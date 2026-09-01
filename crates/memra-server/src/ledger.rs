//! Durable per-request usage and cost receipts.
//!
//! The worker remains the sole source of prompt/cache/completion counts.  The HTTP
//! task appends and syncs the corresponding cost row before publishing a terminal
//! response, so receipt I/O never runs on the CUDA-owner thread.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;

use super::OpenRouterModelMetadata;

const FORMAT: &str = "memra.request-cost.v1";

// ---- outcome census + fault-attribution billing law (owner ruling 2026-08-23) -----------
//
// "if the non response is our fault we should not bill." Every terminal ledger row names
// one of these outcomes, and ONLY the three in `BILLABLE_OUTCOMES` may carry a debit:
//
//   completed         the response was delivered in full                       bills exactly
//   abandoned         the CLIENT walked away mid-generation (user fault)       bills partial
//   deadline_partial  the deadline cut generation and we DELIVERED what we had bills delivered
//   rejected          refused/failed before or during generation (any 4xx/5xx) never bills
//   deadline_exceeded the request's timeout_ms elapsed before we responded     never bills
//   shed_deadline     admission shed: estimated queue wait exceeded deadline   never bills
//   shed_queue        admission shed: absolute queue bound reached             never bills
//   drain_killed      WE killed it at the drain deadline (SIGTERM shutdown)    never bills
//   crashed           the handler panicked while holding the receipt           never bills
//
// The census is closed: `base_row` is the only writer of the `outcome` field, and every
// caller uses one of these literals. `finalize` enforces the debit half at the last write
// point, so a future outcome added without visiting this table cannot bill by accident.
// `deadline_partial` (lane/deadline-partial-20260826) is the third billable outcome and it
// exists BECAUSE the alternative loses information: a deadline-cut partial written as
// `completed` is indistinguishable from a full answer in every census, report and
// reconciliation, with the reason surviving only in an ephemeral log line. It bills, because
// the caller received those tokens — the "our fault, no bill" ruling covers a request we
// failed to answer, not one we answered short and said so in the response.
pub(crate) const BILLABLE_OUTCOMES: [&str; 3] = ["completed", "abandoned", "deadline_partial"];
pub(crate) const ZERO_DEBIT_OUTCOMES: [&str; 6] = [
    "rejected",
    "deadline_exceeded",
    "shed_deadline",
    "shed_queue",
    "drain_killed",
    "crashed",
];

fn outcome_may_debit(outcome: &str) -> bool {
    BILLABLE_OUTCOMES.contains(&outcome)
}

/// Latched by the graceful-drain path the moment the drain deadline expires with requests
/// still in flight: every receipt dropped from that point on was killed by OUR shutdown,
/// not abandoned by its client, so its `Drop` writes `drain_killed` (zero debit) instead
/// of `abandoned` (partial debit). Never cleared — the process is exiting.
static DRAIN_KILL: AtomicBool = AtomicBool::new(false);

pub(crate) fn mark_drain_kill() {
    DRAIN_KILL.store(true, Ordering::SeqCst);
}

fn drain_kill_active() -> bool {
    DRAIN_KILL.load(Ordering::SeqCst)
}

/// Serializes tests that classify a DROPPED receipt (abandoned / crashed / drain_killed):
/// the drain-kill latch is process-global, so a test that sets it must not overlap another
/// test's unfinalized-receipt drop, or that drop is misclassified and the other test flakes.
#[cfg(test)]
pub(crate) static DROP_CLASS_LOCK: Mutex<()> = Mutex::new(());

/// Test-only reset so the drain-kill drop-classification test can restore the latch
/// (production never clears it — the process is exiting).
#[cfg(test)]
pub(crate) fn reset_drain_kill_for_test() {
    DRAIN_KILL.store(false, Ordering::SeqCst);
}
const BUDGET_JOURNAL_FORMAT: &str = "memra.tenant-budget-journal.v1";
const BUDGET_SNAPSHOT_FORMAT: &str = "memra.tenant-budget-snapshot.v1";
const MAX_DECIMAL_SCALE: u32 = 18;
const BUDGET_POLL: Duration = Duration::from_secs(2);
const BUDGET_COMPACT_EVERY: u64 = 128;
const BUDGET_COMPACT_INTERVAL: Duration = Duration::from_secs(30);
const BUDGET_RELOAD_FAIL_CLOSED_AFTER: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Decimal {
    coefficient: u128,
    scale: u32,
}

impl Decimal {
    fn parse(value: &str) -> Result<Self, String> {
        let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
        if whole.is_empty()
            || !whole.bytes().all(|b| b.is_ascii_digit())
            || !fraction.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(format!("invalid non-negative decimal {value:?}"));
        }
        let scale = u32::try_from(fraction.len())
            .map_err(|_| format!("decimal scale is too large in {value:?}"))?;
        if scale > MAX_DECIMAL_SCALE {
            return Err(format!(
                "decimal {value:?} has scale {scale}; maximum supported scale is {MAX_DECIMAL_SCALE}"
            ));
        }
        let digits = format!("{whole}{fraction}");
        let coefficient = digits
            .parse::<u128>()
            .map_err(|_| format!("decimal coefficient overflows u128 in {value:?}"))?;
        Ok(Self { coefficient, scale })
    }

    fn checked_mul_u64(&self, count: u64) -> Result<Self, String> {
        let coefficient = self
            .coefficient
            .checked_mul(count as u128)
            .ok_or_else(|| "request cost multiplication overflowed u128".to_string())?;
        Ok(Self {
            coefficient,
            scale: self.scale,
        })
    }

    fn checked_add(&self, other: &Self) -> Result<Self, String> {
        let scale = self.scale.max(other.scale);
        let left = scale_up(self.coefficient, scale - self.scale)?;
        let right = scale_up(other.coefficient, scale - other.scale)?;
        let coefficient = left
            .checked_add(right)
            .ok_or_else(|| "request cost addition overflowed u128".to_string())?;
        Ok(Self { coefficient, scale })
    }

    fn checked_cmp(&self, other: &Self) -> Result<std::cmp::Ordering, String> {
        let scale = self.scale.max(other.scale);
        Ok(scale_up(self.coefficient, scale - self.scale)?
            .cmp(&scale_up(other.coefficient, scale - other.scale)?))
    }

    fn render(&self) -> String {
        if self.scale == 0 {
            return self.coefficient.to_string();
        }
        let mut digits = self.coefficient.to_string();
        let scale = self.scale as usize;
        if digits.len() <= scale {
            digits.insert_str(0, &"0".repeat(scale + 1 - digits.len()));
        }
        let split = digits.len() - scale;
        digits.insert(split, '.');
        digits
    }

    /// Convert exact USD into the prepaid boundary's integer micro-USD unit. Positive
    /// fractional micros round upward so a stream of tiny requests cannot evade charging.
    fn to_micro_usd_ceil(&self) -> Result<u64, String> {
        let micros = if self.scale <= 6 {
            scale_up(self.coefficient, 6 - self.scale)?
        } else {
            let divisor = pow10(self.scale - 6)?;
            let whole = self.coefficient / divisor;
            let remainder = self.coefficient % divisor;
            whole
                .checked_add(u128::from(remainder != 0))
                .ok_or_else(|| "micro-USD rounding overflowed u128".to_string())?
        };
        u64::try_from(micros).map_err(|_| "request cost exceeds u64 micro-USD".to_string())
    }
}

fn scale_up(value: u128, places: u32) -> Result<u128, String> {
    let factor = pow10(places)?;
    value
        .checked_mul(factor)
        .ok_or_else(|| "request cost decimal alignment overflowed u128".to_string())
}

fn pow10(places: u32) -> Result<u128, String> {
    let mut factor = 1u128;
    for _ in 0..places {
        factor = factor
            .checked_mul(10)
            .ok_or_else(|| "request cost decimal scale overflowed u128".to_string())?;
    }
    Ok(factor)
}

#[derive(Debug, Clone)]
struct PriceSchedule {
    prompt_text: String,
    cached_prompt_text: String,
    completion_text: String,
    request_text: String,
    prompt: Decimal,
    cached_prompt: Decimal,
    completion: Decimal,
    request: Decimal,
}

impl PriceSchedule {
    fn from_metadata(alias: &str, metadata: &OpenRouterModelMetadata) -> Result<Self, String> {
        let required = |field: &str, value: &Option<String>| {
            value.clone().ok_or_else(|| {
                format!("model {alias:?}: {field} is required when MEMRA_REQUEST_LEDGER is enabled")
            })
        };
        let prompt_text = required("pricing.prompt", &metadata.pricing.prompt)?;
        let cached_prompt_text =
            required("pricing.cached_prompt", &metadata.pricing.cached_prompt)?;
        let completion_text = required("pricing.completion", &metadata.pricing.completion)?;
        let request_text = metadata
            .pricing
            .request
            .clone()
            .unwrap_or_else(|| "0".into());
        let schedule = Self {
            prompt: Decimal::parse(&prompt_text)
                .map_err(|e| format!("model {alias:?}: pricing.prompt: {e}"))?,
            cached_prompt: Decimal::parse(&cached_prompt_text)
                .map_err(|e| format!("model {alias:?}: pricing.cached_prompt: {e}"))?,
            completion: Decimal::parse(&completion_text)
                .map_err(|e| format!("model {alias:?}: pricing.completion: {e}"))?,
            request: Decimal::parse(&request_text)
                .map_err(|e| format!("model {alias:?}: pricing.request: {e}"))?,
            prompt_text,
            cached_prompt_text,
            completion_text,
            request_text,
        };
        if schedule.cached_prompt.checked_cmp(&schedule.prompt)? != std::cmp::Ordering::Less {
            return Err(format!(
                "model {alias:?}: pricing.cached_prompt must be lower than pricing.prompt"
            ));
        }
        Ok(schedule)
    }

    fn cost(&self, usage: Usage) -> Result<serde_json::Value, String> {
        let ordinary_prompt = usage
            .prompt_tokens
            .checked_sub(usage.cached_prompt_tokens)
            .ok_or_else(|| {
                format!(
                    "cached prompt tokens {} exceed total prompt tokens {}",
                    usage.cached_prompt_tokens, usage.prompt_tokens
                )
            })?;
        let ordinary_cost = self.prompt.checked_mul_u64(ordinary_prompt)?;
        let cached_cost = self
            .cached_prompt
            .checked_mul_u64(usage.cached_prompt_tokens)?;
        let completion_cost = self.completion.checked_mul_u64(usage.completion_tokens)?;
        let request_cost = self.request.clone();
        let total = ordinary_cost
            .checked_add(&cached_cost)?
            .checked_add(&completion_cost)?
            .checked_add(&request_cost)?;
        Ok(json!({
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "ordinary_prompt_tokens": ordinary_prompt,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.prompt_tokens.checked_add(usage.completion_tokens)
                    .ok_or_else(|| "request token total overflowed u64".to_string())?,
            },
            "unit_prices_usd": {
                "prompt": self.prompt_text,
                "cached_prompt": self.cached_prompt_text,
                "completion": self.completion_text,
                "request": self.request_text,
            },
            "cost_usd": {
                "ordinary_prompt": ordinary_cost.render(),
                "cached_prompt": cached_cost.render(),
                "completion": completion_cost.render(),
                "request": request_cost.render(),
                "total": total.render(),
            },
        }))
    }

    fn worst_case_micro(&self, prompt_tokens: u64, completion_tokens: u64) -> Result<u64, String> {
        let total = self
            .prompt
            .checked_mul_u64(prompt_tokens)?
            .checked_add(&self.completion.checked_mul_u64(completion_tokens)?)?
            .checked_add(&self.request)?;
        total.to_micro_usd_ceil()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: u64,
    pub(crate) cached_prompt_tokens: u64,
    pub(crate) completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct BalanceView {
    pub(crate) tenant: String,
    pub(crate) currency: String,
    pub(crate) balance_micro: i64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct CreditResult {
    pub(crate) tenant: String,
    pub(crate) currency: String,
    pub(crate) amount_micro: i64,
    pub(crate) balance_micro: i64,
    pub(crate) idempotency_key: String,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BudgetAdmissionError {
    Insufficient,
    Unavailable(String),
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub(crate) struct BudgetHealth {
    pub(crate) source_reload_failed: u64,
    pub(crate) source_reload_consecutive: u32,
    pub(crate) source_available: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CreditError {
    Invalid(String),
    Conflict(String),
    Unavailable(String),
}

#[derive(Debug, Clone, Deserialize)]
struct BudgetSourceRow {
    tenant: String,
    currency: String,
    balance_micro: i64,
}

#[derive(Debug, Deserialize, Default)]
struct BudgetSourceFile {
    #[serde(default, alias = "tenants")]
    budgets: Vec<BudgetSourceRow>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BudgetAccount {
    currency: String,
    balance_micro: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BudgetJournalRow {
    format: String,
    kind: String,
    tenant: String,
    currency: String,
    amount_micro: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    balance_after_micro: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exact_cost_usd: Option<String>,
    unix_ms: u64,
}

#[derive(Debug, Serialize)]
struct BudgetSnapshotFile {
    format: &'static str,
    generated_unix_ms: u64,
    balances: Vec<BudgetSnapshotRow>,
}

#[derive(Debug, Serialize)]
struct BudgetSnapshotRow {
    tenant: String,
    currency: String,
    balance_micro: i64,
}

struct BudgetState {
    accounts: HashMap<String, BudgetAccount>,
    source_totals: HashMap<String, BudgetAccount>,
    source_mtime: Option<SystemTime>,
    checked: Instant,
    source_reload_failed: u64,
    source_reload_consecutive: u32,
    source_reload_error: Option<String>,
    debits: HashMap<String, u64>,
    credits: HashMap<String, CreditResult>,
    journal_file: File,
    mutations_since_compact: u64,
    last_compact: Instant,
    compact_next_mutation: bool,
}

struct TenantBudgetsInner {
    source_path: PathBuf,
    journal_path: PathBuf,
    snapshot_path: PathBuf,
    poll: Duration,
    state: Mutex<BudgetState>,
    failed: AtomicBool,
}

#[derive(Clone)]
pub(crate) struct TenantBudgets {
    inner: Arc<TenantBudgetsInner>,
}

/// An in-memory hold for a request's worst-case exact-price ceiling. A budgeted tenant may
/// hold any number of concurrent permits; the balance bounds the SUM of outstanding holds.
/// Terminal worker-truth usage settles the hold to the smaller exact debit; dropping an
/// unfinalized receipt refunds it.
pub(crate) struct BudgetPermit {
    budgets: TenantBudgets,
    tenant: String,
    reserved_micro: u64,
    settled: bool,
}

impl std::fmt::Debug for BudgetPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BudgetPermit")
            .field("tenant", &self.tenant)
            .field("reserved_micro", &self.reserved_micro)
            .finish_non_exhaustive()
    }
}

impl TenantBudgets {
    fn open(source_path: &Path, request_ledger_path: &Path) -> Result<Self, String> {
        let (source_totals, source_mtime) = load_budget_source(source_path)?;
        let journal_path = sidecar_path(request_ledger_path, ".tenant-budget-journal.jsonl");
        let snapshot_path = sidecar_path(request_ledger_path, ".tenant-budget-snapshot.toml");
        repair_jsonl_tail(&journal_path)?;
        let journal_rows: Vec<BudgetJournalRow> = read_jsonl(&journal_path)?;
        let journal_file = open_append_file(&journal_path, "tenant budget journal")?;
        let mut state = BudgetState {
            accounts: source_totals.clone(),
            source_totals,
            source_mtime,
            checked: Instant::now(),
            source_reload_failed: 0,
            source_reload_consecutive: 0,
            source_reload_error: None,
            debits: HashMap::new(),
            credits: HashMap::new(),
            journal_file,
            mutations_since_compact: 0,
            last_compact: Instant::now(),
            compact_next_mutation: true,
        };
        for (line, row) in journal_rows.into_iter().enumerate() {
            replay_budget_row(&mut state, row)
                .map_err(|e| format!("{} line {}: {e}", journal_path.display(), line + 1))?;
        }
        let budgets = Self {
            inner: Arc::new(TenantBudgetsInner {
                source_path: source_path.to_path_buf(),
                journal_path,
                snapshot_path,
                poll: BUDGET_POLL,
                state: Mutex::new(state),
                failed: AtomicBool::new(false),
            }),
        };
        let recovered = budgets.recover_request_ledger(request_ledger_path)?;
        {
            let mut state = budgets.lock_state()?;
            write_budget_snapshot(&budgets.inner.snapshot_path, &state)?;
            state.mutations_since_compact = 0;
            state.last_compact = Instant::now();
            state.compact_next_mutation = true;
        }
        eprintln!(
            "[budget] loaded {} tenant(s) from {}; recovered {recovered} debit(s)",
            budgets.lock_state()?.accounts.len(),
            source_path.display(),
        );
        Ok(budgets)
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, BudgetState>, String> {
        self.inner
            .state
            .lock()
            .map_err(|_| "tenant budget state mutex is poisoned".to_string())
    }

    fn check_available(&self) -> Result<(), String> {
        if self.inner.failed.load(Ordering::Acquire) {
            Err(format!(
                "tenant budget journal {} is latched unavailable",
                self.inner.journal_path.display(),
            ))
        } else {
            Ok(())
        }
    }

    fn maybe_reload(&self) -> Result<(), String> {
        self.check_available()?;
        {
            let state = self.lock_state()?;
            if state.checked.elapsed() < self.inner.poll {
                return source_reload_gate(&state, &self.inner.source_path);
            }
        }
        let mut state = self.lock_state()?;
        if state.checked.elapsed() < self.inner.poll {
            return source_reload_gate(&state, &self.inner.source_path);
        }
        state.checked = Instant::now();
        let reload = match load_budget_source(&self.inner.source_path) {
            Ok((new_totals, new_mtime)) => {
                let changed = new_mtime != state.source_mtime
                    || new_totals != state.source_totals
                    || state.source_reload_consecutive != 0;
                match apply_source_reload(&mut state, new_totals) {
                    Ok(()) => {
                        state.source_mtime = new_mtime;
                        state.source_reload_consecutive = 0;
                        state.source_reload_error = None;
                        if changed {
                            eprintln!(
                                "[budget] source reloaded: {} tenant(s) from {}",
                                state.source_totals.len(),
                                self.inner.source_path.display(),
                            );
                            if let Err(err) =
                                write_budget_snapshot(&self.inner.snapshot_path, &state)
                            {
                                eprintln!("[budget] snapshot refresh failed: {err}");
                                state.compact_next_mutation = true;
                            } else {
                                state.mutations_since_compact = 0;
                                state.last_compact = Instant::now();
                                state.compact_next_mutation = false;
                            }
                        }
                        Ok(())
                    }
                    Err(err) => Err(err),
                }
            }
            Err(err) => Err(err),
        };
        if let Err(err) = reload {
            state.source_reload_failed = state.source_reload_failed.saturating_add(1);
            state.source_reload_consecutive = state.source_reload_consecutive.saturating_add(1);
            state.source_reload_error = Some(err.clone());
            eprintln!(
                "[budget] source reload FAILED ({}/{}: {err}); keeping the previous balances",
                state.source_reload_consecutive, BUDGET_RELOAD_FAIL_CLOSED_AFTER,
            );
        }
        source_reload_gate(&state, &self.inner.source_path)
    }

    pub(crate) fn is_limited(&self, tenant: &str) -> Result<bool, BudgetAdmissionError> {
        self.maybe_reload()
            .map_err(BudgetAdmissionError::Unavailable)?;
        let state = self
            .lock_state()
            .map_err(BudgetAdmissionError::Unavailable)?;
        Ok(state.accounts.contains_key(tenant))
    }

    pub(crate) fn admit(
        &self,
        tenant: &str,
        reserved_micro: u64,
    ) -> Result<Option<BudgetPermit>, BudgetAdmissionError> {
        self.maybe_reload()
            .map_err(BudgetAdmissionError::Unavailable)?;
        let mut state = self
            .lock_state()
            .map_err(BudgetAdmissionError::Unavailable)?;
        let Some(account) = state.accounts.get(tenant) else {
            return Ok(None);
        };
        // CONCURRENT RESERVATIONS (launch blocker fix, 2026-08-15): the old
        // `state.active` one-in-flight-per-tenant gate 429'd every parallel call a
        // paying customer made (an agent's two tool calls; the published c=4/8 TTFT
        // ladder never held for budgeted tenants). The reservation debit below is the
        // real spend bound — each in-flight request holds its worst-case ceiling out
        // of the balance under this same lock, so N parallel requests are admitted
        // exactly while the balance covers the SUM of their reservations.
        let reserved = i64::try_from(reserved_micro).map_err(|_| {
            BudgetAdmissionError::Unavailable(
                "request reservation exceeds i64 micro-USD".to_string(),
            )
        })?;
        if account.balance_micro <= 0 || account.balance_micro < reserved {
            return Err(BudgetAdmissionError::Insufficient);
        }
        let balance_after_reservation =
            account.balance_micro.checked_sub(reserved).ok_or_else(|| {
                BudgetAdmissionError::Unavailable(
                    "tenant reservation underflowed i64 micro-USD".to_string(),
                )
            })?;
        state
            .accounts
            .get_mut(tenant)
            .expect("account existence was checked under the same lock")
            .balance_micro = balance_after_reservation;
        Ok(Some(BudgetPermit {
            budgets: self.clone(),
            tenant: tenant.to_string(),
            reserved_micro,
            settled: false,
        }))
    }

    pub(crate) fn health(&self) -> BudgetHealth {
        let _ = self.maybe_reload();
        match self.inner.state.lock() {
            Ok(state) => BudgetHealth {
                source_reload_failed: state.source_reload_failed,
                source_reload_consecutive: state.source_reload_consecutive,
                source_available: !self.inner.failed.load(Ordering::Acquire)
                    && state.source_reload_consecutive < BUDGET_RELOAD_FAIL_CLOSED_AFTER,
            },
            Err(_) => BudgetHealth {
                source_reload_failed: 0,
                source_reload_consecutive: BUDGET_RELOAD_FAIL_CLOSED_AFTER,
                source_available: false,
            },
        }
    }

    pub(crate) fn balance(&self, tenant: &str) -> Result<Option<BalanceView>, String> {
        self.maybe_reload()?;
        let state = self.lock_state()?;
        Ok(state.accounts.get(tenant).map(|account| BalanceView {
            tenant: tenant.to_string(),
            currency: account.currency.clone(),
            balance_micro: account.balance_micro,
        }))
    }

    pub(crate) fn credit(
        &self,
        tenant: &str,
        amount_micro: i64,
        idempotency_key: &str,
    ) -> Result<CreditResult, CreditError> {
        if !super::auth::tenant_is_valid(tenant) {
            return Err(CreditError::Invalid(
                "tenant must match [A-Za-z0-9_-]+".into(),
            ));
        }
        if amount_micro <= 0 {
            return Err(CreditError::Invalid(
                "amount_micro must be a positive integer".into(),
            ));
        }
        if !valid_idempotency_key(idempotency_key) {
            return Err(CreditError::Invalid(
                "idempotency_key must be 1..128 printable ASCII characters".into(),
            ));
        }
        self.maybe_reload().map_err(CreditError::Unavailable)?;
        let mut state = self.lock_state().map_err(CreditError::Unavailable)?;
        if let Some(original) = state.credits.get(idempotency_key) {
            if original.tenant == tenant && original.amount_micro == amount_micro {
                return Ok(original.clone());
            }
            return Err(CreditError::Conflict(
                "idempotency_key was already used with different credit parameters".into(),
            ));
        }
        let current = state
            .accounts
            .get(tenant)
            .map_or(0, |account| account.balance_micro);
        let balance_after_micro = current.checked_add(amount_micro).ok_or_else(|| {
            CreditError::Invalid("credit would overflow the tenant balance".into())
        })?;
        let result = CreditResult {
            tenant: tenant.to_string(),
            currency: "USD".into(),
            amount_micro,
            balance_micro: balance_after_micro,
            idempotency_key: idempotency_key.to_string(),
        };
        let row = BudgetJournalRow {
            format: BUDGET_JOURNAL_FORMAT.into(),
            kind: "credit".into(),
            tenant: tenant.into(),
            currency: "USD".into(),
            amount_micro: amount_micro as u64,
            request_id: None,
            idempotency_key: Some(idempotency_key.into()),
            balance_after_micro: Some(balance_after_micro),
            exact_cost_usd: None,
            unix_ms: unix_ms(),
        };
        append_budget_row(&self.inner, &mut state, &row).map_err(CreditError::Unavailable)?;
        state.accounts.insert(
            tenant.into(),
            BudgetAccount {
                currency: "USD".into(),
                balance_micro: balance_after_micro,
            },
        );
        state.credits.insert(idempotency_key.into(), result.clone());
        compact_budget_if_due(&self.inner, &mut state);
        Ok(result)
    }

    /// Operator-initiated balance reduction (`POST /admin/tenants/{t}/debit`): balance
    /// migrations between boxes, refunds of mistaken credits, dispute clawbacks. It
    /// journals the SAME `kind: "debit"` row request settlement writes, with the
    /// request_id namespaced `admin-debit:{tenant}:{idempotency_key}` — so replay,
    /// compaction and the debits dedupe map need no new row kind, and an admin debit
    /// can never collide with a request settlement. The balance MAY go negative: a
    /// cross-box migration debits spend that already happened elsewhere, and admission
    /// already refuses to serve a non-positive balance. `reason` is required and
    /// journaled — an unexplained balance change is indistinguishable from a bug.
    ///
    /// Idempotency is the debits map (rebuilt from the journal on restart), and the
    /// TENANT IS PART OF THE KEY: the debits map stores only an amount, so a bare
    /// `admin-debit:{idempotency_key}` id would let a same-key same-amount call
    /// against a DIFFERENT tenant — including an unenrolled one — return a fabricated
    /// 200 for a debit that never happened (and a migration script reusing one key
    /// across tenants would silently skip every tenant after the first). A replay
    /// with the same tenant, key and amount returns Ok with the CURRENT balance, not
    /// the original post-debit balance; a same-tenant-and-key different-amount call
    /// is a Conflict.
    pub(crate) fn admin_debit(
        &self,
        tenant: &str,
        amount_micro: i64,
        idempotency_key: &str,
        reason: &str,
    ) -> Result<CreditResult, CreditError> {
        if !super::auth::tenant_is_valid(tenant) {
            return Err(CreditError::Invalid(
                "tenant must match [A-Za-z0-9_-]+".into(),
            ));
        }
        if amount_micro <= 0 {
            return Err(CreditError::Invalid(
                "amount_micro must be a positive integer".into(),
            ));
        }
        if !valid_idempotency_key(idempotency_key) {
            return Err(CreditError::Invalid(
                "idempotency_key must be 1..128 printable ASCII characters".into(),
            ));
        }
        if reason.trim().is_empty() || reason.len() > 256 || !reason.is_ascii() {
            return Err(CreditError::Invalid(
                "reason must be 1..256 printable ASCII characters".into(),
            ));
        }
        self.maybe_reload().map_err(CreditError::Unavailable)?;
        let mut state = self.lock_state().map_err(CreditError::Unavailable)?;
        let request_id = format!("admin-debit:{tenant}:{idempotency_key}");
        let amount = amount_micro as u64;
        if let Some(previous) = state.debits.get(&request_id) {
            if *previous == amount {
                let balance_micro = state
                    .accounts
                    .get(tenant)
                    .map_or(0, |account| account.balance_micro);
                return Ok(CreditResult {
                    tenant: tenant.to_string(),
                    currency: "USD".into(),
                    amount_micro,
                    balance_micro,
                    idempotency_key: idempotency_key.to_string(),
                });
            }
            return Err(CreditError::Conflict(
                "idempotency_key was already used with a different debit amount".into(),
            ));
        }
        let current = state
            .accounts
            .get(tenant)
            .ok_or_else(|| {
                CreditError::Invalid(format!("cannot debit unenrolled tenant {tenant:?}"))
            })?
            .balance_micro;
        let balance_after_micro = current.checked_sub(amount_micro).ok_or_else(|| {
            CreditError::Invalid("debit would underflow the tenant balance".into())
        })?;
        let row = BudgetJournalRow {
            format: BUDGET_JOURNAL_FORMAT.into(),
            kind: "debit".into(),
            tenant: tenant.into(),
            currency: "USD".into(),
            amount_micro: amount,
            request_id: Some(request_id.clone()),
            idempotency_key: Some(idempotency_key.into()),
            balance_after_micro: Some(balance_after_micro),
            exact_cost_usd: Some(format!("admin-debit reason: {reason}")),
            unix_ms: unix_ms(),
        };
        append_budget_row(&self.inner, &mut state, &row).map_err(CreditError::Unavailable)?;
        state
            .accounts
            .get_mut(tenant)
            .expect("account existence was checked under the same lock")
            .balance_micro = balance_after_micro;
        state.debits.insert(request_id, amount);
        compact_budget_if_due(&self.inner, &mut state);
        Ok(CreditResult {
            tenant: tenant.to_string(),
            currency: "USD".into(),
            amount_micro,
            balance_micro: balance_after_micro,
            idempotency_key: idempotency_key.to_string(),
        })
    }

    fn debit_reserved(
        &self,
        request_id: &str,
        tenant: &str,
        amount_micro: u64,
        refund_micro: u64,
        exact_cost_usd: &str,
    ) -> Result<(), String> {
        self.check_available()?;
        let mut state = self.lock_state()?;
        if let Some(previous) = state.debits.get(request_id) {
            return if *previous == amount_micro {
                Ok(())
            } else {
                Err(format!(
                    "request {request_id} has conflicting budget debits {previous} and {amount_micro}"
                ))
            };
        }
        let current = state
            .accounts
            .get(tenant)
            .ok_or_else(|| format!("request {request_id} debits unenrolled tenant {tenant:?}"))?
            .balance_micro;
        let refund = i64::try_from(refund_micro)
            .map_err(|_| "request reservation refund exceeds i64 micro-USD".to_string())?;
        let balance_after_micro = current
            .checked_add(refund)
            .ok_or_else(|| "tenant balance overflowed while settling reservation".to_string())?;
        let row = BudgetJournalRow {
            format: BUDGET_JOURNAL_FORMAT.into(),
            kind: "debit".into(),
            tenant: tenant.into(),
            currency: "USD".into(),
            amount_micro,
            request_id: Some(request_id.into()),
            idempotency_key: None,
            balance_after_micro: Some(balance_after_micro),
            exact_cost_usd: Some(exact_cost_usd.into()),
            unix_ms: unix_ms(),
        };
        append_budget_row(&self.inner, &mut state, &row)?;
        state
            .accounts
            .get_mut(tenant)
            .expect("account existence was checked under the same lock")
            .balance_micro = balance_after_micro;
        state.debits.insert(request_id.into(), amount_micro);
        compact_budget_if_due(&self.inner, &mut state);
        Ok(())
    }

    fn recover_request_ledger(&self, request_ledger_path: &Path) -> Result<usize, String> {
        let rows: Vec<serde_json::Value> = read_jsonl(request_ledger_path)?;
        let mut recovered = 0;
        for (line, row) in rows.iter().enumerate() {
            let Some(budget) = row.get("budget").filter(|value| !value.is_null()) else {
                continue;
            };
            let request_id = row
                .get("request_id")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "{} line {}: budgeted row has no request_id",
                        request_ledger_path.display(),
                        line + 1
                    )
                })?;
            let tenant = row
                .get("tenant")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "{} line {}: budgeted row has no tenant",
                        request_ledger_path.display(),
                        line + 1
                    )
                })?;
            let amount_micro = budget
                .get("debit_micro")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    format!(
                        "{} line {}: budget.debit_micro is not a u64",
                        request_ledger_path.display(),
                        line + 1
                    )
                })?;
            let exact_cost_usd = row
                .pointer("/cost_usd/total")
                .and_then(|value| value.as_str())
                .ok_or_else(|| {
                    format!(
                        "{} line {}: budgeted row has no exact total cost",
                        request_ledger_path.display(),
                        line + 1
                    )
                })?;
            let seen = self.lock_state()?.debits.contains_key(request_id);
            if !seen {
                self.recover_debit(request_id, tenant, amount_micro, exact_cost_usd)?;
                recovered += 1;
            }
        }
        Ok(recovered)
    }

    fn recover_debit(
        &self,
        request_id: &str,
        tenant: &str,
        amount_micro: u64,
        exact_cost_usd: &str,
    ) -> Result<(), String> {
        let mut state = self.lock_state()?;
        if let Some(previous) = state.debits.get(request_id) {
            return if *previous == amount_micro {
                Ok(())
            } else {
                Err(format!(
                    "request {request_id} has conflicting budget debits {previous} and {amount_micro}"
                ))
            };
        }
        let amount = i64::try_from(amount_micro)
            .map_err(|_| "request debit exceeds i64 micro-USD".to_string())?;
        let current = state
            .accounts
            .get(tenant)
            .ok_or_else(|| format!("request {request_id} debits unenrolled tenant {tenant:?}"))?
            .balance_micro;
        let balance_after_micro = current
            .checked_sub(amount)
            .ok_or_else(|| "tenant balance underflowed i64 micro-USD".to_string())?;
        let row = BudgetJournalRow {
            format: BUDGET_JOURNAL_FORMAT.into(),
            kind: "debit".into(),
            tenant: tenant.into(),
            currency: "USD".into(),
            amount_micro,
            request_id: Some(request_id.into()),
            idempotency_key: None,
            balance_after_micro: Some(balance_after_micro),
            exact_cost_usd: Some(exact_cost_usd.into()),
            unix_ms: unix_ms(),
        };
        append_budget_row(&self.inner, &mut state, &row)?;
        state
            .accounts
            .get_mut(tenant)
            .expect("account existence was checked under the same lock")
            .balance_micro = balance_after_micro;
        state.debits.insert(request_id.into(), amount_micro);
        compact_budget_if_due(&self.inner, &mut state);
        Ok(())
    }

    #[cfg(test)]
    fn with_poll(mut self, poll: Duration) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("test budget manager is uniquely owned")
            .poll = poll;
        self
    }

    #[cfg(test)]
    fn journal_path(&self) -> &Path {
        &self.inner.journal_path
    }
}

impl BudgetPermit {
    fn settle(
        &mut self,
        request_id: &str,
        amount_micro: u64,
        exact_cost_usd: &str,
    ) -> Result<(), String> {
        if amount_micro > self.reserved_micro {
            // HTTP-side tokenization/cap calculation drifted from worker truth. Keep the
            // reservation held and latch later admissions until restart repair applies the
            // durable terminal row; refunding would reopen unpaid serving.
            self.settled = true;
            self.budgets.inner.failed.store(true, Ordering::Release);
            return Err(format!(
                "request {request_id} exact debit {amount_micro} exceeds reservation {}",
                self.reserved_micro,
            ));
        }
        let refund_micro = self.reserved_micro - amount_micro;
        self.budgets.debit_reserved(
            request_id,
            &self.tenant,
            amount_micro,
            refund_micro,
            exact_cost_usd,
        )?;
        self.settled = true;
        Ok(())
    }
}

impl Drop for BudgetPermit {
    fn drop(&mut self) {
        match self.budgets.inner.state.lock() {
            Ok(mut state) => {
                if !self.settled {
                    let refund = match i64::try_from(self.reserved_micro) {
                        Ok(refund) => refund,
                        Err(_) => {
                            self.budgets.inner.failed.store(true, Ordering::Release);
                            eprintln!(
                                "[budget] ERROR: reservation refund exceeds i64 for tenant {:?}",
                                self.tenant,
                            );
                            return;
                        }
                    };
                    let Some(account) = state.accounts.get_mut(&self.tenant) else {
                        self.budgets.inner.failed.store(true, Ordering::Release);
                        eprintln!(
                            "[budget] ERROR: reservation tenant {:?} disappeared before refund",
                            self.tenant,
                        );
                        return;
                    };
                    let Some(balance) = account.balance_micro.checked_add(refund) else {
                        self.budgets.inner.failed.store(true, Ordering::Release);
                        eprintln!(
                            "[budget] ERROR: reservation refund overflow for tenant {:?}",
                            self.tenant,
                        );
                        return;
                    };
                    account.balance_micro = balance;
                }
            }
            Err(_) => {
                self.budgets.inner.failed.store(true, Ordering::Release);
                eprintln!(
                    "[budget] ERROR: could not release admission reservation for tenant {:?}",
                    self.tenant,
                );
            }
        }
    }
}

fn source_reload_gate(state: &BudgetState, path: &Path) -> Result<(), String> {
    if state.source_reload_consecutive < BUDGET_RELOAD_FAIL_CLOSED_AFTER {
        return Ok(());
    }
    Err(format!(
        "tenant budget source {} failed {} consecutive reload polls: {}",
        path.display(),
        state.source_reload_consecutive,
        state
            .source_reload_error
            .as_deref()
            .unwrap_or("unknown reload failure"),
    ))
}

fn valid_idempotency_key(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn load_budget_source(
    path: &Path,
) -> Result<(HashMap<String, BudgetAccount>, Option<SystemTime>), String> {
    let (text, source_mtime) = read_secure_text(path, "tenant budget source")?;
    let rows = if path.extension().and_then(|extension| extension.to_str()) == Some("toml") {
        toml::from_str::<BudgetSourceFile>(&text)
            .map_err(|e| format!("{}: tenant budget TOML: {e}", path.display()))?
            .budgets
    } else {
        text.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(line, value)| {
                serde_json::from_str::<BudgetSourceRow>(value).map_err(|e| {
                    format!(
                        "{} line {}: tenant budget JSON: {e}",
                        path.display(),
                        line + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut totals: HashMap<String, BudgetAccount> = HashMap::new();
    for (index, row) in rows.into_iter().enumerate() {
        if !super::auth::tenant_is_valid(&row.tenant) {
            return Err(format!(
                "{} row {}: tenant must match [A-Za-z0-9_-]+",
                path.display(),
                index + 1,
            ));
        }
        if row.currency != "USD" {
            return Err(format!(
                "{} row {}: currency {:?} is unsupported; ledger prices are USD",
                path.display(),
                index + 1,
                row.currency,
            ));
        }
        if row.balance_micro < 0 {
            return Err(format!(
                "{} row {}: balance_micro must be non-negative",
                path.display(),
                index + 1,
            ));
        }
        let account = totals.entry(row.tenant).or_insert_with(|| BudgetAccount {
            currency: "USD".into(),
            balance_micro: 0,
        });
        account.balance_micro = account
            .balance_micro
            .checked_add(row.balance_micro)
            .ok_or_else(|| format!("{}: tenant credit total overflows i64", path.display()))?;
    }
    Ok((totals, source_mtime))
}

fn apply_source_reload(
    state: &mut BudgetState,
    new_totals: HashMap<String, BudgetAccount>,
) -> Result<(), String> {
    for (tenant, previous) in &state.source_totals {
        let Some(next) = new_totals.get(tenant) else {
            return Err(format!(
                "append-only budget source removed tenant {tenant:?}"
            ));
        };
        if next.currency != previous.currency || next.balance_micro < previous.balance_micro {
            return Err(format!(
                "append-only budget source reduced or changed tenant {tenant:?}"
            ));
        }
    }
    let mut accounts = state.accounts.clone();
    for (tenant, next) in &new_totals {
        let previous = state
            .source_totals
            .get(tenant)
            .map_or(0, |account| account.balance_micro);
        let delta = next.balance_micro - previous;
        if delta == 0 && accounts.contains_key(tenant) {
            continue;
        }
        let account = accounts
            .entry(tenant.clone())
            .or_insert_with(|| BudgetAccount {
                currency: next.currency.clone(),
                balance_micro: 0,
            });
        if account.currency != next.currency {
            return Err(format!("tenant {tenant:?} currency changed during reload"));
        }
        account.balance_micro = account
            .balance_micro
            .checked_add(delta)
            .ok_or_else(|| format!("tenant {tenant:?} balance overflowed during reload"))?;
    }
    state.accounts = accounts;
    state.source_totals = new_totals;
    Ok(())
}

fn replay_budget_row(state: &mut BudgetState, row: BudgetJournalRow) -> Result<(), String> {
    if row.format != BUDGET_JOURNAL_FORMAT {
        return Err(format!("unknown budget journal format {:?}", row.format));
    }
    if row.currency != "USD" {
        return Err(format!("unsupported budget currency {:?}", row.currency));
    }
    let amount = i64::try_from(row.amount_micro)
        .map_err(|_| "budget journal amount exceeds i64".to_string())?;
    match row.kind.as_str() {
        "debit" => {
            let request_id = row
                .request_id
                .ok_or_else(|| "debit row has no request_id".to_string())?;
            if let Some(previous) = state.debits.get(&request_id) {
                if *previous == row.amount_micro {
                    return Ok(());
                }
                return Err(format!("request {request_id:?} has conflicting debit rows"));
            }
            let account = state.accounts.get_mut(&row.tenant).ok_or_else(|| {
                format!(
                    "request {request_id:?} debits unenrolled tenant {:?}",
                    row.tenant,
                )
            })?;
            account.balance_micro = account
                .balance_micro
                .checked_sub(amount)
                .ok_or_else(|| "tenant balance underflowed while replaying debit".to_string())?;
            state.debits.insert(request_id, row.amount_micro);
        }
        "credit" => {
            let idempotency_key = row
                .idempotency_key
                .ok_or_else(|| "credit row has no idempotency_key".to_string())?;
            let balance_after_micro = row
                .balance_after_micro
                .ok_or_else(|| "credit row has no balance_after_micro".to_string())?;
            let result = CreditResult {
                tenant: row.tenant.clone(),
                currency: row.currency.clone(),
                amount_micro: amount,
                balance_micro: balance_after_micro,
                idempotency_key: idempotency_key.clone(),
            };
            if let Some(previous) = state.credits.get(&idempotency_key) {
                if previous == &result {
                    return Ok(());
                }
                return Err(format!(
                    "idempotency key {idempotency_key:?} has conflicting credit rows"
                ));
            }
            let account = state
                .accounts
                .entry(row.tenant)
                .or_insert_with(|| BudgetAccount {
                    currency: "USD".into(),
                    balance_micro: 0,
                });
            account.balance_micro = account
                .balance_micro
                .checked_add(amount)
                .ok_or_else(|| "tenant balance overflowed while replaying credit".to_string())?;
            state.credits.insert(idempotency_key, result);
        }
        other => return Err(format!("unknown budget journal kind {other:?}")),
    }
    Ok(())
}

fn append_budget_row(
    inner: &TenantBudgetsInner,
    state: &mut BudgetState,
    row: &BudgetJournalRow,
) -> Result<(), String> {
    let mut encoded =
        serde_json::to_vec(row).map_err(|e| format!("serialize tenant budget journal row: {e}"))?;
    encoded.push(b'\n');
    if let Err(err) = state.journal_file.write_all(&encoded) {
        inner.failed.store(true, Ordering::Release);
        return Err(format!("append {}: {err}", inner.journal_path.display()));
    }
    if let Err(err) = state.journal_file.sync_data() {
        inner.failed.store(true, Ordering::Release);
        return Err(format!("sync {}: {err}", inner.journal_path.display()));
    }
    state.mutations_since_compact = state.mutations_since_compact.saturating_add(1);
    Ok(())
}

fn compact_budget_if_due(inner: &TenantBudgetsInner, state: &mut BudgetState) {
    // Snapshot the first post-start mutation so a quiet tenant never leaves the derived
    // compact view stale indefinitely. Busy traffic then amortizes snapshots by count/time.
    if !state.compact_next_mutation
        && state.mutations_since_compact < BUDGET_COMPACT_EVERY
        && state.last_compact.elapsed() < BUDGET_COMPACT_INTERVAL
    {
        return;
    }
    match write_budget_snapshot(&inner.snapshot_path, state) {
        Ok(()) => {
            state.mutations_since_compact = 0;
            state.last_compact = Instant::now();
            state.compact_next_mutation = false;
        }
        Err(err) => {
            state.compact_next_mutation = true;
            eprintln!("[budget] snapshot compaction failed: {err}");
        }
    }
}

fn write_budget_snapshot(path: &Path, state: &BudgetState) -> Result<(), String> {
    let mut balances: Vec<BudgetSnapshotRow> = state
        .accounts
        .iter()
        .map(|(tenant, account)| BudgetSnapshotRow {
            tenant: tenant.clone(),
            currency: account.currency.clone(),
            balance_micro: account.balance_micro,
        })
        .collect();
    balances.sort_by(|left, right| left.tenant.cmp(&right.tenant));
    let payload = toml::to_string(&BudgetSnapshotFile {
        format: BUDGET_SNAPSHOT_FORMAT,
        generated_unix_ms: unix_ms(),
        balances,
    })
    .map_err(|e| format!("serialize budget snapshot: {e}"))?;
    atomic_write(path, payload.as_bytes(), "tenant budget snapshot")
}

fn read_secure_text(path: &Path, label: &str) -> Result<(String, Option<SystemTime>), String> {
    let mut file = open_read_file(path, label)?;
    let before = file
        .metadata()
        .map_err(|e| format!("stat {label} {} before read: {e}", path.display()))?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|e| format!("read {label} {}: {e}", path.display()))?;
    let after = file
        .metadata()
        .map_err(|e| format!("stat {label} {} after read: {e}", path.display()))?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err(format!(
            "{label} {} changed while it was being read; retry after the writer finishes",
            path.display(),
        ));
    }
    Ok((text, after.modified().ok()))
}

fn open_read_file(path: &Path, label: &str) -> Result<File, String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|e| format!("open {label} {}: {e}", path.display()))?;
    validate_private_regular_file(&file, path, label)?;
    Ok(file)
}

fn open_append_file(path: &Path, label: &str) -> Result<File, String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let creating = !path.exists();
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o640).custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|e| format!("open {label} {}: {e}", path.display()))?;
    validate_private_regular_file(&file, path, label)?;
    if creating {
        // File data alone is not a durable first journal entry if the directory entry can
        // disappear after a power loss. Persist the create before any acknowledged append.
        sync_parent(path, label)?;
    }
    Ok(file)
}

fn validate_private_regular_file(file: &File, path: &Path, label: &str) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|e| format!("stat {label} {}: {e}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o137 != 0 {
            return Err(format!(
                "{label} {} must have 0600 or 0640-class permissions; found {mode:04o}",
                path.display(),
            ));
        }
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;

    let tmp = sidecar_path(path, ".tmp");
    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    options.mode(0o640).custom_flags(libc::O_NOFOLLOW);
    let mut file = options
        .open(&tmp)
        .map_err(|e| format!("open {label} temp {}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        file.set_permissions(std::fs::Permissions::from_mode(0o640))
            .map_err(|e| format!("chmod {label} temp {}: {e}", tmp.display()))?;
    }
    validate_private_regular_file(&file, &tmp, label)?;
    file.write_all(bytes)
        .map_err(|e| format!("write {label} temp {}: {e}", tmp.display()))?;
    file.sync_all()
        .map_err(|e| format!("sync {label} temp {}: {e}", tmp.display()))?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(|e| {
        format!(
            "rename {label} {} -> {}: {e}",
            tmp.display(),
            path.display()
        )
    })?;
    sync_parent(path, label)
}

fn sync_parent(path: &Path, label: &str) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| format!("sync {label} directory {}: {e}", parent.display()))
}

fn repair_jsonl_tail(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let mut reader = open_read_file(path, "JSONL")?;
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read JSONL {}: {e}", path.display()))?;
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        return Ok(());
    }
    let start = bytes
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    if serde_json::from_slice::<serde_json::Value>(&bytes[start..]).is_ok() {
        let mut file = open_append_file(path, "JSONL")?;
        file.write_all(b"\n")
            .and_then(|()| file.sync_data())
            .map_err(|e| format!("complete JSONL tail {}: {e}", path.display()))?;
        return Ok(());
    }
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt as _;
    let mut options = OpenOptions::new();
    options.write(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|e| format!("open JSONL tail {}: {e}", path.display()))?;
    validate_private_regular_file(&file, path, "JSONL")?;
    file.set_len(start as u64)
        .and_then(|()| file.sync_data())
        .map_err(|e| format!("truncate incomplete JSONL tail {}: {e}", path.display()))?;
    eprintln!(
        "[ledger] recovered incomplete final JSONL row from {} (truncated {} byte(s))",
        path.display(),
        bytes.len() - start,
    );
    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = open_read_file(path, "JSONL")?;
    let mut rows = Vec::new();
    for (line, value) in BufReader::new(file).lines().enumerate() {
        let value = value.map_err(|e| format!("read {} line {}: {e}", path.display(), line + 1))?;
        if value.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&value)
                .map_err(|e| format!("parse {} line {}: {e}", path.display(), line + 1))?,
        );
    }
    Ok(rows)
}

fn json_u64(value: &serde_json::Value, field: &str, zero_based_line: usize) -> Result<u64, String> {
    value
        .get(field)
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            format!(
                "request ledger line {} has no u64 usage.{field}",
                zero_based_line + 1,
            )
        })
}

fn checked_sum(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("tenant usage {label} overflowed u64"))
}

fn civil_day(unix_day: u64) -> String {
    // Howard Hinnant's civil-from-days transform. `unix_day` is non-negative, but the
    // epoch offset crosses the proleptic-calendar origin inside the calculation.
    let Ok(mut z) = i64::try_from(unix_day) else {
        return unix_day.to_string();
    };
    z += 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

#[derive(Clone)]
pub(crate) struct Ledger {
    inner: Arc<LedgerInner>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TenantUsage {
    tenant: String,
    days: Vec<TenantUsageDay>,
}

#[derive(Debug, Serialize)]
struct TenantUsageDay {
    day: String,
    day_start_unix: u64,
    requests: u64,
    prompt_tokens: u64,
    cached_prompt_tokens: u64,
    ordinary_prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost_usd: String,
    debited_micro: u64,
}

#[derive(Default)]
struct UsageAccumulator {
    requests: u64,
    prompt_tokens: u64,
    cached_prompt_tokens: u64,
    ordinary_prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    cost: Option<Decimal>,
    debited_micro: u64,
}

struct LedgerInner {
    path: PathBuf,
    file: Mutex<File>,
    prices: HashMap<String, PriceSchedule>,
    budgets: Option<TenantBudgets>,
    failed: AtomicBool,
}

impl Ledger {
    pub(crate) fn from_env(
        models: &[(String, String, Option<String>)],
        metadata: &HashMap<String, OpenRouterModelMetadata>,
    ) -> Result<Option<Self>, String> {
        let budget_path = std::env::var_os("MEMRA_TENANT_BUDGETS");
        let Some(path) = std::env::var_os("MEMRA_REQUEST_LEDGER") else {
            if budget_path.is_some() {
                return Err(
                    "MEMRA_TENANT_BUDGETS requires MEMRA_REQUEST_LEDGER so terminal costs can be recovered exactly"
                        .into(),
                );
            }
            return Ok(None);
        };
        if path.is_empty() {
            return Err("MEMRA_REQUEST_LEDGER must not be empty".into());
        }
        let mut prices = HashMap::new();
        for (alias, _, _) in models {
            let model_metadata = metadata.get(alias).ok_or_else(|| {
                format!(
                    "model {alias:?}: MEMRA_MODEL_METADATA entry is required when \
                     MEMRA_REQUEST_LEDGER is enabled"
                )
            })?;
            prices.insert(
                alias.clone(),
                PriceSchedule::from_metadata(alias, model_metadata)?,
            );
        }
        let budget_path = budget_path
            .as_deref()
            .map(Path::new)
            .filter(|path| !path.as_os_str().is_empty());
        if std::env::var_os("MEMRA_TENANT_BUDGETS").is_some() && budget_path.is_none() {
            return Err("MEMRA_TENANT_BUDGETS must not be empty".into());
        }
        let ledger = Self::open_with_budgets(Path::new(&path), prices, budget_path)?;
        eprintln!(
            "[ledger] durable request-cost ledger enabled: {}",
            ledger.inner.path.display()
        );
        Ok(Some(ledger))
    }

    #[cfg(test)]
    fn open(path: &Path, prices: HashMap<String, PriceSchedule>) -> Result<Self, String> {
        Self::open_with_budgets(path, prices, None)
    }

    fn open_with_budgets(
        path: &Path,
        prices: HashMap<String, PriceSchedule>,
        budget_path: Option<&Path>,
    ) -> Result<Self, String> {
        repair_jsonl_tail(path)?;
        let file = open_append_file(path, "request ledger")?;
        let budgets = budget_path
            .map(|source| TenantBudgets::open(source, path))
            .transpose()?;
        Ok(Self {
            inner: Arc::new(LedgerInner {
                path: path.to_path_buf(),
                file: Mutex::new(file),
                prices,
                budgets,
                failed: AtomicBool::new(false),
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn start(
        &self,
        request_id: &str,
        tenant: &str,
        model: &str,
        route: &'static str,
        lane: &'static str,
        stream: bool,
    ) -> PendingReceipt {
        self.start_with_budget(request_id, tenant, model, route, lane, stream, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_with_budget(
        &self,
        request_id: &str,
        tenant: &str,
        model: &str,
        route: &'static str,
        lane: &'static str,
        stream: bool,
        budget_permit: Option<BudgetPermit>,
    ) -> PendingReceipt {
        PendingReceipt {
            ledger: self.clone(),
            request_id: request_id.into(),
            tenant: tenant.into(),
            model: model.into(),
            route,
            lane,
            stream,
            started_unix_ms: unix_ms(),
            partial_usage: Usage::default(),
            budget_permit,
            pending_debit: None,
            finalized: false,
            capture: None,
        }
    }

    pub(crate) fn budgets(&self) -> Option<TenantBudgets> {
        self.inner.budgets.clone()
    }

    pub(crate) fn reserve_budget(
        &self,
        tenant: &str,
        model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) -> Result<Option<BudgetPermit>, BudgetAdmissionError> {
        if self.inner.failed.load(Ordering::Acquire) {
            return Err(BudgetAdmissionError::Unavailable(format!(
                "request ledger {} is latched unavailable after an earlier write failure",
                self.inner.path.display(),
            )));
        }
        let Some(budgets) = self.inner.budgets.as_ref() else {
            return Ok(None);
        };
        if !budgets.is_limited(tenant)? {
            return Ok(None);
        }
        let schedule = self.inner.prices.get(model).ok_or_else(|| {
            BudgetAdmissionError::Unavailable(format!(
                "request ledger has no price schedule for {model:?}",
            ))
        })?;
        let reserved_micro = schedule
            .worst_case_micro(prompt_tokens, completion_tokens)
            .map_err(BudgetAdmissionError::Unavailable)?;
        budgets.admit(tenant, reserved_micro)
    }

    pub(crate) fn admin_audit_path(&self) -> PathBuf {
        sidecar_path(&self.inner.path, ".admin-audit.jsonl")
    }

    pub(crate) fn tenant_usage(&self, tenant: &str) -> Result<TenantUsage, String> {
        // A synchronized append is a terminal commit. Hold the writer lock while taking the
        // read snapshot so the admin export never observes half of the current JSONL row.
        let _writer = self
            .inner
            .file
            .lock()
            .map_err(|_| "request ledger writer mutex is poisoned".to_string())?;
        let rows: Vec<serde_json::Value> = read_jsonl(&self.inner.path)?;
        let mut buckets: HashMap<u64, UsageAccumulator> = HashMap::new();
        for (line, row) in rows.iter().enumerate() {
            if row.get("tenant").and_then(|value| value.as_str()) != Some(tenant) {
                continue;
            }
            let Some(usage) = row.get("usage").filter(|value| !value.is_null()) else {
                continue;
            };
            let finished_ms = row
                .get("finished_unix_ms")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| {
                    format!("request ledger line {} has no finished_unix_ms", line + 1)
                })?;
            let day = finished_ms / 86_400_000;
            let bucket = buckets.entry(day).or_default();
            bucket.requests = checked_sum(bucket.requests, 1, "request count")?;
            bucket.prompt_tokens = checked_sum(
                bucket.prompt_tokens,
                json_u64(usage, "prompt_tokens", line)?,
                "prompt tokens",
            )?;
            bucket.cached_prompt_tokens = checked_sum(
                bucket.cached_prompt_tokens,
                json_u64(usage, "cached_prompt_tokens", line)?,
                "cached prompt tokens",
            )?;
            bucket.ordinary_prompt_tokens = checked_sum(
                bucket.ordinary_prompt_tokens,
                json_u64(usage, "ordinary_prompt_tokens", line)?,
                "ordinary prompt tokens",
            )?;
            bucket.completion_tokens = checked_sum(
                bucket.completion_tokens,
                json_u64(usage, "completion_tokens", line)?,
                "completion tokens",
            )?;
            bucket.total_tokens = checked_sum(
                bucket.total_tokens,
                json_u64(usage, "total_tokens", line)?,
                "total tokens",
            )?;
            let total = row
                .pointer("/cost_usd/total")
                .and_then(|value| value.as_str())
                .ok_or_else(|| format!("request ledger line {} has no total cost", line + 1))?;
            let cost = Decimal::parse(total)?;
            bucket.cost = Some(match bucket.cost.take() {
                Some(previous) => previous.checked_add(&cost)?,
                None => cost,
            });
            if let Some(debit) = row
                .pointer("/budget/debit_micro")
                .and_then(|value| value.as_u64())
            {
                bucket.debited_micro =
                    checked_sum(bucket.debited_micro, debit, "debited micro-USD")?;
            }
        }
        let mut buckets: Vec<(u64, UsageAccumulator)> = buckets.into_iter().collect();
        buckets.sort_by_key(|(day, _)| *day);
        let days = buckets
            .into_iter()
            .map(|(day, bucket)| TenantUsageDay {
                day: civil_day(day),
                day_start_unix: day.saturating_mul(86_400),
                requests: bucket.requests,
                prompt_tokens: bucket.prompt_tokens,
                cached_prompt_tokens: bucket.cached_prompt_tokens,
                ordinary_prompt_tokens: bucket.ordinary_prompt_tokens,
                completion_tokens: bucket.completion_tokens,
                total_tokens: bucket.total_tokens,
                cost_usd: bucket.cost.map_or_else(|| "0".into(), |cost| cost.render()),
                debited_micro: bucket.debited_micro,
            })
            .collect();
        Ok(TenantUsage {
            tenant: tenant.to_string(),
            days,
        })
    }

    fn append(&self, row: &serde_json::Value) -> Result<(), String> {
        if self.inner.failed.load(Ordering::Acquire) {
            return Err(format!(
                "request ledger {} is latched unavailable after an earlier write failure",
                self.inner.path.display()
            ));
        }
        let mut encoded =
            serde_json::to_vec(row).map_err(|e| format!("serialize request ledger row: {e}"))?;
        encoded.push(b'\n');
        let mut file = match self.inner.file.lock() {
            Ok(file) => file,
            Err(_) => {
                self.inner.failed.store(true, Ordering::Release);
                return Err("request ledger writer mutex is poisoned".into());
            }
        };
        if let Err(err) = file.write_all(&encoded) {
            self.inner.failed.store(true, Ordering::Release);
            return Err(format!(
                "append request ledger {}: {err}",
                self.inner.path.display()
            ));
        }
        if let Err(err) = file.sync_data() {
            self.inner.failed.store(true, Ordering::Release);
            return Err(format!(
                "sync request ledger {}: {err}",
                self.inner.path.display()
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn for_test(path: &Path, alias: &str, metadata: &OpenRouterModelMetadata) -> Self {
        let schedule = PriceSchedule::from_metadata(alias, metadata).unwrap();
        Self::open(path, HashMap::from([(alias.to_string(), schedule)])).unwrap()
    }

    #[cfg(test)]
    pub(crate) fn for_test_with_budgets(
        path: &Path,
        budget_path: &Path,
        alias: &str,
        metadata: &OpenRouterModelMetadata,
    ) -> Self {
        let schedule = PriceSchedule::from_metadata(alias, metadata).unwrap();
        Self::open_with_budgets(
            path,
            HashMap::from([(alias.to_string(), schedule)]),
            Some(budget_path),
        )
        .unwrap()
    }
}

pub(crate) struct PendingReceipt {
    ledger: Ledger,
    request_id: String,
    tenant: String,
    model: String,
    route: &'static str,
    lane: &'static str,
    stream: bool,
    started_unix_ms: u64,
    partial_usage: Usage,
    budget_permit: Option<BudgetPermit>,
    pending_debit: Option<(u64, String)>,
    finalized: bool,
    capture: Option<CaptureArm>,
}

/// Capture payload riding a receipt: the prompt as received and the completion as
/// generated. Flushed only from a successful `complete()` — rejects, drops, and
/// abandonments discard it with the receipt.
pub(crate) struct CaptureArm {
    store: crate::capture::CaptureStore,
    prompt: serde_json::Value,
    completion: String,
}

impl PendingReceipt {
    /// Attach capture to this request. Callers gate on the store's request-start check;
    /// the settle-time re-check inside `flush_capture` is the authoritative one.
    pub(crate) fn arm_capture(
        &mut self,
        store: crate::capture::CaptureStore,
        prompt: serde_json::Value,
    ) {
        self.capture = Some(CaptureArm {
            store,
            prompt,
            completion: String::new(),
        });
    }

    /// Raw generated text, accumulated at the same sites that count completion tokens.
    /// No-op unless capture is armed.
    pub(crate) fn capture_completion_delta(&mut self, text: &str) {
        if let Some(arm) = self.capture.as_mut() {
            arm.completion.push_str(text);
        }
    }

    /// Settle-time capture decision. The tenant's mark is re-read HERE so clearing it
    /// stops capture for in-flight requests; a `trial` mark additionally requires the
    /// settled balance to be non-negative, so spend past the marked allowance is never
    /// retained (under-capture is the designed failure direction). Fail-open: nothing
    /// in here can fail the request — the row is either enqueued or dropped-with-count.
    fn flush_capture(&mut self, usage: Usage) {
        let Some(arm) = self.capture.take() else {
            return;
        };
        let Some(mode) = arm.store.mode(&self.tenant) else {
            return;
        };
        if mode == crate::capture::CaptureMode::Trial {
            let settled_non_negative = self
                .ledger
                .inner
                .budgets
                .as_ref()
                .and_then(|budgets| budgets.balance(&self.tenant).ok().flatten())
                .is_some_and(|view| view.balance_micro >= 0);
            if !settled_non_negative {
                return;
            }
        }
        arm.store.submit(&json!({
            "format": crate::capture::FORMAT,
            "request_id": self.request_id,
            "started_unix_ms": self.started_unix_ms,
            "finished_unix_ms": unix_ms(),
            "tenant": self.tenant,
            "model": self.model,
            "route": self.route,
            "posture": mode.posture(),
            "prompt": arm.prompt,
            "completion": arm.completion,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        }));
    }

    pub(crate) fn record_prompt_usage(
        &mut self,
        prompt_tokens: u64,
        cached_prompt_tokens: u64,
    ) -> Result<(), String> {
        if cached_prompt_tokens > prompt_tokens {
            return Err(format!(
                "cached prompt tokens {cached_prompt_tokens} exceed total prompt tokens {prompt_tokens}"
            ));
        }
        self.partial_usage.prompt_tokens = prompt_tokens;
        self.partial_usage.cached_prompt_tokens = cached_prompt_tokens;
        Ok(())
    }

    pub(crate) fn record_completion_token(&mut self) -> Result<(), String> {
        self.partial_usage.completion_tokens = self
            .partial_usage
            .completion_tokens
            .checked_add(1)
            .ok_or_else(|| "partial completion token count overflowed u64".to_string())?;
        Ok(())
    }

    pub(crate) fn complete(&mut self, usage: Usage, worker_elapsed_s: f64) -> Result<(), String> {
        let mut row = self.base_row("completed", 200, None);
        self.add_accounting(&mut row, usage)?;
        row["worker_elapsed_s"] = json!(worker_elapsed_s);
        self.finalize(row)?;
        // Only a fully settled request may be retained: the trial guard below reads the
        // POST-debit balance, and a finalize failure must not leave a captured row for a
        // request whose billing state is indeterminate.
        self.flush_capture(usage);
        Ok(())
    }

    pub(crate) fn reject(&mut self, status: u16, error_code: &str) -> Result<(), String> {
        self.settle_unbilled("rejected", status, error_code)
    }

    /// Terminal row for a response CUT by the request deadline and delivered anyway: 200,
    /// outcome `deadline_partial`, and the error_code the response itself carries. Bills the
    /// delivered tokens exactly like `complete`, but stays distinguishable from a full answer
    /// in the census — the reason a review flagged `completed` here as information loss.
    pub(crate) fn complete_deadline_partial(
        &mut self,
        usage: Usage,
        worker_elapsed_s: f64,
    ) -> Result<(), String> {
        let mut row = self.base_row("deadline_partial", 200, Some("deadline_exceeded"));
        self.add_accounting(&mut row, usage)?;
        row["worker_elapsed_s"] = json!(worker_elapsed_s);
        self.finalize(row)?;
        self.flush_capture(usage);
        Ok(())
    }

    /// Terminal row for a request that ends with NO debit by billing policy (owner ruling
    /// 2026-08-23: a non-response that is our fault costs nothing). The row carries the
    /// named outcome (`rejected`, `deadline_exceeded`, `shed_deadline`, `shed_queue`) and
    /// null usage/cost/budget; the un-settled reservation is refunded in full when the
    /// `BudgetPermit` drops.
    pub(crate) fn settle_unbilled(
        &mut self,
        outcome: &'static str,
        status: u16,
        error_code: &str,
    ) -> Result<(), String> {
        debug_assert!(
            !outcome_may_debit(outcome),
            "settle_unbilled cannot write a billable outcome; use complete()"
        );
        let row = self.base_row(outcome, status, Some(error_code));
        self.finalize(row)
    }

    fn base_row(
        &self,
        outcome: &str,
        http_status: u16,
        error_code: Option<&str>,
    ) -> serde_json::Value {
        json!({
            "format": FORMAT,
            "request_id": self.request_id,
            "started_unix_ms": self.started_unix_ms,
            "finished_unix_ms": unix_ms(),
            "tenant": self.tenant,
            "model": self.model,
            "route": self.route,
            "lane": self.lane,
            "stream": self.stream,
            "outcome": outcome,
            "http_status": http_status,
            "error_code": error_code,
            "usage": null,
            "unit_prices_usd": null,
            "cost_usd": null,
            "budget": null,
        })
    }

    fn add_accounting(&mut self, row: &mut serde_json::Value, usage: Usage) -> Result<(), String> {
        let schedule =
            self.ledger.inner.prices.get(&self.model).ok_or_else(|| {
                format!("request ledger has no price schedule for {:?}", self.model)
            })?;
        let accounting = schedule.cost(usage)?;
        row["usage"] = accounting["usage"].clone();
        row["unit_prices_usd"] = accounting["unit_prices_usd"].clone();
        row["cost_usd"] = accounting["cost_usd"].clone();
        if let Some(permit) = self.budget_permit.as_ref() {
            let total = accounting["cost_usd"]["total"]
                .as_str()
                .ok_or_else(|| "request accounting has no decimal total".to_string())?;
            let debit_micro = Decimal::parse(total)?.to_micro_usd_ceil()?;
            row["budget"] = json!({
                "currency": "USD",
                "reserved_micro": permit.reserved_micro,
                "debit_micro": debit_micro,
                "rounding": "ceil_positive_request_to_micro_usd",
            });
            // Kept separately from the JSON value so the journal transaction cannot be
            // redirected by serialized input. The request ledger is synced first; journal
            // replay repairs a crash in the gap between these two durable appends.
            self.pending_debit = Some((debit_micro, total.to_string()));
        }
        Ok(())
    }

    fn finalize(&mut self, mut row: serde_json::Value) -> Result<(), String> {
        if self.finalized {
            return Err(format!(
                "request {} already has a terminal ledger row",
                self.request_id
            ));
        }
        // FAULT-ATTRIBUTION INVARIANT (owner ruling 2026-08-23): only the outcomes in
        // `BILLABLE_OUTCOMES` may debit > 0 — `completed`, `abandoned`, and (since
        // lane/deadline-partial-20260826) `deadline_partial`. Stated as the CONSTANT rather
        // than a hand-copied list: this same sentence has been restated in prose four times
        // in this file's history and went stale the moment the set grew.
        // This is the last write point every terminal row
        // passes through, so an outcome added without visiting the census table cannot
        // bill by accident. The row's `budget` field must be nulled too — journal
        // recovery (`recover_debit`) replays any row with a non-null budget, outcome-blind.
        let outcome = row["outcome"].as_str().unwrap_or("").to_string();
        debug_assert!(
            outcome_may_debit(&outcome) || ZERO_DEBIT_OUTCOMES.contains(&outcome.as_str()),
            "unknown ledger outcome {outcome:?}: add it to the census table (and its \
             debit gate) before writing it"
        );
        if !outcome_may_debit(&outcome) && !row["cost_usd"].is_null() {
            eprintln!(
                "[ledger] ERROR: request {} outcome {outcome:?} carried accounting; \
                 stripped it (only completed/abandoned/deadline_partial may bill)",
                self.request_id
            );
            self.pending_debit = None;
            // ALL FOUR accounting fields go, not just `budget`: recovery replays a
            // non-null `budget` outcome-blind, and any revenue report summing `cost_usd`
            // would otherwise count a request the customer was never charged for. A
            // zero-debit row is byte-shaped like every other one (`base_row` defaults).
            for field in ["usage", "unit_prices_usd", "cost_usd", "budget"] {
                row[field] = serde_json::Value::Null;
            }
        }
        // An append whose sync fails has an indeterminate durability state. Never retry the
        // same request id from Drop and risk a duplicate bill; fail the HTTP completion loud.
        self.finalized = true;
        self.ledger.append(&row)?;
        if let Some(permit) = self.budget_permit.as_mut()
            && let Some((amount_micro, exact_cost_usd)) = self.pending_debit.as_ref()
        {
            permit.settle(&self.request_id, *amount_micro, exact_cost_usd)?;
        }
        Ok(())
    }
}

impl Drop for PendingReceipt {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        // FAULT ATTRIBUTION (owner ruling 2026-08-23): a drop without a terminal row is
        // only a client walk-away when the process is healthy. During a panic unwind or
        // past the drain deadline the non-response is OURS — named outcome, zero debit.
        if std::thread::panicking() {
            let row = self.base_row("crashed", 500, Some("handler_panicked"));
            if let Err(err) = self.finalize(row) {
                eprintln!(
                    "[ledger] ERROR: could not persist crashed request {}: {err}",
                    self.request_id
                );
            }
            return;
        }
        if drain_kill_active() {
            let row = self.base_row("drain_killed", 503, Some("drain_deadline_kill"));
            if let Err(err) = self.finalize(row) {
                eprintln!(
                    "[ledger] ERROR: could not persist drain-killed request {}: {err}",
                    self.request_id
                );
            }
            return;
        }
        let mut row = self.base_row(
            "abandoned",
            499,
            Some("client_disconnected_or_handler_dropped"),
        );
        if let Err(err) = self.add_accounting(&mut row, self.partial_usage) {
            eprintln!(
                "[ledger] ERROR: could not price abandoned request {}: {err}",
                self.request_id
            );
        }
        if let Err(err) = self.finalize(row) {
            eprintln!(
                "[ledger] ERROR: could not persist/debit abandoned request {}: {err}",
                self.request_id
            );
        }
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "memra-{label}-{}-{}",
            std::process::id(),
            super::unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        dir
    }

    fn write_private(path: &Path, contents: &str) {
        std::fs::write(path, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o640)).unwrap();
        }
    }

    fn budget_source(path: &Path, tenant: &str, balance_micro: i64) {
        write_private(
            path,
            &format!(
                "[[budgets]]\ntenant = {tenant:?}\ncurrency = \"USD\"\nbalance_micro = {balance_micro}\n"
            ),
        );
    }

    fn schedule() -> PriceSchedule {
        let metadata = OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.000000289".into()),
                cached_prompt: Some("0.0000000289".into()),
                completion: Some("0.0000024".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        PriceSchedule::from_metadata("q27", &metadata).unwrap()
    }

    #[test]
    fn decimal_math_is_exact_across_different_scales() {
        let accounting = schedule()
            .cost(Usage {
                prompt_tokens: 100,
                cached_prompt_tokens: 90,
                completion_tokens: 5,
            })
            .unwrap();
        assert_eq!(accounting["usage"]["ordinary_prompt_tokens"], 10);
        assert_eq!(accounting["cost_usd"]["ordinary_prompt"], "0.000002890");
        assert_eq!(accounting["cost_usd"]["cached_prompt"], "0.0000026010");
        assert_eq!(accounting["cost_usd"]["completion"], "0.0000120");
        assert_eq!(accounting["cost_usd"]["total"], "0.0000174910");
    }

    #[test]
    fn prepaid_rounding_is_integer_micro_usd_and_never_rounds_positive_cost_down() {
        assert_eq!(Decimal::parse("0").unwrap().to_micro_usd_ceil().unwrap(), 0);
        assert_eq!(
            Decimal::parse("0.000000000001")
                .unwrap()
                .to_micro_usd_ceil()
                .unwrap(),
            1,
        );
        assert_eq!(
            Decimal::parse("0.0000174910")
                .unwrap()
                .to_micro_usd_ceil()
                .unwrap(),
            18,
        );
        assert_eq!(
            Decimal::parse("1.000001")
                .unwrap()
                .to_micro_usd_ceil()
                .unwrap(),
            1_000_001,
        );
    }

    #[test]
    fn worst_case_reservation_refuses_overdraft_and_refunds_unused_hold() {
        let dir = test_dir("budget-reservation");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 40);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        // 100 ordinary prompt tokens + 5 completion tokens cost 40.9 micro-USD.
        assert_eq!(
            ledger
                .reserve_budget("tenant-a", "q27", 100, 5)
                .unwrap_err(),
            BudgetAdmissionError::Insufficient,
        );
        let budgets = ledger.budgets().unwrap();
        budgets.credit("tenant-a", 1, "reservation-topup").unwrap();
        let permit = ledger
            .reserve_budget("tenant-a", "q27", 100, 5)
            .unwrap()
            .unwrap();
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            0
        );
        let mut receipt = ledger.start_with_budget(
            "chatcmpl-reserved",
            "tenant-a",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
            Some(permit),
        );
        receipt
            .complete(
                Usage {
                    prompt_tokens: 100,
                    cached_prompt_tokens: 90,
                    completion_tokens: 5,
                },
                0.25,
            )
            .unwrap();
        drop(receipt);
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            23
        );
        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&request_path).unwrap().trim()).unwrap();
        assert_eq!(row["budget"]["reserved_micro"], 41);
        assert_eq!(row["budget"]["debit_micro"], 18);
        ledger.inner.failed.store(true, Ordering::Release);
        assert!(matches!(
            ledger.reserve_budget("tenant-a", "q27", 1, 1),
            Err(BudgetAdmissionError::Unavailable(message))
                if message.contains("request ledger")
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn capture_flush_honors_posture_balance_and_terminal_outcome() {
        use crate::capture::{CaptureMode, CaptureStore};

        let dir = test_dir("capture-quadrants");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "trial-ok", 1_000);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        let capture_dir = dir.join("capture");
        let store = CaptureStore::open(&capture_dir).unwrap();
        store
            .set_mode("trial-ok", Some(CaptureMode::Trial))
            .unwrap();
        store
            .set_mode("trial-over", Some(CaptureMode::Trial))
            .unwrap();
        store.set_mode("cleared", Some(CaptureMode::Trial)).unwrap();
        store
            .set_mode("rejected", Some(CaptureMode::Consent))
            .unwrap();
        store
            .set_mode("consented", Some(CaptureMode::Consent))
            .unwrap();
        let prompt = json!([{ "role": "user", "content": "hi" }]);
        let usage_small = Usage {
            prompt_tokens: 100,
            cached_prompt_tokens: 90,
            completion_tokens: 5,
        };

        // NEGATIVE 1 — trial mark WITHOUT a budget account: the allowance cannot be
        // proven (balance() is None for the unenrolled), so the flush under-captures.
        // (Settle itself caps the debit at the reservation, so an enrolled tenant can
        // never settle negative through a single request — the balance guard is the
        // defense for exactly this unprovable case and any future negative-balance path.)
        let mut receipt = ledger.start(
            "cap-over",
            "trial-over",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
        );
        receipt.arm_capture(store.clone(), prompt.clone());
        receipt.capture_completion_delta("kept? ");
        receipt.complete(usage_small, 0.1).unwrap();
        drop(receipt);
        assert!(
            budgets.balance("trial-over").unwrap().is_none(),
            "this tenant must be unenrolled for the case to mean anything",
        );

        // NEGATIVE 2 — mark cleared while the request was in flight: opt-out is
        // immediate, the settle-time re-check drops the row.
        let mut receipt = ledger.start(
            "cap-cleared",
            "cleared",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
        );
        receipt.arm_capture(store.clone(), prompt.clone());
        receipt.capture_completion_delta("kept? ");
        store.set_mode("cleared", None).unwrap();
        receipt.complete(usage_small, 0.1).unwrap();
        drop(receipt);

        // NEGATIVE 3 — terminal reject: only completed requests are corpus material.
        let mut receipt = ledger.start(
            "cap-rejected",
            "rejected",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
        );
        receipt.arm_capture(store.clone(), prompt.clone());
        receipt.reject(429, "rate_limit_exceeded").unwrap();
        drop(receipt);

        // POSITIVE 1 — consent mark: captured regardless of budget state (opt-in is
        // the paid-tenant switch; there may be no budget account at all).
        let mut receipt = ledger.start(
            "cap-consent",
            "consented",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
        );
        receipt.arm_capture(store.clone(), prompt.clone());
        receipt.capture_completion_delta("Hello ");
        receipt.capture_completion_delta("world");
        receipt.complete(usage_small, 0.1).unwrap();
        drop(receipt);

        // POSITIVE 2 — trial mark with a non-negative settled balance.
        let permit = ledger.reserve_budget("trial-ok", "q27", 100, 5).unwrap();
        let mut receipt = ledger.start_with_budget(
            "cap-trial",
            "trial-ok",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
            permit,
        );
        receipt.arm_capture(store.clone(), prompt.clone());
        receipt.capture_completion_delta("trial text");
        receipt.complete(usage_small, 0.1).unwrap();
        drop(receipt);

        // The writer is async; wait for the LAST submitted row, then assert the set.
        let rows_path = capture_dir.join("capture-000000.jsonl");
        let rows: Vec<serde_json::Value> = (0..400)
            .find_map(|_| {
                let text = std::fs::read_to_string(&rows_path).ok()?;
                let rows: Vec<serde_json::Value> = text
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                if rows.len() >= 2 {
                    Some(rows)
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    None
                }
            })
            .expect("captured rows never arrived");
        assert_eq!(rows.len(), 2, "exactly the two positive cases: {rows:?}");
        assert_eq!(rows[0]["request_id"], "cap-consent");
        assert_eq!(rows[0]["posture"], "opt_in");
        assert_eq!(rows[0]["prompt"], prompt);
        assert_eq!(rows[0]["completion"], "Hello world");
        assert_eq!(rows[0]["usage"]["prompt_tokens"], 100);
        assert_eq!(rows[1]["request_id"], "cap-trial");
        assert_eq!(rows[1]["posture"], "trial");
        assert_eq!(rows[1]["completion"], "trial text");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn budgeted_completion_debits_exact_cached_price_once() {
        let dir = test_dir("budget-cached");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 1_000);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        assert!(budgets.admit("unlisted", 100).unwrap().is_none());
        let permit = budgets.admit("tenant-a", 100).unwrap().unwrap();
        // CONCURRENT permits: a second in-flight request admits while the balance
        // covers the sum of reservations (100 + 800 <= 1000)...
        let second = budgets.admit("tenant-a", 800).unwrap().unwrap();
        // ...and the balance bounds the SUM: 100 remaining < 101 asked.
        assert_eq!(
            budgets.admit("tenant-a", 101).unwrap_err(),
            BudgetAdmissionError::Insufficient,
        );
        drop(second); // refund the probe reservation
        let mut receipt = ledger.start_with_budget(
            "chatcmpl-budget",
            "tenant-a",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
            Some(permit),
        );
        receipt
            .complete(
                Usage {
                    prompt_tokens: 100,
                    cached_prompt_tokens: 90,
                    completion_tokens: 5,
                },
                0.25,
            )
            .unwrap();
        drop(receipt);
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            982
        );
        let snapshot: toml::Value =
            toml::from_str(&std::fs::read_to_string(&budgets.inner.snapshot_path).unwrap())
                .unwrap();
        let snapshot_balance = snapshot["balances"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["tenant"].as_str() == Some("tenant-a"))
            .and_then(|row| row["balance_micro"].as_integer());
        assert_eq!(snapshot_balance, Some(982));

        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&request_path).unwrap().trim()).unwrap();
        assert_eq!(row["cost_usd"]["total"], "0.0000174910");
        assert_eq!(row["budget"]["debit_micro"], 18);
        let journal: Vec<BudgetJournalRow> = read_jsonl(budgets.journal_path()).unwrap();
        assert_eq!(journal.len(), 1);
        assert_eq!(journal[0].request_id.as_deref(), Some("chatcmpl-budget"));
        assert_eq!(journal[0].amount_micro, 18);
        assert_eq!(journal[0].exact_cost_usd.as_deref(), Some("0.0000174910"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn zero_balance_is_denied_and_abandoned_usage_is_debited() {
        // Drop-classified outcome: serialized against the drain-kill latch test.
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("budget-abandon");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "zero", 0);
        write_private(
            &budget_path,
            &"[[budgets]]\ntenant = \"zero\"\ncurrency = \"USD\"\nbalance_micro = 0\n\n\
                 [[budgets]]\ntenant = \"live\"\ncurrency = \"USD\"\nbalance_micro = 100\n"
                .to_string(),
        );
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        assert_eq!(
            budgets.admit("zero", 1).unwrap_err(),
            BudgetAdmissionError::Insufficient,
        );
        {
            let permit = budgets.admit("live", 20).unwrap().unwrap();
            let mut receipt = ledger.start_with_budget(
                "chatcmpl-abandon-budget",
                "live",
                "q27",
                "/v1/chat/completions",
                "interactive",
                true,
                Some(permit),
            );
            receipt.record_prompt_usage(100, 90).unwrap();
            receipt.record_completion_token().unwrap();
            receipt.record_completion_token().unwrap();
        }
        // Exact ledger cost is 10.291 micro-USD, conservatively rounded to 11.
        assert_eq!(budgets.balance("live").unwrap().unwrap().balance_micro, 89);
        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&request_path).unwrap().trim()).unwrap();
        assert_eq!(row["outcome"], "abandoned");
        assert_eq!(row["cost_usd"]["total"], "0.0000102910");
        assert_eq!(row["budget"]["debit_micro"], 11);
        std::fs::remove_dir_all(dir).unwrap();
    }

    // ---- outcome -> debit census gates (owner ruling 2026-08-23) --------------------
    //
    // Only `completed`, `abandoned` and `deadline_partial` may debit > 0; every other outcome
    // debits ZERO. (`deadline_partial` joined the billable set in
    // lane/deadline-partial-20260826: the deadline cut generation and we DELIVERED the tokens
    // produced, so the caller received value — unlike `deadline_exceeded`, which delivers
    // nothing and bills nothing.)
    // The billable halves have their own gates above (`budgeted_completion_debits_...`,
    // `zero_balance_..._abandoned_usage_is_debited`); each zero-debit outcome gets one
    // gate here, driven with partial usage ALREADY ACCRUED so the gate proves "work
    // happened and the customer still pays nothing", not "nothing happened".

    /// One census setup: budgeted tenant at 1000 micro-USD, a 100-micro reservation, and
    /// a receipt with recorded partial usage (prompt 100 + one completion token).
    fn census_receipt(dir: &Path, request_id: &str) -> (Ledger, PendingReceipt) {
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "t", 1_000);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let permit = ledger.budgets().unwrap().admit("t", 100).unwrap().unwrap();
        let mut receipt = ledger.start_with_budget(
            request_id,
            "t",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
            Some(permit),
        );
        receipt.record_prompt_usage(100, 0).unwrap();
        receipt.record_completion_token().unwrap();
        (ledger, receipt)
    }

    /// Post-settle assertions shared by every zero-debit gate: full refund, a row with
    /// the named outcome and NO accounting, and an empty debit journal.
    fn assert_zero_debit(dir: &Path, ledger: &Ledger, outcome: &str, status: u16, code: &str) {
        let budgets = ledger.budgets().unwrap();
        assert_eq!(
            budgets.balance("t").unwrap().unwrap().balance_micro,
            1_000,
            "{outcome} must refund the reservation in full"
        );
        let row: serde_json::Value = serde_json::from_str(
            std::fs::read_to_string(dir.join("requests.jsonl"))
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(row["outcome"], outcome);
        assert_eq!(row["http_status"], status);
        assert_eq!(row["error_code"], code);
        assert!(
            row["usage"].is_null() && row["cost_usd"].is_null() && row["budget"].is_null(),
            "{outcome} row must carry no accounting: {row}"
        );
        let journal: Vec<BudgetJournalRow> = read_jsonl(budgets.journal_path()).unwrap();
        assert!(
            journal.is_empty(),
            "{outcome} must write no debit journal row: {journal:?}"
        );
    }

    #[test]
    fn outcome_census_is_disjoint() {
        for billable in BILLABLE_OUTCOMES {
            assert!(
                !ZERO_DEBIT_OUTCOMES.contains(&billable),
                "{billable} cannot be in both halves of the census"
            );
        }
    }

    #[test]
    fn rejected_settles_with_zero_debit_and_full_refund() {
        let dir = test_dir("census-rejected");
        let (ledger, mut receipt) = census_receipt(&dir, "req-rejected");
        receipt.reject(500, "engine_error").unwrap();
        drop(receipt);
        assert_zero_debit(&dir, &ledger, "rejected", 500, "engine_error");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deadline_exceeded_settles_with_zero_debit_and_full_refund() {
        let dir = test_dir("census-deadline");
        let (ledger, mut receipt) = census_receipt(&dir, "req-deadline");
        receipt
            .settle_unbilled("deadline_exceeded", 408, "deadline_exceeded")
            .unwrap();
        drop(receipt);
        assert_zero_debit(&dir, &ledger, "deadline_exceeded", 408, "deadline_exceeded");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shed_deadline_settles_with_zero_debit_and_full_refund() {
        let dir = test_dir("census-shed-deadline");
        let (ledger, mut receipt) = census_receipt(&dir, "req-shed-deadline");
        receipt
            .settle_unbilled("shed_deadline", 429, "shed_deadline")
            .unwrap();
        drop(receipt);
        assert_zero_debit(&dir, &ledger, "shed_deadline", 429, "shed_deadline");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shed_queue_settles_with_zero_debit_and_full_refund() {
        let dir = test_dir("census-shed-queue");
        let (ledger, mut receipt) = census_receipt(&dir, "req-shed-queue");
        receipt
            .settle_unbilled("shed_queue", 429, "shed_queue")
            .unwrap();
        drop(receipt);
        assert_zero_debit(&dir, &ledger, "shed_queue", 429, "shed_queue");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_panicking_handler_drop_settles_crashed_with_zero_debit() {
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("census-crashed");
        let (ledger, receipt) = census_receipt(&dir, "req-crashed");
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _held = receipt; // dropped mid-unwind: thread::panicking() is true
            panic!("boom: handler died holding the receipt");
        }));
        assert!(unwound.is_err());
        assert_zero_debit(&dir, &ledger, "crashed", 500, "handler_panicked");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_drain_deadline_kill_drop_settles_drain_killed_with_zero_debit() {
        // Serialized against every other drop-classified test: the latch is process-global
        // and an unrelated receipt dropped inside this window would be misclassified.
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("census-drain-killed");
        let (ledger, receipt) = census_receipt(&dir, "req-drain-killed");
        mark_drain_kill();
        drop(receipt);
        reset_drain_kill_for_test();
        assert_zero_debit(&dir, &ledger, "drain_killed", 503, "drain_deadline_kill");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The finalize invariant proven able to fail (a gate is watched red before it is
    /// trusted): a debit smuggled onto a non-billable outcome is REFUSED at the last
    /// write point — row budget nulled, no journal debit, reservation refunded. Drives
    /// the private `add_accounting` -> `finalize` path directly, because every public
    /// path is already correct by construction and could not exercise the strip.
    #[test]
    fn a_debit_on_a_non_billable_outcome_is_refused_at_finalize() {
        let dir = test_dir("census-invariant");
        let (ledger, mut receipt) = census_receipt(&dir, "req-invariant");
        let usage = receipt.partial_usage;
        let mut row = receipt.base_row("rejected", 500, Some("engine_error"));
        receipt.add_accounting(&mut row, usage).unwrap();
        assert!(
            receipt.pending_debit.is_some(),
            "the smuggle must be armed before finalize can refuse it"
        );
        receipt.finalize(row).unwrap();
        assert!(
            receipt.pending_debit.is_none(),
            "finalize must drop the debit"
        );
        drop(receipt); // release the reservation before reading the balance
        assert_zero_debit(&dir, &ledger, "rejected", 500, "engine_error");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn idempotent_credit_replay_returns_original_result_before_and_after_restart() {
        let dir = test_dir("budget-credit");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 50);
        let original;
        {
            let ledger = Ledger::open_with_budgets(
                &request_path,
                HashMap::from([("q27".into(), schedule())]),
                Some(&budget_path),
            )
            .unwrap();
            let budgets = ledger.budgets().unwrap();
            original = budgets.credit("tenant-a", 25, "credit-event-1").unwrap();
            assert_eq!(original.balance_micro, 75);
            assert_eq!(
                budgets.credit("tenant-a", 25, "credit-event-1").unwrap(),
                original,
            );
            assert!(matches!(
                budgets.credit("tenant-a", 26, "credit-event-1"),
                Err(CreditError::Conflict(_)),
            ));
            assert_eq!(
                read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                    .unwrap()
                    .len(),
                1
            );
        }
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        assert_eq!(
            budgets.credit("tenant-a", 25, "credit-event-1").unwrap(),
            original,
        );
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            75
        );
        assert_eq!(
            read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn admin_debit_reduces_the_balance_journals_a_replayable_row_and_survives_restart() {
        let dir = test_dir("budget-admin-debit");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 100);
        {
            let ledger = Ledger::open_with_budgets(
                &request_path,
                HashMap::from([("q27".into(), schedule())]),
                Some(&budget_path),
            )
            .unwrap();
            let budgets = ledger.budgets().unwrap();
            let result = budgets
                .admin_debit("tenant-a", 30, "migrate-de-1", "DE spend migration")
                .unwrap();
            assert_eq!(result.balance_micro, 70);
            assert_eq!(result.amount_micro, 30);
            let rows = read_jsonl::<BudgetJournalRow>(budgets.journal_path()).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].kind, "debit");
            assert_eq!(
                rows[0].request_id.as_deref(),
                Some("admin-debit:tenant-a:migrate-de-1")
            );
            assert_eq!(
                rows[0].exact_cost_usd.as_deref(),
                Some("admin-debit reason: DE spend migration")
            );
            assert_eq!(rows[0].balance_after_micro, Some(70));
        }
        // Restart: the journal row replays through the SAME debit arm request
        // settlement uses, and the idempotency survives via the debits map.
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            70
        );
        let replay = budgets
            .admin_debit("tenant-a", 30, "migrate-de-1", "DE spend migration")
            .unwrap();
        assert_eq!(replay.balance_micro, 70, "replay must not debit twice");
        assert!(matches!(
            budgets.admin_debit("tenant-a", 31, "migrate-de-1", "DE spend migration"),
            Err(CreditError::Conflict(_)),
        ));
        assert_eq!(
            read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn admin_debit_may_take_the_balance_negative_and_admission_then_refuses() {
        let dir = test_dir("budget-admin-debit-negative");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 50);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        let result = budgets
            .admin_debit(
                "tenant-a",
                80,
                "clawback-1",
                "cross-box migration overshoot case",
            )
            .unwrap();
        assert_eq!(result.balance_micro, -30);
        assert!(
            matches!(
                budgets.admit("tenant-a", 1),
                Err(BudgetAdmissionError::Insufficient)
            ),
            "a negative balance must refuse admission"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn admin_debit_refuses_bad_input_and_unenrolled_tenants() {
        let dir = test_dir("budget-admin-debit-refusals");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 50);
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        assert!(matches!(
            budgets.admin_debit("tenant-a", 0, "k1", "reason"),
            Err(CreditError::Invalid(_)),
        ));
        assert!(matches!(
            budgets.admin_debit("tenant-a", -5, "k1", "reason"),
            Err(CreditError::Invalid(_)),
        ));
        assert!(matches!(
            budgets.admin_debit("tenant-a", 5, "k1", "   "),
            Err(CreditError::Invalid(_)),
        ));
        assert!(matches!(
            budgets.admin_debit("tenant-a", 5, "", "reason"),
            Err(CreditError::Invalid(_)),
        ));
        assert!(
            matches!(
                budgets.admin_debit("nobody", 5, "k1", "reason"),
                Err(CreditError::Invalid(_)),
            ),
            "an unenrolled tenant is UNMETERED — debiting it would silently enroll a ghost"
        );
        assert_eq!(
            read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                .unwrap()
                .len(),
            0,
            "every refusal above must leave the journal untouched"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn admin_debit_replay_is_tenant_bound_so_one_key_cannot_answer_for_another_tenant() {
        let dir = test_dir("budget-admin-debit-tenant-bound");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        write_private(
            &budget_path,
            "[[budgets]]\ntenant = \"tenant-a\"\ncurrency = \"USD\"\nbalance_micro = 100\n\n\
             [[budgets]]\ntenant = \"tenant-b\"\ncurrency = \"USD\"\nbalance_micro = 100\n",
        );
        let ledger = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .unwrap();
        let budgets = ledger.budgets().unwrap();
        // A migration script reusing ONE key across tenants must debit each tenant,
        // not return tenant-a's receipt for tenant-b (the fabricated-200 defect).
        assert_eq!(
            budgets
                .admin_debit("tenant-a", 30, "batch-key", "migration")
                .unwrap()
                .balance_micro,
            70
        );
        assert_eq!(
            budgets
                .admin_debit("tenant-b", 30, "batch-key", "migration")
                .unwrap()
                .balance_micro,
            70,
            "same key, different tenant: a REAL debit, not a replayed receipt"
        );
        // And the same key against an unenrolled tenant is still a refusal, never a 200.
        assert!(matches!(
            budgets.admin_debit("nobody", 30, "batch-key", "migration"),
            Err(CreditError::Invalid(_)),
        ));
        assert_eq!(
            read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                .unwrap()
                .len(),
            2
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn request_ledger_repairs_missing_debit_without_double_charge() {
        let dir = test_dir("budget-recovery");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 100);
        write_private(
            &request_path,
            &format!(
                "{}\n",
                json!({
                    "format": FORMAT,
                    "request_id": "chatcmpl-crash-gap",
                    "tenant": "tenant-a",
                    "cost_usd": {"total": "0.0000174910"},
                    "budget": {"currency": "USD", "debit_micro": 18},
                }),
            ),
        );
        for _ in 0..2 {
            let ledger = Ledger::open_with_budgets(
                &request_path,
                HashMap::from([("q27".into(), schedule())]),
                Some(&budget_path),
            )
            .unwrap();
            let budgets = ledger.budgets().unwrap();
            assert_eq!(
                budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
                82
            );
            assert_eq!(
                read_jsonl::<BudgetJournalRow>(budgets.journal_path())
                    .unwrap()
                    .len(),
                1
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn restart_refuses_a_source_that_forgets_a_previously_debited_tenant() {
        let dir = test_dir("budget-source-truncated");
        let request_path = dir.join("requests.jsonl");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 100);
        {
            let ledger = Ledger::open_with_budgets(
                &request_path,
                HashMap::from([("q27".into(), schedule())]),
                Some(&budget_path),
            )
            .unwrap();
            let permit = ledger
                .budgets()
                .unwrap()
                .admit("tenant-a", 20)
                .unwrap()
                .unwrap();
            let mut receipt = ledger.start_with_budget(
                "chatcmpl-source-truncated",
                "tenant-a",
                "q27",
                "/v1/completions",
                "interactive",
                false,
                Some(permit),
            );
            receipt
                .complete(
                    Usage {
                        prompt_tokens: 1,
                        cached_prompt_tokens: 0,
                        completion_tokens: 1,
                    },
                    0.1,
                )
                .unwrap();
        }
        write_private(&budget_path, "");
        let err = Ledger::open_with_budgets(
            &request_path,
            HashMap::from([("q27".into(), schedule())]),
            Some(&budget_path),
        )
        .err()
        .expect("truncating the append-only source must fail closed on restart");
        assert!(err.contains("debits unenrolled tenant"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn append_only_source_credit_hot_reloads() {
        let dir = test_dir("budget-reload");
        let request_path = dir.join("requests.jsonl");
        write_private(&request_path, "");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 0);
        let budgets = TenantBudgets::open(&budget_path, &request_path)
            .unwrap()
            .with_poll(Duration::ZERO);
        assert_eq!(
            budgets.admit("tenant-a", 1).unwrap_err(),
            BudgetAdmissionError::Insufficient,
        );
        use std::io::Write as _;
        let mut source = OpenOptions::new().append(true).open(&budget_path).unwrap();
        source
            .write_all(
                b"\n[[budgets]]\ntenant = \"tenant-a\"\ncurrency = \"USD\"\nbalance_micro = 25\n",
            )
            .unwrap();
        source.sync_data().unwrap();
        drop(source);
        let permit = budgets.admit("tenant-a", 1).unwrap().unwrap();
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            24
        );
        drop(permit);
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            25
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn repeated_source_reload_failures_retry_then_fail_closed_and_recover() {
        let dir = test_dir("budget-reload-fail-closed");
        let request_path = dir.join("requests.jsonl");
        write_private(&request_path, "");
        let budget_path = dir.join("budgets.toml");
        budget_source(&budget_path, "tenant-a", 10);
        let budgets = TenantBudgets::open(&budget_path, &request_path)
            .unwrap()
            .with_poll(Duration::ZERO);

        std::fs::remove_file(&budget_path).unwrap();
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            10
        );
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            10
        );
        let err = budgets.balance("tenant-a").unwrap_err();
        assert!(err.contains("failed 3 consecutive reload polls"), "{err}");
        let health = budgets.health();
        assert!(health.source_reload_failed >= 3);
        assert!(health.source_reload_consecutive >= 3);
        assert!(!health.source_available);

        // Recovery requires a successful parse, not a distinct timestamp transition.
        budget_source(&budget_path, "tenant-a", 20);
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            20
        );
        let health = budgets.health();
        assert_eq!(health.source_reload_consecutive, 0);
        assert!(health.source_available);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn jsonl_budget_source_adds_integer_rows_and_keeps_absent_tenants_unlimited() {
        let dir = test_dir("budget-jsonl");
        let request_path = dir.join("requests.jsonl");
        write_private(&request_path, "");
        let budget_path = dir.join("budgets.jsonl");
        write_private(
            &budget_path,
            concat!(
                "{\"tenant\":\"tenant-a\",\"currency\":\"USD\",\"balance_micro\":10}\n",
                "{\"tenant\":\"tenant-a\",\"currency\":\"USD\",\"balance_micro\":5}\n",
                "{\"tenant\":\"zero\",\"currency\":\"USD\",\"balance_micro\":0}\n",
            ),
        );
        let budgets = TenantBudgets::open(&budget_path, &request_path).unwrap();
        assert_eq!(
            budgets.balance("tenant-a").unwrap().unwrap().balance_micro,
            15,
        );
        assert_eq!(
            budgets.admit("zero", 1).unwrap_err(),
            BudgetAdmissionError::Insufficient,
        );
        assert!(budgets.admit("unlisted", 1).unwrap().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn utc_day_bucket_conversion_is_stable() {
        assert_eq!(civil_day(0), "1970-01-01");
        assert_eq!(civil_day(20_678), "2026-08-13");
    }

    #[test]
    fn tenant_usage_aggregates_exact_cost_and_debits_by_utc_day() {
        let dir = test_dir("tenant-usage");
        let request_path = dir.join("requests.jsonl");
        let rows = [
            json!({
                "tenant": "tenant-a",
                "finished_unix_ms": 20_678_u64 * 86_400_000,
                "usage": {
                    "prompt_tokens": 10,
                    "cached_prompt_tokens": 4,
                    "ordinary_prompt_tokens": 6,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "cost_usd": {"total": "0.00000310"},
                "budget": {"debit_micro": 4},
            }),
            json!({
                "tenant": "tenant-a",
                "finished_unix_ms": 20_678_u64 * 86_400_000 + 1,
                "usage": {
                    "prompt_tokens": 5,
                    "cached_prompt_tokens": 1,
                    "ordinary_prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
                "cost_usd": {"total": "0.0000012"},
                "budget": null,
            }),
            json!({
                "tenant": "tenant-b",
                "finished_unix_ms": 20_678_u64 * 86_400_000,
                "usage": {
                    "prompt_tokens": 99,
                    "cached_prompt_tokens": 0,
                    "ordinary_prompt_tokens": 99,
                    "completion_tokens": 0,
                    "total_tokens": 99,
                },
                "cost_usd": {"total": "1"},
                "budget": {"debit_micro": 1_000_000},
            }),
        ];
        let contents = rows
            .iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        write_private(&request_path, &contents);
        let ledger = Ledger::open(&request_path, HashMap::new()).unwrap();
        let usage = serde_json::to_value(ledger.tenant_usage("tenant-a").unwrap()).unwrap();
        assert_eq!(usage["tenant"], "tenant-a");
        assert_eq!(usage["days"].as_array().unwrap().len(), 1);
        let day = &usage["days"][0];
        assert_eq!(day["day"], "2026-08-13");
        assert_eq!(day["requests"], 2);
        assert_eq!(day["prompt_tokens"], 15);
        assert_eq!(day["cached_prompt_tokens"], 5);
        assert_eq!(day["completion_tokens"], 3);
        assert_eq!(day["total_tokens"], 18);
        assert_eq!(day["cost_usd"], "0.00000430");
        assert_eq!(day["debited_micro"], 4);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cached_tokens_cannot_exceed_prompt_tokens() {
        let err = schedule()
            .cost(Usage {
                prompt_tokens: 9,
                cached_prompt_tokens: 10,
                completion_tokens: 1,
            })
            .unwrap_err();
        assert!(err.contains("exceed total prompt tokens"));
    }

    #[test]
    fn completed_and_rejected_rows_are_durable_jsonl() {
        let dir = std::env::temp_dir().join(format!("memra-ledger-{}", super::unix_ms()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("requests.jsonl");
        let ledger = Ledger::open(&path, HashMap::from([("q27".into(), schedule())])).unwrap();
        let mut completed = ledger.start(
            "chatcmpl-a",
            "tenant-a",
            "q27",
            "/v1/chat/completions",
            "interactive",
            true,
        );
        completed
            .complete(
                Usage {
                    prompt_tokens: 100,
                    cached_prompt_tokens: 90,
                    completion_tokens: 5,
                },
                0.25,
            )
            .unwrap();
        assert!(
            completed
                .complete(
                    Usage {
                        prompt_tokens: 100,
                        cached_prompt_tokens: 90,
                        completion_tokens: 5,
                    },
                    0.25,
                )
                .unwrap_err()
                .contains("already has a terminal ledger row")
        );
        let mut rejected = ledger.start(
            "chatcmpl-b",
            "tenant-a",
            "q27",
            "/v1/chat/completions",
            "interactive",
            false,
        );
        rejected.reject(429, "rate_limit_exceeded").unwrap();
        drop((completed, rejected, ledger));

        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["format"], FORMAT);
        assert_eq!(rows[0]["cost_usd"]["total"], "0.0000174910");
        assert_eq!(rows[1]["http_status"], 429);
        assert!(rows[1]["cost_usd"].is_null());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn abandoned_row_carries_partial_usage_and_cost() {
        // Drop-classified outcome: serialized against the drain-kill latch test.
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!(
            "memra-ledger-abandoned-{}-{}",
            std::process::id(),
            super::unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("requests.jsonl");
        let ledger = Ledger::open(&path, HashMap::from([("q27".into(), schedule())])).unwrap();
        {
            let mut receipt = ledger.start(
                "chatcmpl-partial",
                "tenant-a",
                "q27",
                "/v1/chat/completions",
                "interactive",
                true,
            );
            receipt.record_prompt_usage(100, 90).unwrap();
            receipt.record_completion_token().unwrap();
            receipt.record_completion_token().unwrap();
        }
        drop(ledger);

        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
        assert_eq!(row["outcome"], "abandoned");
        assert_eq!(row["http_status"], 499);
        assert_eq!(row["usage"]["prompt_tokens"], 100);
        assert_eq!(row["usage"]["cached_prompt_tokens"], 90);
        assert_eq!(row["usage"]["ordinary_prompt_tokens"], 10);
        assert_eq!(row["usage"]["completion_tokens"], 2);
        assert_eq!(row["usage"]["total_tokens"], 102);
        assert_eq!(row["cost_usd"]["ordinary_prompt"], "0.000002890");
        assert_eq!(row["cost_usd"]["cached_prompt"], "0.0000026010");
        assert_eq!(row["cost_usd"]["completion"], "0.0000048");
        assert_eq!(row["cost_usd"]["total"], "0.0000102910");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cancelled_fanout_sibling_is_billed_at_the_cached_rate_after_the_credit_re_emit() {
        // Drop-classified outcome: serialized against the drain-kill latch test.
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // H7 (research/cacheinval-20260813/). A prefix-fanout sibling is admitted with a
        // PROVISIONAL MISS (n_cached = 0) because the group prefix has not been credited yet; the
        // worker credits it afterwards (`s.n_cached += group.prefix_len`) and now re-emits
        // Event::PromptUsage. If a sibling is cancelled before any terminal event,
        // PendingReceipt::drop prices it from the last recorded value, so the re-emit is what
        // decides whether its reused prefix bills as cached or as ordinary input.
        let dir = test_dir("fanout-cancel");
        let path = dir.join("requests.jsonl");
        let ledger = Ledger::open(&path, HashMap::from([("q27".into(), schedule())])).unwrap();
        {
            let mut receipt = ledger.start(
                "chatcmpl-fanout-cancel",
                "tenant-a",
                "q27",
                "/v1/chat/completions",
                "interactive",
                true,
            );
            // admission: provisional miss, prefix not yet credited
            receipt.record_prompt_usage(100, 0).unwrap();
            // fanout restore credits the 90-token group prefix -> the re-emit under test
            receipt.record_prompt_usage(100, 90).unwrap();
            // client cancels: no terminal event, Drop prices the row
        }
        drop(ledger);

        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
        assert_eq!(row["outcome"], "abandoned");
        assert_eq!(row["usage"]["prompt_tokens"], 100);
        // The whole point: the reused prefix is cached, not ordinary input.
        assert_eq!(row["usage"]["cached_prompt_tokens"], 90);
        assert_eq!(row["usage"]["ordinary_prompt_tokens"], 10);
        assert_eq!(row["cost_usd"]["ordinary_prompt"], "0.000002890");
        assert_eq!(row["cost_usd"]["cached_prompt"], "0.0000026010");
        // Total prompt cost 0.0000054910. WITHOUT the re-emit the stale (100, 0) would price all
        // 100 tokens as ordinary input at 0.0000289000 — a 5.26x overcharge on the prompt, in the
        // customer's disfavour. This assertion fails if the re-emit is removed.
        assert_ne!(row["cost_usd"]["ordinary_prompt"], "0.0000289000");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn record_prompt_usage_sets_rather_than_accumulates_so_re_emit_is_idempotent() {
        // The fanout re-emit is only safe because this SETS. If it ever accumulates, a sibling that
        // reaches a terminal event would be double-counted, so pin the semantics here.
        // Drop-classified outcome: serialized against the drain-kill latch test.
        let _serial = DROP_CLASS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("usage-idempotent");
        let path = dir.join("requests.jsonl");
        let ledger = Ledger::open(&path, HashMap::from([("q27".into(), schedule())])).unwrap();
        {
            let mut receipt = ledger.start(
                "chatcmpl-idem",
                "tenant-a",
                "q27",
                "/v1/chat/completions",
                "interactive",
                true,
            );
            receipt.record_prompt_usage(100, 90).unwrap();
            receipt.record_prompt_usage(100, 90).unwrap();
            receipt.record_prompt_usage(100, 90).unwrap();
        }
        drop(ledger);

        let row: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
        assert_eq!(row["usage"]["prompt_tokens"], 100);
        assert_eq!(row["usage"]["cached_prompt_tokens"], 90);
        assert_eq!(row["usage"]["ordinary_prompt_tokens"], 10);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn ledger_open_accepts_0640_class_and_refuses_unsafe_modes() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = std::env::temp_dir().join(format!(
            "memra-ledger-modes-{}-{}",
            std::process::id(),
            super::unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        for mode in [0o600, 0o640] {
            let path = dir.join(format!("accepted-{mode:04o}.jsonl"));
            std::fs::write(&path, b"").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            drop(Ledger::open(&path, HashMap::from([("q27".into(), schedule())])).unwrap());
        }
        for mode in [0o660, 0o644, 0o610] {
            let path = dir.join(format!("refused-{mode:04o}.jsonl"));
            std::fs::write(&path, b"").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            let err = Ledger::open(&path, HashMap::from([("q27".into(), schedule())]))
                .err()
                .expect("unsafe ledger mode must be refused");
            assert!(err.contains(&format!("found {mode:04o}")), "{err}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn ledger_open_does_not_follow_final_symlink() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let dir = std::env::temp_dir().join(format!(
            "memra-ledger-symlink-{}-{}",
            std::process::id(),
            super::unix_ms(),
        ));
        std::fs::create_dir(&dir).unwrap();
        let target = dir.join("target.jsonl");
        let link = dir.join("requests.jsonl");
        std::fs::write(&target, b"").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();
        symlink(&target, &link).unwrap();
        assert!(Ledger::open(&link, HashMap::from([("q27".into(), schedule())]),).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cached_price_must_be_strictly_lower_than_prompt_price() {
        let metadata = OpenRouterModelMetadata {
            pricing: super::super::OpenRouterPricing {
                prompt: Some("0.1".into()),
                cached_prompt: Some("0.10".into()),
                completion: Some("0.2".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let err = PriceSchedule::from_metadata("m", &metadata).unwrap_err();
        assert!(err.contains("must be lower"));
    }
}
