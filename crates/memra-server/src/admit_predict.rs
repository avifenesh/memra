//! Predictive-admission SHADOW instrumentation (lane/d2-engine-gaps-20260831).
//!
//! Engine half of darklanes Arc D2 (`research/engines-kv-oversubscription-20260830/
//! SPEC-SESSION-TIERING.md`, shadow analysis `research/d2-shadow-20260831/RESULTS.md`):
//! the D2 lane measured a Mooncake-style predictive admission predictor offline and
//! named the engine state an enforcing (or live-shadow) build needs. This module
//! supplies that state and the shadow verdict; it NEVER rejects anything. Enforcement
//! is a separate, owner-ratified flip that this module only prepares receipts for.
//!
//! What lives here, by D2 gap number:
//!
//! - G2: [`AdmissionBook`]: per-model in-flight count + booked KV bytes, maintained by
//!   the worker at its single admit/retire seam and published to `/metrics` (operator
//!   scope) as `admission_inflight` / `admission_booked_bytes`. The booked figure is the
//!   engine's OWN admission charge (the request-cost estimate the VRAM gate used,
//!   i.e. the P-maxtok book the ladder implicitly reserves today).
//! - G3: [`CompletionHistory`]: rolling per-(tenant-row, model) completion lengths in
//!   fixed-size rings, with a global per-model fallback ring. Bounded memory by
//!   construction (`HISTORY_WINDOW` samples per ring, `HISTORY_ROW_CAP` tenant rows;
//!   overflow tenants simply use the global fallback). DELIBERATELY VOLATILE: the
//!   history survives nothing across restarts; it is admission-shaping state, not a
//!   ledger, and rebuilding it from live traffic after a boot is the design (the D2
//!   predictor's own 20-sample floor covers the cold window by falling back).
//! - G5: [`ShadowConfig`] + [`shadow_verdict_line`]: the `[admit-predict]` receipt,
//!   one line per request at its first decisive admission consideration, behind
//!   `MEMRA_ADMIT_PREDICT_SHADOW` (default 0 = OFF). Computes the D2 contract's
//!   P-tenant-p95 verdict (slot check first, then the KV book against the budget
//!   arm) and LOGS it, joined to the request id. The budget arm is
//!   `MEMRA_ADMIT_PREDICT_BUDGET_MB` when set (explicit operator override), else
//!   DERIVED AT BOOT by the worker from the engine's own numbers
//!   ([`DerivedShadowBudget`]): the stress-campaign-20260901 FINDING 3 root cause 1
//!   was a hand-derived env value that went stale the moment another lane's deploy
//!   changed the box config (box12 c32: 164 reject-kv against a pre-devpenalty
//!   61,580 MiB arm on traffic that served 200). A config-changing redeploy now
//!   re-derives automatically.
//! - Dual book (FINDING 3 root cause 2, booked-vs-real calibration): every receipt
//!   line carries BOTH `booked_bytes` (the predictive shadow book: each in-flight
//!   request's full kv_hat, prompt + PREDICTED completion, charged up front and held
//!   until retire) and `booked_real` (the engine-charge book: the request-cost
//!   estimates the real VRAM gate actually used at each admit). The predictive book
//!   overbooks by construction at saturation, it holds every in-flight request's
//!   completion KV as if already grown, while the real gate re-reads live free VRAM
//!   (ground truth, only growth-to-date) per arrival. Logging both, instead of
//!   silently re-tuning, lets the D2 false-reject analysis read directly which arm
//!   lied on each would-reject row.
//! - G6: the would-reject lines carry `retry_after_s=`, the value an enforcing 429
//!   would send, computed by [`earliest_completion_retry_s`] from the in-flight
//!   sessions' own predicted remainders and clamped to the shed contract's 1..=60 s
//!   window (`EngineError::rate_limit_after` + `retry_contract_response` are the
//!   byte-compatible response half, wired and tested but not called in shadow mode).
//! - G7: [`ShadowConfig::is_exempt`]: `MEMRA_ADMIT_PREDICT_EXEMPT_TENANTS`, the engine-side
//!   copy of the console's `INTERNAL_ENGINE_TENANTS` list (darklanes
//!   `workers/console/src/activity.ts`); that list is the source of truth and this env
//!   is set FROM it by the deployment. Exempt tenants still enter the book as load;
//!   their would-reject rows are marked `exempt=1` so the confusion matrix can report
//!   them separately as instrument noise (a 429'd probe reads as an outage).
//!
//! Admission-path budget: everything per-request here is arithmetic over fixed-size
//! state. The single percentile computation copies at most `HISTORY_WINDOW` u32 values
//! into a reusable scratch buffer (no steady-state allocation) and runs
//! `select_nth_unstable`, and it runs only while the shadow flag is armed.
//!
//! Scope boundary, stated: this instruments the PRODUCTION admission seam (the batched
//! worker loop). The experimental dsv4 door serves bs=1 on its own dedicated thread and
//! diverts in `handle_cmd` BEFORE that loop, so dsv4 requests appear in neither the
//! book nor the receipts; that door is not a serving-grade path (docs/FLAGS.md §5).

use std::collections::HashMap;

/// Samples kept per completion ring (per tenant-row x model, and per global model row).
/// The D2 report's G3 sizing ("a rolling reservoir (fixed memory, e.g. last 512
/// completions per tenant key)").
pub(crate) const HISTORY_WINDOW: usize = 512;

/// Bound on distinct (tenant-row, model) history rows. Past the cap a new tenant's
/// completions still feed the global per-model ring, and its predictions fall back to
/// P-global-p95: bounded memory beats per-tenant fidelity for row #257.
pub(crate) const HISTORY_ROW_CAP: usize = 256;

/// Tenant-p95 needs at least this much tenant history before it is trusted
/// (D2 contract §5: "floor 20 samples, else fall back to P-global-p95").
pub(crate) const HISTORY_TENANT_FLOOR: usize = 20;

/// The `+8` context slack the engine's own admission ladder adds
/// (`prompt_tokens + max_tokens + 8`); the D2 contract's `ctx_hat` mirrors it.
pub(crate) const CTX_HAT_SLACK: u64 = 8;

// ---------------------------------------------------------------------------
// G5/G6/G7: shadow configuration
// ---------------------------------------------------------------------------

/// Read-once configuration for the shadow predictor. Built at worker start;
/// the flag cannot arm or disarm mid-process.
pub(crate) struct ShadowConfig {
    /// `MEMRA_ADMIT_PREDICT_SHADOW` != "0"/unset. OFF by design: shadow logging is
    /// a measurement act, and unmeasured behavior does not default ON.
    pub armed: bool,
    /// The box KV budget arm the KV check compares against, in bytes.
    /// `MEMRA_ADMIT_PREDICT_BUDGET_MB` when set: the explicit operator override.
    /// When unset, the worker fills this in at boot via [`Self::resolve_budget`]
    /// from the engine's own numbers (effective free VRAM at boot minus the device
    /// prefix-cache budget minus the admission reserve): the values the VRAM gate
    /// and prefix-cache init already read, so a config-changing redeploy re-derives
    /// automatically instead of serving a stale hand-carried constant
    /// (stress-campaign-20260901 FINDING 3 cause 1). Only when BOTH the env is
    /// unset AND the boot derivation is unavailable (free-VRAM query failed) does
    /// the KV arm stay unarmed (`verdict` can only be admit/reject-slot, the line
    /// says `budget_bytes=unset`) so offline analysis can apply any arm to the
    /// logged books.
    pub budget_bytes: Option<u64>,
    /// `MEMRA_ADMIT_PREDICT_EXEMPT_TENANTS`: comma-separated tenant ids, sourced from the
    /// console's `INTERNAL_ENGINE_TENANTS`. Matching is on the tenant ROW identity
    /// (`auth::meter_key` with any `t:` prefix stripped).
    exempt: Vec<String>,
}

impl ShadowConfig {
    pub(crate) fn from_env() -> Self {
        let armed = std::env::var("MEMRA_ADMIT_PREDICT_SHADOW").is_ok_and(|v| v != "0");
        let budget_bytes = std::env::var("MEMRA_ADMIT_PREDICT_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb.saturating_mul(1 << 20));
        let exempt = std::env::var("MEMRA_ADMIT_PREDICT_EXEMPT_TENANTS")
            .ok()
            .map(|v| Self::parse_exempt(&v))
            .unwrap_or_default();
        ShadowConfig {
            armed,
            budget_bytes,
            exempt,
        }
    }

    /// Resolve the KV budget arm at boot and emit the `[admit-predict] shadow armed:`
    /// boot line (the receipt FLAGS.md points at). Call once, after boot calibration,
    /// so the derivation sees the calibrated admission reserve. Precedence:
    /// the env override wins outright; else the boot derivation; else the arm stays
    /// unarmed with the reason stated. The boot line states the resolved value AND
    /// its formula inputs, so a receipt reader can re-check the arithmetic against
    /// the box without shell access.
    pub(crate) fn resolve_budget(&mut self, derived: Option<DerivedShadowBudget>) {
        if !self.armed {
            return;
        }
        let source = if self.budget_bytes.is_some() {
            "env-override(MEMRA_ADMIT_PREDICT_BUDGET_MB)".to_string()
        } else {
            match derived {
                Some(d) => {
                    self.budget_bytes = Some(d.budget_bytes());
                    format!(
                        "derived(effective_free_bytes={} - prefix_cache_budget_bytes={} \
                         - admission_reserve_bytes={})",
                        d.effective_free_bytes,
                        d.prefix_cache_budget_bytes,
                        d.admission_reserve_bytes,
                    )
                }
                None => "unset(boot free-VRAM query failed; KV arm stays unarmed)".to_string(),
            }
        };
        eprintln!(
            "[admit-predict] shadow armed: budget_bytes={} budget_src={} exempt_tenants={} \
             (logging only, nothing is rejected; enforcement is a separate flip)",
            self.budget_bytes.map_or("unset".into(), |b| b.to_string()),
            source,
            self.exempt.len(),
        );
    }

    fn parse_exempt(raw: &str) -> Vec<String> {
        raw.split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// G7: whether this tenant row is enforcement-exempt. `tenant_row` is
    /// `auth::meter_key(cache_ns)`: `t:<tenant>` under a keyring, the raw salt
    /// otherwise; the console list carries bare tenant ids, so the `t:` prefix is
    /// stripped before comparison. Exempt tenants still occupy the book as load.
    pub(crate) fn is_exempt(&self, tenant_row: &str) -> bool {
        let bare = tenant_row.strip_prefix("t:").unwrap_or(tenant_row);
        self.exempt.iter().any(|t| t == bare)
    }
}

/// Boot inputs for the derived shadow budget (unset `MEMRA_ADMIT_PREDICT_BUDGET_MB`).
/// All three are the engine's OWN boot numbers, read at the same seams the real
/// admission machinery reads them, so they cannot go stale against the box the way
/// a hand-derived deployment constant did (stress-campaign-20260901 FINDING 3
/// cause 1: pre-devpenalty 61,580 MiB arm survived a config-changing redeploy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DerivedShadowBudget {
    /// Effective free VRAM at boot: driver free plus pool-cached, the same
    /// quantity the VRAM admission gate reads (`effective_free_bytes`).
    pub effective_free_bytes: u64,
    /// The device prefix-cache byte budget (configured or derived at
    /// `init_prefix_cache_budget`): bytes the cache may hold that session KV
    /// therefore cannot count on.
    pub prefix_cache_budget_bytes: u64,
    /// The admission transient reserve the real gate charges on top of a request's
    /// cost (calibrated per-deployment floor, never below the static constant;
    /// the `MEMRA_ADMIT_RESERVE_MB` teeth door wins where set).
    pub admission_reserve_bytes: u64,
}

impl DerivedShadowBudget {
    /// `budget = effective_free - prefix_cache_budget - admission_reserve`,
    /// saturating: the bytes session KV can actually claim once the prefix cache
    /// and the transient floor take their shares.
    pub(crate) fn budget_bytes(&self) -> u64 {
        self.effective_free_bytes
            .saturating_sub(self.prefix_cache_budget_bytes)
            .saturating_sub(self.admission_reserve_bytes)
    }
}

// ---------------------------------------------------------------------------
// G2: the in-flight book
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Copy)]
struct ModelBook {
    inflight: u64,
    /// Sum of in-flight sessions' engine admission charges (the request-cost
    /// estimate the VRAM gate used at each session's admit).
    booked_bytes: u64,
    /// Sum of in-flight sessions' SHADOW kv_hat charges (P-tenant-p95 book).
    /// Zero for every session admitted while the shadow flag is off.
    shadow_booked_bytes: u64,
}

/// Per-model in-flight count + booked KV bytes, owned by the worker loop (single
/// thread, no locks) and snapshotted into `Metrics` on the publish tick. One admit
/// and one retire seam keep it exact: `active.push` books, `active.remove` unbooks
/// (a step-OOM park unbooks too, since its KV is dropped, and its replay re-books at
/// re-admission).
#[derive(Default)]
pub(crate) struct AdmissionBook {
    models: HashMap<String, ModelBook>,
}

impl AdmissionBook {
    pub(crate) fn admit(&mut self, model: &str, booked_bytes: u64, shadow_kv_hat: u64) {
        let row = self.models.entry(model.to_string()).or_default();
        row.inflight += 1;
        row.booked_bytes = row.booked_bytes.saturating_add(booked_bytes);
        row.shadow_booked_bytes = row.shadow_booked_bytes.saturating_add(shadow_kv_hat);
    }

    pub(crate) fn retire(&mut self, model: &str, booked_bytes: u64, shadow_kv_hat: u64) {
        if let Some(row) = self.models.get_mut(model) {
            row.inflight = row.inflight.saturating_sub(1);
            row.booked_bytes = row.booked_bytes.saturating_sub(booked_bytes);
            row.shadow_booked_bytes = row.shadow_booked_bytes.saturating_sub(shadow_kv_hat);
        }
    }

    pub(crate) fn inflight(&self, model: &str) -> u64 {
        self.models.get(model).map_or(0, |row| row.inflight)
    }

    /// The shadow KV book the D2 decision rule sums: EVERY in-flight request's own
    /// admission-time kv_hat, across all models (they share the card's budget arm).
    pub(crate) fn shadow_booked_total(&self) -> u64 {
        self.models
            .values()
            .map(|row| row.shadow_booked_bytes)
            .sum()
    }

    /// The ENGINE-charge book, same summation shape: every in-flight request's real
    /// admission charge (the cost the VRAM gate used at its admit). The `booked_real=`
    /// column of the receipt: logged beside the predictive book so the false-reject
    /// analysis reads which arm lied (FINDING 3 calibration, booked-vs-real).
    pub(crate) fn booked_total(&self) -> u64 {
        self.models.values().map(|row| row.booked_bytes).sum()
    }

    /// G2 metrics snapshots (operator scope): per-model in-flight and booked bytes.
    pub(crate) fn inflight_snapshot(&self) -> HashMap<String, u64> {
        self.models
            .iter()
            .map(|(model, row)| (model.clone(), row.inflight))
            .collect()
    }

    pub(crate) fn booked_snapshot(&self) -> HashMap<String, u64> {
        self.models
            .iter()
            .map(|(model, row)| (model.clone(), row.booked_bytes))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// G3: rolling completion-length history
// ---------------------------------------------------------------------------

struct Ring {
    buf: Vec<u32>,
    next: usize,
    len: usize,
}

impl Ring {
    fn new() -> Self {
        Ring {
            // Allocated at row creation (once per tenant-row x model), never per request.
            buf: vec![0; HISTORY_WINDOW],
            next: 0,
            len: 0,
        }
    }

    fn push(&mut self, sample: u32) {
        self.buf[self.next] = sample;
        self.next = (self.next + 1) % HISTORY_WINDOW;
        self.len = (self.len + 1).min(HISTORY_WINDOW);
    }
}

/// Rolling per-(tenant-row, model) completion-length state (D2 gap G3). Fed at session
/// retirement with the session's generated token count; consulted (only while the
/// shadow flag is armed) for the causal tenant p95. Restart-volatile BY DESIGN; see
/// the module doc.
#[derive(Default)]
pub(crate) struct CompletionHistory {
    rows: HashMap<(String, String), Ring>,
    global: HashMap<String, Ring>,
    /// Reusable percentile scratch: percentile() copies a ring here and selects in
    /// place, so the admission path allocates nothing in steady state.
    scratch: Vec<u32>,
}

/// Which estimator arm produced the predicted completion length: the `reason=` field
/// of the receipt line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LhatSource {
    /// Per-(tenant, model) causal p95, >= HISTORY_TENANT_FLOOR samples.
    TenantP95,
    /// Global per-model causal p95 (tenant history below the floor).
    GlobalP95,
    /// No usable history at all: fall back to the request's own completion bound
    /// (P-maxtok, today's implicit reservation), the conservative cold-start arm.
    MaxtokFallback,
}

impl LhatSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            LhatSource::TenantP95 => "tenant-p95",
            LhatSource::GlobalP95 => "global-p95",
            LhatSource::MaxtokFallback => "maxtok-fallback",
        }
    }
}

impl CompletionHistory {
    /// Record a retired session's completion length. Called once per retirement
    /// (never per token). Only terminal completions feed the history: the caller
    /// excludes step-OOM parks (they re-admit; nothing completed) and client aborts
    /// (a disconnect length is not a completion length; the D2 offline history was
    /// built from completed rows only).
    pub(crate) fn record(&mut self, tenant_row: &str, model: &str, completion_tokens: u32) {
        self.global
            .entry(model.to_string())
            .or_insert_with(Ring::new)
            .push(completion_tokens);
        let key = (tenant_row.to_string(), model.to_string());
        if let Some(ring) = self.rows.get_mut(&key) {
            ring.push(completion_tokens);
            return;
        }
        if self.rows.len() < HISTORY_ROW_CAP {
            self.rows
                .entry(key)
                .or_insert_with(Ring::new)
                .push(completion_tokens);
        }
        // At the row cap: the sample lives in the global ring only, and this tenant
        // predicts through the global fallback. Bounded memory is the contract.
    }

    /// Nearest-rank p95 of a ring (rank = ceil(0.95 * n), 1-based), via
    /// select_nth_unstable on the reusable scratch buffer. O(n), no allocation in
    /// steady state.
    fn p95(scratch: &mut Vec<u32>, ring: &Ring) -> u32 {
        scratch.clear();
        scratch.extend_from_slice(&ring.buf[..ring.len]);
        let n = scratch.len();
        debug_assert!(n > 0, "p95 caller checks emptiness");
        let rank = ((n as f64) * 0.95).ceil() as usize; // 1-based nearest rank
        let idx = rank.clamp(1, n) - 1;
        let (_, value, _) = scratch.select_nth_unstable(idx);
        *value
    }

    /// The D2 P-tenant-p95 estimator: tenant p95 with >= HISTORY_TENANT_FLOOR samples,
    /// else global per-model p95, else the request's own completion bound; always
    /// clipped to that bound when one exists.
    pub(crate) fn lhat(
        &mut self,
        tenant_row: &str,
        model: &str,
        max_tokens: Option<u64>,
    ) -> (u64, LhatSource) {
        let key = (tenant_row.to_string(), model.to_string());
        let predicted = match self.rows.get(&key) {
            Some(ring) if ring.len >= HISTORY_TENANT_FLOOR => Some((
                Self::p95(&mut self.scratch, ring) as u64,
                LhatSource::TenantP95,
            )),
            _ => match self.global.get(model) {
                Some(ring) if ring.len > 0 => Some((
                    Self::p95(&mut self.scratch, ring) as u64,
                    LhatSource::GlobalP95,
                )),
                _ => None,
            },
        };
        match (predicted, max_tokens) {
            (Some((p95, source)), Some(bound)) => (p95.min(bound), source),
            (Some((p95, source)), None) => (p95, source),
            // Cold start with a bound: book the bound (P-maxtok, the engine's own
            // implicit reservation today). Cold start without one: nothing honest to
            // book; 0 keeps the line explicit (reason=maxtok-fallback, predicted 0).
            (None, bound) => (bound.unwrap_or(0), LhatSource::MaxtokFallback),
        }
    }
}

// ---------------------------------------------------------------------------
// G5: the verdict
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    Admit,
    /// In-flight on this model already holds the session cap (D2:
    /// `would-reject(no-decode-slot-at-completion)`).
    RejectSlot,
    /// Predicted KV book (in-flight kv_hats + this request's kv_hat) exceeds the
    /// configured budget arm (D2: `would-reject(kv-budget-exceeded)`).
    RejectKv,
}

impl Verdict {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Verdict::Admit => "admit",
            Verdict::RejectSlot => "reject-slot",
            Verdict::RejectKv => "reject-kv",
        }
    }
}

/// The D2 contract's per-request KV charge:
/// `kv_hat = B/token x (prompt + L_hat + slack) + fixed`.
pub(crate) fn kv_hat(
    prompt_tokens: u64,
    predicted_completion: u64,
    bytes_per_token: u64,
    fixed_bytes: u64,
) -> u64 {
    bytes_per_token
        .saturating_mul(
            prompt_tokens
                .saturating_add(predicted_completion)
                .saturating_add(CTX_HAT_SLACK),
        )
        .saturating_add(fixed_bytes)
}

/// G6 (shadow half): the Retry-After an enforcing reject would send: the earliest
/// predicted in-flight completion, from each session's own shadow book (predicted
/// total minus generated so far) at the current p50 step latency. Clamped to the shed
/// contract's 1..=60 s window (`retry_contract_response` applies the same clamp on the
/// wire, so the logged value IS the header value). None when nothing is in flight or
/// no latency estimate exists yet (an enforcing path would fall back to the class
/// default (RETRY_AFTER_S_RATE_LIMIT) there, and the line logs `retry_after_s=-`.
pub(crate) fn earliest_completion_retry_s(
    inflight: impl Iterator<Item = (u64, u64)>, // (shadow predicted total, generated so far)
    p50_step_ms: f32,
) -> Option<u64> {
    if p50_step_ms <= 0.0 {
        return None;
    }
    inflight
        .map(|(predicted_total, generated)| predicted_total.saturating_sub(generated))
        .min()
        .map(|remaining_tokens| {
            let secs = (remaining_tokens as f64 * p50_step_ms as f64 / 1000.0).ceil() as u64;
            secs.clamp(1, 60)
        })
}

/// Margin for the first-token deadline gate (`MEMRA_FIRST_TOKEN_DEADLINE_GATE`): the same
/// 150% posture as the non-streaming feasibility gate's `DEADLINE_INFEASIBLE_MARGIN_PCT`:
/// the floor rates are deliberately pessimistic, so only an estimate 1.5x past the
/// remaining deadline refuses. A shared constant is NOT reused across the two gates on
/// purpose: they judge different phases (whole response vs first token) and must be
/// re-tunable independently when their GPU gates land.
pub(crate) const FIRST_TOKEN_DEADLINE_MARGIN_PCT: u64 = 150;

/// First-token deadline feasibility (lane/bench-debts-20260901, competitive-bench engine
/// debt 3; the streaming analog the D2 report §7 named): can this request's FIRST token
/// arrive inside its remaining wire deadline, given the prime backlog already admitted
/// ahead of it? The estimate is the whole pending prefill demand (this prompt plus the
/// sum of every active session's unprimed tokens) at the pessimistic single-stream
/// prefill floor (`MEMRA_PREFILL_FLOOR_TOK_S`; the batched prime shares one card, so the
/// aggregate rate is the same unit). Streaming and non-streaming alike: the wire deadline
/// bounds exactly time-to-first-token for a stream and is a lower bound for a body.
/// Returns `Some(estimated_ms)` when infeasible past the margin, `None` when the request
/// may proceed. Pure arithmetic; the worker owns the inputs and the refusal.
pub(crate) fn first_token_wait_infeasible(
    prompt_tokens: u64,
    backlog_tokens: u64,
    remaining_ms: u64,
    prefill_floor_tok_s: u64,
) -> Option<u64> {
    let est_ms = prompt_tokens
        .saturating_add(backlog_tokens)
        .saturating_mul(1_000)
        / prefill_floor_tok_s.max(1);
    let bound_ms = remaining_ms.saturating_mul(FIRST_TOKEN_DEADLINE_MARGIN_PCT) / 100;
    (est_ms > bound_ms).then_some(est_ms)
}

/// Everything one receipt line says. Assembled by the worker at the request's first
/// decisive admission consideration; formatted by [`shadow_verdict_line`].
pub(crate) struct VerdictLine<'a> {
    pub request_id: &'a str,
    pub tenant_row: &'a str,
    pub model: &'a str,
    pub verdict: Verdict,
    /// Which estimator arm produced `predicted_completion` (the decomposed "why":
    /// the verdict already carries the reject reason per the D2 contract).
    pub reason: LhatSource,
    pub prompt_tokens: Option<u64>,
    pub predicted_completion: u64,
    /// This request's own kv_hat charge (None on slot rejects logged before tokenize).
    pub kv_hat_bytes: Option<u64>,
    /// The in-flight shadow book at decision time, EXCLUDING this request:
    /// the sum of predictive kv_hat charges (prompt + predicted completion,
    /// booked in full at admit).
    pub booked_bytes: u64,
    /// The in-flight ENGINE-charge book at the same instant ([`AdmissionBook::
    /// booked_total`]): the request-cost estimates the real VRAM gate used at each
    /// admit. Carried beside `booked_bytes` so the D2 bar analysis sees both books
    /// on every row and the false-reject read is direct, never a re-tuned guess
    /// (FINDING 3: kv_hat booked 63.5-64 GB at c32 while reality served 200).
    pub booked_real_bytes: u64,
    pub inflight: u64,
    pub cap: u64,
    pub budget_bytes: Option<u64>,
    /// What an enforcing 429 would have sent (would-reject rows only).
    pub retry_after_s: Option<u64>,
    pub exempt: bool,
}

/// One grep-stable receipt line, `[admit-predict]`-prefixed, all fields `key=value`.
/// The confusion matrix is a grep, not a reconstruction (D2 G5).
pub(crate) fn shadow_verdict_line(line: &VerdictLine<'_>) -> String {
    format!(
        "[admit-predict] id={} tenant={:?} model={:?} verdict={} reason={} prompt={} \
         predicted_completion={} kv_hat={} booked_bytes={} booked_real={} inflight={} \
         cap={} budget_bytes={} retry_after_s={} exempt={}",
        line.request_id,
        line.tenant_row,
        line.model,
        line.verdict.as_str(),
        line.reason.as_str(),
        line.prompt_tokens.map_or("-".into(), |v| v.to_string()),
        line.predicted_completion,
        line.kv_hat_bytes.map_or("-".into(), |v| v.to_string()),
        line.booked_bytes,
        line.booked_real_bytes,
        line.inflight,
        line.cap,
        line.budget_bytes.map_or("unset".into(), |v| v.to_string()),
        line.retry_after_s.map_or("-".into(), |v| v.to_string()),
        u8::from(line.exempt),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First-token deadline arithmetic (lane/bench-debts-20260901, competitive-bench
    /// debt 3). The bench shape as concrete rows: at the 2,000 tok/s floor, a 60k-token
    /// prompt behind a 500k-token prime backlog estimates 280 s: infeasible inside a
    /// 90 s wire deadline; the same prompt on an idle box estimates 30 s and proceeds.
    #[test]
    fn first_token_wait_infeasible_arithmetic() {
        // Idle box, 60k prompt, 90 s deadline: 30 s estimate <= 135 s bound -> feasible.
        assert_eq!(first_token_wait_infeasible(60_000, 0, 90_000, 2_000), None);
        // The bench's N56 thrash shape: 500k tokens of prime backlog ahead.
        assert_eq!(
            first_token_wait_infeasible(60_000, 500_000, 90_000, 2_000),
            Some(280_000)
        );
        // The margin is real: an estimate just past the deadline but inside the x1.5
        // band proceeds (floor rates are pessimistic)...
        assert_eq!(first_token_wait_infeasible(200_000, 0, 90_000, 2_000), None);
        // ...and just past the band refuses. 271k tok / 2k tok/s = 135.5 s > 135 s.
        assert_eq!(
            first_token_wait_infeasible(271_000, 0, 90_000, 2_000),
            Some(135_500)
        );
        // A decayed deadline refuses what a fresh one admitted (defer-tick re-check).
        assert_eq!(first_token_wait_infeasible(60_000, 0, 90_000, 2_000), None);
        assert_eq!(
            first_token_wait_infeasible(60_000, 0, 10_000, 2_000),
            Some(30_000)
        );
        // Zero/absurd floor never divides by zero and never admits by overflow
        // (saturating add+mul, floor clamped to 1).
        assert_eq!(
            first_token_wait_infeasible(u64::MAX, u64::MAX, 90_000, 0),
            Some(u64::MAX)
        );
    }

    fn line(verdict: Verdict) -> VerdictLine<'static> {
        VerdictLine {
            request_id: "chatcmpl-abc123",
            tenant_row: "t:acme",
            model: "qwen/qwen3.8-27b",
            verdict,
            reason: LhatSource::TenantP95,
            prompt_tokens: Some(1200),
            predicted_completion: 2906,
            kv_hat_bytes: Some(130_000_000),
            booked_bytes: 9_000_000_000,
            booked_real_bytes: 6_500_000_000,
            inflight: 7,
            cap: 32,
            budget_bytes: Some(35_423 << 20),
            retry_after_s: Some(4),
            exempt: false,
        }
    }

    /// Locks the receipt's field NAMES and joinability: request id, verdict, reason,
    /// predicted_completion, booked_bytes, booked_real (the fields the D2
    /// confusion-matrix grep keys on; the dual book is the FINDING 3 calibration),
    /// plus the shadow-Retry-After and exemption markers.
    #[test]
    fn verdict_line_locks_fields() {
        let s = shadow_verdict_line(&line(Verdict::RejectKv));
        assert!(s.starts_with("[admit-predict] "), "grep-stable prefix: {s}");
        for field in [
            "id=chatcmpl-abc123",
            "tenant=\"t:acme\"",
            "model=\"qwen/qwen3.8-27b\"",
            "verdict=reject-kv",
            "reason=tenant-p95",
            "prompt=1200",
            "predicted_completion=2906",
            "kv_hat=130000000",
            "booked_bytes=9000000000",
            "booked_real=6500000000",
            "inflight=7",
            "cap=32",
            "budget_bytes=37143707648",
            "retry_after_s=4",
            "exempt=0",
        ] {
            assert!(s.contains(field), "line must carry `{field}`: {s}");
        }
        // Exactly one line: a multi-line receipt breaks the grep contract.
        assert_eq!(s.lines().count(), 1);
    }

    #[test]
    fn verdict_line_optional_fields_render_placeholders() {
        let mut l = line(Verdict::RejectSlot);
        l.prompt_tokens = None;
        l.kv_hat_bytes = None;
        l.budget_bytes = None;
        l.retry_after_s = None;
        l.exempt = true;
        let s = shadow_verdict_line(&l);
        for field in [
            "verdict=reject-slot",
            "prompt=-",
            "kv_hat=-",
            "budget_bytes=unset",
            "retry_after_s=-",
            "exempt=1",
        ] {
            assert!(s.contains(field), "line must carry `{field}`: {s}");
        }
    }

    #[test]
    fn book_admit_retire_round_trip() {
        let mut book = AdmissionBook::default();
        book.admit("m", 100, 40);
        book.admit("m", 50, 10);
        book.admit("other", 7, 3);
        assert_eq!(book.inflight("m"), 2);
        assert_eq!(book.booked_snapshot()["m"], 150);
        assert_eq!(book.inflight_snapshot()["m"], 2);
        // Both books sum across models (one card, one budget arm): the predictive
        // kv_hat book and the engine-charge (booked_real) book.
        assert_eq!(book.shadow_booked_total(), 53);
        assert_eq!(book.booked_total(), 157);
        book.retire("m", 100, 40);
        assert_eq!(book.inflight("m"), 1);
        assert_eq!(book.booked_snapshot()["m"], 50);
        assert_eq!(book.shadow_booked_total(), 13);
        assert_eq!(book.booked_total(), 57);
        book.retire("m", 50, 10);
        book.retire("other", 7, 3);
        assert_eq!(book.inflight("m"), 0);
        assert_eq!(book.shadow_booked_total(), 0);
        // Underflow is saturating, never a panic in the serving loop.
        book.retire("m", 1, 1);
        assert_eq!(book.inflight("m"), 0);
    }

    #[test]
    fn history_tenant_floor_then_tenant_p95() {
        let mut h = CompletionHistory::default();
        // Below the 20-sample floor: global fallback.
        for i in 0..HISTORY_TENANT_FLOOR - 1 {
            h.record("t:a", "m", 100 + i as u32);
        }
        h.record("t:b", "m", 9_000); // global ring gains a whale
        let (lhat, source) = h.lhat("t:a", "m", None);
        assert_eq!(source, LhatSource::GlobalP95);
        assert!(lhat >= 118, "global p95 sees the whale: {lhat}");
        // At the floor: tenant p95 (nearest rank ceil(0.95 x 20) = 19 -> the 19th of
        // 20 sorted samples), so the tenant's own single outlier sits above it.
        h.record("t:a", "m", 200);
        let (lhat, source) = h.lhat("t:a", "m", None);
        assert_eq!(source, LhatSource::TenantP95);
        assert_eq!(lhat, 118, "nearest-rank p95 of {{100..=118, 200}}");
    }

    #[test]
    fn history_clips_to_max_tokens_and_cold_start_falls_back() {
        let mut h = CompletionHistory::default();
        for _ in 0..HISTORY_TENANT_FLOOR {
            h.record("t:a", "m", 5_000);
        }
        let (lhat, source) = h.lhat("t:a", "m", Some(256));
        assert_eq!(
            (lhat, source),
            (256, LhatSource::TenantP95),
            "clip to bound"
        );
        // No history at all: the request's own bound (P-maxtok arm), else 0.
        let (lhat, source) = h.lhat("t:zzz", "unknown-model", Some(1024));
        assert_eq!((lhat, source), (1024, LhatSource::MaxtokFallback));
        let (lhat, source) = h.lhat("t:zzz", "unknown-model", None);
        assert_eq!((lhat, source), (0, LhatSource::MaxtokFallback));
    }

    #[test]
    fn history_ring_is_rolling_and_rows_are_bounded() {
        let mut h = CompletionHistory::default();
        // Overfill one ring: old samples must age out.
        for _ in 0..HISTORY_WINDOW {
            h.record("t:a", "m", 10);
        }
        for _ in 0..HISTORY_WINDOW {
            h.record("t:a", "m", 20);
        }
        let (lhat, _) = h.lhat("t:a", "m", None);
        assert_eq!(lhat, 20, "the whole window rolled over");
        // Row cap: row #HISTORY_ROW_CAP+1 gets no per-tenant ring but still feeds global.
        let mut h = CompletionHistory::default();
        for i in 0..HISTORY_ROW_CAP {
            h.record(&format!("t:{i}"), "m", 100);
        }
        assert_eq!(h.rows.len(), HISTORY_ROW_CAP);
        h.record("t:overflow", "m", 100);
        assert_eq!(h.rows.len(), HISTORY_ROW_CAP, "row map is bounded");
        for _ in 0..HISTORY_TENANT_FLOOR + 5 {
            h.record("t:overflow", "m", 300);
        }
        let (_, source) = h.lhat("t:overflow", "m", None);
        assert_eq!(
            source,
            LhatSource::GlobalP95,
            "an over-cap tenant predicts through the global arm"
        );
    }

    #[test]
    fn kv_hat_matches_contract_arithmetic() {
        // kv_hat = B/token x (p + L + 8) + fixed, the D2 §4 decision rule.
        assert_eq!(kv_hat(100, 50, 1_000, 7), 1_000 * (100 + 50 + 8) + 7);
        // Saturates rather than overflowing on absurd inputs.
        assert_eq!(kv_hat(u64::MAX, u64::MAX, 2, 1), u64::MAX);
    }

    #[test]
    fn retry_hint_is_earliest_completion_clamped_to_shed_window() {
        // Two in-flight: 100 remaining and 5 remaining; p50 200 ms/step.
        let hint = earliest_completion_retry_s([(150, 50), (30, 25)].into_iter(), 200.0);
        assert_eq!(hint, Some(1), "5 tokens x 0.2 s = 1 s ceil");
        let hint = earliest_completion_retry_s([(10_000, 0)].into_iter(), 200.0);
        assert_eq!(hint, Some(60), "clamped to the shed contract's 60 s max");
        assert_eq!(earliest_completion_retry_s(std::iter::empty(), 200.0), None);
        assert_eq!(
            earliest_completion_retry_s([(10, 0)].into_iter(), 0.0),
            None,
            "no latency estimate yet"
        );
    }

    /// The derived-budget arithmetic (FINDING 3 cause 1 fix): effective free at boot
    /// minus the prefix-cache budget minus the admission reserve, saturating.
    #[test]
    fn derived_budget_arithmetic() {
        // box12-shaped numbers: 81,920 MiB effective free, 16,384 MiB prefix cache,
        // 1,536 MiB reserve floor -> 64,000 MiB budget.
        let d = DerivedShadowBudget {
            effective_free_bytes: 81_920 << 20,
            prefix_cache_budget_bytes: 16_384 << 20,
            admission_reserve_bytes: 1_536 << 20,
        };
        assert_eq!(d.budget_bytes(), 64_000 << 20);
        // A prefix cache + reserve larger than free saturates to 0 (an unarmed-in-
        // practice budget, never an underflow panic in the serving loop).
        let d = DerivedShadowBudget {
            effective_free_bytes: 1 << 30,
            prefix_cache_budget_bytes: 1 << 30,
            admission_reserve_bytes: 1,
        };
        assert_eq!(d.budget_bytes(), 0);
    }

    /// The env override wins over the derivation; unset env + derivation fills in;
    /// unset env + failed derivation leaves the KV arm unarmed.
    #[test]
    fn resolve_budget_precedence() {
        let derived = DerivedShadowBudget {
            effective_free_bytes: 10 << 30,
            prefix_cache_budget_bytes: 2 << 30,
            admission_reserve_bytes: 1 << 30,
        };
        // Env override present: derivation must not clobber it.
        let mut cfg = ShadowConfig {
            armed: true,
            budget_bytes: Some(1234),
            exempt: Vec::new(),
        };
        cfg.resolve_budget(Some(derived));
        assert_eq!(cfg.budget_bytes, Some(1234));
        // Unset env: the boot derivation arms the KV check.
        let mut cfg = ShadowConfig {
            armed: true,
            budget_bytes: None,
            exempt: Vec::new(),
        };
        cfg.resolve_budget(Some(derived));
        assert_eq!(cfg.budget_bytes, Some(7 << 30));
        // Unset env + no derivation (free-VRAM query failed): stays unarmed.
        let mut cfg = ShadowConfig {
            armed: true,
            budget_bytes: None,
            exempt: Vec::new(),
        };
        cfg.resolve_budget(None);
        assert_eq!(cfg.budget_bytes, None);
        // Disarmed shadow never resolves (no boot line, no budget).
        let mut cfg = ShadowConfig {
            armed: false,
            budget_bytes: None,
            exempt: Vec::new(),
        };
        cfg.resolve_budget(Some(derived));
        assert_eq!(cfg.budget_bytes, None);
    }

    #[test]
    fn exempt_matching_strips_tenant_row_prefix() {
        let cfg = ShadowConfig {
            armed: true,
            budget_bytes: None,
            exempt: ShadowConfig::parse_exempt("watchdog-orn, ten_UBB1F6Lf ,,orn-probe-dry"),
        };
        assert!(cfg.is_exempt("t:watchdog-orn"), "keyring row form");
        assert!(cfg.is_exempt("ten_UBB1F6Lf"), "bare row form");
        assert!(cfg.is_exempt("orn-probe-dry"));
        assert!(!cfg.is_exempt("t:acme"));
        assert!(!cfg.is_exempt(""));
    }
}
