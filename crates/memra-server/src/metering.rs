//! The admission/accounting seam (lane engine-billing-extraction-20260829).
//!
//! The server's job at this boundary is to ADMIT, DENY, and REPORT COUNTS. What
//! admission means — budgets, prices, tenancy policy — is the deployment's business,
//! supplied behind these traits. The stock binary wires the in-repo reference
//! implementation (`ledger::ReferenceMetering`); a deployment-owned binary can wire its own.
//! Everything here speaks tokens and verdicts, never money: the vocabulary is the
//! boundary, and it is what lets the policy half live outside this crate.
//!
//! The seam is `pub`: a deployment-owned binary wires its implementation through
//! [`crate::ServerWiring`] and runs the same server.

use std::any::Any;

/// An opaque reservation handle minted by [`Metering::reserve`] and consumed by
/// [`Metering::open`] of the SAME implementation. The server carries it between the
/// two calls without looking inside; dropping it un-consumed must release whatever
/// it holds (the reference implementation refunds on drop).
pub type Permit = Box<dyn Any + Send>;

/// The request identity a receipt is opened with. All borrowed: this is a view of
/// state the handler already owns, taken for the duration of one `open` call.
pub struct RequestMeta<'a> {
    pub request_id: &'a str,
    pub tenant: &'a str,
    /// The authenticated key's non-secret identification prefix, when the request
    /// carries per-key identity (multi-key ring). What per-PRINCIPAL policy means —
    /// spend caps, quotas — is the implementation's business.
    pub principal: Option<&'a str>,
    pub model: &'a str,
    pub route: &'static str,
    pub lane: &'static str,
    pub stream: bool,
    /// The request's EFFECTIVE completion-token bound at receipt time: the caller's
    /// `max_tokens`, or the model-default/`max_output` it resolved to
    /// (`apply_model_request_limits`); `None` when no bound resolved (context-bounded
    /// generation). D2 gap G4 (darklanes `research/d2-shadow-20260831/RESULTS.md`):
    /// the ledger row carries this so live-shadow replay reads the real bound instead
    /// of recovering it by inverting the billing reservation. An implementation with a
    /// positional row format appends this as a NEW trailing column (column order in an
    /// existing ledger is an API, so the discipline is append-only).
    pub max_tokens: Option<u64>,
    /// The context reservation the budget admission charged for this request:
    /// `prompt_tokens + completion bound`, in tokens (the same quantities handed to
    /// [`Metering::reserve`]). `None` when no reservation ran (limits not enforced, or
    /// the receipt records a pre-reservation rejection). Same G4 append-only column
    /// rule as `max_tokens`.
    pub reserved_ctx: Option<u64>,
}

/// Token counts as the worker measured them. The one usage shape that crosses the
/// seam; anything priced is derived from these on the implementation's side.
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageCounts {
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub completion_tokens: u64,
}

/// Why admission said no. Mirrors the HTTP contract the handlers already speak:
/// `Insufficient`/`Blocked` answer 402 (one recovery action for callers),
/// `Unenrolled` answers 402 with its own code, `Unavailable` is the fail-closed 500.
#[derive(Debug, PartialEq, Eq)]
pub enum AdmitError {
    Insufficient,
    Blocked,
    Unenrolled,
    /// The PRINCIPAL (per-key) spend ceiling is reached while the tenant itself may
    /// still have balance. Its own 402 code: the caller's recovery is raising or
    /// clearing the key's cap, not adding credit.
    PrincipalCapped,
    Unavailable(String),
}

/// Limits-source health for the operator metrics surface. Counts only.
#[derive(Debug, Clone, Copy)]
pub struct LimitsHealth {
    pub source_reload_failed: u64,
    pub source_reload_consecutive: u32,
    pub source_available: bool,
}

/// Per-deployment admission + usage accounting. One object, present iff the
/// deployment configured accounting at all (`AppState.metering: Option<Arc<dyn ..>>`
/// mirrors the old `request_ledger: Option<Ledger>` exactly).
pub trait Metering: Send + Sync {
    /// Whether per-tenant admission limits are configured at all. `false` = every
    /// authenticated tenant is admitted without reservation (counting may still run).
    fn enforces_limits(&self) -> bool;

    /// Whether this tenant is subject to limits. With limits enforced, an unknown
    /// tenant is NOT a free pass — the caller rejects it as unenrolled.
    fn is_limited(&self, tenant: &str) -> Result<bool, AdmitError>;

    /// Reserve headroom for a request's worst case, in tokens. `Ok(Some(permit))`
    /// rides the receipt and is settled to worker-truth usage; `Ok(None)` means the
    /// implementation needs no per-request hold. `principal` is the authenticated
    /// key's non-secret prefix when the request carries one — the hook for per-key
    /// policy (spend caps) on the implementation's side.
    fn reserve(
        &self,
        tenant: &str,
        principal: Option<&str>,
        model: &str,
        prompt_tokens: u64,
        completion_bound: u64,
    ) -> Result<Option<Permit>, AdmitError>;

    /// Open the request's usage receipt. Every terminal outcome settles it through
    /// one of the [`Receipt`] methods; dropping it unfinalized is the abandoned-client
    /// path and must stay safe (the reference implementation prices the partial).
    fn open(&self, meta: &RequestMeta<'_>, permit: Option<Permit>) -> Box<dyn Receipt>;

    /// Whether this tenant's requests are captured (a retention policy the
    /// implementation owns). The one pre-receipt check handlers make so an unmarked
    /// tenant never pays for a prompt copy; the receipt's own [`Receipt::wants_capture`]
    /// is the post-open gate and the implementation's settle-time re-check stays
    /// authoritative.
    fn captures(&self, _tenant: &str) -> bool {
        false
    }

    /// Limits-source health for the operator metrics surface, when limits exist.
    fn limits_health(&self) -> Option<LimitsHealth>;

    /// The graceful-drain deadline expired with requests still in flight: everything
    /// dropped from this moment on was killed by OUR shutdown, not abandoned by its
    /// client. Fault attribution (owner ruling 2026-08-23): the implementation must
    /// settle those drops without billing the caller. Latched — the process is exiting.
    fn drain_kill(&self) {}
}

/// One request's accounting record, admission row to terminal row. Method names
/// deliberately match the reference implementation's inherent methods so the
/// handler code reads identically through the seam.
pub trait Receipt: Send {
    /// Whether this receipt was opened captured. Gates the caller's lazy prompt build;
    /// `false` makes `arm_capture` a no-op.
    fn wants_capture(&self) -> bool {
        false
    }
    /// Attach the prompt payload to a captured request. Where it goes and the
    /// settle-time consent re-check belong to the implementation; the seam never
    /// names a storage type.
    fn arm_capture(&mut self, prompt: serde_json::Value);
    fn capture_completion_delta(&mut self, text: &str);
    fn record_prompt_usage(
        &mut self,
        prompt_tokens: u64,
        cached_prompt_tokens: u64,
    ) -> Result<(), String>;
    fn record_completion_token(&mut self) -> Result<(), String>;
    fn complete(&mut self, usage: UsageCounts, worker_elapsed_s: f64) -> Result<(), String>;
    /// Deadline-partial: billed like `complete` but census-distinct.
    fn complete_deadline_partial(
        &mut self,
        usage: UsageCounts,
        worker_elapsed_s: f64,
    ) -> Result<(), String>;
    fn reject(&mut self, status: u16, error_code: &str) -> Result<(), String>;
    /// Terminal rows with a NAMED zero-debit outcome (`deadline_exceeded`,
    /// `shed_deadline`, `shed_queue`, `shed_queue_wait`) — `reject`'s twin for
    /// outcomes the census distinguishes. Never bills.
    fn settle_unbilled(
        &mut self,
        outcome: &'static str,
        status: u16,
        error_code: &str,
    ) -> Result<(), String>;
}

/// What a metering factory gets to see at construction time: the server's loaded
/// model roster. Deliberately small — an implementation brings its own prices,
/// policies, and storage; the server only vouches for what it serves.
pub struct MeteringInit<'a> {
    pub models: &'a [String],
}

/// Deployment hook: build the metering implementation once models are loaded.
/// `Ok(None)` = no accounting (the stock no-ledger shape). An `Err` is a startup
/// FATAL — accounting configuration never fails open.
pub type MeteringFactory = Box<
    dyn FnOnce(&MeteringInit<'_>) -> Result<Option<std::sync::Arc<dyn Metering>>, String> + Send,
>;
