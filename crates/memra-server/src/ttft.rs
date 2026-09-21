use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const UNSET: u64 = u64::MAX;
const MAX_LIFECYCLE_EVENTS: u64 = 8192;
const MAX_IDENTITY_BYTES: usize = 256;
static NEXT_TRACE_ID: AtomicU64 = AtomicU64::new(1);

/// The sender's observed condition, not an inference about why a client left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReceiverCloseCause {
    ReceiverDropped,
    EventQueueOverflow,
}

impl ReceiverCloseCause {
    fn name(self) -> &'static str {
        match self {
            Self::ReceiverDropped => "receiver_dropped",
            Self::EventQueueOverflow => "event_queue_overflow",
        }
    }
}

/// Supplied only AFTER the caller has released the request's owned resources.
/// Only aborted retirement is instrumented. It does not mean proven client
/// cancellation: inspect close cause and HTTP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetirementOutcome {
    Aborted,
}

impl RetirementOutcome {
    fn name(self) -> &'static str {
        match self {
            Self::Aborted => "aborted",
        }
    }
}

/// The caller's actual after-release hook site, never inferred from phase.
/// This identifies a resource owner; it proves neither absence of GPU work nor
/// cache use. Readers requiring queue retirement must require literal WorkerQueue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetirementSite {
    WorkerQueue,
    ActiveSession,
}

impl RetirementSite {
    fn name(self) -> &'static str {
        match self {
            Self::WorkerQueue => "WorkerQueue",
            Self::ActiveSession => "ActiveSession",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Quantum {
    id: u64,
    route: &'static str,
    rows: usize,
    start_ns: u64,
    end_ns: Option<u64>,
    completed: Option<bool>,
    remaining_chunks: Option<usize>,
}

struct Lifecycle {
    request: Option<(String, String)>,
    client_trace_key: Option<String>,
    worker: Option<(u64, &'static str)>,
    phase: &'static str,
    active_quantum: Option<Quantum>,
    last_quantum: Option<Quantum>,
    quantum_count: u64,
    http_eof: bool,
    http_dropped: bool,
    http_pending_dropped: bool,
    close_cause: Option<ReceiverCloseCause>,
    retirement: Option<RetirementOutcome>,
    retirement_site: Option<RetirementSite>,
    first_error: Option<&'static str>,
    seq: u64,
    suppressed: u64,
    #[cfg(test)]
    events: Arc<Mutex<Vec<TestEvent>>>,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Self {
            request: None,
            client_trace_key: None,
            worker: None,
            phase: "unbound",
            active_quantum: None,
            last_quantum: None,
            quantum_count: 0,
            http_eof: false,
            http_dropped: false,
            http_pending_dropped: false,
            close_cause: None,
            retirement: None,
            retirement_site: None,
            first_error: None,
            seq: 0,
            suppressed: 0,
            #[cfg(test)]
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Lifecycle {
    fn invalidate(&mut self, reason: &'static str) {
        self.first_error.get_or_insert(reason);
    }

    fn bound(&self) -> bool {
        self.request.is_some() && self.worker.is_some()
    }

    fn require_worker(&mut self) {
        if !self.bound() {
            self.invalidate("missing_request_or_worker_binding");
        }
        if self.retirement.is_some() {
            self.invalidate("worker_event_after_retirement");
        }
    }

    fn http_state(&self) -> &'static str {
        if self.http_pending_dropped {
            return "pending_dropped";
        }
        match (self.http_eof, self.http_dropped) {
            (false, false) => "open",
            (true, false) => "normal_eof",
            (false, true) => "dropped_before_eof",
            (true, true) => "dropped_after_eof",
        }
    }
}

// A small std-only JSON string encoder keeps the diagnostic standalone-testable.
// Unlike Rust debug quoting, JSON requires \u0000-style escapes for ALL controls.
fn json_string(value: &str) -> String {
    use std::fmt::Write;
    let mut output = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0}'..='\u{1f}' => {
                let _ = write!(output, "\\u{:04x}", ch as u32);
            }
            _ => output.push(ch),
        }
    }
    output.push('"');
    output
}

fn json_option<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

#[derive(Default)]
struct Meta {
    request_id: String,
    model: String,
    path: String,
    prompt_tokens: usize,
}

/// Per-request TTFT phase trace. Allocated only when MEMRA_TTFT_TRACE=1.
///
/// Every timestamp is a microsecond offset from HTTP request arrival. Atomics let the
/// axum and GPU-worker threads stamp one shared timeline without locks; the metadata
/// mutex is touched only on the debug path and only a few times per request.
///
/// Separate `[request-lifecycle]` JSON records are evidence only. A per-trace
/// mutex serializes HTTP/worker observations AND their output, using elapsed
/// nanoseconds from this Trace's Instant (not CLOCK_MONOTONIC's absolute epoch).
/// `(pid, trace_id)` joins even early unbound events; the external capture binds
/// PID boot/start identity. No prompts, credentials or arbitrary header values
/// are read. The parent may supply ONLY its dedicated diagnostic correlation key
/// through bind_client_trace_key; that key never replaces server request identity.
/// At most 8192 ordinary records, one overflow record and one final trace_end are
/// emitted. Consumers must require the final record, complete bindings, no errors
/// or suppression, and the actual scenario observations; no record qualifies a run.
/// ordinary_event_limit advertises this fixed diagnostic capacity on every event.
/// The caller's preflight must fit its entire frozen target/draft/HTTP event budget
/// within it. This is not a latency/SLO guarantee; nothing is silently coalesced.
/// Actual prime evidence comes ONLY from explicit nonzero quantum calls. Legacy
/// prime timing marks run on cached empty-suffix MTP paths too and are not phase
/// evidence. Missing quantum events imply neither a cache hit nor absence of work:
/// uninstrumented routes remain unqualified. `phase` is the last observed lifecycle
/// state, not a claim of continuous coverage across uninstrumented caller code.
pub struct Trace {
    started: Instant,
    trace_id: u64,
    path: String,
    lifecycle: Mutex<Lifecycle>,
    meta: Mutex<Meta>,
    logged: AtomicBool,
    parsed: AtomicU64,
    submitted: AtomicU64,
    tokenize_start: AtomicU64,
    tokenize_end: AtomicU64,
    prime_start: AtomicU64,
    prime_end: AtomicU64,
    first_decode: AtomicU64,
    first_sse_byte: AtomicU64,
}

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_TTFT_TRACE").as_deref() == Ok("1"))
}

pub fn start(path: &str) -> Option<Arc<Trace>> {
    if !matches!(path, "/v1/completions" | "/v1/chat/completions") {
        return None;
    }
    enabled().then(|| Arc::new(Trace::new(path)))
}

impl Trace {
    #[cfg(test)]
    pub(crate) fn for_test(path: &str) -> Arc<Self> {
        Arc::new(Self::new(path))
    }

    #[cfg(test)]
    pub(crate) fn observe_for_test(&self) -> LifecycleObserver {
        let state = self.lifecycle.lock().unwrap_or_else(|e| e.into_inner());
        LifecycleObserver {
            events: state.events.clone(),
        }
    }

    fn new(path: &str) -> Self {
        let trace = Self {
            started: Instant::now(),
            trace_id: NEXT_TRACE_ID.fetch_add(1, Ordering::Relaxed),
            path: path.to_string(),
            lifecycle: Mutex::new(Lifecycle::default()),
            meta: Mutex::new(Meta {
                path: path.to_string(),
                ..Meta::default()
            }),
            logged: AtomicBool::new(false),
            parsed: AtomicU64::new(UNSET),
            submitted: AtomicU64::new(UNSET),
            tokenize_start: AtomicU64::new(UNSET),
            tokenize_end: AtomicU64::new(UNSET),
            prime_start: AtomicU64::new(UNSET),
            prime_end: AtomicU64::new(UNSET),
            first_decode: AtomicU64::new(UNSET),
            first_sse_byte: AtomicU64::new(UNSET),
        };
        trace.record("trace_start", None, |_, _| {});
        trace
    }
    fn now_us(&self) -> u64 {
        self.started.elapsed().as_micros().min(u64::MAX as u128) as u64
    }

    fn mark(&self, slot: &AtomicU64) -> bool {
        slot.compare_exchange(UNSET, self.now_us(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub fn bind_request(&self, request_id: &str, model: &str) {
        if let Ok(mut meta) = self.meta.lock() {
            meta.request_id = request_id.to_string();
            meta.model = model.to_string();
        }
        self.record("request_bound", None, |state, _| {
            if state.request.is_some() {
                state.invalidate("duplicate_request_binding");
            } else if request_id.is_empty()
                || model.is_empty()
                || request_id.len() > MAX_IDENTITY_BYTES
                || model.len() > MAX_IDENTITY_BYTES
            {
                state.invalidate("invalid_or_oversized_request_binding");
            } else {
                state.request = Some((request_id.to_string(), model.to_string()));
            }
        });
    }

    pub fn mark_parsed(&self) {
        self.mark(&self.parsed);
    }

    /// Optional x-memra-trace-id correlation, not request authority. Only a 32-byte
    /// lowercase hex key is accepted. Invalid values are never copied to evidence.
    /// Bind before the handler future/body ends; do not silently replace a key.
    pub fn bind_client_trace_key(&self, key: &str) {
        self.record("client_trace_key_bound", None, |state, _| {
            if key.len() != 32
                || !key
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            {
                state.invalidate("invalid_client_trace_key");
            } else if state.client_trace_key.is_some() {
                state.invalidate("duplicate_client_trace_key_binding");
            } else if state.http_eof
                || state.http_dropped
                || state.http_pending_dropped
                || state.retirement.is_some()
            {
                state.invalidate("late_client_trace_key_binding");
            } else {
                state.client_trace_key = Some(key.to_string());
            }
        });
    }

    pub fn mark_submitted(&self) {
        self.mark(&self.submitted);
    }

    pub fn mark_tokenize_start(&self) {
        self.mark(&self.tokenize_start);
    }

    pub fn mark_tokenize_end(&self, prompt_tokens: usize) {
        if let Ok(mut meta) = self.meta.lock() {
            meta.prompt_tokens = prompt_tokens;
        }
        self.mark(&self.tokenize_end);
    }

    /// Legacy first-stamp timing only. Cached empty-suffix MTP calls this without
    /// doing any prime work, so it must never enter an actual lifecycle phase.
    pub fn mark_prime_start(&self) {
        self.mark(&self.prime_start);
    }

    /// Legacy first-stamp timing only; not proof that actual prime work completed.
    pub fn mark_prime_end(&self) {
        self.mark(&self.prime_end);
    }

    pub fn mark_first_decode(&self) {
        // A fully cached prompt has no prime call. Represent that as a zero-duration
        // prime at the first decode boundary so the timeline remains ordered.
        if self.prime_start.load(Ordering::Acquire) == UNSET {
            self.mark(&self.prime_start);
        }
        if self.prime_end.load(Ordering::Acquire) == UNSET {
            self.mark(&self.prime_end);
        }
        if self.mark(&self.first_decode) {
            self.record("decode_start", None, |state, _| {
                state.require_worker();
                if !matches!(state.phase, "queued" | "bound" | "prime_finished")
                    || state.active_quantum.is_some()
                {
                    state.invalidate("decode_before_prime_or_quantum_end");
                }
                state.phase = "decode";
            });
        }
    }

    pub fn mark_first_sse_byte(&self) {
        self.mark(&self.first_sse_byte);
        self.log_once("first_sse_byte");
    }

    /// Bind once to the process-local worker-run epoch and actual worker dispatch
    /// route (e.g. shared_gpu_worker), not a speculative/numerical execution mode.
    /// A resumed/requeued request must not silently acquire a new binding on this trace.
    pub fn bind_worker(&self, generation: u64, route: &'static str) {
        self.record("worker_bound", None, |state, _| {
            if state.worker.is_some() {
                state.invalidate("duplicate_worker_binding");
            } else if route.is_empty() || route.len() > MAX_IDENTITY_BYTES {
                state.invalidate("invalid_worker_route");
            } else {
                state.worker = Some((generation, route));
                if state.request.is_none() {
                    state.invalidate("worker_bound_before_request");
                }
                if state.phase == "unbound" {
                    state.phase = "bound";
                }
            }
        });
    }

    /// Only the owner of the actual worker queue may publish this observation.
    pub fn mark_queued(&self) {
        self.record("queued", None, |state, _| {
            state.require_worker();
            if state.phase != "bound" {
                state.invalidate("duplicate_or_out_of_order_queue");
            }
            state.phase = "queued";
        });
    }

    /// #521 seam: call immediately before one actual nonzero prime quantum, not a
    /// prediction or a cache lookup that might skip work. This is the sole prime
    /// phase entry; legacy mark_prime_start is not required. Rows are the actual
    /// quantum's input row count. The producer stamps its own monotonic start.
    pub fn mark_prime_quantum_start(&self, route: &'static str, rows: usize) {
        self.record("prime_quantum_start", None, |state, at| {
            state.require_worker();
            if state.active_quantum.is_some() {
                state.invalidate("overlapping_prime_quantums");
                return; // Preserve the original quantum, including a concurrent HTTP drop.
            }
            if route.is_empty() || route.len() > MAX_IDENTITY_BYTES || rows == 0 {
                state.invalidate("invalid_quantum_route_or_rows");
                return;
            }
            if !matches!(state.phase, "bound" | "queued" | "prime") || state.close_cause.is_some() {
                state.invalidate("quantum_after_prime_finished_decode_or_receiver_close");
            }
            state.phase = "prime";
            state.quantum_count = state.quantum_count.saturating_add(1);
            state.active_quantum = Some(Quantum {
                id: state.quantum_count,
                route,
                rows,
                start_ns: at,
                end_ns: None,
                completed: None,
                remaining_chunks: None,
            });
        });
    }

    /// #521 seam: completed means successful quantum RETURN. Only a successful
    /// return with remaining_chunks=Some(0) proves whole-prime completion and moves
    /// phase to prime_finished. None stays unknown (phase remains prime, with no
    /// active quantum). Legacy mark_prime_end cannot fill that evidence gap.
    /// Never pass a caller-measured duration.
    pub fn mark_prime_quantum_end(&self, completed: bool, remaining_chunks: Option<usize>) {
        self.record("prime_quantum_end", None, |state, at| {
            state.require_worker();
            let Some(mut quantum) = state.active_quantum.take() else {
                state.invalidate("quantum_end_without_start");
                return;
            };
            quantum.end_ns = Some(at);
            quantum.completed = Some(completed);
            quantum.remaining_chunks = remaining_chunks;
            if !completed {
                state.invalidate("failed_prime_quantum");
            }
            state.last_quantum = Some(quantum);
            if completed && remaining_chunks == Some(0) {
                state.phase = "prime_finished";
            }
        });
    }

    /// Called by the HTTP wrapper only when polling the body yields normal EOF.
    /// An error/disconnect/Drop is not this event, even after some output arrived.
    pub fn mark_http_body_eof(&self) {
        self.record("http_body_eof", None, |state, _| {
            if state.request.is_none() {
                state.invalidate("http_eof_without_request_binding");
            }
            if state.http_eof || state.http_dropped || state.http_pending_dropped {
                state.invalidate("duplicate_or_late_http_eof");
                return;
            }
            state.http_eof = true;
        });
    }

    /// The wrapper's Drop calls this even after EOF. It records body lifetime,
    /// not receiver closure, request cancellation, or worker resource retirement.
    pub fn mark_http_body_drop(&self) {
        self.record("http_body_drop", None, |state, _| {
            if state.request.is_none() {
                state.invalidate("http_drop_without_request_binding");
            }
            if state.http_dropped {
                state.invalidate("duplicate_http_body_drop");
            }
            if state.http_pending_dropped {
                state.invalidate("body_drop_after_pending_drop");
            }
            state.http_dropped = true;
        });
    }

    /// The handler future was dropped BEFORE returning a response/body, including
    /// pre-header peek_admission. No body or EOF is invented. The actual receiver
    /// close and worker retirement must still be independently observed.
    pub fn mark_http_pending_drop(&self) {
        self.record("http_pending_drop", None, |state, _| {
            if state.request.is_none() {
                state.invalidate("http_pending_drop_without_request_binding");
            }
            if state.http_pending_dropped || state.http_dropped || state.http_eof {
                state.invalidate("duplicate_or_out_of_order_http_pending_drop");
                return;
            }
            state.http_pending_dropped = true;
        });
    }

    /// Pass the actual EventSender close reason. Overflow takes priority if both
    /// underlying conditions hold. This event NEVER asserts client cancellation.
    pub fn mark_receiver_closed(&self, cause: ReceiverCloseCause) {
        self.record("receiver_closed", Some(cause), |state, _| {
            state.require_worker();
            if state.close_cause.is_some() {
                state.invalidate("duplicate_receiver_close");
            } else {
                state.close_cause = Some(cause);
            }
        });
    }

    /// Record handoff/requeue, without inventing retirement or refreshing identity.
    /// This trace cannot provide a clean single-attempt history after a replay.
    pub fn mark_requeued(&self) {
        self.record("requeued", None, |state, _| {
            state.require_worker();
            state.invalidate("request_requeued");
        });
    }

    /// Explicit AFTER-release observation. A parent guard may call Aborted only
    /// after all Session fields drop, and must not arm on OOM requeue or panic.
    /// Trace/Arc destruction never calls this method on the caller's behalf.
    /// Historical unscoped-record fixture API. Production hooks require an explicit
    /// site via mark_retired_at; fixtures keep unknown/null instead of guessing it.
    #[cfg(test)]
    pub fn mark_retired(&self, outcome: RetirementOutcome) {
        self.record_retirement(outcome, None);
    }

    /// Explicit AFTER-release observation at the actual owner hook. Use WorkerQueue
    /// only after dropping the closed queued Box<Request>; ActiveSession only from
    /// the after-Session-drop guard. A stale queued phase does not imply queue work.
    /// The first retirement's site is immutable, including a legacy unknown site.
    pub fn mark_retired_at(&self, outcome: RetirementOutcome, site: RetirementSite) {
        self.record_retirement(outcome, Some(site));
    }

    fn record_retirement(&self, outcome: RetirementOutcome, site: Option<RetirementSite>) {
        self.record("retired", None, |state, _| {
            state.require_worker();
            if state.retirement.is_some() {
                state.invalidate("duplicate_retirement");
                return;
            }
            if state.active_quantum.is_some() {
                state.invalidate("retirement_before_quantum_return");
            }
            if state.close_cause.is_none() {
                state.invalidate("abort_retirement_without_close_observation");
            }
            state.retirement = Some(outcome);
            state.retirement_site = site;
        });
    }

    fn record(
        &self,
        event: &'static str,
        observed_close_cause: Option<ReceiverCloseCause>,
        update: impl FnOnce(&mut Lifecycle, u64),
    ) {
        use std::io::Write;
        let mut state = match self.lifecycle.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.invalidate("poisoned_lifecycle");
                state
            }
        };
        let elapsed = self.started.elapsed().as_nanos();
        if elapsed >= u64::MAX as u128 {
            state.invalidate("lifecycle_clock_overflow");
        }
        let at = elapsed.min(u64::MAX as u128) as u64;
        update(&mut state, at);
        state.seq = state.seq.saturating_add(1);
        let emitted_event = if event != "trace_end" && state.seq > MAX_LIFECYCLE_EVENTS {
            state.suppressed = state.suppressed.saturating_add(1);
            state.invalidate("lifecycle_event_limit");
            if state.suppressed > 1 {
                return;
            }
            "overflow"
        } else {
            event
        };
        let string = |value: Option<&str>| value.map_or_else(|| "null".to_string(), json_string);
        let quantum = state.active_quantum.or(state.last_quantum);
        let quantum_json = quantum.map_or_else(
            || "null".to_string(),
            |q| format!(
                "{{\"id\":{},\"route\":{},\"rows\":{},\"start_ns\":{},\"end_ns\":{},\"completed\":{},\"remaining_chunks\":{}}}",
                q.id, json_string(q.route), q.rows, q.start_ns, json_option(q.end_ns),
                json_option(q.completed), json_option(q.remaining_chunks)
            ),
        );
        let line = format!(
            "{{\"schema\":\"memra-request-lifecycle-v1\",\"pid\":{},\"trace_id\":{},\"seq\":{},\"ordinary_event_limit\":{},\"clock\":\"trace_elapsed_ns\",\"at_ns\":{},\"event\":{},\"client_trace_key\":{},\"request_id\":{},\"model\":{},\"http_route\":{},\"worker_generation\":{},\"worker_route\":{},\"phase\":{},\"quantum_active\":{},\"quantum\":{},\"http_body\":{},\"receiver_close_cause\":{},\"observed_close_cause\":{},\"retirement\":{},\"retirement_site\":{},\"bindings_complete\":{},\"sequence_valid\":{},\"first_error\":{},\"suppressed_events\":{},\"evidence_only\":true}}",
            std::process::id(),
            self.trace_id,
            state.seq,
            MAX_LIFECYCLE_EVENTS,
            at,
            json_string(emitted_event),
            string(state.client_trace_key.as_deref()),
            string(state.request.as_ref().map(|(id, _)| id.as_str())),
            string(state.request.as_ref().map(|(_, model)| model.as_str())),
            json_string(&self.path),
            json_option(state.worker.map(|(generation, _)| generation)),
            string(state.worker.map(|(_, route)| route)),
            json_string(state.phase),
            state.active_quantum.is_some(),
            quantum_json,
            json_string(state.http_state()),
            string(state.close_cause.map(ReceiverCloseCause::name)),
            string(observed_close_cause.map(ReceiverCloseCause::name)),
            string(state.retirement.map(RetirementOutcome::name)),
            string(state.retirement_site.map(RetirementSite::name)),
            state.bound(),
            state.first_error.is_none(),
            string(state.first_error),
            state.suppressed
        );
        // No diagnostic IO error or poisoned lifecycle mutex may unwind serving.
        // Hold the state lock through the write so different threads cannot publish
        // this trace's seq N+1 before seq N. No per-token calls reach this path.
        // Format the WHOLE record before writing: panic hooks can bypass stderr's
        // usual lock, and write_fmt may split the prefix, payload and newline.
        // A reader must still reject corrupted/missing records from the log sink.
        let record = format!("[request-lifecycle] {line}\n");
        let _ = std::io::stderr().lock().write_all(record.as_bytes());
        #[cfg(test)]
        {
            let snapshot = TestEvent {
                event: emitted_event,
                seq: state.seq,
                at_ns: at,
                phase: state.phase,
                quantum,
                quantum_active: state.active_quantum.is_some(),
                http: state.http_state(),
                valid: state.first_error.is_none(),
                json: line,
            };
            state.events.lock().unwrap().push(snapshot);
        }
    }

    fn end_lifecycle(&self) {
        self.record("trace_end", None, |state, _| {
            if !state.bound() {
                state.invalidate("trace_ended_without_binding");
            }
            if state.retirement.is_none() {
                state.invalidate("trace_ended_without_explicit_retirement");
            }
            if state.active_quantum.is_some() {
                state.invalidate("trace_ended_with_open_quantum");
            }
            if !state.http_dropped && !state.http_pending_dropped {
                state.invalidate("trace_ended_without_http_lifetime_end");
            }
        });
    }

    fn value(slot: &AtomicU64) -> Option<u64> {
        match slot.load(Ordering::Acquire) {
            UNSET => None,
            value => Some(value),
        }
    }

    fn ms(value: Option<u64>) -> String {
        value
            .map(|us| format!("{:.3}", us as f64 / 1_000.0))
            .unwrap_or_else(|| "na".to_string())
    }

    fn delta_ms(start: Option<u64>, end: Option<u64>) -> String {
        match (start, end) {
            (Some(start), Some(end)) => Self::ms(Some(end.saturating_sub(start))),
            _ => "na".to_string(),
        }
    }

    fn log_once(&self, outcome: &str) {
        if self
            .logged
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let parsed = Self::value(&self.parsed);
        let submitted = Self::value(&self.submitted);
        let tokenize_start = Self::value(&self.tokenize_start);
        let tokenize_end = Self::value(&self.tokenize_end);
        let prime_start = Self::value(&self.prime_start);
        let prime_end = Self::value(&self.prime_end);
        let first_decode = Self::value(&self.first_decode);
        let first_sse_byte = Self::value(&self.first_sse_byte);
        let meta = self.meta.lock().ok();
        let request_id = meta
            .as_ref()
            .map(|m| m.request_id.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or("unknown");
        let model = meta
            .as_ref()
            .map(|m| m.model.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or("unknown");
        let path = meta.as_ref().map(|m| m.path.as_str()).unwrap_or("unknown");
        let prompt_tokens = meta.as_ref().map(|m| m.prompt_tokens).unwrap_or(0);

        eprintln!(
            "[ttft] id={request_id} model={model:?} path={path:?} prompt_tokens={prompt_tokens} \
             outcome={outcome} request_parse_ms={} admission_ms={} queue_wait_ms={} \
             tokenize_ms={} prime_wait_ms={} prime_ms={} decode_wait_ms={} sse_handoff_ms={} \
             request_parse_end_ms={} admission_end_ms={} tokenize_start_ms={} tokenize_end_ms={} \
             prime_start_ms={} prime_end_ms={} first_decode_ms={} first_sse_byte_ms={} total_ms={}",
            Self::ms(parsed),
            Self::delta_ms(parsed, submitted),
            Self::delta_ms(submitted, tokenize_start),
            Self::delta_ms(tokenize_start, tokenize_end),
            Self::delta_ms(tokenize_end, prime_start),
            Self::delta_ms(prime_start, prime_end),
            Self::delta_ms(prime_end, first_decode),
            Self::delta_ms(first_decode, first_sse_byte),
            Self::ms(parsed),
            Self::ms(submitted),
            Self::ms(tokenize_start),
            Self::ms(tokenize_end),
            Self::ms(prime_start),
            Self::ms(prime_end),
            Self::ms(first_decode),
            Self::ms(first_sse_byte),
            Self::ms(
                first_sse_byte
                    .or(first_decode)
                    .or(prime_end)
                    .or(tokenize_end)
                    .or(parsed)
            ),
        );
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.log_once("ended_without_sse");
        self.end_lifecycle();
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct TestEvent {
    event: &'static str,
    seq: u64,
    at_ns: u64,
    phase: &'static str,
    quantum: Option<Quantum>,
    quantum_active: bool,
    http: &'static str,
    valid: bool,
    json: String,
}

/// A test-only read view that outlives the Trace without keeping that Trace alive.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct LifecycleObserver {
    events: Arc<Mutex<Vec<TestEvent>>>,
}

#[cfg(test)]
impl LifecycleObserver {
    pub(crate) fn json_lines(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|event| event.json.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn trace() -> Arc<Trace> {
        let trace = Trace::for_test("/v1/chat/completions");
        trace.bind_request("minted-request", "model");
        trace.bind_worker(7, "plain");
        trace.mark_queued();
        trace
    }

    fn observe(trace: &Trace) -> Arc<Mutex<Vec<TestEvent>>> {
        trace.lifecycle.lock().unwrap().events.clone()
    }

    fn event(events: &[TestEvent], name: &str) -> TestEvent {
        events.iter().find(|e| e.event == name).unwrap().clone()
    }

    fn coherent(events: &[TestEvent]) {
        assert!(events.windows(2).all(|w| w[0].seq < w[1].seq));
        assert!(events.windows(2).all(|w| w[0].at_ns <= w[1].at_ns));
        assert!(events.iter().all(|e| e.valid), "{events:?}");
    }

    #[test]
    fn http_drop_during_quantum_keeps_phase_until_actual_return_and_retirement() {
        let trace = trace();
        let events = observe(&trace);
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = trace.clone();
        let thread = std::thread::spawn(move || {
            worker.mark_prime_quantum_start("chunked", 32);
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            worker.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
            worker.mark_prime_quantum_end(true, Some(2));
            worker.mark_retired(RetirementOutcome::Aborted);
        });
        started_rx.recv().unwrap();
        trace.mark_http_body_drop();
        release_tx.send(()).unwrap();
        thread.join().unwrap();
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        let body = event(&events, "http_body_drop");
        let closed = event(&events, "receiver_closed");
        let end = event(&events, "prime_quantum_end");
        let retired = event(&events, "retired");
        assert_eq!(body.phase, "prime");
        assert_eq!(body.http, "dropped_before_eof");
        assert!(body.quantum_active && closed.quantum_active);
        assert_eq!(body.quantum.unwrap().end_ns, None);
        assert_eq!(body.quantum.unwrap().id, end.quantum.unwrap().id);
        assert!(body.at_ns <= closed.at_ns && closed.at_ns <= end.at_ns);
        assert_eq!(end.quantum.unwrap().end_ns, Some(end.at_ns));
        assert_eq!(end.quantum.unwrap().completed, Some(true));
        assert_eq!(end.quantum.unwrap().remaining_chunks, Some(2));
        assert_eq!(
            end.phase, "prime",
            "successful quantum is not whole-prime end"
        );
        assert!(end.at_ns <= retired.at_ns);
        assert!(!retired.quantum_active);
        assert_eq!(events.last().unwrap().event, "trace_end");
        assert!(retired.json.contains("\"retirement\":\"aborted\""));
    }

    #[test]
    fn http_drop_after_quantum_return_keeps_unknown_remaining_and_closed_quantum() {
        let trace = trace();
        let events = observe(&trace);
        let worker = trace.clone();
        std::thread::spawn(move || {
            worker.mark_prime_quantum_start("monolithic", 128);
            worker.mark_prime_quantum_end(true, None);
        })
        .join()
        .unwrap();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        let end = event(&events, "prime_quantum_end");
        let body = event(&events, "http_body_drop");
        assert!(!body.quantum_active);
        assert_eq!(body.quantum.unwrap().end_ns, Some(end.at_ns));
        assert_eq!(body.quantum.unwrap().remaining_chunks, None);
        assert!(body.json.contains("\"remaining_chunks\":null"));
        assert!(end.at_ns <= body.at_ns);
    }

    #[test]
    fn cached_decode_preserves_legacy_stamps_without_synthetic_prime_events() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_parsed();
        trace.mark_submitted();
        trace.mark_tokenize_start();
        trace.mark_tokenize_end(12);
        trace.mark_first_decode();
        let stamps = (
            Trace::value(&trace.prime_start),
            Trace::value(&trace.prime_end),
            Trace::value(&trace.first_decode),
        );
        for _ in 0..100 {
            trace.mark_first_decode();
        }
        assert!(stamps.0.is_some() && stamps.1.is_some() && stamps.2.is_some());
        assert_eq!(
            stamps,
            (
                Trace::value(&trace.prime_start),
                Trace::value(&trace.prime_end),
                Trace::value(&trace.first_decode)
            )
        );
        trace.mark_first_sse_byte();
        trace.mark_http_body_eof();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        assert!(!events.iter().any(|e| e.event.starts_with("prime")));
        assert_eq!(
            events.iter().filter(|e| e.event == "decode_start").count(),
            1
        );
        assert_eq!(event(&events, "http_body_drop").http, "dropped_after_eof");
    }

    #[test]
    fn cached_mtp_legacy_worker_call_replay_is_not_actual_prime_evidence() {
        let trace = trace();
        let events = observe(&trace);
        // AD526-LIFECYCLE-1: step_session marks start before testing an empty
        // suffix; an existing next_pred/pending_tok bypasses the prime walker.
        // Its decode callback then marks end and first decode. Replay that exact
        // existing caller sequence, including HTTP cancellation before callback.
        trace.mark_prime_start();
        let old_prime_start = Trace::value(&trace.prime_start);
        trace.mark_http_body_drop();
        trace.mark_prime_end();
        trace.mark_first_decode();
        let old_prime_end = Trace::value(&trace.prime_end);
        trace.mark_prime_start();
        trace.mark_prime_end();
        assert!(old_prime_start.is_some() && old_prime_end.is_some());
        assert_eq!(old_prime_start, Trace::value(&trace.prime_start));
        assert_eq!(old_prime_end, Trace::value(&trace.prime_end));
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        assert!(!events.iter().any(|e| e.event.starts_with("prime")));
        assert!(
            events
                .iter()
                .all(|e| e.phase != "prime" && e.quantum.is_none())
        );
        assert_eq!(event(&events, "decode_start").phase, "decode");
        let body = event(&events, "http_body_drop");
        assert!(!body.quantum_active && body.phase != "prime");
        // This rules out prime-cancel evidence, not other unobserved work or a
        // cache-hit claim. Only the fixture's source-established path was zero-prime.
    }

    #[test]
    fn unknown_remaining_cannot_be_completed_by_legacy_end_or_inferred_from_decode() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_prime_quantum_start("observed", 8);
        trace.mark_prime_quantum_end(true, None);
        trace.mark_prime_end();
        trace.mark_first_decode(); // Real decode is still recorded, but gap is not repaired.
        trace.mark_http_body_eof();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        assert_eq!(event(&events, "prime_quantum_end").phase, "prime");
        assert_eq!(
            event(&events, "prime_quantum_end")
                .quantum
                .unwrap()
                .remaining_chunks,
            None
        );
        let decode = event(&events, "decode_start");
        assert_eq!(decode.phase, "decode");
        assert!(!decode.valid);
        assert!(!events.iter().any(|e| e.phase == "prime_finished"));
        assert!(!events.last().unwrap().valid);
    }

    #[test]
    fn aborted_retirement_does_not_claim_unknown_prime_completion() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_prime_quantum_start("observed", 8);
        trace.mark_prime_quantum_end(true, None);
        trace.mark_prime_end();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        let retired = event(&events, "retired");
        assert_eq!(retired.phase, "prime");
        assert_eq!(retired.quantum.unwrap().remaining_chunks, None);
        assert!(retired.valid); // An abort retires unfinished work; it does not finish prime.
        assert!(!events.iter().any(|e| e.phase == "prime_finished"));
    }

    #[test]
    fn legacy_end_during_observed_quantum_cannot_close_it_or_move_cancel_phase() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_prime_quantum_start("observed", 16);
        trace.mark_prime_end();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_prime_quantum_end(true, Some(0));
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        let body = event(&events, "http_body_drop");
        assert_eq!(body.phase, "prime");
        assert!(body.quantum_active);
        assert_eq!(body.quantum.unwrap().end_ns, None);
        let end = event(&events, "prime_quantum_end");
        assert_eq!(end.phase, "prime_finished");
        assert!(body.at_ns <= end.at_ns && end.at_ns <= event(&events, "retired").at_ns);
    }

    #[test]
    fn failed_zero_remaining_and_invalid_quantums_do_not_fabricate_completion() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_prime_quantum_start("observed", 8);
        trace.mark_prime_quantum_end(false, Some(0));
        trace.mark_prime_end();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        let end = event(&events, "prime_quantum_end");
        assert_eq!(end.quantum.unwrap().completed, Some(false));
        assert_eq!(end.quantum.unwrap().remaining_chunks, Some(0));
        assert_eq!(end.phase, "prime");
        assert!(!events.iter().any(|e| e.phase == "prime_finished"));
        drop(events);
        for rows in [0, 8] {
            let trace = self::trace();
            let events = observe(&trace);
            if rows != 0 {
                trace.mark_prime_quantum_start("first", 8);
                trace.mark_prime_quantum_end(true, Some(0));
            }
            trace.mark_prime_quantum_start("invalid", rows); // Zero work or after actual completion.
            let last = events.lock().unwrap().last().unwrap().clone();
            assert!(!last.valid);
            if rows == 0 {
                assert_eq!(last.phase, "queued");
                assert!(last.quantum.is_none());
            }
        }
    }

    #[test]
    fn actual_quantums_enter_prime_and_only_known_successful_zero_finishes_it() {
        let trace = trace();
        let events = observe(&trace);
        for remaining in [Some(2), None, Some(0)] {
            // Timing stamps neither start nor finish actual prime evidence.
            trace.mark_prime_start();
            trace.mark_prime_quantum_start("chunked", 16);
            trace.mark_prime_end();
            trace.mark_prime_quantum_end(true, remaining);
        }
        trace.mark_prime_end();
        trace.mark_first_decode();
        for _ in 0..10 {
            trace.mark_prime_start();
            trace.mark_prime_end();
            trace.mark_first_decode();
        }
        trace.mark_http_body_eof();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        assert_eq!(
            events.iter().filter(|e| e.event == "prime_start").count(),
            0
        );
        assert_eq!(events.iter().filter(|e| e.event == "prime_end").count(), 0);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event == "prime_quantum_end")
                .count(),
            3
        );
        let ends: Vec<_> = events
            .iter()
            .filter(|e| e.event == "prime_quantum_end")
            .collect();
        assert_eq!(ends[0].phase, "prime");
        assert_eq!(ends[1].phase, "prime");
        assert_eq!(ends[1].quantum.unwrap().remaining_chunks, None);
        assert_eq!(ends[2].phase, "prime_finished");
        assert_eq!(ends[2].quantum.unwrap().remaining_chunks, Some(0));
    }

    #[test]
    fn close_cause_does_not_equate_overflow_or_normal_eof_with_client_cancel() {
        for cause in [
            ReceiverCloseCause::EventQueueOverflow,
            ReceiverCloseCause::ReceiverDropped,
        ] {
            let trace = trace();
            let events = observe(&trace);
            trace.mark_first_decode();
            trace.mark_http_body_eof();
            trace.mark_http_body_drop();
            trace.mark_receiver_closed(cause);
            trace.mark_retired(RetirementOutcome::Aborted);
            drop(trace);
            let events = events.lock().unwrap();
            coherent(&events);
            let closed = event(&events, "receiver_closed");
            assert_eq!(closed.http, "dropped_after_eof");
            assert!(closed.json.contains(&format!(
                "\"observed_close_cause\":{}",
                json_string(cause.name())
            )));
            assert!(!closed.json.contains("client_cancel"));
        }
    }

    #[test]
    fn normal_eof_after_drop_and_duplicate_close_cannot_erase_earlier_evidence() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_http_body_drop();
        trace.mark_http_body_eof();
        trace.mark_receiver_closed(ReceiverCloseCause::EventQueueOverflow);
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        let last = events.last().unwrap();
        assert!(!last.valid);
        assert_eq!(last.http, "dropped_before_eof");
        let duplicate = events
            .iter()
            .rfind(|e| e.event == "receiver_closed")
            .unwrap();
        assert!(
            duplicate
                .json
                .contains("\"receiver_close_cause\":\"event_queue_overflow\"")
        );
        assert!(
            duplicate
                .json
                .contains("\"observed_close_cause\":\"receiver_dropped\"")
        );
    }

    #[test]
    fn unbound_and_rebound_epochs_never_become_clean_history() {
        let trace = Trace::new("/v1/completions");
        let events = observe(&trace);
        trace.mark_prime_quantum_start("unbound-quantum", 8);
        trace.bind_request("first", "model");
        trace.bind_worker(2, "first-route");
        trace.bind_worker(3, "replacement-route");
        trace.bind_request("second", "other");
        // The old TTFT metadata still follows its old last-bind behavior.
        assert_eq!(trace.meta.lock().unwrap().request_id, "second");
        drop(trace);
        let events = events.lock().unwrap();
        assert!(!events.last().unwrap().valid);
        assert!(
            events
                .last()
                .unwrap()
                .json
                .contains("\"request_id\":\"first\"")
        );
        assert!(
            events
                .last()
                .unwrap()
                .json
                .contains("\"worker_generation\":2")
        );
    }

    #[test]
    fn premature_or_duplicate_retirement_and_failed_quantum_are_not_clean() {
        let cases: [fn(&Trace); 6] = [
            |t| t.mark_prime_quantum_end(true, None),
            |t| t.mark_prime_quantum_start("q", 0),
            |t| {
                t.mark_prime_start();
                t.mark_prime_quantum_start("q", 8);
                t.mark_retired(RetirementOutcome::Aborted);
            },
            |t| {
                t.mark_prime_start();
                t.mark_prime_quantum_start("q", 8);
                t.mark_prime_quantum_start("overlap", 8);
            },
            |t| {
                t.mark_prime_start();
                t.mark_prime_quantum_start("q", 8);
                t.mark_prime_quantum_end(false, None);
                t.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
                t.mark_retired(RetirementOutcome::Aborted);
            },
            |t| {
                t.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
                t.mark_retired(RetirementOutcome::Aborted);
                t.mark_retired(RetirementOutcome::Aborted);
            },
        ];
        for case in cases {
            let trace = trace();
            let events = observe(&trace);
            case(&trace);
            drop(trace);
            assert!(!events.lock().unwrap().last().unwrap().valid);
        }
    }

    #[test]
    fn drop_and_worker_panic_never_synthesize_retirement_or_quantum_end() {
        let trace = trace();
        let events = observe(&trace);
        let worker = trace.clone();
        let result = std::thread::spawn(move || {
            worker.mark_prime_quantum_start("q", 64);
            panic!("controlled worker panic");
        })
        .join();
        assert!(result.is_err());
        trace.mark_http_body_drop();
        drop(trace);
        let events = events.lock().unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e.event, "retired" | "prime_quantum_end"))
        );
        assert!(events.last().unwrap().quantum_active);
        assert!(!events.last().unwrap().valid);
        assert!(events.last().unwrap().json.contains("\"retirement\":null"));
    }

    #[test]
    fn explicit_abort_guard_observes_resource_drop_but_never_arms_on_unwind() {
        struct Resource(Arc<AtomicBool>);
        impl Drop for Resource {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct AfterResources {
            trace: Arc<Trace>,
            released: Arc<AtomicBool>,
        }
        impl Drop for AfterResources {
            fn drop(&mut self) {
                if !std::thread::panicking() {
                    assert!(self.released.load(Ordering::Acquire));
                    self.trace.mark_retired(RetirementOutcome::Aborted);
                }
            }
        }
        for panic in [false, true] {
            let trace = trace();
            let events = observe(&trace);
            let released = Arc::new(AtomicBool::new(false));
            trace.mark_http_body_drop();
            let worker = trace.clone();
            let worker_released = released.clone();
            let outcome = std::thread::spawn(move || {
                // Same declaration order required of the caller's real Session.
                let _after = AfterResources {
                    trace: worker.clone(),
                    released: worker_released.clone(),
                };
                let _resource = Resource(worker_released);
                worker.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
                if panic {
                    panic!("controlled unwind before retirement");
                }
                // Scope exit (including early return) drops resource BEFORE guard.
            })
            .join();
            assert_eq!(outcome.is_err(), panic);
            assert!(released.load(Ordering::Acquire));
            drop(trace);
            let events = events.lock().unwrap();
            assert_eq!(events.iter().any(|e| e.event == "retired"), !panic);
            assert_eq!(events.last().unwrap().valid, !panic);
        }
    }

    #[test]
    fn requeue_does_not_refresh_generation_or_claim_retirement() {
        let trace = trace();
        let events = observe(&trace);
        trace.mark_requeued();
        trace.bind_worker(8, "resumed");
        trace.mark_http_body_drop();
        drop(trace);
        let events = events.lock().unwrap();
        assert!(!events.last().unwrap().valid);
        assert!(!events.iter().any(|e| e.event == "retired"));
        assert!(
            events
                .last()
                .unwrap()
                .json
                .contains("\"worker_generation\":7")
        );
    }

    #[test]
    fn poisoned_state_is_evident_without_poison_unwinding_serving() {
        let trace = trace();
        let events = observe(&trace);
        let other = trace.clone();
        assert!(
            std::thread::spawn(move || {
                let _lock = other.lifecycle.lock().unwrap();
                panic!("controlled state poison");
            })
            .join()
            .is_err()
        );
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        assert!(!events.last().unwrap().valid);
        assert!(events.last().unwrap().json.contains("poisoned_lifecycle"));
    }

    #[test]
    fn event_cap_reports_loss_and_always_retains_final_end_record() {
        let trace = trace();
        let events = observe(&trace);
        assert_eq!(MAX_LIFECYCLE_EVENTS, 8192);
        for _ in 0..(MAX_LIFECYCLE_EVENTS / 2 + 16) {
            trace.mark_prime_quantum_start("q", 1);
            trace.mark_prime_quantum_end(true, None);
        }
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let events = events.lock().unwrap();
        assert_eq!(events.len(), MAX_LIFECYCLE_EVENTS as usize + 2);
        assert_eq!(events.iter().filter(|e| e.event == "overflow").count(), 1);
        for (i, e) in events
            .iter()
            .take(MAX_LIFECYCLE_EVENTS as usize)
            .enumerate()
        {
            assert_eq!(
                e.seq,
                i as u64 + 1,
                "ordinary records must not be coalesced"
            );
        }
        let overflow = &events[MAX_LIFECYCLE_EVENTS as usize];
        assert_eq!(overflow.event, "overflow");
        assert_eq!(overflow.seq, MAX_LIFECYCLE_EVENTS + 1);
        assert!(!overflow.valid);
        assert!(
            events
                .iter()
                .all(|e| e.json.contains("\"ordinary_event_limit\":8192"))
        );
        assert_eq!(events.last().unwrap().event, "trace_end");
        assert!(!events.last().unwrap().valid);
        assert!(
            events
                .last()
                .unwrap()
                .json
                .contains("lifecycle_event_limit")
        );
        assert!(
            !events
                .last()
                .unwrap()
                .json
                .contains("\"suppressed_events\":0")
        );
    }

    #[test]
    fn long_target_and_draft_quantum_protocol_fits_without_suppression() {
        let trace = trace();
        let events = observe(&trace);
        let target_quanta = 131_072 / 1024;
        let draft_quanta = 131_072 / 512;
        let total = target_quanta + draft_quanta;
        for i in 0..total {
            let (route, rows) = if i < target_quanta {
                ("target_prime", 1024)
            } else {
                ("draft_fill", 512)
            };
            trace.mark_prime_quantum_start(route, rows);
            trace.mark_prime_quantum_end(true, Some(total - i - 1));
        }
        trace.mark_first_decode();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired_at(RetirementOutcome::Aborted, RetirementSite::ActiveSession);
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        assert!(events.len() > 256 && events.len() < MAX_LIFECYCLE_EVENTS as usize);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event == "prime_quantum_start")
                .count(),
            total
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event == "prime_quantum_end")
                .count(),
            total
        );
        for route in ["target_prime", "draft_fill"] {
            let rows: usize = events
                .iter()
                .filter(|e| e.event == "prime_quantum_start")
                .filter_map(|e| e.quantum)
                .filter(|q| q.route == route)
                .map(|q| q.rows)
                .sum();
            assert_eq!(rows, 131_072);
        }
        assert!(
            events
                .iter()
                .all(|e| e.json.contains("\"ordinary_event_limit\":8192")
                    && e.json.contains("\"suppressed_events\":0"))
        );
        assert!(!events.iter().any(|e| e.event == "overflow"));
        assert_eq!(events.last().unwrap().event, "trace_end");
        assert!(
            events
                .last()
                .unwrap()
                .json
                .contains("\"retirement_site\":\"ActiveSession\"")
        );
        // Actual producer protocol only: no model/GPU work or latency/SLO assertion.
    }

    #[test]
    fn legacy_retirement_keeps_unknown_site_regardless_of_observed_phase() {
        for decoded in [false, true] {
            let trace = trace();
            let events = observe(&trace);
            if decoded {
                trace.mark_first_decode();
            }
            trace.mark_http_body_drop();
            trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
            trace.mark_retired(RetirementOutcome::Aborted);
            drop(trace);
            let events = events.lock().unwrap();
            coherent(&events);
            assert_eq!(
                event(&events, "retired").phase,
                if decoded { "decode" } else { "queued" }
            );
            assert!(
                events
                    .iter()
                    .all(|e| e.json.contains("\"retirement_site\":null"))
            );
        }
    }

    #[test]
    fn explicit_active_site_does_not_inherit_a_stale_queued_phase() {
        for site in [RetirementSite::WorkerQueue, RetirementSite::ActiveSession] {
            let trace = trace();
            let events = observe(&trace);
            // The producer has not observed any actual prime/decode boundary.
            // In real partially instrumented code this phase can be stale.
            trace.mark_http_pending_drop();
            trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
            trace.mark_retired_at(RetirementOutcome::Aborted, site);
            drop(trace);
            let events = events.lock().unwrap();
            coherent(&events);
            let retired = event(&events, "retired");
            assert_eq!(retired.phase, "queued");
            assert!(retired.quantum.is_none());
            let expected = format!("\"retirement_site\":{}", json_string(site.name()));
            assert!(retired.json.contains(&expected));
            assert!(events.last().unwrap().json.contains(&expected));
            if site == RetirementSite::ActiveSession {
                assert!(!retired.json.contains("WorkerQueue"));
            }
        }
    }

    #[test]
    fn retirement_site_cannot_be_relabelled_even_from_unknown() {
        let sites = [
            None,
            Some(RetirementSite::WorkerQueue),
            Some(RetirementSite::ActiveSession),
        ];
        for first in sites {
            for second in sites {
                let trace = trace();
                let events = observe(&trace);
                trace.mark_http_body_drop();
                trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
                for site in [first, second] {
                    match site {
                        Some(site) => trace.mark_retired_at(RetirementOutcome::Aborted, site),
                        None => trace.mark_retired(RetirementOutcome::Aborted),
                    }
                }
                drop(trace);
                let events = events.lock().unwrap();
                assert!(event(&events, "retired").valid);
                assert!(!events.last().unwrap().valid);
                let value = first.map_or_else(|| "null".to_string(), |s| json_string(s.name()));
                let expected = format!("\"retirement_site\":{value}");
                assert!(
                    events
                        .iter()
                        .filter(|e| e.event == "retired" || e.event == "trace_end")
                        .all(|e| e.json.contains(&expected))
                );
            }
        }
    }

    #[test]
    fn explicit_retirement_site_uses_the_same_abort_validation() {
        for site in [
            None,
            Some(RetirementSite::WorkerQueue),
            Some(RetirementSite::ActiveSession),
        ] {
            let trace = trace();
            let events = observe(&trace);
            trace.mark_http_body_drop();
            // No sender-close observation: naming a hook site must not bypass it.
            match site {
                Some(site) => trace.mark_retired_at(RetirementOutcome::Aborted, site),
                None => trace.mark_retired(RetirementOutcome::Aborted),
            }
            drop(trace);
            let events = events.lock().unwrap();
            let retired = event(&events, "retired");
            assert!(!retired.valid);
            assert!(
                retired
                    .json
                    .contains("abort_retirement_without_close_observation")
            );
        }
    }

    #[test]
    fn json_strings_are_escaped_and_oversized_bindings_refuse_without_truncation() {
        assert_eq!(
            json_string("\"\\\n\t\0\u{001b}é"),
            "\"\\\"\\\\\\u000a\\u0009\\u0000\\u001bé\""
        );
        let trace = Trace::new("/v1/chat/completions");
        let events = observe(&trace);
        trace.bind_request("id\"\n\\", "model\té");
        trace.bind_worker(0, "route\r\"x");
        trace.mark_queued();
        trace.mark_http_body_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::EventQueueOverflow);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        coherent(&events.lock().unwrap());
        let trace = Trace::new("/v1/completions");
        let events = observe(&trace);
        trace.bind_request(&"x".repeat(MAX_IDENTITY_BYTES + 1), "model");
        drop(trace);
        assert!(
            events
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .json
                .contains("\"request_id\":null")
        );
        assert!(start("/not-traced").is_none());
    }

    #[test]
    fn start_remains_gated_by_existing_opt_in() {
        let selected = std::env::var("MEMRA_TTFT_TRACE").as_deref() == Ok("1");
        assert_eq!(start("/v1/completions").is_some(), selected);
        assert!(start("/not-traced").is_none());
    }

    #[test]
    fn pending_drop_while_queued_joins_early_client_key_without_inventing_body() {
        let trace = Trace::for_test("/v1/chat/completions");
        let observer = trace.observe_for_test();
        let events = observe(&trace);
        let key = "0123456789abcdef0123456789abcdef";
        trace.bind_client_trace_key(key); // HTTP header before next.run/minted ID.
        assert!(
            observer
                .json_lines()
                .last()
                .unwrap()
                .contains("\"request_id\":null")
        );
        trace.bind_request("minted-later", "model");
        let worker = trace.clone();
        let (queued_tx, queued_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            worker.bind_worker(9, "shared_gpu_worker");
            worker.mark_queued();
            queued_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            worker.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
            worker.mark_retired(RetirementOutcome::Aborted);
        });
        queued_rx.recv().unwrap();
        trace.mark_http_pending_drop();
        release_tx.send(()).unwrap();
        thread.join().unwrap();
        drop(trace);
        let events = events.lock().unwrap();
        coherent(&events);
        let pending = event(&events, "http_pending_drop");
        assert_eq!(pending.phase, "queued");
        assert_eq!(pending.http, "pending_dropped");
        assert!(pending.quantum.is_none());
        assert!(pending.json.contains(key) && pending.json.contains("minted-later"));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e.event, "http_body_eof" | "http_body_drop"))
        );
        drop(events);
        let final_lines = observer.json_lines();
        assert!(
            final_lines
                .last()
                .unwrap()
                .contains("\"event\":\"trace_end\"")
        );
        assert!(
            final_lines
                .last()
                .unwrap()
                .contains("\"sequence_valid\":true")
        );
    }

    #[test]
    fn client_key_validation_never_echoes_bad_values_or_replaces_first_binding() {
        for bad in [
            "",
            "0123456789abcdef0123456789abcde",
            "0123456789abcdef0123456789abcdef0",
            "0123456789abcdef0123456789abcdeF",
            "0123456789abcdef0123456789abcdeg",
            "Bearer private-invalid-value",
            "\"\nsecret",
            "é123456789abcdef0123456789abcdef",
        ] {
            let trace = Trace::for_test("/v1/completions");
            let observer = trace.observe_for_test();
            trace.bind_client_trace_key(bad);
            drop(trace);
            let lines = observer.json_lines();
            assert!(lines.last().unwrap().contains("invalid_client_trace_key"));
            assert!(
                lines
                    .iter()
                    .all(|line| line.contains("\"client_trace_key\":null"))
            );
            if !bad.is_empty() {
                assert!(!lines.iter().any(|line| line.contains(bad)));
            }
        }
        for second in [
            "0123456789abcdef0123456789abcdef",
            "ffffffffffffffffffffffffffffffff",
        ] {
            let trace = Trace::for_test("/v1/completions");
            let observer = trace.observe_for_test();
            trace.bind_client_trace_key("0123456789abcdef0123456789abcdef");
            trace.bind_client_trace_key(second);
            drop(trace);
            let lines = observer.json_lines();
            assert!(
                lines
                    .last()
                    .unwrap()
                    .contains("duplicate_client_trace_key_binding")
            );
            assert!(
                lines
                    .last()
                    .unwrap()
                    .contains("\"client_trace_key\":\"0123456789abcdef0123456789abcdef\"")
            );
        }
    }

    #[test]
    fn pending_drop_order_and_late_key_cannot_rewrite_http_history() {
        let trace = trace();
        let observer = trace.observe_for_test();
        trace.mark_http_pending_drop();
        trace.bind_client_trace_key("0123456789abcdef0123456789abcdef");
        trace.mark_http_body_eof();
        trace.mark_http_body_drop();
        trace.mark_http_pending_drop();
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let lines = observer.json_lines();
        assert!(
            lines
                .last()
                .unwrap()
                .contains("late_client_trace_key_binding")
        );
        assert!(
            lines
                .last()
                .unwrap()
                .contains("\"http_body\":\"pending_dropped\"")
        );
        assert!(lines.last().unwrap().contains("\"client_trace_key\":null"));
        assert!(lines.last().unwrap().contains("\"sequence_valid\":false"));
        let trace = Trace::for_test("/v1/completions");
        let observer = trace.observe_for_test();
        trace.bind_client_trace_key("0123456789abcdef0123456789abcdef");
        trace.mark_http_pending_drop(); // Key alone supplies no minted request/worker authority.
        drop(trace);
        assert!(
            observer
                .json_lines()
                .last()
                .unwrap()
                .contains("\"bindings_complete\":false")
        );
    }
}
