//! Inference liveness (G5) + GPU-fault detection (G24) — lane/serve-hardening, 2026-08-06.
//!
//! THE GAP THIS CLOSES (provider table-stakes audit §3 G5, G24). `/health` used to answer
//! `{"status":"ok"}` off the AXUM task — i.e. it proved the HTTP listener was alive and
//! nothing else. The GPU worker is ONE `std::thread` owning the CUDA context; if it panicked
//! or the card wedged, HTTP kept answering 200 while every request blocked or errored on a
//! dead channel. That is the exact shape OpenRouter's uptime ladder punishes hardest: a green
//! health check in front of a box answering nothing (`<80%` uptime = fallback-only routing).
//!
//! THE MECHANISM: the worker's scheduler loop stamps a monotonic HEARTBEAT every iteration,
//! together with a PHASE (loading / idle / busy / dead). `/health` is then inference liveness:
//!
//!   * `PHASE_IDLE` — the worker is blocked on `rx.recv()` with no work at all. Staleness is
//!     MEANINGLESS here (an idle server legitimately stamps nothing for hours), so idle is
//!     unconditionally healthy. This distinction is load-bearing: a naive "beat age" check
//!     would report every quiet server as dead.
//!   * `PHASE_BUSY`, work is in flight, so FORWARD PROGRESS must advance. See "BUSY IS NOT
//!     HUNG" below: the beat is one of two progress signals, not the only one. No forward
//!     progress past the stall threshold = the worker is wedged (hung kernel, wedged card,
//!     deadlock) -> 503.
//!   * `PHASE_DEAD` / a recorded fault (worker panic, GPU fault) -> 503 immediately, no
//!     threshold wait.
//!
//! BUSY IS NOT HUNG (memra#50, lane/health-busy-vs-hung, 2026-09-03). The heartbeat is stamped
//! ONCE PER SCHEDULER ITERATION, so it measures "a loop pass ended", not "the worker moved".
//! Those come apart the moment ONE iteration legitimately runs longer than the threshold, and
//! they did, measured: on the glm5 ship-gate stress arm (darklanes
//! `research/glm5-serving-launch-20260901/soak-20260901/RESULT.md`, RED finding 1) waves of 8
//! to 22 admitted sessions carrying 20k-88k-token prompts primed inside one iteration, no beat
//! landed for >120 s while the worker was PROGRESSING NORMALLY, `/health` answered 503
//! `unhealthy` for three guard ticks, and the supervisor SIGTERMed a server with 22 requests
//! in flight: every one lost, plus a 40 s model load, for nothing. Raising the threshold (the
//! deployment's own mitigation, 480 s) buys time and trades it for an 8-minute blind spot on a
//! REAL hang. It does not make the signal honest.
//!
//! THE CONTRACT, in words. A BUSY worker is HEALTHY while it can attest FORWARD PROGRESS, and
//! UNHEALTHY when it cannot attest any for `MEMRA_HEALTH_STALL_S`. Two signals attest it, and
//! the verdict takes the FRESHER of the two (`min` of the ages):
//!   1. the scheduler heartbeat, one loop pass completed (`beat`, `beat_busy`);
//!   2. the engine's forward-progress odometer (`memra_engine::progress`), one PRIME CHUNK
//!      completed, stamped where the chunk's logits are already host-side, i.e. where the
//!      device has demonstrably finished that chunk's work.
//! Only a live worker can move either. So "no completion in N seconds" (which a 90 s prefill
//! satisfies while perfectly healthy) stops being the question, and "no forward progress in N
//! seconds" becomes it. `MEMRA_HEALTH_STALL_S` keeps its name, its default and its 120 s
//! derivation below; what changed is that the quantity it bounds is now progress, not silence.
//!
//! WHAT THIS DESIGN CANNOT DETECT, stated so no operator reads more into a 200:
//!   * A worker looping forever INSIDE one chunk (wedged kernel, hung driver call, deadlock in
//!     one prime call) still takes the full threshold to show up. This fix removes false
//!     restarts; it does not speed up true-hang detection.
//!   * A LIVELOCK that keeps completing chunks without ever finishing a request reads healthy.
//!     Liveness is not progress-toward-the-answer; the first-token deadline and admission own
//!     that question, and memra#50's second and third asks are where they get answered.
//!   * Per-SESSION starvation reads healthy: the odometer is process-global by construction,
//!     which is the right granularity for "should this PROCESS be restarted?" and the wrong
//!     one for any per-request SLO.
//!   * A deployment pinning `MEMRA_PRIME_CHUNK=0` (the monolithic rollback seam) gets chunk
//!     granularity of up to 65,520 tokens, so it is back to sizing the threshold by hand.
//!
//! THE STALL THRESHOLD (`MEMRA_HEALTH_STALL_S`, default 120 s) must cover the longest
//! LEGITIMATE single iteration, which is a max-context prefill tick, not a decode step:
//! a naked lone fresh interactive request may prime up to 8192 tokens in one call, while
//! concurrent sessions remain capped at `PREFILL_TICK_T` (1024) tokens apiece and loop over
//! every active session inside ONE iteration. At the interactive cap
//! (`MEMRA_MAX_SESSIONS` = 64) that is 64 x 1024 = 65,536 primed tokens in a single loop pass;
//! at memra's measured 4k-prefill rate on this rig (1.2k tok/s — research/memra-vs-llama-daily-
//! 20260805) that is ~55 s. 120 s is that worst case with ~2.2x margin, and it is the same
//! number `tools/serve-fleet.sh` already uses for `LOAD_GRACE` — so the app threshold, the
//! bash supervisor's grace, and the systemd `StartLimitIntervalSec` sizing in
//! `deploy/systemd/` are ONE number instead of three. `tick_max_ms` (published on /health)
//! is the live receipt: if a real deployment ever observes a legitimate tick near the
//! threshold, that is measured evidence to raise it, not a guess.
//!
//! READINESS vs LIVENESS (k8s deprecated `/healthz` at v1.16 for `/livez` + `/readyz`;
//! KServe/Triton's Open Inference Protocol v2 is the inference-specific form). Split, per
//! that doctrine:
//!   * `/health` == `/livez` — "should this process be RESTARTED?" Draining is NOT a liveness
//!     failure (the process is exiting on its own; restarting it mid-drain kills in-flight
//!     work), so drain keeps answering 200 with `status:"draining"`.
//!   * `/readyz` — "should traffic be ROUTED here?" 503 while loading, draining, faulted, or
//!     stalled. This is ahead of vLLM (whose `/health` 503s only on `EngineDeadError`, with no
//!     readiness endpoint at all) and TGI (single `/health`).
//!
//! G24 — the GPU-fault watcher lives here too because it feeds the SAME unhealthy flag. Its
//! one hard design constraint: the worst Blackwell wedge class (Xid 119/120, GSP RPC timeout)
//! emits no error to the process AND HANGS THE QUERY TOOLS. So the watcher runs on its own
//! thread, spawns `nvidia-smi` as a CHILD with a hard timeout, and treats its own timeout as
//! the alarm — the health handler only ever reads an atomic, so a hung probe can never block
//! health reporting.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Worker phase (an AtomicU8 so the health handler is lock-free).
pub const PHASE_LOADING: u8 = 0;
/// Blocked on `rx.recv()` with zero active sessions and an empty queue — staleness is
/// meaningless in this phase (see the module doc).
pub const PHASE_IDLE: u8 = 1;
/// Inside the scheduler loop with work in flight — the beat MUST advance.
pub const PHASE_BUSY: u8 = 2;
/// The worker thread is gone (panic caught, or `run()` returned).
pub const PHASE_DEAD: u8 = 3;

/// Advisory runtime peer-probe coverage surfaced by `/readyz`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerProbeIntegrity {
    Ok,
    Deferred(u64),
    Degraded,
}

impl PeerProbeIntegrity {
    pub fn detail(self) -> String {
        match self {
            Self::Ok => "ok".into(),
            Self::Deferred(intervals) => format!("deferred_{intervals}"),
            Self::Degraded => "degraded".into(),
        }
    }
}

pub fn phase_name(p: u8) -> &'static str {
    match p {
        PHASE_LOADING => "loading",
        PHASE_IDLE => "idle",
        PHASE_BUSY => "busy",
        _ => "dead",
    }
}

/// Process-start monotonic baseline. `Instant` is not representable as an atomic, so the
/// heartbeat stores milliseconds since this baseline (monotonic, immune to wall-clock steps —
/// an NTP correction must never look like a wedged GPU).
fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64
}

/// MEMRA_HEALTH_STALL_S (default 120): how long a BUSY worker may go without stamping a beat
/// before `/health` reports unhealthy. See the module doc for the derivation.
///
/// Read once at construction into `WorkerHealth::stall_ms` rather than consulted per call: the
/// verdict must not depend on the environment changing under a running process, and a
/// per-instance value is what makes the stall path testable in bounded time (a
/// process-global OnceLock would force a 120 s sleep to observe a stall at all — i.e. an
/// untested branch, which for the branch that decides "restart this box" is not acceptable).
pub fn stall_threshold_ms() -> u64 {
    std::env::var("MEMRA_HEALTH_STALL_S")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120)
        .max(1)
        * 1000
}

/// What a forward-progress source reports. `age_ms` is milliseconds since the last advance;
/// `rows`/`events` are the odometer's counts, published for operators and never gated on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardProgress {
    pub rows: u64,
    pub events: u64,
    pub age_ms: u64,
}

/// A forward-progress source: `None` means "this process has nothing to attest yet" (a worker
/// that has never primed anything), in which case the verdict falls back to beat age alone.
///
/// It is a stored closure rather than a direct call into `memra_engine::progress` so the stall
/// branch stays deterministically testable on the host: the odometer is a process-global, and
/// a test binary running branches in parallel cannot own one. The production source IS that
/// global; tests inject a local one. Same reasoning as `stall_ms` above, the branch that
/// decides "restart this box" does not get to be the untested one.
pub type ProgressSource = Arc<dyn Fn() -> Option<ForwardProgress> + Send + Sync>;

/// MEMRA_HEALTH_PROGRESS (default 1 = ON): feed the engine's forward-progress odometer into the
/// BUSY stall verdict. `=0` is the rollback seam, pure beat-age semantics, the pre-memra#50
/// behaviour, byte-identical.
///
/// DEFAULT ON, deliberately, with the reason stated (the flags law). OFF is the arm with the
/// measured defect: it declares a busy, progressing worker unhealthy and gets a server with 22
/// requests in flight SIGTERMed (darklanes RESULT.md RED finding 1). ON can only ever make a
/// verdict LESS eager to restart, it takes the fresher of two liveness signals, so the risk
/// it carries is a delayed hang detection, and it carries none: a hung worker advances neither
/// signal, so its detection time is unchanged at `MEMRA_HEALTH_STALL_S`. A default that ships
/// OFF here would mean shipping the known-broken arm to every deployment that does not read
/// FLAGS.md.
fn progress_signal_enabled() -> bool {
    std::env::var("MEMRA_HEALTH_PROGRESS").as_deref() != Ok("0")
}

/// The production forward-progress source: the engine's prime odometer.
fn engine_progress_source() -> ProgressSource {
    Arc::new(|| {
        memra_engine::progress::snapshot().map(|p| ForwardProgress {
            rows: p.rows,
            events: p.events,
            age_ms: p.age_ms,
        })
    })
}

/// Inference-liveness state shared by the worker thread, the GPU watcher, and the HTTP layer.
///
/// Every field the health decision depends on is an ATOMIC: the handler must be able to
/// answer while the worker is wedged, while a probe subprocess is hung, and while another
/// thread holds the reason strings. The `Mutex<String>` reasons are read with `try_lock` and
/// are cosmetic — never load-bearing for the verdict.
pub struct WorkerHealth {
    beat_ms: AtomicU64,
    phase: AtomicU8,
    /// bumped on every (re)spawn of the worker thread — a client can tell a respawn happened.
    generation: AtomicU32,
    /// longest observed scheduler-loop iteration (ms) — the live receipt for the threshold.
    tick_max_ms: AtomicU64,
    /// worker fault latch (panic / init failure). Set once, cleared only by a successful
    /// respawn. The verdict reads THIS, never the reason string.
    faulted: AtomicBool,
    fault_reason: Mutex<String>,
    /// GPU fault latch (Xid class / probe hang) — G24. Separate from `faulted` so /health can
    /// report which half failed, and so a card fault survives a worker respawn.
    gpu_faulted: AtomicBool,
    gpu_reason: Mutex<String>,
    /// non-fatal Xid lines seen (13/31 app errors, 43/45 teardown, 62/63 remap pending) —
    /// counted, not fatal, so an operator can see a card degrading before it wedges.
    xid_warns: AtomicU64,
    /// Consecutive copy-count intervals for which a due peer-integrity probe was deferred by a
    /// live speculative session. This is advisory until `peer_probe_integrity_degraded` latches.
    peer_probe_deferred_intervals: AtomicU64,
    /// Once the configured deferral bound is reached, new speculative admissions are held on the
    /// plain path until a completed probe clears this latch. Existing sessions are untouched.
    peer_probe_integrity_degraded: AtomicBool,
    /// the stall bound this instance enforces, resolved once at construction.
    stall_ms: u64,
    /// forward-progress source consulted alongside the beat (memra#50). `None` = the
    /// `MEMRA_HEALTH_PROGRESS=0` rollback seam: pure beat-age semantics.
    progress: Option<ProgressSource>,
}

pub type SharedHealth = Arc<WorkerHealth>;

impl Default for WorkerHealth {
    fn default() -> Self {
        WorkerHealth {
            beat_ms: AtomicU64::new(now_ms()),
            phase: AtomicU8::new(PHASE_LOADING),
            generation: AtomicU32::new(0),
            tick_max_ms: AtomicU64::new(0),
            faulted: AtomicBool::new(false),
            fault_reason: Mutex::new(String::new()),
            gpu_faulted: AtomicBool::new(false),
            gpu_reason: Mutex::new(String::new()),
            xid_warns: AtomicU64::new(0),
            peer_probe_deferred_intervals: AtomicU64::new(0),
            peer_probe_integrity_degraded: AtomicBool::new(false),
            stall_ms: stall_threshold_ms(),
            progress: progress_signal_enabled().then(engine_progress_source),
        }
    }
}

impl WorkerHealth {
    pub fn new() -> SharedHealth {
        Arc::new(Self::default())
    }

    /// Same, with an explicit stall bound — tests only, so the stall branch is exercised in
    /// milliseconds instead of the 120 s the production default would require.
    #[cfg(test)]
    pub(crate) fn with_stall_ms(stall_ms: u64) -> SharedHealth {
        Arc::new(Self {
            stall_ms,
            // No injected source: the pre-memra#50 beat-age-only arm, which is what the
            // existing stall tests were written against and must keep asserting.
            progress: None,
            ..Default::default()
        })
    }

    /// Same, with an injected forward-progress source, the memra#50 arm. Tests only: the
    /// production source is a process-global odometer, and a red arm for "a busy worker that
    /// is PROGRESSING reads healthy" has to drive progress deterministically.
    #[cfg(test)]
    pub(crate) fn with_stall_and_progress(stall_ms: u64, progress: ProgressSource) -> SharedHealth {
        Arc::new(Self {
            stall_ms,
            progress: Some(progress),
            ..Default::default()
        })
    }

    // ---- worker-side stamps ----

    /// One scheduler-loop iteration completed. Also records the iteration's own duration so
    /// `tick_max_ms` can justify (or refute) the stall threshold from live traffic.
    pub fn beat(&self) {
        let t = now_ms();
        let prev = self.beat_ms.swap(t, Ordering::Release);
        let dt = t.saturating_sub(prev);
        // relaxed max: a lost race only ever under-reports a single tick, and this is a
        // diagnostic, not a gate input.
        if dt > self.tick_max_ms.load(Ordering::Relaxed) {
            self.tick_max_ms.store(dt, Ordering::Relaxed);
        }
    }

    /// Entering / leaving the blocking `rx.recv()` (no work at all), or starting a weight
    /// load. Stamps the beat too, so it is already fresh the moment work arrives — a session
    /// admitted after an hour idle must not look one hour stale for one tick.
    pub fn set_phase(&self, phase: u8) {
        self.beat_ms.store(now_ms(), Ordering::Release);
        self.phase.store(phase, Ordering::Release);
    }

    /// Top of a scheduler-loop iteration with work in flight: BUSY + one beat. The phase
    /// store must NOT reset the beat here (that is why this is not `set_phase` + `beat`) —
    /// `beat()` measures the iteration that just ended, and a stamp in between would report
    /// every tick as 0 ms and make `tick_max_ms` useless as threshold evidence.
    pub fn beat_busy(&self) {
        self.phase.store(PHASE_BUSY, Ordering::Release);
        self.beat();
    }

    /// Worker is up and serving (called once every model is resident and the loop is entered,
    /// and again after a successful respawn — which CLEARS the worker fault latch).
    pub fn mark_ready(&self) {
        if let Ok(mut r) = self.fault_reason.lock() {
            r.clear();
        }
        self.faulted.store(false, Ordering::Release);
        self.set_phase(PHASE_IDLE);
    }

    /// The worker died (panic caught, init failed, or `run()` returned). LATCHES unhealthy:
    /// `/health` and `/readyz` flip within milliseconds, never after a threshold wait.
    pub fn mark_dead(&self, reason: impl Into<String>) {
        let reason = reason.into();
        if let Ok(mut r) = self.fault_reason.lock() {
            *r = reason;
        }
        self.faulted.store(true, Ordering::Release);
        self.phase.store(PHASE_DEAD, Ordering::Release);
    }

    /// A respawn attempt is loading weights: not dead, not ready — `/readyz` 503s, `/health`
    /// stays down until the load lands (the process IS currently answering nothing).
    pub fn mark_respawning(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        self.set_phase(PHASE_LOADING);
    }

    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }

    // ---- runtime peer-probe coverage --------------------------------------------------

    /// Publish a newly observed deferred interval. The worker calls this only when a real
    /// cross-device runtime probe reports `Deferred`; single-device serving never touches it.
    pub fn note_peer_probe_deferral(&self, consecutive_intervals: u64, degraded: bool) {
        self.peer_probe_deferred_intervals
            .store(consecutive_intervals, Ordering::Release);
        if degraded {
            self.peer_probe_integrity_degraded
                .store(true, Ordering::Release);
        }
    }

    /// A completed native probe or validated host-bounce promotion restores coverage.
    pub fn clear_peer_probe_deferral(&self) {
        // Publish the count first: a reader that observes the cleared latch also observes zero.
        self.peer_probe_deferred_intervals
            .store(0, Ordering::Release);
        self.peer_probe_integrity_degraded
            .store(false, Ordering::Release);
    }

    pub fn peer_probe_integrity(&self) -> PeerProbeIntegrity {
        if self.peer_probe_integrity_degraded.load(Ordering::Acquire) {
            PeerProbeIntegrity::Degraded
        } else {
            match self.peer_probe_deferred_intervals.load(Ordering::Acquire) {
                0 => PeerProbeIntegrity::Ok,
                intervals => PeerProbeIntegrity::Deferred(intervals),
            }
        }
    }

    /// False only after the bound has latched and until a completed probe clears it. This gates
    /// speculative session construction, never request admission or an already-live session.
    pub fn peer_probe_allows_spec_admission(&self) -> bool {
        !self.peer_probe_integrity_degraded.load(Ordering::Acquire)
    }

    // ---- GPU watcher side (G24) ----

    /// A fatal GPU fault: an Xid line in the fatal classes, or a probe that HUNG (the
    /// GSP-timeout class raises no Xid and hangs the query tools, so the probe's own timeout
    /// IS the alarm). Latched: a card that wedged does not un-wedge itself.
    pub fn mark_gpu_fault(&self, reason: impl Into<String>) {
        let reason = reason.into();
        eprintln!("[gpu-watch] CRITICAL: {reason}");
        if let Ok(mut r) = self.gpu_reason.lock()
            && r.is_empty()
        {
            *r = reason;
        }
        self.gpu_faulted.store(true, Ordering::Release);
    }

    pub fn note_xid_warn(&self, line: &str) {
        self.xid_warns.fetch_add(1, Ordering::Relaxed);
        eprintln!("[gpu-watch] WARN non-fatal Xid: {line}");
    }

    // ---- verdicts (lock-free) ----

    fn beat_age_ms(&self) -> u64 {
        now_ms().saturating_sub(self.beat_ms.load(Ordering::Acquire))
    }

    /// Milliseconds since this process last attested FORWARD PROGRESS, the fresher of the
    /// scheduler heartbeat and the engine's prime odometer (memra#50; see the module doc's
    /// "BUSY IS NOT HUNG"). Lock-free: two relaxed loads, one acquire load and one
    /// `Instant::now()`, so the verdict can still be computed while the worker is wedged.
    ///
    /// The odometer can only ever make this SMALLER, so this cannot turn a healthy verdict
    /// into an unhealthy one, the change is strictly in the direction of not restarting a
    /// server that is working.
    fn forward_progress_age_ms(&self) -> u64 {
        let beat = self.beat_age_ms();
        match self.progress.as_ref().and_then(|p| p()) {
            Some(p) => beat.min(p.age_ms),
            None => beat,
        }
    }

    /// Is the worker stalled? Only meaningful while BUSY — an idle worker legitimately stamps
    /// nothing, and a loading worker is covered by readiness, not liveness.
    ///
    /// BUSY-but-progressing is NOT stalled: the quantity bounded here is time without forward
    /// progress, not time without a loop pass. That distinction is the whole of memra#50.
    /// `Some(age)` when BUSY and that age exceeds the bound. Returns the age it judged rather
    /// than a bare bool so `live()` reports the number that PRODUCED the verdict: recomputing
    /// it for the message would print a second, later sample.
    fn stalled_for_ms(&self) -> Option<u64> {
        if self.phase.load(Ordering::Acquire) != PHASE_BUSY {
            return None;
        }
        let age = self.forward_progress_age_ms();
        (age > self.stall_ms).then_some(age)
    }

    /// LIVENESS (`/health`, `/livez`): should this process be restarted? Draining is NOT a
    /// liveness failure — the caller passes its own drain flag and we deliberately ignore it
    /// here (see the module doc).
    pub fn live(&self) -> Result<(), String> {
        if self.gpu_faulted.load(Ordering::Acquire) {
            return Err(self
                .gpu_reason
                .try_lock()
                .map(|r| r.clone())
                .unwrap_or_else(|_| "gpu fault".into()));
        }
        if self.faulted.load(Ordering::Acquire) {
            return Err(self
                .fault_reason
                .try_lock()
                .map(|r| r.clone())
                .unwrap_or_else(|_| "worker fault".into()));
        }
        match self.phase.load(Ordering::Acquire) {
            PHASE_DEAD => Err("worker thread is gone".into()),
            PHASE_LOADING => Err("worker is (re)loading weights".into()),
            _ => match self.stalled_for_ms() {
                Some(age) => Err(format!(
                    "worker stalled: no forward progress for {age} ms (beat age {} ms, \
                     threshold {} ms)",
                    self.beat_age_ms(),
                    self.stall_ms
                )),
                None => Ok(()),
            },
        }
    }

    /// READINESS (`/readyz`): should traffic be routed here? Everything liveness rejects, plus
    /// draining (a draining instance is alive but must not receive new work).
    pub fn ready(&self, draining: bool) -> Result<(), String> {
        if draining {
            return Err("draining (shutdown in progress)".into());
        }
        self.live()
    }

    /// The observable state — /health, /readyz, and /metrics all render from this.
    pub fn snapshot(&self) -> HealthSnapshot {
        HealthSnapshot {
            phase: self.phase.load(Ordering::Acquire),
            beat_age_ms: self.beat_age_ms(),
            tick_max_ms: self.tick_max_ms.load(Ordering::Relaxed),
            generation: self.generation(),
            xid_warns: self.xid_warns.load(Ordering::Relaxed),
            stall_threshold_ms: self.stall_ms,
            forward_progress_age_ms: self.forward_progress_age_ms(),
            progress: self.progress.as_ref().and_then(|p| p()),
        }
    }
}

/// Numbers `/health` publishes so an operator can see WHY, not just that.
pub struct HealthSnapshot {
    pub phase: u8,
    pub beat_age_ms: u64,
    pub tick_max_ms: u64,
    pub generation: u32,
    pub xid_warns: u64,
    pub stall_threshold_ms: u64,
    /// The quantity the stall verdict actually bounds (memra#50): the fresher of the beat age
    /// and the prime odometer's age. Equal to `beat_age_ms` on the
    /// `MEMRA_HEALTH_PROGRESS=0` seam and before the first chunk of the process.
    pub forward_progress_age_ms: u64,
    /// The engine odometer itself, `None` until this process completes its first prime chunk
    /// (or when the progress signal is off). Published so an operator watching a long prefill
    /// can see WHICH signal is holding the server healthy.
    pub progress: Option<ForwardProgress>,
}

// ---------------------------------------------------------------------------
// G24 — GPU-fault watcher
// ---------------------------------------------------------------------------

/// MEMRA_GPU_WATCH (default 1): the Xid / probe-hang watcher. `=0` disables.
fn gpu_watch_enabled() -> bool {
    std::env::var("MEMRA_GPU_WATCH").as_deref() != Ok("0")
}

/// MEMRA_GPU_WATCH_S (default 60): probe interval. The audit's published detection
/// commitment is "checks every 60 s" — a fact about instrumentation, not about a human.
fn gpu_watch_interval_s() -> u64 {
    std::env::var("MEMRA_GPU_WATCH_S")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(60)
        .max(1)
}

/// MEMRA_GPU_PROBE_TIMEOUT_S (default 10): the probe's own deadline. THIS IS THE ALARM for
/// the GSP-hang class (Xid 119/120), which emits no Xid and hangs `nvidia-smi` itself.
fn gpu_probe_timeout_s() -> u64 {
    std::env::var("MEMRA_GPU_PROBE_TIMEOUT_S")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10)
        .max(1)
}

/// Xid classes that mean the CARD is unusable — the process must stop taking traffic and let
/// the supervisor restart it (CUDA errors are sticky per process; in-process recovery is not
/// a thing). Sourced from the audit's verified list:
///   48  double-bit ECC error
///   64  row-remap FAILURE ("XID 64 occurred" is also DCGM field 395)
///   79  "GPU has fallen off the bus" (node-fatal)
///   94  contained ECC error
///   95  uncontained ECC error
///   119 / 120  GSP RPC timeout — the Blackwell driver-hang class
const XID_FATAL: &[u32] = &[48, 64, 79, 94, 95, 119, 120];

/// Classify one kernel line. Returns Some((xid, fatal)) for an `NVRM: Xid` line.
pub fn classify_xid(line: &str) -> Option<(u32, bool)> {
    let i = line.find("Xid")?;
    // forms: "NVRM: Xid (PCI:0000:01:00): 119, pid=..." and "NVRM: Xid 119, ..."
    let tail = &line[i + 3..];
    let after = match tail.find(':') {
        // "(PCI:...): 119," — the id follows the LAST colon of the bracketed prefix
        Some(_) if tail.trim_start().starts_with('(') => {
            let close = tail.find(')')?;
            let rest = &tail[close + 1..];
            rest.trim_start().trim_start_matches(':')
        }
        _ => tail,
    };
    let digits: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let xid: u32 = digits.parse().ok()?;
    Some((xid, XID_FATAL.contains(&xid)))
}

/// Run `nvidia-smi` with a HARD deadline, natively (no `timeout(1)` dependency). Returns
/// `Ok(stdout)`, `Err(Hang)` when the deadline passed (the child is killed), or `Err(Spawn)`
/// when the tool is not installed / not executable.
enum ProbeErr {
    Hang,
    Spawn(String),
    Exit(String),
}

fn probe_smi(args: &[&str], deadline: Duration) -> Result<String, ProbeErr> {
    use std::process::{Command, Stdio};
    let mut child = Command::new("nvidia-smi")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ProbeErr::Spawn(e.to_string()))?;
    let t0 = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = child
                    .wait_with_output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                    .unwrap_or_default();
                if status.success() {
                    return Ok(out);
                }
                return Err(ProbeErr::Exit(format!("exit {status}")));
            }
            Ok(None) => {
                if t0.elapsed() >= deadline {
                    // THE ALARM. Kill it so the watcher does not leak a hung child per tick.
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(ProbeErr::Hang);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(ProbeErr::Spawn(e.to_string())),
        }
    }
}

/// Spawn the G24 watcher: kernel-log Xid tail + a timeout-bounded `nvidia-smi` canary, both
/// feeding `WorkerHealth`'s gpu-fault latch. Never blocks the caller; never blocks health.
pub fn spawn_gpu_watch(health: SharedHealth) {
    if !gpu_watch_enabled() {
        eprintln!("[gpu-watch] disabled (MEMRA_GPU_WATCH=0)");
        return;
    }
    spawn_xid_tail(health.clone());
    let interval = Duration::from_secs(gpu_watch_interval_s());
    let deadline = Duration::from_secs(gpu_probe_timeout_s());
    let _ = std::thread::Builder::new()
        .name("memra-gpu-watch".into())
        .spawn(move || {
            // First probe decides whether the canary is available at all. "tool missing" is NOT a
            // fault (CI, containers without the driver toolkit) — the watcher disables that half
            // and says so once, rather than reporting a phantom CRITICAL.
            //
            // FIELD DEGRADATION (measured on this rig, driver 595.84 / RTX 5090 Laptop):
            // `xid.pending` is "not a valid field to query" and every ECC/retired-page field
            // answers `[N/A]` on a consumer part. So the canary asks for the fields when they
            // exist and falls back to a MINIMAL query whose only job is "does the driver answer
            // at all" — which is exactly the signal for the wedge class that matters, since
            // Xid 119/120 hangs the tool regardless of which fields were requested.
            const RICH: &[&str] = &[
                "--query-gpu=timestamp,ecc.errors.uncorrected.volatile.total,\
                                 retired_pages.pending,remapped_rows.failure",
                "--format=csv,noheader",
            ];
            const MIN: &[&str] = &["--query-gpu=timestamp,memory.used", "--format=csv,noheader"];
            let mut args: &[&str] = RICH;
            match probe_smi(RICH, deadline) {
                Ok(_) => {}
                Err(ProbeErr::Hang) => {
                    health.mark_gpu_fault(format!(
                        "nvidia-smi did not answer within {}s at startup — GPU/driver wedge \
                     (the GSP-timeout class raises no Xid and hangs the query tools, so this \
                     timeout IS the fault)",
                        deadline.as_secs()
                    ));
                }
                Err(ProbeErr::Spawn(e)) => {
                    eprintln!(
                        "[gpu-watch] nvidia-smi canary unavailable ({e}); Xid log watch only"
                    );
                    loop_xid_only();
                    return;
                }
                Err(ProbeErr::Exit(_)) => {
                    // fields unsupported on this part -> degrade to the liveness-only query
                    args = MIN;
                    eprintln!(
                        "[gpu-watch] rich ECC/Xid fields unsupported on this GPU; \
                           canary degraded to a driver-liveness query \
                           (the probe's own timeout stays the alarm)"
                    );
                }
            }
            eprintln!(
                "[gpu-watch] on: every {}s, probe deadline {}s, fatal Xid {:?}",
                interval.as_secs(),
                deadline.as_secs(),
                XID_FATAL
            );
            loop {
                std::thread::sleep(interval);
                match probe_smi(args, deadline) {
                    Ok(out) => {
                        // Non-zero uncorrected ECC / a failed row remap is a hardware fault even
                        // without an Xid line reaching us (dmesg may be restricted — it is on
                        // this rig: kernel.dmesg_restrict=1).
                        if let Some(reason) = scan_smi_csv(&out) {
                            health.mark_gpu_fault(reason);
                        }
                    }
                    Err(ProbeErr::Hang) => health.mark_gpu_fault(format!(
                        "nvidia-smi did not answer within {}s — GPU/driver wedge",
                        deadline.as_secs()
                    )),
                    Err(ProbeErr::Spawn(e)) => {
                        eprintln!("[gpu-watch] canary spawn failed ({e}); continuing on Xid only");
                    }
                    Err(ProbeErr::Exit(e)) => {
                        eprintln!(
                            "[gpu-watch] canary exited nonzero ({e}) — not treated as a \
                               fault (a failing QUERY is not a failing card; the hang is)"
                        );
                    }
                }
            }
        });
}

fn loop_xid_only() {
    // The Xid tail runs on its own thread; nothing left to do here.
}

/// Parse the canary CSV for hardware-fault counters. Only DEFINITE values fault: `[N/A]`
/// (unsupported on consumer parts) and non-numeric fields are ignored, never guessed.
fn scan_smi_csv(out: &str) -> Option<String> {
    let line = out.lines().next()?;
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    // RICH order: timestamp, ecc uncorrected volatile total, retired_pages.pending,
    // remapped_rows.failure
    if fields.len() >= 4 {
        if let Ok(ecc) = fields[1].parse::<u64>()
            && ecc > 0
        {
            return Some(format!(
                "uncorrected volatile ECC errors = {ecc} (nvidia-smi)"
            ));
        }
        if fields[3].eq_ignore_ascii_case("yes") || fields[3] == "1" {
            return Some("row-remap FAILURE reported by nvidia-smi (Xid 64 class)".into());
        }
    }
    None
}

/// Tail the kernel log for `NVRM: Xid` lines. Two sources, in preference order:
///   1. `/dev/kmsg` — read directly, no subprocess. Needs CAP_SYSLOG or
///      `kernel.dmesg_restrict=0`; on this rig it is root-only, so it is TRIED and skipped.
///   2. `journalctl -k -f` — works unprivileged here (measured), and is what the systemd
///      deployment has anyway.
///      A watcher that cannot read either source says so ONCE and exits: the `nvidia-smi` canary
///      still covers the hang class, and a silent no-op watcher would be worse than an absent one.
fn spawn_xid_tail(health: SharedHealth) {
    let _ = std::thread::Builder::new()
        .name("memra-xid-tail".into())
        .spawn(move || {
            use std::io::{BufRead, BufReader};
            if let Ok(f) = std::fs::File::open("/dev/kmsg") {
                eprintln!("[gpu-watch] Xid source: /dev/kmsg");
                let mut rd = BufReader::new(f);
                let mut line = String::new();
                loop {
                    line.clear();
                    match rd.read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => handle_xid_line(&health, line.trim_end()),
                        Err(_) => break,
                    }
                }
                return;
            }
            let child = std::process::Command::new("journalctl")
                .args(["-k", "-n", "0", "-f", "--no-pager"])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn();
            match child {
                Ok(mut c) => {
                    eprintln!(
                        "[gpu-watch] Xid source: journalctl -k -f \
                           (/dev/kmsg unreadable — kernel.dmesg_restrict)"
                    );
                    if let Some(out) = c.stdout.take() {
                        for line in BufReader::new(out).lines().map_while(Result::ok) {
                            handle_xid_line(&health, &line);
                        }
                    }
                    let _ = c.wait();
                }
                Err(e) => {
                    eprintln!(
                        "[gpu-watch] no Xid log source (/dev/kmsg unreadable, \
                           journalctl unavailable: {e}) — nvidia-smi canary only"
                    );
                }
            }
        });
}

fn handle_xid_line(health: &SharedHealth, line: &str) {
    if !line.contains("Xid") {
        return;
    }
    match classify_xid(line) {
        Some((xid, true)) => {
            health.mark_gpu_fault(format!("NVRM Xid {xid} (fatal class) — {}", line.trim()))
        }
        Some((_, false)) => health.note_xid_warn(line.trim()),
        None => {}
    }
}

// ---------------------------------------------------------------------------
// systemd integration (sd_notify) — the supervision half of G5
// ---------------------------------------------------------------------------
//
// `Type=notify` + `WatchdogSec=` is the wedged-GPU answer at the SUPERVISION layer: the
// heartbeat that feeds /health also feeds systemd, so a hung card stops WATCHDOG=1 and
// systemd restarts the unit whole. Implemented with std only (a `UnixDatagram` to
// `$NOTIFY_SOCKET`), and a complete no-op when the env var is absent — running under bash
// supervision or a bare shell costs nothing and logs nothing.
//
// LIMITATION, stated rather than hidden: systemd may hand us an ABSTRACT socket (a path
// starting with '@'), which std cannot address without nightly APIs. System units get the
// `/run/systemd/notify` path form, which is the deployment this repo ships; an abstract
// socket disables the notifier with one warning line instead of pretending to work.

fn notify_socket() -> Option<&'static str> {
    static S: OnceLock<Option<String>> = OnceLock::new();
    S.get_or_init(|| {
        let v = std::env::var("NOTIFY_SOCKET").ok()?;
        if v.starts_with('@') {
            eprintln!(
                "[sd-notify] abstract socket {v:?} is not addressable from std — \
                       notifier disabled (use a path-form NOTIFY_SOCKET, i.e. a system unit)"
            );
            return None;
        }
        Some(v)
    })
    .as_deref()
}

/// Send one sd_notify datagram. Best effort by design: supervision must never be able to
/// fail a request path.
pub fn sd_notify(msg: &str) {
    let Some(path) = notify_socket() else { return };
    if let Ok(sock) = std::os::unix::net::UnixDatagram::unbound() {
        let _ = sock.send_to(msg.as_bytes(), path);
    }
}

/// Spawn the systemd watchdog pinger: `WATCHDOG=1` only while the worker is LIVE, at half the
/// interval systemd gave us (`WATCHDOG_USEC`). A wedged worker simply stops being live, the
/// pings stop, and `Restart=` fires — which is the honest outcome, because CUDA errors are
/// sticky per process and a fresh process is the only reliable recovery.
pub fn spawn_sd_watchdog(health: SharedHealth) {
    if notify_socket().is_none() {
        return;
    }
    let usec: u64 = match std::env::var("WATCHDOG_USEC")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(v) if v > 0 => v,
        _ => return, // WatchdogSec= not configured; READY=1/STOPPING=1 still work
    };
    let every = Duration::from_micros(usec / 2);
    eprintln!(
        "[sd-notify] watchdog armed: WATCHDOG_USEC={usec}, pinging every {:.1}s while live",
        every.as_secs_f64()
    );
    let _ = std::thread::Builder::new()
        .name("memra-sd-watchdog".into())
        .spawn(move || {
            loop {
                std::thread::sleep(every);
                match health.live() {
                    Ok(()) => sd_notify("WATCHDOG=1"),
                    Err(why) => eprintln!("[sd-notify] withholding WATCHDOG=1: {why}"),
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xid_classification_covers_both_kernel_line_forms_and_the_fatal_set() {
        // bracketed PCI form (the common one)
        assert_eq!(
            classify_xid("NVRM: Xid (PCI:0000:01:00): 119, pid=1234, GSP RPC timeout"),
            Some((119, true))
        );
        // bare form
        assert_eq!(
            classify_xid("NVRM: Xid 79, GPU has fallen off the bus"),
            Some((79, true))
        );
        // journald prefix in front
        assert_eq!(
            classify_xid(
                "Aug 06 20:00:00 box kernel: NVRM: Xid (PCI:0000:01:00): 48, \
                                 Double-bit ECC"
            ),
            Some((48, true))
        );
        // non-fatal classes are recognized but not fatal
        assert_eq!(
            classify_xid("NVRM: Xid (PCI:0000:01:00): 13, Graphics SM Warp Exception"),
            Some((13, false))
        );
        assert_eq!(
            classify_xid("NVRM: Xid (PCI:0000:01:00): 43, GPU stopped processing"),
            Some((43, false))
        );
        assert_eq!(
            classify_xid("NVRM: Xid (PCI:0000:01:00): 63, Row remap pending"),
            Some((63, false))
        );
        // every audit-listed fatal id classifies fatal
        for id in [48u32, 64, 79, 94, 95, 119, 120] {
            let line = format!("NVRM: Xid (PCI:0000:c1:00): {id}, something");
            assert_eq!(
                classify_xid(&line),
                Some((id, true)),
                "xid {id} must be fatal"
            );
        }
        // not an Xid line
        assert_eq!(
            classify_xid("NVRM: GPU at PCI:0000:01:00 has been initialized"),
            None
        );
    }

    /// A test progress source over a local counter, so the memra#50 arms are deterministic
    /// and do not touch the process-global engine odometer.
    fn test_progress() -> (Arc<AtomicU64>, ProgressSource) {
        // Stores the epoch-ms of the last simulated advance; `u64::MAX` = never advanced.
        let last = Arc::new(AtomicU64::new(u64::MAX));
        let seen = last.clone();
        let src: ProgressSource = Arc::new(move || {
            let v = seen.load(Ordering::Acquire);
            (v != u64::MAX).then(|| ForwardProgress {
                rows: 4096,
                events: 1,
                age_ms: now_ms().saturating_sub(v),
            })
        });
        (last, src)
    }

    /// RED ARM 1 (memra#50), direction "busy must not read hung". A worker that has NOT
    /// stamped a scheduler beat for longer than the stall bound, but IS completing prime
    /// chunks, is HEALTHY. This is the glm5 stress shape in miniature: one scheduler
    /// iteration priming 20k-88k-token prompts for 8-22 sessions, no loop pass ending, the
    /// worker progressing the whole time.
    ///
    /// On the pre-memra#50 code this test FAILS: `stalled()` read beat age alone, so this
    /// asserted `live().is_ok()` on a verdict that was `Err("worker stalled...")`, the exact
    /// 503 that got a server with 22 requests in flight SIGTERMed.
    #[test]
    fn a_busy_worker_that_is_still_priming_chunks_is_live_however_stale_the_beat() {
        let (advance, src) = test_progress();
        // 200 ms bound against 20 ms steps: a 10x margin, so a loaded host cannot turn this
        // into a flake. The bound still has to be REACHED, which the stale beat below does.
        let h = WorkerHealth::with_stall_and_progress(200, src);
        h.mark_ready();
        h.beat_busy();
        // The beat goes stale and STAYS stale: nothing calls beat() again for the rest of the
        // test, exactly as a worker inside one long prefill cannot.
        std::thread::sleep(Duration::from_millis(400));
        assert!(
            h.beat_age_ms() > 200,
            "the beat must be genuinely stale or this asserts nothing"
        );
        // ... but chunks keep landing, each well inside the bound.
        for _ in 0..5 {
            advance.store(now_ms(), Ordering::Release);
            std::thread::sleep(Duration::from_millis(20));
            assert!(
                h.live().is_ok(),
                "a BUSY worker completing prime chunks is progressing, not hung: {:?}",
                h.live()
            );
            assert_eq!(h.snapshot().phase, PHASE_BUSY);
        }
        // And the snapshot says WHICH signal held it healthy, so an operator can see it.
        let snap = h.snapshot();
        assert!(snap.beat_age_ms > 200);
        assert!(
            snap.forward_progress_age_ms <= 200,
            "forward progress age {} must be the fresh number, not the beat age {}",
            snap.forward_progress_age_ms,
            snap.beat_age_ms
        );
        assert!(snap.progress.is_some(), "the odometer must be published");
    }

    /// RED ARM 2 (memra#50), the other direction: the fix must not blind the check. A worker
    /// whose beat AND odometer both freeze is UNHEALTHY within the bound, a genuine hang
    /// (wedged kernel, deadlock inside one chunk) advances neither signal.
    ///
    /// This is the arm that would fail if someone "fixed" memra#50 by making BUSY
    /// unconditionally healthy, or by beating from a thread that is not the worker.
    #[test]
    fn a_worker_whose_progress_also_froze_is_unhealthy_within_the_bound() {
        let (advance, src) = test_progress();
        let h = WorkerHealth::with_stall_and_progress(20, src);
        h.mark_ready();
        h.beat_busy();
        advance.store(now_ms(), Ordering::Release);
        assert!(h.live().is_ok(), "fresh on both signals");
        // Both signals freeze. Nothing else changes.
        std::thread::sleep(Duration::from_millis(60));
        let why = h.live().expect_err("a frozen worker must be unhealthy");
        assert!(
            why.contains("no forward progress"),
            "the reason must name the quantity actually bounded: {why}"
        );
        assert!(h.ready(false).is_err(), "and it must not be routed traffic");
    }

    /// The odometer can only ever make the verdict LESS eager to restart. A source that
    /// reports a STALE progress age must not rescue a stale beat, and must not make a FRESH
    /// beat look stalled either, `min` of the two ages, never `max`, never a replacement.
    #[test]
    fn a_stale_odometer_neither_rescues_nor_condemns() {
        let src: ProgressSource = Arc::new(|| {
            Some(ForwardProgress {
                rows: 1,
                events: 1,
                age_ms: 10_000,
            })
        });
        let h = WorkerHealth::with_stall_and_progress(20, src);
        h.mark_ready();
        h.beat_busy();
        // fresh beat, ancient odometer -> live (the beat is the fresher signal).
        assert!(h.live().is_ok(), "a fresh beat is progress on its own");
        std::thread::sleep(Duration::from_millis(40));
        // stale beat, ancient odometer -> unhealthy.
        assert!(
            h.live().is_err(),
            "neither signal is fresh: this IS a stall"
        );
    }

    /// The `MEMRA_HEALTH_PROGRESS=0` rollback seam: with no source, the verdict is
    /// byte-identical to the pre-memra#50 beat-age semantics. `with_stall_ms` is that arm,
    /// and `idle_is_healthy_at_any_age_but_busy_stalls` below is its assertion; this one
    /// pins that a `None` source is what produces it, so the seam cannot rot silently.
    #[test]
    fn no_progress_source_is_the_pre_fix_beat_age_verdict() {
        let h = WorkerHealth::with_stall_ms(20);
        h.mark_ready();
        h.beat_busy();
        std::thread::sleep(Duration::from_millis(40));
        assert!(
            h.live().is_err(),
            "with the seam pulled, a stale beat alone stalls the worker again"
        );
        let snap = h.snapshot();
        assert_eq!(
            snap.forward_progress_age_ms, snap.beat_age_ms,
            "with no source the published progress age IS the beat age"
        );
        assert!(snap.progress.is_none());
    }

    #[test]
    fn idle_is_healthy_at_any_age_but_busy_stalls() {
        // 20 ms bound so the stall branch is REACHED (not just described): at the production
        // 120 s this test would either sleep two minutes or assert nothing.
        let h = WorkerHealth::with_stall_ms(20);
        h.mark_ready();
        assert!(h.live().is_ok(), "a ready idle worker is live");
        // an old beat while IDLE: still live (an idle server stamps nothing for hours).
        h.phase.store(PHASE_IDLE, Ordering::Release);
        std::thread::sleep(Duration::from_millis(40));
        assert!(
            h.beat_age_ms() > 20,
            "the beat must actually be stale for this to mean anything"
        );
        assert!(h.live().is_ok(), "idle staleness is meaningless");
        // the SAME age while BUSY: stalled. Only the phase changed.
        h.phase.store(PHASE_BUSY, Ordering::Release);
        let why = h
            .live()
            .expect_err("a busy worker with a stale beat is not live");
        assert!(why.contains("stalled"), "{why}");
        // a fresh beat clears it — no latch, so a slow-but-progressing worker recovers by
        // itself instead of needing a restart.
        h.beat();
        assert!(h.live().is_ok());
    }

    #[test]
    fn worker_death_and_gpu_fault_latch_immediately_without_a_threshold_wait() {
        let h = WorkerHealth::new();
        h.mark_ready();
        h.beat(); // fresh beat: staleness cannot be what fails below
        h.mark_dead("worker thread panicked: test");
        let why = h.live().expect_err("a dead worker is never live");
        assert!(why.contains("panicked"), "{why}");
        assert!(h.ready(false).is_err());
        // a successful respawn clears the worker latch...
        h.mark_ready();
        assert!(h.live().is_ok());
        // ...but a GPU fault outlives it (a wedged card does not un-wedge).
        h.mark_gpu_fault("NVRM Xid 119 (fatal class)");
        h.mark_ready();
        let why = h
            .live()
            .expect_err("gpu fault must survive a worker respawn");
        assert!(why.contains("Xid 119"), "{why}");
    }

    #[test]
    fn readiness_is_off_while_draining_but_liveness_stays_on() {
        let h = WorkerHealth::new();
        h.mark_ready();
        assert!(h.live().is_ok());
        assert!(h.ready(false).is_ok());
        // draining: not ready (route away), still live (do NOT restart mid-drain).
        assert!(h.ready(true).is_err());
        assert!(h.live().is_ok());
    }

    #[test]
    fn loading_is_not_live_and_not_ready() {
        let h = WorkerHealth::new(); // starts in PHASE_LOADING
        assert!(h.live().is_err(), "weights are not resident yet");
        assert!(h.ready(false).is_err());
        h.mark_ready();
        assert!(h.live().is_ok());
        h.mark_respawning();
        assert!(h.live().is_err(), "a respawn load answers nothing");
        assert_eq!(h.generation(), 1);
    }

    #[test]
    fn tick_max_records_the_longest_iteration() {
        let h = WorkerHealth::new();
        h.mark_ready();
        h.beat();
        std::thread::sleep(Duration::from_millis(25));
        h.beat();
        let snap = h.snapshot();
        assert!(snap.tick_max_ms >= 20, "tick_max_ms = {}", snap.tick_max_ms);
        assert_eq!(snap.stall_threshold_ms, stall_threshold_ms());
    }

    #[test]
    fn smi_csv_scan_faults_only_on_definite_values() {
        // consumer part: every field [N/A] -> no fault invented
        assert!(scan_smi_csv("2026/08/06 20:00:00.000, [N/A], [N/A], [N/A]").is_none());
        // clean datacenter part
        assert!(scan_smi_csv("2026/08/06 20:00:00.000, 0, 0, No").is_none());
        // real ECC damage
        assert!(
            scan_smi_csv("2026/08/06 20:00:00.000, 3, 0, No").is_some_and(|r| r.contains("ECC"))
        );
        // row-remap failure
        assert!(
            scan_smi_csv("2026/08/06 20:00:00.000, 0, 0, Yes")
                .is_some_and(|r| r.contains("row-remap"))
        );
        // degraded (minimal) query shape: too few fields -> nothing to scan, no fault
        assert!(scan_smi_csv("2026/08/06 20:00:00.000, 512 MiB").is_none());
        assert!(scan_smi_csv("").is_none());
    }
}
