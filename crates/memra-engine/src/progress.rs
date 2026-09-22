//! FORWARD-PROGRESS ODOMETER, the engine's own answer to "is this worker busy, or hung?"
//! (lane/health-busy-vs-hung, memra#50, 2026-09-03).
//!
//! THE DEFECT THIS EXISTS FOR. `/health` used to read ONE signal for a BUSY worker: the age of
//! the scheduler-loop heartbeat (`WorkerHealth::beat`, stamped once per iteration in
//! `worker.rs`). That is a proxy for progress, not progress itself, and the proxy broke the
//! moment ONE iteration legitimately ran longer than the stall threshold. Measured, on the
//! glm5 ship-gate stress arm (darklanes `research/glm5-serving-launch-20260901/soak-20260901/
//! RESULT.md`, RED finding 1): waves of 8 to 22 admitted sessions carrying 20k-88k-token
//! prompts primed inside one scheduler iteration, the beat did not land for >120 s while the
//! worker was PROGRESSING NORMALLY, `/health` answered 503 `unhealthy` for three guard ticks,
//! and the supervisor SIGTERMed a server with 22 requests in flight. A false restart costs
//! every in-flight request plus a full model load.
//!
//! WHAT THIS PUBLISHES, and why it is honest. Every completed PRIME CHUNK stamps this
//! odometer: a token count, an event count, and the monotonic time of the last advance. The
//! stamp sits where the chunk's host-side result already exists, the chunk's logits are a
//! `Vec<f32>`, i.e. a device-to-host copy has already drained that stream (see
//! `prime_chunk_ppn`'s exit-publication note). So an advance is not "the host queued some
//! launches"; it is "the device finished that chunk's work and the host read the answer
//! back". That is the strongest liveness attestation available without a second thread.
//!
//! WHAT IT CANNOT DETECT, stated so nobody reads more into it:
//!   * A worker looping FOREVER INSIDE one chunk (a wedged kernel, a hung driver call, a
//!     deadlock inside a single prime call) advances nothing, so it is caught, but only
//!     after the stall threshold, exactly as before. This buys correctness under load, not
//!     faster hang detection.
//!   * A worker making progress on the WRONG work (a livelock that re-primes the same chunk
//!     forever, a scheduler that starves one session while another runs) reads healthy. This
//!     is a liveness signal, not a fairness or a correctness one.
//!   * Chunk granularity is the resolution: with `MEMRA_PRIME_CHUNK=0` a prompt primes in one
//!     call up to `PRIME_CHUNK_LAUNCH_CAP` (65,520 tokens), so the odometer's own gap can be
//!     a whole 65k-token prime. A deployment that pins the monolithic rollback seam is back
//!     to sizing `MEMRA_HEALTH_STALL_S` from its prefill rate by hand.
//!   * It is PROCESS-GLOBAL, not per-session. One live session priming keeps the process
//!     healthy while another session's work is stuck behind it. That is correct for the
//!     question `/health` asks ("should this process be RESTARTED?") and wrong for any
//!     per-request SLO, which admission and the first-token deadline own instead. The one
//!     exception is a serve route on its own thread (DSv4, memra#500): it installs a
//!     `ProgressSinkScope` and its chunks stamp that route's health record, not this global.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ROWS: AtomicU64 = AtomicU64::new(0);
static EVENTS: AtomicU64 = AtomicU64::new(0);
/// Milliseconds since [`epoch`] at the last advance. `u64::MAX` means "never advanced", which
/// is distinct from "advanced at t=0", a fresh process must not look like a progressing one.
static LAST_MS: AtomicU64 = AtomicU64::new(u64::MAX);

/// Process-start monotonic baseline. Milliseconds since this baseline are storable in an
/// atomic and immune to wall-clock steps: an NTP correction must never look like a wedged GPU.
fn epoch() -> Instant {
    static E: OnceLock<Instant> = OnceLock::new();
    *E.get_or_init(Instant::now)
}

fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64
}

/// One prime chunk completed on this process's worker thread, carrying `rows` token rows.
///
/// Ordering: the counters are Relaxed (diagnostics) but `LAST_MS` is Release and read Acquire,
/// so a reader that observes a fresh timestamp also observes the counts that produced it.
/// Cost is three atomic stores and one `Instant::now()` per CHUNK (not per token, not per
/// kernel), which is noise against a chunk that just moved thousands of token rows.
pub fn note_prime_rows(rows: usize) {
    // A thread that owns its own serve route (the DSv4 serving thread, memra#500) installed a
    // sink: its chunks attest THAT route's progress and stay out of the process-global
    // odometer, which answers for the central worker. Sharing one odometer would let either
    // side's primes hold the other's stall verdict healthy, and would let a route's chunk land
    // between the central shim's two `events()` reads.
    let routed = PROGRESS_SINK.with(|s| match s.borrow().as_ref() {
        Some(sink) => {
            sink(rows);
            true
        }
        None => false,
    });
    if routed {
        return;
    }
    ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    EVENTS.fetch_add(1, Ordering::Relaxed);
    LAST_MS.store(now_ms(), Ordering::Release);
}

/// Completed prime chunks so far. Used by `prime_cache_overlaid` to tell a CHUNKED walk
/// (which already stamped per chunk) from a MONOLITHIC one (which stamped nothing, and whose
/// only honest progress point is the call's own completion).
pub fn events() -> u64 {
    EVENTS.load(Ordering::Relaxed)
}

/// What the odometer has seen. `None` until the first advance, a process that has never
/// primed anything reports nothing rather than reporting an age measured from boot.
pub fn snapshot() -> Option<Progress> {
    let last = LAST_MS.load(Ordering::Acquire);
    if last == u64::MAX {
        return None;
    }
    Some(Progress {
        rows: ROWS.load(Ordering::Relaxed),
        events: EVENTS.load(Ordering::Relaxed),
        age_ms: now_ms().saturating_sub(last),
    })
}

// ---------------------------------------------------------------------------------------------
// ROUTE-OWNED PROGRESS SINK (memra#500, lane/dsv4-route-policies-20260922).
//
// A serve route that runs on its own thread (the DSv4 serving thread) is not the central worker,
// and its liveness is not the central worker's either: the process-global odometer above cannot
// tell whose chunk advanced it. The route installs a sink for the life of its thread through
// `ProgressSinkScope`; every `note_prime_rows` on that thread then feeds the route's own health
// record instead of the global. Same seam shape as the cancellation predicate below, and the same
// property: with no scope installed (the central worker, the CLI, every gate) the call is one
// thread-local read and the global odometer behaves exactly as before.
// ---------------------------------------------------------------------------------------------

/// A route's prime-row sink: completed rows in, stamped on the route's own record.
pub type ProgressSink = Box<dyn Fn(usize)>;

thread_local! {
    static PROGRESS_SINK: RefCell<Option<ProgressSink>> = const { RefCell::new(None) };
}

/// Routes this thread's prime-chunk stamps to `sink` while the guard lives. Nested scopes restore
/// the outer sink on drop.
pub struct ProgressSinkScope {
    prev: Option<ProgressSink>,
}

impl ProgressSinkScope {
    pub fn install(sink: ProgressSink) -> Self {
        let prev = PROGRESS_SINK.with(|s| s.borrow_mut().replace(sink));
        Self { prev }
    }
}

impl Drop for ProgressSinkScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        PROGRESS_SINK.with(|s| *s.borrow_mut() = prev);
    }
}

// ---------------------------------------------------------------------------------------------
// PRIME CANCELLATION POINT (memra#536 item 2, lane/spill-a-20260919 day 16).
//
// THE DEFECT. The worker's only cancellation point for a prime is the tick-top disconnect sweep
// (`worker.rs` "DISCONNECT ABORT"), which runs once per `prefill_tick` call: a disconnected client's
// prompt is primed to the end of the CURRENT take (1024 tokens; 8192 for a sole fresh request; the
// whole prompt for the monolithic class) before the sweep can retire it. Inside the engine call the
// chunk walks below already stop at every internal chunk boundary to stamp the odometer above, so
// that boundary is where a cancellation check costs nothing and leaves the cache at a completed
// chunk. The 2026-09-05 incident (329 `[abort] client disconnected while queued` lines behind one
// prime) is the class.
//
// THE SEAM, and why it is a thread-local rather than a parameter. The engine knows no HTTP channel;
// the worker installs a predicate ("is this session's client gone?") for the duration of ONE prime
// call through `PrimeCancelScope`, on the thread that runs the walk. Every walk asks the same one
// function at its chunk boundary; with no scope installed (the CLI, the gates, every test) the check
// is one thread-local read that answers "not cancelled", so those callers are byte-identical. The
// scope is a guard: it restores the previous predicate on drop, on the `?` paths too.
//
// ONE NUMERIC PROGRAM. The check never changes a chunk schedule: it either lets the walk continue
// exactly as before or returns `PrimeCancelled` AFTER a chunk completed and BEFORE the next starts.
// A request that is not cancelled produces the same bytes (the hit gate and the continuation gate are
// the proof; `tools/prime-cancel-gate.sh` compares the cold and warm digests of the next request
// against a run without the disconnect). A cancelled prime never returns logits, so its caller can
// publish nothing from it; the worker retires the session as aborted (no park, the cache returns to
// the pool at drop) and the capture sites, which run only after an `Ok` prime, are never reached.
//
// WHERE THE CHECK IS AND IS NOT (stated, not implied). It sits in the three sequential walks: the
// serial chunk walk (`prime_cache_overlaid_inner`), the GEMM chunk loop (`step35_prime_cache_batch`
// per chunk) and the single-engine hyper range walk. It is NOT in the pipelined walks (the PP-2 split
// primes with a `next_slot` in flight, the ppN wave walk): returning at a wave boundary there would
// leave another stage's work in flight against a cache the caller is about to drop, which is a
// wider seam (a drain plus the tainted-cache contract) than one check per chunk; those walks keep the
// tick-top sweep as their cancellation point. `prime_cache_batch` (one call for SEVERAL sessions) has
// no per-session cancel either: one member's disconnect cannot stop a wave that is priming its peers.
// ---------------------------------------------------------------------------------------------

thread_local! {
    static PRIME_CANCEL: RefCell<Option<Box<dyn Fn() -> bool>>> = const { RefCell::new(None) };
}

/// Installs a "cancelled?" predicate for the prime calls made on this thread while the guard lives.
/// Nested scopes restore the outer predicate on drop.
pub struct PrimeCancelScope {
    prev: Option<Box<dyn Fn() -> bool>>,
}

impl PrimeCancelScope {
    pub fn install(cancelled: Box<dyn Fn() -> bool>) -> Self {
        let prev = PRIME_CANCEL.with(|c| c.borrow_mut().replace(cancelled));
        Self { prev }
    }
}

impl Drop for PrimeCancelScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        PRIME_CANCEL.with(|c| *c.borrow_mut() = prev);
    }
}

/// The typed outcome of a cancelled prime: which chunk boundary stopped it and how many rows of
/// the call's take were primed (and are now in the cache the caller owns and will drop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimeCancelled {
    /// index (from 0) of the chunk that completed just before the check fired
    pub chunk: usize,
    /// rows of this call primed before the stop
    pub rows_done: usize,
    /// rows this call was asked to prime
    pub rows_total: usize,
}

impl std::fmt::Display for PrimeCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prime cancelled at chunk {} after {} of {} rows (client gone)",
            self.chunk, self.rows_done, self.rows_total
        )
    }
}

impl std::error::Error for PrimeCancelled {}

/// The chunk-boundary check. Called by a walk right after a chunk completed (the odometer stamp) and
/// only when more chunks remain (`rows_done < rows_total`): a finished prime is never "cancelled",
/// its logits return and the worker's sweep retires the session. Answers `Ok(())` with no scope
/// installed or while the predicate says the client is still there.
pub fn prime_cancel_point(
    chunk: usize,
    rows_done: usize,
    rows_total: usize,
) -> Result<(), PrimeCancelled> {
    if rows_done >= rows_total {
        return Ok(());
    }
    let cancelled = PRIME_CANCEL.with(|c| c.borrow().as_ref().is_some_and(|pred| pred()));
    if cancelled {
        Err(PrimeCancelled {
            chunk,
            rows_done,
            rows_total,
        })
    } else {
        Ok(())
    }
}

/// The odometer's observable state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// token rows primed since process start.
    pub rows: u64,
    /// completed prime chunks since process start.
    pub events: u64,
    /// milliseconds since the last completed chunk.
    pub age_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WIRING GATE. The odometer is only as honest as its call sites: a `progress` module that
    /// compiles, publishes and is read by `/health` while NOTHING in the prime path calls it
    /// would make every BUSY worker look hung again, silently, and every unit test above would
    /// still pass. So assert the calls exist in the prime walks, in COMMENT-STRIPPED source
    /// (the module and call-site docs mention `note_prime_rows` by name, and a doc mention is
    /// not an invocation).
    ///
    /// Bound, not an exact count: chunk walks get added. What must never happen is the count
    /// going to zero, or `prime_cache_overlaid` losing its call-granularity stamp, because
    /// either failure is invisible on a host and expensive on a box.
    #[test]
    fn the_prime_walks_actually_call_the_odometer() {
        let src = include_str!("hybrid_forward.rs");
        let code: String = src
            .lines()
            .map(|l| l.trim_start())
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let calls = code.matches("crate::progress::note_prime_rows(").count();
        assert!(
            calls >= 9,
            "the prime walks must stamp the forward-progress odometer (memra#50); found \
             {calls} live call sites in hybrid_forward.rs (7 per-chunk walks plus the two \
             call-granularity shims)"
        );
        // Both ENTRY points need the shim, not just one: `prime_cache_batch` does not route
        // through `prime_cache_overlaid`, and it was the multi-session batched wave prefill's
        // only coverage gap (review of #106).
        assert_eq!(
            code.matches("crate::progress::events()").count(),
            4,
            "both prime entries (prime_cache_overlaid, prime_cache_batch) must compare the \
             event count across the call, or a MONOLITHIC prime on that entry stamps nothing"
        );
        for entry in [
            "fn prime_cache_overlaid_inner(",
            "fn prime_cache_batch_inner(",
        ] {
            assert!(
                code.contains(entry),
                "the shim for {entry} is gone: its entry is stamping nothing"
            );
        }
    }

    /// WIRING GATE, the DSv4 twin (memra#500). The test above reads only `hybrid_forward.rs`, so a
    /// second prime implementation was born passing it: `dsv4_gpu.rs` had no stamp at all and the
    /// gate stayed green. Both DSv4 chunk walks (the plain `continue_prefix_chunked` and the
    /// drafter's `dspark_continue_prefix_chunked`) must stamp per completed chunk, and the
    /// monolithic serve prefill must stamp at call granularity. Comment-stripped, bound from
    /// below.
    #[test]
    fn the_dsv4_prime_walks_actually_call_the_odometer() {
        let src = include_str!("dsv4_gpu.rs");
        let code: String = src
            .lines()
            .map(|l| l.trim_start())
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let calls = code.matches("crate::progress::note_prime_rows(").count();
        assert!(
            calls >= 3,
            "the DSv4 prime walks must stamp the forward-progress odometer (memra#500); found \
             {calls} live call sites in dsv4_gpu.rs (two per-chunk walks plus the monolithic \
             call-granularity stamp)"
        );
    }

    /// An installed sink takes the thread's stamps; the guard restores the outer sink on drop and
    /// the global path resumes after the last guard.
    #[test]
    fn a_progress_sink_scope_routes_this_threads_stamps_and_restores_on_drop() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;
        let outer = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(AtomicUsize::new(0));
        {
            let o = outer.clone();
            let _scope = ProgressSinkScope::install(Box::new(move |r| {
                o.fetch_add(r, Ordering::Relaxed);
            }));
            note_prime_rows(512);
            {
                let i = inner.clone();
                let _inner = ProgressSinkScope::install(Box::new(move |r| {
                    i.fetch_add(r, Ordering::Relaxed);
                }));
                note_prime_rows(64);
            }
            note_prime_rows(8);
        }
        assert_eq!(outer.load(Ordering::Relaxed), 520);
        assert_eq!(inner.load(Ordering::Relaxed), 64);
        // No scope left: the stamp is the global odometer's again (another thread's sink never
        // sees it, and this thread's old sinks are gone).
        note_prime_rows(1);
        assert_eq!(outer.load(Ordering::Relaxed), 520);
        assert!(snapshot().is_some());
    }

    /// The cancellation point answers "not cancelled" with no scope, fires only while a scope says
    /// so, never fires at the last chunk, and the guard restores the outer predicate on drop.
    #[test]
    fn prime_cancel_point_fires_only_under_an_installed_scope_and_never_at_the_end() {
        assert_eq!(prime_cancel_point(0, 256, 1024), Ok(()));
        {
            let _scope = PrimeCancelScope::install(Box::new(|| true));
            assert_eq!(
                prime_cancel_point(2, 768, 1024),
                Err(PrimeCancelled {
                    chunk: 2,
                    rows_done: 768,
                    rows_total: 1024
                })
            );
            // a completed take is never cancelled: its logits return
            assert_eq!(prime_cancel_point(3, 1024, 1024), Ok(()));
            {
                let _inner = PrimeCancelScope::install(Box::new(|| false));
                assert_eq!(prime_cancel_point(0, 1, 2), Ok(()));
            }
            // the inner guard restored the outer predicate
            assert!(prime_cancel_point(0, 1, 2).is_err());
        }
        assert_eq!(prime_cancel_point(0, 1, 2), Ok(()));
        let e: Box<dyn std::error::Error> = Box::new(PrimeCancelled {
            chunk: 1,
            rows_done: 512,
            rows_total: 4096,
        });
        assert!(e.downcast_ref::<PrimeCancelled>().is_some());
        assert_eq!(
            e.to_string(),
            "prime cancelled at chunk 1 after 512 of 4096 rows (client gone)"
        );
    }

    /// WIRING GATE for the cancellation point, the odometer gate's twin: the three sequential walks
    /// must ask `prime_cancel_point` at their chunk boundary, in comment-stripped source. Bound from
    /// below (walks get added); the pipelined walks are deliberately absent (see the module note).
    #[test]
    fn the_sequential_prime_walks_ask_the_cancellation_point() {
        let src = include_str!("hybrid_forward.rs");
        let code: String = src
            .lines()
            .map(|l| l.trim_start())
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        let calls = code.matches("crate::progress::prime_cancel_point(").count();
        assert!(
            calls >= 3,
            "the sequential prime walks must ask the cancellation point at their chunk boundary \
             (memra#536); found {calls} live call sites in hybrid_forward.rs"
        );
    }

    /// The never-advanced state is DISTINCT from a zero age. Asserted because the whole point
    /// of the odometer is that health falls back to beat age when it has nothing to say, and
    /// a `Some(age 0)` on a fresh process would instead declare a never-run worker healthy
    /// forever.
    #[test]
    fn snapshot_is_none_until_the_first_advance_then_counts() {
        // This test owns the process-global only in the sense that it asserts monotonicity,
        // never an absolute value: other tests in the same binary may also advance it.
        let before = snapshot();
        note_prime_rows(4096);
        let after = snapshot().expect("an advance was just stamped");
        match before {
            None => assert_eq!(after.rows, 4096),
            Some(b) => {
                assert!(after.rows.saturating_sub(b.rows) >= 4096);
                assert!(after.events > b.events);
            }
        }
    }
}
