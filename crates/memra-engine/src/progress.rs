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
//!     per-request SLO, which admission and the first-token deadline own instead.

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
