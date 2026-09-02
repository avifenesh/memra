//! Cooperative cancellation for long engine calls (lane/glm5-tp-serve-wiring-20260902,
//! round 3; memra #14).
//!
//! WHY. A serving prime is ONE engine call for the whole queued prompt on an eager-only
//! trunk (glm5_next: `prime_cache` -> `prime_cache_hyper` -> the chunk loop). Nothing inside
//! that call could observe the caller's world, so a client that disconnected mid-prime kept
//! the GPU on a prompt nobody would read, and the worker's per-tick disconnect sweep
//! (`s.tx.is_closed()` at the top of the tick) could not run until the call returned. The
//! second TP-2 box gate (2026-09-02) hit exactly this: a 245,421-token prime on the sharded
//! walk, the client killed at 52 minutes, and the worker never came back to its tick.
//!
//! SHAPE. A THREAD-LOCAL probe the CALLER arms around a call ([`arm`], RAII scope) and the
//! walk polls at its natural boundaries ([`check`]: per prime chunk, per layer inside a
//! chunk, every [`EP_TOKEN_STRIDE`] tokens inside the sequential EP MoE loop). Thread-local
//! because the prime runs on the calling thread and the probe must never leak into a sibling
//! worker's calls; a scope restores whatever was armed before it, so nesting is safe. Polling
//! an unarmed probe is one thread-local read and returns `false`, so every non-serving
//! caller (gates, CLI, tests) is byte- and cost-identical to before this module existed.
//!
//! UNWIND. A tripped poll returns [`Cancelled`] as the call's error; the walk's own `?`
//! unwinds through every layer, freeing the chunk's transients on the way out. The cache
//! keeps the chunks that completed (`cache.pos` advanced per chunk, exactly as a chunked
//! prime that hit an OOM would leave it); the caller marks the session aborted so the
//! partial cache is DROPPED, never parked or published. The site/done/total triple in the
//! error is the receipt: which boundary tripped and how far the call had run.

use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

/// The probe the caller arms: returns `true` once the work should stop.
pub type Probe = Arc<dyn Fn() -> bool + Send + Sync>;

/// Poll cadence inside the sequential EP MoE token loop (the only per-token walk that can
/// take minutes per layer-chunk when the grouped arm falls closed).
pub const EP_TOKEN_STRIDE: usize = 64;

thread_local! {
    static PROBE: RefCell<Option<Probe>> = const { RefCell::new(None) };
}

/// RAII scope of an armed probe; dropping it restores the previously armed probe (or none).
pub struct CancelScope {
    prev: Option<Probe>,
}

impl Drop for CancelScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        PROBE.with(|p| *p.borrow_mut() = prev);
    }
}

/// Arm `probe` for the calling thread until the returned scope drops.
pub fn arm(probe: Probe) -> CancelScope {
    let prev = PROBE.with(|p| p.borrow_mut().replace(probe));
    CancelScope { prev }
}

/// True when a probe is armed on this thread and it reports cancellation.
pub fn requested() -> bool {
    PROBE.with(|p| p.borrow().as_ref().is_some_and(|probe| probe()))
}

/// True when any probe is armed on this thread (diagnostics only).
pub fn armed() -> bool {
    PROBE.with(|p| p.borrow().is_some())
}

/// The cancellation error: which boundary tripped and how far the call had run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled {
    pub site: &'static str,
    pub done: usize,
    pub total: usize,
}

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "prime cancelled by the caller's probe (client disconnected) at {} {}/{}",
            self.site, self.done, self.total
        )
    }
}

impl std::error::Error for Cancelled {}

/// Poll the armed probe at a walk boundary; `Err(Cancelled)` unwinds the call.
pub fn check(site: &'static str, done: usize, total: usize) -> Result<(), Cancelled> {
    if requested() {
        Err(Cancelled { site, done, total })
    } else {
        Ok(())
    }
}

/// True when `err` is a [`Cancelled`] unwind (the caller's abort-vs-error branch).
pub fn is_cancelled(err: &(dyn std::error::Error + 'static)) -> bool {
    err.downcast_ref::<Cancelled>().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn unarmed_probe_never_cancels() {
        assert!(!armed());
        assert!(!requested());
        assert_eq!(check("chunk", 0, 4), Ok(()));
    }

    #[test]
    fn armed_probe_trips_and_names_the_boundary() {
        let flag = Arc::new(AtomicBool::new(false));
        let f = flag.clone();
        let scope = arm(Arc::new(move || f.load(Ordering::Relaxed)));
        assert!(armed());
        assert_eq!(check("chunk", 1, 4), Ok(()));
        flag.store(true, Ordering::Relaxed);
        let err = check("chunk", 2, 4).unwrap_err();
        assert_eq!(
            err,
            Cancelled {
                site: "chunk",
                done: 2,
                total: 4
            }
        );
        let boxed: Box<dyn std::error::Error> = Box::new(err);
        assert!(is_cancelled(boxed.as_ref()));
        assert!(boxed.to_string().contains("at chunk 2/4"));
        drop(scope);
        assert!(!armed());
        assert!(!requested());
    }

    #[test]
    fn nested_scopes_restore_the_outer_probe() {
        let _outer = arm(Arc::new(|| true));
        assert!(requested());
        {
            let _inner = arm(Arc::new(|| false));
            assert!(!requested());
        }
        assert!(requested());
    }

    #[test]
    fn a_plain_error_is_not_a_cancel() {
        let boxed: Box<dyn std::error::Error> = "engine error".into();
        assert!(!is_cancelled(boxed.as_ref()));
    }
}
