//! Per-route load telemetry and admission state for dedicated serving routes (memra#501).
//!
//! The central worker publishes `worker::Metrics` and holds the per-lane
//! `ADMISSION_RESERVATIONS`; the HTTP layer sheds and stamps X-RateLimit-* from those. A
//! dedicated route (today the DSv4 serving thread, one request at a time on a FIFO channel) is
//! neither of them, so before this module its requests were counted against the hybrid
//! lane's 64-session cap, priced with the static 2 s reset, and invisible to /metrics.
//!
//! A [`RouteLoad`] is the route's own book:
//!   * `capacity`: the route's concurrency from its contract (`RouteCapacity`, 1 for DSv4);
//!   * `waiting[lane]`: requests reserved at the HTTP layer and not yet dequeued by the route,
//!     held by a [`RouteTicket`] that travels inside the `Request` and releases on dequeue or
//!     on drop (a request dropped anywhere, including in a dead thread's channel, frees its
//!     slot);
//!   * `inflight[lane]`: the HTTP-side view (submitted, response not finished), so the hybrid
//!     lane's X-RateLimit arithmetic can subtract route traffic it does not serve;
//!   * `running`, the served counters and a service-time window, written by the route thread
//!     through a [`RouteRun`].
//!
//! The registry is process-global, like the admission gauges it sits beside, and keyed by the
//! served model name. Registration is get-or-create so a worker respawn keeps the book (and
//! every outstanding ticket) continuous.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

/// Service-time samples kept for the estimate (most recent first out).
const SERVICE_WINDOW: usize = 64;
/// Round-time samples kept for the p50/p99 row.
const ROUND_WINDOW: usize = 256;

/// A bounded sample window with nearest-rank percentiles.
#[derive(Default)]
struct Window {
    cap: usize,
    samples: VecDeque<u64>,
}

impl Window {
    fn new(cap: usize) -> Self {
        Window {
            cap,
            samples: VecDeque::with_capacity(cap),
        }
    }

    fn push(&mut self, v: u64) {
        if self.samples.len() == self.cap {
            self.samples.pop_front();
        }
        self.samples.push_back(v);
    }

    /// Nearest-rank percentile, `None` while empty.
    fn percentile(&self, pct: u64) -> Option<u64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut v: Vec<u64> = self.samples.iter().copied().collect();
        v.sort_unstable();
        let rank = ((pct as usize * v.len()).div_ceil(100)).clamp(1, v.len());
        Some(v[rank - 1])
    }
}

/// One dedicated route's load book. See the module doc.
pub(crate) struct RouteLoad {
    name: String,
    capacity: usize,
    waiting: [AtomicUsize; 3],
    inflight: [AtomicUsize; 3],
    running: AtomicUsize,
    admitted: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    refused: AtomicU64,
    tokens_out: AtomicU64,
    prompt_tokens_in: AtomicU64,
    cached_tokens_in: AtomicU64,
    rounds: AtomicU64,
    service_ms: Mutex<Window>,
    round_ms: Mutex<Window>,
}

/// What a finished request reports to its route's book.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ServeStats {
    pub tokens_out: usize,
    pub n_prompt: usize,
    pub n_cached: usize,
}

/// The published view of a [`RouteLoad`] (the /metrics `routes` block).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RouteLoadSnapshot {
    pub name: String,
    pub capacity: usize,
    pub waiting: usize,
    pub inflight: usize,
    pub running: usize,
    pub admitted: u64,
    pub completed: u64,
    pub failed: u64,
    /// Admitted, then the client left before the route served it (a memory defer).
    pub cancelled: u64,
    /// Admitted, then turned away by the route's memory door (memra#503).
    pub refused: u64,
    pub tokens_out: u64,
    pub prompt_tokens_in: u64,
    pub cached_tokens_in: u64,
    pub rounds: u64,
    pub service_p50_ms: Option<u64>,
    pub service_p99_ms: Option<u64>,
    pub round_p50_ms: Option<u64>,
    pub round_p99_ms: Option<u64>,
}

impl RouteLoad {
    pub(crate) fn new(name: impl Into<String>, capacity: usize) -> Arc<Self> {
        Arc::new(RouteLoad {
            name: name.into(),
            capacity: capacity.max(1),
            waiting: Default::default(),
            inflight: Default::default(),
            running: AtomicUsize::new(0),
            admitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            tokens_out: AtomicU64::new(0),
            prompt_tokens_in: AtomicU64::new(0),
            cached_tokens_in: AtomicU64::new(0),
            rounds: AtomicU64::new(0),
            service_ms: Mutex::new(Window::new(SERVICE_WINDOW)),
            round_ms: Mutex::new(Window::new(ROUND_WINDOW)),
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn waiting(&self, lane: usize) -> usize {
        self.waiting[lane].load(Ordering::Acquire)
    }

    /// Reserved and not yet dequeued, every lane: the route is one FIFO, so this is the backlog
    /// ahead of a new arrival whatever its lane.
    pub(crate) fn waiting_total(&self) -> usize {
        self.waiting.iter().map(|w| w.load(Ordering::Acquire)).sum()
    }

    pub(crate) fn running(&self) -> usize {
        self.running.load(Ordering::Acquire)
    }

    pub(crate) fn inflight(&self, lane: usize) -> usize {
        self.inflight[lane].load(Ordering::Acquire)
    }

    pub(crate) fn inflight_total(&self) -> usize {
        self.inflight
            .iter()
            .map(|w| w.load(Ordering::Acquire))
            .sum()
    }

    /// Take one waiting slot on `lane` if the count is still `seen` (the caller's admission
    /// decision was made against `seen`, so a concurrent reservation invalidates it and the
    /// caller re-decides). `Err(current)` on a lost race.
    pub(crate) fn try_reserve(
        self: &Arc<Self>,
        lane: usize,
        seen: usize,
    ) -> Result<RouteTicket, usize> {
        self.waiting[lane]
            .compare_exchange(seen, seen + 1, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| RouteTicket {
                load: self.clone(),
                lane,
            })
    }

    /// HTTP-side in-flight slot, held for the life of the response.
    pub(crate) fn enter(self: &Arc<Self>, lane: usize) -> RouteInflight {
        self.inflight[lane].fetch_add(1, Ordering::AcqRel);
        RouteInflight {
            load: self.clone(),
            lane,
        }
    }

    /// The route dequeued a request. `running` rises BEFORE the caller drops the request's
    /// ticket, so the occupancy an admission reads (`running + waiting`) never dips between the
    /// two. Nothing is counted served until [`RouteRun::admit`]: a request whose client left
    /// while it queued is dropped unadmitted and moves no counter.
    pub(crate) fn begin(self: &Arc<Self>) -> RouteRun {
        self.running.fetch_add(1, Ordering::AcqRel);
        RouteRun {
            load: self.clone(),
            t0: Instant::now(),
            admitted: false,
            done: false,
        }
    }

    /// One decode step or speculative round completed, `ms` after the previous one.
    pub(crate) fn note_round(&self, ms: u64) {
        self.rounds.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut w) = self.round_ms.lock() {
            w.push(ms);
        }
    }

    /// Seconds one request occupies the route: the p50 of observed service wall time when there
    /// is any, else `fallback_s` (the `MEMRA_RL_RESET_S` static, default 2). Same clamp as the
    /// hybrid estimate, 1..=600.
    pub(crate) fn service_estimate_s(&self, fallback_s: u64) -> u64 {
        match self.service_ms.lock().ok().and_then(|w| w.percentile(50)) {
            Some(ms) => ms.div_ceil(1000).clamp(1, 600),
            None => fallback_s,
        }
    }

    /// Test seam: record one service-time sample without running a request for it.
    #[cfg(test)]
    pub(crate) fn seed_service_ms(&self, ms: u64) {
        self.service_ms.lock().unwrap().push(ms);
    }

    pub(crate) fn snapshot(&self) -> RouteLoadSnapshot {
        let (service_p50_ms, service_p99_ms) = self
            .service_ms
            .lock()
            .map(|w| (w.percentile(50), w.percentile(99)))
            .unwrap_or_default();
        let (round_p50_ms, round_p99_ms) = self
            .round_ms
            .lock()
            .map(|w| (w.percentile(50), w.percentile(99)))
            .unwrap_or_default();
        RouteLoadSnapshot {
            name: self.name.clone(),
            capacity: self.capacity,
            waiting: self.waiting_total(),
            inflight: self.inflight_total(),
            running: self.running(),
            admitted: self.admitted.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            cancelled: self.cancelled.load(Ordering::Relaxed),
            refused: self.refused.load(Ordering::Relaxed),
            tokens_out: self.tokens_out.load(Ordering::Relaxed),
            prompt_tokens_in: self.prompt_tokens_in.load(Ordering::Relaxed),
            cached_tokens_in: self.cached_tokens_in.load(Ordering::Relaxed),
            rounds: self.rounds.load(Ordering::Relaxed),
            service_p50_ms,
            service_p99_ms,
            round_p50_ms,
            round_p99_ms,
        }
    }
}

fn decrement(counter: &AtomicUsize) {
    let _ = counter.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_sub(1));
}

/// A request's waiting slot on its route. Moved into `Request::route_ticket` before the send,
/// dropped by the route at dequeue, or by whoever drops the request first.
pub(crate) struct RouteTicket {
    load: Arc<RouteLoad>,
    lane: usize,
}

impl RouteTicket {
    pub(crate) fn route(&self) -> &str {
        self.load.name()
    }
}

impl Drop for RouteTicket {
    fn drop(&mut self) {
        decrement(&self.load.waiting[self.lane]);
    }
}

/// The HTTP-side in-flight slot of a route-bound request (rides `InflightGuard`).
pub(crate) struct RouteInflight {
    load: Arc<RouteLoad>,
    lane: usize,
}

impl Drop for RouteInflight {
    fn drop(&mut self) {
        decrement(&self.load.inflight[self.lane]);
    }
}

/// One dequeued request. `admit` starts its service clock, `finish` records it as completed
/// with its service time, `cancel` and `refuse` record a request the route never served (its
/// client left during a memory defer, or the memory door turned it away); dropping it admitted
/// but unfinished (an error event, a panic) counts a failure. Only `finish` feeds the estimate.
pub(crate) struct RouteRun {
    load: Arc<RouteLoad>,
    t0: Instant,
    admitted: bool,
    done: bool,
}

impl RouteRun {
    pub(crate) fn admit(&mut self) {
        if !self.admitted {
            self.admitted = true;
            self.t0 = Instant::now();
            self.load.admitted.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn finish(mut self, stats: ServeStats) {
        self.admit();
        let l = &self.load;
        l.completed.fetch_add(1, Ordering::Relaxed);
        l.tokens_out
            .fetch_add(stats.tokens_out as u64, Ordering::Relaxed);
        l.prompt_tokens_in
            .fetch_add(stats.n_prompt as u64, Ordering::Relaxed);
        l.cached_tokens_in
            .fetch_add(stats.n_cached as u64, Ordering::Relaxed);
        if let Ok(mut w) = l.service_ms.lock() {
            w.push(self.t0.elapsed().as_millis() as u64);
        }
        self.done = true;
    }

    pub(crate) fn cancel(mut self) {
        self.admit();
        self.load.cancelled.fetch_add(1, Ordering::Relaxed);
        self.done = true;
    }

    pub(crate) fn refuse(mut self) {
        self.admit();
        self.load.refused.fetch_add(1, Ordering::Relaxed);
        self.done = true;
    }
}

impl Drop for RouteRun {
    fn drop(&mut self) {
        if self.admitted && !self.done {
            self.load.failed.fetch_add(1, Ordering::Relaxed);
        }
        decrement(&self.load.running);
    }
}

fn registry() -> &'static RwLock<HashMap<String, Arc<RouteLoad>>> {
    static R: OnceLock<RwLock<HashMap<String, Arc<RouteLoad>>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// Get-or-create the route's book. A respawned worker re-registers the same name and keeps the
/// existing book, so counters and outstanding tickets stay continuous across the respawn.
pub(crate) fn register(name: &str, capacity: usize) -> Arc<RouteLoad> {
    let mut map = registry().write().unwrap_or_else(|p| p.into_inner());
    map.entry(name.to_string())
        .or_insert_with(|| RouteLoad::new(name, capacity))
        .clone()
}

/// The book of the dedicated route serving `model`, if one is registered. `None` means the
/// model is served by the central worker and the lane-scoped admission applies.
pub(crate) fn lookup(model: &str) -> Option<Arc<RouteLoad>> {
    registry()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .get(model)
        .cloned()
}

/// Every registered route, sorted by name (a stable /metrics order).
pub(crate) fn all() -> Vec<Arc<RouteLoad>> {
    let mut v: Vec<Arc<RouteLoad>> = registry()
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .values()
        .cloned()
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// HTTP in-flight on `lane` across every dedicated route: what the lane gauge counts that the
/// central worker does not serve.
pub(crate) fn inflight_on_lane(lane: usize) -> usize {
    all().iter().map(|r| r.inflight(lane)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ticket_releases_its_waiting_slot_on_drop() {
        let r = RouteLoad::new("t-ticket", 1);
        let t = r.try_reserve(0, 0).unwrap();
        assert_eq!(t.route(), "t-ticket");
        assert_eq!(r.waiting(0), 1);
        assert_eq!(r.waiting_total(), 1);
        // a decision made against a stale count loses the race and re-decides
        assert_eq!(r.try_reserve(0, 0).err(), Some(1));
        let t2 = r.try_reserve(2, 0).unwrap();
        assert_eq!(r.waiting_total(), 2);
        drop(t);
        drop(t2);
        assert_eq!(r.waiting_total(), 0);
    }

    #[test]
    fn a_finished_run_feeds_the_estimate_and_a_dropped_one_counts_a_failure() {
        let r = RouteLoad::new("t-run", 1);
        assert_eq!(r.service_estimate_s(2), 2, "no signal: the static fallback");
        let mut run = r.begin();
        assert_eq!(r.running(), 1);
        run.admit();
        run.finish(ServeStats {
            tokens_out: 7,
            n_prompt: 30,
            n_cached: 10,
        });
        assert_eq!(r.running(), 0);
        let mut failed = r.begin();
        failed.admit();
        drop(failed);
        // dequeued after its client left: never admitted, moves no counter
        drop(r.begin());
        assert_eq!(r.running(), 0);
        let s = r.snapshot();
        assert_eq!((s.admitted, s.completed, s.failed), (2, 1, 1));
        assert_eq!(
            (s.tokens_out, s.prompt_tokens_in, s.cached_tokens_in),
            (7, 30, 10)
        );
        assert!(s.service_p50_ms.is_some());
        // a sub-second sample rounds up to the 1 s floor, never 0
        assert_eq!(r.service_estimate_s(2), 1);
    }

    #[test]
    fn a_cancelled_or_refused_run_is_neither_served_nor_failed() {
        let r = RouteLoad::new("t-refuse", 1);
        let mut c = r.begin();
        c.admit();
        c.cancel();
        r.begin().refuse();
        assert_eq!(r.running(), 0);
        let s = r.snapshot();
        assert_eq!(
            (s.admitted, s.completed, s.failed, s.cancelled, s.refused),
            (2, 0, 0, 1, 1)
        );
        assert_eq!(
            s.service_p50_ms, None,
            "a turned-away request is not a service sample"
        );
        assert_eq!(r.service_estimate_s(2), 2);
    }

    #[test]
    fn the_estimate_is_the_p50_of_observed_service_time() {
        let r = RouteLoad::new("t-p50", 1);
        {
            let mut w = r.service_ms.lock().unwrap();
            for ms in [4_000, 9_000, 30_000] {
                w.push(ms);
            }
        }
        assert_eq!(r.service_estimate_s(2), 9);
        {
            let mut w = r.service_ms.lock().unwrap();
            w.push(10_000_000);
        }
        assert_eq!(
            r.service_estimate_s(2),
            9,
            "one outlier does not move the median"
        );
    }

    #[test]
    fn windows_are_bounded() {
        let mut w = Window::new(3);
        for v in 1..=10 {
            w.push(v);
        }
        assert_eq!(w.samples.len(), 3);
        assert_eq!(w.percentile(50), Some(9));
        assert_eq!(w.percentile(99), Some(10));
        assert_eq!(Window::new(3).percentile(50), None);
    }

    #[test]
    fn registration_is_get_or_create_and_lookup_is_by_model() {
        let a = register("t-reg-model", 1);
        let _t = a.try_reserve(0, 0).unwrap();
        let b = register("t-reg-model", 1);
        assert!(
            Arc::ptr_eq(&a, &b),
            "a respawn keeps the book and its tickets"
        );
        assert_eq!(lookup("t-reg-model").unwrap().waiting_total(), 1);
        assert!(lookup("t-reg-unknown").is_none());
        let inflight = a.enter(1);
        assert!(inflight_on_lane(1) >= 1);
        drop(inflight);
        assert_eq!(a.inflight(1), 0);
    }
}
