//! Lane model — the yield-gate invariants, native in the scheduler.
//! (ARCHITECTURE-H100.md §3 B3; thresholds mirror the out-of-process sidecar gate so the
//! external sidecar remains a compatible outer layer.)
//!
//! Three lanes:
//!   interactive — the protected class; always admitted; its decode step latency IS the
//!                 SLO sensor (true engine timings, not the sidecar's network-gap proxy).
//!   judge       — prefill-shaped read work; admitted while interactive p99 < 100% of SLO.
//!   harvest     — decode-shaped batch generation; sheds first, at 90% of SLO.
//!
//! Invariants (the B2/D1 lessons):
//!   - shed happens at ADMISSION, never by queueing inside the engine ("the engine's own
//!     queue is where the tail dies") — a shed request gets an immediate retryable error;
//!   - per-lane PREFILL BUDGETS per tick replace vLLM's single global chunked-prefill knob
//!     (the knob that taxed baseline p99 11.6 -> 40.6 ms at zero parasite load);
//!   - interactive is never preempted; harvest yields first.

use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Interactive,
    Judge,
    Harvest,
}

impl Lane {
    /// Parse an `x-lane` header value. Unknown => None (handler answers 400, like the sidecar).
    pub fn parse(v: &str) -> Option<Lane> {
        match v {
            "interactive" => Some(Lane::Interactive),
            "judge" => Some(Lane::Judge),
            "harvest" => Some(Lane::Harvest),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Lane::Interactive => "interactive",
            Lane::Judge => "judge",
            Lane::Harvest => "harvest",
        }
    }
    pub const ALL: [Lane; 3] = [Lane::Interactive, Lane::Judge, Lane::Harvest];
    pub fn idx(&self) -> usize {
        match self {
            Lane::Interactive => 0,
            Lane::Judge => 1,
            Lane::Harvest => 2,
        }
    }
}

/// Windowed percentile over interactive decode-step latencies (ms). Engine ground truth:
/// the worker records the wall time of each batched decode tick that contained at least
/// one interactive session — that IS the interactive TPOT for that tick.
pub struct StepStats {
    window: VecDeque<(Instant, f32)>,
    window_s: f32,
}

impl StepStats {
    pub fn new(window_s: f32) -> Self {
        Self {
            window: VecDeque::with_capacity(4096),
            window_s,
        }
    }
    pub fn record(&mut self, ms: f32) {
        self.window.push_back((Instant::now(), ms));
        if self.window.len() > 16384 {
            self.window.pop_front();
        }
    }
    fn evict(&mut self) {
        let cutoff = self.window_s;
        while let Some(&(t, _)) = self.window.front() {
            if t.elapsed().as_secs_f32() > cutoff {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }
    /// Live sample count after eviction. The statistical mass behind `p()`: with n
    /// samples, p99 selects index round(0.99*(n-1)) — the MAXIMUM for every n <= 50 —
    /// so a small window makes "p99" a synonym for "the one slowest thing that
    /// happened lately", which is exactly the wrong signal to gate on.
    pub fn samples(&mut self) -> usize {
        self.evict();
        self.window.len()
    }
    /// q in [0,100]. None until the window has signal (cold start => lanes admit).
    pub fn p(&mut self, q: f32) -> Option<f32> {
        self.evict();
        if self.window.is_empty() {
            return None;
        }
        let mut v: Vec<f32> = self.window.iter().map(|&(_, g)| g).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let i = ((q / 100.0) * (v.len() - 1) as f32).round() as usize;
        Some(v[i.min(v.len() - 1)])
    }
}

/// Admission policy: thresholds as fractions of the SLO, per lane (yieldgate.py SHED_AT).
pub struct LanePolicy {
    pub slo_p99_ms: f32,
    /// Minimum window samples before the p99 gate may SHED (below it the window is
    /// treated as cold, like `None`). Born 2026-08-28: on an idle prod box whose only
    /// interactive traffic was the watchdog's one-token probe (~5 samples per 30s
    /// window), p99 == the probe's own PRIME-carrying tick (~46-50ms vs the 45ms
    /// harvest threshold), and the health probe itself shed real harvest requests on
    /// an empty machine — 2 of the day's 4 capture calls 429'd, painting the CRM 50%
    /// red. Probe cardinality is noise, not signal; the `starved` sentinel (which
    /// needs no window) still sheds under genuine starvation regardless of this floor.
    pub min_samples: usize,
    /// admission threshold per lane index (interactive unused — always admitted)
    pub shed_at: [f32; 3],
    /// per-tick prefill token budgets [interactive, judge, harvest]; interactive budget is
    /// the tick chunk size (uncapped policy-wise), judge/harvest are the dark-lane budgets.
    pub prefill_budget: [usize; 3],
    /// max resident sessions per lane
    pub max_sessions: [usize; 3],
}

impl LanePolicy {
    pub fn from_env() -> Self {
        let f = |k: &str, d: f32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let u = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        Self {
            slo_p99_ms: f("MEMRA_SLO_P99_MS", 50.0),
            // 32: a one-token watchdog probe contributes ~5-10 samples per 30s window,
            // real interactive load contributes hundreds per second (per-token records).
            // 0 disables the floor (the pre-2026-08-28 behavior).
            min_samples: u("MEMRA_SHED_MIN_SAMPLES", 32),
            shed_at: [
                f32::INFINITY,
                f("MEMRA_SHED_JUDGE", 1.00),
                f("MEMRA_SHED_HARVEST", 0.90),
            ],
            // Dark-lane budgets are PER-TICK STALL BOUNDS, not throughput knobs: a 2048-tok
            // judge chunk = ~230 ms of decode starvation per tick (measured 2026-07-26,
            // native-judge battery: p99 17.6 -> 282 ms at 1 req/s). 256 tok ≈ 30 ms bound.
            prefill_budget: [
                u("MEMRA_PREFILL_TICK", 1024),
                u("MEMRA_PREFILL_JUDGE", 256),
                u("MEMRA_PREFILL_HARVEST", 256),
            ],
            // Interactive cap sizes CONCURRENCY, not batching (decode runs multiple <=8-row
            // chunks per tick). 2026-07-26 battery: cap 4 vs 32 closed-loop clients made
            // QUEUE WAIT the interactive bottleneck (client streams starved in FIFO while
            // the engine idled at p99 17.8ms) — the cap must admit the whole protected set;
            // aggregate decode throughput is the real limit, enforced by the SLO estimator.
            max_sessions: [
                u("MEMRA_LANE_MAX_INTERACTIVE", 32),
                u("MEMRA_LANE_MAX_JUDGE", 4),
                u("MEMRA_LANE_MAX_HARVEST", 8),
            ],
        }
    }

    /// The yield gate: interactive always; judge/harvest against the measured interactive
    /// p99 vs their SLO fraction. No signal yet (cold estimator) => admit, like the sidecar.
    ///
    /// `starved` closes the estimator's blind spot (2026-07-26 native-judge battery, rates
    /// 4-8: interactive decoded ZERO tokens, so no p99 samples arrived, the window aged out,
    /// and the cold-start rule kept admitting judges into total starvation). The worker sets
    /// it when interactive work EXISTS but no interactive decode tick has run within the SLO
    /// age — starvation IS an SLO breach even though the estimator can't see it.
    pub fn admit(&self, lane: Lane, stats: &mut StepStats, starved: bool) -> bool {
        if lane == Lane::Interactive {
            return true;
        }
        if starved {
            return false;
        }
        // Below the floor the window is statistically no signal (p99 of n<=50 samples
        // IS the max sample) — same verdict as the empty-window cold start: admit.
        if stats.samples() < self.min_samples {
            return true;
        }
        match stats.p(99.0) {
            Some(p99) => p99 < self.slo_p99_ms * self.shed_at[lane.idx()],
            None => true,
        }
    }
}

/// Per-lane counters + latency snapshot, exported at /yield/metrics (sidecar-compatible shape).
#[derive(Default, Clone, serde::Serialize)]
pub struct LaneMetrics {
    pub admitted: [u64; 3],
    pub shed: [u64; 3],
    pub completed: [u64; 3],
    pub tokens_out: [u64; 3],
    pub step_p50_ms: f32,
    pub step_p99_ms: f32,
    pub batch_size_last: usize,
}

pub type SharedMetrics = std::sync::Arc<std::sync::Mutex<LaneMetrics>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(min_samples: usize) -> LanePolicy {
        LanePolicy {
            slo_p99_ms: 50.0,
            shed_at: [f32::INFINITY, 1.00, 0.90],
            prefill_budget: [1024, 256, 256],
            max_sessions: [32, 4, 8],
            min_samples,
        }
    }

    /// The 2026-08-28 incident, numerically: an idle box whose window holds only the
    /// watchdog probe's samples — a handful of fast decode ticks plus ONE prime-carrying
    /// tick at ~46ms. p99 of five samples is the max, 46 > 50*0.9 = 45, and the old gate
    /// shed harvest on an empty machine. The floor reads that window as no-signal.
    #[test]
    fn probe_cardinality_window_does_not_shed_harvest() {
        let mut stats = StepStats::new(30.0);
        for ms in [12.0, 11.0, 13.0, 12.0, 46.0] {
            stats.record(ms);
        }
        assert!(
            !policy(0).admit(Lane::Harvest, &mut stats, false),
            "floor disabled reproduces the incident: one probe prime tick sheds harvest"
        );
        assert!(
            policy(32).admit(Lane::Harvest, &mut stats, false),
            "5 samples are noise, not congestion — the floored gate admits"
        );
    }

    /// The floor must not blunt the gate's real job: at genuine load the window carries
    /// hundreds of per-token samples, and a p99 over budget still sheds.
    #[test]
    fn real_congestion_still_sheds_past_the_floor() {
        let mut stats = StepStats::new(30.0);
        for i in 0..300 {
            stats.record(if i % 10 == 0 { 55.0 } else { 30.0 });
        }
        assert!(!policy(32).admit(Lane::Harvest, &mut stats, false));
        // and a healthy p99 at the same mass admits
        let mut healthy = StepStats::new(30.0);
        for _ in 0..300 {
            healthy.record(20.0);
        }
        assert!(policy(32).admit(Lane::Harvest, &mut healthy, false));
    }

    /// Starvation needs no window and no floor: interactive work exists but is not
    /// decoding — shed, whatever the sample count says.
    #[test]
    fn starved_sheds_regardless_of_floor() {
        let mut empty = StepStats::new(30.0);
        assert!(!policy(32).admit(Lane::Harvest, &mut empty, true));
        assert!(!policy(0).admit(Lane::Harvest, &mut empty, true));
    }

    /// Interactive is never gated, floor or not, congestion or not.
    #[test]
    fn interactive_always_admits() {
        let mut stats = StepStats::new(30.0);
        for _ in 0..300 {
            stats.record(500.0);
        }
        assert!(policy(32).admit(Lane::Interactive, &mut stats, false));
    }

    /// At exactly the floor the gate ARMS: the boundary belongs to the shed side, so an
    /// off-by-one cannot quietly re-open the probe-noise hole.
    #[test]
    fn floor_boundary_arms_the_gate() {
        let mut stats = StepStats::new(30.0);
        for _ in 0..32 {
            stats.record(48.0);
        }
        assert!(
            !policy(32).admit(Lane::Harvest, &mut stats, false),
            "n == floor gates"
        );
        let mut below = StepStats::new(30.0);
        for _ in 0..31 {
            below.record(48.0);
        }
        assert!(
            policy(32).admit(Lane::Harvest, &mut below, false),
            "n == floor-1 admits"
        );
    }
}
