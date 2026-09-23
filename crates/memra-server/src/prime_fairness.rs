//! Route-independent policy around the worker's existing tick-top drain/admission.
//! Saved primes yield at their frozen numerical boundaries. A chunk costing more
//! than the service goal grants ready decode peers one worker-wide recovery
//! interval across ordinary worker ticks, whatever the number of saved primes;
//! arrivals and cancellation are still drained at every tick.

use memra_engine::prime_walker::{PrimeProgress, PrimeYieldMode};
use std::time::{Duration, Instant};

/// Structural route support. A bounded route still needs source/model/topology
/// native qualification; this enum is not a receipt or a default selection.
pub enum PrimeRoute {
    SavedWalker,
    TickBounded,
    Unsupported(&'static str),
}

impl PrimeRoute {
    pub fn scheduler(batching: bool) -> Self {
        if batching {
            Self::TickBounded
        } else {
            Self::Unsupported(
                "the legacy scheduler has no saved plain-prime adapter; requires MEMRA_SERVE_BATCH=1",
            )
        }
    }

    pub fn mtp(batching: bool, walker_supported: bool) -> Self {
        if !batching {
            Self::scheduler(false)
        } else if walker_supported {
            Self::SavedWalker
        } else {
            Self::Unsupported("selected MTP plan/topology has no saved prime walker")
        }
    }

    /// Default ON applies where the selected route has a worker boundary.
    /// Unsupported routes retain their existing program unless ON was explicit.
    pub fn cooperative_enabled(self, mode: PrimeYieldMode) -> Result<bool, String> {
        match self {
            Self::Unsupported(reason) if mode == PrimeYieldMode::ExplicitOn => {
                Err(format!("MEMRA_PRIME_YIELD=1 unsupported: {reason}"))
            }
            Self::Unsupported(_) => Ok(false),
            Self::SavedWalker | Self::TickBounded => Ok(mode.enabled()),
        }
    }

    pub fn require_cooperative(self, mode: PrimeYieldMode) -> Result<(), String> {
        self.cooperative_enabled(mode).map(|_| ())
    }
}

/// A scheduler observation, not ownership of the peer or a promise of emission.
pub struct DecodeReadiness {
    pub primed: bool,
    pub queued_rows: usize,
    pub remaining_budget: usize,
    pub channel_open: bool,
    pub output_ready: bool,
}

impl DecodeReadiness {
    pub fn can_advance(&self) -> bool {
        self.primed
            && self.queued_rows == 0
            && self.remaining_budget > 0
            && self.channel_open
            && self.output_ready
    }
}

#[derive(Default)]
pub struct PrimeService {
    pub pending: bool,
    pub peer_quantum: bool,
    pub yields: u64,
    last_advance: Option<(Instant, Duration)>,
}

impl PrimeService {
    /// All route adapters use this driver for both OFF and ON.
    pub fn advance<W: memra_engine::prime_walker::PrimeWalker>(
        &mut self,
        walker: &mut W,
    ) -> Result<bool, memra_engine::prime_walker::PrimeError> {
        // A partially mutated failing chunk must never become a park candidate.
        self.pending = walker.remaining_chunks() > 0;
        let progress = memra_engine::prime_walker::advance_prime(
            walker,
            memra_engine::prime_walker::prime_yield_enabled(),
            memra_engine::prime_walker::trace_chunk,
        )?;
        self.record(progress);
        Ok(!self.pending)
    }

    /// Finalization can still fail (for example a final draft ingestion). Keep its
    /// owned state out of reuse pools until that last fallible operation succeeds.
    pub fn finish<W: memra_engine::prime_walker::PrimeWalker>(
        &mut self,
        walker: W,
    ) -> Result<W::Output, memra_engine::prime_walker::PrimeError> {
        self.pending = true;
        let out = memra_engine::prime_walker::finish_prime(walker)?;
        self.pending = false;
        Ok(out)
    }

    /// Called by every adapter after the generic engine walker returns.
    pub fn record(&mut self, progress: PrimeProgress) {
        self.record_at(progress, Instant::now());
    }

    fn record_at(&mut self, progress: PrimeProgress, completed_at: Instant) {
        self.pending = progress.remaining_chunks > 0;
        if progress.advanced_chunks > 0 && self.pending {
            self.last_advance = Some((completed_at, progress.wall));
            self.yields += 1;
            eprintln!(
                "[prime-yield] count={} remaining={} chunks={} wall_ms={:.3} max_chunk_ms={:.3}",
                self.yields,
                progress.remaining_chunks,
                progress.advanced_chunks,
                progress.wall.as_secs_f64() * 1000.0,
                progress.max_chunk_wall.as_secs_f64() * 1000.0,
            );
        } else if !self.pending {
            self.last_advance = None;
        }
    }

    /// A target of one stops at the first public progress boundary: the initial
    /// seed or at most one committed spec round. Preserve a round's surplus tokens;
    /// this is a cadence bound, never a truncation of committed state.
    pub fn decode_target(&self, usual: usize) -> usize {
        if self.peer_quantum {
            usual.min(1)
        } else {
            usual
        }
    }
}

pub struct PrimePolicy {
    rotation: usize,
    service_goal: Duration,
    first_token_preference_since: Option<Instant>,
}

impl Default for PrimePolicy {
    fn default() -> Self {
        Self {
            rotation: 0,
            service_goal: Duration::from_millis(50),
            first_token_preference_since: None,
        }
    }
}

impl PrimePolicy {
    pub fn for_scheduler(
        batching: bool,
        mode: PrimeYieldMode,
        slo_ms: f32,
    ) -> Result<Self, String> {
        if PrimeRoute::scheduler(batching).cooperative_enabled(mode)? {
            Self::from_slo_ms(slo_ms).map_err(str::to_owned)
        } else {
            Ok(Self::default())
        }
    }

    pub fn from_slo_ms(ms: f32) -> Result<Self, &'static str> {
        let goal = Duration::try_from_secs_f64(f64::from(ms) / 1000.0)
            .map_err(|_| "cooperative prefill requires a finite positive MEMRA_SLO_P99_MS")?;
        if goal.is_zero() {
            return Err("cooperative prefill requires a finite positive MEMRA_SLO_P99_MS");
        }
        Ok(Self {
            service_goal: goal,
            ..Self::default()
        })
    }

    /// A chunk of cost C above service goal S permits ready peers to run for
    /// C-S before any saved prime advances again. `primes` is every live row of
    /// the worker; only pending ones count. The interval is worker-wide: with m
    /// saved primes, one chunk runs per interval instead of m chunks per tick,
    /// so peers keep C-S of service after every chunk and the primes together
    /// keep at least C of every 2C. Each peer still takes one public progress
    /// quantum per worker tick, so this never introduces a long burst.
    ///
    /// When the interval closes, the least recently served pending prime has the
    /// first claim for one further S. A claimant whose route does not run in that
    /// window (another phase, a held batch) delays the others by at most S; after
    /// it any pending prime may advance, so no prime waits on another forever.
    /// Only a prime's own advance opens an interval: arrivals and peer steps never
    /// renew it. A newly admitted prime is not pending before its first chunk (its
    /// walker does not exist yet), so each admission may run that one chunk inside
    /// an open interval; the chunk then opens its own. No ready peer means no delay
    /// (including after peer cancellation),
    /// and chunks within S keep the per-advance program: every prime, every tick.
    ///
    /// This is service sharing, not a hard ITL bound: a frozen chunk cannot be
    /// preempted and its cost may exceed S. Native gates must measure that cost,
    /// peer gaps and the admitted prime's progress separately.
    pub fn defer_for_peer<'a>(
        &self,
        owner: usize,
        primes: impl IntoIterator<Item = (usize, &'a PrimeService)>,
        has_ready_peer: bool,
        now: Instant,
    ) -> bool {
        if !has_ready_peer {
            return false;
        }
        let mut owner_pending = false;
        let mut recovery_end: Option<Instant> = None;
        let mut first_claim: Option<(Option<Instant>, usize)> = None;
        for (row, service) in primes {
            if !service.pending {
                continue;
            }
            owner_pending |= row == owner;
            let claim = (
                service.last_advance.map(|(completed_at, _)| completed_at),
                row,
            );
            if first_claim.is_none_or(|best| claim < best) {
                first_claim = Some(claim);
            }
            if let Some((completed_at, wall)) = service.last_advance
                && wall > self.service_goal
            {
                recovery_end = recovery_end.max(Some(completed_at + (wall - self.service_goal)));
            }
        }
        let Some(end) = recovery_end.filter(|_| owner_pending) else {
            return false;
        };
        now < end
            || (now < end + self.service_goal && first_claim.is_some_and(|(_, row)| row != owner))
    }

    /// A stream of cache hits must not continuously renew the first-token fence
    /// ahead of an already admitted prime. At most one service-goal interval is
    /// reserved for first tokens; then the existing prefill phase gets a turn.
    /// No prefill work means there is no debt to carry into a later request.
    pub fn defer_prefill_for_first_token(
        &mut self,
        requested: bool,
        has_prefill: bool,
        now: Instant,
    ) -> bool {
        if !requested || !has_prefill {
            self.first_token_preference_since = None;
            return false;
        }
        let since = *self.first_token_preference_since.get_or_insert(now);
        if now.saturating_duration_since(since) < self.service_goal {
            true
        } else {
            self.first_token_preference_since = None;
            false
        }
    }

    /// Reorders only already-selected spec rows, never changes a batch partition.
    /// Every row occurs once. New arrivals were admitted at tick top. A pending
    /// prime defers only inside the worker-wide recovery interval and its first
    /// claim window; `defer_for_peer` owns that choice, not this order.
    pub fn order(&mut self, order: &mut [usize], pending: impl Fn(usize) -> bool) {
        if !order.iter().copied().any(&pending) {
            return;
        }
        let count = order.len();
        if count == 0 {
            return;
        }
        order.rotate_left(self.rotation % count);
        // Fresh peers first, then saved primes. Stable within both classes, rotating
        // each tick. A small one-chunk prime can finalize and emit in its same turn.
        order.sort_by_key(|&i| pending(i));
        self.rotation = (self.rotation + 1) % count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defers(policy: &PrimePolicy, service: &PrimeService, ready: bool, now: Instant) -> bool {
        policy.defer_for_peer(0, [(0, service)], ready, now)
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn measured_chunk(service: &mut PrimeService, at: Instant, ms: u64, remaining: usize) {
        service.record_at(
            PrimeProgress {
                advanced_chunks: 1,
                remaining_chunks: remaining,
                wall: Duration::from_millis(ms),
                max_chunk_wall: Duration::from_millis(ms),
            },
            at,
        );
    }

    #[test]
    fn recovery_uses_measured_cost_and_configured_service_goal() {
        let now = Instant::now();
        let mut long = PrimeService::default();
        measured_chunk(&mut long, now, 500, 10);
        let tight = PrimePolicy::from_slo_ms(50.0).unwrap();
        let loose = PrimePolicy::from_slo_ms(200.0).unwrap();
        assert!(defers(
            &tight,
            &long,
            true,
            now + Duration::from_millis(449)
        ));
        assert!(!defers(
            &tight,
            &long,
            true,
            now + Duration::from_millis(450)
        ));
        assert!(!defers(
            &loose,
            &long,
            true,
            now + Duration::from_millis(300)
        ));

        measured_chunk(&mut long, now, 20, 9);
        assert!(!defers(&tight, &long, true, now));
    }

    #[test]
    fn no_ready_peer_or_completed_prime_never_waits() {
        let now = Instant::now();
        let policy = PrimePolicy::default();
        let mut long = PrimeService::default();
        assert!(!defers(&policy, &long, true, now));
        measured_chunk(&mut long, now, 1000, 2);
        // Includes solo work, all peers still priming, and a decode peer that
        // disconnected or exhausted its output budget between ticks.
        assert!(!defers(&policy, &long, false, now));
        measured_chunk(&mut long, now, 1000, 0);
        assert!(!defers(&policy, &long, true, now));
    }

    #[test]
    fn non_runnable_peers_do_not_buy_a_recovery_interval() {
        let policy = PrimePolicy::default();
        let now = Instant::now();
        let mut long = PrimeService::default();
        measured_chunk(&mut long, now, 1000, 3);
        let ready = || DecodeReadiness {
            primed: true,
            queued_rows: 0,
            remaining_budget: 10,
            channel_open: true,
            output_ready: true,
        };
        assert!(defers(&policy, &long, ready().can_advance(), now));
        let peers = [
            DecodeReadiness {
                channel_open: false,
                ..ready()
            },
            DecodeReadiness {
                remaining_budget: 0,
                ..ready()
            },
            DecodeReadiness {
                primed: false,
                ..ready()
            },
            DecodeReadiness {
                queued_rows: 1,
                ..ready()
            },
            DecodeReadiness {
                output_ready: false,
                ..ready()
            },
        ];
        assert!(!defers(
            &policy,
            &long,
            peers.iter().any(DecodeReadiness::can_advance),
            now
        ));
    }

    #[test]
    fn continuous_arrivals_cannot_extend_an_owners_deadline() {
        let policy = PrimePolicy::default();
        let mut long = PrimeService::default();
        let start = Instant::now();
        let mut now = start;
        let mut peer_quanta = 0;
        // A 128k tape frozen in 1024-row chunks. The test measures scheduling,
        // not GPU speed: simulated chunks cost 500ms and peer steps cost 10ms.
        for remaining in (1..128).rev() {
            measured_chunk(&mut long, now, 500, remaining);
            let deadline = now + Duration::from_millis(450);
            while defers(&policy, &long, true, now) {
                // Each tick may contain a different newly arrived ready peer.
                peer_quanta += 1;
                now += Duration::from_millis(10);
            }
            assert_eq!(now, deadline);
        }
        assert_eq!(peer_quanta, 127 * 45);
        // Repeated readiness checks never rewrite last_advance.
        assert_eq!(now.duration_since(start), Duration::from_millis(127 * 450));
    }

    #[test]
    fn concurrent_primes_share_one_recovery_interval() {
        let policy = PrimePolicy::default();
        let t0 = Instant::now();
        let (mut a, mut b) = (PrimeService::default(), PrimeService::default());
        // One tick advanced A (done at 500ms) and then B (done at 1000ms). B's chunk
        // already spent A's own interval, yet peers are still owed C-S after it.
        measured_chunk(&mut a, t0 + ms(500), 500, 10);
        measured_chunk(&mut b, t0 + ms(1000), 500, 10);
        let rows = || [(0, &a), (1, &b)];
        for now in [t0 + ms(1000), t0 + ms(1449)] {
            assert!(policy.defer_for_peer(0, rows(), true, now));
            assert!(policy.defer_for_peer(1, rows(), true, now));
        }
        // A was served least recently, so it has the first claim; B waits one goal.
        let closed = t0 + ms(1450);
        assert!(!policy.defer_for_peer(0, rows(), true, closed));
        assert!(policy.defer_for_peer(1, rows(), true, closed));
        assert!(!policy.defer_for_peer(1, rows(), true, closed + ms(50)));
        // Without a ready peer both keep the per-advance program.
        assert!(!policy.defer_for_peer(0, rows(), false, t0 + ms(1000)));
        assert!(!policy.defer_for_peer(1, rows(), false, t0 + ms(1000)));
    }

    #[test]
    fn many_primes_alternate_with_peers_and_take_turns() {
        // Scheduling simulation, not GPU speed: chunks cost 500ms and a tick of
        // ready peers costs 10ms. Peers run first in a tick, then every prime row.
        let policy = PrimePolicy::default();
        let start = Instant::now();
        let mut now = start;
        let mut primes: Vec<PrimeService> = (0..3).map(|_| PrimeService::default()).collect();
        for prime in &mut primes {
            prime.pending = true;
        }
        let mut chunks = [0usize; 3];
        let mut peer_service: Option<Duration> = None;
        for _ in 0..20_000 {
            now += ms(10);
            if let Some(service) = peer_service.as_mut() {
                *service += ms(10);
            }
            for row in 0..3 {
                let rows: Vec<(usize, &PrimeService)> = primes.iter().enumerate().collect();
                if policy.defer_for_peer(row, rows, true, now) {
                    continue;
                }
                // Peers received C-S of service since the previous chunk, even though
                // three primes are pending and each would otherwise advance this tick.
                assert!(peer_service.is_none_or(|service| service >= ms(450)));
                now += ms(500);
                chunks[row] += 1;
                measured_chunk(&mut primes[row], now, 500, 1_000_000);
                peer_service = Some(Duration::ZERO);
            }
        }
        let (fewest, most) = (chunks.iter().min().unwrap(), chunks.iter().max().unwrap());
        assert!(most - fewest <= 1, "turns must rotate: {chunks:?}");
        // The primes together keep more than half of the worker.
        let prime_time = ms(500) * chunks.iter().sum::<usize>() as u32;
        assert!(prime_time * 2 > now.duration_since(start));
    }

    #[test]
    fn an_absent_first_claimant_delays_other_primes_by_at_most_one_goal() {
        let policy = PrimePolicy::default();
        let t0 = Instant::now();
        let (mut held, mut runnable) = (PrimeService::default(), PrimeService::default());
        measured_chunk(&mut held, t0, 500, 5);
        measured_chunk(&mut runnable, t0 + ms(500), 500, 5);
        let rows = || [(0, &held), (1, &runnable)];
        // Row 0 has the first claim but its route does not run (for example a held
        // dark batch). Row 1 is delayed by one service goal, never indefinitely.
        let closed = t0 + ms(950);
        assert!(policy.defer_for_peer(1, rows(), true, closed));
        assert!(policy.defer_for_peer(1, rows(), true, closed + ms(49)));
        assert!(!policy.defer_for_peer(1, rows(), true, closed + ms(50)));
    }

    #[test]
    fn chunks_within_the_goal_keep_the_per_advance_program() {
        let policy = PrimePolicy::default();
        let t0 = Instant::now();
        let (mut a, mut b) = (PrimeService::default(), PrimeService::default());
        measured_chunk(&mut a, t0, 50, 3);
        measured_chunk(&mut b, t0, 30, 3);
        let rows = || [(0, &a), (1, &b)];
        assert!(!policy.defer_for_peer(0, rows(), true, t0));
        assert!(!policy.defer_for_peer(1, rows(), true, t0));
    }

    #[test]
    fn retired_or_completed_primes_release_the_interval() {
        let policy = PrimePolicy::default();
        let t0 = Instant::now();
        let mut long = PrimeService::default();
        let other = PrimeService {
            pending: true,
            ..Default::default()
        };
        measured_chunk(&mut long, t0, 1000, 3);
        assert!(policy.defer_for_peer(1, [(0, &long), (1, &other)], true, t0));
        // The worker leaves finished rows out, so a cancelled owner holds nothing.
        assert!(!policy.defer_for_peer(1, [(1, &other)], true, t0));
        // A prime that ingested its last chunk no longer holds an interval either.
        measured_chunk(&mut long, t0, 1000, 0);
        assert!(!policy.defer_for_peer(1, [(0, &long), (1, &other)], true, t0));
    }

    #[test]
    fn endless_cached_arrivals_still_grant_initial_prefill_turns() {
        let mut policy = PrimePolicy::default();
        let start = Instant::now();
        let mut turns = Vec::new();
        for tick in 0..600 {
            let now = start + Duration::from_millis(tick * 10);
            if !policy.defer_prefill_for_first_token(true, true, now) {
                turns.push(tick);
            }
        }
        assert_eq!(turns.len(), 100);
        assert_eq!(turns[0], 5);
        assert!(turns.windows(2).all(|ticks| ticks[1] - ticks[0] == 6));
        assert!(!policy.defer_prefill_for_first_token(true, false, start));
        assert!(policy.defer_prefill_for_first_token(true, true, start));
        assert!(!policy.defer_prefill_for_first_token(false, true, start));
    }

    #[test]
    fn invalid_service_goals_refuse_instead_of_disabling_progress() {
        for ms in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
            assert!(PrimePolicy::from_slo_ms(ms).is_err(), "{ms}");
        }
        assert!(PrimePolicy::from_slo_ms(50.0).is_ok());
    }

    #[test]
    fn unsupported_policy_refuses_without_disabling_the_existing_serial_route() {
        for mode in [
            PrimeYieldMode::ImplicitOn,
            PrimeYieldMode::ExplicitOn,
            PrimeYieldMode::Off,
        ] {
            assert_eq!(
                PrimeRoute::SavedWalker.cooperative_enabled(mode).unwrap(),
                mode.enabled()
            );
            assert_eq!(
                PrimeRoute::TickBounded.cooperative_enabled(mode).unwrap(),
                mode.enabled()
            );
        }
        let route = || PrimeRoute::Unsupported("serial request loop has no worker return boundary");
        assert_eq!(
            route()
                .require_cooperative(PrimeYieldMode::ExplicitOn)
                .unwrap_err(),
            "MEMRA_PRIME_YIELD=1 unsupported: serial request loop has no worker return boundary"
        );
        assert!(
            !route()
                .cooperative_enabled(PrimeYieldMode::ImplicitOn)
                .unwrap()
        );
        assert!(!route().cooperative_enabled(PrimeYieldMode::Off).unwrap());
    }

    #[test]
    fn legacy_implicit_and_off_skip_cooperative_slo_validation() {
        for goal in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(PrimePolicy::for_scheduler(false, PrimeYieldMode::ImplicitOn, goal).is_ok());
            assert!(PrimePolicy::for_scheduler(false, PrimeYieldMode::Off, goal).is_ok());
            assert!(PrimePolicy::for_scheduler(true, PrimeYieldMode::Off, goal).is_ok());
            assert!(PrimePolicy::for_scheduler(true, PrimeYieldMode::ImplicitOn, goal).is_err());
        }
        assert!(PrimePolicy::for_scheduler(false, PrimeYieldMode::ExplicitOn, 50.0).is_err());
        assert!(PrimePolicy::for_scheduler(true, PrimeYieldMode::ExplicitOn, 50.0).is_ok());
        assert!(PrimePolicy::for_scheduler(true, PrimeYieldMode::ImplicitOn, 50.0).is_ok());
    }

    #[test]
    fn mtp_selection_requires_both_scheduler_and_walker_capability() {
        for batching in [false, true] {
            for supported in [false, true] {
                let route = || PrimeRoute::mtp(batching, supported);
                assert_eq!(
                    route()
                        .cooperative_enabled(PrimeYieldMode::ImplicitOn)
                        .unwrap(),
                    batching && supported
                );
                assert!(!route().cooperative_enabled(PrimeYieldMode::Off).unwrap());
                match route().cooperative_enabled(PrimeYieldMode::ExplicitOn) {
                    Ok(enabled) => assert!(enabled && batching && supported),
                    Err(_) => assert!(!batching || !supported),
                }
            }
        }
    }

    #[test]
    fn environment_mode_preserves_route_defaults() {
        const CASE: &str = "PRIME_POLICY_TEST_CASE";
        if let Ok(case) = std::env::var(CASE) {
            let expected = match case.as_str() {
                "unset" => PrimeYieldMode::ImplicitOn,
                "on" => PrimeYieldMode::ExplicitOn,
                "off" => PrimeYieldMode::Off,
                _ => panic!("unexpected test case"),
            };
            let actual = memra_engine::prime_walker::prime_yield_mode();
            assert_eq!(actual, expected);
            assert_eq!(
                memra_engine::prime_walker::prime_yield_enabled(),
                expected.enabled()
            );
            assert_eq!(
                PrimeRoute::SavedWalker.cooperative_enabled(actual).unwrap(),
                expected.enabled()
            );
            assert_eq!(
                PrimeRoute::Unsupported("serial route")
                    .require_cooperative(actual)
                    .is_err(),
                expected == PrimeYieldMode::ExplicitOn
            );
            let legacy = PrimeRoute::mtp(false, true).cooperative_enabled(actual);
            if expected == PrimeYieldMode::ExplicitOn {
                assert!(legacy.is_err());
            } else {
                assert!(!legacy.unwrap());
            }
            return;
        }
        // Separate processes exercise the actual environment read without mutating
        // process-global environment or the OnceLock under parallel Rust tests.
        for (case, value) in [("unset", None), ("on", Some("1")), ("off", Some("0"))] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command.args([
                "--exact",
                "prime_fairness::tests::environment_mode_preserves_route_defaults",
            ]);
            command.env(CASE, case).env_remove("MEMRA_PRIME_YIELD");
            if let Some(value) = value {
                command.env("MEMRA_PRIME_YIELD", value);
            }
            let output = command.output().unwrap();
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed;"),
                "{case}: subprocess must execute exactly the requested test"
            );
            assert!(
                output.status.success(),
                "{case}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn serving_guards_use_request_mode_and_legacy_mtp_uses_route_policy() {
        let source = include_str!("worker.rs");
        let production = source.split("#[cfg(test)]\nmod tests").next().unwrap();
        assert_eq!(
            production
                .matches(".require_cooperative(memra_engine::prime_walker::prime_yield_mode())")
                .count(),
            3
        );
        assert!(
            !production.contains(
                ".require_cooperative(memra_engine::prime_walker::prime_yield_enabled())"
            )
        );
        assert!(production.contains("PrimePolicy::for_scheduler(\n        serve_batching(),\n        memra_engine::prime_walker::prime_yield_mode(),"));
        assert!(production.contains("PrimeRoute::mtp(\n                serve_batching(),\n                lm.model.mtp_prime_walk_supported(),"));
        let dsv4 = include_str!("dsv4_serve.rs");
        assert!(
            dsv4.contains(".require_cooperative(memra_engine::prime_walker::prime_yield_mode())?;")
        );
        assert!(
            !dsv4.contains(
                ".require_cooperative(memra_engine::prime_walker::prime_yield_enabled())"
            )
        );
    }

    #[test]
    fn failed_advance_or_final_ingestion_never_makes_a_park_boundary() {
        struct Fails {
            remaining: usize,
        }
        impl memra_engine::prime_walker::PrimeWalker for Fails {
            type Output = ();
            fn remaining_chunks(&self) -> usize {
                self.remaining
            }
            fn advance_chunk(
                &mut self,
            ) -> Result<
                memra_engine::prime_walker::PrimeChunk,
                memra_engine::prime_walker::PrimeError,
            > {
                Err("partial chunk failure".into())
            }
            fn finish(self) -> Result<(), memra_engine::prime_walker::PrimeError> {
                Err("final ingestion failure".into())
            }
        }
        let mut service = PrimeService::default();
        assert!(service.advance(&mut Fails { remaining: 1 }).is_err());
        assert!(service.pending);
        let mut service = PrimeService::default();
        assert!(service.finish(Fails { remaining: 0 }).is_err());
        assert!(service.pending);
    }

    #[test]
    fn new_peer_precedes_resumed_prime_and_neither_can_starve() {
        let mut policy = PrimePolicy::default();
        let mut long_remaining = 100;
        for arrival in 1..101 {
            let mut order = vec![0, arrival];
            policy.order(&mut order, |i| i == 0);
            assert_eq!(order, [arrival, 0]);
            long_remaining -= 1;
        }
        assert_eq!(long_remaining, 0);
    }

    #[test]
    fn peers_rotate_without_duplicates_or_extra_quanta() {
        let mut policy = PrimePolicy::default();
        let mut heads = Vec::new();
        for _ in 0..4 {
            let mut order = vec![0, 1, 2, 3];
            policy.order(&mut order, |_| true);
            heads.push(order[0]);
            order.sort_unstable();
            assert_eq!(order, [0, 1, 2, 3]);
        }
        assert_eq!(heads, [0, 1, 2, 3]);
    }

    #[test]
    fn no_pending_prime_preserves_existing_order_and_burst() {
        let mut order = vec![3, 1, 2];
        PrimePolicy::default().order(&mut order, |_| false);
        assert_eq!(order, [3, 1, 2]);
        let mut service = PrimeService::default();
        assert_eq!(service.decode_target(32), 32);
        service.peer_quantum = true;
        assert_eq!(service.decode_target(32), 1);
        assert_eq!(service.decode_target(0), 0);
    }
}
