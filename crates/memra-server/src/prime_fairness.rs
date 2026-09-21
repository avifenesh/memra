//! Route-independent policy around the worker's existing tick-top drain/admission.
//! Saved primes yield at their frozen numerical boundaries. An expensive chunk
//! grants ready decode peers a recovery interval across ordinary worker ticks;
//! arrivals and cancellation are still drained at every tick.

use memra_engine::prime_walker::PrimeProgress;
use std::time::{Duration, Instant};

/// Structural route support. A bounded route still needs source/model/topology
/// native qualification; this enum is not a receipt or a default selection.
pub enum PrimeRoute {
    SavedWalker,
    TickBounded,
    Unsupported(&'static str),
}

impl PrimeRoute {
    pub fn require_cooperative(self, enabled: bool) -> Result<(), String> {
        if enabled && let Self::Unsupported(reason) = self {
            Err(format!("MEMRA_PRIME_YIELD=1 unsupported: {reason}"))
        } else {
            Ok(())
        }
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
    /// C-S before this owner advances again. Each peer still takes one public
    /// progress quantum per worker tick, so this never introduces a long burst.
    /// The deadline belongs to the owner's last advance: other requests cannot
    /// renew it. No ready peer means no delay (including after peer cancellation).
    ///
    /// This is service sharing, not a hard ITL bound: a frozen chunk cannot be
    /// preempted and its cost may exceed S. Native gates must measure that cost,
    /// peer gaps and the admitted prime's progress separately.
    pub fn defer_for_peer(
        &self,
        service: &PrimeService,
        has_ready_peer: bool,
        now: Instant,
    ) -> bool {
        service.pending
            && has_ready_peer
            && service.last_advance.is_some_and(|(completed_at, wall)| {
                now.saturating_duration_since(completed_at) < wall.saturating_sub(self.service_goal)
            })
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
    /// prime can defer only until its own fixed recovery interval expires.
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
        assert!(tight.defer_for_peer(&long, true, now + Duration::from_millis(449)));
        assert!(!tight.defer_for_peer(&long, true, now + Duration::from_millis(450)));
        assert!(!loose.defer_for_peer(&long, true, now + Duration::from_millis(300)));

        measured_chunk(&mut long, now, 20, 9);
        assert!(!tight.defer_for_peer(&long, true, now));
    }

    #[test]
    fn no_ready_peer_or_completed_prime_never_waits() {
        let now = Instant::now();
        let policy = PrimePolicy::default();
        let mut long = PrimeService::default();
        assert!(!policy.defer_for_peer(&long, true, now));
        measured_chunk(&mut long, now, 1000, 2);
        // Includes solo work, all peers still priming, and a decode peer that
        // disconnected or exhausted its output budget between ticks.
        assert!(!policy.defer_for_peer(&long, false, now));
        measured_chunk(&mut long, now, 1000, 0);
        assert!(!policy.defer_for_peer(&long, true, now));
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
        assert!(policy.defer_for_peer(&long, ready().can_advance(), now));
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
        assert!(!policy.defer_for_peer(&long, peers.iter().any(DecodeReadiness::can_advance), now));
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
            while policy.defer_for_peer(&long, true, now) {
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
        assert!(PrimeRoute::SavedWalker.require_cooperative(true).is_ok());
        assert!(PrimeRoute::TickBounded.require_cooperative(true).is_ok());
        let route = || PrimeRoute::Unsupported("serial request loop has no worker return boundary");
        assert_eq!(
            route().require_cooperative(true).unwrap_err(),
            "MEMRA_PRIME_YIELD=1 unsupported: serial request loop has no worker return boundary"
        );
        assert!(route().require_cooperative(false).is_ok());
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
