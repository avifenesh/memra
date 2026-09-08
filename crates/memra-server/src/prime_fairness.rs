//! Route-independent policy around the worker's existing tick-top drain/admission.
//! A prime advances once per tick. Its peers get at most one service quantum before
//! its next advance; the next tick drains arrivals before resuming the owner.

use memra_engine::prime_walker::PrimeProgress;

#[derive(Default)]
pub struct PrimeService {
    pub pending: bool,
    pub peer_quantum: bool,
    pub yields: u64,
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
        self.pending = progress.remaining_chunks > 0;
        if progress.advanced_chunks > 0 && self.pending {
            self.yields += 1;
            eprintln!(
                "[prime-yield] count={} remaining={} chunks={} wall_ms={:.3} max_chunk_ms={:.3}",
                self.yields,
                progress.remaining_chunks,
                progress.advanced_chunks,
                progress.wall.as_secs_f64() * 1000.0,
                progress.max_chunk_wall.as_secs_f64() * 1000.0,
            );
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

#[derive(Default)]
pub struct PrimePolicy {
    rotation: usize,
}

impl PrimePolicy {
    /// Reorders only already-selected spec rows, never changes a batch partition.
    /// Every row occurs once. New arrivals were admitted at tick top; all primes
    /// still get an advance even under an unending arrival stream.
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
