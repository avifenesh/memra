//! Cooperative prime execution. A route owns its caches, frozen ranges, captures and
//! partial draft batches. Both serving arms drive the same `advance_chunk` operation.
//! No CUDA work, thread, numeric policy or route selection lives in this utility.

use std::time::{Duration, Instant};

pub type PrimeError = Box<dyn std::error::Error>;

/// One existing numerical chunk. Phase names are static diagnostics, never request data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimeChunk {
    pub phase: &'static str,
    pub rows: usize,
}

/// An adapter owns all state needed between chunks, including any pending 256-row
/// ingestion carry. Freeze its range list before the first advance. Each successful
/// advance consumes exactly one range and leaves no live borrow of shared scratch.
/// It must fence any device work whose buffers a peer could reuse before returning.
/// Finalization consumes the state and must reject an incomplete prime. It may perform
/// once-only boundary sampling/publication, but must not hide another prompt-length loop.
pub trait PrimeWalker {
    type Output;

    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError>;
    fn remaining_chunks(&self) -> usize;
    fn finish(self) -> Result<Self::Output, PrimeError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimeProgress {
    pub advanced_chunks: usize,
    pub remaining_chunks: usize,
    pub wall: Duration,
    pub max_chunk_wall: Duration,
}

/// The one door read shared by all adapters. Default OFF until route receipts qualify it.
pub fn prime_yield_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_PRIME_YIELD").as_deref() == Ok("1"))
}

/// OFF drains the frozen tape; ON executes one chunk. The observer sees EVERY actual
/// chunk in both arms, so chunk-wall receipts never infer cost from whole-request TTFT.
pub fn advance_prime<W: PrimeWalker>(
    walker: &mut W,
    yield_after_chunk: bool,
    mut observe: impl FnMut(PrimeChunk, Duration),
) -> Result<PrimeProgress, PrimeError> {
    let mut progress = PrimeProgress::default();
    while walker.remaining_chunks() > 0 {
        let before = walker.remaining_chunks();
        let start = Instant::now();
        let chunk = walker.advance_chunk()?;
        let wall = start.elapsed();
        if walker.remaining_chunks() != before - 1 {
            return Err("prime walker must consume exactly one frozen chunk per advance".into());
        }
        observe(chunk, wall);
        progress.advanced_chunks += 1;
        progress.wall += wall;
        progress.max_chunk_wall = progress.max_chunk_wall.max(wall);
        if yield_after_chunk {
            break;
        }
    }
    progress.remaining_chunks = walker.remaining_chunks();
    Ok(progress)
}

pub fn finish_prime<W: PrimeWalker>(walker: W) -> Result<W::Output, PrimeError> {
    if walker.remaining_chunks() != 0 {
        return Err("cannot finalize an incomplete prime".into());
    }
    walker.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        ranges: Vec<std::ops::Range<usize>>,
        cursor: usize,
        tape: Vec<usize>,
        fail_at: Option<usize>,
    }

    impl Fake {
        fn new() -> Self {
            Self {
                // Non-divisible tail and a capture boundary inside an ingestion batch.
                ranges: vec![0..1024, 1024..1153, 1153..2048, 2048..2071],
                cursor: 0,
                tape: Vec::new(),
                fail_at: None,
            }
        }
    }

    impl PrimeWalker for Fake {
        type Output = Vec<usize>;

        fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
            if self.fail_at == Some(self.cursor) {
                return Err("fake chunk failure".into());
            }
            let range = self.ranges.get(self.cursor).ok_or("prime exhausted")?;
            self.tape.extend(range.clone());
            self.cursor += 1;
            Ok(PrimeChunk {
                phase: "fake",
                rows: range.len(),
            })
        }

        fn remaining_chunks(&self) -> usize {
            self.ranges.len() - self.cursor
        }

        fn finish(self) -> Result<Self::Output, PrimeError> {
            Ok(self.tape)
        }
    }

    #[test]
    fn yielded_and_drained_execute_identical_frozen_tapes() {
        let mut off = Fake::new();
        let mut on = Fake::new();
        let mut off_rows = Vec::new();
        let mut on_rows = Vec::new();
        let drained = advance_prime(&mut off, false, |c, _| off_rows.push(c.rows)).unwrap();
        assert_eq!(drained.advanced_chunks, 4);
        while on.remaining_chunks() > 0 {
            let progress = advance_prime(&mut on, true, |c, _| on_rows.push(c.rows)).unwrap();
            assert_eq!(progress.advanced_chunks, 1);
        }
        assert_eq!(off_rows, on_rows);
        assert_eq!(finish_prime(off).unwrap(), finish_prime(on).unwrap());
    }

    #[test]
    fn error_does_not_advance_or_finalize_the_partial_state() {
        let mut walker = Fake::new();
        walker.fail_at = Some(1);
        assert!(advance_prime(&mut walker, false, |_, _| {}).is_err());
        assert_eq!(walker.remaining_chunks(), 3);
        assert_eq!(walker.tape, (0..1024).collect::<Vec<_>>());
        assert!(finish_prime(walker).is_err());
    }

    #[test]
    fn finished_walker_does_no_more_work() {
        let mut walker = Fake::new();
        advance_prime(&mut walker, false, |_, _| {}).unwrap();
        let progress = advance_prime(&mut walker, true, |_, _| panic!("extra chunk")).unwrap();
        assert_eq!(progress.advanced_chunks, 0);
        assert_eq!(finish_prime(walker).unwrap().len(), 2071);
    }
}
