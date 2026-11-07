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
/// advance consumes one range (or, through `advance_chunks`, adjacent ranges run as one exact
/// call) and leaves no live borrow of shared scratch.
/// It must fence any device work whose buffers a peer could reuse before returning.
/// Finalization consumes the state and must reject an incomplete prime. It may perform
/// once-only boundary sampling/publication, but must not hide another prompt-length loop.
pub trait PrimeWalker {
    type Output;

    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError>;
    fn remaining_chunks(&self) -> usize;
    fn finish(self) -> Result<Self::Output, PrimeError>;

    /// One advance that may run up to `max_ranges` adjacent frozen ranges as one numerical
    /// call, for a walker whose trunk computes the same bits at any range split (the serial
    /// GDN hybrid trunk at GDN-grid-aligned boundaries). The default runs one range. The driver
    /// passes more than one only when no peer is admitted or waiting, so a wider call never
    /// lengthens a peer's wait.
    fn advance_chunks(&mut self, max_ranges: usize) -> Result<PrimeChunk, PrimeError> {
        let _ = max_ranges;
        self.advance_chunk()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimeProgress {
    pub advanced_chunks: usize,
    pub remaining_chunks: usize,
    pub wall: Duration,
    pub max_chunk_wall: Duration,
}

/// The one door read shared by all adapters. Default ON since 2026-09-22 (memra#521): an owned
/// walker advances one frozen chunk per tick and the worker returns to its peers between chunks.
/// `MEMRA_PRIME_YIELD=0` is the rollback seam (drain the same frozen tape in one call); both arms
/// execute the same range program, so the seam changes interleaving, never bytes. Receipts:
/// `research/prefill-fairness-20260908/` (three independent 5090 positives on the MTP route),
/// `research/prime-fairness-default-20260922/` (the decision cell on the 5090 and a PRO 6000).
pub fn prime_yield_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_PRIME_YIELD").as_deref() != Ok("0"))
}

/// Shared observer used by synchronous engine callers and cooperative serving adapters.
/// Existing tick tracing enables measurement in both arms without a second door.
pub fn trace_chunk(chunk: PrimeChunk, wall: Duration) {
    if std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "[prime-chunk] phase={} rows={} wall_ms={:.3}",
            chunk.phase,
            chunk.rows,
            wall.as_secs_f64() * 1000.0
        );
    }
}

/// OFF drains the frozen tape; ON executes one chunk. The observer sees EVERY actual
/// chunk in both arms, so chunk-wall receipts never infer cost from whole-request TTFT.
pub fn advance_prime<W: PrimeWalker>(
    walker: &mut W,
    yield_after_chunk: bool,
    observe: impl FnMut(PrimeChunk, Duration),
) -> Result<PrimeProgress, PrimeError> {
    advance_prime_wide(walker, yield_after_chunk, 1, observe)
}

/// [`advance_prime`] where each advance may merge up to `max_ranges` adjacent frozen ranges
/// (`PrimeWalker::advance_chunks`). `max_ranges` 1 is `advance_prime` exactly.
pub fn advance_prime_wide<W: PrimeWalker>(
    walker: &mut W,
    yield_after_chunk: bool,
    max_ranges: usize,
    mut observe: impl FnMut(PrimeChunk, Duration),
) -> Result<PrimeProgress, PrimeError> {
    let max_ranges = max_ranges.max(1);
    let mut progress = PrimeProgress::default();
    while walker.remaining_chunks() > 0 {
        let before = walker.remaining_chunks();
        let start = Instant::now();
        let chunk = if max_ranges > 1 {
            walker.advance_chunks(max_ranges)?
        } else {
            walker.advance_chunk()?
        };
        let wall = start.elapsed();
        let consumed = before.saturating_sub(walker.remaining_chunks());
        if consumed == 0 || consumed > max_ranges || walker.remaining_chunks() > before {
            return Err(if max_ranges == 1 {
                "prime walker must consume exactly one frozen chunk per advance".into()
            } else {
                "prime walker must consume between one and max_ranges frozen chunks per advance"
                    .into()
            });
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
    let start = Instant::now();
    let result = walker.finish();
    if std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1") {
        eprintln!(
            "[prime-finalize] walker={} wall_ms={:.3} success={}",
            std::any::type_name::<W>(),
            start.elapsed().as_secs_f64() * 1000.0,
            result.is_ok()
        );
    }
    result
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
