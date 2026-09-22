//! Diagnostic observations around actual saved-walker operations. The wrapper
//! does not select, resize, skip or finalize a chunk on the scheduler's behalf.

use memra_engine::prime_walker::{PrimeChunk, PrimeError, PrimeWalker};

use crate::ttft::Trace;

#[derive(Clone, Copy)]
pub(crate) enum PrimeScope {
    /// This walker contains every remaining prime chunk for the request.
    RemainingPrompt,
    /// A plain-prime segment ends before the request's remaining queued input.
    /// Later segment geometry is not frozen yet, so the total count is unknown.
    SegmentWithContinuation,
}

pub(crate) struct ObservedPrime<'a, W> {
    walker: W,
    trace: Option<&'a Trace>,
    scope: PrimeScope,
    final_quantum_open: bool,
}

pub(crate) fn observe_prime<W>(
    walker: W,
    trace: Option<&Trace>,
    scope: PrimeScope,
) -> ObservedPrime<'_, W> {
    ObservedPrime {
        walker,
        trace,
        scope,
        final_quantum_open: false,
    }
}

/// One already-selected direct prefill call, including a one-token tail after a
/// saved segment. The supplied rows describe this call's actual input slice.
/// Future direct-call partitioning is unknown until the worker selects it.
pub(crate) fn observe_call<T>(
    trace: Option<&Trace>,
    chunk: PrimeChunk,
    scope: PrimeScope,
    call: impl FnOnce() -> Result<T, PrimeError>,
) -> Result<T, PrimeError> {
    let trace = trace.filter(|_| chunk.rows > 0);
    if let Some(trace) = trace {
        trace.mark_prime_quantum_start(chunk.phase, chunk.rows);
    }
    let result = call();
    if let Some(trace) = trace {
        if let Some(cancelled) = result
            .as_ref()
            .err()
            .and_then(|err| err.downcast_ref::<memra_engine::progress::PrimeCancelled>())
        {
            // Exact typed error only: message matching or an arbitrary Err must
            // never turn engine failure into a qualified cancellation boundary.
            trace.mark_prime_quantum_cancelled(
                cancelled.chunk,
                cancelled.rows_done,
                cancelled.rows_total,
            );
        } else {
            let remaining = match (result.is_ok(), scope) {
                (true, PrimeScope::RemainingPrompt) => Some(0),
                _ => None,
            };
            trace.mark_prime_quantum_end(result.is_ok(), remaining);
        }
    }
    result
}

impl<W: PrimeWalker> PrimeWalker for ObservedPrime<'_, W> {
    type Output = W::Output;

    fn next_chunk(&self) -> Option<PrimeChunk> {
        self.walker.next_chunk()
    }

    fn remaining_chunks(&self) -> usize {
        self.walker.remaining_chunks()
    }

    fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
        // A queue length, legacy TTFT mark or unknown/zero metadata is not proof
        // of a nonzero operation. No trace is opened for those shapes.
        let chunk = self
            .trace
            .and_then(|_| self.walker.next_chunk())
            .filter(|chunk| chunk.rows > 0);
        let observation = self.trace.zip(chunk);
        if let Some((trace, chunk)) = observation {
            trace.mark_prime_quantum_start(chunk.phase, chunk.rows);
        }
        let result = self.walker.advance_chunk();
        if let Some((trace, _)) = observation {
            let chunks = self.walker.remaining_chunks();
            if result.is_ok() && chunks == 0 {
                // The same worker turn must still finalize this last chunk.
                // DFlash, for example, flushes its bounded ingestion carry in
                // finish(). Keep this host quantum open until that return, so
                // neither Some(0) nor success can precede that fallible work.
                self.final_quantum_open = true;
                return result;
            }
            // After failure, partially consumed state is not a reliable count
            // of remaining work. A segment's future continuation is unknown too.
            let remaining = match (result.is_ok(), self.scope) {
                (true, PrimeScope::RemainingPrompt) => Some(chunks),
                _ => None,
            };
            trace.mark_prime_quantum_end(result.is_ok(), remaining);
        }
        result
    }

    fn finish(self) -> Result<Self::Output, PrimeError> {
        let result = self.walker.finish();
        if self.final_quantum_open
            && let Some(trace) = self.trace
        {
            let remaining = match (result.is_ok(), self.scope) {
                (true, PrimeScope::RemainingPrompt) => Some(0),
                _ => None,
            };
            trace.mark_prime_quantum_end(result.is_ok(), remaining);
        }
        // No quantum was opened for empty/cache-only state, so its finalization
        // cannot manufacture prime evidence. This also is not a resource-release
        // or GPU-quiescence claim; those have separate owner-side observations.
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttft::{ReceiverCloseCause, RetirementOutcome};
    use memra_engine::prime_walker::{advance_prime, finish_prime};
    use std::sync::Arc;

    struct Tape {
        rows: Vec<usize>,
        cursor: usize,
        exposed: bool,
        fail: bool,
        fail_finish: bool,
        trace: Option<Arc<Trace>>,
        http_drop_in_advance: bool,
    }

    impl Tape {
        fn new(rows: Vec<usize>) -> Self {
            Self {
                rows,
                cursor: 0,
                exposed: true,
                fail: false,
                fail_finish: false,
                trace: None,
                http_drop_in_advance: false,
            }
        }
    }

    impl PrimeWalker for Tape {
        type Output = Vec<usize>;

        fn next_chunk(&self) -> Option<PrimeChunk> {
            self.exposed.then_some(())?;
            self.rows.get(self.cursor).map(|&rows| PrimeChunk {
                phase: "frozen-fixture",
                rows,
            })
        }

        fn remaining_chunks(&self) -> usize {
            self.rows.len() - self.cursor
        }

        fn advance_chunk(&mut self) -> Result<PrimeChunk, PrimeError> {
            if let Some(trace) = &self.trace {
                let events = trace.observe_for_test().json_lines();
                assert!(events.last().unwrap().contains("\"quantum_active\":true"));
                if self.http_drop_in_advance {
                    trace.mark_http_pending_drop();
                }
            }
            let rows = self.rows[self.cursor];
            self.cursor += 1;
            if self.fail {
                // A partially changed cursor must not mint known completion.
                return Err("failed after changing state".into());
            }
            Ok(PrimeChunk {
                phase: "frozen-fixture",
                rows,
            })
        }

        fn finish(self) -> Result<Self::Output, PrimeError> {
            if self.fail_finish {
                Err("final ingestion failed".into())
            } else {
                Ok(self.rows)
            }
        }
    }

    fn trace() -> Arc<Trace> {
        let trace = Trace::for_test("/v1/completions");
        trace.bind_request("minted-prime-observer", "fixture-model");
        trace.bind_worker(521, "shared_gpu_worker");
        trace.mark_queued();
        trace
    }

    fn ends(trace: &Trace) -> Vec<String> {
        trace
            .observe_for_test()
            .json_lines()
            .into_iter()
            .filter(|line| line.contains("\"event\":\"prime_quantum_end\""))
            .collect()
    }

    #[test]
    fn actual_trace_wraps_each_frozen_chunk_in_both_driver_modes() {
        for yield_each in [false, true] {
            let trace = trace();
            let observer = trace.observe_for_test();
            let mut tape = Tape::new(vec![1024, 129, 895, 23]);
            tape.trace = Some(trace.clone());
            let mut walker = observe_prime(tape, Some(&trace), PrimeScope::RemainingPrompt);
            while walker.remaining_chunks() > 0 {
                advance_prime(&mut walker, yield_each, |_, _| {}).unwrap();
            }
            assert_eq!(finish_prime(walker).unwrap(), [1024, 129, 895, 23]);
            let records = ends(&trace);
            assert_eq!(records.len(), 4);
            for (i, line) in records.iter().enumerate() {
                assert!(line.contains("\"completed\":true"));
                assert!(line.contains(&format!("\"remaining_chunks\":{}", 3 - i)));
                assert!(line.contains("\"sequence_valid\":true"));
            }
            assert!(records[3].contains("\"phase\":\"prime_finished\""));
            trace.mark_first_decode();
            trace.mark_http_body_eof();
            trace.mark_http_body_drop();
            trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
            trace.mark_retired(RetirementOutcome::Aborted);
            drop(trace);
            let records = observer.json_lines();
            assert!(records.last().unwrap().contains("\"event\":\"trace_end\""));
            assert!(records.last().unwrap().contains("\"sequence_valid\":true"));
        }
    }

    #[test]
    fn partial_failure_closes_the_actual_quantum_without_claiming_completion() {
        let trace = trace();
        let mut tape = Tape::new(vec![17]);
        tape.fail = true;
        tape.trace = Some(trace.clone());
        let mut walker = observe_prime(tape, Some(&trace), PrimeScope::RemainingPrompt);
        assert!(advance_prime(&mut walker, true, |_, _| {}).is_err());
        assert_eq!(walker.remaining_chunks(), 0);
        let records = ends(&trace);
        assert_eq!(records.len(), 1);
        assert!(records[0].contains("\"completed\":false"));
        assert!(records[0].contains("\"remaining_chunks\":null"));
        assert!(records[0].contains("\"quantum_active\":false"));
        assert!(records[0].contains("\"phase\":\"prime\""));
    }

    #[test]
    fn final_ingestion_must_return_before_last_quantum_can_complete() {
        let trace = trace();
        let mut tape = Tape::new(vec![23]);
        tape.fail_finish = true;
        let mut walker = observe_prime(tape, Some(&trace), PrimeScope::RemainingPrompt);
        advance_prime(&mut walker, true, |_, _| {}).unwrap();
        assert!(ends(&trace).is_empty());
        assert!(
            trace
                .observe_for_test()
                .json_lines()
                .last()
                .unwrap()
                .contains("\"quantum_active\":true")
        );
        assert_eq!(
            finish_prime(walker).unwrap_err().to_string(),
            "final ingestion failed"
        );
        let records = ends(&trace);
        assert_eq!(records.len(), 1);
        assert!(records[0].contains("\"completed\":false"));
        assert!(records[0].contains("\"remaining_chunks\":null"));
        assert!(records[0].contains("\"phase\":\"prime\""));
    }

    #[test]
    fn later_plain_segment_is_not_a_finished_prompt() {
        let trace = trace();
        let mut first = observe_prime(
            Tape::new(vec![1024]),
            Some(&trace),
            PrimeScope::SegmentWithContinuation,
        );
        advance_prime(&mut first, true, |_, _| {}).unwrap();
        finish_prime(first).unwrap();
        let first_end = ends(&trace).pop().unwrap();
        assert!(first_end.contains("\"remaining_chunks\":null"));
        assert!(first_end.contains("\"phase\":\"prime\""));
        let mut last = observe_prime(
            Tape::new(vec![128]),
            Some(&trace),
            PrimeScope::RemainingPrompt,
        );
        advance_prime(&mut last, true, |_, _| {}).unwrap();
        finish_prime(last).unwrap();
        let last_end = ends(&trace).pop().unwrap();
        assert!(last_end.contains("\"remaining_chunks\":0"));
        assert!(last_end.contains("\"phase\":\"prime_finished\""));
        assert!(last_end.contains("\"sequence_valid\":true"));
    }

    #[test]
    fn actual_one_token_tail_closes_a_saved_segment_without_a_new_walker() {
        let trace = trace();
        let mut segment = observe_prime(
            Tape::new(vec![1024]),
            Some(&trace),
            PrimeScope::SegmentWithContinuation,
        );
        advance_prime(&mut segment, true, |_, _| {}).unwrap();
        finish_prime(segment).unwrap();
        let mut calls = 0;
        let result = observe_call(
            Some(&trace),
            PrimeChunk {
                phase: "plain-prompt-token",
                rows: 1,
            },
            PrimeScope::RemainingPrompt,
            || {
                calls += 1;
                Ok(42)
            },
        )
        .unwrap();
        assert_eq!((result, calls), (42, 1));
        let records = ends(&trace);
        assert_eq!(records.len(), 2);
        assert!(records[0].contains("\"remaining_chunks\":null"));
        assert!(records[1].contains("\"rows\":1"));
        assert!(records[1].contains("\"phase\":\"prime_finished\""));
        assert!(records[1].contains("\"sequence_valid\":true"));
    }

    #[test]
    fn direct_call_error_and_zero_work_never_mint_completion() {
        let trace = trace();
        let err = observe_call::<()>(
            Some(&trace),
            PrimeChunk {
                phase: "plain-prime-call",
                rows: 9,
            },
            PrimeScope::RemainingPrompt,
            || Err("direct failure".into()),
        );
        assert_eq!(err.unwrap_err().to_string(), "direct failure");
        observe_call(
            Some(&trace),
            PrimeChunk {
                phase: "empty",
                rows: 0,
            },
            PrimeScope::RemainingPrompt,
            || Ok(()),
        )
        .unwrap();
        let records = ends(&trace);
        assert_eq!(records.len(), 1);
        assert!(records[0].contains("\"completed\":false"));
        assert!(records[0].contains("\"remaining_chunks\":null"));
    }

    #[test]
    fn empty_zero_and_unavailable_metadata_cannot_invent_actual_prime() {
        let trace = trace();
        for (rows, exposed) in [(vec![], true), (vec![0], true), (vec![17], false)] {
            let mut tape = Tape::new(rows);
            tape.exposed = exposed;
            let mut walker = observe_prime(tape, Some(&trace), PrimeScope::RemainingPrompt);
            advance_prime(&mut walker, false, |_, _| {}).unwrap();
            finish_prime(walker).unwrap();
        }
        trace.mark_prime_start();
        trace.mark_prime_end();
        assert!(ends(&trace).is_empty());
        assert!(
            !trace
                .observe_for_test()
                .json_lines()
                .iter()
                .any(|line| line.contains("prime_quantum_start"))
        );
    }

    #[test]
    fn client_drop_inside_real_quantum_is_not_early_retirement() {
        let trace = trace();
        let observer = trace.observe_for_test();
        let mut tape = Tape::new(vec![128, 128]);
        tape.trace = Some(trace.clone());
        tape.http_drop_in_advance = true;
        let mut walker = observe_prime(tape, Some(&trace), PrimeScope::RemainingPrompt);
        advance_prime(&mut walker, true, |_, _| {}).unwrap();
        drop(walker); // request-owned state released before explicit retirement
        trace.mark_receiver_closed(ReceiverCloseCause::ReceiverDropped);
        trace.mark_retired(RetirementOutcome::Aborted);
        drop(trace);
        let records = observer.json_lines();
        let at = |event: &str| {
            records
                .iter()
                .position(|line| line.contains(&format!("\"event\":\"{event}\"")))
                .unwrap()
        };
        assert!(at("prime_quantum_start") < at("http_pending_drop"));
        assert!(at("http_pending_drop") < at("prime_quantum_end"));
        assert!(at("prime_quantum_end") < at("receiver_closed"));
        assert!(at("receiver_closed") < at("retired"));
        assert!(records.last().unwrap().contains("\"sequence_valid\":true"));
        // This fixture observes actual prime work, not queued cancellation.
    }

    #[test]
    fn disabled_trace_leaves_the_tape_unchanged() {
        let mut walker = observe_prime(Tape::new(vec![3, 5]), None, PrimeScope::RemainingPrompt);
        advance_prime(&mut walker, false, |_, _| {}).unwrap();
        assert_eq!(finish_prime(walker).unwrap(), [3, 5]);
    }
}

#[cfg(test)]
#[path = "prime_observation_cancel_tests.rs"]
mod cancellation_tests;
