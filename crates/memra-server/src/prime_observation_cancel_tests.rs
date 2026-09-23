//! CPU tests of actual Scope/observer/Trace calls, not model or GPU execution.

use super::*;
use crate::ttft::{ReceiverCloseCause, RetirementOutcome, RetirementSite};
use memra_engine::progress::{PrimeCancelScope, PrimeCancelled, prime_cancel_point};
use std::sync::{Arc, Barrier};

fn bound(name: &str) -> Arc<Trace> {
    let trace = Trace::for_test("/v1/completions");
    trace.bind_client_trace_key("0123456789abcdef0123456789abcdef");
    trace.bind_request(name, "fixture");
    trace.bind_worker(1, "shared_gpu_worker");
    trace.mark_queued();
    trace
}

fn chunk() -> PrimeChunk {
    PrimeChunk {
        phase: "plain-prime-call",
        rows: 64,
    }
}

fn retire(trace: &Trace, cause: ReceiverCloseCause) {
    trace.mark_receiver_closed(cause);
    trace.mark_retired_at(RetirementOutcome::Aborted, RetirementSite::ActiveSession);
}

#[test]
fn actual_typed_error_is_preserved_and_never_completes_the_call() {
    let trace = bound("partial-return");
    let _scope = PrimeCancelScope::install(Box::new(|| true));
    let mut calls = 0;
    let result: Result<(), PrimeError> =
        observe_call(Some(&trace), chunk(), PrimeScope::RemainingPrompt, || {
            calls += 1;
            trace.mark_http_pending_drop();
            prime_cancel_point(0, 32, 64)?;
            panic!("partial cancellation must still return the original error");
        });
    let error = result.unwrap_err();
    assert_eq!(calls, 1);
    assert_eq!(
        error.downcast_ref::<PrimeCancelled>(),
        Some(&PrimeCancelled {
            chunk: 0,
            rows_done: 32,
            rows_total: 64
        })
    );
    let lines = trace.observe_for_test().json_lines();
    let last = lines.last().unwrap();
    assert!(last.contains("\"event\":\"prime_quantum_cancelled\""));
    assert!(
        last.contains("\"cancelled_at_boundary\":{\"chunk\":0,\"rows_done\":32,\"rows_total\":64}")
    );
    assert!(last.contains("\"completed\":false,\"remaining_chunks\":null"));
    assert!(last.contains("\"phase\":\"prime_cancelled\""));
    assert!(last.contains("\"sequence_valid\":true"));
    assert!(!last.contains("prime_finished"));
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("\"event\":\"retired\""))
    );
}

#[test]
fn actual_http_thread_drop_precedes_worker_partial_return() {
    let trace = bound("threaded-partial-return");
    let entered = Arc::new(Barrier::new(2));
    let dropped = Arc::new(Barrier::new(2));
    let worker_trace = trace.clone();
    let worker_entered = entered.clone();
    let worker_dropped = dropped.clone();
    let worker = std::thread::spawn(move || {
        let _scope = PrimeCancelScope::install(Box::new(|| true));
        let result: Result<(), PrimeError> = observe_call(
            Some(&worker_trace),
            chunk(),
            PrimeScope::RemainingPrompt,
            || {
                worker_entered.wait();
                worker_dropped.wait();
                prime_cancel_point(0, 32, 64)?;
                Ok(())
            },
        );
        assert!(result.unwrap_err().is::<PrimeCancelled>());
    });
    entered.wait();
    trace.mark_http_body_drop();
    dropped.wait();
    worker.join().unwrap();
    retire(&trace, ReceiverCloseCause::ReceiverDropped);
    let observer = trace.observe_for_test();
    drop(trace);
    let lines = observer.json_lines();
    let position = |event: &str| {
        lines
            .iter()
            .position(|line| line.contains(&format!("\"event\":\"{event}\"")))
            .unwrap()
    };
    assert!(position("prime_quantum_start") < position("http_body_drop"));
    assert!(position("http_body_drop") < position("prime_quantum_cancelled"));
    assert!(position("prime_quantum_cancelled") < position("receiver_closed"));
    assert!(position("receiver_closed") < position("retired"));
    assert!(position("retired") < position("trace_end"));
    assert!(lines.last().unwrap().contains("\"sequence_valid\":true"));
}

#[test]
fn no_scope_false_predicate_and_last_chunk_preserve_success() {
    for mode in ["absent", "false", "last"] {
        let trace = bound(mode);
        let _scope = match mode {
            "absent" => None,
            "false" => Some(PrimeCancelScope::install(Box::new(|| false))),
            _ => Some(PrimeCancelScope::install(Box::new(|| true))),
        };
        let out = observe_call(Some(&trace), chunk(), PrimeScope::RemainingPrompt, || {
            if mode != "last" {
                prime_cancel_point(0, 32, 64)?;
            }
            prime_cancel_point(1, 64, 64)?;
            Ok(73)
        })
        .unwrap();
        assert_eq!(out, 73);
        let lines = trace.observe_for_test().json_lines();
        assert!(
            !lines
                .iter()
                .any(|line| line.contains("\"event\":\"prime_quantum_cancelled\""))
        );
        let last = lines.last().unwrap();
        assert!(
            last.contains(
                "\"completed\":true,\"remaining_chunks\":0,\"cancelled_at_boundary\":null"
            )
        );
        assert!(last.contains("\"phase\":\"prime_finished\""));
        // A later HTTP drop is after actual completion, not an active-prime cancel.
        trace.mark_http_body_drop();
        assert!(
            trace
                .observe_for_test()
                .json_lines()
                .last()
                .unwrap()
                .contains("\"quantum_active\":false")
        );
    }
}

#[derive(Debug)]
struct Wrapped(PrimeCancelled);

impl std::fmt::Display for Wrapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for Wrapped {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[test]
fn similar_message_and_wrapped_error_are_not_exact_typed_cancellation() {
    for wrapped in [false, true] {
        let trace = bound("ordinary-error");
        let c = PrimeCancelled {
            chunk: 0,
            rows_done: 32,
            rows_total: 64,
        };
        let error: PrimeError = if wrapped {
            Box::new(Wrapped(c))
        } else {
            c.to_string().into()
        };
        let result: Result<(), PrimeError> =
            observe_call(Some(&trace), chunk(), PrimeScope::RemainingPrompt, || {
                trace.mark_http_pending_drop();
                Err(error)
            });
        assert!(!result.unwrap_err().is::<PrimeCancelled>());
        let lines = trace.observe_for_test().json_lines();
        let last = lines.last().unwrap();
        assert!(last.contains("\"event\":\"prime_quantum_end\""));
        assert!(last.contains("\"first_error\":\"failed_prime_quantum\""));
        assert!(last.contains("\"cancelled_at_boundary\":null"));
    }
}

#[test]
fn invalid_typed_boundary_or_missing_start_never_produces_valid_evidence() {
    for (chunk, done, total) in [
        (0, 0, 64),
        (0, 64, 64),
        (0, 65, 64),
        (0, 32, 65),
        (32, 32, 64),
    ] {
        let trace = bound("invalid-boundary");
        let result: Result<(), PrimeError> = observe_call(
            Some(&trace),
            self::chunk(),
            PrimeScope::RemainingPrompt,
            || {
                Err(Box::new(PrimeCancelled {
                    chunk,
                    rows_done: done,
                    rows_total: total,
                }))
            },
        );
        assert!(result.is_err());
        assert!(
            trace
                .observe_for_test()
                .json_lines()
                .last()
                .unwrap()
                .contains("\"first_error\":\"invalid_prime_cancellation_boundary\"")
        );
    }
    let trace = bound("missing-start");
    trace.mark_prime_quantum_cancelled(0, 32, 64);
    assert!(
        trace
            .observe_for_test()
            .json_lines()
            .last()
            .unwrap()
            .contains("\"first_error\":\"quantum_end_without_start\"")
    );
}

#[test]
fn typed_return_cannot_be_followed_by_more_prime_decode_or_normal_eof() {
    for next in ["prime", "decode", "eof", "duplicate", "requeue"] {
        let trace = bound(next);
        let _scope = PrimeCancelScope::install(Box::new(|| true));
        let _: Result<(), PrimeError> =
            observe_call(Some(&trace), chunk(), PrimeScope::RemainingPrompt, || {
                trace.mark_http_pending_drop();
                prime_cancel_point(0, 32, 64)?;
                Ok(())
            });
        match next {
            "prime" => trace.mark_prime_quantum_start("plain-prime-call", 64),
            "decode" => trace.mark_first_decode(),
            "eof" => trace.mark_http_body_eof(),
            "duplicate" => trace.mark_prime_quantum_cancelled(0, 32, 64),
            _ => trace.mark_requeued(),
        }
        assert!(
            trace
                .observe_for_test()
                .json_lines()
                .last()
                .unwrap()
                .contains("\"sequence_valid\":false")
        );
    }
}

#[test]
fn trace_drop_and_panic_do_not_invent_partial_return_or_retirement() {
    let trace = bound("panic");
    let observer = trace.observe_for_test();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = PrimeCancelScope::install(Box::new(|| true));
        let _: Result<(), PrimeError> =
            observe_call(Some(&trace), chunk(), PrimeScope::RemainingPrompt, || {
                trace.mark_http_pending_drop();
                panic!("inside actual call")
            });
    }));
    assert!(result.is_err());
    assert_eq!(prime_cancel_point(0, 32, 64), Ok(())); // scope unwound
    drop(trace);
    let lines = observer.json_lines();
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("\"event\":\"prime_quantum_cancelled\""))
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.contains("\"event\":\"retired\""))
    );
    assert!(lines.last().unwrap().contains("\"sequence_valid\":false"));
}

#[test]
fn actual_source_replay_cases() {
    // The optional output writes real Trace JSON for the Python replay harness.
    // Every case's Rust assertions run regardless of whether export is requested.
    let output = std::env::var_os("MEMRA_TYPED_TRACE_FIXTURES").map(std::path::PathBuf::from);
    for case in [
        "partial_pending",
        "partial_body",
        "overflow",
        "ordinary_error",
        "late_drop",
        "completed_return",
        "unknown_remaining",
        "missing_retirement",
        "early_retirement",
    ] {
        let trace = bound(case);
        let observer = trace.observe_for_test();
        let _scope = PrimeCancelScope::install(Box::new(|| true));
        let scope = if case == "unknown_remaining" {
            PrimeScope::SegmentWithContinuation
        } else {
            PrimeScope::RemainingPrompt
        };
        let result: Result<(), PrimeError> = observe_call(Some(&trace), chunk(), scope, || {
            if case != "late_drop" {
                if case == "partial_body" {
                    trace.mark_http_body_drop();
                } else {
                    trace.mark_http_pending_drop();
                }
            }
            if case == "early_retirement" {
                retire(&trace, ReceiverCloseCause::ReceiverDropped);
            }
            if case == "ordinary_error" {
                return Err("ordinary engine failure".into());
            }
            if !matches!(case, "completed_return" | "unknown_remaining") {
                prime_cancel_point(0, 32, 64)?;
            }
            Ok(())
        });
        assert_eq!(
            result.is_ok(),
            matches!(case, "completed_return" | "unknown_remaining")
        );
        if case == "late_drop" {
            trace.mark_http_pending_drop();
        }
        if !matches!(case, "missing_retirement" | "early_retirement") {
            retire(
                &trace,
                if case == "overflow" {
                    ReceiverCloseCause::EventQueueOverflow
                } else {
                    ReceiverCloseCause::ReceiverDropped
                },
            );
        }
        drop(trace);
        let lines = observer.json_lines();
        assert!(lines.last().unwrap().contains("\"event\":\"trace_end\""));
        if let Some(output) = &output {
            std::fs::create_dir_all(output).unwrap();
            let raw: String = lines
                .iter()
                .map(|line| format!("[request-lifecycle] {line}\n"))
                .collect();
            std::fs::write(output.join(format!("{case}.jsonl")), raw).unwrap();
        }
    }
}
