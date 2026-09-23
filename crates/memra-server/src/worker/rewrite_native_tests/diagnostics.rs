//! Test-only, invocation-scoped evidence from the real error producer.
//! The public error remains redacted. No model query or rewrite guard runs here.
use super::super::{ErrClass, Event, EventReceiver, engine_client_message};
use super::{ProbeResult, events, require};
use std::cell::RefCell;

#[derive(Debug)]
pub(super) struct ProducerDiagnostic {
    class: ErrClass,
    message: String,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<ProducerDiagnostic>>> = const { RefCell::new(None) };
}

pub(in crate::worker) fn record_engine_error(class: ErrClass, message: &str) {
    CAPTURE.with(|slot| {
        if let Some(records) = slot.borrow_mut().as_mut() {
            records.push(ProducerDiagnostic {
                class,
                message: message.to_string(),
            });
        }
    });
}

struct CaptureScope;

impl CaptureScope {
    fn begin() -> ProbeResult<Self> {
        CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            require(slot.is_none(), "NESTED_ENGINE_DIAGNOSTIC_CAPTURE")?;
            *slot = Some(Vec::new());
            Ok(Self)
        })
    }

    fn finish(self) -> ProbeResult<Vec<ProducerDiagnostic>> {
        CAPTURE.with(|slot| {
            slot.borrow_mut()
                .take()
                .ok_or_else(|| "ENGINE_DIAGNOSTIC_CAPTURE_LOST".into())
        })
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        CAPTURE.with(|slot| *slot.borrow_mut() = None);
    }
}

pub(super) fn capture<T>(call: impl FnOnce() -> T) -> ProbeResult<(T, Vec<ProducerDiagnostic>)> {
    let scope = CaptureScope::begin()?;
    let result = call();
    Ok((result, scope.finish()?))
}

pub(super) fn refusal(
    result: ProbeResult<bool>,
    rx: &mut EventReceiver,
    expected: &str,
    diagnostics: Vec<ProducerDiagnostic>,
) -> ProbeResult<String> {
    let mut errors = Vec::new();
    match result {
        Ok(false) => {}
        Ok(true) => return Err("DRIFT_CALL_SUCCEEDED".into()),
        Err(error) => errors.push(error.to_string()),
    }
    let mut public_errors = 0;
    for event in events(rx) {
        match event {
            Event::Error(error) => {
                require(
                    error.class == ErrClass::Engine
                        && error.message == engine_client_message(ErrClass::Engine)
                        && error.param.is_none()
                        && error.retry_after_s.is_none(),
                    format!("WRONG_PUBLIC_REFUSAL: {error:?}"),
                )?;
                public_errors += 1;
            }
            other => return Err(format!("DRIFT_PUBLISHED_SUCCESS_EVENT: {other:?}").into()),
        }
    }
    require(
        public_errors == diagnostics.len(),
        "PUBLIC_REFUSAL_DIAGNOSTIC_COUNT_MISMATCH",
    )?;
    for diagnostic in diagnostics {
        require(
            diagnostic.class == ErrClass::Engine,
            "WRONG_PRODUCER_REFUSAL_CLASS",
        )?;
        errors.push(diagnostic.message);
    }
    require(
        errors.len() == 1 && errors[0].contains(expected),
        format!("WRONG_REFUSAL: {errors:?}"),
    )?;
    Ok(errors.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{EngineError, event_channel};

    #[test]
    fn actual_producer_preserves_public_redaction_and_exact_cause() {
        let (tx, mut rx) = event_channel();
        let raw = "loaded executable mappings changed: never-executed.bin";
        let (result, diagnostics) = capture(|| {
            let error = EngineError::engine(raw);
            assert_eq!(error.message, engine_client_message(ErrClass::Engine));
            assert!(!error.message.contains("never-executed.bin"));
            tx.send(Event::Error(error)).unwrap();
            Ok(false)
        })
        .unwrap();
        assert_eq!(refusal(result, &mut rx, raw, diagnostics).unwrap(), raw);
        assert!(events(&mut rx).is_empty());
    }

    #[test]
    fn generic_public_error_cannot_substitute_for_same_invocation_cause() {
        let (tx, mut rx) = event_channel();
        // An earlier matching producer call must not supply a later invocation's evidence.
        let old = EngineError::engine("snapshot was revoked");
        let (result, diagnostics) = capture(|| {
            tx.send(Event::Error(old)).unwrap();
            Ok(false)
        })
        .unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());

        let (result, diagnostics) = capture(|| {
            tx.send(Event::Error(EngineError::engine("an unrelated failure")))
                .unwrap();
            Ok(false)
        })
        .unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());

        let (result, diagnostics) = capture(|| {
            let _ = EngineError::engine("snapshot was revoked");
            Ok(false)
        })
        .unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());
    }

    #[test]
    fn direct_errors_remain_valid_but_success_events_and_wrong_classes_refuse() {
        let (tx, mut rx) = event_channel();
        let (result, diagnostics) = capture(|| Err("snapshot was revoked".into())).unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_ok());
        for mode in 0..3 {
            let (result, diagnostics) = capture(|| {
                if mode == 0 {
                    tx.send(Event::Token {
                        id: 3,
                        text: String::new(),
                    })
                    .unwrap();
                } else {
                    let message = if mode == 1 {
                        "CUDA_ERROR_OUT_OF_MEMORY"
                    } else {
                        "snapshot was revoked"
                    };
                    tx.send(Event::Error(EngineError::engine(message))).unwrap();
                }
                Ok(mode == 2)
            })
            .unwrap();
            assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());
            events(&mut rx);
        }
    }

    #[test]
    fn missing_duplicate_or_malformed_public_witnesses_refuse() {
        for mode in 0..8 {
            let (tx, mut rx) = event_channel();
            let (result, diagnostics) = capture(|| {
                if mode == 0 {
                    return Ok(false);
                }
                let mut error = EngineError::engine("snapshot was revoked");
                match mode {
                    1 => {
                        tx.send(Event::Error(EngineError::engine("snapshot was revoked")))
                            .unwrap();
                    }
                    2 => {
                        let _ = EngineError::engine("snapshot was revoked");
                    }
                    3 => tx.send(Event::Error(error.clone())).unwrap(),
                    4 => error.message = "snapshot was revoked".to_string(),
                    5 => error.param = Some("messages"),
                    6 => error.retry_after_s = Some(1),
                    7 => error.class = ErrClass::Overloaded,
                    _ => unreachable!(),
                }
                tx.send(Event::Error(error)).unwrap();
                Ok(false)
            })
            .unwrap();
            assert!(
                refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err(),
                "accepted malformed witness mode {mode}"
            );
        }
    }

    #[test]
    fn restored_phase_and_direct_errors_require_their_own_cause() {
        let (tx, mut rx) = event_channel();
        let (result, diagnostics) = capture(|| {
            tx.send(Event::Error(EngineError::engine(
                "loaded executable mappings changed: never-executed.bin",
            )))
            .unwrap();
            Ok(false)
        })
        .unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());

        let (result, diagnostics) = capture(|| Err("an unrelated failure".into())).unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());
        let (result, diagnostics) = capture(|| {
            let _ = EngineError::engine("snapshot was revoked");
            Err("snapshot was revoked".into())
        })
        .unwrap();
        assert!(refusal(result, &mut rx, "snapshot was revoked", diagnostics).is_err());

        // A direct return has no public channel error or producer record to invent.
        let (result, diagnostics) = capture(|| Err("snapshot was revoked".into())).unwrap();
        assert_eq!(
            refusal(result, &mut rx, "snapshot was revoked", diagnostics).unwrap(),
            "snapshot was revoked"
        );
    }

    #[test]
    fn capture_is_scoped_across_nested_attempts_panics_and_threads() {
        let (_, records) = capture(|| {
            assert!(capture(|| ()).is_err());
            let _ = EngineError::engine("outer invocation");
            std::thread::spawn(|| {
                let (_, records) = capture(|| EngineError::engine("other thread")).unwrap();
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].message, "other thread");
            })
            .join()
            .unwrap();
        })
        .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].message, "outer invocation");
        assert!(
            std::panic::catch_unwind(|| {
                let _ = capture(|| {
                    let _ = EngineError::engine("discard on unwind");
                    panic!("capture cleanup control");
                });
            })
            .is_err()
        );
        assert!(capture(|| ()).unwrap().1.is_empty());
        let (early, records) = capture(|| -> ProbeResult<()> {
            let _ = EngineError::engine("early returned error");
            Err("return before normal caller completion".into())
        })
        .unwrap();
        assert!(early.is_err());
        assert_eq!(records.len(), 1);
        assert!(capture(|| ()).unwrap().1.is_empty());
    }
}
