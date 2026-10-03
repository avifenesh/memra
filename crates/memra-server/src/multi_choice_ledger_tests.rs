// Owned CPU router fixture. The real HTTP handlers, row collector and capacity guards
// run against scripted worker events and a failing billing receipt, without a model.
#[derive(Default, Debug)]
struct ChoiceLedgerTrace {
    prompt_calls: usize,
    token_calls: usize,
    success_calls: usize,
    rejected: Vec<(u16, String)>,
    dropped: usize,
}

struct ChoiceFailMeter {
    trace: Arc<std::sync::Mutex<ChoiceLedgerTrace>>,
    prompt: bool,
    fail_token: usize,
}

impl metering::Metering for ChoiceFailMeter {
    fn enforces_limits(&self) -> bool {
        false
    }
    fn is_limited(&self, _: &str) -> Result<bool, metering::AdmitError> {
        Ok(true)
    }
    fn reserve(
        &self,
        _: &str,
        _: Option<&str>,
        _: &str,
        _: u64,
        _: u64,
    ) -> Result<Option<metering::Permit>, metering::AdmitError> {
        Ok(None)
    }
    fn open(
        &self,
        _: &metering::RequestMeta<'_>,
        _: Option<metering::Permit>,
    ) -> Box<dyn metering::Receipt> {
        Box::new(ChoiceFailReceipt {
            trace: self.trace.clone(),
            prompt: self.prompt,
            fail_token: self.fail_token,
        })
    }
    fn limits_health(&self) -> Option<metering::LimitsHealth> {
        None
    }
}

struct ChoiceFailReceipt {
    trace: Arc<std::sync::Mutex<ChoiceLedgerTrace>>,
    prompt: bool,
    fail_token: usize,
}

impl metering::Receipt for ChoiceFailReceipt {
    fn arm_capture(&mut self, _: serde_json::Value) {}
    fn capture_completion_delta(&mut self, _: &str) {}
    fn record_prompt_usage(&mut self, _: u64, _: u64) -> Result<(), String> {
        self.trace.lock().unwrap().prompt_calls += 1;
        if self.prompt {
            Err("owned prompt callback failure".into())
        } else {
            Ok(())
        }
    }
    fn record_completion_token(&mut self) -> Result<(), String> {
        let mut trace = self.trace.lock().unwrap();
        trace.token_calls += 1;
        if trace.token_calls == self.fail_token {
            Err("owned completion callback failure".into())
        } else {
            Ok(())
        }
    }
    fn complete(&mut self, _: metering::UsageCounts, _: f64) -> Result<(), String> {
        self.trace.lock().unwrap().success_calls += 1;
        Ok(())
    }
    fn complete_deadline_partial(
        &mut self,
        _: metering::UsageCounts,
        _: f64,
    ) -> Result<(), String> {
        self.trace.lock().unwrap().success_calls += 1;
        Ok(())
    }
    fn reject(&mut self, status: u16, code: &str) -> Result<(), String> {
        self.trace
            .lock()
            .unwrap()
            .rejected
            .push((status, code.into()));
        Ok(())
    }
    fn settle_unbilled(&mut self, _: &'static str, _: u16, _: &str) -> Result<(), String> {
        panic!("callback failures must reject, never settle as success or client abandon")
    }
}

impl Drop for ChoiceFailReceipt {
    fn drop(&mut self) {
        self.trace.lock().unwrap().dropped += 1;
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)] // isolate the actual handlers' process-global admission counters
async fn choice_http_ledger_callback_failures_are_named_and_release_owned_capacity() {
    let _guard = global_counter_writer_guard();
    let mut failures = Vec::new();
    for chat in [false, true] {
        for (prompt, stream, after_header) in [
            (true, false, false),
            (true, true, false),
            (true, true, true),
            (false, false, false),
            (false, true, false),
            (false, true, true),
        ] {
            let trace = Arc::new(std::sync::Mutex::new(ChoiceLedgerTrace::default()));
            let mut state = fake_worker_state();
            state.caps = Arc::new(HashMap::from([(
                "m".into(),
                ModelCaps {
                    max_choices: 4,
                    chat_ok: true,
                    context_length: 1024,
                    n_vocab: 32,
                    ..Default::default()
                },
            )]));
            state.metering = Some(Arc::new(ChoiceFailMeter {
                trace: trace.clone(),
                prompt,
                fail_token: if after_header { 2 } else { 1 },
            }));
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
            state.cmd_tx = cmd_tx;
            let (resume_tx, resume_rx) = std::sync::mpsc::channel();
            let (closed_tx, closed_rx) = tokio::sync::oneshot::channel();
            let worker = std::thread::spawn(move || {
                let Cmd::GenerateChoices(mut requests) = cmd_rx.recv().unwrap() else {
                    panic!("the actual handler must submit the choice group")
                };
                for request in &mut requests {
                    worker::release_pending_admit();
                    worker::release_request_reservation(request);
                    if let Some(ready) = request.constraint_ready.take() {
                        let _ = ready.send(Ok(()));
                    }
                }
                let first = usize::from(prompt && after_header);
                let usage = || Event::PromptUsage {
                    n_prompt: 5,
                    n_cached: 0,
                };
                let token = |id| Event::Token {
                    id,
                    text: "owned".into(),
                };
                let _ = requests[first].tx.send(usage());
                if after_header {
                    let _ = requests[first].tx.send(token(1));
                    // No remaining event can race the preheader decision. Release only
                    // after the router returns the streaming HTTP response to this test.
                    resume_rx
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                }
                for (index, request) in requests.iter().enumerate() {
                    if index != first {
                        let _ = request.tx.send(usage());
                    }
                    let _ = request.tx.send(token(2));
                    let _ = request.tx.send(Event::TokenSnapshot(vec![2]));
                    let _ = request.tx.send(Event::Done {
                        stop_reason: "MaxNew".into(),
                        n_tokens: 1,
                        n_prompt: 5,
                        n_cached: 0,
                        elapsed_s: 0.01,
                        spec: None,
                    });
                }
                // Hold every producer open so EOF cannot mask the callback failure.
                let _ = closed_tx.send(requests);
            });
            let mut payload = json!({"model":"m", "n":2, "max_tokens":2,
                "seed":500, "timeout_ms":1000, "stream":stream});
            if chat {
                payload["messages"] = json!([{"role":"user","content":"fixture"}]);
            } else {
                payload["prompt"] = json!("fixture");
            }
            if stream {
                payload["stream_options"] = json!({"include_usage":true});
            }
            let path = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                bg914_http(bg914_router(state.clone()), "POST", path, None, payload),
            )
            .await
            .unwrap();
            let status = response.status();
            if after_header {
                resume_tx.send(()).unwrap();
            }
            let bytes = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                axum::body::to_bytes(response.into_body(), 1024 * 1024),
            )
            .await
            .unwrap()
            .unwrap();
            let wire = String::from_utf8(bytes.to_vec()).unwrap();
            let requests = closed_rx.await.unwrap();
            worker.join().unwrap();
            let closed = requests.iter().all(|r| r.tx.is_closed());
            let counts_zero = state
                .inflight
                .iter()
                .all(|n| n.load(std::sync::atomic::Ordering::SeqCst) == 0)
                && state.tenant_inflight.lock().unwrap().is_empty()
                && worker::PENDING_ADMITS.load(std::sync::atomic::Ordering::Acquire) == 0
                && worker::ADMISSION_RESERVATIONS
                    .iter()
                    .all(|n| n.load(std::sync::atomic::Ordering::Acquire) == 0);
            let packets: Vec<serde_json::Value> = if after_header {
                wire.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect()
            } else {
                vec![serde_json::from_str(&wire).unwrap()]
            };
            let errors: Vec<_> = packets.iter().filter_map(|p| p.get("error")).collect();
            let seen = trace.lock().unwrap();
            let pass = status.as_u16() == if after_header { 200 } else { 500 }
                && errors.len() == 1
                && errors[0]["code"] == "request_ledger_unavailable"
                && errors[0]["type"] == "server_error"
                && !wire.contains("[DONE]")
                && packets
                    .iter()
                    .all(|p| p.get("usage").is_none_or(serde_json::Value::is_null))
                && seen.success_calls == 0
                && seen.dropped == 1
                && seen.rejected == vec![(500, "request_ledger_unavailable".into())]
                && seen.prompt_calls == 1
                && seen.token_calls
                    == if prompt {
                        usize::from(after_header)
                    } else {
                        if after_header { 2 } else { 1 }
                    }
                && closed
                && counts_zero;
            eprintln!(
                "CHOICE_LEDGER_CASE {}",
                json!({"chat":chat,"prompt":prompt,"stream":stream,
                "after_header":after_header,"status":status.as_u16(),"wire":wire,
                "trace":format!("{seen:?}"),"receivers_closed":closed,"capacity_released":counts_zero,"pass":pass})
            );
            if !pass {
                failures.push((chat, prompt, stream, after_header));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "callback classification/cleanup failures: {failures:?}"
    );
}
