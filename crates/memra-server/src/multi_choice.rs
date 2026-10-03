//! Request-scoped choice policy and lifecycle. No model arithmetic lives here.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) const MAX_CHOICES: usize = 8;

pub(crate) fn count(value: Option<usize>) -> Result<usize, &'static str> {
    let n = value.unwrap_or(1);
    if (1..=MAX_CHOICES).contains(&n) {
        Ok(n)
    } else {
        Err("n must be an integer from 1 through 8")
    }
}

pub(crate) fn supported_count(
    value: Option<usize>,
    caps: Option<&crate::worker::ModelCaps>,
) -> Result<usize, String> {
    let count = count(value).map_err(str::to_owned)?;
    let available = caps.map_or(1, |c| c.max_choices.max(1));
    if count > available {
        return Err(format!(
            "n-choice shared prefill is unavailable for this model/boot above n={available}; selected spec, vision, parallel or unsupported state routes require n=1"
        ));
    }
    Ok(count)
}

/// A supplied seed names independent per-choice streams, including at u64's boundary.
pub(crate) fn choice_seed(seed: u64, index: usize) -> u64 {
    seed.wrapping_add(index as u64)
}

#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) id: String,
    pub(crate) count: usize,
    admitted: AtomicUsize,
    failed: AtomicBool,
}

impl Group {
    pub(crate) fn new(id: String, count: usize) -> Arc<Self> {
        assert!((2..=MAX_CHOICES).contains(&count));
        Arc::new(Self {
            id,
            count,
            admitted: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.count
            .saturating_sub(self.admitted.load(Ordering::Acquire))
    }

    pub(crate) fn admit(&self) {
        let old = self.admitted.fetch_add(1, Ordering::AcqRel);
        assert!(old < self.count, "one admission per choice");
    }

    pub(crate) fn fail(&self) {
        self.failed.store(true, Ordering::Release);
    }

    pub(crate) fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Choice {
    pub(crate) group: Arc<Group>,
    pub(crate) index: usize,
    /// Only the leader may prefill. A follower becomes ready after its full restore.
    pub(crate) restored: bool,
}

impl Choice {
    pub(crate) fn waiting(&self) -> bool {
        self.group.remaining() != 0 || (self.index != 0 && !self.restored)
    }
}

use crate::metering::{Receipt, UsageCounts};
use crate::toolcall::{ParsedToolCall, Piece, ToolStreamParser};
use crate::worker::{EngineError, Event, EventReceiver};
use crate::{Envelope, StopScrubber};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::pin::Pin;

type IndexedStream = Pin<Box<dyn futures_core::Stream<Item = (usize, Event)> + Send>>;

pub(crate) fn multiplex(receivers: Vec<EventReceiver>) -> IndexedStream {
    let streams: Vec<IndexedStream> = receivers.into_iter().enumerate().map(|(index, mut rx)| {
        Box::pin(async_stream::stream! {
            let mut terminal = false;
            while let Some(event) = rx.recv().await {
                terminal = matches!(&event, Event::Done { .. } | Event::Error(_));
                yield (index, event);
                if terminal { break; }
            }
            if !terminal {
                yield (index, Event::Error(EngineError::engine("choice worker closed without a terminal result")));
            }
        }) as IndexedStream
    }).collect();
    Box::pin(futures_util::stream::select_all(streams))
}

struct Row {
    parser: Option<ToolStreamParser>,
    scrubber: Option<StopScrubber>,
    text: String,
    reasoning: String,
    calls: Vec<ParsedToolCall>,
    tokens: Vec<u32>,
    prompt: Option<(usize, usize)>,
    snapshot: bool,
    finish: Option<&'static str>,
    elapsed: f64,
    role_sent: bool,
}

pub(crate) struct Rows {
    rows: Vec<Row>,
    chat: bool,
    completed: usize,
    started: std::time::Instant,
}

impl Rows {
    pub(crate) fn new(chat: bool, parsers: Vec<Option<ToolStreamParser>>, stop: &[String]) -> Self {
        assert!((2..=MAX_CHOICES).contains(&parsers.len()));
        Self {
            rows: parsers
                .into_iter()
                .map(|parser| Row {
                    parser,
                    scrubber: (!stop.is_empty()).then(|| StopScrubber::new(stop.to_vec())),
                    text: String::new(),
                    reasoning: String::new(),
                    calls: Vec::new(),
                    tokens: Vec::new(),
                    prompt: None,
                    snapshot: false,
                    finish: None,
                    elapsed: 0.0,
                    role_sent: false,
                })
                .collect(),
            chat,
            completed: 0,
            started: std::time::Instant::now(),
        }
    }

    pub(crate) fn complete(&self) -> bool {
        self.completed == self.rows.len()
    }

    pub(crate) fn consume(
        &mut self,
        index: usize,
        event: Event,
        receipt: &mut Option<Box<dyn Receipt>>,
    ) -> Result<Vec<Value>, EngineError> {
        let row = self
            .rows
            .get_mut(index)
            .ok_or_else(|| EngineError::engine("choice index is out of range"))?;
        if row.finish.is_some() {
            return Err(EngineError::engine(
                "choice produced an event after its terminal result",
            ));
        }
        let mut pieces = Vec::new();
        let mut finish = None;
        match event {
            Event::PromptUsage { n_prompt, n_cached } => {
                if n_cached > n_prompt {
                    return Err(EngineError::engine("invalid choice cached-token count"));
                }
                row.prompt = Some((n_prompt, n_cached));
                if index == 0
                    && let Some(r) = receipt.as_mut()
                {
                    r.record_prompt_usage(n_prompt as u64, n_cached as u64)
                        .map_err(EngineError::engine)?;
                }
            }
            Event::Token { id, text } => {
                if row.snapshot {
                    return Err(EngineError::engine(
                        "choice emitted a token after its token snapshot",
                    ));
                }
                if row.prompt.is_none() {
                    return Err(EngineError::engine(
                        "choice emitted a token before prompt accounting",
                    ));
                }
                row.tokens.push(id);
                if let Some(r) = receipt.as_mut() {
                    r.record_completion_token().map_err(EngineError::engine)?;
                    if r.wants_capture() {
                        r.capture_completion_delta(&format!(
                            "{}\n",
                            json!({"index":index,"text":text})
                        ));
                    }
                }
                pieces = row
                    .parser
                    .as_mut()
                    .map_or_else(|| vec![Piece::Content(text.clone())], |p| p.push(&text));
            }
            Event::TokenSnapshot(tokens) => {
                if row.snapshot || tokens != row.tokens {
                    return Err(EngineError::engine(
                        "choice token snapshot differs from emitted token events",
                    ));
                }
                row.snapshot = true;
            }
            Event::Done {
                stop_reason,
                n_tokens,
                n_prompt,
                n_cached,
                elapsed_s,
                spec,
            } => {
                if !row.snapshot
                    || n_tokens != row.tokens.len()
                    || row.prompt != Some((n_prompt, n_cached))
                    || spec.is_some()
                {
                    return Err(EngineError::engine(
                        "choice terminal usage differs from observed events",
                    ));
                }
                pieces = row
                    .parser
                    .as_mut()
                    .map_or_else(Vec::new, ToolStreamParser::finish);
                row.elapsed = elapsed_s;
                finish = Some(crate::stop_reason_to_finish(&stop_reason));
            }
            Event::Error(error) => return Err(error),
            Event::DeadlineExceeded { ms } => {
                return Err(EngineError::engine(format!(
                    "choice first-token deadline exceeded after {ms} ms"
                )));
            }
            Event::PromptCapture { .. } => {
                return Err(EngineError::engine(
                    "unexpected capture event on n-choice generation",
                ));
            }
        }
        let mut deltas = Vec::new();
        for piece in pieces {
            match piece {
                Piece::Content(text) => {
                    let text = row
                        .scrubber
                        .as_mut()
                        .map_or_else(|| text.clone(), |s| s.push(&text));
                    if !text.is_empty() {
                        row.text.push_str(&text);
                        deltas.push(if self.chat {
                            json!({"content":text})
                        } else {
                            json!({"text":text})
                        });
                    }
                }
                Piece::Reasoning(text) => {
                    row.reasoning.push_str(&text);
                    if !text.is_empty() {
                        deltas.push(json!({"reasoning":text,"reasoning_details":[{"type":"reasoning.text","text":text}]}));
                    }
                }
                Piece::Call(call) => {
                    let mut value = crate::tool_call_json(&call);
                    value["index"] = json!(row.calls.len());
                    row.calls.push(call);
                    deltas.push(json!({"tool_calls":[value]}));
                }
            }
        }
        if let Some(reason) = finish {
            if let Some(scrubber) = row.scrubber.as_mut() {
                let tail = scrubber.finish();
                if !tail.is_empty() {
                    row.text.push_str(&tail);
                    deltas.push(if self.chat {
                        json!({"content":tail})
                    } else {
                        json!({"text":tail})
                    });
                }
            }
            row.finish = Some(if row.calls.is_empty() {
                reason
            } else {
                "tool_calls"
            });
            self.completed += 1;
        }
        let mut packets = Vec::new();
        for mut delta in deltas {
            if self.chat && !row.role_sent {
                delta["role"] = json!("assistant");
                row.role_sent = true;
            }
            packets.push(if self.chat {
                json!({"index":index,"delta":delta,"finish_reason":null})
            } else {
                json!({"index":index,"text":delta["text"],"finish_reason":null})
            });
        }
        if let Some(reason) = row.finish {
            let delta = if self.chat && !row.role_sent {
                json!({"role":"assistant"})
            } else {
                json!({})
            };
            row.role_sent = true;
            packets.push(if self.chat {
                json!({"index":index,"delta":delta,"finish_reason":reason})
            } else {
                json!({"index":index,"text":"","finish_reason":reason})
            });
        }
        Ok(packets)
    }

    pub(crate) fn usage(&self) -> UsageCounts {
        let (prompt, cached) = self.rows[0].prompt.unwrap_or((0, 0));
        UsageCounts {
            prompt_tokens: prompt as u64,
            cached_prompt_tokens: cached as u64,
            completion_tokens: self.rows.iter().map(|r| r.tokens.len() as u64).sum(),
        }
    }

    pub(crate) fn usage_json(&self) -> Value {
        let u = self.usage();
        crate::usage_json(
            u.prompt_tokens as usize,
            u.completion_tokens as usize,
            u.cached_prompt_tokens as usize,
            self.elapsed(),
            None,
        )
    }

    pub(crate) fn elapsed(&self) -> f64 {
        let completed = self.rows.iter().map(|r| r.elapsed).fold(0.0, f64::max);
        if self.complete() {
            completed
        } else {
            completed.max(self.started.elapsed().as_secs_f64())
        }
    }

    pub(crate) fn body(&self, env: &Envelope, model: &str, partial: bool) -> Value {
        let choices: Vec<Value> = self
            .rows
            .iter()
            .enumerate()
            .map(|(index, r)| {
                let finish = if partial && r.finish.is_none() {
                    "error"
                } else {
                    r.finish.unwrap_or("error")
                };
                if self.chat {
                    let content = if r.text.is_empty() && !r.calls.is_empty() {
                        Value::Null
                    } else {
                        json!(r.text)
                    };
                    let mut message = json!({"role":"assistant","content":content});
                    if !r.reasoning.is_empty() {
                        message["reasoning"] = json!(r.reasoning);
                        message["reasoning_details"] =
                            json!([{"type":"reasoning.text","text":r.reasoning}]);
                    }
                    if !r.calls.is_empty() {
                        message["tool_calls"] =
                            Value::Array(r.calls.iter().map(crate::tool_call_json).collect());
                    }
                    json!({"index":index,"message":message,"finish_reason":finish})
                } else {
                    json!({"index":index,"text":r.text,"finish_reason":finish})
                }
            })
            .collect();
        env.stamp(
            json!({"object":if self.chat { "chat.completion" } else { "text_completion" },
            "model":model,"choices":choices,"usage":self.usage_json()}),
        )
    }
}

/// One HTTP request against the tenant cap, N rows against worker and route capacity.
struct Guard {
    parent: crate::InflightGuard,
    extra: usize,
    _routes: Vec<crate::route_telemetry::RouteInflight>,
}

impl Guard {
    fn new(
        parent: crate::InflightGuard,
        count: usize,
        route: Option<Arc<crate::route_telemetry::RouteLoad>>,
    ) -> Self {
        let extra = count - 1;
        parent.counts[parent.idx].fetch_add(extra, Ordering::SeqCst);
        let routes = route.map_or_else(Vec::new, |r| {
            (0..extra).map(|_| r.enter(parent.idx)).collect()
        });
        Self {
            parent,
            extra,
            _routes: routes,
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        self.parent.counts[self.parent.idx].fetch_sub(self.extra, Ordering::SeqCst);
    }
}

pub(crate) struct Reply {
    pub(crate) model: String,
    pub(crate) chat: bool,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    pub(crate) env: Envelope,
    pub(crate) deadline: crate::RequestDeadline,
    pub(crate) parsers: Vec<Option<ToolStreamParser>>,
}

#[allow(clippy::too_many_arguments)] // the two HTTP admission callers share this request contract
pub(crate) async fn respond(
    st: &crate::AppState,
    tenant: &crate::auth::TenantCtx,
    mut request: crate::worker::Request,
    rx: EventReceiver,
    reply: Reply,
    capture_prompt: impl FnOnce() -> Option<Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let Reply {
        model,
        chat,
        stream,
        include_usage,
        env,
        deadline,
        parsers,
    } = reply;
    let count = parsers.len();
    let lane = request.lane;
    let route = if chat {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    };
    let bound = crate::effective_max_tokens(&request).and_then(|n| n.checked_mul(count));
    let budget = match crate::admit_tenant_budget_choices(st, tenant, &mut request, count) {
        Ok(budget) => budget,
        Err(rejection) => {
            let (response, code) = rejection.into_response();
            let receipt = crate::start_request_receipt(
                st, &env, tenant, &model, route, lane, stream, bound, None, None,
            );
            return crate::ledger_rejected(receipt, response, code, &env.id);
        }
    };
    let receipt = crate::start_request_receipt(
        st,
        &env,
        tenant,
        &model,
        route,
        lane,
        stream,
        bound,
        budget.reserved_ctx,
        budget.permit,
    );
    let mut receipt = receipt;
    if receipt.as_ref().is_some_and(|r| r.wants_capture())
        && let Some(prompt) = capture_prompt()
    {
        if let Some(r) = receipt.as_mut() {
            r.arm_capture(json!({"n":count,"request":prompt}));
        }
    }
    let stop = request.stop_strings.clone();
    let (mut requests, receivers) = match crate::worker::choice_requests(request, rx, count) {
        Ok(rows) => rows,
        Err(error) => {
            return crate::ledger_rejected(
                receipt,
                crate::engine_error_response(&error),
                crate::engine_error_code(error.class),
                &env.id,
            );
        }
    };
    let (parent, reading) = match crate::acquire_request_slot(st, lane, Some(&model), tenant, &env)
    {
        Ok(slot) => slot,
        Err(response) => {
            return crate::ledger_rejected(receipt, response, "rate_limit_exceeded", &env.id);
        }
    };
    let guard = Guard::new(parent, count, reading.route.clone());
    let tenant_count = st
        .tenant_inflight
        .lock()
        .ok()
        .and_then(|m| m.get(&tenant.tenant).copied())
        .unwrap_or(1);
    let reading = match reading.route {
        Some(r) => crate::RateLimit::at_admit_route(r, tenant, tenant_count),
        None => {
            let all = st.inflight[lane.idx()].load(Ordering::SeqCst);
            let routed: usize = crate::served_routes(st)
                .iter()
                .map(|r| r.inflight(lane.idx()))
                .sum();
            crate::RateLimit::at_admit(
                lane,
                all.saturating_sub(routed),
                &st.metrics,
                tenant,
                tenant_count,
            )
        }
    };
    let mut pending = Vec::with_capacity(count);
    for request in &mut requests {
        match crate::reserve_pending_admit(st, lane, &reading, deadline.preheader(stream)) {
            Ok(mut reservation) => {
                reservation.bind(request);
                pending.push(reservation);
            }
            Err((response, outcome)) => {
                return crate::ledger_unbilled(
                    receipt,
                    reading.attach(response),
                    outcome,
                    outcome,
                    &env.id,
                );
            }
        }
    }
    if st
        .cmd_tx
        .send(crate::worker::Cmd::GenerateChoices(requests))
        .is_err()
    {
        return crate::ledger_rejected(
            receipt,
            reading.attach(crate::worker_unavailable_response()),
            "worker_unavailable",
            &env.id,
        );
    }
    for reservation in pending {
        reservation.commit();
    }
    let mut input = multiplex(receivers);
    let mut rows = Rows::new(chat, parsers, &stop);
    let mut initial = Vec::new();
    if stream {
        loop {
            let event =
                match tokio::time::timeout_at(deadline.preheader(true).at, input.next()).await {
                    Ok(Some(event)) => event,
                    Ok(None) => {
                        return crate::ledger_rejected(
                            receipt,
                            reading.attach(crate::worker_unavailable_response()),
                            "worker_fault",
                            &env.id,
                        );
                    }
                    Err(_) => {
                        return crate::ledger_unbilled(
                            receipt,
                            reading.attach(crate::admission_deadline_response(deadline, true)),
                            "deadline_exceeded",
                            "deadline_exceeded",
                            &env.id,
                        );
                    }
                };
            let ready = matches!(&event.1, Event::Token { .. } | Event::Done { .. });
            match rows.consume(event.0, event.1, &mut receipt) {
                Ok(packets) => initial.extend(packets),
                Err(error) => {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(crate::engine_error_response(&error)),
                        crate::engine_error_code(error.class),
                        &env.id,
                    );
                }
            }
            if ready {
                break;
            }
        }
        let response = stream_reply(
            input,
            rows,
            initial,
            model,
            env.clone(),
            receipt,
            guard,
            include_usage,
        );
        return reading.attach(crate::with_request_id(&env.id, response));
    }
    while !rows.complete() {
        match tokio::time::timeout_at(deadline.at, input.next()).await {
            Ok(Some((index, event))) => {
                if let Err(error) = rows.consume(index, event, &mut receipt) {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(crate::engine_error_response(&error)),
                        crate::engine_error_code(error.class),
                        &env.id,
                    );
                }
            }
            Ok(None) => {
                return crate::ledger_rejected(
                    receipt,
                    reading.attach(crate::worker_unavailable_response()),
                    "worker_fault",
                    &env.id,
                );
            }
            Err(_) => {
                let usage = rows.usage();
                if usage.completion_tokens == 0 {
                    return crate::ledger_unbilled(
                        receipt,
                        reading.attach(crate::deadline_exceeded_response(deadline.ms, false)),
                        "deadline_exceeded",
                        "deadline_exceeded",
                        &env.id,
                    );
                }
                if let Some(r) = receipt.as_mut()
                    && r.complete_deadline_partial(usage, rows.elapsed()).is_err()
                {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(crate::request_ledger_error_response()),
                        "request_ledger_unavailable",
                        &env.id,
                    );
                }
                let mut body = rows.body(&env, &model, true);
                body["error"] = crate::deadline_exceeded_error(deadline.ms, false);
                drop(input);
                drop(guard);
                return reading.attach(crate::with_request_id(
                    &env.id,
                    axum::Json(body).into_response(),
                ));
            }
        }
    }
    if let Some(r) = receipt.as_mut()
        && r.complete(rows.usage(), rows.elapsed()).is_err()
    {
        return crate::ledger_rejected(
            receipt,
            reading.attach(crate::request_ledger_error_response()),
            "request_ledger_unavailable",
            &env.id,
        );
    }
    let body = rows.body(&env, &model, false);
    drop(input);
    drop(guard);
    reading.attach(crate::with_request_id(
        &env.id,
        axum::Json(body).into_response(),
    ))
}

#[allow(clippy::too_many_arguments)] // the stream owns all state needed for cancellation/drop
fn stream_reply(
    mut input: IndexedStream,
    mut rows: Rows,
    initial: Vec<Value>,
    model: String,
    env: Envelope,
    mut receipt: Option<Box<dyn Receipt>>,
    guard: Guard,
    include_usage: bool,
) -> axum::response::Response {
    use axum::response::sse::{Event as SseEvent, KeepAlive};
    use axum::response::{IntoResponse, Sse};
    let stream = async_stream::stream! {
        let _guard = guard;
        let object = if rows.chat { "chat.completion.chunk" } else { "text_completion" };
        let packet = |choice| {
            let mut payload = env.stamp(json!({"object":object,"model":model,"choices":[choice]}));
            if include_usage { payload["usage"] = Value::Null; }
            payload
        };
        for choice in initial { yield Ok::<_, std::convert::Infallible>(SseEvent::default().data(packet(choice).to_string())); }
        let mut final_packets = Vec::new();
        while !rows.complete() {
            let Some((index, event)) = input.next().await else {
                let error = EngineError::engine("n-choice worker closed before all terminal results");
                if let Some(r) = receipt.as_mut() { let _ = r.reject(500, "worker_fault"); }
                yield Ok(SseEvent::default().data(crate::engine_error_body(&error).to_string()));
                return;
            };
            match rows.consume(index, event, &mut receipt) {
                Ok(packets) => {
                    if rows.complete() { final_packets = packets; }
                    else { for choice in packets { yield Ok(SseEvent::default().data(packet(choice).to_string())); } }
                }

                Err(error) => {
                    if let Some(r) = receipt.as_mut() { let _ = r.reject(crate::class_http(error.class).0.as_u16(), crate::engine_error_code(error.class)); }
                    yield Ok(SseEvent::default().data(crate::engine_error_body(&error).to_string()));
                    return;
                }
            }
        }
        if let Some(r) = receipt.as_mut() && r.complete(rows.usage(), rows.elapsed()).is_err() {
            if let Some(r) = receipt.as_mut() { let _ = r.reject(500, "request_ledger_unavailable"); }
            yield Ok(SseEvent::default().data(crate::request_ledger_error_body().to_string()));
            return;
        }
        for choice in final_packets {
            let final_row = choice.get("finish_reason").is_some_and(|v| !v.is_null());
            let mut value = packet(choice);
            if !include_usage && final_row { value["usage"] = rows.usage_json(); }
            yield Ok(SseEvent::default().data(value.to_string()));
        }
        if include_usage {
            let usage = env.stamp(json!({"object":object,"model":model,"choices":[],"usage":rows.usage_json()}));
            yield Ok(SseEvent::default().data(usage.to_string()));
        }

        yield Ok(SseEvent::default().data("[DONE]"));
    };
    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(5))
                .text("keepalive"),
        )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Trace {
        prompts: Vec<(u64, u64)>,
        tokens: usize,
    }

    struct TraceReceipt(Arc<std::sync::Mutex<Trace>>);

    impl Receipt for TraceReceipt {
        fn arm_capture(&mut self, _: Value) {}
        fn capture_completion_delta(&mut self, _: &str) {}
        fn record_prompt_usage(&mut self, p: u64, c: u64) -> Result<(), String> {
            self.0.lock().unwrap().prompts.push((p, c));
            Ok(())
        }
        fn record_completion_token(&mut self) -> Result<(), String> {
            self.0.lock().unwrap().tokens += 1;
            Ok(())
        }
        fn complete(&mut self, _: UsageCounts, _: f64) -> Result<(), String> {
            Ok(())
        }
        fn complete_deadline_partial(&mut self, _: UsageCounts, _: f64) -> Result<(), String> {
            Ok(())
        }
        fn reject(&mut self, _: u16, _: &str) -> Result<(), String> {
            Ok(())
        }
        fn settle_unbilled(&mut self, _: &'static str, _: u16, _: &str) -> Result<(), String> {
            Ok(())
        }
    }

    fn done(n: usize) -> Event {
        Event::Done {
            stop_reason: "MaxNew".into(),
            n_tokens: n,
            n_prompt: 10,
            n_cached: 2,
            elapsed_s: 1.0,
            spec: None,
        }
    }

    #[test]
    fn interleaved_rows_count_prompt_once_and_outputs_sum_before_all_terminate() {
        let trace = Arc::new(std::sync::Mutex::new(Trace::default()));
        let mut receipt: Option<Box<dyn Receipt>> = Some(Box::new(TraceReceipt(trace.clone())));
        let mut rows = Rows::new(false, vec![None, None], &[]);
        for index in 0..2 {
            rows.consume(
                index,
                Event::PromptUsage {
                    n_prompt: 10,
                    n_cached: 2,
                },
                &mut receipt,
            )
            .unwrap();
        }
        rows.consume(
            1,
            Event::Token {
                id: 5,
                text: "b".into(),
            },
            &mut receipt,
        )
        .unwrap();
        rows.consume(
            0,
            Event::Token {
                id: 4,
                text: "a".into(),
            },
            &mut receipt,
        )
        .unwrap();
        rows.consume(0, Event::TokenSnapshot(vec![4]), &mut receipt)
            .unwrap();
        let finish = rows.consume(0, done(1), &mut receipt).unwrap();
        assert_eq!(finish.last().unwrap()["index"], 0);
        assert!(!rows.complete());
        rows.consume(
            1,
            Event::Token {
                id: 7,
                text: "c".into(),
            },
            &mut receipt,
        )
        .unwrap();
        rows.consume(1, Event::TokenSnapshot(vec![5, 7]), &mut receipt)
            .unwrap();
        rows.consume(1, done(2), &mut receipt).unwrap();
        assert!(rows.complete());
        let u = rows.usage();
        assert_eq!(
            (u.prompt_tokens, u.cached_prompt_tokens, u.completion_tokens),
            (10, 2, 3)
        );
        let seen = trace.lock().unwrap();
        assert_eq!(seen.prompts, vec![(10, 2)]);
        assert_eq!(seen.tokens, 3);
        let body = rows.body(&Envelope::new(false), "fixture", false);
        assert_eq!(body["choices"][0]["text"], "a");
        assert_eq!(body["choices"][1]["text"], "bc");
        assert_eq!(body["usage"]["total_tokens"], 13);
    }

    #[test]
    fn missing_snapshot_wrong_terminal_count_and_late_events_are_not_success() {
        let mut receipt = None;
        let mut rows = Rows::new(false, vec![None, None], &[]);
        rows.consume(
            0,
            Event::PromptUsage {
                n_prompt: 10,
                n_cached: 2,
            },
            &mut receipt,
        )
        .unwrap();
        rows.consume(
            0,
            Event::Token {
                id: 4,
                text: "a".into(),
            },
            &mut receipt,
        )
        .unwrap();
        assert!(rows.consume(0, done(1), &mut receipt).is_err());
        assert!(
            rows.consume(0, Event::TokenSnapshot(vec![9]), &mut receipt)
                .is_err()
        );
        rows.consume(0, Event::TokenSnapshot(vec![4]), &mut receipt)
            .unwrap();
        assert!(rows.consume(0, done(2), &mut receipt).is_err());
        rows.consume(0, done(1), &mut receipt).unwrap();
        assert!(rows.consume(0, done(1), &mut receipt).is_err());
        assert!(
            rows.consume(
                0,
                Event::Token {
                    id: 8,
                    text: "late".into()
                },
                &mut receipt
            )
            .is_err()
        );
        assert!(!rows.complete());
    }

    #[tokio::test]
    async fn dropping_the_whole_response_closes_every_choice_channel() {
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..3 {
            let (tx, rx) = crate::worker::event_channel();
            senders.push(tx);
            receivers.push(rx);
        }
        let mut input = multiplex(receivers);
        senders[2]
            .send(Event::Token {
                id: 7,
                text: "row2".into(),
            })
            .unwrap();
        assert!(matches!(
            input.next().await,
            Some((2, Event::Token { id: 7, .. }))
        ));
        drop(input);
        assert!(senders.iter().all(crate::worker::EventSender::is_closed));
    }

    #[test]
    fn one_tenant_request_holds_n_decode_slots_and_all_release_on_drop() {
        let counts = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        let tenants = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let lane = crate::lanes::Lane::Interactive;
        let (parent, _, _) = crate::InflightGuard::try_acquire(
            counts.clone(),
            lane,
            tenants.clone(),
            "tenant",
            Some(1),
        )
        .unwrap();
        let guard = Guard::new(parent, 3, None);
        assert_eq!(counts[lane.idx()].load(Ordering::SeqCst), 3);
        assert_eq!(tenants.lock().unwrap().get("tenant"), Some(&1));
        assert!(
            crate::InflightGuard::try_acquire(
                counts.clone(),
                lane,
                tenants.clone(),
                "tenant",
                Some(1)
            )
            .is_err()
        );
        drop(guard);
        assert_eq!(counts[lane.idx()].load(Ordering::SeqCst), 0);
        assert_eq!(
            tenants.lock().unwrap().get("tenant").copied().unwrap_or(0),
            0
        );
    }

    #[test]
    fn unavailable_model_capability_refuses_multiple_choices_but_preserves_n1() {
        assert_eq!(supported_count(None, None), Ok(1));
        assert_eq!(supported_count(Some(1), None), Ok(1));
        assert!(supported_count(Some(2), None).is_err());
        let caps = crate::worker::ModelCaps {
            max_choices: 4,
            ..Default::default()
        };
        assert_eq!(supported_count(Some(4), Some(&caps)), Ok(4));
        assert!(supported_count(Some(5), Some(&caps)).is_err());
    }

    #[test]
    fn explicit_bounds_and_legacy_default() {
        assert_eq!(count(None), Ok(1));
        assert_eq!(count(Some(1)), Ok(1));
        assert_eq!(count(Some(8)), Ok(8));
        assert!(count(Some(0)).is_err());
        assert!(count(Some(9)).is_err());
        assert!(count(Some(usize::MAX)).is_err());
    }

    #[test]
    fn seeded_rows_are_independent_and_choice_zero_is_unchanged() {
        assert_eq!(choice_seed(73, 0), 73);
        assert_eq!(choice_seed(73, 3), 76);
        assert_eq!(choice_seed(u64::MAX, 1), 0);
    }

    #[test]
    fn no_row_can_prime_before_the_whole_group_is_admitted() {
        let group = Group::new("request".into(), 3);
        let leader = Choice {
            group: group.clone(),
            index: 0,
            restored: false,
        };
        let mut follower = Choice {
            group: group.clone(),
            index: 1,
            restored: false,
        };
        group.admit();
        group.admit();
        assert!(leader.waiting());
        assert_eq!(group.remaining(), 1);
        group.admit();
        assert!(!leader.waiting());
        assert!(follower.waiting());
        follower.restored = true;
        assert!(!follower.waiting());
        group.fail();
        assert!(leader.group.failed());
        assert!(follower.group.failed());
    }
}
