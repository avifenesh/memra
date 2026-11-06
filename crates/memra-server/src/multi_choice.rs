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

/// The leader retains ordinary reuse. Followers receive only its final prompt image,
/// so an independent device/host prefix or parked-session restore would violate the
/// existing prefix_restore fresh-destination contract.
pub(crate) fn reuse_eligible(choice: Option<&Choice>) -> bool {
    choice.is_none_or(|choice| choice.index == 0)
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
    partial_finalized: bool,
}

/// Receipt failures belong to the HTTP accounting contract, not the model executor.
#[derive(Debug)]
pub(crate) enum RowFailure {
    Engine(EngineError),
    Ledger(String),
}

impl RowFailure {
    fn engine(message: impl Into<String>) -> Self {
        Self::Engine(EngineError::engine(message))
    }

    fn ledger(message: String) -> Self {
        Self::Ledger(message)
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Engine(error) => crate::engine_error_code(error.class),
            Self::Ledger(_) => "request_ledger_unavailable",
        }
    }

    fn status(&self) -> u16 {
        match self {
            Self::Engine(error) => crate::class_http(error.class).0.as_u16(),
            Self::Ledger(_) => 500,
        }
    }

    fn response(&self, request_id: &str) -> axum::response::Response {
        match self {
            Self::Engine(error) => crate::engine_error_response(error),
            Self::Ledger(message) => {
                eprintln!("[ledger] ERROR: request {request_id} partial receipt failed: {message}");
                crate::request_ledger_error_response()
            }
        }
    }

    fn body(&self, request_id: &str) -> Value {
        match self {
            Self::Engine(error) => crate::engine_error_body(error),
            Self::Ledger(message) => {
                eprintln!("[ledger] ERROR: request {request_id} partial receipt failed: {message}");
                crate::request_ledger_error_body()
            }
        }
    }
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
                    partial_finalized: false,
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

    /// A worker error that reaches a row before its prompt accounting is a refusal or fault
    /// before admission: log it like n = 1 does.
    fn log_pre_admission(&self, index: usize, event: &Event) {
        if let Event::Error(error) = event
            && self.rows.get(index).is_some_and(|row| row.prompt.is_none())
        {
            log_pre_admission_error(index, error);
        }
    }

    pub(crate) fn consume(
        &mut self,
        index: usize,
        event: Event,
        receipt: &mut Option<Box<dyn Receipt>>,
    ) -> Result<Vec<Value>, RowFailure> {
        let row = self
            .rows
            .get_mut(index)
            .ok_or_else(|| RowFailure::engine("choice index is out of range"))?;
        if row.finish.is_some() || row.partial_finalized {
            return Err(RowFailure::engine(
                "choice produced an event after its terminal result",
            ));
        }
        let mut pieces = Vec::new();
        let mut finish = None;
        match event {
            Event::PromptUsage { n_prompt, n_cached } => {
                if n_cached > n_prompt {
                    return Err(RowFailure::engine("invalid choice cached-token count"));
                }
                row.prompt = Some((n_prompt, n_cached));
                if index == 0
                    && let Some(r) = receipt.as_mut()
                {
                    r.record_prompt_usage(n_prompt as u64, n_cached as u64)
                        .map_err(RowFailure::ledger)?;
                }
            }
            Event::Token { id, text } => {
                if row.snapshot {
                    return Err(RowFailure::engine(
                        "choice emitted a token after its token snapshot",
                    ));
                }
                if row.prompt.is_none() {
                    return Err(RowFailure::engine(
                        "choice emitted a token before prompt accounting",
                    ));
                }
                row.tokens.push(id);
                if let Some(r) = receipt.as_mut() {
                    r.record_completion_token().map_err(RowFailure::ledger)?;
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
                    return Err(RowFailure::engine(
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
                    return Err(RowFailure::engine(
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
            Event::Error(error) => return Err(RowFailure::Engine(error)),
            Event::DeadlineExceeded { ms } => {
                return Err(RowFailure::engine(format!(
                    "choice first-token deadline exceeded after {ms} ms"
                )));
            }
            Event::PromptCapture { .. } => {
                return Err(RowFailure::engine(
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

    /// A deadline delivers every already-consumed byte without inventing more token work.
    /// Completed rows are immutable; a partial row can be finalized only once.
    pub(crate) fn finalize_partial(&mut self) {
        for row in &mut self.rows {
            if row.finish.is_some() || row.partial_finalized {
                continue;
            }
            let pieces = row
                .parser
                .as_mut()
                .map_or_else(Vec::new, ToolStreamParser::finish);
            for piece in pieces {
                match piece {
                    Piece::Content(text) => {
                        let text = row
                            .scrubber
                            .as_mut()
                            .map_or_else(|| text.clone(), |s| s.push(&text));
                        row.text.push_str(&text);
                    }
                    Piece::Reasoning(text) => row.reasoning.push_str(&text),
                    Piece::Call(call) => row.calls.push(call),
                }
            }
            if let Some(scrubber) = row.scrubber.as_mut() {
                row.text.push_str(&scrubber.finish());
            }
            row.partial_finalized = true;
        }
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

/// The n = 1 `[admission] error before admission` line, with the row's index.
fn log_pre_admission_error(index: usize, error: &EngineError) {
    eprintln!(
        "[admission] error before admission: code={} choice={index} message={:?}",
        crate::engine_error_code(error.class),
        error.message
    );
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
    let bound = crate::effective_max_tokens(&request).and_then(|n| n.checked_mul(count as u64));
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
        && let Some(r) = receipt.as_mut()
    {
        r.arm_capture(json!({"n":count,"request":prompt}));
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
    // An extended stream (first-token deadline past the shipped pre-header window, with an SSE
    // prefill commit) commits headers at `window` and keeps waiting, queued or prefilling, under
    // SSE keepalives until its whole first-token deadline, as n = 1 does. So the queue shed
    // judges the group's wait estimate against that deadline, not the commit.
    let window = deadline.preheader(stream);
    let commits_early = window.ms < deadline.ms;
    let mut pending = match crate::reserve_pending_choices(st, lane, &reading, deadline, count) {
        Ok(reservations) => reservations,
        Err((response, outcome)) => {
            return crate::ledger_unbilled(
                receipt,
                reading.attach(response),
                outcome,
                outcome,
                &env.id,
            );
        }
    };
    for (request, reservation) in requests.iter_mut().zip(&mut pending) {
        reservation.bind(request);
    }
    // A group that may commit before its first token must not commit before its schema
    // compiled, the n = 1 rule: every constrained row reports its own compile.
    let constraint_ready: Vec<_> = if commits_early {
        requests
            .iter_mut()
            .enumerate()
            .filter(|(_, request)| request.grammar.is_some())
            .map(|(index, request)| {
                let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
                request.constraint_ready = Some(ready_tx);
                (index, ready_rx)
            })
            .collect()
    } else {
        Vec::new()
    };
    crate::meter_admit(&env, tenant, &model, lane);
    if let Some(trace) = requests.first().and_then(|request| request.ttft.as_ref()) {
        trace.mark_submitted();
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
    // The compile wait counts against the pre-header window, as on n = 1. The rows compile
    // independently, so the first refusal answers without waiting for the others.
    let compiled_by = tokio::time::Instant::now()
        + crate::constrained::CONSTRAINT_COMPILE_TIMEOUT.min(window.remaining());
    let mut compiles: futures_util::stream::FuturesUnordered<_> = constraint_ready
        .into_iter()
        .map(|(index, ready)| async move { (index, ready.await) })
        .collect();
    loop {
        match tokio::time::timeout_at(compiled_by, compiles.next()).await {
            Ok(None) => break,
            Ok(Some((_, Ok(Ok(()))))) => {}
            Ok(Some((index, Ok(Err(error))))) => {
                log_pre_admission_error(index, &error);
                return crate::ledger_rejected(
                    receipt,
                    reading.attach(crate::engine_error_response(&error)),
                    crate::engine_error_code(error.class),
                    &env.id,
                );
            }
            Ok(Some((_, Err(_)))) => {
                return crate::ledger_rejected(
                    receipt,
                    reading.attach(crate::worker_unavailable_response()),
                    "worker_unavailable",
                    &env.id,
                );
            }
            Err(_) if window.remaining().is_zero() => {
                return crate::ledger_unbilled(
                    receipt,
                    reading.attach(crate::admission_deadline_response(deadline, true)),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                );
            }
            Err(_) => {
                return crate::ledger_rejected(
                    receipt,
                    reading.attach(crate::engine_error_response(
                        &crate::worker::constraint_timeout_error(),
                    )),
                    "constraint_compile_timeout",
                    &env.id,
                );
            }
        }
    }
    let mut input = multiplex(receivers);
    let mut rows = Rows::new(chat, parsers, &stop);
    let mut initial = Vec::new();
    if stream {
        let mut committed = false;
        loop {
            let event = match tokio::time::timeout_at(window.at, input.next()).await {
                Ok(Some(event)) => event,
                Ok(None) => {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(crate::worker_unavailable_response()),
                        "worker_fault",
                        &env.id,
                    );
                }
                // An extended stream: the window was its commit, not its deadline.
                Err(_) if commits_early => {
                    committed = true;
                    break;
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
            rows.log_pre_admission(event.0, &event.1);
            match rows.consume(event.0, event.1, &mut receipt) {
                Ok(packets) => initial.extend(packets),
                Err(error) => {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(error.response(&env.id)),
                        error.code(),
                        &env.id,
                    );
                }
            }
            if ready {
                break;
            }
        }
        if committed {
            let shared = crate::prefill_receipt::SharedReceipt::wrap(&mut receipt);
            input = commit_until_first_delivery(input, deadline, shared);
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
                rows.log_pre_admission(index, &event);
                if let Err(error) = rows.consume(index, event, &mut receipt) {
                    return crate::ledger_rejected(
                        receipt,
                        reading.attach(error.response(&env.id)),
                        error.code(),
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
                rows.finalize_partial();
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
                body["error"] = crate::deadline_exceeded_error(deadline.ms, false)["error"].clone();
                // A 200 that carries an error is a failed response, as n = 1 counts it.
                if let Some(observation) = crate::http_metrics::current() {
                    observation.failed();
                }
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

/// The group twin of `forward_prefill_until_first_delivery`: once an extended group stream has
/// committed headers before any row produced its first token, its first-token deadline keeps
/// running whether or not the SSE body is polled. At the deadline the shared receipt settles the
/// zero-debit `deadline_exceeded` outcome, every row's channel closes (the cancel the worker
/// retires queued and active rows on), and the stream ends with the synthetic `DeadlineExceeded`
/// event the body answers in the wire dialect. The first token, terminal or error of any row
/// disarms the deadline: the stream deadline bounds time to first token only.
fn commit_until_first_delivery(
    mut input: IndexedStream,
    deadline: crate::RequestDeadline,
    mut receipt: Option<crate::prefill_receipt::SharedReceipt>,
) -> IndexedStream {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(usize, Event)>();
    tokio::spawn(async move {
        let expiry = tokio::time::sleep_until(deadline.at);
        tokio::pin!(expiry);
        let mut delivered = false;
        loop {
            let event = tokio::select! {
                biased;
                () = &mut expiry, if !delivered => {
                    if let Some(mut receipt) = receipt.take()
                        && let Err(error) =
                            receipt.settle_unbilled("deadline_exceeded", 408, "deadline_exceeded")
                    {
                        eprintln!("[ledger] ERROR: choice deadline settlement failed: {error}");
                    }
                    let _ = tx.send((0, Event::DeadlineExceeded { ms: deadline.ms }));
                    return; // dropping `input` closes every row's channel
                }
                () = tx.closed() => return,
                event = input.next() => event,
            };
            let Some((index, event)) = event else { return };
            if !delivered {
                if let Event::Error(error) = &event {
                    // After the commit a refusal or fault reaches the client as an SSE error
                    // event; log it like the pre-commit refusals.
                    eprintln!(
                        "[admission] error after the SSE commit: code={} choice={index} message={:?}",
                        crate::engine_error_code(error.class),
                        error.message
                    );
                }
                if matches!(
                    &event,
                    Event::Token { .. } | Event::Done { .. } | Event::Error(_)
                ) {
                    delivered = true;
                    receipt = None; // the body owns the terminal settlement from here
                }
            }
            if tx.send((index, event)).is_err() {
                return;
            }
        }
    });
    Box::pin(async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield event;
        }
    })
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
    // Report the outcome to the HTTP layer as the n = 1 SSE body does: without it the
    // middleware saw every group stream end without a terminal, an error and truncated.
    let http_observation = crate::http_metrics::current();
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
                if let Some(observation) = &http_observation { observation.failed(); }
                yield Ok(SseEvent::default().data(crate::engine_error_body(&error).to_string()));
                yield Ok(SseEvent::default().data("[DONE]"));
                return;
            };
            if let Event::DeadlineExceeded { ms } = event {
                // A committed group's first-token deadline (`commit_until_first_delivery`
                // already settled the shared receipt; this returns that settlement). Like every
                // error after the headers, and like an OpenAI-shape n = 1 stream: a `data:` error, then
                // `[DONE]` (n > 1 is always OpenAI-shaped).
                let settled = receipt.as_mut().map_or(Ok(()), |r| {
                    r.settle_unbilled("deadline_exceeded", 408, "deadline_exceeded")
                });
                let payload = if settled.is_err() {
                    crate::request_ledger_error_body()
                } else {
                    crate::deadline_exceeded_error(ms, true)
                };
                if let Some(observation) = &http_observation { observation.failed(); }
                yield Ok(SseEvent::default().data(payload.to_string()));
                yield Ok(SseEvent::default().data("[DONE]"));
                return;
            }
            match rows.consume(index, event, &mut receipt) {
                Ok(packets) => {
                    if rows.complete() { final_packets = packets; }
                    else { for choice in packets { yield Ok(SseEvent::default().data(packet(choice).to_string())); } }
                }

                Err(error) => {
                    if let Some(r) = receipt.as_mut() { let _ = r.reject(error.status(), error.code()); }
                    if let Some(observation) = &http_observation { observation.failed(); }
                    yield Ok(SseEvent::default().data(error.body(&env.id).to_string()));
                    yield Ok(SseEvent::default().data("[DONE]"));
                    return;
                }
            }
        }
        if let Some(r) = receipt.as_mut() && r.complete(rows.usage(), rows.elapsed()).is_err() {
            if let Some(r) = receipt.as_mut() { let _ = r.reject(500, "request_ledger_unavailable"); }
            if let Some(observation) = &http_observation { observation.failed(); }
            yield Ok(SseEvent::default().data(crate::request_ledger_error_body().to_string()));
            yield Ok(SseEvent::default().data("[DONE]"));
            return;
        }
        // Published before the terminal frames: a client may close as soon as it has them.
        if let Some(observation) = &http_observation { observation.terminal(); }
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
    fn deadline_flushes_stop_prefix_once_and_preserves_completed_rows_and_usage() {
        let mut rows = Rows::new(false, vec![None, None], &["END".into()]);
        let mut receipt = None;
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
            0,
            Event::Token {
                id: 7,
                text: "kept".into(),
            },
            &mut receipt,
        )
        .unwrap();
        rows.consume(0, Event::TokenSnapshot(vec![7]), &mut receipt)
            .unwrap();
        rows.consume(0, done(1), &mut receipt).unwrap();
        for (id, ch) in "abcE".chars().enumerate() {
            rows.consume(
                1,
                Event::Token {
                    id: id as u32,
                    text: ch.to_string(),
                },
                &mut receipt,
            )
            .unwrap();
        }
        assert_eq!(rows.rows[1].text, "abc");
        let before = rows.usage();
        rows.finalize_partial();
        rows.finalize_partial();
        let after = rows.usage();
        assert_eq!(
            (
                before.prompt_tokens,
                before.cached_prompt_tokens,
                before.completion_tokens
            ),
            (
                after.prompt_tokens,
                after.cached_prompt_tokens,
                after.completion_tokens
            )
        );
        let body = rows.body(&Envelope::new(false), "fixture", true);
        assert_eq!(body["choices"][0]["text"], "kept");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["choices"][1]["text"], "abcE");
        assert_eq!(body["choices"][1]["finish_reason"], "error");
        assert_eq!(body["usage"]["completion_tokens"], 5);
        assert!(
            rows.consume(
                1,
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

    #[test]
    fn deadline_flushes_reasoning_and_unterminated_tool_buffers_as_the_n1_parser_does() {
        let reasoning = ToolStreamParser::new(Default::default(), true);
        let tool = ToolStreamParser::new(Default::default(), false);
        let mut rows = Rows::new(true, vec![Some(reasoning), Some(tool)], &[]);
        let mut receipt = None;
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
            0,
            Event::Token {
                id: 7,
                text: "reasoning<".into(),
            },
            &mut receipt,
        )
        .unwrap();
        let partial = "<tool_call>\n<function=weather>\n<parameter=city>\nParis";
        rows.consume(
            1,
            Event::Token {
                id: 8,
                text: partial.into(),
            },
            &mut receipt,
        )
        .unwrap();
        let before = rows.usage().completion_tokens;
        rows.finalize_partial();
        let first = rows.body(&Envelope::new(true), "fixture", true);
        rows.finalize_partial();
        let second = rows.body(&Envelope::new(true), "fixture", true);
        assert_eq!(first["choices"], second["choices"]);
        assert_eq!(first["choices"][0]["message"]["reasoning"], "reasoning<");
        assert_eq!(first["choices"][1]["message"]["content"], partial);
        assert_eq!(rows.usage().completion_tokens, before);
        assert_eq!(before, 2);
    }

    /// Terminal settlements a committed group's receipt saw, and whether it died unsettled
    /// (the billable client-abandon path).
    #[derive(Default)]
    struct Settled {
        terminals: Vec<String>,
        abandoned: bool,
    }

    struct SettleReceipt(Arc<std::sync::Mutex<Settled>>, bool);

    impl SettleReceipt {
        fn settle(&mut self, terminal: String) -> Result<(), String> {
            self.1 = true;
            self.0.lock().unwrap().terminals.push(terminal);
            Ok(())
        }
    }

    impl Receipt for SettleReceipt {
        fn arm_capture(&mut self, _: Value) {}
        fn capture_completion_delta(&mut self, _: &str) {}
        fn record_prompt_usage(&mut self, _: u64, _: u64) -> Result<(), String> {
            Ok(())
        }
        fn record_completion_token(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn complete(&mut self, u: UsageCounts, _: f64) -> Result<(), String> {
            self.settle(format!("complete {}", u.completion_tokens))
        }
        fn complete_deadline_partial(&mut self, _: UsageCounts, _: f64) -> Result<(), String> {
            self.settle("deadline_partial".into())
        }
        fn reject(&mut self, status: u16, code: &str) -> Result<(), String> {
            self.settle(format!("reject {status} {code}"))
        }
        fn settle_unbilled(
            &mut self,
            outcome: &'static str,
            status: u16,
            code: &str,
        ) -> Result<(), String> {
            self.settle(format!("unbilled {outcome} {status} {code}"))
        }
    }

    impl Drop for SettleReceipt {
        fn drop(&mut self) {
            if !self.1 {
                self.0.lock().unwrap().abandoned = true;
            }
        }
    }

    /// A two-row chat group committed before its first token with a 300 ms first-token
    /// deadline: the senders, the SSE response and the receipt's settlement log.
    fn committed_group() -> (
        Vec<crate::worker::EventSender>,
        axum::response::Response,
        Arc<std::sync::Mutex<Settled>>,
    ) {
        let (mut senders, mut receivers) = (Vec::new(), Vec::new());
        for _ in 0..2 {
            let (tx, rx) = crate::worker::event_channel();
            senders.push(tx);
            receivers.push(rx);
        }
        let log = Arc::new(std::sync::Mutex::new(Settled::default()));
        let mut receipt: Option<Box<dyn Receipt>> =
            Some(Box::new(SettleReceipt(log.clone(), false)));
        let shared = crate::prefill_receipt::SharedReceipt::wrap(&mut receipt);
        let input = commit_until_first_delivery(
            multiplex(receivers),
            crate::RequestDeadline::starting_now(300),
            shared,
        );
        let counts = Arc::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        let tenants = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let (parent, _, _) = crate::InflightGuard::try_acquire(
            counts,
            crate::lanes::Lane::Interactive,
            tenants,
            "tenant",
            Some(1),
        )
        .unwrap();
        let response = stream_reply(
            input,
            Rows::new(true, vec![None, None], &[]),
            Vec::new(),
            "m".into(),
            Envelope::new(true),
            receipt,
            Guard::new(parent, 2, None),
            false,
        );
        (senders, response, log)
    }

    async fn data_lines(response: axum::response::Response) -> Vec<String> {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    async fn all_closed(senders: &[crate::worker::EventSender]) {
        for tx in senders {
            tokio::time::timeout(std::time::Duration::from_secs(5), tx.closed())
                .await
                .expect("every row's channel closes");
        }
    }

    fn usage() -> Event {
        Event::PromptUsage {
            n_prompt: 10,
            n_cached: 2,
        }
    }

    /// GB10 n > 1 (2026-11-06): after a group commits headers, its first-token deadline keeps
    /// running. Expiry settles the receipt zero-debit and closes every row even when the body is
    /// never polled; a polled body ends with the timeout error, then `[DONE]`; the first token of
    /// any row disarms the deadline; a refusal after the commit is an error event, then `[DONE]`.
    #[tokio::test(start_paused = true)]
    async fn a_committed_group_keeps_its_first_token_deadline() {
        // Unpolled: the deadline settles and cancels with nobody reading the body.
        let (senders, response, log) = committed_group();
        senders[0].send(usage()).unwrap();
        all_closed(&senders).await;
        drop(response);
        {
            let settled = log.lock().unwrap();
            assert_eq!(
                settled.terminals,
                ["unbilled deadline_exceeded 408 deadline_exceeded"]
            );
            assert!(!settled.abandoned, "a timed-out group is never an abandon");
        }

        // Polled: the wire's terminal event, and the one settlement.
        let (senders, response, log) = committed_group();
        senders[1].send(usage()).unwrap();
        let lines = data_lines(response).await;
        assert_eq!(lines.len(), 2, "the timeout error, then [DONE]: {lines:?}");
        assert_eq!(lines[1], "[DONE]");
        let error: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(error["error"]["type"], "timeout");
        assert_eq!(error["error"]["code"], "deadline_exceeded");
        let message = error["error"]["message"].as_str().unwrap();
        assert!(message.contains("deadline of 300 ms"), "{message}");
        assert!(message.contains("first token"), "{message}");
        assert!(message.contains("not billed"), "{message}");
        all_closed(&senders).await;
        assert_eq!(
            log.lock().unwrap().terminals,
            ["unbilled deadline_exceeded 408 deadline_exceeded"]
        );

        // The first token of one row spends the deadline for the whole group.
        let (senders, response, log) = committed_group();
        senders[0].send(usage()).unwrap();
        senders[0]
            .send(Event::Token {
                id: 4,
                text: "a".into(),
            })
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert!(
            senders.iter().all(|tx| !tx.is_closed()),
            "the group runs on"
        );
        senders[0].send(Event::TokenSnapshot(vec![4])).unwrap();
        senders[0].send(done(1)).unwrap();
        senders[1].send(usage()).unwrap();
        senders[1]
            .send(Event::Token {
                id: 5,
                text: "b".into(),
            })
            .unwrap();
        senders[1].send(Event::TokenSnapshot(vec![5])).unwrap();
        senders[1].send(done(1)).unwrap();
        let lines = data_lines(response).await;
        assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
        assert!(!lines.iter().any(|l| l.contains("\"error\"")), "{lines:?}");
        assert_eq!(log.lock().unwrap().terminals, ["complete 2"]);

        // A refusal after the commit reaches the client as the group's error event.
        let (senders, response, log) = committed_group();
        senders[1]
            .send(Event::Error(EngineError::rate_limit("lane full")))
            .unwrap();
        let lines = data_lines(response).await;
        assert_eq!(lines.len(), 2, "the refusal, then [DONE]: {lines:?}");
        assert_eq!(lines[1], "[DONE]");
        let error: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(error["error"]["code"], "rate_limit_exceeded");
        all_closed(&senders).await;
        assert_eq!(
            log.lock().unwrap().terminals,
            ["reject 429 rate_limit_exceeded"]
        );
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
    fn only_the_group_leader_may_restore_independent_cached_state() {
        let group = Group::new("cached-parent".into(), 4);
        let leader = Choice {
            group: group.clone(),
            index: 0,
            restored: false,
        };
        let mut follower = Choice {
            group,
            index: 1,
            restored: false,
        };
        assert!(reuse_eligible(None));
        assert!(reuse_eligible(Some(&leader)));
        assert!(!reuse_eligible(Some(&follower)));
        follower.restored = true;
        assert!(!reuse_eligible(Some(&follower)));
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
