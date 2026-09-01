//! Shared admission core for the TRANSLATION SURFACES (`/v1/messages`, `/v1/responses`).
//!
//! Both surfaces translate their wire format into the exact `ChatCompletionReq` the
//! OpenAI chat surface parses, then flow through THIS driver — the same canonical-model,
//! tenant-auth, budget-admission, ledger-receipt, rate-limit and worker-admission
//! sequence `chat_completions` runs, in the same order. Accounting parity is the whole
//! point: a request on any surface produces the same `[meter]` admit line, the same
//! ledger receipt discipline (admission row -> prompt usage -> per-token record ->
//! terminal complete/reject), the same prepaid-budget reservation and the same capture
//! posture. Only the response RENDERING differs, and that lives in each surface's module
//! (`anthropic.rs`, `responses_api.rs`).
//!
//! Errors leave here in OpenAI error shape (the shape every shared helper produces); the
//! Anthropic surface reshapes bodies at its boundary (`anthropic::reshape_error`) while
//! preserving status and retry headers. The Responses surface uses the OpenAI shape as-is
//! (the Responses API error body is the same `{"error": {...}}` object).
//!
//! ADDITIVE ONLY: `chat_completions` keeps its own inline copy of this sequence,
//! byte-identical to before this module existed. This is a parallel entry into the same
//! plumbing, not a refactor of the existing surface.

use axum::http::HeaderMap;
use axum::response::Response;

use crate::worker::{Cmd, Event};
use crate::{
    AppState, ChatCompletionReq, Envelope, InflightGuard, RateLimit, auth, constrained, metering,
    toolcall::{ParsedToolCall, Piece, ToolStreamParser},
    worker,
};

/// Everything a surface needs to render its response after the request was admitted:
/// the worker's event stream plus the accounting/limits state that must ride the
/// response (receipt discipline, in-flight slot, rate-limit headers).
pub(crate) struct Admission {
    pub rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    pub receipt: Option<Box<dyn crate::metering::Receipt>>,
    pub guard: InflightGuard,
    pub rl: RateLimit,
    /// Some only when a `<tools>` block was rendered or the model has a think tail —
    /// the same arming law as the chat surface (`build_chat_request_with_trace`).
    pub parser: Option<ToolStreamParser>,
    pub stop_strings: Vec<String>,
    /// The request's effective deadline (lane/deadline-billing): already spent on the
    /// admission wait in here; the surface spends the remainder on its response phase
    /// (complete response when blocking, time-to-first-token when streaming).
    pub deadline: crate::RequestDeadline,
}

/// Resolve the surface's auth token against the SAME tenant keyring / single-key /
/// open-server law as the chat surface (`auth::authenticate_with`). `candidates` lets a
/// surface accept more than one header spelling (Anthropic clients send `x-api-key`,
/// Claude Code sends `Authorization: Bearer`): each candidate is tried in order and the
/// first that authenticates wins. No candidate at all falls through to the open-server
/// rule exactly like a missing bearer on the chat surface.
pub(crate) fn authenticate_candidates(
    api_auth: &crate::ApiAuth,
    candidates: &[Option<&str>],
) -> Result<auth::TenantCtx, auth::AuthDenied> {
    let present: Vec<&str> = candidates.iter().filter_map(|c| *c).collect();
    if present.is_empty() {
        return auth::authenticate_with(api_auth.keyring, api_auth.single_key.as_deref(), None);
    }
    let mut last = auth::AuthDenied::Unknown;
    for candidate in present {
        match auth::authenticate_with(
            api_auth.keyring,
            api_auth.single_key.as_deref(),
            Some(candidate),
        ) {
            Ok(tenant) => return Ok(tenant),
            Err(why) => last = why,
        }
    }
    Err(last)
}

/// Drive one TRANSLATED chat request through the exact admission sequence of
/// `chat_completions`: namespace -> role check -> lane -> capture snapshot -> plan build
/// -> model limits -> drain gate -> budget admission -> ledger receipt -> capture arm ->
/// constraint channel -> rate-limit slot -> meter line -> worker submit -> constraint
/// wait -> admission peek. Every rejection settles its receipt through `ledger_rejected`
/// (same status/error-code rows the chat surface writes) and returns the OpenAI-shaped
/// response for the surface to reshape.
pub(crate) async fn admit_translated(
    st: &AppState,
    headers: &HeaderMap,
    env: &Envelope,
    tenant: &auth::TenantCtx,
    req: ChatCompletionReq,
    route: &'static str,
    ttft: Option<std::sync::Arc<crate::ttft::Trace>>,
) -> Result<Admission, Response> {
    let cache_ns = match crate::tenant_namespace(tenant, &req.cache_salt) {
        Ok(ns) => ns,
        Err(msg) => return Err(crate::bad_request(msg, Some("cache_salt"))),
    };
    // Request deadline (lane/deadline-billing): the ONE `parse_timeout_ms` body every
    // surface validates through; the translators pass the field through untouched so a
    // wrong type or range refuses HERE with the same named 400 as the chat surface.
    let deadline = match crate::parse_timeout_ms(req.timeout_ms.as_ref()) {
        Ok(ms) => crate::RequestDeadline::starting_now(ms),
        Err(msg) => return Err(crate::bad_request(&msg, Some("timeout_ms"))),
    };
    // Defensive twin of the chat surface's role gate: the translations only construct
    // system/user/assistant/tool turns, but a translation bug must surface as a clean
    // 400 here, never as a template render fault mid-worker.
    if req.messages.is_empty()
        || req.messages.iter().any(|message| {
            !matches!(
                message.role.as_str(),
                "system" | "developer" | "user" | "assistant" | "tool"
            )
        })
    {
        return Err(crate::bad_request(
            "messages must use system/developer/user/assistant/tool roles",
            Some("messages"),
        ));
    }
    let lane = match crate::lane_for_tenant(headers, tenant) {
        Ok(l) => l,
        Err(resp) => return Err(resp),
    };
    let model = req.model.clone();
    let stream = req.stream;
    // Read BEFORE the plan build consumes `req`: the feasibility gate judges only a
    // caller-DECLARED max_tokens (an omitted one is resolved downstream to the model max,
    // which is not a number the caller chose).
    let declared_max_tokens = req.max_tokens.is_some();
    // Capture snapshot BEFORE the plan build consumes the request — same posture as the
    // chat surface: only marked tenants pay for the copy. What is captured is the
    // TRANSLATED messages array (the internal chat shape), documented in
    // docs/API-SURFACES.md.
    let capture_prompt = st
        .metering
        .as_ref()
        .filter(|m| m.captures(&tenant.tenant))
        .map(|_| crate::capture_chat_messages(&req.messages));
    let vision_preprocess_permit = if crate::request_has_vision(&req) {
        match crate::VISION_PREPROCESS_SEMAPHORE.acquire().await {
            Ok(permit) => Some(permit),
            Err(_) => {
                return Err(crate::error_response(
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "vision preprocessing is unavailable",
                    "server_error",
                    None,
                ));
            }
        }
    } else {
        None
    };
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let affinity = crate::affinity_key(&req.session_id, &req.user, headers);
    let mut plan = match crate::build_chat_request_with_trace(
        req,
        st.caps.get(&model),
        tx,
        lane,
        affinity,
        ttft,
        st.openrouter_metadata
            .get(&model)
            .and_then(|m| m.default_reasoning_effort.as_deref()),
        // Same per-model vendor sampling defaults as /v1/chat/completions and /v1/completions:
        // the ONE `AppState::sampling_defaults` body every surface handler calls. This function
        // is the shared admission for the translated surfaces (/v1/messages, /v1/responses), so
        // all four agree by construction, not by three copies matching.
        &st.sampling_defaults(&model),
    ) {
        Ok(plan) => plan,
        Err(err) => return Err(crate::bad_request(&err, None)),
    };
    plan.request.cache_ns = cache_ns;
    plan.request.request_id = env.id.clone();
    plan.request.wire_deadline = Some(deadline.at.into_std());
    if let Err((message, param)) = crate::apply_model_request_limits(
        &mut plan.request,
        st.openrouter_metadata.get(&model),
        st.caps.get(&model),
    ) {
        return Err(crate::bad_request(&message, Some(param)));
    }
    // FEASIBILITY GATE — the SAME body /v1/chat/completions and /v1/completions call, so all
    // four surfaces refuse an impossible non-streaming request identically (standard-surface
    // law). This function is the shared admission for the translated surfaces, so one call
    // here covers /v1/messages and /v1/responses both.
    //
    // A review caught the first version of this lane landing the gate on only two of the four
    // surfaces while its own comment claimed otherwise.
    if let Err(msg) = crate::nonstream_deadline_gate(
        &plan.request,
        stream,
        deadline,
        declared_max_tokens,
        st.budget_tokenizers
            .as_ref()
            .and_then(|t| t.get(&model))
            .map(std::sync::Arc::as_ref),
    ) {
        return Err(crate::error_response_coded(
            axum::http::StatusCode::BAD_REQUEST,
            &msg,
            "invalid_request_error",
            Some("max_tokens"),
            Some("nonstream_deadline_infeasible"),
        ));
    }
    plan.vision_memory = match crate::reserve_vision_memory(&plan) {
        Ok(permit) => permit,
        Err(err) => return Err(crate::vision_memory_error_response(err, Some("messages"))),
    };
    if crate::draining() {
        let receipt = crate::start_request_receipt(
            st,
            env,
            tenant,
            &model,
            route,
            lane,
            stream,
            crate::effective_max_tokens(&plan.request),
            None,
            None,
        );
        return Err(crate::ledger_rejected(
            receipt,
            crate::drain_response(),
            "draining",
            &env.id,
        ));
    }
    let budget = match crate::admit_tenant_budget(st, tenant, &mut plan.request) {
        Ok(budget) => budget,
        Err(rejection) => {
            let (response, error_code) = rejection.into_response();
            let receipt = crate::start_request_receipt(
                st,
                env,
                tenant,
                &model,
                route,
                lane,
                stream,
                crate::effective_max_tokens(&plan.request),
                None,
                None,
            );
            return Err(crate::ledger_rejected(
                receipt, response, error_code, &env.id,
            ));
        }
    };
    let receipt = crate::start_request_receipt(
        st,
        env,
        tenant,
        &model,
        route,
        lane,
        stream,
        crate::effective_max_tokens(&plan.request),
        budget.reserved_ctx,
        budget.permit,
    );
    let receipt = if let Some(prompt) = capture_prompt {
        crate::arm_capture(receipt, move || prompt)
    } else {
        receipt
    };
    // Take the HTTP slot BEFORE vision decode. A rate-limited request must not expand a canvas,
    // while the separate preprocessing permit above keeps GIF/base64 planning bounded.
    let (guard, rl) = match crate::acquire_request_slot(st, lane, tenant, env) {
        Ok(slot) => slot,
        Err(resp) => {
            return Err(crate::ledger_rejected(
                receipt,
                resp,
                "rate_limit_exceeded",
                &env.id,
            ));
        }
    };
    // BACKPRESSURE (lane/deadline-billing): shed at submission — never after — when the
    // queue is at its bound or the estimated wait cannot fit the request's deadline.
    let pending_admit = match crate::reserve_pending_admit(st, lane, &rl, deadline) {
        Ok(guard) => guard,
        Err((resp, outcome)) => {
            return Err(crate::ledger_unbilled(
                receipt,
                rl.attach(resp),
                outcome,
                outcome,
                &env.id,
            ));
        }
    };
    // Vision phase 2 (hermes decode-bomb finding, fixed 2026-08-23): canvases expand only after
    // budget and slot admission priced the header-planned pad runs. The process-wide memory
    // permit moves into the worker request and survives streaming until completion/cancellation.
    if let Err(err) = crate::decode_pending_vision(&mut plan) {
        return Err(crate::ledger_rejected(
            receipt,
            rl.attach(crate::bad_request(&err, Some("messages"))),
            "invalid_request_error",
            &env.id,
        ));
    }
    plan.request.vision_memory = plan.vision_memory.take();
    drop(vision_preprocess_permit);
    let constraint_ready = if plan.request.grammar.is_some() {
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        plan.request.constraint_ready = Some(ready_tx);
        Some(ready_rx)
    } else {
        None
    };
    crate::meter_admit(env, tenant, &model, lane);
    let stop_strings = plan.request.stop_strings.clone();
    let parser = plan.parser;
    if st
        .cmd_tx
        .send(Cmd::Generate(Box::new(plan.request)))
        .is_err()
    {
        drop(pending_admit);
        return Err(crate::ledger_rejected(
            receipt,
            rl.attach(crate::worker_unavailable_response()),
            "worker_unavailable",
            &env.id,
        ));
    }
    pending_admit.commit();
    if let Some(ready) = constraint_ready {
        // Bounded by the request's own deadline too — a sub-5s timeout_ms must not be
        // overshot by the compile window (same law as the chat surface).
        let bound = constrained::CONSTRAINT_COMPILE_TIMEOUT.min(deadline.remaining());
        match tokio::time::timeout(bound, ready).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(err))) => {
                return Err(crate::ledger_rejected(
                    receipt,
                    rl.attach(crate::engine_error_response(&err)),
                    crate::engine_error_code(err.class),
                    &env.id,
                ));
            }
            Ok(Err(_)) => {
                return Err(crate::ledger_rejected(
                    receipt,
                    rl.attach(crate::worker_unavailable_response()),
                    "worker_unavailable",
                    &env.id,
                ));
            }
            Err(_) if deadline.remaining().is_zero() => {
                return Err(crate::ledger_unbilled(
                    receipt,
                    rl.attach(crate::deadline_exceeded_response(deadline.ms, stream)),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                ));
            }
            Err(_) => {
                return Err(crate::ledger_rejected(
                    receipt,
                    rl.attach(crate::engine_error_response(
                        &worker::constraint_timeout_error(),
                    )),
                    "constraint_compile_timeout",
                    &env.id,
                ));
            }
        }
    }
    // DEADLINE: the admission wait counts against timeout_ms; dropping rx on a miss IS
    // the cancel (the worker prunes closed-channel requests at the next tick).
    let rx = match tokio::time::timeout_at(deadline.at, crate::peek_admission(rx)).await {
        Ok(Ok(rx)) => rx,
        Ok(Err((resp, error_code))) => {
            return Err(crate::ledger_rejected(
                receipt,
                rl.attach(resp),
                error_code,
                &env.id,
            ));
        }
        Err(_) => {
            return Err(crate::ledger_unbilled(
                receipt,
                rl.attach(crate::deadline_exceeded_response(deadline.ms, stream)),
                "deadline_exceeded",
                "deadline_exceeded",
                &env.id,
            ));
        }
    };
    Ok(Admission {
        rx,
        receipt,
        guard,
        rl,
        parser,
        stop_strings,
        deadline,
    })
}

/// The terminal snapshot of one generation, surface-agnostic: what a non-streaming
/// renderer needs to build its response body.
pub(crate) struct FinalChat {
    pub text: String,
    pub reasoning: String,
    pub calls: Vec<ParsedToolCall>,
    pub stop_reason: String,
    /// Which client stop sequence fired (earliest match in the final text), for surfaces
    /// that report it (Anthropic `stop_sequence`). The matched text is already truncated
    /// out of `text`, exactly like the chat surface's `truncate_at_stop`.
    pub matched_stop: Option<String>,
    pub n_tokens: usize,
    pub n_prompt: usize,
    pub n_cached: usize,
    /// Kept for parity with the chat surface's terminal snapshot; neither dialect
    /// renders them today (Anthropic/Responses usage objects have no elapsed/spec slot).
    #[allow(dead_code)]
    pub elapsed_s: f64,
    #[allow(dead_code)]
    pub spec: Option<worker::SpecUsage>,
}

/// Why a blocking collect could not produce a `FinalChat`. In both arms the ledger
/// receipt has ALREADY been settled (rejected) — the surface only renders the error.
pub(crate) enum CollectError {
    /// A receipt sync failed: the fail-closed billing 500 (`request_ledger_error_*`).
    Ledger,
    /// The engine classified a fault (receipt rejected with its class/status).
    Engine(worker::EngineError),
}

/// Drain the worker's event stream to a terminal snapshot with EXACTLY the receipt
/// discipline of the chat surface's `blocking_response_with_receipt`: prompt usage
/// recorded when published, one completion record per token, raw-text capture deltas,
/// terminal complete/reject synced before any HTTP body is produced.
pub(crate) async fn collect_final(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Event>,
    receipt: &mut Option<Box<dyn crate::metering::Receipt>>,
    mut parser: Option<ToolStreamParser>,
    stop_strings: &[String],
    env: &Envelope,
) -> Result<FinalChat, CollectError> {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<ParsedToolCall> = Vec::new();
    let consume = |pieces: Vec<Piece>,
                   text: &mut String,
                   reasoning: &mut String,
                   calls: &mut Vec<ParsedToolCall>| {
        for piece in pieces {
            match piece {
                Piece::Content(t) => text.push_str(&t),
                Piece::Reasoning(t) => reasoning.push_str(&t),
                Piece::Call(c) => calls.push(c),
            }
        }
    };
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::PromptCapture { .. } => {} // embeddings/rerank surface only
            Event::PromptUsage { n_prompt, n_cached } => {
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.record_prompt_usage(n_prompt as u64, n_cached as u64)
                {
                    eprintln!(
                        "[ledger] ERROR: request {} partial prompt receipt failed: {err}",
                        env.id
                    );
                    // Settle as rejected (best effort) so Drop cannot classify OUR
                    // bookkeeping failure as a billable client abandon.
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return Err(CollectError::Ledger);
                }
            }
            Event::Token { id: _, text: delta } => {
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.record_completion_token()
                {
                    eprintln!(
                        "[ledger] ERROR: request {} partial completion receipt failed: {err}",
                        env.id
                    );
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return Err(CollectError::Ledger);
                }
                if let Some(receipt) = receipt.as_mut() {
                    receipt.capture_completion_delta(&delta);
                }
                match parser.as_mut() {
                    Some(p) => consume(p.push(&delta), &mut text, &mut reasoning, &mut calls),
                    None => text.push_str(&delta),
                }
            }
            Event::TokenSnapshot(_) => {}
            Event::Done {
                stop_reason,
                n_tokens,
                n_prompt,
                n_cached,
                elapsed_s,
                spec,
            } => {
                if let Some(p) = parser.as_mut() {
                    consume(p.finish(), &mut text, &mut reasoning, &mut calls);
                }
                // Earliest stop-sequence match wins, exactly like `truncate_at_stop` —
                // but the matched sequence is kept for surfaces that report it.
                let matched_stop = stop_strings
                    .iter()
                    .filter_map(|stop| text.find(stop).map(|at| (at, stop)))
                    .min_by_key(|(at, _)| *at)
                    .map(|(at, stop)| {
                        text.truncate(at);
                        stop.clone()
                    });
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.complete(
                        metering::UsageCounts {
                            prompt_tokens: n_prompt as u64,
                            cached_prompt_tokens: n_cached as u64,
                            completion_tokens: n_tokens as u64,
                        },
                        elapsed_s,
                    )
                {
                    eprintln!(
                        "[ledger] ERROR: request {} completion receipt failed: {err}",
                        env.id
                    );
                    // A pricing failure inside complete() leaves the receipt unfinalized;
                    // settle it rejected (best effort) so Drop cannot bill OUR failure.
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return Err(CollectError::Ledger);
                }
                return Ok(FinalChat {
                    text,
                    reasoning,
                    calls,
                    stop_reason,
                    matched_stop,
                    n_tokens,
                    n_prompt,
                    n_cached,
                    elapsed_s,
                    spec,
                });
            }
            Event::Error(err) => {
                if let Some(receipt) = receipt.as_mut()
                    && let Err(ledger_err) = receipt.reject(
                        crate::class_http(err.class).0.as_u16(),
                        crate::engine_error_code(err.class),
                    )
                {
                    eprintln!(
                        "[ledger] ERROR: request {} failure receipt failed: {ledger_err}",
                        env.id
                    );
                    return Err(CollectError::Ledger);
                }
                return Err(CollectError::Engine(err));
            }
        }
    }
    // Channel closed without Done/Error: worker restart in progress (same law as the
    // chat surface — 503 + Retry-After, receipt rejected).
    let e = worker::EngineError::overloaded(
        "worker closed the stream without completing (worker restart in progress)",
    );
    if let Some(receipt) = receipt.as_mut()
        && let Err(ledger_err) = receipt.reject(
            crate::class_http(e.class).0.as_u16(),
            crate::engine_error_code(e.class),
        )
    {
        eprintln!(
            "[ledger] ERROR: request {} closed-stream receipt failed: {ledger_err}",
            env.id
        );
        return Err(CollectError::Ledger);
    }
    Err(CollectError::Engine(e))
}

/// Streaming stop scrubber with matched-stop reporting: the chat surface's holdback law
/// (release text only once it can no longer start a stop string; a completed stop
/// truncates) plus `matched()` so the Anthropic surface can report `stop_sequence`.
pub(crate) struct SurfaceScrubber {
    stops: Vec<String>,
    buf: String,
    matched: Option<String>,
}

impl SurfaceScrubber {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            buf: String::new(),
            matched: None,
        }
    }

    /// Feed a content delta; returns the text now safe to emit.
    pub fn push(&mut self, text: &str) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        self.buf.push_str(text);
        if let Some((i, stop)) = self
            .stops
            .iter()
            .filter_map(|s| self.buf.find(s.as_str()).map(|at| (at, s.clone())))
            .min_by_key(|(at, _)| *at)
        {
            self.matched = Some(stop);
            let out = self.buf[..i].to_string();
            self.buf.clear();
            return out;
        }
        let keep = self
            .stops
            .iter()
            .map(|s| crate::partial_stop_suffix(&self.buf, s))
            .max()
            .unwrap_or(0);
        let emit_to = self.buf.len() - keep;
        let out = self.buf[..emit_to].to_string();
        self.buf.drain(..emit_to);
        out
    }

    /// End of stream: release held-back text (it never became a stop).
    pub fn finish(&mut self) -> String {
        if self.matched.is_some() {
            self.buf.clear();
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }

    /// Which stop sequence fired, if any.
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }
}

// ---- test support (shared by the surface modules' golden tests) -------------------------

/// Collect an SSE response body into (event-name, data-json) frames. Comment/keep-alive
/// lines are skipped; `[DONE]`-style bare sentinels would arrive as (event, Null).
#[cfg(test)]
pub(crate) async fn sse_frames(resp: Response) -> Vec<(String, serde_json::Value)> {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("sse body");
    let text = String::from_utf8(bytes.to_vec()).expect("utf8 sse body");
    let mut frames = Vec::new();
    for chunk in text.split("\n\n") {
        let mut event = String::new();
        let mut data: Option<serde_json::Value> = None;
        for line in chunk.lines() {
            if let Some(name) = line.strip_prefix("event: ") {
                event = name.trim().to_string();
            } else if let Some(payload) = line.strip_prefix("data: ") {
                data = serde_json::from_str(payload).ok();
            }
        }
        if let Some(data) = data {
            frames.push((event, data));
        }
    }
    frames
}

#[cfg(test)]
pub(crate) fn test_envelope(id: &str) -> Envelope {
    Envelope {
        id: id.to_string(),
        created: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrubber_holds_back_partial_stops_and_reports_the_match() {
        let mut sc = SurfaceScrubber::new(vec!["STOP".into()]);
        assert_eq!(sc.push("hello S"), "hello "); // "S" held back (possible stop prefix)
        assert_eq!(sc.push("T"), ""); // "ST" still a prefix
        assert_eq!(sc.push("ill going"), "STill going"); // not a stop after all
        assert_eq!(sc.push(" STOP more"), " ");
        assert_eq!(sc.matched(), Some("STOP"));
        assert_eq!(sc.push("anything"), ""); // post-stop: nothing leaks
        assert_eq!(sc.finish(), "");
    }

    #[test]
    fn scrubber_finish_releases_heldback_text_when_no_stop_fired() {
        let mut sc = SurfaceScrubber::new(vec!["<end>".into()]);
        assert_eq!(sc.push("tail <e"), "tail ");
        assert_eq!(sc.finish(), "<e");
        assert_eq!(sc.matched(), None);
    }

    #[tokio::test]
    async fn collect_final_truncates_at_stop_and_names_the_matched_sequence() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 7,
            n_cached: 2,
        })
        .unwrap();
        tx.send(Event::Token {
            id: 1,
            text: "one STOP two".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Callback".into(),
            n_tokens: 4,
            n_prompt: 7,
            n_cached: 2,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let mut receipt = None;
        let fin = collect_final(
            &mut rx,
            &mut receipt,
            None,
            &["STOP".to_string()],
            &test_envelope("t"),
        )
        .await
        .ok()
        .expect("final");
        assert_eq!(fin.text, "one ");
        assert_eq!(fin.matched_stop.as_deref(), Some("STOP"));
        assert_eq!((fin.n_prompt, fin.n_cached, fin.n_tokens), (7, 2, 4));
    }

    #[tokio::test]
    async fn collect_final_surfaces_engine_faults_as_classified_errors() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Error(worker::EngineError::overloaded("no room")))
            .unwrap();
        drop(tx);
        let mut receipt = None;
        match collect_final(&mut rx, &mut receipt, None, &[], &test_envelope("t")).await {
            Err(CollectError::Engine(e)) => assert_eq!(e.message, "no room"),
            _ => panic!("expected engine error"),
        }
    }

    #[test]
    fn authenticate_candidates_accepts_either_header_and_stays_open_when_unconfigured() {
        // Open server: no keyring, no single key -> default tenant regardless of headers.
        let open = crate::ApiAuth::default();
        assert!(authenticate_candidates(&open, &[]).is_ok());
        assert!(authenticate_candidates(&open, &[Some("whatever")]).is_ok());
        // Single-key server: the key may arrive via EITHER candidate slot (bearer or
        // x-api-key); a wrong first candidate must not shadow a correct second one.
        let keyed = crate::ApiAuth {
            keyring: None,
            single_key: Some(std::sync::Arc::from("sk-test")),
        };
        assert!(authenticate_candidates(&keyed, &[Some("sk-test"), None]).is_ok());
        assert!(authenticate_candidates(&keyed, &[Some("wrong"), Some("sk-test")]).is_ok());
        assert!(authenticate_candidates(&keyed, &[Some("wrong")]).is_err());
        assert!(authenticate_candidates(&keyed, &[]).is_err());
    }
}
