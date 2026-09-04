//! `/v1/messages` — the Anthropic Messages API served as a TRANSLATION SURFACE over the
//! chat-completions core (lane/api-surfaces, 2026-08-17).
//!
//! Why it exists: agentic clients that speak the Anthropic wire format (Messages +
//! `x-api-key`/`Authorization: Bearer`, SSE event vocabulary `message_start` ..
//! `message_stop`) can point straight at this server. The surface is TRANSLATION ONLY:
//! the request is rewritten into the exact internal `ChatCompletionReq` the OpenAI chat
//! surface parses, admission/billing/capture flow through `surfaces::admit_translated`
//! (the same sequence `chat_completions` runs), and only the response rendering differs.
//!
//! Honesty gates match the house law: semantic features this engine cannot honor return
//! a clear 400 with the Anthropic error body — never a silent downgrade. Anthropic
//! server-side tools (web_search etc.), `tool_choice: any/tool` (needs constrained
//! decoding), non-base64 image sources and `mcp_servers` all refuse loudly.
//!
//! Error bodies everywhere on this surface are the Anthropic shape:
//! `{"type":"error","error":{"type":"...","message":"..."}}` — statuses and retry
//! headers are preserved from the shared core (`reshape_error`).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header::CONTENT_TYPE};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::surfaces::{self, CollectError, SurfaceScrubber};
use crate::toolcall::Piece;
use crate::worker::Event;
use crate::{AppState, ChatCompletionReq, Envelope, Extension, TtftRequestTrace};

// ---- error shape -----------------------------------------------------------------------

/// Anthropic error type vocabulary by HTTP status (the documented mapping; statuses the
/// engine emits that Anthropic never does — 402, 503 — take the nearest honest type).
fn status_error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        // Deadline miss (lane/deadline-billing): the same `timeout` type as the OpenAI
        // surfaces — one semantic everywhere, and the message carries the billing promise.
        408 => "timeout",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        s if s >= 500 => "api_error",
        _ => "invalid_request_error",
    }
}

fn error_body(etype: &str, message: &str, request_id: &str) -> Value {
    json!({
        "type": "error",
        "error": { "type": etype, "message": message },
        "request_id": request_id,
    })
}

/// Stamp the request id as BOTH `x-request-id` (house convention, every surface) and
/// `request-id` (the Anthropic SDK's spelling — it surfaces this one to callers).
pub(crate) fn with_anthropic_request_id(id: &str, resp: Response) -> Response {
    let mut resp = crate::with_request_id(id, resp);
    if let Ok(v) = axum::http::HeaderValue::from_str(id) {
        resp.headers_mut()
            .insert(axum::http::HeaderName::from_static("request-id"), v);
    }
    resp
}

fn error_response(status: StatusCode, message: &str, request_id: &str) -> Response {
    let mut resp = (
        status,
        axum::Json(error_body(status_error_type(status), message, request_id)),
    )
        .into_response();
    if status.is_client_error()
        && status != StatusCode::TOO_MANY_REQUESTS
        && status != StatusCode::REQUEST_TIMEOUT
        && status != StatusCode::CONFLICT
    {
        resp.headers_mut().insert(
            "x-should-retry",
            axum::http::HeaderValue::from_static("false"),
        );
    }
    resp
}

fn bad_request(message: &str, request_id: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, request_id)
}

/// Rewrap an OpenAI-shaped error response (what every shared helper produces) into the
/// Anthropic error body, PRESERVING status and headers (Retry-After / retry-after-ms /
/// x-should-retry are part of the retry contract, not the body shape). Claude Code's
/// retry/degrade logic matches on Anthropic-shaped bodies, so every refusal on this
/// surface must leave through here or through `error_response`.
pub(crate) async fn reshape_error(resp: Response, request_id: &str) -> Response {
    let (mut parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, 1 << 20)
        .await
        .unwrap_or_default();
    let message = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
    let body = error_body(status_error_type(parts.status), &message, request_id).to_string();
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);
    parts.headers.insert(
        CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, axum::body::Body::from(body))
}

// ---- request translation ---------------------------------------------------------------

/// Flatten a `tool_result.content` value to text: string, null, or `{type:"text"}` blocks.
fn tool_result_text(content: Option<&Value>) -> Result<String, String> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => out.push_str(
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .ok_or("tool_result text block has no text field")?,
                    ),
                    other => {
                        return Err(format!(
                            "tool_result content block type {other:?} is not supported \
                             (text blocks only)"
                        ));
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(format!(
            "tool_result content must be a string or array, got {other}"
        )),
    }
}

/// The `system` field: a string, or an array of `{type:"text"}` blocks (cache_control
/// markers are accepted and ignored — prefix caching here is automatic, not opt-in).
fn system_text(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => out.push_str(
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .ok_or("system text block has no text field")?,
                    ),
                    other => {
                        return Err(format!("system block type {other:?} is not supported"));
                    }
                }
            }
            Ok(out)
        }
        other => Err(format!(
            "system must be a string or an array of text blocks, got {other}"
        )),
    }
}

/// Translate one Anthropic Messages request into the internal OpenAI-chat request value.
/// Every unsupported semantic feature errors HERE with a message naming the field.
pub(crate) fn translate(v: &Value) -> Result<Value, String> {
    let obj = v.as_object().ok_or("request body must be a JSON object")?;
    let model = obj
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or("model: field required")?;
    let max_tokens = obj
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .ok_or("max_tokens: field required (integer >= 1)")?;
    if max_tokens == 0 {
        return Err("max_tokens must be >= 1".into());
    }
    if obj
        .get("mcp_servers")
        .and_then(|m| m.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return Err(
            "mcp_servers is not supported (server-side MCP does not run here); \
                    call tools client-side"
                .into(),
        );
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(system) = obj.get("system").filter(|s| !s.is_null()) {
        messages.push(json!({ "role": "system", "content": system_text(system)? }));
    }
    let turns = obj
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("messages: field required (array)")?;
    for (i, msg) in turns.iter().enumerate() {
        let role = msg
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| format!("messages[{i}].role: field required"))?;
        // "system" mid-conversation is legal on the current Messages API; the internal
        // chat shape accepts system turns anywhere, so it passes straight through.
        if !matches!(role, "user" | "assistant" | "system") {
            return Err(format!(
                "messages[{i}].role must be \"user\", \"assistant\" or \"system\", got {role:?}"
            ));
        }
        let content = msg.get("content").unwrap_or(&Value::Null);
        match content {
            Value::String(s) => messages.push(json!({ "role": role, "content": s })),
            Value::Array(blocks) => {
                // Anthropic packs tool_use/tool_result as CONTENT BLOCKS; the internal
                // chat shape wants tool_calls on assistant turns and role:"tool" turns
                // for results. Split each message accordingly, preserving block order.
                let mut parts: Vec<Value> = Vec::new(); // text/image parts for this turn
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut tool_turns: Vec<Value> = Vec::new();
                for (j, block) in blocks.iter().enumerate() {
                    let at = || format!("messages[{i}].content[{j}]");
                    match block.get("type").and_then(|t| t.as_str()) {
                        Some("text") => parts.push(json!({
                            "type": "text",
                            "text": block.get("text").and_then(|t| t.as_str())
                                .ok_or_else(|| format!("{}: text block has no text", at()))?,
                        })),
                        Some("image") => {
                            let source = block
                                .get("source")
                                .ok_or_else(|| format!("{}: image block has no source", at()))?;
                            if source.get("type").and_then(|t| t.as_str()) != Some("base64") {
                                return Err(format!(
                                    "{}: only base64 image sources are supported \
                                     (http(s) fetch is disabled)",
                                    at()
                                ));
                            }
                            let media = source
                                .get("media_type")
                                .and_then(|m| m.as_str())
                                .ok_or_else(|| {
                                    format!("{}: image source has no media_type", at())
                                })?;
                            let data = source
                                .get("data")
                                .and_then(|d| d.as_str())
                                .ok_or_else(|| format!("{}: image source has no data", at()))?;
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": format!("data:{media};base64,{data}") },
                            }));
                        }
                        Some("tool_use") => {
                            if role != "assistant" {
                                return Err(format!(
                                    "{}: tool_use blocks are only valid on assistant messages",
                                    at()
                                ));
                            }
                            tool_calls.push(json!({
                                "id": block.get("id").and_then(|x| x.as_str()).unwrap_or(""),
                                "function": {
                                    "name": block.get("name").and_then(|n| n.as_str())
                                        .ok_or_else(|| format!("{}: tool_use has no name", at()))?,
                                    "arguments": block.get("input").cloned()
                                        .unwrap_or_else(|| json!({})),
                                },
                            }));
                        }
                        Some("tool_result") => {
                            if role != "user" {
                                return Err(format!(
                                    "{}: tool_result blocks are only valid on user messages",
                                    at()
                                ));
                            }
                            let text = tool_result_text(block.get("content"))
                                .map_err(|e| format!("{}: {e}", at()))?;
                            tool_turns.push(json!({
                                "role": "tool",
                                "content": text,
                                "tool_call_id": block.get("tool_use_id")
                                    .and_then(|x| x.as_str()).unwrap_or(""),
                            }));
                        }
                        // Assistant thinking history cannot be re-rendered into a chat
                        // template (it is not part of any template's message grammar);
                        // dropping it matches how the template itself strips prior think
                        // segments from history. Documented in docs/API-SURFACES.md.
                        Some("thinking") | Some("redacted_thinking") => {}
                        // Mid-conversation system content: the internal shape has system
                        // TURNS, so the block becomes its own turn ahead of this message.
                        Some("mid_conv_system") => {
                            let text = tool_result_text(block.get("content"))
                                .map_err(|e| format!("{}: {e}", at()))?;
                            tool_turns.push(json!({ "role": "system", "content": text }));
                        }
                        other => {
                            return Err(format!(
                                "{}: content block type {other:?} is not supported",
                                at()
                            ));
                        }
                    }
                }
                // tool_result turns first (Anthropic requires them at the head of the
                // user message; the internal shape wants them as standalone tool turns).
                messages.extend(tool_turns);
                if !parts.is_empty() || !tool_calls.is_empty() {
                    let mut turn = json!({ "role": role, "content": parts });
                    if !tool_calls.is_empty() {
                        turn["tool_calls"] = Value::Array(tool_calls);
                    }
                    messages.push(turn);
                }
            }
            other => {
                return Err(format!(
                    "messages[{i}].content must be a string or array of blocks, got {other}"
                ));
            }
        }
    }

    let mut tools: Vec<Value> = Vec::new();
    if let Some(ts) = obj.get("tools").and_then(|t| t.as_array()) {
        for (i, t) in ts.iter().enumerate() {
            match t.get("type").and_then(|x| x.as_str()) {
                None | Some("custom") => {}
                Some(server_tool) => {
                    return Err(format!(
                        "tools[{i}]: server tool type {server_tool:?} is not supported \
                         (client-defined tools only)"
                    ));
                }
            }
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").and_then(|n| n.as_str())
                        .ok_or_else(|| format!("tools[{i}].name: field required"))?,
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({})),
                },
            }));
        }
    }

    let tool_choice = match obj.get("tool_choice") {
        None | Some(Value::Null) => Value::Null,
        Some(tc) => match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some(other @ ("any" | "tool")) => {
                return Err(format!(
                    "tool_choice type {other:?} is not supported (forcing a tool call \
                     needs constrained decoding); use \"auto\" or \"none\""
                ));
            }
            _ => return Err(format!("bad tool_choice: {tc}")),
        },
    };

    // Extended thinking + output_config.effort map to the ONE reasoning surface
    // (`parse_think`'s table). thinking.type: enabled -> thinking ON (model-native
    // mechanism), disabled -> thinking OFF, adaptive -> the model's own default (Claude Code
    // sends {"type":"adaptive"} unconditionally for unrecognized model ids, and "the model
    // decides" IS what our model-default arm does, so that one is honoured rather than
    // tolerated). Any OTHER `type` string is now a named 400: it used to fall through the same
    // lenient arm and be silently ignored, which is the accepted-and-ignored class the
    // standard-surface law bans (lane/reasoning-schema-20260823).
    //
    // output_config.effort is the Anthropic effort lever (Claude Code sends it on every
    // request — "xhigh" by default on current models; vLLM's Anthropic surface forwards
    // it verbatim). It flows through AS `reasoning.effort` so parse_think's allowlist
    // validates it and none/minimal suppress thinking exactly as on the other two
    // surfaces. Issue #31: this field used to be dropped here, so /v1/messages accepted
    // every string (bogus/banana/"" -> 200) and no value had any effect.
    //
    // PRECEDENCE when both are present: thinking.type wins the on/off switch (it is the
    // documented Anthropic lever, and it maps to `reasoning.enabled`, which parse_think
    // gives explicit-switch precedence); the validated effort still rides as the level
    // for level-consuming templates.
    let switch = match obj.get("thinking") {
        None | Some(Value::Null) => None,
        Some(th) => {
            // `budget_tokens` CANNOT be honoured on this server, and the owner ruled it must
            // not be faked: reasoning tokens are output tokens, `max_tokens` is the ONE output
            // budget covering reasoning and content together, and building a separate
            // reasoning budget is explicitly out of scope. There is also no engine lever for
            // it — nothing can force-close a `<think>` span after N tokens (GenParams carries
            // only max_new/max_ctx/eos, and a `</think>` stop string ends the whole request
            // with an empty answer). So it is a named 400 rather than the silent accept it was:
            // a caller who sets a thinking budget to control spend must not be told 200 and
            // then billed for an unbounded reasoning block.
            // A NON-OBJECT `thinking` (`true`, `"enabled"`, `[]`) used to slip through every
            // check below, because `get()` on a non-object is always `None` — the same
            // accepted-and-ignored shape the chat surface refuses with "reasoning must be an
            // object". One schema, one answer.
            let th_map = th
                .as_object()
                .ok_or_else(|| format!("thinking must be an object, got {th}"))?;
            // ...and unknown KEYS inside a well-formed object were silently dropped, including
            // a hosted-reseller's `max_thinking_tokens` spelling of the budget. Iterated like
            // `output_config` below, so SERVING.md's "every reasoning field this server cannot
            // act on is a named 400, on every surface" is true rather than aspirational.
            if let Some(other) = th_map
                .keys()
                .find(|k| !matches!(k.as_str(), "type" | "budget_tokens"))
            {
                return Err(format!(
                    "thinking.{other} is not supported by this server; the supported keys are \
                     type and budget_tokens (and budget_tokens itself cannot be honoured — see \
                     below)"
                ));
            }
            if let Some(budget) = th.get("budget_tokens")
                && !budget.is_null()
            {
                return Err(format!(
                    "thinking.budget_tokens ({budget}) is not supported by this server: \
                         reasoning tokens are output tokens here and max_tokens is the ONE \
                         output budget covering reasoning and content together, so there is no \
                         separate reasoning budget to cap. Send max_tokens for the budget, and \
                         output_config.effort (low|medium|high) to spend less of it on \
                         reasoning, or thinking:{{\"type\":\"disabled\"}} for none at all"
                ));
            }
            // Matched on the RAW value, not through `as_str()`: that conflated "absent" with
            // "wrong type", so `{"type": 3}` landed in the unset arm and returned 200 while an
            // unknown *string* type was already a 400.
            match th_map.get("type") {
                None | Some(Value::Null) => None,
                Some(Value::String(t)) => match t.as_str() {
                    "enabled" => Some(true),
                    "disabled" => Some(false),
                    // Anthropic's "the model decides" — which is exactly this server's
                    // model-default arm, so it is honoured, not ignored.
                    "adaptive" => None,
                    other => {
                        return Err(format!(
                            "thinking.type {other:?} is not supported (enabled|disabled|adaptive)"
                        ));
                    }
                },
                Some(other) => {
                    return Err(format!("thinking.type must be a string, got {other}"));
                }
            }
        }
    };
    let effort = match obj.get("output_config") {
        None | Some(Value::Null) => None,
        Some(cfg) => {
            // Every other `output_config` key was silently ignored. Name them: an output_config
            // this server cannot act on must not return 200 as though it had. The `as_object`
            // guard is an `ok_or_else`, not an `if let` — a NON-object `output_config`
            // ("high", []) used to skip the key check entirely and be ignored wholesale, which
            // is the same defect wearing a scalar.
            let cfg_map = cfg
                .as_object()
                .ok_or_else(|| format!("output_config must be an object, got {cfg}"))?;
            if let Some(other) = cfg_map.keys().find(|k| k.as_str() != "effort") {
                return Err(format!(
                    "output_config.{other} is not supported by this server; the only \
                     supported key is effort"
                ));
            }
            match cfg_map.get("effort") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(other) => {
                    return Err(format!(
                        "output_config.effort must be a string, got {other}"
                    ));
                }
            }
        }
    };
    let reasoning = {
        let mut r = serde_json::Map::new();
        if let Some(enabled) = switch {
            r.insert("enabled".into(), json!(enabled));
        }
        if let Some(effort) = effort {
            r.insert("effort".into(), json!(effort));
        }
        if r.is_empty() {
            Value::Null
        } else {
            Value::Object(r)
        }
    };

    let mut out = json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": obj.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
    });
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
    }
    if !tool_choice.is_null() {
        out["tool_choice"] = tool_choice;
    }
    if !reasoning.is_null() {
        out["reasoning"] = reasoning;
    }
    for (theirs, ours) in [
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("stop_sequences", "stop"),
        // Deadline (lane/deadline-billing): passed through UNVALIDATED — the one
        // `parse_timeout_ms` body in the shared admission refuses bad values with the
        // same named 400 on every surface.
        ("timeout_ms", "timeout_ms"),
    ] {
        if let Some(v) = obj.get(theirs).filter(|v| !v.is_null()) {
            out[ours] = v.clone();
        }
    }
    // metadata.user_id is the caller's stable conversation identity — the same session
    // affinity nomination the chat surface reads from `user`.
    if let Some(user_id) = obj
        .get("metadata")
        .and_then(|m| m.get("user_id"))
        .and_then(|u| u.as_str())
    {
        out["user"] = json!(user_id);
    }
    Ok(out)
}

// ---- response rendering ----------------------------------------------------------------

/// Anthropic stop_reason: tool calls win (the client must run them), then a fired stop
/// sequence, then the token budget, else a natural end of turn.
fn stop_reason(worker_reason: &str, has_calls: bool, matched_stop: bool) -> &'static str {
    if has_calls {
        return "tool_use";
    }
    if matched_stop {
        return "stop_sequence";
    }
    match worker_reason {
        "MaxNew" | "ContextFull" => "max_tokens",
        _ => "end_turn",
    }
}

/// `tool_use.input` must be a JSON object. The emission parser only surfaces calls whose
/// arguments parsed (malformed blocks pass through as content), so the fallback arm is
/// defensive: the raw string is preserved under `_raw_arguments` rather than dropped.
fn tool_input(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(v @ Value::Object(_)) => v,
        _ => json!({ "_raw_arguments": arguments }),
    }
}

fn usage_json(n_prompt: usize, n_tokens: usize, n_cached: usize) -> Value {
    // Honest cache accounting: `cache_read_input_tokens` is worker-truth (prompt tokens
    // whose KV was resumed from a cache instead of computed). There is no separate
    // "cache write" billing tier here, so cache_creation_input_tokens is honestly 0.
    json!({
        "input_tokens": n_prompt.saturating_sub(n_cached),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": n_cached,
        "output_tokens": n_tokens,
    })
}

fn message_json(env: &Envelope, model: &str, fin: &surfaces::FinalChat) -> Value {
    let mut content: Vec<Value> = Vec::new();
    if !fin.reasoning.is_empty() {
        content.push(json!({ "type": "thinking", "thinking": fin.reasoning, "signature": "" }));
    }
    if !fin.text.is_empty() {
        content.push(json!({ "type": "text", "text": fin.text }));
    }
    for call in &fin.calls {
        content.push(json!({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": tool_input(&call.arguments),
        }));
    }
    json!({
        "id": env.id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(&fin.stop_reason, !fin.calls.is_empty(), fin.matched_stop.is_some()),
        "stop_sequence": fin.matched_stop,
        "usage": usage_json(fin.n_prompt, fin.n_tokens, fin.n_cached),
    })
}

/// One named SSE frame: `event: <type>` + `data: {"type": <type>, ...}` — the Anthropic
/// framing (clients dispatch on both).
fn frame(data: Value) -> SseEvent {
    let name = data
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("message_delta")
        .to_string();
    SseEvent::default().event(name).data(data.to_string())
}

// ---- handler ---------------------------------------------------------------------------

/// POST /v1/messages. `anthropic-version` is accepted and not enforced (this surface has
/// exactly one wire dialect); auth accepts BOTH `x-api-key` and `Authorization: Bearer`
/// against the same tenant keyring as every other surface.
#[cfg(test)]
pub(crate) async fn messages(
    state: State<AppState>,
    headers: HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    body: Bytes,
) -> Response {
    messages_with_admission(state, headers, trace, None, body).await
}

pub(crate) async fn messages_admitted(
    state: State<AppState>,
    headers: HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    body_admission: Option<Extension<crate::BodyAdmissionGuard>>,
    body: Bytes,
) -> Response {
    messages_with_admission(state, headers, trace, body_admission, body).await
}

async fn messages_with_admission(
    State(st): State<AppState>,
    headers: HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    body_admission: Option<Extension<crate::BodyAdmissionGuard>>,
    body: Bytes,
) -> Response {
    let env = Envelope {
        id: format!("msg_{}", crate::gen_hex128()),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            return with_anthropic_request_id(
                &env.id,
                bad_request(&format!("invalid JSON: {err}"), &env.id),
            );
        }
    };
    let translated = match translate(&parsed) {
        Ok(v) => v,
        Err(msg) => return with_anthropic_request_id(&env.id, bad_request(&msg, &env.id)),
    };
    let mut req: ChatCompletionReq = match serde_json::from_value(translated) {
        Ok(r) => r,
        Err(err) => {
            return with_anthropic_request_id(
                &env.id,
                bad_request(&format!("invalid request: {err}"), &env.id),
            );
        }
    };
    match crate::canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return with_anthropic_request_id(
                &env.id,
                reshape_error(
                    crate::model_not_found_response(&st.models, &req.model),
                    &env.id,
                )
                .await,
            );
        }
    }
    let ttft = trace.and_then(|Extension(trace)| trace.0);
    if let Some(trace) = ttft.as_ref() {
        trace.mark_parsed();
        trace.bind_request(&env.id, &req.model);
    }
    let token_header = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());
    let tenant = match surfaces::authenticate_candidates(
        &st.api_auth,
        &[crate::bearer_token(&headers), token_header],
    ) {
        Ok(tenant) => tenant,
        Err(why) => {
            return with_anthropic_request_id(
                &env.id,
                reshape_error(crate::authentication_error(why), &env.id).await,
            );
        }
    };
    let model = req.model.clone();
    let stream = req.stream;
    let admitted = surfaces::admit_translated(
        &st,
        &headers,
        &env,
        &tenant,
        req,
        "/v1/messages",
        ttft,
        body_admission
            .as_ref()
            .map(|Extension(admission)| admission),
    )
    .await;
    let admission = match admitted {
        Ok(a) => a,
        Err(resp) => {
            return with_anthropic_request_id(&env.id, reshape_error(resp, &env.id).await);
        }
    };
    let surfaces::Admission {
        mut rx,
        mut receipt,
        guard,
        rl,
        parser,
        stop_strings,
        deadline,
    } = admission;
    if stream {
        // Streaming: timeout_ms bounds TIME-TO-FIRST-TOKEN only. The ordinary <=90 s
        // path stays pre-header; an explicitly extended deployment may commit during
        // prefill for SSE keepalives and terminate a later miss in-band, zero debit.
        let rx = match crate::peek_first_token(rx, deadline).await {
            Ok(rx) => rx,
            Err(()) => {
                drop(guard);
                return rl.attach(with_anthropic_request_id(
                    &env.id,
                    crate::ledger_unbilled(
                        receipt,
                        reshape_error(
                            crate::deadline_exceeded_response(deadline.ms, true),
                            &env.id,
                        )
                        .await,
                        "deadline_exceeded",
                        "deadline_exceeded",
                        &env.id,
                    ),
                ));
            }
        };
        let resp = messages_sse(
            rx,
            receipt,
            env.clone(),
            model,
            parser,
            stop_strings,
            Some(guard),
        )
        .into_response();
        return rl.attach(with_anthropic_request_id(&env.id, resp));
    }
    // Non-streaming: the deadline covers the COMPLETE response. On a miss the collect
    // future is dropped and rx with it (cancelling generation at the worker's next tick);
    // the receipt settles deadline_exceeded, debit ZERO.
    //
    // THIS SURFACE KEEPS THE 408 DELIBERATELY (lane/deadline-partial-20260826). The
    // OpenAI-dialect surfaces now DELIVER what was generated when the deadline lands,
    // carried by `finish_reason: "error"` + an error object — a shape the OpenRouter dialect
    // defines and this server already speaks. The Anthropic dialect has no such shape: its
    // `stop_reason` enum is end_turn | max_tokens | stop_sequence | tool_use | pause_turn |
    // refusal | model_context_window_exceeded, with NO time value, so a partial here could
    // only be labelled `max_tokens` — the same lie this lane refused to tell on the other
    // surfaces (it tells the caller to ask for more tokens when the truth is that it must
    // stream). Anthropic itself answers a long non-streaming request with an error and
    // requires streaming for long work, so erroring is also the dialect-faithful behaviour.
    // What DID change for this surface is the feasibility gate in `surfaces::admit_translated`: an
    // impossible request is now refused in ~0.1 s with actionable advice instead of burning
    // the full deadline first.
    let collected = tokio::time::timeout_at(
        deadline.at,
        surfaces::collect_final(&mut rx, &mut receipt, parser, &stop_strings, &env),
    )
    .await;
    let fin = match collected {
        Ok(Ok(fin)) => fin,
        Ok(Err(CollectError::Ledger)) => {
            drop(guard);
            return rl.attach(with_anthropic_request_id(
                &env.id,
                reshape_error(crate::request_ledger_error_response(), &env.id).await,
            ));
        }
        Ok(Err(CollectError::Engine(e))) => {
            drop(guard);
            return rl.attach(with_anthropic_request_id(
                &env.id,
                reshape_error(crate::engine_error_response(&e), &env.id).await,
            ));
        }
        Ok(Err(CollectError::Deadline(ms))) => {
            drop(guard);
            return rl.attach(with_anthropic_request_id(
                &env.id,
                reshape_error(crate::deadline_exceeded_response(ms, false), &env.id).await,
            ));
        }
        Err(_) => {
            drop(rx); // the cancel signal: the worker prunes the closed channel
            drop(guard);
            return rl.attach(with_anthropic_request_id(
                &env.id,
                crate::ledger_unbilled(
                    receipt,
                    reshape_error(
                        crate::deadline_exceeded_response(deadline.ms, false),
                        &env.id,
                    )
                    .await,
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                ),
            ));
        }
    };
    let resp = axum::Json(message_json(&env, &model, &fin)).into_response();
    drop(guard);
    rl.attach(with_anthropic_request_id(&env.id, resp))
}

/// The Anthropic streaming vocabulary over the worker's event stream, with the SAME
/// receipt discipline as the chat SSE path (prompt usage retained on disconnect, one
/// completion record per token, terminal complete/reject before the stream closes).
///
/// Event order: message_start (real input_tokens — published at admission, before the
/// first token) -> ping -> content blocks (thinking / text / tool_use, indexed, opened
/// and closed as the generation moves between them) -> message_delta (stop_reason +
/// final usage) -> message_stop. Mid-stream faults emit a named `error` event and close.
// unused_assignments: the block state machine (open/index/started) is written by a
// macro at every transition; the compiler flags the writes of the FINAL transition as
// dead. Real state, false positive.
#[allow(clippy::too_many_arguments, unused_assignments)]
fn messages_sse(
    mut rx: crate::worker::EventReceiver,
    mut receipt: Option<Box<dyn crate::metering::Receipt>>,
    env: Envelope,
    model: String,
    mut parser: Option<crate::toolcall::ToolStreamParser>,
    stop_strings: Vec<String>,
    guard: Option<crate::InflightGuard>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    let mut scrub = (!stop_strings.is_empty()).then(|| SurfaceScrubber::new(stop_strings.clone()));
    let stream = async_stream::stream! {
        let _guard = guard;
        // Block state: index of the NEXT block; what's open now.
        #[derive(PartialEq, Clone, Copy)]
        enum Open { None, Thinking, Text }
        let mut index: usize = 0;
        let mut open = Open::None;
        let mut started = false;
        let mut prompt_usage: (usize, usize) = (0, 0); // (n_prompt, n_cached)
        macro_rules! ensure_started {
            () => {
                if !started {
                    started = true;
                    yield Ok(frame(json!({
                        "type": "message_start",
                        "message": {
                            "id": env.id, "type": "message", "role": "assistant",
                            "model": model, "content": [],
                            "stop_reason": null, "stop_sequence": null,
                            "usage": usage_json(prompt_usage.0, 0, prompt_usage.1),
                        },
                    })));
                    yield Ok(frame(json!({ "type": "ping" })));
                }
            };
        }
        macro_rules! close_block {
            () => {
                if open != Open::None {
                    if open == Open::Thinking {
                        // The real API closes thinking blocks with a signature delta;
                        // emit the frame so strict clients see the full grammar. There
                        // is no signing key here — the signature is honestly empty.
                        yield Ok(frame(json!({
                            "type": "content_block_delta", "index": index,
                            "delta": { "type": "signature_delta", "signature": "" },
                        })));
                    }
                    yield Ok(frame(json!({ "type": "content_block_stop", "index": index })));
                    index += 1;
                    open = Open::None;
                }
            };
        }
        // Renders one parsed Piece into zero or more frames.
        macro_rules! piece_frames {
            ($piece:expr) => {{
                match $piece {
                    Piece::Content(text) => {
                        let text = match scrub.as_mut() {
                            Some(sc) => sc.push(&text),
                            None => text,
                        };
                        if !text.is_empty() {
                            if open != Open::Text {
                                close_block!();
                                yield Ok(frame(json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "text", "text": "" },
                                })));
                                open = Open::Text;
                            }
                            yield Ok(frame(json!({
                                "type": "content_block_delta", "index": index,
                                "delta": { "type": "text_delta", "text": text },
                            })));
                        }
                    }
                    Piece::Reasoning(text) => {
                        if open != Open::Thinking {
                            close_block!();
                            yield Ok(frame(json!({
                                "type": "content_block_start", "index": index,
                                "content_block": { "type": "thinking", "thinking": "" },
                            })));
                            open = Open::Thinking;
                        }
                        yield Ok(frame(json!({
                            "type": "content_block_delta", "index": index,
                            "delta": { "type": "thinking_delta", "thinking": text },
                        })));
                    }
                    Piece::Call(call) => {
                        close_block!();
                        yield Ok(frame(json!({
                            "type": "content_block_start", "index": index,
                            "content_block": {
                                "type": "tool_use", "id": call.id, "name": call.name,
                                "input": {},
                            },
                        })));
                        yield Ok(frame(json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": call.arguments,
                            },
                        })));
                        yield Ok(frame(json!({
                            "type": "content_block_stop", "index": index,
                        })));
                        index += 1;
                    }
                }
            }};
        }
        macro_rules! stream_fault {
            ($etype:expr, $message:expr) => {
                yield Ok(frame(json!({
                    "type": "error",
                    "error": { "type": $etype, "message": $message },
                })));
            };
        }
        // Set by every arm that BREAKS with its receipt handled; false when the worker
        // closed the channel without Done/Error — settled rejected below, never billed.
        let mut terminal = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                Event::PromptCapture { .. } => {} // embeddings/rerank surface only
                Event::PromptUsage { n_prompt, n_cached } => {
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.record_prompt_usage(
                            n_prompt as u64,
                            n_cached as u64,
                        )
                    {
                        eprintln!(
                            "[ledger] ERROR: request {} partial prompt receipt failed: {err}",
                            env.id
                        );
                        // Settle as rejected (best effort) so Drop cannot classify OUR
                        // bookkeeping failure as a billable client abandon.
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        stream_fault!(
                            "api_error",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    prompt_usage = (n_prompt, n_cached);
                    ensure_started!();
                }
                Event::DeadlineExceeded { ms } => {
                    let ledger_error = if let Some(receipt) = receipt.as_mut() {
                        receipt
                            .settle_unbilled(
                                "deadline_exceeded",
                                StatusCode::REQUEST_TIMEOUT.as_u16(),
                                "deadline_exceeded",
                            )
                            .err()
                    } else {
                        None
                    };
                    if ledger_error.is_some() {
                        stream_fault!(
                            "api_error",
                            "request completion could not be committed to the billing ledger"
                        );
                    } else {
                        stream_fault!(
                            "timeout",
                            crate::deadline_exceeded_message(ms, true)
                        );
                    }
                    terminal = true;
                    break;
                }
                Event::Token { id: _, text } => {
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.record_completion_token()
                    {
                        eprintln!(
                            "[ledger] ERROR: request {} partial completion receipt failed: {err}",
                            env.id
                        );
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        stream_fault!(
                            "api_error",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    if let Some(receipt) = receipt.as_mut() {
                        receipt.capture_completion_delta(&text);
                    }
                    ensure_started!();
                    match parser.as_mut() {
                        Some(p) => {
                            for piece in p.push(&text) {
                                piece_frames!(piece);
                            }
                        }
                        None => piece_frames!(Piece::Content(text)),
                    }
                }
                Event::TokenSnapshot(_) => {}
                Event::Done { stop_reason: reason, n_tokens, n_prompt, n_cached, elapsed_s, spec: _ } => {
                    let mut n_calls = 0;
                    if let Some(p) = parser.as_mut() {
                        for piece in p.finish() {
                            piece_frames!(piece);
                        }
                        n_calls = p.n_calls();
                    }
                    if let Some(sc) = scrub.as_mut() {
                        let tail = sc.finish();
                        if !tail.is_empty() {
                            piece_frames!(Piece::Content(tail));
                        }
                    }
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.complete(
                            crate::metering::UsageCounts {
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
                        // Pricing failure inside complete() leaves the receipt
                        // unfinalized; settle rejected so Drop cannot bill OUR failure.
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        stream_fault!(
                            "api_error",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    ensure_started!();
                    close_block!();
                    let matched = scrub.as_ref().and_then(|sc| sc.matched().map(str::to_string));
                    yield Ok(frame(json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(&reason, n_calls > 0, matched.is_some()),
                            "stop_sequence": matched,
                        },
                        "usage": usage_json(n_prompt, n_tokens, n_cached),
                    })));
                    yield Ok(frame(json!({ "type": "message_stop" })));
                    terminal = true;
                    break;
                }
                Event::Error(err) => {
                    let ledger_error = if let Some(receipt) = receipt.as_mut() {
                        receipt
                            .reject(
                                crate::class_http(err.class).0.as_u16(),
                                crate::engine_error_code(err.class),
                            )
                            .err()
                    } else {
                        None
                    };
                    if let Some(ref ledger_error) = ledger_error {
                        eprintln!(
                            "[ledger] ERROR: request {} failure receipt failed: {ledger_error}",
                            env.id
                        );
                        stream_fault!(
                            "api_error",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    let (status, _, _) = crate::class_http(err.class);
                    stream_fault!(status_error_type(status), err.message);
                    terminal = true;
                    break;
                }
            }
        }
        if !terminal {
            // Channel closed without Done/Error: worker restart — OUR fault, settled
            // rejected with debit ZERO (fault-attribution ruling 2026-08-23; this used to
            // fall through to Drop and bill the partial stream as a client "abandon").
            let e = crate::worker::EngineError::overloaded(
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
            }
            let (status, _, _) = crate::class_http(e.class);
            stream_fault!(status_error_type(status), e.message);
        }
    };
    Sse::new(stream).keep_alive(
        // Long prefill streams nothing until admission; comment keep-alives every 5s are
        // legal SSE and ignored by Anthropic SDK parsers (ping EVENTS are sent once at
        // message_start, matching the real API).
        axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(5)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::{sse_frames, test_envelope};
    use crate::toolcall::ToolStreamParser;
    use std::collections::HashMap;

    /// The Claude-Code-shaped request (system block array with cache_control, tool_use /
    /// tool_result content blocks, adaptive thinking, metadata.user_id) translates into
    /// the exact internal chat shape — and that shape deserializes into the same
    /// ChatCompletionReq the OpenAI surface parses.
    #[test]
    fn translate_maps_the_full_agentic_request_shape() {
        let translated = translate(&json!({
            "model": "m",
            "max_tokens": 128,
            "system": [
                {"type": "text", "text": "sys A", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": " + B"}
            ],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "calling"},
                    {"type": "thinking", "thinking": "hmm", "signature": ""},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "sunny"}]},
                    {"type": "text", "text": "and?"}
                ]}
            ],
            "tools": [{"name": "get_weather", "description": "d",
                       "input_schema": {"type": "object",
                                        "properties": {"city": {"type": "string"}}},
                       "cache_control": {"type": "ephemeral"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
            "stop_sequences": ["STOP"],
            "temperature": 0.5, "top_p": 0.9, "top_k": 40,
            "thinking": {"type": "adaptive"},
            "metadata": {"user_id": "u-1"},
            "stream": true
        }))
        .expect("translate");
        let msgs = translated["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "sys A + B");
        assert_eq!(msgs[1]["content"], "hi");
        // assistant turn: thinking block dropped (template law), text + tool_calls kept.
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"][0]["text"], "calling");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            msgs[2]["tool_calls"][0]["function"]["arguments"]["city"],
            "Paris"
        );
        // tool_result becomes a standalone tool turn AHEAD of the remaining user text.
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["content"], "sunny");
        assert_eq!(msgs[3]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[4]["role"], "user");
        assert_eq!(msgs[4]["content"][0]["text"], "and?");
        // tools/tool_choice/sampling passthrough.
        assert_eq!(translated["tools"][0]["type"], "function");
        assert_eq!(translated["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            translated["tools"][0]["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
        assert_eq!(translated["tool_choice"], "auto");
        assert_eq!(translated["stop"], json!(["STOP"]));
        assert_eq!(translated["top_k"], 40);
        assert_eq!(translated["max_tokens"], 128);
        assert_eq!(translated["user"], "u-1");
        assert_eq!(translated["stream"], true);
        // adaptive thinking = the model's own default: no reasoning override.
        assert!(translated.get("reasoning").is_none());
        // The whole thing parses as the internal request type.
        let req: ChatCompletionReq = serde_json::from_value(translated).expect("internal shape");
        assert_eq!(req.model, "m");
        assert_eq!(req.max_tokens, Some(128));
    }

    #[test]
    fn omitted_sampling_stays_absent_through_translation() {
        // STANDARD-SURFACE LAW, /v1/messages half (lane/vendor-default-sampling, 2026-08-19).
        // The per-model vendor default substitutes only for fields the client left out, so the
        // translator must carry "the client said nothing" through as an ABSENT key. If it
        // zero-filled or 1.0-filled here, an Anthropic-surface client could never receive the
        // model's vendor-recommended sampling — /v1/messages would silently disagree with
        // /v1/chat/completions on the same model.
        let translated = translate(&json!({
            "model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "x"}]
        }))
        .expect("translate");
        for key in ["temperature", "top_p", "top_k"] {
            assert!(
                translated.get(key).is_none(),
                "omitted {key} must translate to an ABSENT key, not a filled one: {translated}"
            );
        }
        let req: ChatCompletionReq = serde_json::from_value(translated).expect("internal shape");
        assert_eq!(
            req.temperature, None,
            "omitted => None, so the vendor default applies"
        );
        assert_eq!(req.top_p, None);
        assert_eq!(req.top_k, None);

        // An EXPLICIT zero must survive translation as an explicit zero — this is the path a
        // client takes to demand greedy on the Anthropic surface.
        let greedy = translate(&json!({
            "model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0
        }))
        .expect("translate");
        let req: ChatCompletionReq = serde_json::from_value(greedy).expect("internal shape");
        assert_eq!(
            req.temperature,
            Some(0.0),
            "an explicit temperature 0 must reach the engine as Some(0.0) = true greedy"
        );

        // A JSON null is the SDK's way of saying "not set" and must not become Some(0.0).
        let nulled = translate(&json!({
            "model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "x"}],
            "temperature": null, "top_p": null, "top_k": null
        }))
        .expect("translate");
        let req: ChatCompletionReq = serde_json::from_value(nulled).expect("internal shape");
        assert_eq!(req.temperature, None, "null is absence, not greedy");
        assert_eq!(req.top_p, None);
        assert_eq!(req.top_k, None);
    }

    #[test]
    fn translate_refuses_what_the_engine_cannot_honor() {
        // Anthropic server-side tools do not run here.
        let err = translate(&json!({
            "model": "m", "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        }))
        .unwrap_err();
        assert!(err.contains("server tool"), "got: {err}");
        // Forcing a tool call needs constrained decoding.
        let err = translate(&json!({
            "model": "m", "max_tokens": 1,
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "any"}
        }))
        .unwrap_err();
        assert!(err.contains("constrained"), "got: {err}");
        // URL image sources would require server-side fetch (disabled everywhere).
        let err = translate(&json!({
            "model": "m", "max_tokens": 1,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
            ]}]
        }))
        .unwrap_err();
        assert!(err.contains("base64"), "got: {err}");
        // max_tokens is required on this API.
        let err = translate(&json!({
            "model": "m", "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap_err();
        assert!(err.contains("max_tokens"), "got: {err}");
        // thinking enabled/disabled map; the mapping is exercised via translate output.
        let on = translate(&json!({
            "model": "m", "max_tokens": 1, "thinking": {"type": "enabled"},
            "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap();
        assert_eq!(on["reasoning"]["enabled"], true);
        // `budget_tokens` is a NAMED 400 (lane/reasoning-schema-20260823). It used to be
        // accepted and never read, validated or enforced — a client capping thinking spend got
        // 200 and an unbounded reasoning block. There is no engine lever to cap a segment
        // (GenParams is max_new/max_ctx/eos, and a `</think>` stop string ends the whole
        // request), and by owner ruling reasoning tokens ARE output tokens under the single
        // `max_tokens` budget, so the field is unhonourable by design rather than unfinished.
        let err = translate(&json!({
            "model": "m", "max_tokens": 1, "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap_err();
        assert!(err.contains("thinking.budget_tokens"), "got: {err}");
        assert!(
            err.contains("max_tokens is the ONE output budget"),
            "got: {err}"
        );
        // and an unknown thinking.type is named rather than silently treated as the default.
        let err = translate(&json!({
            "model": "m", "max_tokens": 1, "thinking": {"type": "extended"},
            "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap_err();
        assert!(err.contains("thinking.type"), "got: {err}");
        // "adaptive" IS honoured — Anthropic's "the model decides" is this server's default arm.
        translate(&json!({
            "model": "m", "max_tokens": 1, "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "x"}]
        }))
        .unwrap();
    }

    #[test]
    fn malformed_thinking_and_output_config_refuse_like_the_other_surfaces() {
        // Found by review of this lane's first cut, and every row here was a 200 that changed
        // nothing — the accepted-and-ignored class the lane exists to remove, and a live
        // cross-surface divergence: chat already refused the analogous malformations of
        // `reasoning` by name while /v1/messages waved them through.
        let req = |extra: Value| {
            let mut body = json!({
                "model": "m", "max_tokens": 16,
                "messages": [{"role": "user", "content": "x"}]
            });
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            body
        };
        for (extra, want) in [
            // a NON-OBJECT thinking slipped past every check, because get() on a non-object is
            // always None.
            (json!({"thinking": true}), "thinking must be an object"),
            (json!({"thinking": "enabled"}), "thinking must be an object"),
            (json!({"thinking": []}), "thinking must be an object"),
            // a WRONG-TYPED type read as "absent", while an unknown STRING type already 400'd.
            (
                json!({"thinking": {"type": 3}}),
                "thinking.type must be a string",
            ),
            // unknown keys inside a well-formed object were dropped — including a hosted-reseller's
            // spelling of the budget, which is exactly the field we refuse by name.
            (
                json!({"thinking": {"type": "enabled", "max_thinking_tokens": 8192}}),
                "thinking.max_thinking_tokens",
            ),
            // ...and a non-object output_config skipped the key check entirely.
            (
                json!({"output_config": "high"}),
                "output_config must be an object",
            ),
            (
                json!({"output_config": []}),
                "output_config must be an object",
            ),
        ] {
            let err = translate(&req(extra.clone()))
                .err()
                .unwrap_or_else(|| panic!("{extra} must not be accepted-and-ignored"));
            assert!(err.contains(want), "{extra}: wanted {want:?}, got: {err}");
        }
        // The well-formed shapes still pass, including the ones stock clients send.
        for extra in [
            json!({"thinking": {"type": "adaptive"}}),
            json!({"thinking": {"type": "enabled"}}),
            json!({"thinking": {"type": "disabled"}}),
            json!({"thinking": null}),
            json!({"output_config": {"effort": "xhigh"}}),
            json!({"output_config": {}}),
            // JSON null reads as "not set" on every surface, which is the divergence this lane
            // also closed on the chat side.
            json!({"output_config": {"effort": null}}),
            json!({"thinking": {"type": "enabled", "budget_tokens": null}}),
        ] {
            translate(&req(extra.clone()))
                .unwrap_or_else(|e| panic!("{extra} must be served: {e}"));
        }
    }

    /// Issue #31 (standard-surface law): `output_config.effort` flows onto the ONE
    /// reasoning surface — translated to `reasoning.effort` VERBATIM so `parse_think`'s
    /// allowlist is the validator (a second table here is how surfaces drift). A mutation
    /// that drops the field again fails HERE by name; the acceptance/effect halves are
    /// pinned end-to-end in main.rs
    /// `same_effort_value_resolves_identically_on_every_surface`.
    #[test]
    fn output_config_effort_flows_to_the_one_reasoning_surface() {
        let req = |extra: Value| {
            let mut body = json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": "x"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            body
        };
        // Every string — valid, clamp-alias, or garbage — must reach parse_think's table.
        for value in [
            "none", "minimal", "low", "medium", "high", "xhigh", "banana", "",
        ] {
            let out = translate(&req(json!({"output_config": {"effort": value}}))).unwrap();
            assert_eq!(
                out["reasoning"]["effort"], value,
                "output_config.effort {value:?} must translate to reasoning.effort — \
                 dropping it re-opens issue #31 (accepted-then-ignored on /v1/messages)"
            );
            assert!(
                out["reasoning"].get("enabled").is_none(),
                "effort alone must not invent an explicit switch"
            );
        }
        // PRECEDENCE: thinking.type is the documented Anthropic lever — it maps to the
        // explicit switch (which parse_think gives precedence) and the effort rides along.
        let both = translate(&req(json!({
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"}
        })))
        .unwrap();
        assert_eq!(both["reasoning"]["enabled"], false);
        assert_eq!(both["reasoning"]["effort"], "high");
        let both = translate(&req(json!({
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "none"}
        })))
        .unwrap();
        assert_eq!(both["reasoning"]["enabled"], true);
        assert_eq!(both["reasoning"]["effort"], "none");
        // adaptive keeps the model's own default switch; effort still flows.
        let adaptive = translate(&req(json!({
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"}
        })))
        .unwrap();
        assert!(adaptive["reasoning"].get("enabled").is_none());
        assert_eq!(adaptive["reasoning"]["effort"], "xhigh");
        // No reasoning expression at all -> no reasoning override (model default).
        let unset = translate(&req(json!({}))).unwrap();
        assert!(unset.get("reasoning").is_none());
        // output_config fields OTHER than effort used to be accepted-and-ignored; they are now
        // named 400s (lane/reasoning-schema-20260823) — an output_config this server cannot act
        // on must not answer 200 as though it had.
        let err = translate(&req(json!({"output_config": {"something_else": 1}}))).unwrap_err();
        assert!(err.contains("output_config.something_else"), "got: {err}");
        // A non-string effort is a type error named at the field, not a silent drop.
        let err = translate(&req(json!({"output_config": {"effort": 3}}))).unwrap_err();
        assert!(err.contains("output_config.effort"), "got: {err}");
    }

    #[test]
    fn message_json_renders_blocks_stop_reason_and_honest_usage() {
        let env = test_envelope("msg_test1");
        // Tool calls win the stop_reason (the client must execute them).
        let fin = surfaces::FinalChat {
            text: "I'll check.".into(),
            reasoning: "let me think".into(),
            calls: vec![crate::toolcall::ParsedToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
            }],
            stop_reason: "Eos".into(),
            matched_stop: None,
            n_tokens: 9,
            n_prompt: 20,
            n_cached: 5,
            elapsed_s: 0.2,
            spec: None,
        };
        let v = message_json(&env, "m", &fin);
        assert_eq!(v["id"], "msg_test1");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"][0]["type"], "thinking");
        assert_eq!(v["content"][0]["thinking"], "let me think");
        assert_eq!(v["content"][1]["type"], "text");
        assert_eq!(v["content"][1]["text"], "I'll check.");
        assert_eq!(v["content"][2]["type"], "tool_use");
        assert_eq!(v["content"][2]["id"], "call_1");
        assert_eq!(v["content"][2]["input"]["city"], "Paris");
        assert_eq!(v["stop_reason"], "tool_use");
        // Anthropic input_tokens EXCLUDE cache reads; the cache field carries them.
        assert_eq!(v["usage"]["input_tokens"], 15);
        assert_eq!(v["usage"]["cache_read_input_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens"], 9);
        // Stop-sequence and budget mappings.
        assert_eq!(stop_reason("Eos", false, true), "stop_sequence");
        assert_eq!(stop_reason("MaxNew", false, false), "max_tokens");
        assert_eq!(stop_reason("Eos", false, false), "end_turn");
    }

    #[test]
    fn thinking_text_is_returned_when_reasoning_is_on_and_absent_when_off() {
        // OWNER ACCEPTANCE GATE (2026-08-23, "also thinking content should be returned, not only
        // the content itself"): on this surface reasoning is delivered as a `thinking` content
        // block — never stripped server-side — and a reasoning-off generation produces NO
        // thinking block rather than an empty one. Billing unchanged either way: reasoning
        // tokens are output tokens.
        let fin = |reasoning: &str| surfaces::FinalChat {
            text: "answer".into(),
            reasoning: reasoning.into(),
            calls: Vec::new(),
            stop_reason: "Eos".into(),
            matched_stop: None,
            n_tokens: 4,
            n_prompt: 10,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        };
        let on = message_json(&test_envelope("msg_on"), "m", &fin("a plan"));
        assert_eq!(on["content"][0]["type"], "thinking");
        assert_eq!(on["content"][0]["thinking"], "a plan");
        assert_eq!(on["content"][1]["type"], "text");
        let off = message_json(&test_envelope("msg_off"), "m", &fin(""));
        assert_eq!(
            off["content"][0]["type"], "text",
            "no thinking block when reasoning is off"
        );
        assert_eq!(off["content"].as_array().unwrap().len(), 1);
    }

    /// GOLDEN TRANSCRIPT (text): the exact Anthropic event grammar over a plain stream —
    /// message_start (admission-truth input tokens) -> ping -> one text block -> a
    /// cumulative-usage message_delta -> message_stop.
    #[tokio::test]
    async fn sse_text_stream_speaks_the_anthropic_grammar() {
        let (tx, rx) = crate::worker::event_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 10,
            n_cached: 4,
        })
        .unwrap();
        tx.send(Event::Token {
            id: 1,
            text: "Hel".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "lo".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 2,
            n_prompt: 10,
            n_cached: 4,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let resp = messages_sse(
            rx,
            None,
            test_envelope("msg_g1"),
            "m".into(),
            None,
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let names: Vec<&str> = frames.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        let start = &frames[0].1;
        assert_eq!(start["message"]["id"], "msg_g1");
        assert_eq!(start["message"]["usage"]["input_tokens"], 6);
        assert_eq!(start["message"]["usage"]["cache_read_input_tokens"], 4);
        assert_eq!(frames[2].1["content_block"]["type"], "text");
        assert_eq!(frames[3].1["delta"]["text"], "Hel");
        assert_eq!(frames[4].1["delta"]["text"], "lo");
        let delta = &frames[6].1;
        assert_eq!(delta["delta"]["stop_reason"], "end_turn");
        assert_eq!(delta["usage"]["output_tokens"], 2);
        // every data payload carries its own type (clients dispatch on either).
        for (name, data) in &frames {
            assert_eq!(data["type"], json!(name));
        }
    }

    #[tokio::test]
    async fn committed_prefill_deadline_uses_the_anthropic_timeout_grammar() {
        let (tx, rx) = crate::worker::event_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 218_000,
            n_cached: 0,
        })
        .unwrap();
        tx.send(Event::DeadlineExceeded { ms: 180_000 }).unwrap();
        drop(tx);
        let resp = messages_sse(
            rx,
            None,
            test_envelope("msg_deadline"),
            "m".into(),
            None,
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let failed = frames.last().expect("deadline terminal frame");
        assert_eq!(failed.0, "error");
        assert_eq!(failed.1["error"]["type"], "timeout");
        assert!(
            failed.1["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not billed")
        );
    }

    /// GOLDEN TRANSCRIPT (tool round-trip): a template-law tool emission becomes a
    /// tool_use block (input via input_json_delta) and the final stop_reason is
    /// "tool_use" — the exact contract an agentic client's tool loop hangs on.
    #[tokio::test]
    async fn sse_tool_call_stream_produces_tool_use_blocks_and_stop_reason() {
        let (tx, rx) = crate::worker::event_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 5,
            n_cached: 0,
        })
        .unwrap();
        tx.send(Event::Token {
            id: 1,
            text: "On it. ".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
                   </parameter>\n</function>\n</tool_call>"
                .into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 2,
            n_prompt: 5,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let mut schemas: HashMap<String, HashMap<String, String>> = HashMap::new();
        schemas.insert(
            "get_weather".into(),
            [("city".to_string(), "string".to_string())].into(),
        );
        let parser = ToolStreamParser::new(schemas, false);
        let resp = messages_sse(
            rx,
            None,
            test_envelope("msg_g2"),
            "m".into(),
            Some(parser),
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let names: Vec<&str> = frames.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "ping",
                "content_block_start", // text
                "content_block_delta", // "On it. "
                "content_block_stop",  // text closes when the call opens
                "content_block_start", // tool_use
                "content_block_delta", // input_json_delta
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        let call_start = &frames[5].1;
        assert_eq!(call_start["content_block"]["type"], "tool_use");
        assert_eq!(call_start["content_block"]["name"], "get_weather");
        assert_eq!(call_start["content_block"]["input"], json!({}));
        assert!(
            call_start["content_block"]["id"]
                .as_str()
                .unwrap()
                .starts_with("call_")
        );
        let args = &frames[6].1["delta"];
        assert_eq!(args["type"], "input_json_delta");
        let parsed: Value = serde_json::from_str(args["partial_json"].as_str().unwrap()).unwrap();
        assert_eq!(parsed, json!({"city": "Paris"}));
        assert_eq!(frames[8].1["delta"]["stop_reason"], "tool_use");
    }

    /// Mid-stream faults surface as the Anthropic `error` event, typed by class.
    #[tokio::test]
    async fn sse_midstream_fault_emits_the_anthropic_error_event() {
        let (tx, rx) = crate::worker::event_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 3,
            n_cached: 0,
        })
        .unwrap();
        tx.send(Event::Error(crate::worker::EngineError::overloaded(
            "vram exhausted",
        )))
        .unwrap();
        drop(tx);
        let resp = messages_sse(
            rx,
            None,
            test_envelope("msg_g3"),
            "m".into(),
            None,
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let (name, data) = frames.last().unwrap();
        assert_eq!(name, "error");
        assert_eq!(data["error"]["type"], "overloaded_error");
        assert_eq!(data["error"]["message"], "vram exhausted");
    }
}
