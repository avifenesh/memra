//! `/v1/responses` — the OpenAI Responses API served as a TRANSLATION SURFACE over the
//! chat-completions core (lane/api-surfaces, 2026-08-17).
//!
//! Why it exists: agentic clients whose provider config only speaks `wire_api =
//! "responses"` (Codex CLI ≥0.147 removed the chat wire entirely) can point straight at
//! this server. The surface is STATELESS translation only: `input` items are rewritten
//! into the internal `ChatCompletionReq`, admission/billing/capture flow through
//! `surfaces::admit_translated` (the same sequence `chat_completions` runs), and the
//! response is rendered in the Responses vocabulary (`response.created` ..
//! `response.completed`).
//!
//! SUBSET LAW (honesty gate): stateful features this server cannot honor refuse with a
//! clear 400 — `previous_response_id`, `store: true`, `conversation`, `background`,
//! `item_reference` input items, `truncation: "auto"`. A stateless client that resends
//! full context each turn (`store: false`, the Codex custom-provider posture) is fully
//! supported. Accepted-and-ignored (non-semantic here): `include`, `parallel_tool_calls`,
//! `reasoning.summary`, `stream_options`, `client_metadata`, `metadata`, `service_tier`,
//! `text.verbosity`. Non-function TOOL types (`web_search`, `namespace`, `custom`) are
//! dropped from the toolset with a log line — stock clients send them unconditionally,
//! so refusing would refuse every default-config request; a dropped tool is one the
//! model never sees, not one that half-works.
//!
//! Wire facts this rendering is built against (verified from the Codex 0.147.0 source):
//! clients dispatch on the `type` INSIDE the SSE data JSON (the `event:` line and
//! `sequence_number` are decorative); `response.output_item.added` must precede any
//! delta for that item; the FULL item on `response.output_item.done` is the
//! authoritative content; `response.completed` must carry `response.id`; reasoning items
//! must carry both `summary` and `encrypted_content` keys; `function_call.arguments` is
//! a JSON STRING. Errors before the stream commits use the standard OpenAI error body.

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::surfaces::{self, CollectError, SurfaceScrubber};
use crate::toolcall::Piece;
use crate::worker::Event;
use crate::{AppState, ChatCompletionReq, Envelope, Extension, TtftRequestTrace};

// ---- request translation ---------------------------------------------------------------

/// Flatten a `function_call_output.output` value: a plain string, or an array of
/// `input_text` content items (the multimodal form — text parts only are supported).
fn call_output_text(v: Option<&Value>) -> Result<String, String> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("input_text") | Some("output_text") | Some("text") => out.push_str(
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .ok_or("output content item has no text field")?,
                    ),
                    other => {
                        return Err(format!(
                            "function_call_output content item type {other:?} is not \
                             supported (text only)"
                        ));
                    }
                }
            }
            Ok(out)
        }
        Some(other) => Err(format!(
            "function_call_output.output must be a string or content array, got {other}"
        )),
    }
}

/// Message content: a string, or an array of typed parts.
fn message_content(v: Option<&Value>, at: &str) -> Result<Value, String> {
    match v {
        None | Some(Value::Null) => Ok(json!("")),
        Some(Value::String(s)) => Ok(json!(s)),
        Some(Value::Array(parts)) => {
            let mut out: Vec<Value> = Vec::new();
            for (j, p) in parts.iter().enumerate() {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("input_text") | Some("output_text") | Some("text")
                    | Some("summary_text") => out.push(json!({
                        "type": "text",
                        "text": p.get("text").and_then(|t| t.as_str())
                            .ok_or_else(|| format!("{at}.content[{j}]: part has no text"))?,
                    })),
                    Some("input_image") => out.push(json!({
                        "type": "image_url",
                        "image_url": { "url": p.get("image_url").and_then(|u| u.as_str())
                            .ok_or_else(|| format!("{at}.content[{j}]: input_image has no image_url"))? },
                    })),
                    Some("refusal") => {}
                    other => {
                        return Err(format!(
                            "{at}.content[{j}]: part type {other:?} is not supported"
                        ));
                    }
                }
            }
            Ok(Value::Array(out))
        }
        Some(other) => Err(format!(
            "{at}.content must be a string or array, got {other}"
        )),
    }
}

/// One clear refusal line for a stateful/unsupported field.
fn refuse(field: &str, why: &str) -> String {
    format!("{field} is not supported on this stateless server: {why}")
}

/// Translate one Responses API request into the internal OpenAI-chat request value.
pub(crate) fn translate(v: &Value) -> Result<Value, (String, Option<String>)> {
    let obj = v
        .as_object()
        .ok_or(("request body must be a JSON object".to_string(), None))?;
    let model = obj.get("model").and_then(|m| m.as_str()).ok_or((
        "model: field required".to_string(),
        Some("model".to_string()),
    ))?;

    // HONESTY GATES: stateful features refuse loudly, never silently misbehave.
    let gate =
        |field: &'static str, why: &'static str| (refuse(field, why), Some(field.to_string()));
    if obj
        .get("previous_response_id")
        .is_some_and(|x| !x.is_null())
    {
        return Err(gate(
            "previous_response_id",
            "responses are not stored; resend the full conversation in `input` each turn \
             (store:false semantics)",
        ));
    }
    if obj.get("store").and_then(|s| s.as_bool()) == Some(true) {
        return Err(gate(
            "store",
            "responses are not persisted server-side; set store:false and resend full \
             context each turn",
        ));
    }
    if obj.get("conversation").is_some_and(|x| !x.is_null()) {
        return Err(gate(
            "conversation",
            "server-side conversation state does not exist here",
        ));
    }
    if obj.get("background").and_then(|b| b.as_bool()) == Some(true) {
        return Err(gate("background", "background responses are not supported"));
    }
    if obj.get("truncation").and_then(|t| t.as_str()) == Some("auto") {
        return Err(gate(
            "truncation",
            "server-side context truncation is not performed; only \"disabled\" is honest",
        ));
    }
    if obj.get("prompt").is_some_and(|x| !x.is_null()) {
        return Err(gate("prompt", "stored prompt templates do not exist here"));
    }

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = obj.get("instructions").and_then(|i| i.as_str())
        && !instructions.is_empty()
    {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    // Consecutive function_call items merge into ONE assistant turn (what an OpenAI chat
    // client would have sent as a single tool_calls array).
    let mut pending_calls: Vec<Value> = Vec::new();
    let flush_calls = |messages: &mut Vec<Value>, pending: &mut Vec<Value>| {
        if !pending.is_empty() {
            messages.push(json!({
                "role": "assistant",
                "content": "",
                "tool_calls": std::mem::take(pending),
            }));
        }
    };
    match obj.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                let at = format!("input[{i}]");
                let itype = item
                    .get("type")
                    .and_then(|t| t.as_str())
                    .or_else(|| item.get("role").map(|_| "message"))
                    .ok_or_else(|| (format!("{at}: item has no type"), None))?;
                match itype {
                    "message" => {
                        flush_calls(&mut messages, &mut pending_calls);
                        let role = item
                            .get("role")
                            .and_then(|r| r.as_str())
                            .ok_or_else(|| (format!("{at}: message has no role"), None))?;
                        if !matches!(role, "user" | "assistant" | "system" | "developer") {
                            return Err((format!("{at}: role {role:?} is not supported"), None));
                        }
                        let content =
                            message_content(item.get("content"), &at).map_err(|e| (e, None))?;
                        messages.push(json!({ "role": role, "content": content }));
                    }
                    "function_call" => {
                        let name = item
                            .get("name")
                            .and_then(|n| n.as_str())
                            .ok_or_else(|| (format!("{at}: function_call has no name"), None))?;
                        let arguments =
                            item.get("arguments")
                                .and_then(|a| a.as_str())
                                .ok_or_else(|| {
                                    (
                                        format!(
                                            "{at}: function_call.arguments must be a JSON string"
                                        ),
                                        None,
                                    )
                                })?;
                        pending_calls.push(json!({
                            "id": item.get("call_id").and_then(|c| c.as_str()).unwrap_or(""),
                            "function": { "name": name, "arguments": arguments },
                        }));
                    }
                    "function_call_output" => {
                        flush_calls(&mut messages, &mut pending_calls);
                        let text = call_output_text(item.get("output"))
                            .map_err(|e| (format!("{at}: {e}"), None))?;
                        messages.push(json!({
                            "role": "tool",
                            "content": text,
                            "tool_call_id": item.get("call_id").and_then(|c| c.as_str())
                                .unwrap_or(""),
                        }));
                    }
                    // Reasoning items are this server's own PRIOR output echoed back by a
                    // stateless client; templates re-render history without think
                    // segments, so they are consumed here by design (docs/API-SURFACES.md).
                    "reasoning" => {}
                    "item_reference" => {
                        return Err((
                            refuse(
                                "item_reference",
                                "responses are not stored, so items \
                                    cannot be referenced by id; inline the item",
                            ),
                            None,
                        ));
                    }
                    other => {
                        return Err((format!("{at}: item type {other:?} is not supported"), None));
                    }
                }
            }
            flush_calls(&mut messages, &mut pending_calls);
        }
        None | Some(Value::Null) => {
            return Err((
                "input: field required".to_string(),
                Some("input".to_string()),
            ));
        }
        Some(other) => {
            return Err((
                format!("input must be a string or array of items, got {other}"),
                Some("input".to_string()),
            ));
        }
    }

    let mut tools: Vec<Value> = Vec::new();
    let mut dropped_tools: Vec<String> = Vec::new();
    if let Some(ts) = obj.get("tools").and_then(|t| t.as_array()) {
        for (i, t) in ts.iter().enumerate() {
            match t.get("type").and_then(|x| x.as_str()) {
                Some("function") => {}
                // Non-function tool types (server-executed `web_search`, agent-side
                // `namespace` groups, freeform `custom` grammars) cannot run here — and
                // stock Responses clients send some of them UNCONDITIONALLY, so a 400
                // would refuse every default-config request. They are DROPPED from the
                // toolset instead: the model never sees them in its tools block, so it
                // cannot call them — a narrower capability set, never a broken call.
                // Logged per request and documented in docs/API-SURFACES.md.
                other => {
                    dropped_tools.push(format!(
                        "{}:{}",
                        other.unwrap_or("?"),
                        t.get("name").and_then(|n| n.as_str()).unwrap_or("-"),
                    ));
                    continue;
                }
            }
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").and_then(|n| n.as_str())
                        .ok_or_else(|| (format!("tools[{i}].name: field required"), None))?,
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("parameters").cloned().unwrap_or_else(|| json!({})),
                },
            }));
        }
    }
    if !dropped_tools.is_empty() {
        eprintln!(
            "[responses] dropped non-function tools (model will not see them): {}",
            dropped_tools.join(", ")
        );
    }

    let tool_choice = match obj.get("tool_choice") {
        None | Some(Value::Null) => Value::Null,
        Some(Value::String(s)) if s == "auto" || s == "none" => json!(s),
        Some(other) => {
            return Err((
                format!(
                    "tool_choice {other} is not supported (forcing a tool call needs \
                     constrained decoding); use \"auto\" or \"none\""
                ),
                Some("tool_choice".to_string()),
            ));
        }
    };

    // reasoning.effort maps onto the ONE reasoning surface via the ONE allowlist. Membership
    // is validated here so the error names the Responses-shaped param, but the RAW value flows
    // through: canonicalization is model-capability-aware (dsv4 has a "max" rung above
    // "high") and therefore happens at plan build in parse_think.
    // STRICT `reasoning` object, matching the chat surface key-for-key (one schema means one
    // answer to the same malformed request on every surface). Before this lane a non-object
    // `reasoning` was silently ignored here while the identical body 400'd on chat, and every
    // unknown key — including `summary`, which this server does not implement — was dropped
    // behind a 200.
    let reasoning_obj = match obj.get("reasoning") {
        None | Some(Value::Null) => None,
        Some(Value::Object(map)) => Some(map),
        Some(_) => {
            return Err((
                "reasoning must be an object".to_string(),
                Some("reasoning".to_string()),
            ));
        }
    };
    if let Some(map) = reasoning_obj {
        for key in map.keys() {
            match key.as_str() {
                "effort" => {}
                // `summary` selects how reasoning is SUMMARISED. This server does not summarise:
                // it delivers the model's reasoning text verbatim, always (owner ruling —
                // reasoning is billed output, so it is never withheld or abridged). Accepting a
                // summary mode we do not implement would be accept-and-ignore; "auto" is the
                // one value our behaviour honestly satisfies.
                // `generate_summary` is OpenAI's DEPRECATED alias of `summary`; same semantics,
                // so same rule. Named explicitly rather than left to the unknown-key arm, because
                // refusing a legacy spelling of a field we do accept would 400 a stock client
                // over vocabulary rather than over anything we cannot do.
                "summary" | "generate_summary" => {
                    // Matched on the RAW value: `as_str()` turned a wrong-typed `{"summary": 3}`
                    // into `None` and read it as unset, so it returned 200 and changed nothing —
                    // while `reasoning.effort: 3` was already a named 400 thirty lines below.
                    match map.get(key.as_str()) {
                        None | Some(Value::Null) => {}
                        Some(Value::String(mode)) if mode == "auto" => {}
                        Some(other) => {
                            return Err((
                                format!(
                                    "reasoning.{key} {other} is not supported by this server: it \
                                     does not summarise reasoning, it delivers the model's \
                                     reasoning text verbatim in the reasoning item. Only \"auto\" \
                                     describes that"
                                ),
                                Some("reasoning".to_string()),
                            ));
                        }
                    }
                }
                other => {
                    return Err((
                        format!(
                            "reasoning.{other} is not a field this server implements (it would \
                             change nothing about the request); the supported keys are effort \
                             and summary"
                        ),
                        Some("reasoning".to_string()),
                    ));
                }
            }
        }
    }
    let reasoning_effort = match reasoning_obj.and_then(|r| r.get("effort")) {
        None | Some(Value::Null) => Value::Null,
        Some(Value::String(raw)) => match crate::canonical_effort(raw) {
            Some(_) => json!(raw),
            None => {
                return Err((
                    format!("reasoning.effort {raw:?} is not supported"),
                    Some("reasoning".to_string()),
                ));
            }
        },
        Some(other) => {
            return Err((
                format!("reasoning.effort must be a string, got {other}"),
                Some("reasoning".to_string()),
            ));
        }
    };

    // text.format -> the chat surface's response_format (same constrained decoder).
    let response_format = match obj.get("text").and_then(|t| t.get("format")) {
        None | Some(Value::Null) => Value::Null,
        Some(f) => match f.get("type").and_then(|t| t.as_str()) {
            Some("text") => Value::Null,
            Some("json_object") => json!({ "type": "json_object" }),
            Some("json_schema") => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": f.get("name").cloned().unwrap_or(json!("response")),
                    "strict": f.get("strict").cloned().unwrap_or(Value::Null),
                    "schema": f.get("schema").cloned().unwrap_or_else(|| json!({})),
                },
            }),
            other => {
                return Err((
                    format!("text.format type {other:?} is not supported"),
                    Some("text".to_string()),
                ));
            }
        },
    };

    let mut out = json!({
        "model": model,
        "messages": messages,
        "stream": obj.get("stream").and_then(|s| s.as_bool()).unwrap_or(false),
    });
    if !tools.is_empty() {
        out["tools"] = Value::Array(tools);
    }
    if !tool_choice.is_null() {
        out["tool_choice"] = tool_choice;
    }
    if !reasoning_effort.is_null() {
        out["reasoning_effort"] = reasoning_effort;
    }
    if !response_format.is_null() {
        out["response_format"] = response_format;
    }
    if let Some(max) = obj.get("max_output_tokens").filter(|m| !m.is_null()) {
        out["max_tokens"] = max.clone();
    }
    // timeout_ms (lane/deadline-billing) passes through UNVALIDATED — the one
    // `parse_timeout_ms` body in the shared admission refuses bad values with the same
    // named 400 on every surface.
    for key in ["temperature", "top_p", "user", "timeout_ms"] {
        if let Some(v) = obj.get(key).filter(|v| !v.is_null()) {
            out[key] = v.clone();
        }
    }
    // prompt_cache_key is the client's stable per-conversation identity — the same
    // session-affinity nomination the chat surface reads from session_id. It does NOT
    // become a cache_salt: prefix-cache isolation stays tenant-scoped, so the shared
    // instructions prefix keeps its cross-session cache hits.
    if let Some(key) = obj.get("prompt_cache_key").and_then(|k| k.as_str()) {
        out["session_id"] = json!(key);
    }
    Ok(out)
}

// ---- response rendering ----------------------------------------------------------------

fn usage_json(n_prompt: usize, n_tokens: usize, n_cached: usize) -> Value {
    json!({
        "input_tokens": n_prompt,
        "input_tokens_details": { "cached_tokens": n_cached },
        "output_tokens": n_tokens,
        "total_tokens": n_prompt + n_tokens,
    })
}

/// The response envelope: `status` is "in_progress" | "completed" | "incomplete" |
/// "failed"; `output`/`usage`/`error`/`incomplete_details` are filled per state.
#[allow(clippy::too_many_arguments)]
fn response_json(
    env: &Envelope,
    model: &str,
    status: &str,
    output: &[Value],
    usage: Option<Value>,
    error: Value,
    incomplete: Value,
) -> Value {
    json!({
        "id": env.id,
        "object": "response",
        "created_at": env.created,
        "status": status,
        "model": model,
        "output": output,
        "error": error,
        "incomplete_details": incomplete,
        "parallel_tool_calls": false,
        "previous_response_id": null,
        "store": false,
        "tool_choice": "auto",
        "usage": usage,
    })
}

fn text_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    })
}

fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [{ "type": "summary_text", "text": text }],
        "encrypted_content": null,
    })
}

fn call_item(id: &str, call: &crate::toolcall::ParsedToolCall) -> Value {
    json!({
        "id": id,
        "type": "function_call",
        "status": "completed",
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    })
}

/// Incomplete-by-token-budget is a first-class terminal state on this API (clients treat
/// it as "the model was cut off", distinct from completed).
fn incomplete_reason(worker_reason: &str) -> Option<&'static str> {
    match worker_reason {
        "MaxNew" | "ContextFull" => Some("max_output_tokens"),
        _ => None,
    }
}

/// One SSE frame: the Responses dialect repeats the type inside the data JSON; clients
/// dispatch on that field, but the `event:` line is set too (the official API sends both).
fn frame(mut data: Value, seq: &mut u64) -> SseEvent {
    let name = data
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("response.completed")
        .to_string();
    *seq += 1;
    data["sequence_number"] = json!(*seq);
    SseEvent::default().event(name).data(data.to_string())
}

// ---- handler ---------------------------------------------------------------------------

/// POST /v1/responses. Auth is the standard `Authorization: Bearer` against the same
/// tenant keyring as every other surface; errors use the OpenAI error body throughout.
pub(crate) async fn responses(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    body: Bytes,
) -> Response {
    let env = Envelope {
        id: format!("resp_{}", crate::gen_hex128()),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            return crate::with_request_id(
                &env.id,
                crate::bad_request(&format!("invalid JSON: {err}"), None),
            );
        }
    };
    let translated = match translate(&parsed) {
        Ok(v) => v,
        Err((msg, param)) => {
            return crate::with_request_id(&env.id, crate::bad_request(&msg, param.as_deref()));
        }
    };
    let mut req: ChatCompletionReq = match serde_json::from_value(translated) {
        Ok(r) => r,
        Err(err) => {
            return crate::with_request_id(
                &env.id,
                crate::bad_request(&format!("invalid request: {err}"), None),
            );
        }
    };
    match crate::canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return crate::with_request_id(
                &env.id,
                crate::model_not_found_response(&st.models, &req.model),
            );
        }
    }
    let ttft = trace.and_then(|Extension(trace)| trace.0);
    if let Some(trace) = ttft.as_ref() {
        trace.mark_parsed();
        trace.bind_request(&env.id, &req.model);
    }
    let tenant =
        match surfaces::authenticate_candidates(&st.api_auth, &[crate::bearer_token(&headers)]) {
            Ok(tenant) => tenant,
            Err(why) => {
                return crate::with_request_id(&env.id, crate::authentication_error(why));
            }
        };
    let model = req.model.clone();
    let stream = req.stream;
    let admission =
        match surfaces::admit_translated(&st, &headers, &env, &tenant, req, "/v1/responses", ttft)
            .await
        {
            Ok(a) => a,
            Err(resp) => return crate::with_request_id(&env.id, resp),
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
        // Streaming: timeout_ms bounds TIME-TO-FIRST-TOKEN only (held pre-header so a
        // miss is an honest 408) — same law as the chat surface.
        let rx = match crate::peek_first_token(rx, deadline).await {
            Ok(rx) => rx,
            Err(()) => {
                drop(guard);
                return rl.attach(crate::ledger_unbilled(
                    receipt,
                    crate::deadline_exceeded_response(deadline.ms, true),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                ));
            }
        };
        let resp = responses_sse(
            rx,
            receipt,
            env.clone(),
            model,
            parser,
            stop_strings,
            Some(guard),
        )
        .into_response();
        return rl.attach(crate::with_request_id(&env.id, resp));
    }
    // Non-streaming: the deadline covers the COMPLETE response. On a miss the collect
    // future is dropped and rx with it (cancelling generation at the worker's next tick);
    // the receipt settles deadline_exceeded, debit ZERO.
    // Deadline posture, stated so it is a decision and not an omission
    // (lane/deadline-partial-20260826): this surface keeps the 408 while the OpenAI
    // chat/completions dialect now delivers the partial. The Responses shape COULD express
    // one (`status: "incomplete"` + `incomplete_details.reason`), but OpenAI defines that
    // reason as max_output_tokens | content_filter only, so a deadline value would be an
    // extension callers do not parse — and the feasibility gate in `surfaces::admit_translated` now
    // refuses the impossible request up front on this surface too, which is where nearly all
    // of these went. Revisit if a caller asks for partials here specifically.
    let collected = tokio::time::timeout_at(
        deadline.at,
        surfaces::collect_final(&mut rx, &mut receipt, parser, &stop_strings, &env),
    )
    .await;
    let fin = match collected {
        Ok(Ok(fin)) => fin,
        Ok(Err(CollectError::Ledger)) => {
            drop(guard);
            return rl.attach(crate::with_request_id(
                &env.id,
                crate::request_ledger_error_response(),
            ));
        }
        Ok(Err(CollectError::Engine(e))) => {
            drop(guard);
            return rl.attach(crate::with_request_id(
                &env.id,
                crate::engine_error_response(&e),
            ));
        }
        Err(_) => {
            drop(rx); // the cancel signal: the worker prunes the closed channel
            drop(guard);
            return rl.attach(crate::ledger_unbilled(
                receipt,
                crate::deadline_exceeded_response(deadline.ms, false),
                "deadline_exceeded",
                "deadline_exceeded",
                &env.id,
            ));
        }
    };
    let mut output: Vec<Value> = Vec::new();
    if !fin.reasoning.is_empty() {
        output.push(reasoning_item(
            &format!("rs_{}", crate::gen_hex128()),
            &fin.reasoning,
        ));
    }
    if !fin.text.is_empty() {
        output.push(text_item(
            &format!("msg_{}", crate::gen_hex128()),
            &fin.text,
        ));
    }
    for call in &fin.calls {
        output.push(call_item(&format!("fc_{}", crate::gen_hex128()), call));
    }
    let (status, incomplete) = match incomplete_reason(&fin.stop_reason) {
        Some(reason) => ("incomplete", json!({ "reason": reason })),
        None => ("completed", Value::Null),
    };
    let body = response_json(
        &env,
        &model,
        status,
        &output,
        Some(usage_json(fin.n_prompt, fin.n_tokens, fin.n_cached)),
        Value::Null,
        incomplete,
    );
    let resp = axum::Json(body).into_response();
    drop(guard);
    rl.attach(crate::with_request_id(&env.id, resp))
}

/// The Responses streaming vocabulary over the worker's event stream, with the SAME
/// receipt discipline as the chat SSE path. Grammar honored (client-verified):
/// `response.created` first; every item opens with `response.output_item.added` BEFORE
/// its deltas and closes with a full-item `response.output_item.done`; the stream ends
/// with exactly one of `response.completed` (id + usage), `response.incomplete`
/// (token-budget cutoff) or `response.failed` (fault), then the connection closes.
// unused_assignments: the item state machine (open/open_buf/output_index) is written by
// a macro at every transition; the compiler flags the writes of the FINAL transition as
// dead. Real state, false positive.
#[allow(clippy::too_many_arguments, unused_assignments)]
fn responses_sse(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    mut receipt: Option<crate::ledger::PendingReceipt>,
    env: Envelope,
    model: String,
    mut parser: Option<crate::toolcall::ToolStreamParser>,
    stop_strings: Vec<String>,
    guard: Option<crate::InflightGuard>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    let mut scrub = (!stop_strings.is_empty()).then(|| SurfaceScrubber::new(stop_strings.clone()));
    let stream = async_stream::stream! {
        let _guard = guard;
        #[derive(PartialEq, Clone, Copy)]
        enum Open { None, Text, Reasoning }
        let mut seq: u64 = 0;
        let mut open = Open::None;
        let mut open_id = String::new();
        let mut open_buf = String::new(); // full text of the open item (done frames carry it)
        let mut output_index: usize = 0;
        let mut output: Vec<Value> = Vec::new(); // completed items, for response.completed
        let mut prompt_usage: (usize, usize) = (0, 0);
        yield Ok(frame(json!({
            "type": "response.created",
            "response": response_json(&env, &model, "in_progress", &[], None,
                                      Value::Null, Value::Null),
        }), &mut seq));
        macro_rules! close_item {
            () => {
                match open {
                    Open::None => {}
                    Open::Text => {
                        yield Ok(frame(json!({
                            "type": "response.output_text.done",
                            "item_id": open_id, "output_index": output_index,
                            "content_index": 0, "text": open_buf,
                        }), &mut seq));
                        yield Ok(frame(json!({
                            "type": "response.content_part.done",
                            "item_id": open_id, "output_index": output_index,
                            "content_index": 0,
                            "part": { "type": "output_text", "text": open_buf,
                                      "annotations": [] },
                        }), &mut seq));
                        let item = text_item(&open_id, &open_buf);
                        yield Ok(frame(json!({
                            "type": "response.output_item.done",
                            "output_index": output_index, "item": item,
                        }), &mut seq));
                        output.push(item);
                        output_index += 1;
                        open = Open::None;
                        open_buf = String::new();
                    }
                    Open::Reasoning => {
                        yield Ok(frame(json!({
                            "type": "response.reasoning_summary_text.done",
                            "item_id": open_id, "output_index": output_index,
                            "summary_index": 0, "text": open_buf,
                        }), &mut seq));
                        let item = reasoning_item(&open_id, &open_buf);
                        yield Ok(frame(json!({
                            "type": "response.output_item.done",
                            "output_index": output_index, "item": item,
                        }), &mut seq));
                        output.push(item);
                        output_index += 1;
                        open = Open::None;
                        open_buf = String::new();
                    }
                }
            };
        }
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
                                close_item!();
                                open = Open::Text;
                                open_id = format!("msg_{}", crate::gen_hex128());
                                yield Ok(frame(json!({
                                    "type": "response.output_item.added",
                                    "output_index": output_index,
                                    "item": { "id": open_id, "type": "message",
                                              "role": "assistant",
                                              "status": "in_progress", "content": [] },
                                }), &mut seq));
                                yield Ok(frame(json!({
                                    "type": "response.content_part.added",
                                    "item_id": open_id, "output_index": output_index,
                                    "content_index": 0,
                                    "part": { "type": "output_text", "text": "",
                                              "annotations": [] },
                                }), &mut seq));
                            }
                            open_buf.push_str(&text);
                            yield Ok(frame(json!({
                                "type": "response.output_text.delta",
                                "item_id": open_id, "output_index": output_index,
                                "content_index": 0, "delta": text,
                            }), &mut seq));
                        }
                    }
                    Piece::Reasoning(text) => {
                        if open != Open::Reasoning {
                            close_item!();
                            open = Open::Reasoning;
                            open_id = format!("rs_{}", crate::gen_hex128());
                            yield Ok(frame(json!({
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": { "id": open_id, "type": "reasoning",
                                          "summary": [], "encrypted_content": null },
                            }), &mut seq));
                            yield Ok(frame(json!({
                                "type": "response.reasoning_summary_part.added",
                                "item_id": open_id, "output_index": output_index,
                                "summary_index": 0,
                                "part": { "type": "summary_text", "text": "" },
                            }), &mut seq));
                        }
                        open_buf.push_str(&text);
                        yield Ok(frame(json!({
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": open_id, "output_index": output_index,
                            "summary_index": 0, "delta": text,
                        }), &mut seq));
                    }
                    Piece::Call(call) => {
                        close_item!();
                        let id = format!("fc_{}", crate::gen_hex128());
                        yield Ok(frame(json!({
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": { "id": id, "type": "function_call",
                                      "status": "in_progress", "call_id": call.id,
                                      "name": call.name, "arguments": "" },
                        }), &mut seq));
                        yield Ok(frame(json!({
                            "type": "response.function_call_arguments.delta",
                            "item_id": id, "output_index": output_index,
                            "delta": call.arguments,
                        }), &mut seq));
                        yield Ok(frame(json!({
                            "type": "response.function_call_arguments.done",
                            "item_id": id, "output_index": output_index,
                            "arguments": call.arguments,
                        }), &mut seq));
                        let item = call_item(&id, &call);
                        yield Ok(frame(json!({
                            "type": "response.output_item.done",
                            "output_index": output_index, "item": item,
                        }), &mut seq));
                        output.push(item);
                        output_index += 1;
                    }
                }
            }};
        }
        macro_rules! stream_fault {
            ($code:expr, $message:expr) => {
                yield Ok(frame(json!({
                    "type": "response.failed",
                    "response": response_json(&env, &model, "failed", &output, None,
                        json!({ "code": $code, "message": $message }), Value::Null),
                }), &mut seq));
            };
        }
        // Set by every arm that BREAKS with its receipt handled; false when the worker
        // closed the channel without Done/Error — settled rejected below, never billed.
        let mut terminal = false;
        while let Some(ev) = rx.recv().await {
            match ev {
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
                            "request_ledger_unavailable",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    prompt_usage = (n_prompt, n_cached);
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
                            "request_ledger_unavailable",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    if let Some(receipt) = receipt.as_mut() {
                        receipt.capture_completion_delta(&text);
                    }
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
                    if let Some(p) = parser.as_mut() {
                        for piece in p.finish() {
                            piece_frames!(piece);
                        }
                    }
                    if let Some(sc) = scrub.as_mut() {
                        let tail = sc.finish();
                        if !tail.is_empty() {
                            piece_frames!(Piece::Content(tail));
                        }
                    }
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.complete(
                            crate::ledger::Usage {
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
                            "request_ledger_unavailable",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    close_item!();
                    let _ = prompt_usage; // Done carries the authoritative counts
                    let usage = Some(usage_json(n_prompt, n_tokens, n_cached));
                    match incomplete_reason(&reason) {
                        Some(cut) => {
                            yield Ok(frame(json!({
                                "type": "response.incomplete",
                                "response": response_json(&env, &model, "incomplete",
                                    &output, usage, Value::Null,
                                    json!({ "reason": cut })),
                            }), &mut seq));
                        }
                        None => {
                            yield Ok(frame(json!({
                                "type": "response.completed",
                                "response": response_json(&env, &model, "completed",
                                    &output, usage, Value::Null, Value::Null),
                            }), &mut seq));
                        }
                    }
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
                            "request_ledger_unavailable",
                            "request completion could not be committed to the billing ledger"
                        );
                        terminal = true;
                        break;
                    }
                    stream_fault!(crate::engine_error_code(err.class), err.message);
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
            stream_fault!(crate::engine_error_code(e.class), e.message);
        }
    };
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(5)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surfaces::{sse_frames, test_envelope};
    use crate::toolcall::ToolStreamParser;
    use std::collections::HashMap;

    /// The Codex-shaped request (instructions, typed input items, flattened function
    /// tools, store:false, include, prompt_cache_key, client_metadata) translates into
    /// the exact internal chat shape and deserializes as ChatCompletionReq.
    #[test]
    fn translate_maps_the_codex_request_shape() {
        let translated = translate(&json!({
            "model": "m",
            "instructions": "You are an agent.",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "fix the bug"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Looking."}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "exec_command", "arguments": "{\"cmd\":\"echo hi\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "hi"},
                {"type": "reasoning", "summary": [], "encrypted_content": null}
            ],
            "tools": [{"type": "function", "name": "exec_command", "strict": false,
                       "description": "run", "parameters": {"type": "object"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "reasoning": {"summary": "auto"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "sess-1",
            "max_output_tokens": 256,
            "client_metadata": {"session_id": "x"}
        }))
        .expect("translate");
        let msgs = translated["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are an agent.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"][0]["text"], "fix the bug");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"][0]["text"], "Looking.");
        // function_call -> assistant tool_calls turn; arguments stay a JSON string.
        assert_eq!(msgs[3]["role"], "assistant");
        assert_eq!(msgs[3]["tool_calls"][0]["id"], "call_1");
        assert_eq!(msgs[3]["tool_calls"][0]["function"]["name"], "exec_command");
        assert_eq!(
            msgs[3]["tool_calls"][0]["function"]["arguments"],
            "{\"cmd\":\"echo hi\"}"
        );
        // function_call_output -> tool turn keyed by call_id.
        assert_eq!(msgs[4]["role"], "tool");
        assert_eq!(msgs[4]["content"], "hi");
        assert_eq!(msgs[4]["tool_call_id"], "call_1");
        // reasoning item consumed; exactly 5 turns.
        assert_eq!(msgs.len(), 5);
        // flattened tool -> nested chat tool.
        assert_eq!(translated["tools"][0]["function"]["name"], "exec_command");
        assert_eq!(translated["tool_choice"], "auto");
        assert_eq!(translated["max_tokens"], 256);
        // prompt_cache_key nominates session affinity, not a cache salt.
        assert_eq!(translated["session_id"], "sess-1");
        assert!(translated.get("cache_salt").is_none());
        // reasoning.summary alone sets no effort override.
        assert!(translated.get("reasoning_effort").is_none());
        let req: ChatCompletionReq = serde_json::from_value(translated).expect("internal shape");
        assert_eq!(req.model, "m");
        assert!(req.stream);
    }

    #[test]
    fn omitted_sampling_stays_absent_through_translation() {
        // STANDARD-SURFACE LAW, /v1/responses half (lane/vendor-default-sampling, 2026-08-19).
        // Omission must survive translation as absence so the per-model vendor default is
        // reachable here too; an explicit 0 must survive as an explicit 0 so greedy stays
        // reachable. Same contract the /v1/messages translator carries.
        let translated = translate(&json!({ "model": "m", "input": "hi" })).expect("translate");
        for key in ["temperature", "top_p"] {
            assert!(
                translated.get(key).is_none(),
                "omitted {key} must translate to an ABSENT key: {translated}"
            );
        }
        let req: ChatCompletionReq = serde_json::from_value(translated).expect("internal shape");
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);

        let greedy = translate(&json!({ "model": "m", "input": "hi", "temperature": 0 }))
            .expect("translate");
        let req: ChatCompletionReq = serde_json::from_value(greedy).expect("internal shape");
        assert_eq!(
            req.temperature,
            Some(0.0),
            "an explicit temperature 0 must reach the engine as true greedy"
        );

        let nulled =
            translate(&json!({ "model": "m", "input": "hi", "temperature": null, "top_p": null }))
                .expect("translate");
        let req: ChatCompletionReq = serde_json::from_value(nulled).expect("internal shape");
        assert_eq!(req.temperature, None, "null is absence, not greedy");
        assert_eq!(req.top_p, None);
    }

    #[test]
    fn translate_refuses_stateful_features_with_clear_messages() {
        let base = |extra: Value| {
            let mut v = json!({ "model": "m", "input": "hi" });
            for (k, val) in extra.as_object().unwrap() {
                v[k] = val.clone();
            }
            v
        };
        for (field, extra) in [
            (
                "previous_response_id",
                json!({"previous_response_id": "resp_x"}),
            ),
            ("store", json!({"store": true})),
            ("conversation", json!({"conversation": "conv_1"})),
            ("background", json!({"background": true})),
            ("truncation", json!({"truncation": "auto"})),
            ("prompt", json!({"prompt": {"id": "p1"}})),
        ] {
            let (msg, _) = translate(&base(extra)).unwrap_err();
            assert!(msg.contains(field), "expected {field} named in: {msg}");
        }
        // item_reference input items are stateful by construction.
        let (msg, _) = translate(&json!({
            "model": "m",
            "input": [{"type": "item_reference", "id": "msg_1"}]
        }))
        .unwrap_err();
        assert!(msg.contains("item_reference"), "got: {msg}");
        // Non-function tool types (web_search, namespace groups) cannot run here, but
        // stock clients send them unconditionally — they are DROPPED from the toolset
        // (the model never sees them), never a 400 that would refuse every default
        // config, and function tools around them survive.
        let v = translate(&json!({
            "model": "m", "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "namespace", "name": "multi_agent_v1"},
                {"type": "function", "name": "exec_command", "parameters": {"type": "object"}}
            ]
        }))
        .unwrap();
        let tools = v["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "exec_command");
        // store:false and absent both pass.
        assert!(translate(&json!({"model": "m", "input": "hi", "store": false})).is_ok());
        assert!(translate(&json!({"model": "m", "input": "hi"})).is_ok());
    }

    #[test]
    fn reasoning_object_is_strict_and_matches_the_chat_surface() {
        // ONE SCHEMA (lane/reasoning-schema-20260823): the same malformed `reasoning` must get the
        // same answer on every surface. Before this lane a non-object `reasoning` was silently
        // ignored HERE while the identical body 400'd on /v1/chat/completions, and every unknown
        // key — including `summary`, which this server does not implement — was dropped behind a
        // 200.
        let req = |reasoning: Value| json!({"model": "m", "input": "hi", "reasoning": reasoning});
        // What codex actually sends is accepted, both the current and the DEPRECATED spelling of
        // the summary field: we deliver reasoning verbatim, which is what "auto" describes.
        for ok in [
            json!({"effort": "xhigh", "summary": "auto"}),
            json!({"summary": "auto"}),
            json!({"generate_summary": "auto"}),
            json!({"effort": "low"}),
        ] {
            translate(&req(ok.clone())).unwrap_or_else(|e| panic!("{ok} must be served: {e:?}"));
        }
        // A summary MODE we cannot perform is named, not silently ignored: this server does not
        // summarise, so promising "concise" would be a promise we break on every request.
        for bad in [
            json!({"summary": "concise"}),
            json!({"generate_summary": "detailed"}),
        ] {
            let (msg, field) = translate(&req(bad.clone())).unwrap_err();
            assert!(msg.contains("not supported"), "{bad}: {msg}");
            assert!(
                msg.contains("verbatim"),
                "{bad}: the refusal must say what we DO do: {msg}"
            );
            assert_eq!(field.as_deref(), Some("reasoning"));
        }
        // A wrong-TYPED summary is a refusal too, not an unset: `as_str()` used to turn it into
        // None and read it as absent, which is a 200 that changes nothing.
        for bad in [json!({"summary": 3}), json!({"generate_summary": true})] {
            let (msg, _) = translate(&req(bad.clone()))
                .err()
                .unwrap_or_else(|| panic!("{bad} must not be accepted-and-ignored"));
            assert!(msg.contains("not supported"), "{bad}: {msg}");
        }
        // JSON null is "not set" for a KEY on every surface — the last key-level divergence this
        // lane closed (chat used to 400 `{"effort": null}` while this surface read it as unset).
        for ok in [
            json!({"effort": null}),
            json!({"summary": null}),
            json!({"effort": "low", "summary": null}),
        ] {
            translate(&req(ok.clone())).unwrap_or_else(|e| panic!("{ok} must be served: {e:?}"));
        }
        // Unknown keys and wrong types refuse by name, exactly as on the chat surface.
        let (msg, _) = translate(&req(json!({"max_tokens": 1024}))).unwrap_err();
        assert!(msg.contains("reasoning.max_tokens"), "{msg}");
        let (msg, _) = translate(&req(json!({"banana": 1}))).unwrap_err();
        assert!(msg.contains("reasoning.banana"), "{msg}");
        let (msg, _) = translate(&req(json!({"effort": 3}))).unwrap_err();
        assert!(msg.contains("must be a string"), "{msg}");
        // And a NON-OBJECT reasoning is the same 400 chat gives, closing that divergence.
        let (msg, _) = translate(&req(json!("high"))).unwrap_err();
        assert!(msg.contains("must be an object"), "{msg}");
    }

    #[test]
    fn translate_maps_reasoning_effort_and_text_format() {
        let v = translate(&json!({
            "model": "m", "input": "hi", "reasoning": {"effort": "xhigh", "summary": "auto"}
        }))
        .unwrap();
        // validated for membership, forwarded RAW: the caps-aware clamp (dsv4 "max"
        // exemption) happens once, at plan build (parse_think).
        assert_eq!(v["reasoning_effort"], "xhigh");
        let v = translate(&json!({
            "model": "m", "input": "hi", "reasoning": {"effort": "none"}
        }))
        .unwrap();
        assert_eq!(v["reasoning_effort"], "none");
        // out-of-table value still 400s at translation, naming the Responses param.
        let err = translate(&json!({
            "model": "m", "input": "hi", "reasoning": {"effort": "banana"}
        }))
        .unwrap_err();
        assert_eq!(err.1.as_deref(), Some("reasoning"));
        let v = translate(&json!({
            "model": "m", "input": "hi",
            "text": {"format": {"type": "json_schema", "name": "out", "strict": true,
                                 "schema": {"type": "object"}}}
        }))
        .unwrap();
        assert_eq!(v["response_format"]["type"], "json_schema");
        assert_eq!(v["response_format"]["json_schema"]["name"], "out");
        assert_eq!(
            v["response_format"]["json_schema"]["schema"]["type"],
            "object"
        );
        let v = translate(&json!({
            "model": "m", "input": "hi", "text": {"format": {"type": "text"}}
        }))
        .unwrap();
        assert!(v.get("response_format").is_none());
    }

    /// GOLDEN TRANSCRIPT (text): the exact Responses event grammar — created first,
    /// output_item.added BEFORE any delta, full item on output_item.done, terminal
    /// response.completed carrying id + usage.
    #[tokio::test]
    async fn sse_text_stream_speaks_the_responses_grammar() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
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
        let resp = responses_sse(
            rx,
            None,
            test_envelope("resp_g1"),
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
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        assert_eq!(frames[0].1["response"]["status"], "in_progress");
        assert_eq!(frames[3].1["delta"], "Hel");
        assert_eq!(frames[5].1["text"], "Hello");
        let done_item = &frames[7].1["item"];
        assert_eq!(done_item["type"], "message");
        assert_eq!(done_item["content"][0]["type"], "output_text");
        assert_eq!(done_item["content"][0]["text"], "Hello");
        let completed = &frames[8].1["response"];
        assert_eq!(completed["id"], "resp_g1");
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["usage"]["input_tokens"], 10);
        assert_eq!(
            completed["usage"]["input_tokens_details"]["cached_tokens"],
            4
        );
        assert_eq!(completed["usage"]["output_tokens"], 2);
        assert_eq!(completed["usage"]["total_tokens"], 12);
        assert_eq!(completed["output"][0]["content"][0]["text"], "Hello");
        // sequence numbers are monotonic; every payload self-describes its type.
        let mut last = 0;
        for (name, data) in &frames {
            assert_eq!(data["type"], json!(name));
            let seq = data["sequence_number"].as_u64().unwrap();
            assert!(seq > last);
            last = seq;
        }
    }

    /// GOLDEN TRANSCRIPT (tool round-trip): a template-law tool emission becomes a
    /// function_call item whose authoritative form rides output_item.done — call_id,
    /// name, and STRING arguments — followed by response.completed.
    #[tokio::test]
    async fn reasoning_text_is_returned_as_a_reasoning_item_and_absent_when_off() {
        // OWNER ACCEPTANCE GATE (2026-08-23): on this surface reasoning is delivered as a
        // `reasoning` output item (`summary[0].text`, streamed as
        // `response.reasoning_summary_text.delta`) — never stripped server-side — and a
        // reasoning-off generation produces NO reasoning item.
        let drive = |think_text: bool| async move {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            tx.send(Event::PromptUsage {
                n_prompt: 10,
                n_cached: 0,
            })
            .unwrap();
            let body = if think_text {
                "a plan</think>\n\nanswer"
            } else {
                "answer"
            };
            tx.send(Event::Token {
                id: 1,
                text: body.into(),
            })
            .unwrap();
            tx.send(Event::Done {
                stop_reason: "Eos".into(),
                n_tokens: 3,
                n_prompt: 10,
                n_cached: 0,
                elapsed_s: 0.1,
                spec: None,
            })
            .unwrap();
            drop(tx);
            // reasoning-on = the think-open prompt arms the reasoning splitter; off = no parser
            // (the NoThink path builds none), so the stream is the plain-text grammar.
            let parser = think_text.then(crate::toolcall::ToolStreamParser::reasoning_only);
            let resp = responses_sse(
                rx,
                None,
                test_envelope("resp_think"),
                "m".into(),
                parser,
                Vec::new(),
                None,
            )
            .into_response();
            sse_frames(resp).await
        };
        let on = drive(true).await;
        let names: Vec<&str> = on.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"response.reasoning_summary_text.delta"),
            "reasoning must stream as summary-text deltas: {names:?}"
        );
        let completed = &on
            .iter()
            .find(|(n, _)| n == "response.completed")
            .unwrap()
            .1;
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(
            output[0]["summary"][0]["text"], "a plan",
            "the completed response must carry the reasoning item verbatim"
        );
        assert_eq!(output[1]["type"], "message");
        let off = drive(false).await;
        let names: Vec<&str> = off.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            !names.iter().any(|n| n.contains("reasoning")),
            "a reasoning-off stream must carry no reasoning events: {names:?}"
        );
        let completed = &off
            .iter()
            .find(|(n, _)| n == "response.completed")
            .unwrap()
            .1;
        let output = completed["response"]["output"].as_array().unwrap();
        assert!(
            output.iter().all(|item| item["type"] != "reasoning"),
            "a reasoning-off response must carry no reasoning item"
        );
    }

    #[tokio::test]
    async fn sse_tool_call_stream_produces_function_call_items() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 5,
            n_cached: 0,
        })
        .unwrap();
        tx.send(Event::Token {
            id: 1,
            text: "<tool_call>\n<function=exec_command>\n<parameter=cmd>\necho probe\n\
                   </parameter>\n</function>\n</tool_call>"
                .into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 1,
            n_prompt: 5,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let mut schemas: HashMap<String, HashMap<String, String>> = HashMap::new();
        schemas.insert(
            "exec_command".into(),
            [("cmd".to_string(), "string".to_string())].into(),
        );
        let parser = ToolStreamParser::new(schemas, false);
        let resp = responses_sse(
            rx,
            None,
            test_envelope("resp_g2"),
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
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed"
            ]
        );
        let added = &frames[1].1["item"];
        assert_eq!(added["type"], "function_call");
        assert_eq!(added["name"], "exec_command");
        assert_eq!(added["arguments"], "");
        let item = &frames[4].1["item"];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["name"], "exec_command");
        assert!(item["call_id"].as_str().unwrap().starts_with("call_"));
        assert!(item["id"].as_str().unwrap().starts_with("fc_"));
        let args: Value = serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"cmd": "echo probe"}));
        // the terminal response also carries the item (spec parity).
        assert_eq!(
            frames[5].1["response"]["output"][0]["type"],
            "function_call"
        );
    }

    /// Token-budget cutoffs are response.incomplete (max_output_tokens), never a fake
    /// "completed"; faults are response.failed.
    #[tokio::test]
    async fn sse_terminal_states_are_honest() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Token {
            id: 1,
            text: "part".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "MaxNew".into(),
            n_tokens: 1,
            n_prompt: 2,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let resp = responses_sse(
            rx,
            None,
            test_envelope("resp_g3"),
            "m".into(),
            None,
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let (name, data) = frames.last().unwrap();
        assert_eq!(name, "response.incomplete");
        assert_eq!(data["response"]["status"], "incomplete");
        assert_eq!(
            data["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Error(crate::worker::EngineError::overloaded(
            "vram exhausted",
        )))
        .unwrap();
        drop(tx);
        let resp = responses_sse(
            rx,
            None,
            test_envelope("resp_g4"),
            "m".into(),
            None,
            Vec::new(),
            None,
        )
        .into_response();
        let frames = sse_frames(resp).await;
        let (name, data) = frames.last().unwrap();
        assert_eq!(name, "response.failed");
        assert_eq!(data["response"]["status"], "failed");
        assert_eq!(data["response"]["error"]["code"], "overloaded");
        assert_eq!(data["response"]["error"]["message"], "vram exhausted");
    }

    #[test]
    fn non_streaming_response_shape_and_incomplete_mapping() {
        let env = test_envelope("resp_ns");
        let body = response_json(
            &env,
            "m",
            "completed",
            &[text_item("msg_1", "hi")],
            Some(usage_json(3, 1, 0)),
            Value::Null,
            Value::Null,
        );
        assert_eq!(body["object"], "response");
        assert_eq!(body["id"], "resp_ns");
        assert_eq!(body["store"], false);
        assert_eq!(body["output"][0]["content"][0]["text"], "hi");
        assert_eq!(body["usage"]["total_tokens"], 4);
        assert_eq!(incomplete_reason("MaxNew"), Some("max_output_tokens"));
        assert_eq!(incomplete_reason("ContextFull"), Some("max_output_tokens"));
        assert_eq!(incomplete_reason("Eos"), None);
    }
}
