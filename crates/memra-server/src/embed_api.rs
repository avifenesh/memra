//! `/v1/embeddings` + `/v1/rerank` — the CAPTURE surfaces (lane/embed-serve).
//!
//! Both surfaces run a plain PREFILL of a causal LM and read the final prompt position:
//! embeddings return the last-token post-final-norm hidden state (the Qwen3-Embedding
//! pooling convention: last token, then L2 normalization, with optional MRL truncation),
//! rerank returns a relevance score from the "yes"/"no" token logits (the Qwen3-Reranker
//! convention). No decode step runs (`max_new: 0`).
//!
//! Admission parity with every other surface, on purpose: the same canonical-model,
//! tenant-auth, lane, budget-admission, ledger-receipt, rate-limit-slot, backpressure and
//! `[meter]` sequence as `/v1/completions` — one admitted worker request PER INPUT, so an
//! embeddings array bills per input exactly like N single calls. Capture requests bypass
//! every KV reuse tier and the spec path in the worker (a cache hit would skip the prime
//! the capture reads from), and prime alone — see `worker::CaptureSpec`.
//!
//! LANE LAW (owner, 2026-08-26): these are the subordinate-priority surfaces. They take
//! whatever lane the request/keyring resolves (batch-class keys land on harvest and are
//! shed under interactive load by the SLO admission — that IS the product contract:
//! embeddings/rerank may be slow; the paying decode lanes are never taxed).

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use crate::worker::{CaptureSpec, Cmd, Event};
use crate::{AppState, Envelope, auth, ledger, worker};

/// OpenAI `/v1/embeddings` request. `input` accepts a string or an array of strings
/// (token-id arrays are not accepted — the worker owns tokenization on this surface).
#[derive(Deserialize)]
pub(crate) struct EmbeddingsReq {
    pub model: String,
    pub input: EmbeddingsInput,
    /// MRL truncation: keep the first N dimensions, then re-normalize (the matryoshka
    /// convention the Qwen3-Embedding family is trained for).
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Only "float" is served; "base64" is refused honestly.
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

/// Cohere-shaped `/v1/rerank` request (the de-facto rerank wire format).
#[derive(Deserialize)]
pub(crate) struct RerankReq {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Task instruction rendered into the judge prompt (vendor default below).
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub return_documents: Option<bool>,
    #[serde(default)]
    pub timeout_ms: Option<serde_json::Value>,
}

/// Vendor default instruction (Qwen3-Reranker model card).
const RERANK_DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// The Qwen3-Reranker judge prompt, rendered EXACTLY as the vendor's usage example
/// renders it (system + user + forced empty think block). The score is read from the
/// final position's "yes"/"no" logits.
fn rerank_prompt(instruction: &str, query: &str, document: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on \
         the Query and the Instruct provided. Note that the answer can only be \"yes\" \
         or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: \
         {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n\
         </think>\n\n"
    )
}

/// One admitted capture request: the full `/v1/completions` admission sequence with a
/// prefill-only worker request, collected to its `PromptCapture` + `Done` pair.
struct CaptureOutcome {
    hidden: Option<Vec<f32>>,
    logits: Vec<f32>,
    n_prompt: usize,
}

/// Admit one capture request and collect its result. Every rejection settles its ledger
/// receipt exactly like the blocking completions path.
#[allow(clippy::too_many_arguments)]
async fn run_capture(
    st: &AppState,
    headers: &HeaderMap,
    env: &Envelope,
    tenant: &auth::TenantCtx,
    model: &str,
    prompt_text: String,
    capture: CaptureSpec,
    route: &'static str,
    deadline: &crate::RequestDeadline,
) -> Result<CaptureOutcome, Response> {
    let cache_ns = match crate::tenant_namespace(tenant, &None::<String>) {
        Ok(ns) => ns,
        Err(msg) => return Err(crate::bad_request(msg, Some("cache_salt"))),
    };
    // BATCH-CLASS BY CONSTRUCTION (owner subordinate-priority law, 2026-08-26): every
    // capture request rides the HARVEST lane regardless of the key's class or any
    // x-lane header — the published contract is batch-class (shed-and-retry under
    // interactive load), and an interactive-lane capture request would tax the paying
    // decode lanes (measured: -8%/req/s to -50%, coresident lane cells 1/3). The
    // header/key lane resolution still runs so its 403s and validation stay identical.
    if let Err(resp) = crate::lane_for_tenant(headers, tenant) {
        return Err(resp);
    }
    let lane = crate::lanes::Lane::Harvest;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let mut request = worker::Request {
        model: model.to_string(),
        prompt_ids: Vec::new(),
        prompt_text,
        chat: false,
        chat_turns: Vec::new(),
        tools_json: Vec::new(),
        tools_struct: Vec::new(),
        think: memra_tokenizer::chat::ThinkMode::Default,
        reasoning_effort: None,
        params: memra_engine::decode::GenParams {
            max_new: 0, // prefill-only: the capture IS the response
            max_ctx: None,
            eos: Vec::new(),
        },
        sampler_cfg: memra_engine::sampler::SamplerConfig::default(),
        stop_strings: Vec::new(),
        trace_id: None,
        max_prompt_tokens: None,
        cache_ns,
        affinity: None,
        lane,
        grammar: None,
        prepared_constraint: None,
        constraint_ready: None,
        oom_retries: 0,
        spec_k_replay: None,
        prepared_prompt: None,
        images: Vec::new(),
        gemma_images: Vec::new(),
        capture: Some(capture),
        vision_memory: None,
        ttft: None,
        tx,
    };
    if let Err((message, param)) = crate::apply_model_request_limits(
        &mut request,
        st.openrouter_metadata.get(model),
        st.caps.get(model),
    ) {
        return Err(crate::bad_request(&message, Some(param)));
    }
    if crate::draining() {
        let receipt =
            crate::start_request_receipt(st, env, tenant, model, route, lane, false, None);
        return Err(crate::ledger_rejected(
            receipt,
            crate::drain_response(),
            "draining",
            &env.id,
        ));
    }
    let budget_permit = match crate::admit_tenant_budget(st, tenant, &mut request) {
        Ok(permit) => permit,
        Err(rejection) => {
            let (response, error_code) = rejection.into_response();
            let receipt =
                crate::start_request_receipt(st, env, tenant, model, route, lane, false, None);
            return Err(crate::ledger_rejected(
                receipt, response, error_code, &env.id,
            ));
        }
    };
    let receipt =
        crate::start_request_receipt(st, env, tenant, model, route, lane, false, budget_permit);
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
    let pending_admit = match crate::reserve_pending_admit(st, lane, &rl, *deadline) {
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
    crate::meter_admit(env, tenant, model, lane);
    if st.cmd_tx.send(Cmd::Generate(Box::new(request))).is_err() {
        drop(pending_admit);
        return Err(crate::ledger_rejected(
            receipt,
            rl.attach(crate::worker_unavailable_response()),
            "worker_unavailable",
            &env.id,
        ));
    }
    pending_admit.commit();
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
                rl.attach(crate::deadline_exceeded_response(deadline.ms, false)),
                "deadline_exceeded",
                "deadline_exceeded",
                &env.id,
            ));
        }
    };
    collect_capture(rx, receipt, guard, rl, env, deadline).await
}

/// Drain the worker's event stream for one capture request with the blocking-path
/// receipt discipline: prompt usage recorded, terminal `Done` completes the receipt
/// (completion_tokens is 0 by construction), `Error` settles it rejected.
async fn collect_capture(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    mut receipt: Option<ledger::PendingReceipt>,
    _guard: crate::InflightGuard,
    rl: crate::RateLimit,
    env: &Envelope,
    deadline: &crate::RequestDeadline,
) -> Result<CaptureOutcome, Response> {
    let mut hidden: Option<Vec<f32>> = None;
    let mut logits: Vec<f32> = Vec::new();
    let mut got_capture = false;
    let mut n_prompt;
    loop {
        let ev = match tokio::time::timeout_at(deadline.at, rx.recv()).await {
            Ok(Some(ev)) => ev,
            Ok(None) => {
                if let Some(mut receipt) = receipt.take() {
                    let _ = receipt.reject(503, "worker_unavailable");
                }
                return Err(rl.attach(crate::worker_unavailable_response()));
            }
            Err(_) => {
                if let Some(mut receipt) = receipt.take() {
                    let _ = receipt.reject(408, "deadline_exceeded");
                }
                return Err(rl.attach(crate::deadline_exceeded_response(deadline.ms, false)));
            }
        };
        match ev {
            Event::PromptUsage {
                n_prompt: np,
                n_cached,
            } => {
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.record_prompt_usage(np as u64, n_cached as u64)
                {
                    eprintln!(
                        "[ledger] ERROR: request {} partial prompt receipt failed: {err}",
                        env.id
                    );
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return Err(rl.attach(crate::request_ledger_error_response()));
                }
            }
            Event::PromptCapture {
                hidden: h,
                logits: l,
            } => {
                hidden = h;
                logits = l;
                got_capture = true;
            }
            Event::Token { .. } | Event::TokenSnapshot(_) => {}
            Event::Done {
                n_prompt: np,
                n_cached,
                elapsed_s,
                ..
            } => {
                n_prompt = np;
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.complete(
                        ledger::Usage {
                            prompt_tokens: np as u64,
                            cached_prompt_tokens: n_cached as u64,
                            completion_tokens: 0,
                        },
                        elapsed_s,
                    )
                {
                    eprintln!(
                        "[ledger] ERROR: request {} completion receipt failed: {err}",
                        env.id
                    );
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return Err(rl.attach(crate::request_ledger_error_response()));
                }
                if !got_capture {
                    // Worker finished without a capture event — engine-side bug, not
                    // the caller's fault.
                    return Err(rl.attach(crate::engine_error_response(
                        &worker::EngineError::engine(
                            "capture request finished without a PromptCapture event",
                        ),
                    )));
                }
                return Ok(CaptureOutcome {
                    hidden,
                    logits,
                    n_prompt,
                });
            }
            Event::Error(err) => {
                // G6: the class decides the status — same law as the completions path.
                if let Some(receipt) = receipt.as_mut() {
                    let _ = receipt.reject(
                        crate::class_http(err.class).0.as_u16(),
                        crate::engine_error_code(err.class),
                    );
                }
                return Err(rl.attach(crate::engine_error_response(&err)));
            }
        }
    }
}

/// L2-normalize in place; returns the input untouched when its norm is 0.
fn l2_normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

pub(crate) async fn embeddings(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<EmbeddingsReq>,
) -> Response {
    let env = Envelope::new(false);
    match crate::canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return crate::with_request_id(
                &env.id,
                crate::model_not_found_response(&st.models, &req.model),
            );
        }
    }
    if req.encoding_format.as_deref().is_some_and(|f| f != "float") {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                "only encoding_format=\"float\" is served",
                Some("encoding_format"),
            ),
        );
    }
    let inputs: Vec<String> = match req.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    const MAX_INPUTS: usize = 32;
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                &format!("input must carry 1..={MAX_INPUTS} strings"),
                Some("input"),
            ),
        );
    }
    if req.dimensions.is_some_and(|d| d == 0) {
        return crate::with_request_id(
            &env.id,
            crate::bad_request("dimensions must be >= 1", Some("dimensions")),
        );
    }
    let tenant = match crate::authenticate(&st.api_auth, &headers) {
        Ok(t) => t,
        Err(resp) => return crate::with_request_id(&env.id, resp),
    };
    let deadline = match crate::parse_timeout_ms(req.timeout_ms.as_ref()) {
        Ok(ms) => crate::RequestDeadline::starting_now(ms),
        Err(msg) => {
            return crate::with_request_id(&env.id, crate::bad_request(&msg, Some("timeout_ms")));
        }
    };
    let mut data = Vec::with_capacity(inputs.len());
    let mut prompt_tokens = 0usize;
    for (index, input) in inputs.into_iter().enumerate() {
        let outcome = match run_capture(
            &st,
            &headers,
            &env,
            &tenant,
            &req.model,
            input,
            CaptureSpec {
                hidden: true,
                logit_pieces: Vec::new(),
            },
            "/v1/embeddings",
            &deadline,
        )
        .await
        {
            Ok(o) => o,
            Err(resp) => return crate::with_request_id(&env.id, resp),
        };
        let Some(mut vector) = outcome.hidden else {
            // The worker's tokenwise fallback has no hidden stack (eager-only model) —
            // this model cannot serve embeddings.
            return crate::with_request_id(
                &env.id,
                crate::bad_request(
                    "this model cannot serve embeddings (no prime-path hidden state)",
                    Some("model"),
                ),
            );
        };
        // Qwen3-Embedding convention: L2-normalize; MRL truncation keeps the first
        // `dimensions` and re-normalizes.
        if let Some(d) = req.dimensions {
            if d > vector.len() {
                return crate::with_request_id(
                    &env.id,
                    crate::bad_request(
                        &format!("dimensions exceeds the model width {}", vector.len()),
                        Some("dimensions"),
                    ),
                );
            }
            vector.truncate(d);
        }
        l2_normalize(&mut vector);
        prompt_tokens += outcome.n_prompt;
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": vector,
        }));
    }
    let body = json!({
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": { "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens },
    });
    crate::with_request_id(&env.id, Json(body).into_response())
}

pub(crate) async fn rerank(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<RerankReq>,
) -> Response {
    let env = Envelope::new(false);
    match crate::canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return crate::with_request_id(
                &env.id,
                crate::model_not_found_response(&st.models, &req.model),
            );
        }
    }
    const MAX_DOCS: usize = 64;
    if req.documents.is_empty() || req.documents.len() > MAX_DOCS {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                &format!("documents must carry 1..={MAX_DOCS} strings"),
                Some("documents"),
            ),
        );
    }
    let tenant = match crate::authenticate(&st.api_auth, &headers) {
        Ok(t) => t,
        Err(resp) => return crate::with_request_id(&env.id, resp),
    };
    let deadline = match crate::parse_timeout_ms(req.timeout_ms.as_ref()) {
        Ok(ms) => crate::RequestDeadline::starting_now(ms),
        Err(msg) => {
            return crate::with_request_id(&env.id, crate::bad_request(&msg, Some("timeout_ms")));
        }
    };
    let instruction = req
        .instruction
        .as_deref()
        .unwrap_or(RERANK_DEFAULT_INSTRUCTION)
        .to_string();
    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(req.documents.len());
    let mut total_tokens = 0usize;
    for (index, document) in req.documents.iter().enumerate() {
        let outcome = match run_capture(
            &st,
            &headers,
            &env,
            &tenant,
            &req.model,
            rerank_prompt(&instruction, &req.query, document),
            CaptureSpec {
                hidden: false,
                logit_pieces: vec!["yes".to_string(), "no".to_string()],
            },
            "/v1/rerank",
            &deadline,
        )
        .await
        {
            Ok(o) => o,
            Err(resp) => return crate::with_request_id(&env.id, resp),
        };
        let (yes, no) = match outcome.logits.as_slice() {
            [y, n] if *y > f32::MIN && *n > f32::MIN => (*y as f64, *n as f64),
            _ => {
                // "yes"/"no" are not single tokens in this model's vocabulary — this
                // model cannot serve the yes/no rerank read.
                return crate::with_request_id(
                    &env.id,
                    crate::bad_request(
                        "this model cannot serve rerank (\"yes\"/\"no\" are not single \
                         vocabulary tokens)",
                        Some("model"),
                    ),
                );
            }
        };
        // P(yes) over the {yes, no} pair — the vendor scoring rule, max-subtracted.
        let m = yes.max(no);
        let score = ((yes - m).exp()) / ((yes - m).exp() + (no - m).exp());
        scored.push((index, score));
        total_tokens += outcome.n_prompt;
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_n = req.top_n.unwrap_or(scored.len()).min(scored.len());
    let return_documents = req.return_documents.unwrap_or(false);
    let results: Vec<serde_json::Value> = scored[..top_n]
        .iter()
        .map(|(index, score)| {
            let mut row = json!({ "index": index, "relevance_score": score });
            if return_documents {
                row["document"] = json!({ "text": req.documents[*index] });
            }
            row
        })
        .collect();
    let body = json!({
        "model": req.model,
        "results": results,
        "usage": { "total_tokens": total_tokens },
    });
    crate::with_request_id(&env.id, Json(body).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_prompt_renders_the_vendor_judge_format() {
        let p = rerank_prompt("inst", "q", "d");
        assert!(p.starts_with("<|im_start|>system\nJudge whether the Document"));
        assert!(p.contains("<Instruct>: inst\n<Query>: q\n<Document>: d<|im_end|>"));
        // the forced empty think block is part of the scored distribution
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn l2_normalize_produces_a_unit_vector_and_survives_zero() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0f32, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    #[test]
    fn mrl_truncation_renormalizes_the_prefix() {
        // truncate-then-renormalize: unit norm over the KEPT dims, not the original
        let mut v = vec![1.0f32, 1.0, 1.0, 1.0];
        v.truncate(2);
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rerank_score_is_p_yes_over_the_pair() {
        let (yes, no) = (2.0f64, 0.0f64);
        let m = yes.max(no);
        let score = ((yes - m).exp()) / ((yes - m).exp() + (no - m).exp());
        assert!((score - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
    }
}
