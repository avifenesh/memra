//! memra-server (BASE-4): a minimal OpenAI-ish HTTP server that serves 2-4 concurrent agents across
//! DIFFERENT models on one endpoint via a single GPU worker thread + step-interleave scheduler.
//!
//! Architecture (see worker.rs): axum runs on a tokio runtime; ONE dedicated std::thread owns the
//! Engine + every loaded HybridModel (CUDA context is thread-affine). Handlers submit `Cmd`s over a
//! std mpsc channel and receive tokens back over a per-request tokio mpsc channel.
//!
//! Endpoints (the full set — `router()` below is the authority):
//!   GET  /health, GET /livez     -> the SAME handler (`health_live`): INFERENCE liveness, not
//!                                     process liveness. {"status":"ok"|"draining"|"unhealthy",
//!                                     "models":[...], "worker":{phase, beat_age_ms, tick_max_ms,
//!                                     stall_threshold_ms, generation, xid_warnings}} + a
//!                                     top-level "detail" on a red. Draining stays 200; dead /
//!                                     GPU-faulted / stalled / loading is 503 (serve-hardening
//!                                     2026-08-06).
//!   GET  /readyz                 -> routability, same payload shape with
//!                                     "status":"ready"|"not_ready". Unready is NOT a restart
//!                                     request — draining and loading are healthy-but-unroutable.
//!   GET  /models                 -> {"data":[{"id":name},...]}  (OpenAI-ish);
//!                                     ?schema=openrouter -> Provider Monitor schema 2.4,
//!                                     ?schema=openmodels -> OpenModels provider feed.
//!   GET  /v1/models              -> existing catalog-style model list (context_length,
//!                                     architecture, pricing stub, top_provider; serve-tail).
//!   GET  /metrics                -> flat serving counters + step latency percentiles.
//!   GET  /yield/metrics          -> per-lane x-lane QoS counters + engine-truth step p50/p99
//!                                     (lane/qos-p95 2026-08-02).
//!   POST /v1/completions         -> {model,prompt|prompt_ids,max_tokens,temperature?,top_p?,top_k?,
//!                                     seed?,stop?,chat?,stream?,cache_salt?}. stream=true => SSE
//!                                     token-by-token; else a single JSON {text,tokens,stop_reason}.
//!   POST /v1/chat/completions    -> OpenAI chat messages rendered by the GGUF chat template;
//!                                     OpenAI message/chunk response shapes. `tools`/`tool_choice`
//!                                     (auto|none) + role:"tool" turns render through the
//!                                     template's own <tools> branch; emitted <tool_call> blocks
//!                                     parse into OpenAI `tool_calls` (+"tool_calls" finish);
//!                                     `reasoning_effort`/`reasoning` map onto the template's
//!                                     think switch (serve-tools lane, 2026-08-02).
//!
//! CONFIG: MEMRA_MODELS="name=/path.gguf[+/draft.gguf],name2=hf:owner/repo,name3=/hf_ckpt_dir"
//! (comma-separated; `+draft.gguf` attaches that model's regime draft — docs/DRAFT-REGIME.md).
//! A model path may be a GGUF file OR an HF safetensors checkpoint directory
//! (config.json + model.safetensors[.index.json] — the run-safetensors load path; serve-st
//! lane 2026-08-04). Defaults to the BASE-4 test pair (main=27B, judge=9B) if unset.
//! MEMRA_ADDR sets the bind addr.
//!
//! LIFECYCLE: SIGTERM = graceful drain (gap-scan F11) — new completion requests 503 with
//! Retry-After, /health reports "draining", in-flight requests (streams included) finish
//! up to MEMRA_DRAIN_S (default 30s), then the process exits 0. Completion responses carry
//! X-RateLimit-Limit/-Remaining/-Reset (concurrency-slot semantics; gap-scan F12).

/// x-lane QoS (lane/dl-metering gate, QoS-only extraction 2026-08-02): lane types, SLO
/// admission policy, engine-truth step stats live in the memra-lanes crate so out-of-process
/// controllers (the sidecar shape) can share them.
///
/// `pub`: the key file format, lifecycle helpers, and single-key path are the API a
/// deployment-owned binary provisions against (engine-billing-extraction-20260829).
pub mod auth;
pub(crate) mod constrained;
/// Dead-darklane background jobs (lane/darklane-training, 2026-08-07): valley detection over
/// worker truth (phase + beat age + pending admits) and a yield-first background job runner —
/// a lane class BELOW every serving lane. Engine mechanics only; policy lives product-side.
pub(crate) mod darklane;
/// Inference-liveness state (lane/serve-hardening, gaps G5 + G24): the worker heartbeat every
/// health answer is derived from, the Xid/GPU-fault watcher, and the sd_notify half of the
/// systemd contract. Process liveness is NOT inference liveness — this module is the difference.
pub(crate) mod health;
pub(crate) mod lanes {
    pub use memra_lanes::*;
}
/// Predictive-admission SHADOW instrumentation (darklanes Arc D2 engine gaps,
/// lane/d2-engine-gaps-20260831): the per-model in-flight book, the rolling
/// completion-length history, and the `[admit-predict]` receipt line behind
/// `MEMRA_ADMIT_PREDICT_SHADOW` (default 0). Logs verdicts, never enforces.
mod admit_predict;
/// CPU affinity for the GPU worker thread (`MEMRA_WORKER_CPUSET`, alias
/// `MEMRA_WORKER_AFFINITY` honored, default OFF —
/// lane/glm5-host-audit 2026-09-01). Engine-wide, not one family's: every served family's
/// decode tick runs on the single `memra-gpu-worker` thread this module can pin, and that
/// thread was measured migrating across L3 domains on a 12-CCD EPYC while 192 unpinned tokio
/// workers shared the same CPUs. Machine config, so it defaults OFF and stays a seam.
mod affinity;
/// Translation surfaces (lane/api-surfaces, 2026-08-17): the Anthropic Messages API and
/// the OpenAI Responses API served over the SAME chat-completions core — same tenant
/// auth, budget admission, ledger receipts, metering and capture posture; only the wire
/// rendering differs. `surfaces` is the shared admission driver; the other two are the
/// per-dialect request translations and response renderers.
mod anthropic;
/// The `system_fingerprint` identity, shared with `build.rs` (which `include!`s this same
/// file to bake the value). Compiled into the crate so the fingerprint tests can re-derive
/// the id from the working tree instead of pinning a second copy of the algorithm.
#[allow(dead_code)] // one implementation, two callers: each uses a subset.
mod build_id;
mod dsv4_serve;
mod embed_api;
/// The admission/accounting seam: the server admits, denies, and reports counts;
/// what admission MEANS — budgets, prices, tenancy policy — is a deployment concern,
/// supplied behind `metering::Metering` through `ServerWiring`. The stock binary
/// ships NO accounting (only the engine is open; the business tier lives in the
/// deployment's own binary — engine-billing-extraction-20260829, owner razor
/// 2026-08-29: "only engine is open, business is private").
pub mod metering;
mod responses_api;
mod surfaces;
mod toolcall;
mod ttft;
mod worker;

use std::collections::HashMap;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use axum::{
    Extension, Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, FromRequest, Query, Request as AxumRequest, State},
    http::{
        HeaderMap, StatusCode,
        header::{CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING},
    },
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
    routing::{get, post},
};
use futures_core::Stream as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tower::ServiceExt as _;

use memra_engine::decode::GenParams;
use memra_engine::sampler::SamplerConfig;
use memra_tokenizer::{
    Tokenizer,
    chat::{self, ThinkMode, ToolCall as TmplToolCall, Turn as TmplTurn},
};
use toolcall::{ParsedToolCall, Piece, ToolStreamParser};
use worker::{Cmd, Event, ModelCaps, Request, SharedMetrics};

/// Explicit HTTP body ceiling for every inference route (hermes finding, 2026-08-19).
/// axum's DefaultBodyLimit is 2 MiB, which silently capped the ADVERTISED surface: a
/// 262,144-token prompt sent as `prompt_ids` is ~2.8 MiB of JSON on its own, and the
/// vision envelope (base64 data URIs) is far past that — sold features died at the
/// extractor with a shapeless 413. Budget, itemized from the advertised maxima:
///
///   prompt   262,144 tokens x 16 B/token JSON-escaped upper bound     =   4 MiB
///   images   VISION_MAX_IMAGES (8) x 12 MiB raw x 4/3 base64          = 128 MiB
///   videos   2 x 12 MiB raw GIF x 4/3 base64                          =  32 MiB
///   message/tools envelope headroom                                    =   4 MiB
///                                                            requirement 168 MiB
///
/// Ceiling: 192 MiB — covers the requirement with headroom while staying finite (the
/// per-lane concurrency slots bound how many of these can buffer at once). Applies to
/// EVERY route on the app router, including `/v1/messages`' raw `Bytes` path (the
/// `DefaultBodyLimit` extension reaches `Bytes` and `Json` extractors alike).
///
/// The "12 MiB raw" per-image line item is ENFORCED, not just budgeted: both data-URI
/// decoders (`vision_pre::decode_data_uri`, `vision_gemma::gemma_decode_data_uri`)
/// refuse a payload past `vision_pre::IMG_MAX_RAW_BYTES` by encoded LENGTH, before any
/// decode allocation, with a named 400 (hermes review finding 48f96cb4cd37e436: until
/// then only this body ceiling bounded the decode, which runs in the content walkers
/// BEFORE slot admission, so one image could expand ~144 MiB of host bytes pre-check).
const MAX_BODY_BYTES: usize = 192 * 1024 * 1024;
const MAX_BODY_ADMISSIONS: usize = 4;
const MAX_SMALL_BODY_ADMISSIONS: usize = 32;
// Small JSON requests are already bounded by the extractor and should not wait behind a
// deliberately slow large upload. They use their own finite pool; unknown-length/chunked bodies
// still take the large-body path.
#[allow(clippy::identity_op)] // allow: the explicit +0/*1/>>0 terms document the lane/byte symmetry of the reference layout
const BODY_ADMISSION_BYPASS_BYTES: usize = 1 * 1024 * 1024;
const BODY_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);
const BODY_READ_RATE_BYTES_PER_SEC: u64 = 2 * 1024 * 1024;
const BODY_READ_TIMEOUT_MAX: std::time::Duration = std::time::Duration::from_secs(180);
const BODY_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const BODY_ADMISSION_RETRY_AFTER_S: u64 = 1;
const MAX_STOP_SEQUENCES: usize = 16;
const MAX_STOP_SEQUENCE_BYTES: usize = 1_024;
const MAX_STOP_SEQUENCES_BYTES: usize = 4 * 1_024;
const MAX_CLIENT_IDENTIFIER_BYTES: usize = 256;
const MAX_HTTP_CONNECTIONS: usize = 1_024;
const MAX_HTTP2_STREAMS_PER_CONNECTION: u32 = 128;
const HTTP1_HEADER_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const HTTP_CONNECTION_MAX_LIFETIME: std::time::Duration = std::time::Duration::from_secs(300);

fn body_admission_semaphore() -> Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    SEMAPHORE
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_BODY_ADMISSIONS)))
        .clone()
}

fn small_body_admission_semaphore() -> Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    SEMAPHORE
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(MAX_SMALL_BODY_ADMISSIONS)))
        .clone()
}

#[derive(Clone)]
pub(crate) struct BodyAdmissionGuard {
    permit: Arc<Mutex<Option<tokio::sync::OwnedSemaphorePermit>>>,
}

impl BodyAdmissionGuard {
    fn new(permit: tokio::sync::OwnedSemaphorePermit) -> Self {
        Self {
            permit: Arc::new(Mutex::new(Some(permit))),
        }
    }

    pub(crate) fn release(&self) {
        if let Ok(mut permit) = self.permit.lock() {
            permit.take();
        }
    }
}

pub(crate) struct BodyAdmissionLease(Option<BodyAdmissionGuard>);

impl BodyAdmissionLease {
    fn release(&mut self) {
        if let Some(admission) = self.0.take() {
            admission.release();
        }
    }

    pub(crate) fn guard(&self) -> Option<&BodyAdmissionGuard> {
        self.0.as_ref()
    }
}

impl Drop for BodyAdmissionLease {
    fn drop(&mut self) {
        self.release();
    }
}

pub(crate) struct AdmittedJson<T>(pub(crate) T, pub(crate) BodyAdmissionLease);

#[axum::async_trait]
impl<S, T> FromRequest<S> for AdmittedJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = axum::extract::rejection::JsonRejection;

    async fn from_request(req: AxumRequest, state: &S) -> Result<Self, Self::Rejection> {
        let admission = req.extensions().get::<BodyAdmissionGuard>().cloned();
        let parsed = Json::<T>::from_request(req, state).await;
        parsed.map(|Json(value)| Self(value, BodyAdmissionLease(admission)))
    }
}

fn declared_body_length(req: &AxumRequest) -> Option<usize> {
    req.headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

fn body_requires_admission(req: &AxumRequest) -> bool {
    // A transfer-encoding header means the wire length is not bounded by Content-Length (and a
    // conflicting pair must take the conservative path), so chunked/unknown bodies never bypass
    // the large-upload gate.
    if req.headers().contains_key(TRANSFER_ENCODING) {
        return true;
    }
    declared_body_length(req).is_none_or(|length| length > BODY_ADMISSION_BYPASS_BYTES)
}

/// Keep the body parser bounded without making the documented 192 MiB envelope require an
/// implausibly fast uplink. The base is still a strict deadline for unknown-length bodies; a
/// declared length earns a pessimistic 2 MiB/s transfer budget, capped at three minutes.
fn body_read_timeout(req: &AxumRequest) -> std::time::Duration {
    let Some(length) = declared_body_length(req) else {
        return BODY_READ_TIMEOUT;
    };
    let bytes = length as u64;
    let extra_seconds =
        bytes.saturating_add(BODY_READ_RATE_BYTES_PER_SEC - 1) / BODY_READ_RATE_BYTES_PER_SEC;
    let seconds = BODY_READ_TIMEOUT
        .as_secs()
        .saturating_add(extra_seconds)
        .min(BODY_READ_TIMEOUT_MAX.as_secs());
    std::time::Duration::from_secs(seconds)
}

/// Reshape the extractor-produced 413 (a plain-text axum rejection) into the standard
/// OpenAI error object every SDK parses. Runs OUTSIDE the routes so both the
/// content-length refusal and the mid-read stream cutoff surface identically: a clean
/// HTTP 413 with our JSON shape — never a hang, never a bare connection reset.
async fn shape_payload_too_large(req: AxumRequest, next: Next) -> Response {
    let resp = next.run(req).await;
    if resp.status() != StatusCode::PAYLOAD_TOO_LARGE {
        return resp;
    }
    error_response_coded(
        StatusCode::PAYLOAD_TOO_LARGE,
        &format!(
            "request body exceeds the {} MiB limit",
            MAX_BODY_BYTES / (1024 * 1024)
        ),
        "invalid_request_error",
        None,
        Some("request_too_large"),
    )
}

/// The one place the body-size policy is applied (tested directly in `body_limit_tests`;
/// `main` wires the app router through here).
fn apply_body_limit(app: Router) -> Router {
    app.layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(middleware::from_fn(shape_payload_too_large))
}

fn protected_inference_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/auth/check"
            | "/v1/completions"
            | "/v1/chat/completions"
            | "/v1/messages"
            | "/v1/responses"
            | "/v1/embeddings"
            | "/v1/rerank"
    )
}

/// Give middleware refusals the same request-id and body contract as the handler they
/// replace. In particular, `/v1/messages` must carry the Anthropic body plus both request-id
/// header spellings even when the body has not been read yet.
async fn shape_inference_early_response(path: &str, response: Response) -> Response {
    let request_id = Envelope::new(path != "/v1/completions");
    if path == "/v1/messages" {
        anthropic::with_anthropic_request_id(
            &request_id.id,
            anthropic::reshape_error(response, &request_id.id).await,
        )
    } else {
        with_request_id(&request_id.id, response)
    }
}

/// Authenticate inference requests from headers before any route extractor is allowed to poll
/// the body. This covers every tenant-authenticated inference surface; catalog, health, metrics,
/// and admin policies have distinct public/auth contracts. The route handlers retain their own
/// authentication checks for defense in depth and for dialect-specific error shaping.
async fn authenticate_inference_before_body(
    State(st): State<AppState>,
    mut req: AxumRequest,
    next: Next,
) -> Response {
    if !protected_inference_path(req.uri().path()) {
        return next.run(req).await;
    }
    let path = req.uri().path().to_string();
    // Reject an advertised oversize before touching either admission pool. Otherwise a caller
    // could fill the pool's active slots and waiter queue with requests that the inner extractor
    // would reject as 413 anyway.
    if declared_body_length(&req).is_some_and(|length| length > MAX_BODY_BYTES) {
        return shape_inference_early_response(
            &path,
            error_response_coded(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!(
                    "request body exceeds the {} MiB limit",
                    MAX_BODY_BYTES / (1024 * 1024)
                ),
                "invalid_request_error",
                None,
                Some("request_too_large"),
            ),
        )
        .await;
    }
    let headers = req.headers();
    let bearer = bearer_token(headers);
    let auth = if matches!(path.as_str(), "/v1/messages" | "/v1/auth/check") {
        let api_key = headers
            .get("x-api-key")
            .and_then(|value| value.to_str().ok());
        surfaces::authenticate_candidates(&st.api_auth, &[bearer, api_key])
    } else {
        surfaces::authenticate_candidates(&st.api_auth, &[bearer])
    };
    if let Err(why) = auth {
        return shape_inference_early_response(&path, authentication_error(why)).await;
    }
    // Keep the large, authenticated body parser itself bounded. The route-level request slot is
    // intentionally acquired after JSON/vision validation so ordinary 400s do not consume it;
    // this separate permit prevents a low-cap key from queueing unbounded 192 MiB parses before
    // that later gate while retaining the advertised body ceiling and 413 contract. Small,
    // explicitly sized bodies use a separate finite pool so a slow large upload cannot head-of-
    // line block ordinary requests, while neither class can create unbounded parser tasks.
    // Acquisition is deliberately fail-fast; Tokio's async waiter queue is not a resource bound.
    let body_deadline = tokio::time::Instant::now() + body_read_timeout(&req);
    let body_admission = if body_requires_admission(&req) {
        body_admission_semaphore()
    } else {
        small_body_admission_semaphore()
    };
    let body_permit = match body_admission.try_acquire_owned() {
        Ok(permit) => permit,
        Err(tokio::sync::TryAcquireError::Closed) => {
            let response = retry_contract_response(
                error_response_coded(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "request body admission is unavailable",
                    "server_error",
                    None,
                    Some("body_admission_unavailable"),
                ),
                Some(BODY_ADMISSION_RETRY_AFTER_S),
            );
            return shape_inference_early_response(&path, response).await;
        }
        Err(tokio::sync::TryAcquireError::NoPermits) => {
            let response = retry_contract_response(
                error_response_coded(
                    StatusCode::TOO_MANY_REQUESTS,
                    "request body admission is busy",
                    "rate_limit_error",
                    None,
                    Some("body_admission_busy"),
                ),
                Some(BODY_ADMISSION_RETRY_AFTER_S),
            );
            return shape_inference_early_response(&path, response).await;
        }
    };
    // Typed handlers retain this shared guard through semantic traversal, prompt construction,
    // tokenization, and request-slot admission, then release it before any generation wait. Raw
    // translation surfaces do the same through their shared admission path. The middleware keeps
    // a fallback clone so extractor rejection and non-body routes cannot leak a permit.
    let body_admission_guard = BodyAdmissionGuard::new(body_permit);
    req.extensions_mut().insert(body_admission_guard.clone());
    let body = std::mem::replace(req.body_mut(), Body::empty());
    let mut body = Box::pin(body.into_data_stream());
    let body_timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let body_timed_out_flag = body_timed_out.clone();
    let guarded_body = async_stream::stream! {
        loop {
            let remaining = body_deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                body_timed_out_flag.store(true, std::sync::atomic::Ordering::Release);
                yield Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "request body read deadline exceeded",
                ));
                break;
            }
            let poll = std::future::poll_fn(|cx| body.as_mut().poll_next(cx));
            let frame = match tokio::time::timeout(BODY_IDLE_TIMEOUT.min(remaining), poll).await {
                Ok(frame) => frame,
                Err(_) => {
                    body_timed_out_flag.store(true, std::sync::atomic::Ordering::Release);
                    yield Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "request body idle timeout exceeded",
                    ));
                    break;
                }
            };
            match frame {
                Some(Ok(bytes)) => yield Ok(bytes),
                Some(Err(error)) => {
                    yield Err(std::io::Error::other(error.to_string()));
                    break;
                }
                None => break,
            }
        }
    };
    *req.body_mut() = Body::from_stream(guarded_body);
    let response = next.run(req).await;
    body_admission_guard.release();
    if body_timed_out.load(std::sync::atomic::Ordering::Acquire) {
        let request_id = Envelope::new(path != "/v1/completions");
        let timeout = error_response_coded(
            StatusCode::REQUEST_TIMEOUT,
            "request body read timed out",
            "invalid_request_error",
            None,
            Some("request_body_timeout"),
        );
        return if path == "/v1/messages" {
            anthropic::with_anthropic_request_id(
                &request_id.id,
                anthropic::reshape_error(timeout, &request_id.id).await,
            )
        } else {
            with_request_id(&request_id.id, timeout)
        };
    }
    if path == "/v1/messages" && response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let request_id = Envelope::new(true);
        return anthropic::with_anthropic_request_id(
            &request_id.id,
            anthropic::reshape_error(response, &request_id.id).await,
        );
    }
    response
}

#[cfg(test)]
mod body_limit_tests {
    use super::*;

    static BODY_ADMISSION_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// A router with the REAL body policy (`apply_body_limit`, the exact helper `main`
    /// wires) over both extractor shapes the inference routes use: `Json` (completions /
    /// chat) and raw `Bytes` (`/v1/messages`).
    fn test_app() -> Router {
        let app = Router::new()
            .route(
                "/bytes",
                post(|b: axum::body::Bytes| async move { b.len().to_string() }),
            )
            .route(
                "/json",
                post(
                    |AdmittedJson(v, _admission): AdmittedJson<serde_json::Value>| async move {
                        v["pad"].as_str().unwrap_or("").len().to_string()
                    },
                ),
            );
        apply_body_limit(app)
    }

    fn streamed_body(chunks: usize) -> Body {
        // one shared 1 MiB chunk, cloned (Bytes clones are refcounted — no O(n) alloc);
        // streaming means NO Content-Length, exercising the mid-read cutoff path.
        let chunk = axum::body::Bytes::from(vec![b'x'; 1024 * 1024]);
        Body::from_stream(async_stream::stream! {
            for _ in 0..chunks {
                yield Ok::<_, std::io::Error>(chunk.clone());
            }
        })
    }

    #[tokio::test]
    async fn bodies_past_the_old_2mib_default_are_accepted() {
        // 3 MiB — over axum's 2 MiB default that silently capped the advertised
        // 262k-token + vision surface, comfortably under MAX_BODY_BYTES.
        for (path, body) in [
            ("/bytes", Body::from(vec![b'x'; 3 * 1024 * 1024])),
            (
                "/json",
                Body::from(
                    serde_json::to_vec(&json!({ "pad": "x".repeat(3 * 1024 * 1024) })).unwrap(),
                ),
            ),
        ] {
            let resp = test_app()
                .oneshot(
                    axum::http::Request::post(path)
                        .header(CONTENT_TYPE, "application/json")
                        .body(body)
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK, "{path}");
        }
    }

    #[tokio::test]
    async fn body_at_exactly_the_limit_is_accepted() {
        let resp = test_app()
            .oneshot(
                axum::http::Request::post("/bytes")
                    .body(streamed_body(MAX_BODY_BYTES / (1024 * 1024)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), MAX_BODY_BYTES.to_string().as_bytes());
    }

    #[tokio::test]
    async fn oversize_body_is_a_clean_413_in_our_error_shape() {
        // one chunk past the ceiling; both extractor shapes must answer the SAME way —
        // an HTTP 413 carrying the standard OpenAI error object (never axum's bare-text
        // rejection, never a hang or reset).
        for path in ["/bytes", "/json"] {
            let resp = test_app()
                .oneshot(
                    axum::http::Request::post(path)
                        .header(CONTENT_TYPE, "application/json")
                        .body(streamed_body(MAX_BODY_BYTES / (1024 * 1024) + 1))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE, "{path}");
            assert_eq!(
                resp.headers().get("x-should-retry").map(|v| v.as_bytes()),
                Some(b"false".as_ref()),
                "{path}: retrying identical bytes cannot fix a 413"
            );
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON error shape");
            assert_eq!(v["error"]["type"], "invalid_request_error", "{path}");
            assert_eq!(v["error"]["code"], "request_too_large", "{path}");
            assert!(
                v["error"]["message"].as_str().unwrap().contains("192 MiB"),
                "{path}: message names the limit"
            );
        }
    }

    #[tokio::test]
    async fn authenticated_body_admission_is_finite() {
        let _test_lock = BODY_ADMISSION_TEST_LOCK.lock().await;
        let semaphore = body_admission_semaphore();
        let mut permits = Vec::new();
        for _ in 0..MAX_BODY_ADMISSIONS {
            permits.push(semaphore.clone().acquire_owned().await.unwrap());
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), semaphore.acquire())
                .await
                .is_err(),
            "body parser admission must not be unbounded"
        );
        drop(permits);
        assert!(semaphore.acquire().await.is_ok());
    }

    #[tokio::test]
    async fn small_body_admission_is_finite_and_separate() {
        let _test_lock = BODY_ADMISSION_TEST_LOCK.lock().await;
        let large = body_admission_semaphore();
        let small = small_body_admission_semaphore();
        let mut small_permits = Vec::new();
        for _ in 0..MAX_SMALL_BODY_ADMISSIONS {
            small_permits.push(small.clone().acquire_owned().await.unwrap());
        }
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), small.acquire())
                .await
                .is_err(),
            "small body parser admission must be bounded"
        );
        assert!(
            large.clone().try_acquire().is_ok(),
            "small uploads must not consume large-upload permits"
        );
        drop(small_permits);
        assert!(small.acquire().await.is_ok());
    }

    #[test]
    fn small_declared_bodies_bypass_large_upload_admission() {
        let request = axum::http::Request::post("/v1/chat/completions")
            .header(CONTENT_LENGTH, "2048")
            .body(Body::empty())
            .unwrap();
        assert!(!body_requires_admission(&request));

        let request = axum::http::Request::post("/v1/chat/completions")
            .header(
                CONTENT_LENGTH,
                (BODY_ADMISSION_BYPASS_BYTES + 1).to_string(),
            )
            .body(Body::empty())
            .unwrap();
        assert!(body_requires_admission(&request));

        let request = axum::http::Request::post("/v1/chat/completions")
            .header(CONTENT_LENGTH, "2048")
            .header(TRANSFER_ENCODING, "chunked")
            .body(Body::empty())
            .unwrap();
        assert!(body_requires_admission(&request));
    }

    #[test]
    fn declared_body_timeout_scales_with_upload_size_and_has_a_cap() {
        let unknown = axum::http::Request::post("/v1/chat/completions")
            .body(Body::empty())
            .unwrap();
        assert_eq!(body_read_timeout(&unknown), BODY_READ_TIMEOUT);

        let large = axum::http::Request::post("/v1/chat/completions")
            .header(CONTENT_LENGTH, MAX_BODY_BYTES.to_string())
            .body(Body::empty())
            .unwrap();
        assert!(body_read_timeout(&large) > BODY_READ_TIMEOUT);
        assert_eq!(body_read_timeout(&large), BODY_READ_TIMEOUT_MAX);

        let absurd = axum::http::Request::post("/v1/chat/completions")
            .header(CONTENT_LENGTH, u64::MAX.to_string())
            .body(Body::empty())
            .unwrap();
        assert_eq!(body_read_timeout(&absurd), BODY_READ_TIMEOUT_MAX);
    }

    #[tokio::test]
    async fn early_body_refusals_keep_dialect_ids_and_retry_contracts() {
        let too_large = shape_inference_early_response(
            "/v1/messages",
            error_response_coded(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request body exceeds the 192 MiB limit",
                "invalid_request_error",
                None,
                Some("request_too_large"),
            ),
        )
        .await;
        assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let house_id = too_large.headers()["x-request-id"].clone();
        assert_eq!(too_large.headers()["request-id"], house_id);
        assert_eq!(too_large.headers()["x-should-retry"], "false");
        let body = axum::body::to_bytes(too_large.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["type"], "error");
        assert_eq!(payload["request_id"], house_id.to_str().unwrap());

        let busy = shape_inference_early_response(
            "/v1/chat/completions",
            retry_contract_response(
                error_response_coded(
                    StatusCode::TOO_MANY_REQUESTS,
                    "request body admission is busy",
                    "rate_limit_error",
                    None,
                    Some("body_admission_busy"),
                ),
                Some(BODY_ADMISSION_RETRY_AFTER_S),
            ),
        )
        .await;
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(!busy.headers()["x-request-id"].is_empty());
        assert_eq!(busy.headers()["retry-after"], "1");
        assert_eq!(busy.headers()["retry-after-ms"], "1000");
        assert!(busy.headers().get("x-should-retry").is_none());
        let body = axum::body::to_bytes(busy.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["error"]["code"], "body_admission_busy");
    }

    #[tokio::test]
    async fn vision_preprocess_admission_is_fail_fast_and_retryable() {
        let semaphore = Box::leak(Box::new(tokio::sync::Semaphore::new(1)));
        let held = semaphore.try_acquire().unwrap();
        let busy = try_vision_preprocess_with(true, semaphore).unwrap_err();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(busy.headers()["retry-after"], "1");
        drop(held);
        assert!(
            try_vision_preprocess_with(true, semaphore)
                .unwrap()
                .is_some()
        );
        assert!(
            try_vision_preprocess_with(false, semaphore)
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn typed_json_retains_body_admission_until_handler_validation_releases_it() {
        #[derive(Clone)]
        struct Signals {
            parsed: Arc<tokio::sync::Notify>,
            finish: Arc<tokio::sync::Notify>,
            semaphore: Arc<tokio::sync::Semaphore>,
        }

        let semaphore = Arc::new(tokio::sync::Semaphore::new(1));
        let guard = BodyAdmissionGuard::new(semaphore.clone().try_acquire_owned().unwrap());
        let signals = Signals {
            parsed: Arc::new(tokio::sync::Notify::new()),
            finish: Arc::new(tokio::sync::Notify::new()),
            semaphore: semaphore.clone(),
        };
        let app = Router::new()
            .route(
                "/",
                post(
                    |Extension(signals): Extension<Signals>,
                     AdmittedJson(_, mut admission): AdmittedJson<serde_json::Value>| async move {
                        assert_eq!(
                            signals.semaphore.available_permits(),
                            0,
                            "typed deserialization alone must not release post-parse admission"
                        );
                        admission.release();
                        assert_eq!(signals.semaphore.available_permits(), 1);
                        signals.parsed.notify_one();
                        signals.finish.notified().await;
                        "ok"
                    },
                ),
            )
            .layer(Extension(signals.clone()))
            .layer(Extension(guard));
        let response = tokio::spawn(
            app.oneshot(
                axum::http::Request::post("/")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"value":1}"#))
                    .unwrap(),
            ),
        );
        signals.parsed.notified().await;
        assert_eq!(
            semaphore.available_permits(),
            1,
            "validated work must release admission before generation waits"
        );
        signals.finish.notify_one();
        assert_eq!(response.await.unwrap().unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn transport_closes_stalled_headers_and_caps_connections() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve_bounded_http_with_limits(
            listener,
            Router::new().route("/", get(|| async { "ok" })).route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(std::time::Duration::from_millis(140)).await;
                    "slow-ok"
                }),
            ),
            async move {
                let _ = shutdown_rx.await;
            },
            std::time::Duration::from_millis(30),
            1,
            std::time::Duration::from_millis(80),
        ));

        let mut stalled = tokio::net::TcpStream::connect(address).await.unwrap();
        stalled.write_all(b"GET / HT").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let mut excess = tokio::net::TcpStream::connect(address).await.unwrap();
        let mut bytes = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_millis(250),
            excess.read_to_end(&mut bytes),
        )
        .await
        .expect("connection beyond the cap must be closed promptly")
        .unwrap();

        bytes.clear();
        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            stalled.read_to_end(&mut bytes),
        )
        .await
        .expect("stalled request headers must hit the configured deadline")
        .unwrap();

        let mut idle = tokio::net::TcpStream::connect(address).await.unwrap();
        idle.write_all(b"GET / HTTP/1.1\r\nHost: local\r\n\r\n")
            .await
            .unwrap();
        bytes.clear();
        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            idle.read_to_end(&mut bytes),
        )
        .await
        .expect("an idle keep-alive connection must hit the maximum lifetime")
        .unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("200 OK"));

        let mut active = tokio::net::TcpStream::connect(address).await.unwrap();
        active
            .write_all(b"GET /slow HTTP/1.1\r\nHost: local\r\n\r\n")
            .await
            .unwrap();
        bytes.clear();
        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            active.read_to_end(&mut bytes),
        )
        .await
        .expect("an active response must finish across the connection age boundary")
        .unwrap();
        let active_response = String::from_utf8_lossy(&bytes);
        assert!(active_response.contains("200 OK"), "{active_response}");
        assert!(active_response.contains("slow-ok"), "{active_response}");

        // HTTP/2 keepalive constructs its timer during the handshake. If the H2 builder
        // does not receive a TokioTimer, hyper panics in the connection task and the
        // response future sees a dropped connection instead of this 200.
        let h2_stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let (mut h2_client, h2_connection) = h2::client::handshake(h2_stream).await.unwrap();
        let h2_driver = tokio::spawn(h2_connection);
        let request = axum::http::Request::builder()
            .uri(format!("http://{address}/"))
            .body(())
            .unwrap();
        let (response, _) = h2_client.send_request(request, true).unwrap();
        let response = tokio::time::timeout(std::time::Duration::from_millis(500), response)
            .await
            .expect("HTTP/2 handshake and response must complete")
            .expect("HTTP/2 connection must stay alive through the response");
        assert_eq!(response.status(), StatusCode::OK);
        drop(h2_client);
        h2_driver.abort();
        let _ = h2_driver.await;

        let _ = shutdown_tx.send(());
        server.await.unwrap().unwrap();
    }
}

#[derive(Clone, Default)]
struct TtftRequestTrace(Option<Arc<ttft::Trace>>);

fn is_sse_data_frame(bytes: &[u8]) -> bool {
    bytes
        .windows(b"data:".len())
        .any(|window| window == b"data:")
}

async fn ttft_request_start(mut req: AxumRequest, next: Next) -> Response {
    let trace = ttft::start(req.uri().path());
    req.extensions_mut().insert(TtftRequestTrace(trace.clone()));
    let response = next.run(req).await;
    let Some(trace) = trace else {
        return response;
    };
    let is_sse = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if !is_sse {
        return response;
    }

    // Stamp the first serialized application data frame as Hyper polls it. Axum's
    // keepalive comments can precede a long prefill, so non-data frames do not count.
    let (parts, body) = response.into_parts();
    let mut body = Box::pin(body.into_data_stream());
    let stream = async_stream::stream! {
        while let Some(frame) =
            std::future::poll_fn(|cx| body.as_mut().poll_next(cx)).await
        {
            if frame
                .as_ref()
                .is_ok_and(|bytes| is_sse_data_frame(bytes))
            {
                trace.mark_first_sse_byte();
            }
            yield frame;
        }
    };
    Response::from_parts(parts, Body::from_stream(stream))
}

const OPENROUTER_SCHEMA_VERSION: &str = "2.4";
const JSON_SAFE_INTEGER_MAX: u64 = 9_007_199_254_740_991;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenRouterMetadataFile {
    #[serde(default)]
    models: HashMap<String, OpenRouterModelMetadata>,
    /// Machine-validated future offers. These never enter a model feed or request path until the
    /// operator moves the entry into `models` and loads the same alias through `MEMRA_MODELS`.
    #[serde(default)]
    planned_models: HashMap<String, OpenRouterModelMetadata>,
    /// Router-marketplace provider identity (TrustedRouter contract v2). Rendered at the top
    /// of /v1/models next to the server-truth error contract; absent = no provider block.
    #[serde(default)]
    provider: Option<ProviderMetadata>,
}

/// Operator-declared provider identity for the /v1/models contract-v2 header. Everything a
/// router needs to route AROUND us (status page, contacts, regions) is declared here; the
/// error contract itself (429/503/Retry-After/quota code) is server truth and not configurable.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderMetadata {
    id: String,
    #[serde(default)]
    status_url: Option<String>,
    #[serde(default)]
    support_contact: Option<String>,
    #[serde(default)]
    incident_contact: Option<String>,
    #[serde(default)]
    regions: Vec<String>,
}

/// Contract-v2 lifecycle block (RFC 3339 timestamps). A model without one is "active".
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct LifecycleMetadata {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    deprecation_at: Option<String>,
    #[serde(default)]
    retirement_at: Option<String>,
    #[serde(default)]
    replacement_model_id: Option<String>,
}

/// Contract-v2 reliability block: how long a router should wait before failing over.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReliabilityMetadata {
    #[serde(default)]
    first_token_timeout_seconds: Option<u64>,
    #[serde(default)]
    completion_timeout_seconds: Option<u64>,
    #[serde(default)]
    stream_idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    capacity_scope: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenRouterModelMetadata {
    /// Contract-v2 per-model blocks (see the ProviderMetadata docs above).
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    lifecycle: Option<LifecycleMetadata>,
    #[serde(default)]
    reliability: Option<ReliabilityMetadata>,
    #[serde(default)]
    hugging_face_id: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    quantization: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    max_prompt_length: Option<u64>,
    #[serde(default)]
    max_output_length: Option<u64>,
    /// Request default when max_tokens is omitted. Keeping this separate from the provider maximum
    /// prevents an advertised 262k ceiling from reserving a 262k KV cache for every ordinary call.
    #[serde(default)]
    default_output_length: Option<u64>,
    #[serde(default)]
    pricing: OpenRouterPricing,
    #[serde(default)]
    capacity: OpenRouterCapacity,
    #[serde(default)]
    is_ready: Option<bool>,
    #[serde(default)]
    is_free: Option<bool>,
    #[serde(default)]
    discount_to_user: Option<f64>,
    #[serde(default)]
    openrouter_slug: Option<String>,
    #[serde(default)]
    datacenters: Vec<OpenRouterDatacenter>,
    /// Extra INPUT modalities beyond the implicit "text" (lane/vision: ["image"]).
    /// Each renders as its own input-modality object in the feed; image tokens bill
    /// at the prompt token price (pads are ordinary prompt tokens).
    #[serde(default)]
    input_modalities: Vec<String>,
    /// Which API surface this model actually serves: "chat" (default), "embedding",
    /// or "rerank". This is a PUBLISHED CONTRACT, not a hint — the catalog row a
    /// client SDK reads is built from it, so it is declared rather than inferred.
    ///
    /// It exists because the row used to be a hardcoded `"type": "chat"` with
    /// `endpoints: ["chat/completions"]` for every registered model. On 2026-08-28
    /// that advertised qwen3-embedding-8b and qwen3-reranker-8b as chat models with
    /// `tools: true`, `streaming: true` and no mention of /v1/embeddings or
    /// /v1/rerank — the two surfaces they actually serve. A client that believed
    /// the catalog would call the wrong endpoint with the wrong body shape.
    ///
    /// Embedding/rerank capability is decided at RUNTIME (does the prime path yield
    /// hidden state), which cannot be read at catalog-build time; the contract we
    /// publish must therefore be stated by the deployment, not guessed.
    #[serde(default)]
    surface: Option<String>,
    #[serde(default)]
    zdr: Option<bool>,
    #[serde(default)]
    hipaa: Option<bool>,
    /// SERVING-DEPLOYMENT default for the OpenAI `reasoning_effort` field when a chat
    /// request leaves reasoning UNSET (owner ruling 2026-08-19: gemma-4 serves think-ON
    /// by default — think-on scored 80.81 GPQA vs 76.26 think-off on the served mint;
    /// qwen's template already defaults ON without any knob). Applied by `parse_think`
    /// exactly as if the client had sent this value, so the rendered prompt is
    /// byte-identical to the explicit request. Explicit client reasoning
    /// (`reasoning_effort`, `reasoning.effort`, `reasoning.enabled`) always wins; the
    /// template's own vendor-law rendering semantics are untouched — this only moves
    /// which ThinkMode an unset request resolves to for THIS deployment.
    #[serde(default)]
    default_reasoning_effort: Option<String>,
    /// VENDOR-RECOMMENDED SAMPLING for requests that expressed NOTHING (owner ruling
    /// 2026-08-19: "we don't have to serve greedy, we measure greedy but we serve what the
    /// user chooses" / "we default to what are the recommendations" / "greedy can create
    /// issues"). Each key substitutes for exactly one omitted sampling field, on EVERY
    /// surface (`/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`)
    /// through the single `resolve_sampler_config` law. An explicit client value always
    /// wins — including an explicit `temperature: 0`, which still produces true greedy.
    ///
    /// The value belongs to the MODEL VENDOR, not to us: put the citation in the TOML
    /// comment next to it so nobody later "cleans up" a deliberate number. Boot-validated
    /// (see `validate_openrouter_metadata`): a typo'd default must fail before GPU load,
    /// never become a per-request 400 storm under the watchdog.
    ///
    /// `default_temperature` REFUSES 0.0 on purpose. A zero here would reinstate exactly the
    /// greedy-by-default hazard this key exists to remove — silently, deployment-wide, for
    /// every omitting client. Greedy stays reachable the honest way: the client sends
    /// `temperature: 0`.
    #[serde(default)]
    default_temperature: Option<f32>,
    #[serde(default)]
    default_top_p: Option<f32>,
    /// 0 = disabled (keep all) — the same convention the request field uses.
    #[serde(default)]
    default_top_k: Option<usize>,
    #[serde(default)]
    default_min_p: Option<f32>,
    #[serde(default)]
    default_presence_penalty: Option<f32>,
    #[serde(default)]
    default_frequency_penalty: Option<f32>,
    /// OpenRouter/HF-convention multiplicative penalty; 1.0 = off.
    #[serde(default)]
    default_repetition_penalty: Option<f32>,
    /// SECOND VENDOR SAMPLING ARM for the model's NON-THINKING mode (owner ruling
    /// 2026-08-24: "do what is correct" — served models default to the VENDOR's
    /// recommendation, and some vendors publish TWO recommendations, one per thinking
    /// mode; qwen3.8's card gives thinking 1.0/0.95/20 and non-thinking 0.7/0.80/20 +
    /// presence_penalty 1.5). The flat `default_*` keys above stay the PRIMARY arm —
    /// what every request got before this table existed — and this table, when
    /// declared, is what a request whose RESOLVED thinking mode is OFF gets for the
    /// sampling fields it left unset (`ModelSamplingDefaults::for_mode`). Off is the
    /// resolved `ThinkMode::NoThink`, whichever spelling produced it: `reasoning_effort:
    /// "none"|"minimal"`, `enable_thinking:false`, `chat_template_kwargs.
    /// enable_thinking:false`, `reasoning:{enabled:false}`, `include_reasoning:false`,
    /// Anthropic `thinking.type:"disabled"`, or an operator `default_reasoning_effort =
    /// "none"` resolving an unset request. An explicit client value is NEVER overridden
    /// by either arm, and an explicit `temperature: 0` still produces true greedy.
    ///
    /// A model WITHOUT this table is byte-identical to before it existed: one arm,
    /// every mode. Same boot-validation posture and ranges as the flat keys (a typo'd
    /// arm fails before GPU load), and an EMPTY declared table is refused — declaring
    /// the arm and recommending nothing would silently hand thinking-off traffic the
    /// bare API-standard defaults while looking configured.
    #[serde(default)]
    non_thinking_sampling: Option<SamplingArmMetadata>,
}

/// One declared sampling arm (`non_thinking_sampling`): the same seven vendor keys as the
/// flat `default_*` set, unprefixed because the table name already says which arm they
/// belong to. `None` = the vendor recommends nothing for that field in this mode — it
/// falls through to the API-standard default, never to the other arm (arms are separate
/// vendor programs; blending them would serve numbers no vendor published).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingArmMetadata {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
}

impl SamplingArmMetadata {
    fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.min_p.is_none()
            && self.presence_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.repetition_penalty.is_none()
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    cached_prompt: Option<String>,
    #[serde(default)]
    cache_write: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    internal_reasoning: Option<String>,
    #[serde(default)]
    request: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenRouterCapacity {
    #[serde(default)]
    prompt_tpm: Option<u64>,
    #[serde(default)]
    cached_prompt_tpm: Option<u64>,
    #[serde(default)]
    completion_tpm: Option<u64>,
    #[serde(default)]
    request_rpm: Option<u64>,
    #[serde(default)]
    concurrency: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct OpenRouterDatacenter {
    country_code: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    region: Option<String>,
}

impl OpenRouterMetadataFile {
    fn parse(
        text: &str,
    ) -> Result<
        (
            HashMap<String, OpenRouterModelMetadata>,
            Option<ProviderMetadata>,
        ),
        String,
    > {
        let file: Self =
            toml::from_str(text).map_err(|e| format!("models metadata TOML parse: {e}"))?;
        for (alias, metadata) in &file.models {
            validate_openrouter_metadata(alias, metadata)?;
        }
        for (alias, metadata) in &file.planned_models {
            validate_openrouter_metadata(alias, metadata)?;
            if file.models.contains_key(alias) {
                return Err(format!(
                    "model alias {alias:?} appears in both models and planned_models"
                ));
            }
        }
        if let Some(provider) = &file.provider {
            if provider.id.is_empty() {
                return Err("provider.id must be a non-empty slug".into());
            }
            // The contract wants URIs, not bare addresses: mailto:ops@example.com or https://…
            for (field, value) in [
                ("provider.support_contact", &provider.support_contact),
                ("provider.incident_contact", &provider.incident_contact),
            ] {
                if let Some(value) = value
                    && !value.contains(':')
                {
                    return Err(format!(
                        "{field} must be a URI (mailto:… or https://…), got {value:?}"
                    ));
                }
            }
        }
        Ok((file.models, file.provider))
    }

    #[cfg(test)]
    fn from_toml(text: &str) -> Result<HashMap<String, OpenRouterModelMetadata>, String> {
        Self::parse(text).map(|(models, _)| models)
    }
}

/// Decimal-shift a per-token USD price string six places left (the per-1M-token price)
/// without floating point: "0.00000038" -> "0.38", "0.0000026" -> "2.60". Keeps at least
/// two fraction digits — the router contract's examples are "0.50"-style strings.
fn per_million_price(per_token: &str) -> Option<String> {
    if !valid_price_string(per_token) {
        return None;
    }
    let (whole, frac) = match per_token.split_once('.') {
        Some((whole, frac)) => (whole, frac),
        None => (per_token, ""),
    };
    let mut digits = format!("{whole}{frac}");
    let point = whole.len() + 6;
    while digits.len() < point {
        digits.push('0');
    }
    let (int_part, frac_part) = digits.split_at(point);
    let int_part = int_part.trim_start_matches('0');
    let int_part = if int_part.is_empty() { "0" } else { int_part };
    let mut frac_out = frac_part.trim_end_matches('0').to_string();
    while frac_out.len() < 2 {
        frac_out.push('0');
    }
    Some(format!("{int_part}.{frac_out}"))
}

fn valid_price_string(value: &str) -> bool {
    let mut parts = value.split('.');
    let whole = parts.next().unwrap_or_default();
    let fraction = parts.next();
    !whole.is_empty()
        && whole.bytes().all(|b| b.is_ascii_digit())
        && fraction.is_none_or(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        && parts.next().is_none()
}

fn validate_openrouter_metadata(
    alias: &str,
    metadata: &OpenRouterModelMetadata,
) -> Result<(), String> {
    if alias.is_empty() {
        return Err("models metadata contains an empty model alias".into());
    }
    // Fail at BOOT, not per-request: a typo'd default must never turn into a 400 storm
    // (or a silent no-op) after the box restarts under the watchdog.
    if let Some(effort) = metadata.default_reasoning_effort.as_deref()
        && !matches!(effort, "none" | "minimal" | "low" | "medium" | "high")
    {
        return Err(format!(
            "model {alias:?}: default_reasoning_effort {effort:?} is not a \
             reasoning_effort level (none|minimal|low|medium|high)"
        ));
    }
    validate_sampling_defaults(alias, metadata)?;
    for m in &metadata.input_modalities {
        if m != "image" && m != "video" {
            return Err(format!(
                "model {alias:?}: input_modalities entry {m:?} not served (image/video)"
            ));
        }
    }
    if let Some(sfc) = metadata.surface.as_deref()
        && !matches!(sfc, "chat" | "embedding" | "rerank")
    {
        return Err(format!(
            "model {alias:?}: surface {sfc:?} is not a served surface (chat|embedding|rerank)"
        ));
    }
    if let Some(q) = metadata.quantization.as_deref()
        && !matches!(
            q,
            "int4"
                | "int8"
                | "fp4"
                | "mxfp4"
                | "nvfp4"
                | "fp6"
                | "fp8"
                | "mxfp8"
                | "fp16"
                | "bf16"
                | "fp32"
        )
    {
        return Err(format!(
            "model {alias:?}: quantization {q:?} is not in the OpenRouter schema 2.4 enum"
        ));
    }
    for (field, value) in [
        ("pricing.prompt", metadata.pricing.prompt.as_deref()),
        (
            "pricing.cached_prompt",
            metadata.pricing.cached_prompt.as_deref(),
        ),
        (
            "pricing.cache_write",
            metadata.pricing.cache_write.as_deref(),
        ),
        ("pricing.completion", metadata.pricing.completion.as_deref()),
        (
            "pricing.internal_reasoning",
            metadata.pricing.internal_reasoning.as_deref(),
        ),
        ("pricing.request", metadata.pricing.request.as_deref()),
    ] {
        if let Some(value) = value
            && !valid_price_string(value)
        {
            return Err(format!(
                "model {alias:?}: {field} must be a non-negative per-unit USD decimal string"
            ));
        }
    }
    for (field, value) in [
        ("created", metadata.created),
        ("max_prompt_length", metadata.max_prompt_length),
        ("max_output_length", metadata.max_output_length),
        ("default_output_length", metadata.default_output_length),
        ("capacity.prompt_tpm", metadata.capacity.prompt_tpm),
        (
            "capacity.cached_prompt_tpm",
            metadata.capacity.cached_prompt_tpm,
        ),
        ("capacity.completion_tpm", metadata.capacity.completion_tpm),
        ("capacity.request_rpm", metadata.capacity.request_rpm),
        ("capacity.concurrency", metadata.capacity.concurrency),
    ] {
        if let Some(value) = value
            && value > JSON_SAFE_INTEGER_MAX
        {
            return Err(format!(
                "model {alias:?}: {field} exceeds OpenRouter's JSON safe-integer maximum"
            ));
        }
    }
    for (field, value) in [
        ("max_prompt_length", metadata.max_prompt_length),
        ("max_output_length", metadata.max_output_length),
        ("default_output_length", metadata.default_output_length),
        ("capacity.prompt_tpm", metadata.capacity.prompt_tpm),
        (
            "capacity.cached_prompt_tpm",
            metadata.capacity.cached_prompt_tpm,
        ),
        ("capacity.completion_tpm", metadata.capacity.completion_tpm),
        ("capacity.request_rpm", metadata.capacity.request_rpm),
        ("capacity.concurrency", metadata.capacity.concurrency),
    ] {
        if value == Some(0) {
            return Err(format!(
                "model {alias:?}: {field} must be greater than zero when declared"
            ));
        }
    }
    if let (Some(default), Some(maximum)) =
        (metadata.default_output_length, metadata.max_output_length)
        && default > maximum
    {
        return Err(format!(
            "model {alias:?}: default_output_length {default} exceeds max_output_length {maximum}"
        ));
    }
    if metadata.default_output_length.is_some() && metadata.max_output_length.is_none() {
        return Err(format!(
            "model {alias:?}: default_output_length requires max_output_length"
        ));
    }
    if let Some(discount) = metadata.discount_to_user
        && (!discount.is_finite() || discount >= 1.0)
    {
        return Err(format!(
            "model {alias:?}: discount_to_user must be finite and less than 1"
        ));
    }
    if metadata
        .openrouter_slug
        .as_deref()
        .is_some_and(str::is_empty)
    {
        return Err(format!(
            "model {alias:?}: openrouter_slug must not be empty when declared"
        ));
    }
    for dc in &metadata.datacenters {
        if dc.country_code.len() != 2 || !dc.country_code.bytes().all(|b| b.is_ascii_uppercase()) {
            return Err(format!(
                "model {alias:?}: datacenter country_code {:?} must be two uppercase ASCII letters",
                dc.country_code
            ));
        }
    }
    Ok(())
}

/// Boot validation for the vendor-recommended sampling defaults (lane/vendor-default-sampling,
/// 2026-08-19). Same posture as `default_reasoning_effort`: FAIL BEFORE GPU LOAD. A bad number
/// here would otherwise apply to every omitting client on a box that came back under the
/// watchdog, which is the worst possible place to discover a typo.
///
/// Ranges are the real API ranges, not taste:
/// - `default_temperature` must be FINITE, > 0.0, <= 2.0. Zero is refused on purpose — see the
///   field docs: a zero default is greedy-by-default wearing a config hat, and it is exactly
///   the hazard the owner ruled out. Greedy is reached by an explicit client `temperature: 0`.
/// - `default_top_p` in (0.0, 1.0]; 1.0 = disabled, 0.0 would mask every token.
/// - `default_top_k` 0 = disabled (keep all); any positive k is a real truncation.
/// - `default_min_p` in [0.0, 1.0); 0.0 = disabled, 1.0 would keep only the argmax.
/// - `default_presence_penalty` / `default_frequency_penalty` in [-2.0, 2.0] (OpenAI's range).
/// - `default_repetition_penalty` finite and > 0.0; 1.0 = off. Zero would zero every logit.
fn validate_sampling_defaults(
    alias: &str,
    metadata: &OpenRouterModelMetadata,
) -> Result<(), String> {
    validate_sampling_arm(
        alias,
        &[
            "default_temperature",
            "default_top_p",
            "default_min_p",
            "default_presence_penalty",
            "default_frequency_penalty",
            "default_repetition_penalty",
        ],
        metadata.default_temperature,
        metadata.default_top_p,
        metadata.default_min_p,
        metadata.default_presence_penalty,
        metadata.default_frequency_penalty,
        metadata.default_repetition_penalty,
    )?;
    if let Some(arm) = &metadata.non_thinking_sampling {
        // A DECLARED-but-empty arm is refused: it would silently hand every
        // thinking-off request the bare API-standard defaults while the file looks
        // configured. Either recommend something or delete the table.
        if arm.is_empty() {
            return Err(format!(
                "model {alias:?}: non_thinking_sampling declares no fields — declare at \
                 least one vendor recommendation or delete the table"
            ));
        }
        validate_sampling_arm(
            alias,
            &[
                "non_thinking_sampling.temperature",
                "non_thinking_sampling.top_p",
                "non_thinking_sampling.min_p",
                "non_thinking_sampling.presence_penalty",
                "non_thinking_sampling.frequency_penalty",
                "non_thinking_sampling.repetition_penalty",
            ],
            arm.temperature,
            arm.top_p,
            arm.min_p,
            arm.presence_penalty,
            arm.frequency_penalty,
            arm.repetition_penalty,
        )?;
    }
    Ok(())
}

/// The range law for ONE sampling arm — the flat `default_*` keys and the
/// `non_thinking_sampling` table go through this same body so the two arms cannot
/// drift apart in what they accept (a zero temperature is refused on BOTH, for the
/// same greedy-by-default reason). `keys` carries the six TOML key names in field
/// order purely so the refusal names the exact key the operator wrote.
#[allow(clippy::too_many_arguments)]
fn validate_sampling_arm(
    alias: &str,
    keys: &[&str; 6],
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
) -> Result<(), String> {
    if let Some(t) = temperature
        && (!t.is_finite() || t <= 0.0 || t > 2.0)
    {
        return Err(format!(
            "model {alias:?}: {} {t} must be finite and in (0, 2]. \
                 A zero DEFAULT would make greedy decoding the deployment-wide behavior for \
                 every request that omits temperature (owner ruling 2026-08-19: we serve the \
                 vendor recommendation, not greedy); clients reach greedy by sending an \
                 explicit temperature 0.",
            keys[0]
        ));
    }
    if let Some(p) = top_p
        && (!p.is_finite() || p <= 0.0 || p > 1.0)
    {
        return Err(format!(
            "model {alias:?}: {} {p} must be finite and in (0, 1] (1.0 = disabled)",
            keys[1]
        ));
    }
    if let Some(m) = min_p
        && (!m.is_finite() || !(0.0..1.0).contains(&m))
    {
        return Err(format!(
            "model {alias:?}: {} {m} must be finite and in [0, 1) (0.0 = disabled)",
            keys[2]
        ));
    }
    for (field, value) in [(keys[3], presence_penalty), (keys[4], frequency_penalty)] {
        if let Some(v) = value
            && (!v.is_finite() || !(-2.0..=2.0).contains(&v))
        {
            return Err(format!(
                "model {alias:?}: {field} {v} must be finite and in [-2, 2]"
            ));
        }
    }
    if let Some(r) = repetition_penalty
        && (!r.is_finite() || r <= 0.0)
    {
        return Err(format!(
            "model {alias:?}: {} {r} must be finite and \
             greater than zero (1.0 = off)",
            keys[5]
        ));
    }
    Ok(())
}

fn load_openrouter_metadata(
    models: &[(String, String, Option<String>)],
) -> Result<
    (
        HashMap<String, OpenRouterModelMetadata>,
        Option<ProviderMetadata>,
    ),
    String,
> {
    let path = match std::env::var("MEMRA_MODEL_METADATA") {
        Ok(path) => path,
        Err(_) => return Ok((HashMap::new(), None)),
    };
    let p = std::path::Path::new(&path);
    if !p.is_file() {
        return Err(format!(
            "MEMRA_MODEL_METADATA={path:?} is not an existing TOML file"
        ));
    }
    let text =
        std::fs::read_to_string(p).map_err(|e| format!("MEMRA_MODEL_METADATA {path:?}: {e}"))?;
    let (metadata, provider) = OpenRouterMetadataFile::parse(&text)
        .map_err(|e| format!("MEMRA_MODEL_METADATA {path:?}: {e}"))?;
    for alias in metadata.keys() {
        if !models.iter().any(|(name, _, _)| name == alias) {
            return Err(format!(
                "MEMRA_MODEL_METADATA {path:?}: model alias {alias:?} is not present in MEMRA_MODELS"
            ));
        }
    }
    eprintln!(
        "[server] OpenRouter metadata loaded: {} model(s) from {path}",
        metadata.len()
    );
    Ok((metadata, provider))
}

#[derive(Clone)]
struct AppState {
    cmd_tx: Sender<Cmd>,
    models: Arc<Vec<String>>,
    caps: Arc<HashMap<String, ModelCaps>>,
    openrouter_metadata: Arc<HashMap<String, OpenRouterModelMetadata>>,
    /// Contract-v2 provider identity from the metadata file (None = no provider block).
    provider_metadata: Arc<Option<ProviderMetadata>>,
    /// Optional admission + usage accounting behind the metering seam. Terminal usage is
    /// synced before the HTTP completion is published; the CUDA-owner worker never performs
    /// accounting I/O. None ⇔ no accounting configured (the old `request_ledger: None`).
    /// The stock binary wires `ledger::Ledger`; limits enforcement (the old
    /// `tenant_budgets`) is the same object answering `enforces_limits()`.
    metering: Option<Arc<dyn metering::Metering>>,
    /// HTTP-side tokenizer copies used only when prepaid enforcement is enabled. Reservations
    /// price the same rendered prompt before worker admission, without moving auth into worker.rs.
    budget_tokenizers: Option<Arc<HashMap<String, Arc<Tokenizer>>>>,
    /// Immutable request-auth sources resolved before model load. The keyring itself
    /// hot-reloads internally; the source selection must not drift after bind validation.
    api_auth: ApiAuth,
    /// Metrics are open only for the no-key loopback development shape.
    metrics_auth: MetricsAuth,
    metrics: SharedMetrics,
    /// live per-lane in-flight request gauge (HTTP-layer view: submitted and not yet
    /// finished, queued-at-worker included) — drives the X-RateLimit-* headers and the
    /// graceful-drain completion barrier (serve-tail lane, gap-scan F11/F12).
    inflight: InflightCounts,
    /// per-tenant in-flight gauge (lane/api-keys): keyed by tenant id, same RAII life as
    /// the lane gauge — drives per-key rate-limit overrides + their headers.
    tenant_inflight: TenantGauge,
    /// inference liveness (lane/serve-hardening, G5): the GPU worker's heartbeat + phase +
    /// fault latches, shared with the worker thread and the Xid watcher. /health, /livez and
    /// /readyz read ONLY this — never "the process is up".
    health: health::SharedHealth,
    /// dead-darklane background job observability (lane/darklane-training): the runner's
    /// shared counters + its yield mode, for the /metrics "bg" block. None when MEMRA_BG_JOB
    /// is unset — the block is absent and the payload byte-identical to pre-lane.
    bg: Option<(Arc<darklane::BgJobState>, &'static str)>,
}

impl AppState {
    /// THE per-request vendor-defaults lookup: every surface handler resolves this model's
    /// omitted-field sampling defaults through this one body (operator metadata first, arch
    /// caps second — `SamplingDefaults::resolve`). Handlers call this instead of composing
    /// the two sources at their own call site so a surface CANNOT quietly consult fewer
    /// sources than its siblings: that asymmetry is exactly how `/v1/completions` used to
    /// ship temperature 1.0 against the Step-3.7 arch caps (0.5/0.9) the chat path applied
    /// (hermes `d991b51699218285`; the resolver itself landed with
    /// lane/vendor-default-sampling, 8e9f37a1b7). The worker-truth teeth live in
    /// `same_omitted_request_resolves_identically_on_all_four_surfaces`.
    ///
    /// Returns BOTH vendor arms (lane/per-mode-sampling, 2026-08-24); which one a request
    /// gets is decided by its resolved thinking mode inside the one builder
    /// (`ModelSamplingDefaults::for_mode`), never at a surface's own call site.
    fn sampling_defaults(&self, model: &str) -> ModelSamplingDefaults {
        ModelSamplingDefaults::resolve(self.openrouter_metadata.get(model), self.caps.get(model))
    }
}

#[derive(Clone, Default)]
struct ApiAuth {
    keyring: Option<&'static auth::KeyStore>,
    single_key: Option<Arc<str>>,
}

impl ApiAuth {
    fn from_env() -> Result<ApiAuth, String> {
        let single_key = match std::env::var("MEMRA_API_KEY") {
            Ok(key) if key.is_empty() => return Err("MEMRA_API_KEY must not be empty".into()),
            Ok(key) => Some(Arc::from(key)),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("MEMRA_API_KEY must be valid UTF-8".into());
            }
        };
        Ok(ApiAuth {
            keyring: auth::global(),
            single_key,
        })
    }

    fn configured(&self) -> bool {
        self.keyring.is_some() || self.single_key.is_some()
    }
}

#[derive(Clone, Default)]
struct MetricsAuth {
    required: bool,
    token: Option<Arc<str>>,
}

impl MetricsAuth {
    fn new(bind_loopback: bool, api_auth_configured: bool, token: Option<String>) -> MetricsAuth {
        let token = token.map(Arc::from);
        MetricsAuth {
            required: !bind_loopback || api_auth_configured || token.is_some(),
            token,
        }
    }
}

fn resolve_bind_addr(addr: &str) -> Result<(SocketAddr, bool), String> {
    let mut resolved = addr
        .to_socket_addrs()
        .map_err(|e| format!("MEMRA_ADDR={addr:?} cannot be resolved: {e}"))?;
    let first = resolved
        .next()
        .ok_or_else(|| format!("MEMRA_ADDR={addr:?} resolved to no socket addresses"))?;
    let mut loopback = first.ip().to_canonical().is_loopback();
    for socket in resolved {
        loopback &= socket.ip().to_canonical().is_loopback();
    }
    Ok((first, loopback))
}

fn bind_is_loopback(addr: &str) -> Result<bool, String> {
    resolve_bind_addr(addr).map(|(_, loopback)| loopback)
}

fn validate_bind_security(
    addr: &str,
    api_auth_configured: bool,
    allow_open_bind: bool,
) -> Result<bool, String> {
    let loopback = bind_is_loopback(addr)?;
    if !loopback && !api_auth_configured && !allow_open_bind {
        return Err(format!(
            "refusing unauthenticated non-loopback bind {addr:?}; configure MEMRA_API_KEY or \
             MEMRA_API_KEYS, or set MEMRA_ALLOW_OPEN_BIND=1 for an explicit development override"
        ));
    }
    Ok(loopback)
}

// ---- rate-limit headers (serve-tail lane, 2026-08-04; gap-scan F12) ----
//
// X-RateLimit-Limit / -Remaining / -Reset on /v1/completions and /v1/chat/completions,
// with CONCURRENCY-SLOT semantics (this server admission-caps concurrent sessions; it has
// no request/min or token/min budget to report — inventing one would be dishonest):
//   Limit     = the lane's configured admission cap — the same values the worker's own
//               admission gate enforces (interactive: MEMRA_MAX_SESSIONS batched /
//               MAX_ACTIVE legacy; judge/harvest: LanePolicy max_sessions).
//   Remaining = free slots at submission time (cap minus in-flight, this request
//               included). Interactive beyond the cap QUEUES (never shed), so Remaining 0
//               means "you will wait", not "you will be rejected".
//   Reset     = seconds until a slot is ESTIMATED free: 0 while slots are free; else the
//               live meter's mean service time (tokens/request x p50 step latency) when
//               it has signal, else MEMRA_RL_RESET_S (default 2). Honestly coarse — a
//               hint, not a promise.
// Dark-lane 429 sheds carry the same trio (Retry-After was already there).

type InflightCounts = Arc<[std::sync::atomic::AtomicUsize; 3]>;

/// Per-tenant in-flight gauge (lane/api-keys): tenant id -> live request count. Entries
/// are removed at zero so the map stays bounded by concurrent tenants, not tenant history.
type TenantGauge = Arc<std::sync::Mutex<HashMap<String, usize>>>;

/// RAII in-flight slot: increments the lane + tenant gauges at submission, decrements
/// both when the response is complete — dropped at handler exit (blocking) or when the
/// SSE stream finishes/disconnects (moved into the stream).
struct InflightGuard {
    counts: InflightCounts,
    idx: usize,
    tenants: TenantGauge,
    tenant: String,
}

impl InflightGuard {
    /// Atomically enforce a binding tenant cap, then return the guard + the (lane, tenant)
    /// in-flight counts INCLUDING this request. The tenant mutex closes the two-arrivals-at-
    /// once race: at cap, exactly one request wins and the other returns the existing count.
    fn try_acquire(
        counts: InflightCounts,
        lane: lanes::Lane,
        tenants: TenantGauge,
        tenant: &str,
        tenant_cap: Option<usize>,
    ) -> Result<(Self, usize, usize), usize> {
        let idx = lane.idx();
        let nt = {
            let mut m = tenants.lock().unwrap();
            let e = m.entry(tenant.to_string()).or_insert(0);
            if tenant_cap.is_some_and(|cap| *e >= cap) {
                return Err(*e);
            }
            *e += 1;
            *e
        };
        let n = counts[idx].fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        Ok((
            InflightGuard {
                counts,
                idx,
                tenants,
                tenant: tenant.to_string(),
            },
            n,
            nt,
        ))
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counts[self.idx].fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        let mut m = self.tenants.lock().unwrap();
        if let Some(e) = m.get_mut(&self.tenant) {
            *e -= 1;
            if *e == 0 {
                m.remove(&self.tenant);
            }
        }
    }
}

/// The lane's configured admission cap — mirrors the worker's admission gate exactly
/// (worker.rs step 2): interactive = MEMRA_MAX_SESSIONS (64) batched / MAX_ACTIVE legacy;
/// judge/harvest = LanePolicy::from_env().max_sessions. Read once.
fn lane_cap(lane: lanes::Lane) -> usize {
    static CAPS: std::sync::OnceLock<[usize; 3]> = std::sync::OnceLock::new();
    CAPS.get_or_init(|| {
        let batching = std::env::var("MEMRA_SERVE_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true);
        let interactive = if batching {
            std::env::var("MEMRA_MAX_SESSIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64)
        } else {
            worker::MAX_ACTIVE
        };
        let p = lanes::LanePolicy::from_env();
        [interactive, p.max_sessions[1], p.max_sessions[2]]
    })[lane.idx()]
}

/// Coarse next-slot estimate (seconds): mean tokens/request x p50 step latency from the
/// live meter when it has signal, else the MEMRA_RL_RESET_S static (default 2).
fn reset_estimate_s(m: &worker::Metrics) -> u64 {
    if m.completed > 0 && m.step_p50_ms > 0.0 {
        let mean_toks = m.tokens_out as f64 / m.completed as f64;
        return ((mean_toks * m.step_p50_ms as f64 / 1000.0).ceil() as u64).clamp(1, 600);
    }
    static D: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("MEMRA_RL_RESET_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

// ---- request deadline + deadline-aware admission (lane/deadline-billing-20260823) --------
//
// Owner ruling (2026-08-23): "we can add a timeout param to the api with default timeout
// documented correctly, and if the time pass and we didnt responed in time we fail and we
// dont bill. if the non response is our fault we should not bill. we need to have
// backpressure and circut breaker."
//
// The circuit breaker itself lives at the router (per-isolate breaker + load spill on the
// X-RateLimit readings); THIS side's whole contribution to it is honest, prompt 429s with
// Retry-After. Do not build a second breaker here.

/// `timeout_ms` bounds. The 90 s maximum is a PLATFORM fact, not a preference: Cloudflare's
/// proxy returns 524 at ~100 s of time-to-headers for a non-streaming response, so any
/// promise past 90 s would be broken upstream of this server no matter what it does. The
/// default equals the maximum — "we answer inside 90 s or you don't pay" is the documented
/// contract for every request, including ones that never heard of the parameter.
pub(crate) const TIMEOUT_MS_MIN: u64 = 1_000;
pub(crate) const TIMEOUT_MS_MAX: u64 = 90_000;
pub(crate) const TIMEOUT_MS_DEFAULT: u64 = 90_000;

/// `MEMRA_TIMEOUT_MS_MAX` — measurement-cell override of the deadline ceiling (docs/FLAGS.md
/// row of the same name). The 90 s ceiling is a PLATFORM fact of the fronted product route
/// (Cloudflare 524 at ~100 s of time-to-headers), so raising it is only honest on a
/// direct-to-server connection, which is exactly the offline capacity/prefill measurement
/// shape it exists for (lane/glm53-1m-demo: a ~1M-token monolithic prime runs for hours, and
/// that cell's question is capacity and correctness, not latency). Unset, unparseable, or
/// below `TIMEOUT_MS_MIN` => the shipped ceiling, behavior byte-identical to before this
/// function existed. When set, the default follows it, preserving the documented
/// "default equals the maximum" contract for requests that never pass the parameter.
pub(crate) fn timeout_ms_max() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MEMRA_TIMEOUT_MS_MAX")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&ms| ms >= TIMEOUT_MS_MIN)
            .unwrap_or(TIMEOUT_MS_MAX)
    })
}

/// Validate `timeout_ms` (all four surfaces call this ONE body — standard-surface law).
/// Absent/null => the documented default. Wrong type or out of range => the named-400
/// message, which always states the range and the streaming escape hatch.
pub(crate) fn parse_timeout_ms(v: Option<&serde_json::Value>) -> Result<u64, String> {
    let max = timeout_ms_max();
    let Some(v) = v.filter(|v| !v.is_null()) else {
        // Default equals the maximum, including under the measurement-cell override.
        return Ok(max);
    };
    let Some(ms) = v.as_u64() else {
        return Err(format!(
            "timeout_ms must be an integer number of milliseconds in \
             {TIMEOUT_MS_MIN}..={max}, got {v}; for work longer than \
             {max} ms use \"stream\": true — the deadline then bounds only the \
             time to first token and the stream may run as long as it needs"
        ));
    };
    if !(TIMEOUT_MS_MIN..=max).contains(&ms) {
        return Err(format!(
            "timeout_ms {ms} is outside the accepted range \
             {TIMEOUT_MS_MIN}..={max} (milliseconds). {max} is a \
             platform ceiling, not a preference: the fronting proxy fails a non-streaming \
             response whose headers take ~100 s (HTTP 524), so promising more would be a \
             lie. For work longer than {max} ms use \"stream\": true — the \
             deadline then bounds only the time to first token and the stream may run as \
             long as it needs"
        ));
    }
    Ok(ms)
}

/// One request's effective deadline: the instant it expires plus the declared value (for
/// error messages that must name the deadline the caller actually got).
#[derive(Clone, Copy)]
pub(crate) struct RequestDeadline {
    pub(crate) at: tokio::time::Instant,
    pub(crate) ms: u64,
}

impl RequestDeadline {
    pub(crate) fn starting_now(ms: u64) -> Self {
        Self {
            at: tokio::time::Instant::now() + std::time::Duration::from_millis(ms),
            ms,
        }
    }

    pub(crate) fn remaining(&self) -> std::time::Duration {
        self.at
            .saturating_duration_since(tokio::time::Instant::now())
    }
}

/// 408 for a missed deadline: standard error object, `type: "timeout"`,
/// `code: "deadline_exceeded"`, message naming the effective deadline and the billing
/// promise. 408 is deliberately retryable (exempt from `x-should-retry: false` — SDKs
/// retry it by default) and carries no Retry-After: the miss says nothing about when a
/// retry would fit, and a made-up window would be a promise this server cannot keep.
pub(crate) fn deadline_exceeded_response(ms: u64, stream: bool) -> Response {
    let what = if stream {
        "the first token was produced"
    } else {
        "the response completed"
    };
    let msg = format!(
        "deadline of {ms} ms (timeout_ms; default {TIMEOUT_MS_DEFAULT}) elapsed before \
         {what}; generation was cancelled and this request is not billed"
    );
    error_response_coded(
        StatusCode::REQUEST_TIMEOUT,
        &msg,
        "timeout",
        Some("timeout_ms"),
        Some("deadline_exceeded"),
    )
}

// ---- non-streaming feasibility gate (lane/deadline-partial-20260826) ---------------
//
// Owner report 2026-08-26: "we have an issue with non streaming and timeouts, if someone
// sends 30k token input, he get a timeout ... thats a customer expirience", and the
// ruling: "the 90s cap doesnt make sense, it should or return in batches that it can work
// under 90s or limit is full context".
//
// MEASURED SHAPE (darklanes research/nonstream-deadline-20260826): at 30,278 prompt
// tokens through the customer path, non-streaming answered 200 at 4096 out (52.0 s),
// 5120 (61.9 s) and 6144 (71.5 s), and 408'd at 8192 (90.7 s) and 16384 (91.5 s), while
// the SAME 8192-token work streamed 200 in 93.8 s — past the deadline. So the wall clock
// never bounded the box, only one response shape, and 90 s of generated tokens were
// discarded to produce the error.
//
// Two gates answer the ruling. This one is the "limit is knowable" half: refuse a
// non-streaming request we can SEE will not finish, immediately, naming the max_tokens
// that fits — instead of burning the full deadline and discarding the work. The other
// half (deliver what was generated when the deadline lands anyway) is in
// `blocking_response_with_receipt`.
//
// WHY A CONSERVATIVE ESTIMATE PLUS A MARGIN, not a promise: throughput is shape-dependent
// (the same box does ~100 tok/s on verbose prose and 300+ on digits), so a tight estimate
// would refuse requests that would have succeeded — and a false refusal is worse than a
// slow success. The floors below are deliberately BELOW anything measured, and the gate
// only fires when even the pessimistic estimate exceeds the deadline by MARGIN. On the
// measured ladder that boundary lands between 6144 (allowed; really 71.5 s) and 8192
// (refused; really a 408), which is the behaviour the receipts ask for.
//
// INDUSTRY CHECK (owner: "check how other enddoints handle non streaming answers"):
// Anthropic enforces the same idea client-side — its SDK raises
// "Streaming is required for operations that may take longer than 10 minutes" BEFORE
// sending — and OpenAI, Google, Azure and the hosted resellers all decline to publish a server-side duration
// ceiling and push long work to streaming or an async/batch surface. Refusing early with
// an actionable message is the precedented behaviour; silently truncating is not.

/// Pessimistic prefill rate for the feasibility estimate, tokens/second. The api-router
/// uses the same 2k floor for its own header-timeout budget; measured prefill on the
/// serving cards is ~2.9k tok/s at 30k tokens, so this under-promises on purpose.
/// Override: `MEMRA_PREFILL_FLOOR_TOK_S`.
pub(crate) const PREFILL_FLOOR_TOK_S: u64 = 2_000;

/// Pessimistic decode rate for the feasibility estimate, tokens/second. The slowest arm
/// measured through the customer path on the current fleet is ~100 tok/s (verbose prose at
/// 30k context); 60 leaves room for a busier box without refusing honest work.
/// Override: `MEMRA_DECODE_FLOOR_TOK_S`.
pub(crate) const DECODE_FLOOR_TOK_S: u64 = 60;

/// How far past the deadline the pessimistic estimate must land before this gate refuses,
/// in percent. 150 = "refuse only when even the floor-rate estimate needs 1.5x the
/// deadline"; anything closer is attempted and covered by partial delivery.
pub(crate) const DEADLINE_INFEASIBLE_MARGIN_PCT: u64 = 150;

/// A BOOLEAN flag, which needs its own reader precisely BECAUSE `env_u64` filters to
/// POSITIVE values: reading an off-switch through that reader made `=0` fall back to the
/// default, so the documented rollback seam did nothing. Caught by the bench gate — arm 7
/// ran with `MEMRA_NONSTREAM_DEADLINE_GATE=0` set and was still refused — which is the only
/// reason the FLAGS.md row is not a lie. `0`/`off`/`false` = off; anything else = on.
fn env_flag_on(name: &'static str, default_on: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false"
        ),
        Err(_) => default_on,
    }
}

/// A POSITIVE numeric knob (a rate): zero and garbage fall back to the default, because a
/// zero rate would divide by zero in the estimate. NEVER read a boolean through this.
pub(crate) fn env_u64(name: &'static str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Prompt size in tokens for the feasibility estimate ONLY — never for billing, never for
/// admission accounting, both of which count with the real tokenizer at their own sites.
///
/// Exact when the caller sent `prompt_ids` or a budget tokenizer for this model is loaded
/// (production always has one). The character fallback DELIBERATELY UNDER-COUNTS at
/// `bytes / CHARS_PER_TOKEN_FLOOR`: an over-count inflates the prefill term and refuses
/// requests that would have succeeded, while an under-count merely lets a doomed request
/// through to partial delivery. The bench gate caught this — a bytes/4 proxy read a real
/// 30,278-token prompt as 51,277 (that text runs ~6.8 chars/token), a 69% over-count in
/// the false-refusal direction.
const CHARS_PER_TOKEN_FLOOR: usize = 6;

pub(crate) fn prompt_tokens_estimate(
    request: &worker::Request,
    tokenizer: Option<&Tokenizer>,
) -> u64 {
    if !request.prompt_ids.is_empty() {
        return request.prompt_ids.len() as u64;
    }
    let mut text = String::new();
    text.push_str(&request.prompt_text);
    for turn in &request.chat_turns {
        text.push_str(&turn.content);
    }
    for tool in &request.tools_json {
        text.push_str(tool);
    }
    if let Some(tokenizer) = tokenizer {
        return tokenizer.encode(text.as_str(), false).len() as u64;
    }
    (text.len() / CHARS_PER_TOKEN_FLOOR) as u64
}

/// The `max_tokens` that WOULD fit this request's remaining deadline at the floor rates,
/// after paying for prefill. `None` when prefill alone cannot fit — that request has no
/// feasible completion length at all.
pub(crate) fn deadline_fitting_max_tokens(prompt_tokens: u64, remaining_ms: u64) -> Option<u64> {
    let prefill_ms = prompt_tokens
        .saturating_mul(1_000)
        .checked_div(env_u64("MEMRA_PREFILL_FLOOR_TOK_S", PREFILL_FLOOR_TOK_S))
        .unwrap_or(u64::MAX);
    let decode_ms = remaining_ms.checked_sub(prefill_ms)?;
    if decode_ms == 0 {
        return None;
    }
    Some(decode_ms.saturating_mul(env_u64("MEMRA_DECODE_FLOOR_TOK_S", DECODE_FLOOR_TOK_S)) / 1_000)
}

/// Refuse a non-streaming request whose pessimistic estimate exceeds its deadline by
/// `DEADLINE_INFEASIBLE_MARGIN_PCT`. Returns the 400 message; the caller answers with a
/// named 400 (`code: "nonstream_deadline_infeasible"`), which costs no slot, opens no
/// receipt, and burns no GPU — the point of the gate.
///
/// Streaming is never gated: its deadline bounds only time-to-first-token and the stream
/// may run as long as it needs, which is exactly what this message tells the caller.
/// Off switch: `MEMRA_NONSTREAM_DEADLINE_GATE=0` (then an infeasible request runs and is
/// covered by partial delivery instead).
pub(crate) fn nonstream_deadline_gate(
    request: &worker::Request,
    stream: bool,
    deadline: RequestDeadline,
    caller_declared_max_tokens: bool,
    tokenizer: Option<&Tokenizer>,
) -> Result<(), String> {
    if stream || !env_flag_on("MEMRA_NONSTREAM_DEADLINE_GATE", true) {
        return Ok(());
    }
    let max_new = request.params.max_new as u64;
    // ONLY a caller-declared max_tokens is judged. An omitted cap is the owner's "limit is
    // full context" case: `apply_model_request_limits` has already resolved it to the
    // model's max_output (32768 on the q38 registry), so gating it would refuse the single
    // MOST COMMON customer shape — a request with no max_tokens at all — over a number the
    // caller never chose and cannot act on. The bench gate caught exactly that (arm 5).
    // Those requests run and are covered by partial delivery instead.
    if !caller_declared_max_tokens || max_new == worker::MAX_NEW_CTX_BOUNDED as u64 || max_new == 0
    {
        return Ok(());
    }
    let prompt_tokens = prompt_tokens_estimate(request, tokenizer);
    let remaining_ms = deadline.remaining().as_millis() as u64;
    let prefill_ms = prompt_tokens.saturating_mul(1_000)
        / env_u64("MEMRA_PREFILL_FLOOR_TOK_S", PREFILL_FLOOR_TOK_S).max(1);
    let decode_ms = max_new.saturating_mul(1_000)
        / env_u64("MEMRA_DECODE_FLOOR_TOK_S", DECODE_FLOOR_TOK_S).max(1);
    let est_ms = prefill_ms.saturating_add(decode_ms);
    let bound_ms = remaining_ms.saturating_mul(DEADLINE_INFEASIBLE_MARGIN_PCT) / 100;
    if est_ms <= bound_ms {
        return Ok(());
    }
    let fits = deadline_fitting_max_tokens(prompt_tokens, remaining_ms);
    let advice = match fits {
        Some(fits) if fits > 0 => format!(
            "lower max_tokens to about {fits} for this prompt, or set \"stream\": true — a \
             stream's deadline bounds only the time to first token, so it may run as long \
             as it needs"
        ),
        _ => format!(
            "this prompt ({prompt_tokens} tok) needs most of the deadline before the first \
             token, so no max_tokens fits: set \"stream\": true"
        ),
    };
    Err(format!(
        "a non-streaming request for {max_new} tokens on a ~{prompt_tokens}-token prompt \
         needs an estimated ~{}s, which does not fit the {remaining_ms} ms timeout_ms \
         deadline (max {TIMEOUT_MS_MAX} ms — a platform ceiling: the fronting proxy fails \
         a non-streaming response whose headers take ~100 s). Refused before any GPU work \
         rather than after the deadline: {advice}",
        est_ms / 1_000,
    ))
}

/// Absolute per-lane queue bound (the backpressure backstop): `MEMRA_MAX_QUEUE_DEPTH`, default
/// 4x the selected lane's session cap. At the bound, new requests shed with a 429 (`shed_queue`,
/// never billed) instead of entering an unbounded handler/worker channel. Read once.
fn max_queue_depth(cap: usize) -> usize {
    static D: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        std::env::var("MEMRA_MAX_QUEUE_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
    })
    .unwrap_or(cap.saturating_mul(4))
}

/// Absolute queue-wait ceiling for the interactive lane: `MEMRA_QUEUE_WAIT_CEILING_S`
/// (default **0 = OFF by design**, darklanes#5). At `N > 0`, an interactive request whose
/// estimated queue wait exceeds `N` seconds sheds 429 (`shed_queue_wait`, never billed)
/// with `Retry-After` = the estimate, even when the caller's own deadline could absorb the
/// wait. `0`, absent, or unparsable = off (today's silent-queue behavior). Read once.
/// Full doc: docs/FLAGS.md row.
fn queue_wait_ceiling_s() -> u64 {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("MEMRA_QUEUE_WAIT_CEILING_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Deadline-aware admission for the interactive lane, which QUEUES beyond the session cap
/// (never sheds) — so before this gate a saturated box accepted every request and simply
/// answered late. At submission time (never after — an admitted request is never shed):
///
///   (a) absolute bound: backlog >= `max_queue_depth` => 429 `shed_queue`;
///   (b) deadline test: estimated queue wait > the request's remaining deadline =>
///       429 `shed_deadline`, Retry-After = the estimate;
///   (c) wait ceiling (opt-in, darklanes#5): `MEMRA_QUEUE_WAIT_CEILING_S` set to N > 0
///       and estimated queue wait > N => 429 `shed_queue_wait`, Retry-After = the
///       estimate. Independent of the caller's deadline: (b) never fires for a patient
///       caller, which is exactly how prod queued 133-137 s in silence.
///
/// The estimate reuses the SAME machinery as X-RateLimit-Reset (mean tokens/request x p50
/// step latency), scaled by how many cap-wide waves of queued requests are ahead. Honestly
/// coarse — a hint, not a promise — and the shed messages say so. Judge/harvest lanes
/// already shed at cap inside the worker; this gate is interactive-only.
/// Atomically reserve one slot in the handler-to-worker queue. The older
/// the estimator-based backpressure check it replaced is gone, but a
/// successful admission must use this compare-exchange immediately before the
/// command send so concurrent handlers cannot all pass one stale snapshot.
pub(crate) struct PendingAdmissionGuard {
    reserved: bool,
    lane: lanes::Lane,
}

impl PendingAdmissionGuard {
    /// Transfer the reservation to the worker. The command-channel gauge is released when the
    /// worker pops the command; the hard queue reservation remains until actual model admission
    /// or terminal rejection. Dropping a guard before send rolls both counters back.
    pub(crate) fn commit(mut self) {
        self.reserved = false;
        std::mem::forget(self);
    }
}

impl Drop for PendingAdmissionGuard {
    fn drop(&mut self) {
        if self.reserved {
            worker::release_pending_admit();
            worker::release_admission_reservation(self.lane);
        }
    }
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
pub(crate) fn reserve_pending_admit(
    st: &AppState,
    lane: lanes::Lane,
    rl: &RateLimit,
    deadline: RequestDeadline,
) -> Result<PendingAdmissionGuard, (Response, &'static str)> {
    reserve_pending_admit_with_ceiling(st, lane, rl, deadline, queue_wait_ceiling_s())
}

/// `reserve_pending_admit` with the queue-wait ceiling passed explicitly, so both arms of
/// the flag are unit-testable in one process (the env read above is a OnceLock). Every
/// production ingress goes through the wrapper; only tests call this directly.
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn reserve_pending_admit_with_ceiling(
    st: &AppState,
    lane: lanes::Lane,
    rl: &RateLimit,
    deadline: RequestDeadline,
    ceiling_s: u64,
) -> Result<PendingAdmissionGuard, (Response, &'static str)> {
    // The queue bound is a capacity safety property, not a quota-only feature. A key with
    // remaining rate-limit headroom can still open hundreds of concurrent requests; applying
    // the same bound to every interactive request keeps the normal and DSV4 unbounded channels
    // finite even before a per-key window reaches zero.
    let cap = lane_cap(lane).max(1);
    let bound = max_queue_depth(cap);
    let reservations_for_lane = &worker::ADMISSION_RESERVATIONS[lane.idx()];
    loop {
        let m = st.metrics.lock().map(|m| m.clone()).unwrap_or_default();
        let reservations = reservations_for_lane.load(std::sync::atomic::Ordering::Acquire);
        // Every production ingress reserves before sending, and step-OOM requeues re-arm their
        // lane explicitly. Keep this count lane-local: a harvest flood must never make an
        // interactive request appear queued.
        let backlog = reservations;
        let est_wait_s = reset_estimate_s(&m).saturating_mul((backlog / cap + 1) as u64);
        if backlog >= bound {
            let msg = format!(
                "{} queue is at its bound ({backlog} queued, bound {bound}); this \
                 request was not admitted and is not billed; retry after ~{est_wait_s}s (a \
                 coarse estimate, not a promise)",
                lane.as_str()
            );
            let resp = retry_contract_response(
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(error_body(
                        &msg,
                        "rate_limit_error",
                        None,
                        Some("shed_queue"),
                    )),
                )
                    .into_response(),
                Some(est_wait_s),
            );
            return Err((resp, "shed_queue"));
        }
        let remaining_ms = deadline.remaining().as_millis() as u64;
        // A request with a free slot (remaining > 0 and no queued work) is admitted
        // immediately; do not apply the coarse reset estimate to it. Once the lane is
        // full or another request is queued, the estimate represents real waiting time.
        let waits_for_capacity = rl.remaining == 0 || backlog > 0;
        if lane == lanes::Lane::Interactive
            && waits_for_capacity
            && est_wait_s.saturating_mul(1_000) > remaining_ms
        {
            let msg = format!(
                "estimated queue wait ~{est_wait_s}s exceeds this request's remaining \
                 timeout_ms deadline ({remaining_ms} ms); this request was not admitted and \
                 is not billed; retry after ~{est_wait_s}s or raise timeout_ms (a coarse \
                 estimate, not a promise)"
            );
            let resp = retry_contract_response(
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(error_body(
                        &msg,
                        "rate_limit_error",
                        None,
                        Some("shed_deadline"),
                    )),
                )
                    .into_response(),
                Some(est_wait_s),
            );
            return Err((resp, "shed_deadline"));
        }
        // QUEUE-WAIT CEILING (darklanes#5, opt-in): the deadline test above never fires
        // for a patient caller, so a burst past the session cap queued interactively for
        // 133-137 s of pre-header silence on prod (2026-09-01) without a single 429. With
        // `MEMRA_QUEUE_WAIT_CEILING_S` = N > 0, a projected wait past N sheds here with the
        // same retry contract instead of making the caller discover the wait by enduring
        // it. Same trigger posture as (b): only a request that actually waits is judged
        // (a free slot with an empty lane admits immediately, estimate not applied).
        if lane == lanes::Lane::Interactive
            && waits_for_capacity
            && ceiling_s > 0
            && est_wait_s > ceiling_s
        {
            let msg = format!(
                "estimated queue wait ~{est_wait_s}s exceeds this deployment's queue-wait \
                 ceiling ({ceiling_s}s); this request was not admitted and is not billed; \
                 retry after ~{est_wait_s}s (a coarse estimate, not a promise)"
            );
            let resp = retry_contract_response(
                (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(error_body(
                        &msg,
                        "rate_limit_error",
                        None,
                        Some("shed_queue_wait"),
                    )),
                )
                    .into_response(),
                Some(est_wait_s),
            );
            return Err((resp, "shed_queue_wait"));
        }
        if reservations_for_lane
            .compare_exchange(
                reservations,
                reservations.saturating_add(1),
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
        {
            // Keep the command-channel signal for speculative-burst yield decisions. It is
            // released when the worker pops the command, while the hard reservation above is
            // held until actual model admission or terminal rejection.
            worker::PENDING_ADMITS.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Ok(PendingAdmissionGuard {
                reserved: true,
                lane,
            });
        }
    }
}

// ---- graceful drain (serve-tail lane, 2026-08-04; gap-scan F11) ----
//
// SIGTERM flips the drain flag: new requests on the completion routes get an immediate
// 503 + Retry-After (never queued), /health reports "draining" (the LB is_ready signal),
// and the drain task waits on the in-flight gauge (the same HTTP-layer counts the
// rate-limit headers use — streams hold their slot until fully written) up to
// MEMRA_DRAIN_S (default 30s), then shuts the listener down and the process exits 0.
// Fleet restarts stop being SIGKILL-class in-flight loss (the chaos-receipt gap).

/// Process-wide drain flag (set by the SIGTERM task, read by every admission gate).
static DRAINING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn draining() -> bool {
    DRAINING.load(std::sync::atomic::Ordering::SeqCst)
}

/// MEMRA_DRAIN_S (default 30): how long a draining server waits for in-flight requests.
fn drain_deadline_s() -> u64 {
    static D: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("MEMRA_DRAIN_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30)
    })
}

/// 503 for a request that arrived during drain: OpenAI error object + Retry-After
/// (the drain window — by then this instance is gone and its replacement is up).
///
/// Goes through the SAME retry contract as every engine-fault class (G6): a `code` clients can
/// branch on, the `retry-after-ms` twin openai-python reads FIRST, and the value clamped to
/// 60 s because litellm ignores anything above that and openai-python abandons the retry past
/// 120 s. It predates the taxonomy and was the one 503 on the surface still emitting a bare
/// `Retry-After` with no code and no ms twin — i.e. a client that trusted `retry-after-ms`
/// exclusively saw no window at all on the most predictable outage memra has.
fn drain_response() -> Response {
    let resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(error_body(
            "server is draining (shutdown in progress); retry",
            "server_error",
            None,
            Some("draining"),
        )),
    )
        .into_response();
    retry_contract_response(resp, Some(drain_deadline_s()))
}

/// One request's header values, computed at submission time (the "at admit" snapshot).
struct RateLimit {
    limit: usize,
    remaining: usize,
    reset_s: u64,
}

impl RateLimit {
    /// Per-tenant override law (lane/api-keys): the effective cap is
    /// min(tenant_override, global lane cap) — the GLOBAL cap stays authoritative (an
    /// override can only narrow, never widen). Remaining is the tighter of the two
    /// headrooms (tenant cap minus tenant in-flight vs lane cap minus lane in-flight).
    fn at_admit(
        lane: lanes::Lane,
        n_inflight: usize,
        metrics: &SharedMetrics,
        tenant: &auth::TenantCtx,
        n_tenant: usize,
    ) -> Self {
        let global = lane_cap(lane);
        let Some(t) = tenant.rate_limit.filter(|&t| t < global) else {
            return Self::compute(global, n_inflight, metrics);
        };
        let headroom = t
            .saturating_sub(n_tenant)
            .min(global.saturating_sub(n_inflight));
        // compute() derives remaining as limit - n; feed it the effective occupancy.
        Self::compute(t, t - headroom, metrics)
    }

    fn compute(limit: usize, n_inflight: usize, metrics: &SharedMetrics) -> Self {
        let remaining = limit.saturating_sub(n_inflight);
        let reset_s = if remaining > 0 {
            0
        } else {
            let m = metrics.lock().map(|m| m.clone()).unwrap_or_default();
            reset_estimate_s(&m)
        };
        RateLimit {
            limit,
            remaining,
            reset_s,
        }
    }

    /// Stamp the X-RateLimit-* trio onto a response.
    fn attach(&self, mut resp: Response) -> Response {
        let h = resp.headers_mut();
        for (k, v) in [
            ("x-ratelimit-limit", self.limit as u64),
            ("x-ratelimit-remaining", self.remaining as u64),
            ("x-ratelimit-reset", self.reset_s),
        ] {
            if let Ok(v) = axum::http::HeaderValue::from_str(&v.to_string()) {
                h.insert(axum::http::HeaderName::from_static(k), v);
            }
        }
        resp
    }
}

/// Take the HTTP-layer request slot or reject a tenant whose configured override is already
/// full. Global interactive capacity still queues as before; this gate exists only when the
/// key's override is narrower than the lane cap.
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn acquire_request_slot(
    st: &AppState,
    lane: lanes::Lane,
    tenant: &auth::TenantCtx,
    env: &Envelope,
) -> Result<(InflightGuard, RateLimit), Response> {
    let global = lane_cap(lane);
    let tenant_cap = tenant.rate_limit.filter(|&cap| cap < global);
    match InflightGuard::try_acquire(
        st.inflight.clone(),
        lane,
        st.tenant_inflight.clone(),
        &tenant.tenant,
        tenant_cap,
    ) {
        Ok((guard, n_inflight, n_tenant)) => {
            let rl = RateLimit::at_admit(lane, n_inflight, &st.metrics, tenant, n_tenant);
            Ok((guard, rl))
        }
        Err(n_tenant) => {
            let n_inflight = st.inflight[lane.idx()].load(std::sync::atomic::Ordering::SeqCst);
            let rl = RateLimit::at_admit(lane, n_inflight, &st.metrics, tenant, n_tenant);
            let error =
                worker::EngineError::rate_limit("api key concurrent request limit reached; retry");
            Err(rl.attach(with_request_id(&env.id, engine_error_response(&error))))
        }
    }
}

/// POST /v1/completions request body.
#[derive(Deserialize)]
struct CompletionReq {
    model: String,
    #[serde(default)]
    prompt: String,
    /// raw token-id prompt (the exact-token validation-gate path; bypasses the tokenizer).
    #[serde(default)]
    prompt_ids: Vec<u32>,
    /// Omitted (gap-scan F2) => context-bounded (session ctx - prompt, model-capped), the
    /// OpenAI default-when-omitted semantics — NOT a silent 128-token truncation.
    #[serde(default)]
    max_tokens: Option<usize>,
    /// Omitted (dogfood F4) => NOT 0.0/greedy. `serde(default)` on an f32 yielded 0.0, which
    /// silently locked every temperature-omitting client (the owner's own agentic pill) into
    /// deterministic argmax: same context in, same token out, identical tool-call cycles
    /// forever. Explicit `"temperature": 0` still means greedy — that's a caller decision.
    ///
    /// `Option`, not `f32` (lane/vendor-default-sampling, 2026-08-19): the resolver must be able
    /// to tell "the client said nothing" from "the client said a number", because an omitted
    /// field is what the model's own vendor recommendation substitutes for. A bare `f32` cannot
    /// express that distinction — which is precisely how this surface came to disagree with
    /// `/v1/chat/completions`, where the same fields had already been made `Option`. Every
    /// sampling field below is `Option` for the same reason: they resolve through the ONE
    /// `resolve_sampler_config` law that all four surfaces share.
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Not an OpenAI parameter (OpenRouter/HF convention); explicit 0 = disabled = keep all.
    #[serde(default)]
    top_k: Option<usize>,
    /// Not an OpenAI parameter (OpenRouter/HF convention); explicit 0.0 = disabled.
    #[serde(default)]
    min_p: Option<f32>,
    /// OpenAI penalties (gap-scan F3): implemented in SamplerConfig all along, now plumbed.
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// OpenRouter/HF-convention multiplicative penalty (explicit 1.0 = off).
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// Omitted (dogfood F4, second half) => a FRESH RANDOM seed per request. `Option`, not
    /// `u64`: `serde(default)` gave 0, which is a perfectly valid FIXED seed, so every
    /// seed-omitting client replayed one single sampled stream — the same loop the
    /// temperature default caused, surviving the temperature fix. OpenAI's `seed` is
    /// explicitly best-effort determinism WHEN SUPPLIED; omitting it must not pin the RNG.
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: StopSequences,
    /// Unsupported-but-semantic fields (gap-scan F4): captured so they 400 loudly instead
    /// of being silently swallowed by serde (policy: clean 400s, not silent downgrades).
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    best_of: Option<usize>,
    /// wrap the prompt in the model's chat template (single user turn).
    #[serde(default)]
    chat: bool,
    /// stream tokens via SSE; else return one JSON when done.
    #[serde(default)]
    stream: bool,
    /// optional hard context cap.
    #[serde(default)]
    max_ctx: Option<usize>,
    /// Stable calibration-record identity written only when confidence tracing is enabled.
    #[serde(default)]
    trace_id: Option<String>,
    /// PC-ISO prefix-cache namespace (vLLM `cache_salt` convention, optional): requests
    /// only share cached prefixes with requests carrying the SAME salt. Absent/"" = the
    /// default single-tenant namespace (pre-PC-ISO behavior). See `cache_namespace`.
    #[serde(default)]
    cache_salt: Option<String>,
    /// SESSION AFFINITY explicit tier (lane/session-affinity): the caller's own name for
    /// this conversation. See `affinity_key`. `session_id` is the explicit spelling;
    /// `user` is OpenAI's field that real clients already send.
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    user: Option<String>,
    /// Request deadline in milliseconds (lane/deadline-billing-20260823) — see
    /// `parse_timeout_ms` for the range, the platform ceiling, and the billing promise.
    /// Kept as a raw `Value` so a wrong type is OUR named 400, not serde's body-wide one.
    #[serde(default)]
    timeout_ms: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    /// string, null, or an array of `{type:"text",text}` parts (OpenAI content shapes).
    #[serde(default)]
    content: serde_json::Value,
    /// OpenAI assistant-history tool calls, re-rendered into the template on the next turn.
    #[serde(default)]
    tool_calls: Vec<ReqToolCall>,
    /// role:"tool" pairing. The qwen/step dialects pair positionally; the gemma4 tooluse
    /// dialect resolves the response NAME by matching this against the assistant call id.
    #[serde(default)]
    tool_call_id: Option<String>,
    /// role:"tool" function name (some clients send it) — gemma4 fallback when the id does
    /// not resolve. Harmless to the positional dialects.
    #[serde(default)]
    name: Option<String>,
    /// Assistant-history reasoning echoed back by a stateless client (OpenRouter shape). The
    /// gemma4 and dsv4 arms re-render it into the prompt; the qwen arm does NOT.
    ///
    /// That last part used to be documented as "their templates carry no history-reasoning
    /// grammar", and for qwen3.8 that is FALSE (lane/reasoning-schema-20260823): its template
    /// reads `message.reasoning_content` and replays it inside a `<think>` block by default. So
    /// this field is silently dropped on that dialect where the vendor would have used it, which
    /// is a named follow-up — `chat_template_kwargs.preserve_thinking` refuses for the same
    /// reason. Recorded here rather than left as a comment that reads as if nothing were missing.
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
}

#[derive(Deserialize)]
struct ReqToolCall {
    #[serde(default)]
    #[allow(dead_code)]
    id: Option<String>,
    function: ReqToolFunction,
}

#[derive(Deserialize)]
struct ReqToolFunction {
    name: String,
    /// OpenAI sends a JSON-encoded STRING; inline objects are accepted too.
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Clone, Default, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
    #[default]
    None,
}

impl StopSequences {
    /// Empty elements are dropped HERE, at the one ingestion choke point (hermes finding,
    /// fixed 2026-08-23): `"".contains`/`find("")` match at every position, so an empty
    /// stop element ended every decode at the first token and `truncate_at_stop` cut the
    /// whole completion to "". OpenAI treats empty stop strings as invalid; dropping them
    /// matches the None/omitted semantics without 400ing batch clients that pad arrays.
    fn into_vec(self) -> Vec<String> {
        let stops = match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
            Self::None => Vec::new(),
        };
        stops.into_iter().filter(|s| !s.is_empty()).collect()
    }

    fn validate(&self) -> Result<(), String> {
        let stops: &[String] = match self {
            Self::One(stop) => std::slice::from_ref(stop),
            Self::Many(stops) => stops,
            Self::None => &[],
        };
        if stops.len() > MAX_STOP_SEQUENCES {
            return Err(format!(
                "stop accepts at most {MAX_STOP_SEQUENCES} sequences"
            ));
        }
        let mut total = 0usize;
        for stop in stops {
            let bytes = stop.len();
            if bytes > MAX_STOP_SEQUENCE_BYTES {
                return Err(format!(
                    "each stop sequence must be at most {MAX_STOP_SEQUENCE_BYTES} UTF-8 bytes"
                ));
            }
            total = total
                .checked_add(bytes)
                .ok_or_else(|| "stop sequence byte count overflowed".to_string())?;
        }
        if total > MAX_STOP_SEQUENCES_BYTES {
            return Err(format!(
                "stop sequences must total at most {MAX_STOP_SEQUENCES_BYTES} UTF-8 bytes"
            ));
        }
        Ok(())
    }
}

/// OpenAI-compatible multi-turn chat request. `tools`/`tool_choice`/role:"tool" are accepted
/// (serve-tools lane, 2026-08-02): tool schemas render into the model chat template's own
/// <tools> branch and emitted `<tool_call>` blocks parse back into OpenAI `tool_calls` — the
/// model's GGUF chat template remains the sole source of prompt formatting, and the tools
/// path is TEMPLATE + PARSING only (zero engine changes).
#[derive(Deserialize)]
struct ChatCompletionReq {
    model: String,
    messages: Vec<ChatMessage>,
    /// Omitted (gap-scan F2) => context-bounded (session ctx - prompt, model-capped), the
    /// OpenAI default-when-omitted semantics — NOT a silent 128-token truncation.
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    /// Kept as Option so loaded-model capabilities can apply a provider-published default only
    /// when the caller omitted the field. Explicit values, including 0 and 1, remain authoritative.
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// Not an OpenAI parameter (OpenRouter/HF convention); explicit 0 = disabled = keep all.
    /// `Option` so a vendor `default_top_k` can fill the OMITTED case while an explicit 0
    /// stays an explicit "keep all" (lane/vendor-default-sampling, 2026-08-19).
    #[serde(default)]
    top_k: Option<usize>,
    /// Not an OpenAI parameter (OpenRouter/HF convention); explicit 0.0 = disabled.
    #[serde(default)]
    min_p: Option<f32>,
    /// OpenAI penalties (gap-scan F3): implemented in SamplerConfig all along, now plumbed.
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    /// OpenRouter/HF-convention multiplicative penalty (explicit 1.0 = off).
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// Omitted (dogfood F4, second half) => a FRESH RANDOM seed per request. See CompletionReq.
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: StopSequences,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_ctx: Option<usize>,
    /// OpenAI `response_format` (constrained decoding, lane/constrained 2026-08-03):
    /// `{"type":"text"}` (no-op), `{"type":"json_object"}`, and
    /// `{"type":"json_schema","json_schema":{...,"schema":{...}}}` are supported — the
    /// grammar masks logits per decode step (llguidance). Unknown types 400 loudly.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default)]
    n: Option<usize>,
    /// OpenAI tool schemas: `[{"type":"function","function":{name,description?,parameters?}}]`.
    #[serde(default)]
    tools: Vec<serde_json::Value>,
    /// "auto" (default) | "none". "required"/named-function need constrained decoding -> 400.
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    /// OpenAI reasoning effort — ONE surface, per-arch native mapping (see `parse_think`'s
    /// table): low|medium|high = thinking ON at that budget, none|minimal = thinking OFF,
    /// absent = the model's own default. Binary-switch templates (qwen enable_thinking,
    /// gemma4) take the on/off half; level-consuming templates (step35 `Reasoning:`,
    /// hy3 `reasoning_effort:`) also receive the level.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// OpenRouter object form. Exactly THREE keys are understood — `effort`, `enabled`,
    /// `exclude` — and every other key is a named 400 (`parse_reasoning_object`), including
    /// `max_tokens`. Until lane/reasoning-schema-20260823 this was a bare `Value` whose
    /// unknown keys were silently ignored: `reasoning:{max_tokens:1024}` returned 200 and
    /// changed nothing, which is the accepted-and-ignored class the standard-surface law bans.
    /// `reasoning.max_tokens` in particular cannot be honoured here by owner ruling — reasoning
    /// is output and `max_tokens` is the ONE output budget covering it, so there is no separate
    /// reasoning budget to spend against.
    #[serde(default)]
    reasoning: Option<serde_json::Value>,
    /// OpenRouter legacy switch — and on this server it STOPS REASONING rather than hiding it.
    ///
    /// OWNER RULING (2026-08-23): *"we have to actually reason or not reason"*. Reasoning is
    /// compute and output, billed as output, so a flag that merely withheld the text meant we
    /// spent the compute, billed the customer, and delivered less than we charged for. That
    /// third state — generate, bill, withhold — is gone: `include_reasoning:false` and
    /// `reasoning.exclude:true` are now first-class ALIASES of reasoning-off
    /// (`reasoning.enabled:false`), mapping into the one schema as exactly that. There is no
    /// suppression mode left in the server, so there is nothing to hide because nothing is
    /// produced, and the caller gets the cheaper and faster request they asked for.
    ///
    /// Consequence a caller should know: on a model whose template cannot turn reasoning off,
    /// `include_reasoning:false` is now the same named 400 as any other off-request, instead of
    /// a 200 that quietly billed for a hidden reasoning block.
    #[serde(default)]
    include_reasoning: Option<bool>,
    /// vLLM/HF-idiom thinking switch, accepted here as a first-class ALIAS of the
    /// OpenAI/OpenRouter switch (`reasoning.enabled`) — same precedence, same table
    /// (`parse_think`). It exists because the whole vLLM-shaped ecosystem sends it and we
    /// used to drop it: `ChatCompletionReq` has no `deny_unknown_fields`, so
    /// `enable_thinking:false` was accepted with 200 and silently ignored while the model
    /// went on reasoning (lane/reasoning-control-20260823, receipted on the live endpoint).
    /// Silent acceptance of an ignored parameter is banned; this field is now wired, and
    /// a model whose template cannot honour it REFUSES with a named error.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// vLLM `chat_template_kwargs`. This server renders templates in Rust rather than
    /// executing jinja, so it cannot honour arbitrary kwargs — the ONLY key it understands
    /// is `enable_thinking`. Every other key is a loud 400 naming the key, never a silent
    /// drop: passing a kwarg that changes nothing is the same defect as `enable_thinking`
    /// being ignored, one level down.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Value>,
    /// PC-ISO prefix-cache namespace (vLLM `cache_salt` convention, optional): requests
    /// only share cached prefixes with requests carrying the SAME salt. Absent/"" = the
    /// default single-tenant namespace (pre-PC-ISO behavior). See `cache_namespace`.
    #[serde(default)]
    cache_salt: Option<String>,
    /// SESSION AFFINITY explicit tier — see `CompletionReq::session_id` / `affinity_key`.
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    user: Option<String>,
    /// Request deadline in milliseconds (lane/deadline-billing-20260823), identical on all
    /// four surfaces (the translators pass it through to this field). See
    /// `parse_timeout_ms` for the range, the platform ceiling, and the billing promise.
    /// Raw `Value` so a wrong type is OUR named 400, not serde's body-wide one.
    #[serde(default)]
    timeout_ms: Option<serde_json::Value>,
}
fn one() -> f32 {
    1.0
}
/// OpenAI's documented default for an omitted `temperature` on every completion surface, and
/// the LAST resort in `resolve_sampler_config`: it applies only when neither the client, the
/// operator's vendor block, nor the engine's arch caps expressed anything. Kept distinct from
/// `one()` so the intent is greppable: this is a COMPAT default, not a coincidence that it
/// equals the top_p disable value.
fn default_temperature() -> f32 {
    1.0
}

/// Per-model sampling defaults for OMITTED request fields — the vendor's own recommendation
/// for this model, resolved once per request (lane/vendor-default-sampling, 2026-08-19).
///
/// Owner ruling: "we don't have to serve greedy, we measure greedy but we serve what the user
/// chooses" / "we default to what are the recommendations" / "greedy can create issues". So the
/// value a client gets when it says nothing is the MODEL VENDOR's published recommendation, not
/// greedy and not a house guess.
///
/// Two sources, in this precedence:
/// 1. `MEMRA_MODEL_METADATA`'s per-model `default_*` keys — operator-declared for THIS
///    deployment, boot-validated, carrying the vendor citation in the TOML comment.
/// 2. `ModelCaps`' arch-keyed defaults (`chat_temperature_default` / `chat_top_p_default`) —
///    the engine's own built-in knowledge for architectures that publish API defaults
///    (step35 = StepFun's 0.5/0.9). Kept as the fallback so a box with no metadata file
///    behaves exactly as it did before this lane.
///
/// A `None` field means "nothing was recommended for this parameter" and falls through to the
/// API-standard default. Per the lane brief: where a vendor recommends nothing we leave the
/// API-standard value alone rather than inventing one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct SamplingDefaults {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
}

impl SamplingDefaults {
    /// Metadata wins over caps: the operator's declaration is about the artifact actually
    /// loaded on this box, while the arch cap is a family-level guess made at spawn.
    fn resolve(metadata: Option<&OpenRouterModelMetadata>, caps: Option<&ModelCaps>) -> Self {
        SamplingDefaults {
            temperature: metadata
                .and_then(|m| m.default_temperature)
                .or_else(|| caps.and_then(|c| c.chat_temperature_default)),
            top_p: metadata
                .and_then(|m| m.default_top_p)
                .or_else(|| caps.and_then(|c| c.chat_top_p_default)),
            top_k: metadata.and_then(|m| m.default_top_k),
            min_p: metadata.and_then(|m| m.default_min_p),
            frequency_penalty: metadata.and_then(|m| m.default_frequency_penalty),
            presence_penalty: metadata.and_then(|m| m.default_presence_penalty),
            repetition_penalty: metadata.and_then(|m| m.default_repetition_penalty),
        }
    }
}

/// BOTH of a model's vendor sampling arms, resolved once per request (lane/per-mode-sampling,
/// 2026-08-24). Some vendors publish two recommendations — one for thinking mode, one for
/// non-thinking (qwen3.8: 1.0/0.95/20 thinking vs 0.7/0.80/20 + presence 1.5 non-thinking).
/// memra used to carry ONE default per model, so a request that turned thinking OFF was
/// still served the thinking arm's numbers; per the repo law "served models default to the
/// VENDOR's recommendation", the correct default for a thinking-off request whose sampling
/// params are unset is the vendor's non-thinking arm.
///
/// `thinking` is the PRIMARY arm — exactly what `SamplingDefaults::resolve` returned before
/// this type existed (flat `default_*` metadata keys, arch caps fallback). `non_thinking` is
/// present only when the operator declared a `non_thinking_sampling` table; a single-arm
/// model resolves every mode to `thinking` and is byte-identical to before.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ModelSamplingDefaults {
    thinking: SamplingDefaults,
    non_thinking: Option<SamplingDefaults>,
}

impl ModelSamplingDefaults {
    fn resolve(metadata: Option<&OpenRouterModelMetadata>, caps: Option<&ModelCaps>) -> Self {
        ModelSamplingDefaults {
            thinking: SamplingDefaults::resolve(metadata, caps),
            // The non-thinking arm is the operator's declaration ALONE — no arch-caps
            // fallback and no field-by-field inheritance from the thinking arm. The two
            // arms are separate vendor programs; a field the vendor left out of one arm
            // falls to the API-standard default exactly like an undeclared flat key.
            non_thinking: metadata
                .and_then(|m| m.non_thinking_sampling.as_ref())
                .map(|arm| SamplingDefaults {
                    temperature: arm.temperature,
                    top_p: arm.top_p,
                    top_k: arm.top_k,
                    min_p: arm.min_p,
                    frequency_penalty: arm.frequency_penalty,
                    presence_penalty: arm.presence_penalty,
                    repetition_penalty: arm.repetition_penalty,
                }),
        }
    }

    /// THE arm-selection law: the request's RESOLVED thinking mode picks the arm.
    /// `NoThink` — produced by any off spelling (`reasoning_effort:"none"|"minimal"`,
    /// `enable_thinking:false`, `chat_template_kwargs.enable_thinking:false`,
    /// `reasoning:{enabled:false}`, `include_reasoning:false`, Anthropic
    /// `thinking.type:"disabled"`), by an operator `default_reasoning_effort = "none"`
    /// resolving an unset request, or by the response_format constraint forcing the
    /// think switch off — takes the non-thinking arm when one is declared. `Default`
    /// deliberately does NOT: it means "the template's own mode", and every model that
    /// carries a non-thinking arm today defaults thinking ON; a deployment whose unset
    /// case should be non-thinking says so with `default_reasoning_effort = "none"`,
    /// which resolves to `NoThink` upstream and lands here. Models without the arm
    /// return `thinking` for every mode — the exact pre-lane behavior.
    fn for_mode(&self, think: ThinkMode) -> &SamplingDefaults {
        match (think, &self.non_thinking) {
            (ThinkMode::NoThink, Some(non_thinking)) => non_thinking,
            _ => &self.thinking,
        }
    }

    /// A single-arm carrier for surfaces/tests that resolve without per-mode metadata —
    /// behaviorally the pre-lane `SamplingDefaults` value, on every mode.
    #[cfg(test)] // only test surfaces resolve without per-mode metadata today
    fn single(thinking: SamplingDefaults) -> Self {
        ModelSamplingDefaults {
            thinking,
            non_thinking: None,
        }
    }
}

/// The client's own sampling expression: `Some` = the client said this, `None` = the client said
/// nothing. Every surface funnels its body into this shape so there is exactly ONE place where
/// an omitted field becomes a number (standard-surface law: `/v1/completions`,
/// `/v1/chat/completions`, `/v1/messages` and `/v1/responses` must not disagree, and the way to
/// guarantee that is to give them one resolver rather than three matching ones).
#[derive(Debug, Clone, Copy, Default)]
struct ClientSampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
}

impl From<&CompletionReq> for ClientSampling {
    fn from(r: &CompletionReq) -> Self {
        ClientSampling {
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            min_p: r.min_p,
            frequency_penalty: r.frequency_penalty,
            presence_penalty: r.presence_penalty,
            repetition_penalty: r.repetition_penalty,
            seed: r.seed,
        }
    }
}

impl From<&ChatCompletionReq> for ClientSampling {
    fn from(r: &ChatCompletionReq) -> Self {
        ClientSampling {
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            min_p: r.min_p,
            frequency_penalty: r.frequency_penalty,
            presence_penalty: r.presence_penalty,
            repetition_penalty: r.repetition_penalty,
            seed: r.seed,
        }
    }
}

/// THE resolution law. Client value > vendor/operator default > API-standard default.
///
/// The one invariant that must never bend: an EXPLICIT `temperature: 0` produces true greedy,
/// because `Some(0.0)` short-circuits before any default is consulted. Greedy is a caller
/// decision and stays exactly reachable; it just stops being what an omitting client gets.
fn resolve_sampler_config(client: ClientSampling, defaults: &SamplingDefaults) -> SamplerConfig {
    sampler_config(
        client
            .temperature
            .or(defaults.temperature)
            .unwrap_or_else(default_temperature),
        client.top_k.or(defaults.top_k).unwrap_or(0),
        client.top_p.or(defaults.top_p).unwrap_or_else(one),
        client.min_p.or(defaults.min_p).unwrap_or(0.0),
        client
            .frequency_penalty
            .or(defaults.frequency_penalty)
            .unwrap_or(0.0),
        client
            .presence_penalty
            .or(defaults.presence_penalty)
            .unwrap_or(0.0),
        client
            .repetition_penalty
            .or(defaults.repetition_penalty)
            .unwrap_or_else(one),
        client.seed,
    )
}

#[derive(Serialize)]
struct CompletionResp {
    model: String,
    text: String,
    tokens: Vec<u32>,
    /// Worker stop reason. `Deadline` (lane/deadline-partial-20260826) means the request's
    /// `timeout_ms` cut generation and the text above is what had been produced — the native
    /// twin of the OpenAI shapes' `finish_reason: "error"`.
    stop_reason: String,
    /// Present ONLY on a deadline-cut partial, carrying the same message/code/metadata the
    /// OpenAI shapes put in their `error` object. Absent on every normal completion, so the
    /// shape is unchanged for them. Without this the native surface learned nothing
    /// actionable from a cut — flagged by review.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<serde_json::Value>,
    n_tokens: usize,
    /// worker-truth prompt accounting (prompt caching): total prompt tokens, and how many
    /// were served from cache (continuation pool / spec resume / cross-request prefix cache).
    prompt_tokens: usize,
    cached_tokens: usize,
    elapsed_s: f64,
}

/// OpenAI-schema usage object, shared by every response shape. `prompt_tokens_details.
/// cached_tokens` is the marketplace prompt-caching field (cache reads bill at a discount;
/// the value is worker-truth — tokens whose KV was resumed instead of computed).
/// `spec` (lane/accept-telemetry) is an ADDITIVE extension: this request's spec-decode
/// rounds/drafted/accepted + acceptance rate. Present only when the request actually ran
/// spec rounds — official SDKs ignore unknown usage fields (extra fields ok, existing
/// fields untouched), and spec-off responses are byte-identical to before.
fn usage_json(
    n_prompt: usize,
    n_tokens: usize,
    n_cached: usize,
    elapsed_s: f64,
    spec: Option<worker::SpecUsage>,
) -> serde_json::Value {
    let mut u = json!({
        "prompt_tokens": n_prompt,
        "completion_tokens": n_tokens,
        "total_tokens": n_prompt + n_tokens,
        "prompt_tokens_details": { "cached_tokens": n_cached },
        "elapsed_s": elapsed_s,
    });
    if let Some(sp) = spec {
        u["spec"] = json!({
            "rounds": sp.rounds,
            "drafted": sp.drafted,
            "accepted": sp.accepted,
            "acceptance_rate": if sp.drafted > 0 {
                sp.accepted as f64 / sp.drafted as f64 } else { 0.0 },
        });
    }
    u
}

// ---- OpenAI response envelope (serve-compat lane, 2026-08-03; gap-scan F1) ----
//
// The official `openai` SDKs pydantic-validate every response: `ChatCompletion` /
// `ChatCompletionChunk` REQUIRE `id: str` and `created: int`, so a response without them
// is rejected client-side before the caller ever sees the content. Every OpenAI-shape
// completion and every stream chunk therefore carries `id` + `created` +
// `system_fingerprint`; the id doubles as the `x-request-id` response header (vLLM
// convention, serving_engine.py) for support/tracing. The memra-native response shape
// (non-chat, MEMRA_COMPAT unset) is untouched — validation harnesses depend on it.

/// Backend-config fingerprint: `memra-<crate version>-<content id>`, baked by `build.rs`
/// from the crate version plus a digest of the workspace's compiled inputs. Together with
/// `seed`, responses are checkable for determinism across deploys — the OpenAI
/// `system_fingerprint` contract.
///
/// It is derived from file CONTENT, not from git history, and that is the whole point:
///
/// - **It cannot degrade to a label.** The old form was `concat!("memra-", <git sha>)`, and
///   a git failure inside darklanes' release container silently baked the literal
///   `unknown`. Prod served `system_fingerprint: memra-unknown` to every request for a
///   deploy generation, which also meant darklanes' `tools/check-claim-builds.mjs --live`
///   had nothing to verify published performance pins against. See `build.rs` for the
///   receipt chain.
/// - **It survives a history rewrite.** Rewriting commits changes every SHA while the bytes
///   of the tree stay put, so a fingerprint quoted in a published claim, a research
///   receipt, or a customer's own response keeps naming the same build afterwards.
///
/// Deliberately NOT in the value: a build timestamp. Two builds of the same source must
/// produce the same fingerprint, because `check-claim-builds` compares it for EQUALITY
/// against a published pin and a per-rebuild value would churn every pin. Build time is an
/// artifact-registry fact (the filename and the file's mtime), not an identity.
pub const SYSTEM_FINGERPRINT: &str = concat!(
    "memra-",
    env!("CARGO_PKG_VERSION"),
    "-",
    env!("MEMRA_BUILD_ID")
);

/// How `SYSTEM_FINGERPRINT`'s id was derived: `source-tree` (real) or `degraded`.
pub const BUILD_ID_SRC: &str = env!("MEMRA_BUILD_ID_SRC");

/// Why the id is degraded. Empty when it is not.
pub const BUILD_ID_NOTE: &str = env!("MEMRA_BUILD_ID_NOTE");

/// The build's git sha when the build could read a repo, else `unknown`. An EXTRA
/// provenance field: convenient, never the identity. A shipped binary outlives the commit it
/// was cut from, and after an authorized history rewrite the sha names nothing at all.
pub const BUILD_GIT_SHA: &str = env!("MEMRA_BUILD_SHA");

/// One line of build provenance, printed at boot by EVERY binary that links this server
/// (the stock bin and darklanes' deployment bin both enter through `serve_with`).
pub fn build_identity_line() -> String {
    format!("[server] build: {SYSTEM_FINGERPRINT} (id: {BUILD_ID_SRC}, git: {BUILD_GIT_SHA})")
}

/// 128 random-ish hex bits: two RandomState-seeded hashes over a process counter + time.
/// Uniqueness class (request ids), not crypto.
fn gen_hex128() -> String {
    use std::hash::{BuildHasher, Hasher};
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h1 = std::collections::hash_map::RandomState::new().build_hasher();
    h1.write_u64(n);
    h1.write_u64(t);
    let mut h2 = std::collections::hash_map::RandomState::new().build_hasher();
    h2.write_u64(t.rotate_left(17));
    h2.write_u64(n);
    format!("{:016x}{:016x}", h1.finish(), h2.finish())
}

/// One request's envelope identity: the completion `id` (`chatcmpl-…` chat, `cmpl-…`
/// text) + `created` unix seconds, shared by the response and every chunk of its stream.
#[derive(Clone)]
struct Envelope {
    id: String,
    created: u64,
}

impl Envelope {
    fn new(chat: bool) -> Self {
        Envelope {
            id: format!(
                "{}-{}",
                if chat { "chatcmpl" } else { "cmpl" },
                gen_hex128()
            ),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// The ledger identity of ONE admitted capture inside a multi-item capture request
    /// (`/v1/embeddings` with N inputs, `/v1/rerank` with N documents): `<parent id>.<index>`.
    ///
    /// Every capture runs the full admission sequence and opens its own receipt, so it is
    /// a separately priced request to the budget ledger. The ledger keys debits by request
    /// id as a REPLAY GUARD: a second debit under an already-debited id is swallowed when
    /// the amount matches and refused (`conflicting budget debits`) when it does not. N
    /// captures sharing the parent id therefore billed as one capture when their costs
    /// rounded equal and failed the whole request with HTTP 500 when they did not
    /// (darklanes research/fleet-consolidation-tx-20260902/INCIDENT-rerank-ledger-conflict.md,
    /// 2026-09-02: rerank documents of 80 and 81 prompt tokens at $0.05/1M -> debits 4 and 5).
    /// A distinct child id per capture makes each capture debit exactly once. The HTTP
    /// response and `x-request-id` keep the parent id; children nest under it by prefix
    /// (`starts_with("<parent>.")`, never the bare parent: hex ids carry no `.`, so the dotted
    /// prefix cannot alias another parent or another child) for reconciliation and log
    /// attribution.
    fn capture_child(&self, index: usize) -> Self {
        Envelope {
            id: format!("{}.{index}", self.id),
            created: self.created,
        }
    }

    /// Stamp the envelope fields onto one completion/chunk payload.
    fn stamp(&self, mut v: serde_json::Value) -> serde_json::Value {
        v["id"] = json!(self.id);
        v["created"] = json!(self.created);
        v["system_fingerprint"] = json!(SYSTEM_FINGERPRINT);
        v
    }
}

/// Attach the request id as the `x-request-id` response header.
fn with_request_id(id: &str, mut resp: Response) -> Response {
    if let Ok(v) = axum::http::HeaderValue::from_str(id) {
        resp.headers_mut()
            .insert(axum::http::HeaderName::from_static("x-request-id"), v);
    }
    resp
}

/// OpenAI-compat mapping (2026-07-05, serve-parity arc): the pi daily client speaks
/// `openai-completions` — POST /v1/completions with the OpenAI body, expecting
/// `{choices:[{text, finish_reason, index}], usage:{...}}` and, when streaming, OpenAI SSE
/// chunks (`data: {choices:[{text}]}` ... `data: [DONE]`). pi renders the chat template
/// CLIENT-side (thinkingFormat qwen-chat-template), so raw-prompt completions is the whole
/// contract. MEMRA_COMPAT=openai (default when MEMRA_API_KEY is set — the pi setup) switches the
/// response shape; the native memra shape stays default otherwise (validation harnesses use it).
fn openai_compat() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| match std::env::var("MEMRA_COMPAT").as_deref() {
        Ok("openai") => true,
        Ok(_) => false,
        Err(_) => std::env::var("MEMRA_API_KEY").is_ok(),
    })
}

/// PC-ISO (lane/pc-iso, 2026-08-02): extract the raw cache namespace for request builders —
/// the vLLM `cache_salt` design (research/cache-tools-20260802/REPORT.md §4): the explicit
/// `cache_salt` body field (OpenAI-compatible extension), else "" — the default
/// single-tenant namespace, byte-identical to pre-PC-ISO behavior. The HTTP handlers validate
/// this value with `validate_cache_namespace` before any Request reaches the worker. When a
/// keyring is configured (MEMRA_API_KEYS) the handlers wrap it in the tenant scope —
/// `tenant_namespace` -> `t:<tenant>\x1f<salt>` (lane/api-keys) — so per-key identity
/// DOES fold in now; without a keyring the validated raw form passes through unchanged.
/// Cross-request KV reuse (prefix cache, continuation pool, spec pool)
/// only ever matches entries with an IDENTICAL namespace, so the `cached_tokens` hit oracle
/// can only reveal the caller's own namespace's history (CacheProbe/PROMPTPEEK mitigation).
fn cache_namespace(cache_salt: &Option<String>) -> String {
    cache_salt.clone().unwrap_or_default()
}

const CACHE_SALT_MAX_BYTES: usize = 64;

fn validate_cache_namespace(
    cache_salt: &Option<String>,
    keyring_configured: bool,
) -> Result<String, &'static str> {
    let raw = cache_namespace(cache_salt);
    if raw.len() > CACHE_SALT_MAX_BYTES {
        return Err("cache_salt must be at most 64 bytes");
    }
    if !keyring_configured && raw.starts_with("t:") {
        return Err("cache_salt must not use the reserved t: prefix without a keyring");
    }
    if !raw
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'+' | b'/' | b'='))
    {
        return Err("cache_salt contains unsupported characters");
    }
    Ok(raw)
}

/// SESSION AFFINITY explicit tier (lane/session-affinity, 2026-08-05): the caller's own name
/// for this conversation, if it supplies one. A named conversation resumes its parked session
/// directly — no fingerprint guess needed. Accepted conventions, in priority order:
///   1. `session_id` body field — the explicit spelling.
///   2. `user` body field — OpenAI's own field; real clients already send a stable per-user
///      (often per-conversation) value here, so honoring it costs the caller nothing.
///   3. `x-session-id` request header — the convention proxies in front of vLLM/TGI use.
///      Body beats header: the body is the caller's own statement of identity, while a header can
///      be rewritten by an intermediary. Blank/whitespace values are treated as absent (a client
///      sending `"user": ""` must not collapse every conversation onto one session).
///
/// The key is NOT authoritative over tokens. It only NOMINATES a parked session for the exact
/// token-diff test in the worker (`affinity_match`), and only within the request's own
/// (model, cache_ns) pool — so a reused or guessed id can cost a wasted probe, never a wrong
/// resume and never cross-tenant reach.
fn affinity_key(
    session_id: &Option<String>,
    user: &Option<String>,
    headers: &axum::http::HeaderMap,
) -> Result<Option<String>, String> {
    let clean = |s: &str| -> Result<Option<String>, String> {
        let t = s.trim();
        if t.is_empty() {
            Ok(None)
        } else if t.len() > MAX_CLIENT_IDENTIFIER_BYTES {
            Err(format!(
                "session identity must be at most {MAX_CLIENT_IDENTIFIER_BYTES} UTF-8 bytes"
            ))
        } else if t.chars().any(char::is_control) {
            Err("session identity must not contain control characters".into())
        } else {
            Ok(Some(t.to_string()))
        }
    };
    if let Some(value) = session_id.as_deref()
        && let Some(value) = clean(value)?
    {
        return Ok(Some(value));
    }
    if let Some(value) = user.as_deref()
        && let Some(value) = clean(value)?
    {
        return Ok(Some(value));
    }
    match headers.get("x-session-id") {
        Some(value) => clean(
            value
                .to_str()
                .map_err(|_| "x-session-id must contain visible ASCII or UTF-8 text")?,
        ),
        None => Ok(None),
    }
}

fn validate_client_identifier(value: Option<&str>, name: &str) -> Result<(), String> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.len() > MAX_CLIENT_IDENTIFIER_BYTES {
        return Err(format!(
            "{name} must be at most {MAX_CLIENT_IDENTIFIER_BYTES} UTF-8 bytes"
        ));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{name} must not contain control characters"));
    }
    Ok(())
}

/// OpenAI error body: `{"error": {"message", "type", "param", "code"}}` — the object
/// shape every OpenAI SDK parses (gap-scan F1; the old `{"error": "<string>"}` made
/// clients show a blank error). `type` follows the OpenAI vocabulary:
/// invalid_request_error / authentication_error / not_found_error / server_error.
fn error_body(
    message: &str,
    etype: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> serde_json::Value {
    json!({ "error": {
        "message": message,
        "type": etype,
        "param": param,
        "code": code,
    } })
}

fn error_response(status: StatusCode, message: &str, etype: &str, param: Option<&str>) -> Response {
    error_response_coded(status, message, etype, param, None)
}

/// Same, with an explicit OpenAI `code`. Handler-layer refusals (auth, lane, request parsing)
/// land here; engine-produced faults land in `engine_error_response`. Both attach
/// `x-should-retry: false` on a 4xx that retrying the identical bytes cannot fix, so the two
/// halves of the surface behave identically to a client that retries by status alone.
fn error_response_coded(
    status: StatusCode,
    message: &str,
    etype: &str,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let mut resp = (status, Json(error_body(message, etype, param, code))).into_response();
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

fn bad_request(message: &str, param: Option<&str>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        param,
    )
}

// ---- engine-fault taxonomy -> HTTP (lane/serve-hardening, G6) --------------------------
//
// WHAT THIS REPLACES. Every worker failure — CUDA errors, VRAM exhaustion, admission sheds,
// tokenizer failures, graph faults — used to funnel into ONE line: `bad_request(&msg, None)`,
// i.e. HTTP 400 invalid_request_error. That is wrong in both directions and both directions
// cost money:
//   * a client SDK never retries a 400 (openai-python retries 408/409/429/>=500 only), so a
//     transient capacity blip became a hard user-visible failure with no retry;
//   * a router cannot tell "your request was malformed" from "my GPU fell over", so it keeps
//     sending traffic to a broken box instead of failing over.
// The class now comes from the PRODUCER (worker.rs::EngineError), not from re-guessing at the
// HTTP layer, with exactly one deliberate text rule (`is_cuda_oom` -> Overloaded).
//
// THE RETRY CONTRACT, verified against the client code rather than the docs:
//   * `Retry-After` is INTEGER seconds (RFC 9110 §10.2.3 delay-seconds — a float here is
//     simply unparseable), and openai-python ABANDONS the retry entirely if the value exceeds
//     its MAX_RETRY_AFTER_DELAY of 120 s. litellm honors the header only for 0 < v <= 60.
//     So every value memra emits is an integer and <= 60.
//   * `retry-after-ms` is read FIRST by openai-python, which lets us express sub-second
//     backoff to SDKs that support it while the integer header stays correct for everyone
//     else. Both are sent; they agree.
//   * `x-should-retry: false` is openai-python's explicit override, used where retrying is
//     provably pointless (a 400-class fault), so a client that retries by status alone does
//     not hammer a request that can never succeed.
const RETRY_AFTER_S_RATE_LIMIT: u64 = 2; // QoS shed: the lane's own budget window
const RETRY_AFTER_S_OVERLOADED: u64 = 5; // VRAM/capacity: needs a session to finish first

/// Status + OpenAI `type` + `code` for one engine error class.
fn class_http(class: worker::ErrClass) -> (StatusCode, &'static str, Option<&'static str>) {
    use worker::ErrClass as C;
    match class {
        C::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error", None),
        C::ContextLength => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("context_length_exceeded"),
        ),
        C::ModelNotFound => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("model_not_found"),
        ),
        C::RateLimit => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            Some("rate_limit_exceeded"),
        ),
        C::Overloaded => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            Some("overloaded"),
        ),
        C::Engine => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            Some("engine_error"),
        ),
    }
}

/// Retry-After seconds for a class, or None when retrying cannot help.
fn class_retry_after_s(class: worker::ErrClass) -> Option<u64> {
    use worker::ErrClass as C;
    match class {
        C::RateLimit => Some(RETRY_AFTER_S_RATE_LIMIT),
        C::Overloaded => Some(RETRY_AFTER_S_OVERLOADED),
        // An engine fault is not time-bounded: this process may need to be restarted. Say
        // nothing rather than promise a window we cannot honor — the SDK's own exponential
        // backoff (500s are retryable by default) is the honest behavior here.
        C::Engine | C::InvalidRequest | C::ContextLength | C::ModelNotFound => None,
    }
}

/// The JSON body for an engine error, shared by the blocking and the streaming paths so a
/// client sees the SAME object either way.
fn engine_error_body(e: &worker::EngineError) -> serde_json::Value {
    let (_, etype, code) = class_http(e.class);
    error_body(&e.message, etype, e.param, code)
}

/// Full HTTP response for an engine error: status, OpenAI body, and the retry headers.
/// A producer-computed `retry_after_s` (D2 gap G6: the predictive-admission reject's
/// earliest predicted in-flight completion) overrides the per-class default; both take
/// the SAME `retry_contract_response` path, so the header pair stays byte-compatible
/// with the shed contract regardless of who chose the value.
fn engine_error_response(e: &worker::EngineError) -> Response {
    engine_error_response_with_retry_after(
        e,
        e.retry_after_s.or_else(|| class_retry_after_s(e.class)),
    )
}

fn engine_error_response_with_retry_after(
    e: &worker::EngineError,
    retry_after_s: Option<u64>,
) -> Response {
    let (status, _, _) = class_http(e.class);
    let resp = (status, Json(engine_error_body(e))).into_response();
    retry_contract_response(resp, retry_after_s)
}

/// Apply memra's retry headers to any response body.
fn retry_contract_response(mut resp: Response, retry_after_s: Option<u64>) -> Response {
    let status = resp.status();
    let h = resp.headers_mut();
    match retry_after_s {
        Some(secs) => {
            // Integer seconds in the SDK-honored 1..=60 window (see the contract note above).
            let secs = secs.clamp(1, 60);
            if let Ok(v) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                h.insert(axum::http::header::RETRY_AFTER, v);
            }
            if let Ok(v) = axum::http::HeaderValue::from_str(&(secs * 1000).to_string()) {
                h.insert("retry-after-ms", v);
            }
        }
        None if status.is_client_error() => {
            // A malformed request, an unknown model, an over-long prompt: retrying the
            // identical bytes cannot succeed. Say so explicitly.
            h.insert(
                "x-should-retry",
                axum::http::HeaderValue::from_static("false"),
            );
        }
        None => {}
    }
    resp
}

fn worker_unavailable_response() -> Response {
    engine_error_response_with_retry_after(
        &worker::EngineError::overloaded("worker unavailable"),
        Some(worker::WORKER_RESPAWN_BACKOFF_BASE_S),
    )
}

fn stop_reason_to_finish(r: &str) -> &'static str {
    match r {
        "Eos" | "Callback" => "stop",
        "MaxNew" | "ContextFull" => "length",
        _ => "stop",
    }
}

// ---- tools surface helpers (serve-tools lane, 2026-08-02) ----

/// Flatten an OpenAI `content` value to text: string, null (-> ""), or `{type:"text"}` parts.
fn content_to_text(v: &serde_json::Value) -> Result<String, String> {
    match v {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") | None => match p.get("text").and_then(|t| t.as_str()) {
                        Some(t) => out.push_str(t),
                        None => return Err("content part has no text field".into()),
                    },
                    Some(other) => {
                        return Err(format!(
                            "unsupported content part type {other:?} (text only)"
                        ));
                    }
                }
            }
            Ok(out)
        }
        _ => Err("content must be a string, null, or an array of text parts".into()),
    }
}

/// Vision PLACEMENT admissibility, published by the worker at boot for EVERY vision family
/// (worker.rs `vision_placement_admissible`) and read at every MEDIA PART below
/// (`vision_placement_admits`), never by the family switches: those route the content
/// walkers, and step37's text-separator law lives only in its walker, so folding the
/// placement into a switch would move prompt bytes on text-only traffic (revuto, #46).
///
/// A loaded tower is not sufficient to serve images: the overlay's rows have to be resident
/// in the CUDA context of the engine that embeds (pp stage 0 under a per-stage-stream ppN
/// split), and `MEMRA_VISION_OVERLAY_PUBLISH=0` forbids putting them there. Deciding that
/// ONCE at boot and refusing at the waist is what lane/glm53-vision-ppn shipped for glm5 —
/// but the door it reads is the first line of `EmbedOverlay::new_published` for all four
/// families, so a gemma4 / qwen-VL / step37 deployment with the same pin (or a mistyped door
/// value) booted clean and 500'd MID-PREFILL on a live request, the exact failure removed for
/// glm5. step37 serves vision in production, which made that a live exposure (memra #25).
///
/// `true` until the worker publishes: readiness gates customer traffic behind the worker's
/// spawn, and a unit test that never spawns a worker must see the pre-lane program.
pub(crate) static VISION_PLACEMENT_SERVING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

fn vision_placement_serving() -> bool {
    VISION_PLACEMENT_SERVING.load(std::sync::atomic::Ordering::Acquire)
}

/// The one placement gate every media-accepting arm passes BEFORE it plans anything: an
/// `image_url`/`video_url` part on a placement that cannot deliver an overlay to embedding
/// intake refuses with a named 400 here, at the waist, instead of 500ing mid-prefill. Pure so
/// its contract is unit-tested without touching process state; `vision_placement_admits` is
/// the live wrapper that feeds the worker's decision in. `kind` is `"image"` or `"video"`.
fn vision_media_admissible(placement: bool, kind: &str) -> Result<(), String> {
    if placement {
        Ok(())
    } else {
        Err(format!(
            "{kind} input is not enabled on this deployment (vision overlay placement \
             inadmissible at boot: see the worker's IMAGE INPUT DISABLED line)"
        ))
    }
}

fn vision_placement_admits(kind: &str) -> Result<(), String> {
    vision_media_admissible(vision_placement_serving(), kind)
}

/// Vision enablement (lane/vision): the worker loads the tower iff MEMRA_VISION_DIR is
/// set, so the HTTP layer accepts image parts under exactly the same condition. Armed-only
/// by design: the placement half is applied per media part (`vision_placement_admits`).
fn vision_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_VISION_DIR").is_ok()
            && std::env::var("MEMRA_VISION").as_deref() != Ok("0")
    })
}

/// Gemma-4 vision seam (lane/gemma-vision): a deployment serves ONE vision family
/// (one model per GPU), so this process-wide switch decides which placeholder + prep
/// the image parts take. Default OFF — gemma image input refuses until an operator
/// sets MEMRA_GEMMA_VISION=1 with a gemma4v mmproj at MEMRA_GEMMA_MMPROJ.
fn gemma_vision_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_GEMMA_VISION").as_deref() == Ok("1")
            && std::env::var("MEMRA_GEMMA_MMPROJ").is_ok()
    })
}

/// glm5_next vision serving decision, published by the worker at spawn (worker.rs tower
/// load) and read by the HTTP intake. DEFAULT ON (owner order 2026-08-30,
/// lane/glm5-vision-default-on): true iff a glm5 tower actually loaded — from the served
/// glm5_next artifact's own `model.visual.*` tensors by default, from
/// MEMRA_GLM5_VISION_DIR when set; false when the artifact carries no tower or
/// MEMRA_GLM5_VISION=0 (the rollback seam). Not an env read: the intake must route image
/// parts to the glm5 planner exactly when the worker can prime them. Already folds in the
/// placement decision (`VISION_PLACEMENT_SERVING`): the worker stores
/// `tower loaded && placement admissible`.
pub(crate) static GLM5_VISION_SERVING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// glm5_next vision seam (lane/glm5-vision): same one-family-per-deployment law as the
/// gemma seam. See `GLM5_VISION_SERVING` for the decision's source of truth.
fn glm5_vision_enabled() -> bool {
    GLM5_VISION_SERVING.load(std::sync::atomic::Ordering::Acquire)
}

/// step37 vision seam (lane/step37-vision): same one-vision-family-per-process law as
/// the two above. The worker loads the perception_encoder tower from the serving
/// artifact's own directory iff MEMRA_STEP_VISION_DIR is set (the vision tensors live
/// unquantized inside the checkpoint), so the HTTP layer accepts image parts under
/// exactly the same condition; MEMRA_STEP_VISION=0 is the kill switch (both sides).
/// Armed-only by design: this switch selects the step content walker, whose TEXT separator
/// law must not move with the placement; image parts pass `vision_placement_admits` inside.
fn step_vision_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_STEP_VISION_DIR").is_ok()
            && std::env::var("MEMRA_STEP_VISION").as_deref() != Ok("0")
    })
}

/// Per-request image cap (v1 envelope; the context cap bounds total vision tokens).
const VISION_MAX_IMAGES: usize = 8;

/// Bound the host memory retained by decoded vision patches. The previous per-image pixel cap
/// allowed eight Qwen images to materialize roughly 3 GiB of f32 patch rows before the HTTP
/// concurrency gate ran. A process-wide reservation keeps both one request and concurrent
/// requests within a finite budget; the request slot remains a separate serving/QoS control.
pub(crate) const MAX_VISION_PATCH_BYTES: usize = 1 << 30; // 1 GiB
static VISION_PATCH_BYTES_IN_USE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// GIF/video preprocessing is bounded separately from request admission because its decoder must
/// discover sampled frames and timestamps while constructing the prompt plan. Serializing this
/// phase prevents multiple requests from simultaneously holding their transient RGB canvases.
pub(crate) static VISION_PREPROCESS_SEMAPHORE: tokio::sync::Semaphore =
    tokio::sync::Semaphore::const_new(1);

// Axum handlers use `Response` as their rejection type. Boxing this rare 429/503 response
// would add allocation and conversion at every `?` boundary for no reduction in retained state.
#[allow(clippy::result_large_err)]
pub(crate) fn try_vision_preprocess(
    required: bool,
) -> Result<Option<tokio::sync::SemaphorePermit<'static>>, Response> {
    try_vision_preprocess_with(required, &VISION_PREPROCESS_SEMAPHORE)
}

#[allow(clippy::result_large_err)]
fn try_vision_preprocess_with(
    required: bool,
    semaphore: &'static tokio::sync::Semaphore,
) -> Result<Option<tokio::sync::SemaphorePermit<'static>>, Response> {
    if !required {
        return Ok(None);
    }
    match semaphore.try_acquire() {
        Ok(permit) => Ok(Some(permit)),
        Err(tokio::sync::TryAcquireError::NoPermits) => Err(retry_contract_response(
            error_response_coded(
                StatusCode::TOO_MANY_REQUESTS,
                "vision preprocessing is busy",
                "rate_limit_error",
                Some("messages"),
                Some("vision_preprocess_busy"),
            ),
            Some(BODY_ADMISSION_RETRY_AFTER_S),
        )),
        Err(tokio::sync::TryAcquireError::Closed) => Err(error_response_coded(
            StatusCode::SERVICE_UNAVAILABLE,
            "vision preprocessing is unavailable",
            "server_error",
            Some("messages"),
            Some("vision_preprocess_unavailable"),
        )),
    }
}

pub(crate) struct VisionMemoryPermit {
    bytes: usize,
}

#[derive(Debug)]
pub(crate) enum VisionMemoryError {
    Request(String),
    Capacity(String),
}

impl Drop for VisionMemoryPermit {
    fn drop(&mut self) {
        if self.bytes != 0 {
            VISION_PATCH_BYTES_IN_USE.fetch_sub(self.bytes, std::sync::atomic::Ordering::AcqRel);
        }
    }
}

fn try_reserve_vision_memory(
    bytes: usize,
) -> Result<Option<VisionMemoryPermit>, VisionMemoryError> {
    if bytes == 0 {
        return Ok(None);
    }
    if bytes > MAX_VISION_PATCH_BYTES {
        return Err(VisionMemoryError::Request(format!(
            "vision preprocessing requires {bytes} bytes of patch memory, exceeding the {} MiB request limit",
            MAX_VISION_PATCH_BYTES / (1024 * 1024)
        )));
    }
    let mut in_use = VISION_PATCH_BYTES_IN_USE.load(std::sync::atomic::Ordering::Acquire);
    loop {
        let Some(next) = in_use.checked_add(bytes) else {
            return Err(VisionMemoryError::Capacity(
                "vision patch memory reservation overflowed".into(),
            ));
        };
        if next > MAX_VISION_PATCH_BYTES {
            return Err(VisionMemoryError::Capacity(format!(
                "vision preprocessing is at capacity ({} MiB reserved; request needs {} MiB)",
                in_use / (1024 * 1024),
                bytes / (1024 * 1024)
            )));
        }
        match VISION_PATCH_BYTES_IN_USE.compare_exchange_weak(
            in_use,
            next,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => return Ok(Some(VisionMemoryPermit { bytes })),
            Err(actual) => in_use = actual,
        }
    }
}

pub(crate) fn vision_memory_error_response(
    error: VisionMemoryError,
    param: Option<&str>,
) -> Response {
    match error {
        VisionMemoryError::Request(message) => bad_request(&message, param),
        VisionMemoryError::Capacity(message) => retry_contract_response(
            error_response_coded(
                StatusCode::SERVICE_UNAVAILABLE,
                &message,
                "server_error",
                None,
                Some("vision_memory_busy"),
            ),
            Some(RETRY_AFTER_S_OVERLOADED),
        ),
    }
}

/// One qwen vision unit as PLANNED at request build — pre-admission, header-only
/// (hermes decode-bomb finding, fixed 2026-08-23). `Still` carries the raw bytes plus
/// the grid its header plans to; the pixels decode in `decode_pending_vision`, AFTER
/// budget admission. `Video` carries a metadata-only GIF plan (sampled timestamps and grids);
/// frame pixels decode in `decode_pending_vision` after admission as well.
enum PendingVisionUnit {
    Still {
        bytes: Vec<u8>,
        gh: usize,
        gw: usize,
    },
    Video {
        bytes: Vec<u8>,
        groups: Vec<memra_engine::vision_pre::PlannedVideoGroup>,
        video: usize,
    },
}

/// The gemma twin of `PendingVisionUnit::Still` (gemma has no video input).
struct PendingGemmaImage {
    bytes: Vec<u8>,
    gw: usize,
    gh: usize,
}

/// The glm5_next twin (lane/glm5-vision). Video arms are censused but NOT served —
/// out of scope for the lane; `video_url` on a glm5 deployment refuses loudly.
struct PendingGlm5Image {
    bytes: Vec<u8>,
    gh: usize,
    gw: usize,
}

/// The step37 twin: header-planned tiling (crop count + newline mask) awaiting its
/// post-admission pixel decode. step37 has no video input either.
struct PendingStepImage {
    bytes: Vec<u8>,
    plan: memra_engine::vision_step::StepImagePlan,
}

/// step37 arm of `content_to_text_vision` (fires only when `step_vision_enabled()`).
/// Two vendor laws live here and nowhere else (chat_template.jinja at the pinned rev,
/// `render_message_content`): adjacent TEXT parts join with ONE space, and an image
/// part resets that separator (text directly after an image abuts it). Each image
/// renders as its exact expansion — the processor law, crops FIRST then the main view:
/// `<patch_start>` + 81 pads + `<patch_end>` (+ `<patch_newline>` per full tile row,
/// except a trailing one), then `<im_start>` + 169 pads + `<im_end>`. The worker
/// re-derives the runs from the TOKENIZED prompt and aligns them with `step_images`,
/// so user text faking pad tokens fails validation loudly. Data URIs only (SSRF off).
fn content_to_text_vision_step(
    v: &serde_json::Value,
    step_images: &mut Vec<PendingStepImage>,
) -> Result<String, String> {
    use memra_engine::vision_step::{SV_MAIN_ROWS, SV_TILE_ROWS};
    let parts = match v {
        serde_json::Value::Array(parts) => parts,
        _ => return content_to_text(v),
    };
    let mut out = String::new();
    let mut needs_sep = false;
    for p in parts {
        match p.get("type").and_then(|t| t.as_str()) {
            Some("text") | None => match p.get("text").and_then(|t| t.as_str()) {
                Some(t) => {
                    if needs_sep {
                        out.push(' ');
                    }
                    out.push_str(t);
                    needs_sep = true;
                }
                None => return Err("content part has no text field".into()),
            },
            Some("image_url") => {
                vision_placement_admits("image")?;
                let url = p
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            u.as_str()
                        } else {
                            u.get("url").and_then(|x| x.as_str())
                        }
                    })
                    .ok_or("image_url part has no url")?;
                if !url.starts_with("data:") {
                    return Err(
                        "image_url must be a base64 data URI (http(s) fetch is disabled)".into(),
                    );
                }
                if step_images.len() >= VISION_MAX_IMAGES {
                    return Err(format!("too many images (max {VISION_MAX_IMAGES})"));
                }
                // PLAN, don't decode (hermes decode-bomb law): the expansion derives
                // from HEADER dims; the canvas expands only after budget admission
                // (decode_pending_vision).
                let bytes = memra_engine::vision_pre::decode_data_uri(url)
                    .map_err(|e| format!("image {}: {e}", step_images.len() + 1))?;
                let plan = memra_engine::vision_step::step_plan_image(&bytes)
                    .map_err(|e| format!("image {}: {e}", step_images.len() + 1))?;
                for i in 0..plan.n_tiles {
                    out.push_str("<patch_start>");
                    for _ in 0..SV_TILE_ROWS {
                        out.push_str("<im_patch>");
                    }
                    out.push_str("<patch_end>");
                    if plan.newline_mask[i] {
                        out.push_str("<patch_newline>");
                    }
                }
                out.push_str("<im_start>");
                for _ in 0..SV_MAIN_ROWS {
                    out.push_str("<im_patch>");
                }
                out.push_str("<im_end>");
                step_images.push(PendingStepImage { bytes, plan });
                needs_sep = false;
            }
            Some("video_url") => {
                return Err("step37 has no video input (image-only processor)".into());
            }
            Some(other) => {
                return Err(format!("unsupported content part type {other:?}"));
            }
        }
    }
    Ok(out)
}

/// `content_to_text` twin that also accepts `image_url` parts: each image is PLANNED
/// here (header dims -> pre-decode pixel admission -> grid) and renders as its exact pad
/// run — `<|vision_start|>` + `<|image_pad|>` x n_tokens + `<|vision_end|>` — at its
/// position in the part order; the pixel decode itself runs after budget admission
/// (`decode_pending_vision`). The worker re-derives the runs from the TOKENIZED prompt
/// and aligns them 1:1 with `images`, so user text faking pad tokens fails validation
/// loudly. v1 posture: data URIs only — http(s) fetch stays off (SSRF), video parts
/// follow images.
fn content_to_text_vision(
    v: &serde_json::Value,
    images: &mut Vec<PendingVisionUnit>,
    gemma_images: &mut Vec<PendingGemmaImage>,
    glm5_images: &mut Vec<PendingGlm5Image>,
    step_images: &mut Vec<PendingStepImage>,
    next_video: &mut usize,
) -> Result<String, String> {
    // step37 deployments take their own walker: its placeholder expansion AND its
    // text-part separator law come from the step template, and both differ from the
    // qwen/gemma arms below. Fires only when the operator armed the step seam.
    if step_vision_enabled() {
        return content_to_text_vision_step(v, step_images);
    }
    let parts = match v {
        serde_json::Value::Array(parts) => parts,
        _ => return content_to_text(v),
    };
    let mut out = String::new();
    for p in parts {
        match p.get("type").and_then(|t| t.as_str()) {
            Some("text") | None => match p.get("text").and_then(|t| t.as_str()) {
                Some(t) => out.push_str(t),
                None => return Err("content part has no text field".into()),
            },
            Some("image_url") if glm5_vision_enabled() => {
                let url = p
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            u.as_str()
                        } else {
                            u.get("url").and_then(|x| x.as_str())
                        }
                    })
                    .ok_or("image_url part has no url")?;
                if !url.starts_with("data:") {
                    return Err(
                        "image_url must be a base64 data URI (http(s) fetch is disabled)".into(),
                    );
                }
                if glm5_images.len() >= VISION_MAX_IMAGES {
                    return Err(format!("too many images (max {VISION_MAX_IMAGES})"));
                }
                // PLAN, don't decode (hermes decode-bomb law): header dims -> pre-decode
                // pixel admission -> grid; the placeholder run derives from the grid and
                // the canvas expands only after budget admission (decode_pending_vision).
                let bytes = memra_engine::vision_pre::decode_data_uri(url)
                    .map_err(|e| format!("image {}: {e}", glm5_images.len() + 1))?;
                let (gh, gw) = memra_engine::vision_glm5::glm5_plan_image(&bytes)
                    .map_err(|e| format!("image {}: {e}", glm5_images.len() + 1))?;
                // glm5_next placeholder run: <|begin_of_image|> + n x <|image|> +
                // <|end_of_image|> — the upstream Glm5NextProcessor.replace_image_token
                // expansion, rendered here so the tokenized prompt matches upstream.
                out.push_str("<|begin_of_image|>");
                for _ in 0..memra_engine::vision_glm5::n_merged_for_grid(gh, gw) {
                    out.push_str("<|image|>");
                }
                out.push_str("<|end_of_image|>");
                glm5_images.push(PendingGlm5Image { bytes, gh, gw });
            }
            Some("video_url") if glm5_vision_enabled() => {
                return Err(
                    "glm5 video input is not served (tensor census only; image input is the \
                     supported surface)"
                        .into(),
                );
            }
            Some("image_url") if gemma_vision_enabled() => {
                vision_placement_admits("image")?;
                let url = p
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            u.as_str()
                        } else {
                            u.get("url").and_then(|x| x.as_str())
                        }
                    })
                    .ok_or("image_url part has no url")?;
                if !url.starts_with("data:") {
                    return Err(
                        "image_url must be a base64 data URI (http(s) fetch is disabled)".into(),
                    );
                }
                if gemma_images.len() >= VISION_MAX_IMAGES {
                    return Err(format!("too many images (max {VISION_MAX_IMAGES})"));
                }
                // PLAN, don't decode (hermes decode-bomb finding, fixed 2026-08-23): the
                // pad run derives from HEADER dims + the pre-decode pixel admission; the
                // canvas expands only after budget admission (decode_pending_vision).
                let bytes = memra_engine::vision_gemma::gemma_decode_data_uri(url)
                    .map_err(|e| format!("image {}: {e}", gemma_images.len() + 1))?;
                let (gw, gh) = memra_engine::vision_gemma::gemma_plan_image(&bytes)
                    .map_err(|e| format!("image {}: {e}", gemma_images.len() + 1))?;
                // gemma-4 placeholder: <|image> + n_soft * <|image|> + <image|>
                out.push_str("<|image>");
                for _ in 0..memra_engine::vision_gemma::n_soft_for_grid(gw, gh) {
                    out.push_str("<|image|>");
                }
                out.push_str("<image|>");
                gemma_images.push(PendingGemmaImage { bytes, gw, gh });
            }
            Some("image_url") => {
                if !vision_enabled() {
                    return Err("image input is not enabled on this deployment".into());
                }
                vision_placement_admits("image")?;
                let url = p
                    .get("image_url")
                    .and_then(|u| {
                        if u.is_string() {
                            u.as_str()
                        } else {
                            u.get("url").and_then(|x| x.as_str())
                        }
                    })
                    .ok_or("image_url part has no url")?;
                if !url.starts_with("data:") {
                    return Err(
                        "image_url must be a base64 data URI (http(s) fetch is disabled)".into(),
                    );
                }
                if images
                    .iter()
                    .filter(|u| matches!(u, PendingVisionUnit::Still { .. }))
                    .count()
                    >= VISION_MAX_IMAGES
                {
                    return Err(format!("too many images (max {VISION_MAX_IMAGES})"));
                }
                // PLAN, don't decode (hermes decode-bomb finding, fixed 2026-08-23):
                // header dims -> pre-decode pixel admission -> grid; the pad run derives
                // from the grid, and the canvas expands only after budget admission
                // (decode_pending_vision).
                let bytes = memra_engine::vision_pre::decode_data_uri(url)
                    .map_err(|e| format!("image {}: {e}", images.len() + 1))?;
                let (gh, gw) = memra_engine::vision_pre::plan_image_bytes(&bytes)
                    .map_err(|e| format!("image {}: {e}", images.len() + 1))?;
                out.push_str("<|vision_start|>");
                for _ in 0..memra_engine::vision_pre::n_tokens_for_grid(gh, gw) {
                    out.push_str("<|image_pad|>");
                }
                out.push_str("<|vision_end|>");
                images.push(PendingVisionUnit::Still { bytes, gh, gw });
            }
            Some("video_url") if gemma_vision_enabled() => {
                return Err("gemma-4 has no video input (image-only projector)".into());
            }
            Some("video_url") => {
                if !vision_enabled() {
                    return Err("video input is not enabled on this deployment".into());
                }
                vision_placement_admits("video")?;
                let url = p
                    .get("video_url")
                    .and_then(|u| {
                        if u.is_string() {
                            u.as_str()
                        } else {
                            u.get("url").and_then(|x| x.as_str())
                        }
                    })
                    .ok_or("video_url part has no url")?;
                if !url.starts_with("data:") {
                    return Err(
                        "video_url must be a base64 data URI (http(s) fetch is disabled)".into(),
                    );
                }
                if *next_video >= 2 {
                    return Err("too many videos (max 2)".into());
                }
                // v1 container: animated GIF (metadata planned here; frames decoded after
                // admission, in-process, with no ffmpeg dependency).
                let bytes = memra_engine::vision_pre::decode_data_uri(url)?;
                let vid = memra_engine::vision_pre::plan_video_gif(&bytes)
                    .map_err(|e| format!("video: {e}"))?;
                let vidx = *next_video;
                *next_video += 1;
                // HF Qwen3VL placeholder: `<t.t seconds>` + one pad run PER temporal group
                for group in &vid.groups {
                    out.push_str(&format!("<{:.1} seconds>", group.timestamp));
                    out.push_str("<|vision_start|>");
                    for _ in 0..memra_engine::vision_pre::n_tokens_for_grid(group.gh, group.gw) {
                        out.push_str("<|video_pad|>");
                    }
                    out.push_str("<|vision_end|>");
                }
                // Only metadata is retained in the plan; frame pixels are decoded after budget,
                // memory, and request-slot admission in `decode_pending_vision`.
                images.push(PendingVisionUnit::Video {
                    bytes,
                    groups: vid.groups,
                    video: vidx,
                });
            }
            Some(other) => {
                return Err(format!("unsupported content part type {other:?}"));
            }
        }
    }
    Ok(out)
}

/// Render a JSON value the way the reference template's `tojson` does (python json.dumps:
/// `", "` / `": "` separators, insertion-order keys — serde_json preserve_order — non-ASCII
/// left raw). The tools block is prompt bytes, so the training-time convention is the law.
fn pyjson(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::Object(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::Value::String(k.clone()).to_string());
                out.push_str(": ");
                pyjson(val, out);
            }
            out.push('}');
        }
        serde_json::Value::Array(a) => {
            out.push('[');
            for (i, val) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                pyjson(val, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn pyjson_str(v: &serde_json::Value) -> String {
    let mut s = String::new();
    pyjson(v, &mut s);
    s
}

/// Sampler wiring shared by both bodies (gap-scan F3): the penalties existed in
/// SamplerConfig end-to-end (host sampler + spec rejection-sampling verify) — this is
/// pure request-struct plumbing. Every serving path uses the same bounded history window:
/// speculative sampling already caps its O(n²) history form at `PEN_WINDOW_MAX`, so the host
/// and sparse-device paths must use that exact bound too. Otherwise a spec-to-plain demotion
/// changes penalty logits mid-request (Hermes `da99e50ec4750599`).
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn sampler_config(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
    seed: Option<u64>,
) -> SamplerConfig {
    let penalties_on =
        frequency_penalty != 0.0 || presence_penalty != 0.0 || repetition_penalty != 1.0;
    SamplerConfig {
        temperature,
        top_k,
        top_p,
        min_p,
        penalty_last_n: if penalties_on {
            memra_engine::spec::PEN_WINDOW_MAX
        } else {
            0
        },
        penalty_repeat: repetition_penalty,
        penalty_freq: frequency_penalty,
        penalty_present: presence_penalty,
        // Omitted seed => fresh entropy per request (dogfood F4). An explicit seed — including
        // an explicit 0 — is honored exactly, so every determinism gate keeps its behavior.
        seed: seed.unwrap_or_else(fresh_seed),
    }
}

/// Non-zero per-request entropy for seed-omitting clients. Nanosecond clock mixed with a
/// process-lifetime counter through SplitMix64's finalizer: two requests in the same
/// nanosecond tick (batched arrivals) still get distinct streams, which a bare clock read
/// would not guarantee. Not crypto — this only has to avoid replaying one stream forever.
fn fresh_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut z = nanos
        .wrapping_add(n.wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    // seed 0 is a legal explicit value but a poor accidental one; keep it reachable only
    // when the caller asks for it.
    if z == 0 { 0x9E3779B97F4A7C15 } else { z }
}

/// Honesty gate (gap-scan F4): semantic params we cannot honor are explicit 400s with the
/// offending param named — never silent downgrades (a client sending response_format:
/// json_object would get unvalidated free text and no error). Cosmetic fields (`user`,
/// `stream_options`) stay accept-and-ignore.
fn reject_unsupported(fields: &[(&str, bool, &str)]) -> Result<(), (String, String)> {
    for (param, present, why) in fields {
        if *present {
            return Err((format!("{param} is not supported{why}"), param.to_string()));
        }
    }
    Ok(())
}

#[derive(PartialEq)]
enum ToolChoice {
    Auto,
    None,
}

fn parse_tool_choice(v: &Option<serde_json::Value>) -> Result<ToolChoice, String> {
    match v {
        None | Some(serde_json::Value::Null) => Ok(ToolChoice::Auto),
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "auto" => Ok(ToolChoice::Auto),
            "none" => Ok(ToolChoice::None),
            "required" => Err("tool_choice \"required\" is not supported (no constrained \
                               decoding); use \"auto\""
                .into()),
            other => Err(format!("bad tool_choice {other:?} (auto|none)")),
        },
        Some(serde_json::Value::Object(_)) => {
            Err("named-function tool_choice is not supported; use \"auto\"".into())
        }
        Some(other) => Err(format!("bad tool_choice: {other}")),
    }
}

/// Map OpenAI `reasoning_effort` / OpenRouter `reasoning` onto the model's native thinking
/// control — ONE serve surface, per-arch mechanism (owner directive 2026-08-07: every
/// supported model is a thinking model).
///
/// The OpenAI/OpenRouter convention for reasoning-capable models: `low|medium|high` all mean
/// reasoning ON at that budget; `none|minimal` request (near-)zero reasoning; OpenRouter's
/// `reasoning: {enabled: false}` is the explicit off. Absent means the MODEL'S OWN default —
/// unless the operator declared `default_reasoning_effort` for the model in
/// MEMRA_MODEL_METADATA (`default_effort` here), in which case the UNSET case — and only
/// the unset case — resolves as if the client had sent that value (same match arms below,
/// so the downstream Request is byte-identical to the explicit request). Any explicit
/// client reasoning field wins over the deployment default:
///
/// | field value        | ThinkMode | effort level | qwen class      | gemma4        | hy3        | step35            |
/// |--------------------|-----------|--------------|-----------------|---------------|------------|-------------------|
/// | (absent)           | Default   | None         | think ON (tmpl) | think OFF     | no_think   | tail always open  |
/// | none / minimal     | NoThink   | "low"        | closed <think>  | closed channel| no_think   | Reasoning: low    |
/// | low                | Think     | "low"        | open <think>    | <\|think\|> ON| low        | Reasoning: low    |
/// | medium             | Think     | "medium"     | open <think>    | <\|think\|> ON| low (clamp)| Reasoning: medium |
/// | high               | Think     | "high"       | open <think>    | <\|think\|> ON| high       | Reasoning: high   |
/// | xhigh/max/ultra    | Think     | "high"       | open <think>    | <\|think\|> ON| high       | Reasoning: high   |
/// | {enabled: false}   | NoThink   | "low"        | closed <think>  | closed channel| no_think   | Reasoning: low    |
/// | {enabled: true}    | Think     | None         | open <think>    | <\|think\|> ON| low        | (tmpl default)    |
///
/// Returns `(think, effort_level, client_explicit)`. `effort_level` rides `Request::reasoning_effort` only
/// for templates that consume a level string (`ModelCaps::effort_levels`: step35, hy3;
/// `ModelCaps::dsv4`: the encoding_dsv4 effort ladder — on the 0731 encoding low = default
/// no prefix, high = a real prompt prefix, medium renders as the default level, and the
/// native "max" rung IS reachable: dsv4 is the one loaded template that distinguishes a
/// tier above "high" (0731: high -> ABSOLUTE_MAX, max -> BEYOND_MAX prefixes), so the
/// above-high aliases canonicalize to "max" for it instead of clamping — see
/// `canonical_effort_for` (hermes 2026-08-23: the unconditional clamp silently lost the
/// BEYOND_MAX tier for dsv4 clients); binary-switch templates are carried by `ThinkMode`
/// alone, so their prompts cannot be perturbed by a level they never read.
///
/// PRECEDENCE (issue #31, standard-surface law): an EXPLICIT boolean switch — OpenRouter
/// `reasoning.enabled`, or Anthropic `thinking.type` which `anthropic::translate` maps
/// onto it — wins the on/off decision over the switch an effort level implies; the effort
/// value is STILL validated against the one table (an invalid value is a 400 on every
/// surface, never a silent accept) and still supplies the level for level-consuming
/// templates. `vllm_switch` is the same kind of explicit boolean, arriving under the
/// vLLM/HF names (`enable_thinking`, `chat_template_kwargs.enable_thinking`); two explicit
/// switches that DISAGREE are a 400 rather than a coin-flip.
///
/// `client_explicit` (third return) says the CLIENT expressed a reasoning control itself —
/// false when the mode came only from the operator's `default_reasoning_effort`. Callers
/// use it to decide whether an unhonourable request is the client's 400 or the operator's
/// problem: refusing every request on a switchless template because of a deployment
/// default would take a model offline for a config choice the caller never made.
fn parse_think(
    reasoning_effort: &Option<String>,
    reasoning: &Option<serde_json::Value>,
    vllm_switch: Option<bool>,
    suppress_switch: Option<bool>,
    default_effort: Option<&str>,
    max_tier: bool,
) -> Result<(ThinkMode, Option<String>, bool), String> {
    let mut effort = reasoning_effort.clone();
    let ReasoningObject {
        mut enabled,
        effort: object_effort,
        exclude,
    } = parse_reasoning_object(reasoning)?;
    if let Some(e) = object_effort {
        effort = Some(e);
    }
    // vLLM-idiom switch (`enable_thinking` / `chat_template_kwargs.enable_thinking`) is the
    // same kind of explicit boolean as `reasoning.enabled`. Two explicit switches that
    // disagree get a 400: picking one silently would make the ignored one exactly the
    // accepted-and-ignored parameter this lane exists to remove.
    match (enabled, vllm_switch) {
        (Some(a), Some(b)) if a != b => {
            return Err(format!(
                "contradictory reasoning switches: reasoning.enabled={a} and \
                 enable_thinking={b} — send one"
            ));
        }
        (None, Some(b)) => enabled = Some(b),
        _ => {}
    }
    // SUPPRESSION IS OFF (owner ruling 2026-08-23, "we have to actually reason or not reason").
    // `include_reasoning:false` and `reasoning.exclude:true` used to hide the reasoning text
    // while the model still generated and we still billed it. They are now spellings of the
    // off-switch, folded onto the SAME boolean axis as `reasoning.enabled` — so they inherit
    // its precedence, its contradiction rule, and its named refusal on templates that cannot
    // honour an off-request. `include_reasoning:true` / `exclude:false` say "deliver it", which
    // is now the only behaviour, so they express no switch at all rather than pinning ON.
    //
    // Runs AFTER the vLLM fold on purpose: `enable_thinking:true` + `include_reasoning:false` is
    // a contradiction, and reaching it here means the refusal below NAMES include_reasoning
    // instead of blaming a `reasoning.enabled` the caller never sent.
    let suppress = match (exclude, suppress_switch) {
        (Some(true), _) | (_, Some(false)) => Some(false),
        _ => None,
    };
    match (enabled, suppress) {
        (Some(true), Some(false)) => {
            return Err(
                "contradictory reasoning switches: reasoning is enabled but \
                 include_reasoning:false / reasoning.exclude:true asks for no reasoning — \
                 on this server not delivering reasoning means not generating it, so send one"
                    .into(),
            );
        }
        (None, Some(b)) => enabled = Some(b),
        _ => {}
    }
    // Did the CLIENT itself ask for a reasoning mode? Recorded before the deployment
    // default is substituted, so the operator's default can never be mistaken for a
    // caller's explicit request.
    let client_explicit = effort.is_some() || enabled.is_some();
    // Deployment default: ONLY when the client expressed nothing at all — no effort on
    // either surface AND no `reasoning.enabled` in either direction. Substituting into
    // `effort` before the match keeps one mapping table: the resolved request cannot
    // diverge from an explicit request carrying the same value.
    if effort.is_none() && enabled.is_none() {
        effort = default_effort.map(str::to_string);
    }
    // Validate BEFORE the switch precedence below, so an out-of-table value is rejected
    // even when it arrives next to an explicit enabled/disabled (issue #31: /v1/messages
    // accepted every string because its value never reached this table; the old
    // `enabled == false` early-return here skipped validation the same way).
    let effort_arm = match effort.as_deref() {
        None => None,
        Some(raw) => {
            let level = canonical_effort_for(raw, max_tier).ok_or_else(|| {
                format!(
                    "bad reasoning_effort {raw:?} \
                     (none|minimal|low|medium|high; xhigh/max/ultra clamp to the \
                     highest level this model's template distinguishes)"
                )
            })?;
            Some(match level {
                "none" | "minimal" => (ThinkMode::NoThink, "low"),
                "low" => (ThinkMode::Think, "low"),
                "medium" => (ThinkMode::Think, "medium"),
                "max" => (ThinkMode::Think, "max"),
                _ => (ThinkMode::Think, "high"),
            })
        }
    };
    let (think, level) = match (enabled, effort_arm) {
        // OpenRouter "thinking off" / Anthropic thinking.type "disabled": the strongest
        // off-request any surface can express — it wins over a coexisting effort level.
        (Some(false), _) => (ThinkMode::NoThink, Some("low".to_string())),
        (Some(true), arm) => (ThinkMode::Think, arm.map(|(_, level)| level.to_string())),
        (None, Some((think, level))) => (think, Some(level.to_string())),
        (None, None) => (ThinkMode::Default, None),
    };
    Ok((think, level, client_explicit))
}

/// The three keys of the OpenRouter `reasoning` object this server understands.
struct ReasoningObject {
    enabled: Option<bool>,
    effort: Option<String>,
    exclude: Option<bool>,
}

/// Parse the OpenRouter `reasoning` object STRICTLY — every key named, every unknown key a 400.
///
/// THE DEFECT THIS CLOSES (lane/reasoning-schema-20260823): `reasoning` is typed
/// `Option<serde_json::Value>`, so serde structurally cannot reject a key, and only `enabled`,
/// `effort` and `exclude` were ever read. Anything else — most importantly OpenRouter's real
/// `reasoning.max_tokens` — was accepted with 200 and changed nothing. That is the same
/// accepted-and-ignored class PR #33 closed one level up for `chat_template_kwargs`, and the
/// same law applies: a key this server cannot act on is a named refusal, not a silent drop.
///
/// The wrong-TYPE cases are refusals too, and that also removes a cross-surface divergence:
/// `reasoning.effort: 3` used to fall through `as_str()` to `None` and be silently ignored on
/// chat, while the Anthropic surface's `output_config.effort` 400'd on exactly the same
/// mistake. One schema means one answer to the same malformed request on every surface.
///
/// `reasoning.max_tokens` gets its own message rather than the generic unknown-key one: it is
/// a real field a real client sends, so the refusal has to say WHY we will not pretend to
/// honour it (owner ruling: reasoning is output, `max_tokens` is the single output budget
/// covering it, and there is no separate reasoning budget on this server).
fn parse_reasoning_object(
    reasoning: &Option<serde_json::Value>,
) -> Result<ReasoningObject, String> {
    let mut out = ReasoningObject {
        enabled: None,
        effort: None,
        exclude: None,
    };
    let Some(v) = reasoning else { return Ok(out) };
    let obj = match v {
        serde_json::Value::Null => return Ok(out),
        serde_json::Value::Object(obj) => obj,
        _ => return Err("reasoning must be an object".into()),
    };
    for (key, value) in obj {
        // An explicit JSON null means "not set" for a KEY exactly as it already does for the whole
        // object — that is how several SDKs serialise an unset optional field, and `{"effort":
        // null}` used to be a 400 here while `/v1/responses` and `/v1/messages` both read it as
        // unset. The skip is scoped to the keys we IMPLEMENT, per arm: a first cut applied it
        // before this match, which meant `{"max_tokens": null}` and `{"banana": null}` returned
        // 200 — smuggling an unhonourable key past its own refusal by nulling the value, which is
        // the very class this function exists to close.
        match key.as_str() {
            "enabled" => {
                if !value.is_null() {
                    out.enabled = Some(
                        value
                            .as_bool()
                            .ok_or("reasoning.enabled must be true or false")?,
                    );
                }
            }
            "exclude" => {
                if !value.is_null() {
                    out.exclude = Some(
                        value
                            .as_bool()
                            .ok_or("reasoning.exclude must be true or false")?,
                    );
                }
            }
            "effort" => {
                if !value.is_null() {
                    out.effort = Some(
                        value
                            .as_str()
                            .ok_or("reasoning.effort must be a string")?
                            .to_string(),
                    );
                }
            }
            "max_tokens" => {
                return Err(
                    "reasoning.max_tokens is not supported by this server: reasoning tokens \
                     are output tokens here, and max_tokens is the ONE output budget covering \
                     reasoning and content together — there is no separate reasoning budget to \
                     spend against, so honouring this field is impossible rather than merely \
                     unimplemented. Use max_tokens for the budget, and reasoning.effort (or \
                     reasoning.enabled:false) to spend less of it on reasoning"
                        .into(),
                );
            }
            other => {
                return Err(format!(
                    "reasoning.{other} is not a field this server implements (it would change \
                     nothing about the request); the supported keys are enabled, effort and \
                     exclude"
                ));
            }
        }
    }
    Ok(out)
}

/// vLLM `chat_template_kwargs` -> the kwargs this renderer can honour.
///
/// The renderer is Rust, not jinja, so a kwarg it does not implement changes NOTHING about
/// the prompt. Accepting such a kwarg with 200 is the accepted-and-ignored defect one level
/// down from `enable_thinking`, so every unknown key is a 400 that names the key. Returns
/// the `enable_thinking` value when present.
///
/// `preserve_thinking` is Qwen3.8's THIRD official thinking kwarg (Qwen/Qwen3.8-27B card;
/// Qwen's own quickstart sends `{"enable_thinking": True, "preserve_thinking": True}`). It
/// governs whether PRIOR assistant turns replay their `<think>` block into the prompt.
///
/// The renderer's ladder arm now implements the vendor DEFAULT (lane/dflash2-session-reuse):
/// the template's replay condition is `preserve_thinking is undefined or preserve_thinking is
/// true or …`, so the absent default is replay — every prior assistant turn renders
/// `<think>\n{reasoning_content|trim}\n</think>\n\n` before its content, empty when the client
/// sent no reasoning. `true` therefore names exactly what this server renders and is ACCEPTED.
///
/// `false` (strip the block for turns at or before the last real user query) remains
/// unimplemented and refused: it needs the template's `last_query_index` walk, and silently
/// serving the replay bytes under a strip request would be a lie about the prompt.
fn parse_template_kwargs(kwargs: &Option<serde_json::Value>) -> Result<Option<bool>, String> {
    let Some(v) = kwargs else { return Ok(None) };
    let obj = match v {
        serde_json::Value::Null => return Ok(None),
        serde_json::Value::Object(obj) => obj,
        _ => return Err("chat_template_kwargs must be an object".into()),
    };
    let mut switch = None;
    for (key, value) in obj {
        match key.as_str() {
            "enable_thinking" => {
                switch = Some(
                    value
                        .as_bool()
                        .ok_or("chat_template_kwargs.enable_thinking must be true or false")?,
                );
            }
            "preserve_thinking" => {
                let preserve = value
                    .as_bool()
                    .ok_or("chat_template_kwargs.preserve_thinking must be true or false")?;
                if !preserve {
                    return Err(
                        "chat_template_kwargs.preserve_thinking:false is not supported by this \
                         server: the renderer implements the vendor DEFAULT (replay every prior \
                         assistant turn's <think> block, empty when no reasoning was sent) but \
                         not the strip arm — serving replay bytes under a strip request would \
                         misdescribe the prompt. Omit the flag or send true"
                            .into(),
                    );
                }
                // true == the vendor default the renderer implements; nothing to carry.
            }
            other => {
                return Err(format!(
                    "chat_template_kwargs.{other} is not supported by this server's \
                     template renderer (it would change nothing about the prompt); the only \
                     supported key is enable_thinking (preserve_thinking is RECOGNISED but \
                     refuses in both directions — see its own message)"
                ));
            }
        }
    }
    Ok(switch)
}

/// Reconcile the two vLLM spellings of the thinking switch: top-level `enable_thinking` and
/// `chat_template_kwargs.enable_thinking`. Both present and disagreeing is a 400 — see
/// `parse_think`'s contradiction rule, same reason.
fn resolve_vllm_think_switch(
    enable_thinking: Option<bool>,
    kwargs: &Option<serde_json::Value>,
) -> Result<Option<bool>, String> {
    let from_kwargs = parse_template_kwargs(kwargs)?;
    match (enable_thinking, from_kwargs) {
        (Some(a), Some(b)) if a != b => Err(format!(
            "contradictory reasoning switches: enable_thinking={a} and \
             chat_template_kwargs.enable_thinking={b} — send one"
        )),
        (Some(a), _) => Ok(Some(a)),
        (None, b) => Ok(b),
    }
}

/// Canonical reasoning-effort table — the ONE allowlist every surface consults: chat
/// `reasoning_effort`, OpenRouter/`/v1/responses` `reasoning.effort`, Anthropic
/// `/v1/messages` `output_config.effort`. Returns the canonical level, or None for a
/// value outside the set (the caller's 400). `xhigh`/`max`/`ultra` clamp to the highest
/// level the model's template distinguishes — because real default-config clients send
/// them (codex sends `xhigh` on /v1/responses; Claude Code sends `xhigh` on /v1/messages
/// on current models): rejecting them refuses stock CLI sessions, and accepting them on
/// SOME surfaces only was issue #31's divergence.
///
/// `dsv4_max`: deepseek-v4 is the ONE loaded template with a rung ABOVE "high" (0731
/// encoding: "high" -> DS_EFFORT_ABSOLUTE_MAX, "max" -> DS_EFFORT_BEYOND_MAX prefixes;
/// preview: "high" no-op, "max" -> ABSOLUTE_MAX — `dsv4_effort_prefix`). For it the
/// above-high aliases canonicalize to "max"; clamping them to "high" silently discarded
/// a real tier (hermes finding, fixed 2026-08-23). Every other template's highest rung
/// is "high", so the clamp there stays correct and byte-identical to before.
///
/// `minimal` = OFF here, and that is a deliberate divergence from Qwen's hosted API (which
/// maps minimal to low with reasoning on briefly): this server's schema promises that its
/// no-reasoning side is real. See the mapping table in SERVING.md.
pub(crate) fn canonical_effort_for(value: &str, max_tier: bool) -> Option<&'static str> {
    match value {
        "none" => Some("none"),
        "minimal" => Some("minimal"),
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        // `max_tier` = this model's template distinguishes a rung ABOVE `high`, so the
        // above-high aliases canonicalize to "max" instead of clamping into "high" and losing
        // the tier. True for deepseek-v4 0731 (high -> ABSOLUTE_MAX, max -> BEYOND_MAX) and for
        // GLM-5.3-Flash (low|high|max, `max` its own default). Every binary-switch and
        // three-rung template keeps the clamp — it cannot render a level it does not define.
        "xhigh" | "max" | "ultra" => Some(if max_tier { "max" } else { "high" }),
        _ => None,
    }
}

/// Membership + non-dsv4 canonicalization (the pre-exemption table; see
/// `canonical_effort_for` for the dsv4 "max" rung).
pub(crate) fn canonical_effort(value: &str) -> Option<&'static str> {
    canonical_effort_for(value, false)
}

/// serde_json::Value -> chat::Val (serde-free tree for the gemma4 tooluse arm). `Num` keeps
/// the value's exact numeric text so the rendered bytes match jinja's `{{ number }}`.
fn json_to_val(v: &serde_json::Value) -> chat::Val {
    match v {
        serde_json::Value::Null => chat::Val::Null,
        serde_json::Value::Bool(b) => chat::Val::Bool(*b),
        serde_json::Value::Number(n) => chat::Val::Num(n.to_string()),
        serde_json::Value::String(s) => chat::Val::Str(s.clone()),
        serde_json::Value::Array(a) => chat::Val::Arr(a.iter().map(json_to_val).collect()),
        // preserve_order is on (Cargo.toml): the object iterates in client key order, which
        // the gemma dialect then dictsorts — ties keep this order, matching jinja.
        serde_json::Value::Object(o) => chat::Val::Obj(
            o.iter()
                .map(|(k, val)| (k.clone(), json_to_val(val)))
                .collect(),
        ),
    }
}

/// Validate tool schemas and pre-serialize them for the template's <tools> block; also produce
/// the gemma4 tooluse dialect's typed `function` objects, and extract declared parameter types
/// (function -> parameter -> type) for argument coercion.
#[allow(clippy::type_complexity)]
fn prepare_tools(
    tools: &[serde_json::Value],
) -> Result<
    (
        Vec<String>,
        Vec<chat::Val>,
        HashMap<String, HashMap<String, String>>,
    ),
    String,
> {
    let mut tools_json = Vec::with_capacity(tools.len());
    let mut tools_struct = Vec::with_capacity(tools.len());
    let mut schemas: HashMap<String, HashMap<String, String>> = HashMap::new();
    for t in tools {
        let f = t
            .get("function")
            .ok_or("each tool needs a function object")?;
        let name = f
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or("each tool needs function.name")?;
        let mut params: HashMap<String, String> = HashMap::new();
        if let Some(props) = f
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (p, def) in props {
                if let Some(ty) = def.get("type").and_then(|t| t.as_str()) {
                    params.insert(p.clone(), ty.to_string());
                }
            }
        }
        schemas.insert(name.to_string(), params);
        tools_json.push(pyjson_str(t));
        // gemma4 arm reads the FUNCTION object (name/description/parameters/response).
        tools_struct.push(json_to_val(f));
    }
    Ok((tools_json, tools_struct, schemas))
}

/// Re-render an assistant-history tool call for the template. Value law mirrors the
/// template's `args_value | tojson if mapping/sequence else | string`: strings raw,
/// objects/arrays python-style JSON; scalars use their JSON text (`true`/`3`/`null` —
/// JSON spelling, not python's, so a parse round-trip stays self-consistent).
fn render_req_tool_call(tc: &ReqToolCall) -> Result<TmplToolCall, String> {
    let parsed: serde_json::Value = match &tc.function.arguments {
        serde_json::Value::Null => json!({}),
        serde_json::Value::String(s) if s.trim().is_empty() => json!({}),
        serde_json::Value::String(s) => serde_json::from_str(s)
            .map_err(|e| format!("tool_calls arguments is not valid JSON: {e}"))?,
        v @ serde_json::Value::Object(_) => v.clone(),
        _ => return Err("tool_calls arguments must be a JSON object".into()),
    };
    let obj = parsed
        .as_object()
        .ok_or("tool_calls arguments must decode to a JSON object")?;
    let params = obj
        .iter()
        .map(|(k, v)| {
            let rendered = match v {
                serde_json::Value::String(s) => s.clone(),
                v @ (serde_json::Value::Object(_) | serde_json::Value::Array(_)) => pyjson_str(v),
                scalar => scalar.to_string(),
            };
            (k.clone(), rendered)
        })
        .collect();
    // gemma4 tooluse dialect: typed args (dictsorted + dialect-rendered by the renderer) and
    // the call id (matched to a following tool turn's tool_call_id to name the response).
    let args = obj
        .iter()
        .map(|(k, v)| (k.clone(), json_to_val(v)))
        .collect();
    Ok(TmplToolCall {
        name: tc.function.name.clone(),
        params,
        args,
        id: tc.id.clone(),
    })
}

/// OpenAI response entry for one parsed call.
fn tool_call_json(c: &ParsedToolCall) -> serde_json::Value {
    json!({ "id": c.id, "type": "function",
            "function": { "name": c.name, "arguments": c.arguments } })
}

/// The whole server as a library entry point (BASE-4 stays: this crate is the
/// async-only seam; the bin in `src/main.rs` is one line deep). Public so a
/// deployment-owned binary can wrap the same server with its own wiring.
async fn serve_bounded_http_with_limits<F>(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: F,
    header_read_timeout: std::time::Duration,
    max_connections: usize,
    connection_max_lifetime: std::time::Duration,
) -> std::io::Result<()>
where
    F: std::future::Future<Output = ()> + Send,
{
    let connections = Arc::new(tokio::sync::Semaphore::new(max_connections));
    let (connection_shutdown, _) = tokio::sync::watch::channel(false);
    let mut connection_tasks = tokio::task::JoinSet::new();
    let mut shutdown = Box::pin(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            joined = connection_tasks.join_next(), if !connection_tasks.is_empty() => {
                if let Some(Err(error)) = joined {
                    eprintln!("[server] connection task failed: {error}");
                }
            }
            accepted = listener.accept() => {
                let (stream, _) = match accepted {
                    Ok(connection) => connection,
                    Err(error) => {
                        eprintln!("[server] accept failed: {error}");
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    }
                };
                let permit = match connections.clone().try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        drop(stream);
                        continue;
                    }
                };
                let service = app.clone().map_request(
                    |request: hyper::Request<hyper::body::Incoming>| request.map(Body::new),
                );
                let service = hyper_util::service::TowerToHyperService::new(service);
                let io = hyper_util::rt::TokioIo::new(stream);
                let mut builder = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                );
                builder
                    .http1()
                    .timer(hyper_util::rt::TokioTimer::new())
                    .header_read_timeout(header_read_timeout)
                    .max_headers(64);
                builder
                    .http2()
                    .timer(hyper_util::rt::TokioTimer::new())
                    .max_concurrent_streams(MAX_HTTP2_STREAMS_PER_CONNECTION)
                    .keep_alive_interval(Some(std::time::Duration::from_secs(30)))
                    .keep_alive_timeout(std::time::Duration::from_secs(10));
                let mut connection = Box::pin(builder
                    .serve_connection_with_upgrades(io, service)
                    .into_owned());
                let mut shutdown_rx = connection_shutdown.subscribe();
                connection_tasks.spawn(async move {
                    let _permit = permit;
                    tokio::select! {
                        result = connection.as_mut() => {
                            let _ = result;
                        }
                        _ = tokio::time::sleep(connection_max_lifetime) => {
                            // Stop accepting new requests at the age boundary, but let every
                            // active response (including long SSE) finish. A hard timeout here
                            // truncated valid generations and made connection age part of the
                            // response contract.
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                        _ = shutdown_rx.changed() => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
        }
    }
    drop(listener);
    let _ = connection_shutdown.send(true);
    let drained = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while connection_tasks.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        connection_tasks.abort_all();
        eprintln!("[server] WARN: HTTP connections exceeded the 5s graceful close deadline");
    }
    Ok(())
}

async fn serve_bounded_http<F>(
    listener: tokio::net::TcpListener,
    app: Router,
    shutdown: F,
) -> std::io::Result<()>
where
    F: std::future::Future<Output = ()> + Send,
{
    serve_bounded_http_with_limits(
        listener,
        app,
        shutdown,
        HTTP1_HEADER_READ_TIMEOUT,
        MAX_HTTP_CONNECTIONS,
        HTTP_CONNECTION_MAX_LIFETIME,
    )
    .await
}

#[tokio::main]
pub async fn serve_main() -> Result<(), Box<dyn std::error::Error>> {
    serve_with(ServerWiring::stock()).await
}

/// How a metering implementation reaches the server.
enum MeteringWiring {
    /// No accounting: every request is admitted (auth still applies), nothing is
    /// counted or billed. Only the engine is open; admission policy, billing,
    /// capture, and provisioning are the deployment binary's business.
    Stock,
    /// Deployment-supplied factory, plus whatever surfaces the deployment runs
    /// beside the engine. It CLAIMS the env vars it consumes itself
    /// (`ServerWiring::claiming`); any deployment-surface var left unclaimed is a
    /// startup FATAL, because set-but-unread configuration must not fail open.
    Custom(metering::MeteringFactory),
}

/// Deployment wiring for a custom binary. `serve_main` is exactly
/// `serve_with(ServerWiring::reference())`; a deployment-owned binary substitutes
/// its own metering and hooks the runtime handles it needs.
pub struct ServerWiring {
    metering: MeteringWiring,
    /// Called once, when the worker is live (models loaded, commands accepted),
    /// with the runtime handles a deployment-side surface needs. Not awaited.
    on_ready: Option<Box<dyn FnOnce(RuntimeHandles) + Send>>,
    /// Reference-only env vars this deployment consumes ITSELF (its own admin, its
    /// own capture). Anything on the fatal list and not claimed is a startup FATAL
    /// under custom wiring — set-but-unread configuration never fails open.
    claimed_env: Vec<&'static str>,
}

impl ServerWiring {
    /// The stock open-engine server: no accounting, no admin listener, no capture.
    pub fn stock() -> Self {
        ServerWiring {
            metering: MeteringWiring::Stock,
            on_ready: None,
            claimed_env: Vec::new(),
        }
    }

    /// A server whose admission/accounting is the factory's. See
    /// [`MeteringWiring::Custom`] for what this deliberately turns off.
    pub fn with_metering(factory: metering::MeteringFactory) -> Self {
        ServerWiring {
            metering: MeteringWiring::Custom(factory),
            on_ready: None,
            claimed_env: Vec::new(),
        }
    }

    /// Declare that the deployment consumes this reference-only env var itself
    /// (e.g. its own admin listener reads `MEMRA_ADMIN_ADDR`), disarming the
    /// custom-wiring startup FATAL for exactly that var.
    pub fn claiming(mut self, var: &'static str) -> Self {
        self.claimed_env.push(var);
        self
    }

    pub fn on_ready(mut self, hook: impl FnOnce(RuntimeHandles) + Send + 'static) -> Self {
        self.on_ready = Some(Box::new(hook));
        self
    }
}

/// Runtime handles handed to [`ServerWiring::on_ready`] — the narrow set of
/// engine-runtime operations a deployment-side admin surface needs.
pub struct RuntimeHandles {
    pub trim: TrimHandle,
    /// Tenant lifecycle purge (lane/kv-tenancy-compaction-20260831): the deployment
    /// admin surface calls this from its key-revocation and tenant-deletion paths.
    pub purge: PurgeHandle,
    /// Host-tier deploy handoff (lane/host-tier-deploy-warmth-20260901): the deployment
    /// admin surface exposes these as `POST /admin/kv-host/export` (called by
    /// serve-deploy on the DRAINED old slot after the edge flip) and
    /// `POST /admin/kv-host/import` (called on the promoted slot right after). Both are
    /// inert unless MEMRA_KV_HOST_HANDOFF names a path on the slot.
    pub kv_handoff: HostHandoffHandle,
    /// Flips to `true` when the graceful drain completes (the moment the in-tree
    /// admin listener stops). A deployment-side surface MUST end and drop its
    /// [`TrimHandle`] AND [`PurgeHandle`] on this signal: each handle wraps a worker
    /// command sender, and the GPU worker only exits when every sender is dropped.
    pub shutdown: tokio::sync::watch::Receiver<bool>,
}

/// Ask the worker to trim its pools (the engine half of `/admin/trim`). Cloneable;
/// answers with the worker's own trim report.
#[derive(Clone)]
pub struct TrimHandle {
    cmd_tx: Sender<Cmd>,
}

impl TrimHandle {
    /// 503-shaped errors as strings: worker down, or no answer within 30s.
    pub async fn trim(&self) -> Result<serde_json::Value, String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.cmd_tx.send(Cmd::TrimPools(tx)).is_err() {
            return Err("worker is down".into());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(report)) => Ok(json!(report)),
            _ => Err("worker did not answer the trim within 30s".into()),
        }
    }
}

/// Purge one tenant's parked KV state (the engine half of a deployment admin
/// `/admin/tenants/{tenant}/purge`; lane/kv-tenancy-compaction-20260831, tiering spec
/// §0.5). Contract notes for the deployment surface: the path parameter is `{tenant}`
/// (the keyring tenant id, the same string `--gen-key <tenant>` took), never
/// `{tenant_id}`; fire it from key revocation AND tenant deletion; a report with
/// `device_pinned_left > 0` means in-flight sessions still lease device entries in the
/// tenant's namespaces, so re-fire after the drain. Cloneable, same lifetime contract
/// as [`TrimHandle`]: drop it on the shutdown signal.
#[derive(Clone)]
pub struct PurgeHandle {
    cmd_tx: Sender<Cmd>,
}

impl PurgeHandle {
    /// 503-shaped errors as strings: worker down, or no answer within 30s.
    pub async fn purge_tenant(&self, tenant: &str) -> Result<serde_json::Value, String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let cmd = Cmd::PurgeTenantHost {
            tenant: tenant.to_string(),
            tx,
        };
        if self.cmd_tx.send(cmd).is_err() {
            return Err("worker is down".into());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(report)) => Ok(json!(report)),
            _ => Err("worker did not answer the purge within 30s".into()),
        }
    }
}

/// Host-tier deploy handoff (lane/host-tier-deploy-warmth-20260901): the engine half of a
/// deployment admin `POST /admin/kv-host/export` / `POST /admin/kv-host/import` pair.
/// Contract notes for the deployment surface: export is called ONLY on the drained old
/// slot (it refuses under traffic unless `force`, and the write stalls that slot's ticks
/// for its duration, expected and harmless when drained); import answers as soon as the
/// file header validates, then re-materializes entries one per tick in the background
/// (watch `prefix_host_handoff_*` in /metrics for completion). Same lifetime contract as
/// [`TrimHandle`]: drop it on the shutdown signal.
#[derive(Clone)]
pub struct HostHandoffHandle {
    cmd_tx: Sender<Cmd>,
}

impl HostHandoffHandle {
    /// Errors as strings: worker down, refused, or no answer. The timeout is generous by
    /// design: tens of GB of drain-demote + NVMe write happen inside the reply.
    pub async fn export(&self, force: bool) -> Result<serde_json::Value, String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self
            .cmd_tx
            .send(Cmd::ExportHostHandoff { force, tx })
            .is_err()
        {
            return Err("worker is down".into());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(900), rx).await {
            Ok(Ok(Ok(report))) => Ok(json!(report)),
            Ok(Ok(Err(refused))) => Err(refused),
            _ => Err("worker did not answer the export within 900s".into()),
        }
    }

    /// Begin the drip import; answers with the validated header (fast: no entry bytes are
    /// read yet) or the refusal reason.
    pub async fn import(&self) -> Result<serde_json::Value, String> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.cmd_tx.send(Cmd::ImportHostHandoff { tx }).is_err() {
            return Err("worker is down".into());
        }
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
            Ok(Ok(Ok(start))) => Ok(json!(start)),
            Ok(Ok(Err(refused))) => Err(refused),
            _ => Err("worker did not answer the import within 30s".into()),
        }
    }
}

pub async fn serve_with(wiring: ServerWiring) -> Result<(), Box<dyn std::error::Error>> {
    // Key lifecycle CLI (lane/api-keys): `--gen-key <tenant>` / `--revoke-key <prefix>`
    // manage the keyring and exit — no engine, no GPU, no model load.
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--version` prints the build identity and exits: no engine, no GPU, no model load. So
    // the fingerprint of a DEPLOYED artifact is checkable on any box, and in the release
    // container that produced it, without touching a serving stack. That check is the one
    // that would have caught `memra-unknown` before it reached a customer.
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("memra-server {}", env!("CARGO_PKG_VERSION"));
        println!("system_fingerprint {SYSTEM_FINGERPRINT}");
        println!("build_id_src {BUILD_ID_SRC}");
        println!("git_sha {BUILD_GIT_SHA}");
        if !BUILD_ID_NOTE.is_empty() {
            println!("degraded {BUILD_ID_NOTE}");
        }
        return Ok(());
    }
    if let Some(code) = auth::run_cli(&args) {
        std::process::exit(code);
    }
    // Build provenance is the FIRST line of every boot. An unknown fingerprint is how this
    // defect hid: a build with a meaningless identity looked exactly like a good one, on
    // both sides of the deploy.
    eprintln!("{}", build_identity_line());
    if BUILD_ID_SRC != build_id::BUILD_ID_SRC_TREE {
        eprintln!(
            "[server] WARNING: build identity is DEGRADED: {BUILD_ID_NOTE}. \
             system_fingerprint {SYSTEM_FINGERPRINT} carries a version-only id, so it does \
             NOT identify the source this binary was compiled from and published \
             performance pins cannot be verified against it (darklanes \
             tools/check-claim-builds.mjs --live). Rebuild where the workspace source tree \
             is readable."
        );
    }
    // Keyring (MEMRA_API_KEYS): parsed once here so a bad config is a startup FATAL,
    // not a per-request surprise. Absent = single-key/open behavior, unchanged.
    auth::init_from_env();
    let api_auth = match ApiAuth::from_env() {
        Ok(auth) => auth,
        Err(err) => {
            eprintln!("[server] FATAL: {err}");
            std::process::exit(1);
        }
    };
    let addr = std::env::var("MEMRA_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let allow_open_bind = std::env::var("MEMRA_ALLOW_OPEN_BIND").as_deref() == Ok("1");
    let (bind_addr, bind_loopback) = match resolve_bind_addr(&addr) {
        Ok(resolved) => resolved,
        Err(err) => {
            eprintln!("[server] FATAL: {err}");
            std::process::exit(1);
        }
    };
    // The refusal goes through validate_bind_security — the SAME function the
    // exposed_open_bind_is_refused_before_server_start test exercises. It used to be
    // duplicated inline here, so the test was pinning a copy of the gate rather than
    // the gate itself (dead_code exposed the split).
    if let Err(message) = validate_bind_security(&addr, api_auth.configured(), allow_open_bind) {
        eprintln!("[server] FATAL: {message}");
        std::process::exit(1);
    }
    if !bind_loopback && !api_auth.configured() {
        eprintln!(
            "[server] WARNING: MEMRA_ALLOW_OPEN_BIND=1 permits open completion routes on {addr}; \
             metrics remain bearer-protected"
        );
    }
    let metrics_token = match std::env::var("MEMRA_METRICS_TOKEN") {
        Ok(token) if token.is_empty() => {
            eprintln!("[server] FATAL: MEMRA_METRICS_TOKEN must not be empty");
            std::process::exit(1);
        }
        Ok(token) => Some(token),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            eprintln!("[server] FATAL: MEMRA_METRICS_TOKEN must be valid UTF-8");
            std::process::exit(1);
        }
    };
    let metrics_auth = MetricsAuth::new(bind_loopback, api_auth.configured(), metrics_token);

    let models = parse_models_config();
    let (openrouter_metadata, provider_metadata) = match load_openrouter_metadata(&models) {
        Ok(loaded) => loaded,
        Err(err) => {
            eprintln!("[server] FATAL: {err}");
            std::process::exit(1);
        }
    };
    // The metering seam splits here. The STOCK server ships no accounting: only the
    // engine is open, and admission policy / billing / capture / the provisioning
    // surface are the deployment binary's business (owner razor 2026-08-29). Their
    // env vars are startup FATALs unless the wiring CLAIMS them — set-but-unread
    // configuration never fails open.
    let metering_obj: Option<Arc<dyn metering::Metering>> = {
        let factory = match wiring.metering {
            MeteringWiring::Stock => None,
            MeteringWiring::Custom(factory) => Some(factory),
        };
        for deployment_only in [
            "MEMRA_REQUEST_LEDGER",
            "MEMRA_TENANT_BUDGETS",
            "MEMRA_ADMIN_ADDR",
            "MEMRA_ADMIN_TOKEN_FILE",
            "MEMRA_CAPTURE_DIR",
        ] {
            if std::env::var_os(deployment_only).is_some()
                && !wiring.claimed_env.contains(&deployment_only)
            {
                eprintln!(
                    "[server] FATAL: {deployment_only} is a deployment-binary surface; this \
                     build ships no accounting/admin/capture. Wire a Metering implementation \
                     through ServerWiring and claim the vars it consumes."
                );
                std::process::exit(1);
            }
        }
        match factory {
            None => None,
            Some(factory) => {
                let model_ids: Vec<String> =
                    models.iter().map(|(name, _, _)| name.clone()).collect();
                match factory(&metering::MeteringInit { models: &model_ids }) {
                    Ok(metering_obj) => metering_obj,
                    Err(err) => {
                        eprintln!("[server] FATAL: metering wiring: {err}");
                        std::process::exit(1);
                    }
                }
            }
        }
    };
    let budget_tokenizers = if metering_obj
        .as_ref()
        .is_some_and(|manager| manager.enforces_limits())
    {
        match load_budget_tokenizers(&models) {
            Ok(tokenizers) => Some(tokenizers),
            Err(err) => {
                eprintln!("[server] FATAL: prepaid reservation tokenizers: {err}");
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    eprintln!("[server] starting; models config = {models:?}");

    // Inference-liveness state (G5). Created BEFORE the worker so the whole weight load is
    // observable as PHASE_LOADING rather than as a gap: /livez and /readyz answer honestly
    // from the first accepted connection, which is what a supervisor's Type=notify +
    // WatchdogSec contract and a load balancer's readiness probe both need.
    let health_state = health::WorkerHealth::new();
    // GPU-fault watchers (G24) start before the load too: an Xid that fires DURING a 120 s
    // weight load is exactly the case a post-load watcher misses. spawn_gpu_watch owns the
    // Xid tail as well (one call, two threads).
    health::spawn_gpu_watch(health_state.clone());
    health::spawn_sd_watchdog(health_state.clone());

    // Spawn the GPU worker thread and block until every model is loaded (or it fails).
    let (cmd_tx, model_names, caps, metrics, worker_thread) =
        match worker::spawn(models, health_state.clone()) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("[server] FATAL: worker init failed: {err}");
                health_state.mark_dead(format!("worker init failed: {err}"));
                health::sd_notify(&format!("STATUS=worker init failed: {err}"));
                std::process::exit(1);
            }
        };
    eprintln!("[server] worker ready; serving models: {model_names:?}");

    // Deployment hook: the worker is live, hand over the runtime handles — INCLUDING
    // the drain shutdown signal. The TrimHandle wraps a worker command sender, and the
    // worker's exit condition is "all senders dropped": a deployment surface that
    // holds its handle past the shutdown signal recreates the v0.116.0 38-minute
    // worker-join hang (the billing parity battery caught exactly that on the first
    // deployment-binary arm, 2026-08-29).
    let (drain_shutdown_tx, drain_shutdown_rx) = tokio::sync::watch::channel(false);
    if let Some(on_ready) = wiring.on_ready {
        on_ready(RuntimeHandles {
            trim: TrimHandle {
                cmd_tx: cmd_tx.clone(),
            },
            purge: PurgeHandle {
                cmd_tx: cmd_tx.clone(),
            },
            kv_handoff: HostHandoffHandle {
                cmd_tx: cmd_tx.clone(),
            },
            shutdown: drain_shutdown_rx.clone(),
        });
    }

    // Dead-darklane background job runner (MEMRA_BG_JOB; lane/darklane-training): armed
    // only after the worker is ready — a weight load is PHASE_LOADING, never a valley.
    let bg_handle = darklane::spawn_from_env(health_state.clone());
    let bg_state = bg_handle.as_ref().map(|h| {
        let mode = darklane::BgConfig::from_env()
            .map(|c| c.yield_mode.as_str())
            .unwrap_or("stop");
        (h.state.clone(), mode)
    });

    let state = AppState {
        cmd_tx,
        models: model_names,
        caps,
        openrouter_metadata: Arc::new(openrouter_metadata),
        provider_metadata: Arc::new(provider_metadata),
        metering: metering_obj,
        budget_tokenizers,
        api_auth,
        metrics_auth,
        metrics,
        inflight: Arc::new(Default::default()),
        tenant_inflight: Arc::new(Default::default()),
        health: health_state.clone(),
        bg: bg_state,
    };
    let inflight_handle = state.inflight.clone();
    // For the drain-kill fault-attribution latch: the drain future outlives the
    // router that consumes `state`.
    let drain_metering = state.metering.clone();
    // LOAD-GUARD DEMAND SEAM (lane/sampled-restore-load-guard). The worker cannot see a request
    // that has passed this boundary but not yet reached its channel — which is exactly the head
    // of an arriving fan-out, the one row a tick-top reading of `active + queue` cannot refuse.
    // Registering the gauge (not a copy of it) keeps one source of truth.
    worker::register_http_inflight(state.inflight.clone());
    let app = Router::new()
        // /health is the historical name (every memra script polls it) and stays the
        // LIVENESS probe; /livez + /readyz are the k8s-doctrine split (healthz deprecated
        // upstream at v1.16). Readiness ≠ liveness: draining or a not-yet-loaded model
        // takes the box out of ROTATION without asking a supervisor to kill it.
        .route("/health", get(health_live))
        .route("/livez", get(health_live))
        .route("/readyz", get(health_ready))
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models_v1))
        .route("/v1/auth/check", get(auth_check))
        .route("/v1/completions", post(completions_admitted))
        .route("/v1/embeddings", post(embed_api::embeddings_admitted))
        .route("/v1/rerank", post(embed_api::rerank_admitted))
        .route("/v1/chat/completions", post(chat_completions_admitted))
        // Translation surfaces (lane/api-surfaces): Anthropic Messages + OpenAI
        // Responses over the same core. Axum matches the PATH only, so the
        // `?beta=true` query some clients append arrives here too.
        .route("/v1/messages", post(anthropic::messages_admitted))
        .route("/v1/responses", post(responses_api::responses_admitted))
        .route("/metrics", get(get_metrics))
        .route("/yield/metrics", get(yield_metrics))
        .with_state(state.clone());
    // Body-size policy (hermes finding): explicit ceiling sized to the advertised
    // 262k-token + vision surface, with 413s reshaped to the standard error object.
    let app = apply_body_limit(app);
    // Header-only auth runs outside the body-limit/extractor stack. Invalid callers therefore
    // cannot spend the 192 MiB parser budget, while valid callers retain the advertised 413.
    let app = app.layer(middleware::from_fn_with_state(
        state,
        authenticate_inference_before_body,
    ));
    let app = if ttft::enabled() {
        app.layer(middleware::from_fn(ttft_request_start))
    } else {
        app
    };

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    eprintln!("[server] listening on http://{bind_addr}");
    drop(drain_shutdown_rx);
    // READY=1 only AFTER the models are resident and the socket is bound — the whole point of
    // Type=notify is that "started" means "can serve". A no-op when NOTIFY_SOCKET is unset
    // (i.e. every non-systemd run), so it costs nothing outside a unit.
    health::sd_notify("READY=1\nSTATUS=serving");
    // GRACEFUL DRAIN (gap-scan F11): SIGTERM flips the drain flag (new completion
    // requests 503 immediately; /health reports "draining"), then the shutdown future
    // resolves once every in-flight request finished (the HTTP-layer gauge — streams
    // hold their slot until fully written) or the MEMRA_DRAIN_S deadline (default 30s)
    // passed. axum's graceful shutdown stops accepting, lets tracked connections finish
    // their current response, and returns — exit 0 (in-flight loss only past deadline).
    let inflight = inflight_handle;
    let signal_admin_shutdown = drain_shutdown_tx.clone();
    let serve_result = serve_bounded_http(listener, app, async move {
        let mut sigterm =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[server] WARN: no SIGTERM handler ({err}); drain disabled");
                    std::future::pending::<()>().await;
                    unreachable!()
                }
            };
        sigterm.recv().await;
        DRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = signal_admin_shutdown.send(true);
        // STOPPING=1 + EXTEND_TIMEOUT_USEC: tell systemd the stop is deliberate and how
        // long the drain may legitimately take, so TimeoutStopSec does not SIGKILL a
        // healthy drain mid-stream (audit's systemd section).
        health::sd_notify(&format!(
            "STOPPING=1\nSTATUS=draining\nEXTEND_TIMEOUT_USEC={}",
            (drain_deadline_s() + 5) * 1_000_000
        ));
        let n: usize = inflight
            .iter()
            .map(|c| c.load(std::sync::atomic::Ordering::SeqCst))
            .sum();
        eprintln!(
            "[server] SIGTERM: draining ({n} in flight, deadline {}s)",
            drain_deadline_s()
        );
        let deadline = std::time::Duration::from_secs(drain_deadline_s());
        let t0 = std::time::Instant::now();
        loop {
            let n: usize = inflight
                .iter()
                .map(|c| c.load(std::sync::atomic::Ordering::SeqCst))
                .sum();
            if n == 0 {
                eprintln!(
                    "[server] drain complete in {:.1}s; exiting",
                    t0.elapsed().as_secs_f64()
                );
                break;
            }
            if t0.elapsed() >= deadline {
                eprintln!(
                    "[server] drain deadline ({}s) hit with {n} in flight; exiting",
                    drain_deadline_s()
                );
                // Fault attribution (owner ruling 2026-08-23): everything still in
                // flight past this point is killed by OUR shutdown. Latch the
                // classification so their receipts settle `drain_killed` (debit
                // ZERO) instead of `abandoned` (partial-billed client walk-away).
                // Through the seam: a custom implementation that never heard this
                // would partial-bill every drain-killed request.
                if let Some(metering) = drain_metering.as_ref() {
                    metering.drain_kill();
                }
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    })
    .await;
    // Drain complete: tell every deployment-side surface to end and drop its
    // TrimHandle (see the worker-join note below).
    let _ = drain_shutdown_tx.send(true);
    serve_result?;
    // Background job cleanup on the graceful path: SIGCONT+SIGTERM(+KILL past grace) the
    // job's process group — a SIGSTOPped orphan would stay frozen forever. The ungraceful
    // path (server SIGKILL) is covered by PDEATHSIG on the child.
    if let Some(h) = bg_handle {
        h.shutdown();
    }
    // The Router owned the last command sender in the stock build; a deployment
    // surface's TrimHandle clone must die on the drain signal above, or the worker's
    // "all senders dropped" exit condition never fires and the join below hangs
    // forever on graceful SIGTERM (v0.116.0 admin_cmd_tx incident; re-caught by the
    // billing parity battery 2026-08-29). Once serve returns it is gone, so the GPU
    // worker retires any sessions that finished concurrently with the HTTP drain. Keep main
    // alive until that cleanup completes: returning first lets CUDA deinitialize underneath a
    // pending-token flush (observed with paired speculative sessions on graceful SIGTERM).
    worker_thread.join().map_err(|_| {
        std::io::Error::other("GPU worker thread panicked during graceful shutdown")
    })?;
    eprintln!("[server] GPU worker shutdown complete");
    Ok(())
}

/// Validate a resolved model-plan path BEFORE the worker thread spins up: a FILE loads as
/// GGUF; a DIRECTORY must be an HF safetensors checkpoint (`config.json` +
/// `model.safetensors` or `model.safetensors.index.json` — the run-safetensors load path)
/// or a memra repack dir (`manifest.json`). A clear error at parse time beats a worker
/// load failure after the Engine is already up.
fn validate_model_path(path: &str) -> Result<(), String> {
    let p = std::path::Path::new(path);
    if !p.exists() {
        return Err(format!("model path {path:?} does not exist"));
    }
    if p.is_file() {
        return Ok(()); // GGUF file (the worker's file branch)
    }
    if p.join("manifest.json").exists() {
        return Ok(()); // memra repack/overlay dir
    }
    let has_st =
        p.join("model.safetensors").exists() || p.join("model.safetensors.index.json").exists();
    if !has_st {
        return Err(format!(
            "model dir {path:?} is not a servable checkpoint: want model.safetensors or \
             model.safetensors.index.json + config.json (HF safetensors dir), or \
             manifest.json (memra repack dir)"
        ));
    }
    if !p.join("config.json").exists() {
        return Err(format!(
            "model dir {path:?} has safetensors weights but no config.json"
        ));
    }
    Ok(())
}

/// MEMRA_MODELS="name=/path.gguf[+/draft.gguf],name2=hf:owner/repo,name3=/hf_ckpt_dir".
/// Falls back to the BASE-4 test pair. `+<draft.gguf>` after a model path attaches that
/// model's regime draft (docs/DRAFT-REGIME.md) — per model, not the global MEMRA_MTP_DRAFT
/// env, so a multi-model server gives each model its own draft. Both parts accept hf: specs.
/// A model path may also be an HF safetensors checkpoint DIRECTORY (serve-st lane,
/// 2026-08-04) — validated by `validate_model_path`, loaded through the same
/// SafetensorsSource seam as run-safetensors/run-gen.
fn parse_models_config() -> Vec<(String, String, Option<String>)> {
    if let Ok(spec) = std::env::var("MEMRA_MODELS") {
        let mut out = Vec::new();
        for entry in spec.split(',').filter(|s| !s.trim().is_empty()) {
            if let Some((name, path)) = entry.split_once('=') {
                // Paths accept hf:owner/repo[:file] specs — resolved (downloaded on first
                // use) before the worker sees them.
                let (mpath, dpath) = match path.trim().split_once('+') {
                    Some((m, d)) => (m.trim(), Some(d.trim())),
                    None => (path.trim(), None),
                };
                let resolve = |p: &str| {
                    memra_gguf::hf::resolve_arg(p).unwrap_or_else(|err| {
                        eprintln!("[server] FATAL: model {name:?}: {err}");
                        std::process::exit(1);
                    })
                };
                let mpath = resolve(mpath);
                if let Err(err) = validate_model_path(&mpath) {
                    eprintln!("[server] FATAL: model {name:?}: {err}");
                    std::process::exit(1);
                }
                // The DRAFT path gets the same parse-time existence check as the model path
                // (lane/step-draft, 2026-08-07). It did not, and the asymmetry cost a class of
                // late failure: a typo'd or unmounted drafter path survived parse, survived the
                // hf resolve, and only failed after the worker had already spent the whole
                // trunk load on the GPU — so on a busy card the operator got
                // `CUDA_ERROR_OUT_OF_MEMORY` on the TRUNK and never learned the drafter path
                // was wrong at all. Found by this lane's own gate arm D. A drafter must be a
                // FILE: `load_draft` opens it as a GGUF, so the dir forms `validate_model_path`
                // admits are not valid here.
                let dpath = dpath.map(|d| {
                    let d = resolve(d);
                    let p = std::path::Path::new(&d);
                    if !p.exists() {
                        eprintln!(
                            "[server] FATAL: model {name:?}: drafter path {d:?} does not \
                                   exist (MEMRA_MODELS '+draft' attach). Refusing to start \
                                   rather than serving plain decode under a config that asked \
                                   for speculative decoding."
                        );
                        std::process::exit(1);
                    }
                    if !p.is_file() {
                        eprintln!(
                            "[server] FATAL: model {name:?}: drafter path {d:?} is not a \
                                   file — a '+draft' attach must be a NextN/MTP GGUF file."
                        );
                        std::process::exit(1);
                    }
                    d
                });
                out.push((name.trim().to_string(), mpath, dpath));
            } else {
                eprintln!(
                    "[server] WARN: bad MEMRA_MODELS entry {entry:?} (want name=/path[+/draft]); skipping"
                );
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    // Default: the BASE-4 test pair (main=27B, judge=9B).
    vec![
        (
            "main".into(),
            "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf".into(),
            None,
        ),
        (
            "judge".into(),
            "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf".into(),
            None,
        ),
    ]
}

fn load_budget_tokenizers(
    models: &[(String, String, Option<String>)],
) -> Result<Arc<HashMap<String, Arc<Tokenizer>>>, String> {
    let mut tokenizers = HashMap::new();
    for (alias, path, _) in models {
        let path = std::path::Path::new(path);
        let tokenizer = if path.is_dir() {
            let tokenizer_dir = if path.join("manifest.json").exists() {
                let repack = memra_gguf::source::Hy3RepackSource::open(path).map_err(|err| {
                    format!("model {alias:?}: open repack tokenizer source: {err}")
                })?;
                repack
                    .source_dir()
                    .filter(|source| source.join("tokenizer.json").exists())
                    .unwrap_or(path)
                    .to_path_buf()
            } else {
                path.to_path_buf()
            };
            Tokenizer::from_hf_dir(&tokenizer_dir)
                .map_err(|err| format!("model {alias:?}: reservation tokenizer: {err}"))?
        } else {
            let gguf = memra_gguf::GgufFile::open(path)
                .map_err(|err| format!("model {alias:?}: open reservation tokenizer: {err}"))?;
            Tokenizer::from_gguf(&gguf)
                .map_err(|err| format!("model {alias:?}: reservation tokenizer: {err}"))?
        };
        tokenizers.insert(alias.clone(), Arc::new(tokenizer));
    }
    Ok(Arc::new(tokenizers))
}

/// Shared body for both probes: the honest state, plus the numbers that explain it.
fn health_payload(st: &AppState, status: &str, detail: Option<&str>) -> serde_json::Value {
    let s = st.health.snapshot();
    let mut v = json!({
        "status": status,
        "models": *st.models,
        "worker": {
            "phase": health::phase_name(s.phase),
            "beat_age_ms": s.beat_age_ms,
            "tick_max_ms": s.tick_max_ms,
            "stall_threshold_ms": s.stall_threshold_ms,
            "generation": s.generation,
            "xid_warnings": s.xid_warns,
        },
    });
    if let Some(d) = detail {
        v["detail"] = json!(d);
    }
    v
}

/// `/readyz` adds peer-integrity coverage as an advisory. Even `degraded` stays HTTP 200 while
/// the worker is otherwise ready: new speculative sessions are held on the safe plain path, so
/// draining all traffic would discard usable plain capacity instead of helping self-recovery.
fn readiness_payload(st: &AppState, status: &str, detail: Option<&str>) -> serde_json::Value {
    let mut v = health_payload(st, status, detail);
    v["peer_probe_integrity"] = json!(st.health.peer_probe_integrity().detail());
    v
}

/// Header-only credential preflight for the edge router. It deliberately has no
/// body extractor: a router can prove a bearer is known before deciding whether
/// to buffer a large model-selection request.
async fn auth_check() -> impl IntoResponse {
    StatusCode::NO_CONTENT
}

/// LIVENESS (`/health`, `/livez`) — INFERENCE liveness, not process liveness (G5).
///
/// WHAT CHANGED AND WHY. The old handler returned 200 whenever the HTTP task was scheduled:
/// a panicked GPU worker, a wedged GPU, a poisoned CUDA context — all reported "ok" forever,
/// on a box that answered nothing. Now the answer is derived ONLY from worker state: a
/// heartbeat the scheduler loop stamps every iteration, the panic/GPU fault latches, and the
/// load phase.
///
/// 503 (dead / GPU-faulted / stalled / still loading) is deliberately a
/// SUPERVISOR-ACTIONABLE signal — the only recovery for a sticky CUDA fault is restarting the
/// process, so this endpoint is what makes `Restart=on-failure` + a liveness probe work.
///
/// DRAINING stays **200**: a drain is a healthy, deliberate shutdown, and answering 503 here
/// would invite a supervisor to kill the process in the middle of finishing in-flight
/// streams. Rotation is `/readyz`'s job — that is the whole reason the two are separate.
async fn health_live(State(st): State<AppState>) -> impl IntoResponse {
    if draining() {
        // "draining" = the LB/orchestrator not-ready signal (gap-scan F11): the process is
        // finishing in-flight work and will exit; route new traffic elsewhere.
        return (StatusCode::OK, Json(health_payload(&st, "draining", None))).into_response();
    }
    match st.health.live() {
        Ok(()) => (StatusCode::OK, Json(health_payload(&st, "ok", None))).into_response(),
        Err(why) => retry_contract_response(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(health_payload(&st, "unhealthy", Some(&why))),
            )
                .into_response(),
            Some(worker::WORKER_RESPAWN_BACKOFF_BASE_S),
        ),
    }
}

/// READINESS (`/readyz`) — "should this instance receive traffic right now?"
///
/// Ready = model loaded AND worker alive AND not draining. Unready is NOT a request for a
/// restart: draining and still-loading are both perfectly healthy states that simply must not
/// be routed to. k8s doctrine (`/livez` + `/readyz`; `healthz` deprecated at v1.16), and ahead
/// of both vLLM (no readiness endpoint) and TGI (single `/health`).
///
/// Queue pressure deliberately does NOT flip readiness: memra's interactive lane queues FIFO
/// and never sheds, so a deep queue is work in progress, not unreadiness. Capacity backpressure
/// belongs on the request path as 429/503 (G6), where a client can act on it.
async fn health_ready(State(st): State<AppState>) -> impl IntoResponse {
    let is_draining = draining();
    match st.health.ready(is_draining) {
        Ok(()) => (StatusCode::OK, Json(readiness_payload(&st, "ready", None))).into_response(),
        Err(why) => retry_contract_response(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(readiness_payload(&st, "not_ready", Some(&why))),
            )
                .into_response(),
            Some(if is_draining {
                drain_deadline_s()
            } else {
                worker::WORKER_RESPAWN_BACKOFF_BASE_S
            }),
        ),
    }
}

#[derive(Clone, Copy)]
struct DualPpMetricsSnapshot {
    stage_ns: [u64; 4],
    stage_samples: [usize; 4],
    dropped_timing_samples: usize,
    overlaps: usize,
    slot_pairs: usize,
    slot_uses: [usize; 2],
    slot_collisions: usize,
}

impl DualPpMetricsSnapshot {
    fn current() -> Self {
        let (stage_ns, stage_samples) = memra_engine::pp::dual_pp_timing_snapshot();
        let (slot_pairs, slot_uses, slot_collisions) = memra_engine::pp::dual_pp_slot_snapshot();
        Self {
            stage_ns,
            stage_samples,
            dropped_timing_samples: memra_engine::pp::dual_pp_timing_dropped(),
            overlaps: memra_engine::pp::dual_pp_overlaps(),
            slot_pairs,
            slot_uses,
            slot_collisions,
        }
    }

    fn populated(self) -> bool {
        self.stage_samples.iter().any(|&n| n > 0)
            || self.dropped_timing_samples > 0
            || self.slot_pairs > 0
            || self.slot_collisions > 0
    }
}

fn insert_dual_pp_metrics(
    body: &mut serde_json::Value,
    metrics_scope: &MetricsScope,
    snapshot: impl FnOnce() -> DualPpMetricsSnapshot,
) {
    // Dual wave/slot counts reveal live capacity and the two-device topology. Completion
    // credentials never evaluate the snapshot closure, even when the process is dual-active.
    if !metrics_scope.operator() {
        return;
    }
    let snapshot = snapshot();
    if !snapshot.populated() {
        return;
    }
    let timings: serde_json::Map<String, serde_json::Value> = memra_engine::pp::DUAL_PP_STAGE_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let total_ms = snapshot.stage_ns[i] as f64 / 1_000_000.0;
            (
                name.to_string(),
                json!({
                    "samples": snapshot.stage_samples[i],
                    "total_ms": total_ms,
                    "mean_ms": if snapshot.stage_samples[i] > 0 {
                        total_ms / snapshot.stage_samples[i] as f64
                    } else { 0.0 },
                }),
            )
        })
        .collect();
    body["dual_pp"] = json!({
        "overlaps": snapshot.overlaps,
        "slot_pairs": snapshot.slot_pairs,
        "slot_uses": snapshot.slot_uses,
        "slot_collisions": snapshot.slot_collisions,
        "cuda_event_spans": timings,
        "dropped_timing_samples": snapshot.dropped_timing_samples,
    });
}

#[derive(Clone, Copy)]
struct PpWaveMetricsSnapshot {
    ticks: usize,
    cells: usize,
    overlaps: usize,
}

impl PpWaveMetricsSnapshot {
    fn current() -> Self {
        let (ticks, cells, overlaps) = memra_engine::pp::pp_wave_snapshot();
        Self {
            ticks,
            cells,
            overlaps,
        }
    }
}

fn insert_pp_wave_metrics(
    body: &mut serde_json::Value,
    metrics_scope: &MetricsScope,
    snapshot: impl FnOnce() -> PpWaveMetricsSnapshot,
) {
    if !metrics_scope.operator() {
        return;
    }
    let snapshot = snapshot();
    if snapshot.ticks == 0 && snapshot.cells == 0 {
        return;
    }
    body["pp_wave"] = json!({
        "ticks": snapshot.ticks,
        "cells": snapshot.cells,
        "overlaps": snapshot.overlaps,
    });
}

fn insert_spec_acceptance_metrics(
    body: &mut serde_json::Value,
    metrics_scope: &MetricsScope,
    snapshot: impl FnOnce() -> HashMap<String, memra_engine::spec::SpecTelemetry>,
) {
    // Acceptance shape is process-wide model telemetry. As with dual_pp, tenant credentials
    // return before evaluating the snapshot closure so they cannot observe other workloads.
    if !metrics_scope.operator() {
        return;
    }
    let snapshot = snapshot();
    if snapshot.is_empty() {
        return;
    }

    let mut tau = serde_json::Map::new();
    let mut by_position = serde_json::Map::new();
    for (model, telemetry) in snapshot {
        if telemetry.rounds == 0 {
            continue;
        }
        let n_pos = telemetry
            .pos_drafted
            .iter()
            .rposition(|&n| n > 0)
            .map_or(0, |position| position + 1);
        tau.insert(model.clone(), json!(telemetry.tau()));
        by_position.insert(
            model,
            json!({
                "window_seconds": worker::SPEC_METRICS_WINDOW_S,
                "rounds": telemetry.rounds,
                "offered": telemetry.pos_drafted[..n_pos].to_vec(),
                "accepted": telemetry.pos_accepted[..n_pos].to_vec(),
                "accept_rate": (0..n_pos).map(|position| {
                    let offered = telemetry.pos_drafted[position];
                    if offered > 0 {
                        telemetry.pos_accepted[position] as f64 / offered as f64
                    } else {
                        0.0
                    }
                }).collect::<Vec<f64>>(),
            }),
        );
    }
    if !tau.is_empty() {
        body["spec_tau"] = serde_json::Value::Object(tau);
        body["spec_accept_by_position"] = serde_json::Value::Object(by_position);
    }
}

fn insert_peer_probe_metrics(
    body: &mut serde_json::Value,
    metrics_scope: &MetricsScope,
    snapshot: impl FnOnce() -> memra_engine::pp::PeerProbeMetrics,
) {
    // Probe bypass/failure state and boundary traffic are process-wide safety telemetry.
    // Completion credentials must not learn cross-tenant traffic or device topology.
    if !metrics_scope.operator() {
        return;
    }
    let snapshot = snapshot();
    body["peer_probe_bypassed"] = json!(snapshot.bypassed);
    body["peer_probe_boundary_copies"] = json!(snapshot.boundary_copies);
    body["peer_probe_runtime_reprobes"] = json!(snapshot.runtime_probes);
    body["peer_probe_runtime_failures"] = json!(snapshot.runtime_failures);
    body["peer_probe_deferred_total"] = json!(snapshot.deferred_total);
    body["peer_probe_integrity_degraded"] = json!(snapshot.integrity_degraded);
    body["peer_probe_degraded_to_host_bounce"] = json!(snapshot.degraded_to_host_bounce);
}

/// Flat serving counters + engine-truth step latency percentiles.
async fn get_metrics(State(st): State<AppState>, headers: HeaderMap) -> Response {
    let metrics_scope = match authorize_metrics(&st.api_auth, &st.metrics_auth, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    let m = st.metrics.lock().map(|m| m.clone()).unwrap_or_default();
    // These counters describe the whole process, not the authenticated tenant. Preserve them for
    // the legacy single-key completion domain, but fail closed when a multi-tenant keyring caller
    // has no explicit operator scrape token.
    let mut body = if metrics_scope.process_wide() {
        json!({
            "admitted": m.admitted,
            "completed": m.completed,
            "tokens_out": m.tokens_out,
            "step_p50_ms": m.step_p50_ms,
            "step_p99_ms": m.step_p99_ms,
            // worker-truth prompt caching split (cached = resumed from any KV cache tier).
            "prompt_tokens_in": m.prompt_tokens_in,
            "cached_tokens_in": m.cached_tokens_in,
            // computed = actually primed; the denominator of the revenue multiplier
            // (billed prompt tokens / computed prompt tokens — tools/cache_economics.py).
            "computed_tokens_in": m.prompt_tokens_in.saturating_sub(m.cached_tokens_in),
            // Whole-session cache and admission observability (lane/cx-cachespec): cumulative
            // counters locate a latency slope; gauges show whether retired state is accumulating.
            "admission_session_defers": m.admission_session_defers,
            "admission_vram_defers": m.admission_vram_defers,
            "step_oom_parks": m.step_oom_parks,
            "continuation_pool_hits": m.continuation_pool_hits,
            "continuation_pool_evictions": m.continuation_pool_evictions,
            "plain_affinity_rewinds": m.plain_affinity_rewinds,
            "served_dspark": m.served_dspark,
            "served_spec": m.served_spec,
            "served_plain": m.served_plain,
            "spec_pool_hits": m.spec_pool_hits,
            "spec_pool_misses": m.spec_pool_misses,
            "spec_pool_affinity_rewinds": m.spec_pool_affinity_rewinds,
            "spec_pool_evictions": m.spec_pool_evictions,
            // lane/session-resume-sampler-predicate-20260820: the production answer to "does real
            // multi-turn traffic change sampler mid-session". Subset of spec_pool_misses.
            "spec_pool_sampler_refusals": m.spec_pool_sampler_refusals,
        })
    } else {
        json!({})
    };
    // Global prefix shape/volume and current capacity/VRAM are operator-only surfaces. The legacy
    // single-key domain retains its cumulative counters, while keyring completion credentials get
    // only their permitted tenant rows, including that tenant's own cache-hit ratio.
    if metrics_scope.operator() {
        if let Some(budget_health) = st.metering.as_ref().and_then(|m| m.limits_health()) {
            body["budget_source_reload_failed"] = json!(budget_health.source_reload_failed);
            body["budget_source_reload_consecutive"] =
                json!(budget_health.source_reload_consecutive);
            body["budget_source_available"] = json!(budget_health.source_available);
        }
        // Token-weighted global hit ratio + full prefix-cache probe/churn counters.
        body["cache_hit_token_ratio"] = json!(if m.prompt_tokens_in > 0 {
            m.cached_tokens_in as f64 / m.prompt_tokens_in as f64
        } else {
            0.0
        });
        body["prefix_cache_hits"] = json!(m.prefix_hits);
        body["prefix_cache_misses"] = json!(m.prefix_misses);
        body["prefix_cache_inserts"] = json!(m.prefix_inserts);
        body["prefix_cache_evictions"] = json!(m.prefix_evictions);
        body["prefix_cache_skips_budget"] = json!(m.prefix_skips_budget);
        body["prefix_cache_skips_pinned"] = json!(m.prefix_skips_pinned);
        body["prefix_cache_hit_tokens"] = json!(m.prefix_hit_tokens);
        // Pinned-host spill tier behind the prefix cache (lane/kv-host-spill-20260830;
        // MEMRA_KV_HOST_MB, default 0 = off). *_ms are cumulative copy wall-time: the
        // tick-stall receipt for the pod battery.
        body["prefix_host_entries"] = json!(m.prefix_host_entries);
        body["prefix_host_bytes"] = json!(m.prefix_host_bytes);
        body["prefix_host_demotions"] = json!(m.prefix_host_demotions);
        body["prefix_host_promotions"] = json!(m.prefix_host_promotions);
        body["prefix_host_demote_ms"] = json!(m.prefix_host_demote_ms);
        body["prefix_host_promote_ms"] = json!(m.prefix_host_promote_ms);
        body["prefix_host_rejected_allocs"] = json!(m.prefix_host_rejected_allocs);
        body["prefix_host_purges"] = json!(m.prefix_host_purges);
        body["prefix_host_purged_entries"] = json!(m.prefix_host_purged_entries);
        body["prefix_host_purged_bytes"] = json!(m.prefix_host_purged_bytes);
        body["prefix_host_tenant_rejects"] = json!(m.prefix_host_tenant_rejects);
        // Agent-pause demotion (MEMRA_KV_PAUSE_DEMOTE, lane/kv-pause-demote-20260831):
        // pause_demotes is a subset of prefix_host_demotions; pause_cancels counts armed
        // candidates whose session returned before the timer (or left nothing demotable).
        body["prefix_host_pause_demotes"] = json!(m.prefix_host_pause_demotes);
        body["prefix_host_pause_cancels"] = json!(m.prefix_host_pause_cancels);
        body["prefix_host_handoff_exports"] = json!(m.prefix_host_handoff_exports);
        body["prefix_host_handoff_imported_entries"] =
            json!(m.prefix_host_handoff_imported_entries);
        body["prefix_host_handoff_imported_bytes"] = json!(m.prefix_host_handoff_imported_bytes);
        body["prefix_host_handoff_skips"] = json!(m.prefix_host_handoff_skips);
        // KV budget flex (MEMRA_KV_FLEX, lane/kv-flex-20260831, tiering spec Arc G):
        // borrowed_bytes = current device prefix-cache residency above its configured
        // floor; sheds/shed_ms = borrowed-slice reclaims and their CUMULATIVE wall-time
        // (ms per shed = shed_ms / sheds, the capture-arrival zero-tax receipt).
        body["kv_flex_borrowed_bytes"] = json!(m.kv_flex_borrowed_bytes);
        body["kv_flex_sheds"] = json!(m.kv_flex_sheds);
        body["kv_flex_shed_ms"] = json!(m.kv_flex_shed_ms);
        // One sample per prefix-cache probe: served length on a hit, best LCP on a miss.
        // `edges` are lower bounds; the last bucket is unbounded.
        body["lcp_histogram"] = json!({
            "edges": worker::LCP_HIST_EDGES.to_vec(),
            "counts": m.lcp_hist.to_vec(),
        });
        // Valley signal (lane/darklane-training): seconds the worker has been COMPLETELY idle
        // (no active sessions, no queued admissions, no pending HTTP handoffs) — worker truth
        // via health phase + beat age + the PENDING_ADMITS gauge, no new hot-path cost.
        let idle_s = darklane::ValleySignal::new(st.health.clone()).idle_seconds();
        body["prefix_cache_entries"] = json!(m.prefix_entries);
        body["prefix_cache_bytes"] = json!(m.prefix_bytes);
        body["active_sessions"] = json!(m.active_sessions);
        body["queued_requests"] = json!(m.queued_requests);
        // Predictive-admission book (D2 gap G2, lane/d2-engine-gaps-20260831): per-model
        // in-flight sessions and the sum of their engine admission charges. Operator
        // scope: per-model load shape is cross-tenant information.
        body["admission_inflight"] = json!(m.admission_inflight);
        body["admission_booked_bytes"] = json!(m.admission_booked_bytes);
        body["continuation_pool_entries"] = json!(m.continuation_pool_entries);
        body["spec_pool_entries"] = json!(m.spec_pool_entries);
        body["cuda_driver_free_bytes"] = json!(m.cuda_driver_free_bytes);
        body["cuda_pool_reserved_bytes"] = json!(m.cuda_pool_reserved_bytes);
        body["cuda_pool_used_bytes"] = json!(m.cuda_pool_used_bytes);
        body["cuda_pool_cached_bytes"] = json!(m.cuda_pool_cached_bytes);
        if !m.constraint_compiler_fail_closed.is_empty() {
            body["constraint_compiler_fail_closed"] = serde_json::Value::Object(
                m.constraint_compiler_fail_closed
                    .iter()
                    .map(|(model, gauge)| {
                        let value = u8::from(gauge.load(std::sync::atomic::Ordering::Acquire));
                        (model.clone(), json!(value))
                    })
                    .collect(),
            );
        }
        body["serve_idle_seconds"] = json!((idle_s * 1000.0).round() / 1000.0);
    }
    // Per-tenant prompt/cached breakdown (composes with PC-ISO tenancy): keyring
    // deployments key rows by tenant (`t:<tenant>`), no-keyring by raw cache_salt
    // ("" = the default namespace). ABSENT until the first admit, so a fresh server's
    // /metrics is otherwise unchanged. Bounded rows; overflow aggregates in "(other)".
    if !m.ns_tokens.is_empty() {
        let tenants: serde_json::Map<String, serde_json::Value> = m
            .ns_tokens
            .iter()
            .filter(|(ns, _)| metrics_scope.includes(ns))
            .map(|(ns, [p, c])| {
                (
                    ns.clone(),
                    json!({
                        "prompt_tokens_in": p,
                        "cached_tokens_in": c,
                        "cache_hit_token_ratio": if *p > 0 { *c as f64 / *p as f64 } else { 0.0 },
                    }),
                )
            })
            .collect();
        if !tenants.is_empty() {
            body["tenants"] = serde_json::Value::Object(tenants);
        }
    }
    let adsd_suspect_total: serde_json::Map<String, serde_json::Value> = m
        .adsd_suspect_total
        .iter()
        .filter(|(tenant, _)| metrics_scope.includes(tenant))
        .map(|(tenant, total)| (tenant.clone(), json!(total)))
        .collect();
    if !adsd_suspect_total.is_empty() {
        body["adsd_suspect_total"] = serde_json::Value::Object(adsd_suspect_total);
    }
    // Background-job state is operator-only and absent unless MEMRA_BG_JOB armed the runner.
    if metrics_scope.operator()
        && let Some((bg, mode)) = &st.bg
    {
        body["bg"] = bg.to_json(mode);
    }
    // Spec-decode acceptance telemetry (lane/accept-telemetry — the llama.cpp #26389 /
    // vLLM per-draft-position counter schema). Per model, cumulative since model load
    // (models load once per process — counters reset on restart, never mid-run). The
    // block is ABSENT until a spec burst runs: spec-off deployments see the exact
    // pre-lane payload. accept_rate_per_pos[j] = P(position j accepted | round offered
    // position j) — sane spec decode decays monotonically from pos 0.
    if metrics_scope.operator() {
        let spec: serde_json::Map<String, serde_json::Value> = m
            .spec
            .iter()
            .map(|(model, t)| {
                let n_pos = t
                    .pos_drafted
                    .iter()
                    .rposition(|&d| d > 0)
                    .map_or(0, |p| p + 1);
                (
                    model.clone(),
                    json!({
                        "rounds": t.rounds,
                        "drafted": t.drafted,
                        "accepted": t.accepted,
                        "acceptance_rate": if t.drafted > 0 {
                            t.accepted as f64 / t.drafted as f64 } else { 0.0 },
                        "tokens_per_round": if t.rounds > 0 {
                            (t.accepted + t.rounds) as f64 / t.rounds as f64 } else { 0.0 },
                        "pos_drafted": t.pos_drafted[..n_pos].to_vec(),
                        "pos_accepted": t.pos_accepted[..n_pos].to_vec(),
                        "accept_rate_per_pos": (0..n_pos).map(|j| if t.pos_drafted[j] > 0 {
                            t.pos_accepted[j] as f64 / t.pos_drafted[j] as f64 } else { 0.0 })
                            .collect::<Vec<f64>>(),
                    }),
                )
            })
            .collect();
        if !spec.is_empty() {
            body["spec"] = serde_json::Value::Object(spec);
        }
    }
    insert_spec_acceptance_metrics(&mut body, &metrics_scope, || m.spec_window.clone());
    insert_dual_pp_metrics(&mut body, &metrics_scope, DualPpMetricsSnapshot::current);
    insert_pp_wave_metrics(&mut body, &metrics_scope, PpWaveMetricsSnapshot::current);
    insert_peer_probe_metrics(
        &mut body,
        &metrics_scope,
        memra_engine::pp::peer_probe_metrics,
    );
    Json(body).into_response()
}

#[derive(Debug, Default, Deserialize)]
struct ModelsQuery {
    #[serde(default)]
    schema: Option<String>,
}

fn models_openai_body(models: &[String]) -> serde_json::Value {
    let data: Vec<_> = models
        .iter()
        .map(|m| json!({ "id": m, "object": "model" }))
        .collect();
    json!({ "object": "list", "data": data })
}

/// The surface a model actually serves, defaulting to chat. All THREE catalog
/// feeds (`/v1/models`, `/models?schema=openrouter`, `/models?schema=openmodels`)
/// resolve it through here so they can never disagree about the same model — the
/// disagreement being exactly what a split fix would have created.
fn declared_surface(metadata: Option<&OpenRouterModelMetadata>) -> &'static str {
    match metadata.and_then(|m| m.surface.as_deref()) {
        Some("embedding") => "embedding",
        Some("rerank") => "rerank",
        _ => "chat",
    }
}

fn openrouter_supported_parameters(
    caps: Option<&ModelCaps>,
    max_output_length: Option<u64>,
    is_chat: bool,
) -> serde_json::Value {
    let mut parameters = serde_json::Map::new();
    // EVERY parameter below is a completion-request field. /v1/embeddings takes
    // {input, dimensions, encoding_format} and /v1/rerank takes {query, documents,
    // top_n} — neither accepts sampling, stop, seed, max_tokens, json_mode or
    // structured_outputs. Publishing them off the chat surface would repeat, on this
    // feed, the contradiction this change exists to remove: /v1/models declaring
    // structured_output=false for an embedder while this feed advertises
    // structured_outputs as an accepted boolean for the same model.
    if !is_chat {
        return serde_json::Value::Object(parameters);
    }
    for name in [
        "temperature",
        "top_p",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "stop",
    ] {
        parameters.insert(name.into(), json!({ "type": "unknown" }));
    }
    parameters.insert("top_k".into(), json!({ "type": "integer", "min": 0 }));
    parameters.insert(
        "seed".into(),
        json!({ "type": "integer", "min": 0, "max": JSON_SAFE_INTEGER_MAX }),
    );
    let mut max_tokens = json!({ "type": "integer", "min": 1, "unit": "token" });
    if let Some(max) = max_output_length {
        max_tokens["max"] = json!(max);
    }
    parameters.insert("max_tokens".into(), max_tokens);
    // Constrained decoding is NOT universal, and this catalog used to say it was. The dsv4
    // route refuses `response_format` by name. A template whose `<think>` tail opens
    // unconditionally with no `enable_thinking` switch is refused ONLY when its think-close
    // token contract is unknown (`ModelCaps::think_close` empty — GLM-5.3-Flash): with a known
    // close sequence, POST-THINK constrained decoding serves it (think runs unconstrained, the
    // grammar engages at the close token — lane/step37-postthink-grammar). This predicate
    // mirrors the ACTUAL refusal in `build_chat_request`, not a template heuristic: v0.123.0
    // shipped the heuristic form and advertised `structured_output: false` for step37 while the
    // server was serving schema-valid `response_format` on it (found by the 2026-09-01 claim
    // re-seal; live-verified both ways). Same predicate as the contract-v2 row's
    // `structured_output`, so the two catalogs cannot disagree about one model. Off the chat
    // surface (embedders, rerankers) nothing chat-shaped is advertised at all.
    if is_chat
        && caps.is_some_and(|c| {
            !c.dsv4 && !(c.qwen_think && !c.think_switch && c.think_close.is_empty())
        })
    {
        parameters.insert("json_mode".into(), json!({ "type": "boolean" }));
        parameters.insert("structured_outputs".into(), json!({ "type": "boolean" }));
    }
    if is_chat && caps.is_some_and(|c| c.tools_branch) {
        parameters.insert("tools".into(), json!({ "type": "boolean" }));
        parameters.insert(
            "tool_choice".into(),
            json!({ "type": "enum", "values": ["auto", "none"] }),
        );
    }
    if is_chat && caps.is_some_and(|c| c.qwen_think || c.effort_levels || c.gemma_think) {
        parameters.insert("reasoning".into(), json!({ "type": "boolean" }));
    }
    // glm5 has a three-rung effort ladder, and issue #75 made publishing it part
    // of the fix: an OpenRouter client tuning depth needs to see low|high|max as
    // the levels, not discover by experiment. `medium` is accepted and mapped to
    // high (`glm5_effort_level`), but the native rungs are what this feed states,
    // and an enum here that lists medium would advertise a rung the template does
    // not define. (glm5 also matches the generic `reasoning` boolean arm above
    // through qwen_think/effort_levels; that advertisement is a separate question
    // from this ladder, tracked in its own issue.)
    if is_chat && caps.is_some_and(|c| c.glm5) {
        parameters.insert(
            "reasoning_effort".into(),
            json!({ "type": "enum", "values": ["low", "high", "max"] }),
        );
    }
    serde_json::Value::Object(parameters)
}

/// The context window a catalog row is allowed to CLAIM: the checkpoint's trained
/// `context_length` capped by the deployment's operational envelope
/// (`max_prompt_length + max_output_length`) when the metadata pins both.
///
/// The trained figure is a training fact, not a serving claim. Admission already refuses a
/// `max_ctx` beyond the pinned envelope (`apply_model_request_limits`: "a tiny request could
/// reserve the model's full trained context and bypass the production shape's VRAM admission
/// contract"), but until 2026-08-30 every catalog body still advertised the raw trained value —
/// so a deployment whose shape cannot serve that window published it anyway. The receipt that
/// forced this: GLM-5.3-Flash declares 1,048,576 trained, and the 3-card resident serving shape
/// cannot prime it — the 1M deep prime died `layer 31: DSA k-pool selection failed:
/// DriverError(CUDA_ERROR_OUT_OF_MEMORY)` at a 97,242 MiB per-card peak
/// (`research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md`). A row must never
/// advertise a window the deployment has not pinned as admissible; with no envelope pinned the
/// trained value stands (a bare dev boot is not a customer catalog).
fn published_context_length(
    caps: Option<&ModelCaps>,
    metadata: Option<&OpenRouterModelMetadata>,
) -> Option<u64> {
    let trained = caps
        .map(|c| c.context_length as u64)
        .filter(|&value| value > 0)?;
    let envelope = metadata.and_then(|m| {
        let prompt = m.max_prompt_length?;
        let output = m.max_output_length?;
        prompt.checked_add(output)
    });
    Some(envelope.map_or(trained, |envelope| trained.min(envelope)))
}

fn model_entry_openrouter(
    name: &str,
    caps: Option<&ModelCaps>,
    metadata: Option<&OpenRouterModelMetadata>,
) -> serde_json::Value {
    let empty = OpenRouterModelMetadata::default();
    let metadata = metadata.unwrap_or(&empty);
    let context_length =
        published_context_length(caps, Some(metadata)).filter(|&v| v <= JSON_SAFE_INTEGER_MAX);
    let tokenizer = caps
        .map(|c| c.tokenizer.as_str())
        .filter(|tokenizer| !tokenizer.is_empty());

    let mut input = serde_json::Map::new();
    input.insert("type".into(), json!("text"));
    let mut supported_inputs = serde_json::Map::new();
    if let Some(value) = context_length {
        supported_inputs.insert(
            "max_context_length".into(),
            json!({ "value": value, "unit": "token" }),
        );
    }
    if let Some(value) = metadata.max_prompt_length {
        supported_inputs.insert(
            "max_prompt_length".into(),
            json!({ "value": value, "unit": "token" }),
        );
    }
    if !supported_inputs.is_empty() {
        input.insert(
            "supported_inputs".into(),
            serde_json::Value::Object(supported_inputs),
        );
    }
    let mut input_pricing = Vec::new();
    for (kind, cost) in [
        ("prompt", metadata.pricing.prompt.as_deref()),
        ("cached_prompt", metadata.pricing.cached_prompt.as_deref()),
        ("cache_write", metadata.pricing.cache_write.as_deref()),
    ] {
        if let Some(cost) = cost {
            input_pricing.push(json!({
                "type": kind,
                "unit": "token",
                "cost_usd": cost,
            }));
        }
    }
    if !input_pricing.is_empty() {
        input.insert("pricing".into(), serde_json::Value::Array(input_pricing));
    }
    let mut input_capacity = Vec::new();
    for (kind, value) in [
        ("prompt", metadata.capacity.prompt_tpm),
        ("cached_prompt", metadata.capacity.cached_prompt_tpm),
    ] {
        if let Some(value) = value {
            input_capacity.push(json!({
                "type": kind,
                "unit": "token",
                "per": "minute",
                "value": value,
            }));
        }
    }
    if !input_capacity.is_empty() {
        input.insert("capacity".into(), serde_json::Value::Array(input_capacity));
    }

    let or_surface = declared_surface(Some(metadata));
    let or_is_chat = or_surface == "chat";
    let mut output = serde_json::Map::new();
    // These strings come from the vendored Provider Monitor 2.4 schema this feed
    // stamps itself with — research/gateway-20260812/raw/sources/
    // openrouter-provider-schema-v2.4-20260812.json, `OutputModality`, a closed
    // oneOf whose branches enum `type` to text|image|video|speech|transcription|
    // embeddings|rerank|audio. They are NOT ours to choose: the wire enum is PLURAL
    // `embeddings` while the models.toml key is singular `embedding`, and there is no
    // `score` modality at all. A row matching no branch fails the whole document.
    output.insert(
        "type".into(),
        json!(match or_surface {
            "embedding" => "embeddings",
            "rerank" => "rerank",
            _ => "text",
        }),
    );
    output.insert(
        "supported_parameters".into(),
        openrouter_supported_parameters(caps, metadata.max_output_length, or_is_chat),
    );
    // The embeddings and rerank branches declare NO `streaming` property and are
    // additionalProperties:false, so the key must be ABSENT there — `false` is as
    // invalid as `true`. Chat keeps the byte-identical `true`.
    if or_is_chat {
        output.insert("streaming".into(), json!(true));
    }
    // Same rule as /v1/models' max_output_tokens: a surface that emits no completion
    // tokens advertises no ceiling, or a client reads it as a max_tokens to send.
    if let Some(value) = metadata.max_output_length
        && or_is_chat
    {
        output.insert(
            "max_length".into(),
            json!({ "value": value, "unit": "token" }),
        );
    }
    let mut output_pricing = Vec::new();
    for (kind, cost) in [
        ("completion", metadata.pricing.completion.as_deref()),
        (
            "internal_reasoning",
            metadata.pricing.internal_reasoning.as_deref(),
        ),
    ] {
        if let Some(cost) = cost {
            output_pricing.push(json!({
                "type": kind,
                "unit": "token",
                "cost_usd": cost,
            }));
        }
    }
    if !output_pricing.is_empty() {
        output.insert("pricing".into(), serde_json::Value::Array(output_pricing));
    }
    let mut output_capacity = Vec::new();
    if let Some(value) = metadata.capacity.completion_tpm {
        output_capacity.push(json!({
            "type": "completion",
            "unit": "token",
            "per": "minute",
            "value": value,
        }));
    }
    if let Some(value) = metadata.capacity.concurrency {
        output_capacity.push(json!({
            "type": "concurrency",
            "unit": "request",
            "value": value,
        }));
    }
    if !output_capacity.is_empty() {
        output.insert("capacity".into(), serde_json::Value::Array(output_capacity));
    }

    let mut entry = serde_json::Map::new();
    entry.insert("schema_version".into(), json!(OPENROUTER_SCHEMA_VERSION));
    entry.insert("id".into(), json!(name));
    entry.insert("name".into(), json!(name));
    if let Some(value) = metadata.hugging_face_id.as_deref() {
        entry.insert("hugging_face_id".into(), json!(value));
    }
    if let Some(value) = metadata.created {
        entry.insert("created".into(), json!(value));
    }
    if let Some(value) = metadata.quantization.as_deref() {
        entry.insert("quantization".into(), json!(value));
    }
    if let Some(value) = tokenizer {
        entry.insert("tokenizer".into(), json!(value));
    }
    if let Some(value) = metadata.description.as_deref() {
        entry.insert("description".into(), json!(value));
    }
    let mut input_modalities = vec![serde_json::Value::Object(input)];
    for m in &metadata.input_modalities {
        let mut extra = serde_json::Map::new();
        extra.insert("type".into(), json!(m));
        if let Some(cost) = metadata.pricing.prompt.as_deref() {
            // image content bills as ordinary prompt tokens (the pad run IS the prompt)
            extra.insert(
                "pricing".into(),
                json!([{ "type": "prompt", "unit": "token", "cost_usd": cost }]),
            );
        }
        input_modalities.push(serde_json::Value::Object(extra));
    }
    entry.insert(
        "input_modalities".into(),
        serde_json::Value::Array(input_modalities),
    );
    entry.insert(
        "output_modalities".into(),
        serde_json::Value::Array(vec![serde_json::Value::Object(output)]),
    );
    if let Some(cost) = metadata.pricing.request.as_deref() {
        entry.insert(
            "pricing".into(),
            json!([{ "type": "request", "unit": "request", "cost_usd": cost }]),
        );
    }
    if let Some(value) = metadata.capacity.request_rpm {
        entry.insert(
            "capacity".into(),
            json!([{
                "type": "request",
                "unit": "request",
                "per": "minute",
                "value": value,
            }]),
        );
    }
    if let Some(value) = metadata.is_ready {
        entry.insert("is_ready".into(), json!(value));
    }
    if let Some(value) = metadata.is_free {
        entry.insert("is_free".into(), json!(value));
    }
    if let Some(value) = metadata.discount_to_user {
        entry.insert("discount_to_user".into(), json!(value));
    }
    if let Some(value) = metadata.openrouter_slug.as_deref() {
        entry.insert("openrouter".into(), json!({ "slug": value }));
    }
    if !metadata.datacenters.is_empty() {
        entry.insert("datacenters".into(), json!(metadata.datacenters));
    }
    let mut compliance = serde_json::Map::new();
    if let Some(value) = metadata.zdr {
        compliance.insert("zdr".into(), json!(value));
    }
    if let Some(value) = metadata.hipaa {
        compliance.insert("hipaa".into(), json!(value));
    }
    if !compliance.is_empty() {
        entry.insert("compliance".into(), serde_json::Value::Object(compliance));
    }
    serde_json::Value::Object(entry)
}

fn models_openrouter_body(st: &AppState) -> serde_json::Value {
    let data: Vec<_> = st
        .models
        .iter()
        .map(|model| {
            model_entry_openrouter(model, st.caps.get(model), st.openrouter_metadata.get(model))
        })
        .collect();
    json!({ "data": data })
}

fn model_entry_openmodels(
    name: &str,
    caps: Option<&ModelCaps>,
    metadata: Option<&OpenRouterModelMetadata>,
) -> Result<serde_json::Value, String> {
    let metadata = metadata.ok_or_else(|| {
        format!("OpenModels feed requires MEMRA_MODEL_METADATA for model {name:?}")
    })?;
    let context_length = published_context_length(caps, Some(metadata))
        .filter(|&value| value <= JSON_SAFE_INTEGER_MAX)
        .ok_or_else(|| format!("OpenModels feed requires context_length for model {name:?}"))?;
    let created = metadata
        .created
        .ok_or_else(|| format!("OpenModels feed requires created for model {name:?}"))?;
    let max_output_length = metadata
        .max_output_length
        .ok_or_else(|| format!("OpenModels feed requires max_output_length for model {name:?}"))?;
    let prompt = metadata
        .pricing
        .prompt
        .as_deref()
        .ok_or_else(|| format!("OpenModels feed requires pricing.prompt for model {name:?}"))?;
    let completion =
        metadata.pricing.completion.as_deref().ok_or_else(|| {
            format!("OpenModels feed requires pricing.completion for model {name:?}")
        })?;
    let input_cache_read = metadata.pricing.cached_prompt.as_deref().ok_or_else(|| {
        format!("OpenModels feed requires pricing.cached_prompt for model {name:?}")
    })?;
    let is_ready = metadata
        .is_ready
        .ok_or_else(|| format!("OpenModels feed requires is_ready for model {name:?}"))?;
    let is_free = metadata
        .is_free
        .ok_or_else(|| format!("OpenModels feed requires is_free for model {name:?}"))?;
    let discount_to_user = metadata
        .discount_to_user
        .ok_or_else(|| format!("OpenModels feed requires discount_to_user for model {name:?}"))?;

    let mut pricing = serde_json::Map::new();
    pricing.insert("prompt".into(), json!(prompt));
    pricing.insert("completion".into(), json!(completion));
    pricing.insert("input_cache_read".into(), json!(input_cache_read));
    if let Some(value) = metadata.pricing.request.as_deref() {
        pricing.insert("request".into(), json!(value));
    }

    let om_surface = declared_surface(Some(metadata));
    let om_is_chat = om_surface == "chat";
    let mut supported_features = Vec::new();
    if om_is_chat && caps.is_some_and(|c| c.tools_branch) {
        supported_features.push("tool_calling");
    }
    if om_is_chat && caps.is_some_and(|c| c.qwen_think || c.effort_levels || c.gemma_think) {
        supported_features.push("reasoning");
    }

    let mut entry = serde_json::Map::new();
    entry.insert("id".into(), json!(name));
    entry.insert("name".into(), json!(name));
    entry.insert("created".into(), json!(created));
    entry.insert("input_modalities".into(), json!(["text"]));
    entry.insert(
        "output_modalities".into(),
        json!(match om_surface {
            "embedding" => ["embeddings"],
            "rerank" => ["rerank"],
            _ => ["text"],
        }),
    );
    entry.insert("context_length".into(), json!(context_length));
    entry.insert("max_output_length".into(), json!(max_output_length));
    // OpenModels' current snapshot importer defaults an omitted currency to CNY.
    // Declare the USD unit used by every pricing string so it cannot apply FX conversion.
    entry.insert("currency".into(), json!("USD"));
    entry.insert("pricing".into(), serde_json::Value::Object(pricing));
    entry.insert("supported_features".into(), json!(supported_features));
    entry.insert("is_ready".into(), json!(is_ready));
    entry.insert("is_free".into(), json!(is_free));
    entry.insert("discount_to_user".into(), json!(discount_to_user));
    Ok(serde_json::Value::Object(entry))
}

fn models_openmodels_body(st: &AppState) -> Result<serde_json::Value, String> {
    let data: Result<Vec<_>, _> = st
        .models
        .iter()
        .map(|model| {
            model_entry_openmodels(model, st.caps.get(model), st.openrouter_metadata.get(model))
        })
        .collect();
    Ok(json!({ "data": data? }))
}

async fn list_models(State(st): State<AppState>, Query(query): Query<ModelsQuery>) -> Response {
    match query.schema.as_deref() {
        None | Some("openai") => Json(models_openai_body(st.models.as_ref())).into_response(),
        Some("openrouter") => Json(models_openrouter_body(&st)).into_response(),
        Some("openmodels") => match models_openmodels_body(&st) {
            Ok(body) => Json(body).into_response(),
            Err(error) => bad_request(&error, Some("schema")),
        },
        Some(schema) => bad_request(
            &format!(
                "unsupported models schema {schema:?}; expected openai, openrouter, or openmodels"
            ),
            Some("schema"),
        ),
    }
}

/// One /v1/models entry in EXACTLY the router-marketplace contract-v2 shape — no extra
/// keys ("Do not design a custom catalog or pricing format"; the checker rejects
/// unknown fields). The richer OpenRouter/OpenModels shapes stay on /models?schema=.
/// Values are worker truth from the loaded plan (ModelCaps probed at spawn) plus the
/// model's MEMRA_MODEL_METADATA entry — the same source the request ledger bills from,
/// so the advertised price can never drift from the charged one. Prices render as
/// per-1M-token decimal STRINGS via exact decimal shift; null when a rate does not apply.
fn model_entry_v1(
    name: &str,
    caps: Option<&ModelCaps>,
    metadata: Option<&OpenRouterModelMetadata>,
) -> serde_json::Value {
    let ctx = published_context_length(caps, metadata);
    // Same thinking-capability predicate as the OpenRouter catalog body: any of the
    // three template dialects (qwen think tail, level-consuming effort string, gemma
    // thought channel) means the model reasons and the reasoning knobs are live.
    let thinking = caps.is_some_and(|c| c.qwen_think || c.effort_levels || c.gemma_think || c.dsv4);
    // rung-3 model-row honesty: the dsv4 route refuses response_format by name and
    // serves no prefix cache (n_cached honestly 0) — its row must not claim either.
    let is_dsv4 = caps.is_some_and(|c| c.dsv4);
    let per_1m = |v: Option<&str>| match v.and_then(per_million_price) {
        Some(p) => json!(p),
        None => serde_json::Value::Null,
    };
    let owned_by = metadata
        .and_then(|m| m.owned_by.as_deref())
        .unwrap_or_else(|| name.split('/').next().unwrap_or(name));
    let mut input_modalities = vec!["text"];
    if let Some(meta) = metadata {
        input_modalities.extend(meta.input_modalities.iter().map(String::as_str));
    }
    let lifecycle = metadata.and_then(|m| m.lifecycle.as_ref());
    let reliability = metadata.and_then(|m| m.reliability.as_ref());
    // The row a client SDK reads to decide HOW to call this model. A non-chat model
    // advertised as chat sends the caller to the wrong endpoint with the wrong body,
    // so type/endpoints/output_modalities/capabilities all follow the declared surface
    // rather than a hardcoded chat literal (2026-08-28: qwen3-embedding-8b and
    // qwen3-reranker-8b were published as chat models with tools+streaming).
    let surface = declared_surface(metadata);
    let (model_type, endpoints, output_modalities) = match surface {
        // `type` mirrors the models.toml vocabulary (singular, like `surface`);
        // output modalities use the SAME wire enum the 2.4 schema pins, because
        // inventing a second vocabulary is what produced `score` in the first place.
        "embedding" => ("embedding", vec!["embeddings"], vec!["embeddings"]),
        "rerank" => ("rerank", vec!["rerank"], vec!["rerank"]),
        _ => ("chat", vec!["chat/completions"], vec!["text"]),
    };
    let is_chat = surface == "chat";
    json!({
        "id": name,
        "name": name,
        "object": "model",
        "owned_by": owned_by,
        "type": model_type,
        "context_length": ctx,
        // A non-chat surface emits no completion tokens; advertising an output ceiling
        // for it invites a max_tokens the endpoint will never honour.
        "max_output_tokens": if is_chat { metadata.and_then(|m| m.max_output_length) } else { None },
        "endpoints": endpoints,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "capabilities": {
            // Every chat-shaped capability is FALSE off the chat surface: an embedder
            // does not stream, does not call tools, and does not reason.
            "streaming": is_chat,
            "tools": is_chat && caps.is_some_and(|c| c.tools_branch),
            // A switchless force-open `<think>` tail refuses `response_format` ONLY when
            // its think-close contract is unknown (`think_close` empty — GLM-5.3-Flash);
            // with a known close sequence POST-THINK constrained decoding serves it
            // (lane/step37-postthink-grammar), so the advertisement mirrors the actual
            // `build_chat_request` refusal. The heuristic form of this predicate shipped in
            // v0.123.0 and advertised false for step37 while the server served schema-valid
            // constrained output on it.
            "structured_output": is_chat
                && !is_dsv4
                && !caps.is_some_and(|c| c.qwen_think && !c.think_switch && c.think_close.is_empty()),
            "reasoning": is_chat && thinking,
            "prompt_caching": is_chat && !is_dsv4,
        },
        "pricing": {
            "currency": "USD",
            "unit": "per_1m_tokens",
            "input": per_1m(metadata.and_then(|m| m.pricing.prompt.as_deref())),
            "output": per_1m(metadata.and_then(|m| m.pricing.completion.as_deref())),
            "cached_input": per_1m(metadata.and_then(|m| m.pricing.cached_prompt.as_deref())),
            "cache_write": per_1m(metadata.and_then(|m| m.pricing.cache_write.as_deref())),
            // Per-REQUEST minimum in USD (not a token rate): our request price, "0" default.
            "minimum_request": metadata
                .and_then(|m| m.pricing.request.as_deref())
                .unwrap_or("0"),
        },
        "lifecycle": {
            "status": lifecycle.and_then(|l| l.status.as_deref()).unwrap_or("active"),
            "deprecation_at": lifecycle.and_then(|l| l.deprecation_at.as_deref()),
            "retirement_at": lifecycle.and_then(|l| l.retirement_at.as_deref()),
            "replacement_model_id": lifecycle.and_then(|l| l.replacement_model_id.as_deref()),
        },
        "reliability": {
            "first_token_timeout_seconds":
                reliability.and_then(|r| r.first_token_timeout_seconds).unwrap_or(120),
            "completion_timeout_seconds":
                reliability.and_then(|r| r.completion_timeout_seconds).unwrap_or(900),
            "stream_idle_timeout_seconds":
                reliability.and_then(|r| r.stream_idle_timeout_seconds).unwrap_or(60),
            "capacity_scope":
                reliability.and_then(|r| r.capacity_scope.as_deref()).unwrap_or("model_region"),
        },
    })
}

/// GET /v1/models — the existing OpenAI/OpenRouter catalog listing, enriched with per-model
/// metadata from the loaded plan (context length, tokenizer, instruct family).
async fn list_models_v1(State(st): State<AppState>) -> impl IntoResponse {
    let data: Vec<_> = st
        .models
        .iter()
        .map(|m| model_entry_v1(m, st.caps.get(m), st.openrouter_metadata.get(m)))
        .collect();
    let mut body = json!({
        "object": "list",
        "contract_version": "2.0",
        "data": data,
    });
    // Provider block (contract v2): operator identity from the metadata file, error
    // contract from server truth — 429 rate limits and 503 overload both carry
    // Retry-After (+ the retry-after-ms twin), quota exhaustion is the stable
    // insufficient_balance code on 402, and every response echoes x-request-id.
    if let Some(provider) = st.provider_metadata.as_ref() {
        body["provider"] = json!({
            "id": provider.id,
            "status_url": provider.status_url,
            "support_contact": provider.support_contact,
            "incident_contact": provider.incident_contact,
            "regions": provider.regions,
            "request_id_header": "x-request-id",
            "error_contract": {
                "rate_limit_status": 429,
                "overload_status": 503,
                "retry_after_header": "Retry-After",
                "account_quota_error_codes": ["insufficient_balance"],
            },
        });
    }
    Json(body)
}

/// Per-lane counters + engine-truth interactive step latency (sidecar-compatible shape —
/// the x-lane QoS gate's receipts endpoint).
async fn yield_metrics(State(st): State<AppState>, headers: HeaderMap) -> Response {
    let metrics_scope = match authorize_metrics(&st.api_auth, &st.metrics_auth, &headers) {
        Ok(scope) => scope,
        Err(response) => return response,
    };
    if !metrics_scope.process_wide() {
        return error_response(
            StatusCode::FORBIDDEN,
            "completion api keys do not authorize process-wide yield metrics; configure \
             MEMRA_METRICS_TOKEN",
            "authentication_error",
            None,
        );
    }
    let m = st.metrics.lock().map(|m| m.clone()).unwrap_or_default();
    let lane = |i: usize| {
        json!({
            "admitted": m.lane_admitted[i], "shed": m.lane_shed[i],
            "completed": m.lane_completed[i], "tokens_out": m.lane_tokens[i],
        })
    };
    let mut body = json!({
        "lanes": {
            "interactive": lane(0), "judge": lane(1), "harvest": lane(2),
        },
        "interactive_step_ms": { "p50": m.step_p50_ms, "p99": m.step_p99_ms },
    });
    if metrics_scope.operator() {
        body["batch_size_last"] = json!(m.batch_size_last);
    }
    Json(body).into_response()
}

/// Wait for the worker's admission verdict before committing a streaming response. Successful
/// admission publishes `PromptUsage` immediately, so this does not wait for a potentially slow
/// first token. Queueing intentionally keeps the request pre-header until capacity is available.
///
/// WHY THE PEEK MATTERS MORE THAN IT LOOKS (audit §OpenRouter uptime): once the first byte of
/// a 200 is written, the response is COMMITTED — a router cannot fail over, and a mid-stream
/// death counts against uptime. Catching an admission refusal here converts a would-be
/// mid-stream failure into a clean pre-header 429/503 that the client's own retry handles.
///
/// The 429 body now goes through `engine_error_body` (G6). It used to be
/// `{"error": "<string>"}` — a BARE STRING where every OpenAI SDK expects an object, which
/// made shed errors render as a blank message in every client that parses the standard shape.
async fn peek_admission(
    mut rx: worker::EventReceiver,
) -> Result<worker::EventReceiver, (Response, &'static str)> {
    match rx.recv().await {
        // Any pre-admission failure — a shed, a rejected allocation, a load fault — is
        // answered as a normal HTTP error with its own class instead of being smuggled into a
        // stream. Classification is the producer's (worker::EngineError), so this no longer
        // string-matches a "shed:" prefix that only ever existed as an in-band sentinel.
        Some(Event::Error(e)) => {
            let error_code = engine_error_code(e.class);
            Err((engine_error_response(&e), error_code))
        }
        first => {
            let (tx2, rx2) = worker::event_channel();
            if let Some(ev) = first {
                let _ = tx2.send(ev);
            }
            tokio::spawn(forward_events(rx, tx2));
            Ok(rx2)
        }
    }
}

/// Pump worker events to the response side, and — the part that is load-bearing for
/// cancellation — drop the worker-side receiver AS SOON AS the consumer goes away, not at
/// the next event.
///
/// A plain `while let Some(ev) = rx.recv().await { tx2.send(ev) }` loop only discovers a
/// dropped consumer when the NEXT event arrives, so a request producing nothing yet (a
/// long prefill) kept its worker channel open indefinitely: the abort the worker looks for
/// (`req.tx.is_closed()`) never appeared, and neither a client disconnect nor a deadline
/// miss could actually cancel it. Selecting on `tx2.closed()` closes that gap for every
/// consumer-side exit — client hang-up, deadline, or handler return.
async fn forward_events(mut rx: worker::EventReceiver, tx2: worker::EventSender) {
    loop {
        tokio::select! {
            biased;
            () = tx2.closed() => break,
            ev = rx.recv() => match ev {
                Some(ev) => {
                    if tx2.send(ev).is_err() {
                        break;
                    }
                }
                None => break,
            },
        }
    }
}

/// STREAMING TTFT DEADLINE (lane/deadline-billing-20260823): hold the response PRE-HEADER
/// until the first generated event (token, done, or fault) or the deadline, whichever is
/// first. A deadline miss can then be an honest, retryable 408 — once the first byte of a
/// 200 is written the response is COMMITTED (see `peek_admission`), and a mid-stream error
/// chunk is neither a status a router can act on nor a promise-keeping "you don't pay"
/// signal. This extends the existing pre-header posture (queueing already holds
/// pre-header until admission) through prefill: headers now commit at first token, which
/// is bounded by the deadline (<= 90 s), inside the fronting proxy's ~100 s
/// time-to-headers ceiling.
///
/// Pre-token events (PromptUsage) are buffered and re-injected in order, so the stream
/// consumer's receipt discipline is unchanged. On a miss the receiver — and with it the
/// worker-side event channel — is dropped, which IS the cancel signal: the worker retires
/// closed-channel requests queued or active at the next tick.
async fn peek_first_token(
    mut rx: worker::EventReceiver,
    deadline: RequestDeadline,
) -> Result<worker::EventReceiver, ()> {
    let mut buffered: Vec<Event> = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline.at, rx.recv()).await {
            Err(_) => return Err(()), // deadline elapsed; dropping rx cancels generation
            Ok(None) => break,        // worker gone: the stream's closed-channel law handles it
            Ok(Some(ev)) => {
                let first_delivery = matches!(
                    ev,
                    Event::Token { .. } | Event::Done { .. } | Event::Error(_)
                );
                buffered.push(ev);
                if first_delivery {
                    break;
                }
            }
        }
    }
    let (tx2, rx2) = worker::event_channel();
    for ev in buffered {
        let _ = tx2.send(ev);
    }
    tokio::spawn(forward_events(rx, tx2));
    Ok(rx2)
}

/// Build the (GenParams, SamplerConfig, stop, prompt) from a request body.
#[cfg(test)]
/// Test helper: the raw-prompt build with NO per-model vendor defaults declared, i.e. the
/// API-standard fallback path. Tests that exercise the vendor-default substitution pass their
/// own `SamplingDefaults` to `build_request_with_trace` directly.
fn build_request(
    req: &CompletionReq,
    tx: worker::EventSender,
    lane: lanes::Lane,
    affinity: Option<String>,
) -> Request {
    build_request_with_trace(req, tx, lane, affinity, None, &SamplingDefaults::default())
}

fn build_request_with_trace(
    req: &CompletionReq,
    tx: worker::EventSender,
    lane: lanes::Lane,
    affinity: Option<String>,
    ttft: Option<Arc<ttft::Trace>>,
    sampling_defaults: &SamplingDefaults,
) -> Request {
    let params = GenParams {
        max_new: req.max_tokens.unwrap_or(worker::MAX_NEW_CTX_BOUNDED),
        max_ctx: req.max_ctx,
        eos: Vec::new(), // worker adds the model's own eos id
    };
    // Same resolver the chat/messages/responses surfaces use — the raw-prompt surface gets the
    // model's vendor-recommended sampling for omitted fields too (standard-surface law). Before
    // this lane it could not: its fields were bare `f32`s, so "omitted" was indistinguishable
    // from "1.0" and the per-model default was silently unreachable here.
    let sampler_cfg = resolve_sampler_config(req.into(), sampling_defaults);
    Request {
        model: req.model.clone(),
        prompt_ids: req.prompt_ids.clone(),
        prompt_text: req.prompt.clone(),
        chat: req.chat,
        chat_turns: Vec::new(),
        tools_json: Vec::new(),
        tools_struct: Vec::new(),
        think: ThinkMode::Default,
        reasoning_effort: None, // /v1/completions is a raw-prompt surface (no template render)
        params,
        sampler_cfg,
        stop_strings: req.stop.clone().into_vec(),
        trace_id: req.trace_id.clone(),
        // Stamped with the envelope id by the handler before submission (the builder
        // does not see the envelope).
        request_id: String::new(),
        admit_predict_logged: false,
        max_prompt_tokens: None,
        cache_ns: cache_namespace(&req.cache_salt),
        affinity,
        lane,
        grammar: None, // /v1/completions carries no response_format (chat surface only)
        prepared_constraint: None,
        constraint_ready: None,
        oom_retries: 0, // step-OOM park budget: fresh from the HTTP layer (lane/admit-oom)
        spec_k_replay: None,
        prepared_prompt: None,
        capture: None,      // set only by the embeddings/rerank routes
        images: Vec::new(), // /v1/completions is a raw-text surface
        gemma_images: Vec::new(),
        glm5_images: Vec::new(),
        step_images: Vec::new(),
        vision_memory: None,
        wire_deadline: None, // stamped by the handler at submission (with request_id)
        ttft,
        tx,
    }
}

/// Everything the chat handler derives from the request body before submitting to the
/// worker: the worker Request plus the parser arming state for the response side.
struct ChatPlan {
    request: Request,
    /// Some(parser) when a <tools> block was rendered — the ONLY case the emission parser
    /// runs (non-tools traffic keeps byte-identical streams, chunk boundaries included).
    parser: Option<ToolStreamParser>,
    /// Header-planned vision units awaiting their post-admission pixel decode
    /// (`decode_pending_vision`) — see the hermes decode-bomb fix, 2026-08-23.
    pending_images: Vec<PendingVisionUnit>,
    pending_gemma: Vec<PendingGemmaImage>,
    pending_glm5: Vec<PendingGlm5Image>,
    pending_step: Vec<PendingStepImage>,
    /// Process-wide patch-memory reservation carried into the worker request. It is released when
    /// the worker drops the request after completion or cancellation, so streaming responses do
    /// not reopen the pre-admission memory window.
    vision_memory: Option<VisionMemoryPermit>,
}

pub(crate) fn request_has_vision(req: &ChatCompletionReq) -> bool {
    req.messages.iter().any(|message| {
        message.content.as_array().is_some_and(|parts| {
            parts.iter().any(|part| {
                matches!(
                    part.get("type").and_then(serde_json::Value::as_str),
                    Some("image_url" | "video_url")
                )
            })
        })
    })
}

fn planned_vision_bytes(plan: &ChatPlan) -> Result<usize, String> {
    let mut total = 0usize;
    let mut add = |bytes: usize| {
        total = total.checked_add(bytes).ok_or_else(|| {
            "vision patch memory reservation overflowed while planning".to_string()
        })?;
        Ok::<(), String>(())
    };
    for unit in &plan.pending_images {
        let bytes = match unit {
            PendingVisionUnit::Still { gh, gw, .. } => gh
                .checked_mul(*gw)
                .and_then(|n| n.checked_mul(memra_engine::vision::V_PATCH_IN))
                .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?,
            PendingVisionUnit::Video { groups, .. } => {
                groups.iter().try_fold(0usize, |total, group| {
                    let bytes = group
                        .gh
                        .checked_mul(group.gw)
                        .and_then(|n| n.checked_mul(memra_engine::vision::V_PATCH_IN))
                        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
                        .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?;
                    total.checked_add(bytes).ok_or_else(|| {
                        "vision patch memory reservation overflowed while planning".to_string()
                    })
                })?
            }
        };
        add(bytes)?;
    }
    for unit in &plan.pending_gemma {
        let bytes = unit
            .gw
            .checked_mul(unit.gh)
            .and_then(|n| n.checked_mul(memra_engine::vision_gemma::GV_PATCH_IN))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?;
        add(bytes)?;
    }
    for unit in &plan.pending_glm5 {
        let bytes = unit
            .gh
            .checked_mul(unit.gw)
            .and_then(|n| n.checked_mul(memra_engine::vision_glm5::G5V_PATCH_IN))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?;
        add(bytes)?;
    }
    for unit in &plan.pending_step {
        use memra_engine::vision_step::{SV_GRID_MAIN, SV_GRID_TILE, SV_PATCH_IN};
        // one 52x52 main view + n_tiles 36x36 crops, 588 f32 per patch row
        let patches = unit
            .plan
            .n_tiles
            .checked_mul(SV_GRID_TILE * SV_GRID_TILE)
            .and_then(|n| n.checked_add(SV_GRID_MAIN * SV_GRID_MAIN))
            .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?;
        let bytes = patches
            .checked_mul(SV_PATCH_IN)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "vision patch memory reservation overflowed".to_string())?;
        add(bytes)?;
    }
    Ok(total)
}

pub(crate) fn reserve_vision_memory(
    plan: &ChatPlan,
) -> Result<Option<VisionMemoryPermit>, VisionMemoryError> {
    let bytes = planned_vision_bytes(plan).map_err(VisionMemoryError::Request)?;
    try_reserve_vision_memory(bytes)
}

#[cfg(test)]
fn build_chat_request(
    req: ChatCompletionReq,
    caps: Option<&ModelCaps>,
    tx: worker::EventSender,
    lane: lanes::Lane,
    affinity: Option<String>,
) -> Result<ChatPlan, String> {
    // Test helper: no operator metadata, so the arch caps are the only default source — the
    // pre-lane behavior. Vendor-default tests pass their own `ModelSamplingDefaults`.
    let defaults = ModelSamplingDefaults::resolve(None, caps);
    build_chat_request_with_trace(req, caps, tx, lane, affinity, None, None, &defaults)
}

/// `default_effort` is the model's operator-declared `default_reasoning_effort`
/// (MEMRA_MODEL_METADATA) — the serve callers pass it from the metadata map; None keeps
/// the model template's own default for the unset case (every model without the knob is
/// byte-identical to before the knob existed).
///
/// `sampling_defaults` is the same idea for the sampling fields (lane/vendor-default-sampling,
/// 2026-08-19): the model vendor's recommendation, substituted only into fields the client left
/// out. Built by `ModelSamplingDefaults::resolve` from the operator metadata block plus the
/// arch caps, and passed rather than computed here so the raw-prompt surface can share the
/// exact same resolver. It carries BOTH vendor arms (lane/per-mode-sampling, 2026-08-24);
/// the request's RESOLVED thinking mode picks the arm below, AFTER `parse_think` and the
/// constraint gate have settled it — so the arm always matches the mode the model actually
/// runs in, on every surface that funnels through this builder.
#[allow(clippy::too_many_arguments)]
fn build_chat_request_with_trace(
    req: ChatCompletionReq,
    caps: Option<&ModelCaps>,
    tx: worker::EventSender,
    lane: lanes::Lane,
    affinity: Option<String>,
    ttft: Option<Arc<ttft::Trace>>,
    default_effort: Option<&str>,
    sampling_defaults: &ModelSamplingDefaults,
) -> Result<ChatPlan, String> {
    req.stop.validate()?;
    // The client's own expression is snapshotted here; the omitted fields resolve to a
    // vendor arm only once the thinking mode is final (see `sampler_cfg` below).
    let client_sampling: ClientSampling = (&req).into();
    let tool_choice = parse_tool_choice(&req.tool_choice)?;
    // Template honesty gate (serve-st lane, 2026-08-04): a directory checkpoint
    // (safetensors/repack) with NO chat template cannot honestly serve chat — 400 with a
    // clear message instead of silently rendering fallback ChatML the model never saw.
    // GGUF models keep the historical fallback (chat_ok=true there regardless).
    if let Some(c) = caps
        && !c.chat_ok
    {
        return Err(format!(
            "model {:?} has no chat template (checkpoint carries neither \
                 tokenizer_config.json chat_template nor chat_template.jinja) — \
                 /v1/chat/completions unavailable; use /v1/completions with a raw prompt",
            req.model
        ));
    }
    let vllm_switch = resolve_vllm_think_switch(req.enable_thinking, &req.chat_template_kwargs)?;
    let (mut think, effort_level, think_client_explicit) = parse_think(
        &req.reasoning_effort,
        &req.reasoning,
        vllm_switch,
        req.include_reasoning,
        default_effort,
        // Templates with a real rung ABOVE `high`: deepseek-v4's BEYOND_MAX prefix and
        // GLM-5.3-Flash's `Reasoning Effort: Max` (its own default). Clamping xhigh/max/ultra
        // into `high` on these silently drops the tier the client asked for.
        caps.is_some_and(|c| c.dsv4 || c.glm5),
    )?;
    // Does this model's template express a reasoning DEPTH at all, and can it be turned off?
    // Both are template-probed capabilities, never inferred from the family name (house law:
    // a control is never assumed from a shared loader, format or lineage).
    let level_template = caps
        .map(|c| c.effort_levels || c.dsv4 || c.qwen_effort || c.glm5)
        .unwrap_or(false);
    // SILENT-IGNORE GATE (lane/reasoning-control-20260823, corrected here). A client that
    // explicitly asked for reasoning OFF, on a model whose template opens a `<think>` tail it
    // cannot close, cannot be served that request: the prompt would render think-open anyway
    // and the reply would stream a full reasoning block behind a 200. That is the owner's named
    // unacceptable case — asking for non-reasoning and getting reasoning — so it is a named 400.
    // Scoped to a CLIENT-explicit off-request (`think_client_explicit`): a deployment
    // `default_reasoning_effort` must never 400 a caller who sent nothing.
    //
    // TWO DIALECTS ARE EXEMPT, and both were false positives of the marker pair as PR #33 shipped
    // it (found by review before release, no customer ever saw them):
    //   - `dsv4`: the deepseek-v4 renderer honours NoThink through its own `chat` thinking mode
    //     (a closed `</think>`), so it needs no `enable_thinking` marker to turn reasoning off.
    //     Latent rather than live today only because encoding-keyed artifacts carry no template
    //     string; keyed here explicitly so it cannot become live by accident.
    //   - a template with NO think tail at all (`!qwen_think`) — gemma4's thought channel and
    //     hy3's `no_think` header both close cleanly and never matched this gate.
    // step35 is deliberately NOT exempt even though it consumes effort levels: its `<think>` tail
    // is unconditional, so its documented `none|minimal -> "Reasoning: low"` clamp answered an
    // off-request WITH reasoning at the lowest rung. That is the unacceptable case wearing a
    // clamp, and the 400 replaces it.
    if think_client_explicit
        && think == ThinkMode::NoThink
        && let Some(c) = caps
        && c.qwen_think
        && !c.think_switch
        && !c.dsv4
    {
        return Err(format!(
            "model {:?} cannot disable reasoning: its chat template opens a think \
                     tail unconditionally and carries no enable_thinking switch, so \
                     reasoning_effort/enable_thinking cannot turn it off on this model",
            req.model
        ));
    }
    // GRADATION ON A BINARY MODEL: TRANSLATE, never refuse (coordinator ruling 2026-08-23,
    // resolving two owner rulings that pulled against each other). A first cut of this lane
    // REFUSED a graded level on a model whose template has no depth input — the construction
    // proof being that low/medium/high render bytes identical to an unset request there. The
    // refusal was correct arithmetic and the wrong law: the owner explicitly authorised
    // normalisation ("it can be translated into one schema that we use"), the standard-surface
    // law makes real-CLI round-trips a launch gate, and stock codex (`reasoning.effort:"xhigh"`)
    // and stock Claude Code (`output_config.effort:"xhigh"`) send a graded level on EVERY
    // request — the 400 broke default-config agent sessions against ornith, the exact model we
    // serve to agents.
    //
    // The owner's unacceptable case is asking for NON-reasoning and getting reasoning. A caller
    // sending `xhigh` asked for reasoning and gets reasoning — the translation keeps the
    // promise. So the mapping, documented here and in SERVING.md rather than implied:
    //
    //   graded level (low|medium|high|xhigh) on a binary-switch model  =>  reasoning ON.
    //
    // No code runs here to do it: `parse_think` already resolved every ON rung to
    // `ThinkMode::Think`, and the `level_template` delivery gate below drops the rung string for
    // templates with no ladder — so the rendered prompt is byte-identical to an explicit
    // `reasoning:{"enabled":true}` by construction (pinned by
    // `a_graded_level_on_a_binary_model_translates_to_reasoning_on`). The named 400s stay for
    // what is genuinely unhonourable: unknown keys, wrong types, contradictions, and the
    // off-request a template cannot honour (the gate above).
    // Effort-level templates: the client's reasoning_effort is a RENDER input, not a think
    // switch — step35/hy3 (`effort_levels`: "Reasoning: {level}\n\n" / header level), qwen3.8
    // (`qwen_effort`: the `xhigh|medium|low` instruction sentence at the head of the system
    // turn) and deepseek-v4 (`dsv4`: the encoding's effort-prompt prefix, resolved against the
    // artifact's detected encoding revision — 0731 ladder low/high/max where "high" is a
    // REAL prefix; the preview treats "high" as its documented no-op and "medium" renders
    // as the default level under both, the never-corrupt clamp). Gate on the capability so
    // every other model's prompt stays byte-identical.
    let reasoning_effort = if level_template { effort_level } else { None };
    // response_format -> grammar spec (constrained decoding). None/text = unconstrained,
    // the exact legacy path; unknown/malformed forms are loud 400s.
    let grammar = constrained::parse_response_format(req.response_format.as_ref())?;
    // GRAMMAR x THINK (measured live 2026-08-03): the grammar masks from the FIRST
    // generated token, so an open <think> tail can never be closed — the forced JSON
    // lands in the think segment and `content` comes back empty. Constrained requests
    // force the template's no-think switch — that path is byte-identical to before this
    // lane. A think-tail template WITHOUT the switch serves POST-THINK constrained
    // decoding instead (lane/step37-postthink-grammar, 2026-08-30) when its think-close
    // token contract is derivable (`ModelCaps::think_close`): the think phase runs
    // unconstrained exactly as the model was trained (EOS banned, so the response cannot
    // end inside think), and the grammar clamps every token from the close on. The worker
    // arms the gate at admission from the same load-time contract; nothing else is
    // plumbed through the request. A think-forced template with NO derivable close
    // contract keeps the loud 400 (honesty gate), never a silent
    // constrain-from-token-1 stream.
    if grammar.is_some()
        && let Some(c) = caps
        && c.qwen_think
        && think != ThinkMode::NoThink
    {
        if c.think_switch {
            think = ThinkMode::NoThink;
        } else if c.think_close.is_empty() {
            return Err(
                "response_format requires the model's think channel to close \
                                before the grammar can engage, but this chat template has \
                                neither an enable_thinking switch nor a recognizable \
                                think-close token sequence"
                    .into(),
            );
        }
        // else: POST-THINK constrained decoding — think stays ON (the
        // template's only honest mode); the worker engages the grammar at the
        // close token(s).
    }

    // PER-MODE VENDOR DEFAULTS (lane/per-mode-sampling, 2026-08-24): the thinking mode is
    // final from here on, so this is the one point where an omitted sampling field becomes
    // a number — the resolved mode picks the vendor arm, then the same client-wins law as
    // ever (`resolve_sampler_config`: client value > arm default > API-standard). A model
    // without a `non_thinking_sampling` table gets its single arm for every mode,
    // byte-identical to when this call sat at the top of the function.
    let sampler_cfg = resolve_sampler_config(client_sampling, sampling_defaults.for_mode(think));

    // tool_choice "none" = OpenAI "the model will not call tools": the prompt renders
    // WITHOUT the tools block (byte-identical to a no-tools request) and no parser runs.
    let (tools_json, tools_struct, schemas) =
        if !req.tools.is_empty() && tool_choice == ToolChoice::Auto {
            prepare_tools(&req.tools)?
        } else {
            (Vec::new(), Vec::new(), HashMap::new())
        };

    let mut turns: Vec<TmplTurn> = Vec::with_capacity(req.messages.len());
    let mut images: Vec<PendingVisionUnit> = Vec::new();
    let mut gemma_images: Vec<PendingGemmaImage> = Vec::new();
    let mut glm5_images: Vec<PendingGlm5Image> = Vec::new();
    let mut step_images: Vec<PendingStepImage> = Vec::new();
    let mut next_video = 0usize;
    for msg in &req.messages {
        let content = content_to_text_vision(
            &msg.content,
            &mut images,
            &mut gemma_images,
            &mut glm5_images,
            &mut step_images,
            &mut next_video,
        )
        .map_err(|e| format!("{} message: {e}", msg.role))?;
        let tool_calls = msg
            .tool_calls
            .iter()
            .map(render_req_tool_call)
            .collect::<Result<Vec<_>, _>>()?;
        if !tool_calls.is_empty() && msg.role != "assistant" {
            return Err("tool_calls are only valid on assistant messages".into());
        }
        // OpenAI's `developer` role is their o-series rename of `system`; chat templates
        // know only `system`, so normalize here (matches OpenAI's own equivalence).
        let role = if msg.role == "developer" {
            "system".to_string()
        } else {
            msg.role.clone()
        };
        turns.push(TmplTurn {
            role,
            content,
            tool_calls,
            // gemma4-only fields; the qwen/step dialects ignore them.
            reasoning: msg.reasoning.clone().filter(|r| !r.is_empty()),
            tool_call_id: msg.tool_call_id.clone(),
            tool_name: msg.name.clone(),
            tool_responses: Vec::new(),
            // dsv4-only fields: the OpenAI serve surface carries no `task` head, and dsv4
            // request-level tools flow via `tools_struct` (folded onto the leading system
            // turn by the dsv4 arm); every other dialect ignores both.
            task: None,
            tools: Vec::new(),
        });
    }

    // Capability gate: reject tools on models whose template has no tools branch BEFORE
    // the request reaches the GPU worker (clean 400 instead of a mid-stream error).
    let has_tool_features = !tools_json.is_empty()
        || turns
            .iter()
            .any(|t| t.role == "tool" || !t.tool_calls.is_empty());
    if has_tool_features && !caps.map(|c| c.tools_branch).unwrap_or(false) {
        return Err(format!(
            "model {:?} chat template has no tools branch",
            req.model
        ));
    }

    // Parser think gate: the rendered prompt ends with an OPEN think tail (template
    // default, not switched off by reasoning_effort on a switch-carrying template).
    let think_open = caps
        .map(|c| c.qwen_think && !(think == ThinkMode::NoThink && c.think_switch))
        .unwrap_or(false);
    // REASONING SEPARATION (gap-scan F13): think-segment text routes to the OpenRouter
    // `reasoning` response field on EVERY chat request against a think-open prompt —
    // content is post-think only. Tools requests keep the full tool-call scanner; non-tools
    // think-open requests get the reasoning-only splitter (post-think text unscanned).
    // Models without a think tail keep a byte-identical no-parser stream.
    //
    // REASONING IS ALWAYS DELIVERED (owner ruling 2026-08-23). There is no longer a
    // suppression path: `include_reasoning:false` and `reasoning.exclude:true` are handled far
    // upstream in `parse_think`, where they turn reasoning OFF instead of hiding it. Reasoning
    // tokens are output tokens and are billed as output, so withholding them was charging for
    // output we did not send; the drop capability is deleted from the parser rather than merely
    // left unreachable, so the third state (generate, bill, withhold) cannot be reintroduced by
    // wiring a flag back to it.
    // gemma4 tooluse dialect: tools rendered into the gemma template need the gemma call
    // parser (`<|tool_call>call:NAME{…}<tool_call|>` + thought channels), NOT the qwen
    // `<tool_call>`/`<parameter=…>` scanner. Keyed on the gemma marker so qwen/step keep
    // their own scanner.
    let gemma_tools = !tools_json.is_empty() && caps.map(|c| c.gemma_think).unwrap_or(false);
    // deepseek-v4 dialect: thinking mode maps to encoding_dsv4's thinking_mode (Default/Think
    // -> thinking, an open `<think>` tail; NoThink -> chat, a closed `</think>`). The parser
    // splits `</think>` reasoning + `<｜DSML｜tool_calls>` blocks. Armed on EVERY dsv4 chat
    // request (like gemma_think): tools present -> full call parser; else a reasoning splitter
    // that also passes content through cleanly.
    let is_dsv4 = caps.map(|c| c.dsv4).unwrap_or(false);
    let dsv4_think_open = is_dsv4 && think != ThinkMode::NoThink;
    let dsv4_tools = is_dsv4 && !tools_struct.is_empty();
    // GLM-5.3-Flash dialect: `<think>` reasoning (unconditional tail, no separator newlines
    // after the close) plus `<tool_call>NAME<arg_key>…` calls. Armed on EVERY glm5 chat request
    // like the gemma/dsv4 arms: with tools the full call parser, without them the reasoning
    // splitter — the qwen scanner's `<function=` body grammar never matches this wire, so
    // before this branch a glm5 tool call would have surfaced VERBATIM as content.
    let glm5 = caps.map(|c| c.glm5).unwrap_or(false);
    // Tencent HY3 dialect: reasoning closes with `</think:opensource>` and calls use the
    // suffixed `<tool_calls:opensource>` protocol. Armed on think-open or tools, like dsv4.
    let is_hy3 = caps.map(|c| c.hy3).unwrap_or(false);
    let hy3_think_open = is_hy3 && think == ThinkMode::Think;
    let hy3_tools = is_hy3 && !tools_json.is_empty();
    let parser = if glm5 {
        Some(ToolStreamParser::glm5(think_open, schemas))
    } else if is_hy3 && (hy3_tools || hy3_think_open) {
        Some(ToolStreamParser::hy3(schemas, hy3_think_open))
    } else if is_dsv4 && (dsv4_tools || dsv4_think_open) {
        Some(ToolStreamParser::dsv4(dsv4_think_open))
    } else if gemma_tools {
        Some(ToolStreamParser::gemma_tools())
    } else if !tools_json.is_empty() {
        Some(ToolStreamParser::new(schemas, think_open))
    } else if think_open {
        Some(ToolStreamParser::reasoning_only())
    } else if caps.map(|c| c.gemma_think).unwrap_or(false) {
        // gemma4 thought-channel dialect (lane/gemma4-serve-gaps): thought text used to
        // land VERBATIM in content — `<|channel>thought\n…` with thinking on, and the tags
        // leaked with it (think-smoke receipt, step-sku lane). Armed on EVERY gemma4 chat
        // request, not just thinking-on: the closed-channel prompt still leaves the model
        // free to open a channel mid-stream (observed live), and the template's own
        // strip_thinking law applies wherever the tags appear. gemma4 templates carry no
        // tools branch, so this arm never competes with the tool scanner.
        Some(ToolStreamParser::gemma_thought())
    } else {
        None
    };

    Ok(ChatPlan {
        request: Request {
            model: req.model,
            prompt_ids: Vec::new(),
            prompt_text: String::new(),
            chat: false,
            chat_turns: turns,
            tools_json,
            tools_struct,
            think,
            reasoning_effort,
            params: GenParams {
                max_new: req.max_tokens.unwrap_or(worker::MAX_NEW_CTX_BOUNDED),
                max_ctx: req.max_ctx,
                eos: Vec::new(),
            },
            sampler_cfg,
            stop_strings: {
                // gemma4 tooluse: the model emits `<|tool_call>call:…<tool_call|>` and would
                // then run past its handoff into a hallucinated `<|tool_response>`; stop when
                // the call completes (scoped to gemma tool requests — never global). The stop
                // token stays in the stream (not a silent eos) so the parser closes the span.
                let mut stops = req.stop.into_vec();
                if gemma_tools {
                    stops.push("<tool_call|>".to_string());
                }
                // deepseek-v4 tool requests: stop when the DSML tool_calls block closes, so the
                // model does not run past its handoff into a hallucinated `<tool_result>`
                // (scoped to dsv4 tool requests, never global; the close stays in the stream so
                // the parser finishes the span — same law as gemma's `<tool_call|>`).
                if dsv4_tools {
                    stops.push("</\u{ff5c}DSML\u{ff5c}tool_calls>".to_string());
                }
                // HY3 tool requests: stop on the native suffixed tool_calls close. Keep the
                // marker in the stream so the parser can close and emit every call.
                if hy3_tools {
                    stops.push("</tool_calls:opensource>".to_string());
                }
                stops
            },
            trace_id: None,
            // Stamped with the envelope id by the handler before submission (the plan
            // builder does not see the envelope).
            request_id: String::new(),
            admit_predict_logged: false,
            max_prompt_tokens: None,
            cache_ns: cache_namespace(&req.cache_salt),
            affinity,
            lane,
            grammar,
            prepared_constraint: None,
            constraint_ready: None,
            oom_retries: 0, // step-OOM park budget: fresh from the HTTP layer (lane/admit-oom)
            spec_k_replay: None,
            prepared_prompt: None,
            // Filled by decode_pending_vision AFTER budget admission (hermes
            // decode-bomb finding, fixed 2026-08-23) — the pad runs above were rendered
            // from header-planned grids, so admission prices the full vision prompt
            // without a single canvas expanding.
            images: Vec::new(),
            gemma_images: Vec::new(),
            glm5_images: Vec::new(),
            step_images: Vec::new(),
            capture: None, // set only by the embeddings/rerank routes
            vision_memory: None,
            wire_deadline: None, // stamped by the handler at submission (with request_id)
            ttft,
            tx,
        },
        parser,
        pending_images: images,
        pending_gemma: gemma_images,
        pending_glm5: glm5_images,
        pending_step: step_images,
        vision_memory: None,
    })
}

/// Phase 2 of the vision path: decode the planned stills into patch rows, AFTER budget
/// admission (hermes decode-bomb finding, fixed 2026-08-23). Order is preserved — the
/// worker aligns pad runs 1:1 with `images`. Each decoded grid must equal its planned
/// grid: the pad runs are already rendered from the plan, so a mismatch (a container
/// whose header lies about dimensions) refuses rather than desyncing runs from units.
fn decode_pending_vision(plan: &mut ChatPlan) -> Result<(), String> {
    for (i, unit) in plan.pending_images.drain(..).enumerate() {
        match unit {
            PendingVisionUnit::Still { bytes, gh, gw } => {
                let prep = memra_engine::vision_pre::prep_image_bytes(&bytes)
                    .map_err(|e| format!("image {}: {e}", i + 1))?;
                if (prep.gh, prep.gw) != (gh, gw) {
                    return Err(format!(
                        "image {}: decoded grid {}x{} differs from its header-planned grid {gh}x{gw} — refusing (pad runs already rendered)",
                        i + 1,
                        prep.gh,
                        prep.gw
                    ));
                }
                plan.request
                    .images
                    .push(memra_engine::vision_pre::VisionUnit { prep, video: None });
            }
            PendingVisionUnit::Video {
                bytes,
                groups,
                video,
            } => {
                let prepared = memra_engine::vision_pre::prep_video_gif(&bytes)
                    .map_err(|e| format!("video {}: {e}", i + 1))?;
                if prepared.groups.len() != groups.len() {
                    return Err(format!(
                        "video {}: decoded {} groups differ from its header-planned {} groups",
                        i + 1,
                        prepared.groups.len(),
                        groups.len()
                    ));
                }
                for ((group, prep), timestamp) in
                    groups.iter().zip(prepared.groups).zip(prepared.timestamps)
                {
                    if (prep.gh, prep.gw) != (group.gh, group.gw) {
                        return Err(format!(
                            "video {}: decoded grid {}x{} differs from its header-planned grid {}x{}",
                            i + 1,
                            prep.gh,
                            prep.gw,
                            group.gh,
                            group.gw
                        ));
                    }
                    if (timestamp - group.timestamp).abs() > 0.001 {
                        return Err(format!(
                            "video {}: decoded timestamp {timestamp:.3} differs from its header-planned timestamp {:.3}",
                            i + 1,
                            group.timestamp
                        ));
                    }
                    plan.request
                        .images
                        .push(memra_engine::vision_pre::VisionUnit {
                            prep,
                            video: Some(video),
                        });
                }
            }
        }
    }
    for (i, unit) in plan.pending_gemma.drain(..).enumerate() {
        let (patches, gw, gh) = memra_engine::vision_gemma::gemma_prep_image(&unit.bytes)
            .map_err(|e| format!("image {}: {e}", i + 1))?;
        if (gw, gh) != (unit.gw, unit.gh) {
            return Err(format!(
                "image {}: decoded grid {gw}x{gh} differs from its header-planned grid {}x{} — refusing (pad runs already rendered)",
                i + 1,
                unit.gw,
                unit.gh
            ));
        }
        plan.request
            .gemma_images
            .push(memra_engine::vision_gemma::GemmaVisionUnit { patches, gw, gh });
    }
    for (i, unit) in plan.pending_glm5.drain(..).enumerate() {
        let (patches, gh, gw) = memra_engine::vision_glm5::glm5_prep_image(&unit.bytes)
            .map_err(|e| format!("image {}: {e}", i + 1))?;
        if (gh, gw) != (unit.gh, unit.gw) {
            return Err(format!(
                "image {}: decoded grid {gh}x{gw} differs from its header-planned grid {}x{} — refusing (placeholder runs already rendered)",
                i + 1,
                unit.gh,
                unit.gw
            ));
        }
        plan.request
            .glm5_images
            .push(memra_engine::vision_glm5::Glm5VisionUnit { patches, gh, gw });
    }
    for (i, unit) in plan.pending_step.drain(..).enumerate() {
        let prepped = memra_engine::vision_step::step_prep_image(&unit.bytes)
            .map_err(|e| format!("image {}: {e}", i + 1))?;
        if prepped.tiles.len() != unit.plan.n_tiles
            || prepped.newline_mask != unit.plan.newline_mask
        {
            return Err(format!(
                "image {}: decoded tiling ({} tiles) differs from its header-planned tiling \
                 ({} tiles) — refusing (pad runs already rendered)",
                i + 1,
                prepped.tiles.len(),
                unit.plan.n_tiles
            ));
        }
        plan.request.step_images.push(prepped);
    }
    Ok(())
}

/// Resolve the request's tenant identity (lane/api-keys, 2026-08-05). The law lives in
/// `auth::authenticate_with`; this wraps the startup-resolved auth sources:
///   MEMRA_API_KEYS keyring match -> that key's tenant/lane-class/rate-limit;
///   MEMRA_API_KEY single-key match -> tenant "default" (back-compat: the daily driver
///     and every serve script keep working unchanged, keyring configured or not);
///   neither configured -> open, tenant "default";
///   otherwise Err: Unknown -> 401 (OpenAI authentication_error), Disabled -> 403.
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

fn authentication_error(why: auth::AuthDenied) -> Response {
    match why {
        auth::AuthDenied::Unknown => error_response(
            StatusCode::UNAUTHORIZED,
            "invalid api key",
            "authentication_error",
            None,
        ),
        auth::AuthDenied::Disabled => error_response(
            StatusCode::FORBIDDEN,
            "api key is disabled",
            "authentication_error",
            None,
        ),
    }
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn authenticate(api_auth: &ApiAuth, headers: &HeaderMap) -> Result<auth::TenantCtx, Response> {
    auth::authenticate_with(
        api_auth.keyring,
        api_auth.single_key.as_deref(),
        bearer_token(headers),
    )
    .map_err(authentication_error)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MetricsScope {
    All,
    CompletionDomain,
    Tenant(String),
}

impl MetricsScope {
    fn operator(&self) -> bool {
        matches!(self, MetricsScope::All)
    }

    fn process_wide(&self) -> bool {
        matches!(self, MetricsScope::All | MetricsScope::CompletionDomain)
    }

    fn includes(&self, tenant_row: &str) -> bool {
        match self {
            MetricsScope::All | MetricsScope::CompletionDomain => true,
            MetricsScope::Tenant(tenant) => tenant == tenant_row,
        }
    }
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn authorize_metrics(
    api_auth: &ApiAuth,
    metrics_auth: &MetricsAuth,
    headers: &HeaderMap,
) -> Result<MetricsScope, Response> {
    if !metrics_auth.required {
        return Ok(MetricsScope::All);
    }
    let Some(candidate) = bearer_token(headers) else {
        return Err(authentication_error(auth::AuthDenied::Unknown));
    };
    if let Some(token) = metrics_auth.token.as_deref() {
        if auth::constant_time_secret_eq(token, candidate) {
            return Ok(MetricsScope::All);
        }
        if api_auth.configured() {
            return match auth::authenticate_with(
                api_auth.keyring,
                api_auth.single_key.as_deref(),
                Some(candidate),
            ) {
                Ok(_) => Err(error_response(
                    StatusCode::FORBIDDEN,
                    "completion api keys do not authorize metrics while \
                     MEMRA_METRICS_TOKEN is configured",
                    "authentication_error",
                    None,
                )),
                Err(why) => Err(authentication_error(why)),
            };
        }
        return Err(authentication_error(auth::AuthDenied::Unknown));
    }
    if api_auth.configured() {
        let tenant = authenticate(api_auth, headers)?;
        return Ok(if api_auth.keyring.is_some() {
            MetricsScope::Tenant(format!("t:{}", tenant.tenant))
        } else {
            // Without a keyring there is one completion tenancy domain. Its metering
            // rows are raw cache_salt values, so they all belong to this caller. It is
            // still a completion credential, not an operator scrape principal.
            MetricsScope::CompletionDomain
        });
    }
    Err(authentication_error(auth::AuthDenied::Unknown))
}

/// Lane resolution with the tenant's lane class applied: interactive-class keys keep the
/// legacy behavior exactly (default interactive, any x-lane honored); batch-class keys
/// DEFAULT to harvest and are refused the protected interactive lane (403, loud — the
/// QoS gate exists to protect interactive from bulk traffic, so a bulk key cannot claim
/// the protected class by omission or by header).
#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn lane_for_tenant(
    headers: &axum::http::HeaderMap,
    tenant: &auth::TenantCtx,
) -> Result<lanes::Lane, Response> {
    let requested = match headers.get("x-lane").map(|v| v.to_str().unwrap_or("?")) {
        None => None,
        // A bad x-lane really is a client bug, so 400 is the right status — but the body has to
        // be an OpenAI-compat error OBJECT like every other refusal on this surface. It used to
        // be a bare `{"error":"unknown x-lane ..."}` string, which makes `e.body["error"]["type"]`
        // an index error in every SDK that parses the standard shape.
        Some(v) => Some(lanes::Lane::parse(v).ok_or_else(|| {
            error_response_coded(
                StatusCode::BAD_REQUEST,
                &format!("unknown x-lane {v:?}; expected one of interactive, judge, harvest"),
                "invalid_request_error",
                Some("x-lane"),
                Some("invalid_lane"),
            )
        })?),
    };
    match tenant.lane_class {
        auth::LaneClass::Interactive => Ok(requested.unwrap_or(lanes::Lane::Interactive)),
        auth::LaneClass::Batch => match requested {
            None => Ok(lanes::Lane::Harvest),
            Some(lanes::Lane::Interactive) => Err(error_response(
                StatusCode::FORBIDDEN,
                "this api key is batch-class: x-lane interactive is not permitted \
                 (use judge or harvest)",
                "authentication_error",
                Some("x-lane"),
            )),
            Some(l) => Ok(l),
        },
    }
}

/// The tenant-scoped PC-ISO namespace: keyring configured -> `t:<tenant>\x1f<salt>`
/// (a tenant's keys share cache, different tenants never — auth::scope_namespace);
/// no keyring -> the validated raw salt. Invalid values fail at the HTTP boundary.
fn tenant_namespace(
    tenant: &auth::TenantCtx,
    cache_salt: &Option<String>,
) -> Result<String, &'static str> {
    let keyring_configured = auth::global().is_some();
    let raw = validate_cache_namespace(cache_salt, keyring_configured)?;
    if keyring_configured {
        Ok(auth::scope_namespace(&tenant.tenant, &raw))
    } else {
        Ok(raw)
    }
}

/// METER SEAM (public-repo half): one flat log line per admitted request with the tenant
/// identity — the private fork's metering layer parses these for per-tenant usage/billing;
/// the public repo only emits. Completion accounting stays on the existing worker-truth
/// usage/abort lines; this line binds request-id -> tenant -> model/lane at admission.
fn meter_admit(env: &Envelope, tenant: &auth::TenantCtx, model: &str, lane: lanes::Lane) {
    eprintln!(
        "[meter] admit id={} tenant={} lane={} model={:?}",
        env.id,
        tenant.tenant,
        lane.as_str(),
        model
    );
}

fn apply_model_request_limits(
    request: &mut Request,
    metadata: Option<&OpenRouterModelMetadata>,
    caps: Option<&ModelCaps>,
) -> Result<(), (String, &'static str)> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let max_prompt = metadata
        .max_prompt_length
        .map(usize::try_from)
        .transpose()
        .map_err(|_| {
            (
                "configured model prompt limit does not fit this platform".into(),
                "model",
            )
        })?;
    let max_output = metadata
        .max_output_length
        .map(usize::try_from)
        .transpose()
        .map_err(|_| {
            (
                "configured model output limit does not fit this platform".into(),
                "model",
            )
        })?;

    request.max_prompt_tokens = max_prompt;
    if let Some(max_output) = max_output {
        if request.params.max_new == worker::MAX_NEW_CTX_BOUNDED {
            request.params.max_new = metadata
                .default_output_length
                .map(usize::try_from)
                .transpose()
                .map_err(|_| {
                    (
                        "configured default output length does not fit this platform".into(),
                        "model",
                    )
                })?
                .unwrap_or(max_output);
        } else if request.params.max_new > max_output {
            return Err((
                format!(
                    "max_tokens {} exceeds configured model maximum {max_output}",
                    request.params.max_new
                ),
                "max_tokens",
            ));
        }
    }

    // `max_ctx` is a memra extension. Refuse a client-selected allocation larger than the
    // advertised prompt+output envelope: otherwise a tiny request could reserve the model's
    // full trained context and bypass the production shape's VRAM admission contract.
    if let (Some(max_prompt), Some(max_output), Some(requested_ctx)) =
        (max_prompt, max_output, request.params.max_ctx)
    {
        let operational_ctx = max_prompt
            .checked_add(max_output)
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| {
                (
                    "configured model context envelope overflowed".into(),
                    "model",
                )
            })?;
        let operational_ctx = caps
            .map(|caps| caps.context_length)
            .filter(|&context| context > 0)
            .map_or(operational_ctx, |context| operational_ctx.min(context));
        if requested_ctx > operational_ctx {
            return Err((
                format!(
                    "max_ctx {requested_ctx} exceeds configured model envelope {operational_ctx}"
                ),
                "max_ctx",
            ));
        }
    }
    Ok(())
}

/// The request's effective completion-token bound for the receipt row (D2 gap G4):
/// `params.max_new` after `apply_model_request_limits` resolution, `None` when it is
/// still the context-bounded sentinel.
fn effective_max_tokens(request: &worker::Request) -> Option<u64> {
    (request.params.max_new != worker::MAX_NEW_CTX_BOUNDED).then_some(request.params.max_new as u64)
}

#[allow(clippy::too_many_arguments)]
fn start_request_receipt(
    st: &AppState,
    env: &Envelope,
    tenant: &auth::TenantCtx,
    model: &str,
    route: &'static str,
    lane: lanes::Lane,
    stream: bool,
    max_tokens: Option<u64>,
    reserved_ctx: Option<u64>,
    budget_permit: Option<metering::Permit>,
) -> Option<Box<dyn metering::Receipt>> {
    st.metering.as_ref().map(|accounting| {
        accounting.open(
            &metering::RequestMeta {
                request_id: &env.id,
                tenant: &tenant.tenant,
                principal: tenant.key_prefix.as_deref(),
                model,
                route,
                lane: lane.as_str(),
                stream,
                max_tokens,
                reserved_ctx,
            },
            budget_permit,
        )
    })
}

/// Attach capture to a successful-admission receipt when the tenant is marked. The
/// prompt payload is built lazily — unmarked tenants (the overwhelming majority of
/// traffic) pay only the receipt's `wants_capture` flag, set once at open. The
/// settle-time re-check inside the implementation remains the authoritative
/// capture decision.
fn arm_capture(
    mut receipt: Option<Box<dyn metering::Receipt>>,
    prompt: impl FnOnce() -> serde_json::Value,
) -> Option<Box<dyn metering::Receipt>> {
    if let Some(receipt) = receipt.as_mut()
        && receipt.wants_capture()
    {
        receipt.arm_capture(prompt());
    }
    receipt
}

/// The capture row's prompt payload: the messages array as the caller sent it
/// (role/content/tool_calls), rebuilt from the parsed request. Content stays the
/// original JSON value, so string and array-of-parts shapes round-trip unchanged.
fn capture_chat_messages(messages: &[ChatMessage]) -> serde_json::Value {
    serde_json::Value::Array(
        messages
            .iter()
            .map(|message| {
                let mut row = json!({ "role": message.role, "content": message.content });
                if !message.tool_calls.is_empty() {
                    row["tool_calls"] = serde_json::Value::Array(
                        message
                            .tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "function": {
                                        "name": call.function.name,
                                        "arguments": call.function.arguments,
                                    },
                                })
                            })
                            .collect(),
                    );
                }
                row
            })
            .collect(),
    )
}

enum BudgetRejection {
    Invalid(String),
    Insufficient,
    Unenrolled,
    /// The authenticated KEY's spend cap is reached (the tenant may still have
    /// balance). Distinct 402 code: the recovery is raising the key's cap.
    PrincipalCapped,
    Unavailable(String),
}

impl BudgetRejection {
    fn into_response(self) -> (Response, &'static str) {
        match self {
            Self::Invalid(message) => (bad_request(&message, Some("prompt")), "invalid_request"),
            Self::Insufficient => (
                error_response_coded(
                    StatusCode::PAYMENT_REQUIRED,
                    "tenant prepaid balance is insufficient for this request",
                    "insufficient_balance",
                    None,
                    Some("insufficient_balance"),
                ),
                "insufficient_balance",
            ),
            Self::Unenrolled => (
                error_response_coded(
                    StatusCode::PAYMENT_REQUIRED,
                    "tenant is not enrolled for prepaid billing",
                    "tenant_not_enrolled",
                    None,
                    Some("tenant_not_enrolled"),
                ),
                "tenant_not_enrolled",
            ),
            Self::PrincipalCapped => (
                error_response_coded(
                    StatusCode::PAYMENT_REQUIRED,
                    "this API key's spend cap is reached; raise or clear the key's cap to continue",
                    "key_spend_cap_reached",
                    None,
                    Some("key_spend_cap_reached"),
                ),
                "key_spend_cap_reached",
            ),
            Self::Unavailable(err) => {
                eprintln!("[budget] ERROR: admission unavailable: {err}");
                (
                    error_response_coded(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "tenant budget accounting is unavailable",
                        "server_error",
                        None,
                        Some("tenant_budget_unavailable"),
                    ),
                    "tenant_budget_unavailable",
                )
            }
        }
    }
}

fn prepare_budget_prompt(
    request: &mut Request,
    tokenizer: Option<&Tokenizer>,
) -> Result<usize, String> {
    if let Some(error) = worker::prompt_source_limit_error(request) {
        return Err(error);
    }
    if request.prepared_prompt.is_none() {
        if let Some(trace) = request.ttft.as_ref() {
            trace.mark_tokenize_start();
        }
        let prompt = if !request.prompt_ids.is_empty() {
            request.prompt_ids.clone()
        } else if !request.chat_turns.is_empty() {
            let tokenizer = tokenizer.ok_or("reservation tokenizer is unavailable")?;
            // The SHARED fast-path predicate (worker::plain_chat_render_path) — this is the
            // render that actually serves: the worker's `prepare` only re-renders when
            // `prepared_prompt` is still None, and this budget-admission path fills it first.
            // v0.109.1's first cut fixed the worker copies only, and the live probe showed
            // why one predicate must exist ONCE: unset q38 chats still served the bare bytes
            // because THIS third copy kept routing them down the legacy render.
            let plain = worker::plain_chat_render_path(
                &request.tools_json,
                &request.think,
                request.reasoning_effort.as_deref(),
                &request.chat_turns,
                tokenizer.has_qwen_effort_ladder(),
            );
            let rendered = if plain {
                let messages: Vec<_> = request
                    .chat_turns
                    .iter()
                    .map(|turn| (turn.role.as_str(), turn.content.as_str()))
                    .collect();
                tokenizer.apply_chat_template(&messages, true)
            } else {
                tokenizer
                    .apply_chat_template_tools_ex(
                        &request.chat_turns,
                        true,
                        &request.tools_json,
                        &request.tools_struct,
                        request.think,
                        request.reasoning_effort.as_deref(),
                    )
                    .map_err(|err| format!("chat template: {err}"))?
            };
            tokenizer.encode(&rendered, true)
        } else if request.chat {
            let tokenizer = tokenizer.ok_or("reservation tokenizer is unavailable")?;
            let rendered =
                tokenizer.apply_chat_template(&[("user", request.prompt_text.as_str())], true);
            tokenizer.encode(&rendered, true)
        } else {
            let tokenizer = tokenizer.ok_or("reservation tokenizer is unavailable")?;
            tokenizer.encode(&request.prompt_text, true)
        };
        if prompt.is_empty() {
            return Err("empty prompt after tokenization".into());
        }
        if let Some(trace) = request.ttft.as_ref() {
            trace.mark_tokenize_end(prompt.len());
        }
        request.prepared_prompt = Some(prompt);
    }
    let prompt_tokens = request
        .prepared_prompt
        .as_ref()
        .expect("budget prompt was prepared")
        .len();
    if let Some(limit) = request.max_prompt_tokens
        && prompt_tokens > limit
    {
        return Err(format!(
            "prompt ({prompt_tokens} tok) exceeds configured model maximum ({limit})"
        ));
    }
    Ok(prompt_tokens)
}

fn budget_completion_bound(
    request: &Request,
    prompt_tokens: usize,
    caps: Option<&ModelCaps>,
) -> Result<usize, String> {
    let max_new = request.params.max_new;
    let requested_ctx = match (request.params.max_ctx, max_new) {
        (Some(cap), _) => cap,
        (None, worker::MAX_NEW_CTX_BOUNDED) => {
            let server_ctx = std::env::var("MEMRA_CTX")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8192usize);
            let mut cap = server_ctx;
            if prompt_tokens.saturating_add(16) > cap {
                cap = prompt_tokens.saturating_add(server_ctx);
            }
            cap
        }
        (None, max_new) => prompt_tokens
            .checked_add(max_new)
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| "request context bound overflowed".to_string())?,
    };
    let ctx_cap = caps
        .map(|caps| caps.context_length)
        .filter(|&context| context > 0)
        .map_or(requested_ctx, |context| requested_ctx.min(context));
    if prompt_tokens >= ctx_cap {
        return Err(format!(
            "prompt ({prompt_tokens} tok) >= context cap ({ctx_cap})"
        ));
    }
    Ok(max_new.min(ctx_cap - prompt_tokens))
}

/// What budget admission produced for the receipt row: the reservation permit and the
/// context it charged (D2 gap G4's "reserved ctx": `prompt_tokens + completion bound`,
/// the same quantities handed to `Metering::reserve`). `reserved_ctx` is `None` exactly
/// when no reservation ran.
struct BudgetAdmission {
    permit: Option<metering::Permit>,
    reserved_ctx: Option<u64>,
}

// Manual: `Permit` is `Box<dyn Any>`; the presence bit is the useful debug fact.
impl std::fmt::Debug for BudgetAdmission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetAdmission")
            .field("permit", &self.permit.is_some())
            .field("reserved_ctx", &self.reserved_ctx)
            .finish()
    }
}

fn admit_tenant_budget(
    st: &AppState,
    tenant: &auth::TenantCtx,
    request: &mut Request,
) -> Result<BudgetAdmission, BudgetRejection> {
    let Some(accounting) = st.metering.as_ref().filter(|m| m.enforces_limits()) else {
        return Ok(BudgetAdmission {
            permit: None,
            reserved_ctx: None,
        });
    };
    match accounting.is_limited(&tenant.tenant) {
        Ok(false) => return Err(BudgetRejection::Unenrolled),
        Ok(true) => {}
        Err(metering::AdmitError::Unavailable(err)) => {
            return Err(BudgetRejection::Unavailable(err));
        }
        Err(other) => {
            return Err(BudgetRejection::Unavailable(format!(
                "unexpected budget enrollment result: {other:?}"
            )));
        }
    }
    let tokenizer = st
        .budget_tokenizers
        .as_ref()
        .and_then(|tokenizers| tokenizers.get(&request.model))
        .map(Arc::as_ref);
    if request.prompt_ids.is_empty() && tokenizer.is_none() {
        return Err(BudgetRejection::Unavailable(format!(
            "no reservation tokenizer for model {:?}",
            request.model
        )));
    }
    let prompt_tokens =
        prepare_budget_prompt(request, tokenizer).map_err(BudgetRejection::Invalid)?;
    let completion_tokens =
        budget_completion_bound(request, prompt_tokens, st.caps.get(&request.model))
            .map_err(BudgetRejection::Invalid)?;
    let prompt_tokens = u64::try_from(prompt_tokens)
        .map_err(|_| BudgetRejection::Unavailable("prompt token count exceeds u64".into()))?;
    let completion_tokens = u64::try_from(completion_tokens)
        .map_err(|_| BudgetRejection::Unavailable("completion token bound exceeds u64".into()))?;
    match accounting.reserve(
        &tenant.tenant,
        tenant.key_prefix.as_deref(),
        &request.model,
        prompt_tokens,
        completion_tokens,
    ) {
        Ok(permit) => Ok(BudgetAdmission {
            permit,
            reserved_ctx: Some(prompt_tokens.saturating_add(completion_tokens)),
        }),
        Err(metering::AdmitError::Insufficient) => Err(BudgetRejection::Insufficient),
        Err(metering::AdmitError::PrincipalCapped) => Err(BudgetRejection::PrincipalCapped),
        // Provisioning-policy blocks intentionally reuse the prepaid 402 shape:
        // callers need one recovery action (add credit), while operators can read
        // the distinct admission mode from the authenticated admin surface.
        Err(metering::AdmitError::Blocked) => Err(BudgetRejection::Insufficient),
        Err(metering::AdmitError::Unenrolled) => Err(BudgetRejection::Unenrolled),
        Err(metering::AdmitError::Unavailable(err)) => Err(BudgetRejection::Unavailable(err)),
    }
}

fn request_ledger_error_response() -> Response {
    error_response_coded(
        StatusCode::INTERNAL_SERVER_ERROR,
        "request completion could not be committed to the billing ledger",
        "server_error",
        None,
        Some("request_ledger_unavailable"),
    )
}

fn request_ledger_error_body() -> serde_json::Value {
    error_body(
        "request completion could not be committed to the billing ledger",
        "server_error",
        None,
        Some("request_ledger_unavailable"),
    )
}

fn ledger_rejected(
    mut receipt: Option<Box<dyn metering::Receipt>>,
    response: Response,
    error_code: &str,
    request_id: &str,
) -> Response {
    let status = response.status().as_u16();
    if let Some(receipt) = receipt.as_mut()
        && let Err(err) = receipt.reject(status, error_code)
    {
        eprintln!("[ledger] ERROR: request {request_id} rejection receipt failed: {err}");
        return with_request_id(request_id, request_ledger_error_response());
    }
    with_request_id(request_id, response)
}

/// Settle a receipt with a NAMED zero-debit outcome (`deadline_exceeded`, `shed_deadline`,
/// `shed_queue`, `shed_queue_wait`) — `ledger_rejected`'s twin for terminal rows whose outcome the billing
/// census distinguishes from a plain rejection. Never bills (enforced again in
/// `ledger::PendingReceipt::finalize`).
fn ledger_unbilled(
    mut receipt: Option<Box<dyn metering::Receipt>>,
    response: Response,
    outcome: &'static str,
    error_code: &str,
    request_id: &str,
) -> Response {
    let status = response.status().as_u16();
    if let Some(receipt) = receipt.as_mut()
        && let Err(err) = receipt.settle_unbilled(outcome, status, error_code)
    {
        eprintln!("[ledger] ERROR: request {request_id} {outcome} receipt failed: {err}");
        return with_request_id(request_id, request_ledger_error_response());
    }
    with_request_id(request_id, response)
}

fn engine_error_code(class: worker::ErrClass) -> &'static str {
    use worker::ErrClass as C;
    match class {
        C::InvalidRequest => "invalid_request",
        C::ContextLength => "context_length_exceeded",
        C::ModelNotFound => "model_not_found",
        C::RateLimit => "rate_limit_exceeded",
        C::Overloaded => "overloaded",
        C::Engine => "engine_error",
    }
}

/// Canonicalize a requested model id to a LOADED alias, tolerating a stripped vendor prefix.
///
/// Marketplaces normalize model ids before calling upstream. Onlist lists
/// `qwen/qwen3.6-35b-a3b` but probes us for `qwen3.6-35b-a3b`, which produced
/// `unknown model "qwen3.6-35b-a3b"; loaded: ["qwen/qwen3.6-27b", "qwen/qwen3.6-35b-a3b"]`.
/// The engine was right and the mapping was wrong, but the listing side offers no upstream-id
/// override, so inbound tolerance belongs here.
///
/// An EXACT alias always wins, so nothing already working can change meaning. Otherwise, if
/// exactly ONE loaded alias's segment after the last `/` equals the request, that alias is used.
/// **Ambiguity is deliberately not resolved**: if two loaded aliases share a suffix
/// (`a/m` and `b/m`), the request stays unknown rather than silently routing to the wrong
/// weights and billing under the wrong model. `/v1/models` continues to advertise canonical ids
/// only — this is request tolerance, not a second public name.
/// The immediate 400 for a model id that resolves to nothing. This MUST fire before
/// prepaid budget admission: a budgeted tenant's reservation path needs the model's
/// tokenizer, so an unresolved id used to surface as a 503 "budget accounting is
/// unavailable" — a customer's typo dressed up as our outage. Same class/code the
/// worker's own roster rejection uses, so the error shape is identical either way.
fn model_not_found_response(models: &[String], requested: &str) -> Response {
    error_response_coded(
        StatusCode::BAD_REQUEST,
        &format!("unknown model {requested:?}; loaded: {models:?}"),
        "invalid_request_error",
        Some("model"),
        Some("model_not_found"),
    )
}

/// prompt_ids OOV gate (hermes, fixed 2026-08-19): `/v1/completions` accepts a raw
/// token-id prompt (`prompt_ids`, the exact-token validation-gate path) and NOTHING
/// bounded those ids against the model's vocabulary — an out-of-vocab id rode through
/// admission into the embed gather, an attacker-chosen row index past the embedding
/// table. Checked at INTAKE against worker-probed tokenizer truth (`ModelCaps::n_vocab`):
/// a clean 400 naming the first offending id, before the request costs a queue slot or
/// reaches the worker. `n_vocab == 0` (unknown) skips the gate — honest-unknown, the
/// same convention as every other caps field.
fn validate_prompt_ids(ids: &[u32], caps: Option<&ModelCaps>) -> Result<(), String> {
    let Some(n_vocab) = caps.map(|c| c.n_vocab).filter(|&n| n > 0) else {
        return Ok(());
    };
    if let Some((pos, &id)) = ids
        .iter()
        .enumerate()
        .find(|&(_, &id)| id as usize >= n_vocab)
    {
        return Err(format!(
            "prompt_ids[{pos}] = {id} is out of vocabulary (model vocab size {n_vocab})"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod prompt_ids_tests {
    use super::*;

    #[test]
    fn prompt_ids_are_bounded_by_the_model_vocab_at_intake() {
        let caps = ModelCaps {
            n_vocab: 8,
            ..Default::default()
        };
        // in bounds: every id < n_vocab, boundary included.
        assert!(validate_prompt_ids(&[0, 3, 7], Some(&caps)).is_ok());
        assert!(validate_prompt_ids(&[], Some(&caps)).is_ok());
        // out of bounds: first offender named by position and value.
        let err = validate_prompt_ids(&[1, 8, 2], Some(&caps)).unwrap_err();
        assert!(err.contains("prompt_ids[1] = 8"), "{err}");
        assert!(err.contains("vocab size 8"), "{err}");
        let err = validate_prompt_ids(&[u32::MAX], Some(&caps)).unwrap_err();
        assert!(err.contains("4294967295"), "{err}");
        // unknown vocab (0) or unknown model: honest-unknown, no gate.
        let unknown = ModelCaps::default();
        assert!(validate_prompt_ids(&[u32::MAX], Some(&unknown)).is_ok());
        assert!(validate_prompt_ids(&[u32::MAX], None).is_ok());
    }
}

fn canonical_model_id(models: &[String], requested: &str) -> Option<String> {
    if models.iter().any(|m| m == requested) {
        return Some(requested.to_string());
    }
    if requested.is_empty() || requested.contains('/') {
        return None;
    }
    let mut matches = models.iter().filter(|m| {
        m.rsplit('/')
            .next()
            .is_some_and(|suffix| suffix == requested)
    });
    match (matches.next(), matches.next()) {
        (Some(only), None) => Some(only.clone()),
        _ => None,
    }
}

async fn completions_admitted(
    state: State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    AdmittedJson(req, admission): AdmittedJson<CompletionReq>,
) -> Response {
    completions_with_admission(state, headers, trace, Json(req), Some(admission)).await
}

#[cfg(test)]
async fn completions(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    request: Json<CompletionReq>,
) -> Response {
    completions_with_admission(State(st), headers, trace, request, None).await
}

async fn completions_with_admission(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    Json(mut req): Json<CompletionReq>,
    mut body_admission: Option<BodyAdmissionLease>,
) -> Response {
    let env = Envelope::new(false);
    if let Err(msg) = req.stop.validate() {
        return with_request_id(&env.id, bad_request(&msg, Some("stop")));
    }
    if let Err(msg) = validate_client_identifier(req.trace_id.as_deref(), "trace_id") {
        return with_request_id(&env.id, bad_request(&msg, Some("trace_id")));
    }
    match canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return with_request_id(&env.id, model_not_found_response(&st.models, &req.model));
        }
    }
    // API key: OpenAI-style `Authorization: Bearer <key>` -> tenant identity
    // (MEMRA_API_KEYS keyring and/or the MEMRA_API_KEY single key; nothing set = open).
    let ttft = trace.and_then(|Extension(trace)| trace.0);
    if let Some(trace) = ttft.as_ref() {
        trace.mark_parsed();
        trace.bind_request(&env.id, &req.model);
    }
    let tenant = match authenticate(&st.api_auth, &headers) {
        Ok(t) => t,
        Err(resp) => return with_request_id(&env.id, resp),
    };
    let cache_ns = match tenant_namespace(&tenant, &req.cache_salt) {
        Ok(ns) => ns,
        Err(msg) => return with_request_id(&env.id, bad_request(msg, Some("cache_salt"))),
    };
    // HONESTY GATE (gap-scan F4): semantic params we can't honor 400 loudly.
    if let Err((msg, param)) = reject_unsupported(&[
        (
            "logit_bias",
            req.logit_bias.is_some(),
            " (device-side sampling has no bias hook yet)",
        ),
        ("logprobs", req.logprobs.is_some(), ""),
        (
            "n",
            req.n.is_some_and(|n| n != 1),
            " for n != 1 (single choice only)",
        ),
        (
            "best_of",
            req.best_of.is_some_and(|n| n != 1),
            " (single choice only)",
        ),
    ]) {
        return with_request_id(&env.id, bad_request(&msg, Some(&param)));
    }
    // OOV gate (hermes): raw prompt_ids are bounded by the model's vocabulary HERE,
    // before the request costs a slot or reaches the worker's embed gather.
    if let Err(msg) = validate_prompt_ids(&req.prompt_ids, st.caps.get(&req.model)) {
        return with_request_id(&env.id, bad_request(&msg, Some("prompt_ids")));
    }
    // Request deadline (lane/deadline-billing): validated with the other request params
    // (a named 400 costs no slot and opens no receipt), armed from this point on.
    let deadline = match parse_timeout_ms(req.timeout_ms.as_ref()) {
        Ok(ms) => RequestDeadline::starting_now(ms),
        Err(msg) => return with_request_id(&env.id, bad_request(&msg, Some("timeout_ms"))),
    };
    let lane = match lane_for_tenant(&headers, &tenant) {
        Ok(l) => l,
        Err(resp) => return resp,
    };
    let (tx, rx) = worker::event_channel();
    let model = req.model.clone();
    let stream = req.stream;
    let affinity = match affinity_key(&req.session_id, &req.user, &headers) {
        Ok(affinity) => affinity,
        Err(msg) => return with_request_id(&env.id, bad_request(&msg, Some("session_id"))),
    };
    let mut request = build_request_with_trace(
        &req,
        tx,
        lane,
        affinity,
        ttft.clone(),
        // /v1/completions is a raw-prompt surface: no template render, no thinking
        // control, `ThinkMode::Default` always — so the arm law resolves it to the
        // primary (thinking) arm through the same `for_mode` body the chat builder uses.
        st.sampling_defaults(&model).for_mode(ThinkMode::Default),
    );
    request.cache_ns = cache_ns;
    request.request_id = env.id.clone();
    // The wire deadline rides to the worker beside the receipt identity, so the
    // first-token deadline gate judges the REMAINING deadline at its own tick.
    request.wire_deadline = Some(deadline.at.into_std());
    if let Err((message, param)) = apply_model_request_limits(
        &mut request,
        st.openrouter_metadata.get(&model),
        st.caps.get(&model),
    ) {
        return with_request_id(&env.id, bad_request(&message, Some(param)));
    }
    // FEASIBILITY GATE: a non-streaming request we can see will not finish inside its
    // deadline is refused HERE — before a slot, a receipt or any GPU work — with the
    // max_tokens that would fit. Costs nothing and replaces a 90 s wait for a 408 that
    // threw away every token it had generated.
    if let Err(msg) = nonstream_deadline_gate(
        &request,
        req.stream,
        deadline,
        req.max_tokens.is_some(),
        st.budget_tokenizers
            .as_ref()
            .and_then(|t| t.get(&req.model))
            .map(Arc::as_ref),
    ) {
        return with_request_id(
            &env.id,
            error_response_coded(
                StatusCode::BAD_REQUEST,
                &msg,
                "invalid_request_error",
                Some("max_tokens"),
                Some("nonstream_deadline_infeasible"),
            ),
        );
    }
    // DRAIN GATE (gap-scan F11): preserve the existing shutdown contract before
    // consulting tenant balances or touching any slot/queue state.
    if draining() {
        let receipt = start_request_receipt(
            &st,
            &env,
            &tenant,
            &req.model,
            "/v1/completions",
            lane,
            req.stream,
            effective_max_tokens(&request),
            None,
            None,
        );
        return ledger_rejected(receipt, drain_response(), "draining", &env.id);
    }
    let budget = match admit_tenant_budget(&st, &tenant, &mut request) {
        Ok(budget) => budget,
        Err(rejection) => {
            let (response, error_code) = rejection.into_response();
            let receipt = start_request_receipt(
                &st,
                &env,
                &tenant,
                &req.model,
                "/v1/completions",
                lane,
                req.stream,
                effective_max_tokens(&request),
                None,
                None,
            );
            return ledger_rejected(receipt, response, error_code, &env.id);
        }
    };
    let receipt = start_request_receipt(
        &st,
        &env,
        &tenant,
        &req.model,
        "/v1/completions",
        lane,
        req.stream,
        effective_max_tokens(&request),
        budget.reserved_ctx,
        budget.permit,
    );
    let receipt = arm_capture(receipt, || json!({ "prompt": req.prompt }));
    // RATE-LIMIT SNAPSHOT (gap-scan F12): take the in-flight slot at submission time;
    // the guard rides the response (stream included) and frees the slot at completion.
    let (guard, rl) = match acquire_request_slot(&st, lane, &tenant, &env) {
        Ok(slot) => slot,
        Err(resp) => {
            return ledger_rejected(receipt, resp, "rate_limit_exceeded", &env.id);
        }
    };
    if let Some(admission) = body_admission.as_mut() {
        admission.release();
    }
    // BACKPRESSURE (lane/deadline-billing): shed at submission — never after — when the
    // queue is at its bound or the estimated wait cannot fit the request's deadline.
    let pending_admit = match reserve_pending_admit(&st, lane, &rl, deadline) {
        Ok(guard) => guard,
        Err((resp, outcome)) => {
            return ledger_unbilled(receipt, rl.attach(resp), outcome, outcome, &env.id);
        }
    };
    meter_admit(&env, &tenant, &model, lane);
    let stop_strings = request.stop_strings.clone();

    // Admission yield (lane/admission-latency): raise the pending-admit gauge BEFORE the
    // send — an in-flight spec burst polls it at every round boundary and ends early so
    // this request's admission wait stops scaling with MEMRA_SPEC_BURST. The worker
    // decrements at pop (handle_cmd).
    if let Some(trace) = ttft.as_ref() {
        trace.mark_submitted();
    }
    if st.cmd_tx.send(Cmd::Generate(Box::new(request))).is_err() {
        drop(pending_admit);
        return ledger_rejected(
            receipt,
            rl.attach(worker_unavailable_response()),
            "worker_unavailable",
            &env.id,
        );
    }
    pending_admit.commit();
    // DEADLINE: the admission wait counts against timeout_ms (a queued request that can
    // no longer answer in time is a miss). Dropping rx on a miss IS the cancel — the
    // worker prunes closed-channel requests still queued at the next tick.
    let rx = match tokio::time::timeout_at(deadline.at, peek_admission(rx)).await {
        Ok(Ok(rx)) => rx,
        Ok(Err((resp, error_code))) => {
            return ledger_rejected(receipt, rl.attach(resp), error_code, &env.id);
        }
        Err(_) => {
            return ledger_unbilled(
                receipt,
                rl.attach(deadline_exceeded_response(deadline.ms, stream)),
                "deadline_exceeded",
                "deadline_exceeded",
                &env.id,
            );
        }
    };

    let resp = if stream {
        // Streaming: timeout_ms bounds TIME-TO-FIRST-TOKEN only. Once the first token has
        // streamed the parameter is spent — a client that walks away mid-stream is the
        // existing "abandoned" path (user fault, partial billed, owner-ratified).
        let rx = match peek_first_token(rx, deadline).await {
            Ok(rx) => rx,
            Err(()) => {
                return ledger_unbilled(
                    receipt,
                    rl.attach(deadline_exceeded_response(deadline.ms, true)),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                );
            }
        };
        sse_response_with_receipt(
            rx,
            model,
            false,
            None,
            env.clone(),
            stop_strings,
            Some(guard),
            receipt,
        )
        .into_response()
    } else {
        // Non-streaming: the deadline is handled INSIDE the collector, which delivers what
        // was generated (billed) instead of discarding it. The old shape here was
        // `timeout_at(deadline.at, collect)`, whose miss dropped the future and threw away
        // up to 90 s of tokens to answer a 408 — the 2026-08-26 customer report. A
        // zero-token miss still answers 408 unbilled, from in there.
        let mut receipt = receipt;
        let resp = blocking_response_with_receipt(
            rx,
            model,
            false,
            stop_strings,
            None,
            env.clone(),
            &mut receipt,
            Some(deadline),
        )
        .await;
        drop(guard); // response complete or cut — free the slot before headers
        resp.into_response()
    };
    rl.attach(with_request_id(&env.id, resp))
}

async fn chat_completions_admitted(
    state: State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    AdmittedJson(req, admission): AdmittedJson<ChatCompletionReq>,
) -> Response {
    chat_completions_with_admission(state, headers, trace, Json(req), Some(admission)).await
}

#[cfg(test)]
async fn chat_completions(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    request: Json<ChatCompletionReq>,
) -> Response {
    chat_completions_with_admission(State(st), headers, trace, request, None).await
}

async fn chat_completions_with_admission(
    State(st): State<AppState>,
    headers: axum::http::HeaderMap,
    trace: Option<Extension<TtftRequestTrace>>,
    Json(mut req): Json<ChatCompletionReq>,
    mut body_admission: Option<BodyAdmissionLease>,
) -> Response {
    let env = Envelope::new(true);
    // Canonicalize before ANY downstream use: metadata limits, caps, cache namespace, ledger
    // pricing and the worker's roster all key off this id and must agree on one spelling.
    // An id that resolves to nothing refuses HERE — before budget admission (see
    // model_not_found_response for why the ordering is the whole point).
    match canonical_model_id(&st.models, &req.model) {
        Some(canonical) => req.model = canonical,
        None => {
            return with_request_id(&env.id, model_not_found_response(&st.models, &req.model));
        }
    }
    let ttft = trace.and_then(|Extension(trace)| trace.0);
    if let Some(trace) = ttft.as_ref() {
        trace.mark_parsed();
        trace.bind_request(&env.id, &req.model);
    }
    let tenant = match authenticate(&st.api_auth, &headers) {
        Ok(t) => t,
        Err(resp) => return with_request_id(&env.id, resp),
    };
    let cache_ns = match tenant_namespace(&tenant, &req.cache_salt) {
        Ok(ns) => ns,
        Err(msg) => return with_request_id(&env.id, bad_request(msg, Some("cache_salt"))),
    };
    if req.messages.is_empty()
        || req.messages.iter().any(|message| {
            !matches!(
                message.role.as_str(),
                "system" | "developer" | "user" | "assistant" | "tool"
            )
        })
    {
        return with_request_id(
            &env.id,
            bad_request(
                "messages must use system/developer/user/assistant/tool roles",
                Some("messages"),
            ),
        );
    }
    // HONESTY GATE (gap-scan F4): semantic params we can't honor 400 loudly, never
    // silent downgrades. response_format json_object/json_schema are now REAL
    // (constrained decoding, lane/constrained) — parsed below; bad forms 400 with the
    // parser's own message.
    if let Err((msg, param)) = reject_unsupported(&[
        (
            "logit_bias",
            req.logit_bias.is_some(),
            " (device-side sampling has no bias hook yet)",
        ),
        (
            "logprobs",
            req.logprobs
                .as_ref()
                .is_some_and(|v| v.as_bool() != Some(false)),
            "",
        ),
        ("top_logprobs", req.top_logprobs.is_some(), ""),
        (
            "n",
            req.n.is_some_and(|n| n != 1),
            " for n != 1 (single choice only)",
        ),
    ]) {
        return with_request_id(&env.id, bad_request(&msg, Some(&param)));
    }
    // Request deadline (lane/deadline-billing): validated with the other request params
    // (a named 400 costs no slot and opens no receipt), armed from this point on.
    let deadline = match parse_timeout_ms(req.timeout_ms.as_ref()) {
        Ok(ms) => RequestDeadline::starting_now(ms),
        Err(msg) => return with_request_id(&env.id, bad_request(&msg, Some("timeout_ms"))),
    };
    let lane = match lane_for_tenant(&headers, &tenant) {
        Ok(l) => l,
        Err(resp) => return resp,
    };
    let model = req.model.clone();
    let stream = req.stream;
    // Snapshot the capture payload BEFORE the plan build consumes the request. Only
    // marked tenants pay for the copy; everyone else gets a lock-read and a None.
    let capture_prompt = st
        .metering
        .as_ref()
        .filter(|m| m.captures(&tenant.tenant))
        .map(|_| capture_chat_messages(&req.messages));
    // Read BEFORE the plan build consumes `req`: the feasibility gate judges only a
    // caller-DECLARED max_tokens (an omitted one is resolved to the model max downstream,
    // which is not a number the caller chose).
    let declared_max_tokens = req.max_tokens.is_some();
    // Preprocessing has its own bounded permit. GIFs must be decoded while the plan is built so
    // their sampled timestamps can render the prompt, while still images decode later; serializing
    // this phase keeps their transient canvases from multiplying outside request admission.
    let vision_preprocess_permit = match try_vision_preprocess(request_has_vision(&req)) {
        Ok(permit) => permit,
        Err(response) => return with_request_id(&env.id, response),
    };
    let (tx, rx) = worker::event_channel();
    let affinity = match affinity_key(&req.session_id, &req.user, &headers) {
        Ok(affinity) => affinity,
        Err(msg) => return with_request_id(&env.id, bad_request(&msg, Some("session_id"))),
    };
    let mut plan = match build_chat_request_with_trace(
        req,
        st.caps.get(&model),
        tx,
        lane,
        affinity,
        ttft.clone(),
        st.openrouter_metadata
            .get(&model)
            .and_then(|m| m.default_reasoning_effort.as_deref()),
        &st.sampling_defaults(&model),
    ) {
        Ok(plan) => plan,
        Err(err) => {
            return with_request_id(&env.id, bad_request(&err, None));
        }
    };
    plan.request.cache_ns = cache_ns;
    plan.request.request_id = env.id.clone();
    plan.request.wire_deadline = Some(deadline.at.into_std());
    if let Err((message, param)) = apply_model_request_limits(
        &mut plan.request,
        st.openrouter_metadata.get(&model),
        st.caps.get(&model),
    ) {
        return with_request_id(&env.id, bad_request(&message, Some(param)));
    }
    // FEASIBILITY GATE — same body as the /v1/completions surface (standard-surface law:
    // one implementation, every entry path). See nonstream_deadline_gate.
    if let Err(msg) = nonstream_deadline_gate(
        &plan.request,
        stream,
        deadline,
        declared_max_tokens,
        st.budget_tokenizers
            .as_ref()
            .and_then(|t| t.get(&model))
            .map(Arc::as_ref),
    ) {
        return with_request_id(
            &env.id,
            error_response_coded(
                StatusCode::BAD_REQUEST,
                &msg,
                "invalid_request_error",
                Some("max_tokens"),
                Some("nonstream_deadline_infeasible"),
            ),
        );
    }
    plan.vision_memory = match reserve_vision_memory(&plan) {
        Ok(permit) => permit,
        Err(err) => {
            return with_request_id(&env.id, vision_memory_error_response(err, Some("messages")));
        }
    };
    // DRAIN GATE (gap-scan F11): preserve the existing shutdown contract before
    // consulting tenant balances or touching any slot/queue state.
    if draining() {
        let receipt = start_request_receipt(
            &st,
            &env,
            &tenant,
            &model,
            "/v1/chat/completions",
            lane,
            stream,
            effective_max_tokens(&plan.request),
            None,
            None,
        );
        return ledger_rejected(receipt, drain_response(), "draining", &env.id);
    }
    let budget = match admit_tenant_budget(&st, &tenant, &mut plan.request) {
        Ok(budget) => budget,
        Err(rejection) => {
            let (response, error_code) = rejection.into_response();
            let receipt = start_request_receipt(
                &st,
                &env,
                &tenant,
                &model,
                "/v1/chat/completions",
                lane,
                stream,
                effective_max_tokens(&plan.request),
                None,
                None,
            );
            return ledger_rejected(receipt, response, error_code, &env.id);
        }
    };
    let receipt = start_request_receipt(
        &st,
        &env,
        &tenant,
        &model,
        "/v1/chat/completions",
        lane,
        stream,
        effective_max_tokens(&plan.request),
        budget.reserved_ctx,
        budget.permit,
    );
    let receipt = if let Some(prompt) = capture_prompt {
        arm_capture(receipt, move || prompt)
    } else {
        receipt
    };
    // RATE-LIMIT SNAPSHOT (gap-scan F12): slot taken at submission (post-validation —
    // a 400 never held a slot); freed when the response completes (guard). It is deliberately
    // acquired BEFORE vision decode so a rejected/rate-limited request cannot expand canvases.
    let (guard, rl) = match acquire_request_slot(&st, lane, &tenant, &env) {
        Ok(slot) => slot,
        Err(resp) => {
            return ledger_rejected(receipt, resp, "rate_limit_exceeded", &env.id);
        }
    };
    if let Some(admission) = body_admission.as_mut() {
        admission.release();
    }
    // BACKPRESSURE (lane/deadline-billing): shed at submission — never after — when the
    // queue is at its bound or the estimated wait cannot fit the request's deadline.
    let pending_admit = match reserve_pending_admit(&st, lane, &rl, deadline) {
        Ok(guard) => guard,
        Err((resp, outcome)) => {
            return ledger_unbilled(receipt, rl.attach(resp), outcome, outcome, &env.id);
        }
    };
    // Vision phase 2 (hermes decode-bomb finding, fixed 2026-08-23): the canvases expand
    // only HERE — after budget admission and request-slot admission priced the header-planned
    // pad runs. The process-wide memory permit moves into the worker request below and survives
    // streaming responses until completion/cancellation.
    if let Err(err) = decode_pending_vision(&mut plan) {
        return ledger_rejected(
            receipt,
            rl.attach(bad_request(&err, Some("messages"))),
            "invalid_request_error",
            &env.id,
        );
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
    meter_admit(&env, &tenant, &model, lane);
    let stop_strings = plan.request.stop_strings.clone();
    // Admission yield (lane/admission-latency): gauge up before send — see completions.
    if let Some(trace) = ttft.as_ref() {
        trace.mark_submitted();
    }
    if st
        .cmd_tx
        .send(Cmd::Generate(Box::new(plan.request)))
        .is_err()
    {
        drop(pending_admit);
        return ledger_rejected(
            receipt,
            rl.attach(worker_unavailable_response()),
            "worker_unavailable",
            &env.id,
        );
    }
    pending_admit.commit();
    // A constrained stream must not commit HTTP 200 before its schema has compiled. This wait
    // is asynchronous; the compiler runs on its bounded model thread and the GPU worker keeps
    // stepping. Timeout/invalid schema therefore remains a clean pre-header 503/400. The wait
    // is additionally bounded by the request's own deadline (a sub-5s timeout_ms must not be
    // overshot by the compile window).
    if let Some(ready) = constraint_ready {
        let bound = constrained::CONSTRAINT_COMPILE_TIMEOUT.min(deadline.remaining());
        match tokio::time::timeout(bound, ready).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(err))) => {
                return ledger_rejected(
                    receipt,
                    rl.attach(engine_error_response(&err)),
                    engine_error_code(err.class),
                    &env.id,
                );
            }
            Ok(Err(_)) => {
                return ledger_rejected(
                    receipt,
                    rl.attach(worker_unavailable_response()),
                    "worker_unavailable",
                    &env.id,
                );
            }
            Err(_) if deadline.remaining().is_zero() => {
                return ledger_unbilled(
                    receipt,
                    rl.attach(deadline_exceeded_response(deadline.ms, stream)),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                );
            }
            Err(_) => {
                return ledger_rejected(
                    receipt,
                    rl.attach(engine_error_response(&worker::constraint_timeout_error())),
                    "constraint_compile_timeout",
                    &env.id,
                );
            }
        }
    }
    // DEADLINE: the admission wait counts against timeout_ms — see `completions`.
    let rx = match tokio::time::timeout_at(deadline.at, peek_admission(rx)).await {
        Ok(Ok(rx)) => rx,
        Ok(Err((resp, error_code))) => {
            return ledger_rejected(receipt, rl.attach(resp), error_code, &env.id);
        }
        Err(_) => {
            return ledger_unbilled(
                receipt,
                rl.attach(deadline_exceeded_response(deadline.ms, stream)),
                "deadline_exceeded",
                "deadline_exceeded",
                &env.id,
            );
        }
    };
    let resp = if stream {
        // Streaming: timeout_ms bounds TIME-TO-FIRST-TOKEN only — see `completions`.
        let rx = match peek_first_token(rx, deadline).await {
            Ok(rx) => rx,
            Err(()) => {
                return ledger_unbilled(
                    receipt,
                    rl.attach(deadline_exceeded_response(deadline.ms, true)),
                    "deadline_exceeded",
                    "deadline_exceeded",
                    &env.id,
                );
            }
        };
        sse_response_with_receipt(
            rx,
            model,
            true,
            plan.parser,
            env.clone(),
            stop_strings,
            Some(guard),
            receipt,
        )
        .into_response()
    } else {
        // Non-streaming: the deadline is handled INSIDE the collector, which delivers what
        // was generated instead of discarding it — see `completions`.
        let mut receipt = receipt;
        let resp = blocking_response_with_receipt(
            rx,
            model,
            true,
            stop_strings,
            plan.parser,
            env.clone(),
            &mut receipt,
            Some(deadline),
        )
        .await;
        drop(guard); // response complete or cut — free the slot before headers
        resp.into_response()
    };
    rl.attach(with_request_id(&env.id, resp))
}

/// Streaming (SSE): forward each Token as an SSE `data:` line; emit a final `done` event.
/// `parser`: Some only for tools-armed chat requests — content routes through the tool-call
/// parser and parsed calls stream as OpenAI `tool_calls` deltas (one header chunk carrying
/// id/type/name, one arguments chunk), with `finish_reason:"tool_calls"` on the final chunk.
/// ENVELOPE (gap-scan F1): every OpenAI-shape chunk is stamped with the request's
/// id/created/system_fingerprint; the FIRST chat delta carries `role:"assistant"` (SDK
/// stream-accumulator contract); mid-stream worker errors go out as a `data:` error chunk
/// (OpenAI clients never parse named SSE events) followed by [DONE].
#[cfg(test)]
fn sse_response(
    rx: worker::EventReceiver,
    model: String,
    chat: bool,
    parser: Option<ToolStreamParser>,
    env: Envelope,
    stop_strings: Vec<String>,
    guard: Option<InflightGuard>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    sse_response_with_receipt(rx, model, chat, parser, env, stop_strings, guard, None)
}

#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
fn sse_response_with_receipt(
    mut rx: worker::EventReceiver,
    model: String,
    chat: bool,
    mut parser: Option<ToolStreamParser>,
    env: Envelope,
    stop_strings: Vec<String>,
    guard: Option<InflightGuard>,
    mut receipt: Option<Box<dyn metering::Receipt>>,
) -> Sse<impl futures_core::Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    // STOP-LEAK holdback (gap-scan F9), OpenAI shapes only: content deltas buffer until
    // they can't start a stop string; matched stop text is excluded exactly like the
    // non-stream shape. The memra-native stream stays byte-identical (no scrubber).
    let mut scrub = (!stop_strings.is_empty() && (chat || openai_compat()))
        .then(|| StopScrubber::new(stop_strings));
    let stream = async_stream::stream! {
        // in-flight slot rides the stream: freed when the stream completes or the
        // client disconnects (drop) — the rate-limit gauge + drain barrier source.
        let _guard = guard;
        let mut call_index: usize = 0;
        // first chat delta carries the role (applied to whatever delta comes first —
        // content, reasoning, or the tool-call header).
        let mut role_sent = false;
        macro_rules! chat_chunk {
            ($delta:expr, $finish:expr) => {{
                let mut delta = $delta;
                if chat && !role_sent {
                    role_sent = true;
                    delta["role"] = json!("assistant");
                }
                env.stamp(json!({ "object": "chat.completion.chunk", "model": model,
                                  "choices": [{ "index": 0, "delta": delta,
                                                "finish_reason": $finish }] }))
                    .to_string()
            }};
        }
        // renders Piece -> chat.completion.chunk payloads (tools-armed path only).
        macro_rules! piece_chunks {
            ($piece:expr) => {{
                let mut payloads: Vec<String> = Vec::new();
                match $piece {
                    Piece::Content(text) => {
                        let text = match scrub.as_mut() {
                            Some(sc) => sc.push(&text),
                            None => text,
                        };
                        if !text.is_empty() {
                            payloads.push(chat_chunk!(json!({ "content": text }),
                                                      serde_json::Value::Null));
                        }
                    }
                    // OR reasoning dialect (gap-scan F13): think text streams as
                    // delta.reasoning, never as content (stop strings scrub content only,
                    // same as the non-stream truncate law).
                    Piece::Reasoning(text) => payloads.push(
                        chat_chunk!(json!({ "reasoning": text }), serde_json::Value::Null)),
                    Piece::Call(call) => {
                        payloads.push(chat_chunk!(json!({ "tool_calls": [{
                            "index": call_index, "id": call.id, "type": "function",
                            "function": { "name": call.name, "arguments": "" } }] }),
                            serde_json::Value::Null));
                        payloads.push(chat_chunk!(json!({ "tool_calls": [{
                            "index": call_index,
                            "function": { "arguments": call.arguments } }] }),
                            serde_json::Value::Null));
                        call_index += 1;
                    }
                }
                payloads
            }};
        }
        // Set by every arm that BREAKS with its receipt handled; false when the loop ends
        // because the worker closed the channel without Done/Error (worker restart) — the
        // post-loop arm below settles that as rejected, debit zero, never "abandoned".
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
                        let payload = request_ledger_error_body().to_string();
                        if chat || openai_compat() {
                            yield Ok(SseEvent::default().data(payload));
                            yield Ok(SseEvent::default().data("[DONE]"));
                        } else {
                            yield Ok(SseEvent::default().event("error").data(payload));
                        }
                        terminal = true;
                        break;
                    }
                }
                Event::Token { id, text } => {
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.record_completion_token()
                    {
                        eprintln!(
                            "[ledger] ERROR: request {} partial completion receipt failed: {err}",
                            env.id
                        );
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        let payload = request_ledger_error_body().to_string();
                        if chat || openai_compat() {
                            yield Ok(SseEvent::default().data(payload));
                            yield Ok(SseEvent::default().data("[DONE]"));
                        } else {
                            yield Ok(SseEvent::default().event("error").data(payload));
                        }
                        terminal = true;
                        break;
                    }
                    // Capture accumulates the RAW generated text — before tool parsing
                    // and stop-scrub holdback — which is the model output a corpus wants.
                    if let Some(receipt) = receipt.as_mut() {
                        receipt.capture_completion_delta(&text);
                    }
                    if let Some(p) = parser.as_mut() {
                        for piece in p.push(&text) {
                            for payload in piece_chunks!(piece) {
                                yield Ok(SseEvent::default().data(payload));
                            }
                        }
                        continue;
                    }
                    let text = match scrub.as_mut() {
                        Some(sc) => sc.push(&text),
                        None => text,
                    };
                    if text.is_empty() && scrub.is_some() {
                        continue; // held back (possible stop prefix) or post-stop
                    }
                    let payload = if chat {
                        chat_chunk!(json!({ "content": text }), serde_json::Value::Null)
                    } else if openai_compat() {
                        env.stamp(json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": text, "finish_reason": null }] }))
                            .to_string()
                    } else {
                        json!({ "model": model, "id": id, "text": text }).to_string()
                    };
                    yield Ok(SseEvent::default().data(payload));
                }
                // Blocking native responses use this terminal snapshot to recover every id
                // from coalesced speculative rounds. SSE already emitted the corresponding
                // text and intentionally has no terminal token-array surface.
                Event::TokenSnapshot(_) => {}
                Event::Done { stop_reason, n_tokens, n_prompt, n_cached, elapsed_s, spec } => {
                    let mut finish = stop_reason_to_finish(&stop_reason);
                    if let Some(p) = parser.as_mut() {
                        for piece in p.finish() {
                            for payload in piece_chunks!(piece) {
                                yield Ok(SseEvent::default().data(payload));
                            }
                        }
                        if p.n_calls() > 0 { finish = "tool_calls"; }
                    }
                    // stop-scrubber flush: held-back text that never became a stop.
                    if let Some(sc) = scrub.as_mut() {
                        let tail = sc.finish();
                        if !tail.is_empty() {
                            let payload = if chat {
                                chat_chunk!(json!({ "content": tail }),
                                            serde_json::Value::Null)
                            } else {
                                env.stamp(json!({ "object": "text_completion",
                                    "model": model,
                                    "choices": [{ "index": 0, "text": tail,
                                                  "finish_reason": null }] })).to_string()
                            };
                            yield Ok(SseEvent::default().data(payload));
                        }
                    }
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
                        // A pricing failure inside complete() leaves the receipt
                        // unfinalized; settle it rejected (best effort — a no-op when
                        // the append itself already latched) so Drop cannot bill it.
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        let payload = request_ledger_error_body().to_string();
                        if chat || openai_compat() {
                            yield Ok(SseEvent::default().data(payload));
                            yield Ok(SseEvent::default().data("[DONE]"));
                        } else {
                            yield Ok(SseEvent::default().event("error").data(payload));
                        }
                        terminal = true;
                        break;
                    }
                    if chat || openai_compat() {
                        let usage = usage_json(n_prompt, n_tokens, n_cached, elapsed_s, spec);
                        let fin = if chat {
                            let mut v = env.stamp(json!({
                                "object": "chat.completion.chunk", "model": model,
                                "choices": [{ "index": 0, "delta": {},
                                              "finish_reason": finish }],
                                "usage": usage }));
                            // zero-token stream: the role must still arrive (SDK contract).
                            if !role_sent {
                                v["choices"][0]["delta"]["role"] = json!("assistant");
                            }
                            v
                        } else {
                            env.stamp(json!({ "object": "text_completion", "model": model,
                                "choices": [{ "index": 0, "text": "",
                                              "finish_reason": finish }],
                                "usage": usage }))
                        }.to_string();
                        yield Ok(SseEvent::default().data(fin));
                        yield Ok(SseEvent::default().data("[DONE]"));
                    } else {
                        let payload = json!({
                            "stop_reason": stop_reason, "n_tokens": n_tokens,
                            "prompt_tokens": n_prompt, "cached_tokens": n_cached,
                            "elapsed_s": elapsed_s
                        }).to_string();
                        yield Ok(SseEvent::default().event("done").data(payload));
                    }
                    terminal = true;
                    break;
                }
                Event::Error(err) => {
                    // MID-STREAM FAILURE (G6). The response status is already 200 and the
                    // headers are gone, so there is no status code left to change: the ONLY
                    // honest signal is an error object in the stream followed by closing the
                    // connection. Both happen here — the `break` ends the generator, which
                    // drops the SSE body and closes.
                    //
                    // The class-derived type/code now travels with it (previously hardcoded
                    // "server_error" for every cause, so a client could not tell an
                    // out-of-VRAM from a context-length mistake once streaming had begun).
                    let ledger_error = if let Some(receipt) = receipt.as_mut() {
                        receipt
                            .reject(class_http(err.class).0.as_u16(), engine_error_code(err.class))
                            .err()
                    } else {
                        None
                    };
                    if let Some(ref ledger_error) = ledger_error {
                        eprintln!(
                            "[ledger] ERROR: request {} failure receipt failed: {ledger_error}",
                            env.id
                        );
                    }
                    let payload = if ledger_error.is_some() {
                        request_ledger_error_body().to_string()
                    } else {
                        engine_error_body(&err).to_string()
                    };
                    if chat || openai_compat() {
                        // OpenAI clients only parse `data:` lines — a named `event: error`
                        // reads as a silent hang. Error object as the final data chunk.
                        yield Ok(SseEvent::default().data(payload));
                        yield Ok(SseEvent::default().data("[DONE]"));
                    } else {
                        // Native (non-OpenAI) surface keeps its named `error` event: its
                        // clients are memra's own tools, which do parse named events.
                        yield Ok(SseEvent::default().event("error").data(payload));
                    }
                    terminal = true;
                    break;
                }
            }
        }
        if !terminal {
            // Channel closed without Done/Error: the worker thread is gone (panicked or
            // restarting) — OUR fault, so the receipt settles rejected with debit ZERO
            // (fault-attribution ruling 2026-08-23; this used to fall through to Drop and
            // bill the partial stream as a client "abandon"), and the failure is LOUD:
            // the same error object the blocking path returns, as the final chunk.
            let e = worker::EngineError::overloaded(
                "worker closed the stream without completing (worker restart in progress)",
            );
            if let Some(receipt) = receipt.as_mut()
                && let Err(ledger_err) = receipt.reject(
                    class_http(e.class).0.as_u16(),
                    engine_error_code(e.class),
                )
            {
                eprintln!(
                    "[ledger] ERROR: request {} closed-stream receipt failed: {ledger_err}",
                    env.id
                );
            }
            let payload = engine_error_body(&e).to_string();
            if chat || openai_compat() {
                yield Ok(SseEvent::default().data(payload));
                yield Ok(SseEvent::default().data("[DONE]"));
            } else {
                yield Ok(SseEvent::default().event("error").data(payload));
            }
        }
    };
    Sse::new(stream).keep_alive(
        // OR cancels + fails over on silent phases (fetch timeout) — long-prompt prefill
        // streams nothing for many seconds before first token. SSE comment every 5s.
        axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(5)),
    )
}

/// Blocking JSON: collect all tokens, return one {text, tokens, stop_reason} when done.
fn truncate_at_stop(text: &mut String, stop_strings: &[String]) {
    if let Some(offset) = stop_strings.iter().filter_map(|stop| text.find(stop)).min() {
        text.truncate(offset);
    }
}

/// Longest PROPER prefix of `tag` (on tag char boundaries) that `s` ends with — the
/// char-boundary-safe twin of toolcall's ASCII-tag helper (stop strings are client text).
fn partial_stop_suffix(s: &str, tag: &str) -> usize {
    let mut best = 0;
    for (k, _) in tag.char_indices().skip(1) {
        if k <= s.len() && s.ends_with(&tag[..k]) {
            best = k;
        }
    }
    best
}

/// STREAMING STOP SCRUBBER (gap-scan F9): the worker emits the token delta BEFORE its
/// stop check, so streams used to leak the stop text (and same-token overshoot) that
/// non-stream clients never see. Content deltas route through this holdback buffer:
/// text is released only once it can no longer be the start of a stop string, and a
/// completed stop truncates exactly like the non-stream `truncate_at_stop`.
struct StopScrubber {
    stops: Vec<String>,
    buf: String,
    done: bool,
}

impl StopScrubber {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            buf: String::new(),
            done: false,
        }
    }

    /// Feed a content delta; returns the text now safe to emit.
    fn push(&mut self, text: &str) -> String {
        if self.done {
            return String::new();
        }
        self.buf.push_str(text);
        if let Some(i) = self
            .stops
            .iter()
            .filter_map(|s| self.buf.find(s.as_str()))
            .min()
        {
            self.done = true;
            let out = self.buf[..i].to_string();
            self.buf.clear();
            return out;
        }
        let keep = self
            .stops
            .iter()
            .map(|s| partial_stop_suffix(&self.buf, s))
            .max()
            .unwrap_or(0);
        let emit_to = self.buf.len() - keep;
        let out = self.buf[..emit_to].to_string();
        self.buf.drain(..emit_to);
        out
    }

    /// End of stream: release held-back text (it never became a stop).
    fn finish(&mut self) -> String {
        if self.done {
            self.buf.clear();
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
async fn blocking_response(
    rx: worker::EventReceiver,
    model: String,
    chat: bool,
    stop_strings: Vec<String>,
    parser: Option<ToolStreamParser>,
    env: Envelope,
) -> Response {
    blocking_response_with_receipt(rx, model, chat, stop_strings, parser, env, &mut None, None)
        .await
}

/// Everything the non-streaming JSON shapes need. ONE body builds the response for both
/// the normal completion and the deadline-partial path, so the two can never drift into
/// different shapes for the same surface (standard-surface law).
struct BlockingPayload<'a> {
    env: &'a Envelope,
    model: String,
    chat: bool,
    finish: &'static str,
    text: String,
    reasoning: String,
    calls: Vec<ParsedToolCall>,
    tokens: Vec<u32>,
    stop_reason: String,
    n_prompt: usize,
    n_tokens: usize,
    n_cached: usize,
    elapsed_s: f64,
    spec: Option<worker::SpecUsage>,
    /// Set ONLY when the request's deadline landed mid-generation and we are delivering
    /// what was produced. Carries the OpenRouter-dialect error object that rides a
    /// `finish_reason: "error"` partial, so a caller can tell "cut by time" from "hit
    /// max_tokens" — which `finish_reason: "length"` alone cannot say, and which no
    /// provider's finish-reason enum has a value for.
    deadline_error: Option<serde_json::Value>,
}

fn blocking_payload(p: BlockingPayload<'_>) -> Response {
    let BlockingPayload {
        env,
        model,
        chat,
        finish,
        text,
        reasoning,
        calls,
        tokens,
        stop_reason,
        n_prompt,
        n_tokens,
        n_cached,
        elapsed_s,
        spec,
        deadline_error,
    } = p;
    if chat {
        // OpenAI shape: content is null on a pure tool-call turn.
        let content = if !calls.is_empty() && text.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::Value::String(text)
        };
        let mut message = json!({ "role": "assistant", "content": content });
        // OR reasoning dialect (gap-scan F13): think text is a dedicated
        // message field (+ reasoning_details), content is post-think only.
        if !reasoning.is_empty() {
            message["reasoning"] = json!(reasoning);
            message["reasoning_details"] = json!([{
                "type": "reasoning.text", "text": reasoning }]);
        }
        if !calls.is_empty() {
            message["tool_calls"] =
                serde_json::Value::Array(calls.iter().map(tool_call_json).collect());
        }
        let mut body = json!({
            "object": "chat.completion", "model": model,
            "choices": [{ "index": 0,
                          "message": message,
                          "finish_reason": finish }],
            "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s, spec)
        });
        if let Some(err) = deadline_error {
            body["choices"][0]["native_finish_reason"] = json!("deadline_exceeded");
            body["error"] = err;
        }
        return Json(env.stamp(body)).into_response();
    }
    if openai_compat() {
        let mut body = json!({
            "object": "text_completion", "model": model,
            "choices": [{ "index": 0, "text": text,
                          "finish_reason": finish }],
            "usage": usage_json(n_prompt, n_tokens, n_cached, elapsed_s, spec)
        });
        if let Some(err) = deadline_error {
            body["choices"][0]["native_finish_reason"] = json!("deadline_exceeded");
            body["error"] = err;
        }
        return Json(env.stamp(body)).into_response();
    }
    Json(CompletionResp {
        model,
        text,
        tokens,
        stop_reason,
        error: deadline_error,
        n_tokens,
        prompt_tokens: n_prompt,
        cached_tokens: n_cached,
        elapsed_s,
    })
    .into_response()
}

/// Collect a complete non-streaming response.
///
/// `receipt` is BORROWED (lane/deadline-billing): it outlives this future so a deadline can
/// be settled with a named outcome rather than left to `Drop`, which would classify OUR cut
/// as an `abandoned` client. What changed in lane/deadline-partial-20260826 is WHERE the
/// deadline is handled and what it settles: no production handler wraps this future in
/// `timeout_at` any more (both pass `Some(deadline)` and the race is inside the loop below;
/// the `None` path is the `#[cfg(test)]` shim), and a MID-GENERATION miss settles the
/// BILLABLE `deadline_partial` because the caller received those tokens. Only a zero-token
/// miss settles `deadline_exceeded`, debit zero.
///
/// `deadline` is the request's own deadline and is handled HERE rather than by wrapping
/// this future in `timeout_at`. That wrapper was the 2026-08-26 customer bug: a miss
/// DROPPED this future, so every token already generated was discarded and the caller got
/// a 408 after the full 90 s (darklanes research/nonstream-deadline-20260826). Now the
/// deadline is a race inside the loop: whatever has been generated is DELIVERED, as an
/// OpenRouter-dialect partial (`finish_reason: "error"` + an `error` object naming
/// `error_type: "timeout"`), and billed for the tokens the caller actually received.
///
/// `finish_reason: "length"` would have been the cheaper lie: no provider's finish-reason
/// enum has a time value (OpenAI, Anthropic, Google and the hosted resellers all mean max_tokens by
/// "length"/MAX_TOKENS), so reporting a time cut as "length" tells the caller to ask for
/// more tokens when the truth is that it needs to stream. Only a zero-token miss still
/// answers 408 unbilled — there is nothing to deliver.
#[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
async fn blocking_response_with_receipt(
    mut rx: worker::EventReceiver,
    model: String,
    chat: bool,
    stop_strings: Vec<String>,
    mut parser: Option<ToolStreamParser>,
    env: Envelope,
    receipt: &mut Option<Box<dyn metering::Receipt>>,
    deadline: Option<RequestDeadline>,
) -> Response {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tokens: Vec<u32> = Vec::new();
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
    // Remembered for the deadline path, which has no Done event to read them from.
    let started = std::time::Instant::now();
    let mut seen_prompt: usize = 0;
    let mut seen_cached: usize = 0;
    let mut seen_tokens: usize = 0;
    loop {
        let ev = match deadline {
            Some(d) => tokio::select! {
                biased;
                ev = rx.recv() => ev,
                () = tokio::time::sleep_until(d.at) => {
                    // Stop the worker at its next tick by dropping the channel, then
                    // deliver what we have.
                    drop(rx);
                    if seen_tokens == 0 {
                        // NAMED outcome, not `rejected`: every sibling deadline path in
                        // this server writes `deadline_exceeded`, and a review caught this
                        // one-word census regression.
                        if let Some(receipt) = receipt.as_mut()
                            && let Err(err) = receipt.settle_unbilled(
                                "deadline_exceeded",
                                StatusCode::REQUEST_TIMEOUT.as_u16(),
                                "deadline_exceeded",
                            )
                        {
                            eprintln!(
                                "[ledger] ERROR: request {} deadline receipt failed: {err}",
                                env.id
                            );
                            return request_ledger_error_response();
                        }
                        return deadline_exceeded_response(d.ms, false);
                    }
                    if let Some(p) = parser.as_mut() {
                        consume(p.finish(), &mut text, &mut reasoning, &mut calls);
                    }
                    truncate_at_stop(&mut text, &stop_strings);
                    let elapsed_s = started.elapsed().as_secs_f64();
                    // BILLED: the caller received these tokens. The unbilled promise
                    // covers a request we failed to answer, not one we answered short.
                    if let Some(receipt) = receipt.as_mut()
                        && let Err(err) = receipt.complete_deadline_partial(
                            metering::UsageCounts {
                                prompt_tokens: seen_prompt as u64,
                                cached_prompt_tokens: seen_cached as u64,
                                completion_tokens: seen_tokens as u64,
                            },
                            elapsed_s,
                        )
                    {
                        eprintln!(
                            "[ledger] ERROR: request {} partial-deadline receipt failed: {err}",
                            env.id
                        );
                        let _ = receipt.reject(500, "request_ledger_unavailable");
                        return request_ledger_error_response();
                    }
                    eprintln!(
                        "[deadline] request {} delivered PARTIAL: {} tokens in {:.1}s of a \
                         {} ms deadline (prompt {}); non-streaming caller advised to stream",
                        env.id, seen_tokens, elapsed_s, d.ms, seen_prompt
                    );
                    let err_obj = json!({
                        "message": format!(
                            "deadline of {} ms (timeout_ms; default {}) elapsed mid-generation; \
                             the {} tokens produced before the cut are delivered above and are \
                             billed. Set \"stream\": true for work this long — a stream's \
                             deadline bounds only the time to first token — or lower max_tokens.",
                            d.ms, TIMEOUT_MS_DEFAULT, seen_tokens
                        ),
                        "code": "deadline_exceeded",
                        "metadata": { "error_type": "timeout", "provider_name": "memra" }
                    });
                    return blocking_payload(BlockingPayload {
                        env: &env,
                        model,
                        chat,
                        finish: "error",
                        text,
                        reasoning,
                        calls,
                        tokens,
                        stop_reason: "Deadline".to_string(),
                        n_prompt: seen_prompt,
                        n_tokens: seen_tokens,
                        n_cached: seen_cached,
                        elapsed_s,
                        spec: None,
                        deadline_error: Some(err_obj),
                    });
                }
            },
            None => rx.recv().await,
        };
        let Some(ev) = ev else { break };
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
                    // Settle the receipt as rejected (best effort) so its Drop cannot
                    // classify OUR bookkeeping failure as a billable client abandon.
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return request_ledger_error_response();
                }
                seen_prompt = n_prompt;
                seen_cached = n_cached;
            }
            Event::Token { id, text: delta } => {
                if let Some(receipt) = receipt.as_mut()
                    && let Err(err) = receipt.record_completion_token()
                {
                    eprintln!(
                        "[ledger] ERROR: request {} partial completion receipt failed: {err}",
                        env.id
                    );
                    let _ = receipt.reject(500, "request_ledger_unavailable");
                    return request_ledger_error_response();
                }
                // Raw generated text, pre-parse and pre-stop-truncation (see the SSE twin).
                if let Some(receipt) = receipt.as_mut() {
                    receipt.capture_completion_delta(&delta);
                }
                tokens.push(id);
                seen_tokens += 1;
                match parser.as_mut() {
                    Some(p) => consume(p.push(&delta), &mut text, &mut reasoning, &mut calls),
                    None => text.push_str(&delta),
                }
            }
            Event::TokenSnapshot(ids) => tokens = ids,
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
                truncate_at_stop(&mut text, &stop_strings);
                let finish = if calls.is_empty() {
                    stop_reason_to_finish(&stop_reason)
                } else {
                    "tool_calls"
                };
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
                    return request_ledger_error_response();
                }
                return blocking_payload(BlockingPayload {
                    env: &env,
                    model,
                    chat,
                    finish,
                    text,
                    reasoning,
                    calls,
                    tokens,
                    stop_reason,
                    n_prompt,
                    n_tokens,
                    n_cached,
                    elapsed_s,
                    spec,
                    deadline_error: None,
                });
            }
            Event::Error(err) => {
                // G6: the class decides the status. This single line used to be
                // `bad_request(&msg, None)` — every CUDA fault, VRAM exhaustion and admission
                // shed reported as 400 invalid_request_error, which no SDK retries.
                if let Some(receipt) = receipt.as_mut()
                    && let Err(ledger_err) = receipt.reject(
                        class_http(err.class).0.as_u16(),
                        engine_error_code(err.class),
                    )
                {
                    eprintln!(
                        "[ledger] ERROR: request {} failure receipt failed: {ledger_err}",
                        env.id
                    );
                    return request_ledger_error_response();
                }
                return engine_error_response(&err);
            }
        }
    }
    // The worker's Event channel closed without a Done or an Error: the worker thread is gone
    // (panicked and unrecoverable, or shutting down). 503 + Retry-After, not 500: this is a
    // process-level condition the supervisor is already acting on, and a client's retry may
    // well land on a restarted process.
    let e = worker::EngineError::overloaded(
        "worker closed the stream without completing (worker restart in progress)",
    );
    if let Some(receipt) = receipt.as_mut()
        && let Err(ledger_err) =
            receipt.reject(class_http(e.class).0.as_u16(), engine_error_code(e.class))
    {
        eprintln!(
            "[ledger] ERROR: request {} closed-stream receipt failed: {ledger_err}",
            env.id
        );
        return request_ledger_error_response();
    }
    engine_error_response(&e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Multi-item capture requests (`/v1/embeddings` N inputs, `/v1/rerank` N documents)
    /// give every capture its own ledger identity under the parent envelope: distinct per
    /// index, prefixed by the parent id, same `created`. The ledger keys debits by request
    /// id as a replay guard, so siblings sharing the parent id billed as one capture or
    /// failed the request (`conflicting budget debits`); see `Envelope::capture_child`.
    #[test]
    fn capture_children_are_distinct_ledger_identities_under_the_parent_id() {
        let parent = Envelope::new(false);
        assert!(parent.id.starts_with("cmpl-"));
        let a = parent.capture_child(0);
        let b = parent.capture_child(1);
        let c = parent.capture_child(2);
        assert_eq!(a.id, format!("{}.0", parent.id));
        assert_eq!(b.id, format!("{}.1", parent.id));
        assert_eq!(c.id, format!("{}.2", parent.id));
        assert_ne!(a.id, b.id);
        assert_ne!(b.id, c.id);
        for child in [&a, &b, &c] {
            assert!(
                child.id.starts_with(&parent.id),
                "child nests under the parent by prefix"
            );
            assert_ne!(
                child.id, parent.id,
                "a child never reuses the parent's ledger id"
            );
            assert_eq!(child.created, parent.created);
        }
        // The same index always derives the same child: a retry of one capture stays a
        // replay to the ledger instead of a fresh debit.
        assert_eq!(parent.capture_child(1).id, b.id);
    }

    /// What the handler is OBLIGED to tell any metering implementation, recorded as a
    /// flat event log. These tests used to run the in-tree prepaid ledger and assert
    /// its JSONL rows; that implementation is a deployment concern now (only the
    /// engine is open), so the public teeth assert the SEAM CALLS — which terminal
    /// method fired, with which worker-truth counts. Row/money assertions live with
    /// the implementation, and the cross-binary billing parity battery covers the
    /// composed behavior end to end.
    #[derive(Debug, Clone, PartialEq)]
    enum MeterEvent {
        Reserve {
            tenant: String,
            principal: Option<String>,
            model: String,
        },
        Open {
            request_id: String,
            tenant: String,
            model: String,
            route: &'static str,
            stream: bool,
            with_permit: bool,
        },
        PromptUsage {
            prompt: u64,
            cached: u64,
        },
        Token,
        CapturePrompt(serde_json::Value),
        CaptureDelta(String),
        Complete {
            prompt: u64,
            cached: u64,
            completion: u64,
        },
        DeadlinePartial {
            prompt: u64,
            cached: u64,
            completion: u64,
        },
        Reject {
            status: u16,
            code: String,
        },
        Unbilled {
            outcome: &'static str,
            status: u16,
            code: String,
        },
        /// The receipt died unfinalized — the abandoned-client path. The counts are
        /// whatever the handler had recorded by then.
        Dropped {
            prompt: u64,
            cached: u64,
            completion: u64,
        },
    }

    /// Scripted admission answers, consumed in order; an empty script admits with no
    /// permit (the "limits off / nothing reserved" shape).
    enum ReserveScript {
        Admit { with_permit: bool },
        Insufficient,
        Blocked,
        PrincipalCapped,
    }

    struct MockMetering {
        events: Arc<std::sync::Mutex<Vec<MeterEvent>>>,
        limits: bool,
        limited: bool,
        reserve_script: std::sync::Mutex<std::collections::VecDeque<ReserveScript>>,
        captures: bool,
    }

    impl MockMetering {
        fn admit_all() -> Arc<Self> {
            Arc::new(MockMetering {
                events: Arc::new(std::sync::Mutex::new(Vec::new())),
                limits: false,
                limited: true,
                reserve_script: std::sync::Mutex::new(std::collections::VecDeque::new()),
                captures: false,
            })
        }

        fn with_limits(script: Vec<ReserveScript>) -> Arc<Self> {
            Arc::new(MockMetering {
                events: Arc::new(std::sync::Mutex::new(Vec::new())),
                limits: true,
                limited: true,
                reserve_script: std::sync::Mutex::new(script.into()),
                captures: false,
            })
        }

        fn capturing() -> Arc<Self> {
            Arc::new(MockMetering {
                events: Arc::new(std::sync::Mutex::new(Vec::new())),
                limits: false,
                limited: true,
                reserve_script: std::sync::Mutex::new(std::collections::VecDeque::new()),
                captures: true,
            })
        }

        fn events(&self) -> Vec<MeterEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl metering::Metering for MockMetering {
        fn enforces_limits(&self) -> bool {
            self.limits
        }

        fn is_limited(&self, _tenant: &str) -> Result<bool, metering::AdmitError> {
            Ok(self.limited)
        }

        fn reserve(
            &self,
            tenant: &str,
            principal: Option<&str>,
            model: &str,
            _prompt_tokens: u64,
            _completion_bound: u64,
        ) -> Result<Option<metering::Permit>, metering::AdmitError> {
            self.events.lock().unwrap().push(MeterEvent::Reserve {
                tenant: tenant.into(),
                principal: principal.map(str::to_owned),
                model: model.into(),
            });
            match self.reserve_script.lock().unwrap().pop_front() {
                None | Some(ReserveScript::Admit { with_permit: false }) => Ok(None),
                Some(ReserveScript::Admit { with_permit: true }) => {
                    Ok(Some(Box::new(()) as metering::Permit))
                }
                Some(ReserveScript::Insufficient) => Err(metering::AdmitError::Insufficient),
                Some(ReserveScript::Blocked) => Err(metering::AdmitError::Blocked),
                Some(ReserveScript::PrincipalCapped) => Err(metering::AdmitError::PrincipalCapped),
            }
        }

        fn open(
            &self,
            meta: &metering::RequestMeta<'_>,
            permit: Option<metering::Permit>,
        ) -> Box<dyn metering::Receipt> {
            self.events.lock().unwrap().push(MeterEvent::Open {
                request_id: meta.request_id.into(),
                tenant: meta.tenant.into(),
                model: meta.model.into(),
                route: meta.route,
                stream: meta.stream,
                with_permit: permit.is_some(),
            });
            Box::new(MockReceipt {
                events: self.events.clone(),
                wants_capture: self.captures,
                prompt: 0,
                cached: 0,
                completion: 0,
                finalized: false,
            })
        }

        fn captures(&self, _tenant: &str) -> bool {
            self.captures
        }

        fn limits_health(&self) -> Option<metering::LimitsHealth> {
            self.limits.then_some(metering::LimitsHealth {
                source_reload_failed: 0,
                source_reload_consecutive: 0,
                source_available: true,
            })
        }
    }

    struct MockReceipt {
        events: Arc<std::sync::Mutex<Vec<MeterEvent>>>,
        wants_capture: bool,
        prompt: u64,
        cached: u64,
        completion: u64,
        finalized: bool,
    }

    impl metering::Receipt for MockReceipt {
        fn wants_capture(&self) -> bool {
            self.wants_capture
        }

        fn arm_capture(&mut self, prompt: serde_json::Value) {
            self.events
                .lock()
                .unwrap()
                .push(MeterEvent::CapturePrompt(prompt));
        }

        fn capture_completion_delta(&mut self, text: &str) {
            if self.wants_capture {
                self.events
                    .lock()
                    .unwrap()
                    .push(MeterEvent::CaptureDelta(text.into()));
            }
        }

        fn record_prompt_usage(&mut self, prompt: u64, cached: u64) -> Result<(), String> {
            self.prompt = prompt;
            self.cached = cached;
            self.events
                .lock()
                .unwrap()
                .push(MeterEvent::PromptUsage { prompt, cached });
            Ok(())
        }

        fn record_completion_token(&mut self) -> Result<(), String> {
            self.completion += 1;
            self.events.lock().unwrap().push(MeterEvent::Token);
            Ok(())
        }

        fn complete(
            &mut self,
            usage: metering::UsageCounts,
            _worker_elapsed_s: f64,
        ) -> Result<(), String> {
            self.finalized = true;
            self.events.lock().unwrap().push(MeterEvent::Complete {
                prompt: usage.prompt_tokens,
                cached: usage.cached_prompt_tokens,
                completion: usage.completion_tokens,
            });
            Ok(())
        }

        fn complete_deadline_partial(
            &mut self,
            usage: metering::UsageCounts,
            _worker_elapsed_s: f64,
        ) -> Result<(), String> {
            self.finalized = true;
            self.events
                .lock()
                .unwrap()
                .push(MeterEvent::DeadlinePartial {
                    prompt: usage.prompt_tokens,
                    cached: usage.cached_prompt_tokens,
                    completion: usage.completion_tokens,
                });
            Ok(())
        }

        fn reject(&mut self, status: u16, error_code: &str) -> Result<(), String> {
            self.finalized = true;
            self.events.lock().unwrap().push(MeterEvent::Reject {
                status,
                code: error_code.into(),
            });
            Ok(())
        }

        fn settle_unbilled(
            &mut self,
            outcome: &'static str,
            status: u16,
            error_code: &str,
        ) -> Result<(), String> {
            self.finalized = true;
            self.events.lock().unwrap().push(MeterEvent::Unbilled {
                outcome,
                status,
                code: error_code.into(),
            });
            Ok(())
        }
    }

    impl Drop for MockReceipt {
        fn drop(&mut self) {
            if !self.finalized {
                self.events.lock().unwrap().push(MeterEvent::Dropped {
                    prompt: self.prompt,
                    cached: self.cached,
                    completion: self.completion,
                });
            }
        }
    }

    /// Serializes every test that READS or FLIPS `MEMRA_NONSTREAM_DEADLINE_GATE`. The
    /// off-switch arm mutates process-global env, and the other gate tests call the gate and
    /// would observe that mutation if they ran in parallel — DRAIN_LOCK does not cover them
    /// because they have no reason to touch the drain flag. Flagged by review.
    static GATE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire GATE_ENV_LOCK surviving a poisoned peer, and restore the baseline it
    /// guards: `MEMRA_NONSTREAM_DEADLINE_GATE` unset (the documented default). The
    /// off-switch arm can panic between its `set_var` and its `remove_var`, and a plain
    /// `.unwrap()` would then hand every peer a PoisonError — the DRAIN_LOCK cascade of
    /// 2026-09-01 (one flake, 21 reds), same class. Recovery is sound because the env
    /// var is the only state under this lock and this resets it.
    fn gate_env_lock() -> std::sync::MutexGuard<'static, ()> {
        let guard = GATE_ENV_LOCK.lock().unwrap_or_else(|poisoned| {
            // Un-latch the flag too: poison otherwise persists forever, and only call
            // sites routed through this helper would survive it.
            GATE_ENV_LOCK.clear_poison();
            poisoned.into_inner()
        });
        unsafe { std::env::remove_var("MEMRA_NONSTREAM_DEADLINE_GATE") };
        guard
    }

    /// A Request shaped for the feasibility-gate tests: `max_new` declared, prompt given as
    /// raw ids so the estimate is exact rather than a byte proxy.
    fn gate_request(max_new: usize, prompt_ids: usize) -> worker::Request {
        let req: CompletionReq = serde_json::from_value(json!({
            "model": "qwen/qwen3.8-27b",
            "prompt_ids": vec![7u32; prompt_ids],
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let mut request = build_request(&req, tx, lanes::Lane::Interactive, None);
        request.params.max_new = max_new;
        request
    }

    /// The gate's boundary must sit where the MEASURED ladder sits. Numbers from
    /// darklanes research/nonstream-deadline-20260826, 30,278-token prompt through the
    /// customer path: 4096 out took 52.0 s, 5120 61.9 s, 6144 71.5 s (all 200), 8192
    /// 90.7 s and 16384 91.5 s (both 408). So the gate must ALLOW up to 6144 and REFUSE
    /// 8192 and 16384 — a gate that refuses 6144 would break a request that works, and one
    /// that allows 16384 would keep the bug.
    #[test]
    fn the_feasibility_gate_boundary_matches_the_measured_ladder() {
        let prompt = 30_278u64;
        let deadline_ms = TIMEOUT_MS_DEFAULT;
        let margin = |max_new: u64| {
            let prefill_ms = prompt * 1_000 / PREFILL_FLOOR_TOK_S;
            let decode_ms = max_new * 1_000 / DECODE_FLOOR_TOK_S;
            (prefill_ms + decode_ms) <= deadline_ms * DEADLINE_INFEASIBLE_MARGIN_PCT / 100
        };
        for allowed in [64u64, 2048, 4096, 5120, 6144] {
            assert!(margin(allowed), "{allowed} measured OK and must be allowed");
        }
        for refused in [8192u64, 16384, 262_144] {
            assert!(
                !margin(refused),
                "{refused} measured as a 408 and must be refused"
            );
        }
    }

    #[test]
    fn the_gate_names_a_max_tokens_that_actually_fits() {
        // At 30k prompt the floors leave ~75 s of decode inside a 90 s deadline, so the
        // advice must be a positive number well under the measured 7.8k ceiling.
        let fits = deadline_fitting_max_tokens(30_278, TIMEOUT_MS_DEFAULT).unwrap();
        assert!(
            fits > 0 && fits < 7_800,
            "advice {fits} must fit the measured ceiling"
        );
        // A prompt so large that prefill alone eats the deadline has NO feasible length.
        assert_eq!(
            deadline_fitting_max_tokens(400_000, TIMEOUT_MS_DEFAULT),
            None
        );
    }

    #[test]
    fn streaming_is_never_gated_and_the_gate_can_be_switched_off() {
        let req = gate_request(262_144, 30_000);
        let deadline = RequestDeadline::starting_now(TIMEOUT_MS_DEFAULT);
        // Non-streaming: refused, and the message has to be actionable, not just "no".
        let err = nonstream_deadline_gate(&req, false, deadline, true, None).unwrap_err();
        assert!(
            err.contains("stream"),
            "message must name the streaming alternative: {err}"
        );
        assert!(
            err.contains("max_tokens"),
            "message must name the knob: {err}"
        );
        // Streaming: the same request is fine — its deadline bounds only first-token time.
        assert!(nonstream_deadline_gate(&req, true, deadline, true, None).is_ok());
        // THE OFF SWITCH, ACTUALLY EXERCISED. This test's NAME claimed this behaviour while
        // asserting only the streaming half, and the seam was in fact DEAD: the flag was read
        // through a positive-only numeric reader, so `=0` fell back to the default and the
        // gate kept firing. The bench gate found it (arm 7 ran with the flag set to 0 and was
        // still refused); this arm is why it cannot come back.
        let _l = gate_env_lock(); // mutates process env
        for off in ["0", "off", "false"] {
            unsafe { std::env::set_var("MEMRA_NONSTREAM_DEADLINE_GATE", off) };
            assert!(
                nonstream_deadline_gate(&req, false, deadline, true, None).is_ok(),
                "MEMRA_NONSTREAM_DEADLINE_GATE={off} must disable the gate"
            );
        }
        unsafe { std::env::set_var("MEMRA_NONSTREAM_DEADLINE_GATE", "1") };
        assert!(nonstream_deadline_gate(&req, false, deadline, true, None).is_err());
        unsafe { std::env::remove_var("MEMRA_NONSTREAM_DEADLINE_GATE") };
        assert!(
            nonstream_deadline_gate(&req, false, deadline, true, None).is_err(),
            "unset means ON (the documented default)"
        );
    }

    /// TEETH FOR THE STANDARD-SURFACE CLAIM. The first version of this lane wired the
    /// feasibility gate into /v1/completions and /v1/chat/completions only, while its own
    /// comment claimed "one implementation, every entry path" — /v1/messages and
    /// /v1/responses kept the discard-and-408 shape. A review caught it. This asserts the
    /// call is present on the translated surfaces' SHARED admission body too, read from
    /// comment-stripped source so a mention in prose cannot satisfy it.
    #[test]
    fn the_feasibility_gate_is_wired_on_every_surface_not_just_the_two_i_remembered() {
        // Comment-stripped so a mention in prose cannot satisfy this, and scoped to each
        // HANDLER BODY so the gate's own definition, this test's needle literal, and the
        // test-module calls cannot satisfy it either. The first version asserted only
        // `source.contains(needle)`, which could never fail while the function existed in the
        // file at all — a review caught it, and it is the wiring-assertions-match-prose trap
        // this repo has been bitten by before.
        let strip = |src: &str| -> String {
            src.lines()
                .map(|line| match line.find("//") {
                    Some(i) => line[..i].to_string(),
                    None => line.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        /// The slice from a function's signature to the start of the next top-level item.
        fn body<'a>(src: &'a str, signature: &str) -> &'a str {
            let start = src
                .find(signature)
                .unwrap_or_else(|| panic!("{signature} not found — did the handler get renamed?"));
            let rest = &src[start + signature.len()..];
            let end = rest.find("\nasync fn ").unwrap_or(rest.len());
            let end = rest[..end].find("\npub(crate) async fn ").unwrap_or(end);
            &rest[..end]
        }
        let main_src = strip(include_str!("lib.rs"));
        let surfaces_src = strip(include_str!("surfaces.rs"));
        for (surface, src, signature) in [
            (
                "/v1/completions",
                &main_src,
                "async fn completions_with_admission(",
            ),
            (
                "/v1/chat/completions",
                &main_src,
                "async fn chat_completions_with_admission(",
            ),
            (
                "/v1/messages + /v1/responses (shared admission)",
                &surfaces_src,
                "pub(crate) async fn admit_translated(",
            ),
        ] {
            let handler = body(src, signature);
            assert!(
                handler.contains("nonstream_deadline_gate("),
                "{surface} must CALL the feasibility gate inside {signature}"
            );
            // And it must run AFTER the model limits resolve max_tokens, or it would judge a
            // cap that does not exist yet.
            let limits = handler
                .find("apply_model_request_limits(")
                .unwrap_or_else(|| panic!("{surface}: no apply_model_request_limits call"));
            let gate = handler.find("nonstream_deadline_gate(").unwrap();
            assert!(
                limits < gate,
                "{surface}: the gate must run after apply_model_request_limits"
            );
        }
    }

    /// The native (non-OpenAI) response shape must carry the deadline signal too. The first
    /// version of `blocking_payload` dropped the error object on that branch, so a cut
    /// response looked complete apart from an undocumented stop_reason — flagged by review.
    #[test]
    fn the_native_shape_carries_the_deadline_error_and_omits_it_otherwise() {
        let err = json!({"code": "deadline_exceeded",
                         "metadata": {"error_type": "timeout"}});
        let cut = CompletionResp {
            model: "m".into(),
            text: "partial".into(),
            tokens: vec![1, 2],
            stop_reason: "Deadline".into(),
            error: Some(err.clone()),
            n_tokens: 2,
            prompt_tokens: 9,
            cached_tokens: 0,
            elapsed_s: 1.0,
        };
        let v = serde_json::to_value(&cut).unwrap();
        assert_eq!(v["stop_reason"], "Deadline");
        assert_eq!(v["error"]["code"], "deadline_exceeded");
        assert_eq!(v["error"]["metadata"]["error_type"], "timeout");
        // A normal completion must be byte-unchanged: no `error` key at all.
        let whole = CompletionResp {
            error: None,
            stop_reason: "Eos".into(),
            ..cut
        };
        let v = serde_json::to_value(&whole).unwrap();
        assert!(
            v.get("error").is_none(),
            "a complete response must not grow an error key: {v}"
        );
    }

    #[test]
    fn a_ctx_bounded_request_is_not_gated_because_context_is_its_only_limit() {
        let _l = gate_env_lock();
        // Owner ruling 2026-08-26: "or limit is full context". A caller who sent no
        // max_tokens has declared no length for the gate to judge; partial delivery covers
        // it instead of a refusal the caller cannot act on.
        let req = gate_request(worker::MAX_NEW_CTX_BOUNDED, 30_000);
        assert!(
            nonstream_deadline_gate(
                &req,
                false,
                RequestDeadline::starting_now(TIMEOUT_MS_DEFAULT),
                false,
                None,
            )
            .is_ok(),
            "an omitted max_tokens is never gated — context is its only limit"
        );
        // THE BENCH-GATE DEFECT, pinned: a request whose omitted cap has already been
        // RESOLVED to the model maximum must still not be gated. Before this, the gate saw
        // a concrete 32768 it thought the caller had chosen and 400'd the most common
        // customer shape (arm 5, darklanes research/nonstream-deadline-20260826).
        let resolved = gate_request(32_768, 30_000);
        assert!(
            nonstream_deadline_gate(
                &resolved,
                false,
                RequestDeadline::starting_now(TIMEOUT_MS_DEFAULT),
                false,
                None,
            )
            .is_ok(),
            "a resolved-but-undeclared cap is not the caller's number to be refused over"
        );
        // And a caller who DID declare that cap on the same prompt IS refused.
        assert!(
            nonstream_deadline_gate(
                &resolved,
                false,
                RequestDeadline::starting_now(TIMEOUT_MS_DEFAULT),
                true,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn the_prompt_estimate_is_exact_for_ids_and_a_proxy_otherwise() {
        let req = gate_request(64, 1234);
        assert_eq!(prompt_tokens_estimate(&req, None), 1234, "ids are exact");
        let mut text = gate_request(64, 0);
        text.prompt_ids.clear();
        text.prompt_text = "x".repeat(6_000);
        assert_eq!(
            prompt_tokens_estimate(&text, None),
            1_000,
            "the fallback under-counts on purpose (bytes/6): an over-count refuses work \
             that would have succeeded"
        );
    }

    #[test]
    fn vision_memory_reservation_is_bounded_and_released() {
        let permit = try_reserve_vision_memory(MAX_VISION_PATCH_BYTES).unwrap();
        let Err(capacity) = try_reserve_vision_memory(1) else {
            panic!("a full process vision budget admitted another request");
        };
        assert!(matches!(capacity, VisionMemoryError::Capacity(_)));
        let response = vision_memory_error_response(capacity, Some("messages"));
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "5");
        assert_eq!(response.headers()["retry-after-ms"], "5000");
        drop(permit);
        assert!(try_reserve_vision_memory(1).is_ok());
        let Err(request) = try_reserve_vision_memory(MAX_VISION_PATCH_BYTES + 1) else {
            panic!("an over-limit vision request was admitted");
        };
        assert!(matches!(request, VisionMemoryError::Request(_)));
        let response = vision_memory_error_response(request, Some("messages"));
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()["x-should-retry"], "false");
        let _ = try_reserve_vision_memory(1);
    }

    #[test]
    fn header_auth_gate_covers_only_inference_dialects() {
        for path in [
            "/v1/auth/check",
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
            "/v1/embeddings",
            "/v1/rerank",
        ] {
            assert!(protected_inference_path(path), "{path}");
        }
        for path in ["/health", "/readyz", "/models", "/v1/models", "/metrics"] {
            assert!(!protected_inference_path(path), "{path}");
        }
    }
    /// The serve-shape capture seam: a request driven through the REAL blocking response
    /// path (the same consumer the HTTP handler awaits) feeds the armed prompt payload
    /// and EVERY completion delta into the receipt, byte-exact — and an unarmed receipt
    /// gets nothing. Where the payload is retained, and for whom, is the metering
    /// implementation's business (tested with it; the parity battery compares the
    /// composed capture files across binaries).
    #[tokio::test]
    async fn served_completion_capture_is_byte_exact_and_armed_receipts_only() {
        use crate::metering::Metering as _;
        let prompt = json!([{ "role": "user", "content": "capture me — exactly" }]);

        let drive = |receipt: Option<Box<dyn metering::Receipt>>| async {
            let (tx, rx) = worker::event_channel();
            tx.send(Event::PromptUsage {
                n_prompt: 7,
                n_cached: 0,
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
                stop_reason: "eos".into(),
                n_tokens: 2,
                n_prompt: 7,
                n_cached: 0,
                elapsed_s: 0.05,
                spec: None,
            })
            .unwrap();
            drop(tx);
            let mut receipt = receipt;
            blocking_response_with_receipt(
                rx,
                "m".into(),
                true,
                Vec::new(),
                None,
                Envelope::new(true),
                &mut receipt,
                None,
            )
            .await
        };

        // Unarmed receipt (the unmarked-tenant shape): the seam must not feed it a byte.
        let plain = MockMetering::admit_all();
        let receipt = plain.open(
            &metering::RequestMeta {
                request_id: "cap-unmarked",
                tenant: "unmarked",
                principal: None,
                model: "m",
                route: "/v1/chat/completions",
                lane: "interactive",
                stream: false,
                max_tokens: None,
                reserved_ctx: None,
            },
            None,
        );
        let response = drive(Some(receipt)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            !plain.events().iter().any(|e| matches!(
                e,
                MeterEvent::CaptureDelta(_) | MeterEvent::CapturePrompt(_)
            )),
            "an unarmed receipt must see no capture traffic: {:?}",
            plain.events()
        );

        // Armed receipt: the prompt payload lands byte-exact and the deltas reassemble
        // the completion byte-exact, alongside the terminal usage.
        let capturing = MockMetering::capturing();
        let mut receipt = capturing.open(
            &metering::RequestMeta {
                request_id: "cap-marked",
                tenant: "marked",
                principal: None,
                model: "m",
                route: "/v1/chat/completions",
                lane: "interactive",
                stream: false,
                max_tokens: None,
                reserved_ctx: None,
            },
            None,
        );
        assert!(receipt.wants_capture());
        receipt.arm_capture(prompt.clone());
        let response = drive(Some(receipt)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "Hello");

        let events = capturing.events();
        assert!(
            events.contains(&MeterEvent::CapturePrompt(prompt.clone())),
            "prompt must arm byte-exact: {events:?}"
        );
        let completion: String = events
            .iter()
            .filter_map(|e| match e {
                MeterEvent::CaptureDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            completion, "Hello",
            "the deltas must reassemble the served completion byte-exact: {events:?}"
        );
        assert!(
            events.contains(&MeterEvent::Complete {
                prompt: 7,
                cached: 0,
                completion: 2,
            }),
            "worker-truth usage settles alongside the capture: {events:?}"
        );
    }

    fn tool_caps() -> ModelCaps {
        ModelCaps {
            tools_branch: true,
            qwen_think: true,
            think_switch: true,
            chat_ok: true,
            ..Default::default()
        }
    }

    /// A qwen-class model that ALSO carries the qwen3.8 reasoning-effort ladder — the shape of
    /// the deployed `qwen/qwen3.8-27b`. Distinct from `tool_caps()` (ornith's shape: the same
    /// binary switch, no depth input) because that difference is exactly what decides whether a
    /// graded level is honoured or refused.
    fn ladder_caps() -> ModelCaps {
        ModelCaps {
            qwen_effort: true,
            ..tool_caps()
        }
    }

    fn gemma_tool_caps() -> ModelCaps {
        ModelCaps {
            tools_branch: true,
            gemma_think: true,
            chat_ok: true,
            instruct_type: Some("gemma".into()),
            ..Default::default()
        }
    }

    fn hy3_tool_caps() -> ModelCaps {
        ModelCaps {
            tools_branch: true,
            hy3: true,
            chat_ok: true,
            effort_levels: true,
            instruct_type: Some("hy3".into()),
            ..Default::default()
        }
    }

    fn gemma_template(kind: &str) -> String {
        let file = match kind {
            "qat" => "qat-trunk-template.jinja",
            _ => "official-tooluse-template.jinja",
        };
        let path = format!(
            "{}/../../research/gemma4-tools-20260817/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// Translate a fixture request (OpenAI shape + optional Google-native `tool_responses`)
    /// into the renderer's inputs, REUSING the real serve helpers (`prepare_tools`,
    /// `render_req_tool_call`, `content_to_text`, `json_to_val`, `parse_think`) so this stays
    /// a faithful mirror of `build_chat_request`, not a second implementation.
    fn render_fixture(request: &serde_json::Value, template: &str) -> String {
        let tools_arr = request
            .get("tools")
            .and_then(|t| t.as_array())
            .cloned()
            .unwrap_or_default();
        let (tools_json, tools_struct, _schemas) = if tools_arr.is_empty() {
            (Vec::new(), Vec::new(), HashMap::new())
        } else {
            prepare_tools(&tools_arr).unwrap()
        };
        let effort = request
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
            .map(String::from);
        let (think, _lvl, _explicit) =
            parse_think(&effort, &None, None, None, None, false).unwrap();

        let mut turns: Vec<TmplTurn> = Vec::new();
        for msg in request["messages"].as_array().unwrap() {
            let role = msg["role"].as_str().unwrap();
            let role = if role == "developer" { "system" } else { role };
            let content =
                content_to_text(msg.get("content").unwrap_or(&serde_json::Value::Null)).unwrap();
            let tool_calls = msg
                .get("tool_calls")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .map(|tc| {
                            let rtc: ReqToolCall = serde_json::from_value(tc.clone()).unwrap();
                            render_req_tool_call(&rtc).unwrap()
                        })
                        .collect()
                })
                .unwrap_or_default();
            let tool_responses = msg
                .get("tool_responses")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .map(|tr| {
                            (
                                tr.get("name").and_then(|n| n.as_str()).unwrap().to_string(),
                                json_to_val(&tr["response"]),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            turns.push(TmplTurn {
                role: role.to_string(),
                content,
                tool_calls,
                reasoning: msg
                    .get("reasoning")
                    .and_then(|r| r.as_str())
                    .map(String::from)
                    .filter(|s| !s.is_empty()),
                tool_call_id: msg
                    .get("tool_call_id")
                    .and_then(|s| s.as_str())
                    .map(String::from),
                tool_name: msg.get("name").and_then(|s| s.as_str()).map(String::from),
                tool_responses,
                task: None,
                tools: Vec::new(),
            });
        }
        chat::apply_chat_template_tools_ex(
            Some(template),
            &turns,
            true,
            &tools_json,
            &tools_struct,
            think,
            None,
            None,
        )
        .unwrap()
    }

    /// Byte-parity oracle gate: every research/gemma4-tools-20260817/fixtures/* pair, rendered
    /// through the memra gemma4 arm, must equal the bytes the OFFICIAL jinja produced under
    /// jinja2 (gen_fixtures.py). The jinja is the LAW; this is what makes it enforceable.
    #[test]
    fn gemma4_tools_fixtures_match_the_official_jinja() {
        let dir = format!(
            "{}/../../research/gemma4-tools-20260817/fixtures",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read fixtures dir {dir}: {e}"))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .collect();
        entries.sort();
        assert!(
            entries.len() >= 14,
            "expected >=14 fixtures, found {}",
            entries.len()
        );
        let (mut official, mut qat) = (0u32, 0u32);
        for d in entries {
            let input: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(d.join("input.json")).unwrap())
                    .unwrap();
            let expected = std::fs::read_to_string(d.join("expected.txt")).unwrap();
            let kind = input
                .get("template")
                .and_then(|t| t.as_str())
                .unwrap_or("official");
            match kind {
                "qat" => qat += 1,
                _ => official += 1,
            }
            let tmpl = gemma_template(kind);
            let got = render_fixture(&input["request"], &tmpl);
            assert_eq!(
                got, expected,
                "fixture {:?} diverged from the jinja oracle",
                d
            );
        }
        assert!(
            official >= 12 && qat >= 2,
            "coverage: {official} official, {qat} qat"
        );
    }

    /// The REAL serve pipeline (`build_chat_request`) renders gemma4 tool DEFINITIONS + a
    /// tool-call/response cycle byte-identically to the fixture oracle — proving the OpenAI
    /// chat surface (and, via the shared path, /v1/messages + /v1/responses) flows tools to
    /// the gemma trunk. Native-only fixtures (Google `tool_responses`) are covered by the
    /// oracle test above, not here (the OpenAI request shape cannot express them).
    #[test]
    fn gemma4_tools_flow_through_build_chat_request() {
        let tmpl = gemma_template("official");
        for name in [
            "01-system-tools-basic",
            "04-single-call-cycle",
            "07-multi-cycle-agentic",
        ] {
            let path = format!(
                "{}/../../research/gemma4-tools-20260817/fixtures/{name}/input.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let input: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let expected_path = format!(
                "{}/../../research/gemma4-tools-20260817/fixtures/{name}/expected.txt",
                env!("CARGO_MANIFEST_DIR")
            );
            let expected = std::fs::read_to_string(&expected_path).unwrap();
            let req: ChatCompletionReq = serde_json::from_value(input["request"].clone()).unwrap();
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                req,
                Some(&gemma_tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .unwrap();
            let got = chat::apply_chat_template_tools_ex(
                Some(&tmpl),
                &plan.request.chat_turns,
                true,
                &plan.request.tools_json,
                &plan.request.tools_struct,
                plan.request.think,
                plan.request.reasoning_effort.as_deref(),
                None,
            )
            .unwrap();
            assert_eq!(got, expected, "pipeline render diverged for {name}");
        }
    }

    // ---- GLM-5.3-Flash (`glm5_next`) surface (lane/glm53-flash-bringup, 2026-08-27) --------
    // THE STANDARD-SURFACE LAW for this model: three wire formats plus tools, all through the
    // vendor's own template bytes. Before this arm, every glm5 marker was ALSO a qwen marker,
    // so `apply_chat_template_tools_ex` fell through to the ChatML arm and served `<|im_start|>`
    // turns to a checkpoint whose special vocabulary does not contain them — fluent, because
    // GLM follows the qwen tool-format instruction it was handed in-context, and invisible
    // without a byte oracle. The oracle is the checkpoint's own chat_template.jinja.

    fn glm5_template() -> String {
        let path = format!(
            "{}/../../research/glm53-flash-bringup-20260827/chat_template.jinja",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// The caps the worker probes off that template — copied from the live boot line
    /// (`tools=true think=true think_switch=false chat_ok=true effort_levels=true
    /// qwen_effort=false gemma_think=false dsv4=false ctx=1048576 tok="glm4"`), plus the
    /// `glm5` dialect flag this lane added.
    fn glm5_caps() -> ModelCaps {
        ModelCaps {
            tools_branch: true,
            qwen_think: true,
            think_switch: false,
            chat_ok: true,
            context_length: 1_048_576,
            tokenizer: "glm4".into(),
            instruct_type: Some("glm".into()),
            effort_levels: true,
            glm5: true,
            ..Default::default()
        }
    }

    /// One fixture request through the REAL serve pipeline, rendered with the vendor template.
    fn glm5_render(body: serde_json::Value) -> Result<String, String> {
        let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, Some(&glm5_caps()), tx, lanes::Lane::Interactive, None)?;
        chat::apply_chat_template_tools_ex(
            Some(&glm5_template()),
            &plan.request.chat_turns,
            true,
            &plan.request.tools_json,
            &plan.request.tools_struct,
            plan.request.think,
            plan.request.reasoning_effort.as_deref(),
            None,
        )
    }

    /// Byte-parity oracle gate: every research/glm53-flash-bringup-20260827/surface-fixtures/*
    /// pair, run through `build_chat_request` + the glm5 arm, must equal the bytes the VENDOR
    /// jinja produced under jinja2 (gen_surface_fixtures.py). The jinja is the LAW; this is
    /// what makes it enforceable.
    #[test]
    fn glm5_fixtures_match_the_vendor_jinja() {
        let dir = format!(
            "{}/../../research/glm53-flash-bringup-20260827/surface-fixtures",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read fixtures dir {dir}: {e}"))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .collect();
        entries.sort();
        assert!(
            entries.len() >= 22,
            "expected >=22 fixtures, found {}",
            entries.len()
        );
        for d in entries {
            let input: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(d.join("input.json")).unwrap())
                    .unwrap();
            let expected = std::fs::read_to_string(d.join("expected.txt")).unwrap();
            let got = glm5_render(input["request"].clone())
                .unwrap_or_else(|e| panic!("fixture {d:?} refused: {e}"));
            assert_eq!(
                got, expected,
                "fixture {d:?} diverged from the jinja oracle"
            );
        }
    }

    /// THE DEFECT THIS ARM EXISTS TO CLOSE. The GLM template contains `<think>`,
    /// `add_generation_prompt` AND `<tools>`, so every qwen marker check matches it. Without
    /// the glm5 dispatch the renderer emitted ChatML — tokens this checkpoint does not carry as
    /// specials at all (`extra_special_tokens` is `[gMASK] <sop> <|system|> <|user|>
    /// <|assistant|> <|observation|>` …), so the whole frame tokenized as ordinary text.
    #[test]
    fn glm5_never_renders_chatml() {
        let tmpl = glm5_template();
        // The markers that used to win the dispatch are all really there.
        assert!(tmpl.contains("<think>") && tmpl.contains("add_generation_prompt"));
        assert!(tmpl.contains("<tools>"));
        assert!(chat::template_is_glm5(&tmpl));
        for body in [
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                   "tools": [{"type": "function", "function": {"name": "f",
                              "parameters": {"type": "object", "properties": {}}}}]}),
        ] {
            let got = glm5_render(body).unwrap();
            assert!(
                !got.contains("<|im_start|>") && !got.contains("<|im_end|>"),
                "glm5 rendered ChatML frames: {got:?}"
            );
            assert!(
                got.starts_with("[gMASK]<sop><|system|>Reasoning Effort: "),
                "{got:?}"
            );
            assert!(got.ends_with("<|assistant|><think>"), "{got:?}");
        }
    }

    /// `reasoning_effort` must reach the TEMPLATE (a rendered system line), never the sampler,
    /// and the model's `max` rung — a real tier ABOVE `high`, and its own default — must
    /// survive `canonical_effort_for` instead of clamping into `high`.
    #[test]
    fn glm5_reasoning_effort_renders_and_keeps_its_max_tier() {
        for (sent, line) in [
            (None, "Max"),
            (Some("low"), "Low"),
            // no medium rung in this ladder: the middle ask maps UP to the middle
            // rung (owner ruling 2026-09-02, issue #75). Never through the
            // template's `else` arm, which is Max: answering "reason less" with
            // the deepest setting.
            (Some("medium"), "High"),
            (Some("high"), "High"),
            (Some("xhigh"), "Max"),
            (Some("max"), "Max"),
            (Some("ultra"), "Max"),
        ] {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            if let Some(v) = sent {
                body["reasoning_effort"] = json!(v);
            }
            let got = glm5_render(body).unwrap();
            assert!(
                got.starts_with(&format!("[gMASK]<sop><|system|>Reasoning Effort: {line}<|")),
                "reasoning_effort {sent:?} should render {line:?}: {got:?}"
            );
        }
        // The level is a RENDER input, not a sampler knob: two efforts that render different
        // system lines must leave the sampler identical.
        let sampler_of = |v: &str| {
            let req: ChatCompletionReq = serde_json::from_value(
                // seed pinned: it is drawn fresh per request, and this assertion is about
                // whether the effort level perturbs the SAMPLER, not about the draw.
                json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": v, "seed": 7}),
            )
            .unwrap();
            let (tx, _rx) = worker::event_channel();
            let plan =
                build_chat_request(req, Some(&glm5_caps()), tx, lanes::Lane::Interactive, None)
                    .unwrap();
            format!("{:?}", plan.request.sampler_cfg)
        };
        assert_eq!(sampler_of("low"), sampler_of("max"));
        // And the canonical table itself keeps the tier for this model's key.
        assert_eq!(canonical_effort_for("max", true), Some("max"));
        assert_eq!(canonical_effort_for("xhigh", true), Some("max"));
        assert_eq!(canonical_effort_for("max", false), Some("high"));
    }

    /// The off-request this template genuinely cannot honour stays a NAMED 400 (it opens
    /// `<think>` unconditionally and has no `enable_thinking`), and an out-of-table level
    /// stays a 400 — neither becomes a silent downgrade now that the level is delivered.
    #[test]
    fn glm5_refuses_what_its_template_cannot_honour() {
        for (value, needle) in [
            ("none", "cannot disable reasoning"),
            ("minimal", "cannot disable reasoning"),
            ("bogus", "bad reasoning_effort"),
        ] {
            let err = glm5_render(json!({"model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": value}))
            .err()
            .unwrap_or_else(|| panic!("reasoning_effort {value:?} must be refused"));
            assert!(err.contains(needle), "{value}: {err}");
        }
    }

    /// THE STANDARD-SURFACE LAW at the byte level, for this model: the same semantic request
    /// expressed in each of the three wire vocabularies — including a tool definition and a
    /// full call/result cycle — must render the SAME glm5 prompt bytes.
    #[test]
    fn one_glm5_request_renders_identical_bytes_on_all_three_surfaces() {
        // TWO parallel calls whose results come back in REVERSED order. That shape is what
        // makes this test discriminate: the glm5 arm re-orders an `<|observation|>` run onto
        // the preceding assistant turn's `tool_calls` order, but ONLY when every result's id
        // resolves (`glm5_can_sort`) — otherwise it renders in message order. With one call
        // both branches emit identical bytes, so a translation surface that silently dropped
        // `tool_call_id` would still pass. With two, reversed, it cannot.
        let chat = json!({
            "model": "m",
            "reasoning_effort": "high",
            "messages": [
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": null,
                 "tool_calls": [
                     {"id": "c1", "type": "function",
                      "function": {"name": "get_weather",
                                   "arguments": "{\"city\": \"Paris\"}"}},
                     {"id": "c2", "type": "function",
                      "function": {"name": "get_weather",
                                   "arguments": "{\"city\": \"Rome\"}"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "rome:27"},
                {"role": "tool", "tool_call_id": "c1", "content": "paris:21"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get the current weather for a city",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]
        });
        let responses = responses_api::translate(&json!({
            "model": "m",
            "reasoning": {"effort": "high"},
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Weather in Paris and Rome?"}]},
                {"type": "function_call", "call_id": "c1", "name": "get_weather",
                 "arguments": "{\"city\": \"Paris\"}"},
                {"type": "function_call", "call_id": "c2", "name": "get_weather",
                 "arguments": "{\"city\": \"Rome\"}"},
                {"type": "function_call_output", "call_id": "c2", "output": "rome:27"},
                {"type": "function_call_output", "call_id": "c1", "output": "paris:21"}
            ],
            "tools": [{"type": "function", "name": "get_weather",
                       "description": "Get the current weather for a city",
                       "parameters": {"type": "object",
                                      "properties": {"city": {"type": "string"}},
                                      "required": ["city"]}}]
        }))
        .expect("/v1/responses translate");
        let messages = anthropic::translate(&json!({
            "model": "m",
            "max_tokens": 256,
            "output_config": {"effort": "high"},
            "messages": [
                {"role": "user", "content": "Weather in Paris and Rome?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "c1", "name": "get_weather",
                     "input": {"city": "Paris"}},
                    {"type": "tool_use", "id": "c2", "name": "get_weather",
                     "input": {"city": "Rome"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c2", "content": "rome:27"},
                    {"type": "tool_result", "tool_use_id": "c1", "content": "paris:21"}]}
            ],
            "tools": [{"name": "get_weather",
                       "description": "Get the current weather for a city",
                       "input_schema": {"type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"]}}]
        }))
        .expect("/v1/messages translate");
        let want = glm5_render(chat).expect("chat");
        // The tool cycle really did render the native dialect, not a qwen-shaped fallback.
        assert!(
            want.contains(
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>\
                 </tool_call><tool_call>get_weather<arg_key>city</arg_key>\
                 <arg_value>Rome</arg_value></tool_call>"
            ),
            "{want:?}"
        );
        // The ids resolved, so the run was re-ordered onto CALL order (Paris, Rome), not the
        // message order the client sent (Rome, Paris). That is the byte this test discriminates
        // on: any surface that loses `tool_call_id` renders the pair the other way round.
        assert!(
            want.contains(
                "<|observation|><tool_response>paris:21</tool_response>\
                 <tool_response>rome:27</tool_response>"
            ),
            "{want:?}"
        );
        assert!(
            want.contains("<|system|>Reasoning Effort: High"),
            "{want:?}"
        );
        for (surface, body) in [
            ("/v1/responses", responses),
            ("/v1/messages", messages.clone()),
        ] {
            let got = glm5_render(body).unwrap_or_else(|e| panic!("{surface}: {e}"));
            assert_eq!(
                got, want,
                "{surface} rendered DIFFERENT glm5 prompt bytes than /v1/chat/completions"
            );
        }
        // NEGATIVE CONTROL — the equality above only means something if losing the ids really
        // changes the bytes. Strip `tool_call_id` from the result turns (what a translation
        // surface that dropped it would hand the renderer) and the run must fall back to
        // MESSAGE order, diverging. Without this, a `can_sort` that silently answered `false`
        // everywhere would keep the whole test green.
        let mut idless = messages;
        for m in idless["messages"].as_array_mut().unwrap() {
            if m["role"] == "tool" {
                m.as_object_mut().unwrap().remove("tool_call_id");
            }
        }
        let got = glm5_render(idless).expect("id-less render");
        assert_ne!(
            got, want,
            "dropping tool_call_id must change the rendered order — this test cannot detect \
             a surface that loses ids otherwise"
        );
        assert!(
            got.contains(
                "<|observation|><tool_response>rome:27</tool_response>\
                 <tool_response>paris:21</tool_response>"
            ),
            "{got:?}"
        );
    }

    /// The chat path must arm the GLM parser, not the qwen `<function=` scanner — otherwise
    /// every native call surfaces VERBATIM as content behind a 200.
    #[test]
    fn glm5_chat_arms_the_native_tool_parser() {
        let req: ChatCompletionReq = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather",
                       "parameters": {"type": "object",
                                      "properties": {"city": {"type": "string"}}}}}]}))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, Some(&glm5_caps()), tx, lanes::Lane::Interactive, None)
            .unwrap();
        let mut parser = plan.parser.expect("glm5 tools request must carry a parser");
        let pieces = parser.push(
            "reasoning here</think><tool_call>get_weather<arg_key>city</arg_key>\
             <arg_value>Paris</arg_value></tool_call>",
        );
        let calls: Vec<_> = pieces
            .iter()
            .filter_map(|p| match p {
                toolcall::Piece::Call(c) => Some((c.name.as_str(), c.arguments.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            vec![("get_weather", r#"{"city":"Paris"}"#)],
            "{pieces:?}"
        );
        assert!(
            pieces
                .iter()
                .any(|p| matches!(p, toolcall::Piece::Reasoning(r) if r == "reasoning here")),
            "{pieces:?}"
        );
        // and nothing leaked into content.
        assert!(
            !pieces
                .iter()
                .any(|p| matches!(p, toolcall::Piece::Content(_))),
            "{pieces:?}"
        );
        // A NON-tools glm5 request must still carry a parser: this template's `<think>` tail is
        // unconditional, so without one the whole reasoning block lands in `content` with the
        // `</think>` tag in it. (The wiring half of `glm5_without_tools_is_a_reasoning_splitter_only`.)
        let req: ChatCompletionReq = serde_json::from_value(
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, Some(&glm5_caps()), tx, lanes::Lane::Interactive, None)
            .unwrap();
        let mut parser = plan
            .parser
            .expect("glm5 non-tools request must still split reasoning");
        let pieces = parser.push("weighing it</think>The answer.");
        assert!(
            pieces
                .iter()
                .any(|p| matches!(p, toolcall::Piece::Reasoning(r) if r == "weighing it")),
            "{pieces:?}"
        );
        assert!(
            pieces
                .iter()
                .any(|p| matches!(p, toolcall::Piece::Content(c) if c == "The answer.")),
            "{pieces:?}"
        );
    }

    /// The worker's PLAIN fast path maps turns to `(role, content)` tuples and drops
    /// `reasoning` — so on a dialect that replays prior reasoning into the prompt it would
    /// render different bytes than the tools path for the same request. GLM-5.3-Flash is such a
    /// dialect (`<think>{reasoning}</think>` on every assistant turn, unconditionally), and the
    /// two paths must never disagree: a re-render that does not match its own live stream is
    /// also what stops a parked session from ever resuming (lane/dflash2-session-reuse).
    #[test]
    fn glm5_plain_fast_path_never_drops_replayed_reasoning() {
        let with_reasoning = vec![
            chat::Turn {
                role: "user".into(),
                content: "a".into(),
                ..Default::default()
            },
            chat::Turn {
                role: "assistant".into(),
                content: "A".into(),
                reasoning: Some("I considered a.".into()),
                ..Default::default()
            },
            chat::Turn {
                role: "user".into(),
                content: "b".into(),
                ..Default::default()
            },
        ];
        // The predicate must refuse the fast path for this shape...
        assert!(!worker::plain_chat_render_path(
            &[],
            &chat::ThinkMode::Default,
            None,
            &with_reasoning,
            false,
        ));
        // ...and the same turns WITHOUT reasoning still take it (the fast path is not disabled
        // wholesale — only for the shape it cannot render faithfully).
        let plain_turns: Vec<chat::Turn> = with_reasoning
            .iter()
            .cloned()
            .map(|mut t| {
                t.reasoning = None;
                t
            })
            .collect();
        assert!(worker::plain_chat_render_path(
            &[],
            &chat::ThinkMode::Default,
            None,
            &plain_turns,
            false,
        ));
        // And the bytes the two paths would produce really do differ on this dialect, so the
        // predicate above is load-bearing rather than defensive.
        let tmpl = glm5_template();
        let via_tools = chat::apply_chat_template_tools_ex(
            Some(&tmpl),
            &with_reasoning,
            true,
            &[],
            &[],
            chat::ThinkMode::Default,
            None,
            None,
        )
        .unwrap();
        let msgs: Vec<(&str, &str)> = with_reasoning
            .iter()
            .map(|t| (t.role.as_str(), t.content.as_str()))
            .collect();
        let via_plain = chat::apply_chat_template_str(Some(&tmpl), &msgs, true);
        assert!(
            via_tools.contains("<think>I considered a.</think>"),
            "{via_tools:?}"
        );
        assert_ne!(via_tools, via_plain);
        // On the no-reasoning shape the two paths are byte-identical, which is what makes
        // keeping the fast path there safe.
        let plain_msgs: Vec<(&str, &str)> = plain_turns
            .iter()
            .map(|t| (t.role.as_str(), t.content.as_str()))
            .collect();
        assert_eq!(
            chat::apply_chat_template_tools_ex(
                Some(&tmpl),
                &plain_turns,
                true,
                &[],
                &[],
                chat::ThinkMode::Default,
                None,
                None,
            )
            .unwrap(),
            chat::apply_chat_template_str(Some(&tmpl), &plain_msgs, true)
        );
    }

    /// `/v1/models` must not advertise a capability the server refuses by name. A template
    /// whose `<think>` tail opens unconditionally with no `enable_thinking` switch cannot take
    /// constrained decoding at all — the request 400s — so the row says `false`.
    #[test]
    fn glm5_model_row_does_not_claim_structured_output() {
        let caps = glm5_caps();
        let row = model_entry_v1("zai/glm-5.3-flash", Some(&caps), None);
        assert_eq!(row["capabilities"]["structured_output"], json!(false));
        assert_eq!(row["capabilities"]["tools"], json!(true));
        assert_eq!(row["capabilities"]["reasoning"], json!(true));
        // and the refusal the row now matches is real.
        let err = glm5_render(json!({"model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}}))
        .expect_err("response_format must be refused on a switchless think template");
        // Post-think constrained decoding (lane/step37-postthink-grammar) widened the refusal
        // text: glm5's template has neither the switch nor a derivable think-close contract,
        // so the refusal (and the false row) stand; only the message grew.
        assert!(
            err.contains("neither an enable_thinking switch nor a recognizable"),
            "{err}"
        );
        // A model that CAN close its think tail keeps the true claim.
        let switchable = model_entry_v1("q", Some(&tool_caps()), None);
        assert_eq!(switchable["capabilities"]["structured_output"], json!(true));
        // The OpenRouter catalog must not disagree with the contract-v2 row about one model:
        // it advertised `json_mode` + `structured_outputs` unconditionally.
        let glm_params = openrouter_supported_parameters(Some(&caps), None, true);
        assert!(
            glm_params.get("structured_outputs").is_none(),
            "{glm_params}"
        );
        // THE step37 SHAPE (v0.123.0 regression, found by the 2026-09-01 claim re-seal):
        // switchless force-open think WITH a derivable think-close contract is SERVED via
        // post-think constrained decoding, so both catalogs must say true. v0.123.0's
        // heuristic predicate advertised false here while the live server returned
        // schema-valid response_format output on the same model.
        let step_like = ModelCaps {
            chat_ok: true,
            qwen_think: true,
            think_switch: false,
            think_close: vec![128799],
            ..caps.clone()
        };
        let step_row = model_entry_v1("stepfun/step-3.7-flash", Some(&step_like), None);
        assert_eq!(step_row["capabilities"]["structured_output"], json!(true));
        let step_params = openrouter_supported_parameters(Some(&step_like), None, true);
        assert!(
            step_params.get("structured_outputs").is_some(),
            "{step_params}"
        );
        assert!(glm_params.get("json_mode").is_none(), "{glm_params}");
        assert!(glm_params.get("tools").is_some(), "{glm_params}");
        // Issue #75: glm5's published levels are its native rungs. The enum is
        // glm5-scoped, not a generic effort advertisement.
        assert_eq!(
            glm_params.get("reasoning_effort"),
            Some(&json!({ "type": "enum", "values": ["low", "high", "max"] })),
            "{glm_params}"
        );
        let qwen_params = openrouter_supported_parameters(Some(&tool_caps()), None, true);
        assert!(
            qwen_params.get("structured_outputs").is_some(),
            "{qwen_params}"
        );
        assert!(qwen_params.get("json_mode").is_some(), "{qwen_params}");
        assert!(
            qwen_params.get("reasoning_effort").is_none(),
            "{qwen_params}"
        );
    }

    /// The catalog must not advertise the checkpoint's trained context as a serving claim.
    /// glm5 declares 1,048,576 trained, and the 3-card resident shape measurably cannot prime
    /// it (`research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md`: the 1M deep
    /// prime died `layer 31: DSA k-pool selection failed: CUDA_ERROR_OUT_OF_MEMORY`). When the
    /// deployment pins its operational envelope (`max_prompt_length` + `max_output_length`),
    /// every catalog body publishes that envelope, not the trained figure; with no envelope
    /// pinned the trained value stands.
    #[test]
    fn catalog_context_claim_is_capped_by_the_deployment_envelope() {
        let caps = glm5_caps();
        assert_eq!(caps.context_length, 1_048_576);
        let metadata = OpenRouterModelMetadata {
            max_prompt_length: Some(126_976),
            max_output_length: Some(4_096),
            ..Default::default()
        };
        // Envelope pinned below trained -> the envelope is the claim, on all three bodies.
        let row = model_entry_v1("zai/glm-5.3-flash", Some(&caps), Some(&metadata));
        assert_eq!(row["context_length"], json!(131_072));
        let or_row = model_entry_openrouter("zai/glm-5.3-flash", Some(&caps), Some(&metadata));
        assert_eq!(
            or_row["input_modalities"][0]["supported_inputs"]["max_context_length"]["value"],
            json!(131_072)
        );
        assert_eq!(
            published_context_length(Some(&caps), Some(&metadata)),
            Some(131_072)
        );
        // No envelope (or half an envelope) -> the trained value stands unchanged.
        assert_eq!(published_context_length(Some(&caps), None), Some(1_048_576));
        let half = OpenRouterModelMetadata {
            max_output_length: Some(4_096),
            ..Default::default()
        };
        assert_eq!(
            published_context_length(Some(&caps), Some(&half)),
            Some(1_048_576)
        );
        // An envelope above trained never inflates the claim.
        let wide = OpenRouterModelMetadata {
            max_prompt_length: Some(2_000_000),
            max_output_length: Some(2_000_000),
            ..Default::default()
        };
        assert_eq!(
            published_context_length(Some(&caps), Some(&wide)),
            Some(1_048_576)
        );
    }

    // ---- deepseek-v4 (encoding_dsv4) template arm (lane 5, 2026-08-18) --------------------
    // The oracle IS encoding_dsv4.py. Byte parity is the only acceptance (GGUF template-mint
    // law). Two gates: the generated matrix (research/dsv4-template-20260818/gen_fixtures.py,
    // 25 cases across 3 modes x {single,multi,system,tools,tool-results,tasks,reminder}) and
    // the artifact's AUTHORITATIVE encoding/tests/test_output_{1..4}. Plus a tokenization
    // cross-check: rendered bytes -> memra token ids == the official HF tokenizer ids.

    fn dsv4_sentinel() -> String {
        let path = format!(
            "{}/../../research/dsv4-template-20260818/dsv4-chat-template.sentinel.jinja",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// Build a dsv4 `TmplTurn` from a serve-shape (`reasoning`) OR OpenAI-shape
    /// (`reasoning_content`) message value, reusing the real serve helpers so this mirrors
    /// `build_chat_request`, not a second implementation. Per-turn `tools` (search-pipeline
    /// developer tools) are read from the message; the `task` head is read too.
    fn dsv4_turn(msg: &serde_json::Value) -> TmplTurn {
        let role = msg["role"].as_str().unwrap().to_string();
        let content =
            content_to_text(msg.get("content").unwrap_or(&serde_json::Value::Null)).unwrap();
        let reasoning = msg
            .get("reasoning")
            .or_else(|| msg.get("reasoning_content"))
            .and_then(|r| r.as_str())
            .map(String::from)
            .filter(|s| !s.is_empty());
        let tool_calls = msg
            .get("tool_calls")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .map(|tc| {
                        let rtc: ReqToolCall = serde_json::from_value(tc.clone()).unwrap();
                        render_req_tool_call(&rtc).unwrap()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let tools = msg
            .get("tools")
            .and_then(|a| a.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.get("function").map(json_to_val))
                    .collect()
            })
            .unwrap_or_default();
        TmplTurn {
            role,
            content,
            tool_calls,
            reasoning,
            tool_call_id: msg
                .get("tool_call_id")
                .and_then(|s| s.as_str())
                .map(String::from),
            tool_name: msg.get("name").and_then(|s| s.as_str()).map(String::from),
            tool_responses: Vec::new(),
            task: msg.get("task").and_then(|s| s.as_str()).map(String::from),
            tools,
        }
    }

    fn dsv4_req_tools(v: Option<&serde_json::Value>) -> Vec<chat::Val> {
        v.and_then(|t| t.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.get("function").map(json_to_val))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Byte-parity runner over one generated fixture dir (gen_fixtures.py), rendered under
    /// the given encoding revision. Both revisions' matrices run through the SAME arm —
    /// only the `Dsv4Encoding` differs (0731 re-gate, ENCODING-DIFF.md).
    fn dsv4_run_fixture_dir(subdir: &str, encoding: chat::Dsv4Encoding, min_fixtures: usize) {
        let dir = format!(
            "{}/../../research/dsv4-template-20260818/{subdir}",
            env!("CARGO_MANIFEST_DIR")
        );
        let tmpl = dsv4_sentinel();
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read fixtures dir {dir}: {e}"))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .collect();
        entries.sort();
        assert!(
            entries.len() >= min_fixtures,
            "expected >={min_fixtures} fixtures, found {}",
            entries.len()
        );
        for d in &entries {
            let input: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(d.join("input.json")).unwrap())
                    .unwrap();
            let expected = std::fs::read_to_string(d.join("expected.txt")).unwrap();
            let turns: Vec<TmplTurn> = input["turns"]
                .as_array()
                .unwrap()
                .iter()
                .map(dsv4_turn)
                .collect();
            let think = match input["think"].as_str().unwrap() {
                "chat" => ThinkMode::NoThink,
                _ => ThinkMode::Think,
            };
            let effort = input
                .get("reasoning_effort")
                .and_then(|v| v.as_str())
                .map(String::from);
            let req_tools = dsv4_req_tools(input.get("req_tools"));
            let agp = input["add_generation_prompt"].as_bool().unwrap_or(true);
            let got = chat::apply_chat_template_tools_ex(
                Some(&tmpl),
                &turns,
                agp,
                &[],
                &req_tools,
                think,
                effort.as_deref(),
                Some(encoding),
            )
            .unwrap();
            assert_eq!(got, expected, "fixture {:?} diverged from the oracle", d);
        }
    }

    #[test]
    fn dsv4_template_fixtures_match_the_oracle() {
        dsv4_run_fixture_dir("fixtures", chat::Dsv4Encoding::Preview, 20);
    }

    /// 0731 re-gate (support-checklist item 3): the full mode x effort x shape matrix
    /// generated from the OFFICIAL 0731 encoding_dsv4.py (ref-0731/encoding/), including
    /// explicit low/high/max rungs of the remapped ladder — "high" is a REAL prefix here
    /// (the preview's "max" text) and "max" is the new stronger text. The preview matrix
    /// above keeps passing untouched (regression: both encodings stay supported).
    #[test]
    fn dsv4_0731_fixtures_match_the_oracle() {
        dsv4_run_fixture_dir("fixtures-0731", chat::Dsv4Encoding::V0731, 40);
    }

    #[test]
    fn dsv4_artifact_fixtures_are_byte_identical() {
        // The NVFP4 artifact's encoding/tests are AUTHORITATIVE (SEMANTICS.md §6). Case 1 has
        // a top-level `tools` merged onto messages[0] (test_encoding_dsv4.py); case 3 carries
        // tools on its developer message; think mode is thinking for 1-3, chat for 4.
        let base = format!(
            "{}/../../research/dsv4-template-20260818/ref/artifact-encoding/tests",
            env!("CARGO_MANIFEST_DIR")
        );
        let tmpl = dsv4_sentinel();
        for (n, think) in [
            (1u32, ThinkMode::Think),
            (2, ThinkMode::Think),
            (3, ThinkMode::Think),
            (4, ThinkMode::NoThink),
        ] {
            let td: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!("{base}/test_input_{n}.json")).unwrap(),
            )
            .unwrap();
            let (messages, tools) = if td.is_object() {
                (td["messages"].clone(), td.get("tools").cloned())
            } else {
                (td.clone(), None)
            };
            let mut turns: Vec<TmplTurn> = Vec::new();
            for (i, msg) in messages.as_array().unwrap().iter().enumerate() {
                let mut t = dsv4_turn(msg);
                if i == 0
                    && let Some(tl) = &tools
                {
                    t.tools = tl
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter_map(|x| x.get("function").map(json_to_val))
                        .collect();
                }
                turns.push(t);
            }
            let expected = std::fs::read_to_string(format!("{base}/test_output_{n}.txt")).unwrap();
            // The 4 authoritative fixtures are byte-identical between the preview and 0731
            // artifacts (verified by diff, ENCODING-DIFF.md) and carry no reasoning_effort,
            // so they must render identically under BOTH encoding revisions.
            for encoding in [chat::Dsv4Encoding::Preview, chat::Dsv4Encoding::V0731] {
                let got = chat::apply_chat_template_tools_ex(
                    Some(&tmpl),
                    &turns,
                    true,
                    &[],
                    &[],
                    think,
                    None,
                    Some(encoding),
                )
                .unwrap();
                assert_eq!(
                    got, expected,
                    "artifact fixture {n} diverged from the oracle under {encoding:?}"
                );
            }
        }
    }

    #[test]
    fn dsv4_default_thinkmode_renders_thinking() {
        // Default == Think for dsv4 (the model has no template-own chat default; thinking is
        // the honest serve default — TEMPLATE-SEMANTICS.md finding #1). NoThink == chat.
        let tmpl = dsv4_sentinel();
        let turns = vec![TmplTurn {
            role: "user".into(),
            content: "Hi".into(),
            ..Default::default()
        }];
        let dflt = chat::apply_chat_template_tools_ex(
            Some(&tmpl),
            &turns,
            true,
            &[],
            &[],
            ThinkMode::Default,
            None,
            None,
        )
        .unwrap();
        let think = chat::apply_chat_template_tools_ex(
            Some(&tmpl),
            &turns,
            true,
            &[],
            &[],
            ThinkMode::Think,
            None,
            None,
        )
        .unwrap();
        assert_eq!(dflt, think);
        assert!(
            dflt.ends_with("<\u{ff5c}Assistant\u{ff5c}><think>"),
            "{dflt:?}"
        );
        let chat_mode = chat::apply_chat_template_tools_ex(
            Some(&tmpl),
            &turns,
            true,
            &[],
            &[],
            ThinkMode::NoThink,
            None,
            None,
        )
        .unwrap();
        assert!(
            chat_mode.ends_with("<\u{ff5c}Assistant\u{ff5c}></think>"),
            "{chat_mode:?}"
        );
    }

    /// Rendered bytes -> memra token ids must equal the official HF tokenizer ids banked
    /// next to the fixtures (gen: HF `tokenizers` over ref/tokenizer.json — one sha across
    /// preview/0731 source/mint, so ONE ref dir serves both matrices). Proves the
    /// deepseek-v3 pre-tokenizer detection + BPE are integer-exact for dsv4.
    fn dsv4_run_tokenization_crosscheck(subdir: &str) {
        let base = format!(
            "{}/../../research/dsv4-template-20260818",
            env!("CARGO_MANIFEST_DIR")
        );
        let refdir = std::path::Path::new(&base).join("ref");
        let tok = memra_tokenizer::Tokenizer::from_hf_dir(&refdir)
            .expect("load dsv4 tokenizer from ref dir");
        assert_eq!(tok.pre(), "deepseek-v3", "pre-tokenizer family detection");
        let banked: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{base}/{subdir}/tokenization-crosscheck.json"))
                .unwrap(),
        )
        .unwrap();
        let obj = banked.as_object().unwrap();
        assert!(obj.len() >= 3, "expected >=3 cross-check fixtures");
        for (name, ids_v) in obj {
            let rendered =
                std::fs::read_to_string(format!("{base}/{subdir}/{name}/expected.txt")).unwrap();
            let want: Vec<u32> = ids_v
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = tok.encode(&rendered, true);
            assert_eq!(got, want, "tokenization diverged for {name}");
        }
    }

    #[test]
    fn dsv4_tokenization_crosscheck_matches_official_ids() {
        dsv4_run_tokenization_crosscheck("fixtures");
    }

    /// 0731 re-gate: id parity on fixtures that carry the REMAPPED effort prefixes (the
    /// new "Beyond maximum" text and the high rung's prefix) — the only new bytes 0731's
    /// encoding introduces to the rendered surface.
    #[test]
    fn dsv4_0731_tokenization_crosscheck_matches_official_ids() {
        dsv4_run_tokenization_crosscheck("fixtures-0731");
    }

    #[test]
    fn dsv4_tool_result_long_runs_render_tokenize_roundtrip() {
        // Regression guard for llama.cpp #26965 (recon: research/deepseek-flash-20260818/
        // RECON.md): upstream's deepseek-v3-class pre-tokenizer runs through backtracking
        // std::regex and stack-overflows on long uniform ASCII runs inside tool results
        // ('Z' x 131072). memra's port (unicode::split_deepseek_v3) is an iterative scan —
        // no regex engine, no recursion — so a dsv4 chat whose tool RESULT carries a giant
        // uniform run must render, tokenize, and round-trip (decode(encode(x)) == x)
        // within a sane bound. Id parity vs the official HF tokenizer on the 131k case is
        // a receipts-time cross-check (see RECEIPTS.md), not a gate here: the gate is our
        // own crash-safety + round-trip.
        let base = format!(
            "{}/../../research/dsv4-template-20260818",
            env!("CARGO_MANIFEST_DIR")
        );
        let refdir = std::path::Path::new(&base).join("ref");
        let tok = memra_tokenizer::Tokenizer::from_hf_dir(&refdir)
            .expect("load dsv4 tokenizer from ref dir");
        assert_eq!(tok.pre(), "deepseek-v3", "pre-tokenizer family detection");
        let tmpl = dsv4_sentinel();
        let req_tools = dsv4_req_tools(Some(&serde_json::json!([
            {"type": "function", "function": {
                "name": "get_data",
                "description": "Fetch a blob",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                               "required": ["key"]}
            }}
        ])));

        let cases: Vec<(&str, String)> = vec![
            ("ascii-letter-131k", "Z".repeat(131_072)), // the issue's exact reproducer
            ("ascii-letter-1m", "Z".repeat(1_048_576)),
            ("space-131k", " ".repeat(131_072)),
            ("digit-131k", "7".repeat(131_072)),
            (
                "mixed-runs",
                format!(
                    "{}{}{}{}",
                    "Z".repeat(65_536),
                    " ".repeat(65_536),
                    "7".repeat(65_536),
                    "\n".repeat(65_536)
                ),
            ),
            ("cjk-64k", "中".repeat(65_536)),
            ("accented-letter-64k", "é".repeat(65_536)),
        ];
        for (name, blob) in &cases {
            let msgs = serde_json::json!([
                {"role": "system", "content": "You are a tool-using assistant."},
                {"role": "user", "content": "Fetch the blob."},
                {"role": "assistant", "reasoning": "Use get_data.", "content": "",
                 "tool_calls": [{"id": "call_001", "type": "function",
                                 "function": {"name": "get_data",
                                              "arguments": "{\"key\": \"blob\"}"}}]},
                {"role": "tool", "tool_call_id": "call_001", "content": blob}
            ]);
            let turns: Vec<TmplTurn> = msgs.as_array().unwrap().iter().map(dsv4_turn).collect();
            let rendered = chat::apply_chat_template_tools_ex(
                Some(&tmpl),
                &turns,
                true,
                &[],
                &req_tools,
                ThinkMode::Think,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{name}: render failed: {e}"));
            assert!(
                rendered.contains(blob.as_str()),
                "{name}: tool result missing from render"
            );
            let t0 = std::time::Instant::now();
            let ids = tok.encode(&rendered, true);
            let encode_dt = t0.elapsed();
            assert!(!ids.is_empty(), "{name}: empty encode");
            let back = tok.decode(&ids);
            assert_eq!(back, rendered, "{name}: decode(encode(x)) != x");
            // linear-ish, not the quadratic/backtracking blowup: debug builds land in
            // single-digit seconds even for the 1M case; 60s catches a blowup without
            // flaking a loaded box.
            assert!(
                encode_dt < std::time::Duration::from_secs(60),
                "{name}: encode took {encode_dt:?}"
            );
            // receipts-time HF cross-check bridge: dump rendered bytes + memra ids for the
            // 131k reproducer so a scratch `tokenizers` venv can verify id parity
            // (research/dsv4-template-20260818/RECEIPTS.md, long-run hardening section).
            if *name == "ascii-letter-131k"
                && let Ok(dir) = std::env::var("DSV4_LONGRUN_DUMP_DIR")
            {
                std::fs::write(format!("{dir}/rendered-131k.txt"), &rendered).unwrap();
                let csv: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                std::fs::write(format!("{dir}/memra-ids-131k.csv"), csv.join(",")).unwrap();
            }
        }
    }

    #[test]
    fn models_v1_entry_advertises_thinking_support() {
        // Thinking model (step35 dialect: effort_levels): reasoning must be discoverable
        // from the contract-v2 capability booleans.
        let step_caps = ModelCaps {
            effort_levels: true,
            ..tool_caps()
        };
        let entry = model_entry_v1("stepfun/step-3.7-flash", Some(&step_caps), None);
        assert_eq!(entry["capabilities"]["reasoning"], true);
        assert_eq!(entry["capabilities"]["tools"], true);

        // Non-thinking, non-tools model: neither capability may be advertised.
        let plain = ModelCaps {
            chat_ok: true,
            ..Default::default()
        };
        let entry = model_entry_v1("plain", Some(&plain), None);
        assert_eq!(entry["capabilities"]["reasoning"], false);
        assert_eq!(entry["capabilities"]["tools"], false);
        // Caps-unknown model: honest falses, streaming always true.
        let entry = model_entry_v1("unknown", None, None);
        assert_eq!(entry["capabilities"]["reasoning"], false);
        assert_eq!(entry["capabilities"]["streaming"], true);
    }

    #[test]
    fn chat_request_preserves_turns_and_openai_stop_forms() {
        let payload = serde_json::json!({
            "model": "plain_quant",
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "developer", "content": "dev rules"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "work"}
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "stop": "<stop>"
        });
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, None, tx, lanes::Lane::Interactive, None).unwrap();
        let request = plan.request;
        assert!(
            plan.parser.is_none(),
            "no tools -> no parser (isolation contract)"
        );
        assert!(request.tools_json.is_empty());
        assert_eq!(request.think, ThinkMode::Default);
        assert_eq!(request.model, "plain_quant");
        assert_eq!(request.params.max_new, 64);
        // OMITTED max_tokens (gap-scan F2): the context-bounded sentinel, not 128.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}]
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, None, tx, lanes::Lane::Interactive, None).unwrap();
        assert_eq!(plan.request.params.max_new, worker::MAX_NEW_CTX_BOUNDED);
        // max_completion_tokens alias still honored exactly.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "max_completion_tokens": 7
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .params
                .max_new,
            7
        );
        // completions body: same omission law.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "prompt": "task"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_request(&req, tx, lanes::Lane::Interactive, None)
                .params
                .max_new,
            worker::MAX_NEW_CTX_BOUNDED
        );
        let turns: Vec<(String, String)> = request
            .chat_turns
            .iter()
            .map(|t| (t.role.clone(), t.content.clone()))
            .collect();
        assert_eq!(
            turns,
            vec![
                ("system".into(), "rules".into()),
                ("system".into(), "dev rules".into()), // developer -> system normalization
                ("user".into(), "task".into()),
                ("assistant".into(), "work".into()),
            ]
        );
        assert!(request.chat_turns.iter().all(|t| t.tool_calls.is_empty()));
        assert_eq!(request.stop_strings, vec!["<stop>"]);

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": ["a", "b"]
        }))
        .unwrap();
        assert_eq!(req.stop.into_vec(), vec!["a", "b"]);

        // TOOTH (hermes finding, fixed 2026-08-23): an empty stop element matches every
        // decode ("".contains == always true; find("") == Some(0) truncated the whole
        // completion). Empties drop at ingestion; real elements survive.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": ["", "real", ""]
        }))
        .unwrap();
        assert_eq!(req.stop.into_vec(), vec!["real"]);
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": ""
        }))
        .unwrap();
        assert!(req.stop.into_vec().is_empty());

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "plain_quant", "messages": [{"role": "user", "content": "task"}],
            "stop": null
        }))
        .unwrap();
        assert!(req.stop.into_vec().is_empty());
    }

    #[test]
    fn stop_sequence_limits_bound_count_individual_and_aggregate_work() {
        let at_limit = StopSequences::Many(vec!["x".repeat(256); MAX_STOP_SEQUENCES]);
        assert!(at_limit.validate().is_ok());
        assert!(
            StopSequences::Many(vec![String::new(); MAX_STOP_SEQUENCES + 1])
                .validate()
                .unwrap_err()
                .contains("at most")
        );
        assert!(
            StopSequences::One("x".repeat(MAX_STOP_SEQUENCE_BYTES + 1))
                .validate()
                .unwrap_err()
                .contains("each stop")
        );
        assert!(
            StopSequences::Many(vec!["x".repeat(300); MAX_STOP_SEQUENCES])
                .validate()
                .unwrap_err()
                .contains("total at most")
        );
    }

    #[tokio::test]
    async fn chat_response_has_openai_message_shape() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "hello".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 1,
            n_prompt: 42,
            n_cached: 30,
            elapsed_s: 0.5,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let response = blocking_response(
            rx,
            "plain_quant".into(),
            true,
            Vec::new(),
            None,
            Envelope::new(true),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["object"], "chat.completion");
        // OpenAI envelope (gap-scan F1): the official SDK pydantic-REQUIRES id + created.
        assert!(payload["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert!(payload["created"].as_u64().unwrap() > 1_700_000_000);
        // Shape, not prefix: `starts_with("memra-")` is what this line used to assert, and
        // `memra-unknown` passes that, which is how a meaningless fingerprint sat inside a
        // tested surface all the way to prod.
        let fingerprint = payload["system_fingerprint"].as_str().unwrap();
        assert!(
            build_id::fingerprint_is_well_formed(fingerprint),
            "system_fingerprint {fingerprint:?} is not memra-<version>-<12 hex>"
        );
        assert_eq!(payload["choices"][0]["message"]["role"], "assistant");
        assert_eq!(payload["choices"][0]["message"]["content"], "hello");
        assert_eq!(payload["choices"][0]["finish_reason"], "stop");
        // OpenAI prompt-caching usage schema (worker-truth cached vs computed split).
        assert_eq!(payload["usage"]["prompt_tokens"], 42);
        assert_eq!(payload["usage"]["completion_tokens"], 1);
        assert_eq!(payload["usage"]["total_tokens"], 43);
        assert_eq!(
            payload["usage"]["prompt_tokens_details"]["cached_tokens"],
            30
        );
        // ADDITIVE contract (lane/accept-telemetry): a non-spec request carries NO usage.spec
        // — the pre-lane usage object byte-for-byte.
        assert!(payload["usage"].get("spec").is_none());
    }

    #[tokio::test]
    async fn native_response_uses_terminal_token_snapshot_for_coalesced_events() {
        let (tx, rx) = worker::event_channel();
        // A speculative round may commit four ids but expose one detokenized text delta.
        tx.send(Event::Token {
            id: 4,
            text: "hello".into(),
        })
        .unwrap();
        tx.send(Event::TokenSnapshot(vec![1, 2, 3, 4])).unwrap();
        tx.send(Event::Done {
            stop_reason: "MaxNew".into(),
            n_tokens: 4,
            n_prompt: 2,
            n_cached: 0,
            elapsed_s: 0.5,
            spec: None,
        })
        .unwrap();
        drop(tx);

        let response = blocking_response(
            rx,
            "plain_quant".into(),
            false,
            Vec::new(),
            None,
            Envelope::new(false),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["text"], "hello");
        assert_eq!(payload["tokens"], serde_json::json!([1, 2, 3, 4]));
        assert_eq!(payload["n_tokens"], 4);
    }

    /// usage.spec (lane/accept-telemetry): spec-decode requests carry this request's own
    /// acceptance summary as an additive usage extension; every existing field is untouched.
    #[tokio::test]
    async fn chat_usage_carries_spec_acceptance_summary() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "hello".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 1,
            n_prompt: 42,
            n_cached: 0,
            elapsed_s: 0.5,
            spec: Some(worker::SpecUsage {
                rounds: 10,
                drafted: 30,
                accepted: 21,
            }),
        })
        .unwrap();
        drop(tx);
        let response = blocking_response(
            rx,
            "plain_quant".into(),
            true,
            Vec::new(),
            None,
            Envelope::new(true),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let sp = &payload["usage"]["spec"];
        assert_eq!(sp["rounds"], 10);
        assert_eq!(sp["drafted"], 30);
        assert_eq!(sp["accepted"], 21);
        assert!((sp["acceptance_rate"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        // existing fields untouched next to the extension.
        assert_eq!(payload["usage"]["total_tokens"], 43);
    }

    fn weather_request(extra: serde_json::Value) -> ChatCompletionReq {
        let mut payload = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object",
                               "properties": {"city": {"type": "string"},
                                              "days": {"type": "integer"}},
                               "required": ["city"]}}}],
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                payload[k] = v.clone();
            }
        }
        serde_json::from_value(payload).unwrap()
    }

    /// glm5 twin of `vision_decode_is_deferred_and_grid_pinned`: the placeholder run is
    /// rendered from the header-planned grid; the decoded grid must equal it, and a
    /// mismatch refuses instead of desyncing runs from units (lane/glm5-vision).
    #[test]
    fn glm5_vision_decode_is_deferred_and_grid_pinned() {
        let (tx, _rx) = worker::event_channel();
        let req: ChatCompletionReq = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let mut plan = build_chat_request(
            req,
            Some(&ModelCaps {
                chat_ok: true,
                ..Default::default()
            }),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        // 112x112 BMP: identity smart_resize (28-aligned, inside the 16..3072 budget) ->
        // grid 8x8 patches, 16 merged tokens (the det112 fixture geometry).
        let bmp = |w: u32, h: u32| -> Vec<u8> {
            let row = (w * 3).div_ceil(4) * 4;
            let size = 54 + row * h;
            let mut b = vec![0x42u8, 0x4d];
            b.extend_from_slice(&size.to_le_bytes());
            b.extend_from_slice(&[0; 4]);
            b.extend_from_slice(&54u32.to_le_bytes());
            b.extend_from_slice(&40u32.to_le_bytes());
            b.extend_from_slice(&w.to_le_bytes());
            b.extend_from_slice(&h.to_le_bytes());
            b.extend_from_slice(&1u16.to_le_bytes());
            b.extend_from_slice(&24u16.to_le_bytes());
            b.extend_from_slice(&[0u8; 24]);
            b.extend(std::iter::repeat_n(0x7fu8, (row * h) as usize));
            b
        };
        let bytes = bmp(112, 112);
        let (gh, gw) = memra_engine::vision_glm5::glm5_plan_image(&bytes).unwrap();
        assert_eq!((gh, gw), (8, 8), "identity resize grid");
        assert_eq!(memra_engine::vision_glm5::n_merged_for_grid(gh, gw), 16);
        plan.pending_glm5.push(PendingGlm5Image {
            bytes: bytes.clone(),
            gh,
            gw,
        });
        decode_pending_vision(&mut plan).unwrap();
        assert_eq!(plan.request.glm5_images.len(), 1);
        let unit = &plan.request.glm5_images[0];
        assert_eq!((unit.gh, unit.gw), (gh, gw));
        assert_eq!(
            unit.patches.len(),
            gh * gw * memra_engine::vision_glm5::G5V_PATCH_IN
        );
        // A grid mismatch refuses instead of desyncing placeholder runs from units.
        plan.request.glm5_images.clear();
        plan.pending_glm5.push(PendingGlm5Image {
            bytes,
            gh: gh + 2,
            gw,
        });
        let err = decode_pending_vision(&mut plan).unwrap_err();
        assert!(err.contains("header-planned"), "got: {err}");
    }

    #[test]
    fn vision_decode_is_deferred_and_grid_pinned() {
        // TOOTH (hermes decode-bomb findings, fixed 2026-08-23): the plan phase renders
        // pad runs from HEADER dims only; canvases expand in decode_pending_vision,
        // which runs after admit_tenant_budget in chat_completions/admit_translated.
        // Build a plain plan, then drive phase 2 directly.
        let (tx, _rx) = worker::event_channel();
        let req: ChatCompletionReq = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let mut plan = build_chat_request(
            req,
            Some(&ModelCaps {
                chat_ok: true,
                ..Default::default()
            }),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        // A planned still decodes into request.images when its grid matches the plan.
        // Hand-built 64x64 24bpp BMP (no image-crate dep in this crate): 54-byte header
        // + 64*64*3 pixel bytes (row stride 192 is 4-aligned, no padding).
        let bmp = |w: i32, h: i32, with_pixels: bool| -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(b"BM");
            b.extend_from_slice(&54u32.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&54u32.to_le_bytes());
            b.extend_from_slice(&40u32.to_le_bytes());
            b.extend_from_slice(&w.to_le_bytes());
            b.extend_from_slice(&h.to_le_bytes());
            b.extend_from_slice(&1u16.to_le_bytes());
            b.extend_from_slice(&24u16.to_le_bytes());
            b.extend_from_slice(&[0u8; 24]);
            if with_pixels {
                b.extend(std::iter::repeat_n(0x7fu8, (w * h * 3) as usize));
            }
            b
        };
        let bytes = bmp(64, 64, true);
        let (gh, gw) = memra_engine::vision_pre::plan_image_bytes(&bytes).unwrap();
        plan.pending_images.push(PendingVisionUnit::Still {
            bytes: bytes.clone(),
            gh,
            gw,
        });
        decode_pending_vision(&mut plan).unwrap();
        assert_eq!(plan.request.images.len(), 1);
        assert_eq!(
            (
                plan.request.images[0].prep.gh,
                plan.request.images[0].prep.gw
            ),
            (gh, gw),
            "decoded grid must equal the header-planned grid the pad run was rendered from"
        );
        // A grid mismatch refuses instead of desyncing pad runs from units.
        plan.request.images.clear();
        plan.pending_images.push(PendingVisionUnit::Still {
            bytes,
            gh: gh + 2,
            gw,
        });
        let err = decode_pending_vision(&mut plan).unwrap_err();
        assert!(err.contains("header-planned"), "got: {err}");
        // Defense in depth: even if a bomb reached phase 2, the decode re-admits the
        // header budget and refuses pre-decode with the named error.
        let bomb = bmp(16_000, 16_000, false);
        plan.pending_images.clear();
        plan.pending_images.push(PendingVisionUnit::Still {
            bytes: bomb,
            gh: 2,
            gw: 2,
        });
        let err = decode_pending_vision(&mut plan).unwrap_err();
        assert!(err.contains("exceeds the decode budget"), "got: {err}");
    }

    #[test]
    fn tools_request_renders_client_key_order_and_arms_parser() {
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(
            weather_request(json!({})),
            Some(&tool_caps()),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        assert!(plan.parser.is_some());
        assert_eq!(plan.request.tools_json.len(), 1);
        // client key order preserved + python-dumps separators (the template's tojson law).
        assert_eq!(
            plan.request.tools_json[0],
            "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \
             \"description\": \"Get current weather\", \"parameters\": {\"type\": \"object\", \
             \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \
             \"integer\"}}, \"required\": [\"city\"]}}}"
        );
    }

    #[test]
    fn hy3_tools_and_reasoning_flow_through_the_real_chat_plan() {
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(
            weather_request(json!({"reasoning_effort": "high"})),
            Some(&hy3_tool_caps()),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        assert_eq!(plan.request.think, ThinkMode::Think);
        assert_eq!(plan.request.reasoning_effort.as_deref(), Some("high"));
        assert!(
            plan.request
                .stop_strings
                .iter()
                .any(|stop| stop == "</tool_calls:opensource>")
        );
        let rendered = chat::apply_chat_template_tools_ex(
            Some("... hy_User ... <tools> ..."),
            &plan.request.chat_turns,
            true,
            &plan.request.tools_json,
            &plan.request.tools_struct,
            plan.request.think,
            plan.request.reasoning_effort.as_deref(),
            None,
        )
        .unwrap();
        assert!(rendered.contains("<tool_calls:opensource>"));
        assert!(rendered.ends_with("<think:opensource>"));

        let mut parser = plan.parser.expect("HY3 tools arm its native parser");
        let pieces = parser.push(concat!(
            "Need weather.</think:opensource>",
            "<tool_calls:opensource><tool_call:opensource>get_weather",
            "<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n",
            "<arg_value:opensource>Paris</arg_value:opensource>\n",
            "</tool_call:opensource></tool_calls:opensource>",
        ));
        assert!(pieces.contains(&Piece::Reasoning("Need weather.".into())));
        assert!(pieces.iter().any(|piece| matches!(piece, Piece::Call(call)
            if call.name == "get_weather" && call.arguments == r#"{"city":"Paris"}"#)));
    }

    #[test]
    fn tool_choice_none_strips_tools_and_parser() {
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(
            weather_request(json!({"tool_choice": "none"})),
            Some(&tool_caps()),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        // tools stripped: no tool-call scanning; the think-open prompt still arms the
        // reasoning-only splitter (F13) — a <tool_call> in post-think prose stays prose.
        let mut p = plan
            .parser
            .expect("think-open chat arms the reasoning splitter");
        let pieces = p.push("x</think>\n\n<tool_call> stays prose");
        assert_eq!(
            pieces,
            vec![
                Piece::Reasoning("x".into()),
                Piece::Content("<tool_call> stays prose".into()),
            ]
        );
        assert!(plan.request.tools_json.is_empty());
        // unsupported tool_choice forms are clean 400s, not silent downgrades.
        let (tx, _rx) = worker::event_channel();
        assert!(
            build_chat_request(
                weather_request(json!({"tool_choice": "required"})),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None
            )
            .is_err()
        );
        let (tx, _rx) = worker::event_channel();
        assert!(
            build_chat_request(
                weather_request(json!({"tool_choice":
            {"type": "function", "function": {"name": "get_weather"}}})),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn model_plan_accepts_st_dir_and_rejects_bogus_dir() {
        let root = std::env::temp_dir().join(format!("memra_plan_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);

        // (a) single-file ST checkpoint dir: config.json + model.safetensors.
        let st = root.join("st_single");
        std::fs::create_dir_all(&st).unwrap();
        std::fs::write(st.join("config.json"), "{}").unwrap();
        std::fs::write(st.join("model.safetensors"), b"x").unwrap();
        assert!(validate_model_path(st.to_str().unwrap()).is_ok());

        // (b) sharded ST checkpoint dir: config.json + model.safetensors.index.json.
        let sh = root.join("st_sharded");
        std::fs::create_dir_all(&sh).unwrap();
        std::fs::write(sh.join("config.json"), "{}").unwrap();
        std::fs::write(sh.join("model.safetensors.index.json"), "{}").unwrap();
        assert!(validate_model_path(sh.to_str().unwrap()).is_ok());

        // (c) repack dir: manifest.json alone qualifies.
        let rp = root.join("repack");
        std::fs::create_dir_all(&rp).unwrap();
        std::fs::write(rp.join("manifest.json"), "{}").unwrap();
        assert!(validate_model_path(rp.to_str().unwrap()).is_ok());

        // (d) bogus dir (no weights): clear error naming what was expected.
        let bogus = root.join("bogus");
        std::fs::create_dir_all(&bogus).unwrap();
        let err = validate_model_path(bogus.to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("model.safetensors"),
            "error should say what is missing: {err}"
        );
        assert!(
            err.contains("manifest.json"),
            "error should mention the repack form: {err}"
        );

        // (e) ST weights but no config.json: distinct clear error.
        let nc = root.join("no_config");
        std::fs::create_dir_all(&nc).unwrap();
        std::fs::write(nc.join("model.safetensors"), b"x").unwrap();
        let err = validate_model_path(nc.to_str().unwrap()).unwrap_err();
        assert!(
            err.contains("config.json"),
            "error should name config.json: {err}"
        );

        // (f) nonexistent path.
        let err = validate_model_path(root.join("nowhere").to_str().unwrap()).unwrap_err();
        assert!(err.contains("does not exist"), "{err}");

        // (g) plain file = GGUF branch, accepted as-is.
        let f = root.join("model.gguf");
        std::fs::write(&f, b"g").unwrap();
        assert!(validate_model_path(f.to_str().unwrap()).is_ok());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn chat_on_templateless_dir_checkpoint_is_rejected_with_clear_message() {
        // serve-st v1 honesty gate: a dir checkpoint whose tokenizer carries no chat
        // template probes chat_ok=false -> every chat request 400s BEFORE the worker.
        let caps = ModelCaps {
            tools_branch: false,
            qwen_think: false,
            think_switch: false,
            chat_ok: false,
            ..Default::default()
        };
        let payload = serde_json::json!({
            "model": "st_model",
            "messages": [{"role": "user", "content": "hello"}],
        });
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = worker::event_channel();
        let err = match build_chat_request(req, Some(&caps), tx, lanes::Lane::Interactive, None) {
            Err(e) => e,
            Ok(_) => panic!("templateless dir checkpoint must reject chat"),
        };
        assert!(
            err.contains("no chat template"),
            "message should name the cause: {err}"
        );
        assert!(
            err.contains("/v1/completions"),
            "message should point at the raw-prompt escape hatch: {err}"
        );
    }

    #[test]
    fn tools_on_model_without_tools_branch_is_rejected() {
        let (tx, _rx) = worker::event_channel();
        let caps = ModelCaps {
            chat_ok: true,
            ..Default::default()
        };
        assert!(
            build_chat_request(
                weather_request(json!({})),
                Some(&caps),
                tx,
                lanes::Lane::Interactive,
                None
            )
            .is_err()
        );
        let (tx, _rx) = worker::event_channel();
        assert!(
            build_chat_request(
                weather_request(json!({})),
                None,
                tx,
                lanes::Lane::Interactive,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn reasoning_effort_maps_to_think_switch() {
        // The reasoning-capable-model convention (owner directive 2026-08-07):
        // low|medium|high = thinking ON at that budget; none|minimal = thinking OFF;
        // absent = the model's own default. `low` used to map to NoThink — that read the
        // OpenAI field as a "how much" dial with off at the bottom, which contradicts how
        // reasoning models ship (low IS a reasoning mode).
        for (extra, want) in [
            (json!({}), ThinkMode::Default),
            (json!({"reasoning_effort": "low"}), ThinkMode::Think),
            (json!({"reasoning_effort": "none"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "minimal"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "high"}), ThinkMode::Think),
            (json!({"reasoning_effort": "medium"}), ThinkMode::Think),
            (json!({"reasoning": {"enabled": false}}), ThinkMode::NoThink),
            (json!({"reasoning": {"effort": "low"}}), ThinkMode::Think),
            (json!({"reasoning": {"enabled": true}}), ThinkMode::Think),
            // Clamp aliases (issue #31): levels above "high" mean thinking ON at the
            // highest level any loaded template distinguishes. Real default-config
            // clients send these (codex xhigh; Claude Code xhigh via /v1/messages).
            (json!({"reasoning_effort": "xhigh"}), ThinkMode::Think),
            (json!({"reasoning_effort": "max"}), ThinkMode::Think),
            (json!({"reasoning_effort": "ultra"}), ThinkMode::Think),
            // Explicit-switch precedence (issue #31): enabled/disabled — the field
            // Anthropic thinking.type translates onto — wins over the switch the
            // effort level implies.
            (
                json!({"reasoning": {"enabled": true, "effort": "none"}}),
                ThinkMode::Think,
            ),
            (
                json!({"reasoning": {"enabled": false, "effort": "high"}}),
                ThinkMode::NoThink,
            ),
        ] {
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                weather_request(extra.clone()),
                // A LADDER-carrying model (qwen3.8 shape), so every rung of the table is
                // exercised as a real render input here. On a model with no depth input the
                // same rungs TRANSLATE onto the binary axis as reasoning ON — that mapping has
                // its own test (`a_graded_level_on_a_binary_model_translates_to_reasoning_on`).
                Some(&ladder_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .unwrap();
            assert_eq!(plan.request.think, want, "extra={extra}");
        }
        // An out-of-table value is a 400 on EVERY expression of the field — including
        // next to an explicit switch (the old enabled==false early-return skipped
        // validation, the same silent-accept class /v1/messages had in issue #31).
        for extra in [
            json!({"reasoning_effort": "extreme"}),
            json!({"reasoning": {"effort": "banana"}}),
            json!({"reasoning": {"enabled": false, "effort": "banana"}}),
            json!({"reasoning": {"enabled": true, "effort": ""}}),
        ] {
            let (tx, _rx) = worker::event_channel();
            assert!(
                build_chat_request(
                    weather_request(extra.clone()),
                    Some(&tool_caps()),
                    tx,
                    lanes::Lane::Interactive,
                    None
                )
                .is_err(),
                "extra={extra} must be rejected by the one allowlist"
            );
        }
        // The clamp really lands on "high" for level-consuming templates, and the
        // whole canonical table is what `canonical_effort` says it is.
        for (raw, want) in [
            ("none", Some("none")),
            ("minimal", Some("minimal")),
            ("low", Some("low")),
            ("medium", Some("medium")),
            ("high", Some("high")),
            ("xhigh", Some("high")),
            ("max", Some("high")),
            ("ultra", Some("high")),
            ("banana", None),
            ("", None),
            ("HIGH", None),
        ] {
            assert_eq!(canonical_effort(raw), want, "canonical_effort({raw:?})");
        }
        // dsv4 exemption (hermes 2026-08-23): the one template with a rung above "high"
        // gets the above-high aliases as "max"; the rest of the table is identical.
        for (raw, want) in [
            ("none", Some("none")),
            ("minimal", Some("minimal")),
            ("low", Some("low")),
            ("medium", Some("medium")),
            ("high", Some("high")),
            ("xhigh", Some("max")),
            ("max", Some("max")),
            ("ultra", Some("max")),
            ("banana", None),
            ("", None),
            ("MAX", None),
        ] {
            assert_eq!(
                canonical_effort_for(raw, true),
                want,
                "canonical_effort_for({raw:?}, dsv4)"
            );
        }
    }

    #[test]
    fn dsv4_reasoning_effort_max_survives_canonicalization() {
        // TOOTH (hermes finding e98463…/parse_think-collapse, fixed 2026-08-23): dsv4's
        // 0731 encoding renders DIFFERENT prompt prefixes for "high" (ABSOLUTE_MAX) and
        // "max" (BEYOND_MAX) — collapsing max->high at the server silently discarded the
        // top tier. A dsv4-caps plan must carry "max" through to the renderer; every
        // non-dsv4 template still clamps to "high".
        let dsv4_caps = ModelCaps {
            chat_ok: true,
            dsv4: true,
            ..Default::default()
        };
        let build = |caps: &ModelCaps, effort: &str| {
            let (tx, _rx) = worker::event_channel();
            let req: ChatCompletionReq = serde_json::from_value(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": effort,
            }))
            .unwrap();
            build_chat_request(req, Some(caps), tx, lanes::Lane::Interactive, None)
        };
        for raw in ["max", "xhigh", "ultra"] {
            let plan = build(&dsv4_caps, raw).unwrap();
            assert_eq!(
                plan.request.reasoning_effort.as_deref(),
                Some("max"),
                "dsv4 {raw:?} must reach the renderer as the max rung"
            );
            assert_eq!(plan.request.think, chat::ThinkMode::Think);
        }
        // "high" stays "high" on dsv4 (a distinct rung, not an alias).
        let plan = build(&dsv4_caps, "high").unwrap();
        assert_eq!(plan.request.reasoning_effort.as_deref(), Some("high"));
        // Non-dsv4 level-consuming template: above-high still clamps to "high".
        let step_caps = ModelCaps {
            chat_ok: true,
            effort_levels: true,
            ..Default::default()
        };
        let plan = build(&step_caps, "max").unwrap();
        assert_eq!(plan.request.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn default_reasoning_effort_flips_only_the_unset_request() {
        // Owner ruling 2026-08-19 (darklanes gemma GPQA recovery board, step 2): gemma-4
        // serves think-ON by default — 80.81 GPQA think-on vs 76.26 think-off on the
        // served mint. Mechanism: a per-model MEMRA_MODEL_METADATA knob
        // (`default_reasoning_effort`) resolved at plan build. ONLY a request that
        // expressed no reasoning preference flips; every explicit client choice is
        // honored unchanged.
        let build = |extra: serde_json::Value, default_effort: Option<&str>| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request_with_trace(
                weather_request(extra),
                Some(&ladder_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
                None,
                default_effort,
                &ModelSamplingDefaults::default(),
            )
            .unwrap()
        };
        for (extra, want) in [
            // the ONE case the knob owns: nothing expressed on either surface.
            (json!({}), ThinkMode::Think),
            // `reasoning.exclude:true` is no longer "unset" and no longer a display flag: it
            // is an OFF-switch (owner ruling 2026-08-23 — not delivering reasoning means not
            // generating it), so it beats the operator default exactly like reasoning.enabled.
            (json!({"reasoning": {"exclude": true}}), ThinkMode::NoThink),
            (json!({"include_reasoning": false}), ThinkMode::NoThink),
            // ...and the "deliver it" direction expresses no switch, so the default still wins.
            (json!({"reasoning": {"exclude": false}}), ThinkMode::Think),
            (json!({"include_reasoning": true}), ThinkMode::Think),
            // explicit OFF stays off, on both surfaces.
            (json!({"reasoning_effort": "none"}), ThinkMode::NoThink),
            (json!({"reasoning_effort": "minimal"}), ThinkMode::NoThink),
            (json!({"reasoning": {"enabled": false}}), ThinkMode::NoThink),
            // explicit ON stays exactly the client's request.
            (json!({"reasoning_effort": "low"}), ThinkMode::Think),
            (json!({"reasoning_effort": "high"}), ThinkMode::Think),
            (json!({"reasoning": {"enabled": true}}), ThinkMode::Think),
        ] {
            let plan = build(extra.clone(), Some("high"));
            assert_eq!(plan.request.think, want, "extra={extra}");
        }
        // the knob can also pin thinking OFF by default; explicit ON still wins over it.
        assert_eq!(
            build(json!({}), Some("none")).request.think,
            ThinkMode::NoThink
        );
        assert_eq!(
            build(json!({"reasoning_effort": "high"}), Some("none"))
                .request
                .think,
            ThinkMode::Think
        );
        // no knob (every model without a metadata entry — qwen etc.): unset stays the
        // template's own default. Together with `reasoning_effort_maps_to_think_switch`
        // above, this is the byte-identical regression guard for knobless deployments.
        assert_eq!(build(json!({}), None).request.think, ThinkMode::Default);
    }

    /// A qwen-class template that carries all three markers the renderer keys on:
    /// `<think>` + `add_generation_prompt` (think tail), `enable_thinking` (the switch),
    /// `<tools>` (tools branch). Shape-equivalent to the deployed q38 / ornith15 GGUF
    /// templates, whose live `think_switch=true` is receipted in darklanes
    /// research/reasoning-control-20260823/THINKING.md.
    const SWITCHED_QWEN_TMPL: &str = "<tools> ... add_generation_prompt ... \
         {%- if enable_thinking is defined and enable_thinking is false %}'<think>\\n\\n</think>\\n\\n'\
         {%- else %}'<think>\\n'{%- endif %}";

    #[test]
    fn vllm_enable_thinking_switch_is_wired_not_ignored() {
        // THE DEFECT THIS CLOSES (lane/reasoning-control-20260823): `ChatCompletionReq` has
        // no `deny_unknown_fields`, so the whole vLLM-shaped ecosystem's thinking switch —
        // top-level `enable_thinking` and `chat_template_kwargs.enable_thinking` — was
        // deserialized away and the request served with reasoning ON behind a 200. Measured
        // on the live endpoint against both served models before the fix.
        let build = |extra: serde_json::Value| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request(
                weather_request(extra),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
        };
        for (extra, want) in [
            (json!({"enable_thinking": false}), ThinkMode::NoThink),
            (json!({"enable_thinking": true}), ThinkMode::Think),
            (
                json!({"chat_template_kwargs": {"enable_thinking": false}}),
                ThinkMode::NoThink,
            ),
            (
                json!({"chat_template_kwargs": {"enable_thinking": true}}),
                ThinkMode::Think,
            ),
            // the vLLM switch is an EXPLICIT switch, so it beats the switch an effort level
            // implies — the same precedence `reasoning.enabled` already had (issue #31).
            (
                json!({"enable_thinking": false, "reasoning_effort": "high"}),
                ThinkMode::NoThink,
            ),
            // agreement between the two spellings is fine.
            (
                json!({"enable_thinking": false,
                       "chat_template_kwargs": {"enable_thinking": false}}),
                ThinkMode::NoThink,
            ),
        ] {
            let plan = build(extra.clone()).unwrap_or_else(|e| {
                panic!("{extra} must be accepted and honored, got 400: {e}");
            });
            assert_eq!(
                plan.request.think, want,
                "{extra} was ACCEPTED AND IGNORED — the banned silent-accept class"
            );
        }
        // and it reaches the PROMPT BYTES, not just the plan: the closed think pair is what
        // the template's `enable_thinking is false` branch emits.
        let render = |extra: serde_json::Value| -> String {
            let plan = build(extra).unwrap();
            chat::apply_chat_template_tools_ex(
                Some(SWITCHED_QWEN_TMPL),
                &plan.request.chat_turns,
                true,
                &plan.request.tools_json,
                &plan.request.tools_struct,
                plan.request.think,
                plan.request.reasoning_effort.as_deref(),
                None,
            )
            .unwrap()
        };
        let off = render(json!({"enable_thinking": false}));
        assert!(
            off.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "enable_thinking:false must render the CLOSED think pair: {off:?}"
        );
        let on = render(json!({}));
        assert!(
            on.ends_with("<|im_start|>assistant\n<think>\n"),
            "an unset request must still render the template's OPEN think tail: {on:?}"
        );
        assert_eq!(
            off,
            render(json!({"chat_template_kwargs": {"enable_thinking": false}})),
            "both vLLM spellings must render byte-identically"
        );
        assert_eq!(
            off,
            render(json!({"reasoning_effort": "none"})),
            "the vLLM spelling must render byte-identically to the OpenAI spelling"
        );
    }

    #[test]
    fn unknown_chat_template_kwarg_refuses_by_name() {
        // This renderer is Rust, not jinja: a kwarg it does not implement changes nothing
        // about the prompt, so accepting it with 200 is the same defect one level down.
        let build = |extra: serde_json::Value| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request(
                weather_request(extra),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
        };
        let refusal = |extra: serde_json::Value, why: &str| -> String {
            build(extra).err().unwrap_or_else(|| panic!("{why}"))
        };
        let err = refusal(
            json!({"chat_template_kwargs": {"add_generation_prompt": false}}),
            "an unimplementable template kwarg must not be accepted",
        );
        assert!(
            err.contains("add_generation_prompt") && err.contains("enable_thinking"),
            "the refusal must name the offending key AND the supported one: {err}"
        );
        let err = refusal(
            json!({"chat_template_kwargs": "enable_thinking=false"}),
            "a non-object chat_template_kwargs must not be accepted",
        );
        assert!(
            err.contains("must be an object"),
            "refusal must say what shape is expected: {err}"
        );
        let err = refusal(
            json!({"chat_template_kwargs": {"enable_thinking": "false"}}),
            "a stringly-typed switch must not be accepted",
        );
        assert!(
            err.contains("true or false"),
            "refusal must name the expected type: {err}"
        );
        // an explicitly-null kwargs bag is "nothing expressed", not an error.
        let plan = build(json!({"chat_template_kwargs": null}))
            .expect("null chat_template_kwargs is the unset case");
        assert_eq!(plan.request.think, ThinkMode::Default);
    }

    // ============ THE ONE REASONING SCHEMA (lane/reasoning-schema-20260823) ===============
    //
    // Owner rulings this section enforces, in their order of severity:
    //   1. a reasoning parameter that returns 200 must have an EFFECT — measured on prompt bytes;
    //   2. every surface spelling maps into ONE internal schema, identically on all three APIs;
    //   3. asking for non-reasoning and getting reasoning is impossible — off is a real
    //      generation decision, and where it cannot be honoured it is a named 400;
    //   4. reasoning is compute and output, so it is never withheld after being billed.
    //
    // The lab is the authority on each model's controls (never inferred from lineage or a shared
    // loader): Qwen/Qwen3.8-27B's card documents `reasoning_effort` = xhigh (default) | medium |
    // low; Ornith AI documents `enable_thinking` and nothing else.

    /// The DEPLOYED qwen3.8 template, byte-identical in the BF16 and NVFP4-Q5K mints.
    const Q38_TMPL: &str =
        include_str!("../../../research/reasoning-schema-20260823/qwen38-27b.chat_template.jinja");

    /// Build a plan and render it through the template the caps describe — the only assertion
    /// that cannot lie about whether a parameter had an effect.
    fn render_with(
        tmpl: &str,
        caps: &ModelCaps,
        extra: serde_json::Value,
        default_effort: Option<&str>,
    ) -> Result<String, String> {
        let mut payload = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                payload[k] = v.clone();
            }
        }
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request_with_trace(
            req,
            Some(caps),
            tx,
            lanes::Lane::Interactive,
            None,
            None,
            default_effort,
            &ModelSamplingDefaults::default(),
        )?;
        Ok(chat::apply_chat_template_tools_ex(
            Some(tmpl),
            &plan.request.chat_turns,
            true,
            &plan.request.tools_json,
            &plan.request.tools_struct,
            plan.request.think,
            plan.request.reasoning_effort.as_deref(),
            None,
        )
        .unwrap())
    }

    #[test]
    fn qwen38_effort_ladder_reaches_prompt_bytes_through_the_whole_api() {
        // THE HEADLINE DEFECT. `reasoning_effort: low|medium|high` was parsed, validated, and
        // then DISCARDED on every qwen3.8 request: the delivery gate asked for
        // `effort_levels || dsv4`, and `effort_levels` probes the substring
        // `reasoning_effort is defined`, which this template does not contain (it spells its
        // input `reasoning_effort|default('xhigh')`). So the level never reached the render and
        // the template's own `xhigh` default never rendered either.
        let r = |extra: serde_json::Value| render_with(Q38_TMPL, &ladder_caps(), extra, None);
        let xhigh = "Reasoning effort is set to xhigh.";
        let low = "Reasoning effort is set to low.";
        // Each rung lands on the sentence the VENDOR's template defines for it.
        assert!(r(json!({"reasoning_effort": "low"})).unwrap().contains(low));
        assert!(
            r(json!({"reasoning_effort": "high"}))
                .unwrap()
                .contains(xhigh)
        );
        // `medium` is the vendor's zero-steering rung: it injects nothing at all. That is the
        // template's own choice, and it is ALSO the byte history of every pre-lane q38 request.
        let medium = r(json!({"reasoning_effort": "medium"})).unwrap();
        assert!(!medium.contains("Reasoning effort is set to"), "{medium:?}");
        // ...so the three rungs are three DIFFERENT prompts. Effect, proven on bytes.
        let low_p = r(json!({"reasoning_effort": "low"})).unwrap();
        let high_p = r(json!({"reasoning_effort": "high"})).unwrap();
        assert_ne!(low_p, high_p);
        assert_ne!(low_p, medium);
        assert_ne!(high_p, medium);
        // The clamp aliases are ONE rung by the vendor's own hosted-API mapping (high/max/xhigh
        // -> xhigh), so they must not become a fourth prompt.
        for alias in ["xhigh", "max", "ultra"] {
            assert_eq!(r(json!({"reasoning_effort": alias})).unwrap(), high_p);
        }
        // THE SERVING-BEHAVIOUR CHANGE, pinned so it cannot land unnoticed: an UNSET request
        // now renders the vendor's xhigh default, where before it rendered nothing.
        assert_eq!(r(json!({})).unwrap(), high_p);
        // ...and the documented no-op migration: an operator default of "medium" restores the
        // exact pre-lane bytes without touching a line of code.
        assert_eq!(
            render_with(Q38_TMPL, &ladder_caps(), json!({}), Some("medium")).unwrap(),
            medium
        );
        // Thinking OFF carries no effort sentence even with a level named — the vendor wraps the
        // whole instruction block in `enable_thinking is undefined or is true`.
        let off = r(json!({"reasoning_effort": "none"})).unwrap();
        assert!(off.ends_with("<think>\n\n</think>\n\n"), "{off:?}");
        assert!(!off.contains("Reasoning effort is set to"), "{off:?}");
    }

    #[test]
    fn the_effort_sentence_is_measurable_on_the_deployed_binary_without_a_deploy() {
        // METHODOLOGY GATE for the live cell in darklanes
        // research/reasoning-schema-20260823/SCHEMA.md §5. That measurement had to answer "does
        // each rung change what the model DOES" against a binary that predates this branch, so it
        // sent each rung's instruction sentence as a SYSTEM MESSAGE instead. That is only a valid
        // substitute if the two render the same bytes — otherwise the numbers describe a prompt no
        // customer will ever get and the whole cell is decoration.
        //
        // Note WHERE the ladder is keyed, because a first attempt at this test got it wrong: the
        // renderer probes the TEMPLATE (`template_has_qwen_effort`), while `ModelCaps::qwen_effort`
        // only decides whether the level STRING is handed to it. So "the deployed binary" cannot be
        // modelled by clearing the cap — it is modelled by a template that carries no ladder at
        // all, which is what the pre-lane renderer effectively was.
        const LOW_SENTENCE: &str = "Reasoning effort is set to low. Keep your thinking brief and \
focused, moving directly to the conclusion without unnecessary elaboration.";
        let expected = format!(
            "<|im_start|>system\n{LOW_SENTENCE}<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        // RIGHT SIDE — this branch: the level, no system message.
        let after_fix = render_with(
            Q38_TMPL,
            &ladder_caps(),
            json!({"reasoning_effort": "low"}),
            None,
        )
        .unwrap();
        assert_eq!(
            after_fix, expected,
            "the shipped prompt for reasoning_effort:\"low\""
        );
        // LEFT SIDE — a ladder-less template, sentence carried in a system message: byte-identical,
        // and this is exactly the request the live cell sent to the deployed endpoint.
        const ORNITH_TMPL: &str = include_str!(
            "../../../research/reasoning-schema-20260823/ornith15.chat_template.jinja"
        );
        let on_deployed_binary = render_with(
            ORNITH_TMPL,
            &tool_caps(),
            json!({"messages": [{"role": "system", "content": LOW_SENTENCE},
                                {"role": "user", "content": "hi"}]}),
            None,
        )
        .unwrap();
        assert_eq!(
            on_deployed_binary, expected,
            "the live cell's system-message stand-in must render the SAME bytes as the post-fix \
             level, or its reasoning-volume numbers do not describe the shipped prompt"
        );
        // And the baseline the cell measured against: a ladder-less template injects no instruction
        // at all, which is why `medium` — the vendor's zero-steering rung — is the pre-lane bytes.
        let ladderless_unset = render_with(ORNITH_TMPL, &tool_caps(), json!({}), None).unwrap();
        assert!(
            !ladderless_unset.contains("Reasoning effort is set to"),
            "pre-lane q38 injected no effort instruction at any level: {ladderless_unset:?}"
        );
        assert_eq!(
            ladderless_unset,
            render_with(
                Q38_TMPL,
                &ladder_caps(),
                json!({"reasoning_effort": "medium"}),
                None
            )
            .unwrap(),
            "medium is the vendor's zero-steering rung and therefore the pre-lane byte baseline"
        );
    }

    #[test]
    fn include_reasoning_false_stops_reasoning_it_does_not_hide_it() {
        // OWNER RULING 2026-08-23: *"we have to actually reason or not reason"*. Reasoning is
        // compute and output, billed as output, so a flag that only withheld the text charged
        // the customer for output we never sent. `include_reasoning:false` and
        // `reasoning.exclude:true` are now spellings of reasoning-OFF, and the proof is that the
        // PROMPT closes the think pair — a test that only checked a response-shaping flag would
        // have passed against the old, banned behaviour.
        let off = render_with(
            Q38_TMPL,
            &ladder_caps(),
            json!({"reasoning_effort": "none"}),
            None,
        )
        .unwrap();
        for extra in [
            json!({"include_reasoning": false}),
            json!({"reasoning": {"exclude": true}}),
        ] {
            let got = render_with(Q38_TMPL, &ladder_caps(), extra.clone(), None).unwrap();
            assert!(
                got.ends_with("<think>\n\n</think>\n\n"),
                "{extra} must render the CLOSED think pair, not a hidden reasoning block: {got:?}"
            );
            assert_eq!(got, off, "{extra} must be byte-identical to reasoning-off");
        }
        // A suppression request that CONTRADICTS an on-switch refuses, and the message names the
        // field the caller actually sent — the two folds are ordered so that
        // `enable_thinking:true` + `include_reasoning:false` is reported against
        // include_reasoning, not against a `reasoning.enabled` that was never in the body.
        for extra in [
            json!({"enable_thinking": true, "include_reasoning": false}),
            json!({"reasoning": {"enabled": true}, "include_reasoning": false}),
            json!({"reasoning": {"enabled": true, "exclude": true}}),
        ] {
            let e = render_with(Q38_TMPL, &ladder_caps(), extra.clone(), None)
                .err()
                .unwrap_or_else(|| panic!("{extra} must be refused as contradictory"));
            assert!(e.contains("contradictory"), "{extra}: {e}");
            assert!(
                e.contains("include_reasoning") || e.contains("exclude"),
                "{extra}: the refusal must name the suppression field the caller sent: {e}"
            );
        }
        // The "deliver it" direction is the only behaviour, so it expresses no switch at all and
        // leaves the model's own default alone.
        let dflt = render_with(Q38_TMPL, &ladder_caps(), json!({}), None).unwrap();
        for extra in [
            json!({"include_reasoning": true}),
            json!({"reasoning": {"exclude": false}}),
        ] {
            assert_eq!(
                render_with(Q38_TMPL, &ladder_caps(), extra.clone(), None).unwrap(),
                dflt,
                "{extra} must not perturb the model's default"
            );
        }
        // And on a model that CANNOT turn reasoning off, hiding is not a fallback — it is the
        // same named refusal as any other off-request, instead of a 200 that billed for a
        // reasoning block the caller never saw.
        let switchless = ModelCaps {
            think_switch: false,
            ..tool_caps()
        };
        let err = render_with(
            Q38_TMPL,
            &switchless,
            json!({"include_reasoning": false}),
            None,
        )
        .expect_err("include_reasoning:false must not silently bill for hidden reasoning");
        assert!(err.contains("cannot disable reasoning"), "{err}");
    }

    #[test]
    fn the_reasoning_object_refuses_every_key_it_cannot_honour() {
        let build = |extra: serde_json::Value| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request(
                weather_request(extra),
                Some(&ladder_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
        };
        let err = |extra: serde_json::Value, why: &str| -> String {
            build(extra).err().unwrap_or_else(|| panic!("{why}"))
        };
        // `reasoning.max_tokens` is a REAL OpenRouter field that was accepted and never read.
        // It is unhonourable by owner ruling, not merely unimplemented: reasoning tokens are
        // output tokens under the single `max_tokens` budget, so there is no second budget.
        let e = err(
            json!({"reasoning": {"max_tokens": 1024}}),
            "reasoning.max_tokens must not be accepted-and-ignored",
        );
        assert!(e.contains("reasoning.max_tokens"), "{e}");
        assert!(e.contains("ONE output budget"), "{e}");
        // ...and NULLING an unhonourable key must not smuggle it past its own refusal. A first cut
        // of the null-as-unset convention applied the skip before the key match, so these two
        // returned 200 and changed nothing — the exact class this function closes, reintroduced by
        // the fix for a different divergence.
        for extra in [
            json!({"reasoning": {"max_tokens": null}}),
            json!({"reasoning": {"banana": null}}),
        ] {
            let e = err(
                extra.clone(),
                "a null-valued unhonourable key must still refuse",
            );
            assert!(
                e.contains("max_tokens") || e.contains("banana"),
                "{extra}: {e}"
            );
        }
        // Any other unknown key: named, like the chat_template_kwargs law one level up.
        let e = err(
            json!({"reasoning": {"budget": 5}}),
            "an unknown reasoning key must not be accepted",
        );
        assert!(
            e.contains("reasoning.budget") && e.contains("enabled"),
            "{e}"
        );
        // WRONG TYPES are refusals too — and this removes a cross-surface divergence: these
        // used to fall through `as_bool()`/`as_str()` to None and be silently ignored on chat,
        // while /v1/messages already 400'd on the same mistake.
        for (extra, want) in [
            (json!({"reasoning": {"enabled": "false"}}), "true or false"),
            (json!({"reasoning": {"exclude": 1}}), "true or false"),
            (json!({"reasoning": {"effort": 3}}), "must be a string"),
        ] {
            let e = err(
                extra.clone(),
                "a wrong-typed reasoning key must not be ignored",
            );
            assert!(e.contains(want), "{extra}: {e}");
        }
        // The three keys we DO implement still work, and an explicit null is "unset" — for a KEY
        // as well as for the whole object. That last part closes the final cross-surface
        // divergence: `{"effort": null}` used to 400 here while /v1/responses and /v1/messages
        // both read it as unset, so the same body got two answers.
        for extra in [
            json!({"reasoning": {"enabled": true}}),
            json!({"reasoning": {"effort": "low"}}),
            json!({"reasoning": {"exclude": false}}),
            json!({"reasoning": null}),
            json!({"reasoning": {"effort": null}}),
            json!({"reasoning": {"enabled": null, "exclude": null}}),
        ] {
            build(extra.clone()).unwrap_or_else(|e| panic!("{extra} must be served: {e}"));
        }
    }

    #[test]
    fn a_graded_level_on_a_binary_model_translates_to_reasoning_on() {
        // THE TRANSLATION RULING (coordinator, 2026-08-23). On ornith's shape — the same binary
        // `enable_thinking` guard as qwen, no depth input, thinking ON by default — a graded
        // level folds onto the binary axis as reasoning ON. A first cut REFUSED it (the
        // construction proof below shows the level cannot move this template's bytes), but the
        // refusal broke stock codex and Claude Code sessions, both of which send `xhigh` on
        // every request; the owner authorised translation into the one schema, and a caller who
        // asked for reasoning and gets reasoning has their promise kept.
        const ORNITH_TMPL: &str = include_str!(
            "../../../research/reasoning-schema-20260823/ornith15.chat_template.jinja"
        );
        // The construction fact the translation documents (and the old refusal rested on): a
        // level cannot move this template's bytes, so translated requests render byte-identical
        // to an explicit boolean ON.
        let explicit_on = render_with(
            ORNITH_TMPL,
            &tool_caps(),
            json!({"reasoning": {"enabled": true}}),
            None,
        )
        .unwrap();
        assert!(explicit_on.ends_with("<think>\n"), "{explicit_on:?}");
        for extra in [
            json!({"reasoning_effort": "low"}),
            json!({"reasoning_effort": "medium"}),
            json!({"reasoning_effort": "high"}),
            // the stock-CLI spellings the first cut's refusal would have broken:
            json!({"reasoning_effort": "xhigh"}),
            json!({"reasoning": {"effort": "xhigh"}}),
        ] {
            let got = render_with(ORNITH_TMPL, &tool_caps(), extra.clone(), None)
                .unwrap_or_else(|e| panic!("{extra} must TRANSLATE to reasoning-on, got 400: {e}"));
            assert_eq!(
                got, explicit_on,
                "{extra} must render byte-identical to reasoning:{{enabled:true}} — the \
                 documented translation, not a decorative accept"
            );
        }
        // The binary controls this model's lab defines keep working: off, on, unset.
        for extra in [
            json!({}),
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "minimal"}),
            json!({"enable_thinking": false}),
        ] {
            render_with(ORNITH_TMPL, &tool_caps(), extra.clone(), None)
                .unwrap_or_else(|e| panic!("{extra} must still be served: {e}"));
        }
        // ...and `minimal` stays OFF — our schema's deliberate divergence from Qwen's
        // minimal->low, decided 2026-08-23: the no-reasoning side of our schema is real.
        let minimal = render_with(
            ORNITH_TMPL,
            &tool_caps(),
            json!({"reasoning_effort": "minimal"}),
            None,
        )
        .unwrap();
        assert!(
            minimal.ends_with("<think>\n\n</think>\n\n"),
            "minimal must close the think pair (OFF), not clamp to a reasoning level: {minimal:?}"
        );
        // A model WITH the ladder still gets its real rungs — the translation is keyed on the
        // template's capability, never on the field being present.
        let ladder_low = render_with(
            Q38_TMPL,
            &ladder_caps(),
            json!({"reasoning_effort": "low"}),
            None,
        )
        .unwrap();
        assert!(
            ladder_low.contains("Reasoning effort is set to low."),
            "{ladder_low:?}"
        );
        assert_ne!(
            ladder_low,
            render_with(
                Q38_TMPL,
                &ladder_caps(),
                json!({"reasoning_effort": "high"}),
                None
            )
            .unwrap(),
            "the ladder model's rungs stay distinct prompts"
        );
    }

    #[test]
    fn one_semantic_reasoning_request_renders_identical_bytes_on_all_three_surfaces() {
        // THE STANDARD-SURFACE LAW, at the byte level. `/v1/responses` and `/v1/messages` are
        // translation surfaces over the chat core, so "the same request" means: each surface's
        // OWN vocabulary for a semantic intent must land on the same internal schema and
        // therefore the same prompt. A parameter honoured on one format and ignored on another is
        // the same defect wearing a different hat — and issue #31 was exactly that.
        //
        // This is the byte half. The schema half (surface -> `(ThinkMode, effort_level)` as the
        // WORKER sees it, through the real handlers) is
        // `same_effort_value_resolves_identically_on_every_surface`. Together they close the
        // chain surface -> schema -> bytes.
        let render_chat = |body: serde_json::Value| -> Result<String, String> {
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                req,
                Some(&ladder_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )?;
            Ok(chat::apply_chat_template_tools_ex(
                Some(Q38_TMPL),
                &plan.request.chat_turns,
                true,
                &plan.request.tools_json,
                &plan.request.tools_struct,
                plan.request.think,
                plan.request.reasoning_effort.as_deref(),
                None,
            )
            .unwrap())
        };
        // Each row: one semantic intent, spelled the way each surface's own clients spell it.
        //   chat            = OpenAI / OpenRouter / vLLM
        //   /v1/responses   = OpenAI Responses (what codex speaks)
        //   /v1/messages    = Anthropic Messages (what Claude Code speaks)
        for (intent, chat_body, responses_body, messages_body) in [
            (
                "reasoning OFF",
                json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": "none"}),
                json!({"model": "m", "input": "hi", "reasoning": {"effort": "none"}}),
                json!({"model": "m", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "hi"}],
                       "thinking": {"type": "disabled"}}),
            ),
            (
                "reasoning ON at the top rung",
                json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": "xhigh"}),
                json!({"model": "m", "input": "hi", "reasoning": {"effort": "xhigh"}}),
                json!({"model": "m", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "hi"}],
                       "output_config": {"effort": "xhigh"}}),
            ),
            (
                "reasoning ON at the bottom rung",
                json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "reasoning_effort": "low"}),
                json!({"model": "m", "input": "hi", "reasoning": {"effort": "low"}}),
                json!({"model": "m", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "hi"}],
                       "output_config": {"effort": "low"}}),
            ),
            (
                "the model's own default",
                json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
                json!({"model": "m", "input": "hi"}),
                json!({"model": "m", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "hi"}]}),
            ),
        ] {
            let chat = render_chat(chat_body).unwrap_or_else(|e| panic!("{intent} on chat: {e}"));
            let via_responses = responses_api::translate(&responses_body)
                .unwrap_or_else(|e| panic!("{intent} on /v1/responses: {e:?}"));
            let via_messages = anthropic::translate(&messages_body)
                .unwrap_or_else(|e| panic!("{intent} on /v1/messages: {e}"));
            for (surface, translated) in [
                ("/v1/responses", via_responses),
                ("/v1/messages", via_messages),
            ] {
                let got = render_chat(translated)
                    .unwrap_or_else(|e| panic!("{intent} via {surface}: {e}"));
                assert_eq!(
                    got, chat,
                    "{intent}: {surface} rendered DIFFERENT prompt bytes than \
                     /v1/chat/completions — the parameter is honoured on one format and not \
                     the other"
                );
            }
        }
        // And the refusals agree too: an intent no model can honour must not be a 400 on one
        // surface and a 200 on another.
        let switchless = ModelCaps {
            think_switch: false,
            ..ladder_caps()
        };
        let render_switchless = |body: serde_json::Value| -> Result<String, String> {
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            let plan =
                build_chat_request(req, Some(&switchless), tx, lanes::Lane::Interactive, None)?;
            Ok(format!("{:?}", plan.request.think))
        };
        for (surface, body) in [
            (
                "/v1/responses",
                responses_api::translate(&json!({
                    "model": "m", "input": "hi", "reasoning": {"effort": "none"}}))
                .unwrap(),
            ),
            (
                "/v1/messages",
                anthropic::translate(&json!({
                    "model": "m", "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "disabled"}}))
                .unwrap(),
            ),
        ] {
            let err = render_switchless(body)
                .err()
                .unwrap_or_else(|| panic!("{surface} must refuse an unhonourable off-request"));
            assert!(err.contains("cannot disable reasoning"), "{surface}: {err}");
        }
    }

    #[test]
    fn preserve_thinking_true_is_the_implemented_default_and_false_refuses() {
        // Qwen3.8's THIRD official thinking kwarg (its own quickstart sends
        // `{"enable_thinking": True, "preserve_thinking": True}`). The ladder renderer now
        // implements the vendor DEFAULT (replay every prior assistant turn's <think> block;
        // lane/dflash2-session-reuse), so `true` names exactly what the server renders and
        // must be ACCEPTED — Qwen's own quickstart payload has to serve. `false` (the strip
        // arm, with its last_query_index walk) stays unimplemented and refuses: serving
        // replay bytes under a strip request would misdescribe the prompt.
        let build = |extra: serde_json::Value| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request(
                weather_request(extra),
                Some(&ladder_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
        };
        build(json!({"chat_template_kwargs": {"preserve_thinking": true}}))
            .expect("preserve_thinking:true is the vendor default the renderer implements");
        let e = build(json!({"chat_template_kwargs": {"preserve_thinking": false}}))
            .err()
            .expect("preserve_thinking:false (the strip arm) must refuse");
        assert!(e.contains("preserve_thinking"), "{e}");
        assert!(e.contains("strip"), "{e}");
        // Omitting it still serves — refusing the absent case would refuse every multi-turn
        // request — and the switch in the same bag keeps working.
        assert_eq!(
            build(json!({"chat_template_kwargs": {"enable_thinking": false}}))
                .unwrap()
                .request
                .think,
            ThinkMode::NoThink
        );
        // a non-bool is still a type error, not a silent drop.
        let e = build(json!({"chat_template_kwargs": {"preserve_thinking": "false"}}))
            .err()
            .expect("a stringly-typed preserve_thinking must not be accepted");
        assert!(e.contains("true or false"), "{e}");
    }

    #[test]
    fn dsv4_is_exempt_from_the_switchless_off_refusal() {
        // The dsv4 renderer honours reasoning-off through its own `chat` thinking mode, so it
        // needs no `enable_thinking` marker to turn reasoning off. PR #33's marker pair
        // (`qwen_think && !think_switch`) would have refused it — latent only because
        // encoding-keyed artifacts carry no template string. Keyed explicitly so it cannot
        // become live by accident.
        let dsv4_caps = ModelCaps {
            qwen_think: true,
            think_switch: false,
            dsv4: true,
            ..tool_caps()
        };
        for extra in [
            json!({"reasoning_effort": "none"}),
            json!({"reasoning": {"enabled": false}}),
            json!({"enable_thinking": false}),
            json!({"include_reasoning": false}),
        ] {
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                weather_request(extra.clone()),
                Some(&dsv4_caps),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .unwrap_or_else(|e| panic!("{extra} must be served on dsv4: {e}"));
            assert_eq!(plan.request.think, ThinkMode::NoThink, "extra={extra}");
        }
    }

    #[test]
    fn contradictory_think_switches_refuse_instead_of_picking_one() {
        // Two explicit switches that disagree: silently honoring one makes the other an
        // accepted-and-ignored parameter, which is the whole class this lane removes.
        let build = |extra: serde_json::Value| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request(
                weather_request(extra),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
        };
        for extra in [
            json!({"enable_thinking": true, "reasoning": {"enabled": false}}),
            json!({"enable_thinking": false, "reasoning": {"enabled": true}}),
            json!({"enable_thinking": false, "chat_template_kwargs": {"enable_thinking": true}}),
        ] {
            match build(extra.clone()) {
                Err(err) => assert!(
                    err.contains("contradictory"),
                    "the refusal must say the switches contradict: {err}"
                ),
                Ok(plan) => panic!(
                    "{extra} must be rejected as contradictory; it silently resolved to {:?}",
                    plan.request.think
                ),
            }
        }
        // agreeing switches, and a switch next to an EFFORT LEVEL, are not contradictions.
        for extra in [
            json!({"enable_thinking": false, "reasoning": {"enabled": false}}),
            json!({"enable_thinking": true, "reasoning": {"enabled": true}}),
            json!({"enable_thinking": false, "reasoning": {"effort": "high"}}),
        ] {
            build(extra.clone())
                .unwrap_or_else(|e| panic!("{extra} is not a contradiction, but got 400: {e}"));
        }
    }

    #[test]
    fn explicit_reasoning_off_on_a_switchless_template_refuses_loudly() {
        // The latent twin of the vLLM defect: on a template whose think tail is
        // UNCONDITIONAL (`qwen_think` with no `enable_thinking`), NoThink has always been a
        // documented no-op — which at the API boundary means 200 + a full reasoning block
        // for a caller who asked for none. Now a named 400.
        let switchless = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            think_switch: false,
            chat_ok: true,
            ..Default::default()
        };
        let build = |extra: serde_json::Value, caps: &ModelCaps, default_effort: Option<&str>| {
            let (tx, _rx) = worker::event_channel();
            build_chat_request_with_trace(
                weather_request(extra),
                Some(caps),
                tx,
                lanes::Lane::Interactive,
                None,
                None,
                default_effort,
                &ModelSamplingDefaults::default(),
            )
        };
        for extra in [
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "minimal"}),
            json!({"reasoning": {"enabled": false}}),
            json!({"enable_thinking": false}),
            json!({"chat_template_kwargs": {"enable_thinking": false}}),
        ] {
            let err = build(extra.clone(), &switchless, None)
                .err()
                .unwrap_or_else(|| {
                    panic!(
                        "{extra} on a switchless think template must not be accepted-and-ignored"
                    )
                });
            assert!(
                err.contains("cannot disable reasoning"),
                "the refusal must say the model cannot disable reasoning: {err}"
            );
        }
        // Everything else on the same model is untouched: thinking-ON requests, unset
        // requests, and — critically — an OPERATOR default of "none", which must never turn
        // into a 400 for a caller who expressed nothing.
        for (extra, default_effort) in [
            (json!({}), None),
            // a client-named LEVEL translates onto the binary axis as reasoning ON (coordinator
            // ruling 2026-08-23) — this template reasons by default, so the promise is kept.
            (json!({"reasoning_effort": "high"}), None),
            (json!({"reasoning": {"enabled": true}}), None),
            (json!({"enable_thinking": true}), None),
            (json!({}), Some("none")),
            (json!({}), Some("minimal")),
            (json!({}), Some("high")),
        ] {
            build(extra.clone(), &switchless, default_effort).unwrap_or_else(|e| {
                panic!("{extra} (default={default_effort:?}) must still be served: {e}")
            });
        }
        // A model WITH the switch serves the same off-request normally — the refusal is
        // keyed on the template, never on the field being present.
        assert_eq!(
            build(json!({"enable_thinking": false}), &tool_caps(), None)
                .unwrap()
                .request
                .think,
            ThinkMode::NoThink
        );
    }

    #[test]
    fn gemma4_default_think_on_renders_byte_identical_to_explicit_think_on() {
        // Template-render identity gate: with the knob active, an UNSET request's
        // rendered prompt equals the explicit think-on request's prompt byte-for-byte —
        // the knob substitutes into the SAME parse_think mapping before the plan is
        // built; it does not grow a second render path. The vendor template's own
        // rendering semantics are untouched: explicit-off and knobless deployments still
        // render the CLOSED thought channel.
        let gemma_caps = ModelCaps {
            tools_branch: true,
            chat_ok: true,
            gemma_think: true,
            instruct_type: Some("gemma".into()),
            ..Default::default()
        };
        let render =
            |tmpl: &str, extra: serde_json::Value, default_effort: Option<&str>| -> String {
                let mut payload = serde_json::json!({
                    "model": "google/gemma-4-31b-it",
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                });
                if let Some(obj) = extra.as_object() {
                    for (k, v) in obj {
                        payload[k] = v.clone();
                    }
                }
                let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
                let (tx, _rx) = worker::event_channel();
                let plan = build_chat_request_with_trace(
                    req,
                    Some(&gemma_caps),
                    tx,
                    lanes::Lane::Interactive,
                    None,
                    None,
                    default_effort,
                    &ModelSamplingDefaults::default(),
                )
                .unwrap();
                chat::apply_chat_template_tools_ex(
                    Some(tmpl),
                    &plan.request.chat_turns,
                    true,
                    &plan.request.tools_json,
                    &plan.request.tools_struct,
                    plan.request.think,
                    plan.request.reasoning_effort.as_deref(),
                    None, // gemma template — no dsv4 encoding revision
                )
                .unwrap()
            };
        let official = gemma_template("official");
        let unset_with_knob = render(&official, json!({}), Some("high"));
        let explicit_on = render(&official, json!({"reasoning_effort": "high"}), None);
        assert_eq!(
            unset_with_knob, explicit_on,
            "knob render must be byte-identical to the explicit think-on render"
        );
        assert!(
            unset_with_knob.starts_with("<|turn>system\n<|think|>\n"),
            "think-on injects the <|think|> system token: {unset_with_knob:?}"
        );
        assert!(
            unset_with_knob.ends_with("<|turn>model\n"),
            "think-on generation turn is OPEN: {unset_with_knob:?}"
        );
        // explicit off under the knob = byte-identical to explicit off without it. On the
        // OFFICIAL tooluse trunk the vendor law for thinking-off is a bare open model
        // turn with NO <|think|> system token (closed_tail is the QAT-trunk variant).
        let explicit_off_with_knob =
            render(&official, json!({"reasoning_effort": "none"}), Some("high"));
        let explicit_off = render(&official, json!({"reasoning_effort": "none"}), None);
        assert_eq!(explicit_off_with_knob, explicit_off);
        assert!(
            !explicit_off_with_knob.contains("<|think|>")
                && explicit_off_with_knob.ends_with("<|turn>model\n"),
            "explicit off keeps the official template's thinking-off bytes: \
             {explicit_off_with_knob:?}"
        );
        // knobless unset = the template's own default (today's serving bytes).
        let unset_no_knob = render(&official, json!({}), None);
        assert_eq!(
            unset_no_knob, explicit_off,
            "knobless unset stays the template's own thinking-off default"
        );
        assert_ne!(unset_no_knob, unset_with_knob);
        // QAT-trunk variant: its thinking-off generation prompt appends the CLOSED
        // thought channel — the knob must not perturb that vendor law either.
        let qat = gemma_template("qat");
        assert!(
            render(&qat, json!({}), None).ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
            "QAT knobless unset keeps the closed-channel default"
        );
        assert_eq!(
            render(&qat, json!({}), Some("high")),
            render(&qat, json!({"reasoning_effort": "high"}), None),
            "QAT knob render must equal the explicit think-on render"
        );
    }

    #[test]
    fn default_reasoning_effort_is_validated_at_metadata_load() {
        // A typo'd knob fails at BOOT (metadata parse), never per-request.
        let parsed = OpenRouterMetadataFile::from_toml(
            r#"
[models.g]
default_reasoning_effort = "high"
"#,
        )
        .unwrap();
        assert_eq!(
            parsed.get("g").unwrap().default_reasoning_effort.as_deref(),
            Some("high")
        );
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.g]
default_reasoning_effort = "always"
"#,
        )
        .unwrap_err();
        assert!(err.contains("default_reasoning_effort"), "{err}");
    }

    #[test]
    fn reasoning_effort_maps_to_effort_level_on_step35_class_templates() {
        // ModelCaps::effort_levels=true (the step35 dialect): the SAME client field becomes
        // a render input (Request::reasoning_effort) — low/medium/high pass through, absent
        // stays None (the template's own default: no `Reasoning:` line).
        //
        // THE REAL CAPS INTERSECTION (lane/reasoning-schema-20260823, found by review of PR #33
        // before release). This used to inherit `think_switch: true` from `tool_caps()` — a
        // combination NO real step35 template can produce, since its `<think>` tail is
        // unconditional and it carries no `enable_thinking`. Probing the shipped template
        // (research/step37-bringup-20260802/raw/chat_template.jinja) gives
        // `qwen_think=true, think_switch=false, effort_levels=true`, so that is what the test
        // asserts against — otherwise CI is blind to what a live step35 actually does.
        let effort_caps = ModelCaps {
            effort_levels: true,
            think_switch: false,
            ..tool_caps()
        };
        for (extra, want) in [
            (json!({}), None),
            (json!({"reasoning_effort": "low"}), Some("low")),
            (json!({"reasoning_effort": "medium"}), Some("medium")),
            (json!({"reasoning_effort": "high"}), Some("high")),
            (json!({"reasoning": {"effort": "high"}}), Some("high")),
            // clamp aliases render as the highest level the template distinguishes
            (json!({"reasoning_effort": "xhigh"}), Some("high")),
            (json!({"reasoning": {"effort": "max"}}), Some("high")),
        ] {
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                weather_request(extra.clone()),
                Some(&effort_caps),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .unwrap();
            assert_eq!(
                plan.request.reasoning_effort.as_deref(),
                want,
                "extra={extra}"
            );
        }
        // AN OFF-REQUEST ON STEP35 IS NOW A NAMED 400, NOT A CLAMP TO THE LOWEST RUNG.
        // It used to resolve `none`/`minimal`/`reasoning.enabled:false` to `Reasoning: low` —
        // i.e. a caller who asked for NO reasoning was served reasoning at the lowest level,
        // behind a 200. That is the owner's named unacceptable case (2026-08-23: asking for
        // non-reasoning and getting reasoning must be impossible), and step35's `<think>` tail
        // is unconditional, so the honest answer is a refusal naming the model.
        for extra in [
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "minimal"}),
            json!({"reasoning": {"enabled": false}}),
            json!({"enable_thinking": false}),
            json!({"include_reasoning": false}),
        ] {
            let (tx, _rx) = worker::event_channel();
            let err = build_chat_request(
                weather_request(extra.clone()),
                Some(&effort_caps),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .err()
            .unwrap_or_else(|| panic!("{extra} must not be clamped to a reasoning level"));
            assert!(
                err.contains("cannot disable reasoning"),
                "extra={extra}: {err}"
            );
        }
        // effort_levels=false AND the template reasons by default (the ornith/qwen-class shape):
        // a client-named level TRANSLATES onto the binary axis as reasoning ON (coordinator
        // ruling 2026-08-23 — a first cut refused these, which broke stock codex/Claude Code
        // sessions against ornith). The level string is dropped by the delivery gate, so the
        // prompt is byte-identical to explicit-ON by construction; the byte proof lives in
        // `a_graded_level_on_a_binary_model_translates_to_reasoning_on`.
        for extra in [
            json!({"reasoning_effort": "high"}),
            json!({"reasoning": {"effort": "low"}}),
        ] {
            let (tx, _rx) = worker::event_channel();
            let plan = build_chat_request(
                weather_request(extra.clone()),
                Some(&tool_caps()),
                tx,
                lanes::Lane::Interactive,
                None,
            )
            .unwrap_or_else(|e| panic!("{extra} must translate, not refuse: {e}"));
            assert_eq!(plan.request.think, ThinkMode::Think, "extra={extra}");
            assert_eq!(plan.request.reasoning_effort, None, "extra={extra}");
        }
        // and an unset request on that class still renders the template's own default.
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(
            weather_request(json!({})),
            Some(&tool_caps()),
            tx,
            lanes::Lane::Interactive,
            None,
        )
        .unwrap();
        assert_eq!(plan.request.reasoning_effort, None);
    }

    #[test]
    fn assistant_history_tool_calls_and_tool_role_render_into_turns() {
        let payload = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_x", "type": "function", "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
                {"role": "tool", "tool_call_id": "call_x", "content": "{\"temp_c\": 21}"}
            ],
        });
        let req: ChatCompletionReq = serde_json::from_value(payload).unwrap();
        let (tx, _rx) = worker::event_channel();
        let plan = build_chat_request(req, Some(&tool_caps()), tx, lanes::Lane::Interactive, None)
            .unwrap();
        let turns = &plan.request.chat_turns;
        assert_eq!(turns[1].tool_calls.len(), 1);
        assert_eq!(turns[1].tool_calls[0].name, "get_weather");
        assert_eq!(
            turns[1].tool_calls[0].params,
            vec![("city".into(), "Paris".into()), ("days".into(), "3".into())]
        );
        assert_eq!(turns[2].role, "tool");
        assert_eq!(turns[2].content, "{\"temp_c\": 21}");
        // no tools field on this follow-up turn: no tool-call scanning — but the think-open
        // prompt still arms the reasoning-only splitter (gap-scan F13).
        let mut p = plan
            .parser
            .expect("think-open chat arms the reasoning splitter");
        let pieces = p.push("thought</think>\n\nanswer <tool_call> is prose here");
        assert_eq!(
            pieces,
            vec![
                Piece::Reasoning("thought".into()),
                Piece::Content("answer <tool_call> is prose here".into()),
            ]
        );
    }

    #[tokio::test]
    async fn blocking_tools_response_carries_tool_calls_and_finish_reason() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "plan</think>\n\n".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "<tool_call>\n<function=get_weather>\n\
<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
                .into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 2,
            n_prompt: 40,
            n_cached: 0,
            elapsed_s: 0.5,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let parser = ToolStreamParser::new(HashMap::new(), true);
        let response = blocking_response(
            rx,
            "m".into(),
            true,
            Vec::new(),
            Some(parser),
            Envelope::new(true),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["choices"][0]["finish_reason"], "tool_calls");
        // reasoning separation (gap-scan F13): think text -> message.reasoning (+details),
        // content is post-think only (null here — a pure tool-call turn).
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::Value::Null
        );
        assert_eq!(payload["choices"][0]["message"]["reasoning"], "plan");
        assert_eq!(
            payload["choices"][0]["message"]["reasoning_details"][0]["text"],
            "plan"
        );
        let call = &payload["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
        // THE INTERSECTION (integrate-cache): a tools response's usage carries the same
        // worker-truth prompt/cached split as any other shape — one source of truth.
        assert_eq!(payload["usage"]["prompt_tokens"], 40);
        assert_eq!(payload["usage"]["completion_tokens"], 2);
        assert_eq!(payload["usage"]["total_tokens"], 42);
        assert_eq!(
            payload["usage"]["prompt_tokens_details"]["cached_tokens"],
            0
        );
    }

    #[test]
    fn cache_salt_plumbs_to_the_worker_namespace() {
        // PC-ISO: explicit cache_salt -> the request's cache namespace, on BOTH bodies.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task", "cache_salt": "tenant-a"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_request(&req, tx, lanes::Lane::Interactive, None).cache_ns,
            "tenant-a"
        );

        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}],
            "cache_salt": "tenant-b"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .cache_ns,
            "tenant-b"
        );

        // no salt -> "" (the default single-tenant namespace; pre-PC-ISO behavior).
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_request(&req, tx, lanes::Lane::Interactive, None).cache_ns,
            ""
        );
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}]
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert_eq!(
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .cache_ns,
            ""
        );
    }

    #[test]
    fn cache_salt_validation_rejects_oversized_value() {
        let salt = Some("a".repeat(CACHE_SALT_MAX_BYTES + 1));
        assert_eq!(
            validate_cache_namespace(&salt, false),
            Err("cache_salt must be at most 64 bytes")
        );
    }

    #[test]
    fn cache_salt_validation_rejects_reserved_open_namespace() {
        let salt = Some("t:acme\u{1f}private".to_string());
        assert_eq!(
            validate_cache_namespace(&salt, false),
            Err("cache_salt must not use the reserved t: prefix without a keyring")
        );
    }

    #[test]
    fn cache_salt_validation_accepts_normal_value() {
        let raw = "tenant-A_7.c2VjcmV0LXNjb3Bl+/=";
        let salt = Some(raw.to_string());
        assert_eq!(validate_cache_namespace(&salt, false).unwrap(), raw);
        assert_eq!(validate_cache_namespace(&None, false).unwrap(), "");
        let max_raw = "a".repeat(CACHE_SALT_MAX_BYTES);
        let max = Some(max_raw.clone());
        assert_eq!(validate_cache_namespace(&max, false).unwrap(), max_raw);
    }

    #[test]
    fn cache_salt_validation_rejects_unsupported_characters() {
        let salt = Some("tenant salt".to_string());
        assert_eq!(
            validate_cache_namespace(&salt, false),
            Err("cache_salt contains unsupported characters")
        );
    }

    #[test]
    fn affinity_key_honors_both_client_conventions_in_priority_order() {
        use axum::http::HeaderMap;
        let hdr = |v: &str| {
            let mut h = HeaderMap::new();
            h.insert("x-session-id", v.parse().unwrap());
            h
        };
        let empty = HeaderMap::new();
        let s = |v: &str| Some(v.to_string());
        // each convention alone.
        assert_eq!(
            affinity_key(&s("explicit"), &None, &empty).unwrap(),
            s("explicit")
        );
        assert_eq!(
            affinity_key(&None, &s("openai-user"), &empty).unwrap(),
            s("openai-user")
        );
        assert_eq!(
            affinity_key(&None, &None, &hdr("hdr-id")).unwrap(),
            s("hdr-id")
        );
        // priority: session_id > user > header. Body beats header because a header can be
        // rewritten by an intermediary.
        assert_eq!(affinity_key(&s("a"), &s("b"), &hdr("c")).unwrap(), s("a"));
        assert_eq!(affinity_key(&None, &s("b"), &hdr("c")).unwrap(), s("b"));
        // blank/whitespace is ABSENT, not a key — a client sending "user": "" must not
        // collapse every conversation onto one shared session.
        assert_eq!(affinity_key(&s("  "), &s(""), &hdr("  ")).unwrap(), None);
        assert_eq!(affinity_key(&s(""), &s("real"), &empty).unwrap(), s("real"));
        // trimmed.
        assert_eq!(
            affinity_key(&s(" padded "), &None, &empty).unwrap(),
            s("padded")
        );
        // nothing supplied -> implicit tier (fingerprint) in the worker.
        assert_eq!(affinity_key(&None, &None, &empty).unwrap(), None);
        assert!(
            affinity_key(
                &s(&"x".repeat(MAX_CLIENT_IDENTIFIER_BYTES + 1)),
                &None,
                &empty,
            )
            .unwrap_err()
            .contains("at most")
        );
        assert!(
            affinity_key(&s("forged\nlog"), &None, &empty)
                .unwrap_err()
                .contains("control")
        );
    }

    #[test]
    fn affinity_key_plumbs_to_the_worker_request_on_both_bodies() {
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task", "session_id": "conv-1"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let key = affinity_key(&req.session_id, &req.user, &axum::http::HeaderMap::new()).unwrap();
        assert_eq!(
            build_request(&req, tx, lanes::Lane::Interactive, key)
                .affinity
                .as_deref(),
            Some("conv-1")
        );
        // OpenAI `user` on the chat body.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}],
            "user": "conv-2"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let key = affinity_key(&req.session_id, &req.user, &axum::http::HeaderMap::new()).unwrap();
        assert_eq!(
            build_chat_request(req, None, tx, lanes::Lane::Interactive, key)
                .unwrap()
                .request
                .affinity
                .as_deref(),
            Some("conv-2")
        );
        // absent on both -> None (implicit tier).
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        assert!(
            build_request(&req, tx, lanes::Lane::Interactive, None)
                .affinity
                .is_none()
        );
    }

    /// Drain an Sse response into its `data:` payload lines (keep-alive comments skipped).
    async fn sse_data_lines(resp: Response) -> Vec<String> {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    #[tokio::test]
    async fn chat_returns_reasoning_text_when_on_and_no_field_when_off() {
        // OWNER ACCEPTANCE GATE (2026-08-23, "also thinking content should be returned, not only
        // the content itself"): on the chat surface reasoning is delivered — non-streaming as
        // `message.reasoning` (+ `message.reasoning_details`), streaming as `delta.reasoning` —
        // and a reasoning-off generation carries NO reasoning field rather than an empty one.
        // Billing unchanged either way: reasoning tokens are output tokens.
        let feed = |think: bool| {
            let (tx, rx) = worker::event_channel();
            let body = if think {
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
            rx
        };
        // NON-STREAMING, reasoning on (the think-open prompt arms the splitter).
        let resp = blocking_response(
            feed(true),
            "m".into(),
            true,
            Vec::new(),
            Some(ToolStreamParser::reasoning_only()),
            Envelope::new(true),
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["reasoning"], "a plan");
        assert_eq!(
            v["choices"][0]["message"]["reasoning_details"][0]["text"],
            "a plan"
        );
        assert_eq!(v["choices"][0]["message"]["content"], "answer");
        // NON-STREAMING, reasoning off: the NoThink path builds no parser, and the response
        // carries no reasoning field at all.
        let resp = blocking_response(
            feed(false),
            "m".into(),
            true,
            Vec::new(),
            None,
            Envelope::new(true),
        )
        .await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            v["choices"][0]["message"].get("reasoning").is_none(),
            "a reasoning-off response must carry no reasoning field: {v}"
        );
        assert_eq!(v["choices"][0]["message"]["content"], "answer");
        // STREAMING, reasoning on: think text arrives as delta.reasoning, never as content.
        let resp = sse_response(
            feed(true),
            "m".into(),
            true,
            Some(ToolStreamParser::reasoning_only()),
            Envelope::new(true),
            Vec::new(),
            None,
        )
        .into_response();
        let lines = sse_data_lines(resp).await;
        let chunks: Vec<serde_json::Value> = lines[..lines.len() - 1]
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let reasoning: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["reasoning"].as_str())
            .collect();
        assert_eq!(
            reasoning, "a plan",
            "think text must stream as delta.reasoning"
        );
        let content: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, "answer", "content must exclude the think segment");
        // STREAMING, reasoning off: no delta carries a reasoning key.
        let resp = sse_response(
            feed(false),
            "m".into(),
            true,
            None,
            Envelope::new(true),
            Vec::new(),
            None,
        )
        .into_response();
        let lines = sse_data_lines(resp).await;
        for l in &lines[..lines.len() - 1] {
            let c: serde_json::Value = serde_json::from_str(l).unwrap();
            assert!(
                c["choices"][0]["delta"].get("reasoning").is_none(),
                "a reasoning-off stream must carry no reasoning deltas: {c}"
            );
        }
    }

    #[tokio::test]
    async fn stream_chunks_carry_envelope_and_first_delta_role() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "he".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "llo".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 2,
            n_prompt: 10,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let resp = sse_response(
            rx,
            "m".into(),
            true,
            None,
            Envelope::new(true),
            Vec::new(),
            None,
        )
        .into_response();
        let lines = sse_data_lines(resp).await;
        assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<serde_json::Value> = lines[..lines.len() - 1]
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // every chunk: id (chatcmpl-, SAME id) + created + system_fingerprint + object.
        let id = chunks[0]["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("chatcmpl-"));
        for c in &chunks {
            assert_eq!(c["id"], id.as_str());
            assert!(c["created"].as_u64().unwrap() > 1_700_000_000);
            let fingerprint = c["system_fingerprint"].as_str().unwrap();
            assert!(
                build_id::fingerprint_is_well_formed(fingerprint),
                "chunk system_fingerprint {fingerprint:?} is not memra-<version>-<12 hex>"
            );
            assert_eq!(c["object"], "chat.completion.chunk");
        }
        // FIRST delta carries role:"assistant" (SDK accumulator contract); later ones don't.
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "he");
        assert!(chunks[1]["choices"][0]["delta"].get("role").is_none());
        // final chunk: finish_reason + usage.
        let fin = chunks.last().unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "stop");
        assert_eq!(fin["usage"]["prompt_tokens"], 10);
    }

    #[tokio::test]
    async fn stream_token_events_equal_usage_on_every_finish_path() {
        for (stop_reason, expected_finish) in [
            ("Eos", "stop"),
            ("Callback", "stop"),
            ("MaxNew", "length"),
            ("ContextFull", "length"),
        ] {
            let (tx, rx) = worker::event_channel();
            // EOS deliberately has empty text: it is still one generated, streamed, and
            // accounted token id. This is the exact Q35 sellgate terminal-token case.
            tx.send(Event::Token {
                id: 248_046,
                text: String::new(),
            })
            .unwrap();
            tx.send(Event::Done {
                stop_reason: stop_reason.into(),
                n_tokens: 1,
                n_prompt: 8,
                n_cached: 8,
                elapsed_s: 0.1,
                spec: None,
            })
            .unwrap();
            drop(tx);

            let resp = sse_response(
                rx,
                "m".into(),
                true,
                None,
                Envelope::new(true),
                Vec::new(),
                None,
            )
            .into_response();
            let lines = sse_data_lines(resp).await;
            assert_eq!(lines.last().map(String::as_str), Some("[DONE]"));
            let chunks: Vec<serde_json::Value> = lines[..lines.len() - 1]
                .iter()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let token_events = chunks
                .iter()
                .filter(|chunk| chunk["choices"][0]["finish_reason"].is_null())
                .count();
            let terminal = chunks.last().unwrap();
            assert_eq!(token_events, 1, "{stop_reason} SSE token count");
            assert_eq!(terminal["usage"]["completion_tokens"], token_events);
            assert_eq!(terminal["choices"][0]["finish_reason"], expected_finish);
        }
    }

    #[tokio::test]
    async fn stream_excludes_stop_text_like_non_stream_does() {
        // gap-scan F9: the worker emits the delta BEFORE its stop check — the stream
        // shape must still exclude the stop text (and same-token overshoot) exactly
        // like the non-stream truncate. Stop spans two token events here.
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "answer\nPro".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "blem: leaked prompt".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Callback".into(),
            n_tokens: 2,
            n_prompt: 8,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let resp = sse_response(
            rx,
            "m".into(),
            true,
            None,
            Envelope::new(true),
            vec!["Problem:".into()],
            None,
        )
        .into_response();
        let lines = sse_data_lines(resp).await;
        let content: String = lines
            .iter()
            .filter(|l| *l != "[DONE]")
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|c| {
                c["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(content, "answer\n");

        // held-back text that never becomes a stop is flushed at Done.
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "ends in Pro".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Eos".into(),
            n_tokens: 1,
            n_prompt: 8,
            n_cached: 0,
            elapsed_s: 0.1,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let resp = sse_response(
            rx,
            "m".into(),
            true,
            None,
            Envelope::new(true),
            vec!["Problem:".into()],
            None,
        )
        .into_response();
        let lines = sse_data_lines(resp).await;
        let content: String = lines
            .iter()
            .filter(|l| *l != "[DONE]")
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|c| {
                c["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(content, "ends in Pro");
    }

    #[tokio::test]
    async fn stream_worker_error_is_a_data_chunk_not_a_named_event() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Error(worker::EngineError::engine("boom")))
            .unwrap();
        drop(tx);
        let resp = sse_response(
            rx,
            "m".into(),
            true,
            None,
            Envelope::new(true),
            Vec::new(),
            None,
        )
        .into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        // OpenAI clients only parse `data:` lines — no named `event: error` on the chat shape.
        assert!(
            !body.contains("event: error"),
            "named SSE event leaked: {body}"
        );
        let lines: Vec<&str> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .collect();
        let err: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(err["error"]["message"], "boom");
        assert_eq!(err["error"]["type"], "server_error");
        assert_eq!(err["error"]["code"], "engine_error");
        assert_eq!(lines.last(), Some(&"[DONE]"));
    }

    #[test]
    fn ttft_sse_marker_ignores_keepalive_comments() {
        assert!(!is_sse_data_frame(b": keep-alive\n\n"));
        assert!(is_sse_data_frame(b"data: {\"choices\":[]}\n\n"));
        assert!(is_sse_data_frame(
            b"event: error\ndata: {\"error\":\"failed\"}\n\n"
        ));
    }

    #[tokio::test]
    async fn error_bodies_use_the_openai_object_shape() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Error(worker::EngineError::model_not_found(
            "unknown model \"x\"",
        )))
        .unwrap();
        drop(tx);
        let response =
            blocking_response(rx, "m".into(), true, Vec::new(), None, Envelope::new(true)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // {"error": {message, type, param, code}} — the object every OpenAI SDK parses.
        assert_eq!(payload["error"]["message"], "unknown model \"x\"");
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["param"], "model");
        assert_eq!(payload["error"]["code"], "model_not_found");
    }

    // ---- G6 taxonomy (lane/serve-hardening) --------------------------------------------
    //
    // The mapping is the deliverable, so it is asserted class by class rather than through
    // one happy-path example. Before this lane EVERY row below answered 400
    // invalid_request_error, which no OpenAI-compatible SDK retries.

    fn retry_after(resp: &Response) -> Option<String> {
        resp.headers()
            .get(axum::http::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    }

    // ---- timeout_ms + deadline-aware admission (lane/deadline-billing-20260823) ------

    async fn body_value(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    /// POST one chat request through the FULL handler, retrying the server's contention
    /// refusals until the request is actually ADMITTED.
    ///
    /// `reserve_pending_admit` reads the process-global lane backlog
    /// (`worker::ADMISSION_RESERVATIONS`) and the test runner is parallel: any sibling
    /// test's in-flight reservation window puts `backlog > 0` under this request, and
    /// with a fresh state's empty metrics the queue-wait estimate is the 2 s static —
    /// more than the minimum 1000 ms deadline these tests declare, so the request sheds
    /// 429 `shed_deadline` before admission. Schedule-dependent and load-amplified: on a
    /// loaded box the windows stretch, and the deadline tests observed 429 where they
    /// asserted 408 (the 2026-09-01 accrace flake). The shed is the server's documented,
    /// unbilled refusal-under-load — so the honest test answer is to treat it as "try
    /// again", never as the outcome: the caller's assertions still require the ADMITTED
    /// request to prove its 408/billing contract, and a 429 that is not a shed stays a
    /// loud failure.
    async fn chat_completion_admitted(st: &AppState, req: serde_json::Value) -> Response {
        let mut last_shed = serde_json::Value::Null;
        for _ in 0..50 {
            let resp = chat_completions(
                State(st.clone()),
                HeaderMap::new(),
                None,
                Json(serde_json::from_value(req.clone()).unwrap()),
            )
            .await;
            if resp.status() != StatusCode::TOO_MANY_REQUESTS {
                return resp;
            }
            let body = body_value(resp).await;
            let code = body["error"]["code"].as_str().unwrap_or_default();
            assert!(
                code.starts_with("shed_"),
                "only a contention shed may be retried; any other 429 is a finding: {body}"
            );
            last_shed = body;
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        // The shed message names the estimate and the remaining deadline, so triage can
        // tell a genuinely saturated run from a shed regression that never clears.
        panic!(
            "still shed after 50 attempts — either load the retry budget cannot absorb \
             or a shed that no longer clears; last refusal: {last_shed}"
        );
    }

    #[test]
    fn timeout_ms_parses_clamps_nothing_and_names_every_refusal() {
        // Absent / explicit null => the DOCUMENTED default, not "no deadline".
        assert_eq!(parse_timeout_ms(None).unwrap(), TIMEOUT_MS_DEFAULT);
        assert_eq!(
            parse_timeout_ms(Some(&serde_json::Value::Null)).unwrap(),
            TIMEOUT_MS_DEFAULT
        );
        // In-range values are honored EXACTLY (no clamping — an out-of-range value is a
        // refusal, because silently shortening a caller's deadline is the accepted-and-
        // ignored class the standard-surface law bans).
        for ms in [TIMEOUT_MS_MIN, 5_000, 45_000, TIMEOUT_MS_MAX] {
            assert_eq!(parse_timeout_ms(Some(&json!(ms))).unwrap(), ms);
        }
        // Out of range both ways: named 400 stating the range AND the streaming hatch.
        for bad in [0u64, TIMEOUT_MS_MIN - 1, TIMEOUT_MS_MAX + 1, 600_000] {
            let err = parse_timeout_ms(Some(&json!(bad))).expect_err("out of range must refuse");
            assert!(err.contains("timeout_ms"), "{err}");
            assert!(
                err.contains(&TIMEOUT_MS_MIN.to_string())
                    && err.contains(&TIMEOUT_MS_MAX.to_string()),
                "the message must state the range: {err}"
            );
            assert!(
                err.contains("stream"),
                "the message must point at streaming for longer work: {err}"
            );
        }
        // Unknown types refuse too (never a silent default).
        for bad in [json!("30s"), json!(1.5), json!(true), json!({}), json!([])] {
            let err = parse_timeout_ms(Some(&bad)).expect_err("bad type must refuse");
            assert!(
                err.contains("timeout_ms") && err.contains("stream"),
                "{err}"
            );
        }
        // Negative numbers are not u64 — same named refusal, not a panic.
        assert!(parse_timeout_ms(Some(&json!(-1))).is_err());
    }

    /// The named 400 is IDENTICAL on all four surfaces (standard-surface law) and costs
    /// neither a slot nor a ledger receipt.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_bad_timeout_ms_is_the_same_named_400_on_every_surface() {
        let _l = drain_lock();
        let st = fake_worker_state();

        let comp = completions(
            State(st.clone()),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m", "prompt": "t", "timeout_ms": 90_001}))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(comp.status(), StatusCode::BAD_REQUEST);
        let chat = chat_completions(
            State(st.clone()),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}],
                    "timeout_ms": 90_001}))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(chat.status(), StatusCode::BAD_REQUEST);
        let resp_api = responses_api::responses(
            State(st.clone()),
            HeaderMap::new(),
            None,
            axum::body::Bytes::from(
                json!({"model": "m", "input": "t", "timeout_ms": 90_001}).to_string(),
            ),
        )
        .await;
        assert_eq!(resp_api.status(), StatusCode::BAD_REQUEST);
        let msgs = anthropic::messages(
            State(st.clone()),
            HeaderMap::new(),
            None,
            axum::body::Bytes::from(
                json!({"model": "m", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "t"}],
                       "timeout_ms": 90_001})
                .to_string(),
            ),
        )
        .await;
        assert_eq!(msgs.status(), StatusCode::BAD_REQUEST);

        // OpenAI-shaped surfaces name the param; all four name the field in the message.
        for (surface, resp) in [
            ("/v1/completions", comp),
            ("/v1/chat/completions", chat),
            ("/v1/responses", resp_api),
        ] {
            let body = body_value(resp).await;
            assert_eq!(body["error"]["type"], "invalid_request_error", "{surface}");
            assert_eq!(body["error"]["param"], "timeout_ms", "{surface}");
            let m = body["error"]["message"].as_str().unwrap();
            assert!(
                m.contains("90000") && m.contains("stream"),
                "{surface}: {m}"
            );
        }
        // Anthropic shape: no param slot, so the message carries it.
        let body = body_value(msgs).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let m = body["error"]["message"].as_str().unwrap();
        assert!(m.contains("timeout_ms") && m.contains("stream"), "{m}");
    }

    /// Wrong TYPE refuses too — the reasoning-schema philosophy, one surface shown end to
    /// end (the parser gate above covers the type matrix).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_non_integer_timeout_ms_is_a_named_400() {
        let _l = drain_lock();
        let st = fake_worker_state();
        let resp = chat_completions(
            State(st),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}],
                    "timeout_ms": "30s"}))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = body_value(resp).await;
        assert_eq!(body["error"]["param"], "timeout_ms");
    }

    /// NON-STREAMING deadline: the response delivers the partial with our standard error
    /// object (`code: "deadline_exceeded"`), generation is CANCELLED (the worker's channel
    /// is closed — observed via the receiver the fake worker holds), and the receipt
    /// settles through `complete_deadline_partial` with the delivered counts — the
    /// census-distinct billable outcome, never plain `complete`.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_missed_non_stream_deadline_delivers_the_partial_bills_it_and_cancels_generation() {
        let _l = drain_lock();
        // A worker that publishes prompt usage and ONE token, then never finishes — the
        // shape a real deadline miss has (work done, no terminal event in time). It keeps
        // the request's sender so the handler's drop of rx is observable as a closed
        // channel: that closure IS the cancel signal the worker acts on at its next tick.
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        let cancel_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_cancel = cancel_seen.clone();
        let health = health::WorkerHealth::new();
        let h = health.clone();
        std::thread::spawn(move || {
            h.mark_ready();
            while let Ok(Cmd::Generate(req)) = cmd_rx.recv() {
                worker::release_pending_admit();
                worker::release_admission_reservation(req.lane);
                let _ = req.tx.send(Event::PromptUsage {
                    n_prompt: 1,
                    n_cached: 0,
                });
                let _ = req.tx.send(Event::Token {
                    id: 1,
                    text: "partial".into(),
                });
                // The abort signal a real worker watches for at every tick: the request's
                // event channel closing. Set the flag the test polls when it appears.
                for _ in 0..5_000 {
                    if req.tx.is_closed() {
                        worker_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });
        for _ in 0..2_000 {
            if health.live().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut st = fake_worker_state();
        st.cmd_tx = cmd_tx;
        st.health = health;
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());

        let resp = chat_completion_admitted(
            &st,
            json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}],
                "timeout_ms": 1_000}),
        )
        .await;

        // CONTRACT CHANGED 2026-08-26 (owner report: a 30k-token non-streaming request
        // timed out). This used to assert a 408 with the generated tokens DISCARDED. The
        // deadline now DELIVERS what was produced, because throwing away 90 s of a
        // customer's tokens to answer an error is the bug, not the safety valve.
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_value(resp).await;
        assert!(
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("partial"),
            "the tokens generated before the cut must be delivered: {body}"
        );
        // OpenRouter dialect, and deliberately NOT finish_reason "length": no provider's
        // finish-reason enum has a time value, so reporting a time cut as "length" would
        // tell the caller to ask for more tokens when the truth is that it must stream.
        assert_eq!(body["choices"][0]["finish_reason"], "error");
        assert_eq!(
            body["choices"][0]["native_finish_reason"],
            "deadline_exceeded"
        );
        assert_eq!(body["error"]["code"], "deadline_exceeded");
        assert_eq!(body["error"]["metadata"]["error_type"], "timeout");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("1000") && message.contains("stream"),
            "the partial must name the deadline and the streaming alternative: {message}"
        );
        assert_eq!(body["usage"]["completion_tokens"], 1);

        // GENERATION CANCELLED: the worker saw its event channel close. Polled with an
        // AWAIT (not a blocking recv): the event forwarder that owns the worker-side
        // receiver is a tokio task, and a blocking wait on this single-threaded test
        // runtime would starve the very task whose exit closes the channel.
        let mut cancelled = false;
        for _ in 0..500 {
            if cancel_seen.load(std::sync::atomic::Ordering::SeqCst) {
                cancelled = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            cancelled,
            "the deadline must CANCEL generation (worker's event channel closed)"
        );

        // SEAM: the delivered tokens settle through the census-distinct terminal —
        // `complete_deadline_partial`, never plain `complete`. Writing `completed` here
        // (the first version of this lane) lost the deadline everywhere except an
        // ephemeral log line — a review caught it.
        let events = mock.events();
        assert!(
            events.contains(&MeterEvent::DeadlinePartial {
                prompt: 1,
                cached: 0,
                completion: 1,
            }),
            "the partial must settle as a deadline-partial with worker-truth counts: {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, MeterEvent::Complete { .. })),
            "a deadline cut must stay distinguishable from a full answer: {events:?}"
        );
    }

    /// The other half of the same contract: a deadline that lands with NOTHING generated
    /// still answers 408 and still bills zero. There is no partial to deliver, so the
    /// original promise ("we answer inside the deadline or you don't pay") stands.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_deadline_missed_before_any_token_is_still_408_and_unbilled() {
        let _l = drain_lock();
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        let health = health::WorkerHealth::new();
        let h = health.clone();
        std::thread::spawn(move || {
            h.mark_ready();
            // Prompt usage only: admitted, prefilling, and NOT ONE token emitted before
            // the deadline — the shape of a prompt too large to prefill in the window.
            while let Ok(Cmd::Generate(req)) = cmd_rx.recv() {
                worker::release_pending_admit();
                worker::release_admission_reservation(req.lane);
                let _ = req.tx.send(Event::PromptUsage {
                    n_prompt: 1,
                    n_cached: 0,
                });
                for _ in 0..5_000 {
                    if req.tx.is_closed() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });
        for _ in 0..2_000 {
            if health.live().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut st = fake_worker_state();
        st.cmd_tx = cmd_tx;
        st.health = health;
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());
        let resp = chat_completion_admitted(
            &st,
            json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}],
                "timeout_ms": 1_000}),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
        // Still retryable, still no invented Retry-After.
        assert!(resp.headers().get("x-should-retry").is_none());
        assert_eq!(retry_after(&resp), None);
        let body = body_value(resp).await;
        assert_eq!(body["error"]["code"], "deadline_exceeded");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not billed"),
            "the zero-token 408 keeps the billing promise: {body}"
        );
        let events = mock.events();
        assert!(
            events.contains(&MeterEvent::Unbilled {
                outcome: "deadline_exceeded",
                status: 408,
                code: "deadline_exceeded".into(),
            }),
            "the named zero-debit census outcome, not the generic reject — every sibling \
             deadline path settles this one: {events:?}"
        );
    }

    /// STREAMING, deadline MISSED before the first token: still a pre-header 408 and no
    /// bill — nothing was delivered, so there is nothing to charge for.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_stream_that_misses_ttft_is_a_preheader_408_and_not_billed() {
        let _l = drain_lock();
        // Admits (publishes prompt usage) but produces NO token — a prefill that overruns.
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        let health = health::WorkerHealth::new();
        let h = health.clone();
        std::thread::spawn(move || {
            h.mark_ready();
            while let Ok(Cmd::Generate(req)) = cmd_rx.recv() {
                worker::release_pending_admit();
                worker::release_admission_reservation(req.lane);
                let _ = req.tx.send(Event::PromptUsage {
                    n_prompt: 1,
                    n_cached: 0,
                });
                while !req.tx.is_closed() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });
        for _ in 0..2_000 {
            if health.live().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut st = fake_worker_state();
        st.cmd_tx = cmd_tx;
        st.health = health;
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());

        let resp = chat_completion_admitted(
            &st,
            json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}],
                "stream": true, "timeout_ms": 1_000}),
        )
        .await;
        // PRE-HEADER: a real status, not a 200 with an error chunk — the whole reason the
        // TTFT peek exists (a committed 200 leaves no status for a router to act on).
        assert_eq!(resp.status(), StatusCode::REQUEST_TIMEOUT);
        let body = body_value(resp).await;
        assert_eq!(body["error"]["code"], "deadline_exceeded");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("first token"),
            "the streaming message must say the deadline bounded TIME TO FIRST TOKEN: {body}"
        );
        let events = mock.events();
        assert!(
            events.contains(&MeterEvent::Unbilled {
                outcome: "deadline_exceeded",
                status: 408,
                code: "deadline_exceeded".into(),
            }),
            "a TTFT miss must settle unbilled under the deadline outcome: {events:?}"
        );
    }

    /// STREAMING, first token DELIVERED inside the deadline: the parameter is SPENT. A
    /// stream whose remaining tokens take longer than timeout_ms still completes and
    /// bills in full — post-first-token immunity, the other half of the streaming rule.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_stream_is_immune_to_the_deadline_after_its_first_token() {
        let _l = drain_lock();
        // 4 tokens, 400ms apart: the first arrives well inside a 1s deadline and the
        // stream then runs ~1.6s — past it. The stream must still finish normally.
        let mut st = fake_worker_state_with_steps(4, std::time::Duration::from_millis(400));
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());
        let resp = chat_completion_admitted(
            &st,
            json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}],
                "stream": true, "timeout_ms": 1_000}),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "TTFT was met — 200 is correct"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("the stream must run to completion past the deadline");
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("[DONE]"), "stream did not complete: {text}");
        let events = mock.events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, MeterEvent::Complete { completion: 4, .. })),
            "a stream past its deadline after first token still settles as COMPLETE with \
             all four tokens: {events:?}"
        );
    }

    /// `worker::ADMISSION_RESERVATIONS` / `worker::PENDING_ADMITS` are PROCESS GLOBALS and
    /// the test runner is parallel: two admission tests pumping the same lane counter race,
    /// and the loser reads the winner's swapped value (caught live in a co-tenant-loaded
    /// local-ci window 2026-08-30 — `deadline_shed_is_interactive_only...` shed on a free
    /// slot because a sibling had the interactive counter at max_queue_depth for that
    /// instant). Every test that WRITES these counters serializes here.
    fn admission_counters_guard() -> std::sync::MutexGuard<'static, ()> {
        static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());
        COUNTERS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Put an admission counter back on DROP — including the drop that unwinds a failed
    /// assertion. The swap tests below used to restore with a trailing `store(prev)`
    /// AFTER their asserts, so one red left the process-global lane backlog pinned at the
    /// swapped value (e.g. max_queue_depth) and every later-admitted request in the run
    /// shed 429 — the 2026-09-01 one-flake-becomes-21-reds cascade, counter form.
    struct CounterRestore<'a>(&'a std::sync::atomic::AtomicUsize, usize);
    impl Drop for CounterRestore<'_> {
        fn drop(&mut self) {
            self.0.store(self.1, std::sync::atomic::Ordering::Release);
        }
    }

    /// `reserve_pending_admit` on the interactive lane, retrying through the TRANSIENT
    /// contention shed: the lane backlog is a process-global reading
    /// (`worker::ADMISSION_RESERVATIONS`) and the runner is parallel, so a sibling
    /// handler test's in-flight reservation puts `backlog > 0` for an instant and the
    /// wait estimate then deadline-sheds a tight deadline — schedule-dependent,
    /// load-amplified (the 2026-09-01 class). A PERSISTENT shed is not contention and
    /// still fails the caller's assert: whatever pins the backlog for all 50 attempts
    /// (e.g. a cross-lane leak) is a finding. Any refusal other than the deadline shed
    /// panics immediately.
    #[allow(clippy::result_large_err)] // allow: passes reserve_pending_admit's own contract through unchanged
    fn reserve_interactive_through_contention(
        st: &AppState,
        rl: &RateLimit,
        deadline_ms: u64,
    ) -> Result<PendingAdmissionGuard, (Response, &'static str)> {
        let reserve = || {
            reserve_pending_admit(
                st,
                lanes::Lane::Interactive,
                rl,
                RequestDeadline::starting_now(deadline_ms),
            )
        };
        let mut g = reserve();
        for _ in 0..50 {
            match &g {
                Ok(_) => break,
                Err((_, "shed_deadline")) => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    g = reserve();
                }
                Err((_, outcome)) => panic!("unexpected refusal: {outcome}"),
            }
        }
        g
    }

    /// BACKPRESSURE, absolute bound: at MEMRA_MAX_QUEUE_DEPTH the request sheds with 429 +
    /// Retry-After, outcome `shed_queue`, no bill, X-RateLimit trio present.
    #[test]
    fn the_queue_bound_sheds_with_429_retry_after_and_the_ratelimit_trio() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        let prev = counter.swap(max_queue_depth(cap), std::sync::atomic::Ordering::AcqRel);
        let _restore = CounterRestore(counter, prev);
        let rl = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 1,
        };
        let (resp, outcome) = reserve_pending_admit(
            &st,
            lane,
            &rl,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        )
        .map(|_| ())
        .expect_err("a backlog at the bound must shed");
        assert_eq!(outcome, "shed_queue");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            retry_after(&resp).is_some(),
            "a shed must carry Retry-After so the router's spill can act on it"
        );
        // The trio rides the shed exactly like every other 429 on this surface.
        let stamped = rl.attach(resp);
        for h in [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
        ] {
            assert!(stamped.headers().get(h).is_some(), "missing {h}");
        }
    }

    /// BACKPRESSURE, deadline test: the SAME loaded lane admits a request whose deadline
    /// can absorb the estimated wait and sheds one whose deadline cannot — the shed is
    /// keyed on the caller's own deadline, not on load alone.
    #[test]
    fn admission_sheds_only_when_the_estimated_wait_cannot_fit_the_deadline() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 1_000;
            m.step_p50_ms = 10.0; // mean service ~1s
        }
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        let prev = counter.swap(cap, std::sync::atomic::Ordering::AcqRel); // one wave ahead
        let _restore = CounterRestore(counter, prev);
        let rl = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 1,
        };
        // A 90s deadline absorbs a ~2s wait: ADMIT (never shed a request that can wait).
        let admitted = reserve_pending_admit(
            &st,
            lane,
            &rl,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        );
        assert!(
            admitted.is_ok(),
            "a request whose deadline covers the estimate must be admitted"
        );
        drop(admitted); // release the reservation the admit took
        // A 1s deadline cannot: SHED, with the estimate as Retry-After.
        let (resp, outcome) = reserve_pending_admit(
            &st,
            lane,
            &rl,
            RequestDeadline::starting_now(TIMEOUT_MS_MIN),
        )
        .map(|_| ())
        .expect_err("a deadline shorter than the estimated wait must shed");
        assert_eq!(outcome, "shed_deadline");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(retry_after(&resp).is_some());
    }

    /// Free capacity never deadline-sheds, and neither do the dark lanes (they shed at cap
    /// inside the worker — the deadline gate here is interactive-only by design).
    #[test]
    fn deadline_shed_is_interactive_only_and_silent_with_free_slots() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let cap = lane_cap(lanes::Lane::Interactive);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 100_000; // an enormous estimate...
            m.step_p50_ms = 100.0;
        }
        // ...but a free slot and an empty lane mean no wait to estimate.
        let free = RateLimit {
            limit: cap,
            remaining: 1,
            reset_s: 0,
        };
        // Retried through the transient sibling-reservation shed (see the helper): this
        // enormous estimate sheds even the minimum deadline whenever the process-global
        // backlog reads > 0 for an instant. The assertion still requires the free-slot
        // admit to prove itself.
        let g = reserve_interactive_through_contention(&st, &free, TIMEOUT_MS_MIN);
        assert!(
            g.is_ok(),
            "free capacity must admit regardless of the estimate"
        );
        drop(g);
        // Loaded, but a dark-lane request: the worker's own lane gate owns those, and the
        // deadline shed must not fire off the interactive lane.
        let full = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 5,
        };
        for lane in [lanes::Lane::Judge, lanes::Lane::Harvest] {
            let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
            let prev = counter.swap(1, std::sync::atomic::Ordering::AcqRel); // backlog > 0
            let _restore = CounterRestore(counter, prev);
            let g = reserve_pending_admit(
                &st,
                lane,
                &full,
                RequestDeadline::starting_now(TIMEOUT_MS_MIN),
            );
            assert!(
                g.is_ok(),
                "{lane:?} must not be deadline-shed by the interactive gate"
            );
            drop(g);
        }
    }

    /// THE DEFECT SHAPE, kept as the flag-off contract (darklanes#5; prod measured
    /// 2026-09-01: 133-137 s of pre-header silence, never a 429). The engine queue is
    /// saturated (a full wave of reservations ahead), the HTTP lane still has slots,
    /// and the caller's deadline can absorb the estimated wait: no arm sheds, the
    /// request queues silently. With `MEMRA_QUEUE_WAIT_CEILING_S` absent or 0 this is
    /// today's behavior byte-for-byte, and this test is what holds that line.
    #[test]
    fn a_saturated_queue_with_free_http_slots_queues_silently_without_a_ceiling() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 1_000; // mean 100 tok/request...
            m.step_p50_ms = 100.0; // ...x 100 ms = ~10 s/wave; one wave ahead => ~20 s
        }
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        let prev = counter.swap(cap, std::sync::atomic::Ordering::AcqRel); // one wave ahead
        let _restore = CounterRestore(counter, prev);
        // The HTTP lane is NOT full: a free slot remains, but the wave ahead means this
        // request still waits ~20 s for engine capacity.
        let free = RateLimit {
            limit: cap,
            remaining: 1,
            reset_s: 0,
        };
        let g = reserve_pending_admit(
            &st,
            lane,
            &free,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        );
        assert!(
            g.is_ok(),
            "flag off: a ~20 s projected wait whose deadline can absorb it queues \
             silently (no 429) - the darklanes#5 defect shape, preserved by default"
        );
        drop(g);
    }

    /// QUEUE-WAIT CEILING, shed arm: the exact defect shape above (saturated engine
    /// queue, free HTTP slot, patient deadline), but with a ceiling below the estimate:
    /// 429, `code: shed_queue_wait`, Retry-After = the estimate (with its ms twin), and
    /// the X-RateLimit trio rides the shed like every other 429 on this surface.
    #[test]
    fn the_queue_wait_ceiling_sheds_with_429_retry_after_and_the_ratelimit_trio() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 1_000; // mean 100 tok/request...
            m.step_p50_ms = 100.0; // ...x 100 ms = ~10 s/wave; one wave ahead => ~20 s
        }
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        let prev = counter.swap(cap, std::sync::atomic::Ordering::AcqRel); // one wave ahead
        let _restore = CounterRestore(counter, prev);
        let free = RateLimit {
            limit: cap,
            remaining: 1,
            reset_s: 0,
        };
        let (resp, outcome) = reserve_pending_admit_with_ceiling(
            &st,
            lane,
            &free,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
            5, // ceiling 5 s, estimate ~20 s
        )
        .map(|_| ())
        .expect_err("a projected wait past the ceiling must shed");
        assert_eq!(outcome, "shed_queue_wait");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            retry_after(&resp).as_deref(),
            Some("20"),
            "Retry-After must carry the estimate (~10 s/wave x 2 waves)"
        );
        assert_eq!(
            resp.headers()
                .get("retry-after-ms")
                .and_then(|v| v.to_str().ok()),
            Some("20000"),
            "the ms twin must match"
        );
        let stamped = free.attach(resp);
        for h in [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
        ] {
            assert!(stamped.headers().get(h).is_some(), "missing {h}");
        }
    }

    /// QUEUE-WAIT CEILING, admit arm + lane scope: an estimate UNDER the ceiling still
    /// queues exactly as before (the ceiling is a ceiling, not a load switch), and the
    /// dark lanes are never judged by it (the worker's own lane gate owns those).
    #[test]
    fn the_queue_wait_ceiling_admits_under_it_and_never_touches_dark_lanes() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 1_000;
            m.step_p50_ms = 100.0; // ~10 s/wave; one wave ahead => ~20 s
        }
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        let prev = counter.swap(cap, std::sync::atomic::Ordering::AcqRel);
        let _restore = CounterRestore(counter, prev);
        let free = RateLimit {
            limit: cap,
            remaining: 1,
            reset_s: 0,
        };
        let g = reserve_pending_admit_with_ceiling(
            &st,
            lane,
            &free,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
            60, // ceiling 60 s, estimate ~20 s
        );
        assert!(
            g.is_ok(),
            "an estimate under the ceiling must admit and queue as before"
        );
        drop(g);
        // Dark lanes: a backlog and a 1 s ceiling, and still no shed from this gate.
        let full = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 5,
        };
        for dark in [lanes::Lane::Judge, lanes::Lane::Harvest] {
            let counter = &worker::ADMISSION_RESERVATIONS[dark.idx()];
            let prev = counter.swap(1, std::sync::atomic::Ordering::AcqRel); // backlog > 0
            let _restore = CounterRestore(counter, prev);
            let g = reserve_pending_admit_with_ceiling(
                &st,
                dark,
                &full,
                RequestDeadline::starting_now(TIMEOUT_MS_MAX),
                1,
            );
            assert!(
                g.is_ok(),
                "{dark:?} must not be shed by the interactive queue-wait ceiling"
            );
            drop(g);
        }
    }

    /// QUEUE-WAIT CEILING, arm precedence: with the ceiling set, the existing arms still
    /// answer first and unchanged. A backlog at the absolute bound stays `shed_queue`;
    /// a deadline shorter than the estimate stays `shed_deadline`.
    #[test]
    fn the_queue_wait_ceiling_leaves_the_existing_shed_arms_first_and_unchanged() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let lane = lanes::Lane::Interactive;
        let cap = lane_cap(lane);
        {
            let mut m = st.metrics.lock().unwrap();
            m.completed = 10;
            m.tokens_out = 1_000;
            m.step_p50_ms = 100.0;
        }
        let rl = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 1,
        };
        let counter = &worker::ADMISSION_RESERVATIONS[lane.idx()];
        // At the absolute bound: shed_queue wins even with a 1 s ceiling armed.
        let prev = counter.swap(max_queue_depth(cap), std::sync::atomic::Ordering::AcqRel);
        let _restore = CounterRestore(counter, prev);
        assert!(matches!(
            reserve_pending_admit_with_ceiling(
                &st,
                lane,
                &rl,
                RequestDeadline::starting_now(TIMEOUT_MS_MAX),
                1,
            ),
            Err((_, "shed_queue"))
        ));
        // Below the bound with a too-short deadline: shed_deadline wins over the ceiling.
        counter.store(cap, std::sync::atomic::Ordering::Release);
        assert!(matches!(
            reserve_pending_admit_with_ceiling(
                &st,
                lane,
                &rl,
                RequestDeadline::starting_now(TIMEOUT_MS_MIN),
                1,
            ),
            Err((_, "shed_deadline"))
        ));
    }

    /// QUEUE-WAIT CEILING wiring: the production wrapper feeds the OnceLock env read into
    /// the judged path (wiring-assertions law: anchored on the INVOCATION in
    /// comment-stripped text, scoped to the wrapper body so this test's own literals
    /// cannot satisfy it).
    #[test]
    fn the_queue_wait_ceiling_is_wired_through_the_production_wrapper() {
        let src = include_str!("lib.rs");
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        let start = code
            .find("pub(crate) fn reserve_pending_admit(")
            .expect("the production wrapper exists");
        let rest = &code[start..];
        let end = rest.find("\nfn ").unwrap_or(rest.len());
        let wrapper = &rest[..end];
        assert!(
            wrapper.contains(
                "reserve_pending_admit_with_ceiling(st, lane, rl, deadline, queue_wait_ceiling_s())"
            ),
            "every production ingress must judge the ceiling the env read armed"
        );
    }

    #[test]
    fn pending_admission_reservation_is_atomic_and_rolls_back_on_drop() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let cap = lane_cap(lanes::Lane::Interactive);
        let bound = max_queue_depth(cap);
        assert!(bound > 0, "the queue bound must admit at least one request");
        let rl = RateLimit {
            limit: cap,
            remaining: 0,
            reset_s: 1,
        };
        let _ = worker::PENDING_ADMITS.fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |_| Some(0),
        );
        let counter = &worker::ADMISSION_RESERVATIONS[lanes::Lane::Interactive.idx()];
        let _restore = CounterRestore(counter, 0);
        counter.store(bound - 1, std::sync::atomic::Ordering::Release);
        let guard = reserve_pending_admit(
            &st,
            lanes::Lane::Interactive,
            &rl,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        )
        .expect("the final queue slot should be reservable");
        assert_eq!(
            worker::PENDING_ADMITS.load(std::sync::atomic::Ordering::Acquire),
            1
        );
        assert_eq!(counter.load(std::sync::atomic::Ordering::Acquire), bound);
        drop(guard);
        assert_eq!(
            worker::PENDING_ADMITS.load(std::sync::atomic::Ordering::Acquire),
            0
        );
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Acquire),
            bound - 1
        );

        counter.store(bound, std::sync::atomic::Ordering::Release);
        let rejected = reserve_pending_admit(
            &st,
            lanes::Lane::Interactive,
            &rl,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        );
        assert!(matches!(rejected, Err((_, "shed_queue"))));
    }

    #[test]
    fn admission_reservations_are_lane_scoped() {
        let _counters = admission_counters_guard();
        let st = fake_worker_state();
        let harvest = lanes::Lane::Harvest;
        let interactive = lanes::Lane::Interactive;
        let harvest_counter = &worker::ADMISSION_RESERVATIONS[harvest.idx()];
        let interactive_counter = &worker::ADMISSION_RESERVATIONS[interactive.idx()];
        let _restore = CounterRestore(harvest_counter, 0);
        harvest_counter.store(
            max_queue_depth(lane_cap(harvest)),
            std::sync::atomic::Ordering::Release,
        );
        interactive_counter.store(0, std::sync::atomic::Ordering::Release);
        let free = RateLimit {
            limit: lane_cap(interactive),
            remaining: 1,
            reset_s: 0,
        };
        // Two arms, because the harvest bound (max_queue_depth of its cap 8 = 32) is far
        // below every interactive threshold: a cross-lane backlog leak (a lane.idx()
        // slip in reserve_pending_admit) would put 32 on the interactive reading — never
        // enough for its shed_queue bound (256), and only 2 s of estimated wait. So the
        // MAX arm proves the path is open, and the MIN arm is the teeth: with the leak,
        // that pinned 2 s estimate deadline-sheds a 1000 ms request on EVERY attempt and
        // outlasts the retry budget; healthy, backlog 0 + a free slot admits with no
        // estimate applied at all. The retry absorbs only the TRANSIENT sibling
        // reservation (load-flaked run 2 of the 2026-09-01 triple), which clears between
        // attempts — the harvest counter this test pins does not.
        let guard = reserve_pending_admit(
            &st,
            interactive,
            &free,
            RequestDeadline::starting_now(TIMEOUT_MS_MAX),
        )
        .expect("a full harvest queue must not consume interactive capacity");
        drop(guard);
        let tight = reserve_interactive_through_contention(&st, &free, TIMEOUT_MS_MIN);
        assert!(
            tight.is_ok(),
            "a full harvest queue must not deadline-shed a tight interactive request \
             (a backlog that outlasts the retry budget here is a cross-lane leak, not \
             contention)"
        );
        drop(tight);
        let harvest_rl = RateLimit {
            limit: lane_cap(harvest),
            remaining: 0,
            reset_s: 1,
        };
        assert!(matches!(
            reserve_pending_admit(
                &st,
                harvest,
                &harvest_rl,
                RequestDeadline::starting_now(TIMEOUT_MS_MAX)
            ),
            Err((_, "shed_queue"))
        ));
    }

    #[test]
    fn taxonomy_maps_every_class_to_its_status_and_code() {
        use worker::{EngineError as E, ErrClass as C};
        let cases: Vec<(worker::EngineError, StatusCode, &str, &str)> = vec![
            (
                E::invalid_param("bad json", "response_format"),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "",
            ),
            (
                E::context_length("prompt (9000 tok) >= context cap (8192)"),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "context_length_exceeded",
            ),
            (
                E::model_not_found("unknown model \"nope\""),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "model_not_found",
            ),
            (
                E::rate_limit("lane judge is at capacity, retry"),
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "rate_limit_exceeded",
            ),
            (
                E::overloaded("no VRAM for a new session"),
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "overloaded",
            ),
            (
                E::engine("graph step failed: launch error"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "engine_error",
            ),
        ];
        for (err, want_status, want_type, want_code) in cases {
            let (status, etype, code) = class_http(err.class);
            assert_eq!(status, want_status, "{:?}", err);
            assert_eq!(etype, want_type, "{:?}", err);
            if !want_code.is_empty() {
                assert_eq!(code, Some(want_code), "{:?}", err);
            }
            // the rendered body agrees with the mapping
            let body = engine_error_body(&err);
            assert_eq!(body["error"]["message"], err.message);
            assert_eq!(body["error"]["type"], want_type);
        }
        // and no class is silently missing from the match
        for c in [
            C::InvalidRequest,
            C::ContextLength,
            C::ModelNotFound,
            C::RateLimit,
            C::Overloaded,
            C::Engine,
        ] {
            let (s, t, _) = class_http(c);
            assert!(s.is_client_error() || s.is_server_error(), "{c:?} -> {s}");
            assert!(!t.is_empty());
        }
    }

    #[test]
    fn a_cuda_oom_message_is_capacity_503_not_a_500() {
        // The one deliberate text rule: the driver's own OOM text promotes an engine fault to
        // Overloaded, because the box ran out of VRAM (a retryable capacity condition) rather
        // than hitting a bug. Same predicate the step-OOM park path uses, so the two paths
        // cannot disagree about what an OOM is.
        let e = worker::EngineError::engine(
            "step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")",
        );
        let resp = engine_error_response(&e);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry_after(&resp).as_deref(), Some("5"));
    }

    #[test]
    fn retry_headers_follow_the_sdk_contract() {
        // openai-python reads retry-after-ms FIRST, then retry-after, and ABANDONS the retry
        // if the delay exceeds 120 s; litellm honors retry-after only for 0 < v <= 60. So:
        // integer seconds, <= 60, with a matching millisecond twin.
        for e in [
            worker::EngineError::rate_limit("shed"),
            worker::EngineError::overloaded("no VRAM"),
        ] {
            let resp = engine_error_response(&e);
            let ra = retry_after(&resp).expect("retryable class must carry Retry-After");
            let secs: u64 = ra
                .parse()
                .expect("Retry-After must be integer delay-seconds");
            assert!(
                secs > 0 && secs <= 60,
                "Retry-After {secs}s outside the honored window"
            );
            let ms = resp
                .headers()
                .get("retry-after-ms")
                .unwrap()
                .to_str()
                .unwrap();
            assert_eq!(
                ms.parse::<u64>().unwrap(),
                secs * 1000,
                "the two headers disagree"
            );
            assert!(
                resp.headers().get("x-should-retry").is_none(),
                "a retryable class must not say x-should-retry: false"
            );
        }
    }

    /// D2 gap G6 (lane/d2-engine-gaps-20260831): the predictive-admission would-reject
    /// path must be byte-compatible with the existing shed contract. Both flow through
    /// `retry_contract_response`, and this gate pins that: same status, byte-identical
    /// retry header pair, same body schema with `type=rate_limit_error`; only the
    /// `code` names the producer. Shadow mode LOGS the horizon; this is the response
    /// the enforcing flip sends, qualified before any flip exists.
    #[tokio::test]
    async fn admit_predict_reject_matches_shed_contract() {
        // Today's shed 429, exactly as reserve_pending_admit shapes it.
        let shed = retry_contract_response(
            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(error_body(
                    "interactive queue is at its bound",
                    "rate_limit_error",
                    None,
                    Some("shed_queue"),
                )),
            )
                .into_response(),
            Some(7),
        );
        // The enforcing predictor's would-reject: the producer-computed horizon rides
        // the SAME machinery.
        let predict = engine_error_response(&worker::EngineError::rate_limit_after(
            "predicted KV-to-completion exceeds the box budget; retry",
            7,
        ));
        assert_eq!(shed.status(), predict.status());
        for header in ["retry-after", "retry-after-ms"] {
            assert_eq!(
                shed.headers().get(header),
                predict.headers().get(header),
                "header {header} must be byte-identical to the shed contract"
            );
        }
        let shed_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(shed.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let predict_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(predict.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(shed_body["error"]["type"], predict_body["error"]["type"]);
        assert_eq!(predict_body["error"]["type"], "rate_limit_error");
        let shed_keys: Vec<&String> = shed_body["error"].as_object().unwrap().keys().collect();
        let predict_keys: Vec<&String> =
            predict_body["error"].as_object().unwrap().keys().collect();
        assert_eq!(shed_keys, predict_keys, "same body schema, key for key");
        assert_eq!(predict_body["error"]["code"], "rate_limit_exceeded");

        // The producer horizon obeys the shed clamp window (integer seconds, <= 60)...
        let clamped = engine_error_response(&worker::EngineError::rate_limit_after("m", 400));
        assert_eq!(retry_after(&clamped).as_deref(), Some("60"));
        // ...and its absence keeps the historical class default (no regression).
        let plain = engine_error_response(&worker::EngineError::rate_limit("m"));
        assert_eq!(retry_after(&plain).as_deref(), Some("2"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn command_send_failure_obeys_the_retry_contract() {
        let _l = drain_lock();
        let mut st = fake_worker_state();
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        drop(cmd_rx);
        st.cmd_tx = cmd_tx;

        let completion = completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "prompt": "test"
                }))
                .unwrap(),
            ),
        )
        .await;
        let chat = chat_completions(
            State(st),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "messages": [{"role": "user", "content": "test"}]
                }))
                .unwrap(),
            ),
        )
        .await;

        for resp in [completion, chat] {
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(retry_after(&resp).as_deref(), Some("2"));
            assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "2000");
            assert_ne!(
                resp.headers()
                    .get("x-should-retry")
                    .and_then(|v| v.to_str().ok()),
                Some("false")
            );
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(payload["error"]["type"], "server_error");
            assert_eq!(payload["error"]["code"], "overloaded");
        }
    }

    #[test]
    fn unfixable_client_errors_say_x_should_retry_false() {
        // Retrying the identical bytes cannot succeed, and a client that retries on status
        // alone would hammer for nothing. openai-python honors this override explicitly.
        for e in [
            worker::EngineError::model_not_found("unknown model \"x\""),
            worker::EngineError::context_length("prompt too long"),
            worker::EngineError::invalid_param("bad", "messages"),
        ] {
            let resp = engine_error_response(&e);
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
            assert_eq!(resp.headers().get("x-should-retry").unwrap(), "false");
            assert!(
                retry_after(&resp).is_none(),
                "a 400 must not promise a retry window"
            );
        }
    }

    #[tokio::test]
    async fn a_closed_worker_channel_is_503_not_500() {
        // The worker thread died (panicked, unrecoverable) mid-request: the Event channel
        // closes with neither Done nor Error. The client's retry may land on a restarted
        // process, so this is capacity-class with a window — not a bare 500.
        let (tx, rx) = worker::event_channel();
        drop(tx);
        let resp =
            blocking_response(rx, "m".into(), true, Vec::new(), None, Envelope::new(true)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry_after(&resp).as_deref(), Some("5"));
    }

    #[tokio::test]
    async fn a_dark_lane_shed_is_429_with_an_openai_object_body() {
        // The admission peek used to answer `{"error": "<string>"}` — a bare string where every SDK
        // expects an object, which renders as a blank message client-side.
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Error(worker::EngineError::rate_limit(
            "lane judge shed: interactive p99 over budget, retry",
        )))
        .unwrap();
        let (resp, error_code) = peek_admission(rx)
            .await
            .expect_err("a shed must not be forwarded into the stream");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error_code, "rate_limit_exceeded");
        assert_eq!(retry_after(&resp).as_deref(), Some("2"));
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            payload["error"].is_object(),
            "bare-string error body: {payload}"
        );
        assert_eq!(payload["error"]["type"], "rate_limit_error");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("shed")
        );
    }

    #[tokio::test]
    async fn interactive_admission_error_is_a_preheader_429() {
        // An unattainable long-context request must remain retryable even when the client asked
        // for streaming; committing a 200 before this worker verdict would prevent failover.
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Error(worker::EngineError::rate_limit(
            "KV capacity unavailable",
        )))
        .unwrap();
        let (resp, error_code) = peek_admission(rx)
            .await
            .expect_err("admission error must stay pre-header");
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error_code, "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn admission_peek_preserves_context_error_for_the_ledger() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Error(worker::EngineError::context_length(
            "prompt exceeds configured model maximum",
        )))
        .unwrap();
        let (resp, error_code) = peek_admission(rx)
            .await
            .expect_err("context rejection must stay pre-header");
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error_code, "context_length_exceeded");
    }

    #[tokio::test]
    async fn admission_peek_replays_prompt_usage_without_waiting_for_a_token() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::PromptUsage {
            n_prompt: 262_143,
            n_cached: 0,
        })
        .unwrap();
        let mut replay = peek_admission(rx).await.expect("successful admission");
        assert!(matches!(
            replay.recv().await,
            Some(Event::PromptUsage {
                n_prompt: 262_143,
                n_cached: 0
            }),
        ));
    }

    #[test]
    fn penalties_plumb_from_http_to_sampler_config() {
        // gap-scan F3: the fields existed in SamplerConfig all along — assert the HTTP
        // layer actually delivers them, with the one cross-path history window armed.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "task"}],
            "frequency_penalty": 0.5, "presence_penalty": 0.25, "repetition_penalty": 1.1
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let cfg = build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
            .unwrap()
            .request
            .sampler_cfg;
        assert_eq!(cfg.penalty_freq, 0.5);
        assert_eq!(cfg.penalty_present, 0.25);
        assert_eq!(cfg.penalty_repeat, 1.1);
        assert_eq!(cfg.penalty_last_n, memra_engine::spec::PEN_WINDOW_MAX);

        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task", "frequency_penalty": 1.5
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let cfg = build_request(&req, tx, lanes::Lane::Interactive, None).sampler_cfg;
        assert_eq!(cfg.penalty_freq, 1.5);
        assert_eq!(cfg.penalty_last_n, memra_engine::spec::PEN_WINDOW_MAX);

        // no penalties set -> window off, byte-identical legacy config.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "task"
        }))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let cfg = build_request(&req, tx, lanes::Lane::Interactive, None).sampler_cfg;
        assert_eq!(cfg.penalty_last_n, 0);
        assert_eq!(cfg.penalty_repeat, 1.0);
    }

    #[test]
    fn omitted_temperature_is_openai_default_not_greedy() {
        // dogfood F4: `#[serde(default)] temperature: f32` yielded 0.0 = greedy, so any
        // client that omits temperature (the owner's own agentic pill, the OpenAI SDK's
        // documented "leave it out" path) got locked into deterministic argmax — same
        // context in, same token out, identical tool-call cycles forever. OpenAI's
        // default-when-omitted is 1.0 on BOTH surfaces.
        //
        // SCOPE, after lane/vendor-default-sampling (2026-08-19): this test now pins the
        // API-STANDARD FALLBACK — the path taken when NO per-model vendor default is declared
        // and the model's arch publishes none either (`SamplingDefaults::default()`, which is
        // what `build_chat_request`/`build_request` pass here). That path must stay exactly as
        // it was: 1.0 / 1.0 / 0 / 0, pure-temp, never greedy. A SERVED model's omitted request
        // resolves to its vendor recommendation instead — see
        // `vendor_sampling_defaults_fill_only_the_omitted_fields` and
        // `vendor_defaults_leave_the_pure_temp_sampled_spec_regime`. Both laws are live at once:
        // "no declaration = OpenAI-compatible", "declaration = the vendor's own numbers".
        let chat_temp = |body: serde_json::Value| {
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .sampler_cfg
                .temperature
        };
        let comp_temp = |body: serde_json::Value| {
            let req: CompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_request(&req, tx, lanes::Lane::Interactive, None)
                .sampler_cfg
                .temperature
        };

        // OMITTED => 1.0 (sampled), all the way through to the SamplerConfig.
        assert_eq!(
            chat_temp(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}]})),
            1.0,
            "omitted chat temperature must be the OpenAI 1.0 default, not 0.0/greedy"
        );
        assert_eq!(
            comp_temp(serde_json::json!({
            "model": "m", "prompt": "t"})),
            1.0,
            "omitted completions temperature must be the OpenAI 1.0 default"
        );

        // EXPLICIT 0 still means greedy — a caller asking for determinism gets it.
        assert_eq!(
            chat_temp(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "temperature": 0.0})),
            0.0,
            "explicit temperature 0 must stay greedy"
        );
        assert_eq!(
            comp_temp(serde_json::json!({
            "model": "m", "prompt": "t", "temperature": 0})),
            0.0,
            "explicit temperature 0 must stay greedy"
        );
        // and the greedy predicate agrees (this is what gates the spec/graph arms).
        assert!(
            memra_engine::sampler::Sampler::new(sampler_config(
                0.0,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                Some(0)
            ))
            .is_greedy()
        );
        assert!(
            !memra_engine::sampler::Sampler::new(sampler_config(
                1.0,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                Some(0)
            ))
            .is_greedy()
        );

        // explicit non-default values still pass through untouched.
        assert_eq!(
            chat_temp(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "temperature": 0.7})),
            0.7
        );

        // OMITTED filter defaults: top_p disabled at 1.0 (OpenAI default), top_k/min_p
        // disabled at 0 (not OpenAI params — OpenRouter/HF convention, 0 = keep all).
        // An omitted-temperature request must therefore be PURE temperature-1.0 sampling.
        let req: CompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "t"}))
        .unwrap();
        let (tx, _rx) = worker::event_channel();
        let cfg = build_request(&req, tx, lanes::Lane::Interactive, None).sampler_cfg;
        assert_eq!(cfg.top_p, 1.0, "omitted top_p = OpenAI 1.0 = disabled");
        assert_eq!(cfg.top_k, 0, "omitted top_k = disabled");
        assert_eq!(cfg.min_p, 0.0, "omitted min_p = disabled");
        assert_eq!(cfg.penalty_last_n, 0, "omitted penalties = window off");
        // and it lands in the PURE-TEMP sampled-spec regime — the one that keeps the
        // in-graph sampled draft chain (spec.rs `pure_temp`). Filters/penalties would still
        // be spec-eligible but would drop the draft to the eager chain, so the default
        // request shape must stay in the fast regime.
        assert!(
            memra_engine::sampler::Sampler::new(cfg).is_spec_sampling(),
            "the omitted-temperature default must ride sampled spec's pure-temp regime"
        );
    }

    #[test]
    fn step35_chat_uses_published_sampling_defaults_only_when_omitted() {
        let caps = ModelCaps {
            chat_temperature_default: Some(0.5),
            chat_top_p_default: Some(0.9),
            chat_ok: true,
            ..Default::default()
        };
        let cfg = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "model": "step35",
                "messages": [{"role": "user", "content": "task"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request(req, Some(&caps), tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .sampler_cfg
        };

        let omitted = cfg(serde_json::json!({}));
        assert_eq!(omitted.temperature, 0.5);
        assert_eq!(omitted.top_p, 0.9);

        let explicit_temp = cfg(serde_json::json!({"temperature": 0.7}));
        assert_eq!(explicit_temp.temperature, 0.7);
        assert_eq!(
            explicit_temp.top_p, 0.9,
            "omitting top_p must retain StepFun's nucleus default"
        );

        let explicit = cfg(serde_json::json!({"temperature": 0.0, "top_p": 1.0}));
        assert_eq!(
            explicit.temperature, 0.0,
            "explicit greedy must remain authoritative"
        );
        assert_eq!(
            explicit.top_p, 1.0,
            "explicit untruncated sampling must remain authoritative"
        );
    }

    /// qwen/qwen3.8-27b's own model card, § Best Practices / § API Usage Tip (thinking mode —
    /// the mode our template defaults to): temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
    /// presence_penalty 0.0, repetition_penalty 1.0.
    fn qwen38_vendor_defaults() -> SamplingDefaults {
        SamplingDefaults {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            frequency_penalty: None,
        }
    }

    /// google/gemma-4-31B-it's own model card, § Best Practices / 1. Sampling Parameters
    /// ("Use the following standardized sampling configuration across all use cases"):
    /// temperature 1.0, top_p 0.95, top_k 64. Google recommends nothing for min_p or the
    /// penalties, so those stay None -> API-standard (never invented).
    fn gemma4_vendor_defaults() -> SamplingDefaults {
        SamplingDefaults {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(64),
            ..Default::default()
        }
    }

    #[test]
    fn vendor_sampling_defaults_fill_only_the_omitted_fields() {
        // Owner ruling 2026-08-19: "we don't have to serve greedy, we measure greedy but we
        // serve what the user chooses" / "we default to what are the recommendations" /
        // "greedy can create issues". So an OMITTING client gets the model vendor's own
        // published numbers, and every explicit client value still wins.
        let d = ModelSamplingDefaults::single(gemma4_vendor_defaults());
        let chat = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "model": "google/gemma-4-31b-it",
                "messages": [{"role": "user", "content": "task"}],
                // pin the seed so two configs are comparable field-by-field.
                "seed": 7
            });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request_with_trace(
                req,
                Some(&ModelCaps {
                    chat_ok: true,
                    ..Default::default()
                }),
                tx,
                lanes::Lane::Interactive,
                None,
                None,
                None,
                &d,
            )
            .unwrap()
            .request
            .sampler_cfg
        };

        // OMITTED EVERYTHING => the vendor's recommendation, not greedy and not 1.0/1.0/0/0.
        let omitted = chat(serde_json::json!({}));
        assert_eq!(omitted.temperature, 1.0, "gemma-4 card temperature");
        assert_eq!(omitted.top_p, 0.95, "gemma-4 card top_p");
        assert_eq!(omitted.top_k, 64, "gemma-4 card top_k");
        // Google recommends no min_p / penalties: API-standard, NOT invented.
        assert_eq!(omitted.min_p, 0.0, "undeclared min_p stays API-standard");
        assert_eq!(omitted.penalty_repeat, 1.0);
        assert_eq!(omitted.penalty_freq, 0.0);
        assert_eq!(omitted.penalty_present, 0.0);
        assert_eq!(omitted.penalty_last_n, 0, "no penalty => no history window");
        assert!(
            !memra_engine::sampler::Sampler::new(omitted).is_greedy(),
            "the vendor default must NOT be greedy — that is the whole point of the lane"
        );

        // EXPLICIT temperature 0 => TRUE GREEDY, vendor default notwithstanding. This is the
        // invariant every determinism gate we own depends on.
        let greedy = chat(serde_json::json!({"temperature": 0}));
        assert_eq!(
            greedy.temperature, 0.0,
            "explicit temperature 0 stays greedy"
        );
        assert!(
            memra_engine::sampler::Sampler::new(greedy).is_greedy(),
            "an explicit temperature 0 must satisfy the greedy predicate that gates the \
             spec/graph exactness arms"
        );

        // Each explicit field wins ALONE — the others still take the vendor value.
        let one_field = chat(serde_json::json!({"top_k": 3}));
        assert_eq!(one_field.top_k, 3, "explicit top_k wins");
        assert_eq!(
            one_field.temperature, 1.0,
            "omitting temperature still takes the vendor value"
        );
        assert_eq!(one_field.top_p, 0.95, "omitting top_p still takes vendor");

        // Explicit DISABLING values are honored, not mistaken for absence: top_k 0 = keep all,
        // top_p 1.0 = untruncated. A client must be able to switch the vendor filters OFF.
        let disabled = chat(serde_json::json!({"top_k": 0, "top_p": 1.0}));
        assert_eq!(
            disabled.top_k, 0,
            "an explicit top_k 0 means KEEP ALL, not 'unset'"
        );
        assert_eq!(
            disabled.top_p, 1.0,
            "an explicit top_p 1.0 means untruncated"
        );

        // Explicit penalties are honored and arm the one cross-path bounded window.
        let penal = chat(serde_json::json!({"presence_penalty": 1.5}));
        assert_eq!(penal.penalty_present, 1.5);
        assert_eq!(penal.penalty_last_n, memra_engine::spec::PEN_WINDOW_MAX);
    }

    #[test]
    fn vendor_sampling_defaults_are_identical_on_every_surface() {
        // STANDARD-SURFACE LAW. Before this lane the surfaces DISAGREED: the chat body's
        // temperature/top_p were `Option` and consulted the per-model default, while
        // /v1/completions used bare `f32`s with `serde(default)` — so "omitted" was
        // indistinguishable from "1.0" there and the per-model default was unreachable on the
        // raw-prompt surface. Both bodies now funnel into ONE `resolve_sampler_config`.
        //
        // /v1/messages and /v1/responses are covered transitively and by construction: both
        // translate into a ChatCompletionReq and call the same `build_chat_request_with_trace`
        // with the same `ModelSamplingDefaults` (see surfaces.rs). Their own tests pin the other
        // half of the contract — that an omitted field translates to an ABSENT field rather
        // than a zero-filled one.
        let d = qwen38_vendor_defaults();
        let md = ModelSamplingDefaults::single(d);
        let comp = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "model": "qwen/qwen3.8-27b", "prompt": "task", "seed": 11 });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let req: CompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_request_with_trace(&req, tx, lanes::Lane::Interactive, None, None, &d).sampler_cfg
        };
        let chat = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "model": "qwen/qwen3.8-27b",
                "messages": [{"role": "user", "content": "task"}],
                "seed": 11 });
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request_with_trace(
                req,
                Some(&ModelCaps {
                    chat_ok: true,
                    ..Default::default()
                }),
                tx,
                lanes::Lane::Interactive,
                None,
                None,
                None,
                &md,
            )
            .unwrap()
            .request
            .sampler_cfg
        };

        for extra in [
            serde_json::json!({}),
            serde_json::json!({"temperature": 0}),
            serde_json::json!({"temperature": 0.0}),
            serde_json::json!({"temperature": 0.7}),
            serde_json::json!({"top_p": 1.0}),
            serde_json::json!({"top_k": 0}),
            serde_json::json!({"min_p": 0.05}),
            serde_json::json!({"repetition_penalty": 1.1}),
            serde_json::json!({"frequency_penalty": 0.5}),
            serde_json::json!({"presence_penalty": 1.5}),
            serde_json::json!({
                "temperature": 0.3, "top_p": 0.5, "top_k": 7, "min_p": 0.02,
                "frequency_penalty": 0.1, "presence_penalty": 0.2,
                "repetition_penalty": 1.05 }),
        ] {
            let c = comp(extra.clone());
            let h = chat(extra.clone());
            assert_eq!(
                (
                    c.temperature,
                    c.top_p,
                    c.top_k,
                    c.min_p,
                    c.penalty_repeat,
                    c.penalty_freq,
                    c.penalty_present,
                    c.penalty_last_n,
                    c.seed
                ),
                (
                    h.temperature,
                    h.top_p,
                    h.top_k,
                    h.min_p,
                    h.penalty_repeat,
                    h.penalty_freq,
                    h.penalty_present,
                    h.penalty_last_n,
                    h.seed
                ),
                "/v1/completions and /v1/chat/completions disagree on {extra} — \
                 standard-surface-law violation"
            );
        }

        // and the vendor values really are what the omitting request lands on, on BOTH.
        let omitted = comp(serde_json::json!({}));
        assert_eq!(
            omitted.temperature, 1.0,
            "qwen3.8 card thinking temperature"
        );
        assert_eq!(omitted.top_p, 0.95, "qwen3.8 card top_p");
        assert_eq!(omitted.top_k, 20, "qwen3.8 card top_k");
        // explicit greedy survives on the raw-prompt surface too.
        assert!(
            memra_engine::sampler::Sampler::new(comp(serde_json::json!({"temperature": 0})))
                .is_greedy()
        );
    }

    /// WORKER-TRUTH surface parity (hermes `d991b51699218285`): the SAME omitted-sampling
    /// request, sent through all four REAL handlers, must reach the worker with the SAME
    /// effective sampling. The builder-level test above proves the two request builders
    /// agree when handed one `SamplingDefaults`; this one proves the HANDLERS do —
    /// including each surface's own per-request `AppState::sampling_defaults` lookup and
    /// the /v1/messages + /v1/responses translations, which that test only covered "by
    /// construction". The pinned scenario is the finding's exact one: a model whose arch
    /// caps carry the Step-3.7 vendor recommendation (0.5/0.9) and a client that says
    /// nothing. Pre-resolver, /v1/completions never consulted ModelCaps and shipped
    /// temperature 1.0 against the 0.5/0.9 the chat path applied; a surface that stops
    /// consulting caps, resolves through a different body, or zero-fills an omitted field
    /// in translation diverges HERE and fails by name.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn same_omitted_request_resolves_identically_on_all_four_surfaces() {
        let _l = drain_lock();
        let step_caps = ModelCaps {
            chat_ok: true,
            chat_temperature_default: Some(0.5),
            chat_top_p_default: Some(0.9),
            ..Default::default()
        };
        let (cfg_tx, cfg_rx) = std::sync::mpsc::channel::<WorkerSaw>();
        let st = fake_worker_state_full(
            1,
            std::time::Duration::ZERO,
            HashMap::from([("m".to_string(), step_caps)]),
            Some(cfg_tx),
        );
        // Everything a distribution-side comparison can see, EXCEPT the seed: an omitted
        // seed is fresh entropy per request BY CONTRACT
        // (`omitted_seed_is_fresh_entropy_not_a_pinned_zero`), so surfaces must NOT agree
        // on it.
        let fields = |saw: &WorkerSaw| {
            let c = &saw.sampler_cfg;
            (
                c.temperature,
                c.top_p,
                c.top_k,
                c.min_p,
                c.penalty_repeat,
                c.penalty_freq,
                c.penalty_present,
                c.penalty_last_n,
            )
        };
        let worker_saw = |surface: &str| {
            cfg_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("{surface}: request never reached the worker"))
        };

        let resp = completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(serde_json::from_value(serde_json::json!({"model": "m", "prompt": "t"})).unwrap()),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/completions rejected the omitted-sampling request"
        );
        let comp = worker_saw("/v1/completions");

        let resp = chat_completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}]}))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/chat/completions rejected the omitted-sampling request"
        );
        let chat = worker_saw("/v1/chat/completions");

        let resp = anthropic::messages(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            axum::body::Bytes::from(
                serde_json::json!({
                    "model": "m", "max_tokens": 16,
                    "messages": [{"role": "user", "content": "t"}]})
                .to_string(),
            ),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/messages rejected the omitted-sampling request"
        );
        let msg = worker_saw("/v1/messages");

        let resp = responses_api::responses(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            axum::body::Bytes::from(serde_json::json!({"model": "m", "input": "t"}).to_string()),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/responses rejected the omitted-sampling request"
        );
        let rsp = worker_saw("/v1/responses");

        for (surface, cfg) in [
            ("/v1/completions", &comp),
            ("/v1/messages", &msg),
            ("/v1/responses", &rsp),
        ] {
            assert_eq!(
                fields(cfg),
                fields(&chat),
                "{surface} resolved DIFFERENT effective sampling than /v1/chat/completions \
                 for the same omitted-sampling request — standard-surface-law violation \
                 (hermes d991b51699218285)"
            );
        }
        // ...and the value every surface lands on IS the Step vendor recommendation, not
        // the API-standard 1.0/1.0 the pre-resolver completions surface shipped.
        assert_eq!(
            (comp.sampler_cfg.temperature, comp.sampler_cfg.top_p),
            (0.5, 0.9),
            "an omitting client must get the model's vendor caps (Step-3.7: 0.5/0.9) on \
             EVERY surface, not the API-standard 1.0/1.0 (hermes d991b51699218285)"
        );
    }

    /// WORKER-TRUTH effort parity (issue #31, standard-surface law): the SAME
    /// reasoning-effort value, expressed in each surface's own field —
    /// `reasoning_effort` on /v1/chat/completions, `reasoning.effort` on /v1/responses,
    /// `output_config.effort` on /v1/messages — must produce the SAME acceptance
    /// decision AND the same resolved (ThinkMode, effort_level) at the worker boundary.
    /// Before this lane /v1/messages accepted EVERY string (bogus/banana/"" -> 200) and
    /// silently ignored the parameter: `anthropic::translate` never read
    /// `output_config.effort`, so it was dropped before `parse_think` — a mutation that
    /// restores the drop fails every row of this test by name.
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn same_effort_value_resolves_identically_on_every_surface() {
        let _l = drain_lock();
        // effort_levels caps so the level string is worker-visible too (step35 dialect);
        // ThinkMode alone would still catch the switch half on binary templates.
        let caps = ModelCaps {
            chat_ok: true,
            effort_levels: true,
            ..Default::default()
        };
        let (saw_tx, saw_rx) = std::sync::mpsc::channel::<WorkerSaw>();
        let st = fake_worker_state_full(
            1,
            std::time::Duration::ZERO,
            HashMap::from([("m".to_string(), caps)]),
            Some(saw_tx),
        );
        let send = |st: AppState, surface: &'static str, effort: &'static str| async move {
            match surface {
                "/v1/chat/completions" => {
                    chat_completions(
                        State(st),
                        axum::http::HeaderMap::new(),
                        None,
                        Json(
                            serde_json::from_value(serde_json::json!({
                                "model": "m", "max_tokens": 8,
                                "reasoning_effort": effort,
                                "messages": [{"role": "user", "content": "t"}]}))
                            .unwrap(),
                        ),
                    )
                    .await
                }
                "/v1/responses" => {
                    responses_api::responses(
                        State(st),
                        axum::http::HeaderMap::new(),
                        None,
                        axum::body::Bytes::from(
                            serde_json::json!({
                                "model": "m", "max_output_tokens": 8, "input": "t",
                                "reasoning": {"effort": effort}})
                            .to_string(),
                        ),
                    )
                    .await
                }
                "/v1/messages" => {
                    anthropic::messages(
                        State(st),
                        axum::http::HeaderMap::new(),
                        None,
                        axum::body::Bytes::from(
                            serde_json::json!({
                                "model": "m", "max_tokens": 8,
                                "messages": [{"role": "user", "content": "t"}],
                                "output_config": {"effort": effort}})
                            .to_string(),
                        ),
                    )
                    .await
                }
                other => panic!("unknown surface {other}"),
            }
        };
        const SURFACES: [&str; 3] = ["/v1/chat/completions", "/v1/responses", "/v1/messages"];

        // Accepted rows: same 200, same worker-truth (ThinkMode, effort_level) on all
        // three surfaces. none/minimal REALLY suppress thinking on /v1/messages now.
        for (effort, want_think, want_level) in [
            ("none", ThinkMode::NoThink, Some("low")),
            ("minimal", ThinkMode::NoThink, Some("low")),
            ("low", ThinkMode::Think, Some("low")),
            ("medium", ThinkMode::Think, Some("medium")),
            ("high", ThinkMode::Think, Some("high")),
            // the issue's divergent row: xhigh was 400 on chat, 200 on the other two.
            ("xhigh", ThinkMode::Think, Some("high")),
        ] {
            for surface in SURFACES {
                let resp = send(st.clone(), surface, effort).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::OK,
                    "{surface} rejected effort {effort:?} — the surfaces' allowlists \
                     diverged again (issue #31)"
                );
                let saw = saw_rx
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .unwrap_or_else(|_| {
                        panic!("{surface}: effort {effort:?} request never reached the worker")
                    });
                assert_eq!(
                    (saw.think, saw.reasoning_effort.as_deref()),
                    (want_think, want_level),
                    "{surface} resolved effort {effort:?} to a DIFFERENT worker-truth \
                     reasoning surface — the parameter was dropped or remapped before \
                     parse_think (issue #31 regression)"
                );
            }
        }

        // Rejected rows: the SAME 400 decision on all three surfaces — /v1/messages
        // accepting a value the other surfaces refuse is exactly issue #31.
        for effort in ["bogus", "banana", ""] {
            for surface in SURFACES {
                let resp = send(st.clone(), surface, effort).await;
                assert_eq!(
                    resp.status(),
                    StatusCode::BAD_REQUEST,
                    "{surface} accepted effort {effort:?} — silent-accept regression \
                     (issue #31: the value never reached parse_think's allowlist)"
                );
                // Each surface still speaks its own documented error envelope.
                let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
                    .await
                    .unwrap();
                let v: serde_json::Value = serde_json::from_slice(&body)
                    .unwrap_or_else(|_| panic!("{surface}: non-JSON 400 body for {effort:?}"));
                match surface {
                    "/v1/messages" => {
                        assert_eq!(v["type"], "error", "{surface} error envelope");
                        assert_eq!(
                            v["error"]["type"], "invalid_request_error",
                            "{surface} error type"
                        );
                    }
                    _ => {
                        assert!(
                            v["error"]["message"].is_string(),
                            "{surface} OpenAI-shaped error body: {v}"
                        );
                    }
                }
            }
        }

        // Anthropic precedence at the HTTP boundary: thinking.type wins the switch when
        // both levers are present (documented Anthropic semantics), and the effort is
        // still validated rather than silently dropped.
        let resp = anthropic::messages(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            axum::body::Bytes::from(
                serde_json::json!({
                    "model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "t"}],
                    "thinking": {"type": "enabled"},
                    "output_config": {"effort": "none"}})
                .to_string(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let saw = saw_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("thinking+effort request never reached the worker");
        assert_eq!(
            saw.think,
            ThinkMode::Think,
            "thinking.type (the documented Anthropic lever) must win the switch over \
             output_config.effort"
        );
        let resp = anthropic::messages(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            axum::body::Bytes::from(
                serde_json::json!({
                    "model": "m", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "t"}],
                    "thinking": {"type": "enabled"},
                    "output_config": {"effort": "banana"}})
                .to_string(),
            ),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "an invalid effort must 400 even next to an explicit thinking.type — \
             precedence must not re-open the silent-accept hole"
        );
    }

    #[test]
    fn vendor_sampling_defaults_are_boot_validated() {
        // Same posture as default_reasoning_effort: a typo'd default fails at metadata parse
        // (before GPU load), never as a per-request 400 storm after a watchdog restart.
        let parsed = OpenRouterMetadataFile::from_toml(
            r#"
[models.g]
default_temperature = 1.0
default_top_p = 0.95
default_top_k = 64
default_min_p = 0.0
default_presence_penalty = 0.0
default_frequency_penalty = 0.0
default_repetition_penalty = 1.0
"#,
        )
        .unwrap();
        let g = parsed.get("g").unwrap();
        assert_eq!(g.default_temperature, Some(1.0));
        assert_eq!(g.default_top_p, Some(0.95));
        assert_eq!(g.default_top_k, Some(64));

        // A ZERO default temperature is refused ON PURPOSE: it would reinstate
        // greedy-by-default deployment-wide, silently, for every omitting client — exactly the
        // hazard this lane exists to remove. Greedy stays reachable per-request.
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.g]
default_temperature = 0.0
"#,
        )
        .unwrap_err();
        assert!(err.contains("default_temperature"), "{err}");
        assert!(
            err.contains("greedy"),
            "the refusal must say WHY a zero default is refused: {err}"
        );

        for bad in [
            "default_temperature = 2.5",
            "default_temperature = -1.0",
            "default_top_p = 0.0",
            "default_top_p = 1.5",
            "default_min_p = 1.0",
            "default_min_p = -0.1",
            "default_presence_penalty = 3.0",
            "default_frequency_penalty = -2.5",
            "default_repetition_penalty = 0.0",
        ] {
            let err =
                OpenRouterMetadataFile::from_toml(&format!("[models.g]\n{bad}\n")).unwrap_err();
            let key = bad.split(' ').next().unwrap();
            assert!(err.contains(key), "{bad} must be refused by name: {err}");
        }

        // DEPLOY-ORDER TRAP (the same one default_reasoning_effort created):
        // `deny_unknown_fields` means an OLDER binary FAILS BOOT on a config carrying these
        // new keys. Binary first, then config — never the other way round.
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.g]
default_temperture = 1.0
"#,
        )
        .unwrap_err();
        assert!(
            err.contains("unknown field"),
            "an unknown key must be fatal, which is what makes binary-first ordering \
             mandatory: {err}"
        );
    }

    #[test]
    fn non_thinking_sampling_arm_is_boot_validated() {
        // Same posture as the flat keys: a typo'd arm fails at metadata parse, before GPU
        // load. The arm goes through the SAME range law (validate_sampling_arm), so the two
        // arms cannot drift apart in what they accept.
        let parsed = OpenRouterMetadataFile::from_toml(
            r#"
[models.q]
default_temperature = 1.0
default_top_p = 0.95
default_top_k = 20

[models.q.non_thinking_sampling]
temperature = 0.7
top_p = 0.8
top_k = 20
presence_penalty = 1.5
"#,
        )
        .unwrap();
        let arm = parsed
            .get("q")
            .unwrap()
            .non_thinking_sampling
            .as_ref()
            .unwrap();
        assert_eq!(arm.temperature, Some(0.7));
        assert_eq!(arm.top_p, Some(0.8));
        assert_eq!(arm.top_k, Some(20));
        assert_eq!(arm.presence_penalty, Some(1.5));
        assert_eq!(
            arm.min_p, None,
            "undeclared arm fields stay undeclared, never invented"
        );

        // A zero arm temperature is refused for the same reason as the flat key: it would be
        // greedy-by-default for every thinking-off omitting client. The refusal names the
        // exact nested key the operator wrote.
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.q]
[models.q.non_thinking_sampling]
temperature = 0.0
"#,
        )
        .unwrap_err();
        assert!(err.contains("non_thinking_sampling.temperature"), "{err}");
        assert!(err.contains("greedy"), "{err}");

        // A DECLARED-but-empty arm is refused: it would silently hand thinking-off traffic
        // the bare API-standard defaults while the file looks configured.
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.q]
[models.q.non_thinking_sampling]
"#,
        )
        .unwrap_err();
        assert!(err.contains("non_thinking_sampling"), "{err}");
        assert!(err.contains("declare"), "{err}");

        // Out-of-range arm values are named with their full nested key.
        for bad in [
            "temperature = 2.5",
            "top_p = 0.0",
            "top_p = 1.5",
            "min_p = 1.0",
            "presence_penalty = 3.0",
            "frequency_penalty = -2.5",
            "repetition_penalty = 0.0",
        ] {
            let err = OpenRouterMetadataFile::from_toml(&format!(
                "[models.q]\n[models.q.non_thinking_sampling]\n{bad}\n"
            ))
            .unwrap_err();
            let key = bad.split(' ').next().unwrap();
            assert!(
                err.contains(&format!("non_thinking_sampling.{key}")),
                "the refusal for {bad:?} must name the nested key: {err}"
            );
        }

        // DEPLOY-ORDER TRAP, inherited on purpose: the arm table is deny_unknown_fields too,
        // and an OLDER binary fails boot on the whole `non_thinking_sampling` table itself —
        // binary first, then config, exactly like the flat keys.
        let err = OpenRouterMetadataFile::from_toml(
            r#"
[models.q]
[models.q.non_thinking_sampling]
temperture = 0.7
"#,
        )
        .unwrap_err();
        assert!(err.contains("unknown field"), "{err}");
    }

    /// qwen/qwen3.8-27b's own model card publishes a SECOND sampling arm for
    /// thinking-disabled use (retrieved 2026-08-24): temperature 0.7, top_p 0.80,
    /// top_k 20, presence_penalty 1.5. min_p and the other penalties are not
    /// separately recommended for this arm.
    fn qwen38_non_thinking_defaults() -> SamplingDefaults {
        SamplingDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(20),
            presence_penalty: Some(1.5),
            ..Default::default()
        }
    }

    fn qwen38_two_arm_defaults() -> ModelSamplingDefaults {
        ModelSamplingDefaults {
            thinking: qwen38_vendor_defaults(),
            non_thinking: Some(qwen38_non_thinking_defaults()),
        }
    }

    /// The served qwen3.8 template's caps shape: think tail on by default WITH the
    /// enable_thinking switch, so an explicit off-request is honorable (no 400 from the
    /// silent-ignore gate).
    fn qwen38_caps() -> ModelCaps {
        ModelCaps {
            chat_ok: true,
            qwen_think: true,
            think_switch: true,
            ..Default::default()
        }
    }

    /// Field-tuple key for comparing two SamplerConfigs exactly (the struct itself is not
    /// PartialEq; the seed is pinned by the test bodies so it participates too).
    fn sampler_key(c: &SamplerConfig) -> (f32, f32, usize, f32, f32, f32, f32, usize, u64) {
        (
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.penalty_present,
            c.penalty_freq,
            c.penalty_repeat,
            c.penalty_last_n,
            c.seed,
        )
    }

    fn build_with_arms(
        defaults: &ModelSamplingDefaults,
        caps: &ModelCaps,
        default_effort: Option<&str>,
        extra: serde_json::Value,
    ) -> Request {
        let mut body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "task"}],
            // pinned so two builds of the same body are comparable field-by-field.
            "seed": 3
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
        let (tx, _rx) = worker::event_channel();
        build_chat_request_with_trace(
            req,
            Some(caps),
            tx,
            lanes::Lane::Interactive,
            None,
            None,
            default_effort,
            defaults,
        )
        .unwrap()
        .request
    }

    #[test]
    fn resolved_thinking_mode_picks_the_vendor_sampling_arm() {
        // THE RESOLUTION MATRIX (owner ruling 2026-08-24): mode x set/unset x both model
        // shapes. Two models: qwen3.8 (vendor publishes TWO arms) and an ornith-shaped
        // single-arm model (Ornith-1.5 documents NO non-thinking arm) — the latter must be
        // unaffected by every row of the matrix.
        let two_arm = qwen38_two_arm_defaults();
        let single_arm = ModelSamplingDefaults::single(qwen38_vendor_defaults());
        let caps = qwen38_caps();

        // Every live off-spelling resolves to NoThink and takes the NON-THINKING arm.
        let off_spellings = [
            serde_json::json!({"reasoning_effort": "none"}),
            serde_json::json!({"enable_thinking": false}),
            serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}}),
            serde_json::json!({"reasoning": {"enabled": false}}),
        ];
        for extra in &off_spellings {
            let r = build_with_arms(&two_arm, &caps, None, extra.clone());
            assert_eq!(r.think, ThinkMode::NoThink, "{extra}");
            let c = &r.sampler_cfg;
            assert_eq!(c.temperature, 0.7, "{extra}: non-thinking card temperature");
            assert_eq!(c.top_p, 0.8, "{extra}: non-thinking card top_p");
            assert_eq!(c.top_k, 20, "{extra}: non-thinking card top_k");
            assert_eq!(
                c.penalty_present, 1.5,
                "{extra}: non-thinking presence_penalty"
            );
            assert_eq!(
                c.penalty_last_n,
                memra_engine::spec::PEN_WINDOW_MAX,
                "{extra}: the arm's presence penalty uses the cross-path history window"
            );
            assert_eq!(
                c.min_p, 0.0,
                "{extra}: the arm recommends no min_p — API standard, never the other arm's"
            );

            // The SAME off-request on the single-arm model keeps the single arm — the arm
            // machinery must be invisible to a model that never declared a second arm.
            let s = build_with_arms(&single_arm, &caps, None, extra.clone());
            assert_eq!(s.think, ThinkMode::NoThink, "{extra}");
            assert_eq!(s.sampler_cfg.temperature, 1.0, "{extra}: single-arm model");
            assert_eq!(s.sampler_cfg.top_p, 0.95, "{extra}: single-arm model");
            assert_eq!(
                s.sampler_cfg.penalty_present, 0.0,
                "{extra}: single-arm model"
            );
        }

        // Thinking ON — explicitly or by the template's own default — keeps the PRIMARY arm,
        // on both models.
        for extra in [
            serde_json::json!({}),
            serde_json::json!({"enable_thinking": true}),
            serde_json::json!({"reasoning_effort": "high"}),
            serde_json::json!({"reasoning": {"enabled": true}}),
        ] {
            for defaults in [&two_arm, &single_arm] {
                let c = build_with_arms(defaults, &caps, None, extra.clone()).sampler_cfg;
                assert_eq!(c.temperature, 1.0, "{extra}: thinking card temperature");
                assert_eq!(c.top_p, 0.95, "{extra}: thinking card top_p");
                assert_eq!(c.top_k, 20, "{extra}: thinking card top_k");
                assert_eq!(
                    c.penalty_present, 0.0,
                    "{extra}: thinking arm has no presence"
                );
            }
        }

        // An operator `default_reasoning_effort = "none"` resolves the UNSET case to
        // NoThink upstream, so the unset case lands on the non-thinking arm...
        let c = build_with_arms(&two_arm, &caps, Some("none"), serde_json::json!({})).sampler_cfg;
        assert_eq!(
            c.temperature, 0.7,
            "deployment-default off = non-thinking arm"
        );
        // ...and an explicit client ON next to that deployment default wins it back.
        let c = build_with_arms(
            &two_arm,
            &caps,
            Some("none"),
            serde_json::json!({"enable_thinking": true}),
        )
        .sampler_cfg;
        assert_eq!(
            c.temperature, 1.0,
            "explicit ON beats the deployment default"
        );

        // SET params are NEVER overridden, whichever arm applies; only unset fields take it.
        let c = build_with_arms(
            &two_arm,
            &caps,
            None,
            serde_json::json!({"enable_thinking": false, "temperature": 0.55}),
        )
        .sampler_cfg;
        assert_eq!(c.temperature, 0.55, "explicit temperature survives the arm");
        assert_eq!(c.top_p, 0.8, "unset top_p still takes the non-thinking arm");
        let c = build_with_arms(
            &two_arm,
            &caps,
            None,
            serde_json::json!({
                "reasoning_effort": "none", "top_p": 0.99, "presence_penalty": 0.0}),
        )
        .sampler_cfg;
        assert_eq!(c.top_p, 0.99, "explicit top_p wins");
        assert_eq!(
            c.penalty_present, 0.0,
            "an explicit presence_penalty 0.0 wins over the arm's 1.5 — a disabling value \
             is a value, not an absence"
        );
        assert_eq!(
            c.penalty_last_n, 0,
            "all penalties off => no history window"
        );
        assert_eq!(c.top_k, 20, "unset top_k still takes the arm");

        // Explicit temperature 0 stays TRUE GREEDY under the non-thinking arm too — the one
        // invariant every determinism gate depends on bends for no arm.
        let c = build_with_arms(
            &two_arm,
            &caps,
            None,
            serde_json::json!({"enable_thinking": false, "temperature": 0}),
        )
        .sampler_cfg;
        assert!(
            memra_engine::sampler::Sampler::new(c).is_greedy(),
            "explicit temperature 0 must stay greedy on the non-thinking arm"
        );

        // The same explicit-set matrix on the SINGLE-ARM model: identical to the two-arm
        // model's thinking rows, untouched by every off-request.
        let c = build_with_arms(
            &single_arm,
            &caps,
            None,
            serde_json::json!({"enable_thinking": false, "temperature": 0.55}),
        )
        .sampler_cfg;
        assert_eq!(c.temperature, 0.55);
        assert_eq!(
            c.top_p, 0.95,
            "single-arm model: unset top_p takes its one arm"
        );
    }

    #[test]
    fn sampling_arms_never_blend_field_by_field() {
        // The two arms are separate vendor programs. A field the vendor left out of the
        // non-thinking arm falls to the API-STANDARD default — never to the thinking arm's
        // value and never to the arch cap — because a blended config would be numbers no
        // vendor ever published.
        let parsed = OpenRouterMetadataFile::from_toml(
            r#"
[models.m]
default_temperature = 1.0
default_min_p = 0.05

[models.m.non_thinking_sampling]
temperature = 0.6
"#,
        )
        .unwrap();
        let caps = ModelCaps {
            chat_temperature_default: Some(0.5),
            chat_top_p_default: Some(0.9),
            ..Default::default()
        };
        let d = ModelSamplingDefaults::resolve(parsed.get("m"), Some(&caps));
        let client = ClientSampling {
            seed: Some(1),
            ..Default::default()
        };

        let off = resolve_sampler_config(client, d.for_mode(ThinkMode::NoThink));
        assert_eq!(off.temperature, 0.6, "the arm's own field applies");
        assert_eq!(
            off.min_p, 0.0,
            "min_p undeclared on the arm = API standard, NOT the thinking arm's 0.05"
        );
        assert_eq!(
            off.top_p, 1.0,
            "top_p undeclared on the arm = API standard, NOT the arch cap's 0.9"
        );

        // Default and Think keep the primary arm, caps fallback included.
        for mode in [ThinkMode::Default, ThinkMode::Think] {
            let on = resolve_sampler_config(client, d.for_mode(mode));
            assert_eq!(on.temperature, 1.0);
            assert_eq!(on.min_p, 0.05);
            assert_eq!(on.top_p, 0.9, "primary arm keeps the arch-cap fallback");
        }
    }

    #[test]
    fn single_arm_models_and_thinking_on_requests_match_the_pre_arm_law_exactly() {
        // BYTE-IDENTITY PIN. Two populations must be exactly what they were before the arm
        // existed: (a) every request against a single-arm model (Ornith-1.5 documents NO
        // non-thinking arm), (b) thinking-on requests against the two-arm model. "Before"
        // is the one-resolver law verbatim — resolve_sampler_config(client, the one arm) —
        // so each build is compared against that expression computed directly. Sampling
        // resolution consumes no render input and produces none: chat_turns/tools/think/
        // effort are built from the request alone, so sampler equality here IS render
        // byte-identity (think/effort are additionally asserted per body).
        let caps = qwen38_caps();
        let single_arm = ModelSamplingDefaults::single(qwen38_vendor_defaults());
        let two_arm = qwen38_two_arm_defaults();

        let bodies = [
            serde_json::json!({}),
            serde_json::json!({"enable_thinking": true}),
            serde_json::json!({"reasoning_effort": "high"}),
            serde_json::json!({"reasoning_effort": "none"}),
            serde_json::json!({"enable_thinking": false}),
            serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}}),
            serde_json::json!({"temperature": 0.3, "top_p": 0.5}),
            serde_json::json!({"enable_thinking": false, "temperature": 0}),
        ];
        for extra in &bodies {
            // (a) the single-arm model: every mode, byte-equal to the pre-arm resolver.
            let r = build_with_arms(&single_arm, &caps, None, extra.clone());
            let mut client = ClientSampling {
                seed: Some(3),
                ..Default::default()
            };
            if let Some(t) = extra.get("temperature").and_then(|v| v.as_f64()) {
                client.temperature = Some(t as f32);
            }
            if let Some(p) = extra.get("top_p").and_then(|v| v.as_f64()) {
                client.top_p = Some(p as f32);
            }
            let pre_arm = resolve_sampler_config(client, &qwen38_vendor_defaults());
            assert_eq!(
                sampler_key(&r.sampler_cfg),
                sampler_key(&pre_arm),
                "{extra}: single-arm model diverged from the pre-arm resolution law"
            );

            // (b) thinking-on / unset bodies: the TWO-arm model is byte-equal to the
            // single-arm build — think mode, effort string and sampler all included.
            if r.think != ThinkMode::NoThink {
                let t = build_with_arms(&two_arm, &caps, None, extra.clone());
                assert_eq!(t.think, r.think, "{extra}");
                assert_eq!(t.reasoning_effort, r.reasoning_effort, "{extra}");
                assert_eq!(
                    sampler_key(&t.sampler_cfg),
                    sampler_key(&r.sampler_cfg),
                    "{extra}: a thinking-on request must not feel the non-thinking arm"
                );
            }
        }
    }

    #[test]
    fn constraint_forced_nothink_takes_the_non_thinking_arm() {
        // response_format on a switch-carrying think template forces the think switch off
        // (the grammar x think law above build_chat_request_with_trace). The model then
        // GENUINELY runs non-thinking, so the vendor's non-thinking arm is the honest
        // default for the sampling fields such a request left unset — the arm is selected
        // AFTER the constraint gate settles the mode, and this pins that ordering.
        let r = build_with_arms(
            &qwen38_two_arm_defaults(),
            &qwen38_caps(),
            None,
            serde_json::json!({"response_format": {"type": "json_object"}}),
        );
        assert_eq!(
            r.think,
            ThinkMode::NoThink,
            "constraint forces the switch off"
        );
        assert_eq!(
            r.sampler_cfg.temperature, 0.7,
            "and the arm follows the real mode"
        );
        assert_eq!(r.sampler_cfg.penalty_present, 1.5);
    }

    #[test]
    fn metadata_sampling_defaults_outrank_arch_caps_but_never_the_client() {
        // Two default sources exist: the operator's per-model metadata block and the engine's
        // arch-keyed caps (step35 = StepFun's published 0.5/0.9). The operator's declaration is
        // about the artifact actually loaded on THIS box, so it wins; the cap remains the
        // fallback so a metadata-less box behaves exactly as it did before this lane.
        let caps = ModelCaps {
            chat_temperature_default: Some(0.5),
            chat_top_p_default: Some(0.9),
            chat_ok: true,
            ..Default::default()
        };
        let metadata = OpenRouterModelMetadata {
            default_temperature: Some(1.0),
            default_top_p: Some(0.95),
            default_top_k: Some(64),
            ..Default::default()
        };

        let caps_only = SamplingDefaults::resolve(None, Some(&caps));
        assert_eq!(caps_only.temperature, Some(0.5), "arch cap is the fallback");
        assert_eq!(caps_only.top_p, Some(0.9));
        assert_eq!(caps_only.top_k, None, "caps declare no top_k");

        let both = SamplingDefaults::resolve(Some(&metadata), Some(&caps));
        assert_eq!(
            both.temperature,
            Some(1.0),
            "metadata outranks the arch cap"
        );
        assert_eq!(both.top_p, Some(0.95));
        assert_eq!(both.top_k, Some(64));

        // Partial metadata falls through to the cap field by field, not wholesale.
        let partial = SamplingDefaults::resolve(
            Some(&OpenRouterModelMetadata {
                default_temperature: Some(0.7),
                ..Default::default()
            }),
            Some(&caps),
        );
        assert_eq!(partial.temperature, Some(0.7));
        assert_eq!(
            partial.top_p,
            Some(0.9),
            "an undeclared metadata field must fall through to the cap, not to 1.0"
        );

        // No metadata AND no caps = the pre-lane API-standard path, byte-for-byte.
        assert_eq!(
            SamplingDefaults::resolve(None, None),
            SamplingDefaults::default()
        );
    }

    #[test]
    fn vendor_defaults_leave_the_pure_temp_sampled_spec_regime() {
        // COST OF THE CHANGE, pinned so it is never a surprise (lane/vendor-default-sampling,
        // 2026-08-19). Both served models' vendor recommendations carry TRUNCATION FILTERS
        // (qwen3.8: top_p 0.95 + top_k 20; gemma-4: top_p 0.95 + top_k 64), and the in-graph
        // sampled draft chain samples from the RAW softmax — it can hold no per-row filter
        // stats, so spec.rs engages `graph_s` only in the pure-temp regime and otherwise falls
        // back to the EAGER draft chain (memra-sampling `is_spec_sampling`, spec.rs `pure_temp`).
        //
        // Nothing about exactness changes: filters are applied symmetrically to draft q and
        // target p under the rejection verify, so these requests stay spec-ELIGIBLE and
        // distribution-exact. What changes is which draft chain runs — and it changes for the
        // DEFAULT request shape, i.e. the one most customers send. That trade is the owner's
        // call, not this test's; the test exists so the flip is measured, not discovered.
        let resolved = |d: &SamplingDefaults| {
            resolve_sampler_config(
                ClientSampling {
                    seed: Some(1),
                    ..Default::default()
                },
                d,
            )
        };

        // Pre-lane default shape (no per-model key declared): pure temp, in-graph draft.
        assert!(
            memra_engine::sampler::Sampler::new(resolved(&SamplingDefaults::default()))
                .is_spec_sampling(),
            "the API-standard default must stay in the fast pure-temp regime"
        );

        for (name, d) in [
            ("qwen/qwen3.8-27b", qwen38_vendor_defaults()),
            ("google/gemma-4-31b-it", gemma4_vendor_defaults()),
        ] {
            let sampler = memra_engine::sampler::Sampler::new(resolved(&d));
            assert!(
                !sampler.is_greedy(),
                "{name}: vendor default must not be greedy"
            );
            assert!(
                !sampler.is_spec_sampling(),
                "{name}: vendor top_p/top_k DO leave the pure-temp regime — if this ever \
                 starts passing, either the vendor numbers changed or the in-graph draft \
                 learned filters, and the perf note in docs/SERVING.md needs revisiting"
            );
        }

        // A client that wants the fast regime back can still ask for it explicitly.
        let opted_out = resolve_sampler_config(
            ClientSampling {
                top_p: Some(1.0),
                top_k: Some(0),
                seed: Some(1),
                ..Default::default()
            },
            &qwen38_vendor_defaults(),
        );
        assert!(
            memra_engine::sampler::Sampler::new(opted_out).is_spec_sampling(),
            "explicitly disabling the filters must restore the pure-temp regime"
        );
    }

    #[test]
    fn omitted_seed_is_fresh_entropy_not_a_pinned_zero() {
        // dogfood F4, SECOND HALF — found only by driving the live server. Fixing the
        // temperature default is NOT sufficient: `#[serde(default)] seed: u64` gave 0, a
        // perfectly valid FIXED seed, so a temp-1.0 request with seed omitted still replayed
        // one single sampled stream. Measured on the pre-fix binary: 4/4 byte-identical
        // completions at temperature 1.0 with seed omitted (receipts in
        // research/sampledspec-20260804/). The loop survives the temperature fix alone.
        let comp_seed = |body: serde_json::Value| {
            let req: CompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_request(&req, tx, lanes::Lane::Interactive, None)
                .sampler_cfg
                .seed
        };
        let chat_seed = |body: serde_json::Value| {
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
                .unwrap()
                .request
                .sampler_cfg
                .seed
        };

        // OMITTED seed: successive requests must NOT share a seed (that was the loop), and
        // must not be the old pinned 0.
        let a = comp_seed(serde_json::json!({"model": "m", "prompt": "t"}));
        let b = comp_seed(serde_json::json!({"model": "m", "prompt": "t"}));
        let c = chat_seed(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}]}));
        assert_ne!(
            a, 0,
            "omitted seed must not be the pinned 0 that caused the loop"
        );
        assert_ne!(b, 0);
        assert_ne!(c, 0);
        assert_ne!(
            a, b,
            "two seed-omitting requests must get DIFFERENT streams"
        );
        assert_ne!(a, c);

        // EXPLICIT seed is honored exactly — including an explicit 0, which every
        // determinism gate in tools/ and research/ relies on.
        assert_eq!(
            comp_seed(serde_json::json!({
            "model": "m", "prompt": "t", "seed": 0})),
            0,
            "explicit seed 0 must stay 0 — the determinism gates depend on it"
        );
        assert_eq!(
            comp_seed(serde_json::json!({
            "model": "m", "prompt": "t", "seed": 12345})),
            12345
        );
        assert_eq!(
            chat_seed(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "seed": 777})),
            777
        );
        // explicit seed is reproducible across calls (the gate contract).
        assert_eq!(
            comp_seed(serde_json::json!({"model": "m", "prompt": "t", "seed": 42})),
            comp_seed(serde_json::json!({"model": "m", "prompt": "t", "seed": 42}))
        );

        // fresh_seed itself: never 0, and distinct across rapid successive calls (the
        // same-nanosecond batched-arrival case the counter mix exists for).
        let seeds: std::collections::HashSet<u64> = (0..256).map(|_| fresh_seed()).collect();
        assert_eq!(
            seeds.len(),
            256,
            "fresh_seed must not collide across rapid calls"
        );
        assert!(!seeds.contains(&0));
    }

    #[test]
    fn response_format_builds_grammar_only_when_present() {
        // NO-OP CONTRACT (lane/constrained): absent / {"type":"text"} => grammar None —
        // the worker Request is field-identical to a pre-lane request, no llguidance
        // object is ever built. json_object / json_schema arm the grammar.
        let mk = |rf: Option<serde_json::Value>| {
            let mut body = serde_json::json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}]});
            if let Some(rf) = rf {
                body["response_format"] = rf;
            }
            let req: ChatCompletionReq = serde_json::from_value(body).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request(req, None, tx, lanes::Lane::Interactive, None)
        };
        assert!(mk(None).unwrap().request.grammar.is_none());
        assert!(
            mk(Some(serde_json::json!({"type": "text"})))
                .unwrap()
                .request
                .grammar
                .is_none()
        );
        assert!(matches!(
            mk(Some(serde_json::json!({"type": "json_object"})))
                .unwrap()
                .request
                .grammar,
            Some(constrained::GrammarSpec::JsonObject)
        ));
        assert!(matches!(
            mk(Some(serde_json::json!({"type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}})))
            .unwrap()
            .request
            .grammar,
            Some(constrained::GrammarSpec::JsonSchema(_))
        ));
        // unknown type: loud error, never silent.
        assert!(mk(Some(serde_json::json!({"type": "yaml"}))).is_err());
    }

    /// GRAMMAR x THINK admit/refuse table (lane/step37-postthink-grammar, 2026-08-30).
    /// Three template classes, three verdicts:
    ///   switch-carrying (qwen): think forced OFF, grammar from token 1 — byte-identical
    ///     to the pre-lane path;
    ///   think-forced WITH a derivable close contract (step37): ADMITTED, think stays ON
    ///     (post-think two-phase — the worker arms the gate from the same load-time
    ///     contract);
    ///   think-forced with NO derivable close contract: the loud 400 stays — never a
    ///     silent constrain-from-token-1 stream.
    #[test]
    fn response_format_think_table_switch_postthink_refusal() {
        let mk = |caps: &ModelCaps| {
            let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
                "model": "m", "messages": [{"role": "user", "content": "t"}],
                "response_format": {"type": "json_object"}}))
            .unwrap();
            let (tx, _rx) = worker::event_channel();
            build_chat_request(req, Some(caps), tx, lanes::Lane::Interactive, None)
        };
        // qwen class: enable_thinking switch — grammar path forces NoThink, unchanged.
        let switch = ModelCaps {
            chat_ok: true,
            qwen_think: true,
            think_switch: true,
            ..Default::default()
        };
        let plan = mk(&switch).unwrap();
        assert_eq!(
            plan.request.think,
            memra_tokenizer::chat::ThinkMode::NoThink,
            "switch-carrying template must keep the grammar-from-token-1 path"
        );
        assert!(plan.request.grammar.is_some());

        // step37 class: think-forced, close contract derivable — admitted, think ON.
        let postthink = ModelCaps {
            chat_ok: true,
            qwen_think: true,
            think_switch: false,
            think_close: vec![128799],
            ..Default::default()
        };
        let plan = mk(&postthink).unwrap();
        assert_ne!(
            plan.request.think,
            memra_tokenizer::chat::ThinkMode::NoThink,
            "post-think constrained request must keep the think channel ON"
        );
        assert!(plan.request.grammar.is_some());

        // think-forced, NO contract: the loud refusal stays.
        let no_contract = ModelCaps {
            chat_ok: true,
            qwen_think: true,
            think_switch: false,
            think_close: Vec::new(),
            ..Default::default()
        };
        let err = match mk(&no_contract) {
            Err(err) => err,
            Ok(_) => panic!("think-forced template with no close contract must refuse"),
        };
        assert!(
            err.contains("think-close"),
            "refusal must name the missing close contract: {err}"
        );
    }

    #[test]
    fn unsupported_semantic_params_are_named_rejections() {
        // gap-scan F4: fields serde used to swallow now deserialize into rejection slots.
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "response_format": {"type": "json_object"}
        }))
        .unwrap();
        assert!(req.response_format.is_some());
        let req: ChatCompletionReq = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [{"role": "user", "content": "t"}],
            "response_format": {"type": "text"}, "logprobs": false, "n": 1,
            "user": "u-1", "stream_options": {"include_usage": true}
        }))
        .unwrap();
        // the no-op forms + cosmetic fields: all fine (accept-and-ignore class).
        assert_eq!(req.response_format.as_ref().unwrap()["type"], "text");
        assert_eq!(req.logprobs.as_ref().unwrap().as_bool(), Some(false));
        assert_eq!(req.n, Some(1));
        // the gate law itself: present -> named error, absent -> Ok.
        assert!(reject_unsupported(&[("logit_bias", false, "")]).is_ok());
        let (msg, param) = reject_unsupported(&[("logit_bias", true, " (why)")]).unwrap_err();
        assert_eq!(param, "logit_bias");
        assert_eq!(msg, "logit_bias is not supported (why)");
    }

    #[test]
    fn completions_accept_openai_stop_forms() {
        for (value, expected) in [
            (serde_json::json!("Problem:"), vec!["Problem:"]),
            (
                serde_json::json!(["Question:", "Problem:"]),
                vec!["Question:", "Problem:"],
            ),
            (serde_json::Value::Null, Vec::<&str>::new()),
        ] {
            let req: CompletionReq = serde_json::from_value(serde_json::json!({
                "model": "plain_quant", "prompt": "task", "stop": value
            }))
            .unwrap();
            assert_eq!(req.stop.into_vec(), expected);
        }
    }

    /// Fake GPU worker: consumes Generate commands and answers each with one Token +
    /// Done — handler-level tests (headers, drain) without a GPU or a loaded model.
    ///
    /// It also drives the SAME health handle the real worker does (mark_ready at "load"
    /// completion, beat_busy per iteration), which is what lets the /health and /readyz tests
    /// exercise the real handlers instead of a mock.
    fn fake_worker_state() -> AppState {
        fake_worker_state_with_steps(1, std::time::Duration::ZERO)
    }

    fn fake_worker_state_with_steps(steps: usize, step_delay: std::time::Duration) -> AppState {
        fake_worker_state_full(steps, step_delay, HashMap::new(), None)
    }

    /// What the fake worker SAW for one admitted request — the worker-truth fields the
    /// surface-parity tests compare: the resolved sampling AND the resolved reasoning
    /// surface (issue #31: /v1/messages dropped `output_config.effort` before this point,
    /// so only a worker-boundary tap can prove the effect half of effort parity).
    struct WorkerSaw {
        sampler_cfg: SamplerConfig,
        think: ThinkMode,
        reasoning_effort: Option<String>,
    }

    /// Fake worker with per-model `caps` and a WORKER-TRUTH tap: each admitted request's
    /// resolved `WorkerSaw` snapshot is sent on `saw_tx` the moment the worker receives
    /// it — i.e. what the engine would actually run with, after every
    /// surface/translation/default layer has run. Surface-parity tests read this instead
    /// of a build helper so a divergence ANYWHERE in a handler path (not just in the
    /// shared resolver) fails the test.
    fn fake_worker_state_full(
        steps: usize,
        step_delay: std::time::Duration,
        caps: HashMap<String, ModelCaps>,
        saw_tx: Option<std::sync::mpsc::Sender<WorkerSaw>>,
    ) -> AppState {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        let health = health::WorkerHealth::new();
        let h = health.clone();
        std::thread::spawn(move || {
            h.mark_ready();
            while let Ok(Cmd::Generate(mut req)) = cmd_rx.recv() {
                if let Some(tx) = &saw_tx {
                    let _ = tx.send(WorkerSaw {
                        sampler_cfg: req.sampler_cfg.clone(),
                        think: req.think,
                        reasoning_effort: req.reasoning_effort.clone(),
                    });
                }
                // Mirror handle_cmd: handlers reserve both the burst-yield gauge and the hard
                // queue bound before send. A fake worker must release both at its admission
                // boundary or leak process-global state into unrelated tests.
                worker::release_pending_admit();
                worker::release_admission_reservation(req.lane);
                h.beat_busy();
                if let Some(ready) = req.constraint_ready.take() {
                    let _ = ready.send(Ok(()));
                }
                let _ = req.tx.send(Event::PromptUsage {
                    n_prompt: 1,
                    n_cached: 0,
                });
                // Capture requests (embeddings/rerank) read the prompt's last position: the
                // real worker answers PromptCapture before Done, and the route 500s without
                // it. A fixed two-wide hidden state and a yes>no logit pair are enough for
                // the handler-level tests (unit-norm pooling, top-index ordering).
                if let Some(spec) = req.capture.as_ref() {
                    let _ = req.tx.send(Event::PromptCapture {
                        hidden: spec.hidden.then(|| vec![1.0, 0.0]),
                        logits: if spec.logit_pieces.is_empty() {
                            Vec::new()
                        } else {
                            vec![2.0, 0.0]
                        },
                    });
                }
                for step in 0..steps {
                    h.beat_busy();
                    let text = if steps == 1 { "ok" } else { "x" };
                    let _ = req.tx.send(Event::Token {
                        id: step as u32 + 1,
                        text: text.into(),
                    });
                    if !step_delay.is_zero() {
                        std::thread::sleep(step_delay);
                    }
                }
                let _ = req.tx.send(Event::Done {
                    stop_reason: "Eos".into(),
                    n_tokens: steps,
                    n_prompt: 1,
                    n_cached: 0,
                    elapsed_s: 0.01,
                    spec: None,
                });
                h.set_phase(health::PHASE_IDLE);
            }
        });
        // The spawn above is the "load"; wait for its ready stamp so a health assertion is not
        // racing the thread start (the real path blocks on ready_tx for the same reason).
        for _ in 0..2000 {
            if health.live().is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AppState {
            cmd_tx,
            models: Arc::new(vec!["m".into()]),
            caps: Arc::new(caps),
            openrouter_metadata: Arc::new(HashMap::new()),
            provider_metadata: Arc::new(None),
            metering: None,

            budget_tokenizers: None,
            api_auth: ApiAuth::default(),
            metrics_auth: MetricsAuth::default(),
            metrics: SharedMetrics::default(),
            inflight: Arc::new(Default::default()),
            tenant_inflight: Arc::new(Default::default()),
            health,
            bg: None,
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn deep_schema_fails_while_normal_decode_keeps_stepping() {
        let _l = drain_lock();
        let st = fake_worker_state_with_steps(64, std::time::Duration::from_millis(5));
        let normal_state = st.clone();
        let normal = tokio::spawn(async move {
            chat_completions(
                State(normal_state),
                axum::http::HeaderMap::new(),
                None,
                Json(
                    serde_json::from_value(serde_json::json!({
                        "model": "m",
                        "messages": [{"role": "user", "content": "keep decoding"}],
                    }))
                    .unwrap(),
                ),
            )
            .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;

        let mut deep = serde_json::json!({"type": "string"});
        for _ in 0..(constrained::MAX_SCHEMA_DEPTH / 2 + 1) {
            deep = serde_json::json!({"allOf": [deep]});
        }
        let bad = chat_completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": "bad schema"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"schema": deep},
                    },
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        assert_eq!(bad.headers().get("x-should-retry").unwrap(), "false");
        let bytes = axum::body::to_bytes(bad.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("maximum nesting depth")
        );
        assert!(
            !normal.is_finished(),
            "bad schema stalled or replaced the normal decode"
        );

        let normal_response = normal.await.unwrap();
        assert_eq!(normal_response.status(), StatusCode::OK);
        let snapshot = st.health.snapshot();
        assert!(
            st.health.live().is_ok(),
            "normal decode left health stalled"
        );
        assert!(snapshot.beat_age_ms < snapshot.stall_threshold_ms);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn valid_response_format_preflight_preserves_generation() {
        let _l = drain_lock();
        let response = chat_completions(
            State(fake_worker_state()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": "valid schema"}],
                    "response_format": {"type": "json_object"},
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["choices"][0]["message"]["content"], "ok");
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn unknown_model_refuses_model_not_found_before_admission() {
        let _l = drain_lock();
        // The fake worker answers ANY admitted request with "ok", so a model_not_found
        // response proves the handler refused BEFORE worker admission — and a fortiori
        // before prepaid budget reservation, which sits between (the live bug: a typo'd
        // model id on a budgeted tenant surfaced as a 503 about budget accounting).
        let response = chat_completions(
            State(fake_worker_state()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "qwen/qwen3.8-27b-typo",
                    "messages": [{"role": "user", "content": "hi"}],
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["error"]["code"], "model_not_found");
        assert_eq!(payload["error"]["type"], "invalid_request_error");

        // Same law on the text-completions surface.
        let response = completions(
            State(fake_worker_state()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "nope",
                    "prompt": "hi",
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["error"]["code"], "model_not_found");
    }

    const METRICS_KEY_ACME: &str = "completion-acme-secret";
    const METRICS_KEY_BLUE: &str = "completion-blue-secret";

    fn multi_key_metrics_state(metrics_token: Option<&str>) -> AppState {
        let spec = format!(
            "acme:{},blue:{}",
            auth::sha256_hex(METRICS_KEY_ACME),
            auth::sha256_hex(METRICS_KEY_BLUE),
        );
        let keyring = Box::leak(Box::new(auth::KeyStore::from_spec(&spec).unwrap()));
        let mut st = fake_worker_state();
        st.api_auth.keyring = Some(keyring);
        st.metrics_auth = MetricsAuth::new(
            true,
            st.api_auth.configured(),
            metrics_token.map(str::to_string),
        );
        {
            let mut metrics = st.metrics.lock().unwrap();
            metrics.admitted = 17;
            metrics.prompt_tokens_in = 400;
            metrics.cached_tokens_in = 60;
            metrics.prefix_hits = 2;
            metrics.prefix_misses = 3;
            metrics.prefix_inserts = 5;
            metrics.prefix_evictions = 7;
            metrics.prefix_skips_budget = 9;
            metrics.prefix_skips_pinned = 10;
            metrics.prefix_hit_tokens = 11;
            metrics.lcp_hist[4] = 13;
            metrics.ns_tokens.insert("t:acme".into(), [100, 40]);
            metrics.ns_tokens.insert("t:blue".into(), [300, 20]);
            metrics.adsd_suspect_total.insert("t:acme".into(), 1);
            metrics.adsd_suspect_total.insert("t:blue".into(), 2);
            metrics.prefix_entries = 29;
            metrics.prefix_bytes = 31;
            metrics.active_sessions = 3;
            metrics.queued_requests = 5;
            metrics.admission_inflight.insert("m".into(), 4);
            metrics
                .admission_booked_bytes
                .insert("m".into(), 41_000_000);
            metrics.continuation_pool_entries = 7;
            metrics.spec_pool_entries = 11;
            metrics.cuda_driver_free_bytes = 13;
            metrics.cuda_pool_reserved_bytes = 17;
            metrics.cuda_pool_used_bytes = 19;
            metrics.cuda_pool_cached_bytes = 23;
            metrics.batch_size_last = 37;
            metrics.spec.insert(
                "m".into(),
                memra_engine::spec::SpecTelemetry {
                    rounds: 2,
                    drafted: 6,
                    accepted: 4,
                    ..Default::default()
                },
            );
            let mut spec_window = memra_engine::spec::SpecTelemetry {
                rounds: 4,
                drafted: 12,
                accepted: 6,
                ..Default::default()
            };
            spec_window.pos_drafted[..3].copy_from_slice(&[4, 4, 4]);
            spec_window.pos_accepted[..3].copy_from_slice(&[3, 2, 1]);
            metrics.spec_window.insert("m".into(), spec_window);
            metrics.constraint_compiler_fail_closed.insert(
                "m".into(),
                Arc::new(std::sync::atomic::AtomicBool::new(true)),
            );
        }
        st
    }

    async fn metrics_json(st: AppState, bearer: &str) -> serde_json::Value {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", format!("Bearer {bearer}").parse().unwrap());
        let response = get_metrics(State(st), headers).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    async fn yield_metrics_json(st: AppState, bearer: &str) -> serde_json::Value {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", format!("Bearer {bearer}").parse().unwrap());
        let response = yield_metrics(State(st), headers).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn exposed_open_bind_is_refused_before_server_start() {
        assert!(validate_bind_security("127.0.0.1:8080", false, false).unwrap());
        assert!(validate_bind_security("[::1]:8080", false, false).unwrap());

        let err = validate_bind_security("0.0.0.0:8000", false, false).unwrap_err();
        assert!(err.contains("refusing unauthenticated non-loopback bind"));
        assert!(err.contains("MEMRA_API_KEY"));
        assert!(err.contains("MEMRA_ALLOW_OPEN_BIND=1"));
        assert!(validate_bind_security("[::]:8000", false, false).is_err());

        assert!(!validate_bind_security("0.0.0.0:8000", true, false).unwrap());
        assert!(!validate_bind_security("0.0.0.0:8000", false, true).unwrap());
    }

    #[tokio::test]
    async fn keyed_metrics_require_and_accept_api_bearer() {
        let mut st = fake_worker_state();
        st.api_auth.single_key = Some(Arc::from("completion-secret"));
        st.metrics_auth = MetricsAuth::new(true, st.api_auth.configured(), None);

        let response = get_metrics(State(st.clone()), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let response = yield_metrics(State(st.clone()), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer completion-secret".parse().unwrap());
        assert_eq!(
            get_metrics(State(st.clone()), headers.clone())
                .await
                .status(),
            StatusCode::OK,
        );
        let body = metrics_json(st.clone(), "completion-secret").await;
        assert!(
            body.get("admitted").is_some(),
            "the legacy single-key domain keeps cumulative counters",
        );
        assert!(
            body.get("active_sessions").is_none(),
            "a static completion key is not an operator metrics principal",
        );
        assert_eq!(
            yield_metrics(State(st), headers).await.status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn keyring_metrics_bearer_sees_only_its_tenant_rows() {
        let st = multi_key_metrics_state(None);
        let body = metrics_json(st.clone(), METRICS_KEY_ACME).await;
        assert_eq!(
            body.as_object().unwrap().len(),
            2,
            "completion metrics must contain only tenant-scoped rows",
        );
        let tenants = body["tenants"].as_object().unwrap();
        assert_eq!(tenants.len(), 1);
        assert_eq!(tenants["t:acme"]["prompt_tokens_in"], 100);
        assert!(!tenants.contains_key("t:blue"));
        let adsd = body["adsd_suspect_total"].as_object().unwrap();
        assert_eq!(adsd.len(), 1);
        assert_eq!(adsd["t:acme"], 1);
        assert!(!adsd.contains_key("t:blue"));

        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            format!("Bearer {METRICS_KEY_ACME}").parse().unwrap(),
        );
        assert_eq!(
            yield_metrics(State(st), headers).await.status(),
            StatusCode::FORBIDDEN,
            "the process-wide yield view requires an operator metrics token",
        );
    }

    #[tokio::test]
    async fn tenant_metrics_hide_capacity_and_aggregate_spec() {
        let body = metrics_json(multi_key_metrics_state(None), METRICS_KEY_ACME).await;
        for operator_only in [
            "prefix_cache_entries",
            "prefix_cache_bytes",
            "prefix_cache_skips_budget",
            "prefix_cache_skips_pinned",
            "active_sessions",
            "queued_requests",
            "admission_inflight",
            "admission_booked_bytes",
            "continuation_pool_entries",
            "spec_pool_entries",
            "cuda_driver_free_bytes",
            "cuda_pool_reserved_bytes",
            "cuda_pool_used_bytes",
            "cuda_pool_cached_bytes",
            "constraint_compiler_fail_closed",
            "serve_idle_seconds",
            "spec",
            "spec_tau",
            "spec_accept_by_position",
            "dual_pp",
            "pp_wave",
            "peer_probe_bypassed",
            "peer_probe_boundary_copies",
            "peer_probe_runtime_reprobes",
            "peer_probe_runtime_failures",
            "peer_probe_deferred_total",
            "peer_probe_integrity_degraded",
            "peer_probe_degraded_to_host_bounce",
        ] {
            assert!(
                body.get(operator_only).is_none(),
                "tenant metrics must not expose operator field {operator_only}",
            );
        }
    }

    #[test]
    fn populated_spec_acceptance_metrics_are_operator_only() {
        for scope in [
            MetricsScope::CompletionDomain,
            MetricsScope::Tenant("t:acme".into()),
        ] {
            let mut body = json!({});
            insert_spec_acceptance_metrics(&mut body, &scope, || {
                panic!("tenant scope evaluated the process-wide spec snapshot")
            });
            assert!(body.get("spec_tau").is_none(), "{scope:?} leaked spec tau");
            assert!(
                body.get("spec_accept_by_position").is_none(),
                "{scope:?} leaked the accept histogram"
            );
        }

        let mut telemetry = memra_engine::spec::SpecTelemetry {
            rounds: 4,
            drafted: 12,
            accepted: 6,
            ..Default::default()
        };
        telemetry.pos_drafted[..3].copy_from_slice(&[4, 4, 4]);
        telemetry.pos_accepted[..3].copy_from_slice(&[3, 2, 1]);
        let mut body = json!({});
        insert_spec_acceptance_metrics(&mut body, &MetricsScope::All, || {
            HashMap::from([("model-a".to_string(), telemetry)])
        });
        assert_eq!(body["spec_tau"]["model-a"], 1.5);
        let histogram = &body["spec_accept_by_position"]["model-a"];
        assert_eq!(histogram["window_seconds"], worker::SPEC_METRICS_WINDOW_S);
        assert_eq!(histogram["rounds"], 4);
        assert_eq!(histogram["offered"], json!([4, 4, 4]));
        assert_eq!(histogram["accepted"], json!([3, 2, 1]));
        assert_eq!(histogram["accept_rate"], json!([0.75, 0.5, 0.25]));
    }

    #[test]
    fn populated_dual_pp_metrics_are_operator_only() {
        let populated = DualPpMetricsSnapshot {
            stage_ns: [1_000_000, 2_000_000, 3_000_000, 4_000_000],
            stage_samples: [1, 1, 1, 1],
            dropped_timing_samples: 0,
            overlaps: 17,
            slot_pairs: 19,
            slot_uses: [19, 19],
            slot_collisions: 0,
        };
        for scope in [
            MetricsScope::CompletionDomain,
            MetricsScope::Tenant("t:acme".into()),
        ] {
            let mut body = json!({});
            insert_dual_pp_metrics(&mut body, &scope, || populated);
            assert!(
                body.get("dual_pp").is_none(),
                "{scope:?} leaked dual PP topology"
            );
        }

        let mut body = json!({});
        insert_dual_pp_metrics(&mut body, &MetricsScope::All, || populated);
        assert_eq!(body["dual_pp"]["overlaps"], 17);
        assert_eq!(body["dual_pp"]["slot_pairs"], 19);
        assert_eq!(body["dual_pp"]["slot_uses"], json!([19, 19]));
        assert_eq!(body["dual_pp"]["slot_collisions"], 0);
        assert_eq!(
            body["dual_pp"]["cuda_event_spans"]["wave_a_stage0"]["mean_ms"],
            1.0
        );
    }

    #[test]
    fn populated_pp_wave_metrics_are_operator_only() {
        let populated = PpWaveMetricsSnapshot {
            ticks: 11,
            cells: 96,
            overlaps: 37,
        };
        for scope in [
            MetricsScope::CompletionDomain,
            MetricsScope::Tenant("t:acme".into()),
        ] {
            let mut body = json!({});
            insert_pp_wave_metrics(&mut body, &scope, || populated);
            assert!(
                body.get("pp_wave").is_none(),
                "{scope:?} leaked PP wave topology"
            );
        }

        let mut body = json!({});
        insert_pp_wave_metrics(&mut body, &MetricsScope::All, || populated);
        assert_eq!(body["pp_wave"]["ticks"], 11);
        assert_eq!(body["pp_wave"]["cells"], 96);
        assert_eq!(body["pp_wave"]["overlaps"], 37);
    }

    #[test]
    fn peer_probe_metrics_are_operator_only() {
        let populated = memra_engine::pp::PeerProbeMetrics {
            bypassed: 1,
            boundary_copies: 8_192,
            runtime_probes: 1,
            runtime_failures: 0,
            deferred_total: 4,
            integrity_degraded: true,
            degraded_to_host_bounce: true,
        };
        for scope in [
            MetricsScope::CompletionDomain,
            MetricsScope::Tenant("t:acme".into()),
        ] {
            let mut body = json!({});
            insert_peer_probe_metrics(&mut body, &scope, || populated);
            assert!(body.get("peer_probe_bypassed").is_none());
        }

        let mut body = json!({});
        insert_peer_probe_metrics(&mut body, &MetricsScope::All, || populated);
        assert_eq!(body["peer_probe_bypassed"], 1);
        assert_eq!(body["peer_probe_boundary_copies"], 8_192);
        assert_eq!(body["peer_probe_runtime_reprobes"], 1);
        assert_eq!(body["peer_probe_runtime_failures"], 0);
        assert_eq!(body["peer_probe_deferred_total"], 4);
        assert_eq!(body["peer_probe_integrity_degraded"], true);
        assert_eq!(body["peer_probe_degraded_to_host_bounce"], true);
    }

    #[tokio::test]
    async fn prefix_aggregate_metrics_are_operator_only_but_tenant_ratio_remains() {
        let tenant_body = metrics_json(multi_key_metrics_state(None), METRICS_KEY_ACME).await;
        for operator_only in [
            "lcp_histogram",
            "cache_hit_token_ratio",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "prefix_cache_inserts",
            "prefix_cache_evictions",
            "prefix_cache_skips_budget",
            "prefix_cache_skips_pinned",
            "prefix_cache_hit_tokens",
        ] {
            assert!(
                tenant_body.get(operator_only).is_none(),
                "tenant metrics must not expose global prefix field {operator_only}",
            );
        }
        assert_eq!(tenant_body["tenants"].as_object().unwrap().len(), 1);
        assert_eq!(tenant_body["tenants"]["t:acme"]["prompt_tokens_in"], 100);
        assert_eq!(tenant_body["tenants"]["t:acme"]["cached_tokens_in"], 40);
        assert_eq!(
            tenant_body["tenants"]["t:acme"]["cache_hit_token_ratio"],
            0.4
        );

        let operator_body = metrics_json(
            multi_key_metrics_state(Some("scrape-secret")),
            "scrape-secret",
        )
        .await;
        assert_eq!(operator_body["prefix_cache_hits"], 2);
        assert_eq!(operator_body["prefix_cache_misses"], 3);
        assert_eq!(operator_body["prefix_cache_inserts"], 5);
        assert_eq!(operator_body["prefix_cache_evictions"], 7);
        assert_eq!(operator_body["prefix_cache_skips_budget"], 9);
        assert_eq!(operator_body["prefix_cache_skips_pinned"], 10);
        assert_eq!(operator_body["prefix_cache_hit_tokens"], 11);
        assert_eq!(operator_body["cache_hit_token_ratio"], 0.15);
        assert_eq!(operator_body["lcp_histogram"]["counts"][4], 13);
    }

    #[tokio::test]
    async fn configured_metrics_token_is_exclusive_and_sees_all_tenants() {
        let st = multi_key_metrics_state(Some("scrape-secret"));
        let mut completion_headers = HeaderMap::new();
        completion_headers.insert(
            "authorization",
            format!("Bearer {METRICS_KEY_ACME}").parse().unwrap(),
        );
        assert_eq!(
            get_metrics(State(st.clone()), completion_headers.clone())
                .await
                .status(),
            StatusCode::FORBIDDEN,
        );
        assert_eq!(
            yield_metrics(State(st.clone()), completion_headers)
                .await
                .status(),
            StatusCode::FORBIDDEN,
        );

        let body = metrics_json(st.clone(), "scrape-secret").await;
        let tenants = body["tenants"].as_object().unwrap();
        assert_eq!(tenants.len(), 2);
        assert!(tenants.contains_key("t:acme"));
        assert!(tenants.contains_key("t:blue"));
        assert_eq!(body["adsd_suspect_total"]["t:acme"], 1);
        assert_eq!(body["adsd_suspect_total"]["t:blue"], 2);
        assert_eq!(body["active_sessions"], 3);
        assert_eq!(body["queued_requests"], 5);
        // D2 gap G2: the per-model admission book is an operator surface.
        assert_eq!(body["admission_inflight"]["m"], 4);
        assert_eq!(body["admission_booked_bytes"]["m"], 41_000_000);
        assert_eq!(body["prefix_cache_bytes"], 31);
        assert_eq!(body["cuda_driver_free_bytes"], 13);
        assert_eq!(body["constraint_compiler_fail_closed"]["m"], 1);
        assert_eq!(body["spec"]["m"]["drafted"], 6);
        assert_eq!(body["spec_tau"]["m"], 1.5);
        assert_eq!(
            body["spec_accept_by_position"]["m"]["accepted"],
            json!([3, 2, 1])
        );
        let yield_body = yield_metrics_json(st, "scrape-secret").await;
        assert_eq!(yield_body["batch_size_last"], 37);
    }

    #[tokio::test]
    async fn metrics_token_protects_public_override_without_api_keys() {
        let mut st = fake_worker_state();
        st.metrics_auth = MetricsAuth::new(false, false, Some("scrape-secret".into()));

        assert_eq!(
            get_metrics(State(st.clone()), HeaderMap::new())
                .await
                .status(),
            StatusCode::UNAUTHORIZED,
        );
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer scrape-secret".parse().unwrap());
        assert_eq!(
            get_metrics(State(st.clone()), headers.clone())
                .await
                .status(),
            StatusCode::OK,
        );
        assert_eq!(
            yield_metrics(State(st), headers).await.status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn no_key_loopback_metrics_remain_open_for_development() {
        let mut st = fake_worker_state();
        st.metrics_auth = MetricsAuth::new(true, false, None);
        let response = get_metrics(State(st.clone()), HeaderMap::new()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            body.get("active_sessions").is_some(),
            "no-key loopback development keeps full operator visibility",
        );
        assert_eq!(
            yield_metrics(State(st), HeaderMap::new()).await.status(),
            StatusCode::OK,
        );
    }

    #[test]
    fn rate_limit_math_remaining_hits_zero_at_cap_and_reset_arms() {
        let metrics = SharedMetrics::default();
        // free slots: remaining counts down, reset stays 0.
        let rl = RateLimit::compute(4, 1, &metrics);
        assert_eq!((rl.limit, rl.remaining, rl.reset_s), (4, 3, 0));
        let rl = RateLimit::compute(4, 3, &metrics);
        assert_eq!(rl.remaining, 1);
        // at cap: remaining 0, reset arms (static default — no meter signal here).
        let rl = RateLimit::compute(4, 4, &metrics);
        assert_eq!(rl.remaining, 0);
        assert!(rl.reset_s > 0, "reset must arm when no slots are free");
        // over cap (queued interactive): saturates at 0, never underflows.
        assert_eq!(RateLimit::compute(4, 9, &metrics).remaining, 0);
        // meter signal: reset = mean tokens/request x p50 step, ceil seconds.
        let m = worker::Metrics {
            completed: 2,
            tokens_out: 200,
            step_p50_ms: 20.0,
            ..Default::default()
        };
        assert_eq!(reset_estimate_s(&m), 2); // 100 tok x 20ms = 2.0s
    }

    #[test]
    fn inflight_guard_counts_up_and_frees_on_drop() {
        let counts: InflightCounts = Arc::new(Default::default());
        let tenants: TenantGauge = Arc::new(Default::default());
        let (g1, n1, t1) = InflightGuard::try_acquire(
            counts.clone(),
            lanes::Lane::Interactive,
            tenants.clone(),
            "acme",
            None,
        )
        .unwrap();
        let (g2, n2, t2) = InflightGuard::try_acquire(
            counts.clone(),
            lanes::Lane::Interactive,
            tenants.clone(),
            "acme",
            None,
        )
        .unwrap();
        assert_eq!((n1, n2), (1, 2));
        // tenant gauge counts per tenant, across lanes.
        assert_eq!((t1, t2), (1, 2));
        // lanes are independent gauges; a different tenant starts at 1.
        let (gj, nj, tj) = InflightGuard::try_acquire(
            counts.clone(),
            lanes::Lane::Judge,
            tenants.clone(),
            "blue",
            None,
        )
        .unwrap();
        assert_eq!((nj, tj), (1, 1));
        drop(g1);
        drop(gj);
        assert_eq!(counts[0].load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(counts[1].load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(tenants.lock().unwrap().get("acme"), Some(&1));
        // tenant entries are removed at zero (bounded by CONCURRENT tenants).
        assert!(tenants.lock().unwrap().get("blue").is_none());
        drop(g2);
        assert_eq!(counts[0].load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(tenants.lock().unwrap().is_empty());
    }

    #[test]
    fn tenant_concurrency_cap_is_atomic_across_arrivals() {
        let counts: InflightCounts = Arc::new(Default::default());
        let tenants: TenantGauge = Arc::new(Default::default());
        let start = Arc::new(std::sync::Barrier::new(3));
        let attempted = Arc::new(std::sync::Barrier::new(3));
        let mut joins = Vec::new();
        for _ in 0..2 {
            let counts = counts.clone();
            let tenants = tenants.clone();
            let start = start.clone();
            let attempted = attempted.clone();
            joins.push(std::thread::spawn(move || {
                start.wait();
                let result = InflightGuard::try_acquire(
                    counts,
                    lanes::Lane::Interactive,
                    tenants,
                    "preview_001",
                    Some(1),
                );
                let won = result.is_ok();
                attempted.wait(); // winner holds its guard until both arrivals attempted.
                drop(result);
                won
            }));
        }
        start.wait();
        attempted.wait();
        let wins = joins
            .into_iter()
            .map(|join| join.join().unwrap())
            .filter(|won| *won)
            .count();
        assert_eq!(wins, 1, "exactly one simultaneous request may pass cap=1");
        assert_eq!(counts[0].load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(tenants.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn tenant_concurrency_cap_rejects_before_worker_admission() {
        let st = fake_worker_state();
        let tenant = auth::TenantCtx {
            tenant: "preview_001".into(),
            lane_class: auth::LaneClass::Interactive,
            rate_limit: Some(1),
            key_prefix: None,
        };
        let first_env = Envelope::new(true);
        let (guard, first_rl) =
            match acquire_request_slot(&st, lanes::Lane::Interactive, &tenant, &first_env) {
                Ok(slot) => slot,
                Err(_) => panic!("the first request must acquire the tenant slot"),
            };
        assert_eq!((first_rl.limit, first_rl.remaining), (1, 0));

        let second_env = Envelope::new(true);
        let response =
            match acquire_request_slot(&st, lanes::Lane::Interactive, &tenant, &second_env) {
                Err(response) => response,
                Ok(_) => panic!("the second request must be rejected at the tenant cap"),
            };
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()["retry-after"], "2");
        assert_eq!(response.headers()["retry-after-ms"], "2000");
        assert_eq!(response.headers()["x-ratelimit-limit"], "1");
        assert_eq!(response.headers()["x-ratelimit-remaining"], "0");
        assert_eq!(response.headers()["x-request-id"], second_env.id);
        assert_eq!(
            st.inflight[0].load(std::sync::atomic::Ordering::SeqCst),
            1,
            "rejected request must not consume a lane slot"
        );
        assert_eq!(
            st.tenant_inflight
                .lock()
                .unwrap()
                .get("preview_001")
                .copied(),
            Some(1),
            "rejected request must not increment the tenant gauge"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["error"]["type"], "rate_limit_error");
        assert_eq!(payload["error"]["code"], "rate_limit_exceeded");
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("concurrent request limit")
        );

        drop(guard);
        let _ = InflightGuard::try_acquire(
            st.inflight.clone(),
            lanes::Lane::Interactive,
            st.tenant_inflight.clone(),
            "preview_001",
            Some(1),
        )
        .expect("slot must reopen after the in-flight request completes");
    }

    #[test]
    fn tenant_rate_limit_override_is_min_with_global_cap() {
        let metrics = SharedMetrics::default();
        let unlimited = auth::TenantCtx::default_tenant();
        let capped = auth::TenantCtx {
            tenant: "acme".into(),
            lane_class: auth::LaneClass::Interactive,
            rate_limit: Some(2),
            key_prefix: None,
        };
        let global = lane_cap(lanes::Lane::Interactive);
        // no override: the global lane cap reports as before.
        let rl = RateLimit::at_admit(lanes::Lane::Interactive, 1, &metrics, &unlimited, 1);
        assert_eq!((rl.limit, rl.remaining), (global, global - 1));
        // override binds: limit = the tenant cap, remaining counts the TENANT gauge.
        let rl = RateLimit::at_admit(lanes::Lane::Interactive, 5, &metrics, &capped, 1);
        assert_eq!((rl.limit, rl.remaining), (2, 1));
        let rl = RateLimit::at_admit(lanes::Lane::Interactive, 5, &metrics, &capped, 2);
        assert_eq!(rl.remaining, 0);
        assert!(rl.reset_s > 0, "reset must arm at the tenant cap too");
        // the GLOBAL cap stays authoritative: a saturated lane zeroes the tenant's
        // remaining even below its own cap, and an override above the global cap is
        // ignored (min(t, global) — a key cannot widen the lane).
        let rl = RateLimit::at_admit(lanes::Lane::Interactive, global, &metrics, &capped, 0);
        assert_eq!(rl.remaining, 0);
        let wide = auth::TenantCtx {
            rate_limit: Some(global + 100),
            ..capped.clone()
        };
        let rl = RateLimit::at_admit(lanes::Lane::Interactive, 1, &metrics, &wide, 1);
        assert_eq!((rl.limit, rl.remaining), (global, global - 1));
    }

    #[test]
    fn batch_class_keys_default_to_harvest_and_cannot_claim_interactive() {
        let batch = auth::TenantCtx {
            tenant: "bulk".into(),
            lane_class: auth::LaneClass::Batch,
            rate_limit: None,
            key_prefix: None,
        };
        let interactive = auth::TenantCtx::default_tenant();
        let hdr = |v: Option<&str>| {
            let mut h = axum::http::HeaderMap::new();
            if let Some(v) = v {
                h.insert("x-lane", axum::http::HeaderValue::from_str(v).unwrap());
            }
            h
        };
        // interactive-class: legacy behavior exactly (default interactive, header honored).
        assert_eq!(
            lane_for_tenant(&hdr(None), &interactive).unwrap(),
            lanes::Lane::Interactive
        );
        assert_eq!(
            lane_for_tenant(&hdr(Some("judge")), &interactive).unwrap(),
            lanes::Lane::Judge
        );
        // batch-class: defaults to harvest; judge ok; interactive is a loud 403.
        assert_eq!(
            lane_for_tenant(&hdr(None), &batch).unwrap(),
            lanes::Lane::Harvest
        );
        assert_eq!(
            lane_for_tenant(&hdr(Some("judge")), &batch).unwrap(),
            lanes::Lane::Judge
        );
        let resp = lane_for_tenant(&hdr(Some("interactive")), &batch).unwrap_err();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        // unknown lane still 400s for everyone.
        let resp = lane_for_tenant(&hdr(Some("turbo")), &interactive).unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn handler_layer_refusals_are_openai_objects_with_x_should_retry() {
        // The lane refusals were the last bare-string error bodies on the surface:
        // `{"error": "unknown x-lane ..."}` indexes as a string in every SDK that reads
        // error.type / error.code. Both lane refusals now go through error_response_coded,
        // and both are unfixable-by-retry 4xx, so both must also say so in a header.
        let hdr = |v: &str| {
            let mut h = axum::http::HeaderMap::new();
            h.insert("x-lane", axum::http::HeaderValue::from_str(v).unwrap());
            h
        };
        let body = |resp: Response| async move {
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };

        let resp = lane_for_tenant(&hdr("turbo"), &auth::TenantCtx::default_tenant()).unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(resp.headers().get("x-should-retry").unwrap(), "false");
        let payload = body(resp).await;
        assert!(
            payload["error"].is_object(),
            "bare-string error body: {payload}"
        );
        assert_eq!(payload["error"]["type"], "invalid_request_error");
        assert_eq!(payload["error"]["param"], "x-lane");
        assert_eq!(payload["error"]["code"], "invalid_lane");

        let batch = auth::TenantCtx {
            tenant: "bulk".into(),
            lane_class: auth::LaneClass::Batch,
            rate_limit: None,
            key_prefix: None,
        };
        let resp = lane_for_tenant(&hdr("interactive"), &batch).unwrap_err();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert_eq!(resp.headers().get("x-should-retry").unwrap(), "false");
        let payload = body(resp).await;
        assert_eq!(payload["error"]["type"], "authentication_error");
        assert_eq!(payload["error"]["param"], "x-lane");
    }

    /// Serializes tests that read or flip the process-global DRAINING flag (the drain
    /// test must not 503 a concurrently-running handler test).
    static DRAIN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire DRAIN_LOCK surviving a poisoned peer, and restore the baseline it guards.
    ///
    /// 2026-09-01 (accrace close): one load-flaky deadline test panicked while holding
    /// this lock, and every later acquirer's `.unwrap()` then failed with PoisonError —
    /// one flake became 21 reds and buried its own cause under twenty unrelated ones.
    /// The lock guards the process-global DRAINING flag, not any invariant of the
    /// panicked test's own data, so recovering the guard is sound as long as the flag is
    /// put back to the "not draining" baseline every acquirer assumes; the drain tests
    /// that want it up set it themselves AFTER acquiring. Same poison-recovery idiom as
    /// `admission_counters_guard`. This normalization also retires the per-test
    /// `DRAINING.store(false, ..)` resets the 2026-08-09 flake introduced — the baseline
    /// now has one owner.
    fn drain_lock() -> std::sync::MutexGuard<'static, ()> {
        let guard = DRAIN_LOCK.lock().unwrap_or_else(|poisoned| {
            // Un-latch the flag too: poison otherwise persists forever, and only call
            // sites routed through this helper would survive it.
            DRAIN_LOCK.clear_poison();
            poisoned.into_inner()
        });
        DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
        guard
    }

    /// Put DRAINING back down on drop — including the drop that unwinds a failed
    /// assertion. The flag is read by every handler, INCLUDING in tests that have no
    /// reason to hold DRAIN_LOCK: a drain test that panicked between its `store(true)`
    /// and its reset would 503 every concurrently-running handler test until the next
    /// `drain_lock()` acquisition normalized the flag.
    struct DrainingRestore;
    impl Drop for DrainingRestore {
        fn drop(&mut self) {
            DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn responses_carry_rate_limit_headers_and_slot_frees() {
        let _l = drain_lock();
        let st = fake_worker_state();
        // non-stream chat: headers present, remaining = cap - 1 (this request held
        // the only slot), slot freed after completion.
        let resp = chat_completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}]
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let h = resp.headers();
        let limit: usize = h["x-ratelimit-limit"].to_str().unwrap().parse().unwrap();
        let remaining: usize = h["x-ratelimit-remaining"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(remaining, limit - 1);
        assert_eq!(h["x-ratelimit-reset"], "0");
        assert_eq!(
            st.inflight[0].load(std::sync::atomic::Ordering::SeqCst),
            0,
            "slot must free at completion"
        );
        // streaming completions: headers on the SSE response too; slot freed once the
        // body is drained (the guard rides the stream).
        let resp = completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "prompt": "t", "stream": true
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().contains_key("x-ratelimit-limit"));
        assert!(resp.headers().contains_key("x-ratelimit-remaining"));
        assert!(resp.headers().contains_key("x-ratelimit-reset"));
        assert_eq!(
            st.inflight[0].load(std::sync::atomic::Ordering::SeqCst),
            1,
            "stream in flight holds the slot"
        );
        let _ = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            st.inflight[0].load(std::sync::atomic::Ordering::SeqCst),
            0,
            "slot must free when the stream completes"
        );
    }

    /// REGRESSION FENCE for the 2026-09-02 rerank/embeddings ledger incident: a multi-item
    /// capture request opens ONE receipt PER ITEM, each under its own child id
    /// `<x-request-id>.<index>`, and settles every one of them. Under the old shared parent
    /// id this test's `opened` list read `[parent, parent, parent]`, which the darklanes
    /// ledger's replay guard turned into one debit (equal costs) or a 500 (unequal costs).
    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn multi_item_capture_requests_open_one_receipt_per_item_under_child_ids() {
        let _l = drain_lock();
        let mut st = fake_worker_state();
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());

        let resp = embed_api::embeddings_admitted(
            State(st.clone()),
            HeaderMap::new(),
            AdmittedJson(
                serde_json::from_value(json!({"model": "m", "input": ["a", "bb", "ccc"]})).unwrap(),
                BodyAdmissionLease(None),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let parent = resp.headers()["x-request-id"].to_str().unwrap().to_string();
        assert!(
            !parent.contains('.'),
            "the caller sees the parent id: {parent}"
        );
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["data"].as_array().map(Vec::len), Some(3));
        let events = mock.events();
        let opened: Vec<(String, &'static str)> = events
            .iter()
            .filter_map(|e| match e {
                MeterEvent::Open {
                    request_id, route, ..
                } => Some((request_id.clone(), *route)),
                _ => None,
            })
            .collect();
        assert_eq!(
            opened,
            vec![
                (format!("{parent}.0"), "/v1/embeddings"),
                (format!("{parent}.1"), "/v1/embeddings"),
                (format!("{parent}.2"), "/v1/embeddings"),
            ],
            "one receipt per input, each under its own child id: {events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, MeterEvent::Complete { .. }))
                .count(),
            3,
            "every input settles its own receipt: {events:?}"
        );

        let resp = embed_api::rerank_admitted(
            State(st),
            HeaderMap::new(),
            AdmittedJson(
                serde_json::from_value(
                    json!({"model": "m", "query": "q", "documents": ["d0", "d1"]}),
                )
                .unwrap(),
                BodyAdmissionLease(None),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let parent = resp.headers()["x-request-id"].to_str().unwrap().to_string();
        let opened: Vec<String> = mock
            .events()
            .into_iter()
            .skip(events.len())
            .filter_map(|e| match e {
                MeterEvent::Open {
                    request_id,
                    route: "/v1/rerank",
                    ..
                } => Some(request_id),
                _ => None,
            })
            .collect();
        assert_eq!(opened, vec![format!("{parent}.0"), format!("{parent}.1")]);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn handlers_sync_worker_truth_usage_and_cost_before_terminal_response() {
        let _l = drain_lock();
        let mut st = fake_worker_state();
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());

        let nonstream = chat_completions(
            State(st.clone()),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m",
                    "messages": [{"role": "user", "content": "t"}],
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(nonstream.status(), StatusCode::OK);
        let nonstream_id = nonstream.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_string();

        let stream = completions(
            State(st),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m",
                    "prompt": "t",
                    "stream": true,
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(stream.status(), StatusCode::OK);
        let stream_id = stream.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_string();
        let _ = axum::body::to_bytes(stream.into_body(), usize::MAX)
            .await
            .unwrap();

        // Both requests opened receipts under THEIR request ids (the x-request-id the
        // caller saw) and settled COMPLETE with worker-truth counts before the terminal
        // response was published.
        let events = mock.events();
        let opened: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                MeterEvent::Open { request_id, .. } => Some(request_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(opened, vec![nonstream_id.as_str(), stream_id.as_str()]);
        let completes = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    MeterEvent::Complete {
                        prompt: 1,
                        cached: 0,
                        completion: 1,
                    }
                )
            })
            .count();
        assert_eq!(
            completes, 2,
            "both surfaces settle complete with worker-truth usage: {events:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn completion_admission_supports_metered_blocked_and_paid_transitions() {
        let _l = drain_lock();
        // The handler's admission obligations, scripted at the seam: a denial maps to
        // the 402 contract and settles a REJECT receipt; an admission (with or without
        // a reservation permit) serves and settles COMPLETE, permit threaded through to
        // open(). Which MODES produce which answers is the implementation's business
        // and is tested with it (plus the cross-binary parity battery).
        let mock = MockMetering::with_limits(vec![
            ReserveScript::Insufficient,
            ReserveScript::Admit { with_permit: false },
            ReserveScript::Blocked,
            ReserveScript::Admit { with_permit: true },
        ]);
        let mut st = fake_worker_state();
        st.metering = Some(mock.clone());

        // Limits-source health reaches the operator metrics surface through the seam.
        let metrics = get_metrics(State(st.clone()), HeaderMap::new()).await;
        assert_eq!(metrics.status(), StatusCode::OK);
        let metrics_body = axum::body::to_bytes(metrics.into_body(), usize::MAX)
            .await
            .unwrap();
        let metrics_body: serde_json::Value = serde_json::from_slice(&metrics_body).unwrap();
        assert_eq!(metrics_body["budget_source_reload_failed"], 0);
        assert_eq!(metrics_body["budget_source_reload_consecutive"], 0);
        assert_eq!(metrics_body["budget_source_available"], true);

        let request = || {
            Json(
                serde_json::from_value::<CompletionReq>(json!({
                    "model": "m",
                    "prompt_ids": [1],
                    "max_tokens": 1,
                }))
                .unwrap(),
            )
        };

        let denied = completions(State(st.clone()), HeaderMap::new(), None, request()).await;
        assert_eq!(denied.status(), StatusCode::PAYMENT_REQUIRED);
        let denied_body = axum::body::to_bytes(denied.into_body(), usize::MAX)
            .await
            .unwrap();
        let denied_body: serde_json::Value = serde_json::from_slice(&denied_body).unwrap();
        assert_eq!(denied_body["error"]["type"], "insufficient_balance");
        assert_eq!(denied_body["error"]["code"], "insufficient_balance");

        let included = completions(State(st.clone()), HeaderMap::new(), None, request()).await;
        assert_eq!(included.status(), StatusCode::OK);

        // A Blocked denial deliberately reuses the prepaid 402 shape: callers get one
        // recovery action; the distinct admission mode is an operator-surface fact.
        let blocked = completions(State(st.clone()), HeaderMap::new(), None, request()).await;
        assert_eq!(blocked.status(), StatusCode::PAYMENT_REQUIRED);

        let admitted = completions(State(st.clone()), HeaderMap::new(), None, request()).await;
        assert_eq!(admitted.status(), StatusCode::OK);

        let events = mock.events();
        let terminal: Vec<&MeterEvent> = events
            .iter()
            .filter(|e| matches!(e, MeterEvent::Reject { .. } | MeterEvent::Complete { .. }))
            .collect();
        assert_eq!(
            terminal.len(),
            4,
            "four requests, four terminal settles: {events:?}"
        );
        assert!(matches!(
            terminal[0],
            MeterEvent::Reject { status: 402, .. }
        ));
        assert!(matches!(terminal[1], MeterEvent::Complete { .. }));
        assert!(matches!(
            terminal[2],
            MeterEvent::Reject { status: 402, .. }
        ));
        assert!(matches!(terminal[3], MeterEvent::Complete { .. }));
        // The reservation permit made it through to open() on the paid admission.
        let permits: Vec<bool> = events
            .iter()
            .filter_map(|e| match e {
                MeterEvent::Open { with_permit, .. } => Some(*with_permit),
                _ => None,
            })
            .collect();
        assert_eq!(
            permits,
            vec![false, false, false, true],
            "the permit rides the receipt exactly when reserve minted one: {events:?}"
        );
    }

    /// A capped KEY answers its own 402 code (the recovery is raising the cap, not
    /// adding credit) and the authenticated key's prefix crossed the seam to reserve
    /// — the per-key-policy hook (stage 4, engine-billing-extraction-20260829).
    #[tokio::test]
    async fn a_capped_key_answers_its_own_402_and_the_principal_crosses_the_seam() {
        let mock = MockMetering::with_limits(vec![ReserveScript::PrincipalCapped]);
        let mut st = fake_worker_state();
        st.metering = Some(mock.clone());
        let tenant = auth::TenantCtx {
            tenant: "acme".into(),
            lane_class: auth::LaneClass::Interactive,
            rate_limit: None,
            key_prefix: Some("mk-acme-testprefix00".into()),
        };
        let mut request = gate_request(1, 1);
        let rejection = admit_tenant_budget(&st, &tenant, &mut request)
            .expect_err("a capped key must be refused at admission");
        assert!(matches!(rejection, BudgetRejection::PrincipalCapped));
        let (response, outcome) = rejection.into_response();
        assert_eq!(outcome, "key_spend_cap_reached");
        assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);
        let body = body_value(response).await;
        assert_eq!(body["error"]["code"], "key_spend_cap_reached");
        assert!(
            body["error"]["message"].as_str().unwrap().contains("cap"),
            "the 402 must point at the KEY's cap, not tenant credit: {body}"
        );
        let events = mock.events();
        assert!(
            events.contains(&MeterEvent::Reserve {
                tenant: "acme".into(),
                principal: Some("mk-acme-testprefix00".into()),
                model: "qwen/qwen3.8-27b".into(),
            }),
            "the key prefix must reach reserve: {events:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn streaming_client_disconnect_records_partial_usage_and_cost() {
        let _l = drain_lock();
        let mut st = fake_worker_state_with_steps(4, std::time::Duration::from_millis(100));
        let mock = MockMetering::admit_all();
        st.metering = Some(mock.clone());

        let response = completions(
            State(st),
            HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(json!({
                    "model": "m",
                    "prompt": "disconnect after one delta",
                    "stream": true,
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let request_id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_string();
        let mut body = Box::pin(response.into_body().into_data_stream());
        let first = std::future::poll_fn(|cx| body.as_mut().poll_next(cx))
            .await
            .expect("stream ended before first delta")
            .expect("stream body failed");
        assert!(
            is_sse_data_frame(&first),
            "first frame was not SSE data: {first:?}"
        );
        drop(body);

        // The receipt died UNFINALIZED with the partial counts recorded — the
        // abandoned-client seam contract. Give the dropped stream a beat to unwind.
        let mut dropped = None;
        for _ in 0..500 {
            if let Some(event) = mock
                .events()
                .into_iter()
                .find(|e| matches!(e, MeterEvent::Dropped { .. }))
            {
                dropped = Some(event);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        let events = mock.events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, MeterEvent::Open { request_id: id, .. } if id == &request_id)),
            "the receipt was opened under the caller-visible request id: {events:?}"
        );
        assert_eq!(
            dropped,
            Some(MeterEvent::Dropped {
                prompt: 1,
                cached: 0,
                completion: 1,
            }),
            "a client disconnect must leave the partial counts on the dropped receipt \
             (the implementation prices that drop): {events:?}"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn draining_rejects_new_requests_with_503_and_retry_after() {
        let _l = drain_lock();
        let st = fake_worker_state();
        // RAII, not just the trailing reset below: a panic while the flag is up would
        // 503 every concurrently-running handler test (they read DRAINING lock-free).
        let _down = DrainingRestore;
        DRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
        // both completion routes: immediate 503 + Retry-After, no slot held.
        let resp = chat_completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}]
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        // The drain 503 obeys the same retry contract as every taxonomy class: an integer
        // Retry-After <= 60, the retry-after-ms twin openai-python reads FIRST (its absence
        // was a real gap — a client trusting only the ms header saw NO window on memra's most
        // predictable outage), both agreeing, and a `code` clients can branch on.
        let ra = resp.headers()["retry-after"].to_str().unwrap().to_string();
        let ra_s: u64 = ra
            .parse()
            .expect("Retry-After must be integer delay-seconds");
        assert!(
            ra_s > 0 && ra_s <= 60,
            "Retry-After {ra_s}s is outside the honored window"
        );
        let ra_ms: u64 = resp.headers()["retry-after-ms"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(ra_ms, ra_s * 1000, "the two retry headers must agree");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap()
                .contains("draining")
        );
        assert_eq!(payload["error"]["type"], "server_error");
        assert_eq!(payload["error"]["code"], "draining");
        let resp = completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "prompt": "t"
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(resp.headers().contains_key("retry-after"));
        assert_eq!(
            st.inflight[0].load(std::sync::atomic::Ordering::SeqCst),
            0,
            "rejected requests must not hold slots"
        );
        // /health flips to "draining" but stays 200 — a drain is a HEALTHY shutdown, and 503
        // here would invite a supervisor to SIGKILL a process that is finishing streams.
        let resp = health_live(State(st.clone())).await.into_response();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "a drain must not look like a liveness fault"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "draining");
        // Rotation is /readyz's job: unready while draining, so the LB stops sending.
        let resp = health_ready(State(st.clone())).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry_s = drain_deadline_s().clamp(1, 60);
        let retry_s_text = retry_s.to_string();
        let retry_ms_text = (retry_s * 1000).to_string();
        assert_eq!(retry_after(&resp).as_deref(), Some(retry_s_text.as_str()));
        assert_eq!(
            resp.headers().get("retry-after-ms").unwrap(),
            retry_ms_text.as_str()
        );
        assert_ne!(
            resp.headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "not_ready");
        assert!(payload["detail"].as_str().unwrap().contains("draining"));
        DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
        // flag cleared: requests admit again (the gate is the flag, nothing latent).
        let resp = chat_completions(
            State(st.clone()),
            axum::http::HeaderMap::new(),
            None,
            Json(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "messages": [{"role": "user", "content": "t"}]
                }))
                .unwrap(),
            ),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ---- G5: /health reports INFERENCE liveness, not process liveness -------------------

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn health_is_green_only_while_the_worker_is_alive() {
        // /readyz reads the process-global DRAINING flag, which the drain test toggles —
        // serialize against it or this races (measured: an interleaved run saw 503 here).
        let _l = drain_lock();
        let st = fake_worker_state();
        // loaded + alive: 200 ok, and the payload explains WHY (phase + heartbeat age vs the
        // threshold), so an operator reading a green never has to guess.
        let resp = health_live(State(st.clone())).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "ok");
        assert_eq!(payload["worker"]["phase"], "idle");
        assert!(payload["worker"]["stall_threshold_ms"].as_u64().unwrap() > 0);
        let ready = health_ready(State(st.clone())).await.into_response();
        assert_eq!(ready.status(), StatusCode::OK);

        // THE REGRESSION THIS PINS. Kill inference the way a panic does — the health handle
        // is marked dead, the HTTP task keeps running, the process is entirely fine. The old
        // handler returned `{"status":"ok"}` here, forever, on a box answering nothing.
        st.health.mark_dead("worker thread panicked: test-injected");
        let resp = health_live(State(st.clone())).await.into_response();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a dead worker MUST NOT report a healthy liveness"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "unhealthy");
        // the cause is QUOTED, not inferred — the panic text travels to the operator
        assert!(
            payload["detail"]
                .as_str()
                .unwrap()
                .contains("test-injected"),
            "cause not surfaced: {payload}"
        );
        let ready = health_ready(State(st.clone())).await.into_response();
        assert_eq!(
            ready.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "dead is also not ready"
        );

        // Latency of the flip: a fault latch, not a timeout — no staleness threshold to wait
        // out, which is what makes this usable as a k8s livenessProbe.
        st.health.mark_ready();
        assert_eq!(
            health_live(State(st.clone()))
                .await
                .into_response()
                .status(),
            StatusCode::OK,
            "mark_ready must clear the latch (a successful respawn)"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn readyz_peer_probe_integrity_is_present_and_advisory() {
        let _l = drain_lock();
        let st = fake_worker_state();

        let ready = health_ready(State(st.clone())).await.into_response();
        assert_eq!(ready.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(ready.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["peer_probe_integrity"], "ok");

        st.health.note_peer_probe_deferral(2, false);
        let deferred = health_ready(State(st.clone())).await.into_response();
        assert_eq!(deferred.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(deferred.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["peer_probe_integrity"], "deferred_2");

        st.health.note_peer_probe_deferral(4, true);
        let degraded = health_ready(State(st.clone())).await.into_response();
        assert_eq!(
            degraded.status(),
            StatusCode::OK,
            "peer degradation is advisory while plain serving remains healthy"
        );
        let bytes = axum::body::to_bytes(degraded.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["peer_probe_integrity"], "degraded");

        st.health.mark_dead("test-injected worker failure");
        let unready = health_ready(State(st)).await.into_response();
        assert_eq!(unready.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(unready.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            payload["peer_probe_integrity"], "degraded",
            "the advisory field must also survive an unrelated readiness failure"
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn liveness_failure_obeys_the_retry_contract() {
        // drain_lock() serializes AND resets the flag: health_live returns 200 ("draining")
        // whenever the process-global DRAINING flag is up, so any test asserting a
        // health_live 503 races the drain tests without it (the a_wedged flake, 2026-08-09
        // — schedule-dependent).
        let _l = drain_lock();
        let st = fake_worker_state();
        st.health
            .mark_dead("worker thread panicked: retry-contract-test");

        let resp = health_live(State(st)).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry_after(&resp).as_deref(), Some("2"));
        assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "2000");
        assert_ne!(
            resp.headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn readiness_failure_obeys_the_retry_contract() {
        let _l = drain_lock();
        let st = fake_worker_state();
        st.health
            .mark_dead("worker thread panicked: retry-contract-test");

        let resp = health_ready(State(st)).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry_after(&resp).as_deref(), Some("2"));
        assert_eq!(resp.headers().get("retry-after-ms").unwrap(), "2000");
        assert_ne!(
            resp.headers()
                .get("x-should-retry")
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // allow: DRAIN_LOCK serializes this test against its shared-state peers; holding across the awaits is the point
    async fn a_wedged_gpu_flips_health_even_though_the_worker_thread_is_fine() {
        // G24: Xid 119/120 hangs nvidia-smi and emits no Xid line; the watcher's probe
        // timeout is the alarm. The worker thread may still be looping (blocked in a driver
        // call), so the heartbeat alone would never catch this — the GPU latch does.
        //
        // drain_lock() serializes + resets (2026-08-09 flake): health_live short-circuits to
        // 200 ("draining") on the process-global DRAINING flag, so this test's 503 assertions
        // race the drain tests when tokio schedules them concurrently — it failed only in
        // full-suite runs, never solo, and the same suite on the identical commit passes or
        // fails by schedule. Same serialization the other drain-flag readers already take.
        let _l = drain_lock();
        let st = fake_worker_state();
        assert_eq!(
            health_live(State(st.clone()))
                .await
                .into_response()
                .status(),
            StatusCode::OK
        );
        st.health
            .mark_gpu_fault("nvidia-smi probe exceeded 10s deadline (GSP hang class)");
        let resp = health_live(State(st.clone())).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            payload["detail"]
                .as_str()
                .unwrap()
                .contains("probe exceeded")
        );
        // A GPU fault survives mark_ready deliberately: a respawned worker on a wedged card
        // is not recovery, and only a fresh process (new CUDA context) can be.
        st.health.mark_ready();
        assert_eq!(
            health_live(State(st.clone()))
                .await
                .into_response()
                .status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "a GPU fault must not be cleared by an in-process respawn"
        );
    }

    #[test]
    fn v1_models_entry_keeps_catalog_shape_with_honest_nulls() {
        // KNOWN plan metadata populates every OR-schema field from worker truth.
        let caps = ModelCaps {
            tools_branch: true,
            hy3: false,
            qwen_think: true,
            think_switch: true,
            chat_ok: true,
            context_length: 262144,
            tokenizer: "qwen2".into(),
            instruct_type: Some("chatml".into()),
            effort_levels: false,
            qwen_effort: false,
            gemma_think: false,
            dsv4: false,
            glm5: false,
            chat_temperature_default: None,
            chat_top_p_default: None,
            n_vocab: 151_936,
            think_close: Vec::new(),
        };
        let e = model_entry_v1("main", Some(&caps), None);
        assert_eq!(e["id"], "main");
        assert_eq!(e["name"], "main");
        assert_eq!(e["object"], "model");
        assert_eq!(e["context_length"], 262144);
        // no metadata -> null prices (unpriced), no cache keys invented.
        assert!(e["pricing"]["input"].is_null());
        assert!(e["pricing"]["output"].is_null());

        // METADATA present -> /v1/models advertises the SAME prices the ledger bills
        // (the launch bug: a priced, vision-serving endpoint reported "0" text-only).
        let meta = OpenRouterModelMetadata {
            pricing: OpenRouterPricing {
                prompt: Some("0.00000038".into()),
                cached_prompt: Some("0.0000002".into()),
                completion: Some("0.0000026".into()),
                ..Default::default()
            },
            input_modalities: vec!["image".into(), "video".into()],
            max_output_length: Some(32768),
            ..Default::default()
        };
        let e = model_entry_v1("main", Some(&caps), Some(&meta));
        // Contract-v2 pricing: per-1M string prices (decimal shift of the SAME metadata),
        // null cache_write (not configured), lifecycle default active, reliability defaults.
        assert_eq!(e["pricing"]["currency"], "USD");
        assert_eq!(e["pricing"]["unit"], "per_1m_tokens");
        assert_eq!(e["pricing"]["input"], "0.38");
        assert_eq!(e["pricing"]["output"], "2.60");
        assert_eq!(e["pricing"]["cached_input"], "0.20");
        assert!(e["pricing"]["cache_write"].is_null());
        assert_eq!(e["pricing"]["minimum_request"], "0");
        assert_eq!(e["owned_by"], "main");
        assert_eq!(e["type"], "chat");
        assert_eq!(e["max_output_tokens"], 32768);
        assert_eq!(e["endpoints"], json!(["chat/completions"]));
        assert_eq!(e["input_modalities"], json!(["text", "image", "video"]));
        assert_eq!(e["output_modalities"], json!(["text"]));
        assert_eq!(e["capabilities"]["streaming"], true);
        assert_eq!(e["capabilities"]["tools"], true);
        assert_eq!(e["lifecycle"]["status"], "active");
        assert!(e["lifecycle"]["deprecation_at"].is_null());
        assert_eq!(e["reliability"]["first_token_timeout_seconds"], 120);
        assert_eq!(e["reliability"]["capacity_scope"], "model_region");
        // EXACT key set — the contract forbids extra fields ("Do not design a custom
        // catalog"): no created, architecture, supported_parameters, top_provider, and
        // no legacy per-token pricing keys.
        let mut keys: Vec<&str> = e.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "capabilities",
                "context_length",
                "endpoints",
                "id",
                "input_modalities",
                "lifecycle",
                "max_output_tokens",
                "name",
                "object",
                "output_modalities",
                "owned_by",
                "pricing",
                "reliability",
                "type",
            ],
            "unexpected /v1/models entry keys"
        );
        let mut price_keys: Vec<&str> = e["pricing"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        price_keys.sort_unstable();
        assert_eq!(
            price_keys,
            [
                "cache_write",
                "cached_input",
                "currency",
                "input",
                "minimum_request",
                "output",
                "unit",
            ],
            "unexpected /v1/models pricing keys"
        );

        // UNKNOWN metadata (no caps / empty fields) -> honest nulls, never invented.
        let e = model_entry_v1("m", None, None);
        assert!(e["context_length"].is_null());
        assert!(e["max_output_tokens"].is_null());
        let bare = ModelCaps::default(); // caps present, fields unknown (0/""/None)
        let e = model_entry_v1("m", Some(&bare), None);
        assert!(e["context_length"].is_null());
    }

    /// 2026-08-28: qwen3-embedding-8b and qwen3-reranker-8b were published on
    /// /v1/models as `type: "chat"`, `endpoints: ["chat/completions"]`, with
    /// `tools: true` and `streaming: true`. Neither serves chat at all. A client SDK
    /// reading that row calls the wrong endpoint with the wrong body shape, so the
    /// declared surface — not a hardcoded literal — decides the row.
    #[test]
    fn catalog_row_follows_the_declared_surface() {
        let caps = ModelCaps {
            tools_branch: true,
            ..Default::default()
        };

        let embed = OpenRouterModelMetadata {
            surface: Some("embedding".into()),
            max_output_length: Some(1),
            ..Default::default()
        };
        let e = model_entry_v1("qwen", Some(&caps), Some(&embed));
        assert_eq!(e["type"], "embedding");
        assert_eq!(e["endpoints"], json!(["embeddings"]));
        assert_eq!(e["output_modalities"], json!(["embeddings"]));
        assert_eq!(e["capabilities"]["streaming"], false);
        assert_eq!(
            e["capabilities"]["tools"], false,
            "an embedder has no tools"
        );
        assert_eq!(e["capabilities"]["reasoning"], false);
        assert_eq!(e["capabilities"]["structured_output"], false);
        assert_eq!(e["capabilities"]["prompt_caching"], false);
        assert!(
            e["max_output_tokens"].is_null(),
            "a surface that emits no completion tokens must not advertise a ceiling"
        );

        let rerank = OpenRouterModelMetadata {
            surface: Some("rerank".into()),
            ..Default::default()
        };
        let r = model_entry_v1("qwen", Some(&caps), Some(&rerank));
        assert_eq!(r["type"], "rerank");
        assert_eq!(r["endpoints"], json!(["rerank"]));
        assert_eq!(r["output_modalities"], json!(["rerank"]));
        assert_eq!(r["capabilities"]["tools"], false);
        assert_eq!(r["capabilities"]["reasoning"], false);

        // Absent surface stays chat, byte-for-byte with the pre-change row: every
        // existing deployment's models.toml omits the field.
        let chat = OpenRouterModelMetadata {
            max_output_length: Some(32768),
            ..Default::default()
        };
        let c = model_entry_v1("main", Some(&caps), Some(&chat));
        assert_eq!(c["type"], "chat");
        assert_eq!(c["endpoints"], json!(["chat/completions"]));
        assert_eq!(c["output_modalities"], json!(["text"]));
        assert_eq!(c["capabilities"]["tools"], true);
        assert_eq!(c["max_output_tokens"], 32768);
    }

    /// The surface is a published contract, so a typo must fail the config load
    /// rather than silently publishing a chat row for an embedder.
    #[test]
    fn unknown_surface_is_rejected_at_config_load() {
        let bad = OpenRouterModelMetadata {
            surface: Some("embeddings".into()), // plural: the near-miss typo
            ..Default::default()
        };
        let err = validate_openrouter_metadata("qwen/qwen3-embedding-8b", &bad)
            .expect_err("an unknown surface must not load");
        assert!(err.contains("surface"), "{err}");

        for good in ["chat", "embedding", "rerank"] {
            let ok = OpenRouterModelMetadata {
                surface: Some(good.into()),
                ..Default::default()
            };
            assert!(
                validate_openrouter_metadata("m", &ok).is_ok(),
                "{good} must load"
            );
        }
    }

    #[test]
    fn per_million_price_is_exact_decimal_shift() {
        // The live prices: per-token strings -> per-1M contract strings, no floats anywhere.
        assert_eq!(per_million_price("0.00000038").as_deref(), Some("0.38"));
        assert_eq!(per_million_price("0.0000026").as_deref(), Some("2.60"));
        assert_eq!(per_million_price("0.0000002").as_deref(), Some("0.20"));
        assert_eq!(per_million_price("0").as_deref(), Some("0.00"));
        assert_eq!(per_million_price("1.5").as_deref(), Some("1500000.00"));
        assert_eq!(per_million_price("0.000000125").as_deref(), Some("0.125"));
        assert_eq!(per_million_price("not-a-price"), None);
        assert_eq!(per_million_price(""), None);
    }

    #[test]
    fn metadata_provider_block_parses_and_validates() {
        let (_, provider) = OpenRouterMetadataFile::parse(
            r#"
            [provider]
            id = "tiyuvta"
            status_url = "https://status.tiyuvta.ai"
            support_contact = "mailto:support@tiyuvta.ai"
            incident_contact = "mailto:incidents@tiyuvta.ai"
            regions = ["eu-central"]
            "#,
        )
        .unwrap();
        let provider = provider.unwrap();
        assert_eq!(provider.id, "tiyuvta");
        assert_eq!(provider.regions, vec!["eu-central"]);
        // empty id refuses at boot, not at request time
        let err = OpenRouterMetadataFile::parse("[provider]\nid = \"\"\n").unwrap_err();
        assert!(err.contains("provider.id"), "{err}");
        // a bare email is not a URI — the contract wants mailto:/https: schemes
        let err = OpenRouterMetadataFile::parse(
            "[provider]\nid = \"x\"\nsupport_contact = \"ops@example.com\"\n",
        )
        .unwrap_err();
        assert!(err.contains("must be a URI"), "{err}");
        // absent block is not an error
        let (_, provider) = OpenRouterMetadataFile::parse("").unwrap();
        assert!(provider.is_none());
    }

    #[test]
    fn models_openai_default_body_stays_byte_identical() {
        let body = models_openai_body(&["main".into(), "judge".into()]);
        let bytes = serde_json::to_vec(&body).unwrap();
        assert_eq!(
            bytes,
            br#"{"object":"list","data":[{"id":"main","object":"model"},{"id":"judge","object":"model"}]}"#
        );
    }

    #[test]
    fn canonical_model_id_tolerates_a_marketplace_stripping_the_vendor_prefix() {
        // The exact live failure: Onlist listed qwen/qwen3.6-35b-a3b and probed for the bare name.
        let loaded = vec![
            "qwen/qwen3.6-27b".to_string(),
            "qwen/qwen3.6-35b-a3b".to_string(),
        ];
        assert_eq!(
            canonical_model_id(&loaded, "qwen3.6-35b-a3b").as_deref(),
            Some("qwen/qwen3.6-35b-a3b"),
        );
        assert_eq!(
            canonical_model_id(&loaded, "qwen3.6-27b").as_deref(),
            Some("qwen/qwen3.6-27b"),
        );
        // An exact alias must keep resolving to itself, unchanged.
        assert_eq!(
            canonical_model_id(&loaded, "qwen/qwen3.6-35b-a3b").as_deref(),
            Some("qwen/qwen3.6-35b-a3b"),
        );
        // A genuinely unknown id stays unknown, so the worker still emits model_not_found.
        assert_eq!(canonical_model_id(&loaded, "gpt-4o"), None);
        assert_eq!(canonical_model_id(&loaded, "vendor/qwen3.6-35b-a3b"), None);
        assert_eq!(canonical_model_id(&loaded, ""), None);
    }

    #[test]
    fn canonical_model_id_refuses_an_ambiguous_suffix_rather_than_guessing() {
        // Two vendors publishing the same model name must NOT be silently disambiguated: routing to
        // the wrong weights would also bill under the wrong model's price schedule.
        let loaded = vec!["a/shared-name".to_string(), "b/shared-name".to_string()];
        assert_eq!(canonical_model_id(&loaded, "shared-name"), None);
        // Each exact id still resolves.
        assert_eq!(
            canonical_model_id(&loaded, "a/shared-name").as_deref(),
            Some("a/shared-name")
        );
        assert_eq!(
            canonical_model_id(&loaded, "b/shared-name").as_deref(),
            Some("b/shared-name")
        );
        // An unprefixed alias is matched exactly, not by suffix games.
        let bare = vec!["solo".to_string()];
        assert_eq!(canonical_model_id(&bare, "solo").as_deref(), Some("solo"));
    }

    #[test]
    fn openrouter_models_entry_serializes_complete_metadata() {
        let metadata = OpenRouterMetadataFile::from_toml(
            r#"
[models.main]
hugging_face_id = "Qwen/Qwen3.6-27B"
created = 1786032000
quantization = "nvfp4"
description = "Qwen3.6 27B served by memra."
max_prompt_length = 245760
max_output_length = 16384
default_output_length = 4096
is_ready = true
is_free = false
discount_to_user = 0.1
openrouter_slug = "qwen/qwen3.6-27b"
datacenters = [{ country_code = "US", region = "us-east" }]
zdr = true
hipaa = false

[models.main.pricing]
prompt = "0.000000234"
cached_prompt = "0.0000000585"
cache_write = "0.000000234"
completion = "0.000001872"
internal_reasoning = "0.000001872"
request = "0.01"

[models.main.capacity]
prompt_tpm = 1000000
cached_prompt_tpm = 2000000
completion_tpm = 500000
request_rpm = 1000
concurrency = 64
"#,
        )
        .unwrap();
        let caps = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            think_switch: true,
            chat_ok: true,
            context_length: 262144,
            tokenizer: "qwen2".into(),
            instruct_type: Some("chatml".into()),
            ..Default::default()
        };
        let entry = model_entry_openrouter("main", Some(&caps), metadata.get("main"));

        assert_eq!(entry["schema_version"], "2.4");
        assert_eq!(entry["id"], "main");
        assert_eq!(entry["name"], "main");
        assert_eq!(entry["hugging_face_id"], "Qwen/Qwen3.6-27B");
        assert_eq!(entry["created"], 1786032000u64);
        assert_eq!(entry["quantization"], "nvfp4");
        assert_eq!(entry["tokenizer"], "qwen2");
        assert_eq!(entry["description"], "Qwen3.6 27B served by memra.");
        assert!(
            entry.get("object").is_none(),
            "OpenRouter schema 2.4 rejects unknown OpenAI fields"
        );

        let input = &entry["input_modalities"][0];
        assert_eq!(input["type"], "text");
        assert_eq!(
            input["supported_inputs"]["max_context_length"]["value"],
            262144
        );
        assert_eq!(
            input["supported_inputs"]["max_prompt_length"]["value"],
            245760
        );
        let input_prices = input["pricing"].as_array().unwrap();
        let input_price = |kind: &str| {
            input_prices
                .iter()
                .find(|price| price["type"] == kind)
                .unwrap()
        };
        assert_eq!(input_price("prompt")["cost_usd"], "0.000000234");
        assert_eq!(input_price("cached_prompt")["cost_usd"], "0.0000000585");
        assert_eq!(input_price("cache_write")["cost_usd"], "0.000000234");
        assert_eq!(input["capacity"][0]["value"], 1000000);
        assert_eq!(input["capacity"][1]["value"], 2000000);

        let output = &entry["output_modalities"][0];
        assert_eq!(output["type"], "text");
        assert_eq!(output["max_length"]["value"], 16384);
        assert_eq!(output["streaming"], true);
        assert_eq!(output["supported_parameters"]["tools"]["type"], "boolean");
        assert_eq!(
            output["supported_parameters"]["structured_outputs"]["type"],
            "boolean"
        );
        assert_eq!(
            output["supported_parameters"]["reasoning"]["type"],
            "boolean"
        );
        assert_eq!(output["pricing"][0]["type"], "completion");
        assert_eq!(output["pricing"][0]["cost_usd"], "0.000001872");
        assert_eq!(output["pricing"][1]["type"], "internal_reasoning");
        assert_eq!(output["capacity"][0]["value"], 500000);
        assert_eq!(output["capacity"][1]["type"], "concurrency");
        assert_eq!(output["capacity"][1]["value"], 64);

        assert_eq!(entry["pricing"][0]["type"], "request");
        assert_eq!(entry["pricing"][0]["cost_usd"], "0.01");
        assert_eq!(entry["capacity"][0]["value"], 1000);
        assert_eq!(entry["is_ready"], true);
        assert_eq!(entry["is_free"], false);
        assert_eq!(entry["discount_to_user"], 0.1);
        assert_eq!(entry["openrouter"]["slug"], "qwen/qwen3.6-27b");
        assert_eq!(entry["datacenters"][0]["country_code"], "US");
        assert_eq!(entry["compliance"]["zdr"], true);
        assert_eq!(entry["compliance"]["hipaa"], false);
    }

    /// The deploy registry moved to the private operations repo (owner boundary call,
    /// 2026-08-16); the SHAPE these tests pin is engine contract, so they keep a local
    /// fixture with the same staged/active structure and the same values the assertions
    /// below already publish.
    const GATEWAY_REGISTRY_FIXTURE: &str = r#"
[models."qwen/qwen3.6-35b-a3b"]
hugging_face_id = "Qwen/Qwen3.6-35B-A3B"
created = 1777260255
quantization = "int4"
description = "Qwen3.6 35B-A3B fixture entry."
max_prompt_length = 262144
max_output_length = 262144
default_output_length = 8192
is_ready = true
is_free = false
discount_to_user = 0.0
openrouter_slug = "qwen/qwen3.6-35b-a3b"
zdr = false
hipaa = false

[[models."qwen/qwen3.6-35b-a3b".datacenters]]
country_code = "CA"
region = "Ontario"

[models."qwen/qwen3.6-35b-a3b".pricing]
prompt = "0.0000000931"
cached_prompt = "0.0000000652"
completion = "0.0000009025"

[models."qwen/qwen3.6-35b-a3b".capacity]
prompt_tpm = 780000
cached_prompt_tpm = 310000
completion_tpm = 9600
request_rpm = 160
concurrency = 16

[planned_models."qwen/qwen3.8-27b"]
description = "Planned fixture entry; must never be emitted."
max_prompt_length = 262144
max_output_length = 262144
default_output_length = 8192
is_ready = false
is_free = false
discount_to_user = 0.0
openrouter_slug = "qwen/qwen3.8-27b"
zdr = false
hipaa = false

[planned_models."qwen/qwen3.8-27b".pricing]
prompt = "0.0000002745"
cached_prompt = "0.0000001922"
completion = "0.0000022800"

[planned_models."google/gemma-4-26b-a4b-it"]
hugging_face_id = "google/gemma-4-26B-A4B-it"
created = 1775227989
quantization = "int4"
description = "Planned fixture entry; must never be emitted."
max_prompt_length = 262144
max_output_length = 262144
default_output_length = 8192
is_ready = false
is_free = false
discount_to_user = 0.0
openrouter_slug = "google/gemma-4-26b-a4b-it"
zdr = false
hipaa = false

[planned_models."google/gemma-4-26b-a4b-it".pricing]
prompt = "0.0000000665"
cached_prompt = "0.0000000466"
completion = "0.0000003230"
"#;

    #[test]
    fn gateway_registry_generates_the_staged_active_shape() {
        let metadata = OpenRouterMetadataFile::from_toml(GATEWAY_REGISTRY_FIXTURE).unwrap();
        let caps = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            think_switch: true,
            chat_ok: true,
            context_length: 262144,
            tokenizer: "qwen2".into(),
            instruct_type: Some("chatml".into()),
            ..Default::default()
        };
        let q35_entry = model_entry_openrouter(
            "qwen/qwen3.6-35b-a3b",
            Some(&caps),
            metadata.get("qwen/qwen3.6-35b-a3b"),
        );
        assert_eq!(q35_entry["created"], 1777260255u64);
        assert_eq!(q35_entry["quantization"], "int4");
        assert_eq!(q35_entry["is_ready"], true);
        assert_eq!(
            q35_entry["input_modalities"][0]["supported_inputs"]["max_context_length"]["value"],
            262144
        );
        assert_eq!(
            q35_entry["input_modalities"][0]["supported_inputs"]["max_prompt_length"]["value"],
            262144
        );
        assert_eq!(
            q35_entry["output_modalities"][0]["max_length"]["value"],
            262144
        );
        let prices = q35_entry["input_modalities"][0]["pricing"]
            .as_array()
            .unwrap();
        assert_eq!(prices[0]["cost_usd"], "0.0000000931");
        assert_eq!(prices[1]["cost_usd"], "0.0000000652");
        // Capacity is the MEASURED sold-shape floor (2026-08-13, research/canonflip-20260813):
        // 4,860-token prompt + 60 output, single RTX PRO 6000 WS. These five move together and
        // only with a measurement — see the comment block in deploy/gateway/q27-models.toml.
        assert_eq!(
            q35_entry["input_modalities"][0]["capacity"][0]["value"],
            780000
        );
        assert_eq!(
            q35_entry["input_modalities"][0]["capacity"][1]["value"],
            310000
        );
        assert_eq!(
            q35_entry["output_modalities"][0]["supported_parameters"]["max_tokens"]["max"],
            262144
        );
        assert_eq!(
            q35_entry["output_modalities"][0]["capacity"][0]["value"],
            9600
        );
        assert_eq!(
            q35_entry["output_modalities"][0]["capacity"][1]["value"],
            16
        );
        assert_eq!(
            q35_entry["output_modalities"][0]["pricing"][0]["cost_usd"],
            "0.0000009025"
        );
        assert_eq!(q35_entry["capacity"][0]["value"], 160); // request_rpm, sold-shape floor
        assert_eq!(q35_entry["datacenters"][0]["country_code"], "CA");

        assert_eq!(
            metadata.len(),
            1,
            "planned models must never enter the active map"
        );
        assert!(!metadata.contains_key("qwen/qwen3.6-27b"));
        assert!(!metadata.contains_key("qwen/qwen3.8-27b"));
        assert!(!metadata.contains_key("google/gemma-4-26b-a4b-it"));

        let openmodels = model_entry_openmodels(
            "qwen/qwen3.6-35b-a3b",
            Some(&caps),
            metadata.get("qwen/qwen3.6-35b-a3b"),
        )
        .unwrap();
        assert_eq!(openmodels["currency"], "USD");
        assert_eq!(openmodels["max_output_length"], 262144);
        assert_eq!(openmodels["is_ready"], true);
        assert_eq!(openmodels["is_free"], false);
        assert_eq!(openmodels["discount_to_user"], 0.0);
    }

    #[test]
    fn gateway_registry_limits_are_live_request_limits() {
        let metadata_file = OpenRouterMetadataFile::from_toml(GATEWAY_REGISTRY_FIXTURE).unwrap();
        let metadata = metadata_file.get("qwen/qwen3.6-35b-a3b").unwrap();
        let caps = ModelCaps {
            context_length: 262_144,
            ..Default::default()
        };
        let build = |value: serde_json::Value| {
            let req: CompletionReq = serde_json::from_value(value).unwrap();
            let (tx, _rx) = worker::event_channel();
            build_request(&req, tx, lanes::Lane::Interactive, None)
        };

        let mut omitted = build(json!({
            "model": "qwen/qwen3.6-35b-a3b",
            "prompt_ids": [1, 2, 3]
        }));
        apply_model_request_limits(&mut omitted, Some(metadata), Some(&caps)).unwrap();
        assert_eq!(omitted.params.max_new, 8_192);
        assert_eq!(omitted.max_prompt_tokens, Some(262_144));

        let mut field_top = build(json!({
            "model": "qwen/qwen3.6-35b-a3b",
            "prompt_ids": [1],
            "max_tokens": 262144
        }));
        apply_model_request_limits(&mut field_top, Some(metadata), Some(&caps)).unwrap();
        assert_eq!(field_top.params.max_new, 262_144);
        assert_eq!(
            budget_completion_bound(&field_top, 100, Some(&caps)).unwrap(),
            262_044,
            "the field-top output request is accepted but bounded by remaining trained context",
        );

        let mut too_much_output = build(json!({
            "model": "qwen/qwen3.6-35b-a3b",
            "prompt_ids": [1],
            "max_tokens": 262145
        }));
        let (message, param) =
            apply_model_request_limits(&mut too_much_output, Some(metadata), Some(&caps))
                .unwrap_err();
        assert_eq!(param, "max_tokens");
        assert!(message.contains("262145"));

        let mut oversized_allocation = build(json!({
            "model": "qwen/qwen3.6-35b-a3b",
            "prompt_ids": [1],
            "max_tokens": 1,
            "max_ctx": 262145
        }));
        let (_, param) =
            apply_model_request_limits(&mut oversized_allocation, Some(metadata), Some(&caps))
                .unwrap_err();
        assert_eq!(param, "max_ctx");
    }

    #[test]
    fn planned_registry_entries_are_validated_but_never_activated() {
        let parsed = OpenRouterMetadataFile::from_toml(
            r#"
[planned_models.future]
max_output_length = 262144
default_output_length = 8192

[planned_models.future.pricing]
prompt = "0.0000001"
"#,
        )
        .unwrap();
        assert!(parsed.is_empty());

        let error = OpenRouterMetadataFile::from_toml(
            r#"
[planned_models.future]
default_output_length = 8192
"#,
        )
        .unwrap_err();
        assert!(error.contains("requires max_output_length"));
    }

    /// The reviewer's catch on PR #61: gating only /v1/models would have left the
    /// two feeds the SITE and llms.txt advertise publishing the same wrong contract
    /// for the same model. All three feeds resolve the surface through
    /// `declared_surface`, so they cannot disagree.
    #[test]
    fn every_catalog_feed_honours_the_declared_surface() {
        let metadata = OpenRouterMetadataFile::from_toml(
            r#"
[models."qwen/qwen3-embedding-8b"]
surface = "embedding"
created = 1787961600
max_output_length = 1
is_ready = true
is_free = false
discount_to_user = 0.0

[models."qwen/qwen3-embedding-8b".pricing]
prompt = "0.00000001"
cached_prompt = "0.0"
completion = "0.0"

[models."main"]
created = 1787443200
max_output_length = 32768
is_ready = true
is_free = false
discount_to_user = 0.0

[models."main".pricing]
prompt = "0.00000025"
cached_prompt = "0.00000009"
completion = "0.0000012"
"#,
        )
        .unwrap();
        let caps = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            // A switchless thinker (GLM-5.3-Flash, step35) legitimately advertises no
            // structured output — the grammar can never close the unconditional <think>
            // tail. This fixture is the SERVED shape: a qwen with the enable_thinking
            // switch, which honours response_format, so the chat assertions below stand.
            think_switch: true,
            chat_ok: true,
            context_length: 32768,
            ..Default::default()
        };
        let embed = metadata.get("qwen/qwen3-embedding-8b");
        let chat = metadata.get("main");

        // /models?schema=openrouter — the feed the site and llms.txt advertise
        let or = model_entry_openrouter("qwen/qwen3-embedding-8b", Some(&caps), embed);
        let out = &or["output_modalities"][0];
        assert_eq!(out["type"], "embeddings", "openrouter feed: {or}");
        assert!(
            out.get("streaming").is_none(),
            "the embeddings branch declares no streaming property (additionalProperties:false): {out}"
        );
        // EVERY completion-request field is absent, not just tools/reasoning:
        // /v1/embeddings takes {input, dimensions, encoding_format} and nothing here.
        // Publishing max_tokens/structured_outputs for an embedder would contradict
        // /v1/models, which reports structured_output=false for the same model.
        let params = &out["supported_parameters"];
        assert_eq!(
            params.as_object().map(|o| o.len()),
            Some(0),
            "no completion parameter belongs on an embedder row: {params}"
        );
        for field in [
            "tools",
            "tool_choice",
            "reasoning",
            "max_tokens",
            "json_mode",
            "structured_outputs",
            "stop",
            "temperature",
            "seed",
        ] {
            assert!(params[field].is_null(), "{field} leaked onto an embedder");
        }
        assert!(
            out["max_length"].is_null(),
            "a surface emitting no completion tokens advertises no ceiling: {out}"
        );

        // /models?schema=openmodels
        let om = model_entry_openmodels("qwen/qwen3-embedding-8b", Some(&caps), embed)
            .expect("openmodels entry builds");
        assert_eq!(om["output_modalities"], json!(["embeddings"]));
        let features = om["supported_features"].as_array().unwrap();
        assert!(
            !features
                .iter()
                .any(|f| f == "tool_calling" || f == "reasoning"),
            "chat-only features leaked onto an embedder: {features:?}"
        );

        // /v1/models — the surface this change started from
        let v1 = model_entry_v1("qwen/qwen3-embedding-8b", Some(&caps), embed);
        assert_eq!(v1["type"], "embedding");
        assert_eq!(v1["capabilities"]["tools"], false);

        // and a chat model keeps every chat affordance on all three
        let or_chat = model_entry_openrouter("main", Some(&caps), chat);
        let out_chat = &or_chat["output_modalities"][0];
        assert_eq!(out_chat["type"], "text");
        assert_eq!(out_chat["streaming"], true);
        assert!(!out_chat["supported_parameters"]["tools"].is_null());
        assert!(!out_chat["supported_parameters"]["max_tokens"].is_null());
        assert!(!out_chat["supported_parameters"]["structured_outputs"].is_null());
        assert_eq!(out_chat["max_length"]["value"], 32768u64);
        let om_chat = model_entry_openmodels("main", Some(&caps), chat).expect("chat entry builds");
        assert_eq!(om_chat["output_modalities"], json!(["text"]));
        assert!(
            om_chat["supported_features"]
                .as_array()
                .unwrap()
                .iter()
                .any(|f| f == "tool_calling")
        );
        assert_eq!(model_entry_v1("main", Some(&caps), chat)["type"], "chat");
    }

    /// The values on the openrouter feed are NOT ours to choose: they must match the
    /// Provider Monitor 2.4 schema this feed stamps itself with. Round 3 of review #61
    /// caught `embedding`/`score`/`streaming:false` — all invented by analogy with the
    /// text modality, all rejected by the vendored schema's closed `OutputModality`
    /// oneOf. This test reads that pinned file, so the next invented value fails here
    /// instead of in a provider's validator.
    #[test]
    fn openrouter_output_modality_matches_the_vendored_2_4_schema() {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../research/gateway-20260812/raw/sources/",
            "openrouter-provider-schema-v2.4-20260812.json"
        ))
        .expect("vendored Provider Monitor 2.4 schema is in-tree");
        let schema: serde_json::Value = serde_json::from_str(&raw).expect("schema parses");
        let branches = schema["components"]["schemas"]["OutputModality"]["oneOf"]
            .as_array()
            .expect("OutputModality is a oneOf");

        let metadata = OpenRouterMetadataFile::from_toml(
            r#"
[models."embed"]
surface = "embedding"
created = 1787961600
max_output_length = 1
is_ready = true
is_free = false
discount_to_user = 0.0

[models."embed".pricing]
prompt = "0.00000001"
cached_prompt = "0.0"
completion = "0.0"

[models."rr"]
surface = "rerank"
created = 1787961600
max_output_length = 1
is_ready = true
is_free = false
discount_to_user = 0.0

[models."rr".pricing]
prompt = "0.00000003"
cached_prompt = "0.0"
completion = "0.0"

[models."chatty"]
created = 1787443200
max_output_length = 32768
is_ready = true
is_free = false
discount_to_user = 0.0

[models."chatty".pricing]
prompt = "0.00000025"
cached_prompt = "0.00000009"
completion = "0.0000012"
"#,
        )
        .unwrap();
        let caps = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            chat_ok: true,
            context_length: 32768,
            ..Default::default()
        };

        for (alias, want_type) in [
            ("embed", "embeddings"),
            ("rr", "rerank"),
            ("chatty", "text"),
        ] {
            let row = model_entry_openrouter(alias, Some(&caps), metadata.get(alias));
            let modality = &row["output_modalities"][0];
            assert_eq!(modality["type"], want_type, "{alias}: {row}");

            // exactly one branch may accept this type, and it must accept every key we emit
            let branch = branches
                .iter()
                .find(|b| b["properties"]["type"]["enum"][0] == want_type)
                .unwrap_or_else(|| panic!("{want_type:?} is not an OutputModality branch"));
            let allowed: std::collections::BTreeSet<&str> = branch["properties"]
                .as_object()
                .expect("branch properties")
                .keys()
                .map(String::as_str)
                .collect();
            for key in modality.as_object().expect("modality object").keys() {
                assert!(
                    allowed.contains(key.as_str()),
                    "{alias}: {key:?} is not a property of the {want_type:?} branch \
                     (additionalProperties:false); allowed = {allowed:?}"
                );
            }
            for req in branch["required"].as_array().into_iter().flatten() {
                let req = req.as_str().expect("required entry is a string");
                assert!(
                    modality.get(req).is_some(),
                    "{alias}: required property {req:?} missing from the {want_type:?} branch"
                );
            }
        }
    }

    #[test]
    fn openrouter_models_entry_omits_undeclared_optional_fields() {
        let entry = model_entry_openrouter("minimal", None, None);
        let object = entry.as_object().unwrap();
        for field in [
            "hugging_face_id",
            "created",
            "quantization",
            "tokenizer",
            "description",
            "pricing",
            "capacity",
            "is_ready",
            "is_free",
            "discount_to_user",
            "openrouter",
            "datacenters",
            "compliance",
        ] {
            assert!(
                !object.contains_key(field),
                "optional field {field} must be absent, not null"
            );
        }
        assert_eq!(entry["schema_version"], "2.4");
        assert_eq!(entry["input_modalities"][0]["type"], "text");
        assert!(
            entry["input_modalities"][0]
                .get("supported_inputs")
                .is_none()
        );
        assert!(entry["input_modalities"][0].get("pricing").is_none());
        assert!(entry["input_modalities"][0].get("capacity").is_none());
        assert_eq!(entry["output_modalities"][0]["type"], "text");
        assert_eq!(entry["output_modalities"][0]["streaming"], true);
        assert!(entry["output_modalities"][0]["supported_parameters"].is_object());
        assert!(entry["output_modalities"][0].get("max_length").is_none());
        assert!(entry["output_modalities"][0].get("pricing").is_none());
        assert!(entry["output_modalities"][0].get("capacity").is_none());
    }

    #[test]
    fn openmodels_entry_serializes_standard_provider_shape() {
        let metadata = OpenRouterMetadataFile::from_toml(
            r#"
[models."qwen/qwen3.6-27b"]
created = 1786032000
max_output_length = 16384
is_ready = true
is_free = false
discount_to_user = 0.05

[models."qwen/qwen3.6-27b".pricing]
prompt = "0.000000291"
cached_prompt = "0.000000291"
completion = "0.000002763"
request = "0"
"#,
        )
        .unwrap();
        let caps = ModelCaps {
            tools_branch: true,
            qwen_think: true,
            chat_ok: true,
            context_length: 262144,
            ..Default::default()
        };
        let entry = model_entry_openmodels(
            "qwen/qwen3.6-27b",
            Some(&caps),
            metadata.get("qwen/qwen3.6-27b"),
        )
        .unwrap();

        assert_eq!(entry["id"], "qwen/qwen3.6-27b");
        assert_eq!(entry["name"], "qwen/qwen3.6-27b");
        assert_eq!(entry["created"], 1786032000u64);
        assert_eq!(entry["input_modalities"], json!(["text"]));
        assert_eq!(entry["output_modalities"], json!(["text"]));
        assert_eq!(entry["context_length"], 262144u64);
        assert_eq!(entry["max_output_length"], 16384u64);
        assert_eq!(entry["currency"], "USD");
        assert_eq!(entry["pricing"]["prompt"], "0.000000291");
        assert_eq!(entry["pricing"]["completion"], "0.000002763");
        assert_eq!(entry["pricing"]["input_cache_read"], "0.000000291");
        assert_eq!(entry["pricing"]["request"], "0");
        assert_eq!(
            entry["supported_features"],
            json!(["tool_calling", "reasoning"])
        );
        assert_eq!(entry["is_ready"], true);
        assert_eq!(entry["is_free"], false);
        assert_eq!(entry["discount_to_user"], 0.05);
        assert!(entry.get("schema_version").is_none());
        assert!(entry.get("quantization").is_none());
    }

    #[test]
    fn openmodels_entry_rejects_missing_operator_metadata() {
        let caps = ModelCaps {
            context_length: 262144,
            ..Default::default()
        };
        let error = model_entry_openmodels("qwen/qwen3.6-27b", Some(&caps), None).unwrap_err();
        assert_eq!(
            error,
            "OpenModels feed requires MEMRA_MODEL_METADATA for model \"qwen/qwen3.6-27b\""
        );
    }

    #[tokio::test]
    async fn blocking_response_excludes_stop_text_across_token_events() {
        let (tx, rx) = worker::event_channel();
        tx.send(Event::Token {
            id: 1,
            text: "answer\nPro".into(),
        })
        .unwrap();
        tx.send(Event::Token {
            id: 2,
            text: "blem: leaked prompt".into(),
        })
        .unwrap();
        tx.send(Event::Done {
            stop_reason: "Callback".into(),
            n_tokens: 2,
            n_prompt: 8,
            n_cached: 0,
            elapsed_s: 0.5,
            spec: None,
        })
        .unwrap();
        drop(tx);
        let response = blocking_response(
            rx,
            "plain_quant".into(),
            false,
            vec!["Problem:".into()],
            None,
            Envelope::new(false),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["text"], "answer\n");
        assert_eq!(payload["stop_reason"], "Callback");
    }

    /// step37 content walker (lane/step37-vision): the vendor template's separator law
    /// plus the exact per-image expansion, on a real (embedded) 64x64 PNG data URI —
    /// square and small, so the plan is tile-free: <im_start> + 169 pads + <im_end>.
    #[test]
    fn step_walker_expansion_and_separator_law() {
        // 64x64 flat-color PNG, pre-encoded (no base64 dep in this crate).
        const PNG64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAY0lEQVR4nO3PQQ3AIADAQEANmlCD9IngcVnSU9DOe/b4s6UDXjWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgfeKYAYIDsx/LAAAAAElFTkSuQmCC";
        let uri = format!("data:image/png;base64,{PNG64}");
        let content = serde_json::json!([
            {"type": "text", "text": "look at"},
            {"type": "text", "text": "this:"},
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "what is it?"},
        ]);
        let mut pending: Vec<PendingStepImage> = Vec::new();
        let out = content_to_text_vision_step(&content, &mut pending).unwrap();
        let mut expansion = String::from("<im_start>");
        for _ in 0..memra_engine::vision_step::SV_MAIN_ROWS {
            expansion.push_str("<im_patch>");
        }
        expansion.push_str("<im_end>");
        // adjacent text parts join with ONE space; the image resets the separator, so
        // the trailing text abuts the expansion with no space.
        assert_eq!(out, format!("look at this:{expansion}what is it?"));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].plan.n_tiles, 0);
        assert_eq!(pending[0].plan.n_prompt_tokens(), 171);

        // video parts refuse (step37 is image-only), http URLs refuse (SSRF off).
        let vid = serde_json::json!([{ "type": "video_url", "video_url": {"url": uri} }]);
        assert!(content_to_text_vision_step(&vid, &mut Vec::new()).is_err());
        let http = serde_json::json!([
            {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}
        ]);
        assert!(content_to_text_vision_step(&http, &mut Vec::new()).is_err());
    }
}

/// The `system_fingerprint` identity gates (lane/real-system-fingerprint-20260901).
///
/// These exist because the field's only assertion used to be `starts_with("memra-")`, which
/// `memra-unknown` satisfies. Prod served that literal to every customer request for a
/// deploy generation and the test suite was green the whole time.
#[cfg(test)]
mod build_identity_tests {
    use super::{BUILD_GIT_SHA, BUILD_ID_NOTE, BUILD_ID_SRC, SYSTEM_FINGERPRINT, build_id};

    /// The baked fingerprint a customer sees: present, shaped, and not the degraded label.
    #[test]
    fn baked_fingerprint_is_real_and_well_formed() {
        assert!(!SYSTEM_FINGERPRINT.is_empty());
        assert_ne!(SYSTEM_FINGERPRINT, "memra-unknown");
        assert!(
            !SYSTEM_FINGERPRINT.contains("unknown"),
            "fingerprint {SYSTEM_FINGERPRINT:?} still carries the degraded literal"
        );
        assert!(
            build_id::fingerprint_is_well_formed(SYSTEM_FINGERPRINT),
            "fingerprint {SYSTEM_FINGERPRINT:?} is not memra-<version>-<12 hex>"
        );
        // The documented shape names the crate version, so a version bump is visible in the
        // field without reading the id.
        assert!(
            SYSTEM_FINGERPRINT.starts_with(concat!("memra-", env!("CARGO_PKG_VERSION"), "-")),
            "fingerprint {SYSTEM_FINGERPRINT:?} does not name this crate version"
        );
    }

    /// Regression pin on the exact value that shipped, plus the OLD shape it replaced:
    /// `memra-<sha>` must not validate either, or a stale-git build could pass the gate.
    #[test]
    fn the_shape_check_rejects_what_shipped_to_prod() {
        assert!(!build_id::fingerprint_is_well_formed("memra-unknown"));
        assert!(!build_id::fingerprint_is_well_formed(
            "memra-0.123.0-unknown"
        ));
        // The pre-lane form: bare 12-hex git sha, no version component. Assembled rather
        // than written out because `tools/public-boundary-policy.toml`'s `live_fingerprint`
        // rule treats a literal `memra-<12 hex>` as deployment identity leaking into the
        // public repo, and it is right to: that shape used to BE a serving build's id.
        let old_form = format!("memra-{}", "0".repeat(12));
        assert!(!build_id::fingerprint_is_well_formed(&old_form));
        assert!(!build_id::fingerprint_is_well_formed(""));
        assert!(!build_id::fingerprint_is_well_formed("memra-"));
        assert!(!build_id::fingerprint_is_well_formed("memra-0.123.0-"));
        // Wrong id width, and uppercase hex (the renderer emits lowercase).
        assert!(!build_id::fingerprint_is_well_formed("memra-0.123.0-abc"));
        assert!(!build_id::fingerprint_is_well_formed(
            "memra-0.123.0-ABCDEF012345"
        ));
        assert!(!build_id::fingerprint_is_well_formed(
            "memra-0.123.0-zzzzzzzzzzzz"
        ));
        // ...and accepts the real shape.
        assert!(build_id::fingerprint_is_well_formed(
            "memra-0.123.0-4b1f9c02d7a3"
        ));
    }

    /// The identity is a FUNCTION OF THE SOURCE, so two builds of the same tree agree.
    ///
    /// A test cannot run cargo twice, so it does the equivalent and stronger thing: it
    /// re-derives the id from the working tree with the same implementation `build.rs`
    /// used, in a different process, at a different time, from a different working
    /// directory. If the baked id were a function of the build ENVIRONMENT (which a git
    /// lookup is) this would not match.
    #[test]
    fn build_id_is_rederivable_from_the_source_tree() {
        let root = build_id::workspace_root(env!("CARGO_MANIFEST_DIR"));
        let scan = root.as_deref().and_then(build_id::content_id);
        match scan {
            Some(scan) => {
                assert_eq!(
                    BUILD_ID_SRC,
                    build_id::BUILD_ID_SRC_TREE,
                    "the source tree is readable, so the baked id must come from it"
                );
                assert!(BUILD_ID_NOTE.is_empty(), "note set on a non-degraded build");
                let expected =
                    format!(concat!("memra-", env!("CARGO_PKG_VERSION"), "-{}"), scan.id);
                assert_eq!(
                    SYSTEM_FINGERPRINT,
                    expected,
                    "baked fingerprint disagrees with a re-derivation over {} files: the id \
                     is not a pure function of the source tree, or the build script did not \
                     re-run after an edit",
                    scan.files.len()
                );
                assert!(scan.files.len() > 100, "suspiciously small hashed file set");
            }
            None => {
                // Not a pass by omission: an unreadable tree MUST have produced the
                // degraded marker and a stated reason, and the fingerprint must still be
                // shaped (asserted by baked_fingerprint_is_real_and_well_formed).
                assert_eq!(BUILD_ID_SRC, build_id::BUILD_ID_SRC_DEGRADED);
                assert!(
                    !BUILD_ID_NOTE.is_empty(),
                    "a degraded build must state its reason so the boot WARN can print it"
                );
            }
        }
    }

    /// The id is not the git sha, in either direction: the identity must not be history, and
    /// the sha must stay available as a separate extra field.
    #[test]
    fn identity_is_independent_of_git_history() {
        let id = SYSTEM_FINGERPRINT.rsplit_once('-').unwrap().1;
        assert_ne!(
            id, BUILD_GIT_SHA,
            "the content id equals the git sha; the identity must not be history, it has to \
             survive a rewrite that changes every commit"
        );
        assert!(
            !SYSTEM_FINGERPRINT.contains(BUILD_GIT_SHA),
            "the git sha leaked into the customer-visible fingerprint {SYSTEM_FINGERPRINT:?}"
        );
        // The extra field is still populated: either a repo was visible to this build, or it
        // honestly reads `unknown`. Never empty, and never the identity.
        assert!(!BUILD_GIT_SHA.is_empty());
    }

    /// Determinism of the digest itself: same bytes in, same id out, and any change in
    /// content, path, or ordering-relevant input changes it.
    #[test]
    fn content_digest_is_deterministic_and_change_sensitive() {
        let a = build_id::degraded_build_id("memra-server", "0.123.0");
        let b = build_id::degraded_build_id("memra-server", "0.123.0");
        assert_eq!(a, b, "the digest is not deterministic");
        assert_eq!(a.len(), build_id::BUILD_ID_HEX);
        assert!(
            a.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
        assert_ne!(a, build_id::degraded_build_id("memra-server", "0.123.1"));
        assert_ne!(a, build_id::degraded_build_id("memra-serve", "r0.123.0"));
        // Fixed width even when the leading nibbles are zero.
        assert_eq!(build_id::render_build_id(0).len(), build_id::BUILD_ID_HEX);
        assert_eq!(
            build_id::render_build_id(0),
            "0".repeat(build_id::BUILD_ID_HEX)
        );
    }

    /// Two scans of the same unchanged tree in one process agree: the in-process half of
    /// "stable across two builds of the same source".
    #[test]
    fn two_scans_of_one_tree_agree() {
        let Some(root) = build_id::workspace_root(env!("CARGO_MANIFEST_DIR")) else {
            assert_eq!(BUILD_ID_SRC, build_id::BUILD_ID_SRC_DEGRADED);
            return;
        };
        let first = build_id::content_id(&root).expect("first scan");
        let second = build_id::content_id(&root).expect("second scan");
        assert_eq!(first.id, second.id);
        assert_eq!(first.files.len(), second.files.len());
    }
}

/// memra #25: the vision PLACEMENT decision applies to every family whose overlay path reads
/// `MEMRA_VISION_OVERLAY_PUBLISH`, not glm5 alone. step37 serves vision in production; with
/// a glm5-only guard it could boot clean and 500 mid-prefill. The decision gates MEDIA PARTS
/// only: the family switches route the content walkers (step37's text-separator law lives in
/// its walker alone), so text-only prompt bytes never move with the placement.
#[cfg(test)]
mod vision_placement_gate_tests {
    use super::vision_media_admissible;

    #[test]
    fn a_media_part_is_admitted_only_when_the_placement_admits() {
        assert_eq!(vision_media_admissible(true, "image"), Ok(()));
        assert_eq!(vision_media_admissible(true, "video"), Ok(()));
        let err = vision_media_admissible(false, "image").unwrap_err();
        assert!(
            err.starts_with("image input is not enabled on this deployment"),
            "same named refusal the armed-off path gives, so clients see one contract: {err}"
        );
        assert!(
            err.contains("placement"),
            "the refusal names its cause: {err}"
        );
        let err = vision_media_admissible(false, "video").unwrap_err();
        assert!(
            err.starts_with("video input is not enabled on this deployment"),
            "{err}"
        );
    }

    fn live_src() -> String {
        let src: String = include_str!("lib.rs")
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let end = src
            .find("\nmod vision_placement_gate_tests")
            .expect("this test module exists");
        src[..end].to_string()
    }

    /// The comment-stripped body of one top-level item, from `head` to the first column-0 `}`.
    fn item_body<'a>(live: &'a str, head: &str) -> &'a str {
        let start = live
            .find(head)
            .unwrap_or_else(|| panic!("{head} not found — did it get renamed?"));
        let body = &live[start..];
        let end = body.find("\n}\n").expect("item body closes");
        &body[..end]
    }

    /// A char-boundary-safe prefix of at most `n` chars.
    fn head_of(s: &str, n: usize) -> &str {
        match s.char_indices().nth(n) {
            Some((i, _)) => &s[..i],
            None => s,
        }
    }

    /// The family switches select the content walker, and step37's TEXT separator law exists
    /// only in its walker; a switch that folds the placement in changes rendered prompt bytes
    /// for text-only requests whenever the placement is inadmissible (revuto finding on #46).
    /// Anchored on comment-stripped source (wiring-assertions law).
    #[test]
    fn no_family_switch_reads_the_placement_decision() {
        let live = live_src();
        for switch in [
            "fn vision_enabled()",
            "fn gemma_vision_enabled()",
            "fn step_vision_enabled()",
        ] {
            let body = item_body(&live, switch);
            assert!(
                !body.contains("vision_placement_serving")
                    && !body.contains("vision_placement_admits"),
                "{switch} routes text rendering; it must stay keyed on the operator knobs alone"
            );
        }
        let walker = item_body(&live, "fn content_to_text_vision(");
        assert!(
            walker.contains(
                "if step_vision_enabled() {\n        return content_to_text_vision_step(v, step_images);"
            ),
            "the step walker dispatch is keyed on the armed switch alone"
        );
    }

    /// Every arm that ACCEPTS a media part passes the placement gate before it plans anything,
    /// so an inadmissible placement refuses at the waist for every family, never mid-prefill.
    #[test]
    fn every_media_accepting_arm_passes_the_placement_gate() {
        let live = live_src();
        let step = item_body(&live, "fn content_to_text_vision_step(");
        let arm = step
            .split("Some(\"image_url\") => {")
            .nth(1)
            .expect("the step walker has an image arm");
        assert!(
            head_of(arm, 120).contains("vision_placement_admits(\"image\")?;"),
            "the step image arm must pass the placement gate first: {}",
            head_of(arm, 120)
        );
        let walker = item_body(&live, "fn content_to_text_vision(");
        for (head, kind) in [
            (
                "Some(\"image_url\") if gemma_vision_enabled() => {",
                "image",
            ),
            ("Some(\"image_url\") => {", "image"),
            ("Some(\"video_url\") => {", "video"),
        ] {
            let arm = walker
                .split(head)
                .nth(1)
                .unwrap_or_else(|| panic!("{head} is not an arm of the walker"));
            let window = head_of(arm, 400);
            assert!(
                window.contains(&format!("vision_placement_admits(\"{kind}\")?;")),
                "{head} must pass the placement gate before planning anything: {window}"
            );
        }
        // glm5 needs no arm-level gate: its switch reads GLM5_VISION_SERVING, which the worker
        // stores as `tower loaded && placement admissible`, so on an inadmissible placement the
        // glm5 arm never fires and the part falls through to the generic named refusal.
        assert!(live.contains("GLM5_VISION_SERVING.load(std::sync::atomic::Ordering::Acquire)"));
        // The live wrapper feeds the worker's published decision to the pure gate.
        let gate = item_body(&live, "fn vision_placement_admits(");
        assert!(gate.contains("vision_media_admissible(vision_placement_serving(), kind)"));
    }
}
