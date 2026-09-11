//! `/v1/audio/*` — the streaming-transcription surface (lane/audio-endpoint-g8).
//!
//! WHAT THIS IS AND IS NOT. Before this module the v1 router carried models, auth,
//! completions, embeddings, rerank, chat, messages and responses, and nothing else: memra
//! had a CPU-reference speech spine for two families and **no way to reach it over HTTP at
//! all**. That absence is what made G7's live half unarmable and `streams@SLO` unmeasurable
//! (`docs/SPEECH.md` §1, §5.1).
//!
//! This module lands the part of step 3 that does not need a GPU: the session lifecycle, the
//! typed admission shedding, and the decode contract on the wire. It deliberately does NOT
//! fake the transcription itself. No speech operation has a CUDA kernel yet, so an endpoint
//! that answered with text would be answering with something other than the model — and a
//! serving surface that answers fluently with nothing behind it is the single most expensive
//! failure shape in this repository's history. With no engine bound, every path here fails
//! CLOSED with `engine_unbound`, and says which capability is missing.
//!
//! What is therefore real and testable today: routing, the session and frame contract, the
//! resident-session cap, the per-lane bounded queue, the shed taxonomy, the refusal of a
//! sampling parameter on a transcription request, and the temperature a response is required
//! to report. What arrives with step 2: the bytes.
//!
//! THE SHED CONTRACT IS THE POINT. The 2026-09-10 capacity cell lost 2 of 4 streams at c4 on
//! a bounded queue that overflowed silently. A capacity limit that is reached is product
//! behaviour; one that eats a stream is a defect. Every refusal on this surface carries a
//! code, the limit it hit, and the value observed (`memra_lanes::audio_stream::ShedCode`).

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use memra_lanes::audio_stream::{
    AudioMetrics, AudioPolicy, AudioScheduler, AudioShed, FRAME_MS, ShedCode,
};

use crate::{AppState, Envelope, auth};

/// The live session table. One lock, held only across bookkeeping — no device work and no
/// I/O happens under it.
pub(crate) type SharedAudio = Arc<Mutex<AudioScheduler>>;

/// Build the session table from the environment. `engine_bound` is false until a speech
/// pack is resident, which is every deployment today.
pub(crate) fn shared_audio(engine_bound: bool) -> SharedAudio {
    Arc::new(Mutex::new(AudioScheduler::new(
        AudioPolicy::from_env(),
        engine_bound,
    )))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A typed shed becomes an HTTP refusal that keeps every field. `retry_after` is set for the
/// retryable arms so a client backs off instead of hammering a full box.
fn shed_response(shed: &AudioShed) -> Response {
    let status =
        StatusCode::from_u16(shed.code.http_status()).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
    let body = json!({
        "error": {
            "message": shed.message(),
            "type": "audio_admission_error",
            "code": shed.code.as_str(),
            "limit": shed.limit,
            "observed": shed.observed,
            "session": shed.session,
        }
    });
    let mut resp = (status, Json(body)).into_response();
    if shed.code.retryable() {
        resp.headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
        resp.headers_mut().insert(
            "x-should-retry",
            axum::http::HeaderValue::from_static("true"),
        );
    }
    resp
}

/// The decode a transcription request actually resolved to, carried on every response.
///
/// G7's engine obligation, in one struct: "a transcription response must report the
/// temperature that produced it — without that field the contract cannot be checked from the
/// outside at all, which is the same reason HTTP 200 is not a receipt for the text models"
/// (`docs/SPEECH.md` §5.1).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct ResolvedDecode {
    pub deterministic: bool,
    pub strategy: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam_size: Option<u32>,
    /// The rung that produced this response — not the ladder, the rung. A ladder that was
    /// declared and never climbed reports 0.0.
    pub temperature: f32,
    /// The declared ladder, so a client can tell "deterministic" from "deterministic until a
    /// quality trip".
    pub temperature_ladder: Vec<f32>,
    pub serves_partials: bool,
}

/// Resolve the registry's declared ASR decode for `model`, or say why it cannot be served.
/// Boot already refused a silent or self-contradictory entry (`validate_asr_decode_contract`),
/// so the only failure reachable here is "this model is not a transcription model".
fn resolve_decode(st: &AppState, model: &str) -> Result<ResolvedDecode, Box<Response>> {
    let set = st.metadata();
    let Some(md) = set.models.get(model) else {
        return Err(Box::new(crate::bad_request(
            &format!(
                "model {model:?} carries no registry metadata, so its ASR decode is undeclared \
                 and cannot be served"
            ),
            Some("model"),
        )));
    };
    if md.surface.as_deref() != Some("transcription") {
        return Err(Box::new(crate::bad_request(
            &format!(
                "model {model:?} does not serve the transcription surface (declared {:?})",
                md.surface.as_deref().unwrap_or("chat")
            ),
            Some("model"),
        )));
    }
    let ladder = md.decode_temperature_ladder.clone().unwrap_or_default();
    Ok(ResolvedDecode {
        deterministic: md.decode_deterministic.unwrap_or(true),
        strategy: md
            .decode_strategy
            .clone()
            .unwrap_or_else(|| "greedy".to_string()),
        beam_size: md.decode_beam_size,
        temperature: ladder.first().copied().unwrap_or(0.0),
        temperature_ladder: ladder,
        serves_partials: md.serves_partials.unwrap_or(false),
    })
}

/// `POST /v1/audio/transcriptions` request. JSON only for now: the multipart form an OpenAI
/// SDK sends is named as step 2's work rather than half-parsed here, and a multipart request
/// is refused with that sentence instead of a 415 nobody can act on.
#[derive(Deserialize)]
pub(crate) struct TranscriptionReq {
    pub model: String,
    /// Base64 audio. Named `file` to match the OpenAI field a client already sets.
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    /// Accepted ONLY where the registry's declared ladder contains it. Present so an
    /// explicit client request can be honoured or refused with a reason, never ignored.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub response_format: Option<String>,
}

pub(crate) async fn transcriptions(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TranscriptionReq>,
) -> Response {
    let env = Envelope::new(false);
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if ct.starts_with("multipart/") {
        return crate::with_request_id(
            &env.id,
            crate::error_response(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "this endpoint accepts application/json with base64 audio in `file`; the \
                 multipart form an OpenAI SDK sends is not served yet",
                "invalid_request_error",
                Some("file"),
            ),
        );
    }
    let mut model = req.model.clone();
    match crate::canonical_model_id(&st.models, &model) {
        Some(canonical) => model = canonical,
        None => {
            return crate::with_request_id(
                &env.id,
                crate::model_not_found_response(&st.models, &model),
            );
        }
    }
    let decode = match resolve_decode(&st, &model) {
        Ok(d) => d,
        Err(resp) => return crate::with_request_id(&env.id, *resp),
    };
    // NO SAMPLING KEY MAY RESOLVE INTO A TRANSCRIPTION REQUEST. `top_p` is refused outright
    // (no hosted ASR API exposes it); `temperature` is refused unless it names a rung the
    // registry actually declared, because "the client asked for it" is how the ladder gets
    // climbed by accident.
    if req.top_p.is_some() {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                "top_p is not a transcription parameter — no ASR family in scope recommends \
                 nucleus sampling and no hosted ASR API exposes it",
                Some("top_p"),
            ),
        );
    }
    if let Some(t) = req.temperature
        && !decode.temperature_ladder.contains(&t)
        && t != 0.0
    {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                &format!(
                    "temperature {t} is not a rung of this model's declared ladder {:?}; the \
                     ladder is quality-failure recovery, not a sampling knob",
                    decode.temperature_ladder
                ),
                Some("temperature"),
            ),
        );
    }
    if req.response_format.as_deref().is_some_and(|f| f != "json") {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                "only response_format=\"json\" is served",
                Some("response_format"),
            ),
        );
    }
    // A language tag that is present and empty is a client bug worth naming: ASR language
    // selection changes the decode's conditioning, so silently treating "" as auto-detect
    // would make two different requests look like one.
    if req.language.as_deref().is_some_and(str::is_empty) {
        return crate::with_request_id(
            &env.id,
            crate::bad_request(
                "language must be a non-empty tag when present, or omitted for auto-detection",
                Some("language"),
            ),
        );
    }
    if req.file.as_deref().is_none_or(str::is_empty) {
        return crate::with_request_id(
            &env.id,
            crate::bad_request("file must carry base64 audio", Some("file")),
        );
    }
    if let Err(resp) = crate::authenticate(&st.api_auth, &headers) {
        return crate::with_request_id(&env.id, resp);
    }
    // FAIL CLOSED. No speech operation has a CUDA kernel, so there is nothing to run and a
    // 200 here would be a lie about the model. The refusal names the capability.
    let shed = AudioShed {
        code: ShedCode::EngineUnbound,
        limit: 0,
        observed: 0,
        session: None,
    };
    let mut resp = shed_response(&shed);
    // The resolved decode rides even the refusal: an operator checking G7's live half needs
    // to see WHICH decode the registry resolved to before the engine exists.
    resp.headers_mut().insert(
        "x-memra-asr-decode",
        axum::http::HeaderValue::from_str(&format!(
            "{};temperature={};partials={};language={}",
            decode.strategy,
            decode.temperature,
            decode.serves_partials,
            req.language.as_deref().unwrap_or("auto")
        ))
        .unwrap_or(axum::http::HeaderValue::from_static("unparseable")),
    );
    crate::with_request_id(&env.id, resp)
}

#[derive(Deserialize)]
pub(crate) struct OpenSessionReq {
    pub model: String,
}

/// `POST /v1/audio/sessions` — open one streaming lane, or be refused with a code.
pub(crate) async fn open_session(
    State(st): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<OpenSessionReq>,
) -> Response {
    let env = Envelope::new(false);
    let mut model = req.model.clone();
    match crate::canonical_model_id(&st.models, &model) {
        Some(canonical) => model = canonical,
        None => {
            return crate::with_request_id(
                &env.id,
                crate::model_not_found_response(&st.models, &model),
            );
        }
    }
    let decode = match resolve_decode(&st, &model) {
        Ok(d) => d,
        Err(resp) => return crate::with_request_id(&env.id, *resp),
    };
    let tenant = match crate::authenticate(&st.api_auth, &headers) {
        Ok(t) => t,
        Err(resp) => return crate::with_request_id(&env.id, resp),
    };
    let lane = match lane_for(&headers, &tenant) {
        Ok(l) => l,
        Err(resp) => return crate::with_request_id(&env.id, *resp),
    };
    let id = env.id.clone();
    let (result, live, cap) = {
        let mut sched = st.audio.lock().expect("audio session table");
        let r = sched.open(&id, lane, now_ms());
        let cap = sched.policy.max_sessions;
        (r, sched.live(), cap)
    };
    match result {
        Err(shed) => crate::with_request_id(&env.id, shed_response(&shed)),
        Ok(()) => crate::with_request_id(
            &env.id,
            Json(json!({
                "id": id,
                "object": "audio.session",
                "model": model,
                "lane": lane.as_str(),
                "frame_ms": FRAME_MS,
                "queue_frames": queue_cap(&st),
                "sessions_live": live,
                "sessions_max": cap,
                "decode": decode,
            }))
            .into_response(),
        ),
    }
}

fn queue_cap(st: &AppState) -> usize {
    st.audio
        .lock()
        .map(|s| s.policy.queue_frames)
        .unwrap_or_default()
}

fn lane_for(
    headers: &HeaderMap,
    tenant: &auth::TenantCtx,
) -> Result<memra_lanes::Lane, Box<Response>> {
    crate::lane_for_tenant(headers, tenant).map_err(Box::new)
}

#[derive(Deserialize)]
pub(crate) struct FramesReq {
    /// Number of `FRAME_MS` frames being offered. The audio bytes themselves are not read
    /// yet — see the module note; the admission contract is what this surface owns today.
    #[serde(default = "one")]
    pub frames: usize,
}

fn one() -> usize {
    1
}

/// `POST /v1/audio/sessions/{id}/frames` — offer audio. Refuses with `queue_overflow` when
/// this lane has fallen behind real time, which is THE c4 code.
pub(crate) async fn append_frames(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(req): Json<FramesReq>,
) -> Response {
    let env = Envelope::new(false);
    if let Err(resp) = crate::authenticate(&st.api_auth, &headers) {
        return crate::with_request_id(&env.id, resp);
    }
    if req.frames == 0 || req.frames > 256 {
        return crate::with_request_id(
            &env.id,
            crate::bad_request("frames must be 1..=256", Some("frames")),
        );
    }
    let mut depth = 0usize;
    let mut accepted = 0usize;
    let now = now_ms();
    let mut sched = st.audio.lock().expect("audio session table");
    for _ in 0..req.frames {
        match sched.offer(&id, now) {
            Ok(d) => {
                depth = d;
                accepted += 1;
            }
            Err(shed) => {
                // A MULTI-FRAME OFFER CAN BE PARTIALLY ACCEPTED, and the client has to be
                // told how much: frames already queued are not rolled back (they are real
                // audio in a real queue), so a refusal that reported only "429" would leave
                // the caller unable to resume without either losing or duplicating audio.
                // That is the same class of silence as dropping a stream.
                drop(sched);
                let mut resp = shed_response(&shed);
                resp.headers_mut().insert(
                    "x-memra-audio-frames-accepted",
                    axum::http::HeaderValue::from_str(&accepted.to_string())
                        .unwrap_or(axum::http::HeaderValue::from_static("0")),
                );
                return crate::with_request_id(&env.id, resp);
            }
        }
    }
    let m = metrics_json(sched.metrics());
    drop(sched);
    crate::with_request_id(
        &env.id,
        Json(json!({
            "id": id,
            "object": "audio.session.frames",
            "queue_depth": depth,
            "frames_accepted": accepted,
            "audio_ms": req.frames as u64 * FRAME_MS,
            "scheduler": m,
        }))
        .into_response(),
    )
}

/// `POST /v1/audio/sessions/{id}/close` — flush and release the slot.
pub(crate) async fn close_session(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let env = Envelope::new(false);
    if let Err(resp) = crate::authenticate(&st.api_auth, &headers) {
        return crate::with_request_id(&env.id, resp);
    }
    let mut sched = st.audio.lock().expect("audio session table");
    match sched.close(&id) {
        Err(shed) => crate::with_request_id(&env.id, shed_response(&shed)),
        Ok(()) => {
            let live = sched.live();
            drop(sched);
            crate::with_request_id(
                &env.id,
                Json(json!({"id": id, "object": "audio.session.closed", "sessions_live": live}))
                    .into_response(),
            )
        }
    }
}

/// `GET /v1/audio/sessions` — the operator view: the drive shape in force, the caps, and the
/// per-code shed counters. Shed is reported per code because "we refused 900 things" and "we
/// refused 900 things because one lane fell behind" are different operational facts.
pub(crate) async fn list_sessions(State(st): State<AppState>, headers: HeaderMap) -> Response {
    let env = Envelope::new(false);
    if let Err(resp) = crate::authenticate(&st.api_auth, &headers) {
        return crate::with_request_id(&env.id, resp);
    }
    let sched = st.audio.lock().expect("audio session table");
    let body = json!({
        "object": "list",
        // The drive in FORCE, read off the policy the table was built with — not
        // `DriveShape::from_env()`, which answers what the environment says NOW and would
        // report a shape the running scheduler is not using.
        "drive": sched.policy.drive.as_str(),
        "sessions_live": sched.live(),
        "sessions_max": sched.policy.max_sessions,
        "queue_frames": sched.policy.queue_frames,
        "frame_ms": FRAME_MS,
        "final_slo_ms": sched.policy.final_slo_ms,
        "scheduler": metrics_json(sched.metrics()),
    });
    drop(sched);
    crate::with_request_id(&env.id, Json(body).into_response())
}

fn metrics_json(m: &AudioMetrics) -> serde_json::Value {
    json!({
        "opened": m.opened,
        "closed": m.closed,
        "frames_in": m.frames_in,
        "frames_stepped": m.frames_stepped,
        "steps": m.steps,
        "batch_last": m.batch_last,
        "batch_max": m.batch_max,
        "max_queue_depth": m.max_queue_depth,
        "step_duty": m.step_duty(),
        "realtime_streams": m.realtime_streams(),
        "shed": {
            "sessions_exceeded": m.shed_sessions_exceeded,
            "queue_overflow": m.shed_queue_overflow,
            "unknown_session": m.shed_unknown_session,
            "engine_unbound": m.shed_engine_unbound,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OpenRouterModelMetadata, validate_asr_decode_contract};
    use memra_lanes::audio_stream::DriveShape;

    // ---- G7 AT LOAD: the red arms ------------------------------------------------------
    //
    // Each `refuses_*` case below is a registry stanza a person writes, and each one is a
    // shape whose darklanes twin is banked verbatim in
    // `research/g7-asr-decode-20260910/red-arm-demonstration.txt`. They are here because a
    // gate can only check the file it was pointed at: the binary has to refuse on its own.

    fn asr() -> OpenRouterModelMetadata {
        OpenRouterModelMetadata {
            surface: Some("transcription".into()),
            decode_deterministic: Some(true),
            decode_strategy: Some("greedy_batch".into()),
            decode_temperature_ladder: Some(vec![]),
            serves_partials: Some(true),
            ..Default::default()
        }
    }

    fn refusal(md: &OpenRouterModelMetadata) -> String {
        validate_asr_decode_contract("m", md).expect_err("must refuse")
    }

    /// THE SHIPPED SHAPES MUST PASS, or the gate is just a wall. Two arms, both real:
    /// the streaming RNNT student (NeMo `greedy_batch`, emits partials, no ladder) and the
    /// batch Whisper path (faster-whisper beam 5 at temperature 0, no partials).
    #[test]
    fn accepts_the_two_registry_shapes_we_would_actually_serve() {
        validate_asr_decode_contract("nemotron-rnnt", &asr()).expect("streaming RNNT");
        let whisper = OpenRouterModelMetadata {
            decode_strategy: Some("beam".into()),
            decode_beam_size: Some(5),
            serves_partials: Some(false),
            ..asr()
        };
        validate_asr_decode_contract("whisper-large-v3", &whisper).expect("batch Whisper");
        // And a declared ladder, exactly, with the honest determinism flag.
        let ladder = OpenRouterModelMetadata {
            decode_deterministic: Some(false),
            decode_temperature_ladder: Some(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            ..asr()
        };
        validate_asr_decode_contract("whisper-ladder", &ladder).expect("declared ladder");
    }

    /// An entry that says nothing. This is the default a person gets by not thinking about
    /// it, and it is the one that inherits faster-whisper's six-rung ladder.
    #[test]
    fn refuses_an_entry_that_declares_nothing() {
        let silent = OpenRouterModelMetadata {
            surface: Some("transcription".into()),
            ..Default::default()
        };
        let msg = refusal(&silent);
        assert!(msg.contains("decode_deterministic"), "{msg}");
        assert!(
            msg.contains("0.2, 0.4, 0.6, 0.8, 1.0"),
            "the refusal must name what silence inherits: {msg}"
        );
    }

    /// A text model's vendor-sampling stanza copied onto an ASR row.
    #[test]
    fn refuses_a_text_sampling_stanza_copied_across() {
        for md in [
            OpenRouterModelMetadata {
                default_temperature: Some(0.7),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_top_p: Some(0.8),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_top_k: Some(20),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_min_p: Some(0.05),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_presence_penalty: Some(1.5),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_frequency_penalty: Some(0.5),
                ..asr()
            },
            OpenRouterModelMetadata {
                default_repetition_penalty: Some(1.1),
                ..asr()
            },
        ] {
            let msg = refusal(&md);
            assert!(msg.contains("must not declare default_"), "{msg}");
        }
    }

    /// Beam on a streaming partial path: NeMo collapses the beam to accept a partial, so
    /// the declared width is fiction across every chunk boundary.
    #[test]
    fn refuses_beam_on_a_partial_emitting_path() {
        for strategy in ["beam", "default_beam_search"] {
            let md = OpenRouterModelMetadata {
                decode_strategy: Some(strategy.into()),
                decode_beam_size: Some(5),
                serves_partials: Some(true),
                ..asr()
            };
            let msg = refusal(&md);
            assert!(msg.contains("cannot serve streaming partials"), "{msg}");
            assert!(
                msg.contains("kept_hyps"),
                "the refusal must cite the collapse: {msg}"
            );
        }
    }

    /// A ladder that starts hot samples the FIRST attempt of every request.
    #[test]
    fn refuses_a_ladder_that_starts_hot() {
        let md = OpenRouterModelMetadata {
            decode_deterministic: Some(false),
            decode_temperature_ladder: Some(vec![0.2, 0.4]),
            ..asr()
        };
        assert!(refusal(&md).contains("must start at 0.0"));
    }

    /// A repeated or falling rung re-runs the decode that already failed its quality test.
    #[test]
    fn refuses_a_non_monotonic_ladder() {
        for rungs in [vec![0.0, 0.2, 0.2], vec![0.0, 0.4, 0.2], vec![0.0, 0.0]] {
            let md = OpenRouterModelMetadata {
                decode_deterministic: Some(false),
                decode_temperature_ladder: Some(rungs),
                ..asr()
            };
            assert!(refusal(&md).contains("strictly increasing"));
        }
    }

    /// The two self-contradictions. "Deterministic" and "deterministic until a quality
    /// trip, then sampled" are different products.
    #[test]
    fn refuses_a_determinism_flag_that_contradicts_the_ladder() {
        let lying_true = OpenRouterModelMetadata {
            decode_deterministic: Some(true),
            decode_temperature_ladder: Some(vec![0.0, 0.2]),
            ..asr()
        };
        assert!(refusal(&lying_true).contains("contradicts a ladder"));
        let lying_false = OpenRouterModelMetadata {
            decode_deterministic: Some(false),
            decode_temperature_ladder: Some(vec![]),
            ..asr()
        };
        assert!(refusal(&lying_false).contains("declares nondeterminism"));
    }

    /// A beam family with no width, and a width on a family that has no beam.
    #[test]
    fn refuses_a_beam_width_that_does_not_match_the_strategy() {
        let no_width = OpenRouterModelMetadata {
            decode_strategy: Some("beam".into()),
            serves_partials: Some(false),
            ..asr()
        };
        assert!(refusal(&no_width).contains("must declare decode_beam_size"));
        let stray_width = OpenRouterModelMetadata {
            decode_beam_size: Some(5),
            ..asr()
        };
        assert!(refusal(&stray_width).contains("meaningless"));
        let zero = OpenRouterModelMetadata {
            decode_strategy: Some("beam".into()),
            decode_beam_size: Some(0),
            serves_partials: Some(false),
            ..asr()
        };
        assert!(refusal(&zero).contains("not a beam width"));
    }

    /// An unknown strategy name, including the NeMo decoders that REFUSE a partial
    /// hypothesis outright — declaring one would look configured and fail at runtime.
    #[test]
    fn refuses_a_strategy_name_that_is_not_served() {
        for name in ["maes", "tsd", "alsd", "nsc", "sampling", ""] {
            let md = OpenRouterModelMetadata {
                decode_strategy: Some(name.into()),
                ..asr()
            };
            assert!(
                refusal(&md).contains("is not a served ASR decode"),
                "{name}"
            );
        }
    }

    /// A decode contract on a chat row protects nothing and reads as if it did.
    #[test]
    fn refuses_decode_keys_on_a_non_transcription_surface() {
        let chat = OpenRouterModelMetadata {
            surface: Some("chat".into()),
            decode_deterministic: Some(true),
            ..Default::default()
        };
        assert!(refusal(&chat).contains("only meaningful on surface"));
        // and a bare chat row with no decode keys is untouched
        validate_asr_decode_contract(
            "chat",
            &OpenRouterModelMetadata {
                surface: Some("chat".into()),
                ..Default::default()
            },
        )
        .expect("an ordinary chat row must be unaffected");
    }

    // ---- the served surface ------------------------------------------------------------

    fn asr_state() -> AppState {
        use std::collections::HashMap;
        use std::sync::{Arc, RwLock};
        let (cmd_tx, _rx) = std::sync::mpsc::channel::<crate::worker::Cmd>();
        let mut models = HashMap::new();
        models.insert("asr".to_string(), asr());
        models.insert(
            "chatty".to_string(),
            OpenRouterModelMetadata {
                surface: Some("chat".into()),
                ..Default::default()
            },
        );
        AppState {
            cmd_tx,
            models: Arc::new(vec!["asr".into(), "chatty".into()]),
            caps: Arc::new(HashMap::new()),
            openrouter_metadata: Arc::new(RwLock::new(Arc::new(crate::ModelMetadataSet {
                models,
                provider: None,
            }))),
            metering: None,
            budget_tokenizers: None,
            api_auth: crate::ApiAuth::default(),
            metrics_auth: crate::MetricsAuth::default(),
            metrics: crate::SharedMetrics::default(),
            inflight: Arc::new(Default::default()),
            tenant_inflight: Arc::new(Default::default()),
            health: crate::health::WorkerHealth::new(),
            bg: None,
            audio: shared_audio(false),
        }
    }

    async fn body(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json body")
    }

    /// The endpoint EXISTS and it fails CLOSED. No speech operation has a CUDA kernel, so
    /// the only honest answer is a typed 503 that names the missing capability — never a
    /// 200 with text that did not come from the model.
    #[tokio::test]
    async fn transcription_fails_closed_with_the_resolved_decode_attached() {
        let st = asr_state();
        let resp = transcriptions(
            State(st),
            HeaderMap::new(),
            Json(TranscriptionReq {
                model: "asr".into(),
                file: Some("AAAA".into()),
                language: Some("he".into()),
                temperature: None,
                top_p: None,
                response_format: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        // The resolved decode rides even the refusal, so G7's live half can see WHICH
        // decode the registry resolved to before the engine exists.
        let decode = resp
            .headers()
            .get("x-memra-asr-decode")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert_eq!(
            decode,
            "greedy_batch;temperature=0;partials=true;language=he"
        );
        let b = body(resp).await;
        assert_eq!(b["error"]["code"], "engine_unbound");
        assert_eq!(b["error"]["type"], "audio_admission_error");
    }

    /// No sampling key may resolve into a transcription request, and the refusal says why
    /// rather than shrugging. `top_p` is refused outright; a temperature that is not a
    /// declared rung is refused by name.
    #[tokio::test]
    async fn transcription_refuses_sampling_parameters() {
        let st = asr_state();
        let req = |t: Option<f32>, p: Option<f32>| TranscriptionReq {
            model: "asr".into(),
            file: Some("AAAA".into()),
            language: None,
            temperature: t,
            top_p: p,
            response_format: None,
        };
        let resp = transcriptions(
            State(st.clone()),
            HeaderMap::new(),
            Json(req(None, Some(0.9))),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            body(resp).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not a transcription parameter")
        );
        let resp = transcriptions(
            State(st.clone()),
            HeaderMap::new(),
            Json(req(Some(0.4), None)),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            body(resp).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not a rung")
        );
        // temperature 0 is always honest, and reaches the engine-unbound refusal
        let resp = transcriptions(State(st), HeaderMap::new(), Json(req(Some(0.0), None))).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// A chat model on the transcription route is refused by NAME, not by producing
    /// nonsense: this is the same class of error as advertising an embedder as a chat model.
    #[tokio::test]
    async fn transcription_refuses_a_model_that_does_not_serve_the_surface() {
        let resp = transcriptions(
            State(asr_state()),
            HeaderMap::new(),
            Json(TranscriptionReq {
                model: "chatty".into(),
                file: Some("AAAA".into()),
                language: None,
                temperature: None,
                top_p: None,
                response_format: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(
            body(resp).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not serve the transcription surface")
        );
    }

    /// A multipart body — what an OpenAI SDK actually sends — is refused with the sentence
    /// that says what to send instead, not a bare 415.
    #[tokio::test]
    async fn transcription_names_the_multipart_gap_instead_of_guessing() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("multipart/form-data; boundary=x"),
        );
        let resp = transcriptions(
            State(asr_state()),
            headers,
            Json(TranscriptionReq {
                model: "asr".into(),
                file: None,
                language: None,
                temperature: None,
                top_p: None,
                response_format: None,
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert!(
            body(resp).await["error"]["message"]
                .as_str()
                .unwrap()
                .contains("base64 audio in `file`")
        );
    }

    /// Opening a stream is refused with a CODE and a limit, never dropped. With no engine
    /// bound the refusal is `engine_unbound`; the capacity arm is exercised against the
    /// scheduler directly in `memra_lanes::audio_stream`.
    #[tokio::test]
    async fn session_open_refuses_typed_when_no_engine_is_bound() {
        let resp = open_session(
            State(asr_state()),
            HeaderMap::new(),
            Json(OpenSessionReq {
                model: "asr".into(),
            }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "1");
        let b = body(resp).await;
        assert_eq!(b["error"]["code"], "engine_unbound");
    }

    /// A frame for a session nobody opened is a 404 with a code — the caller has to be able
    /// to learn its session is gone. The c4 shape's sin was answering nothing at all.
    #[tokio::test]
    async fn frames_for_an_unknown_session_are_typed_404() {
        let resp = append_frames(
            State(asr_state()),
            HeaderMap::new(),
            Path("ghost".into()),
            Json(FramesReq { frames: 1 }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let b = body(resp).await;
        assert_eq!(b["error"]["code"], "unknown_session");
        assert_eq!(b["error"]["session"], "ghost");
    }

    /// The full lifecycle against a BOUND scheduler: open, offer to the queue cap, be
    /// refused with `queue_overflow` carrying the cap, then close. This is the c4 failure
    /// as a product behaviour instead of a lost stream.
    #[tokio::test]
    async fn queue_overflow_is_a_typed_refusal_that_names_the_cap() {
        let mut st = asr_state();
        // A bound engine with a deliberately tiny queue: the geometry of the incident,
        // scaled down so the test states its own arithmetic.
        st.audio = Arc::new(Mutex::new(AudioScheduler::new(
            AudioPolicy {
                max_sessions: 1,
                queue_frames: 2,
                drive: DriveShape::Fused,
                final_slo_ms: 800,
            },
            true,
        )));
        let opened = open_session(
            State(st.clone()),
            HeaderMap::new(),
            Json(OpenSessionReq {
                model: "asr".into(),
            }),
        )
        .await;
        assert_eq!(opened.status(), StatusCode::OK);
        let ob = body(opened).await;
        let id = ob["id"].as_str().expect("session id").to_string();
        assert_eq!(ob["frame_ms"], 80);
        assert_eq!(ob["queue_frames"], 2);
        assert_eq!(ob["decode"]["strategy"], "greedy_batch");
        assert_eq!(ob["decode"]["deterministic"], true);

        let ok = append_frames(
            State(st.clone()),
            HeaderMap::new(),
            Path(id.clone()),
            Json(FramesReq { frames: 2 }),
        )
        .await;
        assert_eq!(ok.status(), StatusCode::OK);
        assert_eq!(body(ok).await["queue_depth"], 2);

        let shed = append_frames(
            State(st.clone()),
            HeaderMap::new(),
            Path(id.clone()),
            Json(FramesReq { frames: 1 }),
        )
        .await;
        assert_eq!(shed.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            shed.headers().get("x-memra-audio-frames-accepted").unwrap(),
            "0",
            "a refusal must say how many frames it took before refusing"
        );
        assert_eq!(shed.headers().get("retry-after").unwrap(), "1");
        let sb = body(shed).await;
        assert_eq!(sb["error"]["code"], "queue_overflow");
        assert_eq!(sb["error"]["limit"], 2);
        assert_eq!(sb["error"]["observed"], 2);

        // A second session while the cap is 1: the OTHER limit, with its own code.
        let full = open_session(
            State(st.clone()),
            HeaderMap::new(),
            Json(OpenSessionReq {
                model: "asr".into(),
            }),
        )
        .await;
        assert_eq!(full.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body(full).await["error"]["code"], "sessions_exceeded");

        let closed = close_session(State(st.clone()), HeaderMap::new(), Path(id.clone())).await;
        assert_eq!(closed.status(), StatusCode::OK);
        assert_eq!(body(closed).await["sessions_live"], 0);
        // and closing twice is a typed 404, not a silent success
        let again = close_session(State(st.clone()), HeaderMap::new(), Path(id)).await;
        assert_eq!(again.status(), StatusCode::NOT_FOUND);

        // The operator view reports the shed counters PER CODE.
        let listed = list_sessions(State(st), HeaderMap::new()).await;
        let lb = body(listed).await;
        assert_eq!(lb["drive"], "fused");
        assert_eq!(lb["scheduler"]["shed"]["queue_overflow"], 1);
        assert_eq!(lb["scheduler"]["shed"]["sessions_exceeded"], 1);
        assert_eq!(lb["scheduler"]["frames_in"], 2);
    }

    /// A multi-frame offer that overruns the cap must report how much it TOOK. Without
    /// that the caller cannot resume without losing or duplicating audio, which is the
    /// same silence as dropping a stream.
    #[tokio::test]
    async fn a_partially_accepted_offer_reports_what_it_took() {
        let mut st = asr_state();
        st.audio = Arc::new(Mutex::new(AudioScheduler::new(
            AudioPolicy {
                max_sessions: 1,
                queue_frames: 3,
                drive: DriveShape::Fused,
                final_slo_ms: 800,
            },
            true,
        )));
        let opened = open_session(
            State(st.clone()),
            HeaderMap::new(),
            Json(OpenSessionReq {
                model: "asr".into(),
            }),
        )
        .await;
        let id = body(opened).await["id"].as_str().unwrap().to_string();
        // Offer five frames into a three-frame queue: three land, then the refusal.
        let resp = append_frames(
            State(st.clone()),
            HeaderMap::new(),
            Path(id),
            Json(FramesReq { frames: 5 }),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers().get("x-memra-audio-frames-accepted").unwrap(),
            "3"
        );
        let b = body(resp).await;
        assert_eq!(b["error"]["code"], "queue_overflow");
        assert_eq!(b["error"]["observed"], 3);
    }

    /// `frames` is bounded: a client cannot offer an unbounded batch and call it one call.
    #[tokio::test]
    async fn frame_count_is_bounded() {
        for n in [0usize, 257] {
            let resp = append_frames(
                State(asr_state()),
                HeaderMap::new(),
                Path("x".into()),
                Json(FramesReq { frames: n }),
            )
            .await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "frames={n}");
        }
    }
}
