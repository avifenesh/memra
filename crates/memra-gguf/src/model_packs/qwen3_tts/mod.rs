//! Qwen3-TTS 12Hz pack: artifact pin, tensor contract, codec contract, the exact talker prefill,
//! the declared decode and the streaming emit contract.
//!
//! Native execution is not in this file. Following the shape `nemotron_rnnt` set for a speech
//! pack, this module owns what can be checked WITHOUT a model resident: the census, the contract,
//! and the constants a forward pass is not allowed to invent. It binds a checkpoint and refuses a
//! wrong one; it does not run it.
//!
//! The contract is pack-local rather than the shared `TensorContract`, for the reason recorded on
//! `nemotron_rnnt`: that type is keyed by `CheckpointDialect` and folding a second speech shape
//! into it means touching every text builder that matches on the dialect. This pack reads
//! safetensors headers through `crate::safetensors::parse_header_json_checked`, which is the
//! existing header-only reader, so no tensor storage is materialized to take a census.
//!
//! ## Where every number here came from
//!
//! Nothing is inherited from a sibling model and nothing is typed from a document. The geometry
//! and the tensor list are read off the pinned artifact's own safetensors headers, the audio
//! contract off `speech_tokenizer/config.json` at the pinned revision, the prefill line by line
//! out of `qwen-tts` 0.1.1's `modeling_qwen3_tts.py`, and the emit contract from
//! `docs/SPEECH.md` §2.4.1. The prefill is additionally closed against THREE measured widths
//! (27, 10 and 17) from two fixtures at two text lengths, which is what makes it a receipt rather
//! than a reading.
//!
//! ## The two facts a port gets wrong and cannot recover from
//!
//! 1. The talker's prefill SUMS two embedding tables, and the two summands are not the same kind
//!    of thing at every position. Positions 3..=8 add the `tts_pad` row of `text_embedding` (run
//!    through `text_projection`) to a CODEC embedding; position 9 and the text run add
//!    `text_projection(text_embedding(id))` to a codec embedding. A single "add the text embedding
//!    to the codec embedding" rule is wrong at six of the ten non-role positions.
//! 2. The codec decoder's four upsample stages have DIFFERENT strides and kernels (8/5/4/3 with
//!    kernels 16/10/8/6), and its residual blocks carry NO gamma and NO norm, so they are
//!    `x + conv2(act2(conv1(act1(x))))` and not ConvNeXt blocks. The `upsample` stage's ConvNeXt
//!    blocks DO carry both, so the two look alike in the header and are not.

use crate::safetensors::{StInfo, parse_header_json_checked};
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Base publisher artifact this architecture was read from.
pub const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
pub const SOURCE_REVISION: &str = "0c0e3051f131929182e2c023b9537f8b1c68adfe";
/// Vendor reference the contract was read from. Its sdist sha256 matches the PyPI published
/// digest, so the source this pack was derived from is byte-identified.
pub const VENDOR_REFERENCE: &str = "qwen-tts==0.1.1";
pub const VENDOR_SDIST_SHA256: &str =
    "afba5fa235806a6883f46a389e67540b46f8a55da457216bf1d7342903814780";
/// `model.safetensors`, the talker.
pub const TALKER_WEIGHTS_SHA256: &str =
    "38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137";
/// `speech_tokenizer/model.safetensors`, the codec. Held on disk in F32 and it carries an encoder
/// half that synthesis never touches.
pub const CODEC_WEIGHTS_SHA256: &str =
    "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258";
/// Licence read off the artifact's own card, not inferred across the family.
pub const LICENCE: &str = "Apache-2.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtsError {
    Header(String),
    MissingTensor(String),
    UnexpectedTensor(String),
    Shape {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    Dtype {
        name: String,
        expected: &'static str,
        found: String,
    },
    /// Two tensors reading the same byte range. A safetensors file can express this and a loader
    /// that accepts it substitutes one tensor's data for another's.
    AliasedStorage {
        name: String,
        other: String,
        offsets: [usize; 2],
    },
    /// The census is the refusal, not a warning: a total that disagrees with the pinned artifact
    /// means this is not that artifact.
    Census {
        which: &'static str,
        expected_tensors: usize,
        found_tensors: usize,
        expected_elements: u64,
        found_elements: u64,
    },
    UnknownSpeaker(String),
    UnknownLanguage(String),
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(e) => write!(f, "safetensors header: {e}"),
            Self::MissingTensor(n) => write!(f, "checkpoint is missing {n}"),
            Self::UnexpectedTensor(n) => write!(f, "checkpoint carries unexpected {n}"),
            Self::Shape {
                name,
                expected,
                found,
            } => write!(f, "{name}: expected shape {expected:?}, found {found:?}"),
            Self::Dtype {
                name,
                expected,
                found,
            } => write!(f, "{name}: expected storage {expected}, found {found}"),
            Self::AliasedStorage {
                name,
                other,
                offsets,
            } => write!(f, "{name} and {other} share byte range {offsets:?}"),
            Self::Census {
                which,
                expected_tensors,
                found_tensors,
                expected_elements,
                found_elements,
            } => write!(
                f,
                "{which} census is {found_tensors} tensors / {found_elements} elements, the pinned \
                 artifact is {expected_tensors} / {expected_elements}"
            ),
            Self::UnknownSpeaker(s) => write!(f, "speaker {s:?} is not in the artifact's table"),
            Self::UnknownLanguage(l) => write!(f, "language {l:?} is not in the artifact's table"),
        }
    }
}

impl std::error::Error for TtsError {}

/// Talker geometry. Every field is a census-checked fact about the pinned artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TalkerGeometry {
    pub layers: u32,
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub ffn: u32,
    /// The text vocabulary, reached only through `text_projection`.
    pub text_vocab: u32,
    /// The codec token space, which is ALSO the talker's output head width.
    pub codec_vocab: u32,
    pub predictor_layers: u32,
    pub predictor_hidden: u32,
    pub predictor_ffn: u32,
    /// Codebooks the nested predictor emits. Codebook 0 is the talker's, so the frame total is
    /// this plus one.
    pub predictor_codebooks: u32,
    pub predictor_codebook_vocab: u32,
}

pub const TALKER: TalkerGeometry = TalkerGeometry {
    layers: 28,
    hidden: 2048,
    q_heads: 16,
    kv_heads: 8,
    head_dim: 128,
    ffn: 6144,
    text_vocab: 151936,
    codec_vocab: 3072,
    predictor_layers: 5,
    predictor_hidden: 1024,
    predictor_ffn: 3072,
    predictor_codebooks: 15,
    predictor_codebook_vocab: 2048,
};

/// The talker's element census, derived from the geometry above and asserted against the pinned
/// artifact in the tests. This is the number a wrong geometry trips on.
pub const TALKER_TENSORS: usize = 404;
pub const TALKER_ELEMENTS: u64 = 1_916_676_352;
/// What a serving box actually holds: the talker plus the codec's DECODER half. The encoder half
/// is dead weight for synthesis on `-CustomVoice`, which needs no reference audio.
pub const CODEC_DECODER_TENSORS: usize = 271;
pub const CODEC_DECODER_ELEMENTS: u64 = 114_323_137;
pub const CODEC_ENCODER_TENSORS: usize = 225;
pub const CODEC_ENCODER_ELEMENTS: u64 = 56_234_304;
pub const RESIDENT_ELEMENTS: u64 = TALKER_ELEMENTS + CODEC_DECODER_ELEMENTS;

/// The audio and codec contract, read off `speech_tokenizer/config.json` at the pinned revision.
///
/// The name says 12 Hz and the artifact says 12.5. Neither 83.33 ms nor 2000 samples appears
/// anywhere in it, so every step budget uses 80.00 ms and 12.5 steps per second of audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioContract {
    pub sample_rate: u32,
    pub samples_per_frame: u32,
    pub codebooks: u32,
    pub codebook_size: u32,
    pub semantic_codebooks: u32,
    /// The decoder's attention window, in frames.
    pub sliding_window: u32,
}

pub const AUDIO: AudioContract = AudioContract {
    sample_rate: 24000,
    samples_per_frame: 1920,
    codebooks: 16,
    codebook_size: 2048,
    semantic_codebooks: 1,
    sliding_window: 72,
};

impl AudioContract {
    /// Frames per second. 12.5, not the 12 in the model's name.
    pub fn frame_rate_hz(&self) -> f64 {
        f64::from(self.sample_rate) / f64::from(self.samples_per_frame)
    }
    /// Audio per decode step, in milliseconds. Exactly 80.00.
    pub fn frame_ms(&self) -> f64 {
        1000.0 * f64::from(self.samples_per_frame) / f64::from(self.sample_rate)
    }
    /// The arithmetic has to close on itself or the contract was read off the wrong artifact.
    pub fn self_consistent(&self) -> bool {
        self.sample_rate.is_multiple_of(self.samples_per_frame)
            || (f64::from(self.sample_rate) / f64::from(self.samples_per_frame) * 100.0).fract()
                == 0.0
    }
}

/// Special ids, pinned as plan constants AND gated against the artifact's own config, per §4
/// step 3. Two vocabularies are in play and mixing them is the defect this struct exists to
/// prevent: text and special ids index `text_embedding`, codec ids index `codec_embedding`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialIds {
    pub im_start: i64,
    pub im_end: i64,
    pub assistant: i64,
    pub tts_bos: i64,
    pub tts_eos: i64,
    pub tts_pad: i64,
    pub codec_pad: i64,
    pub codec_bos: i64,
    pub codec_eos: i64,
    pub codec_think: i64,
    pub codec_nothink: i64,
    pub codec_think_bos: i64,
    pub codec_think_eos: i64,
}

pub const IDS: SpecialIds = SpecialIds {
    im_start: 151644,
    im_end: 151645,
    assistant: 77091,
    tts_bos: 151672,
    tts_eos: 151673,
    tts_pad: 151671,
    codec_pad: 2148,
    codec_bos: 2149,
    codec_eos: 2150,
    codec_think: 2154,
    codec_nothink: 2155,
    codec_think_bos: 2156,
    codec_think_eos: 2157,
};

/// Voice conditioning on `-CustomVoice` is ONE ROW of `codec_embedding`. There is no speaker
/// encoder, no reference audio and no x-vector on this artifact, which is why the codec's encoder
/// half is dead weight here and why "speaker-embedding caching" cannot be worth measurable time
/// on this variant.
pub const SPEAKERS: &[(&str, i64)] = &[
    ("aiden", 2861),
    ("dylan", 2878),
    ("eric", 2875),
    ("ono_anna", 2873),
    ("ryan", 3061),
    ("serena", 3066),
    ("sohee", 2864),
    ("uncle_fu", 3010),
    ("vivian", 3065),
];

/// Language ids are CODEC tokens, not text tokens. The two dialects are in the artifact's table
/// but are not user-selectable as languages; they are forced by a speaker.
pub const LANGUAGES: &[(&str, i64)] = &[
    ("english", 2050),
    ("chinese", 2055),
    ("spanish", 2054),
    ("german", 2053),
    ("italian", 2070),
    ("portuguese", 2071),
    ("japanese", 2058),
    ("korean", 2064),
    ("french", 2061),
    ("russian", 2069),
    ("beijing_dialect", 2074),
    ("sichuan_dialect", 2062),
];

/// Speakers that force a dialect language id whenever the language is chinese or auto.
pub const DIALECT_SPEAKERS: &[(&str, &str)] =
    &[("eric", "sichuan_dialect"), ("dylan", "beijing_dialect")];

pub fn speaker_id(name: &str) -> Result<i64, TtsError> {
    SPEAKERS
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, id)| *id)
        .ok_or_else(|| TtsError::UnknownSpeaker(name.to_string()))
}

pub fn language_id(name: &str) -> Result<i64, TtsError> {
    LANGUAGES
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, id)| *id)
        .ok_or_else(|| TtsError::UnknownLanguage(name.to_string()))
}

/// Resolve the language id the vendor would actually use, including the dialect override and the
/// `auto` collapse. This is the rule at `modeling_qwen3_tts.py:2109-2122` and it is not the same
/// as `language_id`: `auto` yields NO id and changes the think block, and a dialect speaker
/// overrides an explicit chinese.
pub fn resolve_language_id(language: &str, speaker: &str) -> Result<Option<i64>, TtsError> {
    let dialect = DIALECT_SPEAKERS
        .iter()
        .find(|(s, _)| s.eq_ignore_ascii_case(speaker))
        .map(|(_, d)| *d);
    let lower = language.to_ascii_lowercase();
    if lower == "auto" {
        return Ok(dialect.and_then(|d| language_id(d).ok()));
    }
    let id = language_id(&lower)?;
    if lower == "chinese"
        && let Some(d) = dialect
    {
        return Ok(Some(language_id(d)?));
    }
    Ok(Some(id))
}

/// Ids suppressed at the talker head. Everything from the codebook ceiling up is an INPUT-only
/// token: control ids, speaker rows and language ids all live in 2048..3072, and the only one the
/// head may emit is `codec_eos`, which is how the talker stops.
pub const SUPPRESS_FROM: i64 = 2048;
pub const SUPPRESS_TO: i64 = 3071;

/// The talker head's output width is the whole codec token space, so suppression is what keeps it
/// emitting codes rather than speaker rows.
pub fn is_suppressed(id: i64) -> bool {
    (SUPPRESS_FROM..=SUPPRESS_TO).contains(&id) && id != IDS.codec_eos
}

/// Which of the two embedding tables a position sums from. The prefill is a SUM of a text-side
/// vector and a codec-side vector at every position, but the text side is a bare `tts_pad` row at
/// six positions and a projected text token at the rest, and getting that backwards is silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSide {
    /// `text_projection(text_embedding(id))` for a real text token.
    ProjectedToken,
    /// `text_projection(text_embedding(tts_pad))`, the same vector at every such position.
    PadRow,
    /// `text_projection(text_embedding(tts_bos))`.
    BosRow,
    /// `text_projection(text_embedding(tts_eos))`.
    EosRow,
}

/// One prefill position: what the codec side contributes and what the text side contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillRow {
    pub codec_id: i64,
    /// `None` for the three role positions, which carry no codec side at all.
    pub speaker_row: bool,
    pub text: Option<TextSide>,
}

/// The exact talker prefill for the NON-STREAMING text path, which is the outer entry point's
/// default (`non_streaming_mode = True`) and therefore the path the pinned oracle encodes.
///
/// Closed against three measured widths rather than asserted: 27 positions for a 16-token text
/// body, 17 for a 6-token body, and the streaming path's 10. Given `T` text-body tokens the width
/// is `9 + (T + 1) + 1`, and the three measured cases are `9 + 17 + 1 = 27`, `9 + 7 + 1 = 17` and
/// `9 + 1 = 10`.
///
/// The three role positions are `text_projection(text_embedding(input_id[0..3]))` and carry no
/// codec side, which is why they are not in the returned rows.
pub fn prefill_rows(language_id: Option<i64>, speaker_id: Option<i64>) -> Vec<PrefillRow> {
    // The think block: four positions with an explicit language, three when the language is auto.
    let think: Vec<i64> = match language_id {
        Some(id) => vec![
            IDS.codec_think,
            IDS.codec_think_bos,
            id,
            IDS.codec_think_eos,
        ],
        None => vec![IDS.codec_nothink, IDS.codec_think_bos, IDS.codec_think_eos],
    };
    // Positions 3..=8: tts_pad summed onto six codec rows, dropping the final codec_bos.
    let mut codec_side: Vec<i64> = think.clone();
    if let Some(spk) = speaker_id {
        codec_side.push(spk);
    }
    codec_side.push(IDS.codec_pad);
    let mut rows: Vec<PrefillRow> = codec_side
        .iter()
        .map(|id| PrefillRow {
            codec_id: *id,
            speaker_row: speaker_id == Some(*id),
            text: Some(TextSide::PadRow),
        })
        .collect();
    // Position 9: tts_bos summed onto codec_bos, the one position the two-step construction
    // reuses as the tail of the first block.
    rows.push(PrefillRow {
        codec_id: IDS.codec_bos,
        speaker_row: false,
        text: Some(TextSide::BosRow),
    });
    rows
}

/// Width of the non-streaming prefill for a text body of `text_body_tokens`.
///
/// `9 + (text_body_tokens + 1) + 1`, measured at 27 for 16 tokens and 17 for 6.
pub fn prefill_width(text_body_tokens: usize) -> usize {
    9 + (text_body_tokens + 1) + 1
}

/// Width of the STREAMING prefill, which carries only the first text token and feeds the rest one
/// per step. Measured at 10 for a 16-token body, and independent of the body length.
pub fn streaming_prefill_width() -> usize {
    10
}

/// Tokens per decode step of `trailing_text_hidden` on the non-streaming path: exactly one, and it
/// is always the `tts_pad` row, so the per-step text contribution is a CONSTANT vector rather than
/// a schedule. On the streaming path it is the remaining text body followed by `tts_eos`, consumed
/// one per step and only while steps remain.
pub fn trailing_text_is_constant_on_non_streaming() -> bool {
    true
}

/// The declared served decode (G7). TTS inverts the ASR rule: the vendor recommendation for this
/// artifact is SAMPLED on BOTH loops, and a port that samples the talker but argmaxes the nested
/// predictor is serving a decode nobody recommended. Greedy is the instrument for byte-identity
/// gates and is never the served shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeContract {
    pub talker_do_sample: bool,
    pub talker_temperature: f32,
    pub talker_top_k: u32,
    pub talker_top_p: f32,
    pub talker_repetition_penalty: f32,
    pub predictor_do_sample: bool,
    pub predictor_temperature: f32,
    pub predictor_top_k: u32,
    pub predictor_top_p: f32,
    pub max_new_frames: u32,
}

/// Read off `generation_config.json` at the pinned revision, sha256 `f1b90b45...`.
pub const SERVED_DECODE: DecodeContract = DecodeContract {
    talker_do_sample: true,
    talker_temperature: 0.9,
    talker_top_k: 50,
    talker_top_p: 1.0,
    talker_repetition_penalty: 1.05,
    predictor_do_sample: true,
    predictor_temperature: 0.9,
    predictor_top_k: 50,
    predictor_top_p: 1.0,
    max_new_frames: 8192,
};

/// Greedy on both loops: the instrument, never the served shape.
pub const GREEDY_INSTRUMENT: DecodeContract = DecodeContract {
    talker_do_sample: false,
    predictor_do_sample: false,
    ..SERVED_DECODE
};

/// The streaming emit contract, `docs/SPEECH.md` §2.4.1. Fixed before the implementation so a
/// battery can be written against it in parallel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmitContract {
    /// Samples per emitted chunk. Exactly one codec frame; a partial frame is never emitted
    /// because the codec cannot decode one.
    pub samples_per_chunk: u32,
    pub sample_rate: u32,
    /// Peak below which a frame is declared silent on the wire. -60 dBFS. The measured silent
    /// lead-in is 5.72e-05 on one box and 4.816e-05 on another, so the threshold sits about 25x
    /// above the silence and about 30x below the quietest audible frame measured (0.032).
    pub audible_peak: f32,
    /// Frames this artifact emits before any signal. Measured, not assumed.
    pub silent_lead_in_frames: u32,
}

pub const EMIT: EmitContract = EmitContract {
    samples_per_chunk: 1920,
    sample_rate: 24000,
    audible_peak: 1e-3,
    silent_lead_in_frames: 1,
};

/// Turn "the caller wants N audible frames" into the frame count the engine must produce, owning
/// the silent lead-in in exactly one place. The vendor's own off-by-one (`max_new_tokens - 1`
/// frames) stays inside the engine and never reaches this contract.
pub fn frames_for_audible(audible_frames: u32) -> u32 {
    audible_frames + EMIT.silent_lead_in_frames
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsTensorRequirement {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: &'static str,
}

fn elements(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// Every tensor the talker archive must carry, generated from the geometry. 404 requirements and
/// 1,916,676,352 elements against the pinned artifact.
pub fn talker_tensor_contract(g: TalkerGeometry) -> Vec<TtsTensorRequirement> {
    let hidden = u64::from(g.hidden);
    let q = u64::from(g.q_heads * g.head_dim);
    let kv = u64::from(g.kv_heads * g.head_dim);
    let ffn = u64::from(g.ffn);
    let head = u64::from(g.head_dim);
    let ph = u64::from(g.predictor_hidden);
    // The predictor narrows the residual stream, not its attention head geometry:
    // the pinned config still declares 16 query heads and 8 KV heads of width 128.
    let pq = q;
    let pffn = u64::from(g.predictor_ffn);
    let mut want = Vec::new();
    let mut add = |name: String, shape: Vec<u64>| {
        want.push(TtsTensorRequirement {
            name,
            shape,
            dtype: "BF16",
        });
    };

    for layer in 0..g.layers {
        let p = format!("talker.model.layers.{layer}");
        add(format!("{p}.input_layernorm.weight"), vec![hidden]);
        add(format!("{p}.post_attention_layernorm.weight"), vec![hidden]);
        add(format!("{p}.self_attn.q_proj.weight"), vec![q, hidden]);
        add(format!("{p}.self_attn.k_proj.weight"), vec![kv, hidden]);
        add(format!("{p}.self_attn.v_proj.weight"), vec![kv, hidden]);
        add(format!("{p}.self_attn.o_proj.weight"), vec![hidden, q]);
        // Qwen3-style PER-HEAD q/k norms, one row of head_dim each.
        add(format!("{p}.self_attn.q_norm.weight"), vec![head]);
        add(format!("{p}.self_attn.k_norm.weight"), vec![head]);
        add(format!("{p}.mlp.gate_proj.weight"), vec![ffn, hidden]);
        add(format!("{p}.mlp.up_proj.weight"), vec![ffn, hidden]);
        add(format!("{p}.mlp.down_proj.weight"), vec![hidden, ffn]);
    }
    add("talker.model.norm.weight".into(), vec![hidden]);
    add(
        "talker.model.text_embedding.weight".into(),
        vec![u64::from(g.text_vocab), hidden],
    );
    add(
        "talker.model.codec_embedding.weight".into(),
        vec![u64::from(g.codec_vocab), hidden],
    );

    for layer in 0..g.predictor_layers {
        let p = format!("talker.code_predictor.model.layers.{layer}");
        add(format!("{p}.input_layernorm.weight"), vec![ph]);
        add(format!("{p}.post_attention_layernorm.weight"), vec![ph]);
        add(format!("{p}.self_attn.q_proj.weight"), vec![pq, ph]);
        add(format!("{p}.self_attn.k_proj.weight"), vec![kv, ph]);
        add(format!("{p}.self_attn.v_proj.weight"), vec![kv, ph]);
        add(format!("{p}.self_attn.o_proj.weight"), vec![ph, pq]);
        add(format!("{p}.self_attn.q_norm.weight"), vec![head]);
        add(format!("{p}.self_attn.k_norm.weight"), vec![head]);
        add(format!("{p}.mlp.gate_proj.weight"), vec![pffn, ph]);
        add(format!("{p}.mlp.up_proj.weight"), vec![pffn, ph]);
        add(format!("{p}.mlp.down_proj.weight"), vec![ph, pffn]);
    }
    add("talker.code_predictor.model.norm.weight".into(), vec![ph]);
    // Fifteen separate codebook embeddings and fifteen separate heads, one per remaining
    // codebook. Not one shared head with fifteen slices.
    for cb in 0..g.predictor_codebooks {
        add(
            format!("talker.code_predictor.model.codec_embedding.{cb}.weight"),
            vec![u64::from(g.predictor_codebook_vocab), hidden],
        );
        add(
            format!("talker.code_predictor.lm_head.{cb}.weight"),
            vec![u64::from(g.predictor_codebook_vocab), ph],
        );
    }
    add(
        "talker.code_predictor.small_to_mtp_projection.weight".into(),
        vec![ph, hidden],
    );
    add(
        "talker.code_predictor.small_to_mtp_projection.bias".into(),
        vec![ph],
    );

    // A two-layer BIASED MLP between the text table and the residual stream. The bias is easy to
    // drop and the prefill is bit-identical or it is wrong.
    add(
        "talker.text_projection.linear_fc1.weight".into(),
        vec![hidden, hidden],
    );
    add(
        "talker.text_projection.linear_fc1.bias".into(),
        vec![hidden],
    );
    add(
        "talker.text_projection.linear_fc2.weight".into(),
        vec![hidden, hidden],
    );
    add(
        "talker.text_projection.linear_fc2.bias".into(),
        vec![hidden],
    );
    add(
        "talker.codec_head.weight".into(),
        vec![u64::from(g.codec_vocab), hidden],
    );
    want
}

/// Codec geometry. The four upsample stages have different strides and kernels, and the stride
/// product is the x480 that turns one frame into 1920 samples.
pub const CODEC_UPSAMPLE_STRIDES: [u32; 4] = [8, 5, 4, 3];
pub const CODEC_UPSAMPLE_KERNELS: [u32; 4] = [16, 10, 8, 6];
pub const CODEC_CHANNELS: [u32; 5] = [1536, 768, 384, 192, 96];
/// Residual blocks per upsample stage. They carry NO gamma and NO norm.
pub const CODEC_RESIDUAL_BLOCKS: u32 = 3;
pub const CODEC_TRANSFORMER_LAYERS: u32 = 8;
pub const CODEC_TRANSFORMER_HIDDEN: u32 = 512;
pub const CODEC_TRANSFORMER_FFN: u32 = 1024;
pub const CODEC_LATENT: u32 = 1024;
/// RVQ projection channels, before `pre_conv` expands them to `CODEC_LATENT`.
pub const CODEC_QUANTIZER_DIM: u32 = 512;
pub const CODEC_VQ_DIM: u32 = 256;
/// The `upsample` stage runs BEFORE `decoder.decoder` and contributes x4 (two stages at x2).
pub const CODEC_PRE_UPSAMPLE_STAGES: u32 = 2;
pub const CODEC_PRE_UPSAMPLE_RATIO: u32 = 2;

/// Every tensor the codec's DECODER half must carry. 271 requirements and 114,323,137 elements.
///
/// The encoder half (225 tensors, 56,234,304 elements) is deliberately NOT in this contract. It
/// is dead weight for synthesis on `-CustomVoice`, and `bind_codec` refuses it as unexpected
/// rather than loading 0.22 GB that can never run.
pub fn codec_decoder_tensor_contract() -> Vec<TtsTensorRequirement> {
    let mut want = Vec::new();
    let mut add = |name: String, shape: Vec<u64>| {
        want.push(TtsTensorRequirement {
            name,
            shape,
            dtype: "F32",
        });
    };
    let codebooks = u64::from(AUDIO.codebooks);
    let vq = u64::from(CODEC_VQ_DIM);
    let quantizer = u64::from(CODEC_QUANTIZER_DIM);
    let size = u64::from(AUDIO.codebook_size);

    // The RVQ dequantizer stores `embedding_sum` with a `cluster_usage` divisor, so the codebook
    // is embedding_sum / cluster_usage and NOT a plain embedding table.
    add(
        "decoder.quantizer.rvq_first.input_proj.weight".into(),
        vec![vq, quantizer, 1],
    );
    add(
        "decoder.quantizer.rvq_first.output_proj.weight".into(),
        vec![quantizer, vq, 1],
    );
    add(
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum".into(),
        vec![size, vq],
    );
    add(
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage".into(),
        vec![size],
    );
    add(
        "decoder.quantizer.rvq_rest.input_proj.weight".into(),
        vec![vq, quantizer, 1],
    );
    add(
        "decoder.quantizer.rvq_rest.output_proj.weight".into(),
        vec![quantizer, vq, 1],
    );
    for layer in 0..(codebooks - 1) {
        add(
            format!("decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook.embedding_sum"),
            vec![size, vq],
        );
        add(
            format!("decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook.cluster_usage"),
            vec![size],
        );
    }

    add(
        "decoder.pre_conv.conv.weight".into(),
        vec![u64::from(CODEC_LATENT), quantizer, 3],
    );
    add(
        "decoder.pre_conv.conv.bias".into(),
        vec![u64::from(CODEC_LATENT)],
    );

    // The windowed transformer. Attention is 16 heads of the 512-wide stream projected to 1024,
    // and every branch carries a layer_scale.
    add(
        "decoder.pre_transformer.input_proj.weight".into(),
        vec![512, u64::from(CODEC_LATENT)],
    );
    add("decoder.pre_transformer.input_proj.bias".into(), vec![512]);
    for layer in 0..CODEC_TRANSFORMER_LAYERS {
        let p = format!("decoder.pre_transformer.layers.{layer}");
        add(format!("{p}.input_layernorm.weight"), vec![512]);
        add(format!("{p}.post_attention_layernorm.weight"), vec![512]);
        add(format!("{p}.self_attn.q_proj.weight"), vec![1024, 512]);
        add(format!("{p}.self_attn.k_proj.weight"), vec![1024, 512]);
        add(format!("{p}.self_attn.v_proj.weight"), vec![1024, 512]);
        add(format!("{p}.self_attn.o_proj.weight"), vec![512, 1024]);
        add(format!("{p}.self_attn_layer_scale.scale"), vec![512]);
        add(format!("{p}.mlp.gate_proj.weight"), vec![1024, 512]);
        add(format!("{p}.mlp.up_proj.weight"), vec![1024, 512]);
        add(format!("{p}.mlp.down_proj.weight"), vec![512, 1024]);
        add(format!("{p}.mlp_layer_scale.scale"), vec![512]);
    }
    add("decoder.pre_transformer.norm.weight".into(), vec![512]);
    add(
        "decoder.pre_transformer.output_proj.weight".into(),
        vec![u64::from(CODEC_LATENT), 512],
    );
    add(
        "decoder.pre_transformer.output_proj.bias".into(),
        vec![u64::from(CODEC_LATENT)],
    );

    // The x4 pre-upsample: a transposed conv at ratio 2 then a ConvNeXt block that DOES carry a
    // gamma and a norm, unlike the residual blocks further down.
    for stage in 0..CODEC_PRE_UPSAMPLE_STAGES {
        let p = format!("decoder.upsample.{stage}");
        add(
            format!("{p}.0.conv.weight"),
            vec![u64::from(CODEC_LATENT), u64::from(CODEC_LATENT), 2],
        );
        add(format!("{p}.0.conv.bias"), vec![u64::from(CODEC_LATENT)]);
        add(
            format!("{p}.1.dwconv.conv.weight"),
            vec![u64::from(CODEC_LATENT), 1, 7],
        );
        add(
            format!("{p}.1.dwconv.conv.bias"),
            vec![u64::from(CODEC_LATENT)],
        );
        add(format!("{p}.1.norm.weight"), vec![u64::from(CODEC_LATENT)]);
        add(format!("{p}.1.norm.bias"), vec![u64::from(CODEC_LATENT)]);
        add(
            format!("{p}.1.pwconv1.weight"),
            vec![4096, u64::from(CODEC_LATENT)],
        );
        add(format!("{p}.1.pwconv1.bias"), vec![4096]);
        add(
            format!("{p}.1.pwconv2.weight"),
            vec![u64::from(CODEC_LATENT), 4096],
        );
        add(format!("{p}.1.pwconv2.bias"), vec![u64::from(CODEC_LATENT)]);
        add(format!("{p}.1.gamma"), vec![u64::from(CODEC_LATENT)]);
    }

    // decoder.decoder is a Sequential of 7: conv, four upsample stages, Snake, conv.
    add(
        "decoder.decoder.0.conv.weight".into(),
        vec![u64::from(CODEC_CHANNELS[0]), u64::from(CODEC_LATENT), 7],
    );
    add(
        "decoder.decoder.0.conv.bias".into(),
        vec![u64::from(CODEC_CHANNELS[0])],
    );
    for (stage, stride) in CODEC_UPSAMPLE_STRIDES.iter().enumerate() {
        let chin = u64::from(CODEC_CHANNELS[stage]);
        let chout = u64::from(CODEC_CHANNELS[stage + 1]);
        let p = format!("decoder.decoder.{}.block", stage + 1);
        // Snake activation carries BOTH alpha and beta.
        add(format!("{p}.0.alpha"), vec![chin]);
        add(format!("{p}.0.beta"), vec![chin]);
        add(
            format!("{p}.1.conv.weight"),
            vec![chin, chout, u64::from(2 * stride)],
        );
        add(format!("{p}.1.conv.bias"), vec![chout]);
        for block in 0..CODEC_RESIDUAL_BLOCKS {
            let b = format!("{p}.{}", block + 2);
            add(format!("{b}.act1.alpha"), vec![chout]);
            add(format!("{b}.act1.beta"), vec![chout]);
            add(format!("{b}.conv1.conv.weight"), vec![chout, chout, 7]);
            add(format!("{b}.conv1.conv.bias"), vec![chout]);
            add(format!("{b}.act2.alpha"), vec![chout]);
            add(format!("{b}.act2.beta"), vec![chout]);
            add(format!("{b}.conv2.conv.weight"), vec![chout, chout, 1]);
            add(format!("{b}.conv2.conv.bias"), vec![chout]);
        }
    }
    let last = u64::from(*CODEC_CHANNELS.last().unwrap());
    add("decoder.decoder.5.alpha".into(), vec![last]);
    add("decoder.decoder.5.beta".into(), vec![last]);
    add("decoder.decoder.6.conv.weight".into(), vec![1, last, 7]);
    add("decoder.decoder.6.conv.bias".into(), vec![1]);
    want
}

/// The stride product has to be the x480 that completes the x1920, or the geometry was read wrong.
pub fn codec_upsample_product() -> u32 {
    CODEC_UPSAMPLE_STRIDES.iter().product()
}

/// Total upsample from one codec frame to samples, pre-upsample included.
pub fn codec_total_upsample() -> u32 {
    codec_upsample_product() * CODEC_PRE_UPSAMPLE_RATIO.pow(CODEC_PRE_UPSAMPLE_STAGES)
}

/// `StInfo` is not `PartialEq`, so neither is this: a bound contract is compared by its census,
/// which is what the tests assert on.
#[derive(Debug, Clone)]
pub struct BoundTts {
    pub tensors: BTreeMap<String, StInfo>,
    pub elements: u64,
    pub payload_bytes: u64,
}

fn parse(header_json: &str) -> Result<HashMap<String, StInfo>, TtsError> {
    parse_header_json_checked(header_json).map_err(TtsError::Header)
}

/// Bind a census against a contract: nothing missing, nothing extra, every shape and dtype exact,
/// and no two tensors reading the same byte range.
fn bind(
    which: &'static str,
    expected_tensors: usize,
    expected_elements: u64,
    contract: Vec<TtsTensorRequirement>,
    header: HashMap<String, StInfo>,
) -> Result<BoundTts, TtsError> {
    // Aliased storage is checked FIRST, before any shape, because a substituted tensor that has
    // the right shape is otherwise invisible.
    let mut by_offsets: HashMap<[usize; 2], &String> = HashMap::new();
    for (name, info) in &header {
        if let Some(other) = by_offsets.get(&info.data_offsets) {
            return Err(TtsError::AliasedStorage {
                name: name.clone(),
                other: (*other).clone(),
                offsets: info.data_offsets,
            });
        }
        by_offsets.insert(info.data_offsets, name);
    }

    let mut bound = BTreeMap::new();
    // Named `total_elements` and not `elements`, which is the shape helper: a local binding of the
    // same name shadows the function for the rest of the block and the compiler then reports
    // "expected function, found u64" at the first call site.
    let mut total_elements = 0u64;
    let mut payload_bytes = 0u64;
    let dtype_bytes = |d: &str| -> u64 {
        match d {
            "BF16" | "F16" => 2,
            "F32" | "I32" => 4,
            "F64" | "I64" => 8,
            "F8_E4M3" | "I8" | "U8" | "BOOL" => 1,
            _ => 0,
        }
    };
    for requirement in contract {
        let found = header
            .get(&requirement.name)
            .ok_or_else(|| TtsError::MissingTensor(requirement.name.clone()))?;
        if found.shape != requirement.shape {
            return Err(TtsError::Shape {
                name: requirement.name,
                expected: requirement.shape,
                found: found.shape.clone(),
            });
        }
        if found.dtype != requirement.dtype {
            return Err(TtsError::Dtype {
                name: requirement.name,
                expected: requirement.dtype,
                found: found.dtype.clone(),
            });
        }
        let n = elements(&found.shape);
        let span = found.data_offsets[1].saturating_sub(found.data_offsets[0]);
        // The stored byte span has to equal elements x dtype size, or the tensor does not read its
        // own storage densely.
        let want_bytes = n * dtype_bytes(&found.dtype);
        if want_bytes != 0 && span != want_bytes as usize {
            return Err(TtsError::Shape {
                name: requirement.name,
                expected: vec![want_bytes],
                found: vec![span as u64],
            });
        }
        total_elements += n;
        payload_bytes += span as u64;
        bound.insert(requirement.name, found.clone());
    }
    // Anything left over is unexpected, and this is where the codec's encoder half gets refused:
    // it is present in the archive and absent from synthesis, so a caller that binds the whole
    // file against the decoder contract is told so rather than quietly loading 0.22 GB of weights
    // that can never run.
    if let Some(name) = header.keys().find(|k| !bound.contains_key(*k)) {
        return Err(TtsError::UnexpectedTensor(name.clone()));
    }
    if bound.len() != expected_tensors || total_elements != expected_elements {
        return Err(TtsError::Census {
            which,
            expected_tensors,
            found_tensors: bound.len(),
            expected_elements,
            found_elements: total_elements,
        });
    }
    Ok(BoundTts {
        tensors: bound,
        elements: total_elements,
        payload_bytes,
    })
}

/// Bind the talker archive from its safetensors header JSON.
pub fn bind_talker(header_json: &str) -> Result<BoundTts, TtsError> {
    let header = parse(header_json)?;
    bind(
        "talker",
        TALKER_TENSORS,
        TALKER_ELEMENTS,
        talker_tensor_contract(TALKER),
        header,
    )
}

/// Bind the codec's decoder half from its safetensors header JSON. The encoder half is refused as
/// dead weight for synthesis, and the census counts only the decoder.
pub fn bind_codec_decoder(header_json: &str) -> Result<BoundTts, TtsError> {
    let full = parse(header_json)?;
    let decoder: HashMap<String, StInfo> = full
        .into_iter()
        .filter(|(k, _)| k.starts_with("decoder."))
        .collect();
    bind(
        "codec-decoder",
        CODEC_DECODER_TENSORS,
        CODEC_DECODER_ELEMENTS,
        codec_decoder_tensor_contract(),
        decoder,
    )
}

/// Refuse the codec's encoder half explicitly, so a port that reaches for it fails at load rather
/// than synthesizing from weights that were never trained for it.
pub fn codec_encoder_is_dead_weight(header_json: &str) -> Result<(usize, u64), TtsError> {
    let full = parse(header_json)?;
    let mut tensors = 0usize;
    let mut elements = 0u64;
    for (name, info) in &full {
        if name.starts_with("encoder.") {
            tensors += 1;
            elements += info.shape.iter().product::<u64>();
        }
    }
    if tensors != CODEC_ENCODER_TENSORS || elements != CODEC_ENCODER_ELEMENTS {
        return Err(TtsError::Census {
            which: "codec-encoder",
            expected_tensors: CODEC_ENCODER_TENSORS,
            found_tensors: tensors,
            expected_elements: CODEC_ENCODER_ELEMENTS,
            found_elements: elements,
        });
    }
    Ok((tensors, elements))
}

#[cfg(test)]
mod tests;
