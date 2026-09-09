//! Whisper large-v3 metadata pack. Native speech execution is still unsupported.
//!
//! `PACK` accepts the checkpoint's speech config directly, without routing it through the
//! text-only `ModelConfig` defaults. CLI speech onboarding is a follow-up to this skeleton.
//! Source bytes stay F32. FP16 conversion is a later explicit, receipted artifact step.

use super::{Gate, NativeSupport};
use crate::config::JsonObj;
use crate::model_plan::speech::*;
use crate::model_plan::{ModelPlan, NormKind, NormPlan, WeightTransform};
use crate::safetensors::{self, StInfo, StModel};
use crate::tensor_contract::*;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

pub const SOURCE_REPOSITORY: &str = "ivrit-ai/whisper-large-v3";
pub const SOURCE_REVISION: &str = "766847c9795b3b5cc0d42f8476199c711d5cee21";
pub const PACK: WhisperPack = WhisperPack;

/// Speech has no chat template. Tokenizer, language/task prefix and generation policy replace it.
pub struct WhisperPack;

impl WhisperPack {
    pub const FAMILY: &'static str = "whisper";
    pub const ALIASES: &'static [&'static str] = &["whisper", "whisper-large-v3"];
    pub const SUPPORT: Option<NativeSupport> = None;
    pub const GATES: &'static [Gate] = &[
        Gate::Config,
        Gate::TokenizerTemplate,
        Gate::TensorCensus,
        Gate::TinyParity,
        Gate::CheckpointParity,
        Gate::RewriteParity,
        Gate::Serve,
    ];

    /// Deliberately accept only the audited large-v3 geometry for the first executable step.
    /// Turbo and trained successors need their own config/census receipts before admission.
    pub fn compile_plan(&self, config: &str, frontend: &str) -> Result<ModelPlan, String> {
        let c = JsonObj::parse(config);
        let f = JsonObj::parse(frontend);
        for (key, value) in [
            ("model_type", "\"whisper\""),
            ("activation_function", "\"gelu\""),
            ("is_encoder_decoder", "true"),
            ("scale_embedding", "false"),
            ("torch_dtype", "\"float32\""),
            ("d_model", "1280"),
            ("encoder_layers", "32"),
            ("decoder_layers", "32"),
            ("encoder_attention_heads", "20"),
            ("decoder_attention_heads", "20"),
            ("encoder_ffn_dim", "5120"),
            ("decoder_ffn_dim", "5120"),
            ("max_source_positions", "1500"),
            ("max_target_positions", "448"),
            ("num_mel_bins", "128"),
            ("vocab_size", "51866"),
            ("decoder_start_token_id", "50258"),
            ("eos_token_id", "50257"),
        ] {
            if c.raw(key) != Some(value) {
                return Err(format!(
                    "unsupported Whisper config {key}: expected {value}"
                ));
            }
        }
        // These absent HF defaults are semantic, not permission to accept a changed graph.
        if c.raw("tie_word_embeddings").is_some_and(|v| v != "true")
            || c.raw("layer_norm_eps")
                .is_some_and(|v| v.parse::<f32>().ok() != Some(1e-5))
        {
            return Err("unsupported Whisper output tying or LayerNorm epsilon".into());
        }
        for (key, value) in [
            ("feature_extractor_type", "\"WhisperFeatureExtractor\""),
            ("feature_size", "128"),
            ("sampling_rate", "16000"),
            ("n_fft", "400"),
            ("hop_length", "160"),
            ("n_samples", "480000"),
            ("nb_max_frames", "3000"),
            ("chunk_length", "30"),
            ("padding_side", "\"right\""),
            ("padding_value", "0.0"),
        ] {
            if f.raw(key) != Some(value) {
                return Err(format!(
                    "unsupported Whisper frontend {key}: expected {value}"
                ));
            }
        }
        Ok(WhisperPlan {
            frontend: LogMelPlan {
                sample_rate: 16000,
                fft_size: 400,
                hop_length: 160,
                mel_bins: 128,
                max_samples: 480000,
                max_frames: 3000,
                periodic_hann: true,
                centered_reflect_padding: true,
                drop_last_stft_frame: true,
                slaney_mel_filters: true,
                power: 2,
                log10_floor: 1e-10,
                dynamic_range: 8.0,
                normalization_offset: 4.0,
                normalization_divisor: 4.0,
            },
            conv_stem: [
                AudioConv1dPlan {
                    input_channels: 128,
                    output_channels: 1280,
                    kernel: 3,
                    stride: 1,
                    padding: 1,
                    bias: true,
                },
                AudioConv1dPlan {
                    input_channels: 1280,
                    output_channels: 1280,
                    kernel: 3,
                    stride: 2,
                    padding: 1,
                    bias: true,
                },
            ],
            hidden_size: 1280,
            encoder_layers: 32,
            decoder_layers: 32,
            encoder_heads: 20,
            decoder_heads: 20,
            encoder_ffn: 5120,
            decoder_ffn: 5120,
            source_positions: 1500,
            target_positions: 448,
            vocab_size: 51866,
            norm: NormPlan {
                kind: NormKind::LayerNorm,
                epsilon: 1e-5,
                weight_transform: WeightTransform::Identity,
            },
            activation: SpeechActivation::GeluErf,
            encoder_position: AudioPositionKind::CheckpointSinusoidal,
            decoder_position: AudioPositionKind::LearnedAbsolute,
            attention: WhisperAttentionPlan {
                encoder_bidirectional: true,
                decoder_causal: true,
                cross_attention: true,
                query_bias: true,
                key_bias: false,
                value_bias: true,
                output_bias: true,
                qk_scale: 0.125,
                pre_norm_serial_residual: true,
                tied_output_embedding: true,
            },
            state: WhisperStatePlan {
                encoder_recomputed_per_utterance: true,
                decoder_self_kv_per_layer: true,
                encoder_cross_kv_per_decoder_layer: true,
                max_decoder_tokens: 448,
            },
            decode: AsrDecodePlan {
                beam_size: 1,
                deterministic: true,
                start_token: 50258,
                language_token: 50279,
                task_token: 50360,
                no_timestamps_token: 50364,
                eos_token: 50257,
                generation_policy_qualified: false,
            },
        }
        .into_model_plan())
    }
}

#[allow(clippy::result_large_err)] // allow: keep the shared tensor contract diagnostic type
pub fn tensor_contract(
    plan: &WhisperPlan,
    dialect: CheckpointDialect,
) -> Result<TensorContract, TensorContractError> {
    if dialect != CheckpointDialect::HfSafetensors {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "Whisper safetensors source required",
        });
    }
    let h = u64::from(plan.hidden_size);
    let mut requirements = Vec::new();
    let mut add = |name: String, shape: Vec<u64>| {
        requirements.push(TensorRequirement {
            id: TensorId::Family {
                family: "whisper",
                key: name.strip_prefix("model.").unwrap().to_owned(),
            },
            names: vec![name],
            match_mode: TensorMatch::OneOf,
            shape,
            owner: TensorOwner::Global,
            transform: TensorTransform::Identity,
            quant: QuantConstraint::ExactFloat(FloatType::F32),
            auxiliaries: None,
            required: true,
        });
    };
    for (i, conv) in plan.conv_stem.iter().enumerate() {
        add(
            format!("model.encoder.conv{}.weight", i + 1),
            vec![
                u64::from(conv.output_channels),
                u64::from(conv.input_channels),
                u64::from(conv.kernel),
            ],
        );
        add(
            format!("model.encoder.conv{}.bias", i + 1),
            vec![u64::from(conv.output_channels)],
        );
    }
    add(
        "model.encoder.embed_positions.weight".into(),
        vec![u64::from(plan.source_positions), h],
    );
    add(
        "model.decoder.embed_positions.weight".into(),
        vec![u64::from(plan.target_positions), h],
    );
    add(
        "model.decoder.embed_tokens.weight".into(),
        vec![u64::from(plan.vocab_size), h],
    );
    // proj_out is tied to decoder.embed_tokens and is absent from this exact artifact.
    for (tower, layers, ffn) in [
        ("encoder", plan.encoder_layers, plan.encoder_ffn),
        ("decoder", plan.decoder_layers, plan.decoder_ffn),
    ] {
        for suffix in ["weight", "bias"] {
            add(format!("model.{tower}.layer_norm.{suffix}"), vec![h]);
        }
        for layer in 0..layers {
            let p = format!("model.{tower}.layers.{layer}");
            let attentions: &[&str] = if tower == "decoder" {
                &["self_attn", "encoder_attn"]
            } else {
                &["self_attn"]
            };
            for attn in attentions {
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    add(format!("{p}.{attn}.{proj}.weight"), vec![h, h]);
                    if proj != "k_proj" {
                        add(format!("{p}.{attn}.{proj}.bias"), vec![h]);
                    }
                }
                for suffix in ["weight", "bias"] {
                    add(format!("{p}.{attn}_layer_norm.{suffix}"), vec![h]);
                }
            }
            for suffix in ["weight", "bias"] {
                add(format!("{p}.final_layer_norm.{suffix}"), vec![h]);
            }
            add(format!("{p}.fc1.weight"), vec![u64::from(ffn), h]);
            add(format!("{p}.fc1.bias"), vec![u64::from(ffn)]);
            add(format!("{p}.fc2.weight"), vec![h, u64::from(ffn)]);
            add(format!("{p}.fc2.bias"), vec![h]);
        }
    }
    Ok(TensorContract {
        dialect,
        requirements,
    })
}

pub struct WhisperCheckpoint {
    pub plan: ModelPlan,
    pub tensors: BoundTensorContract,
    /// Mapped source bytes, never converted or uploaded by this metadata loader.
    pub weights: StModel,
}

impl WhisperCheckpoint {
    pub fn open(directory: &Path) -> Result<Self, String> {
        let config =
            std::fs::read_to_string(directory.join("config.json")).map_err(|e| e.to_string())?;
        let frontend = std::fs::read_to_string(directory.join("preprocessor_config.json"))
            .map_err(|e| e.to_string())?;
        let plan = PACK.compile_plan(&config, &frontend)?;
        let weights = StModel::open(directory).map_err(|e| e.to_string())?;
        let census = weights
            .names()
            .map(|name| census_entry(name, weights.info(name).unwrap()))
            .collect::<Result<Vec<_>, _>>()?;
        let tensors = bind(&plan, &census)?;
        Ok(Self {
            plan,
            tensors,
            weights,
        })
    }
}

fn census_entry(name: &str, info: &StInfo) -> Result<TensorCensusEntry, String> {
    if info.dtype != "F32" {
        return Err(format!(
            "Whisper source tensor {name} must be F32, got {}",
            info.dtype
        ));
    }
    Ok(TensorCensusEntry {
        name: name.into(),
        shape: info.shape.clone(),
        storage: StorageLayout::Float(FloatType::F32),
        physical_bytes: (info.data_offsets[1] - info.data_offsets[0]) as u64,
    })
}

fn bind(plan: &ModelPlan, census: &[TensorCensusEntry]) -> Result<BoundTensorContract, String> {
    TensorContract::for_plan(
        plan,
        CheckpointDialect::HfSafetensors,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .and_then(|c| c.bind(census))
    .map_err(|e| e.to_string())
}

/// Inspect actual safetensors range headers without manufacturing a multi-GB weight file.
/// `file_size` is the full shard size from the pinned artifact manifest, not header length.
/// This validates structure/census only; it cannot authenticate absent weight bytes.
pub struct ShardMetadata<'a> {
    pub name: &'a str,
    pub header: &'a [u8],
    pub file_size: u64,
}

pub fn inspect_metadata(
    config: &str,
    frontend: &str,
    index: &str,
    shards: &[ShardMetadata<'_>],
) -> Result<(ModelPlan, BoundTensorContract), String> {
    let plan = PACK.compile_plan(config, frontend)?;
    let weight_map = safetensors::parse_index_weight_map_json_checked(index)?;
    let mut seen = BTreeMap::new();
    let mut census = Vec::new();
    for shard in shards {
        let prefix: [u8; 8] = shard
            .header
            .get(..8)
            .ok_or("truncated safetensors prefix")?
            .try_into()
            .unwrap();
        let len = u64::from_le_bytes(prefix);
        if len > 100_000_000 || len.checked_add(8) != Some(shard.header.len() as u64) {
            return Err("invalid safetensors header length".into());
        }
        let payload = shard
            .file_size
            .checked_sub(len + 8)
            .ok_or("shard smaller than header")?;
        let payload = usize::try_from(payload).map_err(|e| e.to_string())?;
        let json = std::str::from_utf8(&shard.header[8..]).map_err(|e| e.to_string())?;
        let infos: HashMap<String, StInfo> = safetensors::parse_header_json_checked(json)?;
        safetensors::validate_tensor_extents(&infos, payload).map_err(|e| e.to_string())?;
        for (name, info) in infos {
            if weight_map.get(&name).map(String::as_str) != Some(shard.name) {
                return Err(format!("index/shard mismatch for {name}"));
            }
            if seen.insert(name.clone(), shard.name).is_some() {
                return Err(format!("duplicate physical tensor {name}"));
            }
            census.push(census_entry(&name, &info)?);
        }
    }
    if seen.len() != weight_map.len() {
        return Err("index names missing from supplied headers".into());
    }
    let tensors = bind(&plan, &census)?;
    Ok((plan, tensors))
}

#[cfg(test)]
mod tests;
