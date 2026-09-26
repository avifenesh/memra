//! Checkpoint tensor schema for the pinned MiMo V2.6 audio weights.
//!
//! Sources: XiaomiMiMo/MiMo-V2.6-Flash-RL at
//! 3b38d063180c3e4aed9691fdc735f3d10b266ee4 and
//! tiyuvta/MiMo-V2.6-Flash-RL-NVFP4 at
//! 58edbd0c60ace653512b8f423bf41813331ef9e3. Their audio headers agree:
//! 75 `audio_encoder` and 20 `speech_embeddings` BF16 tensors. The 16,384-wide
//! projection MLP and BF16 storage are checkpoint facts, not fields of
//! `MiMoAudioConfig`. Callers must bind these rows against a full HF census.
//! This sidecar does not establish an executable audio path or model support.

use crate::config::MiMoAudioConfig;
use crate::tensor_contract::{
    FloatType, QuantConstraint, TensorContractError, TensorId, TensorMatch, TensorOwner,
    TensorRequirement, TensorTransform,
};

const PROJECTION_INTERMEDIATE: u64 = 16_384;

fn pinned_geometry(audio: &MiMoAudioConfig, text_hidden_size: u32) -> bool {
    text_hidden_size == 4_096
        && audio.add_post_norm
        && audio.audio_channels == 20
        && audio.audio_segment_size == 6_000
        && audio.group_size == 4
        && audio.input_full_attention
        && audio.input_local_attn_heads == 16
        && audio.input_local_dim == 1_024
        && audio.input_local_head_dim == 64
        && audio.input_local_hidden_dropout.to_bits() == 0.0f32.to_bits()
        && audio.input_local_intermediate_size == 4_096
        && audio.input_local_layers == 6
        && audio.out_hidden_size == 4_096
        && audio.partial_rotary_factor.to_bits() == 1.0f32.to_bits()
        && audio.projection_layers == 2
        && audio.rope_theta.to_bits() == 640_000.0f32.to_bits()
        && audio.speech_vocab_size == 1_280
        && audio.speech_zeroemb_idx == 1_024
}

fn row(name: String, shape: Vec<u64>) -> TensorRequirement {
    TensorRequirement {
        id: TensorId::Family {
            family: "mimo_v2_audio",
            key: name.clone(),
        },
        names: vec![name],
        match_mode: TensorMatch::OneOf,
        shape,
        // TensorOwner has no audio tower variant. These tensors are outside
        // text layers and belong to the top-level checkpoint.
        owner: TensorOwner::Global,
        transform: TensorTransform::Identity,
        quant: QuantConstraint::ExactFloat(FloatType::Bf16),
        auxiliaries: None,
        required: true,
    }
}

/// Return the exact HF safetensors audio rows of the two pinned revisions.
///
/// `text_hidden_size` is the enclosing model's hidden size. Every audio
/// config field is checked because this schema contains checkpoint-derived
/// constants that are not safe to extrapolate to another geometry.
#[allow(clippy::result_large_err)]
#[allow(dead_code)] // consumed by the subsequent full MiMo mint inspector, never a load path
pub(crate) fn pinned_audio_requirements(
    audio: &MiMoAudioConfig,
    text_hidden_size: u32,
) -> Result<Vec<TensorRequirement>, TensorContractError> {
    if !pinned_geometry(audio, text_hidden_size) {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "MiMo V2.6 audio config differs from pinned source and mint revisions",
        });
    }

    let h = u64::from(audio.input_local_dim);
    let f = u64::from(audio.input_local_intermediate_size);
    let out = u64::from(audio.out_hidden_size);
    let vocab = u64::from(audio.speech_vocab_size);
    let mut rows = Vec::with_capacity(75 + audio.audio_channels as usize);

    for layer in 0..audio.input_local_layers {
        let prefix = format!("audio_encoder.input_local_transformer.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![h]),
            ("mlp.down_proj.weight", vec![h, f]),
            ("mlp.gate_proj.weight", vec![f, h]),
            ("mlp.up_proj.weight", vec![f, h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("self_attn.k_proj.bias", vec![h]),
            ("self_attn.k_proj.weight", vec![h, h]),
            ("self_attn.o_proj.weight", vec![h, h]),
            ("self_attn.q_proj.bias", vec![h]),
            ("self_attn.q_proj.weight", vec![h, h]),
            ("self_attn.v_proj.bias", vec![h]),
            ("self_attn.v_proj.weight", vec![h, h]),
        ] {
            rows.push(row(format!("{prefix}.{suffix}"), shape));
        }
    }
    rows.push(row(
        "audio_encoder.input_local_transformer.norm.weight".to_owned(),
        vec![h],
    ));
    rows.push(row(
        "audio_encoder.projection.mlp.0.weight".to_owned(),
        vec![PROJECTION_INTERMEDIATE, out],
    ));
    rows.push(row(
        "audio_encoder.projection.mlp.2.weight".to_owned(),
        vec![out, PROJECTION_INTERMEDIATE],
    ));

    for channel in 0..audio.audio_channels {
        rows.push(row(
            format!("speech_embeddings.{channel}.weight"),
            vec![vocab, h],
        ));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use crate::tensor_contract::{
        CheckpointDialect, StorageLayout, TensorCensusEntry, TensorContract,
    };
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;

    fn pinned_config() -> (MiMoAudioConfig, u32) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        (config.mimo.unwrap().audio_config.unwrap(), config.n_embd)
    }

    fn census(rows: &[TensorRequirement]) -> Vec<TensorCensusEntry> {
        rows.iter()
            .map(|r| TensorCensusEntry {
                name: r.names[0].clone(),
                shape: r.shape.clone(),
                storage: StorageLayout::Float(FloatType::Bf16),
                physical_bytes: r.shape.iter().product::<u64>() * 2,
            })
            .collect()
    }

    #[test]
    fn pinned_rows_cover_each_named_audio_tensor_once() {
        let (audio, text_hidden_size) = pinned_config();
        let rows = pinned_audio_requirements(&audio, text_hidden_size).unwrap();
        assert_eq!(rows.len(), 95);
        assert_eq!(
            rows.iter()
                .filter(|r| r.names[0].starts_with("audio_encoder."))
                .count(),
            75
        );
        assert_eq!(
            rows.iter()
                .filter(|r| r.names[0].starts_with("speech_embeddings."))
                .count(),
            20
        );
        let names: BTreeSet<_> = rows.iter().map(|r| r.names[0].as_str()).collect();
        let ids: BTreeSet<_> = rows.iter().map(|r| &r.id).collect();
        assert_eq!(names.len(), 95);
        assert_eq!(ids.len(), 95);
        for r in &rows {
            assert_eq!(
                r.id,
                TensorId::Family {
                    family: "mimo_v2_audio",
                    key: r.names[0].clone(),
                }
            );
            assert_eq!(r.match_mode, TensorMatch::OneOf);
            assert_eq!(r.owner, TensorOwner::Global);
            assert_eq!(r.transform, TensorTransform::Identity);
            assert_eq!(r.quant, QuantConstraint::ExactFloat(FloatType::Bf16));
            assert_eq!(r.auxiliaries, None);
            assert!(r.required);
        }
        // Digest of sorted "name\0dtype\0comma-separated-shape\n" records
        // from both pinned revisions' 95 audio headers.
        let mut sorted = rows.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|r| &r.names[0]);
        let mut manifest = String::new();
        for r in sorted {
            manifest.push_str(&r.names[0]);
            manifest.push('\0');
            manifest.push_str("BF16");
            manifest.push('\0');
            manifest.push_str(
                &r.shape
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            manifest.push('\n');
        }
        assert_eq!(
            format!("{:x}", Sha256::digest(manifest.as_bytes())),
            "6101c0a9b1be56671cdd3548c8bd51fe16ae9584edefa6864f0d9104f055d4f6"
        );
    }

    #[test]
    fn changed_config_fails_closed() {
        let (audio, text_hidden_size) = pinned_config();
        assert!(pinned_audio_requirements(&audio, text_hidden_size + 1).is_err());
        let mut changed = audio.clone();
        changed.input_local_layers += 1;
        assert!(pinned_audio_requirements(&changed, text_hidden_size).is_err());
        let mut changed = audio.clone();
        changed.projection_layers += 1;
        assert!(pinned_audio_requirements(&changed, text_hidden_size).is_err());
        let mut changed = audio.clone();
        changed.speech_vocab_size += 1;
        assert!(pinned_audio_requirements(&changed, text_hidden_size).is_err());
        let mut changed = audio.clone();
        changed.rope_theta += 1.0;
        assert!(pinned_audio_requirements(&changed, text_hidden_size).is_err());
    }

    #[test]
    fn strict_binding_rejects_missing_extra_shape_and_dtype() {
        let (audio, text_hidden_size) = pinned_config();
        let rows = pinned_audio_requirements(&audio, text_hidden_size).unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: rows.clone(),
        };
        let complete = census(&rows);
        assert_eq!(contract.bind(&complete).unwrap().tensors.len(), 95);

        let mut missing = complete.clone();
        missing.remove(0);
        assert!(matches!(
            contract.bind(&missing),
            Err(TensorContractError::Missing { .. })
        ));

        let mut extra = complete.clone();
        extra.push(TensorCensusEntry {
            name: "speech_embeddings.20.weight".to_owned(),
            shape: vec![1_280, 1_024],
            storage: StorageLayout::Float(FloatType::Bf16),
            physical_bytes: 2_621_440,
        });
        assert!(matches!(
            contract.bind(&extra),
            Err(TensorContractError::Extra { .. })
        ));

        let mut wrong_shape = complete.clone();
        wrong_shape[0].shape[0] += 1;
        assert!(matches!(
            contract.bind(&wrong_shape),
            Err(TensorContractError::ShapeMismatch { .. })
        ));

        let mut wrong_dtype = complete;
        wrong_dtype[0].storage = StorageLayout::Float(FloatType::F16);
        assert!(matches!(
            contract.bind(&wrong_dtype),
            Err(TensorContractError::QuantLayoutMismatch { .. })
        ));
    }
}
