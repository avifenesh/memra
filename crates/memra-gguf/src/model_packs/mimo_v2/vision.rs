//! Checkpoint tensor schema for the pinned MiMo V2.6 vision tower.
//!
//! Sources: XiaomiMiMo/MiMo-V2.6-Flash-RL at
//! 3b38d063180c3e4aed9691fdc735f3d10b266ee4 and
//! tiyuvta/MiMo-V2.6-Flash-RL-NVFP4 at
//! 58edbd0c60ace653512b8f423bf41813331ef9e3. Both revisions carry the
//! same 364 `visual.*` BF16 tensor headers. The asymmetric fused-QKV and
//! attention-output widths, merger widths, and BF16 storage are checkpoint
//! facts, not general properties of `MiMoVisionConfig`. Callers must bind
//! these rows against a full HF census. This sidecar does not establish an
//! executable vision path or model support.

use crate::config::MiMoVisionConfig;
use crate::tensor_contract::{
    FloatType, QuantConstraint, TensorContractError, TensorId, TensorMatch, TensorOwner,
    TensorRequirement, TensorTransform, VisionTensor,
};

const FULL_ATTENTION_BLOCKS: [u32; 4] = [0, 9, 18, 27];
const WINDOW_ATTENTION_TYPES: [i32; 28] = [
    -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1,
];
const FUSED_QKV_ROWS: u64 = 3_072;
const ATTENTION_PROJECTION_INPUT: u64 = 2_048;
const MERGER_WIDTH: u64 = 5_120;

fn pinned_geometry(vision: &MiMoVisionConfig, text_hidden_size: u32) -> bool {
    text_hidden_size == 4_096
        && vision.depth == 28
        && vision.fullatt_block_indexes == FULL_ATTENTION_BLOCKS
        && vision.hidden_act == "silu"
        && vision.hidden_size == 1_280
        && vision.in_chans == 3
        && vision.intermediate_size == 4_608
        && vision.num_heads == 32
        && vision.num_key_value_heads == 8
        && vision.num_query_groups == 4
        && vision.out_hidden_size == 4_096
        && vision.patch_size == 16
        && vision.spatial_merge_size == 2
        && vision.spatial_patch_size == 16
        && vision.temporal_patch_size == 2
        && vision.tokens_per_second == 2
        && vision.use_sink
        && vision.visual_token_window_size == 64
        && vision.vit_window_attn_types == WINDOW_ATTENTION_TYPES
        && vision.window_size == 128
}

fn row(id: TensorId, name: String, shape: Vec<u64>, layer: Option<u32>) -> TensorRequirement {
    TensorRequirement {
        id,
        names: vec![name],
        match_mode: TensorMatch::OneOf,
        shape,
        owner: TensorOwner::Vision(layer),
        transform: TensorTransform::Identity,
        quant: QuantConstraint::ExactFloat(FloatType::Bf16),
        auxiliaries: None,
        required: true,
    }
}

fn typed_row(
    tensor: VisionTensor,
    name: String,
    shape: Vec<u64>,
    layer: Option<u32>,
) -> TensorRequirement {
    row(TensorId::Vision { layer, tensor }, name, shape, layer)
}

fn family_row(name: String, shape: Vec<u64>, layer: Option<u32>) -> TensorRequirement {
    row(
        TensorId::Family {
            family: "mimo_v2_vision",
            key: name.clone(),
        },
        name,
        shape,
        layer,
    )
}

/// Return the exact `visual.*` HF safetensors rows of the two pinned revisions.
///
/// `text_hidden_size` is the enclosing model's hidden size. Every vision
/// config field is checked because these checkpoint-derived shapes and names
/// must not be extrapolated to a different geometry or attention pattern.
#[allow(clippy::result_large_err)]
pub(crate) fn pinned_vision_requirements(
    vision: &MiMoVisionConfig,
    text_hidden_size: u32,
) -> Result<Vec<TensorRequirement>, TensorContractError> {
    if !pinned_geometry(vision, text_hidden_size) {
        return Err(TensorContractError::UnsupportedPlanOperation {
            operation: "MiMo V2.6 vision config differs from pinned source and mint revisions",
        });
    }

    let h = u64::from(vision.hidden_size);
    let ff = u64::from(vision.intermediate_size);
    let out = u64::from(vision.out_hidden_size);
    let mut rows = Vec::with_capacity(364);
    for layer in 0..vision.depth {
        let prefix = format!("visual.blocks.{layer}");
        let l = Some(layer);
        for (tensor, suffix, shape) in [
            (VisionTensor::AttentionOutputBias, "attn.proj.bias", vec![h]),
            (
                VisionTensor::AttentionOutput,
                "attn.proj.weight",
                vec![h, ATTENTION_PROJECTION_INPUT],
            ),
            (
                VisionTensor::FusedQkvBias,
                "attn.qkv.bias",
                vec![FUSED_QKV_ROWS],
            ),
            (
                VisionTensor::FusedQkv,
                "attn.qkv.weight",
                vec![FUSED_QKV_ROWS, h],
            ),
            (VisionTensor::MlpDownBias, "mlp.down_proj.bias", vec![h]),
            (VisionTensor::MlpDown, "mlp.down_proj.weight", vec![h, ff]),
            (VisionTensor::MlpGateBias, "mlp.gate_proj.bias", vec![ff]),
            (VisionTensor::MlpGate, "mlp.gate_proj.weight", vec![ff, h]),
            (VisionTensor::MlpUpBias, "mlp.up_proj.bias", vec![ff]),
            (VisionTensor::MlpUp, "mlp.up_proj.weight", vec![ff, h]),
            (VisionTensor::InputNorm, "norm1.weight", vec![h]),
            (VisionTensor::PreMlpNorm, "norm2.weight", vec![h]),
        ] {
            rows.push(typed_row(tensor, format!("{prefix}.{suffix}"), shape, l));
        }
        if !FULL_ATTENTION_BLOCKS.contains(&layer) {
            // There is no VisionTensor::AttentionSink. Preserve one semantic
            // identity per physical sink without inventing a generic ID.
            rows.push(family_row(format!("{prefix}.attn.sinks"), vec![32], l));
        }
    }
    rows.push(family_row(
        "visual.merger.ln_q.weight".to_owned(),
        vec![h],
        None,
    ));
    rows.push(family_row(
        "visual.merger.mlp.0.weight".to_owned(),
        vec![MERGER_WIDTH, MERGER_WIDTH],
        None,
    ));
    rows.push(family_row(
        "visual.merger.mlp.2.weight".to_owned(),
        vec![out, MERGER_WIDTH],
        None,
    ));
    rows.push(typed_row(
        VisionTensor::PatchProjection,
        "visual.patch_embed.proj.weight".to_owned(),
        vec![
            h,
            u64::from(vision.in_chans),
            u64::from(vision.temporal_patch_size),
            u64::from(vision.spatial_patch_size),
            u64::from(vision.spatial_patch_size),
        ],
        None,
    ));
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

    fn pinned_config() -> (MiMoVisionConfig, u32) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        (config.mimo.unwrap().vision_config.unwrap(), config.n_embd)
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
    fn pinned_rows_cover_each_visual_tensor_once() {
        let (vision, text_hidden_size) = pinned_config();
        let rows = pinned_vision_requirements(&vision, text_hidden_size).unwrap();
        assert_eq!(rows.len(), 364);
        assert_eq!(
            rows.iter()
                .filter(|r| r.names[0].ends_with(".attn.sinks"))
                .count(),
            24
        );
        let names: BTreeSet<_> = rows.iter().map(|r| r.names[0].as_str()).collect();
        let ids: BTreeSet<_> = rows.iter().map(|r| &r.id).collect();
        assert_eq!(names.len(), 364);
        assert_eq!(ids.len(), 364);
        for r in &rows {
            assert_eq!(r.names.len(), 1);
            assert!(r.names[0].starts_with("visual."));
            assert_eq!(r.match_mode, TensorMatch::OneOf);
            assert_eq!(r.transform, TensorTransform::Identity);
            assert_eq!(r.quant, QuantConstraint::ExactFloat(FloatType::Bf16));
            assert_eq!(r.auxiliaries, None);
            assert!(r.required);
            let block = r.names[0]
                .strip_prefix("visual.blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|index| index.parse::<u32>().ok());
            assert_eq!(r.owner, TensorOwner::Vision(block));
        }
        // Sorted "name\0dtype\0comma-separated-shape\n" records from both
        // pinned revisions' 364 visual headers.
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
            "5cee93652d5cbb1939085700819b1fdef75740cb6a7080cd5d6f7a6c6187572d"
        );
    }

    #[test]
    fn changed_config_fails_closed() {
        let (vision, text_hidden_size) = pinned_config();
        assert!(pinned_vision_requirements(&vision, text_hidden_size + 1).is_err());
        macro_rules! changed {
            ($field:ident, $value:expr) => {{
                let mut altered = vision.clone();
                altered.$field = $value;
                assert!(
                    pinned_vision_requirements(&altered, text_hidden_size).is_err(),
                    "accepted changed {}",
                    stringify!($field)
                );
            }};
        }
        changed!(depth, 29);
        changed!(fullatt_block_indexes, vec![0, 9, 18]);
        changed!(hidden_act, "gelu".to_owned());
        changed!(hidden_size, 1_281);
        changed!(in_chans, 4);
        changed!(intermediate_size, 4_609);
        changed!(num_heads, 16);
        changed!(num_key_value_heads, 4);
        changed!(num_query_groups, 2);
        changed!(out_hidden_size, 4_097);
        changed!(patch_size, 14);
        changed!(spatial_merge_size, 1);
        changed!(spatial_patch_size, 14);
        changed!(temporal_patch_size, 1);
        changed!(tokens_per_second, 1);
        changed!(use_sink, false);
        changed!(visual_token_window_size, 32);
        let mut window_types = vision.vit_window_attn_types.clone();
        window_types[5] = 0;
        changed!(vit_window_attn_types, window_types);
        changed!(window_size, 64);
    }

    #[test]
    fn strict_binding_rejects_missing_extra_shape_and_dtype() {
        let (vision, text_hidden_size) = pinned_config();
        let rows = pinned_vision_requirements(&vision, text_hidden_size).unwrap();
        let contract = TensorContract {
            dialect: CheckpointDialect::HfSafetensors,
            requirements: rows.clone(),
        };
        let complete = census(&rows);
        assert_eq!(contract.bind(&complete).unwrap().tensors.len(), 364);

        let mut missing = complete.clone();
        missing.remove(0);
        assert!(matches!(
            contract.bind(&missing),
            Err(TensorContractError::Missing { .. })
        ));

        let mut extra = complete.clone();
        extra.push(TensorCensusEntry {
            name: "visual.blocks.0.attn.sinks".to_owned(),
            shape: vec![32],
            storage: StorageLayout::Float(FloatType::Bf16),
            physical_bytes: 64,
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
