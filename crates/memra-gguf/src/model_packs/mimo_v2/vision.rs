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

use crate::config::{Arch, MiMoVisionConfig, ModelConfig};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiMoPatchOrder {
    Row,
    Column,
}

/// Source ViT attention, distinct from the text mixer's causal sink denominator.
/// `sink_first_key` adds a learned score bias to the first patch in each image;
/// a masked first patch remains masked. Q/K already carry two-axis RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiMoVisionAttentionPlan {
    pub layer: u32,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub symmetric_window: Option<usize>,
    pub sink_first_key: bool,
    pub patch_order: MiMoPatchOrder,
}

/// One source `grid_thw` row after temporal pairs and 16x16 spatial patches
/// have been formed. Height and width count patch rows, not pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiMoVisionGrid {
    pub frames: u32,
    pub height: u32,
    pub width: u32,
}

/// The source ViT's patch order and two-axis position IDs, before RoPE
/// frequencies, QKV projection, and any vision execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiMoVisionLayout {
    /// Row-order `(height, width)` position of each projected patch. Every
    /// four consecutive patches form one 2x2 spatial merge unit.
    pub row_positions: Vec<(u32, u32)>,
    /// Cumulative patch ends for attention. A frame never attends to another
    /// frame or image in the publisher's vision tower.
    pub frame_ends: Vec<u32>,
    /// Selection indices at four-patch merge-unit granularity: column-order
    /// unit `i` reads row-order unit `column_groups[i]`.
    pub column_groups: Vec<u32>,
    /// Selection indices that restore row order after a column-order block.
    pub reverse_column_groups: Vec<u32>,
    pub output_tokens: u32,
}

/// Bounds a single diagnostic layout before materializing its indices.
/// This is not an image/video decoder or a serving request admission limit.
const MAX_LAYOUT_PATCHES: usize = 50_000;

/// Reproduce `rot_pos_emb`, `get_window_index_1d(col=True)`, and the per-frame
/// `cu_seqlens` of the pinned publisher source. The source image processor
/// must provide already patchified pixel rows in four-patch merge-unit order.
pub fn pinned_vision_layout(
    config: &ModelConfig,
    grids: &[MiMoVisionGrid],
) -> Result<MiMoVisionLayout, &'static str> {
    let vision = config
        .mimo
        .as_ref()
        .and_then(|mimo| mimo.vision_config.as_ref())
        .ok_or("MiMo vision config is missing")?;
    if config.arch != Arch::MiMoV2 || !pinned_geometry(vision, config.n_embd) {
        return Err("MiMo vision geometry differs from the pinned source");
    }
    if grids.is_empty() {
        return Err("MiMo vision grid is empty");
    }
    let mut total_patches = 0usize;
    let mut total_frames = 0usize;
    for grid in grids {
        if grid.frames == 0 || grid.height == 0 || grid.width == 0 {
            return Err("MiMo vision grid has a zero extent");
        }
        if grid.height % 2 != 0 || grid.width % 2 != 0 {
            return Err("MiMo vision grid is not divisible by the 2x2 merger");
        }
        let patches = (grid.frames as usize)
            .checked_mul(grid.height as usize)
            .and_then(|value| value.checked_mul(grid.width as usize))
            .ok_or("MiMo vision patch count overflow")?;
        total_patches = total_patches
            .checked_add(patches)
            .ok_or("MiMo vision total patch count overflow")?;
        total_frames = total_frames
            .checked_add(grid.frames as usize)
            .ok_or("MiMo vision frame count overflow")?;
    }
    if total_patches > MAX_LAYOUT_PATCHES {
        return Err("MiMo vision diagnostic layout exceeds 50000 patches");
    }
    let total_patches_u32 =
        u32::try_from(total_patches).map_err(|_| "MiMo vision patch count exceeds u32")?;
    let total_groups = total_patches / 4;
    let mut row_positions = Vec::new();
    let mut frame_ends = Vec::new();
    let mut column_groups = Vec::new();
    row_positions
        .try_reserve_exact(total_patches)
        .map_err(|_| "MiMo vision position allocation failed")?;
    frame_ends
        .try_reserve_exact(total_frames)
        .map_err(|_| "MiMo vision frame allocation failed")?;
    column_groups
        .try_reserve_exact(total_groups)
        .map_err(|_| "MiMo vision column allocation failed")?;

    let mut patch_end = 0u32;
    let mut group_start = 0u32;
    for grid in grids {
        let group_h = grid.height / 2;
        let group_w = grid.width / 2;
        let frame_patches = grid.height * grid.width;
        for _ in 0..grid.frames {
            for group_row in 0..group_h {
                for group_col in 0..group_w {
                    for sub_row in 0..2 {
                        for sub_col in 0..2 {
                            row_positions.push((group_row * 2 + sub_row, group_col * 2 + sub_col));
                        }
                    }
                }
            }
            // `torch.arange(...).reshape(t, group_h, group_w).transpose(1,2)`
            // selects complete merge units by column, then by row.
            for group_col in 0..group_w {
                for group_row in 0..group_h {
                    column_groups.push(group_start + group_row * group_w + group_col);
                }
            }
            patch_end += frame_patches;
            group_start += group_h * group_w;
            frame_ends.push(patch_end);
        }
    }
    if patch_end != total_patches_u32
        || row_positions.len() != total_patches
        || column_groups.len() != total_groups
    {
        return Err("MiMo vision layout bookkeeping differs from grid");
    }
    let mut reverse_column_groups = vec![0; total_groups];
    for (column, &row) in column_groups.iter().enumerate() {
        reverse_column_groups[row as usize] = column as u32;
    }
    Ok(MiMoVisionLayout {
        row_positions,
        frame_ends,
        column_groups,
        reverse_column_groups,
        output_tokens: total_patches_u32 / 4,
    })
}

pub fn pinned_attention_plan(
    config: &ModelConfig,
    layer: u32,
) -> Result<MiMoVisionAttentionPlan, &'static str> {
    let vision = config
        .mimo
        .as_ref()
        .and_then(|mimo| mimo.vision_config.as_ref())
        .ok_or("MiMo vision config is missing")?;
    if config.arch != Arch::MiMoV2 || !pinned_geometry(vision, config.n_embd) {
        return Err("MiMo vision geometry differs from the pinned source");
    }
    if layer >= vision.depth {
        return Err("MiMo vision layer is out of range");
    }
    let global = FULL_ATTENTION_BLOCKS.contains(&layer);
    Ok(MiMoVisionAttentionPlan {
        layer,
        query_heads: 32,
        kv_heads: 8,
        head_dim: 64,
        symmetric_window: (!global).then_some(64),
        sink_first_key: !global,
        patch_order: if vision.vit_window_attn_types[layer as usize] == 1 {
            MiMoPatchOrder::Column
        } else {
            MiMoPatchOrder::Row
        },
    })
}

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

    #[test]
    fn publisher_grid_layout_keeps_merge_units_and_frame_boundaries() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        let layout = pinned_vision_layout(
            &config,
            &[
                MiMoVisionGrid {
                    frames: 2,
                    height: 4,
                    width: 6,
                },
                MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 4,
                },
            ],
        )
        .unwrap();
        assert_eq!(layout.frame_ends, [24, 48, 56]);
        assert_eq!(layout.output_tokens, 14);
        assert_eq!(
            layout.row_positions[0..8],
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
            ]
        );
        assert_eq!(
            layout.row_positions[12..16],
            [(2, 0), (2, 1), (3, 0), (3, 1)]
        );
        assert_eq!(layout.row_positions[0..24], layout.row_positions[24..48]);
        assert_eq!(
            layout.column_groups,
            [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, 12, 13]
        );
        assert_eq!(
            layout.reverse_column_groups,
            [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11, 12, 13]
        );
        for (column, &row) in layout.column_groups.iter().enumerate() {
            assert_eq!(layout.reverse_column_groups[row as usize], column as u32);
        }
    }

    #[test]
    fn publisher_grid_layout_refuses_invalid_grid_and_plan() {
        let mut config =
            ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        assert!(pinned_vision_layout(&config, &[]).is_err());
        for grid in [
            MiMoVisionGrid {
                frames: 0,
                height: 2,
                width: 2,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 3,
                width: 2,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 1,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 226,
                width: 226,
            },
        ] {
            assert!(pinned_vision_layout(&config, &[grid]).is_err());
        }
        config
            .mimo
            .as_mut()
            .unwrap()
            .vision_config
            .as_mut()
            .unwrap()
            .patch_size = 14;
        assert!(
            pinned_vision_layout(
                &config,
                &[MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .is_err()
        );
    }

    #[test]
    fn pinned_attention_program_tracks_full_window_and_patch_order() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!("fixtures/config.json")));
        for layer in 0..28 {
            let plan = pinned_attention_plan(&config, layer).unwrap();
            let global = [0, 9, 18, 27].contains(&layer);
            assert_eq!(plan.query_heads, 32);
            assert_eq!(plan.kv_heads, 8);
            assert_eq!(plan.head_dim, 64);
            assert_eq!(plan.symmetric_window, (!global).then_some(64));
            assert_eq!(plan.sink_first_key, !global);
            assert_eq!(
                plan.patch_order,
                if [5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26].contains(&layer) {
                    MiMoPatchOrder::Column
                } else {
                    MiMoPatchOrder::Row
                }
            );
        }
        assert!(pinned_attention_plan(&config, 28).is_err());
        let mut wrong = config;
        wrong
            .mimo
            .as_mut()
            .unwrap()
            .vision_config
            .as_mut()
            .unwrap()
            .visual_token_window_size = 32;
        assert!(pinned_attention_plan(&wrong, 1).is_err());
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
