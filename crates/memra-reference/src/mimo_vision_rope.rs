//! Axial MiMo vision RoPE after QKV projection, in the patch order of one
//! pinned vision block. This portable BF16-rounded path is an operator oracle.

use memra_gguf::model_packs::mimo_v2::vision::{
    MiMoPatchOrder, MiMoVisionAttentionPlan, MiMoVisionLayout,
};

const Q_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const HALF: usize = HEAD_DIM / 2;
const MAX_PATCHES: usize = 1_024;

fn check_plan(plan: &MiMoVisionAttentionPlan) -> Result<(), &'static str> {
    let column = [5, 6, 7, 8, 14, 15, 16, 17, 23, 24, 25, 26].contains(&plan.layer);
    if plan.layer >= 28
        || plan.query_heads != Q_HEADS
        || plan.kv_heads != KV_HEADS
        || plan.head_dim != HEAD_DIM
        || (plan.patch_order == MiMoPatchOrder::Column) != column
    {
        return Err("MiMo axial RoPE plan differs from pinned vision blocks");
    }
    Ok(())
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

/// Return source `(height,width)` positions in the current block's row or
/// column patch order. Each column selection moves an intact 2x2 merge unit.
pub fn ordered_positions(
    plan: &MiMoVisionAttentionPlan,
    layout: &MiMoVisionLayout,
) -> Result<Vec<(u32, u32)>, &'static str> {
    check_plan(plan)?;
    let patches = layout.row_positions.len();
    let units = patches / 4;
    if patches == 0
        || patches > MAX_PATCHES
        || !patches.is_multiple_of(4)
        || layout.output_tokens as usize != units
        || layout.column_groups.len() != units
        || layout.reverse_column_groups.len() != units
        || layout.frame_ends.is_empty()
        || layout.frame_ends.last().copied() != Some(patches as u32)
    {
        return Err("MiMo axial RoPE layout differs from bounded patch geometry");
    }
    let mut previous = 0;
    for &end in &layout.frame_ends {
        if end <= previous || (end - previous) as usize > 256 || !(end - previous).is_multiple_of(4)
        {
            return Err("MiMo axial RoPE frame boundaries differ from vision attention");
        }
        previous = end;
    }
    let mut seen = vec![false; units];
    for (column, &row) in layout.column_groups.iter().enumerate() {
        let row = row as usize;
        if row >= units || seen[row] || layout.reverse_column_groups[row] as usize != column {
            return Err("MiMo axial RoPE column selection is not a permutation");
        }
        seen[row] = true;
    }
    if plan.patch_order == MiMoPatchOrder::Row {
        return Ok(layout.row_positions.clone());
    }
    let mut positions = Vec::with_capacity(patches);
    for &unit in &layout.column_groups {
        let first = unit as usize * 4;
        positions.extend_from_slice(&layout.row_positions[first..first + 4]);
    }
    Ok(positions)
}

/// Cosine and sine for the first 32 dimensions of each head, in the block's
/// patch order. The second 32 dimensions reuse the same phase values.
pub fn phase_rows(
    plan: &MiMoVisionAttentionPlan,
    layout: &MiMoVisionLayout,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let positions = ordered_positions(plan, layout)?;
    let mut cos = Vec::with_capacity(positions.len() * HALF);
    let mut sin = Vec::with_capacity(positions.len() * HALF);
    for &(height, width) in &positions {
        for dimension in 0..HALF {
            let position = if dimension < 16 { height } else { width };
            let frequency = 10_000.0f32.powf(-((dimension % 16) as f32) / 16.0);
            let angle = position as f32 * frequency;
            let (s, c) = angle.sin_cos();
            cos.push(c);
            sin.push(s);
        }
    }
    Ok((cos, sin))
}

/// Apply source two-axis RoPE to `[patch,head,64]` Q/K and return BF16-rounded
/// f32 values. The first 16 half-pairs use height, the next 16 width.
pub fn rotate_qk(
    plan: &MiMoVisionAttentionPlan,
    layout: &MiMoVisionLayout,
    q: &[f32],
    k: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (cos, sin) = phase_rows(plan, layout)?;
    let patches = cos.len() / HALF;
    if q.len() != patches * Q_HEADS * HEAD_DIM
        || k.len() != patches * KV_HEADS * HEAD_DIM
        || q.iter().chain(k).any(|value| !value.is_finite())
    {
        return Err("MiMo axial RoPE Q/K extent or values differ from source".into());
    }
    let mut q_out = vec![0.0f32; q.len()];
    let mut k_out = vec![0.0f32; k.len()];
    for patch in 0..patches {
        for dimension in 0..HALF {
            let cosine = cos[patch * HALF + dimension];
            let sine = sin[patch * HALF + dimension];
            for (heads, input, output) in [(Q_HEADS, q, &mut q_out), (KV_HEADS, k, &mut k_out)] {
                for head in 0..heads {
                    let base = (patch * heads + head) * HEAD_DIM;
                    let first = input[base + dimension];
                    let second = input[base + dimension + HALF];
                    output[base + dimension] = round_bf16(first * cosine - second * sine);
                    output[base + dimension + HALF] = round_bf16(second * cosine + first * sine);
                }
            }
        }
    }
    if q_out.iter().chain(&k_out).any(|value| !value.is_finite()) {
        return Err("MiMo axial RoPE output is non-finite".into());
    }
    Ok((q_out, k_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::vision::{
        MiMoVisionGrid, pinned_attention_plan, pinned_vision_layout,
    };

    fn config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        )))
    }

    #[test]
    fn height_and_width_rope_axes_stay_separate() {
        let config = config();
        let layout = pinned_vision_layout(
            &config,
            &[MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 2,
            }],
        )
        .unwrap();
        let plan = pinned_attention_plan(&config, 0).unwrap();
        let mut q = vec![0.0f32; 4 * Q_HEADS * HEAD_DIM];
        let mut k = vec![0.0f32; 4 * KV_HEADS * HEAD_DIM];
        let patch2 = 2 * Q_HEADS * HEAD_DIM;
        q[patch2] = 1.0;
        q[patch2 + 16] = 1.0;
        let patch3 = 3 * KV_HEADS * HEAD_DIM;
        k[patch3] = 1.0;
        let (q, k) = rotate_qk(&plan, &layout, &q, &k).unwrap();
        assert!((q[patch2] - 1.0f32.cos()).abs() < 0.005);
        assert!((q[patch2 + HALF] - 1.0f32.sin()).abs() < 0.005);
        assert_eq!(q[patch2 + 16], 1.0);
        assert_eq!(q[patch2 + 16 + HALF], 0.0);
        assert!((k[patch3] - 1.0f32.cos()).abs() < 0.005);
    }

    #[test]
    fn column_order_moves_complete_merge_units_before_rope() {
        let config = config();
        let layout = pinned_vision_layout(
            &config,
            &[MiMoVisionGrid {
                frames: 1,
                height: 4,
                width: 6,
            }],
        )
        .unwrap();
        let plan = pinned_attention_plan(&config, 5).unwrap();
        let positions = ordered_positions(&plan, &layout).unwrap();
        assert_eq!(positions[0..4], layout.row_positions[0..4]);
        assert_eq!(positions[4..8], layout.row_positions[12..16]);
        assert_eq!(positions[8..12], layout.row_positions[4..8]);
        let mut broken = layout.clone();
        broken.column_groups[1] = broken.column_groups[0];
        assert!(ordered_positions(&plan, &broken).is_err());
    }
}
