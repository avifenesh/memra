//! Portable MiMo ViT attention arithmetic over already projected, rotated Q/K/V.
//! The image patcher, axial RoPE, block MLP, and merger have separate gates.

use memra_gguf::model_packs::mimo_v2::vision::MiMoVisionAttentionPlan;

use crate::speech::encoder::gelu_erf;

fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

/// Portable publisher merger over BF16-valued rows. Small widths let tests
/// check the arithmetic; the GPU path separately admits only 1280 -> 5120
/// -> 4096. The three missing checkpoint biases are exactly zero.
pub fn patch_merger_bf16(
    input: &[f32],
    context_dim: usize,
    output_dim: usize,
    norm_weight: &[f32],
    mlp_0: &[f32],
    mlp_2: &[f32],
) -> Result<Vec<f32>, String> {
    let merged_width = context_dim
        .checked_mul(4)
        .ok_or("MiMo merger width overflows")?;
    let patches = input.len().checked_div(context_dim).unwrap_or(0);
    let tokens = patches / 4;
    if context_dim == 0
        || context_dim > 1_280
        || output_dim == 0
        || output_dim > 4_096
        || !(4..=1_024).contains(&patches)
        || !patches.is_multiple_of(4)
        || patches.checked_mul(context_dim) != Some(input.len())
        || norm_weight.len() != context_dim
        || merged_width.checked_mul(merged_width) != Some(mlp_0.len())
        || merged_width.checked_mul(output_dim) != Some(mlp_2.len())
    {
        return Err("MiMo merger reference extent differs from bounded rows".into());
    }
    if input
        .iter()
        .chain(norm_weight)
        .chain(mlp_0)
        .chain(mlp_2)
        .any(|value| !value.is_finite() || (value.to_bits() & 0xffff) != 0)
    {
        return Err("MiMo merger reference needs finite BF16-valued operands".into());
    }
    let mut normalized = vec![0.0f32; input.len()];
    for row in 0..patches {
        let values = &input[row * context_dim..(row + 1) * context_dim];
        let mean = values.iter().copied().sum::<f32>() / context_dim as f32;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / context_dim as f32;
        let inverse = (variance + 1e-6).sqrt().recip();
        for col in 0..context_dim {
            normalized[row * context_dim + col] =
                bf16((values[col] - mean) * inverse * norm_weight[col]);
        }
    }
    let mut activated = vec![0.0f32; tokens * merged_width];
    for token in 0..tokens {
        for out in 0..merged_width {
            let mut sum = 0.0f32;
            for col in 0..merged_width {
                sum = normalized[token * merged_width + col]
                    .mul_add(mlp_0[out * merged_width + col], sum);
            }
            activated[token * merged_width + out] = bf16(gelu_erf(bf16(sum)));
        }
    }
    let mut output = vec![0.0f32; tokens * output_dim];
    for token in 0..tokens {
        for out in 0..output_dim {
            let mut sum = 0.0f32;
            for col in 0..merged_width {
                sum = activated[token * merged_width + col]
                    .mul_add(mlp_2[out * merged_width + col], sum);
            }
            output[token * output_dim + out] = bf16(sum);
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("MiMo merger reference output is non-finite".into());
    }
    Ok(output)
}

/// Source ViT's column walk over 2x2 spatial merge blocks. Each block keeps
/// its four patch rows together; the source restores row order before block 27.
pub fn column_patch_indices(grids: &[[usize; 3]]) -> Result<Vec<usize>, String> {
    if grids.is_empty() {
        return Err("MiMo vision needs at least one patch grid".into());
    }
    let mut indices = Vec::new();
    let mut offset = 0usize;
    for &[frames, height, width] in grids {
        if frames == 0
            || height == 0
            || width == 0
            || !height.is_multiple_of(2)
            || !width.is_multiple_of(2)
        {
            return Err("MiMo vision grids need positive frames and even spatial patches".into());
        }
        let rows = height / 2;
        let columns = width / 2;
        let blocks_per_frame = rows
            .checked_mul(columns)
            .ok_or("MiMo vision block grid overflows")?;
        let patch_count = frames
            .checked_mul(height)
            .and_then(|count| count.checked_mul(width))
            .ok_or("MiMo vision patch grid overflows")?;
        let next_offset = offset
            .checked_add(patch_count)
            .ok_or("MiMo vision combined patch grids overflow")?;
        for frame in 0..frames {
            for column in 0..columns {
                for row in 0..rows {
                    let block = frame * blocks_per_frame + row * columns + column;
                    let start = offset + block * 4;
                    indices.extend(start..start + 4);
                }
            }
        }
        offset = next_offset;
    }
    Ok(indices)
}

/// Inputs and output are `[patch, head, dim]`; K/V use `kv_heads`.
/// `lengths` splits independent images or frames. Window visibility is
/// symmetric, and the learned sink biases key zero within each split.
pub fn preprojected_attention(
    plan: &MiMoVisionAttentionPlan,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lengths: &[usize],
    sink: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    if plan.query_heads != 32 || plan.kv_heads != 8 || plan.head_dim != 64 || plan.layer >= 28 {
        return Err("MiMo vision reference requires pinned attention geometry".into());
    }
    let global = [0, 9, 18, 27].contains(&plan.layer);
    if plan.symmetric_window != (!global).then_some(64)
        || plan.sink_first_key != !global
        || sink.is_some() != plan.sink_first_key
    {
        return Err("MiMo vision window or first-key sink differs from pinned source".into());
    }
    if lengths.is_empty() || lengths.contains(&0) {
        return Err("MiMo vision reference needs nonempty image lengths".into());
    }
    let patches = lengths
        .iter()
        .try_fold(0usize, |total, &length| total.checked_add(length))
        .ok_or("MiMo vision patch count overflows")?;
    let q_width = plan.query_heads * plan.head_dim;
    let kv_width = plan.kv_heads * plan.head_dim;
    let q_elements = patches
        .checked_mul(q_width)
        .ok_or("MiMo vision query extent overflows")?;
    let kv_elements = patches
        .checked_mul(kv_width)
        .ok_or("MiMo vision KV extent overflows")?;
    if q.len() != q_elements
        || k.len() != kv_elements
        || v.len() != kv_elements
        || sink.is_some_and(|bias| bias.len() != plan.query_heads)
        || q.iter().chain(k).chain(v).any(|value| !value.is_finite())
        || sink.is_some_and(|bias| bias.iter().any(|value| !value.is_finite()))
    {
        return Err("MiMo vision Q/K/V or sink extent is invalid".into());
    }

    let mut output = vec![0.0f32; q_elements];
    let group = plan.query_heads / plan.kv_heads;
    let scale = 1.0f32 / (plan.head_dim as f32).sqrt();
    let mut offset = 0;
    for &length in lengths {
        for query in 0..length {
            for head in 0..plan.query_heads {
                let kv_head = head / group;
                let q_start = (offset + query) * q_width + head * plan.head_dim;
                let q_row = &q[q_start..q_start + plan.head_dim];
                let mut scores = Vec::with_capacity(length);
                for key in 0..length {
                    if plan
                        .symmetric_window
                        .is_some_and(|window| query.abs_diff(key) > window)
                    {
                        continue;
                    }
                    let k_start = (offset + key) * kv_width + kv_head * plan.head_dim;
                    let k_row = &k[k_start..k_start + plan.head_dim];
                    let dot: f32 = q_row.iter().zip(k_row).map(|(a, b)| a * b).sum();
                    let bias = if key == 0 {
                        sink.map_or(0.0, |values| values[head])
                    } else {
                        0.0
                    };
                    let score = dot * scale + bias;
                    if !score.is_finite() {
                        return Err("MiMo vision attention score is not finite".into());
                    }
                    scores.push((key, score));
                }
                let max = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let normalizer: f32 = scores.iter().map(|(_, score)| (score - max).exp()).sum();
                if !normalizer.is_finite() || normalizer <= 0.0 {
                    return Err("MiMo vision attention softmax is invalid".into());
                }
                for (key, score) in scores {
                    let weight = (score - max).exp() / normalizer;
                    let v_start = (offset + key) * kv_width + kv_head * plan.head_dim;
                    let v_row = &v[v_start..v_start + plan.head_dim];
                    let out_row = &mut output[q_start..q_start + plan.head_dim];
                    for (out, value) in out_row.iter_mut().zip(v_row) {
                        *out += weight * value;
                    }
                }
            }
        }
        offset += length;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("MiMo vision attention output is not finite".into());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::vision::pinned_attention_plan;

    fn plan(layer: u32) -> MiMoVisionAttentionPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        )));
        pinned_attention_plan(&config, layer).unwrap()
    }

    fn planes(values: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut v = vec![0.0f32; values.len() * 8 * 64];
        for (patch, &value) in values.iter().enumerate() {
            for head in 0..8 {
                v[(patch * 8 + head) * 64] = value;
            }
        }
        (vec![0.0; values.len() * 32 * 64], vec![0.0; v.len()], v)
    }

    #[test]
    fn merger_uses_layernorm_erf_gelu_zero_bias_and_four_row_groups() {
        let input = [
            1.0, 3.0, 4.0, 0.0, 2.0, 2.0, -1.0, 5.0, // first merge unit
            4.0, 0.0, 1.0, 3.0, 2.0, 2.0, -1.0, 5.0, // second merge unit
        ];
        let mut first = vec![0.0; 8 * 8];
        for channel in 0..8 {
            first[channel * 8 + channel] = 1.0;
        }
        let mut second = vec![0.0; 2 * 8];
        second[0] = 1.0;
        second[8 + 2] = 1.0;
        let output = patch_merger_bf16(&input, 2, 2, &[3.0, 1.0], &first, &second).unwrap();
        assert_eq!(
            output,
            [
                bf16(gelu_erf(-3.0)),
                bf16(gelu_erf(3.0)),
                bf16(gelu_erf(3.0)),
                bf16(gelu_erf(-3.0)),
            ]
        );
        assert!(output[0] < 0.0); // RMSNorm would keep the first row positive.
        assert_eq!(output[0].to_bits(), 0xbb85_0000); // tanh GELU gives 0xbb6e_0000.
    }

    #[test]
    fn merger_rounds_first_linear_and_erf_gelu_before_final_linear() {
        let input = [0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mut first = vec![0.0f32; 8 * 8];
        first[0] = 1.710_937_5;
        let mut second = vec![0.0f32; 8];
        second[0] = 3.828_125;
        let output = patch_merger_bf16(&input, 2, 1, &[2.1875, 1.0], &first, &second).unwrap();
        let product = -2.1875 * 1.710_937_5;
        let first = bf16(product);
        let activated = bf16(gelu_erf(first));
        let expected = bf16(activated * second[0]);
        let skipped_linear_round = bf16(bf16(gelu_erf(product)) * second[0]);
        let skipped_gelu_round = bf16(gelu_erf(first) * second[0]);
        assert_ne!(expected, skipped_linear_round);
        assert_ne!(expected, skipped_gelu_round);
        assert_eq!(output, [expected]);
    }

    #[test]
    fn merger_rejects_bad_extents_nonfinite_and_unrounded_operands() {
        let mut first = vec![0.0; 8 * 8];
        first[0] = 1.0;
        let mut input = [0.0f32; 8];
        assert!(patch_merger_bf16(&input, 2, 1, &[1.0, 1.0], &first, &[0.0; 8]).is_ok());
        assert!(patch_merger_bf16(&input[..7], 2, 1, &[1.0, 1.0], &first, &[0.0; 8]).is_err());
        assert!(patch_merger_bf16(&input, 2, 1, &[1.0], &first, &[0.0; 8]).is_err());
        assert!(patch_merger_bf16(&input, 2, 1, &[1.0, 1.0], &first[..7], &[0.0; 8]).is_err());
        input[0] = f32::NAN;
        assert!(patch_merger_bf16(&input, 2, 1, &[1.0, 1.0], &first, &[0.0; 8]).is_err());
        input[0] = 1.001;
        assert!(patch_merger_bf16(&input, 2, 1, &[1.0, 1.0], &first, &[0.0; 8]).is_err());
    }

    #[test]
    fn symmetric_window_masks_first_key_before_adding_sink() {
        let values: Vec<f32> = (0..66).map(|index| index as f32).collect();
        let (q, k, v) = planes(&values);
        let sink = vec![2.0f32.ln(); 32];
        let local = preprojected_attention(&plan(1), &q, &k, &v, &[66], Some(&sink)).unwrap();
        let full = preprojected_attention(&plan(0), &q, &k, &v, &[66], None).unwrap();
        assert!((local[0] - (2080.0 / 66.0)).abs() < 1e-4);
        assert!((local[65 * 32 * 64] - 33.0).abs() < 1e-4);
        assert!((full[65 * 32 * 64] - 32.5).abs() < 1e-4);
    }

    #[test]
    fn first_key_sink_restarts_at_each_image_boundary() {
        let (q, k, v) = planes(&[0.0, 3.0, 10.0, 13.0]);
        let sink = vec![2.0f32.ln(); 32];
        let output = preprojected_attention(&plan(1), &q, &k, &v, &[2, 2], Some(&sink)).unwrap();
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[2 * 32 * 64] - 11.0).abs() < 1e-6);
        assert!(preprojected_attention(&plan(1), &q, &k, &v, &[2, 2], None).is_err());
        assert!(preprojected_attention(&plan(1), &q, &k, &v, &[4, 1], Some(&sink)).is_err());
    }

    #[test]
    fn column_order_moves_complete_merge_blocks_and_preserves_frame_boundaries() {
        let indices = column_patch_indices(&[[1, 4, 6], [1, 2, 2]]).unwrap();
        let blocks = [0, 3, 1, 4, 2, 5];
        let mut expected = Vec::new();
        for block in blocks {
            expected.extend(block * 4..block * 4 + 4);
        }
        expected.extend(24..28);
        assert_eq!(indices, expected);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..28).collect::<Vec<_>>());
        assert!(column_patch_indices(&[[1, 3, 4]]).is_err());
    }
}
