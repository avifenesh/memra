//! Portable RVQ oracle for the pinned MiMo bundled audio tokenizer.
//!
//! The source casts post-downsample hidden states to F32 before RVQ. This
//! mirrors `EuclideanCodebook.quantize`, then `layer.decode` and F32 residual
//! subtraction. It deliberately retains the input norm and source expression.

pub const MAX_TOKENS: usize = 256;
pub const MAX_WIDTH: usize = 1_024;
pub const MAX_BINS: usize = 1_024;
pub const RVQ_DEPTHS: usize = 20;

use crate::mimo_audio::GroupedAudioCodes;

/// One source `encoder.quantizer.vq.layers.{depth}._codebook.embed` in F32.
pub struct F32Codebook<'a> {
    pub embed: &'a [f32],
    pub bins: usize,
}

#[derive(Debug, PartialEq)]
pub struct Rvq20Result {
    /// Token-major `[T,20]`: ID `(token,depth)` is `code_ids[token*20+depth]`.
    /// Source `torch.stack(all_indices)` is depth-major `[20,T]`, with that
    /// same ID at `source[depth*T+token]`.
    pub code_ids: Vec<u16>,
    pub residual: Vec<f32>,
    pub tokens: usize,
}

/// Pinned per-depth bin schedule. The same schedule is checked by the
/// resident weight accessor before any GPU encode runs.
pub fn expected_bins(depth: usize) -> Option<usize> {
    match depth {
        0 | 1 => Some(1_024),
        2 => Some(256),
        3..=19 => Some(128),
        _ => None,
    }
}

#[derive(Debug, PartialEq)]
pub struct OneCodebookResult {
    pub code_ids: Vec<u32>,
    pub residual: Vec<f32>,
}

/// Source expression: `-(x.pow(2).sum - (2*x) @ embed + embed.pow(2).sum)`.
///
/// The three reductions use scalar F32 accumulation here. A GPU BLAS backend
/// can choose another reduction tree, so near ties need source GPU evidence.
fn source_score(x: &[f32], embed: &[f32], x_norm: f32) -> f32 {
    let mut dot = 0.0f32;
    let mut embed_norm = 0.0f32;
    for (&value, &weight) in x.iter().zip(embed) {
        dot += (2.0 * value) * weight;
        embed_norm += weight * weight;
    }
    -((x_norm - dot) + embed_norm)
}

/// Encode exactly one codebook and subtract its decoded embedding row.
///
/// `residual` is contiguous `[tokens,width]`; `embed` is contiguous
/// `[bins,width]`. The caller binds a specific source depth and its bin count.
pub fn encode_one(
    residual: &[f32],
    tokens: usize,
    width: usize,
    embed: &[f32],
    bins: usize,
) -> Result<OneCodebookResult, String> {
    if !(1..=MAX_TOKENS).contains(&tokens)
        || !(1..=MAX_WIDTH).contains(&width)
        || !(1..=MAX_BINS).contains(&bins)
        || residual.len() != tokens * width
        || embed.len() != bins * width
        || !residual.iter().all(|value| value.is_finite())
        || !embed.iter().all(|value| value.is_finite())
    {
        return Err("MiMo RVQ one-codebook geometry or F32 finiteness changed".into());
    }

    let mut result = OneCodebookResult {
        code_ids: Vec::with_capacity(tokens),
        residual: Vec::with_capacity(residual.len()),
    };
    for x in residual.chunks_exact(width) {
        let mut x_norm = 0.0f32;
        for &value in x {
            x_norm += value * value;
        }
        if !x_norm.is_finite() {
            return Err("MiMo RVQ one-codebook input norm is non-finite".into());
        }
        let mut best_score = f32::NEG_INFINITY;
        let mut best_index = 0usize;
        for (index, row) in embed.chunks_exact(width).enumerate() {
            let score = source_score(x, row, x_norm);
            if !score.is_finite() {
                return Err("MiMo RVQ one-codebook score is non-finite".into());
            }
            // `torch.max(dim=-1).indices` returns the first exact tie.
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        result.code_ids.push(best_index as u32);
        let selected = &embed[best_index * width..(best_index + 1) * width];
        for (&value, &weight) in x.iter().zip(selected) {
            let next = value - weight;
            if !next.is_finite() {
                return Err("MiMo RVQ one-codebook residual is non-finite".into());
            }
            result.residual.push(next);
        }
    }
    Ok(result)
}

/// Source `ResidualVectorQuantization.encode`, fixed to depths `0..20`.
///
/// Input is post-downsample contiguous F32 `[tokens,width]`. All twenty
/// source codebooks are checked before the first residual subtraction.
pub fn encode_20(
    input: &[f32],
    tokens: usize,
    width: usize,
    codebooks: &[F32Codebook<'_>],
) -> Result<Rvq20Result, String> {
    if !(1..=MAX_TOKENS).contains(&tokens)
        || !(1..=MAX_WIDTH).contains(&width)
        || input.len() != tokens * width
        || !input.iter().all(|value| value.is_finite())
        || codebooks.len() != RVQ_DEPTHS
    {
        return Err("MiMo RVQ twenty-depth input or codebook count changed".into());
    }
    for (depth, codebook) in codebooks.iter().enumerate() {
        if expected_bins(depth) != Some(codebook.bins)
            || codebook.embed.len() != codebook.bins * width
            || !codebook.embed.iter().all(|value| value.is_finite())
        {
            return Err(format!("MiMo RVQ depth {depth} F32 codebook changed"));
        }
    }

    let mut residual = input.to_vec();
    let mut code_ids = vec![0u16; tokens * RVQ_DEPTHS];
    for (depth, codebook) in codebooks.iter().enumerate() {
        let step = encode_one(&residual, tokens, width, codebook.embed, codebook.bins)?;
        for (token, &id) in step.code_ids.iter().enumerate() {
            code_ids[token * RVQ_DEPTHS + depth] = id as u16;
        }
        residual = step.residual;
    }
    Ok(Rvq20Result {
        code_ids,
        residual,
        tokens,
    })
}

/// Convert token-major `[T,20]` IDs to the patch encoder's row-major
/// `[groups,4,20]`, repeating the final token row in the last group.
pub fn group_for_patch(code_ids: &[u16], tokens: usize) -> Result<GroupedAudioCodes, String> {
    if !(1..=MAX_TOKENS).contains(&tokens) || code_ids.len() != tokens * RVQ_DEPTHS {
        return Err("MiMo RVQ grouped-code token extent changed".into());
    }
    for (index, &id) in code_ids.iter().enumerate() {
        if usize::from(id) >= expected_bins(index % RVQ_DEPTHS).unwrap() {
            return Err(format!(
                "MiMo RVQ grouped-code ID invalid at element {index}"
            ));
        }
    }
    let groups = tokens.div_ceil(4);
    let mut codes = Vec::with_capacity(groups * 4 * RVQ_DEPTHS);
    codes.extend_from_slice(code_ids);
    let last = &code_ids[(tokens - 1) * RVQ_DEPTHS..tokens * RVQ_DEPTHS];
    for _ in tokens..groups * 4 {
        codes.extend_from_slice(last);
    }
    Ok(GroupedAudioCodes { groups, codes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_codebook_selects_source_scores_then_subtracts_rows() {
        let output = encode_one(
            &[0.875, -0.5, -1.0, 0.25],
            2,
            2,
            &[1.0, -0.25, -1.0, 0.5, 0.0, 0.0],
            3,
        )
        .unwrap();
        assert_eq!(output.code_ids, [0, 1]);
        assert_eq!(output.residual, [-0.125, -0.25, 0.0, -0.25]);
    }

    #[test]
    fn exact_tie_keeps_first_index() {
        let output = encode_one(&[0.0], 1, 1, &[1.0, -1.0], 2).unwrap();
        assert_eq!(output.code_ids, [0]);
        assert_eq!(output.residual, [-1.0]);
    }

    #[test]
    fn dropping_input_norm_changes_f32_argmax_ties() {
        // At this magnitude F32 rounds x_norm - 2 back to x_norm.
        // Omitting x_norm would prefer the second row.
        let output = encode_one(&[10_000.0], 1, 1, &[0.0, 0.0001], 2).unwrap();
        assert_eq!(output.code_ids, [0]);
        assert_eq!(output.residual, [10_000.0]);
        assert!(2.0 * 10_000.0 * 0.0001 > 0.0001f32.powi(2));
    }

    #[test]
    fn squared_distance_can_choose_a_different_code_in_f32() {
        let x = 10_000.0f32;
        let near = 0.0003f32;
        let output = encode_one(&[x], 1, 1, &[0.0, near], 2).unwrap();
        assert_eq!(output.code_ids, [1]);
        // Direct squared distance rounds both differences to the same F32
        // input. That arithmetic would choose code 0 on a first-index tie.
        assert_eq!((x - 0.0).powi(2), (x - near).powi(2));
    }

    #[test]
    fn source_score_keeps_input_norm_and_source_parentheses() {
        let x = [3.0f32, 4.0];
        let row = [1.0f32, 2.0];
        let score = source_score(&x, &row, 25.0);
        assert_eq!(score, -((25.0 - 22.0) + 5.0));
    }

    #[test]
    fn refuses_bounds_nonfinite_and_overflow() {
        assert!(encode_one(&[], 0, 1, &[0.0], 1).is_err());
        assert!(encode_one(&[0.0], 1, 1, &[], 0).is_err());
        assert!(encode_one(&[0.0], 1, 2, &[0.0], 1).is_err());
        assert!(encode_one(&[f32::NAN], 1, 1, &[0.0], 1).is_err());
        assert!(encode_one(&[0.0], 1, 1, &[f32::INFINITY], 1).is_err());
        assert!(encode_one(&[f32::MAX], 1, 1, &[0.0], 1).is_err());
    }

    fn synthetic_codebooks<'a>(embeds: &'a [Vec<f32>]) -> Vec<F32Codebook<'a>> {
        embeds
            .iter()
            .enumerate()
            .map(|(depth, embed)| F32Codebook {
                embed,
                bins: expected_bins(depth).unwrap(),
            })
            .collect()
    }

    #[test]
    fn twenty_depths_preserve_source_order_residual_and_grouped_layout() {
        let mut embeds = (0..RVQ_DEPTHS)
            .map(|depth| vec![0.0; expected_bins(depth).unwrap()])
            .collect::<Vec<_>>();
        embeds[0][1] = 2.0;
        embeds[1][1] = 1.0;
        embeds[2][1] = -0.25;
        let input = [1.75, 0.75];
        let result = encode_20(&input, 2, 1, &synthetic_codebooks(&embeds)).unwrap();
        assert_eq!(result.residual, [0.0, 0.0]);
        assert_eq!(&result.code_ids[..3], &[1, 0, 1]);
        assert_eq!(&result.code_ids[20..23], &[0, 1, 1]);
        assert!(result.code_ids.iter().skip(3).take(17).all(|&id| id == 0));

        let mut source_depth_major = Vec::with_capacity(2 * RVQ_DEPTHS);
        for depth in 0..RVQ_DEPTHS {
            for token in 0..2 {
                source_depth_major.push(result.code_ids[token * RVQ_DEPTHS + depth]);
            }
        }
        assert_eq!(&source_depth_major[..6], &[1, 0, 0, 1, 1, 1]);

        let grouped = group_for_patch(&result.code_ids, result.tokens).unwrap();
        assert_eq!(grouped.groups, 1);
        assert_eq!(&grouped.codes[..40], &result.code_ids);
        assert_eq!(&grouped.codes[40..60], &result.code_ids[20..40]);
        assert_eq!(&grouped.codes[60..80], &result.code_ids[20..40]);

        embeds[0][1] = 1.0;
        embeds[1][1] = 2.0;
        let reordered = encode_20(&input, 2, 1, &synthetic_codebooks(&embeds)).unwrap();
        assert_ne!(reordered.code_ids, result.code_ids);
        assert_ne!(reordered.residual, result.residual);
    }

    #[test]
    fn tie_at_first_depth_changes_the_next_depth_input() {
        let mut embeds = (0..RVQ_DEPTHS)
            .map(|depth| vec![0.0; expected_bins(depth).unwrap()])
            .collect::<Vec<_>>();
        embeds[0][0] = 1.0;
        embeds[0][1] = -1.0;
        embeds[0][2..].fill(3.0);
        embeds[1][1] = -1.0;
        let result = encode_20(&[0.0], 1, 1, &synthetic_codebooks(&embeds)).unwrap();
        assert_eq!(&result.code_ids[..2], &[0, 1]);
        assert_eq!(result.residual, [0.0]);
    }

    #[test]
    fn twenty_depth_final_residual_is_not_forced_to_zero() {
        let mut embeds = (0..RVQ_DEPTHS)
            .map(|depth| vec![0.0; expected_bins(depth).unwrap()])
            .collect::<Vec<_>>();
        embeds[0][1] = 1.0;
        let result = encode_20(&[0.875], 1, 1, &synthetic_codebooks(&embeds)).unwrap();
        assert_eq!(result.code_ids[0], 1);
        assert!(result.code_ids[1..].iter().all(|&id| id == 0));
        assert_eq!(result.residual, [-0.125]);
    }

    #[test]
    fn patch_grouping_repeats_last_row_across_group_boundary() {
        let mut ids = vec![0u16; 5 * RVQ_DEPTHS];
        for token in 0..5 {
            ids[token * RVQ_DEPTHS] = token as u16;
        }
        let grouped = group_for_patch(&ids, 5).unwrap();
        assert_eq!(grouped.groups, 2);
        assert_eq!(&grouped.codes[..5 * RVQ_DEPTHS], &ids);
        for row in 5..8 {
            assert_eq!(
                &grouped.codes[row * RVQ_DEPTHS..(row + 1) * RVQ_DEPTHS],
                &ids[4 * RVQ_DEPTHS..5 * RVQ_DEPTHS]
            );
        }
    }

    #[test]
    fn late_bad_codebook_and_grouped_ids_are_refused() {
        let mut embeds = (0..RVQ_DEPTHS)
            .map(|depth| vec![0.0; expected_bins(depth).unwrap()])
            .collect::<Vec<_>>();
        embeds[19][0] = f32::NAN;
        assert!(encode_20(&[0.0], 1, 1, &synthetic_codebooks(&embeds)).is_err());
        embeds[19] = vec![0.0; 127];
        assert!(encode_20(&[0.0], 1, 1, &synthetic_codebooks(&embeds)).is_err());
        embeds[19].push(0.0);
        let mut codebooks = synthetic_codebooks(&embeds);
        codebooks[19].bins = 256;
        assert!(encode_20(&[0.0], 1, 1, &codebooks).is_err());
        assert!(group_for_patch(&[0; 19], 1).is_err());
        let mut ids = vec![0; RVQ_DEPTHS];
        ids[19] = 128;
        assert!(group_for_patch(&ids, 1).is_err());
    }
}
