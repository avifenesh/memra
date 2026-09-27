//! Portable one-codebook oracle for the pinned MiMo bundled audio tokenizer.
//!
//! The source casts post-downsample hidden states to F32 before RVQ. This
//! mirrors `EuclideanCodebook.quantize`, then `layer.decode` and F32 residual
//! subtraction. It deliberately retains the input norm and source expression.

pub const MAX_TOKENS: usize = 256;
pub const MAX_WIDTH: usize = 1_024;
pub const MAX_BINS: usize = 1_024;

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
}
