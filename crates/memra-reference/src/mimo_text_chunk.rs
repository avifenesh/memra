//! Portable oracle for a fresh MiMo text attention chunk.
//!
//! Q and K are already RoPE-rotated. V already includes the source's 0.707
//! factor. All rows belong to one sequence starting at position zero. This
//! does not read a preceding KV cache or write a future one.

use memra_gguf::model_plan::{
    AttentionPlan, AttentionScale, TensorPresence, ValueNorm, ValueProjection,
};

pub const MAX_CHUNK: usize = 256;
pub const QUERY_HEADS: usize = 64;
pub const QK_DIM: usize = 192;
pub const VALUE_DIM: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkShape {
    pub kv_heads: usize,
    pub window: usize,
}

pub fn validate_request(
    attention: &AttentionPlan,
    chunk: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    sink_len: Option<usize>,
) -> Result<ChunkShape, &'static str> {
    let (plan, window) = match attention {
        AttentionPlan::Full(plan) => (plan, 0),
        AttentionPlan::SlidingWindow { attention, window } => (attention, *window as usize),
        _ => return Err("MiMo chunk requires full or sliding text attention"),
    };
    let math = plan
        .mimo_math
        .ok_or("MiMo chunk attention lacks family math")?;
    if math.fused_qkv_checkpoint_shards != Some(4)
        || math.value_scale_before_cache.to_bits() != 0.707f32.to_bits()
        || plan.query_heads as usize != QUERY_HEADS
        || plan.key_head_dim as usize != QK_DIM
        || plan.value_head_dim as usize != VALUE_DIM
        || plan.qk_norm != TensorPresence::Absent
        || plan.scale != AttentionScale::InverseSqrtKeyDim
        || plan.value_projection != ValueProjection::Separate
        || plan.value_norm != ValueNorm::None
    {
        return Err("MiMo chunk differs from pinned text attention geometry");
    }
    let kv_heads = match (plan.kv_heads, window, math.sink, sink_len) {
        (4, 0, TensorPresence::Absent, None) => 4,
        (8, 128, TensorPresence::Required, Some(QUERY_HEADS)) => 8,
        _ => return Err("MiMo chunk KV heads, window, or denominator sink mismatch"),
    };
    if !(1..=MAX_CHUNK).contains(&chunk) {
        return Err("MiMo chunk accepts 1..=256 fresh positions");
    }
    if query_len != chunk * QUERY_HEADS * QK_DIM
        || key_len != chunk * kv_heads * QK_DIM
        || value_len != chunk * kv_heads * VALUE_DIM
    {
        return Err("MiMo chunk Q/K/V extent mismatch");
    }
    Ok(ChunkShape { kv_heads, window })
}

/// Independent two-pass softmax oracle with a causal mask and optional SWA128.
pub fn preprojected_attention(
    attention: &AttentionPlan,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sink: Option<&[f32]>,
    chunk: usize,
) -> Result<Vec<f32>, String> {
    let shape = validate_request(
        attention,
        chunk,
        query.len(),
        key.len(),
        value.len(),
        sink.map(<[f32]>::len),
    )?;
    if query
        .iter()
        .chain(key)
        .chain(value)
        .chain(sink.into_iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err("MiMo chunk input is non-finite".into());
    }
    let scale = 1.0 / (QK_DIM as f32).sqrt();
    let mut output = vec![0.0; chunk * QUERY_HEADS * VALUE_DIM];
    for position in 0..chunk {
        let first = if shape.window == 0 {
            0
        } else {
            (position + 1).saturating_sub(shape.window)
        };
        for head in 0..QUERY_HEADS {
            let kv_head = head / (QUERY_HEADS / shape.kv_heads);
            let q_base = (position * QUERY_HEADS + head) * QK_DIM;
            let q_row = &query[q_base..q_base + QK_DIM];
            let mut scores = Vec::with_capacity(position + 1 - first);
            for token in first..=position {
                let k_base = (token * shape.kv_heads + kv_head) * QK_DIM;
                let dot: f32 = q_row
                    .iter()
                    .zip(&key[k_base..k_base + QK_DIM])
                    .map(|(&q, &k)| q * k)
                    .sum();
                let score = dot * scale;
                if !score.is_finite() {
                    return Err("MiMo chunk score is non-finite".into());
                }
                scores.push(score);
            }
            let max = scores
                .iter()
                .copied()
                .chain(sink.map(|s| s[head]))
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores.iter().map(|score| (score - max).exp()).sum::<f32>()
                + sink.map_or(0.0, |s| (s[head] - max).exp());
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err("MiMo chunk denominator is invalid".into());
            }
            let out_base = (position * QUERY_HEADS + head) * VALUE_DIM;
            for (offset, &score) in scores.iter().enumerate() {
                let v_base = ((first + offset) * shape.kv_heads + kv_head) * VALUE_DIM;
                let weight = (score - max).exp() / denominator;
                for dim in 0..VALUE_DIM {
                    output[out_base + dim] += weight * value[v_base + dim];
                }
            }
            if output[out_base..out_base + VALUE_DIM]
                .iter()
                .any(|x| !x.is_finite())
            {
                return Err("MiMo chunk output is non-finite".into());
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_plan::ModelPlan;

    fn plans() -> (AttentionPlan, AttentionPlan) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        )));
        let plan = ModelPlan::compile(&config).unwrap();
        (
            plan.layers[0].attention.clone(),
            plan.layers[1].attention.clone(),
        )
    }

    #[test]
    fn pinned_modes_and_bounds() {
        let (global, local) = plans();
        assert_eq!(
            validate_request(&global, 1, 64 * 192, 4 * 192, 4 * 128, None),
            Ok(ChunkShape {
                kv_heads: 4,
                window: 0
            })
        );
        assert_eq!(
            validate_request(&local, 1, 64 * 192, 8 * 192, 8 * 128, Some(64)),
            Ok(ChunkShape {
                kv_heads: 8,
                window: 128
            })
        );
        assert!(validate_request(&global, 0, 0, 0, 0, None).is_err());
        assert!(validate_request(&global, 257, 0, 0, 0, None).is_err());
        assert!(validate_request(&global, 1, 64 * 192, 4 * 192, 4 * 128, Some(64)).is_err());
        assert!(validate_request(&local, 1, 64 * 192, 8 * 192, 8 * 128, None).is_err());
        let mut wrong = local;
        if let AttentionPlan::SlidingWindow { window, .. } = &mut wrong {
            *window = 64;
        }
        assert!(validate_request(&wrong, 1, 64 * 192, 8 * 192, 8 * 128, Some(64)).is_err());
    }

    #[test]
    fn causal_grouped_window_and_denominator_sink() {
        let (global, local) = plans();
        for (plan, kv_heads) in [(global, 4), (local, 8)] {
            let n = 130;
            let q = vec![0.0; n * QUERY_HEADS * QK_DIM];
            let k = vec![0.0; n * kv_heads * QK_DIM];
            let mut v = vec![0.0; n * kv_heads * VALUE_DIM];
            for token in 0..n {
                for kv in 0..kv_heads {
                    v[(token * kv_heads + kv) * VALUE_DIM] = token as f32 + kv as f32 * 1000.0;
                }
            }
            let sink = (kv_heads == 8).then(|| vec![0.0; QUERY_HEADS]);
            let got = preprojected_attention(&plan, &q, &k, &v, sink.as_deref(), n).unwrap();
            assert_eq!(got[0], 0.0, "first query sees only key zero");
            let head = QUERY_HEADS / kv_heads;
            assert_eq!(
                got[head * VALUE_DIM],
                1000.0 / if sink.is_some() { 2.0 } else { 1.0 }
            );
            let last = (n - 1) * QUERY_HEADS * VALUE_DIM;
            let first = if kv_heads == 8 { 2 } else { 0 };
            let mean = (first + n - 1) as f32 / 2.0;
            let denominator = (n - first) as f32 + f32::from(sink.is_some());
            let expected = mean * (n - first) as f32 / denominator;
            assert!(
                (got[last] - expected).abs() < 1e-4,
                "{} vs {expected}",
                got[last]
            );
        }
    }

    #[test]
    fn nonfinite_input_and_score_refuse() {
        let (global, _) = plans();
        let mut q = vec![0.0; QUERY_HEADS * QK_DIM];
        let mut k = vec![0.0; 4 * QK_DIM];
        let v = vec![0.0; 4 * VALUE_DIM];
        q[0] = f32::NAN;
        assert!(preprojected_attention(&global, &q, &k, &v, None, 1).is_err());
        q[0] = 1e30;
        k[0] = 1e30;
        assert!(preprojected_attention(&global, &q, &k, &v, None, 1).is_err());
    }
}
