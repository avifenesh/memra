//! Portable oracles for MiMo text attention chunks.
//!
//! Q and K are already RoPE-rotated. V already includes the source's 0.707
//! factor. The fresh-only entry point starts at position zero. The continuing
//! entry point reads decoded model-owned Q8_0-K/NVFP4-V prefix rows and F32
//! fresh rows. Neither entry point writes a future KV cache.

use memra_gguf::model_plan::{
    AttentionPlan, AttentionScale, TensorPresence, ValueNorm, ValueProjection,
};

pub const MAX_CHUNK: usize = 256;
pub const MAX_CONTINUING_CHUNK: usize = 128;
pub const MAX_TOTAL_POSITIONS: usize = 1_048_576;
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

/// Validate a continuation with `prefix` cached positions followed by `chunk` fresh positions.
///
/// The Q extent is `[chunk, 64, 192]`. Prefix and fresh K/V extents are
/// `[positions, kv_heads, 192 or 128]`, with the head count fixed by the plan.
/// `prefix + chunk` must fit the source's combined context limit.
#[allow(clippy::too_many_arguments)]
pub fn validate_continuing_request(
    attention: &AttentionPlan,
    prefix: usize,
    chunk: usize,
    query_len: usize,
    prefix_key_len: usize,
    prefix_value_len: usize,
    key_len: usize,
    value_len: usize,
    sink_len: Option<usize>,
) -> Result<ChunkShape, &'static str> {
    if !(1..=MAX_CONTINUING_CHUNK).contains(&chunk) {
        return Err("MiMo continuation accepts 1..=128 fresh positions");
    }
    let total = prefix
        .checked_add(chunk)
        .ok_or("MiMo continuation position count overflows")?;
    if total > MAX_TOTAL_POSITIONS {
        return Err("MiMo continuation exceeds combined context limit");
    }
    let shape = validate_request(attention, chunk, query_len, key_len, value_len, sink_len)?;
    let expected_prefix_keys = prefix
        .checked_mul(shape.kv_heads)
        .and_then(|rows| rows.checked_mul(QK_DIM))
        .ok_or("MiMo continuation prefix K extent overflows")?;
    let expected_prefix_values = prefix
        .checked_mul(shape.kv_heads)
        .and_then(|rows| rows.checked_mul(VALUE_DIM))
        .ok_or("MiMo continuation prefix V extent overflows")?;
    if prefix_key_len != expected_prefix_keys || prefix_value_len != expected_prefix_values {
        return Err("MiMo continuation prefix K/V extent mismatch");
    }
    Ok(shape)
}

/// Causal attention for fresh rows after an already cached prefix.
///
/// Prefix K/V are F32 values decoded from the model-owned Q8_0-K/NVFP4-V
/// cache. Fresh Q/K/V are F32 current-chunk values. Each fresh row `i` has
/// absolute position `prefix + i`, and sees keys through that position only.
/// A local row sees `max(0, position + 1 - 128)..=position`. The sink is one
/// logit per query head in the softmax denominator, with no associated V row.
/// Output is row-major `[chunk, 64, 128]`.
#[allow(clippy::too_many_arguments)]
pub fn preprojected_continuing_attention(
    attention: &AttentionPlan,
    prefix_key: &[f32],
    prefix_value: &[f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sink: Option<&[f32]>,
    prefix: usize,
    chunk: usize,
) -> Result<Vec<f32>, String> {
    let shape = validate_continuing_request(
        attention,
        prefix,
        chunk,
        query.len(),
        prefix_key.len(),
        prefix_value.len(),
        key.len(),
        value.len(),
        sink.map(<[f32]>::len),
    )?;
    if prefix_key
        .iter()
        .chain(prefix_value)
        .chain(query)
        .chain(key)
        .chain(value)
        .chain(sink.into_iter().flatten())
        .any(|x| !x.is_finite())
    {
        return Err("MiMo continuation input is non-finite".into());
    }
    let scale = 1.0 / (QK_DIM as f32).sqrt();
    let mut output = vec![0.0; chunk * QUERY_HEADS * VALUE_DIM];
    let mut scores = Vec::new();
    for fresh_position in 0..chunk {
        let position = prefix + fresh_position;
        let first = if shape.window == 0 {
            0
        } else {
            (position + 1).saturating_sub(shape.window)
        };
        for head in 0..QUERY_HEADS {
            let kv_head = head / (QUERY_HEADS / shape.kv_heads);
            let q_base = (fresh_position * QUERY_HEADS + head) * QK_DIM;
            let q_row = &query[q_base..q_base + QK_DIM];
            scores.clear();
            for token in first..=position {
                let (keys, row) = if token < prefix {
                    (prefix_key, token)
                } else {
                    (key, token - prefix)
                };
                let k_base = (row * shape.kv_heads + kv_head) * QK_DIM;
                let dot: f32 = q_row
                    .iter()
                    .zip(&keys[k_base..k_base + QK_DIM])
                    .map(|(&q, &k)| q * k)
                    .sum();
                let score = dot * scale;
                if !score.is_finite() {
                    return Err("MiMo continuation score is non-finite".into());
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
                return Err("MiMo continuation denominator is invalid".into());
            }
            let out_base = (fresh_position * QUERY_HEADS + head) * VALUE_DIM;
            for (offset, &score) in scores.iter().enumerate() {
                let token = first + offset;
                let (values, row) = if token < prefix {
                    (prefix_value, token)
                } else {
                    (value, token - prefix)
                };
                let v_base = (row * shape.kv_heads + kv_head) * VALUE_DIM;
                let weight = (score - max).exp() / denominator;
                for dim in 0..VALUE_DIM {
                    output[out_base + dim] += weight * values[v_base + dim];
                }
            }
            if output[out_base..out_base + VALUE_DIM]
                .iter()
                .any(|x| !x.is_finite())
            {
                return Err("MiMo continuation output is non-finite".into());
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

    // Independent one-query calculation over exactly the keys available at
    // this absolute position. f64 accumulation helps expose indexing mistakes
    // without sharing the chunk oracle's F32 reduction.
    fn sequential_one_query(
        q_row: &[f32],
        all_key: &[f32],
        all_value: &[f32],
        sink: Option<f32>,
        position: usize,
        head: usize,
        shape: ChunkShape,
    ) -> [f32; VALUE_DIM] {
        let first = if shape.window == 0 {
            0
        } else {
            (position + 1).saturating_sub(shape.window)
        };
        let kv_head = head / (QUERY_HEADS / shape.kv_heads);
        let scale = f64::from(1.0 / (QK_DIM as f32).sqrt());
        let scores: Vec<f64> = (first..=position)
            .map(|token| {
                let k_base = (token * shape.kv_heads + kv_head) * QK_DIM;
                q_row
                    .iter()
                    .zip(&all_key[k_base..k_base + QK_DIM])
                    .map(|(&q, &k)| f64::from(q) * f64::from(k))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let maximum = scores
            .iter()
            .copied()
            .chain(sink.map(f64::from))
            .fold(f64::NEG_INFINITY, f64::max);
        let denominator = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .sum::<f64>()
            + sink.map_or(0.0, |score| (f64::from(score) - maximum).exp());
        let mut out = [0.0f64; VALUE_DIM];
        for (offset, score) in scores.iter().enumerate() {
            let v_base = ((first + offset) * shape.kv_heads + kv_head) * VALUE_DIM;
            let probability = (score - maximum).exp() / denominator;
            for (sum, &v) in out.iter_mut().zip(&all_value[v_base..v_base + VALUE_DIM]) {
                *sum += probability * f64::from(v);
            }
        }
        out.map(|v| v as f32)
    }

    #[test]
    fn continuing_chunk_matches_sequential_one_query_at_prefix_offsets() {
        let (global, local) = plans();
        for (plan, kv_heads, window) in [(global, 4, 0), (local, 8, 128)] {
            for prefix in [0, 127, 128, 319] {
                let chunk = 2;
                let total = prefix + chunk;
                let mut all_key = vec![0.0; total * kv_heads * QK_DIM];
                let mut all_value = vec![0.0; total * kv_heads * VALUE_DIM];
                for token in 0..total {
                    for kv_head in 0..kv_heads {
                        let k_base = (token * kv_heads + kv_head) * QK_DIM;
                        for dim in 0..QK_DIM {
                            all_key[k_base + dim] =
                                ((token * 11 + kv_head * 7 + dim * 3) % 43) as f32 / 80.0 - 0.25;
                        }
                        let v_base = (token * kv_heads + kv_head) * VALUE_DIM;
                        for dim in 0..VALUE_DIM {
                            all_value[v_base + dim] =
                                token as f32 * 0.03 + kv_head as f32 * 0.2 + dim as f32 * 0.001;
                        }
                    }
                }
                let query: Vec<f32> = (0..chunk * QUERY_HEADS * QK_DIM)
                    .map(|index| {
                        let token = index / (QUERY_HEADS * QK_DIM);
                        let head = (index / QK_DIM) % QUERY_HEADS;
                        let dim = index % QK_DIM;
                        ((token * 19 + head * 5 + dim * 7) % 37) as f32 / 64.0 - 0.28
                    })
                    .collect();
                let sink = (window != 0).then(|| {
                    (0..QUERY_HEADS)
                        .map(|head| head as f32 * 0.005 - 0.2)
                        .collect::<Vec<_>>()
                });
                let prefix_k_end = prefix * kv_heads * QK_DIM;
                let prefix_v_end = prefix * kv_heads * VALUE_DIM;
                let got = preprojected_continuing_attention(
                    &plan,
                    &all_key[..prefix_k_end],
                    &all_value[..prefix_v_end],
                    &query,
                    &all_key[prefix_k_end..],
                    &all_value[prefix_v_end..],
                    sink.as_deref(),
                    prefix,
                    chunk,
                )
                .unwrap();
                let shape = ChunkShape { kv_heads, window };
                for fresh_position in 0..chunk {
                    for head in 0..QUERY_HEADS {
                        let q_base = (fresh_position * QUERY_HEADS + head) * QK_DIM;
                        let expected = sequential_one_query(
                            &query[q_base..q_base + QK_DIM],
                            &all_key,
                            &all_value,
                            sink.as_ref().map(|s| s[head]),
                            prefix + fresh_position,
                            head,
                            shape,
                        );
                        let out_base = (fresh_position * QUERY_HEADS + head) * VALUE_DIM;
                        for dim in 0..VALUE_DIM {
                            let actual = got[out_base + dim];
                            assert!(
                                (actual - expected[dim]).abs() < 2e-4,
                                "prefix={prefix} window={window} row={fresh_position} head={head} dim={dim}: {actual} vs {}",
                                expected[dim]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn continuing_local_window_excludes_oldest_prefix_and_sink_has_no_value() {
        let (_, local) = plans();
        let prefix = 127;
        let chunk = 2;
        let query = vec![0.0; chunk * QUERY_HEADS * QK_DIM];
        let prefix_key = vec![0.0; prefix * 8 * QK_DIM];
        let key = vec![0.0; chunk * 8 * QK_DIM];
        let mut prefix_value = vec![0.0; prefix * 8 * VALUE_DIM];
        let mut value = vec![0.0; chunk * 8 * VALUE_DIM];
        prefix_value[0] = 129.0; // absolute token 0, KV head 0
        prefix_value[VALUE_DIM] = 1290.0; // absolute token 0, KV head 1
        value[0] = 387.0; // absolute token 127, KV head 0
        value[8 * VALUE_DIM] = 258.0; // absolute token 128, KV head 0
        let sink = vec![0.0; QUERY_HEADS];
        let got = preprojected_continuing_attention(
            &local,
            &prefix_key,
            &prefix_value,
            &query,
            &key,
            &value,
            Some(&sink),
            prefix,
            chunk,
        )
        .unwrap();
        let row = QUERY_HEADS * VALUE_DIM;
        assert!((got[0] - 4.0).abs() < 1e-5); // (129 + 387) / (128 keys + sink)
        assert!((got[row] - 5.0).abs() < 1e-5); // (387 + 258) / 129
        assert!((got[8 * VALUE_DIM] - 10.0).abs() < 1e-5);
        assert_eq!(got[row + 8 * VALUE_DIM], 0.0);
        assert_eq!(got[VALUE_DIM - 1], 0.0); // sink supplies no V row
    }

    #[test]
    fn continuing_bounds_shapes_plan_and_nonfinite_values_refuse() {
        let (global, local) = plans();
        let prefix = 1;
        let chunk = 1;
        let mut query = vec![0.0; QUERY_HEADS * QK_DIM];
        let mut prefix_key = vec![0.0; 4 * QK_DIM];
        let mut prefix_value = vec![0.0; 4 * VALUE_DIM];
        let mut key = prefix_key.clone();
        let mut value = prefix_value.clone();
        let lengths = [
            query.len(),
            prefix_key.len(),
            prefix_value.len(),
            key.len(),
            value.len(),
        ];
        let check = |plan: &AttentionPlan,
                     prefix: usize,
                     chunk: usize,
                     lengths: [usize; 5],
                     sink_len: Option<usize>| {
            validate_continuing_request(
                plan, prefix, chunk, lengths[0], lengths[1], lengths[2], lengths[3], lengths[4],
                sink_len,
            )
        };
        assert_eq!(
            check(&global, prefix, chunk, lengths, None),
            Ok(ChunkShape {
                kv_heads: 4,
                window: 0
            })
        );
        assert!(check(&global, prefix, 0, lengths, None).is_err());
        assert!(check(&global, prefix, 129, lengths, None).is_err());
        assert_eq!(
            check(&global, MAX_TOTAL_POSITIONS, 1, lengths, None),
            Err("MiMo continuation exceeds combined context limit")
        );
        assert_eq!(
            check(&global, usize::MAX, 1, lengths, None),
            Err("MiMo continuation position count overflows")
        );
        assert_eq!(
            check(
                &global,
                MAX_TOTAL_POSITIONS - MAX_CONTINUING_CHUNK,
                MAX_CONTINUING_CHUNK,
                [
                    MAX_CONTINUING_CHUNK * QUERY_HEADS * QK_DIM,
                    (MAX_TOTAL_POSITIONS - MAX_CONTINUING_CHUNK) * 4 * QK_DIM,
                    (MAX_TOTAL_POSITIONS - MAX_CONTINUING_CHUNK) * 4 * VALUE_DIM,
                    MAX_CONTINUING_CHUNK * 4 * QK_DIM,
                    MAX_CONTINUING_CHUNK * 4 * VALUE_DIM,
                ],
                None,
            ),
            Ok(ChunkShape {
                kv_heads: 4,
                window: 0
            })
        );
        assert!(check(&global, prefix + 1, chunk, lengths, None).is_err());
        for field in 0..lengths.len() {
            let mut malformed = lengths;
            malformed[field] -= 1;
            assert!(
                check(&global, prefix, chunk, malformed, None).is_err(),
                "extent {field} was accepted"
            );
        }
        assert!(check(&global, prefix, chunk, lengths, Some(QUERY_HEADS)).is_err());
        assert!(check(&local, prefix, chunk, lengths, None).is_err());
        let mut wrong = global.clone();
        if let AttentionPlan::Full(plan) = &mut wrong {
            plan.query_heads = 32;
        }
        assert!(check(&wrong, prefix, chunk, lengths, None).is_err());

        let run = |query: &[f32],
                   prefix_key: &[f32],
                   prefix_value: &[f32],
                   key: &[f32],
                   value: &[f32]| {
            preprojected_continuing_attention(
                &global,
                prefix_key,
                prefix_value,
                query,
                key,
                value,
                None,
                prefix,
                chunk,
            )
        };
        query[0] = f32::NAN;
        assert!(run(&query, &prefix_key, &prefix_value, &key, &value).is_err());
        query[0] = 0.0;
        prefix_key[0] = f32::INFINITY;
        assert!(run(&query, &prefix_key, &prefix_value, &key, &value).is_err());
        prefix_key[0] = 0.0;
        prefix_value[0] = f32::NAN;
        assert!(run(&query, &prefix_key, &prefix_value, &key, &value).is_err());
        prefix_value[0] = 0.0;
        query[0] = 1e30;
        key[0] = 1e30;
        assert!(run(&query, &prefix_key, &prefix_value, &key, &value).is_err());
        query[0] = 0.0;
        key[0] = 0.0;
        value[0] = f32::NEG_INFINITY;
        assert!(run(&query, &prefix_key, &prefix_value, &key, &value).is_err());
        let bad_sink = [f32::NAN; QUERY_HEADS];
        assert!(
            preprojected_continuing_attention(
                &local,
                &[],
                &[],
                &query,
                &vec![0.0; 8 * QK_DIM],
                &vec![0.0; 8 * VALUE_DIM],
                Some(&bad_sink),
                0,
                1,
            )
            .is_err()
        );
    }
}
