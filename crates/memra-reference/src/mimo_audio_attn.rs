//! Portable attention for the pinned MiMo audio patch encoder.
//!
//! Q and K have already received RoPE, and Q/K/V have already been projected.
//! This covers one of the six local transformer layers, before its output
//! projection. BF16 rounding, QKV projection, and the bundled codec are
//! separate checks.

use memra_gguf::model_packs::mimo_v2::audio::MiMoAudioPatchPlan;

pub const MAX_GROUPS: usize = 1_500;
const GROUP_SIZE: usize = 4;
const HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const WIDTH: usize = HEADS * HEAD_DIM;

pub fn validate_request(
    plan: &MiMoAudioPatchPlan,
    layer: usize,
    groups: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
) -> Result<usize, &'static str> {
    if plan.local_layers != 6
        || layer >= plan.local_layers
        || plan.group_size != GROUP_SIZE
        || plan.local_hidden != WIDTH
        || plan.local_heads != HEADS
        || plan.local_head_dim != HEAD_DIM
        || !plan.local_full_attention
        || plan.local_rope_theta.to_bits() != 640_000.0f32.to_bits()
    {
        return Err("MiMo audio attention differs from pinned local transformer");
    }
    if !(1..=MAX_GROUPS).contains(&groups) {
        return Err("MiMo audio attention groups are outside 1..=1500");
    }
    let elements = groups
        .checked_mul(GROUP_SIZE)
        .and_then(|count| count.checked_mul(WIDTH))
        .ok_or("MiMo audio attention extent overflows")?;
    if q_len != elements || k_len != elements || v_len != elements {
        return Err("MiMo audio Q/K/V extents differ from [groups,4,16,64]");
    }
    Ok(elements)
}

/// Row-major `[groups, 4, 16, 64]` Q/K/V to the same-shaped f32 result.
///
/// Each of the four positions sees all four keys in its own group. There is
/// no causal mask, sink, or reused KV state.
pub fn preprojected_attention(
    plan: &MiMoAudioPatchPlan,
    layer: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    groups: usize,
) -> Result<Vec<f32>, String> {
    let elements = validate_request(plan, layer, groups, q.len(), k.len(), v.len())?;
    if q.iter().chain(k).chain(v).any(|value| !value.is_finite()) {
        return Err("MiMo audio attention input is not finite".into());
    }
    let mut output = vec![0.0f32; elements];
    for group in 0..groups {
        for query in 0..GROUP_SIZE {
            for head in 0..HEADS {
                let q_base = ((group * GROUP_SIZE + query) * HEADS + head) * HEAD_DIM;
                let q_row = &q[q_base..q_base + HEAD_DIM];
                let mut scores = [0.0f32; GROUP_SIZE];
                for (key, score) in scores.iter_mut().enumerate() {
                    let k_base = ((group * GROUP_SIZE + key) * HEADS + head) * HEAD_DIM;
                    let k_row = &k[k_base..k_base + HEAD_DIM];
                    let dot = q_row
                        .iter()
                        .zip(k_row)
                        .fold(0.0f32, |sum, (&a, &b)| sum + a * b);
                    *score = dot * 0.125;
                    if !score.is_finite() {
                        return Err("MiMo audio attention score is not finite".into());
                    }
                }
                let max = scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
                let mut weights = scores.map(|score| (score - max).exp());
                let denominator: f32 = weights.iter().sum();
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err("MiMo audio attention softmax is invalid".into());
                }
                for weight in &mut weights {
                    *weight /= denominator;
                }
                let out_row = &mut output[q_base..q_base + HEAD_DIM];
                for (key, &weight) in weights.iter().enumerate() {
                    let v_base = ((group * GROUP_SIZE + key) * HEADS + head) * HEAD_DIM;
                    for (out, &value) in out_row.iter_mut().zip(&v[v_base..v_base + HEAD_DIM]) {
                        *out += weight * value;
                    }
                }
                if out_row.iter().any(|value| !value.is_finite()) {
                    return Err("MiMo audio attention output is not finite".into());
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::audio::pinned_patch_plan;

    fn plan() -> MiMoAudioPatchPlan {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(
            "../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        )));
        pinned_patch_plan(&config).unwrap()
    }

    fn base(group: usize, token: usize, head: usize) -> usize {
        ((group * GROUP_SIZE + token) * HEADS + head) * HEAD_DIM
    }

    #[test]
    fn all_four_tokens_are_visible_and_groups_and_heads_stay_separate() {
        let mut q = vec![0.0; 2 * GROUP_SIZE * WIDTH];
        let mut k = q.clone();
        let mut v = q.clone();
        for token in 0..GROUP_SIZE {
            v[base(0, token, 0)] = [1.0, 3.0, 7.0, 11.0][token];
            v[base(0, token, 1)] = [2.0, 6.0, 10.0, 14.0][token];
            v[base(1, token, 0)] = [10.0, 14.0, 18.0, 22.0][token];
        }
        q[base(0, 0, 0)] = 8.0;
        k[base(0, 1, 0)] = 2.0f32.ln();
        q[base(0, 3, 1)] = 8.0;
        k[base(0, 0, 1)] = 3.0f32.ln();
        let result = preprojected_attention(&plan(), 5, &q, &k, &v, 2).unwrap();
        assert!((result[base(0, 0, 0)] - 5.0).abs() < 1e-6);
        assert!((result[base(0, 1, 0)] - 5.5).abs() < 1e-6);
        assert!((result[base(0, 3, 1)] - 6.0).abs() < 1e-6);
        assert!((result[base(1, 0, 0)] - 16.0).abs() < 1e-6);
        assert_eq!(result[base(0, 0, 2)], 0.0);
    }

    #[test]
    fn boundary_and_source_geometry_refuse() {
        let zeros = vec![0.0; GROUP_SIZE * WIDTH];
        assert!(preprojected_attention(&plan(), 0, &zeros, &zeros, &zeros, 1).is_ok());
        assert!(preprojected_attention(&plan(), 6, &zeros, &zeros, &zeros, 1).is_err());
        assert!(preprojected_attention(&plan(), 0, &[], &[], &[], 0).is_err());
        assert!(validate_request(&plan(), 0, MAX_GROUPS + 1, 0, 0, 0).is_err());
        assert!(preprojected_attention(&plan(), 0, &zeros[1..], &zeros, &zeros, 1).is_err());
        let mut causal = plan();
        causal.local_full_attention = false;
        assert!(preprojected_attention(&causal, 0, &zeros, &zeros, &zeros, 1).is_err());
        let mut changed = plan();
        changed.local_head_dim = 128;
        assert!(preprojected_attention(&changed, 0, &zeros, &zeros, &zeros, 1).is_err());
    }

    #[test]
    fn nonfinite_inputs_and_finite_overflow_refuse() {
        let mut q = vec![0.0; GROUP_SIZE * WIDTH];
        let mut k = q.clone();
        let v = q.clone();
        q[0] = f32::NAN;
        assert!(preprojected_attention(&plan(), 0, &q, &k, &v, 1).is_err());
        q[0] = 1e30;
        k[0] = 1e30;
        assert!(preprojected_attention(&plan(), 0, &q, &k, &v, 1).is_err());
    }
}
