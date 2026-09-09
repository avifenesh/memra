//! The calibrated Qwen prefill program belongs to an artifact, not a dtype.
use std::collections::BTreeMap;

use crate::{GgmlType, GgufFile};

pub const PROGRAM: &str = "qwen35-prefill-nvfp4-a4-v1";
pub const PROGRAM_KEY: &str = "memra.activation_program";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPhase {
    Prefill,
    Decode,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearActivation {
    W4A8,
    CalibratedNvfp4 { multiplier: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefillFp4 {
    /// GGUF weight names, with calibrated activation dequantization multipliers.
    scales: BTreeMap<String, f32>,
}

/// The 400 large projections. Narrow alpha/beta, MTP, head and drafter are absent
/// by construction, even though some also have NVFP4 weights.
pub fn projection_names() -> Vec<String> {
    let mut names = Vec::with_capacity(400);
    for layer in 0..64 {
        for projection in ["ffn_gate", "ffn_up", "ffn_down"] {
            names.push(format!("blk.{layer}.{projection}.weight"));
        }
        let mixer: &[&str] = if layer % 4 == 3 {
            &["attn_q", "attn_k", "attn_v", "attn_output"]
        } else {
            &["attn_qkv", "attn_gate", "ssm_out"]
        };
        for projection in mixer {
            names.push(format!("blk.{layer}.{projection}.weight"));
        }
    }
    names
}

impl PrefillFp4 {
    pub fn from_gguf(g: &GgufFile) -> Result<Option<Self>, String> {
        let Some(value) = g.metadata.get(PROGRAM_KEY) else {
            return Ok(None);
        };
        if value.as_str() != Some(PROGRAM) {
            return Err("unsupported or malformed memra.activation_program".into());
        }
        let geometry = [
            ("block_count", 65),
            ("nextn_predict_layers", 1),
            ("embedding_length", 5120),
            ("feed_forward_length", 17408),
            ("full_attention_interval", 4),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 4),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
        ];
        if g.arch() != Some("qwen35")
            || geometry
                .iter()
                .any(|(key, expected)| g.meta_arch(key).and_then(|v| v.as_u64()) != Some(*expected))
        {
            return Err("calibrated prefill artifact has incompatible Qwen geometry".into());
        }
        let mut scales = BTreeMap::new();
        for name in projection_names() {
            let weight = g.find(&name).ok_or_else(|| format!("missing {name}"))?;
            if weight.ggml_type != GgmlType::NVFP4 || weight.ne.len() != 2 {
                return Err(format!(
                    "{name}: calibrated activation requires a 2-D NVFP4 weight"
                ));
            }
            let projection = name.split('.').nth(2).unwrap();
            let shape = match projection {
                "ffn_gate" | "ffn_up" => [5120, 17408],
                "ffn_down" => [17408, 5120],
                "attn_q" => [5120, 12288],
                "attn_k" | "attn_v" => [5120, 1024],
                "attn_output" | "ssm_out" => [6144, 5120],
                "attn_qkv" => [5120, 10240],
                "attn_gate" => [5120, 6144],
                _ => unreachable!("projection_names defines the complete program"),
            };
            if weight.ne != shape {
                return Err(format!(
                    "{name}: activation program expected shape {shape:?}"
                ));
            }
            let stem = name.strip_suffix(".weight").unwrap();
            let auxiliary = format!("{stem}.input_scale");
            let tensor = g
                .find(&auxiliary)
                .ok_or_else(|| format!("missing {auxiliary}"))?;
            if tensor.ggml_type != GgmlType::F32 || tensor.ne != [1] {
                return Err(format!("{auxiliary}: expected F32 [1] multiplier"));
            }
            let bytes: [u8; 4] = g
                .tensor_data(tensor)
                .try_into()
                .map_err(|_| format!("{auxiliary}: expected four bytes"))?;
            let multiplier = f32::from_le_bytes(bytes);
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return Err(format!(
                    "{auxiliary}: multiplier must be finite and positive"
                ));
            }
            scales.insert(name, multiplier);
        }
        for tensor in &g.tensors {
            if let Some(stem) = tensor.name.strip_suffix(".input_scale")
                && !scales.contains_key(&format!("{stem}.weight"))
            {
                return Err(format!("unexpected activation scale {}", tensor.name));
            }
        }
        Ok(Some(Self { scales }))
    }

    pub fn scales(&self) -> &BTreeMap<String, f32> {
        &self.scales
    }
}

/// Used by runtime dispatch. Row count is deliberately not an input: one-row
/// prefill tails and wide speculative verification have different semantics.
pub fn select_activation(
    program: Option<&PrefillFp4>,
    weight: &str,
    phase: LinearPhase,
) -> LinearActivation {
    if phase == LinearPhase::Prefill
        && let Some(multiplier) = program.and_then(|p| p.scales.get(weight))
    {
        return LinearActivation::CalibratedNvfp4 {
            multiplier: *multiplier,
        };
    }
    LinearActivation::W4A8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_less_artifact_never_selects_a4() {
        for phase in [
            LinearPhase::Prefill,
            LinearPhase::Decode,
            LinearPhase::Verify,
        ] {
            for name in projection_names() {
                assert_eq!(
                    select_activation(None, &name, phase),
                    LinearActivation::W4A8
                );
            }
        }
    }

    #[test]
    fn scaled_artifact_is_prefill_only_and_preserves_precision_islands() {
        let program = PrefillFp4 {
            scales: projection_names().into_iter().map(|n| (n, 0.125)).collect(),
        };
        assert_eq!(program.scales.len(), 400);
        for name in projection_names() {
            assert_eq!(
                select_activation(Some(&program), &name, LinearPhase::Prefill),
                LinearActivation::CalibratedNvfp4 { multiplier: 0.125 }
            );
            for phase in [LinearPhase::Decode, LinearPhase::Verify] {
                assert_eq!(
                    select_activation(Some(&program), &name, phase),
                    LinearActivation::W4A8
                );
            }
        }
        for name in [
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight",
            "blk.64.ffn_gate.weight",
            "output.weight",
            "token_embd.weight",
            "draft.blk.0.ffn_gate.weight",
        ] {
            assert_eq!(
                select_activation(Some(&program), name, LinearPhase::Prefill),
                LinearActivation::W4A8
            );
        }
    }
}
