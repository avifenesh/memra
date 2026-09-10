//! The calibrated Qwen prefill program belongs to an artifact, not a dtype.
use std::collections::BTreeMap;

use crate::{GgmlType, GgufFile};

pub const PROGRAM: &str = "qwen35-prefill-nvfp4-a4-v1";
pub const PROGRAM_KEY: &str = "memra.activation_program";
/// OUR suffix, deliberately not ".input_scale": that one is a reserved quant auxiliary
/// (`QuantAuxTensor::InputScale`, and `is_quant_auxiliary` on the safetensors source)
/// meaning a ModelOpt static W4A8 activation scale. Reusing it would make the census
/// describe a checkpoint we did not mint.
pub const SCALE_SUFFIX: &str = ".a4_input_scale";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPhase {
    Prefill,
    Decode,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearActivation {
    W4A8,
    CalibratedNvfp4 { multiplier: f32, slot: u32 },
}

/// The program's 400 linears are numbered in BTreeMap (name) order. The slot travels with the
/// multiplier onto the weight so a runtime receipt can say WHICH projections executed the
/// program, not just how many GEMMs it issued: a missed prefill call site is a specific
/// projection, and a count alone cannot name it.
pub const PROGRAM_SLOTS: usize = 400;

#[derive(Clone, PartialEq)]
pub struct PrefillFp4 {
    /// GGUF weight names, with calibrated activation dequantization multipliers.
    scales: BTreeMap<String, f32>,
}

/// A plan receipt has to say WHICH calibration it is holding, not just that it holds one, or two
/// artifacts calibrated from different corpora read identically in the log. The derived Debug
/// would dump 400 lines to do that, so the digest stands in for the map: same 400 names and same
/// 400 bit patterns, same digest.
impl std::fmt::Debug for PrefillFp4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillFp4")
            .field("program", &PROGRAM)
            .field("scales", &self.scales.len())
            .field("digest", &format_args!("{:016x}", self.digest()))
            .finish()
    }
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
        // GATE-HARNESS ONLY (MEMRA_A4_DISABLE): refuse the program HERE, before the weights are
        // loaded. Clearing `cfg.prefill_activation` after `HybridModel::load` does nothing --
        // every weight has already been stamped and `matmul_prefill` reads the STAMP, not the
        // config -- so a control written that way silently measures the A4 arm twice.
        if std::env::var_os("MEMRA_A4_DISABLE").is_some() {
            return Ok(None);
        }
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
            let auxiliary = format!("{stem}{SCALE_SUFFIX}");
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
            if let Some(stem) = tensor.name.strip_suffix(SCALE_SUFFIX)
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

    /// Slot index of a weight in the program, in the same BTreeMap order `slot_names` returns.
    pub fn slot(&self, weight: &str) -> Option<u32> {
        self.scales
            .keys()
            .position(|name| name == weight)
            .map(|i| i as u32)
    }

    /// The program's weight names, slot-indexed.
    pub fn slot_names(&self) -> Vec<&str> {
        self.scales.keys().map(String::as_str).collect()
    }

    /// FNV-1a over every (name, IEEE-754 bits) pair in BTreeMap order. Bits, not float equality:
    /// two multipliers that differ in the last mantissa bit are two different calibrations.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        for (name, multiplier) in &self.scales {
            eat(name.as_bytes());
            eat(&multiplier.to_bits().to_le_bytes());
        }
        h
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
        && let Some(program) = program
        && let Some(multiplier) = program.scales.get(weight)
    {
        return LinearActivation::CalibratedNvfp4 {
            multiplier: *multiplier,
            slot: program
                .slot(weight)
                .expect("a name found in the map has a position in the map"),
        };
    }
    LinearActivation::W4A8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(multiplier: f32) -> PrefillFp4 {
        PrefillFp4 {
            scales: projection_names()
                .into_iter()
                .map(|n| (n, multiplier))
                .collect(),
        }
    }

    #[test]
    fn digest_separates_two_calibrations_and_the_receipt_carries_it() {
        let a = program(0.125);
        let b = program(0.125);
        assert_eq!(
            a.digest(),
            b.digest(),
            "same 400 pairs must digest the same"
        );

        // One multiplier, one mantissa bit apart: a different calibration, a different digest.
        let mut c = program(0.125);
        let name = projection_names()[7].clone();
        c.scales
            .insert(name.clone(), f32::from_bits(0.125f32.to_bits() + 1));
        assert_ne!(a.digest(), c.digest());

        // Same 400 multipliers under a DIFFERENT name is also a different program.
        let mut d = program(0.125);
        d.scales.remove(&name);
        d.scales.insert("blk.0.ssm_alpha.weight".into(), 0.125);
        assert_ne!(a.digest(), d.digest());

        let receipt = format!("{a:?}");
        assert!(receipt.contains(PROGRAM), "{receipt}");
        assert!(receipt.contains("400"), "{receipt}");
        assert!(
            receipt.contains(&format!("{:016x}", a.digest())),
            "the plan receipt must carry the digest, not just the count: {receipt}"
        );
        assert!(
            !receipt.contains("blk.0.ffn_gate"),
            "the receipt must not dump 400 scale lines: {receipt}"
        );
    }

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
        let program = program(0.125);
        assert_eq!(program.scales.len(), 400);
        for name in projection_names() {
            assert_eq!(
                select_activation(Some(&program), &name, LinearPhase::Prefill),
                LinearActivation::CalibratedNvfp4 {
                    multiplier: 0.125,
                    slot: program.slot(&name).expect("a program name has a slot")
                }
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
