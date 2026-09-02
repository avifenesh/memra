//! qwen4_exp GPU eager vs memra-reference — tiny exactness gate, three arms.
//!
//! Arm A (fixture): the pack's deterministic tiny fixture through `from_reference_weights`.
//! Arm B (bf16 dir): a synthesized tiny BF16 safetensors dir through the FULL
//!   pack/plan/contract loader (`Qwen4ExpGpu::load_from_dir`) — name mapping, shape/dtype
//!   checks, n-gram shard concatenation, transforms, the (1+w) fold; the reference side
//!   consumes `read_checkpoint(..).into_reference_weights()`, so the gate proves the
//!   loader's weights and the GPU forward agree end to end. (Transform/fold correctness
//!   against upstream transformers is the tinyparity/goldens lane, not this gate.)
//! Arm C (nvfp4 dir): the same dir with modelopt-NVFP4 stacked TRUNK expert banks (random
//!   valid codes/scales, pow2 macros): device-resident quantized banks + per-routed-expert
//!   kernel dequant vs the host `dequant_nvfp4_expert` decoder feeding the reference.
//! Arm D (nvfp4 per-expert dir): the REAL mint's tensor-name shape (census receipt
//!   raw/nvfp4-census-names.tsv) — un-fused per-expert projections with the modelopt
//!   sibling set incl. `input_scale` (validated, unused: W4A16 eager) and the UNSHARDED
//!   n-gram table; binds through the pack's PerExpertModelopt dialect contract. The tiny
//!   down_proj (ff=8) derives BF16 per geometry; gate/up ride the NVFP4 path.
//!
//! Every arm runs BOTH phases: full prefill row-compare AND the cache-vs-full decode
//! invariance (prefill N, then M incremental steps vs the full-sequence reference rows).
//! Token program: mid-prompt EOS + decode-side EOS (PLE n-gram segment resets), long
//! enough that the tiny indexer budget (2 blocks) actually drops blocks.
//!
//! Exactness class: f32 GPU vs f32 host reference; policy max_abs <= 0.01,
//! max_rel <= 0.01 (denominator max(1, |ref|)), argmax match on EVERY compared row — the
//! modelplan_reference_gate policy. Measured maxima are banked in the receipt.
//!
//! Usage: qwen4exp-gpu-gate <receipt.tsv>

// lane/clippy-zero-restore-20260901: the gate's comparison tuples are deliberately explicit
// (naming them buys nothing in a one-file gate binary), and the summaries vec is built by
// pushes because each push is a named gate arm carrying its own comment block.
#![allow(clippy::type_complexity, clippy::vec_init_then_push)]

use memra_engine::Engine;
use memra_engine::qwen4exp_gpu::{LoadOptions, Qwen4ExpGpu, read_checkpoint, read_checkpoint_with};
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::{AttentionPlan, ModelPlan, RopeFactors};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, QuantConstraint, TensorMatch,
};
use memra_reference::{ReferenceWeights, deterministic_fixture, execute};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// The pack's tiny config VERBATIM (model_packs/qwen4_exp tiny_plan); drift fails loudly
// via the plan-equality assert in main.
const TINY_CONFIG: &str = r#"{"model_type":"qwen4_exp","text_config":{
    "model_type":"qwen4_exp_text","eos_token_id":63,
    "num_hidden_layers":4,"hidden_size":16,
    "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
    "intermediate_size":32,"vocab_size":64,"max_position_embeddings":64,
    "rms_norm_eps":0.000001,"full_attention_interval":4,
    "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":0.25,
    "mrope_section":[1,1,2],"mrope_interleaved":true},
    "linear_conv_kernel_dim":4,"linear_key_head_dim":4,"linear_value_head_dim":4,
    "linear_num_key_heads":1,"linear_num_value_heads":2,
    "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":8,
    "shared_expert_intermediate_size":8,
    "indexer_n_heads":1,"indexer_kv_heads":1,"indexer_head_dim":4,
    "indexer_compress_ratio":4,"indexer_budget":8,
    "hc_count":2,"hc_lowrank":4,
    "ngram_size":3,"heads_per_ngram":2,"ngram_vocab_size_base":64,
    "make_ngram_vocab_size_divisible_by":128,"split_ngram_parts":2,
    "ple_layer_ids":[2],"ple_embed_dim":16,"ple_conv_kernel_size":4,
    "output_gate_type":"sigmoid",
    "mtp":{"num_hidden_layers":1,"rope_theta":10000},"mtp_num_hidden_layers":1}}"#;

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best
}

struct RowStats {
    max_abs: f32,
    max_rel: f32,
    /// Reference-row magnitude (max |logit|) — the scale the abs tolerance reads against.
    ref_absmax: f32,
    argmax_match: bool,
}

fn compare_row(reference: &[f32], candidate: &[f32]) -> RowStats {
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut ref_absmax = 0.0f32;
    for (&r, &c) in reference.iter().zip(candidate) {
        let abs = (r - c).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / r.abs().max(1.0));
        ref_absmax = ref_absmax.max(r.abs());
    }
    RowStats {
        max_abs,
        max_rel,
        ref_absmax,
        argmax_match: argmax(reference) == argmax(candidate),
    }
}

// ---------------------------------------------------------------- synthesized checkpoints

fn hash_name(name: &str) -> u64 {
    let mut value = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.bytes() {
        value ^= byte as u64;
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    value
}

/// Deterministic values, `center + uniform(-scale, scale)` — the fixture generator's
/// hash chain keyed by (name, index).
fn gen_f32(name: &str, elements: usize, center: f32, scale: f32) -> Vec<f32> {
    let salt = hash_name(name);
    (0..elements)
        .map(|index| {
            let mut value = index as u64 ^ salt.wrapping_mul(0x9e37_79b9);
            value ^= value >> 16;
            value = value.wrapping_mul(0x045d_9f3b);
            value ^= value >> 16;
            let unit = (value as u32) as f32 / u32::MAX as f32;
            center + (2.0 * unit - 1.0) * scale
        })
        .collect()
}

fn gen_bytes(name: &str, elements: usize, lo: u8, hi: u8) -> Vec<u8> {
    let salt = hash_name(name);
    (0..elements)
        .map(|index| {
            let mut value = index as u64 ^ salt.wrapping_mul(0x9e37_79b9);
            value ^= value >> 13;
            value = value.wrapping_mul(0x045d_9f3b);
            (lo as u64 + value % (hi as u64 - lo as u64 + 1)) as u8
        })
        .collect()
}

/// f32 -> bf16 round-to-nearest-even (the dspark_extract helper).
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
        .collect()
}

struct StEntry {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

/// Minimal single-file safetensors writer (8-byte LE header length + JSON + data).
fn write_safetensors(path: &Path, entries: &[StEntry]) -> Res<()> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (index, entry) in entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let end = offset + entry.bytes.len();
        let shape = entry
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            entry.name, entry.dtype, shape, offset, end
        ));
        offset = end;
    }
    header.push('}');
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for entry in entries {
        out.extend_from_slice(&entry.bytes);
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn first_primes_at_least(base: i64, count: usize) -> Vec<i64> {
    let is_prime = |n: i64| {
        if n < 2 {
            return false;
        }
        let mut d = 2;
        while d * d <= n {
            if n % d == 0 {
                return false;
            }
            d += 1;
        }
        true
    };
    let mut out = Vec::with_capacity(count);
    let mut candidate = base.max(2);
    while out.len() < count {
        if is_prime(candidate) {
            out.push(candidate);
        }
        candidate += 1;
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum DirKind {
    /// The BF16 export: fused 3D banks, sharded n-gram table.
    Bf16Fused,
    /// Fused STACKED modelopt-NVFP4 trunk gate_up banks (the pre-mint assumption; kept
    /// as the stacked-reader fixture).
    Nvfp4Stacked,
    /// The REAL mint shape: per-expert modelopt projections + unsharded table
    /// (PerExpertModelopt dialect contract drives the name set).
    Nvfp4PerExpert,
}

/// Synthesize a complete tiny checkpoint dir from the pack contract for `kind`: every
/// requirement gets deterministic values (MTP rows included, so the dir is
/// census-complete). NVFP4 rows get random valid e2m1 codes, positive non-NaN e4m3
/// scales, pow2 macro 2^-5, and (per-expert) an `input_scale` sibling.
fn synthesize_dir(dir: &Path, cfg: &ModelConfig, plan: &ModelPlan, kind: DirKind) -> Res<()> {
    use memra_gguf::model_packs::qwen4_exp::{ExpertDialect, tensor_contract_for};
    use memra_gguf::tensor_contract::{LayerTensor, TensorId, TensorOwner};
    let pack = memra_gguf::model_packs::for_config(cfg).ok_or("no pack for tiny config")?;
    let contract = match kind {
        DirKind::Nvfp4PerExpert => {
            tensor_contract_for(cfg, plan, ExpertDialect::PerExpertModelopt)?
        }
        _ => pack.compile_tensor_contract(
            cfg,
            plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )?,
    };
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("config.json"), TINY_CONFIG)?;
    let n_trunk = plan.layers.len() as u32;
    let mut entries: Vec<StEntry> = Vec::new();
    for requirement in &contract.requirements {
        let elements = |shape: &[u64]| shape.iter().map(|&d| d as usize).product::<usize>();
        if requirement.match_mode == TensorMatch::All {
            // n-gram shards.
            for name in &requirement.names {
                let n = elements(&requirement.shape);
                entries.push(StEntry {
                    name: name.clone(),
                    dtype: "BF16",
                    shape: requirement.shape.clone(),
                    bytes: bf16_bytes(&gen_f32(name, n, 0.0, 0.2)),
                });
            }
            continue;
        }
        let name = requirement.names[0].clone();
        if matches!(requirement.quant, QuantConstraint::I64) {
            // The n-gram index buffers must be mutually consistent with the table rows
            // (LOADED, never re-derived — SEMANTICS.md §PLE): sizes = the first
            // `ngram_heads` primes >= ngram_vocab_size_base (the census-shape math),
            // offsets their prefix sums; ids then stay inside the table.
            let q = cfg.qwen4exp.as_ref().ok_or("tiny config lost qwen4exp")?;
            let heads = q.ngram_heads() as usize;
            let ints: Vec<i64> = if name.ends_with("layer_multipliers") {
                (0..q.ngram_size as i64)
                    .map(|i| 1_000_003 + 2 * i * 31)
                    .collect()
            } else {
                let sizes = first_primes_at_least(q.ngram_vocab_size_base as i64, heads);
                if name.ends_with("ngram_heads_vocab_sizes") {
                    sizes
                } else {
                    let mut offsets = Vec::with_capacity(heads);
                    let mut total = 0i64;
                    for &size in &sizes {
                        offsets.push(total);
                        total += size;
                    }
                    offsets
                }
            };
            entries.push(StEntry {
                name,
                dtype: "I64",
                shape: requirement.shape.clone(),
                bytes: ints.iter().flat_map(|v| v.to_le_bytes()).collect(),
            });
            continue;
        }
        // Per-expert NVFP4 rows (arm D): the dialect contract declares them
        // QuantConstraint::Nvfp4 with LOGICAL shapes; write the modelopt triplet plus
        // the input_scale sibling the mint carries.
        if requirement.quant == QuantConstraint::Nvfp4 {
            let (out_f, in_f) = (requirement.shape[0], requirement.shape[1]);
            let stem = name
                .strip_suffix(".weight")
                .ok_or("NVFP4 row without .weight")?;
            entries.push(StEntry {
                name: name.clone(),
                dtype: "U8",
                shape: vec![out_f, in_f / 2],
                bytes: gen_bytes(&name, (out_f * in_f / 2) as usize, 0, 255),
            });
            entries.push(StEntry {
                name: format!("{stem}.weight_scale"),
                dtype: "F8_E4M3",
                shape: vec![out_f, in_f / 16],
                bytes: gen_bytes(
                    &format!("{name}.weight_scale"),
                    (out_f * in_f / 16) as usize,
                    0x28,
                    0x40,
                ),
            });
            entries.push(StEntry {
                name: format!("{stem}.weight_scale_2"),
                dtype: "F32",
                shape: vec![],
                // The REAL mint's macro class: modelopt amax-derived NON-pow2 (this is
                // the first value the fleet box measured, layers.0 down_proj). Gates the
                // macro-post-upcast f32 dequant chain against the host decoder; arm C
                // keeps a pow2 macro so both classes stay covered.
                bytes: 5.9945243e-5f32.to_le_bytes().to_vec(),
            });
            entries.push(StEntry {
                name: format!("{stem}.input_scale"),
                dtype: "F32",
                shape: vec![],
                bytes: 0.0078125f32.to_le_bytes().to_vec(), // validated, unused (W4A16)
            });
            continue;
        }
        // NVFP4 trunk expert banks (arm C): the gate_up half only — the tiny down bank's
        // in_f (= ff = 8) cannot carry modelopt's per-16 scale groups, so geometry keeps
        // it BF16 (the loader's halves mix freely; on the artifact ff = 640 and BOTH
        // halves ride the same code/scale/macro path this arm gates).
        let trunk_gate_up = matches!(
            (&requirement.id, requirement.owner),
            (
                TensorId::Layer { index, tensor },
                TensorOwner::Layer(_)
            ) if *index < n_trunk && *tensor == LayerTensor::MoeExpertGateUpBank
        );
        if kind == DirKind::Nvfp4Stacked && trunk_gate_up {
            let (n_expert, out_f, in_f) = (
                requirement.shape[0],
                requirement.shape[1],
                requirement.shape[2],
            );
            let codes = gen_bytes(&name, (n_expert * out_f * in_f / 2) as usize, 0, 255);
            // e4m3 scale bytes in [0x28, 0x40]: positive, non-NaN, ~[0.04, 2.0].
            let scales = gen_bytes(
                &format!("{name}.weight_scale"),
                (n_expert * out_f * in_f / 16) as usize,
                0x28,
                0x40,
            );
            entries.push(StEntry {
                name: name.clone(),
                dtype: "U8",
                shape: vec![n_expert, out_f, in_f / 2],
                bytes: codes,
            });
            entries.push(StEntry {
                name: format!("{name}.weight_scale"),
                dtype: "F8_E4M3",
                shape: vec![n_expert, out_f, in_f / 16],
                bytes: scales,
            });
            entries.push(StEntry {
                name: format!("{name}.weight_scale_2"),
                dtype: "F32",
                shape: vec![n_expert],
                bytes: vec![0.03125f32; n_expert as usize] // 2^-5: pow2 per the dequant law
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            });
            continue;
        }
        // Float rows: zero-centered norms (the loader folds +1), raw linear_attn.norm near
        // 1, A_log small-negative (post -exp decay in a sane band), 1/sqrt(in) projections.
        let n = elements(&requirement.shape);
        let (center, scale) = if name.ends_with("linear_attn.norm.weight") {
            (1.0, 0.1)
        } else if name.contains("norm") && name.ends_with(".weight") {
            (0.0, 0.2)
        } else if name.ends_with("linear_attn.A_log") {
            (-0.7, 0.3)
        } else if name.ends_with("linear_attn.dt_bias") {
            (0.0, 0.1)
        } else {
            let in_f = *requirement.shape.last().unwrap() as f32;
            (0.0, 1.0 / in_f.sqrt())
        };
        entries.push(StEntry {
            name: name.clone(),
            dtype: "BF16",
            shape: requirement.shape.clone(),
            bytes: bf16_bytes(&gen_f32(&name, n, center, scale)),
        });
    }
    write_safetensors(&dir.join("model.safetensors"), &entries)?;
    Ok(())
}

/// Scratch dir with drop cleanup (tmp hygiene: the task that creates it deletes it).
struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------- arm runner

struct ArmResult {
    prefill_worst: (f32, f32),
    decode_worst: (f32, f32),
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    label: &str,
    e: &Engine,
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    model: &Qwen4ExpGpu,
    prompt: &[u32],
    decode_feed: &[u32],
    lines: &mut Vec<String>,
    failures: &mut usize,
) -> Res<ArmResult> {
    const MAX_ABS: f32 = 0.01;
    const MAX_REL: f32 = 0.01;
    let n = prompt.len();
    let vocab = plan.vocab_size as usize;
    let all_tokens: Vec<u32> = prompt.iter().chain(decode_feed.iter()).copied().collect();
    // ONE full-sequence reference run — causality makes its row i the oracle for both the
    // prefill rows and every decode step.
    let reference = execute(plan, weights, &all_tokens)?;
    let mut state = model.alloc_state(e, all_tokens.len())?;
    let mut record = |phase: &str, row: usize, stats: &RowStats, failures: &mut usize| {
        let passed = stats.max_abs <= MAX_ABS && stats.max_rel <= MAX_REL && stats.argmax_match;
        if !passed {
            *failures += 1;
        }
        lines.push(format!(
            "{label}\t{phase}\trow={row}\tmax_abs={:.3e}\tmax_rel={:.3e}\tref_absmax={:.3e}\targmax_match={}\tpass={passed}",
            stats.max_abs, stats.max_rel, stats.ref_absmax, stats.argmax_match
        ));
    };

    let prefill = model.prefill(e, prompt, &mut state)?;
    assert_eq!(prefill.len(), n * vocab);
    let mut prefill_worst = (0.0f32, 0.0f32);
    for row in 0..n {
        let stats = compare_row(
            &reference.logits[row * vocab..(row + 1) * vocab],
            &prefill[row * vocab..(row + 1) * vocab],
        );
        prefill_worst.0 = prefill_worst.0.max(stats.max_abs);
        prefill_worst.1 = prefill_worst.1.max(stats.max_rel);
        record("prefill", row, &stats, failures);
    }

    let mut decode_worst = (0.0f32, 0.0f32);
    for (step, &token) in decode_feed.iter().enumerate() {
        let logits = model.decode_step(e, token, &mut state)?;
        assert_eq!(logits.len(), vocab);
        let row = n + step;
        let stats = compare_row(&reference.logits[row * vocab..(row + 1) * vocab], &logits);
        decode_worst.0 = decode_worst.0.max(stats.max_abs);
        decode_worst.1 = decode_worst.1.max(stats.max_rel);
        record("decode-step", row, &stats, failures);
    }
    Ok(ArmResult {
        prefill_worst,
        decode_worst,
    })
}

/// MTP draft-forward parity arm (mtp-spec lane): the engine draft — one BATCHED pass
/// (the replay shape) AND a fresh single-step chain (the K-step decode shape) — vs the
/// reference MTP arm, both fed the REFERENCE trunk wide state (pure draft-program
/// comparison; `pos_off = 0` matches the reference's row-indexed positions). Logits
/// rows take the modelplan policy (abs/rel/argmax); the wide CARRIER (the K>1 seed)
/// takes abs/rel.
#[allow(clippy::too_many_arguments)]
fn run_mtp_arm(
    label: &str,
    e: &Engine,
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    model: &Qwen4ExpGpu,
    tokens: &[u32],
    lines: &mut Vec<String>,
    failures: &mut usize,
) -> Res<String> {
    const MAX_ABS: f32 = 0.01;
    const MAX_REL: f32 = 0.01;
    let vocab = plan.vocab_size as usize;
    let t = tokens.len();
    let reference = execute(plan, weights, tokens)?;
    let mtp_ref = reference
        .mtp
        .first()
        .ok_or("reference produced no MTP output")?;
    let trunk_wide = reference
        .layer_hidden
        .last()
        .ok_or("reference produced no trunk hidden")?;
    let wide = trunk_wide.len() / t;
    let wide_dev = e.htod(trunk_wide)?;
    let mut worst = (0.0f32, 0.0f32);
    let record = |phase: &str,
                  row: usize,
                  stats: &RowStats,
                  with_argmax: bool,
                  worst: &mut (f32, f32),
                  failures: &mut usize,
                  lines: &mut Vec<String>| {
        let passed = stats.max_abs <= MAX_ABS
            && stats.max_rel <= MAX_REL
            && (!with_argmax || stats.argmax_match);
        if !passed {
            *failures += 1;
        }
        worst.0 = worst.0.max(stats.max_abs);
        worst.1 = worst.1.max(stats.max_rel);
        lines.push(format!(
            "{label}\t{phase}\trow={row}\tmax_abs={:.3e}\tmax_rel={:.3e}\tref_absmax={:.3e}\targmax_match={}\tpass={passed}",
            stats.max_abs, stats.max_rel, stats.ref_absmax, stats.argmax_match
        ));
    };

    // Batched pass (the replay shape).
    let mut batched = model.mtp_state(e, t + 1)?;
    let (logits_d, carrier_d) =
        model.mtp_draft_forward(e, tokens, &wide_dev, 0, &mut batched, 0, true)?;
    let logits = e.dtoh_view(&logits_d.slice(0..t * vocab))?;
    let carrier = e.dtoh_view(&carrier_d.slice(0..t * wide))?;
    for row in 0..t {
        let stats = compare_row(
            &mtp_ref.logits[row * vocab..(row + 1) * vocab],
            &logits[row * vocab..(row + 1) * vocab],
        );
        record(
            "mtp-batched",
            row,
            &stats,
            true,
            &mut worst,
            failures,
            lines,
        );
    }
    for row in 0..t {
        let stats = compare_row(
            &mtp_ref.hidden[row * wide..(row + 1) * wide],
            &carrier[row * wide..(row + 1) * wide],
        );
        record(
            "mtp-carrier",
            row,
            &stats,
            false,
            &mut worst,
            failures,
            lines,
        );
    }

    // Single-step chain (the K-step decode shape) on a FRESH draft state: the draft
    // cache-vs-full invariance.
    let mut chained = model.mtp_state(e, t + 1)?;
    for row in 0..t {
        let (ld, cd) = model.mtp_draft_forward(
            e,
            &tokens[row..row + 1],
            &wide_dev,
            row,
            &mut chained,
            0,
            true,
        )?;
        let step_logits = e.dtoh_view(&ld.slice(0..vocab))?;
        let stats = compare_row(
            &mtp_ref.logits[row * vocab..(row + 1) * vocab],
            &step_logits,
        );
        record("mtp-step", row, &stats, true, &mut worst, failures, lines);
        model.mtp_recycle(&mut chained, ld, cd);
    }
    Ok(format!(
        "{label}: draft parity worst abs {:.3e} rel {:.3e} over batched+carrier+steps",
        worst.0, worst.1
    ))
}

/// The DEFERRED-round identity arm (mtp11): plain greedy vs the host chain vs the
/// deferred chain, at pmin 0 and under the p-min guard (both deferred guard arms),
/// optionally with a REVERSED full-width trim (the trim-rank chain table + d2t map path).
/// Every spec run must equal plain BYTE FOR BYTE and every arm of a config must agree
/// on the admission counters (rounds, drafted, accepted, guard_stops, zero_draft) —
/// the deferred round is the same picks by construction, so any drift is a failure.
#[allow(clippy::too_many_arguments)]
fn run_defer_arm(
    label: &str,
    e: &Engine,
    model: &mut Qwen4ExpGpu,
    vocab: usize,
    prompt: &[u32],
    lines: &mut Vec<String>,
    failures: &mut usize,
    with_trim: bool,
) -> Res<String> {
    use memra_engine::qwen4exp_gpu::SpecOpts;
    let max_new = 24usize;
    let spec_k = 3usize;
    let cap = prompt.len() + max_new + spec_k + 4;
    // Plain greedy chain: the byte-identity oracle.
    let mut plain_state = model.alloc_state(e, cap)?;
    let logits = model.prefill(e, prompt, &mut plain_state)?;
    let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]) as u32;
    let mut plain = vec![next];
    for _ in 1..max_new {
        let row = model.decode_step(e, next, &mut plain_state)?;
        next = argmax(&row) as u32;
        plain.push(next);
    }
    let mut checked = 0usize;
    let check_config_k = |model: &Qwen4ExpGpu,
                          sk: usize,
                          cfg_name: &str,
                          arms: &[(&str, SpecOpts)],
                          lines: &mut Vec<String>,
                          failures: &mut usize|
     -> Res<()> {
        let mut base: Option<(Vec<u32>, (usize, u64, u64, usize, usize))> = None;
        for (arm_name, opts) in arms {
            let scap = prompt.len() + max_new + sk + 4;
            let mut ss = model.alloc_state(e, scap)?;
            let mut ds = model.mtp_state(e, scap)?;
            let report = model.spec_generate_ext(
                e, e, prompt, max_new, sk, &mut ss, &mut ds, None, *opts, None,
            )?;
            let counters = (
                report.rounds,
                report.drafted,
                report.accepted,
                report.guard_stops,
                report.zero_draft_rounds,
            );
            let vs_plain = report.tokens == plain;
            let vs_base = match base.as_ref() {
                Some((toks, ctrs)) => &report.tokens == toks && counters == *ctrs,
                None => true,
            };
            let pass = vs_plain && vs_base;
            if !pass {
                *failures += 1;
            }
            lines.push(format!(
                "{label}\t{cfg_name}\t{arm_name}\tk={sk}\ttokens={max_new}\trounds={}\tdrafted={}\taccepted={}\tguard_stops={}\tzero_draft={}\tvs_plain={vs_plain}\tvs_host={vs_base}\tpass={pass}",
                report.rounds,
                report.drafted,
                report.accepted,
                report.guard_stops,
                report.zero_draft_rounds,
            ));
            if base.is_none() {
                base = Some((report.tokens, counters));
            }
        }
        Ok(())
    };
    let check_config =
        |model: &Qwen4ExpGpu,
         cfg_name: &str,
         arms: &[(&str, SpecOpts)],
         lines: &mut Vec<String>,
         failures: &mut usize|
         -> Res<()> { check_config_k(model, spec_k, cfg_name, arms, lines, failures) };
    let host = SpecOpts::default();
    let defer = SpecOpts {
        defer: true,
        ..Default::default()
    };
    // No-guard config: host vs deferred.
    model.arm_spec_devchain(e)?;
    check_config(
        model,
        "pmin0",
        &[("host", host), ("defer", defer)],
        lines,
        failures,
    )?;
    checked += 2;
    let g = |pmin: f32, defer: bool, gsync: bool| SpecOpts {
        pmin,
        defer,
        defer_guard_sync: gsync,
        ..Default::default()
    };
    // All-stop guard config (every round zero-draft): the PMIN0 zero-draft semantics.
    check_config(
        model,
        "pmin0.5",
        &[
            ("host", g(0.5, false, false)),
            ("defer", g(0.5, true, false)),
            ("defer-gsync", g(0.5, true, true)),
        ],
        lines,
        failures,
    )?;
    checked += 3;
    // MID-CHAIN guard config: find a (k, pmin) whose host run stops at some j >= 1
    // (guard_stops > zero_draft rounds proves a mid-chain stop) — the post-hoc
    // truncation's semantic risk. A gate that never reaches j >= 1 would be a wiring
    // assertion, so the absence of such a point is a FAILURE, not a skip.
    let mut mixed = None;
    'sweep: for &sk in &[spec_k, 6usize, 2] {
        for &p in &[
            0.4f32, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02, 0.01, 0.002,
        ] {
            let scap = prompt.len() + max_new + sk + 4;
            let mut ss = model.alloc_state(e, scap)?;
            let mut ds = model.mtp_state(e, scap)?;
            let r = model.spec_generate_ext(
                e,
                e,
                prompt,
                max_new,
                sk,
                &mut ss,
                &mut ds,
                None,
                g(p, false, false),
                None,
            )?;
            if r.guard_stops > r.zero_draft_rounds && r.drafted > 0 {
                mixed = Some((sk, p));
                break 'sweep;
            }
        }
    }
    match mixed {
        Some((sk, p)) => {
            let name = format!("k{sk}+pmin{p}-midchain");
            check_config_k(
                model,
                sk,
                &name,
                &[
                    ("host", g(p, false, false)),
                    ("defer", g(p, true, false)),
                    ("defer-gsync", g(p, true, true)),
                ],
                lines,
                failures,
            )?;
            checked += 3;
        }
        None => {
            // The deterministic fixture's intra-round confidence never crosses a
            // threshold its first pick passed, so a mid-chain INTEGRATION stop is
            // unreachable here. Coverage for that walk = the guard-trunc pin below
            // (arbitrary windows incl. dips) + the real-model `--defer-ab` counter
            // identity, where the ship pmin stops mid-chain constantly. Stated, not
            // silently skipped.
            lines.push(format!(
                "{label}\tmidchain-pmin\tNONE\tcovered_by=guard-trunc-pin+box-defer-ab\t(no swept (k,pmin) mixes on the deterministic fixture)"
            ));
        }
    }
    // The deferred guard's truncation walk pinned on windows the fixture cannot
    // produce: predicate (strict p < pmin, boundary passes) and index must match the
    // host chain's sequential stop rule exactly.
    {
        use memra_engine::qwen4exp_gpu::spec_guard_trunc;
        let cases: &[(&[f32], f32, usize, &str)] = &[
            (
                &[0.9, 0.9, 0.1, 0.9],
                0.3,
                2,
                "mid-chain dip, later recovery ignored",
            ),
            (&[0.1, 0.9, 0.9], 0.3, 0, "zero-draft round"),
            (&[0.9, 0.8, 0.7], 0.3, 3, "no stop"),
            (
                &[0.3, 0.29],
                0.3,
                1,
                "boundary: p == pmin passes (strict <)",
            ),
            (&[], 0.3, 0, "empty window"),
            (&[0.5, 0.4, 0.3, 0.2], 0.35, 2, "monotone decay"),
        ];
        let mut ok = true;
        for (probs, pmin, want, what) in cases {
            let got = spec_guard_trunc(probs, *pmin);
            if got != *want {
                ok = false;
                *failures += 1;
                lines.push(format!(
                    "{label}\tguard-trunc-pin\t{what}\tgot={got}\twant={want}\tpass=false"
                ));
            }
        }
        if ok {
            lines.push(format!(
                "{label}\tguard-trunc-pin\t{} windows incl. mid-chain dip\tpass=true",
                cases.len()
            ));
            checked += 1;
        }
    }
    if with_trim {
        // Reversed FULL-WIDTH trim: d2t[i] = vocab-1-i, a non-identity permutation of
        // every row — the trim-rank chain table and the drain's d2t map both move.
        let rev: Vec<u32> = (0..vocab as u32).rev().collect();
        model.build_draft_trim(e, &rev)?;
        model.arm_spec_devchain(e)?;
        let p = mixed.map(|(_, p)| p).unwrap_or(0.5);
        check_config(
            model,
            &format!("trim-rev+pmin{p}"),
            &[
                ("host", g(p, false, false)),
                ("defer", g(p, true, false)),
                ("defer-gsync", g(p, true, true)),
            ],
            lines,
            failures,
        )?;
        checked += 3;
        model.clear_draft_trim();
    }
    model.clear_spec_devchain();
    Ok(format!(
        "{label}: deferred-round identity over {checked} arms (byte identity vs plain + counter identity vs host, incl. a mid-chain guard stop)"
    ))
}

fn main() -> Res<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // `--write-bank-goldens` MINTS the bank-byte fingerprints instead of checking them.
    // It exists to be run on the PRE-STREAMING loader (see the bank-bytes arm); running it
    // on the streaming loader would rubber-stamp whatever that loader produces, so the
    // receipt records which build minted the file.
    let write_bank_goldens = argv.iter().any(|a| a == "--write-bank-goldens");
    let receipt_path = argv
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .ok_or("usage: qwen4exp-gpu-gate <receipt.tsv> [--write-bank-goldens]")?;
    let pack =
        memra_gguf::model_packs::by_alias("qwen4_exp").ok_or("qwen4_exp pack is not registered")?;
    let plan = pack.compile_tiny_plan()?;
    let cfg = ModelConfig::from_hf(&HfConfig::parse(TINY_CONFIG));
    // The embedded tiny config must stay the pack's tiny config — drift fails loudly.
    assert_eq!(
        pack.compile_plan(&cfg)?,
        plan,
        "gate tiny config drifted from the pack tiny_plan"
    );

    // Token program (vocab 64, eos 63): 18-token prompt with a mid-prompt EOS, 7 decode
    // steps with a SECOND EOS (PLE n-gram segment reset must also hold incrementally). At
    // position p the tiny indexer sees (p+1)/4 complete blocks vs budget 2, so every
    // position past 11 exercises real block dropping.
    let prompt: Vec<u32> = vec![
        1, 7, 13, 2, 41, 9, 30, 5, 22, 63, 11, 3, 47, 8, 19, 26, 4, 35,
    ];
    let decode_feed: Vec<u32> = vec![12, 55, 63, 6, 28, 40, 17];

    // Gate instrumentation: force not-yet-default seams for correctness runs
    // (MEMRA_Q4E_SEAMS, flags law — receipts precede the default flip).
    memra_engine::qwen4exp_gpu::apply_env_seams();
    // Reference-parity arms compare against the exact f32 oracle, so they run the
    // EXACTNESS-INSTRUMENT cache arms (kvq off, idxq f32) regardless of the serving
    // defaults — the quantized-cache paths carry their own dedicated arms below and
    // in the kvq lane's real-gate batteries (KVQ-CELL.md). Without this pin the
    // ship-default flip (kvq ON, idxq q8) leaks cross-config quant drift into
    // same-config gates: tiny margins brushed the 0.01 policy and flipped one
    // near-tie argmax (receipt /tmp/q48fn-tinygate-flip.tsv, 2026-08-31).
    // An explicit MEMRA_Q4E_SEAMS entry still wins (armed correctness runs).
    let seams_env = std::env::var("MEMRA_Q4E_SEAMS").unwrap_or_default();
    if !seams_env.contains("kvq") {
        memra_engine::qwen4exp_gpu::set_kv_quant(false);
    }
    if !seams_env.contains("idxq") {
        memra_engine::qwen4exp_gpu::set_idxq("f32");
    }
    let mut lines: Vec<String> = Vec::new();
    let mut failures = 0usize;
    let mut summaries: Vec<String> = Vec::new();

    // Arm BANK-BYTES: the loader's MEMORY-ORDERING gate (issue #48). The streaming loader
    // reads each layer's expert bank off the safetensors mmap inside the upload loop
    // instead of pre-materializing all 48 layers on host, which is what OOM-killed the
    // real gate at 179.7 GB anon-RSS on a 180 GB-RAM box. Moving WHEN bytes are read must
    // not move WHICH bytes: this arm digests every bank projection and compares against
    // goldens MINTED FROM THE PRE-STREAMING LOADER on the same deterministic fixtures
    // (`bank-bytes-goldens.tsv`, banked next to this receipt; mint procedure in
    // research/qwen4exp-bringup-20260829/loader/LOADER-STREAM.md).
    //
    // Coverage is every branch of the bank read: fused f32 (dequantized), fused raw bf16
    // (`host_bf16_banks`), fused NVFP4 code/scale row split, the MTP bank's DeviceBf16
    // residency at index >= n_trunk, per-expert modelopt stacking, and — with a 12-expert
    // variant config — the EXPERT ORDER the lexicographic census trap attacks
    // (`experts.10` sorts before `experts.2`, so an arrival-ordered stack silently
    // mis-assigns experts on any E > 9; the tiny plan's E=8 cannot see it).
    {
        let goldens_path = Path::new(&receipt_path)
            .parent()
            .unwrap_or(Path::new("."))
            .join("bank-bytes-goldens.tsv");
        // A 12-expert twin of the tiny config: same everything else, E > 9 so the
        // per-expert name set sorts lexicographically out of numeric order.
        let cfg12_json = TINY_CONFIG.replace("\"num_experts\":8", "\"num_experts\":12");
        assert!(
            cfg12_json != TINY_CONFIG,
            "tiny config lost its num_experts row — the E>9 order pin is not being built"
        );
        let arms: [(&str, DirKind, LoadOptions, &str); 5] = [
            (
                "fused-f32",
                DirKind::Bf16Fused,
                LoadOptions::default(),
                TINY_CONFIG,
            ),
            (
                "fused-hostbf16",
                DirKind::Bf16Fused,
                LoadOptions {
                    host_bf16_banks: true,
                    ..Default::default()
                },
                TINY_CONFIG,
            ),
            (
                "fused-mtp-devbf16",
                DirKind::Bf16Fused,
                LoadOptions {
                    load_mtp: true,
                    ..Default::default()
                },
                TINY_CONFIG,
            ),
            (
                "nvfp4-stacked",
                DirKind::Nvfp4Stacked,
                LoadOptions::default(),
                TINY_CONFIG,
            ),
            (
                "nvfp4-perexpert-e12",
                DirKind::Nvfp4PerExpert,
                LoadOptions::default(),
                cfg12_json.as_str(),
            ),
        ];
        let mut measured: Vec<String> = Vec::new();
        for (label, kind, opts, arm_config) in arms {
            let acfg = ModelConfig::from_hf(&HfConfig::parse(arm_config));
            let aplan = pack.compile_plan(&acfg)?;
            let dir = TempDir(std::env::temp_dir().join(format!(
                "qwen4exp-gate-bankbytes-{label}-{}",
                std::process::id()
            )));
            synthesize_dir(&dir.0, &acfg, &aplan, kind)?;
            // `synthesize_dir` always writes the pack's tiny config; the E=12 arm needs the
            // dir's config.json to describe the dir the loader is about to re-derive its
            // contract from, or the loader binds an 8-expert contract to 12-expert rows.
            std::fs::write(dir.0.join("config.json"), arm_config)?;
            let checkpoint = read_checkpoint_with(&dir.0, opts)?;
            let prints = checkpoint.bank_fingerprints()?;
            if prints.is_empty() {
                return Err(format!("bank-bytes {label}: loader produced no banks").into());
            }
            for print in prints {
                measured.push(format!(
                    "{label}\t{}\t{}\t{}\t{}\t{}",
                    print.layer, print.projection, print.kind, print.bytes, print.digest
                ));
            }
        }
        if write_bank_goldens {
            let mut body = String::from(
                "# qwen4exp bank byte fingerprints — sha256 over each expert-bank \
                 projection's uploaded payload\n# arm\tlayer\tprojection\tkind\tbytes\tsha256\n",
            );
            for row in &measured {
                body.push_str(row);
                body.push('\n');
            }
            std::fs::write(&goldens_path, body)?;
            println!(
                "bank-bytes goldens written: {} ({} rows)",
                goldens_path.display(),
                measured.len()
            );
            // Minting is a GPU-free operation on purpose: it runs on the pre-streaming
            // build, whose only job here is to state what the bytes WERE.
            return Ok(());
        }
        let banked = std::fs::read_to_string(&goldens_path).map_err(|error| {
            format!(
                "bank-bytes goldens {} unreadable ({error}) — mint them with \
                 --write-bank-goldens on the PRE-STREAMING loader, never on this one",
                goldens_path.display()
            )
        })?;
        let expected: Vec<&str> = banked
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .collect();
        let mut mismatches = 0usize;
        if expected.len() != measured.len() {
            mismatches += 1;
            lines.push(format!(
                "bank-bytes\trows_banked={}\trows_measured={}\tpass=false",
                expected.len(),
                measured.len()
            ));
        }
        for (index, row) in measured.iter().enumerate() {
            let want = expected.get(index).copied().unwrap_or("<missing>");
            if want != row.as_str() {
                mismatches += 1;
                lines.push(format!(
                    "bank-bytes\tbanked={want}\tmeasured={row}\tpass=false"
                ));
            }
        }
        if mismatches > 0 {
            failures += mismatches;
        }
        lines.push(format!(
            "bank-bytes\tarms=5\trows={}\tmismatches={mismatches}\tpass={}",
            measured.len(),
            mismatches == 0
        ));
        summaries.push(format!(
            "bank-bytes: {} bank projections across 5 loader arms {} the pre-streaming \
             goldens ({} mismatches)",
            measured.len(),
            if mismatches == 0 {
                "BYTE-IDENTICAL to"
            } else {
                "DIVERGED from"
            },
            mismatches
        ));
    }

    let engine = Engine::new(0)?;

    // Arm 0: the grouped-decode kernel's own oracle (kernel vs host decoder chain). The
    // tiny arms below cannot reach `qmatvec_nvfp4_modelopt_sel_f32` — the tiny down
    // projection is BF16 by geometry, so the grouped MoE path never engages at this
    // scale — this synthetic arm gates the kernel directly (perf lane, PROFILE-0).
    summaries.push(memra_engine::qwen4exp_gpu::gate_nvfp4_sel_matvec(&engine)?);

    // Arm 0a2: the SUB-WARP pair-group sel kernels (`selgroup`, downsel lane) at the REAL
    // MoE geometry. Arm 0 above runs tiny shapes where `pairs` is 1 or 2; the defect this
    // seam fixes only exists at pairs=20 (down, lanes 20-31 idle) and pairs=80 (gate/up
    // tail), so the arm has to carry the real widths. Also pins the degenerate
    // (g=32, rows=4) shape BIT-IDENTICAL to the shipped v3 / gufuse kernels, which is what
    // makes the seam a rollback.
    summaries.push(memra_engine::qwen4exp_gpu::gate_nvfp4_sel_group(&engine)?);

    // Arm 0b: the bf16 trunk matvec kernel oracle (kernel vs host f32 matvec over
    // identical bf16-widened weights). The fixture arm's random f32 weights fail the
    // representability guard (bf16 twins skipped there by value, and the tiny
    // hc_lowrank=4 fails the %8 geometry guard for the up projection everywhere), so
    // the dir arms exercise the resident-path plumbing and THIS arm gates the kernel's
    // arithmetic directly, including the batch/stride modes the tiny dims never reach.
    summaries.push(memra_engine::qwen4exp_gpu::gate_qmatvec_bf16(&engine)?);

    // Arm 0c: the hcmicro kernels at REAL geometry (streams 4, hidden 2560) vs the
    // classic composition — the tiny plan cannot reach that shape (perf7 incident).
    summaries.push(memra_engine::qwen4exp_gpu::gate_hc_micro_kernels(&engine)?);

    // Arm 0d: the perf-round-3 GDN kernels at REAL geometry — the decode-step scan twin
    // vs the naive scan (tolerance, accumulation class) and the fused norm+gate vs its
    // chain (bit-identity). The tiny plan (hk 4) cannot reach the step twin's warp guard.
    summaries.push(memra_engine::qwen4exp_gpu::gate_gdn_step_kernels(&engine)?);

    // Arm 0e: the round-4 hyper-gate diet at REAL geometry (streams 4, hidden 2560,
    // rank 320) vs the classic fused chain — the tiny plan's rank 4 fails the diet's
    // %8 geometry guard, so only this arm reaches the three diet kernels.
    summaries.push(memra_engine::qwen4exp_gpu::gate_hc_diet_kernels(&engine)?);

    // Arm 0f: the device-router oracle at REAL geometry (devtwin lane, 512 experts /
    // top-10) — the tiny plan's bank never reaches the grouped NVFP4 dispatch, so this
    // arm gates `qwen4exp_route_topk_f32` directly against `host_route_softmax_topk`:
    // selection ids + emitted order EXACT (hard fail), weights ULP-bounded, tie
    // batteries (boundary-straddling duplicates / all-equal / underflow) included.
    summaries.push(memra_engine::qwen4exp_gpu::gate_route_kernel(&engine)?);

    // Arm 0h: the kvq/idxq kernel oracles at REAL geometry (kv_dim 512, idx_dim 128 +
    // padded-tail widths) — append-quantize vs host twins (byte), dequant vs host twins
    // (bit), the FUSED quantized block-list attention vs the dequant-rows composition
    // (bit — the storage contract), and the indexer q8/bf16 appenders vs the host cache
    // twins (the idxcache interleave contract).
    summaries.push(memra_engine::qwen4exp_gpu::gate_kvq_kernels(&engine)?);

    // Arm A: deterministic tiny fixture.
    let fixture = deterministic_fixture(&plan)?;
    let mut model = Qwen4ExpGpu::from_reference_weights(&engine, &plan, &fixture.weights)?;
    let result = run_arm(
        "fixture",
        &engine,
        &plan,
        &fixture.weights,
        &model,
        &prompt,
        &decode_feed,
        &mut lines,
        &mut failures,
    )?;
    summaries.push(format!(
        "fixture: prefill worst abs {:.3e} rel {:.3e}; decode worst abs {:.3e} rel {:.3e}",
        result.prefill_worst.0,
        result.prefill_worst.1,
        result.decode_worst.0,
        result.decode_worst.1
    ));

    // Arm E: MTP draft parity on the fixture (F32 device draft bank).
    summaries.push(run_mtp_arm(
        "mtp-fixture",
        &engine,
        &plan,
        &fixture.weights,
        &model,
        &prompt,
        &mut lines,
        &mut failures,
    )?);

    // Arms B/C: synthesized tiny checkpoint dirs through the full loader.
    for (label, kind) in [
        ("dir-bf16", DirKind::Bf16Fused),
        ("dir-nvfp4-stacked", DirKind::Nvfp4Stacked),
        ("dir-nvfp4-perexpert", DirKind::Nvfp4PerExpert),
    ] {
        let dir = TempDir(
            std::env::temp_dir().join(format!("qwen4exp-gate-{label}-{}", std::process::id())),
        );
        synthesize_dir(&dir.0, &cfg, &plan, kind)?;
        let checkpoint = read_checkpoint(&dir.0)?;
        assert_eq!(
            checkpoint.plan, plan,
            "dir loader compiled a different plan"
        );
        let model = Qwen4ExpGpu::load_from_dir(&engine, &dir.0)?;
        let reference_weights = checkpoint.into_reference_weights()?;
        // The loader materializes the TRUNK only (MTP execution is deferred), so the
        // reference twin runs the trunk-only plan — MTP is a post-trunk side branch and
        // the trunk logits are identical with or without it.
        let mut trunk_plan = plan.clone();
        trunk_plan.mtp_blocks.clear();
        let result = run_arm(
            label,
            &engine,
            &trunk_plan,
            &reference_weights,
            &model,
            &prompt,
            &decode_feed,
            &mut lines,
            &mut failures,
        )?;
        summaries.push(format!(
            "{label}: prefill worst abs {:.3e} rel {:.3e}; decode worst abs {:.3e} rel {:.3e}",
            result.prefill_worst.0,
            result.prefill_worst.1,
            result.decode_worst.0,
            result.decode_worst.1
        ));
    }

    // Arm DIET (trunk_f32_diet): on a dir-bf16 model (twins resident), freeing the f32
    // originals must change NOTHING the bf16 paths compute — prefill+decode rows
    // BIT-IDENTICAL pre/post diet, freed > 0. (The fixture arm's random f32 weights
    // fail the bf16 representability guard, so only the dir model reaches the diet.)
    {
        let dir = TempDir(
            std::env::temp_dir().join(format!("qwen4exp-gate-diet-dir-{}", std::process::id())),
        );
        synthesize_dir(&dir.0, &cfg, &plan, DirKind::Bf16Fused)?;
        let mut dmodel = Qwen4ExpGpu::load_from_dir(&engine, &dir.0)?;
        let run_rows = |m: &Qwen4ExpGpu| -> Res<Vec<f32>> {
            let mut state = m.alloc_state(&engine, prompt.len() + decode_feed.len() + 2)?;
            let mut rows = m.prefill(&engine, &prompt, &mut state)?;
            for &tok in &decode_feed {
                rows.extend(m.decode_step(&engine, tok, &mut state)?);
            }
            Ok(rows)
        };
        let before = run_rows(&dmodel)?;
        let freed = dmodel.trunk_f32_diet(&engine)?;
        let after = run_rows(&dmodel)?;
        let identical = before
            .iter()
            .zip(&after)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let pass = identical && freed > 0;
        if !pass {
            failures += 1;
        }
        lines.push(format!(
            "trunk-diet\tfreed_bytes={freed}\tbits_identical={identical}\tpass={pass}"
        ));
        summaries.push(format!(
            "trunk-diet: dir-bf16 pre/post rows {} (freed {} bytes)",
            if identical {
                "BIT-IDENTICAL"
            } else {
                "DIVERGED"
            },
            freed
        ));
    }

    // Arm F: the dir-bf16 loader path WITH load_mtp — the mtp.* namespace through the
    // contract into engine-resident draft weights (DeviceBf16 expert bank), gated
    // against the reference MTP arm on the loader's own materialization.
    {
        let dir = TempDir(
            std::env::temp_dir().join(format!("qwen4exp-gate-mtp-dir-{}", std::process::id())),
        );
        synthesize_dir(&dir.0, &cfg, &plan, DirKind::Bf16Fused)?;
        let opts = LoadOptions {
            load_mtp: true,
            ..Default::default()
        };
        let mut model = Qwen4ExpGpu::load_from_dir_with(&engine, &dir.0, opts)?;
        assert!(model.has_mtp(), "load_mtp did not materialize the draft");
        let reference_weights = read_checkpoint_with(&dir.0, opts)?.into_reference_weights()?;
        summaries.push(run_mtp_arm(
            "mtp-dir-bf16",
            &engine,
            &plan,
            &reference_weights,
            &model,
            &prompt,
            &mut lines,
            &mut failures,
        )?);
        // mtp11: the deferred round on the dir-bf16 model — its embed dequants from
        // bf16 bytes, so this is the bf16 BIT-CLEAN chain-table path (the fixture's
        // random f32 embeds exercise the f32 fallback in arm G2 below).
        summaries.push(run_defer_arm(
            "mtp-spec-defer-dirbf16",
            &engine,
            &mut model,
            plan.vocab_size as usize,
            &prompt,
            &mut lines,
            &mut failures,
            false,
        )?);
    }

    // Arm G: SPEC decode on the fixture — the exactness law at tiny scale: the spec
    // loop's output must equal the plain greedy chain TOKEN FOR TOKEN (verify chunks
    // run the per-token-exact row programs; the tiny plan exercises the fused-chain
    // and naive-scan fallbacks of the exact dispatch, and the K>1 carrier chain).
    {
        let vocab = plan.vocab_size as usize;
        let max_new = 24usize;
        let spec_k = 3usize;
        let cap = prompt.len() + max_new + spec_k + 4;
        let mut plain_state = model.alloc_state(&engine, cap)?;
        let logits = model.prefill(&engine, &prompt, &mut plain_state)?;
        let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]) as u32;
        let mut plain = vec![next];
        for _ in 1..max_new {
            let row = model.decode_step(&engine, next, &mut plain_state)?;
            next = argmax(&row) as u32;
            plain.push(next);
        }
        let mut spec_state = model.alloc_state(&engine, cap)?;
        let mut draft_state = model.mtp_state(&engine, cap)?;
        let report = model.spec_generate(
            &engine,
            &prompt,
            max_new,
            spec_k,
            &mut spec_state,
            &mut draft_state,
            None,
        )?;
        let matched = report.tokens == plain;
        if !matched {
            failures += 1;
        }
        lines.push(format!(
            "mtp-spec-tiny\tbyte-identity\tk={spec_k}\ttokens={max_new}\trounds={}\taccepted={}/{}\tspec={:?}\tplain={:?}\tpass={matched}",
            report.rounds, report.accepted, report.drafted, report.tokens, plain
        ));
        summaries.push(format!(
            "mtp-spec-tiny: spec-vs-plain byte-identity {} (k={spec_k}, {max_new} tokens, \
             accepted {}/{} over {} rounds)",
            if matched { "PASS" } else { "FAIL" },
            report.accepted,
            report.drafted,
            report.rounds
        ));
    }

    // Arm G2 (mtp11): the DEFERRED round on the fixture — byte identity vs plain and
    // counter identity vs the host chain, at pmin 0 and under the guard (post-hoc +
    // sequential sub-arm), plus the reversed full-width trim (trim-rank table + d2t
    // map). The fixture's random f32 embeds take the f32 fallback table.
    summaries.push(run_defer_arm(
        "mtp-spec-defer",
        &engine,
        &mut model,
        plan.vocab_size as usize,
        &prompt,
        &mut lines,
        &mut failures,
        true,
    )?);

    // Arm G3 (mtp11): SHORT-prompt armed-prefill BIT pin — a prompt that fits inside
    // k_cap must prefill through the SAME fused program armed or unarmed (the gen-157
    // latent defect's structural statement: the exact-row programs are for verify
    // chunks at base > 0, never the prefill). Prefill both ways, decode one step with
    // the same fed token, bit-compare every row.
    {
        let vocab = plan.vocab_size as usize;
        let short = &prompt[0..3];
        let mut ua = model.alloc_state(&engine, short.len() + 4)?;
        let la = model.prefill(&engine, short, &mut ua)?;
        let mut aa = model.alloc_state(&engine, short.len() + 8)?;
        model.spec_arm(&engine, &mut aa, 4)?; // k_cap 4 >= prompt len 3: the edge
        let lb = model.prefill(&engine, short, &mut aa)?;
        let fed = argmax(&la[(short.len() - 1) * vocab..]) as u32;
        let ra = model.decode_step(&engine, fed, &mut ua)?;
        let rb = model.decode_step(&engine, fed, &mut aa)?;
        let pre_same =
            la.len() == lb.len() && la.iter().zip(&lb).all(|(a, b)| a.to_bits() == b.to_bits());
        let step_same =
            ra.len() == rb.len() && ra.iter().zip(&rb).all(|(a, b)| a.to_bits() == b.to_bits());
        let pass = pre_same && step_same;
        if !pass {
            failures += 1;
        }
        lines.push(format!(
            "mtp-armed-prefill-bit\tprompt_len=3\tk_cap=4\tprefill_bits={pre_same}\tstep_bits={step_same}\tpass={pass}"
        ));
        summaries.push(format!(
            "mtp-armed-prefill-bit: short-prompt armed vs unarmed prefill {} (prefill rows + 1 step, bitwise)",
            if pass { "BIT-IDENTICAL" } else { "DIVERGED" }
        ));
    }

    // Arm G4 (mtp12, `vfuse`): the FUSED verify program at the VERIFY-CHUNK shape.
    //
    // Acceptance policy for this arm is deliberately NOT byte identity, and the reason is
    // the owner law: greedy/bit exactness is the INSTRUMENT. The exact per-row arm stays
    // as the byte-identity gate arm; `vfuse` is a DIFFERENT program (fused dense mats,
    // fused hyper gate, chunk scan, m=t indexer/PLE projections) that is allowed to differ
    // in bits. What it is not allowed to do is differ in MEANING, so it is gated like the
    // reference class: tolerance vs the reference executor plus argmax per row.
    //
    // Three things this arm pins that no other arm can:
    // 1. the fused program EXECUTES at 1 < t <= k_cap (no geometry refusal from the
    //    kernels that were written for prefill widths),
    // 2. the argmax sink is fed on the fused arm (so a cost A/B compares programs and not
    //    readback sizes),
    // 3. `verify_rewind` REFUSES after a fused chunk, by name. That failure path is
    //    executed here rather than described, because the fused chunk scan leaves no
    //    per-column stash and a rewind that silently "worked" would corrupt a spec run.
    {
        let vocab = plan.vocab_size as usize;
        let seq: Vec<u32> = prompt.iter().chain(decode_feed.iter()).copied().collect();
        let reference = execute(&plan, &fixture.weights, &seq)?;
        let n = prompt.len();
        let chunk_len = 4usize;
        let chunk = &decode_feed[0..chunk_len];
        let chunk_rows = |fused: bool| -> Res<Vec<f32>> {
            memra_engine::qwen4exp_gpu::set_verify_fused(fused);
            let mut state = model.alloc_state(&engine, seq.len() + 2)?;
            model.spec_arm(&engine, &mut state, chunk_len + 1)?;
            let _ = model.prefill(&engine, &prompt, &mut state)?;
            let rows = model.prefill(&engine, chunk, &mut state)?;
            memra_engine::qwen4exp_gpu::set_verify_fused(false);
            Ok(rows)
        };
        let exact_rows = chunk_rows(false)?;
        let fused_rows = chunk_rows(true)?;
        // (1) + meaning: every fused row vs the REFERENCE under the modelplan policy.
        let mut worst = (0.0f32, 0.0f32);
        let mut argmax_rows = 0usize;
        let mut ok = true;
        // Arm-vs-arm delta, informational: how far the fused program sits from the exact
        // one. A thin-margin argmax flip here would be EXPECTED and reported, not hidden.
        let mut worst_vs_exact = 0.0f32;
        for row in 0..chunk_len {
            let stats = compare_row(
                &reference.logits[(n + row) * vocab..(n + row + 1) * vocab],
                &fused_rows[row * vocab..(row + 1) * vocab],
            );
            worst.0 = worst.0.max(stats.max_abs);
            worst.1 = worst.1.max(stats.max_rel);
            argmax_rows += usize::from(stats.argmax_match);
            ok &= stats.max_abs <= 0.01 && stats.max_rel <= 0.01 && stats.argmax_match;
            for (a, b) in exact_rows[row * vocab..(row + 1) * vocab]
                .iter()
                .zip(&fused_rows[row * vocab..(row + 1) * vocab])
            {
                worst_vs_exact = worst_vs_exact.max((a - b).abs());
            }
        }
        // (2) the argmax sink on the fused arm: 4t-byte device argmax == host argmax of
        // the same rows.
        memra_engine::qwen4exp_gpu::set_verify_fused(true);
        let mut sink_state = model.alloc_state(&engine, seq.len() + 2)?;
        model.spec_arm(&engine, &mut sink_state, chunk_len + 1)?;
        let _ = model.prefill(&engine, &prompt, &mut sink_state)?;
        model.set_verify_want_argmax(&mut sink_state, true)?;
        let _ = model.prefill(&engine, chunk, &mut sink_state)?;
        let sink = model.verify_argmax_rows(&sink_state)?.to_vec();
        let sink_want: Vec<u32> = (0..chunk_len)
            .map(|r| argmax(&fused_rows[r * vocab..(r + 1) * vocab]) as u32)
            .collect();
        let sink_ok = sink == sink_want;
        ok &= sink_ok;
        // (3) rewind after a fused chunk must refuse, and say why.
        let rewind_err = model
            .verify_rewind(&engine, &mut sink_state, 1)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        let refused = rewind_err.contains("vfuse");
        ok &= refused;
        memra_engine::qwen4exp_gpu::set_verify_fused(false);
        if !ok {
            failures += 1;
        }
        lines.push(format!(
            "mtp-vfuse\tt={chunk_len}\tk_cap={}\tmax_abs={:.3e}\tmax_rel={:.3e}\targmax={argmax_rows}/{chunk_len}\tworst_vs_exact={worst_vs_exact:.3e}\tsink_ok={sink_ok}\trewind_refused={refused}\tpass={ok}",
            chunk_len + 1,
            worst.0,
            worst.1,
        ));
        summaries.push(format!(
            "mtp-vfuse: fused verify chunk vs reference {} (argmax {argmax_rows}/{chunk_len}, \
             worst abs {:.3e} rel {:.3e}; vs exact arm {worst_vs_exact:.3e}); argmax sink \
             {}; rewind refusal {}",
            if ok { "PASS" } else { "FAIL" },
            worst.0,
            worst.1,
            if sink_ok { "fed" } else { "WRONG" },
            if refused { "loud" } else { "MISSING" },
        ));
    }

    // Arm H: verify-REWIND invariance on the fixture — chunk 4 tokens through the
    // exact verify path, rewind to keep ∈ {1,2,3}, then decode the remaining feed
    // step-by-step; every row (chunk rows 0..keep AND the post-rewind steps) must
    // match the full-sequence reference under the modelplan policy. Exercises the
    // GDN state/conv rebuild and the PLE history rebuild (the feed crosses an EOS
    // segment reset) — the partial-accept path the spec arm's 0-acceptance fixture
    // rounds cannot reach.
    {
        let vocab = plan.vocab_size as usize;
        let seq: Vec<u32> = prompt.iter().chain(decode_feed.iter()).copied().collect();
        let reference = execute(&plan, &fixture.weights, &seq)?;
        let n = prompt.len();
        let chunk_len = 4usize;
        for keep in 1..chunk_len {
            let mut state = model.alloc_state(&engine, seq.len() + 2)?;
            model.spec_arm(&engine, &mut state, chunk_len + 1)?;
            let _ = model.prefill(&engine, &prompt, &mut state)?;
            let chunk = &decode_feed[0..chunk_len];
            let chunk_logits = model.prefill(&engine, chunk, &mut state)?;
            let mut worst = (0.0f32, 0.0f32);
            let mut ok = true;
            for row in 0..keep {
                let stats = compare_row(
                    &reference.logits[(n + row) * vocab..(n + row + 1) * vocab],
                    &chunk_logits[row * vocab..(row + 1) * vocab],
                );
                worst.0 = worst.0.max(stats.max_abs);
                worst.1 = worst.1.max(stats.max_rel);
                ok &= stats.max_abs <= 0.01 && stats.max_rel <= 0.01 && stats.argmax_match;
            }
            model.verify_rewind(&engine, &mut state, keep)?;
            for (step, &token) in decode_feed[keep..].iter().enumerate() {
                let row = n + keep + step;
                let logits = model.decode_step(&engine, token, &mut state)?;
                let stats = compare_row(&reference.logits[row * vocab..(row + 1) * vocab], &logits);
                worst.0 = worst.0.max(stats.max_abs);
                worst.1 = worst.1.max(stats.max_rel);
                ok &= stats.max_abs <= 0.01 && stats.max_rel <= 0.01 && stats.argmax_match;
            }
            if !ok {
                failures += 1;
            }
            lines.push(format!(
                "mtp-rewind\tkeep={keep}\tmax_abs={:.3e}\tmax_rel={:.3e}\tpass={ok}",
                worst.0, worst.1
            ));
            summaries.push(format!(
                "mtp-rewind keep={keep}: chunk+rewind+decode vs reference worst abs {:.3e} rel {:.3e} ({})",
                worst.0,
                worst.1,
                if ok { "PASS" } else { "FAIL" }
            ));
        }
    }

    // Arm KVQ (kvq lane): the quantized QSA KV cache (K=q8_0/V=q5_1) on the fixture.
    // Tiny random weights make a cross-config tolerance claim meaningless (the real-
    // checkpoint envelope is the box gate), so these arms assert EXACT properties only:
    // determinism, spec-vs-plain byte identity same-config, and a REPORTED (finiteness-
    // gated) envelope vs the f32 twin.
    {
        use memra_engine::qwen4exp_gpu::{set_idxq, set_kv_quant};
        let vocab = plan.vocab_size as usize;
        let run_rows = |model: &Qwen4ExpGpu| -> Res<Vec<f32>> {
            let mut state = model.alloc_state(&engine, prompt.len() + decode_feed.len() + 2)?;
            let mut rows = model.prefill(&engine, &prompt, &mut state)?;
            for &tok in &decode_feed {
                rows.extend(model.decode_step(&engine, tok, &mut state)?);
            }
            Ok(rows)
        };
        // f32 twin first (default seams), then the quant arm twice (determinism).
        set_kv_quant(false);
        let f32_rows = run_rows(&model)?;
        set_kv_quant(true);
        let qa = run_rows(&model)?;
        let qb = run_rows(&model)?;
        set_kv_quant(false);
        let deterministic = qa.iter().zip(&qb).all(|(a, b)| a.to_bits() == b.to_bits());
        let finite = qa.iter().all(|x| x.is_finite());
        let mut worst = (0.0f32, 0.0f32);
        let mut argmax_flips = 0usize;
        let n_rows = qa.len() / vocab;
        for r in 0..n_rows {
            let s = compare_row(
                &f32_rows[r * vocab..(r + 1) * vocab],
                &qa[r * vocab..(r + 1) * vocab],
            );
            worst.0 = worst.0.max(s.max_abs);
            worst.1 = worst.1.max(s.max_rel);
            if !s.argmax_match {
                argmax_flips += 1;
            }
        }
        let pass = deterministic && finite;
        if !pass {
            failures += 1;
        }
        lines.push(format!(
            "kvq-fixture\tdeterministic={deterministic}\tfinite={finite}\tenvelope_abs={:.3e}\tenvelope_rel={:.3e}\targmax_flips={argmax_flips}/{n_rows}\tpass={pass}",
            worst.0, worst.1
        ));
        summaries.push(format!(
            "kvq-fixture: determinism {} / finite {}; envelope vs f32 twin abs {:.3e} rel {:.3e}, argmax flips {argmax_flips}/{n_rows} (REPORT — the hard cross-config gate is the real checkpoint)",
            if deterministic { "PASS" } else { "FAIL" },
            if finite { "PASS" } else { "FAIL" },
            worst.0,
            worst.1
        ));

        // Spec byte identity, kvq armed SAME-CONFIG (spec and plain both quantized).
        set_kv_quant(true);
        {
            let max_new = 24usize;
            let spec_k = 3usize;
            let cap = prompt.len() + max_new + spec_k + 4;
            let mut plain_state = model.alloc_state(&engine, cap)?;
            let logits = model.prefill(&engine, &prompt, &mut plain_state)?;
            let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]) as u32;
            let mut plain = vec![next];
            for _ in 1..max_new {
                let row = model.decode_step(&engine, next, &mut plain_state)?;
                next = argmax(&row) as u32;
                plain.push(next);
            }
            let mut spec_state = model.alloc_state(&engine, cap)?;
            let mut draft_state = model.mtp_state(&engine, cap)?;
            let report = model.spec_generate(
                &engine,
                &prompt,
                max_new,
                spec_k,
                &mut spec_state,
                &mut draft_state,
                None,
            )?;
            let matched = report.tokens == plain;
            if !matched {
                failures += 1;
            }
            lines.push(format!(
                "kvq-spec-byte-identity\tk={spec_k}\ttokens={max_new}\taccepted={}/{}\tpass={matched}",
                report.accepted, report.drafted
            ));
            summaries.push(format!(
                "kvq-spec: spec-vs-plain byte identity {} with the quantized cache on BOTH arms \
                 (k={spec_k}, {max_new} tokens, accepted {}/{})",
                if matched { "PASS" } else { "FAIL" },
                report.accepted,
                report.drafted
            ));
        }
        set_kv_quant(false);

        // Arm IDXQ: quantized indexer raw-key cache. The exact system-level pin is the
        // idxcache ON/OFF interleave: with idxq armed, the host-quantized and device-
        // quantized cache rows must be interchangeable — decode logits BIT-IDENTICAL
        // between the two arms (this is what makes the host/device quantize twins a
        // contract rather than a convention).
        for mode in ["q8", "bf16"] {
            set_idxq(mode);
            memra_engine::qwen4exp_gpu::set_idx_cache(true);
            let on_rows = run_rows(&model)?;
            memra_engine::qwen4exp_gpu::set_idx_cache(false);
            let off_rows = run_rows(&model)?;
            memra_engine::qwen4exp_gpu::set_idx_cache(true);
            let identical = on_rows
                .iter()
                .zip(&off_rows)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !identical {
                failures += 1;
            }
            lines.push(format!(
                "idxq-{mode}-interleave\tidxcache_on_vs_off_bits={identical}\tpass={identical}"
            ));
            summaries.push(format!(
                "idxq-{mode}: idxcache ON vs OFF (host- vs device-quantized rows) {}",
                if identical {
                    "BIT-IDENTICAL"
                } else {
                    "DIVERGED"
                }
            ));
            // Selection-flip envelope vs the f32 cache (REPORT + finiteness).
            let mut worst = (0.0f32, 0.0f32);
            let mut argmax_flips = 0usize;
            for r in 0..n_rows {
                let s = compare_row(
                    &f32_rows[r * vocab..(r + 1) * vocab],
                    &on_rows[r * vocab..(r + 1) * vocab],
                );
                worst.0 = worst.0.max(s.max_abs);
                worst.1 = worst.1.max(s.max_rel);
                if !s.argmax_match {
                    argmax_flips += 1;
                }
            }
            lines.push(format!(
                "idxq-{mode}-envelope\tabs={:.3e}\trel={:.3e}\targmax_flips={argmax_flips}/{n_rows}",
                worst.0, worst.1
            ));
        }
        // Spec byte identity with idxq=q8 armed (same-config both arms).
        set_idxq("q8");
        {
            let max_new = 24usize;
            let spec_k = 3usize;
            let cap = prompt.len() + max_new + spec_k + 4;
            let mut plain_state = model.alloc_state(&engine, cap)?;
            let logits = model.prefill(&engine, &prompt, &mut plain_state)?;
            let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]) as u32;
            let mut plain = vec![next];
            for _ in 1..max_new {
                let row = model.decode_step(&engine, next, &mut plain_state)?;
                next = argmax(&row) as u32;
                plain.push(next);
            }
            let mut spec_state = model.alloc_state(&engine, cap)?;
            let mut draft_state = model.mtp_state(&engine, cap)?;
            let report = model.spec_generate(
                &engine,
                &prompt,
                max_new,
                spec_k,
                &mut spec_state,
                &mut draft_state,
                None,
            )?;
            let matched = report.tokens == plain;
            if !matched {
                failures += 1;
            }
            lines.push(format!(
                "idxq-q8-spec-byte-identity\tk={spec_k}\ttokens={max_new}\taccepted={}/{}\tpass={matched}",
                report.accepted, report.drafted
            ));
            summaries.push(format!(
                "idxq-q8-spec: spec-vs-plain byte identity {} (quantized raw-key cache both arms)",
                if matched { "PASS" } else { "FAIL" }
            ));
        }
        set_idxq("f32");
    }

    // Arm Y (yarn-tiny): the fixture with RopeFactors::Yarn on every full-attention rope
    // (trunk + MTP), gated GPU-vs-reference like arm A. Tiny n_rot = 2 means ONE freq
    // pair, and the truncate clamp pins its ramp to pure extrapolation — so this arm
    // exercises the mscale-on-cos/sin path, the ffm kernel, the host indexer twin, and
    // the reference yarn plumbing, NOT the divisor numerics (those are pinned in
    // memra-gguf's yarn_divisors test against the banked transformers receipt, and the
    // real-checkpoint yarn goldens cover dim 64 end to end).
    let set_yarn = |plan: &mut ModelPlan, factors: RopeFactors| {
        for layer in plan.layers.iter_mut() {
            if let AttentionPlan::Full(attention) = &mut layer.attention {
                attention.rope.factors = factors;
            }
        }
        for mtp in plan.mtp_blocks.iter_mut() {
            if let AttentionPlan::Full(attention) = &mut mtp.layer.attention {
                attention.rope.factors = factors;
            }
        }
    };
    {
        let mut yarn_plan = plan.clone();
        set_yarn(
            &mut yarn_plan,
            RopeFactors::Yarn {
                factor: 2.0,
                original_context: 8,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        );
        let yarn_model =
            Qwen4ExpGpu::from_reference_weights(&engine, &yarn_plan, &fixture.weights)?;
        let result = run_arm(
            "fixture-yarn",
            &engine,
            &yarn_plan,
            &fixture.weights,
            &yarn_model,
            &prompt,
            &decode_feed,
            &mut lines,
            &mut failures,
        )?;
        summaries.push(format!(
            "fixture-yarn (factor 2, mscale 1.0693): prefill worst abs {:.3e} rel {:.3e}; \
             decode worst abs {:.3e} rel {:.3e}",
            result.prefill_worst.0,
            result.prefill_worst.1,
            result.decode_worst.0,
            result.decode_worst.1
        ));
    }

    // Arm Y2 (yarn-identity): factor 1.0 must be BYTE-IDENTICAL to the plain-rope model —
    // the ffm kernel with an all-ones divisor table and mscale 1.0 reproduces rope_neox
    // bit-for-bit (the kernel contract), and the helper returns EXACT ones for
    // factor <= 1 by construction. Prefill rows AND every decode step compared as bits.
    {
        let mut identity_plan = plan.clone();
        set_yarn(
            &mut identity_plan,
            RopeFactors::Yarn {
                factor: 1.0,
                original_context: 64,
                beta_fast: 32.0,
                beta_slow: 1.0,
            },
        );
        let identity_model =
            Qwen4ExpGpu::from_reference_weights(&engine, &identity_plan, &fixture.weights)?;
        let vocab = plan.vocab_size as usize;
        let cap = prompt.len() + decode_feed.len() + 2;
        let mut rows_base: Vec<Vec<f32>> = Vec::new();
        let mut rows_yarn: Vec<Vec<f32>> = Vec::new();
        for (m, sink) in [(&model, &mut rows_base), (&identity_model, &mut rows_yarn)] {
            let mut state = m.alloc_state(&engine, cap)?;
            let prefill = m.prefill(&engine, &prompt, &mut state)?;
            for row in 0..prompt.len() {
                sink.push(prefill[row * vocab..(row + 1) * vocab].to_vec());
            }
            for &token in &decode_feed {
                sink.push(m.decode_step(&engine, token, &mut state)?);
            }
        }
        let mut identical = true;
        for (row, (a, b)) in rows_base.iter().zip(rows_yarn.iter()).enumerate() {
            let same = a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            if !same {
                identical = false;
                failures += 1;
                lines.push(format!(
                    "yarn-identity\trow={row}\tbit_identical=false\tpass=false"
                ));
            }
        }
        if identical {
            lines.push(format!(
                "yarn-identity\trows={}\tbit_identical=true\tpass=true",
                rows_base.len()
            ));
        }
        summaries.push(format!(
            "yarn-identity: factor-1.0 yarn vs plain rope {} ({} rows bit-compared)",
            if identical { "BIT-IDENTICAL" } else { "FAIL" },
            rows_base.len()
        ));
    }

    // Arm 0f: the block-list attention kernel oracle at real QSA geometry (long-context
    // lane) — bit identity vs the masked kernel + the past-the-bound host-twin case.
    summaries.push(memra_engine::qwen4exp_gpu::gate_sdpa_blocklist(&engine)?);

    // Arm 0g: the device QSA index scorer vs the host twin — BIT identity per score plus
    // top-k set equality (what the selection actually consumes).
    summaries.push(memra_engine::qwen4exp_gpu::gate_qsa_index_score(&engine)?);

    // Arm 0i: the device QSA indexer top-k SELECTION vs `top_blocks_ascending` — ids AND
    // ascending order EXACT at REAL budget 512 up to 65,536 blocks (the 262k window's
    // fill/4), ragged sub-batch slabs, and the tie classes (all-zero — structural on a
    // relu-sum score — boundary-straddling duplicates, and the total_cmp domain).
    summaries.push(memra_engine::qwen4exp_gpu::gate_qsa_index_topk(&engine)?);

    // Arm 0j: the PLE n-gram id CACHE vs the full `host_ngram_ids` twin — ids EXACT (a wrong
    // id gathers a different embedding row, so there is no tolerance), over decode-shaped and
    // prefill-shaped growth, eos segment resets, all-eos, repeated rewinds to DIVERGING
    // prefixes, and shorter-unrelated-sequence state reuse. Host-only: no engine needed.
    summaries.push(memra_engine::qwen4exp_gpu::gate_ple_ngram_cache()?);
    summaries.push(memra_engine::qwen4exp_gpu::gate_seam_table()?);

    // Arm R (mtp-spec-ring): the long-context spec shape — chunked co-prefill (chunk 8
    // over the 18-token prompt) + a RING-bounded wide stash (16 rows, so replay reads
    // cross the ring seam) — must stay BYTE-IDENTICAL to the plain greedy chain (commits
    // are always target rows; the ring only re-slots the draft's seed source).
    {
        let vocab = plan.vocab_size as usize;
        let max_new = 24usize;
        let spec_k = 3usize;
        let cap = prompt.len() + max_new + spec_k + 4;
        let mut plain_state = model.alloc_state(&engine, cap)?;
        let logits = model.prefill(&engine, &prompt, &mut plain_state)?;
        let mut next = argmax(&logits[(prompt.len() - 1) * vocab..]) as u32;
        let mut plain = vec![next];
        for _ in 1..max_new {
            let row = model.decode_step(&engine, next, &mut plain_state)?;
            next = argmax(&row) as u32;
            plain.push(next);
        }
        let mut spec_state = model.alloc_state(&engine, cap)?;
        let mut draft_state = model.mtp_state(&engine, cap)?;
        let opts = memra_engine::qwen4exp_gpu::SpecOpts {
            prefill_chunk: Some(8),
            wide_ring: Some(16),
            ..Default::default()
        };
        let report = model.spec_generate_ext(
            &engine,
            &engine,
            &prompt,
            max_new,
            spec_k,
            &mut spec_state,
            &mut draft_state,
            None,
            opts,
            None,
        )?;
        let matched = report.tokens == plain;
        if !matched {
            failures += 1;
        }
        lines.push(format!(
            "mtp-spec-ring\tbyte-identity\tchunk=8\tring=16\ttokens={max_new}\trounds={}\tspec={:?}\tplain={:?}\tpass={matched}",
            report.rounds, report.tokens, plain
        ));
        summaries.push(format!(
            "mtp-spec-ring: chunked co-prefill (8) + wide ring (16) spec-vs-plain \
             byte-identity {} ({} rounds)",
            if matched { "PASS" } else { "FAIL" },
            report.rounds
        ));
    }

    // Arm L (fixture-longatt): the WHOLE tiny program with the block-list attention
    // FORCED (production auto only engages past the masked kernel's smem bound, which
    // the tiny scale never reaches) — prefill rows and every decode step must be
    // BIT-IDENTICAL to the base model's masked-kernel run. The tiny indexer budget (2)
    // drops blocks past position 11, so scored-form lists (not just full prefixes) flow
    // through rowsel_positions into the kernel.
    {
        memra_engine::qwen4exp_gpu::set_longatt("force");
        let vocab = plan.vocab_size as usize;
        let cap = prompt.len() + decode_feed.len() + 2;
        let mut rows_long: Vec<Vec<f32>> = Vec::new();
        {
            let mut state = model.alloc_state(&engine, cap)?;
            let prefill = model.prefill(&engine, &prompt, &mut state)?;
            for row in 0..prompt.len() {
                rows_long.push(prefill[row * vocab..(row + 1) * vocab].to_vec());
            }
            for &token in &decode_feed {
                rows_long.push(model.decode_step(&engine, token, &mut state)?);
            }
        }
        memra_engine::qwen4exp_gpu::set_longatt("auto");
        let mut rows_base: Vec<Vec<f32>> = Vec::new();
        {
            let mut state = model.alloc_state(&engine, cap)?;
            let prefill = model.prefill(&engine, &prompt, &mut state)?;
            for row in 0..prompt.len() {
                rows_base.push(prefill[row * vocab..(row + 1) * vocab].to_vec());
            }
            for &token in &decode_feed {
                rows_base.push(model.decode_step(&engine, token, &mut state)?);
            }
        }
        let mut identical = true;
        for (row, (a, b)) in rows_base.iter().zip(rows_long.iter()).enumerate() {
            let same = a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits());
            if !same {
                identical = false;
                failures += 1;
                lines.push(format!(
                    "fixture-longatt\trow={row}\tbit_identical=false\tpass=false"
                ));
            }
        }
        if identical {
            lines.push(format!(
                "fixture-longatt\trows={}\tbit_identical=true\tpass=true",
                rows_base.len()
            ));
        }
        summaries.push(format!(
            "fixture-longatt: forced block-list attention vs masked {} ({} rows bit-compared)",
            if identical { "BIT-IDENTICAL" } else { "FAIL" },
            rows_base.len()
        ));
    }

    // Arm P (prefill-extend): the chunked long-context prefill driver — chunked
    // prefill_extend (head skipped mid-chunk, last-row lm_head) then decode, vs the
    // one-shot prefill + decode. TOLERANCE class, not bit class: chunking changes the
    // cuBLASLt GEMM shapes (m = chunk vs m = t), the same accumulation-order family the
    // verify lane documented — the modelplan policy (0.01/0.01 + argmax) applies.
    {
        let vocab = plan.vocab_size as usize;
        let cap = prompt.len() + decode_feed.len() + 2;
        let mut one_state = model.alloc_state(&engine, cap)?;
        let one = model.prefill(&engine, &prompt, &mut one_state)?;
        let one_last = &one[(prompt.len() - 1) * vocab..prompt.len() * vocab];
        let mut chunk_state = model.alloc_state(&engine, cap)?;
        let chunked = model.prefill_extend(&engine, &prompt, &mut chunk_state, 5)?;
        let mut worst = (0.0f32, 0.0f32);
        let mut ok = chunked.len() == vocab;
        let fold = |s: RowStats, ok: &mut bool, worst: &mut (f32, f32)| {
            worst.0 = worst.0.max(s.max_abs);
            worst.1 = worst.1.max(s.max_rel);
            *ok &= s.max_abs <= 0.01 && s.max_rel <= 0.01 && s.argmax_match;
        };
        fold(compare_row(one_last, &chunked), &mut ok, &mut worst);
        for &token in &decode_feed {
            let a = model.decode_step(&engine, token, &mut one_state)?;
            let b = model.decode_step(&engine, token, &mut chunk_state)?;
            fold(compare_row(&a, &b), &mut ok, &mut worst);
        }
        if !ok {
            failures += 1;
        }
        lines.push(format!(
            "prefill-extend\tchunk=5\tmax_abs={:.3e}\tmax_rel={:.3e}\tpass={ok}",
            worst.0, worst.1
        ));
        summaries.push(format!(
            "prefill-extend: chunked (5) vs one-shot prefill + decode worst abs {:.3e} rel \
             {:.3e} ({})",
            worst.0,
            worst.1,
            if ok { "PASS" } else { "FAIL" }
        ));
    }

    let executable = std::fs::read(std::env::current_exe()?)?;
    let sha256 = Sha256::digest(&executable)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let all_tokens: Vec<u32> = prompt.iter().chain(decode_feed.iter()).copied().collect();
    let mut receipt = format!(
        "# qwen4exp-gpu-gate\tbinary_sha256={sha256}\tpolicy=max_abs<=0.01 max_rel<=0.01 argmax\tprompt_len={}\tdecode_steps={}\tvocab={}\tplan=tiny(qwen4_exp pack)\tarms=fixture,mtp-fixture,dir-bf16,dir-nvfp4-stacked,dir-nvfp4-perexpert,mtp-dir-bf16,mtp-spec-tiny,mtp-spec-defer,mtp-spec-defer-dirbf16,mtp-armed-prefill-bit,mtp-vfuse,fixture-yarn,yarn-identity,bank-bytes\ttokens={all_tokens:?}\n",
        prompt.len(),
        decode_feed.len(),
        plan.vocab_size
    );
    for line in &lines {
        receipt.push_str(line);
        receipt.push('\n');
    }
    for summary in &summaries {
        receipt.push_str(&format!("# summary\t{summary}\n"));
    }
    receipt.push_str(&format!("# verdict\tfailures={failures}\n"));
    std::fs::write(&receipt_path, &receipt)?;

    if failures > 0 {
        eprintln!(
            "qwen4exp-gpu-gate FAILED: {failures} rows out of tolerance (receipt {receipt_path})"
        );
        std::process::exit(1);
    }
    for summary in &summaries {
        println!("qwen4exp-gpu-gate PASS [{summary}]");
    }
    println!("receipt: {receipt_path}");
    Ok(())
}
