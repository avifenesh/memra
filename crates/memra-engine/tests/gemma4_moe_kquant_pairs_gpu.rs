//! memra#1234: a gemma4_moe model with k-quant expert banks must prefill what it decodes.
//!
//! WHY. `gemma4_moe` admits every `expert_dp4a_supported` qtype to its device arms: Q4_0, IQ3_S,
//! IQ4_XS, Q3_K, Q4_K and Q6_K. Three of its programs decode only Q4_0, IQ4_XS and IQ3_S: the
//! prefill pairs arm's int8-MMA expert GEMM (t >= 16; any other qtype is read as IQ3_S), the
//! pairs arm's dp4a program, which always called the decode-once `moe_pairs_matvec_q8_dec` (any
//! other qtype is read as IQ4_XS), and the verify CSR twin of the gate/up kernel (t = 2..10;
//! read as IQ3_S). A k-quant bank was read with the wrong layout: a silently wrong prefill and
//! a silently wrong speculative verify. The fix is the qwen arm's per-qtype dispatch at all
//! three.
//!
//! WHAT THIS GATE HOLDS. A two-layer gemma4_moe fixture with gemma-4-26B-A4B's expert geometry
//! (128 experts, top 8) in five expert-bank variants:
//!   * `Q4_0 / Q4_0`, ff 704: the control, the shipped QAT class (the arm was always right here);
//!   * `Q3_K / Q4_0`, ff 704: the layout a stock llama.cpp Q3_K quantization gives the 26B. Its
//!     down k (704) is not a multiple of the 256-value k-quant superblock, so the quantizer
//!     falls back to a 32-value type there, and Q3_K's fallback is Q4_0;
//!   * `Q4_K / Q4_0`, ff 704: Q4_K gate/up. (Stock llama.cpp falls back from Q4_K to Q5_0 and
//!     from Q6_K to Q8_0, which the device arms do not admit, so such a 26B GGUF keeps the
//!     host-staged loop; this variant and the next hold the dispatch for the other k-quants);
//!   * `Q6_K / Q6_K`, ff 512: a k-quant down too (a 256-multiple ff, so the down bank can be one);
//!   * `Q4_0 / Q4_0`, ff 512: that variant's control.
//!
//! Each variant runs forward at T = 1 (the decode kernels), 8 (the verify CSR twin), 12 (the
//! verify rows twin) and 16, 40, 130 (the pairs arm), on the int8-MMA default and under
//! `MEMRA_GEMMA_MOE_MMA=0` (the dp4a arm). Every T is compared with `memra_reference::execute`
//! over the DEQUANTIZED weights the engine reads, so the only error left is the engine's own
//! activation and KV quantization. The score of a width is the MEDIAN over its rows of the
//! row's largest logit error: a near-tie in the router's top-k that activation quantization
//! flips moves a few rows a lot in any variant (the printed maximum shows it), while a bank
//! read with the wrong layout moves every row. The k-quant variants must stay within `BAND`
//! times their control's score at the same width and under `CAP` whatever the control did, and
//! at most `SHARE` of a width's rows may be above `CAP` (a misread confined to a few experts
//! moves a minority of rows). Before the fix T = 8 and the pairs-arm widths are far outside.
//! A second check primes 16 tokens (the pairs arm) and compares the last row with 16 decode
//! steps (the decode kernels): two programs of one numeric class, within the same band.
//!
//! PLACEMENT. Each k-quant variant then runs the same widths and four wide prefills with the
//! experts in a 360-slot SLRU cache (memra#1225's tight arm: pinned admissions, and chunked
//! expert runs when a wide prefill selects more than 120 experts) and must be bit-identical to
//! its resident run: the `_em` pair matvec that k-quant banks now take walks one expert segment
//! per block row like `_dec`.
//!
//! The evidence is fixture-only: no k-quant gemma4_moe GGUF is on any unit we test on. The
//! `MEMRA_GEMMA_CSR`, `MEMRA_GEMMA_MOE_FAST`, `MEMRA_GEMMA_MOE_PAIRS` and `MEMRA_MOE_F16G` doors
//! must be unset: they take a variant off the programs this gate holds.
//!
//! GPU-gated (`#[ignore]`). Run under the 5090 lease with `NVIDIA_TF32_OVERRIDE=0` and
//! `-- --ignored --test-threads=1`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId,
    TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const VOCAB: u32 = 512;
const MAX_CTX: usize = 1024;
const WIDTH: usize = 256;
const EXPERTS: usize = 128;
const TOP_K: usize = 8;
/// A k-quant variant's error against the reference may be at most this many times its control's.
const BAND: f32 = 3.0;
/// Absolute floor under the band, for widths where the control is near exact.
const FLOOR: f32 = 2e-3;
/// Absolute ceiling on a k-quant score, so a noisy control cannot loosen the bound: right
/// scores are near 0.002 and a misread bank is above 0.08.
const CAP: f32 = 1e-2;
/// The largest share of a width's rows that may be above `CAP`. Router top-k flips put up to a
/// quarter of a width's rows there in any variant, the controls included (3 of 12 rows at the
/// worst); a misread of four of the 128 experts reaches 40% of the rows of a wide prefill
/// (`1 - (124 / 128)^16` over two layers of top 8).
const SHARE: f32 = 1.0 / 3.0;
/// Rollbacks and doors that would take a variant off the programs this gate holds.
const FOREIGN_ENV: [&str; 4] = [
    "MEMRA_GEMMA_CSR",
    "MEMRA_GEMMA_MOE_FAST",
    "MEMRA_GEMMA_MOE_PAIRS",
    "MEMRA_MOE_F16G",
];
const WIDTHS: [usize; 6] = [1, 8, 12, 16, 40, 130];
/// The placement check's (width, token seed) rows: the reference widths, then wide prefills,
/// which can select more experts than the tight cache pins at once (120 of 128).
const PLACEMENT_ROWS: [(usize, u64); 10] = [
    (1, 0),
    (8, 0),
    (12, 0),
    (16, 0),
    (40, 0),
    (130, 0),
    (600, 0),
    (1000, 0),
    (1000, 1),
    (1000, 2),
];
/// memra#1225's floor: the blocks a call below the prefill width can need at once.
const TIGHT_SLOTS: usize = 3 * if EXPERTS < 15 * TOP_K {
    EXPERTS
} else {
    15 * TOP_K
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Variant {
    gate_up: GgmlType,
    down: GgmlType,
    ff: usize,
}

const CONTROL_704: Variant = Variant {
    gate_up: GgmlType::Q4_0,
    down: GgmlType::Q4_0,
    ff: 704,
};
const Q3K_704: Variant = Variant {
    gate_up: GgmlType::Q3_K,
    down: GgmlType::Q4_0,
    ff: 704,
};
const Q4K_704: Variant = Variant {
    gate_up: GgmlType::Q4_K,
    down: GgmlType::Q4_0,
    ff: 704,
};
const CONTROL_512: Variant = Variant {
    gate_up: GgmlType::Q4_0,
    down: GgmlType::Q4_0,
    ff: 512,
};
const Q6K_512: Variant = Variant {
    gate_up: GgmlType::Q6_K,
    down: GgmlType::Q6_K,
    ff: 512,
};

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made yet; call_once serializes the write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

fn mini_config(ff: usize) -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&format!(
        r#"{{"model_type":"gemma4","text_config":{{"model_type":"gemma4_text",
        "num_hidden_layers":2,"hidden_size":{WIDTH},"num_attention_heads":2,
        "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":256,
        "global_head_dim":512,"intermediate_size":{WIDTH},"moe_intermediate_size":{ff},
        "num_experts":{EXPERTS},"top_k_experts":{TOP_K},"vocab_size":{VOCAB},
        "max_position_embeddings":{MAX_CTX},"rms_norm_eps":0.000001,
        "sliding_window":64,"final_logit_softcapping":30,"hidden_activation":"gelu_pytorch_tanh",
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{{"full_attention":{{"rope_theta":1000000,
        "partial_rotary_factor":0.25}},"sliding_attention":{{"rope_theta":10000}}}}}}}}"#
    )))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    let pack = memra_gguf::model_packs::for_config(config).expect("a pack matches the mini config");
    assert_eq!(
        pack.family, "gemma4_moe",
        "the mini config is the MoE gemma"
    );
    pack.compile_plan(config)
        .expect("mini gemma4_moe plan compiles")
}

/// See `gemma4_chunked_prime_gpu.rs`: the GGUF contract needs `rope_freqs.weight` for a
/// `PartialRotary` rope, which `deterministic_fixture` only synthesizes for checkpoint factors.
fn add_partial_rotary_factors(plan: &ModelPlan, weights: &mut BTreeMap<TensorId, ReferenceTensor>) {
    use memra_gguf::model_plan::{AttentionPlan, RopeFactors};
    if weights.contains_key(&TensorId::RopeFactors) {
        return;
    }
    let mut table: Option<Vec<f32>> = None;
    for layer in &plan.layers {
        let rope = match &layer.attention {
            AttentionPlan::Full(a) | AttentionPlan::SlidingWindow { attention: a, .. } => &a.rope,
            _ => continue,
        };
        if let RopeFactors::PartialRotary { factor } = rope.factors {
            let width = rope.dimensions as usize / 2;
            let keep = (width as f32 * factor.clamp(0.0, 1.0)).round() as usize;
            let t: Vec<f32> = (0..width)
                .map(|i| if i < keep { 1.0 } else { 1.0e30 })
                .collect();
            match &table {
                Some(prev) if prev.len() >= t.len() => {}
                _ => table = Some(t),
            }
        }
    }
    if let Some(t) = table {
        let n = t.len();
        weights.insert(
            TensorId::RopeFactors,
            ReferenceTensor::new(vec![n], t).expect("rope factors tensor"),
        );
    }
}

fn f16(x: f32) -> [u8; 2] {
    memra_gguf::nvfp4_repack::f32_to_f16_bits(x).to_le_bytes()
}

/// The f32 value an f16 stores (what the engine decodes the scale as).
fn f16_round(x: f32) -> f32 {
    let b = memra_gguf::nvfp4_repack::f32_to_f16_bits(x);
    memra_gguf::dequant::dequantize(GgmlType::F16, &b.to_le_bytes(), 1)[0]
}

/// ggml's reference Q4_0 encoder (see `moe_slru_gemma_placement_gpu.rs`).
fn f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 32 * 18);
    for blk in data.chunks_exact(32) {
        let max = blk
            .iter()
            .fold(0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16(d));
        let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
        for j in 0..16 {
            out.push(q(blk[j]) | (q(blk[j + 16]) << 4));
        }
    }
    out
}

/// A valid block_q3_K encoder (a direct fit): per 256 values, sixteen 16-value sub-blocks with a
/// 6-bit scale (stored + 32) under an fp16 `d`, 3-bit quants split into `qs` (low two bits) and
/// `hmask` (the high bit), packed as ggml's dequantizer reads them.
fn f32_to_q3_k(data: &[f32]) -> Vec<u8> {
    assert_eq!(
        data.len() % 256,
        0,
        "Q3_K encodes whole 256-value superblocks"
    );
    let mut out = Vec::with_capacity(data.len() / 256 * 110);
    for sb in data.chunks_exact(256) {
        // The quant range is -4..=3, so a sub-block's step covers both ends.
        let step: Vec<f32> = sb
            .chunks_exact(16)
            .map(|c| {
                let lo = c.iter().fold(0f32, |m, &v| m.min(v));
                let hi = c.iter().fold(0f32, |m, &v| m.max(v));
                (-lo / 4.0).max(hi / 3.0)
            })
            .collect();
        let d = f16_round(step.iter().fold(0f32, |m, &v| m.max(v)) / 31.0);
        let sc: Vec<u8> = step
            .iter()
            .map(|&v| {
                let eff = if d > 0.0 {
                    (v / d).round().clamp(0.0, 31.0)
                } else {
                    0.0
                };
                eff as u8 + 32
            })
            .collect();
        let (mut hmask, mut qs, mut scales) = ([0u8; 32], [0u8; 64], [0u8; 12]);
        for (y, &x) in sb.iter().enumerate() {
            let dl = d * (sc[y / 16] as f32 - 32.0);
            let q = if dl > 0.0 {
                (x / dl).round().clamp(-4.0, 3.0) as i32
            } else {
                0
            };
            let u = (q + 4) as u8;
            let (half, r) = (y / 128, y % 128);
            let (j, l) = (r / 32, r % 32);
            qs[half * 32 + l] |= (u & 3) << (2 * j);
            hmask[l] |= (u >> 2) << (half * 4 + j);
        }
        for b in 0..4 {
            scales[b] = (sc[b] & 0xF) | ((sc[8 + b] & 0xF) << 4);
            scales[4 + b] = (sc[4 + b] & 0xF) | ((sc[12 + b] & 0xF) << 4);
            scales[8 + b] = (sc[b] >> 4)
                | ((sc[4 + b] >> 4) << 2)
                | ((sc[8 + b] >> 4) << 4)
                | ((sc[12 + b] >> 4) << 6);
        }
        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&f16(d));
    }
    out
}

/// A valid block_q4_K encoder (not ggml's search, a direct min/max fit): per 256 values, eight
/// 32-value sub-blocks with a 6-bit scale and min under fp16 super-scales `d` and `dmin`, packed
/// as `get_scale_min_k4` reads them.
fn f32_to_q4_k(data: &[f32]) -> Vec<u8> {
    assert_eq!(
        data.len() % 256,
        0,
        "Q4_K encodes whole 256-value superblocks"
    );
    let mut out = Vec::with_capacity(data.len() / 256 * 144);
    for sb in data.chunks_exact(256) {
        let mut scale = [0f32; 8];
        let mut minv = [0f32; 8];
        for (j, sub) in sb.chunks_exact(32).enumerate() {
            let lo = sub.iter().fold(0f32, |m, &v| m.min(v));
            let hi = sub.iter().fold(f32::MIN, |m, &v| m.max(v));
            scale[j] = (hi - lo) / 15.0;
            minv[j] = -lo;
        }
        let d = f16_round(scale.iter().fold(0f32, |m, &v| m.max(v)) / 63.0);
        let dmin = f16_round(minv.iter().fold(0f32, |m, &v| m.max(v)) / 63.0);
        let q6 = |v: f32, s: f32| {
            if s > 0.0 {
                (v / s).round().clamp(0.0, 63.0) as u8
            } else {
                0
            }
        };
        let sc: Vec<u8> = scale.iter().map(|&v| q6(v, d)).collect();
        let mn: Vec<u8> = minv.iter().map(|&v| q6(v, dmin)).collect();
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = sc[j] | ((sc[j + 4] >> 4) << 6);
            scales[j + 4] = mn[j] | ((mn[j + 4] >> 4) << 6);
            scales[j + 8] = (sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4);
        }
        let mut qs = [0u8; 128];
        for g in 0..4 {
            for half in 0..2 {
                let j = 2 * g + half;
                let (dl, ml) = (d * sc[j] as f32, dmin * mn[j] as f32);
                for l in 0..32 {
                    let x = sb[j * 32 + l];
                    let q = if dl > 0.0 {
                        ((x + ml) / dl).round().clamp(0.0, 15.0) as u8
                    } else {
                        0
                    };
                    qs[g * 32 + l] |= q << (4 * half);
                }
            }
        }
        out.extend_from_slice(&f16(d));
        out.extend_from_slice(&f16(dmin));
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qs);
    }
    out
}

/// A valid block_q6_K encoder: per 256 values, sixteen 16-value sub-blocks with an int8 scale
/// under an fp16 `d`, 6-bit quants in `ql`/`qh` in ggml's layout.
fn f32_to_q6_k(data: &[f32]) -> Vec<u8> {
    assert_eq!(
        data.len() % 256,
        0,
        "Q6_K encodes whole 256-value superblocks"
    );
    let mut out = Vec::with_capacity(data.len() / 256 * 210);
    for sb in data.chunks_exact(256) {
        let sub: Vec<f32> = sb
            .chunks_exact(16)
            .map(|c| c.iter().fold(0f32, |m, &v| m.max(v.abs())) / 31.0)
            .collect();
        let d = f16_round(sub.iter().fold(0f32, |m, &v| m.max(v)) / 127.0);
        let sc: Vec<i8> = sub
            .iter()
            .map(|&s| {
                if d > 0.0 {
                    (s / d).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                }
            })
            .collect();
        let q: Vec<u8> = (0..256)
            .map(|y| {
                let s = d * sc[y / 16] as f32;
                let v = if s != 0.0 {
                    (sb[y] / s).round().clamp(-32.0, 31.0)
                } else {
                    0.0
                };
                (v as i32 + 32) as u8
            })
            .collect();
        let (mut ql, mut qh) = ([0u8; 128], [0u8; 64]);
        for n2 in 0..2 {
            for l in 0..32 {
                let y = n2 * 128 + l;
                let (q1, q2, q3, q4) = (q[y], q[y + 32], q[y + 64], q[y + 96]);
                ql[n2 * 64 + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[n2 * 64 + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[n2 * 32 + l] =
                    (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend(sc.iter().map(|&v| v as u8));
        out.extend_from_slice(&f16(d));
    }
    out
}

fn encode(ty: GgmlType, data: &[f32]) -> Vec<u8> {
    match ty {
        GgmlType::Q4_0 => f32_to_q4_0(data),
        GgmlType::Q3_K => f32_to_q3_k(data),
        GgmlType::Q4_K => f32_to_q4_k(data),
        GgmlType::Q6_K => f32_to_q6_k(data),
        GgmlType::Q8_0 => memra_gguf::nvfp4_repack::f32_to_q8_0(data),
        other => panic!("no encoder for {other:?}"),
    }
}

/// Uniform values in [-0.5, 0.5) from a splitmix64 stream seeded by the layer index.
fn spread_router(n: usize, id: &TensorId) -> Vec<f32> {
    let mut s: u64 = 0x1234_5EED
        ^ match id {
            TensorId::Layer { index, .. } => *index as u64,
            _ => 0,
        };
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ty: GgmlType,
}

struct FixtureSource {
    config: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&t.bytes),
            ggml_type: t.ty,
            ne: t.ne.clone(),
        })
    }
    fn tensor_census(&self) -> Result<memra_gguf::source::TensorCensus, String> {
        Ok(memra_gguf::source::census_from_views(
            self.tensors
                .iter()
                .map(|(name, t)| (name.as_str(), t.ty, t.ne.as_slice(), t.bytes.len() as u64)),
        ))
    }
}

/// The variant's GGUF-named source, and the reference weights: every quantized tensor replaced
/// by its dequantized values, so the reference computes the function the engine's weights hold.
fn fixture(
    v: Variant,
) -> (
    ModelConfig,
    ModelPlan,
    FixtureSource,
    BTreeMap<TensorId, ReferenceTensor>,
) {
    let config = mini_config(v.ff);
    let plan = mini_plan(&config);
    let mut weights = deterministic_fixture(&plan)
        .expect("deterministic gemma4_moe fixture")
        .weights;
    add_partial_rotary_factors(&plan, &mut weights);
    let contract = TensorContract::for_plan(
        &plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini gemma4_moe plan");
    let mut tensors = BTreeMap::new();
    let mut reference = weights.clone();
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let n: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(n, tensor.data.len(), "fixture {:?} shape", req.id);
        let ty = match req.id {
            TensorId::Layer {
                tensor: LayerTensor::MoeRouter,
                ..
            } => GgmlType::F32,
            TensorId::Layer {
                tensor: LayerTensor::MoeExpertGateUpBank,
                ..
            } => v.gate_up,
            TensorId::Layer {
                tensor: LayerTensor::MoeExpertDownBank,
                ..
            } => v.down,
            TensorId::Layer { .. } if req.shape.len() == 2 && req.shape[0].is_multiple_of(32) => {
                GgmlType::Q8_0
            }
            _ => GgmlType::F32,
        };
        let data = if matches!(
            req.id,
            TensorId::Layer {
                tensor: LayerTensor::MoeRouter,
                ..
            }
        ) {
            spread_router(n, &req.id)
        } else {
            tensor.data.clone()
        };
        let bytes = if ty == GgmlType::F32 {
            data.iter().flat_map(|x| x.to_le_bytes()).collect()
        } else {
            encode(ty, &data)
        };
        let held = if ty == GgmlType::F32 {
            data
        } else {
            memra_gguf::dequant::dequantize(ty, &bytes, n)
        };
        reference.insert(
            req.id.clone(),
            ReferenceTensor::new(tensor.shape.clone(), held).expect("reference tensor"),
        );
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: bytes.clone(),
                    ne: req.shape.clone(),
                    ty,
                },
            );
        }
    }
    let source = FixtureSource {
        config: config.clone(),
        tensors,
    };
    (config, plan, source, reference)
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32) % VOCAB
        })
        .collect()
}

/// Pins the expert placement for the next model load: the resident slab, or a `TIGHT_SLOTS`
/// SLRU cache.
fn set_placement(tight: bool) {
    // SAFETY: callers hold `gpu_guard()`; nothing else in this binary touches these vars and no
    // other thread runs engine code.
    unsafe {
        std::env::set_var("MEMRA_MOE_CACHE", "1");
        std::env::set_var("MEMRA_MOE_RESIDENT", if tight { "0" } else { "1" });
        std::env::set_var(
            "MEMRA_MOE_SLOTS",
            if tight {
                TIGHT_SLOTS.to_string()
            } else {
                String::new()
            },
        );
    }
}

/// A width's row errors: each row's max-abs difference from `want` over the max-abs of `want`.
#[derive(Clone, Copy, Debug)]
struct Score {
    median: f32,
    max: f32,
    /// Share of the rows above `CAP`.
    above: f32,
}

fn row_errors(got: &[f32], want: &[f32]) -> Score {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    let mut rows: Vec<f32> = got
        .chunks_exact(VOCAB as usize)
        .zip(want.chunks_exact(VOCAB as usize))
        .map(|(g, w)| max_abs_diff(g, w) / scale)
        .collect();
    assert!(!rows.is_empty(), "no logit rows to compare");
    rows.sort_by(f32::total_cmp);
    Score {
        median: rows[rows.len() / 2],
        max: rows[rows.len() - 1],
        above: rows.iter().filter(|&&r| r.is_nan() || r > CAP).count() as f32 / rows.len() as f32,
    }
}

/// The largest element difference; infinite when any difference is a NaN.
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    got.iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0f32,
            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
        )
}

/// Max-abs difference over the max-abs of `want`.
fn relative(got: &[f32], want: &[f32]) -> f32 {
    max_abs_diff(got, want) / want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

/// Per width: the engine's forward logits against the reference; then the 16-token prime's last
/// row against 16 decode steps. Resident placement.
fn errors(v: Variant) -> (Vec<Score>, f32) {
    force_true_f32();
    set_placement(false);
    let (config, plan, source, reference) = fixture(v);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini gemma4_moe model loads from the contract");
    let mut per_width = Vec::new();
    for &n in &WIDTHS {
        let ids = tokens(n, 0x1234_0000 ^ n as u64);
        let got = model.forward(&engine, &ids).expect("GPU forward");
        let want = memra_reference::execute(&plan, &reference, &ids)
            .expect("reference execute")
            .logits;
        per_width.push(row_errors(&got, &want));
    }
    let ids = tokens(16, 0x1234_0016);
    let new_cache = || {
        memra_engine::cache::Cache::new_planned(&engine, &model.cfg, &plan, MAX_CTX)
            .expect("cache for the mini gemma4_moe model")
    };
    let mut primed = new_cache();
    let (prime_last, _, _) = model
        .prime_cache(&engine, &ids, &mut primed, 0)
        .expect("prime");
    let mut stepped = new_cache();
    let mut decode_last = Vec::new();
    for &tk in &ids {
        decode_last = model
            .decode_step(&engine, tk, &mut stepped)
            .expect("decode step");
    }
    let _ = config;
    (per_width, relative(&prime_last, &decode_last))
}

/// The variant's forward logits for `PLACEMENT_ROWS` on one placement, with the SLRU arm's
/// (layer row, pinned, chunked) call counts.
fn placement_rows(v: Variant, tight: bool) -> (Vec<Vec<f32>>, (u64, u64, u64)) {
    force_true_f32();
    set_placement(tight);
    let before = memra_engine::moe_slru_pairs_calls();
    let (_, _, source, _) = fixture(v);
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("mini gemma4_moe model loads from the contract");
    let rows = PLACEMENT_ROWS
        .iter()
        .map(|&(n, seed)| {
            model
                .forward(&engine, &tokens(n, 0x1234_0000 ^ n as u64 ^ (seed << 32)))
                .expect("GPU forward")
        })
        .collect();
    let after = memra_engine::moe_slru_pairs_calls();
    (
        rows,
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
    )
}

#[test]
#[ignore = "needs a CUDA device; run under the 5090 lease"]
fn kquant_expert_banks_prefill_what_they_decode() {
    let _gpu = gpu_guard();
    println!(
        "[kquant] pairs program: {}",
        if std::env::var("MEMRA_GEMMA_MOE_MMA").as_deref() == Ok("0") {
            "dp4a (MEMRA_GEMMA_MOE_MMA=0)"
        } else {
            "int8 MMA default"
        }
    );
    for var in FOREIGN_ENV {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set: this gate holds the default device programs, unset it"
        );
    }
    let show = |e: &[Score]| {
        e.iter()
            .map(|s| format!("{:.4}/{:.4}/{:.2}", s.median, s.max, s.above))
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("[kquant] per width {WIDTHS:?}: median/max/share of rows above {CAP}");
    let mut failures = Vec::new();
    let mut controls: Vec<(Variant, (Vec<Score>, f32))> = Vec::new();
    for (control, variant) in [
        (CONTROL_704, Q3K_704),
        (CONTROL_704, Q4K_704),
        (CONTROL_512, Q6K_512),
    ] {
        if !controls.iter().any(|(c, _)| *c == control) {
            let run = errors(control);
            println!(
                "[kquant] control {control:?}: vs reference {}, prime-vs-decode {:e}",
                show(&run.0),
                run.1
            );
            controls.push((control, run));
        }
        let (c_err, c_pd) = controls
            .iter()
            .find(|(c, _)| *c == control)
            .map(|(_, run)| run.clone())
            .expect("the control ran");
        let (v_err, v_pd) = errors(variant);
        println!(
            "[kquant] variant {variant:?}: vs reference {}, prime-vs-decode {v_pd:e}",
            show(&v_err)
        );
        for (i, &n) in WIDTHS.iter().enumerate() {
            let (c_med, v) = (c_err[i].median, v_err[i]);
            let bound = (BAND * c_med).clamp(FLOOR, CAP);
            if !c_med.is_finite() {
                failures.push(format!(
                    "{control:?} T={n}: the control itself is not finite"
                ));
            }
            if v.median.is_nan() || v.median > bound {
                failures.push(format!(
                    "{variant:?} T={n}: median row error {:e} against the reference, bound \
                     {bound:e} (control {c_med:e})",
                    v.median
                ));
            }
            // One row is all or nothing; its score is the median above.
            if n > 1 && v.above > SHARE {
                failures.push(format!(
                    "{variant:?} T={n}: {:.2} of the rows are above {CAP} against the \
                     reference, at most {SHARE:.2} may be (control {:.2})",
                    v.above, c_err[i].above
                ));
            }
        }
        let bound = (BAND * c_pd).clamp(FLOOR, 2.0 * CAP);
        if !c_pd.is_finite() {
            failures.push(format!(
                "{control:?}: the control's prime-vs-decode is not finite"
            ));
        }
        if v_pd.is_nan() || v_pd > bound {
            failures.push(format!(
                "{variant:?}: 16-token prime vs 16 decode steps {v_pd:e}, bound {bound:e} \
                 (control {c_pd:e})"
            ));
        }

        let (resident, r_modes) = placement_rows(variant, false);
        let (tight, (row, pinned, chunked)) = placement_rows(variant, true);
        let differ: Vec<(usize, u64)> = PLACEMENT_ROWS
            .iter()
            .zip(resident.iter().zip(&tight))
            .filter(|(_, (r, t))| {
                r.len() != t.len() || r.iter().zip(*t).any(|(x, y)| x.to_bits() != y.to_bits())
            })
            .map(|(&row, _)| row)
            .collect();
        println!(
            "[kquant] variant {variant:?}: {TIGHT_SLOTS}-slot SLRU calls row={row} \
             pinned={pinned} chunked={chunked}; {} of {} (width, seed) rows differ from \
             resident: {differ:?}",
            differ.len(),
            PLACEMENT_ROWS.len()
        );
        if r_modes != (0, 0, 0) {
            failures.push(format!(
                "{variant:?}: the resident run took the SLRU arm {r_modes:?}"
            ));
        }
        if !(pinned > 0 && chunked > 0) {
            failures.push(format!(
                "{variant:?}: the tight SLRU run did not pin and chunk (row={row} \
                 pinned={pinned} chunked={chunked}); the placement check is vacuous"
            ));
        }
        if !differ.is_empty() {
            failures.push(format!(
                "{variant:?}: the SLRU placement differs from resident at (T, seed) {differ:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "k-quant expert banks do not compute the model: {failures:?}"
    );
}

/// CPU companion: the k-quant encoders produce blocks the engine's dequantizer reads back close
/// to the input, so the reference and the engine read the same function.
#[test]
fn the_kquant_encoders_round_trip() {
    let data: Vec<f32> = spread_router(4096, &TensorId::TokenEmbedding);
    for ty in [
        GgmlType::Q4_0,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q6_K,
    ] {
        let back = memra_gguf::dequant::dequantize(ty, &encode(ty, &data), data.len());
        let err = relative(&back, &data);
        println!("{ty:?}: round-trip relative error {err:e}");
        // Max-abs error over the max-abs value of uniform data: about half a step of the type's
        // bit width over a block's range, plus the scale's own rounding.
        let limit = match ty {
            GgmlType::Q6_K => 0.05,
            GgmlType::Q3_K => 0.25,
            _ => 0.15,
        };
        assert!(err < limit, "{ty:?} round trip {err:e} >= {limit}");
    }
}
