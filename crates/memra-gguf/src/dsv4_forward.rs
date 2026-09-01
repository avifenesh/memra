//! DeepSeek-V4-Flash CPU paired-forward (lane 3, research/dsv4-flash-loader-20260818).
//!
//! Semantic source (THE LAW): the official reference `inference/model.py` + `kernel.py`
//! as resolved in darklanes research/deepseek-flash-20260818/SEMANTICS.md (every mechanism
//! file:line-cited). The numeric contract is the lane-2 fixture generator
//! (`dsv4_cpu_ref_fixtures.py`): a float32 pipeline over weights dequantized with the
//! lane-1 oracle semantics (dsv4.rs decoders, bit-exact vs numpy), prefill-only,
//! start_pos = 0. This module is that same program in Rust:
//!
//!   - all GEMMs run on dequantized f32 weights (GPU activation FP8/FP4 GEMM quant is
//!     NOT emulated — same contract as the fixtures);
//!   - the three KV QAT round-trips ARE emulated because they are value semantics
//!     (model.py:506, :372, :368-370, :413-416): per-64 FP8 (pow2-ceil scale, clamp
//!     ±448, e4m3 RNE) on window/compressor kv nope dims, and per-32 FP4 (pow2-ceil,
//!     clamp ±6, e2m1 RNE) after a Sylvester Hadamard on the indexer q / indexer kv;
//!   - [`ActQuantVariant`] selects the kernel.py:88 fork: `RefFp8Round` = the reference
//!     law (cast through FP8), `ClampOnly` = the NVFP4 artifact's kernel (cast through
//!     out_dtype, i.e. FP8 rounding disabled). ONE variant per run — never mixed.
//!
//! Precision doctrine (why the gate is tolerance-based, not bit-exact): the fixtures
//! were banked with torch 2.13 CPU (blocked/vectorized f32 reductions, possibly FMA).
//! This port accumulates every dot product and long reduction in f64 and rounds once to
//! f32, so |rust − torch| ≤ |rust − exact| + |exact − torch| ≈ torch's own blocked-f32
//! rounding error (~sqrt(k)·2⁻²⁴ per length-k dot). Elementwise transcendental math
//! (exp/sigmoid/cos/sin/powf) runs in f32 like torch's, with ≤ a-few-ULP library skew.
//! The QAT round-trips are exact given equal inputs (pow2 scales, grid values); a tiny
//! upstream difference can flip one code at a rounding boundary, which is why the gate
//! carries a small named flip allowance on the quantized-kv arrays.
//!
//! Everything geometric is DERIVED from `DeepSeekV4Config` (compress_ratios,
//! num_hash_layers, hc_mult, o_groups, ...) — never hardcoded counts (lane-1 law).
//! Loading refuses loudly (named tensor) on NaN, missing tensors, or shape surprises.

use std::collections::BTreeMap;
use std::path::Path;

use crate::config::{DeepSeekV4Config, JsonObj, ModelConfig};
use crate::dsv4::{dequant_fp8_blk128, dequant_mxfp4_expert, dequant_nvfp4_expert};
use crate::nvfp4_repack::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use crate::safetensors::StModel;

/// kernel.py:88 fork. `RefFp8Round` = reference law (act_quant inplace casts through
/// FP8 → RNE round-trip). `ClampOnly` = the NVFP4 artifact's kernel (casts through
/// out_dtype/bf16 → the FP8 rounding of the window/compressor-KV QAT sim is disabled;
/// the f32 fixture contract treats the bf16 cast as identity, like every other bf16
/// hop it does not emulate). The FP4 sim is identical in both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActQuantVariant {
    RefFp8Round,
    ClampOnly,
}

impl ActQuantVariant {
    /// Fixture JSON `variant` field → variant. Anything else refuses.
    pub fn from_fixture_tag(tag: &str) -> Self {
        match tag {
            "ref" => ActQuantVariant::RefFp8Round,
            "clamp-only" => ActQuantVariant::ClampOnly,
            other => panic!("unknown fixture variant {other:?} (want \"ref\" | \"clamp-only\")"),
        }
    }
}

// ============================ lane 7: expert-GEMM numeric class ============================

/// bf16 unit roundoff (lane-4 threshold doctrine).
pub const U_BF16: f64 = 1.0 / 256.0; // 2^-8
/// e4m3 unit roundoff (lane-7 native class: 3 mantissa bits → spacing 2⁻³, RNE 2⁻⁴).
pub const U_FP8: f64 = 1.0 / 16.0; // 2^-4

/// The lane-7 arm seam, read ONCE per process: `MEMRA_DSV4_EXPERT_ARM=native` selects
/// the natively-quantized expert GEMM class on BOTH sides of every gate — the GPU
/// forward (memra-engine dsv4_gpu) runs the native NVFP4/MXFP4 kernels and this CPU
/// oracle emulates the reference expert act-quant (model.py:113-115: act_quant per-128
/// with pow2-ceil scales and REAL FP8 rounding — the GEMM path rounds in BOTH kernel.py
/// variants; the clamp-only fork is inplace-KV-QAT-only). One seam for both sides so an
/// invocation cannot mix numeric classes.
pub fn expert_arm_native() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("MEMRA_DSV4_EXPERT_ARM")
            .map(|v| v == "native")
            .unwrap_or(false)
    })
}

/// Per-array drift coefficient of the lane-4 random-walk doctrine, generalized to two
/// hop classes (lane-7 derivation, banked): C = √(d_b·u_b² + d_q·u_q²). The bf16 class
/// is C(d, 0) == u_b·√d (lane 4 unchanged).
pub fn drift_coeff(d_b: f64, d_q: f64) -> f64 {
    (d_b * U_BF16 * U_BF16 + d_q * U_FP8 * U_FP8).sqrt()
}

/// Quantized-hop count upstream of a fixture array (2 per MoE sub-block: the w1/w3
/// input quant and the w2 input quant of that layer's routed experts). Mirrors the
/// lane-4 `depth_of` naming contract.
pub fn quant_depth_of(name: &str) -> f64 {
    let core = name.strip_prefix("c160_").unwrap_or(name);
    if core == "final_logits_last" {
        return 86.0;
    }
    if core == "mtp_logits_last" {
        return 88.0;
    }
    let n: f64 = core
        .strip_prefix("layer")
        .and_then(|r| r.split('_').next())
        .and_then(|x| x.parse().ok())
        .unwrap_or(0.0);
    if core.contains("_out") && !core.contains("attn_out") {
        2.0 * (n + 1.0)
    } else {
        // attn_out, compressor/indexer arrays: only layers < n contribute MoE hops
        2.0 * n
    }
}

// ============================ parallel helper (no new deps) ============================

/// Run `f(row_index, row)` over `out` split into `row_len` rows across available cores.
/// Each row is produced by exactly one closure call — deterministic output regardless of
/// thread count (parallelism never changes any accumulation order).
pub fn par_rows<F>(out: &mut [f32], row_len: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    assert_eq!(out.len() % row_len.max(1), 0);
    let rows = out.len().checked_div(row_len).unwrap_or(0);
    if rows == 0 {
        return;
    }
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(rows);
    let chunk_rows = rows.div_ceil(n_threads);
    std::thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(chunk_rows * row_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (ri, row) in chunk.chunks_mut(row_len).enumerate() {
                    f(ci * chunk_rows + ri, row);
                }
            });
        }
    });
}

/// Map `f` over 0..n on all cores, collecting results in index order.
fn par_map<T: Send, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T + Sync,
{
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    if n == 0 {
        return Vec::new();
    }
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
        .min(n);
    let chunk = n.div_ceil(n_threads);
    std::thread::scope(|s| {
        for (ci, slot_chunk) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (ri, slot) in slot_chunk.iter_mut().enumerate() {
                    *slot = Some(f(ci * chunk + ri));
                }
            });
        }
    });
    out.into_iter().map(|x| x.unwrap()).collect()
}

// ============================ numeric primitives ============================

/// f64-accumulated dot, rounded once to f32 (precision doctrine in the module header).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f64;
    for (x, y) in a.iter().zip(b) {
        acc += *x as f64 * *y as f64;
    }
    acc as f32
}

/// x [s, k] @ w[n, k]ᵀ -> [s, n]; every output element is one f64-accumulated dot.
/// Parallel over w rows — call [`matmul_serial`] from inside an already-parallel region
/// (nested scoped spawns would oversubscribe the box).
pub fn matmul(x: &[f32], s: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), s * k, "matmul x shape");
    assert_eq!(w.len(), n * k, "matmul w shape");
    // compute transposed [n, s] so threads own contiguous chunks, then transpose.
    let mut out_t = vec![0f32; n * s];
    par_rows(&mut out_t, s, |j, row| {
        let wr = &w[j * k..(j + 1) * k];
        for (i, o) in row.iter_mut().enumerate() {
            *o = dot(&x[i * k..(i + 1) * k], wr);
        }
    });
    if s == 1 {
        return out_t;
    }
    let mut out = vec![0f32; s * n];
    for j in 0..n {
        for i in 0..s {
            out[i * n + j] = out_t[j * s + i];
        }
    }
    out
}

/// Single-threaded [`matmul`] (identical arithmetic and accumulation order — every
/// output element is one sequential f64 dot either way).
pub fn matmul_serial(x: &[f32], s: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(x.len(), s * k, "matmul x shape");
    assert_eq!(w.len(), n * k, "matmul w shape");
    let mut out = vec![0f32; s * n];
    for i in 0..s {
        let xr = &x[i * k..(i + 1) * k];
        for j in 0..n {
            out[i * n + j] = dot(xr, &w[j * k..(j + 1) * k]);
        }
    }
    out
}

/// RMSNorm rows (model.py:191-196): `w * (x * rsqrt(mean(x²) + eps))`, f32 elementwise,
/// f64 mean accumulation.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    assert_eq!(x.len() % d, 0, "rmsnorm width");
    let mut out = vec![0f32; x.len()];
    for (row, orow) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut acc = 0f64;
        for v in row {
            acc += (*v as f64) * (*v as f64);
        }
        let rsq = 1.0f32 / ((acc / d as f64) as f32 + eps).sqrt();
        for i in 0..d {
            orow[i] = w[i] * (row[i] * rsq);
        }
    }
    out
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// torch.nn.functional.softplus (beta=1, threshold=20): x>20 → x, else log1p(exp(x)).
#[inline]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

// ============================ RoPE (model.py:199-244) ============================

/// Per-position complex rope table: `cs[pos * half + k] = (cos, sin)` of `pos·f_k`.
pub struct FreqsCis {
    pub half: usize,
    pub cs: Vec<(f32, f32)>,
}

/// model.py precompute_freqs_cis: base frequencies with optional YaRN frequency
/// interpolation (no mscale anywhere — model.py:464). `orig_ctx == 0` disables YaRN
/// (the ratio-0 layers: theta 10000, pure sliding window).
pub fn precompute_freqs_cis(
    dim: usize,
    seqlen: usize,
    orig_ctx: u32,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> FreqsCis {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|k| 1.0f32 / base.powf((2 * k) as f32 / dim as f32))
        .collect();
    if orig_ctx > 0 {
        // find_correction_dim/-range in f64 (python math.log), then f32 ramp mixing.
        let fcd = |num_rot: f64| -> f64 {
            dim as f64 * (orig_ctx as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = fcd(beta_fast as f64).floor().max(0.0);
        let high = fcd(beta_slow as f64).ceil().min((dim - 1) as f64);
        let (mn, mx) = (low, if low == high { high + 0.001 } else { high });
        let denom = (mx - mn) as f32;
        for (k, f) in freqs.iter_mut().enumerate() {
            let ramp = ((k as f32 - mn as f32) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut cs = Vec::with_capacity(seqlen * half);
    for t in 0..seqlen {
        for f in &freqs {
            let ang = t as f32 * *f;
            cs.push((ang.cos(), ang.sin()));
        }
    }
    FreqsCis { half, cs }
}

/// In-place interleaved complex rope on the LAST `rd` dims of each `dim`-wide vector
/// (model.py:232-244). `x` is [n_pos, n_vec, dim]; `positions[p]` selects the freq row.
/// `inverse` uses the conjugate (the de-rotation at the query position, model.py:534).
#[allow(clippy::too_many_arguments)]
pub fn apply_rope(
    x: &mut [f32],
    n_pos: usize,
    n_vec: usize,
    dim: usize,
    rd: usize,
    fc: &FreqsCis,
    positions: &[usize],
    inverse: bool,
) {
    assert_eq!(x.len(), n_pos * n_vec * dim);
    assert_eq!(positions.len(), n_pos);
    assert_eq!(fc.half * 2, rd, "freq table half must be rd/2");
    for (p, &pos) in positions.iter().enumerate() {
        let frow = &fc.cs[pos * fc.half..(pos + 1) * fc.half];
        for v in 0..n_vec {
            let base = (p * n_vec + v) * dim + (dim - rd);
            for (k, &(c, s0)) in frow.iter().enumerate() {
                let s = if inverse { -s0 } else { s0 };
                let x0 = x[base + 2 * k];
                let x1 = x[base + 2 * k + 1];
                x[base + 2 * k] = x0 * c - x1 * s;
                x[base + 2 * k + 1] = x0 * s + x1 * c;
            }
        }
    }
}

// ============================ QAT sims (kernel.py act_quant / fp4_act_quant) ============================

/// fast_round_scale (kernel.py:22-37): 2^ceil(log2(x)) via exact f32 bit manipulation.
/// Caller guarantees x is a positive normal (amax floors 1e-4 / 6·2⁻¹²⁶ ensure it).
#[inline]
pub fn pow2_ceil(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & ((1 << 23) - 1);
    let e = exp - 127 + i32::from(man != 0);
    (e as f32).exp2()
}

/// kernel.py act_quant(..., round_scale=True, inplace=True) on contiguous groups of
/// `block`: per-group pow2 scale (amax·(1/448) ceil-pow2, amax floored 1e-4), clamp
/// ±448, then FP8-E4M3 RNE round-trip (ref) or nothing more (clamp-only fork, §7.6).
pub fn act_quant(x: &mut [f32], block: usize, variant: ActQuantVariant) {
    assert_eq!(x.len() % block, 0, "act_quant block divisibility");
    let inv = (1.0f64 / 448.0) as f32;
    for g in x.chunks_exact_mut(block) {
        let mut amax = 0f32;
        for v in g.iter() {
            amax = amax.max(v.abs());
        }
        amax = amax.max(1e-4);
        let s = pow2_ceil(amax * inv);
        for v in g.iter_mut() {
            let q = (*v / s).clamp(-448.0, 448.0);
            let q = match variant {
                ActQuantVariant::RefFp8Round => fp8_e4m3_to_f32(f32_to_fp8_e4m3(q)),
                ActQuantVariant::ClampOnly => q,
            };
            *v = q * s;
        }
    }
}

/// Round f32 onto the e2m1 grid, round-to-nearest, ties to even mantissa bit
/// (fixture generator e2m1_rne; identical thresholds).
#[inline]
pub fn e2m1_rne(v: f32) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = v.abs();
    let mut idx = 0usize;
    for t in [0.25f32, 1.25, 2.5, 5.0] {
        // midpoint below an ODD-mantissa upper neighbour: go up only if strictly greater
        idx += usize::from(a > t);
    }
    for t in [0.75f32, 1.75, 3.5] {
        // midpoint below an EVEN-mantissa upper neighbour: ties go up
        idx += usize::from(a >= t);
    }
    if v < 0.0 { -GRID[idx] } else { GRID[idx] }
}

/// kernel.py fp4_act_quant(..., inplace=True) on contiguous groups of 32: pow2 scale
/// (amax·(1/6) ceil-pow2, amax floored 6·2⁻¹²⁶), clamp ±6, e2m1 RNE round-trip.
/// Identical in both kernel variants.
pub fn fp4_act_quant(x: &mut [f32], block: usize) {
    assert_eq!(x.len() % block, 0, "fp4_act_quant block divisibility");
    let inv = (1.0f64 / 6.0) as f32;
    let floor = 6.0f32 * 2f32.powi(-126);
    for g in x.chunks_exact_mut(block) {
        let mut amax = 0f32;
        for v in g.iter() {
            amax = amax.max(v.abs());
        }
        amax = amax.max(floor);
        let s = pow2_ceil(amax * inv);
        for v in g.iter_mut() {
            *v = e2m1_rne((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

/// Sylvester-order Walsh–Hadamard transform, scale d^-0.5, in place on each `d`-chunk
/// (fast_hadamard_transform semantics; model.py:247-251). `d` must be a power of two.
pub fn hadamard(x: &mut [f32], d: usize) {
    assert!(d.is_power_of_two(), "hadamard needs pow2 dim");
    assert_eq!(x.len() % d, 0);
    let scale = (d as f32).powf(-0.5);
    for row in x.chunks_exact_mut(d) {
        let mut h = 1;
        while h < d {
            let mut base = 0;
            while base < d {
                for i in base..base + h {
                    let a = row[i];
                    let b = row[i + h];
                    row[i] = a + b;
                    row[i + h] = a - b;
                }
                base += 2 * h;
            }
            h *= 2;
        }
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

// ============================ index builders (model.py:255-276) ============================

/// Sliding-window topk indices: row i covers max(0, i-win+1)..=i, -1 padding above i.
/// Shape [s, min(s, win)].
pub fn window_topk_idxs(win: usize, s: usize) -> (Vec<i64>, usize) {
    let w = s.min(win);
    let mut m = vec![-1i64; s * w];
    for i in 0..s {
        let start = i.saturating_sub(win - 1);
        for j in 0..w {
            let v = (start + j) as i64;
            m[i * w + j] = if v > i as i64 { -1 } else { v };
        }
    }
    (m, w)
}

/// Deterministic all-completed-blocks indices for coarse layers: block j attendable by
/// query i iff j < (i+1)/ratio; else -1. Shape [s, s/ratio], entries offset by `offset`.
pub fn compress_topk_idxs(ratio: usize, s: usize, offset: usize) -> (Vec<i64>, usize) {
    let nb = s / ratio;
    let mut m = vec![-1i64; s * nb];
    for i in 0..s {
        let lim = (i + 1) / ratio;
        for j in 0..nb {
            m[i * nb + j] = if j >= lim { -1 } else { (j + offset) as i64 };
        }
    }
    (m, nb)
}

// ============================ weight loading ============================

/// The opened artifact + parsed config. All tensor reads go through [`Dsv4Model::tensor_f32`]
/// which mirrors the fixture generator's `get()`: quant sibling sets decode through the
/// lane-1 oracle decoders, BF16/F32 load verbatim, NaN refuses by name.
pub struct Dsv4Model {
    pub st: StModel,
    pub mc: ModelConfig,
}

fn bf16_to_f32_vec(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

impl Dsv4Model {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ModelConfig::from_config_json(&dir.join("config.json"))
        }))
        .map_err(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "config.json parser panicked".into())
        })?;
        let mc = parsed.map_err(|error| format!("parse config.json: {error}"))?;
        if !mc.arch.is_dsv4() {
            return Err(format!(
                "dsv4 forward: model_type is not deepseek_v4 (arch {:?})",
                mc.arch
            ));
        }
        let st = StModel::open(dir).map_err(|error| format!("open safetensors model: {error}"))?;
        Ok(Dsv4Model { st, mc })
    }

    pub fn cfg(&self) -> &DeepSeekV4Config {
        self.mc.dsv4.as_ref().expect("dsv4 config")
    }

    /// Raw tensor presence (0731 lane: the GPU loader discriminates NextN vs DSpark
    /// `mtp.*` namespaces on stored structure, lane-1 style).
    pub fn has(&self, name: &str) -> bool {
        self.st.raw(name).is_some()
    }

    /// Any tensor stem or raw name -> f32 (fixture generator `get()` semantics).
    pub fn tensor_f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (shape, v) = self.tensor_f32_inner(name);
        for x in &v {
            assert!(!x.is_nan(), "NaN in decoded {name}");
        }
        (shape, v)
    }

    fn tensor_f32_inner(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        if self.has(&format!("{name}.weight_scale_2")) {
            let (wi, wb) = self
                .st
                .raw(&format!("{name}.weight"))
                .expect("nvfp4 weight");
            let (_, sb) = self
                .st
                .raw(&format!("{name}.weight_scale"))
                .expect("nvfp4 scale");
            let (_, s2b) = self
                .st
                .raw(&format!("{name}.weight_scale_2"))
                .expect("nvfp4 scale_2");
            let rows = wi.shape[0] as usize;
            let cols = wi.shape[1] as usize * 2;
            let s2 = f32::from_le_bytes(s2b.try_into().expect("scale_2 4B"));
            return (
                vec![rows, cols],
                dequant_nvfp4_expert(wb, sb, s2, rows, cols),
            );
        }
        if self.has(&format!("{name}.scale")) {
            let (wi, wb) = self
                .st
                .raw(&format!("{name}.weight"))
                .expect("quant weight");
            let (_, sb) = self.st.raw(&format!("{name}.scale")).expect("quant scale");
            let rows = wi.shape[0] as usize;
            if wi.dtype == "I8" {
                let cols = wi.shape[1] as usize * 2;
                return (vec![rows, cols], dequant_mxfp4_expert(wb, sb, rows, cols));
            }
            assert_eq!(
                wi.dtype, "F8_E4M3",
                "{name}: unexpected quant dtype {}",
                wi.dtype
            );
            let cols = wi.shape[1] as usize;
            return (vec![rows, cols], dequant_fp8_blk128(wb, sb, rows, cols));
        }
        let full;
        let key = if self.has(name) {
            name
        } else {
            full = format!("{name}.weight");
            &full
        };
        let (info, raw) = self
            .st
            .raw(key)
            .unwrap_or_else(|| panic!("tensor {name} not found (tried {key})"));
        let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
        let v = match info.dtype.as_str() {
            "BF16" => bf16_to_f32_vec(raw),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => panic!("unhandled dtype {other} for {name}"),
        };
        (shape, v)
    }

    pub fn tensor_i64(&self, name: &str) -> (Vec<usize>, Vec<i64>) {
        let (info, raw) = self
            .st
            .raw(name)
            .unwrap_or_else(|| panic!("tensor {name} not found"));
        assert_eq!(info.dtype, "I64", "{name}: expected I64");
        let shape: Vec<usize> = info.shape.iter().map(|&x| x as usize).collect();
        let v = raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, v)
    }

    /// Decode embedding rows for `ids` (BF16 rows -> f32, no arithmetic — bit-exact
    /// against the fixture's embed_out).
    pub fn embed_rows(&self, ids: &[u32]) -> Vec<f32> {
        let (info, raw) = self.st.raw("embed.weight").expect("embed.weight");
        assert_eq!(info.dtype, "BF16");
        let h = info.shape[1] as usize;
        let mut out = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            let row = &raw[(id as usize) * h * 2..((id as usize) + 1) * h * 2];
            out.extend(bf16_to_f32_vec(row));
        }
        out
    }

    /// logits = x [4096] @ head.weight [v, 4096]ᵀ, decoding BF16 rows on the fly.
    pub fn head_logits(&self, x: &[f32]) -> Vec<f32> {
        let (info, raw) = self.st.raw("head.weight").expect("head.weight");
        assert_eq!(info.dtype, "BF16");
        let v = info.shape[0] as usize;
        let h = info.shape[1] as usize;
        assert_eq!(x.len(), h);
        let mut out = vec![0f32; v];
        par_rows(&mut out, 1, |j, o| {
            let row = bf16_to_f32_vec(&raw[j * h * 2..(j + 1) * h * 2]);
            o[0] = dot(x, &row);
        });
        out
    }
}

// ============================ hyper-connections (model.py:663-735, kernel hc_split_sinkhorn) ============================

/// One hc parameter family: fn [rows, hc_mult·hidden], base [rows], scale [3 or 1].
pub struct HcSet {
    pub fn_w: Vec<f32>,
    pub base: Vec<f32>,
    pub scale: Vec<f32>,
    pub rows: usize,
}

/// hc_split_sinkhorn (kernel.py:372-438): rows [0:hc]=pre, [hc:2hc]=post, [2hc:]=comb;
/// comb = row-softmax + eps, column-normalize, then iters-1 more (row, col) pairs.
/// Returns (pre [s,hc], post [s,hc], comb [s,hc,hc]).
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    s: usize,
    hc: usize,
    scale: &[f32],
    base: &[f32],
    iters: u32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = (2 + hc) * hc;
    assert_eq!(mixes.len(), s * rows);
    assert_eq!(base.len(), rows);
    assert_eq!(scale.len(), 3);
    let mut pre = vec![0f32; s * hc];
    let mut post = vec![0f32; s * hc];
    let mut comb = vec![0f32; s * hc * hc];
    for t in 0..s {
        let m = &mixes[t * rows..(t + 1) * rows];
        for c in 0..hc {
            pre[t * hc + c] = sigmoid_f32(m[c] * scale[0] + base[c]) + eps;
            post[t * hc + c] = 2.0 * sigmoid_f32(m[hc + c] * scale[1] + base[hc + c]);
        }
        let cm = &mut comb[t * hc * hc..(t + 1) * hc * hc];
        for j in 0..hc {
            for k in 0..hc {
                cm[j * hc + k] = m[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
            }
        }
        // row softmax + eps
        for j in 0..hc {
            let row = &mut cm[j * hc..(j + 1) * hc];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v = *v / sum + eps;
            }
        }
        // column normalize, then (iters-1) x (row, col)
        let col_norm = |cm: &mut [f32]| {
            for k in 0..hc {
                let mut sum = 0f32;
                for j in 0..hc {
                    sum += cm[j * hc + k];
                }
                for j in 0..hc {
                    cm[j * hc + k] /= sum + eps;
                }
            }
        };
        let row_norm = |cm: &mut [f32]| {
            for j in 0..hc {
                let mut sum = 0f32;
                for k in 0..hc {
                    sum += cm[j * hc + k];
                }
                for k in 0..hc {
                    cm[j * hc + k] /= sum + eps;
                }
            }
        };
        col_norm(cm);
        for _ in 0..iters.saturating_sub(1) {
            row_norm(cm);
            col_norm(cm);
        }
    }
    (pre, post, comb)
}

/// hc_pre (model.py:673-681): collapse hc copies -> 1. x is [s, hc, d].
/// Returns (y [s,d], post [s,hc], comb [s,hc,hc]).
pub fn hc_pre(
    x: &[f32],
    s: usize,
    hc: usize,
    d: usize,
    set: &HcSet,
    iters: u32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w = hc * d;
    assert_eq!(x.len(), s * w);
    assert_eq!(set.rows, (2 + hc) * hc, "hc rows must be (2+hc)*hc");
    let mut mixes = matmul(x, s, w, &set.fn_w, set.rows);
    for t in 0..s {
        let xf = &x[t * w..(t + 1) * w];
        let mut acc = 0f64;
        for v in xf {
            acc += (*v as f64) * (*v as f64);
        }
        let rsq = 1.0f32 / ((acc / w as f64) as f32 + eps).sqrt();
        for v in &mut mixes[t * set.rows..(t + 1) * set.rows] {
            *v *= rsq;
        }
    }
    let (pre, post, comb) = hc_split_sinkhorn(&mixes, s, hc, &set.scale, &set.base, iters, eps);
    let mut y = vec![0f32; s * d];
    for t in 0..s {
        for c in 0..hc {
            let p = pre[t * hc + c];
            let src = &x[(t * hc + c) * d..(t * hc + c + 1) * d];
            let dst = &mut y[t * d..(t + 1) * d];
            for i in 0..d {
                dst[i] += p * src[i];
            }
        }
    }
    (y, post, comb)
}

/// hc_post (model.py:683-686): new[k] = post[k]·f + Σ_j comb[j,k]·residual[j].
pub fn hc_post(
    f: &[f32],
    residual: &[f32],
    s: usize,
    hc: usize,
    d: usize,
    post: &[f32],
    comb: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; s * hc * d];
    for t in 0..s {
        for k in 0..hc {
            let dst = &mut out[(t * hc + k) * d..(t * hc + k + 1) * d];
            let pf = post[t * hc + k];
            let frow = &f[t * d..(t + 1) * d];
            for i in 0..d {
                dst[i] = pf * frow[i];
            }
            for j in 0..hc {
                let c = comb[t * hc * hc + j * hc + k];
                let rrow = &residual[(t * hc + j) * d..(t * hc + j + 1) * d];
                for i in 0..d {
                    dst[i] += c * rrow[i];
                }
            }
        }
    }
    out
}

/// ParallelHead.hc_head (model.py:728-735): pre-only collapse (sigmoid gates + hc_eps).
pub fn hc_head(
    x: &[f32],
    s: usize,
    hc: usize,
    d: usize,
    set: &HcSet,
    eps: f32,
    hc_eps: f32,
) -> Vec<f32> {
    let w = hc * d;
    assert_eq!(set.rows, hc, "hc_head rows must be hc_mult");
    assert_eq!(set.scale.len(), 1);
    let mut mixes = matmul(x, s, w, &set.fn_w, hc);
    let mut y = vec![0f32; s * d];
    for t in 0..s {
        let xf = &x[t * w..(t + 1) * w];
        let mut acc = 0f64;
        for v in xf {
            acc += (*v as f64) * (*v as f64);
        }
        let rsq = 1.0f32 / ((acc / w as f64) as f32 + eps).sqrt();
        for c in 0..hc {
            let mix = mixes[t * hc + c] * rsq;
            let pre = sigmoid_f32(mix * set.scale[0] + set.base[c]) + hc_eps;
            mixes[t * hc + c] = pre;
            let src = &x[(t * hc + c) * d..(t * hc + c + 1) * d];
            let dst = &mut y[t * d..(t + 1) * d];
            for i in 0..d {
                dst[i] += pre * src[i];
            }
        }
    }
    y
}

// ============================ compressor (model.py:279-377) ============================

/// Learned gated softmax pooling over ratio-blocks. Two shape classes (lane-1 census):
/// fine ratio-4 pools over 8 positions via the coff=2 overlap; coarse ratio-128 plain.
pub struct CompressorW {
    pub ratio: usize,
    pub d: usize,
    pub latent: usize,
    pub overlap: bool,
    pub rotate: bool,
    pub wkv: Vec<f32>,
    pub wgate: Vec<f32>,
    pub norm_w: Vec<f32>,
    pub ape: Vec<f32>,
}

impl CompressorW {
    pub fn load(model: &Dsv4Model, prefix: &str, ratio: usize, d: usize, rotate: bool) -> Self {
        let (wkv_shape, wkv) = model.tensor_f32(&format!("{prefix}.wkv.weight"));
        let (_, wgate) = model.tensor_f32(&format!("{prefix}.wgate.weight"));
        let (_, norm_w) = model.tensor_f32(&format!("{prefix}.norm.weight"));
        let (ape_shape, ape) = model.tensor_f32(&format!("{prefix}.ape"));
        let latent = wkv_shape[0];
        let overlap = ratio == 4; // coff = 2 iff ratio == 4 (model.py:290-292)
        assert_eq!(
            latent,
            if overlap { 2 * d } else { d },
            "{prefix}: latent width"
        );
        assert_eq!(ape_shape, vec![ratio, latent], "{prefix}.ape shape");
        CompressorW {
            ratio,
            d,
            latent,
            overlap,
            rotate,
            wkv,
            wgate,
            norm_w,
            ape,
        }
    }

    /// Prefill forward: x [s, hidden] (the post-attn_norm activations, f32) ->
    /// Some(([nb, d], nb)) or None when s < ratio. QAT applied per §7.5:
    /// attention side (rotate=false): per-64 FP8 on nope dims; indexer side
    /// (rotate=true): Hadamard then per-32 FP4 on all dims.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        s: usize,
        hidden: usize,
        fc: &FreqsCis,
        rd: usize,
        eps: f32,
        variant: ActQuantVariant,
    ) -> Option<(Vec<f32>, usize)> {
        let (ratio, d, latent) = (self.ratio, self.d, self.latent);
        if s < ratio {
            return None;
        }
        let kv = matmul(x, s, hidden, &self.wkv, latent);
        let mut score = matmul(x, s, hidden, &self.wgate, latent);
        let cutoff = s - s % ratio;
        let nb = cutoff / ratio;
        // score[j, p, :] += ape[p, :]
        for j in 0..nb {
            for p in 0..ratio {
                let row = &mut score[(j * ratio + p) * latent..(j * ratio + p + 1) * latent];
                for (v, a) in row.iter_mut().zip(&self.ape[p * latent..(p + 1) * latent]) {
                    *v += *a;
                }
            }
        }
        // gather pooling positions per block: overlap doubles them (prev block through
        // dims [0:d], current through [d:2d]; block 0's prev half is -inf-masked).
        let positions = if self.overlap { 2 * ratio } else { ratio };
        let mut out = vec![0f32; nb * d];
        for j in 0..nb {
            // per output channel: softmax over the pooling positions, then weighted sum
            for c in 0..d {
                let mut sc = Vec::with_capacity(positions);
                let mut kvv = Vec::with_capacity(positions);
                if self.overlap {
                    for p in 0..ratio {
                        // first ratio slots: previous block, channel c (dims [0:d])
                        if j == 0 {
                            sc.push(f32::NEG_INFINITY);
                            kvv.push(0.0f32);
                        } else {
                            sc.push(score[((j - 1) * ratio + p) * latent + c]);
                            kvv.push(kv[((j - 1) * ratio + p) * latent + c]);
                        }
                    }
                    for p in 0..ratio {
                        // last ratio slots: current block, channel d + c
                        sc.push(score[(j * ratio + p) * latent + d + c]);
                        kvv.push(kv[(j * ratio + p) * latent + d + c]);
                    }
                } else {
                    for p in 0..ratio {
                        sc.push(score[(j * ratio + p) * latent + c]);
                        kvv.push(kv[(j * ratio + p) * latent + c]);
                    }
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0f64;
                let mut num = 0f64;
                for p in 0..positions {
                    let e = (sc[p] - mx).exp();
                    den += e as f64;
                    num += e as f64 * kvv[p] as f64;
                }
                out[j * d + c] = (num / den) as f32;
            }
        }
        let mut out = rmsnorm(&out, &self.norm_w, eps);
        let positions_rope: Vec<usize> = (0..nb).map(|j| j * ratio).collect();
        apply_rope(&mut out, nb, 1, d, rd, fc, &positions_rope, false);
        if self.rotate {
            hadamard(&mut out, d);
            fp4_act_quant(&mut out, 32);
        } else {
            for row in out.chunks_exact_mut(d) {
                act_quant(&mut row[..d - rd], 64, variant);
            }
        }
        Some((out, nb))
    }
}

// ============================ indexer (model.py:380-433) ============================

pub struct IndexerW {
    pub wq_b: Vec<f32>,         // [heads*hd, q_lora]
    pub weights_proj: Vec<f32>, // [heads, hidden]
    pub compressor: CompressorW,
    pub heads: usize,
    pub hd: usize,
    pub topk: usize,
}

pub struct IndexerOut {
    /// [s, slots] selected indices offset into the concatenated kv (-1 = masked);
    /// slots = min(index_topk, n_blocks).
    pub idxs: Vec<i64>,
    pub slots: usize,
    /// captured fixture arrays: the FP4-quantized compressed kv [nb, hd] and the
    /// post-mask index score [s, nb] (nb = block count, its own column width).
    pub indexer_kv: Option<(Vec<f32>, usize)>,
    pub index_score: Option<(Vec<f32>, usize)>,
}

impl IndexerW {
    pub fn load(model: &Dsv4Model, prefix: &str, ratio: usize) -> Self {
        let d = model.cfg();
        let heads = d.index_n_heads as usize;
        let hd = d.index_head_dim as usize;
        let (_, wq_b) = model.tensor_f32(&format!("{prefix}.wq_b"));
        let (_, weights_proj) = model.tensor_f32(&format!("{prefix}.weights_proj.weight"));
        let compressor = CompressorW::load(model, &format!("{prefix}.compressor"), ratio, hd, true);
        IndexerW {
            wq_b,
            weights_proj,
            compressor,
            heads,
            hd,
            topk: d.index_topk as usize,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        qr: &[f32],
        s: usize,
        hidden: usize,
        q_lora: usize,
        offset: usize,
        fc: &FreqsCis,
        rd: usize,
        eps: f32,
        variant: ActQuantVariant,
        capture: bool,
    ) -> IndexerOut {
        let (heads, hd) = (self.heads, self.hd);
        let ratio = self.compressor.ratio;
        let mut q = matmul(qr, s, q_lora, &self.wq_b, heads * hd); // [s, heads, hd]
        let positions: Vec<usize> = (0..s).collect();
        apply_rope(&mut q, s, heads, hd, rd, fc, &positions, false);
        hadamard(&mut q, hd);
        fp4_act_quant(&mut q, 32);
        let ckv = self.compressor.forward(x, s, hidden, fc, rd, eps, variant);
        let indexer_kv = if capture { ckv.clone() } else { None };
        // weights_proj is BF16 (model.py:394); scale = hd^-0.5 * heads^-0.5 (f64 -> f32)
        let scale = ((self.hd as f64).powf(-0.5) * (self.heads as f64).powf(-0.5)) as f32;
        let mut weights = matmul(x, s, hidden, &self.weights_proj, heads);
        for v in &mut weights {
            *v *= scale;
        }
        let Some((ckv, nb)) = ckv else {
            return IndexerOut {
                idxs: Vec::new(),
                slots: 0,
                indexer_kv,
                index_score: None,
            };
        };
        // score[t, j] = sum_h relu(q[t,h]·ckv[j]) * weights[t,h]; then causal mask -inf
        let mut score = vec![0f32; s * nb];
        par_rows(&mut score, nb, |t, row| {
            let lim = (t + 1) / ratio;
            for (j, o) in row.iter_mut().enumerate() {
                if j >= lim {
                    *o = f32::NEG_INFINITY;
                    continue;
                }
                let mut acc = 0f64;
                for h in 0..heads {
                    let sc = dot(
                        &q[(t * heads + h) * hd..(t * heads + h + 1) * hd],
                        &ckv[j * hd..(j + 1) * hd],
                    );
                    let r = sc.max(0.0);
                    acc += (r * weights[t * heads + h]) as f64;
                }
                *o = acc as f32;
            }
        });
        let index_score = if capture {
            Some((score.clone(), nb))
        } else {
            None
        };
        // topk over blocks (k = min(index_topk, nb)), value desc / index asc on ties,
        // then re-mask: block >= (t+1)/ratio -> -1, else + offset (model.py:508-510).
        let k = self.topk.min(nb);
        let mut idxs = vec![-1i64; s * k];
        for t in 0..s {
            let lim = (t + 1) / ratio;
            let mut order: Vec<usize> = (0..nb).collect();
            order.sort_by(|&a, &b| {
                score[t * nb + b]
                    .partial_cmp(&score[t * nb + a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            for (slot, &j) in order.iter().take(k).enumerate() {
                idxs[t * k + slot] = if j >= lim { -1 } else { (j + offset) as i64 };
            }
        }
        IndexerOut {
            idxs,
            slots: k,
            indexer_kv,
            index_score,
        }
    }
}

// ============================ attention (model.py:436-544, kernel sparse_attn) ============================

pub struct AttnW {
    pub layer_id: u32,
    pub ratio: usize,
    pub sink: Vec<f32>,
    pub wq_a: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub compressor: Option<CompressorW>,
    pub indexer: Option<IndexerW>,
    pub fc: FreqsCis,
}

/// Per-layer capture of the fixture arrays produced inside attention.
#[derive(Default)]
pub struct AttnCapture {
    pub compressor_kv: Option<(Vec<f32>, usize)>,
    pub indexer_kv: Option<(Vec<f32>, usize)>,
    pub index_score: Option<(Vec<f32>, usize)>,
}

impl AttnW {
    pub fn load(model: &Dsv4Model, prefix: &str, layer_id: u32, max_seq: usize) -> Self {
        let d = model.cfg();
        let ratio = d.compress_ratio(layer_id) as usize;
        let rd = d.qk_rope_head_dim as usize;
        let hd = d.head_dim as usize;
        // theta/yarn selection (model.py:475-482): compressor layers ride
        // compress_rope_theta + YaRN; ratio-0 layers theta rope_theta, YaRN off.
        let (theta, orig) = if ratio != 0 {
            (d.compress_rope_theta, d.rope_yarn_orig_ctx)
        } else {
            (model.mc.rope_freq_base, 0)
        };
        let fc = precompute_freqs_cis(
            rd,
            max_seq,
            orig,
            theta,
            d.rope_yarn_factor,
            d.rope_yarn_beta_fast,
            d.rope_yarn_beta_slow,
        );
        // parallel dequant of the five FP8 linears (each decode is serial + proven).
        let names = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"];
        let mut mats = par_map(names.len(), |i| {
            model.tensor_f32(&format!("{prefix}.{}", names[i])).1
        });
        let wo_b = mats.pop().unwrap();
        let wo_a = mats.pop().unwrap();
        let wkv = mats.pop().unwrap();
        let wq_b = mats.pop().unwrap();
        let wq_a = mats.pop().unwrap();
        let (_, sink) = model.tensor_f32(&format!("{prefix}.attn_sink"));
        let (_, q_norm) = model.tensor_f32(&format!("{prefix}.q_norm.weight"));
        let (_, kv_norm) = model.tensor_f32(&format!("{prefix}.kv_norm.weight"));
        let compressor = if ratio != 0 {
            Some(CompressorW::load(
                model,
                &format!("{prefix}.compressor"),
                ratio,
                hd,
                false,
            ))
        } else {
            None
        };
        let indexer = if ratio == 4 {
            Some(IndexerW::load(model, &format!("{prefix}.indexer"), ratio))
        } else {
            None
        };
        AttnW {
            layer_id,
            ratio,
            sink,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            wo_a,
            wo_b,
            compressor,
            indexer,
            fc,
        }
    }

    /// Prefill forward: x [s, hidden] -> [s, hidden]. Mirrors model.py:496-542 exactly
    /// (shared K==V latent, in-place rope, window ∪ compressed top-k, attn_sink
    /// denominator, query-position de-rotation, grouped wo).
    pub fn forward(
        &self,
        model: &Dsv4Model,
        x: &[f32],
        s: usize,
        variant: ActQuantVariant,
        capture: Option<&mut AttnCapture>,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let heads = model.mc.n_head as usize;
        let hd = d.head_dim as usize;
        let rd = d.qk_rope_head_dim as usize;
        let q_lora = d.q_lora_rank as usize;
        let win = d.sliding_window as usize;
        let o_groups = d.o_groups as usize;
        let o_lora = d.o_lora_rank as usize;
        let eps = model.mc.rms_eps;
        let positions: Vec<usize> = (0..s).collect();

        // q path (model.py:496-499)
        let qr = rmsnorm(&matmul(x, s, hidden, &self.wq_a, q_lora), &self.q_norm, eps);
        let mut q = matmul(&qr, s, q_lora, &self.wq_b, heads * hd); // [s, heads, hd]
        for head in q.chunks_exact_mut(hd) {
            // weightless per-head RMS over the FULL head dim (model.py:498)
            let mut acc = 0f64;
            for v in head.iter() {
                acc += (*v as f64) * (*v as f64);
            }
            let rsq = 1.0f32 / ((acc / hd as f64) as f32 + eps).sqrt();
            for v in head.iter_mut() {
                *v *= rsq;
            }
        }
        apply_rope(&mut q, s, heads, hd, rd, &self.fc, &positions, false);

        // shared kv latent = K = V (model.py:502-506) + window QAT on nope dims
        let mut kv = rmsnorm(&matmul(x, s, hidden, &self.wkv, hd), &self.kv_norm, eps);
        apply_rope(&mut kv, s, 1, hd, rd, &self.fc, &positions, false);
        for row in kv.chunks_exact_mut(hd) {
            act_quant(&mut row[..hd - rd], 64, variant);
        }

        let (mut idxs, mut slots) = window_topk_idxs(win, s);
        let mut n_kv = s;
        let mut cap_local = AttnCapture::default();
        if self.ratio != 0 {
            let offset = s;
            let (cidx, cslots) = if let Some(ix) = &self.indexer {
                let out = ix.forward(
                    x,
                    &qr,
                    s,
                    hidden,
                    q_lora,
                    offset,
                    &self.fc,
                    rd,
                    eps,
                    variant,
                    capture.is_some(),
                );
                cap_local.indexer_kv = out.indexer_kv;
                cap_local.index_score = out.index_score;
                (out.idxs, out.slots)
            } else {
                compress_topk_idxs(self.ratio, s, offset)
            };
            // concat per row
            if cslots > 0 {
                let mut merged = vec![-1i64; s * (slots + cslots)];
                for t in 0..s {
                    merged[t * (slots + cslots)..t * (slots + cslots) + slots]
                        .copy_from_slice(&idxs[t * slots..(t + 1) * slots]);
                    merged[t * (slots + cslots) + slots..(t + 1) * (slots + cslots)]
                        .copy_from_slice(&cidx[t * cslots..(t + 1) * cslots]);
                }
                idxs = merged;
                slots += cslots;
            }
            let ckv = self
                .compressor
                .as_ref()
                .expect("ratio != 0 implies compressor")
                .forward(x, s, hidden, &self.fc, rd, eps, variant);
            cap_local.compressor_kv = ckv.clone();
            if let Some((ckv, nb)) = ckv {
                kv.extend_from_slice(&ckv);
                n_kv += nb;
            }
        }
        if let Some(c) = capture {
            *c = cap_local;
        }

        // sparse_attn (kernel.py:308-348): online softmax over the gathered rows,
        // attn_sink contributes denominator mass only. f32 scores, f64 sums.
        let scale = (hd as f64).powf(-0.5) as f32;
        let mut o = vec![0f32; s * heads * hd];
        par_rows(&mut o, heads * hd, |t, orow| {
            let ti = &idxs[t * slots..(t + 1) * slots];
            for h in 0..heads {
                let qv = &q[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                let mut scores = vec![f32::NEG_INFINITY; slots];
                for (sl, &ix) in ti.iter().enumerate() {
                    if ix >= 0 {
                        debug_assert!((ix as usize) < n_kv);
                        scores[sl] = dot(qv, &kv[ix as usize * hd..(ix as usize + 1) * hd]) * scale;
                    }
                }
                let mut m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                m = m.max(-1e30);
                let mut denom = 0f64;
                let mut acc = vec![0f64; hd];
                for (sl, &ix) in ti.iter().enumerate() {
                    if ix < 0 {
                        continue;
                    }
                    let e = (scores[sl] - m).exp();
                    denom += e as f64;
                    let krow = &kv[ix as usize * hd..(ix as usize + 1) * hd];
                    for i in 0..hd {
                        acc[i] += e as f64 * krow[i] as f64;
                    }
                }
                denom += (self.sink[h] - m).exp() as f64;
                let dst = &mut orow[h * hd..(h + 1) * hd];
                for i in 0..hd {
                    dst[i] = (acc[i] / denom) as f32;
                }
            }
        });
        // de-rotation at the query position (model.py:534)
        apply_rope(&mut o, s, heads, hd, rd, &self.fc, &positions, true);

        // grouped low-rank output projection (model.py:537-542): heads 8g..8g+7 form
        // group g (o reshaped [s, o_groups, heads/o_groups * hd]); per-group einsum
        // against wo_a viewed [o_groups, o_lora, group_width]; flatten -> wo_b.
        let gw = heads / o_groups * hd; // group width (4096)
        let mut og = vec![0f32; s * o_groups * o_lora];
        par_rows(&mut og, o_lora, |tg, row| {
            let (t, g) = (tg / o_groups, tg % o_groups);
            let src = &o[t * heads * hd + g * gw..t * heads * hd + (g + 1) * gw];
            let wg = &self.wo_a[g * o_lora * gw..(g + 1) * o_lora * gw];
            for (r, out_v) in row.iter_mut().enumerate() {
                *out_v = dot(src, &wg[r * gw..(r + 1) * gw]);
            }
        });
        matmul(&og, s, o_groups * o_lora, &self.wo_b, hidden)
    }
}

// ============================ MoE (model.py:546-643) ============================

pub struct MoeW {
    pub prefix: String,
    pub hash: bool,
    pub gate_w: Vec<f32>,          // [n_experts, hidden]
    pub bias: Option<Vec<f32>>,    // score layers only (selection-only correction)
    pub tid2eid: Option<Vec<i64>>, // hash layers only [vocab, top_k]
    pub shared: [Vec<f32>; 3],     // w1, w2, w3
}

impl MoeW {
    pub fn load(model: &Dsv4Model, prefix: &str, layer_id: u32) -> Self {
        let d = model.cfg();
        let hash = d.is_hash_layer(layer_id);
        let (_, gate_w) = model.tensor_f32(&format!("{prefix}.gate.weight"));
        let bias = if hash {
            None
        } else {
            Some(model.tensor_f32(&format!("{prefix}.gate.bias")).1)
        };
        let tid2eid = if hash {
            Some(model.tensor_i64(&format!("{prefix}.gate.tid2eid")).1)
        } else {
            None
        };
        let shared =
            ["w1", "w2", "w3"].map(|m| model.tensor_f32(&format!("{prefix}.shared_experts.{m}")).1);
        MoeW {
            prefix: prefix.to_string(),
            hash,
            gate_w,
            bias,
            tid2eid,
            shared,
        }
    }

    /// SwiGLU expert with swiglu_limit clamps (model.py:596-606): up two-sided, gate
    /// one-sided (max only), routing weight multiplied BEFORE w2. All f32 with
    /// f64-accumulated GEMM dots.
    ///
    /// Lane 7: `quantize` (routed experts under the native arm) emulates the reference
    /// expert-GEMM activation quantization (model.py:113-115 → kernel.py act_quant
    /// non-inplace, K:92-96): per-128 pow2-ceil scale + REAL FP8 e4m3 RNE round-trip
    /// on the w1/w3 input AND on h AFTER the routing-weight multiply (M:604-606 order),
    /// before w2. Weights stay the lane-1 exact f32 decode; products of e4m3-grid
    /// activations × decoded weights are exact, so the f64-accumulated dot is the
    /// ideal-order reference of the SAME quantized arithmetic. Banked deviations
    /// (mirroring the GPU arm): quantize from f32 (not from a bf16-rounded copy) and
    /// the shared expert stays unquantized.
    #[allow(clippy::too_many_arguments)]
    fn expert_forward(
        x: &[f32],
        t: usize,
        hidden: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        inter: usize,
        limit: f32,
        weights: Option<&[f32]>,
        parallel: bool,
        quantize: bool,
    ) -> Vec<f32> {
        // routed experts run inside par_map (one task per expert) and MUST stay
        // single-threaded; the shared expert runs at top level and parallelizes.
        // Arithmetic is identical either way (one sequential f64 dot per element).
        let mm = if parallel { matmul } else { matmul_serial };
        let xq: Vec<f32>;
        let xin: &[f32] = if quantize {
            let mut v = x.to_vec();
            act_quant(&mut v, 128, ActQuantVariant::RefFp8Round);
            xq = v;
            &xq
        } else {
            x
        };
        let gate = mm(xin, t, hidden, w1, inter);
        let up = mm(xin, t, hidden, w3, inter);
        let mut h = vec![0f32; t * inter];
        for i in 0..t * inter {
            let u = up[i].clamp(-limit, limit);
            let g = gate[i].min(limit);
            h[i] = g * sigmoid_f32(g) * u;
        }
        if let Some(w) = weights {
            for (ti, hrow) in h.chunks_exact_mut(inter).enumerate() {
                for v in hrow.iter_mut() {
                    *v *= w[ti];
                }
            }
        }
        if quantize {
            act_quant(&mut h, 128, ActQuantVariant::RefFp8Round);
        }
        mm(&h, t, inter, w2, hidden)
    }

    /// x [s, hidden], ids [s] -> [s, hidden]. Scores sqrt(softplus(x·g)) in f32
    /// (f64-accumulated gate GEMM); bias is selection-only; weights renormalized then
    /// × routed_scaling_factor; hash layers select via tid2eid; shared expert added
    /// last, unweighted (model.py:643).
    pub fn forward(&self, model: &Dsv4Model, x: &[f32], s: usize, ids: &[u32]) -> Vec<f32> {
        let moe = model.mc.moe.as_ref().expect("moe block");
        let d = model.cfg();
        let hidden = model.mc.n_embd as usize;
        let ne = moe.expert_count as usize;
        let topk = moe.expert_used_count as usize;
        let inter = moe.expert_ff_length as usize;
        let limit = d.swiglu_limit;
        let route_scale = d.routed_scaling_factor;

        let mut scores = matmul(x, s, hidden, &self.gate_w, ne);
        for v in &mut scores {
            *v = softplus_f32(*v).sqrt(); // sqrtsoftplus (model.py:570-571)
        }
        // selection
        let mut indices = vec![0usize; s * topk];
        if let Some(tid2eid) = &self.tid2eid {
            for t in 0..s {
                let row = &tid2eid[ids[t] as usize * topk..(ids[t] as usize + 1) * topk];
                let mut seen = std::collections::BTreeSet::new();
                for (k, &e) in row.iter().enumerate() {
                    assert!(
                        (0..ne as i64).contains(&e),
                        "{}: tid2eid[{}][{}] = {} outside [0,{})",
                        self.prefix,
                        ids[t],
                        k,
                        e,
                        ne
                    );
                    // torch `y[idx] += v` is last-wins on duplicate rows — a duplicate
                    // expert id would make the reference semantics ambiguous. Measured
                    // absent on all 3 hash layers (0 duplicate rows / 129280); refuse
                    // if a different artifact ever ships one.
                    assert!(
                        seen.insert(e),
                        "{}: duplicate expert id {} in tid2eid row {} — ambiguous vs torch last-wins",
                        self.prefix,
                        e,
                        ids[t]
                    );
                    indices[t * topk + k] = e as usize;
                }
            }
        } else {
            let bias = self.bias.as_ref().expect("score layer needs gate.bias");
            for t in 0..s {
                let mut order: Vec<usize> = (0..ne).collect();
                let biased: Vec<f32> = (0..ne).map(|e| scores[t * ne + e] + bias[e]).collect();
                order.sort_by(|&a, &b| {
                    biased[b]
                        .partial_cmp(&biased[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                for k in 0..topk {
                    indices[t * topk + k] = order[k];
                }
            }
        }
        // combination weights: ORIGINAL scores at the selected experts, renormalized,
        // × route_scale (bias never enters weights — model.py:573-582)
        let mut weights = vec![0f32; s * topk];
        for t in 0..s {
            let mut sum = 0f32;
            for k in 0..topk {
                let w = scores[t * ne + indices[t * topk + k]];
                weights[t * topk + k] = w;
                sum += w;
            }
            for k in 0..topk {
                weights[t * topk + k] = weights[t * topk + k] / sum * route_scale;
            }
        }
        // per-expert token groups, computed in parallel, accumulated in ascending
        // expert order (the reference loops sorted(set(indices))) then shared last.
        let mut uniq: Vec<usize> = indices.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let contribs = par_map(uniq.len(), |ui| {
            let e = uniq[ui];
            let toks: Vec<(usize, usize)> = (0..s * topk)
                .filter(|i| indices[*i] == e)
                .map(|i| (i / topk, i % topk))
                .collect();
            let mut xg = Vec::with_capacity(toks.len() * hidden);
            let mut wg = Vec::with_capacity(toks.len());
            for &(t, k) in &toks {
                xg.extend_from_slice(&x[t * hidden..(t + 1) * hidden]);
                wg.push(weights[t * topk + k]);
            }
            let (_, w1) = model.tensor_f32(&format!("{}.experts.{e}.w1", self.prefix));
            let (_, w2) = model.tensor_f32(&format!("{}.experts.{e}.w2", self.prefix));
            let (_, w3) = model.tensor_f32(&format!("{}.experts.{e}.w3", self.prefix));
            let y = Self::expert_forward(
                &xg,
                toks.len(),
                hidden,
                &w1,
                &w2,
                &w3,
                inter,
                limit,
                Some(&wg),
                false,
                expert_arm_native(),
            );
            (toks, y)
        });
        let mut y = vec![0f32; s * hidden];
        for (toks, contrib) in &contribs {
            for (row, &(t, _)) in toks.iter().enumerate() {
                let dst = &mut y[t * hidden..(t + 1) * hidden];
                for i in 0..hidden {
                    dst[i] += contrib[row * hidden + i];
                }
            }
        }
        // shared expert(s), unweighted and unscaled (model.py:643); stays unquantized
        // under the native arm (banked lane-7 deviation — mirrors the GPU arm's
        // FP8-linear stay-bf16 decision)
        let sh = Self::expert_forward(
            x,
            s,
            hidden,
            &self.shared[0],
            &self.shared[1],
            &self.shared[2],
            self.shared[0].len() / hidden,
            limit,
            None,
            true,
            false,
        );
        for i in 0..s * hidden {
            y[i] += sh[i];
        }
        y
    }
}

// ============================ block + drivers (model.py:646-826) ============================

pub struct BlockW {
    pub attn: AttnW,
    pub moe: MoeW,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn: HcSet,
    pub hc_ffn: HcSet,
}

/// Fixture arrays captured from one block forward.
#[derive(Default)]
pub struct BlockCapture {
    pub attn_out: Option<Vec<f32>>,
    pub attn: AttnCapture,
}

impl BlockW {
    pub fn load(model: &Dsv4Model, prefix: &str, layer_id: u32, max_seq: usize) -> Self {
        let hc_load = |fam: &str| -> HcSet {
            let (shape, fn_w) = model.tensor_f32(&format!("{prefix}.hc_{fam}_fn"));
            let (_, base) = model.tensor_f32(&format!("{prefix}.hc_{fam}_base"));
            let (_, scale) = model.tensor_f32(&format!("{prefix}.hc_{fam}_scale"));
            HcSet {
                rows: shape[0],
                fn_w,
                base,
                scale,
            }
        };
        BlockW {
            attn: AttnW::load(model, &format!("{prefix}.attn"), layer_id, max_seq),
            moe: MoeW::load(model, &format!("{prefix}.ffn"), layer_id),
            attn_norm: model.tensor_f32(&format!("{prefix}.attn_norm.weight")).1,
            ffn_norm: model.tensor_f32(&format!("{prefix}.ffn_norm.weight")).1,
            hc_attn: hc_load("attn"),
            hc_ffn: hc_load("ffn"),
        }
    }

    /// One block over the hc state: x [s, hc_mult, hidden] -> same shape.
    pub fn forward(
        &self,
        model: &Dsv4Model,
        x: &[f32],
        s: usize,
        ids: &[u32],
        variant: ActQuantVariant,
        capture: Option<&mut BlockCapture>,
    ) -> Vec<f32> {
        let d = model.cfg();
        let hc = d.hc_mult as usize;
        let hidden = model.mc.n_embd as usize;
        let eps = model.mc.rms_eps;
        let iters = d.hc_sinkhorn_iters;
        let hc_eps = d.hc_eps;

        // attention sub-block
        let (h, post, comb) = hc_pre(x, s, hc, hidden, &self.hc_attn, iters, hc_eps);
        let h = rmsnorm(&h, &self.attn_norm, eps);
        let mut attn_cap = AttnCapture::default();
        let want_cap = capture.is_some();
        let h = self
            .attn
            .forward(model, &h, s, variant, want_cap.then_some(&mut attn_cap));
        if let Some(c) = capture {
            c.attn_out = Some(h.clone());
            c.attn = attn_cap;
        }
        let x = hc_post(&h, x, s, hc, hidden, &post, &comb);

        // ffn sub-block
        let (h, post, comb) = hc_pre(&x, s, hc, hidden, &self.hc_ffn, iters, hc_eps);
        let h = rmsnorm(&h, &self.ffn_norm, eps);
        let h = self.moe.forward(model, &h, s, ids);
        hc_post(&h, &x, s, hc, hidden, &post, &comb)
    }
}

/// Expand embeddings into the hc state (model.py:805): [s, hidden] -> [s, hc, hidden].
pub fn hc_expand(e: &[f32], s: usize, hc: usize, hidden: usize) -> Vec<f32> {
    let mut h = Vec::with_capacity(s * hc * hidden);
    for t in 0..s {
        for _ in 0..hc {
            h.extend_from_slice(&e[t * hidden..(t + 1) * hidden]);
        }
    }
    h
}

/// Trunk head: hc_head collapse -> final RMSNorm -> last-position logits
/// (model.py:713-726, 808).
pub fn trunk_logits_last(model: &Dsv4Model, h: &[f32], s: usize) -> Vec<f32> {
    let d = model.cfg();
    let hc = d.hc_mult as usize;
    let hidden = model.mc.n_embd as usize;
    let set = HcSet {
        rows: model.tensor_f32("hc_head_fn").0[0],
        fn_w: model.tensor_f32("hc_head_fn").1,
        base: model.tensor_f32("hc_head_base").1,
        scale: model.tensor_f32("hc_head_scale").1,
    };
    let collapsed = hc_head(h, s, hc, hidden, &set, model.mc.rms_eps, d.hc_eps);
    let final_h = rmsnorm(
        &collapsed,
        &model.tensor_f32("norm.weight").1,
        model.mc.rms_eps,
    );
    model.head_logits(&final_h[(s - 1) * hidden..s * hidden])
}

/// MTP path (model.py:738-766, call shape = model.py:826 — same ids as the trunk; the
/// spec-decode token-shift convention is NOT claimed here, matching the fixture caveat).
pub fn mtp_logits_last(
    model: &Dsv4Model,
    h_trunk: &[f32],
    s: usize,
    ids: &[u32],
    variant: ActQuantVariant,
    max_seq: usize,
) -> Vec<f32> {
    let d = model.cfg();
    let hc = d.hc_mult as usize;
    let hidden = model.mc.n_embd as usize;
    let eps = model.mc.rms_eps;
    let n_trunk = model.mc.n_layer - model.mc.nextn_predict_layers;
    // NOTE single-MTP path (num_nextn_predict_layers == 1 on Flash); a deeper chain
    // would thread h through mtp.k sequentially.
    assert_eq!(
        model.mc.nextn_predict_layers, 1,
        "single MTP layer expected"
    );
    let p = "mtp.0";
    let e = rmsnorm(
        &model.embed_rows(ids),
        &model.tensor_f32(&format!("{p}.enorm.weight")).1,
        eps,
    );
    let xm_h = rmsnorm(
        h_trunk,
        &model.tensor_f32(&format!("{p}.hnorm.weight")).1,
        eps,
    );
    let (_, e_proj) = model.tensor_f32(&format!("{p}.e_proj"));
    let (_, h_proj) = model.tensor_f32(&format!("{p}.h_proj"));
    let ep = matmul(&e, s, hidden, &e_proj, hidden); // [s, hidden]
    let hp = matmul(&xm_h, s * hc, hidden, &h_proj, hidden); // per-copy [s*hc, hidden]
    let mut xm = vec![0f32; s * hc * hidden];
    for t in 0..s {
        for c in 0..hc {
            let dst = &mut xm[(t * hc + c) * hidden..(t * hc + c + 1) * hidden];
            let er = &ep[t * hidden..(t + 1) * hidden];
            let hr = &hp[(t * hc + c) * hidden..(t * hc + c + 1) * hidden];
            for i in 0..hidden {
                dst[i] = er[i] + hr[i];
            }
        }
    }
    let blk = BlockW::load(model, p, n_trunk, max_seq);
    let xm = blk.forward(model, &xm, s, ids, variant, None);
    let set = HcSet {
        rows: model.tensor_f32(&format!("{p}.hc_head_fn")).0[0],
        fn_w: model.tensor_f32(&format!("{p}.hc_head_fn")).1,
        base: model.tensor_f32(&format!("{p}.hc_head_base")).1,
        scale: model.tensor_f32(&format!("{p}.hc_head_scale")).1,
    };
    let collapsed = hc_head(&xm, s, hc, hidden, &set, eps, d.hc_eps);
    let final_h = rmsnorm(
        &collapsed,
        &model.tensor_f32(&format!("{p}.norm.weight")).1,
        eps,
    );
    model.head_logits(&final_h[(s - 1) * hidden..s * hidden])
}

// ============================ fixture spec + npz reading ============================

/// One banked fixture file pair (json + npz): the gate's ground truth.
pub struct FixtureSpec {
    pub variant: ActQuantVariant,
    pub variant_tag: String,
    pub npz_path: std::path::PathBuf,
    pub tokens_32: Vec<u32>,
    pub tokens_160: Option<Vec<u32>>,
    /// name -> (shape, sha256 of little-endian f32 payload)
    pub arrays: BTreeMap<String, (Vec<usize>, String)>,
    pub top20_ids: Vec<u32>,
    /// Absent on trunk-only fixture sets (0731 has no NextN head; its `mtp.*` namespace
    /// is the DSpark drafter — a later lane's oracle, not this fixture contract).
    pub mtp_top20_ids: Option<Vec<u32>>,
    /// Measured final-logits max-abs between the ref and clamp-only variants of the SAME
    /// generator run (the contract fork the logits bound must catch). Banked by the 0731
    /// generator; absent on the preview lane-2 sets.
    pub contract_fork_final_logits_maxabs: Option<f64>,
}

impl FixtureSpec {
    pub fn load(json_path: &Path) -> Self {
        let txt = std::fs::read_to_string(json_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", json_path.display()));
        let j = JsonObj::parse(&txt);
        let tag = j.string("variant").expect("fixture json: variant");
        let npz = j.string("npz").expect("fixture json: npz");
        let tokens_32 = j.u32_array("tokens_32").expect("fixture json: tokens_32");
        let tokens_160 = j.u32_array("tokens_160");
        let arrays_obj = j.object("arrays").expect("fixture json: arrays");
        let mut arrays = BTreeMap::new();
        for (name, _) in arrays_obj.fields() {
            let a = arrays_obj.object(name).expect("array entry object");
            let shape: Vec<usize> = a
                .u64_array("shape")
                .expect("array shape")
                .into_iter()
                .map(|x| x as usize)
                .collect();
            let sha = a.string("sha256_le_f32").expect("array sha256");
            arrays.insert(name.to_string(), (shape, sha));
        }
        let top20_ids = j
            .object("top20")
            .and_then(|o| o.u32_array("ids"))
            .expect("fixture json: top20.ids");
        // Optional: absent on 0731-lineage trunk-only fixture sets. When the JSON has the
        // object, ids must be present and well-formed (a malformed entry still refuses).
        let mtp_top20_ids = j
            .object("mtp_top20")
            .map(|o| o.u32_array("ids").expect("fixture json: mtp_top20.ids"));
        let contract_fork_final_logits_maxabs = j.f64("contract_fork_final_logits_maxabs");
        FixtureSpec {
            variant: ActQuantVariant::from_fixture_tag(&tag),
            variant_tag: tag,
            npz_path: json_path.parent().unwrap_or(Path::new(".")).join(npz),
            tokens_32,
            tokens_160,
            arrays,
            top20_ids,
            mtp_top20_ids,
            contract_fork_final_logits_maxabs,
        }
    }
}

/// Minimal NPZ (zip of .npy) reader for np.savez output: parses the central directory
/// (zip64-aware), requires stored (uncompressed) entries, then the v1/v2 .npy header
/// ('<f4', C-order). Returns name -> (shape, data, sha256-hex of the raw payload) so
/// callers can pin file integrity against the fixture JSON.
pub fn read_npz(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>, String)> {
    use sha2::{Digest, Sha256};
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let rd_u16 = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]) as usize;
    let rd_u32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()) as u64;
    let rd_u64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    // find EOCD (scan back over the (empty) comment window)
    let mut eocd = None;
    let mut i = buf.len().saturating_sub(22);
    loop {
        if rd_u32(i) == 0x06054b50 {
            eocd = Some(i);
            break;
        }
        if i == 0 || buf.len() - i > 22 + 65535 {
            break;
        }
        i -= 1;
    }
    let eocd = eocd.expect("npz: no end-of-central-directory record");
    let mut n_entries = rd_u16(eocd + 10) as u64;
    let mut cd_off = rd_u32(eocd + 16);
    // zip64 EOCD locator directly precedes EOCD when present
    if eocd >= 20 && rd_u32(eocd - 20) == 0x07064b50 {
        let z64 = rd_u64(eocd - 20 + 8) as usize;
        assert_eq!(rd_u32(z64), 0x06064b50, "npz: bad zip64 EOCD");
        n_entries = rd_u64(z64 + 32);
        cd_off = rd_u64(z64 + 48);
    }
    let mut out = BTreeMap::new();
    let mut p = cd_off as usize;
    for _ in 0..n_entries {
        assert_eq!(rd_u32(p), 0x02014b50, "npz: bad central directory entry");
        let method = rd_u16(p + 10);
        let mut csize = rd_u32(p + 20);
        let mut usize_ = rd_u32(p + 24);
        let name_len = rd_u16(p + 28);
        let extra_len = rd_u16(p + 30);
        let comment_len = rd_u16(p + 32);
        let mut lho = rd_u32(p + 42);
        let name = String::from_utf8_lossy(&buf[p + 46..p + 46 + name_len]).into_owned();
        // zip64 extra field (id 0x0001): order = usize, csize, lho for any 0xFFFFFFFF
        let mut e = p + 46 + name_len;
        let e_end = e + extra_len;
        while e + 4 <= e_end {
            let (id, sz) = (rd_u16(e), rd_u16(e + 2));
            if id == 0x0001 {
                let mut q = e + 4;
                if usize_ == 0xFFFFFFFF {
                    usize_ = rd_u64(q);
                    q += 8;
                }
                if csize == 0xFFFFFFFF {
                    csize = rd_u64(q);
                    q += 8;
                }
                if lho == 0xFFFFFFFF {
                    lho = rd_u64(q);
                }
            }
            e += 4 + sz;
        }
        assert_eq!(
            method, 0,
            "npz: {name} is compressed; np.savez writes stored"
        );
        assert_eq!(csize, usize_, "npz: {name} stored sizes disagree");
        // local header: skip its own (possibly different) name/extra lengths
        let l = lho as usize;
        assert_eq!(rd_u32(l), 0x04034b50, "npz: bad local header for {name}");
        let lname = rd_u16(l + 26);
        let lextra = rd_u16(l + 28);
        let data = &buf[l + 30 + lname + lextra..l + 30 + lname + lextra + usize_ as usize];
        // .npy header
        assert_eq!(&data[..6], b"\x93NUMPY", "npz: {name} not npy");
        let (major, hlen, hstart) = if data[6] == 1 {
            (
                1u8,
                u16::from_le_bytes([data[8], data[9]]) as usize,
                10usize,
            )
        } else {
            (
                data[6],
                u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
                12usize,
            )
        };
        assert!(major >= 1, "npy version");
        let hdr = String::from_utf8_lossy(&data[hstart..hstart + hlen]).into_owned();
        assert!(
            hdr.contains("'descr': '<f4'"),
            "npz: {name} dtype is not <f4: {hdr}"
        );
        assert!(
            hdr.contains("'fortran_order': False"),
            "npz: {name} fortran order: {hdr}"
        );
        let shape_str = hdr
            .split("'shape':")
            .nth(1)
            .and_then(|s| s.split('(').nth(1))
            .and_then(|s| s.split(')').next())
            .unwrap_or_else(|| panic!("npz: {name} shape parse: {hdr}"));
        let shape: Vec<usize> = shape_str
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect();
        let payload = &data[hstart + hlen..];
        let n: usize = shape.iter().product::<usize>().max(1);
        assert_eq!(
            payload.len(),
            n * 4,
            "npz: {name} payload size vs shape {shape:?}"
        );
        let vals: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut h = Sha256::new();
        h.update(payload);
        let sha = h
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let base = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(base, (shape, vals, sha));
        p += 46 + name_len + extra_len + comment_len;
    }
    out
}

// ============================ tests (pure math; no artifact needed) ============================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_sylvester_order_and_involution() {
        // H_4 (Sylvester) rows: ++++, +-+-, ++--, +--+ ; scale 1/2.
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0];
        hadamard(&mut x, 4);
        assert_eq!(x, vec![0.5, 0.5, 0.5, 0.5]);
        let mut y = vec![1.0f32, 2.0, 3.0, 4.0];
        hadamard(&mut y, 4);
        assert_eq!(y, vec![5.0, -1.0, -2.0, 0.0]); // (1±2±3±4)/2 in Sylvester order
        hadamard(&mut y, 4); // orthonormal -> involution
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn e2m1_rne_ties_to_even() {
        // midpoints: 0.25 -> 0 (even), 0.75 -> 1.0 (even), 1.25 -> 1.0, 1.75 -> 2.0,
        // 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0
        let cases = [
            (0.25f32, 0.0f32),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (5.00001, 6.0),
            (-0.75, -1.0),
            (7.0, 6.0),
            (0.0, 0.0),
        ];
        for (v, want) in cases {
            assert_eq!(e2m1_rne(v), want, "e2m1_rne({v})");
        }
    }

    #[test]
    fn pow2_ceil_exact_and_between() {
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(2.0), 2.0);
        assert_eq!(pow2_ceil(1.0001), 2.0);
        assert_eq!(pow2_ceil(0.25), 0.25);
        assert_eq!(pow2_ceil(0.2500001), 0.5);
        assert_eq!(pow2_ceil(3.9), 4.0);
    }

    #[test]
    fn act_quant_grid_values_roundtrip_exact() {
        // integer values in [-6, 6]: amax 6 -> s = pow2_ceil(6/448) = 2^-6, so q = 64·v
        // lands on e4m3-representable magnitudes {0, 64, 128, 192, 256, 320, 384} — the
        // ref FP8 round-trip must be the identity on them.
        let mut x: Vec<f32> = (0..64).map(|i| (i % 13) as f32 - 6.0).collect();
        let want = x.clone();
        act_quant(&mut x, 64, ActQuantVariant::RefFp8Round);
        for (a, b) in x.iter().zip(&want) {
            assert_eq!(a, b, "grid value must round-trip");
        }
        // clamp-only variant is identity here too
        let mut y = want.clone();
        act_quant(&mut y, 64, ActQuantVariant::ClampOnly);
        assert_eq!(y, want);
        // and a value OFF the grid must round (RNE): 6.03125·64 = 386 -> nearest 384
        let mut z = vec![0f32; 64];
        z[0] = 6.03125; // becomes amax
        act_quant(&mut z, 64, ActQuantVariant::RefFp8Round);
        assert_eq!(z[0], 384.0 * 2f32.powi(-6), "off-grid value must RNE-round");
        // ties-to-even: 248 sits exactly between 240 and 256 -> even mantissa (256)
        let mut w = vec![0f32; 64];
        w[0] = 4.0; // amax 4 -> s = 2^-6... make amax explicit
        w[1] = 248.0 * 2f32.powi(-6); // 3.875
        // amax = 4 -> s = pow2_ceil(4/448) = 2^-6; q1 = 248 -> ties to 256
        act_quant(&mut w, 64, ActQuantVariant::RefFp8Round);
        assert_eq!(w[1], 256.0 * 2f32.powi(-6), "tie must go to even mantissa");
    }

    #[test]
    fn window_and_compress_idxs_match_reference_shapes() {
        let (m, w) = window_topk_idxs(4, 6);
        assert_eq!(w, 4);
        // row 0: 0,-1,-1,-1 ; row 3: 0,1,2,3 ; row 5: 2,3,4,5
        assert_eq!(&m[0..4], &[0, -1, -1, -1]);
        assert_eq!(&m[12..16], &[0, 1, 2, 3]);
        assert_eq!(&m[20..24], &[2, 3, 4, 5]);
        let (c, nb) = compress_topk_idxs(2, 6, 100);
        assert_eq!(nb, 3);
        // row i sees blocks j < (i+1)/2: row0 none, row1 {0}, row4 {0,1}, row5 {0,1,2}
        assert_eq!(&c[0..3], &[-1, -1, -1]);
        assert_eq!(&c[3..6], &[100, -1, -1]);
        assert_eq!(&c[12..15], &[100, 101, -1]);
        assert_eq!(&c[15..18], &[100, 101, 102]);
    }

    #[test]
    fn sinkhorn_comb_is_doubly_stochastic() {
        let hc = 4;
        let rows = (2 + hc) * hc;
        let mixes: Vec<f32> = (0..rows).map(|i| (i as f32 * 0.37).sin()).collect();
        let base: Vec<f32> = (0..rows).map(|i| (i as f32 * 0.11).cos()).collect();
        let scale = vec![0.7f32, 1.3, 0.9];
        let (pre, post, comb) = hc_split_sinkhorn(&mixes, 1, hc, &scale, &base, 20, 1e-6);
        assert!(pre.iter().all(|&v| v > 0.0));
        assert!(post.iter().all(|&v| (0.0..2.0).contains(&v)));
        for j in 0..hc {
            let rs: f32 = (0..hc).map(|k| comb[j * hc + k]).sum();
            assert!((rs - 1.0).abs() < 1e-3, "row {j} sum {rs}");
        }
        for k in 0..hc {
            let cs: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            assert!((cs - 1.0).abs() < 1e-3, "col {k} sum {cs}");
        }
    }

    #[test]
    fn freqs_cis_no_yarn_matches_closed_form() {
        let fc = precompute_freqs_cis(64, 8, 0, 10000.0, 16.0, 32.0, 1.0);
        assert_eq!(fc.half, 32);
        // pos 0: all (1, 0)
        for k in 0..32 {
            assert_eq!(fc.cs[k], (1.0, 0.0));
        }
        // pos 3, k=0: angle 3
        let (c, s) = fc.cs[3 * 32];
        assert!((c - 3f32.cos()).abs() < 1e-6 && (s - 3f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_roundtrip_inverse() {
        let fc = precompute_freqs_cis(64, 16, 0, 10000.0, 16.0, 32.0, 1.0);
        let mut x: Vec<f32> = (0..2 * 512)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.1)
            .collect();
        let orig = x.clone();
        let positions = vec![3usize, 7];
        apply_rope(&mut x, 2, 1, 512, 64, &fc, &positions, false);
        assert_ne!(x, orig);
        apply_rope(&mut x, 2, 1, 512, 64, &fc, &positions, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5);
        }
        // nope dims untouched
        assert_eq!(&x[..448], &orig[..448]);
    }

    #[test]
    fn matmul_f64_accum_small() {
        // x [2,3] @ w [2,3]^T
        let x = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 2.0];
        let w = vec![1.0f32, 0.0, -1.0, 2.0, 1.0, 0.5];
        let out = matmul(&x, 2, 3, &w, 2);
        assert_eq!(out, vec![-2.0, 5.5, -1.5, 1.0]);
    }

    #[test]
    fn npz_reader_parses_synthetic_stored_zip() {
        // hand-built stored zip with one npy entry: shape (2,), values [1.5, -2.0]
        let dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }\n";
        let mut payload = Vec::new();
        payload.extend_from_slice(b"\x93NUMPY\x01\x00");
        payload.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        payload.extend_from_slice(dict);
        payload.extend_from_slice(&1.5f32.to_le_bytes());
        payload.extend_from_slice(&(-2.0f32).to_le_bytes());
        let name = b"arr.npy";
        let mut zip = Vec::new();
        // local header
        zip.extend_from_slice(&0x04034b50u32.to_le_bytes());
        zip.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // ver, flags, method, time, date
        zip.extend_from_slice(&0u32.to_le_bytes()); // crc (unchecked)
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        zip.extend_from_slice(name);
        zip.extend_from_slice(&payload);
        let cd_off = zip.len();
        // central directory
        zip.extend_from_slice(&0x02014b50u32.to_le_bytes());
        zip.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        zip.extend_from_slice(&0u32.to_le_bytes()); // crc
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        zip.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip.extend_from_slice(&[0u8; 12]); // extra, comment, disk, int attr, ext attr
        zip.extend_from_slice(&0u32.to_le_bytes()); // local header offset
        zip.extend_from_slice(name);
        let cd_len = zip.len() - cd_off;
        // EOCD
        zip.extend_from_slice(&0x06054b50u32.to_le_bytes());
        zip.extend_from_slice(&[0u8; 4]);
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&1u16.to_le_bytes());
        zip.extend_from_slice(&(cd_len as u32).to_le_bytes());
        zip.extend_from_slice(&(cd_off as u32).to_le_bytes());
        zip.extend_from_slice(&0u16.to_le_bytes());
        let dir = std::env::temp_dir().join(format!("memra_dsv4_npz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.npz");
        std::fs::write(&p, &zip).unwrap();
        let m = read_npz(&p);
        std::fs::remove_dir_all(&dir).ok();
        let (shape, vals, _sha) = &m["arr"];
        assert_eq!(shape, &vec![2usize]);
        assert_eq!(vals, &vec![1.5f32, -2.0]);
    }
}
