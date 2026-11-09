//! DeepSeek-V4.1-Flash CPU forward primitives (the `deepseek_v41` reference program).
//!
//! Semantic source: the vendor reference `inference/model.py`, `engram.py` and `kernel.py` of
//! `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision 1bae0e1088993b9b82de08f59de5be5a34e42170.
//! This is a sibling of [`crate::dsv4_forward`], not a parameterization of it: the DSv4 module
//! runs its fixtures in f32 and treats every bf16 cast as identity, while V4.1 is gated against
//! the vendor program run unchanged, whose activations are bf16 between operations. So this
//! module rounds to bf16 at every point the vendor program stores a bf16 tensor, and applies the
//! vendor quantize-dequantize steps exactly:
//!
//! - every FP8/FP4 linear input: `act_quant(x, 32, "ue8m0")`, a power-of-two scale per 32
//!   elements (amax floored at 1e-4, `2^ceil(log2(amax / 448))`), clamp to +-448, e4m3 with
//!   ties to even;
//! - window KV and the attention output before `wo_a`: the same FP8 quantize-dequantize;
//! - the indexer query and key: FP4 e2m1 per 32 with a power-of-two scale;
//! - the compressed main KV: FP4 e2m1 per 16 with an e4m3 scale (`amax / 6` rounded to e4m3);
//! - sparse attention: the tilelang kernel's online softmax over blocks of 64 selected entries,
//!   bf16 probabilities into the PV product, an fp32 denominator with the sink added once.
//!
//! Primitive semantics: the vendor program fixes the operations, their order, the dtype every
//! tensor is stored in and every quantization point, but not how a GEMM or a reduction orders its
//! f32 accumulation or how exp, sigmoid, cos and the like approximate their result. Those free
//! choices are pinned to the canonical one: every dot product, sum and mean accumulates in f64
//! and rounds once to f32 (then to bf16 where the vendor output is bf16), and every
//! transcendental elementwise function is evaluated in f64 and rounded once. The tiny gate's
//! oracle runs the vendor model.py with the same primitive semantics
//! (`research/modelplan-onboarding-dsv41-20261109/oracle/exact.py`), so the two agree to the
//! bit. Without that, a one-ulp difference in an f32 accumulation flips a bf16 rounding, and the
//! bf16 index scores tie often enough that the flip changes a top-k selection.
//!
//! The Engram hash multipliers are drawn by the vendor with numpy's `default_rng(10007 * layer)`
//! (PCG64 seeded through `SeedSequence`, Lemire-bounded 64-bit integers). [`engram_multipliers`]
//! is a bit-exact port, pinned against the values the vendor code derives for the 1017 config.

use crate::dsv4::e8m0_to_f32;
use crate::dsv4_forward::{
    FreqsCis, act_quant as dsv4_act_quant, dot, e2m1_rne, matmul, pow2_ceil,
};
use crate::nvfp4_repack::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};

// ============================ element formats ============================

/// f32 -> bf16 -> f32, round to nearest with ties to even (torch `c10::BFloat16`).
#[inline]
pub fn bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(rounding) & 0xffff_0000)
}

/// [`bf16`] in place over a slice.
pub fn bf16_in_place(x: &mut [f32]) {
    for value in x {
        *value = bf16(*value);
    }
}

/// bf16 bit pattern (as stored in a safetensors BF16 tensor) -> f32.
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// f32 -> bf16 bit pattern, rounding as [`bf16`].
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    (bf16(x).to_bits() >> 16) as u16
}

/// e4m3 byte -> f32 (the vendor `float8_e4m3fn`).
#[inline]
pub fn e4m3(byte: u8) -> f32 {
    fp8_e4m3_to_f32(byte)
}

/// f32 -> e4m3 byte, nearest with ties to even, saturating at +-448.
#[inline]
pub fn to_e4m3(x: f32) -> u8 {
    f32_to_fp8_e4m3(x)
}

/// e8m0 byte -> f32 scale, `2^(byte - 127)`.
#[inline]
pub fn e8m0(byte: u8) -> f32 {
    e8m0_to_f32(byte)
}

/// Power-of-two scale -> its e8m0 byte. The scale must be a positive power of two.
#[inline]
pub fn to_e8m0(scale: f32) -> u8 {
    let bits = scale.to_bits();
    debug_assert!(
        scale > 0.0 && bits & 0x7f_ffff == 0,
        "e8m0 scale must be 2^k"
    );
    ((bits >> 23) & 0xff) as u8
}

/// e2m1 nibble -> f32 (`0..7` = 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign).
#[inline]
pub fn e2m1(nibble: u8) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = GRID[usize::from(nibble & 7)];
    if nibble & 8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// f32 already on the e2m1 grid -> its nibble.
#[inline]
pub fn to_e2m1(x: f32) -> u8 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = x.abs();
    let code = GRID
        .iter()
        .position(|&grid| grid == magnitude)
        .expect("value is on the e2m1 grid") as u8;
    if x.is_sign_negative() && magnitude != 0.0 {
        code | 8
    } else {
        code
    }
}

// ============================ quantize-dequantize ============================

/// Vendor `act_quant(x, 32, "ue8m0", inplace=True)` on contiguous groups of `block`.
/// Identical arithmetic to the DSv4 reference variant (`ActQuantVariant::RefFp8Round`).
pub fn act_quant_fp8(x: &mut [f32], block: usize) {
    dsv4_act_quant(x, block, crate::dsv4_forward::ActQuantVariant::RefFp8Round);
}

/// Vendor `fp4_act_quant(x, block, inplace=True)` with an e8m0 (power-of-two) scale.
pub fn fp4_quant_ue8m0(x: &mut [f32], block: usize) {
    crate::dsv4_forward::fp4_act_quant(x, block);
}

/// Vendor `fp4_act_quant(x, block, inplace=True, scale_dtype=float8_e4m3fn)`: amax floored at
/// `6 * 2^-9`, scale `e4m3(amax / 6)`, clamp to +-6, e2m1 with ties to even.
pub fn fp4_quant_e4m3(x: &mut [f32], block: usize) {
    assert_eq!(x.len() % block, 0, "fp4 e4m3 quant block divisibility");
    let floor = 6.0f32 * 2f32.powi(-9);
    for group in x.chunks_exact_mut(block) {
        let amax = group.iter().fold(0f32, |acc, value| acc.max(value.abs()));
        let scale = e4m3(to_e4m3(amax.max(floor) / 6.0));
        for value in group.iter_mut() {
            *value = e2m1_rne((*value / scale).clamp(-6.0, 6.0)) * scale;
        }
    }
}

// ============================ linears and norms ============================

/// `F.linear` in f32: `x [rows, k] @ w[n, k]^T`, f64-accumulated dots rounded to f32.
pub fn linear_f32(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    matmul(x, rows, k, w, n)
}

/// `F.linear` on bf16 operands: f32 accumulation, bf16 output.
pub fn linear_bf16(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = matmul(x, rows, k, w, n);
    bf16_in_place(&mut out);
    out
}

/// Vendor `linear()` over an FP8 or FP4 weight (`w` dequantized): the input is quantized per 32
/// (`act_quant`), the GEMM accumulates in f32 and the output is bf16.
pub fn linear_quant(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut quantized = x.to_vec();
    act_quant_fp8(&mut quantized, 32);
    linear_bf16(&quantized, rows, k, w, n)
}

/// Vendor `RMSNorm` on rows of `w.len()`: f32 statistics, `weight * (x * rsqrt(mean + eps))`,
/// bf16 output.
pub fn rmsnorm_bf16(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = crate::dsv4_forward::rmsnorm(x, w, eps);
    bf16_in_place(&mut out);
    out
}

/// `torch.rsqrt(x.square().mean(-1) + eps)` over one row.
#[inline]
pub fn rsqrt_mean_square(row: &[f32], eps: f32) -> f32 {
    let sum: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    1.0 / ((sum / row.len() as f64) as f32 + eps).sqrt()
}

// ============================ RoPE ============================

/// Vendor `precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast,
/// beta_slow)`: YaRN only when `original_seq_len > 0`, the ramp denominator floored at 1e-3.
#[allow(clippy::too_many_arguments)]
pub fn precompute_freqs_cis(
    dim: usize,
    seqlen: usize,
    original_seq_len: u32,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> FreqsCis {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|k| 1.0f32 / pow_cr(base, (2 * k) as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        let corrected_dim = |rotations: f64| -> f64 {
            dim as f64
                * (f64::from(original_seq_len) / (rotations * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * f64::from(base).ln())
        };
        let low = corrected_dim(f64::from(beta_fast)).floor().max(0.0);
        let high = corrected_dim(f64::from(beta_slow))
            .ceil()
            .min((dim - 1) as f64);
        let denominator = (high - low).max(1e-3) as f32;
        for (k, frequency) in freqs.iter_mut().enumerate() {
            let ramp = ((k as f32 - low as f32) / denominator).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *frequency = *frequency / factor * (1.0 - smooth) + *frequency * smooth;
        }
    }
    let mut cs = Vec::with_capacity(seqlen * half);
    for position in 0..seqlen {
        for frequency in &freqs {
            let angle = f64::from(position as f32 * *frequency);
            cs.push((angle.cos() as f32, angle.sin() as f32));
        }
    }
    FreqsCis { half, cs }
}

/// Vendor `apply_rotary_emb(x[..., -rd:], freqs, inverse)` on `vectors` rows of `width`, each
/// row `r` at `positions[r / per_position]`, written back to bf16.
pub fn rope_bf16(
    x: &mut [f32],
    width: usize,
    rope_dim: usize,
    per_position: usize,
    positions: &[usize],
    freqs: &FreqsCis,
    inverse: bool,
) {
    assert_eq!(x.len(), positions.len() * per_position * width);
    crate::dsv4_forward::apply_rope(
        x,
        positions.len(),
        per_position,
        width,
        rope_dim,
        freqs,
        positions,
        inverse,
    );
    for row in x.chunks_exact_mut(width) {
        bf16_in_place(&mut row[width - rope_dim..]);
    }
}

// ============================ sparse attention ============================

/// Entries per online-softmax block in the vendor `sparse_attn_kernel`.
pub const SPARSE_ATTN_BLOCK: usize = 64;

/// Vendor `sparse_attn(q, kv, attn_sink, topk_idxs, scale)` for one query row set: `q` is
/// `[queries, heads, dim]` (bf16), `kv` is `[rows, dim]` (bf16), `indices` is `[queries, k]`
/// (`-1` = empty slot). The kernel walks the selected entries in blocks of 64 with a running
/// max: each block's probabilities are `exp(s - running_max)` in f32, cast to bf16 for the PV
/// product, while the denominator sums the f32 values; the running sums rescale by
/// `exp(old_max - new_max)` at each block, and the sink joins the denominator once at the end
/// as `exp(sink - final_max)`. The running max starts at `-1e30`, so a row with no valid entry
/// returns zeros. Output is bf16.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attention(
    q: &[f32],
    queries: usize,
    heads: usize,
    dim: usize,
    kv: &[f32],
    sink: &[f32],
    indices: &[i64],
    k: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), queries * heads * dim);
    assert_eq!(kv.len() % dim, 0);
    assert_eq!(indices.len(), queries * k);
    assert_eq!(sink.len(), heads);
    let rows = kv.len() / dim;
    let mut out = vec![0f32; queries * heads * dim];
    for query in 0..queries {
        let selected = &indices[query * k..(query + 1) * k];
        for &index in selected {
            assert!(
                index < rows as i64,
                "sparse attention index {index} outside {rows} kv rows"
            );
        }
        for head in 0..heads {
            let q_row = &q[(query * heads + head) * dim..(query * heads + head + 1) * dim];
            let mut running_max = -1e30f32;
            let mut denominator = 0f32;
            let mut accumulator = vec![0f32; dim];
            for block in selected.chunks(SPARSE_ATTN_BLOCK) {
                let scores: Vec<f32> = block
                    .iter()
                    .map(|&index| {
                        if index < 0 {
                            f32::NEG_INFINITY
                        } else {
                            let row = &kv[index as usize * dim..(index as usize + 1) * dim];
                            dot(q_row, row) * scale
                        }
                    })
                    .collect();
                let block_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let previous = running_max;
                running_max = running_max.max(block_max);
                let rescale = exp_cr(previous - running_max);
                let probabilities: Vec<f32> = scores
                    .iter()
                    .map(|&score| exp_cr(score - running_max))
                    .collect();
                let block_sum: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
                denominator = denominator * rescale + block_sum as f32;
                let mut product = vec![0f64; dim];
                for (&index, &probability) in block.iter().zip(&probabilities) {
                    if index < 0 {
                        continue;
                    }
                    let weight = f64::from(bf16(probability));
                    let row = &kv[index as usize * dim..(index as usize + 1) * dim];
                    for (value, &element) in product.iter_mut().zip(row) {
                        *value += weight * f64::from(element);
                    }
                }
                for (value, &block_value) in accumulator.iter_mut().zip(&product) {
                    *value = *value * rescale + block_value as f32;
                }
            }
            denominator += exp_cr(sink[head] - running_max);
            let destination =
                &mut out[(query * heads + head) * dim..(query * heads + head + 1) * dim];
            for (slot, &value) in destination.iter_mut().zip(&accumulator) {
                *slot = bf16(value / denominator);
            }
        }
    }
    out
}

// ============================ top-k ============================

/// The `k` largest entries of `scores` by value (ties to the lower index), returned in
/// ascending index order: the vendor `topk(k, sorted=False).indices.sort()`. The vendor
/// tie-break is torch's, so parity fixtures must be tie-free at every selection boundary
/// ([`has_boundary_tie`]).
pub fn topk_ascending(scores: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut selected: Vec<usize> = order.into_iter().take(k).collect();
    selected.sort_unstable();
    selected
}

/// True when the k-th and (k+1)-th largest finite scores are equal: the selection then depends
/// on the implementation's tie-break.
pub fn has_boundary_tie(scores: &[f32], k: usize) -> bool {
    if k == 0 || k >= scores.len() {
        return false;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[k - 1].is_finite() && sorted[k - 1] == sorted[k]
}

/// Vendor `select_candidate_blocks` for one query row: score each block of `block_size`
/// entries by its best entry (`-inf` pads the tail), pin the block holding the newest
/// reachable entry (`(reachable - 1) / block_size`; none when nothing is reachable), keep the
/// `top_blocks` best and drop picks that came back `-inf`. Returns the per-entry mask.
pub fn select_candidate_blocks(
    row: &[f32],
    reachable: usize,
    top_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let width = row.len();
    let blocks = width.div_ceil(block_size);
    let mut scores: Vec<f32> = (0..blocks)
        .map(|block| {
            row[block * block_size..((block + 1) * block_size).min(width)]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    if reachable > 0 {
        scores[(reachable - 1) / block_size] = f32::INFINITY;
    }
    let keep = topk_ascending(&scores, top_blocks.min(blocks));
    let mut mask = vec![false; width];
    for block in keep {
        if scores[block] > f32::NEG_INFINITY {
            for entry in mask
                .iter_mut()
                .take(((block + 1) * block_size).min(width))
                .skip(block * block_size)
            {
                *entry = true;
            }
        }
    }
    mask
}

// ============================ single-pass mHC ============================

/// Vendor `Block.hc_mixes`: one RMS statistic over the flattened `streams * hidden` row (with
/// `norm_eps`), `mixes = (x @ fn^T) * rsqrt`, then `hc_split_sinkhorn` with `hc_eps`. Returns
/// (pre [tokens, streams], post [tokens, streams], comb [tokens, streams, streams]).
#[allow(clippy::too_many_arguments)]
pub fn hc_mixes(
    x: &[f32],
    tokens: usize,
    streams: usize,
    hidden: usize,
    function: &[f32],
    scale: &[f32],
    base: &[f32],
    iterations: u32,
    norm_eps: f32,
    hc_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let width = streams * hidden;
    let rows = (2 + streams) * streams;
    let mut mixes = matmul(x, tokens, width, function, rows);
    for token in 0..tokens {
        let rsqrt = rsqrt_mean_square(&x[token * width..(token + 1) * width], norm_eps);
        for value in &mut mixes[token * rows..(token + 1) * rows] {
            *value *= rsqrt;
        }
    }
    hc_split_sinkhorn(&mixes, tokens, streams, scale, base, iterations, hc_eps)
}

/// Vendor `hc_split_sinkhorn` (kernel.py): rows `[0:hc]` pre = `sigmoid(m * s0 + b) + eps`,
/// `[hc:2hc]` post = `2 * sigmoid(m * s1 + b)`, the rest comb = row softmax `+ eps`, a column
/// normalization, then `iterations - 1` (row, column) normalizations, each `x / (sum + eps)`.
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    tokens: usize,
    hc: usize,
    scale: &[f32],
    base: &[f32],
    iterations: u32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = (2 + hc) * hc;
    assert_eq!(mixes.len(), tokens * rows);
    let mut pre = vec![0f32; tokens * hc];
    let mut post = vec![0f32; tokens * hc];
    let mut comb = vec![0f32; tokens * hc * hc];
    for token in 0..tokens {
        let m = &mixes[token * rows..(token + 1) * rows];
        for c in 0..hc {
            pre[token * hc + c] = sigmoid(m[c] * scale[0] + base[c]) + eps;
            post[token * hc + c] = 2.0 * sigmoid(m[hc + c] * scale[1] + base[hc + c]);
        }
        let cm = &mut comb[token * hc * hc..(token + 1) * hc * hc];
        for j in 0..hc {
            let logits: Vec<f32> = (0..hc)
                .map(|k| m[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k])
                .collect();
            for (k, p) in softmax_cr(&logits).into_iter().enumerate() {
                cm[j * hc + k] = p + eps;
            }
        }
        let column = |cm: &mut [f32]| {
            for k in 0..hc {
                let sum = (0..hc).map(|j| f64::from(cm[j * hc + k])).sum::<f64>() as f32;
                for j in 0..hc {
                    cm[j * hc + k] /= sum + eps;
                }
            }
        };
        let row = |cm: &mut [f32]| {
            for j in 0..hc {
                let sum = (0..hc).map(|k| f64::from(cm[j * hc + k])).sum::<f64>() as f32;
                for k in 0..hc {
                    cm[j * hc + k] /= sum + eps;
                }
            }
        };
        column(cm);
        for _ in 0..iterations.saturating_sub(1) {
            row(cm);
            column(cm);
        }
    }
    (pre, post, comb)
}

/// Vendor `Block.hc_pre`: `sum(pre * x, dim=streams)` in f32, bf16 output `[tokens, hidden]`.
pub fn hc_pre(x: &[f32], tokens: usize, streams: usize, hidden: usize, pre: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; tokens * hidden];
    for token in 0..tokens {
        for i in 0..hidden {
            let sum: f64 = (0..streams)
                .map(|stream| {
                    f64::from(
                        pre[token * streams + stream] * x[(token * streams + stream) * hidden + i],
                    )
                })
                .sum();
            out[token * hidden + i] = bf16(sum as f32);
        }
    }
    out
}

/// Vendor `Block.hc_post`: `post[k] * f + sum_j comb[j, k] * residual[j]` (the stream sum first,
/// as the vendor expression associates), bf16 output `[tokens, streams, hidden]`.
pub fn hc_post(
    f: &[f32],
    residual: &[f32],
    tokens: usize,
    streams: usize,
    hidden: usize,
    post: &[f32],
    comb: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; tokens * streams * hidden];
    for token in 0..tokens {
        for k in 0..streams {
            let weight = post[token * streams + k];
            for i in 0..hidden {
                let mixed: f64 = (0..streams)
                    .map(|j| {
                        f64::from(
                            comb[(token * streams + j) * streams + k]
                                * residual[(token * streams + j) * hidden + i],
                        )
                    })
                    .sum();
                out[(token * streams + k) * hidden + i] =
                    bf16(weight * f[token * hidden + i] + mixed as f32);
            }
        }
    }
    out
}

// ============================ MoE ============================

/// `exp`, evaluated in f64 and rounded once.
#[inline]
pub fn exp_cr(x: f32) -> f32 {
    f64::from(x).exp() as f32
}

/// `base ^ exponent`, evaluated in f64 and rounded once.
#[inline]
pub fn pow_cr(base: f32, exponent: f32) -> f32 {
    f64::from(base).powf(f64::from(exponent)) as f32
}

/// torch `F.softplus` (beta 1, threshold 20), evaluated in f64 and rounded once.
#[inline]
pub fn softplus(x: f32) -> f32 {
    let x = f64::from(x);
    (if x > 20.0 { x } else { x.exp().ln_1p() }) as f32
}

/// torch `F.silu`, `x / (1 + exp(-x))`, evaluated in f64 and rounded once.
#[inline]
pub fn silu(x: f32) -> f32 {
    let x = f64::from(x);
    (x / (1.0 + (-x).exp())) as f32
}

/// torch `sigmoid`, `1 / (1 + exp(-x))`, evaluated in f64 and rounded once.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-f64::from(x)).exp())) as f32
}

/// torch `softmax` over one row, evaluated in f64 and rounded once per element.
pub fn softmax_cr(row: &[f32]) -> Vec<f32> {
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &v| acc.max(f64::from(v)));
    let exps: Vec<f64> = row.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
}

/// Vendor `Gate.forward` for one token (score `sqrtsoftplus`, `gate_temp` 1): the bias steers
/// selection only, the weights are the raw scores, renormalized (`+ 1e-20`) when `normalize`
/// and top-k > 1, then scaled. Returns (experts in selection order, weights).
pub fn route_sqrtsoftplus(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    normalize: bool,
    route_scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    let scores: Vec<f32> = logits.iter().map(|&value| softplus(value).sqrt()).collect();
    let biased: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();
    let mut order: Vec<usize> = (0..biased.len()).collect();
    order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
    let experts: Vec<usize> = order.into_iter().take(top_k).collect();
    let mut weights: Vec<f32> = experts.iter().map(|&expert| scores[expert]).collect();
    if normalize && top_k > 1 {
        let sum = weights.iter().map(|&weight| f64::from(weight)).sum::<f64>() as f32 + 1e-20;
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    for weight in &mut weights {
        *weight *= route_scale;
    }
    (experts, weights)
}

/// Vendor `Expert.forward` on `rows` tokens of `x` (bf16): `w1`/`w3` quantized linears (bf16
/// outputs read as f32), up clamped to +-limit and gate from above when `limit > 0`,
/// `silu(gate) * up`, times the routing weight when given, cast to bf16, then `w2`.
#[allow(clippy::too_many_arguments)]
pub fn expert(
    x: &[f32],
    rows: usize,
    hidden: usize,
    inter: usize,
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    limit: f32,
    weights: Option<&[f32]>,
) -> Vec<f32> {
    let gate = linear_quant(x, rows, hidden, w1, inter);
    let up = linear_quant(x, rows, hidden, w3, inter);
    let mut act = vec![0f32; rows * inter];
    for row in 0..rows {
        for i in 0..inter {
            let mut g = gate[row * inter + i];
            let mut u = up[row * inter + i];
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            let mut value = silu(g) * u;
            if let Some(weights) = weights {
                value *= weights[row];
            }
            act[row * inter + i] = bf16(value);
        }
    }
    linear_quant(&act, rows, inter, w2, hidden)
}

// ============================ Engram ============================

/// The vendor bucket primes: for every engram layer slot, `max_ngram - 1` n-gram sizes of
/// `heads` primes, drawn in order above `bucket_base - 1` and never reused. `[slot][ngram][head]`.
pub fn engram_primes(
    slots: usize,
    max_ngram: usize,
    heads: usize,
    bucket_base: u64,
) -> Vec<Vec<Vec<u64>>> {
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n.is_multiple_of(2) {
            return n == 2;
        }
        let mut divisor = 3;
        while divisor * divisor <= n {
            if n.is_multiple_of(divisor) {
                return false;
            }
            divisor += 2;
        }
        true
    }
    let mut seen = std::collections::BTreeSet::new();
    (0..slots)
        .map(|_| {
            (1..max_ngram)
                .map(|_| {
                    let mut current = bucket_base.saturating_sub(1);
                    (0..heads)
                        .map(|_| {
                            let mut candidate = current + 1;
                            while !is_prime(candidate) || seen.contains(&candidate) {
                                candidate += 1;
                            }
                            seen.insert(candidate);
                            current = candidate;
                            candidate
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// numpy `SeedSequence(entropy).generate_state(4, uint64)` for a non-negative integer seed and
/// no spawn key.
fn seed_sequence_state(entropy: u64) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;
    let mut words = Vec::new();
    let mut rest = entropy;
    if rest == 0 {
        words.push(0u32);
    }
    while rest > 0 {
        words.push(rest as u32);
        rest >>= 32;
    }
    let mut hash_const = INIT_A;
    let hashmix = |value: u32, hash_const: &mut u32| -> u32 {
        let mut value = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(*hash_const);
        value ^ (value >> XSHIFT)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    };
    let mut pool = [0u32; 4];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..4 {
        for destination in 0..4 {
            if source != destination {
                let hashed = hashmix(pool[source], &mut hash_const);
                pool[destination] = mix(pool[destination], hashed);
            }
        }
    }
    for &word in words.iter().skip(4) {
        for slot in pool.iter_mut() {
            let hashed = hashmix(word, &mut hash_const);
            *slot = mix(*slot, hashed);
        }
    }
    let mut hash_const = INIT_B;
    let mut state = [0u32; 8];
    for (i, slot) in state.iter_mut().enumerate() {
        let mut value = pool[i % 4] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        value = value.wrapping_mul(hash_const);
        *slot = value ^ (value >> XSHIFT);
    }
    let mut out = [0u64; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u64::from(state[2 * i]) | (u64::from(state[2 * i + 1]) << 32);
    }
    out
}

/// numpy `PCG64` (XSL-RR 128/64) seeded from a [`seed_sequence_state`].
struct Pcg64 {
    state: u128,
    increment: u128,
}

impl Pcg64 {
    const MULTIPLIER: u128 = (0x2360_ed05_1fc6_5da4u128 << 64) | 0x4385_df64_9fcc_f645u128;

    fn seeded(entropy: u64) -> Self {
        let words = seed_sequence_state(entropy);
        let initial_state = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let sequence = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let mut generator = Self {
            state: 0,
            increment: (sequence << 1) | 1,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initial_state);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let rotation = (self.state >> 122) as u32;
        (((self.state >> 64) as u64) ^ (self.state as u64)).rotate_right(rotation)
    }

    /// numpy `bounded_lemire_uint64(rng)`: a uniform draw in `0..=range`.
    fn bounded(&mut self, range: u64) -> u64 {
        let exclusive = range + 1;
        let mut product = u128::from(self.next_u64()) * u128::from(exclusive);
        let mut leftover = product as u64;
        if leftover < exclusive {
            let threshold = (u64::MAX - range) % exclusive;
            while leftover < threshold {
                product = u128::from(self.next_u64()) * u128::from(exclusive);
                leftover = product as u64;
            }
        }
        (product >> 64) as u64
    }
}

/// Vendor `compute_hash_multipliers` for one engram layer: `default_rng(10007 * layer)
/// .integers(0, bound, size=max_ngram, dtype=int64) * 2 + 1` with
/// `bound = max(1, (i64::MAX // compressed_vocab) // 2)`. Only the 64-bit Lemire path numpy
/// takes for `bound - 1 > u32::MAX` is ported; smaller vocab-derived bounds are refused.
pub fn engram_multipliers(
    layer: u32,
    max_ngram: usize,
    compressed_vocab: u64,
) -> Result<Vec<i64>, String> {
    let bound = ((i64::MAX as u64 / compressed_vocab) / 2).max(1);
    let range = bound - 1;
    if range <= u64::from(u32::MAX) {
        return Err(format!(
            "engram multiplier bound {bound} takes numpy's 32-bit draw path, which is not ported"
        ));
    }
    let mut generator = Pcg64::seeded(10007 * u64::from(layer));
    Ok((0..max_ngram)
        .map(|_| generator.bounded(range) as i64 * 2 + 1)
        .collect())
}

/// Vendor `NgramHashState.forward` for one position, given the compressed-id history `history`
/// (positions `0..=position`, `None` = a dead token such as an image slot): the
/// `max_ngram - 1` n-gram hashes per head, `[ngram][head]` flattened, each offset into the
/// layer's table (the cumulative prime sums).
pub fn engram_hash_row(
    history: &[Option<i64>],
    position: usize,
    pad: i64,
    multipliers: &[i64],
    primes: &[Vec<u64>],
) -> Vec<u64> {
    let max_ngram = multipliers.len();
    let mut tokens = Vec::with_capacity(max_ngram);
    let mut blocked = false;
    for shift in 0..max_ngram {
        let source = if position >= shift {
            history[position - shift]
        } else {
            history[0]
        };
        blocked = blocked || position < shift || source.is_none();
        tokens.push(if blocked {
            pad
        } else {
            source.expect("unblocked")
        });
    }
    let mut out = Vec::with_capacity((max_ngram - 1) * primes.first().map_or(0, Vec::len));
    let mut offset = 0u64;
    let mut rolling = tokens[0].wrapping_mul(multipliers[0]);
    for (ngram, heads) in primes.iter().enumerate() {
        rolling ^= tokens[ngram + 1].wrapping_mul(multipliers[ngram + 1]);
        for &prime in heads {
            out.push(rolling as u64 % prime + offset);
            offset += prime;
        }
    }
    out
}

/// Vendor `Engram.forward` gate for one token and stream: `h` and `key` rows of `hidden`, the
/// `q_weight * k_weight` row, `value` the shared value row. Returns the bf16 output row.
#[allow(clippy::too_many_arguments)]
pub fn engram_mix(
    h: &[f32],
    key: &[f32],
    weight: &[f32],
    value: &[f32],
    norm_eps: f32,
    gate_clamp: f32,
    out: &mut [f32],
) {
    let hidden = h.len();
    let rstd = rsqrt_mean_square(h, norm_eps) * rsqrt_mean_square(key, norm_eps);
    let sum: f64 = h
        .iter()
        .zip(weight)
        .zip(key)
        .map(|((&h, &w), &k)| f64::from(h * w * k))
        .sum();
    let dot = (sum as f32) * rstd * ((hidden as f64).powf(-0.5) as f32);
    let gate = sigmoid(dot.abs().max(gate_clamp).sqrt().copysign(dot));
    for i in 0..hidden {
        out[i] = bf16(h[i] + gate * value[i]);
    }
}

// ============================ checkpoint dequantization ============================

/// FP8 e4m3 `[rows, cols]` with an e8m0 scale per `block_rows x 32` block (`block_rows` 32 for
/// dense weights, 1 for Engram table rows) -> f32.
pub fn dequant_fp8(
    weight: &[u8],
    scale: &[u8],
    rows: usize,
    cols: usize,
    block_rows: usize,
) -> Vec<f32> {
    let scale_cols = cols.div_ceil(32);
    assert_eq!(weight.len(), rows * cols, "fp8 weight bytes");
    assert_eq!(
        scale.len(),
        rows.div_ceil(block_rows) * scale_cols,
        "fp8 scale grid"
    );
    let mut out = vec![0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = e4m3(weight[row * cols + col])
                * e8m0(scale[(row / block_rows) * scale_cols + col / 32]);
        }
    }
    out
}

/// FP4 e2m1 pairs packed in bytes (`[rows, cols / 2]`, low nibble first) with an e8m0 scale per
/// 32 inputs (`[rows, cols / 32]`) -> f32 `[rows, cols]`.
pub fn dequant_fp4(packed: &[u8], scale: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(packed.len(), rows * cols / 2, "fp4 packed bytes");
    assert_eq!(scale.len(), rows * cols / 32, "fp4 scale grid");
    let mut out = vec![0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let byte = packed[(row * cols + col) / 2];
            let nibble = if col % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            out[row * cols + col] = e2m1(nibble) * e8m0(scale[row * (cols / 32) + col / 32]);
        }
    }
    out
}

/// Quantize f32 `[rows, cols]` the way the checkpoint stores an FP8 weight: one power-of-two
/// e8m0 scale per `block_rows x 32` block (`2^ceil(log2(amax / 448))`, amax floored at 1e-4),
/// e4m3 codes. Returns (codes, scales). Used to build checkpoint-format fixtures.
pub fn quantize_fp8(
    values: &[f32],
    rows: usize,
    cols: usize,
    block_rows: usize,
) -> (Vec<u8>, Vec<u8>) {
    let scale_rows = rows.div_ceil(block_rows);
    let scale_cols = cols.div_ceil(32);
    let mut scales = vec![0u8; scale_rows * scale_cols];
    let mut scale_values = vec![0f32; scale_rows * scale_cols];
    let inverse = (1.0f64 / 448.0) as f32;
    for block_row in 0..scale_rows {
        for block_col in 0..scale_cols {
            let mut amax = 0f32;
            for row in block_row * block_rows..((block_row + 1) * block_rows).min(rows) {
                for col in block_col * 32..((block_col + 1) * 32).min(cols) {
                    amax = amax.max(values[row * cols + col].abs());
                }
            }
            let scale = pow2_ceil(amax.max(1e-4) * inverse);
            scales[block_row * scale_cols + block_col] = to_e8m0(scale);
            scale_values[block_row * scale_cols + block_col] = scale;
        }
    }
    let codes = (0..rows * cols)
        .map(|index| {
            let (row, col) = (index / cols, index % cols);
            let scale = scale_values[(row / block_rows) * scale_cols + col / 32];
            to_e4m3((values[index] / scale).clamp(-448.0, 448.0))
        })
        .collect();
    (codes, scales)
}

/// Quantize f32 `[rows, cols]` the way the checkpoint stores an FP4 expert weight: one
/// power-of-two e8m0 scale per 32 inputs (`2^ceil(log2(amax / 6))`), e2m1 codes packed low
/// nibble first. Returns (packed bytes, scales).
pub fn quantize_fp4(values: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(cols % 32, 0, "fp4 expert inputs are a multiple of 32");
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 32];
    let inverse = (1.0f64 / 6.0) as f32;
    let floor = 6.0f32 * 2f32.powi(-126);
    for row in 0..rows {
        for group in 0..cols / 32 {
            let slice = &values[row * cols + group * 32..row * cols + (group + 1) * 32];
            let amax = slice.iter().fold(0f32, |acc, value| acc.max(value.abs()));
            let scale = pow2_ceil(amax.max(floor) * inverse);
            scales[row * (cols / 32) + group] = to_e8m0(scale);
            for (offset, &value) in slice.iter().enumerate() {
                let col = group * 32 + offset;
                let nibble = to_e2m1(e2m1_rne((value / scale).clamp(-6.0, 6.0)));
                let index = (row * cols + col) / 2;
                if col % 2 == 0 {
                    packed[index] |= nibble;
                } else {
                    packed[index] |= nibble << 4;
                }
            }
        }
    }
    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 1.0);
        // 1 + 2^-8 is exactly halfway between 1 and 1 + 2^-7: ties go to the even mantissa (1.0).
        assert_eq!(bf16(1.0 + 2f32.powi(-8)), 1.0);
        assert_eq!(bf16(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2.0 * 2f32.powi(-7));
        assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    }

    /// numpy `default_rng(10007 * layer).integers(0, bound, size=4, dtype=int64) * 2 + 1` for
    /// the 1017 config (layers 1 and 14, compressed vocab 99092), as the vendor `engram.py`
    /// derives them at load (dumped by the lane's `engram_constants.py`).
    #[test]
    fn engram_multipliers_match_the_vendor_numpy_draw() {
        assert_eq!(
            engram_multipliers(1, 4, 99_092).unwrap(),
            [
                76632096046245,
                4839876093313,
                35959672319349,
                73987337458391
            ]
        );
        assert_eq!(
            engram_multipliers(14, 4, 99_092).unwrap(),
            [
                67716810739261,
                51510806800915,
                30921347202721,
                82619226485591
            ]
        );
    }

    /// The vendor prime layout for the 1017 config: the first and last bucket of each layer and
    /// the per-layer row sums the checkpoint's `engram_num_embeddings` carries.
    #[test]
    fn engram_primes_match_the_vendor_layout() {
        let primes = engram_primes(2, 4, 8, 16_000_000);
        assert_eq!(primes[0][0][0], 16_000_057);
        assert_eq!(primes[1][0][0], 16_000_477);
        let sums: Vec<u64> = primes
            .iter()
            .map(|layer| layer.iter().flatten().sum())
            .collect();
        assert_eq!(sums, [384_006_168, 384_016_682]);
    }

    #[test]
    fn engram_hash_pads_missing_history_and_offsets_buckets() {
        let primes = vec![vec![7, 11], vec![13, 17]];
        let multipliers = [3, 5, 9];
        let history = [Some(4), Some(6)];
        // Position 0: shifts 1 and 2 are before the start, so both use the pad id.
        let row = engram_hash_row(&history, 0, 2, &multipliers, &primes);
        let h2 = (4 * 3) ^ (2 * 5);
        let h3 = h2 ^ (2 * 9);
        assert_eq!(
            row,
            [
                h2 as u64 % 7,
                h2 as u64 % 11 + 7,
                h3 as u64 % 13 + 18,
                h3 as u64 % 17 + 31
            ]
        );
        // A dead token blocks every longer look-back too.
        let dead = [Some(4), None, Some(6)];
        let row = engram_hash_row(&dead, 2, 2, &multipliers, &primes);
        let h2 = (6 * 3) ^ (2 * 5);
        assert_eq!(row[0], h2 as u64 % 7);
    }

    /// Checkpoint-format quantization lands every value within half a grid step of its input
    /// (e4m3: 3 mantissa bits, e2m1: 1), with power-of-two scales, and dequantization reads the
    /// layout back (the 32x32 block grid, low nibble first).
    #[test]
    fn fp8_and_fp4_checkpoint_quantization_reads_back() {
        let values: Vec<f32> = (0..64 * 40)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let (codes, scales) = quantize_fp8(&values, 64, 40, 32);
        assert_eq!(scales.len(), 2 * 2);
        let dequant = dequant_fp8(&codes, &scales, 64, 40, 32);
        for (&value, &read) in values.iter().zip(&dequant) {
            assert!(
                (value - read).abs() <= value.abs() / 16.0 + 2f32.powi(-9) * 0.05,
                "{value} {read}"
            );
        }
        let values: Vec<f32> = (0..4 * 64)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 3.0)
            .collect();
        let (packed, scales) = quantize_fp4(&values, 4, 64);
        let dequant = dequant_fp4(&packed, &scales, 4, 64);
        for (&value, &read) in values.iter().zip(&dequant) {
            let scale = e8m0(scales[0]);
            assert!(
                (value - read).abs() <= value.abs() / 4.0 + 0.25 * scale,
                "{value} {read}"
            );
        }
        // Low nibble first: code 1 (0.5) then code 15 (-6) at scale 2^0.
        let read = dequant_fp4(&[0xf1; 16], &[127], 1, 32);
        assert_eq!((read[0], read[1]), (0.5, -6.0));
    }

    /// One block of entries: the online softmax equals the direct sink softmax with bf16
    /// probabilities; empty slots contribute nothing, and an all-empty row returns zeros.
    #[test]
    fn sparse_attention_single_block_matches_the_direct_form() {
        let dim = 8;
        let q: Vec<f32> = (0..2 * dim)
            .map(|i| bf16((i as f32 * 0.37).sin()))
            .collect();
        let kv: Vec<f32> = (0..5 * dim)
            .map(|i| bf16((i as f32 * 0.21).cos()))
            .collect();
        let sink = [0.3f32];
        let indices = [0i64, 2, 4, -1, -1, -1, -1, -1];
        let out = sparse_attention(&q, 2, 1, dim, &kv, &sink, &indices, 4, 0.5);
        let scores: Vec<f32> = [0usize, 2, 4]
            .iter()
            .map(|&row| crate::dsv4_forward::dot(&q[..dim], &kv[row * dim..(row + 1) * dim]) * 0.5)
            .collect();
        let max = scores.iter().copied().fold(-1e30f32, f32::max);
        let probabilities: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let denominator = probabilities.iter().sum::<f32>() + (sink[0] - max).exp();
        for i in 0..dim {
            let mut acc = 0f64;
            for (&row, &p) in [0usize, 2, 4].iter().zip(&probabilities) {
                acc += f64::from(bf16(p)) * f64::from(kv[row * dim + i]);
            }
            let expected = bf16(acc as f32 / denominator);
            assert!(
                (out[i] - expected).abs() <= expected.abs() * 2f32.powi(-7),
                "{i}"
            );
        }
        assert!(out[dim..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn candidate_blocks_pin_the_newest_block_and_drop_unreachable_picks() {
        let inf = f32::NEG_INFINITY;
        // Seven reachable entries in blocks of 2; the newest (entry 6) sits in block 3.
        let row = [5.0, 1.0, 0.5, 0.2, 9.0, 3.0, 0.1, inf];
        let mask = select_candidate_blocks(&row, 7, 2, 2);
        // Block 3 is pinned; the best other block is block 2 (score 9).
        assert_eq!(mask, [false, false, false, false, true, true, true, true]);
        // Fewer reachable blocks than the budget: -inf blocks are not kept.
        let row = [2.0, inf, inf, inf];
        let mask = select_candidate_blocks(&row, 1, 2, 2);
        assert_eq!(mask, [true, true, false, false]);
    }

    #[test]
    fn topk_is_by_value_then_returned_in_position_order() {
        let scores = [0.5, 3.0, -1.0, 2.0, 3.5];
        assert_eq!(topk_ascending(&scores, 3), [1, 3, 4]);
        assert!(!has_boundary_tie(&scores, 3));
        assert!(has_boundary_tie(&[1.0, 2.0, 2.0, 0.0], 1));
        assert!(!has_boundary_tie(&[1.0, 2.0, 2.0, 0.0], 2));
        assert!(!has_boundary_tie(
            &[1.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
            2
        ));
    }
}
