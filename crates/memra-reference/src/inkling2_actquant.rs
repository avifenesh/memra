//! Inkling 2 Mini: the activation-quant numeric program of the routed experts (the checkpoint's
//! declared default class: NVFP4 W4A4 on layers 0..19, MXFP8 W8A8 on layers 20..23).
//!
//! The specification is the checkpoint's `hf_quant_config.json` (per-token amax, block 16, e2m1
//! values, e4m3 block scales, `fouroversix_mae` scale selection; MXFP8 block 32 with ue8m0 scales)
//! as implemented by the pinned serving image's ops. Every quantizer function here is one IEEE
//! fp32 operation per source operation, in the source order, and is proven bit-identical to the
//! torch port of those ops by the golden vectors in `fixtures/inkling2-actquant-goldens.bin`
//! (`research/modelplan-onboarding-inkling2mini-20261121/actquant/gen_goldens.py`).
//!
//! Two parts are models, not proofs, and say so:
//! - the GEMM accumulator. The checkpoint declares fp32 accumulation and no order. The functions
//!   named `index_order` (and the goldens) add in ascending index order. The sm_120 tensor core
//!   does not: [`dot_chain_rz`] (W4A4) and [`dot_chain_trunc`] (W8A8) are what the card computes,
//!   measured there and pinned by a committed device capture. Every model stays inside
//!   `(K - 1) * 2^-23 * sum |a_i w_i|` of the exact sum.
//!
//! SiTU is not the serving image's approx fp32 sequence: it is the mathematical value rounded once
//! to bf16 ([`situ_value`]), the program the kernels compute bit for bit. Non-finite inputs follow
//! the source's operations: maxNum / minNum reductions and clamps (a NaN operand is dropped), the
//! NaN-propagating MXFP8 clamp and its exponent-255 NaN scale, NaN codes as the sm_120 casts give
//! them.

/// Largest finite fp32, the clamp of every reciprocal in the NVFP4 quantizer.
pub const FP32_MAX: f32 = f32::MAX;

const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// fp32 -> bf16 -> fp32, round to nearest even.
#[inline]
pub fn bf16(x: f32) -> f32 {
    memra_gguf::dsv41_forward::bf16(x)
}

/// fp32 -> e2m1 nibble: round to nearest, ties to the even code, saturating at 6; bit 3 is the
/// fp32 sign bit, so -0 and negative values that round to zero encode as 0x8. NaN of either sign
/// encodes as 0x7, what `cvt.rn.satfinite.e2m1x2.f32` gives on sm_120 (the gate's `ink2-cvt-hw`).
#[inline]
pub fn e2m1_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 7;
    }
    let a = x.abs();
    let code = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    code | (((x.to_bits() >> 31) as u8) << 3)
}

/// e2m1 nibble -> fp32.
#[inline]
pub fn e2m1_decode(nibble: u8) -> f32 {
    let magnitude = E2M1_GRID[usize::from(nibble & 7)];
    if nibble & 8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// e4m3 byte -> fp32 (`float8_e4m3fn`: `0x7f` and `0xff` are NaN).
#[inline]
pub fn e4m3_decode(byte: u8) -> f32 {
    if byte & 0x7f == 0x7f {
        return f32::NAN;
    }
    let exponent = i32::from((byte >> 3) & 0xf);
    let mantissa = f32::from(byte & 7);
    let magnitude = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if byte & 0x80 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// fp32 -> e4m3 byte: round to nearest, ties to even, subnormals kept, saturating at 448, the sign
/// bit carried from the fp32 sign bit (`cvt.rn.satfinite`); NaN of either sign encodes as `0x7f`,
/// what the sm_120 conversion gives (the gate's `ink2-cvt-hw`).
#[inline]
pub fn e4m3_encode(x: f32) -> u8 {
    let sign = ((x.to_bits() >> 31) as u8) << 7;
    if x.is_nan() {
        return 0x7f;
    }
    let a = f64::from(x.abs()).min(448.0);
    let code = if a < 2f64.powi(-6) {
        (a * 512.0).round_ties_even() as u8
    } else {
        let exponent = a.log2().floor() as i32;
        // a is a fp32 value at most 448: the scaled mantissa is exact in f64
        let mut exponent = exponent;
        let mut mantissa = (a / 2f64.powi(exponent) * 8.0).round_ties_even() as i32;
        if mantissa < 8 {
            // log2 rounded up across a power of two
            exponent -= 1;
            mantissa = (a / 2f64.powi(exponent) * 8.0).round_ties_even() as i32;
        }
        if mantissa >= 16 {
            exponent += 1;
            mantissa = 8;
        }
        ((((exponent + 7) << 3) | (mantissa - 8)) as u8).min(0x7e)
    };
    code | sign
}

/// One row quantized to NVFP4.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Row {
    /// e2m1 nibbles, two per byte, the even element in the low nibble: `[K / 2]`.
    pub codes: Vec<u8>,
    /// e4m3 block scales, one per 16 elements: `[K / 16]`.
    pub scales: Vec<u8>,
    /// The row's max magnitude; the row scale of the GEMM output is built from it.
    pub amax: f32,
}

fn pairwise_error16(error: &[f32; 16]) -> f32 {
    let mut level = *error;
    let mut width = 16;
    while width > 1 {
        for i in 0..width / 2 {
            level[i] = level[2 * i] + level[2 * i + 1];
        }
        width /= 2;
    }
    level[0]
}

/// NVFP4 per-row activation quantization with the four-over-six MAE scale selection.
///
/// `x` is one row of bf16 values widened to fp32, its length a multiple of 16. There is no amax
/// epsilon: a zero row gives encoder scale `FP32_MAX`, zero block scales and zero codes.
pub fn nvfp4_act_quant(x: &[f32]) -> Nvfp4Row {
    let canonical = |v: f32| {
        if v.is_nan() {
            f32::from_bits(0x7fc0_0000)
        } else {
            v
        }
    };
    assert!(
        x.len().is_multiple_of(16),
        "NVFP4 rows are whole blocks of 16"
    );
    // maxNum over the row (a NaN element is dropped); an all-NaN row gives the quiet NaN 0x7fc00000
    let amax = canonical(x.iter().fold(f32::NAN, |m, v| m.max(v.abs())));
    let encoder = (1536.0f32 / amax).min(FP32_MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * (1.0f32 / 1536.0f32);
    let enc6 = encoder * (1.0f32 / 6.0f32);
    let enc4 = encoder * 0.25f32;
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; x.len() / 16];
    for (b, block) in x.chunks_exact(16).enumerate() {
        let block_max = block.iter().fold(f32::NAN, |m, v| m.max(v.abs()));
        let candidate = |enc: f32| {
            let byte = e4m3_encode((block_max * enc).min(448.0));
            let scale = e4m3_decode(byte);
            let inverse = (1.0f32 / (scale * decoder)).min(FP32_MAX);
            let mut nibbles = [0u8; 16];
            let mut error = [0f32; 16];
            for i in 0..16 {
                nibbles[i] = e2m1_encode(block[i] * inverse);
                error[i] = ((e2m1_decode(nibbles[i]) * scale) * outer - block[i]).abs();
            }
            (byte, nibbles, pairwise_error16(&error))
        };
        let (byte6, nibbles6, error6) = candidate(enc6);
        let (byte4, nibbles4, error4) = candidate(enc4);
        // strict: a tie keeps the 6-based candidate
        let (byte, nibbles) = if error4 < error6 {
            (byte4, nibbles4)
        } else {
            (byte6, nibbles6)
        };
        scales[b] = byte;
        for i in 0..8 {
            codes[b * 8 + i] = nibbles[2 * i] | (nibbles[2 * i + 1] << 4);
        }
    }
    Nvfp4Row {
        codes,
        scales,
        amax,
    }
}

/// The values the W4 GEMM multiplies: `val(code) * e4m3(scale)` per element (no row factor).
pub fn nvfp4_dequant(codes: &[u8], scales: &[u8]) -> Vec<f32> {
    assert_eq!(codes.len(), scales.len() * 8);
    let mut out = Vec::with_capacity(codes.len() * 2);
    for (i, byte) in codes.iter().enumerate() {
        let scale = e4m3_decode(scales[i / 8]);
        out.push(e2m1_decode(byte & 0xf) * scale);
        out.push(e2m1_decode(byte >> 4) * scale);
    }
    out
}

/// MXFP8 activation quantization: e4m3 values with one round-up power-of-two (ue8m0) scale per 32.
/// Returns the e4m3 bytes `[K]` and the biased exponents `[K / 32]`.
pub fn mxfp8_act_quant(x: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert!(
        x.len().is_multiple_of(32),
        "MXFP8 rows are whole blocks of 32"
    );
    let min_scale = 2f32.powi(-127);
    let mut values = Vec::with_capacity(x.len());
    let mut exponents = Vec::with_capacity(x.len() / 32);
    for block in x.chunks_exact(32) {
        let amax = block.iter().fold(f32::NAN, |m, v| m.max(v.abs()));
        let raw = amax * (1.0f32 / 448.0f32);
        let rounded = ((raw.to_bits() as i32).wrapping_add(0x007f_ffff) & 0x7f80_0000) >> 23;
        let exponent = if raw.is_nan() {
            255
        } else if raw <= min_scale {
            0
        } else {
            rounded.min(254)
        };
        // exponent 255 (a block whose amax is NaN): NaN scale, zero inverse (mxfp8.py:23,25)
        let scale = if exponent == 255 {
            f32::NAN
        } else {
            f32::from_bits((exponent as u32) << 23).max(min_scale)
        };
        let inverse = if exponent == 0 {
            2f32.powi(127)
        } else if exponent == 255 {
            0.0
        } else {
            1.0f32 / scale
        };
        for v in block {
            // the clamp propagates NaN (propagate_nan=ALL in the source)
            let q = v * inverse;
            values.push(e4m3_encode(if q.is_nan() {
                q
            } else {
                q.clamp(-448.0, 448.0)
            }));
        }
        exponents.push(exponent as u8);
    }
    (values, exponents)
}

/// The values the W8 GEMM multiplies: `e4m3(value) * 2^(exponent - 127)`, exponent byte 0 read as
/// `2^-127`.
pub fn mxfp8_dequant(values: &[u8], exponents: &[u8]) -> Vec<f32> {
    assert_eq!(values.len(), exponents.len() * 32);
    values
        .iter()
        .enumerate()
        .map(|(i, byte)| {
            let e = u32::from(exponents[i / 32]);
            let scale = f32::from_bits(if e == 0 { 0x0040_0000 } else { e << 23 });
            e4m3_decode(*byte) * scale
        })
        .collect()
}

/// fp32 dot product, products then additions in ascending index order. The reference model of
/// the expert GEMM's accumulator; see the module note.
pub fn dot_index_order(a: &[f32], w: &[f32]) -> f32 {
    assert_eq!(a.len(), w.len());
    let mut acc = 0f32;
    for (x, y) in a.iter().zip(w) {
        acc += x * y;
    }
    acc
}

/// The sm_120 block-scaled tensor-core accumulator as measured on the card: one fp32 accumulator
/// chained over ascending `step`-wide K steps, each step adding the exact sum of its products to
/// the accumulator with a single rounding toward zero. On an RTX 5090 Laptop (sm_120a) this equals
/// the `kind::mxf4nvf4` m16n8k64 chain bit for bit (`step` 64) on every element the gate runs and on
/// the committed device capture. It is a model of that hardware, not part of the checkpoint's
/// specification, and it does not describe the `kind::mxf8f6f4` e4m3 chain.
pub fn dot_chain_rz(a: &[f32], w: &[f32], step: usize) -> f32 {
    assert_eq!(a.len(), w.len());
    let mut acc = 0f32;
    for (ca, cw) in a.chunks(step).zip(w.chunks(step)) {
        // The products are exact in fp32. Their step sum is exact in f64 while the step's products
        // span fewer than 53 bits, which holds for e2m1 values under block scales within about
        // 2^20 of each other; wider steps are outside what the model was measured on.
        let mut sum = 0f64;
        for (x, y) in ca.iter().zip(cw) {
            sum += f64::from(*x) * f64::from(*y);
        }
        acc = add_round_toward_zero(acc, sum);
    }
    acc
}

/// The exponent the `kind::mxf8f6f4` tensor core gives each MXFP8 operand: the e4m3 field
/// exponent (`E - 7`; a subnormal or zero code counts as `-6`, not renormalised) plus its ue8m0
/// block exponent (`byte - 127`, byte 0 read as `-127`). One per element.
pub fn mxfp8_operand_exponents(values: &[u8], exponents: &[u8]) -> Vec<i32> {
    assert_eq!(values.len(), exponents.len() * 32);
    values
        .iter()
        .enumerate()
        .map(|(i, byte)| {
            let field = i32::from((byte >> 3) & 0xf);
            let e = i32::from(exponents[i / 32]);
            (if field == 0 { -6 } else { field - 7 }) + if e == 0 { -127 } else { e - 127 }
        })
        .collect()
}

/// The sm_120 `kind::mxf8f6f4` e4m3 chain as measured on the card (K step 32, `guard` 25): within
/// a step every product and the accumulator are truncated toward zero to the quantum
/// `2^(emax - guard)`, where `emax` is the largest of the accumulator's fp32 exponent and the
/// products' operand-exponent sums over the products whose operands are both nonzero. The operand
/// exponents are the hardware's ([`mxfp8_operand_exponents`]): the e4m3 field exponent, so a
/// subnormal code sits at `-6`. The truncated terms are summed exactly and the sum is truncated to
/// fp32. Bit-identical to the device on the committed capture (subnormal-heavy records included)
/// and on every element the gate runs. A model of that hardware, not of the checkpoint.
pub fn dot_chain_trunc(
    a: &[f32],
    a_exp: &[i32],
    w: &[f32],
    w_exp: &[i32],
    step: usize,
    guard: i32,
) -> f32 {
    assert!(a.len() == w.len() && a.len() == a_exp.len() && w.len() == w_exp.len());
    let fp32_exponent = |x: f32| ((f64::from(x).to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut acc = 0f32;
    for s0 in (0..a.len()).step_by(step) {
        let s1 = (s0 + step).min(a.len());
        let mut emax = i32::MIN;
        for i in s0..s1 {
            if a[i] != 0.0 && w[i] != 0.0 {
                emax = emax.max(a_exp[i] + w_exp[i]);
            }
        }
        if acc != 0.0 {
            emax = emax.max(fp32_exponent(acc));
        }
        if emax == i32::MIN {
            continue;
        }
        let shift = emax - guard;
        let q = 2f64.powi(shift);
        // every term is below 2^(guard + 2) quanta: the integer sum is exact
        let mut sum = (f64::from(acc) / q).trunc() as i64;
        for i in s0..s1 {
            sum += ((f64::from(a[i]) * f64::from(w[i])) / q).trunc() as i64;
        }
        let mut magnitude = sum.unsigned_abs();
        let mut scale = shift;
        let bits = 64 - magnitude.leading_zeros() as i32;
        if bits > 24 {
            magnitude >>= bits - 24; // toward zero
            scale += bits - 24;
        }
        let value = (magnitude as f64 * 2f64.powi(scale)) as f32;
        acc = if sum < 0 { -value } else { value };
    }
    acc
}

/// `acc + sum` rounded once toward zero to fp32, without a double rounding: the f64 sum is split
/// into its rounded value and its exact error first.
fn add_round_toward_zero(acc: f32, sum: f64) -> f32 {
    let a = f64::from(acc);
    let hi = a + sum;
    // TwoSum: hi + lo == a + sum exactly
    let bb = hi - a;
    let lo = (a - (hi - bb)) + (sum - bb);
    let mut r = hi as f32;
    if !r.is_finite() {
        return if hi.is_finite() {
            f32::MAX.copysign(r)
        } else {
            r
        };
    }
    let d = f64::from(r) - hi; // exact
    let positive = hi > 0.0 || (hi == 0.0 && lo > 0.0);
    let negative = hi < 0.0 || (hi == 0.0 && lo < 0.0);
    if (positive && d > lo) || (negative && d < lo) {
        // |r| is above the exact value: one step toward zero
        r = f32::from_bits(r.to_bits() - 1);
    }
    r
}

/// The NVFP4 GEMM's per-row output factor: `(act_amax * w_amax) * rcp(1536 * denominator)`.
#[inline]
pub fn nvfp4_row_scale(act_amax: f32, w_amax: f32, denominator: f32) -> f32 {
    (act_amax * w_amax) * (1.0f32 / (1536.0f32 * denominator))
}

/// NVFP4 expert weights as stored: `[E, N, K / 2]` codes, `[E, N, K / 16]` e4m3 scales, a per-expert
/// amax and one outer-scale denominator.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Experts<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub amax: &'a [f32],
    pub denominator: f32,
    pub n: usize,
    pub k: usize,
}

/// MXFP8 expert weights as stored: `[E, N, K]` e4m3 bytes and `[E, N, K / 32]` ue8m0 exponents.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Experts<'a> {
    pub values: &'a [u8],
    pub exponents: &'a [u8],
    pub n: usize,
    pub k: usize,
}

/// The fp32 accumulators `[N]` of one quantized row against one NVFP4 expert (before the row
/// scale).
pub fn w4a4_accumulate(row: &Nvfp4Row, w: &Nvfp4Experts<'_>, expert: usize) -> Vec<f32> {
    let a = nvfp4_dequant(&row.codes, &row.scales);
    assert_eq!(a.len(), w.k);
    (0..w.n)
        .map(|n| {
            let r = expert * w.n + n;
            let wr = nvfp4_dequant(
                &w.codes[r * w.k / 2..(r + 1) * w.k / 2],
                &w.scales[r * w.k / 16..(r + 1) * w.k / 16],
            );
            dot_index_order(&a, &wr)
        })
        .collect()
}

/// [`w4a4_accumulate`] with the sm_120 tensor-core chain ([`dot_chain_rz`], K step 64) in place of
/// the index-order sum.
pub fn w4a4_accumulate_chain(row: &Nvfp4Row, w: &Nvfp4Experts<'_>, expert: usize) -> Vec<f32> {
    let a = nvfp4_dequant(&row.codes, &row.scales);
    assert_eq!(a.len(), w.k);
    (0..w.n)
        .map(|n| {
            let r = expert * w.n + n;
            let wr = nvfp4_dequant(
                &w.codes[r * w.k / 2..(r + 1) * w.k / 2],
                &w.scales[r * w.k / 16..(r + 1) * w.k / 16],
            );
            dot_chain_rz(&a, &wr, 64)
        })
        .collect()
}

/// How an expert GEMM's fp32 accumulator is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accumulation {
    /// Products then additions in ascending index order (the goldens' software model).
    IndexOrder,
    /// What the sm_120a kernels compute: [`dot_chain_rz`] (W4A4) or [`dot_chain_trunc`] (W8A8).
    Sm120,
}

/// One quantized row through one NVFP4 expert: accumulate, expert scale 1.0, row scale, bf16.
pub fn w4a4_gemm_row(row: &Nvfp4Row, w: &Nvfp4Experts<'_>, expert: usize) -> Vec<f32> {
    w4a4_gemm_row_with(row, w, expert, Accumulation::IndexOrder)
}

/// [`w4a4_gemm_row`] with the accumulator of [`Accumulation`].
pub fn w4a4_gemm_row_with(
    row: &Nvfp4Row,
    w: &Nvfp4Experts<'_>,
    expert: usize,
    accumulation: Accumulation,
) -> Vec<f32> {
    let scale = nvfp4_row_scale(row.amax, w.amax[expert], w.denominator);
    let acc = match accumulation {
        Accumulation::IndexOrder => w4a4_accumulate(row, w, expert),
        Accumulation::Sm120 => w4a4_accumulate_chain(row, w, expert),
    };
    acc.into_iter()
        .map(|acc| bf16((acc * 1.0f32) * scale))
        .collect()
}

/// The fp32 accumulators `[N]` of one MXFP8 row against one MXFP8 expert.
pub fn w8a8_accumulate(
    values: &[u8],
    exponents: &[u8],
    w: &Mxfp8Experts<'_>,
    expert: usize,
) -> Vec<f32> {
    w8a8_accumulate_with(values, exponents, w, expert, Accumulation::IndexOrder)
}

/// [`w8a8_accumulate`] with the accumulator of [`Accumulation`].
pub fn w8a8_accumulate_with(
    values: &[u8],
    exponents: &[u8],
    w: &Mxfp8Experts<'_>,
    expert: usize,
    accumulation: Accumulation,
) -> Vec<f32> {
    let a = mxfp8_dequant(values, exponents);
    let a_exp = mxfp8_operand_exponents(values, exponents);
    assert_eq!(a.len(), w.k);
    (0..w.n)
        .map(|n| {
            let r = expert * w.n + n;
            let (wv, we) = (
                &w.values[r * w.k..(r + 1) * w.k],
                &w.exponents[r * w.k / 32..(r + 1) * w.k / 32],
            );
            let wr = mxfp8_dequant(wv, we);
            match accumulation {
                Accumulation::IndexOrder => dot_index_order(&a, &wr),
                Accumulation::Sm120 => {
                    dot_chain_trunc(&a, &a_exp, &wr, &mxfp8_operand_exponents(wv, we), 32, 25)
                }
            }
        })
        .collect()
}

/// One MXFP8 row through one MXFP8 expert: accumulate, bf16.
pub fn w8a8_gemm_row(
    values: &[u8],
    exponents: &[u8],
    w: &Mxfp8Experts<'_>,
    expert: usize,
) -> Vec<f32> {
    w8a8_gemm_row_with(values, exponents, w, expert, Accumulation::IndexOrder)
}

/// [`w8a8_gemm_row`] with the accumulator of [`Accumulation`].
pub fn w8a8_gemm_row_with(
    values: &[u8],
    exponents: &[u8],
    w: &Mxfp8Experts<'_>,
    expert: usize,
    accumulation: Accumulation,
) -> Vec<f32> {
    w8a8_accumulate_with(values, exponents, w, expert, accumulation)
        .into_iter()
        .map(bf16)
        .collect()
}

/// The gate factor of SiTU as a real function of the bf16 gate, evaluated in f64:
/// `cap * tanh(g / cap) * sigmoid(g)`, `sigmoid(g) = 1 / (1 + exp(-g))`.
pub fn situ_gate_factor(gate: f32, cap: f32) -> f64 {
    let (g, c) = (f64::from(gate), f64::from(cap));
    c * (g / c).tanh() * (1.0 / (1.0 + (-g).exp()))
}

/// The up factor of SiTU as a real function of the bf16 up value, in f64: `cap * tanh(u / cap)`.
pub fn situ_up_factor(up: f32, cap: f32) -> f64 {
    let (u, c) = (f64::from(up), f64::from(cap));
    c * (u / c).tanh()
}

/// SiTU's two one-input factors for every bf16 bit pattern (65,536 entries each, f64), indexed by
/// the bf16 bits of the gate and of the up value. The kernels read the same tables.
#[derive(Debug, Clone, PartialEq)]
pub struct SituTables {
    pub gate_cap: f32,
    pub up_cap: f32,
    pub gate: Vec<f64>,
    pub up: Vec<f64>,
}

/// Builds [`SituTables`] for the given caps (the checkpoint's are 4 and 25).
pub fn situ_tables(gate_cap: f32, up_cap: f32) -> SituTables {
    let value = |bits: u32| f32::from_bits(bits << 16);
    SituTables {
        gate_cap,
        up_cap,
        gate: (0..=0xffffu32)
            .map(|b| situ_gate_factor(value(b), gate_cap))
            .collect(),
        up: (0..=0xffffu32)
            .map(|b| situ_up_factor(value(b), up_cap))
            .collect(),
    }
}

/// f64 -> bf16 (held as fp32), rounded once to nearest even: the f64 value is first rounded to
/// fp32 by round-to-odd (toward zero, the last bit set when inexact), which keeps the one bf16
/// rounding exact. NaN gives the quiet NaN `0x7fc0`.
pub fn bf16_from_f64(x: f64) -> f32 {
    if x.is_nan() {
        return f32::from_bits(0x7fc0_0000);
    }
    let r = x as f32;
    if r.is_infinite() {
        return r;
    }
    let mut b = r.to_bits();
    if f64::from(r) != x {
        if f64::from(r).abs() > x.abs() {
            b -= 1;
        }
        b |= 1;
    }
    bf16(f32::from_bits(b))
}

/// One SiTU output: `bf16(gate_factor(g) * up_factor(u) * gamma)`, the products in f64 (in this
/// order) and the only rounding the one to bf16. `g` and `u` are bf16 values held as fp32.
#[inline]
pub fn situ_value(gate: f32, up: f32, gamma: f32, t: &SituTables) -> f32 {
    let (g, u) = (
        (gate.to_bits() >> 16) as usize,
        (up.to_bits() >> 16) as usize,
    );
    bf16_from_f64((t.gate[g] * t.up[u]) * f64::from(gamma))
}

/// Softcapped SwiGLU with the routing weight folded in, over interleaved `(gate, up)` columns
/// (column `2j` is gate `j`, `2j + 1` is up `j`): `bf16(cap_g tanh(g / cap_g) sigmoid(g) *
/// cap_u tanh(u / cap_u) * gamma)`, the mathematical value rounded once to bf16 (see
/// [`situ_value`]). The routing weight is multiplied in here, before the down projection and
/// before its quantization. The serving image's fp32 sequence of approximate instructions is a
/// different, less accurate program and is not reproduced.
pub fn situ_row(gate_up: &[f32], gamma: f32, tables: &SituTables) -> Vec<f32> {
    gate_up
        .chunks_exact(2)
        .map(|pair| situ_value(pair[0], pair[1], gamma, tables))
        .collect()
}

/// What one token's routed experts produce.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBlockOut {
    /// bf16 gate/up rows, `[top_k][2F]` interleaved.
    pub gate_up: Vec<Vec<f32>>,
    /// bf16 SiTU rows with the routing weight, `[top_k][F]`.
    pub y: Vec<Vec<f32>>,
    /// bf16 down rows, `[top_k][H]`.
    pub down: Vec<Vec<f32>>,
    /// The token's output: the down rows summed in fp32 in slot order, rounded once to bf16.
    pub out: Vec<f32>,
}

fn combine(down: &[Vec<f32>]) -> Vec<f32> {
    let mut acc = vec![0f32; down[0].len()];
    for row in down {
        for (a, v) in acc.iter_mut().zip(row) {
            *a += v;
        }
    }
    acc.into_iter().map(bf16).collect()
}

/// One token through its routed NVFP4 experts: `x` is quantized once, each selected expert runs
/// the interleaved gate/up GEMM, SiTU times its routing weight, a per-routed-row quantization and
/// the down GEMM.
pub fn expert_block_w4a4(
    x: &[f32],
    selected: &[usize],
    gamma: &[f32],
    w13: &Nvfp4Experts<'_>,
    w2: &Nvfp4Experts<'_>,
    tables: &SituTables,
    accumulation: Accumulation,
) -> ExpertBlockOut {
    let xq = nvfp4_act_quant(x);
    let (mut gate_up, mut ys, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for (&expert, &g) in selected.iter().zip(gamma) {
        let gu = w4a4_gemm_row_with(&xq, w13, expert, accumulation);
        let y = situ_row(&gu, g, tables);
        down.push(w4a4_gemm_row_with(
            &nvfp4_act_quant(&y),
            w2,
            expert,
            accumulation,
        ));
        gate_up.push(gu);
        ys.push(y);
    }
    let out = combine(&down);
    ExpertBlockOut {
        gate_up,
        y: ys,
        down,
        out,
    }
}

/// One token through its routed MXFP8 experts (the W8A8 twin of [`expert_block_w4a4`]).
pub fn expert_block_w8a8(
    x: &[f32],
    selected: &[usize],
    gamma: &[f32],
    w13: &Mxfp8Experts<'_>,
    w2: &Mxfp8Experts<'_>,
    tables: &SituTables,
    accumulation: Accumulation,
) -> ExpertBlockOut {
    let (xv, xe) = mxfp8_act_quant(x);
    let (mut gate_up, mut ys, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for (&expert, &g) in selected.iter().zip(gamma) {
        let gu = w8a8_gemm_row_with(&xv, &xe, w13, expert, accumulation);
        let y = situ_row(&gu, g, tables);
        let (yv, ye) = mxfp8_act_quant(&y);
        down.push(w8a8_gemm_row_with(&yv, &ye, w2, expert, accumulation));
        gate_up.push(gu);
        ys.push(y);
    }
    let out = combine(&down);
    ExpertBlockOut {
        gate_up,
        y: ys,
        down,
        out,
    }
}

/// The golden-vector container (`INK2AQ01`): named records of tagged byte blobs.
pub mod goldens {
    use std::collections::BTreeMap;

    /// The committed goldens.
    pub const BYTES: &[u8] = include_bytes!("fixtures/inkling2-actquant-goldens.bin");

    /// What an RTX 5090 Laptop (sm_120a) computed for the golden GEMM records: per record the fp32
    /// accumulators (`acc`) and the bf16 output (`out`) of `ink2-actq-gate --capture`.
    pub const SM120_CAPTURE: &[u8] = include_bytes!("fixtures/inkling2-actquant-sm120-capture.bin");

    /// One record: tag -> bytes.
    pub type Record = BTreeMap<String, Vec<u8>>;

    /// Parses a container into its records.
    pub fn parse(bytes: &[u8]) -> Result<BTreeMap<String, Record>, String> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8], String> {
            let s = bytes
                .get(at..at + n)
                .ok_or_else(|| format!("goldens truncated at {at}"))?;
            at += n;
            Ok(s)
        };
        if take(8)? != b"INK2AQ01" {
            return Err("not an INK2AQ01 container".into());
        }
        let u32le = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
        let u16le = |s: &[u8]| u16::from_le_bytes([s[0], s[1]]) as usize;
        let records = u32le(take(4)?);
        let mut out = BTreeMap::new();
        for _ in 0..records {
            let n = u16le(take(2)?);
            let name = String::from_utf8(take(n)?.to_vec()).map_err(|e| e.to_string())?;
            let blobs = u32le(take(4)?);
            let mut record = Record::new();
            for _ in 0..blobs {
                let n = u16le(take(2)?);
                let tag = String::from_utf8(take(n)?.to_vec()).map_err(|e| e.to_string())?;
                let n = u32le(take(4)?);
                record.insert(tag, take(n)?.to_vec());
            }
            out.insert(name, record);
        }
        Ok(out)
    }

    /// Serializes records into a container.
    pub fn write(records: &BTreeMap<String, Record>) -> Vec<u8> {
        let mut out = b"INK2AQ01".to_vec();
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        for (name, record) in records {
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(record.len() as u32).to_le_bytes());
            for (tag, data) in record {
                out.extend_from_slice(&(tag.len() as u16).to_le_bytes());
                out.extend_from_slice(tag.as_bytes());
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(data);
            }
        }
        out
    }

    /// Little-endian u16 blob.
    pub fn u16s(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
    /// Little-endian u32 blob.
    pub fn u32s(b: &[u8]) -> Vec<u32> {
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    /// Little-endian fp32 blob.
    pub fn f32s(b: &[u8]) -> Vec<f32> {
        u32s(b).into_iter().map(f32::from_bits).collect()
    }
    /// Little-endian f64 blob.
    pub fn f64s(b: &[u8]) -> Vec<f64> {
        b.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
            .collect()
    }
    /// bf16 bit patterns widened to fp32.
    pub fn bf16s(b: &[u8]) -> Vec<f32> {
        u16s(b)
            .into_iter()
            .map(|v| f32::from_bits(u32::from(v) << 16))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::goldens::{bf16s, f32s, parse, u32s};
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn e2m1_and_e4m3_rounding_match_the_goldens() {
        let g = parse(goldens::BYTES).unwrap();
        let r = &g["e2m1/grid"];
        let x = f32s(&r["x"]);
        assert!(x.len() >= 100);
        for (v, code) in x.iter().zip(&r["code"]) {
            assert_eq!(e2m1_encode(*v), *code, "e2m1({v:e})");
        }
        for name in ["e4m3/grid", "e4m3/satfinite"] {
            let r = &g[name];
            let x = f32s(&r["x"]);
            for (v, byte) in x.iter().zip(&r["byte"]) {
                assert_eq!(e4m3_encode(*v), *byte, "{name}: e4m3({v:e})");
            }
        }
        for byte in 0..=0x7eu8 {
            assert_eq!(e4m3_encode(e4m3_decode(byte)), byte);
            assert_eq!(e4m3_encode(e4m3_decode(byte | 0x80)), byte | 0x80);
        }
    }

    #[test]
    fn nvfp4_act_quant_is_bit_identical_to_the_goldens() {
        let g = parse(goldens::BYTES).unwrap();
        let mut rows = 0;
        for (name, r) in g.iter().filter(|(n, _)| n.starts_with("nvfp4q/")) {
            let d = u32s(&r["dims"]);
            let (m, k) = (d[0] as usize, d[1] as usize);
            let x = bf16s(&r["x"]);
            let amax = f32s(&r["amax"]);
            for i in 0..m {
                let q = nvfp4_act_quant(&x[i * k..(i + 1) * k]);
                assert_eq!(q.amax.to_bits(), amax[i].to_bits(), "{name} row {i} amax");
                assert_eq!(
                    q.scales,
                    r["scales"][i * k / 16..(i + 1) * k / 16],
                    "{name} row {i} scales"
                );
                assert_eq!(
                    q.codes,
                    r["codes"][i * k / 2..(i + 1) * k / 2],
                    "{name} row {i} codes"
                );
                rows += 1;
            }
        }
        assert!(rows >= 100, "{rows} rows");
    }

    #[test]
    fn mxfp8_act_quant_is_bit_identical_to_the_goldens() {
        let g = parse(goldens::BYTES).unwrap();
        let mut rows = 0;
        for (name, r) in g.iter().filter(|(n, _)| n.starts_with("mxfp8q/")) {
            let d = u32s(&r["dims"]);
            let (m, k) = (d[0] as usize, d[1] as usize);
            let x = bf16s(&r["x"]);
            for i in 0..m {
                let (values, exps) = mxfp8_act_quant(&x[i * k..(i + 1) * k]);
                assert_eq!(
                    exps,
                    r["exps"][i * k / 32..(i + 1) * k / 32],
                    "{name} row {i} exps"
                );
                assert_eq!(
                    values,
                    r["values"][i * k..(i + 1) * k],
                    "{name} row {i} values"
                );
                rows += 1;
            }
        }
        assert!(rows >= 40, "{rows} rows");
    }

    #[test]
    fn expert_gemms_match_the_index_order_goldens() {
        let g = parse(goldens::BYTES).unwrap();
        for (name, r) in g.iter().filter(|(n, _)| n.starts_with("w4a4/")) {
            let d = u32s(&r["dims"]);
            let (n, k, rows) = (d[1] as usize, d[2] as usize, d[3] as usize);
            let wamax = f32s(&r["wamax"]);
            let w = Nvfp4Experts {
                codes: &r["wcodes"],
                scales: &r["wscales"],
                amax: &wamax,
                denominator: f32s(&r["denom"])[0],
                n,
                k,
            };
            let x = bf16s(&r["x"]);
            let expert = u32s(&r["expert"]);
            let (acc, rowscale, out) = (f32s(&r["acc"]), f32s(&r["rowscale"]), bf16s(&r["out"]));
            for i in 0..rows {
                let q = nvfp4_act_quant(&x[i * k..(i + 1) * k]);
                let e = expert[i] as usize;
                assert_eq!(
                    bits(&w4a4_accumulate(&q, &w, e)),
                    bits(&acc[i * n..(i + 1) * n]),
                    "{name} row {i} acc"
                );
                assert_eq!(
                    nvfp4_row_scale(q.amax, wamax[e], w.denominator).to_bits(),
                    rowscale[i].to_bits()
                );
                assert_eq!(
                    bits(&w4a4_gemm_row(&q, &w, e)),
                    bits(&out[i * n..(i + 1) * n]),
                    "{name} row {i} out"
                );
            }
        }
        for (name, r) in g.iter().filter(|(n, _)| n.starts_with("w8a8/")) {
            let d = u32s(&r["dims"]);
            let (n, k, rows) = (d[1] as usize, d[2] as usize, d[3] as usize);
            let w = Mxfp8Experts {
                values: &r["w"],
                exponents: &r["wexps"],
                n,
                k,
            };
            let x = bf16s(&r["x"]);
            let expert = u32s(&r["expert"]);
            let (acc, out) = (f32s(&r["acc"]), bf16s(&r["out"]));
            for i in 0..rows {
                let (v, e) = mxfp8_act_quant(&x[i * k..(i + 1) * k]);
                let ex = expert[i] as usize;
                assert_eq!(
                    bits(&w8a8_accumulate(&v, &e, &w, ex)),
                    bits(&acc[i * n..(i + 1) * n]),
                    "{name} row {i} acc"
                );
                assert_eq!(
                    bits(&w8a8_gemm_row(&v, &e, &w, ex)),
                    bits(&out[i * n..(i + 1) * n]),
                    "{name} row {i} out"
                );
            }
        }
    }

    #[test]
    fn expert_block_matches_the_chain_goldens() {
        let g = parse(goldens::BYTES).unwrap();
        let tables = situ_tables(4.0, 25.0);
        for kind in ["nvfp4", "mxfp8"] {
            let r = &g[&format!("chain/{kind}")];
            let d = u32s(&r["dims"]);
            let (h, f, tokens, topk) = (d[1] as usize, d[2] as usize, d[3] as usize, d[4] as usize);
            let x = bf16s(&r["x"]);
            let sel = u32s(&r["sel"]);
            let gamma = f32s(&r["gamma"]);
            let (gate_up, y, down, out) = (
                bf16s(&r["gate_up"]),
                bf16s(&r["y"]),
                bf16s(&r["down"]),
                bf16s(&r["out"]),
            );
            for t in 0..tokens {
                let s: Vec<usize> = sel[t * topk..(t + 1) * topk]
                    .iter()
                    .map(|v| *v as usize)
                    .collect();
                let gm = &gamma[t * topk..(t + 1) * topk];
                let xt = &x[t * h..(t + 1) * h];
                let got = if kind == "nvfp4" {
                    let (a13, a2) = (f32s(&r["w13amax"]), f32s(&r["w2amax"]));
                    let w13 = Nvfp4Experts {
                        codes: &r["w13codes"],
                        scales: &r["w13scales"],
                        amax: &a13,
                        denominator: f32s(&r["w13denom"])[0],
                        n: 2 * f,
                        k: h,
                    };
                    let w2 = Nvfp4Experts {
                        codes: &r["w2codes"],
                        scales: &r["w2scales"],
                        amax: &a2,
                        denominator: f32s(&r["w2denom"])[0],
                        n: h,
                        k: f,
                    };
                    expert_block_w4a4(xt, &s, gm, &w13, &w2, &tables, Accumulation::IndexOrder)
                } else {
                    let w13 = Mxfp8Experts {
                        values: &r["w13"],
                        exponents: &r["w13exps"],
                        n: 2 * f,
                        k: h,
                    };
                    let w2 = Mxfp8Experts {
                        values: &r["w2"],
                        exponents: &r["w2exps"],
                        n: h,
                        k: f,
                    };
                    expert_block_w8a8(xt, &s, gm, &w13, &w2, &tables, Accumulation::IndexOrder)
                };
                for slot in 0..topk {
                    let row = t * topk + slot;
                    assert_eq!(
                        bits(&got.gate_up[slot]),
                        bits(&gate_up[row * 2 * f..(row + 1) * 2 * f]),
                        "{kind} gate_up {row}"
                    );
                    assert_eq!(
                        bits(&got.y[slot]),
                        bits(&y[row * f..(row + 1) * f]),
                        "{kind} y {row}"
                    );
                    assert_eq!(
                        bits(&got.down[slot]),
                        bits(&down[row * h..(row + 1) * h]),
                        "{kind} down {row}"
                    );
                }
                assert_eq!(
                    bits(&got.out),
                    bits(&out[t * h..(t + 1) * h]),
                    "{kind} out {t}"
                );
            }
        }
    }

    /// The tensor-core chain models against what the card computed, without a GPU: W4A4 is the
    /// exact step sum added with one rounding toward zero, W8A8 the truncating model at guard 25,
    /// and the bf16 epilogue follows from the captured accumulators.
    #[test]
    fn chain_models_equal_the_sm120_capture() {
        let g = parse(goldens::BYTES).unwrap();
        let cap = parse(goldens::SM120_CAPTURE).unwrap();
        let mut elements = 0;
        for (name, c) in &cap {
            let r = &g[name];
            let d = u32s(&r["dims"]);
            let (n, k, rows) = (d[1] as usize, d[2] as usize, d[3] as usize);
            let x = bf16s(&r["x"]);
            let expert = u32s(&r["expert"]);
            let (acc, out) = (f32s(&c["acc"]), bf16s(&c["out"]));
            let index_order = f32s(&r["acc"]);
            let mut differs_from_index_order = 0;
            for i in 0..rows {
                let e = expert[i] as usize;
                let xr = &x[i * k..(i + 1) * k];
                let (a, a_exp, scale) = if name.starts_with("w4a4/") {
                    let q = nvfp4_act_quant(xr);
                    let scale = nvfp4_row_scale(q.amax, f32s(&r["wamax"])[e], f32s(&r["denom"])[0]);
                    (nvfp4_dequant(&q.codes, &q.scales), Vec::new(), Some(scale))
                } else {
                    let (v, s) = mxfp8_act_quant(xr);
                    (mxfp8_dequant(&v, &s), mxfp8_operand_exponents(&v, &s), None)
                };
                for col in 0..n {
                    let row = e * n + col;
                    let (w, model) = if name.starts_with("w4a4/") {
                        let w = nvfp4_dequant(
                            &r["wcodes"][row * k / 2..(row + 1) * k / 2],
                            &r["wscales"][row * k / 16..(row + 1) * k / 16],
                        );
                        let model = dot_chain_rz(&a, &w, 64);
                        (w, model)
                    } else {
                        let (wv, we) = (
                            &r["w"][row * k..(row + 1) * k],
                            &r["wexps"][row * k / 32..(row + 1) * k / 32],
                        );
                        let w = mxfp8_dequant(wv, we);
                        let w_exp = mxfp8_operand_exponents(wv, we);
                        let model = dot_chain_trunc(&a, &a_exp, &w, &w_exp, 32, 25);
                        (w, model)
                    };
                    let got = acc[i * n + col];
                    assert_eq!(
                        model.to_bits(),
                        got.to_bits(),
                        "{name} row {i} col {col}: {model:e} vs device {got:e}"
                    );
                    let want = match scale {
                        Some(s) => bf16((got * 1.0f32) * s),
                        None => bf16(got),
                    };
                    assert_eq!(
                        want.to_bits(),
                        out[i * n + col].to_bits(),
                        "{name} row {i} col {col} epilogue"
                    );
                    differs_from_index_order +=
                        usize::from(index_order[i * n + col].to_bits() != got.to_bits());
                    let _ = w;
                    elements += 1;
                }
            }
            // the index-order sum is not the device's program: the capture has to show it
            assert!(
                differs_from_index_order > 0,
                "{name}: the capture equals the index-order sum everywhere"
            );
        }
        assert!(elements >= 1000, "{elements} elements");
    }

    #[test]
    fn a_zero_row_has_no_amax_epsilon() {
        let q = nvfp4_act_quant(&[0.0; 32]);
        assert_eq!(q.amax, 0.0);
        assert!(q.scales.iter().all(|s| *s == 0) && q.codes.iter().all(|c| *c == 0));
        let q = nvfp4_act_quant(&[-0.0; 32]);
        assert!(q.codes.iter().all(|c| *c == 0x88), "-0 keeps its sign bit");
    }

    /// The one rounding of SiTU: round-to-odd to fp32 then nearest-even to bf16 equals the
    /// nearest bf16 of the f64 value (ties to even), on values that sit on, beside and between
    /// bf16 midpoints, across the normal and subnormal ranges.
    #[test]
    fn bf16_from_f64_rounds_once() {
        let nearest = |x: f64| -> f32 {
            // the two bf16 neighbours of x by search over bit patterns
            let lo = {
                let mut b = bf16(x as f32).to_bits() >> 16;
                let v = |b: u32| f64::from(f32::from_bits(b << 16));
                while v(b) > x {
                    b = if x > 0.0 { b - 1 } else { b + 1 };
                }
                while v(if x >= 0.0 { b + 1 } else { b.wrapping_sub(1) }) <= x && b != 0x8000 {
                    b = if x >= 0.0 { b + 1 } else { b - 1 };
                }
                b
            };
            let v = |b: u32| f64::from(f32::from_bits(b << 16));
            let hi = if x >= 0.0 { lo + 1 } else { lo - 1 };
            let (dl, dh) = ((x - v(lo)).abs(), (v(hi) - x).abs());
            let pick = if dl < dh || (dl == dh && lo & 1 == 0) {
                lo
            } else {
                hi
            };
            f32::from_bits(pick << 16)
        };
        let mut checked = 0;
        for bits in (0x0001u32..0x7f00).step_by(7) {
            for sign in [1.0f64, -1.0] {
                let a = f64::from(f32::from_bits(bits << 16));
                let b = f64::from(f32::from_bits((bits + 1) << 16));
                let mid = (a + b) / 2.0;
                for x in [
                    mid,
                    mid * (1.0 + 2f64.powi(-40)),
                    mid * (1.0 - 2f64.powi(-40)),
                    a * (1.0 + 2f64.powi(-30)),
                ] {
                    let x = x * sign;
                    let got = bf16_from_f64(x);
                    assert_eq!(got.to_bits(), nearest(x).to_bits(), "{x:e}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000);
        assert_eq!(bf16_from_f64(f64::NAN).to_bits(), 0x7fc0_0000);
        assert_eq!(bf16_from_f64(-0.0).to_bits(), 0x8000_0000);
    }

    /// SiTU is the mathematical function rounded once: no fp32 rounding of a factor, and the
    /// tables cover every bf16 input including the non-finite ones.
    #[test]
    fn situ_is_rounded_once() {
        let t = situ_tables(4.0, 25.0);
        assert_eq!(t.gate.len(), 65536);
        let pos_inf = 0x7f80usize;
        assert_eq!(t.gate[pos_inf], 4.0);
        assert_eq!(t.up[pos_inf], 25.0);
        assert!(t.gate[0xff80] == 0.0 && t.gate[0xff80].is_sign_negative());
        // a case where rounding the factors to fp32 first moves the bf16 result: the program
        // keeps them in f64
        let mut moved = 0;
        for gb in (0x3c00u32..0x4100).step_by(3) {
            for ub in (0x3c00u32..0x4100).step_by(5) {
                let (g, u) = (f32::from_bits(gb << 16), f32::from_bits(ub << 16));
                let once = situ_value(g, u, 1.0, &t);
                let staged = bf16((t.gate[gb as usize] as f32) * (t.up[ub as usize] as f32));
                moved += usize::from(once.to_bits() != staged.to_bits());
                let exact = situ_gate_factor(g, 4.0) * situ_up_factor(u, 25.0);
                assert_eq!(once.to_bits(), bf16_from_f64(exact).to_bits());
            }
        }
        assert!(
            moved > 0,
            "the staged fp32 rounding never differed: the sample is too narrow"
        );
    }

    /// dot_chain_trunc takes the e4m3 field exponent: a subnormal code counts as -6. Step 1 leaves an
    /// accumulator with its last bit set at exponent e; step 2 subtracts 2^e as 0x08 under a block
    /// scale times the negative smallest subnormal (-2^-9): field exponent sum e + 3, normalised e.
    /// The hardware's quantum 2^(e - 22) drops the accumulator's last bit; the normalised reading's
    /// 2^(e - 25) keeps it.
    #[test]
    fn chain_trunc_reads_the_field_exponent() {
        let k = 64;
        let mut found = false;
        for seed in 1u32..5000 {
            let mut x = seed.wrapping_mul(2654435761);
            let mut next = || {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x
            };
            let mut av: Vec<u8> = (0..32).map(|_| 1 + (next() % 0x7d) as u8).collect();
            let mut wv: Vec<u8> = (0..32).map(|_| 1 + (next() % 0x7d) as u8).collect();
            let mut e = vec![127u8; k / 32];
            let (a1, w1) = (mxfp8_dequant(&av, &e[..1]), mxfp8_dequant(&wv, &e[..1]));
            let (ae1, we1) = (
                mxfp8_operand_exponents(&av, &e[..1]),
                mxfp8_operand_exponents(&wv, &e[..1]),
            );
            let step1 = dot_chain_trunc(&a1, &ae1, &w1, &we1, 32, 25);
            let e_acc = ((step1.to_bits() >> 23) & 0xff) as i32 - 127;
            // step 2: the code 0x08 (field exponent -6) under the block exponent that puts the field
            // sum with the subnormal 2^-9 at e_acc + 3; the normalised sum is e_acc
            let block = e_acc + 15 + 127;
            if step1.to_bits() & 1 == 0 || !(1..=254).contains(&block) {
                continue;
            }
            e[1] = block as u8;
            av.push(0x08);
            av.extend([0u8; 31]);
            wv.push(0x81); // -2^-9: the result falls a binade, so the dropped bit stays visible
            wv.extend([0u8; 31]);
            let ew = vec![127u8; k / 32];
            let (a, w) = (mxfp8_dequant(&av, &e), mxfp8_dequant(&wv, &ew));
            let (a_exp, w_exp) = (
                mxfp8_operand_exponents(&av, &e),
                mxfp8_operand_exponents(&wv, &ew),
            );
            // the weights' exponents read from the normalised value instead
            let normalised: Vec<i32> = w
                .iter()
                .zip(&w_exp)
                .map(|(v, e)| {
                    if *v == 0.0 {
                        *e
                    } else {
                        ((f64::from(*v).to_bits() >> 52) & 0x7ff) as i32 - 1023
                    }
                })
                .collect();
            assert_eq!(w_exp[32], -6);
            assert_eq!(normalised[32], -9);
            assert_eq!(a_exp[32] + w_exp[32], e_acc + 3);
            let hw = dot_chain_trunc(&a, &a_exp, &w, &w_exp, 32, 25);
            let other = dot_chain_trunc(&a, &a_exp, &w, &normalised, 32, 25);
            let exact: f64 = a
                .iter()
                .zip(&w)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            assert!((f64::from(hw) - exact).abs() <= sum_magnitude(&a, &w) * 64.0 * 2f64.powi(-23));
            assert_ne!(
                hw.to_bits(),
                other.to_bits(),
                "seed {seed}: the two exponent readings agree"
            );
            found = true;
            break;
        }
        assert!(found, "no seed built the case");
    }

    fn sum_magnitude(a: &[f32], w: &[f32]) -> f64 {
        a.iter()
            .zip(w)
            .map(|(x, y)| (f64::from(*x) * f64::from(*y)).abs())
            .sum()
    }
}
