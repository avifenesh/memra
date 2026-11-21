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
//! - the GEMM accumulates in fp32 in ascending index order. The tensor core's addition order is
//!   hardware defined, so a kernel is compared with this under a per-element bound.
//! - SiTU evaluates `exp2` and `tanh` in f64 and rounds once to fp32.

/// Largest finite fp32, the clamp of every reciprocal in the NVFP4 quantizer.
pub const FP32_MAX: f32 = f32::MAX;

const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// fp32 -> bf16 -> fp32, round to nearest even.
#[inline]
pub fn bf16(x: f32) -> f32 {
    memra_gguf::dsv41_forward::bf16(x)
}

/// fp32 -> e2m1 nibble: round to nearest, ties to the even code, saturating at 6; bit 3 is the
/// fp32 sign bit, so -0 and negative values that round to zero encode as 0x8.
#[inline]
pub fn e2m1_encode(x: f32) -> u8 {
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

/// e4m3 byte -> fp32 (`float8_e4m3fn`; the NaN codes never occur in this program).
#[inline]
pub fn e4m3_decode(byte: u8) -> f32 {
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
/// bit carried from the fp32 sign bit (`cvt.rn.satfinite`). NaN is not part of this program.
#[inline]
pub fn e4m3_encode(x: f32) -> u8 {
    let sign = ((x.to_bits() >> 31) as u8) << 7;
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
    assert!(
        x.len().is_multiple_of(16),
        "NVFP4 rows are whole blocks of 16"
    );
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let encoder = (1536.0f32 / amax).min(FP32_MAX);
    let decoder = 1.0f32 / encoder;
    let outer = amax * (1.0f32 / 1536.0f32);
    let enc6 = encoder * (1.0f32 / 6.0f32);
    let enc4 = encoder * 0.25f32;
    let mut codes = vec![0u8; x.len() / 2];
    let mut scales = vec![0u8; x.len() / 16];
    for (b, block) in x.chunks_exact(16).enumerate() {
        let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
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
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let raw = amax * (1.0f32 / 448.0f32);
        let rounded = ((raw.to_bits() as i32).wrapping_add(0x007f_ffff) & 0x7f80_0000) >> 23;
        let exponent = if raw <= min_scale {
            0
        } else {
            rounded.min(254)
        };
        let scale = f32::from_bits((exponent as u32) << 23).max(min_scale);
        let inverse = if exponent == 0 {
            2f32.powi(127)
        } else {
            1.0f32 / scale
        };
        for v in block {
            values.push(e4m3_encode((v * inverse).clamp(-448.0, 448.0)));
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

/// One quantized row through one NVFP4 expert: accumulate, expert scale 1.0, row scale, bf16.
pub fn w4a4_gemm_row(row: &Nvfp4Row, w: &Nvfp4Experts<'_>, expert: usize) -> Vec<f32> {
    let scale = nvfp4_row_scale(row.amax, w.amax[expert], w.denominator);
    w4a4_accumulate(row, w, expert)
        .into_iter()
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
    let a = mxfp8_dequant(values, exponents);
    assert_eq!(a.len(), w.k);
    (0..w.n)
        .map(|n| {
            let r = expert * w.n + n;
            let wr = mxfp8_dequant(
                &w.values[r * w.k..(r + 1) * w.k],
                &w.exponents[r * w.k / 32..(r + 1) * w.k / 32],
            );
            dot_index_order(&a, &wr)
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
    w8a8_accumulate(values, exponents, w, expert)
        .into_iter()
        .map(bf16)
        .collect()
}

/// `cap * tanh(x / cap)`, tanh in f64 rounded once.
#[inline]
fn softcap(x: f32, cap: f32) -> f32 {
    cap * (f64::from(x / cap).tanh() as f32)
}

/// Softcapped SwiGLU with the routing weight folded in, rounded once to bf16:
/// `bf16(((softcap(gate) * sigmoid(gate)) * softcap(up)) * gamma)` over interleaved `(gate, up)`
/// columns (column `2j` is gate `j`, `2j + 1` is up `j`). The routing weight is multiplied in here,
/// before the down projection and before its quantization.
pub fn situ_row(gate_up: &[f32], gamma: f32, gate_cap: f32, up_cap: f32) -> Vec<f32> {
    gate_up
        .chunks_exact(2)
        .map(|pair| {
            let (gate, up) = (pair[0], pair[1]);
            let t = gate * f32::from_bits(0xbfb8_aa3b); // -log2(e)
            let sigmoid = 1.0f32 / (((f64::from(t)).exp2() as f32) + 1.0f32);
            bf16(((softcap(gate, gate_cap) * sigmoid) * softcap(up, up_cap)) * gamma)
        })
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
    caps: (f32, f32),
) -> ExpertBlockOut {
    let xq = nvfp4_act_quant(x);
    let (mut gate_up, mut ys, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for (&expert, &g) in selected.iter().zip(gamma) {
        let gu = w4a4_gemm_row(&xq, w13, expert);
        let y = situ_row(&gu, g, caps.0, caps.1);
        down.push(w4a4_gemm_row(&nvfp4_act_quant(&y), w2, expert));
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
    caps: (f32, f32),
) -> ExpertBlockOut {
    let (xv, xe) = mxfp8_act_quant(x);
    let (mut gate_up, mut ys, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for (&expert, &g) in selected.iter().zip(gamma) {
        let gu = w8a8_gemm_row(&xv, &xe, w13, expert);
        let y = situ_row(&gu, g, caps.0, caps.1);
        let (yv, ye) = mxfp8_act_quant(&y);
        down.push(w8a8_gemm_row(&yv, &ye, w2, expert));
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
                    expert_block_w4a4(xt, &s, gm, &w13, &w2, (4.0, 25.0))
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
                    expert_block_w8a8(xt, &s, gm, &w13, &w2, (4.0, 25.0))
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

    #[test]
    fn a_zero_row_has_no_amax_epsilon() {
        let q = nvfp4_act_quant(&[0.0; 32]);
        assert_eq!(q.amax, 0.0);
        assert!(q.scales.iter().all(|s| *s == 0) && q.codes.iter().all(|c| *c == 0));
        let q = nvfp4_act_quant(&[-0.0; 32]);
        assert!(q.codes.iter().all(|c| *c == 0x88), "-0 keeps its sign bit");
    }
}
