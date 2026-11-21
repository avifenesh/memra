//! Inkling 2 Mini routed experts on the GPU: the act-quant quantizers and the W4A4 / W8A8 expert
//! GEMMs of `cu/inkling2_actq.cu`, their gates against
//! [`memra_reference::inkling2_actquant`] and its golden vectors, and a microbench.
//!
//! Which layers run this class is the plan's choice, per layer: every entry here takes the numeric
//! kind per call and holds no model state.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_reference::inkling2_actquant as reference;

use crate::Engine;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

unsafe extern "C" {
    fn memra_ink2_actq_built() -> i32;
    fn memra_ink2_actq(
        kind: i32,
        situ: i32,
        x: *const c_void,
        gamma: *const f32,
        rows: *const u32,
        nrows: i32,
        values: *mut c_void,
        scales: *mut c_void,
        amax: *mut f32,
        y: *mut c_void,
        k: i32,
        gate_cap: f32,
        up_cap: f32,
        mode: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_ink2_cvt_probe(
        x: *const f32,
        e4m3: *mut u8,
        e2m1: *mut u8,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_ink2_group_words(nb: i32) -> i32;
    fn memra_ink2_expert_gemm(
        kind: i32,
        nb: i32,
        act: *const c_void,
        act_scales: *const c_void,
        act_amax: *const f32,
        w: *const c_void,
        w_scales: *const c_void,
        w_amax: *const f32,
        denominator: f32,
        groups: *const u32,
        ngroups: i32,
        out: *mut c_void,
        acc: *mut f32,
        n: i32,
        k: i32,
        wpb: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// The two activation-quant formats of the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ink2Quant {
    /// e2m1 values, e4m3 scale per 16, per-row amax, four-over-six selection (W4A4).
    Nvfp4,
    /// e4m3 values, round-up ue8m0 scale per 32 (W8A8).
    Mxfp8,
}

impl Ink2Quant {
    fn kind(self) -> i32 {
        match self {
            Self::Nvfp4 => 0,
            Self::Mxfp8 => 1,
        }
    }
    /// Value bytes of one quantized row of width `k`.
    pub fn value_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 => k / 2,
            Self::Mxfp8 => k,
        }
    }
    /// Scale bytes of one quantized row of width `k`.
    pub fn scale_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 => k / 16,
            Self::Mxfp8 => k / 32,
        }
    }
}

/// Quantized activation rows on the device.
pub struct Ink2Rows {
    pub quant: Ink2Quant,
    pub rows: usize,
    pub k: usize,
    pub values: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// Per-row amax (NVFP4; a single unused element for MXFP8).
    pub amax: CudaSlice<f32>,
}

/// One projection of the experts as stored in the checkpoint, on the device.
pub struct Ink2Weights {
    pub quant: Ink2Quant,
    pub n: usize,
    pub k: usize,
    pub values: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// Per-expert amax (NVFP4; one unused element for MXFP8).
    pub amax: CudaSlice<f32>,
    /// The outer-scale denominator (NVFP4).
    pub denominator: f32,
}

/// Whether this build carries the block-scaled expert GEMMs (the 120a family).
pub fn ink2_actq_built() -> bool {
    unsafe { memra_ink2_actq_built() == 1 }
}

/// Group routed rows by expert into records of `{expert, count, row[8 * nb]}`. Rows of one
/// expert keep their order; an expert with more than `8 * nb` rows takes several groups.
pub fn ink2_groups(expert_of_row: &[u32], nb: usize) -> (Vec<u32>, usize) {
    let cap = 8 * nb;
    let mut by: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (row, e) in expert_of_row.iter().enumerate() {
        by.entry(*e).or_default().push(row as u32);
    }
    let mut out = Vec::new();
    let mut groups = 0;
    for (e, rows) in by {
        for chunk in rows.chunks(cap) {
            out.push(e);
            out.push(chunk.len() as u32);
            out.extend_from_slice(chunk);
            out.extend(std::iter::repeat_n(0u32, cap - chunk.len()));
            groups += 1;
        }
    }
    debug_assert_eq!(
        out.len(),
        groups * unsafe { memra_ink2_group_words(nb as i32) } as usize
    );
    (out, groups)
}

impl Engine {
    /// Allocate the outputs of a quantization of `rows` rows of width `k`.
    pub fn ink2_rows_alloc(&self, quant: Ink2Quant, rows: usize, k: usize) -> Res<Ink2Rows> {
        Ok(Ink2Rows {
            quant,
            rows,
            k,
            values: self.alloc_u8(rows * quant.value_bytes(k))?,
            scales: self.alloc_u8(rows * quant.scale_bytes(k))?,
            amax: self.zeros(rows.max(1))?,
        })
    }

    /// Quantize bf16 rows `x` (`[rows][k]`, two bytes each) into `out`. Asynchronous on the stream.
    pub fn ink2_act_quant_into(&self, x: &CudaSlice<u8>, out: &mut Ink2Rows) -> Res<()> {
        if x.len() != out.rows * out.k * 2 {
            return Err("ink2 act quant: input is not [rows][k] bf16".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let rc = unsafe {
            memra_ink2_actq(
                out.quant.kind(),
                0,
                x.device_ptr(&stream).0 as *const c_void,
                std::ptr::null(),
                std::ptr::null(),
                out.rows as i32,
                out.values.device_ptr_mut(&stream).0 as *mut c_void,
                out.scales.device_ptr_mut(&stream).0 as *mut c_void,
                out.amax.device_ptr_mut(&stream).0 as *mut f32,
                std::ptr::null_mut(),
                out.k as i32,
                0.0,
                0.0,
                0,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_ink2_actq returned {rc}").into());
        }
        Ok(())
    }

    /// The down projection's input: softcapped SwiGLU of interleaved bf16 `gate_up` (`[rows][2k]`)
    /// times each row's routing weight, rounded to bf16 and quantized into `out`. `y` (optional,
    /// `[rows][k]` bf16) receives the bf16 rows. `approx` selects the PTX approx instructions (a
    /// measurement arm; the default is the fp32 library functions).
    #[allow(clippy::too_many_arguments)]
    pub fn ink2_situ_quant_into(
        &self,
        gate_up: &CudaSlice<u8>,
        gamma: &CudaSlice<f32>,
        caps: (f32, f32),
        approx: bool,
        out: &mut Ink2Rows,
        y: Option<&mut CudaSlice<u8>>,
    ) -> Res<()> {
        if gate_up.len() != out.rows * out.k * 4 || gamma.len() < out.rows {
            return Err(
                "ink2 situ quant: input is not [rows][2k] bf16 with a weight per row".into(),
            );
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let y_ptr = match y {
            Some(y) => {
                if y.len() != out.rows * out.k * 2 {
                    return Err("ink2 situ quant: y is not [rows][k] bf16".into());
                }
                y.device_ptr_mut(&stream).0 as *mut c_void
            }
            None => std::ptr::null_mut(),
        };
        let rc = unsafe {
            memra_ink2_actq(
                out.quant.kind(),
                1,
                gate_up.device_ptr(&stream).0 as *const c_void,
                gamma.device_ptr(&stream).0 as *const f32,
                std::ptr::null(),
                out.rows as i32,
                out.values.device_ptr_mut(&stream).0 as *mut c_void,
                out.scales.device_ptr_mut(&stream).0 as *mut c_void,
                out.amax.device_ptr_mut(&stream).0 as *mut f32,
                y_ptr,
                out.k as i32,
                caps.0,
                caps.1,
                i32::from(approx),
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_ink2_actq (situ) returned {rc}").into());
        }
        Ok(())
    }

    /// The hardware fp32 -> e4m3 and fp32 -> e2m1 conversions of `x`.
    pub fn ink2_cvt_probe(&self, x: &[f32]) -> Res<(Vec<u8>, Vec<u8>)> {
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let xd = self.htod(x)?;
        let mut a = self.alloc_u8(x.len())?;
        let mut b = self.alloc_u8(x.len())?;
        let rc = unsafe {
            memra_ink2_cvt_probe(
                xd.device_ptr(&stream).0 as *const f32,
                a.device_ptr_mut(&stream).0 as *mut u8,
                b.device_ptr_mut(&stream).0 as *mut u8,
                x.len() as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_ink2_cvt_probe returned {rc}").into());
        }
        Ok((self.dtoh_u8(&a)?, self.dtoh_u8(&b)?))
    }

    /// The grouped expert GEMM: quantized rows `act` against `w`, bf16 `out` (`[rows][n]`, two bytes
    /// each); `acc` (optional, `[rows][n]`) receives the fp32 accumulators. `groups` is from
    /// [`ink2_groups`] with the same `nb`. `wpb` (warps per block) takes 0 for the default. Asynchronous
    /// on the stream.
    #[allow(clippy::too_many_arguments)]
    pub fn ink2_expert_gemm_into(
        &self,
        w: &Ink2Weights,
        act: &Ink2Rows,
        groups: &CudaSlice<u32>,
        ngroups: usize,
        nb: usize,
        wpb: usize,
        out: &mut CudaSlice<u8>,
        acc: Option<&mut CudaSlice<f32>>,
    ) -> Res<()> {
        if w.quant != act.quant || w.k != act.k || out.len() != act.rows * w.n * 2 {
            return Err("ink2 expert gemm: operand shapes disagree".into());
        }
        let words = unsafe { memra_ink2_group_words(nb as i32) } as usize;
        if groups.len() < ngroups * words {
            return Err("ink2 expert gemm: group table shorter than its count".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let acc_ptr = match acc {
            Some(a) => {
                if a.len() != act.rows * w.n {
                    return Err("ink2 expert gemm: acc is not [rows][n]".into());
                }
                a.device_ptr_mut(&stream).0 as *mut f32
            }
            None => std::ptr::null_mut(),
        };
        let rc = unsafe {
            memra_ink2_expert_gemm(
                w.quant.kind(),
                nb as i32,
                act.values.device_ptr(&stream).0 as *const c_void,
                act.scales.device_ptr(&stream).0 as *const c_void,
                act.amax.device_ptr(&stream).0 as *const f32,
                w.values.device_ptr(&stream).0 as *const c_void,
                w.scales.device_ptr(&stream).0 as *const c_void,
                w.amax.device_ptr(&stream).0 as *const f32,
                w.denominator,
                groups.device_ptr(&stream).0 as *const u32,
                ngroups as i32,
                out.device_ptr_mut(&stream).0 as *mut c_void,
                acc_ptr,
                w.n as i32,
                w.k as i32,
                wpb as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_ink2_expert_gemm returned {rc}").into());
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------
// gates

fn bf16_bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_from_bytes(b: &[u8]) -> Vec<f32> {
    reference::goldens::bf16s(b)
}

fn count_diff<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Distance in bf16 steps between two bf16 values held as fp32.
fn bf16_ulps(a: f32, b: f32) -> i64 {
    let key = |v: f32| {
        let bits = (v.to_bits() >> 16) as i64;
        if bits & 0x8000 != 0 {
            -(bits & 0x7fff)
        } else {
            bits
        }
    };
    (key(a) - key(b)).abs()
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Roughly normal, bf16-rounded.
    fn gauss(&mut self, scale: f32) -> f32 {
        let s: f32 = (0..6).map(|_| self.unit()).sum::<f32>() - 3.0;
        reference::bf16(s * scale * 1.4)
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(n);
        v
    }
}

/// Device weights with their host copies: values, scales, per-expert amax.
pub type Ink2WeightsWithHost = (Ink2Weights, Vec<u8>, Vec<u8>, Vec<f32>);

/// Random expert weights in the checkpoint's layout. NVFP4 scale bytes stay in a mid range and
/// MXFP8 values avoid the NaN codes.
pub fn ink2_random_weights(
    e: &Engine,
    quant: Ink2Quant,
    experts: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Res<Ink2WeightsWithHost> {
    let mut rng = XorShift(seed | 1);
    let mut values = rng.bytes(experts * n * quant.value_bytes(k));
    let mut scales = rng.bytes(experts * n * quant.scale_bytes(k));
    let amax: Vec<f32> = (0..experts).map(|_| 0.5 + rng.unit()).collect();
    match quant {
        Ink2Quant::Nvfp4 => scales.iter_mut().for_each(|s| *s = 0x20 + (*s % 0x50)),
        Ink2Quant::Mxfp8 => {
            values.iter_mut().for_each(|v| {
                if *v & 0x7f == 0x7f {
                    *v &= 0x80;
                }
            });
            scales.iter_mut().for_each(|s| *s = 110 + (*s % 14));
        }
    }
    let w = Ink2Weights {
        quant,
        n,
        k,
        values: e.htod_bytes(&values)?,
        scales: e.htod_bytes(&scales)?,
        amax: e.htod(&amax)?,
        denominator: 1344.0,
    };
    Ok((w, values, scales, amax))
}

fn quantize_on_device(
    e: &Engine,
    quant: Ink2Quant,
    x: &[f32],
    rows: usize,
    k: usize,
) -> Res<Ink2Rows> {
    let xd = e.htod_bytes(&bf16_bytes(x))?;
    let mut out = e.ink2_rows_alloc(quant, rows, k)?;
    e.ink2_act_quant_into(&xd, &mut out)?;
    e.stream().synchronize()?;
    Ok(out)
}

fn verdict(lines: &mut Vec<String>, fails: &mut usize, ok: bool, line: String) {
    if !ok {
        *fails += 1;
    }
    lines.push(format!("{line} {}", if ok { "OK" } else { "FAIL" }));
}

/// The per-element bound on an fp32 accumulation of `k` exact products in any order with any
/// rounding direction per addition: `(k - 1) * 2^-23 * sum |a_i w_i|`.
fn sum_bound(k: usize, magnitude: f64) -> f64 {
    (k as f64 - 1.0) * 2f64.powi(-23) * magnitude
}

/// Software models of one MMA chain over `step`-wide K steps, to say what reproduces the hardware.
fn chain_models(a: &[f32], w: &[f32], step: usize) -> [f32; 5] {
    let mut seq = 0f32; // index order, one fp32 add per product
    let mut step_rn = 0f32; // exact step sum rounded to fp32, then one fp32 add
    let mut fused_rn = 0f32; // round(acc + exact step sum), nearest
    for (ca, cw) in a.chunks(step).zip(w.chunks(step)) {
        let mut exact = 0f64;
        for (x, y) in ca.iter().zip(cw) {
            seq += x * y;
            exact += f64::from(*x) * f64::from(*y);
        }
        step_rn += exact as f32;
        fused_rn = (f64::from(fused_rn) + exact) as f32;
    }
    let fused_rz = reference::dot_chain_rz(a, w, step); // the reference's tensor-core chain model
    let trunc = reference::dot_chain_trunc(a, w, step, 25); // the truncating tensor-core model
    [seq, step_rn, fused_rn, fused_rz, trunc]
}

struct GemmStats {
    elements: usize,
    equal_index_order: usize,
    model_equal: [usize; 5],
    max_ratio: f64,
    over_bound: usize,
    max_ulp: u32,
}

#[allow(clippy::too_many_arguments)]
fn gemm_stats(
    acc: &[f32],
    a_rows: &[Vec<f32>],
    w_rows: &dyn Fn(usize, usize) -> Vec<f32>,
    expert: &[u32],
    n: usize,
    k: usize,
    step: usize,
) -> GemmStats {
    let mut s = GemmStats {
        elements: 0,
        equal_index_order: 0,
        model_equal: [0; 5],
        max_ratio: 0.0,
        over_bound: 0,
        max_ulp: 0,
    };
    for (r, a) in a_rows.iter().enumerate() {
        for col in 0..n {
            let w = w_rows(expert[r] as usize, col);
            let got = acc[r * n + col];
            let models = chain_models(a, &w, step);
            let (mut exact, mut mag) = (0f64, 0f64);
            for (x, y) in a.iter().zip(&w) {
                let p = f64::from(*x) * f64::from(*y);
                exact += p;
                mag += p.abs();
            }
            s.elements += 1;
            for (i, m) in models.iter().enumerate() {
                if m.to_bits() == got.to_bits() {
                    s.model_equal[i] += 1;
                }
            }
            if models[0].to_bits() == got.to_bits() {
                s.equal_index_order += 1;
            }
            let err = (f64::from(got) - exact).abs();
            if mag > 0.0 {
                s.max_ratio = s.max_ratio.max(err / mag);
            }
            if err > sum_bound(k, mag) {
                s.over_bound += 1;
            }
            let ulp = (i64::from(got.to_bits() as i32) - i64::from(models[0].to_bits() as i32))
                .unsigned_abs();
            if got.is_sign_negative() == models[0].is_sign_negative() {
                s.max_ulp = s.max_ulp.max(ulp.min(u64::from(u32::MAX)) as u32);
            }
        }
    }
    let _ = k;
    s
}

/// One GEMM case on the device against the reference: the accumulators under the summation bound,
/// the bf16 epilogue bit for bit from the device's own accumulators, and what software chain equals
/// the hardware.
#[allow(clippy::too_many_arguments)]
fn gemm_case(
    e: &Engine,
    name: &str,
    quant: Ink2Quant,
    wv: &[u8],
    ws: &[u8],
    wamax: &[f32],
    denominator: f32,
    n: usize,
    k: usize,
    x: &[f32],
    expert: &[u32],
    nb: usize,
    lines: &mut Vec<String>,
    fails: &mut usize,
) -> Res<()> {
    let rows = expert.len();
    let w = Ink2Weights {
        quant,
        n,
        k,
        values: e.htod_bytes(wv)?,
        scales: e.htod_bytes(ws)?,
        amax: e.htod(wamax)?,
        denominator,
    };
    let act = quantize_on_device(e, quant, x, rows, k)?;
    let (groups, ngroups) = ink2_groups(expert, nb);
    let gd = e.htod_u32_v(&groups)?;
    let mut out = e.alloc_u8(rows * n * 2)?;
    let mut acc = e.zeros(rows * n)?;
    e.ink2_expert_gemm_into(&w, &act, &gd, ngroups, nb, 0, &mut out, Some(&mut acc))?;
    let acc = e.dtoh(&acc)?;
    let out = bf16_from_bytes(&e.dtoh_u8(&out)?);
    // operands as the reference dequantizes them, from the reference's own quantization
    let (vb, sb) = (quant.value_bytes(k), quant.scale_bytes(k));
    let mut a_rows = Vec::new();
    let mut row_scale = Vec::new();
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        match quant {
            Ink2Quant::Nvfp4 => {
                let q = reference::nvfp4_act_quant(xr);
                row_scale.push(reference::nvfp4_row_scale(
                    q.amax,
                    wamax[expert[r] as usize],
                    denominator,
                ));
                a_rows.push(reference::nvfp4_dequant(&q.codes, &q.scales));
            }
            Ink2Quant::Mxfp8 => {
                let (v, s) = reference::mxfp8_act_quant(xr);
                row_scale.push(1.0);
                a_rows.push(reference::mxfp8_dequant(&v, &s));
            }
        }
    }
    let w_rows = |ex: usize, col: usize| {
        let r = ex * n + col;
        match quant {
            Ink2Quant::Nvfp4 => {
                reference::nvfp4_dequant(&wv[r * vb..(r + 1) * vb], &ws[r * sb..(r + 1) * sb])
            }
            Ink2Quant::Mxfp8 => {
                reference::mxfp8_dequant(&wv[r * vb..(r + 1) * vb], &ws[r * sb..(r + 1) * sb])
            }
        }
    };
    let step = if quant == Ink2Quant::Nvfp4 { 64 } else { 32 };
    let s = gemm_stats(&acc, &a_rows, &w_rows, expert, n, k, step);
    // epilogue: bf16((acc * 1.0) * row scale) from the device accumulators, and against the reference out
    let (mut epi_bad, mut ref_diff, mut ref_max_ulp) = (0usize, 0usize, 0i64);
    for r in 0..rows {
        let wr_out: Vec<f32> = (0..n)
            .map(|c| reference::dot_index_order(&a_rows[r], &w_rows(expert[r] as usize, c)))
            .collect();
        for c in 0..n {
            let want = match quant {
                Ink2Quant::Nvfp4 => reference::bf16(acc[r * n + c] * row_scale[r]),
                Ink2Quant::Mxfp8 => reference::bf16(acc[r * n + c]),
            };
            if want.to_bits() != out[r * n + c].to_bits() {
                epi_bad += 1;
            }
            let reference_out = match quant {
                Ink2Quant::Nvfp4 => reference::bf16(wr_out[c] * row_scale[r]),
                Ink2Quant::Mxfp8 => reference::bf16(wr_out[c]),
            };
            let d = bf16_ulps(reference_out, out[r * n + c]);
            if d != 0 {
                ref_diff += 1;
            }
            ref_max_ulp = ref_max_ulp.max(d);
        }
    }
    // The accumulators under the summation bound and the epilogue bit for bit from them are the whole
    // claim. The bf16 distance from the index-order reference is reported, not gated: a sum that
    // cancels can sit many bf16 steps from its reference inside the same absolute bound.
    // W4A4 also has to equal the reference's tensor-core chain model bit for bit (model 3 is
    // `reference::dot_chain_rz` at this K step).
    // and W8A8 the truncating model (model 4, `reference::dot_chain_trunc` at guard 25).
    let chain_ok = match quant {
        Ink2Quant::Nvfp4 => s.model_equal[3] == s.elements,
        Ink2Quant::Mxfp8 => s.model_equal[4] == s.elements,
    };
    let ok = s.over_bound == 0 && epi_bad == 0 && chain_ok;
    verdict(
        lines,
        fails,
        ok,
        format!(
            "{name} [n={n} k={k} rows={rows} nb={nb}] acc: over-bound={}/{} max|err|/sum|p|={:.3e} (bound {:.3e}) \
             bit-equal index-order={} step-rn={} fused-rn={} chain-rz={} chain-trunc25={} max-ulp-vs-index-order={}; \
             epilogue bit-bad={epi_bad}; bf16 vs reference: differ={ref_diff} max={ref_max_ulp} step",
            s.over_bound,
            s.elements,
            s.max_ratio,
            (k as f64 - 1.0) * 2f64.powi(-23),
            s.equal_index_order,
            s.model_equal[1],
            s.model_equal[2],
            s.model_equal[3],
            s.model_equal[4],
            s.max_ulp,
        ),
    );
    Ok(())
}

/// The device accumulators and bf16 outputs of the golden GEMM records (one record each, tags
/// `acc` and `out`), in the goldens' container: what the tensor core computed, for the reference's
/// CPU test of its chain model.
pub fn ink2_actq_capture(e: &Engine) -> Res<Vec<u8>> {
    use reference::goldens::{bf16s, f32s, parse, u32s};
    if !ink2_actq_built() {
        return Err("no block-scaled target in this build".into());
    }
    let g = parse(reference::goldens::BYTES)?;
    let mut records = std::collections::BTreeMap::new();
    for (name, r) in g
        .iter()
        .filter(|(n, _)| n.starts_with("w4a4/") || n.starts_with("w8a8/"))
    {
        let quant = if name.starts_with("w4a4/") {
            Ink2Quant::Nvfp4
        } else {
            Ink2Quant::Mxfp8
        };
        let d = u32s(&r["dims"]);
        let (n, k) = (d[1] as usize, d[2] as usize);
        let expert = u32s(&r["expert"]);
        let rows = expert.len();
        let (wv, ws, wamax, denominator) = match quant {
            Ink2Quant::Nvfp4 => (
                &r["wcodes"],
                &r["wscales"],
                f32s(&r["wamax"]),
                f32s(&r["denom"])[0],
            ),
            Ink2Quant::Mxfp8 => (&r["w"], &r["wexps"], vec![1.0; d[0] as usize], 1.0),
        };
        let w = Ink2Weights {
            quant,
            n,
            k,
            values: e.htod_bytes(wv)?,
            scales: e.htod_bytes(ws)?,
            amax: e.htod(&wamax)?,
            denominator,
        };
        let act = quantize_on_device(e, quant, &bf16s(&r["x"]), rows, k)?;
        let (groups, ngroups) = ink2_groups(&expert, 1);
        let gd = e.htod_u32_v(&groups)?;
        let mut out = e.alloc_u8(rows * n * 2)?;
        let mut acc = e.zeros(rows * n)?;
        e.ink2_expert_gemm_into(&w, &act, &gd, ngroups, 1, 0, &mut out, Some(&mut acc))?;
        let acc: Vec<u8> = e.dtoh(&acc)?.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut record = reference::goldens::Record::new();
        record.insert("acc".into(), acc);
        record.insert("out".into(), e.dtoh_u8(&out)?);
        records.insert(name.clone(), record);
    }
    Ok(reference::goldens::write(&records))
}

/// The act-quant gates. Returns the verdict lines (each starts with its cell name and ends in
/// `OK` or `FAIL`) and the failure count.
pub fn ink2_actq_gate(e: &Engine) -> Res<(Vec<String>, usize)> {
    use reference::goldens::{bf16s, f32s, parse, u32s};
    let g = parse(reference::goldens::BYTES)?;
    let mut lines = Vec::new();
    let mut fails = 0usize;

    // --- quantizers against the goldens, bit for bit
    for (cell, prefix, quant) in [
        ("ink2-actq-nvfp4", "nvfp4q/", Ink2Quant::Nvfp4),
        ("ink2-actq-mxfp8", "mxfp8q/", Ink2Quant::Mxfp8),
    ] {
        let (mut rows_total, mut bad_values, mut bad_scales, mut bad_amax, mut records) =
            (0, 0, 0, 0, 0);
        for (_, r) in g.iter().filter(|(n, _)| n.starts_with(prefix)) {
            let d = u32s(&r["dims"]);
            let (m, k) = (d[0] as usize, d[1] as usize);
            let out = quantize_on_device(e, quant, &bf16s(&r["x"]), m, k)?;
            let (values, scales) = (e.dtoh_u8(&out.values)?, e.dtoh_u8(&out.scales)?);
            match quant {
                Ink2Quant::Nvfp4 => {
                    bad_values += count_diff(&values, &r["codes"]);
                    bad_scales += count_diff(&scales, &r["scales"]);
                    let amax: Vec<u32> = e.dtoh(&out.amax)?.iter().map(|v| v.to_bits()).collect();
                    let want: Vec<u32> = f32s(&r["amax"]).iter().map(|v| v.to_bits()).collect();
                    bad_amax += count_diff(&amax[..m], &want);
                }
                Ink2Quant::Mxfp8 => {
                    bad_values += count_diff(&values, &r["values"]);
                    bad_scales += count_diff(&scales, &r["exps"]);
                }
            }
            rows_total += m;
            records += 1;
        }
        verdict(
            &mut lines,
            &mut fails,
            records > 0 && bad_values + bad_scales + bad_amax == 0,
            format!(
                "{cell} [{records} golden records, {rows_total} rows] value-bytes-bad={bad_values} scale-bytes-bad={bad_scales} amax-bad={bad_amax}"
            ),
        );
    }

    // --- the hardware conversions against the same goldens (the open fact)
    if ink2_actq_built() {
        let mut bad4 = 0;
        let mut total4 = 0;
        for name in ["e4m3/grid", "e4m3/satfinite"] {
            let r = &g[name];
            let x = f32s(&r["x"]);
            let (hw, _) = e.ink2_cvt_probe(&x)?;
            bad4 += count_diff(&hw, &r["byte"]);
            total4 += x.len();
        }
        let r = &g["e2m1/grid"];
        let x = f32s(&r["x"]);
        let (_, hw2) = e.ink2_cvt_probe(&x)?;
        let bad2 = count_diff(&hw2, &r["code"]);
        verdict(
            &mut lines,
            &mut fails,
            bad4 == 0 && bad2 == 0,
            format!(
                "ink2-cvt-hw cvt.rn.satfinite.e4m3x2.f32 vs RNE goldens bad={bad4}/{total4}; cvt.rn.satfinite.e2m1x2.f32 vs goldens bad={bad2}/{}",
                x.len()
            ),
        );
    } else {
        lines.push("SKIP ink2-cvt-hw (no block-scaled target in this build)".into());
    }

    // --- SiTU fused with the quantizer: bf16 rows within one step of the f64-function reference,
    //     and the quantization of the device's own rows bit for bit
    for (cell, quant) in [
        ("ink2-situ-actq-nvfp4", Ink2Quant::Nvfp4),
        ("ink2-situ-actq-mxfp8", Ink2Quant::Mxfp8),
    ] {
        let (rows, k) = (16usize, 1152usize);
        let mut rng = XorShift(0x5177_2026_1121);
        let gate_up: Vec<f32> = (0..rows * 2 * k)
            .map(|i| rng.gauss(if (i / (2 * k)) % 4 == 3 { 12.0 } else { 2.0 }))
            .collect();
        let gamma: Vec<f32> = (0..rows).map(|_| 0.05 + 3.0 * rng.unit()).collect();
        let gd = e.htod_bytes(&bf16_bytes(&gate_up))?;
        let gm = e.htod(&gamma)?;
        let mut stats = Vec::new();
        let mut quant_bad = 0usize;
        let mut max_step = 0i64;
        for approx in [false, true] {
            if approx && !ink2_actq_built() {
                continue;
            }
            let mut out = e.ink2_rows_alloc(quant, rows, k)?;
            let mut y = e.alloc_u8(rows * k * 2)?;
            e.ink2_situ_quant_into(&gd, &gm, (4.0, 25.0), approx, &mut out, Some(&mut y))?;
            let y = bf16_from_bytes(&e.dtoh_u8(&y)?);
            let (values, scales) = (e.dtoh_u8(&out.values)?, e.dtoh_u8(&out.scales)?);
            let (mut differ, mut worst) = (0usize, 0i64);
            for r in 0..rows {
                let want =
                    reference::situ_row(&gate_up[r * 2 * k..(r + 1) * 2 * k], gamma[r], 4.0, 25.0);
                for (a, b) in want.iter().zip(&y[r * k..(r + 1) * k]) {
                    let d = bf16_ulps(*a, *b);
                    differ += usize::from(d != 0);
                    worst = worst.max(d);
                }
                let yr = &y[r * k..(r + 1) * k];
                let (vb, sb) = (quant.value_bytes(k), quant.scale_bytes(k));
                let (rv, rs) = match quant {
                    Ink2Quant::Nvfp4 => {
                        let q = reference::nvfp4_act_quant(yr);
                        (q.codes, q.scales)
                    }
                    Ink2Quant::Mxfp8 => reference::mxfp8_act_quant(yr),
                };
                quant_bad += count_diff(&rv, &values[r * vb..(r + 1) * vb])
                    + count_diff(&rs, &scales[r * sb..(r + 1) * sb]);
            }
            if !approx {
                max_step = worst;
            }
            stats.push(format!(
                "{}: bf16 differ={differ}/{} max={worst} step",
                if approx {
                    "ptx-approx (measurement)"
                } else {
                    "library"
                },
                rows * k
            ));
        }
        verdict(
            &mut lines,
            &mut fails,
            quant_bad == 0 && max_step <= 1,
            format!(
                "{cell} [rows={rows} k={k}] {}; quantization of the device rows bit-bad={quant_bad}",
                stats.join("; ")
            ),
        );
    }

    // --- expert GEMMs
    if ink2_actq_built() {
        for (prefix, quant, cell) in [
            ("w4a4/", Ink2Quant::Nvfp4, "ink2-w4a4-gemm"),
            ("w8a8/", Ink2Quant::Mxfp8, "ink2-w8a8-gemm"),
        ] {
            for (name, r) in g.iter().filter(|(n, _)| n.starts_with(prefix)) {
                let d = u32s(&r["dims"]);
                let (n, k) = (d[1] as usize, d[2] as usize);
                let x = bf16s(&r["x"]);
                let expert = u32s(&r["expert"]);
                let (wv, ws, wamax, denom) = match quant {
                    Ink2Quant::Nvfp4 => (
                        &r["wcodes"],
                        &r["wscales"],
                        f32s(&r["wamax"]),
                        f32s(&r["denom"])[0],
                    ),
                    Ink2Quant::Mxfp8 => (&r["w"], &r["wexps"], vec![1.0; d[0] as usize], 1.0),
                };
                let nbs: &[usize] = if quant == Ink2Quant::Nvfp4 {
                    &[1, 2, 4]
                } else {
                    &[1, 2]
                };
                for nb in nbs {
                    gemm_case(
                        e,
                        &format!("{cell} golden {name}"),
                        quant,
                        wv,
                        ws,
                        &wamax,
                        denom,
                        n,
                        k,
                        &x,
                        &expert,
                        *nb,
                        &mut lines,
                        &mut fails,
                    )?;
                }
            }
            // the model's shapes: decode rows (top-8 of the experts, 1 and 8 tokens) and a prefill group
            for (n, k, tag) in [(2304usize, 2304usize, "w13"), (2304, 1152, "w2")] {
                let experts = 12usize;
                let (_, wv, ws, wamax) =
                    ink2_random_weights(e, quant, experts, n, k, 0xabc0 + n as u64 + k as u64)?;
                let mut rng = XorShift(0x9e37 + k as u64);
                for (rows, nb, shape) in [
                    (8usize, 1usize, "decode M=1"),
                    (64, 1, "decode M=8"),
                    (
                        40,
                        if quant == Ink2Quant::Nvfp4 { 4 } else { 2 },
                        "prefill group",
                    ),
                ] {
                    let x: Vec<f32> = (0..rows * k)
                        .map(|i| rng.gauss(1.0 + (i / k) as f32))
                        .collect();
                    let expert: Vec<u32> = (0..rows)
                        .map(|r| {
                            if shape == "prefill group" {
                                (r % 2) as u32
                            } else {
                                ((r * 5 + r / 8) % experts) as u32
                            }
                        })
                        .collect();
                    gemm_case(
                        e,
                        &format!("{cell} {tag} {shape}"),
                        quant,
                        &wv,
                        &ws,
                        &wamax,
                        1344.0,
                        n,
                        k,
                        &x,
                        &expert,
                        nb,
                        &mut lines,
                        &mut fails,
                    )?;
                }
            }
        }
    } else {
        lines.push("SKIP ink2-w4a4-gemm (no block-scaled target in this build)".into());
        lines.push("SKIP ink2-w8a8-gemm (no block-scaled target in this build)".into());
    }
    Ok((lines, fails))
}

/// Bytes one expert GEMM call reads from the weights for `distinct` experts.
fn weight_bytes(quant: Ink2Quant, n: usize, k: usize, distinct: usize) -> usize {
    distinct * n * (quant.value_bytes(k) + quant.scale_bytes(k))
}

/// Microbench of the expert GEMMs and the quantizers at the model's shapes. Every repetition reads
/// a different set of experts out of `experts`, so the weights come from DRAM, not from L2.
pub fn ink2_actq_bench(e: &Engine, experts: usize, reps: usize) -> Res<Vec<String>> {
    let mut lines = Vec::new();
    if !ink2_actq_built() {
        return Ok(vec![
            "ink2-actq-bench: no block-scaled target in this build".into(),
        ]);
    }
    let hidden = 2304usize;
    let ff = 1152usize;
    // The harness's own cost per call (argument marshalling, the launch, one trivial block): a GEMM
    // of 16 outputs over K = 64. The kernel rows below include it; "net" subtracts it.
    let overhead = {
        let (w, _, _, _) = ink2_random_weights(e, Ink2Quant::Nvfp4, 1, 16, 64, 5)?;
        let act = quantize_on_device(e, Ink2Quant::Nvfp4, &[1.0f32; 64], 1, 64)?;
        let (groups, ngroups) = ink2_groups(&[0], 1);
        let gd = e.htod_u32_v(&groups)?;
        let mut out = e.alloc_u8(32)?;
        let mut best = f64::MAX;
        // also the warm-up: the card leaves its idle clocks during these calls
        for _ in 0..40 {
            let t0 = std::time::Instant::now();
            for _ in 0..2000 {
                e.ink2_expert_gemm_into(&w, &act, &gd, ngroups, 1, 1, &mut out, None)?;
            }
            e.stream().synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() * 1e6 / 2000.0);
        }
        best
    };
    lines.push(format!("ink2-actq-bench call overhead (16 x 64 GEMM, one block): {overhead:.2} us/call (best of 40 x 2000 calls)"));
    for quant in [Ink2Quant::Nvfp4, Ink2Quant::Mxfp8] {
        for (n, k, tag) in [(2 * ff, hidden, "w13 gate/up"), (hidden, ff, "w2 down")] {
            let (w, _, _, _) = ink2_random_weights(e, quant, experts, n, k, 77 + k as u64)?;
            let mut rng = XorShift(4242);
            for (tokens, nb) in [
                (1usize, 1usize),
                (2, 1),
                (4, 1),
                (8, 1),
                (256, if quant == Ink2Quant::Nvfp4 { 4 } else { 2 }),
            ] {
                let rows = tokens * 8;
                let x: Vec<f32> = (0..rows * k).map(|_| rng.gauss(1.0)).collect();
                let act = quantize_on_device(e, quant, &x, rows, k)?;
                // per repetition: each token takes 8 distinct experts, rotating through the pool
                let sets = if tokens <= 8 {
                    (experts / (8 * tokens)).max(1)
                } else {
                    1
                };
                let mut tables = Vec::new();
                let mut distinct_total = 0usize;
                for s in 0..sets {
                    let expert: Vec<u32> = (0..rows)
                        .map(|r| {
                            if tokens <= 8 {
                                ((s * rows + r) % experts) as u32
                            } else {
                                (rng.next() % experts as u64) as u32
                            }
                        })
                        .collect();
                    let mut d: Vec<u32> = expert.clone();
                    d.sort_unstable();
                    d.dedup();
                    distinct_total += d.len();
                    let (groups, ngroups) = ink2_groups(&expert, nb);
                    tables.push((e.htod_u32_v(&groups)?, ngroups));
                }
                let mut out = e.alloc_u8(rows * n * 2)?;
                let distinct = distinct_total as f64 / sets as f64;
                let bytes = weight_bytes(quant, n, k, 1) as f64 * distinct;
                let flops = 2.0 * rows as f64 * n as f64 * k as f64;
                let mut arms = Vec::new();
                let mut best_all = (f64::MAX, 0usize);
                for wpb in [1usize, 2, 4, 8] {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        e.ink2_expert_gemm_into(
                            &w,
                            &act,
                            &tables[0].0,
                            tables[0].1,
                            nb,
                            wpb,
                            &mut out,
                            None,
                        )?;
                        e.stream().synchronize()?;
                        let t0 = std::time::Instant::now();
                        for i in 0..reps {
                            let (t, ng) = &tables[i % sets];
                            e.ink2_expert_gemm_into(&w, &act, t, *ng, nb, wpb, &mut out, None)?;
                        }
                        e.stream().synchronize()?;
                        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
                    }
                    arms.push(format!("w{wpb}={best:.1}"));
                    if best < best_all.0 {
                        best_all = (best, wpb);
                    }
                }
                let (best, wpb) = best_all;
                lines.push(format!(
                    "ink2-actq-bench {quant:?} {tag} tokens={tokens} rows={rows} nb={nb}: best {best:.1} us/call at wpb={wpb}, {:.2} MB weights ({distinct:.1} experts), {:.0} GB/s ({:.0} GB/s net of the call overhead), {:.2} TFLOP/s; us/call by arm: {} (best of 5 x {reps} calls, {sets} expert sets)",
                    bytes / 1e6,
                    bytes / best / 1e3,
                    bytes / (best - overhead).max(0.1) / 1e3,
                    flops / best / 1e6,
                    arms.join(" "),
                ));
            }
        }
        // quantizers
        for (rows, k, situ) in [
            (1usize, hidden, false),
            (8, hidden, false),
            (8, ff, true),
            (64, ff, true),
            (2048, ff, true),
        ] {
            let mut rng = XorShift(99);
            let width = if situ { 2 * k } else { k };
            let x: Vec<f32> = (0..rows * width).map(|_| rng.gauss(2.0)).collect();
            let xd = e.htod_bytes(&bf16_bytes(&x))?;
            let gm = e.htod(&vec![1.0f32; rows])?;
            let mut out = e.ink2_rows_alloc(quant, rows, k)?;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    if situ {
                        e.ink2_situ_quant_into(&xd, &gm, (4.0, 25.0), false, &mut out, None)?;
                    } else {
                        e.ink2_act_quant_into(&xd, &mut out)?;
                    }
                }
                e.stream().synchronize()?;
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
            }
            lines.push(format!(
                "ink2-actq-bench {quant:?} quantize{} rows={rows} k={k}: {best:.1} us/call (best of 3 x {reps} calls)",
                if situ { " + SiTU" } else { "" }
            ));
        }
    }
    Ok(lines)
}
