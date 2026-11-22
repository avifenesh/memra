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
        tg: *const f64,
        tu: *const f64,
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

/// The expert GEMM reads its weights in MMA-fragment order (`cu/inkling2_actq.cu`). A projection
/// stored row-major as `rows` rows of `rowbytes` bytes (32 bytes a row per K step: the e2m1 codes of
/// 64 inputs or the e4m3 values of 32) becomes tiles of 16 rows; per tile and K step 512 bytes, lane
/// `L = 4g + t` holding row g bytes `[4t, 4t + 4)` of the step, row g + 8 the same, then both rows at
/// `16 + 4t`. A layout change only: every byte is kept, so every product is the one stored.
pub fn ink2_frag_values(src: &[u8], rows: usize, rowbytes: usize) -> Result<Vec<u8>, String> {
    if !rows.is_multiple_of(16) || !rowbytes.is_multiple_of(32) || src.len() != rows * rowbytes {
        return Err(format!(
            "fragment order: {} bytes are not {rows} rows (a multiple of 16) of {rowbytes} (a multiple of 32)",
            src.len()
        ));
    }
    let mut out = vec![0u8; src.len()];
    by_tiles(&mut out, src, 16 * rowbytes, |o, s| {
        for step in 0..rowbytes / 32 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let d = step * 512 + lane * 16;
                let parts = [
                    (g, 4 * t),
                    (g + 8, 4 * t),
                    (g, 16 + 4 * t),
                    (g + 8, 16 + 4 * t),
                ];
                for (q, (r, c)) in parts.into_iter().enumerate() {
                    let at = r * rowbytes + step * 32 + c;
                    o[d + 4 * q..d + 4 * q + 4].copy_from_slice(&s[at..at + 4]);
                }
            }
        }
    });
    Ok(out)
}

/// The weight scales of [`ink2_frag_values`]'s tiles: row-major `rows` x `rowbytes` (a multiple of 4:
/// one 4-byte word a K step for NVFP4, a word per 4 steps for MXFP8) becomes `[tile][word][16 rows]`
/// of 4-byte words.
pub fn ink2_frag_scales(src: &[u8], rows: usize, rowbytes: usize) -> Result<Vec<u8>, String> {
    if !rows.is_multiple_of(16) || !rowbytes.is_multiple_of(4) || src.len() != rows * rowbytes {
        return Err(format!(
            "fragment order: {} scale bytes are not {rows} rows (a multiple of 16) of {rowbytes} (a multiple of 4)",
            src.len()
        ));
    }
    let mut out = vec![0u8; src.len()];
    by_tiles(&mut out, src, 16 * rowbytes, |o, s| {
        for w in 0..rowbytes / 4 {
            for r in 0..16 {
                let d = (w * 16 + r) * 4;
                o[d..d + 4].copy_from_slice(&s[r * rowbytes + 4 * w..r * rowbytes + 4 * w + 4]);
            }
        }
    });
    Ok(out)
}

/// `f(out tile, src tile)` over equal tiles of both, on up to 8 threads.
fn by_tiles(out: &mut [u8], src: &[u8], tile: usize, f: impl Fn(&mut [u8], &[u8]) + Sync) {
    let tiles = out.len() / tile;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 8)
        .min(tiles.max(1));
    let per = tiles.div_ceil(threads).max(1) * tile;
    std::thread::scope(|sc| {
        for (oc, scs) in out.chunks_mut(per).zip(src.chunks(per)) {
            let f = &f;
            sc.spawn(move || {
                for (o, s) in oc.chunks_mut(tile).zip(scs.chunks(tile)) {
                    f(o, s);
                }
            });
        }
    });
}

/// One projection of the experts on the device, in the expert GEMM's fragment order
/// ([`ink2_frag_values`], [`ink2_frag_scales`]).
pub struct Ink2Weights {
    pub quant: Ink2Quant,
    /// Experts in the buffers: values `[experts][n][value_bytes(k)]` as stored, then in fragment order;
    /// scales likewise; amax `[experts]`.
    pub experts: usize,
    pub n: usize,
    pub k: usize,
    pub values: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// Per-expert amax (NVFP4; one unused element for MXFP8).
    pub amax: CudaSlice<f32>,
    /// The outer-scale denominator (NVFP4).
    pub denominator: f32,
}

impl Ink2Weights {
    /// Uploads a projection stored row-major (`values` `[experts][n][value_bytes(k)]`, `scales`
    /// `[experts][n][scale_bytes(k)]`) in fragment order.
    #[allow(clippy::too_many_arguments)] // allow: one projection's shape and buffers
    pub fn upload(
        e: &Engine,
        quant: Ink2Quant,
        experts: usize,
        n: usize,
        k: usize,
        values: &[u8],
        scales: &[u8],
        amax: &[f32],
        denominator: f32,
    ) -> Res<Self> {
        let rows = experts * n;
        Ok(Self {
            quant,
            experts,
            n,
            k,
            values: e.htod_bytes(&ink2_frag_values(values, rows, quant.value_bytes(k))?)?,
            scales: e.htod_bytes(&ink2_frag_scales(scales, rows, quant.scale_bytes(k))?)?,
            amax: e.htod(amax)?,
            denominator,
        })
    }
}

/// SiTU's two factor tables on the device (65,536 f64 each, from
/// [`reference::situ_tables`]), with the host copy the gates compare against.
pub struct Ink2SituTables {
    pub host: reference::SituTables,
    pub gate: CudaSlice<u8>,
    pub up: CudaSlice<u8>,
}

/// A group table on the device, built from a host expert-per-row list: the wrapper checks its
/// experts and rows against the operands before every launch.
pub struct Ink2Groups {
    pub table: CudaSlice<u32>,
    pub ngroups: usize,
    pub nb: usize,
    /// One past the largest expert id and the largest row in the table.
    pub experts_end: usize,
    pub rows_end: usize,
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
    /// Upload SiTU's factor tables for the given caps (the checkpoint's are 4 and 25).
    pub fn ink2_situ_tables(&self, gate_cap: f32, up_cap: f32) -> Res<Ink2SituTables> {
        let host = reference::situ_tables(gate_cap, up_cap);
        let bytes = |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        Ok(Ink2SituTables {
            gate: self.htod_bytes(&bytes(&host.gate))?,
            up: self.htod_bytes(&bytes(&host.up))?,
            host,
        })
    }

    /// Upload the group table of [`ink2_groups`] for `expert_of_row` and `nb` B tiles.
    pub fn ink2_groups_upload(&self, expert_of_row: &[u32], nb: usize) -> Res<Ink2Groups> {
        if !(nb == 1 || nb == 2 || nb == 4) {
            return Err(format!("ink2 groups: nb {nb} is not 1, 2 or 4").into());
        }
        let (table, ngroups) = ink2_groups(expert_of_row, nb);
        if ngroups == 0 {
            return Err("ink2 groups: no rows".into());
        }
        Ok(Ink2Groups {
            table: self.htod_u32_v(&table)?,
            ngroups,
            nb,
            experts_end: expert_of_row.iter().max().map_or(0, |e| *e as usize + 1),
            rows_end: expert_of_row.len(),
        })
    }

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
                std::ptr::null(),
                std::ptr::null(),
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("memra_ink2_actq returned {rc}").into());
        }
        Ok(())
    }

    /// The down projection's input: softcapped SwiGLU of interleaved bf16 `gate_up` (`[rows][2k]`)
    /// times each row's routing weight, the mathematical value rounded once to bf16
    /// ([`reference::situ_value`]), and quantized into `out`. `y` (optional, `[rows][k]` bf16)
    /// receives the bf16 rows.
    pub fn ink2_situ_quant_into(
        &self,
        gate_up: &CudaSlice<u8>,
        gamma: &CudaSlice<f32>,
        tables: &Ink2SituTables,
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
                tables.gate.device_ptr(&stream).0 as *const f64,
                tables.up.device_ptr(&stream).0 as *const f64,
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
    /// each); `acc` (optional, `[rows][n]`) receives the fp32 accumulators. `wpb` (warps per block)
    /// takes 0 for the default. Every buffer length, the group table's experts and rows, and the
    /// launch shape are checked here, so no argument reaches memory outside its buffer.
    /// Asynchronous on the stream.
    pub fn ink2_expert_gemm_into(
        &self,
        w: &Ink2Weights,
        act: &Ink2Rows,
        groups: &Ink2Groups,
        wpb: usize,
        out: &mut CudaSlice<u8>,
        acc: Option<&mut CudaSlice<f32>>,
    ) -> Res<()> {
        let (q, n, k) = (w.quant, w.n, w.k);
        if q != act.quant || k != act.k || out.len() != act.rows * n * 2 {
            return Err("ink2 expert gemm: operand shapes disagree".into());
        }
        let step = if q == Ink2Quant::Nvfp4 { 64 } else { 128 };
        if n == 0
            || n % 16 != 0
            || k == 0
            || k % step != 0
            || n > i32::MAX as usize
            || k > i32::MAX as usize
        {
            return Err(format!(
                "ink2 expert gemm: n {n} (multiple of 16) or k {k} (multiple of {step}) refused"
            )
            .into());
        }
        if w.values.len() != w.experts * n * q.value_bytes(k)
            || w.scales.len() != w.experts * n * q.scale_bytes(k)
            || (q == Ink2Quant::Nvfp4 && w.amax.len() < w.experts)
        {
            return Err("ink2 expert gemm: weight buffers are not [experts][n][k]".into());
        }
        if act.values.len() != act.rows * q.value_bytes(k)
            || act.scales.len() != act.rows * q.scale_bytes(k)
            || (q == Ink2Quant::Nvfp4 && act.amax.len() < act.rows)
        {
            return Err("ink2 expert gemm: activation buffers are not [rows][k]".into());
        }
        if groups.experts_end > w.experts || groups.rows_end > act.rows {
            return Err(format!(
                "ink2 expert gemm: the group table names expert {} of {} or row {} of {}",
                groups.experts_end, w.experts, groups.rows_end, act.rows
            )
            .into());
        }
        let (nb, ngroups) = (groups.nb, groups.ngroups);
        let words = unsafe { memra_ink2_group_words(nb as i32) } as usize;
        if groups.table.len() != ngroups * words || ngroups > i32::MAX as usize || wpb > 8 {
            return Err("ink2 expert gemm: group table or launch shape refused".into());
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
                groups.table.device_ptr(&stream).0 as *const u32,
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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

/// Devices on which the tensor-core chain models were measured bit for bit (the device capture and
/// the gate's cells). On any other device the gate reports the model match and holds only the
/// summation bound and the epilogue: an unmeasured chain is not a gate.
pub const INK2_CHAIN_MEASURED: &[&str] = &["NVIDIA GeForce RTX 5090 Laptop GPU"];

fn chain_measured(e: &Engine) -> (bool, String) {
    let name = e.gpu.ctx.name().unwrap_or_default();
    (INK2_CHAIN_MEASURED.contains(&name.as_str()), name)
}

/// Device weights with their host copies: values, scales, per-expert amax.
pub type Ink2WeightsWithHost = (Ink2Weights, Vec<u8>, Vec<u8>, Vec<f32>);

/// Random expert weights in the checkpoint's layout. NVFP4 scale bytes stay in a mid range and
/// MXFP8 values avoid the NaN codes; `subnormal` makes 97% of the MXFP8 codes e4m3 subnormals.
pub fn ink2_random_weights(
    e: &Engine,
    quant: Ink2Quant,
    experts: usize,
    n: usize,
    k: usize,
    seed: u64,
    subnormal: bool,
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
            if subnormal {
                for v in values.iter_mut() {
                    if rng.unit() < 0.97 {
                        *v = (*v & 0x80) | (1 + (*v % 7));
                    }
                }
            }
            scales.iter_mut().for_each(|s| *s = 110 + (*s % 14));
        }
    }
    let w = Ink2Weights::upload(e, quant, experts, n, k, &values, &scales, &amax, 1344.0)?;
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

/// A quantized operand row as the reference reads it: the dequantized values and, for MXFP8, the
/// operand exponents the tensor core sees.
struct Operand {
    values: Vec<f32>,
    exponents: Vec<i32>,
}

fn operand(quant: Ink2Quant, codes: &[u8], scales: &[u8]) -> Operand {
    match quant {
        Ink2Quant::Nvfp4 => Operand {
            values: reference::nvfp4_dequant(codes, scales),
            exponents: Vec::new(),
        },
        Ink2Quant::Mxfp8 => Operand {
            values: reference::mxfp8_dequant(codes, scales),
            exponents: reference::mxfp8_operand_exponents(codes, scales),
        },
    }
}

/// The reference's model of the sm_120a accumulator for one output.
fn chain_model(quant: Ink2Quant, a: &Operand, w: &Operand) -> f32 {
    match quant {
        Ink2Quant::Nvfp4 => reference::dot_chain_rz(&a.values, &w.values, 64),
        Ink2Quant::Mxfp8 => {
            reference::dot_chain_trunc(&a.values, &a.exponents, &w.values, &w.exponents, 32, 25)
        }
    }
}

/// One GEMM case on the device against the reference: the accumulators equal the reference's
/// chain model bit for bit (on a measured device) and sit under the summation bound, and the bf16
/// epilogue follows bit for bit from the device's own accumulators.
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
    let experts = wv.len() / (n * quant.value_bytes(k));
    let w = Ink2Weights::upload(e, quant, experts, n, k, wv, ws, wamax, denominator)?;
    let act = quantize_on_device(e, quant, x, rows, k)?;
    let groups = e.ink2_groups_upload(expert, nb)?;
    let mut out = e.alloc_u8(rows * n * 2)?;
    let mut acc = e.zeros(rows * n)?;
    e.ink2_expert_gemm_into(&w, &act, &groups, 0, &mut out, Some(&mut acc))?;
    let acc = e.dtoh(&acc)?;
    let out = bf16_from_bytes(&e.dtoh_u8(&out)?);
    let (vb, sb) = (quant.value_bytes(k), quant.scale_bytes(k));
    let (mut elements, mut model_equal, mut index_equal, mut over_bound, mut epi_bad) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut max_ratio = 0f64;
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        let ex = expert[r] as usize;
        let (a, row_scale) = match quant {
            Ink2Quant::Nvfp4 => {
                let q = reference::nvfp4_act_quant(xr);
                let s = reference::nvfp4_row_scale(q.amax, wamax[ex], denominator);
                (operand(quant, &q.codes, &q.scales), s)
            }
            Ink2Quant::Mxfp8 => {
                let (v, s) = reference::mxfp8_act_quant(xr);
                (operand(quant, &v, &s), 1.0)
            }
        };
        for col in 0..n {
            let rr = ex * n + col;
            let wo = operand(
                quant,
                &wv[rr * vb..(rr + 1) * vb],
                &ws[rr * sb..(rr + 1) * sb],
            );
            let got = acc[r * n + col];
            elements += 1;
            model_equal += usize::from(chain_model(quant, &a, &wo).to_bits() == got.to_bits());
            index_equal += usize::from(
                reference::dot_index_order(&a.values, &wo.values).to_bits() == got.to_bits(),
            );
            let (mut exact, mut mag) = (0f64, 0f64);
            for (p, q) in a.values.iter().zip(&wo.values) {
                let t = f64::from(*p) * f64::from(*q);
                exact += t;
                mag += t.abs();
            }
            let err = (f64::from(got) - exact).abs();
            if mag > 0.0 {
                max_ratio = max_ratio.max(err / mag);
            }
            over_bound += usize::from(err > sum_bound(k, mag));
            let want = match quant {
                Ink2Quant::Nvfp4 => reference::bf16((got * 1.0f32) * row_scale),
                Ink2Quant::Mxfp8 => reference::bf16(got),
            };
            epi_bad += usize::from(want.to_bits() != out[r * n + col].to_bits());
        }
    }
    let (measured, device) = chain_measured(e);
    let chain_ok = !measured || model_equal == elements;
    verdict(
        lines,
        fails,
        over_bound == 0 && epi_bad == 0 && chain_ok,
        format!(
            "{name} [n={n} k={k} rows={rows} nb={nb}] acc: chain-model bit-equal={model_equal}/{elements} ({}) \
             index-order bit-equal={index_equal}; over-bound={over_bound} max|err|/sum|p|={max_ratio:.3e} (bound {:.3e}); \
             epilogue bit-bad={epi_bad}",
            if measured {
                "gated".to_string()
            } else {
                format!("reported only: the chain is not measured on {device}")
            },
            (k as f64 - 1.0) * 2f64.powi(-23),
        ),
    );
    Ok(())
}

/// The device accumulators and bf16 outputs of the golden GEMM records (one record each, tags
/// `acc` and `out`), in the goldens' container: what the tensor core computed, for the reference's
/// CPU test of its chain models.
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
        let (experts, n, k) = (d[0] as usize, d[1] as usize, d[2] as usize);
        let expert = u32s(&r["expert"]);
        let rows = expert.len();
        let (wv, ws, wamax, denominator) = match quant {
            Ink2Quant::Nvfp4 => (
                &r["wcodes"],
                &r["wscales"],
                f32s(&r["wamax"]),
                f32s(&r["denom"])[0],
            ),
            Ink2Quant::Mxfp8 => (&r["w"], &r["wexps"], vec![1.0; experts], 1.0),
        };
        let w = Ink2Weights::upload(e, quant, experts, n, k, wv, ws, &wamax, denominator)?;
        let act = quantize_on_device(e, quant, &bf16s(&r["x"]), rows, k)?;
        let groups = e.ink2_groups_upload(&expert, 1)?;
        let mut out = e.alloc_u8(rows * n * 2)?;
        let mut acc = e.zeros(rows * n)?;
        e.ink2_expert_gemm_into(&w, &act, &groups, 0, &mut out, Some(&mut acc))?;
        let acc: Vec<u8> = e.dtoh(&acc)?.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut record = reference::goldens::Record::new();
        record.insert("acc".into(), acc);
        record.insert("out".into(), e.dtoh_u8(&out)?);
        records.insert(name.clone(), record);
    }
    Ok(reference::goldens::write(&records))
}

/// SiTU fused with the quantizer against the reference, bit for bit: the bf16 rows and the
/// quantization of the device's own rows. Also counts, on the CPU side, the outputs whose f64 value
/// lies within 2^-40 (relative) of a bf16 rounding boundary: a nonzero count would make the one
/// rounding depend on the f64 factors' last bits.
#[allow(clippy::too_many_arguments)]
fn situ_case(
    e: &Engine,
    tables: &Ink2SituTables,
    quant: Ink2Quant,
    name: &str,
    gate_up: &[u16],
    gamma: &[f32],
    k: usize,
    lines: &mut Vec<String>,
    fails: &mut usize,
) -> Res<()> {
    let rows = gamma.len();
    let gu: Vec<u8> = gate_up.iter().flat_map(|v| v.to_le_bytes()).collect();
    let gd = e.htod_bytes(&gu)?;
    let gm = e.htod(gamma)?;
    let mut out = e.ink2_rows_alloc(quant, rows, k)?;
    let mut y = e.alloc_u8(rows * k * 2)?;
    e.ink2_situ_quant_into(&gd, &gm, tables, &mut out, Some(&mut y))?;
    let y = bf16_from_bytes(&e.dtoh_u8(&y)?);
    let (values, scales) = (e.dtoh_u8(&out.values)?, e.dtoh_u8(&out.scales)?);
    let t = &tables.host;
    let (mut differ, mut quant_bad, mut near) = (0usize, 0usize, 0usize);
    let (vb, sb) = (quant.value_bytes(k), quant.scale_bytes(k));
    for r in 0..rows {
        let row: Vec<f32> = gate_up[r * 2 * k..(r + 1) * 2 * k]
            .iter()
            .map(|b| f32::from_bits(u32::from(*b) << 16))
            .collect();
        let want = reference::situ_row(&row, gamma[r], t);
        let yr = &y[r * k..(r + 1) * k];
        differ += count_diff(&bits(&want), &bits(yr));
        for j in 0..k {
            let z = (t.gate[gate_up[r * 2 * k + 2 * j] as usize]
                * t.up[gate_up[r * 2 * k + 2 * j + 1] as usize])
                * f64::from(gamma[r]);
            if z.is_finite() && z != 0.0 {
                // distance to the nearest bf16 midpoint, relative
                let lo = f64::from(f32::from_bits(((z as f32).to_bits() >> 16) << 16));
                let ulp = f64::from(f32::from_bits((((z as f32).to_bits() >> 16) + 1) << 16)) - lo;
                let mid = lo + ulp / 2.0;
                if ((z - mid) / z).abs() < 2f64.powi(-40) {
                    near += 1;
                }
            }
        }
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
    verdict(
        lines,
        fails,
        differ == 0 && quant_bad == 0,
        format!(
            "{name} [rows={rows} k={k}] bf16 rows differ from the f64 reference={differ}/{}; quantization of the device rows bit-bad={quant_bad}; outputs within 2^-40 of a bf16 midpoint={near}",
            rows * k
        ),
    );
    Ok(())
}

/// The routed experts of the golden `chain/*` records end to end on the device (quantize, w13,
/// SiTU and its quantization, w2; the slot combine on the host) against the reference's expert
/// block with the sm_120a accumulators, bit for bit at every stage.
fn expert_block_case(
    e: &Engine,
    tables: &Ink2SituTables,
    kind: &str,
    r: &reference::goldens::Record,
    lines: &mut Vec<String>,
    fails: &mut usize,
) -> Res<()> {
    use reference::goldens::{bf16s, f32s, u32s};
    let quant = if kind == "nvfp4" {
        Ink2Quant::Nvfp4
    } else {
        Ink2Quant::Mxfp8
    };
    let d = u32s(&r["dims"]);
    let (experts, h, f, tokens, topk) = (
        d[0] as usize,
        d[1] as usize,
        d[2] as usize,
        d[3] as usize,
        d[4] as usize,
    );
    let x = bf16s(&r["x"]);
    let sel = u32s(&r["sel"]);
    let gamma = f32s(&r["gamma"]);
    let tag = |s: &str| {
        if kind == "nvfp4" {
            format!("{s}codes")
        } else {
            s.to_string()
        }
    };
    let stag = |s: &str| {
        if kind == "nvfp4" {
            format!("{s}scales")
        } else {
            format!("{s}exps")
        }
    };
    let (a13, a2) = if kind == "nvfp4" {
        (f32s(&r["w13amax"]), f32s(&r["w2amax"]))
    } else {
        (vec![1.0; experts], vec![1.0; experts])
    };
    let (d13, d2) = if kind == "nvfp4" {
        (f32s(&r["w13denom"])[0], f32s(&r["w2denom"])[0])
    } else {
        (1.0, 1.0)
    };
    let upload = |name: &str, n: usize, k: usize, amax: &[f32], den: f32| -> Res<Ink2Weights> {
        Ink2Weights::upload(
            e,
            quant,
            experts,
            n,
            k,
            &r[&tag(name)],
            &r[&stag(name)],
            amax,
            den,
        )
    };
    let w13 = upload("w13", 2 * f, h, &a13, d13)?;
    let w2 = upload("w2", h, f, &a2, d2)?;
    // one routed row per (token, slot): its own copy of the token's quantized input
    let rows = tokens * topk;
    let xr: Vec<f32> = (0..rows)
        .flat_map(|row| x[(row / topk) * h..(row / topk + 1) * h].to_vec())
        .collect();
    let act = quantize_on_device(e, quant, &xr, rows, h)?;
    let groups = e.ink2_groups_upload(&sel, 1)?;
    let mut gu = e.alloc_u8(rows * 2 * f * 2)?;
    e.ink2_expert_gemm_into(&w13, &act, &groups, 0, &mut gu, None)?;
    let gm = e.htod(&gamma)?;
    let mut yq = e.ink2_rows_alloc(quant, rows, f)?;
    let mut y = e.alloc_u8(rows * f * 2)?;
    e.ink2_situ_quant_into(&gu, &gm, tables, &mut yq, Some(&mut y))?;
    let mut down = e.alloc_u8(rows * h * 2)?;
    e.ink2_expert_gemm_into(&w2, &yq, &groups, 0, &mut down, None)?;
    let (gu, y, down) = (
        bf16_from_bytes(&e.dtoh_u8(&gu)?),
        bf16_from_bytes(&e.dtoh_u8(&y)?),
        bf16_from_bytes(&e.dtoh_u8(&down)?),
    );
    let (mut bad_gu, mut bad_y, mut bad_down, mut bad_out) = (0, 0, 0, 0);
    let wv13 = (&r[&tag("w13")], &r[&stag("w13")]);
    let wv2 = (&r[&tag("w2")], &r[&stag("w2")]);
    for t in 0..tokens {
        let s: Vec<usize> = sel[t * topk..(t + 1) * topk]
            .iter()
            .map(|v| *v as usize)
            .collect();
        let gmm = &gamma[t * topk..(t + 1) * topk];
        let xt = &x[t * h..(t + 1) * h];
        let want = if quant == Ink2Quant::Nvfp4 {
            let w13r = reference::Nvfp4Experts {
                codes: wv13.0,
                scales: wv13.1,
                amax: &a13,
                denominator: d13,
                n: 2 * f,
                k: h,
            };
            let w2r = reference::Nvfp4Experts {
                codes: wv2.0,
                scales: wv2.1,
                amax: &a2,
                denominator: d2,
                n: h,
                k: f,
            };
            reference::expert_block_w4a4(
                xt,
                &s,
                gmm,
                &w13r,
                &w2r,
                &tables.host,
                reference::Accumulation::Sm120,
            )
        } else {
            let w13r = reference::Mxfp8Experts {
                values: wv13.0,
                exponents: wv13.1,
                n: 2 * f,
                k: h,
            };
            let w2r = reference::Mxfp8Experts {
                values: wv2.0,
                exponents: wv2.1,
                n: h,
                k: f,
            };
            reference::expert_block_w8a8(
                xt,
                &s,
                gmm,
                &w13r,
                &w2r,
                &tables.host,
                reference::Accumulation::Sm120,
            )
        };
        let mut comb = vec![0f32; h];
        for slot in 0..topk {
            let row = t * topk + slot;
            bad_gu += count_diff(
                &bits(&want.gate_up[slot]),
                &bits(&gu[row * 2 * f..(row + 1) * 2 * f]),
            );
            bad_y += count_diff(&bits(&want.y[slot]), &bits(&y[row * f..(row + 1) * f]));
            let dr = &down[row * h..(row + 1) * h];
            bad_down += count_diff(&bits(&want.down[slot]), &bits(dr));
            for (c, v) in comb.iter_mut().zip(dr) {
                *c += v;
            }
        }
        let comb: Vec<f32> = comb.into_iter().map(reference::bf16).collect();
        bad_out += count_diff(&bits(&want.out), &bits(&comb));
    }
    let (measured, device) = chain_measured(e);
    let ok = !measured || bad_gu + bad_y + bad_down + bad_out == 0;
    verdict(
        lines,
        fails,
        ok,
        format!(
            "ink2-expert-block-{kind} [tokens={tokens} top-k={topk} h={h} f={f}] vs the reference block (sm_120a accumulators): gate_up bad={bad_gu} y bad={bad_y} down bad={bad_down} out bad={bad_out}{}",
            if measured {
                String::new()
            } else {
                format!(" (reported only: the chain is not measured on {device})")
            }
        ),
    );
    Ok(())
}

/// The act-quant gates. Returns the verdict lines (each starts with its cell name and ends in
/// `OK` or `FAIL`) and the failure count.
pub fn ink2_actq_gate(e: &Engine) -> Res<(Vec<String>, usize)> {
    use reference::goldens::{bf16s, f32s, parse, u32s};
    let g = parse(reference::goldens::BYTES)?;
    let mut lines = Vec::new();
    let mut fails = 0usize;

    // --- quantizers against the goldens, bit for bit (the non-finite records included)
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

    // --- the hardware conversions against the same goldens (the open fact); NaN reported
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
        let (n4, n2) = e.ink2_cvt_probe(&[f32::NAN, -f32::NAN])?;
        verdict(
            &mut lines,
            &mut fails,
            bad4 == 0 && bad2 == 0 && n4 == [0x7f, 0x7f] && n2 == [0x7, 0x7],
            format!(
                "ink2-cvt-hw cvt.rn.satfinite.e4m3x2.f32 vs RNE goldens bad={bad4}/{total4}; cvt.rn.satfinite.e2m1x2.f32 vs goldens bad={bad2}/{}; NaN, -NaN: hw e4m3 {:#04x} {:#04x}, hw e2m1 {:#x} {:#x} (the software rounding: 0x7f 0x7f, 0x7 0x7)",
                x.len(),
                n4[0],
                n4[1],
                n2[0],
                n2[1]
            ),
        );
    } else {
        lines.push("SKIP ink2-cvt-hw (no block-scaled target in this build)".into());
    }

    // --- SiTU fused with the quantizer, bit for bit against the f64 reference
    let tables = e.ink2_situ_tables(4.0, 25.0)?;
    for (cell, quant) in [
        ("ink2-situ-actq-nvfp4", Ink2Quant::Nvfp4),
        ("ink2-situ-actq-mxfp8", Ink2Quant::Mxfp8),
    ] {
        // every bf16 bit pattern as a gate and as an up value (64 rows x 1024; the ups permuted)
        let mut rng = XorShift(0x5177_2026_1121);
        let (rows, k) = (64usize, 1024usize);
        let mut ups: Vec<u16> = (0..=0xffffu16).collect();
        for i in (1..ups.len()).rev() {
            ups.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let gu: Vec<u16> = (0..rows * k).flat_map(|i| [i as u16, ups[i]]).collect();
        let gamma: Vec<f32> = (0..rows).map(|_| 0.05 + 3.0 * rng.unit()).collect();
        situ_case(
            e,
            &tables,
            quant,
            &format!("{cell} every-bf16"),
            &gu,
            &gamma,
            k,
            &mut lines,
            &mut fails,
        )?;
        // the dense window: every gate with exponent field 117..133 against every such up
        let pat: Vec<u16> = (0..2u16)
            .flat_map(|s| {
                (117u16..134)
                    .flat_map(move |ex| (0..128u16).map(move |m| (s << 15) | (ex << 7) | m))
            })
            .collect();
        let kk = pat.len();
        let gu: Vec<u16> = (0..kk * kk)
            .flat_map(|i| [pat[i % kk], pat[i / kk]])
            .collect();
        let gamma: Vec<f32> = (0..kk).map(|_| 0.05 + 3.0 * rng.unit()).collect();
        situ_case(
            e,
            &tables,
            quant,
            &format!("{cell} window-117..133"),
            &gu,
            &gamma,
            kk,
            &mut lines,
            &mut fails,
        )?;
        // a decode-like sample
        let (rows, k) = (16usize, 1152usize);
        let gu: Vec<u16> = (0..rows * 2 * k)
            .map(|i| {
                (rng.gauss(if (i / (2 * k)) % 4 == 3 { 12.0 } else { 2.0 })
                    .to_bits()
                    >> 16) as u16
            })
            .collect();
        let gamma: Vec<f32> = (0..rows).map(|_| 0.05 + 3.0 * rng.unit()).collect();
        situ_case(
            e,
            &tables,
            quant,
            &format!("{cell} gauss"),
            &gu,
            &gamma,
            k,
            &mut lines,
            &mut fails,
        )?;
    }

    // --- expert GEMMs and the whole expert block
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
            // the model's shapes: decode rows (top-8 of the experts, 1 and 8 tokens) and a prefill
            // group; for W8A8 also weights that are 97% e4m3 subnormal codes
            let variants: &[bool] = if quant == Ink2Quant::Mxfp8 {
                &[false, true]
            } else {
                &[false]
            };
            for &subnormal in variants {
                for (n, k, tag) in [(2304usize, 2304usize, "w13"), (2304, 1152, "w2")] {
                    let experts = 12usize;
                    let (_, wv, ws, wamax) = ink2_random_weights(
                        e,
                        quant,
                        experts,
                        n,
                        k,
                        0xabc0 + n as u64 + k as u64 + u64::from(subnormal),
                        subnormal,
                    )?;
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
                        // heavy-tailed rows: many act codes land in the e4m3 subnormal range
                        let x: Vec<f32> = (0..rows * k)
                            .map(|i| {
                                let v = rng.gauss(1.0 + (i / k) as f32);
                                if subnormal {
                                    reference::bf16(v * (3.0 * rng.gauss(1.0)).exp())
                                } else {
                                    v
                                }
                            })
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
                            &format!(
                                "{cell} {tag} {shape}{}",
                                if subnormal { " subnormal-weights" } else { "" }
                            ),
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
        }
        for kind in ["nvfp4", "mxfp8"] {
            expert_block_case(
                e,
                &tables,
                kind,
                &g[&format!("chain/{kind}")],
                &mut lines,
                &mut fails,
            )?;
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

/// A phase marker for the 250 ms telemetry log: `#MARK <unix seconds> <name> start|end`.
fn mark(lines: &mut Vec<String>, name: &str, edge: &str) {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let line = format!("#MARK {t:.3} {name} {edge}");
    println!("{line}");
    lines.push(line);
}

/// Microbench of the expert GEMMs and the quantizers at the model's shapes. Every repetition reads
/// a different set of experts out of `experts`, so the weights come from DRAM, not from L2. Each
/// figure is bytes (or FLOPs) over the measured time per call of back-to-back asynchronous calls;
/// nothing is subtracted. Every phase prints `#MARK` lines for the telemetry cut.
pub fn ink2_actq_bench(e: &Engine, experts: usize, reps: usize) -> Res<Vec<String>> {
    let mut lines = Vec::new();
    if !ink2_actq_built() {
        return Ok(vec![
            "ink2-actq-bench: no block-scaled target in this build".into(),
        ]);
    }
    let hidden = 2304usize;
    let ff = 1152usize;
    // warm-up: the card leaves its idle clocks during these calls
    {
        let (w, _, _, _) = ink2_random_weights(e, Ink2Quant::Nvfp4, 8, 2 * ff, hidden, 5, false)?;
        let act = quantize_on_device(e, Ink2Quant::Nvfp4, &vec![1.0f32; 8 * hidden], 8, hidden)?;
        let groups = e.ink2_groups_upload(&(0..8).collect::<Vec<u32>>(), 1)?;
        let mut out = e.alloc_u8(8 * 2 * ff * 2)?;
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_secs_f64() < 2.0 {
            for _ in 0..200 {
                e.ink2_expert_gemm_into(&w, &act, &groups, 0, &mut out, None)?;
            }
            e.stream().synchronize()?;
        }
    }
    for quant in [Ink2Quant::Nvfp4, Ink2Quant::Mxfp8] {
        for (n, k, tag) in [(2 * ff, hidden, "w13"), (hidden, ff, "w2")] {
            let (w, _, _, _) = ink2_random_weights(e, quant, experts, n, k, 77 + k as u64, false)?;
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
                    tables.push(e.ink2_groups_upload(&expert, nb)?);
                }
                let mut out = e.alloc_u8(rows * n * 2)?;
                let distinct = distinct_total as f64 / sets as f64;
                let bytes = weight_bytes(quant, n, k, 1) as f64 * distinct;
                let flops = 2.0 * rows as f64 * n as f64 * k as f64;
                let phase = format!("{quant:?}-{tag}-t{tokens}");
                mark(&mut lines, &phase, "start");
                let mut arms = Vec::new();
                let mut best_all = (f64::MAX, 0usize);
                for wpb in [1usize, 2, 4, 8] {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        e.ink2_expert_gemm_into(&w, &act, &tables[0], wpb, &mut out, None)?;
                        e.stream().synchronize()?;
                        let t0 = std::time::Instant::now();
                        for i in 0..reps {
                            e.ink2_expert_gemm_into(
                                &w,
                                &act,
                                &tables[i % sets],
                                wpb,
                                &mut out,
                                None,
                            )?;
                        }
                        e.stream().synchronize()?;
                        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
                    }
                    arms.push(format!("w{wpb}={best:.1}"));
                    if best < best_all.0 {
                        best_all = (best, wpb);
                    }
                }
                mark(&mut lines, &phase, "end");
                let (best, wpb) = best_all;
                lines.push(format!(
                    "ink2-actq-bench {quant:?} {tag} tokens={tokens} rows={rows} nb={nb}: best {best:.1} us/call at wpb={wpb}, {:.2} MB weights ({distinct:.1} experts), {:.0} GB/s, {:.2} TFLOP/s; us/call by arm: {} (best of 5 x {reps} calls, {sets} expert sets)",
                    bytes / 1e6,
                    bytes / best / 1e3,
                    flops / best / 1e6,
                    arms.join(" "),
                ));
            }
        }
        // quantizers
        let tables = e.ink2_situ_tables(4.0, 25.0)?;
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
            let phase = format!("{quant:?}-quant{}-r{rows}", if situ { "-situ" } else { "" });
            mark(&mut lines, &phase, "start");
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    if situ {
                        e.ink2_situ_quant_into(&xd, &gm, &tables, &mut out, None)?;
                    } else {
                        e.ink2_act_quant_into(&xd, &mut out)?;
                    }
                }
                e.stream().synchronize()?;
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
            }
            mark(&mut lines, &phase, "end");
            let moved =
                (rows * width * 2 + rows * (quant.value_bytes(k) + quant.scale_bytes(k))) as f64;
            lines.push(format!(
                "ink2-actq-bench {quant:?} quantize{} rows={rows} k={k}: {best:.1} us/call, {:.1} GB/s of input and output (best of 5 x {reps} calls)",
                if situ { " + SiTU" } else { "" },
                moved / best / 1e3,
            ));
        }
    }
    Ok(lines)
}
