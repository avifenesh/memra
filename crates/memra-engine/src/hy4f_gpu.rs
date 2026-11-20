//! Hy4-flash on one GPU, the first native route: the text program of the typed plan
//! (`memra_gguf::model_plan::hy_v4_flash`) over the artifact's own safetensors shards, for the
//! `hy4f-run` checkpoint runner.
//!
//! Numeric class `hy-v4-flash-source-program`: the transformers program (PR #50903 at
//! b865850d0532) computed in f32. The weight GEMMs and the attention core run on the GPU
//! (`cu/hy4f_gpu.cu`); the weights stay in their stored formats (BF16, F32, FP8 128x128 blocks,
//! NVFP4 with its per-expert `weight_scale_2`) and are never formed as rounded dequantized
//! values: codes multiply the inputs and the scales multiply the partial sums. The small
//! per-token operations (norms, RoPE, hyper-connections, router, indexer pooling and scoring,
//! Engram gathers and gates) run on the host in f64 over f32 activations, which are stored as
//! f32 between operations.
//!
//! The request is scored teacher-forced in one layer-by-layer pass: the prefill call
//! `tokens[..decode_from]`, then one decode call per later token. Every operation but attention
//! acts per token, and the GEMM and attention kernels compute each row on its own, so the pass
//! runs every position of a layer together and gives each query the keys its call would see: a
//! prefill query sees the causal prefix of the prefill, a decode query at position `t` the
//! `t + 1` cached keys. An indexed layer runs its indexer for a query whose call holds more
//! than `index_topk` keys (`kv_length <= index_topk` attends densely), over the groups that call
//! holds; a sliding layer reads its last `window` keys. Each layer's weights are read once.
//!
//! Correctness first; the tuned route (BF16 tensor-core GEMMs, a carried cache, real decode) is
//! a later step.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use memra_gguf::config::HyV4FlashConfig;
use memra_gguf::hy_v4_flash_engram as hashing;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, GroupTieOrder, HyV4AttentionKind, HyV4AttentionPlan,
    HyV4EngramPlan, HyV4ResidualPlan, HyV4WeightFormat, MlpPlan, ModelPlan, NormPlan,
    ResidualTopology, RouterPlan,
};
use memra_gguf::safetensors::StModel;
use std::sync::Arc;

pub const NUMERIC_CLASS: &str = "hy-v4-flash-source-program";

pub mod ffi {
    use std::os::raw::c_void;
    // Return codes: 0 ok, 10000 + cudaError_t, 44xxx for a refused contract (cu/hy4f_gpu.cu).
    unsafe extern "C" {
        pub fn memra_hy4f_gemm(
            x: *const c_void,
            ldx: i64,
            w: *const c_void,
            scales: *const c_void,
            grid_cols: i32,
            scale2: f32,
            fmt: i32,
            round_bf16: i32,
            y: *mut c_void,
            ldy: i64,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4f_attn(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            sinks: *const c_void,
            slots: *const c_void,
            t: i32,
            l: i32,
            h: i32,
            dk: i32,
            dv: i32,
            scale: f32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

const FMT_BF16: i32 = 0;
const FMT_F32: i32 = 1;
const FMT_FP8_BLOCK: i32 = 2;
const FMT_NVFP4: i32 = 3;
/// Keys one attention row may read (`ATTN_MAX_KEYS` in the kernel).
const ATTN_MAX_KEYS: usize = 4096;

fn ck(name: &str, rc: i32) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        r if r >= 10000 => Err(format!("{name}: CUDA error {}", r - 10000)),
        r => Err(format!("{name}: contract refused ({r})")),
    }
}

/// Wrong programs the checkpoint gate must refuse (fault injection of the gate, not runtime
/// options).
#[derive(Debug, Clone, Copy, Default)]
pub struct RedArms {
    /// The Engram memories add nothing.
    pub engram_off: bool,
    /// The indexer keeps the higher index among equal group logits.
    pub ties_reversed: bool,
    /// Every quantized weight multiplies as bf16(its dequantized value), as a BF16 loader does.
    pub dequant_bf16: bool,
}

/// A weight as stored, on the device.
pub struct DevW {
    data: CudaSlice<u8>,
    scales: Option<CudaSlice<u8>>,
    fmt: i32,
    /// output rows, input columns
    n: usize,
    k: usize,
    grid_cols: i32,
    /// `weight_scale_2` per matrix (per expert when stacked; 1.0 for other formats)
    scale2: Vec<f32>,
    experts: usize,
}

pub struct Gpu {
    st: Arc<CudaStream>,
    pub red: RedArms,
}

fn dptr<T>(s: &CudaSlice<T>, st: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr(st);
    p
}

fn dptr_mut<T>(s: &mut CudaSlice<T>, st: &CudaStream) -> u64 {
    let (p, _g) = s.device_ptr_mut(st);
    p
}

impl Gpu {
    pub fn new(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("cuda context: {e:?}"))?;
        Ok(Self {
            st: ctx.default_stream(),
            red: RedArms::default(),
        })
    }

    fn up<T: cudarc::driver::DeviceRepr>(&self, v: &[T]) -> Result<CudaSlice<T>, String> {
        self.st.clone_htod(v).map_err(|e| format!("htod: {e:?}"))
    }

    fn down<T: cudarc::driver::DeviceRepr + Default + Clone>(
        &self,
        v: &CudaSlice<T>,
    ) -> Result<Vec<T>, String> {
        let out = self.st.clone_dtoh(v).map_err(|e| format!("dtoh: {e:?}"))?;
        self.st.synchronize().map_err(|e| format!("sync: {e:?}"))?;
        Ok(out)
    }

    /// `x [m, k] @ W[e].T` in f32 (`e` the expert of a stacked weight).
    pub fn gemm(&self, x: &[f32], w: &DevW, e: usize) -> Result<Vec<f32>, String> {
        let k = w.k;
        if !x.len().is_multiple_of(k) {
            return Err(format!("gemm: input of {} values for k={k}", x.len()));
        }
        let m = x.len() / k;
        if m == 0 {
            return Ok(Vec::new());
        }
        if e >= w.experts {
            return Err(format!("gemm: expert {e} of {}", w.experts));
        }
        let (n, st) = (w.n, &self.st);
        let per = match w.fmt {
            FMT_BF16 => n * k * 2,
            FMT_F32 => n * k * 4,
            FMT_FP8_BLOCK => n * k,
            _ => n * k / 2,
        };
        let sper = if w.fmt == FMT_NVFP4 { n * k / 16 } else { 0 };
        let wp = dptr(&w.data, st) + (e * per) as u64;
        let sp = w
            .scales
            .as_ref()
            .map_or(0, |s| dptr(s, st) + (e * sper) as u64);
        let mut out = Vec::with_capacity(m * n);
        // bound the output buffer: the head is [tokens, 120832]
        let rows = (1usize << 26).div_ceil(n).clamp(1, m);
        for r0 in (0..m).step_by(rows) {
            let mm = rows.min(m - r0);
            let xd = self.up(&x[r0 * k..(r0 + mm) * k])?;
            let mut yd = st
                .alloc_zeros::<f32>(mm * n)
                .map_err(|e| format!("alloc: {e:?}"))?;
            let rc = unsafe {
                ffi::memra_hy4f_gemm(
                    dptr(&xd, st) as *const _,
                    k as i64,
                    wp as *const _,
                    sp as *const _,
                    w.grid_cols,
                    w.scale2[e],
                    w.fmt,
                    i32::from(self.red.dequant_bf16 && w.fmt >= FMT_FP8_BLOCK),
                    dptr_mut(&mut yd, st) as *mut _,
                    n as i64,
                    mm as i32,
                    n as i32,
                    k as i32,
                    st.cu_stream() as *mut _,
                )
            };
            ck("memra_hy4f_gemm", rc)?;
            out.extend(self.down(&yd)?);
        }
        Ok(out)
    }

    /// Attention over per-query key slots (`slots [t, l]`, -1 pads a row).
    #[allow(clippy::too_many_arguments)]
    fn attn(
        &self,
        q: &[f32],
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        sinks: &[f32],
        slots: &[i32],
        l: usize,
        heads: usize,
        dk: usize,
        dv: usize,
        scale: f32,
    ) -> Result<Vec<f32>, String> {
        let t = q.len() / (heads * dk);
        let st = &self.st;
        let (qd, sd, ld) = (self.up(q)?, self.up(sinks)?, self.up(slots)?);
        let mut od = st
            .alloc_zeros::<f32>(t * heads * dv)
            .map_err(|e| format!("alloc: {e:?}"))?;
        let rc = unsafe {
            ffi::memra_hy4f_attn(
                dptr(&qd, st) as *const _,
                dptr(k, st) as *const _,
                dptr(v, st) as *const _,
                dptr(&sd, st) as *const _,
                dptr(&ld, st) as *const _,
                t as i32,
                l as i32,
                heads as i32,
                dk as i32,
                dv as i32,
                scale,
                dptr_mut(&mut od, st) as *mut _,
                st.cu_stream() as *mut _,
            )
        };
        ck("memra_hy4f_attn", rc)?;
        self.down(&od)
    }
}

// ---- the checkpoint --------------------------------------------------------------------------

pub struct Ckpt {
    pub st: StModel,
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// e4m3 (fn) to f32, exactly.
pub fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 7) as f32;
    if exp == 0xF && (b & 7) == 7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// A tensor's byte range in its shard file.
pub struct FileRows {
    file: Arc<std::fs::File>,
    start: usize,
    len: usize,
}

impl FileRows {
    fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), String> {
        use std::os::unix::fs::FileExt;
        if offset + out.len() > self.len {
            return Err(format!(
                "read of {} bytes at {offset} past a {}-byte tensor",
                out.len(),
                self.len
            ));
        }
        self.file
            .read_exact_at(out, (self.start + offset) as u64)
            .map_err(|e| format!("pread: {e}"))
    }
}

impl Ckpt {
    pub fn open(dir: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            st: StModel::open(dir).map_err(|e| format!("open {}: {e}", dir.display()))?,
        })
    }

    fn raw(&self, name: &str) -> Result<(&memra_gguf::safetensors::StInfo, &[u8]), String> {
        self.st
            .raw(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))
    }

    fn shaped(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&[u8], String> {
        let (info, bytes) = self.raw(name)?;
        let got: Vec<usize> = info.shape.iter().map(|v| *v as usize).collect();
        if info.dtype != dtype || got != shape {
            return Err(format!(
                "{name}: {} {:?} in the checkpoint, the plan needs {dtype} {shape:?}",
                info.dtype, got
            ));
        }
        Ok(bytes)
    }

    /// A BF16 or F32 tensor as f32 values (exact), shape-checked.
    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let (info, _) = self.raw(name)?;
        match info.dtype.as_str() {
            "BF16" => Ok(bf16_to_f32(self.shaped(name, "BF16", shape)?)),
            "F32" => Ok(le_f32(self.shaped(name, "F32", shape)?)),
            d => Err(format!("{name}: {d} where the plan reads BF16 or F32")),
        }
    }

    /// A tensor read in place from its shard file (shape-checked), for gathers of a few rows.
    fn rows(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<FileRows, String> {
        self.shaped(name, dtype, shape)?;
        let (_, file, start, len) = self
            .st
            .raw_extent(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        Ok(FileRows { file, start, len })
    }

    fn i32s(&self, name: &str, shape: &[usize]) -> Result<Vec<i32>, String> {
        Ok(self
            .shaped(name, "I32", shape)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// A linear weight `[n, k]` on the device in the format the plan names for it.
    fn weight(
        &self,
        gpu: &Gpu,
        name: &str,
        n: usize,
        k: usize,
        format: HyV4WeightFormat,
    ) -> Result<DevW, String> {
        let (info, _) = self.raw(name)?;
        match (format, info.dtype.as_str()) {
            (HyV4WeightFormat::Bf16, "BF16") => Ok(DevW {
                data: gpu.up(self.shaped(name, "BF16", &[n, k])?)?,
                scales: None,
                fmt: FMT_BF16,
                n,
                k,
                grid_cols: 0,
                scale2: vec![1.0],
                experts: 1,
            }),
            (HyV4WeightFormat::Bf16, "F32") => Ok(DevW {
                data: gpu.up(self.shaped(name, "F32", &[n, k])?)?,
                scales: None,
                fmt: FMT_F32,
                n,
                k,
                grid_cols: 0,
                scale2: vec![1.0],
                experts: 1,
            }),
            (HyV4WeightFormat::Fp8Block { block: 128 }, "F8_E4M3") => {
                let stem = name.strip_suffix(".weight").unwrap_or(name);
                let grid = [n.div_ceil(128), k.div_ceil(128)];
                Ok(DevW {
                    data: gpu.up(self.shaped(name, "F8_E4M3", &[n, k])?)?,
                    scales: Some(gpu.up(self.shaped(
                        &format!("{stem}.weight_scale_inv"),
                        "F32",
                        &grid,
                    )?)?),
                    fmt: FMT_FP8_BLOCK,
                    n,
                    k,
                    grid_cols: grid[1] as i32,
                    scale2: vec![1.0],
                    experts: 1,
                })
            }
            (f, d) => Err(format!(
                "{name}: stored as {d}, the plan names {f:?}; no substitution is allowed"
            )),
        }
    }

    /// A stacked NVFP4 expert bank `[experts, n, k]`.
    fn experts(
        &self,
        gpu: &Gpu,
        stem: &str,
        experts: usize,
        n: usize,
        k: usize,
        group: usize,
    ) -> Result<DevW, String> {
        if group != 16 {
            return Err(format!("{stem}: NVFP4 group {group}, the kernel runs 16"));
        }
        Ok(DevW {
            data: gpu.up(self.shaped(&format!("{stem}.weight"), "U8", &[experts, n, k / 2])?)?,
            scales: Some(gpu.up(self.shaped(
                &format!("{stem}.weight_scale"),
                "F8_E4M3",
                &[experts, n, k / 16],
            )?)?),
            fmt: FMT_NVFP4,
            n,
            k,
            grid_cols: 0,
            scale2: le_f32(self.shaped(&format!("{stem}.weight_scale_2"), "F32", &[experts])?),
            experts,
        })
    }
}

// ---- host operations (f64 over f32 activations) ----------------------------------------------

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn f32s(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Weighted RMS norm per row of `weight.len()`, `weight * (x * rsqrt(mean(x^2) + eps))`.
fn rms(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
    let w = weight.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(w) {
        let inv =
            1.0 / (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / w as f64 + eps).sqrt();
        out.extend(
            row.iter()
                .zip(weight)
                .map(|(v, g)| (*g as f64 * (*v as f64 * inv)) as f32),
        );
    }
    out
}

fn norm_eps(n: NormPlan) -> f64 {
    n.epsilon as f64
}

/// Interleaved RoPE of `row` (adjacent pairs) at `position`, `inv_freq_i = 1 / base^(2i/d)`.
fn rope(row: &mut [f32], position: usize, base: f64) {
    let d = row.len();
    for i in 0..d / 2 {
        let inv = 1.0 / base.powf((2 * i) as f64 / d as f64);
        let (s, c) = (position as f64 * inv).sin_cos();
        let (x0, x1) = (row[2 * i] as f64, row[2 * i + 1] as f64);
        row[2 * i] = (x0 * c - x1 * s) as f32;
        row[2 * i + 1] = (x1 * c + x0 * s) as f32;
    }
}

/// Rows `0..n` split across threads; `f(row) -> values` fills `width` outputs a row.
fn par_rows<F>(n: usize, width: usize, f: F) -> Vec<f32>
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    let mut out = vec![0f32; n * width];
    if n == 0 {
        return out;
    }
    let threads = std::thread::available_parallelism()
        .map_or(4, |v| v.get())
        .min(16);
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (c, chunk) in out.chunks_mut(per * width.max(1)).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, row) in chunk.chunks_mut(width.max(1)).enumerate() {
                    f(c * per + i, row);
                }
            });
        }
    });
    out
}

struct Hc {
    function: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
}

/// `HYV4FlashHyperConnection`: (post [t, n], collapsed input [t, h]).
fn hc_pre(s: &[f32], hc: &Hc, res: &HyV4ResidualPlan, h: usize) -> (Vec<f64>, Vec<f32>) {
    let n = res.streams as usize;
    let t = s.len() / (n * h);
    let mut post = vec![0f64; t * n];
    let mut x = vec![0f32; t * h];
    for r in 0..t {
        let flat = &s[r * n * h..(r + 1) * n * h];
        let inv = 1.0
            / (flat.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n * h) as f64
                + res.mix_norm_epsilon as f64)
                .sqrt();
        let mix: Vec<f64> = (0..2 * n)
            .map(|j| {
                flat.iter()
                    .zip(&hc.function[j * n * h..(j + 1) * n * h])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * inv
            })
            .collect();
        let pre: Vec<f64> = (0..n)
            .map(|j| sigmoid(mix[j] * hc.scale[0] as f64 + hc.base[j] as f64) + res.epsilon as f64)
            .collect();
        for j in 0..n {
            post[r * n + j] = res.post_magnitude as f64
                * sigmoid(mix[n + j] * hc.scale[1] as f64 + hc.base[n + j] as f64)
                + res.epsilon as f64;
        }
        for d in 0..h {
            let v: f64 = (0..n).map(|j| pre[j] * flat[j * h + d] as f64).sum();
            x[r * h + d] = v as f32;
        }
    }
    (post, x)
}

/// `post * out + residual` per stream.
fn write_back(post: &[f64], out: &[f32], s: &mut [f32], n: usize, h: usize) {
    let t = out.len() / h;
    for r in 0..t {
        for j in 0..n {
            let p = post[r * n + j];
            for d in 0..h {
                let i = (r * n + j) * h + d;
                s[i] = (p * out[r * h + d] as f64 + s[i] as f64) as f32;
            }
        }
    }
}

// ---- one layer -------------------------------------------------------------------------------

/// What a call recorded: per query its recalled groups (sorted, the oracle's text), the group
/// logits it ranked, per Engram token its table rows.
#[derive(Default)]
pub struct LayerRecord {
    /// per query position: (groups sorted, logits over the call's groups)
    pub recalled: Vec<Option<(Vec<i32>, Vec<f32>)>>,
    /// per position: the Engram table rows (head offsets included)
    pub engram_rows: Option<Vec<Vec<u64>>>,
    /// MoE layers, per position: the selected experts (sorted) and the selection scores
    pub routes: Option<Vec<(Vec<i32>, Vec<f32>)>>,
}

pub struct Request<'a> {
    pub tokens: &'a [u32],
    pub decode_from: usize,
}

impl Request<'_> {
    /// Keys the call that scores position `t` holds.
    pub fn kv_length(&self, t: usize) -> usize {
        if t < self.decode_from {
            self.decode_from
        } else {
            t + 1
        }
    }
}

pub struct Runner<'a> {
    pub gpu: Gpu,
    pub ck: &'a Ckpt,
    pub plan: &'a ModelPlan,
    pub hy: &'a HyV4FlashConfig,
}

fn attention_plan(layer: &memra_gguf::model_plan::LayerPlan) -> Result<&HyV4AttentionPlan, String> {
    match &layer.attention {
        AttentionPlan::HyV4(a) => Ok(a),
        _ => Err(format!(
            "layer {} is not a Hy4-flash attention",
            layer.index
        )),
    }
}

impl Runner<'_> {
    /// Layer `il` on the streams `s` (`[t, n, h]`) of every position of the request.
    pub fn layer(&self, il: usize, req: &Request, s: &mut [f32]) -> Result<LayerRecord, String> {
        let layer = &self.plan.layers[il];
        let h = self.plan.hidden_size as usize;
        let ResidualTopology::HyV4(res) = &layer.residual else {
            return Err(format!("layer {il} is not a Hy4-flash residual"));
        };
        let n = res.streams as usize;
        let t = req.tokens.len();
        let p = format!("model.layers.{il}.");
        let mut rec = LayerRecord::default();
        if let Some(ep) = &res.engram {
            let (out, rows) = self.engram(il, ep, req, s)?;
            rec.engram_rows = Some(rows);
            if !self.gpu.red.engram_off {
                for (a, b) in s.iter_mut().zip(&out) {
                    *a = (*a as f64 + *b as f64) as f32;
                }
            }
        }
        let hc = |sub: &str| -> Result<Hc, String> {
            Ok(Hc {
                function: self
                    .ck
                    .f32s(&format!("{p}{sub}.hc_pre.hc_fn"), &[2 * n, n * h])?,
                base: self
                    .ck
                    .f32s(&format!("{p}{sub}.hc_pre.hc_base"), &[2 * n])?,
                scale: self.ck.f32s(&format!("{p}{sub}.hc_pre.hc_scale"), &[2])?,
            })
        };
        let (post, x) = hc_pre(s, &hc("hc_attn_layer")?, res, h);
        let x = rms(
            &x,
            &self.ck.f32s(&format!("{p}input_layernorm.weight"), &[h])?,
            norm_eps(layer.pre_attention_norm),
        );
        let (a, recalled) = self.attention(il, req, &x)?;
        rec.recalled = recalled;
        write_back(&post, &a, s, n, h);
        let (post, x) = hc_pre(s, &hc("hc_mlp_layer")?, res, h);
        let x = rms(
            &x,
            &self
                .ck
                .f32s(&format!("{p}post_attention_layernorm.weight"), &[h])?,
            norm_eps(layer.pre_mlp_norm),
        );
        let (m, routes) = self.mlp(il, res, &x)?;
        rec.routes = routes;
        write_back(&post, &m, s, n, h);
        debug_assert_eq!(s.len(), t * n * h);
        Ok(rec)
    }

    #[allow(clippy::type_complexity)]
    fn attention(
        &self,
        il: usize,
        req: &Request,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<Option<(Vec<i32>, Vec<f32>)>>), String> {
        let layer = &self.plan.layers[il];
        let a = attention_plan(layer)?;
        let (h, t) = (self.plan.hidden_size as usize, req.tokens.len());
        let heads = a.query_heads as usize;
        let (nope, rd, vd) = (
            a.qk_nope_head_dim as usize,
            a.qk_rope_head_dim as usize,
            a.value_head_dim as usize,
        );
        let (ql, kvl) = (a.q_lora_rank as usize, a.kv_lora_rank as usize);
        let dk = nope + rd;
        let at = |s: &str| format!("model.layers.{il}.self_attn.{s}");
        let g = &self.gpu;
        let bf = HyV4WeightFormat::Bf16;
        let pw = a.projection_weights;
        let gate = g.gemm(
            x,
            &self
                .ck
                .weight(g, &at("linear_gate.weight"), heads * vd, h, pw)?,
            0,
        )?;
        let qa = g.gemm(x, &self.ck.weight(g, &at("q_a_proj.weight"), ql, h, bf)?, 0)?;
        // `q_a_layernorm`, `kv_a_layernorm` and `post_proj_norm` are `rms_norm_eps` RMS norms
        let eps = self.hy.rms_norm_eps as f64;
        let q_resid = rms(&qa, &self.ck.f32s(&at("q_a_layernorm.weight"), &[ql])?, eps);
        let mut q = g.gemm(
            &q_resid,
            &self
                .ck
                .weight(g, &at("q_b_proj.weight"), heads * dk, ql, pw)?,
            0,
        )?;
        let ckv = g.gemm(
            x,
            &self
                .ck
                .weight(g, &at("kv_a_proj_with_mqa.weight"), kvl + rd, h, bf)?,
            0,
        )?;
        let kv_norm = self.ck.f32s(&at("kv_a_layernorm.weight"), &[kvl])?;
        let base = a.rope.base as f64;
        let mut latent = Vec::with_capacity(t * kvl);
        let mut krot = Vec::with_capacity(t * rd);
        for r in 0..t {
            let row = &ckv[r * (kvl + rd)..(r + 1) * (kvl + rd)];
            latent.extend(rms(&row[..kvl], &kv_norm, eps));
            let mut kr = row[kvl..].to_vec();
            rope(&mut kr, r, base);
            krot.extend(kr);
            for hd in 0..heads {
                let o = (r * heads + hd) * dk + nope;
                rope(&mut q[o..o + rd], r, base);
            }
        }
        let kvb = g.gemm(
            &latent,
            &self
                .ck
                .weight(g, &at("kv_b_proj.weight"), heads * (nope + vd), kvl, bf)?,
            0,
        )?;
        let mut kfull = vec![0f32; t * heads * dk];
        let mut vfull = vec![0f32; t * heads * vd];
        for r in 0..t {
            for hd in 0..heads {
                let src = &kvb[(r * heads + hd) * (nope + vd)..(r * heads + hd + 1) * (nope + vd)];
                let ko = (r * heads + hd) * dk;
                kfull[ko..ko + nope].copy_from_slice(&src[..nope]);
                kfull[ko + nope..ko + dk].copy_from_slice(&krot[r * rd..(r + 1) * rd]);
                let vo = (r * heads + hd) * vd;
                vfull[vo..vo + vd].copy_from_slice(&src[nope..]);
            }
        }
        // the keys each query reads
        let mut recalled: Vec<Option<(Vec<i32>, Vec<f32>)>> = vec![None; t];
        let (slots, l) = match a.kind {
            HyV4AttentionKind::Sliding { window } => {
                let w = window as usize;
                let mut slots = vec![-1i32; t * w];
                for r in 0..t {
                    let start = (r + 1).saturating_sub(w);
                    for (j, k) in (start..=r).enumerate() {
                        slots[r * w + j] = k as i32;
                    }
                }
                (slots, w)
            }
            HyV4AttentionKind::Indexed(ip) => {
                let sel = self.indexer(il, &ip, req, x, &q_resid)?;
                let l = (ip.top_k as usize).max(
                    sel.iter()
                        .flatten()
                        .map(|v| v.0.len() * ip.pool as usize)
                        .max()
                        .unwrap_or(0),
                );
                if l > ATTN_MAX_KEYS {
                    return Err(format!(
                        "layer {il}: {l} keys a row, the kernel reads {ATTN_MAX_KEYS}"
                    ));
                }
                let pool = ip.pool as usize;
                let mut slots = vec![-1i32; t * l];
                for r in 0..t {
                    let row = &mut slots[r * l..(r + 1) * l];
                    match &sel[r] {
                        None => {
                            for (j, k) in (0..=r).enumerate() {
                                row[j] = k as i32;
                            }
                        }
                        Some((groups, _)) => {
                            let kv = req.kv_length(r);
                            let mut keep = vec![false; r + 1];
                            for gi in groups {
                                for o in 0..pool {
                                    let k = (*gi as usize * pool + o).min(kv - 1);
                                    if k <= r {
                                        keep[k] = true;
                                    }
                                }
                            }
                            for (j, k) in keep
                                .iter()
                                .enumerate()
                                .filter(|(_, v)| **v)
                                .map(|(k, _)| k)
                                .enumerate()
                            {
                                row[j] = k as i32;
                            }
                        }
                    }
                }
                recalled = sel;
                (slots, l)
            }
        };
        let kd = g.up(&kfull)?;
        let vdv = g.up(&vfull)?;
        let mut sinks = self.ck.f32s(&at("learnable_sink_param"), &[heads])?;
        if a.sink == memra_gguf::model_plan::TensorPresence::Absent {
            sinks = vec![f32::NEG_INFINITY; heads];
        }
        let scale = (dk as f64).powf(-0.5) as f32;
        let mut o = Vec::with_capacity(t * heads * vd);
        let chunk = 256;
        for r0 in (0..t).step_by(chunk) {
            let rr = chunk.min(t - r0);
            o.extend(g.attn(
                &q[r0 * heads * dk..(r0 + rr) * heads * dk],
                &kd,
                &vdv,
                &sinks,
                &slots[r0 * l..(r0 + rr) * l],
                l,
                heads,
                dk,
                vd,
                scale,
            )?);
        }
        for (v, gt) in o.iter_mut().zip(&gate) {
            *v = (*v as f64 * sigmoid(*gt as f64)) as f32;
        }
        let out = g.gemm(
            &o,
            &self.ck.weight(g, &at("o_proj.weight"), h, heads * vd, pw)?,
            0,
        )?;
        let norm = self.ck.f32s(&at("post_proj_norm.weight"), &[h])?;
        Ok((rms(&out, &norm, norm_eps(a.post_projection_norm)), recalled))
    }

    /// `HYV4FlashCompressedIndexer` for every query: `None` where the query's call attends
    /// densely, else its recalled groups (the top-k plus its own group, sorted) and the group
    /// logits of its call.
    #[allow(clippy::type_complexity)]
    fn indexer(
        &self,
        il: usize,
        ip: &memra_gguf::model_plan::IndexPoolPlan,
        req: &Request,
        x: &[f32],
        q_resid: &[f32],
    ) -> Result<Vec<Option<(Vec<i32>, Vec<f32>)>>, String> {
        let t = req.tokens.len();
        let (ih, id, rd, pool) = (
            ip.heads as usize,
            ip.head_dim as usize,
            ip.rope_head_dim as usize,
            ip.pool as usize,
        );
        let h = self.plan.hidden_size as usize;
        let ql = self.hy.q_lora_rank as usize;
        let a = attention_plan(&self.plan.layers[il])?;
        let base = a.rope.base as f64;
        let eps = self.hy.rms_norm_eps as f64;
        let c = |s: &str| format!("model.layers.{il}.self_attn.compressed_indexer.{s}");
        let g = &self.gpu;
        let bf = HyV4WeightFormat::Bf16;
        if (0..t).all(|r| req.kv_length(r) <= ip.top_k as usize) {
            return Ok(vec![None; t]);
        }
        let tk = rms(
            &g.gemm(
                x,
                &self
                    .ck
                    .weight(g, &c("token_key_projection.weight"), id, h, bf)?,
                0,
            )?,
            &self.ck.f32s(&c("token_key_norm.weight"), &[id])?,
            eps,
        );
        let kvp = f64s(&self.ck.f32s(&c("key_value_projection.weight"), &[id, id])?);
        let kgp = f64s(&self.ck.f32s(&c("key_gate_projection.weight"), &[id, id])?);
        let ape = f64s(&self.ck.f32s(&c("intra_group_ape"), &[pool, id])?);
        let key_norm = self.ck.f32s(&c("key_norm.weight"), &[id])?;
        // pooled group keys of every complete group, rotated at the group's first position
        let groups = t / pool;
        let gk = par_rows(groups, id, |gi, out| {
            let rows: Vec<&[f32]> = (0..pool)
                .map(|s| &tk[(gi * pool + s) * id..(gi * pool + s + 1) * id])
                .collect();
            let lin = |w: &[f64], x: &[f32], o: usize| -> f64 {
                x.iter()
                    .zip(&w[o * id..(o + 1) * id])
                    .map(|(a, b)| *a as f64 * b)
                    .sum()
            };
            let mut pooled = vec![0f64; id];
            for d in 0..id {
                let gates: Vec<f64> = (0..pool)
                    .map(|s| lin(&kgp, rows[s], d) + ape[s * id + d])
                    .collect();
                let m = gates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = gates.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = e.iter().sum();
                pooled[d] = (0..pool).map(|s| lin(&kvp, rows[s], d) * (e[s] / z)).sum();
            }
            let pf = f32s(&pooled);
            let mut k = rms(&pf, &key_norm, eps);
            rope(&mut k[id - rd..], gi * pool, base);
            out.copy_from_slice(&k);
        });
        let mut qi = g.gemm(
            q_resid,
            &self
                .ck
                .weight(g, &c("query_projection.weight"), ih * id, ql, bf)?,
            0,
        )?;
        for r in 0..t {
            for hd in 0..ih {
                let o = (r * ih + hd) * id + id - rd;
                rope(&mut qi[o..o + rd], r, base);
            }
        }
        let hw = g.gemm(
            x,
            &self
                .ck
                .weight(g, &c("head_weight_projection.weight"), ih, h, bf)?,
            0,
        )?;
        let hscale = (ih as f64).powf(-0.5) * (id as f64).powf(-0.5);
        let ties = if self.gpu.red.ties_reversed {
            None
        } else {
            Some(ip.ties)
        };
        let k_groups = (ip.top_k / ip.pool) as usize - 1;
        let out: Vec<Option<(Vec<i32>, Vec<f32>)>> = {
            let rows: Vec<usize> = (0..t).collect();
            let mut res = vec![None; t];
            let threads = 16usize;
            let per = t.div_ceil(threads);
            std::thread::scope(|s| {
                for (ci, chunk) in res.chunks_mut(per).enumerate() {
                    let (gk, qi, hw, rows) = (&gk, &qi, &hw, &rows);
                    s.spawn(move || {
                        for (j, slot) in chunk.iter_mut().enumerate() {
                            let r = rows[ci * per + j];
                            let kv = req.kv_length(r);
                            if kv <= ip.top_k as usize {
                                continue;
                            }
                            let ng = kv / pool;
                            let own = r / pool;
                            let mut logits = vec![f64::NEG_INFINITY; ng];
                            for (gi, lg) in logits.iter_mut().enumerate().take(own.min(ng)) {
                                let key = &gk[gi * id..(gi + 1) * id];
                                let mut acc = 0f64;
                                for hd in 0..ih {
                                    let qrow = &qi[(r * ih + hd) * id..(r * ih + hd + 1) * id];
                                    let dot: f64 = qrow
                                        .iter()
                                        .zip(key)
                                        .map(|(a, b)| *a as f64 * *b as f64)
                                        .sum();
                                    acc += hw[r * ih + hd] as f64 * hscale * dot.max(0.0);
                                }
                                *lg = acc;
                            }
                            // the program's scores are F32 (`.float()`); the pick reads them
                            let logits: Vec<f64> =
                                logits.iter().map(|v| *v as f32 as f64).collect();
                            let kk = k_groups.min(ng);
                            let mut picked = match ties {
                                Some(GroupTieOrder::LowerIndexFirst) => {
                                    memra_reference::hy_v4_flash::group_top_k(
                                        &logits,
                                        kk,
                                        GroupTieOrder::LowerIndexFirst,
                                    )
                                }
                                None => {
                                    let mut order: Vec<usize> = (0..ng).collect();
                                    order.sort_by(|a, b| {
                                        logits[*b].total_cmp(&logits[*a]).then(b.cmp(a))
                                    });
                                    order.truncate(kk);
                                    order
                                }
                            };
                            picked.push(own);
                            let mut sel: Vec<i32> = picked.iter().map(|v| *v as i32).collect();
                            sel.sort_unstable();
                            *slot = Some((sel, f32s(&logits)));
                        }
                    });
                }
            });
            res
        };
        Ok(out)
    }

    #[allow(clippy::type_complexity)]
    fn mlp(
        &self,
        il: usize,
        res: &HyV4ResidualPlan,
        x: &[f32],
    ) -> Result<(Vec<f32>, Option<Vec<(Vec<i32>, Vec<f32>)>>), String> {
        let layer = &self.plan.layers[il];
        let h = self.plan.hidden_size as usize;
        let t = x.len() / h;
        let m = |s: &str| format!("model.layers.{il}.mlp.{s}");
        let g = &self.gpu;
        let bf = HyV4WeightFormat::Bf16;
        let swiglu = |gate: &[f32], up: &[f32], limit: Option<f64>| -> Vec<f32> {
            gate.iter()
                .zip(up)
                .map(|(a, b)| {
                    let (mut a, mut b) = (*a as f64, *b as f64);
                    if let Some(l) = limit {
                        a = a.min(l);
                        b = b.clamp(-l, l);
                    }
                    (silu(a) * b) as f32
                })
                .collect()
        };
        match &layer.mlp {
            MlpPlan::Dense(d) => {
                let inter = d.intermediate_size as usize;
                if d.activation != ActivationPlan::Silu {
                    return Err(format!("layer {il}: dense activation {:?}", d.activation));
                }
                let ga = g.gemm(
                    x,
                    &self.ck.weight(g, &m("gate_proj.weight"), inter, h, bf)?,
                    0,
                )?;
                let up = g.gemm(
                    x,
                    &self.ck.weight(g, &m("up_proj.weight"), inter, h, bf)?,
                    0,
                )?;
                let y = g.gemm(
                    &swiglu(&ga, &up, None),
                    &self.ck.weight(g, &m("down_proj.weight"), h, inter, bf)?,
                    0,
                )?;
                Ok((
                    rms(
                        &y,
                        &self.ck.f32s(&m("post_fc2_norm.weight"), &[h])?,
                        norm_eps(res.mlp_output_norm),
                    ),
                    None,
                ))
            }
            MlpPlan::LatentMoe(lm) => {
                let e = lm.moe.expert_count as usize;
                let topk = lm.moe.experts_per_token as usize;
                let inter = lm.moe.expert_intermediate_size as usize;
                let lat = lm.latent_size as usize;
                let RouterPlan::Sigmoid {
                    normalize_selected,
                    scaling_factor,
                    selection_bias,
                } = lm.moe.router
                else {
                    return Err(format!("layer {il}: router {:?}", lm.moe.router));
                };
                let ActivationPlan::SwiGluPreClamped { limit } = lm.moe.activation else {
                    return Err(format!(
                        "layer {il}: expert activation {:?}",
                        lm.moe.activation
                    ));
                };
                let HyV4WeightFormat::Nvfp4W4A16 { group } = lm.expert_weights else {
                    return Err(format!("layer {il}: experts {:?}", lm.expert_weights));
                };
                let logits = g.gemm(x, &self.ck.weight(g, &m("gate.weight"), e, h, bf)?, 0)?;
                let bias = if selection_bias {
                    self.ck.f32s(&m("gate.e_score_correction_bias"), &[e])?
                } else {
                    vec![0.0; e]
                };
                let mut routes: Vec<Vec<(usize, f64)>> = Vec::with_capacity(t);
                let mut record = Vec::with_capacity(t);
                for r in 0..t {
                    // the program's scores are F32 (`sigmoid` of the F32 logits, plus the F32
                    // bias); the pick reads them
                    let scores: Vec<f64> = logits[r * e..(r + 1) * e]
                        .iter()
                        .map(|v| sigmoid(*v as f64) as f32 as f64)
                        .collect();
                    let choice: Vec<f64> = scores
                        .iter()
                        .zip(&bias)
                        .map(|(s, b)| (s + *b as f64) as f32 as f64)
                        .collect();
                    let sel = memra_reference::hy_v4_flash::group_top_k(
                        &choice,
                        topk,
                        GroupTieOrder::LowerIndexFirst,
                    );
                    let mut picked: Vec<i32> = sel.iter().map(|i| *i as i32).collect();
                    picked.sort_unstable();
                    record.push((picked, f32s(&choice)));
                    let mut w: Vec<f64> = sel.iter().map(|i| scores[*i]).collect();
                    if normalize_selected {
                        let z: f64 = w.iter().sum::<f64>() + 1e-20;
                        w.iter_mut().for_each(|v| *v /= z);
                    }
                    w.iter_mut().for_each(|v| *v *= scaling_factor as f64);
                    routes.push(sel.into_iter().zip(w).collect());
                }
                let routed_in = g.gemm(
                    x,
                    &self
                        .ck
                        .weight(g, &m("routed_expert_down_proj.weight"), lat, h, bf)?,
                    0,
                )?;
                let gate_w =
                    self.ck
                        .experts(g, &m("experts.gate_proj"), e, inter, lat, group as usize)?;
                let up_w =
                    self.ck
                        .experts(g, &m("experts.up_proj"), e, inter, lat, group as usize)?;
                let down_w =
                    self.ck
                        .experts(g, &m("experts.down_proj"), e, lat, inter, group as usize)?;
                let mut acc = vec![0f64; t * lat];
                for ex in 0..e {
                    let hits: Vec<(usize, f64)> = routes
                        .iter()
                        .enumerate()
                        .filter_map(|(r, rt)| {
                            rt.iter().find(|(i, _)| *i == ex).map(|(_, w)| (r, *w))
                        })
                        .collect();
                    if hits.is_empty() {
                        continue;
                    }
                    let mut xin = Vec::with_capacity(hits.len() * lat);
                    for (r, _) in &hits {
                        xin.extend_from_slice(&routed_in[r * lat..(r + 1) * lat]);
                    }
                    let ga = g.gemm(&xin, &gate_w, ex)?;
                    let up = g.gemm(&xin, &up_w, ex)?;
                    let y = g.gemm(&swiglu(&ga, &up, Some(limit as f64)), &down_w, ex)?;
                    for (j, (r, w)) in hits.iter().enumerate() {
                        for d in 0..lat {
                            // `F.linear(..) * weight`, then `index_add_`
                            acc[r * lat + d] += y[j * lat + d] as f64 * w;
                        }
                    }
                }
                drop((gate_w, up_w, down_w));
                let routed = g.gemm(
                    &f32s(&acc),
                    &self
                        .ck
                        .weight(g, &m("routed_expert_up_proj.weight"), h, lat, bf)?,
                    0,
                )?;
                let sh = lm
                    .moe
                    .shared
                    .as_ref()
                    .ok_or(format!("layer {il}: no shared expert"))?
                    .intermediate_size as usize;
                let sg = g.gemm(
                    x,
                    &self
                        .ck
                        .weight(g, &m("shared_experts.gate_proj.weight"), sh, h, bf)?,
                    0,
                )?;
                let su = g.gemm(
                    x,
                    &self
                        .ck
                        .weight(g, &m("shared_experts.up_proj.weight"), sh, h, bf)?,
                    0,
                )?;
                let sd = g.gemm(
                    &swiglu(&sg, &su, None),
                    &self
                        .ck
                        .weight(g, &m("shared_experts.down_proj.weight"), h, sh, bf)?,
                    0,
                )?;
                let sum: Vec<f32> = routed
                    .iter()
                    .zip(&sd)
                    .map(|(a, b)| (*a as f64 + *b as f64) as f32)
                    .collect();
                Ok((
                    rms(
                        &sum,
                        &self.ck.f32s(&m("combine_output_norm.weight"), &[h])?,
                        norm_eps(res.mlp_output_norm),
                    ),
                    Some(record),
                ))
            }
            other => Err(format!("layer {il}: MLP {other:?}")),
        }
    }

    /// `HYV4FlashEngramLayer` on every position: the residual to add and each position's table
    /// rows.
    #[allow(clippy::type_complexity)]
    fn engram(
        &self,
        il: usize,
        ep: &HyV4EngramPlan,
        req: &Request,
        s: &[f32],
    ) -> Result<(Vec<f32>, Vec<Vec<u64>>), String> {
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let t = req.tokens.len();
        let (heads, dim) = (ep.heads as usize, ep.head_dim as usize);
        let width = heads * dim;
        let e = |s: &str| format!("model.layers.{il}.engram_layer.{s}");
        let slot = self
            .hy
            .engram_slot(il as u32)
            .ok_or(format!("layer {il} carries no Engram in the config"))?;
        let moduli = &self.hy.engram_head_vocab_sizes[slot];
        let lookup = self
            .ck
            .i32s(&e("lookup_table"), &[ep.lookup_table_size as usize])?;
        let vocab = hashing::compressed_vocab(ep, &lookup)?;
        let multipliers = hashing::hash_multipliers(ep, vocab)?;
        let offsets = hashing::head_offsets(moduli);
        let positions: Vec<u32> = (0..t as u32).collect();
        let ngrams = hashing::ngram_tokens(ep, req.tokens, &positions);
        let hashes = hashing::hash_ids(ep, &lookup, &multipliers, moduli, &ngrams)?;
        let rows: Vec<Vec<u64>> = hashes
            .chunks(heads)
            .map(|r| r.iter().zip(&offsets).map(|(a, b)| a + b).collect())
            .collect();
        let mask = hashing::engram_mask(ep, req.tokens);
        let rows_total = ep.table_rows as usize;
        // The ~16.5 GB table stays on disk: each row and its scale are read with `pread`, so the
        // gather never faults the mapping in (fault readahead pulls far more than a row).
        let table = self.ck.rows(
            &e("multi_head_embedding.embedding.weight"),
            "F8_E4M3",
            &[rows_total, dim],
        )?;
        let scales = self.ck.rows(
            &e("multi_head_embedding.embedding.weight_scale"),
            "F32",
            &[rows_total],
        )?;
        let eps = ep.norm_epsilon.value() as f64;
        let head_norms: Vec<Vec<f32>> = (0..heads)
            .map(|k| {
                self.ck
                    .f32s(&e(&format!("lookup_per_head_norms.{k}.weight")), &[dim])
            })
            .collect::<Result<_, _>>()?;
        let mut emb = Vec::with_capacity(t * width);
        let (mut code, mut sb) = (vec![0u8; dim], [0u8; 4]);
        for row in &rows {
            for (k, &ri) in row.iter().enumerate() {
                let ri = ri as usize;
                if ri >= rows_total {
                    return Err(format!(
                        "layer {il}: Engram row {ri} past the table's {rows_total}"
                    ));
                }
                table.read(ri * dim, &mut code)?;
                scales.read(ri * 4, &mut sb)?;
                let sc = f32::from_le_bytes(sb);
                // the stored row, `f32(e4m3 * row scale)`
                let v: Vec<f32> = code.iter().map(|c| e4m3(*c) * sc).collect();
                emb.extend(rms(&v, &head_norms[k], eps));
            }
        }
        let g = &self.gpu;
        let bf = HyV4WeightFormat::Bf16;
        let linear = |name: &str| -> Result<Vec<f32>, String> {
            let y = g.gemm(
                &emb,
                &self
                    .ck
                    .weight(g, &e(&format!("{name}.weight")), h, width, bf)?,
                0,
            )?;
            let b = self.ck.f32s(&e(&format!("{name}.bias")), &[h])?;
            Ok(y.chunks(h)
                .flat_map(|r| {
                    r.iter()
                        .zip(&b)
                        .map(|(a, c)| (*a as f64 + *c as f64) as f32)
                        .collect::<Vec<_>>()
                })
                .collect())
        };
        let value = linear("value_proj")?;
        let ones = vec![1f32; h];
        let pre_gamma = self.ck.f32s(&e("unified_pre_gamma"), &[h])?;
        let main: Vec<f32> = rms(&value, &ones, eps)
            .iter()
            .enumerate()
            .map(|(i, v)| (pre_gamma[i % h] as f64 * *v as f64) as f32)
            .collect();
        let mut out = vec![0f32; t * n * h];
        for hc in 0..n {
            let key = rms(
                &linear(&format!("key_projs.{hc}"))?,
                &self.ck.f32s(&e(&format!("norm_key.{hc}.weight")), &[h])?,
                eps,
            );
            let qn = self.ck.f32s(&e(&format!("norm_query.{hc}.weight")), &[h])?;
            let scale = self
                .ck
                .f32s(&e(&format!("unified_per_hc_scales.{hc}")), &[h])?;
            for r in 0..t {
                let stream = &s[(r * n + hc) * h..(r * n + hc + 1) * h];
                let query = rms(stream, &qn, eps);
                let dot: f64 = key[r * h..(r + 1) * h]
                    .iter()
                    .zip(&query)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>()
                    * (h as f64).powf(-0.5);
                let sign = if dot > 0.0 {
                    1.0
                } else if dot < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let gate = sigmoid(dot.abs().max(1e-6).sqrt() * sign);
                if !mask[r] {
                    continue;
                }
                for d in 0..h {
                    let mv = main[r * h + d] as f64;
                    out[(r * n + hc) * h + d] = (gate * mv + scale[d] as f64 * mv) as f32;
                }
            }
        }
        Ok((out, rows))
    }

    /// The embedding of every position on every stream (`[t, n, h]`).
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let h = self.plan.hidden_size as usize;
        let v = self.plan.vocab_size as usize;
        let n = self.hy.hc_mult as usize;
        let table = self
            .ck
            .shaped("model.embed_tokens.weight", "BF16", &[v, h])?;
        let mut s = Vec::with_capacity(tokens.len() * n * h);
        for &tk in tokens {
            let tk = tk as usize;
            if tk >= v {
                return Err(format!("token {tk} past the vocabulary {v}"));
            }
            let row = bf16_to_f32(&table[tk * h * 2..(tk + 1) * h * 2]);
            for _ in 0..n {
                s.extend_from_slice(&row);
            }
        }
        Ok(s)
    }

    /// `hc_head`, `model.norm` and the F32 head: logits `[t, vocab]`.
    pub fn head(&self, s: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.plan.hidden_size as usize;
        let v = self.plan.vocab_size as usize;
        let n = self.hy.hc_mult as usize;
        let t = s.len() / (n * h);
        let numerics = self.plan.hy_v4_flash.ok_or("not a hy_v4_flash plan")?;
        if !numerics.output_projection_f32 {
            return Err("the Hy4-flash head is computed in F32".into());
        }
        let fn_ = self.ck.f32s("model.hc_head.hc_head_fn", &[n, n * h])?;
        let base = self.ck.f32s("model.hc_head.hc_head_base", &[n])?;
        let scale = self.ck.f32s("model.hc_head.hc_head_scale", &[1])?;
        let mix_eps = self.hy.rms_norm_eps as f64;
        let mut x = vec![0f32; t * h];
        for r in 0..t {
            let flat = &s[r * n * h..(r + 1) * n * h];
            let inv = 1.0
                / (flat.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / (n * h) as f64
                    + mix_eps)
                    .sqrt();
            let pre: Vec<f64> = (0..n)
                .map(|j| {
                    let mix: f64 = flat
                        .iter()
                        .zip(&fn_[j * n * h..(j + 1) * n * h])
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * inv;
                    sigmoid(mix * scale[0] as f64 + base[j] as f64) + numerics.exit_epsilon as f64
                })
                .collect();
            for d in 0..h {
                x[r * h + d] = (0..n).map(|j| pre[j] * flat[j * h + d] as f64).sum::<f64>() as f32;
            }
        }
        let x = rms(
            &x,
            &self.ck.f32s("model.norm.weight", &[h])?,
            norm_eps(self.plan.output_norm),
        );
        let g = &self.gpu;
        g.gemm(
            &x,
            &self
                .ck
                .weight(g, "lm_head.weight", v, h, HyV4WeightFormat::Bf16)?,
            0,
        )
    }
}
