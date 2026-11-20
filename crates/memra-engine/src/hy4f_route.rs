//! Hy4-flash on one GPU with the weights resident and a carried cache: the route the `hy4f-gpu`
//! binary runs (`cu/hy4f_route.cu`).
//!
//! Numeric class `hy-v4-flash-bf16x2-activations`: every weight GEMM carries its input as two
//! BF16 terms (`hi = bf16(x)`, `lo = bf16(x - hi)`, 16 significant bits) and runs both on the
//! tensor cores with F32 accumulation, the weight as the artifact stores it (BF16;
//! FP8 E4M3 codes with the 128x128 block scale on each 128-input partial sum; NVFP4 codes times
//! their E4M3 group scale, exact in BF16, with the expert's `weight_scale_2` on the finished
//! sum). The program's F32 islands stay F32: the router logits and the head are F32 GEMMs over
//! F32 inputs (PR 7's `memra_hy4f_gemm`), and the hyper-connection streams, norms, RoPE, softmax,
//! router scores, indexer pooling and group logits and the Engram gates are F32 kernels.
//!
//! The weights are uploaded once, in their stored formats (about 87 GiB). The Engram tables stay
//! on disk: each call hashes its n-grams on the host (integers, the step 5 code), reads each row
//! and its scale with `pread`, and the GPU decodes and normalizes them. The cache carries, per
//! layer, the latent and the rope key (576 F32 values a token; the last 512 positions in a ring
//! on sliding layers) and, on the indexed layers, the indexer's token key (128 F32) and the
//! pooled key of every closed group (128 F32), pooled when the group's fourth key arrives.
//!
//! Why two BF16 terms: one position of the `long` checkpoint request (756) amplifies a
//! perturbation of the linear inputs about a thousandfold between layers 17 and 39 in every
//! backend measured. With single BF16 inputs this route lands 13.3 logits from the f64 oracle
//! there and moves 16 argmaxes of 2,112 (the pinned program's own BF16 backend: 18.9 and 51), so
//! no BF16-input program can hold the argmax at every position
//! (research/modelplan-onboarding-hy4flash-20261119/gpu/).
//!
//! A call scores tokens `p0..p0 + t` against everything cached before them: a prefill call with
//! `t` rows, a decode call with one. Every kernel computes a row on its own in an order that
//! does not depend on the other rows of the call, so a token gets the same bits in a prefill as
//! in a decode call.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr};
use memra_gguf::config::HyV4FlashConfig;
use memra_gguf::hy_v4_flash_engram as hashing;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, HyV4AttentionKind, HyV4AttentionPlan, HyV4EngramPlan,
    HyV4ResidualPlan, HyV4WeightFormat, IndexPoolPlan, MlpPlan, ModelPlan, ResidualTopology,
    RouterPlan,
};
use memra_gguf::safetensors::StModel;
use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub const NUMERIC_CLASS: &str = "hy-v4-flash-bf16x2-activations";

#[allow(clippy::too_many_arguments)]
mod ffi {
    use std::os::raw::c_void;
    // Return codes: 0 ok, 10000 + cudaError_t, 45xxx for a refused contract (cu/hy4f_route.cu).
    unsafe extern "C" {
        pub fn memra_hy4r_gemm(
            x: *const f32,
            ldx: i64,
            xbs: i64,
            w: *const c_void,
            wbs: i64,
            sc: *const c_void,
            scbs: i64,
            s2: *const f32,
            fmt: i32,
            grid_cols: i32,
            y: *mut f32,
            ldy: i64,
            ybs: i64,
            m: i32,
            n: i32,
            k: i32,
            batch: i32,
            tiles: *const i32,
            ntiles: i32,
            red_dq: i32,
            small: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_rms(
            x: *const f32,
            ldx: i64,
            w: *const f32,
            y: *mut f32,
            ldy: i64,
            rows: i32,
            d: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_rope(
            x: *mut f32,
            ldrow: i64,
            ldhead: i64,
            rows: i32,
            heads: i32,
            off: i32,
            half: i32,
            cs: *const c_void,
            posrow: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_rope_table(
            out: *mut c_void,
            pos0: i32,
            rows: i32,
            half: i32,
            inv: *const f64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_hc(
            s: *const f32,
            fun: *const f32,
            base: *const f32,
            scale: *const f32,
            rows: i32,
            nstream: i32,
            h: i32,
            fns: i32,
            mix_eps: f32,
            eps: f32,
            mag: f32,
            post: *mut f32,
            x: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_write_back(
            s: *mut f32,
            post: *const f32,
            a: *const f32,
            rows: i32,
            nstream: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_act(
            g: *const f32,
            u: *const f32,
            out: *mut f32,
            n: i64,
            lim: f32,
            mode: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_add(
            y: *mut f32,
            a: *const f32,
            b: *const f32,
            bias: *const f32,
            n: i64,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_copy(
            src: *const f32,
            lds: i64,
            dst: *mut c_void,
            ldd: i64,
            rows: i32,
            cols: i32,
            to_bf16: i32,
            dstrow: *const i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_gather_bf(
            src: *const c_void,
            lds: i64,
            rows: *const i32,
            dst: *mut c_void,
            ldd: i64,
            n: i32,
            words: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_embed(
            table: *const c_void,
            tok: *const i32,
            s: *mut f32,
            rows: i32,
            nstream: i32,
            h: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_topk(
            vals: *const f32,
            ldv: i64,
            rows: i32,
            n: i32,
            k: i32,
            reverse_ties: i32,
            out: *mut i32,
            ldo: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_router_scores(
            logits: *const f32,
            bias: *const f32,
            scores: *mut f32,
            choice: *mut f32,
            rows: i32,
            e: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_router_weights(
            scores: *const f32,
            rows: i32,
            e: i32,
            picks: *const i32,
            k: i32,
            normalize: i32,
            scaling: f32,
            w: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_moe_group(
            picks: *const i32,
            rows: i32,
            k: i32,
            ne: i32,
            bm: i32,
            max_tiles: i32,
            x: *const f32,
            d: i32,
            counts: *mut i32,
            cursor: *mut i32,
            tiles: *mut i32,
            rowmap: *mut i32,
            xg: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_moe_combine(
            y: *const f32,
            d: i32,
            rowmap: *const i32,
            w: *const f32,
            rows: i32,
            k: i32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_attn(
            q: *const f32,
            kv: *const c_void,
            list: *const i32,
            ldl: i32,
            lens: *const i32,
            qpos: *const i32,
            mode: i32,
            window: i32,
            ring: i32,
            base: i32,
            rows: i32,
            h: i32,
            c: i32,
            scale: f32,
            sinks: *const f32,
            po: *mut f32,
            pml: *mut f32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_pool(
            tk: *const c_void,
            dim: i32,
            pool: i32,
            kvp: *const f32,
            kgp: *const f32,
            ape: *const f32,
            key_norm: *const f32,
            eps: f32,
            rd: i32,
            cs: *const c_void,
            cs_pos0: i32,
            g0: i32,
            n: i32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_index_logits(
            qi: *const f32,
            hw: *const f32,
            gk: *const f32,
            qpos: *const i32,
            rows: i32,
            heads: i32,
            dim: i32,
            pool: i32,
            ng: i32,
            hscale: f32,
            out: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_recall(
            picks: *const i32,
            kp: i32,
            qpos: *const i32,
            rows: i32,
            pool: i32,
            sel: *mut i32,
            list: *mut i32,
            ldl: i32,
            lens: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_recall_forced(
            sel: *const i32,
            ks: i32,
            qpos: *const i32,
            rows: i32,
            pool: i32,
            list: *mut i32,
            ldl: i32,
            lens: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_engram_rows(
            codes: *const c_void,
            scales: *const f32,
            norms: *const f32,
            rows: i32,
            heads: i32,
            dim: i32,
            eps: f32,
            emb: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_engram_mix(
            s: *mut f32,
            keyraw: *const f32,
            main: *const f32,
            norm_key: *const f32,
            norm_query: *const f32,
            scales: *const f32,
            mask: *const c_void,
            rows: i32,
            nstream: i32,
            h: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_h2d(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_d2h(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_memset(
            dst: *mut c_void,
            v: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_sync(stream: *mut c_void) -> i32;
    }
}

use crate::hy4f_gpu::ffi::memra_hy4f_gemm;

const FMT_BF16: i32 = 0;
const FMT_FP8: i32 = 2;
const FMT_NVFP4: i32 = 3;
/// Queries one attention launch covers (bounds the chunk partials).
const ATTN_SLAB: usize = 256;
/// Keys one attention chunk holds (`AT_CHUNK` in the kernel).
const ATTN_CHUNK: usize = 256;
/// Rows of the head GEMM per launch.
const HEAD_ROWS: usize = 256;

fn ck(name: &str, rc: i32) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        r if r >= 10000 => Err(format!("{name}: CUDA error {}", r - 10000)),
        r => Err(format!("{name}: contract refused ({r})")),
    }
}

macro_rules! cu {
    ($f:ident ( $($a:expr),* $(,)? )) => {
        ck(stringify!($f), unsafe { ffi::$f($($a),*) })
    };
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
    /// A decode call writes its latent and rope key one cache row late (the query reads the row
    /// its position names, which holds no key yet).
    pub kv_off_by_one: bool,
    /// The indexer pools a group when its third key arrives instead of its fourth.
    pub pool_early: bool,
}

// ---- device memory ----------------------------------------------------------------------------

pub struct DBuf {
    _s: CudaSlice<u8>,
    p: u64,
    bytes: usize,
    /// the device's count of live bytes, decremented when the buffer is freed
    live: Arc<AtomicUsize>,
}

impl Drop for DBuf {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

impl DBuf {
    fn f(&self) -> *mut f32 {
        self.p as *mut f32
    }
    fn i(&self) -> *mut i32 {
        self.p as *mut i32
    }
    fn v(&self) -> *mut std::os::raw::c_void {
        self.p as *mut _
    }
    fn at(&self, bytes: usize) -> *mut std::os::raw::c_void {
        (self.p + bytes as u64) as *mut _
    }
}

pub struct Dev {
    st: Arc<CudaStream>,
    sp: *mut std::os::raw::c_void,
    /// bytes ever allocated (growths included)
    pub allocated: std::cell::Cell<usize>,
    /// bytes allocated and not yet freed
    live: Arc<AtomicUsize>,
    /// host output the running call admitted and has not allocated yet
    host_reserve: std::cell::Cell<usize>,
}

impl Dev {
    pub fn new(device: usize) -> Result<Self, String> {
        let ctx = CudaContext::new(device).map_err(|e| format!("cuda context: {e:?}"))?;
        unsafe { ctx.disable_event_tracking() };
        let st = ctx.default_stream();
        let sp = st.cu_stream() as *mut _;
        Ok(Self {
            st,
            sp,
            allocated: std::cell::Cell::new(0),
            live: Arc::new(AtomicUsize::new(0)),
            host_reserve: std::cell::Cell::new(0),
        })
    }

    /// Device bytes the route holds now (weights, sessions' caches and scratch).
    pub fn live_bytes(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    pub fn alloc(&self, bytes: usize) -> Result<DBuf, String> {
        let bytes = bytes.max(16);
        let s = unsafe { self.st.alloc::<u8>(bytes) }
            .map_err(|e| format!("alloc of {bytes} bytes: {e:?}"))?;
        let p = {
            let (p, _g) = s.device_ptr(&self.st);
            p
        };
        self.allocated.set(self.allocated.get() + bytes);
        self.live.fetch_add(bytes, Ordering::Relaxed);
        Ok(DBuf {
            _s: s,
            p,
            bytes,
            live: self.live.clone(),
        })
    }

    fn zeros(&self, bytes: usize) -> Result<DBuf, String> {
        let b = self.alloc(bytes)?;
        cu!(memra_hy4r_memset(b.v(), 0, b.bytes, self.sp))?;
        Ok(b)
    }

    fn up_bytes(&self, dst: *mut std::os::raw::c_void, src: &[u8]) -> Result<(), String> {
        cu!(memra_hy4r_h2d(
            dst,
            src.as_ptr() as *const _,
            src.len(),
            self.sp
        ))
    }

    fn upload<T: Copy>(&self, v: &[T]) -> Result<DBuf, String> {
        let bytes = std::mem::size_of_val(v);
        let b = self.alloc(bytes)?;
        let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
        self.up_bytes(b.v(), raw)?;
        Ok(b)
    }

    fn put<T: Copy>(&self, dst: *mut std::os::raw::c_void, v: &[T]) -> Result<(), String> {
        let raw = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
        self.up_bytes(dst, raw)
    }

    fn down<T: Copy + Default>(
        &self,
        src: *const std::os::raw::c_void,
        n: usize,
    ) -> Result<Vec<T>, String> {
        let mut out = vec![T::default(); n];
        cu!(memra_hy4r_d2h(
            out.as_mut_ptr() as *mut _,
            src,
            n * std::mem::size_of::<T>(),
            self.sp
        ))?;
        Ok(out)
    }

    pub fn sync(&self) -> Result<(), String> {
        cu!(memra_hy4r_sync(self.sp))
    }
}

/// A grown-on-demand scratch buffer.
#[derive(Default)]
struct Grow(Option<DBuf>);

/// Unified memory a grown buffer must leave free.
const HEADROOM: usize = 4 << 30;

impl Grow {
    /// The buffer, grown to `bytes` when it is smaller; a growth is refused unless unified
    /// memory holds it with `HEADROOM` to spare (the weights are already resident), so every
    /// scratch allocation of a call is admitted where it happens.
    fn get(&mut self, dev: &Dev, bytes: usize) -> Result<&DBuf, String> {
        if self.0.as_ref().is_none_or(|b| b.bytes < bytes) {
            let held = self.0.as_ref().map_or(0, |b| b.bytes);
            // the call's host output is still to come: keep it free too
            admit(
                bytes.saturating_sub(held) + dev.host_reserve.get(),
                "a scratch buffer",
            )?;
            self.0 = None;
            self.0 = Some(dev.alloc(bytes)?);
        }
        Ok(self.0.as_ref().unwrap())
    }
}

/// Refuse a new allocation of `bytes` unless MemAvailable holds it plus `HEADROOM`.
fn admit(bytes: usize, what: &str) -> Result<(), String> {
    let avail = mem_available().unwrap_or(0);
    if avail < bytes + HEADROOM {
        return Err(format!(
            "{what} of {:.2} GiB does not fit: MemAvailable is {:.1} GiB with {} GiB kept free; split the prefill",
            bytes as f64 / (1u64 << 30) as f64,
            avail as f64 / (1u64 << 30) as f64,
            HEADROOM >> 30
        ));
    }
    Ok(())
}

// ---- weights ----------------------------------------------------------------------------------

/// A linear weight `[n, k]` (or a stacked expert bank `[experts, n, k]`) in its stored format.
pub struct W {
    data: DBuf,
    scales: Option<DBuf>,
    scale2: Option<DBuf>,
    fmt: i32,
    n: usize,
    k: usize,
    grid_cols: i32,
    /// bytes per expert (banks)
    wbs: usize,
    scbs: usize,
}

/// The checkpoint, read with `pread` into a staging buffer; the page cache of what was read is
/// dropped after each tensor (`POSIX_FADV_DONTNEED`), so the upload does not double the
/// resident set.
pub struct Ckpt {
    pub st: StModel,
    stage: std::cell::RefCell<Vec<u8>>,
    pub bytes_read: std::cell::Cell<usize>,
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

impl Ckpt {
    pub fn open(dir: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            st: StModel::open(dir).map_err(|e| format!("open {}: {e}", dir.display()))?,
            stage: std::cell::RefCell::new(vec![0u8; 64 << 20]),
            bytes_read: std::cell::Cell::new(0),
        })
    }

    fn check(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<usize, String> {
        let (info, _) = self
            .st
            .raw(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        let got: Vec<usize> = info.shape.iter().map(|v| *v as usize).collect();
        if info.dtype != dtype || got != shape {
            return Err(format!(
                "{name}: {} {:?} in the checkpoint, the plan needs {dtype} {shape:?}",
                info.dtype, got
            ));
        }
        let (_, _, _, len) = self
            .st
            .raw_extent(name)
            .ok_or(format!("{name}: no extent"))?;
        Ok(len)
    }

    fn dtype(&self, name: &str) -> Result<String, String> {
        Ok(self
            .st
            .raw(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?
            .0
            .dtype
            .clone())
    }

    /// The tensor's bytes, streamed through `f(offset, chunk)`.
    fn stream(
        &self,
        name: &str,
        mut f: impl FnMut(usize, &[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let (_, file, start, len) = self
            .st
            .raw_extent(name)
            .ok_or_else(|| format!("the checkpoint has no tensor {name}"))?;
        let mut stage = self.stage.borrow_mut();
        let mut off = 0;
        while off < len {
            let n = (len - off).min(stage.len());
            file.read_exact_at(&mut stage[..n], (start + off) as u64)
                .map_err(|e| format!("{name}: pread: {e}"))?;
            f(off, &stage[..n])?;
            off += n;
        }
        self.bytes_read.set(self.bytes_read.get() + len);
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                start as i64,
                len as i64,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        Ok(())
    }

    fn host(&self, name: &str) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        self.stream(name, |_, c| {
            out.extend_from_slice(c);
            Ok(())
        })?;
        Ok(out)
    }

    fn upload(&self, dev: &Dev, name: &str, dtype: &str, shape: &[usize]) -> Result<DBuf, String> {
        let len = self.check(name, dtype, shape)?;
        let b = dev.alloc(len)?;
        self.stream(name, |off, c| dev.up_bytes(b.at(off), c))?;
        Ok(b)
    }

    /// A BF16 or F32 tensor as f32 values (exact), shape-checked.
    pub fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        match self.dtype(name)?.as_str() {
            "BF16" => {
                self.check(name, "BF16", shape)?;
                Ok(bf16_to_f32(&self.host(name)?))
            }
            "F32" => {
                self.check(name, "F32", shape)?;
                Ok(le_f32(&self.host(name)?))
            }
            d => Err(format!("{name}: {d} where the plan reads BF16 or F32")),
        }
    }

    fn vec(&self, dev: &Dev, name: &str, shape: &[usize]) -> Result<DBuf, String> {
        dev.upload(&self.f32s(name, shape)?)
    }

    fn i32s(&self, name: &str, shape: &[usize]) -> Result<Vec<i32>, String> {
        self.check(name, "I32", shape)?;
        Ok(self
            .host(name)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// A linear weight `[n, k]` in the format the plan names for it.
    fn weight(
        &self,
        dev: &Dev,
        name: &str,
        n: usize,
        k: usize,
        format: HyV4WeightFormat,
    ) -> Result<W, String> {
        let dtype = self.dtype(name)?;
        match (format, dtype.as_str()) {
            (HyV4WeightFormat::Bf16, "BF16") => Ok(W {
                data: self.upload(dev, name, "BF16", &[n, k])?,
                scales: None,
                scale2: None,
                fmt: FMT_BF16,
                n,
                k,
                grid_cols: 0,
                wbs: 0,
                scbs: 0,
            }),
            (HyV4WeightFormat::Fp8Block { block: 128 }, "F8_E4M3") => {
                let stem = name.strip_suffix(".weight").unwrap_or(name);
                let grid = [n.div_ceil(128), k.div_ceil(128)];
                Ok(W {
                    data: self.upload(dev, name, "F8_E4M3", &[n, k])?,
                    scales: Some(self.upload(
                        dev,
                        &format!("{stem}.weight_scale_inv"),
                        "F32",
                        &grid,
                    )?),
                    scale2: None,
                    fmt: FMT_FP8,
                    n,
                    k,
                    grid_cols: grid[1] as i32,
                    wbs: 0,
                    scbs: 0,
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
        dev: &Dev,
        stem: &str,
        experts: usize,
        n: usize,
        k: usize,
        group: usize,
    ) -> Result<W, String> {
        if group != 16 {
            return Err(format!("{stem}: NVFP4 group {group}, the kernel runs 16"));
        }
        Ok(W {
            data: self.upload(dev, &format!("{stem}.weight"), "U8", &[experts, n, k / 2])?,
            scales: Some(self.upload(
                dev,
                &format!("{stem}.weight_scale"),
                "F8_E4M3",
                &[experts, n, k / 16],
            )?),
            scale2: Some(self.upload(dev, &format!("{stem}.weight_scale_2"), "F32", &[experts])?),
            fmt: FMT_NVFP4,
            n,
            k,
            grid_cols: 0,
            wbs: n * k / 2,
            scbs: n * k / 16,
        })
    }
}

struct Hc {
    fun: DBuf,
    base: DBuf,
    scale: DBuf,
}

struct Indexer {
    plan: IndexPoolPlan,
    tk: W,
    tk_norm: DBuf,
    kvp: DBuf,
    kgp: DBuf,
    ape: DBuf,
    key_norm: DBuf,
    q: W,
    hw: W,
}

struct Attn {
    plan: HyV4AttentionPlan,
    gate: W,
    qa: W,
    qa_norm: DBuf,
    qb: W,
    kva: W,
    kva_norm: DBuf,
    /// `kv_b_proj`'s nope rows per head, transposed: `[heads, kv_lora, nope]` BF16 (exact)
    kb_t: DBuf,
    /// `kv_b_proj` as stored, for its value rows
    kvb: DBuf,
    o: W,
    post_norm: DBuf,
    sinks: DBuf,
    indexer: Option<Indexer>,
}

#[allow(clippy::large_enum_variant)]
enum Mlp {
    Dense {
        gate: W,
        up: W,
        down: W,
        inter: usize,
        norm: DBuf,
    },
    Moe {
        router: W,
        bias: Option<DBuf>,
        rdown: W,
        rup: W,
        gate: W,
        up: W,
        down: W,
        sgate: W,
        sup: W,
        sdown: W,
        norm: DBuf,
        experts: usize,
        topk: usize,
        lat: usize,
        shared: usize,
        limit: f32,
        normalize: bool,
        scaling: f32,
    },
}

struct Engram {
    plan: HyV4EngramPlan,
    lookup: Vec<i32>,
    multipliers: Vec<i64>,
    moduli: Vec<u64>,
    offsets: Vec<u64>,
    /// the shard mappings (the checkpoint's own read-only `mmap`) and the tensors' byte
    /// offsets in them: the tables stay on disk and the page cache holds the rows read
    table: (Arc<memmap2::Mmap>, usize),
    scales: (Arc<memmap2::Mmap>, usize),
    head_norms: DBuf,
    value: W,
    value_bias: DBuf,
    keys: Vec<(W, DBuf)>,
    pre_gamma: DBuf,
    norm_key: DBuf,
    norm_query: DBuf,
    hc_scales: DBuf,
}

struct Layer {
    res: HyV4ResidualPlan,
    in_norm: DBuf,
    post_attn_norm: DBuf,
    in_eps: f32,
    mlp_eps: f32,
    hc_attn: Hc,
    hc_mlp: Hc,
    attn: Attn,
    mlp: Mlp,
    engram: Option<Engram>,
}

struct Cache {
    /// latent + rope key rows, F32 `[rows, 576]`
    kv: DBuf,
    /// ring length on sliding layers, 0 on linear (indexed) caches
    ring: usize,
    tk: Option<DBuf>,
    gk: Option<DBuf>,
}

/// What one call recorded per layer.
#[derive(Default, Clone)]
pub struct LayerRec {
    /// indexed layers that ran the indexer: (k, groups, recalled [q, k], logits [q, groups])
    pub recalled: Option<(usize, usize, Vec<i32>, Vec<f32>)>,
    /// Engram layers: the table rows per query
    pub engram_rows: Option<Vec<Vec<u64>>>,
    /// MoE layers: (picks [q, topk] sorted, selection scores [q, experts])
    pub routes: Option<(usize, usize, Vec<i32>, Vec<f32>)>,
}

/// Selections a call applies instead of its own (the checkpoint gate's forced runs): per
/// layer, the router's experts `[q, topk]` and the indexer's recalled groups `[q, k]`.
#[derive(Default)]
pub struct Forced {
    pub router: HashMap<usize, Vec<i32>>,
    pub indexer: HashMap<usize, Vec<i32>>,
}

#[derive(Default)]
struct Scratch {
    tok: Grow,
    qpos: Grow,
    posrow: Grow,
    cs: [Grow; 2],
    s: Grow,
    post: Grow,
    x: Grow,
    xn: Grow,
    gate: Grow,
    qa: Grow,
    qr: Grow,
    q: Grow,
    ckv: Grow,
    lat: Grow,
    qcat: Grow,
    olat: Grow,
    o: Grow,
    og: Grow,
    ao: Grow,
    a: Grow,
    lin: Grow,
    lin_rows: Grow,
    ring_rows: Grow,
    po: Grow,
    pml: Grow,
    tkv: Grow,
    qi: Grow,
    hw: Grow,
    ilog: Grow,
    ipick: Grow,
    isel: Grow,
    ilist: Grow,
    ilens: Grow,
    rl: Grow,
    rsc: Grow,
    rch: Grow,
    rpick: Grow,
    rw: Grow,
    rin: Grow,
    xg: Grow,
    mg: Grow,
    mu: Grow,
    ma: Grow,
    my: Grow,
    counts: Grow,
    cursor: Grow,
    tiles: Grow,
    rowmap: Grow,
    macc: Grow,
    routed: Grow,
    sg: Grow,
    su: Grow,
    sa: Grow,
    sd: Grow,
    sum: Grow,
    m: Grow,
    dg: Grow,
    du: Grow,
    da: Grow,
    ecodes: Grow,
    escales: Grow,
    emask: Grow,
    emb: Grow,
    evalue: Grow,
    ekeys: Grow,
    emain: Grow,
    hx: Grow,
    hn: Grow,
    logits: Grow,
}

/// The GEMM launchers, borrowing only the device and the red arms.
#[derive(Clone, Copy)]
struct Gm<'d> {
    dev: &'d Dev,
    red: RedArms,
}

impl Gm<'_> {
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        x: *const f32,
        ldx: usize,
        w: &W,
        y: *mut f32,
        ldy: usize,
        m: usize,
    ) -> Result<(), String> {
        let red = i32::from(self.red.dequant_bf16 && w.fmt != FMT_BF16);
        cu!(memra_hy4r_gemm(
            x,
            ldx as i64,
            0,
            w.data.v(),
            0,
            w.scales.as_ref().map_or(std::ptr::null_mut(), |s| s.v()),
            0,
            std::ptr::null(),
            w.fmt,
            w.grid_cols,
            y,
            ldy as i64,
            0,
            m as i32,
            w.n as i32,
            w.k as i32,
            1,
            std::ptr::null(),
            0,
            red,
            i32::from(m <= 16),
            self.dev.sp
        ))
    }

    /// The F32 GEMM of the program's F32 islands (router logits, head): F32 inputs, BF16 weight.
    fn gemm_f32(&self, x: *const f32, w: &W, y: *mut f32, m: usize) -> Result<(), String> {
        if w.fmt != FMT_BF16 {
            return Err("the F32 islands read BF16 weights".into());
        }
        self.gemm_f32_raw(x, w.data.p, w.n, w.k, y, m)
    }

    fn gemm_f32_raw(
        &self,
        x: *const f32,
        wp: u64,
        n: usize,
        k: usize,
        y: *mut f32,
        m: usize,
    ) -> Result<(), String> {
        let rc = unsafe {
            memra_hy4f_gemm(
                x as *const _,
                k as i64,
                wp as *const _,
                std::ptr::null(),
                0,
                1.0,
                0,
                0,
                y as *mut _,
                n as i64,
                m as i32,
                n as i32,
                k as i32,
                self.dev.sp,
            )
        };
        ck("memra_hy4f_gemm", rc)
    }
}

pub struct Route<'a> {
    pub dev: Dev,
    plan: &'a ModelPlan,
    hy: &'a HyV4FlashConfig,
    pub red: RedArms,
    layers: Vec<Layer>,
    embed: DBuf,
    lm_head: DBuf,
    hc_head: Hc,
    final_norm: DBuf,
    final_eps: f32,
    /// the request whose cache the calls read and write (`call_in` swaps another one in)
    sess: Session,
    /// per RoPE base: (base, device inv_freq)
    bases: Vec<(f32, DBuf)>,
    sc: Scratch,
    /// the current call records its selections (a device sync per layer)
    recording: bool,
}

/// One request's carried state: its cache rows and its token history. The weights and the
/// call scratch are the route's and shared; a call reads and writes only its session's rows, so
/// requests served one call at a time through one route keep their numbers apart.
pub struct Session {
    caches: Vec<Cache>,
    cap: usize,
    /// tokens seen so far (the Engram n-grams read the history)
    history: Vec<u32>,
    /// a call failed mid-way: refuse until `reset`
    broken: bool,
}

impl Session {
    /// Positions the session has scored.
    pub fn positions(&self) -> usize {
        self.history.len()
    }

    /// Positions the cache holds at most.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Whether a call failed mid-way (every later call refuses until `reset`).
    pub fn broken(&self) -> bool {
        self.broken
    }

    /// Start a new request on the same rows: every cache row a later call reads is written by
    /// it first.
    pub fn reset(&mut self) {
        self.history.clear();
        self.broken = false;
    }
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

/// /proc/meminfo MemAvailable in bytes.
pub fn mem_available() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

impl<'a> Route<'a> {
    /// Upload the weights and size the cache for `cap` positions.
    pub fn load(
        dev: Dev,
        ck: &Ckpt,
        plan: &'a ModelPlan,
        hy: &'a HyV4FlashConfig,
        cap: usize,
        layers: usize,
    ) -> Result<Self, String> {
        let h = plan.hidden_size as usize;
        let v = plan.vocab_size as usize;
        let n = hy.hc_mult as usize;
        let numerics = plan.hy_v4_flash.ok_or("not a hy_v4_flash plan")?;
        if !numerics.output_projection_f32 {
            return Err("the Hy4-flash head is computed in F32".into());
        }
        let bf = HyV4WeightFormat::Bf16;
        let mut out_layers = Vec::new();
        let mut bases: Vec<(f32, DBuf)> = Vec::new();
        for il in 0..layers {
            let lp = &plan.layers[il];
            let ResidualTopology::HyV4(res) = &lp.residual else {
                return Err(format!("layer {il} is not a Hy4-flash residual"));
            };
            let res = *res;
            let p = format!("model.layers.{il}.");
            let hc = |sub: &str| -> Result<Hc, String> {
                Ok(Hc {
                    fun: ck.vec(&dev, &format!("{p}{sub}.hc_pre.hc_fn"), &[2 * n, n * h])?,
                    base: ck.vec(&dev, &format!("{p}{sub}.hc_pre.hc_base"), &[2 * n])?,
                    scale: ck.vec(&dev, &format!("{p}{sub}.hc_pre.hc_scale"), &[2])?,
                })
            };
            let a = *attention_plan(lp)?;
            if !bases.iter().any(|(b, _)| *b == a.rope.base) {
                let rd = a.qk_rope_head_dim as usize;
                let inv: Vec<f64> = (0..rd / 2)
                    .map(|i| 1.0 / (a.rope.base as f64).powf((2 * i) as f64 / rd as f64))
                    .collect();
                bases.push((a.rope.base, dev.upload(&inv)?));
            }
            let at = |s: &str| format!("{p}self_attn.{s}");
            let heads = a.query_heads as usize;
            let (nope, rd, vd) = (
                a.qk_nope_head_dim as usize,
                a.qk_rope_head_dim as usize,
                a.value_head_dim as usize,
            );
            let (ql, kvl) = (a.q_lora_rank as usize, a.kv_lora_rank as usize);
            // the kernels and strides are written for this one head geometry
            if (nope, rd, kvl, vd) != (192, 64, 512, 256) || a.rope.base <= 0.0 {
                return Err(format!(
                    "layer {il}: heads of nope {nope}, rope {rd}, latent {kvl}, value {vd}; the route runs 192, 64, 512, 256"
                ));
            }
            let pw = a.projection_weights;
            let kvb_host = {
                ck.check(&at("kv_b_proj.weight"), "BF16", &[heads * (nope + vd), kvl])?;
                ck.host(&at("kv_b_proj.weight"))?
            };
            // [heads, kvl, nope]: element (h, c, d) = kv_b[h * (nope + vd) + d][c]
            let mut kb_t = vec![0u8; heads * kvl * nope * 2];
            for hd in 0..heads {
                for d in 0..nope {
                    let src = (hd * (nope + vd) + d) * kvl;
                    for c in 0..kvl {
                        let dst = (hd * kvl + c) * nope + d;
                        kb_t[2 * dst] = kvb_host[2 * (src + c)];
                        kb_t[2 * dst + 1] = kvb_host[2 * (src + c) + 1];
                    }
                }
            }
            let indexer = match a.kind {
                HyV4AttentionKind::Indexed(ip) => {
                    let c = |s: &str| format!("{p}self_attn.compressed_indexer.{s}");
                    let (ih, id) = (ip.heads as usize, ip.head_dim as usize);
                    Some(Indexer {
                        plan: ip,
                        tk: ck.weight(&dev, &c("token_key_projection.weight"), id, h, bf)?,
                        tk_norm: ck.vec(&dev, &c("token_key_norm.weight"), &[id])?,
                        kvp: ck.vec(&dev, &c("key_value_projection.weight"), &[id, id])?,
                        kgp: ck.vec(&dev, &c("key_gate_projection.weight"), &[id, id])?,
                        ape: ck.vec(&dev, &c("intra_group_ape"), &[ip.pool as usize, id])?,
                        key_norm: ck.vec(&dev, &c("key_norm.weight"), &[id])?,
                        q: ck.weight(&dev, &c("query_projection.weight"), ih * id, ql, bf)?,
                        hw: ck.weight(&dev, &c("head_weight_projection.weight"), ih, h, bf)?,
                    })
                }
                HyV4AttentionKind::Sliding { .. } => None,
            };
            let sinks = if a.sink == memra_gguf::model_plan::TensorPresence::Absent {
                dev.upload(&vec![f32::NEG_INFINITY; heads])?
            } else {
                ck.vec(&dev, &at("learnable_sink_param"), &[heads])?
            };
            let attn = Attn {
                plan: a,
                gate: ck.weight(&dev, &at("linear_gate.weight"), heads * vd, h, pw)?,
                qa: ck.weight(&dev, &at("q_a_proj.weight"), ql, h, bf)?,
                qa_norm: ck.vec(&dev, &at("q_a_layernorm.weight"), &[ql])?,
                qb: ck.weight(&dev, &at("q_b_proj.weight"), heads * (nope + rd), ql, pw)?,
                kva: ck.weight(&dev, &at("kv_a_proj_with_mqa.weight"), kvl + rd, h, bf)?,
                kva_norm: ck.vec(&dev, &at("kv_a_layernorm.weight"), &[kvl])?,
                kb_t: dev.upload(&kb_t)?,
                kvb: dev.upload(&kvb_host)?,
                o: ck.weight(&dev, &at("o_proj.weight"), h, heads * vd, pw)?,
                post_norm: ck.vec(&dev, &at("post_proj_norm.weight"), &[h])?,
                sinks,
                indexer,
            };
            let m = |s: &str| format!("{p}mlp.{s}");
            let mlp = match &lp.mlp {
                MlpPlan::Dense(d) => {
                    if d.activation != ActivationPlan::Silu {
                        return Err(format!("layer {il}: dense activation {:?}", d.activation));
                    }
                    let inter = d.intermediate_size as usize;
                    Mlp::Dense {
                        gate: ck.weight(&dev, &m("gate_proj.weight"), inter, h, bf)?,
                        up: ck.weight(&dev, &m("up_proj.weight"), inter, h, bf)?,
                        down: ck.weight(&dev, &m("down_proj.weight"), h, inter, bf)?,
                        inter,
                        norm: ck.vec(&dev, &m("post_fc2_norm.weight"), &[h])?,
                    }
                }
                MlpPlan::LatentMoe(lm) => {
                    let e = lm.moe.expert_count as usize;
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
                    if inter != lat {
                        return Err(format!("layer {il}: expert width {inter} != latent {lat}"));
                    }
                    let sh = lm
                        .moe
                        .shared
                        .as_ref()
                        .ok_or(format!("layer {il}: no shared expert"))?
                        .intermediate_size as usize;
                    Mlp::Moe {
                        router: ck.weight(&dev, &m("gate.weight"), e, h, bf)?,
                        bias: if selection_bias {
                            Some(ck.vec(&dev, &m("gate.e_score_correction_bias"), &[e])?)
                        } else {
                            None
                        },
                        rdown: ck.weight(&dev, &m("routed_expert_down_proj.weight"), lat, h, bf)?,
                        rup: ck.weight(&dev, &m("routed_expert_up_proj.weight"), h, lat, bf)?,
                        gate: ck.experts(
                            &dev,
                            &m("experts.gate_proj"),
                            e,
                            inter,
                            lat,
                            group as usize,
                        )?,
                        up: ck.experts(
                            &dev,
                            &m("experts.up_proj"),
                            e,
                            inter,
                            lat,
                            group as usize,
                        )?,
                        down: ck.experts(
                            &dev,
                            &m("experts.down_proj"),
                            e,
                            lat,
                            inter,
                            group as usize,
                        )?,
                        sgate: ck.weight(&dev, &m("shared_experts.gate_proj.weight"), sh, h, bf)?,
                        sup: ck.weight(&dev, &m("shared_experts.up_proj.weight"), sh, h, bf)?,
                        sdown: ck.weight(&dev, &m("shared_experts.down_proj.weight"), h, sh, bf)?,
                        norm: ck.vec(&dev, &m("combine_output_norm.weight"), &[h])?,
                        experts: e,
                        topk: lm.moe.experts_per_token as usize,
                        lat,
                        shared: sh,
                        limit,
                        normalize: normalize_selected,
                        scaling: scaling_factor,
                    }
                }
                other => return Err(format!("layer {il}: MLP {other:?}")),
            };
            let engram = match &res.engram {
                Some(ep) => Some(Self::load_engram(&dev, ck, hy, il, ep, h, n)?),
                None => None,
            };
            out_layers.push(Layer {
                res,
                in_norm: ck.vec(&dev, &format!("{p}input_layernorm.weight"), &[h])?,
                post_attn_norm: ck.vec(
                    &dev,
                    &format!("{p}post_attention_layernorm.weight"),
                    &[h],
                )?,
                in_eps: lp.pre_attention_norm.epsilon,
                mlp_eps: lp.pre_mlp_norm.epsilon,
                hc_attn: hc("hc_attn_layer")?,
                hc_mlp: hc("hc_mlp_layer")?,
                attn,
                mlp,
                engram,
            });
            eprintln!(
                "hy4f-gpu: layer {il:2} resident, {:.1} GiB on the device",
                dev.allocated.get() as f64 / (1u64 << 30) as f64
            );
        }
        let embed = ck.upload(&dev, "model.embed_tokens.weight", "BF16", &[v, h])?;
        let lm_head = ck.upload(&dev, "lm_head.weight", "BF16", &[v, h])?;
        let hc_head = Hc {
            fun: ck.vec(&dev, "model.hc_head.hc_head_fn", &[n, n * h])?,
            base: ck.vec(&dev, "model.hc_head.hc_head_base", &[n])?,
            scale: ck.vec(&dev, "model.hc_head.hc_head_scale", &[1])?,
        };
        let final_norm = ck.vec(&dev, "model.norm.weight", &[h])?;
        let sess = Self::alloc_session(&dev, &out_layers, cap)?;
        dev.sync()?;
        Ok(Self {
            dev,
            plan,
            hy,
            red: RedArms::default(),
            layers: out_layers,
            embed,
            lm_head,
            hc_head,
            final_norm,
            final_eps: plan.output_norm.epsilon,
            sess,
            bases,
            sc: Scratch::default(),
            recording: false,
        })
    }

    /// The cache of one request at `cap` positions: per layer the latent and rope key rows (a
    /// ring of the window on sliding layers), and on indexed layers the token keys and the
    /// pooled group keys. Every byte is zeroed.
    fn alloc_session(dev: &Dev, layers: &[Layer], cap: usize) -> Result<Session, String> {
        let mut caches = Vec::new();
        for (ring, tk, gk) in Self::cache_shape(layers, cap) {
            caches.push(Cache {
                kv: dev.zeros(if ring > 0 { ring } else { cap + 1 } * 576 * 4)?,
                ring,
                tk: if tk > 0 { Some(dev.zeros(tk)?) } else { None },
                gk: if gk > 0 { Some(dev.zeros(gk)?) } else { None },
            });
        }
        Ok(Session {
            caches,
            cap,
            history: Vec::new(),
            broken: false,
        })
    }

    /// Per layer: (ring rows, token-key bytes, group-key bytes) of a cache at `cap` positions.
    fn cache_shape(layers: &[Layer], cap: usize) -> Vec<(usize, usize, usize)> {
        layers
            .iter()
            .map(|l| {
                let ring = match l.attn.plan.kind {
                    HyV4AttentionKind::Sliding { window } => window as usize,
                    HyV4AttentionKind::Indexed(_) => 0,
                };
                match &l.attn.indexer {
                    Some(ix) => {
                        let id = ix.plan.head_dim as usize;
                        (
                            ring,
                            (cap + 1) * id * 4,
                            (cap / ix.plan.pool as usize + 1) * id * 4,
                        )
                    }
                    None => (ring, 0, 0),
                }
            })
            .collect()
    }

    /// Device bytes a session of `cap` positions allocates ([`Self::new_session`]), each buffer
    /// at least the allocator's 16 bytes as `Dev::alloc` rounds it.
    pub fn session_bytes(&self, cap: usize) -> usize {
        Self::cache_shape(&self.layers, cap)
            .into_iter()
            .map(|(ring, tk, gk)| {
                let kv = if ring > 0 { ring } else { cap + 1 } * 576 * 4;
                kv.max(16)
                    + if tk > 0 { tk.max(16) } else { 0 }
                    + if gk > 0 { gk.max(16) } else { 0 }
            })
            .sum()
    }

    /// A new request's session of `cap` positions, admitted against MemAvailable with the
    /// route's headroom (refused rather than allocated past it).
    pub fn new_session(&self, cap: usize) -> Result<Session, String> {
        admit(self.session_bytes(cap), "a session's cache")?;
        let s = Self::alloc_session(&self.dev, &self.layers, cap)?;
        self.dev.sync()?;
        Ok(s)
    }

    /// The scratch a call of `rows` tokens grows for the indexer's group logits when its
    /// session holds up to `cap` positions (`rows * (cap / pool)` F32 values): the one call
    /// scratch whose size follows the context rather than the call. Every other scratch buffer
    /// is bounded by the call's rows, the window and the indexer budget.
    pub fn index_scratch_bytes(&self, rows: usize, cap: usize) -> usize {
        self.layers
            .iter()
            .filter_map(|l| l.attn.indexer.as_ref())
            .map(|ix| rows * (cap / ix.plan.pool as usize + 1) * 4)
            .max()
            .unwrap_or(0)
    }

    /// Bytes the indexer's group-logit scratch holds now (it grows with the longest context a
    /// call has scored and is kept): what a session's [`Self::index_scratch_bytes`] need not
    /// allocate again.
    pub fn index_scratch_held(&self) -> usize {
        self.sc.ilog.0.as_ref().map_or(0, |b| b.bytes)
    }

    /// Score `tokens` on `sess` (another request's session): [`Self::call`] with that session's
    /// cache and history swapped in for the call and back out after it.
    pub fn call_in(
        &mut self,
        sess: &mut Session,
        tokens: &[u32],
        all_logits: bool,
        rec: Option<&mut Vec<(usize, LayerRec)>>,
        forced: Option<&Forced>,
    ) -> Result<Vec<f32>, String> {
        std::mem::swap(&mut self.sess, sess);
        let out = self.call(tokens, all_logits, rec, forced);
        std::mem::swap(&mut self.sess, sess);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn load_engram(
        dev: &Dev,
        ck: &Ckpt,
        hy: &HyV4FlashConfig,
        il: usize,
        ep: &HyV4EngramPlan,
        h: usize,
        n: usize,
    ) -> Result<Engram, String> {
        let e = |s: &str| format!("model.layers.{il}.engram_layer.{s}");
        let (heads, dim) = (ep.heads as usize, ep.head_dim as usize);
        let width = heads * dim;
        let slot = hy
            .engram_slot(il as u32)
            .ok_or(format!("layer {il} carries no Engram in the config"))?;
        let moduli = hy.engram_head_vocab_sizes[slot].clone();
        let lookup = ck.i32s(&e("lookup_table"), &[ep.lookup_table_size as usize])?;
        let vocab = hashing::compressed_vocab(ep, &lookup)?;
        let multipliers = hashing::hash_multipliers(ep, vocab)?;
        let offsets = hashing::head_offsets(&moduli);
        let rows = ep.table_rows as usize;
        let tname = e("multi_head_embedding.embedding.weight");
        let sname = e("multi_head_embedding.embedding.weight_scale");
        ck.check(&tname, "F8_E4M3", &[rows, dim])?;
        ck.check(&sname, "F32", &[rows])?;
        let (tm, _, ts, tlen) = ck.st.raw_extent(&tname).ok_or("no table extent")?;
        let (sm, _, ss, slen) = ck.st.raw_extent(&sname).ok_or("no scale extent")?;
        if tlen != rows * dim || slen != rows * 4 || ts + tlen > tm.len() || ss + slen > sm.len() {
            return Err(format!(
                "layer {il}: the Engram table extents are not {rows} rows"
            ));
        }
        // rows are read at random: no read-ahead around them
        tm.advise_range(memmap2::Advice::Random, ts, tlen)
            .map_err(|e| format!("layer {il}: madvise the Engram table: {e}"))?;
        let mut norms = Vec::with_capacity(width);
        for k in 0..heads {
            norms.extend(ck.f32s(&e(&format!("lookup_per_head_norms.{k}.weight")), &[dim])?);
        }
        let bf = HyV4WeightFormat::Bf16;
        let mut keys = Vec::new();
        let (mut nk, mut nq, mut hs) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            keys.push((
                ck.weight(dev, &e(&format!("key_projs.{j}.weight")), h, width, bf)?,
                ck.vec(dev, &e(&format!("key_projs.{j}.bias")), &[h])?,
            ));
            nk.extend(ck.f32s(&e(&format!("norm_key.{j}.weight")), &[h])?);
            nq.extend(ck.f32s(&e(&format!("norm_query.{j}.weight")), &[h])?);
            hs.extend(ck.f32s(&e(&format!("unified_per_hc_scales.{j}")), &[h])?);
        }
        Ok(Engram {
            plan: *ep,
            lookup,
            multipliers,
            moduli,
            offsets,
            table: (tm, ts),
            scales: (sm, ss),
            head_norms: dev.upload(&norms)?,
            value: ck.weight(dev, &e("value_proj.weight"), h, width, bf)?,
            value_bias: ck.vec(dev, &e("value_proj.bias"), &[h])?,
            keys,
            pre_gamma: ck.vec(dev, &e("unified_pre_gamma"), &[h])?,
            norm_key: dev.upload(&nk)?,
            norm_query: dev.upload(&nq)?,
            hc_scales: dev.upload(&hs)?,
        })
    }

    pub fn positions(&self) -> usize {
        self.sess.positions()
    }

    /// Start a new request: every cache row a later call reads is written by it first.
    pub fn reset(&mut self) {
        self.sess.reset();
    }

    /// The host memory a call's output takes: its logits, and with records the host copies of
    /// every layer's selections.
    fn host_bytes(&self, t: usize, all_logits: bool, record: bool) -> usize {
        let v = self.plan.vocab_size as usize;
        let mut host = if all_logits { t } else { 1 } * v * 4;
        if record {
            for l in &self.layers {
                if let Some(ix) = &l.attn.indexer {
                    let ks = (ix.plan.top_k / ix.plan.pool) as usize;
                    host += t * (ks + self.sess.cap / ix.plan.pool as usize + 1) * 4;
                }
                if let Mlp::Moe { experts, topk, .. } = &l.mlp {
                    host += t * (experts + topk) * 4;
                }
                if let Some(eg) = &l.engram {
                    // one `Vec<u64>` of the heads' row ids a token
                    host += t * (eg.plan.heads as usize * 8 + std::mem::size_of::<Vec<u64>>());
                }
            }
        }
        host
    }

    fn base_slot(&self, base: f32) -> usize {
        self.bases.iter().position(|(b, _)| *b == base).unwrap()
    }

    /// Score positions `p0..p0 + tokens.len()` (`tokens` the new tokens; the route keeps the
    /// history), with optional records and forced selections. Returns the logits of the scored
    /// rows `[rows, vocab]` (every row with `all_logits`, else the last).
    pub fn call(
        &mut self,
        tokens: &[u32],
        all_logits: bool,
        rec: Option<&mut Vec<(usize, LayerRec)>>,
        forced: Option<&Forced>,
    ) -> Result<Vec<f32>, String> {
        // a call that fails may have written cache rows (a sliding ring row of an earlier
        // position among them): the route refuses every call until `reset` starts a new request
        if self.sess.broken {
            return Err(
                "an earlier call failed mid-way; reset the route and replay the request".into(),
            );
        }
        let out = self.call_inner(tokens, all_logits, rec, forced);
        self.dev.host_reserve.set(0);
        if out.is_err() {
            self.sess.broken = true;
        }
        out
    }

    fn call_inner(
        &mut self,
        tokens: &[u32],
        all_logits: bool,
        mut rec: Option<&mut Vec<(usize, LayerRec)>>,
        forced: Option<&Forced>,
    ) -> Result<Vec<f32>, String> {
        self.recording = rec.is_some();
        let t = tokens.len();
        let p0 = self.sess.history.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if p0 + t > self.sess.cap {
            return Err(format!(
                "{} positions past the cache's {}",
                p0 + t,
                self.sess.cap
            ));
        }
        // the call's host output (logits, records); its device scratch is admitted where it
        // grows (`Grow::get`)
        let host = self.host_bytes(t, all_logits, rec.is_some());
        admit(host, "the call's host output")?;
        self.dev.host_reserve.set(host);
        let v = self.plan.vocab_size as usize;
        for &tk in tokens {
            if tk as usize >= v {
                return Err(format!("token {tk} past the vocabulary {v}"));
            }
        }
        self.sess.history.extend_from_slice(tokens);
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let dev = &self.dev;
        let sc = &mut self.sc;
        // per-call tables
        let tok: Vec<i32> = tokens.iter().map(|v| *v as i32).collect();
        let tokb = sc.tok.get(dev, t * 4)?.p;
        dev.put(tokb as *mut _, &tok)?;
        let qpos: Vec<i32> = (p0..p0 + t).map(|p| p as i32).collect();
        let qposb = sc.qpos.get(dev, t * 4)?.p;
        dev.put(qposb as *mut _, &qpos)?;
        let cs_pos0 = p0.saturating_sub(3);
        let cs_rows = p0 + t - cs_pos0;
        let posrow: Vec<i32> = (p0..p0 + t).map(|p| (p - cs_pos0) as i32).collect();
        let posrowb = sc.posrow.get(dev, t * 4)?.p;
        dev.put(posrowb as *mut _, &posrow)?;
        for (bi, (_, inv)) in self.bases.iter().enumerate() {
            let b = sc.cs[bi].get(dev, cs_rows * 32 * 8)?;
            cu!(memra_hy4r_rope_table(
                b.v(),
                cs_pos0 as i32,
                cs_rows as i32,
                32,
                inv.p as *const f64,
                dev.sp
            ))?;
        }
        let s = sc.s.get(dev, t * n * h * 4)?.p as *mut f32;
        cu!(memra_hy4r_embed(
            self.embed.v(),
            tokb as *const i32,
            s,
            t as i32,
            n as i32,
            h as i32,
            dev.sp
        ))?;
        for il in 0..self.layers.len() {
            let mut lr = LayerRec::default();
            self.layer(il, p0, t, s, &mut lr, forced)?;
            if let Some(r) = rec.as_deref_mut() {
                r.push((il, lr));
            }
        }
        // the exit: hc_head, the final norm, the F32 head
        let dev = &self.dev;
        let sc = &mut self.sc;
        let hx = sc.hx.get(dev, t * h * 4)?.f();
        let numerics = self.plan.hy_v4_flash.unwrap();
        cu!(memra_hy4r_hc(
            s,
            self.hc_head.fun.f(),
            self.hc_head.base.f(),
            self.hc_head.scale.f(),
            t as i32,
            n as i32,
            h as i32,
            n as i32,
            self.hy.rms_norm_eps,
            numerics.exit_epsilon,
            0.0,
            std::ptr::null_mut(),
            hx,
            dev.sp
        ))?;
        let hn = sc.hn.get(dev, t * h * 4)?.f();
        cu!(memra_hy4r_rms(
            hx,
            h as i64,
            self.final_norm.f(),
            hn,
            h as i64,
            t as i32,
            h as i32,
            self.final_eps,
            dev.sp
        ))?;
        let (r0, rows) = if all_logits { (0, t) } else { (t - 1, 1) };
        let lg = self
            .sc
            .logits
            .get(&self.dev, HEAD_ROWS.min(rows) * v * 4)?
            .f();
        let mut out = Vec::with_capacity(rows * v);
        for c0 in (r0..r0 + rows).step_by(HEAD_ROWS) {
            let cr = HEAD_ROWS.min(r0 + rows - c0);
            Gm {
                dev: &self.dev,
                red: self.red,
            }
            .gemm_f32_raw(hn.wrapping_add(c0 * h), self.lm_head.p, v, h, lg, cr)?;
            out.extend(self.dev.down::<f32>(lg as *const _, cr * v)?);
        }
        Ok(out)
    }

    fn layer(
        &mut self,
        il: usize,
        p0: usize,
        t: usize,
        s: *mut f32,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
    ) -> Result<(), String> {
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let res = self.layers[il].res;
        if self.layers[il].engram.is_some() {
            self.engram(il, p0, t, s, rec)?;
        }
        let dev = &self.dev;
        let l = &self.layers[il];
        let post = self.sc.post.get(dev, t * n * 4)?.f();
        let x = self.sc.x.get(dev, t * h * 4)?.f();
        let xn = self.sc.xn.get(dev, t * h * 4)?.f();
        cu!(memra_hy4r_hc(
            s,
            l.hc_attn.fun.f(),
            l.hc_attn.base.f(),
            l.hc_attn.scale.f(),
            t as i32,
            n as i32,
            h as i32,
            2 * n as i32,
            res.mix_norm_epsilon,
            res.epsilon,
            res.post_magnitude,
            post,
            x,
            dev.sp
        ))?;
        cu!(memra_hy4r_rms(
            x,
            h as i64,
            l.in_norm.f(),
            xn,
            h as i64,
            t as i32,
            h as i32,
            l.in_eps,
            dev.sp
        ))?;
        let a = self.attention(il, p0, t, xn, rec, forced)?;
        let dev = &self.dev;
        let l = &self.layers[il];
        cu!(memra_hy4r_write_back(
            s, post, a, t as i32, n as i32, h as i32, dev.sp
        ))?;
        cu!(memra_hy4r_hc(
            s,
            l.hc_mlp.fun.f(),
            l.hc_mlp.base.f(),
            l.hc_mlp.scale.f(),
            t as i32,
            n as i32,
            h as i32,
            2 * n as i32,
            res.mix_norm_epsilon,
            res.epsilon,
            res.post_magnitude,
            post,
            x,
            dev.sp
        ))?;
        cu!(memra_hy4r_rms(
            x,
            h as i64,
            l.post_attn_norm.f(),
            xn,
            h as i64,
            t as i32,
            h as i32,
            l.mlp_eps,
            dev.sp
        ))?;
        let m = self.mlp(il, t, xn, rec, forced)?;
        let post = self.sc.post.get(&self.dev, t * n * 4)?.f();
        cu!(memra_hy4r_write_back(
            s,
            post,
            m,
            t as i32,
            n as i32,
            h as i32,
            self.dev.sp
        ))?;
        Ok(())
    }

    fn attention(
        &mut self,
        il: usize,
        p0: usize,
        t: usize,
        xn: *mut f32,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
    ) -> Result<*mut f32, String> {
        let h = self.plan.hidden_size as usize;
        let eps = self.hy.rms_norm_eps;
        let bi = self.base_slot(self.layers[il].attn.plan.rope.base);
        let dev = &self.dev;
        let sc = &mut self.sc;
        let l = &self.layers[il];
        let a = &l.attn;
        let heads = a.plan.query_heads as usize;
        let (ql, kvl) = (a.plan.q_lora_rank as usize, a.plan.kv_lora_rank as usize);
        let hd = 256usize;
        let cs = sc.cs[bi].get(dev, 0)?.p as *const std::os::raw::c_void;
        let posrow = sc.posrow.get(dev, 0)?.i();
        let qpos = sc.qpos.get(dev, 0)?.i();
        let gate = sc.gate.get(dev, t * heads * hd * 4)?.f();
        let qa = sc.qa.get(dev, t * ql * 4)?.f();
        let qr = sc.qr.get(dev, t * ql * 4)?.f();
        let q = sc.q.get(dev, t * heads * hd * 4)?.f();
        let ckv = sc.ckv.get(dev, t * 576 * 4)?.f();
        let lat = sc.lat.get(dev, t * 576 * 4)?.f();
        let qcat = sc.qcat.get(dev, t * heads * 576 * 4)?.f();
        let olat = sc.olat.get(dev, t * heads * 512 * 4)?.f();
        let o = sc.o.get(dev, t * heads * hd * 4)?.f();
        let og = sc.og.get(dev, t * heads * hd * 4)?.f();
        let ao = sc.ao.get(dev, t * h * 4)?.f();
        let aout = sc.a.get(dev, t * h * 4)?.f();
        let sp = dev.sp;
        let route = Gm {
            dev: &self.dev,
            red: self.red,
        };
        route.gemm(xn, h, &a.gate, gate, heads * hd, t)?;
        route.gemm(xn, h, &a.qa, qa, ql, t)?;
        cu!(memra_hy4r_rms(
            qa,
            ql as i64,
            a.qa_norm.f(),
            qr,
            ql as i64,
            t as i32,
            ql as i32,
            eps,
            sp
        ))?;
        route.gemm(qr, ql, &a.qb, q, heads * hd, t)?;
        cu!(memra_hy4r_rope(
            q,
            (heads * hd) as i64,
            hd as i64,
            t as i32,
            heads as i32,
            192,
            32,
            cs,
            posrow,
            sp
        ))?;
        route.gemm(xn, h, &a.kva, ckv, 576, t)?;
        cu!(memra_hy4r_rms(
            ckv,
            576,
            a.kva_norm.f(),
            lat,
            576,
            t as i32,
            kvl as i32,
            eps,
            sp
        ))?;
        cu!(memra_hy4r_copy(
            ckv.wrapping_add(512),
            576,
            lat.wrapping_add(512) as *mut _,
            576,
            t as i32,
            64,
            0,
            std::ptr::null(),
            sp
        ))?;
        cu!(memra_hy4r_rope(
            lat, 576, 576, t as i32, 1, 512, 32, cs, posrow, sp
        ))?;
        // absorbed queries: q_lat[h] = kv_b_nope[h]^T q_nope[h]
        cu!(memra_hy4r_gemm(
            q,
            (heads * hd) as i64,
            hd as i64,
            a.kb_t.v(),
            (kvl * 192 * 2) as i64,
            std::ptr::null(),
            0,
            std::ptr::null(),
            FMT_BF16,
            0,
            qcat,
            (heads * 576) as i64,
            576,
            t as i32,
            kvl as i32,
            192,
            heads as i32,
            std::ptr::null(),
            0,
            0,
            i32::from(t <= 16),
            sp
        ))?;
        cu!(memra_hy4r_copy(
            q.wrapping_add(192),
            hd as i64,
            qcat.wrapping_add(512) as *mut _,
            576,
            (t * heads) as i32,
            64,
            0,
            std::ptr::null(),
            sp
        ))?;
        // the cache rows of this call
        let cache = &self.sess.caches[il];
        let late = usize::from(self.red.kv_off_by_one && t == 1);
        let kv_len = p0 + t;
        let scale = (256f64).powf(-0.5) as f32;
        let sinks = a.sinks.f();
        // keys: (mode, kv pointer, window, ring, base)
        let mut list_mode = false;
        let (kvp, window, ring, base);
        if cache.ring > 0 {
            let w = cache.ring;
            let HyV4AttentionKind::Sliding { window: win } = a.plan.kind else {
                unreachable!()
            };
            if win as usize != w {
                return Err("ring and window differ".into());
            }
            if t == 1 {
                let slot = [((p0 + late) % w) as i32];
                let sb = sc.ring_rows.get(dev, 4)?.i();
                dev.put(sb as *mut _, &slot)?;
                cu!(memra_hy4r_copy(
                    lat,
                    576,
                    cache.kv.v(),
                    576,
                    1,
                    576,
                    0,
                    sb,
                    sp
                ))?;
                kvp = cache.kv.p;
                window = w;
                ring = w;
                base = 0;
            } else {
                // a linear copy: the ring's earlier rows, then this call's rows
                let b0 = p0.saturating_sub(w - 1);
                let lin = sc.lin.get(dev, (p0 - b0 + t) * 576 * 4)?;
                if p0 > b0 {
                    let rows: Vec<i32> = (b0..p0).map(|p| (p % w) as i32).collect();
                    let rb = sc.lin_rows.get(dev, rows.len() * 4)?.i();
                    dev.put(rb as *mut _, &rows)?;
                    cu!(memra_hy4r_gather_bf(
                        cache.kv.v(),
                        576,
                        rb,
                        lin.v(),
                        576,
                        rows.len() as i32,
                        576,
                        sp
                    ))?;
                }
                cu!(memra_hy4r_copy(
                    lat,
                    576,
                    lin.at((p0 - b0) * 576 * 4),
                    576,
                    t as i32,
                    576,
                    0,
                    std::ptr::null(),
                    sp
                ))?;
                // the ring keeps the last `w` positions of the call
                let keep = t.min(w);
                let rows: Vec<i32> = (p0 + t - keep..p0 + t).map(|p| (p % w) as i32).collect();
                let rb = sc.ring_rows.get(dev, rows.len() * 4)?.i();
                dev.put(rb as *mut _, &rows)?;
                cu!(memra_hy4r_copy(
                    lat.wrapping_add((t - keep) * 576),
                    576,
                    cache.kv.v(),
                    576,
                    keep as i32,
                    576,
                    0,
                    rb,
                    sp
                ))?;
                kvp = lin.p;
                window = w;
                ring = 0;
                base = b0;
            }
        } else {
            let rows: Vec<i32> = (p0..p0 + t).map(|p| (p + late) as i32).collect();
            let rb = sc.ring_rows.get(dev, rows.len() * 4)?.i();
            dev.put(rb as *mut _, &rows)?;
            cu!(memra_hy4r_copy(
                lat,
                576,
                cache.kv.v(),
                576,
                t as i32,
                576,
                0,
                rb,
                sp
            ))?;
            kvp = cache.kv.p;
            window = 0;
            ring = 0;
            base = 0;
        }
        // the indexer
        let mut lens_p: *mut i32 = std::ptr::null_mut();
        let mut list_p: *mut i32 = std::ptr::null_mut();
        let mut max_len = 0usize;
        if let Some(ix) = &a.indexer {
            let ip = ix.plan;
            let (ih, id, pool) = (ip.heads as usize, ip.head_dim as usize, ip.pool as usize);
            let tkv = sc.tkv.get(dev, t * id * 4)?.f();
            route.gemm(xn, h, &ix.tk, tkv, id, t)?;
            cu!(memra_hy4r_rms(
                tkv,
                id as i64,
                ix.tk_norm.f(),
                tkv,
                id as i64,
                t as i32,
                id as i32,
                eps,
                sp
            ))?;
            let rows: Vec<i32> = (p0..p0 + t).map(|p| p as i32).collect();
            let rb = sc.ring_rows.get(dev, rows.len() * 4)?.i();
            dev.put(rb as *mut _, &rows)?;
            let tkc = cache.tk.as_ref().unwrap();
            let gkc = cache.gk.as_ref().unwrap();
            cu!(memra_hy4r_copy(
                tkv,
                id as i64,
                tkc.v(),
                id as i64,
                t as i32,
                id as i32,
                0,
                rb,
                sp
            ))?;
            // groups whose last key is in this call close now (the red arm: their third)
            let last = if self.red.pool_early {
                pool - 2
            } else {
                pool - 1
            };
            let g_lo = (p0 + pool - 1 - last) / pool;
            let g_hi_end = (p0 + t + pool - 1 - last) / pool; // exclusive: groups with g*pool+last < p0+t
            if g_hi_end > g_lo {
                cu!(memra_hy4r_pool(
                    tkc.v(),
                    id as i32,
                    pool as i32,
                    ix.kvp.f(),
                    ix.kgp.f(),
                    ix.ape.f(),
                    ix.key_norm.f(),
                    eps,
                    ip.rope_head_dim as i32,
                    cs,
                    cs_pos0_of(p0) as i32,
                    g_lo as i32,
                    (g_hi_end - g_lo) as i32,
                    gkc.f(),
                    sp
                ))?;
            }
            if kv_len > ip.top_k as usize {
                let qi = sc.qi.get(dev, t * ih * id * 4)?.f();
                route.gemm(qr, ql, &ix.q, qi, ih * id, t)?;
                cu!(memra_hy4r_rope(
                    qi,
                    (ih * id) as i64,
                    id as i64,
                    t as i32,
                    ih as i32,
                    (id - ip.rope_head_dim as usize) as i32,
                    (ip.rope_head_dim / 2) as i32,
                    cs,
                    posrow,
                    sp
                ))?;
                let hw = sc.hw.get(dev, t * ih * 4)?.f();
                route.gemm(xn, h, &ix.hw, hw, ih, t)?;
                let ng = kv_len / pool;
                let ilog = sc.ilog.get(dev, t * ng * 4)?.f();
                let hscale = ((ih as f64).powf(-0.5) * (id as f64).powf(-0.5)) as f32;
                cu!(memra_hy4r_index_logits(
                    qi,
                    hw,
                    gkc.f(),
                    qpos,
                    t as i32,
                    ih as i32,
                    id as i32,
                    pool as i32,
                    ng as i32,
                    hscale,
                    ilog,
                    sp
                ))?;
                let kp = ((ip.top_k as usize) / pool - 1).min(ng);
                let ks = kp + 1;
                let ipick = sc.ipick.get(dev, t * kp * 4)?.i();
                cu!(memra_hy4r_topk(
                    ilog,
                    ng as i64,
                    t as i32,
                    ng as i32,
                    kp as i32,
                    i32::from(self.red.ties_reversed),
                    ipick,
                    kp as i64,
                    sp
                ))?;
                let isel = sc.isel.get(dev, t * ks * 4)?.i();
                let ldl = ks * pool;
                let ilist = sc.ilist.get(dev, t * ldl * 4)?.i();
                let ilens = sc.ilens.get(dev, t * 4)?.i();
                cu!(memra_hy4r_recall(
                    ipick,
                    kp as i32,
                    qpos,
                    t as i32,
                    pool as i32,
                    isel,
                    ilist,
                    ldl as i32,
                    ilens,
                    sp
                ))?;
                if self.recording || forced.is_some() {
                    rec.recalled = Some((
                        ks,
                        ng,
                        dev.down::<i32>(isel as *const _, t * ks)?,
                        dev.down::<f32>(ilog as *const _, t * ng)?,
                    ));
                }
                if let Some(f) = forced {
                    let fsel = f.indexer.get(&il).ok_or(format!(
                        "layer {il}: the forcing tape has no indexer record"
                    ))?;
                    // each query's forced recall: `ks` sorted groups of the call (the own group
                    // may be the open one, `ng`)
                    if fsel.len() != t * ks
                        || fsel.chunks(ks).any(|row| {
                            row.windows(2).any(|w| w[0] > w[1])
                                || row.iter().any(|g| *g < 0 || *g as usize > ng)
                        })
                    {
                        return Err(format!(
                            "layer {il}: the forced recall is not {ks} sorted groups of the call's {ng} a query"
                        ));
                    }
                    dev.put(isel as *mut _, fsel)?;
                    cu!(memra_hy4r_recall_forced(
                        isel,
                        ks as i32,
                        qpos,
                        t as i32,
                        pool as i32,
                        ilist,
                        ldl as i32,
                        ilens,
                        sp
                    ))?;
                }
                list_mode = true;
                lens_p = ilens;
                list_p = ilist;
                max_len = ldl;
            }
        }
        let qn = t;
        let keys_max = if list_mode {
            max_len
        } else if window > 0 {
            window.min(kv_len)
        } else {
            kv_len
        };
        let chunks = keys_max.div_ceil(ATTN_CHUNK).max(1);
        let slab = ATTN_SLAB.min(qn);
        let po = sc.po.get(dev, slab * heads * chunks * 512 * 4)?.f();
        let pml = sc.pml.get(dev, slab * heads * chunks * 2 * 4)?.f();
        let ldl = max_len;
        for s0 in (0..qn).step_by(ATTN_SLAB) {
            let rows = ATTN_SLAB.min(qn - s0);
            cu!(memra_hy4r_attn(
                qcat.wrapping_add(s0 * heads * 576),
                kvp as *const _,
                if list_mode {
                    list_p.wrapping_add(s0 * ldl)
                } else {
                    std::ptr::null_mut()
                },
                ldl as i32,
                if list_mode {
                    lens_p.wrapping_add(s0)
                } else {
                    std::ptr::null_mut()
                },
                qpos.wrapping_add(s0),
                i32::from(list_mode),
                window as i32,
                ring as i32,
                base as i32,
                rows as i32,
                heads as i32,
                chunks as i32,
                scale,
                sinks,
                po,
                pml,
                olat.wrapping_add(s0 * heads * 512),
                sp
            ))?;
        }
        // values: o[h] = kv_b_value[h] o_lat[h]
        cu!(memra_hy4r_gemm(
            olat,
            (heads * 512) as i64,
            512,
            a.kvb.at(192 * kvl * 2),
            (448 * kvl * 2) as i64,
            std::ptr::null(),
            0,
            std::ptr::null(),
            FMT_BF16,
            0,
            o,
            (heads * hd) as i64,
            hd as i64,
            t as i32,
            256,
            kvl as i32,
            heads as i32,
            std::ptr::null(),
            0,
            0,
            i32::from(t <= 16),
            sp
        ))?;
        cu!(memra_hy4r_act(
            gate,
            o,
            og,
            (t * heads * hd) as i64,
            -1.0,
            1,
            sp
        ))?;
        route.gemm(og, heads * hd, &a.o, ao, h, t)?;
        cu!(memra_hy4r_rms(
            ao,
            h as i64,
            a.post_norm.f(),
            aout,
            h as i64,
            t as i32,
            h as i32,
            a.plan.post_projection_norm.epsilon,
            sp
        ))?;
        Ok(aout)
    }

    fn mlp(
        &mut self,
        il: usize,
        t: usize,
        xn: *mut f32,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
    ) -> Result<*mut f32, String> {
        let h = self.plan.hidden_size as usize;
        let dev = &self.dev;
        let sc = &mut self.sc;
        let sp = dev.sp;
        let l = &self.layers[il];
        let eps = l.res.mlp_output_norm.epsilon;
        let route = Gm {
            dev: &self.dev,
            red: self.red,
        };
        let out = sc.m.get(dev, t * h * 4)?.f();
        match &l.mlp {
            Mlp::Dense {
                gate,
                up,
                down,
                inter,
                norm,
            } => {
                let g = sc.dg.get(dev, t * inter * 4)?.f();
                let u = sc.du.get(dev, t * inter * 4)?.f();
                let act = sc.da.get(dev, t * inter * 4)?.f();
                route.gemm(xn, h, gate, g, *inter, t)?;
                route.gemm(xn, h, up, u, *inter, t)?;
                cu!(memra_hy4r_act(g, u, act, (t * inter) as i64, -1.0, 0, sp))?;
                let y = sc.sum.get(dev, t * h * 4)?.f();
                route.gemm(act, *inter, down, y, h, t)?;
                cu!(memra_hy4r_rms(
                    y,
                    h as i64,
                    norm.f(),
                    out,
                    h as i64,
                    t as i32,
                    h as i32,
                    eps,
                    sp
                ))?;
            }
            Mlp::Moe {
                router,
                bias,
                rdown,
                rup,
                gate,
                up,
                down,
                sgate,
                sup,
                sdown,
                norm,
                experts,
                topk,
                lat,
                shared,
                limit,
                normalize,
                scaling,
            } => {
                let (e, k, lat, sh) = (*experts, *topk, *lat, *shared);
                let rl = sc.rl.get(dev, t * e * 4)?.f();
                route.gemm_f32(xn, router, rl, t)?;
                let rsc = sc.rsc.get(dev, t * e * 4)?.f();
                let rch = sc.rch.get(dev, t * e * 4)?.f();
                cu!(memra_hy4r_router_scores(
                    rl,
                    bias.as_ref().map_or(std::ptr::null(), |b| b.f()),
                    rsc,
                    rch,
                    t as i32,
                    e as i32,
                    sp
                ))?;
                let rpick = sc.rpick.get(dev, t * k * 4)?.i();
                cu!(memra_hy4r_topk(
                    rch, e as i64, t as i32, e as i32, k as i32, 0, rpick, k as i64, sp
                ))?;
                if self.recording || forced.is_some() {
                    rec.routes = Some((
                        k,
                        e,
                        dev.down::<i32>(rpick as *const _, t * k)?,
                        dev.down::<f32>(rch as *const _, t * e)?,
                    ));
                }
                if let Some(f) = forced {
                    let fp = f
                        .router
                        .get(&il)
                        .ok_or(format!("layer {il}: the forcing tape has no router record"))?;
                    // each query's forced experts: `k` of this layer, sorted, unique, in range
                    if fp.len() != t * k
                        || fp.chunks(k).any(|row| {
                            row.windows(2).any(|w| w[0] >= w[1])
                                || row.iter().any(|x| *x < 0 || *x as usize >= e)
                        })
                    {
                        return Err(format!(
                            "layer {il}: the forced experts are not {k} sorted unique experts of {e} a query"
                        ));
                    }
                    dev.put(rpick as *mut _, fp)?;
                }
                let rw = sc.rw.get(dev, t * k * 4)?.f();
                cu!(memra_hy4r_router_weights(
                    rsc,
                    t as i32,
                    e as i32,
                    rpick,
                    k as i32,
                    i32::from(*normalize),
                    *scaling,
                    rw,
                    sp
                ))?;
                let rin = sc.rin.get(dev, t * lat * 4)?.f();
                route.gemm(xn, h, rdown, rin, lat, t)?;
                let rows = t * k;
                let small = rows <= 16;
                let bm = if small { 16 } else { 64 };
                let max_tiles = rows.div_ceil(bm) + e.min(rows);
                let counts = sc.counts.get(dev, e * 4)?.i();
                let cursor = sc.cursor.get(dev, e * 4)?.i();
                let tiles = sc.tiles.get(dev, max_tiles * 3 * 4)?.i();
                let rowmap = sc.rowmap.get(dev, rows * 4)?.i();
                let xg = sc.xg.get(dev, rows * lat * 4)?.f();
                cu!(memra_hy4r_moe_group(
                    rpick,
                    t as i32,
                    k as i32,
                    e as i32,
                    bm as i32,
                    max_tiles as i32,
                    rin,
                    lat as i32,
                    counts,
                    cursor,
                    tiles,
                    rowmap,
                    xg,
                    sp
                ))?;
                let mg = sc.mg.get(dev, rows * lat * 4)?.f();
                let mu = sc.mu.get(dev, rows * lat * 4)?.f();
                let ma = sc.ma.get(dev, rows * lat * 4)?.f();
                let my = sc.my.get(dev, rows * lat * 4)?.f();
                let grouped = |x: *const f32, w: &W, y: *mut f32| -> Result<(), String> {
                    let red = i32::from(route.red.dequant_bf16);
                    cu!(memra_hy4r_gemm(
                        x,
                        w.k as i64,
                        0,
                        w.data.v(),
                        w.wbs as i64,
                        w.scales.as_ref().unwrap().v(),
                        w.scbs as i64,
                        w.scale2.as_ref().unwrap().f(),
                        w.fmt,
                        0,
                        y,
                        w.n as i64,
                        0,
                        rows as i32,
                        w.n as i32,
                        w.k as i32,
                        1,
                        tiles,
                        max_tiles as i32,
                        red,
                        i32::from(small),
                        sp
                    ))
                };
                grouped(xg, gate, mg)?;
                grouped(xg, up, mu)?;
                cu!(memra_hy4r_act(
                    mg,
                    mu,
                    ma,
                    (rows * lat) as i64,
                    *limit,
                    0,
                    sp
                ))?;
                grouped(ma, down, my)?;
                let macc = sc.macc.get(dev, t * lat * 4)?.f();
                cu!(memra_hy4r_moe_combine(
                    my, lat as i32, rowmap, rw, t as i32, k as i32, macc, sp
                ))?;
                let routed = sc.routed.get(dev, t * h * 4)?.f();
                route.gemm(macc, lat, rup, routed, h, t)?;
                let sg = sc.sg.get(dev, t * sh * 4)?.f();
                let su = sc.su.get(dev, t * sh * 4)?.f();
                let sa = sc.sa.get(dev, t * sh * 4)?.f();
                let sd = sc.sd.get(dev, t * h * 4)?.f();
                route.gemm(xn, h, sgate, sg, sh, t)?;
                route.gemm(xn, h, sup, su, sh, t)?;
                cu!(memra_hy4r_act(sg, su, sa, (t * sh) as i64, -1.0, 0, sp))?;
                route.gemm(sa, sh, sdown, sd, h, t)?;
                let sum = sc.sum.get(dev, t * h * 4)?.f();
                cu!(memra_hy4r_add(
                    sum,
                    routed,
                    sd,
                    std::ptr::null(),
                    (t * h) as i64,
                    h as i32,
                    sp
                ))?;
                cu!(memra_hy4r_rms(
                    sum,
                    h as i64,
                    norm.f(),
                    out,
                    h as i64,
                    t as i32,
                    h as i32,
                    eps,
                    sp
                ))?;
            }
        }
        Ok(out)
    }

    /// `HYV4FlashEngramLayer` on the call's rows: the n-gram hashes and the table reads on the
    /// host, the rest on the device; adds into the streams.
    fn engram(
        &mut self,
        il: usize,
        p0: usize,
        t: usize,
        s: *mut f32,
        rec: &mut LayerRec,
    ) -> Result<(), String> {
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let eg = self.layers[il].engram.as_ref().unwrap();
        let ep = &eg.plan;
        let (heads, dim) = (ep.heads as usize, ep.head_dim as usize);
        let width = heads * dim;
        // n-grams over the history: a window that starts `max_ngram - 1` tokens back
        let back = (ep.max_ngram as usize).saturating_sub(1);
        let w0 = p0.saturating_sub(back);
        let ids = &self.sess.history[w0..p0 + t];
        let positions: Vec<u32> = (w0..p0 + t).map(|p| p as u32).collect();
        let ngrams = hashing::ngram_tokens(ep, ids, &positions);
        let hashes = hashing::hash_ids(ep, &eg.lookup, &eg.multipliers, &eg.moduli, &ngrams)?;
        let skip = p0 - w0;
        let rows: Vec<Vec<u64>> = hashes
            .chunks(heads)
            .skip(skip)
            .map(|r| r.iter().zip(&eg.offsets).map(|(a, b)| a + b).collect())
            .collect();
        let mask: Vec<u8> = hashing::engram_mask(ep, &self.sess.history[p0..p0 + t])
            .into_iter()
            .map(u8::from)
            .collect();
        let total = ep.table_rows as usize;
        let mut codes = vec![0u8; t * width];
        let mut scales = vec![0f32; t * heads];
        {
            // each row's bytes from the mapped shard, its scale from the mapped scale tensor
            let tb = &eg.table.0[eg.table.1..eg.table.1 + total * dim];
            let sbytes = &eg.scales.0[eg.scales.1..eg.scales.1 + total * 4];
            let flat: Vec<(usize, u64)> = rows
                .iter()
                .flat_map(|r| r.iter().copied())
                .enumerate()
                .collect();
            for (_, ri) in &flat {
                if *ri as usize >= total {
                    return Err(format!(
                        "layer {il}: Engram row {ri} past the table's {total}"
                    ));
                }
            }
            let threads = 16usize.min(flat.len()).max(1);
            let per = flat.len().div_ceil(threads);
            #[allow(clippy::type_complexity)]
            let results: Vec<Result<Vec<(usize, Vec<u8>, f32)>, String>> =
                std::thread::scope(|sc| {
                    let hs: Vec<_> = flat
                        .chunks(per)
                        .map(|chunk| {
                            sc.spawn(move || {
                                let mut out = Vec::with_capacity(chunk.len());
                                for &(i, ri) in chunk {
                                    let r = ri as usize;
                                    let c = tb[r * dim..(r + 1) * dim].to_vec();
                                    let sb = &sbytes[r * 4..r * 4 + 4];
                                    out.push((
                                        i,
                                        c,
                                        f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]),
                                    ));
                                }
                                Ok(out)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
            for r in results {
                for (i, c, sv) in r? {
                    codes[i * dim..(i + 1) * dim].copy_from_slice(&c);
                    scales[i] = sv;
                }
            }
        }
        rec.engram_rows = Some(rows);
        if self.red.engram_off {
            return Ok(());
        }
        let dev = &self.dev;
        let sc = &mut self.sc;
        let sp = dev.sp;
        let eps = ep.norm_epsilon.value();
        let cb = sc.ecodes.get(dev, t * width)?.v();
        dev.put(cb, &codes)?;
        let sb = sc.escales.get(dev, t * heads * 4)?.f();
        dev.put(sb as *mut _, &scales)?;
        let mb = sc.emask.get(dev, t)?.v();
        dev.put(mb, &mask)?;
        let emb = sc.emb.get(dev, t * width * 4)?.f();
        cu!(memra_hy4r_engram_rows(
            cb,
            sb,
            eg.head_norms.f(),
            t as i32,
            heads as i32,
            dim as i32,
            eps,
            emb,
            sp
        ))?;
        let value = sc.evalue.get(dev, t * h * 4)?.f();
        let route = Gm {
            dev: &self.dev,
            red: self.red,
        };
        route.gemm(emb, width, &eg.value, value, h, t)?;
        cu!(memra_hy4r_add(
            value,
            value,
            std::ptr::null(),
            eg.value_bias.f(),
            (t * h) as i64,
            h as i32,
            sp
        ))?;
        let main = sc.emain.get(dev, t * h * 4)?.f();
        // `pre_gamma * unified_rmsnorm(value)`: the weightless norm's `1 * (x * inv)` is `x * inv`
        // exactly, so the gamma enters as the norm's weight
        cu!(memra_hy4r_rms(
            value,
            h as i64,
            eg.pre_gamma.f(),
            main,
            h as i64,
            t as i32,
            h as i32,
            eps,
            sp
        ))?;
        let keys = sc.ekeys.get(dev, n * t * h * 4)?.f();
        for (j, (kw, kb)) in eg.keys.iter().enumerate() {
            let kj = keys.wrapping_add(j * t * h);
            route.gemm(emb, width, kw, kj, h, t)?;
            cu!(memra_hy4r_add(
                kj,
                kj,
                std::ptr::null(),
                kb.f(),
                (t * h) as i64,
                h as i32,
                sp
            ))?;
        }
        cu!(memra_hy4r_engram_mix(
            s,
            keys,
            main,
            eg.norm_key.f(),
            eg.norm_query.f(),
            eg.hc_scales.f(),
            mb,
            t as i32,
            n as i32,
            h as i32,
            eps,
            sp
        ))?;
        Ok(())
    }
}

fn cs_pos0_of(p0: usize) -> usize {
    p0.saturating_sub(3)
}

/// Which layers of the plan run the indexer (for the tape's header).
pub fn indexed_layers(plan: &ModelPlan) -> Vec<usize> {
    plan.layers
        .iter()
        .filter(|l| matches!(&l.attention, AttentionPlan::HyV4(x) if matches!(x.kind, HyV4AttentionKind::Indexed(_))))
        .map(|l| l.index as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn closing_groups_cover_each_group_once() {
        // groups whose last key (offset `pool - 1`) lands in [p0, p0 + t) close in that call
        let pool = 4usize;
        let last = pool - 1;
        let mut seen = vec![0u32; 64];
        let mut p0 = 0;
        for t in [5usize, 1, 1, 1, 3, 7, 1, 1, 2, 9, 1, 1, 1, 1, 13] {
            let lo = (p0 + pool - 1 - last) / pool;
            let hi = (p0 + t + pool - 1 - last) / pool;
            for (g, c) in seen.iter_mut().enumerate().take(hi).skip(lo) {
                assert!(g * pool + last >= p0 && g * pool + last < p0 + t);
                *c += 1;
            }
            p0 += t;
        }
        for (g, c) in seen.iter().enumerate() {
            assert_eq!(*c, u32::from(g * pool + last < p0), "group {g}");
        }
    }
}
