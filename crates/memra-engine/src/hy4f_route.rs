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
    /// One output of `memra_hy4r_gemm_multi` (`RSeg` in the kernel).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct RSeg {
        pub x: *const f32,
        pub ldx: i64,
        pub w: *const c_void,
        pub sc: *const c_void,
        pub s2: *const f32,
        pub y: *mut f32,
        pub ldy: i64,
        pub n: i32,
        pub grid_cols: i32,
        pub gx0: i32,
    }

    unsafe extern "C" {
        pub fn memra_hy4r_gemm_multi(
            segs: *const RSeg,
            nseg: i32,
            fmt: i32,
            xbs: i64,
            wbs: i64,
            scbs: i64,
            ybs: i64,
            m: i32,
            k: i32,
            batch: i32,
            tiles: *const i32,
            ntiles: i32,
            red_dq: i32,
            tile_rows: i32,
            stream: *mut c_void,
        ) -> i32;
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
            tile_rows: i32,
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
            dots: *mut f32,
            norm_w: *const f32,
            norm_eps: f32,
            xn: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_gemm_f32(
            x: *const f32,
            ldx: i64,
            w: *const c_void,
            y: *mut f32,
            ldy: i64,
            m: i32,
            n: i32,
            k: i32,
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
        pub fn memra_hy4r_route(
            logits: *const f32,
            bias: *const f32,
            scores: *mut f32,
            choice: *mut f32,
            rows: i32,
            e: i32,
            k: i32,
            picks: *mut i32,
            normalize: i32,
            scaling: f32,
            w: *mut f32,
            plan: i32,
            tiles: *mut i32,
            rowmap: *mut i32,
            max_tiles: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_moe_gather(
            x: *const f32,
            d: i32,
            rowmap: *const i32,
            n: i32,
            k: i32,
            xg: *mut f32,
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
        pub fn memra_hy4r_tile_bytes(rows: i64, rb: i64) -> i64;
        pub fn memra_hy4r_tile(
            src: *const c_void,
            rows: i64,
            rb: i64,
            src_ld: i64,
            src_bs: i64,
            dst: *mut c_void,
            dst_bs: i64,
            batch: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_hy4r_pinned(p: *mut *mut c_void, bytes: usize) -> i32;
        pub fn memra_hy4r_pinned_free(p: *mut c_void) -> i32;
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
/// Outputs one small-row launch takes (`R_MAXSEG`).
const MULTI_MAX: usize = 6;

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

    /// Hand the device pool's freed blocks back to the system (`cuMemPoolTrimTo(0)`): the server's
    /// engine pins the pool's release threshold high, so a freed session's cache would otherwise
    /// stay mapped and out of MemAvailable. Live allocations are untouched.
    pub fn trim_pool(&self) -> Result<(), String> {
        self.sync()?;
        use cudarc::driver::sys;
        unsafe {
            let mut pool = std::ptr::null_mut();
            let rc = sys::cuDeviceGetDefaultMemPool(&mut pool, self.st.context().cu_device());
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("the device's default memory pool: {rc:?}"));
            }
            let rc = sys::cuMemPoolTrimTo(pool, 0);
            if rc != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("trim the device memory pool: {rc:?}"));
            }
        }
        Ok(())
    }

    /// Free a staging buffer (stream-ordered) and take it off the resident count.
    fn release(&self, b: DBuf) {
        self.allocated
            .set(self.allocated.get().saturating_sub(b.bytes));
        drop(b);
    }

    /// `batch` entries of `rows` rows of `rb` bytes (`src_ld` apart, entries `src_bs` apart) in
    /// the GEMM weights' tile layout (`r_toff` in the kernel): returns the buffer and an entry's
    /// bytes.
    fn tiled(
        &self,
        src: *const std::os::raw::c_void,
        rows: usize,
        rb: usize,
        src_ld: usize,
        src_bs: usize,
        batch: usize,
    ) -> Result<(DBuf, usize), String> {
        let per = unsafe { ffi::memra_hy4r_tile_bytes(rows as i64, rb as i64) } as usize;
        let b = self.alloc(per * batch)?;
        cu!(memra_hy4r_tile(
            src,
            rows as i64,
            rb as i64,
            src_ld as i64,
            src_bs as i64,
            b.v(),
            per as i64,
            batch as i32,
            self.sp
        ))?;
        Ok((b, per))
    }

    /// `tiled` for a batch whose GEMM calls stride it as plain rows (`rows * rb` bytes an
    /// entry): refused unless the tile layout adds no padding rows (rows a multiple of 32).
    fn tiled_dense(
        &self,
        src: *const std::os::raw::c_void,
        rows: usize,
        rb: usize,
        src_ld: usize,
        src_bs: usize,
        batch: usize,
    ) -> Result<DBuf, String> {
        let (b, per) = self.tiled(src, rows, rb, src_ld, src_bs, batch)?;
        if per != rows * rb {
            return Err(format!(
                "tiled batch of {rows} rows x {rb} bytes is {per} bytes an entry, not {}",
                rows * rb
            ));
        }
        Ok(b)
    }
}

/// Page-locked host staging, grown on demand: an upload from it runs in stream order without
/// waiting for the stream. Growth first waits for the stream (an earlier upload may still read
/// the old buffer).
#[derive(Default)]
struct Pinned {
    p: Option<std::ptr::NonNull<u8>>,
    bytes: usize,
}

impl Pinned {
    fn get(&mut self, dev: &Dev, bytes: usize) -> Result<&mut [u8], String> {
        if self.bytes < bytes {
            dev.sync()?;
            self.free();
            admit(bytes, "a pinned staging buffer")?;
            let mut p: *mut std::os::raw::c_void = std::ptr::null_mut();
            cu!(memra_hy4r_pinned(&mut p, bytes))?;
            self.p = std::ptr::NonNull::new(p as *mut u8);
            self.bytes = bytes;
        }
        let p = self.p.ok_or("pinned staging: no buffer")?;
        Ok(unsafe { std::slice::from_raw_parts_mut(p.as_ptr(), bytes) })
    }

    fn free(&mut self) {
        if let Some(p) = self.p.take() {
            unsafe { ffi::memra_hy4r_pinned_free(p.as_ptr() as *mut _) };
        }
        self.bytes = 0;
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        self.free();
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
    /// device staging for a GEMM weight's stored rows before it is tiled, reused across tensors
    dstage: std::cell::RefCell<Option<DBuf>>,
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
            dstage: std::cell::RefCell::new(None),
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

    /// A tensor's stored bytes into the device staging buffer (grown as needed); returns its
    /// device pointer. Valid until the next staging upload (stream-ordered).
    fn upload_staged(
        &self,
        dev: &Dev,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<*mut std::os::raw::c_void, String> {
        let len = self.check(name, dtype, shape)?;
        let p = self.staging(dev, len)?;
        self.stream(name, |off, c| {
            dev.up_bytes((p as u64 + off as u64) as *mut _, c)
        })?;
        Ok(p)
    }

    /// The device staging buffer, at least `len` bytes.
    fn staging(&self, dev: &Dev, len: usize) -> Result<*mut std::os::raw::c_void, String> {
        let mut ds = self.dstage.borrow_mut();
        if ds.as_ref().is_none_or(|b| b.bytes < len) {
            if let Some(old) = ds.take() {
                dev.release(old);
            }
            *ds = Some(dev.alloc(len)?);
        }
        Ok(ds.as_ref().unwrap().v())
    }

    /// Host bytes into the device staging buffer.
    fn stage_host(&self, dev: &Dev, bytes: &[u8]) -> Result<*mut std::os::raw::c_void, String> {
        let p = self.staging(dev, bytes.len())?;
        dev.up_bytes(p, bytes)?;
        Ok(p)
    }

    /// Free the device staging buffer (after the last weight).
    pub fn release_staging(&self, dev: &Dev) {
        if let Some(b) = self.dstage.borrow_mut().take() {
            dev.release(b);
        }
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

    /// A GEMM weight `[n, k]` in the format the plan names for it, resident in the tile layout
    /// the route's GEMMs read (`Dev::tiled`).
    fn weight(
        &self,
        dev: &Dev,
        name: &str,
        n: usize,
        k: usize,
        format: HyV4WeightFormat,
    ) -> Result<W, String> {
        self.weight_rows(dev, name, n, k, format, true)
    }

    /// A linear weight `[n, k]` in the format the plan names for it: tiled for the route's
    /// GEMMs, or with its rows as stored (the F32 islands' GEMM reads those).
    fn weight_rows(
        &self,
        dev: &Dev,
        name: &str,
        n: usize,
        k: usize,
        format: HyV4WeightFormat,
        tiled: bool,
    ) -> Result<W, String> {
        let dtype = self.dtype(name)?;
        let data = |dt: &str, rb: usize| -> Result<DBuf, String> {
            if tiled {
                let p = self.upload_staged(dev, name, dt, &[n, k])?;
                Ok(dev.tiled(p, n, rb, rb, 0, 1)?.0)
            } else {
                self.upload(dev, name, dt, &[n, k])
            }
        };
        match (format, dtype.as_str()) {
            (HyV4WeightFormat::Bf16, "BF16") => Ok(W {
                data: data("BF16", k * 2)?,
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
                    data: data("F8_E4M3", k)?,
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
        let rows =
            self.upload_staged(dev, &format!("{stem}.weight"), "U8", &[experts, n, k / 2])?;
        let (data, per) = dev.tiled(rows, n, k / 2, k / 2, n * k / 2, experts)?;
        Ok(W {
            data,
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
            wbs: per,
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
    /// `kv_b_proj`'s nope rows per head, transposed: `[heads, kv_lora, nope]` BF16 (exact),
    /// tiled per head
    kb_t: DBuf,
    /// `kv_b_proj`'s value rows per head `[heads, 256, kv_lora]`, tiled per head
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
    /// staging for this layer's uploads (its own, so a later layer's host writes never race an
    /// earlier layer's pending copy)
    stage: Pinned,
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
    /// the call's index tables (`CallIdx`)
    idx: Grow,
    /// ring slots of a verify call's backup and rollback (the drafter's rounds)
    ring_rows: Grow,
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
    /// an Engram layer's codes, row scales and mask (one upload)
    ecodes: Grow,
    emb: Grow,
    evalue: Grow,
    ekeys: Grow,
    emain: Grow,
    hx: Grow,
    hn: Grow,
    /// a batched call's rows the head scores, gathered
    hsel: Grow,
    logits: Grow,
    /// the hyper-connection reductions of a call's rows: 256 partials each
    hcd: Grow,
}

/// A call's index tables in one device buffer (`CallIdx::build` lays them out on the host, the
/// call uploads them once): device pointers into it, each `[rows]` i32 over the call's rows in
/// part order unless noted.
#[derive(Default, Clone)]
struct CallIdx {
    tok: u64,
    qpos: u64,
    /// the row of the position in the call's cos/sin table (its part's sub-table)
    posrow: u64,
    /// linear cache rows the call writes (`p + late`)
    kv_rows: u64,
    /// per part, per ring width
    rings: Vec<Vec<RingIdx>>,
}

#[derive(Default, Clone, Copy)]
struct RingIdx {
    w: usize,
    /// decode: the ring slot of the position (`(p0 + late) % w`), one value
    slot: u64,
    /// prefill: ring slots of the earlier positions in the window `[p0 - b0]`
    lin: u64,
    /// prefill: ring slots of the call's last `min(t, w)` positions
    keep: u64,
}

impl CallIdx {
    /// The host table and the offsets (in i32 elements, the device base still to add). A call of
    /// one part lays it out as a one-session call always has.
    fn build(segs: &[Seg], rings: &[usize]) -> (Vec<i32>, Self) {
        let mut v: Vec<i32> = Vec::new();
        let at = |v: &mut Vec<i32>, it: &mut dyn Iterator<Item = i32>| -> u64 {
            let o = v.len() as u64;
            v.extend(it);
            o
        };
        let span = |s: &Seg| s.p0..s.p0 + s.t();
        let tok = at(
            &mut v,
            &mut segs.iter().flat_map(|s| s.tokens.iter().map(|x| *x as i32)),
        );
        let qpos = at(
            &mut v,
            &mut segs.iter().flat_map(|s| span(s).map(|p| p as i32)),
        );
        let posrow = at(
            &mut v,
            &mut segs
                .iter()
                .flat_map(|s| span(s).map(move |p| (s.cs0 + p - s.cs_pos0) as i32)),
        );
        let kv_rows = at(
            &mut v,
            &mut segs
                .iter()
                .flat_map(|s| span(s).map(move |p| (p + s.late) as i32)),
        );
        let mut out = Vec::new();
        for sg in segs {
            let (p0, t, late) = (sg.p0, sg.t(), sg.late);
            let mut per = Vec::new();
            for &w in rings {
                let slot = at(&mut v, &mut std::iter::once(((p0 + late) % w) as i32));
                let b0 = p0.saturating_sub(w - 1);
                let lin = at(&mut v, &mut (b0..p0).map(|p| (p % w) as i32));
                let keep = t.min(w);
                let kp = at(&mut v, &mut (p0 + t - keep..p0 + t).map(|p| (p % w) as i32));
                per.push(RingIdx {
                    w,
                    slot,
                    lin,
                    keep: kp,
                });
            }
            out.push(per);
        }
        let ci = Self {
            tok,
            qpos,
            posrow,
            kv_rows,
            rings: out,
        };
        (v, ci)
    }

    fn at(mut self, base: u64) -> Self {
        let f = |o: u64| base + o * 4;
        self.tok = f(self.tok);
        self.qpos = f(self.qpos);
        self.posrow = f(self.posrow);
        self.kv_rows = f(self.kv_rows);
        for per in &mut self.rings {
            for r in per {
                r.slot = f(r.slot);
                r.lin = f(r.lin);
                r.keep = f(r.keep);
            }
        }
        self
    }

    fn ring(&self, j: usize, w: usize) -> Result<RingIdx, String> {
        self.rings
            .get(j)
            .and_then(|per| per.iter().find(|r| r.w == w))
            .copied()
            .ok_or_else(|| format!("no index table for a ring of {w} in part {j}"))
    }
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
            0,
            self.dev.sp
        ))
    }

    /// Several plain GEMMs of one format and K over `m` rows: `(x, ldx, weight, y, ldy)`. A
    /// decode call (`m <= 16`) runs them as one launch (`memra_hy4r_gemm_multi`), each output's
    /// bits those of its own launch; a prefill call runs them one by one.
    fn gemm_multi(
        &self,
        outs: &[(*const f32, usize, &W, *mut f32, usize)],
        m: usize,
    ) -> Result<(), String> {
        let (fmt, k) = (outs[0].2.fmt, outs[0].2.k);
        let uniform = outs.iter().all(|o| o.2.fmt == fmt && o.2.k == k);
        if m > 16 || outs.len() == 1 || outs.len() > MULTI_MAX || !uniform {
            for &(x, ldx, w, y, ldy) in outs {
                self.gemm(x, ldx, w, y, ldy, m)?;
            }
            return Ok(());
        }
        let segs: Vec<ffi::RSeg> = outs
            .iter()
            .map(|&(x, ldx, w, y, ldy)| ffi::RSeg {
                x,
                ldx: ldx as i64,
                w: w.data.v(),
                sc: w.scales.as_ref().map_or(std::ptr::null_mut(), |s| s.v()),
                s2: std::ptr::null(),
                y,
                ldy: ldy as i64,
                n: w.n as i32,
                grid_cols: w.grid_cols,
                gx0: 0,
            })
            .collect();
        let red = i32::from(self.red.dequant_bf16 && fmt != FMT_BF16);
        cu!(memra_hy4r_gemm_multi(
            segs.as_ptr(),
            segs.len() as i32,
            fmt,
            0,
            0,
            0,
            0,
            m as i32,
            k as i32,
            1,
            std::ptr::null(),
            0,
            red,
            0,
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
        if m <= 16 {
            // up to four rows a launch (a batched decode call's router and head rows): every
            // row's chain is its own, so the rows' bits do not depend on the launch they share
            for r0 in (0..m).step_by(4) {
                let mr = 4.min(m - r0);
                cu!(memra_hy4r_gemm_f32(
                    x.wrapping_add(r0 * k),
                    k as i64,
                    wp as *const _,
                    y.wrapping_add(r0 * n),
                    n as i64,
                    mr as i32,
                    n as i32,
                    k as i32,
                    self.dev.sp
                ))?;
            }
            return Ok(());
        }
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
    /// the route's own session, the one [`Self::call`] scores on
    sess: Session,
    /// the current call's sessions and rows, in row order (one for [`Self::call`] and
    /// [`Self::call_in`], one per part for [`Self::call_batch`]); empty between calls
    segs: Vec<Seg>,
    /// per RoPE base: (base, device inv_freq)
    bases: Vec<(f32, DBuf)>,
    sc: Scratch,
    /// the current call records its selections (a device sync per layer)
    recording: bool,
    /// the current call has begun writing its sessions (their histories, then their caches)
    wrote: bool,
    /// the current call's index tables
    ci: CallIdx,
    /// the Engram table readers
    reader: ReadPool,
    /// the DFly2 drafter, when loaded (`dfly::Route::load_drafter`)
    dfly: Option<dfly::Drafter>,
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
    /// the drafter's target index layers: the recall lists of this session's last call's last
    /// rows (`dfly::Route::draft_in` reads them)
    snaps: HashMap<usize, dfly::Snap>,
    /// the first row of this session's rows in its last call (a batched call's rows follow the
    /// parts' order): its last hidden rows sit there in the call's output
    row0: usize,
}

/// One session's rows in a call.
struct Seg {
    sess: Session,
    tokens: Vec<u32>,
    all_logits: bool,
    /// the session's positions before the call
    p0: usize,
    /// the first of its rows among the call's
    row0: usize,
    /// its cos/sin rows: their first position and their first row in the call's table
    cs_pos0: usize,
    cs0: usize,
    /// the `kv_off_by_one` red arm on a one-token part
    late: usize,
}

impl Seg {
    fn t(&self) -> usize {
        self.tokens.len()
    }
}

/// A call's per-row buffers the attention's per-part work reads (each part at its rows).
#[derive(Clone, Copy)]
struct SegRows {
    lat: *mut f32,
    tkv: *mut f32,
    qi: *mut f32,
    hw: *mut f32,
    qcat: *mut f32,
    olat: *mut f32,
}

/// One request's rows of a batched call ([`Route::call_batch`]).
pub struct Part<'a> {
    pub sess: &'a mut Session,
    pub tokens: &'a [u32],
    /// every row's logits; else the last row's
    pub all_logits: bool,
}

impl Session {
    /// A session with no cache: the place a session's slot holds while a call has it.
    fn detached() -> Self {
        Self {
            caches: Vec::new(),
            cap: 0,
            history: Vec::new(),
            broken: false,
            snaps: HashMap::new(),
            row0: 0,
        }
    }

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
                kb_t: {
                    let rows = ck.stage_host(&dev, &kb_t)?;
                    dev.tiled_dense(rows, kvl, nope * 2, nope * 2, kvl * nope * 2, heads)?
                },
                kvb: {
                    // the value rows of each head (192..448 of its 448), tiled per head
                    let rows = ck.stage_host(&dev, &kvb_host)?;
                    dev.tiled_dense(
                        (rows as u64 + (nope * kvl * 2) as u64) as *const _,
                        vd,
                        kvl * 2,
                        kvl * 2,
                        (nope + vd) * kvl * 2,
                        heads,
                    )?
                },
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
                        router: ck.weight_rows(&dev, &m("gate.weight"), e, h, bf, false)?,
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
        ck.release_staging(&dev);
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
            segs: Vec::new(),
            bases,
            sc: Scratch::default(),
            recording: false,
            wrote: false,
            ci: CallIdx::default(),
            reader: ReadPool::new(16),
            dfly: None,
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
            snaps: HashMap::new(),
            row0: 0,
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
            .sum::<usize>()
            + self.snap_bytes()
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
    /// cache and history in the call's place of the route's own.
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

    /// Score several requests' rows in one call: each part's tokens on its own session, its
    /// positions following its own history, its logits as [`Self::call_in`] would return them.
    /// The weights are read once for all the rows; the cache, the indexer and the attention run
    /// part by part on each session's own rows. Every row computes on its own (the route's
    /// row-independence), so each part's logits and cache rows are those of its own
    /// [`Self::call_in`] bit for bit. A call that fails marks every part's session broken.
    pub fn call_batch(&mut self, parts: &mut [Part<'_>]) -> Result<Vec<Vec<f32>>, String> {
        if parts.is_empty() {
            return Ok(Vec::new());
        }
        if !self.segs.is_empty() {
            return Err("a call is already holding sessions".into());
        }
        for p in parts.iter_mut() {
            self.segs.push(Seg {
                sess: std::mem::replace(p.sess, Session::detached()),
                tokens: p.tokens.to_vec(),
                all_logits: p.all_logits,
                p0: 0,
                row0: 0,
                cs_pos0: 0,
                cs0: 0,
                late: 0,
            });
        }
        // a panic below leaves the sessions here and every later call refused (the serve route
        // stops on a panic)
        let out = self.run_segs(None, None);
        for (p, sg) in parts.iter_mut().zip(self.segs.drain(..)) {
            *p.sess = sg.sess;
        }
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
            stage: Pinned::default(),
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
    fn host_bytes(&self, t: usize, all_logits: bool, record: bool, cap: usize) -> usize {
        let v = self.plan.vocab_size as usize;
        let mut host = if all_logits { t } else { 1 } * v * 4;
        if record {
            for l in &self.layers {
                if let Some(ix) = &l.attn.indexer {
                    let ks = (ix.plan.top_k / ix.plan.pool) as usize;
                    host += t * (ks + cap / ix.plan.pool as usize + 1) * 4;
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
        if !self.segs.is_empty() {
            return Err("a call is already holding sessions".into());
        }
        self.segs.push(Seg {
            sess: std::mem::replace(&mut self.sess, Session::detached()),
            tokens: tokens.to_vec(),
            all_logits,
            p0: 0,
            row0: 0,
            cs_pos0: 0,
            cs0: 0,
            late: 0,
        });
        let out = self.run_segs(rec, forced);
        self.sess = self.segs.pop().expect("the call's one part").sess;
        out.map(|mut v| v.pop().unwrap_or_default())
    }

    /// The call of the sessions in `self.segs`.
    fn run_segs(
        &mut self,
        rec: Option<&mut Vec<(usize, LayerRec)>>,
        forced: Option<&Forced>,
    ) -> Result<Vec<Vec<f32>>, String> {
        // a call that fails may have written cache rows (a sliding ring row of an earlier
        // position among them): the route refuses every call on its sessions until `reset`
        // starts a new request
        if self.segs.iter().any(|s| s.sess.broken) {
            return Err(
                "an earlier call failed mid-way; reset the route and replay the request".into(),
            );
        }
        self.wrote = false;
        let out = self.call_inner(rec, forced);
        self.dev.host_reserve.set(0);
        // a call refused before it wrote anything (a part past its cache or the vocabulary, the
        // host output refused) leaves its sessions as they were
        if out.is_err() && self.wrote {
            for s in &mut self.segs {
                s.sess.broken = true;
            }
            // copies queued before the failure may still read the pinned staging buffers:
            // drain the stream before a later call reuses them (the call's error stands)
            let _ = self.dev.sync();
        }
        out
    }

    fn call_inner(
        &mut self,
        mut rec: Option<&mut Vec<(usize, LayerRec)>>,
        forced: Option<&Forced>,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.recording = rec.is_some();
        if self.segs.len() > 1 && (rec.is_some() || forced.is_some()) {
            return Err("records and forced selections take a one-session call".into());
        }
        let t: usize = self.segs.iter().map(Seg::t).sum();
        if t == 0 {
            return Ok(vec![Vec::new(); self.segs.len()]);
        }
        let v = self.plan.vocab_size as usize;
        let mut host = 0usize;
        let (mut row0, mut cs0) = (0usize, 0usize);
        for sg in &mut self.segs {
            let (p0, st) = (sg.sess.history.len(), sg.t());
            if st == 0 {
                return Err("a batched call's part has no tokens".into());
            }
            if p0 + st > sg.sess.cap {
                return Err(format!(
                    "{} positions past the cache's {}",
                    p0 + st,
                    sg.sess.cap
                ));
            }
            if let Some(tk) = sg.tokens.iter().find(|tk| **tk as usize >= v) {
                return Err(format!("token {tk} past the vocabulary {v}"));
            }
            sg.p0 = p0;
            sg.row0 = row0;
            sg.cs_pos0 = cs_pos0_of(p0);
            sg.cs0 = cs0;
            sg.late = usize::from(self.red.kv_off_by_one && st == 1);
            row0 += st;
            cs0 += p0 + st - sg.cs_pos0;
        }
        // the call's host output (logits, records); its device scratch is admitted where it
        // grows (`Grow::get`)
        for sg in &self.segs {
            host += self.host_bytes(sg.t(), sg.all_logits, rec.is_some(), sg.sess.cap);
        }
        admit(host, "the call's host output")?;
        self.dev.host_reserve.set(host);
        // from here the call writes its sessions
        self.wrote = true;
        for sg in &mut self.segs {
            sg.sess.history.extend_from_slice(&sg.tokens);
            sg.sess.row0 = sg.row0;
        }
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let dev = &self.dev;
        let sc = &mut self.sc;
        // the call's index tables, one upload while the stream is idle (the previous call ended
        // on its logits copy), so no copy inside the call waits for the stream
        let mut rings: Vec<usize> = self.segs[0]
            .sess
            .caches
            .iter()
            .map(|c| c.ring)
            .filter(|r| *r > 0)
            .collect();
        rings.sort_unstable();
        rings.dedup();
        let (table, ci) = CallIdx::build(&self.segs, &rings);
        let idxb = sc.idx.get(dev, table.len() * 4)?.p;
        dev.put(idxb as *mut _, &table)?;
        self.ci = ci.at(idxb);
        let tokb = self.ci.tok;
        // the cos/sin table: each part's positions from 3 before its first, a sub-table a part
        for (bi, (_, inv)) in self.bases.iter().enumerate() {
            let b = sc.cs[bi].get(dev, cs0 * 32 * 8)?;
            for sg in &self.segs {
                cu!(memra_hy4r_rope_table(
                    b.at(sg.cs0 * 32 * 8),
                    sg.cs_pos0 as i32,
                    (sg.p0 + sg.t() - sg.cs_pos0) as i32,
                    32,
                    inv.p as *const f64,
                    dev.sp
                ))?;
            }
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
        // the Engram layers' table reads run on host threads beside the first layers
        let jobs = (0..self.layers.len())
            .filter(|il| self.layers[*il].engram.is_some())
            .map(|il| self.engram_job(il))
            .collect::<Result<Vec<_>, String>>()?;
        let mut fetch = if jobs.is_empty() {
            None
        } else {
            Some(engram_fetch(&self.reader, jobs)?)
        };
        let mut fetched: HashMap<usize, EngramData> = HashMap::new();
        for il in 0..self.layers.len() {
            let mut lr = LayerRec::default();
            let ed = if self.layers[il].engram.is_some() {
                if let Some(f) = fetch.take() {
                    fetched.extend(f.wait()?);
                }
                fetched.remove(&il)
            } else {
                None
            };
            self.layer(il, t, s, &mut lr, forced, ed)?;
            if let Some(r) = rec.as_deref_mut() {
                r.push((il, lr));
            }
        }
        // the exit: hc_head, the final norm, the F32 head
        let dev = &self.dev;
        let sc = &mut self.sc;
        let hx = sc.hx.get(dev, t * h * 4)?.f();
        let hcd = sc.hcd.get(dev, t * 17 * 256 * 4)?.f();
        let numerics = self.plan.hy_v4_flash.unwrap();
        let hn = sc.hn.get(dev, t * h * 4)?.f();
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
            hcd,
            self.final_norm.f(),
            self.final_eps,
            hn,
            dev.sp
        ))?;
        let gm = Gm {
            dev: &self.dev,
            red: self.red,
        };
        if self.segs.len() == 1 {
            let (r0, rows) = if self.segs[0].all_logits {
                (0, t)
            } else {
                (t - 1, 1)
            };
            let lg = self
                .sc
                .logits
                .get(&self.dev, HEAD_ROWS.min(rows) * v * 4)?
                .f();
            let mut out = Vec::with_capacity(rows * v);
            for c0 in (r0..r0 + rows).step_by(HEAD_ROWS) {
                let cr = HEAD_ROWS.min(r0 + rows - c0);
                gm.gemm_f32_raw(hn.wrapping_add(c0 * h), self.lm_head.p, v, h, lg, cr)?;
                out.extend(self.dev.down::<f32>(lg as *const _, cr * v)?);
            }
            return Ok(vec![out]);
        }
        // several parts: their rows the head scores (each part's all or its last), gathered in
        // part order, one head pass over them; each row's logits are its own row's (the head's
        // rows compute on their own)
        let want: Vec<(usize, usize)> = self
            .segs
            .iter()
            .map(|sg| {
                if sg.all_logits {
                    (sg.row0, sg.t())
                } else {
                    (sg.row0 + sg.t() - 1, 1)
                }
            })
            .collect();
        let rows: usize = want.iter().map(|w| w.1).sum();
        let hsel = self.sc.hsel.get(&self.dev, rows * h * 4)?.f();
        let mut at = 0usize;
        for &(r0, nr) in &want {
            cu!(memra_hy4r_copy(
                hn.wrapping_add(r0 * h),
                h as i64,
                hsel.wrapping_add(at * h) as *mut _,
                h as i64,
                nr as i32,
                h as i32,
                0,
                std::ptr::null(),
                self.dev.sp
            ))?;
            at += nr;
        }
        let lg = self
            .sc
            .logits
            .get(&self.dev, HEAD_ROWS.min(rows) * v * 4)?
            .f();
        let mut all = Vec::with_capacity(rows * v);
        for c0 in (0..rows).step_by(HEAD_ROWS) {
            let cr = HEAD_ROWS.min(rows - c0);
            gm.gemm_f32_raw(hsel.wrapping_add(c0 * h), self.lm_head.p, v, h, lg, cr)?;
            all.extend(self.dev.down::<f32>(lg as *const _, cr * v)?);
        }
        let mut out = Vec::with_capacity(want.len());
        let mut at = 0usize;
        for &(_, nr) in &want {
            out.push(all[at * v..(at + nr) * v].to_vec());
            at += nr;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        il: usize,
        t: usize,
        s: *mut f32,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
        engram: Option<EngramData>,
    ) -> Result<(), String> {
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let res = self.layers[il].res;
        if self.layers[il].engram.is_some() {
            let data = engram.ok_or(format!("layer {il}: no Engram reads"))?;
            self.engram(il, t, s, rec, data)?;
        }
        let dev = &self.dev;
        let l = &self.layers[il];
        let post = self.sc.post.get(dev, t * n * 4)?.f();
        let x = self.sc.x.get(dev, t * h * 4)?.f();
        let xn = self.sc.xn.get(dev, t * h * 4)?.f();
        let hcd = self.sc.hcd.get(dev, t * 17 * 256 * 4)?.f();
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
            hcd,
            l.in_norm.f(),
            l.in_eps,
            xn,
            dev.sp
        ))?;
        let a = self.attention(il, t, xn, rec, forced)?;
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
            hcd,
            l.post_attn_norm.f(),
            l.mlp_eps,
            xn,
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
        t: usize,
        xn: *mut f32,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
    ) -> Result<*mut f32, String> {
        let h = self.plan.hidden_size as usize;
        let eps = self.hy.rms_norm_eps;
        let bi = self.base_slot(self.layers[il].attn.plan.rope.base);
        // the parts whose keys pass the indexer budget run it; the projections it reads are
        // computed for every row when any part needs them (each row on its own)
        let over: Vec<bool> = match &self.layers[il].attn.indexer {
            Some(ix) => self
                .segs
                .iter()
                .map(|sg| kv_len_of(sg.p0, sg.t()) > ix.plan.top_k as usize)
                .collect(),
            None => vec![false; self.segs.len()],
        };
        let any_over = over.iter().any(|x| *x);
        let dev = &self.dev;
        let sc = &mut self.sc;
        let l = &self.layers[il];
        let a = &l.attn;
        let heads = a.plan.query_heads as usize;
        let (ql, kvl) = (a.plan.q_lora_rank as usize, a.plan.kv_lora_rank as usize);
        let hd = 256usize;
        let cs = sc.cs[bi].get(dev, 0)?.p as *const std::os::raw::c_void;
        let posrow = self.ci.posrow as *mut i32;
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
        // the projections of xn: one launch for those of one format (a decode call)
        let (tkv, hwb) = match &a.indexer {
            Some(ix) => {
                let (ih, id) = (ix.plan.heads as usize, ix.plan.head_dim as usize);
                (
                    sc.tkv.get(dev, t * id * 4)?.f(),
                    sc.hw.get(dev, t * ih * 4)?.f(),
                )
            }
            None => (std::ptr::null_mut(), std::ptr::null_mut()),
        };
        {
            let mut outs: Vec<(*const f32, usize, &W, *mut f32, usize)> = vec![
                (xn, h, &a.gate, gate, heads * hd),
                (xn, h, &a.qa, qa, ql),
                (xn, h, &a.kva, ckv, 576),
            ];
            if let Some(ix) = &a.indexer {
                let (ih, id) = (ix.plan.heads as usize, ix.plan.head_dim as usize);
                outs.push((xn, h, &ix.tk, tkv, id));
                if any_over {
                    outs.push((xn, h, &ix.hw, hwb, ih));
                }
            }
            let (same, other): (Vec<_>, Vec<_>) =
                outs.into_iter().partition(|o| o.2.fmt == FMT_BF16);
            if !same.is_empty() {
                route.gemm_multi(&same, t)?;
            }
            if !other.is_empty() {
                route.gemm_multi(&other, t)?;
            }
        }
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
            0,
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
        // the indexer's per-row work: the token keys' norm and the queries
        let mut qi: *mut f32 = std::ptr::null_mut();
        if let Some(ix) = &a.indexer {
            let ip = ix.plan;
            let (ih, id) = (ip.heads as usize, ip.head_dim as usize);
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
            if any_over {
                qi = sc.qi.get(dev, t * ih * id * 4)?.f();
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
            }
        }
        // each part on its own session: its cache rows, its indexer selection, its attention
        let rows = SegRows {
            lat,
            tkv,
            qi,
            hw: hwb,
            qcat,
            olat,
        };
        for (j, &ov) in over.iter().enumerate() {
            self.attend_seg(il, j, bi, ov, rows, rec, forced)?;
            self.dfly_after_seg(il, j)?;
        }
        let l = &self.layers[il];
        let a = &l.attn;
        let route = Gm {
            dev: &self.dev,
            red: self.red,
        };
        // values: o[h] = kv_b_value[h] o_lat[h]
        cu!(memra_hy4r_gemm(
            olat,
            (heads * 512) as i64,
            512,
            a.kvb.v(),
            (256 * kvl * 2) as i64,
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
            0,
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

    /// Part `j`'s attention at layer `il` (its rows of the call's per-row buffers `rows`): its
    /// cache rows written, the indexer's selection over its own keys (`over`: its keys pass the
    /// budget), the attention of its queries over its own cache into its rows of `rows.olat`.
    #[allow(clippy::too_many_arguments)]
    fn attend_seg(
        &mut self,
        il: usize,
        j: usize,
        bi: usize,
        over: bool,
        rows: SegRows,
        rec: &mut LayerRec,
        forced: Option<&Forced>,
    ) -> Result<(), String> {
        let eps = self.hy.rms_norm_eps;
        let dev = &self.dev;
        let sc = &mut self.sc;
        let sg = &self.segs[j];
        let (p0, t, r0) = (sg.p0, sg.t(), sg.row0);
        let l = &self.layers[il];
        let a = &l.attn;
        let heads = a.plan.query_heads as usize;
        let sp = dev.sp;
        // the part's rows of the call's buffers and index tables
        let lat = rows.lat.wrapping_add(r0 * 576);
        let qcat = rows.qcat.wrapping_add(r0 * heads * 576);
        let olat = rows.olat.wrapping_add(r0 * heads * 512);
        let qpos = (self.ci.qpos as *mut i32).wrapping_add(r0);
        let kv_rows = (self.ci.kv_rows as *const i32).wrapping_add(r0);
        // the part's cos/sin sub-table (its first row is position `cs_pos0`)
        let cs = sc.cs[bi].get(dev, 0)?.at(sg.cs0 * 32 * 8) as *const std::os::raw::c_void;
        // the cache rows of this call
        let cache = &sg.sess.caches[il];
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
            let ri = self.ci.ring(j, w)?;
            if t == 1 {
                let sb = ri.slot as *const i32;
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
                    let rb = ri.lin as *const i32;
                    cu!(memra_hy4r_gather_bf(
                        cache.kv.v(),
                        576,
                        rb,
                        lin.v(),
                        576,
                        (p0 - b0) as i32,
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
                let rb = ri.keep as *const i32;
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
            cu!(memra_hy4r_copy(
                lat,
                576,
                cache.kv.v(),
                576,
                t as i32,
                576,
                0,
                kv_rows,
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
            let tkv = rows.tkv.wrapping_add(r0 * id);
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
                qpos,
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
            if over {
                let qi = rows.qi.wrapping_add(r0 * ih * id);
                let hw = rows.hw.wrapping_add(r0 * ih);
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
            let nq = ATTN_SLAB.min(qn - s0);
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
                nq as i32,
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
        Ok(())
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
                let rpick = sc.rpick.get(dev, t * k * 4)?.i();
                let rw = sc.rw.get(dev, t * k * 4)?.f();
                let rows = t * k;
                // up to 16 tokens: an expert's tile holds at most one row a token, so the experts
                // run on the small-row kernel's tiles of 16 (a batched decode call's rows)
                let small = t <= 16;
                let bm = if small { 16 } else { 64 };
                let max_tiles = rows.div_ceil(bm) + e.min(rows);
                let tiles = sc.tiles.get(dev, max_tiles * 3 * 4)?.i();
                let rowmap = sc.rowmap.get(dev, rows * 4)?.i();
                // free picks of a small router: one block a row scores, picks and weighs, and
                // with one row plans the experts' tiles; forced picks keep the separate kernels
                let one_block = forced.is_none() && k <= 32 && e <= 1024 && k <= e;
                let planned = one_block && t == 1;
                if one_block {
                    cu!(memra_hy4r_route(
                        rl,
                        bias.as_ref().map_or(std::ptr::null(), |b| b.f()),
                        rsc,
                        rch,
                        t as i32,
                        e as i32,
                        k as i32,
                        rpick,
                        i32::from(*normalize),
                        *scaling,
                        rw,
                        i32::from(planned),
                        tiles,
                        rowmap,
                        max_tiles as i32,
                        sp
                    ))?;
                    if self.recording {
                        rec.routes = Some((
                            k,
                            e,
                            dev.down::<i32>(rpick as *const _, t * k)?,
                            dev.down::<f32>(rch as *const _, t * e)?,
                        ));
                    }
                } else {
                    cu!(memra_hy4r_router_scores(
                        rl,
                        bias.as_ref().map_or(std::ptr::null(), |b| b.f()),
                        rsc,
                        rch,
                        t as i32,
                        e as i32,
                        sp
                    ))?;
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
                }
                let rin = sc.rin.get(dev, t * lat * 4)?.f();
                let sg = sc.sg.get(dev, t * sh * 4)?.f();
                let su = sc.su.get(dev, t * sh * 4)?.f();
                let sa = sc.sa.get(dev, t * sh * 4)?.f();
                // the latent down projection and the shared expert's gate and up read xn
                route.gemm_multi(
                    &[
                        (xn, h, rdown, rin, lat),
                        (xn, h, sgate, sg, sh),
                        (xn, h, sup, su, sh),
                    ],
                    t,
                )?;
                cu!(memra_hy4r_act(sg, su, sa, (t * sh) as i64, -1.0, 0, sp))?;
                let xg = sc.xg.get(dev, rows * lat * 4)?.f();
                if planned {
                    cu!(memra_hy4r_moe_gather(
                        rin,
                        lat as i32,
                        rowmap,
                        rows as i32,
                        k as i32,
                        xg,
                        sp
                    ))?;
                } else {
                    let counts = sc.counts.get(dev, e * 4)?.i();
                    let cursor = sc.cursor.get(dev, e * 4)?.i();
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
                }
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
                        t.min(bm) as i32,
                        sp
                    ))
                };
                if small && gate.wbs == up.wbs && gate.scbs == up.scbs && gate.n == up.n {
                    // a decode call: the experts' gate and up in one launch
                    let seg = |w: &W, y: *mut f32| ffi::RSeg {
                        x: xg,
                        ldx: w.k as i64,
                        w: w.data.v(),
                        sc: w.scales.as_ref().unwrap().v(),
                        s2: w.scale2.as_ref().unwrap().f(),
                        y,
                        ldy: w.n as i64,
                        n: w.n as i32,
                        grid_cols: 0,
                        gx0: 0,
                    };
                    let segs = [seg(gate, mg), seg(up, mu)];
                    cu!(memra_hy4r_gemm_multi(
                        segs.as_ptr(),
                        2,
                        gate.fmt,
                        0,
                        gate.wbs as i64,
                        gate.scbs as i64,
                        0,
                        rows as i32,
                        gate.k as i32,
                        1,
                        tiles,
                        max_tiles as i32,
                        i32::from(route.red.dequant_bf16),
                        t.min(bm) as i32,
                        sp
                    ))?;
                } else {
                    grouped(xg, gate, mg)?;
                    grouped(xg, up, mu)?;
                }
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
                let sd = sc.sd.get(dev, t * h * 4)?.f();
                // the latent up projection and the shared expert's down projection
                route.gemm_multi(&[(macc, lat, rup, routed, h), (sa, sh, sdown, sd, h)], t)?;
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

    /// An Engram layer's reads for the call's rows: the n-gram hashes (on the host, the step 5
    /// code) over each part's own history and the table rows they name, checked against the
    /// table, in the call's row order.
    fn engram_job(&self, il: usize) -> Result<EngramJob, String> {
        let eg = self.layers[il].engram.as_ref().unwrap();
        let ep = &eg.plan;
        let heads = ep.heads as usize;
        let mut rows: Vec<Vec<u64>> = Vec::new();
        let mut mask: Vec<u8> = Vec::new();
        for sg in &self.segs {
            let (p0, t, history) = (sg.p0, sg.t(), &sg.sess.history);
            // n-grams over the history: a window that starts `max_ngram - 1` tokens back
            let back = (ep.max_ngram as usize).saturating_sub(1);
            let w0 = p0.saturating_sub(back);
            let ids = &history[w0..p0 + t];
            let positions: Vec<u32> = (w0..p0 + t).map(|p| p as u32).collect();
            let ngrams = hashing::ngram_tokens(ep, ids, &positions);
            let hashes = hashing::hash_ids(ep, &eg.lookup, &eg.multipliers, &eg.moduli, &ngrams)?;
            let skip = p0 - w0;
            rows.extend(
                hashes
                    .chunks(heads)
                    .skip(skip)
                    .map(|r| r.iter().zip(&eg.offsets).map(|(a, b)| a + b).collect()),
            );
            mask.extend(
                hashing::engram_mask(ep, &history[p0..p0 + t])
                    .into_iter()
                    .map(u8::from),
            );
        }
        let total = ep.table_rows as usize;
        if let Some(ri) = rows.iter().flatten().find(|ri| **ri as usize >= total) {
            return Err(format!(
                "layer {il}: Engram row {ri} past the table's {total}"
            ));
        }
        Ok(EngramJob {
            il,
            rows,
            mask,
            table: eg.table.clone(),
            scales: eg.scales.clone(),
            dim: ep.head_dim as usize,
        })
    }

    /// `HYV4FlashEngramLayer` on the call's rows, the table rows already read (`engram_fetch`);
    /// adds into the streams.
    fn engram(
        &mut self,
        il: usize,
        t: usize,
        s: *mut f32,
        rec: &mut LayerRec,
        data: EngramData,
    ) -> Result<(), String> {
        let h = self.plan.hidden_size as usize;
        let n = self.hy.hc_mult as usize;
        let eg = self.layers[il].engram.as_ref().unwrap();
        let ep = &eg.plan;
        let (heads, dim) = (ep.heads as usize, ep.head_dim as usize);
        let width = heads * dim;
        let EngramData {
            rows,
            codes,
            scales,
            mask,
        } = data;
        if codes.len() != t * width || scales.len() != t * heads || mask.len() != t {
            return Err(format!(
                "layer {il}: the Engram reads are not the call's rows"
            ));
        }
        rec.engram_rows = Some(rows);
        if self.red.engram_off {
            return Ok(());
        }
        let dev = &self.dev;
        let sc = &mut self.sc;
        let sp = dev.sp;
        let eps = ep.norm_epsilon.value();
        // one upload of codes | scales | mask through the layer's pinned staging
        let (oc, os, om) = (0, (t * width).div_ceil(16) * 16, 0);
        let om = om + os + (t * heads * 4).div_ceil(16) * 16;
        let total = om + t;
        let stage = self.layers[il]
            .engram
            .as_mut()
            .unwrap()
            .stage
            .get(dev, total)?;
        stage[oc..oc + t * width].copy_from_slice(&codes);
        for (i, v) in scales.iter().enumerate() {
            stage[os + 4 * i..os + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        stage[om..om + t].copy_from_slice(&mask);
        let eb = sc.ecodes.get(dev, total)?;
        cu!(memra_hy4r_h2d(
            eb.v(),
            stage.as_ptr() as *const _,
            total,
            sp
        ))?;
        let cb = eb.at(oc);
        let sb = eb.at(os) as *mut f32;
        let mb = eb.at(om);
        let eg = self.layers[il].engram.as_ref().unwrap();
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
        let keys = sc.ekeys.get(dev, n * t * h * 4)?.f();
        {
            // the value and the stream keys read the same rows: one launch (a decode call)
            let mut outs: Vec<(*const f32, usize, &W, *mut f32, usize)> =
                vec![(emb, width, &eg.value, value, h)];
            for (j, (kw, _)) in eg.keys.iter().enumerate() {
                outs.push((emb, width, kw, keys.wrapping_add(j * t * h), h));
            }
            route.gemm_multi(&outs, t)?;
        }
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
        for (j, (_, kb)) in eg.keys.iter().enumerate() {
            let kj = keys.wrapping_add(j * t * h);
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

/// One Engram layer's table reads for a call (`Route::engram_job`).
struct EngramJob {
    il: usize,
    rows: Vec<Vec<u64>>,
    mask: Vec<u8>,
    table: (Arc<memmap2::Mmap>, usize),
    scales: (Arc<memmap2::Mmap>, usize),
    dim: usize,
}

/// What `engram_fetch` read for a layer: the rows' FP8 codes `[t, heads, dim]` and scales.
struct EngramData {
    rows: Vec<Vec<u64>>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    mask: Vec<u8>,
}

/// Host threads for the Engram table reads, started with the route: a call's reads go out as
/// tasks (a few rows each) and its results come back on the call's channel. Spawning the threads
/// per call cost more than the reads (about 8 ms a decode call for 128 page-cached `pread`s).
struct ReadPool {
    tx: Option<std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl ReadPool {
    fn new(threads: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let rx = Arc::new(std::sync::Mutex::new(rx));
        let workers = (0..threads)
            .map(|_| {
                let rx = rx.clone();
                std::thread::spawn(move || {
                    loop {
                        let task = match rx.lock() {
                            Ok(r) => r.recv(),
                            Err(_) => return,
                        };
                        match task {
                            Ok(f) => f(),
                            Err(_) => return,
                        }
                    }
                })
            })
            .collect();
        Self {
            tx: Some(tx),
            workers,
        }
    }

    fn submit(&self, f: Box<dyn FnOnce() + Send>) -> Result<(), String> {
        self.tx
            .as_ref()
            .ok_or("the Engram reader is stopped")?
            .send(f)
            .map_err(|_| "the Engram reader is gone".to_string())
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        self.tx = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Rows a read task takes.
const ENGRAM_TASK_ROWS: usize = 4;

type EngramRead = Result<Vec<(usize, usize, Vec<u8>, f32)>, String>;

/// Every Engram layer's table rows of a call, read from the shard mappings on the pool's threads; returns
/// at once, the reads running beside the GPU's first layers (the hashes need only the token
/// ids). `EngramFetch::wait` collects them.
fn engram_fetch(pool: &ReadPool, jobs: Vec<EngramJob>) -> Result<EngramFetch, String> {
    let (tx, rx) = std::sync::mpsc::channel::<EngramRead>();
    let mut tasks = 0usize;
    for (j, job) in jobs.iter().enumerate() {
        let flat: Vec<(usize, u64)> = job.rows.iter().flatten().copied().enumerate().collect();
        for chunk in flat.chunks(ENGRAM_TASK_ROWS) {
            let chunk = chunk.to_vec();
            let (table, scales, dim) = (job.table.clone(), job.scales.clone(), job.dim);
            let tx = tx.clone();
            pool.submit(Box::new(move || {
                let read = || -> EngramRead {
                    // each row's bytes from the mapped shard, its scale from the mapped scale
                    // tensor (`engram_job` checked the rows against the table)
                    let mut res = Vec::with_capacity(chunk.len());
                    for (i, ri) in chunk {
                        let r = ri as usize;
                        let c = table
                            .0
                            .get(table.1 + r * dim..table.1 + (r + 1) * dim)
                            .ok_or("an Engram row past its mapping")?
                            .to_vec();
                        let sb = scales
                            .0
                            .get(scales.1 + r * 4..scales.1 + r * 4 + 4)
                            .ok_or("an Engram scale past its mapping")?;
                        res.push((j, i, c, f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]])));
                    }
                    Ok(res)
                };
                let _ = tx.send(read());
            }))?;
            tasks += 1;
        }
    }
    Ok(EngramFetch { jobs, rx, tasks })
}

/// A call's Engram reads in flight.
struct EngramFetch {
    jobs: Vec<EngramJob>,
    rx: std::sync::mpsc::Receiver<EngramRead>,
    tasks: usize,
}

impl EngramFetch {
    fn wait(self) -> Result<Vec<(usize, EngramData)>, String> {
        let mut out: Vec<(Vec<u8>, Vec<f32>)> = self
            .jobs
            .iter()
            .map(|j| {
                let rows: usize = j.rows.iter().map(Vec::len).sum();
                (vec![0u8; rows * j.dim], vec![0f32; rows])
            })
            .collect();
        for _ in 0..self.tasks {
            let got = self
                .rx
                .recv()
                .map_err(|_| "an Engram read task was dropped".to_string())??;
            for (j, i, c, sv) in got {
                let dim = self.jobs[j].dim;
                out[j].0[i * dim..(i + 1) * dim].copy_from_slice(&c);
                out[j].1[i] = sv;
            }
        }
        Ok(self
            .jobs
            .into_iter()
            .zip(out)
            .map(|(job, (codes, scales))| {
                (
                    job.il,
                    EngramData {
                        rows: job.rows,
                        codes,
                        scales,
                        mask: job.mask,
                    },
                )
            })
            .collect())
    }
}

/// The keys a call's last query sees: everything before it and the call's rows.
fn kv_len_of(p0: usize, t: usize) -> usize {
    p0 + t
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

mod dfly;
pub use dfly::{
    DraftOut, DraftPick, DraftRed, DraftReq, Drafter, ExternalTarget, MAX_DRAFTS, RingBackup,
};
