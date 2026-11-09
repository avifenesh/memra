//! DeepSeek-V4.1-Flash on one GPU: the V4.1 program's kernels (`cu/dsv41_gpu.cu`) and a
//! single-card layer that runs them.
//!
//! A sibling of [`crate::dsv4_gpu`], not a parameterization of it: the DSv4 route is wired to two
//! cards, 43 layers, d 4096, compress ratios {4, 128} and 128-wide quant blocks, and keeps its
//! activations in f32. V4.1 keeps the reference program's bf16 activations, quantizes every
//! FP8/FP4 linear input per 32 values, shares one compressed KV and one indexer across a CSA2
//! layer group, narrows its decoder indexers through a candidate pool, carries the mHC pre-mix
//! from the previous FFN and adds Engram n-gram lookups at two layers. The semantic source is the
//! vendor `inference/model.py` at rev 1bae0e10; `crates/memra-gguf/src/model_plan/deepseek_v41.rs`
//! is the typed plan these kernels execute.
//!
//! Scope today: every V4.1 operation as a kernel plus the prefill/decode layer
//! ([`Ctx::block_forward`]) fed from the safetensors checkpoint, and the same layer split across
//! N ranks ([`tp`]: tensor-parallel attention, expert-parallel MoE, a vocab-parallel head; gathers
//! only, so the N-rank program writes these bytes). Serving is not here yet.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use memra_gguf::safetensors::StModel;
use std::os::raw::c_void;
use std::sync::Arc;

pub mod engram;
pub mod tp;

pub mod ffi {
    use std::os::raw::c_void;
    // Return codes: 0 ok, 10000 + cudaError_t, 41xxx for a refused contract (cu/dsv41_gpu.cu).
    unsafe extern "C" {
        pub fn memra_v41_act_quant(
            x: *const c_void,
            ldx: i64,
            rowidx: *const i32,
            q: *mut c_void,
            ldq: i64,
            s: *mut c_void,
            lds: i64,
            y: *mut c_void,
            ldy: i64,
            rows: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_fp4_qdq(
            x: *mut c_void,
            ld: i64,
            rows: i32,
            n: i32,
            group: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_rmsnorm(
            x: *const c_void,
            ldx: i64,
            w: *const c_void,
            y: *mut c_void,
            ldy: i64,
            rows: i32,
            d: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_rope(
            x: *mut c_void,
            ld_row: i64,
            rows: i32,
            heads: i32,
            hd: i32,
            cosb: *const f32,
            sinb: *const f32,
            pos_base: i32,
            pos_step: i32,
            pos: *const i32,
            inverse: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_f8(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            b: *const c_void,
            ldb: i64,
            b_s: *const c_void,
            ldbs: i64,
            bscale2: f32,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_bf16(
            a: *const c_void,
            lda: i64,
            s_az: i64,
            b: *const c_void,
            ldb: i64,
            s_bz: i64,
            c: *mut c_void,
            ldc: i64,
            s_cz: i64,
            out_f32: i32,
            m: i32,
            n: i32,
            k: i32,
            batch: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_sparse_attn(
            q: *const c_void,
            ldq: i64,
            kv: *const c_void,
            idx: *const i32,
            ldi: i64,
            topk: i32,
            sink: *const f32,
            scale: f32,
            o: *mut c_void,
            ldo: i64,
            m: i32,
            heads: i32,
            hd: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_comp_pool2(
            kv: *const f32,
            sc: *const f32,
            ld: i64,
            out: *mut c_void,
            ldo: i64,
            groups: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score(
            q: *const c_void,
            ldq: i64,
            k: *const c_void,
            t: i32,
            w: *const c_void,
            ldw: i64,
            wscale: f32,
            start_pos: i32,
            ratio: i32,
            cand: *const u8,
            ldc: i64,
            cb: i32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_topk(
            score: *const c_void,
            lds: i64,
            t: i32,
            k: i32,
            out_w: i32,
            offset: i32,
            out: *mut i32,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_candidates(
            score: *const c_void,
            lds: i64,
            t: i32,
            cb: i32,
            kblocks: i32,
            start_pos: i32,
            ratio: i32,
            keep: *mut u8,
            ldk: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_route(
            raw: *const f32,
            ldr: i64,
            bias: *const f32,
            ne: i32,
            topk: i32,
            norm: i32,
            route_scale: f32,
            sel: *mut i32,
            selw: *mut f32,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_swiglu(
            gate: *const c_void,
            up: *const c_void,
            ld: i64,
            wrow: *const f32,
            limit: f32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            inter: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_scatter_add(
            src: *const c_void,
            lds: i64,
            rows_of: *const i32,
            y: *mut f32,
            ldy: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_add_cast(
            y: *const f32,
            add: *const c_void,
            out: *mut c_void,
            n: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_mixes(
            x: *const c_void,
            fn_: *const f32,
            scale: *const f32,
            base: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            iters: i32,
            norm_eps: f32,
            hc_eps: f32,
            mixes: *mut f32,
            pre: *mut f32,
            post: *mut f32,
            comb: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_pre(
            x: *const c_void,
            mix: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            y: *mut c_void,
            ldy: i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_hc_post(
            f: *const c_void,
            ldf: i64,
            res: *const c_void,
            post: *const f32,
            comb: *const f32,
            rows: i32,
            hc: i32,
            d: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_engram_dequant(
            rows: *const c_void,
            sc: *const c_void,
            n_rows: i64,
            hd: i32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_engram_gate(
            h: *const c_void,
            kv: *const c_void,
            qw: *const c_void,
            kw: *const c_void,
            rows: i32,
            hc: i32,
            d: i32,
            eps: f32,
            inv_sqrt_d: f32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gather_rows(
            src: *const c_void,
            lds: i64,
            rowidx: *const i32,
            dst: *mut c_void,
            ldd: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_gemm_ffma(
            a: *const c_void,
            lda: i64,
            b: *const c_void,
            ldb: i64,
            c: *mut f32,
            ldc: i64,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_cast(
            x: *const c_void,
            y: *mut c_void,
            n: i64,
            to_f32: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `grp`: device array of `ng` groups `{i64 row0, i32 rows, f32 scale2, u64 w, u64 s}`.
        pub fn memra_v41_gemm_f8_grouped(
            a: *const c_void,
            lda: i64,
            a_s: *const c_void,
            ldas: i64,
            grp: *const c_void,
            ng: i32,
            max_rows: i32,
            ldb: i64,
            ldbs: i64,
            wfmt: i32,
            c: *mut c_void,
            ldc: i64,
            out_f32: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_moe_combine(
            slots: *const c_void,
            lds: i64,
            order: *const i32,
            topk: i32,
            shared: *const c_void,
            ldsh: i64,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            d: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score_part(
            q: *const c_void,
            ldq: i64,
            k: *const c_void,
            t: i32,
            w: *const c_void,
            ldw: i64,
            wscale: f32,
            heads: i32,
            part: *mut f32,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn memra_v41_index_score_finish(
            parts: *const c_void,
            n: i32,
            ldp: i64,
            t: i32,
            start_pos: i32,
            ratio: i32,
            cand: *const u8,
            ldc: i64,
            cb: i32,
            out: *mut c_void,
            ldo: i64,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// `rc` from a `memra_v41_*` launcher as a `Result`.
pub fn ck(name: &str, rc: i32) -> Result<(), String> {
    match rc {
        0 => Ok(()),
        r if r >= 10000 => Err(format!("{name}: CUDA error {}", r - 10000)),
        r => Err(format!("{name}: contract refused ({r})")),
    }
}

/// A device address inside a live [`Buf`] (or null). Only `Buf` makes one, so a kernel argument
/// always names device memory; the host never dereferences it. Sizes stay the caller's contract,
/// as with every CUDA launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dp(u64);

impl Dp {
    pub const NULL: Dp = Dp(0);
    pub fn c(self) -> *const c_void {
        self.0 as *const c_void
    }
    pub fn m(self) -> *mut c_void {
        self.0 as *mut c_void
    }
    pub fn i(self) -> *const i32 {
        self.0 as *const i32
    }
    pub fn im(self) -> *mut i32 {
        self.0 as *mut i32
    }
    pub fn f(self) -> *const f32 {
        self.0 as *const f32
    }
    pub fn fm(self) -> *mut f32 {
        self.0 as *mut f32
    }
    pub fn u8c(self) -> *const u8 {
        self.0 as *const u8
    }
    pub fn u8m(self) -> *mut u8 {
        self.0 as *mut u8
    }
    pub fn addr(self) -> u64 {
        self.0
    }
}

/// A device byte buffer. Every V4.1 kernel takes untyped pointers; the element type is the
/// caller's contract (bf16 as raw u16, fp8 and e8m0 as bytes, f32, i32).
pub struct Buf {
    pub s: CudaSlice<u8>,
    pub len: usize,
}

impl Buf {
    pub fn zeros(st: &Arc<CudaStream>, len: usize) -> Result<Self, String> {
        let s = st
            .alloc_zeros::<u8>(len.max(1))
            .map_err(|e| format!("alloc {len}: {e:?}"))?;
        Ok(Self { s, len })
    }
    pub fn from_bytes(st: &Arc<CudaStream>, b: &[u8]) -> Result<Self, String> {
        if b.is_empty() {
            return Self::zeros(st, 0);
        }
        let s = st
            .clone_htod(b)
            .map_err(|e| format!("htod {}: {e:?}", b.len()))?;
        Ok(Self { s, len: b.len() })
    }
    pub fn from_f32(st: &Arc<CudaStream>, v: &[f32]) -> Result<Self, String> {
        Self::from_bytes(st, &f32_bytes(v))
    }
    pub fn from_i32(st: &Arc<CudaStream>, v: &[i32]) -> Result<Self, String> {
        Self::from_bytes(st, &i32_bytes(v))
    }
    pub fn p(&self, st: &Arc<CudaStream>) -> Dp {
        let (p, _g) = self.s.device_ptr(st);
        Dp(p)
    }
    /// Device address `off` bytes in.
    pub fn at(&self, st: &Arc<CudaStream>, off: usize) -> Dp {
        assert!(off <= self.len, "Buf::at past the end");
        Dp(self.p(st).0 + off as u64)
    }
    pub fn host(&self, st: &Arc<CudaStream>) -> Result<Vec<u8>, String> {
        let mut v = st.clone_dtoh(&self.s).map_err(|e| format!("dtoh: {e:?}"))?;
        v.truncate(self.len);
        Ok(v)
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn bytes_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
pub fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}
/// fp32 -> bf16 bits, round to nearest even (finite inputs).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7FC0;
    }
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (r >> 16) as u16
}

/// float8_e4m3fn -> f32 (0x7F/0xFF are NaN).
pub fn e4m3_to_f32(c: u8) -> f32 {
    let s = if c & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((c >> 3) & 0xF) as i32;
    let m = (c & 7) as f32;
    if e == 0xF && (c & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}
/// float8_e8m0fnu -> f32 (255 is NaN).
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        2f32.powi(b as i32 - 127)
    }
}

/// The kernels behind one stream, plus the RoPE tables the attention layers read.
pub struct Ops {
    pub st: Arc<CudaStream>,
}

macro_rules! launch {
    ($name:ident ( $($a:expr),* $(,)? )) => {
        ck(stringify!($name), unsafe { ffi::$name($($a),*) })
    };
}

impl Ops {
    pub fn new(st: Arc<CudaStream>) -> Self {
        Self { st }
    }
    /// The stream handle for a launch, with its context made current first: the launchers use the
    /// runtime API, which launches on the current device, and a rank's kernels must land on its card.
    pub fn sv(&self) -> *mut c_void {
        // a failed bind leaves the previous context current and the launch then fails on the
        // stream/device mismatch, so the error is not lost
        let _ = self.st.context().bind_to_thread();
        self.st.cu_stream() as *mut c_void
    }
    pub fn sync(&self) -> Result<(), String> {
        self.st.synchronize().map_err(|e| format!("sync: {e:?}"))
    }
    pub fn buf(&self, len: usize) -> Result<Buf, String> {
        Buf::zeros(&self.st, len)
    }

    /// act_quant(x, 32, "ue8m0") of `rows` bf16 rows of `k` (gathered through `rowidx` when it
    /// is not null) into e4m3 codes `[rows][k]` and ue8m0 scales `[rows][k/32]`.
    #[allow(clippy::too_many_arguments)]
    pub fn act_quant(
        &self,
        x: Dp,
        ldx: usize,
        rowidx: Dp,
        rows: usize,
        k: usize,
        q: Dp,
        s: Dp,
    ) -> Result<(), String> {
        launch!(memra_v41_act_quant(
            x.c(),
            ldx as i64,
            rowidx.i(),
            q.m(),
            k as i64,
            s.m(),
            (k / 32) as i64,
            std::ptr::null_mut(),
            0,
            rows as i32,
            k as i32,
            self.sv()
        ))
    }
    /// In-place fp8 quantize-dequantize (window KV, attention output before `wo_a`).
    pub fn fp8_qdq(&self, x: Dp, ld: usize, rows: usize, k: usize) -> Result<(), String> {
        launch!(memra_v41_act_quant(
            x.c(),
            ld as i64,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            0,
            x.m(),
            ld as i64,
            rows as i32,
            k as i32,
            self.sv()
        ))
    }
    pub fn fp4_qdq(
        &self,
        x: Dp,
        ld: usize,
        rows: usize,
        n: usize,
        group: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_fp4_qdq(
            x.m(),
            ld as i64,
            rows as i32,
            n as i32,
            group as i32,
            self.sv()
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        &self,
        x: Dp,
        ldx: usize,
        w: &Buf,
        y: Dp,
        ldy: usize,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), String> {
        launch!(memra_v41_rmsnorm(
            x.c(),
            ldx as i64,
            w.p(&self.st).c(),
            y.m(),
            ldy as i64,
            rows as i32,
            d as i32,
            eps,
            self.sv()
        ))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: Dp,
        ld_row: usize,
        rows: usize,
        heads: usize,
        hd: usize,
        t: &RopeTable,
        pos_base: usize,
        pos_step: usize,
        inverse: bool,
    ) -> Result<(), String> {
        if rows > 0 && pos_base + (rows - 1) * pos_step >= t.n {
            return Err(format!("rope: position beyond the table ({})", t.n));
        }
        launch!(memra_v41_rope(
            x.m(),
            ld_row as i64,
            rows as i32,
            heads as i32,
            hd as i32,
            t.cos.p(&self.st).f(),
            t.sin.p(&self.st).f(),
            pos_base as i32,
            pos_step as i32,
            std::ptr::null(),
            inverse as i32,
            self.sv()
        ))
    }
    /// `out[rows][w.n]` = linear(x, w) for an act-quantized input (`aq`, `as_`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q(
        &self,
        aq: Dp,
        as_: Dp,
        rows: usize,
        w: &QW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
    ) -> Result<(), String> {
        let (ldb, ldbs, fmt, s2) = match w.fmt {
            QFmt::Fp8Block32 => (w.k, w.k / 32, 0, 1.0),
            QFmt::Mxfp4 => (w.k / 2, w.k / 32, 1, 1.0),
            QFmt::Nvfp4Minted { scale2 } => (w.k / 2, w.k / 16, 2, scale2),
        };
        launch!(memra_v41_gemm_f8(
            aq.c(),
            w.k as i64,
            as_.c(),
            (w.k / 32) as i64,
            w.w.p(&self.st).c(),
            ldb as i64,
            w.s.p(&self.st).c(),
            ldbs as i64,
            s2,
            fmt,
            out.m(),
            ldc as i64,
            out_f32 as i32,
            rows as i32,
            w.n as i32,
            w.k as i32,
            self.sv()
        ))
    }
    /// linear(x, w) on bf16 rows: per-32 activation quant into `scr`, then the block GEMM.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_q(
        &self,
        x: Dp,
        ldx: usize,
        rows: usize,
        w: &QW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
        scr: &mut Scratch,
    ) -> Result<(), String> {
        let (q, s) = scr.actq(self, rows, w.k)?;
        self.act_quant(x, ldx, Dp::NULL, rows, w.k, q, s)?;
        self.gemm_q(q, s, rows, w, out, ldc, out_f32)
    }
    /// F.linear on bf16 weights with fp32 accumulation (batched over `batch` with element strides).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
        &self,
        a: Dp,
        lda: usize,
        s_az: usize,
        b: Dp,
        ldb: usize,
        s_bz: usize,
        c: Dp,
        ldc: usize,
        s_cz: usize,
        out_f32: bool,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_gemm_bf16(
            a.c(),
            lda as i64,
            s_az as i64,
            b.c(),
            ldb as i64,
            s_bz as i64,
            c.m(),
            ldc as i64,
            s_cz as i64,
            out_f32 as i32,
            m as i32,
            n as i32,
            k as i32,
            batch as i32,
            self.sv()
        ))
    }
    /// F.linear on a bf16 weight. A bf16 output is the reference's bf16 GEMM (tensor cores, fp32
    /// accumulation); an fp32 output is its fp32 GEMM on the same values (CUDA-core FMA chain).
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        x: Dp,
        ldx: usize,
        rows: usize,
        w: &BW,
        out: Dp,
        ldc: usize,
        out_f32: bool,
    ) -> Result<(), String> {
        if out_f32 {
            return launch!(memra_v41_gemm_ffma(
                x.c(),
                ldx as i64,
                w.w.p(&self.st).c(),
                w.k as i64,
                out.fm(),
                ldc as i64,
                rows as i32,
                w.n as i32,
                w.k as i32,
                self.sv()
            ));
        }
        self.gemm_bf16(
            x,
            ldx,
            0,
            w.w.p(&self.st),
            w.k,
            0,
            out,
            ldc,
            0,
            false,
            rows,
            w.n,
            w.k,
            1,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        q: Dp,
        ldq: usize,
        kv: Dp,
        idx: Dp,
        ldi: usize,
        topk: usize,
        sink: Dp,
        scale: f32,
        o: Dp,
        ldo: usize,
        m: usize,
        heads: usize,
    ) -> Result<(), String> {
        launch!(memra_v41_sparse_attn(
            q.c(),
            ldq as i64,
            kv.c(),
            idx.i(),
            ldi as i64,
            topk as i32,
            sink.f(),
            scale,
            o.m(),
            ldo as i64,
            m as i32,
            heads as i32,
            512,
            self.sv()
        ))
    }
}

/// Weight formats of the block-quantized linears.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QFmt {
    /// e4m3 `[n][k]`, one ue8m0 scale per 32x32 block `[ceil(n/32)][k/32]`.
    Fp8Block32,
    /// e2m1 pairs `[n][k/2]` (low nibble first), one ue8m0 scale per row and 32 inputs.
    Mxfp4,
    /// The MXFP4 values re-expressed as NVFP4: e4m3 scales per 16 inputs `[n][k/16]`, each 32-block's
    /// pair equal, times `scale2`. Value-identical to the MXFP4 source by construction.
    Nvfp4Minted { scale2: f32 },
}

/// A block-quantized weight on the device.
pub struct QW {
    pub w: Buf,
    pub s: Buf,
    pub n: usize,
    pub k: usize,
    pub fmt: QFmt,
}

/// A bf16 weight `[n][k]` on the device.
pub struct BW {
    pub w: Buf,
    pub n: usize,
    pub k: usize,
}

/// Reusable activation-quant scratch (codes + scales), grown on demand.
#[derive(Default)]
pub struct Scratch {
    q: Option<Buf>,
    s: Option<Buf>,
}

impl Scratch {
    pub fn actq(&mut self, ops: &Ops, rows: usize, k: usize) -> Result<(Dp, Dp), String> {
        let nq = rows * k;
        let ns = rows * k / 32;
        if self.q.as_ref().is_none_or(|b| b.len < nq) {
            self.q = Some(ops.buf(nq)?);
        }
        if self.s.as_ref().is_none_or(|b| b.len < ns) {
            self.s = Some(ops.buf(ns)?);
        }
        Ok((
            self.q.as_ref().unwrap().p(&ops.st),
            self.s.as_ref().unwrap().p(&ops.st),
        ))
    }
}

/// fp32 RoPE cos/sin tables `[n][32]` for one (theta, YaRN) choice, built the way the reference
/// `precompute_freqs_cis` builds them (fp32 frequencies, YaRN ramp in fp32, fp32 angles).
pub struct RopeTable {
    pub cos: Buf,
    pub sin: Buf,
    pub n: usize,
}

/// Host side of [`RopeTable`]: `(cos, sin)` row-major `[n][dim/2]`.
pub fn rope_tables_host(
    dim: usize,
    n: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        let corrected = |rot: f64| -> f64 {
            dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * (base as f64).ln())
        };
        let low = corrected(beta_fast as f64).floor().max(0.0);
        let high = corrected(beta_slow as f64).ceil().min((dim - 1) as f64);
        let span = (high - low).max(1e-3) as f32;
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
            let smooth = 1.0f32 - ramp;
            *f = *f / factor * (1.0f32 - smooth) + *f * smooth;
        }
    }
    let mut c = vec![0f32; n * half];
    let mut s = vec![0f32; n * half];
    for p in 0..n {
        for i in 0..half {
            let a = p as f32 * freqs[i];
            c[p * half + i] = (a as f64).cos() as f32;
            s[p * half + i] = (a as f64).sin() as f32;
        }
    }
    (c, s)
}

impl RopeTable {
    pub fn new(
        ops: &Ops,
        n: usize,
        original_seq_len: usize,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self, String> {
        let (c, s) = rope_tables_host(64, n, original_seq_len, base, factor, beta_fast, beta_slow);
        Ok(Self {
            cos: Buf::from_f32(&ops.st, &c)?,
            sin: Buf::from_f32(&ops.st, &s)?,
            n,
        })
    }
}

// ------------------------------------------------------------------------------------------------
// checkpoint loading

fn raw<'a>(
    stm: &'a StModel,
    name: &str,
) -> Result<(&'a memra_gguf::safetensors::StInfo, &'a [u8]), String> {
    stm.raw(name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

fn want(
    info: &memra_gguf::safetensors::StInfo,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<(), String> {
    if info.dtype != dtype || info.shape != shape {
        return Err(format!(
            "{name}: {} {:?}, want {dtype} {shape:?}",
            info.dtype, info.shape
        ));
    }
    Ok(())
}

/// FP8 32x32 dense weight `{p}.weight` / `{p}.scale`.
pub fn load_fp8(ops: &Ops, stm: &StModel, p: &str, n: usize, k: usize) -> Result<QW, String> {
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(
        is,
        p,
        "F8_E8M0",
        &[n.div_ceil(32) as u64, k.div_ceil(32) as u64],
    )?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, bw)?,
        s: Buf::from_bytes(&ops.st, bs)?,
        n,
        k,
        fmt: QFmt::Fp8Block32,
    })
}

/// Rows `rows` of the FP8 32x32 dense weight `{p}` (`[n][k]` in the checkpoint): the weight rows
/// and their scale-block rows, so `rows` starts on a 32-row block and ends on one or at `n`.
pub fn load_fp8_rows(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<QW, String> {
    if rows == (0..n) {
        return load_fp8(ops, stm, p, n, k);
    }
    if !rows.start.is_multiple_of(32)
        || (!rows.end.is_multiple_of(32) && rows.end != n)
        || rows.end > n
    {
        return Err(format!("{p}: rows {rows:?} cut a 32-row scale block"));
    }
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    let kb = k.div_ceil(32);
    want(is, p, "F8_E8M0", &[n.div_ceil(32) as u64, kb as u64])?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, &bw[rows.start * k..rows.end * k])?,
        s: Buf::from_bytes(
            &ops.st,
            &bs[rows.start / 32 * kb..rows.end.div_ceil(32) * kb],
        )?,
        n: rows.len(),
        k,
        fmt: QFmt::Fp8Block32,
    })
}

/// FP4 expert weight `{p}.weight` (I8 pairs) / `{p}.scale` (ue8m0 per 32), optionally minted.
pub fn load_fp4(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    mint: bool,
) -> Result<QW, String> {
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "I8", &[n as u64, (k / 2) as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(is, p, "F8_E8M0", &[n as u64, (k / 32) as u64])?;
    if !mint {
        return Ok(QW {
            w: Buf::from_bytes(&ops.st, bw)?,
            s: Buf::from_bytes(&ops.st, bs)?,
            n,
            k,
            fmt: QFmt::Mxfp4,
        });
    }
    let (sc, scale2) = mint_mxfp4_scales(bs).map_err(|e| format!("{p}: {e}"))?;
    Ok(QW {
        w: Buf::from_bytes(&ops.st, bw)?,
        s: Buf::from_bytes(&ops.st, &sc)?,
        n,
        k,
        fmt: QFmt::Nvfp4Minted { scale2 },
    })
}

/// MXFP4 -> NVFP4 scale mint (the `dsv4_mint` cast, per tensor): each ue8m0 scale 2^(b-127)
/// becomes two equal e4m3 per-16 scales 2^(b-127)/scale2 with scale2 = 2^(max_b-127-8), exact when
/// the tensor's scales span at most 17 octaves (e4m3 holds 2^-9 .. 2^8 as powers of two).
pub fn mint_mxfp4_scales(sc: &[u8]) -> Result<(Vec<u8>, f32), String> {
    let mut hi = 0u8;
    for &b in sc {
        if b == 0xFF {
            return Err("NaN ue8m0 scale".into());
        }
        hi = hi.max(b);
    }
    let base = hi as i32 - 127 - 8;
    let mut out = Vec::with_capacity(sc.len() * 2);
    for &b in sc {
        let q = b as i32 - 127 - base; // exponent of the e4m3 scale
        let code = e4m3_pow2_code(q).ok_or_else(|| {
            format!(
                "scale 2^{} outside the e4m3 power-of-two range after the mint",
                b as i32 - 127
            )
        })?;
        out.push(code);
        out.push(code);
    }
    Ok((out, 2f32.powi(base)))
}

/// e4m3 code of 2^q, q in [-9, 8].
fn e4m3_pow2_code(q: i32) -> Option<u8> {
    match q {
        -9..=-7 => Some(1u8 << (q + 9)), // subnormals 2^-9, 2^-8, 2^-7: mantissa 1, 2, 4
        -6..=8 => Some(((q + 7) as u8) << 3),
        _ => None,
    }
}

pub fn load_bf16(ops: &Ops, stm: &StModel, name: &str, n: usize, k: usize) -> Result<BW, String> {
    let (i, b) = raw(stm, name)?;
    want(i, name, "BF16", &[n as u64, k as u64])?;
    Ok(BW {
        w: Buf::from_bytes(&ops.st, b)?,
        n,
        k,
    })
}
/// Rows `rows` of the bf16 weight `name` `[n][k]` (the vocab-parallel head).
pub fn load_bf16_rows(
    ops: &Ops,
    stm: &StModel,
    name: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<BW, String> {
    let (i, b) = raw(stm, name)?;
    want(i, name, "BF16", &[n as u64, k as u64])?;
    if rows.end > n {
        return Err(format!("{name}: rows {rows:?} past {n}"));
    }
    Ok(BW {
        w: Buf::from_bytes(&ops.st, &b[rows.start * k * 2..rows.end * k * 2])?,
        n: rows.len(),
        k,
    })
}
pub fn load_vec_bf16(ops: &Ops, stm: &StModel, name: &str, n: usize) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "BF16" || i.shape.iter().product::<u64>() != n as u64 {
        return Err(format!(
            "{name}: {} {:?}, want BF16 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, b)
}
/// Elements `r` of the F32 vector `name` of `n`.
pub fn load_vec_f32_range(
    ops: &Ops,
    stm: &StModel,
    name: &str,
    n: usize,
    r: std::ops::Range<usize>,
) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "F32" || i.shape.iter().product::<u64>() != n as u64 || r.end > n {
        return Err(format!(
            "{name}: {} {:?}, want F32 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, &b[r.start * 4..r.end * 4])
}
pub fn load_vec_f32(ops: &Ops, stm: &StModel, name: &str, n: usize) -> Result<Buf, String> {
    let (i, b) = raw(stm, name)?;
    if i.dtype != "F32" || i.shape.iter().product::<u64>() != n as u64 {
        return Err(format!(
            "{name}: {} {:?}, want F32 of {n}",
            i.dtype, i.shape
        ));
    }
    Buf::from_bytes(&ops.st, b)
}

/// `wo_a`: FP8 32x32 in the checkpoint, dequantized to bf16 at load like the reference
/// (`[n_groups * o_lora_rank][heads * head_dim / n_groups]`; each product is exact in bf16).
pub fn load_wo_a(ops: &Ops, stm: &StModel, p: &str, n: usize, k: usize) -> Result<BW, String> {
    load_wo_a_rows(ops, stm, p, n, k, 0..n)
}

/// Rows `rows` of `wo_a`, dequantized like [`load_wo_a`].
pub fn load_wo_a_rows(
    ops: &Ops,
    stm: &StModel,
    p: &str,
    n: usize,
    k: usize,
    rows: std::ops::Range<usize>,
) -> Result<BW, String> {
    if rows.end > n {
        return Err(format!("{p}: rows {rows:?} past {n}"));
    }
    let (iw, bw) = raw(stm, &format!("{p}.weight"))?;
    want(iw, p, "F8_E4M3", &[n as u64, k as u64])?;
    let (is, bs) = raw(stm, &format!("{p}.scale"))?;
    want(
        is,
        p,
        "F8_E8M0",
        &[n.div_ceil(32) as u64, k.div_ceil(32) as u64],
    )?;
    let kb = k.div_ceil(32);
    let lut: Vec<f32> = (0..=255u8).map(e4m3_to_f32).collect();
    let mut out = Vec::with_capacity(rows.len() * k * 2);
    for r in rows.clone() {
        for c in 0..k {
            let v = lut[bw[r * k + c] as usize] * e8m0_to_f32(bs[(r / 32) * kb + c / 32]);
            let bits = f32_to_bf16_bits(v);
            debug_assert_eq!(
                f32::from_bits((bits as u32) << 16),
                v,
                "wo_a dequant must be exact in bf16"
            );
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    Ok(BW {
        w: Buf::from_bytes(&ops.st, &out)?,
        n: rows.len(),
        k,
    })
}

// ------------------------------------------------------------------------------------------------
// one layer

/// V4.1 geometry the layer needs (from `DeepSeekV41Config` + the base config).
#[derive(Debug, Clone)]
pub struct Geo {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub window: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub cand_blocks: usize,
    pub cand_block: usize,
    pub cand_source: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub compress_ratios: Vec<usize>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub engram_layers: Vec<usize>,
    pub engram_heads: usize,
    pub engram_head_dim: usize,
    pub engram_ngram: usize,
    pub engram_rows: Vec<usize>,
    pub engram_bucket_base: u64,
    pub engram_compressed_vocab: u64,
    pub engram_pad_id: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub yarn_factor: f32,
    pub yarn_orig: usize,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
}

impl Geo {
    pub fn from_config(mc: &memra_gguf::config::ModelConfig) -> Result<Self, String> {
        let c = mc.dsv41.as_ref().ok_or("not a deepseek_v41 config")?;
        let u = |v: &[u32]| v.iter().map(|&x| x as usize).collect::<Vec<_>>();
        Ok(Self {
            dim: mc.n_embd as usize,
            heads: mc.n_head as usize,
            head_dim: c.head_dim as usize,
            rope_dim: c.qk_rope_head_dim as usize,
            q_lora: c.q_lora_rank as usize,
            o_groups: c.o_groups as usize,
            o_lora: c.o_lora_rank as usize,
            window: c.sliding_window as usize,
            index_heads: c.index_n_heads as usize,
            index_dim: c.index_head_dim as usize,
            index_topk: c.index_topk as usize,
            cand_blocks: c.candidate_topk_blocks as usize,
            cand_block: c.candidate_block_size as usize,
            cand_source: c.candidate_source_layer_id as usize,
            n_experts: c.n_routed_experts as usize,
            topk: c.num_experts_per_tok as usize,
            moe_inter: c.moe_intermediate_size as usize,
            route_scale: c.routed_scaling_factor,
            swiglu_limit: c.swiglu_limit,
            hc: c.hc_mult as usize,
            sinkhorn_iters: c.hc_sinkhorn_iters as usize,
            hc_eps: c.hc_eps,
            norm_eps: mc.rms_eps,
            compress_ratios: u(&c.compress_ratios),
            kv_sources: u(&c.kv_source_layer_ids),
            index_sources: u(&c.index_source_layer_ids),
            engram_layers: u(&c.engram_layer_ids),
            engram_heads: c.engram_n_heads as usize,
            engram_head_dim: c.engram_head_dim as usize,
            engram_ngram: c.engram_max_ngram_size as usize,
            engram_rows: u(&c.engram_num_embeddings),
            engram_bucket_base: c.engram_vocab_size as u64,
            engram_compressed_vocab: c.engram_compressed_vocab_size as u64,
            engram_pad_id: c.engram_pad_token_id as usize,
            vocab: mc.n_vocab as usize,
            rope_theta: mc.rope_freq_base,
            compress_rope_theta: c.compress_rope_theta,
            yarn_factor: c.rope_yarn_factor,
            yarn_orig: c.rope_yarn_orig_ctx as usize,
            yarn_beta_fast: c.rope_yarn_beta_fast,
            yarn_beta_slow: c.rope_yarn_beta_slow,
        })
    }
    pub fn engram_cols(&self) -> usize {
        (self.engram_ngram - 1) * self.engram_heads
    }
}

pub struct CompW {
    pub wkv: BW,
    pub wgate: Option<BW>,
    pub norm: Buf,
}

pub struct IdxW {
    pub wq_b: QW,
    pub wproj: BW,
    pub wk: Option<BW>,
    pub k_norm: Option<Buf>,
}

pub struct EngramW {
    pub wkv: QW,
    pub qw: Buf,
    pub kw: Buf,
    /// table tensor names (`.weight` fp8 rows, `.scale` ue8m0) read from the host mmap
    pub table: String,
    pub rows: usize,
}

/// One rank's share of a layer under N-rank tensor and expert parallelism (`n` = 1: the whole layer
/// on one card). Attention heads, output groups, `wo_b` output rows, routed experts and head rows
/// split into `n` equal contiguous ranges; everything else is replicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Part {
    pub rank: usize,
    pub n: usize,
}

impl Part {
    pub const WHOLE: Part = Part { rank: 0, n: 1 };
    /// This rank's range of `total` items (`total` divisible by `n`, checked by [`Part::check`]).
    pub fn range(&self, total: usize) -> std::ops::Range<usize> {
        let p = total / self.n;
        self.rank * p..(self.rank + 1) * p
    }
    /// The partition laws of the V4.1 layer at `n` ranks: whole heads per rank in sparse-attention
    /// tiles of 16, whole output groups, `wo_b` rows in 32-row scale blocks, whole experts and an
    /// even vocabulary. A degree that breaks one is refused, never forced.
    pub fn check(g: &Geo, n: usize) -> Result<(), String> {
        let fail = |what: String| Err(format!("{n} ranks: {what}"));
        if n == 0 {
            return fail("no ranks".into());
        }
        if !g.heads.is_multiple_of(n) || !(g.heads / n).is_multiple_of(16) {
            return fail(format!(
                "{} attention heads do not split into whole 16-head tiles",
                g.heads
            ));
        }
        if !g.o_groups.is_multiple_of(n) || !g.heads.is_multiple_of(g.o_groups) {
            return fail(format!("{} output groups do not split evenly", g.o_groups));
        }
        if !g.dim.is_multiple_of(n) || !(g.dim / n).is_multiple_of(32) {
            return fail(format!(
                "wo_b rows {} do not split into 32-row scale blocks",
                g.dim
            ));
        }
        if !g.n_experts.is_multiple_of(n) {
            return fail(format!("{} experts do not split evenly", g.n_experts));
        }
        if !g.vocab.is_multiple_of(n) {
            return fail(format!("vocabulary {} does not split evenly", g.vocab));
        }
        Ok(())
    }
}

/// One V4.1 trunk layer's weights on the device (one rank's share under [`Part`]).
pub struct LayerW {
    pub id: usize,
    pub part: Part,
    /// this card's attention heads `[head0, head0 + heads_l)`
    pub head0: usize,
    pub heads_l: usize,
    /// this card's output groups `[group0, group0 + groups_l)`
    pub group0: usize,
    pub groups_l: usize,
    /// this card's `wo_b` output rows start (`wo_b.n` of them)
    pub dim0: usize,
    /// `experts[i]` is routed expert `expert0 + i`
    pub expert0: usize,
    pub ratio: usize,
    pub kv_source: bool,
    pub index_source: bool,
    pub cand_source: bool,
    pub uses_cand: bool,
    pub hc_attn_fn: Buf,
    pub hc_attn_base: Buf,
    pub hc_attn_scale: Buf,
    pub hc_ffn_fn: Buf,
    pub hc_ffn_base: Buf,
    pub hc_ffn_scale: Buf,
    pub attn_norm: Buf,
    pub ffn_norm: Buf,
    pub wq_a: QW,
    pub q_norm: Buf,
    pub wq_b: QW,
    pub wkv: QW,
    pub kv_norm: Buf,
    pub wo_a: BW,
    pub wo_b: QW,
    pub sink: Buf,
    pub comp: Option<CompW>,
    pub idx: Option<IdxW>,
    pub gate_w: BW,
    pub gate_b: Buf,
    pub experts: Vec<[QW; 3]>,
    pub shared: [QW; 3],
    pub engram: Option<EngramW>,
}

impl LayerW {
    pub fn load(ops: &Ops, stm: &StModel, g: &Geo, id: usize, mint: bool) -> Result<Self, String> {
        Self::load_part(ops, stm, g, id, mint, Part::WHOLE)
    }

    /// This rank's share of layer `id` (see [`Part`]).
    pub fn load_part(
        ops: &Ops,
        stm: &StModel,
        g: &Geo,
        id: usize,
        mint: bool,
        part: Part,
    ) -> Result<Self, String> {
        Part::check(g, part.n)?;
        let heads = part.range(g.heads);
        let groups = part.range(g.o_groups);
        let dims = part.range(g.dim);
        let experts = part.range(g.n_experts);
        let p = format!("layers.{id}");
        let ratio = g.compress_ratios[id];
        let kv_source = g.kv_sources.contains(&id);
        let index_source = g.index_sources.contains(&id);
        let hcd = g.hc * g.dim;
        let nmix = (2 + g.hc) * g.hc;
        let comp = if kv_source {
            Some(CompW {
                wkv: load_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.compressor.wkv.weight"),
                    g.head_dim,
                    g.dim,
                )?,
                wgate: if ratio > 1 {
                    Some(load_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.compressor.wgate.weight"),
                        g.head_dim,
                        g.dim,
                    )?)
                } else {
                    None
                },
                norm: load_vec_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.compressor.norm.weight"),
                    g.head_dim,
                )?,
            })
        } else {
            None
        };
        let idx = if index_source {
            Some(IdxW {
                wq_b: load_fp8(
                    ops,
                    stm,
                    &format!("{p}.attn.indexer.wq_b"),
                    g.index_heads * g.index_dim,
                    g.q_lora,
                )?,
                wproj: load_bf16(
                    ops,
                    stm,
                    &format!("{p}.attn.indexer.weights_proj.weight"),
                    g.index_heads,
                    g.dim,
                )?,
                wk: if kv_source {
                    Some(load_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.indexer.wk.weight"),
                        g.index_dim,
                        g.head_dim,
                    )?)
                } else {
                    None
                },
                k_norm: if kv_source {
                    Some(load_vec_bf16(
                        ops,
                        stm,
                        &format!("{p}.attn.indexer.k_norm.weight"),
                        g.index_dim,
                    )?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let expert0 = experts.start;
        let mut experts_w = Vec::with_capacity(experts.len());
        for e in experts {
            let ep = format!("{p}.ffn.experts.{e}");
            experts_w.push([
                load_fp4(ops, stm, &format!("{ep}.w1"), g.moe_inter, g.dim, mint)?,
                load_fp4(ops, stm, &format!("{ep}.w2"), g.dim, g.moe_inter, mint)?,
                load_fp4(ops, stm, &format!("{ep}.w3"), g.moe_inter, g.dim, mint)?,
            ]);
        }
        let sp = format!("{p}.ffn.shared_experts");
        let shared = [
            load_fp8(ops, stm, &format!("{sp}.w1"), g.moe_inter, g.dim)?,
            load_fp8(ops, stm, &format!("{sp}.w2"), g.dim, g.moe_inter)?,
            load_fp8(ops, stm, &format!("{sp}.w3"), g.moe_inter, g.dim)?,
        ];
        let engram = if let Some(ei) = g.engram_layers.iter().position(|&l| l == id) {
            let cols = g.engram_cols();
            let table = format!("{p}.engram.embed");
            let rows = stm
                .info(&format!("{table}.weight"))
                .ok_or("engram table missing")?
                .shape[0] as usize;
            if rows != g.engram_rows[ei] {
                return Err(format!(
                    "{table}: {rows} rows, config says {}",
                    g.engram_rows[ei]
                ));
            }
            Some(EngramW {
                wkv: load_fp8(
                    ops,
                    stm,
                    &format!("{p}.engram.wkv"),
                    g.dim * (g.hc + 1),
                    cols * g.engram_head_dim,
                )?,
                qw: load_vec_bf16(ops, stm, &format!("{p}.engram.q_weight"), g.hc * g.dim)?,
                kw: load_vec_bf16(ops, stm, &format!("{p}.engram.k_weight"), g.hc * g.dim)?,
                table,
                rows,
            })
        } else {
            None
        };
        let hd = g.head_dim;
        let gin = g.heads * hd / g.o_groups;
        Ok(Self {
            id,
            part,
            head0: heads.start,
            heads_l: heads.len(),
            group0: groups.start,
            groups_l: groups.len(),
            dim0: dims.start,
            expert0,
            ratio,
            kv_source,
            index_source,
            cand_source: id == g.cand_source,
            uses_cand: g.cand_source < id && ratio > 0,
            hc_attn_fn: load_vec_f32(ops, stm, &format!("{p}.hc_attn_fn"), nmix * hcd)?,
            hc_attn_base: load_vec_f32(ops, stm, &format!("{p}.hc_attn_base"), nmix)?,
            hc_attn_scale: load_vec_f32(ops, stm, &format!("{p}.hc_attn_scale"), 3)?,
            hc_ffn_fn: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_fn"), nmix * hcd)?,
            hc_ffn_base: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_base"), nmix)?,
            hc_ffn_scale: load_vec_f32(ops, stm, &format!("{p}.hc_ffn_scale"), 3)?,
            attn_norm: load_vec_bf16(ops, stm, &format!("{p}.attn_norm.weight"), g.dim)?,
            ffn_norm: load_vec_bf16(ops, stm, &format!("{p}.ffn_norm.weight"), g.dim)?,
            wq_a: load_fp8(ops, stm, &format!("{p}.attn.wq_a"), g.q_lora, g.dim)?,
            q_norm: load_vec_bf16(ops, stm, &format!("{p}.attn.q_norm.weight"), g.q_lora)?,
            wq_b: load_fp8_rows(
                ops,
                stm,
                &format!("{p}.attn.wq_b"),
                g.heads * hd,
                g.q_lora,
                heads.start * hd..heads.end * hd,
            )?,
            wkv: load_fp8(ops, stm, &format!("{p}.attn.wkv"), g.head_dim, g.dim)?,
            kv_norm: load_vec_bf16(ops, stm, &format!("{p}.attn.kv_norm.weight"), g.head_dim)?,
            wo_a: load_wo_a_rows(
                ops,
                stm,
                &format!("{p}.attn.wo_a"),
                g.o_groups * g.o_lora,
                gin,
                groups.start * g.o_lora..groups.end * g.o_lora,
            )?,
            wo_b: load_fp8_rows(
                ops,
                stm,
                &format!("{p}.attn.wo_b"),
                g.dim,
                g.o_groups * g.o_lora,
                dims,
            )?,
            sink: load_vec_f32_range(ops, stm, &format!("{p}.attn.attn_sink"), g.heads, heads)?,
            comp,
            idx,
            gate_w: load_bf16(
                ops,
                stm,
                &format!("{p}.ffn.gate.weight"),
                g.n_experts,
                g.dim,
            )?,
            gate_b: load_vec_f32(ops, stm, &format!("{p}.ffn.gate.bias"), g.n_experts)?,
            experts: experts_w,
            shared,
            engram,
        })
    }
}

/// The CSA2 state one layer hands down the stack (vendor `SharedAttentionRuntime`): the group's
/// compressed KV and index keys from its kv source, the selection from the last index source,
/// the candidate blocks from the candidate source.
#[derive(Default)]
pub struct Shared {
    pub compress_kv: Option<Buf>, // bf16 [n_c][512], RoPE'd and fp4 g16 quant-dequantized
    pub n_c: usize,
    pub index_k: Option<Buf>, // bf16 [n_c][128], RoPE'd and fp4 g32 quant-dequantized
    pub topk: Option<Buf>,    // i32 [s][topk_w], positions already offset by the window length
    pub topk_w: usize,
    pub cand: Option<Buf>, // u8 [s][n_blocks]
    pub cand_ld: usize,
    /// row capacity of `compress_kv` / `index_k` (decode appends rows)
    pub compress_cap: usize,
    pub index_k_cap: usize,
}

/// One layer's own decode state: the sliding-window ring (bf16 `[window][head_dim]`, position p at
/// slot p % window) and, on a ratio-2 kv source, the open pair's fp32 kv/gate rows.
#[derive(Default)]
pub struct LayerState {
    pub ring: Option<Buf>,
    pub c_kv: Option<Buf>,
    pub c_sc: Option<Buf>,
}

/// Grow `buf` (rows of `row_bytes`) to hold `rows`, keeping its first `cap` rows.
fn ensure_rows(
    ops: &Ops,
    buf: &mut Option<Buf>,
    cap: &mut usize,
    rows: usize,
    row_bytes: usize,
    hint: usize,
) -> Result<(), String> {
    if buf.is_some() && rows <= *cap {
        return Ok(());
    }
    let new_cap = rows.max(hint).max(*cap * 2);
    let nb = ops.buf(new_cap * row_bytes)?;
    if let Some(old) = buf.as_ref() {
        let keep = (*cap).min(old.len / row_bytes) * row_bytes;
        copy_d2d(ops, old, 0, &nb, 0, keep)?;
    }
    *buf = Some(nb);
    *cap = new_cap;
    Ok(())
}

/// What one prefill layer produced besides its output, for gates.
#[derive(Default)]
pub struct Trace {
    pub attn_mix: Option<Buf>, // f32 [s][24]
    pub attn_in: Option<Buf>,  // bf16 [s][dim], after attn_norm
    pub qr: Option<Buf>,
    pub latent: Option<Buf>, // pre-RoPE compressed latent
    pub idx_all: Option<Buf>,
    pub kv_all: Option<Buf>,
    pub q: Option<Buf>,
    pub attn_o: Option<Buf>,
    pub attn_out: Option<Buf>,
    pub mid: Option<Buf>,
    pub moe_in: Option<Buf>,
    pub route_sel: Vec<i32>,
    pub route_w: Vec<f32>,
    pub moe_out: Option<Buf>,
    pub engram_out: Option<Buf>,
}

/// What the keys side of one attention call hands its query side: `qr`, the KV rows the heads
/// read (`[window part | compressed part]`) and the selection width; the selection itself stays in
/// the [`Shared`] state the keys side wrote.
pub struct AttnKeys {
    pub qr: Buf,
    pub kv_all: Buf,
    pub ww: usize,
    pub tw: usize,
    pub tot: usize,
    pub win: usize,
    pub comp: bool,
}

/// Runtime pieces shared by every layer on the card.
pub struct Ctx {
    pub ops: Ops,
    pub g: Geo,
    pub rope_base: RopeTable,
    pub rope_comp: RopeTable,
    pub scr: Scratch,
    pub stage: Option<engram::Stage>,
    pub trace: bool,
    /// positions the RoPE tables and cache capacities cover
    pub max_pos: usize,
    /// prefill tokens per query-side slice (attention, indexer selection, MoE, Engram); every
    /// sliced step acts per row, so the slice size never changes a result
    pub chunk: usize,
    /// the query rows this card's indexer selects (None: all of them). An N-rank prefill gives
    /// each rank a contiguous range and gathers the selection rows (`dsv41_gpu::tp`); the
    /// selection is per row, so the split never changes a result
    pub index_rows: Option<std::ops::Range<usize>>,
}

impl Ctx {
    pub fn new(st: Arc<CudaStream>, g: Geo, max_pos: usize) -> Result<Self, String> {
        let ops = Ops::new(st);
        let rope_base = RopeTable::new(
            &ops,
            max_pos,
            0,
            g.rope_theta,
            g.yarn_factor,
            g.yarn_beta_fast,
            g.yarn_beta_slow,
        )?;
        let rope_comp = RopeTable::new(
            &ops,
            max_pos,
            g.yarn_orig,
            g.compress_rope_theta,
            g.yarn_factor,
            g.yarn_beta_fast,
            g.yarn_beta_slow,
        )?;
        Ok(Self {
            ops,
            g,
            rope_base,
            rope_comp,
            scr: Scratch::default(),
            stage: None,
            trace: false,
            max_pos,
            chunk: 8192,
            index_rows: None,
        })
    }

    /// One prefill row of the window selection (vendor `get_window_topk_idxs`, start_pos 0): the
    /// query at position p sees positions max(0, p-win+1) ..= p in order, -1 where that runs past p.
    pub fn window_row(p: usize, win: usize, row: &mut [i32]) {
        let start = (p + 1).saturating_sub(win);
        for (j, v) in row.iter_mut().enumerate() {
            let k = start + j;
            *v = if k > p { -1 } else { k as i32 };
        }
    }

    /// The window selection of a whole prefill of `s` tokens: `([s][min(s, win)], width)`.
    pub fn window_idx(s: usize, win: usize) -> (Vec<i32>, usize) {
        let w = s.min(win);
        let mut v = vec![-1i32; s * w];
        for (i, row) in v.chunks_mut(w.max(1)).enumerate().take(s) {
            Self::window_row(i, win, row);
        }
        (v, w)
    }

    /// Engram (vendor `Engram.forward`) on the stream in place, `chunk` tokens at a time: gather
    /// the hashed table rows from the host mmap into pinned staging, H2D, dequant, `wkv`, the
    /// normalized gate. `hashes` is `[s][cols]`.
    pub fn engram(
        &mut self,
        stm: &StModel,
        lw: &LayerW,
        h: &Buf,
        hashes: &[i64],
        s: usize,
        tr: &mut Trace,
    ) -> Result<(), String> {
        let ew = lw.engram.as_ref().ok_or("layer has no engram")?;
        let g = self.g.clone();
        let cols = g.engram_cols();
        let hd = g.engram_head_dim;
        if hashes.len() != s * cols {
            return Err(format!(
                "engram: {} hash ids for {s} x {cols}",
                hashes.len()
            ));
        }
        let c = self.chunk.min(s).max(1);
        if self.stage.as_ref().is_none_or(|x| x.cap < c * cols) {
            self.stage = Some(engram::Stage::new(&self.ops, (c * cols).max(4096), hd)?);
        }
        let st = self.ops.st.clone();
        let nkv = g.dim * (g.hc + 1);
        let emb = self.ops.buf(c * cols * hd * 2)?;
        let kv = self.ops.buf(c * nkv * 2)?;
        let row_bytes = g.hc * g.dim * 2;
        for r0 in (0..s).step_by(c) {
            let n = c.min(s - r0);
            let stage = self.stage.as_mut().unwrap();
            stage.load(
                &self.ops,
                stm,
                &ew.table,
                ew.rows,
                &hashes[r0 * cols..(r0 + n) * cols],
            )?;
            ck("engram_dequant", unsafe {
                ffi::memra_v41_engram_dequant(
                    stage.d_rows.p(&st).m(),
                    stage.d_sc.p(&st).m(),
                    (n * cols) as i64,
                    hd as i32,
                    emb.p(&st).m(),
                    self.ops.sv(),
                )
            })?;
            self.ops.linear_q(
                emb.p(&st),
                cols * hd,
                n,
                &ew.wkv,
                kv.p(&st),
                nkv,
                false,
                &mut self.scr,
            )?;
            let hp = h.at(&st, r0 * row_bytes);
            ck("engram_gate", unsafe {
                ffi::memra_v41_engram_gate(
                    hp.m(),
                    kv.p(&st).m(),
                    ew.qw.p(&st).m(),
                    ew.kw.p(&st).m(),
                    n as i32,
                    g.hc as i32,
                    g.dim as i32,
                    g.norm_eps,
                    (g.dim as f32).powf(-0.5),
                    hp.m(),
                    self.ops.sv(),
                )
            })?;
            // the staging buffers are refilled next chunk: the copies and kernels must be done
            self.ops.sync()?;
        }
        if self.trace {
            tr.engram_out = Some(clone_buf(&self.ops, h)?);
        }
        Ok(())
    }

    /// Index keys of the latent rows that just completed on a kv source (vendor `Indexer.forward`,
    /// its key half): `k_norm(wk(latent))`, RoPE at each group's first position, fp4 g32, written
    /// at rows `[row0, row0 + nrows)` of the shared index keys.
    fn index_keys(
        &mut self,
        lw: &LayerW,
        lat: &Buf,
        row0: usize,
        nrows: usize,
        sh: &mut Shared,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let iw = lw.idx.as_ref().unwrap();
        let (Some(wk), Some(kn)) = (iw.wk.as_ref(), iw.k_norm.as_ref()) else {
            return Ok(());
        };
        let ratio = lw.ratio;
        let rb = g.index_dim * 2;
        ensure_rows(
            &self.ops,
            &mut sh.index_k,
            &mut sh.index_k_cap,
            row0 + nrows,
            rb,
            self.max_pos / ratio,
        )?;
        let kp = sh.index_k.as_ref().unwrap().at(&st, row0 * rb);
        self.ops
            .linear_bf16(lat.p(&st), g.head_dim, nrows, wk, kp, g.index_dim, false)?;
        self.ops.rmsnorm(
            kp,
            g.index_dim,
            kn,
            kp,
            g.index_dim,
            nrows,
            g.index_dim,
            g.norm_eps,
        )?;
        self.ops.rope(
            kp,
            g.index_dim,
            nrows,
            1,
            g.index_dim,
            &self.rope_comp,
            row0 * ratio,
            ratio,
            false,
        )?;
        self.ops.fp4_qdq(kp, g.index_dim, nrows, g.index_dim, 32)?;
        Ok(())
    }

    /// The indexer's selection for query rows `[r0, r1)` of a call of `s` tokens from position
    /// `sp` (vendor `Indexer.forward`, its query half, row-sliced like the oracle's long-context
    /// indexer: every step acts per query row). The fp4 query, the bf16 score over the `n_c`
    /// visible keys, the candidate pool on the candidate source, top-k shifted by the window
    /// length `offset`; written into rows `r0..r1` of `sh.topk` (and `sh.cand`).
    #[allow(clippy::too_many_arguments)]
    fn index_select(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        qr: &Buf,
        r0: usize,
        r1: usize,
        sp: usize,
        s: usize,
        offset: usize,
        sh: &mut Shared,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let iw = lw.idx.as_ref().unwrap();
        let ratio = lw.ratio;
        let n_c = (sp + s) / ratio;
        let rows = r1 - r0;
        let ik = sh
            .index_k
            .as_ref()
            .ok_or("indexer: no index keys from a kv source")?;
        let qd = g.index_heads * g.index_dim;
        let q = self.ops.buf(rows * qd * 2)?;
        self.ops.linear_q(
            qr.at(&st, r0 * g.q_lora * 2),
            g.q_lora,
            rows,
            &iw.wq_b,
            q.p(&st),
            qd,
            false,
            &mut self.scr,
        )?;
        self.ops.rope(
            q.p(&st),
            qd,
            rows,
            g.index_heads,
            g.index_dim,
            &self.rope_comp,
            sp + r0,
            1,
            false,
        )?;
        self.ops.fp4_qdq(q.p(&st), qd, rows, qd, 32)?;
        let w = self.ops.buf(rows * g.index_heads * 2)?;
        self.ops.linear_bf16(
            x.at(&st, r0 * g.dim * 2),
            g.dim,
            rows,
            &iw.wproj,
            w.p(&st),
            g.index_heads,
            false,
        )?;
        // the kernel applies weights * (index_head_dim^-0.5 * index_heads^-0.5) as a bf16 product
        let wscale = ((g.index_dim as f64).powf(-0.5) * (g.index_heads as f64).powf(-0.5)) as f32;
        let score = self.ops.buf(rows * n_c * 2)?;
        let (cand, ldc) = if lw.uses_cand {
            let c = sh.cand.as_ref().ok_or("indexer: candidate pool missing")?;
            (c.at(&st, r0 * sh.cand_ld).u8c(), sh.cand_ld)
        } else {
            (std::ptr::null(), 0)
        };
        ck("index_score", unsafe {
            ffi::memra_v41_index_score(
                q.p(&st).m(),
                qd as i64,
                ik.p(&st).m(),
                n_c as i32,
                w.p(&st).m(),
                g.index_heads as i64,
                wscale,
                (sp + r0) as i32,
                ratio as i32,
                cand,
                ldc as i64,
                g.cand_block as i32,
                score.p(&st).m(),
                n_c as i64,
                rows as i32,
                self.ops.sv(),
            )
        })?;
        if lw.cand_source {
            let keep = sh
                .cand
                .as_ref()
                .ok_or("indexer: candidate pool not allocated")?;
            ck("candidates", unsafe {
                ffi::memra_v41_candidates(
                    score.p(&st).m(),
                    n_c as i64,
                    n_c as i32,
                    g.cand_block as i32,
                    g.cand_blocks as i32,
                    (sp + r0) as i32,
                    ratio as i32,
                    keep.at(&st, r0 * sh.cand_ld).u8m(),
                    sh.cand_ld as i64,
                    rows as i32,
                    self.ops.sv(),
                )
            })?;
        }
        let tw = sh.topk_w;
        let out = sh.topk.as_ref().ok_or("indexer: selection not allocated")?;
        ck("topk", unsafe {
            ffi::memra_v41_topk(
                score.p(&st).m(),
                n_c as i64,
                n_c as i32,
                tw as i32,
                tw as i32,
                offset as i32,
                out.at(&st, r0 * tw * 4).im(),
                tw as i64,
                rows as i32,
                self.ops.sv(),
            )
        })?;
        Ok(())
    }

    /// CSA2 compressor (vendor `Compressor.forward`) for `s` tokens from `sp`: the pre-RoPE latent
    /// rows of every group that completes, with their first row index, or None while a ratio-2
    /// pair is still half full. Ratio 1 is a bf16 projection; ratio 2 pools pairs by a per-channel
    /// softmax gate in fp32, with the open pair's fp32 rows held in `ls`.
    fn compressor(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        ls: &mut LayerState,
    ) -> Result<Option<(Buf, usize, usize)>, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let cw = lw.comp.as_ref().unwrap();
        let ratio = lw.ratio;
        let hd = g.head_dim;
        let norm = |c: &mut Self, lat: &Buf, n: usize| {
            c.ops
                .rmsnorm(lat.p(&st), hd, &cw.norm, lat.p(&st), hd, n, hd, g.norm_eps)
        };
        if ratio == 1 {
            let lat = self.ops.buf(s * hd * 2)?;
            self.ops
                .linear_bf16(x.p(&st), g.dim, s, &cw.wkv, lat.p(&st), hd, false)?;
            norm(self, &lat, s)?;
            return Ok(Some((lat, sp, s)));
        }
        if ratio != 2 {
            return Err(format!("compressor ratio {ratio} has no pooling kernel"));
        }
        let kvf = self.ops.buf(s * hd * 4)?;
        let scf = self.ops.buf(s * hd * 4)?;
        self.ops
            .linear_bf16(x.p(&st), g.dim, s, &cw.wkv, kvf.p(&st), hd, true)?;
        self.ops.linear_bf16(
            x.p(&st),
            g.dim,
            s,
            cw.wgate.as_ref().unwrap(),
            scf.p(&st),
            hd,
            true,
        )?;
        if ls.c_kv.is_none() {
            ls.c_kv = Some(self.ops.buf(2 * hd * 4)?);
            ls.c_sc = Some(self.ops.buf(2 * hd * 4)?);
        }
        let (ckv, csc) = (ls.c_kv.as_ref().unwrap(), ls.c_sc.as_ref().unwrap());
        let pool = |c: &mut Self, kv: &Buf, sc: &Buf, n: usize, lat: &Buf| -> Result<(), String> {
            ck("comp_pool2", unsafe {
                ffi::memra_v41_comp_pool2(
                    kv.p(&st).f(),
                    sc.p(&st).f(),
                    hd as i64,
                    lat.p(&st).m(),
                    hd as i64,
                    n as i32,
                    hd as i32,
                    c.ops.sv(),
                )
            })
        };
        if sp == 0 {
            let n = s / 2;
            if s % 2 == 1 {
                // the trailing token waits in slot 0 for its pair
                copy_d2d(&self.ops, &kvf, (s - 1) * hd * 4, ckv, 0, hd * 4)?;
                copy_d2d(&self.ops, &scf, (s - 1) * hd * 4, csc, 0, hd * 4)?;
            }
            if n == 0 {
                return Ok(None);
            }
            let lat = self.ops.buf(n * hd * 2)?;
            pool(self, &kvf, &scf, n, &lat)?;
            norm(self, &lat, n)?;
            return Ok(Some((lat, 0, n)));
        }
        if s != 1 {
            return Err("compressor: chunked decode is not implemented".into());
        }
        let slot = sp % 2;
        copy_d2d(&self.ops, &kvf, 0, ckv, slot * hd * 4, hd * 4)?;
        copy_d2d(&self.ops, &scf, 0, csc, slot * hd * 4, hd * 4)?;
        if !(sp + 1).is_multiple_of(2) {
            return Ok(None);
        }
        let lat = self.ops.buf(hd * 2)?;
        pool(self, ckv, csc, 1, &lat)?;
        norm(self, &lat, 1)?;
        Ok(Some((lat, sp / 2, 1)))
    }

    /// Attention (vendor `Attention.forward`) on the normed input x `[s][dim]` from position `sp`:
    /// a prefill chunk at 0, or one decode token. Returns `[s][dim]` bf16. The keys side (window
    /// KV, compressor, index keys) runs over all `s` tokens; the query side (q, the indexer's
    /// selection, attention, the output projections) runs `chunk` query rows at a time, every
    /// step of it being per query row.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<Buf, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let k = self.attn_keys(lw, x, sp, s, sh, ls, tr)?;
        let qdim = lw.heads_l * g.head_dim;
        let c = self.chunk.min(s).max(1);
        let out = self.ops.buf(s * g.dim * 2)?;
        let q = self.ops.buf(c * qdim * 2)?;
        let o = self.ops.buf(c * qdim * 2)?;
        let oa_ld = g.o_groups * g.o_lora;
        let oa = self.ops.buf(c * oa_ld * 2)?;
        let single = c == s;
        for r0 in (0..s).step_by(c) {
            let rows = c.min(s - r0);
            self.attn_heads(
                lw,
                &k,
                sh,
                sp,
                r0,
                rows,
                &q,
                &o,
                oa.p(&st),
                oa_ld,
                single,
                tr,
            )?;
            self.attn_wo_b(
                lw,
                oa.p(&st),
                oa_ld,
                rows,
                out.at(&st, r0 * g.dim * 2),
                g.dim,
            )?;
            // the host index rows are rebuilt next chunk
            self.ops.sync()?;
        }
        Ok(out)
    }

    /// The keys side of [`Self::attention`], identical on every rank: `qr`, this call's window KV
    /// into the ring, the compressor and index keys on a source, the indexer's selection, the
    /// compressed KV append, and the KV rows `[window part | compressed part]` the heads read.
    #[allow(clippy::too_many_arguments)]
    fn attn_keys(
        &mut self,
        lw: &LayerW,
        x: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<AttnKeys, String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let hd = g.head_dim;
        let ratio = lw.ratio;
        let win = g.window;
        let comp = ratio > 0;
        if sp > 0 && s != 1 {
            return Err("attention: chunked decode is not implemented".into());
        }
        let qr = self.ops.buf(s * g.q_lora * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            s,
            &lw.wq_a,
            qr.p(&st),
            g.q_lora,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            qr.p(&st),
            g.q_lora,
            &lw.q_norm,
            qr.p(&st),
            g.q_lora,
            s,
            g.q_lora,
            g.norm_eps,
        )?;
        // this call's window KV: kv_norm(wkv(x)), RoPE, fp8 per-32 quant-dequant
        let kvw = self.ops.buf(s * hd * 2)?;
        self.ops.linear_q(
            x.p(&st),
            g.dim,
            s,
            &lw.wkv,
            kvw.p(&st),
            hd,
            false,
            &mut self.scr,
        )?;
        self.ops.rmsnorm(
            kvw.p(&st),
            hd,
            &lw.kv_norm,
            kvw.p(&st),
            hd,
            s,
            hd,
            g.norm_eps,
        )?;
        self.ops.rope(
            kvw.p(&st),
            hd,
            s,
            1,
            hd,
            if comp {
                &self.rope_comp
            } else {
                &self.rope_base
            },
            sp,
            1,
            false,
        )?;
        self.ops.fp8_qdq(kvw.p(&st), hd, s, hd)?;
        if ls.ring.is_none() {
            ls.ring = Some(self.ops.buf(win * hd * 2)?);
        }
        let ring = ls.ring.as_ref().unwrap();
        // the ring holds position p at slot p % win; a prefill seeds it with its last win rows
        let p0 = (sp + s).saturating_sub(win).max(sp);
        for p in p0..sp + s {
            copy_d2d(
                &self.ops,
                &kvw,
                (p - sp) * hd * 2,
                ring,
                (p % win) * hd * 2,
                hd * 2,
            )?;
        }
        let (win_rows, ww) = if sp == 0 { (s, s.min(win)) } else { (win, win) };
        let n_c = (sp + s).checked_div(ratio).unwrap_or(0);
        let kv_all = self.ops.buf((win_rows + n_c).max(1) * hd * 2)?;
        if sp == 0 {
            copy_d2d(&self.ops, &kvw, 0, &kv_all, 0, s * hd * 2)?;
        } else {
            copy_d2d(&self.ops, ring, 0, &kv_all, 0, win * hd * 2)?;
        }
        drop(kvw);
        let c = self.chunk.min(s).max(1);
        if ratio > 0 {
            let latent = if lw.kv_source {
                self.compressor(lw, x, sp, s, ls)?
            } else {
                None
            };
            if self.trace {
                tr.latent = match &latent {
                    Some((l, _, _)) => Some(clone_buf(&self.ops, l)?),
                    None => None,
                };
            }
            if lw.index_source {
                if n_c == 0 {
                    sh.topk = None;
                    sh.topk_w = 0;
                } else {
                    if let Some((lat, row0, nrows)) = latent.as_ref() {
                        self.index_keys(lw, lat, *row0, *nrows, sh)?;
                    }
                    let tw = g.index_topk.min(n_c);
                    sh.topk = Some(self.ops.buf(s * tw * 4)?);
                    sh.topk_w = tw;
                    if lw.cand_source {
                        let nb = n_c.div_ceil(g.cand_block);
                        sh.cand = Some(self.ops.buf(s * nb)?);
                        sh.cand_ld = nb;
                    }
                    let span = self.index_rows.clone().unwrap_or(0..s);
                    for r0 in (span.start..span.end.min(s)).step_by(c) {
                        let r1 = (r0 + c).min(span.end.min(s));
                        self.index_select(lw, x, &qr, r0, r1, sp, s, win_rows, sh)?;
                    }
                }
            }
            if let Some((lat, row0, nrows)) = latent {
                self.ops.rope(
                    lat.p(&st),
                    hd,
                    nrows,
                    1,
                    hd,
                    &self.rope_comp,
                    row0 * ratio,
                    ratio,
                    false,
                )?;
                self.ops.fp4_qdq(lat.p(&st), hd, nrows, hd, 16)?;
                ensure_rows(
                    &self.ops,
                    &mut sh.compress_kv,
                    &mut sh.compress_cap,
                    row0 + nrows,
                    hd * 2,
                    self.max_pos / ratio,
                )?;
                copy_d2d(
                    &self.ops,
                    &lat,
                    0,
                    sh.compress_kv.as_ref().unwrap(),
                    row0 * hd * 2,
                    nrows * hd * 2,
                )?;
                sh.n_c = row0 + nrows;
            }
            if n_c > 0 {
                let ckv = sh
                    .compress_kv
                    .as_ref()
                    .ok_or("attention: no compressed KV from the group source")?;
                if sh.n_c < n_c {
                    return Err(format!(
                        "attention: compressed KV holds {} rows, need {n_c}",
                        sh.n_c
                    ));
                }
                copy_d2d(&self.ops, ckv, 0, &kv_all, win_rows * hd * 2, n_c * hd * 2)?;
            }
        }
        let tw = if ratio > 0 && n_c > 0 { sh.topk_w } else { 0 };
        let tot = ww + tw;
        if tw > 0 && sh.topk.is_none() {
            return Err("attention: no selection from the index source".into());
        }
        Ok(AttnKeys {
            qr,
            kv_all,
            ww,
            tw,
            tot,
            win,
            comp,
        })
    }

    /// The query side of [`Self::attention`] for rows `[r0, r0 + rows)` over this card's heads
    /// (`lw.heads_l` from `lw.head0`; all of them on one card): q, RoPE, the selection rows, sparse
    /// attention, inverse RoPE, the fp8 quantize-dequantize of o and the grouped `wo_a` of this
    /// card's output groups, written at element column `lw.group0 * o_lora` of `oa` (row pitch
    /// `oa_ld` elements). Every step is per row and per head.
    #[allow(clippy::too_many_arguments)]
    fn attn_heads(
        &mut self,
        lw: &LayerW,
        k: &AttnKeys,
        sh: &Shared,
        sp: usize,
        r0: usize,
        rows: usize,
        q: &Buf,
        o: &Buf,
        oa: Dp,
        oa_ld: usize,
        single: bool,
        tr: &mut Trace,
    ) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let hd = g.head_dim;
        let heads = lw.heads_l;
        let qdim = heads * hd;
        let gk = g.heads * hd / g.o_groups; // 4096 inputs per group
        let scale = (hd as f64).powf(-0.5) as f32;
        let (ww, tw, tot, win) = (k.ww, k.tw, k.tot, k.win);
        self.ops.linear_q(
            k.qr.at(&st, r0 * g.q_lora * 2),
            g.q_lora,
            rows,
            &lw.wq_b,
            q.p(&st),
            qdim,
            false,
            &mut self.scr,
        )?;
        self.ops.rope(
            q.p(&st),
            qdim,
            rows,
            heads,
            hd,
            if k.comp {
                &self.rope_comp
            } else {
                &self.rope_base
            },
            sp + r0,
            1,
            false,
        )?;
        // index rows: the window part from the host, the compressed part from the selection
        let mut idx_host = vec![-1i32; rows * tot];
        for i in 0..rows {
            let row = &mut idx_host[i * tot..i * tot + ww];
            if sp == 0 {
                Self::window_row(r0 + i, win, row);
            } else {
                // decode: the whole ring, oldest slot first, -1 for slots not reached yet
                let oldest = sp % win + 1;
                for (v, slot) in row.iter_mut().zip((oldest..win).chain(0..oldest)) {
                    *v = if slot > sp { -1 } else { slot as i32 };
                }
            }
        }
        let idx = Buf::from_i32(&st, &idx_host)?;
        if tw > 0 {
            let sel = sh
                .topk
                .as_ref()
                .ok_or("attention: no selection from the index source")?;
            copy_2d(
                &self.ops,
                sel,
                r0,
                tw * 4,
                &idx,
                ww * 4,
                tot * 4,
                tw * 4,
                rows,
            )?;
        }
        self.ops.sparse_attn(
            q.p(&st),
            qdim,
            k.kv_all.p(&st),
            idx.p(&st),
            tot,
            tot,
            lw.sink.p(&st),
            scale,
            o.p(&st),
            qdim,
            rows,
            heads,
        )?;
        if self.trace && single {
            tr.qr = Some(clone_buf(&self.ops, &k.qr)?);
            tr.q = Some(clone_buf(&self.ops, q)?);
            tr.kv_all = Some(clone_buf(&self.ops, &k.kv_all)?);
            tr.idx_all = Some(clone_buf(&self.ops, &idx)?);
            tr.attn_o = Some(clone_buf(&self.ops, o)?);
        }
        self.ops.rope(
            o.p(&st),
            qdim,
            rows,
            heads,
            hd,
            if k.comp {
                &self.rope_comp
            } else {
                &self.rope_base
            },
            sp + r0,
            1,
            true,
        )?;
        self.ops.fp8_qdq(o.p(&st), qdim, rows, qdim)?;
        self.ops.gemm_bf16(
            o.p(&st),
            qdim,
            gk,
            lw.wo_a.w.p(&st),
            gk,
            g.o_lora * gk,
            Dp(oa.addr() + (lw.group0 * g.o_lora * 2) as u64),
            oa_ld,
            g.o_lora,
            false,
            rows,
            g.o_lora,
            gk,
            lw.groups_l,
        )?;
        Ok(())
    }

    /// `wo_b` over the gathered `oa` rows (pitch `oa_ld` elements): this card's output rows
    /// `[lw.dim0, lw.dim0 + lw.wo_b.n)` written at that column of `out` (row pitch `ldc` elements).
    fn attn_wo_b(
        &mut self,
        lw: &LayerW,
        oa: Dp,
        oa_ld: usize,
        rows: usize,
        out: Dp,
        ldc: usize,
    ) -> Result<(), String> {
        self.ops.linear_q(
            oa,
            oa_ld,
            rows,
            &lw.wo_b,
            Dp(out.addr() + (lw.dim0 * 2) as u64),
            ldc,
            false,
            &mut self.scr,
        )
    }

    /// MoE (vendor `MoE.forward`): the sqrtsoftplus gate with its selection bias, routed MXFP4
    /// (or minted NVFP4) experts in ascending id order, the shared FP8 expert, bf16 out.
    fn moe(&mut self, lw: &LayerW, x: Dp, s: usize, out: Dp, tr: &mut Trace) -> Result<(), String> {
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let ne = g.n_experts;
        let raw = self.ops.buf(s * ne * 4)?;
        self.ops
            .linear_bf16(x, g.dim, s, &lw.gate_w, raw.p(&st), ne, true)?;
        let sel = self.ops.buf(s * g.topk * 4)?;
        let selw = self.ops.buf(s * g.topk * 4)?;
        ck("route", unsafe {
            ffi::memra_v41_route(
                raw.p(&st).f(),
                ne as i64,
                lw.gate_b.p(&st).f(),
                ne as i32,
                g.topk as i32,
                1,
                g.route_scale,
                sel.p(&st).im(),
                selw.p(&st).fm(),
                s as i32,
                self.ops.sv(),
            )
        })?;
        let sel_h = bytes_i32(&sel.host(&st)?);
        let selw_h = bytes_f32(&selw.host(&st)?);
        let mut rows_of: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
        for t in 0..s {
            for j in 0..g.topk {
                let e = sel_h[t * g.topk + j];
                if e < 0 || e as usize >= ne {
                    return Err(format!("route: expert {e}"));
                }
                rows_of[e as usize].push((t as i32, selw_h[t * g.topk + j]));
            }
        }
        if self.trace {
            tr.route_sel.extend_from_slice(&sel_h);
            tr.route_w.extend_from_slice(&selw_h);
        }
        let y = self.ops.buf(s * g.dim * 4)?;
        let inter = g.moe_inter;
        let maxn = rows_of.iter().map(|v| v.len()).max().unwrap_or(0).max(s);
        let gb = self.ops.buf(maxn * inter * 2)?;
        let ub = self.ops.buf(maxn * inter * 2)?;
        let hb = self.ops.buf(maxn * inter * 2)?;
        let db = self.ops.buf(maxn * g.dim * 2)?;
        let aq = self.ops.buf(maxn * g.dim)?;
        let as_ = self.ops.buf(maxn * g.dim / 32)?;
        for (e, rows) in rows_of.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let n = rows.len();
            let ridx: Vec<i32> = rows.iter().map(|r| r.0).collect();
            let rw: Vec<f32> = rows.iter().map(|r| r.1).collect();
            let dri = Buf::from_i32(&st, &ridx)?;
            let drw = Buf::from_f32(&st, &rw)?;
            let [w1, w2, w3] = &lw.experts[e];
            self.ops
                .act_quant(x, g.dim, dri.p(&st), n, g.dim, aq.p(&st), as_.p(&st))?;
            self.ops
                .gemm_q(aq.p(&st), as_.p(&st), n, w1, gb.p(&st), inter, false)?;
            self.ops
                .gemm_q(aq.p(&st), as_.p(&st), n, w3, ub.p(&st), inter, false)?;
            ck("swiglu", unsafe {
                ffi::memra_v41_swiglu(
                    gb.p(&st).m(),
                    ub.p(&st).m(),
                    inter as i64,
                    drw.p(&st).f(),
                    g.swiglu_limit,
                    hb.p(&st).m(),
                    inter as i64,
                    n as i32,
                    inter as i32,
                    self.ops.sv(),
                )
            })?;
            self.ops.linear_q(
                hb.p(&st),
                inter,
                n,
                w2,
                db.p(&st),
                g.dim,
                false,
                &mut self.scr,
            )?;
            ck("scatter_add", unsafe {
                ffi::memra_v41_scatter_add(
                    db.p(&st).m(),
                    g.dim as i64,
                    dri.p(&st).i(),
                    y.p(&st).fm(),
                    g.dim as i64,
                    n as i32,
                    g.dim as i32,
                    self.ops.sv(),
                )
            })?;
            // the per-expert index and weight buffers must outlive their kernels
            self.ops.sync()?;
        }
        let [s1, s2, s3] = &lw.shared;
        self.ops
            .linear_q(x, g.dim, s, s1, gb.p(&st), inter, false, &mut self.scr)?;
        self.ops
            .linear_q(x, g.dim, s, s3, ub.p(&st), inter, false, &mut self.scr)?;
        ck("swiglu", unsafe {
            ffi::memra_v41_swiglu(
                gb.p(&st).m(),
                ub.p(&st).m(),
                inter as i64,
                std::ptr::null(),
                g.swiglu_limit,
                hb.p(&st).m(),
                inter as i64,
                s as i32,
                inter as i32,
                self.ops.sv(),
            )
        })?;
        self.ops.linear_q(
            hb.p(&st),
            inter,
            s,
            s2,
            db.p(&st),
            g.dim,
            false,
            &mut self.scr,
        )?;
        ck("add_cast", unsafe {
            ffi::memra_v41_add_cast(
                y.p(&st).f(),
                db.p(&st).m(),
                out.m(),
                (s * g.dim) as i64,
                self.ops.sv(),
            )
        })?;
        self.ops.sync()
    }

    /// Single-pass mHC mixes of the stream `x` `[s][hc][dim]`: `(mixes, pre, post, comb)`.
    pub fn hc_mix(
        &mut self,
        fn_: &Buf,
        sc: &Buf,
        base: &Buf,
        x: &Buf,
        s: usize,
    ) -> Result<(Buf, Buf, Buf, Buf), String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let nmix = (2 + hc) * hc;
        let mixes = self.ops.buf(s * nmix * 4)?;
        let pre = self.ops.buf(s * hc * 4)?;
        let post = self.ops.buf(s * hc * 4)?;
        let comb = self.ops.buf(s * hc * hc * 4)?;
        ck("hc_mixes", unsafe {
            ffi::memra_v41_hc_mixes(
                x.p(&st).m(),
                fn_.p(&st).f(),
                sc.p(&st).f(),
                base.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                g.sinkhorn_iters as i32,
                g.norm_eps,
                g.hc_eps,
                mixes.p(&st).fm(),
                pre.p(&st).fm(),
                post.p(&st).fm(),
                comb.p(&st).fm(),
                self.ops.sv(),
            )
        })?;
        Ok((mixes, pre, post, comb))
    }

    /// The collapse of the stream `h` with `mix` `[s][hc]`, then RMSNorm with `norm`: `[s][dim]`.
    pub fn hc_pre_norm(&mut self, h: &Buf, mix: &Buf, norm: &Buf, s: usize) -> Result<Buf, String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let x = self.ops.buf(s * d * 2)?;
        ck("hc_pre", unsafe {
            ffi::memra_v41_hc_pre(
                h.p(&st).m(),
                mix.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                x.p(&st).m(),
                d as i64,
                self.ops.sv(),
            )
        })?;
        self.ops
            .rmsnorm(x.p(&st), d, norm, x.p(&st), d, s, d, g.norm_eps)?;
        Ok(x)
    }

    /// The expansion of a sublayer output `f` `[s][dim]` into the residual stream `res`.
    pub fn hc_post(
        &mut self,
        f: &Buf,
        res: &Buf,
        post: &Buf,
        comb: &Buf,
        s: usize,
    ) -> Result<Buf, String> {
        let g = &self.g;
        let st = self.ops.st.clone();
        let (hc, d) = (g.hc, g.dim);
        let out = self.ops.buf(s * hc * d * 2)?;
        ck("hc_post", unsafe {
            ffi::memra_v41_hc_post(
                f.p(&st).m(),
                d as i64,
                res.p(&st).m(),
                post.p(&st).f(),
                comb.p(&st).f(),
                s as i32,
                hc as i32,
                d as i32,
                out.p(&st).m(),
                self.ops.sv(),
            )
        })?;
        Ok(out)
    }

    /// One trunk block (vendor `Block.forward`) for `s` tokens from position `sp`: a prefill chunk
    /// at 0 or one decode token. `h` is the bf16 stream `[s][hc][dim]` after this layer's Engram
    /// (if any), `pre_mix` the carried f32 `[s][hc]` from the previous FFN; `ls` is this layer's
    /// own window ring and compressor state, `sh` the CSA2 state handed down the stack. Returns the
    /// new stream and the FFN's pre-mix for the next layer.
    #[allow(clippy::too_many_arguments)]
    pub fn block_forward(
        &mut self,
        lw: &LayerW,
        h: &Buf,
        pre_mix: &Buf,
        sp: usize,
        s: usize,
        sh: &mut Shared,
        ls: &mut LayerState,
        tr: &mut Trace,
    ) -> Result<(Buf, Buf), String> {
        if lw.part.n != 1 {
            return Err(
                "block_forward runs a whole layer; a rank's share runs in dsv41_gpu::tp".into(),
            );
        }
        let g = self.g.clone();
        let st = self.ops.st.clone();
        let d = g.dim;
        let (amix, a_pre, a_post, a_comb) =
            self.hc_mix(&lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base, h, s)?;
        let x = self.hc_pre_norm(h, pre_mix, &lw.attn_norm, s)?;
        if self.trace {
            tr.attn_mix = Some(amix);
            tr.attn_in = Some(clone_buf(&self.ops, &x)?);
        }
        let a = self.attention(lw, &x, sp, s, sh, ls, tr)?;
        let h1 = self.hc_post(&a, h, &a_post, &a_comb, s)?;
        let (_fmix, f_pre, f_post, f_comb) =
            self.hc_mix(&lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base, &h1, s)?;
        let x2 = self.hc_pre_norm(&h1, &a_pre, &lw.ffn_norm, s)?;
        // the MoE runs `chunk` tokens at a time: each token's experts still add in ascending id order
        let m = self.ops.buf(s * d * 2)?;
        // larger slices here: the per-expert loop runs once per slice
        let cm = (4 * self.chunk).min(s).max(1);
        for r0 in (0..s).step_by(cm) {
            let rows = cm.min(s - r0);
            self.moe(lw, x2.at(&st, r0 * d * 2), rows, m.at(&st, r0 * d * 2), tr)?;
        }
        let h2 = self.hc_post(&m, &h1, &f_post, &f_comb, s)?;
        if self.trace {
            tr.attn_out = Some(a);
            tr.mid = Some(h1);
            tr.moe_in = Some(x2);
            tr.moe_out = Some(m);
        }
        self.ops.sync()?;
        Ok((h2, f_pre))
    }
}

fn clone_buf(ops: &Ops, b: &Buf) -> Result<Buf, String> {
    let o = ops.buf(b.len)?;
    copy_d2d(ops, b, 0, &o, 0, b.len)?;
    Ok(o)
}

/// `dst[doff..doff+n] = src[soff..soff+n]` on the stream.
pub fn copy_d2d(
    ops: &Ops,
    src: &Buf,
    soff: usize,
    dst: &Buf,
    doff: usize,
    n: usize,
) -> Result<(), String> {
    if soff + n > src.len || doff + n > dst.len {
        return Err(format!("copy_d2d: {n} bytes out of range"));
    }
    if n == 0 {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst.p(&ops.st).addr() + doff as u64,
            src.p(&ops.st).addr() + soff as u64,
            n,
            ops.sv() as cudarc::driver::sys::CUstream,
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("memcpy d2d: {r:?}"));
    }
    Ok(())
}

/// Copy `rows` rows of `width` bytes from `src` (pitch `spitch`, starting at row `src_row`) into `dst` at byte column
/// `dcol` (pitch `dpitch`).
#[allow(clippy::too_many_arguments)]
pub fn copy_2d(
    ops: &Ops,
    src: &Buf,
    src_row: usize,
    spitch: usize,
    dst: &Buf,
    dcol: usize,
    dpitch: usize,
    width: usize,
    rows: usize,
) -> Result<(), String> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    use cudarc::driver::sys::*;
    let p = CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: src_row,
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src.p(&ops.st).addr(),
        srcArray: std::ptr::null_mut(),
        srcPitch: spitch,
        dstXInBytes: dcol,
        dstY: 0,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst.p(&ops.st).addr(),
        dstArray: std::ptr::null_mut(),
        dstPitch: dpitch,
        WidthInBytes: width,
        Height: rows,
    };
    let r = unsafe { cuMemcpy2DAsync_v2(&p, ops.sv() as CUstream) };
    if r != CUresult::CUDA_SUCCESS {
        return Err(format!("memcpy2d: {r:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_is_exact_and_refuses_what_e4m3_cannot_hold() {
        // 2^-3, 2^4 and 2^-9 relative to a tensor maximum of 2^4: 13 octaves, inside e4m3's 17
        let sc = [124u8, 131, 118];
        let (codes, s2) = mint_mxfp4_scales(&sc).unwrap();
        assert_eq!(codes.len(), 6);
        for (i, &b) in sc.iter().enumerate() {
            assert_eq!(
                codes[2 * i],
                codes[2 * i + 1],
                "both per-16 halves carry the per-32 scale"
            );
            assert_eq!(
                e4m3_to_f32(codes[2 * i]) * s2,
                e8m0_to_f32(b),
                "scale {b} survives the mint"
            );
        }
        // 18 octaves of spread cannot be one e4m3 power of two each
        assert!(mint_mxfp4_scales(&[100u8, 118]).is_err());
        assert!(mint_mxfp4_scales(&[0xFF]).is_err());
    }

    #[test]
    fn window_rows_match_the_reference_prefill_selection() {
        let (v, w) = Ctx::window_idx(5, 3);
        assert_eq!(w, 3);
        assert_eq!(v, vec![0, -1, -1, 0, 1, -1, 0, 1, 2, 1, 2, 3, 2, 3, 4]);
    }
}
