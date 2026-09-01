//! memra inference runtime. Correctness-first: every GPU op is validated against a
//! CPU reference before any sm_120 fast-path replaces it.

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{CudaContext, CudaStream, sys as cu};
use std::sync::Arc;

pub use memra_gguf;

/// CPU reference matmul for a linear layer y = x @ W^T.
/// Conventions (ggml/GGUF): a weight tensor with ne=[in, out] is stored row-major as
/// `out` rows of `in` contiguous elements — i.e. W[o*in + i]. A linear layer computes
/// y[o] = sum_i x[i] * W[o*in + i], for each of `out` outputs. Batched over `m` tokens:
///   x: [m, in] row-major (x[t*in + i]); w: [out, in] row-major (w[o*in + i]); y: [m, out].
#[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
pub fn cpu_linear(x: &[f32], w: &[f32], m: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * in_f);
    assert_eq!(w.len(), out_f * in_f);
    let mut y = vec![0f32; m * out_f];
    for t in 0..m {
        for o in 0..out_f {
            let mut acc = 0f32;
            let xr = &x[t * in_f..t * in_f + in_f];
            let wr = &w[o * in_f..o * in_f + in_f];
            for i in 0..in_f {
                acc += xr[i] * wr[i];
            }
            y[t * out_f + o] = acc;
        }
    }
    y
}

/// GPU runtime handle: a context + stream + cuBLASLt.
pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    /// The MAIN compute stream. PRIVATE since M1 increment 2: every launch site reads
    /// `stream()` so the pp2 per-stage stream override (below) is a single seam. Naked
    /// paths (no override pushed) get exactly this stream back — behavior unchanged.
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlasLT>,
    /// TOKEN-PIPELINE phase streams (step37 chain): two extra stream/cuBLASLt pairs so
    /// alternate tokens' host-issue rides disjoint streams. Lazily built (first
    /// enter_main under an active decode phase); None everywhere else — zero cost.
    #[allow(clippy::type_complexity)]
    // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    phase: std::sync::Mutex<Option<[(Arc<CudaStream>, Arc<CudaBlasLT>); 2]>>,
}

// ---------------------------------------------------------------------------------------
// M1-PP2 increment 2: AMBIENT STREAM OVERRIDE (per-stage CUDA streams).
//
// The engine's entire launch surface reads `Gpu::stream()`. A pipeline stage redirects it
// by pushing a per-stage stream onto this thread-local stack for the stage's host-issue
// scope (RAII guard pops it). Decode is single-threaded host-issue, so thread-local is the
// natural scope; cost when the stack is empty (every naked path) is one TLS lookup + a
// branch + one Arc clone per launch — nanoseconds against a kernel launch.
//
// SAFETY CONTRACT (the multi-stream law): cudarc's per-arg event tracking stays DISABLED
// (see memra-engine Engine::new) — cross-stream ordering is the OVERRIDER's job, via
// explicit CudaEvents (pp2's boundary TX/RX choreography). The async mem pool is configured
// below with opportunistic reuse OFF + internal dependencies ON, so a block freed on stream
// A and re-allocated on stream B carries a driver-inserted dependency — alloc reuse cannot
// race across stages. Buffers that one stream writes and another reads must be evented by
// the caller; pp2 routes ALL cross-stage bytes through its persistent boundary slots.
// ---------------------------------------------------------------------------------------
struct StreamBinding {
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlasLT>,
}

thread_local! {
    static STREAM_OVERRIDE: std::cell::RefCell<Vec<StreamBinding>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    static DECODE_PHASE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

thread_local! {
    /// RANK0 STREAM MERGE (step37): while set to a (ctx, stream, blas) binding, enter_main
    /// on the MATCHING context binds THIS stream instead of the gpu's own main stream —
    /// the same-device rank's work then rides the model engine's stream and every
    /// e<->rank0 event hop becomes same-stream program order.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    static RANK0_REDIRECT: std::cell::RefCell<Option<(usize, Arc<CudaStream>, Arc<CudaBlasLT>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Install/clear the rank0 redirect (ctx ordinal + stream/blas of the model engine).
pub fn set_rank0_redirect(binding: Option<(usize, Arc<CudaStream>, Arc<CudaBlasLT>)>) {
    RANK0_REDIRECT.with(|c| *c.borrow_mut() = binding);
}

/// RAII scope for the rank0 redirect: clears on drop (panic-safe).
pub struct Rank0RedirectGuard(());
pub fn rank0_redirect_scope(
    ordinal: usize,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlasLT>,
) -> Rank0RedirectGuard {
    set_rank0_redirect(Some((ordinal, stream, blas)));
    Rank0RedirectGuard(())
}
impl Drop for Rank0RedirectGuard {
    fn drop(&mut self) {
        set_rank0_redirect(None);
    }
}
fn rank0_redirect_for(ordinal: usize) -> Option<(Arc<CudaStream>, Arc<CudaBlasLT>)> {
    RANK0_REDIRECT.with(|c| {
        c.borrow()
            .as_ref()
            .filter(|(o, ..)| *o == ordinal)
            .map(|(_, s, b)| (s.clone(), b.clone()))
    })
}

/// TOKEN-PIPELINE phase (step37 chain): while `Some(p)`, `enter_main` binds each gpu's
/// phase-p stream instead of its main stream, so alternate tokens' rank-local work rides
/// disjoint streams. Cross-stream ordering is the SETTER's job (the multi-stream law
/// above): the chain wires per-layer KV events between phases.
pub fn set_decode_phase(p: Option<usize>) {
    DECODE_PHASE.with(|c| c.set(p));
}
pub fn decode_phase() -> Option<usize> {
    DECODE_PHASE.with(|c| c.get())
}

/// RAII scope: while alive, `Gpu::stream()` on THIS thread returns the pushed stream.
/// Nest freely (stack). Popping on Drop keeps panic paths consistent.
pub struct StreamOverride(());

/// A rank-local CUDA scope nested inside another engine's PP stage scope.
///
/// CUDA contexts are a per-thread stack. The stream override alone is not enough: every rank-local
/// allocation and launch must make that rank's context current, then restore the caller's context
/// before the PP owner resumes issuing work.
pub struct GpuMainOverride {
    stream: Option<StreamOverride>,
    expected_ctx: cu::CUcontext,
}

/// Push a matched stream/cuBLASLt binding for the current thread until the guard drops.
pub fn push_stream_override(stream: Arc<CudaStream>, blas: Arc<CudaBlasLT>) -> StreamOverride {
    STREAM_OVERRIDE.with(|o| o.borrow_mut().push(StreamBinding { stream, blas }));
    StreamOverride(())
}

impl Drop for StreamOverride {
    fn drop(&mut self) {
        STREAM_OVERRIDE.with(|o| {
            o.borrow_mut().pop();
        });
    }
}

impl Drop for GpuMainOverride {
    fn drop(&mut self) {
        // Restore the ambient PP stream/cuBLAS binding before restoring its CUDA context.
        drop(self.stream.take());
        // cudarc binds the context of every stream operation with cuCtxSetCurrent. NVIDIA defines
        // that call as replacing the top entry of an existing context stack, so a cross-context
        // operation may replace the slot that enter_main pushed. Put the owning context back in
        // that slot before popping it; the untouched PP context beneath it then becomes current.
        let set_rc = unsafe { cu::cuCtxSetCurrent(self.expected_ctx) };
        let mut popped = std::ptr::null_mut();
        let pop_rc = unsafe { cu::cuCtxPopCurrent_v2(&mut popped) };
        if set_rc != cu::CUresult::CUDA_SUCCESS
            || pop_rc != cu::CUresult::CUDA_SUCCESS
            || popped != self.expected_ctx
        {
            let message = format!(
                "rank-local CUDA context restore failed: set_rc={set_rc:?} pop_rc={pop_rc:?} \
                 expected={:?} popped={popped:?}",
                self.expected_ctx,
            );
            if std::thread::panicking() {
                eprintln!("{message}");
            } else {
                panic!("{message}");
            }
        }
    }
}

impl Gpu {
    /// The stream every engine op launches on: the thread's override if one is pushed
    /// (pp2 stage scopes), else the main compute stream. By-value Arc so callers hold a
    /// stable handle across the call regardless of later pushes/pops.
    #[inline]
    pub fn stream(&self) -> Arc<CudaStream> {
        STREAM_OVERRIDE
            .with(|o| o.borrow().last().map(|binding| binding.stream.clone()))
            .unwrap_or_else(|| self.stream.clone())
    }

    /// The cuBLASLt handle bound to the same stream returned by `stream()`.
    #[inline]
    pub fn blas(&self) -> Arc<CudaBlasLT> {
        STREAM_OVERRIDE
            .with(|o| o.borrow().last().map(|binding| binding.blas.clone()))
            .unwrap_or_else(|| self.blas.clone())
    }

    /// The main compute stream, override-blind (graph capture pins itself here; the pp2
    /// runtime uses it to fence stage streams against load-time state).
    #[inline]
    pub fn main_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Enter this GPU's own context and matched main stream/cuBLASLt binding, even when the
    /// calling thread currently carries a pipeline-stage stream override for another engine.
    ///
    /// Multi-context TP/EP helpers invoke rank-local engines from inside a PP stage scope. Without
    /// this nested binding, `stream()` would inherit the PP owner's stream and launch rank-local
    /// pointers through the wrong CUDA context.
    pub fn enter_main(&self) -> Result<GpuMainOverride, Box<dyn std::error::Error>> {
        let rc = unsafe { cu::cuCtxPushCurrent_v2(self.ctx.cu_ctx()) };
        if rc != cu::CUresult::CUDA_SUCCESS {
            return Err(format!("rank-local CUDA context push failed: {rc:?}").into());
        }
        // Token-pipeline phase / rank0 redirect: bind the override stream when armed.
        let (stream, blas) = if let Some(pair) = rank0_redirect_for(self.ctx.ordinal()) {
            pair
        } else {
            match decode_phase() {
                Some(p) => self.phase_pair(p)?,
                None => (self.stream.clone(), self.blas.clone()),
            }
        };
        Ok(GpuMainOverride {
            stream: Some(push_stream_override(stream, blas)),
            expected_ctx: self.ctx.cu_ctx(),
        })
    }

    /// The phase-`p` stream/cuBLASLt pair, lazily created. Caller must have this gpu's
    /// context current (enter_main does; external callers use enter_main first).
    pub fn phase_pair(
        &self,
        p: usize,
    ) -> Result<(Arc<CudaStream>, Arc<CudaBlasLT>), Box<dyn std::error::Error>> {
        let mut guard = self
            .phase
            .lock()
            .map_err(|_| "gpu phase lock is poisoned")?;
        if guard.is_none() {
            let s0 = self.ctx.new_stream()?;
            let s1 = self.ctx.new_stream()?;
            let b0 = Arc::new(CudaBlasLT::new(s0.clone())?);
            let b1 = Arc::new(CudaBlasLT::new(s1.clone())?);
            *guard = Some([(s0, b0), (s1, b1)]);
        }
        let arr = guard.as_ref().expect("armed above");
        Ok((arr[p & 1].0.clone(), arr[p & 1].1.clone()))
    }
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        // A NON-BLOCKING created stream (NOT the legacy NULL/default stream): the NULL stream cannot be
        // CUDA-graph captured (cuStreamBeginCapture -> CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). All
        // engine kernels launch on this stream, so making it capturable enables the decode CUDA-graph
        // capture/replay path (CUDA-GRAPH-PLAN Phase 3). Behaviorally identical for the existing single-
        // stream paths (just a real stream id instead of NULL).
        let stream = ctx.new_stream()?;
        // DETERMINISM FIX (decode bit-stability): the default stream-ordered async memory pool
        // (cuMemAllocAsync/cuMemFreeAsync, used by cudarc's `alloc`/`alloc_zeros`) reuses freed
        // blocks OPPORTUNISTICALLY — it hands a freed block to the next alloc as soon as the HOST
        // observes the GPU has passed the free, WITHOUT inserting a stream dependency. Whether that
        // reuse happens is a function of how far the async GPU has progressed at host-alloc time, so
        // it is timing-dependent. Our decode path launches kernels through the raw launch builder
        // (no cudarc read/write event tracking on the args), so the per-step scratch buffers are
        // freed-and-reused inside the async window: under opportunistic reuse a buffer can be
        // recycled and overwritten by a later kernel while an earlier kernel that still references
        // the same physical block is in flight — a WAR/RAW hazard that produces RUN-TO-RUN
        // nondeterministic results (two identical prompt primes diverge; per-step sync hides it).
        // Disable opportunistic reuse and require the pool to insert INTERNAL stream dependencies
        // before reusing a freed block. This makes every reuse stream-ordered and deterministic with
        // negligible cost (one-time pool config; the dependency is the same ordering the single
        // stream already implies, just made explicit). The release threshold is set to MAX so freed
        // blocks stay in the pool (no give-back to the OS between steps -> stable reuse, no per-step
        // cuMemMap churn).
        // (A/B-verified perf-neutral: decode ~80 tok/s with this on, off, or absent — full-power noise
        // band. The real determinism fix is the SSM ping-pong in decode.rs; this is cheap belt-and-
        // suspenders against any other per-step async-pool reuse hazard.)
        unsafe {
            use cudarc::driver::sys;
            let dev = ctx.cu_device();
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            if sys::cuDeviceGetDefaultMemPool(&mut pool, dev) == sys::CUresult::CUDA_SUCCESS
                && !pool.is_null()
            {
                let off: std::os::raw::c_int = 0;
                let _ = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
                    &off as *const _ as *mut std::os::raw::c_void,
                );
                let on: std::os::raw::c_int = 1;
                let _ = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                    &on as *const _ as *mut std::os::raw::c_void,
                );
                let thresh: u64 = u64::MAX;
                let _ = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &thresh as *const _ as *mut std::os::raw::c_void,
                );
            }
        }
        let blas = Arc::new(CudaBlasLT::new(stream.clone())?);
        Ok(Self {
            ctx,
            stream,
            blas,
            phase: std::sync::Mutex::new(None),
        })
    }

    /// GPU linear y = x @ W^T using cuBLASLt (f32), matching `cpu_linear` exactly.
    ///
    /// Layout reasoning (cuBLASLt is column-major):
    /// We want y[m,out] row-major = y^T[out,m] column-major. Treat:
    ///   - x[m,in] row-major == x^T[in,m] col-major  (an in×m col-major matrix)
    ///   - w[out,in] row-major == w^T[in,out] col-major (an in×out col-major matrix)
    ///     Compute C[out,m] col-major = W_colmajor(out×in) * X_colmajor(in×m)
    ///     => set A = w (interpreted col-major as in×out, so transa to get out×in),
    ///     B = x (col-major in×m), C = y (col-major out×m == y[m,out] row-major).
    ///     cfg: m_=out, n_=m_tokens, k=in. A is in×out (lda=in, transa=true -> out×in),
    ///     B is in×m (ldb=in), C is out×m (ldc=out).
    pub fn linear_f32(
        &self,
        x: &cudarc::driver::CudaSlice<f32>,
        w: &cudarc::driver::CudaSlice<f32>,
        m_tokens: usize,
        in_f: usize,
        out_f: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let stream = self.stream();
        let mut c = stream.alloc_zeros::<f32>(m_tokens * out_f)?;
        let cfg = MatmulConfig {
            transa: true, // A stored in×out col-major -> use as out×in
            transb: false,
            transc: false,
            m: out_f as u64,
            n: m_tokens as u64,
            k: in_f as u64,
            alpha: 1.0,
            lda: in_f as i64, // A leading dim = in (col-major in×out)
            ldb: in_f as i64, // B leading dim = in (col-major in×m)
            beta: 0.0,
            ldc: out_f as i64, // C leading dim = out (col-major out×m)
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let blas = self.blas();
        unsafe {
            blas.matmul(cfg, w, x, &mut c, None, None)?;
        }
        let y = stream.clone_dtoh(&c)?;
        stream.synchronize()?;
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_linear_tiny() {
        // m=1, in=2, out=2; x=[1,2], W=[[1,0],[0,1]] (identity) -> y=[1,2]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0]; // row0=[1,0], row1=[0,1]
        let y = cpu_linear(&x, &w, 1, 2, 2);
        assert_eq!(y, vec![1.0, 2.0]);
        // W=[[1,1],[2,0]] -> y[0]=1*1+2*1=3, y[1]=1*2+2*0=2
        let w2 = vec![1.0, 1.0, 2.0, 0.0];
        let y2 = cpu_linear(&x, &w2, 1, 2, 2);
        assert_eq!(y2, vec![3.0, 2.0]);
    }
}
