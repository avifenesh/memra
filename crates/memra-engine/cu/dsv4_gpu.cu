// dsv4_gpu.cu — DeepSeek-V4-Flash GPU trunk kernels (lane 4, correctness bring-up).
//
// Semantic source: the official reference model.py/kernel.py as resolved in darklanes
// research/deepseek-flash-20260818/SEMANTICS.md; the numeric contract is the lane-3 CPU
// oracle (crates/memra-gguf/src/dsv4_forward.rs) — every kernel here mirrors that
// oracle's arithmetic: f32 elementwise math, f64 accumulation for dots and long
// reductions, exact QAT grid math (pow2-ceil scales, e2m1 RNE thresholds), and the
// modelopt e4m3 NaN-code -> 0.0 weight convention.
//
// Rung declaration (banked in RECEIPTS.md "Lane 4" BEFORE this file was written):
// weights ride bf16 (all dequants are EXACT in bf16 — <=5 significand bits x pow2);
// activations stay f32 host-visible and are cast to bf16 ONLY at the inputs of the
// non-f32-island GEMMs (cuBLASLt bf16 x bf16 -> f32, CUBLAS_COMPUTE_32F). f32 islands
// (SEMANTICS §7.2) run in the dedicated f32/f64 kernels below.
//
// Existing-kernel survey (why these are new, per the reuse-or-new law):
//   - rms_norm_f32 (kernels.cu) reduces in f32; the oracle contract is an f64 mean —
//     memra_dsv4_rmsnorm keeps oracle parity.
//   - swiglu_clamped_mul_scaled_f32 (hybrid.cu) clamps AFTER silu (min(silu(g), limit));
//     dsv4 clamps the gate BEFORE silu (silu(min(g, limit))) — a real semantic fork.
//   - rope_neox* pair (i, i+half); dsv4 rope is interleaved complex pairs (2k, 2k+1) on
//     the LAST rd dims with an inverse (conjugate) form — no engine twin.
//   - hc/compressor/indexer/sink-attention/sqrtsoftplus-MoE have no engine arm at all.
//
// House pattern: static-lib TU with extern "C" host launchers (mmq_ffi kind), errors
// 0 ok / 10000+cudaError / cublas bands, stream passed as void*.

#include <cublasLt.h>
// NVTX is a RESEARCH-INSTRUMENT dependency (memra_dsv4_nvtx_push/pop, armed only by
// MEMRA_DSV4_NVTX=1 at runtime) — the header is optional so the TU builds on toolkits
// that ship without nvtx3 (GitHub CI's minimal CUDA install; the v0.98 train's CI
// caught the unconditional include). Without the header the push/pop launchers become
// no-ops: profiling silently absent is acceptable for an opt-in instrument, an engine
// that cannot BUILD is not.
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MEMRA_DSV4_HAVE_NVTX 1
#endif
#endif
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>

#define DSV4_ERR()                                             \
    do {                                                       \
        cudaError_t ce_ = cudaGetLastError();                  \
        if (ce_ != cudaSuccess) return 10000 + (int)ce_;       \
    } while (0)

// ---------------------------------------------------------------- numeric primitives

// e2m1 magnitude table, code = nibble (bit 3 = sign). Mirrors dsv4.rs E2M1.
__device__ __constant__ float DSV4_E2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                               4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                               -2.0f, -3.0f, -4.0f, -6.0f};

// fp8_e4m3_to_f32 (nvfp4_repack.rs:103): NaN code (mag 0x7F) -> 0.0 (modelopt WEIGHT
// convention).
__device__ __forceinline__ float dsv4_e4m3(uint8_t x) {
    uint8_t mag = x & 0x7F;
    if (mag == 0x7F) return 0.0f;
    float sign = (x & 0x80) ? -1.0f : 1.0f;
    int exp = (mag >> 3) & 0xF;
    float man = (float)(mag & 0x7);
    float raw = (exp == 0) ? (man / 8.0f) * exp2f(-6.0f)
                           : (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
    return sign * raw;
}

// pow2_ceil (dsv4_forward.rs / kernel.py fast_round_scale): 2^ceil(log2(x)), exact bit math.
__device__ __forceinline__ float dsv4_pow2_ceil(float x) {
    unsigned bits = __float_as_uint(x);
    int exp = (int)((bits >> 23) & 0xFF);
    unsigned man = bits & ((1u << 23) - 1);
    int e = exp - 127 + (man != 0 ? 1 : 0);
    return exp2f((float)e);
}

// e2m1_rne (dsv4_forward.rs:328): nearest on the e2m1 grid, ties to even mantissa bit.
__device__ __forceinline__ float dsv4_e2m1_rne(float v) {
    const float GRID[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float a = fabsf(v);
    int idx = 0;
    // midpoints below ODD-mantissa upper neighbours: go up only if strictly greater
    idx += (a > 0.25f); idx += (a > 1.25f); idx += (a > 2.5f); idx += (a > 5.0f);
    // midpoints below EVEN-mantissa upper neighbours: ties go up
    idx += (a >= 0.75f); idx += (a >= 1.75f); idx += (a >= 3.5f);
    return (v < 0.0f) ? -GRID[idx] : GRID[idx];
}

__device__ __forceinline__ float dsv4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// block tree-reduce of double partials in shared memory (deterministic: fixed tree).
__device__ __forceinline__ double dsv4_block_sum(double v, double* sh) {
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    return sh[0];
}

__device__ __forceinline__ float dsv4_block_max(float v, float* sh) {
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sh[tid] = fmaxf(sh[tid], sh[tid + off]);
        __syncthreads();
    }
    return sh[0];
}

// ---------------------------------------------------------------- dequant kernels

// modelopt NVFP4 expert -> bf16: out[r,c] = (E2M1[code] * e4m3(scale[r, c/16])) * scale_2.
// Same multiply order as dsv4.rs dequant_nvfp4_expert; the product is EXACT in bf16
// (<=5 significand bits x pow2 scale_2 — scale_2 pow2 asserted by the Rust loader).
// One thread per byte (2 output elements). elem 2i -> LOW nibble (modelopt).
extern "C" __global__ void dsv4_nvfp4_deq_bf16_kernel(const uint8_t* __restrict__ w,
                                                      const uint8_t* __restrict__ sc,
                                                      float scale2, int rows, int cols,
                                                      __nv_bfloat16* __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;   // byte index
    long nbytes = (long)rows * (cols / 2);
    if (i >= nbytes) return;
    int cb = cols / 2;
    int r = (int)(i / cb);
    int c0 = 2 * (int)(i % cb);
    uint8_t byte = w[i];
    const uint8_t* srow = sc + (long)r * (cols / 16);
    float s0 = dsv4_e4m3(srow[c0 / 16]);
    float s1 = dsv4_e4m3(srow[(c0 + 1) / 16]);
    float v0 = DSV4_E2M1[byte & 0x0F] * s0 * scale2;
    float v1 = DSV4_E2M1[byte >> 4] * s1 * scale2;
    out[(long)r * cols + c0] = __float2bfloat16(v0);
    out[(long)r * cols + c0 + 1] = __float2bfloat16(v1);
}

extern "C" int memra_dsv4_nvfp4_deq_bf16(const void* w, const void* sc, float scale2,
                                         int rows, int cols, void* out, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long nbytes = (long)rows * (cols / 2);
    int threads = 256;
    long blocks = (nbytes + threads - 1) / threads;
    dsv4_nvfp4_deq_bf16_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint8_t*)w, (const uint8_t*)sc, scale2, rows, cols, (__nv_bfloat16*)out);
    DSV4_ERR();
    return 0;
}

// OCP MXFP4 (MTP experts) -> bf16: E2M1[code] * 2^(scale_byte - 127). An 0xFF scale byte
// propagates NaN (e8m0_to_f32 semantics) — the loader NaN-sweeps after dequant.
extern "C" __global__ void dsv4_mxfp4_deq_bf16_kernel(const uint8_t* __restrict__ w,
                                                      const uint8_t* __restrict__ sc,
                                                      int rows, int cols,
                                                      __nv_bfloat16* __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long nbytes = (long)rows * (cols / 2);
    if (i >= nbytes) return;
    int cb = cols / 2;
    int r = (int)(i / cb);
    int c0 = 2 * (int)(i % cb);
    uint8_t byte = w[i];
    const uint8_t* srow = sc + (long)r * (cols / 32);
    uint8_t b0 = srow[c0 / 32], b1 = srow[(c0 + 1) / 32];
    float s0 = (b0 == 0xFF) ? nanf("") : exp2f((float)b0 - 127.0f);
    float s1 = (b1 == 0xFF) ? nanf("") : exp2f((float)b1 - 127.0f);
    out[(long)r * cols + c0] = __float2bfloat16(DSV4_E2M1[byte & 0x0F] * s0);
    out[(long)r * cols + c0 + 1] = __float2bfloat16(DSV4_E2M1[byte >> 4] * s1);
}

extern "C" int memra_dsv4_mxfp4_deq_bf16(const void* w, const void* sc, int rows, int cols,
                                         void* out, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long nbytes = (long)rows * (cols / 2);
    int threads = 256;
    long blocks = (nbytes + threads - 1) / threads;
    dsv4_mxfp4_deq_bf16_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint8_t*)w, (const uint8_t*)sc, rows, cols, (__nv_bfloat16*)out);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- casts / gathers

extern "C" __global__ void dsv4_cvt_bf16_kernel(const float* __restrict__ x,
                                                __nv_bfloat16* __restrict__ o, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2bfloat16(x[i]);
}

extern "C" int memra_dsv4_cvt_bf16(const float* x, void* o, long n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_cvt_bf16_kernel<<<(unsigned)blocks, threads, 0, stream>>>(x, (__nv_bfloat16*)o, n);
    DSV4_ERR();
    return 0;
}

// bf16 rows of the embed table -> f32 rows for the given token ids (bit-exact decode).
extern "C" __global__ void dsv4_embed_rows_kernel(const uint16_t* __restrict__ table,
                                                  const int* __restrict__ ids,
                                                  float* __restrict__ out, int n_ids,
                                                  int ncols) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n_ids * ncols) return;
    int row = (int)(i / ncols), c = (int)(i % ncols);
    uint16_t h = table[(long)ids[row] * ncols + c];
    out[i] = __uint_as_float(((unsigned)h) << 16);
}

extern "C" int memra_dsv4_embed_rows(const void* table_bf16, const int* ids, float* out,
                                     int n_ids, int ncols, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)n_ids * ncols;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_embed_rows_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint16_t*)table_bf16, ids, out, n_ids, ncols);
    DSV4_ERR();
    return 0;
}

// gather bf16 rows by index: out[g, :] = x[idx[g], :] (expert token-group assembly).
extern "C" __global__ void dsv4_gather_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                   const int* __restrict__ idx,
                                                   __nv_bfloat16* __restrict__ out, int g,
                                                   int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)g * d) return;
    int r = (int)(i / d), c = (int)(i % d);
    out[i] = x[(long)idx[r] * d + c];
}

extern "C" int memra_dsv4_gather_bf16(const void* x, const int* idx, void* out, int g, int d,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)g * d;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_gather_bf16_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)x, idx, (__nv_bfloat16*)out, g, d);
    DSV4_ERR();
    return 0;
}

// scatter-add f32 rows: y[idx[g], :] += contrib[g, :]. Row indices are UNIQUE within one
// launch (a token selects an expert at most once — tid2eid duplicates refused at load),
// so no atomics: deterministic.
extern "C" __global__ void dsv4_scatter_add_kernel(float* __restrict__ y,
                                                   const float* __restrict__ contrib,
                                                   const int* __restrict__ idx, int g, int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)g * d) return;
    int r = (int)(i / d), c = (int)(i % d);
    y[(long)idx[r] * d + c] += contrib[i];
}

extern "C" int memra_dsv4_scatter_add(float* y, const float* contrib, const int* idx, int g,
                                      int d, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)g * d;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_scatter_add_kernel<<<(unsigned)blocks, threads, 0, stream>>>(y, contrib, idx, g, d);
    DSV4_ERR();
    return 0;
}

extern "C" __global__ void dsv4_add_inplace_kernel(float* __restrict__ y,
                                                   const float* __restrict__ x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

extern "C" int memra_dsv4_add_inplace(float* y, const float* x, long n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_add_inplace_kernel<<<(unsigned)blocks, threads, 0, stream>>>(y, x, n);
    DSV4_ERR();
    return 0;
}

// copy a column window: dst[t, 0..n) = src[t, col_off .. col_off+n) (stride = src row width)
extern "C" __global__ void dsv4_take_cols_kernel(const float* __restrict__ src,
                                                 float* __restrict__ dst, int s, int n,
                                                 long stride, long col_off) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)s * n) return;
    int t = (int)(i / n), c = (int)(i % n);
    dst[i] = src[(long)t * stride + col_off + c];
}

extern "C" int memra_dsv4_take_cols(const float* src, float* dst, int s, int n, long stride,
                                    long col_off, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)s * n;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_take_cols_kernel<<<(unsigned)blocks, threads, 0, stream>>>(src, dst, s, n, stride,
                                                                    col_off);
    DSV4_ERR();
    return 0;
}

// place into a column window: dst[t, col_off .. col_off+n) = src[t, 0..n)
extern "C" __global__ void dsv4_place_cols_kernel(const float* __restrict__ src,
                                                  float* __restrict__ dst, int s, int n,
                                                  long stride, long col_off) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)s * n) return;
    int t = (int)(i / n), c = (int)(i % n);
    dst[(long)t * stride + col_off + c] = src[i];
}

extern "C" int memra_dsv4_place_cols(const float* src, float* dst, int s, int n, long stride,
                                     long col_off, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)s * n;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_place_cols_kernel<<<(unsigned)blocks, threads, 0, stream>>>(src, dst, s, n, stride,
                                                                     col_off);
    DSV4_ERR();
    return 0;
}

// hc expand (model.py:805): h[t, c, :] = e[t, :] for all hc copies c.
extern "C" __global__ void dsv4_repeat_hc_kernel(const float* __restrict__ e,
                                                 float* __restrict__ h, int s, int hc, int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)s * hc * d) return;
    int c = (int)(i % d);
    int t = (int)(i / ((long)hc * d));
    h[i] = e[(long)t * d + c];
}

extern "C" int memra_dsv4_repeat_hc(const float* e, float* h, int s, int hc, int d,
                                    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * hc * d;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_repeat_hc_kernel<<<(unsigned)blocks, threads, 0, stream>>>(e, h, s, hc, d);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- f32-island GEMM

// y[t, j] = dot(x[t, :], w[j, :]) with f64 partials + f64 fixed-tree block reduction —
// the f32-island GEMM (gate scoring, compressor wkv/wgate, hc mixes, head logits).
// w is f32 (w_is_bf16 == 0) or bf16 (== 1, decoded exactly in-kernel). Block per (j, t).
extern "C" __global__ void dsv4_dots_f32_kernel(const float* __restrict__ x,
                                                const void* __restrict__ w, int w_is_bf16,
                                                float* __restrict__ y, int s, int k, int n) {
    int j = blockIdx.x;
    int t = blockIdx.y;
    if (j >= n || t >= s) return;
    const float* xr = x + (long)t * k;
    double acc = 0.0;
    if (w_is_bf16) {
        const uint16_t* wr = (const uint16_t*)w + (long)j * k;
        for (int i = threadIdx.x; i < k; i += blockDim.x)
            acc += (double)xr[i] * (double)__uint_as_float(((unsigned)wr[i]) << 16);
    } else {
        const float* wr = (const float*)w + (long)j * k;
        for (int i = threadIdx.x; i < k; i += blockDim.x)
            acc += (double)xr[i] * (double)wr[i];
    }
    extern __shared__ double shd[];
    double tot = dsv4_block_sum(acc, shd);
    if (threadIdx.x == 0) y[(long)t * n + j] = (float)tot;
}

extern "C" int memra_dsv4_dots_f32(const float* x, const void* w, int w_is_bf16, float* y,
                                   int s, int k, int n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dim3 grid((unsigned)n, (unsigned)s);
    dsv4_dots_f32_kernel<<<grid, threads, threads * sizeof(double), stream>>>(x, w, w_is_bf16,
                                                                              y, s, k, n);
    DSV4_ERR();
    return 0;
}

// Lane-9 rung C — OWNER-GATED FORK (RECEIPTS.md "Lane 9", ruling 2026-08-19): the
// f32-accumulation SERVING arm for the island dots. The f64 kernel above stays the
// switchable oracle-truth arm (seam MEMRA_DSV4_DOTS_ARM, default f64); this arm is
// the reference's own numeric class and is gated by the fork discipline (decode-gate
// + teacher-forcing vs the CPU quantized oracle, in-band near-ties only).
// Deterministic fixed tree — the gated dsv4_gemv_bf16 accumulation class: thread t
// owns contiguous 8-element chunks at (t*8 + j*8*blockDim), sequential in-chunk and
// ascending across its chunks, then a blockDim-leaf halving tree. Vectorized loads
// (float4 / uint4); k % 8 == 0 enforced by the launcher. Block per (j, t).
extern "C" __global__ void dsv4_dots_f32acc_kernel(const float* __restrict__ x,
                                                   const void* __restrict__ w, int w_is_bf16,
                                                   float* __restrict__ y, int s, int k,
                                                   int n) {
    int j = blockIdx.x;
    int t = blockIdx.y;
    if (j >= n || t >= s) return;
    const float* xr = x + (long)t * k;
    float part = 0.0f;
    if (w_is_bf16) {
        const uint16_t* wr = (const uint16_t*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            uint4 wv = *(const uint4*)(wr + i0);
            float4 xa = *(const float4*)(xr + i0);
            float4 xb = *(const float4*)(xr + i0 + 4);
            unsigned ww[4] = {wv.x, wv.y, wv.z, wv.w};
            float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
            for (int q2 = 0; q2 < 4; q2++) {
                float w0 = __uint_as_float((ww[q2] & 0xFFFFu) << 16);
                float w1 = __uint_as_float(ww[q2] & 0xFFFF0000u);
                part += xs[2 * q2] * w0;
                part += xs[2 * q2 + 1] * w1;
            }
        }
    } else {
        const float* wr = (const float*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            float4 wa = *(const float4*)(wr + i0);
            float4 wb = *(const float4*)(wr + i0 + 4);
            float4 xa = *(const float4*)(xr + i0);
            float4 xb = *(const float4*)(xr + i0 + 4);
            part += xa.x * wa.x;
            part += xa.y * wa.y;
            part += xa.z * wa.z;
            part += xa.w * wa.w;
            part += xb.x * wb.x;
            part += xb.y * wb.y;
            part += xb.z * wb.z;
            part += xb.w * wb.w;
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) y[(long)t * n + j] = red[0];
}

extern "C" int memra_dsv4_dots_f32acc(const float* x, const void* w, int w_is_bf16, float* y,
                                      int s, int k, int n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40012;  // vector-chunk contract (all island k are 64-multiples)
    int threads = 128;
    dim3 grid((unsigned)n, (unsigned)s);
    dsv4_dots_f32acc_kernel<<<grid, threads, 0, stream>>>(x, w, w_is_bf16, y, s, k, n);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- cuBLASLt bf16 GEMM

namespace {
struct Bf16Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulAlgo_t algo;
};
std::mutex g_mu_dsv4;
cublasLtHandle_t g_lt_dsv4[16] = {};                                   // per-device handles
std::map<std::tuple<int, int, int, int>, Bf16Plan>* g_plans_dsv4 = nullptr;  // leaked
}  // namespace

// y[m,n] row-major f32 = x[m,k] bf16 @ W[n,k]^T bf16, CUBLAS_COMPUTE_32F.
// Col-major view: D[n,m] = A^T(W as k x n) * B(x as k x m). Mirrors memra_f16_pp_gemm_pre.
// `dev` selects the per-device Lt handle (two-card placement).
extern "C" int memra_dsv4_gemm_bf16(const void* w_bf16, const void* x_bf16, float* y_f32,
                                    int m, int n, int k, int dev, void* ws, size_t ws_bytes,
                                    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> guard(g_mu_dsv4);
    if (dev < 0 || dev >= 16) return 30001;
    if (!g_lt_dsv4[dev]) {
        cublasStatus_t s = cublasLtCreate(&g_lt_dsv4[dev]);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    }
    cublasLtHandle_t lt = g_lt_dsv4[dev];
    if (!g_plans_dsv4) g_plans_dsv4 = new std::map<std::tuple<int, int, int, int>, Bf16Plan>();
    auto key = std::make_tuple(dev, m, n, k);
    auto it = g_plans_dsv4->find(key);
    if (it == g_plans_dsv4->end()) {
        Bf16Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k);  // W: k x n col-major
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k);  // x: k x m col-major
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, n, m, n);   // y: n x m col-major
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur,
                                           &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_plans_dsv4->emplace(key, p).first;
    }
    const Bf16Plan& p = it->second;
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasLtMatmul(lt, p.op, &alpha, w_bf16, p.la, x_bf16, p.lb, &beta,
                                      y_f32, p.ld, y_f32, p.ld, &p.algo, ws, ws_bytes, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}

// ---------------------------------------------------------------- norms / rope / QAT

// RMSNorm rows, oracle arithmetic (dsv4_forward.rs rmsnorm): f64 sum of squares, mean cast
// to f32, +eps and sqrt in f32; y = w * (x * rsq). w == nullptr -> weightless (q head RMS
// uses the dedicated kernel below; this one is the [rows, ncols] w-weighted form).
extern "C" __global__ void dsv4_rmsnorm_kernel(const float* __restrict__ x,
                                               const float* __restrict__ w,
                                               float* __restrict__ dst, int ncols, float eps) {
    int row = blockIdx.x;
    const float* xr = x + (long)row * ncols;
    float* dr = dst + (long)row * ncols;
    double acc = 0.0;
    // lane-9 (BIT-EXACT): batch 8 strided loads into registers before the dependent
    // f64 chain consumes them — the i order per thread is UNCHANGED (i, i+B, i+2B, …),
    // so every square and every add is the same; only load latency gets overlapped
    // (rung-0: 13.1 µs/inst on a 1.8 µs f64-issue floor, single-block latency-exposed).
    int i = threadIdx.x;
    int B = blockDim.x;
    for (; i + 7 * B < ncols; i += 8 * B) {
        float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
        float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
        acc += (double)v0 * (double)v0;
        acc += (double)v1 * (double)v1;
        acc += (double)v2 * (double)v2;
        acc += (double)v3 * (double)v3;
        acc += (double)v4 * (double)v4;
        acc += (double)v5 * (double)v5;
        acc += (double)v6 * (double)v6;
        acc += (double)v7 * (double)v7;
    }
    for (; i < ncols; i += B) {
        double v = (double)xr[i];
        acc += v * v;
    }
    extern __shared__ double shd[];
    double tot = dsv4_block_sum(acc, shd);
    float mean = (float)(tot / (double)ncols);
    float rsq = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x)
        dr[i] = (w ? w[i] : 1.0f) * (xr[i] * rsq);
    // NOTE weightless form multiplies by 1.0f — identical to the oracle's `x * rsq`.
}

extern "C" int memra_dsv4_rmsnorm(const float* x, const float* w, float* dst, int rows,
                                  int ncols, float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_rmsnorm_kernel<<<(unsigned)rows, threads, threads * sizeof(double), stream>>>(
        x, w, dst, ncols, eps);
    DSV4_ERR();
    return 0;
}

// Interleaved complex rope on the LAST rd dims of each dim-wide vector
// (dsv4_forward.rs apply_rope / model.py:232-244). x is [n_pos, n_vec, dim];
// cs is [table_len, rd] as (cos, sin) pairs; positions[p] selects the row.
// inverse != 0 uses the conjugate (query-position de-rotation, model.py:534).
extern "C" __global__ void dsv4_rope_kernel(float* __restrict__ x, int n_pos, int n_vec,
                                            int dim, int rd, const float* __restrict__ cs,
                                            const int* __restrict__ positions, int inverse) {
    int half = rd / 2;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)n_pos * n_vec * half;
    if (i >= tot) return;
    int kk = (int)(i % half);
    long pv = i / half;
    int v = (int)(pv % n_vec);
    int p = (int)(pv / n_vec);
    const float* row = cs + (long)positions[p] * rd + 2 * kk;
    float c = row[0];
    float s0 = row[1];
    float s = inverse ? -s0 : s0;
    long base = ((long)p * n_vec + v) * dim + (dim - rd) + 2 * kk;
    float x0 = x[base], x1 = x[base + 1];
    x[base] = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

extern "C" int memra_dsv4_rope(float* x, int n_pos, int n_vec, int dim, int rd,
                               const float* cs, const int* positions, int inverse,
                               void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)n_pos * n_vec * (rd / 2);
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_rope_kernel<<<(unsigned)blocks, threads, 0, stream>>>(x, n_pos, n_vec, dim, rd, cs,
                                                               positions, inverse);
    DSV4_ERR();
    return 0;
}

// Weightless per-head RMS over the FULL head dim (model.py:498), in place.
// x viewed as [rows, d]; oracle: rsq = 1/sqrt((f32)(mean64) + eps), x *= rsq.
extern "C" __global__ void dsv4_headrms_kernel(float* __restrict__ x, int d, float eps) {
    int row = blockIdx.x;
    float* xr = x + (long)row * d;
    double acc = 0.0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        double v = (double)xr[i];
        acc += v * v;
    }
    extern __shared__ double shd[];
    double tot = dsv4_block_sum(acc, shd);
    float rsq = 1.0f / sqrtf((float)(tot / (double)d) + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) xr[i] *= rsq;
}

extern "C" int memra_dsv4_headrms(float* x, int rows, int d, float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_headrms_kernel<<<(unsigned)rows, threads, threads * sizeof(double), stream>>>(x, d,
                                                                                       eps);
    DSV4_ERR();
    return 0;
}

// act_quant QAT sim (kernel.py act_quant, dsv4_forward.rs act_quant): per contiguous
// `block` group inside x[r*stride .. r*stride+prefix_len): pow2-ceil scale of
// amax*(1/448) (amax floored 1e-4), clamp +-448, then either nothing more (clamp_only,
// the ARTIFACT contract) or an e4m3 RNE round-trip (ref law). One warp-block per group.
// Grid layout note (lane-6 fix): rows ride blockIdx.x (limit 2^31-1) and the group
// index rides blockIdx.y — the original (group, rows) layout hit the 65535 grid.y
// ceiling at rows = s·heads > 65535 (first seen at a 1024-token prefill: fp4 on the
// indexer q has rows = s·64). Pure index swap; per-group arithmetic unchanged.
extern "C" __global__ void dsv4_act_quant_kernel(float* __restrict__ x, long stride,
                                                 int block, int clamp_only) {
    int r = blockIdx.x;
    int g = blockIdx.y;   // group within the row prefix
    float* grp = x + (long)r * stride + (long)g * block;
    __shared__ float shf[128];
    float a = 0.0f;
    for (int i = threadIdx.x; i < block; i += blockDim.x) a = fmaxf(a, fabsf(grp[i]));
    float amax = dsv4_block_max(a, shf);
    amax = fmaxf(amax, 1e-4f);
    const float inv = (float)(1.0 / 448.0);
    float s = dsv4_pow2_ceil(amax * inv);
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        float q = fminf(fmaxf(grp[i] / s, -448.0f), 448.0f);
        if (!clamp_only) {
            // reference law: FP8-E4M3 RNE round-trip. Inputs are pre-clamped to +-448 so
            // SATFINITE saturation == the oracle's saturate-to-448; RNE ties-to-even
            // mantissa == ties-to-even code on the monotone e4m3 grid.
            __nv_fp8_storage_t c8 = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
            q = __half2float(__nv_cvt_fp8_to_halfraw(c8, __NV_E4M3));
        }
        grp[i] = q * s;
    }
}

extern "C" int memra_dsv4_act_quant(float* x, int rows, long stride, int prefix_len,
                                    int block, int clamp_only, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (prefix_len % block != 0) return 40001;
    dim3 grid((unsigned)rows, (unsigned)(prefix_len / block));
    dsv4_act_quant_kernel<<<grid, 64, 0, stream>>>(x, stride, block, clamp_only);
    DSV4_ERR();
    return 0;
}

// fp4_act_quant QAT sim (identical in both kernel variants): per-32 groups, pow2-ceil of
// amax*(1/6) (amax floored 6*2^-126), clamp +-6, e2m1 RNE round-trip.
extern "C" __global__ void dsv4_fp4_act_quant_kernel(float* __restrict__ x, long stride) {
    int r = blockIdx.x;   // rows on x (lane-6 grid fix, see act_quant note)
    int g = blockIdx.y;
    float* grp = x + (long)r * stride + (long)g * 32;
    __shared__ float shf[128];
    float a = 0.0f;
    for (int i = threadIdx.x; i < 32; i += blockDim.x) a = fmaxf(a, fabsf(grp[i]));
    float amax = dsv4_block_max(a, shf);
    float floorv = 6.0f * ldexpf(1.0f, -126);
    amax = fmaxf(amax, floorv);
    const float inv = (float)(1.0 / 6.0);
    float s = dsv4_pow2_ceil(amax * inv);
    for (int i = threadIdx.x; i < 32; i += blockDim.x)
        grp[i] = dsv4_e2m1_rne(fminf(fmaxf(grp[i] / s, -6.0f), 6.0f)) * s;
}

extern "C" int memra_dsv4_fp4_act_quant(float* x, int rows, long stride, int len,
                                        void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (len % 32 != 0) return 40002;
    dim3 grid((unsigned)rows, (unsigned)(len / 32));
    dsv4_fp4_act_quant_kernel<<<grid, 32, 0, stream>>>(x, stride);
    DSV4_ERR();
    return 0;
}

// Sylvester-order Walsh-Hadamard, in place per d-chunk. Same butterfly structure as the
// oracle's sequential loop (pairs disjoint per stage -> the parallel adds are the SAME
// adds: bit-exact). `scale` (= d^-0.5) is computed on the HOST with the oracle's own f32
// powf so no CUDA-vs-Rust libm ULP skew can enter. d power of two, <= 1024.
extern "C" __global__ void dsv4_hadamard_kernel(float* __restrict__ x, int d, float scale) {
    extern __shared__ float sh[];
    int row = blockIdx.x;
    float* xr = x + (long)row * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sh[i] = xr[i];
    __syncthreads();
    for (int h = 1; h < d; h *= 2) {
        // pair (i, i+h) for i in blocks of 2h; thread handles pair index p
        int npairs = d / 2;
        for (int p = threadIdx.x; p < npairs; p += blockDim.x) {
            int base = (p / h) * 2 * h;
            int i = base + (p % h);
            float a = sh[i], b = sh[i + h];
            sh[i] = a + b;
            sh[i + h] = a - b;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x) xr[i] = sh[i] * scale;
}

extern "C" int memra_dsv4_hadamard(float* x, int rows, int d, float scale, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (d > 1024 || (d & (d - 1)) != 0) return 40003;
    int threads = d / 2 < 128 ? d / 2 : 128;
    dsv4_hadamard_kernel<<<(unsigned)rows, threads, d * sizeof(float), stream>>>(x, d, scale);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- compressor / indexer

// Gated softmax pooling over ratio-blocks (model.py:279-377; oracle CompressorW::forward).
// kv/score: [s, latent] f32 (raw GEMM outputs; ape added HERE). out: [nb, d].
// overlap != 0 (fine r=4): position slots = prev block via dims [0:d] (block 0 -> -inf),
// current block via dims [d:2d]. One thread per (j, c), sequential f64 num/den like the
// oracle.
extern "C" __global__ void dsv4_compressor_pool_kernel(const float* __restrict__ kv,
                                                       const float* __restrict__ score,
                                                       const float* __restrict__ ape,
                                                       float* __restrict__ out, int nb,
                                                       int ratio, int d, int latent,
                                                       int overlap) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)nb * d) return;
    int j = (int)(i / d);
    int c = (int)(i % d);
    int positions = overlap ? 2 * ratio : ratio;
    float mx = -INFINITY;
    // pass 1: max of gated scores at this channel
    for (int p = 0; p < positions; p++) {
        float sc;
        if (overlap) {
            if (p < ratio) {
                sc = (j == 0) ? -INFINITY
                              : score[((long)(j - 1) * ratio + p) * latent + c] +
                                    ape[(long)p * latent + c];
            } else {
                int pp = p - ratio;
                sc = score[((long)j * ratio + pp) * latent + d + c] +
                     ape[(long)pp * latent + d + c];
            }
        } else {
            sc = score[((long)j * ratio + p) * latent + c] + ape[(long)p * latent + c];
        }
        mx = fmaxf(mx, sc);
    }
    double den = 0.0, num = 0.0;
    for (int p = 0; p < positions; p++) {
        float sc, kvv;
        if (overlap) {
            if (p < ratio) {
                if (j == 0) {
                    sc = -INFINITY;
                    kvv = 0.0f;
                } else {
                    sc = score[((long)(j - 1) * ratio + p) * latent + c] +
                         ape[(long)p * latent + c];
                    kvv = kv[((long)(j - 1) * ratio + p) * latent + c];
                }
            } else {
                int pp = p - ratio;
                sc = score[((long)j * ratio + pp) * latent + d + c] +
                     ape[(long)pp * latent + d + c];
                kvv = kv[((long)j * ratio + pp) * latent + d + c];
            }
        } else {
            sc = score[((long)j * ratio + p) * latent + c] + ape[(long)p * latent + c];
            kvv = kv[((long)j * ratio + p) * latent + c];
        }
        float e = expf(sc - mx);
        den += (double)e;
        num += (double)e * (double)kvv;
    }
    out[i] = (float)(num / den);
}

extern "C" int memra_dsv4_compressor_pool(const float* kv, const float* score,
                                          const float* ape, float* out, int nb, int ratio,
                                          int d, int latent, int overlap, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)nb * d;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_compressor_pool_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        kv, score, ape, out, nb, ratio, d, latent, overlap);
    DSV4_ERR();
    return 0;
}

// Indexer scoring (model.py:380-433; oracle IndexerW::forward): score[t, j] =
// sum_h relu(f32(dot64(q[t,h], ckv[j]))) * (w[t,h] * wscale), causal mask -inf where
// j >= lim. lim0 < 0: prefill law, lim = (t+1)/ratio with LOCAL t (model.py:425).
// lim0 >= 0: decode law — the local t is 0, causality lives in the store
// (nb = (pos+1)/ratio, model.py:415), so the caller passes lim0 = nb (lane 6).
//
// Lane-8 reparallelization (rung A, BIT-EXACT): one BLOCK per (t, j), one THREAD per
// head. The per-head dot stays ONE sequential f64 chain over x (the oracle expression
// verbatim) and the head sum stays thread-0 sequential in h order — every value and
// every accumulation order is unchanged; only the thread mapping moved. The original
// one-THREAD-per-(t,j) shape ran the decode step on ~60 threads total (rung-0 profile:
// 1.066 ms per fine layer, 22.4 ms/step — the #1 GPU consumer).
extern "C" __global__ void dsv4_indexer_score_kernel(const float* __restrict__ q,
                                                     const float* __restrict__ ckv,
                                                     const float* __restrict__ w,
                                                     float wscale, float* __restrict__ score,
                                                     int s, int heads, int hd, int nb,
                                                     int ratio, int lim0) {
    long i = blockIdx.x;
    if (i >= (long)s * nb) return;
    int t = (int)(i / nb);
    int j = (int)(i % nb);
    int lim = (lim0 >= 0) ? lim0 : (t + 1) / ratio;
    extern __shared__ double shh[];
    if (j >= lim) {
        if (threadIdx.x == 0) score[i] = -INFINITY;
        return;
    }
    int h = threadIdx.x;
    if (h < heads) {
        const float* qr = q + ((long)t * heads + h) * hd;
        const float* kr = ckv + (long)j * hd;
        double dacc = 0.0;
        for (int x = 0; x < hd; x++) dacc += (double)qr[x] * (double)kr[x];
        float sc = (float)dacc;
        float r = fmaxf(sc, 0.0f);
        float ws = w[(long)t * heads + h] * wscale;
        shh[h] = (double)(r * ws);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double acc = 0.0;
        for (int hh = 0; hh < heads; hh++) acc += shh[hh];  // oracle h order
        score[i] = (float)acc;
    }
}

extern "C" int memra_dsv4_indexer_score(const float* q, const float* ckv, const float* w,
                                        float wscale, float* score, int s, int heads, int hd,
                                        int nb, int ratio, int lim0, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * nb;
    if (n == 0) return 0;
    if (n > 2147483647L) return 40009;  // grid.x contract
    int threads = heads;  // one thread per head (64); heads <= 1024 asserted by shape
    if (threads > 1024) return 40009;
    dsv4_indexer_score_kernel<<<(unsigned)n, threads, (size_t)heads * sizeof(double), stream>>>(
        q, ckv, w, wscale, score, s, heads, hd, nb, ratio, lim0);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- sparse sink attention

// kernel.py sparse_attn / oracle AttnW::forward inner loop: per (t, h) online softmax over
// the gathered kv rows (idx -1 = masked), attn_sink contributes denominator mass only,
// K == V (shared latent). f32 scores, f64 sums (denominator summed in slot order by
// thread 0 — the oracle's order). Block per (t, h).
extern "C" __global__ void dsv4_sink_attn_kernel(const float* __restrict__ q,
                                                 const float* __restrict__ kv,
                                                 const int* __restrict__ idxs,
                                                 const float* __restrict__ sink,
                                                 float* __restrict__ o, int heads, int hd,
                                                 int slots, float scale) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    const int* ti = idxs + (long)t * slots;
    const float* qv = q + ((long)t * heads + h) * hd;
    extern __shared__ float shs[];       // scores[slots] then e[slots]
    float* scores = shs;
    float* evals = shs + slots;
    __shared__ float shred[128];
    __shared__ double shden;
    // scores (one thread per slot, sequential f64 dot each — oracle dot order)
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) {
        int ix = ti[sl];
        if (ix < 0) {
            scores[sl] = -INFINITY;
        } else {
            const float* kr = kv + (long)ix * hd;
            double acc = 0.0;
            for (int x = 0; x < hd; x++) acc += (double)qv[x] * (double)kr[x];
            scores[sl] = (float)acc * scale;
        }
    }
    __syncthreads();
    // max
    float m = -INFINITY;
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) m = fmaxf(m, scores[sl]);
    m = dsv4_block_max(m, shred);
    m = fmaxf(m, -1e30f);
    // e values + denominator (thread 0, slot order — matches the oracle's sequential sum)
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x)
        evals[sl] = (ti[sl] < 0) ? 0.0f : expf(scores[sl] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        double den = 0.0;
        for (int sl = 0; sl < slots; sl++)
            if (ti[sl] >= 0) den += (double)evals[sl];
        den += (double)expf(sink[h] - m);
        shden = den;
    }
    __syncthreads();
    double den = shden;
    // accumulate output dims (each thread owns dims, slots sequential in slot order)
    float* orow = o + ((long)t * heads + h) * hd;
    for (int x = threadIdx.x; x < hd; x += blockDim.x) {
        double acc = 0.0;
        for (int sl = 0; sl < slots; sl++) {
            int ix = ti[sl];
            if (ix < 0) continue;
            acc += (double)evals[sl] * (double)kv[(long)ix * hd + x];
        }
        orow[x] = (float)(acc / den);
    }
}

extern "C" int memra_dsv4_sink_attn(const float* q, const float* kv, const int* idxs,
                                    const float* sink, float* o, int s, int heads, int hd,
                                    int slots, float scale, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    dim3 grid((unsigned)s, (unsigned)heads);
    size_t smem = (size_t)slots * 2 * sizeof(float);
    dsv4_sink_attn_kernel<<<grid, 128, smem, stream>>>(q, kv, idxs, sink, o, heads, hd, slots,
                                                       scale);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- hyper-connections

// scale mixes[t, :] by rsqrt(mean(x[t, :]^2) + eps) — the hc_pre normalization applied to
// the MIXING COEFFS only (model.py:673-681). x: [s, w], mixes: [s, rows]. Block per t.
extern "C" __global__ void dsv4_rowsq_scale_kernel(const float* __restrict__ x,
                                                   float* __restrict__ mixes, int w, int rows,
                                                   float eps) {
    int t = blockIdx.x;
    const float* xr = x + (long)t * w;
    double acc = 0.0;
    // lane-9 (BIT-EXACT): same 8-wide load batching as dsv4_rmsnorm_kernel — per-thread
    // i order unchanged (rung-0: 30.5 µs/inst on a ~7 µs single-SM f64 floor).
    int i = threadIdx.x;
    int B = blockDim.x;
    for (; i + 7 * B < w; i += 8 * B) {
        float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
        float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
        acc += (double)v0 * (double)v0;
        acc += (double)v1 * (double)v1;
        acc += (double)v2 * (double)v2;
        acc += (double)v3 * (double)v3;
        acc += (double)v4 * (double)v4;
        acc += (double)v5 * (double)v5;
        acc += (double)v6 * (double)v6;
        acc += (double)v7 * (double)v7;
    }
    for (; i < w; i += B) {
        double v = (double)xr[i];
        acc += v * v;
    }
    extern __shared__ double shd[];
    double tot = dsv4_block_sum(acc, shd);
    float rsq = 1.0f / sqrtf((float)(tot / (double)w) + eps);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) mixes[(long)t * rows + i] *= rsq;
}

extern "C" int memra_dsv4_rowsq_scale(const float* x, float* mixes, int s, int w, int rows,
                                      float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_rowsq_scale_kernel<<<(unsigned)s, threads, threads * sizeof(double), stream>>>(
        x, mixes, w, rows, eps);
    DSV4_ERR();
    return 0;
}

// hc_pre collapse: y[t, :] = sum_c pre[t, c] * x[t, c, :] (copy order c ascending, FMA-free
// f32 adds like the oracle).
// grid.y ceiling (hermes finding, fixed 2026-08-23): CUDA grid.y maxes at 65535, and
// these two hc launchers put token counts (hc_collapse: s; hc_post: s*hc — which crosses
// the ceiling at a 16384-token prefill with hc=4) on grid.y. Launches are CHUNKED over a
// y0 base offset instead: pure index offset, bit-identical per element, any s.
#define DSV4_GRID_Y_MAX 65535

extern "C" __global__ void dsv4_hc_collapse_kernel(const float* __restrict__ x,
                                                   const float* __restrict__ pre,
                                                   float* __restrict__ y, int hc, int d,
                                                   int y0) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int t = y0 + blockIdx.y;
    if (i >= d) return;
    float acc = 0.0f;
    for (int c = 0; c < hc; c++)
        acc += pre[(long)t * hc + c] * x[((long)t * hc + c) * d + i];
    y[(long)t * d + i] = acc;
}

extern "C" int memra_dsv4_hc_collapse(const float* x, const float* pre, float* y, int s,
                                      int hc, int d, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    for (int y0 = 0; y0 < s; y0 += DSV4_GRID_Y_MAX) {
        int chunk = s - y0 < DSV4_GRID_Y_MAX ? s - y0 : DSV4_GRID_Y_MAX;
        dim3 grid((unsigned)((d + threads - 1) / threads), (unsigned)chunk);
        dsv4_hc_collapse_kernel<<<grid, threads, 0, stream>>>(x, pre, y, hc, d, y0);
        DSV4_ERR();
    }
    return 0;
}

// hc_post: out[t, k, :] = post[t, k] * f[t, :] + sum_j comb[t, j, k] * residual[t, j, :]
// (j ascending — oracle order).
extern "C" __global__ void dsv4_hc_post_kernel(const float* __restrict__ f,
                                               const float* __restrict__ residual,
                                               const float* __restrict__ post,
                                               const float* __restrict__ comb,
                                               float* __restrict__ out, int hc, int d,
                                               int y0) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int tk = y0 + blockIdx.y;
    if (i >= d) return;
    int t = tk / hc, k = tk % hc;
    float acc = post[(long)t * hc + k] * f[(long)t * d + i];
    for (int j = 0; j < hc; j++)
        acc += comb[((long)t * hc + j) * hc + k] * residual[((long)t * hc + j) * d + i];
    out[((long)t * hc + k) * d + i] = acc;
}

extern "C" int memra_dsv4_hc_post(const float* f, const float* residual, const float* post,
                                  const float* comb, float* out, int s, int hc, int d,
                                  void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    // s*hc rows on grid.y: 16384 tokens x hc=4 = 65536 crossed the 65535 ceiling — the
    // 16k-prefill crash. Chunked over y0 (bit-identical: pure index offset).
    long total = (long)s * hc;
    for (long y0 = 0; y0 < total; y0 += DSV4_GRID_Y_MAX) {
        long chunk = total - y0 < DSV4_GRID_Y_MAX ? total - y0 : DSV4_GRID_Y_MAX;
        dim3 grid((unsigned)((d + threads - 1) / threads), (unsigned)chunk);
        dsv4_hc_post_kernel<<<grid, threads, 0, stream>>>(f, residual, post, comb, out, hc, d,
                                                          (int)y0);
        DSV4_ERR();
    }
    return 0;
}

// ---------------------------------------------------------------- lane 7: native arms

// Reference law (artifact inference/kernel.py, cited in RECEIPTS.md "Lane 7"):
// act_quant (K:40-125, non-inplace GEMM path): per-128 group along K, amax floored
// 1e-4, s = 2^ceil(log2(amax/448)) (round_scale — pow2, K:36-37), codes =
// e4m3_RNE(clamp(x/s, ±448)) — REAL FP8 rounding in BOTH kernel variants (the
// clamp-only artifact fork at kernel.py:88 applies only to the inplace KV-QAT sims).
// Emits codes + f32 scales (E8M0 storage in the reference is exact for pow2 — no-op).
// Grid: rows on grid.x (lane-6 ceiling lesson), K/128 groups on grid.y.
extern "C" __global__ void dsv4_act_quant_fp8_kernel(const float* __restrict__ x,
                                                     uint8_t* __restrict__ codes,
                                                     float* __restrict__ scales, int kdim) {
    int r = blockIdx.x;
    int g = blockIdx.y;
    const float* grp = x + (long)r * kdim + (long)g * 128;
    uint8_t* out = codes + (long)r * kdim + (long)g * 128;
    __shared__ float shf[128];
    float a = 0.0f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) a = fmaxf(a, fabsf(grp[i]));
    float amax = dsv4_block_max(a, shf);
    amax = fmaxf(amax, 1e-4f);
    const float inv = (float)(1.0 / 448.0);
    float s = dsv4_pow2_ceil(amax * inv);
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        float q = fminf(fmaxf(grp[i] / s, -448.0f), 448.0f);
        uint8_t c = (uint8_t)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        // zero-sign canonicalization (lane-7 gate-a run-2 finding, banked): a tiny
        // negative that RNE-rounds to zero yields IEEE -0.0 = 0x80 here, while the
        // house CPU encoder canonicalizes zero to +0 = 0x00. Decoded values are
        // identical (±0.0) — canonicalize so codes are bit-stable across the pair.
        if ((c & 0x7F) == 0) c = 0;
        out[i] = c;
    }
    if (threadIdx.x == 0) scales[(long)r * (kdim / 128) + g] = s;
}

extern "C" int memra_dsv4_act_quant_fp8(const float* x, void* codes, float* scales, int rows,
                                        int kdim, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (kdim % 128 != 0) return 40004;
    dim3 grid((unsigned)rows, (unsigned)(kdim / 128));
    dsv4_act_quant_fp8_kernel<<<grid, 64, 0, stream>>>(x, (uint8_t*)codes, scales, kdim);
    DSV4_ERR();
    return 0;
}

// Lossless FP8-QAT -> half transport for grouped tensor-core prefill. The scale
// is a power of two, NOT the generic gather's arbitrary row amax. Every element
// is round-tripped; the caller refuses any row that cannot be represented.
extern "C" __global__ void dsv4_fp8_gather_half_kernel(
    const uint8_t* __restrict__ codes, const float* __restrict__ scales,
    const int* __restrict__ row_ids, __half* __restrict__ out,
    float* __restrict__ row_scale, int* __restrict__ row_status, int cols) {
    const int row = blockIdx.x, src = row_ids ? row_ids[row] : row;
    const int groups = cols / 128;
    __shared__ float red[256];
    __shared__ int bad;
    if (threadIdx.x == 0) bad = 0;
    float local = 0.0f;
    for (int g = threadIdx.x; g < groups; g += blockDim.x)
        local = fmaxf(local, scales[(long)src * groups + g]);
    float max_scale = dsv4_block_max(local, red);
    // 448*128 = 57344, below half's finite limit; smaller groups retain 22
    // exponent bits of room before a nonzero value may underflow.
    float rs = ldexpf(max_scale, -7);
    if (!(rs > 0.0f) || !isfinite(rs)) atomicOr(&bad, 1);
    for (int x = threadIdx.x; x < cols; x += blockDim.x) {
        uint8_t code = codes[(long)src * cols + x];
        float v = dsv4_e4m3(code) * scales[(long)src * groups + x / 128];
        __half h = __float2half_rn(v / rs);
        float back = __half2float(h) * rs;
        if ((code & 127) == 127 || !isfinite(v) || __float_as_uint(back) != __float_as_uint(v))
            atomicOr(&bad, 1);
        out[(long)row * cols + x] = h;
    }
    __syncthreads();
    if (threadIdx.x == 0) { row_scale[row] = rs; row_status[row] = bad; }
}

extern "C" int memra_dsv4_fp8_gather_half(
    const void* codes, const float* scales, const int* row_ids, void* out,
    float* row_scale, int* row_status, int rows, int cols, void* stream_v) {
    if (rows < 1 || cols < 128 || cols % 128 != 0) return 40004;
    dsv4_fp8_gather_half_kernel<<<rows, 256, 0, (cudaStream_t)stream_v>>>(
        (const uint8_t*)codes, scales, row_ids, (__half*)out, row_scale, row_status, cols);
    DSV4_ERR();
    return 0;
}

extern "C" __global__ void dsv4_scale_rows_kernel(float* y, const float* scale,
                                                 int rows, int cols) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (long)rows * cols) y[i] *= scale[i / cols];
}

extern "C" int memra_dsv4_scale_rows(float* y, const float* scale, int rows, int cols,
                                     void* stream_v) {
    if (rows < 1 || cols < 1) return 40004;
    long n = (long)rows * cols;
    dsv4_scale_rows_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream_v>>>(
        y, scale, rows, cols);
    DSV4_ERR();
    return 0;
}

// Native quantized expert GEMM: out[g, n] f32 = A_fp8[g, K] @ W_fp4[n, K]^T, the
// kernel.py fp4_gemm arithmetic (K:441-536) on the artifact's as-stored slabs:
//   kind 0 (NVFP4 trunk): per-16 groups, group scale = e4m3(sc[j]) * scale_2
//   kind 1 (MXFP4 MTP):   per-32 groups, group scale = 2^(sc[j] - 127) (0xFF refused at load)
// Every decoded product is EXACT in f32 (e4m3 <=4 sig bits x e2m1 <=2); group scales x
// act scale are exact pow2 products for this artifact (lane-1 measured law + act_quant
// pow2 by construction). fp4_gemm applies scales per 32-K sub-block; this arm applies
// them per WEIGHT GROUP (16/32) — identical scaled-product set, different f32 summation
// grouping only (banked deviation, same lawful-reorder class as the tilelang tile).
// Deterministic + CPU-mirrorable: thread t owns groups j ≡ t (mod blockDim), sequential
// ascending; per-group inner sum sequential f32; partials reduced by a fixed halving
// tree. Block per (n, g) output element on (grid.x, grid.y).
//
// Lane-8 rung A (BIT-EXACT vectorization): the per-byte global loads + divergent
// __constant__ LUT + per-element e4m3 exp2f of the original body ran ~8x off weight
// bandwidth (rung-0 profile: 20.6 µs per GEMM, 15.9 ms/step). The rewrite loads
// weights/activations with 8/16-byte vectors and decodes through per-block SHARED
// tables whose entries equal dsv4_e4m3()/DSV4_E2M1 exactly — identical products,
// identical i order inside each group, identical group ownership and halving tree.

// shared decode tables: e4m3_tab[256] then pair_tab[256] (lo,hi e2m1 of a weight byte).
__device__ __forceinline__ void dsv4_fp4_tables(float* e4m3_tab, float2* pair_tab) {
    for (int c = threadIdx.x; c < 256; c += blockDim.x) {
        e4m3_tab[c] = dsv4_e4m3((uint8_t)c);
        pair_tab[c] = make_float2(DSV4_E2M1[c & 0x0F], DSV4_E2M1[(c >> 4) & 0x0F]);
    }
    __syncthreads();
}

// one weight group's inner sum, ascending i order (kk even = LOW nibble) — the original
// sequential expression with vector loads. k0 is 16/32-aligned so the 8B/16B loads are
// aligned by construction (slab strides are multiples of 8/16 bytes).
__device__ __forceinline__ float dsv4_fp4_group_sub(const uint8_t* wrow,
                                                    const uint8_t* arow, int k0, int gs,
                                                    const float* e4m3_tab,
                                                    const float2* pair_tab) {
    float sub = 0.0f;
    int chunks = gs / 16;  // 1 (nvfp4) or 2 (mxfp4): 8 weight bytes + 16 act bytes each
    for (int c = 0; c < chunks; c++) {
        uint2 wv = *(const uint2*)(wrow + (k0 >> 1) + c * 8);
        uint4 av = *(const uint4*)(arow + k0 + c * 16);
        unsigned ww[2] = {wv.x, wv.y};
        unsigned aw[4] = {av.x, av.y, av.z, av.w};
        for (int b = 0; b < 8; b++) {
            unsigned byte = (ww[b >> 2] >> ((b & 3) * 8)) & 0xFF;
            float2 wp = pair_tab[byte];
            unsigned a0 = (aw[(2 * b) >> 2] >> (((2 * b) & 3) * 8)) & 0xFF;
            unsigned a1 = (aw[(2 * b + 1) >> 2] >> (((2 * b + 1) & 3) * 8)) & 0xFF;
            sub += e4m3_tab[a0] * wp.x;  // i = 2b   (low nibble)
            sub += e4m3_tab[a1] * wp.y;  // i = 2b+1 (high nibble)
        }
    }
    return sub;
}

// dynamic smem layout for both fp4 GEMM kernels: e4m3_tab[256] | pair_tab[256] | red[threads]
#define DSV4_FP4_SMEM(threads) ((256 + 512 + (threads)) * sizeof(float))

// ---- lane-9 ALU decoders: bit-constructions of the SAME values as dsv4_e4m3() and
// DSV4_E2M1[] — exhaustive 256+16 device equality probe banked (RECEIPTS.md "Lane 9").
// They exist so the hot indirect-GEMM loop decodes in integer ALU instead of
// data-dependent (conflict-serialized) shared-memory lookups.
__device__ __forceinline__ float dsv4_e4m3_alu(unsigned x) {
    unsigned mag = x & 0x7Fu;
    if (mag == 0x7Fu) return 0.0f;  // modelopt NaN code -> +0.0 (sign dropped)
    unsigned exp = mag >> 3, man = mag & 7u;
    float v = exp ? __uint_as_float(((exp + 120u) << 23) | (man << 20))
                  : (float)man * 0x1p-9f;
    return (x & 0x80u) ? -v : v;
}

__device__ __forceinline__ float dsv4_e2m1_alu(unsigned nib) {
    unsigned mag = nib & 7u;
    unsigned bits = (mag >= 2u) ? (((126u + (mag >> 1)) << 23) | ((mag & 1u) << 22))
                                : (mag ? 0x3F000000u : 0u);
    return __uint_as_float(bits | ((nib & 8u) << 28));
}

// group inner sum with ALU decode — value-identical to dsv4_fp4_group_sub (same vector
// loads, same ascending i order, same adds; only the decode source changes from the
// smem tables to the proven-equal bit constructions).
// MEASURED REGRESSION (lane-9, banked): swapping this into dsv4_fp4_gemm_sel_kernel
// moved it 37.8 -> 45.9 µs/inst — the smem lookups were NOT the binding resource
// (activation codes cluster, so the "random" reads broadcast well); the ALU decode
// only added integer-issue pressure. Kept as the negative-result reference; the sel
// kernel uses the tables.
__device__ __forceinline__ float dsv4_fp4_group_sub_alu(const uint8_t* wrow,
                                                        const uint8_t* arow, int k0, int gs) {
    float sub = 0.0f;
    int chunks = gs / 16;
    for (int c = 0; c < chunks; c++) {
        uint2 wv = *(const uint2*)(wrow + (k0 >> 1) + c * 8);
        uint4 av = *(const uint4*)(arow + k0 + c * 16);
        unsigned ww[2] = {wv.x, wv.y};
        unsigned aw[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned byte = (ww[b >> 2] >> ((b & 3) * 8)) & 0xFF;
            unsigned a0 = (aw[(2 * b) >> 2] >> (((2 * b) & 3) * 8)) & 0xFF;
            unsigned a1 = (aw[(2 * b + 1) >> 2] >> (((2 * b + 1) & 3) * 8)) & 0xFF;
            sub += dsv4_e4m3_alu(a0) * dsv4_e2m1_alu(byte & 0x0Fu);       // i = 2b
            sub += dsv4_e4m3_alu(a1) * dsv4_e2m1_alu((byte >> 4) & 0x0Fu); // i = 2b+1
        }
    }
    return sub;
}

extern "C" __global__ void dsv4_fp4_gemm_kernel(const uint8_t* __restrict__ a,
                                                const float* __restrict__ as_,
                                                const uint8_t* __restrict__ w,
                                                const uint8_t* __restrict__ wsc, float scale2,
                                                int kind, float* __restrict__ out, int n,
                                                int kdim) {
    int col = blockIdx.x; // output feature (row of W)
    int row = blockIdx.y; // activation row
    int gs = (kind == 0) ? 16 : 32;
    int ngroups = kdim / gs;
    extern __shared__ float shg[];
    float* e4m3_tab = shg;
    float2* pair_tab = (float2*)(shg + 256);
    float* red = shg + 256 + 512;
    dsv4_fp4_tables(e4m3_tab, pair_tab);
    const uint8_t* arow = a + (long)row * kdim;
    const float* asrow = as_ + (long)row * (kdim / 128);
    const uint8_t* wrow = w + (long)col * (kdim / 2);
    const uint8_t* srow = wsc + (long)col * ngroups;
    float part = 0.0f;
    for (int j = threadIdx.x; j < ngroups; j += blockDim.x) {
        int k0 = j * gs;
        float sub = dsv4_fp4_group_sub(wrow, arow, k0, gs, e4m3_tab, pair_tab);
        float ws = (kind == 0) ? e4m3_tab[srow[j]] * scale2
                               : ((srow[j] == 0xFF) ? nanf("") : exp2f((float)srow[j] - 127.0f));
        float sc = ws * asrow[k0 / 128];
        part += sub * sc;
    }
    int tid = threadIdx.x;
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) out[(long)row * n + col] = red[0];
}

extern "C" int memra_dsv4_fp4_gemm(const void* a_codes, const float* a_scales, const void* w,
                                   const void* wsc, float scale2, int kind, float* out, int g,
                                   int n, int kdim, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (kdim % 128 != 0 || (kind != 0 && kind != 1)) return 40005;
    if (g > 65535 || n > 2147483647) return 40006; // grid-dim contract (lane-6 lesson)
    dim3 grid((unsigned)n, (unsigned)g);
    int threads = 128;
    dsv4_fp4_gemm_kernel<<<grid, threads, DSV4_FP4_SMEM(threads), stream>>>(
        (const uint8_t*)a_codes, a_scales, (const uint8_t*)w, (const uint8_t*)wsc, scale2,
        kind, out, n, kdim);
    DSV4_ERR();
    return 0;
}

// byte-row gather: out[r, :] = x[idx[r], :] (row_bytes wide) — expert token-group
// assembly for fp8 codes and (as raw bytes) their f32 scale rows; per-row quantization
// commutes with gathering exactly.
extern "C" __global__ void dsv4_gather_rows_u8_kernel(const uint8_t* __restrict__ x,
                                                      const int* __restrict__ idx,
                                                      uint8_t* __restrict__ out, int g,
                                                      long row_bytes) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)g * row_bytes) return;
    long r = i / row_bytes;
    long c = i % row_bytes;
    out[i] = x[(long)idx[r] * row_bytes + c];
}

extern "C" int memra_dsv4_gather_rows_u8(const void* x, const int* idx, void* out, int g,
                                         long row_bytes, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)g * row_bytes;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_gather_rows_u8_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const uint8_t*)x, idx, (uint8_t*)out, g, row_bytes);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------- expert activation

// SwiGLU with swiglu_limit clamps (model.py:596-606; oracle expert_forward): up two-sided
// clamp, gate ONE-sided (min only), silu(g)*u, then optional per-row routing weight.
extern "C" __global__ void dsv4_swiglu_kernel(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              float* __restrict__ dst, int inter, float limit,
                                              const float* __restrict__ wrow, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = fminf(fmaxf(up[i], -limit), limit);
    float g = fminf(gate[i], limit);
    float h = g * dsv4_sigmoid(g) * u;
    if (wrow) h *= wrow[i / inter];
    dst[i] = h;
}

extern "C" int memra_dsv4_swiglu(const float* gate, const float* up, float* dst, int rows,
                                 int inter, float limit, const float* wrow, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)rows * inter;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_swiglu_kernel<<<(unsigned)blocks, threads, 0, stream>>>(gate, up, dst, inter, limit,
                                                                 wrow, n);
    DSV4_ERR();
    return 0;
}

// ================================================================ lane 8 — device-resident
// decode step (RECEIPTS.md "Lane 8", plan of record banked before this section).
// Everything below serves the s=1 decode path ONLY; prefill stays on the lane-4 kernels.
// Realization notes (banked): device expf/log1pf differ from host libm at the ulp level —
// the Sinkhorn and router kernels are declared REALIZATION FORKS gated at class bounds;
// the top-k and combine kernels are integer-exact / order-preserving and must be
// BIT-EXACT vs the host path.

// rope at ONE scalar position (same arithmetic as dsv4_rope_kernel; cs row = pos).
extern "C" __global__ void dsv4_rope_at_kernel(float* __restrict__ x, int n_vec, int dim,
                                               int rd, const float* __restrict__ cs, int pos,
                                               int inverse) {
    int half = rd / 2;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)n_vec * half;
    if (i >= tot) return;
    int kk = (int)(i % half);
    int v = (int)(i / half);
    const float* row = cs + (long)pos * rd + 2 * kk;
    float c = row[0];
    float s0 = row[1];
    float s = inverse ? -s0 : s0;
    long base = (long)v * dim + (dim - rd) + 2 * kk;
    float x0 = x[base], x1 = x[base + 1];
    x[base] = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

extern "C" int memra_dsv4_rope_at(float* x, int n_vec, int dim, int rd, const float* cs,
                                  int pos, int inverse, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)n_vec * (rd / 2);
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_rope_at_kernel<<<(unsigned)blocks, threads, 0, stream>>>(x, n_vec, dim, rd, cs, pos,
                                                                  inverse);
    DSV4_ERR();
    return 0;
}

// hc_split_sinkhorn at s=1 — the host function's SEQUENTIAL loop order verbatim on one
// thread (hc=4: 24 mixes, 20 iterations — trivially cheap; the win is killing the
// D2H sync + host compute + 3 H2D per sub-block). expf/log fork banked.
// Parallel layout (v2, bit-identical values to the single-thread v1): thread c owns
// pre/post[c]; thread j owns row j of every row op (max/exp/sum/div sequential in k —
// the host per-row order); thread k owns column k of every col op (sequential in j).
// Rows and columns are independent in the host loops, so this is the SAME arithmetic.
extern "C" __global__ void dsv4_hc_sinkhorn_kernel(const float* __restrict__ mixes,
                                                   const float* __restrict__ scale,
                                                   const float* __restrict__ base,
                                                   float* __restrict__ pre,
                                                   float* __restrict__ post,
                                                   float* __restrict__ comb, int hc, int iters,
                                                   float eps) {
    int t = threadIdx.x;
    if (t < hc) {
        pre[t] = dsv4_sigmoid(mixes[t] * scale[0] + base[t]) + eps;
        post[t] = 2.0f * dsv4_sigmoid(mixes[hc + t] * scale[1] + base[hc + t]);
    }
    for (int i = t; i < hc * hc; i += blockDim.x)
        comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
    __syncthreads();
    // row softmax + eps (per-row order preserved; rows independent)
    if (t < hc) {
        float* row = comb + t * hc;
        float mx = -INFINITY;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, row[k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) {
            row[k] = expf(row[k] - mx);
            sum += row[k];
        }
        for (int k = 0; k < hc; k++) row[k] = row[k] / sum + eps;
    }
    __syncthreads();
    // col_norm, then (iters-1) x (row_norm, col_norm) — host closure order
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            if (t < hc) {
                float sum = 0.0f;
                for (int k = 0; k < hc; k++) sum += comb[t * hc + k];
                for (int k = 0; k < hc; k++) comb[t * hc + k] /= sum + eps;
            }
            __syncthreads();
        }
        if (t < hc) {
            float sum = 0.0f;
            for (int j = 0; j < hc; j++) sum += comb[j * hc + t];
            for (int j = 0; j < hc; j++) comb[j * hc + t] /= sum + eps;
        }
        __syncthreads();
    }
}

extern "C" int memra_dsv4_hc_sinkhorn(const float* mixes, const float* scale,
                                      const float* base, float* pre, float* post, float* comb,
                                      int hc, int iters, float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    dsv4_hc_sinkhorn_kernel<<<1, 32, 0, stream>>>(mixes, scale, base, pre, post, comb, hc,
                                                  iters, eps);
    DSV4_ERR();
    return 0;
}

// head hc gate: pre[c] = sigmoid(m[c]*scale[0] + base[c]) + eps (head_logits_row host loop).
extern "C" __global__ void dsv4_hc_head_pre_kernel(const float* __restrict__ mixes,
                                                   const float* __restrict__ scale,
                                                   const float* __restrict__ base,
                                                   float* __restrict__ pre, int hc, float eps) {
    int c = threadIdx.x;
    if (c >= hc) return;
    pre[c] = dsv4_sigmoid(mixes[c] * scale[0] + base[c]) + eps;
}

extern "C" int memra_dsv4_hc_head_pre(const float* mixes, const float* scale,
                                      const float* base, float* pre, int hc, float eps,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    dsv4_hc_head_pre_kernel<<<1, 32, 0, stream>>>(mixes, scale, base, pre, hc, eps);
    DSV4_ERR();
    return 0;
}

// torch softplus (beta=1, threshold=20) — dsv4_gpu.rs softplus_f32 twin.
__device__ __forceinline__ float dsv4_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

// MoE routing at s=1 — route_host verbatim on one thread: scores = sqrt(softplus(raw)),
// selection = tid2eid row (hash layers) or top-k of biased scores by repeated argmax
// with strict > (== the host full-sort take-k under value-desc/index-asc), weights =
// score/sum*route_scale. Emits sel (selection order), selw, and order = slot ids
// sorted by ascending expert id (the legacy scatter_add accumulation order).
extern "C" __global__ void dsv4_route_kernel(const float* __restrict__ raw,
                                             const float* __restrict__ bias,
                                             const int* __restrict__ tid2eid,
                                             const int* __restrict__ tok, int ne, int topk,
                                             float route_scale, int* __restrict__ sel,
                                             float* __restrict__ selw,
                                             int* __restrict__ order) {
    // v2 (bit-identical to the single-thread v1): the per-element transcendentals
    // (softplus/sqrt — the expensive part) are computed ONCE in parallel; the
    // selection walks precomputed values on one thread (pure compares).
    __shared__ float sc[256];  // sqrt(softplus(raw[c]))
    __shared__ float bs[256];  // biased scores (score-routed layers)
    int t = threadIdx.x;
    for (int c = t; c < ne; c += blockDim.x) {
        float s = sqrtf(dsv4_softplus(raw[c]));
        sc[c] = s;
        bs[c] = (bias != nullptr) ? s + bias[c] : 0.0f;
    }
    __syncthreads();
    if (tid2eid == nullptr) {
        // v3 (lane 9, BIT-EXACT): the repeated argmax runs as a PARALLEL (value, index)
        // tree with the host tie rule (strict >, ties keep the lowest index) —
        // max-with-lowest-index is associative (the dsv4_argmax argument), so every
        // round selects the SAME id as the v2 single-thread scan and the mask evolves
        // identically. Values are only compared, never modified.
        __shared__ float rv[128];
        __shared__ int ri[128];
        __shared__ unsigned long long mask[4];
        if (t < 4) mask[t] = 0ull;
        __syncthreads();
        for (int k = 0; k < topk; k++) {
            float bv = 0.0f;
            int bi = -1;
            for (int c = t; c < ne; c += blockDim.x) {
                if (mask[c >> 6] & (1ull << (c & 63))) continue;
                float v = bs[c];
                // c ascends per thread, so strict > keeps the lowest index in-thread
                if (bi < 0 || v > bv) {
                    bv = v;
                    bi = c;
                }
            }
            rv[t] = bv;
            ri[t] = bi;
            __syncthreads();
            for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (t < off) {
                    bool take = (ri[t + off] >= 0) &&
                                (ri[t] < 0 || rv[t + off] > rv[t] ||
                                 (rv[t + off] == rv[t] && ri[t + off] < ri[t]));
                    if (take) {
                        rv[t] = rv[t + off];
                        ri[t] = ri[t + off];
                    }
                }
                __syncthreads();
            }
            if (t == 0) {
                sel[k] = ri[0];
                mask[ri[0] >> 6] |= (1ull << (ri[0] & 63));
            }
            __syncthreads();
        }
    }
    if (t != 0) return;
    if (tid2eid != nullptr) {
        const int* row = tid2eid + (long)tok[0] * topk;
        for (int k = 0; k < topk; k++) sel[k] = row[k];
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        float w = sc[sel[k]];
        selw[k] = w;
        sum += w;
    }
    for (int k = 0; k < topk; k++) selw[k] = selw[k] / sum * route_scale;
    // order = slot indices sorted by expert id ascending (selection ids are distinct)
    for (int k = 0; k < topk; k++) order[k] = k;
    for (int a = 1; a < topk; a++) {
        int o = order[a];
        int b = a - 1;
        while (b >= 0 && sel[order[b]] > sel[o]) {
            order[b + 1] = order[b];
            b--;
        }
        order[b + 1] = o;
    }
}

extern "C" int memra_dsv4_route(const float* raw, const float* bias, const int* tid2eid,
                                const int* tok, int ne, int topk, float route_scale, int* sel,
                                float* selw, int* order, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (ne > 256 || topk > 32) return 40007;
    dsv4_route_kernel<<<1, 128, 0, stream>>>(raw, bias, tid2eid, tok, ne, topk, route_scale,
                                             sel, selw, order);
    DSV4_ERR();
    return 0;
}

// Indirect multi-expert fp4 GEMM: dsv4_fp4_gemm_kernel's PER-OUTPUT arithmetic verbatim
// (same thread-strided group ownership, same sequential in-group order, same halving
// tree), expert id read from sel[blockIdx.y] on device — one launch covers the whole
// active-expert set of a projection (lane-8 fused dispatch; bit-exact vs the per-expert
// launches by construction). a_stride_rows = 0 (shared x codes) or 1 (per-slot h rows).
// DSV4 selected-expert output columns owned by one CTA. Per-column arithmetic is
// independent and unchanged; widening this only amortizes activation/table setup.
// Eight columns was exact-gated on the target pair but rejected: 75.48 vs 75.65 s
// long-prefill is noise while short DSpark regressed 41.66 -> 40.71 tok/s.
#define DSV4_FP4_SEL_CPB 4

// Lane-9 rung 2 (BIT-EXACT multi-column blocking): one block owns consecutive
// output features of one slot — the lane-8 one-col blocks did 2 KB of weight work
// per 768-float smem-table init + 7-sync tree (rung-0 lane-9 profile: 37.2 µs/inst,
// 4.79 ms/step vs the ~2.6 ms weight-bandwidth floor). Per column NOTHING moves:
// thread t owns groups j ≡ t (mod blockDim) sequential ascending, the in-group order
// is dsv4_fp4_group_sub verbatim, the expression part += sub * (ws * as) is
// unchanged, and each column runs the SAME 128-leaf halving tree (sequentially, one
// per column). The activation vector is reused across the CPB columns and the
// table init is amortized CPB-fold.
template <bool WarpReduce>
__global__ void dsv4_fp4_gemm_sel_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ as_,
    const uint8_t* __restrict__ w_base, const uint8_t* __restrict__ sc_base,
    const float* __restrict__ s2, const int* __restrict__ sel, int proj, int a_stride_rows,
    int kind, float* __restrict__ out, int n, int kdim, long wstride, long sstride,
    int a_group) {
    const int CPB = DSV4_FP4_SEL_CPB;
    int col0 = blockIdx.x * CPB;   // first output feature of this block
    int slot = blockIdx.y;         // active-expert slot
    int eid = sel[slot];
    int gs = (kind == 0) ? 16 : 32;
    int ngroups = kdim / gs;
    extern __shared__ float selected_smem[];
    float* e4m3_tab = selected_smem;
    float2* pair_tab = (float2*)(selected_smem + 256);
    float* red = selected_smem + 256 + 512;
    dsv4_fp4_tables(e4m3_tab, pair_tab);
    // iteration-3 batched verify: a_group > 0 means the slot set covers a_group slots per
    // ACTIVATION row (T positions x topk experts), so slot / a_group is the position whose
    // activation this slot consumes. a_group == 0 keeps the pinned single-position
    // semantics EXACTLY (0 -> shared row 0, 1 -> per-slot row) — same expression, same
    // value, so the gated bytes of every existing call are unchanged.
    long arow_i = (a_group > 0) ? (long)(slot / a_group) : (a_stride_rows ? (long)slot : 0L);
    const uint8_t* arow = a + arow_i * kdim;
    const float* asrow = as_ + arow_i * (kdim / 128);
    const uint8_t* wb = w_base + ((long)eid * 3 + proj) * wstride;
    const uint8_t* sb = sc_base + ((long)eid * 3 + proj) * sstride;
    float scale2 = (kind == 0) ? s2[eid * 3 + proj] : 0.0f;
    float part[CPB] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = threadIdx.x; j < ngroups; j += blockDim.x) {
        int k0 = j * gs;
        float as = asrow[k0 / 128];
#pragma unroll
        for (int c = 0; c < CPB; c++) {
            int col = col0 + c;
            if (col >= n) break;
            const uint8_t* wrow = wb + (long)col * (kdim / 2);
            const uint8_t* srow = sb + (long)col * ngroups;
            float sub = dsv4_fp4_group_sub(wrow, arow, k0, gs, e4m3_tab, pair_tab);
            float ws = (kind == 0)
                           ? e4m3_tab[srow[j]] * scale2
                           : ((srow[j] == 0xFF) ? nanf("") : exp2f((float)srow[j] - 127.0f));
            float sc = ws * as;
            part[c] += sub * sc;
        }
    }
    int tid = threadIdx.x;
    if constexpr (WarpReduce) {
        // The original tree first pairs t with t+64, then t with t+32,
        // then halves the remaining 32 lanes. Reducing each warp first
        // would reassociate the sum. Stage all four columns once, then
        // reproduce those SAME seven additions in the first warp.
#pragma unroll
        for (int c = 0; c < CPB; ++c) red[c * 128 + tid] = part[c];
        __syncthreads();
        if (tid < 32) {
#pragma unroll
            for (int c = 0; c < CPB; ++c) {
                float* r = red + c * 128;
                float v = (r[tid] + r[tid + 64]) + (r[tid + 32] + r[tid + 96]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const float other = __shfl_down_sync(0xffffffffu, v, off);
                    if (tid < off) v += other;
                }
                if (tid == 0 && col0 + c < n) out[(long)slot * n + col0 + c] = v;
            }
        }
    } else {
        for (int c = 0; c < CPB; c++) {
            int col = col0 + c;
            if (col >= n) break;
            __syncthreads();  // red free from the previous column's tree
            red[tid] = part[c];
            __syncthreads();
            for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (tid < off) red[tid] += red[tid + off];
                __syncthreads();
            }
            if (tid == 0) out[(long)slot * n + col] = red[0];
        }
    }
}

// Default off while the exact RTX PRO 6000 serving/performance gates are pending.
// Invalid values fail at the launcher, including when the first call is captured.
static int dsv4_fp4_reduce_arm() {
    static const int arm = [] {
        const char* v = std::getenv("MEMRA_DSV4_FP4_REDUCE");
        if (!v || !v[0] || std::strcmp(v, "block") == 0) return 0;
        if (std::strcmp(v, "warp") == 0) return 1;
        return -1;
    }();
    return arm;
}

// The native Rust model passes its own arm, allowing an isolated one-load gate
// to alternate arms without changing the process environment or global state.
extern "C" int memra_dsv4_fp4_gemm_sel_g_arm(const void* a_codes, const float* a_scales,
                                           const void* w_base, const void* sc_base,
                                           const float* s2, const int* sel, int proj,
                                           int a_stride_rows, int kind, float* out, int slots,
                                           int n, int kdim, long wstride, long sstride,
                                           int a_group, int reduce_arm, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (kdim % 128 != 0 || (kind != 0 && kind != 1)) return 40005;
    if (slots > 65535 || n > 2147483647) return 40006;
    if (a_group < 0) return 40006;
    if (reduce_arm != 0 && reduce_arm != 1) return 40021;
    dim3 grid((unsigned)((n + DSV4_FP4_SEL_CPB - 1) / DSV4_FP4_SEL_CPB),
              (unsigned)slots);
    int threads = 128;
    if (reduce_arm) {
        dsv4_fp4_gemm_sel_kernel<true><<<grid, threads, DSV4_FP4_SMEM(threads * DSV4_FP4_SEL_CPB), stream>>>(
            (const uint8_t*)a_codes, a_scales, (const uint8_t*)w_base, (const uint8_t*)sc_base,
            s2, sel, proj, a_stride_rows, kind, out, n, kdim, wstride, sstride, a_group);
    } else {
        dsv4_fp4_gemm_sel_kernel<false><<<grid, threads, DSV4_FP4_SMEM(threads), stream>>>(
            (const uint8_t*)a_codes, a_scales, (const uint8_t*)w_base, (const uint8_t*)sc_base,
            s2, sel, proj, a_stride_rows, kind, out, n, kdim, wstride, sstride, a_group);
    }
    DSV4_ERR();
    return 0;
}

// Preserve the original C interfaces and their process-env behavior for existing
// callers. Both share the explicit launcher, so launch geometry cannot drift.
extern "C" int memra_dsv4_fp4_gemm_sel(const void* a_codes, const float* a_scales,
                                       const void* w_base, const void* sc_base,
                                       const float* s2, const int* sel, int proj,
                                       int a_stride_rows, int kind, float* out, int slots,
                                       int n, int kdim, long wstride, long sstride,
                                       void* stream_v) {
    return memra_dsv4_fp4_gemm_sel_g_arm(a_codes, a_scales, w_base, sc_base,
        s2, sel, proj, a_stride_rows, kind, out, slots, n, kdim, wstride, sstride,
        0, dsv4_fp4_reduce_arm(), stream_v);
}

// Batched-verify twin: `a_group` slots share one activation row (T positions x topk).
// Same kernel, same grid geometry per slot, same per-output accumulation order — the
// only change is WHICH activation row a slot reads, so every slot's dot is bit-identical
// to the single-position launch that would have computed it.
extern "C" int memra_dsv4_fp4_gemm_sel_g(const void* a_codes, const float* a_scales,
                                         const void* w_base, const void* sc_base,
                                         const float* s2, const int* sel, int proj,
                                         int a_stride_rows, int kind, float* out, int slots,
                                         int n, int kdim, long wstride, long sstride,
                                         int a_group, void* stream_v) {
    return memra_dsv4_fp4_gemm_sel_g_arm(a_codes, a_scales, w_base, sc_base,
        s2, sel, proj, a_stride_rows, kind, out, slots, n, kdim, wstride, sstride,
        a_group, dsv4_fp4_reduce_arm(), stream_v);
}

// Routed-expert combine: y[i] = sum over slots in ascending-expert-id order (acc starts
// at 0.0f — the legacy zeroed-y + sequential scatter_add sum order, bit-exact).
extern "C" __global__ void dsv4_combine_rows_kernel(const float* __restrict__ contrib,
                                                    const int* __restrict__ order, int topk,
                                                    float* __restrict__ y, long d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float acc = 0.0f;
    for (int k = 0; k < topk; k++) acc += contrib[(long)order[k] * d + i];
    y[i] = acc;
}

extern "C" int memra_dsv4_combine_rows(const float* contrib, const int* order, int topk,
                                       float* y, long d, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    long blocks = (d + threads - 1) / threads;
    dsv4_combine_rows_kernel<<<(unsigned)blocks, threads, 0, stream>>>(contrib, order, topk, y,
                                                                       d);
    DSV4_ERR();
    return 0;
}

// Decode index list, window part (+ coarse part when nb >= 0): the block_decode host
// builder verbatim — pos >= win-1: ring slots [sp+1..win) then [0..sp]; else [0..pos]
// with -1 pads. Coarse layers append blocks 0..nb-1 at +win. Fine layers stop at win
// (the top-k kernel fills [win, win+kk)).
extern "C" __global__ void dsv4_build_idx_kernel(int* __restrict__ idx, int pos, int win,
                                                 int nb, int cap) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= cap) return;
    if (k < win) {
        if (pos >= win - 1) {
            int sp = pos % win;
            int head = win - 1 - sp;  // count of slots [sp+1, win)
            idx[k] = (k < head) ? (sp + 1 + k) : (k - head);
        } else {
            idx[k] = (k <= pos) ? k : -1;
        }
    } else {
        int c = k - win;
        idx[k] = (nb >= 0 && c < nb) ? (win + c) : -1;
    }
}

extern "C" int memra_dsv4_build_idx(int* idx, int pos, int win, int nb, int cap,
                                    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    int blocks = (cap + threads - 1) / threads;
    dsv4_build_idx_kernel<<<blocks, threads, 0, stream>>>(idx, pos, win, nb, cap);
    DSV4_ERR();
    return 0;
}

// Fine-layer top-k block selection on device: keys = (~orderable(score) << 32) | index,
// single-block bitonic sort ascending == host sort (value desc, index asc) EXACTLY
// (integer comparator on distinct keys — no float arithmetic, so the selection and its
// order are bit-identical to the host path). Pads (0xFFFF...) sort last. Writes
// idx_out[t] = index_t + win for t < kk. Capacity contract: nb <= 4096 (s <= ~16k at
// ratio 4); the long-context indexer rewrite is a different lane.
extern "C" __global__ void dsv4_topk_idx_kernel(const float* __restrict__ score, int nb,
                                                int kk, int win, int* __restrict__ idx_out,
                                                int npow2) {
    extern __shared__ unsigned long long keys[];
    int tid = threadIdx.x;
    for (int i = tid; i < npow2; i += blockDim.x) {
        unsigned long long key = 0xFFFFFFFFFFFFFFFFull;
        if (i < nb) {
            unsigned b = __float_as_uint(score[i]);
            unsigned ord = b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
            key = (((unsigned long long)(~ord)) << 32) | (unsigned)i;
        }
        keys[i] = key;
    }
    __syncthreads();
    for (int k = 2; k <= npow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < npow2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    unsigned long long a = keys[i], b = keys[ixj];
                    if ((a > b) == up) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int t = tid; t < kk; t += blockDim.x)
        idx_out[t] = (int)(keys[t] & 0xFFFFFFFFull) + win;
}

extern "C" int memra_dsv4_topk_idx(const float* score, int nb, int kk, int win, int* idx_out,
                                   void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (nb <= 0 || nb > 4096 || kk > nb) return 40008;
    int npow2 = 1;
    while (npow2 < nb) npow2 <<= 1;
    size_t smem = (size_t)npow2 * sizeof(unsigned long long);
    dsv4_topk_idx_kernel<<<1, 512, smem, stream>>>(score, nb, kk, win, idx_out, npow2);
    DSV4_ERR();
    return 0;
}

// Batched exact twin for chunked prefill: one independent bitonic block per query
// row. Scores past that row's causal block limit are already -inf; sorting the common
// nb_max therefore preserves the same top-k keys and value-desc/index-asc order as
// launching dsv4_topk_idx_kernel with the shorter per-row nb.
extern "C" __global__ void dsv4_topk_idx_m_kernel(
        const float* __restrict__ score, int nb, int topk, int win,
        int* __restrict__ idx_out, int idx_stride, int pos0, int ratio, int npow2) {
    extern __shared__ unsigned long long keys[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* srow = score + (long)row * nb;
    int lim = min(nb, (pos0 + row + 1) / ratio);
    int kk = min(topk, lim);
    for (int i = tid; i < npow2; i += blockDim.x) {
        unsigned long long key = 0xFFFFFFFFFFFFFFFFull;
        if (i < nb) {
            unsigned b = __float_as_uint(srow[i]);
            unsigned ord = b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
            key = (((unsigned long long)(~ord)) << 32) | (unsigned)i;
        }
        keys[i] = key;
    }
    __syncthreads();
    for (int k = 2; k <= npow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < npow2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    unsigned long long a = keys[i], b = keys[ixj];
                    if ((a > b) == up) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    int* out = idx_out + (long)row * idx_stride + win;
    for (int i = tid; i < kk; i += blockDim.x)
        out[i] = (int)(keys[i] & 0xFFFFFFFFull) + win;
}

extern "C" int memra_dsv4_topk_idx_m(const float* score, int s, int nb, int topk,
                                      int win, int* idx_out, int idx_stride, int pos0,
                                      int ratio, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s <= 0 || nb <= 0 || nb > 4096 || topk <= 0 || ratio <= 0 ||
        idx_stride < win + min(topk, nb)) return 40008;
    int npow2 = 1;
    while (npow2 < nb) npow2 <<= 1;
    size_t smem = (size_t)npow2 * sizeof(unsigned long long);
    dsv4_topk_idx_m_kernel<<<(unsigned)s, 512, smem, stream>>>(
        score, nb, topk, win, idx_out, idx_stride, pos0, ratio, npow2);
    DSV4_ERR();
    return 0;
}

// Hierarchical exact top-k for the 1M indexer. Each stage bitonic-sorts independent
// 4096-key partitions and retains 512 keys. Repeating until <=4096 is lossless for a
// global top-512: anything outside a partition's top-512 cannot enter the global
// top-512. Keys carry the ORIGINAL index, so value-desc/index-asc ordering is identical
// to dsv4_topk_idx_kernel across every merge level.
constexpr int DSV4_TOPK_STREAM_CHUNK = 4096;
constexpr int DSV4_TOPK_STREAM_KEEP = 512;

extern "C" __global__ void dsv4_topk_stream_scores_kernel(
        const float* __restrict__ score, int nb, unsigned long long* __restrict__ out,
        int out_stride) {
    extern __shared__ unsigned long long keys[];
    int part = blockIdx.x;
    int row = blockIdx.y;
    int base = part * DSV4_TOPK_STREAM_CHUNK;
    int tid = threadIdx.x;
    const float* srow = score + (long)row * nb;
    for (int i = tid; i < DSV4_TOPK_STREAM_CHUNK; i += blockDim.x) {
        int j = base + i;
        unsigned long long key = 0xFFFFFFFFFFFFFFFFull;
        if (j < nb) {
            unsigned b = __float_as_uint(srow[j]);
            unsigned ord = b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
            key = (((unsigned long long)(~ord)) << 32) | (unsigned)j;
        }
        keys[i] = key;
    }
    __syncthreads();
    for (int k = 2; k <= DSV4_TOPK_STREAM_CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < DSV4_TOPK_STREAM_CHUNK; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    unsigned long long a = keys[i], b = keys[ixj];
                    if ((a > b) == up) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    unsigned long long* dst = out + (long)row * out_stride
                                    + (long)part * DSV4_TOPK_STREAM_KEEP;
    for (int i = tid; i < DSV4_TOPK_STREAM_KEEP; i += blockDim.x) dst[i] = keys[i];
}

extern "C" __global__ void dsv4_topk_stream_keys_kernel(
        const unsigned long long* __restrict__ in, int nkeys, int in_stride,
        unsigned long long* __restrict__ out, int out_stride) {
    extern __shared__ unsigned long long keys[];
    int part = blockIdx.x;
    int row = blockIdx.y;
    int base = part * DSV4_TOPK_STREAM_CHUNK;
    int tid = threadIdx.x;
    const unsigned long long* src = in + (long)row * in_stride;
    for (int i = tid; i < DSV4_TOPK_STREAM_CHUNK; i += blockDim.x) {
        int j = base + i;
        keys[i] = (j < nkeys) ? src[j] : 0xFFFFFFFFFFFFFFFFull;
    }
    __syncthreads();
    for (int k = 2; k <= DSV4_TOPK_STREAM_CHUNK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < DSV4_TOPK_STREAM_CHUNK; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    unsigned long long a = keys[i], b = keys[ixj];
                    if ((a > b) == up) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    unsigned long long* dst = out + (long)row * out_stride
                                    + (long)part * DSV4_TOPK_STREAM_KEEP;
    for (int i = tid; i < DSV4_TOPK_STREAM_KEEP; i += blockDim.x) dst[i] = keys[i];
}

extern "C" __global__ void dsv4_topk_stream_final_kernel(
        const unsigned long long* __restrict__ in, int nkeys, int in_stride,
        int topk, int win, int* __restrict__ idx_out, int idx_stride, int npow2) {
    extern __shared__ unsigned long long keys[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const unsigned long long* src = in + (long)row * in_stride;
    for (int i = tid; i < npow2; i += blockDim.x)
        keys[i] = (i < nkeys) ? src[i] : 0xFFFFFFFFFFFFFFFFull;
    __syncthreads();
    for (int k = 2; k <= npow2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < npow2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    unsigned long long a = keys[i], b = keys[ixj];
                    if ((a > b) == up) {
                        keys[i] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    int* dst = idx_out + (long)row * idx_stride + win;
    for (int i = tid; i < topk; i += blockDim.x)
        dst[i] = (int)(keys[i] & 0xFFFFFFFFull) + win;
}

extern "C" int memra_dsv4_topk_idx_stream_m(
        const float* score, int s, int nb, int topk, int win, int* idx_out,
        int idx_stride, unsigned long long* work_a, unsigned long long* work_b,
        int work_stride, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s <= 0 || nb <= 4096 || topk <= 0 || topk > DSV4_TOPK_STREAM_KEEP ||
        idx_stride < win + topk) return 40008;
    int parts = (nb + DSV4_TOPK_STREAM_CHUNK - 1) / DSV4_TOPK_STREAM_CHUNK;
    int nkeys = parts * DSV4_TOPK_STREAM_KEEP;
    if (work_stride < nkeys) return 40008;
    constexpr size_t stage_smem = DSV4_TOPK_STREAM_CHUNK * sizeof(unsigned long long);
    dsv4_topk_stream_scores_kernel<<<dim3((unsigned)parts, (unsigned)s), 512,
                                        stage_smem, stream>>>(
        score, nb, work_a, work_stride);
    DSV4_ERR();
    unsigned long long* cur = work_a;
    unsigned long long* next = work_b;
    while (nkeys > DSV4_TOPK_STREAM_CHUNK) {
        parts = (nkeys + DSV4_TOPK_STREAM_CHUNK - 1) / DSV4_TOPK_STREAM_CHUNK;
        dsv4_topk_stream_keys_kernel<<<dim3((unsigned)parts, (unsigned)s), 512,
                                      stage_smem, stream>>>(
            cur, nkeys, work_stride, next, work_stride);
        DSV4_ERR();
        nkeys = parts * DSV4_TOPK_STREAM_KEEP;
        unsigned long long* tmp = cur;
        cur = next;
        next = tmp;
    }
    int npow2 = 1;
    while (npow2 < nkeys) npow2 <<= 1;
    dsv4_topk_stream_final_kernel<<<(unsigned)s, 512,
                                    (size_t)npow2 * sizeof(unsigned long long), stream>>>(
        cur, nkeys, work_stride, topk, win, idx_out, idx_stride, npow2);
    DSV4_ERR();
    return 0;
}

// ---- lane-8 sink attention, decode shape (s=1), split into three kernels so the
// slot dots / per-head softmax / output accumulation each get real parallelism
// (the monolithic kernel ran 64 blocks re-gathering every kv row per head: 214 µs
// per layer, 9.2 ms/step at rung 0). BIT-EXACT vs dsv4_sink_attn_kernel:
//   - per-(slot,head) score = the same single sequential f64 dot over hd (from an
//     smem copy of the row — identical values), * scale in f32;
//   - per-head max via fmax (exact, order-free), evals = expf(score - m) elementwise
//     (pads carry score -inf => evals +0.0, the legacy explicit 0.0f);
//   - denominator = thread-0 sequential over slots in slot order (pads add +0.0 —
//     bit-inert) + the sink term, kept in f64;
//   - output per (head, dim) = the same sequential f64 chain over slots in slot
//     order, slots with evals == 0.0 skipped exactly as the legacy ix < 0 skip
//     (an underflowed eval contributes a ±0.0 product — bit-inert either way).

// K1: scores[h * slots + sl] = f32(dot64(q_h, kv[ix_sl])) * scale, -inf for pads.
// Block per slot; kv row staged to smem once; one thread per head.
//
// Lane-9 verdict (both takes banked, then REVERTED to this lane-8 shape): ncu shows
// this kernel at 0.2% DRAM / 8.3% SM — pure f64 dependency latency (~346 cy/element).
// Take 2 (float4 q vectors) measured IDENTICAL (67.9 µs/inst); take 3 (4 slots per
// block, hoisted products, 4 interleaved chains) measured 3.1× WORSE (211.9 µs/inst —
// the wider f64 register set defeats nvcc's scheduling). The bit-exact f64 chain at
// one thread per (slot, head) is measured-best at ~67.9 µs/inst; further movement on
// this kernel needs the f32-island ruling extended to attention dots (owner call
// flagged in the lane-9 report).
extern "C" __global__ void dsv4_sink_scores_kernel(const float* __restrict__ q,
                                                   const float* __restrict__ kv,
                                                   const int* __restrict__ idxs,
                                                   float* __restrict__ scores, int heads,
                                                   int hd, int slots, float scale) {
    int sl = blockIdx.x;
    if (sl >= slots) return;
    int ix = idxs[sl];
    extern __shared__ float kvs[];
    if (ix < 0) {
        for (int h = threadIdx.x; h < heads; h += blockDim.x)
            scores[(long)h * slots + sl] = -INFINITY;
        return;
    }
    for (int x = threadIdx.x; x < hd; x += blockDim.x) kvs[x] = kv[(long)ix * hd + x];
    __syncthreads();
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const float* qv = q + (long)h * hd;
        double acc = 0.0;
        for (int x = 0; x < hd; x++) acc += (double)qv[x] * (double)kvs[x];
        scores[(long)h * slots + sl] = (float)acc * scale;
    }
}

// K2: per head — m, evals, f64 denominator (+ sink term). Block per head.
extern "C" __global__ void dsv4_sink_soft_kernel(const float* __restrict__ scores,
                                                 const float* __restrict__ sink,
                                                 float* __restrict__ evals,
                                                 double* __restrict__ den, int slots) {
    int h = blockIdx.x;
    const float* srow = scores + (long)h * slots;
    float* erow = evals + (long)h * slots;
    __shared__ float shred[128];
    float m = -INFINITY;
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) m = fmaxf(m, srow[sl]);
    m = dsv4_block_max(m, shred);
    m = fmaxf(m, -1e30f);
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x)
        erow[sl] = (srow[sl] == -INFINITY) ? 0.0f : expf(srow[sl] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        double d = 0.0;
        for (int sl = 0; sl < slots; sl++) d += (double)erow[sl];  // pads add +0.0
        d += (double)expf(sink[h] - m);
        den[h] = d;
    }
}

// K3: o[h, x] = f64 sum over slots (slot order) of evals * kv, / den. Block per
// (x-chunk, head-chunk): threads (h, x) — kv columns staged per slot-tile in slot
// order; each (h, x) keeps ONE sequential f64 chain (the legacy expression).
//
// Lane-9 rung 1 (BIT-EXACT reshape): head-chunks of 8 instead of 64 — the lane-8
// grid put 72 blocks on 188 SMs (rung-0 lane-9 profile: 73.5 µs/inst, 3.16 ms/step);
// this one launches 8× the blocks (64 threads each), everything else — tile staging
// order, per-(h, x) sequential slot chain, ev == 0.0 skip — unchanged.
extern "C" __global__ void dsv4_sink_out_kernel(const float* __restrict__ kv,
                                                const int* __restrict__ idxs,
                                                const float* __restrict__ evals,
                                                const double* __restrict__ den,
                                                float* __restrict__ o, int heads, int hd,
                                                int slots) {
    // grid.x: x-chunks of 8 dims; grid.y: head-chunks of 8
    const int XC = 8, HC = 8;
    int x0 = blockIdx.x * XC;
    int h0 = blockIdx.y * HC;
    int tx = threadIdx.x % XC;   // dim within chunk
    int th = threadIdx.x / XC;   // head within chunk
    int x = x0 + tx;
    int h = h0 + th;
    __shared__ float kvt[32 * XC];  // slot-tile x dim-chunk
    double acc = 0.0;
    for (int t0 = 0; t0 < slots; t0 += 32) {
        int tl = min(32, slots - t0);
        // stage kv[ix[t0..t0+tl], x0..x0+XC] (threads cooperate; pads stage 0)
        for (int i = threadIdx.x; i < tl * XC; i += blockDim.x) {
            int sl = t0 + i / XC;
            int xx = x0 + i % XC;
            int ix = idxs[sl];
            kvt[i] = (ix < 0 || xx >= hd) ? 0.0f : kv[(long)ix * hd + xx];
        }
        __syncthreads();
        if (x < hd && h < heads) {
            const float* erow = evals + (long)h * slots;
            for (int i = 0; i < tl; i++) {
                float ev = erow[t0 + i];
                if (ev == 0.0f) continue;  // pads + underflows (legacy ix<0 skip class)
                acc += (double)ev * (double)kvt[i * XC + tx];
            }
        }
        __syncthreads();
    }
    if (x < hd && h < heads) o[(long)h * hd + x] = (float)(acc / den[h]);
}

extern "C" int memra_dsv4_sink_attn_dec(const float* q, const float* kv, const int* idxs,
                                        const float* sink, float* scores, float* evals,
                                        double* den, float* o, int heads, int hd, int slots,
                                        float scale, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (slots <= 0) return 40010;
    dsv4_sink_scores_kernel<<<(unsigned)slots, 64, (size_t)hd * sizeof(float), stream>>>(
        q, kv, idxs, scores, heads, hd, slots, scale);
    DSV4_ERR();
    dsv4_sink_soft_kernel<<<(unsigned)heads, 128, 0, stream>>>(scores, sink, evals, den,
                                                               slots);
    DSV4_ERR();
    dim3 grid((unsigned)((hd + 7) / 8), (unsigned)((heads + 7) / 8));
    dsv4_sink_out_kernel<<<grid, 64, 0, stream>>>(kv, idxs, evals, den, o, heads, hd, slots);
    DSV4_ERR();
    return 0;
}

// Deterministic bf16 GEMV (lane-8, m=1 decode): y[n] f32 = W[n,k] bf16 @ x[k] bf16.
// Replaces the cuBLASLt m=1 GEMMs on the device decode path ONLY (rung-0/2 profile:
// nvjet+splitK ~10 ms/step, ~2.3x off weight bandwidth at these shapes). This is a
// CLASS-II realization fork (f32 accumulation order differs from the cuBLASLt plans,
// exactly the lane-6 legitimate-reorder class) — decode-gate + oracle adjudicated.
// Fixed accumulation: thread t owns contiguous 8-element chunks at (t*8 + j*8*blockDim),
// sequential in-chunk and across its chunks, then a 128-wide halving tree. bf16
// products are computed in f32 (bf16 x bf16 exact to f32 product rounding — the same
// per-product precision class as CUBLAS_COMPUTE_32F).
// (lane-9 note, banked: a two-rows-per-block variant measured 13.44 vs 13.04 µs/inst
// — reverted; the one-row shape is measured-best despite ncu's 48% DRAM on the
// wo_a class. GEMV physics on this card class, consistent with lane-8's cuBLASLt
// contradiction.)
extern "C" __global__ void dsv4_gemv_bf16_kernel(const uint16_t* __restrict__ w,
                                                 const uint16_t* __restrict__ x,
                                                 float* __restrict__ y, int n, int k) {
    int row = blockIdx.x;
    if (row >= n) return;
    const uint16_t* wr = w + (long)row * k;
    float part = 0.0f;
    for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
        uint4 wv = *(const uint4*)(wr + i0);
        uint4 xv = *(const uint4*)(x + i0);
        unsigned ww[4] = {wv.x, wv.y, wv.z, wv.w};
        unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
        for (int j = 0; j < 4; j++) {
            float w0 = __uint_as_float((ww[j] & 0xFFFFu) << 16);
            float w1 = __uint_as_float(ww[j] & 0xFFFF0000u);
            float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
            float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
            part += w0 * x0;
            part += w1 * x1;
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) y[row] = red[0];
}

extern "C" int memra_dsv4_gemv_bf16(const void* w_bf16, const void* x_bf16, float* y, int n,
                                    int k, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40011;  // uint4 vector contract (all dsv4 k are 64-multiples)
    dsv4_gemv_bf16_kernel<<<(unsigned)n, 128, 0, stream>>>((const uint16_t*)w_bf16,
                                                           (const uint16_t*)x_bf16, y, n, k);
    DSV4_ERR();
    return 0;
}

// ---- iteration-5 FP8 dense arm (MEMRA_DSV4_DENSE_ARM=fp8): the FP8-blk linears read
// AS-STORED (e4m3 codes [n, k] + host-decoded f32 block scales [ceil(n/128), sc_cols],
// sc_cols = ceil(k/128), scales exact pow2 from the e8m0 grid) instead of the load-time
// bf16 dequant slab — halving the dense weight traffic that is 79.9% of a decode step's
// bytes. BIT-IDENTITY law, not a numerics fork: e4m3(code) * scale is EXACT in f32 (pow2
// scale, <=4 significand bits — the same exactness the loader's f32_to_bf16_exact
// refusal proves for the bf16 slab), and the accumulation below is
// dsv4_gemv_bf16_kernel's VERBATIM (8-element chunks per thread, sequential in-chunk
// pairs, 128-leaf halving tree). A chunk of 8 never straddles a 128-wide scale block
// (8 | 128), so one scale serves each chunk. Body duplicated, not shared — the section
// law: gated arms' generated code cannot move.
extern "C" __global__ void dsv4_gemv_fp8_kernel(const uint8_t* __restrict__ w,
                                                const float* __restrict__ sc, int sc_cols,
                                                const uint16_t* __restrict__ x,
                                                float* __restrict__ y, int n, int k) {
    int row = blockIdx.x;
    if (row >= n) return;
    // e4m3 decode via smem LUT (the expert kernels' own pattern): table values ARE
    // dsv4_e4m3's — a transport of the decode, not a numeric change. The arithmetic
    // decode costs exp2f + a divide per element and measured ~7% where halved bytes
    // priced ~30% (box4 A/B alt-1); the LUT restores the byte-bound rate bit-inertly.
    __shared__ float e4m3_tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4m3_tab[i] = dsv4_e4m3((uint8_t)i);
    __syncthreads();
    const uint8_t* wr = w + (long)row * k;
    const float* srow = sc + (long)(row >> 7) * sc_cols;
    float part = 0.0f;
    // Unroll-by-2 with EARLY weight loads: both chunks' loads issue before either
    // chunk's arithmetic (memory-level parallelism for the halved-width uint2 reads),
    // then the chunks accumulate IN THE ORIGINAL ORDER — the per-thread += sequence
    // into `part` is the single-chunk loop's verbatim, so the change is load
    // scheduling only and the result is bit-identical.
    int stride = blockDim.x * 8;
    int i0 = threadIdx.x * 8;
    for (; i0 + stride < k; i0 += 2 * stride) {
        int i1 = i0 + stride;
        uint2 wva = *(const uint2*)(wr + i0);
        uint2 wvb = *(const uint2*)(wr + i1);
        uint4 xva = *(const uint4*)(x + i0);
        uint4 xvb = *(const uint4*)(x + i1);
        float sa = srow[i0 >> 7];
        float sb = srow[i1 >> 7];
        unsigned wba[2] = {wva.x, wva.y};
        unsigned wbb[2] = {wvb.x, wvb.y};
        unsigned xwa[4] = {xva.x, xva.y, xva.z, xva.w};
        unsigned xwb[4] = {xvb.x, xvb.y, xvb.z, xvb.w};
        for (int j = 0; j < 4; j++) {
            float w0 = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sa;
            float w1 = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sa;
            float x0 = __uint_as_float((xwa[j] & 0xFFFFu) << 16);
            float x1 = __uint_as_float(xwa[j] & 0xFFFF0000u);
            part += w0 * x0;
            part += w1 * x1;
        }
        for (int j = 0; j < 4; j++) {
            float w0 = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sb;
            float w1 = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sb;
            float x0 = __uint_as_float((xwb[j] & 0xFFFFu) << 16);
            float x1 = __uint_as_float(xwb[j] & 0xFFFF0000u);
            part += w0 * x0;
            part += w1 * x1;
        }
    }
    for (; i0 < k; i0 += stride) {
        uint2 wv = *(const uint2*)(wr + i0);
        uint4 xv = *(const uint4*)(x + i0);
        float s = srow[i0 >> 7];
        unsigned wb[2] = {wv.x, wv.y};
        unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
        for (int j = 0; j < 4; j++) {
            float w0 = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * s;
            float w1 = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * s;
            float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
            float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
            part += w0 * x0;
            part += w1 * x1;
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    if (tid == 0) y[row] = red[0];
}

extern "C" int memra_dsv4_gemv_fp8(const void* w_codes, const float* sc_f32, int sc_cols,
                                   const void* x_bf16, float* y, int n, int k,
                                   void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40011;
    if (sc_cols <= 0) return 40012;
    dsv4_gemv_fp8_kernel<<<(unsigned)n, 128, 0, stream>>>(
        (const uint8_t*)w_codes, sc_f32, sc_cols, (const uint16_t*)x_bf16, y, n, k);
    DSV4_ERR();
    return 0;
}

// Greedy argmax with the host tie rule (first strict max == lowest index among equals).
// max-with-lowest-index is associative and commutative — any reduction tree gives the
// same answer, so this is deterministic AND host-equal by value.
extern "C" __global__ void dsv4_argmax_kernel(const float* __restrict__ v, long n,
                                              int* __restrict__ out) {
    __shared__ float bv[256];
    __shared__ long bi[256];
    int tid = threadIdx.x;
    float best = -INFINITY;
    long besti = -1;
    for (long i = tid; i < n; i += blockDim.x) {
        float x = v[i];
        // i is strictly increasing per thread, so `>` alone keeps the lowest index of
        // any within-thread tie; the tree below applies the cross-thread tie rule.
        if (besti < 0 || x > best) {
            best = x;
            besti = i;
        }
    }
    bv[tid] = best;
    bi[tid] = besti;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            bool take = (bi[tid + off] >= 0) &&
                        (bi[tid] < 0 || bv[tid + off] > bv[tid] ||
                         (bv[tid + off] == bv[tid] && bi[tid + off] < bi[tid]));
            if (take) {
                bv[tid] = bv[tid + off];
                bi[tid] = bi[tid + off];
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = (int)bi[0];
}

extern "C" int memra_dsv4_argmax(const float* v, long n, int* out, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    dsv4_argmax_kernel<<<1, 256, 0, stream>>>(v, n, out);
    DSV4_ERR();
    return 0;
}

// =====================================================================================
// 0731 re-gate extension rung (owner-authorized 2026-08-19, pending ratification;
// derivation + gates in RECEIPTS.md "Lane 0731-regate"): f32-accumulation TWINS for the
// remaining f64 dependency chains on the DEVICE decode path — sink scores/soft/out,
// rmsnorm, headrms, rowsq_scale, indexer_score. Seam: MEMRA_DSV4_DOTS_ARM=f32x (implies
// the lane-9 f32 dots arm). The f64 kernels above are UNTOUCHED and stay the default
// oracle-truth arm; the lane-9 `f32` arm's bytes are also untouched. Every twin keeps
// the SAME launch geometry, the SAME per-thread element order and the SAME reduction
// topology as its f64 twin — the fork is a pure accumulator-type substitution on the
// identical expression DAG (the lane-9 rung-C numeric class: reference stacks accumulate
// f32 on these chains). Gated by decode-gate + CPU teacher-forcing (in-band near-ties
// only) or the rung reverts.

// float twin of dsv4_block_sum (same fixed tree).
__device__ __forceinline__ float dsv4_block_sum_f32(float v, float* sh) {
    int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    return sh[0];
}

// twin of dsv4_rmsnorm_kernel: float sum of squares (same 8-wide load batching, same
// per-thread i order), mean and rsqrt in f32.
extern "C" __global__ void dsv4_rmsnorm_f32acc_kernel(const float* __restrict__ x,
                                                      const float* __restrict__ w,
                                                      float* __restrict__ dst, int ncols,
                                                      float eps) {
    int row = blockIdx.x;
    const float* xr = x + (long)row * ncols;
    float* dr = dst + (long)row * ncols;
    float acc = 0.0f;
    int i = threadIdx.x;
    int B = blockDim.x;
    for (; i + 7 * B < ncols; i += 8 * B) {
        float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
        float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
        acc += v0 * v0;
        acc += v1 * v1;
        acc += v2 * v2;
        acc += v3 * v3;
        acc += v4 * v4;
        acc += v5 * v5;
        acc += v6 * v6;
        acc += v7 * v7;
    }
    for (; i < ncols; i += B) {
        float v = xr[i];
        acc += v * v;
    }
    extern __shared__ float shf32[];
    float tot = dsv4_block_sum_f32(acc, shf32);
    float mean = tot / (float)ncols;
    float rsq = 1.0f / sqrtf(mean + eps);
    for (int i = threadIdx.x; i < ncols; i += blockDim.x)
        dr[i] = (w ? w[i] : 1.0f) * (xr[i] * rsq);
}

extern "C" int memra_dsv4_rmsnorm_f32acc(const float* x, const float* w, float* dst, int rows,
                                         int ncols, float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_rmsnorm_f32acc_kernel<<<(unsigned)rows, threads, threads * sizeof(float), stream>>>(
        x, w, dst, ncols, eps);
    DSV4_ERR();
    return 0;
}

// twin of dsv4_headrms_kernel.
extern "C" __global__ void dsv4_headrms_f32acc_kernel(float* __restrict__ x, int d,
                                                      float eps) {
    int row = blockIdx.x;
    float* xr = x + (long)row * d;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xr[i];
        acc += v * v;
    }
    extern __shared__ float shf32[];
    float tot = dsv4_block_sum_f32(acc, shf32);
    float rsq = 1.0f / sqrtf(tot / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) xr[i] *= rsq;
}

extern "C" int memra_dsv4_headrms_f32acc(float* x, int rows, int d, float eps,
                                         void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_headrms_f32acc_kernel<<<(unsigned)rows, threads, threads * sizeof(float), stream>>>(
        x, d, eps);
    DSV4_ERR();
    return 0;
}

// twin of dsv4_rowsq_scale_kernel.
extern "C" __global__ void dsv4_rowsq_scale_f32acc_kernel(const float* __restrict__ x,
                                                          float* __restrict__ mixes, int w,
                                                          int rows, float eps) {
    int t = blockIdx.x;
    const float* xr = x + (long)t * w;
    float acc = 0.0f;
    int i = threadIdx.x;
    int B = blockDim.x;
    for (; i + 7 * B < w; i += 8 * B) {
        float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
        float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
        acc += v0 * v0;
        acc += v1 * v1;
        acc += v2 * v2;
        acc += v3 * v3;
        acc += v4 * v4;
        acc += v5 * v5;
        acc += v6 * v6;
        acc += v7 * v7;
    }
    for (; i < w; i += B) {
        float v = xr[i];
        acc += v * v;
    }
    extern __shared__ float shf32[];
    float tot = dsv4_block_sum_f32(acc, shf32);
    float rsq = 1.0f / sqrtf(tot / (float)w + eps);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) mixes[(long)t * rows + i] *= rsq;
}

extern "C" int memra_dsv4_rowsq_scale_f32acc(const float* x, float* mixes, int s, int w,
                                             int rows, float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    dsv4_rowsq_scale_f32acc_kernel<<<(unsigned)s, threads, threads * sizeof(float), stream>>>(
        x, mixes, w, rows, eps);
    DSV4_ERR();
    return 0;
}

// twin of dsv4_indexer_score_kernel: float per-head dot chain (same x order), float
// thread-0 head sum (same h order).
extern "C" __global__ void dsv4_indexer_score_f32acc_kernel(
    const float* __restrict__ q, const float* __restrict__ ckv, const float* __restrict__ w,
    float wscale, float* __restrict__ score, int s, int heads, int hd, int nb, int ratio,
    int lim0) {
    long i = blockIdx.x;
    if (i >= (long)s * nb) return;
    int t = (int)(i / nb);
    int j = (int)(i % nb);
    int lim = (lim0 >= 0) ? lim0 : (t + 1) / ratio;
    extern __shared__ float shhf[];
    if (j >= lim) {
        if (threadIdx.x == 0) score[i] = -INFINITY;
        return;
    }
    int h = threadIdx.x;
    if (h < heads) {
        const float* qr = q + ((long)t * heads + h) * hd;
        const float* kr = ckv + (long)j * hd;
        float dacc = 0.0f;
        for (int x = 0; x < hd; x++) dacc += qr[x] * kr[x];
        float r = fmaxf(dacc, 0.0f);
        float ws = w[(long)t * heads + h] * wscale;
        shhf[h] = r * ws;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int hh = 0; hh < heads; hh++) acc += shhf[hh];  // oracle h order
        score[i] = acc;
    }
}

extern "C" int memra_dsv4_indexer_score_f32acc(const float* q, const float* ckv,
                                               const float* w, float wscale, float* score,
                                               int s, int heads, int hd, int nb, int ratio,
                                               int lim0, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * nb;
    if (n == 0) return 0;
    if (n > 2147483647L) return 40009;
    int threads = heads;
    if (threads > 1024) return 40009;
    dsv4_indexer_score_f32acc_kernel<<<(unsigned)n, threads, (size_t)heads * sizeof(float),
                                       stream>>>(q, ckv, w, wscale, score, s, heads, hd, nb,
                                                 ratio, lim0);
    DSV4_ERR();
    return 0;
}

// Absolute-position batched twin for chunked prefill. Arithmetic is verbatim from
// dsv4_indexer_score_f32acc_kernel; only the causal limit includes pos0 so all query
// rows can share one launch and one common nb_max.
extern "C" __global__ void dsv4_indexer_score_f32acc_pos_m_kernel(
    const float* __restrict__ q, const float* __restrict__ ckv,
    const float* __restrict__ w, float wscale, float* __restrict__ score,
    int s, int heads, int hd, int nb, int ratio, int pos0) {
    long i = blockIdx.x;
    if (i >= (long)s * nb) return;
    int t = (int)(i / nb);
    int j = (int)(i % nb);
    int lim = (pos0 + t + 1) / ratio;
    extern __shared__ float shhf[];
    if (j >= lim) {
        if (threadIdx.x == 0) score[i] = -INFINITY;
        return;
    }
    int h = threadIdx.x;
    if (h < heads) {
        const float* qr = q + ((long)t * heads + h) * hd;
        const float* kr = ckv + (long)j * hd;
        float dacc = 0.0f;
        for (int x = 0; x < hd; x++) dacc += qr[x] * kr[x];
        float r = fmaxf(dacc, 0.0f);
        float ws = w[(long)t * heads + h] * wscale;
        shhf[h] = r * ws;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int hh = 0; hh < heads; hh++) acc += shhf[hh];
        score[i] = acc;
    }
}

extern "C" int memra_dsv4_indexer_score_f32acc_pos_m(
    const float* q, const float* ckv, const float* w, float wscale, float* score,
    int s, int heads, int hd, int nb, int ratio, int pos0, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * nb;
    if (s <= 0 || nb <= 0 || ratio <= 0 || pos0 < 0 || n > 2147483647L ||
        heads <= 0 || heads > 1024) return 40009;
    dsv4_indexer_score_f32acc_pos_m_kernel<<<
        (unsigned)n, heads, (size_t)heads * sizeof(float), stream>>>(
        q, ckv, w, wscale, score, s, heads, hd, nb, ratio, pos0);
    DSV4_ERR();
    return 0;
}

// Head/candidate-blocked indexer, derived from Memra's MLA head-blocked scorer.
// Unlike MLA, DSV4's witness is compiled -fmad=false: each dot term must round
// the multiply BEFORE the add. Preserve ascending x and h; no reassociation.
// One thread owns one candidate and all 64 heads. Query slabs are shared across
// 128 candidates; contiguous key reads stage a bank-conflict-free transpose.
extern "C" __global__ __launch_bounds__(128) void dsv4_indexer_score_tiled_kernel(
    const float* __restrict__ q, const float* __restrict__ ckv,
    const float* __restrict__ w, float wscale, float* __restrict__ score,
    int nb, int ratio, int lim0, int pos0) {
    constexpr int H = 64, HD = 128, TPB = 128, KC = 16, KP = TPB + 1;
    const int tid = threadIdx.x, t = blockIdx.y;
    const int j0 = blockIdx.x * TPB, j = j0 + tid;
    const int lim = pos0 >= 0 ? (pos0 + t + 1) / ratio
                             : (lim0 >= 0 ? lim0 : (t + 1) / ratio);
    if (j0 >= lim) {
        if (j < nb) score[(long)t * nb + j] = -INFINITY;
        return;
    }
    __shared__ __align__(16) float qs[KC * H];
    __shared__ float ks[KC * KP];
    float dots[H];
#pragma unroll
    for (int h = 0; h < H; ++h) dots[h] = 0.0f;
    for (int x0 = 0; x0 < HD; x0 += KC) {
        __syncthreads();
        for (int e = tid; e < KC * H; e += TPB) {
            int x = e / H, h = e % H;
            qs[e] = q[((long)t * H + h) * HD + x0 + x];
        }
        for (int e = tid; e < TPB * KC; e += TPB) {
            int local = e / KC, x = e % KC;
            ks[x * KP + local] = j0 + local < nb
                ? ckv[(long)(j0 + local) * HD + x0 + x] : 0.0f;
        }
        __syncthreads();
#pragma unroll 4
        for (int x = 0; x < KC; ++x) {
            const float key = ks[x * KP + tid];
#pragma unroll
            for (int h = 0; h < H; h += 4) {
                const float4 query = *(const float4*)(qs + x * H + h);
                dots[h] = __fadd_rn(dots[h], __fmul_rn(query.x, key));
                dots[h + 1] = __fadd_rn(dots[h + 1], __fmul_rn(query.y, key));
                dots[h + 2] = __fadd_rn(dots[h + 2], __fmul_rn(query.z, key));
                dots[h + 3] = __fadd_rn(dots[h + 3], __fmul_rn(query.w, key));
            }
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float ws = __fmul_rn(w[(long)t * H + h], wscale);
        acc = __fadd_rn(acc, __fmul_rn(fmaxf(dots[h], 0.0f), ws));
    }
    if (j < nb) score[(long)t * nb + j] = j < lim ? acc : -INFINITY;
}

extern "C" int memra_dsv4_indexer_score_tiled(
    const float* q, const float* ckv, const float* w, float wscale, float* score,
    int s, int heads, int hd, int nb, int ratio, int lim0, int pos0, void* stream_v) {
    if (s <= 0 || s > 64 || heads != 64 || hd != 128 || nb <= 0 ||
        nb > 262147 || ratio <= 0 || pos0 < -1 || pos0 > 1048576 ||
        lim0 < -1 || lim0 > nb) return 40009;
    dsv4_indexer_score_tiled_kernel<<<dim3((nb + 127) / 128, s), 128, 0,
        (cudaStream_t)stream_v>>>(q, ckv, w, wscale, score, nb, ratio, lim0, pos0);
    DSV4_ERR();
    return 0;
}

// twins of the sink dec trio. den rides a FLOAT view of the caller's f64 workspace
// (written by K2, read by K3 within one call — format internal to this entry point).
extern "C" __global__ void dsv4_sink_scores_f32acc_kernel(const float* __restrict__ q,
                                                          const float* __restrict__ kv,
                                                          const int* __restrict__ idxs,
                                                          float* __restrict__ scores,
                                                          int heads, int hd, int slots,
                                                          float scale) {
    int sl = blockIdx.x;
    if (sl >= slots) return;
    int ix = idxs[sl];
    extern __shared__ float kvs[];
    if (ix < 0) {
        for (int h = threadIdx.x; h < heads; h += blockDim.x)
            scores[(long)h * slots + sl] = -INFINITY;
        return;
    }
    for (int x = threadIdx.x; x < hd; x += blockDim.x) kvs[x] = kv[(long)ix * hd + x];
    __syncthreads();
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const float* qv = q + (long)h * hd;
        float acc = 0.0f;
        for (int x = 0; x < hd; x++) acc += qv[x] * kvs[x];
        scores[(long)h * slots + sl] = acc * scale;
    }
}

extern "C" __global__ void dsv4_sink_soft_f32acc_kernel(const float* __restrict__ scores,
                                                        const float* __restrict__ sink,
                                                        float* __restrict__ evals,
                                                        float* __restrict__ den, int slots) {
    int h = blockIdx.x;
    const float* srow = scores + (long)h * slots;
    float* erow = evals + (long)h * slots;
    __shared__ float shred[128];
    float m = -INFINITY;
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) m = fmaxf(m, srow[sl]);
    m = dsv4_block_max(m, shred);
    m = fmaxf(m, -1e30f);
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x)
        erow[sl] = (srow[sl] == -INFINITY) ? 0.0f : expf(srow[sl] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        float d = 0.0f;
        for (int sl = 0; sl < slots; sl++) d += erow[sl];  // pads add +0.0
        d += expf(sink[h] - m);
        den[h] = d;
    }
}

extern "C" __global__ void dsv4_sink_out_f32acc_kernel(const float* __restrict__ kv,
                                                       const int* __restrict__ idxs,
                                                       const float* __restrict__ evals,
                                                       const float* __restrict__ den,
                                                       float* __restrict__ o, int heads,
                                                       int hd, int slots) {
    const int XC = 8, HC = 8;
    int x0 = blockIdx.x * XC;
    int h0 = blockIdx.y * HC;
    int tx = threadIdx.x % XC;
    int th = threadIdx.x / XC;
    int x = x0 + tx;
    int h = h0 + th;
    __shared__ float kvt[32 * XC];
    float acc = 0.0f;
    for (int t0 = 0; t0 < slots; t0 += 32) {
        int tl = min(32, slots - t0);
        for (int i = threadIdx.x; i < tl * XC; i += blockDim.x) {
            int sl = t0 + i / XC;
            int xx = x0 + i % XC;
            int ix = idxs[sl];
            kvt[i] = (ix < 0 || xx >= hd) ? 0.0f : kv[(long)ix * hd + xx];
        }
        __syncthreads();
        if (x < hd && h < heads) {
            const float* erow = evals + (long)h * slots;
            for (int i = 0; i < tl; i++) {
                float ev = erow[t0 + i];
                if (ev == 0.0f) continue;
                acc += ev * kvt[i * XC + tx];
            }
        }
        __syncthreads();
    }
    if (x < hd && h < heads) o[(long)h * hd + x] = acc / den[h];
}

extern "C" int memra_dsv4_sink_attn_dec_f32acc(const float* q, const float* kv,
                                               const int* idxs, const float* sink,
                                               float* scores, float* evals, float* den,
                                               float* o, int heads, int hd, int slots,
                                               float scale, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (slots <= 0) return 40010;
    dsv4_sink_scores_f32acc_kernel<<<(unsigned)slots, 64, (size_t)hd * sizeof(float), stream>>>(
        q, kv, idxs, scores, heads, hd, slots, scale);
    DSV4_ERR();
    dsv4_sink_soft_f32acc_kernel<<<(unsigned)heads, 128, 0, stream>>>(scores, sink, evals,
                                                                      den, slots);
    DSV4_ERR();
    dim3 grid((unsigned)((hd + 7) / 8), (unsigned)((heads + 7) / 8));
    dsv4_sink_out_f32acc_kernel<<<grid, 64, 0, stream>>>(kv, idxs, evals, den, o, heads, hd,
                                                         slots);
    DSV4_ERR();
    return 0;
}

// ---------------------------------------------------------------------------
// Iteration 3 (GPU DSpark drafter path) kernels.

// hc-state mean over the copy dim (the M:917-921 DSpark trunk tap): [s, hc, hidden]
// -> [s, hidden]. Oracle parity: f32 accumulation in copy order, one divide — the
// exact loop of dsv4_decode.rs::hc_mean.
extern "C" __global__ void dsv4_hc_mean_kernel(const float* __restrict__ h,
                                               float* __restrict__ out, int s, int hc,
                                               int hidden) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)s * hidden;
    if (i >= n) return;
    int t = (int)(i / hidden);
    int c = (int)(i % hidden);
    float acc = 0.0f;
    for (int k = 0; k < hc; k++) acc += h[((long)t * hc + k) * hidden + c];
    out[i] = acc / (float)hc;
}

extern "C" int memra_dsv4_hc_mean(const float* h, float* out, int s, int hc, int hidden,
                                  void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * hidden;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_hc_mean_kernel<<<(unsigned)blocks, threads, 0, stream>>>(h, out, s, hc, hidden);
    DSV4_ERR();
    return 0;
}

// Model-entry stream expand: [s, hidden] -> [s, hc, hidden], every copy identical
// (memra_gguf::dsv4_forward::hc_expand). The exact inverse of the mean collapse above, and
// the last piece of the hc device program that had no kernel; it is a pure gather, so the
// TU's -fmad=false parity contract does not reach it. Indexed like every other kernel in
// this family: element (t, k, i) at ((t*hc + k)*hidden + i).
extern "C" __global__ void dsv4_hc_expand_kernel(const float* __restrict__ e,
                                                 float* __restrict__ out, long n, int hc,
                                                 int hidden) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long t = i / ((long)hc * hidden);
    long c = i % hidden;
    out[i] = e[t * hidden + c];
}

extern "C" int memra_dsv4_hc_expand(const float* e, float* out, int s, int hc, int hidden,
                                    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * hc * hidden;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    dsv4_hc_expand_kernel<<<(unsigned)blocks, threads, 0, stream>>>(e, out, n, hc, hidden);
    DSV4_ERR();
    return 0;
}

// build_idx with the §3.1 in-round redirect (batched T=k+1 verify): identical to
// dsv4_build_idx_kernel EXCEPT that a window slot whose backing position q (the
// largest q <= pos with q % win == slot) falls inside the open round [pos0, pos]
// resolves to the TRANSIENT batch row trans_base + (q - pos0) instead of the ring
// slot — the ring stays read-only during a round (ring-hazard decision, iteration-3
// receipts). trans_base is the first transient row id in the layer's kvc
// (win + cap_blocks); rows there were written by this round's kv path in position
// order, so every query reads exactly the floats sequential decode would.
extern "C" __global__ void dsv4_build_idx_redirect_kernel(int* __restrict__ idx, int pos,
                                                          int win, int nb, int cap,
                                                          int pos0, int trans_base) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= cap) return;
    int v;
    if (k < win) {
        if (pos >= win - 1) {
            int sp = pos % win;
            int head = win - 1 - sp;
            v = (k < head) ? (sp + 1 + k) : (k - head);
        } else {
            v = (k <= pos) ? k : -1;
        }
        if (v >= 0) {
            // backing position of ring slot v for a query at pos
            int q = pos - ((pos % win - v + win) % win);
            if (q >= pos0) v = trans_base + (q - pos0);
        }
    } else {
        int c = k - win;
        v = (nb >= 0 && c < nb) ? (win + c) : -1;
    }
    idx[k] = v;
}

extern "C" int memra_dsv4_build_idx_redirect(int* idx, int pos, int win, int nb, int cap,
                                             int pos0, int trans_base, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 128;
    int blocks = (cap + threads - 1) / threads;
    dsv4_build_idx_redirect_kernel<<<blocks, threads, 0, stream>>>(idx, pos, win, nb, cap,
                                                                   pos0, trans_base);
    DSV4_ERR();
    return 0;
}

// Batched redirect builder for one contiguous teacher-forced transaction. One flat
// launch replaces T scalar launches; each (row,slot) expression is verbatim from
// dsv4_build_idx_redirect_kernel. fine!=0 leaves the compressed tail padded for the
// batched top-k kernel; coarse layers append every causally complete ratio block.
extern "C" __global__ void dsv4_build_idx_redirect_m_kernel(
        int* __restrict__ idx, int pos0, int s, int win, int ratio, int cap,
        int stride, int trans_base, int fine) {
    long z = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= (long)s * cap) return;
    int row = (int)(z / cap);
    int k = (int)(z % cap);
    int pos = pos0 + row;
    int v;
    if (k < win) {
        if (pos >= win - 1) {
            int sp = pos % win;
            int head = win - 1 - sp;
            v = (k < head) ? (sp + 1 + k) : (k - head);
        } else {
            v = (k <= pos) ? k : -1;
        }
        if (v >= 0) {
            int q = pos - ((pos % win - v + win) % win);
            if (q >= pos0) v = trans_base + (q - pos0);
        }
    } else {
        int c = k - win;
        int nb = (fine || ratio <= 0) ? -1 : (pos + 1) / ratio;
        v = (nb >= 0 && c < nb) ? (win + c) : -1;
    }
    idx[(long)row * stride + k] = v;
}

extern "C" int memra_dsv4_build_idx_redirect_m(
        int* idx, int pos0, int s, int win, int ratio, int cap, int stride,
        int trans_base, int fine, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)s * cap;
    if (s <= 0 || pos0 < 0 || win <= 0 || cap < win || stride < cap ||
        n > 2147483647L) return 40020;
    int threads = 128;
    int blocks = (int)((n + threads - 1) / threads);
    dsv4_build_idx_redirect_m_kernel<<<blocks, threads, 0, stream>>>(
        idx, pos0, s, win, ratio, cap, stride, trans_base, fine);
    DSV4_ERR();
    return 0;
}

// =====================================================================================
// Iteration 3, rung 4 — BATCHED T=k+1 VERIFY kernels (device speculative verify).
//
// Design law (banked in the iteration-3 receipts BEFORE this section was written):
// the batched verify must be BIT-EXACT against T sequential single-position decode
// steps, because the greedy spec==plain identity law is the lane's verdict instrument
// and a numeric fork between the verify pass and the plain pass would break it
// silently at every near-tie. That is achievable here — and only here — because the
// device decode path's dense projections are OUR OWN kernels (dsv4_gemv_bf16,
// dsv4_dots_f32*), not cuBLASLt: a batched GEMV that hoists the WEIGHT load across T
// activation rows keeps each (row, column) dot's element order and reduction tree
// exactly as the m=1 kernel had them, so the value is identical while the weight
// traffic is paid ONCE per round instead of once per token. cuBLASLt is deliberately
// NOT used on this path (its m-order changes split-K plans and shifts logits by
// 0.18-3.08 — banked).
//
// Every kernel below is a TWIN of a pinned kernel above with one added row/query
// dimension. The pinned kernels are byte-untouched (the only edit in this file to an
// existing kernel is fp4_gemm_sel's `a_group`, whose 0 case is the same expression).
// The bodies are duplicated rather than refactored into shared __device__ helpers
// precisely so the gated arms' generated code cannot move.
// =====================================================================================

#define DSV4_TMAX 32  // DSpark uses <=6; bounded prefill prices widths through 32

// ---- batched bf16 GEMV: y[m, n] = x[m, k] @ W[n, k]^T, weight row loaded once.
// Per (t, row): thread tid owns contiguous 8-element chunks at (tid*8 + j*8*blockDim),
// sequential in-chunk pairs, then the 128-leaf halving tree — dsv4_gemv_bf16_kernel's
// accumulation verbatim.
// M is a TEMPLATE parameter, not a runtime one: `part[]` must live in registers (a
// runtime row count forces local-memory indexing and the whole amortization is lost).
// One instantiation per depth, dispatched by the launcher's switch.
template <int M>
__global__ void dsv4_gemv_bf16_m_kernel(const uint16_t* __restrict__ w,
                                        const uint16_t* __restrict__ x, float* __restrict__ y,
                                        int n, int k, int xstride, int ystride) {
    int row = blockIdx.x;
    if (row >= n) return;
    const uint16_t* wr = w + (long)row * k;
    float part[M];
#pragma unroll
    for (int t = 0; t < M; t++) part[t] = 0.0f;
    for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
        uint4 wv = *(const uint4*)(wr + i0);
        unsigned ww[4] = {wv.x, wv.y, wv.z, wv.w};
        float wu[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wu[2 * j] = __uint_as_float((ww[j] & 0xFFFFu) << 16);
            wu[2 * j + 1] = __uint_as_float(ww[j] & 0xFFFF0000u);
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x + (long)t * xstride + i0);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wu[2 * j] * x0;
                acc += wu[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
#pragma unroll
    for (int t = 0; t < M; t++) {
        __syncthreads();  // red free from the previous row's tree
        red[tid] = part[t];
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) y[(long)t * ystride + row] = red[0];
    }
}

#define DSV4_GEMV_M_CASE(MM)                                                          \
    case MM:                                                                          \
        dsv4_gemv_bf16_m_kernel<MM><<<(unsigned)n, 128, 0, stream>>>(                 \
            (const uint16_t*)w_bf16, (const uint16_t*)x_bf16, y, n, k, xstride,       \
            ystride);                                                                 \
        break;

// xstride/ystride in ELEMENTS (0 = packed: k and n). The grouped output projection
// reads slices of a wider activation row and writes slices of a wider output row, so
// the strides are load-bearing there and packed everywhere else.
extern "C" int memra_dsv4_gemv_bf16_m(const void* w_bf16, const void* x_bf16, float* y, int m,
                                      int n, int k, int xstride, int ystride,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40011;
    if (m < 1) return 40020;
    if (xstride <= 0) xstride = k;
    if (ystride <= 0) ystride = n;
    if (xstride % 8 != 0) return 40011;
    if (m > DSV4_TMAX) {
        // Wide prefill transaction: tile the exact registered-row kernel instead of
        // growing its register array without bound. Each output row keeps the same
        // element order and 128-leaf reduction tree; tiles are independent and may
        // share the enclosing causal attention transaction.
        const uint16_t* x = (const uint16_t*)x_bf16;
        for (int base = 0; base < m; base += 8) {
            int tile = min(8, m - base);
            int rc = memra_dsv4_gemv_bf16_m(
                w_bf16, x + (long)base * xstride, y + (long)base * ystride,
                tile, n, k, xstride, ystride, stream_v);
            if (rc != 0) return rc;
        }
        return 0;
    }
    switch (m) {
        DSV4_GEMV_M_CASE(1)
        DSV4_GEMV_M_CASE(2)
        DSV4_GEMV_M_CASE(3)
        DSV4_GEMV_M_CASE(4)
        DSV4_GEMV_M_CASE(5)
        DSV4_GEMV_M_CASE(6)
        DSV4_GEMV_M_CASE(7)
        DSV4_GEMV_M_CASE(8)
        DSV4_GEMV_M_CASE(9)
        DSV4_GEMV_M_CASE(10)
        DSV4_GEMV_M_CASE(11)
        DSV4_GEMV_M_CASE(12)
        DSV4_GEMV_M_CASE(13)
        DSV4_GEMV_M_CASE(14)
        DSV4_GEMV_M_CASE(15)
        DSV4_GEMV_M_CASE(16)
        DSV4_GEMV_M_CASE(17)
        DSV4_GEMV_M_CASE(18)
        DSV4_GEMV_M_CASE(19)
        DSV4_GEMV_M_CASE(20)
        DSV4_GEMV_M_CASE(21)
        DSV4_GEMV_M_CASE(22)
        DSV4_GEMV_M_CASE(23)
        DSV4_GEMV_M_CASE(24)
        DSV4_GEMV_M_CASE(25)
        DSV4_GEMV_M_CASE(26)
        DSV4_GEMV_M_CASE(27)
        DSV4_GEMV_M_CASE(28)
        DSV4_GEMV_M_CASE(29)
        DSV4_GEMV_M_CASE(30)
        DSV4_GEMV_M_CASE(31)
        DSV4_GEMV_M_CASE(32)
        default:
            return 40020;
    }
    DSV4_ERR();
    return 0;
}

// ---- iteration-5 FP8 dense arm, batched twin: dsv4_gemv_bf16_m_kernel's accumulation
// VERBATIM (weight chunk hoisted across the t rows, per-(t,j) order unchanged); the only
// delta is the weight decode — e4m3 code x exact pow2 block scale, the same exact value
// the bf16 slab holds. See dsv4_gemv_fp8_kernel's header note for the bit-identity law.
template <int M>
__global__ void dsv4_gemv_fp8_m_kernel(const uint8_t* __restrict__ w,
                                       const float* __restrict__ sc, int sc_cols,
                                       const uint16_t* __restrict__ x, float* __restrict__ y,
                                       int n, int k, int xstride, int ystride) {
    int row = blockIdx.x;
    if (row >= n) return;
    // smem e4m3 LUT — see dsv4_gemv_fp8_kernel's note (bit-inert decode transport).
    __shared__ float e4m3_tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4m3_tab[i] = dsv4_e4m3((uint8_t)i);
    __syncthreads();
    const uint8_t* wr = w + (long)row * k;
    const float* srow = sc + (long)(row >> 7) * sc_cols;
    float part[M];
#pragma unroll
    for (int t = 0; t < M; t++) part[t] = 0.0f;
    // Unroll-by-2, early weight loads — the m=1 twin's note applies: load scheduling
    // only, per-(t)-accumulation order verbatim, bit-identical.
    int stride = blockDim.x * 8;
    int i0 = threadIdx.x * 8;
    for (; i0 + stride < k; i0 += 2 * stride) {
        int i1 = i0 + stride;
        uint2 wva = *(const uint2*)(wr + i0);
        uint2 wvb = *(const uint2*)(wr + i1);
        float sa = srow[i0 >> 7];
        float sb = srow[i1 >> 7];
        unsigned wba[2] = {wva.x, wva.y};
        unsigned wbb[2] = {wvb.x, wvb.y};
        float wua[8], wub[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wua[2 * j] = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sa;
            wua[2 * j + 1] = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sa;
            wub[2 * j] = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sb;
            wub[2 * j + 1] = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sb;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x + (long)t * xstride + i0);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wua[2 * j] * x0;
                acc += wua[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x + (long)t * xstride + i1);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wub[2 * j] * x0;
                acc += wub[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
    }
    for (; i0 < k; i0 += stride) {
        uint2 wv = *(const uint2*)(wr + i0);
        float s = srow[i0 >> 7];
        unsigned wb[2] = {wv.x, wv.y};
        float wu[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wu[2 * j] = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * s;
            wu[2 * j + 1] = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * s;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x + (long)t * xstride + i0);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wu[2 * j] * x0;
                acc += wu[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
#pragma unroll
    for (int t = 0; t < M; t++) {
        __syncthreads();  // red free from the previous row's tree
        red[tid] = part[t];
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) y[(long)t * ystride + row] = red[0];
    }
}

#define DSV4_GEMV_FP8_M_CASE(MM)                                                     \
    case MM:                                                                         \
        dsv4_gemv_fp8_m_kernel<MM><<<(unsigned)n, 128, 0, stream>>>(                 \
            (const uint8_t*)w_codes, sc_f32, sc_cols, (const uint16_t*)x_bf16, y, n, \
            k, xstride, ystride);                                                    \
        break;

extern "C" int memra_dsv4_gemv_fp8_m(const void* w_codes, const float* sc_f32, int sc_cols,
                                     const void* x_bf16, float* y, int m, int n, int k,
                                     int xstride, int ystride, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40011;
    if (m < 1) return 40020;
    if (sc_cols <= 0) return 40012;
    if (xstride <= 0) xstride = k;
    if (ystride <= 0) ystride = n;
    if (xstride % 8 != 0) return 40011;
    if (m > DSV4_TMAX) {
        const uint16_t* x = (const uint16_t*)x_bf16;
        for (int base = 0; base < m; base += 8) {
            int tile = min(8, m - base);
            int rc = memra_dsv4_gemv_fp8_m(
                w_codes, sc_f32, sc_cols, x + (long)base * xstride,
                y + (long)base * ystride, tile, n, k, xstride, ystride, stream_v);
            if (rc != 0) return rc;
        }
        return 0;
    }
    switch (m) {
        DSV4_GEMV_FP8_M_CASE(1)
        DSV4_GEMV_FP8_M_CASE(2)
        DSV4_GEMV_FP8_M_CASE(3)
        DSV4_GEMV_FP8_M_CASE(4)
        DSV4_GEMV_FP8_M_CASE(5)
        DSV4_GEMV_FP8_M_CASE(6)
        DSV4_GEMV_FP8_M_CASE(7)
        DSV4_GEMV_FP8_M_CASE(8)
        DSV4_GEMV_FP8_M_CASE(9)
        DSV4_GEMV_FP8_M_CASE(10)
        DSV4_GEMV_FP8_M_CASE(11)
        DSV4_GEMV_FP8_M_CASE(12)
        DSV4_GEMV_FP8_M_CASE(13)
        DSV4_GEMV_FP8_M_CASE(14)
        DSV4_GEMV_FP8_M_CASE(15)
        DSV4_GEMV_FP8_M_CASE(16)
        DSV4_GEMV_FP8_M_CASE(17)
        DSV4_GEMV_FP8_M_CASE(18)
        DSV4_GEMV_FP8_M_CASE(19)
        DSV4_GEMV_FP8_M_CASE(20)
        DSV4_GEMV_FP8_M_CASE(21)
        DSV4_GEMV_FP8_M_CASE(22)
        DSV4_GEMV_FP8_M_CASE(23)
        DSV4_GEMV_FP8_M_CASE(24)
        DSV4_GEMV_FP8_M_CASE(25)
        DSV4_GEMV_FP8_M_CASE(26)
        DSV4_GEMV_FP8_M_CASE(27)
        DSV4_GEMV_FP8_M_CASE(28)
        DSV4_GEMV_FP8_M_CASE(29)
        DSV4_GEMV_FP8_M_CASE(30)
        DSV4_GEMV_FP8_M_CASE(31)
        DSV4_GEMV_FP8_M_CASE(32)
        default:
            return 40020;
    }
    DSV4_ERR();
    return 0;
}

// ---- f32-island dots, batched rows with the weight row hoisted (f64 accumulation arm).
// Per (t, j) the element order and the f64 halving tree are dsv4_dots_f32_kernel's.
template <int M>
__global__ void dsv4_dots_f32_mrow_kernel(const float* __restrict__ x,
                                          const void* __restrict__ w, int w_is_bf16,
                                          float* __restrict__ y, int k, int n) {
    int j = blockIdx.x;
    if (j >= n) return;
    double acc[M];
#pragma unroll
    for (int t = 0; t < M; t++) acc[t] = 0.0;
    if (w_is_bf16) {
        const uint16_t* wr = (const uint16_t*)w + (long)j * k;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            double wv = (double)__uint_as_float(((unsigned)wr[i]) << 16);
#pragma unroll
            for (int t = 0; t < M; t++) acc[t] += (double)x[(long)t * k + i] * wv;
        }
    } else {
        const float* wr = (const float*)w + (long)j * k;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            double wv = (double)wr[i];
#pragma unroll
            for (int t = 0; t < M; t++) acc[t] += (double)x[(long)t * k + i] * wv;
        }
    }
    extern __shared__ double shd[];
#pragma unroll
    for (int t = 0; t < M; t++) {
        __syncthreads();  // shd free from the previous row's tree
        double tot = dsv4_block_sum(acc[t], shd);
        if (threadIdx.x == 0) y[(long)t * n + j] = (float)tot;
    }
}

#define DSV4_DOTS_F32_MROW_CASE(MM)                                                    \
    case MM:                                                                           \
        dsv4_dots_f32_mrow_kernel<MM>                                                  \
            <<<(unsigned)n, threads, threads * sizeof(double), stream>>>(x, w, w_is_bf16, \
                                                                        y, k, n);      \
        break;

extern "C" int memra_dsv4_dots_f32_mrow(const float* x, const void* w, int w_is_bf16,
                                        float* y, int s, int k, int n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1) return 40020;
    if (s > DSV4_TMAX) {
        for (int base = 0; base < s; base += 8) {
            int tile = min(8, s - base);
            int rc = memra_dsv4_dots_f32_mrow(
                x + (long)base * k, w, w_is_bf16, y + (long)base * n,
                tile, k, n, stream_v);
            if (rc != 0) return rc;
        }
        return 0;
    }
    int threads = 128;
    switch (s) {
        DSV4_DOTS_F32_MROW_CASE(1)
        DSV4_DOTS_F32_MROW_CASE(2)
        DSV4_DOTS_F32_MROW_CASE(3)
        DSV4_DOTS_F32_MROW_CASE(4)
        DSV4_DOTS_F32_MROW_CASE(5)
        DSV4_DOTS_F32_MROW_CASE(6)
        DSV4_DOTS_F32_MROW_CASE(7)
        DSV4_DOTS_F32_MROW_CASE(8)
        DSV4_DOTS_F32_MROW_CASE(9)
        DSV4_DOTS_F32_MROW_CASE(10)
        DSV4_DOTS_F32_MROW_CASE(11)
        DSV4_DOTS_F32_MROW_CASE(12)
        DSV4_DOTS_F32_MROW_CASE(13)
        DSV4_DOTS_F32_MROW_CASE(14)
        DSV4_DOTS_F32_MROW_CASE(15)
        DSV4_DOTS_F32_MROW_CASE(16)
        DSV4_DOTS_F32_MROW_CASE(17)
        DSV4_DOTS_F32_MROW_CASE(18)
        DSV4_DOTS_F32_MROW_CASE(19)
        DSV4_DOTS_F32_MROW_CASE(20)
        DSV4_DOTS_F32_MROW_CASE(21)
        DSV4_DOTS_F32_MROW_CASE(22)
        DSV4_DOTS_F32_MROW_CASE(23)
        DSV4_DOTS_F32_MROW_CASE(24)
        DSV4_DOTS_F32_MROW_CASE(25)
        DSV4_DOTS_F32_MROW_CASE(26)
        DSV4_DOTS_F32_MROW_CASE(27)
        DSV4_DOTS_F32_MROW_CASE(28)
        DSV4_DOTS_F32_MROW_CASE(29)
        DSV4_DOTS_F32_MROW_CASE(30)
        DSV4_DOTS_F32_MROW_CASE(31)
        DSV4_DOTS_F32_MROW_CASE(32)
        default:
            return 40020;
    }
    DSV4_ERR();
    return 0;
}

// ---- f32-island dots, batched rows, f32-accumulation arm (the ratified f32x class).
// Per (t, j): thread tid owns 8-element chunks at (tid*8 + j*8*blockDim), the same
// in-chunk pair order, then the 128-leaf f32 halving tree — dsv4_dots_f32acc_kernel's.
template <int M>
__global__ void dsv4_dots_f32acc_mrow_kernel(const float* __restrict__ x,
                                             const void* __restrict__ w, int w_is_bf16,
                                             float* __restrict__ y, int k, int n) {
    int j = blockIdx.x;
    if (j >= n) return;
    float part[M];
#pragma unroll
    for (int t = 0; t < M; t++) part[t] = 0.0f;
    if (w_is_bf16) {
        const uint16_t* wr = (const uint16_t*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            uint4 wv = *(const uint4*)(wr + i0);
            unsigned ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int t = 0; t < M; t++) {
                const float* xr = x + (long)t * k;
                float4 xa = *(const float4*)(xr + i0);
                float4 xb = *(const float4*)(xr + i0 + 4);
                float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float acc = part[t];
#pragma unroll
                for (int q2 = 0; q2 < 4; q2++) {
                    float w0 = __uint_as_float((ww[q2] & 0xFFFFu) << 16);
                    float w1 = __uint_as_float(ww[q2] & 0xFFFF0000u);
                    acc += xs[2 * q2] * w0;
                    acc += xs[2 * q2 + 1] * w1;
                }
                part[t] = acc;
            }
        }
    } else {
        const float* wr = (const float*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            float4 wa = *(const float4*)(wr + i0);
            float4 wb = *(const float4*)(wr + i0 + 4);
#pragma unroll
            for (int t = 0; t < M; t++) {
                const float* xr = x + (long)t * k;
                float4 xa = *(const float4*)(xr + i0);
                float4 xb = *(const float4*)(xr + i0 + 4);
                float acc = part[t];
                acc += xa.x * wa.x;
                acc += xa.y * wa.y;
                acc += xa.z * wa.z;
                acc += xa.w * wa.w;
                acc += xb.x * wb.x;
                acc += xb.y * wb.y;
                acc += xb.z * wb.z;
                acc += xb.w * wb.w;
                part[t] = acc;
            }
        }
    }
    __shared__ float red[128];
    int tid = threadIdx.x;
#pragma unroll
    for (int t = 0; t < M; t++) {
        __syncthreads();
        red[tid] = part[t];
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        if (tid == 0) y[(long)t * n + j] = red[0];
    }
}

#define DSV4_DOTS_F32ACC_MROW_CASE(MM)                                                 \
    case MM:                                                                           \
        dsv4_dots_f32acc_mrow_kernel<MM>                                               \
            <<<(unsigned)n, threads, 0, stream>>>(x, w, w_is_bf16, y, k, n);           \
        break;

extern "C" int memra_dsv4_dots_f32acc_mrow(const float* x, const void* w, int w_is_bf16,
                                           float* y, int s, int k, int n, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (k % 8 != 0) return 40012;
    if (s < 1) return 40020;
    if (s > DSV4_TMAX) {
        for (int base = 0; base < s; base += 8) {
            int tile = min(8, s - base);
            int rc = memra_dsv4_dots_f32acc_mrow(
                x + (long)base * k, w, w_is_bf16, y + (long)base * n,
                tile, k, n, stream_v);
            if (rc != 0) return rc;
        }
        return 0;
    }
    int threads = 128;
    switch (s) {
        DSV4_DOTS_F32ACC_MROW_CASE(1)
        DSV4_DOTS_F32ACC_MROW_CASE(2)
        DSV4_DOTS_F32ACC_MROW_CASE(3)
        DSV4_DOTS_F32ACC_MROW_CASE(4)
        DSV4_DOTS_F32ACC_MROW_CASE(5)
        DSV4_DOTS_F32ACC_MROW_CASE(6)
        DSV4_DOTS_F32ACC_MROW_CASE(7)
        DSV4_DOTS_F32ACC_MROW_CASE(8)
        DSV4_DOTS_F32ACC_MROW_CASE(9)
        DSV4_DOTS_F32ACC_MROW_CASE(10)
        DSV4_DOTS_F32ACC_MROW_CASE(11)
        DSV4_DOTS_F32ACC_MROW_CASE(12)
        DSV4_DOTS_F32ACC_MROW_CASE(13)
        DSV4_DOTS_F32ACC_MROW_CASE(14)
        DSV4_DOTS_F32ACC_MROW_CASE(15)
        DSV4_DOTS_F32ACC_MROW_CASE(16)
        DSV4_DOTS_F32ACC_MROW_CASE(17)
        DSV4_DOTS_F32ACC_MROW_CASE(18)
        DSV4_DOTS_F32ACC_MROW_CASE(19)
        DSV4_DOTS_F32ACC_MROW_CASE(20)
        DSV4_DOTS_F32ACC_MROW_CASE(21)
        DSV4_DOTS_F32ACC_MROW_CASE(22)
        DSV4_DOTS_F32ACC_MROW_CASE(23)
        DSV4_DOTS_F32ACC_MROW_CASE(24)
        DSV4_DOTS_F32ACC_MROW_CASE(25)
        DSV4_DOTS_F32ACC_MROW_CASE(26)
        DSV4_DOTS_F32ACC_MROW_CASE(27)
        DSV4_DOTS_F32ACC_MROW_CASE(28)
        DSV4_DOTS_F32ACC_MROW_CASE(29)
        DSV4_DOTS_F32ACC_MROW_CASE(30)
        DSV4_DOTS_F32ACC_MROW_CASE(31)
        DSV4_DOTS_F32ACC_MROW_CASE(32)
        default:
            return 40020;
    }
    DSV4_ERR();
    return 0;
}

// ---- hc Sinkhorn, one block per position (dsv4_hc_sinkhorn_kernel's body verbatim on
// the position's own mixes/pre/post/comb slices).
extern "C" __global__ void dsv4_hc_sinkhorn_m_kernel(const float* __restrict__ mixes_all,
                                                     const float* __restrict__ scale,
                                                     const float* __restrict__ base,
                                                     float* __restrict__ pre_all,
                                                     float* __restrict__ post_all,
                                                     float* __restrict__ comb_all, int hc,
                                                     int iters, float eps) {
    int p = blockIdx.x;
    const float* mixes = mixes_all + (long)p * (2 + hc) * hc;
    float* pre = pre_all + (long)p * hc;
    float* post = post_all + (long)p * hc;
    float* comb = comb_all + (long)p * hc * hc;
    int t = threadIdx.x;
    if (t < hc) {
        pre[t] = dsv4_sigmoid(mixes[t] * scale[0] + base[t]) + eps;
        post[t] = 2.0f * dsv4_sigmoid(mixes[hc + t] * scale[1] + base[hc + t]);
    }
    for (int i = t; i < hc * hc; i += blockDim.x)
        comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
    __syncthreads();
    if (t < hc) {
        float* row = comb + t * hc;
        float mx = -INFINITY;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, row[k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) {
            row[k] = expf(row[k] - mx);
            sum += row[k];
        }
        for (int k = 0; k < hc; k++) row[k] = row[k] / sum + eps;
    }
    __syncthreads();
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            if (t < hc) {
                float sum = 0.0f;
                for (int k = 0; k < hc; k++) sum += comb[t * hc + k];
                for (int k = 0; k < hc; k++) comb[t * hc + k] /= sum + eps;
            }
            __syncthreads();
        }
        if (t < hc) {
            float sum = 0.0f;
            for (int j = 0; j < hc; j++) sum += comb[j * hc + t];
            for (int j = 0; j < hc; j++) comb[j * hc + t] /= sum + eps;
        }
        __syncthreads();
    }
}

extern "C" int memra_dsv4_hc_sinkhorn_m(const float* mixes, const float* scale,
                                        const float* base, float* pre, float* post,
                                        float* comb, int s, int hc, int iters, float eps,
                                        void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1) return 40020;
    dsv4_hc_sinkhorn_m_kernel<<<(unsigned)s, 32, 0, stream>>>(mixes, scale, base, pre, post,
                                                              comb, hc, iters, eps);
    DSV4_ERR();
    return 0;
}

// ---- head hc gate, batched positions.
extern "C" __global__ void dsv4_hc_head_pre_m_kernel(const float* __restrict__ mixes,
                                                     const float* __restrict__ scale,
                                                     const float* __restrict__ base,
                                                     float* __restrict__ pre, int hc,
                                                     float eps) {
    int p = blockIdx.x;
    int c = threadIdx.x;
    if (c >= hc) return;
    pre[(long)p * hc + c] =
        dsv4_sigmoid(mixes[(long)p * hc + c] * scale[0] + base[c]) + eps;
}

extern "C" int memra_dsv4_hc_head_pre_m(const float* mixes, const float* scale,
                                        const float* base, float* pre, int s, int hc,
                                        float eps, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1) return 40020;
    dsv4_hc_head_pre_m_kernel<<<(unsigned)s, 32, 0, stream>>>(mixes, scale, base, pre, hc,
                                                              eps);
    DSV4_ERR();
    return 0;
}

// ---- hc pre-chain FUSED (lane/glm5-decode-diet, 2026-08-31): rowsq_scale + Sinkhorn +
// collapse in ONE launch per (site, token) — the launch-diet census's top mHC increment
// (mhc-sites: 3.306 ms/token GPU + 362 launches/token on the glm5 serving arm; the 20
// serial Sinkhorn iterations ran as their own 18.3 us launch behind a 15.5 us rowsq
// launch per site, 90 sites/token).
//
// BIT-IDENTITY BY CONSTRUCTION, asserted bytewise by the gate
// (crates/memra-engine/tests/hc_fused_pre_gpu.rs):
//   * stage 1 is dsv4_rowsq_scale_kernel's 8-wide f64 reduction VERBATIM at the same
//     blockDim=128 (the reduction tree depends on blockDim, so the launcher pins it);
//     the scaled mixes are staged in SHARED instead of written back — the same single
//     f32 multiply, rounded by the store either way, and nothing downstream reads the
//     scaled mixes buffer once Sinkhorn has consumed it (hyper.rs drops it).
//   * stage 2 is dsv4_hc_sinkhorn_m_kernel's body VERBATIM on shared-memory operands
//     (shared vs global residency does not change f32 arithmetic); the i-strided init
//     partition changes with blockDim, elementwise op unchanged.
//   * stage 3 is dsv4_hc_collapse_kernel's per-element expression VERBATIM (c ascending,
//     one thread per element) reading the SAME pre values the unfused chain reads back
//     from global.
//
// BIT-PRESERVING EARLY EXIT (the "fewer-iteration fixed point at t=1" question, answered
// without changing bits): one Sinkhorn iteration is a deterministic map comb -> f(comb),
// so if a full (row_norm, col_norm) application leaves every comb bit unchanged, every
// later iteration reproduces the same bits by induction — breaking is invisible in the
// output. The exit fires ONLY on bitwise stationarity (uint compare, NaN-safe); it is not
// a tolerance and needs no flag of its own. `niters` (nullable) records the executed
// iteration count per token — the gate's convergence receipt.
//
// Host seam: MEMRA_HC_FUSED_PRE (default OFF, read per call in hyper.rs::pre_finish).
#define DSV4_HC_MAX 8

extern "C" __global__ void dsv4_hc_pre_fused_kernel(
        const float* __restrict__ x, const float* __restrict__ mixes_all,
        const float* __restrict__ scale, const float* __restrict__ base,
        float* __restrict__ pre_all, float* __restrict__ post_all,
        float* __restrict__ comb_all, float* __restrict__ y, int w, int rows, int hc, int d,
        int iters, float eps, int* __restrict__ niters) {
    int p = blockIdx.x;
    int t = threadIdx.x;
    int B = blockDim.x;

    // ---- stage 1: rowsq — dsv4_rowsq_scale_kernel's reduction VERBATIM (8-wide batching,
    // f64 accumulate, fixed shared tree at blockDim=128).
    const float* xr = x + (long)p * w;
    double acc = 0.0;
    {
        int i = t;
        for (; i + 7 * B < w; i += 8 * B) {
            float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
            float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
            acc += (double)v0 * (double)v0;
            acc += (double)v1 * (double)v1;
            acc += (double)v2 * (double)v2;
            acc += (double)v3 * (double)v3;
            acc += (double)v4 * (double)v4;
            acc += (double)v5 * (double)v5;
            acc += (double)v6 * (double)v6;
            acc += (double)v7 * (double)v7;
        }
        for (; i < w; i += B) {
            double v = (double)xr[i];
            acc += v * v;
        }
    }
    __shared__ double shd[128];
    double tot = dsv4_block_sum(acc, shd);
    float rsq = 1.0f / sqrtf((float)(tot / (double)w) + eps);

    // Scaled mixes staged in shared: the unfused kernel's `mixes[i] *= rsq` store, same
    // single f32 multiply, rounded identically by the shared store.
    __shared__ float smix[(2 + DSV4_HC_MAX) * DSV4_HC_MAX];
    const float* mixes = mixes_all + (long)p * rows;
    for (int i = t; i < rows; i += B) smix[i] = mixes[i] * rsq;
    __syncthreads();

    // ---- stage 2: Sinkhorn — dsv4_hc_sinkhorn_m_kernel's body VERBATIM on shared operands.
    float* pre = pre_all + (long)p * hc;
    float* post = post_all + (long)p * hc;
    float* combg = comb_all + (long)p * hc * hc;
    __shared__ float spre[DSV4_HC_MAX];
    __shared__ float comb[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ float sprev[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ unsigned schanged;
    if (t < hc) {
        float pv = dsv4_sigmoid(smix[t] * scale[0] + base[t]) + eps;
        pre[t] = pv;
        spre[t] = pv;
        post[t] = 2.0f * dsv4_sigmoid(smix[hc + t] * scale[1] + base[hc + t]);
    }
    for (int i = t; i < hc * hc; i += B)
        comb[i] = smix[2 * hc + i] * scale[2] + base[2 * hc + i];
    __syncthreads();
    if (t < hc) {
        float* row = comb + t * hc;
        float mx = -INFINITY;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, row[k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) {
            row[k] = expf(row[k] - mx);
            sum += row[k];
        }
        for (int k = 0; k < hc; k++) row[k] = row[k] / sum + eps;
    }
    __syncthreads();
    int done = 0;
    for (int it = 0; it < iters; it++) {
        if (it > 0) {
            // Snapshot for the stationarity check; reset the shared flag behind a barrier.
            for (int i = t; i < hc * hc; i += B) sprev[i] = comb[i];
            if (t == 0) schanged = 0u;
            __syncthreads();
            if (t < hc) {
                float sum = 0.0f;
                for (int k = 0; k < hc; k++) sum += comb[t * hc + k];
                for (int k = 0; k < hc; k++) comb[t * hc + k] /= sum + eps;
            }
            __syncthreads();
        }
        if (t < hc) {
            float sum = 0.0f;
            for (int j = 0; j < hc; j++) sum += comb[j * hc + t];
            for (int j = 0; j < hc; j++) comb[j * hc + t] /= sum + eps;
        }
        __syncthreads();
        done = it + 1;
        if (it > 0) {
            unsigned ch = 0u;
            for (int i = t; i < hc * hc; i += B)
                ch |= (unsigned)(__float_as_uint(sprev[i]) != __float_as_uint(comb[i]));
            if (ch) atomicOr(&schanged, 1u);
            __syncthreads();
            // Register copy, then a barrier BEFORE anyone can reset schanged next iteration
            // (the double-barrier flag pattern — a racing reset would fork the block).
            unsigned stop = (schanged == 0u);
            __syncthreads();
            if (stop) break; // bitwise-stationary: every later iteration is identity
        }
    }
    if (niters && t == 0) niters[p] = done;
    for (int i = t; i < hc * hc; i += B) combg[i] = comb[i];
    __syncthreads();

    // ---- stage 3: collapse — dsv4_hc_collapse_kernel's per-element expression VERBATIM
    // (copy order c ascending, f32 adds; spre holds the exact bits `pre` was written with).
    float* yr = y + (long)p * d;
    for (int i = t; i < d; i += B) {
        float acc2 = 0.0f;
        for (int c = 0; c < hc; c++) acc2 += spre[c] * xr[(long)c * d + i];
        yr[i] = acc2;
    }
}

extern "C" int memra_dsv4_hc_pre_fused(const float* x, const float* mixes, const float* scale,
                                       const float* base, float* pre, float* post, float* comb,
                                       float* y, int s, int hc, int d, int iters, float eps,
                                       int* niters, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1 || hc < 1 || hc > DSV4_HC_MAX || d < 1 || iters < 1) return 40021;
    int w = hc * d;
    int rows = (2 + hc) * hc;
    // 128 threads is LOAD-BEARING: dsv4_rowsq_scale's reduction tree shape (and therefore
    // its bits) is a function of blockDim, and the unfused launcher pins 128.
    dsv4_hc_pre_fused_kernel<<<(unsigned)s, 128, 0, stream>>>(x, mixes, scale, base, pre, post,
                                                              comb, y, w, rows, hc, d, iters,
                                                              eps, niters);
    DSV4_ERR();
    return 0;
}

// ---- hc pre-chain FUSED v2 (MEMRA_HC_FUSED_PRE=2, lane/b200-sinkhorn-fusion-20260902
// follow-up). Same stage 1 (rowsq) and stage 3 (collapse) as dsv4_hc_pre_fused_kernel
// above, copied VERBATIM — rowsq's reduction tree is a function of blockDim=128 (see that
// kernel's own "128 threads is LOAD-BEARING" note) so it cannot move to a warp without
// becoming a different numeric class, and collapse already runs at full block width for
// throughput (its per-output-element value does not depend on thread count at all: each
// `y[i]` is one thread's own sequential loop over `spre[0..hc-1]`, so blockDim never
// touches its bits — "share a warp with rowsq" was checked and would only cost
// parallelism on d=4096 elements, not gain anything, so it is left at full block width).
//
// The B200 nsys census that opened this follow-up (research/b200-sinkhorn-fusion-20260902/
// LANE.md) measured dsv4_hc_pre_fused_kernel itself at 32.8 us/launch average with
// MEMRA_HC_FUSED_PRE=1 in serving, now the second-largest kernel: at t=1 the real
// per-token math (hc=4, d=4096) is small, so almost all of that 32.8 us is the stage-2
// Sinkhorn loop's up-to-20 `__syncthreads()` PAIRS — a 128-thread, up-to-4-warp barrier,
// paid twenty-odd times to synchronize work that only threads t<hc (and the t<hc*hc
// snapshot/writeback strided loops) ever touch.
//
// THE ONE CHANGE: for hc<=4, rows=(2+hc)*hc<=24 and hc*hc<=16 — every shared-memory
// index stage 2 ever reads or writes (smix[0..rows-1], comb/sprev[0..hc*hc-1],
// spre[0..hc-1], schanged) is < 32, i.e. lives entirely inside warp 0's lane range
// (t=threadIdx.x 0..31). Stage 2 below is dsv4_hc_pre_fused_kernel's stage 2 body
// VERBATIM — same operands, same operations, same sequential summation order inside
// each `if (t < hc)` / `if (t < hc*hc)` arm — wrapped in `if (t < 32)` (warp-uniform: all
// 32 lanes of warp 0 take it together, the other three warps skip it together, so there
// is no intra-warp divergence anywhere) with every `__syncthreads()` in that scope
// replaced by `__syncwarp()`. This is a SYNCHRONIZATION-PRIMITIVE SUBSTITUTION ONLY: no
// operand, no operator, no summation order changes, so the values stage 2 produces are
// bit-identical to v1's by construction — a barrier does not touch a mantissa. Threads
// 32..127 did no work in stage 2 in v1 either (every write there is already gated on
// t<hc or t<hc*hc, both <32 for this hc range); they simply no longer pay for barriers
// guarding work they were never part of.
//
// ONE real `__syncthreads()` remains, between stage 2 and stage 3: collapse reads
// `spre[]` with the FULL 128-thread block, and warp 0's writes must become visible
// across warp boundaries — `__syncwarp()` cannot do that (it only orders memory within
// its own warp), so this barrier is not optional and is not a place to save more time.
//
// hc>4 (rows>32, e.g. hc=5 -> rows=35): the warp-0-only invariant above no longer holds
// (some shared indices this kernel would touch live in warp 1+), and this lane did not
// build a multi-warp partial-sync scheme for it — unverifiable on a GPU-less worktree
// and not the shape this follow-up was asked to speed up (GLM-5.3-Flash is hc=4). The
// host wrapper below falls back to calling dsv4_hc_pre_fused_kernel (v1) for hc>4, so
// MEMRA_HC_FUSED_PRE=2 is always correct, only sometimes faster than =1.
extern "C" __global__ void dsv4_hc_pre_fused_v2_kernel(
        const float* __restrict__ x, const float* __restrict__ mixes_all,
        const float* __restrict__ scale, const float* __restrict__ base,
        float* __restrict__ pre_all, float* __restrict__ post_all,
        float* __restrict__ comb_all, float* __restrict__ y, int w, int rows, int hc, int d,
        int iters, float eps, int* __restrict__ niters) {
    int p = blockIdx.x;
    int t = threadIdx.x;
    int B = blockDim.x;

    // ---- stage 1: rowsq — VERBATIM dsv4_hc_pre_fused_kernel, unchanged.
    const float* xr = x + (long)p * w;
    double acc = 0.0;
    {
        int i = t;
        for (; i + 7 * B < w; i += 8 * B) {
            float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
            float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
            acc += (double)v0 * (double)v0;
            acc += (double)v1 * (double)v1;
            acc += (double)v2 * (double)v2;
            acc += (double)v3 * (double)v3;
            acc += (double)v4 * (double)v4;
            acc += (double)v5 * (double)v5;
            acc += (double)v6 * (double)v6;
            acc += (double)v7 * (double)v7;
        }
        for (; i < w; i += B) {
            double v = (double)xr[i];
            acc += v * v;
        }
    }
    __shared__ double shd[128];
    double tot = dsv4_block_sum(acc, shd);
    float rsq = 1.0f / sqrtf((float)(tot / (double)w) + eps);

    __shared__ float smix[(2 + DSV4_HC_MAX) * DSV4_HC_MAX];
    const float* mixes = mixes_all + (long)p * rows;
    for (int i = t; i < rows; i += B) smix[i] = mixes[i] * rsq;

    float* pre = pre_all + (long)p * hc;
    float* post = post_all + (long)p * hc;
    float* combg = comb_all + (long)p * hc * hc;
    __shared__ float spre[DSV4_HC_MAX];
    __shared__ float comb[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ float sprev[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ unsigned schanged;
    int done = 0;

    // ---- stage 2: Sinkhorn, WARP-0-ONLY (valid because the caller only reaches this
    // kernel when hc<=4 — see memra_dsv4_hc_pre_fused_v2). Every write and every read
    // below lives at shared index < 32.
    if (t < 32) {
        __syncwarp(); // smix writes above (by lanes < rows <= 24) visible to all 32 lanes
        if (t < hc) {
            float pv = dsv4_sigmoid(smix[t] * scale[0] + base[t]) + eps;
            pre[t] = pv;
            spre[t] = pv;
            post[t] = 2.0f * dsv4_sigmoid(smix[hc + t] * scale[1] + base[hc + t]);
        }
        if (t < hc * hc) comb[t] = smix[2 * hc + t] * scale[2] + base[2 * hc + t];
        __syncwarp();
        if (t < hc) {
            float* row = comb + t * hc;
            float mx = -INFINITY;
            for (int k = 0; k < hc; k++) mx = fmaxf(mx, row[k]);
            float sum = 0.0f;
            for (int k = 0; k < hc; k++) {
                row[k] = expf(row[k] - mx);
                sum += row[k];
            }
            for (int k = 0; k < hc; k++) row[k] = row[k] / sum + eps;
        }
        __syncwarp();
        for (int it = 0; it < iters; it++) {
            if (it > 0) {
                if (t < hc * hc) sprev[t] = comb[t];
                if (t == 0) schanged = 0u;
                __syncwarp();
                if (t < hc) {
                    float sum = 0.0f;
                    for (int k = 0; k < hc; k++) sum += comb[t * hc + k];
                    for (int k = 0; k < hc; k++) comb[t * hc + k] /= sum + eps;
                }
                __syncwarp();
            }
            if (t < hc) {
                float sum = 0.0f;
                for (int j = 0; j < hc; j++) sum += comb[j * hc + t];
                for (int j = 0; j < hc; j++) comb[j * hc + t] /= sum + eps;
            }
            __syncwarp();
            done = it + 1;
            if (it > 0) {
                unsigned ch = 0u;
                if (t < hc * hc)
                    ch = (unsigned)(__float_as_uint(sprev[t]) != __float_as_uint(comb[t]));
                if (ch) atomicOr(&schanged, 1u);
                __syncwarp();
                unsigned stop = (schanged == 0u);
                __syncwarp();
                if (stop) break; // bitwise-stationary: every later iteration is identity
            }
        }
        if (niters && t == 0) niters[p] = done;
        if (t < hc * hc) combg[t] = comb[t];
    }
    __syncthreads(); // cross-warp: stage 3 (full block) needs spre[] visible everywhere

    // ---- stage 3: collapse — VERBATIM dsv4_hc_pre_fused_kernel, unchanged.
    float* yr = y + (long)p * d;
    for (int i = t; i < d; i += B) {
        float acc2 = 0.0f;
        for (int c = 0; c < hc; c++) acc2 += spre[c] * xr[(long)c * d + i];
        yr[i] = acc2;
    }
}

extern "C" int memra_dsv4_hc_pre_fused_v2(const float* x, const float* mixes, const float* scale,
                                          const float* base, float* pre, float* post,
                                          float* comb, float* y, int s, int hc, int d,
                                          int iters, float eps, int* niters, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1 || hc < 1 || hc > DSV4_HC_MAX || d < 1 || iters < 1) return 40021;
    int w = hc * d;
    int rows = (2 + hc) * hc;
    if (rows > 32) {
        // hc>4: the warp-0-only invariant above does not hold. Fall back to v1 rather
        // than an unverified multi-warp scheme — see the kernel doc comment above.
        return memra_dsv4_hc_pre_fused(x, mixes, scale, base, pre, post, comb, y, s, hc, d,
                                       iters, eps, niters, stream_v);
    }
    dsv4_hc_pre_fused_v2_kernel<<<(unsigned)s, 128, 0, stream>>>(x, mixes, scale, base, pre, post,
                                                                 comb, y, w, rows, hc, d, iters,
                                                                 eps, niters);
    DSV4_ERR();
    return 0;
}

// ---- MoE routing, one block per position (dsv4_route_kernel's body verbatim on the

// =====================================================================================
// dsv4_hc_pre_fused_v3 — THE SAME KERNEL, GIVEN THE REST OF THE BLOCK.
// (door MEMRA_HC_PRE_BLOCK, lane/b200-hcpre-wide-20260903, default 128 = v2 verbatim)
// =====================================================================================
//
// WHY. `memra_dsv4_hc_pre_fused_v2` launches `<<<s, 128>>>`, one block per ROW. At t=1
// decode s is 1, so the whole call is ONE block of 128 threads on a 148-SM B200: 147 SMs
// idle. nsys on the 2x B200 pair, current best posture, 2026-09-03, measures it as the
// single largest kernel in the decode profile:
//
//   dsv4_hc_pre_fused_v2_kernel   17.5%   23,220 launches   31.1 us avg
//
// 23,220 launches over 256 profiled tokens is 90.7 per token, which is exactly 2 per layer
// across 45 layers (the attn site and the mlp site). So 90 x 31.1 us = 2.8 ms of an 18.4 ms
// token, 15% of the token, on 1/148th of the GPU.
//
// The kernel is not doing 31 us of work. It moves ~128 KB: stage 1 reads hc*d floats
// (4 x 4096 = 64 KB) and stage 3 reads them again to write d. 128 KB in 31 us is 4.1 GB/s.
// One block of 4 warps cannot hold enough loads in flight to cover HBM latency; the fix is
// warps, not arithmetic. Stage 2 is untouched by any of this: it is warp-0-only by the
// hc<=4 invariant its own comment states, and it stays exactly where it was.
//
// WHAT CHANGES, AND THE HONEST EXACTNESS STATEMENT. Stage 3 is bit-identical at any block
// size: each output element sums the same hc terms in the same order, and only WHICH thread
// computes it moves. Stage 1 is NOT: `dsv4_block_sum` reduces over blockDim.x, so a wider
// block gives each thread a different subset of the row and the double accumulation order
// changes. In practice the f32 narrowing of 1/sqrt(tot/w + eps) absorbs a last-ulp double
// difference, but that is an expectation, not a construction, so this is a NAMED NUMERIC
// CLASS `hc_pre_rowsq_blockwide` and it ships behind a door at default 128 (= v2's own
// partition, bit-identical) until an argmax gate and a greedy tape say otherwise.
#define DSV4_HC_PRE_V3_MAXBLOCK 1024

extern "C" __global__ void dsv4_hc_pre_fused_v3_kernel(
        const float* __restrict__ x, const float* __restrict__ mixes_all,
        const float* __restrict__ scale, const float* __restrict__ base,
        float* __restrict__ pre_all, float* __restrict__ post_all,
        float* __restrict__ comb_all, float* __restrict__ y, int w, int rows, int hc, int d,
        int iters, float eps, int* __restrict__ niters, int sink_reg) {
    int p = blockIdx.x;
    int t = threadIdx.x;
    int B = blockDim.x;

    // ---- stage 1: rowsq — VERBATIM dsv4_hc_pre_fused_kernel, unchanged.
    const float* xr = x + (long)p * w;
    double acc = 0.0;
    {
        int i = t;
        for (; i + 7 * B < w; i += 8 * B) {
            float v0 = xr[i], v1 = xr[i + B], v2 = xr[i + 2 * B], v3 = xr[i + 3 * B];
            float v4 = xr[i + 4 * B], v5 = xr[i + 5 * B], v6 = xr[i + 6 * B], v7 = xr[i + 7 * B];
            acc += (double)v0 * (double)v0;
            acc += (double)v1 * (double)v1;
            acc += (double)v2 * (double)v2;
            acc += (double)v3 * (double)v3;
            acc += (double)v4 * (double)v4;
            acc += (double)v5 * (double)v5;
            acc += (double)v6 * (double)v6;
            acc += (double)v7 * (double)v7;
        }
        for (; i < w; i += B) {
            double v = (double)xr[i];
            acc += v * v;
        }
    }
    __shared__ double shd[DSV4_HC_PRE_V3_MAXBLOCK];
    double tot = dsv4_block_sum(acc, shd);
    float rsq = 1.0f / sqrtf((float)(tot / (double)w) + eps);

    __shared__ float smix[(2 + DSV4_HC_MAX) * DSV4_HC_MAX];
    const float* mixes = mixes_all + (long)p * rows;
    for (int i = t; i < rows; i += B) smix[i] = mixes[i] * rsq;

    float* pre = pre_all + (long)p * hc;
    float* post = post_all + (long)p * hc;
    float* combg = comb_all + (long)p * hc * hc;
    __shared__ float spre[DSV4_HC_MAX];
    __shared__ float comb[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ float sprev[DSV4_HC_MAX * DSV4_HC_MAX];
    __shared__ unsigned schanged;
    int done = 0;

    // ---- stage 2R: THE SAME SINKHORN, IN REGISTERS (sink_reg != 0).
    //
    // WHY. nsys on 2x B200, 2026-09-03, measured this kernel at both block widths and the
    // split falls out of the two numbers: 128 threads -> 31.194 us, 1024 threads -> 26.609 us.
    // Stages 1 and 3 scale with the block; stage 2 does not (it is warp-0-only at every
    // width). Solving S + P = 31.194 and S + P/8 = 26.609 gives P = 5.24 us and
    // S = 25.95 us: the Sinkhorn is 83% of the kernel, which is 90 launches x 25.95 us =
    // 2.34 ms of an 18.44 ms token, 12.7% of the token, to normalise an hc x hc matrix
    // (16 floats at hc=4) for hc_sinkhorn_iters = 20 rounds.
    //
    // It is not arithmetic. Per round the shared path does ~2*hc dependent shared loads per
    // lane plus six __syncwarp and a shared atomicOr, on ONE warp with no other warp resident
    // to cover the latency — every dependent shared round trip is fully exposed.
    //
    // WHAT THIS DOES. comb lives one element per lane (lane l holds comb[l], l < hc*hc <= 16),
    // and every row/column sum is gathered with __shfl_sync IN THE SAME ORDER the shared loop
    // used. That is the whole exactness argument and it is why this is NOT a numeric class:
    // the shared path computes `for (k = 0; k < hc; ++k) sum += comb[t*hc+k]`, and this
    // computes `for (k = 0; k < hc; ++k) sum += __shfl_sync(mask, cv, r*hc+k)` — the same
    // addends, in the same sequence, into the same running float. A tree reduction would have
    // been fewer instructions and a different association; it is deliberately not used.
    //
    // Every lane of the warp executes every __shfl_sync (the mask is full and the shuffles sit
    // outside the `l < hc*hc` guard); lanes past the matrix carry a clamped index and a zero
    // value and never write. `niters`, `pre`, `post`, `spre` and `combg` keep their meanings.
    // MEASURED AND REMOVED: an ALL-REGISTER Sinkhorn arm (every lane holding the whole 4x4
    // matrix, no shuffles, no ballot) was built here and LOST on 2x B200 -- t=1 98 us against the
    // shuffle arm's 93, t=4 117 against 102, bit-identical throughout. The reasoning that produced
    // it ("the lanes are idle, so redundant compute is free") confused LATENCY with THROUGHPUT:
    // only warp 0 runs this stage either way, so holding all 16 elements per lane multiplies that
    // one warp's divides by hc*hc -- 640 per lane per call against 40 -- and the extra arithmetic
    // cancels the saved dependent latency. It did not spill (48 registers, ncu).
    //
    // And the premise was wrong anyway. Sweeping MEMRA_HC_GATE_ITERS against THIS kernel with
    // sink_reg=1 gives 90 us at 1 iteration and 91 at 40: the served Sinkhorn costs ~26 ns per
    // iteration, about 0.5 us of a 12.7 us kernel. The ~700 ns/iteration that motivated the arm
    // was the v2 kernel's SHARED-MEMORY Sinkhorn at block 128, a path this one does not take.
    // The cost here is stages 1 and 3 -- 130 KB moved on ONE block, ~11 GB/s, because the grid is
    // the sequence length and decode has s = 1. See TRAP:decode-kernel-launched-per-sequence-position.
    if (sink_reg && t < 32) {
        const unsigned MASK = 0xffffffffu;
        __syncwarp();
        if (t < hc) {
            float pv = dsv4_sigmoid(smix[t] * scale[0] + base[t]) + eps;
            pre[t] = pv;
            spre[t] = pv;
            post[t] = 2.0f * dsv4_sigmoid(smix[hc + t] * scale[1] + base[hc + t]);
        }
        int n2 = hc * hc;
        int r = (t < n2) ? t / hc : 0;
        int c = (t < n2) ? t - r * hc : 0;
        float cv = (t < n2) ? (smix[2 * hc + t] * scale[2] + base[2 * hc + t]) : 0.0f;
        __syncwarp();

        // initial row softmax — same max order, same post-exp accumulation order
        float mx = -INFINITY;
        for (int k = 0; k < hc; k++) mx = fmaxf(mx, __shfl_sync(MASK, cv, r * hc + k));
        float e = expf(cv - mx);
        float sum = 0.0f;
        for (int k = 0; k < hc; k++) sum += __shfl_sync(MASK, e, r * hc + k);
        cv = e / sum + eps;

        int done = 0;
        for (int it = 0; it < iters; it++) {
            float prev = cv;
            if (it > 0) {
                float rs = 0.0f;
                for (int k = 0; k < hc; k++) rs += __shfl_sync(MASK, cv, r * hc + k);
                cv = cv / (rs + eps);
            }
            float cs = 0.0f;
            for (int j = 0; j < hc; j++) cs += __shfl_sync(MASK, cv, j * hc + c);
            cv = cv / (cs + eps);
            done = it + 1;
            if (it > 0) {
                // bitwise-stationary, exactly the shared path's test, one ballot instead of a
                // shared atomicOr: every later iteration would be the identity.
                unsigned ch = (t < n2 && __float_as_uint(prev) != __float_as_uint(cv)) ? 1u : 0u;
                if (__any_sync(MASK, ch) == 0) break;
            }
        }
        if (niters && t == 0) niters[p] = done;
        if (t < n2) combg[t] = cv;
    }
    // ---- stage 2: Sinkhorn, WARP-0-ONLY (valid because the caller only reaches this
    // kernel when hc<=4 — see memra_dsv4_hc_pre_fused_v2). Every write and every read
    // below lives at shared index < 32. Skipped when the register path above ran.
    if (!sink_reg && t < 32) {
        __syncwarp(); // smix writes above (by lanes < rows <= 24) visible to all 32 lanes
        if (t < hc) {
            float pv = dsv4_sigmoid(smix[t] * scale[0] + base[t]) + eps;
            pre[t] = pv;
            spre[t] = pv;
            post[t] = 2.0f * dsv4_sigmoid(smix[hc + t] * scale[1] + base[hc + t]);
        }
        if (t < hc * hc) comb[t] = smix[2 * hc + t] * scale[2] + base[2 * hc + t];
        __syncwarp();
        if (t < hc) {
            float* row = comb + t * hc;
            float mx = -INFINITY;
            for (int k = 0; k < hc; k++) mx = fmaxf(mx, row[k]);
            float sum = 0.0f;
            for (int k = 0; k < hc; k++) {
                row[k] = expf(row[k] - mx);
                sum += row[k];
            }
            for (int k = 0; k < hc; k++) row[k] = row[k] / sum + eps;
        }
        __syncwarp();
        for (int it = 0; it < iters; it++) {
            if (it > 0) {
                if (t < hc * hc) sprev[t] = comb[t];
                if (t == 0) schanged = 0u;
                __syncwarp();
                if (t < hc) {
                    float sum = 0.0f;
                    for (int k = 0; k < hc; k++) sum += comb[t * hc + k];
                    for (int k = 0; k < hc; k++) comb[t * hc + k] /= sum + eps;
                }
                __syncwarp();
            }
            if (t < hc) {
                float sum = 0.0f;
                for (int j = 0; j < hc; j++) sum += comb[j * hc + t];
                for (int j = 0; j < hc; j++) comb[j * hc + t] /= sum + eps;
            }
            __syncwarp();
            done = it + 1;
            if (it > 0) {
                unsigned ch = 0u;
                if (t < hc * hc)
                    ch = (unsigned)(__float_as_uint(sprev[t]) != __float_as_uint(comb[t]));
                if (ch) atomicOr(&schanged, 1u);
                __syncwarp();
                unsigned stop = (schanged == 0u);
                __syncwarp();
                if (stop) break; // bitwise-stationary: every later iteration is identity
            }
        }
        if (niters && t == 0) niters[p] = done;
        if (t < hc * hc) combg[t] = comb[t];
    }
    __syncthreads(); // cross-warp: stage 3 (full block) needs spre[] visible everywhere

    // ---- stage 3: collapse — VERBATIM dsv4_hc_pre_fused_kernel, unchanged.
    float* yr = y + (long)p * d;
    for (int i = t; i < d; i += B) {
        float acc2 = 0.0f;
        for (int c = 0; c < hc; c++) acc2 += spre[c] * xr[(long)c * d + i];
        yr[i] = acc2;
    }
}

extern "C" int memra_dsv4_hc_pre_fused_v3(const float* x, const float* mixes,
                                          const float* scale, const float* base, float* pre,
                                          float* post, float* comb, float* y, int s, int hc,
                                          int d, int iters, float eps, int* niters, int block,
                                          int sink_reg, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1 || hc < 1 || hc > DSV4_HC_MAX || d < 1 || iters < 1) return 40021;
    // Power of two, at least one warp for the stage-2 invariant, at most the shared array.
    if (block < 32 || block > DSV4_HC_PRE_V3_MAXBLOCK || (block & (block - 1)) != 0) return 40023;
    int w = hc * d;
    int rows = (2 + hc) * hc;
    if (rows > 32) {
        // hc>4: the warp-0-only invariant does not hold, exactly as in v2. Fall back to v1
        // rather than an unverified multi-warp scheme.
        return memra_dsv4_hc_pre_fused(x, mixes, scale, base, pre, post, comb, y, s, hc, d,
                                       iters, eps, niters, stream_v);
    }
    // The register Sinkhorn addresses comb by LANE (lane l holds comb[l]), so it needs the
    // whole matrix inside one warp: hc*hc <= 32. At the hc <= 4 this launcher already enforces
    // that always holds, but it is checked rather than assumed, and a violation falls back to
    // the shared path instead of reading a lane that does not exist.
    int sr = (sink_reg && hc * hc <= 32) ? 1 : 0;
    dsv4_hc_pre_fused_v3_kernel<<<(unsigned)s, (unsigned)block, 0, stream>>>(
        x, mixes, scale, base, pre, post, comb, y, w, rows, hc, d, iters, eps, niters, sr);
    DSV4_ERR();
    return 0;
}
// position's own raw/sel/selw/order slices and its OWN token id — the hash layers'
// tid2eid row is per token, which is exactly why a round needs a token ARRAY).
extern "C" __global__ void dsv4_route_m_kernel(const float* __restrict__ raw_all,
                                               const float* __restrict__ bias,
                                               const int* __restrict__ tid2eid,
                                               const int* __restrict__ tok, int ne, int topk,
                                               float route_scale, int* __restrict__ sel_all,
                                               float* __restrict__ selw_all,
                                               int* __restrict__ order_all) {
    int p = blockIdx.x;
    const float* raw = raw_all + (long)p * ne;
    int* sel = sel_all + (long)p * topk;
    float* selw = selw_all + (long)p * topk;
    int* order = order_all + (long)p * topk;
    __shared__ float sc[256];
    __shared__ float bs[256];
    int t = threadIdx.x;
    for (int c = t; c < ne; c += blockDim.x) {
        float s = sqrtf(dsv4_softplus(raw[c]));
        sc[c] = s;
        bs[c] = (bias != nullptr) ? s + bias[c] : 0.0f;
    }
    __syncthreads();
    if (tid2eid == nullptr) {
        __shared__ float rv[128];
        __shared__ int ri[128];
        __shared__ unsigned long long mask[4];
        if (t < 4) mask[t] = 0ull;
        __syncthreads();
        for (int k = 0; k < topk; k++) {
            float bv = 0.0f;
            int bi = -1;
            for (int c = t; c < ne; c += blockDim.x) {
                if (mask[c >> 6] & (1ull << (c & 63))) continue;
                float v = bs[c];
                if (bi < 0 || v > bv) {
                    bv = v;
                    bi = c;
                }
            }
            rv[t] = bv;
            ri[t] = bi;
            __syncthreads();
            for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
                if (t < off) {
                    bool take = (ri[t + off] >= 0) &&
                                (ri[t] < 0 || rv[t + off] > rv[t] ||
                                 (rv[t + off] == rv[t] && ri[t + off] < ri[t]));
                    if (take) {
                        rv[t] = rv[t + off];
                        ri[t] = ri[t + off];
                    }
                }
                __syncthreads();
            }
            if (t == 0) {
                sel[k] = ri[0];
                mask[ri[0] >> 6] |= (1ull << (ri[0] & 63));
            }
            __syncthreads();
        }
    }
    if (t != 0) return;
    if (tid2eid != nullptr) {
        const int* row = tid2eid + (long)tok[p] * topk;
        for (int k = 0; k < topk; k++) sel[k] = row[k];
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        float w = sc[sel[k]];
        selw[k] = w;
        sum += w;
    }
    for (int k = 0; k < topk; k++) selw[k] = selw[k] / sum * route_scale;
    for (int k = 0; k < topk; k++) order[k] = k;
    for (int a = 1; a < topk; a++) {
        int o = order[a];
        int b = a - 1;
        while (b >= 0 && sel[order[b]] > sel[o]) {
            order[b + 1] = order[b];
            b--;
        }
        order[b + 1] = o;
    }
}

extern "C" int memra_dsv4_route_m(const float* raw, const float* bias, const int* tid2eid,
                                  const int* tok, int s, int ne, int topk, float route_scale,
                                  int* sel, float* selw, int* order, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (ne > 256 || topk > 32) return 40007;
    if (s < 1) return 40020;
    dsv4_route_m_kernel<<<(unsigned)s, 128, 0, stream>>>(raw, bias, tid2eid, tok, ne, topk,
                                                         route_scale, sel, selw, order);
    DSV4_ERR();
    return 0;
}

// ---- routed-expert combine, batched positions. Slot layout is [position][expert slot]
// (T*topk rows of `contrib`), `order` holds position-LOCAL slot indices, so position p
// sums contrib rows p*topk + order[p*topk + k] in ascending-expert-id order — the
// pinned kernel's sum order, per position.
extern "C" __global__ void dsv4_combine_rows_m_kernel(const float* __restrict__ contrib,
                                                      const int* __restrict__ order, int topk,
                                                      float* __restrict__ y, long d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y;
    if (i >= d) return;
    const int* orow = order + (long)p * topk;
    const float* crow = contrib + (long)p * topk * d;
    float acc = 0.0f;
    for (int k = 0; k < topk; k++) acc += crow[(long)orow[k] * d + i];
    y[(long)p * d + i] = acc;
}

extern "C" int memra_dsv4_combine_rows_m(const float* contrib, const int* order, int topk,
                                         float* y, long d, int s, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (s < 1) return 40020;
    int threads = 256;
    dim3 grid((unsigned)((d + threads - 1) / threads), (unsigned)s);
    dsv4_combine_rows_m_kernel<<<grid, threads, 0, stream>>>(contrib, order, topk, y, d);
    DSV4_ERR();
    return 0;
}

// ---- sink attention, decode shape, batched QUERIES (nq). Each query carries its own
// idx list (the §3.1 redirect makes them differ within a round), its own q/o rows and
// its own scores/evals/den slices. Bodies are the pinned trio's verbatim.
extern "C" __global__ void dsv4_sink_scores_mq_kernel(const float* __restrict__ q_all,
                                                      const float* __restrict__ kv,
                                                      const int* __restrict__ idxs_all,
                                                      float* __restrict__ scores_all,
                                                      int heads, int hd, int slots,
                                                      int idx_stride, float scale) {
    int sl = blockIdx.x;
    int p = blockIdx.y;
    if (sl >= slots) return;
    const int* idxs = idxs_all + (long)p * idx_stride;
    const float* q = q_all + (long)p * heads * hd;
    float* scores = scores_all + (long)p * heads * slots;
    int ix = idxs[sl];
    extern __shared__ float kvs[];
    if (ix < 0) {
        for (int h = threadIdx.x; h < heads; h += blockDim.x)
            scores[(long)h * slots + sl] = -INFINITY;
        return;
    }
    for (int x = threadIdx.x; x < hd; x += blockDim.x) kvs[x] = kv[(long)ix * hd + x];
    __syncthreads();
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const float* qv = q + (long)h * hd;
        double acc = 0.0;
        for (int x = 0; x < hd; x++) acc += (double)qv[x] * (double)kvs[x];
        scores[(long)h * slots + sl] = (float)acc * scale;
    }
}

extern "C" __global__ void dsv4_sink_soft_mq_kernel(const float* __restrict__ scores_all,
                                                    const float* __restrict__ sink,
                                                    float* __restrict__ evals_all,
                                                    double* __restrict__ den_all, int heads,
                                                    int slots) {
    int h = blockIdx.x;
    int p = blockIdx.y;
    const float* srow = scores_all + (long)p * heads * slots + (long)h * slots;
    float* erow = evals_all + (long)p * heads * slots + (long)h * slots;
    double* den = den_all + (long)p * heads;
    __shared__ float shred[128];
    float m = -INFINITY;
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) m = fmaxf(m, srow[sl]);
    m = dsv4_block_max(m, shred);
    m = fmaxf(m, -1e30f);
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x)
        erow[sl] = (srow[sl] == -INFINITY) ? 0.0f : expf(srow[sl] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        double d = 0.0;
        for (int sl = 0; sl < slots; sl++) d += (double)erow[sl];
        d += (double)expf(sink[h] - m);
        den[h] = d;
    }
}

extern "C" __global__ void dsv4_sink_out_mq_kernel(const float* __restrict__ kv,
                                                   const int* __restrict__ idxs_all,
                                                   const float* __restrict__ evals_all,
                                                   const double* __restrict__ den_all,
                                                   float* __restrict__ o_all, int heads,
                                                   int hd, int slots, int idx_stride) {
    const int XC = 8, HC = 8;
    int x0 = blockIdx.x * XC;
    int h0 = blockIdx.y * HC;
    int p = blockIdx.z;
    const int* idxs = idxs_all + (long)p * idx_stride;
    const float* evals = evals_all + (long)p * heads * slots;
    const double* den = den_all + (long)p * heads;
    float* o = o_all + (long)p * heads * hd;
    int tx = threadIdx.x % XC;
    int th = threadIdx.x / XC;
    int x = x0 + tx;
    int h = h0 + th;
    __shared__ float kvt[32 * XC];
    double acc = 0.0;
    for (int t0 = 0; t0 < slots; t0 += 32) {
        int tl = min(32, slots - t0);
        for (int i = threadIdx.x; i < tl * XC; i += blockDim.x) {
            int sl = t0 + i / XC;
            int xx = x0 + i % XC;
            int ix = idxs[sl];
            kvt[i] = (ix < 0 || xx >= hd) ? 0.0f : kv[(long)ix * hd + xx];
        }
        __syncthreads();
        if (x < hd && h < heads) {
            const float* erow = evals + (long)h * slots;
            for (int i = 0; i < tl; i++) {
                float ev = erow[t0 + i];
                if (ev == 0.0f) continue;
                acc += (double)ev * (double)kvt[i * XC + tx];
            }
        }
        __syncthreads();
    }
    if (x < hd && h < heads) o[(long)h * hd + x] = (float)(acc / den[h]);
}

// C4-only residency rewrite. The indexer's logical row order is preserved exactly;
// only its addresses change. One row is fetched once and reused by all query heads.
// Host reads are volatile so rollback/re-emission cannot reuse stale cached bytes.
__global__ void dsv4_c4_gather_kernel(
    const float* device, const float* host, const int* indices,
    float* out, int* out_indices, int slots, int stride, int live_rows,
    int logical_transient, int transient_rows) {
    const int slot = blockIdx.x, q = blockIdx.y, lane = threadIdx.x;
    const int index = indices[q * stride + slot];
    const float* source = nullptr;
    if (index >= 0 && index < 128) source = device + (long)index * 512;
    else if (index >= 128 && index < 128 + live_rows)
        source = host + (long)(index - 128) * 512;
    else if (index >= logical_transient && index < logical_transient + transient_rows)
        source = device + (long)(128 + index - logical_transient) * 512;
    else assert(index == -1); // invalid positive indices must not silently become pads
    const int row = q * slots + slot;
    if (lane == 0) out_indices[q * stride + slot] = index == -1 ? -1 : row;
    auto dst = reinterpret_cast<unsigned*>(out + (long)row * 512);
    auto src = reinterpret_cast<const volatile unsigned*>(source);
    for (int x = lane; x < 512; x += blockDim.x) dst[x] = source ? src[x] : 0;
}

extern "C" int memra_dsv4_c4_gather(
    const float* device, const float* host, const int* indices,
    float* out, int* out_indices, int nq, int slots, int stride, int live_rows,
    int logical_transient, int transient_rows, void* stream_v) {
    if (!device || !host || !indices || !out || !out_indices || nq < 1 || nq > 512
        || slots < 1 || slots > 640 || stride < slots || live_rows < 0
        || logical_transient < 128 + live_rows || transient_rows < 0) return 40010;
    dsv4_c4_gather_kernel<<<dim3(slots, nq), 128, 0, (cudaStream_t)stream_v>>>(
        device, host, indices, out, out_indices, slots, stride, live_rows,
        logical_transient, transient_rows);
    DSV4_ERR();
    return 0;
}

extern "C" int memra_dsv4_sink_attn_dec_mq(const float* q, const float* kv, const int* idxs,
                                           const float* sink, float* scores, float* evals,
                                           double* den, float* o, int nq, int heads, int hd,
                                           int slots, int idx_stride, float scale,
                                           void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (slots <= 0 || nq < 1) return 40010;
    dim3 g1((unsigned)slots, (unsigned)nq);
    dsv4_sink_scores_mq_kernel<<<g1, 64, (size_t)hd * sizeof(float), stream>>>(
        q, kv, idxs, scores, heads, hd, slots, idx_stride, scale);
    DSV4_ERR();
    dim3 g2((unsigned)heads, (unsigned)nq);
    dsv4_sink_soft_mq_kernel<<<g2, 128, 0, stream>>>(scores, sink, evals, den, heads, slots);
    DSV4_ERR();
    dim3 g3((unsigned)((hd + 7) / 8), (unsigned)((heads + 7) / 8), (unsigned)nq);
    dsv4_sink_out_mq_kernel<<<g3, 64, 0, stream>>>(kv, idxs, evals, den, o, heads, hd, slots,
                                                   idx_stride);
    DSV4_ERR();
    return 0;
}

// ---- sink attention, decode shape, batched queries, f32-accumulation arm (f32x).
extern "C" __global__ void dsv4_sink_scores_mq_f32acc_kernel(const float* __restrict__ q_all,
                                                             const float* __restrict__ kv,
                                                             const int* __restrict__ idxs_all,
                                                             float* __restrict__ scores_all,
                                                             int heads, int hd, int slots,
                                                             int idx_stride, float scale) {
    int sl = blockIdx.x;
    int p = blockIdx.y;
    if (sl >= slots) return;
    const int* idxs = idxs_all + (long)p * idx_stride;
    const float* q = q_all + (long)p * heads * hd;
    float* scores = scores_all + (long)p * heads * slots;
    int ix = idxs[sl];
    extern __shared__ float kvs[];
    if (ix < 0) {
        for (int h = threadIdx.x; h < heads; h += blockDim.x)
            scores[(long)h * slots + sl] = -INFINITY;
        return;
    }
    for (int x = threadIdx.x; x < hd; x += blockDim.x) kvs[x] = kv[(long)ix * hd + x];
    __syncthreads();
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const float* qv = q + (long)h * hd;
        float acc = 0.0f;
        for (int x = 0; x < hd; x++) acc += qv[x] * kvs[x];
        scores[(long)h * slots + sl] = acc * scale;
    }
}

extern "C" __global__ void dsv4_sink_soft_mq_f32acc_kernel(const float* __restrict__ scores_all,
                                                           const float* __restrict__ sink,
                                                           float* __restrict__ evals_all,
                                                           float* __restrict__ den_all,
                                                           int heads, int slots) {
    int h = blockIdx.x;
    int p = blockIdx.y;
    const float* srow = scores_all + (long)p * heads * slots + (long)h * slots;
    float* erow = evals_all + (long)p * heads * slots + (long)h * slots;
    float* den = den_all + (long)p * heads;
    __shared__ float shred[128];
    float m = -INFINITY;
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x) m = fmaxf(m, srow[sl]);
    m = dsv4_block_max(m, shred);
    m = fmaxf(m, -1e30f);
    for (int sl = threadIdx.x; sl < slots; sl += blockDim.x)
        erow[sl] = (srow[sl] == -INFINITY) ? 0.0f : expf(srow[sl] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        float d = 0.0f;
        for (int sl = 0; sl < slots; sl++) d += erow[sl];
        d += expf(sink[h] - m);
        den[h] = d;
    }
}

extern "C" __global__ void dsv4_sink_out_mq_f32acc_kernel(const float* __restrict__ kv,
                                                          const int* __restrict__ idxs_all,
                                                          const float* __restrict__ evals_all,
                                                          const float* __restrict__ den_all,
                                                          float* __restrict__ o_all, int heads,
                                                          int hd, int slots, int idx_stride) {
    const int XC = 8, HC = 8;
    int x0 = blockIdx.x * XC;
    int h0 = blockIdx.y * HC;
    int p = blockIdx.z;
    const int* idxs = idxs_all + (long)p * idx_stride;
    const float* evals = evals_all + (long)p * heads * slots;
    const float* den = den_all + (long)p * heads;
    float* o = o_all + (long)p * heads * hd;
    int tx = threadIdx.x % XC;
    int th = threadIdx.x / XC;
    int x = x0 + tx;
    int h = h0 + th;
    __shared__ float kvt[32 * XC];
    float acc = 0.0f;
    for (int t0 = 0; t0 < slots; t0 += 32) {
        int tl = min(32, slots - t0);
        for (int i = threadIdx.x; i < tl * XC; i += blockDim.x) {
            int sl = t0 + i / XC;
            int xx = x0 + i % XC;
            int ix = idxs[sl];
            kvt[i] = (ix < 0 || xx >= hd) ? 0.0f : kv[(long)ix * hd + xx];
        }
        __syncthreads();
        if (x < hd && h < heads) {
            const float* erow = evals + (long)h * slots;
            for (int i = 0; i < tl; i++) {
                float ev = erow[t0 + i];
                if (ev == 0.0f) continue;
                acc += ev * kvt[i * XC + tx];
            }
        }
        __syncthreads();
    }
    if (x < hd && h < heads) o[(long)h * hd + x] = acc / den[h];
}

extern "C" int memra_dsv4_sink_attn_dec_mq_f32acc(const float* q, const float* kv,
                                                  const int* idxs, const float* sink,
                                                  float* scores, float* evals, float* den,
                                                  float* o, int nq, int heads, int hd,
                                                  int slots, int idx_stride, float scale,
                                                  void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (slots <= 0 || nq < 1) return 40010;
    dim3 g1((unsigned)slots, (unsigned)nq);
    dsv4_sink_scores_mq_f32acc_kernel<<<g1, 64, (size_t)hd * sizeof(float), stream>>>(
        q, kv, idxs, scores, heads, hd, slots, idx_stride, scale);
    DSV4_ERR();
    dim3 g2((unsigned)heads, (unsigned)nq);
    dsv4_sink_soft_mq_f32acc_kernel<<<g2, 128, 0, stream>>>(scores, sink, evals, den, heads,
                                                            slots);
    DSV4_ERR();
    dim3 g3((unsigned)((hd + 7) / 8), (unsigned)((heads + 7) / 8), (unsigned)nq);
    dsv4_sink_out_mq_f32acc_kernel<<<g3, 64, 0, stream>>>(kv, idxs, evals, den, o, heads, hd,
                                                          slots, idx_stride);
    DSV4_ERR();
    return 0;
}

// ---- row scatter: dst[dst_rows[i], :] = src[i, :] for i in [0, n) (verify-round commit
// of the transient window-kv rows into their ring slots; one launch, no host loop, and
// the source rows are disjoint from the destinations by the ring-hazard construction).
extern "C" __global__ void dsv4_scatter_rows_kernel(const float* __restrict__ src,
                                                    float* __restrict__ dst,
                                                    const int* __restrict__ dst_rows, int n,
                                                    int d) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)n * d) return;
    int r = (int)(i / d);
    int c = (int)(i % d);
    dst[(long)dst_rows[r] * d + c] = src[(long)r * d + c];
}

extern "C" int memra_dsv4_scatter_rows(const float* src, float* dst, const int* dst_rows,
                                       int n, int d, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (n < 1) return 40020;
    long tot = (long)n * d;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    dsv4_scatter_rows_kernel<<<(unsigned)blocks, threads, 0, stream>>>(src, dst, dst_rows, n,
                                                                       d);
    DSV4_ERR();
    return 0;
}

// =============================================== iteration-5: NVTX phase shim (F itemisation)
// The drafted round costs F + 0.272*T plain steps and the whole drafted gap is F, so F has to
// be itemised before it can be attacked. These two launchers let the Rust round driver name
// its phases; `nsys -t cuda,nvtx` then reports GPU-busy per phase (nvtx_gpu_proj_sum) and host
// wall per phase (nvtx_sum), whose DIFFERENCE is the exposed launch/sync stall.
// NVTX v3 is header-only and push/pop cost a table-indirect no-op when no tool is attached,
// so this is safe to leave compiled in; the Rust side additionally gates it on an env knob.
extern "C" int memra_dsv4_nvtx_push(const char *name) {
#ifndef MEMRA_DSV4_HAVE_NVTX
    (void)name;
    return 0;
#else
    nvtxRangePushA(name);
    return 0;
#endif
}
extern "C" int memra_dsv4_nvtx_pop() {
#ifndef MEMRA_DSV4_HAVE_NVTX
    return 0;
#else
    nvtxRangePop();
    return 0;
#endif
}

// ===================================== iteration-5: gather one row by a DEVICE-RESIDENT index
// The DSpark markov chain needs row `idx[slot]` of markov_w1, where `idx[slot]` is the argmax
// the previous chain step just wrote on the device. The shipped path read that index back to
// the host (4-byte D2H + a full stream drain, five times a round) purely to compute the source
// offset of a memcpy. Doing the indirection on the device removes the drain and copies the
// SAME bytes, so the chain stays bit-identical.
extern "C" __global__ void dsv4_gather_row_by_idx_kernel(const float *__restrict__ src,
                                                        const int *__restrict__ idx, int slot,
                                                        float *__restrict__ dst, int cols) {
    long long r = (long long)idx[slot];
    for (long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x; c < (long long)cols;
         c += (long long)blockDim.x * gridDim.x) {
        dst[c] = src[r * (long long)cols + c];
    }
}

extern "C" int memra_dsv4_gather_row_by_idx(const float *src, const int *idx, int slot,
                                           float *dst, int cols, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    dsv4_gather_row_by_idx_kernel<<<(unsigned)blocks, threads, 0, stream>>>(src, idx, slot, dst,
                                                                           cols);
    DSV4_ERR();
    return 0;
}

// ============================== iteration-5: row-blocked twin of dsv4_dots_f32_kernel (BIT-EXACT)
//
// `dsv4_dots_f32_kernel` puts ONE BLOCK PER OUTPUT ROW. For the DSpark markov bias GEMV
// (n = vocab = 129,280, k = rank = 256) that is 129,280 blocks each reading 1 KB and then paying a
// 7-level __syncthreads halving tree -- measured 318 us, 416 GB/s, 26% of roofline, i.e. bound by
// per-block tree LATENCY, not by bandwidth.
//
// This twin changes ONLY the block shape: R rows per block, blockDim = (128, R), row r owned by
// threadIdx.y == r with its own 128-double smem slab. The per-thread strided accumulation over
// threadIdx.x, the leaf count, and the halving tree are the SAME, so every output is
// bit-identical to the original -- only the launch geometry moved. Out-of-range rows still walk
// the tree with a zero leaf: `dsv4_block_sum` contains __syncthreads(), so every thread of the
// block must reach every barrier.
template <int R>
__global__ void dsv4_dots_f32_rowblk_kernel(const float *__restrict__ x,
                                            const void *__restrict__ w, int w_is_bf16,
                                            float *__restrict__ y, int s, int k, int n) {
    const int j = blockIdx.x * R + (int)threadIdx.y;
    const int t = blockIdx.y;
    extern __shared__ double shd_rb[];
    double *sh = shd_rb + (long)threadIdx.y * blockDim.x;
    double acc = 0.0;
    if (j < n && t < s) {
        const float *xr = x + (long)t * k;
        if (w_is_bf16) {
            const uint16_t *wr = (const uint16_t *)w + (long)j * k;
            for (int i = threadIdx.x; i < k; i += blockDim.x)
                acc += (double)xr[i] * (double)__uint_as_float(((unsigned)wr[i]) << 16);
        } else {
            const float *wr = (const float *)w + (long)j * k;
            for (int i = threadIdx.x; i < k; i += blockDim.x)
                acc += (double)xr[i] * (double)wr[i];
        }
    }
    double tot = dsv4_block_sum(acc, sh);
    if (threadIdx.x == 0 && j < n && t < s) y[(long)t * n + j] = (float)tot;
}

extern "C" int memra_dsv4_dots_f32_rowblk(const float *x, const void *w, int w_is_bf16, float *y,
                                          int s, int k, int n, void *stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int R = 8, TH = 128;
    dim3 block((unsigned)TH, (unsigned)R);
    dim3 grid((unsigned)((n + R - 1) / R), (unsigned)s);
    size_t sm = (size_t)R * TH * sizeof(double);
    dsv4_dots_f32_rowblk_kernel<R><<<grid, block, sm, stream>>>(x, w, w_is_bf16, y, s, k, n);
    DSV4_ERR();
    return 0;
}
