// f16_prefill.cu — MEMRA_PP_F16 prefill GEMM: cuBLASLt FP16 TN on a resident fp16 dequant
// mirror of Q8_0 weights.
//
// Provenance: tools/bench_lt_f16.cu probe (2026-07-26, H100 box): 611-687 TF at the 9B m=512
// prefill shapes vs the vendored MMQ class those Q8_0 weights ride (253/168/90/247/236/82us
// per-shape medians) = 3.2-3.7x per launch. The exact-int8 wgmma arc (v0-v4 + ceiling probe,
// ARCHITECTURE-H100.md) established WHY a fold-free fp16 GEMM is the right swing: Q8_0's
// per-32-block scale fold serializes Hopper's warpgroup MMA pipe (ptxas C7514); fp16 with f32
// accumulate has zero mid-loop accumulator reads and streams at full tensor-core rate.
//
// NUMERIC CONFIG (new, explicit, gated — GDN-chunked/MEMRA_PP_FP8 precedent): weight int8 -> fp16
// dequant is EXACT for the int8 part (|q|<=127 needs 7 mantissa bits; fp16 has 11) — the only
// rounding is d(half)*q products vs the s32-exact-then-f32-fold law, plus activation f32->fp16
// (rel ~2^-11 per element, NO per-32 rescale — finer-grained than q8_1's int8 in that sense).
// Decode (m=1..8) NEVER reaches this path — the decode==verify law is untouched.
//
// Host C-ABI (called from f16_ffi.rs), mirrors fp8_prefill.cu exactly:
//   memra_f16_pp_gemm : f32->fp16 activation convert + cublasLtMatmul TN, all on one stream,
//                      (m,n,k)-cached descriptors + heuristic algo under a mutex.
//   memra_q8_0_dequant_f16 : GGUF Q8_0 34B blocks -> row-major fp16 mirror (load-time, per tensor).

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

// ---- device kernels -------------------------------------------------------------------------

// f32 -> fp16 elementwise (activations; fp16 max 65504 >> activation range, no scale needed).
// float4/half4-vectorized grid-stride (elementwise -> bit-identical; H100 sweep 2026-07-26).
extern "C" __global__ void memra_f16_cvt_kernel(const float* __restrict__ x,
                                               __half* __restrict__ o, size_t n) {
    size_t n4 = n / 4;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = *(const float4*)(x + i * 4);
        __half2 lo = __floats2half2_rn(v.x, v.y);
        __half2 hi = __floats2half2_rn(v.z, v.w);
        *(__half2*)(o + i * 4) = lo;
        *(__half2*)(o + i * 4 + 2) = hi;
    }
    // tail (n % 4) by the first threads
    size_t t0 = n4 * 4;
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (tid < n - t0) o[t0 + tid] = __float2half(x[t0 + tid]);
}

// GGUF Q8_0 (34B blocks: half d + 32 int8) -> row-major fp16. One thread per 32-block.
extern "C" __global__ void memra_q8f16_dequant_kernel(const unsigned char* __restrict__ src,
                                                     __half* __restrict__ dst,
                                                     size_t nblk_total, int nblk_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nblk_total) return;
    const unsigned char* b = src + i * 34;
    float d = __half2float(*(const __half*)b);
    const signed char* q = (const signed char*)(b + 2);
    __half* o = dst + i * 32;   // block i = row (i/nblk_row), k-block (i%nblk_row): dst is dense
    #pragma unroll
    for (int k = 0; k < 32; k++) o[k] = __float2half(d * (float)q[k]);
}

extern "C" int memra_q8_0_dequant_f16(const void* w_q8, void* w_f16, long out_f, long nblk_row,
                                     void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nblk_total = (size_t)out_f * (size_t)nblk_row;
    int threads = 256;
    size_t blocks = (nblk_total + threads - 1) / threads;
    memra_q8f16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q8, (__half*)w_f16, nblk_total, (int)nblk_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// GGUF Q4_0 (18B blocks: half d + 16 nibble bytes) -> row-major fp16 (campaign A, 2026-07-31:
// the gemma QAT trunk's f16 mirror — int4 magnitudes are exact in fp16, rounding only at d*q;
// same accuracy class as the promoted Q8_0 mirror). One thread per 32-block.
extern "C" __global__ void memra_q4f16_dequant_kernel(const unsigned char* __restrict__ src,
                                                     __half* __restrict__ dst,
                                                     size_t nblk_total, int nblk_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nblk_total) return;
    const unsigned char* b = src + i * 18;
    float d = __half2float(*(const __half*)b);
    const unsigned char* qs = b + 2;
    __half* o = dst + i * 32;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        const int lo = (qs[k] & 0x0F) - 8;
        const int hi = (qs[k] >> 4) - 8;
        o[k]      = __float2half(d * (float)lo);
        o[k + 16] = __float2half(d * (float)hi);
    }
}

extern "C" int memra_q4_0_dequant_f16(const void* w_q4, void* w_f16, long out_f, long nblk_row,
                                     void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nblk_total = (size_t)out_f * (size_t)nblk_row;
    int threads = 256;
    size_t blocks = (nblk_total + threads - 1) / threads;
    memra_q4f16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q4, (__half*)w_f16, nblk_total, (int)nblk_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// GGUF Q6_K (210B superblocks: ql[128] qh[64] i8 scales[16] fp16 d, 256 vals) -> row-major
// fp16 (round 47: the q27 Q4_K_M mix packs attn_v/ffn_down/head as Q6_K — its 6.7ms/call
// dequant-GEMMs were the prefill wall; no Q6_K MMQ exists in the vendored set). Indexing
// verified against qmatvec.cu's q6_K decode (line ~144). One thread per superblock
// (load-time only).
extern "C" __global__ void memra_q6kf16_dequant_kernel(const unsigned char* __restrict__ src,
                                                       __half* __restrict__ dst,
                                                       size_t nsb_total, int nsb_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nsb_total) return;
    const unsigned char* b = src + i * 210;
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char* sc = (const signed char*)(b + 192);
    float d = __half2float(*(const __half*)(b + 208));
    __half* o = dst + i * 256;
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            int is = l >> 4;
            int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            o[l +  0] = __float2half(d * (float)sc[is + 0] * (float)q1);
            o[l + 32] = __float2half(d * (float)sc[is + 2] * (float)q2);
            o[l + 64] = __float2half(d * (float)sc[is + 4] * (float)q3);
            o[l + 96] = __float2half(d * (float)sc[is + 6] * (float)q4);
        }
        o += 128; ql += 64; qh += 32; sc += 8;
    }
}

extern "C" int memra_q6_K_dequant_f16(const void* w_q6, void* w_f16, long out_f, long nsb_row,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nsb_total = (size_t)out_f * (size_t)nsb_row;
    int threads = 256;
    size_t blocks = (nsb_total + threads - 1) / threads;
    memra_q6kf16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q6, (__half*)w_f16, nsb_total, (int)nsb_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// GGUF Q4_K (144B superblocks: fp16 d + fp16 dmin + u8 scales[12] (6-bit packed) + u8 qs[128],
// 256 vals) -> row-major fp16 (round 49: the q27 trunk BULK — 294 Q4_K tensors ride
// mul_mat_q_q45k int8-MMA; the Lt f16 lane beats that class at large m, campaign-A Q4_0
// precedent). value = d*sc*q - dmin*mn per 32-group; unpack verified against qmatvec.cu's
// deq_q4_k / q4k_scale_min (get_scale_min_k4 6-bit packing, line ~111): groups 0-3 read
// sc[j]&63 / sc[j+4]&63, groups 4-7 splice the high 2 bits of sc[j-4]/sc[j] above the low
// nibble of sc[j+4]. Each 64-val chunk shares 32 qs bytes: even group = low nibble, odd =
// high. One thread per superblock (load-time only).
extern "C" __global__ void memra_q4kf16_dequant_kernel(const unsigned char* __restrict__ src,
                                                       __half* __restrict__ dst,
                                                       size_t nsb_total, int nsb_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nsb_total) return;
    const unsigned char* b = src + i * 144;
    float d = __half2float(*(const __half*)b);
    float dmin = __half2float(*(const __half*)(b + 2));
    const unsigned char* sc = b + 4;
    const unsigned char* qs = b + 16;
    __half* o = dst + i * 256;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        unsigned char s8, m8;
        if (g < 4) { s8 = sc[g] & 63; m8 = sc[g + 4] & 63; }
        else {
            s8 = (sc[g + 4] & 0xF) | ((sc[g - 4] >> 6) << 4);
            m8 = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
        }
        const unsigned char* q = qs + (g >> 1) * 32;
        float dl = d * (float)s8, ml = dmin * (float)m8;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            int v = (g & 1) ? (q[l] >> 4) : (q[l] & 0xF);
            o[g * 32 + l] = __float2half(dl * (float)v - ml);
        }
    }
}

extern "C" int memra_q4_K_dequant_f16(const void* w_q4k, void* w_f16, long out_f, long nsb_row,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nsb_total = (size_t)out_f * (size_t)nsb_row;
    int threads = 256;
    size_t blocks = (nsb_total + threads - 1) / threads;
    memra_q4kf16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q4k, (__half*)w_f16, nsb_total, (int)nsb_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// GGUF Q5_K (176B superblocks: fp16 d + fp16 dmin + u8 scales[12] (6-bit, same
// get_scale_min_k4 as Q4_K) + u8 qh[32] + u8 ql[128], 256 vals) -> row-major fp16
// (round 49b: q27's 48 ssm_out projections — the last mul_mat_q_q45k prefill class).
// value = d*sc*(nib | qh_bit<<4) - dmin*mn; unpack verified against qmatvec.cu's
// deq_q5_k (line ~213): the qh bit index for group g, lane l is simply bit g of qh[l].
// One thread per superblock (load-time only).
extern "C" __global__ void memra_q5kf16_dequant_kernel(const unsigned char* __restrict__ src,
                                                       __half* __restrict__ dst,
                                                       size_t nsb_total, int nsb_row) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nsb_total) return;
    const unsigned char* b = src + i * 176;
    float d = __half2float(*(const __half*)b);
    float dmin = __half2float(*(const __half*)(b + 2));
    const unsigned char* sc = b + 4;
    const unsigned char* qh = b + 16;
    const unsigned char* ql = b + 48;
    __half* o = dst + i * 256;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        unsigned char s8, m8;
        if (g < 4) { s8 = sc[g] & 63; m8 = sc[g + 4] & 63; }
        else {
            s8 = (sc[g + 4] & 0xF) | ((sc[g - 4] >> 6) << 4);
            m8 = (sc[g + 4] >> 4) | ((sc[g] >> 6) << 4);
        }
        const unsigned char* q = ql + (g >> 1) * 32;
        float dl = d * (float)s8, ml = dmin * (float)m8;
        #pragma unroll
        for (int l = 0; l < 32; l++) {
            int nib = (g & 1) ? (q[l] >> 4) : (q[l] & 0xF);
            int w = nib | (((qh[l] >> g) & 1) << 4);
            o[g * 32 + l] = __float2half(dl * (float)w - ml);
        }
    }
}

extern "C" int memra_q5_K_dequant_f16(const void* w_q5k, void* w_f16, long out_f, long nsb_row,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    size_t nsb_total = (size_t)out_f * (size_t)nsb_row;
    int threads = 256;
    size_t blocks = (nsb_total + threads - 1) / threads;
    memra_q5kf16_dequant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const unsigned char*)w_q5k, (__half*)w_f16, nsb_total, (int)nsb_row);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// ---- host: cached cuBLASLt plans (fp8_prefill.cu pattern) ------------------------------------


// Per-device cuBLASLt handles. A cublasLtHandle_t is bound to the device that was current at
// cublasLtCreate; using it with another device current returned CUBLAS_STATUS_EXECUTION_FAILED
// (13) on every bf16 GEMM issued from PP stage 1 on a 2x B200 SXM pair (2026-09-02, lane
// glm5-b200), while the SM120 PP pairs tolerated the shared handle. Each device gets its own
// handle, created lazily with that device current; plan caches carry the device in their key.
static const int kMemraMaxDevices = 64;
static inline int memra_lt_current_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMemraMaxDevices) return 0;
    return dev;
}
static inline cublasStatus_t memra_lt_handle_for_device(cublasLtHandle_t* slots, int dev,
                                                        cublasLtHandle_t* out) {
    if (!slots[dev]) {
        cublasStatus_t s = cublasLtCreate(&slots[dev]);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    *out = slots[dev];
    return CUBLAS_STATUS_SUCCESS;
}

namespace {
struct F16Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulAlgo_t algo;
};
std::mutex g_mu16;
cublasLtHandle_t g_lt16_dev[kMemraMaxDevices] = {};
std::map<std::tuple<int, int, int, int>, F16Plan>* g_plans16 = nullptr;  // leaked (process-lifetime)
}  // namespace

// Standalone f32->fp16 activation convert (grouped-dispatch entry: hybrid layers run 2-4
// GEMMs on ONE activation — convert once, feed memra_f16_pp_gemm_pre per weight).
extern "C" int memra_f16_cvt(const float* x_f32, void* xh_f16, size_t nelem, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = 256;
    size_t want = (nelem + threads - 1) / threads;
    int blocks = (int)(want < 1024 ? want : 1024);
    if (blocks < 1) blocks = 1;
    memra_f16_cvt_kernel<<<blocks, threads, 0, stream>>>(x_f32, (__half*)xh_f16, nelem);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

// One FP16 prefill GEMM: y[m,n] row-major = x[m,k] @ W[n,k]^T, f32 accumulate/output.
// Col-major view: D[n,m] = A^T(W as k x n, opT) * B(xh as k x m, opN), lda=ldb=k, ldd=n.
// Returns 0 ok; 1xxxx = cudaError from the convert; 2xxxx = no heuristic; 3xxxx = matmul status.
extern "C" int memra_f16_pp_gemm_pre(
    const void* w_f16,      // device [n, k] row-major fp16 mirror
    const void* xh_f16,     // device [m, k] fp16 activation (pre-converted)
    float* y_f32,           // device out [m, n] row-major f32 (fully overwritten)
    int m, int n, int k,
    void* ws, size_t ws_bytes,
    void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> guard(g_mu16);
    const int dev = memra_lt_current_device();
    cublasLtHandle_t g_lt16 = nullptr;
    {
        cublasStatus_t s = memra_lt_handle_for_device(g_lt16_dev, dev, &g_lt16);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
    }
    if (!g_plans16) g_plans16 = new std::map<std::tuple<int, int, int, int>, F16Plan>();
    auto key = std::make_tuple(dev, m, n, k);
    auto it = g_plans16->find(key);
    if (it == g_plans16->end()) {
        F16Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, k, n, k);  // W: k x n col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, k, m, k);  // act: k x m col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, n, m, n);  // out: n x m col-major, ld=n
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(g_lt16, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_plans16->emplace(key, p).first;
    }
    F16Plan& plan = it->second;
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(g_lt16, plan.op, &alpha, w_f16, plan.la, xh_f16, plan.lb,
                                      &beta, y_f32, plan.ld, y_f32, plan.ld, &plan.algo,
                                      ws, ws_bytes, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}

// Combined convert + GEMM (the single-consumer path and the kernel-check raw entry).
extern "C" int memra_f16_pp_gemm(
    const void* w_f16, const float* x_f32, void* xh_f16, float* y_f32,
    int m, int n, int k, void* ws, size_t ws_bytes, void* stream_v) {
    int rc = memra_f16_cvt(x_f32, xh_f16, (size_t)m * (size_t)k, stream_v);
    if (rc != 0) return rc;
    return memra_f16_pp_gemm_pre(w_f16, xh_f16, y_f32, m, n, k, ws, ws_bytes, stream_v);
}

// ---- BF16 RESIDENT PREFILL (2026-08-28, step37 prime) ----------------------------------------
// The step37 trunk is BF16 in the checkpoint, so it has no Q8_0 fp16 mirror and every prefill
// projection fell to linear_bf16_chunked_inner: dequant the WHOLE weight to f32, then an f32
// (non-tensor-core) GEMM. Measured: ~1,000 tok/s prefill against vLLM's ~15,000 on the same
// class of card. cuBLASLt speaks CUDA_R_16BF natively and the checkpoint bytes are ALREADY the
// k x n col-major operand the TN form wants, so the fix needs no mirror and no extra VRAM:
// convert the activation once, hand the resident weight straight to the tensor cores.
//
// NUMERIC CONFIG: weight bytes are untouched (bit-identical to the checkpoint — strictly closer
// to the reference than the f32-dequant path, which round-trips through f32). Rounding enters
// only at the activation f32->bf16 cast (8 mantissa bits) and is bounded by the same
// argmax/exactness gates as every other prefill GEMM door. Decode (m<16) never reaches here.

extern "C" __global__ void memra_bf16_cvt_kernel(const float* __restrict__ x,
                                                 __nv_bfloat16* __restrict__ o, size_t n) {
    size_t n4 = n / 4;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = *(const float4*)(x + i * 4);
        __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
        __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
        *(__nv_bfloat162*)(o + i * 4) = lo;
        *(__nv_bfloat162*)(o + i * 4 + 2) = hi;
    }
    size_t t0 = n4 * 4;
    for (size_t i = t0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        o[i] = __float2bfloat16(x[i]);
    }
}

extern "C" int memra_bf16_cvt(const float* x_f32, void* xb_bf16, size_t nelem, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    const int threads = 256;
    size_t want = (nelem + threads - 1) / threads;
    int blocks = (int)(want < 1024 ? want : 1024);
    if (blocks < 1) blocks = 1;
    memra_bf16_cvt_kernel<<<blocks, threads, 0, stream>>>(x_f32, (__nv_bfloat16*)xb_bf16, nelem);
    cudaError_t ce = cudaGetLastError();
    return ce == cudaSuccess ? 0 : 10000 + (int)ce;
}

namespace {
std::mutex g_mubf;
cublasLtHandle_t g_ltbf_dev[kMemraMaxDevices] = {};
std::map<std::tuple<int, int, int, int>, F16Plan>* g_plansbf = nullptr;  // leaked (process-lifetime)
}  // namespace

// One BF16 prefill GEMM on a PRE-CONVERTED activation: y[m,n] row-major = x[m,k] @ W[n,k]^T,
// f32 accumulate/output. Col-major view: D[n,m] = A^T(W as k x n, opT) * B(xb as k x m, opN).
extern "C" int memra_bf16_pp_gemm_pre(
    const void* w_bf16, const void* xb_bf16, float* y_f32,
    int m, int n, int k, void* ws, size_t ws_bytes, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> lk(g_mubf);
    const int dev = memra_lt_current_device();
    cublasLtHandle_t g_ltbf = nullptr;
    {
        cublasStatus_t cs = memra_lt_handle_for_device(g_ltbf_dev, dev, &g_ltbf);
        if (cs != CUBLAS_STATUS_SUCCESS) return 40000 + (int)cs;
    }
    if (!g_plansbf) g_plansbf = new std::map<std::tuple<int, int, int, int>, F16Plan>();
    auto key = std::make_tuple(dev, m, n, k);
    auto it = g_plansbf->find(key);
    if (it == g_plansbf->end()) {
        F16Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k);  // W: k x n col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k);  // act: k x m col-major, ld=k
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, n, m, n);   // out: n x m col-major, ld=n
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(g_ltbf, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_plansbf->emplace(key, p).first;
    }
    F16Plan& plan = it->second;
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(g_ltbf, plan.op, &alpha, w_bf16, plan.la, xb_bf16, plan.lb,
                                      &beta, y_f32, plan.ld, y_f32, plan.ld, &plan.algo,
                                      ws, ws_bytes, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}

// ---- MLA absorb / decompress: STRIDED-BATCHED BF16 tensor-core GEMM -------------------------
// (lane/glm5-mla-tc-prefill, 2026-08-30.) The launch-diet census attributed 44.5 + 43.6
// ms/layer-chunk to memra_mla_absorb_q_kernel / memra_mla_decompress_v_kernel — per-position
// shared-memory dot programs whose math is a per-HEAD GEMM: for every head h,
//   absorb     : q_lat[:,h,:]  [t, r]  = q_nope[:,h,:] [t, dn] @ W_uk[h] [r, dn]^T
//   decompress : attn[:,h,:]   [t, dv] = o_lat[:,h,:]  [t, r]  @ W_uv[h] [dv, r]^T
// Both are ONE cublasLtMatmul with CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = n_head: the per-head
// activation views are row-major slices of the [t, n_head, d] plane (row stride n_head*d,
// batch offset d), which the layout's ld + STRIDED_BATCH_OFFSET express exactly. The weight is
// the conversion-split plane ([n, k] row-major per head, batch stride n*k) converted to bf16 by
// the caller. Output dtype is a flag: bf16 when it feeds the attention kernel directly (absorb),
// f32 when it re-enters the f32 stream (decompress).
//
// NUMERIC CONFIG: bf16 operands, f32 accumulate — the MEMRA_PP_BF16 class; the caller's flag row
// (MEMRA_MLA_TC_PREFILL) carries the calibrated band. Decode never reaches this path.

namespace {
std::mutex g_musb;
cublasLtHandle_t g_ltsb_dev[kMemraMaxDevices] = {};
// keyed on every shape/stride/dtype degree of freedom — a plan reused across a different stride
// would silently read the wrong rows.
std::map<std::tuple<int, int, int, int, long, long, long, long, int, int>, F16Plan>* g_planssb =
    nullptr;  // leaked (process-lifetime)
}  // namespace

extern "C" int memra_bf16_gemm_sb(
    const void* w_bf16,   // per-head [n, k] row-major bf16, batch stride n*k elements
    const void* x_bf16,   // per-head [m, k] row-major bf16 view: row stride x_rs, batch offset x_bs
    void* y,              // per-head [m, n] row-major view: row stride y_rs, batch offset y_bs
    int m, int n, int k,
    long x_rs, long x_bs, long y_rs, long y_bs,
    int batch, int y_is_bf16,
    void* ws, size_t ws_bytes, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> lk(g_musb);
    const int dev = memra_lt_current_device();
    cublasLtHandle_t g_ltsb = nullptr;
    {
        cublasStatus_t cs = memra_lt_handle_for_device(g_ltsb_dev, dev, &g_ltsb);
        if (cs != CUBLAS_STATUS_SUCCESS) return 40000 + (int)cs;
    }
    if (!g_planssb)
        g_planssb = new std::map<std::tuple<int, int, int, int, long, long, long, long, int, int>,
                                 F16Plan>();
    auto key = std::make_tuple(dev, m, n, k, x_rs, x_bs, y_rs, y_bs, batch, y_is_bf16);
    auto it = g_planssb->find(key);
    if (it == g_planssb->end()) {
        F16Plan p{};
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (s != CUBLAS_STATUS_SUCCESS) return (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        // Col-major view of the row-major contract (memra_bf16_pp_gemm_pre form, + batching):
        //   D[n, m] = W^T (W stored k x n col-major, ld=k) * X (k x m col-major, ld=x_rs)
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n, k);
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, x_rs);
        cublasLtMatrixLayoutCreate(&p.ld, y_is_bf16 ? CUDA_R_16BF : CUDA_R_32F, n, m, y_rs);
        int32_t bc = batch;
        int64_t sa = (int64_t)n * k, sb = (int64_t)x_bs, sd = (int64_t)y_bs;
        cublasLtMatrixLayout_t lays[3] = {p.la, p.lb, p.ld};
        int64_t strides[3] = {sa, sb, sd};
        for (int i = 0; i < 3; ++i) {
            cublasLtMatrixLayoutSetAttribute(lays[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
                                             sizeof(bc));
            cublasLtMatrixLayoutSetAttribute(lays[i],
                                             CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                             &strides[i], sizeof(strides[i]));
        }
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(g_ltsb, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur,
                                           &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        if (s != CUBLAS_STATUS_SUCCESS || nh == 0) return 20000 + (int)s;
        p.algo = heur.algo;
        it = g_planssb->emplace(key, p).first;
    }
    F16Plan& plan = it->second;
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(g_ltsb, plan.op, &alpha, w_bf16, plan.la, x_bf16, plan.lb,
                                      &beta, y, plan.ld, y, plan.ld, &plan.algo, ws, ws_bytes,
                                      stream);
    if (s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}

// Convert + GEMM in one call (single-projection callers).
extern "C" int memra_bf16_pp_gemm(
    const void* w_bf16, const float* x_f32, void* xb_bf16, float* y_f32,
    int m, int n, int k, void* ws, size_t ws_bytes, void* stream_v) {
    int rc = memra_bf16_cvt(x_f32, xb_bf16, (size_t)m * (size_t)k, stream_v);
    if (rc != 0) return rc;
    return memra_bf16_pp_gemm_pre(w_bf16, xb_bf16, y_f32, m, n, k, ws, ws_bytes, stream_v);
}
