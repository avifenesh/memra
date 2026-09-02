// moe_f16_batched.cu — the MEMRA_B200_PRIME_V2 arm-3 expert GEMM: dequant ONCE into a
// persistent f16 slab, pad the CSR to a uniform rows-per-expert, and run ONE cuBLASLt
// STRIDED-BATCHED matmul per projection on OUR stream.
//
// WHY THIS EXISTS, measured rather than assumed (2x B200 SXM, 2026-09-02, boot A):
// a 4096-token MoE layer chunk is ~1.24 TFLOP in 16.5 ms = 75 TFLOP/s = 3.4% of the part's
// 2.2 PFLOPS bf16 peak, while the ~2.7 GB of expert bytes it touches is 0.34 ms at 8 TB/s.
// The shipped direct-from-quant kernel (moe_f16_grouped.cu) is therefore neither compute- nor
// byte-bound: it is ~30x from BOTH floors, and it is what pins every prompt depth to the same
// ~2,500 tok/s plateau. The structural reason is its instruction class: every mma it issues is
// `mma.sync.aligned.m16n8k16` — the sm_80-portable WARP MMA — and sm_100a's real rate lives in
// tcgen05, which that path cannot reach. cuBLASLt CAN reach it. So this arm buys the 5th-gen
// tensor cores without writing a tcgen05 kernel, and pays for them in memory traffic.
//
// THE TRADE, stated numerically so it can be checked rather than believed. Per MoE layer at
// 4096 tokens (288 experts, 8 used, in_f 4096, out_f 2048): the shipped path reads 1.36 GB of
// packed NVFP4 per projection and materializes nothing; this arm reads the same 1.36 GB and
// WRITES 4.75 GB of f16, which the GEMM then reads back. Three projections: ~18.3 GB moved
// (2.3 ms at 8 TB/s) plus a GEMM that should land near 1.8 ms. ~4.1 ms against 16.5 ms. The
// arm is a LOSS if cuBLASLt does not clear roughly 300 TFLOP/s on these shapes, which is
// exactly what the gate's per-phase line and the box A/B are for.
//
// BUCKETED, because glm5's routing is skewed BY DESIGN. Measured on the real artifact
// 2026-09-02: Gini 0.575 over 288 experts, the busiest expert takes 77% of a layer's tokens and
// the median 1.3%. One uniform-n batch over all 288 therefore pads to the busiest expert's row
// count and costs 7.17x the real work (`288 x n_pad 816 = 235,008 padded rows against 32,768
// real`) — the first cut of this arm declined on every layer of every boot for exactly that
// reason, which is the decline line doing its job. The fix is the standard one: sort the active
// experts by row count, cut them into K contiguous buckets so each bucket's rows are within the
// pad ceiling of each other, and issue ONE strided-batched call per bucket per projection. The
// heavy head experts land in small (often singleton) buckets; the long tail shares one. K is
// chosen per call from the measured counts, never assumed.
//
// The pad/unpad kernels do NOT know about buckets. They walk a `pad_map`: one entry per padded
// row, holding either the CSR pair index that row carries or -1 for padding. That keeps them
// one launch each whatever K is, and it is the only structure that has to be right for the
// bucket layout to be correct — the GEMM just needs each bucket's slice to be contiguous, which
// the caller arranges by dequanting the expert slab in BUCKET ORDER.
//
// WHY STRIDED-BATCHED AND NOT GROUPED. `cublasGemmGroupedBatchedEx` takes variable n per group
// and needs no padding — and it issues on cuBLAS-INTERNAL streams that are NOT ordered with the
// caller's. On the glm5 walk's 283-group shape that race silently destroyed the trunk and took
// the GPU worker down with it (2026-09-02 boot D; the mode-1 refusal in
// `moe_ffn_grouped_prefill_sigmoid` carries the full account). `cublasLtMatmul` is ordinary
// stream-ordered work. The price of that ordering is a uniform n, i.e. padding every expert's
// pair block up to the widest one — 10.5% of the GEMM at the measured max_m=115 against
// n_pad=128, and REFUSED by the caller when routing skew would make it worse.
//
// NUMERIC CLASS: the f16-mirror grouped class, unchanged. A is the same dequanted-to-f16 weight
// the shipped workspace path produces (`memra_moe_f16g_dequant`, reused verbatim — this file
// adds no dequant of its own), B is the same amax-normalized f16 activation the shipped path
// gathers (`memra_moe_f16g_act`, reused verbatim — the pad kernel below only MOVES those rows),
// accumulation is f32, and the per-pair amax scale is folded on the way out exactly as the sk
// kernel folds it. What changes is the GEMM's reduction ORDER. So the bar is the band, not bits.
//
// PAD ROWS ARE ZERO AND THEIR OUTPUTS ARE NEVER READ. The pad kernel zero-fills rows
// `j >= m_e`, and the unpad kernel walks the CSR, so a padded output row has no consumer at all.
// The zero fill is belt-and-braces against a NaN in uninitialized memory reaching the GEMM's
// accumulator through a denormal path; correctness does not depend on it, and it costs one
// memset-shaped write per layer.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>

// ---- pad: CSR-order f16 activations -> the bucketed padded plane, driven by `pad_map` ----
// One block per PADDED row. `pad_map[r]` is the CSR pair index that row carries, or -1 for a
// pad row (zero-filled). Bucket-agnostic by construction.
static __global__ void moe_pad_act_map_kernel(
        const __half* __restrict__ act, const int* __restrict__ pad_map,
        __half* __restrict__ dst, int in_f){
    const int r = blockIdx.x + blockIdx.y * gridDim.x;
    const int p = pad_map[r];
    __half* d = dst + (size_t)r * in_f;
    if(p < 0){
        for(int v = threadIdx.x; v < in_f; v += blockDim.x) d[v] = __float2half(0.0f);
        return;
    }
    const __half* s = act + (size_t)p * in_f;
    for(int v = threadIdx.x; v < in_f; v += blockDim.x) d[v] = s[v];
}

// ---- unpad: padded f32 plane -> CSR-order [n_pairs][out_f], per-pair amax scale folded ----
// The same fold the sk kernel applies in its epilogue (`acc * s0`). Pad rows have no consumer.
static __global__ void moe_unpad_scale_map_kernel(
        const float* __restrict__ y_pad, const int* __restrict__ pad_map,
        const float* __restrict__ row_scale, float* __restrict__ y, int out_f){
    const int r = blockIdx.x + blockIdx.y * gridDim.x;
    const int p = pad_map[r];
    if(p < 0) return;
    const float s = row_scale[p];
    const float* src = y_pad + (size_t)r * out_f;
    float* d = y + (size_t)p * out_f;
    for(int c = threadIdx.x; c < out_f; c += blockDim.x) d[c] = src[c] * s;
}

// grid.x is capped at 65,535 by nothing, but grid.y is — and the padded row count passes
// 65,535 at glm5 widths, so the launcher splits across x and y and the kernels recombine.
static inline dim3 moe_pad_grid(int rows){
    const int x = rows < 32768 ? rows : 32768;
    const int y = (rows + x - 1) / x;
    return dim3((unsigned)x, (unsigned)y, 1);
}

extern "C" int memra_moe_pad_act_map_f16(const void* act_f16, const int* pad_map,
        void* dst_f16, int n_rows_padded, int in_f, void* stream){
    if(n_rows_padded <= 0 || in_f <= 0) return 1;
    cudaStream_t st = (cudaStream_t)stream;
    moe_pad_act_map_kernel<<<moe_pad_grid(n_rows_padded), 256, 0, st>>>(
        (const __half*)act_f16, pad_map, (__half*)dst_f16, in_f);
    cudaError_t e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

extern "C" int memra_moe_unpad_scale_map_f32(const float* y_pad, const int* pad_map,
        const float* row_scale, float* y, int n_rows_padded, int out_f, void* stream){
    if(n_rows_padded <= 0 || out_f <= 0) return 1;
    cudaStream_t st = (cudaStream_t)stream;
    moe_unpad_scale_map_kernel<<<moe_pad_grid(n_rows_padded), 256, 0, st>>>(
        y_pad, pad_map, row_scale, y, out_f);
    cudaError_t e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

// ---- cached cuBLASLt strided-batched plans (the f16_prefill.cu pattern, same reasons) ----
//
// Per-device handles: a cublasLtHandle_t is bound to the device current at cublasLtCreate, and a
// shared one returned CUBLAS_STATUS_EXECUTION_FAILED from PP stage 1 on a 2x B200 pair
// (2026-09-02, f16_prefill.cu's own note). The plan key carries the device for the same reason.
static const int kMemraMaxDevicesB = 64;
namespace {
struct BPlan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, ld;
    cublasLtMatmulAlgo_t algo;
    bool have_algo;
};
std::mutex g_mu_b;
cublasLtHandle_t g_ltb_dev[kMemraMaxDevicesB] = {};
std::map<std::tuple<int, int, int, int, int>, BPlan>* g_plans_b = nullptr;  // process-lifetime
}  // namespace

// ONE strided-batched f16 matmul per projection, on the CALLER'S stream.
//
// Column-major mapping, derived once and asserted by the gate rather than trusted:
//   per batch g, D[out_f, n_pad] = A^T * B with
//   A = W + g*out_f*in_f   (row-major [out_f][in_f] == col-major in_f x out_f, lda = in_f, opT),
//   B = act + g*n_pad*in_f (row-major [n_pad][in_f]  == col-major in_f x n_pad,  ldb = in_f, opN),
//   D = y   + g*n_pad*out_f(row-major [n_pad][out_f] == col-major out_f x n_pad, ldd = out_f).
// That is the SAME mapping mode 1's grouped call uses per group (ma=out_f, na=m_e, ka=in_f,
// lda=ldb=in_f, ldc=out_f); only the batching mechanism and the output dtype differ.
// `*_off` are ELEMENT offsets into the three planes, so one bucket's slice is addressed without
// the caller doing pointer arithmetic on device handles. Each bucket's experts are contiguous in
// the slab (the caller dequants in bucket order), which is what makes the stride uniform.
extern "C" int memra_moe_bgemm_f16_strided(
        const void* w_f16, size_t w_off, const void* act_f16, size_t act_off,
        float* y_f32, size_t y_off,
        int batch, int n_pad, int in_f, int out_f,
        void* ws, size_t ws_bytes, void* stream_v){
    const int n_active = batch;
    if(batch <= 0 || n_pad <= 0 || in_f <= 0 || out_f <= 0) return 1;
    const __half* A = (const __half*)w_f16 + w_off;
    const __half* B = (const __half*)act_f16 + act_off;
    float* D = y_f32 + y_off;
    cudaStream_t stream = (cudaStream_t)stream_v;
    std::lock_guard<std::mutex> guard(g_mu_b);
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMemraMaxDevicesB) dev = 0;
    if(!g_ltb_dev[dev]){
        if(cublasLtCreate(&g_ltb_dev[dev]) != CUBLAS_STATUS_SUCCESS) return 3;
    }
    cublasLtHandle_t lt = g_ltb_dev[dev];
    if(!g_plans_b) g_plans_b = new std::map<std::tuple<int,int,int,int,int>, BPlan>();
    auto key = std::make_tuple(dev, n_active, n_pad, in_f, out_f);
    auto it = g_plans_b->find(key);
    if(it == g_plans_b->end()){
        BPlan p{};
        p.have_algo = false;
        cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if(s != CUBLAS_STATUS_SUCCESS) return 10000 + (int)s;
        cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16F, in_f, out_f, in_f);
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16F, in_f, n_pad, in_f);
        cublasLtMatrixLayoutCreate(&p.ld, CUDA_R_32F, out_f, n_pad, out_f);
        const int32_t batch = n_active;
        const int64_t sa = (int64_t)out_f * in_f;
        const int64_t sb = (int64_t)n_pad * in_f;
        const int64_t sd = (int64_t)n_pad * out_f;
        cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
        cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
        cublasLtMatrixLayoutSetAttribute(p.ld, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
        cublasLtMatrixLayoutSetAttribute(p.la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa));
        cublasLtMatrixLayoutSetAttribute(p.lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb));
        cublasLtMatrixLayoutSetAttribute(p.ld, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sd, sizeof(sd));
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &ws_bytes, sizeof(ws_bytes));
        cublasLtMatmulHeuristicResult_t heur;
        int nh = 0;
        s = cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nh);
        cublasLtMatmulPreferenceDestroy(pref);
        // A shape cuBLASLt has no algorithm for is a DECLINE, not an error: the caller falls
        // back to the shipped kernel. Announced once per shape by the Rust side.
        if(s != CUBLAS_STATUS_SUCCESS || nh == 0) return 2;
        p.algo = heur.algo;
        p.have_algo = true;
        it = g_plans_b->emplace(key, p).first;
    }
    BPlan& plan = it->second;
    if(!plan.have_algo) return 2;
    float alpha = 1.f, beta = 0.f;
    cublasStatus_t s = cublasLtMatmul(lt, plan.op, &alpha, A, plan.la, B, plan.lb,
                                      &beta, D, plan.ld, D, plan.ld, &plan.algo,
                                      ws, ws_bytes, stream);
    if(s != CUBLAS_STATUS_SUCCESS) return 30000 + (int)s;
    return 0;
}
