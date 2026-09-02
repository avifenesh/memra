// memra engine Stage-1 kernels: correctness-first, all f32, no tensor cores.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// Math matches llama.cpp ggml CUDA ops node-for-node (norm.cu, rope.cu).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// PDL entry hook (SOTA item 2, 2026-07-13). Under a plain launch this is a documented
// no-op (grid dependencies are complete before any block starts). Under a PROGRAMMATIC
// graph edge (the MEMRA_PDL post-capture rewrite) it orders this kernel's global reads
// after the producer kernel's writes while still letting the grid launch overlap the
// producer's drain (~120ns/kernel on sm_120, pdl_probe). sm_90+ only; the sm_89
// portable arm compiles it out.
#if !defined(MEMRA_PORTABLE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MEMRA_PDL_ENTRY() cudaGridDependencySynchronize()
#else
#define MEMRA_PDL_ENTRY()
#endif

// ---- GPU-resident greedy argmax over logits[n_vocab] -> token_out[0] (u32). ----
// CUDA-GRAPH-PLAN Phase 1: removes the per-step dtoh(logits)+synchronize host barrier (the hard
// graph-capture blocker). Single CTA, 256 threads. Tie-break = SMALLEST index wins, bit-identical
// to the host argmax (forward.rs `if v > bv` strictly-greater keeps the first max). Each thread
// scans a strided slice keeping (best_val,best_id); reduce keeps the lower id on equal value.
// SUPERSEDED for the live path by the parallel 2-pass kernels below (argmax_partial_f32 +
// argmax_final_f32): one 256-thread block scanning 248K logits on ONE SM is HBM-starved (~448us/
// token, ncu clock-locked). Kept as the bit-exact single-CTA reference (same tie-break contract).
extern "C" __global__ void argmax_logits_f32_to_u32(
        const float* __restrict__ logits, uint32_t* __restrict__ token_out, int n_vocab) {
    int tid = threadIdx.x;
    float best_v = -3.402823466e38f;   // -FLT_MAX (matches f32::NEG_INFINITY seed)
    int   best_i = 0x7fffffff;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        // strictly-greater takes the new value; on a tie keep the smaller index.
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    // warp butterfly reduce: max value, smallest index on tie.
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) token_out[0] = (uint32_t)best_i;
    }
}

// ---- PARALLEL argmax (2-pass, multi-CTA). RANK1 LEVER: the single-CTA argmax above scans the full
// 248320-vocab logits with ONE 256-thread block on ONE SM — memory-starved, ~426us/token. This pair
// fans the scan across NB blocks so HBM is saturated, then a 1-block final reduce picks the winner.
// BIT-IDENTICAL to the single-CTA kernel and to host `argmax` (forward.rs `v>bv`): strictly-greater
// takes the new value, ties keep the SMALLEST index. Pass 1 -> (part_v[NB], part_i[NB]); pass 2
// reduces those NB partials into token_out[0]. Both passes are plain launches (graph-capturable).
//
// Greedy-token softmax probability, pass 1: partial sums of exp(logit - max) where max =
// logits[tok] (tok = the argmax token, already on device). p(tok) = 1 / sum. Feeds the
// spec-decode p-min confidence gate (stop drafting when the head is unsure — the mechanism
// behind the serve script's --spec-draft-p-min): one extra ~1-4MB logits read per draft token.
extern "C" __global__ void prob_of_token_partial_f32(
        const float* __restrict__ logits, const uint32_t* __restrict__ tok,
        float* __restrict__ part_s, int n_vocab) {
    const float mx = logits[tok[0]];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = gtid; i < n_vocab; i += gstride) sum += expf(logits[i] - mx);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float ss[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? ss[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) part_s[blockIdx.x] = sum;
    }
}
// pass 2: p = 1 / sum(partials).
extern "C" __global__ void prob_of_token_final_f32(
        const float* __restrict__ part_s, float* __restrict__ p_out, int nb) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < nb; i += blockDim.x) sum += part_s[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float ss[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) ss[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        sum = (lane < nwarps) ? ss[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) p_out[0] = 1.0f / sum;
    }
}

// Pass 1: block b, thread tid scans logits[b*blockDim + tid : n_vocab : NB*blockDim] keeping
// (best_v, smallest best_i), block-reduces, writes part_v[b]/part_i[b].
extern "C" __global__ void argmax_partial_f32(
        const float* __restrict__ logits, float* __restrict__ part_v, int* __restrict__ part_i,
        int n_vocab) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;
    int gstride = gridDim.x * blockDim.x;
    float best_v = -3.402823466e38f;
    int   best_i = 0x7fffffff;
    for (int i = gtid; i < n_vocab; i += gstride) {
        float v = logits[i];
        if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { part_v[blockIdx.x] = best_v; part_i[blockIdx.x] = best_i; }
    }
}

// Pass 2: ONE block reduces the NB partials into token_out[0]. Same tie-break (smallest index).
// nb = number of pass-1 blocks. Launch with block_dim >= 32 (256 used); strided over nb.
extern "C" __global__ void argmax_final_f32(
        const float* __restrict__ part_v, const int* __restrict__ part_i,
        uint32_t* __restrict__ token_out, int nb) {
    int tid = threadIdx.x;
    float best_v = -3.402823466e38f;
    int   best_i = 0x7fffffff;
    for (int i = tid; i < nb; i += blockDim.x) {
        float v = part_v[i];
        int   id = part_i[i];
        if (v > best_v || (v == best_v && id < best_i)) { best_v = v; best_i = id; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[32];
    __shared__ int   si[32];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        best_v = (lane < nwarps) ? sv[lane] : -3.402823466e38f;
        best_i = (lane < nwarps) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, best_v, off);
            int   oi = __shfl_xor_sync(0xffffffff, best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) token_out[0] = (uint32_t)best_i;
    }
}

// ---- RMSNorm: one block per row. y = x / sqrt(mean(x^2) + eps) * weight ----
// x: [ncols, nrows] row-major (row stride = ncols). weight: [ncols]. dst same shape as x.
extern "C" __global__ void rms_norm_f32(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ dst, int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;

    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    // block reduce
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// rms_norm + fused fp16 twin emission (task #14 launch diet, 2026-07-26): on the prefill
// path the f32 output feeds nothing but the f16-mirror GEMM group, so emit the fp16 copy
// here and kill the standalone memra_f16_cvt launch (+ its full re-read of dst). The f32
// math and store are VERBATIM rms_norm_f32 (same reduction tree); the fp16 value is the
// same __float2half of the same f32 -> end-to-end BIT-IDENTICAL to norm-then-convert.
extern "C" __global__ void rms_norm_f16out_f32(const float* __restrict__ x, const float* __restrict__ w,
                                               float* __restrict__ dst, __half* __restrict__ dst16,
                                               int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    __half* hr = dst16 + (size_t)row * ncols;

    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float o = xr[i] * scale * w[i];
        dr[i] = o;
        hr[i] = __float2half(o);
    }
}

// f16out twin of add_rms_norm_f32 (round 28): the prefill trunk's residual+norm pair
// in ONE kernel, norm epilogue also emitting the fp16 GEMM operand (task-#17 class).
// BIT-IDENTICAL to add_f32 -> rms_norm_f16out_f32: same IEEE add, same reduction order,
// same __float2half twin.
extern "C" __global__ void add_rms_norm_f16out_f32(const float* __restrict__ a, const float* __restrict__ b,
                                                   const float* __restrict__ w, float* __restrict__ res,
                                                   float* __restrict__ dst, __half* __restrict__ dst16,
                                                   int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    __half* hr = dst16 + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float o = rr[i] * scale * w[i];
        dr[i] = o;
        hr[i] = __float2half(o);
    }
}

// ---- TRUNK-KERNELS ILP TWIN (lane/dspark-trunk-kernels-20260820): rms_norm_f32 with the
// strided element loop unrolled 4-deep. The verify T-row norms launch grid=T (3..8 blocks on a
// 188-SM card) x block=256: 20 strided scalar iterations whose load->fma chain is pure serial
// DRAM latency (measured 11.8us/inst for a 20KB row read — nsys-B verify scope, 130 inst/rd
// = 1.51 ms/rd for the norm pair). Unrolling issues 4 independent loads per round; the
// ACCUMULATION ORDER IS UNCHANGED — each thread still consumes elements i, i+b, i+2b, i+3b in
// that exact sequence into ONE accumulator (same fma(v,v,sum) chain), and the block reduce is
// VERBATIM rms_norm_f32 -> BIT-IDENTICAL for every (ncols, blockDim). The epilogue keeps the
// exact per-element expression xr[i] * scale * w[i]. MEMRA_NORM_ILP=0 reverts to the v1 names.
extern "C" __global__ void rms_norm_f32_v2(const float* __restrict__ x, const float* __restrict__ w,
                                           float* __restrict__ dst, int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    const float* xr = x + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;

    float sum = 0.0f;
    int i = tid;
    for (; i + 3 * bd < ncols; i += 4 * bd) {
        float v0 = xr[i];
        float v1 = xr[i + bd];
        float v2 = xr[i + 2 * bd];
        float v3 = xr[i + 3 * bd];
        sum += v0 * v0; sum += v1 * v1; sum += v2 * v2; sum += v3 * v3;
    }
    for (; i < ncols; i += bd) { float v = xr[i]; sum += v * v; }
    // block reduce — VERBATIM rms_norm_f32
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (bd + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    i = tid;
    for (; i + 3 * bd < ncols; i += 4 * bd) {
        float o0 = xr[i] * scale * w[i];
        float o1 = xr[i + bd] * scale * w[i + bd];
        float o2 = xr[i + 2 * bd] * scale * w[i + 2 * bd];
        float o3 = xr[i + 3 * bd] * scale * w[i + 3 * bd];
        dr[i] = o0; dr[i + bd] = o1; dr[i + 2 * bd] = o2; dr[i + 3 * bd] = o3;
    }
    for (; i < ncols; i += bd) dr[i] = xr[i] * scale * w[i];
}

// ILP twin of add_rms_norm_f32 (same lane): r = a[i]+b[i] is the same IEEE add per element,
// written to res in the same positions; the sum-of-squares consumes the same r values in the
// same per-thread strided order into one accumulator; reduce VERBATIM -> BIT-IDENTICAL.
extern "C" __global__ void add_rms_norm_f32_v2(const float* __restrict__ a, const float* __restrict__ b,
                                               const float* __restrict__ w, float* __restrict__ res,
                                               float* __restrict__ dst, int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    int i = tid;
    for (; i + 3 * bd < ncols; i += 4 * bd) {
        float a0 = ar[i], a1 = ar[i + bd], a2 = ar[i + 2 * bd], a3 = ar[i + 3 * bd];
        float b0 = br[i], b1 = br[i + bd], b2 = br[i + 2 * bd], b3 = br[i + 3 * bd];
        float v0 = a0 + b0, v1 = a1 + b1, v2 = a2 + b2, v3 = a3 + b3;
        rr[i] = v0; rr[i + bd] = v1; rr[i + 2 * bd] = v2; rr[i + 3 * bd] = v3;
        sum += v0 * v0; sum += v1 * v1; sum += v2 * v2; sum += v3 * v3;
    }
    for (; i < ncols; i += bd) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (bd + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    i = tid;
    for (; i + 3 * bd < ncols; i += 4 * bd) {
        float o0 = rr[i] * scale * w[i];
        float o1 = rr[i + bd] * scale * w[i + bd];
        float o2 = rr[i + 2 * bd] * scale * w[i + 2 * bd];
        float o3 = rr[i + 3 * bd] * scale * w[i + 3 * bd];
        dr[i] = o0; dr[i + bd] = o1; dr[i + 2 * bd] = o2; dr[i + 3 * bd] = o3;
    }
    for (; i < ncols; i += bd) dr[i] = rr[i] * scale * w[i];
}

// ---- RANK3 LEVER (add+rmsnorm fuse): residual-add THEN RMSNorm in ONE kernel. ----
// res = a + b  (the residual, written out for the next residual-add); dst = rms_norm(res) * w.
// Fuses e.add(a,b,res) + e.rms_norm(res,w,dst) — removes one launch + one HBM read of `res` per
// residual+norm pair. BIT-IDENTICAL to add_f32 then rms_norm_f32: r=a[i]+b[i] is the same IEEE add,
// and the sum-of-squares reduction reads the same r values in the same per-thread/strided order.
// One block per row (row stride = ncols). a,b,res,dst: [ncols, nrows]; w: [ncols].
// O-PROJ TAIL FUSION M2: the direct-join add (mixed = a0 + a1) folded into the
// residual+post-norm — the body below is add_rms_norm_f32 VERBATIM with the b-operand
// composed as (a0[i] + a1[i]) in a register (plain adds, no contraction possible), so the
// values are BIT-IDENTICAL to add_f32 then add_rms_norm_f32.
extern "C" __global__ void join_add_rms_norm_f32(
        const float* __restrict__ a0, const float* __restrict__ a1,
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ res,
        float* __restrict__ dst, int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float m = a0[i] + a1[i];
        float v = xr[i] + m;
        rr[i] = v;
        sum += v * v;
    }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = rr[i] * scale * w[i];
}

extern "C" __global__ void add_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
                                            const float* __restrict__ w, float* __restrict__ res,
                                            float* __restrict__ dst, int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = rr[i] * scale * w[i];
}

// ---- E4B glue fusion: rms(a, wa) prologue + the add_rms_norm_f32 program — folds the
// per-layer post-attn rms_norm_f32(o) into the tail's residual-add+ffn-norm launch. ----
extern "C" __global__ void rms_pre_add_rms_norm_f32(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        const float* __restrict__ w, float* __restrict__ res,
        float* __restrict__ dst, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    __shared__ float s[128];
    // SINGLE-PHASE (parity with the q8z twin — verify t>1 and decode t=1 must share the
    // reduction algebra bit-for-bit; the 31B depth-spec 45/128 was this mismatch).
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3];
    float scale = rsqrtf(sumv2 / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] * ascale) * wa[i] + br[i];
        rr[i] = v;
        dr[i] = v * scale * w[i];
    }
}

// ---- E4B glue fusion wave 2: the two tail-entry programs with the ffn-norm output zsh
// ALSO emitted q8_1 (fused2 gate/up consume only the quantized pair at t=1; the f32 zsh
// stays written for the off-class fallbacks). Epilogue = quantize_q8_1's program verbatim. ----
extern "C" __global__ void rms_pre_add_rms_norm_q8z_f32(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        const float* __restrict__ w, float* __restrict__ res,
        float* __restrict__ dst,
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    int nblk = ncols / 32;
    __shared__ float s[128];
    // SINGLE-PHASE (wave 4, same algebra as the closing emit; c == 1 here):
    // sum(v^2) with v = a*ascale*wa + b  ==  ascale^2*S2 + 2*ascale*S3 + S4.
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3];
    float scale = rsqrtf(sumv2 / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        rr[i] = (ar[i] * ascale) * wa[i] + br[i];
    }
    __syncthreads();
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        d4[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- E4B glue fusion wave 2: a + b with the sum ALSO emitted q8_1 (resid feeds inp_gate
// through matmul_pre; f32 resid stays written for the later residual add). ----
// ---- E4B FFN-tail EXIT fusion (glue wave 5): resid = attn_out + rms(f0, post_ffw), ----
// emitted f32 + q8_1 pair in ONE launch — replaces rms_norm_f32(f0 -> sn) + add_q8_1_f32(sn,
// attn_out). BIT-IDENTITY: the rms reduction reads f0 in rms_norm_f32's exact strided order
// (same single sum, same block reduce); the per-element value is ((f0[i]*s)*w[i]) + b[i] — the
// identical op chain of the two-kernel pair (the f32 round-trip of sn is exact, removing it
// changes no bits); the quantize section is add_q8_1_f32's float4-quad walk verbatim.
extern "C" __global__ void rms_pre_add_q8_1_f32(
        const float* __restrict__ a,   // f0 (ffn_down output)
        const float* __restrict__ wa,  // post_ffw_norm weight
        const float* __restrict__ b,   // attn_out (the residual carry)
        float* __restrict__ res,       // resid f32 out
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;

    // phase 1: rms_norm_f32's reduction, verbatim.
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);

    // phase 2: add_q8_1_f32's quad walk verbatim, with v = ((a*scale)*wa) + b inline.
    int lane = tid & 31;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    const float* war = wa;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        const float4 a4 = ((const float4*)ar)[i4];
        const float4 w4 = ((const float4*)war)[i4];
        const float4 b4 = ((const float4*)br)[i4];
        float4 v = make_float4(a4.x * scale * w4.x + b4.x, a4.y * scale * w4.y + b4.y,
                               a4.z * scale * w4.z + b4.z, a4.w * scale * w4.w + b4.w);
        ((float4*)rr)[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

extern "C" __global__ void add_q8_1_f32(const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ res,
                                        signed char* __restrict__ out_q, float* __restrict__ out_d,
                                        int ncols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int nblk = ncols / 32;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        const float4 a4 = ((const float4*)ar)[i4];
        const float4 b4 = ((const float4*)br)[i4];
        float4 v = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
        ((float4*)rr)[i4] = v;
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- RMSNorm with FUSED q8_1 quantize epilogue (decode glue-fusion lever). ----
// Computes z = rms_norm(x)*w THEN emits z directly as q8_1 (out_q int8 + out_d f32 per-32 scale),
// so the standalone `quantize_q8_1` launch + the f32 `z` HBM round-trip are removed. The normed
// activation has exactly the matvec(s) as consumers, all on the q8_1 fast path — so producing it
// pre-quantized is free (rms_norm already touches every element). BIT-IDENTICAL to
// rms_norm_f32(x,w,z) then quantize_q8_1(z): the scale `s = rsqrt(mean(x^2)+eps)` reduction reads
// the same x in the same strided order; the normed value is the SAME (x[i]*s)*w[i] association;
// the per-32-block amax/d=amax/127/id=1/d/__float2int_rn rounding is quantize_q8_1's exactly.
// One block per row (decode: nrows=1). ncols must be a multiple of 32 (n_embd always is).
extern "C" __global__ void rms_norm_q8_1(const float* __restrict__ x, const float* __restrict__ w,
                                         signed char* __restrict__ out_q, float* __restrict__ out_d,
                                         int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    int nblk = ncols / 32;
    // pass 1: sum of squares -> scale (identical reduction to rms_norm_f32)
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    // pass 2, WARP-PER-4-BLOCKS float4 (ncu 2026-07-03): lane j reads float4 -> a warp covers 128
    // elements = FOUR 32-blocks per iteration (512B coalesced x/w reads, char4 writes). Each block
    // maps to an 8-lane group; amax reduces within the group (3 shfl_xor steps, width 8). Order of
    // max over the same 32 values is irrelevant -> q8_1 output BIT-IDENTICAL to quantize_q8_1.
    // (Plain warp-per-block regressed here: single-CTA kernel, 8 warps -> too little MLP.)
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)xr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;               // float4 index; 32 lanes * 4 = 128 elems = 4 blocks
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
    // tail (nblk % 4 != 0): scalar warp-per-block for the last <4 blocks.
    for (int blk = (nblk & ~3) + (tid >> 5); blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (xr[i] * scale) * w[i];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
}

// ---- add+RMSNorm with FUSED q8_1 quantize epilogue. res = a+b (written out for the next residual);
// then z = rms_norm(res)*w emitted directly as q8_1. Fuses add_rms_norm + quantize_q8_1 for the FFN
// input path (z feeds ffn_gate/ffn_up matvecs, both q8_1-fast). BIT-IDENTICAL to add_rms_norm_f32
// then quantize_q8_1: r=a[i]+b[i] same IEEE add (and written to `res` for the post-ffn add), the
// sum-of-squares reduction reads the same r, z=(r*scale)*w same association, per-32 q8_1 identical.
// add+RMSNorm emitting BOTH the f32 normed row (z — the MoE router logits input) AND its q8_1
// quantization (the expert dp4a input) in one launch. The MoE layer needs both views of the same
// vector; running add_rms_norm_f32 then quantize_q8_1 costs a launch and re-reads z from HBM.
// BIT-IDENTICAL: z values same IEEE ops as add_rms_norm_f32; q8 blocks same amax/127 rounding as
// quantize_q8_1 over the same z.
extern "C" __global__ void add_rms_norm_zq8(const float* __restrict__ a, const float* __restrict__ b,
                                            const float* __restrict__ w, float* __restrict__ res,
                                            float* __restrict__ z,
                                            signed char* __restrict__ out_q, float* __restrict__ out_d,
                                            int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* zr = z + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    for (int blk = tid >> 5; blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (rr[i] * scale) * w[i];
        zr[i] = v;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
}

extern "C" __global__ void add_rms_norm_q8_1(const float* __restrict__ a, const float* __restrict__ b,
                                             const float* __restrict__ w, float* __restrict__ res,
                                             signed char* __restrict__ out_q, float* __restrict__ out_d,
                                             int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = ar[i] + br[i]; rr[i] = v; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    // pass 2, WARP-PER-4-BLOCKS float4: same coalesced form as rms_norm_q8_1 (see comment there).
    // Reads the just-written `res` row (rr) — bit-identical (same IEEE values back from cache/HBM).
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* r4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = r4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
    for (int blk = (nblk & ~3) + (tid >> 5); blk < nblk; blk += blockDim.x >> 5) {
        int i = blk * 32 + lane;
        float v = (rr[i] * scale) * w[i];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        base_q[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) base_d[blk] = d;
    }
}

// ---- L2 norm per head_dim (no weight). y = x / sqrt(sum(x^2)+eps). one block per row ----
extern "C" __global__ void l2_norm_f32(const float* __restrict__ x, float* __restrict__ dst,
                                       int ncols, float eps) {
    int row = blockIdx.x; int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols; float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale;
}

// l2_norm PREFILL v2 (round 27): warp-per-row float4 — d_state=128 cols = exactly one
// float4 per lane, warp-shuffle reduce, float4 store. The 256-block strided kernel ran
// at 918GB/s (half the threads idle on 128-col rows). NUMERIC CONFIG (explicit+gated,
// MEMRA_L2_V2, the GDN-chunked/mma precedent): the reduction tree order differs from
// l2_norm_f32 — arbitration = greedy-stream/argmax battery, not bit-identity. The same
// values feed K2/K4 either way at ~1e-7 relative. PREFILL-ONLY: decode keeps
// l2_norm_decode (decode==verify law untouched).
// dst16 (nullable): the bf16 twin of the row — the K4 kb16 mirror emitted in-epilogue
// (same __float2bfloat16 values the standalone mirror pass would produce).
extern "C" __global__ void l2_norm_pp_v2_f32(const float* __restrict__ x, float* __restrict__ dst,
                                             __nv_bfloat16* __restrict__ dst16,
                                             int ncols, int nrows, float eps) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    int lane = threadIdx.x & 31;
    const float* xr = x + (size_t)row * ncols;
    float4 v = *(const float4*)(xr + lane * 4);
    float sum = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    float scale = rsqrtf(sum + eps);
    float4 o4 = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
    *(float4*)(dst + (size_t)row * ncols + lane * 4) = o4;
    if (dst16 != nullptr) {
        __nv_bfloat16* h = dst16 + (size_t)row * ncols + lane * 4;
        h[0] = __float2bfloat16(o4.x); h[1] = __float2bfloat16(o4.y);
        h[2] = __float2bfloat16(o4.z); h[3] = __float2bfloat16(o4.w);
    }
}

// ---- RoPE NEOX (full or partial). Pairs x[i] with x[i+n_dims/2]; dims >= n_dims copied. ----
// data layout: [head_dim, n_heads, n_tokens] (head_dim fastest). pos: [n_tokens].
// One thread per (pair i0/2, head, token). grid.x = n_heads*n_tokens, threads = head_dim/2.
extern "C" __global__ void rope_neox_f32(float* __restrict__ x, const int* __restrict__ pos,
                                         int head_dim, int n_dims, int n_heads,
                                         float theta_scale, float freq_scale) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;                 // pair index within head, 0..hd2-1
    if (j >= hd2) return;
    int hr = blockIdx.x;                 // head*token flattened
    int head = hr % n_heads;
    int tok = hr / n_heads;
    (void)head;
    float* base = x + (size_t)hr * head_dim;
    int half = n_dims / 2;
    if (j >= half) {
        // dims >= n_dims are untouched (copy-through is a no-op since in-place)
        return;
    }
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    float c = cosf(theta), s = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * s;
    base[j + half] = x0 * s + x1 * c;
}

// ---- gemma4: 3-way rms_norm — ONE reduction over x, three weight vectors/outputs
// (gemma's attn_out feeds ffn_norm + router-scale + pre_ffw_norm_2). Numerics per output
// identical to rms_norm_f32 (same block reduction, same scale multiply). ----
extern "C" __global__ void rms_norm3_f32(const float* __restrict__ x,
                                         const float* __restrict__ w0, const float* __restrict__ w1,
                                         const float* __restrict__ w2,
                                         float* __restrict__ d0, float* __restrict__ d1,
                                         float* __restrict__ d2, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    // block reduce — the rms_norm_f32 shuffle tree VERBATIM (per-output bit-identity).
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    float* o0 = d0 + (size_t)row * ncols;
    float* o1 = d1 + (size_t)row * ncols;
    float* o2 = d2 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float xh = xr[i] * scale;
        o0[i] = xh * w0[i];
        o1[i] = xh * w1[i];
        o2[i] = xh * w2[i];
    }
}

// ---- gemma4: q/k/v head norms in ONE launch — 3 (src,dst,rows) segments of the same width
// (q_norm rows=nh, k_norm rows=nkv, weightless-V rows=nkv). Segment picked by row index;
// per-row chain = rms_norm_f32 verbatim. ----
extern "C" __global__ void rms_norm_qkv_f32(const float* __restrict__ q, const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            const float* __restrict__ wq, const float* __restrict__ wk,
                                            const float* __restrict__ wv,
                                            float* __restrict__ dq, float* __restrict__ dk,
                                            float* __restrict__ dv,
                                            int ncols, int rq, int rk, float eps) {
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;        w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq; xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// ---- warp-per-row twin of rms_norm_qkv_f32 (prefill T>=16): the block-per-row form runs
// 17k+ tiny blocks of rms_block() threads over 512-col head rows at ~92GB/s (launch/reduce
// latency dominates the 2KB/row payload). Here: 8 warps/block, one ROW per warp, float4
// loads, warp-shuffle reduce only. OWN NUMERIC CONFIG (float4-lane partial sums reduce in a
// different order than the block tree) — battery-gated, MEMRA_QKVNORM_W=0 reverts. ----
extern "C" __global__ void rms_norm_qkv_w4_f32(const float* __restrict__ q, const float* __restrict__ k,
                                               const float* __restrict__ v,
                                               const float* __restrict__ wq, const float* __restrict__ wk,
                                               const float* __restrict__ wv,
                                               float* __restrict__ dq, float* __restrict__ dk,
                                               float* __restrict__ dv,
                                               int ncols, int rq, int rk, int rv, float eps) {
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rq + rk + rv) return;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;              w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq;      xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    const int nc4 = ncols >> 2;
    const float4* x4 = (const float4*)xr;
    float sum = 0.0f;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i];
        sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float scale = rsqrtf(sum / ncols + eps);
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i]; float4 wv4 = w4[i];
        float4 ov;
        ov.x = xv.x * scale * wv4.x; ov.y = xv.y * scale * wv4.y;
        ov.z = xv.z * scale * wv4.z; ov.w = xv.w * scale * wv4.w;
        d4[i] = ov;
    }
}

extern "C" __global__ void rms_norm_qkv_w4b_f32(const float* __restrict__ q, const float* __restrict__ k,
                                               const float* __restrict__ v,
                                               const float* __restrict__ wq, const float* __restrict__ wk,
                                               const float* __restrict__ wv,
                                               float* __restrict__ dq, float* __restrict__ dk,
                                               float* __restrict__ dv,
                                               __nv_bfloat16* __restrict__ dvb,
                                               int ncols, int rq, int rk, int rv, float eps,
                                               int vf16) {
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rq + rk + rv) return;
    const float* xr; const float* w; float* dr;
    if (row < rq)           { xr = q + (size_t)row * ncols;              w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { int r = row - rq;      xr = k + (size_t)r * ncols; w = wk; dr = dk + (size_t)r * ncols; }
    else                    { int r = row - rq - rk; xr = v + (size_t)r * ncols; w = wv; dr = dv + (size_t)r * ncols; }
    const int nc4 = ncols >> 2;
    const float4* x4 = (const float4*)xr;
    float sum = 0.0f;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i];
        sum += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    const float scale = rsqrtf(sum / ncols + eps);
    const float4* w4 = (const float4*)w;
    float4* d4 = (float4*)dr;
    // v rows also emit bf16 (the FA V operand; q/k get theirs post-rope).
    __nv_bfloat16* db = (row >= rq + rk) ? dvb + (size_t)(row - rq - rk) * ncols : nullptr;
    for (int i = lane; i < nc4; i += 32) {
        float4 xv = x4[i]; float4 wv4 = w4[i];
        float4 ov;
        ov.x = xv.x * scale * wv4.x; ov.y = xv.y * scale * wv4.y;
        ov.z = xv.z * scale * wv4.z; ov.w = xv.w * scale * wv4.w;
        d4[i] = ov;
        if (db) {
            if (vf16) {   // f16-P/V door: V operand consumed as __half by the h2/sp16 stamps
                __half* dh = (__half*)db;
                dh[4*i+0] = __float2half(ov.x); dh[4*i+1] = __float2half(ov.y);
                dh[4*i+2] = __float2half(ov.z); dh[4*i+3] = __float2half(ov.w);
            } else {
                db[4*i+0] = __float2bfloat16(ov.x); db[4*i+1] = __float2bfloat16(ov.y);
                db[4*i+2] = __float2bfloat16(ov.z); db[4*i+3] = __float2bfloat16(ov.w);
            }
        }
    }
}

// ---- E4B glue fusion wave 3: rms_norm_qkv + rope_neox2 in ONE launch. Row segments as in
// rms_norm_qkv_f32; after the norm store, q rows (seg 0) and k rows (seg 1) rope in-block
// (rope_neox math verbatim on the normed row; barrier between store and rope read). ----
// cat twin (wave 4b): the q|k|v input is ONE contiguous buffer (the qkv_cat matvec output),
// so the three input segments collapse to base + row*ncols. Outputs stay separate.
extern "C" __global__ void rms_norm_qkv_rope_cat_f32(
        const float* __restrict__ qkv,
        const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int ncols, int rq, int rk,
        const int* __restrict__ pos, int nh_q, int nh_k,
        float theta_scale, float freq_scale, const float* __restrict__ ff,
        float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    const float* xr = qkv + (size_t)row * ncols;
    const float* w; float* dr;
    int seg; int seg_r;
    if (row < rq)           { seg = 0; seg_r = row;           w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { seg = 1; seg_r = row - rq;      w = wk; dr = dk + (size_t)seg_r * ncols; }
    else                    { seg = 2; seg_r = row - rq - rk; w = wv; dr = dv + (size_t)seg_r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
    if (seg == 2) return;
    __syncthreads();
    int half = ncols / 2;
    int j = tid;
    if (j >= half) return;
    int tok = (seg == 0) ? seg_r / nh_q : seg_r / nh_k;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = dr[j];
    float x1 = dr[j + half];
    dr[j]        = x0 * c - x1 * sn;
    dr[j + half] = x0 * sn + x1 * c;
}

extern "C" __global__ void rms_norm_qkv_rope_f32(
        const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
        float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv,
        int ncols, int rq, int rk,
        const int* __restrict__ pos, int nh_q, int nh_k,
        float theta_scale, float freq_scale, const float* __restrict__ ff,
        float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    int seg; int seg_r;
    if (row < rq)           { seg = 0; seg_r = row;           xr = q + (size_t)row * ncols;   w = wq; dr = dq + (size_t)row * ncols; }
    else if (row < rq + rk) { seg = 1; seg_r = row - rq;      xr = k + (size_t)seg_r * ncols; w = wk; dr = dk + (size_t)seg_r * ncols; }
    else                    { seg = 2; seg_r = row - rq - rk; xr = v + (size_t)seg_r * ncols; w = wv; dr = dv + (size_t)seg_r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
    if (seg == 2) return;                   // V: norm only, never roped
    __syncthreads();                        // normed row visible before the rope read
    // rope_neox on the normed row (n_dims == ncols == head_dim here; math verbatim).
    int half = ncols / 2;
    int j = tid;
    if (j >= half) return;
    int tok = (seg == 0) ? seg_r / nh_q : seg_r / nh_k;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = dr[j];
    float x1 = dr[j + half];
    dr[j]        = x0 * c - x1 * sn;
    dr[j + half] = x0 * sn + x1 * c;
}

// ---- gemma4: two rms_norms of two DIFFERENT inputs, same width, one launch
// (post_ffw_norm_1(mlp0) + post_ffw_norm_2(moe0)). grid.x = 2*nrows; per-row verbatim. ----
extern "C" __global__ void rms_norm2x_f32(const float* __restrict__ a, const float* __restrict__ b,
                                          const float* __restrict__ wa, const float* __restrict__ wb,
                                          float* __restrict__ da, float* __restrict__ db,
                                          int ncols, int nrows, float eps) {
    int row = blockIdx.x;
    const float* xr; const float* w; float* dr;
    if (row < nrows) { xr = a + (size_t)row * ncols; w = wa; dr = da + (size_t)row * ncols; }
    else { int r = row - nrows; xr = b + (size_t)r * ncols; w = wb; dr = db + (size_t)r * ncols; }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float x = xr[i]; sum += x * x; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v2 = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v2 += __shfl_down_sync(0xffffffff, v2, o);
        if (tid == 0) s[0] = v2;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = xr[i] * scale * w[i];
}

// ---- gemma4: dst = (a + b) * c — the layer-tail residual add + layer_output_scale in one
// launch. Same IEEE add-then-multiply as add_f32 followed by scale_f32. ----
extern "C" __global__ void add_scale_f32(const float* __restrict__ a, const float* __restrict__ b,
                                         float c, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (a[i] + b[i]) * c;
}

// ---- gemma4 R4: final-logit softcap, y = cap * tanh(y / cap), in place. ----
extern "C" __global__ void softcap_f32(float* __restrict__ y, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(y[i] / cap);
}

// ---- gemma4: suppress-token mask — y[row][ids[j]] = -inf for every logits row, so every
// downstream consumer (host/device argmax, sampler) inherits the model card's forbidden ids. ----
extern "C" __global__ void mask_ids_rows_f32(float* __restrict__ y, const int* __restrict__ ids,
                                             int n_ids, int n_vocab, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_ids * t) y[(size_t)(i / n_ids) * n_vocab + ids[i % n_ids]] = -INFINITY;
}

// ---- gemma4: residual add + layer scale + NEXT layer's attn_norm in one launch.
// res = (a+b)*c (add_scale_f32 verbatim); dst = rms_norm(res, w) (rms_norm_f32 verbatim). ----
extern "C" __global__ void add_scale_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
                                                  float c, const float* __restrict__ w,
                                                  float* __restrict__ res, float* __restrict__ dst,
                                                  int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] + br[i]) * c;
        rr[i] = v;
        sum += v * v;
    }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) dr[i] = rr[i] * scale * w[i];
}

// ---- gemma4: residual add + the THREE attn_out norms in one launch (add then rms_norm3,
// per-element chains verbatim). ----
extern "C" __global__ void add_rms_norm3_f32(const float* __restrict__ a, const float* __restrict__ b,
                                             const float* __restrict__ w0, const float* __restrict__ w1,
                                             const float* __restrict__ w2,
                                             float* __restrict__ res,
                                             float* __restrict__ d0, float* __restrict__ d1,
                                             float* __restrict__ d2, int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = ar[i] + br[i];
        rr[i] = v;
        sum += v * v;
    }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    float* o0 = d0 + (size_t)row * ncols;
    float* o1 = d1 + (size_t)row * ncols;
    float* o2 = d2 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float xh = rr[i] * scale;
        o0[i] = xh * w0[i];
        o1[i] = xh * w1[i];
        o2[i] = xh * w2[i];
    }
}

// ---- gemma4: residual add + layer scale + next attn_norm EMITTED q8_1 (the mixer input is
// consumed only by quantized matmuls). res = (a+b)*c; norm chain = rms_norm_f32; quantize
// epilogue = rms_norm_q8_1's warp-per-4-blocks float4 form (bit-identical to quantize_q8_1). ----
extern "C" __global__ void add_scale_rms_norm_q8_1(const float* __restrict__ a, const float* __restrict__ b,
                                                   float c, const float* __restrict__ w,
                                                   float* __restrict__ res,
                                                   signed char* __restrict__ out_q, float* __restrict__ out_d,
                                                   int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (ar[i] + br[i]) * c;
        rr[i] = v;
        sum += v * v;
    }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- E4B glue fusion (2026-07-12): rms-normalize `a` FIRST (the PLE tail's post_norm of y),
// then the add_scale_rms_norm_q8_1 program verbatim on (a_normed, b). Replaces the separate
// rms_norm_f32(y) launch per layer. Two full-row reductions, one launch. ----
extern "C" __global__ void rms_pre_add_scale_rms_norm_q8_1(
        const float* __restrict__ a, const float* __restrict__ wa,
        const float* __restrict__ b,
        float c, const float* __restrict__ w,
        float* __restrict__ res,
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    __shared__ float s[128];
    // SINGLE-PHASE reductions (wave 4): four simultaneous sums in one pass —
    //   S1 = sum(a^2)            -> ascale
    //   S2 = sum((a*wa)^2), S3 = sum(a*wa*b), S4 = sum(b^2)
    // then sum(v^2) with v = (a*ascale*wa + b)*c expands ALGEBRAICALLY to
    //   c^2 * (ascale^2*S2 + 2*ascale*S3 + S4)
    // — one barrier round instead of two full reduction phases. FP-order differs from the
    // sequential two-phase form (expansion rounding); the argmax/chat gates arbitrate.
    float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float a0 = ar[i]; float b0 = br[i]; float awa = a0 * wa[i];
        s1 += a0 * a0; s2 += awa * awa; s3 += awa * b0; s4 += b0 * b0;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
        s3 += __shfl_down_sync(0xffffffff, s3, o);
        s4 += __shfl_down_sync(0xffffffff, s4, o);
    }
    int wid = tid >> 5;
    if ((tid & 31) == 0) { s[wid] = s1; s[32 + wid] = s2; s[64 + wid] = s3; s[96 + wid] = s4; }
    __syncthreads();
    if (tid < 32) {
        int nw = (blockDim.x + 31) / 32;
        float v1 = (tid < nw) ? s[tid] : 0.0f;
        float v2 = (tid < nw) ? s[32 + tid] : 0.0f;
        float v3 = (tid < nw) ? s[64 + tid] : 0.0f;
        float v4 = (tid < nw) ? s[96 + tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, o);
            v2 += __shfl_down_sync(0xffffffff, v2, o);
            v3 += __shfl_down_sync(0xffffffff, v3, o);
            v4 += __shfl_down_sync(0xffffffff, v4, o);
        }
        if (tid == 0) { s[0] = v1; s[1] = v2; s[2] = v3; s[3] = v4; }
    }
    __syncthreads();
    float ascale = rsqrtf(s[0] / ncols + eps);
    float sumv2 = c * c * (ascale * ascale * s[1] + 2.0f * ascale * s[2] + s[3]);
    float scale = rsqrtf(sumv2 / ncols + eps);
    // store pass: rr written here (the reduction pass no longer writes it).
    for (int i = tid; i < ncols; i += blockDim.x) {
        float an = (ar[i] * ascale) * wa[i];
        rr[i] = (an + br[i]) * c;
    }
    __syncthreads();
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    const float4* w4 = (const float4*)w;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;
        float4 xv = x4[i4];
        float4 wv = w4[i4];
        float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                               (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                              (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- gemma4: residual add + the three attn_out norms with TWO outputs emitted q8_1
// (zsh -> the quantized gate/up pair input; moe_in -> the quantized expert input) and the
// router input f32. Chains: add + rms_norm3 + quantize_q8_1 verbatim per element. ----
extern "C" __global__ void add_rms_norm3_q8z_f32(const float* __restrict__ a, const float* __restrict__ b,
                                                 const float* __restrict__ w0, const float* __restrict__ w1,
                                                 const float* __restrict__ w2,
                                                 float* __restrict__ res,
                                                 signed char* __restrict__ q0, float* __restrict__ d0,
                                                 float* __restrict__ out1,
                                                 signed char* __restrict__ q2, float* __restrict__ d2,
                                                 int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* ar = a + (size_t)row * ncols;
    const float* br = b + (size_t)row * ncols;
    float* rr = res + (size_t)row * ncols;
    int nblk = ncols / 32;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = ar[i] + br[i];
        rr[i] = v;
        sum += v * v;
    }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    // f32 router output (plain rms_norm write)
    float* o1 = out1 + (size_t)row * ncols;
    for (int i = tid; i < ncols; i += blockDim.x) o1[i] = rr[i] * scale * w1[i];
    // q8 outputs: the rms_norm_q8_1 warp-per-4-blocks float4 epilogue, once per weight vector.
    int lane = tid & 31;
    const float4* x4 = (const float4*)rr;
    {
        signed char* bq = q0 + (size_t)row * ncols;
        float* bd = d0 + (size_t)row * nblk;
        const float4* w4 = (const float4*)w0;
        for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
            int i4 = quad * 32 + lane;
            float4 xv = x4[i4];
            float4 wv = w4[i4];
            float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                                   (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            float d = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                                  (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
            ((char4*)bq)[i4] = qv;
            if ((lane & 7) == 0) bd[quad * 4 + (lane >> 3)] = d;
        }
    }
    {
        signed char* bq = q2 + (size_t)row * ncols;
        float* bd = d2 + (size_t)row * nblk;
        const float4* w4 = (const float4*)w2;
        for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
            int i4 = quad * 32 + lane;
            float4 xv = x4[i4];
            float4 wv = w4[i4];
            float4 v = make_float4((xv.x * scale) * wv.x, (xv.y * scale) * wv.y,
                                   (xv.z * scale) * wv.z, (xv.w * scale) * wv.w);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            float d = amax / 127.0f;
            float id = d > 0.0f ? 1.0f / d : 0.0f;
            char4 qv = make_char4((signed char)__float2int_rn(v.x * id), (signed char)__float2int_rn(v.y * id),
                                  (signed char)__float2int_rn(v.z * id), (signed char)__float2int_rn(v.w * id));
            ((char4*)bq)[i4] = qv;
            if ((lane & 7) == 0) bd[quad * 4 + (lane >> 3)] = d;
        }
    }
}

// ---- gemma4: q AND k roped in ONE launch (two segments on grid.x; per-row math =
// rope_neox_f32 / rope_neox_ff_f32 verbatim; ff = nullptr -> plain). ----
extern "C" __global__ void rope_neox2_f32(float* __restrict__ q, float* __restrict__ k,
                                          const int* __restrict__ pos,
                                          int head_dim, int n_dims, int nh_q, int nh_k,
                                          int n_tokens, float theta_scale, float freq_scale,
                                          const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int total_q = nh_q * n_tokens;
    float* base; int tok;
    if (hr < total_q) { base = q + (size_t)hr * head_dim; tok = hr / nh_q; }
    else { int r = hr - total_q; base = k + (size_t)r * head_dim; tok = r / nh_k; }
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * sn;
    base[j + half] = x0 * sn + x1 * c;
}

// bf16-emit twin (31B glue lane 2026-07-23): identical rope math + stores, ALSO emits the
// post-rope values as bf16 (the exact __float2bfloat16 the FA pre-converter applied) — the FA
// operands come out of this launch, killing the separate q/k f32->bf16 convert + re-read.
extern "C" __global__ void rope_neox2_bf16e_f32(float* __restrict__ q, float* __restrict__ k,
                                          __nv_bfloat16* __restrict__ qb, __nv_bfloat16* __restrict__ kb,
                                          const int* __restrict__ pos,
                                          int head_dim, int n_dims, int nh_q, int nh_k,
                                          int n_tokens, float theta_scale, float freq_scale,
                                          const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int total_q = nh_q * n_tokens;
    float* base; __nv_bfloat16* baseb; int tok;
    if (hr < total_q) { base = q + (size_t)hr * head_dim; baseb = qb + (size_t)hr * head_dim; tok = hr / nh_q; }
    else { int r = hr - total_q; base = k + (size_t)r * head_dim; baseb = kb + (size_t)r * head_dim; tok = r / nh_k; }
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) * freq_scale;
    if (ff) theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), sn = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    float y0 = x0 * c - x1 * sn;
    float y1 = x0 * sn + x1 * c;
    base[j]        = y0;
    base[j + half] = y1;
    baseb[j]        = __float2bfloat16(y0);
    baseb[j + half] = __float2bfloat16(y1);
}

// ---- tiny async setters/packers (gemma spec round: zero host-memory transfers) ----
// gemma4-E4B: gather layer il's per-layer-input rows out of the [t][n_layer][n_epl] prologue
// buffer into a dense [t][n_epl] operand (row t at src offset (t*stride + off)).
extern "C" __global__ void copy_rows_strided_f32(
        const float* __restrict__ src, float* __restrict__ dst,
        int row_elems, int n_rows, long src_stride, long src_off) {
    int r = blockIdx.y;
    if (r >= n_rows) return;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < row_elems; j += gridDim.x * blockDim.x)
        dst[(size_t)r * row_elems + j] = src[(size_t)r * src_stride + src_off + j];
}

// Place dense [row][row_elems] source rows into one column range of a strided destination.
// This is the byte-preserving inverse of copy_rows_strided_f32 and performs no arithmetic.
extern "C" __global__ void place_rows_strided_f32(
        const float* __restrict__ src, float* __restrict__ dst,
        int row_elems, int n_rows, long dst_stride, long dst_off) {
    int r = blockIdx.y;
    if (r >= n_rows) return;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < row_elems; j += gridDim.x * blockDim.x)
        dst[(size_t)r * dst_stride + dst_off + j] = src[(size_t)r * row_elems + j];
}

extern "C" __global__ void u32_set_k(unsigned int* __restrict__ dst, unsigned int v, int idx) {
    dst[idx] = v;
}

// FR-Spec trim id translate (gemma async draft round): buf[idx] holds a TRIM-space argmax --
// map it to the full-vocab token id in place (d2t = ranked-id table). Single-slot, async.
extern "C" __global__ void u32_map_k(unsigned int* __restrict__ buf,
                                     const unsigned int* __restrict__ map, int idx) {
    buf[idx] = map[buf[idx]];
}

// pos-row fill from a device counter: dst[i] = ctr[0] + i (verify-stream rope positions —
// no host pos value; one launch per verify).
extern "C" __global__ void i32_iota_from(const int* __restrict__ ctr, int* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = ctr[0] + i;
}

// counter += v (device-slot append advance for t rows; the +1 twin is inc_seqlen).
extern "C" __global__ void i32_add_k(int* __restrict__ d, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) d[0] += v;
}

// i32 twin (device-len counters, graph arc): async single-slot store, value rides the arg.
extern "C" __global__ void i32_set_k(int* dst, int v, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[idx] = v;
}
// pack: out[0..n1) = a[off_a..], out[n1..n1+n2) = b[0..n2) (single dtoh follows).
extern "C" __global__ void u32_pack2(const unsigned int* __restrict__ a, int off_a, int n1,
                                     const unsigned int* __restrict__ b, int n2,
                                     unsigned int* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n1) out[i] = a[off_a + i];
    else if (i < n1 + n2) out[i] = b[i - n1];
}

// ---- gemma4 R1: GELU(tanh approx) * up GLU epilogue. Constants = ggml's GELU_COEF_A /
// SQRT_2_OVER_PI so the activation matches llama.cpp's CUDA gelu op float-for-float. ----
extern "C" __global__ void gelu_tanh_mul_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                             float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float t = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        dst[i] = 0.5f * x * (1.0f + t) * up[i];
    }
}

// ---- E4B/gemma glue fusion (2026-07-12): GELU(tanh)*up with the activation EMITTED q8_1
// (per-32-block amax quantize, bit-identical to quantize_q8_1's rounding — the add_scale
// emit epilogue's program). The down/proj matmul then rides matmul_pre: one launch replaces
// gelu_tanh_mul_f32 + quantize_q8_1. Row-major [nrows, ncols]; ncols % 128 == 0. ----
extern "C" __global__ void gelu_tanh_mul_q8_1(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              float* __restrict__ act,
                                              signed char* __restrict__ out_q,
                                              float* __restrict__ out_d,
                                              int ncols) {
    MEMRA_PDL_ENTRY();
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int nblk = ncols / 32;
    const float* gr = gate + (size_t)row * ncols;
    const float* ur = up + (size_t)row * ncols;
    float* arow = act + (size_t)row * ncols;
    signed char* base_q = out_q + (size_t)row * ncols;
    float* base_d = out_d + (size_t)row * nblk;
    for (int quad = tid >> 5; quad < nblk / 4; quad += blockDim.x >> 5) {
        int i4 = quad * 32 + lane;   // float4 index
        const float4 g4 = ((const float4*)gr)[i4];
        const float4 u4 = ((const float4*)ur)[i4];
        float vx[4] = {g4.x, g4.y, g4.z, g4.w};
        float ux[4] = {u4.x, u4.y, u4.z, u4.w};
        float o[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            float x = vx[e];
            float t = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
            o[e] = 0.5f * x * (1.0f + t) * ux[e];
        }
        ((float4*)arow)[i4] = make_float4(o[0], o[1], o[2], o[3]);
        float amax = fmaxf(fmaxf(fabsf(o[0]), fabsf(o[1])), fmaxf(fabsf(o[2]), fabsf(o[3])));
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        char4 qv = make_char4((signed char)__float2int_rn(o[0] * id), (signed char)__float2int_rn(o[1] * id),
                              (signed char)__float2int_rn(o[2] * id), (signed char)__float2int_rn(o[3] * id));
        ((char4*)base_q)[i4] = qv;
        if ((lane & 7) == 0) base_d[quad * 4 + (lane >> 3)] = d;
    }
}

// ---- gemma4 R9: RoPE NEOX with per-dim freq factors (rope_freqs.weight, global layers).
// theta = pos * base^(-2j/d) / ff[j] (llama rope_ext freq_factors semantics, freq_scale = 1). ----
extern "C" __global__ void rope_neox_ff_f32(float* __restrict__ x, const int* __restrict__ pos,
                                            int head_dim, int n_dims, int n_heads,
                                            float theta_scale, float freq_scale,
                                            const float* __restrict__ ff) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int tok = hr / n_heads;
    float* base = x + (size_t)hr * head_dim;
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta), s = sinf(theta);
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * s;
    base[j + half] = x0 * s + x1 * c;
}

// ---- rope_neox_ff + the YaRN attention factor (qwen4_exp long-context lane):
// cos/sin scaled by mscale = 0.1*ln(factor)+1 (transformers `attention_scaling`), ff =
// per-pair frequency divisors (memra-gguf yarn_frequency_divisors). With ff = ones and
// mscale = 1.0 this reproduces rope_neox_f32 BIT-IDENTICALLY (x/1.0 and x*1.0 are exact) —
// the factor-1.0 identity gate rides on that. ----
extern "C" __global__ void rope_neox_ffm_f32(float* __restrict__ x, const int* __restrict__ pos,
                                             int head_dim, int n_dims, int n_heads,
                                             float theta_scale, float freq_scale,
                                             const float* __restrict__ ff, float mscale) {
    int hd2 = head_dim / 2;
    int j = threadIdx.x;
    if (j >= hd2) return;
    int hr = blockIdx.x;
    int tok = hr / n_heads;
    float* base = x + (size_t)hr * head_dim;
    int half = n_dims / 2;
    if (j >= half) return;
    float theta = (float)pos[tok] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
    float c = cosf(theta) * mscale, s = sinf(theta) * mscale;
    float x0 = base[j];
    float x1 = base[j + half];
    base[j]        = x0 * c - x1 * s;
    base[j + half] = x0 * s + x1 * c;
}

// ---- elementwise ----
// float4-vectorized (H100 sweep 2026-07-26: scalar version ran 43us at m=512 ffn width vs a
// ~20us BW floor). Same op per element, same order — BIT-IDENTICAL to the scalar form; the
// tail loop covers n % 4 (thread grid covers ceil(n/4) lanes of 4).
extern "C" __global__ void silu_mul_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                        float* __restrict__ dst, int n) {
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 g = *(const float4*)(gate + base);
        float4 u = *(const float4*)(up + base);
        float4 o;
        o.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        o.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        o.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        o.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        *(float4*)(dst + base) = o;
    } else {
        for (int i = base; i < n; i++) {
            float g = gate[i];
            dst[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }
}
// f16out twin (task #17, nsys round-26 gap anatomy): the SwiGLU epilogue also emits the fp16
// GEMM operand for the down projection, removing the standalone memra_f16_cvt pass (a full extra
// HBM read+write of act). BIT-IDENTICAL class: dst gets the same floats as silu_mul_f32 and
// dst16[i] = __float2half(dst[i]) == exactly what memra_f16_cvt_kernel would have emitted.
extern "C" __global__ void silu_mul_f16out_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                               float* __restrict__ dst, __half* __restrict__ dst16, int n) {
    int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    int base = i4 * 4;
    if (base + 3 < n) {
        float4 g = *(const float4*)(gate + base);
        float4 u = *(const float4*)(up + base);
        float4 o;
        o.x = (g.x / (1.0f + expf(-g.x))) * u.x;
        o.y = (g.y / (1.0f + expf(-g.y))) * u.y;
        o.z = (g.z / (1.0f + expf(-g.z))) * u.z;
        o.w = (g.w / (1.0f + expf(-g.w))) * u.w;
        *(float4*)(dst + base) = o;
        __half2* h2 = (__half2*)(dst16 + base);
        h2[0] = __halves2half2(__float2half(o.x), __float2half(o.y));
        h2[1] = __halves2half2(__float2half(o.z), __float2half(o.w));
    } else {
        for (int i = base; i < n; i++) {
            float g = gate[i];
            float o = (g / (1.0f + expf(-g))) * up[i];
            dst[i] = o;
            dst16[i] = __float2half(o);
        }
    }
}
// FFN SwiGLU epilogue fusion (RANK3 LEVER 2). Folds the per-tensor NVFP4 macro-scale of the gate
// and up matmuls INTO the silu*mul, removing the two separate `scale_f32` launches per dense FFN
// layer. BIT-IDENTICAL to scale_f32(gate,gs); scale_f32(up,us); silu_mul_f32(gate,up,dst): same
// float ops in the same order — multiply by scale, then silu(g'), then multiply by up'. For
// non-NVFP4 weights gs==us==1.0, so this reduces exactly to silu_mul_f32.
extern "C" __global__ void silu_mul_scaled_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                               float gs, float us, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i] * gs; dst[i] = (g / (1.0f + expf(-g))) * (up[i] * us); }
}
// swigluoai (MiniMax-M3 / GPT-OSS): clamped SwiGLU. Math 1:1 vs llama.cpp
// ggml_cuda_op_swiglu_oai_single (unary.cuh:107): gate clamps ABOVE only, up clamps both sides,
// swish uses alpha inside the sigmoid, and the linear term is (1 + up). gs/us fold the NVFP4
// per-tensor macro-scales exactly like silu_mul_scaled_f32 (gs==us==1.0 for non-NVFP4).
extern "C" __global__ void swigluoai_mul_scaled_f32(const float* __restrict__ gate, const float* __restrict__ up,
                                                    float gs, float us, float alpha, float limit,
                                                    float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = fminf(gate[i] * gs, limit);
        float g = fmaxf(fminf(up[i] * us, limit), -limit);
        dst[i] = (x / (1.0f + expf(-x * alpha))) * (1.0f + g);
    }
}
// RANK2 LEVER (q8_1 quant-fold): FFN SwiGLU epilogue that ALSO emits the q8_1 quantization of its
// own output, so ffn_down's activation is produced pre-quantized and the standalone quantize_q8_1
// launch is removed (1 fewer launch + no f32 `act` HBM round-trip per dense FFN layer). The down-proj
// activation has EXACTLY ONE consumer (ffn_down's matvec), so folding the quant into the producer is
// free — silu*mul already touches every element once; here each thread owns one 32-block, computes
// its 32 silu*mul values, finds amax over the block, and writes q8_1 (aq int8 + ad f32 scale).
// BIT-IDENTICAL q8_1 to scale->silu_mul->quantize_q8_1: same float silu*mul (g*gs, up*us), same
// d=amax/127, same id=1/d, same __float2int_rn rounding. n must be a multiple of 32 (n_ff always is).
// WARP-PER-BLOCK (decode elementwise-soup fix, ncu 2026-07-03): lane j of a warp owns element j of
// one 32-block -> fully coalesced 128B gate/up reads + 32B q8 writes. The old thread-owns-block form
// read 32 SEQUENTIAL floats per thread (32-way uncoalesced) on a nblk-thread grid (384 threads for
// n_ff=12288) and measured 22.7us vs ~0.15us of actual DRAM traffic. amax via __shfl_xor max is
// order-independent (max is associative+commutative) -> d and every q8 value stay BIT-IDENTICAL.
extern "C" __global__ void silu_mul_scaled_q8_1(
        const float* __restrict__ gate, const float* __restrict__ up, float gs, float us,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global 32-block index
    int lane = threadIdx.x & 31;
    int nblk = n / 32;
    if (warp >= nblk) return;
    int i = warp * 32 + lane;
    float g = gate[i] * gs;
    float r = (g / (1.0f + expf(-g))) * (up[i] * us);   // silu(g)*up, bit-identical
    float amax = fabsf(r);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[i] = (signed char)__float2int_rn(r * id);
    if (lane == 0) out_d[warp] = d;
}

// float4-vectorized (elementwise -> bit-identical; H100 sweep 2026-07-26). Tail in-kernel.
// MOE TAIL FUSION M1: (acc0 + acc1) + sh in ONE launch — the direct-join add and the
// shexp-overlap apply were adjacent elementwise launches on e's stream. Per element the
// sequence is exactly [a+b] then [+ sh*1.0] (the add_scaled_rows program with scale[0]=1
// folded), so the values are BIT-IDENTICAL to the split pair.
extern "C" __global__ void add3_f32(const float* __restrict__ a, const float* __restrict__ b,
                                    const float* __restrict__ sh, const float* __restrict__ scale,
                                    float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Exact split-pair ops: add_f32's plain add, then add_scaled_rows' contracted
        // mul-add (nvcc -fmad turns `dst += src*scale` into one FMA) — forced with
        // intrinsics so this kernel cannot round differently from the pair it replaces.
        float v = __fadd_rn(a[i], b[i]);
        dst[i] = __fmaf_rn(sh[i], scale[0], v);
    }
}

extern "C" __global__ void add_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ dst, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 x = *(const float4*)(a + base);
        float4 y = *(const float4*)(b + base);
        *(float4*)(dst + base) = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    } else {
        for (int i = base; i < n; i++) dst[i] = a[i] + b[i];
    }
}
// y[i] *= s. NVFP4 per-tensor macro-scale broadcast over the whole matmul output.
extern "C" __global__ void scale_f32(float* __restrict__ y, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= s;
}
// MEMRA_FULL_PREC dequant-on-use: expand a bf16-resident matmul weight (u16 LE, upper 16 bits of
// f32) to a transient f32 scratch that feeds the SAME cuBLASLt f32 GEMV the Float arm uses. This is
// bit-identical to the load-time bf16->f32 dequant (dequant::bf16_to_f32), just deferred to keep
// VRAM at 2 B/w resident instead of 4.
extern "C" __global__ void bf16_to_f32(const unsigned short* __restrict__ in,
                                       float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float(((unsigned int)in[i]) << 16);
}
extern "C" __global__ void mul_f32(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] * b[i];
}

// ---- naive SDPA for one token-batch, GQA, causal. Correctness oracle (no flash). ----
// Q: [head_dim, n_head, T], K/V: [head_dim, n_head_kv, T_kv]. out: [head_dim, n_head, T].
// One block per (head, query-token). threads cooperate over T_kv. Scores in smem.
extern "C" __global__ void sdpa_naive_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                          const float* __restrict__ V, float* __restrict__ O,
                                          int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                          float scale, int causal) {
    int head = blockIdx.x;
    int qt = blockIdx.y;                 // query token index (0..T-1)
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);   // GQA mapping
    int tid = threadIdx.x;
    extern __shared__ float scores[];    // [T_kv]

    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    // query absolute position = (T_kv - T) + qt  (kv holds past + current)
    int q_pos = (T_kv - T) + qt;

    // scores[t] = scale * dot(q, K[:,kv_head,t])
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    // softmax over scores[0..T_kv) — single thread for simplicity (T_kv small in M0 tests)
    __shared__ float red[1];
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
        red[0] = 0.0f;
    }
    __syncthreads();
    // out[d] = sum_t scores[t] * V[d,kv_head,t]
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

// Global-memory-scores twin of sdpa_naive_f32 (lane/hermes-perf-fixes, 2026-08-23). The smem
// kernel's T_kv*4-byte dynamic shared memory exceeds the 48KB launch bound past T_kv=12288 —
// the measured dspark/full-attn long-ctx crash class (DFLASH2-EVAL B2's non-windowed sibling;
// the windowed layers got sdpa_naive_w_lo, the PLAIN full-attn path had nothing). Same loop
// structure, same reduction order, same single-thread softmax — only the scores row lives in
// a caller-provided [n_head * T * T_kv] f32 workspace instead of shared memory, so the output
// is BYTE-IDENTICAL to sdpa_naive_f32 wherever both launch (kernel_check pins it) and the
// launch bound disappears. __threadfence_block + __syncthreads reproduce the smem kernel's
// intra-block visibility ordering for the gmem row.
extern "C" __global__ void sdpa_naive_gmem_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                               const float* __restrict__ V, float* __restrict__ O,
                                               float* __restrict__ SC,
                                               int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                               float scale, int causal) {
    int head = blockIdx.x;
    int qt = blockIdx.y;                 // query token index (0..T-1)
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);   // GQA mapping
    int tid = threadIdx.x;
    float* scores = SC + ((size_t)qt * n_head + head) * (size_t)T_kv;   // [T_kv]

    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    // query absolute position = (T_kv - T) + qt  (kv holds past + current)
    int q_pos = (T_kv - T) + qt;

    // scores[t] = scale * dot(q, K[:,kv_head,t])
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        scores[t] = acc;
    }
    __threadfence_block();
    __syncthreads();
    // softmax over scores[0..T_kv) — single thread, exactly the smem kernel's order
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
    }
    __threadfence_block();
    __syncthreads();
    // out[d] = sum_t scores[t] * V[d,kv_head,t]
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

// Windowed twin of sdpa_naive_f32 (gemma4 R6 SWA): additionally masks keys OLDER than
// q_pos - (window-1) — llama's sliding-window mask (window keys incl self). window <= 0 = none.
extern "C" __global__ void sdpa_naive_w_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                            const float* __restrict__ V, float* __restrict__ O,
                                            int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                            float scale, int causal, int window) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        if (window > 0 && t < q_pos - (window - 1)) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    __shared__ float red[1];
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
        red[0] = 0.0f;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

// DFlash2 grouped dynamic causal conv (z-lab reference _grouped_dynamic_convolve;
// DFLASH2-EVAL-20260820.md §2.1): a causal ksize-tap depthwise conv over the BLOCK rows
// (block-local — row 0 zero-pads its missing predecessors) with a per-channel BASE
// kernel plus a per-position per-GROUP dynamic coefficient:
//   out[p,c] = sum_{o < ksize, o <= p}
//       (BASE[half][o][c] + DYN[p][half][o][c/group_size]) * X[p-o][c]
// `half` selects the module's prepare (0: convolves the sublayer INPUT) vs finish
// (1: convolves the sublayer OUTPUT) application; BOTH halves ride the same DYN
// projection, computed from the pre-conv input rows.
// X/OUT: [rows, hidden]; DYN: [rows, 2*ksize*groups] (row-major view of
// [rows, 2, ksize, groups]); BASE: [2, ksize, hidden] flattened.
extern "C" __global__ void dflash2_dynconv_f32(const float* __restrict__ X,
                                               const float* __restrict__ DYN,
                                               const float* __restrict__ BASE,
                                               float* __restrict__ OUT,
                                               int rows, int hidden, int group_size,
                                               int ksize, int half) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hidden) return;
    int p = idx / hidden, c = idx % hidden;
    int groups = hidden / group_size;
    int g = c / group_size;
    const float* baseh = BASE + (size_t)half * ksize * hidden;
    const float* dynp = DYN + (size_t)p * 2 * ksize * groups + (size_t)half * ksize * groups;
    float acc = 0.0f;
    for (int o = 0; o < ksize && o <= p; o++) {
        acc += (baseh[o * hidden + c] + dynp[o * groups + g]) * X[(size_t)(p - o) * hidden + c];
    }
    OUT[idx] = acc;
}

// Per-row top-k (k <= 32) over a [n_rows, n_cols] f32 matrix (DFlash2 candidate
// selector: k=16 over the vocab). One block per row: per-thread top-k over strided
// columns (descending insertion list), then a single-thread k-way merge of the
// per-thread sorted lists. Ties break to the LOWER column index. Outputs are
// value-descending; empty slots (n_cols < k) write idx 0xffffffff.
extern "C" __global__ void topk_rows_f32(const float* __restrict__ L, int n_rows,
                                         int n_cols, int k,
                                         float* __restrict__ vals,
                                         unsigned int* __restrict__ idxs) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;
    int nth = blockDim.x;
    const float* base = L + (size_t)row * n_cols;
    const float NEG_INF = __int_as_float(0xff800000);
    const int SENTINEL = 0x7fffffff;
    float lv[32];
    int li[32];
    for (int j = 0; j < k; j++) { lv[j] = NEG_INF; li[j] = SENTINEL; }
    for (int c = tid; c < n_cols; c += nth) {
        float v = base[c];
        if (v > lv[k - 1] || (v == lv[k - 1] && c < li[k - 1])) {
            int j = k - 1;
            while (j > 0 && (lv[j - 1] < v || (lv[j - 1] == v && li[j - 1] > c))) {
                lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--;
            }
            lv[j] = v; li[j] = c;
        }
    }
    extern __shared__ unsigned char tk_smem[];
    float* sv = (float*)tk_smem;                          // [nth*k]
    int* si = (int*)(tk_smem + (size_t)nth * k * 4);      // [nth*k]
    for (int j = 0; j < k; j++) { sv[tid * k + j] = lv[j]; si[tid * k + j] = li[j]; }
    __syncthreads();
    if (tid == 0) {
        int head[1024];
        for (int t = 0; t < nth; t++) head[t] = 0;
        for (int out = 0; out < k; out++) {
            float bv = NEG_INF; int bi = SENTINEL; int bt = -1;
            for (int t = 0; t < nth; t++) {
                if (head[t] >= k) continue;
                float v = sv[t * k + head[t]];
                int ii = si[t * k + head[t]];
                if (ii == SENTINEL) continue;
                if (bt < 0 || v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; bt = t; }
            }
            if (bt >= 0) head[bt]++;
            vals[(size_t)row * k + out] = bv;
            idxs[(size_t)row * k + out] = (bi == SENTINEL) ? 0xffffffffu : (unsigned int)bi;
        }
    }
}

// ---- SHARDED exact top-k twin pair (lane/glm5-matvec door K, MEMRA_TOPK_SHARDS). ----
// topk_rows_f32 puts n_rows blocks on the card — the DFlash2 selector launches 15 blocks on
// a 188-SM part and reads 9.3 MB in 1.31 ms (7 GB/s, diet-battery c8-ship census). Top-k
// under the total order (value desc, column asc) is a DISCRETE SELECTION: any algorithm
// applying the same order yields the identical (value, index) list, so a shard split is
// output-identical by construction — no numeric class, no float reassociation.
//   Stage 1: per-(row, shard) partial top-k over the shard's column range, the standing
//   kernel's insertion/merge code VERBATIM on global column indices.
//   Stage 2: per-row k-way merge across the shard lists, the standing kernel's merge rules
//   verbatim (value desc, lower index on ties, 0xffffffff = exhausted slot).
// NaN handling is inherited unchanged: every comparison against a NaN is false, so a NaN
// never enters a list in either algorithm. Gated by glm5_matvec_doors_gpu (bit-compare vs
// the standing kernel incl. planted-tie fixtures).
extern "C" __global__ void topk_rows_shard_f32(const float* __restrict__ L, int n_rows,
                                               int n_cols, int k, int n_shards,
                                               float* __restrict__ pvals,
                                               unsigned int* __restrict__ pidxs) {
    int row = blockIdx.x;
    int sh = blockIdx.y;
    if (row >= n_rows || sh >= n_shards) return;
    int tid = threadIdx.x;
    int nth = blockDim.x;
    int shard_w = (n_cols + n_shards - 1) / n_shards;
    int c0 = sh * shard_w;
    int c1 = min(c0 + shard_w, n_cols);
    const float* base = L + (size_t)row * n_cols;
    const float NEG_INF = __int_as_float(0xff800000);
    const int SENTINEL = 0x7fffffff;
    float lv[32];
    int li[32];
    for (int j = 0; j < k; j++) { lv[j] = NEG_INF; li[j] = SENTINEL; }
    for (int c = c0 + tid; c < c1; c += nth) {
        float v = base[c];
        if (v > lv[k - 1] || (v == lv[k - 1] && c < li[k - 1])) {
            int j = k - 1;
            while (j > 0 && (lv[j - 1] < v || (lv[j - 1] == v && li[j - 1] > c))) {
                lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--;
            }
            lv[j] = v; li[j] = c;
        }
    }
    extern __shared__ unsigned char tks_smem[];
    float* sv = (float*)tks_smem;                         // [nth*k]
    int* si = (int*)(tks_smem + (size_t)nth * k * 4);     // [nth*k]
    for (int j = 0; j < k; j++) { sv[tid * k + j] = lv[j]; si[tid * k + j] = li[j]; }
    __syncthreads();
    if (tid == 0) {
        int head[1024];
        for (int t = 0; t < nth; t++) head[t] = 0;
        float* ov = pvals + ((size_t)row * n_shards + sh) * k;
        unsigned int* oi = pidxs + ((size_t)row * n_shards + sh) * k;
        for (int out = 0; out < k; out++) {
            float bv = NEG_INF; int bi = SENTINEL; int bt = -1;
            for (int t = 0; t < nth; t++) {
                if (head[t] >= k) continue;
                float v = sv[t * k + head[t]];
                int ii = si[t * k + head[t]];
                if (ii == SENTINEL) continue;
                if (bt < 0 || v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; bt = t; }
            }
            if (bt >= 0) head[bt]++;
            ov[out] = bv;
            oi[out] = (bi == SENTINEL) ? 0xffffffffu : (unsigned int)bi;
        }
    }
}
extern "C" __global__ void topk_rows_shard_merge_f32(const float* __restrict__ pvals,
                                                     const unsigned int* __restrict__ pidxs,
                                                     int n_rows, int n_shards, int k,
                                                     float* __restrict__ vals,
                                                     unsigned int* __restrict__ idxs) {
    int row = blockIdx.x;
    if (row >= n_rows || threadIdx.x != 0) return;
    const float NEG_INF = __int_as_float(0xff800000);
    const float* pv = pvals + (size_t)row * n_shards * k;
    const unsigned int* pi = pidxs + (size_t)row * n_shards * k;
    int head[64];
    for (int s = 0; s < n_shards; s++) head[s] = 0;
    for (int out = 0; out < k; out++) {
        float bv = NEG_INF; unsigned int bi = 0xffffffffu; int bs = -1;
        for (int s = 0; s < n_shards; s++) {
            if (head[s] >= k) continue;
            float v = pv[s * k + head[s]];
            unsigned int ii = pi[s * k + head[s]];
            if (ii == 0xffffffffu) continue;
            if (bs < 0 || v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; bs = s; }
        }
        if (bs >= 0) head[bs]++;
        vals[(size_t)row * k + out] = bv;
        idxs[(size_t)row * k + out] = bi;
    }
}

// Lo-clipped twin of sdpa_naive_w_f32 (lane/dflash2-longctx, DFLASH2-EVAL §10.6(c)): the
// windowed kernel sizes its dynamic shared memory T_kv*4 bytes and SCANS every ctx key,
// so any windowed caller whose T_kv exceeds ~12k rows blows the 48KB default dynamic-smem
// bound and the launch dies with CUDA_ERROR_INVALID_VALUE (measured: DFlash2 drafting hard-
// fails at ctx 16,571/30,157, last success 9,510 — GATES-SMOKE-20260821 B2). Keys below
// kv_lo are outside EVERY query's window (caller passes kv_lo = max(0, (T_kv-T)+1-window),
// the oldest key visible to the OLDEST query row), so this twin never reads them: the score
// loop starts at kv_lo, shared memory holds T_kv-kv_lo floats, and the window/causal masks
// are unchanged. BYTE-IDENTICAL to sdpa_naive_w_f32 for the surviving keys: a masked key's
// score is -1e30 -> expf underflows to exactly +0.0 -> contributes exactly nothing to the
// max, the ascending-t sum, or the output accumulation, so skipping it removes only
// exact-zero terms from the same-order reductions (kernel_check `sdpa_naive_w_lo` pins the
// bit-identity; the >48KB arm pins the legacy launch failure).
extern "C" __global__ void sdpa_naive_w_lo_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                               const float* __restrict__ V, float* __restrict__ O,
                                               int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                               float scale, int causal, int window, int kv_lo) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];    // [T_kv - kv_lo]
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    for (int t = kv_lo + tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (causal && t > q_pos) acc = -1e30f;
        if (window > 0 && t < q_pos - (window - 1)) acc = -1e30f;
        scores[t - kv_lo] = acc;
    }
    __syncthreads();
    int n_sc = T_kv - kv_lo;
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < n_sc; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < n_sc; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < n_sc; t++) scores[t] *= inv;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = kv_lo; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t - kv_lo] * v[d];
        }
        o[d] = acc;
    }
}

// Island twin of sdpa_naive_w_f32 (lane/gemma-vision masked prefill): span_id[t_kv] labels
// each absolute kv position (-1 = text, >=0 = image-island id). A key inside the SAME island
// as the query is visible UNCONDITIONALLY (bidirectional island, per the reference's
// non-causal image batch: llama.cpp mtmd-helper llama_set_causal_attn(false) around each
// image chunk); everything else keeps the causal + sliding-window law. window <= 0 = none.
// Image islands are capped at 280 tokens (< window 1024), so island-internal visibility and
// the window mask never conflict on SWA layers.
extern "C" __global__ void sdpa_naive_island_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                                 const float* __restrict__ V, float* __restrict__ O,
                                                 const int* __restrict__ span_id,
                                                 int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                                 float scale, int window) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    int q_span = span_id[q_pos];
    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        bool same_island = (q_span >= 0) && (span_id[t] == q_span);
        bool causal_ok = (t <= q_pos) && (window <= 0 || t >= q_pos - (window - 1));
        if (!same_island && !causal_ok) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    __shared__ float red[1];
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
        red[0] = 0.0f;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

// MoE router GEMV (MEMRA_ROUTER_KERNEL=1): logits[t][e] = dot(W[e], x[t]) — replaces ~200
// cuBLASLt dispatches/round (4% of the 35B spec round loop, 2026-07-10 MEMRA_PROFILE_SPEC=2).
// One warp per (expert, token); fixed-stride f32 accumulation + standard warp reduce —
// DETERMINISTIC but a DIFFERENT FP order than cuBLAS: new numeric config, the router feeds
// top-k selection (discontinuous) so the full battery + MOE_GATE oracle arbitrate adoption.
extern "C" __global__ void router_gemv_f32(
        const float* __restrict__ w,   // [n_experts, n_embd] row-major
        const float* __restrict__ x,   // [t, n_embd]
        float* __restrict__ y,         // [t, n_experts]
        int n_embd, int n_experts, int t) {
    const int e = blockIdx.x;
    const int tok = blockIdx.y;
    if (e >= n_experts || tok >= t) return;
    const float* wr = w + (size_t) e * n_embd;
    const float* xr = x + (size_t) tok * n_embd;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += 32) s += wr[i] * xr[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if (threadIdx.x == 0) y[(size_t) tok * n_experts + e] = s;
}

// 8-warp twin (2026-07-31, the H100 q35 decode dig): the warp-per-(expert,token) form is
// LATENCY-bound at t=1 — 128 lone-warp CTAs each walking n_embd/32 serial load steps put
// the router at 19.9us/layer = 14.8% of the whole decode step on the 132-SM part. Eight
// warps split the row (n_embd/256 steps) and tree-reduce through smem. NEW FP ORDER vs
// the warp twin (near-tie routing can flip) — battery-gated numeric config; the host
// picks this twin per MEMRA_ROUTER_V2 with the warp form as the rollback seam.
extern "C" __global__ void router_gemv_f32_w8(
        const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
        int n_embd, int n_experts, int t) {
    const int e = blockIdx.x;
    const int tok = blockIdx.y;
    if (e >= n_experts || tok >= t) return;
    const float* wr = w + (size_t) e * n_embd;
    const float* xr = x + (size_t) tok * n_embd;
    float s = 0.0f;
    for (int i = threadIdx.x + threadIdx.y * 32; i < n_embd; i += 256) s += wr[i] * xr[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    __shared__ float ps[8];
    if (threadIdx.x == 0) ps[threadIdx.y] = s;
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float acc = 0.0f;
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) acc += ps[wi];
        y[(size_t) tok * n_experts + e] = acc;
    }
}

// SHEXP GATE dot (2026-07-31, the q35 "cublasLt 40/step" decode dig): qwen35moe's shared-
// expert sigmoid gate is a 1-output dot per token per layer; cuBLASLt served it as an
// m=1,n=t,k=n_embd GEMM whose splitKreduce_kernel ran 40x/step at 14.3us (~10% of the H100
// decode step). One fused launch replaces linear+sigmoid: g[tok] = sigmoid(dot(x[tok],w)).
// Used for BOTH t=1 decode and small-t spec verify, so the two chains match per row by
// construction (one fold order). sigmoid form matches sigmoid_f32 (expf).
extern "C" __global__ void sigmoid_dot_rows_f32(
        const float* __restrict__ x,   // [t, n_embd]
        const float* __restrict__ w,   // [n_embd]
        float* __restrict__ g,         // [t]
        int n_embd, int t) {
    const int tok = blockIdx.x;
    if (tok >= t) return;
    const float* xr = x + (size_t) tok * n_embd;
    float s = 0.0f;
    for (int i = threadIdx.x + threadIdx.y * 32; i < n_embd; i += 256) s += xr[i] * w[i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    __shared__ float ps[8];
    if (threadIdx.x == 0) ps[threadIdx.y] = s;
    __syncthreads();
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        float acc = 0.0f;
#pragma unroll
        for (int wi = 0; wi < 8; ++wi) acc += ps[wi];
        g[tok] = 1.0f / (1.0f + expf(-acc));
    }
}

// FAST-ROUTER batch twin (lane/fast-router, 2026-08-02): the w8 form above is decode's
// m-invariant router, but at prefill m it is a GEMV program at GEMM shape — every
// (expert, token) block re-streams both full rows with zero operand reuse (the concat-prime
// exactness fix routed prefill through it and q35 board-2048 prefill paid -10%). This twin
// keeps EVERY per-row FP chain BIT-IDENTICAL to router_gemv_f32_w8 — same tid-strided k
// order (i = tid + j*256, one serial FFMA chain per output), same shfl_down tree, same
// serial 8-partial fold in warp order — and changes only WHERE operands come from: an
// 8x8 (expert x token) register tile per block turns 16 row-streams into 64 outputs
// (4 FFMAs per load instead of 0.5). m-invariance by construction: a row's chain never
// sees m. Edge tiles clamp row pointers (redundant compute) and guard stores.
template <int TT>
__device__ __forceinline__ void router_gemv_w8_batch_impl(
        const float* __restrict__ w,   // [n_experts, n_embd] row-major
        const float* __restrict__ x,   // [t, n_embd]
        float* __restrict__ y,         // [t, n_experts]
        int n_embd, int n_experts, int t) {
    const int e0 = blockIdx.x * 8;
    const int t0 = blockIdx.y * TT;
    if (e0 >= n_experts || t0 >= t) return;
    const int tid = threadIdx.x + threadIdx.y * 32;
    const float* wr[8]; const float* xr[TT];
#pragma unroll
    for (int a = 0; a < 8; a++) wr[a] = w + (size_t) min(e0 + a, n_experts - 1) * n_embd;
#pragma unroll
    for (int b = 0; b < TT; b++) xr[b] = x + (size_t) min(t0 + b, t - 1) * n_embd;
    float acc[8][TT];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int b = 0; b < TT; b++) acc[a][b] = 0.0f;
    for (int i = tid; i < n_embd; i += 256) {
        float wv[8], xv[TT];
#pragma unroll
        for (int a = 0; a < 8; a++) wv[a] = wr[a][i];
#pragma unroll
        for (int b = 0; b < TT; b++) xv[b] = xr[b][i];
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
            for (int b = 0; b < TT; b++) acc[a][b] += wv[a] * xv[b];
    }
    __shared__ float ps[8][8][TT];      // [warp][a][b]
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int b = 0; b < TT; b++) {
            float s = acc[a][b];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
            if (threadIdx.x == 0) ps[threadIdx.y][a][b] = s;
        }
    __syncthreads();
    // one output per thread: tid 0..8*TT-1 owns (a,b); the 8-partial fold keeps warp order
    // 0..7, identical to the w8 form's ps[0]+..+ps[7] serial fold.
    if (tid < 8 * TT) {
        const int a = tid / TT, b = tid % TT;
        const int e = e0 + a, tok = t0 + b;
        if (e < n_experts && tok < t) {
            float s = 0.0f;
#pragma unroll
            for (int wi = 0; wi < 8; ++wi) s += ps[wi][a][b];
            y[(size_t) tok * n_experts + e] = s;
        }
    }
}

// Tile arbitration (2026-08-02, 5090): TT=16 (half the per-output w-row traffic) measured
// SLOWER than TT=8 at every t (652 vs 448us at t=2048 — the 128-accumulator register
// pressure costs the occupancy the traffic gain needs); killed, crossover-router-tiles.jsonl
// is the record. Remaining gap to the m-DEPENDENT cuBLASLt GEMM (68us at t=2048) is the
// price of the fixed per-row 256-way chain: cuBLAS's k-split is exactly the reduction shape
// the exactness contract bans, and larger register tiles are occupancy-bound. A shared
// new-chain numeric config for decode+prefill would need full battery re-arbitration.
extern "C" __global__ void router_gemv_f32_w8_batch(
        const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
        int n_embd, int n_experts, int t) {
    router_gemv_w8_batch_impl<8>(w, x, y, n_embd, n_experts, t);
}

// (A same-shape 8-token batch twin of sigmoid_dot_rows_f32 was built and proven
// bit-identical on this lane, but measured SLOWER at every prefill t on the 5090 —
// launch-latency-bound out_f=1 op, ~7us/layer at m=2048. Killed per flags doctrine;
// research/fast-router-20260802/crossover-router.jsonl is the record.)

// f32 row permute: dst[idx[i]] = src[i] (the grouped-GEMM lane's CSR -> pair-id reorder).
extern "C" __global__ void rows_permute_f32(const float* __restrict__ src,
                                            const int* __restrict__ idx,
                                            float* __restrict__ dst, int ncols, int nrows){
    int i = blockIdx.x; if(i>=nrows) return;
    const float* s = src + (size_t)i*ncols;
    float* d = dst + (size_t)idx[i]*ncols;
    for(int c=threadIdx.x;c<ncols;c+=blockDim.x) d[c]=s[c];
}

// ROUND-STREAM stage (c) draft-chain pack: (tok, p) into slot j of a u32[2K] buffer — the
// host (or the assemble kernel) reads the whole chain in one go instead of 2 DtoHs per token.
extern "C" __global__ void pack_tok_p(const unsigned int* __restrict__ tok,
                                      const float* __restrict__ p,
                                      unsigned int* __restrict__ out, int slot) {
    if (threadIdx.x == 0) { out[2 * slot] = tok[0]; out[2 * slot + 1] = __float_as_uint(p[0]); }
}

// In-graph trimmed-head token remap: tok[0] = map[tok[0]] — replaces the host read-map-patch
// round trip inside the K-chain draft graph. Exact integer identity with the host map.
extern "C" __global__ void tok_map_u32(unsigned int* __restrict__ tok,
                                       const unsigned int* __restrict__ map) {
    if (threadIdx.x == 0) tok[0] = map[tok[0]];
}

// DSpark semi-AR markov head (dflash lane, 2026-07-13): gather ONE bf16 row of
// markov_w1 [V, rank] by a DEVICE token id into f32 (the rank-256 step vector). The
// sequential draft chain stays on-device (no per-position dtoh).
extern "C" __global__ void gather_row_bf16_f32(const unsigned short* __restrict__ table,
                                               const unsigned int* __restrict__ tok, int idx,
                                               float* __restrict__ dst, int ncols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ncols) {
        unsigned short h = table[(size_t)tok[idx] * ncols + i];
        dst[i] = __uint_as_float(((unsigned int)h) << 16);
    }
}

// bias add on ONE logits row: logits[row0*V .. +V] += bias[0..V]
extern "C" __global__ void add_row_inplace_f32(float* __restrict__ logits,
                                               const float* __restrict__ bias,
                                               int n, long row_off) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[row_off + i] += bias[i];
}

// LATENCY-HIDING ARC (owner angles, 2026-07-10): L2 prefetch of a byte range — issued 1-2
// kernels ahead of the consumer (fa's KV stream), so the latency-bound consumer finds its
// lines L2-warm. Pure scheduling: touches no values, changes no numeric config.
extern "C" __global__ void prefetch_l2_bytes(const unsigned char* __restrict__ p, long n) {
    long i = (long)(blockIdx.x * blockDim.x + threadIdx.x) * 128;
    if (i < n) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(p + i));
    }
}

// ================= vision tower primitives (lane/vision, 2026-08-15) =================
// The qwen3_5_vision ViT needs three generic ops the trunk (RMS-only, causal) never had.
// All f32, correctness-first — the tower is ~2% of a vision request's FLOPs.

// Row LayerNorm WITH bias: y = (x - mean) / sqrt(var + eps) * w + b. One block per row.
extern "C" __global__ void layer_norm_bias_f32(
        const float* __restrict__ x, const float* __restrict__ w,
        const float* __restrict__ b, float* __restrict__ y, int n, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t) row * n;
    float* yr = y + (size_t) row * n;
    __shared__ float sred[256];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    sred[threadIdx.x] = s; __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < blockDim.x; ++i) t += sred[i]; sred[0] = t / n; }
    __syncthreads();
    const float mean = sred[0]; __syncthreads();
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float d = xr[i] - mean; v += d * d; }
    sred[threadIdx.x] = v; __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < blockDim.x; ++i) t += sred[i]; sred[0] = rsqrtf(t / n + eps); }
    __syncthreads();
    const float inv = sred[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))), elementwise.
extern "C" __global__ void gelu_tanh_f32(
        const float* __restrict__ x, float* __restrict__ y, long n) {
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(c));
}

// Row softmax in place over [rows, n] (bidirectional attention scores). One block per row.
extern "C" __global__ void row_softmax_f32(float* __restrict__ x, int n) {
    const int row = blockIdx.x;
    float* xr = x + (size_t) row * n;
    __shared__ float sred[256];
    float m = -3.4e38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, xr[i]);
    sred[threadIdx.x] = m; __syncthreads();
    if (threadIdx.x == 0) { float t = sred[0]; for (int i = 1; i < blockDim.x; ++i) t = fmaxf(t, sred[i]); sred[0] = t; }
    __syncthreads();
    const float mx = sred[0]; __syncthreads();
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float e = __expf(xr[i] - mx); xr[i] = e; s += e; }
    sred[threadIdx.x] = s; __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < blockDim.x; ++i) t += sred[i]; sred[0] = t > 0 ? 1.0f / t : 0.0f; }
    __syncthreads();
    const float inv = sred[0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= inv;
}

// ---- Batched uniform-size D2D copy (engine-bundle slice 1, DSF-ROUNDCOST-20260820 §1.1) ----
// The dspark round's GDN state snapshot/commit was a dribble of ~48-96 tiny memcpyDtoD
// dispatches per round (snap 0.67 ms native + commit 0.25 ms — pure dispatch serialization,
// zero kernels). table = [src_0..src_{n-1}, dst_0..dst_{n-1}] raw device pointers; every
// region is `words` f32. ONE launch replaces n memcpy dispatches; bytes and stream order are
// identical to the memcpy sequence it replaces (regions are disjoint whole allocations).
// Grid (chunks, n): block (bx, r) strides region r. float4 body when both pointers are
// 16B-aligned and words%4==0 (all state buffers are whole 256B-aligned allocations with
// power-of-two row sizes); scalar fallback otherwise so the kernel is safe for ANY region.
extern "C" __global__ void copy_batch_uniform_f32(
        const unsigned long long* __restrict__ table, int n, int words) {
    const int r = blockIdx.y;
    if (r >= n) return;
    const float* __restrict__ src = (const float*)(size_t)table[r];
    float* __restrict__ dst = (float*)(size_t)table[n + r];
    const int stride = gridDim.x * blockDim.x;
    const bool aligned = (((size_t)src | (size_t)dst) & 15) == 0 && (words & 3) == 0;
    if (aligned) {
        const float4* __restrict__ s4 = (const float4*)src;
        float4* __restrict__ d4 = (float4*)dst;
        const int vec = words >> 2;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec; i += stride) d4[i] = s4[i];
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
            dst[i] = src[i];
    }
}

// Speculative TP-cache verified-prefix repair. One block owns one layer and copies its accepted
// quantized K/V byte ranges, then publishes that layer's device length after the bytes are
// visible. table = [k_src x n, v_src x n, k_dst x n, v_dst x n, len_dst x n]. The source may
// be peer memory; the destination and len pointer belong to the launching rank. Exact HY3 TP2
// geometry is k_bytes=544 and v_bytes=384 per accepted row and rank, but the scalar fallback
// keeps the primitive layout-generic.
extern "C" __global__ void copy_batch_uniform_kv_u8_set_len(
        const unsigned long long* __restrict__ table,
        int n,
        int rows,
        int k_row_bytes,
        int v_row_bytes,
        int k_src_stride,
        int v_src_stride,
        int logical_len) {
    const int r = blockIdx.x;
    if (r >= n) return;
    const unsigned char* __restrict__ k_src = (const unsigned char*)(size_t)table[r];
    const unsigned char* __restrict__ v_src = (const unsigned char*)(size_t)table[n + r];
    unsigned char* __restrict__ k_dst = (unsigned char*)(size_t)table[2 * n + r];
    unsigned char* __restrict__ v_dst = (unsigned char*)(size_t)table[3 * n + r];
    int* __restrict__ len_dst = (int*)(size_t)table[4 * n + r];

    for (int row = 0; row < rows; ++row) {
        const unsigned char* __restrict__ ks = k_src + row * k_src_stride;
        unsigned char* __restrict__ kd = k_dst + row * k_row_bytes;
        const bool k_aligned = (((size_t)ks | (size_t)kd) & 15) == 0 && (k_row_bytes & 15) == 0;
        if (k_aligned) {
            const uint4* __restrict__ src = (const uint4*)ks;
            uint4* __restrict__ dst = (uint4*)kd;
            for (int i = threadIdx.x; i < (k_row_bytes >> 4); i += blockDim.x) dst[i] = src[i];
        } else {
            for (int i = threadIdx.x; i < k_row_bytes; i += blockDim.x) kd[i] = ks[i];
        }

        const unsigned char* __restrict__ vs = v_src + row * v_src_stride;
        unsigned char* __restrict__ vd = v_dst + row * v_row_bytes;
        const bool v_aligned = (((size_t)vs | (size_t)vd) & 15) == 0 && (v_row_bytes & 15) == 0;
        if (v_aligned) {
            const uint4* __restrict__ src = (const uint4*)vs;
            uint4* __restrict__ dst = (uint4*)vd;
            for (int i = threadIdx.x; i < (v_row_bytes >> 4); i += blockDim.x) dst[i] = src[i];
        } else {
            for (int i = threadIdx.x; i < v_row_bytes; i += blockDim.x) vd[i] = vs[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) *len_dst = logical_len;
}

// ---- Indirect-source copy (engine-bundle slice 3) ----
// src address is read from a device pointer-table entry at RUN time — the gdn ping-pong
// redirects through the same 6-entry table the scan kernels consume, so a CAPTURED graph
// stays parity-correct after the canonical/alt handles swap between rounds (a baked
// memcpy node would keep reading the capture-time physical buffer — the slice-3 smoke
// divergence). dst is a direct (static slab) address. Same body as copy_batch_uniform.
extern "C" __global__ void copy_indirect_src_f32(
        const unsigned long long* __restrict__ src_entry, float* __restrict__ dst, int words) {
    const float* __restrict__ src = (const float*)(size_t)src_entry[0];
    const int stride = gridDim.x * blockDim.x;
    const bool aligned = (((size_t)src | (size_t)dst) & 15) == 0 && (words & 3) == 0;
    if (aligned) {
        const float4* __restrict__ s4 = (const float4*)src;
        float4* __restrict__ d4 = (float4*)dst;
        const int vec = words >> 2;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec; i += stride) d4[i] = s4[i];
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
            dst[i] = src[i];
    }
}

// =====================================================================================
// qwen4_exp eager-arm kernels (qwen4exp-bringup-20260829, GPU-EAGER phase 7).
// Correctness-class oracles for the gated-residual/QSA/GDN/PLE program — geometry-generic,
// all f32, gated against memra-reference. Tuned twins are later perf-lane work.
// =====================================================================================

// Masked-visibility twin of sdpa_naive_f32 (QSA micro-block overlay, SEMANTICS.md §QSA
// indexer): identical loop structure, GQA mapping, and single-thread softmax; visibility =
// causal AND mask[qt, t]. `mask` is [T, T_kv] row-major u8 (0/1), one row per QUERY token,
// shared across heads (the indexer selects per token, not per head). A masked score takes
// the same -1e30 sentinel the causal arm uses, so expf underflows to exact 0.0f — identical
// to the reference's source-exclusion. The QSA tail rule guarantees >= 1 visible source per
// query; the HOST mask builder asserts that before launch (an all-masked row would 0/0).
extern "C" __global__ void sdpa_naive_mask_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                               const float* __restrict__ V, float* __restrict__ O,
                                               const unsigned char* __restrict__ mask,
                                               int head_dim, int n_head, int n_head_kv, int T, int T_kv,
                                               float scale) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float scores[];    // [T_kv]

    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;
    int q_pos = (T_kv - T) + qt;
    const unsigned char* mrow = mask + (size_t)qt * T_kv;

    for (int t = tid; t < T_kv; t += blockDim.x) {
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        acc *= scale;
        if (t > q_pos || !mrow[t]) acc = -1e30f;
        scores[t] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int t = 0; t < T_kv; t++) mx = fmaxf(mx, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < T_kv; t++) { float e = expf(scores[t] - mx); scores[t] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int t = 0; t < T_kv; t++) scores[t] *= inv;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < T_kv; t++) {
            const float* v = V + ((size_t)t * n_head_kv + kv_head) * head_dim;
            acc += scores[t] * v[d];
        }
        o[d] = acc;
    }
}

// ---- QSA indexer block scoring (qwen4_exp long-context lane): one THREAD per (query
// row, complete micro-block) computes score = sum_h relu(q_h . pooled_k) / sqrt(head_dim)
// in fp32. Thread-per-block (not a warp reduction) is deliberate: the dim loop runs in the
// SAME sequential order as the host twin, so every score is BIT-IDENTICAL to
// `indexer_select_rows`'s host arithmetic and the top-k selects the same set. The pooled
// keys are the device mirror of the host pooled cache (post-mean/norm/rope, one row per
// complete block); queries are the per-row post-norm/post-rope index heads.
// q: [rows, heads*head_dim]; pooled: [n_blocks, head_dim]; out: [rows, n_blocks]. ----
extern "C" __global__ void qsa_index_score_f32(const float* __restrict__ q,
                                              const float* __restrict__ pooled,
                                              float* __restrict__ out,
                                              int heads, int head_dim, int n_blocks,
                                              int rows, float scale) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (block >= n_blocks || row >= rows) return;
    const float* k = pooled + (size_t)block * head_dim;
    const float* qr = q + (size_t)row * heads * head_dim;
    // EXPLICIT IEEE ops, no FMA contraction: the host twin does separate f32 multiply and
    // add, and nvcc's default `-fmad=true` fused the dot loop into fmaf, which differed
    // by 1 ULP on ~1 score in 10^3 (caught by the arm-0g bit gate — a differing score can
    // flip a near-tie block out of the top-k, i.e. change the attended set). Keep these
    // intrinsics if this kernel is ever retuned.
    float score = 0.0f;
    for (int h = 0; h < heads; h++) {
        const float* qh = qr + h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot = __fadd_rn(dot, __fmul_rn(qh[d], k[d]));
        // host twin: `score += dot.max(0.0)` then `score / scale` (DIVISION, matched).
        score = __fadd_rn(score, fmaxf(dot, 0.0f));
    }
    out[(size_t)row * n_blocks + block] = __fdiv_rn(score, scale);
}

// ---- `qsa_index_score_f32` over a DIM-MAJOR pooled plane (the `poolT` layout seam). ----
// Same program, same loop order, same explicit `__fmul_rn`/`__fadd_rn`/`__fdiv_rn` — the ONLY
// difference is the address of `pooled`, so this kernel is BIT-IDENTICAL to the row-major twin
// on a correctly transposed plane. It exists because the row-major read is the worst possible
// pattern for the one axis that grows with context:
//
//   row-major:  lane L reads `pooled[(block0+L)*head_dim + d]` -> lanes are head_dim*4 = 512 B
//               apart, so a warp's single `k[d]` load touches 32 DISTINCT 32-byte sectors and
//               moves 1024 B to use 128 B. 8x sector amplification, every element, every head.
//   dim-major:  lane L reads `pooled_t[d*pitch + (block0+L)]` -> 32 consecutive floats, 128 B,
//               4 sectors, zero waste.
//
// `pitch` is the plane's block capacity (NOT n_blocks): the mirror grows geometrically and a
// dim-major plane cannot be appended to without a fixed row pitch, so the pitch is baked at
// allocation and the transpose is redone only when the capacity changes. Passing n_blocks as
// the pitch would silently read the wrong dim for every d>0 the moment capacity > n_blocks,
// which is the normal state — hence a separate parameter rather than a reuse.
extern "C" __global__ void qsa_index_score_f32_t(const float* __restrict__ q,
                                                const float* __restrict__ pooled_t,
                                                float* __restrict__ out,
                                                int heads, int head_dim, int n_blocks,
                                                int rows, float scale, long pitch) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (block >= n_blocks || row >= rows) return;
    const float* qr = q + (size_t)row * heads * head_dim;
    float score = 0.0f;
    for (int h = 0; h < heads; h++) {
        const float* qh = qr + h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot = __fadd_rn(dot, __fmul_rn(qh[d], pooled_t[(size_t)d * pitch + block]));
        score = __fadd_rn(score, fmaxf(dot, 0.0f));
    }
    out[(size_t)row * n_blocks + block] = __fdiv_rn(score, scale);
}

// ---- Mirror `rows` freshly-appended pooled rows into the dim-major plane. ----
// ONE buffer holds both layouts: `[0, cap_rows*head_dim)` is the row-major mirror the host H2Ds
// into, and `[cap_rows*head_dim, 2*cap_rows*head_dim)` is the dim-major plane. Both regions are
// passed as one pointer so the kernel cannot be handed two aliasing `__restrict__` arguments
// (the regions are disjoint, but promising a compiler otherwise is a trap, not an optimisation).
// Device-to-device: the rows are already on the card from the row-major H2D, so re-sending them
// from the host would double the transfer for nothing.
//
// Pure data movement, no arithmetic — it cannot move a bit of any score. Reads are `head_dim*4`
// strided and writes are coalesced; deliberately left un-tiled because the delta is 512 rows per
// 2,048-token prefill chunk and 0-1 rows per decode step (65,536 elements at the target geometry,
// against the 33.5 MB the score kernel reads per row). A smem-tiled transpose here would be
// optimising the wrong side of a 500x ratio.
extern "C" __global__ void qsa_pooled_transpose_f32(float* buf,
                                                   int rows, int head_dim, int r0, long cap_rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;   // row within the delta
    int d = blockIdx.y;                              // dim
    if (r >= rows || d >= head_dim) return;
    const size_t block = (size_t)r0 + (size_t)r;
    buf[(size_t)cap_rows * head_dim + (size_t)d * cap_rows + block] =
        buf[block * (size_t)head_dim + (size_t)d];
}

// ---- f32 -> ASCENDING u32 order key: `f32::total_cmp` VERBATIM, then shifted into the
// unsigned domain. Rust's total_cmp is `let mut l = bits as i32; l ^= (((l >> 31) as u32)
// >> 1) as i32;` compared as i32 — so the xor mask is 0 for a clear sign bit and
// 0x7fffffff for a set one, and one more xor by 0x80000000 turns the i32 total order into
// an ascending u32. This is a strictly monotone map over the WHOLE f32 domain (both
// zeros, both infinities, every NaN payload), which is why the selection below carries NO
// non-negativity claim. Contrast `qwen4exp_route_topk_f32`, whose `~bits(w)` key is only
// the host comparator BECAUSE its weights are >= +0.0 — a domain-scoped shortcut that
// would silently mis-order here if a score were ever negative or NaN.
static __device__ __forceinline__ unsigned f32_total_asc_u32(float v) {
    unsigned u = __float_as_uint(v);
    unsigned m = ((unsigned)((int)u >> 31)) >> 1; // 0 or 0x7fffffff
    return (u ^ m) ^ 0x80000000u;
}

// ---- QSA indexer top-k SELECTION, device twin (262k perf lane): per query row, the
// pinned top-`budget` micro-block selection over the score slab, emitted ASCENDING by
// block index. This is `top_blocks_ascending` on device, and it exists because the host
// half of the indexer (`qsa.idx_host`) measured **83% of a deep prefill chunk** at a
// 131,072 fill — 51,235 ms of 61,700 — while EVERY GPU section stayed flat
// (research/qwen4exp-bringup-20260829/round2-box-receipts/LADDER.md §4c). Moving it here
// also deletes the score slab's blocking dtoh (up to 128 MB per sub-batch) in favour of a
// rows x budget u32 readback (4 MB at the target geometry).
//
// The order: key = (~f32_total_asc_u32(score) << 32) | block_index, and ASCENDING u64 key
// order IS the host `sel_cmp` (score desc under total_cmp, then block index asc). Keys are
// DISTINCT because the low 32 bits are the unique block index, so the k-th smallest key is
// a single well-defined element and "key <= T" selects exactly k blocks. Ties are the
// point, not an edge case: the zero-score tie class is structural on this family (relu-sum
// scores floor at +0.0 and a deep row has many), and a tie-blind selection would pick a
// DIFFERENT set with identical arithmetic.
//
// Algorithm: 8-pass radix select (one byte per pass, 256 smem bins) narrows the k-th
// smallest key exactly, then one ordered compaction pass emits every block whose key is
// <= it. Deliberately NOT the route kernel's "k rounds of block-wide min": that is O(k*n)
// and the geometries differ by three orders — k=10 over 512 experts there, k=512 over up
// to 65,536 blocks here (33.5 M comparisons per row x 2,048 rows x 12 layers per chunk
// would be SLOWER than the host it replaces). What transfers from that lane is the KEY
// ENCODING, not the loop.
//
// scores: [rows, stride] (row r reads its own prefix of `complete[r]`); complete: [rows]
// i32; out: [rows, budget] i32 block ids, ascending. Launcher pins blockDim 256 and
// guards complete[r] > budget for every row (so k == budget and every out slot is
// written). ----
extern "C" __global__ void qsa_index_topk_u32(const float* __restrict__ scores,
                                             const int* __restrict__ complete,
                                             int* __restrict__ out,
                                             int stride, int budget, int rows) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* s = scores + (size_t)row * stride;
    int n = complete[row];
    int k = budget < n ? budget : n;
    if (k <= 0) return;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    __shared__ unsigned hist[256];
    __shared__ unsigned long long prefix_sh;
    __shared__ int need_sh;
    __shared__ unsigned wcnt[32];
    __shared__ unsigned tile_total;
    __shared__ unsigned base_sh;
    if (tid == 0) {
        prefix_sh = 0ULL;
        need_sh = k;
    }
    __syncthreads();
    // ---- radix select: fix the k-th smallest key one byte at a time, most significant
    // first. Invariant: `prefix_sh` holds the bits already fixed and `need_sh` is the
    // 1-based rank still sought AMONG the keys matching that prefix.
    for (int pass = 0; pass < 8; pass++) {
        int shift = 56 - 8 * pass;
        unsigned long long pref = prefix_sh;
        // Bits above this pass's byte; 0 on pass 0 (nothing fixed yet).
        unsigned long long hi_mask = (pass == 0) ? 0ULL : (~0ULL << (shift + 8));
        for (int b = tid; b < 256; b += nt) hist[b] = 0u;
        __syncthreads();
        for (int i = tid; i < n; i += nt) {
            unsigned long long kk =
                ((unsigned long long)(~f32_total_asc_u32(s[i])) << 32) | (unsigned)i;
            if ((kk & hi_mask) == (pref & hi_mask))
                atomicAdd(&hist[(unsigned)((kk >> shift) & 0xffULL)], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int need = need_sh;
            unsigned run = 0u;
            int b = 0;
            for (; b < 255; b++) {
                unsigned c = hist[b];
                if (run + c >= (unsigned)need) break;
                run += c;
            }
            need_sh = need - (int)run;
            prefix_sh = pref | ((unsigned long long)(unsigned)b << shift);
        }
        __syncthreads();
    }
    unsigned long long thr = prefix_sh;
    // ---- ordered compaction: walk the row ASCENDING in blockDim tiles and append every
    // block whose key is <= the threshold. Sequential tiles + a per-tile warp-ballot rank
    // give ascending block order for free (no sort), which is the contract the host twin's
    // `candidates.sort_unstable()` provides.
    if (tid == 0) base_sh = 0u;
    __syncthreads();
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = (nt + 31) >> 5;
    for (int t0 = 0; t0 < n; t0 += nt) {
        int i = t0 + tid;
        int p = 0;
        if (i < n) {
            unsigned long long kk =
                ((unsigned long long)(~f32_total_asc_u32(s[i])) << 32) | (unsigned)i;
            p = (kk <= thr) ? 1 : 0;
        }
        unsigned mask = __ballot_sync(0xffffffffu, p);
        unsigned r = (unsigned)__popc(mask & ((1u << lane) - 1u));
        if (lane == 0) wcnt[warp] = (unsigned)__popc(mask);
        __syncthreads();
        if (tid == 0) {
            unsigned run = 0u;
            for (int w = 0; w < nwarps; w++) {
                unsigned c = wcnt[w];
                wcnt[w] = run;
                run += c;
            }
            tile_total = run;
        }
        __syncthreads();
        if (p) {
            unsigned pos = base_sh + wcnt[warp] + r;
            if (pos < (unsigned)budget) out[(size_t)row * budget + pos] = i;
        }
        __syncthreads();
        if (tid == 0) base_sh += tile_total;
        __syncthreads();
    }
}

// ---- Row-window column-slice copy (devtwin indexer cache): dst[dst_row + r][0..width)
// = src[r*src_stride + src_col ..], r in [0, rows). Pure element moves — exact bytes,
// no arithmetic; appends the k-part of the indexer projection rows to the DEVICE
// raw-key cache without a host round trip. ----
extern "C" __global__ void copy_rows_col_f32(const float* __restrict__ src,
                                             float* __restrict__ dst,
                                             int rows, int width, long src_stride,
                                             long src_col, long dst_row) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * width;
    if (i >= total) return;
    long r = i / width;
    long c = i - r * width;
    dst[(dst_row + r) * (long)width + c] = src[r * src_stride + src_col + c];
}

// ---- qwen4_exp MoE router, device twin (devtwin lane): per token row, the FULL
// host_route_softmax_topk program on device — softmax over `experts` logits, top-`selected`
// under the pinned tie rule (weight desc, index asc), renormalize with the denominator
// floor. The order-SENSITIVE f32 sums (softmax sum, denominator sum) run on thread 0
// SEQUENTIALLY in index/selection order over smem — the host twin's op order verbatim;
// everything order-FREE is parallel: the max fold (fmaxf is associative+commutative, any
// order is bit-exact), exp, the divisions, and the top-k (k rounds of block-wide strict
// min over distinct 64-bit keys — a total order enumerated smallest-first is
// order-of-evaluation-free). EXPLICIT IEEE intrinsics, no FMA contraction (the
// qsa_index_score lesson above).
//
// exp is the ONE op whose bits are not host-guaranteed: the host twin rounds through libm
// expf; this kernel computes exp in double (CUDA exp(double), <= 1 ulp double error) and
// rounds once to f32 — correctly-rounded f32 except when the double result sits within a
// double-ulp of an f32 rounding boundary (~2^-29 per call). The devtwin contract gates it:
// the selection set+order must equal the host twin EXACTLY and weights within documented
// ULP (gate_route_kernel + the MEMRA_Q4E_ROUTER_AUDIT live twin over real decode rows).
//
// Tie-rule note (why selection cannot run on raw logits): f32 exp maps ~2 adjacent logits
// near the max onto one weight, and the host tie rule breaks WEIGHT ties by index — a
// logits-ordered top-k would resolve those rows differently. The kernel therefore selects
// on the divided softmax weights, exactly like the host.
//
// Weights are >= +0.0 (exp then positive division), so their IEEE bit patterns order as
// unsigned ints == f32 total_cmp on this domain (-0.0/NaN unreachable from finite logits).
//
// logits: [rows, experts]; sel/w_out: [rows, selected]; tok_raw: optional [rows, selected]
// i32 slot->token map (tok[r*selected+j] = tok_base + r), 0 to skip — feeds the gufuse
// tok_map so the verify merged path needs no host-built map. Dynamic smem = experts f32.
extern "C" __global__ void qwen4exp_route_topk_f32(const float* __restrict__ logits,
                                                   int* __restrict__ sel,
                                                   float* __restrict__ w_out,
                                                   unsigned long long tok_raw,
                                                   int experts, int selected, int rows,
                                                   int tok_base, float denom_floor) {
    // Dynamic smem: w[experts] f32, then keys[experts] u64 (experts EVEN keeps the u64
    // slab 8-byte aligned — launcher guard).
    extern __shared__ float w[];
    unsigned long long* keys = (unsigned long long*)(w + experts);
    __shared__ float redf[32];
    __shared__ unsigned long long red64[32];
    __shared__ unsigned long long last_sh;
    __shared__ float bcast;
    __shared__ int top_i[32];
    __shared__ float top_w[32];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float* x = logits + (size_t)row * experts;
    // Max fold, PARALLEL tree: fmaxf is associative and commutative (incl. its
    // NaN-ignoring arm), so ANY reduction order produces the identical bits to the
    // host's sequential fold. (Perf postmortem, v1/v2: v1 ran every phase on thread 0
    // over GLOBAL memory (~54 us/launch on the rig bench); v2 moved data to smem but
    // kept the top-k as a serial insertion over thread-LOCAL arrays, which spill to
    // local memory under dynamic indexing — the phase bisect measured the top-k alone
    // at 48 of the 54 us. v3 parallelizes selection as k rounds of block-wide min over
    // a 64-bit key (below); order-SENSITIVE reductions stay sequential over smem.)
    float mx = -INFINITY;
    for (int i = tid; i < experts; i += blockDim.x) mx = fmaxf(mx, x[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    if ((tid & 31) == 0) redf[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? redf[tid] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
        if (tid == 0) bcast = v;
    }
    __syncthreads();
    mx = bcast;
    for (int i = tid; i < experts; i += blockDim.x)
        w[i] = (float)exp((double)__fsub_rn(x[i], mx));
    __syncthreads();
    if (tid == 0) {
        // Host softmax sum: sequential over index order on the ROUNDED exp values
        // (f32 addition is order-sensitive — this chain IS the host's, over smem).
        float sum = 0.0f;
        for (int i = 0; i < experts; i++) sum = __fadd_rn(sum, w[i]);
        bcast = sum;
    }
    __syncthreads();
    float sum = bcast;
    // Divide (order-free per element) and build the selection key: ascending u64 order
    // of key = (~bits(w) << 32) | idx IS the host comparator (weight desc via
    // total_cmp, index asc) on this domain — weights are >= +0.0 (exp then positive
    // division), where IEEE bit patterns order as unsigned ints. Keys are distinct
    // (idx unique), so k rounds of strict block-min enumerate the selection in the
    // host's emitted order, ties included.
    for (int i = tid; i < experts; i += blockDim.x) {
        float wi = __fdiv_rn(w[i], sum);
        w[i] = wi;
        keys[i] = ((unsigned long long)(~__float_as_uint(wi)) << 32) | (unsigned)i;
    }
    __syncthreads();
    int k = selected < experts ? selected : experts;
    unsigned long long last = 0;
    for (int j = 0; j < k; j++) {
        unsigned long long m = ~0ULL;
        for (int i = tid; i < experts; i += blockDim.x) {
            unsigned long long kb = keys[i];
            if (kb > last && kb < m) m = kb;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = min(m, __shfl_down_sync(0xffffffff, m, off));
        if ((tid & 31) == 0) red64[tid >> 5] = m;
        __syncthreads();
        if (tid < 32) {
            unsigned long long v = (tid < (blockDim.x + 31) / 32) ? red64[tid] : ~0ULL;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v = min(v, __shfl_down_sync(0xffffffff, v, off));
            if (tid == 0) {
                last_sh = v;
                top_i[j] = (int)(v & 0xffffffffu);
                top_w[j] = __uint_as_float(~(unsigned)(v >> 32));
            }
        }
        __syncthreads();
        last = last_sh;
        __syncthreads();
    }
    if (tid == 0) {
        // Host denominator: sum over the SELECTION order, then the floor via f32::max.
        float denom = 0.0f;
        for (int j = 0; j < k; j++) denom = __fadd_rn(denom, top_w[j]);
        denom = fmaxf(denom, denom_floor);
        int* tok = (int*)(size_t)tok_raw;
        for (int j = 0; j < k; j++) {
            sel[(size_t)row * selected + j] = top_i[j];
            w_out[(size_t)row * selected + j] = __fdiv_rn(top_w[j], denom);
            if (tok) tok[(size_t)row * selected + j] = tok_base + row;
        }
    }
}

// ---- Block-list QSA attention (qwen4_exp long-context lane): per query row, attend the
// row's OWN ascending position list instead of a dense [T, T_kv] mask — smem scales with
// the bounded selection (budget*block + tail <= 2052 on real geometry), never with T_kv.
// BIT-IDENTICAL to sdpa_naive_mask_f32 on the same selection: there, a masked entry's
// score is -1e30, whose exp underflows to exactly 0.0f, and adding 0.0f terms (softmax
// sum) or 0.0f*v (weighted V) in the same ascending-position order changes no float —
// dropping them here reproduces every accumulation bit. Phases mirror the masked kernel:
// per-position dots (parallel over selected entries), single-thread max/exp/normalize in
// selection order, per-dim weighted V over selected entries ascending. ----
extern "C" __global__ void sdpa_blocklist_f32(const float* __restrict__ Q, const float* __restrict__ K,
                                              const float* __restrict__ V, float* __restrict__ O,
                                              const int* __restrict__ pos_list,
                                              const int* __restrict__ row_meta,
                                              int head_dim, int n_head, int n_head_kv, int T,
                                              int max_count, float scale) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float smem_raw[];
    int* spos = (int*)smem_raw;            // [max_count]
    float* scores = smem_raw + max_count;  // [max_count]

    int off = row_meta[2 * qt];
    int count = row_meta[2 * qt + 1];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;

    for (int i = tid; i < count; i += blockDim.x) {
        int t = pos_list[off + i];
        spos[i] = t;
        const float* k = K + ((size_t)t * n_head_kv + kv_head) * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * k[d];
        scores[i] = acc * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int i = 0; i < count; i++) mx = fmaxf(mx, scores[i]);
        float sum = 0.0f;
        for (int i = 0; i < count; i++) { float e = expf(scores[i] - mx); scores[i] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int i = 0; i < count; i++) scores[i] *= inv;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < count; i++) {
            const float* v = V + ((size_t)spos[i] * n_head_kv + kv_head) * head_dim;
            acc += scores[i] * v[d];
        }
        o[d] = acc;
    }
}

// Geometry-generic sequential Gated DeltaNet scan — the memra-reference `gated_delta_net`
// twin (gdn_scan_s128 is the tuned d_state=128 form; this one takes any hk/hv, including
// the qwen4_exp tiny fixture's 4/4). Consumes the POST-conv fused qkv TOKEN-major
// [T, conv_dim] (conv_dim = 2*nk*hk + nv*hv, silu already applied by dwconv_causal_f32),
// l2-normalizes q/k per (token, key-head) in-kernel (1/sqrt(sum_sq + eps), the reference
// l2_normalize_rows form), applies beta = sigmoid(beta_raw) and decay = expf(g_log)
// (g_log from gdn_glog_f32: a * softplus(alpha + dt_bias)). The z-gate activation is NOT
// here — the caller composes rms_norm + sigmoid/silu ⊙ (qwen4_exp = sigmoid, SEMANTICS.md
// §GDN). State layout [nv, hv, hk] == ReferenceLayerState::Recurrent::matrix, updated in
// place. Grid (nv), block (hv) — launch EXACT dims (in-loop __syncthreads). hk <= 128.
// Sequential over T per the reference recurrence; eager-arm oracle, not a serving kernel.
extern "C" __global__ void gdn_scan_naive_f32(
        const float* __restrict__ qkv, const float* __restrict__ g_log,
        const float* __restrict__ beta_raw, float* __restrict__ state, float* __restrict__ o,
        int nk, int nv, int hk, int hv, int T, float scale, float eps) {
    int h = blockIdx.x;
    int col = threadIdx.x;
    int kh = h % nk;
    int conv_dim = 2 * nk * hk + nv * hv;
    extern __shared__ float sh[];                 // [2*hk + 2]: qrow, krow, inv-norms
    float* qrow = sh;
    float* krow = sh + hk;
    float* norms = sh + 2 * hk;
    float* srow = state + ((size_t)h * hv + col) * hk;
    float s[128];
    for (int i = 0; i < hk; i++) s[i] = srow[i];
    for (int t = 0; t < T; t++) {
        const float* row = qkv + (size_t)t * conv_dim;
        if (col == 0) {
            float qs = 0.0f, ks = 0.0f;
            for (int i = 0; i < hk; i++) {
                float qv = row[kh * hk + i];
                float kv = row[nk * hk + kh * hk + i];
                qrow[i] = qv; krow[i] = kv;
                qs += qv * qv; ks += kv * kv;
            }
            norms[0] = 1.0f / sqrtf(qs + eps);
            norms[1] = 1.0f / sqrtf(ks + eps);
        }
        __syncthreads();
        float g = expf(g_log[(size_t)t * nv + h]);
        float b = 1.0f / (1.0f + expf(-beta_raw[(size_t)t * nv + h]));
        float vcol = row[2 * nk * hk + h * hv + col];
        float qn = norms[0], kn = norms[1];
        float kv_dot = 0.0f;
        for (int i = 0; i < hk; i++) kv_dot += s[i] * (krow[i] * kn);
        float delta = (vcol - g * kv_dot) * b;
        float attn = 0.0f;
        for (int i = 0; i < hk; i++) {
            float sn = g * s[i] + (krow[i] * kn) * delta;
            s[i] = sn;
            attn += sn * (qrow[i] * qn);
        }
        o[(size_t)t * (nv * hv) + h * hv + col] = attn * scale;
        __syncthreads();                          // qrow/krow rewritten next t
    }
    for (int i = 0; i < hk; i++) srow[i] = s[i];
}

// Decode-step (T==1) twin of gdn_scan_naive_f32 (qwen4_exp perf round 3): the naive
// kernel at t=1 launches `nv` blocks (48 on a 188-SM card) with s[128] per THREAD in
// registers — latency-bound, most of the card idle. This twin flips the parallelism:
// grid (nv, hv) = 6144 blocks at the artifact geometry, block hk threads, each thread
// holding ONE state element. Same math per element (l2 norms with the same 1/sqrt(sum+eps)
// form, decay expf(g_log), beta sigmoid, delta rule, out attn*scale); the three row sums
// (q/k square sums, kv_dot, attn) run as block reduction TREES instead of the naive
// kernel's sequential loops — the accumulation class, gated by its own real-geometry
// oracle (`gate_gdn_scan_step`) + the real-checkpoint gates. The q/k norm reduce is
// recomputed redundantly per value-column block (hv× per head) — 2*hk loads, cheap.
// Geometry: hk % 32 == 0 (full-warp shuffles), hk <= 1024; the tiny plan (hk 4) keeps
// the naive kernel. State layout [nv, hv, hk] updated in place, exactly as naive.
extern "C" __global__ void gdn_scan_step_f32(
        const float* __restrict__ qkv, const float* __restrict__ g_log,
        const float* __restrict__ beta_raw, float* __restrict__ state, float* __restrict__ o,
        int nk, int nv, int hk, int hv, float scale, float eps) {
    int h = blockIdx.x;
    int col = blockIdx.y;
    int kh = h % nk;
    int tid = threadIdx.x;
    float qv = qkv[kh * hk + tid];
    float kv = qkv[nk * hk + kh * hk + tid];
    // Dual block reduce: qs and ks together (one shared round-trip).
    __shared__ float redq[32];
    __shared__ float redk[32];
    float aq = qv * qv, ak = kv * kv;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        aq += __shfl_down_sync(0xffffffff, aq, off);
        ak += __shfl_down_sync(0xffffffff, ak, off);
    }
    if ((tid & 31) == 0) { redq[tid >> 5] = aq; redk[tid >> 5] = ak; }
    __syncthreads();
    if (tid < 32) {
        float vq = (tid < (blockDim.x + 31) / 32) ? redq[tid] : 0.0f;
        float vk = (tid < (blockDim.x + 31) / 32) ? redk[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vq += __shfl_down_sync(0xffffffff, vq, off);
            vk += __shfl_down_sync(0xffffffff, vk, off);
        }
        if (tid == 0) { redq[0] = 1.0f / sqrtf(vq + eps); redk[0] = 1.0f / sqrtf(vk + eps); }
    }
    __syncthreads();
    float qn = qv * redq[0];
    float kn = kv * redk[0];
    float g = expf(g_log[h]);
    float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    float vcol = qkv[2 * nk * hk + h * hv + col];
    float* srow = state + ((size_t)h * hv + col) * hk;
    float s = srow[tid];
    // Every thread has read redq[0]/redk[0] above; fence before they are reused as
    // reduction scratch (a delayed warp must not read a partial store).
    __syncthreads();
    // kv_dot = <state_row, kn>
    float acc = s * kn;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) redq[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? redq[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) redq[0] = v;
    }
    __syncthreads();
    float delta = (vcol - g * redq[0]) * b;
    float sn = g * s + kn * delta;
    srow[tid] = sn;
    // attn = <state_row_new, qn>
    float att = sn * qn;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) att += __shfl_down_sync(0xffffffff, att, off);
    if ((tid & 31) == 0) redk[tid >> 5] = att;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? redk[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) o[(size_t)h * hv + col] = v * scale;
    }
}

// Depthwise causal conv, TOKEN-major, arbitrary dilation, explicit history rows — the
// qwen4_exp eager conv (GDN conv dilation=1; PLE conv dilation=max_ngram with left-reach
// (K-1)*dilation, SEMANTICS.md §PLE). x [T, C] current rows; hist [Th, C] cached left
// context (Th = 0 on a fresh prefill; decode passes the cached tail rows); w [C, K].
// source = t - (K-1-tap)*dilation reads x, then hist (at Th + source), then implicit zero
// (the reference's `source >= 0` skip). mode 0: y = conv; 1: y = silu(conv);
// 2: y += silu(conv) (the PLE residual add).
extern "C" __global__ void dwconv_causal_f32(const float* __restrict__ x, const float* __restrict__ hist,
                                             const float* __restrict__ w, float* __restrict__ y,
                                             int T, int Th, int C, int K, int dilation, int mode) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)T * C) return;
    int t = (int)(idx / C);
    int c = (int)(idx % C);
    float acc = 0.0f;
    for (int tap = 0; tap < K; tap++) {
        int src = t - (K - 1 - tap) * dilation;
        float xv;
        if (src >= 0) {
            xv = x[(size_t)src * C + c];
        } else if (src + Th >= 0) {
            xv = hist[(size_t)(src + Th) * C + c];
        } else {
            continue;
        }
        acc += xv * w[(size_t)c * K + tap];
    }
    if (mode == 0) {
        y[idx] = acc;
    } else {
        float s = acc / (1.0f + expf(-acc));
        if (mode == 1) y[idx] = s; else y[idx] += s;
    }
}

// Selected-experts NVFP4 matvec over the AS-STORED modelopt layout (qwen4_exp perf lane,
// research/qwen4exp-bringup-20260829/perf/): codes [E, out_f, in_f/2] u8 — element g of a
// row sits at byte g/2, low nibble g even / high nibble g odd — and scales
// [E, out_f, in_f/16] UE4M3 (modelopt convention: NaN code mag 0x7F -> 0.0, sign bit
// honored, denormal man/8 * 2^-6), per-expert f32 macro (weight_scale_2) folded in the
// epilogue. W4A16: activations stay f32, products (e2m1_code * ue4m3_scale) are computed
// exactly in f32 (<= 6 significand bits) — the same per-element values as the eager
// dequant->cuBLASLt chain; only ASSOCIATION differs (macro applied after the row sum,
// scale factored per 16-group, warp-strided reduce tree) — the accumulation class, gated
// by the tiny four-arm + real-checkpoint gates. One launch covers every selected expert
// of one projection: grid (out_f, n_sel), block 32 (one warp per output row);
// x_stride = 0 shares one activation row (gate/up), = in_f reads per-slot rows (down).
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_f32(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int o = blockIdx.x;
    int slot = blockIdx.y;
    if (o >= out_f) return;
    int e = sel[slot];
    const unsigned char* crow = codes + ((size_t)e * out_f + o) * (size_t)(in_f / 2);
    const unsigned char* srow = scales + ((size_t)e * out_f + o) * (size_t)(in_f / 16);
    const float* xrow = x + (size_t)slot * (size_t)x_stride;
    int groups = in_f / 16;
    float acc = 0.0f;
    for (int g = threadIdx.x; g < groups; g += 32) {
        unsigned char sb = srow[g];
        int mag = sb & 0x7F;
        float scale;
        if (mag == 0x7F) {
            scale = 0.0f; // modelopt NaN code -> 0.0 (nvfp4_repack::fp8_e4m3_to_f32)
        } else {
            int se = (mag >> 3) & 0xF;
            float man = (float)(mag & 0x7);
            scale = (se == 0) ? (man / 8.0f) * exp2f(-6.0f)
                              : (1.0f + man / 8.0f) * exp2f((float)(se - 7));
            if (sb & 0x80) scale = -scale;
        }
        const unsigned char* cb = crow + (size_t)g * 8;
        float dot = 0.0f;
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte = cb[b];
            int base = g * 16 + 2 * b;
            dot += e2m1[byte & 0x0F] * xrow[base];
            dot += e2m1[byte >> 4] * xrow[base + 1];
        }
        acc += scale * dot;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (threadIdx.x == 0) y[(size_t)slot * out_f + o] = acc * macros[e];
}

// ---- qwen4_exp gated-residual (hyper-connection) fused gates, perf lane ----------------
// PROFILE-0 indicted the read gate at 27.7% of the attributed decode token across 96 calls,
// spent almost entirely in launch boundaries: ~71 launches per gate for 4 norms, 8 GEMVs,
// 16 inject GEMVs and ~25 elementwise/reduce steps. These three kernels take the
// elementwise/reduce work to three launches; the 12 GEMVs stay cuBLASLt (real matmul).
// Buffers are STREAM-MAJOR [streams, t, width] (the module's plane layout).

// low_act[t, rank] = silu(inv_streams * sum_s parts[s, t, rank]).
// Stream sum runs s = 0..streams-1 in order and the scale applies after — the same
// association as the axpy chain + scale_inplace it replaces, so BIT-IDENTICAL; silu matches
// silu_mul_f32's `g / (1 + expf(-g))` with up == 1.0.
extern "C" __global__ void hc_lowrank_reduce_f32(const float* __restrict__ parts,
                                                float* __restrict__ low_act,
                                                int streams, int t, int rank,
                                                float inv_streams) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)t * rank;
    if (idx >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < streams; s++) acc += parts[(long)s * n + idx];
    float g = acc * inv_streams;
    low_act[idx] = g / (1.0f + expf(-g));
}

// mixed[t, hidden] = inv_streams * sum_s sigmoid(gates[s, t, hidden]) * normed[s, t, hidden].
// Same s order and same post-sum scale as the sigmoid/mul/axpy/scale chain it replaces, and
// sigmoid matches sigmoid_f32 — BIT-IDENTICAL per element.
extern "C" __global__ void hc_mix_epilogue_f32(const float* __restrict__ gates,
                                               const float* __restrict__ normed,
                                               float* __restrict__ mixed,
                                               int streams, int t, int hidden,
                                               float inv_streams) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)t * hidden;
    if (idx >= n) return;
    float acc = 0.0f;
    for (int s = 0; s < streams; s++) {
        long o = (long)s * n + idx;
        acc += (1.0f / (1.0f + expf(-gates[o]))) * normed[o];
    }
    mixed[idx] = acc * inv_streams;
}

// inject[s, t] = 2 * sigmoid(inv_streams * <w_row_s, wide_normed_token_t>), where w is
// [streams, streams*hidden] row-major (block_inject_weight) and the wide token vector is the
// stream-major normed planes read at column s2*hidden + d. One block per (s, t); warp-strided
// reduction over the whole wide row. ACCUMULATION CLASS, not bit-identical: the chain it
// replaces ran one cuBLAS GEMV per (s, s2) and summed the four results, so the reduction tree
// differs. Output feeds a sigmoid and then scales a residual write — gated by the tiny
// four-arm gate's per-row tolerance + argmax policy.
extern "C" __global__ void hc_inject_gates_f32(const float* __restrict__ normed,
                                               const float* __restrict__ w,
                                               float* __restrict__ out,
                                               int streams, int t, int hidden,
                                               float inv_streams) {
    int s = blockIdx.x;
    int tok = blockIdx.y;
    if (s >= streams || tok >= t) return;
    int tid = threadIdx.x;
    long n = (long)t * hidden;
    const float* wrow = w + (long)s * streams * hidden;
    float acc = 0.0f;
    for (int s2 = 0; s2 < streams; s2++) {
        const float* nrow = normed + (long)s2 * n + (long)tok * hidden;
        const float* wsub = wrow + (long)s2 * hidden;
        for (int d = tid; d < hidden; d += blockDim.x) acc += wsub[d] * nrow[d];
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    if (tid == 0) {
        float g = red[0] * inv_streams;
        out[(long)s * t + tok] = 2.0f / (1.0f + expf(-g));
    }
}

// ---- qwen4_exp bf16 trunk residency (perf lane item 1) ---------------------------------
// Batched/strided bf16-WEIGHT matvec, f32 activations, f32 accumulate. W bf16
// [batch, out_f, in_f] row-major; x f32 at x + b*x_bstride + tok*x_tstride (element
// strides; x_bstride = 0 shares one activation across the batch — the read gate's up
// projection); y f32 at y + b*y_bstride + tok*out_f + o. bf16 -> f32 widening is EXACT,
// so per-element products equal the f32 chain's when the resident bf16 bytes equal the
// checkpoint's (the loader's representability guard); only the reduction tree differs
// from cuBLASLt gemvx — the accumulation class, gated per row + argmax. Vectorized
// uint4 = 8 bf16 per load: in_f % 8 == 0 required (the loader's geometry guard keeps
// non-conforming tensors f32). grid (out_f, t, batch), block 128.
extern "C" __global__ void qmatvec_bf16w_f32(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t,
        long w_bstride, long x_bstride, long x_tstride, long y_bstride) {
    int o = blockIdx.x;
    int tok = blockIdx.y;
    int b = blockIdx.z;
    if (o >= out_f || tok >= t) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(
        w + (long)b * w_bstride + (size_t)o * (size_t)in_f);
    const float* xrow = x + (long)b * x_bstride + (long)tok * x_tstride;
    int n8 = in_f >> 3;
    float acc = 0.0f;
    for (int g = threadIdx.x; g < n8; g += blockDim.x) {
        uint4 pk = wrow[g];
        const float4* xv = reinterpret_cast<const float4*>(xrow + ((long)g << 3));
        float4 xa = xv[0];
        float4 xb = xv[1];
        acc = fmaf(__uint_as_float(pk.x << 16), xa.x, acc);
        acc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(pk.y << 16), xa.z, acc);
        acc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(pk.z << 16), xb.x, acc);
        acc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(pk.w << 16), xb.z, acc);
        acc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, acc);
    }
    __shared__ float red[32];
    int tid = threadIdx.x;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(long)b * y_bstride + (long)tok * out_f + o] = v;
    }
}

// Device-SELECTED expert twin of qmatvec_bf16w_f32 (devtwin lane, the DeviceBf16 draft
// bank): one launch covers every routed expert of one projection — grid (out_f, 1,
// n_sel), block 128; slot s reads its expert id from the DEVICE sel array (the route
// kernel's output) and its weight rows at sel[s]*out_f*in_f, writing y at s*out_f. The
// per-row program (g loop, 8-lane order, fmaf chain, block reduce tree) is
// qmatvec_bf16w_f32 VERBATIM => outputs are BIT-IDENTICAL to the per-slot
// launch_qmatvec_bf16w_off_into chain it replaces; only the launch count (n_sel -> 1)
// and the host expert-id dependency drop. x_sstride = per-SLOT activation stride in
// elements: 0 shares one row across slots (gate/up read the token's mixed row); in_f
// gives each slot its own row (down reads slot s's act row) — exactly the operand the
// per-slot off_into launch sliced.
extern "C" __global__ void qmatvec_bf16w_sel_f32(
        const unsigned short* __restrict__ w, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y, int in_f, int out_f,
        int n_sel, long x_sstride) {
    int o = blockIdx.x;
    int s = blockIdx.z;
    if (o >= out_f || s >= n_sel) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(
        w + (size_t)sel[s] * (size_t)out_f * (size_t)in_f + (size_t)o * (size_t)in_f);
    const float* xrow = x + (long)s * x_sstride;
    int n8 = in_f >> 3;
    float acc = 0.0f;
    for (int g = threadIdx.x; g < n8; g += blockDim.x) {
        uint4 pk = wrow[g];
        const float4* xv = reinterpret_cast<const float4*>(xrow + ((long)g << 3));
        float4 xa = xv[0];
        float4 xb = xv[1];
        acc = fmaf(__uint_as_float(pk.x << 16), xa.x, acc);
        acc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(pk.y << 16), xa.z, acc);
        acc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(pk.z << 16), xb.x, acc);
        acc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(pk.w << 16), xb.z, acc);
        acc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, acc);
    }
    __shared__ float red[32];
    int tid = threadIdx.x;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)s * (size_t)out_f + o] = v;
    }
}

// Multi-token WEIGHT-SHARED twin of qmatvec_bf16w_f32 (mtp-spec verify chunks): grid
// (out_f, 1, batch), one block per output row; each weight uint4 is loaded ONCE and
// FMA'd into every token's accumulator. Per (row, token) the fma CHAIN (g order, the
// 8-lane order inside g, the block reduce tree) is qmatvec_bf16w_f32 VERBATIM, so
// outputs are BIT-IDENTICAL to per-token launches — only the weight-read count drops
// from t to 1 (the qwen38 t-parallel verify lesson: batch the weight dim, keep per-row
// programs). t <= 12 (register accumulators).
extern "C" __global__ void qmatvec_bf16w_mt_f32(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t,
        long w_bstride, long x_bstride, long x_tstride, long y_bstride) {
    int o = blockIdx.x;
    int b = blockIdx.z;
    if (o >= out_f || t > 12) return;
    const uint4* wrow = reinterpret_cast<const uint4*>(
        w + (long)b * w_bstride + (size_t)o * (size_t)in_f);
    const float* xbase = x + (long)b * x_bstride;
    int n8 = in_f >> 3;
    float acc[12];
    for (int j = 0; j < 12; j++) acc[j] = 0.0f;
    for (int g = threadIdx.x; g < n8; g += blockDim.x) {
        uint4 pk = wrow[g];
        float w0 = __uint_as_float(pk.x << 16);
        float w1 = __uint_as_float(pk.x & 0xffff0000u);
        float w2 = __uint_as_float(pk.y << 16);
        float w3 = __uint_as_float(pk.y & 0xffff0000u);
        float w4 = __uint_as_float(pk.z << 16);
        float w5 = __uint_as_float(pk.z & 0xffff0000u);
        float w6 = __uint_as_float(pk.w << 16);
        float w7 = __uint_as_float(pk.w & 0xffff0000u);
        for (int j = 0; j < t; j++) {
            const float4* xv =
                reinterpret_cast<const float4*>(xbase + (long)j * x_tstride + ((long)g << 3));
            float4 xa = xv[0];
            float4 xb = xv[1];
            acc[j] = fmaf(w0, xa.x, acc[j]);
            acc[j] = fmaf(w1, xa.y, acc[j]);
            acc[j] = fmaf(w2, xa.z, acc[j]);
            acc[j] = fmaf(w3, xa.w, acc[j]);
            acc[j] = fmaf(w4, xb.x, acc[j]);
            acc[j] = fmaf(w5, xb.y, acc[j]);
            acc[j] = fmaf(w6, xb.z, acc[j]);
            acc[j] = fmaf(w7, xb.w, acc[j]);
        }
    }
    __shared__ float red[32];
    int tid = threadIdx.x;
    for (int j = 0; j < t; j++) {
        float a = acc[j];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if ((tid & 31) == 0) red[tid >> 5] = a;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) y[(long)b * y_bstride + (long)j * out_f + o] = v;
        }
        __syncthreads();
    }
}

// bf16-weight twin of hc_inject_gates_f32 (same grid, same loop order, same reduction
// tree — bf16 -> f32 widening is exact, so with representable resident bytes the output
// is BIT-IDENTICAL to the f32 twin). w bf16 [streams, streams*hidden] row-major.
extern "C" __global__ void hc_inject_gates_bf16w_f32(const float* __restrict__ normed,
                                                     const unsigned short* __restrict__ w,
                                                     float* __restrict__ out,
                                                     int streams, int t, int hidden,
                                                     float inv_streams) {
    int s = blockIdx.x;
    int tok = blockIdx.y;
    if (s >= streams || tok >= t) return;
    int tid = threadIdx.x;
    long n = (long)t * hidden;
    const unsigned short* wrow = w + (long)s * streams * hidden;
    float acc = 0.0f;
    for (int s2 = 0; s2 < streams; s2++) {
        const float* nrow = normed + (long)s2 * n + (long)tok * hidden;
        const unsigned short* wsub = wrow + (long)s2 * hidden;
        for (int d = tid; d < hidden; d += blockDim.x)
            acc += __uint_as_float((unsigned)wsub[d] << 16) * nrow[d];
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    if (tid == 0) {
        float g = red[0] * inv_streams;
        out[(long)s * t + tok] = 2.0f / (1.0f + expf(-g));
    }
}

// ---- qwen4_exp grouped sel matvec v2 (perf lane item 3) --------------------------------
// The v1 kernel above reads codes/scales one BYTE at a time (one warp per output row,
// scalar nibble unpack) and lands at ~225-275 GB/s on the modelopt bank — well under the
// card. v2 is the ornith sel-kernel craft ported to this dialect: uint4 code loads
// (32 codes = 2 scale groups per load), float4 activation loads, and TWO output rows per
// warp sharing the activation registers (the MEMRA_SEL_GU_RPW=2 shape). Per-element
// products and the `acc += scale * group_dot` chaining are IDENTICAL to v1; the
// group-dot internal reduction tree and the per-thread group partition differ — the
// accumulation class, gated by the same oracle + real gates. Geometry: in_f % 32 == 0
// and out_f % 2 == 0 (dispatcher falls
// back to v1 otherwise). grid ((out_f+1)/2, n_sel), block 32.
__device__ __forceinline__ float q4e_ue4m3(unsigned char sb) {
    int mag = sb & 0x7F;
    if (mag == 0x7F) return 0.0f; // modelopt NaN code -> 0.0
    int se = (mag >> 3) & 0xF;
    float man = (float)(mag & 0x7);
    float scale = (se == 0) ? (man / 8.0f) * exp2f(-6.0f)
                            : (1.0f + man / 8.0f) * exp2f((float)(se - 7));
    return (sb & 0x80) ? -scale : scale;
}

__device__ __forceinline__ float q4e_dot8(const float* lut, unsigned u, const float* xp) {
    float4 a = reinterpret_cast<const float4*>(xp)[0];
    float4 b = reinterpret_cast<const float4*>(xp)[1];
    float d = 0.0f;
    d += lut[u & 15] * a.x;
    d += lut[(u >> 4) & 15] * a.y;
    d += lut[(u >> 8) & 15] * a.z;
    d += lut[(u >> 12) & 15] * a.w;
    d += lut[(u >> 16) & 15] * b.x;
    d += lut[(u >> 20) & 15] * b.y;
    d += lut[(u >> 24) & 15] * b.z;
    d += lut[(u >> 28) & 15] * b.w;
    return d;
}

extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_f32_v2(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int o0 = blockIdx.x * 2;
    int slot = blockIdx.y;
    if (o0 >= out_f) return;
    int lane = threadIdx.x & 31; // block 32: lane == threadIdx.x (semantics unchanged)
    int e = sel[slot];
    const float* xrow = x + (size_t)slot * (size_t)x_stride;
    size_t row_codes = (size_t)in_f / 2;
    size_t row_scales = (size_t)in_f / 16;
    const uint4* c0 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0) * row_codes);
    const uint4* c1 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 1) * row_codes);
    const unsigned char* s0 = scales + ((size_t)e * out_f + o0) * row_scales;
    const unsigned char* s1 = s0 + row_scales;
    int pairs = in_f / 32; // one uint4 = 32 codes = 2 scale groups
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int p = lane; p < pairs; p += 32) {
        const float* xp = xrow + (size_t)p * 32;
        uint4 ca = c0[p];
        uint4 cb = c1[p];
        float dA0 = q4e_dot8(e2m1, ca.x, xp) + q4e_dot8(e2m1, ca.y, xp + 8);
        float dA1 = q4e_dot8(e2m1, ca.z, xp + 16) + q4e_dot8(e2m1, ca.w, xp + 24);
        float dB0 = q4e_dot8(e2m1, cb.x, xp) + q4e_dot8(e2m1, cb.y, xp + 8);
        float dB1 = q4e_dot8(e2m1, cb.z, xp + 16) + q4e_dot8(e2m1, cb.w, xp + 24);
        acc0 += q4e_ue4m3(s0[2 * p]) * dA0 + q4e_ue4m3(s0[2 * p + 1]) * dA1;
        acc1 += q4e_ue4m3(s1[2 * p]) * dB0 + q4e_ue4m3(s1[2 * p + 1]) * dB1;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
    }
    if (threadIdx.x == 0) {
        float m = macros[e];
        y[(size_t)slot * out_f + o0] = acc0 * m;
        y[(size_t)slot * out_f + o0 + 1] = acc1 * m;
    }
}

// ---- qwen4_exp TP2 direct-join push (perf round 3) ---------------------------------------
// UVA store of a small f32 vector into a PEER device's buffer (dst is a raw device
// address on the other card; P2P enabled by configure_native_p2p). The tp2-join-diet
// direct-join mechanism: the producing card's stream pushes its partial into the
// consumer's resident staging buffer — no pull copy, no cross-device memcpy engine
// turnaround. Values are copied verbatim.
extern "C" __global__ void q4e_push_f32(const float* __restrict__ src,
                                        unsigned long long dst, long n) {
    float* d = reinterpret_cast<float*>((size_t)dst);
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = src[i];
}

// ---- qwen4_exp GDN norm+gate fusion (perf round 3) --------------------------------------
// dst = rms_norm(x, w) * sigmoid(z), rows of `ncols` — one launch replaces the GDN
// mixer's rms_norm + sigmoid + mul chain (3 small kernels serialized per layer). The
// reduction and normalize are rms_norm_f32 VERBATIM (same tree, same scale multiply
// order) and the gate matches sigmoid_f32's expf form; the final multiply has no
// contraction seam (pure multiplies) — end-to-end BIT-IDENTICAL to the chain it
// replaces. grid (nrows), block rms_block().
extern "C" __global__ void rms_sigmul_f32(const float* __restrict__ x, const float* __restrict__ w,
                                          const float* __restrict__ z, float* __restrict__ dst,
                                          int ncols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * ncols;
    const float* zr = z + (size_t)row * ncols;
    float* dr = dst + (size_t)row * ncols;

    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = xr[i]; sum += v * v; }
    // block reduce — VERBATIM rms_norm_f32
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        float sig = 1.0f / (1.0f + expf(-zr[i]));
        dr[i] = xr[i] * scale * w[i] * sig;
    }
}

// ---- qwen4_exp grouped sel matvec v3 (perf round 3) -------------------------------------
// v2 sits at ~340-420 GB/s on the modelopt bank (PROFILE-2 §Residual): each warp owns 2
// rows and, at the artifact's down geometry (in_f 640 -> 20 uint4 pairs), a thread runs
// at most ONE strided iteration — almost no memory-level parallelism per warp. v3 takes
// FOUR output rows per warp sharing the float4 activation registers: per p-iteration a
// thread issues 4 independent uint4 code loads + 4 u16 scale loads against 8 shared
// activation float4 loads, quadrupling outstanding code traffic per warp. Per-element
// products and the per-row `acc += scale * group_dot` chaining are IDENTICAL to v1/v2;
// the per-thread p-partition (threadIdx.x strided by 32) matches v2 exactly — only
// compiler scheduling/fma contraction may differ, the accumulation class, gated by the
// same kernel oracle + real gates. Geometry: in_f % 32 == 0 && out_f % 4 == 0
// (dispatcher falls back v3 -> v2 -> v1). grid (out_f/4, n_sel), block 32.
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_f32_v3(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    // Warp-packed blocks (mtp-spec occupancy fix): each WARP owns 4 rows exactly as the
    // one-warp-per-block form did (same lane program, same shfl tree — bit-identical);
    // block 128 packs 4 warps so SM block-slot limits stop starving the launch. Block 32
    // reduces to the original indexing.
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    int slot = blockIdx.y;
    if (o0 >= out_f) return;
    int e = sel[slot];
    const float* xrow = x + (size_t)slot * (size_t)x_stride;
    size_t row_codes = (size_t)in_f / 2;
    size_t row_scales = (size_t)in_f / 16;
    const uint4* c0 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0) * row_codes);
    const uint4* c1 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 1) * row_codes);
    const uint4* c2 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 2) * row_codes);
    const uint4* c3 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 3) * row_codes);
    // row_scales is even (in_f % 32 == 0), so every scale row starts u16-aligned: one
    // 2-byte load fetches both group scales of a uint4's worth of codes.
    const unsigned short* s0 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0) * row_scales);
    const unsigned short* s1 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 1) * row_scales);
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 2) * row_scales);
    const unsigned short* s3 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 3) * row_scales);
    int pairs = in_f / 32; // one uint4 = 32 codes = 2 scale groups
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int p = lane; p < pairs; p += 32) {
        const float* xp = xrow + (size_t)p * 32;
        uint4 ca = c0[p];
        uint4 cb = c1[p];
        uint4 cc = c2[p];
        uint4 cd = c3[p];
        unsigned short sa = s0[p], sb = s1[p], sc = s2[p], sd = s3[p];
        float dA0 = q4e_dot8(e2m1, ca.x, xp) + q4e_dot8(e2m1, ca.y, xp + 8);
        float dA1 = q4e_dot8(e2m1, ca.z, xp + 16) + q4e_dot8(e2m1, ca.w, xp + 24);
        float dB0 = q4e_dot8(e2m1, cb.x, xp) + q4e_dot8(e2m1, cb.y, xp + 8);
        float dB1 = q4e_dot8(e2m1, cb.z, xp + 16) + q4e_dot8(e2m1, cb.w, xp + 24);
        float dC0 = q4e_dot8(e2m1, cc.x, xp) + q4e_dot8(e2m1, cc.y, xp + 8);
        float dC1 = q4e_dot8(e2m1, cc.z, xp + 16) + q4e_dot8(e2m1, cc.w, xp + 24);
        float dD0 = q4e_dot8(e2m1, cd.x, xp) + q4e_dot8(e2m1, cd.y, xp + 8);
        float dD1 = q4e_dot8(e2m1, cd.z, xp + 16) + q4e_dot8(e2m1, cd.w, xp + 24);
        acc0 += q4e_ue4m3((unsigned char)(sa & 0xFF)) * dA0 + q4e_ue4m3((unsigned char)(sa >> 8)) * dA1;
        acc1 += q4e_ue4m3((unsigned char)(sb & 0xFF)) * dB0 + q4e_ue4m3((unsigned char)(sb >> 8)) * dB1;
        acc2 += q4e_ue4m3((unsigned char)(sc & 0xFF)) * dC0 + q4e_ue4m3((unsigned char)(sc >> 8)) * dC1;
        acc3 += q4e_ue4m3((unsigned char)(sd & 0xFF)) * dD0 + q4e_ue4m3((unsigned char)(sd >> 8)) * dD1;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (lane == 0) {
        float m = macros[e];
        y[(size_t)slot * out_f + o0] = acc0 * m;
        y[(size_t)slot * out_f + o0 + 1] = acc1 * m;
        y[(size_t)slot * out_f + o0 + 2] = acc2 * m;
        y[(size_t)slot * out_f + o0 + 3] = acc3 * m;
    }
}

// ---- qwen4_exp TP2 count-gated MoE tail (perf round 3) -----------------------------------
// The TP2 expert split gives each card a VARIABLE number of selected slots (0..10 per
// token) — a captured graph cannot re-bake launch shapes, so these twins run at a FIXED
// grid (max_sel slots) and read the live slot count from a device-resident PACK blob:
// [max_sel i32 sel (padded)] [max_sel f32 weights (padded)] [1 i32 count], one H2D per
// card per layer into a parked workspace slot. Blocks with slot >= count retire at the
// first instruction; per-slot arithmetic is IDENTICAL to the v3 kernel / the sequential
// axpy chain, so the count-gated tail is bit-identical to the eager tail it replaces.
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_f32_v3c(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, unsigned long long pack, int max_sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride) {
    const int* __restrict__ meta = reinterpret_cast<const int*>((size_t)pack);
    int slot = blockIdx.y;
    if (slot >= meta[2 * max_sel]) return; // live count
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int o0 = blockIdx.x * 4;
    if (o0 >= out_f) return;
    int lane = threadIdx.x & 31; // block 32: lane == threadIdx.x (semantics unchanged)
    int e = meta[slot];
    const float* xrow = x + (size_t)slot * (size_t)x_stride;
    size_t row_codes = (size_t)in_f / 2;
    size_t row_scales = (size_t)in_f / 16;
    const uint4* c0 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0) * row_codes);
    const uint4* c1 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 1) * row_codes);
    const uint4* c2 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 2) * row_codes);
    const uint4* c3 = reinterpret_cast<const uint4*>(codes + ((size_t)e * out_f + o0 + 3) * row_codes);
    const unsigned short* s0 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0) * row_scales);
    const unsigned short* s1 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 1) * row_scales);
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 2) * row_scales);
    const unsigned short* s3 = reinterpret_cast<const unsigned short*>(
        scales + ((size_t)e * out_f + o0 + 3) * row_scales);
    int pairs = in_f / 32;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int p = lane; p < pairs; p += 32) {
        const float* xp = xrow + (size_t)p * 32;
        uint4 ca = c0[p];
        uint4 cb = c1[p];
        uint4 cc = c2[p];
        uint4 cd = c3[p];
        unsigned short sa = s0[p], sb = s1[p], sc = s2[p], sd = s3[p];
        float dA0 = q4e_dot8(e2m1, ca.x, xp) + q4e_dot8(e2m1, ca.y, xp + 8);
        float dA1 = q4e_dot8(e2m1, ca.z, xp + 16) + q4e_dot8(e2m1, ca.w, xp + 24);
        float dB0 = q4e_dot8(e2m1, cb.x, xp) + q4e_dot8(e2m1, cb.y, xp + 8);
        float dB1 = q4e_dot8(e2m1, cb.z, xp + 16) + q4e_dot8(e2m1, cb.w, xp + 24);
        float dC0 = q4e_dot8(e2m1, cc.x, xp) + q4e_dot8(e2m1, cc.y, xp + 8);
        float dC1 = q4e_dot8(e2m1, cc.z, xp + 16) + q4e_dot8(e2m1, cc.w, xp + 24);
        float dD0 = q4e_dot8(e2m1, cd.x, xp) + q4e_dot8(e2m1, cd.y, xp + 8);
        float dD1 = q4e_dot8(e2m1, cd.z, xp + 16) + q4e_dot8(e2m1, cd.w, xp + 24);
        acc0 += q4e_ue4m3((unsigned char)(sa & 0xFF)) * dA0 + q4e_ue4m3((unsigned char)(sa >> 8)) * dA1;
        acc1 += q4e_ue4m3((unsigned char)(sb & 0xFF)) * dB0 + q4e_ue4m3((unsigned char)(sb >> 8)) * dB1;
        acc2 += q4e_ue4m3((unsigned char)(sc & 0xFF)) * dC0 + q4e_ue4m3((unsigned char)(sc >> 8)) * dC1;
        acc3 += q4e_ue4m3((unsigned char)(sd & 0xFF)) * dD0 + q4e_ue4m3((unsigned char)(sd >> 8)) * dD1;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        acc2 += __shfl_down_sync(0xffffffff, acc2, off);
        acc3 += __shfl_down_sync(0xffffffff, acc3, off);
    }
    if (threadIdx.x == 0) {
        float m = macros[e];
        y[(size_t)slot * out_f + o0] = acc0 * m;
        y[(size_t)slot * out_f + o0 + 1] = acc1 * m;
        y[(size_t)slot * out_f + o0 + 2] = acc2 * m;
        y[(size_t)slot * out_f + o0 + 3] = acc3 * m;
    }
}

// Count-gated slot combine twin of axpy_rows_seq_f32: weights + live count from the
// pack blob; the p-order sum is the base kernel's exact sequential chain over the live
// slots (count == 0 writes zeros — the empty-card case).
extern "C" __global__ void axpy_rows_seq_pack_f32(
        const float* __restrict__ x, unsigned long long pack, int max_sel,
        float* __restrict__ y, int width) {
    const int* __restrict__ meta = reinterpret_cast<const int*>((size_t)pack);
    const float* __restrict__ w = reinterpret_cast<const float*>((size_t)pack) + max_sel;
    int count = meta[2 * max_sel];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float acc = 0.0f;
    for (int p = 0; p < count; p++) acc += w[p] * x[(size_t)p * width + i];
    y[i] = acc;
}

// ---- qwen4_exp read/write-gate micro bundle (perf lane, set_hc_micro seam) -------------
// Post item-1/2/3 nsys: the read gate's residue is EXECUTION, not launches — 384 one-
// block rms_norm launches (~3.7us each), 96 inject launches at grid (streams, 1) = 4
// blocks on a 188-SM card (~18us each, latency-bound), 384 add_scaled_rows + 384 inject
// row d2d copies. These kernels batch across streams via a device POINTER TABLE (the
// copy_batch_uniform_f32 precedent) since the stream planes are separate buffers.

// Per-(stream, token) RMSNorm over the plane table into the stream-major normed slab.
// Same math per row as rms_norm_f32 (mean-square, rsqrt, effective weight); the block
// reduction tree is this kernel's own — accumulation class, gated. grid (t, streams),
// block 256.
extern "C" __global__ void hc_norm_planes_f32(const unsigned long long* __restrict__ planes,
                                              const float* __restrict__ w_stack,
                                              float* __restrict__ dst,
                                              int hidden, int t, float eps) {
    int tok = blockIdx.x;
    int s = blockIdx.y;
    const float* x = reinterpret_cast<const float*>(planes[s]) + (size_t)tok * hidden;
    const float* w = w_stack + (size_t)s * hidden;
    float* out = dst + ((size_t)s * t + tok) * hidden;
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[i];
        acc += v * v;
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / (float)hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

// Inject stage 1: partial dots of block_inject rows against the wide normed token, C
// chunks per (s, t) so the launch fills the card (the single-stage kernel ran 4 blocks).
// partials[(s*t + tok)*C + c]; deterministic (no atomics — the greedy instrument needs
// byte-stable replays). w is f32 or bf16 (twin below). grid (streams, t, C), block 256.
extern "C" __global__ void hc_inject_partials_f32(const float* __restrict__ normed,
                                                  const float* __restrict__ w,
                                                  float* __restrict__ partials,
                                                  int streams, int t, int hidden, int chunks) {
    int s = blockIdx.x;
    int tok = blockIdx.y;
    int c = blockIdx.z;
    int tid = threadIdx.x;
    long n = (long)t * hidden;
    long wide = (long)streams * hidden;
    long per = (wide + chunks - 1) / chunks;
    long lo = (long)c * per;
    long hi = min(lo + per, wide);
    const float* wrow = w + (long)s * wide;
    float acc = 0.0f;
    for (long j = lo + tid; j < hi; j += blockDim.x) {
        int s2 = (int)(j / hidden);
        int d = (int)(j % hidden);
        acc += wrow[j] * normed[(long)s2 * n + (long)tok * hidden + d];
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        // Store the BLOCK sum `v` — storing warp 0's `acc` here was the perf7 layer-0
        // corruption: exact at tiny geometry (every element lands in warp 0), wrong the
        // moment the chunk spans more than one warp. Caught by gate_hc_micro_kernels.
        if (tid == 0) partials[((long)s * t + tok) * chunks + c] = v;
    }
}

extern "C" __global__ void hc_inject_partials_bf16w_f32(const float* __restrict__ normed,
                                                        const unsigned short* __restrict__ w,
                                                        float* __restrict__ partials,
                                                        int streams, int t, int hidden,
                                                        int chunks) {
    int s = blockIdx.x;
    int tok = blockIdx.y;
    int c = blockIdx.z;
    int tid = threadIdx.x;
    long n = (long)t * hidden;
    long wide = (long)streams * hidden;
    long per = (wide + chunks - 1) / chunks;
    long lo = (long)c * per;
    long hi = min(lo + per, wide);
    const unsigned short* wrow = w + (long)s * wide;
    float acc = 0.0f;
    for (long j = lo + tid; j < hi; j += blockDim.x) {
        int s2 = (int)(j / hidden);
        int d = (int)(j % hidden);
        acc += __uint_as_float((unsigned)wrow[j] << 16) *
               normed[(long)s2 * n + (long)tok * hidden + d];
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        // Store the BLOCK sum `v` — storing warp 0's `acc` here was the perf7 layer-0
        // corruption: exact at tiny geometry (every element lands in warp 0), wrong the
        // moment the chunk spans more than one warp. Caught by gate_hc_micro_kernels.
        if (tid == 0) partials[((long)s * t + tok) * chunks + c] = v;
    }
}

// Inject stage 2: out[s, t] = 2*sigmoid(inv_streams * sum_c partials) — the sequential
// c-order sum keeps replays byte-stable. grid (streams*t + 255)/256, block 256.
extern "C" __global__ void hc_inject_reduce_f32(const float* __restrict__ partials,
                                                float* __restrict__ out,
                                                int rows, int chunks, float inv_streams) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float acc = 0.0f;
    for (int c = 0; c < chunks; c++) acc += partials[(long)r * chunks + c];
    float g = acc * inv_streams;
    out[r] = 2.0f / (1.0f + expf(-g));
}

// Slab write gate: plane_s[tok, :] += block_out[tok, :] * inj[s*t + tok], one launch for
// all streams (was `streams` add_scaled_rows + `streams` d2d row copies per gate). Same
// single multiply-add per element as add_scaled_rows_f32. grid (ceil(t*hidden/256),
// streams), block 256.
extern "C" __global__ void hc_write_planes_f32(const unsigned long long* __restrict__ planes,
                                               const float* __restrict__ block_out,
                                               const float* __restrict__ inj,
                                               int hidden, int t) {
    int s = blockIdx.y;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)t * hidden) return;
    int tok = (int)(i / hidden);
    float* plane = reinterpret_cast<float*>(planes[s]);
    plane[i] += block_out[i] * inj[(long)s * t + tok];
}

// ---- qwen4_exp projection stack (perf round 4, set_proj_stack seam) ---------------------
// Same-activation trunk projections that today run as SEPARATE qmatvec_bf16w_f32 launches
// (GDN qkv/z/beta/alpha, QSA wq/wk/wv, shared-expert gate/up) collapse into ONE launch
// over a LOAD-TIME row-stacked bf16 mat, with each output row routed to its original
// destination buffer by row range (up to 4 parts, raw device pointers as args — no d2d
// copies, no layout change for consumers). The per-row math is qmatvec_bf16w_f32
// VERBATIM (same uint4=8-bf16 loads, same block-128 two-level reduce, same fma chain),
// so every output value is BIT-IDENTICAL to the per-mat launches it replaces; only the
// grid packing changes. t == 1 (decode); grid (r0+r1+r2+r3, 1, 1), block 128.
extern "C" __global__ void qmatvec_bf16w_multi4_f32(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        unsigned long long y0, int r0, unsigned long long y1, int r1,
        unsigned long long y2, int r2, unsigned long long y3, int r3,
        int in_f) {
    int o = blockIdx.x;
    float* dst;
    int local = o;
    if (local < r0) {
        dst = reinterpret_cast<float*>((size_t)y0);
    } else if ((local -= r0) < r1) {
        dst = reinterpret_cast<float*>((size_t)y1);
    } else if ((local -= r1) < r2) {
        dst = reinterpret_cast<float*>((size_t)y2);
    } else if ((local -= r2) < r3) {
        dst = reinterpret_cast<float*>((size_t)y3);
    } else {
        return;
    }
    const uint4* wrow = reinterpret_cast<const uint4*>(w + (size_t)o * (size_t)in_f);
    const float* xrow = x;
    int n8 = in_f >> 3;
    float acc = 0.0f;
    for (int g = threadIdx.x; g < n8; g += blockDim.x) {
        uint4 pk = wrow[g];
        const float4* xv = reinterpret_cast<const float4*>(xrow + ((long)g << 3));
        float4 xa = xv[0];
        float4 xb = xv[1];
        acc = fmaf(__uint_as_float(pk.x << 16), xa.x, acc);
        acc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, acc);
        acc = fmaf(__uint_as_float(pk.y << 16), xa.z, acc);
        acc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, acc);
        acc = fmaf(__uint_as_float(pk.z << 16), xb.x, acc);
        acc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, acc);
        acc = fmaf(__uint_as_float(pk.w << 16), xb.z, acc);
        acc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, acc);
    }
    __shared__ float red[32];
    int tid = threadIdx.x;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) dst[local] = v;
    }
}

// ---- qwen4_exp hyper-gate diet (perf round 4, set_hc_diet seam) -------------------------
// PROFILE-3 residual: the replicated read gates burn 3.7 ms/card of which only ~0.9 ms is
// bandwidth floor — a 7-launch serial chain (norm, batched down GEMV, lowrank reduce,
// batched up GEMV, mix epilogue, inject partials, inject reduce) whose per-launch grids
// underfill the card. The diet re-fuses the READ GATE into THREE launches at t == 1:
//
//   stage 1: per (row-chunk, stream) block — recompute the stream's RMS scale from the
//            raw plane (redundant per block, deterministic), materialize the normed row
//            in smem, then run this block's DOWN rows + INJECT partial rows against it.
//   stage 2: low_act = silu(mean_s parts) (the hc_lowrank_reduce association VERBATIM)
//            + inj[j] = 2*sigmoid(mean_s2 inj_parts) in one tiny launch.
//   stage 3: per dim-chunk block — the UP dots for all streams from a smem copy of
//            low_act, then mixed[d] = mean_s sigmoid(up_s·low)·(plane_s[d]·inv_s·nw[d])
//            with the per-stream inv scalars from stage 1 (so the recomputed normed
//            values are IDENTICAL to stage 1's).
//
// ACCUMULATION CLASS throughout: the RMS reduce, row dots, and up dots use this file's
// standard two-level trees at different widths than the kernels they replace; sigmoid /
// silu / the s-ascending mean associations match the fused-gate kernels verbatim. Gated
// by gate_hc_diet_kernels (real geometry, vs the classic fused chain) + the real gates.

// stage 1: grid (ceil((rank+n_inj)/rows_pb), t, streams), block 256,
// smem hidden*4 bytes. wdown bf16 [S, rank, hidden]; winj bf16 [S, S*hidden] (row j,
// window s2). parts f32 [t, S, rank]; inj_parts f32 [t, n_inj, S]; inv f32 [t, S].
// The token dim (blockIdx.y, mtp-spec verify chunks) runs the t == 1 program VERBATIM
// per token at plane offset tok*hidden — per-token outputs are bit-identical to t == 1
// launches (decode is tok == 0 of a t == 1 grid, indexing unchanged).
extern "C" __global__ void hc_diet_stage1_f32(
        const unsigned long long* __restrict__ planes, const float* __restrict__ nw,
        const unsigned short* __restrict__ wdown, const unsigned short* __restrict__ winj,
        float* __restrict__ parts, float* __restrict__ inj_parts, float* __restrict__ inv_out,
        int hidden, int rank, int streams, int n_inj, int rows_pb, float eps) {
    extern __shared__ float sm[]; // [hidden] normed row of this (stream, token)
    int s = blockIdx.z;
    int tok = blockIdx.y;
    int tid = threadIdx.x;
    const float* x = reinterpret_cast<const float*>(planes[s]) + (size_t)tok * hidden;
    // RMS sumsq (redundant per block, deterministic — every block reduces the same way).
    float acc = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[i];
        sm[i] = v;
        acc += v * v;
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(red[0] / (float)hidden + eps);
    if (blockIdx.x == 0 && tid == 0) inv_out[(size_t)tok * streams + s] = inv;
    // Normalize in place: sm[d] = x[d] * inv * nw[s*hidden + d].
    const float* nwr = nw + (size_t)s * hidden;
    for (int i = tid; i < hidden; i += blockDim.x) sm[i] = sm[i] * inv * nwr[i];
    __syncthreads();
    // This block's rows: down rows [0, rank), inject rows [rank, rank+n_inj).
    int n8 = hidden >> 3;
    for (int r_local = 0; r_local < rows_pb; r_local++) {
        int row = blockIdx.x * rows_pb + r_local;
        if (row >= rank + n_inj) break;
        const unsigned short* wrow =
            (row < rank) ? wdown + ((size_t)s * rank + row) * (size_t)hidden
                         : winj + ((size_t)(row - rank) * streams + s) * (size_t)hidden;
        const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
        float racc = 0.0f;
        for (int g = tid; g < n8; g += blockDim.x) {
            uint4 pk = w4[g];
            const float4* xv = reinterpret_cast<const float4*>(sm + ((long)g << 3));
            float4 xa = xv[0];
            float4 xb = xv[1];
            racc = fmaf(__uint_as_float(pk.x << 16), xa.x, racc);
            racc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, racc);
            racc = fmaf(__uint_as_float(pk.y << 16), xa.z, racc);
            racc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, racc);
            racc = fmaf(__uint_as_float(pk.z << 16), xb.x, racc);
            racc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, racc);
            racc = fmaf(__uint_as_float(pk.w << 16), xb.z, racc);
            racc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, racc);
        }
        for (int o = 16; o > 0; o >>= 1) racc += __shfl_down_sync(0xffffffff, racc, o);
        if ((tid & 31) == 0) red[tid >> 5] = racc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
            if (tid == 0) {
                if (row < rank) parts[((size_t)tok * streams + s) * rank + row] = v;
                else inj_parts[((size_t)tok * n_inj + (row - rank)) * streams + s] = v;
            }
        }
        __syncthreads();
    }
}

// stage 2: low_act[tok, r] = silu(inv_streams * sum_s parts[tok, s, r]) — the
// hc_lowrank_reduce association VERBATIM (s-ascending sum, post-sum scale, same silu
// form); plus inj_all[j, tok] = 2*sigmoid(inv_streams * sum_s2 inj_parts[tok, j, s2])
// — the [S, t] slab layout hc_write_planes consumes. grid (ceil((rank+n_inj)/256), t);
// per-token math identical to the t == 1 launch.
extern "C" __global__ void hc_diet_stage2_f32(
        const float* __restrict__ parts, const float* __restrict__ inj_parts,
        float* __restrict__ low_act, float* __restrict__ inj_all,
        int rank, int streams, int n_inj, int t, float inv_streams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    if (idx < rank) {
        float acc = 0.0f;
        for (int s = 0; s < streams; s++)
            acc += parts[((size_t)tok * streams + s) * rank + idx];
        float g = acc * inv_streams;
        low_act[(size_t)tok * rank + idx] = g / (1.0f + expf(-g));
    } else if (idx - rank < n_inj) {
        int j = idx - rank;
        float acc = 0.0f;
        for (int s = 0; s < streams; s++)
            acc += inj_parts[((size_t)tok * n_inj + j) * streams + s];
        float g = acc * inv_streams;
        inj_all[(size_t)j * t + tok] = 2.0f / (1.0f + expf(-g));
    }
}

// stage 3: grid (ceil(hidden/dims_pb), t, 1), block 256, smem rank + dims_pb*streams
// floats. wup bf16 [S, hidden, rank]. Each warp takes up-dots k = i*streams + s round
// robin; thread i then finalizes dim d0+i with the hc_mix_epilogue association
// (s-ascending sigmoid*normed sum, post-sum scale), normed recomputed as
// plane_s[tok*hidden + d]*inv[tok,s]*nw[s][d] — identical values to stage 1's smem row.
// Token dim = blockIdx.y (per-token program == the t == 1 launch).
extern "C" __global__ void hc_diet_stage3_f32(
        const unsigned long long* __restrict__ planes, const float* __restrict__ nw,
        const float* __restrict__ inv_in, const unsigned short* __restrict__ wup,
        const float* __restrict__ low_act, float* __restrict__ mixed,
        int hidden, int rank, int streams, int dims_pb, float inv_streams) {
    extern __shared__ float sm[]; // [rank] low_act, then [dims_pb*streams] up dots
    float* gdots = sm + rank;
    int tid = threadIdx.x;
    int tok = blockIdx.y;
    int d0 = blockIdx.x * dims_pb;
    for (int i = tid; i < rank; i += blockDim.x) sm[i] = low_act[(size_t)tok * rank + i];
    __syncthreads();
    int warp = tid >> 5;
    int lane = tid & 31;
    int n_warps = blockDim.x >> 5;
    int n_dots = dims_pb * streams;
    bool vec8 = (rank & 7) == 0;
    for (int k = warp; k < n_dots; k += n_warps) {
        int i = k / streams;
        int s = k - i * streams;
        int d = d0 + i;
        float acc = 0.0f;
        if (d < hidden) {
            const unsigned short* wrow = wup + ((size_t)s * hidden + d) * (size_t)rank;
            if (vec8) {
                const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
                int n8 = rank >> 3;
                for (int g = lane; g < n8; g += 32) {
                    uint4 pk = w4[g];
                    const float4* xv = reinterpret_cast<const float4*>(sm + ((long)g << 3));
                    float4 xa = xv[0];
                    float4 xb = xv[1];
                    acc = fmaf(__uint_as_float(pk.x << 16), xa.x, acc);
                    acc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, acc);
                    acc = fmaf(__uint_as_float(pk.y << 16), xa.z, acc);
                    acc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, acc);
                    acc = fmaf(__uint_as_float(pk.z << 16), xb.x, acc);
                    acc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, acc);
                    acc = fmaf(__uint_as_float(pk.w << 16), xb.z, acc);
                    acc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, acc);
                }
            } else {
                for (int g = lane; g < rank; g += 32)
                    acc = fmaf(__uint_as_float((unsigned)wrow[g] << 16), sm[g], acc);
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        }
        if (lane == 0) gdots[k] = acc;
    }
    __syncthreads();
    if (tid < dims_pb) {
        int d = d0 + tid;
        if (d < hidden) {
            float acc = 0.0f;
            for (int s = 0; s < streams; s++) {
                float g = gdots[tid * streams + s];
                float sig = 1.0f / (1.0f + expf(-g));
                const float* x = reinterpret_cast<const float*>(planes[s]) + (size_t)tok * hidden;
                float normed = x[d] * inv_in[(size_t)tok * streams + s] * nw[(size_t)s * hidden + d];
                acc += sig * normed;
            }
            mixed[(size_t)tok * hidden + d] = acc * inv_streams;
        }
    }
}

// ---- qwen4_exp hc-diet MT (mtp-spec verify: weight-shared token batching) ---------------
// The diet's token grid dim (above) re-reads every down/inject/up weight row PER TOKEN —
// t-linear bytes, the wrong side of the t-parallel law. These twins read each weight row
// ONCE and iterate tokens inside, with every per-(row, token) arithmetic chain the
// stage1/stage3 program VERBATIM:
//   - stage0_mt: the stage1 RMS sumsq reduce EXACTLY (same block 256 stride, same
//     two-level tree, same rsqrtf form) per (token, stream) -> inv[t, s]. Bit-equal to
//     the inv stage1 computes in-block.
//   - stage1_mt: per weight uint4 g, the 8 nw values load once and each token's 8 plane
//     values normalize INLINE as (x*inv)*nw — the same two multiplies stage1 applies
//     when it materializes sm[d] — then the same 8-fma chain per (row, token).
//   - stage3_mt: all T low_act rows live in smem; per (dim, stream) warp the up dot
//     runs per token with the same lane-strided chain + shfl tree; the epilogue is the
//     stage3 form with inv[t, s].
// Outputs are BIT-IDENTICAL per token to the token-grid kernels (gate oracle asserts).

extern "C" __global__ void hc_diet_stage0_mt_f32(
        const unsigned long long* __restrict__ planes, float* __restrict__ inv_out,
        int hidden, int streams, int t, float eps) {
    int tok = blockIdx.x;
    int s = blockIdx.y;
    if (tok >= t || s >= streams) return;
    int tid = threadIdx.x;
    const float* x = reinterpret_cast<const float*>(planes[s]) + (size_t)tok * hidden;
    float acc = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = x[i];
        acc += v * v;
    }
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((tid & 31) == 0) red[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) inv_out[(size_t)tok * streams + s] = rsqrtf(v / (float)hidden + eps);
    }
}

// grid (ceil((rank+n_inj)/rows_pb), 1, streams), block 256. t <= 12.
extern "C" __global__ void hc_diet_stage1_mt_f32(
        const unsigned long long* __restrict__ planes, const float* __restrict__ nw,
        const float* __restrict__ inv_in,
        const unsigned short* __restrict__ wdown, const unsigned short* __restrict__ winj,
        float* __restrict__ parts, float* __restrict__ inj_parts,
        int hidden, int rank, int streams, int n_inj, int rows_pb, int t) {
    int s = blockIdx.z;
    int tid = threadIdx.x;
    if (t > 12) return;
    const float* x0 = reinterpret_cast<const float*>(planes[s]);
    const float* nwr = nw + (size_t)s * hidden;
    float inv[12];
    for (int j = 0; j < t; j++) inv[j] = inv_in[(size_t)j * streams + s];
    int n8 = hidden >> 3;
    __shared__ float red[32];
    for (int r_local = 0; r_local < rows_pb; r_local++) {
        int row = blockIdx.x * rows_pb + r_local;
        if (row >= rank + n_inj) break;
        const unsigned short* wrow =
            (row < rank) ? wdown + ((size_t)s * rank + row) * (size_t)hidden
                         : winj + ((size_t)(row - rank) * streams + s) * (size_t)hidden;
        const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
        const float4* nw4 = reinterpret_cast<const float4*>(nwr);
        float racc[12];
        for (int j = 0; j < 12; j++) racc[j] = 0.0f;
        for (int g = tid; g < n8; g += blockDim.x) {
            uint4 pk = w4[g];
            float w0 = __uint_as_float(pk.x << 16);
            float w1 = __uint_as_float(pk.x & 0xffff0000u);
            float w2 = __uint_as_float(pk.y << 16);
            float w3 = __uint_as_float(pk.y & 0xffff0000u);
            float w4v = __uint_as_float(pk.z << 16);
            float w5 = __uint_as_float(pk.z & 0xffff0000u);
            float w6 = __uint_as_float(pk.w << 16);
            float w7 = __uint_as_float(pk.w & 0xffff0000u);
            float4 na = nw4[2 * g];
            float4 nb = nw4[2 * g + 1];
            for (int j = 0; j < t; j++) {
                const float4* xv =
                    reinterpret_cast<const float4*>(x0 + (size_t)j * hidden + ((long)g << 3));
                float4 xa = xv[0];
                float4 xb = xv[1];
                float iv = inv[j];
                racc[j] = fmaf(w0, (xa.x * iv) * na.x, racc[j]);
                racc[j] = fmaf(w1, (xa.y * iv) * na.y, racc[j]);
                racc[j] = fmaf(w2, (xa.z * iv) * na.z, racc[j]);
                racc[j] = fmaf(w3, (xa.w * iv) * na.w, racc[j]);
                racc[j] = fmaf(w4v, (xb.x * iv) * nb.x, racc[j]);
                racc[j] = fmaf(w5, (xb.y * iv) * nb.y, racc[j]);
                racc[j] = fmaf(w6, (xb.z * iv) * nb.z, racc[j]);
                racc[j] = fmaf(w7, (xb.w * iv) * nb.w, racc[j]);
            }
        }
        for (int j = 0; j < t; j++) {
            float a = racc[j];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
            if ((tid & 31) == 0) red[tid >> 5] = a;
            __syncthreads();
            if (tid < 32) {
                float v = (tid < (blockDim.x + 31) / 32) ? red[tid] : 0.0f;
                for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
                if (tid == 0) {
                    if (row < rank) parts[((size_t)j * streams + s) * rank + row] = v;
                    else inj_parts[((size_t)j * n_inj + (row - rank)) * streams + s] = v;
                }
            }
            __syncthreads();
        }
    }
}

// grid (ceil(hidden/dims_pb), 1, 1), block 256, smem (t*rank + dims_pb*streams*t) floats.
extern "C" __global__ void hc_diet_stage3_mt_f32(
        const unsigned long long* __restrict__ planes, const float* __restrict__ nw,
        const float* __restrict__ inv_in, const unsigned short* __restrict__ wup,
        const float* __restrict__ low_act, float* __restrict__ mixed,
        int hidden, int rank, int streams, int dims_pb, int t, float inv_streams) {
    extern __shared__ float sm[]; // [t*rank] low_act rows, then [dims_pb*streams*t] dots
    float* gdots = sm + (size_t)t * rank;
    int tid = threadIdx.x;
    int d0 = blockIdx.x * dims_pb;
    for (int i = tid; i < t * rank; i += blockDim.x) sm[i] = low_act[i];
    __syncthreads();
    int warp = tid >> 5;
    int lane = tid & 31;
    int n_warps = blockDim.x >> 5;
    int n_dots = dims_pb * streams;
    bool vec8 = (rank & 7) == 0;
    for (int k = warp; k < n_dots; k += n_warps) {
        int i = k / streams;
        int s = k - i * streams;
        int d = d0 + i;
        if (d >= hidden) continue;
        const unsigned short* wrow = wup + ((size_t)s * hidden + d) * (size_t)rank;
        for (int j = 0; j < t; j++) {
            const float* la = sm + (size_t)j * rank;
            float acc = 0.0f;
            if (vec8) {
                const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
                int n8 = rank >> 3;
                for (int g = lane; g < n8; g += 32) {
                    uint4 pk = w4[g];
                    const float4* xv = reinterpret_cast<const float4*>(la + ((long)g << 3));
                    float4 xa = xv[0];
                    float4 xb = xv[1];
                    acc = fmaf(__uint_as_float(pk.x << 16), xa.x, acc);
                    acc = fmaf(__uint_as_float(pk.x & 0xffff0000u), xa.y, acc);
                    acc = fmaf(__uint_as_float(pk.y << 16), xa.z, acc);
                    acc = fmaf(__uint_as_float(pk.y & 0xffff0000u), xa.w, acc);
                    acc = fmaf(__uint_as_float(pk.z << 16), xb.x, acc);
                    acc = fmaf(__uint_as_float(pk.z & 0xffff0000u), xb.y, acc);
                    acc = fmaf(__uint_as_float(pk.w << 16), xb.z, acc);
                    acc = fmaf(__uint_as_float(pk.w & 0xffff0000u), xb.w, acc);
                }
            } else {
                for (int g = lane; g < rank; g += 32)
                    acc = fmaf(__uint_as_float((unsigned)wrow[g] << 16), la[g], acc);
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
            if (lane == 0) gdots[(size_t)k * t + j] = acc;
        }
    }
    __syncthreads();
    if (tid < dims_pb) {
        int d = d0 + tid;
        if (d < hidden) {
            for (int j = 0; j < t; j++) {
                float acc = 0.0f;
                for (int s = 0; s < streams; s++) {
                    float g = gdots[(size_t)(tid * streams + s) * t + j];
                    float sig = 1.0f / (1.0f + expf(-g));
                    const float* x =
                        reinterpret_cast<const float*>(planes[s]) + (size_t)j * hidden;
                    float normed =
                        x[d] * inv_in[(size_t)j * streams + s] * nw[(size_t)s * hidden + d];
                    acc += sig * normed;
                }
                mixed[(size_t)j * hidden + d] = acc * inv_streams;
            }
        }
    }
}

// ---- qwen4_exp fused gate+up+silu sel matvec (perf round 4, set_sel_gufuse seam) --------
// The MoE tail's gate launch + up launch + silu launch collapse into ONE: each warp
// computes 4 GATE rows and 4 UP rows at o0 (independent banks, shared f32 activation
// registers — activations stay f32 per the owner order retiring activation
// quantization) and writes act[slot, o] = silu(gate_o) * up_o. Per-row arithmetic is
// qmatvec_nvfp4_modelopt_sel_f32_v3 VERBATIM (same LUT extracts, same
// `acc += scale * group_dot` chaining, same p-partition, same shfl tree, same
// post-reduce macro multiply) and the epilogue matches silu_mul_f32's
// `g / (1 + expf(-g)) * up` element form — BIT-IDENTICAL to the three-launch chain it
// replaces (asserted by the sel kernel oracle's gufuse mode). Doubles the outstanding
// code loads per warp on top of the fusion. Geometry: in_f % 32 == 0 && ff % 4 == 0.
// grid (ff/4, n_sel), block 32.
__device__ __forceinline__ void q4e_gu_rows4(
        const unsigned char* codes, const unsigned char* scales, const float* e2m1,
        size_t bank_row_off, int o0, int in_f, const float* xrow, int pairs, int lane,
        float* acc /*[4]*/) {
    size_t row_codes = (size_t)in_f / 2;
    size_t row_scales = (size_t)in_f / 16;
    const uint4* c0 = reinterpret_cast<const uint4*>(codes + (bank_row_off + o0) * row_codes);
    const uint4* c1 = reinterpret_cast<const uint4*>(codes + (bank_row_off + o0 + 1) * row_codes);
    const uint4* c2 = reinterpret_cast<const uint4*>(codes + (bank_row_off + o0 + 2) * row_codes);
    const uint4* c3 = reinterpret_cast<const uint4*>(codes + (bank_row_off + o0 + 3) * row_codes);
    const unsigned short* s0 = reinterpret_cast<const unsigned short*>(
        scales + (bank_row_off + o0) * row_scales);
    const unsigned short* s1 = reinterpret_cast<const unsigned short*>(
        scales + (bank_row_off + o0 + 1) * row_scales);
    const unsigned short* s2 = reinterpret_cast<const unsigned short*>(
        scales + (bank_row_off + o0 + 2) * row_scales);
    const unsigned short* s3 = reinterpret_cast<const unsigned short*>(
        scales + (bank_row_off + o0 + 3) * row_scales);
    for (int p = lane; p < pairs; p += 32) {
        const float* xp = xrow + (size_t)p * 32;
        uint4 ca = c0[p];
        uint4 cb = c1[p];
        uint4 cc = c2[p];
        uint4 cd = c3[p];
        unsigned short sa = s0[p], sb = s1[p], sc = s2[p], sd = s3[p];
        float dA0 = q4e_dot8(e2m1, ca.x, xp) + q4e_dot8(e2m1, ca.y, xp + 8);
        float dA1 = q4e_dot8(e2m1, ca.z, xp + 16) + q4e_dot8(e2m1, ca.w, xp + 24);
        float dB0 = q4e_dot8(e2m1, cb.x, xp) + q4e_dot8(e2m1, cb.y, xp + 8);
        float dB1 = q4e_dot8(e2m1, cb.z, xp + 16) + q4e_dot8(e2m1, cb.w, xp + 24);
        float dC0 = q4e_dot8(e2m1, cc.x, xp) + q4e_dot8(e2m1, cc.y, xp + 8);
        float dC1 = q4e_dot8(e2m1, cc.z, xp + 16) + q4e_dot8(e2m1, cc.w, xp + 24);
        float dD0 = q4e_dot8(e2m1, cd.x, xp) + q4e_dot8(e2m1, cd.y, xp + 8);
        float dD1 = q4e_dot8(e2m1, cd.z, xp + 16) + q4e_dot8(e2m1, cd.w, xp + 24);
        acc[0] += q4e_ue4m3((unsigned char)(sa & 0xFF)) * dA0 + q4e_ue4m3((unsigned char)(sa >> 8)) * dA1;
        acc[1] += q4e_ue4m3((unsigned char)(sb & 0xFF)) * dB0 + q4e_ue4m3((unsigned char)(sb >> 8)) * dB1;
        acc[2] += q4e_ue4m3((unsigned char)(sc & 0xFF)) * dC0 + q4e_ue4m3((unsigned char)(sc >> 8)) * dC1;
        acc[3] += q4e_ue4m3((unsigned char)(sd & 0xFF)) * dD0 + q4e_ue4m3((unsigned char)(sd >> 8)) * dD1;
    }
}

// tok_map (mtp-spec verify): a nonzero pointer maps slot -> token, and the slot's
// activation row becomes x + tok_map[slot]*x_tstride — ONE launch covers every verify
// column's routed experts (per-slot row program unchanged => bit-identical to the
// per-token launches it replaces; the weight banks are read once per selected slot
// either way — the launch count is what drops).
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_gu_silu_f32(
        const unsigned char* __restrict__ gcodes, const unsigned char* __restrict__ gscales,
        const float* __restrict__ gmacros,
        const unsigned char* __restrict__ ucodes, const unsigned char* __restrict__ uscales,
        const float* __restrict__ umacros,
        const int* __restrict__ sel, unsigned long long pack, int max_sel,
        const float* __restrict__ x, float* __restrict__ act, int in_f, int ff,
        unsigned long long tok_map, long x_tstride) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int slot = blockIdx.y;
    int e;
    if (pack != 0) {
        const int* meta = reinterpret_cast<const int*>((size_t)pack);
        if (slot >= meta[2 * max_sel]) return; // live count (count-gated TP2 twin)
        e = meta[slot];
    } else {
        e = sel[slot];
    }
    // Warp-packed blocks (see the v3 note): each warp owns 4 rows, block 32 reduces to
    // the original one-warp form — per-row programs unchanged, bit-identical.
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int o0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (o0 >= ff) return;
    int pairs = in_f / 32;
    size_t bank_row = (size_t)e * ff;
    const float* xrow = x;
    if (tok_map != 0) {
        const int* tm = reinterpret_cast<const int*>((size_t)tok_map);
        xrow = x + (long)tm[slot] * x_tstride;
    }
    float gacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float uacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    q4e_gu_rows4(gcodes, gscales, e2m1, bank_row, o0, in_f, xrow, pairs, lane, gacc);
    q4e_gu_rows4(ucodes, uscales, e2m1, bank_row, o0, in_f, xrow, pairs, lane, uacc);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            gacc[r] += __shfl_down_sync(0xffffffff, gacc[r], off);
            uacc[r] += __shfl_down_sync(0xffffffff, uacc[r], off);
        }
    }
    if (lane == 0) {
        float gm = gmacros[e];
        float um = umacros[e];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float g = gacc[r] * gm;
            float u = uacc[r] * um;
            act[(size_t)slot * ff + o0 + r] = g / (1.0f + expf(-g)) * u;
        }
    }
}

// ===================================================================== //
//  qwen4_exp sel matvec: SUB-WARP pair groups (downsel lane, mtp14)     //
// ===================================================================== //
// THE DEFECT. v3/gufuse partition the pair loop over all 32 lanes —
// `for (p = lane; p < pairs; p += 32)` with `pairs = in_f/32` — and then reduce with a
// full 5-step shfl tree. At this artifact's geometry that partition does not fill a warp:
//
//   down  (in_f = expert ff = 640):   pairs = 20  -> lanes 20-31 hold NO pair for the
//                                     whole kernel (62.5% lane occupancy), and each
//                                     active lane runs exactly ONE loop iteration.
//   gate+up (in_f = hidden = 2560):   pairs = 80  -> ceil(80/32) = 3 warp iterations for
//                                     80 lane-iterations of work (83.3% occupancy: a
//                                     3-vs-2 tail).
//
// KNEE:q4e-sel-slots-not-bytes measured the consequence: the section scales with SLOT
// COUNT (10 -> 60 slots costs 4.13x at fixed bytes) and barely at all with weight traffic
// (a 6x distinct-byte cut buys 1.101x), i.e. it is per-slot-work bound, and per-slot work
// is what an idle lane wastes.
//
// THE SHAPE. The pair loop becomes a SUB-WARP of `g` lanes. The warp carries `32/g`
// groups; group `gi` owns `ROWS` consecutive output rows starting at
// `o0 + gi*ROWS`; lane `s = lane & (g-1)` walks `p = s, s+g, s+2g, ...`; the reduce is
// log2(g) `shfl_down` steps INSIDE the group and the group's lane `s == 0` writes. Rows
// per warp is therefore `(32/g) * ROWS` and the launcher tiles `out_f` by it.
//
// `g == 32` with `ROWS == 4` is the v3 / gufuse program EXACTLY — same per-lane pair set
// (`p = lane; p += 32`), same 5-step tree (`off = 16 -> 1`), same write lane, same
// per-row expression tree. That arm is BYTE-COMPARED to the shipped kernels in
// `gate_nvfp4_sel_matvec`, which is what makes this kernel a strict generalization and
// the seam a safe rollback.
//
// Every other `g` changes the ORDER the pairs are summed in (a lane now chains several
// pairs into its accumulator, and the tree is shallower), so it is an ACCUMULATION-CLASS
// change like v3 was against v2 — gated against the host decoder chain on tolerance, not
// against v3 on bits.
//
// `g` must be a power of two in [1, 32]; the launcher enforces that and the exact tiling.
// A group whose rows fall past `out_f` still runs the reduce with zeroed accumulators
// rather than returning early: groups inside ONE warp have different `o0`, so an early
// return would leave `__shfl_down_sync(0xffffffff, ...)` with an incomplete mask.

// Accumulate `ROWS` consecutive rows of ONE bank into `acc[]`, pairs strided by `g` from
// `s`. `q4e_gu_rows4`'s body VERBATIM per row (same LUT extracts, same
// `acc += scale*group_dot` chaining) — only the p-partition moved.
template <int ROWS>
__device__ __forceinline__ void q4e_sel_rows_g(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* e2m1, size_t bank_row_off, int o0, int in_f,
        const float* __restrict__ xrow, int pairs, int s, int g, float* acc /*[ROWS]*/) {
    size_t row_codes = (size_t)in_f / 2;
    size_t row_scales = (size_t)in_f / 16;
    const uint4* c[ROWS];
    const unsigned short* sc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        c[r] = reinterpret_cast<const uint4*>(codes + (bank_row_off + o0 + r) * row_codes);
        // row_scales is even (in_f % 32 == 0), so every scale row starts u16-aligned: one
        // 2-byte load fetches both group scales of a uint4's worth of codes.
        sc[r] = reinterpret_cast<const unsigned short*>(
            scales + (bank_row_off + o0 + r) * row_scales);
    }
    for (int p = s; p < pairs; p += g) {
        const float* xp = xrow + (size_t)p * 32;
        uint4 cw[ROWS];
        unsigned short sw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            cw[r] = c[r][p];
            sw[r] = sc[r][p];
        }
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float d0 = q4e_dot8(e2m1, cw[r].x, xp) + q4e_dot8(e2m1, cw[r].y, xp + 8);
            float d1 = q4e_dot8(e2m1, cw[r].z, xp + 16) + q4e_dot8(e2m1, cw[r].w, xp + 24);
            acc[r] += q4e_ue4m3((unsigned char)(sw[r] & 0xFF)) * d0
                    + q4e_ue4m3((unsigned char)(sw[r] >> 8)) * d1;
        }
    }
}

// Sub-warp reduce: log2(g) shfl_down steps. Lanes at `s >= g/2` read across the group
// boundary and their results are discarded (only `s == 0` is read), which is the standard
// butterfly-down: at every step the lanes that still matter read in-group partials.
template <int ROWS>
__device__ __forceinline__ void q4e_sel_reduce_g(float* acc /*[ROWS]*/, int g) {
    for (int off = g >> 1; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
        }
    }
}

template <int ROWS>
__device__ __forceinline__ void q4e_sel_g_body(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride, int g) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int s = lane & (g - 1);
    int gi = lane >> (__ffs(g) - 1);
    int rows_per_warp = (32 / g) * ROWS;
    int o0 = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * rows_per_warp + gi * ROWS;
    int slot = blockIdx.y;
    int e = sel[slot];
    const float* xrow = x + (size_t)slot * (size_t)x_stride;
    int pairs = in_f / 32;
    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;
    bool live = (o0 + ROWS <= out_f);
    if (live) {
        q4e_sel_rows_g<ROWS>(codes, scales, e2m1, (size_t)e * out_f, o0, in_f, xrow, pairs,
                             s, g, acc);
    }
    q4e_sel_reduce_g<ROWS>(acc, g);
    if (live && s == 0) {
        float m = macros[e];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) y[(size_t)slot * out_f + o0 + r] = acc[r] * m;
    }
}

// grid (out_f / rows_per_warp, n_sel), block 32*warps. `g` power of two in [1,32];
// `rows` in {1,2,4}. (g=32, rows=4) is qmatvec_nvfp4_modelopt_sel_f32_v3 verbatim.
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_g_f32(
        const unsigned char* __restrict__ codes, const unsigned char* __restrict__ scales,
        const float* __restrict__ macros, const int* __restrict__ sel,
        const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, long x_stride, int g, int rows) {
    // Grid-uniform branch (no divergence): the launcher picks one (g, rows) per launch.
    if (rows == 4) {
        q4e_sel_g_body<4>(codes, scales, macros, sel, x, y, in_f, out_f, x_stride, g);
    } else if (rows == 2) {
        q4e_sel_g_body<2>(codes, scales, macros, sel, x, y, in_f, out_f, x_stride, g);
    } else {
        q4e_sel_g_body<1>(codes, scales, macros, sel, x, y, in_f, out_f, x_stride, g);
    }
}

template <int ROWS>
__device__ __forceinline__ void q4e_gu_g_body(
        const unsigned char* __restrict__ gcodes, const unsigned char* __restrict__ gscales,
        const float* __restrict__ gmacros,
        const unsigned char* __restrict__ ucodes, const unsigned char* __restrict__ uscales,
        const float* __restrict__ umacros,
        const int* __restrict__ sel, unsigned long long pack, int max_sel,
        const float* __restrict__ x, float* __restrict__ act, int in_f, int ff,
        unsigned long long tok_map, long x_tstride, int g) {
    const float e2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    int slot = blockIdx.y;
    int e;
    if (pack != 0) {
        const int* meta = reinterpret_cast<const int*>((size_t)pack);
        if (slot >= meta[2 * max_sel]) return; // live count (count-gated TP2 twin)
        e = meta[slot];
    } else {
        e = sel[slot];
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int s = lane & (g - 1);
    int gi = lane >> (__ffs(g) - 1);
    int rows_per_warp = (32 / g) * ROWS;
    int o0 = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * rows_per_warp + gi * ROWS;
    int pairs = in_f / 32;
    size_t bank_row = (size_t)e * ff;
    const float* xrow = x;
    if (tok_map != 0) {
        const int* tm = reinterpret_cast<const int*>((size_t)tok_map);
        xrow = x + (long)tm[slot] * x_tstride;
    }
    float gacc[ROWS];
    float uacc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) { gacc[r] = 0.0f; uacc[r] = 0.0f; }
    bool live = (o0 + ROWS <= ff);
    if (live) {
        q4e_sel_rows_g<ROWS>(gcodes, gscales, e2m1, bank_row, o0, in_f, xrow, pairs, s, g,
                             gacc);
        q4e_sel_rows_g<ROWS>(ucodes, uscales, e2m1, bank_row, o0, in_f, xrow, pairs, s, g,
                             uacc);
    }
    // Interleaving gate/up inside the tree is free for bit-identity: each accumulator's
    // own addition sequence is what fixes its bits, and that sequence is unchanged.
    for (int off = g >> 1; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            gacc[r] += __shfl_down_sync(0xffffffff, gacc[r], off);
            uacc[r] += __shfl_down_sync(0xffffffff, uacc[r], off);
        }
    }
    if (live && s == 0) {
        float gm = gmacros[e];
        float um = umacros[e];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            float gv = gacc[r] * gm;
            float uv = uacc[r] * um;
            act[(size_t)slot * ff + o0 + r] = gv / (1.0f + expf(-gv)) * uv;
        }
    }
}

// grid (ff / rows_per_warp, n_sel), block 32*warps. (g=32, rows=4) is
// qmatvec_nvfp4_modelopt_sel_gu_silu_f32 verbatim, pack and tok_map modes included.
extern "C" __global__ void qmatvec_nvfp4_modelopt_sel_gu_silu_g_f32(
        const unsigned char* __restrict__ gcodes, const unsigned char* __restrict__ gscales,
        const float* __restrict__ gmacros,
        const unsigned char* __restrict__ ucodes, const unsigned char* __restrict__ uscales,
        const float* __restrict__ umacros,
        const int* __restrict__ sel, unsigned long long pack, int max_sel,
        const float* __restrict__ x, float* __restrict__ act, int in_f, int ff,
        unsigned long long tok_map, long x_tstride, int g, int rows) {
    if (rows == 4) {
        q4e_gu_g_body<4>(gcodes, gscales, gmacros, ucodes, uscales, umacros, sel, pack,
                         max_sel, x, act, in_f, ff, tok_map, x_tstride, g);
    } else if (rows == 2) {
        q4e_gu_g_body<2>(gcodes, gscales, gmacros, ucodes, uscales, umacros, sel, pack,
                         max_sel, x, act, in_f, ff, tok_map, x_tstride, g);
    } else {
        q4e_gu_g_body<1>(gcodes, gscales, gmacros, ucodes, uscales, umacros, sel, pack,
                         max_sel, x, act, in_f, ff, tok_map, x_tstride, g);
    }
}

// ===================================================================== //
//  qwen4_exp quantized QSA KV cache (kvq lane, 2026-08-31)              //
// ===================================================================== //
// Owner default K=q8_0 / V=q5_1 — asymmetric because K feeds the score dots + rope
// (symmetric 8-bit keeps dot precision) while V errors average under the attention
// weighting (affine 5-bit suffices). Formats are HARDCODED here (not the flash fatbin's
// MEMRA_KV_K/V macro matrix): the qwen4_exp KV default is a per-family owner decision
// and must not follow the env-selected fatbin variant. The quantize warp programs are
// flash_attn.cu's validated baseline appenders VERBATIM (quant_K_block / quant_V_block
// #else arms), so the cache byte layout matches that lane's oracle history; layout
// reference also cross-checked against llama.cpp mainline's quantized QSA-KV path
// (src/models/qwen4exp.cpp) — same q8_0 34B / q5_1 24B blocks, token-major rows.
// Dequant helpers use EXPLICIT IEEE intrinsics (the qsa_index_score 1-ULP FMA lesson):
// both the row-dequant kernel and the in-attention reads must produce the same bits.

static __device__ __forceinline__ float q4e_warp_amax(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
static __device__ __forceinline__ float q4e_warp_min(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
static __device__ __forceinline__ float q4e_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// q8_0 block: { fp16 d; int8 qs[32] } = 34 B / 32 elems. Whole warp participates;
// `x` is this lane's element (caller zero-pads past the row width).
// ONE deliberate divergence from the flash program: a subnormal-amax block makes
// 1/d overflow to +inf, where device lrintf(inf) is UNDEFINED while the host twin
// saturates — the `isfinite` guard zeroes the block on BOTH sides instead, making the
// host/device quantize contract TOTAL (these caches never interop with flash caches,
// and a subnormal-amax K/raw-key row is numerically dead anyway).
static __device__ __forceinline__ void q4e_quant_q8_block(float x, int lane, uint8_t* __restrict__ blk) {
    float amax = q4e_warp_amax(x);
    float d = amax / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    if (!isfinite(id)) id = 0.0f;
    int q = (int)lrintf(x * id);
    q = max(-127, min(127, q));
    if (lane == 0) *(half*)blk = __float2half(d);
    ((int8_t*)(blk + 2))[lane] = (int8_t)q;
}

// q5_1 block: { fp16 d; fp16 m; u32 qh; u8 qs[16] } = 24 B / 32 elems. Same subnormal
// guard as the q8 arm.
static __device__ __forceinline__ void q4e_quant_q5_block(float x, int lane, uint8_t* __restrict__ blk) {
    float mn = q4e_warp_min(x);
    float mx = q4e_warp_max(x);
    float d = (mx - mn) / 31.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    if (!isfinite(id)) id = 0.0f;
    int q5 = (int)lrintf((x - mn) * id);
    q5 = max(0, min(31, q5));
    uint32_t qh = __ballot_sync(0xffffffffu, (q5 >> 4) & 1);
    if (lane == 0) {
        *(half*)blk        = __float2half(d);
        *(half*)(blk + 2)  = __float2half(mn);
        *(uint32_t*)(blk + 4) = qh;
    }
    uint8_t* qs = blk + 8;
    int nib = q5 & 0x0F;
    int partner_nib = __shfl_sync(0xffffffffu, nib, lane + 16) & 0x0F;
    if (lane < 16) qs[lane] = (uint8_t)(nib | (partner_nib << 4));
}

// Element dequant, EXPLICIT intrinsics (shared by the row-dequant kernel and the
// block-list attention reads — one helper so both consumers produce identical bits).
static __device__ __forceinline__ float q4e_deq_q8(const uint8_t* __restrict__ row, int e) {
    const uint8_t* blk = row + (size_t)(e >> 5) * 34;
    const float d = __half2float(*(const half*)blk);
    return __fmul_rn(d, (float)((const int8_t*)(blk + 2))[e & 31]);
}
static __device__ __forceinline__ float q4e_deq_q5(const uint8_t* __restrict__ row, int e) {
    const uint8_t* blk = row + (size_t)(e >> 5) * 24;
    const float d = __half2float(*(const half*)blk);
    const float m = __half2float(*(const half*)(blk + 2));
    const uint32_t qh = *(const uint32_t*)(blk + 4);
    const uint8_t* qs = blk + 8;
    const int lane = e & 31;
    const int lo = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
    const int q5 = lo | (int)(((qh >> lane) & 1u) << 4);
    return __fmaf_rn(d, (float)q5, m);
}

// Append-quantize T token rows of post-RoPE K (q8_0) and V (q5_1) into the resident
// byte caches at slots [t0, t0+T). grid = (max_blocks, T), block = (32,1,1); one warp
// per 32-elem block, zero-pad past the row width (the flash rows-appender program).
extern "C" __global__ void q4e_kv_append_q8q5_rows(
        const float* __restrict__ k_rows,  // [T, kv_dim_k] token-major
        const float* __restrict__ v_rows,  // [T, kv_dim_v]
        uint8_t* __restrict__ K, uint8_t* __restrict__ V,
        int t0, int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes) {
    const int b = blockIdx.x;
    const int tt = blockIdx.y;
    const int lane = threadIdx.x;
    const int eidx = b * 32 + lane;
    const int t = t0 + tt;
    if (b * 32 < kv_dim_k) {
        float x = (eidx < kv_dim_k) ? k_rows[(size_t)tt * kv_dim_k + eidx] : 0.0f;
        q4e_quant_q8_block(x, lane, K + (size_t)t * k_tok_bytes + (size_t)b * 34);
    }
    if (b * 32 < kv_dim_v) {
        float x = (eidx < kv_dim_v) ? v_rows[(size_t)tt * kv_dim_v + eidx] : 0.0f;
        q4e_quant_q5_block(x, lane, V + (size_t)t * v_tok_bytes + (size_t)b * 24);
    }
}

// Dequant cache rows [r0, r0+rows) into f32 buffers (buffer row i = cache row r0+i).
// The exactness seam for gates and the TP2 migration path; grid = (max_blocks, rows).
extern "C" __global__ void q4e_kv_dequant_rows(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ k_out, float* __restrict__ v_out,
        int r0, int kv_dim_k, int kv_dim_v,
        long k_tok_bytes, long v_tok_bytes) {
    const int b = blockIdx.x;
    const int rr = blockIdx.y;
    const int lane = threadIdx.x;
    const int eidx = b * 32 + lane;
    const int r = r0 + rr;
    if (eidx < kv_dim_k)
        k_out[(size_t)rr * kv_dim_k + eidx] = q4e_deq_q8(K + (size_t)r * k_tok_bytes, eidx);
    if (eidx < kv_dim_v)
        v_out[(size_t)rr * kv_dim_v + eidx] = q4e_deq_q5(V + (size_t)r * v_tok_bytes, eidx);
}

// Block-list QSA attention over the QUANTIZED cache: sdpa_blocklist_f32's program with
// the K/V row reads dequanted in place (q4e_deq_q8 / q4e_deq_q5 — the same helpers as
// the row-dequant kernel, so this kernel is gated BIT-IDENTICAL to the composition
// "q4e_kv_dequant_rows then sdpa_blocklist_f32"). Same phases: per-position dots
// (parallel over selected entries), single-thread max/exp/normalize in selection order,
// per-dim weighted V over selected entries ascending; smem scales with the bounded
// selection, never with T_kv.
extern "C" __global__ void q4e_sdpa_blocklist_q8q5(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ O,
        const int* __restrict__ pos_list, const int* __restrict__ row_meta,
        int head_dim, int n_head, int n_head_kv, int T, int max_count, float scale,
        long k_tok_bytes, long v_tok_bytes) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float smem_raw[];
    int* spos = (int*)smem_raw;            // [max_count]
    float* scores = smem_raw + max_count;  // [max_count]

    int off = row_meta[2 * qt];
    int count = row_meta[2 * qt + 1];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;

    for (int i = tid; i < count; i += blockDim.x) {
        int t = pos_list[off + i];
        spos[i] = t;
        const uint8_t* krow = K + (size_t)t * k_tok_bytes;
        const int e0 = kv_head * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q[d] * q4e_deq_q8(krow, e0 + d);
        scores[i] = acc * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int i = 0; i < count; i++) mx = fmaxf(mx, scores[i]);
        float sum = 0.0f;
        for (int i = 0; i < count; i++) { float e = expf(scores[i] - mx); scores[i] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int i = 0; i < count; i++) scores[i] *= inv;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int e = kv_head * head_dim + d;
        float acc = 0.0f;
        for (int i = 0; i < count; i++)
            acc += scores[i] * q4e_deq_q5(V + (size_t)spos[i] * v_tok_bytes, e);
        o[d] = acc;
    }
}

// ---- `q4e_sdpa_blocklist_q8q5` with the K block scale HOISTED (the `kvhoist` read seam). ----
// BIT-IDENTICAL to the kernel above by construction: same `q[d] * __fmul_rn(d_scale, (float)qi)`
// product, same `acc +=` in the same ascending-d order, same phase 2 and phase 3 code. The only
// change is WHERE the fp16 block scale is loaded from.
//
// Why it exists. `q4e_deq_q8` recomputes `blk` from the element index, so the score loop reloads
// the block's fp16 scale ONCE PER ELEMENT — 32 redundant loads per 32-element block, 2 load
// instructions per element instead of 1. That is the whole measured kvq-at-depth penalty: the
// f32 twin's score loop issues `head_dim` loads per selected position and this one issues
// `2*head_dim`, and because phase 1 is thread-per-position (lanes on 32 different tokens,
// `k_tok_bytes` apart) EVERY load instruction replays 32 ways into 32 distinct sectors. The
// quantized cache reads ~3.8x fewer BYTES than f32 and issues 2x MORE transactions, so the byte
// saving never lands and the extra instruction stream is a straight loss. Hoisting the scale to
// once per block takes phase 1 from 2*head_dim to head_dim + head_dim/32 loads per position.
//
// `head_dim % 32 == 0` and `e0 = kv_head*head_dim` mean the inner run is always a full 32-element
// block, but the code carries the partial-run arithmetic anyway: a geometry where it does not hold
// would otherwise read a neighbouring block's scale, which is a silent wrong-value bug rather than
// a crash. Real-geometry oracle vs the un-hoisted kernel is the gate (tiny geometry would not
// exercise a second block at all — the lane has been bitten twice by tiny-green/real-broken).
extern "C" __global__ void q4e_sdpa_blocklist_q8q5_hoist(
        const float* __restrict__ Q,
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        float* __restrict__ O,
        const int* __restrict__ pos_list, const int* __restrict__ row_meta,
        int head_dim, int n_head, int n_head_kv, int T, int max_count, float scale,
        long k_tok_bytes, long v_tok_bytes) {
    int head = blockIdx.x;
    int qt = blockIdx.y;
    if (head >= n_head || qt >= T) return;
    int kv_head = head / (n_head / n_head_kv);
    int tid = threadIdx.x;
    extern __shared__ float smem_raw[];
    int* spos = (int*)smem_raw;            // [max_count]
    float* scores = smem_raw + max_count;  // [max_count]

    int off = row_meta[2 * qt];
    int count = row_meta[2 * qt + 1];
    const float* q = Q + ((size_t)qt * n_head + head) * head_dim;

    for (int i = tid; i < count; i += blockDim.x) {
        int t = pos_list[off + i];
        spos[i] = t;
        const uint8_t* krow = K + (size_t)t * k_tok_bytes;
        const int e0 = kv_head * head_dim;
        float acc = 0.0f;
        int d = 0;
        while (d < head_dim) {
            const int e = e0 + d;
            const uint8_t* blk = krow + (size_t)(e >> 5) * 34;
            const float dsc = __half2float(*(const half*)blk);
            const int8_t* qs = (const int8_t*)(blk + 2);
            const int lane0 = e & 31;
            int n = 32 - lane0;
            if (n > head_dim - d) n = head_dim - d;
            for (int j = 0; j < n; j++)
                acc += q[d + j] * __fmul_rn(dsc, (float)qs[lane0 + j]);
            d += n;
        }
        scores[i] = acc * scale;
    }
    __syncthreads();
    if (tid == 0) {
        float mx = -1e30f;
        for (int i = 0; i < count; i++) mx = fmaxf(mx, scores[i]);
        float sum = 0.0f;
        for (int i = 0; i < count; i++) { float e = expf(scores[i] - mx); scores[i] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int i = 0; i < count; i++) scores[i] *= inv;
    }
    __syncthreads();
    float* o = O + ((size_t)qt * n_head + head) * head_dim;
    // Phase 3 is the base kernel's phase 3 VERBATIM — `kvhoist` is deliberately a ONE-VARIABLE
    // seam. The same hoist was tried here (block offset / packed-byte index / nibble shift are
    // loop invariants `q4e_deq_q5` recomputes once per selected position) and the SASS said no:
    // per position the base loop issues {1 U16, 1 U8, 0.5 LDG.E} and the hoisted form issued
    // {1 U8, 2 LDG.E} — the base already lets the compiler merge `d` and `m` into one 32-bit
    // load, and spelling the reads out separately cost that merge. Phase 3 is also already
    // coalesced (adjacent lanes take adjacent `e`, so a warp covers one 24-byte block in one
    // sector), so there was no transaction win to buy. Receipted dead end; do not re-try without
    // reading `research/qwen4exp-bringup-20260829/perf/PROFILE-C0.md` §2 first.
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int e = kv_head * head_dim + d;
        float acc = 0.0f;
        for (int i = 0; i < count; i++)
            acc += scores[i] * q4e_deq_q5(V + (size_t)spos[i] * v_tok_bytes, e);
        o[d] = acc;
    }
}

// ---- Indexer raw-key cache, quantized arms (idxq lane) ----
// copy_rows_col_f32's shape with a quantize epilogue: append the k-part columns of the
// indexer projection rows to the DEVICE raw-key cache as q8_0 blocks (rows of
// ceil(width/32) x 34 B) or bf16 (RNE). The host cache materializes from these bytes
// verbatim (dtoh, no re-quant), so host/device interleavings stay bit-identical as long
// as the HOST quantize twin matches these warp programs (pinned by the tiny gate).
extern "C" __global__ void q4e_idx_append_q8(
        const float* __restrict__ src, uint8_t* __restrict__ dst,
        int rows, int width, long src_stride, long src_col, long dst_row) {
    const int b = blockIdx.x;          // 32-elem block within the row
    const int r = blockIdx.y;          // row within this chunk
    const int lane = threadIdx.x;
    const int eidx = b * 32 + lane;
    if (r >= rows || b * 32 >= width) return;
    const long row_bytes = ((width + 31) / 32) * 34;
    float x = (eidx < width) ? src[(size_t)r * src_stride + src_col + eidx] : 0.0f;
    q4e_quant_q8_block(x, lane, dst + (size_t)(dst_row + r) * row_bytes + (size_t)b * 34);
}

extern "C" __global__ void q4e_idx_append_bf16(
        const float* __restrict__ src, uint16_t* __restrict__ dst,
        int rows, int width, long src_stride, long src_col, long dst_row) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * width;
    if (i >= total) return;
    long r = i / width;
    long c = i - r * width;
    float x = src[r * src_stride + src_col + c];
    dst[(dst_row + r) * (long)width + c] = __bfloat16_as_ushort(__float2bfloat16(x));
}
