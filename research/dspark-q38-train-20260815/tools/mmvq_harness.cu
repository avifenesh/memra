// e4m3-blk MMVQ decode-kernel harness (memra sm_100a lane, 2026-08-15).
// Measures qmatvec_e4m3_blk_mmvq (verbatim engine copy, VAR=0) and optimization variants at the
// 27B decode shapes, GB/s of weight traffic, with an exactness check vs the verbatim kernel.
//   VAR 0: engine kernel verbatim (baseline)
//   VAR 1: ROWS=4 -> 2 warps y (occupancy probe: 8 rows/CTA)
//   VAR 2: manual 2-deep software pipeline (prefetch next blk's W+A while converting current)
//   VAR 3: VAR2 + __ldcs streaming weight loads (L2 bypass hint; weights are read-once at m=1)
// Usage: harness <var> [in_f] [out_f]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
#define ROWS 4

__device__ __forceinline__ float2 e4m3x2_to_f32x2(unsigned short w2) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)w2, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&hr));
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, off);
    return v;
}

// ---------- VAR 0: engine verbatim ----------
__device__ __forceinline__ float e4m3_blk_row_dot(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, const float* __restrict__ srow,
        int nblk, int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            int a = au[k];
            bs = fmaf(wlo.x, (float)(signed char)(a & 0xff), bs);
            bs = fmaf(wlo.y, (float)(signed char)((a >> 8) & 0xff), bs);
            bs = fmaf(whi.x, (float)(signed char)((a >> 16) & 0xff), bs);
            bs = fmaf(whi.y, (float)(a >> 24), bs);
        }
        acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
    }
    return acc;
}
extern "C" __global__ void mmvq_v0(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_blk_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                                 ad + (size_t)t * nblk,
                                 blk_scales + (size_t)(o >> 7) * scale_cols, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ---------- VAR 1: 8 rows per CTA ----------
extern "C" __global__ void mmvq_v1(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * 8 + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_blk_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                                 ad + (size_t)t * nblk,
                                 blk_scales + (size_t)(o >> 7) * scale_cols, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ---------- VAR 2/3: 2-deep software pipeline (+ optional streaming loads) ----------
template <bool STREAM>
__device__ __forceinline__ float e4m3_blk_row_dot_pipe(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, const float* __restrict__ srow,
        int nblk, int lane) {
    float acc = 0.0f;
    int blk = lane;
    if (blk >= nblk) return 0.0f;
    uint4 w01, w23; int4 a01, a23;
    auto load = [&](int b) {
        const uint4* w16 = (const uint4*)(wrow + b * 32);
        const int4* aq16 = (const int4*)(arow + b * 32);
        if (STREAM) { w01 = __ldcs(&w16[0]); w23 = __ldcs(&w16[1]); }
        else        { w01 = w16[0];          w23 = w16[1]; }
        a01 = aq16[0]; a23 = aq16[1];
    };
    load(blk);
    for (; blk < nblk; ) {
        uint4 cw01 = w01, cw23 = w23; int4 ca01 = a01, ca23 = a23;
        const int cur = blk;
        blk += 32;
        if (blk < nblk) load(blk);          // prefetch next while converting current
        unsigned wu[8] = { cw01.x, cw01.y, cw01.z, cw01.w, cw23.x, cw23.y, cw23.z, cw23.w };
        int au[8] = { ca01.x, ca01.y, ca01.z, ca01.w, ca23.x, ca23.y, ca23.z, ca23.w };
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            int a = au[k];
            bs = fmaf(wlo.x, (float)(signed char)(a & 0xff), bs);
            bs = fmaf(wlo.y, (float)(signed char)((a >> 8) & 0xff), bs);
            bs = fmaf(whi.x, (float)(signed char)((a >> 16) & 0xff), bs);
            bs = fmaf(whi.y, (float)(a >> 24), bs);
        }
        acc = fmaf(srow[cur >> 2] * adrow[cur], bs, acc);
    }
    return acc;
}
template <bool STREAM>
__device__ __forceinline__ void mmvq_pipe(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_blk_row_dot_pipe<STREAM>(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                                              ad + (size_t)t * nblk,
                                              blk_scales + (size_t)(o >> 7) * scale_cols, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
extern "C" __global__ void mmvq_v2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    mmvq_pipe<false>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}
extern "C" __global__ void mmvq_v3(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    mmvq_pipe<true>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}


// ---------- VAR 9: pure-read ceiling for this access pattern (no convert, no act) ----------
extern "C" __global__ void mmvq_v9(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    unsigned acc = 0;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        acc += w01.x + w01.y + w01.z + w01.w + w23.x + w23.y + w23.z + w23.w;
    }
    acc += __shfl_down_sync(0xFFFFFFFF, acc, 16);
    if (lane == 0) y[o] = (float)acc;
}
// ---------- VAR 4: 64B-per-lane block-pair walk (NEW reduction order: pair-serial) ----------
extern "C" __global__ void mmvq_v4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    const float* srow = blk_scales + (size_t)(o >> 7) * scale_cols;
    float acc = 0.0f;
    for (int p = lane * 2; p < nblk; p += 64) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            int blk = p + h;
            if (blk >= nblk) break;
            const uint4* w16 = (const uint4*)(wrow + blk * 32);
            uint4 w01 = w16[0], w23 = w16[1];
            unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float bs = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
                float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
                int a = au[k];
                bs = fmaf(wlo.x, (float)(signed char)(a & 0xff), bs);
                bs = fmaf(wlo.y, (float)(signed char)((a >> 8) & 0xff), bs);
                bs = fmaf(whi.x, (float)(signed char)((a >> 16) & 0xff), bs);
                bs = fmaf(whi.y, (float)(a >> 24), bs);
            }
            acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}


// ---------- VAR 5: act row converted ONCE to f32 in smem, shared by the CTA's 4 warps ----------
// Same values, same per-lane fmaf chain (lane-strided blk walk unchanged across k-tiles), so the
// output is BIT-IDENTICAL to the engine kernel. Kills 4 I2F + 4 byte-extracts per 4 weights.
#define V5_TILE 4096
extern "C" __global__ void mmvq_v5(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    __shared__ float sa[V5_TILE];
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    int lane = threadIdx.x;
    int tid = threadIdx.y * 32 + lane;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    const float* srow = blk_scales + (size_t)(min(o, out_f - 1) >> 7) * scale_cols;
    float acc = 0.0f;
    for (int k0 = 0; k0 < in_f; k0 += V5_TILE) {
        const int kn = min(V5_TILE, in_f - k0);
        __syncthreads();
        for (int i = tid * 4; i < kn; i += ROWS * 32 * 4) {
            const int a = *(const int*)(arow + k0 + i);
            sa[i + 0] = (float)(signed char)(a & 0xff);
            sa[i + 1] = (float)(signed char)((a >> 8) & 0xff);
            sa[i + 2] = (float)(signed char)((a >> 16) & 0xff);
            sa[i + 3] = (float)(a >> 24);
        }
        __syncthreads();
        if (o >= out_f || t >= m) continue;
        const int nb_t = kn / 32, b0 = k0 / 32;
        for (int bl = lane; bl < nb_t; bl += 32) {
            const int blk = b0 + bl;
            const uint4* w16 = (const uint4*)(wrow + blk * 32);
            uint4 w01 = w16[0], w23 = w16[1];
            unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            const float4* a4 = (const float4*)(sa + bl * 32);
            float bs = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
                float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
                float4 av = a4[k];
                bs = fmaf(wlo.x, av.x, bs);
                bs = fmaf(wlo.y, av.y, bs);
                bs = fmaf(whi.x, av.z, bs);
                bs = fmaf(whi.y, av.w, bs);
            }
            acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
        }
    }
    if (o >= out_f || t >= m) return;
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}


// ---------- ablation probes (NOT candidates; wrong math by design) ----------
// VAR 6: no act loads / no act I2F — constant act. Prices the act side.
// VAR 7: full loads, weight cvt replaced by integer mangle — prices the e4m3->f32 cvt chain.
extern "C" __global__ void mmvq_v6(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const float* adrow = ad + (size_t)t * nblk;
    const float* srow = blk_scales + (size_t)(o >> 7) * scale_cols;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            bs = fmaf(wlo.x, 3.0f, bs);
            bs = fmaf(wlo.y, 5.0f, bs);
            bs = fmaf(whi.x, 7.0f, bs);
            bs = fmaf(whi.y, 9.0f, bs);
        }
        acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
extern "C" __global__ void mmvq_v7(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    const float* srow = blk_scales + (size_t)(o >> 7) * scale_cols;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float wlo_x = (float)(int)(wu[k] & 0xFF);      // fake cvt: I2F instead of fp8 chain
            float wlo_y = (float)(int)((wu[k] >> 8) & 0xFF);
            float whi_x = (float)(int)((wu[k] >> 16) & 0xFF);
            float whi_y = (float)(int)(wu[k] >> 24);
            int a = au[k];
            bs = fmaf(wlo_x, (float)(signed char)(a & 0xff), bs);
            bs = fmaf(wlo_y, (float)(signed char)((a >> 8) & 0xff), bs);
            bs = fmaf(whi_x, (float)(signed char)((a >> 16) & 0xff), bs);
            bs = fmaf(whi_y, (float)(a >> 24), bs);
        }
        acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}


// ---------- VAR 10: act pre-converted to f32 in GLOBAL (one I2F per value per token, done
// upstream by the quantizer in the engine port). Same values, same fmaf order -> BIT-IDENTICAL.
extern "C" __global__ void mmvq_v10(
        const unsigned char* __restrict__ W, const float* __restrict__ aqf,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const float* arow = aqf + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    const float* srow = blk_scales + (size_t)(o >> 7) * scale_cols;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        const float4* a4 = (const float4*)(arow + blk * 32);
        float bs = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            float4 av = a4[k];
            bs = fmaf(wlo.x, av.x, bs);
            bs = fmaf(wlo.y, av.y, bs);
            bs = fmaf(whi.x, av.z, bs);
            bs = fmaf(whi.y, av.w, bs);
        }
        acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

int main(int argc, char** argv) {
    const int var = argc > 1 ? atoi(argv[1]) : 0;
    const int in_f = argc > 2 ? atoi(argv[2]) : 5120;
    const int out_f = argc > 3 ? atoi(argv[3]) : 12288;
    const int m = 1, scale_cols = (in_f + 127) / 128, nblk = in_f / 32;
    const size_t wb = (size_t) out_f * in_f;
    unsigned char* hW = (unsigned char*) malloc(wb);
    signed char* hA = (signed char*) malloc(in_f);
    float* hAd = (float*) malloc(nblk * 4);
    float* hS = (float*) malloc((size_t)((out_f + 127) / 128) * scale_cols * 4);
    srand(7);
    for (size_t i = 0; i < wb; i++) { unsigned char v = rand() & 0xFF; if ((v & 0x7F) == 0x7F) v &= 0x77; hW[i] = v; }
    for (int i = 0; i < in_f; i++) hA[i] = (signed char)(rand() % 255 - 127);
    for (int i = 0; i < nblk; i++) hAd[i] = 0.001f + (rand() % 100) * 1e-4f;
    for (int i = 0; i < ((out_f + 127) / 128) * scale_cols; i++) hS[i] = 0.01f + (rand() % 100) * 1e-3f;
    float* hAf = (float*) malloc(in_f * 4);
    for (int i = 0; i < in_f; i++) hAf[i] = (float) hA[i];
    float* dAf;
    unsigned char* dW; signed char* dA; float *dAd, *dS, *dY, *dY0;
    CK(cudaMalloc(&dW, wb)); CK(cudaMalloc(&dA, in_f)); CK(cudaMalloc(&dAd, nblk * 4));
    CK(cudaMalloc(&dAf, in_f * 4)); CK(cudaMemcpy(dAf, hAf, in_f * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dS, (size_t)((out_f + 127) / 128) * scale_cols * 4));
    CK(cudaMalloc(&dY, out_f * 4)); CK(cudaMalloc(&dY0, out_f * 4));
    CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dA, hA, in_f, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAd, hAd, nblk * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, hS, (size_t)((out_f + 127) / 128) * scale_cols * 4, cudaMemcpyHostToDevice));
    dim3 blk3(32, var == 1 ? 8 : ROWS, 1);
    dim3 grid((out_f + blk3.y - 1) / blk3.y, m, 1);
    dim3 grid0((out_f + ROWS - 1) / ROWS, m, 1);
    auto launch = [&](int v, float* out) {
        if (v == 0) mmvq_v0<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 1) mmvq_v1<<<grid, blk3>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 2) mmvq_v2<<<grid, blk3>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 3) mmvq_v3<<<grid, blk3>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 4) mmvq_v4<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 5) mmvq_v5<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 6) mmvq_v6<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 7) mmvq_v7<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else if (v == 10) mmvq_v10<<<grid0, dim3(32, ROWS, 1)>>>(dW, dAf, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
        else mmvq_v9<<<grid0, dim3(32, ROWS, 1)>>>(dW, dA, dAd, dS, out, in_f, out_f, m, in_f, scale_cols);
    };
    // exactness vs verbatim
    launch(0, dY0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    launch(var, dY); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    float* hY = (float*) malloc(out_f * 4); float* hY0 = (float*) malloc(out_f * 4);
    CK(cudaMemcpy(hY, dY, out_f * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hY0, dY0, out_f * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < out_f; i++) {
        if (var == 9) break;
        if (var == 6 || var == 7) break;
        if (var <= 3 || var == 5 || var == 10) { if (hY[i] != hY0[i]) bad++; }
        else if (fabsf(hY[i] - hY0[i]) > 1e-4f * fabsf(hY0[i]) + 1e-5f) bad++;
    }
    // timing
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 20; i++) launch(var, dY);
    CK(cudaDeviceSynchronize());
    const int R = 200;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < R; i++) launch(var, dY);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    double us = ms * 1e3 / R, gbs = wb / (us * 1e-6) / 1e9;
    printf("var=%d %dx%d: %.2f us/launch  %.0f GB/s weight-traffic  bit_mismatch=%d/%d\n",
           var, in_f, out_f, us, gbs, bad, out_f);
    return bad != 0 && var != 0 ? 1 : 0;
}
