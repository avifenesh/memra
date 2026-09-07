// R8 dense FP8 GEMV candidate, CUDA 13.3 cubin source.
//
// This is deliberately the original 128-thread F32 GEMV program shape: one
// output row per block, each thread's serial product/add order unchanged, and
// the original 128-leaf shared reduction unchanged. The only replacement is
// FP8 decode. Two normalized E4M3 bytes go through native
// cvt.rn.bf16x2.e4m3x2, the exact BF16 words expand to f32, and the original
// f32 scale multiply and BF16-input dot product order follows.

#include <cuda_runtime.h>

#include <cstdint>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

static __device__ __forceinline__ uint8_t normalize_e4m3(uint8_t code) {
    const uint8_t mag = code & 0x7fu;
    return (mag == 0u || mag == 0x7fu) ? 0u : code;
}

static __device__ __forceinline__ uint32_t native_e4m3x2_to_bf16x2(uint16_t packed) {
    uint32_t out = 0;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out) : "h"(packed));
    return out;
}

static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

static __device__ __forceinline__ void add_pair(
        float& part, uint32_t bf16x2, float scale, uint32_t xword) {
    const float w0 = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2)), scale);
    const float w1 = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2 >> 16)), scale);
    const float x0 = __uint_as_float((xword & 0xffffu) << 16);
    const float x1 = __uint_as_float(xword & 0xffff0000u);
    part = __fadd_rn(part, __fmul_rn(w0, x0));
    part = __fadd_rn(part, __fmul_rn(w1, x1));
}

extern "C" __global__ void r8_gemv_fp8_native(
        const uint8_t* __restrict__ w, const float* __restrict__ sc, int sc_cols,
        const uint16_t* __restrict__ x, float* __restrict__ y, int n, int k) {
    const int row = blockIdx.x;
    if (row >= n) return;
    const uint8_t* wr = w + static_cast<long>(row) * k;
    const float* srow = sc + static_cast<long>(row >> 7) * sc_cols;
    float part = 0.0f;
    const int stride = blockDim.x * 8;
    int i0 = threadIdx.x * 8;
    for (; i0 + stride < k; i0 += 2 * stride) {
        const int i1 = i0 + stride;
        const uint2 wva = *reinterpret_cast<const uint2*>(wr + i0);
        const uint2 wvb = *reinterpret_cast<const uint2*>(wr + i1);
        const uint4 xva = *reinterpret_cast<const uint4*>(x + i0);
        const uint4 xvb = *reinterpret_cast<const uint4*>(x + i1);
        const uint32_t wba[2] = {wva.x, wva.y};
        const uint32_t wbb[2] = {wvb.x, wvb.y};
        const uint32_t xwa[4] = {xva.x, xva.y, xva.z, xva.w};
        const uint32_t xwb[4] = {xvb.x, xvb.y, xvb.z, xvb.w};
        const float sa = srow[i0 >> 7];
        const float sb = srow[i1 >> 7];
        float wua[8], wub[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint16_t raw_a = static_cast<uint16_t>(
                (wba[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const uint16_t normalized_a = static_cast<uint16_t>(
                (static_cast<uint16_t>(normalize_e4m3(static_cast<uint8_t>(raw_a >> 8))) << 8)
                | normalize_e4m3(static_cast<uint8_t>(raw_a)));
            const uint16_t raw_b = static_cast<uint16_t>(
                (wbb[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const uint16_t normalized_b = static_cast<uint16_t>(
                (static_cast<uint16_t>(normalize_e4m3(static_cast<uint8_t>(raw_b >> 8))) << 8)
                | normalize_e4m3(static_cast<uint8_t>(raw_b)));
            const uint32_t bf16x2_a = native_e4m3x2_to_bf16x2(normalized_a);
            const uint32_t bf16x2_b = native_e4m3x2_to_bf16x2(normalized_b);
            wua[2 * j] = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2_a)), sa);
            wua[2 * j + 1] = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2_a >> 16)), sa);
            wub[2 * j] = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2_b)), sb);
            wub[2 * j + 1] = __fmul_rn(bf16_bits_to_f32(static_cast<uint16_t>(bf16x2_b >> 16)), sb);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float x0 = __uint_as_float((xwa[j] & 0xffffu) << 16);
            const float x1 = __uint_as_float(xwa[j] & 0xffff0000u);
            part = __fadd_rn(part, __fmul_rn(wua[2 * j], x0));
            part = __fadd_rn(part, __fmul_rn(wua[2 * j + 1], x1));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float x0 = __uint_as_float((xwb[j] & 0xffffu) << 16);
            const float x1 = __uint_as_float(xwb[j] & 0xffff0000u);
            part = __fadd_rn(part, __fmul_rn(wub[2 * j], x0));
            part = __fadd_rn(part, __fmul_rn(wub[2 * j + 1], x1));
        }
    }
    for (; i0 < k; i0 += stride) {
        const uint2 wv = *reinterpret_cast<const uint2*>(wr + i0);
        const uint4 xv = *reinterpret_cast<const uint4*>(x + i0);
        const uint32_t wb[2] = {wv.x, wv.y};
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
        const float scale = srow[i0 >> 7];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint16_t raw = static_cast<uint16_t>(
                (wb[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const uint16_t normalized = static_cast<uint16_t>(
                (static_cast<uint16_t>(normalize_e4m3(static_cast<uint8_t>(raw >> 8))) << 8)
                | normalize_e4m3(static_cast<uint8_t>(raw)));
            add_pair(part, native_e4m3x2_to_bf16x2(normalized), scale, xw[j]);
        }
    }
    __shared__ float red[128];
    const int tid = threadIdx.x;
    red[tid] = part;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] = __fadd_rn(red[tid], red[tid + off]);
        __syncthreads();
    }
    if (tid == 0) y[row] = red[0];
}

int main() { return 0; }
