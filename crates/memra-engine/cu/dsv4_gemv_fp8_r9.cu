// DSV4 R9 native FP8 GEMV decode.
//
// This is intentionally a separate translation unit.  The native
// cvt.rn.bf16x2.e4m3x2 instruction is unavailable to the CUDA 13.1 ptxas used
// by the ordinary engine build, so the build script compiles this file with an
// explicit CUDA 13.3 compiler only when MEMRA_DSV4_NVCC is supplied.  The
// normal build still links the fail-closed C ABI stubs below.
//
// The native kernel is the ordinary, packed M=1 FP8 GEMV only.  It is not the
// grouped wo_a entry point.  Its decode, scale multiply, per-thread product
// order, and 128-leaf reduction are kept in the same order as
// dsv4_gemv_fp8_m_kernel<1,false>.

#include <cuda_runtime.h>

#include <cstdint>

#ifndef MEMRA_DSV4_R9_NATIVE
#define MEMRA_DSV4_R9_NATIVE 0
#endif

namespace {

constexpr int kR9Unavailable = 40023;

#if MEMRA_DSV4_R9_NATIVE

__device__ __forceinline__ std::uint8_t normalize_e4m3(std::uint8_t code) {
    // ModelOpt's weight contract maps only the two E4M3 NaN encodings to +0.
    // In particular, mag==0 keeps the sign bit, preserving -0.
    return ((code & 0x7fu) == 0x7fu) ? 0u : code;
}

__device__ __forceinline__ std::uint32_t native_e4m3x2_to_bf16x2(std::uint16_t packed) {
    std::uint32_t out;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out)
                 : "h"(packed));
    return out;
}

__device__ __forceinline__ float bf16_bits_to_f32(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ void add_pair(
    float& part, std::uint32_t bf16x2, float scale, std::uint32_t xword) {
    const float w0 = __fmul_rn(
        bf16_bits_to_f32(static_cast<std::uint16_t>(bf16x2)), scale);
    const float w1 = __fmul_rn(
        bf16_bits_to_f32(static_cast<std::uint16_t>(bf16x2 >> 16)), scale);
    const float x0 = __uint_as_float((xword & 0xffffu) << 16);
    const float x1 = __uint_as_float(xword & 0xffff0000u);
    part = __fadd_rn(part, __fmul_rn(w0, x0));
    part = __fadd_rn(part, __fmul_rn(w1, x1));
}

__global__ void dsv4_gemv_fp8_r9_m1_kernel(
    const std::uint8_t* __restrict__ w,
    const float* __restrict__ sc,
    int sc_cols,
    const std::uint16_t* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k) {
    const int row = blockIdx.x;
    if (row >= n) return;

    const std::uint8_t* wr = w + static_cast<long>(row) * k;
    const float* srow = sc + static_cast<long>(row >> 7) * sc_cols;
    const std::uint16_t* xrow = x;
    float part = 0.0f;
    const int stride = blockDim.x * 8;
    int i0 = threadIdx.x * 8;

    // Keep the shipped order: decode both chunks, accumulate every i0
    // product, then every i1 product.  This is deliberately not a
    // pair-interleaved rewrite.
    for (; i0 + stride < k; i0 += 2 * stride) {
        const int i1 = i0 + stride;
        const uint2 wva = *reinterpret_cast<const uint2*>(wr + i0);
        const uint2 wvb = *reinterpret_cast<const uint2*>(wr + i1);
        const uint4 xva = *reinterpret_cast<const uint4*>(xrow + i0);
        const uint4 xvb = *reinterpret_cast<const uint4*>(xrow + i1);
        const std::uint32_t wba[2] = {wva.x, wva.y};
        const std::uint32_t wbb[2] = {wvb.x, wvb.y};
        const std::uint32_t xwa[4] = {xva.x, xva.y, xva.z, xva.w};
        const std::uint32_t xwb[4] = {xvb.x, xvb.y, xvb.z, xvb.w};
        const float sa = srow[i0 >> 7];
        const float sb = srow[i1 >> 7];
        float wua[8], wub[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint16_t raw_a = static_cast<std::uint16_t>(
                (wba[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const std::uint16_t raw_b = static_cast<std::uint16_t>(
                (wbb[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const std::uint16_t norm_a = static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(normalize_e4m3(
                     static_cast<std::uint8_t>(raw_a >> 8)))
                 << 8)
                | normalize_e4m3(static_cast<std::uint8_t>(raw_a)));
            const std::uint16_t norm_b = static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(normalize_e4m3(
                     static_cast<std::uint8_t>(raw_b >> 8)))
                 << 8)
                | normalize_e4m3(static_cast<std::uint8_t>(raw_b)));
            const std::uint32_t bf_a = native_e4m3x2_to_bf16x2(norm_a);
            const std::uint32_t bf_b = native_e4m3x2_to_bf16x2(norm_b);
            wua[2 * j] = __fmul_rn(
                bf16_bits_to_f32(static_cast<std::uint16_t>(bf_a)), sa);
            wua[2 * j + 1] = __fmul_rn(
                bf16_bits_to_f32(static_cast<std::uint16_t>(bf_a >> 16)), sa);
            wub[2 * j] = __fmul_rn(
                bf16_bits_to_f32(static_cast<std::uint16_t>(bf_b)), sb);
            wub[2 * j + 1] = __fmul_rn(
                bf16_bits_to_f32(static_cast<std::uint16_t>(bf_b >> 16)), sb);
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
        const uint4 xv = *reinterpret_cast<const uint4*>(xrow + i0);
        const std::uint32_t wb[2] = {wv.x, wv.y};
        const std::uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
        const float scale = srow[i0 >> 7];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint16_t raw = static_cast<std::uint16_t>(
                (wb[j >> 1] >> ((j & 1) * 16)) & 0xffffu);
            const std::uint16_t norm = static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(normalize_e4m3(
                     static_cast<std::uint8_t>(raw >> 8)))
                 << 8)
                | normalize_e4m3(static_cast<std::uint8_t>(raw)));
            add_pair(part, native_e4m3x2_to_bf16x2(norm), scale, xw[j]);
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
    // M=1 is the t=0 row of the existing [M,N] output layout, so the
    // canonical packed destination is y[row], not y[row*ystride].
    if (tid == 0) y[row] = red[0];
}

#endif

}  // namespace

extern "C" int memra_dsv4_gemv_fp8_r9_available() {
#if MEMRA_DSV4_R9_NATIVE
    return 1;
#else
    return 0;
#endif
}

extern "C" int memra_dsv4_gemv_fp8_r9_m1(
    const void* w_codes,
    const float* sc_f32,
    int sc_cols,
    const void* x_bf16,
    float* y,
    int m,
    int n,
    int k,
    int xstride,
    int ystride,
    void* stream_v) {
#if !MEMRA_DSV4_R9_NATIVE
    (void)w_codes;
    (void)sc_f32;
    (void)sc_cols;
    (void)x_bf16;
    (void)y;
    (void)m;
    (void)n;
    (void)k;
    (void)xstride;
    (void)ystride;
    (void)stream_v;
    return kR9Unavailable;
#else
    if (m != 1 || n <= 0 || k <= 0 || k % 8 != 0 || sc_cols <= 0) return 40020;
    if (!((n == 32768 && k == 1024) || (n == 2048 && k == 4096))) return 40020;
    if (sc_cols < (k + 127) / 128) return 40012;
    if (xstride <= 0) xstride = k;
    if (ystride <= 0) ystride = n;
    if (xstride % 8 != 0 || xstride != k || ystride != n) return 40011;
    if (w_codes == nullptr || sc_f32 == nullptr || x_bf16 == nullptr || y == nullptr) {
        return 40020;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_v);
    dsv4_gemv_fp8_r9_m1_kernel<<<static_cast<unsigned>(n), 128, 0, stream>>>(
        static_cast<const std::uint8_t*>(w_codes), sc_f32, sc_cols,
        static_cast<const std::uint16_t*>(x_bf16), y, n, k);
    const cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
#endif
}
