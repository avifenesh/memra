// Shared NVFP4 (ModelOpt split-plane) stream primitives: cp.async copies, the f32-accumulate
// m16n8k16 MMA and the in-register E2M1/E4M3 dequant of the one-token visitor. Included by
// moe_f16_grouped.cu (the stream visitors) and dsv4_gpu.cu (the fused DSV4 MoE, memra #17
// lane), so both TUs run the same instructions. Nothing here contracts f32 mul+add, so the
// TUs' different -fmad settings do not change a bit.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ void sk_cp16(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}

// rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md
//   32.03 cyc/warp-MMA, 77.8 TFLOP/s -- the f32-accumulate throttle: half the 155.2 TFLOP/s the
//   f16-accumulate form reaches (flash_attn.cu:974). NO equal-math swap: ptxas rejects f16
//   m16n8k32 and bf16 .block_scale alike (isa_sibling_check.cu), so no deeper-K sibling exists.
//   f16-accumulate would double the rate but is a NUMERIC change -- and unlike attention's P@V
//   (bounded, post-softmax, 0<=p<=1), this is a full FFN GEMM whose f32 `c` accumulates over the
//   whole in_f reduction, where f16 accumulate would overflow/lose mantissa. Verdict:
//   NOT-APPLICABLE (no equal-math sibling; the accumulator is load-bearing here).
__device__ __forceinline__ void sk_mma(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Four ModelOpt E2M1 codes -> two f16x2 MMA B registers. x holds c0 (k 2t), c1 (k 2t+1),
// c2 (k 2t+8), c3 (k 2t+9) in nibbles 0..3; b0 = {c0, c1}, b1 = {c2, c3}. The f16 magnitudes of
// E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6} all have a zero low byte, so a byte LUT gives the high byte and
// the code's sign moves to bit 15. A negative-zero code decodes to +0, matching kv[8] = 0.
__device__ __forceinline__ void kqs_e2m1_f16x4(uint32_t x, uint32_t& b0, uint32_t& b1){
    const uint32_t mag = x & 0x7777u;
    const uint32_t nz  = (mag + 0x7777u) & 0x8888u;   // nibble bit 3 set iff magnitude != 0
    const uint32_t sg  = x & nz;
    const uint32_t hb  = __byte_perm(0x3E3C3800u, 0x46444240u, mag);
    b0 = __byte_perm(hb, 0u, 0x1404u) | ((sg << 12) & 0x8000u) | ((sg << 24) & 0x80000000u);
    b1 = __byte_perm(hb, 0u, 0x3424u) | ((sg << 4) & 0x8000u) | ((sg << 16) & 0x80000000u);
}

// Clear E4M3FN NaN bytes (|b| == 0x7F) to +0, the value g_e4m3fn_to_float returns for them.
__device__ __forceinline__ uint32_t kqs_e4m3fn_clear_nan(uint32_t w){
    const uint32_t nan = ((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
    return w & ~((nan >> 7) * 0xFFu);
}

// Two NaN-cleared E4M3FN bytes -> f16x2 (byte 0 -> low half). Exact: every finite E4M3 value is
// an f16 value.
__device__ __forceinline__ uint32_t kqs_e4m3x2_f16x2(uint32_t w16){
    uint32_t d;
    const unsigned short v = (unsigned short)w16;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(v));
    return d;
}

__device__ __forceinline__ uint32_t kqs_hmul2(uint32_t a, uint32_t b){
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ void kqs_cp8(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0],[%1],8;" :: "r"(s), "l"(g));
}
