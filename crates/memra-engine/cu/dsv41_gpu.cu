// DeepSeek-V4.1-Flash single-card kernels: the sibling of dsv4_gpu.cu for the V4.1 program.
//
// Nothing here is shared with the DSv4 (0731) route, and nothing in dsv4_gpu.cu changes for it.
// The numeric program is the vendor reference inference/model.py + kernel.py of
// deepseek-ai/DeepSeek-V4.1-Flash-1017 at rev 1bae0e10, with the torch stand-ins of its tilelang
// kernels as the operational oracle:
//   - activations stay bf16 between ops, exactly where the reference keeps them bf16;
//   - every FP8/FP4 linear quantizes its input to e4m3 with one power-of-two (ue8m0) scale per 32
//     values (act_quant(x, 32, "ue8m0")); the GEMM runs each output as one chain of hardware
//     block-scaled MMAs over the 32-wide K blocks in ascending K (the scales applied and the fp32
//     accumulation done inside the tensor core; the reference sums a block from zero and promotes it
//     to a separate fp32 accumulator: the order gate judges the difference);
//   - dense FP8 weights carry one ue8m0 scale per 32x32 block, routed experts are e2m1 pairs with
//     one ue8m0 scale per row and 32 inputs (MXFP4; the minted NVFP4 form is the same values with
//     duplicated per-16 e4m3 scales times a global scale);
//   - the window KV is quantize-dequantized fp8 per 32, the compressed KV fp4 per 16 with e4m3
//     scales, the index query/key fp4 per 32 with ue8m0 scales;
//   - sparse attention feeds bf16 probabilities to PV, sums the denominator in fp32 and adds the
//     sink to the denominator only, in key blocks of 64 with an online max (the tilelang schedule);
//   - the indexer score is bf16 at every step the reference rounds it (einsum out, product, sum).
// Compiled -fmad=false (build.rs) so separate reference products and sums stay separate.
//
// Return codes: 0 ok, 10000 + cudaError_t on a launch error, 41xxx for a refused contract.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include <atomic>

typedef uint16_t v41_bf;  // raw bf16 bits

#define V41_ERR()                                                   \
    do {                                                            \
        cudaError_t e_ = cudaGetLastError();                        \
        return e_ == cudaSuccess ? 0 : 10000 + (int)e_;             \
    } while (0)

// A kernel's dynamic shared-memory limit and its carveout are attributes of the function in a
// context, not of a launch. Each is set once a device, to the most the kernel ever asks for (the
// first time a launch needs more than the default 48 KB), and never to a call's own size: two threads issuing on one card with different sizes would otherwise launch
// above the limit the other had just set (CUDA error 1 or 701 in a pipeline of stages sharing a
// card). `kernel` goes in parentheses when it is a template instance.
#define V41_ONCE_A_DEVICE(stmt)                                          \
    do {                                                                 \
        static std::atomic<unsigned long long> done_{0};                 \
        int dev_ = 0;                                                    \
        cudaGetDevice(&dev_);                                            \
        unsigned long long bit_ = 1ull << (dev_ & 63);                   \
        if (!(done_.load(std::memory_order_relaxed) & bit_)) {           \
            stmt;                                                        \
            done_.fetch_or(bit_, std::memory_order_relaxed);             \
        }                                                                \
    } while (0)
#define V41_SMEM_LIMIT(kernel, bytes) \
    V41_ONCE_A_DEVICE(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes))
#define V41_CARVE_MAX(kernel)                                                                        \
    V41_ONCE_A_DEVICE(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                           cudaSharedmemCarveoutMaxShared))

// ---------------------------------------------------------------------------------------------
// scalar helpers

__device__ __forceinline__ float v41_bf2f(v41_bf u) { return __uint_as_float(((uint32_t)u) << 16); }
__device__ __forceinline__ v41_bf v41_f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

// 2**ceil(log2(v)) for positive normal v, bit-exact with the reference fast_log2_ceil/fast_pow2.
__device__ __forceinline__ float v41_pow2_ceil(float v) {
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
    return __uint_as_float((uint32_t)(e + 127) << 23);
}
__device__ __forceinline__ uint8_t v41_pow2_code(float p) { return (uint8_t)((__float_as_uint(p) >> 23) & 0xFF); }

// float8_e8m0fnu -> fp32 (255 is NaN, 0 is 2^-127)
__device__ __forceinline__ float v41_e8m0(uint8_t b) {
    if (b == 0xFF) return __uint_as_float(0x7FC00000u);
    if (b == 0) return __uint_as_float(0x00400000u);
    return __uint_as_float((uint32_t)b << 23);
}
__device__ __forceinline__ float v41_e4m3f(uint8_t c) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
    return __half2float(__half(h));
}
__device__ __forceinline__ uint8_t v41_f2e4m3(float v) {
    return (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
// A minted NVFP4 weight's per-32 scale (its e4m3 per-16 code times the tensor's power-of-two scale2) is a
// power of two: its ue8m0 code is the fp32 exponent field (0 for a zero scale: 2^-127).
__device__ __forceinline__ uint32_t v41_mint_ue8(uint8_t c, float s2) {
    return (__float_as_uint(__fmul_rn(v41_e4m3f(c), s2)) >> 23) & 0xFF;
}

// nearest e2m1 grid point, ties to the even mantissa (|x| <= 6 already)
__device__ __forceinline__ float v41_round_fp4(float x) {
    float a = fabsf(x), q;
    if (a <= 0.25f) q = 0.f;
    else if (a < 0.75f) q = 0.5f;
    else if (a <= 1.25f) q = 1.f;
    else if (a < 1.75f) q = 1.5f;
    else if (a <= 2.5f) q = 2.f;
    else if (a < 3.5f) q = 3.f;
    else if (a <= 5.f) q = 4.f;
    else q = 6.f;
    return copysignf(q, x);
}

// e2m1 nibble -> the identical e4m3 byte (every e2m1 value is an e4m3 value): one byte permute
// picks each magnitude from the 8-entry table (selector nibble = magnitude, its mode bit clear),
// then each nibble's sign bit moves to its byte's top bit.
__device__ __forceinline__ uint32_t v41_fp4_to_e4m3x4(uint32_t nib16) {
    // nib16: 4 nibbles in the low 16 bits, element 0 in the lowest nibble
    const uint32_t lut_lo = 0x3C383000u;  // 0, 0.5, 1, 1.5
    const uint32_t lut_hi = 0x4C484440u;  // 2, 3, 4, 6
    uint32_t mag = __byte_perm(lut_lo, lut_hi, nib16 & 0x7777u);
    uint32_t sgn = ((nib16 & 0x8u) << 4) | ((nib16 & 0x80u) << 8) | ((nib16 & 0x800u) << 12) |
                   ((nib16 & 0x8000u) << 16);
    return mag | sgn;
}

__device__ __forceinline__ float v41_warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float v41_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block-wide fp32 sum (blockDim.x multiple of 32, <= 1024); every thread gets the total
__device__ float v41_block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = v41_warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t = __fadd_rn(t, red[i]);
    return t;
}

__device__ __forceinline__ uint32_t v41_lds32(const uint8_t* p) { return *(const uint32_t*)p; }

__device__ __forceinline__ void v41_mma_e4m3(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}
__device__ __forceinline__ void v41_mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// The block-scaled MMA of the FP8/FP4 linears (sm_120a family: QMMA.SF): c = (A * sa) . (B * sb) + c over one
// 32-wide K block, A e4m3 [16][32] (row), B e4m3 [32][8] (col), one ue8m0 scale per row of A and per column of B
// (kind::mxf8f6f4, scale_vec::1X), the scaling and the fp32 accumulation inside the tensor core. Scale
// selectors (PTX ISA "Block Scaling for mma.sync", checked on GB10 by bsmma/probe): byte BA of lane t = 2 TA
// of quad g holds row g's code and of lane t = 2 TA + 1 row g + 8's; byte BB of lane t = TB of quad g holds
// column g's code. Every FP8/FP4 GEMM kernel below runs each output element as one chain of these MMAs over
// its K blocks in ascending order, so an element's bytes do not depend on the kernel, the call's rows or
// the rank count.
template <int BA, int TA, int BB, int TB>
__device__ __forceinline__ void v41_mma_bs_sel(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                               uint32_t sb) {
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(TA), "n"(BB),
          "n"(TB));
}
// The K block u (0..3) of a stage: its codes are byte u of the scale words (lanes t = 0 / 1 for the rows,
// t = 0 for the columns). `u` is a constant after unrolling, so the switch folds away.
__device__ __forceinline__ void v41_mma_bs(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                           uint32_t sb, int u) {
    switch (u) {
        case 0: v41_mma_bs_sel<0, 0, 0, 0>(c, a, b0, b1, sa, sb); break;
        case 1: v41_mma_bs_sel<1, 0, 1, 0>(c, a, b0, b1, sa, sb); break;
        case 2: v41_mma_bs_sel<2, 0, 2, 0>(c, a, b0, b1, sa, sb); break;
        default: v41_mma_bs_sel<3, 0, 3, 0>(c, a, b0, b1, sa, sb); break;
    }
}
__device__ __forceinline__ void v41_ldsm_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void v41_ldsm_x4_trans(uint32_t r[4], const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
// Four 8x8 tiles of 16-bit elements in one load. Lane l gives the address of row (l & 7) of tile
// (l >> 3); register i of lane (g, t) then holds elements 2t, 2t + 1 of row g of tile i: the four
// words a per-lane fragment load would fetch one by one.
__device__ __forceinline__ void v41_ldsm_x4(uint32_t r[4], const void* p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
// the A fragment of an m16 tile from rows r0 .. r0 + 15 of `base` (pitch bytes) at byte offset kb:
// {rows g @ kb, rows g + 8 @ kb, rows g @ kb + 16, rows g + 8 @ kb + 16}
__device__ __forceinline__ void v41_ldsm_a(uint32_t af[4], const uint8_t* base, int r0, int pitch, int kb) {
    int lane = threadIdx.x & 31, q = lane >> 3;
    v41_ldsm_x4(af, base + (long)(r0 + (q & 1) * 8 + (lane & 7)) * pitch + kb + (q >> 1) * 16);
}
// the B fragments of two neighbouring n8 tiles from rows c0 .. c0 + 15 of `base` at byte offset kb:
// {tile 0 b0, tile 0 b1, tile 1 b0, tile 1 b1}
__device__ __forceinline__ void v41_ldsm_b2(uint32_t bf[4], const uint8_t* base, int c0, int pitch, int kb) {
    int lane = threadIdx.x & 31, q = lane >> 3;
    v41_ldsm_x4(bf, base + (long)(c0 + (q >> 1) * 8 + (lane & 7)) * pitch + kb + (q & 1) * 16);
}

// ---------------------------------------------------------------------------------------------
// act_quant(x, 32, "ue8m0"): one warp per 32-wide block. q/s may be null (in-place qdq into y);
// y may alias x. rowidx (nullable) gathers input rows.

// x carries no __restrict__: y may be x
__global__ void v41_act_quant_kernel(const v41_bf* x, long ldx, const int* __restrict__ rowidx,
                                     uint8_t* __restrict__ q, long ldq, uint8_t* __restrict__ s, long lds,
                                     v41_bf* y, long ldy, int rows, int kb) {
    long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= (long)rows * kb) return;
    int r = (int)(wid / kb), b = (int)(wid % kb);
    int ri = rowidx ? rowidx[r] : r;
    float v = v41_bf2f(x[(long)ri * ldx + b * 32 + lane]);
    float a = v41_warp_max(fabsf(v));
    a = fmaxf(a, 1e-4f);
    float sc = v41_pow2_ceil(a * (1.0f / 448.0f));
    float t = fminf(fmaxf(v / sc, -448.f), 448.f);
    uint8_t c = v41_f2e4m3(t);
    if (q) q[(long)r * ldq + b * 32 + lane] = c;
    if (s && lane == 0) s[(long)r * lds + b] = v41_pow2_code(sc);
    if (y) y[(long)r * ldy + b * 32 + lane] = v41_f2bf(v41_e4m3f(c) * sc);
}

extern "C" int memra_v41_act_quant(const void* x, long ldx, const int* rowidx, void* q, long ldq, void* s, long lds,
                                   void* y, long ldy, int rows, int k, void* stream) {
    if (k % 32 != 0 || rows < 0) return 41001;
    if (rows == 0) return 0;
    int kb = k / 32;
    long warps = (long)rows * kb;
    long blocks = (warps * 32 + 255) / 256;
    v41_act_quant_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        (const v41_bf*)x, ldx, rowidx, (uint8_t*)q, ldq, (uint8_t*)s, lds, (v41_bf*)y, ldy, rows, kb);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// fp4_act_quant(x, G, inplace=True): G=32 with ue8m0 scales (index Q/K), G=16 with e4m3 scales
// (compressed KV). One thread per group, in place on bf16 rows.

template <int G, bool E4M3>
__global__ void v41_fp4_qdq_kernel(v41_bf* x, long ld, int rows, int ng) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * ng) return;
    int r = (int)(i / ng), gi = (int)(i % ng);
    v41_bf* p = x + (long)r * ld + gi * G;
    float v[G];
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < G; ++j) {
        v[j] = v41_bf2f(p[j]);
        a = fmaxf(a, fabsf(v[j]));
    }
    float s;
    if (E4M3) {
        a = fmaxf(a, 6.f * 0.001953125f);  // 6 * 2^-9: an all-zero group keeps a nonzero scale
        s = v41_e4m3f(v41_f2e4m3(a / 6.0f));
    } else {
        a = fmaxf(a, 6.f * 1.1754943508222875e-38f);  // 6 * 2^-126
        s = v41_pow2_ceil(a * (1.0f / 6.0f));
    }
#pragma unroll
    for (int j = 0; j < G; ++j) {
        float t = fminf(fmaxf(v[j] / s, -6.f), 6.f);
        p[j] = v41_f2bf(v41_round_fp4(t) * s);
    }
}

extern "C" int memra_v41_fp4_qdq(void* x, long ld, int rows, int n, int group, void* stream) {
    if (rows <= 0) return rows < 0 ? 41002 : 0;
    if (group == 32) {
        if (n % 32) return 41002;
        long tot = (long)rows * (n / 32);
        v41_fp4_qdq_kernel<32, false><<<(unsigned)((tot + 127) / 128), 128, 0, (cudaStream_t)stream>>>(
            (v41_bf*)x, ld, rows, n / 32);
    } else if (group == 16) {
        if (n % 16) return 41002;
        long tot = (long)rows * (n / 16);
        v41_fp4_qdq_kernel<16, true><<<(unsigned)((tot + 127) / 128), 128, 0, (cudaStream_t)stream>>>(
            (v41_bf*)x, ld, rows, n / 16);
    } else {
        return 41002;
    }
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// RMSNorm (bf16 in, bf16 weight, fp32 math, bf16 out): y = bf16(w * (x * rsqrt(mean(x^2) + eps))).
// y may alias x.

__global__ void v41_rmsnorm_kernel(const v41_bf* x, long ldx, const v41_bf* __restrict__ w, v41_bf* y, long ldy,
                                   int d, float eps) {
    __shared__ float red[32];
    const v41_bf* xr = x + (long)blockIdx.x * ldx;
    int bd = blockDim.x;
    float ss = 0.f;
    // four strided loads in flight ahead of their adds, which stay in the strided order
    for (int i = threadIdx.x; i < d; i += 4 * bd) {
        float v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) v[u] = i + u * bd < d ? v41_bf2f(xr[i + u * bd]) : 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (i + u * bd < d) ss = __fadd_rn(ss, __fmul_rn(v[u], v[u]));
    }
    float tot = v41_block_sum(ss, red);
    float rs = rsqrtf(__fadd_rn(tot / (float)d, eps));
    v41_bf* yr = y + (long)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < d; i += 4 * bd) {
        float v[4], wv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            bool ok = i + u * bd < d;
            v[u] = ok ? v41_bf2f(xr[i + u * bd]) : 0.f;
            wv[u] = ok ? v41_bf2f(w[i + u * bd]) : 0.f;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u)
            if (i + u * bd < d) yr[i + u * bd] = v41_f2bf(__fmul_rn(wv[u], __fmul_rn(v[u], rs)));
    }
}

// The same program for d <= 32 x blockDim: a thread's strided values (at most 32) and their weights
// are all loaded at once and stay in registers for both passes; the squares are added in the strided
// order as above.
__global__ void v41_rmsnorm32_kernel(const v41_bf* x, long ldx, const v41_bf* __restrict__ w, v41_bf* y, long ldy,
                                     int d, float eps) {
    __shared__ float red[32];
    const v41_bf* xr = x + (long)blockIdx.x * ldx;
    int bd = blockDim.x;
    float v[32], wv[32];
#pragma unroll
    for (int u = 0; u < 32; ++u) {
        int i = threadIdx.x + u * bd;
        bool ok = i < d;
        v[u] = ok ? v41_bf2f(xr[i]) : 0.f;
        wv[u] = ok ? v41_bf2f(w[i]) : 0.f;
    }
    float ss = 0.f;
#pragma unroll
    for (int u = 0; u < 32; ++u)
        if (threadIdx.x + u * bd < d) ss = __fadd_rn(ss, __fmul_rn(v[u], v[u]));
    float tot = v41_block_sum(ss, red);
    float rs = rsqrtf(__fadd_rn(tot / (float)d, eps));
    v41_bf* yr = y + (long)blockIdx.x * ldy;
#pragma unroll
    for (int u = 0; u < 32; ++u) {
        int i = threadIdx.x + u * bd;
        if (i < d) yr[i] = v41_f2bf(__fmul_rn(wv[u], __fmul_rn(v[u], rs)));
    }
}

extern "C" int memra_v41_rmsnorm(const void* x, long ldx, const void* w, void* y, long ldy, int rows, int d, float eps,
                                 void* stream) {
    if (rows <= 0) return rows < 0 ? 41003 : 0;
    int th = d >= 1024 ? 256 : 128;
    if (d <= 32 * th) {
        v41_rmsnorm32_kernel<<<rows, th, 0, (cudaStream_t)stream>>>((const v41_bf*)x, ldx, (const v41_bf*)w, (v41_bf*)y,
                                                                    ldy, d, eps);
        V41_ERR();
    }
    v41_rmsnorm_kernel<<<rows, th, 0, (cudaStream_t)stream>>>((const v41_bf*)x, ldx, (const v41_bf*)w, (v41_bf*)y,
                                                              ldy, d, eps);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// RoPE on the trailing 64 dims of each head vector, adjacent pairs as complex numbers (in place,
// bf16). Row r takes position pos[r] when pos is non-null, else pos_base + r * pos_step. The
// tables are fp32 [positions][32] cos and sin; inverse conjugates the rotation.

__global__ void v41_rope_kernel(v41_bf* x, long ld_row, int heads, int hd, const float* __restrict__ cosb,
                                const float* __restrict__ sinb, int pos_base, int pos_step, const int* __restrict__ pos,
                                int inverse) {
    int r = blockIdx.x;
    int p = pos ? pos[r] : pos_base + r * pos_step;
    int npairs = heads * 32;
    for (int t = threadIdx.x; t < npairs; t += blockDim.x) {
        int h = t >> 5, i = t & 31;
        v41_bf* e = x + (long)r * ld_row + (long)h * hd + (hd - 64) + 2 * i;
        float a = v41_bf2f(e[0]), b = v41_bf2f(e[1]);
        float c = cosb[(long)p * 32 + i], s = sinb[(long)p * 32 + i];
        if (inverse) s = -s;
        float re = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
        float im = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
        e[0] = v41_f2bf(re);
        e[1] = v41_f2bf(im);
    }
}

extern "C" int memra_v41_rope(void* x, long ld_row, int rows, int heads, int hd, const float* cosb, const float* sinb,
                              int pos_base, int pos_step, const int* pos, int inverse, void* stream) {
    if (rows <= 0) return rows < 0 ? 41004 : 0;
    if (hd < 64 || hd % 2) return 41004;
    int th = heads * 32 >= 256 ? 256 : 64;
    v41_rope_kernel<<<rows, th, 0, (cudaStream_t)stream>>>((v41_bf*)x, ld_row, heads, hd, cosb, sinb, pos_base,
                                                           pos_step, pos, inverse);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Block-scaled FP8 GEMM: C[M,N] = sum over 32-wide K blocks of (A_blk . B_blk) * sa[m,kb] * sb(n,kb).
// A: e4m3 codes [M,K] (lda bytes) + ue8m0 scales [M,K/32] (ldas). B by weight format WF:
//   0: e4m3 [N,K] (ldb bytes), ue8m0 per 32x32 block [ceil(N/32), K/32] (ldbs)
//   1: e2m1 pairs [N,K/2] (ldb bytes), ue8m0 per row and 32 inputs [N, K/32] (MXFP4)
//   2: e2m1 pairs [N,K/2], e4m3 per row and 16 inputs [N, K/16] times bscale2 (minted NVFP4;
//      the loader checked that each 32-block's two per-16 scales are equal)
// Each output element is one chain of block-scaled MMAs (v41_mma_bs) over its 32-wide K blocks in
// ascending K: acc = (A_kb * sa) . (B_kb * sb) + acc, the scaling and the fp32 accumulation inside the
// tensor core. (The vendor's fp8_gemm / fp4_gemm sum each K block from zero and promote it to a separate
// fp32 accumulator on CUDA cores; this program moves the promotion into the tensor core, the form of the
// hardware block-scaled MMA. The layer and order gates judge the bytes it moves.) OUTF 0 = bf16, 1 = fp32.
// Tile 64x128x32, 8 warps.

#define V41_GM 64
#define V41_GN 128
#define V41_GS 48  // smem bytes per 32-byte row: conflict-free fragment loads

// A weight's ue8m0 code for output column c (WF 0: its 32-row block's) of K block kb, 0 past N.
template <int WF>
__device__ __forceinline__ uint32_t v41_wcode(const uint8_t* __restrict__ Bs, long ldbs, float bscale2, int c, int N,
                                              int kb) {
    if (c >= N) return 0;
    if (WF == 0) return Bs[(long)(c >> 5) * ldbs + kb];
    if (WF == 1) return Bs[(long)c * ldbs + kb];
    return v41_mint_ue8(Bs[(long)c * ldbs + 2 * kb], bscale2);
}

// One 64x128 output tile at (m0, n0), one K block a shared-memory round, no pipelining: the reference
// schedule the kernel gate holds every dispatched schedule to, byte for byte.
template <int WF, int OUTF>
__device__ __forceinline__ void v41_gemm_f8_tile(const uint8_t* __restrict__ A, long lda,
                                                 const uint8_t* __restrict__ As, long ldas,
                                                 const uint8_t* __restrict__ B, long ldb,
                                                 const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                 void* __restrict__ C, long ldc, int M, int N, int K, int m0,
                                                 int n0, uint8_t* sA, uint8_t* sB, uint8_t* sSa, uint8_t* sSb) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;
    int KB = K / 32;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    for (int kb = 0; kb < KB; ++kb) {
        // A tile: 64 rows x 32 bytes, threads 0..127 one 16-byte half each; row codes by 128..191
        if (tid < 128) {
            int r = tid >> 1, h = tid & 1;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < M) v = *(const uint4*)(A + (long)(m0 + r) * lda + kb * 32 + h * 16);
            *(uint4*)(&sA[r * V41_GS + h * 16]) = v;
        } else if (tid < 192) {
            int r = tid - 128;
            sSa[r] = (m0 + r < M) ? As[(long)(m0 + r) * ldas + kb] : 0;
        }
        {
            int r = tid >> 1, h = tid & 1;
            if (WF == 0) {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (n0 + r < N) v = *(const uint4*)(B + (long)(n0 + r) * ldb + kb * 32 + h * 16);
                *(uint4*)(&sB[r * V41_GS + h * 16]) = v;
            } else {
                uint2 v = make_uint2(0, 0);
                if (n0 + r < N) v = *(const uint2*)(B + (long)(n0 + r) * ldb + kb * 16 + h * 8);
                uint4 o;
                o.x = v41_fp4_to_e4m3x4(v.x & 0xFFFF);
                o.y = v41_fp4_to_e4m3x4(v.x >> 16);
                o.z = v41_fp4_to_e4m3x4(v.y & 0xFFFF);
                o.w = v41_fp4_to_e4m3x4(v.y >> 16);
                *(uint4*)(&sB[r * V41_GS + h * 16]) = o;
            }
        }
        if (tid < 128) sSb[tid] = (uint8_t)v41_wcode<WF>(Bs, ldbs, bscale2, n0 + tid, N, kb);
        __syncthreads();
        uint32_t af[2][4], sa[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int r = wm * 32 + mi * 16;
            af[mi][0] = v41_lds32(&sA[(r + g) * V41_GS + t * 4]);
            af[mi][1] = v41_lds32(&sA[(r + g + 8) * V41_GS + t * 4]);
            af[mi][2] = v41_lds32(&sA[(r + g) * V41_GS + 16 + t * 4]);
            af[mi][3] = v41_lds32(&sA[(r + g + 8) * V41_GS + 16 + t * 4]);
            sa[mi] = sSa[r + g + 8 * (t & 1)];  // lane t = 0: row g, t = 1: row g + 8
        }
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int c = wn * 32 + ni * 8;
            uint32_t b0 = v41_lds32(&sB[(c + g) * V41_GS + t * 4]);
            uint32_t b1 = v41_lds32(&sB[(c + g) * V41_GS + 16 + t * 4]);
            uint32_t sb = sSb[c + g];  // lane t = 0: column g
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) v41_mma_bs(acc[mi][ni], af[mi], b0, b1, sa[mi], sb, 0);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int row = m0 + wm * 32 + mi * 16 + g + (e >= 2 ? 8 : 0);
                int col = n0 + wn * 32 + ni * 8 + 2 * t + (e & 1);
                if (row < M && col < N) {
                    if (OUTF == 0) ((v41_bf*)C)[(long)row * ldc + col] = v41_f2bf(acc[mi][ni][e]);
                    else ((float*)C)[(long)row * ldc + col] = acc[mi][ni][e];
                }
            }
}

// The reference schedule: the tile body above as its own launch. Not dispatched; `memra_v41_gemm_f8_ref`
// runs it so `dsv41-gate kernels` can prove the pipelined and small-row schedules write its bytes.
template <int WF, int OUTF>
__global__ void __launch_bounds__(256) v41_gemm_f8_kernel(const uint8_t* __restrict__ A, long lda,
                                                          const uint8_t* __restrict__ As, long ldas,
                                                          const uint8_t* __restrict__ B, long ldb,
                                                          const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                          void* __restrict__ C, long ldc, int M, int N, int K) {
    __shared__ __align__(16) uint8_t sA[V41_GM * V41_GS];
    __shared__ __align__(16) uint8_t sB[V41_GN * V41_GS];
    __shared__ uint8_t sSa[V41_GM];
    __shared__ uint8_t sSb[V41_GN];
    v41_gemm_f8_tile<WF, OUTF>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, C, ldc, M, N, K, blockIdx.y * V41_GM,
                               blockIdx.x * V41_GN, sA, sB, sSa, sSb);
}

// The device-planned dispatch (`memra_v41_moe_plan`): one entry per tile of an expert with rows
struct V41Tile {
    int row0;    // the expert's first pair
    int rows;    // the expert's pairs
    int m0;      // this tile's first row inside the expert
    int expert;  // index into the rank's expert weight table
};

// one routed expert's three projections (w1, w2, w3 as `LayerW::experts` holds them)
struct V41ExpW {
    const uint8_t* w[3];
    const uint8_t* s[3];
    float scale2[3];
    int pad;
};

// Pipelined block-scaled GEMM (more than V41_SM_MAX rows): the tile kernel's per-element program (one
// chain of block-scaled MMAs over the K blocks in ascending order, on the same fragment lanes) on a BM x 128
// tile whose K blocks stream through a two-stage cp.async ring: the next stage's copies are in flight while
// the current one runs its MMAs. A stage holds 3 K blocks of a 64-row tile (a block then stays under half
// an SM's 100 KB of shared memory, so two blocks, 16 warps, share an SM) and 4 of a 128-row tile (one block
// per SM). Fragments come from ldmatrix on rows padded by 16 bytes (conflict-free); MXFP4 / minted NVFP4
// weights land raw and are expanded to e4m3 once per stage by the whole block (the same values: e2m1 is a
// subset of e4m3). A stage's scale codes are read with plain loads while the stage before it runs and
// stored next to it, one 32-bit word a row and a column (byte u: K block u), which the MMA's selectors read
// directly.

// Debug builds only (-DV41_SMEM_POISON): fill a GEMM kernel's dynamic shared memory with 0xFF bytes before it runs,
// so an output that depends on shared memory the kernel never wrote shows as a byte difference in the kernel gate
// (compute-sanitizer's initcheck covers global memory only).
__device__ __forceinline__ void v41_smem_poison(uint8_t* sm) {
#ifdef V41_SMEM_POISON
    uint32_t n;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(n));
    for (uint32_t i = threadIdx.x * 4; i + 4 <= n; i += blockDim.x * 4) *(uint32_t*)(sm + i) = 0xFFFFFFFFu;
    __syncthreads();
#else
    (void)sm;
#endif
}

__device__ __forceinline__ void v41_cp16(void* smem, const void* gmem, bool pred) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    int n = pred ? 16 : 0;  // 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void v41_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void v41_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <int WF, int BM>
struct V41Pipe {
    static constexpr int BN = 128;
    static constexpr int PK = BM == 128 ? 4 : 3;  // 32-wide K blocks per stage (at most 4: one scale word)
    static constexpr int PS = 2;                  // stages in flight
    static constexpr int PRS = PK * 32 + 16;      // smem bytes per e4m3 row of a stage (+16: conflict-free ldmatrix)
    static constexpr int PRR = PK * 16 + 16;      // smem bytes per raw fp4 row of a stage
    static constexpr int A_B = BM * PRS;                       // e4m3 A rows
    static constexpr int B_B = WF == 0 ? BN * PRS : BN * PRR;  // e4m3 or raw fp4 B rows
    static constexpr int SA_B = BM * 4;                        // a row's ue8m0 codes of the stage's K blocks, one word
    static constexpr int SB_B = BN * 4;                        // a column's (WF 0: its 32-column block's)
    static constexpr int STAGE = A_B + B_B + SA_B + SB_B;
    static constexpr int CONV = WF == 0 ? 0 : BN * PRS;  // the expanded e4m3 B of the current stage
    static constexpr int SMEM = PS * STAGE + CONV;
};

template <int WF, int OUTF, int BM>
__device__ __forceinline__ void v41_gemm_f8_pipe_tile(const uint8_t* __restrict__ A, long lda,
                                                      const uint8_t* __restrict__ As, long ldas,
                                                      const uint8_t* __restrict__ B, long ldb,
                                                      const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                      void* __restrict__ C, long ldc, int M, int N, int K, int m0,
                                                      int n0, uint8_t* sm, const int* __restrict__ rix = nullptr) {
    using P = V41Pipe<WF, BM>;
    // rix: row i of the tile's M is row rix[i] of A and As (the expert tiles over activations quantized
    // once a token); null: row i
    auto arow = [&](int i) -> long { return rix ? (long)rix[i] : (long)i; };
    constexpr int BN = P::BN;
    // a thread copies the same A rows in every stage (its chunks c, c + 256, ...) and reads one row's scale
    // codes: their rows in A are read once here, not once a stage (a load in front of every stage's copies
    // would sit in the pipeline's path)
    constexpr int NCH = (BM * 2 * P::PK + 255) / 256;
    long ar[NCH];
#pragma unroll
    for (int j = 0; j < NCH; ++j) {
        int r = ((int)threadIdx.x + j * 256) / (2 * P::PK);
        ar[j] = (r < BM && m0 + r < M) ? arow(m0 + r) : 0;
    }
    long asrow = ((int)threadIdx.x < BM && m0 + (int)threadIdx.x < M) ? arow(m0 + (int)threadIdx.x) : 0;
    constexpr int MI = BM / 32;  // 16-row MMA tiles per warp (warp rows BM / 2)
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;
    int KB = K / 32;
    int NS = (KB + P::PK - 1) / P::PK;  // stages
    auto stage = [&](int i) { return sm + (i % P::PS) * P::STAGE; };
    // ---- issue the copies of stage i (K blocks i*P::PK ..), zero-filling rows past M / N and K
    auto issue = [&](int i) {
        uint8_t* st = stage(i);
        uint8_t* sA = st;
        uint8_t* sB = st + P::A_B;
        int kb0 = i * P::PK;
        // A: BM rows x CR chunks of 16 B (two per K block)
        constexpr int CR = 2 * P::PK;
        for (int c = tid, j = 0; c < BM * CR; c += 256, ++j) {
            int r = c / CR, q = c % CR;
            int kb = kb0 + (q >> 1);
            bool ok = (m0 + r < M) && (kb < KB);
            const uint8_t* src = ok ? A + ar[j] * lda + kb0 * 32 + q * 16 : A;
            v41_cp16(sA + r * P::PRS + q * 16, src, ok);
        }
        if (WF == 0) {
            for (int c = tid; c < BN * CR; c += 256) {
                int r = c / CR, q = c % CR;
                int kb = kb0 + (q >> 1);
                bool ok = (n0 + r < N) && (kb < KB);
                const uint8_t* src = ok ? B + (long)(n0 + r) * ldb + kb0 * 32 + q * 16 : B;
                v41_cp16(sB + r * P::PRS + q * 16, src, ok);
            }
        } else {
            for (int c = tid; c < BN * P::PK; c += 256) {  // a raw K block is one 16-byte chunk
                int r = c / P::PK, q = c % P::PK;
                int kb = kb0 + q;
                bool ok = (n0 + r < N) && (kb < KB);
                const uint8_t* src = ok ? B + (long)(n0 + r) * ldb + kb0 * 16 + q * 16 : B;
                v41_cp16(sB + r * P::PRR + q * 16, src, ok);
            }
        }
    };
    // ---- scale codes: thread r < BM packs row r's codes of the stage (byte u = K block u), threads 128..
    // a column's; read with plain loads while the stage before runs, stored next to the stage after it
    auto load_sc = [&](int i) -> uint32_t {
        int kb0 = i * P::PK;
        uint32_t code = 0;
        if (i >= NS) return code;
        if (tid < BM) {
            if (m0 + tid < M) {
                const uint8_t* asr = As + asrow * ldas;
                for (int u = 0; u < P::PK; ++u)
                    if (kb0 + u < KB) code |= (uint32_t)asr[kb0 + u] << (8 * u);
            }
        } else if (tid >= 128) {
            int c = n0 + tid - 128;
            for (int u = 0; u < P::PK; ++u)
                if (kb0 + u < KB) code |= v41_wcode<WF>(Bs, ldbs, bscale2, c, N, kb0 + u) << (8 * u);
        }
        return code;
    };
    auto store_sc = [&](int i, uint32_t code) {
        if (i >= NS) return;
        uint8_t* st = stage(i);
        if (tid < BM) ((uint32_t*)(st + P::A_B + P::B_B))[tid] = code;
        else if (tid >= 128) ((uint32_t*)(st + P::A_B + P::B_B + P::SA_B))[tid - 128] = code;
    };

    // A tile's last rows and a launch's last columns are padding: a warp whose rows or columns are all
    // outside (`dead`) computes nothing, and a warp skips each 16-row MMA group that is outside (`lr`).
    bool dead = m0 + wm * (BM / 2) >= M || n0 + wn * 32 >= N;
    bool lr[MI];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) lr[mi] = m0 + wm * (BM / 2) + mi * 16 < M;

    float acc[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // prologue: stages 0 .. PS-2 in flight with their scales stored. Iteration i then issues stage
    // i + PS - 1, loads its scale codes into a register (plain loads, in flight while stage i runs)
    // and stores them at its end, into the slot stage i - 1 left free.
    for (int i = 0; i < P::PS - 1; ++i) {
        if (i < NS) issue(i);
        v41_cp_commit();
        store_sc(i, load_sc(i));
    }
    uint8_t* sBc = sm + P::PS * P::STAGE;
    for (int i = 0; i < NS; ++i) {
        v41_cp_wait<P::PS - 2>();
        __syncthreads();  // stage i landed for every thread; stage i-1's readers are done
        int nx = i + P::PS - 1;
        if (nx < NS) issue(nx);
        v41_cp_commit();
        uint32_t nxt = load_sc(nx);
        uint8_t* st = stage(i);
        const uint8_t* sA = st;
        const uint8_t* sB = st + P::A_B;
        const uint32_t* sSa = (const uint32_t*)(st + P::A_B + P::B_B);
        const uint32_t* sSb = (const uint32_t*)(st + P::A_B + P::B_B + P::SA_B);
        if (WF != 0) {
            // expand this stage's raw fp4 rows to e4m3 once, as the tile kernel's stager does
            for (int c = tid; c < BN * 2 * P::PK; c += 256) {
                int r = c / (2 * P::PK), q = c % (2 * P::PK);  // q: 8 raw bytes -> 16 e4m3 bytes
                uint2 v = *(const uint2*)(sB + r * P::PRR + q * 8);
                uint4 o;
                o.x = v41_fp4_to_e4m3x4(v.x & 0xFFFF);
                o.y = v41_fp4_to_e4m3x4(v.x >> 16);
                o.z = v41_fp4_to_e4m3x4(v.y & 0xFFFF);
                o.w = v41_fp4_to_e4m3x4(v.y >> 16);
                *(uint4*)(sBc + r * P::PRS + q * 16) = o;
            }
            __syncthreads();
            sB = sBc;
        }
        if (!dead) {
            // this lane's scale words of the stage: rows g (lane t = 0) and g + 8 (t = 1) of each 16-row
            // group, column g of each 8-column group; byte u is K block u
            uint32_t sa[MI], sb[4];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) sa[mi] = sSa[wm * (BM / 2) + mi * 16 + g + 8 * (t & 1)];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) sb[ni] = sSb[wn * 32 + ni * 8 + g];
#pragma unroll
            for (int kk = 0; kk < P::PK; ++kk) {
                if (i * P::PK + kk >= KB) break;
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    int q = lane >> 3;
                    int r = wm * (BM / 2) + mi * 16 + (q & 1) * 8 + (lane & 7);
                    v41_ldsm_x4(af[mi], sA + r * P::PRS + kk * 32 + (q >> 1) * 16);
                }
                uint32_t bf[4][2];
#pragma unroll
                for (int nj = 0; nj < 2; ++nj) {
                    int q = lane >> 3;
                    int cn = wn * 32 + nj * 16 + (q >> 1) * 8 + (lane & 7);
                    uint32_t r4[4];
                    v41_ldsm_x4(r4, sB + cn * P::PRS + kk * 32 + (q & 1) * 16);
                    bf[2 * nj][0] = r4[0];
                    bf[2 * nj][1] = r4[1];
                    bf[2 * nj + 1][0] = r4[2];
                    bf[2 * nj + 1][1] = r4[3];
                }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int mi = 0; mi < MI; ++mi)
                        if (lr[mi]) v41_mma_bs(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1], sa[mi], sb[ni], kk);
            }
        }
        // the scales loaded last iteration go next to their stage (its slot was freed by the barrier)
        store_sc(nx, nxt);
    }
    v41_cp_wait<0>();
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int row = m0 + wm * (BM / 2) + mi * 16 + g + (e >= 2 ? 8 : 0);
                int col = n0 + wn * 32 + ni * 8 + 2 * t + (e & 1);
                if (row < M && col < N) {
                    if (OUTF == 0) ((v41_bf*)C)[(long)row * ldc + col] = v41_f2bf(acc[mi][ni][e]);
                    else ((float*)C)[(long)row * ldc + col] = acc[mi][ni][e];
                }
            }
}

template <int WF, int OUTF, int BM>
__global__ void __launch_bounds__(256, BM <= 64 ? 2 : 1) v41_gemm_f8_pipe_kernel(const uint8_t* __restrict__ A, long lda,
                                                               const uint8_t* __restrict__ As, long ldas,
                                                               const uint8_t* __restrict__ B, long ldb,
                                                               const uint8_t* __restrict__ Bs, long ldbs,
                                                               float bscale2, void* __restrict__ C, long ldc, int M,
                                                               int N, int K) {
    extern __shared__ __align__(16) uint8_t v41_pipe_sm[];
    v41_smem_poison(v41_pipe_sm);
    v41_gemm_f8_pipe_tile<WF, OUTF, BM>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, C, ldc, M, N, K, blockIdx.y * BM,
                                        blockIdx.x * 128, v41_pipe_sm);
}

// Small-row block-scaled GEMM (at most 32 rows: decode, verify rows, small batches, and a grouped dispatch
// whose groups hold at most V41_SM_MAX rows). Every output element runs the tile kernel's program: one chain
// of block-scaled MMAs over its K blocks in ascending order. A chain cannot be split over K, so the work
// splits over the output columns instead: the MMA's operands are exchanged (its 16-wide side holds weight
// rows = output columns, its 8-wide side up to 8 activation rows, a warp runs MR such sets of rows), and a
// warp owns CW weight rows (16, 8 or 4; fewer than 16 leaves rows of the MMA zero, so a narrow output still
// spreads over enough warps) and walks all of K. Each warp streams its weight rows, the call's activation
// rows and both scale codes through its own cp.async ring of PS stages of 4 K blocks (no block barrier: the
// warps of a block run independently). The tensor core forms the same products and sums whichever operand
// a factor is in, so a row's bytes do not depend on which kernel ran: `dsv41-gate kernels` checks each of
// these schedules against the reference tile byte for byte.
#define V41_SM_MAX 8
// Calls of V41_SM_MAX + 1 to v41_wide[family] rows keep the small-row bodies, issued once a group of
// rows inside the call: a batch of 9 to 32 sequences is far from filling a 64-row pipelined tile, whose
// fixed cost made a 12-row call cost three 8-row calls (two cards, 20 layers: 12.7 ms at 8 rows, 35.3 at
// 12). Families: 0 block-scaled dense, 1 bf16, 2 fp32 FMA, 3 the one-pass mHC kernel. Rows are
// independent in every one of them, so a row's bytes do not depend on the group it runs in. The limits
// are measured per family and rank count (`wide_rows_for` in src/dsv41_gpu.rs); these are one rank's.
// Atomic: a gate or bench thread sets them while rank threads launch.
static std::atomic<int> v41_wide_rows_a[4] = {24, 16, 48, 32};
static inline int v41_wide(int family) { return v41_wide_rows_a[family].load(std::memory_order_relaxed); }
// A group's limits, or a gate's or bench's choice of them.
extern "C" void memra_v41_wide_rows(int family, int rows) {
    if (family >= 0 && family < 4) v41_wide_rows_a[family].store(rows, std::memory_order_relaxed);
}

__device__ __forceinline__ uint32_t v41_ldg32(const uint8_t* p) { return __ldg((const unsigned int*)p); }
__device__ __forceinline__ uint32_t v41_ldg16(const uint8_t* p) { return (uint32_t)__ldg((const unsigned short*)p); }

__device__ __forceinline__ void v41_cp8(void* smem, const void* gmem, bool pred) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    int n = pred ? 8 : 0;  // 0: the 8 bytes are zero-filled
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void v41_cp4(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

// A warp's ring: PS stages of PK = 4 K blocks, each [CW weight rows][WR] + [RA activation rows][AR] + the
// weight rows' scale codes [CW][SWB] + the activation rows' [RA][4]. RA: the activation rows the launch holds
// (a call's rows, or a tile list's tile rows), so a one-row call keeps one row and the ring goes deep: the
// small-row body is bound by the bytes it has in flight, not by its MMAs.
template <int WF, int CW, int MR>
struct V41Sk {
    static constexpr int PK = 4;
    static constexpr int PS = 6;
    static constexpr int WR = WF == 0 ? PK * 32 + 16 : PK * 16 + 16;  // +16: conflict-free fragment loads
    static constexpr int AR = PK * 32 + 16;
    static constexpr int SWB = WF == 2 ? 8 : 4;  // minted: two e4m3 codes a K block, converted when read
    static constexpr int W_B = CW * WR;
    static constexpr int WS_B = (CW * SWB + 15) & ~15;
    static constexpr int NW = MR == 1 ? 4 : 2;  // warps a block (each holds its own copy of the rows)
    __host__ __device__ static constexpr int a_b(int ra) { return ra * AR; }
    __host__ __device__ static constexpr int as_b(int ra) { return (ra * 4 + 15) & ~15; }
    __host__ __device__ static constexpr int stage(int ra) { return W_B + a_b(ra) + WS_B + as_b(ra); }
    __host__ __device__ static constexpr int warp(int ra) { return PS * stage(ra); }
    __host__ __device__ static constexpr int block(int ra) { return NW * warp(ra); }
};

// The chain of one warp: weight rows rw0 .. rw0 + CW - 1 of B (output columns) against activation rows
// 0 .. M - 1 of A (M <= RA <= 8 MR), all of K (K / 32 a multiple of 4). acc[mr]: lane (g, t) holds weight row g
// against activation rows 8 mr + 2t, 8 mr + 2t + 1 (acc[mr][0], [1]) and weight row g + 8 (acc[mr][2], [3]).
template <int WF, int CW, int MR>
__device__ __forceinline__ void v41_sk_chain(const uint8_t* __restrict__ A, long lda, const uint8_t* __restrict__ As,
                                             long ldas, const uint8_t* __restrict__ B, long ldb,
                                             const uint8_t* __restrict__ Bs, long ldbs, float bscale2, int M, int N,
                                             int K, int rw0, int RA, uint8_t* ring, float acc[MR][4]) {
    using P = V41Sk<WF, CW, MR>;
    const int STAGE = P::stage(RA), A_B = P::a_b(RA), WS_O = P::W_B + A_B, AS_O = WS_O + P::WS_B;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int KB = K / 32, NS = KB / P::PK;
    int nw = min(CW, N - rw0);  // live weight rows (the rest stay zero)
#pragma unroll
    for (int mr = 0; mr < MR; ++mr) acc[mr][0] = acc[mr][1] = acc[mr][2] = acc[mr][3] = 0.f;
    if (nw <= 0) return;
    // rows that are never copied (weight rows past N, activation rows past M) are never read: their fragments
    // and scale codes are zeros in registers
    auto issue = [&](int i) {
        uint8_t* st = ring + (i % P::PS) * STAGE;
        int kb0 = i * P::PK;
        if (WF == 0) {
            for (int c = lane; c < nw * 8; c += 32) {
                int r = c >> 3, q = c & 7;
                v41_cp16(st + r * P::WR + q * 16, B + (long)(rw0 + r) * ldb + kb0 * 32 + q * 16, true);
            }
        } else {
            for (int c = lane; c < nw * 4; c += 32) {
                int r = c >> 2, q = c & 3;
                v41_cp16(st + r * P::WR + q * 16, B + (long)(rw0 + r) * ldb + kb0 * 16 + q * 16, true);
            }
        }
        for (int c = lane; c < M * 8; c += 32) {
            int r = c >> 3, q = c & 7;
            v41_cp16(st + P::W_B + r * P::AR + q * 16, A + (long)r * lda + kb0 * 32 + q * 16, true);
        }
        uint8_t* ws = st + WS_O;
        if (lane < nw) {
            int r = rw0 + lane;
            if (WF == 0) v41_cp4(ws + lane * 4, Bs + (long)(r >> 5) * ldbs + kb0);
            else if (WF == 1) v41_cp4(ws + lane * 4, Bs + (long)r * ldbs + kb0);
            else v41_cp8(ws + lane * 8, Bs + (long)r * ldbs + 2 * kb0, true);
        }
        if (lane < M) v41_cp4(st + AS_O + lane * 4, As + (long)lane * ldas + kb0);
    };
    for (int s = 0; s < P::PS - 1; ++s) {
        if (s < NS) issue(s);
        v41_cp_commit();
    }
    // this lane's weight rows in the MMA (rows g and g + 8 of the 16-wide side) and its scale-code row
    bool w0 = g < nw, w1 = g + 8 < nw;
    int wsr = g + 8 * (t & 1);  // lane t = 0: row g's code, t = 1: row g + 8's
    bool wsok = wsr < nw;
    for (int i = 0; i < NS; ++i) {
        v41_cp_wait<P::PS - 2>();
        __syncwarp();  // stage i landed for every lane; stage i - 1's readers are done
        if (i + P::PS - 1 < NS) issue(i + P::PS - 1);
        v41_cp_commit();
        const uint8_t* st = ring + (i % P::PS) * STAGE;
        const uint8_t* sw = st;
        const uint8_t* sa = st + P::W_B;
        const uint8_t* sws = st + WS_O;
        const uint32_t* sas = (const uint32_t*)(st + AS_O);
        uint32_t wsc = 0;
        if (wsok) {
            if (WF == 2) {
                const uint8_t* p = sws + wsr * 8;
                wsc = v41_mint_ue8(p[0], bscale2) | (v41_mint_ue8(p[2], bscale2) << 8) |
                      (v41_mint_ue8(p[4], bscale2) << 16) | (v41_mint_ue8(p[6], bscale2) << 24);
            } else {
                wsc = *(const uint32_t*)(sws + wsr * 4);
            }
        }
        uint32_t asc[MR];
#pragma unroll
        for (int mr = 0; mr < MR; ++mr) asc[mr] = mr * 8 + g < M ? sas[mr * 8 + g] : 0u;  // lane t = 0: row g's
#pragma unroll
        for (int kk = 0; kk < P::PK; ++kk) {
            uint32_t af[4] = {0u, 0u, 0u, 0u};
            if (WF == 0) {
                if (w0) {
                    af[0] = v41_lds32(sw + g * P::WR + kk * 32 + 4 * t);
                    af[2] = v41_lds32(sw + g * P::WR + kk * 32 + 16 + 4 * t);
                }
                if (w1) {
                    af[1] = v41_lds32(sw + (g + 8) * P::WR + kk * 32 + 4 * t);
                    af[3] = v41_lds32(sw + (g + 8) * P::WR + kk * 32 + 16 + 4 * t);
                }
            } else {
                if (w0) {
                    af[0] = v41_fp4_to_e4m3x4(*(const uint16_t*)(sw + g * P::WR + kk * 16 + 2 * t));
                    af[2] = v41_fp4_to_e4m3x4(*(const uint16_t*)(sw + g * P::WR + kk * 16 + 8 + 2 * t));
                }
                if (w1) {
                    af[1] = v41_fp4_to_e4m3x4(*(const uint16_t*)(sw + (g + 8) * P::WR + kk * 16 + 2 * t));
                    af[3] = v41_fp4_to_e4m3x4(*(const uint16_t*)(sw + (g + 8) * P::WR + kk * 16 + 8 + 2 * t));
                }
            }
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                uint32_t b0 = 0u, b1 = 0u;
                if (mr * 8 + g < M) {
                    const uint8_t* ar = sa + (mr * 8 + g) * P::AR + kk * 32 + 4 * t;
                    b0 = v41_lds32(ar);
                    b1 = v41_lds32(ar + 16);
                }
                v41_mma_bs(acc[mr], af, b0, b1, wsc, asc[mr], kk);
            }
        }
    }
    v41_cp_wait<0>();
}

// The warp's results into C (bf16 or fp32), rows 0 .. M - 1, columns rw0 .. (< N).
template <int OUTF, int CW, int MR>
__device__ __forceinline__ void v41_sk_store(void* __restrict__ C, long ldc, int M, int N, int rw0,
                                             const float acc[MR][4]) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mr = 0; mr < MR; ++mr)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int wr = g + (e >= 2 ? 8 : 0), row = mr * 8 + 2 * t + (e & 1), col = rw0 + wr;
            if (wr < CW && row < M && col < N) {
                if (OUTF == 0) ((v41_bf*)C)[(long)row * ldc + col] = v41_f2bf(acc[mr][e]);
                else ((float*)C)[(long)row * ldc + col] = acc[mr][e];
            }
        }
}

// The dense small-row GEMM: block x = NW warps of CW output columns each.
template <int WF, int OUTF, int CW, int MR>
__global__ void __launch_bounds__(128) v41_gemv_bs_kernel(const uint8_t* __restrict__ A, long lda,
                                                          const uint8_t* __restrict__ As, long ldas,
                                                          const uint8_t* __restrict__ B, long ldb,
                                                          const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                          void* __restrict__ C, long ldc, int M, int N, int K) {
    using P = V41Sk<WF, CW, MR>;
    extern __shared__ __align__(16) uint8_t v41_sk_sm[];
    v41_smem_poison(v41_sk_sm);
    int warp = threadIdx.x >> 5;
    int rw0 = (blockIdx.x * P::NW + warp) * CW;
    float acc[MR][4];
    v41_sk_chain<WF, CW, MR>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, M, N, K, rw0, M, v41_sk_sm + warp * P::warp(M),
                             acc);
    v41_sk_store<OUTF, CW, MR>(C, ldc, M, N, rw0, acc);
}

// The small-row tile list (every tile at most V41_SM_MAX rows): block y runs tile y (none past
// `*ntiles`), its expert's projection `proj`, block x NW warps of CW output columns each.
template <int WF, int OUTF, int CW>
__global__ void __launch_bounds__(128) v41_gemv_bs_tiles_kernel(const uint8_t* __restrict__ A, long lda,
                                                                const uint8_t* __restrict__ As, long ldas,
                                                                const V41Tile* __restrict__ tiles,
                                                                const int* __restrict__ ntiles,
                                                                const V41ExpW* __restrict__ ew, int proj, long ldb,
                                                                long ldbs, void* __restrict__ C, long ldc, int N, int K,
                                                                int BM) {
    using P = V41Sk<WF, CW, 1>;
    extern __shared__ __align__(16) uint8_t v41_sk_sm[];
    v41_smem_poison(v41_sk_sm);
    // the tile is loaded with the count, not after it: one memory round trip, not two (an entry past
    // the count is inside the list's allocation and is not used)
    const V41Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const V41ExpW& w = ew[tl.expert];
    long t0 = (long)tl.row0 + tl.m0;
    int M = min(tl.rows - tl.m0, BM);
    int warp = threadIdx.x >> 5;
    int rw0 = (blockIdx.x * P::NW + warp) * CW;
    float acc[1][4];
    v41_sk_chain<WF, CW, 1>(A + t0 * lda, lda, As + t0 * ldas, ldas, w.w[proj], ldb, w.s[proj], ldbs, w.scale2[proj], M,
                            N, K, rw0, BM, v41_sk_sm + warp * P::warp(BM), acc);
    void* c = OUTF ? (void*)((float*)C + t0 * ldc) : (void*)((v41_bf*)C + t0 * ldc);
    v41_sk_store<OUTF, CW, 1>(c, ldc, M, N, rw0, acc);
}

// The SMs of the current device (cached a device).
static int v41_sms() {
    static int sms[64] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!sms[dev & 63]) cudaDeviceGetAttribute(&sms[dev & 63], cudaDevAttrMultiProcessorCount, dev);
    return sms[dev & 63] > 0 ? sms[dev & 63] : 1;
}
// A warp's output columns for n columns over `groups` row groups (tiles): 16 while that gives at least four
// warps an SM, else 8, else 4 (a narrow output still spreads over the card; the empty rows of the MMA are
// free next to the weight stream).
static int v41_sk_cw(int n, int groups) {
    long want = 4L * v41_sms();
    if ((long)((n + 15) / 16) * groups >= want) return 16;
    if ((long)((n + 7) / 8) * groups >= want) return 8;
    return 4;
}

// The pipelined tile list (64- or 128-row tiles): block y runs tile y (none past `*ntiles`), its expert's
// projection `proj` from the weight table.
template <int WF, int OUTF, int BM>
__global__ void __launch_bounds__(256, BM == 64 ? 2 : 1) v41_gemm_f8_pipe_tiles_kernel(const uint8_t* __restrict__ A, long lda,
                                                                     const uint8_t* __restrict__ As, long ldas,
                                                                     const V41Tile* __restrict__ tiles,
                                                                     const int* __restrict__ ntiles,
                                                                     const V41ExpW* __restrict__ ew, int proj,
                                                                     long ldb, long ldbs, void* __restrict__ C,
                                                                     long ldc, int N, int K,
                                                                     const int* __restrict__ ridx) {
    extern __shared__ __align__(16) uint8_t v41_pipe_sm[];
    v41_smem_poison(v41_pipe_sm);
    // the tile is loaded with the count, not after it: one memory round trip, not two (an entry past
    // the count is inside the list's allocation and is not used)
    const V41Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const V41ExpW& w = ew[tl.expert];
    void* c = OUTF ? (void*)((float*)C + (long)tl.row0 * ldc) : (void*)((v41_bf*)C + (long)tl.row0 * ldc);
    // ridx: the list's row r (a pair, C's row) reads row ridx[r] of A and As (a token's codes, quantized once)
    long a0 = ridx ? 0 : (long)tl.row0;
    v41_gemm_f8_pipe_tile<WF, OUTF, BM>(A + a0 * lda, lda, As + a0 * ldas, ldas, w.w[proj], ldb, w.s[proj], ldbs,
                                        w.scale2[proj], c, ldc, tl.rows, N, K, tl.m0, blockIdx.x * 128, v41_pipe_sm,
                                        ridx ? ridx + tl.row0 : nullptr);
}

// bm: the plan's tile rows (64 or 128: the pipelined body; <= V41_SM_MAX: every tile has at most that
// many rows and the small-row body runs); max_tiles bounds *ntiles.
// ridx (the pipelined body only): the list's row r reads row ridx[r] of a and as.
static int v41_gemm_f8_tiles_impl(const void* a, long lda, const void* as, long ldas, const void* tiles,
                                  const int* ntiles, int max_tiles, const void* ew, int proj, long ldb, long ldbs,
                                  int wfmt, void* c, long ldc, int out_f32, int n, int k, int bm, const int* ridx,
                                  void* stream) {
    if (k % 32 || n <= 0 || lda % 16 || max_tiles < 0 || max_tiles > 65535 || proj < 0 || proj > 2) return 41026;
    if (ldb % 16 || wfmt < 0 || wfmt > 2 || (uintptr_t)a % 16) return 41026;
    if (ridx && bm != 64 && bm != 128) return 41026;
    if (max_tiles == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as;
    const V41Tile* T = (const V41Tile*)tiles;
    const V41ExpW* E = (const V41ExpW*)ew;
    if (bm <= V41_SM_MAX) {
        if ((k / 32) % 4 || ldas % 4 || ldbs % 4) return 41026;
        int cw = v41_sk_cw(n, max_tiles);
#define V41_SKT(WF, OF, CWX)                                                                                   \
    do {                                                                                                       \
        int sm_ = V41Sk<WF, CWX, 1>::block(bm);                                                                \
        if (sm_ > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_bs_tiles_kernel<WF, OF, CWX>), (V41Sk<WF, CWX, 1>::block(V41_SM_MAX))); \
        dim3 gr((n + V41Sk<WF, CWX, 1>::NW * CWX - 1) / (V41Sk<WF, CWX, 1>::NW * CWX), max_tiles);             \
        v41_gemv_bs_tiles_kernel<WF, OF, CWX><<<gr, 128, sm_, st>>>(A, lda, As, ldas, T, ntiles, E, proj, ldb, \
                                                                     ldbs, c, ldc, n, k, bm);                  \
    } while (0)
#define V41_SKT_CW(WF, OF) \
    do { if (cw == 16) V41_SKT(WF, OF, 16); else if (cw == 8) V41_SKT(WF, OF, 8); else V41_SKT(WF, OF, 4); } while (0)
        if (wfmt == 0) { if (out_f32) V41_SKT_CW(0, 1); else V41_SKT_CW(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_SKT_CW(1, 1); else V41_SKT_CW(1, 0); }
        else { if (out_f32) V41_SKT_CW(2, 1); else V41_SKT_CW(2, 0); }
#undef V41_SKT_CW
#undef V41_SKT
        V41_ERR();
    }
    if (bm != 64 && bm != 128) return 41026;
#define V41_PIPE_TL(WF, OF, BMX)                                                                               \
    do {                                                                                                       \
        int sm_ = V41Pipe<WF, BMX>::SMEM;                                                                      \
        V41_SMEM_LIMIT((v41_gemm_f8_pipe_tiles_kernel<WF, OF, BMX>), sm_);                                     \
        V41_CARVE_MAX((v41_gemm_f8_pipe_tiles_kernel<WF, OF, BMX>));                                           \
        dim3 gp((n + 127) / 128, max_tiles);                                                                   \
        v41_gemm_f8_pipe_tiles_kernel<WF, OF, BMX><<<gp, 256, sm_, st>>>(A, lda, As, ldas, T, ntiles, E, proj, ldb, \
                                                                          ldbs, c, ldc, n, k, ridx);           \
    } while (0)
#define V41_PIPE_TL_BM(WF, OF) \
    do { if (bm == 128) V41_PIPE_TL(WF, OF, 128); else V41_PIPE_TL(WF, OF, 64); } while (0)
    if (wfmt == 0) { if (out_f32) V41_PIPE_TL_BM(0, 1); else V41_PIPE_TL_BM(0, 0); }
    else if (wfmt == 1) { if (out_f32) V41_PIPE_TL_BM(1, 1); else V41_PIPE_TL_BM(1, 0); }
    else { if (out_f32) V41_PIPE_TL_BM(2, 1); else V41_PIPE_TL_BM(2, 0); }
#undef V41_PIPE_TL_BM
#undef V41_PIPE_TL
    V41_ERR();
}

extern "C" int memra_v41_gemm_f8_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                                       const int* ntiles, int max_tiles, const void* ew, int proj, long ldb, long ldbs,
                                       int wfmt, void* c, long ldc, int out_f32, int n, int k, int bm, void* stream) {
    return v41_gemm_f8_tiles_impl(a, lda, as, ldas, tiles, ntiles, max_tiles, ew, proj, ldb, ldbs, wfmt, c, ldc,
                                  out_f32, n, k, bm, nullptr, stream);
}

// The tile list over activations quantized once a token: the list's row r (a pair, and c's row) reads row
// ridx[r] of a and as (the pair's token). Tiles of 64 or 128 rows; the bytes are those of the list run over
// the pairs' gathered copies, which hold the same codes.
extern "C" int memra_v41_gemm_f8_tiles_rows(const void* a, long lda, const void* as, long ldas, const int* ridx,
                                            const void* tiles, const int* ntiles, int max_tiles, const void* ew,
                                            int proj, long ldb, long ldbs, int wfmt, void* c, long ldc, int out_f32,
                                            int n, int k, int bm, void* stream) {
    if (!ridx) return 41026;
    return v41_gemm_f8_tiles_impl(a, lda, as, ldas, tiles, ntiles, max_tiles, ew, proj, ldb, ldbs, wfmt, c, ldc,
                                  out_f32, n, k, bm, ridx, stream);
}

extern "C" int memra_v41_gemm_f8(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                 const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                 int m, int n, int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || lda % 16 || ldb % 16 || wfmt < 0 || wfmt > 2) return 41005;
    // every body loads a and b 16 bytes at a time: an unaligned base is refused, not read wrong
    if (((uintptr_t)a | (uintptr_t)b) % 16) return 41005;
    if (m == 0) return 0;
    bool small = (k / 32) % 4 == 0 && ldas % 4 == 0 && ldbs % 4 == 0;
    if (small && m > 32 && m <= v41_wide(0)) {
        // the small-row body, a launch a group of 32 rows
        long ce = out_f32 ? 4 : 2;
        for (int m0 = 0; m0 < m; m0 += 32) {
            int mm = m - m0 < 32 ? m - m0 : 32;
            int rc = memra_v41_gemm_f8((const uint8_t*)a + (long)m0 * lda, lda, (const uint8_t*)as + (long)m0 * ldas,
                                       ldas, b, ldb, bs, ldbs, bscale2, wfmt, (uint8_t*)c + (long)m0 * ldc * ce, ldc,
                                       out_f32, mm, n, k, stream);
            if (rc) return rc;
        }
        return 0;
    }
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as, *B = (const uint8_t*)b, *Bs = (const uint8_t*)bs;
    if (small && m <= 32 && (m <= V41_SM_MAX || m <= v41_wide(0))) {
        int cw = v41_sk_cw(n, 1), mr = (m + 7) / 8;
        if (mr == 3) mr = 4;
#define V41_SK(WF, OF, CWX, MRX)                                                                                \
    do {                                                                                                        \
        int sm_ = V41Sk<WF, CWX, MRX>::block(m);                                                                \
        if (sm_ > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_bs_kernel<WF, OF, CWX, MRX>), (V41Sk<WF, CWX, MRX>::block(8 * MRX))); \
        dim3 gr((n + V41Sk<WF, CWX, MRX>::NW * CWX - 1) / (V41Sk<WF, CWX, MRX>::NW * CWX));                     \
        v41_gemv_bs_kernel<WF, OF, CWX, MRX><<<gr, V41Sk<WF, CWX, MRX>::NW * 32, sm_, st>>>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, \
                                                                    c, ldc, m, n, k);                           \
    } while (0)
#define V41_SK_MR(WF, OF, CWX) \
    do { if (mr == 1) V41_SK(WF, OF, CWX, 1); else if (mr == 2) V41_SK(WF, OF, CWX, 2); else V41_SK(WF, OF, CWX, 4); } while (0)
#define V41_SK_CW(WF, OF) \
    do { if (cw == 16) V41_SK_MR(WF, OF, 16); else if (cw == 8) V41_SK_MR(WF, OF, 8); else V41_SK_MR(WF, OF, 4); } while (0)
        if (wfmt == 0) { if (out_f32) V41_SK_CW(0, 1); else V41_SK_CW(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_SK_CW(1, 1); else V41_SK_CW(1, 0); }
        else { if (out_f32) V41_SK_CW(2, 1); else V41_SK_CW(2, 0); }
#undef V41_SK_CW
#undef V41_SK_MR
#undef V41_SK
        V41_ERR();
    }
    {
        // the pipelined tile: 128 rows once there are that many, else 64
#define V41_PIPE_F8(WF, OF, BMX)                                                                                \
    do {                                                                                                        \
        int sm_ = V41Pipe<WF, BMX>::SMEM;                                                                       \
        V41_SMEM_LIMIT((v41_gemm_f8_pipe_kernel<WF, OF, BMX>), sm_);                                            \
        V41_CARVE_MAX((v41_gemm_f8_pipe_kernel<WF, OF, BMX>));                                                  \
        dim3 gp((n + 127) / 128, (m + BMX - 1) / BMX);                                                          \
        v41_gemm_f8_pipe_kernel<WF, OF, BMX><<<gp, 256, sm_, st>>>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, \
                                                                    c, ldc, m, n, k);                           \
    } while (0)
// (a call of at most 32 rows: a 32-row tile, where all 8 warps hold rows; in a 64-row tile half of them
// would hold padding)
#define V41_PIPE_F8_BM(WF, OF)                                \
    do {                                                      \
        if (m >= 128) V41_PIPE_F8(WF, OF, 128);               \
        else if (m <= 32) V41_PIPE_F8(WF, OF, 32);            \
        else V41_PIPE_F8(WF, OF, 64);                         \
    } while (0)
        if (wfmt == 0) { if (out_f32) V41_PIPE_F8_BM(0, 1); else V41_PIPE_F8_BM(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_PIPE_F8_BM(1, 1); else V41_PIPE_F8_BM(1, 0); }
        else { if (out_f32) V41_PIPE_F8_BM(2, 1); else V41_PIPE_F8_BM(2, 0); }
#undef V41_PIPE_F8_BM
#undef V41_PIPE_F8
        V41_ERR();
    }
}

// The reference schedule (gates only): the same GEMM through `v41_gemm_f8_kernel`, any row count.
extern "C" int memra_v41_gemm_f8_ref(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                     const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                     int m, int n, int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || lda % 16) return 41005;
    if (wfmt == 0 && ldb % 16) return 41005;
    if (wfmt != 0 && ldb % 8) return 41005;
    if (m == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as, *B = (const uint8_t*)b, *Bs = (const uint8_t*)bs;
    dim3 grid((n + V41_GN - 1) / V41_GN, (m + V41_GM - 1) / V41_GM);
#define V41_GEMM_F8(WF, OF) \
    v41_gemm_f8_kernel<WF, OF><<<grid, 256, 0, st>>>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, c, ldc, m, n, k)
    if (wfmt == 0) { if (out_f32) V41_GEMM_F8(0, 1); else V41_GEMM_F8(0, 0); }
    else if (wfmt == 1) { if (out_f32) V41_GEMM_F8(1, 1); else V41_GEMM_F8(1, 0); }
    else if (wfmt == 2) { if (out_f32) V41_GEMM_F8(2, 1); else V41_GEMM_F8(2, 0); }
    else return 41005;
#undef V41_GEMM_F8
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// The order of a dot product over K (the K-segment order). K is cut into V41_KSEG contiguous segments
// of K / V41_KSEG; a segment's sum is one chain in K order with an fp32 accumulator that starts at
// zero; the segment sums are added by one fixed binary tree in index order,
//   ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)),
// in fp32, and rounded to the output type once. Every kernel that takes the form uses these two
// helpers, at every row count and rank count, so a row's bytes do not depend on the path that
// computed it. A K that the segments do not divide into whole steps keeps one segment (the chain).
// WHY: one chain of K / 16 dependent MMAs is bound by the chain's latency; eight chains run through
// the tensor unit side by side.
#define V41_KSEG 8

// k16 steps a segment, or 0 when K takes one chain (`step`: the kernel's K step in elements)
__device__ __forceinline__ int v41_kseg_steps(int K, int step) {
    return (K % (V41_KSEG * step) == 0 && K >= V41_KSEG * step) ? K / (V41_KSEG * 16) : 0;
}

// the tree over the segment sums, element by element
__device__ __forceinline__ float v41_ktree(float s0, float s1, float s2, float s3, float s4, float s5, float s6,
                                           float s7) {
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// ---------------------------------------------------------------------------------------------
// BF16 GEMM with fp32 accumulation (F.linear on bf16 weights, and on bf16 values promoted to
// fp32: bf16 x bf16 products are exact in fp32). C[z][M,N] = A[z][M,K] . B[z][N,K]^T, batched
// over grid.z with element strides. OUTF 0 = bf16, 1 = fp32. Tile 64x128x32, 8 warps. The sum over K
// is in the K-segment order: the tile walks K once, and at the end of a segment folds the segment's
// sum into the tree's saved partials (p1: an odd segment's left neighbour, p2: a pair, p4: the first
// four).

#define V41_HS 80  // smem bytes per 32-element bf16 row

template <int OUTF>
__global__ void __launch_bounds__(256) v41_gemm_bf16_kernel(const v41_bf* __restrict__ A, long lda, long sAz,
                                                            const v41_bf* __restrict__ B, long ldb, long sBz,
                                                            void* __restrict__ C, long ldc, long sCz, int M, int N,
                                                            int K) {
    __shared__ __align__(16) uint8_t sA[V41_GM * V41_HS];
    __shared__ __align__(16) uint8_t sB[V41_GN * V41_HS];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int g = lane >> 2, t = lane & 3;
    int wm = warp >> 2, wn = warp & 3;
    int m0 = blockIdx.y * V41_GM, n0 = blockIdx.x * V41_GN;
    A += (long)blockIdx.z * sAz;
    B += (long)blockIdx.z * sBz;
    float acc[2][4][4], p1[2][4][4], p2[2][4][4], p4[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = p1[i][j][e] = p2[i][j][e] = p4[i][j][e] = 0.f;
    // elements a segment (0: one chain over K)
    int seg = v41_kseg_steps(K, 32) * 16;
    int slen = seg ? seg : K;
    // the K loop a segment at a time (one pass of the outer loop when K takes one chain), so that each
    // fold names its place in the tree at compile time
#pragma unroll
    for (int sidx = 0; sidx < V41_KSEG; ++sidx) {
        if (sidx * (long)slen >= K) break;
        for (int k0 = sidx * slen; k0 < (sidx + 1) * slen; k0 += 32) {
            {  // A: 64 rows x 64 bytes = 256 x 16 B
                int r = tid >> 2, q = tid & 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m0 + r < M) v = *(const uint4*)(A + (long)(m0 + r) * lda + k0 + q * 8);
                *(uint4*)(&sA[r * V41_HS + q * 16]) = v;
            }
#pragma unroll
            for (int rep = 0; rep < 2; ++rep) {  // B: 128 rows x 64 bytes = 512 x 16 B
                int i = tid + rep * 256;
                int r = i >> 2, q = i & 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (n0 + r < N) v = *(const uint4*)(B + (long)(n0 + r) * ldb + k0 + q * 8);
                *(uint4*)(&sB[r * V41_HS + q * 16]) = v;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                int kb = kk * 32;  // byte offset of this k16 half
                // the fragments through ldmatrix, four tiles per load
                uint32_t af[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) v41_ldsm_a(af[mi], sA, wm * 32 + mi * 16, V41_HS, kb);
#pragma unroll
                for (int nj = 0; nj < 2; ++nj) {
                    uint32_t bf[4];
                    v41_ldsm_b2(bf, sB, wn * 32 + nj * 16, V41_HS, kb);
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        v41_mma_bf16(acc[mi][2 * nj], af[mi], bf[0], bf[1]);
                        v41_mma_bf16(acc[mi][2 * nj + 1], af[mi], bf[2], bf[3]);
                    }
                }
            }
            __syncthreads();
        }
        if (seg) {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float v = acc[i][j][e];
                        if (sidx == 0 || sidx == 2 || sidx == 4 || sidx == 6) {
                            p1[i][j][e] = v;  // a pair's left segment
                            v = 0.f;
                        } else if (sidx == 1 || sidx == 5) {
                            p2[i][j][e] = p1[i][j][e] + v;  // the first pair of a four
                            v = 0.f;
                        } else if (sidx == 3) {
                            p4[i][j][e] = p2[i][j][e] + (p1[i][j][e] + v);  // the first four
                            v = 0.f;
                        } else {
                            v = p4[i][j][e] + (p2[i][j][e] + (p1[i][j][e] + v));  // the result
                        }
                        acc[i][j][e] = v;
                    }
        }
    }
    char* Cz = (char*)C + (long)blockIdx.z * sCz * (OUTF == 0 ? 2 : 4);
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int row = m0 + wm * 32 + mi * 16 + g + (e >= 2 ? 8 : 0);
                int col = n0 + wn * 32 + ni * 8 + 2 * t + (e & 1);
                if (row < M && col < N) {
                    if (OUTF == 0) ((v41_bf*)Cz)[(long)row * ldc + col] = v41_f2bf(acc[mi][ni][e]);
                    else ((float*)Cz)[(long)row * ldc + col] = acc[mi][ni][e];
                }
            }
}

// Small-M bf16 GEMM (at most V41_SM_MAX rows): the tile kernel's sum exactly. Its fp32 sum runs
// inside each m16n8k16 MMA over the K steps in order, on the same fragment lanes as the tile (rows
// 0..7 of the tile, column = lane group); a warp owns an 8-column group. The columns and the batch
// give the parallelism. A warp lands its 8 weight rows and the M rows in shared memory in 16-byte
// chunks through cp.async, `su` k16 steps per stage, and reads the fragments from there.
// SEG 0 (a K the segments do not divide: one chain): one warp sums the whole K. A few-column output
// (few warps in the whole launch) is bound by the latency of one fetch after another, so there a
// warp is its own block and lands its whole K slice at once (one stage); a wide output streams
// through a two-stage ring of 8 steps, 4 warps per block.
// SEG 1 (the K-segment order): a block is V41_KSEG warps over one 8-column group, warp w sums
// segment w (its K / V41_KSEG elements, contiguous, a chain from zero in K order), and the segment
// sums meet in shared memory in v41_ktree, in index order. Every warp streams its own contiguous
// slice of each weight row (full lines a stage), and the eight chains run side by side.
#define V41_SMB_U 8

static long v41_gemv_bf16_warp_smem(int m, int su, int stages) {
    return (long)stages * (m + 8) * (su * 32 + 16);
}

// R16: a call of 9 to 16 rows. The tile's 16-wide side then holds 16 rows (rows g + 8 on the same lanes), so
// a warp's chain over K serves twice the rows with the same MMAs and the same weight fetches.
template <int OUTF, int R16 = 0, int SEG = 0>
__global__ void __launch_bounds__(SEG ? 32 * V41_KSEG : 128)
    v41_gemv_bf16_kernel(const v41_bf* __restrict__ A, long lda, long sAz, const v41_bf* __restrict__ B, long ldb,
                         long sBz, void* __restrict__ C, long ldc, long sCz, int M, int N, int K, int su,
                         int stages) {
    extern __shared__ __align__(16) uint8_t v41_smb[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n0 = SEG ? blockIdx.x * 8 : (blockIdx.x * (blockDim.x >> 5) + warp) * 8;
    if (!SEG && n0 >= N) return;  // the whole warp (SEG: the grid has no such block)
    const uint8_t* Ab = (const uint8_t*)(A + (long)blockIdx.z * sAz);
    const uint8_t* Bb = (const uint8_t*)(B + (long)blockIdx.z * sBz);
    bool rowok = g < M;
    bool rowok8 = R16 && g + 8 < M;
    // the warp's part of K: [kb, kb + KL)
    int KL = SEG ? K / V41_KSEG : K;
    long kb = SEG ? (long)warp * KL : 0;
    int P = su * 32 + 16;        // bytes per staged row (+16: bank spread)
    int SB = (M + 8) * P;        // one stage: the M rows, then the 8 weight rows
    uint8_t* wsm = v41_smb + (long)warp * stages * SB;
    int CH = 2 * su;  // 16-byte chunks (8 elements) per row and stage
    auto issue = [&](int i) {
        uint8_t* st = wsm + (i % stages) * SB;
        long k0 = (long)i * su * 16;
        for (int q = lane; q < 8 * CH; q += 32) {
            int r = q / CH, c = q % CH;
            bool ok = (n0 + r < N) && (k0 + c * 8 < KL);
            const uint8_t* src = ok ? Bb + ((long)(n0 + r) * ldb + kb + k0 + c * 8) * 2 : Bb;
            v41_cp16(st + (M + r) * P + c * 16, src, ok);
        }
        for (int q = lane; q < M * CH; q += 32) {
            int r = q / CH, c = q % CH;
            bool ok = k0 + c * 8 < KL;
            const uint8_t* src = ok ? Ab + ((long)r * lda + kb + k0 + c * 8) * 2 : Ab;
            v41_cp16(st + r * P + c * 16, src, ok);
        }
    };
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int ns = (KL + su * 16 - 1) / (su * 16);
    issue(0);
    v41_cp_commit();
    for (int i = 0; i < ns; ++i) {
        if (i + 1 < ns) issue(i + 1);  // ns > 1 only with a two-stage ring
        v41_cp_commit();
        v41_cp_wait<1>();
        __syncwarp();  // stage i landed for every lane
        const uint8_t* st = wsm + (i % stages) * SB;
        const uint8_t* sA = st + (rowok ? g : 0) * P;
        const uint8_t* sA8 = st + (rowok8 ? g + 8 : 0) * P;
        const uint8_t* sB = st + (M + g) * P;
        int k0 = i * su * 16;
        for (int u0 = 0; u0 < su; u0 += 8) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                int u = u0 + v;
                if (u >= su || k0 + 16 * u >= KL) break;
                uint32_t af[4] = {0u, 0u, 0u, 0u};
                if (rowok) {
                    af[0] = v41_lds32(sA + u * 32 + t * 4);
                    af[2] = v41_lds32(sA + u * 32 + 16 + t * 4);
                }
                if (rowok8) {
                    af[1] = v41_lds32(sA8 + u * 32 + t * 4);
                    af[3] = v41_lds32(sA8 + u * 32 + 16 + t * 4);
                }
                uint32_t b0 = v41_lds32(sB + u * 32 + t * 4);
                uint32_t b1 = v41_lds32(sB + u * 32 + 16 + t * 4);
                v41_mma_bf16(acc, af, b0, b1);
            }
        }
    }
    v41_cp_wait<0>();
    if (SEG) {
        // the segment sums, [segment][element][lane] after the stages, into the tree on warp 0
        float* red = (float*)(v41_smb + (long)V41_KSEG * stages * SB);
#pragma unroll
        for (int e = 0; e < 4; ++e) red[(warp * 4 + e) * 32 + lane] = acc[e];
        __syncthreads();
        if (warp != 0) return;
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[e] = v41_ktree(red[(0 * 4 + e) * 32 + lane], red[(1 * 4 + e) * 32 + lane], red[(2 * 4 + e) * 32 + lane],
                               red[(3 * 4 + e) * 32 + lane], red[(4 * 4 + e) * 32 + lane], red[(5 * 4 + e) * 32 + lane],
                               red[(6 * 4 + e) * 32 + lane], red[(7 * 4 + e) * 32 + lane]);
    }
    if (rowok) {
        char* Cz = (char*)C + (long)blockIdx.z * sCz * (OUTF == 0 ? 2 : 4);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            int cn = n0 + 2 * t + e;
            if (cn < N) {
                if (OUTF == 0) ((v41_bf*)Cz)[(long)g * ldc + cn] = v41_f2bf(acc[e]);
                else ((float*)Cz)[(long)g * ldc + cn] = acc[e];
                if (rowok8) {
                    // acc[2], acc[3]: row g + 8 against the same two columns
                    if (OUTF == 0) ((v41_bf*)Cz)[(long)(g + 8) * ldc + cn] = v41_f2bf(acc[2 + e]);
                    else ((float*)Cz)[(long)(g + 8) * ldc + cn] = acc[2 + e];
                }
            }
        }
    }
}

extern "C" int memra_v41_gemm_bf16(const void* a, long lda, long sAz, const void* b, long ldb, long sBz, void* c,
                                   long ldc, long sCz, int out_f32, int m, int n, int k, int batch, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || batch <= 0 || lda % 8 || ldb % 8 || sAz % 8 || sBz % 8) return 41006;
    // every body lands a and b 16 bytes at a time (cp.async): an unaligned base is refused
    if (((uintptr_t)a | (uintptr_t)b) % 16) return 41006;
    if (m == 0) return 0;
    if (m > 2 * V41_SM_MAX && m <= v41_wide(1)) {
        // the small-row body, a launch a group of 16 rows (a batch's planes keep their strides)
        long ce = out_f32 ? 4 : 2;
        for (int m0 = 0; m0 < m; m0 += 2 * V41_SM_MAX) {
            int mm = m - m0 < 2 * V41_SM_MAX ? m - m0 : 2 * V41_SM_MAX;
            int rc = memra_v41_gemm_bf16((const v41_bf*)a + (long)m0 * lda, lda, sAz, b, ldb, sBz,
                                         (uint8_t*)c + (long)m0 * ldc * ce, ldc, sCz, out_f32, mm, n, k, batch, stream);
            if (rc) return rc;
        }
        return 0;
    }
    cudaStream_t st = (cudaStream_t)stream;
    bool r16 = m > V41_SM_MAX && m <= 2 * V41_SM_MAX && m <= v41_wide(1);
    if (m <= V41_SM_MAX || r16) {
        int su, stages, wpb, smem;
        dim3 grid;
        // the K-segment order (v41_kseg_steps' rule): V41_KSEG warps a block over one column group
        bool seg = k % (V41_KSEG * 32) == 0 && k >= V41_KSEG * 32;
        if (seg) {
            // a small launch keeps a warp's whole segment in flight (both stages of its ring), a wide one
            // streams V41_SMB_U steps a stage; fewer steps a stage when the rows need the room
            int ku = k / V41_KSEG / 16;  // k16 steps a segment
            long blocks = (long)((n + 7) / 8) * batch;
            su = (blocks <= 128 && ku > 1) ? (ku + 1) / 2 : (ku < V41_SMB_U ? ku : V41_SMB_U);
            stages = 2;
            while (su > 1 && V41_KSEG * v41_gemv_bf16_warp_smem(m, su, stages) + V41_KSEG * 512 > 96 * 1024)
                su = (su + 1) / 2;
            wpb = V41_KSEG;
            // + the tree's inputs: V41_KSEG warps x 4 sums x 32 lanes in fp32
            smem = (int)(V41_KSEG * v41_gemv_bf16_warp_smem(m, su, stages)) + V41_KSEG * 512;
            grid = dim3((n + 7) / 8, 1, batch);
        } else {
            // one warp per block with its whole K slice staged while the launch is at most 256 warps
            // and the slice fits a block; else 4 warps per block over the two-stage ring
            long whole = v41_gemv_bf16_warp_smem(m, k / 16, 1);
            long warps = (long)((n + 7) / 8) * batch;
            su = V41_SMB_U;
            stages = 2;
            wpb = 4;
            if (warps <= 256 && whole <= 96 * 1024) {
                su = k / 16;
                stages = 1;
                wpb = 1;
            }
            smem = (int)(wpb * v41_gemv_bf16_warp_smem(m, su, stages));
            grid = dim3((n + 8 * wpb - 1) / (8 * wpb), 1, batch);
        }
#define V41_GEMV_BF(OF, SG)                                                                                        \
    do {                                                                                                           \
        if (r16) {                                                                                                 \
            if (smem > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_bf16_kernel<OF, 1, SG>), 96 * 1024);                    \
            v41_gemv_bf16_kernel<OF, 1, SG><<<grid, 32 * wpb, smem, st>>>(                                         \
                (const v41_bf*)a, lda, sAz, (const v41_bf*)b, ldb, sBz, c, ldc, sCz, m, n, k, su, stages);         \
        } else {                                                                                                   \
            if (smem > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_bf16_kernel<OF, 0, SG>), 96 * 1024);                    \
            v41_gemv_bf16_kernel<OF, 0, SG><<<grid, 32 * wpb, smem, st>>>(                                         \
                (const v41_bf*)a, lda, sAz, (const v41_bf*)b, ldb, sBz, c, ldc, sCz, m, n, k, su, stages);         \
        }                                                                                                          \
    } while (0)
        if (seg) {
            if (out_f32) V41_GEMV_BF(1, 1); else V41_GEMV_BF(0, 1);
        } else {
            if (out_f32) V41_GEMV_BF(1, 0); else V41_GEMV_BF(0, 0);
        }
#undef V41_GEMV_BF
        V41_ERR();
    }
    dim3 grid((n + V41_GN - 1) / V41_GN, (m + V41_GM - 1) / V41_GM, batch);
    if (out_f32)
        v41_gemm_bf16_kernel<1><<<grid, 256, 0, st>>>((const v41_bf*)a, lda, sAz, (const v41_bf*)b, ldb, sBz, c, ldc,
                                                      sCz, m, n, k);
    else
        v41_gemm_bf16_kernel<0><<<grid, 256, 0, st>>>((const v41_bf*)a, lda, sAz, (const v41_bf*)b, ldb, sBz, c, ldc,
                                                      sCz, m, n, k);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// F.linear in fp32 on bf16 values (the ratio-2 compressor's wkv/wgate, the MoE router): every
// product is exact, the sum is an fp32 FMA chain in K order on CUDA cores, as in the reference's
// fp32 GEMM. Tensor-core accumulation is not used here: on the compressor it sat ~6x further from
// a float64 recomputation than the reference does (research/modelplan-onboarding-dsv41-20261109/gpu).
// C[M,N] f32, 64x64 tile, 256 threads x 4x4 outputs, K tiles of 32. WIDE (rows 16-byte aligned): a
// thread stages a K tile with one 16-byte load from A and one from B, both in flight, instead of 16
// two-byte loads one after another.

template <bool WIDE>
__global__ void __launch_bounds__(256) v41_gemm_ffma_kernel(const v41_bf* __restrict__ A, long lda,
                                                            const v41_bf* __restrict__ B, long ldb,
                                                            float* __restrict__ C, long ldc, int M, int N, int K) {
    __shared__ float sA[32][65];
    __shared__ float sB[32][65];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 32) {
        if (WIDE) {
            int row = tid >> 2, q = tid & 3;  // 8 elements of one row each
            uint4 av = make_uint4(0, 0, 0, 0), bv = make_uint4(0, 0, 0, 0);
            if (m0 + row < M) av = *(const uint4*)(A + (long)(m0 + row) * lda + k0 + q * 8);
            if (n0 + row < N) bv = *(const uint4*)(B + (long)(n0 + row) * ldb + k0 + q * 8);
            const v41_bf* ae = (const v41_bf*)&av;
            const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                sA[q * 8 + u][row] = v41_bf2f(ae[u]);
                sB[q * 8 + u][row] = v41_bf2f(be[u]);
            }
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int i = tid + r * 256;
                int row = i >> 5, kk = i & 31;
                sA[kk][row] = (m0 + row < M) ? v41_bf2f(A[(long)(m0 + row) * lda + k0 + kk]) : 0.f;
                sB[kk][row] = (n0 + row < N) ? v41_bf2f(B[(long)(n0 + row) * ldb + k0 + kk]) : 0.f;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = sA[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = sB[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = m0 + ty * 4 + i, col = n0 + tx * 4 + j;
            if (row < M && col < N) C[(long)row * ldc + col] = acc[i][j];
        }
}

// Small-M fp32 FMA GEMM (at most V41_SM_MAX rows): the tile kernel's chain exactly, one fma per K
// element in K order per output. A block stages 32 columns of B through shared memory as fp32,
// transposed so the 32 threads of a row read 32 neighbouring words, with the next chunk's loads in
// registers while the current chunk's chains run; thread (row, column) owns one chain.
#define V41_SMF_KC 256

// MR: the rows a block holds, 4 (128 threads) or 8 (256: a call of 5..8 rows in one launch)
template <int MR>
__global__ void __launch_bounds__(MR * 32) v41_gemv_ffma_kernel(const v41_bf* __restrict__ A, long lda,
                                                                const v41_bf* __restrict__ B, long ldb,
                                                                float* __restrict__ C, long ldc, int M, int N,
                                                                int K) {
    // B as [column][K chunk] and A as [row][K chunk], rows 16-byte aligned: a chain reads its 8 next
    // values of each with two 16-byte loads. At 5..8 rows every thread of the block runs a chain and
    // the block is bound by the instructions it issues, 4 loads for 8 fmas instead of 16.
    __shared__ __align__(16) float sB[32][V41_SMF_KC + 4];
    __shared__ __align__(16) float sA[MR][V41_SMF_KC];
    int tid = threadIdx.x, n0 = blockIdx.x * 32;
    // staging roles: B row j = tid >> 2, quarter q = tid & 3 (64 elements = 8 x 16 B), the first 128
    // threads; A: 16 B each
    int j = tid >> 2, q = tid & 3;
    bool bst = tid < 128;
    int ar = tid >> 5, ac = (tid & 31) * 8;  // A row and element offset (MR rows x 256 elements)
    uint4 nb[8], na;
    auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            nb[i] = make_uint4(0, 0, 0, 0);
            if (bst && n0 + j < N && k0 + q * 64 + i * 8 < K)
                nb[i] = __ldg((const uint4*)(B + (long)(n0 + j) * ldb + k0 + q * 64 + i * 8));
        }
        na = make_uint4(0, 0, 0, 0);
        if (ar < M && k0 + ac < K) na = __ldg((const uint4*)(A + (long)ar * lda + k0 + ac));
    };
    int r = tid >> 5, cc = tid & 31;
    float acc = 0.f;
    fetch(0);
    for (int k0 = 0; k0 < K; k0 += V41_SMF_KC) {
        __syncthreads();  // the previous chunk's chains are done with sB / sA
        if (bst) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const v41_bf* e = (const v41_bf*)&nb[i];
                float4* d = (float4*)&sB[j][q * 64 + i * 8];
                d[0] = make_float4(v41_bf2f(e[0]), v41_bf2f(e[1]), v41_bf2f(e[2]), v41_bf2f(e[3]));
                d[1] = make_float4(v41_bf2f(e[4]), v41_bf2f(e[5]), v41_bf2f(e[6]), v41_bf2f(e[7]));
            }
        }
        {
            const v41_bf* e = (const v41_bf*)&na;
            float4* d = (float4*)&sA[ar][ac];
            d[0] = make_float4(v41_bf2f(e[0]), v41_bf2f(e[1]), v41_bf2f(e[2]), v41_bf2f(e[3]));
            d[1] = make_float4(v41_bf2f(e[4]), v41_bf2f(e[5]), v41_bf2f(e[6]), v41_bf2f(e[7]));
        }
        __syncthreads();
        if (k0 + V41_SMF_KC < K) fetch(k0 + V41_SMF_KC);
        int kn = min(V41_SMF_KC, K - k0);
        if (r < M)
            for (int kk = 0; kk < kn; kk += 8) {
                const float4 a0 = *(const float4*)&sA[r][kk], a1 = *(const float4*)&sA[r][kk + 4];
                const float4 b0 = *(const float4*)&sB[cc][kk], b1 = *(const float4*)&sB[cc][kk + 4];
                acc = __fmaf_rn(a0.x, b0.x, acc);
                acc = __fmaf_rn(a0.y, b0.y, acc);
                acc = __fmaf_rn(a0.z, b0.z, acc);
                acc = __fmaf_rn(a0.w, b0.w, acc);
                acc = __fmaf_rn(a1.x, b1.x, acc);
                acc = __fmaf_rn(a1.y, b1.y, acc);
                acc = __fmaf_rn(a1.z, b1.z, acc);
                acc = __fmaf_rn(a1.w, b1.w, acc);
            }
    }
    if (r < M && n0 + cc < N) C[(long)r * ldc + n0 + cc] = acc;
}

// The same chain for a narrow output (the router, the ratio-2 compressor): a block owns 4 columns and
// lands their whole K slice and A's rows in shared memory with every copy in flight, then thread
// (row, column) runs its chain; with K dependent fmas per output the chain is the floor, so the
// columns spread over as many blocks as possible.
#define V41_SMF_NC 4

__global__ void __launch_bounds__(128) v41_gemv_ffma_narrow_kernel(const v41_bf* __restrict__ A, long lda,
                                                                   const v41_bf* __restrict__ B, long ldb,
                                                                   float* __restrict__ C, long ldc, int M, int N,
                                                                   int K) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    int pitch = K + 8;  // elements: rows 16 bytes apart in the bank map
    v41_bf* sB = (v41_bf*)v41_ffma_sm;
    v41_bf* sA = sB + V41_SMF_NC * pitch;
    int tid = threadIdx.x, n0 = blockIdx.x * V41_SMF_NC;
    int chunks = K / 8;
    for (int q = tid; q < V41_SMF_NC * chunks; q += 128) {
        int r = q / chunks, c = q % chunks;
        bool ok = n0 + r < N;
        v41_cp16(sB + r * pitch + c * 8, B + (long)(ok ? n0 + r : 0) * ldb + c * 8, ok);
    }
    for (int q = tid; q < M * chunks; q += 128) {
        int r = q / chunks, c = q % chunks;
        v41_cp16(sA + r * pitch + c * 8, A + (long)r * lda + c * 8, true);
    }
    v41_cp_commit();
    v41_cp_wait<0>();
    __syncthreads();
    if (tid >= M * V41_SMF_NC) return;
    int r = tid / V41_SMF_NC, cc = tid % V41_SMF_NC;
    if (n0 + cc >= N) return;
    const v41_bf* ar = sA + r * pitch;
    const v41_bf* br = sB + cc * pitch;
    float acc = 0.f;
    for (int k = 0; k < K; k += 8) {
        uint4 av = *(const uint4*)(ar + k), bv = *(const uint4*)(br + k);
        const v41_bf* ae = (const v41_bf*)&av;
        const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
        for (int u = 0; u < 8; ++u) acc = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), acc);
    }
    C[(long)r * ldc + n0 + cc] = acc;
}

// The narrow form with K in parts of Kp elements, for a call whose rows and columns of all of K do
// not fit a block (5..8 rows of K = 5120): a part is landed, every chain runs through it, the next
// part is landed; a chain is still its K fmas in K order.
__global__ void __launch_bounds__(128) v41_gemv_ffma_narrow_parts_kernel(const v41_bf* __restrict__ A, long lda,
                                                                         const v41_bf* __restrict__ B, long ldb,
                                                                         float* __restrict__ C, long ldc, int M, int N,
                                                                         int K, int Kp) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    int pitch = Kp + 8;  // elements: rows 16 bytes apart in the bank map
    v41_bf* sB = (v41_bf*)v41_ffma_sm;
    v41_bf* sA = sB + V41_SMF_NC * pitch;
    int tid = threadIdx.x, n0 = blockIdx.x * V41_SMF_NC;
    int r = tid / V41_SMF_NC, cc = tid % V41_SMF_NC;
    bool chain = tid < M * V41_SMF_NC && n0 + cc < N;
    const v41_bf* ar = sA + r * pitch;
    const v41_bf* br = sB + cc * pitch;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += Kp) {
        int kn = min(Kp, K - k0), chunks = kn / 8;
        for (int q = tid; q < V41_SMF_NC * chunks; q += 128) {
            int rr = q / chunks, c = q % chunks;
            bool ok = n0 + rr < N;
            v41_cp16(sB + rr * pitch + c * 8, B + (long)(ok ? n0 + rr : 0) * ldb + k0 + c * 8, ok);
        }
        for (int q = tid; q < M * chunks; q += 128) {
            int rr = q / chunks, c = q % chunks;
            v41_cp16(sA + rr * pitch + c * 8, A + (long)rr * lda + k0 + c * 8, true);
        }
        v41_cp_commit();
        v41_cp_wait<0>();
        __syncthreads();
        if (chain)
            for (int k = 0; k < kn; k += 8) {
                uint4 av = *(const uint4*)(ar + k), bv = *(const uint4*)(br + k);
                const v41_bf* ae = (const v41_bf*)&av;
                const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
                for (int u = 0; u < 8; ++u) acc = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), acc);
            }
        // the next part overwrites what the chains read
        if (k0 + Kp < K) __syncthreads();
    }
    if (chain) C[(long)r * ldc + n0 + cc] = acc;
}

// The same chain for a narrow output of up to 8 rows, its K streamed: a block is one warp that owns 4 columns
// and lane (row, column) = (lane / 4, lane % 4) runs the output's one fma chain over K in K order (the narrow
// kernel's bytes). K arrives through a ring of V41_RING_NS stages of V41_RING_KC elements (the 4 columns and
// the call's rows, lane l landing the l-th 16 bytes of each), stages in flight while the chain runs, and the
// chain reads its next 8 values of each operand one step ahead.
// WHY: an output is K dependent fmas whatever the kernel, so the chain's latency is the call's floor (4 cycles an
// fma on sm_120: 8.3 us for K = 5,120 on an RTX 5090); the narrow kernel added the wait for all of K to land and a
// shared-memory read the chain waited on at each turn of its loop (36 us a call of the router at one row). Here the
// first stage is the only wait, and a block is small enough that both outputs of the compressor's pair run in one
// launch (memra_v41_gemm_ffma2). Measured and dropped: a second warp widening each stage to fp32 for the chain's
// warp (22 us: the hand-over costs more than the widening).
#define V41_RING_KC 256
#define V41_RING_NS 4
#define V41_RING_PITCH (V41_RING_KC + 8)  // elements: a stage's rows 16 bytes apart in the bank map

// one output of a ring launch: the weight, the output and its width; `nblk` blocks of 4 columns
struct V41RingOut {
    const v41_bf* B;
    long ldb;
    float* C;
    long ldc;
    int N;
    int nblk;
};

__global__ void __launch_bounds__(32) v41_gemv_ffma_ring_kernel(const v41_bf* __restrict__ A, long lda, V41RingOut o0,
                                                                V41RingOut o1, int M, int K) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    v41_bf* sm = (v41_bf*)v41_ffma_sm;
    int b = blockIdx.x;
    const bool second = b >= o0.nblk;
    const v41_bf* __restrict__ B = second ? o1.B : o0.B;
    const long ldb = second ? o1.ldb : o0.ldb;
    float* __restrict__ C = second ? o1.C : o0.C;
    const long ldc = second ? o1.ldc : o0.ldc;
    const int N = second ? o1.N : o0.N;
    if (second) b -= o0.nblk;
    const int n0 = b * V41_SMF_NC, rows = V41_SMF_NC + M, lane = threadIdx.x, nk = K / V41_RING_KC;
    // stage t: rows 0..3 the columns, then the call's rows
    auto land = [&](int t) {
        v41_bf* base = sm + (t % V41_RING_NS) * rows * V41_RING_PITCH;
        long k0 = (long)t * V41_RING_KC + lane * 8;
#pragma unroll
        for (int row = 0; row < V41_SMF_NC; ++row) {
            bool ok = n0 + row < N;
            v41_cp16(base + row * V41_RING_PITCH + lane * 8, B + (long)(ok ? n0 + row : 0) * ldb + k0, ok);
        }
        for (int row = 0; row < M; ++row)
            v41_cp16(base + (V41_SMF_NC + row) * V41_RING_PITCH + lane * 8, A + (long)row * lda + k0, true);
    };
#pragma unroll
    for (int t = 0; t < V41_RING_NS - 1; ++t) {
        if (t < nk) land(t);
        v41_cp_commit();
    }
    const int r = lane / V41_SMF_NC, cc = lane % V41_SMF_NC;
    const bool chain = r < M && n0 + cc < N;
    float acc = 0.f;
    for (int t = 0; t < nk; ++t) {
        v41_cp_wait<V41_RING_NS - 2>();
        __syncwarp();  // stage t landed by every lane; every lane is done with stage t - 1's slot
        if (t + V41_RING_NS - 1 < nk) land(t + V41_RING_NS - 1);
        v41_cp_commit();
        if (chain) {
            const v41_bf* base = sm + (t % V41_RING_NS) * rows * V41_RING_PITCH;
            const v41_bf* ar = base + (V41_SMF_NC + r) * V41_RING_PITCH;
            const v41_bf* br = base + cc * V41_RING_PITCH;
            uint4 av = *(const uint4*)ar, bv = *(const uint4*)br;
#pragma unroll
            for (int k = 0; k < V41_RING_KC; k += 8) {
                uint4 an = av, bn = bv;
                if (k + 8 < V41_RING_KC) {
                    an = *(const uint4*)(ar + k + 8);
                    bn = *(const uint4*)(br + k + 8);
                }
                // a 32-bit word holds two bf16 values: the low one widened by a shift, the high one by a mask
                // (one instruction each, where a 16-bit pick and a shift took two)
                const uint32_t aw[4] = {av.x, av.y, av.z, av.w}, bw[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    acc = __fmaf_rn(__uint_as_float(aw[u] << 16), __uint_as_float(bw[u] << 16), acc);
                    acc = __fmaf_rn(__uint_as_float(aw[u] & 0xffff0000u), __uint_as_float(bw[u] & 0xffff0000u), acc);
                }
                av = an;
                bv = bn;
            }
        }
    }
    if (chain) C[(long)r * ldc + n0 + cc] = acc;
}

// the ring kernel takes the call: up to 8 rows, outputs up to 4096 wide, K whole stages, 16-byte rows
// The ring kernel is one warp a block: lane = row x V41_SMF_NC + column, so every row the dispatch admits
// (m <= V41_SM_MAX) must fit the 32 lanes.
static_assert(V41_SM_MAX * V41_SMF_NC <= 32, "v41_gemv_ffma_ring_kernel maps rows x columns onto one warp");
static bool v41_ring_fits(const void* a, long lda, const void* b, long ldb, int m, int n, int k) {
    return m >= 1 && m <= V41_SM_MAX && n <= 4096 && k % V41_RING_KC == 0 && lda % 8 == 0 && ldb % 8 == 0 &&
           ((uintptr_t)a | (uintptr_t)b) % 16 == 0;
}

static int v41_ring_launch(const void* a, long lda, V41RingOut o0, V41RingOut o1, int m, int k, void* stream) {
    int smem = V41_RING_NS * (V41_SMF_NC + m) * V41_RING_PITCH * 2;
    v41_gemv_ffma_ring_kernel<<<o0.nblk + o1.nblk, 32, smem, (cudaStream_t)stream>>>((const v41_bf*)a, lda, o0, o1, m,
                                                                                     k);
    V41_ERR();
}

extern "C" int memra_v41_gemm_ffma(const void* a, long lda, const void* b, long ldb, float* c, long ldc, int m, int n,
                                   int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0) return 41023;
    if (m == 0) return 0;
    if (v41_ring_fits(a, lda, b, ldb, m, n, k)) {
        // a narrow output (the router's gate, the compressor) of up to 8 rows: the chain with K streamed
        V41RingOut o0{(const v41_bf*)b, ldb, c, ldc, n, (n + V41_SMF_NC - 1) / V41_SMF_NC}, o1{nullptr, 0, nullptr, 0,
                                                                                            0, 0};
        return v41_ring_launch(a, lda, o0, o1, m, k, stream);
    }
    // the row forms load 16 bytes at a time from the rows of a and b: their bases too must be aligned
    bool al = lda % 8 == 0 && ldb % 8 == 0 && ((uintptr_t)a | (uintptr_t)b) % 16 == 0;
    if (m > V41_SM_MAX && m <= 32 && m <= v41_wide(2) && n <= 4096 && al) {
        // a narrow output (the router's gate, the compressor) at 9 to 32 rows: the narrow kernel's block is
        // already 4 columns x 32 chains, so the whole call is one launch with K in as many parts as let
        // the block hold the rows (a chain is still its K fmas in K order); else the groups below
        for (int parts = 2; parts <= 8; parts *= 2) {
            int kp = k / parts;
            int smem = (V41_SMF_NC + m) * (kp + 8) * 2;
            if (k % parts == 0 && kp % 8 == 0 && smem <= 99 * 1024) {
                cudaStream_t st = (cudaStream_t)stream;
                if (smem > 48 * 1024) V41_SMEM_LIMIT(v41_gemv_ffma_narrow_parts_kernel, 99 * 1024);
                v41_gemv_ffma_narrow_parts_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, st>>>(
                    (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k, kp);
                V41_ERR();
            }
        }
    }
    if (m > V41_SM_MAX && m <= v41_wide(2) && al) {
        // the small-row body, a launch a group of rows
        for (int m0 = 0; m0 < m; m0 += V41_SM_MAX) {
            int mm = m - m0 < V41_SM_MAX ? m - m0 : V41_SM_MAX;
            int rc = memra_v41_gemm_ffma((const v41_bf*)a + (long)m0 * lda, lda, b, ldb, c + (long)m0 * ldc, ldc, mm, n,
                                         k, stream);
            if (rc) return rc;
        }
        return 0;
    }
    if (m <= V41_SM_MAX && al) {
        // the narrow form when all of K, or its halves, fit a block with the call's rows; else a block
        // of 32 columns with 4 or 8 rows
        cudaStream_t st = (cudaStream_t)stream;
        int kp = k;
        if ((V41_SMF_NC + m) * (kp + 8) * 2 > 99 * 1024) kp = k / 2;
        int smem = (V41_SMF_NC + m) * (kp + 8) * 2;
        if (n <= 4096 && smem <= 99 * 1024 && kp == k) {
            if (smem > 48 * 1024) V41_SMEM_LIMIT(v41_gemv_ffma_narrow_kernel, 99 * 1024);
            v41_gemv_ffma_narrow_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, st>>>(
                (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k);
        } else if (n <= 4096 && smem <= 99 * 1024 && kp % 8 == 0) {
            if (smem > 48 * 1024) V41_SMEM_LIMIT(v41_gemv_ffma_narrow_parts_kernel, 99 * 1024);
            v41_gemv_ffma_narrow_parts_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, st>>>(
                (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k, kp);
        } else if (m <= 4) {
            v41_gemv_ffma_kernel<4><<<(n + 31) / 32, 128, 0, st>>>((const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc,
                                                                    m, n, k);
        } else {
            v41_gemv_ffma_kernel<8><<<(n + 31) / 32, 256, 0, st>>>((const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc,
                                                                    m, n, k);
        }
        V41_ERR();
    }
    dim3 grid((n + 63) / 64, (m + 63) / 64);
    if (al)
        v41_gemm_ffma_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b, ldb,
                                                                           c, ldc, m, n, k);
    else
        v41_gemm_ffma_kernel<false><<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b,
                                                                            ldb, c, ldc, m, n, k);
    V41_ERR();
}

// Two fp32 linears of the same input (the ratio-2 compressor's kv and gate): c0 = a . b0^T [m][n0] and
// c1 = a . b1^T [m][n1], each output the bytes memra_v41_gemm_ffma writes for it. A call the ring kernel takes is
// one launch over both outputs' columns (two chains of K side by side instead of one after the other); any other
// is the two calls.
extern "C" int memra_v41_gemm_ffma2(const void* a, long lda, const void* b0, long ldb0, float* c0, long ldc0, int n0,
                                    const void* b1, long ldb1, float* c1, long ldc1, int n1, int m, int k,
                                    void* stream) {
    if (k % 32 || m < 0 || n0 <= 0 || n1 <= 0) return 41023;
    if (m == 0) return 0;
    if (v41_ring_fits(a, lda, b0, ldb0, m, n0, k) && v41_ring_fits(a, lda, b1, ldb1, m, n1, k)) {
        V41RingOut o0{(const v41_bf*)b0, ldb0, c0, ldc0, n0, (n0 + V41_SMF_NC - 1) / V41_SMF_NC},
            o1{(const v41_bf*)b1, ldb1, c1, ldc1, n1, (n1 + V41_SMF_NC - 1) / V41_SMF_NC};
        return v41_ring_launch(a, lda, o0, o1, m, k, stream);
    }
    int rc = memra_v41_gemm_ffma(a, lda, b0, ldb0, c0, ldc0, m, n0, k, stream);
    return rc ? rc : memra_v41_gemm_ffma(a, lda, b1, ldb1, c1, ldc1, m, n1, k, stream);
}

// ---------------------------------------------------------------------------------------------
// Sink sparse attention (reference sparse_attn): q [m][heads][512] bf16, kv [n][512] bf16 (one
// latent head, K = V), idx [m][topk] (-1 = empty). One block per (query, 16 heads); key blocks of
// 64 in index order with an online max; P is cast to bf16 for PV, the denominator stays fp32, the
// sink joins the denominator only. A row with no valid index yields zeros.

#define V41_AH 16
#define V41_AD 512
#define V41_AK 64
#define V41_ARS 1040  // bytes per 512-element bf16 row (+16: conflict-free fragments)
#define V41_APS 144   // bytes per 64-element bf16 P row
#define V41_ASMEM0 (V41_AH * V41_ARS + V41_AK * V41_ARS + V41_AH * V41_APS + 4 * 16 * 4 * 2)  // + the row's indices

// A query row's keys when they are not one buffer (`memra_v41_sparse_attn_keys`): index i below
// `nwin` is row i of `win` (the window ring, or a prefill's own rows), any other index row i - nwin
// of `comp` (the group's compressed KV). A decode step then reads the ring and the compressed KV
// where they live, and every row of a batch its own sequence's.
struct V41KeyRef {
    const v41_bf* win;
    const v41_bf* comp;
    int nwin;
    int pad;
};

__global__ void __launch_bounds__(128) v41_sparse_attn_kernel(const v41_bf* __restrict__ q, long ldq,
                                                              const v41_bf* __restrict__ kv,
                                                              const V41KeyRef* __restrict__ keys,
                                                              const int* __restrict__ idx, long ldi, int topk,
                                                              const float* __restrict__ sink, float scale,
                                                              v41_bf* __restrict__ o, long ldo) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* sQ = sm;
    uint8_t* sKV = sQ + V41_AH * V41_ARS;
    uint8_t* sP = sKV + V41_AK * V41_ARS;
    float* sMax = (float*)(sP + V41_AH * V41_APS);  // [4 warps][16 rows]
    float* sSum = sMax + 64;
    int* sIdx = (int*)(sSum + 64);  // [nblk * V41_AK]: the row's indices, -1 past topk
    int mq = blockIdx.x, h0 = blockIdx.y * V41_AH;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    // the row's key bases: one buffer, or its reference (read with the tiles below in flight; first
    // used after the indices are in)
    const v41_bf* kwin = kv;
    const v41_bf* kcomp = kv;
    int nwin = 0x7fffffff;
    if (keys) {
        const V41KeyRef kr = keys[mq];
        kwin = kr.win;
        nwin = kr.nwin;
        kcomp = (const v41_bf*)((const uint8_t*)kr.comp - (long)kr.nwin * V41_AD * 2);
    }

    // every copy of a tile in flight at once (cp.async), then one wait: the gathers are latency-bound
    const uint8_t* qg = (const uint8_t*)(q + (long)mq * ldq + (long)h0 * V41_AD);
    for (int i = tid; i < V41_AH * 64; i += 128) {
        int r = i >> 6, c = i & 63;
        v41_cp16(sQ + r * V41_ARS + c * 16, qg + (long)r * V41_AD * 2 + c * 16, true);
    }
    v41_cp_commit();
    // the row's indices once, 8 loads in flight per thread: a key block's gather then waits on one
    // round trip, not on its index load and then its rows
    int nblk = (topk + V41_AK - 1) / V41_AK;
    for (int j0 = 0; j0 < nblk * V41_AK; j0 += 8 * 128) {
        int iv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int j = j0 + u * 128 + tid;
            iv[u] = j < topk ? idx[(long)mq * ldi + j] : -1;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int j = j0 + u * 128 + tid;
            if (j < nblk * V41_AK) sIdx[j] = iv[u];
        }
    }
    __syncthreads();
    float m_run[2] = {-1e30f, -1e30f}, l_run[2] = {0.f, 0.f};
    float oacc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) oacc[i][e] = 0.f;

    for (int bk = 0; bk < nblk; ++bk) {
        const int* bIdx = sIdx + bk * V41_AK;
        for (int i = tid; i < V41_AK * 64; i += 128) {
            int r = i >> 6, c = i & 63;
            int ix = bIdx[r];
            const v41_bf* kb = ix < nwin ? kwin : kcomp;
            const uint8_t* src = (const uint8_t*)(kb + (long)(ix >= 0 ? ix : 0) * V41_AD) + c * 16;
            v41_cp16(sKV + r * V41_ARS + c * 16, src, ix >= 0);  // a missing key lands as zeros
        }
        v41_cp_commit();
        v41_cp_wait<0>();
        __syncthreads();
        // scores for keys [16w, 16w+16): two n8 tiles over K = 512
        float s[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[ni][e] = 0.f;
#pragma unroll 4
        for (int kk = 0; kk < 32; ++kk) {
            int kb = kk * 32;  // bytes
            uint32_t af[4], bf[4];  // two ldmatrix loads where eight per-lane loads fetched the same words
            v41_ldsm_a(af, sQ, 0, V41_ARS, kb);
            v41_ldsm_b2(bf, sKV, w * 16, V41_ARS, kb);
            v41_mma_bf16(s[0], af, bf[0], bf[1]);
            v41_mma_bf16(s[1], af, bf[2], bf[3]);
        }
        float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int key = w * 16 + ni * 8 + 2 * t + (e & 1);
                float v = bIdx[key] >= 0 ? __fmul_rn(s[ni][e], scale) : -INFINITY;
                s[ni][e] = v;
                rmax[e >> 1] = fmaxf(rmax[e >> 1], v);
            }
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            rmax[hh] = fmaxf(rmax[hh], __shfl_xor_sync(0xffffffffu, rmax[hh], 1));
            rmax[hh] = fmaxf(rmax[hh], __shfl_xor_sync(0xffffffffu, rmax[hh], 2));
        }
        if (t == 0) {
            sMax[w * 16 + g] = rmax[0];
            sMax[w * 16 + g + 8] = rmax[1];
        }
        __syncthreads();
        float m_new[2], sc_prev[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            int row = g + hh * 8;
            float bm = fmaxf(fmaxf(sMax[row], sMax[16 + row]), fmaxf(sMax[32 + row], sMax[48 + row]));
            m_new[hh] = fmaxf(m_run[hh], bm);
            sc_prev[hh] = expf(m_run[hh] - m_new[hh]);
        }
        float psum[2] = {0.f, 0.f};
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float p = expf(s[ni][e] - m_new[e >> 1]);
                s[ni][e] = p;
                psum[e >> 1] = __fadd_rn(psum[e >> 1], p);
            }
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            psum[hh] = __fadd_rn(psum[hh], __shfl_xor_sync(0xffffffffu, psum[hh], 1));
            psum[hh] = __fadd_rn(psum[hh], __shfl_xor_sync(0xffffffffu, psum[hh], 2));
        }
        if (t == 0) {
            sSum[w * 16 + g] = psum[0];
            sSum[w * 16 + g + 8] = psum[1];
        }
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            int key = w * 16 + ni * 8 + 2 * t;
            uint32_t lo = (uint32_t)v41_f2bf(s[ni][0]) | ((uint32_t)v41_f2bf(s[ni][1]) << 16);
            uint32_t hi = (uint32_t)v41_f2bf(s[ni][2]) | ((uint32_t)v41_f2bf(s[ni][3]) << 16);
            *(uint32_t*)(sP + g * V41_APS + key * 2) = lo;
            *(uint32_t*)(sP + (g + 8) * V41_APS + key * 2) = hi;
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            int row = g + hh * 8;
            float bs = __fadd_rn(__fadd_rn(__fadd_rn(sSum[row], sSum[16 + row]), sSum[32 + row]), sSum[48 + row]);
            l_run[hh] = __fadd_rn(__fmul_rn(l_run[hh], sc_prev[hh]), bs);
            m_run[hh] = m_new[hh];
        }
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            oacc[nt][0] = __fmul_rn(oacc[nt][0], sc_prev[0]);
            oacc[nt][1] = __fmul_rn(oacc[nt][1], sc_prev[0]);
            oacc[nt][2] = __fmul_rn(oacc[nt][2], sc_prev[1]);
            oacc[nt][3] = __fmul_rn(oacc[nt][3], sc_prev[1]);
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            int kb = kk * 32;
            uint32_t af[4];
            v41_ldsm_a(af, sP, 0, V41_APS, kb);
#pragma unroll
            for (int nt = 0; nt < 16; nt += 2) {
                // two n8 tiles of V per transposed load: lanes 16.. address the next 8 columns
                int c0 = w * 128 + nt * 8 + (lane >> 4) * 8;
                uint32_t vf[4];
                v41_ldsm_x4_trans(vf, sKV + (kk * 16 + (lane & 15)) * V41_ARS + c0 * 2);
                v41_mma_bf16(oacc[nt], af, vf[0], vf[1]);
                v41_mma_bf16(oacc[nt + 1], af, vf[2], vf[3]);
            }
        }
        __syncthreads();
    }
    v41_cp_wait<0>();  // no key block: the Q copies are still drained before exit
    float den[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) den[hh] = __fadd_rn(l_run[hh], expf(sink[h0 + g + hh * 8] - m_run[hh]));
    v41_bf* og = o + (long)mq * ldo + (long)h0 * V41_AD;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        int c = w * 128 + nt * 8 + 2 * t;
        uint32_t lo = (uint32_t)v41_f2bf(oacc[nt][0] / den[0]) | ((uint32_t)v41_f2bf(oacc[nt][1] / den[0]) << 16);
        uint32_t hi = (uint32_t)v41_f2bf(oacc[nt][2] / den[1]) | ((uint32_t)v41_f2bf(oacc[nt][3] / den[1]) << 16);
        *(uint32_t*)(og + (long)g * V41_AD + c) = lo;
        *(uint32_t*)(og + (long)(g + 8) * V41_AD + c) = hi;
    }
}

extern "C" int memra_v41_sparse_attn(const void* q, long ldq, const void* kv, const int* idx, long ldi, int topk,
                                     const float* sink, float scale, void* o, long ldo, int m, int heads, int hd,
                                     void* stream) {
    if (hd != V41_AD || heads % V41_AH || m < 0 || topk < 0 || ldq % 8 || ldo % 8 ||
        ((uintptr_t)q | (uintptr_t)kv | (uintptr_t)o) % 16)
        return 41007;
    if (m == 0) return 0;
    // the tiles plus the row's indices (rounded up to key blocks); a block holds at most 99 KB
    int smem = V41_ASMEM0 + (topk + V41_AK - 1) / V41_AK * V41_AK * 4;
    if (smem > 99 * 1024) return 41007;
    V41_SMEM_LIMIT(v41_sparse_attn_kernel, 99 * 1024);
    dim3 grid(m, heads / V41_AH);
    v41_sparse_attn_kernel<<<grid, 128, smem, (cudaStream_t)stream>>>(
        (const v41_bf*)q, ldq, (const v41_bf*)kv, nullptr, idx, ldi, topk, sink, scale, (v41_bf*)o, ldo);
    V41_ERR();
}

// The same attention with each query row's keys named by `keys[row]` (V41KeyRef) instead of one buffer.
extern "C" int memra_v41_sparse_attn_keys(const void* q, long ldq, const void* keys, const int* idx, long ldi,
                                          int topk, const float* sink, float scale, void* o, long ldo, int m,
                                          int heads, int hd, void* stream) {
    if (hd != V41_AD || heads % V41_AH || m < 0 || topk < 0 || ldq % 8 || ldo % 8 || !keys ||
        ((uintptr_t)q | (uintptr_t)o) % 16)
        return 41007;
    if (m == 0) return 0;
    int smem = V41_ASMEM0 + (topk + V41_AK - 1) / V41_AK * V41_AK * 4;
    if (smem > 99 * 1024) return 41007;
    V41_SMEM_LIMIT(v41_sparse_attn_kernel, 99 * 1024);
    dim3 grid(m, heads / V41_AH);
    v41_sparse_attn_kernel<<<grid, 128, smem, (cudaStream_t)stream>>>(
        (const v41_bf*)q, ldq, nullptr, (const V41KeyRef*)keys, idx, ldi, topk, sink, scale, (v41_bf*)o, ldo);
    V41_ERR();
}

// The index rows the sparse attention reads, built on the device: [rows][tot] = the window part
// (ww slots) then the selection rows r0.. of `sel` (tw wide, already offset by the window length).
// A prefill row (sp == 0) at position p sees max(0, p - win + 1) ..= p in order, -1 past p (vendor
// get_window_topk_idxs at start_pos 0); a decode row (sp > 0) reads the whole ring oldest slot first
// (slot (sp % win + 1 + j) % win), -1 for slots past sp.
__global__ void v41_attn_idx_kernel(const int* __restrict__ sel, int tw, int r0, int rows, int sp, int win, int ww,
                                    int tot, int* __restrict__ out) {
    int i = blockIdx.x;
    int* o = out + (long)i * tot;
    for (int j = threadIdx.x; j < tot; j += blockDim.x) {
        int v;
        if (j < ww) {
            if (sp == 0) {
                int p = r0 + i;
                int start = p + 1 > win ? p + 1 - win : 0;
                int k = start + j;
                v = k > p ? -1 : k;
            } else {
                int slot = (sp % win + 1 + j) % win;
                v = slot > sp ? -1 : slot;
            }
        } else {
            v = sel[(long)(r0 + i) * tw + (j - ww)];
        }
        o[j] = v;
    }
}

extern "C" int memra_v41_attn_idx(const int* sel, int tw, int r0, int rows, int sp, int win, int ww, int tot, int* out,
                                  void* stream) {
    if (rows <= 0) return rows < 0 ? 41027 : 0;
    if (win <= 0 || ww < 0 || tw < 0 || tot != ww + tw || (tw > 0 && !sel) || (sp > 0 && rows != 1)) return 41027;
    v41_attn_idx_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(sel, tw, r0, rows, sp, win, ww, tot, out);
    V41_ERR();
}

// The index rows of several decode tokens run in one forward (a drafted block's verification):
// row i is the decode row of position p = sp + r0 + i, the whole ring oldest slot first, -1 for
// slots past p, read from [the ring as it stood before the forward (win rows) | the forward's own
// KV rows]. A slot's position is the newest q <= p with q % win == slot: one of the forward's own
// rows (q >= sp) is entry win + (q - sp), an older one is the ring's slot. The order, and so the
// blocked softmax's key order, is the one-token program's; then the selection rows.
__global__ void v41_attn_idx_rows_kernel(const int* __restrict__ sel, int tw, int r0, int sp, int win, int tot,
                                         int* __restrict__ out) {
    int i = blockIdx.x;
    int p = sp + r0 + i;
    int* o = out + (long)i * tot;
    for (int j = threadIdx.x; j < tot; j += blockDim.x) {
        int v;
        if (j < win) {
            int slot = (p % win + 1 + j) % win;
            if (slot > p) {
                v = -1;
            } else {
                int q = p - (p - slot) % win;
                v = q >= sp ? win + (q - sp) : slot;
            }
        } else {
            v = sel[(long)(r0 + i) * tw + (j - win)];
        }
        o[j] = v;
    }
}

extern "C" int memra_v41_attn_idx_rows(const int* sel, int tw, int r0, int rows, int sp, int win, int tot, int* out,
                                       void* stream) {
    if (rows <= 0) return rows < 0 ? 41028 : 0;
    if (win <= 0 || tw < 0 || tot != win + tw || (tw > 0 && !sel) || sp <= 0 || r0 < 0) return 41028;
    v41_attn_idx_rows_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(sel, tw, r0, sp, win, tot, out);
    V41_ERR();
}

// The index rows of a decode call whose rows belong to different sequences: row i is the
// one-token decode row of its own position pos[i] (its sequence's ring oldest slot first, -1 for
// slots past the position), read from that sequence's region of the KV rows (`base[i]`: its ring,
// then its compressed rows), then its selection row (already offset to its region), -1 past
// `nsel[i]` (a sequence with no compressed row yet has none).
__global__ void v41_attn_idx_batch_kernel(const int* __restrict__ sel, int tw, const int* __restrict__ pos,
                                          const int* __restrict__ base, const int* __restrict__ nsel, int win,
                                          int tot, int* __restrict__ out) {
    int i = blockIdx.x;
    int p = pos[i], b = base[i], ns = nsel[i];
    int* o = out + (long)i * tot;
    for (int j = threadIdx.x; j < tot; j += blockDim.x) {
        int v;
        if (j < win) {
            int slot = (p % win + 1 + j) % win;
            v = slot > p ? -1 : b + slot;
        } else {
            int jj = j - win;
            v = jj < ns ? sel[(long)i * tw + jj] : -1;
        }
        o[j] = v;
    }
}

extern "C" int memra_v41_attn_idx_batch(const int* sel, int tw, const int* pos, const int* base, const int* nsel,
                                        int rows, int win, int tot, int* out, void* stream) {
    if (rows <= 0) return rows < 0 ? 41031 : 0;
    if (win <= 0 || tw < 0 || tot != win + tw || (tw > 0 && !sel) || !pos || !base || !nsel) return 41031;
    v41_attn_idx_batch_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(sel, tw, pos, base, nsel, win, tot, out);
    V41_ERR();
}

// *dst[r] = src[r] (rows of `words` 16-bit words; src row r at pitch lds words; dst[r] a device
// address, null skips the row): every row of a decode call over several sequences into its own
// sequence's buffer in one launch (its window ring slot, an open pair's slot, an appended
// compressed row or index key). The targets are distinct.
__global__ void v41_put_rows_kernel(const uint16_t* __restrict__ src, long lds,
                                    const unsigned long long* __restrict__ dst, int words) {
    int r = blockIdx.x;
    uint16_t* o = (uint16_t*)dst[r];
    if (!o) return;
    const uint16_t* s = src + (long)r * lds;
    for (int i = threadIdx.x; i < words; i += blockDim.x) o[i] = s[i];
}

extern "C" int memra_v41_put_rows(const void* src, long lds, const void* dst, int rows, int words, void* stream) {
    if (rows <= 0) return rows < 0 ? 41032 : 0;
    if (!src || !dst || words <= 0 || lds < words) return 41032;
    v41_put_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)src, lds,
                                                                (const unsigned long long*)dst, words);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// The draft head's pick: row[i] += bias[i] in fp32 (the Markov head's bias added to the block
// row's logits, vendor `logits[:, i].add_(logits_bias)`), then the largest value and the lowest
// index holding it. One block; out_val / out_idx are one element each.
__global__ void __launch_bounds__(1024) v41_add_argmax_kernel(float* __restrict__ row, const float* __restrict__ bias,
                                                              int n, float* __restrict__ out_val,
                                                              int* __restrict__ out_idx) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = bias ? __fadd_rn(row[i], bias[i]) : row[i];
        if (bias) row[i] = v;
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w)
            if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
                best = sv[w];
                bi = si[w];
            }
        *out_val = best;
        *out_idx = bi;
    }
}

extern "C" int memra_v41_add_argmax(float* row, const float* bias, int n, float* out_val, int* out_idx, void* stream) {
    if (n <= 0 || !row || !out_val || !out_idx) return 41029;
    v41_add_argmax_kernel<<<1, 1024, 0, (cudaStream_t)stream>>>(row, bias, n, out_val, out_idx);
    V41_ERR();
}

// The same pick for a chain of picks on the device, over several blocks: each block reduces its
// slice of the row to the largest value and the lowest index holding it, and the blocks meet in
// one 64-bit key through atomicMin. The key's high word falls as the value rises (the float's
// bits mapped to an unsigned order and inverted, -0 read as +0), its low word is the index, so
// the smallest key is the largest value at its lowest index and the low word alone is the
// token: the next pick's gather reads it as an int. `key` must hold 0xffffffff00000000 before
// the launch (no value, token 0); a row of NaNs leaves it so, and the high word decodes to NaN.
// `store` writes row + bias back (a caller that keeps the logits).
__global__ void __launch_bounds__(256) v41_add_argmax_key_kernel(float* __restrict__ row,
                                                                 const float* __restrict__ bias, int n, int per,
                                                                 int store, unsigned long long* __restrict__ key) {
    __shared__ unsigned long long sk[8];
    int lo = blockIdx.x * per;
    int hi = min(n, lo + per);
    unsigned long long best = ~0ull;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = bias ? __fadd_rn(row[i], bias[i]) : row[i];
        if (bias && store) row[i] = v;
        if (v != v) continue;
        unsigned u = v == 0.0f ? 0u : __float_as_uint(v);
        unsigned asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        unsigned long long k = ((unsigned long long)(~asc) << 32) | (unsigned)i;
        best = k < best ? k : best;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long ok = __shfl_xor_sync(0xffffffffu, best, o);
        best = ok < best ? ok : best;
    }
    if ((threadIdx.x & 31) == 0) sk[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w) best = sk[w] < best ? sk[w] : best;
        // a slice of NaNs holds no key: the row's key keeps what the other slices (or the caller) wrote
        if (best != ~0ull) atomicMin(key, best);
    }
}

extern "C" int memra_v41_add_argmax_key(float* row, const float* bias, int n, int store, void* key, void* stream) {
    if (n <= 0 || !row || !key) return 41033;
    const int per = 4096;
    v41_add_argmax_key_kernel<<<(n + per - 1) / per, 256, 0, (cudaStream_t)stream>>>(
        row, bias, n, per, store, (unsigned long long*)key);
    V41_ERR();
}

// The greedy token of each fp32 logits row of `x` ([rows][n], pitch ld): the largest value and
// the lowest index holding it, one block per row. A row holding a non-finite value reports NaN, so
// the caller refuses it as it refuses such a row of downloaded logits.
__global__ void __launch_bounds__(1024) v41_argmax_rows_kernel(const float* __restrict__ x, long ld, int n,
                                                               float* __restrict__ out_val,
                                                               int* __restrict__ out_idx) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ int sb[32];
    const float* row = x + (long)blockIdx.x * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    int bad = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i];
        if ((__float_as_uint(v) & 0x7f800000u) == 0x7f800000u) bad = 1;
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        bad |= __shfl_xor_sync(0xffffffffu, bad, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sv[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = bi;
        sb[threadIdx.x >> 5] = bad;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w) {
            bad |= sb[w];
            if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
                best = sv[w];
                bi = si[w];
            }
        }
        out_val[blockIdx.x] = bad ? __uint_as_float(0x7fc00000u) : best;
        out_idx[blockIdx.x] = bi;
    }
}

extern "C" int memra_v41_argmax_rows(const float* x, long ld, int rows, int n, float* out_val, int* out_idx,
                                     void* stream) {
    if (rows <= 0) return rows < 0 ? 41030 : 0;
    if (n <= 0 || ld < n || !x || !out_val || !out_idx) return 41030;
    v41_argmax_rows_kernel<<<rows, 1024, 0, (cudaStream_t)stream>>>(x, ld, n, out_val, out_idx);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// The trunk's input on the device: token t's bf16 embedding row, gathered from the resident table, into each of
// its hc stream slots (`h` [s][hc][d]), and the one-hot carried pre-mix (`pm` [s][hc] f32, 1 in slot 0). The
// bytes the host gather, its upload and the hc copies wrote. One block a token, 16 bytes a thread a step; an id
// outside the table (the host refuses those before the launch) leaves its rows zero instead of reading past it.

__global__ void v41_embed_rows_kernel(const uint4* __restrict__ table, long vocab, const int* __restrict__ ids,
                                      int d16, int hc, uint4* __restrict__ h, float* __restrict__ pm) {
    const int t = blockIdx.x;
    const int id = ids[t];
    const bool ok = id >= 0 && id < vocab;
    const uint4* src = table + (long)(ok ? id : 0) * d16;
    uint4* dst = h + (long)t * hc * d16;
    for (int i = threadIdx.x; i < d16; i += blockDim.x) {
        const uint4 v = ok ? src[i] : make_uint4(0u, 0u, 0u, 0u);
        for (int k = 0; k < hc; ++k) dst[(long)k * d16 + i] = v;
    }
    if ((int)threadIdx.x < hc) pm[(long)t * hc + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}

extern "C" int memra_v41_embed_rows(const void* table, long vocab, const int* ids, int s, int d, int hc, void* h,
                                    float* pm, void* stream) {
    if (s <= 0) return s < 0 ? 41048 : 0;
    if (!table || !ids || !h || !pm || vocab <= 0 || d <= 0 || (d * 2) % 16 != 0 || hc <= 0 || hc > 256)
        return 41048;
    if (((unsigned long)table | (unsigned long)h) % 16 != 0) return 41048;
    v41_embed_rows_kernel<<<s, 256, 0, (cudaStream_t)stream>>>((const uint4*)table, vocab, ids, d * 2 / 16, hc,
                                                              (uint4*)h, pm);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// CSA2 compressor pooling at ratio 2: out[j] = bf16(kv[2j] * p0 + kv[2j+1] * p1) with p the
// softmax over the pair of gate scores, per channel (fp32 rows in, bf16 rows out).

__global__ void v41_comp_pool2_kernel(const float* __restrict__ kv, const float* __restrict__ sc, long ld,
                                      v41_bf* __restrict__ out, long ldo, int d) {
    int j = blockIdx.x;
    const float* k0 = kv + (long)(2 * j) * ld;
    const float* k1 = k0 + ld;
    const float* s0 = sc + (long)(2 * j) * ld;
    const float* s1 = s0 + ld;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a = s0[c], b = s1[c];
        float mx = fmaxf(a, b);
        float e0 = expf(a - mx), e1 = expf(b - mx);
        float sum = __fadd_rn(e0, e1);
        float p0 = e0 / sum, p1 = e1 / sum;
        out[(long)j * ldo + c] = v41_f2bf(__fadd_rn(__fmul_rn(k0[c], p0), __fmul_rn(k1[c], p1)));
    }
}

extern "C" int memra_v41_comp_pool2(const float* kv, const float* sc, long ld, void* out, long ldo, int groups, int d,
                                    void* stream) {
    if (groups <= 0) return groups < 0 ? 41008 : 0;
    v41_comp_pool2_kernel<<<groups, 128, 0, (cudaStream_t)stream>>>(kv, sc, ld, (v41_bf*)out, ldo, d);
    V41_ERR();
}

// The same pooling for the rows of a decode call over several sequences, each row's pair where
// its sequence keeps it: kv[j] / sc[j] are device addresses of row j's two fp32 rows (even
// position first, pitch d), out row j its latent. Every row is pooled, whether or not its pair
// is complete this step (the caller drops the rows that are not): the launch does not depend on
// which sequences complete a pair.
__global__ void v41_comp_pool2_keys_kernel(const unsigned long long* __restrict__ kv,
                                           const unsigned long long* __restrict__ sc, v41_bf* __restrict__ out,
                                           long ldo, int d) {
    int j = blockIdx.x;
    const float* k0 = (const float*)kv[j];
    const float* k1 = k0 + d;
    const float* s0 = (const float*)sc[j];
    const float* s1 = s0 + d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a = s0[c], b = s1[c];
        float mx = fmaxf(a, b);
        float e0 = expf(a - mx), e1 = expf(b - mx);
        float sum = __fadd_rn(e0, e1);
        float p0 = e0 / sum, p1 = e1 / sum;
        out[(long)j * ldo + c] = v41_f2bf(__fadd_rn(__fmul_rn(k0[c], p0), __fmul_rn(k1[c], p1)));
    }
}

extern "C" int memra_v41_comp_pool2_keys(const void* kv, const void* sc, void* out, long ldo, int rows, int d,
                                         void* stream) {
    if (rows <= 0) return rows < 0 ? 41034 : 0;
    if (!kv || !sc || !out || d <= 0 || ldo < d) return 41034;
    v41_comp_pool2_keys_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(
        (const unsigned long long*)kv, (const unsigned long long*)sc, (v41_bf*)out, ldo, d);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Indexer score: score[i][t] = bf16( sum_h bf16( relu(bf16(q[i,h,:] . k[t,:])) * w[i,h] ) ), masked
// to -inf where t >= (start_pos + i + 1) / ratio or where the candidate block t/cb is not kept.
// q [rows][32][128] bf16, k [T][128] bf16, w [rows][32] bf16 (already times 128^-0.5 * 32^-0.5).
// One block per (row, 64 keys); 4 warps x 16 keys; both head halves.

#define V41_IRS 272  // bytes per 128-element bf16 row (+16)

#define V41_IQB 8  // query rows per block: the 64-key tile is loaded once for all of them

// MI 16-head MMA tiles (2 = all 32 heads, the model's program). PART 1 writes the fp32 head sum
// unrounded and unmasked (the head-split measurement arm: `memra_v41_index_score_finish` adds the
// ranks' partials, rounds and masks).
//
// A warp's 16 keys stay in registers as MMA fragments for the block's rows (they are per warp, so
// shared memory would only be a detour), loaded once with every load in flight. The Q rows go
// through two shared-memory buffers by cp.async: the next live row's copies are in flight while
// the current row runs its MMAs, and a row waits on one barrier. 18 KB a block.
template <int MI, int PART>
__global__ void __launch_bounds__(128) v41_index_score_kernel(const v41_bf* __restrict__ q, long ldq,
                                                              const v41_bf* __restrict__ k, int T,
                                                              const v41_bf* __restrict__ w, long ldw, float wscale,
                                                              int start_pos,
                                                              int ratio, const uint8_t* __restrict__ cand, long ldc,
                                                              int cb, void* __restrict__ outv, long ldo, int rows) {
    __shared__ __align__(16) uint8_t sQ2[2][32 * V41_IRS];
    __shared__ float sW2[2][32];
    int t0 = blockIdx.x * 64, row0 = blockIdx.y * V41_IQB;
    int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    // A row whose every key of this tile is masked (past its causal limit, or in no kept candidate
    // block) is written -inf without its MMAs: the masked score is -inf whatever the product was.
    auto masked = [&](int row) {
        if (PART) return false;
        if (t0 >= (start_pos + row + 1) / ratio) return true;
        if (!cand) return false;
        int b1 = (min(T, t0 + 64) - 1) / cb;
        for (int bb = t0 / cb; bb <= b1; ++bb)
            if (cand[(long)row * ldc + bb]) return false;
        return true;
    };
    auto write_masked = [&](int row) {
        if (tid < 64 && t0 + tid < T) ((v41_bf*)outv)[(long)row * ldo + t0 + tid] = (v41_bf)0xFF80u;  // bf16 -inf
    };
    // bit rq of live: row row0 + rq runs its MMAs (the same value in every thread)
    unsigned live = 0u;
    for (int rq = 0; rq < V41_IQB; ++rq) {
        int row = row0 + rq;
        if (row >= rows) break;
        if (masked(row)) write_masked(row);
        else live |= 1u << rq;
    }
    if (!live) return;
    // the copies of a row's Q and its head weights into buffer b
    auto issue = [&](int rq, int b) {
        int row = row0 + rq;
        const uint8_t* qg = (const uint8_t*)(q + (long)row * ldq);
        for (int i = tid; i < MI * 16 * 16; i += 128) {
            int r = i >> 4, c = i & 15;
            v41_cp16(sQ2[b] + r * V41_IRS + c * 16, qg + r * 256 + c * 16, true);
        }
        // weights_proj(x) * (index_head_dim^-0.5 * index_heads^-0.5), a bf16 product in the reference
        if (tid < MI * 16) sW2[b][tid] = v41_bf2f(v41_f2bf(__fmul_rn(v41_bf2f(w[(long)row * ldw + tid]), wscale)));
    };
    int cur = __ffs(live) - 1, buf = 0;
    issue(cur, buf);
    v41_cp_commit();
    // this warp's key fragments: keys t0 + 16 wp + 8 ni + g, the 8 k16 steps of the 128 dims
    uint32_t kf[2][8][2];
#pragma unroll
    for (int ni = 0; ni < 2; ++ni) {
        int key = t0 + wp * 16 + ni * 8 + g;
        bool ok = key < T;
        const uint8_t* kp = (const uint8_t*)(k + (long)(ok ? key : 0) * 128);
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            kf[ni][kk][0] = ok ? v41_ldg32(kp + kk * 32 + t * 4) : 0u;
            kf[ni][kk][1] = ok ? v41_ldg32(kp + kk * 32 + 16 + t * 4) : 0u;
        }
    }
    while (true) {
        unsigned rest = live & ~((2u << cur) - 1u);
        int nxt = rest ? __ffs(rest) - 1 : -1;
        v41_cp_wait<0>();
        __syncthreads();  // row cur landed for every thread; the row before it is done with the other buffer
        if (nxt >= 0) issue(nxt, buf ^ 1);
        v41_cp_commit();
        int row = row0 + cur;
        const uint8_t* sQ = sQ2[buf];
        const float* sW = sW2[buf];
        float d[MI][2][4];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) d[mi][ni][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int kb = kk * 32;
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                uint32_t af[4];
                v41_ldsm_a(af, sQ, mi * 16, V41_IRS, kb);
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) v41_mma_bf16(d[mi][ni], af, kf[ni][kk][0], kf[ni][kk][1]);
            }
        }
        // per key column: this thread holds heads {g, g+8, 16+g, 24+g}; sum their terms then across g
        int visible = (start_pos + row + 1) / ratio;
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                float part = 0.f;
#pragma unroll
                for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        int head = mi * 16 + g + hh * 8;
                        float dv = v41_bf2f(v41_f2bf(d[mi][ni][hh * 2 + cc]));
                        dv = fmaxf(dv, 0.f);
                        float term = v41_bf2f(v41_f2bf(__fmul_rn(dv, sW[head])));
                        part = __fadd_rn(part, term);
                    }
                part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 4));
                part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 8));
                part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, 16));
                int key = t0 + wp * 16 + ni * 8 + 2 * t + cc;
                if (g == 0 && key < T) {
                    if (PART) {
                        ((float*)outv)[(long)row * ldo + key] = part;
                    } else {
                        float v = v41_bf2f(v41_f2bf(part));
                        if (key >= visible) v = -INFINITY;
                        if (cand && !cand[(long)row * ldc + key / cb]) v = -INFINITY;
                        ((v41_bf*)outv)[(long)row * ldo + key] = v41_f2bf(v);
                    }
                }
            }
        if (nxt < 0) break;
        cur = nxt;
        buf ^= 1;
    }
    v41_cp_wait<0>();
}

#define V41_INDEX_LAUNCH(MI, PART, ...)                                                                          \
    do {                                                                                                         \
        /* 18 KB a block: the largest shared-memory split, so five blocks share an SM */                         \
        V41_CARVE_MAX((v41_index_score_kernel<MI, PART>));                                                       \
        dim3 grid((T + 63) / 64, (rows + V41_IQB - 1) / V41_IQB);                                                \
        v41_index_score_kernel<MI, PART><<<grid, 128, 0, (cudaStream_t)stream>>>(__VA_ARGS__);                   \
    } while (0)

extern "C" int memra_v41_index_score(const void* q, long ldq, const void* k, int T, const void* w, long ldw,
                                     float wscale, int start_pos, int ratio, const uint8_t* cand, long ldc, int cb, void* out,
                                     long ldo, int rows, void* stream) {
    if (rows <= 0 || T <= 0) return (rows < 0 || T < 0) ? 41009 : 0;
    if (ratio <= 0 || (cand && cb <= 0) || ldq % 8) return 41009;
    V41_INDEX_LAUNCH(2, 0, (const v41_bf*)q, ldq, (const v41_bf*)k, T, (const v41_bf*)w, ldw, wscale, start_pos, ratio,
                     cand, ldc, cb, out, ldo, rows);
    V41_ERR();
}

// Head-split arm (measurement): the fp32 sum over `heads` (16 or 32) local heads of q [rows][heads*128]
// and w [rows][heads], unrounded and unmasked, into part [rows][T] (ldo floats).
extern "C" int memra_v41_index_score_part(const void* q, long ldq, const void* k, int T, const void* w, long ldw,
                                          float wscale, int heads, float* part, long ldo, int rows, void* stream) {
    if (rows <= 0 || T <= 0) return (rows < 0 || T < 0) ? 41018 : 0;
    if ((heads != 16 && heads != 32) || ldq % 8) return 41018;
    if (heads == 16)
        V41_INDEX_LAUNCH(1, 1, (const v41_bf*)q, ldq, (const v41_bf*)k, T, (const v41_bf*)w, ldw, wscale, 0, 1, nullptr,
                         0, 1, part, ldo, rows);
    else
        V41_INDEX_LAUNCH(2, 1, (const v41_bf*)q, ldq, (const v41_bf*)k, T, (const v41_bf*)w, ldw, wscale, 0, 1, nullptr,
                         0, 1, part, ldo, rows);
    V41_ERR();
}
#undef V41_INDEX_LAUNCH

// out[row][key] = bf16 of the ranks' partials added in rank order (((p0 + p1) + p2) ...), then the
// visibility and candidate masks of the full kernel. parts: n device pointers (a device array).
__global__ void v41_index_finish_kernel(const float* const* __restrict__ parts, int n, long ldp, int T, int start_pos,
                                        int ratio, const uint8_t* __restrict__ cand, long ldc, int cb,
                                        v41_bf* __restrict__ out, long ldo) {
    int row = blockIdx.y;
    int key = blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= T) return;
    float v = parts[0][(long)row * ldp + key];
    for (int r = 1; r < n; ++r) v = __fadd_rn(v, parts[r][(long)row * ldp + key]);
    v = v41_bf2f(v41_f2bf(v));
    int visible = (start_pos + row + 1) / ratio;
    if (key >= visible) v = -INFINITY;
    if (cand && !cand[(long)row * ldc + key / cb]) v = -INFINITY;
    out[(long)row * ldo + key] = v41_f2bf(v);
}

extern "C" int memra_v41_index_score_finish(const void* parts, int n, long ldp, int T, int start_pos, int ratio,
                                            const uint8_t* cand, long ldc, int cb, void* out, long ldo, int rows,
                                            void* stream) {
    if (rows <= 0 || T <= 0) return (rows < 0 || T < 0) ? 41019 : 0;
    if (n <= 0 || ratio <= 0 || (cand && cb <= 0) || rows > 65535) return 41019;
    dim3 grid((T + 255) / 256, rows);
    v41_index_finish_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const float* const*)parts, n, ldp, T, start_pos,
                                                                     ratio, cand, ldc, cb, (v41_bf*)out, ldo);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Exact top-k over a bf16 row (radix select on 16-bit order keys). -inf entries never count; when
// fewer than k entries are finite every finite one is taken. Ties at the threshold go to the
// lowest positions. Selected positions are emitted in ascending order.

__device__ __forceinline__ uint32_t v41_ord(v41_bf u) { return (u & 0x8000) ? (uint32_t)(~u & 0xFFFF) : (uint32_t)(u | 0x8000); }
#define V41_NEG_INF_BITS 0xFF80u

// key source: either a score row (blk == 0) or block maxima over blk positions (+inf on `pin`)
struct V41Keys {
    const v41_bf* row;
    int n;     // number of keys
    int blk;   // 0: key i = row[i]; >0: key i = max(row[i*blk .. i*blk+blk-1] within width)
    int width; // score width when blk > 0
    int pin;   // block index forced to +inf (blk > 0), -1 for none
    __device__ __forceinline__ int get(int i) const {  // -1 for -inf (excluded), else the order key
        v41_bf b;
        if (blk == 0) {
            b = row[i];
        } else {
            if (i == pin) return (int)v41_ord(0x7F80);  // +inf
            float m = -INFINITY;
            int e = min(width, (i + 1) * blk);
            for (int j = i * blk; j < e; ++j) m = fmaxf(m, v41_bf2f(row[j]));
            b = v41_f2bf(m);
        }
        if (b == V41_NEG_INF_BITS) return -1;
        return (int)v41_ord(b);
    }
};

// block-wide exclusive scan of a per-thread count (256 threads), returns the prefix; *total gets the sum
__device__ __forceinline__ int v41_scan_count(int v, int* wsum, int* total) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += up;
    }
    __syncthreads();
    if (lane == 31) wsum[w] = inc;
    __syncthreads();
    int off = 0, tot = 0;
    for (int i = 0; i < 8; ++i) {
        if (i < w) off += wsum[i];
        tot += wsum[i];
    }
    *total = tot;
    return off + inc - v;
}

// cum[q] = the number of set bits in bal[0 .. q) for q in [0, nq); returns the total (256 threads; the
// caller puts a barrier before, for bal, and after, for cum)
__device__ __forceinline__ int v41_prefix_popc(const uint32_t* bal, int* cum, int nq, int* wsum) {
    int per = (nq + 255) / 256;
    int q0 = min(nq, (int)threadIdx.x * per), q1 = min(nq, q0 + per);
    int loc = 0;
    for (int q = q0; q < q1; ++q) loc += __popc(bal[q]);
    int tot;
    int pre = v41_scan_count(loc, wsum, &tot);
    for (int q = q0; q < q1; ++q) {
        cum[q] = pre;
        pre += __popc(bal[q]);
    }
    return tot;
}

// selects up to k keys; emit(pos, slot) for each selected position in ascending order; returns count.
// tab: 3 words per warp of every 256 keys (8 x ceil(n / 256) warp slots), or null.
template <typename Emit>
__device__ int v41_select(const V41Keys& K, int k, int* hist, int* wsum, uint32_t* tab, Emit emit) {
    int tid = threadIdx.x;
    // count finite
    __shared__ int s_cnt, s_thr, s_need, s_gt;
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    int c = 0;
    for (int i = tid; i < K.n; i += 256) c += K.get(i) >= 0;
    atomicAdd(&s_cnt, c);
    __syncthreads();
    int nvalid = s_cnt;
    int need = min(k, nvalid);
    int thr = 0, eq_need = 0x7FFFFFFF;  // take everything finite by default
    if (need < nvalid && need > 0) {
        // pass 1: high byte
        for (int i = tid; i < 256; i += 256) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < K.n; i += 256) {
            int key = K.get(i);
            if (key >= 0) atomicAdd(&hist[key >> 8], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = 0, hb = 255;
            for (; hb >= 0; --hb) {
                if (cum + hist[hb] >= need) break;
                cum += hist[hb];
            }
            s_thr = hb;
            s_gt = cum;
        }
        __syncthreads();
        int hb = s_thr, gt_hi = s_gt;
        for (int i = tid; i < 256; i += 256) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < K.n; i += 256) {
            int key = K.get(i);
            if (key >= 0 && (key >> 8) == hb) atomicAdd(&hist[key & 0xFF], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = gt_hi, lb = 255;
            for (; lb >= 0; --lb) {
                if (cum + hist[lb] >= need) break;
                cum += hist[lb];
            }
            s_thr = (hb << 8) | lb;
            s_need = need - cum;  // how many equal to the threshold are taken
        }
        __syncthreads();
        thr = s_thr;
        eq_need = s_need;
    }
    if (need == 0) return 0;
    bool all = need >= nvalid;
    if (tab) {
        // Emit in position order with the keys read once more, strided (a warp reads 32 neighbours):
        // warp w of key block j ballots which of its keys are above the threshold and which equal
        // it; a prefix over the equal ballots ranks every equal key, so the warp can ballot what it
        // takes; a prefix over those gives every taken key its output slot. Six barriers for the
        // row, whatever its length.
        int nb = (K.n + 255) / 256, nq = nb * 8;
        int w = tid >> 5, lane = tid & 31;
        uint32_t below = (1u << lane) - 1u;
        uint32_t* tA = tab;                 // above the threshold, then: taken
        uint32_t* tE = tab + nq;            // equal to the threshold
        int* tC = (int*)(tab + 2 * nq);     // set bits before the slot: of tE, then of tA
        for (int j = 0; j < nb; ++j) {
            int i = j * 256 + tid;
            int key = i < K.n ? K.get(i) : -1;
            bool gt = key >= 0 && (all || key > thr);
            bool eq = key >= 0 && !gt && key == thr;
            uint32_t bg = __ballot_sync(0xffffffffu, gt), be = __ballot_sync(0xffffffffu, eq);
            if (lane == 0) {
                tA[j * 8 + w] = bg;
                tE[j * 8 + w] = be;
            }
        }
        __syncthreads();
        v41_prefix_popc(tE, tC, nq, wsum);
        __syncthreads();
        for (int j = 0; j < nb; ++j) {
            int q = j * 8 + w;
            uint32_t bg = tA[q], be = tE[q];
            int eq_rank = tC[q] + __popc(be & below);
            bool sel = ((bg >> lane) & 1u) || (((be >> lane) & 1u) && eq_rank < eq_need);
            uint32_t bs = __ballot_sync(0xffffffffu, sel);
            if (lane == 0) tA[q] = bs;
        }
        __syncthreads();
        int total = v41_prefix_popc(tA, tC, nq, wsum);
        __syncthreads();
        for (int j = 0; j < nb; ++j) {
            int q = j * 8 + w;
            uint32_t bs = tA[q];
            if ((bs >> lane) & 1u) emit(j * 256 + tid, tC[q] + __popc(bs & below));
        }
        return total;
    }
    // without the tables: thread t owns the contiguous keys [t * per, (t + 1) * per), counts what it
    // takes (the keys above the threshold, and its share of the first eq_need keys equal to it), and
    // two block scans give its first equal key's rank and its first output slot
    int per = (K.n + 255) / 256;
    int i0 = min(K.n, tid * per), i1 = min(K.n, i0 + per);
    // a thread with at most 32 keys remembers each key's class (a block-maximum key is a loop of loads)
    bool few = per <= 32;
    uint32_t gmask = 0u, emask = 0u;
    int gt = 0, eq = 0;
    for (int i = i0; i < i1; ++i) {
        int key = K.get(i);
        if (key >= 0) {
            if (all || key > thr) {
                ++gt;
                if (few) gmask |= 1u << (i - i0);
            } else if (key == thr) {
                ++eq;
                if (few) emask |= 1u << (i - i0);
            }
        }
    }
    int eq_tot;
    int eq_before = v41_scan_count(eq, wsum, &eq_tot);
    int take_eq = all ? 0 : max(0, min(eq, eq_need - eq_before));
    int total;
    int slot = v41_scan_count(gt + take_eq, wsum, &total);
    for (int i = i0; i < i1; ++i) {
        bool sel, iseq;
        if (few) {
            sel = (gmask >> (i - i0)) & 1u;
            iseq = (emask >> (i - i0)) & 1u;
        } else {
            int key = K.get(i);
            sel = key >= 0 && (all || key > thr);
            iseq = key >= 0 && !sel && key == thr;
        }
        if (iseq && take_eq > 0) {
            sel = true;
            --take_eq;
        }
        if (sel) emit(i, slot++);
    }
    return total;
}

// the select's static shared memory: the histogram, the warp sums and its four counters
#define V41_SEL_STATIC (256 * 4 + 8 * 4 + 4 * 4)
// the emit tables of a row of n keys: 12 bytes per warp of every 256 keys; 0 (the scan form) when
// two blocks with their tables and static shared memory would not share an SM's 102.4 KB: rows up
// to 133,632 keys take the tables (48 KB for the 131,072 keys of a ratio-1 index layer at that
// many tokens), longer rows the scan form (one block an SM is not measured)
static int v41_sel_tab_bytes(int n) {
    long b = (long)((n + 255) / 256) * 8 * 12;
    return b > 0 && b <= 51200 - V41_SEL_STATIC ? (int)b : 0;
}

// BIG: the instance for tables past 48 KB with the static shared memory (rows of more than about
// 125k keys): it has its own limit and the whole of the SM's shared memory, so that two of its
// blocks share an SM, and the other instance's launches never see either setting
template <int BIG>
__global__ void __launch_bounds__(256) v41_topk_kernel(const v41_bf* __restrict__ score, long lds, int T, int k,
                                                       int out_w, int offset, int* __restrict__ out, long ldo,
                                                       int tabs) {
    __shared__ int hist[256];
    __shared__ int wsum[8];
    extern __shared__ uint32_t v41_sel_tab[];
    int row = blockIdx.x;
    V41Keys K{score + (long)row * lds, T, 0, T, -1};
    int* orow = out + (long)row * ldo;
    int n = v41_select(K, k, hist, wsum, tabs ? v41_sel_tab : nullptr,
                       [&](int pos, int slot) { orow[slot] = pos + offset; });
    __syncthreads();
    for (int i = n + threadIdx.x; i < out_w; i += 256) orow[i] = -1;
}

extern "C" int memra_v41_topk(const void* score, long lds, int T, int k, int out_w, int offset, int* out, long ldo,
                              int rows, void* stream) {
    if (rows <= 0) return rows < 0 ? 41010 : 0;
    if (k < 0 || k > out_w || T < 0) return 41010;
    // A few rows (decode) run the contiguous form: with the card to itself a block reads its row
    // fastest that way (0.040 against 0.062 ms on a 32,768-key row); many rows at once want the
    // strided table form (10.1 against 4.3 ms for 8,192 such rows).
    int tb = rows <= 4 ? 0 : v41_sel_tab_bytes(T);
    if (tb + V41_SEL_STATIC > 48 * 1024) {
        V41_SMEM_LIMIT(v41_topk_kernel<1>, 51200 - V41_SEL_STATIC);
        V41_CARVE_MAX(v41_topk_kernel<1>);
        v41_topk_kernel<1><<<rows, 256, tb, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, k, out_w, offset, out,
                                                                    ldo, tb > 0);
    } else {
        v41_topk_kernel<0><<<rows, 256, tb, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, k, out_w, offset, out,
                                                                    ldo, tb > 0);
    }
    V41_ERR();
}

// select_candidate_blocks: block score = max over cb positions, the block holding the row's newest
// visible position pinned to +inf, top-k blocks kept when finite. keep [rows][nb] u8. BIG as in
// v41_topk_kernel: the instance for emit tables past 48 KB with the static shared memory (rows of
// more than 128,000 blocks: 1.024M positions at 8 a block), with its own shared-memory limit.
template <int BIG>
__global__ void __launch_bounds__(256) v41_candidates_kernel(const v41_bf* __restrict__ score, long lds, int T,
                                                             int cb, int kblocks, int start_pos, int ratio,
                                                             uint8_t* __restrict__ keep, long ldk, int tabs) {
    __shared__ int hist[256];
    __shared__ int wsum[8];
    extern __shared__ uint32_t v41_sel_tab[];
    int row = blockIdx.x;
    int nb = (T + cb - 1) / cb;
    int visible = (start_pos + row + 1) / ratio;
    int pin = visible > 0 ? (visible - 1) / cb : -1;
    V41Keys K{score + (long)row * lds, nb, cb, T, pin};
    uint8_t* krow = keep + (long)row * ldk;
    for (int i = threadIdx.x; i < nb; i += 256) krow[i] = 0;
    __syncthreads();
    v41_select(K, kblocks, hist, wsum, tabs ? v41_sel_tab : nullptr, [&](int pos, int) { krow[pos] = 1; });
}

extern "C" int memra_v41_candidates(const void* score, long lds, int T, int cb, int kblocks, int start_pos, int ratio,
                                    uint8_t* keep, long ldk, int rows, void* stream) {
    if (rows <= 0) return rows < 0 ? 41011 : 0;
    if (cb <= 0 || ratio <= 0 || kblocks < 0 || T < 0) return 41011;
    int tb = v41_sel_tab_bytes((T + cb - 1) / cb);
    if (tb + V41_SEL_STATIC > 48 * 1024) {
        V41_SMEM_LIMIT(v41_candidates_kernel<1>, 51200 - V41_SEL_STATIC);
        V41_CARVE_MAX(v41_candidates_kernel<1>);
        v41_candidates_kernel<1><<<rows, 256, tb, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, cb, kblocks,
                                                                          start_pos, ratio, keep, ldk, tb > 0);
    } else {
        v41_candidates_kernel<0><<<rows, 256, tb, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, cb, kblocks,
                                                                          start_pos, ratio, keep, ldk, tb > 0);
    }
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// The select over ONE long row (a decode step's top-k and candidate blocks at long context). The
// one-block form above reads the row's keys four times with 256 threads: 225 us for 131,072 keys.
// Here the row is cut into blocks of V41_SN_CH keys and the same select runs as four launches over a
// small zeroed scratch `ws`: the high-byte histogram and the valid count, the low-byte histogram of
// the threshold's high byte, each block's counts above and equal to the threshold, and the emit,
// where a block starts at the slots and the equal-key rank the blocks before it used. The
// selection is the one-block form's: keys above the threshold, then the first equal keys by
// position, positions ascending.
#define V41_SN_CH 8192
#define V41_SN_LO 272
#define V41_SN_CNT 544
#define V41_SN_BLOCKS 256

struct V41Sn {
    int thr, eq_need, need, all;
};

__device__ V41Sn v41_sn_derive(const int* ws, int k, bool lo) {
    V41Sn s;
    int nvalid = ws[256];
    s.need = min(k, nvalid);
    s.all = s.need >= nvalid;
    s.thr = 0;
    s.eq_need = 0x7FFFFFFF;
    if (!s.all && s.need > 0) {
        int cum = 0, hb = 255, lb = 0;
        for (; hb >= 0; --hb) {
            if (cum + ws[hb] >= s.need) break;
            cum += ws[hb];
        }
        if (lo) {
            for (lb = 255; lb >= 0; --lb) {
                if (cum + ws[V41_SN_LO + lb] >= s.need) break;
                cum += ws[V41_SN_LO + lb];
            }
        }
        s.thr = (hb << 8) | lb;
        s.eq_need = s.need - cum;
    }
    return s;
}

// MODE 0: valid count and high-byte histogram; MODE 1: low-byte histogram of the threshold's high byte
template <int MODE>
__global__ void __launch_bounds__(256) v41_sn_hist_kernel(V41Keys K, int k, int* __restrict__ ws) {
    __shared__ int hist[256];
    __shared__ int s_hb;
    int tid = threadIdx.x;
    hist[tid] = 0;
    if (MODE == 1 && tid == 0) {
        V41Sn s = v41_sn_derive(ws, k, false);
        s_hb = (s.all || s.need == 0) ? -1 : (s.thr >> 8);
    }
    __syncthreads();
    int i0 = blockIdx.x * V41_SN_CH, i1 = min(K.n, i0 + V41_SN_CH), cnt = 0;
    int hb = MODE == 1 ? s_hb : 0;
    for (int i = i0 + tid; i < i1; i += 256) {
        int key = K.get(i);
        if (key >= 0) {
            if (MODE == 0) {
                ++cnt;
                atomicAdd(&hist[key >> 8], 1);
            } else if ((key >> 8) == hb) {
                atomicAdd(&hist[key & 0xFF], 1);
            }
        }
    }
    __syncthreads();
    if (hist[tid]) atomicAdd(ws + (MODE ? V41_SN_LO : 0) + tid, hist[tid]);
    if (MODE == 0 && cnt) atomicAdd(ws + 256, cnt);
}

__global__ void __launch_bounds__(256) v41_sn_count_kernel(V41Keys K, int k, int* __restrict__ ws) {
    __shared__ int s_thr, s_all, s_gt, s_eq;
    int tid = threadIdx.x;
    if (tid == 0) {
        V41Sn s = v41_sn_derive(ws, k, true);
        s_thr = s.thr;
        s_all = s.all;
        s_gt = 0;
        s_eq = 0;
    }
    __syncthreads();
    int thr = s_thr;
    bool all = s_all;
    int i0 = blockIdx.x * V41_SN_CH, i1 = min(K.n, i0 + V41_SN_CH), gt = 0, eq = 0;
    for (int i = i0 + tid; i < i1; i += 256) {
        int key = K.get(i);
        if (key >= 0) {
            if (all || key > thr) ++gt;
            else if (key == thr) ++eq;
        }
    }
    if (gt) atomicAdd(&s_gt, gt);
    if (eq) atomicAdd(&s_eq, eq);
    __syncthreads();
    if (tid == 0) {
        ws[V41_SN_CNT + 2 * blockIdx.x] = s_gt;
        ws[V41_SN_CNT + 2 * blockIdx.x + 1] = s_eq;
    }
}

// EMIT 0: positions (plus offset) into out, ascending, -1 past the selection; EMIT 1: keep[pos] = 1
template <int EMIT>
__global__ void __launch_bounds__(256) v41_sn_emit_kernel(V41Keys K, int k, const int* __restrict__ ws,
                                                          int* __restrict__ out, int out_w, int offset,
                                                          uint8_t* __restrict__ keep) {
    __shared__ int s_thr, s_all, s_eqn, s_eqb, s_slot, s_need;
    __shared__ int wsum[8];
    int tid = threadIdx.x, b = blockIdx.x;
    if (tid == 0) {
        V41Sn s = v41_sn_derive(ws, k, true);
        int eqb = 0, slot = 0;
        for (int j = 0; j < b; ++j) {
            int g = ws[V41_SN_CNT + 2 * j], e = ws[V41_SN_CNT + 2 * j + 1];
            slot += g + (s.all ? 0 : max(0, min(e, s.eq_need - eqb)));
            eqb += e;
        }
        s_thr = s.thr;
        s_all = s.all;
        s_eqn = s.eq_need;
        s_eqb = eqb;
        s_slot = slot;
        s_need = s.need;
    }
    int i0b = b * V41_SN_CH, i1b = min(K.n, i0b + V41_SN_CH);
    if (EMIT == 1)
        for (int i = i0b + tid; i < i1b; i += 256) keep[i] = 0;
    __syncthreads();
    int thr = s_thr, eq_need = s_eqn;
    bool all = s_all;
    // the one-block form's emit on this block's keys: a thread owns 32 contiguous keys
    const int per = V41_SN_CH / 256;
    int i0 = min(i1b, i0b + tid * per), i1 = min(i1b, i0 + per);
    uint32_t gmask = 0u, emask = 0u;
    int gt = 0, eq = 0;
    for (int i = i0; i < i1; ++i) {
        int key = K.get(i);
        if (key >= 0) {
            if (all || key > thr) {
                ++gt;
                gmask |= 1u << (i - i0);
            } else if (key == thr) {
                ++eq;
                emask |= 1u << (i - i0);
            }
        }
    }
    int eq_tot;
    int eq_before = s_eqb + v41_scan_count(eq, wsum, &eq_tot);
    int take_eq = all ? 0 : max(0, min(eq, eq_need - eq_before));
    int total;
    int slot = s_slot + v41_scan_count(gt + take_eq, wsum, &total);
    for (int i = i0; i < i1; ++i) {
        bool sel = (gmask >> (i - i0)) & 1u;
        if (((emask >> (i - i0)) & 1u) && take_eq > 0) {
            sel = true;
            --take_eq;
        }
        if (sel) {
            if (EMIT == 0) out[slot++] = i + offset;
            else keep[i] = 1;
        }
    }
    if (EMIT == 0 && b == 0)
        for (int i = s_need + tid; i < out_w; i += 256) out[i] = -1;
}

static int v41_sn_launch(const V41Keys& K, int k, int* ws, int* out, int out_w, int offset, uint8_t* keep,
                         cudaStream_t st) {
    int nb = (K.n + V41_SN_CH - 1) / V41_SN_CH;
    if (nb > V41_SN_BLOCKS || !ws) return 41010;
    v41_sn_hist_kernel<0><<<nb, 256, 0, st>>>(K, k, ws);
    v41_sn_hist_kernel<1><<<nb, 256, 0, st>>>(K, k, ws);
    v41_sn_count_kernel<<<nb, 256, 0, st>>>(K, k, ws);
    if (keep) v41_sn_emit_kernel<1><<<nb, 256, 0, st>>>(K, k, ws, nullptr, 0, 0, keep);
    else v41_sn_emit_kernel<0><<<nb, 256, 0, st>>>(K, k, ws, out, out_w, offset, nullptr);
    V41_ERR();
}

// `memra_v41_topk` for one row, over a zeroed int scratch of V41_SN_CNT + 2 x V41_SN_BLOCKS words.
extern "C" int memra_v41_topk_long(const void* score, int T, int k, int out_w, int offset, int* out, int* ws,
                                   void* stream) {
    if (k < 0 || k > out_w || T <= 0) return 41010;
    V41Keys K{(const v41_bf*)score, T, 0, T, -1};
    return v41_sn_launch(K, k, ws, out, out_w, offset, nullptr, (cudaStream_t)stream);
}

// `memra_v41_candidates` for one row (the query at `start_pos`), over the same scratch.
extern "C" int memra_v41_candidates_long(const void* score, int T, int cb, int kblocks, int start_pos, int ratio,
                                         uint8_t* keep, int* ws, void* stream) {
    if (cb <= 0 || ratio <= 0 || kblocks < 0 || T <= 0 || !keep) return 41011;
    int visible = (start_pos + 1) / ratio;
    V41Keys K{(const v41_bf*)score, (T + cb - 1) / cb, cb, T, visible > 0 ? (visible - 1) / cb : -1};
    return v41_sn_launch(K, kblocks, ws, nullptr, 0, 0, keep, (cudaStream_t)stream);
}

// ---------------------------------------------------------------------------------------------
// MoE gate (sqrtsoftplus): score = sqrt(softplus(raw)); experts by score + bias (top-k, ties to the
// lower id, descending); weights are the unbiased scores, normalized by (sum + 1e-20) only when
// top-k > 1, times route_scale. One warp per token; ne <= 1024, topk <= 16.

// PER: the most experts a lane holds (ne <= 32 * PER). Every loop over a lane's experts has that fixed
// count and is unrolled, so the lane's values stay in registers (an array indexed with a runtime
// bound lives in local memory).
template <int PER>
__global__ void v41_route_kernel(const float* __restrict__ raw, long ldr, const float* __restrict__ bias, int ne,
                                 int topk, int norm, float route_scale, int* __restrict__ sel, float* __restrict__ selw,
                                 int rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const float* r = raw + (long)wid * ldr;
    float sc[PER], bs[PER], rv[PER], bv[PER];
    int per = (ne + 31) / 32;
    // every load of the row and the bias first, so none waits behind another
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int e = lane + 32 * i;
        bool ok = i < per && e < ne;
        rv[i] = ok ? r[e] : 0.f;
        bv[i] = (ok && bias) ? bias[e] : 0.f;
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        int e = lane + 32 * i;
        sc[i] = 0.f;
        bs[i] = -INFINITY;
        if (i < per) {
            float v = rv[i];
            float sp = v > 20.f ? v : log1pf(expf(v));
            sc[i] = sqrtf(sp);
            bs[i] = e < ne ? __fadd_rn(sc[i], bv[i]) : -INFINITY;
        }
    }
    float wsel[16];
    int isel[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        wsel[j] = 0.f;
        isel[j] = 0;
        if (j >= topk) continue;
        float best = -INFINITY;
        int bi = 0x7FFFFFFF;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int e = lane + 32 * i;
            if (i < per && e < ne && (bs[i] > best || (bs[i] == best && e < bi))) {
                best = bs[i];
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            float ob = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        // a row with fewer than topk orderable scores (a NaN raw value: its score never compares)
        // would pick an expert already taken (they sit at -inf, which equals the start); the slot
        // names none instead, which the plan counts as out of range and skips
        if (!(best > -INFINITY)) bi = -1;
        isel[j] = bi;
        float wv = 0.f;
        if (bi >= 0 && (bi & 31) == lane) {
            int slot = bi >> 5;
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (i == slot && i < per) {
                    wv = sc[i];
                    bs[i] = -INFINITY;
                }
        }
        wsel[j] = __shfl_sync(0xffffffffu, wv, bi < 0 ? 0 : bi & 31);
    }
    if (lane == 0) {
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (j < topk) sum = __fadd_rn(sum, wsel[j]);
        float den = __fadd_rn(sum, 1e-20f);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j >= topk) continue;
            float wv = (norm && topk > 1) ? wsel[j] / den : wsel[j];
            sel[(long)wid * topk + j] = isel[j];
            selw[(long)wid * topk + j] = __fmul_rn(wv, route_scale);
        }
    }
}

extern "C" int memra_v41_route(const float* raw, long ldr, const float* bias, int ne, int topk, int norm,
                               float route_scale, int* sel, float* selw, int rows, void* stream) {
    if (ne <= 0 || ne > 1024 || topk <= 0 || topk > 16 || topk > ne) return 41012;
    if (rows <= 0) return rows < 0 ? 41012 : 0;
    if (ne <= 32 * 12)
        v41_route_kernel<12><<<(rows * 32 + 127) / 128, 128, 0, (cudaStream_t)stream>>>(raw, ldr, bias, ne, topk, norm,
                                                                                       route_scale, sel, selw, rows);
    else
        v41_route_kernel<32><<<(rows * 32 + 127) / 128, 128, 0, (cudaStream_t)stream>>>(raw, ldr, bias, ne, topk, norm,
                                                                                       route_scale, sel, selw, rows);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// SwiGLU with the training clamps: up clamped to [-L, L], gate to (-inf, L]; h = silu(g) * u, times
// the routing weight when given, bf16 out. gate/up bf16 [rows][inter] (ld elements).

__device__ __forceinline__ float v41_swiglu1(float g, float u, float limit) {
    if (limit > 0.f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    return __fmul_rn(g / __fadd_rn(1.f, expf(-g)), u);
}

__global__ void v41_swiglu_kernel(const v41_bf* __restrict__ gate, const v41_bf* __restrict__ up, long ld,
                                  const float* __restrict__ wrow, float limit, v41_bf* __restrict__ out, long ldo,
                                  int inter) {
    int r = blockIdx.x;
    float wv = wrow ? wrow[r] : 1.f;
    for (int i = threadIdx.x; i < inter; i += blockDim.x) {
        float h = v41_swiglu1(v41_bf2f(gate[(long)r * ld + i]), v41_bf2f(up[(long)r * ld + i]), limit);
        if (wrow) h = __fmul_rn(wv, h);
        out[(long)r * ldo + i] = v41_f2bf(h);
    }
}

// w1, w3 and the SwiGLU of a small-row tile list in one launch. A block is 4 warps, two pairs; a pair owns CW
// output columns, its first warp runs w1's chain (the small-row body, v41_sk_chain) and its second w3's, at the
// same time; then the up warp hands its sums to the gate warp through shared memory (the same lane holds the
// same element in both), which rounds both to bf16 and stores v41_swiglu_kernel's value (times the pair's
// routing weight) straight into the hidden rows. Per element this is the three launches' program: both sums
// are the small-row body's chains, both are rounded to bf16 before the SwiGLU as the stored tensors were, and
// the SwiGLU is the same function.
template <int WF, int CW>
__global__ void __launch_bounds__(128) v41_swiglu_bs_tiles_kernel(
    const uint8_t* __restrict__ A, long lda, const uint8_t* __restrict__ As, long ldas,
    const V41Tile* __restrict__ tiles, const int* __restrict__ ntiles, const V41ExpW* __restrict__ ew, long ldb,
    long ldbs, const float* __restrict__ wrow, float limit, v41_bf* __restrict__ H, long ldh, int N, int K, int BM) {
    using P = V41Sk<WF, CW, 1>;
    extern __shared__ __align__(16) uint8_t v41_swb_sm[];
    v41_smem_poison(v41_swb_sm);
    const V41Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const V41ExpW& w = ew[tl.expert];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int pair = warp >> 1, up = warp & 1, proj = up ? 2 : 0;
    long t0 = (long)tl.row0 + tl.m0;
    int M = min(tl.rows - tl.m0, BM);
    int rw0 = (blockIdx.x * 2 + pair) * CW;
    float acc[1][4];
    v41_sk_chain<WF, CW, 1>(A + t0 * lda, lda, As + t0 * ldas, ldas, w.w[proj], ldb, w.s[proj], ldbs, w.scale2[proj], M,
                            N, K, rw0, BM, v41_swb_sm + warp * P::warp(BM), acc);
    float* x = (float*)(v41_swb_sm + 4 * P::warp(BM)) + pair * 128;
    if (up) {
#pragma unroll
        for (int e = 0; e < 4; ++e) x[lane * 4 + e] = acc[0][e];
    }
    __syncthreads();
    if (!up) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int wr = g + (e >= 2 ? 8 : 0), row = 2 * t + (e & 1), col = rw0 + wr;
            if (wr < CW && row < M && col < N) {
                float h = v41_swiglu1(v41_bf2f(v41_f2bf(acc[0][e])), v41_bf2f(v41_f2bf(x[lane * 4 + e])), limit);
                if (wrow) h = __fmul_rn(wrow[t0 + row], h);
                H[(t0 + row) * ldh + col] = v41_f2bf(h);
            }
        }
    }
}

// The routed experts' hidden rows of a small-row tile list: h[pair][n] = SwiGLU(w1 a, w3 a) * wrow[pair], bf16,
// for the tiles' experts (FP4, wfmt 1 or 2), every tile of at most bm <= V41_SM_MAX rows.
extern "C" int memra_v41_swiglu_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                                      const int* ntiles, int max_tiles, const void* ew, long ldb, long ldbs,
                                      int wfmt, const float* wrow, float limit, void* h, long ldh, int n, int k,
                                      int bm, void* stream) {
    if (k % 32 || (k / 32) % 4 || n <= 0 || lda % 16 || ldb % 16 || ldas % 4 || ldbs % 4 || max_tiles < 0 ||
        max_tiles > 65535 || bm <= 0 || bm > V41_SM_MAX || (wfmt != 1 && wfmt != 2) || (uintptr_t)a % 16)
        return 41043;
    if (max_tiles == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    // a pair of warps a column group: as many column groups as the plain list would run warps
    int cw = v41_sk_cw(n, max_tiles);
#define V41_SWB(WF, CWX)                                                                                      \
    do {                                                                                                      \
        int sm_ = 4 * V41Sk<WF, CWX, 1>::warp(bm) + 2 * 128 * 4;                                              \
        if (sm_ > 48 * 1024)                                                                                  \
            V41_SMEM_LIMIT((v41_swiglu_bs_tiles_kernel<WF, CWX>), (4 * V41Sk<WF, CWX, 1>::warp(V41_SM_MAX) + 2 * 128 * 4)); \
        dim3 gr((n + 2 * CWX - 1) / (2 * CWX), max_tiles);                                                    \
        v41_swiglu_bs_tiles_kernel<WF, CWX><<<gr, 128, sm_, st>>>(                                            \
            (const uint8_t*)a, lda, (const uint8_t*)as, ldas, (const V41Tile*)tiles, ntiles, (const V41ExpW*)ew, \
            ldb, ldbs, wrow, limit, (v41_bf*)h, ldh, n, k, bm);                                               \
    } while (0)
#define V41_SWB_CW(WF) \
    do { if (cw == 16) V41_SWB(WF, 16); else if (cw == 8) V41_SWB(WF, 8); else V41_SWB(WF, 4); } while (0)
    if (wfmt == 1) V41_SWB_CW(1);
    else V41_SWB_CW(2);
#undef V41_SWB_CW
#undef V41_SWB
    V41_ERR();
}

extern "C" int memra_v41_swiglu(const void* gate, const void* up, long ld, const float* wrow, float limit, void* out,
                                long ldo, int rows, int inter, void* stream) {
    if (rows <= 0) return rows < 0 ? 41013 : 0;
    v41_swiglu_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)gate, (const v41_bf*)up, ld, wrow, limit,
                                                              (v41_bf*)out, ldo, inter);
    V41_ERR();
}

// y[rows_of[r]][:] += float(src[r][:]) (fp32 accumulator; launches in ascending expert order keep
// the reference's per-token summation order)
__global__ void v41_scatter_add_kernel(const v41_bf* __restrict__ src, long lds, const int* __restrict__ rows_of,
                                       float* __restrict__ y, long ldy, int d) {
    int r = blockIdx.x;
    float* yr = y + (long)rows_of[r] * ldy;
    for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = __fadd_rn(yr[i], v41_bf2f(src[(long)r * lds + i]));
}

extern "C" int memra_v41_scatter_add(const void* src, long lds, const int* rows_of, float* y, long ldy, int rows, int d,
                                     void* stream) {
    if (rows <= 0) return rows < 0 ? 41014 : 0;
    v41_scatter_add_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)src, lds, rows_of, y, ldy, d);
    V41_ERR();
}

// out = bf16(y + float(add)) (add nullable)
__global__ void v41_add_cast_kernel(const float* __restrict__ y, const v41_bf* __restrict__ add, v41_bf* __restrict__ out,
                                    long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = y[i];
    if (add) v = __fadd_rn(v, v41_bf2f(add[i]));
    out[i] = v41_f2bf(v);
}

extern "C" int memra_v41_add_cast(const float* y, const void* add, void* out, long n, void* stream) {
    if (n <= 0) return n < 0 ? 41015 : 0;
    v41_add_cast_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, (const v41_bf*)add,
                                                                                       (v41_bf*)out, n);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// The MoE dispatch plan on the device, the host `tp::MoePlan` order exactly: pairs sorted by expert
// with tokens ascending inside an expert, each token's pair indices in ascending expert id, plus the
// tile list of the grouped expert GEMMs (one entry per `bm` rows of an expert that has rows). No
// host round trip: per-chunk expert counts, one block scanning them, then each chunk placing its
// tokens in order. `err` counts expert ids outside [0, ne) (skipped, their order slot -1), and such an id
// sets bit 0 of `*flag` when it is given (the rank's call flag, read with the call's logits).

#define V41_PLAN_CH 256

__global__ void __launch_bounds__(V41_PLAN_CH) v41_moe_count_kernel(const int* __restrict__ sel, int rows, int topk,
                                                                     int ne, int* __restrict__ counts,
                                                                     int* __restrict__ err, int* flag) {
    extern __shared__ int v41_cnt[];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) v41_cnt[e] = 0;
    __syncthreads();
    int t = blockIdx.x * V41_PLAN_CH + threadIdx.x;
    if (t < rows)
        for (int j = 0; j < topk; ++j) {
            int e = sel[(long)t * topk + j];
            if (e < 0 || e >= ne) {
                atomicAdd(err, 1);
                if (flag) atomicOr(flag, 1);
            } else {
                atomicAdd(&v41_cnt[e], 1);
            }
        }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) counts[(long)blockIdx.x * ne + e] = v41_cnt[e];
}

// counts [nch][ne] -> each chunk's base inside its expert (in place); off [ne + 1]; the tile list of
// the experts [e0, e1) (a rank's share), expert index relative to e0
__global__ void __launch_bounds__(1024) v41_moe_scan_kernel(int* __restrict__ counts, int nch, int ne, int bm,
                                                            int e0, int e1, int* __restrict__ off,
                                                            V41Tile* __restrict__ tiles, int* __restrict__ ntiles) {
    __shared__ int s_off[1024], s_til[1024];
    int e = threadIdx.x;
    int run = 0;
    if (e < ne)
        for (int c = 0; c < nch; ++c) {
            int v = counts[(long)c * ne + e];
            counts[(long)c * ne + e] = run;
            run += v;
        }
    s_off[e] = e < ne ? run : 0;
    s_til[e] = (e >= e0 && e < e1) ? (run + bm - 1) / bm : 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, tt = 0;
        for (int x = 0; x < ne; ++x) {
            int c = s_off[x], n = s_til[x];
            s_off[x] = o;
            s_til[x] = tt;
            o += c;
            tt += n;
        }
        off[ne] = o;
        *ntiles = tt;
    }
    __syncthreads();
    if (e < ne) {
        off[e] = s_off[e];
        if (e >= e0 && e < e1)
            for (int i = 0; i * bm < run; ++i) tiles[s_til[e] + i] = V41Tile{s_off[e], run, i * bm, e - e0};
    }
}

// One thread per token of a 256-token chunk. A pair's slot is its expert's base plus the number of
// earlier tokens of the chunk that picked the same expert (a token's experts are distinct): the
// chunk's tokens set their bit in a per-expert bitmap, and a token counts the bits below its own.
__global__ void __launch_bounds__(V41_PLAN_CH) v41_moe_place_kernel(const int* __restrict__ sel,
                                                                    const float* __restrict__ selw, int rows, int topk,
                                                                    int ne, const int* __restrict__ bases,
                                                                    const int* __restrict__ off, int* __restrict__ ptok,
                                                                    float* __restrict__ pw, int* __restrict__ order) {
    extern __shared__ int v41_loc[];  // [ne] the chunk's first slot per expert, then [ne][8] the token bitmap
    unsigned* mask = (unsigned*)(v41_loc + ne);
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += V41_PLAN_CH) v41_loc[e] = off[e] + bases[(long)blockIdx.x * ne + e];
    for (int i = tid; i < ne * (V41_PLAN_CH / 32); i += V41_PLAN_CH) mask[i] = 0u;
    __syncthreads();
    int t = blockIdx.x * V41_PLAN_CH + tid;
    int ev[16];
    float wv[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        ev[j] = -1;
        wv[j] = 0.f;
        if (j < topk && t < rows) {
            int e = sel[(long)t * topk + j];
            wv[j] = selw[(long)t * topk + j];
            if (e >= 0 && e < ne) ev[j] = e;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
        if (ev[j] >= 0) atomicOr(&mask[ev[j] * (V41_PLAN_CH / 32) + (tid >> 5)], 1u << (tid & 31));
    __syncthreads();
    if (t >= rows) return;
    int n = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) n += ev[j] >= 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int e = ev[j];
        if (e < 0) continue;
        const unsigned* m = mask + e * (V41_PLAN_CH / 32);
        int before = __popc(m[tid >> 5] & ((1u << (tid & 31)) - 1u));
#pragma unroll
        for (int w = 0; w < V41_PLAN_CH / 32; ++w)
            if (w < (tid >> 5)) before += __popc(m[w]);
        int p = v41_loc[e] + before;
        ptok[p] = t;
        pw[p] = wv[j];
        int q = 0;  // this expert's rank among the token's experts: its place in the per-token order
#pragma unroll
        for (int k = 0; k < 16; ++k) q += (ev[k] >= 0 && ev[k] < e);
        order[(long)t * topk + q] = p;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
        if (j < topk && j >= n) order[(long)t * topk + j] = -1;
}

// The plan of a call of a few rows (at most V41_PLAN_SMALL pairs) in one block instead of three kernels and
// three zero-fills: the same outputs (the gate compares both forms with the host plan). Nothing is zeroed
// beforehand: the slots past the placed pairs are written here (token 0, weight 0), and err is stored.
#define V41_PLAN_SMALL 256

__global__ void __launch_bounds__(1024) v41_moe_plan_small_kernel(const int* __restrict__ sel,
                                                                  const float* __restrict__ selw, int rows, int topk,
                                                                  int ne, int bm, int e0, int e1, int* __restrict__ off,
                                                                  int* __restrict__ ptok, float* __restrict__ pw,
                                                                  int* __restrict__ order, V41Tile* __restrict__ tiles,
                                                                  int* __restrict__ ntiles, int* __restrict__ err,
                                                                  int* flag) {
    __shared__ int s_sel[V41_PLAN_SMALL], s_off[1024], s_til[1024], s_placed;
    int tid = threadIdx.x, np = rows * topk;
    if (tid < np) {
        int e = sel[tid];
        s_sel[tid] = (e >= 0 && e < ne) ? e : -1;
    }
    __syncthreads();
    int run = 0;  // expert tid's pairs
    if (tid < ne)
        for (int p = 0; p < np; ++p) run += s_sel[p] == tid;
    s_off[tid] = run;
    s_til[tid] = (tid >= e0 && tid < e1) ? (run + bm - 1) / bm : 0;
    __syncthreads();
    if (tid == 0) {
        int o = 0, tt = 0, bad = 0;
        for (int x = 0; x < ne; ++x) {
            int c = s_off[x], n = s_til[x];
            s_off[x] = o;
            s_til[x] = tt;
            o += c;
            tt += n;
        }
        for (int p = 0; p < np; ++p) bad += s_sel[p] < 0;
        off[ne] = o;
        *ntiles = tt;
        *err = bad;
        if (bad && flag) atomicOr(flag, 1);
        s_placed = o;
    }
    __syncthreads();
    if (tid < ne) {
        off[tid] = s_off[tid];
        if (tid >= e0 && tid < e1)
            for (int i = 0; i * bm < run; ++i) tiles[s_til[tid] + i] = V41Tile{s_off[tid], run, i * bm, tid - e0};
    }
    if (tid < np) {
        int t = tid / topk, j = tid % topk, e = s_sel[tid];
        int n = 0, q = 0;  // the token's experts in range, and those below e: e's place in the per-token order
        for (int k = 0; k < topk; ++k) {
            int x = s_sel[t * topk + k];
            n += x >= 0;
            q += x >= 0 && x < e;
        }
        if (e >= 0) {
            int before = 0;  // earlier tokens that picked e (a token's experts are distinct)
            for (int p = 0; p < t * topk; ++p) before += s_sel[p] == e;
            int p = s_off[e] + before;
            ptok[p] = t;
            pw[p] = selw[tid];
            order[t * topk + q] = p;
        }
        if (j >= n) order[t * topk + j] = -1;
        if (tid >= s_placed) {
            ptok[tid] = 0;
            pw[tid] = 0.f;
        }
    }
}

// counts: scratch [ceil(rows / 256)][ne] ints (unused for at most V41_PLAN_SMALL pairs); tiles: room for
// ceil(rows * topk / bm) + min(ne, rows * topk); the tile list covers experts [e0, e1)
extern "C" int memra_v41_moe_plan(const int* sel, const float* selw, int rows, int topk, int ne, int bm, int e0, int e1,
                                  int* counts, int* off, int* ptok, float* pw, int* order, void* tiles, int* ntiles,
                                  int* err, int* flag, void* stream) {
    if (rows <= 0) return rows < 0 ? 41025 : 0;
    if (ne <= 0 || ne > 1024 || topk <= 0 || topk > 16 || bm <= 0 || e0 < 0 || e1 > ne || e0 > e1) return 41025;
    cudaStream_t st = (cudaStream_t)stream;
    if (rows * topk <= V41_PLAN_SMALL) {
        v41_moe_plan_small_kernel<<<1, 1024, 0, st>>>(sel, selw, rows, topk, ne, bm, e0, e1, off, ptok, pw, order,
                                                     (V41Tile*)tiles, ntiles, err, flag);
        V41_ERR();
    }
    int nch = (rows + V41_PLAN_CH - 1) / V41_PLAN_CH;
    v41_moe_count_kernel<<<nch, V41_PLAN_CH, ne * 4, st>>>(sel, rows, topk, ne, counts, err, flag);
    v41_moe_scan_kernel<<<1, 1024, 0, st>>>(counts, nch, ne, bm, e0, e1, off, (V41Tile*)tiles, ntiles);
    v41_moe_place_kernel<<<nch, V41_PLAN_CH, ne * 4 * (1 + V41_PLAN_CH / 32), st>>>(sel, selw, rows, topk, ne, counts,
                                                                                  off, ptok, pw, order);
    V41_ERR();
}

// The routed-expert combine of the grouped dispatch: out[t] = bf16(sum_j float(slots[order[t][j]]) +
// float(shared[t])), j in ascending expert id, fp32 from zero. This is the per-expert loop's sum
// (scatter_add launched in ascending expert order into a zeroed accumulator, then add_cast) in the same
// order, so the result does not depend on which rank computed which expert. order -1 skips a slot.
__global__ void v41_moe_combine_kernel(const v41_bf* __restrict__ slots, long lds, const int* __restrict__ order,
                                       int topk, const v41_bf* __restrict__ shared, long ldsh, v41_bf* __restrict__ out,
                                       long ldo, int d) {
    int t = blockIdx.x;
    const int* o = order + (long)t * topk;
    int i1 = min(d, (int)(blockIdx.y + 1) * 1024);
    for (int i = blockIdx.y * 1024 + threadIdx.x; i < i1; i += blockDim.x) {
        float v = 0.f;
        for (int j = 0; j < topk; ++j) {
            int p = o[j];
            if (p >= 0) v = __fadd_rn(v, v41_bf2f(slots[(long)p * lds + i]));
        }
        if (shared) v = __fadd_rn(v, v41_bf2f(shared[(long)t * ldsh + i]));
        out[(long)t * ldo + i] = v41_f2bf(v);
    }
}

extern "C" int memra_v41_moe_combine(const void* slots, long lds, const int* order, int topk, const void* shared,
                                     long ldsh, void* out, long ldo, int rows, int d, void* stream) {
    if (rows <= 0) return rows < 0 ? 41017 : 0;
    if (topk <= 0 || d <= 0) return 41017;
    dim3 grid(rows, (d + 1023) / 1024);
    v41_moe_combine_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)slots, lds, order, topk,
                                                                   (const v41_bf*)shared, ldsh, (v41_bf*)out, ldo, d);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Single-pass mHC. The stream is bf16 [s][hc][d].
// mixes[s][(2+hc)*hc] = (flat . fn^T) * rsqrt(mean(flat^2) + eps), flat = the hc*d stream in fp32.
//
// The arithmetic is a 256-thread block per token: thread i sums x[i'] * fn[j][i'] (and x[i']^2) over
// i' = i, i + 256, ... in that order, each sum then goes through the xor-butterfly warp sum, and the
// 8 warp sums are added in warp order. The partial pass keeps every one of those sums and orders but
// lays the 256 threads out as 8 virtual warps: block (token group, virtual warp b) runs, in its warp
// j, the chain of mix j (warp nmix: the square sum) for V41_HCT tokens at once, with the virtual
// warp's x values staged once in shared memory for all 25 chains. fn is read once per V41_HCT tokens
// instead of once per token, and a one-token decode spreads over 8 blocks.
// parts[s][8][nmix + 1] holds each virtual warp's sums.

// V41_HCT tokens per block, V41_HCM strided steps staged at a time: 16 x 16 for a prefill, 1 x 40 for
// a one-token decode whose staging does not fit `v41_hc_mixes_part1_kernel`. 25 warps per block, so
// at most 80 registers per thread.
template <int V41_HCT, int V41_HCM>
__global__ void __launch_bounds__(800, 1) v41_hc_mixes_part_kernel(const v41_bf* __restrict__ x,
                                                                   const float* __restrict__ fn, int nmix, int n,
                                                                   int rows, float* __restrict__ parts, int b0) {
    __shared__ __align__(16) v41_bf sx[V41_HCM][V41_HCT][32];  // raw bf16, widened where it is read
    int b = blockIdx.y + b0;  // virtual warp (b0: the first of a rank's share, see memra_v41_hc_mixes_part)
    int s0 = blockIdx.x * V41_HCT;
    int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nt = min(V41_HCT, rows - s0);
    int vt = b * 32 + lane;  // the 256-thread kernel's thread id
    const float* f = fn + (long)(j < nmix ? j : 0) * n;
    float acc[V41_HCT];
#pragma unroll
    for (int t = 0; t < V41_HCT; ++t) acc[t] = 0.f;
    for (int m0 = 0; m0 * 256 < n; m0 += V41_HCM) {
        // this chain's fn values of the chunk, in flight while x is staged
        float fv[V41_HCM];
#pragma unroll
        for (int mm = 0; mm < V41_HCM; ++mm) {
            int i = vt + 256 * (m0 + mm);
            fv[mm] = (j < nmix && i < n) ? __ldg(f + i) : 0.f;
        }
        __syncthreads();  // the previous steps' readers are done with sx
        // this virtual warp's x values of the next V41_HCM steps, once for the 25 chains that read them:
        // 16-byte chunks through cp.async, every copy in flight (a step's 32 values are 64 contiguous
        // bytes of a token's row; n is a multiple of 8 and a chunk past n lands as zeros)
        for (int q = threadIdx.x; q < V41_HCM * nt * 4; q += blockDim.x) {
            int mm = q / (nt * 4), t = (q >> 2) % nt, c = q & 3;
            long i = b * 32 + c * 8 + 256L * (m0 + mm);
            bool ok = i < n;
            v41_cp16(&sx[mm][t][c * 8], x + (ok ? (long)(s0 + t) * n + i : 0), ok);
        }
        v41_cp_commit();
        v41_cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int mm = 0; mm < V41_HCM; ++mm) {
            int i = vt + 256 * (m0 + mm);  // the original thread's i, in its order
            if (i < n) {
#pragma unroll
                for (int t = 0; t < V41_HCT; ++t) {
                    if (t < nt) {
                        float v = v41_bf2f(sx[mm][t][lane]);
                        acc[t] = __fadd_rn(acc[t], j < nmix ? __fmul_rn(v, fv[mm]) : __fmul_rn(v, v));
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < V41_HCT; ++t) {
        if (t < nt) {
            float v = v41_warp_sum(acc[t]);
            if (lane == 0) parts[((long)(s0 + t) * 8 + b) * (nmix + 1) + j] = v;
        }
    }
}

// The partial pass of one token (decode): block (virtual warp b, chain group z) runs 4 chains, one per
// warp, over the whole token. The virtual warp's x values and each chain's fn values land in shared
// memory through cp.async with every copy in flight, so the pass waits on one memory round trip; a
// chain then adds its products in the strided order, as above. smem: steps x (64 + 4 x 128) bytes.
__global__ void __launch_bounds__(128) v41_hc_mixes_part1_kernel(const v41_bf* __restrict__ x,
                                                                 const float* __restrict__ fn, int nmix, int n,
                                                                 float* __restrict__ parts) {
    extern __shared__ __align__(16) uint8_t v41_hc1_sm[];
    // block column = token: each of a few rows (a decode step, a drafted block's verification, a
    // batch of sequences) gets the one-token schedule on its own blocks
    x += (long)blockIdx.x * n;
    parts += (long)blockIdx.x * 8 * (nmix + 1);
    int steps = (n + 255) / 256;
    int b = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int j = blockIdx.z * 4 + w;  // chain: a mix, the square sum (nmix), or none
    v41_bf* sx = (v41_bf*)v41_hc1_sm;                                       // [steps][32]
    float* sf = (float*)(v41_hc1_sm + steps * 64) + (long)w * steps * 32;    // [steps][32], this warp's chain
    for (int q = threadIdx.x; q < steps * 4; q += 128) {
        int m = q >> 2, c = q & 3;
        long i = b * 32 + c * 8 + 256L * m;
        bool ok = i < n;
        v41_cp16(sx + m * 32 + c * 8, x + (ok ? i : 0), ok);
    }
    if (j < nmix) {
        for (int q = lane; q < steps * 8; q += 32) {
            int m = q >> 3, c = q & 7;
            long i = b * 32 + c * 4 + 256L * m;
            bool ok = i < n;
            v41_cp16(sf + m * 32 + c * 4, fn + (long)j * n + (ok ? i : 0), ok);
        }
    }
    v41_cp_commit();
    v41_cp_wait<0>();
    __syncthreads();
    int vt = b * 32 + lane;  // the 256-thread kernel's thread id
    float acc = 0.f;
    for (int m = 0; m < steps; ++m) {
        if (vt + 256 * m < n) {
            float v = v41_bf2f(sx[m * 32 + lane]);
            acc = __fadd_rn(acc, j < nmix ? __fmul_rn(v, sf[m * 32 + lane]) : __fmul_rn(v, v));
        }
    }
    float v = v41_warp_sum(acc);
    if (lane == 0 && j <= nmix) parts[(long)b * (nmix + 1) + j] = v;
}

// The mixes from the partials (the 8 warp sums added in order, times rsqrt(mean square + eps)), then
// pre/post/comb (reference hc_split_sinkhorn) with 16 lanes per token: lane (j, k) holds comb[j][k];
// a row or column sum gathers its hc lanes and adds them in the reference order in each lane, so
// every lane holds exactly the serial value. hc <= 4.
__global__ void __launch_bounds__(256) v41_hc_finish_kernel(const float* __restrict__ parts, int nmix, int n,
                                                            float norm_eps, const float* __restrict__ scale,
                                                            const float* __restrict__ base, int hc, int iters,
                                                            float eps, float* __restrict__ mixes,
                                                            float* __restrict__ pre, float* __restrict__ post,
                                                            float* __restrict__ comb, int rows) {
    __shared__ float sm[16][32];
    int tl = threadIdx.x >> 4, l = threadIdx.x & 15;
    int s = blockIdx.x * 16 + tl;
    bool ok = s < rows;
    if (ok) {
        const float* p = parts + (long)s * 8 * (nmix + 1);
        float tot = 0.f;
        for (int b = 0; b < 8; ++b) tot = __fadd_rn(tot, p[b * (nmix + 1) + nmix]);
        float rs = rsqrtf(__fadd_rn(tot / (float)n, norm_eps));
        for (int jj = l; jj < nmix; jj += 16) {
            float v = 0.f;
            for (int b = 0; b < 8; ++b) v = __fadd_rn(v, p[b * (nmix + 1) + jj]);
            float mv = __fmul_rn(v, rs);
            mixes[(long)s * nmix + jj] = mv;
            sm[tl][jj] = mv;
        }
    }
    __syncwarp();
    int j = l >> 2, k = l & 3;
    bool act = ok && j < hc && k < hc;
    const float* m = sm[tl];
    if (act && k == 0) {
        float a = __fadd_rn(__fmul_rn(m[j], scale[0]), base[j]);
        pre[(long)s * hc + j] = __fadd_rn(1.f / __fadd_rn(1.f, expf(-a)), eps);
    }
    if (act && k == 1) {
        float bb = __fadd_rn(__fmul_rn(m[hc + j], scale[1]), base[hc + j]);
        post[(long)s * hc + j] = __fmul_rn(2.f, 1.f / __fadd_rn(1.f, expf(-bb)));
    }
    float c = 0.f;
    if (act) c = __fadd_rn(__fmul_rn(m[2 * hc + j * hc + k], scale[2]), base[2 * hc + j * hc + k]);
    // the lanes of row j are 4j + k', of column k are 4j' + k (within this token's 16)
    auto row_max = [&](float v) {
        float mx = -INFINITY;
        for (int kk = 0; kk < 4; ++kk) {
            float o = __shfl_sync(0xffffffffu, v, 4 * j + kk, 16);
            if (kk < hc) mx = fmaxf(mx, o);
        }
        return mx;
    };
    auto row_sum = [&](float v) {
        float sum = 0.f;
        for (int kk = 0; kk < 4; ++kk) {
            float o = __shfl_sync(0xffffffffu, v, 4 * j + kk, 16);
            if (kk < hc) sum = __fadd_rn(sum, o);
        }
        return sum;
    };
    auto col_sum = [&](float v) {
        float sum = 0.f;
        for (int jj = 0; jj < 4; ++jj) {
            float o = __shfl_sync(0xffffffffu, v, 4 * jj + k, 16);
            if (jj < hc) sum = __fadd_rn(sum, o);
        }
        return sum;
    };
    float mx = row_max(c);
    c = expf(c - mx);
    float sum = row_sum(c);
    c = __fadd_rn(c / sum, eps);
    float cs = col_sum(c);
    c = c / __fadd_rn(cs, eps);
    for (int it = 0; it < iters - 1; ++it) {
        float rsum = row_sum(c);
        c = c / __fadd_rn(rsum, eps);
        float cs2 = col_sum(c);
        c = c / __fadd_rn(cs2, eps);
    }
    if (act) comb[(long)s * hc * hc + j * hc + k] = c;
}

extern "C" int memra_v41_hc_mixes(const void* x, const float* fn, const float* scale, const float* base, int rows,
                                  int hc, int d, int iters, float norm_eps, float hc_eps, float* parts, float* mixes,
                                  float* pre, float* post, float* comb, void* stream) {
    int nmix = (2 + hc) * hc;
    if (hc <= 0 || hc > 4 || nmix > 31) return 41016;
    if ((hc * d) % 8 || (uintptr_t)x % 16) return 41016;  // the stream is staged in 16-byte chunks
    if (rows <= 0) return rows < 0 ? 41016 : 0;
    cudaStream_t st = (cudaStream_t)stream;
    int n1 = hc * d, smem1 = ((n1 + 255) / 256) * (64 + 4 * 128);
    if ((rows <= V41_SM_MAX || rows <= v41_wide(3)) && smem1 <= 99 * 1024 && n1 % 8 == 0 && (uintptr_t)fn % 16 == 0) {
        // a few rows: the one-token kernel a row (6.7 us, where the 16-token pass takes 53)
        if (smem1 > 48 * 1024) V41_SMEM_LIMIT(v41_hc_mixes_part1_kernel, 99 * 1024);
        v41_hc_mixes_part1_kernel<<<dim3(rows, 8, (nmix + 4) / 4), 128, smem1, st>>>((const v41_bf*)x, fn, nmix, n1,
                                                                                   parts);
    } else if (rows == 1) {
        v41_hc_mixes_part_kernel<1, 40><<<dim3(1, 8), 32 * (nmix + 1), 0, st>>>((const v41_bf*)x, fn, nmix, hc * d,
                                                                               rows, parts, 0);
    } else {
        dim3 g1((rows + 15) / 16, 8);
        v41_hc_mixes_part_kernel<16, 16><<<g1, 32 * (nmix + 1), 0, st>>>((const v41_bf*)x, fn, nmix, hc * d, rows,
                                                                          parts, 0);
    }
    v41_hc_finish_kernel<<<(rows + 15) / 16, 256, 0, st>>>(parts, nmix, hc * d, norm_eps, scale, base, hc, iters,
                                                           hc_eps, mixes, pre, post, comb, rows);
    V41_ERR();
}

// The many-token partial pass for the virtual warps [b0, b0 + nb) only, and the finish on its own. The 8
// virtual warps' sums are independent of each other and every rank holds the whole stream, so on N ranks
// each rank runs 8 / N of them, the sums are gathered, and every rank runs the finish: the same sums in
// the same orders as memra_v41_hc_mixes, a rank reading 1 / N of the mix weights. More than V41_SM_MAX rows
// (below that the one-token kernel a row is one launch and a gather would cost more than it saves).
extern "C" int memra_v41_hc_mixes_part(const void* x, const float* fn, int rows, int hc, int d, int b0, int nb,
                                       float* parts, void* stream) {
    int nmix = (2 + hc) * hc;
    if (hc <= 0 || hc > 4 || nmix > 31 || b0 < 0 || nb <= 0 || b0 + nb > 8) return 41047;
    if ((hc * d) % 8 || (uintptr_t)x % 16) return 41047;
    if (rows <= V41_SM_MAX) return 41047;
    dim3 g1((rows + 15) / 16, nb);
    v41_hc_mixes_part_kernel<16, 16><<<g1, 32 * (nmix + 1), 0, (cudaStream_t)stream>>>((const v41_bf*)x, fn, nmix,
                                                                                       hc * d, rows, parts, b0);
    V41_ERR();
}

extern "C" int memra_v41_hc_mixes_finish(const float* parts, const float* scale, const float* base, int rows, int hc,
                                         int d, int iters, float norm_eps, float hc_eps, float* mixes, float* pre,
                                         float* post, float* comb, void* stream) {
    int nmix = (2 + hc) * hc;
    if (hc <= 0 || hc > 4 || nmix > 31) return 41047;
    if (rows <= 0) return rows < 0 ? 41047 : 0;
    v41_hc_finish_kernel<<<(rows + 15) / 16, 256, 0, (cudaStream_t)stream>>>(parts, nmix, hc * d, norm_eps, scale,
                                                                             base, hc, iters, hc_eps, mixes, pre,
                                                                             post, comb, rows);
    V41_ERR();
}

// y[s][d] = bf16(sum_k mix[s][k] * x[s][k][d])
__global__ void v41_hc_pre_kernel(const v41_bf* __restrict__ x, const float* __restrict__ mix, int hc, int d,
                                  v41_bf* __restrict__ y, long ldy) {
    int s = blockIdx.x;
    const v41_bf* xr = x + (long)s * hc * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < hc; ++k) acc = __fadd_rn(acc, __fmul_rn(mix[(long)s * hc + k], v41_bf2f(xr[(long)k * d + i])));
        y[(long)s * ldy + i] = v41_f2bf(acc);
    }
}

// The same per-element program on 16-byte chunks (rows 16-byte aligned, hc <= 4): a thread owns 8
// neighbouring columns, loads them from every copy with one load each, all in flight, and writes
// them with one store.
__global__ void __launch_bounds__(256) v41_hc_pre8_kernel(const v41_bf* __restrict__ x, const float* __restrict__ mix,
                                                          int hc, int d, v41_bf* __restrict__ y, long ldy) {
    int s = blockIdx.x;
    int c = blockIdx.y * 256 + threadIdx.x;  // 8-column chunk
    if (c * 8 >= d) return;
    const v41_bf* xr = x + (long)s * hc * d;
    uint4 xv[4];
    float mv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        xv[k] = k < hc ? *(const uint4*)(xr + (long)k * d + c * 8) : make_uint4(0, 0, 0, 0);
        mv[k] = k < hc ? mix[(long)s * hc + k] : 0.f;
    }
    uint4 o4;
    v41_bf* oe = (v41_bf*)&o4;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            if (k < hc) acc = __fadd_rn(acc, __fmul_rn(mv[k], v41_bf2f(((const v41_bf*)&xv[k])[u])));
        oe[u] = v41_f2bf(acc);
    }
    *(uint4*)(y + (long)s * ldy + c * 8) = o4;
}

extern "C" int memra_v41_hc_pre(const void* x, const float* mix, int rows, int hc, int d, void* y, long ldy,
                                void* stream) {
    if (rows <= 0) return rows < 0 ? 41017 : 0;
    if (hc > 0 && hc <= 4 && d % 8 == 0 && ldy % 8 == 0 && ((uintptr_t)x | (uintptr_t)y) % 16 == 0) {
        dim3 g8(rows, (d / 8 + 255) / 256);
        v41_hc_pre8_kernel<<<g8, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)x, mix, hc, d, (v41_bf*)y, ldy);
        V41_ERR();
    }
    v41_hc_pre_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)x, mix, hc, d, (v41_bf*)y, ldy);
    V41_ERR();
}

// out[s][k][d] = bf16(post[s][k] * f[s][d] + sum_j comb[s][j][k] * res[s][j][d]); out may alias res
// only when hc*d fits one pass (each thread reads its column of every copy before writing).
__global__ void v41_hc_post_kernel(const v41_bf* __restrict__ f, long ldf, const v41_bf* res,
                                   const float* __restrict__ post, const float* __restrict__ comb, int hc, int d,
                                   v41_bf* out) {
    int s = blockIdx.x;
    const v41_bf* rr = res + (long)s * hc * d;
    v41_bf* orow = out + (long)s * hc * d;
    float pk[4], cm[4][4];
    for (int k = 0; k < hc; ++k) pk[k] = post[(long)s * hc + k];
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) cm[j][k] = comb[(long)s * hc * hc + j * hc + k];
    int i1 = min(d, (int)(blockIdx.y + 1) * 1024);
    for (int i = blockIdx.y * 1024 + threadIdx.x; i < i1; i += blockDim.x) {
        float fv = v41_bf2f(f[(long)s * ldf + i]);
        float rv[4];
        for (int j = 0; j < hc; ++j) rv[j] = v41_bf2f(rr[(long)j * d + i]);
        for (int k = 0; k < hc; ++k) {
            float sum = 0.f;
            for (int j = 0; j < hc; ++j) sum = __fadd_rn(sum, __fmul_rn(cm[j][k], rv[j]));
            orow[(long)k * d + i] = v41_f2bf(__fadd_rn(__fmul_rn(pk[k], fv), sum));
        }
    }
}

// The same per-element program on 16-byte chunks: a thread owns 8 neighbouring columns, loads them
// from f and from every copy of res with one load each (all in flight before the math), and writes
// each output copy with one store. Rows 16-byte aligned (d and ldf multiples of 8). out may alias res:
// a thread reads its columns of every copy before it writes them.
__global__ void __launch_bounds__(256) v41_hc_post8_kernel(const v41_bf* __restrict__ f, long ldf, const v41_bf* res,
                                                           const float* __restrict__ post,
                                                           const float* __restrict__ comb, int hc, int d,
                                                           v41_bf* out) {
    int s = blockIdx.x;
    int c = blockIdx.y * 256 + threadIdx.x;  // 8-column chunk
    if (c * 8 >= d) return;
    const v41_bf* rr = res + (long)s * hc * d;
    v41_bf* orow = out + (long)s * hc * d;
    uint4 fv4 = *(const uint4*)(f + (long)s * ldf + c * 8);
    uint4 rv4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) rv4[j] = j < hc ? *(const uint4*)(rr + (long)j * d + c * 8) : make_uint4(0, 0, 0, 0);
    float pk[4], cm[4][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) pk[k] = k < hc ? post[(long)s * hc + k] : 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) cm[j][k] = (j < hc && k < hc) ? comb[(long)s * hc * hc + j * hc + k] : 0.f;
    const v41_bf* fe = (const v41_bf*)&fv4;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (k >= hc) break;
        uint4 o4;
        v41_bf* oe = (v41_bf*)&o4;
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (j < hc) sum = __fadd_rn(sum, __fmul_rn(cm[j][k], v41_bf2f(((const v41_bf*)&rv4[j])[u])));
            oe[u] = v41_f2bf(__fadd_rn(__fmul_rn(pk[k], v41_bf2f(fe[u])), sum));
        }
        *(uint4*)(orow + (long)k * d + c * 8) = o4;
    }
}

extern "C" int memra_v41_hc_post(const void* f, long ldf, const void* res, const float* post, const float* comb,
                                 int rows, int hc, int d, void* out, void* stream) {
    if (hc <= 0 || hc > 4) return 41018;
    if (rows <= 0) return rows < 0 ? 41018 : 0;
    if (d % 8 == 0 && ldf % 8 == 0 && ((uintptr_t)f | (uintptr_t)res | (uintptr_t)out) % 16 == 0) {
        dim3 g8(rows, (d / 8 + 255) / 256);
        v41_hc_post8_kernel<<<g8, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)f, ldf, (const v41_bf*)res, post, comb,
                                                                  hc, d, (v41_bf*)out);
        V41_ERR();
    }
    dim3 grid(rows, (d + 1023) / 1024);
    v41_hc_post_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)f, ldf, (const v41_bf*)res, post, comb,
                                                               hc, d, (v41_bf*)out);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Engram. dequant: emb[s][c*hd + j] = bf16(e4m3(rows[s][c][j]) * e8m0(sc[s][c][j/32])) from the
// staged table rows. gate: per (token, copy k) over d,
//   rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps), w = q_w[k] * k_w[k]
//   dot = sum(h * w * key) * rstd * d^-0.5, gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
//   out = bf16(h + gate * value), key = kv[s][k*d:], value = kv[s][hc*d:].

__global__ void v41_engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ sc, int hd,
                                          long n, v41_bf* __restrict__ out) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long rowi = i / hd;
    int j = (int)(i % hd);
    float v = __fmul_rn(v41_e4m3f(rows[i]), v41_e8m0(sc[rowi * (hd / 32) + j / 32]));
    out[i] = v41_f2bf(v);
}

extern "C" int memra_v41_engram_dequant(const void* rows, const void* sc, long n_rows, int hd, void* out,
                                        void* stream) {
    if (hd % 32 || n_rows < 0) return 41019;
    long n = n_rows * hd;
    if (n == 0) return 0;
    v41_engram_dequant_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)rows, (const uint8_t*)sc, hd, n, (v41_bf*)out);
    V41_ERR();
}

// h and out carry no __restrict__: the layer writes the stream in place (out == h)
__global__ void v41_engram_gate_kernel(const v41_bf* h, const v41_bf* __restrict__ kv,
                                       const v41_bf* __restrict__ qw, const v41_bf* __restrict__ kw, int hc, int d,
                                       float eps, float inv_sqrt_d, v41_bf* out) {
    __shared__ float red[32];
    int s = blockIdx.x, k = blockIdx.y;
    const v41_bf* hr = h + ((long)s * hc + k) * d;
    const v41_bf* key = kv + (long)s * (hc + 1) * d + (long)k * d;
    const v41_bf* val = kv + (long)s * (hc + 1) * d + (long)hc * d;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float hv = v41_bf2f(hr[i]), kv_ = v41_bf2f(key[i]);
        float wv = __fmul_rn(v41_bf2f(qw[(long)k * d + i]), v41_bf2f(kw[(long)k * d + i]));
        sh = __fadd_rn(sh, __fmul_rn(hv, hv));
        sk = __fadd_rn(sk, __fmul_rn(kv_, kv_));
        sd = __fadd_rn(sd, __fmul_rn(__fmul_rn(hv, wv), kv_));
    }
    float th = v41_block_sum(sh, red);
    float tk = v41_block_sum(sk, red);
    float td = v41_block_sum(sd, red);
    float rstd = __fmul_rn(rsqrtf(__fadd_rn(th / (float)d, eps)), rsqrtf(__fadd_rn(tk / (float)d, eps)));
    float dot = __fmul_rn(__fmul_rn(td, rstd), inv_sqrt_d);
    float sg = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    float gate = 1.f / __fadd_rn(1.f, expf(-sg));
    v41_bf* orow = out + ((long)s * hc + k) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        orow[i] = v41_f2bf(__fadd_rn(v41_bf2f(hr[i]), __fmul_rn(gate, v41_bf2f(val[i]))));
}

extern "C" int memra_v41_engram_gate(const void* h, const void* kv, const void* qw, const void* kw, int rows, int hc,
                                     int d, float eps, float inv_sqrt_d, void* out, void* stream) {
    if (rows <= 0) return rows < 0 ? 41020 : 0;
    dim3 grid(rows, hc);
    v41_engram_gate_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)h, (const v41_bf*)kv,
                                                                   (const v41_bf*)qw, (const v41_bf*)kw, hc, d, eps,
                                                                   inv_sqrt_d, (v41_bf*)out);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// small glue

// dst[r][:] = src[rowidx[r]][:] (bf16 rows)
__global__ void v41_gather_rows_kernel(const v41_bf* __restrict__ src, long lds, const int* __restrict__ rowidx,
                                       v41_bf* __restrict__ dst, long ldd, int d) {
    int r = blockIdx.x;
    const v41_bf* s = src + (long)rowidx[r] * lds;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dst[(long)r * ldd + i] = s[i];
}

extern "C" int memra_v41_gather_rows(const void* src, long lds, const int* rowidx, void* dst, long ldd, int rows, int d,
                                     void* stream) {
    if (rows <= 0) return rows < 0 ? 41021 : 0;
    v41_gather_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)src, lds, rowidx, (v41_bf*)dst, ldd,
                                                                   d);
    V41_ERR();
}

// bf16 -> fp32 and fp32 -> bf16 elementwise
__global__ void v41_bf2f_kernel(const v41_bf* __restrict__ x, float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = v41_bf2f(x[i]);
}
__global__ void v41_f2bf_kernel(const float* __restrict__ x, v41_bf* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = v41_f2bf(x[i]);
}

extern "C" int memra_v41_cast(const void* x, void* y, long n, int to_f32, void* stream) {
    if (n <= 0) return n < 0 ? 41022 : 0;
    unsigned b = (unsigned)((n + 255) / 256);
    if (to_f32) v41_bf2f_kernel<<<b, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)x, (float*)y, n);
    else v41_f2bf_kernel<<<b, 256, 0, (cudaStream_t)stream>>>((const float*)x, (v41_bf*)y, n);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Instruction ceilings (measurement only, `dsv41-bench ceil`): `iters` rounds per warp on register
// operands with no memory traffic in the loop; every result escapes through `out`.
//   0: four independent e4m3 m16n8k32 MMAs accumulating (the tensor pipe's e4m3 rate)
//   1: four e4m3 MMAs from zero (one operand register varied per round), each scaled into its own
//      accumulator as the GEMMs do (acc + (d * sa) * sb): the reference per-32-block schedule
//   2: the same with one fused multiply-add per element (acc = fma(d, s, acc))
//   3: four independent bf16 m16n8k16 MMAs accumulating
//   4: eight independent fp32 fma chains per thread

__device__ __forceinline__ void v41_mma_e4m3_acc(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int KIND>
__global__ void v41_peak_kernel(int iters, float* __restrict__ out) {
    int lane = threadIdx.x & 31;
    uint32_t a[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = 0x38383838u ^ ((uint32_t)(threadIdx.x + i) & 0x07070707u);
    uint32_t b0 = 0x30303030u ^ (uint32_t)(lane & 7), b1 = 0x34343434u ^ (uint32_t)(lane & 3);
    float acc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
    float sa = 1.0f + (float)(lane & 1), sb = 0.5f;
    if (KIND == 0) {
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int j = 0; j < 4; ++j) v41_mma_e4m3_acc(acc[j], a, b0, b1);
    } else if (KIND == 1 || KIND == 2) {
        float s = __fmul_rn(sa, sb);
        for (int it = 0; it < iters; ++it) {
            float d[4][4];
            // operands that change every round: identical from-zero MMAs would be merged by ptxas
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t ai[4] = {a[0] ^ (uint32_t)((it + j) & 0x07070707), a[1], a[2], a[3]};
                v41_mma_e4m3(d[j], ai, b0, b1);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    if (KIND == 1) acc[j][e] = __fadd_rn(acc[j][e], __fmul_rn(__fmul_rn(d[j][e], sa), sb));
                    else acc[j][e] = __fmaf_rn(d[j][e], s, acc[j][e]);
                }
        }
    } else if (KIND == 3) {
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int j = 0; j < 4; ++j) v41_mma_bf16(acc[j], a, b0, b1);
    } else {
        float x[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = 1.0f + (float)j * 1e-3f + (float)lane * 1e-5f;
        float y = 0.999f, z = 1e-3f;
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] = __fmaf_rn(x[j], y, z);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j & 3][j >> 2] = x[j];
    }
    float t = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) t = __fadd_rn(t, acc[j][e]);
    out[(long)blockIdx.x * blockDim.x + threadIdx.x] = t;
}

extern "C" int memra_v41_peak(int kind, int blocks, int threads, int iters, float* out, void* stream) {
    if (blocks <= 0 || threads <= 0 || threads % 32 || threads > 1024 || iters <= 0) return 41024;
    cudaStream_t st = (cudaStream_t)stream;
    switch (kind) {
        case 0: v41_peak_kernel<0><<<blocks, threads, 0, st>>>(iters, out); break;
        case 1: v41_peak_kernel<1><<<blocks, threads, 0, st>>>(iters, out); break;
        case 2: v41_peak_kernel<2><<<blocks, threads, 0, st>>>(iters, out); break;
        case 3: v41_peak_kernel<3><<<blocks, threads, 0, st>>>(iters, out); break;
        case 4: v41_peak_kernel<4><<<blocks, threads, 0, st>>>(iters, out); break;
        default: return 41024;
    }
    V41_ERR();
}

// A gate's fault injection (`SIDE_DELAY_NS` in dsv41_gpu.rs): one thread holds the stream for `ns`
// nanoseconds of the global timer, so what the stream runs next starts that much later.
__global__ void v41_sleep_kernel(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

extern "C" int memra_v41_sleep(unsigned long long ns, void* stream) {
    if (ns == 0) return 0;
    v41_sleep_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(ns);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Small N-rank gathers inside kernels (the device-push transport). With peer access on the pools a
// kernel stores straight into another card's memory. A rank's gather kernel packs its region into
// its slot of every peer's inbox and then stores the gather's sequence number in the slot's word;
// then it spins on the words of its own inbox and unpacks each slot into the producer's region of
// its buffer. No event, no host round trip and no copy call sits between a rank's last kernel of a
// part and its first kernel of the next one.
// Order: every thread fences its data stores to the system before the block's barrier and the
// word is stored after it, so a consumer that sees the word reads the data (word_first reverses
// the two: the gate's control case, which must fail). A slot holds nothing its reader writes, so
// no cache line has a writer on each card, and it is rewritten two gathers later, when its reader
// is done (the caller's law: every rank pushes and pulls in every gather and the slots alternate).

#define V41_PUSH_PEERS 7
#define V41_PUSH_MAX (1 << 20)  // an inbox slot's payload; the caller pushes less (its measured limit)
#define V41_PULL_SPIN (1L << 31)  // polls before a peer is given up for dead (about a minute)

struct V41Push {
    uint8_t* d[V41_PUSH_PEERS];   // each peer's slot for this producer: the payload
    uint32_t* w[V41_PUSH_PEERS];  // and the word
};
struct V41Pull {
    const uint8_t* d[V41_PUSH_PEERS];   // each producer's slot in this rank's inbox: the payload
    const uint32_t* w[V41_PUSH_PEERS];  // and the word
    long off[V41_PUSH_PEERS];           // the producer's region of the rank buffer: byte offset,
    long pitch[V41_PUSH_PEERS];         // row pitch,
    int width[V41_PUSH_PEERS];          // row bytes,
    int rows[V41_PUSH_PEERS];           // rows
};

// rows x width bytes between a region (rows `pitch` apart) and its packed copy. PACK 1: region to
// packed, 0: packed to region. 32-bit words when everything is word aligned, bytes otherwise.
template <int PACK>
__device__ __forceinline__ void v41_blk_move(uint8_t* reg, long pitch, int width, int rows, uint8_t* packed) {
    int total = rows * width;
    bool flat = rows <= 1 || pitch == width;
    if (((((uintptr_t)reg) | ((uintptr_t)packed) | (uintptr_t)pitch | (uintptr_t)width) & 3) == 0) {
        int ww = width >> 2;
        for (int i = threadIdx.x; i < (total >> 2); i += blockDim.x) {
            uint32_t* a = flat ? (uint32_t*)reg + i : (uint32_t*)(reg + (long)(i / ww) * pitch) + (i % ww);
            uint32_t* b = (uint32_t*)packed + i;
            if (PACK) *b = *a;
            else *a = *b;
        }
    } else {
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            uint8_t* a = flat ? reg + i : reg + (long)(i / width) * pitch + (i % width);
            if (PACK) packed[i] = *a;
            else *a = packed[i];
        }
    }
}

// The gather's sequence number is the rank's own count of gathers (`*ctr`, in its device memory) plus one:
// every rank runs the same gathers in the same order, so the counts agree, and no launch carries a number
// the host computed (a captured launch replays unchanged). `alt` is the byte distance from a producer's
// slot of an even gather to its slot of an odd one.
// One launch a rank a gather: the producer's half (pack the region into every peer's slot, fence, then
// the number into the slots' words) and, behind a barrier, the consumer's half (spin on the words of the
// rank's own inbox, unpack each slot into its producer's region of the buffer, count the gather). The two
// halves were two launches issued back to back on the rank's stream; nothing ran between them, so the
// order of every store and load is the same and a launch's latency is paid once.
__global__ void v41_gather_kernel(const uint8_t* src, long pitch, int width, int rows, V41Push dst, V41Pull in,
                                  int npeer, uint32_t* ctr, long alt, uint8_t* buf, int* bad, int word_first) {
    __shared__ int go;
    uint32_t seq = *ctr + 1;
    long o = (seq & 1) ? alt : 0;
    if (word_first) {
        if (threadIdx.x == 0) {
            for (int p = 0; p < npeer; ++p) *(volatile uint32_t*)((uint8_t*)dst.w[p] + o) = seq;
            __threadfence_system();
            // the control holds its data back 50 us, so a consumer that saw the word reads the slot
            // before the data lands: the consumer's fence after its spin would otherwise close the
            // window the race needs, and the control would pass by timing
            unsigned long long t0, t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            do {
                __nanosleep(1000);
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            } while (t - t0 < 50000);
        }
        __syncthreads();
    }
    for (int p = 0; p < npeer; ++p) v41_blk_move<1>((uint8_t*)src, pitch, width, rows, dst.d[p] + o);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (!word_first) {
            for (int p = 0; p < npeer; ++p) *(volatile uint32_t*)((uint8_t*)dst.w[p] + o) = seq;
            __threadfence_system();
        }
        int ok = 0;
        for (long it = 0; it < V41_PULL_SPIN && !ok; ++it) {
            ok = 1;
            for (int p = 0; p < npeer; ++p)
                ok &= (*(volatile const uint32_t*)((const uint8_t*)in.w[p] + o) == seq) ? 1 : 0;
        }
        // the words seen, the payload stores they were ordered after are visible to this thread;
        // the fence orders this block's loads of them after the words (the barrier below hands
        // that to every thread)
        __threadfence_system();
        go = ok;
        if (!ok) atomicOr(bad, 0x10000);  // the rank's transport flag: sticky, the group is lost
    }
    __syncthreads();
    if (go)
        for (int p = 0; p < npeer; ++p)
            v41_blk_move<0>(buf + in.off[p], in.pitch[p], in.width[p], in.rows[p], (uint8_t*)in.d[p] + o);
    __syncthreads();
    if (threadIdx.x == 0) *ctr = seq;
}

static int v41_push_threads(long bytes) {
    long t = ((bytes + 3) / 4 + 31) / 32 * 32;
    return (int)(t < 32 ? 32 : t > 1024 ? 1024 : t);
}

// A small gather on one rank: `rows` rows of `width` bytes at `src` (`pitch` apart) into the `npeer` slots
// of `dst` (a host V41Push: the even gathers' slots; an odd gather's are `alt` bytes on) with the gather's
// number (`*ctr + 1`) in their words; then the wait for that number in the `npeer` slots of `in` (a host
// V41Pull), each unpacked into its region of `buf`, and the gather counted in `*ctr`. A peer that never
// arrives sets bit 16 of *bad and nothing is unpacked.
extern "C" int memra_v41_gather(const void* src, long pitch, int width, int rows, const void* dst, const void* in,
                                int npeer, unsigned* ctr, long alt, void* buf, int* bad, int word_first,
                                void* stream) {
    if (npeer < 0 || npeer > V41_PUSH_PEERS || width < 0 || rows < 0 || (long)rows * width > V41_PUSH_MAX)
        return 41040;
    if (rows > 1 && pitch < width) return 41040;
    if (!ctr || alt < 0) return 41040;
    V41Push d = *(const V41Push*)dst;
    V41Pull s = *(const V41Pull*)in;
    long most = (long)rows * width;
    for (int p = 0; p < npeer; ++p) {
        long b = (long)s.rows[p] * s.width[p];
        if (s.width[p] < 0 || s.rows[p] < 0 || b > V41_PUSH_MAX || (s.rows[p] > 1 && s.pitch[p] < s.width[p]))
            return 41041;
        most = b > most ? b : most;
    }
    v41_gather_kernel<<<1, v41_push_threads(most), 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, pitch, width, rows, d, s, npeer, (uint32_t*)ctr, alt, (uint8_t*)buf, bad, word_first);
    V41_ERR();
}

// A strided region to and from a contiguous block, for the N-rank gathers of strided regions: a peer copy
// of a strided region between two cards is issued a row at a time, of a contiguous block it is one
// transfer. Chunks of 16 bytes (T = uint4), 4 (uint32_t) or 1 (uint8_t), 8 a thread.
template <typename T>
__global__ void __launch_bounds__(256) v41_blk_pack_kernel(uint8_t* reg, long pitch, long cw, long rows,
                                                           uint8_t* packed, int unpack) {
    long i0 = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8, total = cw * rows;
    if (i0 >= total) return;
    long r = i0 / cw, c = i0 % cw;
    T v[8];
    T* at[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        at[u] = nullptr;
        if (i0 + u < total) {
            at[u] = (T*)(reg + r * pitch + c * sizeof(T));
            v[u] = unpack == 1 ? *(const T*)(packed + (i0 + u) * sizeof(T)) : *at[u];
            if (++c == cw) {
                c = 0;
                ++r;
            }
        }
    }
#pragma unroll
    for (int u = 0; u < 8; ++u)
        if (at[u]) {
            if (unpack == 1) *at[u] = v[u];
            else *(T*)(packed + (i0 + u) * sizeof(T)) = v[u];
        }
    // unpack 2: `packed` is a peer card's memory; the stores are fenced to the system before the kernel ends
    if (unpack == 2) __threadfence_system();
}

// rows x width bytes of `reg` (rows `pitch` apart) into `packed` (unpack 0; 2 when `packed` is on a peer
// card, the large gathers of the device-push transport) or back (unpack 1): 16-byte
// chunks when everything is a multiple of 16 bytes and 16-byte aligned, 4-byte ones for multiples of 4,
// bytes otherwise
extern "C" int memra_v41_blk_pack(void* reg, long pitch, long width, long rows, void* packed, int unpack,
                                  void* stream) {
    if (rows <= 0 || width <= 0) return (rows < 0 || width < 0) ? 41045 : 0;
    uintptr_t all = (uintptr_t)reg | (uintptr_t)packed | (uintptr_t)width | (uintptr_t)pitch;
    if (pitch < width) return 41045;
    int unit = all % 16 == 0 ? 16 : all % 4 == 0 ? 4 : 1;
    long blocks = (width / unit * rows + 8 * 256 - 1) / (8 * 256);
    if (blocks > 0x7fffffffL) return 41045;
    if (unit == 16)
        v41_blk_pack_kernel<uint4><<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
            (uint8_t*)reg, pitch, width / 16, rows, (uint8_t*)packed, unpack);
    else if (unit == 4)
        v41_blk_pack_kernel<uint32_t><<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
            (uint8_t*)reg, pitch, width / 4, rows, (uint8_t*)packed, unpack);
    else
        v41_blk_pack_kernel<uint8_t><<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
            (uint8_t*)reg, pitch, width, rows, (uint8_t*)packed, unpack);
    V41_ERR();
}

// Gate helpers for the small gathers: a region of 32-bit words set to a pattern of (seed, word), and the
// number of words of a region that differ from it added to *diff.
__device__ __forceinline__ uint32_t v41_pat(uint32_t seed, uint32_t i) {
    uint32_t x = (i + 1u) * 0x9E3779B1u ^ (seed + 0x7F4A7C15u) * 0x85EBCA6Bu;
    return x ^ (x >> 15);
}
__global__ void v41_pat_kernel(uint32_t* reg, long pitch_w, int width_w, int rows, uint32_t seed, int* diff) {
    int total = rows * width_w, bad = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        uint32_t* a = reg + (long)(i / width_w) * pitch_w + (i % width_w);
        if (diff) bad += *a != v41_pat(seed, i);
        else *a = v41_pat(seed, i);
    }
    if (diff && bad) atomicAdd(diff, bad);
}
extern "C" int memra_v41_pat(void* reg, long pitch_w, int width_w, int rows, unsigned seed, int* diff, void* stream) {
    if (width_w <= 0 || rows <= 0 || pitch_w < width_w) return 41042;
    v41_pat_kernel<<<1, v41_push_threads((long)rows * width_w * 4), 0, (cudaStream_t)stream>>>(
        (uint32_t*)reg, pitch_w, width_w, rows, seed, diff);
    V41_ERR();
}
