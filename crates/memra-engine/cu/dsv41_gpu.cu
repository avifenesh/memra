// DeepSeek-V4.1-Flash single-card kernels: the sibling of dsv4_gpu.cu for the V4.1 program.
//
// Nothing here is shared with the DSv4 (0731) route, and nothing in dsv4_gpu.cu changes for it.
// The numeric program is the vendor reference inference/model.py + kernel.py of
// deepseek-ai/DeepSeek-V4.1-Flash-1017 at rev 1bae0e10, with the torch stand-ins of its tilelang
// kernels as the operational oracle:
//   - activations stay bf16 between ops, exactly where the reference keeps them bf16;
//   - every FP8/FP4 linear quantizes its input to e4m3 with one power-of-two (ue8m0) scale per 32
//     values (act_quant(x, 32, "ue8m0")), and the GEMM sums each 32-wide K block in fp32 before
//     scaling it by the activation and weight scales into a separate fp32 accumulator;
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

__global__ void v41_act_quant_kernel(const v41_bf* __restrict__ x, long ldx, const int* __restrict__ rowidx,
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
// Each K block is one m16n8k32 e4m3 MMA from zero, scaled into a separate fp32 accumulator
// (the reference fp8_gemm/fp4_gemm schedule). OUTF 0 = bf16, 1 = fp32. Tile 64x128x32, 8 warps.

#define V41_GM 64
#define V41_GN 128
#define V41_GS 48  // smem bytes per 32-byte row: conflict-free fragment loads

// One 64x128 output tile at (m0, n0). The plain kernel and the grouped expert kernel both run this
// body, so a row's result is the same whichever launch computes it (rows never mix in a tile).
template <int WF, int OUTF>
__device__ __forceinline__ void v41_gemm_f8_tile(const uint8_t* __restrict__ A, long lda,
                                                 const uint8_t* __restrict__ As, long ldas,
                                                 const uint8_t* __restrict__ B, long ldb,
                                                 const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                 void* __restrict__ C, long ldc, int M, int N, int K, int m0,
                                                 int n0, uint8_t* sA, uint8_t* sB, float* sSa, float* sSb) {
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
        // A tile: 64 rows x 32 bytes, threads 0..127 one 16-byte half each; scales by 128..191
        if (tid < 128) {
            int r = tid >> 1, h = tid & 1;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (m0 + r < M) v = *(const uint4*)(A + (long)(m0 + r) * lda + kb * 32 + h * 16);
            *(uint4*)(&sA[r * V41_GS + h * 16]) = v;
        } else if (tid < 192) {
            int r = tid - 128;
            sSa[r] = (m0 + r < M) ? v41_e8m0(As[(long)(m0 + r) * ldas + kb]) : 0.f;
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
        if (WF == 0) {
            if (tid >= 192 && tid < 196) {
                int j = tid - 192;
                int nb = (n0 >> 5) + j;
                sSb[j] = (n0 + j * 32 < N) ? v41_e8m0(Bs[(long)nb * ldbs + kb]) : 0.f;
            }
        } else if (tid < 128) {
            float s = 0.f;
            if (n0 + tid < N) {
                if (WF == 1) s = v41_e8m0(Bs[(long)(n0 + tid) * ldbs + kb]);
                else s = __fmul_rn(v41_e4m3f(Bs[(long)(n0 + tid) * ldbs + 2 * kb]), bscale2);
            }
            sSb[tid] = s;
        }
        __syncthreads();
        uint32_t af[2][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            int r = wm * 32 + mi * 16;
            af[mi][0] = v41_lds32(&sA[(r + g) * V41_GS + t * 4]);
            af[mi][1] = v41_lds32(&sA[(r + g + 8) * V41_GS + t * 4]);
            af[mi][2] = v41_lds32(&sA[(r + g) * V41_GS + 16 + t * 4]);
            af[mi][3] = v41_lds32(&sA[(r + g + 8) * V41_GS + 16 + t * 4]);
        }
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int c = wn * 32 + ni * 8;
            uint32_t b0 = v41_lds32(&sB[(c + g) * V41_GS + t * 4]);
            uint32_t b1 = v41_lds32(&sB[(c + g) * V41_GS + 16 + t * 4]);
            float sb0, sb1;
            if (WF == 0) {
                sb0 = sb1 = sSb[wn];
            } else {
                sb0 = sSb[c + 2 * t];
                sb1 = sSb[c + 2 * t + 1];
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                float d[4];
                v41_mma_e4m3(d, af[mi], b0, b1);
                int r = wm * 32 + mi * 16;
                float sa0 = sSa[r + g], sa1 = sSa[r + g + 8];
                acc[mi][ni][0] = __fadd_rn(acc[mi][ni][0], __fmul_rn(__fmul_rn(d[0], sa0), sb0));
                acc[mi][ni][1] = __fadd_rn(acc[mi][ni][1], __fmul_rn(__fmul_rn(d[1], sa0), sb1));
                acc[mi][ni][2] = __fadd_rn(acc[mi][ni][2], __fmul_rn(__fmul_rn(d[2], sa1), sb0));
                acc[mi][ni][3] = __fadd_rn(acc[mi][ni][3], __fmul_rn(__fmul_rn(d[3], sa1), sb1));
            }
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

// The reference schedule: the tile body above as its own launch, one 32-wide K block per smem round
// with no pipelining. Not dispatched; `memra_v41_gemm_f8_ref` runs it so `dsv41-gate kernels` can
// prove the pipelined and small-M schedules write its bytes.
template <int WF, int OUTF>
__global__ void __launch_bounds__(256) v41_gemm_f8_kernel(const uint8_t* __restrict__ A, long lda,
                                                          const uint8_t* __restrict__ As, long ldas,
                                                          const uint8_t* __restrict__ B, long ldb,
                                                          const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                          void* __restrict__ C, long ldc, int M, int N, int K) {
    __shared__ __align__(16) uint8_t sA[V41_GM * V41_GS];
    __shared__ __align__(16) uint8_t sB[V41_GN * V41_GS];
    __shared__ float sSa[V41_GM];
    __shared__ float sSb[V41_GN];
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

// Pipelined block-scaled GEMM (more than V41_SM_MAX rows): the same per-element program as the tile
// kernel above (the m16n8k32 e4m3 MMA from zero per 32-wide K block on the same fragment lanes, then
// acc + (d * sa) * sb in ascending K order), on a BM x 128 tile whose K blocks stream through a
// two-stage cp.async ring: the next stage's copies are in flight while the current one runs its
// MMAs. A stage holds 3 K blocks of a 64-row tile (a block then stays under half an SM's 100 KB of
// shared memory, so two blocks, 16 warps, share an SM) and 4 of a 128-row tile (one block per SM):
// the per-stage work (copies, scales, barrier, the fp4 expansion) is spread over more MMAs.
// Fragments come from ldmatrix on rows padded by 16 bytes (conflict-free); MXFP4 / minted NVFP4
// weights land raw and are expanded to e4m3 once per stage by the whole block, exactly as the tile
// kernel's stager expands them. The scales of a stage are read with plain loads while the stage
// before it runs and stored as fp32 next to it. The A rows of a tile are the tile kernel's rows at
// the same fragment positions (BM = 64: warp rows 32; BM = 128: warp rows 64), so every output's
// bytes are the tile kernel's.

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
    static constexpr int PK = BM == 64 ? 3 : 4;   // 32-wide K blocks per stage
    static constexpr int PS = 2;                  // stages in flight
    static constexpr int PRS = PK * 32 + 16;      // smem bytes per e4m3 row of a stage (+16: conflict-free ldmatrix)
    static constexpr int PRR = PK * 16 + 16;      // smem bytes per raw fp4 row of a stage
    static constexpr int A_B = BM * PRS;                           // e4m3 A rows
    static constexpr int B_B = WF == 0 ? BN * PRS : BN * PRR;      // e4m3 or raw fp4 B rows
    static constexpr int SA_B = BM * PK * 4;                       // fp32 row scales
    static constexpr int SB_B = (WF == 0 ? BN / 32 : BN) * PK * 4;  // fp32 column (block) scales
    static constexpr int FL_B = 16;  // per warp: which of the stage's K blocks hold a scale code outside the fused range
    static constexpr int STAGE = A_B + B_B + SA_B + SB_B + FL_B;
    static constexpr int CONV = WF == 0 ? 0 : BN * PRS;  // the expanded e4m3 B of the current stage
    static constexpr int SMEM = PS * STAGE + CONV;
};

template <int WF, int OUTF, int BM>
__device__ __forceinline__ void v41_gemm_f8_pipe_tile(const uint8_t* __restrict__ A, long lda,
                                                      const uint8_t* __restrict__ As, long ldas,
                                                      const uint8_t* __restrict__ B, long ldb,
                                                      const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                      void* __restrict__ C, long ldc, int M, int N, int K, int m0,
                                                      int n0, uint8_t* sm) {
    using P = V41Pipe<WF, BM>;
    constexpr int BN = P::BN;
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
        for (int c = tid; c < BM * CR; c += 256) {
            int r = c / CR, q = c % CR;
            int kb = kb0 + (q >> 1);
            bool ok = (m0 + r < M) && (kb < KB);
            const uint8_t* src = ok ? A + (long)(m0 + r) * lda + kb0 * 32 + q * 16 : A;
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
    // ---- scales: thread r < BM holds row r's codes, threads 128.. the columns' (WF 0: 4 blocks)
    auto load_sc = [&](int i) -> uint32_t {
        int kb0 = i * P::PK;
        uint32_t code = 0;
        if (i >= NS) return code;
        if (tid < BM) {
            if (m0 + tid < M)
                for (int u = 0; u < P::PK; ++u)
                    if (kb0 + u < KB) code |= (uint32_t)As[(long)(m0 + tid) * ldas + kb0 + u] << (8 * u);
        } else if (tid >= 128) {
            int c = tid - 128;
            if (WF == 0) {
                if (c < BN / 32 && n0 + c * 32 < N)
                    for (int u = 0; u < P::PK; ++u)
                        if (kb0 + u < KB)
                            code |= (uint32_t)Bs[(long)((n0 >> 5) + c) * ldbs + kb0 + u] << (8 * u);
            } else if (n0 + c < N) {
                for (int u = 0; u < P::PK; ++u)
                    if (kb0 + u < KB)
                        code |= (uint32_t)Bs[(long)(n0 + c) * ldbs + (WF == 1 ? kb0 + u : 2 * (kb0 + u))] << (8 * u);
            }
        }
        return code;
    };
    // Fused epilogue, same bytes: with ue8m0 scales sa = 2^ea, sb = 2^eb and an MMA sum d that is 0 or
    // of magnitude in [2^-18, 2^23) (32 products of e4m3 values, each a multiple of 2^-18), d * sa and
    // d * sa * sb are exact whenever ea and ea + eb lie in [-108, 104], so acc + (d * sa) * sb rounds
    // once, at the add, exactly as fma(d * sa, sb, acc) and fma(d, sa * sb, acc) round. A K block of
    // the tile takes it when every live row code is in [64, 180] and every live column code in
    // [82, 178] (any pair then has ea in [-108, 104] and ea + eb in [-108, 104]); the thread that
    // stores a code checks it, and each warp leaves one byte per stage with a bit per K block that
    // holds a code outside. Scales of real tensors sit near 2^-10; a K block with an outlier runs the
    // three-step form. A padding row or column has scale 0 and no output.
    auto store_sc = [&](int i, uint32_t code) {
        if (i >= NS) return;
        uint8_t* st = stage(i);
        float* sSa = (float*)(st + P::A_B + P::B_B);
        float* sSb = (float*)(st + P::A_B + P::B_B + P::SA_B);
        uint8_t* sFl = st + P::A_B + P::B_B + P::SA_B + P::SB_B;
        uint32_t bad = 0;
        if (tid < BM) {
            bool ok = m0 + tid < M;
            for (int u = 0; u < P::PK; ++u) {
                uint32_t b = (code >> (8 * u)) & 0xFF;
                sSa[tid * P::PK + u] = ok ? v41_e8m0(b) : 0.f;
                if (ok && (b < 64 || b > 180)) bad |= 1u << u;
            }
        } else if (tid >= 128) {
            int c = tid - 128;
            if (WF == 0) {
                if (c < BN / 32) {
                    bool ok = n0 + c * 32 < N;
                    for (int u = 0; u < P::PK; ++u) {
                        uint32_t b = (code >> (8 * u)) & 0xFF;
                        sSb[c * P::PK + u] = ok ? v41_e8m0(b) : 0.f;
                        if (ok && (b < 82 || b > 178)) bad |= 1u << u;
                    }
                }
            } else {
                bool ok = n0 + c < N;
                for (int u = 0; u < P::PK; ++u) {
                    uint8_t b = (code >> (8 * u)) & 0xFF;
                    float v = WF == 1 ? v41_e8m0(b) : __fmul_rn(v41_e4m3f(b), bscale2);
                    sSb[c * P::PK + u] = ok ? v : 0.f;
                    if (ok && (b < 82 || b > 178)) bad |= 1u << u;
                }
            }
        }
        if (WF != 2) {
            uint32_t wb = 0;
#pragma unroll
            for (int u = 0; u < P::PK; ++u)
                if (__any_sync(0xffffffffu, (bad >> u) & 1)) wb |= 1u << u;
            if (lane == 0) sFl[warp] = (uint8_t)wb;
        }
    };

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
        const float* sSa = (const float*)(st + P::A_B + P::B_B);
        const float* sSb = (const float*)(st + P::A_B + P::B_B + P::SA_B);
        // the 8 warps' flags of this stage: bit u of a byte set = K block u holds a code outside
        uint64_t sfl = WF == 2 ? ~0ull : *(const uint64_t*)(st + P::A_B + P::B_B + P::SA_B + P::SB_B);
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
            // every MMA of this K block first (independent, so the tensor pipe stays fed), then the
            // scale epilogue of each
            float d[MI][4][4];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int mi = 0; mi < MI; ++mi) v41_mma_e4m3(d[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
            float sa[MI][2];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                int r = wm * (BM / 2) + mi * 16;
                sa[mi][0] = sSa[(r + g) * P::PK + kk];
                sa[mi][1] = sSa[(r + g + 8) * P::PK + kk];
            }
            float sb[4][2];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int c = wn * 32 + ni * 8;
                if (WF == 0) {
                    sb[ni][0] = sb[ni][1] = sSb[wn * P::PK + kk];
                } else {
                    sb[ni][0] = sSb[(c + 2 * t) * P::PK + kk];
                    sb[ni][1] = sSb[(c + 2 * t + 1) * P::PK + kk];
                }
            }
            bool fused = ((sfl >> kk) & 0x0101010101010101ull) == 0;  // see store_sc
            if (fused && WF == 0) {
#pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    float s0 = __fmul_rn(sa[mi][0], sb[0][0]), s1 = __fmul_rn(sa[mi][1], sb[0][0]);
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        acc[mi][ni][0] = __fmaf_rn(d[mi][ni][0], s0, acc[mi][ni][0]);
                        acc[mi][ni][1] = __fmaf_rn(d[mi][ni][1], s0, acc[mi][ni][1]);
                        acc[mi][ni][2] = __fmaf_rn(d[mi][ni][2], s1, acc[mi][ni][2]);
                        acc[mi][ni][3] = __fmaf_rn(d[mi][ni][3], s1, acc[mi][ni][3]);
                    }
                }
            } else if (fused) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int mi = 0; mi < MI; ++mi) {
                        float sa0 = sa[mi][0], sa1 = sa[mi][1];
                        acc[mi][ni][0] = __fmaf_rn(__fmul_rn(d[mi][ni][0], sa0), sb[ni][0], acc[mi][ni][0]);
                        acc[mi][ni][1] = __fmaf_rn(__fmul_rn(d[mi][ni][1], sa0), sb[ni][1], acc[mi][ni][1]);
                        acc[mi][ni][2] = __fmaf_rn(__fmul_rn(d[mi][ni][2], sa1), sb[ni][0], acc[mi][ni][2]);
                        acc[mi][ni][3] = __fmaf_rn(__fmul_rn(d[mi][ni][3], sa1), sb[ni][1], acc[mi][ni][3]);
                    }
            } else {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int mi = 0; mi < MI; ++mi) {
                        float sa0 = sa[mi][0], sa1 = sa[mi][1], sb0 = sb[ni][0], sb1 = sb[ni][1];
                        acc[mi][ni][0] = __fadd_rn(acc[mi][ni][0], __fmul_rn(__fmul_rn(d[mi][ni][0], sa0), sb0));
                        acc[mi][ni][1] = __fadd_rn(acc[mi][ni][1], __fmul_rn(__fmul_rn(d[mi][ni][1], sa0), sb1));
                        acc[mi][ni][2] = __fadd_rn(acc[mi][ni][2], __fmul_rn(__fmul_rn(d[mi][ni][2], sa1), sb0));
                        acc[mi][ni][3] = __fadd_rn(acc[mi][ni][3], __fmul_rn(__fmul_rn(d[mi][ni][3], sa1), sb1));
                    }
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
__global__ void __launch_bounds__(256, BM == 64 ? 2 : 1) v41_gemm_f8_pipe_kernel(const uint8_t* __restrict__ A, long lda,
                                                               const uint8_t* __restrict__ As, long ldas,
                                                               const uint8_t* __restrict__ B, long ldb,
                                                               const uint8_t* __restrict__ Bs, long ldbs,
                                                               float bscale2, void* __restrict__ C, long ldc, int M,
                                                               int N, int K) {
    extern __shared__ __align__(16) uint8_t v41_pipe_sm[];
    v41_gemm_f8_pipe_tile<WF, OUTF, BM>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, C, ldc, M, N, K, blockIdx.y * BM,
                                        blockIdx.x * 128, v41_pipe_sm);
}

// Small-M block-scaled GEMM (at most V41_SM_MAX rows: decode, and a grouped dispatch whose groups
// hold that few rows). Every output element is the tile kernel's sequence exactly: the same
// m16n8k32 e4m3 MMA from zero per 32-wide K block on the same fragment lanes and registers (rows
// 0..7 of the MMA tile, column = lane group), the same (d * sa) * sb, the same ascending fp32 sum
// over K blocks. Only the schedule differs: a block owns 8 output columns, its 8 warps split the K
// blocks, each warp writes its blocks' scaled terms to shared memory, and one thread per (row,
// column) adds them in K order. So a row's bytes do not depend on which of the two kernels ran,
// and a one-token decode streams the weight with every SM instead of N/128 tiles walking K.
#define V41_SM_MAX 8
// SU: the K blocks a warp loads ahead of their MMAs (registers). 8 when a warp's share of K is at
// most 8 blocks; 10 for a longer K, where it saves a memory round trip per warp (20 blocks: two
// rounds, not three) at the price of registers.

__device__ __forceinline__ uint32_t v41_ldg32(const uint8_t* p) { return __ldg((const unsigned int*)p); }
__device__ __forceinline__ uint32_t v41_ldg16(const uint8_t* p) { return (uint32_t)__ldg((const unsigned short*)p); }

template <int WF, int OUTF, int SU>
__device__ __forceinline__ void v41_gemv_f8_cols(const uint8_t* __restrict__ A, long lda,
                                                 const uint8_t* __restrict__ As, long ldas,
                                                 const uint8_t* __restrict__ B, long ldb,
                                                 const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                 void* __restrict__ C, long ldc, int M, int N, int K, int n0,
                                                 float* sT) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int KB = K / 32;
    int per = (KB + 7) / 8;
    int kb0 = warp * per, kb1 = min(KB, kb0 + per);
    int col = n0 + g;  // this lane's B column (fragment column g)
    bool colok = col < N;
    bool rowok = g < M;  // this lane's A row (fragment rows g; rows g + 8 never hold one)
    const uint8_t* arow = A + (long)(rowok ? g : 0) * lda;
    const uint8_t* brow = B + (long)(colok ? col : 0) * ldb;
    const uint8_t* srow = As + (long)(rowok ? g : 0) * ldas;
    // the scale columns of this lane's two output columns (n0 + 2t, n0 + 2t + 1)
    int c0 = n0 + 2 * t, c1 = c0 + 1;
    for (int k = kb0; k < kb1; k += SU) {
        // every load of the next SU K blocks first (fragments and scale bytes), then their MMAs
        uint32_t a0[SU], a2[SU], b0[SU], b1[SU];
        uint8_t sac[SU], sbc0[SU], sbc1[SU];
#pragma unroll
        for (int u = 0; u < SU; ++u) {
            int kb = k + u;
            a0[u] = a2[u] = b0[u] = b1[u] = 0;
            sac[u] = sbc0[u] = sbc1[u] = 0;
            if (kb < kb1) {
                if (rowok) {
                    a0[u] = v41_ldg32(arow + kb * 32 + t * 4);
                    a2[u] = v41_ldg32(arow + kb * 32 + 16 + t * 4);
                    sac[u] = srow[kb];
                    if (WF == 0) {
                        if (c0 < N) sbc0[u] = Bs[(long)(n0 >> 5) * ldbs + kb];
                    } else if (WF == 1) {
                        if (c0 < N) sbc0[u] = Bs[(long)c0 * ldbs + kb];
                        if (c1 < N) sbc1[u] = Bs[(long)c1 * ldbs + kb];
                    } else {
                        if (c0 < N) sbc0[u] = Bs[(long)c0 * ldbs + 2 * kb];
                        if (c1 < N) sbc1[u] = Bs[(long)c1 * ldbs + 2 * kb];
                    }
                }
                if (colok) {
                    if (WF == 0) {
                        b0[u] = v41_ldg32(brow + kb * 32 + t * 4);
                        b1[u] = v41_ldg32(brow + kb * 32 + 16 + t * 4);
                    } else {
                        b0[u] = v41_ldg16(brow + kb * 16 + 2 * t);
                        b1[u] = v41_ldg16(brow + kb * 16 + 8 + 2 * t);
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < SU; ++u) {
            int kb = k + u;
            if (kb >= kb1) break;
            uint32_t af[4] = {a0[u], 0u, a2[u], 0u};
            uint32_t bb0 = b0[u], bb1 = b1[u];
            if (WF != 0) {
                bb0 = v41_fp4_to_e4m3x4(bb0);
                bb1 = v41_fp4_to_e4m3x4(bb1);
            }
            float d[4];
            v41_mma_e4m3(d, af, bb0, bb1);
            if (rowok) {
                float sa = v41_e8m0(sac[u]);
                float sb0 = 0.f, sb1 = 0.f;
                if (WF == 0) {
                    float sv = (c0 < N) ? v41_e8m0(sbc0[u]) : 0.f;
                    sb0 = sb1 = sv;
                } else if (WF == 1) {
                    if (c0 < N) sb0 = v41_e8m0(sbc0[u]);
                    if (c1 < N) sb1 = v41_e8m0(sbc1[u]);
                } else {
                    if (c0 < N) sb0 = __fmul_rn(v41_e4m3f(sbc0[u]), bscale2);
                    if (c1 < N) sb1 = __fmul_rn(v41_e4m3f(sbc1[u]), bscale2);
                }
                float* tp = sT + ((long)kb * M + g) * 8 + 2 * t;
                tp[0] = __fmul_rn(__fmul_rn(d[0], sa), sb0);
                tp[1] = __fmul_rn(__fmul_rn(d[1], sa), sb1);
            }
        }
    }
    __syncthreads();
    if (tid < M * 8) {
        int r = tid >> 3, cc = tid & 7;
        float acc = 0.f;
        int kb = 0;
        for (; kb + 8 <= KB; kb += 8) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = sT[((long)(kb + u) * M + r) * 8 + cc];
#pragma unroll
            for (int u = 0; u < 8; ++u) acc = __fadd_rn(acc, v[u]);
        }
        for (; kb < KB; ++kb) acc = __fadd_rn(acc, sT[((long)kb * M + r) * 8 + cc]);
        int cn = n0 + cc;
        if (cn < N) {
            if (OUTF == 0) ((v41_bf*)C)[(long)r * ldc + cn] = v41_f2bf(acc);
            else ((float*)C)[(long)r * ldc + cn] = acc;
        }
    }
}

template <int WF, int OUTF, int SU>
__global__ void __launch_bounds__(256) v41_gemv_f8_kernel(const uint8_t* __restrict__ A, long lda,
                                                          const uint8_t* __restrict__ As, long ldas,
                                                          const uint8_t* __restrict__ B, long ldb,
                                                          const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                          void* __restrict__ C, long ldc, int M, int N, int K) {
    extern __shared__ float v41_sm_t[];
    v41_gemv_f8_cols<WF, OUTF, SU>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, C, ldc, M, N, K, blockIdx.x * 8,
                               v41_sm_t);
}

// The small-M body with the MMA's operands exchanged. The tile is 16 x 8: above, its 8-wide side holds
// 8 weight rows and its 16-wide side the activation rows, of which a call has at most 8, so half of
// every MMA is zeros. Here the 16-wide side holds 16 weight rows (output columns n0 .. n0 + 15) and
// the 8-wide side the activation rows: one MMA per 32-wide K block yields 16 x M terms instead of
// 8 x M. Per output element the program is the same: the e4m3 MMA from
// zero over the same 32 products in the same K order, (d * sa) * sb, the ascending fp32 sum over K
// blocks. The tensor core adding the 32 products the same way whichever operand a factor is in is
// what `dsv41-gate kernels` proves against the reference schedule. It runs every small-M tile list
// (the routed experts of a call of up to V41_SM_MAX rows): there the kernel is bound by the
// instructions it issues per K block and weight row, and this form issues fewer (163 -> 128 us for
// a token's six experts). The dense FP8 GEMMs keep the 8-column body: a single wave of blocks is
// bound by its load rounds, and 16 rows of fragments leave registers for half as many K blocks a
// round (wo_b 44 -> 68 us).
template <int WF, int OUTF, int SU, int ONE>
__device__ __forceinline__ void v41_gemv_f8_cols16(const uint8_t* __restrict__ A, long lda,
                                                   const uint8_t* __restrict__ As, long ldas,
                                                   const uint8_t* __restrict__ B, long ldb,
                                                   const uint8_t* __restrict__ Bs, long ldbs, float bscale2,
                                                   void* __restrict__ C, long ldc, int M, int N, int K, int n0,
                                                   int P, float* sT) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    int KB = K / 32;
    // the K blocks run in P parts so that a part's terms fit the block's shared memory; the sum
    // below carries from part to part, so it is the one ascending sum over all of K (ONE: the
    // instance for P = 1, a decode step's rows, without the loop over parts)
    int KBp = ONE ? KB : (KB + P - 1) / P;
    // this lane's two weight rows (fragment rows g and g + 8) and its activation row (fragment column g)
    int c0 = n0 + g, c1 = c0 + 8;
    bool ok0 = c0 < N, ok1 = c1 < N;
    bool rowok = g < M;
    const uint8_t* w0 = B + (long)(ok0 ? c0 : 0) * ldb;
    const uint8_t* w1 = B + (long)(ok1 ? c1 : 0) * ldb;
    const uint8_t* arow = A + (long)(rowok ? g : 0) * lda;
    // this lane's terms: activation rows 2t and 2t + 1 against its two weight rows
    int r0 = 2 * t, r1 = r0 + 1;
    bool out0 = r0 < M, out1 = r1 < M;
    float acc = 0.f;  // the (row, column) sum of the threads below M * 16
    for (int pa = 0; pa < KB; pa += KBp) {
    int pb = min(KB, pa + KBp);
    int per = (pb - pa + 7) / 8;
    int kb0 = pa + warp * per, kb1 = min(pb, kb0 + per);
    for (int k = kb0; k < kb1; k += SU) {
        uint32_t wl0[SU], wh0[SU], wl1[SU], wh1[SU], b0[SU], b1[SU];
        uint8_t sa0[SU], sa1[SU], sb0[SU], sb1[SU];
#pragma unroll
        for (int u = 0; u < SU; ++u) {
            int kb = k + u;
            wl0[u] = wh0[u] = wl1[u] = wh1[u] = b0[u] = b1[u] = 0;
            sa0[u] = sa1[u] = sb0[u] = sb1[u] = 0;
            if (kb < kb1) {
                if (rowok) {
                    b0[u] = v41_ldg32(arow + kb * 32 + t * 4);
                    b1[u] = v41_ldg32(arow + kb * 32 + 16 + t * 4);
                }
                if (out0) {
                    sa0[u] = As[(long)r0 * ldas + kb];
                    if (out1) sa1[u] = As[(long)r1 * ldas + kb];
                    if (WF == 0) {
                        if (ok0) sb0[u] = Bs[(long)(n0 >> 5) * ldbs + kb];
                    } else if (WF == 1) {
                        if (ok0) sb0[u] = Bs[(long)c0 * ldbs + kb];
                        if (ok1) sb1[u] = Bs[(long)c1 * ldbs + kb];
                    } else {
                        if (ok0) sb0[u] = Bs[(long)c0 * ldbs + 2 * kb];
                        if (ok1) sb1[u] = Bs[(long)c1 * ldbs + 2 * kb];
                    }
                }
                if (WF == 0) {
                    if (ok0) {
                        wl0[u] = v41_ldg32(w0 + kb * 32 + t * 4);
                        wh0[u] = v41_ldg32(w0 + kb * 32 + 16 + t * 4);
                    }
                    if (ok1) {
                        wl1[u] = v41_ldg32(w1 + kb * 32 + t * 4);
                        wh1[u] = v41_ldg32(w1 + kb * 32 + 16 + t * 4);
                    }
                } else {
                    if (ok0) {
                        wl0[u] = v41_ldg16(w0 + kb * 16 + 2 * t);
                        wh0[u] = v41_ldg16(w0 + kb * 16 + 8 + 2 * t);
                    }
                    if (ok1) {
                        wl1[u] = v41_ldg16(w1 + kb * 16 + 2 * t);
                        wh1[u] = v41_ldg16(w1 + kb * 16 + 8 + 2 * t);
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < SU; ++u) {
            int kb = k + u;
            if (kb >= kb1) break;
            uint32_t af[4] = {wl0[u], wl1[u], wh0[u], wh1[u]};
            if (WF != 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) af[i] = v41_fp4_to_e4m3x4(af[i]);
            }
            float d[4];
            v41_mma_e4m3(d, af, b0[u], b1[u]);
            if (out0) {
                float s0 = v41_e8m0(sa0[u]), s1 = out1 ? v41_e8m0(sa1[u]) : 0.f;
                float q0 = 0.f, q1 = 0.f;
                if (WF == 0) {
                    q0 = q1 = ok0 ? v41_e8m0(sb0[u]) : 0.f;
                } else if (WF == 1) {
                    if (ok0) q0 = v41_e8m0(sb0[u]);
                    if (ok1) q1 = v41_e8m0(sb1[u]);
                } else {
                    if (ok0) q0 = __fmul_rn(v41_e4m3f(sb0[u]), bscale2);
                    if (ok1) q1 = __fmul_rn(v41_e4m3f(sb1[u]), bscale2);
                }
                // d[0], d[1]: weight row c0 against activation rows r0, r1; d[2], d[3]: weight row c1
                float* tp = sT + ((long)(kb - pa) * M + r0) * 16 + g;
                tp[0] = __fmul_rn(__fmul_rn(d[0], s0), q0);
                tp[8] = __fmul_rn(__fmul_rn(d[2], s0), q1);
                if (out1) {
                    tp[16] = __fmul_rn(__fmul_rn(d[1], s1), q0);
                    tp[24] = __fmul_rn(__fmul_rn(d[3], s1), q1);
                }
            }
        }
    }
    __syncthreads();
    if (tid < M * 16) {
        int r = tid >> 4, cc = tid & 15;
        int kb = 0, nk = pb - pa;
        for (; kb + 8 <= nk; kb += 8) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = sT[((long)(kb + u) * M + r) * 16 + cc];
#pragma unroll
            for (int u = 0; u < 8; ++u) acc = __fadd_rn(acc, v[u]);
        }
        for (; kb < nk; ++kb) acc = __fadd_rn(acc, sT[((long)kb * M + r) * 16 + cc]);
    }
    // the next part's terms overwrite these
    if (pb < KB) __syncthreads();
    }
    if (tid < M * 16) {
        int r = tid >> 4, cc = tid & 15;
        int cn = n0 + cc;
        if (cn < N) {
            if (OUTF == 0) ((v41_bf*)C)[(long)r * ldc + cn] = v41_f2bf(acc);
            else ((float*)C)[(long)r * ldc + cn] = acc;
        }
    }
}

// the 16-column body's terms, [K blocks of a part][rows][16] fp32: the fewest parts of K that keep
// them within the default 48 KB of a block (two blocks an SM)
static void v41_gemv_smem16(int rows, int k, int* bytes, int* parts) {
    int kb = k / 32, p = 1;
    while ((long)((kb + p - 1) / p) * rows * 16 * 4 > 48 * 1024) ++p;
    *parts = p;
    *bytes = (int)((long)((kb + p - 1) / p) * rows * 16 * 4);
}

static int v41_gemv_smem(int rows, int k, int* bytes) {
    long b = (long)(k / 32) * rows * 8 * 4;
    if (b > 96 * 1024) return 0;
    *bytes = (int)b;
    return 1;
}

// The grouped expert GEMMs over a device-built tile list (`memra_v41_moe_plan`): block y runs tile
// y (none past `*ntiles`), its expert's projection `proj` from the weight table. The pipelined tile
// body for 64- or 128-row tiles, the small-M body for tiles of at most V41_SM_MAX rows; the same
// per-element program as every other dispatch of these GEMMs.
template <int WF, int OUTF, int BM>
__global__ void __launch_bounds__(256, BM == 64 ? 2 : 1) v41_gemm_f8_pipe_tiles_kernel(const uint8_t* __restrict__ A, long lda,
                                                                     const uint8_t* __restrict__ As, long ldas,
                                                                     const V41Tile* __restrict__ tiles,
                                                                     const int* __restrict__ ntiles,
                                                                     const V41ExpW* __restrict__ ew, int proj,
                                                                     long ldb, long ldbs, void* __restrict__ C,
                                                                     long ldc, int N, int K) {
    extern __shared__ __align__(16) uint8_t v41_pipe_sm[];
    // the tile is loaded with the count, not after it: one memory round trip, not two (an entry past
    // the count is inside the list's allocation and is not used)
    const V41Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const V41ExpW& w = ew[tl.expert];
    void* c = OUTF ? (void*)((float*)C + (long)tl.row0 * ldc) : (void*)((v41_bf*)C + (long)tl.row0 * ldc);
    v41_gemm_f8_pipe_tile<WF, OUTF, BM>(A + (long)tl.row0 * lda, lda, As + (long)tl.row0 * ldas, ldas, w.w[proj], ldb,
                                        w.s[proj], ldbs, w.scale2[proj], c, ldc, tl.rows, N, K, tl.m0,
                                        blockIdx.x * 128, v41_pipe_sm);
}

template <int WF, int OUTF, int SU, int ONE>
__global__ void __launch_bounds__(256) v41_gemv_f8t_tiles_kernel(const uint8_t* __restrict__ A, long lda,
                                                                 const uint8_t* __restrict__ As, long ldas,
                                                                 const V41Tile* __restrict__ tiles,
                                                                 const int* __restrict__ ntiles,
                                                                 const V41ExpW* __restrict__ ew, int proj, long ldb,
                                                                 long ldbs, void* __restrict__ C, long ldc, int N,
                                                                 int K, int P) {
    extern __shared__ float v41_sm_t[];
    // the tile is loaded with the count, not after it: one memory round trip, not two (an entry past
    // the count is inside the list's allocation and is not used)
    const V41Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const V41ExpW& w = ew[tl.expert];
    void* c = OUTF ? (void*)((float*)C + (long)tl.row0 * ldc) : (void*)((v41_bf*)C + (long)tl.row0 * ldc);
    v41_gemv_f8_cols16<WF, OUTF, SU, ONE>(A + (long)tl.row0 * lda, lda, As + (long)tl.row0 * ldas, ldas, w.w[proj], ldb,
                                     w.s[proj], ldbs, w.scale2[proj], c, ldc, tl.rows, N, K, blockIdx.x * 16, P,
                                     v41_sm_t);
}

// bm: the plan's tile rows (64 or 128: the pipelined body; <= V41_SM_MAX: every tile has at most that
// many rows and the small-M body runs); max_tiles bounds *ntiles.
extern "C" int memra_v41_gemm_f8_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                                       const int* ntiles, int max_tiles, const void* ew, int proj, long ldb, long ldbs,
                                       int wfmt, void* c, long ldc, int out_f32, int n, int k, int bm, void* stream) {
    if (k % 32 || n <= 0 || lda % 16 || max_tiles < 0 || max_tiles > 65535 || proj < 0 || proj > 2) return 41026;
    if (ldb % 16) return 41026;
    if (max_tiles == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as;
    const V41Tile* T = (const V41Tile*)tiles;
    const V41ExpW* E = (const V41ExpW*)ew;
    if (bm <= V41_SM_MAX) {
        // every small-M tile list: the 16-column body (the 8-column body stays with the dense GEMMs)
        int smem16 = 0, parts = 1;
        v41_gemv_smem16(bm, k, &smem16, &parts);
        dim3 grid((n + 15) / 16, max_tiles);
#define V41_GEMV_TT_P(WF, OF, ONE)                                                                                  \
    v41_gemv_f8t_tiles_kernel<WF, OF, 5, ONE><<<grid, 256, smem16, st>>>(A, lda, As, ldas, T, ntiles, E, proj, ldb, \
                                                                          ldbs, c, ldc, n, k, parts)
#define V41_GEMV_TT(WF, OF) \
    do { if (parts == 1) V41_GEMV_TT_P(WF, OF, 1); else V41_GEMV_TT_P(WF, OF, 0); } while (0)
        if (wfmt == 0) { if (out_f32) V41_GEMV_TT(0, 1); else V41_GEMV_TT(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_GEMV_TT(1, 1); else V41_GEMV_TT(1, 0); }
        else if (wfmt == 2) { if (out_f32) V41_GEMV_TT(2, 1); else V41_GEMV_TT(2, 0); }
        else return 41026;
#undef V41_GEMV_TT
#undef V41_GEMV_TT_P
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
                                                                          ldbs, c, ldc, n, k);                 \
    } while (0)
#define V41_PIPE_TL_BM(WF, OF) \
    do { if (bm == 128) V41_PIPE_TL(WF, OF, 128); else V41_PIPE_TL(WF, OF, 64); } while (0)
    if (wfmt == 0) { if (out_f32) V41_PIPE_TL_BM(0, 1); else V41_PIPE_TL_BM(0, 0); }
    else if (wfmt == 1) { if (out_f32) V41_PIPE_TL_BM(1, 1); else V41_PIPE_TL_BM(1, 0); }
    else if (wfmt == 2) { if (out_f32) V41_PIPE_TL_BM(2, 1); else V41_PIPE_TL_BM(2, 0); }
    else return 41026;
#undef V41_PIPE_TL_BM
#undef V41_PIPE_TL
    V41_ERR();
}

extern "C" int memra_v41_gemm_f8(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                 const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                 int m, int n, int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || lda % 16 || ldb % 16) return 41005;
    if (m == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as, *B = (const uint8_t*)b, *Bs = (const uint8_t*)bs;
    int smem = 0;
    if (m <= V41_SM_MAX && v41_gemv_smem(m, k, &smem)) {
        dim3 grid((n + 7) / 8);
#define V41_GEMV_F8_U(WF, OF, SUX)                                                                              \
    do {                                                                                                        \
        if (smem > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_f8_kernel<WF, OF, SUX>), 96 * 1024);                     \
        v41_gemv_f8_kernel<WF, OF, SUX><<<grid, 256, smem, st>>>(A, lda, As, ldas, B, ldb, Bs, ldbs, bscale2, c, \
                                                                  ldc, m, n, k);                                \
    } while (0)
#define V41_GEMV_F8(WF, OF) \
    do { if (k / 32 > 64) V41_GEMV_F8_U(WF, OF, 10); else V41_GEMV_F8_U(WF, OF, 8); } while (0)
        if (wfmt == 0) { if (out_f32) V41_GEMV_F8(0, 1); else V41_GEMV_F8(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_GEMV_F8(1, 1); else V41_GEMV_F8(1, 0); }
        else if (wfmt == 2) { if (out_f32) V41_GEMV_F8(2, 1); else V41_GEMV_F8(2, 0); }
        else return 41005;
#undef V41_GEMV_F8
#undef V41_GEMV_F8_U
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
#define V41_PIPE_F8_BM(WF, OF) \
    do { if (m >= 128) V41_PIPE_F8(WF, OF, 128); else V41_PIPE_F8(WF, OF, 64); } while (0)
        if (wfmt == 0) { if (out_f32) V41_PIPE_F8_BM(0, 1); else V41_PIPE_F8_BM(0, 0); }
        else if (wfmt == 1) { if (out_f32) V41_PIPE_F8_BM(1, 1); else V41_PIPE_F8_BM(1, 0); }
        else if (wfmt == 2) { if (out_f32) V41_PIPE_F8_BM(2, 1); else V41_PIPE_F8_BM(2, 0); }
        else return 41005;
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
// BF16 GEMM with fp32 accumulation (F.linear on bf16 weights, and on bf16 values promoted to
// fp32: bf16 x bf16 products are exact in fp32). C[z][M,N] = A[z][M,K] . B[z][N,K]^T, batched
// over grid.z with element strides. OUTF 0 = bf16, 1 = fp32. Tile 64x128x32, 8 warps.

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
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 32) {
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

// Small-M bf16 GEMM (at most V41_SM_MAX rows): the tile kernel's chain exactly. Its fp32 sum runs
// inside each m16n8k16 MMA over the K steps in order, so one warp owns an 8-column group over the
// whole K, on the same fragment lanes (rows 0..7 of the tile, column = lane group). The columns and
// the batch give the parallelism. A warp lands its 8 weight rows and the M rows in shared memory in
// 16-byte chunks through cp.async, `su` k16 steps per stage, and reads the fragments from there.
// A few-column output (few warps in the whole launch) is bound by the latency of one fetch after
// another, so there a warp is its own block and lands its whole K slice at once (one stage); a
// wide output streams through a two-stage ring of 8 steps, 4 warps per block.
#define V41_SMB_U 8

static long v41_gemv_bf16_warp_smem(int m, int su, int stages) {
    return (long)stages * (m + 8) * (su * 32 + 16);
}

template <int OUTF>
__global__ void __launch_bounds__(128) v41_gemv_bf16_kernel(const v41_bf* __restrict__ A, long lda, long sAz,
                                                            const v41_bf* __restrict__ B, long ldb, long sBz,
                                                            void* __restrict__ C, long ldc, long sCz, int M, int N,
                                                            int K, int su, int stages) {
    extern __shared__ __align__(16) uint8_t v41_smb[];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 8;
    if (n0 >= N) return;  // the whole warp
    const uint8_t* Ab = (const uint8_t*)(A + (long)blockIdx.z * sAz);
    const uint8_t* Bb = (const uint8_t*)(B + (long)blockIdx.z * sBz);
    bool rowok = g < M;
    int P = su * 32 + 16;        // bytes per staged row (+16: bank spread)
    int SB = (M + 8) * P;        // one stage: the M rows, then the 8 weight rows
    uint8_t* wsm = v41_smb + (long)warp * stages * SB;
    int CH = 2 * su;  // 16-byte chunks (8 elements) per row and stage
    auto issue = [&](int i) {
        uint8_t* st = wsm + (i % stages) * SB;
        long k0 = (long)i * su * 16;
        for (int q = lane; q < 8 * CH; q += 32) {
            int r = q / CH, c = q % CH;
            bool ok = (n0 + r < N) && (k0 + c * 8 < K);
            const uint8_t* src = ok ? Bb + ((long)(n0 + r) * ldb + k0 + c * 8) * 2 : Bb;
            v41_cp16(st + (M + r) * P + c * 16, src, ok);
        }
        for (int q = lane; q < M * CH; q += 32) {
            int r = q / CH, c = q % CH;
            bool ok = k0 + c * 8 < K;
            const uint8_t* src = ok ? Ab + ((long)r * lda + k0 + c * 8) * 2 : Ab;
            v41_cp16(st + r * P + c * 16, src, ok);
        }
    };
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int ns = (K + su * 16 - 1) / (su * 16);
    issue(0);
    v41_cp_commit();
    for (int i = 0; i < ns; ++i) {
        if (i + 1 < ns) issue(i + 1);  // ns > 1 only with a two-stage ring
        v41_cp_commit();
        v41_cp_wait<1>();
        __syncwarp();  // stage i landed for every lane
        const uint8_t* st = wsm + (i % stages) * SB;
        const uint8_t* sA = st + (rowok ? g : 0) * P;
        const uint8_t* sB = st + (M + g) * P;
        int k0 = i * su * 16;
        for (int u0 = 0; u0 < su; u0 += 8) {
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                int u = u0 + v;
                if (u >= su || k0 + 16 * u >= K) break;
                uint32_t af[4] = {0u, 0u, 0u, 0u};
                if (rowok) {
                    af[0] = v41_lds32(sA + u * 32 + t * 4);
                    af[2] = v41_lds32(sA + u * 32 + 16 + t * 4);
                }
                uint32_t b0 = v41_lds32(sB + u * 32 + t * 4);
                uint32_t b1 = v41_lds32(sB + u * 32 + 16 + t * 4);
                v41_mma_bf16(acc, af, b0, b1);
            }
        }
    }
    v41_cp_wait<0>();
    if (rowok) {
        char* Cz = (char*)C + (long)blockIdx.z * sCz * (OUTF == 0 ? 2 : 4);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            int cn = n0 + 2 * t + e;
            if (cn < N) {
                if (OUTF == 0) ((v41_bf*)Cz)[(long)g * ldc + cn] = v41_f2bf(acc[e]);
                else ((float*)Cz)[(long)g * ldc + cn] = acc[e];
            }
        }
    }
}

extern "C" int memra_v41_gemm_bf16(const void* a, long lda, long sAz, const void* b, long ldb, long sBz, void* c,
                                   long ldc, long sCz, int out_f32, int m, int n, int k, int batch, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || batch <= 0 || lda % 8 || ldb % 8 || sAz % 8 || sBz % 8) return 41006;
    if (m == 0) return 0;
    cudaStream_t st = (cudaStream_t)stream;
    if (m <= V41_SM_MAX) {
        // one warp per block with its whole K slice staged while the launch is at most 256 warps
        // and the slice fits a block; else 4 warps per block over the two-stage ring
        long whole = v41_gemv_bf16_warp_smem(m, k / 16, 1);
        long warps = (long)((n + 7) / 8) * batch;
        int su = V41_SMB_U, stages = 2, wpb = 4;
        if (warps <= 256 && whole <= 96 * 1024) {
            su = k / 16;
            stages = 1;
            wpb = 1;
        }
        int smem = (int)(wpb * v41_gemv_bf16_warp_smem(m, su, stages));
        dim3 grid((n + 8 * wpb - 1) / (8 * wpb), 1, batch);
#define V41_GEMV_BF(OF)                                                                                            \
    do {                                                                                                           \
        if (smem > 48 * 1024) V41_SMEM_LIMIT((v41_gemv_bf16_kernel<OF>), 96 * 1024);                               \
        v41_gemv_bf16_kernel<OF><<<grid, 32 * wpb, smem, st>>>((const v41_bf*)a, lda, sAz, (const v41_bf*)b, ldb, \
                                                                sBz, c, ldc, sCz, m, n, k, su, stages);            \
    } while (0)
        if (out_f32) V41_GEMV_BF(1); else V41_GEMV_BF(0);
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
    __shared__ float sB[V41_SMF_KC][33];
    __shared__ float sA[MR][V41_SMF_KC];
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
#pragma unroll
                for (int u = 0; u < 8; ++u) sB[q * 64 + i * 8 + u][j] = v41_bf2f(e[u]);
            }
        }
        {
            const v41_bf* e = (const v41_bf*)&na;
#pragma unroll
            for (int u = 0; u < 8; ++u) sA[ar][ac + u] = v41_bf2f(e[u]);
        }
        __syncthreads();
        if (k0 + V41_SMF_KC < K) fetch(k0 + V41_SMF_KC);
        int kn = min(V41_SMF_KC, K - k0);
        if (r < M)
            for (int kk = 0; kk < kn; kk += 8) {
                float av[8], bv[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    av[u] = sA[r][kk + u];
                    bv[u] = sB[kk + u][cc];
                }
#pragma unroll
                for (int u = 0; u < 8; ++u) acc = __fmaf_rn(av[u], bv[u], acc);
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

extern "C" int memra_v41_gemm_ffma(const void* a, long lda, const void* b, long ldb, float* c, long ldc, int m, int n,
                                   int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0) return 41023;
    if (m == 0) return 0;
    if (m <= V41_SM_MAX && lda % 8 == 0 && ldb % 8 == 0) {
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
    if (lda % 8 == 0 && ldb % 8 == 0 && ((uintptr_t)a | (uintptr_t)b) % 16 == 0)
        v41_gemm_ffma_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b, ldb,
                                                                           c, ldc, m, n, k);
    else
        v41_gemm_ffma_kernel<false><<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b,
                                                                            ldb, c, ldc, m, n, k);
    V41_ERR();
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
    if (hd != V41_AD || heads % V41_AH || m < 0 || topk < 0 || ldq % 8 || ldo % 8) return 41007;
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
    if (hd != V41_AD || heads % V41_AH || m < 0 || topk < 0 || ldq % 8 || ldo % 8 || !keys) return 41007;
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
    if (k > out_w || T < 0) return 41010;
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
// visible position pinned to +inf, top-k blocks kept when finite. keep [rows][nb] u8.
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
    if (cb <= 0 || ratio <= 0) return 41011;
    int tb = v41_sel_tab_bytes((T + cb - 1) / cb);
    v41_candidates_kernel<<<rows, 256, tb, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, cb, kblocks, start_pos,
                                                                   ratio, keep, ldk, tb > 0);
    V41_ERR();
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
        isel[j] = bi;
        float wv = 0.f;
        if ((bi & 31) == lane) {
            int slot = bi >> 5;
#pragma unroll
            for (int i = 0; i < PER; ++i)
                if (i == slot && i < per) {
                    wv = sc[i];
                    bs[i] = -INFINITY;
                }
        }
        wsel[j] = __shfl_sync(0xffffffffu, wv, bi & 31);
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

__global__ void v41_swiglu_kernel(const v41_bf* __restrict__ gate, const v41_bf* __restrict__ up, long ld,
                                  const float* __restrict__ wrow, float limit, v41_bf* __restrict__ out, long ldo,
                                  int inter) {
    int r = blockIdx.x;
    float wv = wrow ? wrow[r] : 1.f;
    for (int i = threadIdx.x; i < inter; i += blockDim.x) {
        float g = v41_bf2f(gate[(long)r * ld + i]);
        float u = v41_bf2f(up[(long)r * ld + i]);
        if (limit > 0.f) {
            u = fminf(fmaxf(u, -limit), limit);
            g = fminf(g, limit);
        }
        float h = __fmul_rn(g / __fadd_rn(1.f, expf(-g)), u);
        if (wrow) h = __fmul_rn(wv, h);
        out[(long)r * ldo + i] = v41_f2bf(h);
    }
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
// tokens in order. `err` counts expert ids outside [0, ne) (skipped, their order slot -1).

#define V41_PLAN_CH 256

__global__ void __launch_bounds__(V41_PLAN_CH) v41_moe_count_kernel(const int* __restrict__ sel, int rows, int topk,
                                                                     int ne, int* __restrict__ counts,
                                                                     int* __restrict__ err) {
    extern __shared__ int v41_cnt[];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) v41_cnt[e] = 0;
    __syncthreads();
    int t = blockIdx.x * V41_PLAN_CH + threadIdx.x;
    if (t < rows)
        for (int j = 0; j < topk; ++j) {
            int e = sel[(long)t * topk + j];
            if (e < 0 || e >= ne) atomicAdd(err, 1);
            else atomicAdd(&v41_cnt[e], 1);
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

// counts: scratch [ceil(rows / 256)][ne] ints; tiles: room for ceil(rows * topk / bm) + min(ne, rows * topk);
// the tile list covers experts [e0, e1)
extern "C" int memra_v41_moe_plan(const int* sel, const float* selw, int rows, int topk, int ne, int bm, int e0, int e1,
                                  int* counts, int* off, int* ptok, float* pw, int* order, void* tiles, int* ntiles,
                                  int* err, void* stream) {
    if (rows <= 0) return rows < 0 ? 41025 : 0;
    if (ne <= 0 || ne > 1024 || topk <= 0 || topk > 16 || bm <= 0 || e0 < 0 || e1 > ne || e0 > e1) return 41025;
    cudaStream_t st = (cudaStream_t)stream;
    int nch = (rows + V41_PLAN_CH - 1) / V41_PLAN_CH;
    v41_moe_count_kernel<<<nch, V41_PLAN_CH, ne * 4, st>>>(sel, rows, topk, ne, counts, err);
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

// N-rank routed-expert rows without a host read of the plan. Every rank writes its experts' rows into region
// `rank` of a buffer of nr regions of np rows, the regions are gathered whole, and each pair's row is then read
// from the region of the rank that owns its expert. Pairs are ordered by expert and rank q owns the experts
// [q * per, (q + 1) * per), so q is the number of boundaries off[r * per], r = 1..nr-1, at or below the pair.
// out[i] = order[i] + q * np (-1 stays -1). *bad is set when the plan placed other than np pairs, the check
// the host made on the offsets it read.
__global__ void v41_moe_owner_kernel(const int* __restrict__ order, int n, const int* __restrict__ off, int per,
                                     int nr, int np, int* __restrict__ out, int* __restrict__ bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && off[(long)per * nr] != np) *bad = 1;
    if (i >= n) return;
    int p = order[i], q = 0;
    for (int r = 1; r < nr; ++r) q += p >= off[(long)r * per];
    out[i] = p < 0 ? p : p + q * np;
}

extern "C" int memra_v41_moe_owner(const int* order, int n, const int* off, int per, int nr, int np, int* out,
                                   int* bad, void* stream) {
    if (n <= 0) return n < 0 ? 41017 : 0;
    if (per <= 0 || nr <= 0 || np <= 0) return 41017;
    v41_moe_owner_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(order, n, off, per, nr, np, out, bad);
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
                                                                   int rows, float* __restrict__ parts) {
    __shared__ __align__(16) v41_bf sx[V41_HCM][V41_HCT][32];  // raw bf16, widened where it is read
    int b = blockIdx.y;  // virtual warp
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
    if (rows <= V41_SM_MAX && smem1 <= 99 * 1024 && n1 % 8 == 0 && (uintptr_t)fn % 16 == 0) {
        // up to 8 rows: the one-token kernel a row (6.7 us, where the 16-token pass takes 53)
        if (smem1 > 48 * 1024) V41_SMEM_LIMIT(v41_hc_mixes_part1_kernel, 99 * 1024);
        v41_hc_mixes_part1_kernel<<<dim3(rows, 8, (nmix + 4) / 4), 128, smem1, st>>>((const v41_bf*)x, fn, nmix, n1,
                                                                                   parts);
    } else if (rows == 1) {
        v41_hc_mixes_part_kernel<1, 40><<<dim3(1, 8), 32 * (nmix + 1), 0, st>>>((const v41_bf*)x, fn, nmix, hc * d,
                                                                               rows, parts);
    } else {
        dim3 g1((rows + 15) / 16, 8);
        v41_hc_mixes_part_kernel<16, 16><<<g1, 32 * (nmix + 1), 0, st>>>((const v41_bf*)x, fn, nmix, hc * d, rows,
                                                                          parts);
    }
    v41_hc_finish_kernel<<<(rows + 15) / 16, 256, 0, st>>>(parts, nmix, hc * d, norm_eps, scale, base, hc, iters,
                                                           hc_eps, mixes, pre, post, comb, rows);
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

__global__ void v41_engram_gate_kernel(const v41_bf* __restrict__ h, const v41_bf* __restrict__ kv,
                                       const v41_bf* __restrict__ qw, const v41_bf* __restrict__ kw, int hc, int d,
                                       float eps, float inv_sqrt_d, v41_bf* __restrict__ out) {
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

// ---------------------------------------------------------------------------------------------
// Small N-rank gathers inside kernels (the device-push transport). With peer access on the pools a
// kernel stores straight into another card's memory. The producer packs its region into its slot
// of every peer's inbox and then stores the gather's sequence number in the slot's word; the
// consumer's kernel spins on the words of its own inbox and unpacks each slot into the producer's
// region of its buffer. No event, no host round trip and no copy call sits between a rank's last
// kernel of a part and its first kernel of the next one.
// Order: every thread fences its data stores to the system before the block's barrier and the
// word is stored after it, so a consumer that sees the word reads the data (word_first reverses
// the two: the gate's control case, which must fail). A slot holds nothing its reader writes, so
// no cache line has a writer on each card, and it is rewritten two gathers later, when its reader
// is done (the caller's law: every rank pushes and pulls in every gather and the slots alternate).

#define V41_PUSH_PEERS 7
#define V41_PUSH_MAX (256 << 10)
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

__global__ void v41_push_kernel(const uint8_t* src, long pitch, int width, int rows, V41Push dst, int npeer,
                                uint32_t seq, int word_first) {
    if (word_first && threadIdx.x == 0) {
        for (int p = 0; p < npeer; ++p) *(volatile uint32_t*)dst.w[p] = seq;
        __threadfence_system();
    }
    for (int p = 0; p < npeer; ++p) v41_blk_move<1>((uint8_t*)src, pitch, width, rows, dst.d[p]);
    __threadfence_system();
    __syncthreads();
    if (!word_first && threadIdx.x == 0) {
        for (int p = 0; p < npeer; ++p) *(volatile uint32_t*)dst.w[p] = seq;
        __threadfence_system();
    }
}

__global__ void v41_pull_kernel(V41Pull in, int npeer, uint32_t seq, uint8_t* dst, int* bad) {
    __shared__ int go;
    if (threadIdx.x == 0) {
        int ok = 0;
        for (long it = 0; it < V41_PULL_SPIN && !ok; ++it) {
            ok = 1;
            for (int p = 0; p < npeer; ++p) ok &= (*(volatile const uint32_t*)in.w[p] == seq) ? 1 : 0;
        }
        go = ok;
        if (!ok) *bad = *bad | 0x10000;  // sticky, read with the step's logits
    }
    __syncthreads();
    if (!go) return;
    for (int p = 0; p < npeer; ++p)
        v41_blk_move<0>(dst + in.off[p], in.pitch[p], in.width[p], in.rows[p], (uint8_t*)in.d[p]);
}

static int v41_push_threads(long bytes) {
    long t = ((bytes + 3) / 4 + 31) / 32 * 32;
    return (int)(t < 32 ? 32 : t > 1024 ? 1024 : t);
}

// The producer's half of a small gather: `rows` rows of `width` bytes at `src` (`pitch` apart) into the
// `npeer` slots of `dst` (a host V41Push), then `seq` into their words.
extern "C" int memra_v41_push(const void* src, long pitch, int width, int rows, const void* dst, int npeer,
                              unsigned seq, int word_first, void* stream) {
    if (npeer < 0 || npeer > V41_PUSH_PEERS || width < 0 || rows < 0 || (long)rows * width > V41_PUSH_MAX)
        return 41040;
    if (rows > 1 && pitch < width) return 41040;
    V41Push d = *(const V41Push*)dst;
    v41_push_kernel<<<1, v41_push_threads((long)rows * width), 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, pitch, width, rows, d, npeer, seq, word_first);
    V41_ERR();
}

// The consumer's half: wait for `seq` in the `npeer` slots of `in` (a host V41Pull), then unpack each into its
// region of `dst`. A peer that never arrives sets bit 16 of *bad and nothing is unpacked.
extern "C" int memra_v41_pull(const void* in, int npeer, unsigned seq, void* dst, int* bad, void* stream) {
    if (npeer < 0 || npeer > V41_PUSH_PEERS) return 41041;
    V41Pull s = *(const V41Pull*)in;
    long most = 0;
    for (int p = 0; p < npeer; ++p) {
        long b = (long)s.rows[p] * s.width[p];
        if (s.width[p] < 0 || s.rows[p] < 0 || b > V41_PUSH_MAX || (s.rows[p] > 1 && s.pitch[p] < s.width[p]))
            return 41041;
        most = b > most ? b : most;
    }
    v41_pull_kernel<<<1, v41_push_threads(most), 0, (cudaStream_t)stream>>>(s, npeer, seq, (uint8_t*)dst, bad);
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
