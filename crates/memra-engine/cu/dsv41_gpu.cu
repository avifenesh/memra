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

typedef uint16_t v41_bf;  // raw bf16 bits

#define V41_ERR()                                                   \
    do {                                                            \
        cudaError_t e_ = cudaGetLastError();                        \
        return e_ == cudaSuccess ? 0 : 10000 + (int)e_;             \
    } while (0)

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

// e2m1 nibble -> the identical e4m3 byte (every e2m1 value is an e4m3 value)
__device__ __forceinline__ uint32_t v41_fp4_to_e4m3x4(uint32_t nib16) {
    // nib16: 4 nibbles in the low 16 bits, element 0 in the lowest nibble
    const uint32_t lut_lo = 0x3C383000u;  // 0, 0.5, 1, 1.5
    const uint32_t lut_hi = 0x4C484440u;  // 2, 3, 4, 6
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t n = (nib16 >> (4 * i)) & 0xF;
        uint32_t m = n & 7;
        uint32_t b = m < 4 ? (lut_lo >> (8 * m)) & 0xFF : (lut_hi >> (8 * (m - 4))) & 0xFF;
        b |= (n & 8) << 4;  // sign
        out |= b << (8 * i);
    }
    return out;
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
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = v41_bf2f(xr[i]);
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    float tot = v41_block_sum(ss, red);
    float rs = rsqrtf(__fadd_rn(tot / (float)d, eps));
    v41_bf* yr = y + (long)blockIdx.x * ldy;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = v41_bf2f(xr[i]);
        yr[i] = v41_f2bf(__fmul_rn(v41_bf2f(w[i]), __fmul_rn(v, rs)));
    }
}

extern "C" int memra_v41_rmsnorm(const void* x, long ldx, const void* w, void* y, long ldy, int rows, int d, float eps,
                                 void* stream) {
    if (rows <= 0) return rows < 0 ? 41003 : 0;
    int th = d >= 1024 ? 256 : 128;
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

// Grouped expert GEMM: one launch for every expert of a layer that has rows. Group z owns rows
// [row0, row0 + rows) of A/As/C and its own weight, scales and minted scale2; blockIdx.y walks its
// 64-row tiles (tiles past the group's rows exit at once). Same tile body as the plain kernel.
struct V41Group {
    long row0;
    int rows;
    float scale2;
    const uint8_t* w;
    const uint8_t* s;
};

template <int WF, int OUTF>
__global__ void __launch_bounds__(256) v41_gemm_f8_grouped_kernel(const uint8_t* __restrict__ A, long lda,
                                                                  const uint8_t* __restrict__ As, long ldas,
                                                                  const V41Group* __restrict__ grp, long ldb,
                                                                  long ldbs, void* __restrict__ C, long ldc, int N,
                                                                  int K) {
    __shared__ __align__(16) uint8_t sA[V41_GM * V41_GS];
    __shared__ __align__(16) uint8_t sB[V41_GN * V41_GS];
    __shared__ float sSa[V41_GM];
    __shared__ float sSb[V41_GN];
    const V41Group gr = grp[blockIdx.z];
    int m0 = blockIdx.y * V41_GM;
    if (m0 >= gr.rows) return;  // the whole block leaves together
    void* c = OUTF ? (void*)((float*)C + gr.row0 * ldc) : (void*)((v41_bf*)C + gr.row0 * ldc);
    v41_gemm_f8_tile<WF, OUTF>(A + gr.row0 * lda, lda, As + gr.row0 * ldas, ldas, gr.w, ldb, gr.s, ldbs, gr.scale2, c,
                               ldc, gr.rows, N, K, m0, blockIdx.x * V41_GN, sA, sB, sSa, sSb);
}

// grp: device array of `ng` V41Group (layout above, 32 bytes each); max_rows bounds every group's rows.
extern "C" int memra_v41_gemm_f8_grouped(const void* a, long lda, const void* as, long ldas, const void* grp, int ng,
                                         int max_rows, long ldb, long ldbs, int wfmt, void* c, long ldc, int out_f32,
                                         int n, int k, void* stream) {
    if (k % 32 || n <= 0 || lda % 16 || max_rows < 0 || ng < 0 || ng > 65535) return 41016;
    if (wfmt == 0 && ldb % 16) return 41016;
    if (wfmt != 0 && ldb % 8) return 41016;
    if (ng == 0 || max_rows == 0) return 0;
    dim3 grid((n + V41_GN - 1) / V41_GN, (max_rows + V41_GM - 1) / V41_GM, ng);
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as;
    const V41Group* G = (const V41Group*)grp;
#define V41_GEMM_GR(WF, OF) \
    v41_gemm_f8_grouped_kernel<WF, OF><<<grid, 256, 0, st>>>(A, lda, As, ldas, G, ldb, ldbs, c, ldc, n, k)
    if (wfmt == 0) { if (out_f32) V41_GEMM_GR(0, 1); else V41_GEMM_GR(0, 0); }
    else if (wfmt == 1) { if (out_f32) V41_GEMM_GR(1, 1); else V41_GEMM_GR(1, 0); }
    else if (wfmt == 2) { if (out_f32) V41_GEMM_GR(2, 1); else V41_GEMM_GR(2, 0); }
    else return 41016;
#undef V41_GEMM_GR
    V41_ERR();
}

extern "C" int memra_v41_gemm_f8(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                 const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                 int m, int n, int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || lda % 16) return 41005;
    if (wfmt == 0 && ldb % 16) return 41005;
    if (wfmt != 0 && ldb % 8) return 41005;
    if (m == 0) return 0;
    dim3 grid((n + V41_GN - 1) / V41_GN, (m + V41_GM - 1) / V41_GM);
    cudaStream_t st = (cudaStream_t)stream;
    const uint8_t *A = (const uint8_t*)a, *As = (const uint8_t*)as, *B = (const uint8_t*)b, *Bs = (const uint8_t*)bs;
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
            uint32_t af[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int r = wm * 32 + mi * 16;
                af[mi][0] = v41_lds32(&sA[(r + g) * V41_HS + kb + t * 4]);
                af[mi][1] = v41_lds32(&sA[(r + g + 8) * V41_HS + kb + t * 4]);
                af[mi][2] = v41_lds32(&sA[(r + g) * V41_HS + kb + 16 + t * 4]);
                af[mi][3] = v41_lds32(&sA[(r + g + 8) * V41_HS + kb + 16 + t * 4]);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int c = wn * 32 + ni * 8;
                uint32_t b0 = v41_lds32(&sB[(c + g) * V41_HS + kb + t * 4]);
                uint32_t b1 = v41_lds32(&sB[(c + g) * V41_HS + kb + 16 + t * 4]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) v41_mma_bf16(acc[mi][ni], af[mi], b0, b1);
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

extern "C" int memra_v41_gemm_bf16(const void* a, long lda, long sAz, const void* b, long ldb, long sBz, void* c,
                                   long ldc, long sCz, int out_f32, int m, int n, int k, int batch, void* stream) {
    if (k % 32 || m < 0 || n <= 0 || batch <= 0 || lda % 8 || ldb % 8 || sAz % 8 || sBz % 8) return 41006;
    if (m == 0) return 0;
    dim3 grid((n + V41_GN - 1) / V41_GN, (m + V41_GM - 1) / V41_GM, batch);
    cudaStream_t st = (cudaStream_t)stream;
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
// C[M,N] f32, 64x64 tile, 256 threads x 4x4 outputs, K tiles of 32.

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
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            int i = tid + r * 256;
            int row = i >> 5, kk = i & 31;
            sA[kk][row] = (m0 + row < M) ? v41_bf2f(A[(long)(m0 + row) * lda + k0 + kk]) : 0.f;
            sB[kk][row] = (n0 + row < N) ? v41_bf2f(B[(long)(n0 + row) * ldb + k0 + kk]) : 0.f;
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

extern "C" int memra_v41_gemm_ffma(const void* a, long lda, const void* b, long ldb, float* c, long ldc, int m, int n,
                                   int k, void* stream) {
    if (k % 32 || m < 0 || n <= 0) return 41023;
    if (m == 0) return 0;
    dim3 grid((n + 63) / 64, (m + 63) / 64);
    v41_gemm_ffma_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc,
                                                                 m, n, k);
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
#define V41_ASMEM (V41_AH * V41_ARS + V41_AK * V41_ARS + V41_AH * V41_APS + 4 * 16 * 4 * 2 + V41_AK * 4)

__global__ void __launch_bounds__(128) v41_sparse_attn_kernel(const v41_bf* __restrict__ q, long ldq,
                                                              const v41_bf* __restrict__ kv, const int* __restrict__ idx,
                                                              long ldi, int topk, const float* __restrict__ sink,
                                                              float scale, v41_bf* __restrict__ o, long ldo) {
    extern __shared__ __align__(16) uint8_t sm[];
    uint8_t* sQ = sm;
    uint8_t* sKV = sQ + V41_AH * V41_ARS;
    uint8_t* sP = sKV + V41_AK * V41_ARS;
    float* sMax = (float*)(sP + V41_AH * V41_APS);  // [4 warps][16 rows]
    float* sSum = sMax + 64;
    int* sIdx = (int*)(sSum + 64);
    int mq = blockIdx.x, h0 = blockIdx.y * V41_AH;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;

    const uint8_t* qg = (const uint8_t*)(q + (long)mq * ldq + (long)h0 * V41_AD);
    for (int i = tid; i < V41_AH * 64; i += 128) {
        int r = i >> 6, c = i & 63;
        *(uint4*)(sQ + r * V41_ARS + c * 16) = *(const uint4*)(qg + (long)r * V41_AD * 2 + c * 16);
    }
    float m_run[2] = {-1e30f, -1e30f}, l_run[2] = {0.f, 0.f};
    float oacc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) oacc[i][e] = 0.f;

    int nblk = (topk + V41_AK - 1) / V41_AK;
    for (int bk = 0; bk < nblk; ++bk) {
        if (tid < V41_AK) {
            int j = bk * V41_AK + tid;
            sIdx[tid] = j < topk ? idx[(long)mq * ldi + j] : -1;
        }
        __syncthreads();
        for (int i = tid; i < V41_AK * 64; i += 128) {
            int r = i >> 6, c = i & 63;
            int ix = sIdx[r];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (ix >= 0) v = *(const uint4*)((const uint8_t*)(kv + (long)ix * V41_AD) + c * 16);
            *(uint4*)(sKV + r * V41_ARS + c * 16) = v;
        }
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
            uint32_t af[4];
            af[0] = v41_lds32(sQ + g * V41_ARS + kb + t * 4);
            af[1] = v41_lds32(sQ + (g + 8) * V41_ARS + kb + t * 4);
            af[2] = v41_lds32(sQ + g * V41_ARS + kb + 16 + t * 4);
            af[3] = v41_lds32(sQ + (g + 8) * V41_ARS + kb + 16 + t * 4);
#pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                int key = w * 16 + ni * 8 + g;
                uint32_t b0 = v41_lds32(sKV + key * V41_ARS + kb + t * 4);
                uint32_t b1 = v41_lds32(sKV + key * V41_ARS + kb + 16 + t * 4);
                v41_mma_bf16(s[ni], af, b0, b1);
            }
        }
        float rmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int key = w * 16 + ni * 8 + 2 * t + (e & 1);
                float v = sIdx[key] >= 0 ? __fmul_rn(s[ni][e], scale) : -INFINITY;
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
            af[0] = v41_lds32(sP + g * V41_APS + kb + t * 4);
            af[1] = v41_lds32(sP + (g + 8) * V41_APS + kb + t * 4);
            af[2] = v41_lds32(sP + g * V41_APS + kb + 16 + t * 4);
            af[3] = v41_lds32(sP + (g + 8) * V41_APS + kb + 16 + t * 4);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                int c0 = w * 128 + nt * 8;
                uint32_t b0, b1;
                v41_ldsm_x2_trans(b0, b1, sKV + (kk * 16 + (lane & 15)) * V41_ARS + c0 * 2);
                v41_mma_bf16(oacc[nt], af, b0, b1);
            }
        }
        __syncthreads();
    }
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
    // per device and cheap: set it on every call so a second card never launches without it
    cudaFuncSetAttribute(v41_sparse_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, V41_ASMEM);
    dim3 grid(m, heads / V41_AH);
    v41_sparse_attn_kernel<<<grid, 128, V41_ASMEM, (cudaStream_t)stream>>>(
        (const v41_bf*)q, ldq, (const v41_bf*)kv, idx, ldi, topk, sink, scale, (v41_bf*)o, ldo);
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

#define V41_IQB 4  // query rows per block: the 64-key tile is loaded once for all of them

// MI 16-head MMA tiles (2 = all 32 heads, the model's program). PART 1 writes the fp32 head sum
// unrounded and unmasked (the head-split measurement arm: `memra_v41_index_score_finish` adds the
// ranks' partials, rounds and masks).
template <int MI, int PART>
__global__ void __launch_bounds__(128) v41_index_score_kernel(const v41_bf* __restrict__ q, long ldq,
                                                              const v41_bf* __restrict__ k, int T,
                                                              const v41_bf* __restrict__ w, long ldw, float wscale,
                                                              int start_pos,
                                                              int ratio, const uint8_t* __restrict__ cand, long ldc,
                                                              int cb, void* __restrict__ outv, long ldo, int rows) {
    __shared__ __align__(16) uint8_t sQ[32 * V41_IRS];
    __shared__ __align__(16) uint8_t sK[64 * V41_IRS];
    __shared__ float sW[32];
    int t0 = blockIdx.x * 64;
    int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    for (int i = tid; i < 64 * 16; i += 128) {
        int r = i >> 4, c = i & 15;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t0 + r < T) v = *(const uint4*)((const uint8_t*)(k + (long)(t0 + r) * 128) + c * 16);
        *(uint4*)(sK + r * V41_IRS + c * 16) = v;
    }
    for (int rq = 0; rq < V41_IQB; ++rq) {
        int row = blockIdx.y * V41_IQB + rq;
        if (row >= rows) break;
        __syncthreads();  // the previous row's readers are done with sQ / sW
        const uint8_t* qg = (const uint8_t*)(q + (long)row * ldq);
        for (int i = tid; i < MI * 16 * 16; i += 128) {
            int r = i >> 4, c = i & 15;
            *(uint4*)(sQ + r * V41_IRS + c * 16) = *(const uint4*)(qg + r * 256 + c * 16);
        }
        // weights_proj(x) * (index_head_dim^-0.5 * index_heads^-0.5), a bf16 product in the reference
        if (tid < MI * 16) sW[tid] = v41_bf2f(v41_f2bf(__fmul_rn(v41_bf2f(w[(long)row * ldw + tid]), wscale)));
        __syncthreads();
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
                int r = mi * 16;
                af[0] = v41_lds32(sQ + (r + g) * V41_IRS + kb + t * 4);
                af[1] = v41_lds32(sQ + (r + g + 8) * V41_IRS + kb + t * 4);
                af[2] = v41_lds32(sQ + (r + g) * V41_IRS + kb + 16 + t * 4);
                af[3] = v41_lds32(sQ + (r + g + 8) * V41_IRS + kb + 16 + t * 4);
    #pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    int key = wp * 16 + ni * 8 + g;
                    uint32_t b0 = v41_lds32(sK + key * V41_IRS + kb + t * 4);
                    uint32_t b1 = v41_lds32(sK + key * V41_IRS + kb + 16 + t * 4);
                    v41_mma_bf16(d[mi][ni], af, b0, b1);
                }
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
    }
}

extern "C" int memra_v41_index_score(const void* q, long ldq, const void* k, int T, const void* w, long ldw,
                                     float wscale, int start_pos, int ratio, const uint8_t* cand, long ldc, int cb, void* out,
                                     long ldo, int rows, void* stream) {
    if (rows <= 0 || T <= 0) return (rows < 0 || T < 0) ? 41009 : 0;
    if (ratio <= 0 || (cand && cb <= 0) || ldq % 8) return 41009;
    dim3 grid((T + 63) / 64, (rows + V41_IQB - 1) / V41_IQB);
    v41_index_score_kernel<2, 0><<<grid, 128, 0, (cudaStream_t)stream>>>((const v41_bf*)q, ldq, (const v41_bf*)k, T,
                                                                         (const v41_bf*)w, ldw, wscale, start_pos, ratio,
                                                                         cand, ldc, cb, out, ldo, rows);
    V41_ERR();
}

// Head-split arm (measurement): the fp32 sum over `heads` (16 or 32) local heads of q [rows][heads*128]
// and w [rows][heads], unrounded and unmasked, into part [rows][T] (ldo floats).
extern "C" int memra_v41_index_score_part(const void* q, long ldq, const void* k, int T, const void* w, long ldw,
                                          float wscale, int heads, float* part, long ldo, int rows, void* stream) {
    if (rows <= 0 || T <= 0) return (rows < 0 || T < 0) ? 41018 : 0;
    if ((heads != 16 && heads != 32) || ldq % 8) return 41018;
    dim3 grid((T + 63) / 64, (rows + V41_IQB - 1) / V41_IQB);
    cudaStream_t st = (cudaStream_t)stream;
    if (heads == 16)
        v41_index_score_kernel<1, 1><<<grid, 128, 0, st>>>((const v41_bf*)q, ldq, (const v41_bf*)k, T,
                                                           (const v41_bf*)w, ldw, wscale, 0, 1, nullptr, 0, 1, part,
                                                           ldo, rows);
    else
        v41_index_score_kernel<2, 1><<<grid, 128, 0, st>>>((const v41_bf*)q, ldq, (const v41_bf*)k, T,
                                                           (const v41_bf*)w, ldw, wscale, 0, 1, nullptr, 0, 1, part,
                                                           ldo, rows);
    V41_ERR();
}

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

// block-wide exclusive scan of a 0/1 flag (256 threads), returns the prefix; *total gets the sum
__device__ __forceinline__ int v41_scan_flag(int f, int* wsum, int* total) {
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    unsigned bal = __ballot_sync(0xffffffffu, f);
    int pre = __popc(bal & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) wsum[w] = __popc(bal);
    __syncthreads();
    int off = 0, tot = 0;
    for (int i = 0; i < 8; ++i) {
        if (i < w) off += wsum[i];
        tot += wsum[i];
    }
    *total = tot;
    return off + pre;
}

// selects up to k keys; emit(pos, slot) for each selected position in ascending order; returns count
template <typename Emit>
__device__ int v41_select(const V41Keys& K, int k, int* hist, int* wsum, Emit emit) {
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
    // emit in position order
    int base = 0, eq_seen = 0;
    for (int i0 = 0; i0 < K.n; i0 += 256) {
        int i = i0 + tid;
        int key = i < K.n ? K.get(i) : -1;
        int is_eq = (key >= 0 && need < nvalid && key == thr) ? 1 : 0;
        int eq_tot;
        int eq_rank = v41_scan_flag(is_eq, wsum, &eq_tot) + eq_seen;
        int sel = 0;
        if (key >= 0) {
            if (need >= nvalid) sel = 1;
            else if (key > thr) sel = 1;
            else if (is_eq && eq_rank < eq_need) sel = 1;
        }
        if (need == 0) sel = 0;
        int sel_tot;
        int slot = v41_scan_flag(sel, wsum, &sel_tot) + base;
        if (sel) emit(i, slot);
        base += sel_tot;
        eq_seen += eq_tot;
    }
    return base;
}

__global__ void __launch_bounds__(256) v41_topk_kernel(const v41_bf* __restrict__ score, long lds, int T, int k,
                                                       int out_w, int offset, int* __restrict__ out, long ldo) {
    __shared__ int hist[256];
    __shared__ int wsum[8];
    int row = blockIdx.x;
    V41Keys K{score + (long)row * lds, T, 0, T, -1};
    int* orow = out + (long)row * ldo;
    int n = v41_select(K, k, hist, wsum, [&](int pos, int slot) { orow[slot] = pos + offset; });
    __syncthreads();
    for (int i = n + threadIdx.x; i < out_w; i += 256) orow[i] = -1;
}

extern "C" int memra_v41_topk(const void* score, long lds, int T, int k, int out_w, int offset, int* out, long ldo,
                              int rows, void* stream) {
    if (rows <= 0) return rows < 0 ? 41010 : 0;
    if (k > out_w || T < 0) return 41010;
    v41_topk_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, k, out_w, offset, out, ldo);
    V41_ERR();
}

// select_candidate_blocks: block score = max over cb positions, the block holding the row's newest
// visible position pinned to +inf, top-k blocks kept when finite. keep [rows][nb] u8.
__global__ void __launch_bounds__(256) v41_candidates_kernel(const v41_bf* __restrict__ score, long lds, int T,
                                                             int cb, int kblocks, int start_pos, int ratio,
                                                             uint8_t* __restrict__ keep, long ldk) {
    __shared__ int hist[256];
    __shared__ int wsum[8];
    int row = blockIdx.x;
    int nb = (T + cb - 1) / cb;
    int visible = (start_pos + row + 1) / ratio;
    int pin = visible > 0 ? (visible - 1) / cb : -1;
    V41Keys K{score + (long)row * lds, nb, cb, T, pin};
    uint8_t* krow = keep + (long)row * ldk;
    for (int i = threadIdx.x; i < nb; i += 256) krow[i] = 0;
    __syncthreads();
    v41_select(K, kblocks, hist, wsum, [&](int pos, int) { krow[pos] = 1; });
}

extern "C" int memra_v41_candidates(const void* score, long lds, int T, int cb, int kblocks, int start_pos, int ratio,
                                    uint8_t* keep, long ldk, int rows, void* stream) {
    if (rows <= 0) return rows < 0 ? 41011 : 0;
    if (cb <= 0 || ratio <= 0) return 41011;
    v41_candidates_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)score, lds, T, cb, kblocks, start_pos,
                                                                  ratio, keep, ldk);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// MoE gate (sqrtsoftplus): score = sqrt(softplus(raw)); experts by score + bias (top-k, ties to the
// lower id, descending); weights are the unbiased scores, normalized by (sum + 1e-20) only when
// top-k > 1, times route_scale. One warp per token; ne <= 1024, topk <= 16.

__global__ void v41_route_kernel(const float* __restrict__ raw, long ldr, const float* __restrict__ bias, int ne,
                                 int topk, int norm, float route_scale, int* __restrict__ sel, float* __restrict__ selw,
                                 int rows) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= rows) return;
    const float* r = raw + (long)wid * ldr;
    float sc[32], bs[32];
    int per = (ne + 31) / 32;
    for (int i = 0; i < per; ++i) {
        int e = lane + 32 * i;
        float v = e < ne ? r[e] : 0.f;
        float sp = v > 20.f ? v : log1pf(expf(v));
        sc[i] = sqrtf(sp);
        bs[i] = e < ne ? __fadd_rn(sc[i], bias ? bias[e] : 0.f) : -INFINITY;
    }
    float wsel[16];
    int isel[16];
    for (int j = 0; j < topk; ++j) {
        float best = -INFINITY;
        int bi = 0x7FFFFFFF;
        for (int i = 0; i < per; ++i) {
            int e = lane + 32 * i;
            if (e < ne && (bs[i] > best || (bs[i] == best && e < bi))) {
                best = bs[i];
                bi = e;
            }
        }
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
            for (int i = 0; i < per; ++i)
                if (i == slot) {
                    wv = sc[i];
                    bs[i] = -INFINITY;
                }
        }
        wsel[j] = __shfl_sync(0xffffffffu, wv, bi & 31);
    }
    if (lane == 0) {
        float sum = 0.f;
        for (int j = 0; j < topk; ++j) sum = __fadd_rn(sum, wsel[j]);
        float den = __fadd_rn(sum, 1e-20f);
        for (int j = 0; j < topk; ++j) {
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
    v41_route_kernel<<<(rows * 32 + 127) / 128, 128, 0, (cudaStream_t)stream>>>(raw, ldr, bias, ne, topk, norm,
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

// The routed-expert combine of the grouped dispatch: out[t] = bf16(sum_j float(slots[order[t][j]]) +
// float(shared[t])), j in ascending expert id, fp32 from zero. This is the per-expert loop's sum
// (scatter_add launched in ascending expert order into a zeroed accumulator, then add_cast) in the same
// order, so the result does not depend on which rank computed which expert. order -1 skips a slot.
__global__ void v41_moe_combine_kernel(const v41_bf* __restrict__ slots, long lds, const int* __restrict__ order,
                                       int topk, const v41_bf* __restrict__ shared, long ldsh, v41_bf* __restrict__ out,
                                       long ldo, int d) {
    int t = blockIdx.x;
    const int* o = order + (long)t * topk;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
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
    v41_moe_combine_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)slots, lds, order, topk,
                                                                   (const v41_bf*)shared, ldsh, (v41_bf*)out, ldo, d);
    V41_ERR();
}

// ---------------------------------------------------------------------------------------------
// Single-pass mHC. The stream is bf16 [s][hc][d].
// mixes[s][(2+hc)*hc] = (flat . fn^T) * rsqrt(mean(flat^2) + eps), flat = the hc*d stream in fp32.

__global__ void v41_hc_mixes_kernel(const v41_bf* __restrict__ x, const float* __restrict__ fn, int nmix, int n,
                                    float eps, float* __restrict__ mixes) {
    __shared__ float red[32];
    __shared__ float part[8][32];
    int s = blockIdx.x;
    const v41_bf* xr = x + (long)s * n;
    float acc[32];
    for (int j = 0; j < nmix; ++j) acc[j] = 0.f;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = v41_bf2f(xr[i]);
        ss = __fadd_rn(ss, __fmul_rn(v, v));
        for (int j = 0; j < nmix; ++j) acc[j] = __fadd_rn(acc[j], __fmul_rn(v, fn[(long)j * n + i]));
    }
    float tot = v41_block_sum(ss, red);
    float rs = rsqrtf(__fadd_rn(tot / (float)n, eps));
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int j = 0; j < nmix; ++j) {
        float v = v41_warp_sum(acc[j]);
        if (lane == 0) part[w][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < nmix) {
        float v = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) v = __fadd_rn(v, part[i][threadIdx.x]);
        mixes[(long)s * nmix + threadIdx.x] = __fmul_rn(v, rs);
    }
}

// pre/post/comb from mixes (reference hc_split_sinkhorn), one thread per token; hc <= 4
__global__ void v41_hc_sinkhorn_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                       const float* __restrict__ base, int hc, int iters, float eps,
                                       float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
                                       int rows) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= rows) return;
    int nm = (2 + hc) * hc;
    const float* m = mixes + (long)s * nm;
    for (int j = 0; j < hc; ++j) {
        float a = __fadd_rn(__fmul_rn(m[j], scale[0]), base[j]);
        pre[(long)s * hc + j] = __fadd_rn(1.f / __fadd_rn(1.f, expf(-a)), eps);
        float b = __fadd_rn(__fmul_rn(m[hc + j], scale[1]), base[hc + j]);
        post[(long)s * hc + j] = __fmul_rn(2.f, 1.f / __fadd_rn(1.f, expf(-b)));
    }
    float c[4][4];
    for (int j = 0; j < hc; ++j) {
        float mx = -INFINITY;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = __fadd_rn(__fmul_rn(m[2 * hc + j * hc + k], scale[2]), base[2 * hc + j * hc + k]);
            mx = fmaxf(mx, c[j][k]);
        }
        float sum = 0.f;
        for (int k = 0; k < hc; ++k) {
            c[j][k] = expf(c[j][k] - mx);
            sum = __fadd_rn(sum, c[j][k]);
        }
        for (int k = 0; k < hc; ++k) c[j][k] = __fadd_rn(c[j][k] / sum, eps);
    }
    for (int k = 0; k < hc; ++k) {  // comb / (comb.sum(-2) + eps)
        float cs = 0.f;
        for (int j = 0; j < hc; ++j) cs = __fadd_rn(cs, c[j][k]);
        for (int j = 0; j < hc; ++j) c[j][k] = c[j][k] / __fadd_rn(cs, eps);
    }
    for (int it = 0; it < iters - 1; ++it) {
        for (int j = 0; j < hc; ++j) {
            float rsum = 0.f;
            for (int k = 0; k < hc; ++k) rsum = __fadd_rn(rsum, c[j][k]);
            for (int k = 0; k < hc; ++k) c[j][k] = c[j][k] / __fadd_rn(rsum, eps);
        }
        for (int k = 0; k < hc; ++k) {
            float cs = 0.f;
            for (int j = 0; j < hc; ++j) cs = __fadd_rn(cs, c[j][k]);
            for (int j = 0; j < hc; ++j) c[j][k] = c[j][k] / __fadd_rn(cs, eps);
        }
    }
    for (int j = 0; j < hc; ++j)
        for (int k = 0; k < hc; ++k) comb[(long)s * hc * hc + j * hc + k] = c[j][k];
}

extern "C" int memra_v41_hc_mixes(const void* x, const float* fn, const float* scale, const float* base, int rows,
                                  int hc, int d, int iters, float norm_eps, float hc_eps, float* mixes, float* pre,
                                  float* post, float* comb, void* stream) {
    int nmix = (2 + hc) * hc;
    if (hc <= 0 || hc > 4 || nmix > 32) return 41016;
    if (rows <= 0) return rows < 0 ? 41016 : 0;
    cudaStream_t st = (cudaStream_t)stream;
    v41_hc_mixes_kernel<<<rows, 256, 0, st>>>((const v41_bf*)x, fn, nmix, hc * d, norm_eps, mixes);
    v41_hc_sinkhorn_kernel<<<(rows + 63) / 64, 64, 0, st>>>(mixes, scale, base, hc, iters, hc_eps, pre, post, comb,
                                                            rows);
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

extern "C" int memra_v41_hc_pre(const void* x, const float* mix, int rows, int hc, int d, void* y, long ldy,
                                void* stream) {
    if (rows <= 0) return rows < 0 ? 41017 : 0;
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
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
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

extern "C" int memra_v41_hc_post(const void* f, long ldf, const void* res, const float* post, const float* comb,
                                 int rows, int hc, int d, void* out, void* stream) {
    if (hc <= 0 || hc > 4) return 41018;
    if (rows <= 0) return rows < 0 ? 41018 : 0;
    v41_hc_post_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const v41_bf*)f, ldf, (const v41_bf*)res, post, comb,
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
//   1: four e4m3 MMAs from zero, each scaled into its own accumulator as the GEMMs do
//      (acc + (d * sa) * sb): the reference per-32-block schedule's instruction mix
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
#pragma unroll
            for (int j = 0; j < 4; ++j) v41_mma_e4m3(d[j], a, b0, b1);
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
