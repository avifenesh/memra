// Inkling 2 Mini routed experts, the checkpoint's declared numeric program on sm_120a:
//   - per-row NVFP4 activation quantization with the four-over-six MAE scale selection, and MXFP8
//     (e4m3 values, one round-up ue8m0 scale per 32), optionally fused with the softcapped SwiGLU and
//     the routing weight (the down projection's input);
//   - the W4A4 expert GEMM on the block-scaled FP4 tensor-core form (kind::mxf4nvf4, m16n8k64, one
//     ue4m3 scale per 16) and the W8A8 one on kind::mxf8f6f4 (m16n8k32, one ue8m0 scale per 32).
//
// The quantizers are the program of memra_reference::inkling2_actquant operation for operation: one
// IEEE fp32 operation per source operation (the *_rn intrinsics are never contracted into an FMA),
// software e2m1 and e4m3 rounding. They are bit-tested against the committed golden vectors.
//
// GEMM design (ours, for this card): the WEIGHTS sit on the MMA's 16-row A operand and the
// activation rows on its 8-column B operand. A decode step has 1..8 rows per expert, so every MMA is
// fully used on the weight side. The weight bytes are the whole cost at M = 1: each warp streams its
// 16 rows in the stored row-major layout through a private cp.async ring (16-byte copies, several K
// steps in flight) and ldmatrix reads a stage back as the A fragment, with no repack. The group's
// activation rows are staged once into shared memory. NB B tiles share one weight read, so a prefill group of up to 8 * NB rows
// costs one pass over its expert.
//
// The accumulator is one fp32 MMA chain over ascending K steps. An index-order fp32 sum does not
// reproduce it; two measured models do, bit for bit on this card, and the gate holds the kernels to
// them: W4A4 adds each step's exact product sum with one rounding toward zero
// (reference::dot_chain_rz), W8A8 truncates the step's terms to 2^(emax - 25) first
// (reference::dot_chain_trunc).

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// build.rs defines MEMRA_INK2_BLOCK_SCALED=1 for the sm_120a family (120a, 121a). The host pass does not
// see the device feature macros, so the launchers' fail-closed refusal keys on this define.
#if defined(MEMRA_INK2_BLOCK_SCALED) && MEMRA_INK2_BLOCK_SCALED
#define INK2_BLOCK_SCALED_MMA 1
#else
#define INK2_BLOCK_SCALED_MMA 0
#endif

namespace {

constexpr float INK2_FP32_MAX = 3.4028234663852886e38f;

__device__ __forceinline__ float ink2_bf16_load(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// fp32 -> bf16 bits, round to nearest even (finite inputs; overflow rounds to infinity)
__device__ __forceinline__ uint16_t ink2_bf16_bits(float x) {
    uint32_t b = __float_as_uint(x);
    b += 0x7fffu + ((b >> 16) & 1u);
    return static_cast<uint16_t>(b >> 16);
}

__device__ __forceinline__ float ink2_bf16_round(float x) { return ink2_bf16_load(ink2_bf16_bits(x)); }

// e2m1: nearest, ties to the even code, saturating; bit 3 is the fp32 sign bit. NaN of either sign is
// 0x7, what cvt.rn.satfinite.e2m1x2.f32 gives on sm_120.
__device__ __forceinline__ uint32_t ink2_e2m1_code(float x) {
    if (x != x) return 7u;
    const float a = fabsf(x);
    const uint32_t c = 7u - ((a <= 0.25f) + (a < 0.75f) + (a <= 1.25f) + (a < 1.75f) + (a <= 2.5f) + (a < 3.5f) + (a <= 5.0f));
    return c | ((__float_as_uint(x) >> 31) << 3);
}

__device__ __forceinline__ float ink2_e2m1_value(uint32_t code) {
    const uint32_t e = (code >> 1) & 3u, m = code & 1u;
    // e = 0: m / 2; else (2 + m) * 2^(e - 2)
    const float mag = e == 0 ? static_cast<float>(m) * 0.5f
                             : static_cast<float>(2u + m) * __uint_as_float((125u + e) << 23);
    return (code & 8u) ? -mag : mag;
}

// e4m3 magnitude code of a in [0, 448]: nearest, ties to even, subnormals kept; NaN is 0x7f
__device__ __forceinline__ uint32_t ink2_e4m3_code(float a) {
    if (a != a) return 0x7fu;
    if (a < 0x1p-6f) return static_cast<uint32_t>(__float2int_rn(__fmul_rn(a, 512.0f)));
    const uint32_t b = __float_as_uint(a);
    uint32_t code = (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
    const uint32_t rem = b & 0xfffffu;
    code += (rem > 0x80000u) || (rem == 0x80000u && (code & 1u));
    return code < 0x7eu ? code : 0x7eu;
}

__device__ __forceinline__ float ink2_e4m3_value(uint32_t code) {
    const uint32_t e = (code >> 3) & 15u, m = code & 7u;
    return e == 0 ? static_cast<float>(m) * 0x1p-9f : static_cast<float>(8u + m) * __uint_as_float((e + 117u) << 23);
}

// ---------------------------------------------------------------------------------------------
// softcapped SwiGLU times the routing weight, the mathematical value rounded once to bf16. SiTU's
// inputs are bf16, so its two one-input factors are tables of f64 values over every bf16 bit pattern,
// built on the host by memra_reference::inkling2_actquant::situ_tables:
//   TG[g] = cap_g tanh(g / cap_g) sigmoid(g), TU[u] = cap_u tanh(u / cap_u)
// The products (TG * TU) * gamma are f64 (IEEE, two multiplies: nothing to contract), and the one
// rounding to bf16 goes through round-to-odd fp32, so it is exact. Bit-identical to situ_value.
__device__ __forceinline__ float ink2_bf16_of_f64(double x) {
    if (x != x) return __uint_as_float(0x7fc00000u);
    const float r = __double2float_rn(x);
    if (isinf(r)) return r;
    uint32_t b = __float_as_uint(r);
    const double rd = static_cast<double>(r);
    if (rd != x) {
        if (fabs(rd) > fabs(x)) b -= 1;
        b |= 1u;
    }
    return ink2_bf16_round(__uint_as_float(b));
}

__device__ __forceinline__ float ink2_situ(uint16_t gate, uint16_t up, float gamma, const double* __restrict__ TG,
                                           const double* __restrict__ TU) {
    return ink2_bf16_of_f64(__dmul_rn(__dmul_rn(__ldg(TG + gate), __ldg(TU + up)), static_cast<double>(gamma)));
}

// NaN-dropping max of magnitudes (IEEE maxNum, what the source's tl.max emits); the reductions start
// from NaN, so a block or row that is all NaN reduces to NaN.
__device__ __forceinline__ float ink2_amax(float m, float v) { return fmaxf(m, fabsf(v)); }

// ---------------------------------------------------------------------------------------------
// One 16-element NVFP4 block: candidate with block scale encoder factor enc (encoder / 6 or / 4)
struct Ink2Cand {
    uint32_t lo, hi;  // 16 nibbles, element 0 lowest
    uint32_t scale;   // e4m3 byte
    float error;
};

__device__ __forceinline__ Ink2Cand ink2_candidate(const float* x, float block_max, float enc, float decoder,
                                                   float outer) {
    Ink2Cand c;
    c.scale = ink2_e4m3_code(fminf(__fmul_rn(block_max, enc), 448.0f));
    const float scale = ink2_e4m3_value(c.scale);
    const float inverse = fminf(__fdiv_rn(1.0f, __fmul_rn(scale, decoder)), INK2_FP32_MAX);
    uint64_t packed = 0;
    float e[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t code = ink2_e2m1_code(__fmul_rn(x[i], inverse));
        packed |= static_cast<uint64_t>(code) << (4 * i);
        e[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(ink2_e2m1_value(code), scale), outer), x[i]));
    }
    // pairwise tree: (e0 + e1), pairs of pairs, ...
#pragma unroll
    for (int w = 8; w >= 1; w >>= 1) {
#pragma unroll
        for (int i = 0; i < w; ++i) e[i] = __fadd_rn(e[2 * i], e[2 * i + 1]);
    }
    c.error = e[0];
    c.lo = static_cast<uint32_t>(packed);
    c.hi = static_cast<uint32_t>(packed >> 32);
    return c;
}

// One block per row. Phase A puts the row (the bf16 input, or its SiTU) into shared memory as fp32
// and reduces the row amax; phase B quantizes whole blocks, one thread each. Shared memory holds the
// row block by block with one pad word per block (a row of BW = 16 or 32 values takes BW + 1 words),
// so the 32 lanes of a warp, each reading element i of its own block, hit 32 distinct banks.
//   SITU : X is [rows_in][2K] interleaved (gate, up), Gamma the row's routing weight, TG / TU the
//          SiTU factor tables
//   NVFP4: Values [.][K / 2] nibbles, Scales [.][K / 16] e4m3, Amax [.]
//   else : Values [.][K] e4m3, Scales [.][K / 32] ue8m0 exponents
// Rows (nullable) lists the row to process per block; without it block b processes row b.
template <bool SITU, bool NVFP4>
__global__ void ink2_actq_kernel(const uint16_t* __restrict__ X, const float* __restrict__ Gamma,
                                 const uint32_t* __restrict__ Rows, uint8_t* __restrict__ Values,
                                 uint8_t* __restrict__ Scales, float* __restrict__ Amax, uint16_t* __restrict__ Y, int K,
                                 const double* __restrict__ TG, const double* __restrict__ TU) {
    constexpr int BW = NVFP4 ? 16 : 32;
    extern __shared__ float ink2_sx[];
    float* sx = ink2_sx;
    float* swmax = ink2_sx + (K / BW) * (BW + 1);
    const int tid = threadIdx.x, nthr = blockDim.x;
    const long row = Rows ? Rows[blockIdx.x] : blockIdx.x;
    auto at = [](int i) { return (i / BW) * (BW + 1) + i % BW; };
    float local = __uint_as_float(0x7fc00000u);
    if constexpr (SITU) {
        const uint16_t* xr = X + row * 2L * K;
        const float gamma = Gamma[row];
        for (int i = tid; i < K; i += nthr) {
            const uint32_t pair = *reinterpret_cast<const uint32_t*>(xr + 2 * i);
            const float y = ink2_situ(static_cast<uint16_t>(pair), static_cast<uint16_t>(pair >> 16), gamma, TG, TU);
            sx[at(i)] = y;
            if (Y) Y[row * K + i] = static_cast<uint16_t>(__float_as_uint(y) >> 16);
            local = ink2_amax(local, y);
        }
    } else {
        const uint16_t* xr = X + row * (long)K;
        for (int i = tid; i < K; i += nthr) {
            const float v = ink2_bf16_load(xr[i]);
            sx[at(i)] = v;
            local = ink2_amax(local, v);
        }
    }
    float amax = __uint_as_float(0x7fc00000u);
    if constexpr (NVFP4) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, o));
        if ((tid & 31) == 0) swmax[tid >> 5] = local;
    }
    __syncthreads();
    if constexpr (NVFP4) {
        for (int w = 0; w < (nthr + 31) / 32; ++w) amax = fmaxf(amax, swmax[w]);
        if (amax != amax) amax = __uint_as_float(0x7fc00000u);  // an all-NaN row: the quiet NaN
        if (tid == 0) Amax[row] = amax;
        const float encoder = fminf(__fdiv_rn(1536.0f, amax), INK2_FP32_MAX);
        const float decoder = __fdiv_rn(1.0f, encoder);
        const float outer = __fmul_rn(amax, 1.0f / 1536.0f);
        const float enc6 = __fmul_rn(encoder, 1.0f / 6.0f);
        const float enc4 = __fmul_rn(encoder, 0.25f);
        for (int b = tid; b < K / 16; b += nthr) {
            float x[16];
            float block_max = __uint_as_float(0x7fc00000u);
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                x[i] = sx[b * 17 + i];
                block_max = ink2_amax(block_max, x[i]);
            }
            const Ink2Cand c6 = ink2_candidate(x, block_max, enc6, decoder, outer);
            const Ink2Cand c4 = ink2_candidate(x, block_max, enc4, decoder, outer);
            const bool four = c4.error < c6.error;  // strict: a tie keeps the 6-based candidate
            uint32_t* out = reinterpret_cast<uint32_t*>(Values + row * (long)(K / 2) + b * 8);
            out[0] = four ? c4.lo : c6.lo;
            out[1] = four ? c4.hi : c6.hi;
            Scales[row * (long)(K / 16) + b] = static_cast<uint8_t>(four ? c4.scale : c6.scale);
        }
    } else {
        for (int b = tid; b < K / 32; b += nthr) {
            const float* blk = sx + b * 33;
            float bmax = __uint_as_float(0x7fc00000u);
#pragma unroll
            for (int i = 0; i < 32; ++i) bmax = ink2_amax(bmax, blk[i]);
            const float raw = __fmul_rn(bmax, 1.0f / 448.0f);
            int exponent = static_cast<int>(((__float_as_uint(raw) + 0x007fffffu) & 0x7f800000u) >> 23);
            exponent = raw != raw ? 255 : raw <= 0x1p-127f ? 0 : (exponent < 254 ? exponent : 254);
            // exponent 255 (an all-NaN block): NaN scale and a zero inverse, as the source
            const float scale = fmaxf(__uint_as_float(static_cast<uint32_t>(exponent) << 23), 0x1p-127f);
            const float inverse = exponent == 0 ? 0x1p127f : exponent == 255 ? 0.0f : __fdiv_rn(1.0f, scale);
            uint32_t* out = reinterpret_cast<uint32_t*>(Values + row * (long)K + b * 32);
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                uint32_t word = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float v = __fmul_rn(blk[w * 4 + i], inverse);
                    // the clamp propagates NaN (propagate_nan=ALL in the source)
                    const float a = v != v ? v : fminf(fabsf(v), 448.0f);
                    // NaN of either sign is 0x7f, as cvt.rn.satfinite.e4m3x2.f32 on sm_120
                    const uint32_t code = v != v ? 0x7fu : ink2_e4m3_code(a) | ((__float_as_uint(v) >> 31) << 7);
                    word |= code << (8 * i);
                }
                out[w] = word;
            }
            Scales[row * (long)(K / 32) + b] = static_cast<uint8_t>(exponent);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The open fact: what the hardware conversions do. cvt.rn.satfinite.e4m3x2.f32 and
// cvt.rn.satfinite.e2m1x2.f32 of each input, one byte or nibble out per element.
__global__ void ink2_cvt_probe_kernel(const float* __restrict__ x, uint8_t* __restrict__ e4m3, uint8_t* __restrict__ e2m1,
                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
#if INK2_BLOCK_SCALED_MMA
    uint32_t a, b;
    const float v = x[i];
    asm("{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q, %2, %1; cvt.u32.u16 %0, q; }" : "=r"(a) : "f"(v), "f"(v));
    asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %2, %1; cvt.u32.u8 %0, q; }" : "=r"(b) : "f"(v), "f"(v));
    e4m3[i] = static_cast<uint8_t>(a & 0xffu);
    e2m1[i] = static_cast<uint8_t>(b & 0xfu);
#else
    e4m3[i] = 0xff;
    e2m1[i] = 0xff;
#endif
}

// ---------------------------------------------------------------------------------------------
// Expert GEMMs. A group is one expert and up to 8 * NB routed rows:
//   Groups[g] = { expert, count, row[0 .. 8 * NB) } (u32)
// Grid (groups, N-blocks); a block is WPB warps of 16 output columns each.
#if INK2_BLOCK_SCALED_MMA

__device__ __forceinline__ void ink2_mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                             uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,1},{%11},{0,1};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

__device__ __forceinline__ void ink2_mma_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                             uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}

// shared-memory row stride in u32 words: at least words, congruent to 4 mod 32, so the 8 rows by 4
// words one B-fragment load touches fall in 32 distinct banks
__host__ __device__ constexpr int ink2_stride(int words) { return words + ((4 - words % 32) + 32) % 32; }

// FP4: K step 64. Per lane L (g = L / 4, t = L % 4), weights on A:
//   a0 = row g, K[8t .. 8t + 8); a1 = row g + 8, same; a2, a3 = the same rows at K + 32
//   scale A: lane t = 2 row g, t = 3 row g + 8 (4 ue4m3 bytes = the step's 4 blocks of 16)
//   b0 = act column g, K[8t ..); b1 at K + 32; scale B: lane t = 1 column g
// FP8: K step 32, 4 bytes per word:
//   a0 = row g, K[4t .. 4t + 4); a2 at K + 16; scale A: lane t = 0 row g, t = 1 row g + 8 (one byte)
//   b0 = act column g, K[4t ..); b1 at K + 16; scale B: lane t = 0 column g
constexpr int INK2_STAGE = 576;  // one ring stage: 16 rows x 32 B of codes + 16 scale words

__device__ __forceinline__ void ink2_cp16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void ink2_cp4(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void ink2_cp_commit() { asm volatile("cp.async.commit_group;" ::); }
template <int PENDING>
__device__ __forceinline__ void ink2_cp_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(PENDING));
}

// Weight stages each warp keeps in flight. 16 was measured beside 8 at every decode and prefill shape
// and won nothing (research/modelplan-onboarding-inkling2mini-20261121/actquant/README.md).
constexpr int RING = 8;

template <bool FP4, int NB, bool ONE = false>
__global__ void ink2_expert_gemm_kernel(const uint32_t* __restrict__ Act, const uint8_t* __restrict__ ActSc,
                                        const float* __restrict__ ActAmax, const uint32_t* __restrict__ W,
                                        const uint8_t* __restrict__ WSc, const float* __restrict__ WAmax, float denominator,
                                        const uint32_t* __restrict__ Groups, uint16_t* __restrict__ Out, float* __restrict__ Acc, int N, int K,
                                        int xdiv) {
    constexpr int KS = FP4 ? 64 : 32;        // K per MMA
    constexpr int WPS = 8;                    // u32 words of one operand row per step (both halves)
    constexpr int COLS = 8 * NB;
    // ONE: every group holds one routed row (a decode step); only that column is staged, the MMA
    // reads zeros for the other seven, as it does from a staged column past a group's count
    constexpr int SCOLS = ONE ? 1 : COLS;
    // a one-row group keeps twice the weight stages in flight (its block is small enough)
    constexpr int RG = ONE ? 2 * RING : RING;
    const int words = K / KS * WPS;           // words per operand row
    const int steps = K / KS;
    const int scw = FP4 ? steps : (steps + 3) / 4;  // u32 words of scales per row
    const int stride = ink2_stride(words);
    extern __shared__ uint32_t ink2_sm[];
    uint32_t* sact = ink2_sm;
    uint32_t* ssc = ink2_sm + SCOLS * stride;
    const int tid = threadIdx.x, nthr = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const uint32_t* grp = Groups + (long)blockIdx.x * (2 + COLS);
    const uint32_t expert = grp[0], count = grp[1];
    const int n0 = (blockIdx.y * (nthr >> 5) + warp) * 16;
    // The weights are the bytes of this kernel. Each warp keeps RING stages in flight in its own
    // shared-memory ring: a stage is one K step of its 16 weight rows, 16 x 32 B of codes copied as 32
    // 16-byte cp.async (lane 2r + h: row r, half h) plus the rows' scale words. ldmatrix.x4 reads a
    // stage back as the A fragment: matrix 0 = rows 0..7 first half, 1 = rows 8..15 first half, 2 and 3
    // the second halves, which is a0..a3.
    // The ring's first stages are issued before the activation is staged: the weight stream does not
    // wait on it, and the MMA sequence of every output is unchanged.
    const int rowbytes = steps * 32;
    const int scrow = FP4 ? 4 * steps : steps;
    const int head = ((SCOLS * (stride + scw) * 4) + 15) & ~15;
    uint8_t* ring = reinterpret_cast<uint8_t*>(ink2_sm) + head + warp * (RG * INK2_STAGE);
    const uint8_t* wrow = reinterpret_cast<const uint8_t*>(W) + ((long)expert * N + n0) * rowbytes;
    const uint8_t* wsrow = WSc + ((long)expert * N + n0) * scrow;
    auto issue = [&](int s) {
        uint8_t* stg = ring + (s % RG) * INK2_STAGE;
        const int r = lane >> 1, h = lane & 1;
        ink2_cp16(stg + r * 32 + h * 16, wrow + (long)r * rowbytes + s * 32 + h * 16);
        // FP4: the step's 4 ue4m3 bytes; FP8: the 4 ue8m0 bytes of steps s & ~3 .. + 3
        if (lane < 16) ink2_cp4(stg + 512 + lane * 4, wsrow + (long)lane * scrow + (FP4 ? s * 4 : (s & ~3)));
    };
    if (n0 < N) {
        for (int s = 0; s < RG - 1; ++s) {
            if (s < steps) issue(s);
            ink2_cp_commit();
        }
    }
    for (int i = tid; i < SCOLS * words; i += nthr) {
        const int j = i / words, w = i - j * words;
        sact[j * stride + w] = (uint32_t)j < count ? Act[(long)(grp[2 + j] / xdiv) * words + w] : 0u;
    }
    for (int i = tid; i < SCOLS * scw; i += nthr) {
        const int j = i / scw, w = i - j * scw;
        uint32_t v = 0;
        if ((uint32_t)j < count) {
            if constexpr (FP4) {
                v = reinterpret_cast<const uint32_t*>(ActSc)[(long)(grp[2 + j] / xdiv) * scw + w];
            } else {
                const uint8_t* p = ActSc + (long)(grp[2 + j] / xdiv) * steps + 4 * w;
                for (int q = 0; q < 4 && 4 * w + q < steps; ++q) v |= static_cast<uint32_t>(p[q]) << (8 * q);
            }
        }
        ssc[j * scw + w] = v;
    }
    __syncthreads();
    if (n0 >= N) return;
    const int g = lane >> 2, t = lane & 3;
    float acc[NB][4];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.0f;
    const int li = lane >> 3, lr = lane & 7;
    const int ldoff = ((li & 1) * 8 + lr) * 32 + (li >> 1) * 16;
    // scale source of this lane: FP4 lanes t = 2, 3; FP8 lanes t = 0, 1 (row g, row g + 8)
    const int st = FP4 ? t - 2 : t;
    const bool has_sa = st == 0 || st == 1;
    const int sr = g + 8 * (st == 1);
    const bool has_sb = FP4 ? t == 1 : t == 0;
    for (int s = 0; s < steps; ++s) {
        ink2_cp_wait<RG - 2>();
        __syncwarp();  // stage s landed for every lane; stage s - 1's readers are done
        if (s + RG - 1 < steps) issue(s + RG - 1);
        ink2_cp_commit();
        const uint8_t* stg = ring + (s % RG) * INK2_STAGE;
        uint32_t a[4];
        {
            const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(stg + ldoff));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(addr));
        }
        uint32_t sa = 0;
        if (has_sa) {
            const uint32_t word = *reinterpret_cast<const uint32_t*>(stg + 512 + sr * 4);
            sa = FP4 ? word : (word >> (8 * (s & 3))) & 0xffu;
        }
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int col = 8 * b + g;
            const bool live = !ONE || col == 0;
            const uint32_t b0 = live ? sact[col * stride + s * WPS + t] : 0u;
            const uint32_t b1 = live ? sact[col * stride + s * WPS + 4 + t] : 0u;
            uint32_t sb = 0;
            if (has_sb && live) {
                if constexpr (FP4) {
                    sb = ssc[col * scw + s];
                } else {
                    sb = (ssc[col * scw + (s >> 2)] >> (8 * (s & 3))) & 0xffu;
                }
            }
            if constexpr (FP4) {
                ink2_mma_fp4(acc[b], a, b0, b1, sa, sb);
            } else {
                ink2_mma_fp8(acc[b], a, b0, b1, sa, sb);
            }
        }
    }
    ink2_cp_wait<0>();
    // d0 = (row g, col 2t), d1 = (row g, col 2t + 1), d2, d3 = row g + 8
    float rcp = 1.0f, wamax = 1.0f;
    if constexpr (FP4) {
        rcp = __fdiv_rn(1.0f, __fmul_rn(1536.0f, denominator));
        wamax = WAmax[expert];
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const uint32_t col = 8 * b + 2 * t + c;
            if (col < count) {
                const uint32_t row = grp[2 + col];
                float v0 = acc[b][c], v1 = acc[b][2 + c];
                if (Acc) {  // the raw accumulators, for the gates
                    Acc[(long)row * N + n0 + g] = v0;
                    Acc[(long)row * N + n0 + g + 8] = v1;
                }
                if constexpr (FP4) {
                    const float rs = __fmul_rn(__fmul_rn(ActAmax[row / xdiv], wamax), rcp);
                    v0 = __fmul_rn(v0, rs);
                    v1 = __fmul_rn(v1, rs);
                }
                uint16_t* o = Out + (long)row * N + n0 + g;
                o[0] = ink2_bf16_bits(v0);
                o[8] = ink2_bf16_bits(v1);
            }
        }
    }
}
#endif  // INK2_BLOCK_SCALED_MMA

template <typename Kern>
int ink2_launch(Kern kernel, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream) {
    if (smem > 48 * 1024) {
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (e != cudaSuccess) return 20000 + static_cast<int>(e);
    }
    const cudaError_t e = cudaLaunchKernel(reinterpret_cast<const void*>(kernel), grid, block, args, smem, stream);
    return e == cudaSuccess ? 0 : 10000 + static_cast<int>(e);
}

}  // namespace

extern "C" {

// 1 when this build carries the block-scaled expert GEMMs and the hardware conversion probe.
int memra_ink2_actq_built() { return INK2_BLOCK_SCALED_MMA; }

// Quantize rows. kind 0 = NVFP4 (K % 16 == 0), 1 = MXFP8 (K % 32 == 0). situ != 0: X is
// [.][2K] interleaved bf16 (gate, up), gamma the per-row routing weight, tg / tu the SiTU factor tables
// (65,536 f64 each, memra_reference situ_tables), y (nullable) receives the bf16 SiTU rows. rows
// (nullable) lists the row of each of the nrows blocks. Asynchronous on stream. Returns 0, a refusal
// (1..9) or 10000 + cudaError.
int memra_ink2_actq(int kind, int situ, const void* x, const float* gamma, const uint32_t* rows, int nrows, void* values,
                    void* scales, float* amax, void* y, int K, const double* tg, const double* tu, cudaStream_t stream) {
    if (!x || !values || !scales || nrows <= 0 || K <= 0) return 1;
    if (kind != 0 && kind != 1) return 2;
    if (K % (kind == 0 ? 16 : 32) != 0) return 3;
    if (kind == 0 && !amax) return 4;
    if (situ && (!gamma || !tg || !tu)) return 5;
    const uint16_t* xp = static_cast<const uint16_t*>(x);
    uint8_t* vp = static_cast<uint8_t*>(values);
    uint8_t* sp = static_cast<uint8_t*>(scales);
    uint16_t* yp = static_cast<uint16_t*>(y);
    void* args[] = {&xp, &gamma, &rows, &vp, &sp, &amax, &yp, &K, &tg, &tu};
    const dim3 grid(nrows), block(256);
    const int bw = kind == 0 ? 16 : 32;
    const size_t smem = (static_cast<size_t>(K / bw) * (bw + 1) + 8) * sizeof(float);
    if (kind == 0) {
        return situ ? ink2_launch(ink2_actq_kernel<true, true>, grid, block, args, smem, stream)
                    : ink2_launch(ink2_actq_kernel<false, true>, grid, block, args, smem, stream);
    }
    return situ ? ink2_launch(ink2_actq_kernel<true, false>, grid, block, args, smem, stream)
                : ink2_launch(ink2_actq_kernel<false, false>, grid, block, args, smem, stream);
}

// The hardware fp32 -> e4m3 and fp32 -> e2m1 conversions of n values (one byte each; 0xff where
// this build has no such instruction).
int memra_ink2_cvt_probe(const float* x, uint8_t* e4m3, uint8_t* e2m1, int n, cudaStream_t stream) {
    if (!x || !e4m3 || !e2m1 || n <= 0) return 1;
    void* args[] = {&x, &e4m3, &e2m1, &n};
    return ink2_launch(ink2_cvt_probe_kernel, dim3((n + 255) / 256), dim3(256), args, 0, stream);
}

// Bytes of one group record for nb B tiles, and the largest nb a kind takes.
int memra_ink2_group_words(int nb) { return 2 + 8 * nb; }

// The grouped expert GEMM. kind 0 = W4A4 (act codes [R][K / 2], act scales [R][K / 16] e4m3, act
// amax [R]; weights [E][N][K / 2], [E][N][K / 16], amax [E], denominator), 1 = W8A8 (act [R][K] e4m3,
// [R][K / 32] ue8m0; weights [E][N][K], [E][N][K / 32]). nb B tiles per group (W4A4: 1, 2 or 4; W8A8:
// 1 or 2); groups is ngroups records of memra_ink2_group_words(nb) u32. out is bf16 [R][N], written
// for the grouped rows only; acc (nullable) receives the fp32 accumulators [R][N] before the row scale.
// wpb warps per block (0 = default: 2 for a decode group, 8 with several B tiles). N % 16 == 0;
// K % 64 == 0 (W4A4) or K % 128 == 0 (W8A8).
int memra_ink2_expert_gemm_x(int kind, int nb, const void* act, const void* act_scales, const float* act_amax,
                             const void* w, const void* w_scales, const float* w_amax, float denominator,
                             const uint32_t* groups, int ngroups, int xdiv, void* out, float* acc, int N, int K,
                             int wpb, cudaStream_t stream);
int memra_ink2_expert_gemm(int kind, int nb, const void* act, const void* act_scales, const float* act_amax,
                           const void* w, const void* w_scales, const float* w_amax, float denominator,
                           const uint32_t* groups, int ngroups, void* out, float* acc, int N, int K, int wpb,
                           cudaStream_t stream) {
    return memra_ink2_expert_gemm_x(kind, nb, act, act_scales, act_amax, w, w_scales, w_amax, denominator, groups,
                                    ngroups, 1, out, acc, N, K, wpb, stream);
}

// The same GEMM with separate input and output rows: a group's row r reads act row r / xdiv and
// writes out row r (xdiv = top_k: the routed pairs of a token read the token's one quantized row).
int memra_ink2_expert_gemm_x(int kind, int nb, const void* act, const void* act_scales, const float* act_amax,
                             const void* w, const void* w_scales, const float* w_amax, float denominator,
                             const uint32_t* groups, int ngroups, int xdiv, void* out, float* acc, int N, int K,
                             int wpb, cudaStream_t stream) {
#if INK2_BLOCK_SCALED_MMA
    if (xdiv <= 0) return 7;
    if (!act || !act_scales || !w || !w_scales || !groups || !out || ngroups <= 0) return 1;
    if (kind != 0 && kind != 1) return 2;
    if (N <= 0 || N % 16 != 0 || K <= 0 || K % (kind == 0 ? 64 : 128) != 0) return 3;
    if (kind == 0 && (!act_amax || !w_amax)) return 4;
    // nb 0: one-row groups of the nb 1 layout (a decode step), only the live column staged
    const bool one = nb == 0;
    if (one) nb = 1;
    if (!(nb == 1 || nb == 2 || (nb == 4 && kind == 0))) return 5;
    if (wpb <= 0) wpb = nb == 1 ? 1 : 8;
    if (wpb > 8) return 6;
    const uint32_t* ap = static_cast<const uint32_t*>(act);
    const uint8_t* asp = static_cast<const uint8_t*>(act_scales);
    const uint32_t* wp = static_cast<const uint32_t*>(w);
    const uint8_t* wsp = static_cast<const uint8_t*>(w_scales);
    uint16_t* op = static_cast<uint16_t*>(out);
    void* args[] = {&ap, &asp, &act_amax, &wp, &wsp, &w_amax, &denominator, &groups, &op, &acc, &N, &K, &xdiv};
    const int ks = kind == 0 ? 64 : 32;
    const int words = K / ks * 8, steps = K / ks;
    const int scw = kind == 0 ? steps : (steps + 3) / 4;
    const size_t head = (static_cast<size_t>(one ? 1 : 8 * nb) * (ink2_stride(words) + scw) * sizeof(uint32_t) + 15) & ~size_t(15);
    const size_t smem = head + static_cast<size_t>(wpb) * (one ? 2 * RING : RING) * INK2_STAGE;
    // groups on grid.x (up to 2^31 - 1), the N blocks of one group on grid.y (N / 16 / wpb, at most 65,535)
    const int nblocks = (N / 16 + wpb - 1) / wpb;
    if (nblocks > 65535) return 7;
    const dim3 grid(ngroups, nblocks), block(32 * wpb);
    if (kind == 0) {
        if (one) return ink2_launch(ink2_expert_gemm_kernel<true, 1, true>, grid, block, args, smem, stream);
        if (nb == 1) return ink2_launch(ink2_expert_gemm_kernel<true, 1>, grid, block, args, smem, stream);
        if (nb == 2) return ink2_launch(ink2_expert_gemm_kernel<true, 2>, grid, block, args, smem, stream);
        return ink2_launch(ink2_expert_gemm_kernel<true, 4>, grid, block, args, smem, stream);
    }
    if (one) return ink2_launch(ink2_expert_gemm_kernel<false, 1, true>, grid, block, args, smem, stream);
    if (nb == 1) return ink2_launch(ink2_expert_gemm_kernel<false, 1>, grid, block, args, smem, stream);
    return ink2_launch(ink2_expert_gemm_kernel<false, 2>, grid, block, args, smem, stream);
#else
    (void)kind; (void)nb; (void)act; (void)act_scales; (void)act_amax; (void)w; (void)w_scales; (void)w_amax;
    (void)denominator; (void)groups; (void)ngroups; (void)xdiv; (void)out; (void)acc; (void)N; (void)K; (void)wpb; (void)stream;
    return 9;  // fail closed: no block-scaled MMA on this target
#endif
}

}  // extern "C"
