// moe_f16_grouped.cu — per-layer expert dequant to f16 + ONE grouped f16 GEMM per projection
// over the CSR expert->token groups (round 46 arc 2, the "vLLM shape" lever from round 44).
//
// WHY: the expert-segmented MMQ kernel re-dequants the same expert weights per
// (out-tile x token-tile x k-block) — at ~65-145 tokens/expert that is up to 12x redundant
// dequant, 13x scalar instructions per mma, and ~50% tile-padding waste (ncu round 46).
// Here the active experts dequant ONCE per (layer, projection) into an f16 workspace and
// cublasGemmGroupedBatchedEx runs every expert's [m_e x out_f x in_f] GEMM in one call at
// tensor-core f16 rate.
//
// NUMERICS: the f16-mirror class (campaign A / battery-adopted): dequanted weight values
// round to f16, f32 accumulate. NOT byte-identical to the MMQ path — argmax + spec gates
// arbitrate, per-model, like every numeric-class promotion. Experimental door
// MEMRA_MOE_F16G=1 until gated.
//
// Column-major mapping (derivation in the round-46 ledger): per group g,
//   C(m=out_f, n=m_e, k=in_f), opA=T opB=N,
//   A = Wf16[g]   (row-major [out_f][in_f], lda=in_f),
//   B = actf16 + pair_off[g]*in_f (pair-major row-major, ldb=in_f),
//   C = y + pair_off[g]*out_f     (f32 pair-major row-major, ldc=out_f).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

#define QT_IQ4_XS 5
#define QT_IQ3_S  6
#define QT_Q4_0   12
#define QT_Q4_K   1
#define QT_Q6_K   2
#define QT_Q3_K   4
#define QT_NVFP4  7
#define QT_NVFP4_V2 107  // slot-major v2 bank permutation of QT_NVFP4 (tp.rs nvfp4_matrix_v2_permute)
#define QT_NVFP4_MODELOPT 108  // DSV4: consecutive packed codes + separate linear E4M3/16 plane

// Host-side engagement receipts for the opt-in packed-store launchers below.  These count an
// enqueue that passed cudaGetLastError, not a speculative gate decision.  The Rust public getters
// expose them to the caller without changing the default kernels or their shared-memory footprint.
static std::atomic<unsigned long long> g_moe_kq_gu_h2_dispatches{0};
static std::atomic<unsigned long long> g_moe_kq_m1_h2_dispatches{0};

__device__ __forceinline__ float g_half_to_float(uint16_t h){ return __half2float(*reinterpret_cast<const __half*>(&h)); }
__constant__ signed char g_kvalues_iq4nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// e2m1 values DOUBLED (the NVFP4 0.5 convention — the UE4M3 scale decode below multiplies by 0.5).
__constant__ signed char g_kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

// UE4M3 (unsigned 4-exp/3-mant, bias 7) -> f32, returns value*0.5 (DOUBLED-table convention).
// Per-value port of the memra-gguf CPU oracle (dequant.rs ue4m3_to_f32); codes 0 / 0x7F -> 0.0.
__device__ __forceinline__ float g_ue4m3_to_float(uint8_t x){
    if(x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? man * exp2f(-9.0f) : (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
    return raw * 0.5f;
}

// ModelOpt stores the micro-scale in signed E4M3FN. DSV4 scale tensors are
// non-negative in the pinned checkpoint, but decoding the sign bit here keeps
// the byte contract complete instead of silently treating it as exponent data.
// Return value*0.5 to pair with the doubled E2M1 codebook above.
__device__ __forceinline__ float g_e4m3fn_to_float(uint8_t x){
    unsigned mag = x & 0x7Fu;
    if(mag == 0x7Fu) return 0.0f;
    unsigned exp = mag >> 3, man = mag & 7u;
    float v = exp ? __uint_as_float(((exp + 120u) << 23) | (man << 20))
                  : (float)man * 0x1p-9f;
    if(x & 0x80u) v = -v;
    return v * 0.5f;
}

// iq3s codebook — verbatim copy of iq3s_grid_const (cu/mmq_iq_experts.cu, itself verbatim from
// ggml-common.h). Duplicated rather than shared via header so the default MMQ TU stays untouched.
static __device__ __constant__ unsigned int g_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// ---- dequant: one expert row slice -> f16. grid.x = out-row, grid.y = active-expert seg.
// 256 threads walk the row's k. Layout out: dst[seg][row][k] row-major (lda = in_f).
static __global__ void dequant_q4_0_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // q4_0: 18B block = f16 d + 16B nibbles for 32 values (lo nibbles vals 0..15, hi 16..31).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int blk=v>>5, l=v&31;
        const uint8_t* b = r + (size_t)blk*18;
        float dscale = g_half_to_float(*(const uint16_t*)b);
        uint8_t q = b[2 + (l&15)];
        int nib = (l<16) ? (q & 0xF) : (q >> 4);
        d[v] = __float2half(dscale * (float)(nib - 8));
    }
}

static __global__ void dequant_iq4xs_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    // iq4_xs: 136B superblock = f16 d, u16 scales_h, 4B scales_l, 128B nibbles (256 vals).
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*136;
        float d_sb = g_half_to_float(*(const uint16_t*)b);
        uint16_t sh = *(const uint16_t*)(b+2);
        const uint8_t* sl = b+4; const uint8_t* qs = b+8;
        int g = l>>5, lg = l&31;
        int ls = ((sl[g>>1]>>(4*(g&1)))&0xf) | (((sh>>(2*g))&3)<<4);
        const uint8_t* gqs = qs + g*16;
        int code = (lg<16) ? (gqs[lg]&0xf) : (gqs[lg-16]>>4);
        d[v] = __float2half(d_sb * (float)(ls-32) * (float)g_kvalues_iq4nl[code]);
    }
}

// ---- round-49 dequant coverage: q35's UD-IQ4_XS bank is a MIX (gate/up IQ3_S x39 + Q3_K x1 +
// IQ4_XS x1; down IQ4_XS x37 + Q6_K x3 + Q4_K x1). The round-47 IQ4_XS/Q4_0-only table admitted
// ~1 of 41 q35 layers — the actual reason the q35 f16g cell measured FLAT. Each kernel is a
// per-value port of the memra-gguf CPU dequant oracle (crates/memra-gguf/src/dequant.rs, itself
// diffed against ggml row dequant on real tensors).

// iq3_s: 110B superblock = f16 d, qs[64], qh[8], signs[32], scales[4]. Grid-codebook + per-byte
// signs; scale = d*(1+2*nibble) per 32-group.
static __global__ void dequant_iq3s_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*110;
        float dd = g_half_to_float(*(const uint16_t*)b);
        const uint8_t* qs = b+2; const uint8_t* qh = b+66;
        const uint8_t* sg = b+74; const uint8_t* sc = b+106;
        int ib32 = l>>5, lg = l&31, l4 = lg>>3, j = lg&7;
        float db = dd * (1.0f + 2.0f*(float)((sc[ib32>>1] >> (4*(ib32&1))) & 0xf));
        int qsb = qs[ib32*8 + 2*l4 + (j>>2)];
        int hshift = (j<4) ? (8-2*l4) : (7-2*l4);
        int idx = qsb | (((int)qh[ib32] << hshift) & 256);
        int gb = (g_iq3s_grid[idx] >> (8*(j&3))) & 0xff;
        float sgn = (sg[ib32*4 + l4] & (1<<j)) ? -1.0f : 1.0f;
        d[v] = __float2half(db * (float)gb * sgn);
    }
}

// ---- shared per-value Q4_K/Q6_K dequant (lane/kquant-tile-loaders): the EXACT per-value
// expressions of the q4k/q6k dequant kernels, factored into __forceinline__ helpers so the
// DIRECT-FROM-QUANT sk tile loaders (below the sk kernels) compute bit-identical f16 values
// to the workspace path by construction — gated bitwise in kernel-check ("f16g-kq-direct"),
// synthetic + real weights.

// q6_K value: b = 210B superblock (ql[128], qh[64], i8 scales[16], f16 d at the END),
// l255 = value index 0..255 within the superblock.
__device__ __forceinline__ __half kq_q6k_val(const uint8_t* __restrict__ b, int l255){
    const uint8_t* ql = b; const uint8_t* qh = b+128;
    const int8_t* sc = (const int8_t*)(b+192);
    float dd = g_half_to_float(*(const uint16_t*)(b+208));
    int n2 = l255>>7, rr = l255&127, q4 = rr>>5, l = rr&31, is = l>>4;
    int qlb = ql[n2*64 + l + (q4&1)*32];
    int nib = (q4<2) ? (qlb & 0xF) : (qlb >> 4);
    int qv = (nib | (((qh[n2*32 + l] >> (2*q4)) & 3) << 4)) - 32;
    return __float2half(dd * (float)sc[n2*8 + is + 2*q4] * (float)qv);
}

// q4_K value: b = 144B superblock (f16 d, f16 dmin, scales[12] 6-bit packed, qs[128]),
// l = value index 0..255 within the superblock.
__device__ __forceinline__ __half kq_q4k_val(const uint8_t* __restrict__ b, int l){
    float dd = g_half_to_float(*(const uint16_t*)b);
    float dmin = g_half_to_float(*(const uint16_t*)(b+2));
    const uint8_t* scs = b+4; const uint8_t* q = b+16;
    int g64 = l>>6, w64 = l&63, is = g64*2 + (w64>>5);
    int sc8, m8;
    if(is < 4){ sc8 = scs[is] & 63;                       m8 = scs[is+4] & 63; }
    else      { sc8 = (scs[is+4] & 0xF) | ((scs[is-4] >> 6) << 4);
                m8  = (scs[is+4] >> 4)  | ((scs[is]   >> 6) << 4); }
    int qb = q[g64*32 + (w64&31)];
    int nib = (w64<32) ? (qb & 0xF) : (qb >> 4);
    return __float2half(dd * (float)sc8 * (float)nib - dmin * (float)m8);
}

// q6_K: 210B superblock = ql[128], qh[64], i8 scales[16], f16 d (d at the END).
static __global__ void dequant_q6k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x)
        d[v] = kq_q6k_val(r + (size_t)(v>>8)*210, v&255);
}

// q4_K: 144B superblock = f16 d, f16 dmin, scales[12] (6-bit packed), qs[128].
static __global__ void dequant_q4k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x)
        d[v] = kq_q4k_val(r + (size_t)(v>>8)*144, v&255);
}

// nvfp4: 36B block = u8 d[4] (UE4M3 sub-scales) + qs[32] nibbles, 64 values (4 sub-blocks of 16;
// value j<8 = lo nibble of qs[sub*8+j], j>=8 = hi nibble of qs[sub*8+j-8]). Per-value port of the
// memra-gguf CPU oracle (dequant.rs dequant_nvfp4) — the ornith15 expert bank class.
static __global__ void dequant_nvfp4_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int blk = v>>6, l = v&63, sub = l>>4, j = l&15;
        const uint8_t* b = r + (size_t)blk*36;
        float dscale = g_ue4m3_to_float(b[sub]);
        uint8_t q = b[4 + sub*8 + (j&7)];
        int code = (j<8) ? (q & 0xF) : (q >> 4);
        d[v] = __float2half(dscale * (float)g_kvalues_mxfp4[code]);
    }
}

// NVFP4 v2 bank layout (tp.rs nvfp4_matrix_v2_permute): per row, slot g (32 values) stores its
// 16 qs bytes at g*16, and the per-16-value UE4M3 scale bytes live in a tail at n_slots*16 + g*2.
// Value-identical to the 36B interleaved form — only the byte permutation differs. Feeding v2
// bytes to the v1 kernel above was the gemm-prime garbage-output bug (2026-08-27).
static __global__ void dequant_nvfp4v2_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    const int n_slots = in_f >> 5;
    const uint8_t* qs0 = r;
    const uint8_t* sc0 = r + (size_t)n_slots*16;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int g = v>>5, l = v&31, sub = l>>4, j = l&15;
        float dscale = g_ue4m3_to_float(sc0[g*2 + sub]);
        uint8_t q = qs0[(size_t)g*16 + sub*8 + (j&7)];
        int code = (j<8) ? (q & 0xF) : (q >> 4);
        d[v] = __float2half(dscale * (float)g_kvalues_mxfp4[code]);
    }
}

// q3_K: 110B superblock = hmask[32], qs[64], scales[12] (6-bit aux dance), f16 d.
static __global__ void dequant_q3k_f16_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, __half* __restrict__ dst,
        int in_f, int out_f, long row_bytes){
    int row = blockIdx.x; int seg = blockIdx.y;
    const uint8_t* W = (const uint8_t*)table[(size_t)proj*n_expert + ex_ids[seg]];
    const uint8_t* r = W + (size_t)row*row_bytes;
    __half* d = dst + ((size_t)seg*out_f + row)*in_f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x){
        int sb=v>>8, l=v&255;
        const uint8_t* b = r + (size_t)sb*110;
        const uint8_t* hm = b; const uint8_t* qs = b+32; const uint8_t* scb = b+96;
        float dd = g_half_to_float(*(const uint16_t*)(b+108));
        int nn = l>>7, rr = l&127, jj = rr>>5, l16 = rr&31;
        int is = nn*8 + jj*2 + (l16>>4);
        // 6-bit scale unpack (ggml aux-word dance; scb byte-loaded — 110B stride is unaligned)
        uint32_t aux0 = scb[0] | (scb[1]<<8) | (scb[2]<<16) | ((uint32_t)scb[3]<<24);
        uint32_t aux1 = scb[4] | (scb[5]<<8) | (scb[6]<<16) | ((uint32_t)scb[7]<<24);
        uint32_t aux2 = scb[8] | (scb[9]<<8) | (scb[10]<<16) | ((uint32_t)scb[11]<<24);
        const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        uint32_t w;
        switch(is>>2){
            case 0:  w = (aux0 & km2)        | (((aux2 >> 0) & km1) << 4); break;
            case 1:  w = (aux1 & km2)        | (((aux2 >> 2) & km1) << 4); break;
            case 2:  w = ((aux0 >> 4) & km2) | (((aux2 >> 4) & km1) << 4); break;
            default: w = ((aux1 >> 4) & km2) | (((aux2 >> 6) & km1) << 4); break;
        }
        int s6 = (int)((w >> (8*(is&3))) & 0xff) - 32;
        int q2 = (qs[nn*32 + l16] >> (2*jj)) & 3;
        int hb = (hm[l16] & (1 << (nn*4 + jj))) ? 0 : 4;
        d[v] = __float2half(dd * (float)s6 * (float)(q2 - hb));
    }
}

// ---- f16 -> f32 elementwise (the grouped GEMM emits f16 C: the grouped API's type
// matrix has no 16F-in/32F-out combo — rc 20015 on H100 cublas 13).
static __global__ void h2f_kernel(const __half* __restrict__ s, float* __restrict__ d, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) d[i] = __half2float(s[i]);
}
// f16 -> f32 with the per-row activation scale folded back (row = pair, C row-major).
static __global__ void h2f_rows_scale_kernel(const __half* __restrict__ src, float* __restrict__ d,
                                             const float* __restrict__ s, int ncols, int nrows){
    int r = blockIdx.x; if(r>=nrows) return;
    float sc = s[r];
    const __half* sr = src + (size_t)r*ncols;
    float* dr = d + (size_t)r*ncols;
    for(int c=threadIdx.x;c<ncols;c+=blockDim.x) dr[c] = __half2float(sr[c]) * sc;
}

// ---- activation gather+convert: f32 x[token][in_f] -> f16 B[pair][in_f] via pair_tok,
// with PER-ROW amax normalization (gemma's late-layer activation spikes overflow raw f16
// — the round-46 NaN find; the MMQ path survives on per-32 q8 scales). The row scale
// folds back into the GEMM output (y row *= s[pair]). pair_tok == nullptr => identity.
static __global__ void gather_act_f16_kernel(
        const float* __restrict__ x, const int* __restrict__ pair_tok,
        __half* __restrict__ dst, float* __restrict__ s, int in_f, int n_pairs){
    int p = blockIdx.x; if(p>=n_pairs) return;
    int src = pair_tok ? pair_tok[p] : p;
    const float* xs = x + (size_t)src*in_f;
    __half* d = dst + (size_t)p*in_f;
    __shared__ float red[256];
    float amax = 0.0f;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) amax = fmaxf(amax, fabsf(xs[v]));
    red[threadIdx.x] = amax; __syncthreads();
    for(int off=128; off>0; off>>=1){
        if(threadIdx.x<off) red[threadIdx.x]=fmaxf(red[threadIdx.x],red[threadIdx.x+off]);
        __syncthreads();
    }
    amax = red[0];
    float inv = (amax>0.0f) ? 1.0f/amax : 0.0f;
    if(threadIdx.x==0) s[p] = amax;
    for(int v=threadIdx.x; v<in_f; v+=blockDim.x) d[v] = __float2half(xs[v]*inv);
}

// ================= single-kernel grouped GEMM (MEMRA_MOE_F16G=2, round 49) =================
//
// cublasGemmGroupedBatchedEx issues through cublas-INTERNAL streams that are not ordered with
// the engine stream (round 47: deterministic NaN race; v1 pays a full stream sync per
// projection — the tax that capped the door at +4-11% g26 / flat q35). This kernel is the
// structural fix: ONE launch on OUR stream runs every CSR group's [m_e x out_f x in_f] GEMM —
// ordered by construction, zero syncs — and C emits f32 DIRECTLY with the per-row act amax
// scale folded in (kills the f16-C + h2f pass the grouped API's type matrix forced).
//
// Form: plain tiled m16n8k16 f16 mma, f32 accumulate (same numeric class as the cublas
// COMPUTE_32F arm). grid.z = CSR group, grid.y = m-tile inside the group (grid is sized for
// the LARGEST group; CTAs past a group's pair count exit in ~2 loads), grid.x = out-tile.
// Dispatch order (x fastest, z slowest) keeps one group's tiles adjacent so the W-tile
// re-reads across its m-tiles hit L2. BM=32 BN=64 BK=32, 4 warps (2x2), cp.async
// double-buffered smem, ldmatrix operand loads. All sm_80-class — portable to every memra
// arch (89/90a/100a/120a), unlike the sm_120a-only CUTLASS static lib.

#define SK_BM 32
#define SK_BN 64
#define SK_BK 32
#define SK_STRIDE (SK_BK + 8)   // +8 halves de-banks ldmatrix rows; keeps 16B alignment

__device__ __forceinline__ void sk_cp16(void* smem, const void* g){
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
// ldmatrix x4: a 16x16 half tile from k-contiguous smem rows. Register i = the 8x8 submatrix
// (rows i&1 ? 8-15 : 0-7, k i&2 ? 8-15 : 0-7) — exactly the m16n8k16 A-operand register order;
// the same load on the [n][k] W tile yields the B operand as n-blocks {r0,r2} (n 0-7) and
// {r1,r3} (n 8-15) (both operands are k-contiguous, so no .trans needed).
__device__ __forceinline__ void sk_ldm16x16(unsigned (&r)[4], const __half* base, int stride){
    const __half* p = base + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
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

static __global__ void moe_f16g_sk_kernel(
        const __half* __restrict__ W,        // [n_active][out_f][in_f] dequant workspace
        const __half* __restrict__ A,        // [n_pairs][in_f] act, CSR pair-major
        float* __restrict__ Y,               // [n_pairs][out_f] f32 out, CSR pair-major
        const float* __restrict__ row_scale, // [n_pairs] act amax (folds back here)
        const int* __restrict__ ex_off,      // [n_active+1] CSR group offsets (device)
        int in_f, int out_f){
    const int g  = blockIdx.z;
    const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
    const int m0 = blockIdx.y * SK_BM;
    if(m0 >= m_e) return;
    const int n0 = blockIdx.x * SK_BN;

    __shared__ __half As[2][SK_BM][SK_STRIDE];
    __shared__ __half Bs[2][SK_BN][SK_STRIDE];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const __half* Ag = A + (size_t)lo * in_f;
    const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;

    // Stage loads: A tile 32 rows x 64B = 1 cp.async16/thread; B tile 64 rows x 64B = 2/thread.
    // Rows past the group's pairs / past out_f clamp to the last valid row — real bytes load,
    // the store guards discard the results.
    const int ar = tid >> 2,  ac  = (tid & 3) * 8;
    const int am = min(m0 + ar, m_e - 1);
    const int br = tid >> 1,  bc  = (tid & 1) * 16;
    const int bn = min(n0 + br, out_f - 1);
    const __half* aga = Ag + (size_t)am * in_f + ac;
    const __half* bga = Wg + (size_t)bn * in_f + bc;

    const int nkb = in_f / SK_BK;
    // prologue: stage 0
    sk_cp16(&As[0][ar][ac],    aga);
    sk_cp16(&Bs[0][br][bc],     bga);
    sk_cp16(&Bs[0][br][bc + 8], bga + 8);
    asm volatile("cp.async.commit_group;");

    float acc[4][4] = {};
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;

    for(int kb = 0; kb < nkb; kb++){
        const int cur = kb & 1;
        if(kb + 1 < nkb){
            const int nxt = cur ^ 1, k0 = (kb + 1) * SK_BK;
            sk_cp16(&As[nxt][ar][ac],    aga + k0);
            sk_cp16(&Bs[nxt][br][bc],     bga + k0);
            sk_cp16(&Bs[nxt][br][bc + 8], bga + k0 + 8);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        #pragma unroll
        for(int kk = 0; kk < 2; kk++){
            unsigned a[4], b0[4], b1[4];
            sk_ldm16x16(a,  &As[cur][wm][kk*16],      SK_STRIDE);
            sk_ldm16x16(b0, &Bs[cur][wn][kk*16],      SK_STRIDE);
            sk_ldm16x16(b1, &Bs[cur][wn + 16][kk*16], SK_STRIDE);
            sk_mma(acc[0], a, b0[0], b0[2]);
            sk_mma(acc[1], a, b0[1], b0[3]);
            sk_mma(acc[2], a, b1[0], b1[2]);
            sk_mma(acc[3], a, b1[1], b1[3]);
        }
        __syncthreads();
    }

    // epilogue: f32 store with the per-pair act scale folded back (mode 1 pays a separate
    // h2f_rows_scale pass for this). C frag: c0/c1 = (row lane/4, col 2*(lane%4)+{0,1});
    // c2/c3 = row+8. acc[nb] covers warp cols wn + nb*8.
    const int r0 = m0 + wm + lane / 4;
    const int cb = n0 + wn + (lane % 4) * 2;
    const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
    const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
    float* y0 = Y + (size_t)(lo + r0) * out_f;
    #pragma unroll
    for(int nb = 0; nb < 4; nb++){
        const int c = cb + nb * 8;
        if(r0 < m_e){
            if(c     < out_f) y0[c]     = acc[nb][0] * s0;
            if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
        }
        if(r0 + 8 < m_e){
            if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
            if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
        }
    }
}

// ============== round-51 (lane/sk-bm128): persistent problem-visitor + BM128 form ==============
//
// The round-49 sk kernel launches grid (ntx, ceil(max_m/BM), n_active) — under q35's ~17x routing
// skew that is ~92% early-exit churn (H100 ncu, research/sk-vs-cublas-20260801: grid (32,36,252)
// = 290k blocks, most exit at `m0 >= m_e` after ~2 loads), and BM=32 re-fetches each expert's W
// tile 4x more through L2 than cutlass's 128x64 (L2 SOL 75-88% — the wall that let cublas win
// 1.32x on H100 while HBM sat at 30%). Two fixes:
//   1. PROBLEM VISITOR (both forms): CTAs grid-stride a flat list of REAL tiles. Per-group tile
//      counts are prefix-summed into smem from the device CSR offsets (thread 0, n_active <=
//      SK_MAX_G); each flat id binary-searches its group. Zero early-exit blocks; the grid is
//      min(total_tiles, SMs x occupancy-API blocks) persistent CTAs — the cutlass waves=1 shape.
//   2. BM=128/BN=64/BK=64 3-STAGE cp.async form (sk128) for large-m groups: 4x fewer W-tile
//      re-fetches through L2 with the SAME sm_80-portable ldmatrix + mma.sync m16n8k16 pipeline
//      (no wgmma/TMA — runs on every memra arch). Small-m groups keep the 32x64 form (a 128-row
//      tile on a ~17-pair group is ~87% padding); the crossover is swept, not guessed
//      (MEMRA_F16G_SK_CROSS, read on the Rust side).
//
// NUMERICS: bit-identical to the round-49 kernel by construction — each output element's k-chain
// is the same ascending sequence of m16n8k16 f32-accumulate steps on the same f16 operands; only
// WHICH CTA computes a tile changes. Gated byte-identical in kernel-check ("f16g-sk" section).
//
// SMEM (sk128): A 128x(64+8) + B 64x(64+8) halves = 18432 + 9216 = 27648 B/stage x3 stages =
// 82944 B dynamic (+ 2052 B static tile-prefix) — needs the >48KB opt-in and fits sm_120a's
// ~99KB. cudaFuncSetAttribute is CHECKED with a device-fit fallback to the 32x64 visitor form
// (the round-49 mmq_iq rc=1001 lesson: never unchecked smem growth). 2 CTA/SM is smem-impossible
// for sk128 on sm_120a (2 x 82944 > ~100KB/SM): it runs 1 CTA/SM x 8 warps — occupancy is not
// the target, pipeline depth is (cutlass wins at 12.5% occupancy).

#define SK_MAX_G 512

// Per-group tile-count prefix into s_pre[0..n_active] (groups whose pair count lies outside
// [mlo, mhi) contribute 0 tiles). Thread 0 serial: n_active <= SK_MAX_G, one cheap pass per CTA.
__device__ __forceinline__ void sk_tile_prefix(int* s_pre, const int* ex_off, int n_active,
                                               int ntx, int bm, int mlo, int mhi){
    if(threadIdx.x == 0 && threadIdx.y == 0){
        int acc = 0; s_pre[0] = 0;
        for(int g = 0; g < n_active; g++){
            int m_e = ex_off[g+1] - ex_off[g];
            if(m_e >= mlo && m_e < mhi) acc += ((m_e + bm - 1)/bm)*ntx;
            s_pre[g+1] = acc;
        }
    }
    __syncthreads();
}
// Largest g with s_pre[g] <= t (empty groups have s_pre[g] == s_pre[g+1] and are skipped over).
__device__ __forceinline__ int sk_tile_group(const int* s_pre, int n_active, int t){
    int lo = 0, hi = n_active - 1;
    while(lo < hi){ int mid = (lo + hi + 1) >> 1; if(s_pre[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
}

// 32x64 visitor form: the round-49 tile body verbatim, driven by the flat tile list.
static __global__ void __launch_bounds__(128)
moe_f16g_sk32v_kernel(
        const __half* __restrict__ W, const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    __shared__ int s_pre[SK_MAX_G + 1];
    // explicit 16B alignment: cp.async.cg 16 + ldmatrix require it, and __half arrays placed
    // after the 2052-B s_pre otherwise land 4-aligned (the exact miss found on first run).
    __shared__ __align__(16) __half As[2][SK_BM][SK_STRIDE];
    __shared__ __align__(16) __half Bs[2][SK_BN][SK_STRIDE];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, mlo, mhi);

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nkb  = in_f / SK_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int ar = tid >> 2,  ac  = (tid & 3) * 8;
    const int br = tid >> 1,  bc  = (tid & 1) * 16;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;
        const int am = min(m0 + ar, m_e - 1);
        const int bn = min(n0 + br, out_f - 1);
        const __half* aga = Ag + (size_t)am * in_f + ac;
        const __half* bga = Wg + (size_t)bn * in_f + bc;

        sk_cp16(&As[0][ar][ac],    aga);
        sk_cp16(&Bs[0][br][bc],     bga);
        sk_cp16(&Bs[0][br][bc + 8], bga + 8);
        asm volatile("cp.async.commit_group;");

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb & 1;
            if(kb + 1 < nkb){
                const int nxt = cur ^ 1, k0 = (kb + 1) * SK_BK;
                sk_cp16(&As[nxt][ar][ac],    aga + k0);
                sk_cp16(&Bs[nxt][br][bc],     bga + k0);
                sk_cp16(&Bs[nxt][br][bc + 8], bga + k0 + 8);
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();
            #pragma unroll
            for(int kk = 0; kk < 2; kk++){
                unsigned a[4], b0[4], b1[4];
                sk_ldm16x16(a,  &As[cur][wm][kk*16],      SK_STRIDE);
                sk_ldm16x16(b0, &Bs[cur][wn][kk*16],      SK_STRIDE);
                sk_ldm16x16(b1, &Bs[cur][wn + 16][kk*16], SK_STRIDE);
                sk_mma(acc[0], a, b0[0], b0[2]);
                sk_mma(acc[1], a, b0[1], b0[3]);
                sk_mma(acc[2], a, b1[0], b1[2]);
                sk_mma(acc[3], a, b1[1], b1[3]);
            }
            __syncthreads();
        }

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
        const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
        float* y0 = Y + (size_t)(lo + r0) * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                if(c     < out_f) y0[c]     = acc[nb][0] * s0;
                if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
            }
            if(r0 + 8 < m_e){
                if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
                if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
            }
        }
    }
}

// 128x64x64 3-stage visitor form. 8 warps in a 4(m) x 2(n) grid — each warp owns a 32x32
// sub-tile (2 m-frags x 2 b-pairs of the same m16n8k16 pipeline). Dynamic smem (>48KB static
// limit): stages-major A then B.
#define SK128_BM 128
#define SK128_BN 64
#define SK128_BK 64
#define SK128_STRIDE (SK128_BK + 8)   // 72 halves = 144 B rows: 16B-aligned, de-banked
#define SK128_STAGES 3
#define SK128_A_ELEMS (SK128_BM * SK128_STRIDE)
#define SK128_B_ELEMS (SK128_BN * SK128_STRIDE)
#define SK128_SMEM_BYTES ((SK128_STAGES * (SK128_A_ELEMS + SK128_B_ELEMS)) * (int)sizeof(__half))

static __global__ void __launch_bounds__(256)
moe_f16g_sk128v_kernel(
        const __half* __restrict__ W, const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    extern __shared__ __align__(16) __half sk128_sm[];
    __half* Asm = sk128_sm;                                   // [stage][128][72]
    __half* Bsm = sk128_sm + SK128_STAGES * SK128_A_ELEMS;    // [stage][64][72]
    __shared__ int s_pre[SK_MAX_G + 1];
    const int ntx = (out_f + SK128_BN - 1) / SK128_BN;
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK128_BM, mlo, mhi);

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..255
    const int nkb  = in_f / SK128_BK;
    const int wm   = (warp & 3) * 32, wn = (warp >> 2) * 32;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK128_BM;
        const int n0 = (local % ntx) * SK128_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;

        // Stage-load geometry: A = 128 rows x 128B = 1024 cp.async16 (4/thread); B = 64 rows x
        // 128B = 512 (2/thread). Chunk c -> row c>>3, col (c&7)*8 halves. Rows past the group's
        // pairs / past out_f clamp to the last valid row — real bytes load, store guards discard.
        const __half* agp[4]; int asr[4], asc[4];
        #pragma unroll
        for(int i = 0; i < 4; i++){
            const int c = tid + i * 256;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        const __half* bgp[2]; int bsr[2], bsc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 256;
            bsr[i] = c >> 3; bsc[i] = (c & 7) * 8;
            const int bn = min(n0 + bsr[i], out_f - 1);
            bgp[i] = Wg + (size_t)bn * in_f + bsc[i];
        }
        #define SK128_LOAD(st, k0) do { \
            __half* a_s = Asm + (st) * SK128_A_ELEMS; \
            __half* b_s = Bsm + (st) * SK128_B_ELEMS; \
            _Pragma("unroll") \
            for(int i = 0; i < 4; i++) sk_cp16(a_s + asr[i]*SK128_STRIDE + asc[i], agp[i] + (k0)); \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) sk_cp16(b_s + bsr[i]*SK128_STRIDE + bsc[i], bgp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        // prologue: stages 0 (+1 when the k extent has a second block)
        SK128_LOAD(0, 0);
        if(nkb > 1) SK128_LOAD(1, SK128_BK);

        float acc[2][4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SK128_STAGES;
            if(kb + 2 < nkb){
                SK128_LOAD((kb + 2) % SK128_STAGES, (kb + 2) * SK128_BK);
                asm volatile("cp.async.wait_group 2;");
            } else if(kb + 1 < nkb){
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();
            const __half* Ab = Asm + cur * SK128_A_ELEMS;
            const __half* Bb = Bsm + cur * SK128_B_ELEMS;
            #pragma unroll
            for(int kk = 0; kk < 4; kk++){
                unsigned a0[4], a1[4], b0[4], b1[4];
                sk_ldm16x16(a0, Ab + (wm     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(a1, Ab + (wm + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b0, Bb + (wn     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b1, Bb + (wn + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_mma(acc[0][0], a0, b0[0], b0[2]);
                sk_mma(acc[0][1], a0, b0[1], b0[3]);
                sk_mma(acc[0][2], a0, b1[0], b1[2]);
                sk_mma(acc[0][3], a0, b1[1], b1[3]);
                sk_mma(acc[1][0], a1, b0[0], b0[2]);
                sk_mma(acc[1][1], a1, b0[1], b0[3]);
                sk_mma(acc[1][2], a1, b1[0], b1[2]);
                sk_mma(acc[1][3], a1, b1[1], b1[3]);
            }
            __syncthreads();
        }
        #undef SK128_LOAD

        #pragma unroll
        for(int mi = 0; mi < 2; mi++){
            const int r0 = m0 + wm + mi * 16 + lane / 4;
            const int cb = n0 + wn + (lane % 4) * 2;
            const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
            const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
            float* y0 = Y + (size_t)(lo + r0) * out_f;
            #pragma unroll
            for(int nb = 0; nb < 4; nb++){
                const int c = cb + nb * 8;
                if(r0 < m_e){
                    if(c     < out_f) y0[c]     = acc[mi][nb][0] * s0;
                    if(c + 1 < out_f) y0[c + 1] = acc[mi][nb][1] * s0;
                }
                if(r0 + 8 < m_e){
                    if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[mi][nb][2] * s1;
                    if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[mi][nb][3] * s1;
                }
            }
        }
    }
}

// ============== lane/sk-tail-form: DEEP-TAIL visitor form (32x64x64, 3-stage) ==============
//
// The H100 ncu pricing (research/sk-bm128-20260801): under q35's routing skew the 32x64x32
// 2-stage tail form above is 41.3 ms = 31% of the sk GEMM stage, and the x8/x16/x24 cross
// fine-sweep REFUTED pushing tail groups onto the 128 form (padding). This is the priced next
// rung: the SAME 32-row tile (zero extra padding on the sub-crossover groups this form exists
// for) with a 64-deep k-block and a 3-stage cp.async pipeline — 2 k-blocks (128 k-values) in
// flight instead of 1x32, and half the __syncthreads per k.
//
// SMEM math (the form pick, computed before building — lane receipts
// research/sk-tail-form-20260802/):
//   this form:   A 32x(64+8) 4608 B + B 64x72 9216 B = 13824 B/stage x3 = 41472 B
//                + 2052 B s_pre = 43524 B STATIC — under the 48 KB static limit (no opt-in);
//                sm_120a (~100 KB smem/SM) keeps 2 CTA/SM, H100 (228 KB) 5 smem-wise.
//   BM=64 alt:   18432 B/stage x3 + 2052 = 57348 B -> >48KB opt-in required, 1 CTA/SM on
//                sm_120a, and up to 2x tile padding on sub-crossover groups — rejected.
//   reg-double-buffer alt: smem-neutral but keeps the 32-k-deep global pipeline and the
//                per-32-k sync cadence — strictly weaker latency cover; rejected.
//
// NUMERICS: bit-identical to the round-49/51 forms by construction — per output element the
// k-chain is the same ascending m16n8k16 f32-accumulate sequence on the same f16 operands
// (each 64-k block runs the same four ascending 16-k mma steps the 32-k form runs in pairs).
// kernel-check "f16g-sk" gates every tail arm maxdiff==0 vs grid-scan (deep and legacy).
// MEMRA_F16G_TAIL=0 (parsed Rust-side, moe_f16g_tail_on) is the rollback seam to the
// 2-stage tail. in_f % 64 != 0 falls back to the 2-stage tail in-launcher.

#define SKT_BK 64
#define SKT_STRIDE (SKT_BK + 8)   // 72 halves = 144 B rows: 16B-aligned, de-banked
#define SKT_STAGES 3

static __global__ void __launch_bounds__(128)
moe_f16g_sktail_kernel(
        const __half* __restrict__ W, const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SKT_STAGES][SK_BN][SKT_STRIDE];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, mlo, mhi);

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..127
    const int nkb  = in_f / SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const __half* Wg = W + (size_t)g * (size_t)out_f * in_f;

        // Stage-load geometry (chunk = 16 B = 8 halves; 8 chunks per 64-half row):
        // A = 32 rows x 8 = 256 chunks (2/thread), B = 64 x 8 = 512 (4/thread). Rows past
        // the group's pairs / past out_f clamp to the last valid row — real bytes load,
        // the store guards discard the results.
        const __half* agp[2]; int asr[2], asc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 128;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        const __half* bgp[4]; int bsr[4], bsc[4];
        #pragma unroll
        for(int i = 0; i < 4; i++){
            const int c = tid + i * 128;
            bsr[i] = c >> 3; bsc[i] = (c & 7) * 8;
            const int bn = min(n0 + bsr[i], out_f - 1);
            bgp[i] = Wg + (size_t)bn * in_f + bsc[i];
        }
        #define SKT_LOAD(st, k0) do { \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            _Pragma("unroll") \
            for(int i = 0; i < 4; i++) sk_cp16(&Bs[st][bsr[i]][bsc[i]], bgp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        // prologue: stages 0 (+1 when the k extent has a second block)
        SKT_LOAD(0, 0);
        if(nkb > 1) SKT_LOAD(1, SKT_BK);

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SKT_STAGES;
            if(kb + 2 < nkb){
                SKT_LOAD((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);
                asm volatile("cp.async.wait_group 2;");
            } else if(kb + 1 < nkb){
                asm volatile("cp.async.wait_group 1;");
            } else {
                asm volatile("cp.async.wait_group 0;");
            }
            __syncthreads();
            #pragma unroll
            for(int kk = 0; kk < 4; kk++){
                unsigned a[4], b0[4], b1[4];
                sk_ldm16x16(a,  &As[cur][wm][kk*16],      SKT_STRIDE);
                sk_ldm16x16(b0, &Bs[cur][wn][kk*16],      SKT_STRIDE);
                sk_ldm16x16(b1, &Bs[cur][wn + 16][kk*16], SKT_STRIDE);
                sk_mma(acc[0], a, b0[0], b0[2]);
                sk_mma(acc[1], a, b0[1], b0[3]);
                sk_mma(acc[2], a, b1[0], b1[2]);
                sk_mma(acc[3], a, b1[1], b1[3]);
            }
            __syncthreads();
        }
        #undef SKT_LOAD

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
        const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
        float* y0 = Y + (size_t)(lo + r0) * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                if(c     < out_f) y0[c]     = acc[nb][0] * s0;
                if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
            }
            if(r0 + 8 < m_e){
                if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
                if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
            }
        }
    }
}

// ============ lane/kquant-tile-loaders: DIRECT-FROM-QUANT sk visitor forms ============
//
// The Ornith-35B pp512 finding (research/q4k-expert-prefill-20260802 §5): at t=512 the
// q4_K/q6_K -> f16 dequant passes are 41.8% of GPU kernel time — a fixed per-(layer,proj)
// cost over the ~all-active expert bank (~44GB f16 write+read per pass at the 858GB/s wall).
// The kill is NO dequant pass: these kernels are the round-51 visitor forms with the B-side
// (weight) cp.async tile loads replaced by dequant-in-register DIRECTLY from the Q4_K/Q6_K
// superblocks in the expert slab (table/ex_ids/row_bytes — the same pointer-table contract
// as the dequant kernels above). The A-side (activation) pipeline is untouched.
// lane/iq-direct-loaders extends the same discipline to IQ4_XS/IQ3_S — the class that is
// 94.8% of q35's expert-bank bytes (the h100-sk-direct coverage pricing): the mode-2 IQ
// workspace pass dies the same way.
//
// NUMERICS: bit-identical to the workspace path BY CONSTRUCTION — the B smem tile holds the
// same f16 values (kq_q4k_val/kq_q6k_val for the k-quants; the iq4_xs/iq3_s dequant kernels'
// exact per-value expressions for the IQ classes) in the same positions, and the mma sequence
// per output element is unchanged. Gated bitwise in kernel-check ("f16g-kq-direct") on
// synthetic + real weights; MEMRA_F16G_DIRECT=0 is the rollback seam back to the
// dequant-workspace path (Rust-side admission).
//
// B tile is SINGLE-buffered (the dequant is synchronous; the trailing __syncthreads of each
// kb iteration already fences the next overwrite) — the A cp.async prefetch stays in flight
// behind the B global quant reads + ALU. sk128's dynamic smem drops to 3xA + 1xB = 64512B.

// Per-thread 16-value window, SOFTWARE-PIPELINED: the raw quant bytes for kb+1 fetch into
// REGISTERS before kb's mma runs (global latency hides behind the tensor-core work), and the
// loop-invariant scale loads/products hoist once per window. A 16-aligned window never
// crosses a q4k/q6k sub-scale boundary (q4k/q6k sub-scales cover 32/16 values; iq4_xs/iq3_s
// scales cover 32 — every class holds). VALUE MATH: the workspace dequant kernels' exact
// DAG — `dd*(float)sc8` / `dd*(float)sc` / `d_sb*(float)(ls-32)` / `dd*(1+2*nib)` are the
// left-assoc first products of those expressions, hoisted unchanged (bitwise-gated vs the
// workspace path in kernel-check "f16g-kq-direct").
//
// IQ classes (lane/iq-direct-loaders): the per-value codebooks ride SHARED memory, staged
// once per launch — divergent per-value lookups serialize on the constant cache, shared is
// banked. IQ4_XS stages the 16 kvalues as f32 bits (the workspace's `(float)kvalues[code]`
// pre-converted — small ints are f32-exact); IQ3_S stages the 512-word grid and resolves the
// window's four grid words AT FETCH (behind the previous kb's mma), so the store is pure
// extract+mul.
// QT_NVFP4_V2 MUST be listed here. kq_stage_codebook stages 16 words for BOTH NVFP4 layouts
// (`QT == QT_NVFP4 || QT == QT_NVFP4_V2`), but this macro sized only the v1 constant, so the v2
// instantiations -- at the time, every run with MEMRA_NVFP4_BANK_V2=1, a door REMOVED
// 2026-08-29 (research/step37-bankv2-removal-20260829); QT_NVFP4_V2 now reaches this GEMM
// only through the always-slot-major EP2 banks or the moe_tp2_repro harness --
// declared `s_cb[1]` and threads
// 1..15 wrote 15 words past the end of the array on every launch. compute-sanitizer memcheck,
// 2026-08-28: 6816 "Invalid __shared__ write of size 4 bytes" in moe_kq_sktail_kernel<107> at
// t=576, 288 at t=4096, naming threads 4 and 5 writing s_cb[4]/s_cb[5]; qt=7 clean at both
// geometries. A shared-memory overrun is not a crash, it is a CORRUPTION whose victim depends on
// what else is resident, which is why it presented as ULP-dense nondeterminism with byte-identical
// inputs (and, in the workspace lane, as 1.6e20 garbage) rather than as a fault.
#define KQ_CB_WORDS(QT) \
    ((QT) == QT_IQ3_S ? 512                                                   \
     : (((QT) == QT_IQ4_XS || (QT) == QT_NVFP4 || (QT) == QT_NVFP4_V2         \
          || (QT) == QT_NVFP4_MODELOPT) ? 16 : 1))
template<int QT>
__device__ __forceinline__ void kq_stage_codebook(uint32_t* s_cb){
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if(QT == QT_IQ3_S){
        const int nthr = blockDim.x * blockDim.y;
        for(int i = tid; i < 512; i += nthr) s_cb[i] = g_iq3s_grid[i];
    } else if(QT == QT_IQ4_XS){
        if(tid < 16) s_cb[tid] = __float_as_uint((float)g_kvalues_iq4nl[tid]);
    } else if(QT == QT_NVFP4 || QT == QT_NVFP4_V2 || QT == QT_NVFP4_MODELOPT){
        // same pre-converted-float trick as IQ4_XS: the workspace kernel's
        // `(float)g_kvalues_mxfp4[code]` — small ints are f32-exact.
        if(tid < 16) s_cb[tid] = __float_as_uint((float)g_kvalues_mxfp4[tid]);
    }
    // no fence here: every caller runs sk_tile_prefix (ends in __syncthreads) next.
}
struct KqRaw {
    uint32_t q[4];      // q4k: 16 nibble bytes (one uint4) | q6k: 16 ql bytes (8x u16)
                        // iq4_xs: the 32-group's 16 qs bytes | iq3_s: 4 RESOLVED grid words
    uint32_t qh[4];     // q6k: 16 qh bytes (8x u16) | iq3_s: qh[0] = 2 sign bytes
    float f1, f2;       // q4k: (dd*sc8, dmin*m8) | q6k: (dd*sc, -) | iq4_xs: (d_sb*(ls-32), -)
                        // iq3_s: (dd*(1+2*nib), -)
    int sel;            // q4k/iq4_xs: hi-nibble half | q6k: q4 (2-bit shift selector)
};
// `in_f` is REQUIRED and has NO default. It used to default to 0, and exactly two of the ten
// call sites (moe_kq_sktail_kernel's kb+1 prefetch pair) relied on that default. QT_NVFP4_V2 is
// the only branch that reads in_f -- it locates the slot-major row's UE4M3 scale tail at
// n_slots*16 -- so in_f=0 sent the scale fetch into the packed-codes region while the 4-bit
// codes stayed correct: right weights, wrong per-16-element scale, on every k-block but kb=0.
// Every other qtype branch ignores in_f, which is why the omission was a silent no-op for v1 and
// a margin-sensitive logits corruption for v2 (research/step37-bankv3-20260901/DIAGNOSIS.md).
// Keep it undefaulted: a defaultable geometry field that only one layout consumes is the hole.
template<int QT>
__device__ __forceinline__ KqRaw kq_fetch(const uint8_t* __restrict__ wrow,
                                          const uint8_t* __restrict__ scrow, int k0v,
                                          const uint32_t* __restrict__ s_cb, int in_f){
    // k0v = 16-aligned value offset within the row (absolute k of the window start)
    constexpr int SBB = (QT == QT_Q4_K) ? 144 : (QT == QT_Q6_K) ? 210
                      : (QT == QT_IQ4_XS) ? 136 : 110;  // unused by the NVFP4 branches
    const uint8_t* b = wrow + (size_t)(k0v >> 8) * SBB;
    const int l0 = k0v & 255;
    KqRaw r;
    if(QT == QT_NVFP4_MODELOPT){
        // ModelOpt split planes: consecutive elements share one packed byte;
        // scrow has one signed-E4M3 byte per 16 K values. The scale_2 macro
        // remains an epilogue row scale owned by the DSV4 caller.
        r.f1 = g_e4m3fn_to_float(scrow[k0v >> 4]);
        r.f2 = 0.0f;
        r.sel = 0;
        const uint8_t* qsp = wrow + (k0v >> 1);
        r.q[0] = *(const uint32_t*)qsp;
        r.q[1] = *(const uint32_t*)(qsp + 4);
        return r;
    }
    if(QT == QT_NVFP4_V2){
        // v2 slot-major bank (tp.rs nvfp4_matrix_v2_permute): slot g's 16 qs bytes at g*16,
        // its two UE4M3 scale bytes in the tail at n_slots*16 + g*2. Same values, same DAG as
        // the v1 branch below and as dequant_nvfp4v2_f16_kernel — only the byte map differs.
        // The 16-value window is one 16-value sub-block, so it never crosses a scale.
        const int n_slots = in_f >> 5;
        const int g = k0v >> 5, sub = (k0v >> 4) & 1;
        r.f1 = g_ue4m3_to_float(wrow[(size_t)n_slots * 16 + g * 2 + sub]);
        r.f2 = 0.0f;
        r.sel = 0;
        const uint8_t* qsp = wrow + (size_t)g * 16 + sub * 8;
        r.q[0] = *(const uint32_t*)qsp;
        r.q[1] = *(const uint32_t*)(qsp + 4);
        return r;
    }
    if(QT == QT_NVFP4){
        // 36B block covers 64 values in 4 UE4M3-scaled sub-blocks of 16 — the 16-value
        // window IS one sub-block (never crosses a scale boundary by construction).
        // dequant_nvfp4_f16_kernel's exact DAG: f1 = g_ue4m3_to_float(d_bytes[sub]);
        // value j: byte qs[sub*8 + (j&7)], code = j<8 ? lo-nibble : hi-nibble.
        const uint8_t* nb = wrow + (size_t)(k0v >> 6) * 36;
        const int sub = (k0v >> 4) & 3;
        r.f1 = g_ue4m3_to_float(nb[sub]);
        r.f2 = 0.0f;
        r.sel = 0;
        const uint8_t* qsp = nb + 4 + sub * 8;   // 4-aligned (36B blocks on 8-aligned rows)
        r.q[0] = *(const uint32_t*)qsp;
        r.q[1] = *(const uint32_t*)(qsp + 4);
        return r;
    }
    if(QT == QT_Q4_K){
        float dd = g_half_to_float(*(const uint16_t*)b);
        float dmin = g_half_to_float(*(const uint16_t*)(b+2));
        const uint8_t* scs = b+4;
        int g64 = l0>>6, w64 = l0&63, is = g64*2 + (w64>>5);
        int sc8, m8;
        if(is < 4){ sc8 = scs[is] & 63;                       m8 = scs[is+4] & 63; }
        else      { sc8 = (scs[is+4] & 0xF) | ((scs[is-4] >> 6) << 4);
                    m8  = (scs[is+4] >> 4)  | ((scs[is]   >> 6) << 4); }
        r.f1 = dd * (float)sc8;
        r.f2 = dmin * (float)m8;
        r.sel = (w64 >= 32);
        uint4 q = *(const uint4*)(b + 16 + g64*32 + (w64 & 31));   // 16B-aligned
        r.q[0] = q.x; r.q[1] = q.y; r.q[2] = q.z; r.q[3] = q.w;
    } else if(QT == QT_IQ4_XS){
        // dequant_iq4xs_f16_kernel's exact DAG: d_sb*(float)(ls-32) is the left-assoc first
        // product, hoisted. Both 16-value halves of a 32-group read the group's SAME 16 qs
        // bytes (sel picks the nibble). 136B superblocks keep rows 8-aligned, not 16 — uint2.
        float d_sb = g_half_to_float(*(const uint16_t*)b);
        uint16_t sh = *(const uint16_t*)(b+2);
        const uint8_t* sl = b+4;
        int g = l0>>5, lg = l0&31;
        int ls = ((sl[g>>1]>>(4*(g&1)))&0xf) | (((sh>>(2*g))&3)<<4);
        r.f1 = d_sb * (float)(ls-32);
        r.f2 = 0.0f;
        r.sel = (lg >= 16);
        const uint8_t* gqs = b + 8 + g*16;
        uint2 q0 = *(const uint2*)gqs;
        uint2 q1 = *(const uint2*)(gqs + 8);
        r.q[0] = q0.x; r.q[1] = q0.y; r.q[2] = q1.x; r.q[3] = q1.y;
    } else if(QT == QT_IQ3_S){
        // dequant_iq3s_f16_kernel's exact DAG: db = dd*(1+2*nib) hoists; the window's four
        // grid words (2 per 8-value chunk) resolve HERE through the shared codebook so those
        // lookups fly behind the previous kb's mma. 110B superblocks: 2-aligned loads only.
        float dd = g_half_to_float(*(const uint16_t*)b);
        int ib32 = l0>>5, l4b = (l0&31)>>3;                     // l4b in {0,2}
        const uint8_t* sc = b+106;
        r.f1 = dd * (1.0f + 2.0f*(float)((sc[ib32>>1] >> (4*(ib32&1))) & 0xf));
        r.f2 = 0.0f;
        r.sel = 0;
        const int qhb = b[66 + ib32];
        #pragma unroll
        for(int c = 0; c < 2; c++){
            const int l4 = l4b + c;
            const uint16_t qs2 = *(const uint16_t*)(b + 2 + ib32*8 + 2*l4);   // 2-aligned
            r.q[2*c]   = s_cb[(qs2 & 0xff) | ((qhb << (8-2*l4)) & 256)];
            r.q[2*c+1] = s_cb[(qs2 >> 8)   | ((qhb << (7-2*l4)) & 256)];
        }
        r.qh[0] = (uint32_t)b[74 + ib32*4 + l4b]
                | ((uint32_t)b[74 + ib32*4 + l4b + 1] << 8);
    } else {
        float dd = g_half_to_float(*(const uint16_t*)(b+208));
        int n2 = l0>>7, rr = l0&127, q4 = rr>>5, l = rr&31;
        int is = l>>4;
        const int8_t* sc = (const int8_t*)(b+192);
        r.f1 = dd * (float)sc[n2*8 + is + 2*q4];
        r.f2 = 0.0f;
        r.sel = q4;
        const uint8_t* qlp = b + n2*64 + l + (q4&1)*32;   // 2-aligned (offsets even)
        const uint8_t* qhp = b + 128 + n2*32 + l;
        #pragma unroll
        for(int i = 0; i < 4; i++){
            r.q[i]  = (uint32_t)*(const uint16_t*)(qlp + 4*i)
                    | ((uint32_t)*(const uint16_t*)(qlp + 4*i + 2) << 16);
            r.qh[i] = (uint32_t)*(const uint16_t*)(qhp + 4*i)
                    | ((uint32_t)*(const uint16_t*)(qhp + 4*i + 2) << 16);
        }
    }
    return r;
}
template<int QT>
__device__ __forceinline__ void kq_store(const KqRaw& r, __half* __restrict__ dst,
                                         const uint32_t* __restrict__ s_cb){
    #pragma unroll
    for(int j = 0; j < 16; j++){
        if(QT == QT_NVFP4_MODELOPT){
            int byte = (r.q[(j >> 1) >> 2] >> (8 * ((j >> 1) & 3))) & 0xff;
            int code = (j & 1) ? (byte >> 4) : (byte & 0xF);
            dst[j] = __float2half(r.f1 * __uint_as_float(s_cb[code]));
        } else if(QT == QT_NVFP4 || QT == QT_NVFP4_V2){
            // bytes qs[0..7] live in q[0..1]; j<8 lo nibble, j>=8 hi nibble (workspace order).
            // v2 fills q[] with the same 8 bytes from its slot-major offset, so the store DAG
            // is shared verbatim — the layouts differ only in where kq_fetch read them.
            
            int byte = (r.q[(j&7)>>2] >> (8*(j&3))) & 0xff;
            int code = (j < 8) ? (byte & 0xF) : (byte >> 4);
            dst[j] = __float2half(r.f1 * __uint_as_float(s_cb[code]));
        } else if(QT == QT_Q4_K){
            int byte = (r.q[j>>2] >> (8*(j&3))) & 0xff;
            int nib = r.sel ? (byte >> 4) : (byte & 0xF);
            dst[j] = __float2half(r.f1 * (float)nib - r.f2);
        } else if(QT == QT_IQ4_XS){
            int byte = (r.q[j>>2] >> (8*(j&3))) & 0xff;
            int code = r.sel ? (byte >> 4) : (byte & 0xF);
            dst[j] = __float2half(r.f1 * __uint_as_float(s_cb[code]));
        } else if(QT == QT_IQ3_S){
            // chunk c = j>>3 (8 values); word = grid1/grid2 of the chunk; byte = j&3.
            int gb = (r.q[(j>>3)*2 + ((j>>2)&1)] >> (8*(j&3))) & 0xff;
            float sgn = ((r.qh[0] >> (8*(j>>3))) & (1u << (j&7))) ? -1.0f : 1.0f;
            dst[j] = __float2half(r.f1 * (float)gb * sgn);
        } else {
            int qlb = (r.q[j>>2]  >> (8*(j&3))) & 0xff;
            int qhb = (r.qh[j>>2] >> (8*(j&3))) & 0xff;
            int nib = (r.sel < 2) ? (qlb & 0xF) : (qlb >> 4);
            int qv = (nib | (((qhb >> (2*r.sel)) & 3) << 4)) - 32;
            dst[j] = __float2half(r.f1 * (float)qv);
        }
    }
}

// ModelOpt-only packed store twin.  The LUT index is the original packed code byte, so one
// entry contains the two adjacent E2M1 values as exact half bits.  The finite E4M3 scale is
// representable in half, and the doubled-integer codebook values are representable exactly; the
// half2 product therefore has the same rounded half result as the scalar float product followed
// by __float2half.  This path is deliberately dynamic-shared only in its opt-in launchers: the
// existing 16-entry f32 shared-LUT kernels keep their original static allocation and instruction
// path when PackedStore=false.
template<bool PackedStore>
__device__ __forceinline__ void kq_stage_half2_lut(uint32_t* __restrict__ s_h2){
    if constexpr(PackedStore){
        const int tid = threadIdx.y * 32 + threadIdx.x;
        for(int i = tid; i < 256; i += blockDim.x * blockDim.y){
            const int lo = i & 0xF;
            const int hi = i >> 4;
            const __half hlo = __float2half((float)g_kvalues_mxfp4[lo]);
            const __half hhi = __float2half((float)g_kvalues_mxfp4[hi]);
            s_h2[i] = (uint32_t)__half_as_ushort(hlo)
                    | ((uint32_t)__half_as_ushort(hhi) << 16);
        }
        __syncthreads();
    }
}

template<int QT, bool PackedStore>
__device__ __forceinline__ void kq_store_variant(const KqRaw& r, __half* __restrict__ dst,
                                                 const uint32_t* __restrict__ s_cb,
                                                 const uint32_t* __restrict__ s_h2){
    if constexpr(PackedStore){
        static_assert(QT == QT_NVFP4_MODELOPT, "packed half2 store is ModelOpt-only");
        const __half scale = __float2half(r.f1);
        const __half2 scale2 = __halves2half2(scale, scale);
        #pragma unroll
        for(int pair = 0; pair < 8; pair++){
            const int byte = (r.q[pair >> 2] >> (8 * (pair & 3))) & 0xff;
            const __half2 codes = *reinterpret_cast<const __half2*>(&s_h2[byte]);
            *reinterpret_cast<__half2*>(dst + pair * 2) = __hmul2(scale2, codes);
        }
    } else {
        kq_store<QT>(r, dst, s_cb);
    }
}

template<int QT>
static __global__ void __launch_bounds__(128)
moe_kq_sk32v_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[2][SK_BM][SK_STRIDE];
    __shared__ __align__(16) __half Bs[SK_BN][SK_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT>(s_cb);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, mlo, mhi);
    if (total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nkb  = in_f / SK_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int ar = tid >> 2,  ac  = (tid & 3) * 8;
    const int brow = tid >> 1, bc0 = (tid & 1) * 16;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const int qplane = (QT == QT_NVFP4_MODELOPT) ? 2 * proj : proj;
        const uint8_t* Wq = (const uint8_t*)table[(size_t)qplane*n_expert + eid];
        const uint8_t* Wsc = (QT == QT_NVFP4_MODELOPT)
            ? (const uint8_t*)table[(size_t)(qplane + 1)*n_expert + eid] : nullptr;
        const int am = min(m0 + ar, m_e - 1);
        const __half* aga = Ag + (size_t)am * in_f + ac;
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* wrow = Wq + (size_t)bn * row_bytes;
        const uint8_t* scrow = Wsc ? Wsc + (size_t)bn * (in_f / 16) : nullptr;

        sk_cp16(&As[0][ar][ac], aga);
        asm volatile("cp.async.commit_group;");
        KqRaw braw = kq_fetch<QT>(wrow, scrow, bc0, s_cb, in_f);   // kb=0 window

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb & 1;
            if(kb + 1 < nkb){
                const int nxt = cur ^ 1, k0n = (kb + 1) * SK_BK;
                sk_cp16(&As[nxt][ar][ac], aga + k0n);
                asm volatile("cp.async.commit_group;");
            }
            // B tile for THIS kb from the pre-fetched registers (previous kb's trailing
            // __syncthreads fences the Bs overwrite), then issue kb+1's raw fetch so those
            // global reads fly behind this kb's mma.
            kq_store<QT>(braw, &Bs[brow][bc0], s_cb);
            if(kb + 1 < nkb) braw = kq_fetch<QT>(wrow, scrow, (kb + 1) * SK_BK + bc0, s_cb, in_f);
            if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else             asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            #pragma unroll
            for(int kk = 0; kk < 2; kk++){
                unsigned a[4], b0[4], b1[4];
                sk_ldm16x16(a,  &As[cur][wm][kk*16],  SK_STRIDE);
                sk_ldm16x16(b0, &Bs[wn][kk*16],       SK_STRIDE);
                sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SK_STRIDE);
                sk_mma(acc[0], a, b0[0], b0[2]);
                sk_mma(acc[1], a, b0[1], b0[3]);
                sk_mma(acc[2], a, b1[0], b1[2]);
                sk_mma(acc[3], a, b1[1], b1[3]);
            }
            __syncthreads();
        }

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
        const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
        float* y0 = Y + (size_t)(lo + r0) * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                if(c     < out_f) y0[c]     = acc[nb][0] * s0;
                if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
            }
            if(r0 + 8 < m_e){
                if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
                if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
            }
        }
    }
}

// 128x64x64 direct form: A keeps the 3-stage cp.async pipeline; B is a single 64x72 tile
// dequanted per kb (256 threads, row = tid>>2, 16-value quarter-row per thread).
// B carries TWO buffers (lane/kq-bdb 2026-08-28): with a single buffer the weight-tile
// dequant sits between two __syncthreads and cannot overlap the mma — see the schedule note
// on moe_kq_sk128v_kernel. 3*128*72 + 2*64*72 halves = 72 KB, still one CTA/SM (occ128=1 was
// already 1 at 63 KB), so this buys the overlap without costing occupancy.
#define KQ128_B_BUFS 2
#define KQ128_SMEM_BYTES ((SK128_STAGES * SK128_A_ELEMS + KQ128_B_BUFS * SK128_B_ELEMS) * (int)sizeof(__half))

template<int QT, bool BDB>
static __global__ void __launch_bounds__(256)
moe_kq_sk128v_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    extern __shared__ __align__(16) __half kq128_sm[];
    __half* Asm = kq128_sm;                                   // [stage][128][72]
    __half* Bsm = kq128_sm + SK128_STAGES * SK128_A_ELEMS;    // [KQ128_B_BUFS][64][72]
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    const int ntx = (out_f + SK128_BN - 1) / SK128_BN;
    kq_stage_codebook<QT>(s_cb);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK128_BM, mlo, mhi);
    if (total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..255
    const int nkb  = in_f / SK128_BK;
    const int wm   = (warp & 3) * 32, wn = (warp >> 2) * 32;
    const int brow = tid >> 2, bc0 = (tid & 3) * 16;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK128_BM;
        const int n0 = (local % ntx) * SK128_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const int qplane = (QT == QT_NVFP4_MODELOPT) ? 2 * proj : proj;
        const uint8_t* Wq = (const uint8_t*)table[(size_t)qplane*n_expert + eid];
        const uint8_t* Wsc = (QT == QT_NVFP4_MODELOPT)
            ? (const uint8_t*)table[(size_t)(qplane + 1)*n_expert + eid] : nullptr;
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* wrow = Wq + (size_t)bn * row_bytes;
        const uint8_t* scrow = Wsc ? Wsc + (size_t)bn * (in_f / 16) : nullptr;

        const __half* agp[4]; int asr[4], asc[4];
        #pragma unroll
        for(int i = 0; i < 4; i++){
            const int c = tid + i * 256;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        #define KQ128_LOAD_A(st, k0) do { \
            __half* a_s = Asm + (st) * SK128_A_ELEMS; \
            _Pragma("unroll") \
            for(int i = 0; i < 4; i++) sk_cp16(a_s + asr[i]*SK128_STRIDE + asc[i], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        KQ128_LOAD_A(0, 0);
        if(nkb > 1) KQ128_LOAD_A(1, SK128_BK);
        KqRaw braw = kq_fetch<QT>(wrow, scrow, bc0, s_cb, in_f);   // kb=0 window
        // BDB: fill B[0] up front and pull kb=1's raw window, so the loop body always stores
        // into the buffer the NEXT iteration reads while the mma consumes this one.
        if(BDB){
            kq_store<QT>(braw, Bsm + brow*SK128_STRIDE + bc0, s_cb);
            if(nkb > 1) braw = kq_fetch<QT>(wrow, scrow, SK128_BK + bc0, s_cb, in_f);
        }

        float acc[2][4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SK128_STAGES;
            if(!BDB){
            if(kb + 2 < nkb) KQ128_LOAD_A((kb + 2) % SK128_STAGES, (kb + 2) * SK128_BK);
            // B tile for THIS kb from the pre-fetched registers (single buffer; previous
            // kb's trailing __syncthreads fences), then issue kb+1's raw fetch so those
            // global reads fly behind this kb's mma.
            kq_store<QT>(braw, Bsm + brow*SK128_STRIDE + bc0, s_cb);
            if(kb + 1 < nkb) braw = kq_fetch<QT>(wrow, scrow, (kb + 1) * SK128_BK + bc0, s_cb, in_f);
            if(kb + 2 < nkb)      asm volatile("cp.async.wait_group 2;");
            else if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else                  asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            } else {
            // DOUBLE-BUFFERED B (lane/kq-bdb): ONE barrier per k-block instead of two, and the
            // dequant of kb+1's weight tile is issued AFTER it and overlaps this kb's mma.
            //
            // Why it is safe with a single barrier. The barrier at the top of iteration kb
            // separates iteration kb-1's mma from everything in kb. Past it:
            //   * KQ128_LOAD_A targets stage (kb+2)%3 == (kb-1)%3, which kb-1's mma just
            //     finished reading — the barrier is exactly what makes that overwrite legal,
            //     which is why the issue moved BELOW it (in the single-buffer schedule the
            //     TRAILING __syncthreads played this role).
            //   * kq_store targets B[(kb+1)&1], last read by kb-1's mma — same argument.
            //   * the mma reads A[kb%3] and B[kb&1], both filled before this barrier.
            // Group accounting: the prologue issues k0 and k1, and iteration kb issues k(kb+2)
            // AFTER the wait, so at the wait the newest issued group is k(kb+1) and "kb is
            // complete" means at most one group outstanding -> wait_group 1.
            if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else             asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if(kb + 2 < nkb) KQ128_LOAD_A((kb + 2) % SK128_STAGES, (kb + 2) * SK128_BK);
            if(kb + 1 < nkb){
                kq_store<QT>(braw, Bsm + ((kb + 1) & 1) * SK128_B_ELEMS
                                       + brow*SK128_STRIDE + bc0, s_cb);
                if(kb + 2 < nkb)
                    braw = kq_fetch<QT>(wrow, scrow, (kb + 2) * SK128_BK + bc0, s_cb, in_f);
            }
            }
            const __half* Ab = Asm + cur * SK128_A_ELEMS;
            const __half* Bb = BDB ? (Bsm + (kb & 1) * SK128_B_ELEMS) : Bsm;
            #pragma unroll
            for(int kk = 0; kk < 4; kk++){
                unsigned a0[4], a1[4], b0[4], b1[4];
                sk_ldm16x16(a0, Ab + (wm     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(a1, Ab + (wm + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b0, Bb + (wn     ) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_ldm16x16(b1, Bb + (wn + 16) * SK128_STRIDE + kk*16, SK128_STRIDE);
                sk_mma(acc[0][0], a0, b0[0], b0[2]);
                sk_mma(acc[0][1], a0, b0[1], b0[3]);
                sk_mma(acc[0][2], a0, b1[0], b1[2]);
                sk_mma(acc[0][3], a0, b1[1], b1[3]);
                sk_mma(acc[1][0], a1, b0[0], b0[2]);
                sk_mma(acc[1][1], a1, b0[1], b0[3]);
                sk_mma(acc[1][2], a1, b1[0], b1[2]);
                sk_mma(acc[1][3], a1, b1[1], b1[3]);
            }
            // Single-buffer schedule needs the trailing fence (the next kq_store overwrites the
            // one B tile this mma just read). BDB's next store targets the OTHER buffer and is
            // fenced by the next iteration's leading barrier instead.
            if(!BDB) __syncthreads();
        }
        #undef KQ128_LOAD_A

        #pragma unroll
        for(int mi = 0; mi < 2; mi++){
            const int r0 = m0 + wm + mi * 16 + lane / 4;
            const int cb = n0 + wn + (lane % 4) * 2;
            const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
            const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
            float* y0 = Y + (size_t)(lo + r0) * out_f;
            #pragma unroll
            for(int nb = 0; nb < 4; nb++){
                const int c = cb + nb * 8;
                if(r0 < m_e){
                    if(c     < out_f) y0[c]     = acc[mi][nb][0] * s0;
                    if(c + 1 < out_f) y0[c + 1] = acc[mi][nb][1] * s0;
                }
                if(r0 + 8 < m_e){
                    if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[mi][nb][2] * s1;
                    if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[mi][nb][3] * s1;
                }
            }
        }
    }
}

// DEEP-TAIL direct form (lane/sk-tail-form): the 32x64x64 3-stage tail with the B side
// dequanted in-register from the Q4_K/Q6_K superblocks — A keeps the 3-stage cp.async
// pipeline (32x72 x3 = 13824 B), B is a single 64x72 tile (9216 B; the trailing
// __syncthreads of each kb fences the overwrite, exactly the kq_sk32v/kq_sk128v contract).
// 128 threads: B row = tid>>1, each thread owns 32 values = TWO 16-value KqRaw windows,
// both software-pipelined one kb ahead behind the mma. Total smem 25092 B static.
// Bit-identical to every other form by construction (same kq_*_val f16 values into the
// same ascending mma k-chain).
// M1=true is a compile-only DSV4 candidate for one-row CSR groups: it keeps
// the same B tile and valid-row (warps 0/2) m16n8k16 chain, but does not issue
// the duplicate invalid-row MMA work from warps 1/3 or load their unused A
// stage. It is deliberately a separate launcher seam, not the rejected scalar
// m=1 visitor; the valid row's accumulation and epilogue remain unchanged.
template<int QT, bool M1 = false, bool PackedStore = false>
static __global__ void __launch_bounds__(128)
moe_kq_sktail_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ Y, const float* __restrict__ row_scale,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int mlo, int mhi, int total_tiles){
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SK_BN][SKT_STRIDE];   // single buffer
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    extern __shared__ uint32_t packed_h2[];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, mlo, mhi);
    if (total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..127
    const int nkb  = in_f / SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;          // 2 threads/row, 32 values each

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const int qplane = (QT == QT_NVFP4_MODELOPT) ? 2 * proj : proj;
        const uint8_t* Wq = (const uint8_t*)table[(size_t)qplane*n_expert + eid];
        const uint8_t* Wsc = (QT == QT_NVFP4_MODELOPT)
            ? (const uint8_t*)table[(size_t)(qplane + 1)*n_expert + eid] : nullptr;
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* wrow = Wq + (size_t)bn * row_bytes;
        const uint8_t* scrow = Wsc ? Wsc + (size_t)bn * (in_f / 16) : nullptr;

        const __half* agp[2]; int asr[2], asc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 128;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        #define KQT_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) if(!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        KQT_LOAD_A(0, 0);
        if(nkb > 1) KQT_LOAD_A(1, SKT_BK);
        KqRaw braw0 = kq_fetch<QT>(wrow, scrow, bc0, s_cb, in_f);        // kb=0 windows
        KqRaw braw1 = kq_fetch<QT>(wrow, scrow, bc0 + 16, s_cb, in_f);

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SKT_STAGES;
            if(kb + 2 < nkb) KQT_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);
            // B tile for THIS kb from the pre-fetched registers (previous kb's trailing
            // __syncthreads fences the overwrite), then issue kb+1's raw fetches so those
            // global reads fly behind this kb's mma.
            kq_store_variant<QT, PackedStore>(braw0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(braw1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                braw0 = kq_fetch<QT>(wrow, scrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                braw1 = kq_fetch<QT>(wrow, scrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if(kb + 2 < nkb)      asm volatile("cp.async.wait_group 2;");
            else if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else                  asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if(!M1 || ((warp & 1) == 0)) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(acc[0], a, b0[0], b0[2]);
                    sk_mma(acc[1], a, b0[1], b0[3]);
                    sk_mma(acc[2], a, b1[0], b1[2]);
                    sk_mma(acc[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
        #undef KQT_LOAD_A

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float s0 = (r0     < m_e) ? row_scale[lo + r0]     : 0.0f;
        const float s1 = (r0 + 8 < m_e) ? row_scale[lo + r0 + 8] : 0.0f;
        float* y0 = Y + (size_t)(lo + r0) * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                if(c     < out_f) y0[c]     = acc[nb][0] * s0;
                if(c + 1 < out_f) y0[c + 1] = acc[nb][1] * s0;
            }
            if(r0 + 8 < m_e){
                if(c     < out_f) y0[(size_t)8 * out_f + c]     = acc[nb][2] * s1;
                if(c + 1 < out_f) y0[(size_t)8 * out_f + c + 1] = acc[nb][3] * s1;
            }
        }
    }
}

// DSV4 matrix plain-decode gate-only fusion (MEMRA_F16G_GU_FUSE=1).  The
// one-token matrix route has m_e=1 on every non-empty local expert group.  Keep
// the shipped ModelOpt FP8-QAT -> normalized-f16 -> f32-MMA numeric program,
// but compute gate and up from the same A tile and apply the existing macro,
// clamp, SiLU and route-weight epilogue directly into H.  Down, intermediate
// FP8 quantization and original-slot scatter remain the common path.
//
// This is deliberately a narrow m_e=1 visitor.  It uses the same kq_fetch /
// kq_store and ascending sk_mma chain as moe_kq_sktail_kernel, so the two
// projection accumulators are separately bit-comparable before the epilogue.
// QT is fixed to QT_NVFP4_MODELOPT by the C entry point.
__device__ __forceinline__ float kq_gu_sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

template<int QT, bool M1 = false, bool PackedStore = false>
static __global__ void __launch_bounds__(128)
moe_kq_sktail_gu_kernel(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ H, const float* __restrict__ row_scale,
        const float* __restrict__ macro_g, const float* __restrict__ macro_u,
        const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit){
    if(QT != QT_NVFP4_MODELOPT) return;
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SK_BN][SKT_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    extern __shared__ uint32_t packed_h2[];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    // m_e=1 only.  Empty and larger groups are left for the rollback visitor.
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, 1, 2);
    if(total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nkb  = in_f / SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        if(m_e != 1) continue;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;
        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const uint8_t* Wg = (const uint8_t*)table[(size_t)0 * n_expert + eid];
        const uint8_t* Sg = (const uint8_t*)table[(size_t)1 * n_expert + eid];
        const uint8_t* Wu = (const uint8_t*)table[(size_t)4 * n_expert + eid];
        const uint8_t* Su = (const uint8_t*)table[(size_t)5 * n_expert + eid];
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* gwrow = Wg + (size_t)bn * row_bytes;
        const uint8_t* gsrow = Sg + (size_t)bn * (in_f / 16);
        const uint8_t* uwrow = Wu + (size_t)bn * row_bytes;
        const uint8_t* usrow = Su + (size_t)bn * (in_f / 16);

        const __half* agp[2]; int asr[2], asc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 128;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        #define KQGU_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) if(!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        KQGU_LOAD_A(0, 0);
        if(nkb > 1) KQGU_LOAD_A(1, SKT_BK);
        KqRaw gb0 = kq_fetch<QT>(gwrow, gsrow, bc0, s_cb, in_f);
        KqRaw gb1 = kq_fetch<QT>(gwrow, gsrow, bc0 + 16, s_cb, in_f);
        KqRaw ub0 = kq_fetch<QT>(uwrow, usrow, bc0, s_cb, in_f);
        KqRaw ub1 = kq_fetch<QT>(uwrow, usrow, bc0 + 16, s_cb, in_f);
        float ag[4][4] = {}, au[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SKT_STAGES;
            if(kb + 2 < nkb) KQGU_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);

            // Gate B tile and its next fetch.  The A tile is shared by both
            // projections; this is the only new fusion relative to the tail.
            kq_store_variant<QT, PackedStore>(gb0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(gb1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                gb0 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                gb1 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if(kb + 2 < nkb)      asm volatile("cp.async.wait_group 2;");
            else if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else                  asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if(!M1 || ((warp & 1) == 0)) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(ag[0], a, b0[0], b0[2]);
                    sk_mma(ag[1], a, b0[1], b0[3]);
                    sk_mma(ag[2], a, b1[0], b1[2]);
                    sk_mma(ag[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();

            // Up uses the same B tile and A tile after gate has consumed it.
            kq_store_variant<QT, PackedStore>(ub0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(ub1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                ub0 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                ub1 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            __syncthreads();
            if(!M1 || ((warp & 1) == 0)) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(au[0], a, b0[0], b0[2]);
                    sk_mma(au[1], a, b0[1], b0[3]);
                    sk_mma(au[2], a, b1[0], b1[2]);
                    sk_mma(au[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
        #undef KQGU_LOAD_A

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const int pair = lo + r0;
        const float rs = (r0 < m_e) ? row_scale[pair] : 0.0f;
        const float mg = (r0 < m_e) ? macro_g[pair] : 0.0f;
        const float mu = (r0 < m_e) ? macro_u[pair] : 0.0f;
        const float rw = (r0 < m_e) ? route_w[pair] : 0.0f;
        float* hrow = H + (size_t)pair * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(r0 < m_e){
                #define KQGU_STORE(col, ga, ua) do { \
                    if((col) < out_f){ \
                        float g = __fmul_rn(__fmul_rn((ga), rs), mg); \
                        float u = __fmul_rn(__fmul_rn((ua), rs), mu); \
                        u = fminf(fmaxf(u, -limit), limit); \
                        g = fminf(g, limit); \
                        float hv = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u); \
                        hrow[(col)] = __fmul_rn(hv, rw); \
                    } \
                } while(0)
                KQGU_STORE(c,     ag[nb][0], au[nb][0]);
                KQGU_STORE(c + 1, ag[nb][1], au[nb][1]);
                #undef KQGU_STORE
            }
        }
    }
}

// Numeric class moe_m1_adaptive_splitk_f32_fixed_order. The device CSR live
// count determines the split count. Integer K-block boundaries partition all K
// exactly, including non-divisors such as 13 slices. MMA order within each slice
// and ascending f32 partial reduction are fixed functions of the routed shape.
__host__ __device__ __forceinline__ int moe_m1_slices(int slots, int out_f, int gu){
    const int tiles = slots * (out_f / SK_BN);
    if(tiles <= 0) return 1;
    const int target = gu ? 1280 : 768;
    if(tiles >= target) return 1;
    const int slices = (target + tiles / 2) / tiles;
    return slices < 1 ? 1 : (slices > 16 ? 16 : slices);
}

template<int Projections>
static __device__ __forceinline__ void
moe_m1_splitk_partial_body(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ partial,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f){
    constexpr int QT = QT_NVFP4_MODELOPT;
    constexpr bool M1 = true, PackedStore = true;
    __shared__ __align__(16) __half As[SKT_STAGES][16][SKT_STRIDE];
    // Only warps 0/2 consume A rows 0..15 in M1; rows 16..31 were unused.
    // Removing those stages lowers static shared storage without changing MMA.
    __shared__ __align__(16) __half Bs[SK_BN][SKT_STRIDE];   // single buffer
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    __shared__ uint32_t packed_h2[256];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    __syncthreads();
    const int live = ex_off[n_active];
    const int slices = moe_m1_slices(live, out_f, Projections == 2);
    const int split = blockIdx.x % slices;
    const int tile = blockIdx.x / slices;
    const int slot = tile / ntx;
    if(slot >= live) return;

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;                        // 0..127
    const int total_kb = in_f / SKT_BK;
    const int begin_kb = split * total_kb / slices;
    const int end_kb = (split + 1) * total_kb / slices;
    const int nkb = end_kb - begin_kb;
    const int kbase = begin_kb * SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;          // 2 threads/row, 32 values each

    {
        // M=1 makes the CSR row offsets the tile map. Seven binary-search
        // reads replace rebuilding a 128-expert prefix in every slice CTA.
        const int g = sk_tile_group(ex_off, n_active, slot);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        const int m0 = 0;
        const int n0 = (tile % ntx) * SK_BN;

        const __half* Ag = A + (size_t)lo * in_f + kbase;
        const int eid = ex_ids[g];
        for(int p = 0; p < Projections; ++p){
        const int qplane = Projections == 2 ? p * 4 : 2 * proj;
        const uint8_t* Wq = (const uint8_t*)table[(size_t)qplane*n_expert + eid];
        const uint8_t* Wsc = (QT == QT_NVFP4_MODELOPT)
            ? (const uint8_t*)table[(size_t)(qplane + 1)*n_expert + eid] : nullptr;
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* wrow = Wq + (size_t)bn * row_bytes + kbase / 2;
        const uint8_t* scrow = Wsc ? Wsc + (size_t)bn * (in_f / 16) + kbase / 16 : nullptr;

        const __half* agp[2]; int asr[2], asc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 128;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        #define KQT_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) if(!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        KQT_LOAD_A(0, 0);
        if(nkb > 1) KQT_LOAD_A(1, SKT_BK);
        KqRaw braw0 = kq_fetch<QT>(wrow, scrow, bc0, s_cb, in_f);        // kb=0 windows
        KqRaw braw1 = kq_fetch<QT>(wrow, scrow, bc0 + 16, s_cb, in_f);

        float acc[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SKT_STAGES;
            if(kb + 2 < nkb) KQT_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);
            // B tile for THIS kb from the pre-fetched registers (previous kb's trailing
            // __syncthreads fences the overwrite), then issue kb+1's raw fetches so those
            // global reads fly behind this kb's mma.
            kq_store_variant<QT, PackedStore>(braw0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(braw1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                braw0 = kq_fetch<QT>(wrow, scrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                braw1 = kq_fetch<QT>(wrow, scrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if(kb + 2 < nkb)      asm volatile("cp.async.wait_group 2;");
            else if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else                  asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if(!M1 || ((warp & 1) == 0)) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(acc[0], a, b0[0], b0[2]);
                    sk_mma(acc[1], a, b0[1], b0[3]);
                    sk_mma(acc[2], a, b1[0], b1[2]);
                    sk_mma(acc[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
        #undef KQT_LOAD_A

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        if(r0 == 0){
            float* dst = partial + ((size_t(lo) * Projections + p) * 16 + split) * out_f;
            #pragma unroll
            for(int nb = 0; nb < 4; ++nb){
                const int c = cb + nb * 8;
                dst[c] = acc[nb][0];
                dst[c + 1] = acc[nb][1];
            }
        }
        __syncthreads();
        } // projection

    }
}

template<int Projections>
static __global__ void __launch_bounds__(128)
moe_m1_splitk_partial_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ partial,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f){
    moe_m1_splitk_partial_body<Projections>(table, proj, n_expert, ex_ids, row_bytes, A, partial, ex_off, n_active, in_f, out_f);
}

template<int Projections>
static __global__ void __launch_bounds__(128)
moe_m1_graph_splitk_partial_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ partial,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f){
    // Uniform CTA exit before shared-memory staging or synchronization.
    const int live = ex_off[n_active];
    const int slices = moe_m1_slices(live, out_f, Projections == 2);
    if(blockIdx.x >= live * ((out_f + SK_BN - 1) / SK_BN) * slices) return;
    moe_m1_splitk_partial_body<Projections>(table, proj, n_expert, ex_ids, row_bytes, A, partial, ex_off, n_active, in_f, out_f);
}


template<int Projections>
static __device__ __forceinline__ void moe_m1_splitk_reduce_body(
        const float* partial, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        int slots, int out_f, float limit, const int* ex_off, int n_active){
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= slots * out_f) return;
    const int row = c / out_f, col = c % out_f;
    // With route readback disabled, slots is only a launch upper bound.
    // The device CSR endpoint owns the true count; never consume tail scratch.
    if(row >= ex_off[n_active]){ out[c] = 0.0f; return; }
    const int slices = moe_m1_slices(ex_off[n_active], out_f, Projections == 2);
    float v[Projections] = {};
    #pragma unroll
    for(int p = 0; p < Projections; ++p){
        #pragma unroll
        for(int split = 0; split < slices; ++split)
            v[p] = __fadd_rn(v[p], partial[((size_t(row) * Projections + p) * 16 + split) * out_f + col]);
    }
    if(Projections == 1){ out[c] = __fmul_rn(v[0], row_scale[row]); }
    else {
        float g = __fmul_rn(__fmul_rn(v[0], row_scale[row]), macro_g[row]);
        float u = __fmul_rn(__fmul_rn(v[1], row_scale[row]), macro_u[row]);
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
        out[c] = __fmul_rn(__fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u), route_w[row]);
    }
}

template<int Projections>
static __global__ void moe_m1_splitk_reduce_kernel(
        const float* partial, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        int slots, int out_f, float limit, const int* ex_off, int n_active){
    moe_m1_splitk_reduce_body<Projections>(partial, out, row_scale, macro_g, macro_u, route_w, slots, out_f, limit, ex_off, n_active);
}

template<int Projections>
static __global__ void moe_m1_graph_splitk_reduce_kernel(
        const float* partial, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        int slots, int out_f, float limit, const int* ex_off, int n_active){
    moe_m1_splitk_reduce_body<Projections>(partial, out, row_scale, macro_g, macro_u, route_w, slots, out_f, limit, ex_off, n_active);
}

// Baseline instrument: same bodies until the per-live bandwidth cell is measured.
template<int Projections>
static __global__ void __launch_bounds__(128)
moe_m1_splitk_fast_partial_kernel(
        const unsigned long long* __restrict__ table, int proj, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ partial,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f){
    // Uniform CTA exit before shared-memory staging or synchronization.
    const int live = ex_off[n_active];
    const int slices = moe_m1_slices(live, out_f, Projections == 2);
    if(blockIdx.x >= live * ((out_f + SK_BN - 1) / SK_BN) * slices) return;
    moe_m1_splitk_partial_body<Projections>(table, proj, n_expert, ex_ids, row_bytes, A, partial, ex_off, n_active, in_f, out_f);
}

template<int Projections>
static __global__ void moe_m1_splitk_fast_reduce_kernel(
        const float* partial, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        int slots, int out_f, float limit, const int* ex_off, int n_active){
    moe_m1_splitk_reduce_body<Projections>(partial, out, row_scale, macro_g, macro_u, route_w, slots, out_f, limit, ex_off, n_active);
}

// Per-QT direct-from-quant launch (lane/iq-direct-loaders: the Q4_K/Q6_K if/else ladder
// became a template — each instantiation keeps its OWN occupancy/attribute statics, since
// the bodies register-differ per quant class). Guards live in memra_moe_kq_gemm_sk.
template<int QT>
static int moe_kq_gemm_sk_launch(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, const void* act_f16, float* y,
        const float* row_scale, const int* ex_off_dev, const int* ex_off_host,
        int n_active, int max_m, int in_f, int out_f, int cross,
        int tail, long row_bytes, cudaStream_t st){
    // Per-instantiation device caps: occ128 -2 unprobed, -1 device-unfit -> this qtype's
    // large groups ride the 32x64 form. occt: the deep-tail direct form; -1 = probe failed.
    // PER-DEVICE probes (2026-08-27, TP2 grouped prime). These were single statics probed on
    // whichever device was current FIRST — and cudaFuncSetAttribute's smem opt-in is
    // per-function-PER-CONTEXT, so the 128-form launch on the second rank's context ran with
    // no opt-in: cudaErrorInvalidValue, the gemm-prime rc=1001. Probe and opt in once per device.
    enum { SK_MAX_DEV = 16 };
    static int sms_d[SK_MAX_DEV] = {0}, occ32_d[SK_MAX_DEV] = {0}, occt_d[SK_MAX_DEV] = {0};
    static int occ128_d[SK_MAX_DEV];
    static int occ128_init = 0;
    if(!occ128_init){ for(int i = 0; i < SK_MAX_DEV; i++) occ128_d[i] = -2; occ128_init = 1; }
    int cur_dev = 0; cudaGetDevice(&cur_dev);
    if(cur_dev < 0 || cur_dev >= SK_MAX_DEV) cur_dev = 0;
    int sms, occ32, occ128, occt;
    if(sms_d[cur_dev] == 0){
        if(cudaDeviceGetAttribute(&sms_d[cur_dev], cudaDevAttrMultiProcessorCount, cur_dev)
           != cudaSuccess || sms_d[cur_dev] <= 0)
            sms_d[cur_dev] = 1;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ32_d[cur_dev], moe_kq_sk32v_kernel<QT>, 128, 0)
           != cudaSuccess || occ32_d[cur_dev] < 1) occ32_d[cur_dev] = 1;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occt_d[cur_dev], moe_kq_sktail_kernel<QT>, 128, 0)
           != cudaSuccess || occt_d[cur_dev] < 1) occt_d[cur_dev] = -1;
    }
    // MEMRA_F16G_BDB (lane/kq-bdb 2026-08-28): double-buffered B in the 128 form. Same tiles and
    // same accumulation order as the shipped schedule — one barrier per k-block instead of two,
    // and kb+1's weight dequant overlaps this kb's mma. The extra 9 KB of smem does not move
    // occ128 off 1. DEFAULT OFF until this lane's A/B lands, per the new-flag law; both
    // instantiations stay compiled so the arms are one env var apart.
    static int bdb = -1;
    if(bdb < 0){
        const char* e = getenv("MEMRA_F16G_BDB");
        bdb = (e && e[0] == '1') ? 1 : 0;
    }
    // Both instantiations get the smem opt-in: cudaFuncSetAttribute is per-FUNCTION and
    // per-context, and the two template arms are different functions (the rc=1001 lesson).
    if(occ128_d[cur_dev] == -2){
        int optin = 0;
        if(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cur_dev) != cudaSuccess)
            optin = 48*1024;
        cudaError_t ae = cudaFuncSetAttribute(moe_kq_sk128v_kernel<QT, false>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, KQ128_SMEM_BYTES);
        cudaError_t ae2 = cudaFuncSetAttribute(moe_kq_sk128v_kernel<QT, true>,
                 cudaFuncAttributeMaxDynamicSharedMemorySize, KQ128_SMEM_BYTES);
        cudaError_t oe = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ128_d[cur_dev],
                 moe_kq_sk128v_kernel<QT, true>, 256, KQ128_SMEM_BYTES);
        if((size_t)KQ128_SMEM_BYTES > (size_t)optin || ae != cudaSuccess || ae2 != cudaSuccess
           || oe != cudaSuccess || occ128_d[cur_dev] < 1)
            occ128_d[cur_dev] = -1;
    }
    sms = sms_d[cur_dev]; occ32 = occ32_d[cur_dev]; occ128 = occ128_d[cur_dev]; occt = occt_d[cur_dev];
    int xcross = cross;
    if(occ128 < 1 || (in_f % SK128_BK) != 0) xcross = 0x7fffffff;
    if(xcross < 1) xcross = 1;
    // WHICH FORM ACTUALLY RAN (2026-08-28): the prime measures ~13.6 TFLOP/s/rank in the MoE,
    // ~10x under what this mma.m16n8k16 3-stage pipeline should reach. A failed smem opt-in
    // silently forces every group onto the 32-row form (4x redundant weight dequant at the
    // m_e~114 of a 4096-token chunk), and the fallback is invisible from outside: occ128 = -1
    // reads exactly like a deliberate shape choice. One line per qtype (the instantiation
    // statics above are already per-QT), so it cannot become log noise.
    {
        static int said = 0;
        if(!said){
            said = 1;
            fprintf(stderr, "[moe-sk-form] dev=%d qt=%d sms=%d occ128=%d occ32=%d occt=%d "
                    "cross=%d xcross=%d bdb=%d in_f=%d out_f=%d n_active=%d max_m=%d\n",
                    cur_dev, QT, sms, occ128, occ32, occt, cross, xcross, bdb, in_f, out_f,
                    n_active, max_m);
        }
    }
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    long t32 = 0, t128 = 0;
    if (ex_off_host == nullptr) {
        // The device prefix defines the exact work count. Fixed persistent grids
        // permit replay with changed routing without reading counts on the CPU.
        if (xcross < 0x7fffffff) {
            const int resident = sms * occ128;
            const int grid = resident;
            if (bdb)
                moe_kq_sk128v_kernel<QT, true><<<grid, dim3(32,8,1), KQ128_SMEM_BYTES, st>>>(
                    table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
                    row_scale, ex_off_dev, n_active, in_f, out_f, xcross,
                    0x7fffffff, -1);
            else
                moe_kq_sk128v_kernel<QT, false><<<grid, dim3(32,8,1), KQ128_SMEM_BYTES, st>>>(
                    table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
                    row_scale, ex_off_dev, n_active, in_f, out_f, xcross,
                    0x7fffffff, -1);
        }
        if (xcross > 1) {
            const int deep = tail != 0 && occt >= 1 && in_f % SKT_BK == 0;
            const int resident = sms * (deep ? occt : occ32);
            const int grid = resident;
            if (deep)
                moe_kq_sktail_kernel<QT><<<grid, dim3(32,4,1), 0, st>>>(
                    table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
                    row_scale, ex_off_dev, n_active, in_f, out_f, 1,
                    xcross, -1);
            else
                moe_kq_sk32v_kernel<QT><<<grid, dim3(32,4,1), 0, st>>>(
                    table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
                    row_scale, ex_off_dev, n_active, in_f, out_f, 1,
                    xcross, -1);
        }
        cudaError_t err = cudaGetLastError();
        return err ? 1000 + (int)err : 0;
    }
    for(int g = 0; g < n_active; g++){
        const int m_e = ex_off_host[g+1] - ex_off_host[g];
        if(m_e <= 0) continue;
        if(m_e >= xcross) t128 += (long)((m_e + SK128_BM - 1)/SK128_BM) * ntx;
        else              t32  += (long)((m_e + SK_BM - 1)/SK_BM) * ntx;
    }
    if(t128 > 0){
        const long cap = (long)sms * occ128;
        const int grid = (int)(t128 < cap ? t128 : cap);
        if(bdb)
            moe_kq_sk128v_kernel<QT, true><<<grid, dim3(32,8,1), KQ128_SMEM_BYTES, st>>>(
                table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y, row_scale,
                ex_off_dev, n_active, in_f, out_f, xcross,
                0x7fffffff, (int)t128);
        else
            moe_kq_sk128v_kernel<QT, false><<<grid, dim3(32,8,1), KQ128_SMEM_BYTES, st>>>(
                table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y, row_scale,
                ex_off_dev, n_active, in_f, out_f, xcross,
                0x7fffffff, (int)t128);
    }
    if(t32 > 0){
        // Deep tail (lane/sk-tail-form) when admitted; in_f % 256 == 0 here so the
        // in_f % 64 requirement always holds — the guard is kept for form.
        const int deep = (tail != 0 && occt >= 1 && (in_f % SKT_BK) == 0);
        const long cap = (long)sms * (deep ? occt : occ32);
        const int grid = (int)(t32 < cap ? t32 : cap);
        if(deep)
            moe_kq_sktail_kernel<QT><<<grid, dim3(32,4,1), 0, st>>>(
                table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y, row_scale,
                ex_off_dev, n_active, in_f, out_f, 1,
                xcross, (int)t32);
        else
            moe_kq_sk32v_kernel<QT><<<grid, dim3(32,4,1), 0, st>>>(
                table, proj, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y, row_scale,
                ex_off_dev, n_active, in_f, out_f, 1,
                xcross, (int)t32);
    }
    cudaError_t e=cudaGetLastError();
    if(e) fprintf(stderr, "[moe-sk-err] kq_sk err=%d(%s) n_active=%d max_m=%d in_f=%d out_f=%d\n",
                  (int)e, cudaGetErrorString(e), n_active, max_m, in_f, out_f);
    return e?1000+(int)e:0;
}

extern "C" {

size_t memra_moe_f16g_w_bytes(int n_active, int out_f, int in_f){
    return (size_t)n_active*out_f*in_f*sizeof(__half);
}
size_t memra_moe_f16g_act_bytes(int n_pairs, int in_f){
    return (size_t)n_pairs*in_f*sizeof(__half);
}

int memra_moe_f16g_dequant(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, void* w_f16, int in_f, int out_f, int n_active,
        int qtype, long row_bytes, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)out_f, (unsigned)n_active, 1), blk(256,1,1);
    if(qtype==QT_NVFP4_V2)
        dequant_nvfp4v2_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q4_0)
        dequant_q4_0_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_IQ4_XS)
        dequant_iq4xs_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_IQ3_S)
        dequant_iq3s_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q6_K)
        dequant_q6k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q4_K)
        dequant_q4k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_Q3_K)
        dequant_q3k_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else if(qtype==QT_NVFP4)
        dequant_nvfp4_f16_kernel<<<grid,blk,0,st>>>(table,proj,n_expert,ex_ids,(__half*)w_f16,in_f,out_f,row_bytes);
    else return 2;   // unsupported qtype => caller falls back to the MMQ arm
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// Single-kernel grouped GEMM (MEMRA_MOE_F16G=2): every CSR group's GEMM on the caller's stream,
// f32 C with the per-pair act scale folded in. ex_off_dev = DEVICE CSR offsets (n_active+1);
// ex_off_host = the same offsets on the HOST (already there at the call site — sizes the visitor
// grids and skips empty arms with no extra transfer); max_m = host-side max group size.
// in_f must be a multiple of 32.
// shape_sel < 0  -> the round-49 grid-scan kernel (rollback arm, MEMRA_F16G_SK=0).
// shape_sel >= 0 -> problem-visitor: groups with m_e >= cross ride the 128x64x64 3-stage form,
//                   smaller groups the 32x64 tail (cross=1 forces all-128, INT_MAX all-32).
//                   Policy/env parsing lives on the Rust side (moe_f16g_sk_params).
// tail != 0      -> the sub-cross groups ride the DEEP tail (32x64x64 3-stage,
//                   lane/sk-tail-form) when in_f % 64 == 0 and the device fits it;
//                   tail == 0 (MEMRA_F16G_TAIL=0) = the round-51 2-stage 32x64x32 tail.
//                   Every arm is byte-identical (kernel-check "f16g-sk").
int memra_moe_f16g_gemm_sk(const void* w_f16, const void* act_f16, float* y,
        const float* row_scale, const int* ex_off_dev, const int* ex_off_host,
        int n_active, int max_m, int in_f, int out_f, int shape_sel, int cross,
        int tail, void* stream){
    if(in_f % SK_BK) return 2;
    if(n_active <= 0 || max_m <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    if(shape_sel < 0 || n_active > SK_MAX_G || ex_off_host == nullptr){
        // round-49 form: grid (ntx, ceil(max_m/BM), n_active), early-exit past a group's pairs.
        dim3 grid((unsigned)((out_f + SK_BN - 1) / SK_BN),
                  (unsigned)((max_m + SK_BM - 1) / SK_BM),
                  (unsigned)n_active);
        dim3 blk(32, 4, 1);
        moe_f16g_sk_kernel<<<grid, blk, 0, st>>>(
            (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev, in_f, out_f);
        cudaError_t e=cudaGetLastError();
    if(e) fprintf(stderr, "[moe-sk-err] memra_moe_f16g_gemm_sk err=%d(%s) n_active=%d max_m=%d in_f=%d out_f=%d\n",
                  (int)e, cudaGetErrorString(e), n_active, max_m, in_f, out_f);
    return e?1000+(int)e:0;
    }
    // Device caps + per-form occupancy, once. occ128: -2 unprobed, -1 device-unfit (fallback to
    // the 32x64 form — the round-49 rc=1001 lesson: SetAttribute CHECKED, never assumed).
    // occt: the deep-tail form (static 43524 B smem — no opt-in needed); -1 = probe failed,
    // tail groups keep the 2-stage form.
    // PER-DEVICE probes — same fix and reason as the kq template above (the smem opt-in is
    // per-function-PER-CONTEXT; a second rank's context launched the 128-form without it:
    // cudaErrorInvalidValue, the gemm-prime rc=1001).
    enum { F16G_MAX_DEV = 16 };
    static int sms_d[F16G_MAX_DEV] = {0}, occ32_d[F16G_MAX_DEV] = {0}, occt_d[F16G_MAX_DEV] = {0};
    static int occ128_d[F16G_MAX_DEV];
    static int occ128_init = 0;
    if(!occ128_init){ for(int i = 0; i < F16G_MAX_DEV; i++) occ128_d[i] = -2; occ128_init = 1; }
    int cur_dev = 0; cudaGetDevice(&cur_dev);
    if(cur_dev < 0 || cur_dev >= F16G_MAX_DEV) cur_dev = 0;
    int sms, occ32, occ128, occt;
    if(sms_d[cur_dev] == 0){
        if(cudaDeviceGetAttribute(&sms_d[cur_dev], cudaDevAttrMultiProcessorCount, cur_dev)
           != cudaSuccess || sms_d[cur_dev] <= 0)
            sms_d[cur_dev] = 1;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ32_d[cur_dev], moe_f16g_sk32v_kernel, 128, 0)
           != cudaSuccess || occ32_d[cur_dev] < 1) occ32_d[cur_dev] = 1;
        if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occt_d[cur_dev], moe_f16g_sktail_kernel, 128, 0)
           != cudaSuccess || occt_d[cur_dev] < 1) occt_d[cur_dev] = -1;
    }
    if(occ128_d[cur_dev] == -2){
        int optin = 0;
        if(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cur_dev) != cudaSuccess)
            optin = 48*1024;
        if((size_t)SK128_SMEM_BYTES > (size_t)optin
           || cudaFuncSetAttribute(moe_f16g_sk128v_kernel,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, SK128_SMEM_BYTES) != cudaSuccess
           || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ128_d[cur_dev], moe_f16g_sk128v_kernel, 256,
                  SK128_SMEM_BYTES) != cudaSuccess
           || occ128_d[cur_dev] < 1)
            occ128_d[cur_dev] = -1;
    }
    sms = sms_d[cur_dev]; occ32 = occ32_d[cur_dev]; occ128 = occ128_d[cur_dev]; occt = occt_d[cur_dev];
    int xcross = cross;
    if(occ128 < 1 || (in_f % SK128_BK) != 0) xcross = 0x7fffffff;  // every group rides 32x64
    if(xcross < 1) xcross = 1;
    const int ntx = (out_f + SK_BN - 1) / SK_BN;                   // BN identical in both forms
    long t32 = 0, t128 = 0;
    for(int g = 0; g < n_active; g++){
        const int m_e = ex_off_host[g+1] - ex_off_host[g];
        if(m_e <= 0) continue;
        if(m_e >= xcross) t128 += (long)((m_e + SK128_BM - 1)/SK128_BM) * ntx;
        else              t32  += (long)((m_e + SK_BM - 1)/SK_BM) * ntx;
    }
    if(t128 > 0){
        const long cap = (long)sms * occ128;
        const int grid = (int)(t128 < cap ? t128 : cap);
        moe_f16g_sk128v_kernel<<<grid, dim3(32,8,1), SK128_SMEM_BYTES, st>>>(
            (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev,
            n_active, in_f, out_f, xcross, 0x7fffffff, (int)t128);
    }
    if(t32 > 0){
        // Deep tail (lane/sk-tail-form) when admitted; identical tile list either way
        // (same BM/BN -> same t32), so the arms are drop-in interchangeable.
        const int deep = (tail != 0 && occt >= 1 && (in_f % SKT_BK) == 0);
        const long cap = (long)sms * (deep ? occt : occ32);
        const int grid = (int)(t32 < cap ? t32 : cap);
        if(deep)
            moe_f16g_sktail_kernel<<<grid, dim3(32,4,1), 0, st>>>(
                (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev,
                n_active, in_f, out_f, 1, xcross, (int)t32);
        else
            moe_f16g_sk32v_kernel<<<grid, dim3(32,4,1), 0, st>>>(
                (const __half*)w_f16, (const __half*)act_f16, y, row_scale, ex_off_dev,
                n_active, in_f, out_f, 1, xcross, (int)t32);
    }
    cudaError_t e=cudaGetLastError();
    if(e) fprintf(stderr, "[moe-sk-err] f16g_sk err=%d(%s) n_active=%d max_m=%d in_f=%d out_f=%d\n",
                  (int)e, cudaGetErrorString(e), n_active, max_m, in_f, out_f);
    return e?1000+(int)e:0;
}

// DIRECT-FROM-QUANT grouped GEMM (lane/kquant-tile-loaders + lane/iq-direct-loaders): the
// visitor forms above with the B side dequanted in-register from the expert superblocks —
// no f16 dequant workspace exists. Bit-identical to memra_moe_f16g_gemm_sk on the dequant
// workspace by construction (kernel-check "f16g-kq-direct" gates it bitwise).
// qtype: 1=Q4_K, 2=Q6_K, 5=IQ4_XS, 6=IQ3_S.
// Visitor forms only (rc=2 on anything else — the caller keeps the workspace path):
// the grid-scan rollback arm (MEMRA_F16G_SK=0) stays on the workspace path unchanged.
int memra_moe_kq_gemm_sk(const unsigned long long* table, int proj, int n_expert,
        const int* ex_ids, const void* act_f16, float* y,
        const float* row_scale, const int* ex_off_dev, const int* ex_off_host,
        int n_active, int max_m, int in_f, int out_f, int qtype, int cross,
        int tail, long row_bytes, void* stream){
    // whole superblocks per k walk: 256-value superblocks for the kq/IQ classes,
    // 64-value blocks for NVFP4 (its 16-value window is one UE4M3 sub-block).
    if(in_f % ((qtype == QT_NVFP4 || qtype == QT_NVFP4_V2
                || qtype == QT_NVFP4_MODELOPT) ? 64 : 256)) return 2;
    if(qtype != QT_Q4_K && qtype != QT_Q6_K && qtype != QT_IQ4_XS && qtype != QT_IQ3_S
       && qtype != QT_NVFP4 && qtype != QT_NVFP4_V2 && qtype != QT_NVFP4_MODELOPT)
        return 2;
    // Device-owned metadata is initially admitted only for the DSV4 split-plane
    // ModelOpt path; every existing generic caller keeps its host-count ABI.
    if(n_active > SK_MAX_G || (ex_off_host == nullptr && qtype != QT_NVFP4_MODELOPT)) return 2;
    if(n_active <= 0 || max_m <= 0) return 0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    switch(qtype){
        case QT_NVFP4:
            return moe_kq_gemm_sk_launch<QT_NVFP4>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        case QT_NVFP4_V2:
            return moe_kq_gemm_sk_launch<QT_NVFP4_V2>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        case QT_NVFP4_MODELOPT:
            return moe_kq_gemm_sk_launch<QT_NVFP4_MODELOPT>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        case QT_Q4_K:
            return moe_kq_gemm_sk_launch<QT_Q4_K>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        case QT_Q6_K:
            return moe_kq_gemm_sk_launch<QT_Q6_K>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        case QT_IQ4_XS:
            return moe_kq_gemm_sk_launch<QT_IQ4_XS>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
        default:
            return moe_kq_gemm_sk_launch<QT_IQ3_S>(table, proj, n_expert, ex_ids, act_f16, y,
                row_scale, ex_off_dev, ex_off_host, n_active, max_m, in_f, out_f, cross,
                tail, row_bytes, st);
    }
}

// Compile-only DSV4 candidate: tensor-core m_e=1 tail.  This is NOT the
// rejected scalar m=1 visitor and does not change the active dispatch.  It
// retains the same f16 operands, ModelOpt NVFP4 kq_fetch/kq_store program and
// valid-row m16n8k16 chain as the shipped tail, while skipping the two warps
// whose 16-row output tiles are provably outside an m_e=1 group.  The ordinary
// FP8 mirror and scatter remain in the caller for the exactness comparison.
int memra_moe_kq_gemm_sk_m1(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* y, const float* row_scale,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !y || !row_scale || !ex_off_dev
       || n_expert <= 0 || n_active <= 0 || n_active > SK_MAX_G
       || in_f <= 0 || out_f <= 0 || in_f % SKT_BK || out_f % SK_BN
       || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true>, 128, 0) != cudaSuccess
       || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true><<<grid, dim3(32,4,1), 0, st>>>(
        table, 1, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
        row_scale, ex_off_dev, n_active, in_f, out_f, 1, 2, -1);
    cudaError_t e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

// Gate/up + weighted SwiGLU fused matrix decode arm.  The caller admits this
// only for one-row transactions, so every non-empty CSR group has m_e=1 and
// the device prefix is sufficient; no host route-count readback is needed.
// Down projection, intermediate FP8 quantization and scatter remain separate.
int memra_moe_kq_gemm_sk_gu(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* h, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        float limit, long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !h || !row_scale || !macro_g || !macro_u
       || !route_w || !ex_off_dev || n_expert <= 0 || n_active <= 0
       || n_active > SK_MAX_G || in_f <= 0 || out_f <= 0 || in_f % SKT_BK
       || out_f % SK_BN || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT>, 128, 0) != cudaSuccess
       || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT><<<grid, dim3(32,4,1), 0, st>>>(
        table, n_expert, ex_ids, row_bytes, (const __half*)act_f16, h, row_scale,
        macro_g, macro_u, route_w, ex_off_dev, n_active, in_f, out_f, -1, limit);
    cudaError_t e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

// GU m_e=1 tensor-core work-elision twin. The valid-row gate/up MMA chain and
// epilogue are unchanged; only the duplicate invalid-row A loads and MMA work
// are skipped. This is a gate-only launch selected by the Rust process-local
// override, never by an environment or serving path.
int memra_moe_kq_gemm_sk_gu_m1(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* h, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        float limit, long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !h || !row_scale || !macro_g || !macro_u
       || !route_w || !ex_off_dev || n_expert <= 0 || n_active <= 0
       || n_active > SK_MAX_G || in_f <= 0 || out_f <= 0 || in_f % SKT_BK
       || out_f % SK_BN || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true>, 128, 0) != cudaSuccess
       || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true><<<grid, dim3(32,4,1), 0, st>>>(
        table, n_expert, ex_ids, row_bytes, (const __half*)act_f16, h, row_scale,
        macro_g, macro_u, route_w, ex_off_dev, n_active, in_f, out_f, -1, limit);
    cudaError_t e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

// Packed ModelOpt store twin for the fused gate/up visitor.  The default GU launcher above keeps
// the original 16-entry f32 shared LUT and zero dynamic shared bytes; this launcher opts into only
// the 256-entry half2 table needed by kq_store_variant<PackedStore=true>.
int memra_moe_kq_gemm_sk_gu_half2(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* h, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        float limit, long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !h || !row_scale || !macro_g || !macro_u
       || !route_w || !ex_off_dev || n_expert <= 0 || n_active <= 0
       || n_active > SK_MAX_G || in_f <= 0 || out_f <= 0 || in_f % SKT_BK
       || out_f % SK_BN || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    constexpr size_t H2_SMEM_BYTES = 256 * sizeof(uint32_t);
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, false, true>, 128,
           H2_SMEM_BYTES) != cudaSuccess || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, false, true>
        <<<grid, dim3(32,4,1), H2_SMEM_BYTES, st>>>(
            table, n_expert, ex_ids, row_bytes, (const __half*)act_f16, h, row_scale,
            macro_g, macro_u, route_w, ex_off_dev, n_active, in_f, out_f, -1, limit);
    cudaError_t e = cudaGetLastError();
    if(!e) g_moe_kq_gu_h2_dispatches.fetch_add(1, std::memory_order_relaxed);
    return e ? 1000 + (int)e : 0;
}

// Composition of the existing GU-M1 work-elision and packed-half2 stores. This
// is the same one-row numeric program as the two gate launchers, instantiated
// as <QT_NVFP4_MODELOPT,true,true>; the caller advances both feature receipts
// for this one successful CUDA enqueue.
int memra_moe_kq_gemm_sk_gu_m1_half2(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* h, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        float limit, long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !h || !row_scale || !macro_g || !macro_u
       || !route_w || !ex_off_dev || n_expert <= 0 || n_active <= 0
       || n_active > SK_MAX_G || in_f <= 0 || out_f <= 0 || in_f % SKT_BK
       || out_f % SK_BN || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    constexpr size_t H2_SMEM_BYTES = 256 * sizeof(uint32_t);
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true, true>, 128,
           H2_SMEM_BYTES) != cudaSuccess || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true, true>
        <<<grid, dim3(32,4,1), H2_SMEM_BYTES, st>>>(
            table, n_expert, ex_ids, row_bytes, (const __half*)act_f16, h, row_scale,
            macro_g, macro_u, route_w, ex_off_dev, n_active, in_f, out_f, -1, limit);
    cudaError_t e = cudaGetLastError();
    if(!e) g_moe_kq_gu_h2_dispatches.fetch_add(1, std::memory_order_relaxed);
    return e ? 1000 + (int)e : 0;
}

// Packed ModelOpt store twin for the tensor-core m_e=1 down-tail candidate.  It is intentionally
// separate from memra_moe_kq_gemm_sk_m1 so the latter's static shared allocation and scalar
// dequant path remain untouched while the gate can compare both complete chains.
int memra_moe_kq_gemm_sk_m1_half2(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* y, const float* row_scale,
        const int* ex_off_dev, int n_active, int in_f, int out_f,
        long row_bytes, void* stream){
    if(!table || !ex_ids || !act_f16 || !y || !row_scale || !ex_off_dev
       || n_expert <= 0 || n_active <= 0 || n_active > SK_MAX_G
       || in_f <= 0 || out_f <= 0 || in_f % SKT_BK || out_f % SK_BN
       || row_bytes != in_f / 2)
        return 40004;
    int dev = 0, sms = 1, occ = 1;
    cudaGetDevice(&dev);
    if(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess
       || sms < 1) sms = 1;
    constexpr size_t H2_SMEM_BYTES = 256 * sizeof(uint32_t);
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
           &occ, moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true, true>, 128,
           H2_SMEM_BYTES) != cudaSuccess || occ < 1) occ = 1;
    const int grid = sms * occ;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true, true>
        <<<grid, dim3(32,4,1), H2_SMEM_BYTES, st>>>(
            table, 1, n_expert, ex_ids, row_bytes, (const __half*)act_f16, y,
            row_scale, ex_off_dev, n_active, in_f, out_f, 1, 2, -1);
    cudaError_t e = cudaGetLastError();
    if(!e) g_moe_kq_m1_h2_dispatches.fetch_add(1, std::memory_order_relaxed);
    return e ? 1000 + (int)e : 0;
}

// Scratch has slots * out_f * 16 * (gu ? 2 : 1) floats. The Rust caller
// admits only unique top-k M=1 routes; slots is an upper bound when route
// readback is disabled. Both kernels use the device CSR prefix as authority.
static int moe_m1_splitk_launch(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream, cudaEvent_t middle){
    if(!table || !ex_ids || !act_f16 || !out || !row_scale || !ex_off || !partial
       || n_expert < 1 || n_active < 1 || n_active > SK_MAX_G
       || slots < 1 || slots > n_active || (gu != 0 && gu != 1)
       || (gu && (!macro_g || !macro_u || !route_w))
       || (in_f != 4096 && in_f != 2048) || (out_f != 4096 && out_f != 2048))
        return 40004;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    // Host slots can be an upper bound, so launch the maximum over possible
    // live counts. The device chooses the actual split without a host drain.
    int grid = 1;
    for(int live = 1; live <= slots; ++live)
        grid = std::max(grid, live * (out_f / SK_BN) * moe_m1_slices(live, out_f, gu));
    if(gu){
        moe_m1_splitk_partial_kernel<2><<<grid, dim3(32,4), 0, st>>>(
            table, 0, n_expert, ex_ids, in_f/2, (const __half*)act_f16,
            partial, ex_off, n_active, in_f, out_f);
    } else {
        moe_m1_splitk_partial_kernel<1><<<grid, dim3(32,4), 0, st>>>(
            table, 1, n_expert, ex_ids, in_f/2, (const __half*)act_f16,
            partial, ex_off, n_active, in_f, out_f);
    }
    cudaError_t e = cudaGetLastError();
    if(e) return 1000 + (int)e;
    if(middle){ e = cudaEventRecord(middle,st); if(e) return 1000 + int(e); }
    if(gu) moe_m1_splitk_reduce_kernel<2><<<(slots*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,slots,out_f,limit,ex_off,n_active);
    else moe_m1_splitk_reduce_kernel<1><<<(slots*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,slots,out_f,limit,ex_off,n_active);
    e = cudaGetLastError();
    return e ? 1000 + (int)e : 0;
}

int memra_moe_m1_splitk(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream){
    return moe_m1_splitk_launch(table,n_expert,ex_ids,act_f16,out,row_scale,
        macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,slots,gu,partial,stream,nullptr);
}
// Gate override is changed only after both rank streams have drained.
static std::atomic<int> g_splitk_fast_override{-1};
static std::atomic<bool> g_splitk_fast_component_enabled{false};
void memra_moe_m1_splitk_fast_set_for_gate(int enabled){ g_splitk_fast_override.store(enabled); }
void memra_moe_m1_splitk_fast_component_set_for_gate(){ g_splitk_fast_component_enabled.store(true); }
int memra_moe_m1_splitk_fast_on(){
    const int override=g_splitk_fast_override.load();
    if(override>=0) return override;
    static const bool enabled=[](){
        const char* raw=std::getenv("MEMRA_DSV4_SPLITK_FAST");
        if(!raw || std::strcmp(raw,"0")==0) return false;
        if(std::strcmp(raw,"1")==0) return true;
        std::fprintf(stderr,"MEMRA_DSV4_SPLITK_FAST requires 0 or 1\n");
        std::abort();
    }();
    return enabled;
}
// Harness-only failures retain the returned status before inspecting/clearing
// CUDA's last-error slot. Normal production launches do not call this logger.
static int splitk_fast_cuda_failure(cudaError_t error, const char* call,
        const char* file, int line, const char* phase, int device, int gu, int live){
    const cudaError_t peek=cudaPeekAtLastError();
    const cudaError_t last=cudaGetLastError();
    fflush(stdout);
    fprintf(stderr,"SPLITK_FAST_CUDA_FAILURE phase=%s file=%s line=%d device=%d gu=%d live=%d call=%s returned=%d/%s/%s peek=%d/%s/%s last=%d/%s/%s\n",
        phase,file,line,device,gu,live,call,int(error),cudaGetErrorName(error),cudaGetErrorString(error),
        int(peek),cudaGetErrorName(peek),cudaGetErrorString(peek),
        int(last),cudaGetErrorName(last),cudaGetErrorString(last));
    fflush(stderr);
    return 1000+int(error);
}
// Fixed graph geometry. Neither launch depends on host routing observations.
static int moe_m1_splitk_fast_launch(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream, bool fast, cudaEvent_t middle=nullptr){
    if(!table || !ex_ids || !act_f16 || !out || !row_scale || !ex_off || !partial
       || n_expert < 1 || n_active < 1 || n_active > SK_MAX_G
       || slots != 6 || slots > n_active || (gu != 0 && gu != 1)
       || (gu && (!macro_g || !macro_u || !route_w))
       || (in_f != 4096 && in_f != 2048) || (out_f != 4096 && out_f != 2048))
        return 40004;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int grid = 6 * (out_f / SK_BN) * 16;
    if(gu) {
        if(fast) moe_m1_splitk_fast_partial_kernel<2><<<grid, dim3(32,4), 0, st>>>(
        table,0,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
        else moe_m1_graph_splitk_partial_kernel<2><<<grid, dim3(32,4), 0, st>>>(
        table,0,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
    }
    else {
        if(fast) moe_m1_splitk_fast_partial_kernel<1><<<grid, dim3(32,4), 0, st>>>(
        table,1,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
        else moe_m1_graph_splitk_partial_kernel<1><<<grid, dim3(32,4), 0, st>>>(
        table,1,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
    }
    cudaError_t e = cudaGetLastError();
    if(e){
        if(middle) return splitk_fast_cuda_failure(e,"cudaGetLastError after partial",__FILE__,__LINE__,"instrument_launch",-1,gu,-1);
        return 1000+int(e);
    }
    if(middle){ e=cudaEventRecord(middle,st); if(e) return splitk_fast_cuda_failure(e,"cudaEventRecord(middle,st)",__FILE__,__LINE__,"instrument_launch",-1,gu,-1); }
    if(gu) {
        if(fast) moe_m1_splitk_fast_reduce_kernel<2><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
        else moe_m1_graph_splitk_reduce_kernel<2><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
    }
    else {
        if(fast) moe_m1_splitk_fast_reduce_kernel<1><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
        else moe_m1_graph_splitk_reduce_kernel<1><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
    }
    e = cudaGetLastError();
    if(e && middle) return splitk_fast_cuda_failure(e,"cudaGetLastError after reducer",__FILE__,__LINE__,"instrument_launch",-1,gu,-1);
    return e ? 1000 + int(e) : 0;
}

// Fixed graph geometry. Neither launch depends on host routing observations.
int memra_moe_m1_graph_splitk(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream){
    return moe_m1_splitk_fast_launch(table,n_expert,ex_ids,act_f16,out,row_scale,
        macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,slots,gu,partial,stream,
        memra_moe_m1_splitk_fast_on());
}

int memra_moe_m1_splitk_fast_component(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream);

// Gate-only real-operand census: wait for a six-live route on each rank and
// projection, then replay all prefixes 0..6 without recapturing either kernel.
static std::atomic<unsigned> g_graph_splitk_component_mask{0};
unsigned memra_moe_m1_graph_splitk_component_mask(){ return g_graph_splitk_component_mask.load(); }
int memra_moe_m1_graph_splitk_component(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream){
    if(g_splitk_fast_component_enabled.load()) return memra_moe_m1_splitk_fast_component(
        table,n_expert,ex_ids,act_f16,out,row_scale,macro_g,macro_u,route_w,
        ex_off,n_active,in_f,out_f,limit,slots,gu,partial,stream);
    int device=0, observed=0;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    #define GS_CHECK(call) do { cudaError_t e=(call); if(e) return 1000+int(e); } while(0)
    GS_CHECK(cudaGetDevice(&device));
    const unsigned bit = 1u << (device*2+gu);
    if(g_graph_splitk_component_mask.load() & bit) return 0;
    GS_CHECK(cudaMemcpyAsync(&observed,ex_off+n_active,4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaStreamSynchronize(st));
    if(observed != 6) return 0;
    struct Buffers {
        float *a=nullptr,*b=nullptr,*p=nullptr;
        __half* activation=nullptr;
        int* offsets=nullptr;
        void* flush=nullptr;
        cudaEvent_t begin=nullptr,end=nullptr;
        cudaGraph_t graph=nullptr;
        cudaGraphExec_t exec=nullptr;
        ~Buffers(){ if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph);
            if(begin) cudaEventDestroy(begin); if(end) cudaEventDestroy(end);
            if(a) cudaFree(a); if(b) cudaFree(b); if(p) cudaFree(p);
            if(activation) cudaFree(activation);
            if(offsets) cudaFree(offsets); if(flush) cudaFree(flush); }
    } b;
    const size_t n=size_t(6)*out_f, pn=n*16*(gu?2:1), flush_bytes=256ull<<20;
    GS_CHECK(cudaMalloc(&b.a,(n+128)*4)); GS_CHECK(cudaMalloc(&b.b,(n+128)*4));
    GS_CHECK(cudaMalloc(&b.p,(pn+128)*4)); GS_CHECK(cudaMalloc(&b.offsets,(n_active+1)*4));
    GS_CHECK(cudaMalloc(&b.flush,flush_bytes));
    GS_CHECK(cudaMalloc(&b.activation,6ull*in_f*sizeof(__half)));
    GS_CHECK(cudaMemcpyAsync(b.activation,act_f16,6ull*in_f*sizeof(__half),cudaMemcpyDeviceToDevice,st));
    GS_CHECK(cudaEventCreate(&b.begin)); GS_CHECK(cudaEventCreate(&b.end));
    std::vector<int> offsets(n_active+1), ids(n_active);
    GS_CHECK(cudaMemcpyAsync(offsets.data(),ex_off,(n_active+1)*4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaMemcpyAsync(ids.data(),ex_ids,n_active*4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaStreamSynchronize(st));
    if(offsets[0]!=0) return 40005;
    for(int g=0;g<n_active;++g){
        if(offsets[g+1]-offsets[g]<0 || offsets[g+1]-offsets[g]>1) return 40005;
        if(offsets[g+1]!=offsets[g]) printf("GRAPH_COMPONENT_OPERAND device=%d gu=%d row=%d expert=%d\n",device,gu,offsets[g],ids[g]);
    }
    auto launch = [&](int arm,int live){
        if(arm==2) return memra_moe_m1_graph_splitk(table,n_expert,ex_ids,b.activation,b.b+64,row_scale,
            macro_g,macro_u,route_w,b.offsets,n_active,in_f,out_f,limit,6,gu,b.p+64,stream);
        if(arm==1) return memra_moe_m1_splitk(table,n_expert,ex_ids,b.activation,b.a+64,row_scale,
            macro_g,macro_u,route_w,b.offsets,n_active,in_f,out_f,limit,std::max(live,1),gu,b.p+64,stream);
        if(gu) return memra_moe_kq_gemm_sk_gu_m1_half2(table,n_expert,ex_ids,b.activation,b.a+64,row_scale,
            macro_g,macro_u,route_w,b.offsets,n_active,in_f,out_f,limit,in_f/2,stream);
        return memra_moe_kq_gemm_sk_m1_half2(table,n_expert,ex_ids,b.activation,b.a+64,row_scale,
            b.offsets,n_active,in_f,out_f,in_f/2,stream);
    };
    GS_CHECK(cudaMemcpyAsync(b.offsets,offsets.data(),(n_active+1)*4,cudaMemcpyHostToDevice,st));
    GS_CHECK(cudaStreamSynchronize(st));
    GS_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    int capture_rc=launch(2,6);
    cudaError_t ended=cudaStreamEndCapture(st,&b.graph);
    if(capture_rc) return capture_rc;
    GS_CHECK(ended);
    GS_CHECK(cudaGraphInstantiate(&b.exec,b.graph,nullptr,nullptr,0));
    for(int live=0;live<=6;++live){
        std::vector<int> prefix=offsets;
        for(auto& v:prefix) v=std::min(v,live);
        GS_CHECK(cudaMemcpyAsync(b.offsets,prefix.data(),(n_active+1)*4,cudaMemcpyHostToDevice,st));
        // Match replay launch geometry in the timing table. The host-adaptive
        // control may capture a different grid per live count; the candidate
        // above retains ONE capture across the complete 0..6 sweep.
        struct Controls {
            cudaGraph_t graphs[2]={}; cudaGraphExec_t execs[2]={};
            ~Controls(){ for(int arm=0;arm<2;++arm){
                if(execs[arm]) cudaGraphExecDestroy(execs[arm]);
                if(graphs[arm]) cudaGraphDestroy(graphs[arm]); } }
        } controls;
        for(int arm=0;arm<2;++arm){
            // Initialize the control launcher's device attributes before capture.
            int rc=launch(arm,live); if(rc) return rc;
            GS_CHECK(cudaStreamSynchronize(st));
            GS_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
            rc=launch(arm,live);
            cudaError_t end=cudaStreamEndCapture(st,&controls.graphs[arm]);
            if(rc) return rc;
            GS_CHECK(end);
            GS_CHECK(cudaGraphInstantiate(&controls.execs[arm],controls.graphs[arm],nullptr,nullptr,0));
        }
        std::vector<float> reference(n), candidate(n), first(n);
        bool have_first=false;
        for(int cold=0;cold<2;++cold){
            double totals[3]={};
            for(int repeat=-2;repeat<10;++repeat){
                for(int arm : {2,0,1,1,0,2}){
                    GS_CHECK(cudaMemsetAsync(b.a,0xff,(n+128)*4,st));
                    GS_CHECK(cudaMemsetAsync(b.b,0xff,(n+128)*4,st));
                    GS_CHECK(cudaMemsetAsync(b.p,0xff,(pn+128)*4,st));
                    if(cold) GS_CHECK(cudaMemsetAsync(b.flush,repeat+3,flush_bytes,st));
                    GS_CHECK(cudaEventRecord(b.begin,st));
                    GS_CHECK(cudaGraphLaunch(arm==2?b.exec:controls.execs[arm],st));
                    GS_CHECK(cudaEventRecord(b.end,st)); GS_CHECK(cudaEventSynchronize(b.end));
                    float ms=0; GS_CHECK(cudaEventElapsedTime(&ms,b.begin,b.end));
                    if(repeat>=0) totals[arm]+=ms*1000;
                    auto& host=arm==2?candidate:reference;
                    GS_CHECK(cudaMemcpyAsync(host.data(),(arm==2?b.b:b.a)+64,n*4,cudaMemcpyDeviceToHost,st));
                    GS_CHECK(cudaStreamSynchronize(st));
                    if(arm==2){
                        for(float v:host) if(!std::isfinite(v)) return 40006;
                        for(size_t i=size_t(live)*out_f;i<n;++i) if(host[i]!=0.0f) return 40007;
                        if(have_first && std::memcmp(first.data(),host.data(),n*4)) return 40007;
                        first=host; have_first=true;
                    } else if(arm==1 && std::memcmp(first.data(),host.data(),size_t(live)*out_f*4)) return 40008;
                    for(int which=0;which<3;++which){
                        float* base=which==0?b.a:which==1?b.b:b.p;
                        size_t len=which==2?pn:n;
                        unsigned guard[128];
                        GS_CHECK(cudaMemcpyAsync(guard,base,64*4,cudaMemcpyDeviceToHost,st));
                        GS_CHECK(cudaMemcpyAsync(guard+64,base+64+len,64*4,cudaMemcpyDeviceToHost,st));
                        GS_CHECK(cudaStreamSynchronize(st));
                        for(unsigned v:guard) if(v!=0xffffffffu) return 40009;
                    }
                }
            }
            printf("GRAPH_COMPONENT device=%d gu=%d live=%d cold=%d flush_bytes=%zu control_us=%.6f adaptive_us=%.6f graph_us=%.6f measurements=20 bit_equal=1 deterministic=1 canaries=1 candidate_capture_count=1 all_timed_arms=replay\n",device,gu,live,cold,cold?flush_bytes:0,totals[0]/20,totals[1]/20,totals[2]/20);
        }
        // The down projection is the consumer of rowwise-packed GU output.
        // Poison only inactive activation rows in a private real-operand copy:
        // the actual captured projection must ignore them through the CSR.
        if(live < 6){
            GS_CHECK(cudaMemsetAsync(b.activation+size_t(live)*in_f,0x7b,size_t(6-live)*in_f*sizeof(__half),st));
            GS_CHECK(cudaGraphLaunch(b.exec,st));
            std::vector<float> poisoned(n);
            GS_CHECK(cudaMemcpyAsync(poisoned.data(),b.b+64,n*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaStreamSynchronize(st));
            if(std::memcmp(first.data(),poisoned.data(),n*4)) return 40010;
            GS_CHECK(cudaMemcpyAsync(b.activation,act_f16,6ull*in_f*sizeof(__half),cudaMemcpyDeviceToDevice,st));
        }
        printf("GRAPH_COMPONENT_TAIL device=%d gu=%d live=%d zero_output_rows=%d inactive_activation_poison_ignored=1\n",device,gu,live,6-live);
    }
    fflush(stdout); // Keep complete C++ receipt rows ahead of Rust coverage logs.
    g_graph_splitk_component_mask.fetch_or(bit);
    #undef GS_CHECK
    return 0;
}

// Instrument plus default-OFF door scaffold: one observed six-live CSR per
// rank/projection supplies derived prefixes 0..6. Also retain the first actual
// low-live (1..5) CSR per rank/projection, without modifying that route.
// Verbatim merged wrapper body from 182819614, renamed only for this harness.
static int splitk_fast_merged_wrapper_control(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream){
    if(!table || !ex_ids || !act_f16 || !out || !row_scale || !ex_off || !partial
       || n_expert < 1 || n_active < 1 || n_active > SK_MAX_G
       || slots != 6 || slots > n_active || (gu != 0 && gu != 1)
       || (gu && (!macro_g || !macro_u || !route_w))
       || (in_f != 4096 && in_f != 2048) || (out_f != 4096 && out_f != 2048))
        return 40004;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int grid = 6 * (out_f / SK_BN) * 16;
    if(gu) moe_m1_graph_splitk_partial_kernel<2><<<grid, dim3(32,4), 0, st>>>(
        table,0,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
    else moe_m1_graph_splitk_partial_kernel<1><<<grid, dim3(32,4), 0, st>>>(
        table,1,n_expert,ex_ids,in_f/2,(const __half*)act_f16,
        partial,ex_off,n_active,in_f,out_f);
    cudaError_t e = cudaGetLastError();
    if(e) return 1000 + int(e);
    if(gu) moe_m1_graph_splitk_reduce_kernel<2><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
    else moe_m1_graph_splitk_reduce_kernel<1><<<(6*out_f+255)/256,256,0,st>>>(
        partial,out,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
    e = cudaGetLastError();
    return e ? 1000 + int(e) : 0;
}

static std::atomic<int> g_splitk_fast_diagnostic_token{-1};
static std::atomic<bool> g_splitk_fast_diagnostic_done{false};
static std::string g_splitk_fast_diagnostic_dir;
void memra_moe_m1_splitk_fast_diagnose_for_gate(int token, const char* dir){
    g_splitk_fast_diagnostic_dir=dir;
    g_splitk_fast_diagnostic_token.store(token);
}
int memra_moe_m1_splitk_fast_diagnostic_done(){ return g_splitk_fast_diagnostic_done.load(); }
static std::atomic<unsigned> g_splitk_fast_component_mask{0};
unsigned memra_moe_m1_splitk_fast_component_mask(){ return g_splitk_fast_component_mask.load(); }
int memra_moe_m1_splitk_fast_component(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* out, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* partial, void* stream){
    int device=-1, observed=-1;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const bool diagnostic=g_splitk_fast_diagnostic_token.load()>=0;
    const char* phase="entry";
    #define GS_CHECK(call) do { cudaError_t e=(call); if(e) return splitk_fast_cuda_failure(e,#call,__FILE__,__LINE__,phase,device,gu,observed); } while(0)
    GS_CHECK(cudaPeekAtLastError());
    GS_CHECK(cudaStreamGetDevice(st,&device));
    int ambient=-1; GS_CHECK(cudaGetDevice(&ambient));
    if(ambient!=device) return 40012;
    for(const void* ptr : {(const void*)table,(const void*)ex_ids,act_f16,(const void*)ex_off}){
        cudaPointerAttributes attr{}; GS_CHECK(cudaPointerGetAttributes(&attr,ptr));
        if(attr.device!=device || attr.type!=cudaMemoryTypeDevice) return 40012;
    }
    if(diagnostic && (device!=0 || !gu || g_splitk_fast_diagnostic_done.load())) return 0;
    const unsigned key = 1u << (device*2+gu);
    const unsigned mask = g_splitk_fast_component_mask.load();
    if((mask & (key | (key<<4))) == (key | (key<<4))) return 0;
    GS_CHECK(cudaMemcpyAsync(&observed,ex_off+n_active,4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaStreamSynchronize(st));
    if(observed<0 || observed>6) return 40005;
    if(observed==0 && !diagnostic) return 0;
    const bool derived = observed==6;
    const unsigned bit = derived ? key : key<<4;
    if(mask & bit) return 0;
    const char* origin = derived ? "derived_prefix" : "observed_low";
    printf("SPLITK_FAST_BANDWIDTH_MODEL modeled effective weight GB/s (unique codes + scales / partial time); 1792 GB/s is a nominal peak, not achieved DRAM bandwidth\n");
    printf("SPLITK_FAST_COMPONENT_ROUTE device=%d gu=%d route_origin=%s source_live=%d independent_route=1\n",device,gu,origin,observed);
    phase="resource_query";
    for(int arm=0;arm<2;++arm){
        const void* kernel=gu
            ? (arm ? (const void*)moe_m1_splitk_fast_partial_kernel<2> : (const void*)moe_m1_graph_splitk_partial_kernel<2>)
            : (arm ? (const void*)moe_m1_splitk_fast_partial_kernel<1> : (const void*)moe_m1_graph_splitk_partial_kernel<1>);
        cudaFuncAttributes attr{}; int blocks=0;
        GS_CHECK(cudaFuncGetAttributes(&attr,kernel));
        GS_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0));
        printf("SPLITK_FAST_RESOURCE device=%d gu=%d route_origin=%s source_live=%d arm=%s threads=128 registers=%d shared_bytes=%zu local_bytes=%zu blocks_per_sm=%d\n",device,gu,origin,observed,arm?"on":"off",attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,blocks);
    }
    struct Buffers {
        float *a=nullptr,*b=nullptr,*p=nullptr;
        __half* activation=nullptr;
        int* offsets=nullptr;
        void* flush=nullptr;
        cudaEvent_t begin=nullptr,middle=nullptr,end=nullptr;
        cudaGraph_t graph=nullptr;
        cudaGraphExec_t exec=nullptr;
        ~Buffers(){ if(exec) cudaGraphExecDestroy(exec); if(graph) cudaGraphDestroy(graph);
            if(begin) cudaEventDestroy(begin); if(middle) cudaEventDestroy(middle); if(end) cudaEventDestroy(end);
            if(a) cudaFree(a); if(b) cudaFree(b); if(p) cudaFree(p);
            if(activation) cudaFree(activation);
            if(offsets) cudaFree(offsets); if(flush) cudaFree(flush); }
    } b;
    phase="allocation_and_operand_copy";
    const size_t n=size_t(6)*out_f, pn=n*16*(gu?2:1), flush_bytes=256ull<<20;
    GS_CHECK(cudaMalloc(&b.a,(n+128)*4)); GS_CHECK(cudaMalloc(&b.b,(n+128)*4));
    GS_CHECK(cudaMalloc(&b.p,(pn+128)*4)); GS_CHECK(cudaMalloc(&b.offsets,(n_active+1)*4));
    GS_CHECK(cudaMalloc(&b.flush,flush_bytes));
    GS_CHECK(cudaMalloc(&b.activation,6ull*in_f*sizeof(__half)));
    GS_CHECK(cudaMemcpyAsync(b.activation,act_f16,6ull*in_f*sizeof(__half),cudaMemcpyDeviceToDevice,st));
    GS_CHECK(cudaEventCreate(&b.begin)); GS_CHECK(cudaEventCreate(&b.middle)); GS_CHECK(cudaEventCreate(&b.end));
    std::vector<int> offsets(n_active+1), ids(n_active);
    GS_CHECK(cudaMemcpyAsync(offsets.data(),ex_off,(n_active+1)*4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaMemcpyAsync(ids.data(),ex_ids,n_active*4,cudaMemcpyDeviceToHost,st));
    GS_CHECK(cudaStreamSynchronize(st));
    if(offsets[0]!=0) return 40005;
    for(int g=0;g<n_active;++g){
        if(offsets[g+1]-offsets[g]<0 || offsets[g+1]-offsets[g]>1) return 40005;
        if(offsets[g+1]!=offsets[g]) printf("SPLITK_FAST_COMPONENT_OPERAND device=%d gu=%d route_origin=%s source_live=%d row=%d expert=%d\n",device,gu,origin,observed,offsets[g],ids[g]);
    }
    std::vector<float> diagnostic_reference(n);
    std::vector<unsigned> diagnostic_partial(pn);
    if(diagnostic){
        phase="route_dump";
        std::vector<int> live_ids;
        for(int g=0;g<n_active;++g) if(offsets[g+1]!=offsets[g]) live_ids.push_back(ids[g]);
        FILE* route=fopen((g_splitk_fast_diagnostic_dir+"/route.txt").c_str(),"wb");
        if(!route){ perror("splitk route dump"); return 40014; }
        fprintf(route,"device=%d gu=%d token=%d observed=%d caller_component_slots=%d launch_slots=6 n_expert=%d n_active=%d in_f=%d out_f=%d limit=%.9g partial_grid=%d partial_block=32x4 reduce_grid=%zu reduce_block=256 output_allocation_bytes=%zu scratch_allocation_bytes=%zu\n",
            device,gu,g_splitk_fast_diagnostic_token.load(),observed,slots,n_expert,n_active,in_f,out_f,limit,
            6*(out_f/SK_BN)*16,(n+255)/256,(n+128)*4,(pn+128)*4);
        fprintf(route,"offsets="); for(int v:offsets) fprintf(route,"%d,",v);
        fprintf(route,"\nids="); for(int v:ids) fprintf(route,"%d,",v);
        fprintf(route,"\nselected="); for(int v:live_ids) fprintf(route,"%d,",v);
        fprintf(route,"\n"); fclose(route);
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_ROUTE device=%d gu=%d token=%d observed=%d selected=",device,gu,g_splitk_fast_diagnostic_token.load(),observed);
        for(int v:live_ids) fprintf(stderr,"%d,",v);
        fprintf(stderr," dump=%s/route.txt\n",g_splitk_fast_diagnostic_dir.c_str());
        if(g_splitk_fast_diagnostic_token.load()!=0 || observed!=4 || live_ids!=std::vector<int>({90,93,95,104})
           || in_f!=4096 || out_f!=2048 || memra_moe_m1_splitk_fast_on()){
            fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_ROUTE_MISMATCH no_forced_route no_kernel_controls\n");
            return 40013;
        }
        auto dump = [&](const char* name,const void* ptr,size_t bytes){
            std::vector<unsigned char> data(bytes);
            GS_CHECK(cudaMemcpyAsync(data.data(),ptr,bytes,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaStreamSynchronize(st));
            FILE* f=fopen((g_splitk_fast_diagnostic_dir+"/"+name).c_str(),"wb");
            if(!f){ perror("splitk operand dump"); return 40014; }
            const bool ok=fwrite(data.data(),1,bytes,f)==bytes;
            const int closed=fclose(f);
            return ok && closed==0 ? 0 : 40014;
        };
        for(auto entry : {std::pair<const char*,const void*>{"activation.f16",act_f16},
                          {"row_scale.f32",row_scale},{"macro_g.f32",macro_g},
                          {"macro_u.f32",macro_u},{"route_weight.f32",route_w}}){
            int rc=dump(entry.first,entry.second,entry.second==act_f16 ? 6ull*in_f*2 : 6ull*4);
            if(rc) return rc;
        }
        auto reset = [&](){
            GS_CHECK(cudaMemsetAsync(b.a,0xff,(n+128)*4,st));
            GS_CHECK(cudaMemsetAsync(b.p,0xff,(pn+128)*4,st));
            return 0;
        };
        auto read_and_guard = [&](bool reference){
            std::vector<float> values(n); std::vector<unsigned> partials(pn);
            GS_CHECK(cudaMemcpyAsync(values.data(),b.a+64,n*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaMemcpyAsync(partials.data(),b.p+64,pn*4,cudaMemcpyDeviceToHost,st));
            unsigned guards[256];
            GS_CHECK(cudaMemcpyAsync(guards,b.a,64*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaMemcpyAsync(guards+64,b.a+64+n,64*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaMemcpyAsync(guards+128,b.p,64*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaMemcpyAsync(guards+192,b.p+64+pn,64*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaStreamSynchronize(st));
            for(unsigned v:guards) if(v!=0xffffffffu){ fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_CANARY_FAILURE phase=%s\n",phase); return 40009; }
            for(float v:values) if(!std::isfinite(v)) return 40006;
            for(size_t i=size_t(observed)*out_f;i<n;++i) if(values[i]!=0.0f) return 40007;
            if(reference){ diagnostic_reference=values; diagnostic_partial=partials; }
            else if(std::memcmp(values.data(),diagnostic_reference.data(),n*4) || partials!=diagnostic_partial){
                fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BIT_MISMATCH phase=%s\n",phase); return 40008;
            }
            fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_CONTROL_PASS phase=%s guarded=1 comparison=%s timing_events=0\n",phase,reference?"reference_saved":"bit_equal");
            return 0;
        };
        phase="merged_wrapper_original_operands_no_events";
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BEGIN phase=%s\n",phase);
        int rc=reset(); if(rc) return rc;
        rc=splitk_fast_merged_wrapper_control(table,n_expert,ex_ids,act_f16,b.a+64,row_scale,
            macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,6,gu,b.p+64,stream);
        if(rc){
            fprintf(stderr,"SPLITK_FAST_MERGED_DEFAULT_FAILURE_P1 rc=%d call=splitk_fast_merged_wrapper_control file=%s line=%d\n",rc,__FILE__,__LINE__);
            if(rc>=1000 && rc<2000) splitk_fast_cuda_failure(cudaError_t(rc-1000),"splitk_fast_merged_wrapper_control",__FILE__,__LINE__,phase,device,gu,observed);
            return rc;
        }
        GS_CHECK(cudaPeekAtLastError()); GS_CHECK(cudaStreamSynchronize(st));
        rc=read_and_guard(true); if(rc) return rc;
        rc=reset(); if(rc) return rc;
        phase="scaffold_wrapper_original_operands_no_events";
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BEGIN phase=%s\n",phase);
        rc=reset(); if(rc) return rc;
        rc=memra_moe_m1_graph_splitk(table,n_expert,ex_ids,act_f16,b.a+64,row_scale,
            macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,6,gu,b.p+64,stream);
        if(rc){
            fprintf(stderr,"SPLITK_FAST_SCAFFOLD_WRAPPER_FAILURE rc=%d call=memra_moe_m1_graph_splitk file=%s line=%d\n",rc,__FILE__,__LINE__);
            if(rc>=1000 && rc<2000) splitk_fast_cuda_failure(cudaError_t(rc-1000),"memra_moe_m1_graph_splitk",__FILE__,__LINE__,phase,device,gu,observed);
            return rc;
        }
        GS_CHECK(cudaPeekAtLastError()); GS_CHECK(cudaStreamSynchronize(st));
        rc=read_and_guard(false); if(rc) return rc;
        rc=reset(); if(rc) return rc;
        phase="production_partial_original_operands_no_events";
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BEGIN phase=%s\n",phase);
        moe_m1_graph_splitk_partial_kernel<2><<<6*(out_f/SK_BN)*16,dim3(32,4),0,st>>>(
            table,0,n_expert,ex_ids,in_f/2,(const __half*)act_f16,b.p+64,ex_off,n_active,in_f,out_f);
        GS_CHECK(cudaPeekAtLastError()); GS_CHECK(cudaStreamSynchronize(st));
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_KERNEL_PASS phase=%s\n",phase);
        phase="production_reducer_original_operands_no_events";
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BEGIN phase=%s\n",phase);
        moe_m1_graph_splitk_reduce_kernel<2><<<(6*out_f+255)/256,256,0,st>>>(
            b.p+64,b.a+64,row_scale,macro_g,macro_u,route_w,6,out_f,limit,ex_off,n_active);
        GS_CHECK(cudaPeekAtLastError()); GS_CHECK(cudaStreamSynchronize(st));
        rc=read_and_guard(false); if(rc) return rc;
        phase="production_wrapper_copied_operands_no_events";
        fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BEGIN phase=%s\n",phase);
        GS_CHECK(cudaMemcpyAsync(b.offsets,offsets.data(),(n_active+1)*4,cudaMemcpyHostToDevice,st));
        rc=reset(); if(rc) return rc;
        rc=memra_moe_m1_graph_splitk(table,n_expert,ex_ids,b.activation,b.a+64,row_scale,
            macro_g,macro_u,route_w,b.offsets,n_active,in_f,out_f,limit,6,gu,b.p+64,stream);
        if(rc){
            if(rc>=1000 && rc<2000) splitk_fast_cuda_failure(cudaError_t(rc-1000),"memra_moe_m1_graph_splitk(copied operands)",__FILE__,__LINE__,phase,device,gu,observed);
            return rc;
        }
        GS_CHECK(cudaPeekAtLastError()); GS_CHECK(cudaStreamSynchronize(st));
        rc=read_and_guard(false); if(rc) return rc;
        fflush(stdout); fflush(stderr);
    }
    phase="instrument_capture_on";
    auto launch = [&](int arm,int live){
        GS_CHECK(cudaEventRecord(b.begin,st));
        int rc=moe_m1_splitk_fast_launch(table,n_expert,ex_ids,b.activation,
            (arm==2?b.b:b.a)+64,row_scale,macro_g,macro_u,route_w,b.offsets,
            n_active,in_f,out_f,limit,6,gu,b.p+64,stream,arm==2,b.middle);
        if(rc){
            if(rc>=1000 && rc<2000) splitk_fast_cuda_failure(cudaError_t(rc-1000),"moe_m1_splitk_fast_launch with middle event",__FILE__,__LINE__,phase,device,gu,observed);
            return rc;
        }
        GS_CHECK(cudaEventRecord(b.end,st));
        return 0;
    };
    GS_CHECK(cudaMemcpyAsync(b.offsets,offsets.data(),(n_active+1)*4,cudaMemcpyHostToDevice,st));
    GS_CHECK(cudaStreamSynchronize(st));
    GS_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    int capture_rc=launch(2,6);
    cudaError_t ended=cudaStreamEndCapture(st,&b.graph);
    if(capture_rc) return capture_rc;
    GS_CHECK(ended);
    GS_CHECK(cudaGraphInstantiate(&b.exec,b.graph,nullptr,nullptr,0));
        // Both fixed-grid arms retain one capture across the entire CSR sweep.
        struct Controls {
            cudaGraph_t graphs[2]={}; cudaGraphExec_t execs[2]={};
            ~Controls(){ for(int arm=0;arm<2;++arm){
                if(execs[arm]) cudaGraphExecDestroy(execs[arm]);
                if(graphs[arm]) cudaGraphDestroy(graphs[arm]); } }
        } controls;
        for(int arm=1;arm<2;++arm){
            // Reproduce r1's uncaptured warmup and subsequent capture unchanged.
            phase="instrument_uncaptured_control_with_events";
            int rc=launch(arm,6); if(rc) return rc;
            GS_CHECK(cudaStreamSynchronize(st));
            phase="instrument_capture_off";
            GS_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
            rc=launch(arm,6);
            cudaError_t end=cudaStreamEndCapture(st,&controls.graphs[arm]);
            if(rc) return rc;
            GS_CHECK(end);
            GS_CHECK(cudaGraphInstantiate(&controls.execs[arm],controls.graphs[arm],nullptr,nullptr,0));
        }
    for(int live=derived?0:observed;live<=observed;++live){
        std::vector<int> prefix=offsets;
        if(derived) for(auto& v:prefix) v=std::min(v,live);
        GS_CHECK(cudaMemcpyAsync(b.offsets,prefix.data(),(n_active+1)*4,cudaMemcpyHostToDevice,st));
        std::vector<float> reference(n), candidate(n), first(n);
        std::vector<unsigned> first_partial(pn), current_partial(pn);
        bool have_first=false;
        for(int cold=0;cold<2;++cold){
            double totals[3]={}, partial_us[3]={}, reduce_us[3]={};
            for(int repeat=-2;repeat<10;++repeat){
                for(int arm : {2,1,1,2}){
                    GS_CHECK(cudaMemsetAsync(b.a,0xff,(n+128)*4,st));
                    GS_CHECK(cudaMemsetAsync(b.b,0xff,(n+128)*4,st));
                    GS_CHECK(cudaMemsetAsync(b.p,0xff,(pn+128)*4,st));
                    if(cold) GS_CHECK(cudaMemsetAsync(b.flush,repeat+3,flush_bytes,st));
                    phase=arm==2?"instrument_replay_on":"instrument_replay_off";
                    GS_CHECK(cudaGraphLaunch(arm==2?b.exec:controls.execs[arm],st));
                    GS_CHECK(cudaEventSynchronize(b.end));
                    phase="instrument_elapsed_total";
                    float ms=0; GS_CHECK(cudaEventElapsedTime(&ms,b.begin,b.end));
                    if(diagnostic || repeat>=0){
                        totals[arm]+=ms*1000;
                        phase="instrument_elapsed_partial";
                        GS_CHECK(cudaEventElapsedTime(&ms,b.begin,b.middle)); partial_us[arm]+=ms*1000;
                        phase="instrument_elapsed_reduce";
                        GS_CHECK(cudaEventElapsedTime(&ms,b.middle,b.end)); reduce_us[arm]+=ms*1000;
                    }
                    auto& host=arm==2?candidate:reference;
                    GS_CHECK(cudaMemcpyAsync(host.data(),(arm==2?b.b:b.a)+64,n*4,cudaMemcpyDeviceToHost,st));
                    GS_CHECK(cudaStreamSynchronize(st));
                    GS_CHECK(cudaMemcpyAsync(current_partial.data(),b.p+64,pn*4,cudaMemcpyDeviceToHost,st));
                    GS_CHECK(cudaStreamSynchronize(st));
                    if(arm==2) first_partial=current_partial;
                    else if(current_partial!=first_partial) return 40011;
                    if(arm==2){
                        for(float v:host) if(!std::isfinite(v)) return 40006;
                        for(size_t i=size_t(live)*out_f;i<n;++i) if(host[i]!=0.0f) return 40007;
                        if(have_first && std::memcmp(first.data(),host.data(),n*4)) return 40007;
                        first=host; have_first=true;
                    } else if(arm==1 && std::memcmp(first.data(),host.data(),n*4)) return 40008;
                    for(int which=0;which<3;++which){
                        float* base=which==0?b.a:which==1?b.b:b.p;
                        size_t len=which==2?pn:n;
                        unsigned guard[128];
                        GS_CHECK(cudaMemcpyAsync(guard,base,64*4,cudaMemcpyDeviceToHost,st));
                        GS_CHECK(cudaMemcpyAsync(guard+64,base+64+len,64*4,cudaMemcpyDeviceToHost,st));
                        GS_CHECK(cudaStreamSynchronize(st));
                        for(unsigned v:guard) if(v!=0xffffffffu) return 40009;
                    }
                    if(diagnostic){
                        if(std::memcmp(host.data(),diagnostic_reference.data(),n*4) || current_partial!=diagnostic_partial){
                            fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_BIT_MISMATCH phase=instrument_replay\n"); return 40008;
                        }
                        if(arm==1){
                            g_splitk_fast_diagnostic_done.store(true);
                            fprintf(stderr,"SPLITK_FAST_DIAGNOSTIC_COMPLETE merged_wrapper=pass scaffold_wrapper=pass separate_kernels=pass copied_operands=pass event_path=pass timing_rows=0\n");
                            return 0;
                        }
                    }
                }
            }
            const int slices=moe_m1_slices(live,out_f,gu);
            const size_t codes=size_t(live)*out_f*in_f*(gu?2:1)/2;
            const size_t scales=codes/8;
            const size_t scratch=size_t(live)*out_f*slices*(gu?2:1)*4;
            // Requested weight bandwidth uses unique code+scale bytes. The
            // activation count is issued cp.async bytes including M1 row replication.
            const size_t activation=size_t(live)*(out_f/SK_BN)*in_f*16*2*(gu?2:1);
            for(int arm : {1,2}){
                const double pus=partial_us[arm]/20, rus=reduce_us[arm]/20;
                printf("SPLITK_FAST_COMPONENT device=%d gu=%d route_origin=%s source_live=%d live=%d slices=%d cold=%d arm=%s flush_bytes=%zu codes_bytes=%zu scales_bytes=%zu weight_bytes=%zu activation_issued_bytes=%zu scratch_write_bytes=%zu partial_us=%.6f reduce_us=%.6f total_us=%.6f modeled_effective_weight_GBs=%.6f pct_nominal_peak_1792=%.6f measurements=20 bit_equal=1 deterministic=1 canaries=1 candidate_capture_count=1 all_timed_arms=replay\n",device,gu,origin,observed,live,slices,cold,arm==2?"on":"off",cold?flush_bytes:0,codes,scales,codes+scales,activation,scratch,pus,rus,totals[arm]/20,(codes+scales)/pus/1000,(codes+scales)/pus/1000/1792*100);
            }
        }
        // The down projection is the consumer of rowwise-packed GU output.
        // Poison only inactive activation rows in a private real-operand copy:
        // the actual captured projection must ignore them through the CSR.
        if(live < 6){
            GS_CHECK(cudaMemsetAsync(b.activation+size_t(live)*in_f,0x7b,size_t(6-live)*in_f*sizeof(__half),st));
            GS_CHECK(cudaGraphLaunch(b.exec,st));
            std::vector<float> poisoned(n);
            GS_CHECK(cudaMemcpyAsync(poisoned.data(),b.b+64,n*4,cudaMemcpyDeviceToHost,st));
            GS_CHECK(cudaStreamSynchronize(st));
            if(std::memcmp(first.data(),poisoned.data(),n*4)) return 40010;
            GS_CHECK(cudaMemcpyAsync(b.activation,act_f16,6ull*in_f*sizeof(__half),cudaMemcpyDeviceToDevice,st));
        }
        printf("SPLITK_FAST_COMPONENT_TAIL device=%d gu=%d route_origin=%s source_live=%d live=%d zero_output_rows=%d inactive_activation_poison_ignored=1\n",device,gu,origin,observed,live,6-live);
    }
    fflush(stdout); // Keep complete C++ receipt rows ahead of Rust coverage logs.
    g_splitk_fast_component_mask.fetch_or(bit);
    #undef GS_CHECK
    return 0;
}

static std::atomic<int> g_splitk_component_token{0};
void memra_moe_m1_splitk_component_token(int token){ g_splitk_component_token.store(token); }

// Diagnostic only: called once per projection and rank on actual routed inputs.
// Output and scratch allocations have 64-word canaries at both ends. Event
// intervals bracket the whole candidate (partial + reduction), one launch per
// arm per ABBA position, 10 cycles, 20 measurements per arm.
int memra_moe_m1_splitk_component(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        const void* act_f16, float* ignored, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        float limit, int slots, int gu, float* ignored_partial, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    const int token = g_splitk_component_token.load();
    int observed=0;
    cudaError_t observed_error = cudaMemcpyAsync(&observed,ex_off+n_active,sizeof(int),cudaMemcpyDeviceToHost,st);
    if(observed_error) return 1000 + int(observed_error);
    observed_error = cudaStreamSynchronize(st);
    if(observed_error) return 1000 + int(observed_error);
    if(observed < 0 || observed > slots) return 40005;
    if(observed == 0){
        int device=0; cudaGetDevice(&device);
        printf("SPLITK_COMPONENT token=%d device=%d gu=%d slots=0 skipped=1\n",token,device,gu);
        return 0;
    }
    slots = observed;
    const size_t n = size_t(slots) * out_f, scratch_n = n * 16 * (gu ? 2 : 1);
    struct Buffers {
        float *a=nullptr, *b=nullptr, *p=nullptr;
        cudaEvent_t begin=nullptr, end=nullptr, middle=nullptr;
        ~Buffers(){ if(middle) cudaEventDestroy(middle); if(begin) cudaEventDestroy(begin); if(end) cudaEventDestroy(end);
            if(a) cudaFree(a); if(b) cudaFree(b); if(p) cudaFree(p); }
    } b;
    #define SK_CHECK(call) do { cudaError_t e = (call); if(e) return 1000 + int(e); } while(0)
    SK_CHECK(cudaMalloc(&b.a,(n+128)*4));
    SK_CHECK(cudaMalloc(&b.b,(n+128)*4));
    SK_CHECK(cudaMalloc(&b.p,(scratch_n+128)*4));
    SK_CHECK(cudaMemsetAsync(b.a,0xff,(n+128)*4,st));
    SK_CHECK(cudaMemsetAsync(b.b,0xff,(n+128)*4,st));
    SK_CHECK(cudaMemsetAsync(b.p,0xff,(scratch_n+128)*4,st));
    SK_CHECK(cudaEventCreate(&b.begin)); SK_CHECK(cudaEventCreate(&b.end)); SK_CHECK(cudaEventCreate(&b.middle));
    std::vector<int> offsets(n_active+1), ids(n_active);
    SK_CHECK(cudaMemcpyAsync(offsets.data(),ex_off,offsets.size()*4,cudaMemcpyDeviceToHost,st));
    SK_CHECK(cudaMemcpyAsync(ids.data(),ex_ids,ids.size()*4,cudaMemcpyDeviceToHost,st));
    SK_CHECK(cudaStreamSynchronize(st));
    int device=0; SK_CHECK(cudaGetDevice(&device));
    cudaFuncAttributes attr; int occ=0, sms=0, l2=0;
    const void* kernel = gu ? (const void*)moe_m1_splitk_partial_kernel<2> : (const void*)moe_m1_splitk_partial_kernel<1>;
    SK_CHECK(cudaFuncGetAttributes(&attr,kernel));
    SK_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kernel,128,0));
    SK_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
    SK_CHECK(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,device));
    printf("SPLITK_RESOURCES device=%d gu=%d sms=%d blocks_per_sm=%d registers=%d shared_bytes=%zu local_bytes=%zu l2_bytes=%d regime=warm_matched_planes\n",device,gu,sms,occ,attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,l2);
    int live=0;
    for(int g=0;g<n_active;++g){
        if(offsets[g+1]-offsets[g] < 0 || offsets[g+1]-offsets[g] > 1) return 40005;
        if(offsets[g+1]!=offsets[g]){ ++live;
            printf("SPLITK_EXPERT token=%d device=%d gu=%d slot=%d expert=%d\n",token,device,gu,offsets[g],ids[g]); }
    }
    if(live != slots || offsets[0]!=0 || offsets[n_active]!=slots) return 40005;
    auto launch = [&](int arm){
        if(arm) return memra_moe_m1_splitk(table,n_expert,ex_ids,act_f16,b.b+64,row_scale,
            macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,slots,gu,b.p+64,stream);
        if(gu) return memra_moe_kq_gemm_sk_gu_m1_half2(table,n_expert,ex_ids,act_f16,
            b.a+64,row_scale,macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,in_f/2,stream);
        return memra_moe_kq_gemm_sk_m1_half2(table,n_expert,ex_ids,act_f16,b.a+64,row_scale,
            ex_off,n_active,in_f,out_f,in_f/2,stream);
    };
    std::vector<float> reference(n), candidate(n), first(n);
    double totals[2]={}; float max_abs=0, max_rel=0, peak=0; bool have_first=false;
    for(int cycle=-2;cycle<10;++cycle){
        const int order[4]={0,1,1,0};
        for(int arm:order){
            SK_CHECK(cudaEventRecord(b.begin,st));
            int rc=launch(arm); if(rc) return rc;
            SK_CHECK(cudaEventRecord(b.end,st)); SK_CHECK(cudaEventSynchronize(b.end));
            float ms=0; SK_CHECK(cudaEventElapsedTime(&ms,b.begin,b.end));
            if(cycle>=0){ totals[arm]+=ms;
                printf("SPLITK_TIMING token=%d device=%d gu=%d cycle=%d arm=%d us=%.6f\n",token,device,gu,cycle,arm,ms*1000); }
            auto& host = arm ? candidate : reference;
            SK_CHECK(cudaMemcpyAsync(host.data(),(arm?b.b:b.a)+64,n*4,cudaMemcpyDeviceToHost,st));
            SK_CHECK(cudaStreamSynchronize(st));
            for(float v:host) if(!std::isfinite(v)) return 40006;
            if(arm){
                if(!have_first){ first=host; have_first=true; }
                else if(std::memcmp(first.data(),host.data(),n*4)) return 40007;
            }
        }
    }
    for(float v:reference) peak=std::max(peak,std::abs(v));
    // K<=4096 has gamma_K ~= 2.45e-4 in f32. The mixed threshold allows
    // cancellation near zero while requiring 0.02% relative agreement for
    // large outputs. The 0.002% plane-scale floor is stricter than gamma_K;
    // this is an empirical component admission bound, not a bit-identity claim.
    bool numeric_ok=true;
    for(size_t i=0;i<n;++i){
        float d=std::abs(candidate[i]-reference[i]);
        float r=d/std::max(std::abs(reference[i]),1e-30f);
        max_abs=std::max(max_abs,d); max_rel=std::max(max_rel,r);
        if(d > 2e-5f*peak + 2e-4f*std::abs(reference[i])) numeric_ok=false;
    }
    for(int which=0;which<3;++which){
        float* base=which==0?b.a:which==1?b.b:b.p;
        size_t len=which==2?scratch_n:n;
        uint32_t guard[128];
        SK_CHECK(cudaMemcpyAsync(guard,base,256,cudaMemcpyDeviceToHost,st));
        SK_CHECK(cudaMemcpyAsync(guard+64,base+64+len,256,cudaMemcpyDeviceToHost,st));
        SK_CHECK(cudaStreamSynchronize(st));
        for(uint32_t v:guard) if(v!=0xffffffff) return 40008;
    }
    SK_CHECK(cudaEventRecord(b.begin,st));
    int phase_rc=moe_m1_splitk_launch(table,n_expert,ex_ids,act_f16,b.b+64,row_scale,
        macro_g,macro_u,route_w,ex_off,n_active,in_f,out_f,limit,slots,gu,b.p+64,stream,b.middle);
    if(phase_rc) return phase_rc;
    SK_CHECK(cudaEventRecord(b.end,st)); SK_CHECK(cudaEventSynchronize(b.end));
    float partial_ms=0, reduce_ms=0;
    SK_CHECK(cudaEventElapsedTime(&partial_ms,b.begin,b.middle));
    SK_CHECK(cudaEventElapsedTime(&reduce_ms,b.middle,b.end));
    printf("SPLITK_PHASE token=%d device=%d gu=%d slots=%d partial_us=%.6f reduce_us=%.6f diagnostic_only=1\n",token,device,gu,slots,partial_ms*1000,reduce_ms*1000);
    printf("SPLITK_COMPONENT token=%d device=%d gu=%d slots=%d slices=%d blocks=%d max_abs=%.9g max_rel=%.9g peak=%.9g oracle_us=%.6f candidate_us=%.6f speedup=%.6f numeric_ok=%d finite=1 deterministic=1 canaries=1 launches_per_arm=20 numeric_class=moe_m1_adaptive_splitk_f32_fixed_order\n",
        token,device,gu,slots,moe_m1_slices(slots,out_f,gu),slots*(out_f/64)*moe_m1_slices(slots,out_f,gu),max_abs,max_rel,peak,totals[0]*50,totals[1]*50,totals[0]/totals[1],numeric_ok);
    fflush(stdout);
    #undef SK_CHECK
    return numeric_ok ? 0 : 40009;
}

unsigned long long memra_moe_kq_gemm_sk_gu_half2_dispatches(){
    return g_moe_kq_gu_h2_dispatches.load(std::memory_order_relaxed);
}

unsigned long long memra_moe_kq_gemm_sk_m1_half2_dispatches(){
    return g_moe_kq_m1_h2_dispatches.load(std::memory_order_relaxed);
}

int memra_moe_f16g_gather_act(const float* x, const int* pair_tok_or_null, void* act_f16,
        float* row_scale, int in_f, int n_pairs, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_pairs,1,1), blk(256,1,1);
    gather_act_f16_kernel<<<grid,blk,0,st>>>(x, pair_tok_or_null, (__half*)act_f16, row_scale,
                                             in_f, n_pairs);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f_scaled(const void* src_f16, float* dst, const float* row_scale,
        int ncols, int nrows, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    h2f_rows_scale_kernel<<<(unsigned)nrows,256,0,st>>>((const __half*)src_f16, dst, row_scale,
                                                        ncols, nrows);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

int memra_moe_f16g_h2f(const void* src_f16, float* dst, size_t n, void* stream){
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    size_t blocks = (n + 255)/256;
    h2f_kernel<<<(unsigned)blocks,256,0,st>>>((const __half*)src_f16, dst, n);
    cudaError_t e=cudaGetLastError(); return e?1000+(int)e:0;
}

// One grouped GEMM over the CSR groups. ex_off_host = HOST copy of the CSR offsets
// (n_active+1 ints). y = f16 [n_pairs, out_f] (pair-major; caller converts via h2f —
// the grouped API's type matrix has no 16F-in/32F-out combo).
int memra_moe_f16g_gemm(const void* w_f16, const void* act_f16, void* y,
        const int* ex_off_host, int n_active, int in_f, int out_f, void* stream){
    // per-device handles: a cuBLAS handle is bound to the device current at cublasCreate, and
    // a shared one fails from the other PP stage on a 2x B200 pair (2026-09-02, f16_prefill.cu).
    static cublasHandle_t handles[64] = {};
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = 0;
    if(!handles[dev]){
        if(cublasCreate(&handles[dev]) != CUBLAS_STATUS_SUCCESS) return 3;
    }
    cublasHandle_t handle = handles[dev];
    cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream));

    // per-group host arrays (n_active <= a few hundred; stack-scale, heap for safety)
    const int G = n_active;
    static thread_local int cap = 0;
    static thread_local cublasOperation_t *ta=nullptr,*tb=nullptr;
    static thread_local int *ma=nullptr,*na=nullptr,*ka=nullptr,*lda=nullptr,*ldb=nullptr,*ldc=nullptr,*gsz=nullptr;
    static thread_local float *al=nullptr,*be=nullptr;
    static thread_local const void **Aa=nullptr,**Ba=nullptr; static thread_local void **Ca=nullptr;
    if(G>cap){
        delete[] ta; delete[] tb; delete[] ma; delete[] na; delete[] ka; delete[] lda; delete[] ldb;
        delete[] ldc; delete[] gsz; delete[] al; delete[] be; delete[] Aa; delete[] Ba; delete[] Ca;
        cap=G;
        ta=new cublasOperation_t[G]; tb=new cublasOperation_t[G];
        ma=new int[G]; na=new int[G]; ka=new int[G]; lda=new int[G]; ldb=new int[G]; ldc=new int[G];
        gsz=new int[G]; al=new float[G]; be=new float[G];
        Aa=new const void*[G]; Ba=new const void*[G]; Ca=new void*[G];
    }
    int g_used = 0;
    for(int g=0; g<G; g++){
        int lo=ex_off_host[g], hi=ex_off_host[g+1], m_e=hi-lo;
        if(m_e<=0) continue;
        ta[g_used]=CUBLAS_OP_T; tb[g_used]=CUBLAS_OP_N;
        ma[g_used]=out_f; na[g_used]=m_e; ka[g_used]=in_f;
        lda[g_used]=in_f; ldb[g_used]=in_f; ldc[g_used]=out_f;
        al[g_used]=1.0f; be[g_used]=0.0f; gsz[g_used]=1;
        Aa[g_used]=(const uint8_t*)w_f16 + (size_t)g*out_f*in_f*sizeof(__half);
        Ba[g_used]=(const uint8_t*)act_f16 + (size_t)lo*in_f*sizeof(__half);
        Ca[g_used]=(uint8_t*)y + (size_t)lo*out_f*sizeof(__half);
        g_used++;
    }
    if(g_used==0) return 0;
    cublasStatus_t s = cublasGemmGroupedBatchedEx(handle, ta, tb, ma, na, ka,
        al, Aa, CUDA_R_16F, lda, Ba, CUDA_R_16F, ldb,
        be, Ca, CUDA_R_16F, ldc, g_used, gsz, CUBLAS_COMPUTE_32F);
    if(s != CUBLAS_STATUS_SUCCESS) return 20000 + (int)s;
    return 0;
}

} // extern "C"

// Runtime-API device bind (2026-08-27, gemm-prime bring-up). Every raw <<<>>> launch in this
// file follows the RUNTIME API's current device, which nothing in this repo had ever moved off
// 0 — the whole FFI surface was root-only until the TP2 grouped prime called it on rank
// engines, where a rank-1 stream plus a device-0 runtime context is cudaErrorInvalidValue
// (the rc=1001 the [moe-sk-err] receipt finally named; geometry exonerated by moe-sk-repro,
// which ran the same shapes OK single-device up to max_m=6639).
extern "C" int memra_bind_device(int dev){
    return (int)cudaSetDevice(dev);
}
