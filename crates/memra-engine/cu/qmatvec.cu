// Resident-quantized matmul: weights stay in GGUF block format in VRAM, dequantized in-register
// inside the kernel (never materialized as f32/f16). Fixes the OOM. Activations are f32 (Stage A:
// correctness-first; Stage B will quantize activations to q8_1 + int8 dp4a like llama.cpp MMVQ/MMQ).
//
// y[m, out] = x[m, in] @ W[out, in]^T,  W is quantized (ggml block layout), x/y are f32.
// Layout: x token-major [m, in] (x[t*in + i]); W row o = out-feature o, `in` elements quantized;
//         y token-major [m, out] (y[t*out + o]). One block per (token, out-row); threads reduce over `in`.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>   // cp.async staging for the v3 GEMV family (sm_80+)
#include <cstdint>

// PDL entry (same contract as kernels.cu): cudaGridDependencySynchronize() orders this
// kernel's loads after the producer's writes while the grid launch overlaps the producer's
// drain. ONLY kernels carrying this macro may be launched with
// CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION. sm_90+ only.
#if !defined(MEMRA_PORTABLE_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MEMRA_PDL_ENTRY() cudaGridDependencySynchronize()
#else
#define MEMRA_PDL_ENTRY()
#endif

__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// IQ3_S grid: 512 u32 entries, each packs 4 unsigned bytes. Verbatim from ggml-common.h:1042.
// STORAGE CLASS (2026-07-06): plain __device__ (global mem, L1-cached), NOT __constant__ —
// the constant cache broadcasts only on uniform addresses and SERIALIZES divergent reads, and
// these grid lookups are divergent by construction (every lane decodes different codes).
// llama's GGML_TABLE_BEGIN is `static const __device__` for exactly this reason.
__device__ unsigned int iq3s_grid_const[512] = {
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
__device__ __forceinline__ unsigned int iq3s_grid_d(int idx) { return iq3s_grid_const[idx]; }

// ---- per-dtype: dequantize element j of weight-row `wrow` (raw bytes) and return its f32 value ----
// Q8_0: block=32, bytes=34 (fp16 d + int8[32]).
__device__ __forceinline__ float deq_q8_0(const uint8_t* row, int j) {
    int blk = j >> 5, off = j & 31;
    const uint8_t* b = row + blk * 34;
    float d = half_to_float(*(const uint16_t*)b);
    int8_t q = (int8_t)b[2 + off];
    return d * (float)q;
}
// Q4_K: superblock=256, bytes=144. {fp16 d, fp16 dmin, u8 scales[12], u8 qs[128]}.
// 8 sub-blocks of 32; 6-bit scale/min via get_scale_min_k4.
__device__ __forceinline__ void q4k_scale_min(const uint8_t* sc, int j, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *d = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); *m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}
__device__ __forceinline__ float deq_q4_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;          // which superblock, idx within
    const uint8_t* b = row + blk * 144;
    float d = half_to_float(*(const uint16_t*)b);
    float dmin = half_to_float(*(const uint16_t*)(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qs = b + 16;
    // ggml q4_K layout: for is in 0..7, group of 32. j = group*32 + l (l 0..31).
    // qs are nibble-packed: 64-elem chunk uses 32 bytes; low nibble first 32, high nibble next 32.
    int group = jj >> 5;       // 0..7
    int l = jj & 31;
    // each 64-block (2 groups) shares 32 qs bytes: group even -> low nibble, odd -> high nibble
    int chunk = group >> 1;    // 0..3  (which 32-byte qs run)
    const uint8_t* q = qs + chunk * 32;
    uint8_t sc, mn;
    q4k_scale_min(scales, group, &sc, &mn);
    float val;
    if ((group & 1) == 0) val = d * (float)sc * (float)(q[l] & 0xF) - dmin * (float)mn;
    else                  val = d * (float)sc * (float)(q[l] >> 4)  - dmin * (float)mn;
    return val;
}
// Q6_K: superblock=256, bytes=210. {u8 ql[128], u8 qh[64], i8 scales[16], fp16 d}.
__device__ __forceinline__ float deq_q6_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 210;
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales = (const int8_t*)(b + 192);
    float d = half_to_float(*(const uint16_t*)(b + 208));
    // ggml q6_K: two halves of 128. n = jj/128 (0/1); within half l=jj%128 ; sub group of 16 -> scale.
    int n = jj >> 7;           // 0 or 1
    int l = jj & 127;          // 0..127
    int il = l & 31;           // position within 32-run
    int run = l >> 5;          // 0..3 within half
    const uint8_t* qlh = ql + n * 64;
    const uint8_t* qhh = qh + n * 32;
    // reconstruct q like ggml dequantize_row_q6_K
    int ql_bits, qh_bits;
    if (run == 0)      { ql_bits = qlh[il] & 0xF;        qh_bits = (qhh[il] >> 0) & 3; }
    else if (run == 1) { ql_bits = qlh[il + 32] & 0xF;   qh_bits = (qhh[il] >> 2) & 3; }
    else if (run == 2) { ql_bits = qlh[il] >> 4;         qh_bits = (qhh[il] >> 4) & 3; }
    else               { ql_bits = qlh[il + 32] >> 4;    qh_bits = (qhh[il] >> 6) & 3; }
    int q = (ql_bits | (qh_bits << 4)) - 32;
    int is = n * 8 + run * 2 + (il >> 4);   // scale index 0..15
    return d * (float)scales[is] * (float)q;
}

// device codebook tables — plain __device__ (L1), NOT __constant__: per-lane indices diverge
// (expert_dot_iq4xs_g does 8 byte-lookups per group per lane), and the constant cache serializes
// divergent reads. Same fix class as iq3s_grid_const (+11.8% 35B decode, 2026-07-06).
// mxfp4 stays __constant__: its consumers go through get_int_from_table_16_d (byte_perm on two
// uniform 8B halves — broadcast-friendly, the constant cache's good case).
// __align__(16): expert_dot_iq4xs_g_v reads this table as four u32 words for the byte_perm
// lookup (get_int_from_table_16_d) — same 16 byte VALUES, alignment attribute only.
__device__ __align__(16) signed char kvalues_iq4nl_d[16] =
    {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
__device__ __constant__ signed char kvalues_mxfp4_d[16] =
    {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

// Fast 4-bit codebook lookup (llama.cpp vecdotq.cuh get_int_from_table_16). Takes 4 packed
// bytes (8 nibbles) in q4; returns int2 where .x = the 4 codebook int8s of the LOW nibbles
// (one per byte, packed) and .y = the 4 codebook int8s of the HIGH nibbles. ~5 __byte_perm
// vs 8 scalar table[] loads — the NVFP4/MXFP4/IQ4 decode hot loop is ALU-bound otherwise.
// CUDA __byte_perm selects bytes by 3-bit indices; the 4th index bit is handled by a 2nd perm.
__device__ __forceinline__ int2 get_int_from_table_16_d(int q4, const signed char* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210u | ((q4 & 0x88888888u) >> 1));
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16u * i;
        const uint32_t low  = __byte_perm(table32[0], table32[1], (uint32_t)q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], (uint32_t)q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// UE4M3 -> f32, software fallback (ggml_cuda_ue4m3_to_fp32 common.cuh:843-854). NaN 0/0x7F -> 0.
__device__ __forceinline__ float ue4m3_to_f32_d(unsigned char x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int   exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? ldexpf(man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}
// HW UE4M3 -> f32 (OCP E4M3, bias 7, NO x0.5). This is what the mxf4nvf4 block_scale MMA decodes
// its sa/sb operand as (verified by probe/fp4_4x_final.cu, maxrel=0). The GGUF NVFP4 micro-scale
// byte fed RAW here decodes to exactly 2x the GGUF value — which is cancelled by the e2m1 nibble
// being GGUF-codebook/2 (GGUF dequant = (2*e2m1_hw)*(0.5*ue4m3_hw) = e2m1_hw*ue4m3_hw). So GGUF
// scale bytes + GGUF e2m1 nibbles fed verbatim == GGUF dequant exactly. (used by quantize_fp4_act).
__device__ __forceinline__ float ue4m3_to_f32_hw(unsigned char x) {
    int   exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    return (exp == 0) ? ldexpf(man / 8.0f, -6) : ldexpf(1.0f + man / 8.0f, exp - 7);
}

// ---- Q5_K f32 deq (oracle for the dp4a kernel) ----
__device__ __forceinline__ float deq_q5_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 176;
    float d    = half_to_float(*(const uint16_t*)b);
    float dmin = half_to_float(*(const uint16_t*)(b + 2));
    const uint8_t* scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int group = jj >> 5;          // 0..7
    int l = jj & 31;
    int chunk = group >> 1;       // shares 32 qs bytes
    const uint8_t* q = ql + chunk * 32;
    uint8_t sc, mn;
    q4k_scale_min(scales, group, &sc, &mn);       // identical 6-bit unpack to Q4_K
    int g64 = group >> 1;
    int half = group & 1;
    int hbit = 2 * g64 + half;
    int nib = (half == 0) ? (q[l] & 0xF) : (q[l] >> 4);
    int h = (qh[l] >> hbit) & 1;
    int w = nib | (h << 4);                        // unsigned 0..31
    return d * (float)sc * (float)w - dmin * (float)mn;
}

// ---- Q3_K f32 deq ----
__device__ __forceinline__ float deq_q3_k(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    const uint8_t* hmask  = b;
    const uint8_t* qs     = b + 32;
    const uint8_t* scbyte = b + 96;
    float d = half_to_float(*(const uint16_t*)(b + 108));
    // unpack 16 6-bit signed scales (aux dance)
    unsigned int aux0 = (scbyte[0]) | (scbyte[1]<<8) | (scbyte[2]<<16) | (scbyte[3]<<24);
    unsigned int aux1 = (scbyte[4]) | (scbyte[5]<<8) | (scbyte[6]<<16) | (scbyte[7]<<24);
    unsigned int aux2 = (scbyte[8]) | (scbyte[9]<<8) | (scbyte[10]<<16) | (scbyte[11]<<24);
    const unsigned int km1 = 0x03030303u, km2 = 0x0f0f0f0fu, tmp = aux2;
    unsigned int n0 = (aux0 & km2) | (((tmp>>0)&km1)<<4);
    unsigned int n1 = (aux1 & km2) | (((tmp>>2)&km1)<<4);
    unsigned int n2 = ((aux0>>4)&km2) | (((tmp>>4)&km1)<<4);
    unsigned int n3 = ((aux1>>4)&km2) | (((tmp>>6)&km1)<<4);
    signed char sc[16];
    { unsigned int w[4] = {n0,n1,n2,n3};
      for (int k=0;k<4;k++){ sc[k*4+0]=(signed char)(w[k]); sc[k*4+1]=(signed char)(w[k]>>8);
                             sc[k*4+2]=(signed char)(w[k]>>16); sc[k*4+3]=(signed char)(w[k]>>24);} }
    // map jj (0..255) back to (half, j-iter, l, shift, m_bit, scale index)
    int half = jj >> 7;             // 0/1 (which 128)
    int rem  = jj & 127;            // 0..127
    int jiter = rem >> 5;           // 0..3 (which of the 4 j-iterations within the half)
    int within = rem & 31;          // 0..31 within the 32-wide j-iteration
    int sublo = within >> 4;        // 0 -> low 16 (sc index is_base), 1 -> high 16 (is_base+1)
    int l = within & 15;
    int shift = 2 * jiter;
    int m_bit_idx = half * 4 + jiter;          // running bit position (0..7)
    int is = (half * 8) + jiter * 2 + sublo;   // scale index 0..15
    const uint8_t* q = qs + half * 32;
    int qidx = sublo * 16 + l;                 // q[l] or q[l+16]
    int hidx = sublo * 16 + l;                 // hmask[l] or hmask[l+16]
    int q2 = (q[qidx] >> shift) & 3;
    int hb = (hmask[hidx] & (1 << m_bit_idx)) ? 0 : 4;
    int w = q2 - hb;
    return d * (float)((int)sc[is] - 32) * (float)w;
}

// ---- IQ4_XS f32 deq ----
__device__ __forceinline__ float deq_iq4_xs(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 136;
    float d = half_to_float(*(const uint16_t*)b);
    unsigned short sh = *(const uint16_t*)(b + 2);
    const uint8_t* sl = b + 4;
    const uint8_t* qs = b + 8;
    int ib = jj >> 5;               // 0..7
    int within = jj & 31;           // 0..31
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    float dl = d * (float)(ls - 32);
    const uint8_t* q = qs + ib * 16;
    int code = (within < 16) ? (q[within] & 0xf) : (q[within - 16] >> 4);
    return dl * (float)kvalues_iq4nl_d[code];
}

// ---- IQ3_S f32 deq ----
__device__ __forceinline__ float deq_iq3_s(const uint8_t* row, int j) {
    int blk = j >> 8, jj = j & 255;
    const uint8_t* b = row + blk * 110;
    float d = half_to_float(*(const uint16_t*)b);
    const uint8_t* qs    = b + 2;     // [64]
    const uint8_t* qh    = b + 66;    // [8]
    const uint8_t* signs = b + 74;    // [32]
    const uint8_t* scales= b + 106;   // [4]
    // Each ib32 group (32 elems) = qh[ib32], 4 sign bytes, 8 qs bytes. 8 elems per l (grid1/grid2).
    int ib32   = jj >> 5;             // 0..7
    int within = jj & 31;             // 0..31
    int l      = within >> 3;         // 0..3  (which qs pair)
    int e      = within & 7;          // 0..7  (grid byte slot)
    // ggml: db for even ib32 uses &0xf, odd uses >>4 of scales[ib32/2]
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const uint8_t* qsb = qs + ib32 * 8;       // 8 qs bytes per ib32
    unsigned char qhb = qh[ib32];
    const uint8_t* sgn = signs + ib32 * 4;
    int qpair = (e < 4) ? (2 * l + 0) : (2 * l + 1);
    int shamt = (e < 4) ? (8 - 2 * l) : (7 - 2 * l);
    int gidx = qsb[qpair] | (((int)qhb << shamt) & 256);
    int jb = e & 3;                            // grid byte 0..3
    unsigned int gw = iq3s_grid_d(gidx);
    int gval = (gw >> (8 * jb)) & 0xff;
    int sbit = (e < 4) ? jb : (jb + 4);
    float sign = (sgn[l] & (1 << sbit)) ? -1.0f : 1.0f;
    return db * (float)gval * sign;
}

// ---- NVFP4 f32 deq ----
__device__ __forceinline__ float deq_nvfp4(const uint8_t* row, int j) {
    int blk = j / 64, jj = j & 63;
    const uint8_t* b = row + blk * 36;
    const uint8_t* d_bytes = b;
    const uint8_t* qs = b + 4;
    int s = jj >> 4;            // sub-block 0..3
    int within = jj & 15;
    int byte = qs[s * 8 + (within & 7)];
    int code = (within < 8) ? (byte & 0xF) : (byte >> 4);
    return (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(d_bytes[s]);
}

// Exact 256-thread W4A16 row walk. Thread tid visits tid, tid+256, ... exactly as the scalar
// kernel, but hoists the invariant block/sub-block address calculation.
__device__ __forceinline__ float dot_nvfp4_bf16_row_256(
        const uint8_t* __restrict__ row,
        const unsigned short* __restrict__ x,
        int in_f) {
    const int tid = threadIdx.x;
    const int within_block = tid & 63;
    const int sub = within_block >> 4;
    const int within_sub = within_block & 15;
    const int q_offset = 4 + sub * 8 + (within_sub & 7);
    const bool high_nibble = within_sub >= 8;
    const int n_blocks = in_f >> 6;
    float acc = 0.0f;
    for (int block = tid >> 6; block < n_blocks; block += 4) {
        const uint8_t* b = row + (long)block * 36;
        const int packed = b[q_offset];
        const int code = high_nibble ? (packed >> 4) : (packed & 0xF);
        const int i = (block << 6) + within_block;
        const float xv = __uint_as_float((unsigned)x[i] << 16);
        acc += (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(b[sub]) * xv;
    }
    return acc;
}

// True gate+up twin of the exact row walk; the two independent accumulators only share each
// BF16 activation load.
__device__ __forceinline__ float2 dot_nvfp4_bf16_dual_row_256(
        const uint8_t* __restrict__ gate_row,
        const uint8_t* __restrict__ up_row,
        const unsigned short* __restrict__ x,
        int in_f) {
    const int tid = threadIdx.x;
    const int within_block = tid & 63;
    const int sub = within_block >> 4;
    const int within_sub = within_block & 15;
    const int q_offset = 4 + sub * 8 + (within_sub & 7);
    const bool high_nibble = within_sub >= 8;
    const int n_blocks = in_f >> 6;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int block = tid >> 6; block < n_blocks; block += 4) {
        const uint8_t* gate = gate_row + (long)block * 36;
        const uint8_t* up = up_row + (long)block * 36;
        const int gate_packed = gate[q_offset];
        const int up_packed = up[q_offset];
        const int gate_code =
            high_nibble ? (gate_packed >> 4) : (gate_packed & 0xF);
        const int up_code =
            high_nibble ? (up_packed >> 4) : (up_packed & 0xF);
        const int i = (block << 6) + within_block;
        const float xv = __uint_as_float((unsigned)x[i] << 16);
        acc.x +=
            (float)kvalues_mxfp4_d[gate_code] * ue4m3_to_f32_d(gate[sub]) * xv;
        acc.y +=
            (float)kvalues_mxfp4_d[up_code] * ue4m3_to_f32_d(up[sub]) * xv;
    }
    return acc;
}

// Two adjacent output rows from gate and up. Four independent accumulator chains preserve the
// scalar element order while sharing each BF16 activation load.
__device__ __forceinline__ float4 dot_nvfp4_bf16_quad_row_256(
        const uint8_t* __restrict__ gate_row0,
        const uint8_t* __restrict__ up_row0,
        const uint8_t* __restrict__ gate_row1,
        const uint8_t* __restrict__ up_row1,
        const unsigned short* __restrict__ x,
        int in_f) {
    const int tid = threadIdx.x;
    const int within_block = tid & 63;
    const int sub = within_block >> 4;
    const int within_sub = within_block & 15;
    const int q_offset = 4 + sub * 8 + (within_sub & 7);
    const bool high_nibble = within_sub >= 8;
    const int n_blocks = in_f >> 6;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int block = tid >> 6; block < n_blocks; block += 4) {
        const uint8_t* gate0 = gate_row0 + (long)block * 36;
        const uint8_t* up0 = up_row0 + (long)block * 36;
        const uint8_t* gate1 = gate_row1 + (long)block * 36;
        const uint8_t* up1 = up_row1 + (long)block * 36;
        const int gate0_packed = gate0[q_offset];
        const int up0_packed = up0[q_offset];
        const int gate1_packed = gate1[q_offset];
        const int up1_packed = up1[q_offset];
        const int gate0_code =
            high_nibble ? (gate0_packed >> 4) : (gate0_packed & 0xF);
        const int up0_code =
            high_nibble ? (up0_packed >> 4) : (up0_packed & 0xF);
        const int gate1_code =
            high_nibble ? (gate1_packed >> 4) : (gate1_packed & 0xF);
        const int up1_code =
            high_nibble ? (up1_packed >> 4) : (up1_packed & 0xF);
        const int i = (block << 6) + within_block;
        const float xv = __uint_as_float((unsigned)x[i] << 16);
        acc.x +=
            (float)kvalues_mxfp4_d[gate0_code] * ue4m3_to_f32_d(gate0[sub]) * xv;
        acc.y +=
            (float)kvalues_mxfp4_d[up0_code] * ue4m3_to_f32_d(up0[sub]) * xv;
        acc.z +=
            (float)kvalues_mxfp4_d[gate1_code] * ue4m3_to_f32_d(gate1[sub]) * xv;
        acc.w +=
            (float)kvalues_mxfp4_d[up1_code] * ue4m3_to_f32_d(up1[sub]) * xv;
    }
    return acc;
}

// ---- Q4_0 f32 deq (gemma4 QAT-Q4_0 checkpoints): 18B block per 32 elems = fp16 d + 16 nibble
// bytes; x[j] = d * (nib - 8); qs[i] holds elems i (lo nibble) and i+16 (hi nibble). ----
__device__ __forceinline__ float deq_q4_0(const uint8_t* row, int j) {
    const uint8_t* blk = row + (j / 32) * 18;
    float d = __half2float(*(const __half*)blk);
    const uint8_t* qs = blk + 2;
    int i = j % 32;
    int q = (i < 16) ? (qs[i] & 0xF) : (qs[i - 16] >> 4);
    return d * (float)(q - 8);
}

// ---- Q2_K f32 deq ----
__device__ __forceinline__ float deq_q2_k(const uint8_t* row, int j) {
    int blk = j >> 8;
    int jj = j & 255;
    const uint8_t* b = row + (long)blk * 84;
    const uint8_t* scales = b;
    const uint8_t* qs = b + 16;
    float d = half_to_float(*(const unsigned short*)(b + 80));
    float dmin = half_to_float(*(const unsigned short*)(b + 82));
    int within = jj & 127;
    int shift = 2 * (within >> 5);
    int q = (qs[(jj >> 7) * 32 + (within & 31)] >> shift) & 3;
    int sc = scales[jj >> 4];
    return d * (float)(sc & 0xf) * (float)q - dmin * (float)(sc >> 4);
}

enum QType { QT_Q8_0 = 0, QT_Q4_K = 1, QT_Q6_K = 2,
             QT_Q5_K = 3, QT_Q3_K = 4, QT_IQ4_XS = 5, QT_IQ3_S = 6, QT_NVFP4 = 7,
             QT_F32 = 8,
             // SPLIT-PLANE repacked NVFP4 (A6 walk-order repack): quant plane
             // [out_f x in_f/64 x 32B] followed by scale plane [out_f x in_f/64 x 4B].
             // Host-side tag only for the Stage-A generic kernel (GpuTensor keeps QT_NVFP4 +
             // an rp flag); deq() cannot express it (needs tensor base + out_f, not a row ptr).
             QT_NVFP4_RP = 9,
             // Slot-major per-row NVFP4 (tp.rs nvfp4_matrix_v2_permute; the resident expert slab
             // layout behind MEMRA_MOE_EXPERT_RP): quants at g*16, UE4M3 scale tail at nsb*16 + g*2.
             QT_NVFP4_V2 = 107,
             // CHECKPOINT-NATIVE FP8-E4M3 (MEMRA_ST_E4M3, lane e4m3dec 2026-07-08): the raw
             // safetensors e4m3 weight bytes [out_f, in_f] row-major (row_bytes == in_f), NO
             // per-32 scales — the per-tensor f32 weight_scale rides the host GpuTensor `scale`
             // (fused at the mmvq write, like the NVFP4 macro-scale). Weight side is EXACT
             // (the checkpoint's own precision; the Q8_0 re-encode this replaces was lossy).
             QT_F8_E4M3 = 10,
             // Raw BF16 row (FULL_PREC embed gather): 2 B/elem; f32 = bits << 16, exact.
             QT_BF16 = 11,
             // gemma4 QAT checkpoints (Q4_0 blocks, host tag 12 = lib.rs QT_Q4_0).
             QT_Q4_0 = 12,
             // Q2_K expert blocks. Generic staged f32-dequant path only for now.
             QT_Q2_K = 13 };

// e4m3 (OCP FP8, signed, bias 7) -> f32 via the native sm_89+ cvt (e4m3x2 -> f16x2 -> f32x2;
// every e4m3 value is exactly representable in f16, and f16 -> f32 is exact, so this chain is
// EXACT). Byte 0 of the ushort -> .x, byte 1 -> .y (little-endian, matches the cvt semantics).
__device__ __forceinline__ float2 e4m3x2_to_f32x2(unsigned short w2) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)w2, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&hr));
}
// Single-byte e4m3 -> f32 (Stage-A deq + scalar tails). Same exact chain as the x2 form.
__device__ __forceinline__ float e4m3_to_f32_d(unsigned char b) {
    __nv_fp8_e4m3 v; v.__x = (__nv_fp8_storage_t)b;
    return (float)v;
}

__device__ __forceinline__ float deq(int qtype, const uint8_t* row, int j) {
    switch (qtype) {
        case QT_Q8_0:   return deq_q8_0(row, j);
        case QT_Q4_K:   return deq_q4_k(row, j);
        case QT_Q6_K:   return deq_q6_k(row, j);
        case QT_Q5_K:   return deq_q5_k(row, j);
        case QT_Q3_K:   return deq_q3_k(row, j);
        case QT_IQ4_XS: return deq_iq4_xs(row, j);
        case QT_IQ3_S:  return deq_iq3_s(row, j);
        case QT_NVFP4:  return deq_nvfp4(row, j);
        case QT_Q2_K:   return deq_q2_k(row, j);
        // Unquantized f32 weight row (safetensors MoE Path A: experts gathered + dequantized to f32
        // host-resident, staged verbatim). `row` is the start of one out-row of `in_f` contiguous f32s.
        case QT_F32:    return ((const float*)row)[j];
        // Checkpoint-native e4m3 (MEMRA_ST_E4M3): 1 byte/element, row_bytes == in_f. The per-tensor
        // weight_scale is applied POST-matmul by the host (scale_inplace), like the NVFP4 macro-scale.
        case QT_F8_E4M3: return e4m3_to_f32_d(row[j]);
        // Raw bf16 (FULL_PREC embed): exact expansion, bit-identical to the host
        // f32::from_bits((bits as u32) << 16) contract.
        case QT_BF16: {
            unsigned int bits = ((const unsigned short*)row)[j];
            return __uint_as_float(bits << 16);
        }
        case QT_Q4_0:   return deq_q4_0(row, j);
    }
    return 0.0f;
}

// ---- Embed-from-device (CUDA-GRAPH-PLAN Phase 1): gather + dequant ONE token row whose id lives
// in a device u32 buffer (the argmax output), so the token never round-trips to host in steady
// state. x_out[j] = deq(qtype, embd_row(token_d[0]), j) for j in [0,n_embd). Bit-identical to host
// EmbedHost::gather (same per-dtype deq path). Single token (decode T=1). row_bytes = bytes/embed-row.
extern "C" __global__ void embed_gather_u32(
        const unsigned char* __restrict__ embd, const unsigned int* __restrict__ token_d,
        float* __restrict__ x_out, int n_embd, int qtype, long row_bytes) {
    unsigned int tok = token_d[0];
    const unsigned char* row = embd + (size_t)tok * row_bytes;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_embd; j += gridDim.x * blockDim.x)
        x_out[j] = deq(qtype, row, j);
}
// T-token variant (spec verify/replay): tokens_d[T] device ids -> x_out[T, n_embd]. grid.y = t.
// Replaces the host-side per-row dequant + ~T*14KB HtoD of EmbedHost::gather on the spec hot loop
// (nsys: cuMemcpyHtoDAsync was 84% of spec API time). Same per-dtype deq -> bit-identical rows.
extern "C" __global__ void embed_gather_u32_t(
        const unsigned char* __restrict__ embd, const unsigned int* __restrict__ tokens_d,
        float* __restrict__ x_out, int n_embd, int qtype, long row_bytes, int t) {
    int ti = blockIdx.y;
    if (ti >= t) return;
    unsigned int tok = tokens_d[ti];
    const unsigned char* row = embd + (size_t)tok * row_bytes;
    float* xr = x_out + (size_t)ti * n_embd;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_embd; j += gridDim.x * blockDim.x)
        xr[j] = deq(qtype, row, j);
}

// ---- Device i32 increment (CUDA-GRAPH-PLAN Phase 1): pos_d[0] += 1 inside the captured graph,
// replacing the per-step host htod_i32(&[pos]). One thread.
extern "C" __global__ void inc_i32(int* __restrict__ p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1; }

// ================= Stage-B: int8 dp4a MMVQ (decode hot path) =================
// Quantize activation row to q8_1 blocks (32 vals -> int8 + fp16 scale d), then weight-int8 dot.
// Activation buffer layout per block i: [32 int8 qs][1 float d]. We pack as: int8 qs in a byte array
// + a parallel float array of per-block d. Done in a tiny kernel below.

// dp4a: 4x int8 dot accumulate (sm_61+). Available on sm_120.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    int r = c;
    for (int i = 0; i < 4; i++) { int8_t x = (a >> (i*8)) & 0xff, y = (b >> (i*8)) & 0xff; r += x*y; }
    return r;
#endif
}

// Quantize an [m, in] f32 activation matrix to q8_1: out_q (int8 [m, in]) + out_d (f32 [m, in/32]).
// One block per (token, block-of-32). amax over 32, d=amax/127, qs=round(x/d).
// WARP-PER-BLOCK (decode elementwise-soup fix, ncu 2026-07-03): lane j owns element j of one
// 32-block -> coalesced 128B read + 32B write, vs the old thread-owns-block 32-way strided form.
// __shfl_xor max reduce is order-independent -> d and q8 values BIT-IDENTICAL to the old kernel.
extern "C" __global__ void quantize_q8_1(const float* __restrict__ x, signed char* __restrict__ out_q,
                                         float* __restrict__ out_d, int in_f, int m) {
    MEMRA_PDL_ENTRY();
    // 64-bit thread-id math (audit Q7 defense-in-depth): m*in_f >= 2^31 needs M >= ~52k tokens
    // at the largest in_f — out of today's chunked range, but the widening is free.
    long long blk = ((long long) blockIdx.x * blockDim.x + threadIdx.x) >> 5; // global 32-block idx
    int lane = threadIdx.x & 31;
    int nblk_row = in_f / 32;
    if (blk >= (long long) m * nblk_row) return;
    int t = (int)(blk / nblk_row);
    int b = (int)(blk % nblk_row);
    size_t off = (size_t)t * in_f + b * 32 + lane;
    float v = x[off];
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[off] = (signed char)__float2int_rn(v * id);
    if (lane == 0) out_d[(size_t)t * nblk_row + b] = d;
}

// ================= Stage-C: FP4 (e2m1) activation quantize for the mxf4 block-scale GEMM =========
// Quantize an [m, in] f32 activation to NVFP4-style e2m1 nibbles + per-16 UE4M3 scale, in the EXACT
// layout the mxf4nvf4 GEMM B-fragment gather wants (verified by probe/fp4_4x_*.cu):
//   aq4 : u32 [m][in_f/8]  — nibble (k&7) of word (k/8) = e2m1 code of activation element k
//   ad4 : u8  [m][in_f/16] — one UE4M3 scale byte per 16-elem K block
// e2m1 magnitudes: {0,0.5,1,1.5,2,3,4,6}; HW value of a nibble == kvalues here are GGUF-codebook
// (=2x HW e2m1); but for the B operand we feed RAW e2m1 nibbles whose HW value is the GGUF/2. So we
// must encode x/d to the *HW* e2m1 grid (max 6.0). The UE4M3 d is chosen so amax/d <= 6.
// HW UE4M3 (OCP E4M3, bias 7, NO x0.5): enc/dec below. Scale stored as the HW byte (NOT the GGUF
// 0.5x form) — the GEMM treats sb as HW UE4M3.
__device__ __forceinline__ int e2m1_encode_hw(float v) {
    // nearest of the 8 signed e2m1 grid points {0,.5,1,1.5,2,3,4,6}. sign bit = bit3.
    float a = fabsf(v);
    int code;
    // round-to-nearest on the irregular grid
    if (a < 0.25f) code = 0;            // 0
    else if (a < 0.75f) code = 1;       // 0.5
    else if (a < 1.25f) code = 2;       // 1.0
    else if (a < 1.75f) code = 3;       // 1.5
    else if (a < 2.5f) code = 4;        // 2.0
    else if (a < 3.5f) code = 5;        // 3.0
    else if (a < 5.0f) code = 6;        // 4.0
    else code = 7;                      // 6.0
    if (code != 0 && v < 0.0f) code |= 0x8;
    return code;
}
// HW UE4M3 encode of a NON-NEGATIVE scale s: round to nearest E4M3 (bias 7, no x0.5). Clamp [2^-9, 448].
__device__ __forceinline__ unsigned char ue4m3_encode_hw(float s) {
    if (!(s > 0.0f)) return 0;
    s = fminf(s, 448.0f);
    int e; float m = frexpf(s, &e);    // s = m*2^e, m in [0.5,1)
    // normalized: s = 2^(E-7)*(1+man/8), E = exponent field (1..15), man 0..7
    int E = e - 1 + 7;                 // since m in [0.5,1): s = 2^(e-1)*(2m), 2m in [1,2)
    float frac = 2.0f * m - 1.0f;      // in [0,1)
    if (E <= 0) {                      // subnormal: s = (man/8)*2^-6
        float q = s * 64.0f * 8.0f;    // man = round(s / 2^-9)
        int man = (int)(q + 0.5f);
        if (man > 7) man = 7;
        return (unsigned char)man;     // E=0
    }
    int man = (int)(frac * 8.0f + 0.5f);
    if (man == 8) { man = 0; E += 1; }
    if (E > 15) { E = 15; man = 7; }
    return (unsigned char)((E << 3) | man);
}
// One CTA-thread per (token, 16-block). amax over 16 -> UE4M3 d (so amax/d ~ 6) -> e2m1 encode.
extern "C" __global__ void quantize_fp4_act(const float* __restrict__ x, unsigned* __restrict__ aq4,
                                            unsigned char* __restrict__ ad4, int in_f, int m) {
    // 64-bit thread-id math (audit Q7 defense-in-depth) — see quantize_q8_1.
    long long b16 = (long long) blockIdx.x * blockDim.x + threadIdx.x; // global 16-block index
    int nb16_row = in_f / 16;
    if (b16 >= (long long) m * nb16_row) return;
    int t = (int)(b16 / nb16_row);
    int blk = (int)(b16 % nb16_row);
    const float* xr = x + (size_t)t * in_f + blk * 16;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++) amax = fmaxf(amax, fabsf(xr[j]));
    // choose d so that amax/d == 6 (the e2m1 max). d ~ amax/6, quantized to UE4M3.
    float dwant = amax > 0.0f ? amax / 6.0f : 0.0f;
    unsigned char db = ue4m3_encode_hw(dwant);
    float d = ue4m3_to_f32_hw(db);
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    ad4[(size_t)t * nb16_row + blk] = db;
    // encode 16 nibbles into two u32 words (k/8 within the 16-block -> word blk*2 + (k/8)).
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned w = 0;
        #pragma unroll
        for (int n = 0; n < 8; n++) {
            int code = e2m1_encode_hw(xr[half * 8 + n] * id);
            w |= ((unsigned)code) << (4 * n);
        }
        aq4[((size_t)t * (in_f / 8)) + blk * 2 + half] = w;
    }
}

// Block reduce shared by the dp4a MMVQ kernels: full-warp shfl, then warp0 sums the per-warp
// partials. Correct for any blockDim.x that is a multiple of 32 (used with 128 = 4 warps).
__device__ __forceinline__ void mmvq_block_reduce_write(float acc, float* __restrict__ y,
                                                        size_t out_idx, int tid) {
    __shared__ float s[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[out_idx] = v;
    }
}

// Vectorized weight-int load: 4 int8 starting at `p` (only 2-byte aligned in Q8_0 -> uint16x2).
// Mirrors llama.cpp get_int_b2 (vecdotq.cuh:18-25). Safe for any 2-byte-aligned source.
__device__ __forceinline__ int get_int_b2(const void* p) {
    // NOT streaming: only 2-byte aligned — an __ldcs pair here split the single 32-bit
    // load and cost −0.8% on the 31B depth cell (2026-07-14 probe); b4 streams instead.
    const unsigned short* u = (const unsigned short*)p;
    return (int)u[0] | ((int)u[1] << 16);
}

// Vectorized weight-int load: 4 int8 starting at `p`, single 32-bit LDG. Mirrors llama get_int_b4
// (vecdotq.cuh). Safe for any 4-byte-aligned source. NVFP4 qss is provably 4-aligned
// (row_bytes=(in_f/64)*36 -> mult of 4; qs=b+4; qss=qs+s*8) so the qs stream qualifies. Do NOT
// widen to int2/LDG.E.64 there: rows are only 8-aligned when in_f%128==0 -> faults on odd in_f/64.
__device__ __forceinline__ int get_int_b4(const void* p) {
    return __ldcs((const int*)p);   // streaming: weight-only helper (see get_int_b2)
}

// Two NVFP4 rows over one q8_1 activation row. Each accumulator retains the established dp4a
// and floating-add order; only the activation bytes and scales are shared.
__device__ __forceinline__ float2 dot_nvfp4_q8_dual_row(
        const unsigned char* __restrict__ row0,
        const unsigned char* __restrict__ row1,
        const signed char* __restrict__ arow,
        const float* __restrict__ adrow,
        int in_f) {
    const int tid = threadIdx.x;
    const int nsb = in_f >> 5;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int g = tid; g < nsb; g += blockDim.x) {
        const int sblk = g >> 1;
        const int which_half = g & 1;
        const unsigned char* b0 = row0 + (long)sblk * 36;
        const unsigned char* b1 = row1 + (long)sblk * 36;
        const int s0 = which_half * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        const int4 a01 = aq16[0], a23 = aq16[1];
        const int aq4[8] = {
            a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w
        };
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; ++sl) {
            const int sub = s0 + sl;
            const unsigned char* q0 = b0 + 4 + sub * 8;
            const unsigned char* q1 = b1 + 4 + sub * 8;
            const int2 a0 =
                get_int_from_table_16_d(get_int_b4(q0), kvalues_mxfp4_d);
            const int2 b0v =
                get_int_from_table_16_d(get_int_b4(q0 + 4), kvalues_mxfp4_d);
            const int2 a1 =
                get_int_from_table_16_d(get_int_b4(q1), kvalues_mxfp4_d);
            const int2 b1v =
                get_int_from_table_16_d(get_int_b4(q1 + 4), kvalues_mxfp4_d);
            const int base = sl * 4;
            int sum0 = 0;
            sum0 = dp4a(a0.x, aq4[base + 0], sum0);
            sum0 = dp4a(b0v.x, aq4[base + 1], sum0);
            sum0 = dp4a(a0.y, aq4[base + 2], sum0);
            sum0 = dp4a(b0v.y, aq4[base + 3], sum0);
            int sum1 = 0;
            sum1 = dp4a(a1.x, aq4[base + 0], sum1);
            sum1 = dp4a(b1v.x, aq4[base + 1], sum1);
            sum1 = dp4a(a1.y, aq4[base + 2], sum1);
            sum1 = dp4a(b1v.y, aq4[base + 3], sum1);
            partial0 += ue4m3_to_f32_d(b0[sub]) * (float)sum0;
            partial1 += ue4m3_to_f32_d(b1[sub]) * (float)sum1;
        }
        acc.x += adrow[g] * partial0;
        acc.y += adrow[g] * partial1;
    }
    return acc;
}

// ============================ Stage-B MMVQ (warp-per-row decode) ============================
// PERF-3 (DECODE-GEMV-PLAN): warp-per-row layout matching llama.cpp mmvq.cu. block=(32,ROWS,1):
// one WARP (threadIdx.y) owns one output row. Reduction is warp-only __shfl_xor_sync (no smem,
// no __syncthreads — removes the cross-warp barrier from the m=1 critical path). The per-element
// DEQUANT BODIES are LIFTED VERBATIM from the matching _dp4a kernels (same get_int_b2/codebook
// math), so the int sumi is bit-for-bit identical; only the layout + reduction order change.
// ROWS_PER_BLOCK = 4 (128 threads, 4 independent rows in flight) is llama's GENERIC ncols_dst=1.
#define MEMRA_MMVQ_ROWS 4

// Warp-only reduce: full-warp shfl-xor (butterfly), all lanes hold the sum. No smem/barrier.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ----- Q8_0 warp-per-row MMVQ. Body lifted from qmatvec_q8_0_dp4a (loop @ ~line 398). -----
extern "C" __global__ void qmatvec_q8_0_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;                              // 0..31
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {        // per-warp contiguous stride (32 lanes)
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);   // 2-byte aligned OK
        const unsigned char* wq = wb + 2;                       // qs: 2-byte aligned -> get_int_b2
        const int4* aq16 = (const int4*)(arow + blk * 32);      // 32-aligned -> 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q8_0 m=1 single-row body shared by the FUSED multi-tensor launches below. LIFTED VERBATIM
// from qmatvec_q8_0_mmvq with t pinned to 0 (decode m==1): same block walk, same dp4a order, same
// warp_reduce_sum, same write -> per (tensor,row) output bits identical to a separate m=1 launch. -----
__device__ __forceinline__ void q8_0_mmvq_row1(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);
        const unsigned char* wq = wb + 2;
        const int4* aq16 = (const int4*)(aq + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * ad[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc;
}

// ----- FUSED Q8_0 m=1 matvec PAIR, UNEQUAL out_f (trunk launch-fusion, 2026-07-05). The 35B trunk
// decode runs ~250 tiny q8_0 m=1 launches/token (2.4-8us, launch-latency class: 44k of the 48-tok
// window's 160k-class launches are this kernel). Same-input projections fold into ONE grid: blocks
// [0,nb0) compute tensor 0, [nb0,nb0+nb1) tensor 1 (the NVFP4 gate+up dual + beta/alpha dual proved
// the recipe; this variant lifts the same-out_f restriction via a block-offset split instead of
// blockIdx.y). Both tensors share in_f (Q8_0 row_bytes is a pure function of in_f -> ONE row_bytes)
// and the SAME q8_1 activation. Per (tensor,row) the body is qmatvec_q8_0_mmvq VERBATIM ->
// BIT-IDENTICAL to two separate m=1 launches. Targets: 35B wqkv+wqkv_gate (8192/4096),
// gate_shexp+up_shexp (512/512). Seam MEMRA_Q8_DUAL=0 (host-side). -----
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0) { W = W0; y = y0; out_f = out0; }
    else         { W = W1; y = y1; out_f = out1; b -= nb0; }
    q8_0_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- FUSED Q8_0 m=1 matvec TRIPLE (wq+wk+wv: same input h, same in_f, out_f 8192/512/512 on
// the 35B full-attn layers). Same block-offset recipe as fused2 with three ranges. -----
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int nb1 = (out1 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; b -= nb0 + nb1; }
    q8_0_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- f32 warp-per-row matvec body for the KDA fused-6 launch below. NOT a lift: the unfused
// arm for a Float-resident weight is cuBLASLt f32 GEMV, whose internal reduction order is not
// reproducible in a hand kernel. This body is a DETERMINISTIC replacement (lane-strided float4
// walk + warp_reduce_sum) -> a reduction-order numeric-class change for exactly these rows, the
// step37 MEMRA_STEP_TP_QKV_FUSED acceptance class ("2.4e-3 -> 4.2e-3"), measured and pinned by
// crates/memra-engine/tests/kda_fused_proj_gpu.rs. Requires in_f % 128 == 0 (32 lanes x float4),
// host-guarded. -----
__device__ __forceinline__ void f32_mmvq_row1(
        const float* __restrict__ W, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    const float* wr = W + (size_t)o * in_f;
    float acc = 0.0f;
    for (int i = lane * 4; i < in_f; i += 32 * 4) {
        float4 w4 = *reinterpret_cast<const float4*>(wr + i);
        float4 x4 = *reinterpret_cast<const float4*>(x + i);
        acc += w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc;
}

// ----- FUSED KDA 6-way projection matvec (lane/glm5-launch-diet, 2026-08-30). The glm5_next KDA
// stage-1 group (wq|wk|wv Q8_0-resident + f_a|g_a|b_proj f32-resident, ONE shared input) runs as
// ONE launch instead of 3x(quantize+mmvq) + 3x cuBLASLt GEMV per layer per token. Same
// block-offset range split as qmatvec_q8_0_mmvq_fused2/3 above, extended to six unequal ranges
// and t rows (blockIdx.y, the 1..=15 decode/verify widths). Per (t,row) the Q8_0 body is
// q8_0_mmvq_row1 on the t-offset activation slices = qmatvec_q8_0_mmvq VERBATIM ->
// BIT-IDENTICAL to the separate MMVQ/batched launches; the f32 rows take f32_mmvq_row1 (numeric
// class change vs cuBLASLt, see above). Both engines upstream ship this exact program shape
// (vLLM in_proj_qkvbfg_a "6 to 1 launches"; SGLang fused_qkvbfg_a_proj) — design copied, no
// kernel code (ENGINE-SURVEY.md C1). Host seam MEMRA_KDA_FUSED_PROJ (default OFF). -----
extern "C" __global__ void qmatvec_kda6_q8f32_mmvq(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m, long row_bytes) {
    int t = blockIdx.y;
    if (t >= m) return;
    int nblk = in_f / 32;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    const float* xrow = x + (size_t)t * in_f;
    int b = blockIdx.x;
    int o = 0; // row within the selected tensor, assigned below
    int nb;
    nb = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    if (b < nb) {
        o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
        q8_0_mmvq_row1(W0, arow, adrow, y0 + (size_t)t * out0, in_f, out0, row_bytes, o);
        return;
    }
    b -= nb;
    nb = (out1 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    if (b < nb) {
        o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
        q8_0_mmvq_row1(W1, arow, adrow, y1 + (size_t)t * out1, in_f, out1, row_bytes, o);
        return;
    }
    b -= nb;
    nb = (out2 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    if (b < nb) {
        o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
        q8_0_mmvq_row1(W2, arow, adrow, y2 + (size_t)t * out2, in_f, out2, row_bytes, o);
        return;
    }
    b -= nb;
    nb = (out3 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    if (b < nb) {
        o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
        f32_mmvq_row1(W3, xrow, y3 + (size_t)t * out3, in_f, out3, o);
        return;
    }
    b -= nb;
    nb = (out4 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    if (b < nb) {
        o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
        f32_mmvq_row1(W4, xrow, y4 + (size_t)t * out4, in_f, out4, o);
        return;
    }
    b -= nb;
    o = b * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
    f32_mmvq_row1(W5, xrow, y5 + (size_t)t * out5, in_f, out5, o);
}

// ----- bf16 4-rows-per-block body for the BF16 fused-6 launch below: the inner loop of
// matvec_bf16_f32acc_x4_rows VERBATIM (same 8-wide uint4 weight loads, same bits<<16
// expansion, same per-thread acc chain, same red[] block tree at the SAME blockDim —
// the launcher pins mmv_block() exactly like matvec_bf16_rows_into). BIT-IDENTICAL per
// row to the unfused kernel by construction; asserted bytewise by
// crates/memra-engine/tests/kda_fused_proj_bf16_gpu.rs. -----
__device__ __forceinline__ void kda6_bf16_rows4(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int blk) {
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blk * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
#pragma unroll 4
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(x + i);
            float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[row] = red[0];
        __syncthreads();
    }
}

// ----- FUSED KDA 6-way projection matvec, BF16 operand arm (lane/glm5-decode-diet lever 3,
// 2026-08-31). The SERVING-RECIPE twin of qmatvec_kda6_q8f32_mmvq above: under
// MEMRA_BF16_MMV=1 the loader admits wq/wk/wv to raw bf16 residency (`admit=bf16_mmv`), so
// the q8 arm refuses there by design and the unfused stage-1 group runs as 3x
// matvec_bf16_f32acc_x4_rows + 3x cuBLASLt f32 GEMV pairs. This kernel runs the six in ONE
// launch: the bf16 ranges take kda6_bf16_rows4 (per-row program VERBATIM -> BIT-IDENTICAL),
// the f32 ranges take f32_mmvq_row1 (the SAME deterministic warp tree the q8 arm's f32 rows
// use — the gated cuBLASLt-replacement numeric class, measured and pinned). Block =
// mmv_block() like the unfused bf16 launcher (the red[] tree shape depends on blockDim, so
// it is part of the bit-identity claim); each block owns 4 rows of one range; on the f32
// ranges warps 1+ compute discarded partials (rows are warp-0-owned) — waste bounded by the
// three small f32 ranges, correctness unaffected. Host seam MEMRA_KDA_FUSED_PROJ (default
// OFF, same door as the q8 arm). -----
extern "C" __global__ void qmatvec_kda6_bf16f32(
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,
        const unsigned short* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    int t = blockIdx.y;
    if (t >= m) return;
    const float* xrow = x + (size_t)t * in_f;
    int b = blockIdx.x;
    int nb;
    nb = (out0 + 3) / 4;
    if (b < nb) {
        kda6_bf16_rows4(W0, xrow, y0 + (size_t)t * out0, in_f, out0, b);
        return;
    }
    b -= nb;
    nb = (out1 + 3) / 4;
    if (b < nb) {
        kda6_bf16_rows4(W1, xrow, y1 + (size_t)t * out1, in_f, out1, b);
        return;
    }
    b -= nb;
    nb = (out2 + 3) / 4;
    if (b < nb) {
        kda6_bf16_rows4(W2, xrow, y2 + (size_t)t * out2, in_f, out2, b);
        return;
    }
    b -= nb;
    nb = (out3 + 3) / 4;
    if (b < nb) {
        for (int p = 0; p < 4; p++)
            f32_mmvq_row1(W3, xrow, y3 + (size_t)t * out3, in_f, out3, b * 4 + p);
        return;
    }
    b -= nb;
    nb = (out4 + 3) / 4;
    if (b < nb) {
        for (int p = 0; p < 4; p++)
            f32_mmvq_row1(W4, xrow, y4 + (size_t)t * out4, in_f, out4, b * 4 + p);
        return;
    }
    b -= nb;
    for (int p = 0; p < 4; p++)
        f32_mmvq_row1(W5, xrow, y5 + (size_t)t * out5, in_f, out5, b * 4 + p);
}

// ----- Q4_K warp-per-row MMVQ. Body lifted from qmatvec_q4_K_dp4a (loop @ ~line 427). -----
extern "C" __global__ void qmatvec_q4_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        const int* q4 = (const int*)(qs + chunk * 32);          // 4-byte aligned
        bool hi = (grp & 1);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int raw = q4[k];
            int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d) * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q5_K warp-per-row MMVQ. Body lifted from qmatvec_q5_K_dp4a (the only major decode matvec that
// still fell to the slow dp4a path at m=1 — 7% of 9B decode). One warp owns one output row; lane
// strides the 32-blocks; warp-only shfl reduce (no smem barrier). Bit-equivalent to qmatvec_q5_K_dp4a
// up to f32 reduction order (same vectorized q5_K unpack + dp4a + min-offset). -----
extern "C" __global__ void qmatvec_q5_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);
            int qh4 = get_int_b2(qh + k * 4);
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;
            int wpack = low | (h << 4);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d)   * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- Q5_K MULTI-ROW-PER-WARP MMVQ (the FR-Spec trimmed draft head is Q5_K 32768 rows — 8% of
// the 27B p3 spec wall at 1.02ms/draft launch, memory-latency bound like the other k-quants). Same
// multirow recipe as q4k_mmvq_multirow: activation int8 loaded ONCE (2x int4), reused across RPW
// rows; RPW weight rows in flight hide the load latency. BIT-IDENTICAL per row to qmatvec_q5_K_mmvq
// (same scale/min unpack, same qh bit extraction, same dp4a order, same warp_reduce_sum). -----
template<int RPW>
__device__ __forceinline__ void q5k_mmvq_multirow(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        // activation loaded ONCE, reused across RPW rows (+ the min-sum, row-independent).
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 176;
            float d_sb    = half_to_float(*(const unsigned short*)b);
            float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qh = b + 16;
            const unsigned char* qs = b + 48;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            const unsigned char* q = qs + g64 * 32;
            int sumi_d = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int q4  = get_int_b2(q  + k * 4);
                int qh4 = get_int_b2(qh + k * 4);
                int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
                int h   = (qh4 >> hbit) & 0x01010101;
                int wpack = low | (h << 4);
                sumi_d = dp4a(wpack, aq4[k], sumi_d);
            }
            acc[r] += d_sb * (float)((int)sc * sumi_d) * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- Q5_K ISSUE-REDUCED MMVQ family (`_il`, q5issue lane 2026-07-08, MEMRA_Q5K_ISSUE=1) ----
// WHY: the q5_K mmvq family sits at ~61% of the bandwidth wall on the 27B lm_head (1030us) —
// the k-quant mmvq ceiling is instruction-ISSUE, not loads (q6krp repack: exactly 0 gain; the
// down8 byte_perm decode: +2.5% e2e). Per 32-elem group the reference body issues 2 LDG.U16
// (d/dmin) + a warp-DIVERGENT grp<4 / grp>=4 scale unpack (2-4 LDG.U8 byte loads, both paths
// serialized every iteration since grp = lane&7 splits the warp) + 16 LDG.32 (get_int_b2 qs/qh)
// = ~21 LSU ops + divergence against 16 dp4a of real work. This body produces the IDENTICAL
// packed ints from the IDENTICAL bytes with 5 LDG.128: one uint4 header (d|dmin|scales[12]),
// 2x uint4 qh (the whole 32B plane), 2x uint4 qs — and replaces the divergent scale branch with
// branchless register extraction (both paths computed + select on the loop-invariant grp>=4).
// BIT-IDENTITY (value-level): the uint4 components ARE the little-endian 32-bit words
// get_int_b2 builds (q5_K block=176B: b, b+16, b+48+g64*32 all 16-aligned when W is);
// (q4 >> sh4) & M with sh4 = hi*4 == the `hi ? (q4>>4)&M : q4&M` select (>>0 is identity);
// the scale/min register math lands the exact scales[] bytes the branchy path loads
// (hdr.y/z/w = scales[0..3]/[4..7]/[8..11], byte j via >>8j). The dp4a chain order (k
// ascending, sumi_d/sumi_sum separate integer chains) and the closing float expression are
// UNCHANGED, so outputs are bit-identical per (token,row).
// ALIGNMENT: q5_K row_bytes = (in_f/256)*176, a multiple of 16 -> every superblock pointer is
// 16-aligned iff W is (cudaMalloc slabs are 256B-aligned; every real dispatch passes the
// tensor-base slice). A GRID-UNIFORM guard falls back to the reference body for exotic bases.
template<int RPW>
__device__ __forceinline__ void q5k_mmvq_multirow_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    if ((((unsigned long long)W | (unsigned long long)row_bytes) & 15ull) != 0ull) {
        q5k_mmvq_multirow<RPW>(W, aq, ad, y, in_f, out_f, m, row_bytes);  // reference fallback
        return;
    }
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    // decode geometry is loop-invariant per lane: g = lane + 32*i -> g&7 == lane&7.
    int grp  = lane & 7;
    int g64  = grp >> 1;
    int sh4  = (grp & 1) * 4;        // 0 for the low-nibble plane, 4 for the high
    int hbit = 2 * g64 + (grp & 1);
    bool hi4 = grp >= 4;
    int sh8  = 8 * (grp & 3);        // byte j of the scale words, j = grp or grp-4
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sumi_sum = 0;                        // row-independent activation sum (shared)
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 176;
            uint4 hdr = *(const uint4*)b;        // d|dmin (4B) + scales[12] in one LDG.128
            float d_sb    = half_to_float((unsigned short)(hdr.x & 0xffffu));
            float dmin_sb = half_to_float((unsigned short)(hdr.x >> 16));
            unsigned by = (hdr.y >> sh8) & 0xffu;   // scales[j]
            unsigned bz = (hdr.z >> sh8) & 0xffu;   // scales[j+4]
            unsigned bw = (hdr.w >> sh8) & 0xffu;   // scales[j+8]
            int sc = hi4 ? (int)((bw & 0xFu) | ((by >> 6) << 4)) : (int)(by & 63u);
            int mn = hi4 ? (int)((bw >> 4)   | ((bz >> 6) << 4)) : (int)(bz & 63u);
            const uint4* qhv = (const uint4*)(b + 16);            // whole 32B qh plane
            uint4 h01 = qhv[0], h23 = qhv[1];
            const uint4* qsv = (const uint4*)(b + 48 + g64 * 32); // 32B nibble plane
            uint4 q01 = qsv[0], q23 = qsv[1];
            int qw[8]  = { (int)q01.x, (int)q01.y, (int)q01.z, (int)q01.w,
                           (int)q23.x, (int)q23.y, (int)q23.z, (int)q23.w };
            int qhw[8] = { (int)h01.x, (int)h01.y, (int)h01.z, (int)h01.w,
                           (int)h23.x, (int)h23.y, (int)h23.z, (int)h23.w };
            int sumi_d = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int low = (qw[k] >> sh4) & 0x0F0F0F0F;
                int h   = (qhw[k] >> hbit) & 0x01010101;
                int wpack = low | (h << 4);
                sumi_d = dp4a(wpack, aq4[k], sumi_d);
            }
            acc[r] += d_sb * (float)(sc * sumi_d) * d8
                    - dmin_sb * (float)(mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a;
    }
}
// Single-row twin: RPW=1 of the multirow body is bit-identical to qmatvec_q5_K_mmvq (the only
// difference is sumi_sum computed before sumi_d — separate exact integer chains; the float
// expression and per-g accumulation order are unchanged). Fallback likewise goes to
// q5k_mmvq_multirow<1>, bit-identical to the reference single-row kernel.
extern "C" __global__ void qmatvec_q5_K_mmvq_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow_il<1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_mr2_il(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_multirow_il<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}



// ----- Q6_K warp-per-row MMVQ. Body lifted from qmatvec_q6_K_dp4a (loop @ ~line 476). -----
extern "C" __global__ void qmatvec_q6_K_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    MEMRA_PDL_ENTRY();
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        // VECTORIZED unpack (was a scalar 4-byte inner loop = ~20 ALU ops/k starving DRAM to 19%).
        // For each k the 4 ql bytes (il=k*4..k*4+3) and 4 qh bytes are CONTIGUOUS -> read each as one
        // 32-bit word (get_int_b2: 2-aligned-safe, q6_K block=210 is even) and extract all 4 nibbles/
        // 2-bit groups with SIMD masks. BIT-IDENTICAL: get_int_b2 packs byte e at bit e*8, exactly the
        // old `<<(e*8)` order; per-byte ql_bits|(qh_bits<<4) and __vsubss4 are unchanged.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ql4 = get_int_b2(qlh + k * 4 + ql_off);          // 4 ql bytes
            int qh4 = get_int_b2(qhh + k * 4);                   // 4 qh bytes
            int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
            int qhn = (qh4 >> qh_sh) & 0x03030303;               // 2-bit group per byte, 0..3
            int vpack = qln | (qhn << 4);                        // per byte = ql_bits|(qh_bits<<4), 0..63
            int wpack = __vsubss4(vpack, 0x20202020);            // subtract 32 per byte (signed sat)
            int a = aq4[k];
            if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
            else       sumi1 = dp4a(wpack, a, sumi1);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- NVFP4 warp-per-row MMVQ. Body lifted from qmatvec_nvfp4_dp4a (loop @ ~line 674). -----
extern "C" __global__ void qmatvec_nvfp4_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;          // which 64-elem block_nvfp4 (36 bytes)
        int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2;
        // activation 32 int8 = 8 ints: load as 2x int4 (16B) -> cuts 8 LDG.E.32 to 2 LDG.E.128,
        // attacking lg_throttle (3.82, LSU queue full). aqb = arow + g*32 is 32-aligned -> int4 safe.
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];   // aq4[0..3]
        int4 a23 = aq16[1];   // aq4[4..7]
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;
            int q4a = get_int_b4(qss);      // P1: single LDG.E.32 (was 4x LDG.E.U8); qss 4-aligned
            int q4b = get_int_b4(qss + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * yscale;
}

// ---- NVFP4 MMVQ, MULTI-ROW-PER-WARP (MLP lever). The single-row mmvq above is m=1 LATENCY-bound
// (ncu: 30-46% DRAM, loads-in-flight starved — one acc chain per warp waits on each weight LDG
// before the next dp4a). This variant has ONE warp compute RPW output rows in ONE pass over the
// shared activation: the activation int8 (loaded once as 2x int4) is REUSED across all RPW rows, and
// RPW independent weight rows are loaded + RPW independent acc chains run per iteration -> RPW x the
// memory-level parallelism, hiding the weight-load latency WITHOUT a cross-warp reduce barrier (the
// barrier was why more-WARPS-per-row was slower; more-ROWS-per-warp has no barrier). Activation
// bytes leave HBM/L2 1x per warp instead of 1x per row. BIT-IDENTICAL per row to qmatvec_nvfp4_mmvq:
// same dp4a order, same ue4m3 scale, same warp_reduce_sum, same write. grid.x sized for RPW rows/warp.
// yscale = the per-tensor NVFP4 macro-scale, applied AT THE WRITE (y = reduced_acc * yscale).
// Bit-identical to the old separate scale_inplace pass (same single IEEE multiply on the same
// value); folding it removes one launch per matvec (53 scale_f32 launches/token on the 9B).
template<int RPW>
__device__ __forceinline__ void nvfp4_mmvq_multirow(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;   // first of this warp's RPW rows
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        int s0 = whichHalf * 2;
        // activation loaded ONCE, reused across all RPW rows.
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];
        int4 a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}
// t=0-pinned single-token body of nvfp4_mmvq_multirow (blockIdx.y is repurposed by the dual
// kernel for tensor select). SAME dp4a order / scales / reduce as the multirow helper -> the
// dual kernel's per-element results are bit-identical to the mr2 kernel at m=1.
template<int RPW>
__device__ __forceinline__ void nvfp4_mmvq_dual_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char*   arow = aq;
    const float*         adrow = ad;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        int s0 = whichHalf * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0];
        int4 a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            const unsigned char* b = W + (long)o * row_bytes + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[o] = a * yscale;
    }
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    nvfp4_mmvq_multirow<2>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}
// DUAL gate+up matvec (mm-fusion, 2026-07-03): the FFN gate and up projections share the SAME
// activation and the same (in_f, out_f) shape; running them as two sequential launches leaves the
// tail of each under-filled and pays two launch latencies. ONE grid computes both: blockIdx.y
// selects the tensor (0=gate -> y0, 1=up -> y1). Per (tensor, row) the body is nvfp4_mmvq_multirow
// verbatim -> BIT-IDENTICAL per output element to two separate launches. (The reference engine
// runs the same fusion as its top 27B decode kernel at 47-50% DRAM vs ~40% for singles.)
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_mr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes, float y0scale, float y1scale) {
    const unsigned char* W = (blockIdx.y == 0) ? W0 : W1;
    float* y = (blockIdx.y == 0) ? y0 : y1;
    float yscale = (blockIdx.y == 0) ? y0scale : y1scale;
    // nvfp4_mmvq_multirow reads blockIdx.y as the token index; decode m==1 -> token 0. Inline the
    // call with t forced to 0 via a shifted grid: we reuse the body by passing m=1 and mapping
    // blockIdx.y ourselves — the helper uses blockIdx.y for t, so temporarily this kernel only
    // supports m==1 (asserted host-side).
    nvfp4_mmvq_dual_row<2>(W, aq, ad, y, in_f, out_f, row_bytes, yscale);
}

// ---- NVFP4 BATCHED matvec, WEIGHT-TILE-RESIDENT across M token columns (the m=2-4 concurrent-decode
// win). The current mmvq launches grid.y=m INDEPENDENT blocks per output row -> the weight row is
// re-read m times from HBM/L2. Here ONE warp owns ONE output row and walks the weight ONCE, doing
// dp4a against ALL m activation columns (m independent accumulators in regs). The weight quant
// bytes + decoded e2m1 values leave HBM/L2 ONCE and serve all m tokens (the activation is tiny: m*32
// int8 per group). So m tokens cost ~1 weight-read instead of m. y is [m, out_f] (token-major, same
// as the per-m kernel writes y[t*out_f+o]). MCOLS is the compile-time batch (2 or 4). For m<MCOLS the
// extra columns are computed against zero-padded activation (caller sizes y for exactly m; we guard).
// BIT-IDENTICAL per (token,row) to qmatvec_nvfp4_mmvq: same dp4a order, same ue4m3 scale, same reduce.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2;
        // decode the weight nibbles ONCE for this group (reused across all m token columns).
        int2 wv[2][2];   // [sl][0]=va, [sl][1]=vb
        float wscale[2];
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;
            int q4a = get_int_b4(qss);
            int q4b = get_int_b4(qss + 4);
            wv[sl][0] = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            wv[sl][1] = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            wscale[sl] = ue4m3_to_f32_d(d_bytes[s]);
        }
        // for each token column: load its 32 int8 activation + per-group scale, dp4a vs the decoded W.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int base = sl * 4;
                int2 va = wv[sl][0], vb = wv[sl][1];
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += wscale[sl] * (float)sumi;
            }
            acc[c] += adg * partial;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 (K=4..7 spec verify, T=5..8). Same template; columns c >= m break out, so m=5 does the
// b4+b1 split's total dp4a work with ONE weight read/decode instead of five (the pre-b8 T=5 path
// was grid.y=m per-row MMVQ — 5 full weight reads — measured as the 27B K=4 cliff).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=16 (lane/rp-on-st, 2026-08-06) — the EXACT-16 SERVE TIER's admission ticket for NVFP4,
// and the actual blocker this lane found. The mixed FP8-ST 27B artifact is 193 NVFP4 dense-MLP
// tensors + 208 per-tensor F8_E4M3; `decode_batch_exact16_ok` walks EVERY matmul, so one class
// without a bit-exact m=9..16 kernel refuses the whole model — measured as
// `decode_step_batch: B=16 > cap 8 with no exact tier ... refused` on the ST checkpoint even
// after the e4m3 classes got theirs. NVFP4 already had the template (b2/b4/b8); only the MCOLS=16
// instantiation was missing, so this is one line of new code and zero new arithmetic: same
// per-group nibble decode, same dp4a order, same ue4m3 weight scale, same warp_reduce_sum, hence
// bit-identical per (token,row) to qmatvec_nvfp4_mmvq at m=1. NOTE the variant families (_pf,
// _r2, _pfr2, _ca) deliberately get NO b16 twin here: they are shape-tuned perf variants whose
// selection is a measured per-shape call, and the exact tier needs the reference form first.
extern "C" __global__ void qmatvec_nvfp4_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, WEIGHT-PREFETCH double-buffer (b4 long_scoreboard fix, 2026-07-03).
// ncu --set full on the REAL 27B verify (12 steady launches): the batched kernel is memory-LATENCY
// bound, not bandwidth bound — long_scoreboard 18-30 stalls/issue (every other reason <=1.7),
// DRAM only 41-51% active, lg_throttle 0.7 (LSU queue fine), L1 hit 94% (activations), L2 hit 18%
// (weights stream from DRAM). Cause: ONE weight-load wavefront (6 LDGs, 18B) in flight per warp per
// k-iteration — half the m=1 mr2 kernel's per-warp weight MLP. Fix: stage the NEXT g-iteration's
// weight words in registers, issuing its 5 LDGs BEFORE consuming the current ones -> 2 weight
// wavefronts in flight per warp. Also folds the 2 scale byte-loads into the superblock's one
// 4-byte scale word (b is 4-aligned; extracted bytes feed the SAME ue4m3_to_f32_d) and the 4 quant
// words are the SAME 16 bytes the reference reads via get_int_b4 x4. BIT-IDENTICAL per (token,row):
// identical dp4a order, scales, adg factor, warp_reduce_sum — only load ISSUE TIME changes.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    // staged weight words for the CURRENT g: 4 quant words (16B at qs + whichHalf*16) + the
    // superblock's 4-byte scale word.
    int q0 = 0, q1 = 0, q2 = 0, q3 = 0, scw = 0;
    int g = lane;
    if (g < nsb) {
        const unsigned char* b = wrow + (long)(g >> 1) * 36;
        const unsigned char* qp = b + 4 + (g & 1) * 16;
        q0 = get_int_b4(qp);      q1 = get_int_b4(qp + 4);
        q2 = get_int_b4(qp + 8);  q3 = get_int_b4(qp + 12);
        scw = get_int_b4(b);
    }
    while (g < nsb) {
        int cq0 = q0, cq1 = q1, cq2 = q2, cq3 = q3, cscw = scw;
        int gn = g + 32;
        if (gn < nsb) {   // issue the NEXT wavefront before consuming the current one
            const unsigned char* bn = wrow + (long)(gn >> 1) * 36;
            const unsigned char* qpn = bn + 4 + (gn & 1) * 16;
            q0 = get_int_b4(qpn);      q1 = get_int_b4(qpn + 4);
            q2 = get_int_b4(qpn + 8);  q3 = get_int_b4(qpn + 12);
            scw = get_int_b4(bn);
        }
        int s0 = (g & 1) * 2;
        // decode ONCE per group, exactly like the reference (sl=0 -> cq0/cq1, sl=1 -> cq2/cq3).
        int2 wv[2][2];
        float wscale[2];
        wv[0][0] = get_int_from_table_16_d(cq0, kvalues_mxfp4_d);
        wv[0][1] = get_int_from_table_16_d(cq1, kvalues_mxfp4_d);
        wv[1][0] = get_int_from_table_16_d(cq2, kvalues_mxfp4_d);
        wv[1][1] = get_int_from_table_16_d(cq3, kvalues_mxfp4_d);
        wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
        wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int base = sl * 4;
                int2 va = wv[sl][0], vb = wv[sl][1];
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += wscale[sl] * (float)sumi;
            }
            acc[c] += adg * partial;
        }
        g = gn;
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_pf(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pf<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, TWO ROWS PER WARP (same long_scoreboard fix by the mr2 route: 2
// independent weight-row streams per warp = 12 weight LDGs in flight instead of 6, and the m
// activation columns are loaded once per warp and reused across BOTH rows). Per (token,row) the
// body is the reference nvfp4_mmvq_batched verbatim -> bit-identical; only the row->warp mapping
// (grid shape) and cross-row interleave change, both exactness-free. Costs ~+14 regs -> one fewer
// resident block; measured against _pf on the DRAM-cold sweep before defaulting.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        // decode BOTH rows' weight groups first (both wavefronts issued together).
        int2 wv[2][2][2];    // [row][sl][a/b]
        float wscale[2][2];  // [row][sl]
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 36;
            const unsigned char* qs = b + 4;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int s = s0 + sl;
                const unsigned char* qss = qs + s * 8;
                wv[r][sl][0] = get_int_from_table_16_d(get_int_b4(qss),     kvalues_mxfp4_d);
                wv[r][sl][1] = get_int_from_table_16_d(get_int_b4(qss + 4), kvalues_mxfp4_d);
                wscale[r][sl] = ue4m3_to_f32_d(b[s]);
            }
        }
        // each token column's activation loaded ONCE, dp4a vs both rows.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// 8-RESIDENT-BLOCK twin of b4_r2: __launch_bounds__(128, 8) squeezes 67 -> 64 regs (STACK:8, no
// LOCAL spill) so 8 blocks fit per SM instead of 7. Same template, bit-identical per (token,row).
// Only wins when the extra residency DROPS the integer wave count of the halved grid — measured
// DRAM-cold m=4: ffn_down 640 blocks 1.11 -> 0.98 waves = 112.5 -> 81.6us (beats pf 90.1);
// ssm_out 44.9 -> 34.1; qkv 1280 blocks 2.23 -> 1.95 waves = 58.1 -> 51.1. When ceil(waves) does
// NOT drop, the reg squeeze is a pure ~3-4% per-block tax (gate/up 81.1 -> 83.9, attn_q 12288
// 61.0 -> 63.4) — the dispatcher compares ceil(waves) at both residencies and picks.
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 twins (K=4..7 spec verify T=5..8). acc[2][8] costs ~+8 regs over b4_r2 — the r2w8
// residency squeeze may spill at MCOLS=8; measured per shape before defaulting (msweep m=5..8).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b8_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, PREFETCH x TWO-ROWS combined (4 weight wavefronts in flight/warp:
// 2 rows x double-buffer). Highest register pressure of the family; measured, not assumed.
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    int q[2][4]; int scw[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) { q[r][0]=q[r][1]=q[r][2]=q[r][3]=0; scw[r]=0; }
    int g = lane;
    if (g < nsb) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)(g >> 1) * 36;
            const unsigned char* qp = b + 4 + (g & 1) * 16;
            q[r][0] = get_int_b4(qp);      q[r][1] = get_int_b4(qp + 4);
            q[r][2] = get_int_b4(qp + 8);  q[r][3] = get_int_b4(qp + 12);
            scw[r] = get_int_b4(b);
        }
    }
    while (g < nsb) {
        int cq[2][4]; int cscw[2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            cq[r][0]=q[r][0]; cq[r][1]=q[r][1]; cq[r][2]=q[r][2]; cq[r][3]=q[r][3];
            cscw[r]=scw[r];
        }
        int gn = g + 32;
        if (gn < nsb) {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                const unsigned char* bn = wrow0 + (long)r * row_bytes + (long)(gn >> 1) * 36;
                const unsigned char* qpn = bn + 4 + (gn & 1) * 16;
                q[r][0] = get_int_b4(qpn);      q[r][1] = get_int_b4(qpn + 4);
                q[r][2] = get_int_b4(qpn + 8);  q[r][3] = get_int_b4(qpn + 12);
                scw[r] = get_int_b4(bn);
            }
        }
        int s0 = (g & 1) * 2;
        int2 wv[2][2][2];
        float wscale[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            wv[r][0][0] = get_int_from_table_16_d(cq[r][0], kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(cq[r][1], kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(cq[r][2], kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(cq[r][3], kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw[r] >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw[r] >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        g = gn;
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pfr2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_pfr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_pfr2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, cp.async SMEM WEIGHT RING (A5, 2026-07-04 — Marlin/CUTLASS multi-stage
// staging pattern). ncu on the post-pf/r2w8 dispatch showed the residual stall is STILL memory
// latency (long_scoreboard 8.8-16.4/issue vs FMA-dep wait <=1.9, DRAM only 48-69%): the register
// double-buffer (pf) and 2-row ILP (r2) top out at 1-2 weight wavefronts in flight per warp because
// every extra wavefront costs ~20 registers. cp.async.cg stages weight bytes global->smem WITHOUT
// register cost, so a STAGES-deep ring holds (STAGES-1) full 576B warp-windows in flight per warp.
// Layout law: one warp k-iteration consumes a CONTIGUOUS 576-byte window (16 NVFP4 36B blocks —
// 32 lanes x half-block) at window g-iter*576 of the row; when row_bytes%16==0 (all trunk shapes:
// in_f%256==0 -> (in_f/64)*36 % 16 == 0) every window is 16B-aligned in GLOBAL space, so the ring
// copies 36 16B cp.async.cg chunks per window. Lanes then read their 5 words (4 quant + 1 scale)
// from smem (LDS, no long_scoreboard). Host dispatch gates _ca on row_bytes%16==0 && nsb%32==0;
// otherwise falls back to pf/r2. BIT-IDENTICAL per (token,row): the staged bytes ARE the global
// bytes (cp.async is a byte copy); identical dp4a order, scales, adg factor, warp_reduce_sum —
// only WHERE the bytes stage changes, not the dot order.
#define CA_WIN 576   // bytes per warp-window: 16 blocks x 36B
__device__ __forceinline__ void cp_async16_g(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template<int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

// Issue one row-window (36 x 16B chunks) into `dst`. Lane L copies chunk L, lanes 0..3 also copy
// chunk 32+L. `src` = wrow + iter*CA_WIN, 16B-aligned by the dispatch gate.
__device__ __forceinline__ void ca_issue_window(unsigned char* dst, const unsigned char* src, int lane) {
    cp_async16_g(dst + lane * 16, src + lane * 16);
    if (lane < 4) cp_async16_g(dst + (32 + lane) * 16, src + (32 + lane) * 16);
}

// WROWS=1: one row/warp, STAGES-deep ring (smem 4 warps x STAGES x 576B).
// WROWS=2: two rows/warp (r2's activation-reuse + halved grid) x STAGES ring on both row streams.
template<int MCOLS, int WROWS, int STAGES>
__device__ __forceinline__ void nvfp4_mmvq_batched_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int niter = nsb >> 5;                       // dispatch gate: nsb%32==0
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    __shared__ __align__(16) unsigned char smw[MEMRA_MMVQ_ROWS][STAGES][WROWS][CA_WIN];
    unsigned char (*ring)[WROWS][CA_WIN] = smw[threadIdx.y];
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    // prologue: ALWAYS commit STAGES-1 groups (empty commits keep the per-thread group count
    // uniform when niter < STAGES, so wait<STAGES-2> below really completes the oldest stage).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < niter) {
            ca_issue_window(&ring[s][0][0], wrow0 + s * CA_WIN, lane);
            if (WROWS == 2 && has1)
                ca_issue_window(&ring[s][1][0], wrow0 + row_bytes + s * CA_WIN, lane);
        }
        cp_async_commit();
    }
    for (int it = 0; it < niter; it++) {
        cp_async_wait<STAGES - 2>();            // oldest committed stage (it) landed
        __syncwarp();
        const unsigned char* wnd0 = &ring[it % STAGES][0][0];
        int g = it * 32 + lane;
        int loff = (lane >> 1) * 36;            // this lane's block within the window
        int qoff = loff + 4 + (lane & 1) * 16;  // its 16B quant half (4B-aligned in smem)
        int s0 = (lane & 1) * 2;
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const unsigned char* wnd = wnd0 + (WROWS == 2 ? r * CA_WIN : 0);
            int cscw = *(const int*)(wnd + loff);
            int cq0 = *(const int*)(wnd + qoff);
            int cq1 = *(const int*)(wnd + qoff + 4);
            int cq2 = *(const int*)(wnd + qoff + 8);
            int cq3 = *(const int*)(wnd + qoff + 12);
            // decode ONCE per group, exactly like pf (sl=0 -> cq0/cq1, sl=1 -> cq2/cq3).
            int2 wv[2][2];
            float wscale[2];
            wv[0][0] = get_int_from_table_16_d(cq0, kvalues_mxfp4_d);
            wv[0][1] = get_int_from_table_16_d(cq1, kvalues_mxfp4_d);
            wv[1][0] = get_int_from_table_16_d(cq2, kvalues_mxfp4_d);
            wv[1][1] = get_int_from_table_16_d(cq3, kvalues_mxfp4_d);
            wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[sl][0], vb = wv[sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        // refill: consume slot (it%STAGES) is done for THIS warp's lanes after the reads above
        // retire; the overwrite targets slot (it+STAGES-1)%STAGES = (it-1)%STAGES, whose reads
        // finished an iteration ago (separated by the next iter's __syncwarp).
        int itn = it + STAGES - 1;
        if (itn < niter) {
            ca_issue_window(&ring[itn % STAGES][0][0], wrow0 + (size_t)itn * CA_WIN, lane);
            if (WROWS == 2 && has1)
                ca_issue_window(&ring[itn % STAGES][1][0], wrow0 + row_bytes + (size_t)itn * CA_WIN, lane);
        }
        cp_async_commit();
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<2, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<4, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_car2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<2, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_car2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_ca<4, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 split-plane matvec, cp.async SOFTWARE-PIPELINED (2026-07-05). The _rp kernels below
// issue a SYNCHRONOUS int4 quant load per g-iter; ncu on 27B decode (b4_rpr2) showed the dominant
// stall is long_scoreboard 7.8-9.0 inst/issue (global-load latency), occupancy only 55% (67 regs
// -> 7 blocks/SM), DRAM 40% — memory-LATENCY bound, not wall-bound. This variant pipelines the
// split-plane windows with cp.async so weight loads for iter it+STAGES-1 are in flight while iter
// it computes. Split-plane makes the window trivially aligned: per warp-iter the quant read is
// 512B contiguous (32 lanes x 16B at rowq + it*512) and the scale read is 64B (16 words at
// rows + it*64). Window = 512B quant + 64B scale = 576B (== CA_WIN). BIT-IDENTICAL to _rp: the
// staged bytes ARE the global bytes, same word order (qw.x..qw.w = cq0..cq3), same scale byte
// extraction, same dp4a order + adg + warp_reduce_sum — only WHERE the bytes stage changes.
#define RP_WIN 576   // 512B quant (32x16B) + 64B scale (4x16B)
__device__ __forceinline__ void ca_issue_window_rp(unsigned char* dst,
        const unsigned char* qsrc, const unsigned char* ssrc, int lane) {
    cp_async16_g(dst + lane * 16, qsrc + lane * 16);          // quant: 32 lanes x 16B = 512B
    if (lane < 4) cp_async16_g(dst + 512 + lane * 16, ssrc + lane * 16);  // scale: 4 lanes x 16B = 64B
}
template<int MCOLS, int WROWS, int STAGES>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    int niter = nsb >> 5;                       // dispatch gate: nsb%32==0
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;   // this warp's row0 quant base
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;    // this warp's row0 scale base
    long qstride = (long)nsb64 * 32;            // +1 row in the quant plane
    long sstride = (long)nsb64 * 4;             // +1 row in the scale plane
    __shared__ __align__(16) unsigned char smw[MEMRA_MMVQ_ROWS][STAGES][WROWS][RP_WIN];
    unsigned char (*ring)[WROWS][RP_WIN] = smw[threadIdx.y];
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < niter) {
            ca_issue_window_rp(&ring[s][0][0], rowq0 + (size_t)s * 512, rows0 + (size_t)s * 64, lane);
            if (WROWS == 2 && has1)
                ca_issue_window_rp(&ring[s][1][0], rowq0 + qstride + (size_t)s * 512,
                                   rows0 + sstride + (size_t)s * 64, lane);
        }
        cp_async_commit();
    }
    for (int it = 0; it < niter; it++) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        int g = it * 32 + lane;
        int s0 = (lane & 1) * 2;
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const unsigned char* wnd = &ring[it % STAGES][r][0];
            int4 qw = *(const int4*)(wnd + lane * 16);
            int cscw = *(const int*)(wnd + 512 + (lane >> 1) * 4);
            int2 wv[2][2];
            float wscale[2];
            wv[0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[sl][0], vb = wv[sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
        int itn = it + STAGES - 1;
        if (itn < niter) {
            ca_issue_window_rp(&ring[itn % STAGES][0][0], rowq0 + (size_t)itn * 512,
                               rows0 + (size_t)itn * 64, lane);
            if (WROWS == 2 && has1)
                ca_issue_window_rp(&ring[itn % STAGES][1][0], rowq0 + qstride + (size_t)itn * 512,
                                   rows0 + sstride + (size_t)itn * 64, lane);
        }
        cp_async_commit();
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<4, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpcar2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<4, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<2, 1, 4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpcar2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ca<2, 2, 3>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- NVFP4 batched matvec, SPLIT-PLANE WALK-ORDER REPACK (A6, 2026-07-04 — Marlin-style offline
// repack). The GGUF 36B block interleaves a 4B scale word with 32B of quants, so a lane's per-g
// weight read is 5 scattered 4B LDGs at 36B stride (the "18B straggle": 4x LDG.32 quants at a
// 4B-aligned address + 1 scale LDG). The repacked layout splits the tensor into two planes:
//   quant plane: out_f rows x (in_f/64) x 32B  — lane's 16B half at row_q + g*16, PERFECTLY
//                16B-aligned; the warp reads 512B contiguous per g-iter = one LDG.128 wavefront;
//   scale plane: out_f rows x (in_f/64) x 4B   — block's scale word at row_s + (g>>1)*4 (the
//                warp reads 64B contiguous; lane pairs broadcast-share a word).
// Same total bytes (36/block), byte-for-byte the same values — only their ADDRESSES move, so the
// decode (same word order cq0..cq3 + same scale-byte extraction as _pf) is BIT-IDENTICAL per
// (token,row). W points at the repacked tensor base; the scale plane starts at
// out_f*(in_f/64)*32 (32B-multiple -> aligned). row_bytes is unused (kept for ABI parity).
template<int MCOLS, int WROWS>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;         // 32-elem groups
    int nsb64 = in_f >> 6;       // 64-elem NVFP4 blocks
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            // ONE 16B load for the quant half (vs 4x LDG.32 at a 36B-stride address), one 4B
            // scale-word load from the dense plane. Word order cq0..cq3 identical to _pf.
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (WROWS == 2 && r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<2, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<2, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// 8-resident-block twin of b4_rpr2 (r2w8 precedent: wins when the extra residency deletes a
// straggler wave of the halved grid).
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b4_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=8 rp twins (K=4..7 spec verify T=5..8 on the default split-plane layout).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=16 SPLIT-PLANE twin (lane/rp-on-st, 2026-08-06). REQUIRED, not optional: rp is a LAYOUT,
// and NVFP4-from-safetensors is resident as split-plane BY DEFAULT (model.rs A1 direct import,
// `rp: true`). The b16 dispatch pins variant = "rp" whenever rp is set, so a base-only b16 would
// have made `func("qmatvec_nvfp4_mmvq_b16_rp")` miss — and feeding split-plane bytes to the
// GGUF-layout b16 instead would silently produce NaN. WROWS=1 matches the b16 launcher's
// ROWS_PER_BLOCK (the r2 schedules do not exist at this width). Body is the same
// nvfp4_mmvq_batched_rp template the b2/b4/b8 rp kernels run -> bit-identical per (token,row).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<16, 1>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- GROUP-4 GDN-tuple BATCHED twins (lane/dspark-trunk-kernels-20260820 slice C). The GDN
// in-projection 4-tuple (wqkv / wqkv_gate / ssm_beta / ssm_alpha — 10240/6144/48/48 rows on the
// q38 trunk) shares ONE activation and one in_f; as four sequential batched launches the two
// 48-row tensors each pay a full 12-block launch + latency round (measured 8.5us apiece,
// nsys-B verify scope; tuple total 2.16 ms/rd over 160 launches/rd) and the four weight
// streams serialize. ONE grid computes all four: blocks map to the CONCATENATED row space
// [n0|n1|n2|n3]; the host gate requires every out_f to be a multiple of rows_per_block(8), so
// each warp's row pair resolves to exactly ONE tensor and runs nvfp4_mmvq_batched_rp's body
// VERBATIM on (W_t, o_local): same 16B quant load, same scale-word decode, same dp4a order,
// same adg factor, same warp_reduce_sum per (token, row). The tensor's NVFP4 macro-scale is
// fused at the write — the same single IEEE multiply the launcher's scale_inplace pass stores
// (the mr2 fused-yscale precedent) -> BIT-IDENTICAL per (tensor, token, row) to the four
// single launches + their scale passes. WROWS=2 for every tensor (bit-identity holds across
// WROWS by the variant law; the 48-row tensors ride the big grid's residency instead of their
// own 12-block launches). Split-plane rp layout ONLY (the sm_120 default trunk).
template<int MCOLS>
__device__ __forceinline__ void nvfp4_mmvq_batched_group4_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        float* __restrict__ y2, float* __restrict__ y3,
        int in_f, int n0, int n1, int n2, int n3, int m,
        float s0, float s1, float s2, float s3) {
    const int WROWS = 2;
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    const unsigned char* W;
    float* y;
    float ys;
    int out_f;
    if (o0 < n0) { W = W0; y = y0; ys = s0; out_f = n0; }
    else if ((o0 -= n0) < n1) { W = W1; y = y1; ys = s1; out_f = n1; }
    else if ((o0 -= n1) < n2) { W = W2; y = y2; ys = s2; out_f = n2; }
    else if ((o0 -= n2) < n3) { W = W3; y = y3; ys = s3; out_f = n3; }
    else return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0w = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0w     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0w + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            // ys==1.0 skips the multiply exactly like the launcher's conditional
            // scale_inplace pass (bit-hygiene: no unconditional *1.0f on the write).
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = (ys == 1.0f) ? a : a * ys;
        }
    }
}
#define MEMRA_GROUP4_WRAP(MC) \
extern "C" __global__ void qmatvec_nvfp4_mmvq_group4_b##MC##_rp( \
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1, \
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3, \
        const signed char* __restrict__ aq, const float* __restrict__ ad, \
        float* __restrict__ y0, float* __restrict__ y1, \
        float* __restrict__ y2, float* __restrict__ y3, \
        int in_f, int n0, int n1, int n2, int n3, int m, \
        float s0, float s1, float s2, float s3) { \
    nvfp4_mmvq_batched_group4_rp<MC>(W0, W1, W2, W3, aq, ad, y0, y1, y2, y3, \
                                     in_f, n0, n1, n2, n3, m, s0, s1, s2, s3); \
}
MEMRA_GROUP4_WRAP(2)
MEMRA_GROUP4_WRAP(4)
MEMRA_GROUP4_WRAP(5)
MEMRA_GROUP4_WRAP(6)
MEMRA_GROUP4_WRAP(7)
MEMRA_GROUP4_WRAP(8)
// MCOLS=16 (lane/orndecode2): the exact-16 decode tier's quartet/trio width. acc[2][16]
// per warp-row pair — measured before promoted, same bit-identity law as every MCOLS.
MEMRA_GROUP4_WRAP(16)
#undef MEMRA_GROUP4_WRAP

// ---- DUAL gate+up BATCHED twins (lane/verify-economics, 2026-08-02). The verify-tier FFN pair
// (gate, up) shares ONE activation and one shape; as two sequential batched launches the two
// independent weight streams serialize (no PDL arm on the batched launcher) and each launch
// pays its own straggler tail. ONE grid computes both: blockIdx.y selects the tensor
// (0 -> W0/y0=gate, 1 -> W1/y1=up); per (tensor, token, row) the body is the SAME template the
// single launch runs -> BIT-IDENTICAL per output element to the two single launches
// (kernel-check pins bitwise on both layouts). Schedules mirror the single-launch auto picks
// for the 27B 5120x17408 pair: split-plane rp layout (the sm_120 default trunk) = b2 rp(r1) /
// b4 rpr2; GGUF layout = b2 base / b4 r2. b2/b4 ONLY (verify T=2..4, the K=1..3 profitable
// window): the b8 dual measured FLAT vs the rpsc singles (x3 interleaved probe,
// research/verify-economics-20260802) and was killed per doctrine. Measured on the live q27
// verify: -0.2..-0.3ms/pass at T=2..4 (9/9 interleaved pairs), +0.8% spec e2e at K=3.
// Mechanism precedent: the m=1 dual (dual_mr2) measured 47-50% DRAM active vs ~40% singles.
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched<2>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                          in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b4_r2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_r2<4>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                             in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<2, 1>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b4_rpr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 2>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}
// Tiny-projection twin for the 27B RP auxiliary dual. Unlike the rpr2 default used by large
// equal-shape pairs, this preserves the tiny singles' one-row-per-warp mapping and combines their
// two 12-block grids into one 24-block launch. The per-row template body is unchanged.
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b4_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<4, 1>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b8_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- rpsc: _rp + per-warp SMEM SCALE PRESTAGE (2026-07-06). ncu on the 27B verify (b4_rpr2):
// 62.4% long_scoreboard, DRAM 36-52%, reg-limited 7 blocks — latency-bound. The steady loop has
// TWO outstanding global dependencies per (row, g-iter): the 16B quant half and the 4B scale word,
// both streaming from DRAM. This twin coalesced-loads each warp's FULL scale rows (nsb64 words,
// <=272 = 1088B/row) into smem ONCE before the loop, so the loop keeps ONE global dependency (the
// quant stream). No register growth (the unroll-2/rpca occupancy trap does not apply — staging is
// a pointer swap). Same values, same dp4a + warp_reduce_sum order -> BIT-IDENTICAL to _rp per
// (token,row). Dispatch gates: in_f % 512 == 0 && in_f/64 <= RP_MAX_NSB64 (all 27B/9B shapes).
#define RP_MAX_NSB64 272   // in_f <= 17408
template<int MCOLS, int WROWS>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_sc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    __shared__ __align__(16) int ssc[MEMRA_MMVQ_ROWS][WROWS][RP_MAX_NSB64];
    // prestage this warp's scale rows (warp-private smem -> __syncwarp, no block barrier).
    int n4 = nsb64 >> 2;                    // dispatch gate: nsb64 % 4 == 0
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4);
        int4* dst = (int4*)&ssc[threadIdx.y][r][0];
        for (int i = lane; i < n4; i += 32) dst[i] = src[i];
    }
    __syncwarp();
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = ssc[threadIdx.y][r][sblk];          // smem, no global dependency
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (WROWS == 2 && r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
// seg twin of the batched body (lane/samplat, 2026-08-21): identical per-(tensor,row,column)
// chain with o0 rebased by the caller's block-range offset — the fused-dispatch form of
// nvfp4_mmvq_batched_rp_sc, exactly as nvfp4_mmvq_fused_seg_rp is the fused form of the m=1
// body. Bit-identical outputs to the standalone bN_rpsc launch per tensor by construction.
template<int MCOLS, int WROWS>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_sc_seg(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, int seg_block0) {
    int o0 = (((int)blockIdx.x - seg_block0) * MEMRA_MMVQ_ROWS + threadIdx.y) * WROWS;
    if (o0 >= out_f) return;
    const bool has1 = (WROWS == 2) && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    __shared__ __align__(16) int ssc[MEMRA_MMVQ_ROWS][WROWS][RP_MAX_NSB64];
    int n4 = nsb64 >> 2;
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4);
        int4* dst = (int4*)&ssc[threadIdx.y][r][0];
        for (int i = lane; i < n4; i += 32) dst[i] = src[i];
    }
    __syncwarp();
    float acc[WROWS][MCOLS];
    #pragma unroll
    for (int r = 0; r < WROWS; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[WROWS][2][2];
        float wscale[WROWS][2];
        #pragma unroll
        for (int r = 0; r < WROWS; r++) {
            if (WROWS == 2 && r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = ssc[threadIdx.y][r][sblk];
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < WROWS; r++) {
                if (WROWS == 2 && r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < WROWS; r++) {
        if (WROWS == 2 && r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}

// fused4 x batched twin (lane/samplat): the GDN projection quartet at the BATCHED tick
// (B=2..8) in ONE launch. The B=8 serve tick paid 310 small mmvq launches (28.4% of the
// tick, box4 nsys); the quartet is 4 of them per GDN layer x 30 layers. Weight rows are
// read ONCE per (row) for all B columns (the bN program), unlike a grid.y=m lift of the
// m=1 fused kernel which would re-read weights B times. Scale-1 tensors only (host gate).
extern "C" __global__ void qmatvec_nvfp4_mmvq_fused4_b8_rpsc(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, int in_f, int out0, int out1, int out2, int out3, int m) {
    const int rows_pb = MEMRA_MMVQ_ROWS * 2; // WROWS=2
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    const int nb1 = (out1 + rows_pb - 1) / rows_pb;
    const int nb2 = (out2 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W0, aq, ad, y0, in_f, out0, m, 0);
    } else if ((int)blockIdx.x < nb0 + nb1) {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W1, aq, ad, y1, in_f, out1, m, nb0);
    } else if ((int)blockIdx.x < nb0 + nb1 + nb2) {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W2, aq, ad, y2, in_f, out2, m, nb0 + nb1);
    } else {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W3, aq, ad, y3, in_f, out3, m, nb0 + nb1 + nb2);
    }
}

// fused3 x batched twin (lane/samplat): the attention q/k/v trio at the batched tick,
// same construction as fused4_b8 (batched seg body, weight rows read once per m columns).
extern "C" __global__ void qmatvec_nvfp4_mmvq_fused3_b8_rpsc(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m) {
    const int rows_pb = MEMRA_MMVQ_ROWS * 2; // WROWS=2
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    const int nb1 = (out1 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W0, aq, ad, y0, in_f, out0, m, 0);
    } else if ((int)blockIdx.x < nb0 + nb1) {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W1, aq, ad, y1, in_f, out1, m, nb0);
    } else {
        nvfp4_mmvq_batched_rp_sc_seg<8, 2>(W2, aq, ad, y2, in_f, out2, m, nb0 + nb1);
    }
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<2, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<4, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<8, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- EXACT-WIDTH b5/b6/b7 twins (lane/vt-fixes fix 1, 2026-08-03). The T=5..7 verify tier
// used to ride the MCOLS=8 kernels: acc[WROWS][8] is statically allocated at ANY m, so an
// m=5 launch pays the 8-wide register/occupancy tax — the measured T=4->5 verify cliff
// (verify-tier-20260802 §3: the whole +5.56ms q27 step is matvec_b). The SAME template at
// MCOLS=m runs the IDENTICAL per-(token,row) column chain (columns c >= m never execute in
// either form; same g-order, same dp4a order, same warp_reduce_sum) -> BIT-IDENTICAL to the
// b8 kernel and to per-m MMVQ. rpsc = the b8-tier auto pick (scale rows prestaged to smem);
// rpr2w8 = the !sc_ok fallback schedule. Dispatch remaps mcols 8 -> m in
// qmatvec_mmvq_batched (MEMRA_B567=0 rollback).
extern "C" __global__ void qmatvec_nvfp4_mmvq_b5_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<5, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b6_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<6, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b7_rpsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_sc<7, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b5_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<5, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b6_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<6, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_nvfp4_mmvq_b7_rpr2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<7, 2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// EXACT-WIDTH DUAL gate+up twins for T=5..7 (vt-fixes fix 1b). At T<=4 the FFN pair rides
// ONE dual launch — two independent weight streams in one grid restored the memory-level
// parallelism (dual_b4_rpr2 measured 84.6% DRAM vs 59.1% for the single); at T>=5 the old
// m<=4 dual gate dropped gate+up onto two serial singles. The verify-economics b8 dual (an
// MCOLS=8 kernel at m=5..8) measured FLAT and was killed — these are DIFFERENT cells:
// exact-width MCOLS=m keeps the register/occupancy shape of the profitable b4 dual. Per
// (tensor, token, row) the body is the single b5/b6/b7 program (blockIdx.y selects the
// tensor) -> BIT-IDENTICAL to the two singles. Split-plane rp only (the daily NVFP4 trunk).
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b5_rpr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<5, 2>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b6_rpr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<6, 2>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_b7_rpr2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp<7, 2>(blockIdx.y == 0 ? W0 : W1, aq, ad, blockIdx.y == 0 ? y0 : y1,
                                in_f, out_f, m, row_bytes);
}

// ---- rpks: K-SPLIT x2 ACROSS WARP PAIRS (2026-07-06). block (32,4) = TWO warp-pairs; a pair
// owns 2 output rows (same 2 independent weight streams per warp as rpr2), the pair's two warps
// split the k-range in half. grid.x = ceil(out_f/4) — 2x rpr2's blocks with the same regs/thread:
// latency hidden by BLOCK-level parallelism instead of per-thread ILP (the LAW fix; unroll-2 and
// rpca lost by growing registers). Reduce order: per-lane serial accumulation over the chunk's
// g's + warp_reduce_sum per (row,col) — identical WITHIN a chunk to _rp — then ONE cross-warp add
// in fixed chunk order (chunk0 + chunk1) via smem. DETERMINISTIC but NOT bit-identical to _rp
// (k-order differs); verify arbitrates exactness — gates are acceptance parity + argmax MATCH.
// SC=true additionally prestages each warp's scale-row HALF into smem (rpsc mechanism).
// Dispatch gates: in_f % 512 == 0 (nsb%16==0 -> aligned half-plane staging) && nsb64 <= 272.
template<int MCOLS, bool SC>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int pair = threadIdx.y >> 1;            // 0..1: which 2-row group of the block
    int kc = threadIdx.y & 1;               // 0..1: which k-chunk of the pair
    int o0 = (blockIdx.x * 2 + pair) * 2;
    const bool act = o0 < out_f;            // inactive warps still reach __syncthreads
    const bool has1 = act && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    int half = nsb >> 1;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    // per-warp scale half-rows: [warp][row][nsb64/2 words] (chunk kc reads sblk in
    // [kc*half/2, kc*half/2 + half/2); local index = sblk - kc*(half>>1)). Sized 1 when SC=false
    // so the plain rpks twin doesn't pay 4.3KB of dead smem per block.
    __shared__ __align__(16) int ssc[4][2][SC ? RP_MAX_NSB64 / 2 : 1];
    int sbase = kc * (half >> 1);
    if (SC && act) {
        int n4 = (half >> 1) >> 2;          // dispatch gate: (nsb/4) % 4 == 0 (in_f % 512 == 0)
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sbase * 4);
            int4* dst = (int4*)&ssc[threadIdx.y][r][0];
            for (int i = lane; i < n4; i += 32) dst[i] = src[i];
        }
        __syncwarp();
    }
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    if (act) {
        int gend = kc * half + half;
        for (int g = kc * half + lane; g < gend; g += 32) {
            int sblk = g >> 1;
            int s0 = (g & 1) * 2;
            int2 wv[2][2][2];
            float wscale[2][2];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
                int4 qw = *qh;
                int cscw = SC ? ssc[threadIdx.y][r][sblk - sbase]
                              : *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
                wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
                wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
                wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
                wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
                wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
                wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
            }
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                const signed char* arow = aq + (size_t)c * in_f;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0];
                int4 a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float adg = ad[(size_t)c * nsb + g];
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    if (r == 1 && !has1) break;
                    float partial = 0.0f;
                    #pragma unroll
                    for (int sl = 0; sl < 2; sl++) {
                        int base = sl * 4;
                        int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                        int sumi = 0;
                        sumi = dp4a(va.x, aq4[base + 0], sumi);
                        sumi = dp4a(vb.x, aq4[base + 1], sumi);
                        sumi = dp4a(va.y, aq4[base + 2], sumi);
                        sumi = dp4a(vb.y, aq4[base + 3], sumi);
                        partial += wscale[r][sl] * (float)sumi;
                    }
                    acc[r][c] += adg * partial;
                }
            }
        }
    }
    // reduce: butterfly per (row,col) inside each chunk-warp, then chunk0 + chunk1 in FIXED order.
    float asum[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) asum[r][c] = warp_reduce_sum(acc[r][c]);
    __shared__ float part[2][2][MCOLS];     // [pair][row][col], written by the kc==1 warp
    if (kc == 1 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 2; r++)
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) part[pair][r][c] = asum[r][c];
    }
    __syncthreads();
    if (act && kc == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            #pragma unroll
            for (int c = 0; c < MCOLS; c++) {
                if (c >= m) break;
                y[(size_t)c * out_f + o0 + r] = asum[r][c] + part[pair][r][c];
            }
        }
    }
}
// ---- rpms: M-SPLIT ACROSS WARP PAIRS (2026-07-06). Same occupancy goal as rpks (2x rpr2's
// blocks: grid = ceil(out_f/4), block (32,4) = 2 pairs x 2 rows) WITHOUT touching the k-reduce
// order: the pair's two warps both walk the FULL k-range of the SAME 2 rows but each owns half
// the m columns (warp kc computes cols [kc*MCOLS/2, (kc+1)*MCOLS/2), c>=m masked). Every
// (token,row) dot keeps the reference per-lane serial chain + warp_reduce_sum -> BIT-IDENTICAL
// to _rp. (The rpks e2e self-consistency FAIL taught: verify logits MUST be bit-identical to the
// decode path — the k-order shift moves greedy argmax at tie margins and run-spec FAILs.) The
// twin warp re-reads the same weight bytes in near-lockstep -> L1/L2 serve the second copy; the
// per-warp column work halves and acc/act registers drop (acc[2][MCOLS/2]). No cross-warp
// reduce, no smem, no __syncthreads — warps fully independent. SC=true prestages the pair's
// scale rows to smem (rpsc mechanism, warp-private so no block barrier).
template<int MCOLS, bool SC>
__device__ __forceinline__ void nvfp4_mmvq_batched_rp_ms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    constexpr int CH = MCOLS / 2;           // columns per warp
    int pair = threadIdx.y >> 1;            // 0..1: which 2-row group of the block
    int kc = threadIdx.y & 1;               // 0..1: which column half
    int o0 = (blockIdx.x * 2 + pair) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int c0 = kc * CH;                       // this warp's first column
    if (c0 >= m) return;                    // whole column half masked
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const unsigned char* rowq0 = qplane + (size_t)o0 * nsb64 * 32;
    const unsigned char* rows0 = splane + (size_t)o0 * nsb64 * 4;
    __shared__ __align__(16) int ssc[4][2][SC ? RP_MAX_NSB64 : 1];
    if (SC) {
        int n4 = nsb64 >> 2;                // dispatch gate: nsb64 % 4 == 0
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* src = (const int4*)(rows0 + (size_t)r * nsb64 * 4);
            int4* dst = (int4*)&ssc[threadIdx.y][r][0];
            for (int i = lane; i < n4; i += 32) dst[i] = src[i];
        }
        __syncwarp();
    }
    float acc[2][CH];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < CH; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int2 wv[2][2][2];
        float wscale[2][2];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const int4* qh = (const int4*)(rowq0 + (size_t)r * nsb64 * 32 + (size_t)g * 16);
            int4 qw = *qh;
            int cscw = SC ? ssc[threadIdx.y][r][sblk]
                          : *(const int*)(rows0 + (size_t)r * nsb64 * 4 + (size_t)sblk * 4);
            wv[r][0][0] = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
            wv[r][0][1] = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
            wv[r][1][0] = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
            wv[r][1][1] = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
            wscale[r][0] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 *  s0     )) & 0xFF));
            wscale[r][1] = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
        }
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            if (c0 + c >= m) break;
            const signed char* arow = aq + (size_t)(c0 + c) * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0];
            int4 a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = ad[(size_t)(c0 + c) * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int base = sl * 4;
                    int2 va = wv[r][sl][0], vb = wv[r][sl][1];
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += wscale[r][sl] * (float)sumi;
                }
                acc[r][c] += adg * partial;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            if (c0 + c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)(c0 + c) * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<4, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpms(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<8, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<2, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<4, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpmsc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ms<8, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<4, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpks(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<8, false>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b2_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<2, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b4_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<4, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_b8_rpksc(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    nvfp4_mmvq_batched_rp_ks<8, true>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ============ SPLIT-PLANE rp twins of the m=1 NVFP4 decode family (A6 integration) ============
// Each is the matching kernel's body with the weight-group loads swapped to the split-plane
// addresses (ONE 16B quant load + one 4B scale word) — identical decode word order (qw.x..qw.w ==
// q4a/q4b of sl=0,1), identical scale-byte extraction, identical dp4a/reduce order per (token,row).

// m>=1 warp-per-row (grid.y = t). Twin of qmatvec_nvfp4_mmvq; also serves decode-exact grid.y=m.
extern "C" __global__ void qmatvec_nvfp4_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    // PDL entry (lane/glm5-nvfp4-row-ilp-20260904): a no-op without the launch attribute; with
    // it, the B200 grid-fill (mr1) launches join the PDL class the mr2 kernel already had.
    MEMRA_PDL_ENTRY();
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* rowq = W + (size_t)o * nsb64 * 32;
    const unsigned char* rows = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int4 qw = *(const int4*)(rowq + (size_t)g * 16);
        int cscw = *(const int*)(rows + (size_t)sblk * 4);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = (sl == 0) ? qw.x : qw.z;
            int q4b = (sl == 0) ? qw.y : qw.w;
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * yscale;
}

// multirow rp body (t from blockIdx.y unless pinned): twin of nvfp4_mmvq_multirow.
template<int RPW, bool PIN_T0>
__device__ __forceinline__ void nvfp4_mmvq_multirow_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = PIN_T0 ? 0 : blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            int4 qw = *(const int4*)(qplane + ((size_t)o * nsb64 + sblk) * 32 + (size_t)(g & 1) * 16);
            int cscw = *(const int*)(splane + ((size_t)o * nsb64 + sblk) * 4);
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int q4a = (sl == 0) ? qw.x : qw.z;
                int q4b = (sl == 0) ? qw.y : qw.w;
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    MEMRA_PDL_ENTRY();
    nvfp4_mmvq_multirow_rp<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}

// ---- ILP twins of the NVFP4 split-plane trunk matvec (lane/glm5-nvfp4-row-ilp-20260904, door
// MEMRA_NVFP4_ROW_ILP, default OFF).
// WHY. `qmatvec_nvfp4_mmvq_mr2_rp` is 74 launches per plain glm5_next t=1 token on the 2x B200
// pair (door-ON census 2026-09-04: 0.9 ms of a ~13.7 ms token; the 4096-row shapes run a
// 512-block grid of four warps = 2,048 warps for a 148-SM part) and root ncu on the rig at
// that shape reads long-scoreboard stalls 68-70% of warp-active cycles, issue-active 37-39%,
// DRAM 30% of peak, occupancy 49%. Each lane walks its groups serially with ONE 16-byte quant
// load and one scale word per row in flight. This body issues FOUR groups' loads per lane
// (RPW rows each) before any table lookup or dp4a, then the shipped single-group tail.
// EXACTNESS by construction: per row the accumulation order is the shipped one
// (`acc[r] += adg * partial` for g = lane, lane+32, ... into one accumulator, `partial` built
// by the same two-sub-block chain), the same bytes are read, the warp tree and the `* yscale`
// epilogue are verbatim; RPW=1 is the per-row program of `qmatvec_nvfp4_mmvq_rp` and RPW=2
// that of `qmatvec_nvfp4_mmvq_mr2_rp`. Gate: b200_matvec_bench family 4 (shipped vs both ILP
// twins bitwise); the box greedy tape holds it at model scale.
template<int RPW, bool PIN_T0>
__device__ __forceinline__ void nvfp4_mmvq_multirow_rp_ilp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = PIN_T0 ? 0 : blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    int g = lane;
    for (; g + 96 < nsb; g += 128) {
        // Four groups' weight words and scale words per row, issued before any math.
        int4 qw[4][RPW];
        int cscw[4][RPW];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int gk = g + 32 * k;
            int sblk = gk >> 1;
            #pragma unroll
            for (int r = 0; r < RPW; r++) {
                int o = o0 + r;
                if (o < out_f) {
                    qw[k][r] = *(const int4*)(qplane + ((size_t)o * nsb64 + sblk) * 32 + (size_t)(gk & 1) * 16);
                    cscw[k][r] = *(const int*)(splane + ((size_t)o * nsb64 + sblk) * 4);
                } else {
                    qw[k][r] = make_int4(0, 0, 0, 0);
                    cscw[k][r] = 0;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int gk = g + 32 * k;
            int s0 = (gk & 1) * 2;
            const int4* aq16 = (const int4*)(arow + (size_t)gk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float adg = adrow[gk];
            #pragma unroll
            for (int r = 0; r < RPW; r++) {
                int o = o0 + r;
                if (o >= out_f) break;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int q4a = (sl == 0) ? qw[k][r].x : qw[k][r].z;
                    int q4b = (sl == 0) ? qw[k][r].y : qw[k][r].w;
                    int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                    int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                    int base = sl * 4;
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += ue4m3_to_f32_d((unsigned char)((cscw[k][r] >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
                }
                acc[r] += adg * partial;
            }
        }
    }
    for (; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            int4 qw = *(const int4*)(qplane + ((size_t)o * nsb64 + sblk) * 32 + (size_t)(g & 1) * 16);
            int cscw = *(const int*)(splane + ((size_t)o * nsb64 + sblk) * 4);
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int q4a = (sl == 0) ? qw.x : qw.z;
                int q4b = (sl == 0) ? qw.y : qw.w;
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_rp_ilp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    MEMRA_PDL_ENTRY();
    nvfp4_mmvq_multirow_rp_ilp<1, false>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}
extern "C" __global__ void qmatvec_nvfp4_mmvq_mr2_rp_ilp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float yscale) {
    MEMRA_PDL_ENTRY();
    nvfp4_mmvq_multirow_rp_ilp<2, false>(W, aq, ad, y, in_f, out_f, m, row_bytes, yscale);
}
// DUAL gate+up rp twin (blockIdx.y selects tensor; m==1 asserted host-side like the original).
extern "C" __global__ void qmatvec_nvfp4_mmvq_dual_mr2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out_f, int m, long row_bytes, float y0scale, float y1scale) {
    const unsigned char* W = (blockIdx.y == 0) ? W0 : W1;
    float* y = (blockIdx.y == 0) ? y0 : y1;
    float yscale = (blockIdx.y == 0) ? y0scale : y1scale;
    nvfp4_mmvq_multirow_rp<2, true>(W, aq, ad, y, in_f, out_f, 1, row_bytes, yscale);
}

// dp4a rp twin (128-thread two-level reduce, grid (out_f, m)). Twin of qmatvec_nvfp4_dp4a.
extern "C" __global__ void qmatvec_nvfp4_dp4a_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* rowq = W + (size_t)o * nsb64 * 32;
    const unsigned char* rows = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        int4 qw = *(const int4*)(rowq + (size_t)g * 16);
        int cscw = *(const int*)(rows + (size_t)sblk * 4);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = (sl == 0) ? qw.x : qw.z;
            int q4b = (sl == 0) ? qw.y : qw.w;
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ============================ k-quant BATCHED weight-resident matvec ============================
// Same structure as nvfp4_mmvq_batched: ONE warp owns ONE output row and walks the weight ONCE,
// decoding each weight group's quant bytes a SINGLE time and dp4a-ing the decoded weight against
// ALL m activation columns (m independent reg accumulators). The weight bytes + decoded ints leave
// HBM/L2 ONCE and serve all m tokens — the m>1 verify/MTP win (vs grid.y=m _dp4a, which re-reads the
// weight m times). y is [m, out_f] token-major. BIT-IDENTICAL per (token,row) to the matching _mmvq
// kernel: the per-element dequant + dp4a order + warp_reduce_sum are lifted verbatim; only the loop
// nest order (group-outer, column-inner) changes, which does not alter any per-(token,row) f32 sum.
// MCOLS is the compile-time batch (2 or 4); m<=MCOLS, the c>=m columns are skipped.

// ----- Q8_0 batched. Per-group reusable: dw + 8 weight ints. Per-column: activation int8 + dp4a. -----
// Row-parameterized body (`o` = the output row this warp owns): LIFTED VERBATIM from the original
// q8_0_mmvq_batched so the per-(token,row) FP chain is unchanged. The plain _b2/_b4/_b8 kernels
// pass the standard blockIdx.x mapping; the FUSED multi-tensor twins below pass the
// block-offset-split mapping (the fused2/fused3 m=1 recipe applied to the batched tier).
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_batched_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);
        const unsigned char* wq = wb + 2;
        int wi[8];                               // decode weight ints ONCE for this block
        #pragma unroll
        for (int k = 0; k < 8; k++) wi[k] = get_int_b2(wq + k * 4);
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
// ---- Q4_0 batched twins (gemma verify t=2..8): per (token,row) chain BIT-IDENTICAL to
// qmatvec_q4_0_mmvq / _mr2 (same dp4a issue order, same d4*(sumi-8*sums)*d8 accumulate). ----
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const unsigned char* b = wrow + (long)blk * 18;
        float d4 = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs = b + 2;
        int lo[4], hi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
            lo[k] = (int)(raw & 0x0F0F0F0Fu);
            hi[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13, the L1TEX fix): same values, same dp4a order.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            const int al[4] = { a01.x, a01.y, a01.z, a01.w };
            const int ah[4] = { a23.x, a23.y, a23.z, a23.w };
            int sumi = 0, sums = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(lo[k], al[k], sumi);
                sumi = dp4a(hi[k], ah[k], sumi);
                sums = dp4a(0x01010101, al[k], sums);
                sums = dp4a(0x01010101, ah[k], sums);
            }
            acc[c] += d4 * (float)(sumi - 8 * sums) * ad[(size_t)c * nblk + blk];
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
// Q4_0 batched MULTIROW (verify trunk lever): 2 rows/warp — activation int4 loads AND the
// row-independent ones-sums computed ONCE per (col, group), reused across both rows. Per
// (token, row) float chain identical to q4_0_mmvq_batched (d4*(sumi-8*sums)*d8 in g order).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* w0 = W + (long)o0 * row_bytes;
    const unsigned char* w1 = W + (long)(o0 + 1) * row_bytes;
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            const unsigned char* b = w0 + (long)blk * 18;
            const unsigned char* qs = b + 2;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                lo0[k] = (int)(raw & 0x0F0F0F0Fu);
                hi0[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            const unsigned char* b = w1 + (long)blk * 18;
            const unsigned char* qs = b + 2;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                lo1[k] = (int)(raw & 0x0F0F0F0Fu);
                hi1[k] = (int)((raw >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(*(const unsigned short*)(w0 + (long)blk * 18));
        float d41 = two ? half_to_float(*(const unsigned short*)(w1 + (long)blk * 18)) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13): the 8 scalar int loads were 4x the L1TEX
            // transactions of the t=1 walk's two 16B loads — L1TEX measured 90% saturated
            // (the b-tier limiter). Same bytes, same order per k — bit-identical.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)c * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) { s0 = dp4a(lo0[k], a[k], s0); s0 = dp4a(hi0[k], a[4 + k], s0); }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) { s1 = dp4a(lo1[k], a[k], s1); s1 = dp4a(hi1[k], a[4 + k], s1); }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float v0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = v0;
        if (two) {
            float v1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = v1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

extern "C" __global__ void qmatvec_q4_0_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


extern "C" __global__ void qmatvec_q8_0_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=16 BASE twin (lane/rp-on-st, 2026-08-06). Until now Q8_0's b16 existed ONLY as the
// split-plane `_rp` form, which made the q8rp mirror the exact-16 tier's admission ticket for
// ANY model carrying a single Q8_0 matmul. On the FP8-ST 27B that is a bad trade measured
// precisely: the diagnostic named `L0.ssm_beta qtype=0 rp4=false` as the refusing tensor, and the
// whole residual Q8_0 class there is 96 tensors / 23.906 MiB = 0.143% of resident 2D weight —
// so admission was costing a mirror walk over the trunk to serve the tier's least significant
// bytes. The mirror's real justification is BANDWIDTH on a Q8_0-dominant GGUF (the 34 B stride's
// sector overfetch, H100 ncu 2026-07-26); it was never meant to be a correctness prerequisite.
// This base form removes that coupling: the tier is now reachable at zero VRAM on any layout.
// Same q8_0_mmvq_batched_row body the b2/b4/b8 kernels run -> bit-identical per (token,row).
extern "C" __global__ void qmatvec_q8_0_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q8_0_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- FUSED Q8_0 BATCHED matvec PAIR/TRIPLE (verify t=2-4 trunk launch-fusion, MEMRA_SPEC_FUSED_T,
// lane/close35b): the m=1 fused2/fused3 block-offset split applied to the batched weight-resident
// tier. Blocks [0,nb0) compute tensor 0, [nb0,nb0+nb1) tensor 1 (fused3: a third range). Per
// (tensor,token,row) the body is q8_0_mmvq_batched_row VERBATIM with the identical row mapping
// (Q8_0 batched_variant is always "base", ROWS=4) -> BIT-IDENTICAL to the separate _b2/_b4
// launches the verify t=2-4 path otherwise runs via matmul_decode_exact, with ONE shared q8_1
// activation quantize and ONE launch instead of two/three. y per tensor is token-major [m, out_f],
// same as the per-tensor kernels. Targets: 35B wqkv+wqkv_gate (8192/4096), wq/wk/wv (8192/512/512),
// gate_shexp+up_shexp (512/512). -----
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_fused2_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0) { W = W0; y = y0; out_f = out0; }
    else         { W = W1; y = y1; out_f = out1; b -= nb0; }
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<2>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
// b8 wrapper (lane/q27-deepdive, 2026-08-05): the SERVING tier. The m=1 fuse2 lever landed
// +0.94% on q27-Q8_0 single-stream, but the batched serve tick (decode_step_batch, c=5..8 ->
// mcols 8) ran the dense-FFN gate+up as two `matmul_pre` -> two _b8 launches. Same template,
// same q8_0_mmvq_batched_row body, MCOLS=8 -> BIT-IDENTICAL per (tensor,token,row) to the two
// qmatvec_q8_0_mmvq_b8 launches, one shared q8_1 activation instead of two re-quantizes.
extern "C" __global__ void qmatvec_q8_0_mmvq_fused2_b8(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q8_0_mmvq_fused2_b<8>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_fused3_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int nb1 = (out1 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; b -= nb0 + nb1; }
    q8_0_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q8_0_mmvq_fused3_b<2>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_fused3_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q8_0_mmvq_fused3_b<4>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}

// ==================== F8-E4M3 (checkpoint-native) warp-per-row MMVQ + batched ====================
// MEMRA_ST_E4M3 decode path (lane e4m3dec, 2026-07-08): F8-E4M3-origin safetensors projections keep
// their RAW checkpoint e4m3 bytes resident ([out_f, in_f] row-major, row_bytes == in_f) instead of
// the lossy Q8_0 re-encode — the weight side of this dot is EXACT w.r.t. the checkpoint. The
// activation is the SAME q8_1 (aq int8 [m,in] + per-32 f32 ad) every fast decode path rides, so the
// fused norm->quantize producer chain is untouched. Per 32-block:
//     bs   = sum_j f32(e4m3(w[j])) * f32(aq[j])      (fmaf chain, fixed j order 0..31)
//     acc += ad[blk] * bs                            (fmaf, lane-strided blk walk like q8_0_mmvq)
// f32 accumulate throughout (e4m3 max 448 * 127 * 32 fits comfortably). The per-tensor f32
// weight_scale is FUSED at the write (`ws` arg, the NVFP4 macro-scale convention).
//
// EXACTNESS LAW: per (token,row) the body is a pure function of (row bytes, that token's q8_1
// row) — grid.y=m verify launches are bit-identical to the m=1 decode launch by construction,
// and the batched _b2/_b4/_b8 twins below replay the IDENTICAL fmaf chain per column.

// One row x one token: the shared body (bit-contract anchor for the m=1, grid.y=m and batched forms).
__device__ __forceinline__ float e4m3_row_dot(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, int nblk, int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        // 32 e4m3 weight bytes: 2x LDG.128 (wrow is 32B-aligned: base alloc 256B, row stride
        // in_f % 32 == 0). 32 int8 activation: 2x LDG.128 (same as the q8_0 twin).
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
            bs = fmaf(whi.y, (float)(a >> 24), bs);   // arithmetic shift: already sign-extended
        }
        acc = fmaf(adrow[blk], bs, acc);
    }
    return acc;
}

extern "C" __global__ void qmatvec_e4m3_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, float ws) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                             ad + (size_t)t * nblk, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc * ws;
}

// ----- F8-E4M3 m=1 single-row body shared by the FUSED multi-tensor launches below. This is
// qmatvec_e4m3_mmvq with t pinned to 0 (decode m==1): same e4m3_row_dot call, same
// warp_reduce_sum, same `* ws` write -> per (tensor,row) output bits identical to a separate
// m=1 launch. Unlike the Q8_0 twin each range carries its OWN per-tensor weight_scale, because
// the checkpoint scale is a per-tensor property (Q8_0's is always 1.0). -----
__device__ __forceinline__ void e4m3_mmvq_row1(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, long row_bytes, float ws, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_row_dot(W + (long)o * row_bytes, aq, ad, nblk, lane);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc * ws;
}

// ----- FUSED F8-E4M3 m=1 matvec PAIR, UNEQUAL out_f (lane/fp8-decode-v1, 2026-08-05). Under
// MEMRA_ST_E4M3 the per-tensor FP8 projections lost EVERY launch fusion the Q8_0/NVFP4 trunk has
// (q8_fused_params requires QT_Q8_0 && scale==1.0; matmul_pre_dual_noscale requires NVFP4) — so
// enabling native e4m3 residency UN-FUSED the trunk: the NV-27B linear-attn wqkv+wqkv_gate pair,
// the beta+alpha dual, the full-attn wq/wk/wv triple and the FFN gate+up dual all fell back to
// separate m=1 launches. Same block-offset recipe as qmatvec_q8_0_mmvq_fused2: blocks [0,nb0)
// compute tensor 0, [nb0,nb0+nb1) tensor 1. Both tensors share in_f (e4m3 row_bytes == in_f -> ONE
// row_bytes) and the SAME q8_1 activation. Per (tensor,row) the body is e4m3_mmvq_row1 ->
// BIT-IDENTICAL to two separate m=1 launches. Seam MEMRA_E4M3_DUAL=0 (host-side). -----
extern "C" __global__ void qmatvec_e4m3_mmvq_fused2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long row_bytes, float ws0, float ws1) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f; float ws;
    if (b < nb0) { W = W0; y = y0; out_f = out0; ws = ws0; }
    else         { W = W1; y = y1; out_f = out1; ws = ws1; b -= nb0; }
    e4m3_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, ws,
                   b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- FUSED F8-E4M3 m=1 matvec TRIPLE (wq+wk+wv: same input h, same in_f). Same block-offset
// recipe as fused2 with three ranges. -----
extern "C" __global__ void qmatvec_e4m3_mmvq_fused3(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long row_bytes,
        float ws0, float ws1, float ws2) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int nb1 = (out1 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f; float ws;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; ws = ws0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; ws = ws1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; ws = ws2; b -= nb0 + nb1; }
    e4m3_mmvq_row1(W, aq, ad, y, in_f, out_f, row_bytes, ws,
                   b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}

// ----- F8-E4M3 batched (b2/b4/b8): ONE warp owns ONE row, weight bytes leave HBM/L2 ONCE for all
// m token columns (the m=2..8 verify/MTP tier — without this the F8 class re-reads its ~GBs of
// weights m times per verify, the known K>=4 spec cliff). Per (token,row) the fmaf chain is the
// e4m3_row_dot body VERBATIM (weights re-converted per column from the SAME registers — cvt is
// deterministic, so the f32 inputs and order are identical) -> bit-identical to grid.y=m _mmvq.
// NOTE: 8-arg signature (no ws) like every other batched kernel — the host launcher applies the
// macro-scale via scale_inplace. -----
// Row-parameterized body (`o` = the output row this warp owns): the plain _b2/_b4/_b8 kernels pass
// the standard blockIdx.x mapping, the FUSED multi-tensor twins below pass the block-offset-split
// mapping (the fused2/fused3 m=1 recipe applied to the batched tier), exactly as the Q8_0 family
// does via q8_0_mmvq_batched_row.
template<int MCOLS>
__device__ __forceinline__ void e4m3_mmvq_batched_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];                 // weight bytes read ONCE for all columns
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        // DEQUANT-HOIST (lane/rp-on-st, 2026-08-06): e4m3 -> f32 depends only on the WEIGHT,
        // never on the activation column, so it belongs outside the column loop. It used to sit
        // inside it, which made the batched kernel do the conversion MCOLS times per k32 block:
        // at MCOLS=16 that is 16x the e4m3x2_to_f32x2 work for one set of weight bytes, and it
        // turned "weight-read-once" into "weight-read-once, weight-CONVERT-sixteen-times". The
        // whole point of the batched tier is amortizing per-weight work across columns, and the
        // conversion is per-weight work. dp4a classes (Q8_0 and friends) never had this problem
        // because their dequant IS the integer dot product.
        // EXACTNESS: bit-identity is preserved by construction — same values, same `bs`
        // accumulation order over k, same fmaf/warp_reduce_sum sequence. Only the point of
        // evaluation moves, and float conversion is not order-dependent. kernel-check's
        // E4M3-BATCHED cells (m=2..8,9,12,16, both shapes) hold bit-bad=0.
        float wf[32];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            wf[k * 4 + 0] = wlo.x; wf[k * 4 + 1] = wlo.y;
            wf[k * 4 + 2] = whi.x; wf[k * 4 + 3] = whi.y;
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float bs = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int a = au[k];
                bs = fmaf(wf[k * 4 + 0], (float)(signed char)(a & 0xff), bs);
                bs = fmaf(wf[k * 4 + 1], (float)(signed char)((a >> 8) & 0xff), bs);
                bs = fmaf(wf[k * 4 + 2], (float)(signed char)((a >> 16) & 0xff), bs);
                bs = fmaf(wf[k * 4 + 3], (float)(a >> 24), bs);
            }
            acc[c] = fmaf(ad[(size_t)c * nblk + blk], bs, acc[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
template<int MCOLS>
__device__ __forceinline__ void e4m3_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// b16 tier (lane/rp-on-st, 2026-08-06): the SAME e4m3_mmvq_batched template at MCOLS=16 — the
// exact-16 serve tier's admission ticket for the per-tensor FP8-ST class. This is the e4m3
// analogue of qmatvec_q8_0_mmvq_b16_rp: on Q8_0 the b16 kernel exists only in the split-plane
// (rp) layout, which is WHY Q8_0 needs the q8rp mirror to reach chunk 16. e4m3 needs no mirror
// at all — its native row-major layout is ALREADY aligned (row_bytes == in_f, 1 B/weight, so
// every 32-weight block is a 32 B-aligned pair of LDG.128s; there is no 34 B GGUF stride to
// un-skew and hence no coalescing deficit for a mirror to fix). Per (token,row) the body is
// e4m3_mmvq_batched_row VERBATIM, so this is BIT-IDENTICAL to the grid.y=m mmvq launch it
// replaces — identical fmaf chain, identical warp_reduce_sum, one weight read for up to 16
// columns instead of 16 full weight re-reads.
extern "C" __global__ void qmatvec_e4m3_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    e4m3_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- FUSED F8-E4M3 BATCHED pair/triple (verify + serve-tick tiers, lane/fp8-decode-v1): the m=1
// fused2/fused3 block-offset split applied to the batched weight-resident tier, mirroring
// q8_0_mmvq_fused2_b / _fused3_b exactly. Per (tensor,token,row) the body is
// e4m3_mmvq_batched_row VERBATIM with the identical row mapping -> BIT-IDENTICAL to the separate
// _b2/_b4/_b8 launches matmul_decode_exact would otherwise run, with ONE shared q8_1 activation.
// 8-arg-style signature (NO ws) like every other batched kernel: the host applies each tensor's
// per-tensor weight_scale to its own output buffer via scale_inplace. -----
template<int MCOLS>
__device__ __forceinline__ void e4m3_mmvq_fused2_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0) { W = W0; y = y0; out_f = out0; }
    else         { W = W1; y = y1; out_f = out1; b -= nb0; }
    e4m3_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_fused2_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    e4m3_mmvq_fused2_b<2>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_fused2_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    e4m3_mmvq_fused2_b<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_fused2_b8(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    e4m3_mmvq_fused2_b<8>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
template<int MCOLS>
__device__ __forceinline__ void e4m3_mmvq_fused3_b(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    int nb0 = (out0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int nb1 = (out1 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; b -= nb0 + nb1; }
    e4m3_mmvq_batched_row<MCOLS>(W, aq, ad, y, in_f, out_f, m, row_bytes,
                                 b * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_fused3_b2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    e4m3_mmvq_fused3_b<2>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}
extern "C" __global__ void qmatvec_e4m3_mmvq_fused3_b4(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    e4m3_mmvq_fused3_b<4>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}

// ============ F8-E4M3 BLOCK-128 SCALE (Qwen-official FP8 class) warp-per-row MMVQ ============
// lane/fp8-blk128-decode, 2026-08-05. The per-block-dequant twin of qmatvec_e4m3_mmvq above.
//
// WHY IT EXISTS: two FP8 safetensors scale classes reach memra. The per-tensor class (modelopt /
// NVIDIA) carries ONE scalar `weight_scale` and is served natively by qmatvec_e4m3_mmvq (fused at
// the write). The BLOCK-128 class (Qwen3.6-FP8 and the DeepSeek-V3 lineage: `weight_block_size
// [128,128]`) carries a BF16 `weight_scale_inv` grid of shape [ceil(out_f/128), ceil(in_f/128)]
// and `scale == 1.0`; before this kernel it had no native consumer at all and every such
// projection paid the ARM B' Q8_0 re-encode (1.0625 B/weight resident + a lossy extra hop).
//
// ARITHMETIC CONTRACT (the host reference in kernel_check.rs implements exactly this, and the
// `E4M3-BLK-MMVQ` gate proves the kernel implements the reference):
//   per k32-block blk in the SAME lane-strided walk qmatvec_e4m3_mmvq uses:
//     bs   = sum_j f32(e4m3(w[j])) * f32(aq[j])         (fmaf chain, fixed j order 0..31)
//     acc  = fmaf(s[blk >> 2] * ad[blk], bs, acc)       (ONE extra f32 mul per 32 weights)
//   s[.] is the row's scale line: `blk_scales + (o >> 7) * scale_cols`, indexed by the k128 block
//   `blk >> 2` (128/32 == 4 k32-blocks share one scale). NO clamp is needed on that index: with
//   in_f % 32 == 0, (in_f/32 - 1) >> 2 == ceil(in_f/128) - 1 for every in_f, so the last k32-block
//   always lands on the last scale column exactly (verified by the ragged-in_f gate cells).
//   The scale is folded PER K32-BLOCK, not per 128, on purpose: that keeps the walk, the register
//   footprint and the fmaf chain byte-for-byte identical in structure to the per-tensor twin, and
//   a lane's consecutive iterations (blk += 32) never revisit a scale column anyway, so a per-128
//   hoist would buy nothing while forcing a different (slower, 4-consecutive-block) walk.
//   The grid is resident as f32 (Fp8BlockScales: one host bf16->f32 decode at load, one htod), so
//   this is an f32 load from L1/L2 — the whole grid is ~21 KB even for the widest 27B projection.
//
// e4m3 DECODE CONVENTION: the HARDWARE semantics (__nv_cvt_fp8x2_to_halfraw2 via e4m3x2_to_f32x2),
// exactly like qmatvec_e4m3_mmvq — magnitude 0x7F is NaN, NOT the modelopt-0.0 that
// cu/fp8_blk_dequant.cu's closed-form host math uses. Consequence, and it is a DISPATCH
// PRECONDITION not an assumption: a resident tensor containing 0x7F/0xFF would poison its block.
// The host arm scans for those codes (fp8_blk_nan_count, already in the tree for the MMQ arm) and
// refuses the native arm for that tensor; the gate's code-coverage cell therefore asserts 254/254
// LEGAL codes, the same bar lane/fp8-mmq-v2 set.
//
// EXACTNESS LAW (unchanged from the per-tensor twin): per (token,row) the body is a pure function
// of (row bytes, that row's scale line, that token's q8_1 row) — the grid.y=m verify launch is
// bit-identical to the m=1 decode launch by construction. There is deliberately NO batched
// _b2/_b4/_b8 twin for this class yet: the m=2..15 tiers fall to the grid.y=m form above, which is
// the exact m=1 program per column (rare tier; exactness over bandwidth, same call the per-tensor
// catch-all makes).

__device__ __forceinline__ float e4m3_blk_row_dot(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, const float* __restrict__ srow,
        int nblk, int lane) {
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        // 32 e4m3 weight bytes: 2x LDG.128; 32 int8 activation: 2x LDG.128 (as the per-tensor twin).
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
            bs = fmaf(whi.y, (float)(a >> 24), bs);   // arithmetic shift: already sign-extended
        }
        acc = fmaf(srow[blk >> 2] * adrow[blk], bs, acc);
    }
    return acc;
}

extern "C" __global__ void qmatvec_e4m3_blk_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;   // this warp's output row
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    float acc = e4m3_blk_row_dot(W + (long)o * row_bytes, aq + (size_t)t * in_f,
                                 ad + (size_t)t * nblk,
                                 blk_scales + (size_t)(o >> 7) * scale_cols, nblk, lane);
    acc = warp_reduce_sum(acc);
    // No `ws` epilogue: the block class's per-tensor scale IS 1.0 by the layout contract (the
    // dispatch refuses anything else), and every scale factor is already folded per k128 above.
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ----- BLOCK-128 e4m3 BATCHED family (lane/rp-on-st, 2026-08-06). The comment on the kernel above
// noted "there is deliberately NO batched _b2/_b4/_b8 twin for this class yet: the m=2..15 tiers
// fall to the grid.y=m form, which is the exact m=1 program per column (rare tier; exactness over
// bandwidth)". That tradeoff was correct while the block-128 class was a single-stream decode lane.
// It is WRONG for serving: at serve concurrency the batched decode tick runs m = chunk width on
// EVERY projection of EVERY layer, so "weight re-read m times" is the steady-state path, not a rare
// tier — a 16x weight-traffic multiplier on the most bandwidth-bound part of the forward.
//
// Same weight-read-once structure as every other batched twin: the 32 e4m3 weight bytes and the
// row's scale line are loaded ONCE per (row, k32-block) and reused across all MCOLS activation
// columns. Per (token,row) the arithmetic is e4m3_blk_row_dot's chain VERBATIM — same lane-strided
// blk walk, same fmaf order, same `srow[blk >> 2] * adrow[blk]` fold, same warp_reduce_sum — so
// each column is BIT-IDENTICAL to the grid.y=m launch and hence to the m=1 decode launch. That is
// the exact property the exact-16 tier's admission requires.
template<int MCOLS>
__device__ __forceinline__ void e4m3_blk_mmvq_batched_row(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ srow,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const uint4* w16 = (const uint4*)(wrow + blk * 32);
        uint4 w01 = w16[0], w23 = w16[1];            // weight bytes read ONCE for all columns
        unsigned wu[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        const float s = srow[blk >> 2];              // scale line read ONCE for all columns
        // DEQUANT-HOIST (lane/rp-on-st, 2026-08-06): same fix as e4m3_mmvq_batched_row — the
        // e4m3 -> f32 conversion is per-WEIGHT, not per-column, so running it inside the column
        // loop cost MCOLS x the conversion work for one weight fetch (16x at the b16 tier).
        // Bit-identity holds by construction: same values, same order, only hoisted.
        float wf[32];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            float2 wlo = e4m3x2_to_f32x2((unsigned short)(wu[k] & 0xFFFF));
            float2 whi = e4m3x2_to_f32x2((unsigned short)(wu[k] >> 16));
            wf[k * 4 + 0] = wlo.x; wf[k * 4 + 1] = wlo.y;
            wf[k * 4 + 2] = whi.x; wf[k * 4 + 3] = whi.y;
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int au[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float bs = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int a = au[k];
                bs = fmaf(wf[k * 4 + 0], (float)(signed char)(a & 0xff), bs);
                bs = fmaf(wf[k * 4 + 1], (float)(signed char)((a >> 8) & 0xff), bs);
                bs = fmaf(wf[k * 4 + 2], (float)(signed char)((a >> 16) & 0xff), bs);
                bs = fmaf(wf[k * 4 + 3], (float)(a >> 24), bs);
            }
            // IDENTICAL fold to e4m3_blk_row_dot: s * ad[blk] first, then one fmaf into acc.
            acc[c] = fmaf(s * ad[(size_t)c * nblk + blk], bs, acc[c]);
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;   // no ws epilogue (scale == 1.0 by contract)
    }
}
template<int MCOLS>
__device__ __forceinline__ void e4m3_blk_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
    if (o >= out_f) return;
    e4m3_blk_mmvq_batched_row<MCOLS>(W, aq, ad, blk_scales + (size_t)(o >> 7) * scale_cols,
                                     y, in_f, out_f, m, row_bytes, o);
}
extern "C" __global__ void qmatvec_e4m3_blk_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    e4m3_blk_mmvq_batched<2>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}
extern "C" __global__ void qmatvec_e4m3_blk_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    e4m3_blk_mmvq_batched<4>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}
extern "C" __global__ void qmatvec_e4m3_blk_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    e4m3_blk_mmvq_batched<8>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}
extern "C" __global__ void qmatvec_e4m3_blk_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, const float* __restrict__ blk_scales,
        float* __restrict__ y, int in_f, int out_f, int m, long row_bytes, int scale_cols) {
    e4m3_blk_mmvq_batched<16>(W, aq, ad, blk_scales, y, in_f, out_f, m, row_bytes, scale_cols);
}

// ----- Q4_K batched. Per-group reusable: d_sb, dmin_sb, sc, mn, 8 decoded wpack. Per-column: act + dp4a
// (incl. the per-column sumi_sum = dp4a(0x01010101, a) min-offset term, which depends on activation). -----
template<int MCOLS>
__device__ __forceinline__ void q4k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        bool hi = (grp & 1);
        const int* q4 = (const int*)(qs + chunk * 32);
        int wpack[8];                            // decode the 4-bit weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int raw = q4[k];
            wpack[k] = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_d = 0, sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sumi_d   = dp4a(wpack[k], aq4[k], sumi_d);
                sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d_sb   * (float)((int)sc * sumi_d) * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=16 (lane/rp-on-st, 2026-08-06) — the 9B NVFP4 GGUF's exact-16 blocker, named by the
// MEMRA_EXACT16_WHY diagnostic as `L0.wqkv qtype=1 rp4=false`. Real NVFP4 GGUFs are MIXED: the
// MLP is NVFP4 while the attention/linear-attn projections stay Q4_K, and the tier's predicate is
// an ALL over every matmul — so Q4_K refused chunk 16 on a model nobody would call a "Q4_K model".
// Same q4k_mmvq_batched body as b2/b4/b8 -> bit-identical per (token,row) to the m=1 mmvq.
extern "C" __global__ void qmatvec_q4_K_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q5_K batched. Per-group reusable: d_sb, dmin_sb, sc, mn, 8 decoded 5-bit wpack. -----
template<int MCOLS>
__device__ __forceinline__ void q5k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        int wpack[8];                            // decode the 5-bit weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);
            int qh4 = get_int_b2(qh + k * 4);
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;
            wpack[k] = low | (h << 4);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_d = 0, sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sumi_d   = dp4a(wpack[k], aq4[k], sumi_d);
                sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d_sb   * (float)((int)sc * sumi_d)   * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// mcols=16 (lane/rp-on-st, 2026-08-06). The FOURTH class the exact-16 diagnostic named, on the
// same 9B NVFP4 GGUF: `L0.wqkv_gate qtype=3`. This is the pattern the lane found — a real mixed
// checkpoint spreads its ~500 matmuls over four or five quant classes (NVFP4 MLP, Q4_K qkv, Q5_K
// gate, Q6_K output, Q8_0 ssm), and the tier's predicate is an ALL, so chunk 16 was unreachable
// for EVERY shipped artifact until every class had a b16. Q5_K never mirrors (no rp twins exist
// for it at any width), so the base form is the whole requirement here. Same q5k_mmvq_batched
// body as b2/b4/b8 -> bit-identical per (token,row) to the m=1 mmvq.
extern "C" __global__ void qmatvec_q5_K_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q6_K batched. Per-group reusable: d, scales, 8 decoded signed wpack. Symmetric (no min). -----
template<int MCOLS>
__device__ __forceinline__ void q6k_mmvq_batched(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        int wpack[8];                            // decode the 6-bit signed weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int ql4 = get_int_b2(qlh + k * 4 + ql_off);
            int qh4 = get_int_b2(qhh + k * 4);
            int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
            int qhn = (qh4 >> qh_sh) & 0x03030303;
            int vpack = qln | (qhn << 4);
            wpack[k] = __vsubss4(vpack, 0x20202020);
        }
        int sc0 = (int)scn[is0], sc1 = (int)scn[is1];
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi0 = 0, sumi1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k < 4) sumi0 = dp4a(wpack[k], aq4[k], sumi0);
                else       sumi1 = dp4a(wpack[k], aq4[k], sumi1);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d * d8 * ( (float)(sumi0 * sc0) + (float)(sumi1 * sc1) );
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b16(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


// ============ k-quant batched TWO-ROWS-PER-WARP variants (2026-07-04, NVFP4 _r2 recipe port) ============
// ncu on the DRAM-cold msweep (9B real shapes, m=4): q4_K b4 long_scoreboard 19.6/issue at DRAM
// 47.7% (L2 weight hit 13%, occupancy 71%), q5_K 16.4/issue at DRAM 38.2% — memory-LATENCY bound
// exactly like the NVFP4 batched family pre-fix (ONE weight wavefront in flight/warp). Same fix:
// each warp owns TWO output rows — 2 independent weight-row streams in flight and the m activation
// columns loaded once, reused across both rows. q6_K gets the template too, but its dominant real
// shape (the 9B lm_head, out_f=248320, 75 waves) measured DRAM 90-91% = BANDWIDTH-bound at the
// wall — build-to-measure only. Q8_0 gets NO r2: its only real batched shapes are the tiny
// out_f=32 ssm_alpha/beta (8-block grids; halving a grid that never fills one SM cannot help).
// BIT-IDENTICAL per (token,row) to the matching base batched kernel: identical scale/min unpack,
// identical wpack decode, identical dp4a order (the per-column sumi_sum is INTEGER and
// row-independent — hoisting it out of the row loop is exact), identical warp_reduce_sum. Only
// the row->warp mapping (grid shape) and cross-row interleave change, both exactness-free.

// ----- Q4_K batched r2 -----
template<int MCOLS>
__device__ __forceinline__ void q4k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        int chunk = grp >> 1;
        bool hi = (grp & 1);
        // decode BOTH rows' weight groups first (both wavefronts issued together).
        float dsb[2], dmn[2];
        int   scv[2], mnv[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 144;
            dsb[r] = half_to_float(*(const unsigned short*)b);
            dmn[r] = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qs     = b + 16;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            scv[r] = sc; mnv[r] = mn;
            const int* q4 = (const int*)(qs + chunk * 32);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int raw = q4[k];
                wpack[r][k] = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            }
        }
        // each token column's activation loaded ONCE, dp4a vs both rows.
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi_d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi_d = dp4a(wpack[r][k], aq4[k], sumi_d);
                acc[r][c] += dsb[r] * (float)(scv[r] * sumi_d)   * d8
                           - dmn[r] * (float)(mnv[r] * sumi_sum) * d8;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q4_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q5_K batched r2 -----
template<int MCOLS>
__device__ __forceinline__ void q5k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3, grp = g & 7;
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        float dsb[2], dmn[2];
        int   scv[2], mnv[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 176;
            dsb[r] = half_to_float(*(const unsigned short*)b);
            dmn[r] = half_to_float(*(const unsigned short*)(b + 2));
            const unsigned char* scales = b + 4;
            const unsigned char* qh = b + 16;
            const unsigned char* qs = b + 48;
            unsigned char sc, mn;
            if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
            else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
                   mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
            scv[r] = sc; mnv[r] = mn;
            const unsigned char* q = qs + g64 * 32;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int q4  = get_int_b2(q  + k * 4);
                int qh4 = get_int_b2(qh + k * 4);
                int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
                int h   = (qh4 >> hbit) & 0x01010101;
                wpack[r][k] = low | (h << 4);
            }
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi_d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi_d = dp4a(wpack[r][k], aq4[k], sumi_d);
                acc[r][c] += dsb[r] * (float)(scv[r] * sumi_d)   * d8
                           - dmn[r] * (float)(mnv[r] * sumi_sum) * d8;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q5_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q5_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q5k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ----- Q6_K batched r2 (built to MEASURE; the 9B lm_head shape is DRAM-wall-bound, see header) -----
template<int MCOLS>
__device__ __forceinline__ void q6k_mmvq_batched_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    if (o0 >= out_f) return;
    const bool has1 = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow0 = W + (long)o0 * row_bytes;
    float acc[2][MCOLS];
    #pragma unroll
    for (int r = 0; r < 2; r++)
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) acc[r][c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        int n   = grp >> 2;
        int run = grp & 3;
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);
        int qh_sh  = run * 2;
        float dv[2];
        int   sc0v[2], sc1v[2];
        int   wpack[2][8];
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r == 1 && !has1) break;
            const unsigned char* b = wrow0 + (long)r * row_bytes + (long)sblk * 210;
            const unsigned char* qlh = b + n * 64;
            const unsigned char* qhh = b + 128 + n * 32;
            const signed char*   scn = (const signed char*)(b + 192) + n * 8;
            dv[r] = half_to_float(*(const unsigned short*)(b + 208));
            sc0v[r] = (int)scn[is0]; sc1v[r] = (int)scn[is1];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int ql4 = get_int_b2(qlh + k * 4 + ql_off);
                int qh4 = get_int_b2(qhh + k * 4);
                int qln = ql_hi ? ((ql4 >> 4) & 0x0F0F0F0F) : (ql4 & 0x0F0F0F0F);
                int qhn = (qh4 >> qh_sh) & 0x03030303;
                int vpack = qln | (qhn << 4);
                wpack[r][k] = __vsubss4(vpack, 0x20202020);
            }
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float d8 = ad[(size_t)c * nsb + g];
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                if (r == 1 && !has1) break;
                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < 4) sumi0 = dp4a(wpack[r][k], aq4[k], sumi0);
                    else       sumi1 = dp4a(wpack[r][k], aq4[k], sumi1);
                }
                acc[r][c] += dv[r] * d8 * ( (float)(sumi0 * sc0v[r]) + (float)(sumi1 * sc1v[r]) );
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        if (r == 1 && !has1) break;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            float a = warp_reduce_sum(acc[r][c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + r] = a;
        }
    }
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b2_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b4_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void __launch_bounds__(128, 8) qmatvec_q6_K_mmvq_b4_r2w8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b8_r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q6k_mmvq_batched_r2<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// Q8_0 weight x q8_1 activation, int8 dp4a. y[m,out] = sum_blocks d_w*d_a*dp4a(w_qs, a_qs).
// W: block_q8_0 rows (34 bytes/block). aq: int8 [m,in]; ad: f32 [m, in/32].
// grid (out, m); block 128 threads (4 warps), each warp strides the in/32 blocks.
extern "C" __global__ void qmatvec_q8_0_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = tid; blk < nblk; blk += blockDim.x) {
        const unsigned char* wb = wrow + blk * 34;
        float dw = half_to_float(*(const unsigned short*)wb);   // weight block scale (2-byte aligned OK)
        const unsigned char* wq = wb + 2;                       // qs: 2-byte aligned -> get_int_b2
        const int4* aq16 = (const int4*)(arow + blk * 32);      // 2x int4 (128-bit), 32-aligned
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = dp4a(get_int_b2(wq + k * 4), aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// Q4_K decode MMVQ (int8 dp4a). Min-offset via the q8_1 activation-sum term.
// y = sum_subblock [ d*sc*d8*dp4a(nibble,a) - dmin*m*d8*sum(a) ]. d/dmin folded PER sub-block
// (a thread's stripe crosses superblocks). Nibble scheme matches deq_q4_k oracle.
extern "C" __global__ void qmatvec_q4_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;                 // total 32-blocks per row
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 144;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs     = b + 16;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int chunk = grp >> 1;
        // qs at byte off 16 in a 144B superblock -> 4-byte aligned; chunk*32 keeps it 4-byte aligned.
        const int* q4 = (const int*)(qs + chunk * 32);
        bool hi = (grp & 1);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // nibble-by-shift over 4 packed weights (llama.cpp vmmq style, vecdotq.cuh:514-515):
            // low nibbles for even groups, high nibbles for odd. 0x0F0F0F0F masks all 4 lanes.
            int raw = q4[k];
            int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d) * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// Q6_K decode MMVQ (symmetric, no min). w=(ql|qh<<4)-32 signed; per-16 signed scales; fp16 d.
// Matches deq_q6_k oracle: n=grp>>2 half, run=grp&3, is=run*2+(il>>4).
extern "C" __global__ void qmatvec_q6_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3;
        int grp  = g & 7;
        const unsigned char* b = wrow + (long)sblk * 210;
        const unsigned char* ql = b;
        const unsigned char* qh = b + 128;
        const signed char*   scales = (const signed char*)(b + 192);
        float d = half_to_float(*(const unsigned short*)(b + 208));
        int n   = grp >> 2;
        int run = grp & 3;
        const unsigned char* qlh = ql + n * 64;
        const unsigned char* qhh = qh + n * 32;
        const signed char*   scn = scales + n * 8;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int is0 = run * 2 + 0;
        int is1 = run * 2 + 1;
        int sumi0 = 0, sumi1 = 0;
        // ql offset for low/high nibble (run 0/1 use bytes [il], run 2/3 use [il+32]);
        // Stage-A deq_q6_k: run0 qlh[il]&0xF, run1 qlh[il+32]&0xF, run2 qlh[il]>>4, run3 qlh[il+32]>>4.
        // => byte offset +32 on ODD runs (1,3); high nibble on runs >=2 (2,3). The offset is (run&1),
        //    NOT (run>=2) — the old (run>=2) swapped run-1<->run-2 ql bytes (rel 0.34 on Q6_K lm_head).
        int ql_off = (run & 1) ? 32 : 0;
        int ql_hi  = (run >= 2);          // true -> high nibble of ql byte
        int qh_sh  = run * 2;             // 0,2,4,6
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            // Build the 4 unsigned 6-bit weights (0..63) packed one per byte, then __vsubss4 the
            // -32 across all 4 lanes in one SIMD op (llama.cpp vecdotq.cuh:638). Saturating sub is
            // exact here: vals are 0..63 so result is -32..31, well within int8.
            unsigned int vpack = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int il = k * 4 + e;
                int ql_bits = ql_hi ? (qlh[il + ql_off] >> 4) : (qlh[il + ql_off] & 0xF);
                int qh_bits = (qhh[il] >> qh_sh) & 3;
                unsigned int w = (unsigned int)(ql_bits | (qh_bits << 4));   // 0..63
                vpack |= (w & 0xff) << (e * 8);
            }
            int wpack = __vsubss4((int)vpack, 0x20202020);   // subtract 32 per byte (signed sat)
            int a = aq4[k];
            if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
            else       sumi1 = dp4a(wpack, a, sumi1);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
    }
    mmvq_block_reduce_write(acc, y, (size_t)t * out_f + o, tid);
}

// ===== Q5_K decode MMVQ (int8 dp4a). Unsigned 5-bit weight + min-offset via q8_1 sum. =====
extern "C" __global__ void qmatvec_q5_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 176;
        float d_sb    = half_to_float(*(const unsigned short*)b);
        float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qh = b + 16;
        const unsigned char* qs = b + 48;
        unsigned char sc, mn;
        if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
        else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
               mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
        int g64 = grp >> 1; bool hi = (grp & 1); int hbit = 2 * g64 + (hi ? 1 : 0);
        const unsigned char* q = qs + g64 * 32;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        // VECTORIZED unpack (was scalar 4-byte inner loop = ~16 ALU ops/k starving DRAM to 31%).
        // The 4 q bytes (idx=k*4..+3) and 4 qh bytes are contiguous -> one get_int_b2 each (2-aligned:
        // q5_K block=176, qs=b+48, qh=b+16 all even). SIMD-extract: low nibble per byte + bit hbit of
        // qh per byte. BIT-IDENTICAL: same byte->bit e*8 packing, same lowbits|(h<<4) per byte.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int q4  = get_int_b2(q  + k * 4);                    // 4 q bytes
            int qh4 = get_int_b2(qh + k * 4);                    // 4 qh bytes
            int low = hi ? ((q4 >> 4) & 0x0F0F0F0F) : (q4 & 0x0F0F0F0F);
            int h   = (qh4 >> hbit) & 0x01010101;                // bit hbit per byte, 0/1
            int wpack = low | (h << 4);                          // per byte 0..31
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d)   * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== Q3_K decode MMVQ (symmetric, signed 3-bit weight, NO min term). =====
// 32-chunk grp covers TWO 16-elem sub-blocks => two scale indices (lo/hi 16).
extern "C" __global__ void qmatvec_q3_K_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 3, grp = g & 7;
        const unsigned char* b = wrow + (long)sblk * 110;
        const unsigned char* hmask  = b;
        const unsigned char* qs     = b + 32;
        const unsigned char* scbyte = b + 96;
        float d = half_to_float(*(const unsigned short*)(b + 108));
        // unpack 16 6-bit signed scales
        unsigned int aux0 = scbyte[0]|(scbyte[1]<<8)|(scbyte[2]<<16)|(scbyte[3]<<24);
        unsigned int aux1 = scbyte[4]|(scbyte[5]<<8)|(scbyte[6]<<16)|(scbyte[7]<<24);
        unsigned int aux2 = scbyte[8]|(scbyte[9]<<8)|(scbyte[10]<<16)|(scbyte[11]<<24);
        const unsigned int km1=0x03030303u, km2=0x0f0f0f0fu, tmp=aux2;
        unsigned int nA[4]={ (aux0&km2)|(((tmp>>0)&km1)<<4), (aux1&km2)|(((tmp>>2)&km1)<<4),
                             ((aux0>>4)&km2)|(((tmp>>4)&km1)<<4), ((aux1>>4)&km2)|(((tmp>>6)&km1)<<4) };
        signed char sc[16];
        for(int kk=0;kk<4;kk++){ sc[kk*4+0]=(signed char)nA[kk]; sc[kk*4+1]=(signed char)(nA[kk]>>8);
                                 sc[kk*4+2]=(signed char)(nA[kk]>>16); sc[kk*4+3]=(signed char)(nA[kk]>>24); }
        // grp -> half/jiter/shift/m_bit/scale-base. half=grp>>2, jiter=grp&3.
        int half = grp >> 2, jiter = grp & 3;
        int shift = 2 * jiter;
        int m_bit_idx = half * 4 + jiter;
        const unsigned char* q  = qs    + half * 32;   // 32-byte qs run for this half
        const unsigned char* hm = hmask;               // hmask not chunked: index by element directly
        int is_lo = half * 8 + jiter * 2 + 0;          // scale for lo 16 elems
        int is_hi = half * 8 + jiter * 2 + 1;          // scale for hi 16 elems
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumlo = 0, sumhi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = 0; bool hiHalf = (k >= 4);     // k0..3 -> lo16, k4..7 -> hi16
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int idx = k * 4 + e;                   // 0..31 within chunk
                int l = idx & 15;
                int sub = idx >> 4;                    // 0 -> q[l], 1 -> q[l+16]
                int q2 = (q[sub * 16 + l] >> shift) & 3;
                int hb = (hm[sub * 16 + l] & (1 << m_bit_idx)) ? 0 : 4;
                int w = q2 - hb;                       // signed -4..3
                wpack |= (w & 0xff) << (e * 8);
            }
            int a = aq4[k];
            if (!hiHalf) sumlo = dp4a(wpack, a, sumlo);
            else         sumhi = dp4a(wpack, a, sumhi);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)sumlo * (float)((int)sc[is_lo] - 32)
                        + (float)sumhi * (float)((int)sc[is_hi] - 32) );
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== NVFP4 decode MMVQ (codebook->int8 dp4a, symmetric, no min). =====
// 32-elem activation block g covers TWO 16-elem NVFP4 sub-blocks (own UE4M3 scale each).
extern "C" __global__ void qmatvec_nvfp4_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;          // which 64-elem block_nvfp4 (36 bytes)
        int whichHalf = g & 1;      // 0 -> sub 0,1 ; 1 -> sub 2,3
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2, s1 = s0 + 1;
        (void)s1;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);  // 2x int4 (128-bit)
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        // sub-block s_local=0 -> activation ints aq4[0..3], s_local=1 -> aq4[4..7]
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;       // 8 qs bytes for this sub-block
            // Codebook the 16 packed 4-bit weights via __byte_perm (get_int_from_table_16_d) instead
            // of 16 scalar kvalues_mxfp4_d[] loads — this loop was ALU-bound (19% of BW ceiling).
            // For 4 packed bytes, .x = low-nibble codes (4 int8s packed) = old wlo*, .y = high-nibble
            // codes = old whi*. P1: qss is 4-aligned (row_bytes=(in_f/64)*36 mult of 4; qs=b+4; qss=+s*8)
            // -> single LDG.E.32 each via get_int_b4 (was 4x LDG.E.U8). int2/64-bit NOT safe: rows only
            // 8-aligned when in_f%128==0.
            int q4a = get_int_b4(qss);
            int q4b = get_int_b4(qss + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);  // .x=wlo0 (elems0..3) .y=whi0 (elems8..11)
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);  // .x=wlo1 (elems4..7) .y=whi1 (elems12..15)
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);   // elems 0..3
            sumi = dp4a(vb.x, aq4[base + 1], sumi);   // elems 4..7
            sumi = dp4a(va.y, aq4[base + 2], sumi);   // elems 8..11
            sumi = dp4a(vb.y, aq4[base + 3], sumi);   // elems 12..15
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===== IQ4_XS decode MMVQ (OPTIONAL perf path; codebook->int8 dp4a, symmetric, no min). =====
// nibble->position split: low nibbles qs[0..15] -> elems 0..15, high -> elems 16..31.
// ---- MoE EXPERT dp4a DOT BODIES (2026-07-06 dp4a arc; HANDOVER "MoE expert dp4a upgrade") ----
// Per-32-elem-group int dots vs a q8_1 activation, separable block scales OUTSIDE the int dot —
// the q4k/q5k mmvq structure. IQ3_S sign trick ported from llama vec_dot_iq3_s_q8_1
// (vecdotq.cuh:1148): signs expand via __vcmpne4 mask -> XOR-sub negation on the packed grid
// bytes. Layout matches memra deq_iq3_s (block 110B: d@0 qs@2 qh@66 signs@74 scales@106).
// Group g = 32 elems; IQ3_S block = 256 elems = 8 groups; IQ4_XS block = 256 elems = 8 groups.
__device__ __forceinline__ float expert_dot_iq3s_g(const unsigned char* wrow, int g,
                                                   const signed char* aqb, float d8) {
    int sblk = g >> 3, ib32 = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    float d = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs    = b + 2  + ib32 * 8;
    unsigned char qh           = b[66 + ib32];
    const unsigned char* signs = b + 74 + ib32 * 4;
    const unsigned char* scales= b + 106;
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
        unsigned char sb = signs[l0 / 2];
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        int grid_l = __vsub4(gl ^ signs0, signs0);
        int grid_h = __vsub4(gh ^ signs1, signs1);
        sumi = dp4a(grid_l, aq4[l0 + 0], sumi);
        sumi = dp4a(grid_h, aq4[l0 + 1], sumi);
    }
    return db * (float)sumi * d8;
}
__device__ __forceinline__ float expert_dot_iq4xs_g(const unsigned char* wrow, int g,
                                                    const signed char* aqb, float d8) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    float d_sb = half_to_float(*(const unsigned short*)b);
    unsigned short sh = *(const unsigned short*)(b + 2);
    const unsigned char* sl = b + 4;
    const unsigned char* qs = b + 8 + ib * 16;
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    int scale = ls - 32;
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int sumi = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int wlo = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
        int whi = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
        sumi = dp4a(wlo, aLo[k], sumi);
        sumi = dp4a(whi, aHi[k], sumi);
    }
    return d_sb * (float)(scale * sumi) * d8;
}
// K-QUANT expert dot bodies (2026-07-06): the UD-IQ4_XS 35B mix puts Q3_K/Q4_K/Q6_K experts on
// the tail layers (blk.38-40) — those layers fell to the f32-dequant _dev arm (80us vs 15us
// launches = 8.5% of the fixed-build decode window). Group-g bodies lifted VERBATIM from
// qmatvec_q3_K_dp4a / qmatvec_q4_K_mmvq / qmatvec_q6_K_mmvq (same unpack, same dp4a order,
// same accumulate expression), so per (row,group) the math matches those kernels bit-for-bit.
__device__ __forceinline__ float expert_dot_q3k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    const unsigned char* hmask  = b;
    const unsigned char* qs     = b + 32;
    const unsigned char* scbyte = b + 96;
    float d = half_to_float(*(const unsigned short*)(b + 108));
    unsigned int aux0 = scbyte[0]|(scbyte[1]<<8)|(scbyte[2]<<16)|(scbyte[3]<<24);
    unsigned int aux1 = scbyte[4]|(scbyte[5]<<8)|(scbyte[6]<<16)|(scbyte[7]<<24);
    unsigned int aux2 = scbyte[8]|(scbyte[9]<<8)|(scbyte[10]<<16)|(scbyte[11]<<24);
    const unsigned int km1=0x03030303u, km2=0x0f0f0f0fu, tmp=aux2;
    unsigned int nA[4]={ (aux0&km2)|(((tmp>>0)&km1)<<4), (aux1&km2)|(((tmp>>2)&km1)<<4),
                         ((aux0>>4)&km2)|(((tmp>>4)&km1)<<4), ((aux1>>4)&km2)|(((tmp>>6)&km1)<<4) };
    signed char sc[16];
    for(int kk=0;kk<4;kk++){ sc[kk*4+0]=(signed char)nA[kk]; sc[kk*4+1]=(signed char)(nA[kk]>>8);
                             sc[kk*4+2]=(signed char)(nA[kk]>>16); sc[kk*4+3]=(signed char)(nA[kk]>>24); }
    int half = grp >> 2, jiter = grp & 3;
    int shift = 2 * jiter;
    int m_bit_idx = half * 4 + jiter;
    const unsigned char* q  = qs + half * 32;
    const unsigned char* hm = hmask;
    int is_lo = half * 8 + jiter * 2 + 0;
    int is_hi = half * 8 + jiter * 2 + 1;
    const int* aq4 = (const int*)aqb;
    int sumlo = 0, sumhi = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int wpack = 0; bool hiHalf = (k >= 4);
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int idx = k * 4 + e;
            int l = idx & 15;
            int sub = idx >> 4;
            int q2 = (q[sub * 16 + l] >> shift) & 3;
            int hb = (hm[sub * 16 + l] & (1 << m_bit_idx)) ? 0 : 4;
            int w = q2 - hb;
            wpack |= (w & 0xff) << (e * 8);
        }
        int a = aq4[k];
        if (!hiHalf) sumlo = dp4a(wpack, a, sumlo);
        else         sumhi = dp4a(wpack, a, sumhi);
    }
    return d * d8 * ( (float)sumlo * (float)((int)sc[is_lo] - 32)
                    + (float)sumhi * (float)((int)sc[is_hi] - 32) );
}
__device__ __forceinline__ float expert_dot_q4k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 144;
    float d_sb    = half_to_float(*(const unsigned short*)b);
    float dmin_sb = half_to_float(*(const unsigned short*)(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs     = b + 16;
    unsigned char sc, mn;
    if (grp < 4) { sc = scales[grp] & 63; mn = scales[grp + 4] & 63; }
    else { sc = (scales[grp + 4] & 0xF) | ((scales[grp - 4] >> 6) << 4);
           mn = (scales[grp + 4] >> 4) | ((scales[grp] >> 6) << 4); }
    int chunk = grp >> 1;
    const int* q4 = (const int*)(qs + chunk * 32);
    bool hi = (grp & 1);
    const int* aq4 = (const int*)aqb;
    int sumi_d = 0, sumi_sum = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int raw = q4[k];
        int wpack = hi ? ((raw >> 4) & 0x0F0F0F0F) : (raw & 0x0F0F0F0F);
        int a = aq4[k];
        sumi_d   = dp4a(wpack, a, sumi_d);
        sumi_sum = dp4a(0x01010101, a, sumi_sum);
    }
    return d_sb * (float)((int)sc * sumi_d) * d8 - dmin_sb * (float)((int)mn * sumi_sum) * d8;
}
__device__ __forceinline__ float expert_dot_q6k_g(const unsigned char* wrow, int g,
                                                  const signed char* aqb, float d8) {
    int sblk = g >> 3, grp = g & 7;
    const unsigned char* b = wrow + (long)sblk * 210;
    const unsigned char* ql = b;
    const unsigned char* qh = b + 128;
    const signed char*   scales = (const signed char*)(b + 192);
    float d = half_to_float(*(const unsigned short*)(b + 208));
    int n   = grp >> 2;
    int run = grp & 3;
    const unsigned char* qlh = ql + n * 64;
    const unsigned char* qhh = qh + n * 32;
    const signed char*   scn = scales + n * 8;
    const int* aq4 = (const int*)aqb;
    int is0 = run * 2 + 0;
    int is1 = run * 2 + 1;
    int sumi0 = 0, sumi1 = 0;
    int ql_off = (run & 1) ? 32 : 0;
    int ql_hi  = (run >= 2);
    int qh_sh  = run * 2;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int il = k * 4;
        int qlw = get_int_b2(qlh + ql_off + il);
        int qhw = get_int_b2(qhh + il);
        int qln = ql_hi ? ((qlw >> 4) & 0x0F0F0F0F) : (qlw & 0x0F0F0F0F);
        int qhn = (qhw >> qh_sh) & 0x03030303;
        int vpack = qln | (qhn << 4);
        int wpack = __vsubss4(vpack, 0x20202020);
        int a = aq4[k];
        if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
        else       sumi1 = dp4a(wpack, a, sumi1);
    }
    return d * d8 * ( (float)(sumi0 * (int)scn[is0]) + (float)(sumi1 * (int)scn[is1]) );
}

// group-dispatching wrapper: qtype -> dot body (compile-time-hot switch, bodies inlined)
// NVFP4 expert dot (2026-07-07, MiniMax-M3): group-g body lifted VERBATIM from qmatvec_nvfp4_mmvq
// (same 36B GGUF block walk, same get_int_from_table_16_d + ue4m3 sub-scales, same dp4a order) —
// bit-identical per (row, group) to the m=1 kernel. The per-expert weight_scale_2 macro is applied
// by the CALLER (ffn_act_scaled / axpy fold), matching the dense-path contract.
// The arithmetic core shared by the interleaved and the split-plane (rp) expert group dots
// (2026-09-04, memra#147): `qh` points at the 16 quant bytes of the group's half-block, `d_bytes`
// at the block's 4 scale bytes, `s0` selects the half's two sub-block scales. Both layouts call
// THIS body so the compiler sees one expression tree at both sites: same ints (get_int_b4 at
// qh+0/+4/+8/+12), same table lookups, same dp4a order, same partial/d8 chain.
// The register form of the core (lane/glm5-moe-rows-ilp-20260904): the group's four quant
// ints and two scale bytes ALREADY LOADED, so a caller can hoist several groups' loads ahead
// of the math. `expert_dot_nvfp4_core` below is this body fed by its own loads: one
// expression tree at every site, which is the whole pinned-arithmetic argument.
__device__ __forceinline__ float expert_dot_nvfp4_core_regs(int q4a0, int q4b0, int q4a1, int q4b1,
                                                            unsigned char d0, unsigned char d1,
                                                            const int* aq4, float d8) {
    float partial = 0.0f;
    #pragma unroll
    for (int sl = 0; sl < 2; sl++) {
        int q4a = sl == 0 ? q4a0 : q4a1;
        int q4b = sl == 0 ? q4b0 : q4b1;
        int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
        int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
        int base = sl * 4;
        int sumi = 0;
        sumi = dp4a(va.x, aq4[base + 0], sumi);
        sumi = dp4a(vb.x, aq4[base + 1], sumi);
        sumi = dp4a(va.y, aq4[base + 2], sumi);
        sumi = dp4a(vb.y, aq4[base + 3], sumi);
        // PINNED arithmetic (2026-09-04): ptxas fused the outer multiply into the caller's add at
        // the split-plane call site and not at the interleaved one (24 of 12288 gate/up and 1577
        // of 4096 down outputs differed by 2-4 ulps with one shared source body). The original
        // shipped bits are: inner multiply-add FUSED, outer d8 multiply SEPARATE. Both are
        // written as .rn intrinsics so no call site can be compiled differently; verified 0 diffs
        // against the untouched-main outputs for shipped, _w4 and rp (bench dump compare).
        partial = __fmaf_rn(ue4m3_to_f32_d(sl == 0 ? d0 : d1), (float)sumi, partial);
    }
    return __fmul_rn(d8, partial);
}
__device__ __forceinline__ float expert_dot_nvfp4_core(const unsigned char* qh,
                                                       const unsigned char* d_bytes, int s0,
                                                       const signed char* aqb, float d8) {
    // sl = 0 reads qh+0/+4 and d_bytes[s0]; sl = 1 reads qh+8/+12 and d_bytes[s0+1]: the
    // exact loads the loop form issued, in registers.
    return expert_dot_nvfp4_core_regs(get_int_b4(qh), get_int_b4(qh + 4), get_int_b4(qh + 8),
                                      get_int_b4(qh + 12), d_bytes[s0], d_bytes[s0 + 1],
                                      (const int*)aqb, d8);
}
__device__ __forceinline__ float expert_dot_nvfp4_g(const unsigned char* wrow, int g,
                                                    const signed char* aqb, float d8) {
    int sblk = g >> 1;
    int whichHalf = g & 1;
    const unsigned char* b = wrow + (long)sblk * 36;
    int s0 = whichHalf * 2;
    // half-block quants: qs = b + 4, the half's 16 bytes start at qs + s0 * 8
    return expert_dot_nvfp4_core(b + 4 + s0 * 8, b, s0, aqb, d8);
}
// Slot-major (QT_NVFP4_V2) row: slot g's 16 quant bytes at g*16, its two UE4M3 scale bytes at
// nsb*16 + g*2 (the same order as the 64-block's d[0..4]: g = 2*sblk + half -> sblk*4 + s0).
// One 16B window per lane-group at a 16B lane stride instead of the 36B straggle; the
// arithmetic is the shared pinned core, so it is bit-identical to expert_dot_nvfp4_g per row
// (bench dump compare 0 diffs, memra#147). This is the resident-expert-slab layout behind
// MEMRA_MOE_EXPERT_RP.
__device__ __forceinline__ float expert_dot_nvfp4_v2_g(const unsigned char* wrow, int g, int nsb,
                                                       const signed char* aqb, float d8) {
    int s0 = (g & 1) * 2;
    return expert_dot_nvfp4_core(wrow + (size_t)g * 16, wrow + (size_t)nsb * 16 + (size_t)(g >> 1) * 4,
                                 s0, aqb, d8);
}

// ---- Q4_0 group dot (gemma4 QAT experts): one 18B block per 32-elem group; the exact
// qmatvec_q4_0_mmvq accumulation chain (dp4a nibbles + inline ones-sum, d*(sumi-8*sums)*d8). ----
__device__ __forceinline__ float expert_dot_q4_0_g(const unsigned char* wrow, int g,
                                                   const signed char* aqb, float d8) {
    const unsigned char* b = wrow + (long)g * 18;
    float d4 = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs = b + 2;
    const int* aq4 = (const int*)aqb;
    int sumi = 0, sums = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t raw;
        memcpy(&raw, qs + 4 * k, 4);
        int lo = (int)(raw & 0x0F0F0F0Fu);
        int hi = (int)((raw >> 4) & 0x0F0F0F0Fu);
        int a_lo = aq4[k];
        int a_hi = aq4[4 + k];
        sumi = dp4a(lo, a_lo, sumi);
        sumi = dp4a(hi, a_hi, sumi);
        sums = dp4a(0x01010101, a_lo, sums);
        sums = dp4a(0x01010101, a_hi, sums);
    }
    return d4 * (float)(sumi - 8 * sums) * d8;
}

// `nsb` (= in_f/32, the row's slot count) locates the slot-major (QT_NVFP4_V2) scale tail; every
// other layout ignores it. Added 2026-09-04 (memra#147) so the resident-slab layout is a
// property of the bytes (the qtype the slab carries), never of the kernel that reads them.
__device__ __forceinline__ float expert_dot_g(int qtype, const unsigned char* wrow, int g,
                                              const signed char* aqb, float d8, int nsb) {
    if (qtype == QT_IQ3_S)  return expert_dot_iq3s_g(wrow, g, aqb, d8);
    if (qtype == QT_IQ4_XS) return expert_dot_iq4xs_g(wrow, g, aqb, d8);
    if (qtype == QT_Q3_K)   return expert_dot_q3k_g(wrow, g, aqb, d8);
    if (qtype == QT_Q4_K)   return expert_dot_q4k_g(wrow, g, aqb, d8);
    if (qtype == QT_Q6_K)   return expert_dot_q6k_g(wrow, g, aqb, d8);
    if (qtype == QT_NVFP4)  return expert_dot_nvfp4_g(wrow, g, aqb, d8);
    if (qtype == QT_NVFP4_V2) return expert_dot_nvfp4_v2_g(wrow, g, nsb, aqb, d8);
    if (qtype == QT_Q4_0)   return expert_dot_q4_0_g(wrow, g, aqb, d8);
    // QT_NVFP4_RP is the TRUNK's per-tensor split-plane layout; an expert slab never carries
    // it (the resident-slab repack is the per-row QT_NVFP4_V2 form). Reaching here with it is a
    // wiring error: poison the dot (NaN) so the gate and the tape scream, never a silent zero.
    if (qtype == QT_NVFP4_RP) return __int_as_float(0x7fc00000);
    return 0.0f; // caller gates on supported qtypes
}

// ---- NVFP4 SPLIT-PLANE expert group dot (lane/moe-expert-rp, memra#147, 2026-09-04) ----
// WHY: root ncu at the served GLM-5.3-Flash geometry measured moe_gate_up_preclamp8_q8(_w4) at
// 24.97 L1 sectors per warp global-load request (4 = coalesced, 32 = one sector per lane) and
// moe_down8_fma_q8_w4 at 18.20: expert_dot_nvfp4_g walks the 36B interleaved GGUF block, so each
// lane's group is 5 scattered 4B loads at a 36B lane stride. The trunk cured the same disease with
// the split-plane `_rp` layout (repack_nvfp4_split: quant plane rows x nsb64 x 32B, then scale
// plane rows x nsb64 x 4B; microprobe m=1 1.34x, bit-identical). This is that cure for the
// resident expert slabs: per expert the SAME byte permutation, applied at upload (DevExps.rp),
// and per group ONE 16B-aligned LDG.128 of quants (lane stride 16B, contiguous across the warp)
// plus one 4B scale word.
// BIT-IDENTITY (value- and order-level with expert_dot_nvfp4_g): qw.x/qw.y/qw.z/qw.w are the
// same little-endian ints get_int_b4 read at qs+0/+4/+8/+12 of the group's half-block, the scale
// byte (cscw >> 8*s) & 0xFF is d_bytes[s], the table lookups, the dp4a order (va.x, vb.x, va.y,
// vb.y per sl) and the closing d8 * partial are unchanged.
// ALIGNMENT: the quant plane row is 16B-aligned when the expert base is (cudaMalloc slabs are
// 256B-aligned, expert_stride = rows*nsb64*36 is a multiple of 16 when nsb64 % 4 == 0, which the
// host door requires: in_f % 256 == 0). Streaming loads: expert bytes are single-use per token.

// ---- IQ4_XS WIDE-LOAD group dot (down8 lane 2026-07-08) ----
// WHY: the 35B down kernel (w8h2) runs at 47% of the byte-math wall (11.1us vs 5.2us) — NOT
// bandwidth-bound. The issue count is: expert_dot_iq4xs_g spends 16 LDG.U8 (qs bytes) + 32
// divergent byte lookups into kvalues_iq4nl_d + ~60 shift/or pack ALU per 32-elem group,
// against just 8 dp4a of real work. This body computes the SAME packed ints from the SAME
// bytes with 2 LDG.64 (qs) + 1 LDG.64 (d/sh/sl header) + 4 uniform u32 table words through
// get_int_from_table_16_d (~5 byte_perm per int pair — the llama.cpp vecdotq recipe, already
// proven bit-clean on the NVFP4/MXFP4 path here).
// BIT-IDENTITY: value-level, not order-level — wlo/whi/scale/d_sb are the exact same values
// expert_dot_iq4xs_g produces (little-endian byte extraction == the scalar byte loads; .x/.y
// of the table lookup == the low/high-nibble scalar packs), the dp4a issue order is unchanged
// (lo,hi per k), and the closing float expression is the same. sumi is exact integer math.
// ALIGNMENT: block=136B and every IQ4_XS row/expert stride here is a multiple of 8, so b is
// 8-aligned whenever the expert slab base is (cudaMalloc slabs are 256B-aligned). A warp-
// uniform guard falls back to the scalar body for any exotic base — same values either way.
__device__ __forceinline__ float expert_dot_iq4xs_g_v(const unsigned char* wrow, int g,
                                                      const signed char* aqb, float d8) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    if (((unsigned long long)b & 7ull) != 0ull)
        return expert_dot_iq4xs_g(wrow, g, aqb, d8);      // non-8-aligned slab: scalar body
    uint2 hdr = __ldcs((const uint2*)b);                  // d(2B) | sh(2B) | sl(4B), one LDG.64 (streaming: single-use expert bytes)
    float d_sb = half_to_float((unsigned short)(hdr.x & 0xffffu));
    unsigned short sh = (unsigned short)(hdr.x >> 16);
    // sl[ib>>1] is byte (ib>>1) of hdr.y (little-endian); fold the byte+nibble shifts.
    int ls = ((hdr.y >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    int scale = ls - 32;
    const int2* qs2 = (const int2*)(b + 8 + ib * 16);     // (8 + ib*16) % 8 == 0 -> 8-aligned
    int2 q01 = __ldcs(&qs2[0]), q23 = __ldcs(&qs2[1]);
    const int* aLo = (const int*)(aqb);
    const int* aHi = (const int*)(aqb + 16);
    int2 v0 = get_int_from_table_16_d(q01.x, kvalues_iq4nl_d);
    int2 v1 = get_int_from_table_16_d(q01.y, kvalues_iq4nl_d);
    int2 v2 = get_int_from_table_16_d(q23.x, kvalues_iq4nl_d);
    int2 v3 = get_int_from_table_16_d(q23.y, kvalues_iq4nl_d);
    int sumi = 0;                                          // same lo,hi dp4a order per k as scalar
    sumi = dp4a(v0.x, aLo[0], sumi); sumi = dp4a(v0.y, aHi[0], sumi);
    sumi = dp4a(v1.x, aLo[1], sumi); sumi = dp4a(v1.y, aHi[1], sumi);
    sumi = dp4a(v2.x, aLo[2], sumi); sumi = dp4a(v2.y, aHi[2], sumi);
    sumi = dp4a(v3.x, aLo[3], sumi); sumi = dp4a(v3.y, aHi[3], sumi);
    return d_sb * (float)(scale * sumi) * d8;
}
// ---- Q4_0 WIDE-LOAD group dot (gemma A4B lane 2026-07-12) ----
// WHY: expert_dot_q4_0_g issues 1 U16 + 16 byte-class loads per 18B block (the q4_0 stride
// is 2 mod 4 — nothing aligns); the gemma MoE pair reads DRAM 50% / SM 40% with the load
// chain as the critical path (same disease the trunk q4rp split-plane cured). This body
// reads the SAME 18 bytes as 6 aligned LDG.32 + funnelshift extraction (REVISION 4b recipe,
// SASS-proven on fa_v4_stage_k): same bytes -> same lo/hi ints -> same dp4a order ->
// bit-identical result.
// OVERREAD: the aligned 24B window reads up to 6B past the block — within a row/slab that is
// the next block's bytes; the LAST block of an expert slab needs the 8B tail pad at the moe
// alloc site (see moe slab alloc).
__device__ __forceinline__ float expert_dot_q4_0_g_v(const unsigned char* wrow, int g,
                                                     const signed char* aqb, float d8) {
    const unsigned char* b = wrow + (long)g * 18;
    const unsigned sh8 = ((unsigned)(size_t)b & 3u) * 8u;
    const uint32_t* ap = (const uint32_t*)((size_t)b & ~(size_t)3);
    // streaming loads (2026-07-14 L2 arc): expert bytes are single-use per token
    uint32_t w0 = __ldcs(&ap[0]), w1 = __ldcs(&ap[1]), w2 = __ldcs(&ap[2]),
             w3 = __ldcs(&ap[3]), w4 = __ldcs(&ap[4]), w5 = __ldcs(&ap[5]);
    uint32_t s0 = __funnelshift_r(w0, w1, sh8);   // bytes b[0..3]
    uint32_t s1 = __funnelshift_r(w1, w2, sh8);   // b[4..7]
    uint32_t s2 = __funnelshift_r(w2, w3, sh8);   // b[8..11]
    uint32_t s3 = __funnelshift_r(w3, w4, sh8);   // b[12..15]
    uint32_t s4 = __funnelshift_r(w4, w5, sh8);   // b[16..19] (2B past the block)
    float d4 = half_to_float((unsigned short)(s0 & 0xffffu));
    // qs word k = bytes b[2+4k .. 5+4k] — one more 16-bit funnel over the byte stream.
    uint32_t q0 = __funnelshift_r(s0, s1, 16), q1 = __funnelshift_r(s1, s2, 16);
    uint32_t q2 = __funnelshift_r(s2, s3, 16), q3 = __funnelshift_r(s3, s4, 16);
    const uint32_t qw[4] = { q0, q1, q2, q3 };
    const int* aq4 = (const int*)aqb;
    int sumi = 0, sums = 0;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int lo = (int)(qw[k] & 0x0F0F0F0Fu);
        int hi = (int)((qw[k] >> 4) & 0x0F0F0F0Fu);
        int a_lo = aq4[k];
        int a_hi = aq4[4 + k];
        sumi = dp4a(lo, a_lo, sumi);
        sumi = dp4a(hi, a_hi, sumi);
        sums = dp4a(0x01010101, a_lo, sums);
        sums = dp4a(0x01010101, a_hi, sums);
    }
    return d4 * (float)(sumi - 8 * sums) * d8;
}

// qtype wrapper: IQ4_XS and Q4_0 take the wide-load bodies; every other qtype = expert_dot_g
// verbatim.
__device__ __forceinline__ float expert_dot_g_v(int qtype, const unsigned char* wrow, int g,
                                                const signed char* aqb, float d8, int nsb) {
    if (qtype == QT_IQ4_XS) return expert_dot_iq4xs_g_v(wrow, g, aqb, d8);
    if (qtype == QT_Q4_0)   return expert_dot_q4_0_g_v(wrow, g, aqb, d8);
    return expert_dot_g(qtype, wrow, g, aqb, d8, nsb);
}

// ---- DECODE-ONCE weight-group extractors (the MMQ tile-decode, split from the dp4a) ----
// The em/dot bodies above re-dequant the weight group on every (group,token) call; the compiler
// can't hoist that across an unrolled token loop (proven NEUTRAL, rung 2). These split the WEIGHT
// decode from the activation dp4a: decode fills wq[8] (32 int8 weight quants packed as 8 int32,
// EXACTLY the values dp4a'd inside expert_dot_*) + a per-group (fscale, iscale). The reuse kernel
// then dp4a's each pre-decoded group against MANY tokens. FP-ORDER: contrib is computed as
// `fscale * (float)(iscale * sumi) * d8` — byte-identical to expert_dot_iq3s_g (iscale=1 =>
// fscale=db) and expert_dot_iq4xs_g (iscale=scale => fscale=d_sb). Per-group accumulate order is
// unchanged, so MEMRA_MOE_GATE byte-identity holds vs the pair-major/sequential paths.
__device__ __forceinline__ void expert_decode_iq3s_g(const unsigned char* wrow, int g,
                                                     int wq[8], int* iscale, float* fscale) {
    int sblk = g >> 3, ib32 = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    float d = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs    = b + 2  + ib32 * 8;
    unsigned char qh           = b[66 + ib32];
    const unsigned char* signs = b + 74 + ib32 * 4;
    const unsigned char* scales= b + 106;
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    *fscale = d * (1.0f + 2.0f * (float)sc_nib);
    *iscale = 1;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
        int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
        unsigned char sb = signs[l0 / 2];
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        wq[l0 + 0] = __vsub4(gl ^ signs0, signs0);
        wq[l0 + 1] = __vsub4(gh ^ signs1, signs1);
    }
}
__device__ __forceinline__ void expert_decode_iq4xs_g(const unsigned char* wrow, int g,
                                                      int wq[8], int* iscale, float* fscale) {
    int sblk = g >> 3, ib = g & 7;
    const unsigned char* b = wrow + (long)sblk * 136;
    *fscale = half_to_float(*(const unsigned short*)b);
    unsigned short sh = *(const unsigned short*)(b + 2);
    const unsigned char* sl = b + 4;
    const unsigned char* qs = b + 8 + ib * 16;
    int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
    *iscale = ls - 32;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        wq[k]   = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
        wq[k+4] = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
    }
}
// NOTE the int-lane pairing: IQ3_S packs [grid_l,grid_h] interleaved (wq[0..7] = l0=0,0,2,2,4,4,6,6)
// and the activation int order in aq matches (aq4[l0], aq4[l0+1]); IQ4_XS packs [lo x4, hi x4] and
// the activation is aLo[0..3] then aHi[0..3]. So the dp4a token loop must feed activation ints in
// the SAME split for each qtype. We store the activation-int layout choice per qtype via a flag.
// Q4_0 decode-once: fold the -8 offset INTO the int weights ((nib-8) in [-8,7], __vsub4) —
// the group int sum then equals the _em chain's (sumi - 8*sums) EXACTLY (integer identity),
// and the float chain fscale*(iscale*sumi)*d8 == d4*(sumi-8*sums)*d8 bit-for-bit.
__device__ __forceinline__ void expert_decode_q4_0_g(const unsigned char* wrow, int g,
                                                     int wq[8], int* iscale, float* fscale) {
    const unsigned char* b = wrow + (long)g * 18;
    *fscale = half_to_float(*(const unsigned short*)b);
    *iscale = 1;
    const unsigned char* qs = b + 2;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
        wq[k]     = __vsub4((int)(raw & 0x0F0F0F0Fu), 0x08080808);
        wq[4 + k] = __vsub4((int)((raw >> 4) & 0x0F0F0F0Fu), 0x08080808);
    }
}

__device__ __forceinline__ void expert_decode_g(int qtype, const unsigned char* wrow, int g,
                                               int wq[8], int* iscale, float* fscale) {
    if (qtype == QT_IQ3_S)  { expert_decode_iq3s_g(wrow, g, wq, iscale, fscale); return; }
    if (qtype == QT_Q4_0)   { expert_decode_q4_0_g(wrow, g, wq, iscale, fscale); return; }
    expert_decode_iq4xs_g(wrow, g, wq, iscale, fscale);
}
// dp4a a pre-decoded weight group (wq[8]) against one token's 32 activation int8 (aqb) with the
// qtype's int pairing. IQ3_S: sequential aq4[0..7]; IQ4_XS: aLo[0..3]=aqb[0..15], aHi[0..3]=aqb[16..31]
// interleaved as (wq[k]*aLo[k], wq[k+4]*aHi[k]) — matches expert_dot_iq4xs_g's dp4a issue order.
__device__ __forceinline__ int expert_dp4a_group(int qtype, const int wq[8], const signed char* aqb) {
    const int* a = (const int*)aqb;
    int sumi = 0;
    if (qtype == QT_IQ3_S) {
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wq[k], a[k], sumi);
    } else { // IQ4_XS: lo half then hi half
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            sumi = dp4a(wq[k],   a[k],     sumi);
            sumi = dp4a(wq[k+4], a[k + 4], sumi);
        }
    }
    return sumi;
}

// q8_1-activation MoE expert matvec (warp-per-row like mmvq): the staged/sequential expert path
// upgrade — replaces the 256-thread f32-dequant qmatvec_f32 (Stage-A) for IQ3_S/IQ4_XS experts.
// FP-ORDER NOTE: different reduction than qmatvec_f32 (int dp4a + per-group f32 accumulate,
// 32-lane warp tree) — logits SHIFT; argmax/run-gen/stream-identity gates arbitrate, and the
// fused _q8 twins below MUST ship in the same commit (MEMRA_MOE_GATE byte-identity pair contract).
extern "C" __global__ void qmatvec_expert_q8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, int qtype, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g(qtype, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ---- MoE PREFILL PAIR-BATCH kernels (2026-07-06, the 16x pp hole) ----
// ONE launch per (proj, layer) covers ALL (token, expert) routed pairs: grid.y = pair index,
// grid.x tiles the expert-FFN rows (MEMRA_MMVQ_ROWS warps/block, warp-per-row). Per pair the body
// is qmatvec_expert_q8 verbatim (same expert_dot_g order per (pair,row) — bit-identity class).
// Replaces the per-expert loop (256 experts x 3-4 launches x tiny m_e = the 1000+ launch/layer
// prefill wall; llama's fused MoE MMQ analog). Inputs: pair_tok[p] (activation row), pair_ex[p]
// (expert id -> device slab ptr table like _dev), q8_1 activations for ALL T tokens.
extern "C" __global__ void moe_pairs_matvec_q8(
        const unsigned long long* __restrict__ table,   // [3, n_expert] slab base ptrs
        int proj,                                        // 0=gate 1=up 2=down (table row)
        const int* __restrict__ pair_tok,                // [n_pairs]
        const int* __restrict__ pair_ex,                 // [n_pairs]
        const signed char* __restrict__ aq,              // [T, in_f] q8_1 (token-major)
        const float* __restrict__ ad,                    // [T, in_f/32]
        float* __restrict__ y,                           // [n_pairs, out_f]
        int in_f, int out_f, int n_expert, int n_pairs, int qtype, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int pr = blockIdx.y;
    if (o >= out_f || pr >= n_pairs) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int tok = pair_tok[pr];
    int ex  = pair_ex[pr];
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float*       adrow = ad + (size_t)tok * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g(qtype, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)pr * out_f + o] = acc;
}
// silu(gate)*up over the pair-major activation buffers (gate/up both [n_pairs, n_ff]).
// EXPERT-MAJOR variant (rung 2): CSR over experts — block = (expert-segment e, row-tile); the
// warp loads each weight GROUP once into registers and dp4a's it against ALL of expert e's
// tokens (weight reuse across the token group = llama-MMQ's core win; the pair-major kernel
// re-read the weight per pair). Same expert_dot_g per (pair,row) — bit-identical output order.
// ex_off: [n_active+1] CSR into ex_pairs (pair ids grouped by expert); ex_ids: [n_active].
extern "C" __global__ void moe_pairs_matvec_q8_em(
        const unsigned long long* __restrict__ table, int proj,
        const int* __restrict__ ex_ids, const int* __restrict__ ex_off,
        const int* __restrict__ ex_pairs, const int* __restrict__ pair_tok,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y,
        int in_f, int out_f, int n_expert, int n_active, int qtype, long row_bytes) {
    int seg = blockIdx.y;                 // active-expert segment
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (seg >= n_active || o >= out_f) return;
    int lane = threadIdx.x;
    int ex = ex_ids[seg];
    int lo = ex_off[seg], hi = ex_off[seg + 1];
    int nsb = in_f >> 5;
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    // accumulators for up to 16 tokens per pass (register cap); loop passes if more.
    for (int base = lo; base < hi; base += 16) {
        int cnt = min(16, hi - base);
        float acc[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            // weight group decoded ONCE (expert_dot_g re-dequants per call — acceptable: the
            // dp4a-int weight ints stay in L1/registers via the compiler across the token loop
            // when cnt is unrolled; the HBM read happens once per g per row-tile pass).
            #pragma unroll 4
            for (int i = 0; i < cnt; i++) {
                int pr = ex_pairs[base + i];
                int tok = pair_tok[pr];
                acc[i] += expert_dot_g(qtype, wrow, g,
                                       aq + (size_t)tok * in_f + (size_t)g * 32,
                                       ad[(size_t)tok * nsb + g], nsb);
            }
        }
        #pragma unroll
        for (int i = 0; i < cnt; i++) {
            float v = warp_reduce_sum(acc[i]);
            if (lane == 0) y[(size_t)ex_pairs[base + i] * out_f + o] = v;
        }
    }
}

// DECODE-ONCE expert-major MMQ (rung 3): same CSR shape as _em, but the weight group is dequanted
// ONCE per (row, group) via expert_decode_g, then dp4a'd against every token of the expert segment.
// This is the actual MMQ win the _em kernel's comment CLAIMED but did not deliver (expert_dot_g
// re-decoded per token — proven NEUTRAL). Here the decode cost amortizes over the token group.
// FP-ORDER: per-group accumulate `acc[i] += fscale*(float)(iscale*sumi)*d8` in the SAME g-strided
// order as _em/pair-major -> byte-identical logits (MEMRA_MOE_GATE pair contract holds).
extern "C" __global__ void moe_pairs_matvec_q8_dec(
        const unsigned long long* __restrict__ table, int proj,
        const int* __restrict__ ex_ids, const int* __restrict__ ex_off,
        const int* __restrict__ ex_pairs, const int* __restrict__ pair_tok,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y,
        int in_f, int out_f, int n_expert, int n_active, int qtype, long row_bytes) {
    int seg = blockIdx.y;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (seg >= n_active || o >= out_f) return;
    int lane = threadIdx.x;
    int ex = ex_ids[seg];
    int lo = ex_off[seg], hi = ex_off[seg + 1];
    int nsb = in_f >> 5;
    const unsigned char* wrow = (const unsigned char*)table[(size_t)proj * n_expert + ex]
                                + (long)o * row_bytes;
    for (int base = lo; base < hi; base += 32) {
        int cnt = min(32, hi - base);
        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) acc[i] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            int wq[8]; int iscale; float fscale;
            expert_decode_g(qtype, wrow, g, wq, &iscale, &fscale);  // ONCE per (row, group)
            #pragma unroll 4
            for (int i = 0; i < cnt; i++) {
                int tok = pair_tok[ex_pairs[base + i]];
                int sumi = expert_dp4a_group(qtype, wq, aq + (size_t)tok * in_f + (size_t)g * 32);
                acc[i] += fscale * (float)(iscale * sumi) * ad[(size_t)tok * nsb + g];
            }
        }
        #pragma unroll
        for (int i = 0; i < cnt; i++) {
            float v = warp_reduce_sum(acc[i]);
            if (lane == 0) y[(size_t)ex_pairs[base + i] * out_f + o] = v;
        }
    }
}

extern "C" __global__ void moe_pairs_silu_mul(
        const float* __restrict__ gate, const float* __restrict__ up,
        float* __restrict__ act, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; act[i] = (g / (1.0f + expf(-g))) * up[i]; }
}
// gemma4 GELU twin of moe_pairs_silu_mul (gelu_tanh_mul_f32 expression).
extern "C" __global__ void moe_pairs_gelu_mul(
        const float* __restrict__ gate, const float* __restrict__ up,
        float* __restrict__ act, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[i] = 0.5f * x * (1.0f + th) * up[i];
    }
}
// scatter: moe_out[tok] += w[pr] * y_down[pr] — slot-ORDERED per token for bit-identity with the
// sequential axpy chain: one block per (token, col-tile); walks the token's pairs in SLOT order
// via the per-token pair list (tok_pairs CSR built on host: for each token its n_used pair ids
// in slot order).
extern "C" __global__ void moe_pairs_scatter(
        const float* __restrict__ y_down,               // [n_pairs, n_embd]
        const float* __restrict__ pair_w,               // [n_pairs]
        const int* __restrict__ tok_pair_off,            // [T+1] CSR offsets
        const int* __restrict__ tok_pair_ids,            // [n_pairs] pair ids, slot-ordered per token
        float* __restrict__ moe_out,                     // [T, n_embd]
        int n_embd) {
    int tok = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) return;
    int lo = tok_pair_off[tok], hi = tok_pair_off[tok + 1];
    float acc = 0.0f;
    for (int i = lo; i < hi; i++) {
        int pr = tok_pair_ids[i];
        acc = __fmaf_rn(pair_w[pr], y_down[(size_t)pr * n_embd + c], acc);
    }
    moe_out[(size_t)tok * n_embd + c] = acc;
}

extern "C" __global__ void qmatvec_iq4_XS_dp4a(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        const signed char* aqb = arow + (size_t)g * 32;
        // Same group value and lo/hi dp4a order as the scalar body, with aligned
        // 64-bit header/quant loads and byte_perm lookup; exotic bases fall back.
        acc += expert_dot_iq4xs_g_v(wrow, g, aqb, adrow[g]);
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// y[m,out] = x[m,in] @ W[out,in]^T. W quantized rows of `row_bytes` each.
// grid: (out, m); block: 256 threads reduce over `in`.
extern "C" __global__ void qmatvec_f32(
        const uint8_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
        int in_f, int out_f, int m, int qtype, long row_bytes) {
    int o = blockIdx.x;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    const uint8_t* wrow = W + (long)o * row_bytes;
    const float* xrow = x + (long)t * in_f;
    float acc = 0.0f;
    if (qtype == QT_NVFP4_RP) {
        // split-plane NVFP4: same per-element value/product order as deq_nvfp4 -> bit-identical.
        int nsb64 = in_f >> 6;
        const uint8_t* qrow = W + (size_t)o * nsb64 * 32;
        const uint8_t* srow = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
        for (int i = tid; i < in_f; i += blockDim.x) {
            int blk = i >> 6, jj = i & 63;
            int s = jj >> 4, within = jj & 15;
            int byte = qrow[blk * 32 + s * 8 + (within & 7)];
            int code = (within < 8) ? (byte & 0xF) : (byte >> 4);
            acc += (float)kvalues_mxfp4_d[code] * ue4m3_to_f32_d(srow[blk * 4 + s]) * xrow[i];
        }
    } else
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qtype, wrow, i) * xrow[i];
    // block reduce
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(long)t * out_f + o] = v;
    }
}

// ================================================================================================
// STAGE-2 GROUPED DECODE (2026-07-04): single-launch 8-expert MoE matvecs for m=1.
//
// The sequential decode path launches 8 experts x (gate,up,silu,down,axpy) = 40 kernels per MoE
// layer per token, each a tiny m=1 matvec (~5 us) — 2533 launches/token total on the 35B, host
// launch time ~7.9 ms/tok vs 11.7 ms/tok GPU time (nsys 2026-07-04). These two kernels fold one
// layer's routed-expert FFN into TWO launches via expert-pointer indirection (the SLRU cache slots
// are fixed-address, so the 8 weight pointers are stable for the whole launch).
//
// BIT-IDENTITY CONTRACT (vs the sequential qmatvec_f32 + silu_mul_f32 + axpy_f32 chain):
//  - each dot reproduces qmatvec_f32's EXACT reduction: same 256-thread striding over in_f, same
//    warp shuffle tree, same s[32] two-level reduce. Identical partial-sum order => identical f32.
//  - the SiLU epilogue is silu_mul_f32's expression on the SAME dot values (f32 store/load of the
//    intermediates is exact, so register-passing them is bit-identical).
//  - the down epilogue reproduces the 8 sequential axpy_f32 accumulations: acc starts 0.0 (the
//    e.zeros moe_out) and chains __fmaf_rn(w[j], y_j, acc) in slot order j=0..7 — the same FMA
//    axpy_f32 compiles to (the A2 slot-scheme argument, byte-identity-gated there).
// ================================================================================================

typedef struct { const unsigned char* p[8]; } wptr8_t;
typedef struct { float v[8]; } f32x8_t;

// One MoE layer's gate+up+SiLU for all 8 routed experts of ONE token in ONE launch.
// act[j*n_ff + o] = silu(gate_j[o] . x) * (up_j[o] . x). grid: (n_ff, n_used); block: 256.
extern "C" __global__ void moe_gate_up_silu8_f32(
        wptr8_t gp, wptr8_t up, const float* __restrict__ x, float* __restrict__ act,
        int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row 0..n_ff-1
    int j = blockIdx.y;              // routed-expert slot 0..n_used-1
    int tid = threadIdx.x;
    __shared__ float s[32];
    __shared__ float g_final;
    // ---- gate dot: EXACT qmatvec_f32 structure ----
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    float acc = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt_g, grow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) g_final = v;
    }
    __syncthreads();                 // s + g_final ready; s reused below
    // ---- up dot: same structure ----
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float acc2 = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc2 += deq(qt_u, urow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc2;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) {
            float g = g_final;
            // silu_mul_f32's exact expression on the exact dot values.
            act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * v;
        }
    }
}

// ---- dp4a _q8 TWINS of the fused MoE kernels (matched pair with qmatvec_expert_q8) ----
// Same grid/block/slot-order/silu expression as the _f32 versions; ONLY the dot changes:
// warp-per-row int dp4a vs the q8_1 activation (aq/ad), block=(32, ROWS) covering n_ff rows
// like the f32 version's grid. Reduction = 32-lane warp tree per row (matches expert_q8).
extern "C" __global__ void moe_gate_up_silu8_q8(
        wptr8_t gp, wptr8_t up, const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act, int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row
    int j = blockIdx.y;              // routed slot
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,j)
    int nsb = in_f >> 5;
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg;
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * accu;
    }
}

// glm5_next PRE-CLAMPED, MACRO-FOLDING twin of moe_gate_up_silu8_q8 (lane/glm53-epilogue,
// 2026-08-28). Dots, grid/block, warp reduction and slot order are the sibling's VERBATIM; two
// things change, and both are semantic requirements of this arch rather than optimizations:
//
//   1. EPILOGUE. glm5_next is ActivationPlan::SwiGluPreClamped, not plain silu(gate)*up. The
//      expression below is swiglu_preclamped_mul_scaled_f32's (hybrid.cu:1061) character for
//      character on the exact dot values:
//          u = clamp(up*us, +-limit);  x = min(gate*gs, limit);  act = silu(x) * u
//      The gate clamp lands BEFORE silu and is ONE-sided. step35's POST form
//      (min(silu(gate*gs), limit) * u) compiles here just as happily and returns
//      plausible-but-wrong logits; the two diverge by up to limit*(1-sigmoid(limit)) per element
//      wherever gate*gs > limit. Gated by glm5_moe_epilogue_gpu.rs's `post-for-pre-clamp` arm.
//   2. PER-SLOT MACRO SCALES. The compressed-tensors NVFP4 expert class carries a per-expert
//      weight_scale_2 that is NOT in the block bytes; the sequential loop folds it through
//      ffn_act_lim's gs/us. gs/us here are the SELECTED experts' scales in router slot order
//      (host-gathered, so no macros[] indirection), 1.0 for a macro-free bank. Dropping them is
//      the ~3e4x fluent-garbage class measured 2026-07-16.
//
// The DOWN projection needs no twin: its macro folds into the routing weight the caller already
// passes to moe_down8_fma_q8 (w[j] * macro_down[sel[j]]), which is exactly what the sequential
// loop's axpy_into does.
extern "C" __global__ void moe_gate_up_preclamp8_q8(
        wptr8_t gp, wptr8_t up, const signed char* __restrict__ aq, const float* __restrict__ ad,
        f32x8_t gs, f32x8_t us, float limit,
        float* __restrict__ act, int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row
    int j = blockIdx.y;              // routed slot
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,j)
    int nsb = in_f >> 5;
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * us.v[j], limit), -limit);
        float x = fminf(accg * gs.v[j], limit);
        act[(size_t)j * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
// act is quantized per-slot by the caller (aq2/ad2 hold n_used rows of q8_1).
extern "C" __global__ void moe_down8_fma_q8(
        wptr8_t dp, f32x8_t w, const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(w.v[j], acc, chain);
    }
    if (lane == 0) dst[o] = chain;
}

// ---- WARP-PACKED twins of the plain-decode glm5_next epilogue pair (lane/b200-matvec-occupancy,
// MEMRA_B200_MATVEC_ARM, 2026-09-02) ----
//
// moe_gate_up_preclamp8_q8 / moe_down8_fma_q8 launch block=(32,1,1) — ONE warp per block, same
// occupancy ceiling the vrest _rows pair hit before its own _w4 twins above: on a card whose
// per-SM resident-block limit is below its resident-warp limit, a 1-warp block caps occupancy at
// (block limit)/(warp limit) instead of 100%, and every wave pays full memory latency instead of
// overlapping it against other resident warps. This is the plain-decode pair the B200 census
// (2026-09-02, GLM-5.3-Flash NVFP4, PP2) measured at 20.0% + 10.5% of GPU time, ~9x its roofline
// byte estimate for gate+up (~6us of NVFP4 traffic vs 54.6us measured) — an occupancy/latency
// signature, not a bandwidth one, on the 148-SM/8-TB/s B200 vs the 188-SM/1.8-TB/s RTX PRO 6000
// these kernels were tuned for.
//
// These twins pack MEMRA_MMVQ_ROWS = 4 warps per block on threadIdx.y — the same standing shape
// as the _rows_w4 pair above — with the per-warp body VERBATIM: identical expert_dot_g g-strided
// chain, identical warp_reduce_sum, identical swiglu_preclamped_mul_scaled_f32 epilogue /
// slot-ordered __fmaf_rn down chain. Packing only changes which block/warp computes a given
// (o,j) or o output; it moves no bits. Selected only behind MEMRA_B200_MATVEC_ARM=1 (default
// OFF; door pending its B200 A/B, docs/FLAGS.md).
extern "C" __global__ void moe_gate_up_preclamp8_q8_w4(
        wptr8_t gp, wptr8_t up, const signed char* __restrict__ aq, const float* __restrict__ ad,
        f32x8_t gs, f32x8_t us, float limit,
        float* __restrict__ act, int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;  // expert-FFN row (packed)
    int j = blockIdx.y;              // routed slot
    if (o >= n_ff) return;
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,j)
    int nsb = in_f >> 5;
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * us.v[j], limit), -limit);
        float x = fminf(accg * gs.v[j], limit);
        act[(size_t)j * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
extern "C" __global__ void moe_down8_fma_q8_w4(
        wptr8_t dp, f32x8_t w, const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(w.v[j], acc, chain);
    }
    if (lane == 0) dst[o] = chain;
}

// ---- VERIFY-ROWS twins of the fused glm5_next epilogue pair (lane/glm5-vrest, 2026-08-31) ----
// ONE launch pair covers ALL t x n_used routed pairs of a spec-verify batch (the pair union
// across the K+1 rows), replacing the per-(token,expert) sequential loop's 49 launches per
// token-layer. Pair p = tok*n_used + j is DENSE slot-major, so per-token slot order is
// structural. Per pair the body is moe_gate_up_preclamp8_q8 / moe_down8_fma_q8 VERBATIM —
// same expert_dot_g g-strided order per (pair,row) (== qmatvec_expert_q8's chain), same warp
// tree, same swiglu_preclamped_mul_scaled_f32 expression on the exact dot values, same
// slot-ordered __fmaf_rn down chain (== the sequential axpy chain) — only the
// pair -> (token activation row, expert pointer, macro pair) indirection is new, and
// indirection moves no bits. Bit-gated per row vs the sequential chain
// (glm5_verify_batch_gpu, swapped-pair + dropped-macro reds).
//
// ptrs: [3*n_pairs] u64 expert ROW-0 addresses, plane-major (gate | up | down), host-built
// from the resident slab base + ex*stride — the sequential slab arm's exact pointer
// arithmetic. scl: [3*n_pairs] f32 plane-major (gs | us | w*macro_down): gate/up macros ride
// the epilogue exactly where ffn_act_lim's gs/us ride the unfused loop; the down macro folds
// into the routing weight exactly where axpy_into folds it.
extern "C" __global__ void moe_gate_up_preclamp8_q8_rows(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs,
        int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;              // expert-FFN row
    int pr = blockIdx.y;             // (token, slot) pair
    if (o >= n_ff || pr >= n_pairs) return;
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,pr)
    int nsb = in_f >> 5;
    int tok = pr / n_used;
    const unsigned char* grow = (const unsigned char*)ptrs[pr] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)ptrs[n_pairs + pr] + (long)o * rb_u;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float* adrow = ad + (size_t)tok * nsb;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = arow + (size_t)g * 32;
        float d8 = adrow[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * scl[n_pairs + pr], limit), -limit);
        float x = fminf(accg * scl[pr], limit);
        act[(size_t)pr * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
// Down + slot-ordered weighted accumulation for EVERY verify row in one launch. Per (token,
// out-row) the body is moe_down8_fma_q8 verbatim: the j=0..n_used-1 __fmaf_rn chain over that
// token's pairs, full overwrite of dst[tok]. aq2/ad2 hold the [n_pairs, in_f] pair-major q8_1
// activations; ptrs/scl are the launch-pair tables above (down plane = [2*n_pairs..)).
extern "C" __global__ void moe_down8_fma_q8_rows(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb) {
    int o = blockIdx.x;
    int tok = blockIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int pr = tok * n_used + j;
        const unsigned char* wrow = (const unsigned char*)ptrs[2 * n_pairs + pr] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)pr * in_f;
        const float* adrow = ad2 + (size_t)pr * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(scl[2 * n_pairs + pr], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}

// ---- WARP-PACKED twins of the verify-rows MoE pair (lane/glm5-matvec, MEMRA_MOE_VROWS_PACK).
// The _rows pair launches block=(32,1,1) — ONE warp per block, so the resident-warp count is
// capped by the blocks/SM limit (<=32 of 48 warp slots, <=67% occupancy) and every launch
// schedules ~65k one-warp blocks (diet-battery c8-ship census: the pair is 26% of decode-round
// GPU at the 57-64%-of-bound class the decode-gap attribution measured). These twins pack
// MEMRA_MMVQ_ROWS = 4 warps per block on threadIdx.y — the qmatvec mmvq family's standing
// shape — with the per-warp body VERBATIM (same expert_dot_g g-strided chain, same
// warp_reduce_sum, same epilogue / slot-ordered __fmaf_rn chain; neither kernel has a
// __syncthreads, so the early return on a ragged row tail is safe). Packing moves no bits:
// output (o, pair) is computed by exactly one warp running exactly the _rows program.
// Bonus locality: the 4 warps of a block share one pair's activation row and read 4 ADJACENT
// expert rows (contiguous bank bytes). Bit-gated vs the unpacked pair (glm5_matvec_doors_gpu).
extern "C" __global__ void moe_gate_up_preclamp8_q8_rows_w4(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs,
        int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;  // expert-FFN row (packed)
    int pr = blockIdx.y;                                 // (token, slot) pair
    if (o >= n_ff || pr >= n_pairs) return;
    int lane = threadIdx.x;                              // 32 lanes, one warp per (o,pr)
    int nsb = in_f >> 5;
    int tok = pr / n_used;
    const unsigned char* grow = (const unsigned char*)ptrs[pr] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)ptrs[n_pairs + pr] + (long)o * rb_u;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float* adrow = ad + (size_t)tok * nsb;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = arow + (size_t)g * 32;
        float d8 = adrow[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * scl[n_pairs + pr], limit), -limit);
        float x = fminf(accg * scl[pr], limit);
        act[(size_t)pr * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
extern "C" __global__ void moe_down8_fma_q8_rows_w4(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;  // output row (packed)
    int tok = blockIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int pr = tok * n_used + j;
        const unsigned char* wrow = (const unsigned char*)ptrs[2 * n_pairs + pr] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)pr * in_f;
        const float* adrow = ad2 + (size_t)pr * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(scl[2 * n_pairs + pr], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}

// ---- ILP twins of the verify-rows MoE pair (lane/glm5-moe-rows-ilp-20260904, door
// MEMRA_MOE_VROWS_ILP, default OFF) ----
//
// WHY. The pair is the largest kernel-time item of a plain glm5_next t=1 token on the 2x B200
// pair (door-ON census 2026-09-04: moe_gate_up_preclamp8_q8_rows 43/token x 47.2 us +
// moe_down8_fma_q8_rows 43/token x 25.7 us = 3.1 ms of a ~13.7 ms token, 22%), and it moves
// ~71 MB + ~36 MB of expert bytes per launch pair: ~1.5 TB/s useful on an 8 TB/s part. Root
// ncu on the rig at the served geometry (darklanes research/glm5-b200-20260902/ncu-rig/moe.csv)
// says WHY: long-scoreboard stalls 55-60% of warp-active cycles, issue-active 36-44%, DRAM at
// 25-46% of peak; the 4-warp packing (_w4) lifted occupancy 47 -> 76% and DRAM% by a quarter,
// and priced +0.6..0.9% on the pair. So it is neither occupancy nor arithmetic: every lane
// walks its groups SERIALLY (g = lane, lane+32, ...; nsb = 128 at in_f 4096 is four rounds
// for gate/up, two for down at in_f 2048), and each round's six weight loads (four quant ints
// + two scale bytes per plane) wait on the previous round's dependent chain.
//
// WHAT. The same per-warp program with the loads of four (then two) groups per lane issued
// BEFORE any of their math: `nvfp4_v1_load_g` reads a group's exact bytes into registers
// (the ints get_int_b4 read at the half-block's +0/+4/+8/+12, the scale bytes d_bytes[s0] and
// [s0+1]) and `expert_dot_nvfp4_core_regs` is the pinned core on those registers. The
// accumulation ORDER per plane is unchanged: accg += dot(g), accg += dot(g+32), ... exactly
// the serial loop's sequence into the same accumulator; accg and accu are separate
// accumulators so their interleaving moves no bits; the warp tree, the pre-clamp epilogue and
// the slot-ordered __fmaf_rn down chain are verbatim. ONLY the interleaved NVFP4 layout
// (QT_NVFP4): the host launcher refuses by name for any other qtype, and the kernel poisons
// its output (NaN) if it is ever reached with another, so a wiring error screams.
// Gate: tests/glm5_matvec_doors_gpu.rs (bitwise vs the shipped pair at the served nsb
// classes, red arms through the door).
struct nvfp4_grp_regs { int q0, q1, q2, q3; unsigned char d0, d1; };
__device__ __forceinline__ nvfp4_grp_regs nvfp4_v1_load_g(const unsigned char* wrow, int g) {
    int sblk = g >> 1;
    int s0 = (g & 1) * 2;
    const unsigned char* b = wrow + (long)sblk * 36;
    const unsigned char* qh = b + 4 + s0 * 8;
    nvfp4_grp_regs r;
    r.q0 = get_int_b4(qh);
    r.q1 = get_int_b4(qh + 4);
    r.q2 = get_int_b4(qh + 8);
    r.q3 = get_int_b4(qh + 12);
    r.d0 = b[s0];
    r.d1 = b[s0 + 1];
    return r;
}
__device__ __forceinline__ float nvfp4_regs_dot(const nvfp4_grp_regs& r, const int* aq4, float d8) {
    return expert_dot_nvfp4_core_regs(r.q0, r.q1, r.q2, r.q3, r.d0, r.d1, aq4, d8);
}
// gate/up per-warp body: two planes, loads hoisted 4 groups deep, then 2, then singles.
__device__ __forceinline__ void moe_gate_up_nvfp4_rows_ilp_body(
        const unsigned char* __restrict__ grow, const unsigned char* __restrict__ urow,
        const signed char* __restrict__ arow, const float* __restrict__ adrow, int nsb, int lane,
        float& accg, float& accu) {
    int g = lane;
    for (; g + 96 < nsb; g += 128) {
        nvfp4_grp_regs g0 = nvfp4_v1_load_g(grow, g);
        nvfp4_grp_regs g1 = nvfp4_v1_load_g(grow, g + 32);
        nvfp4_grp_regs g2 = nvfp4_v1_load_g(grow, g + 64);
        nvfp4_grp_regs g3 = nvfp4_v1_load_g(grow, g + 96);
        nvfp4_grp_regs u0 = nvfp4_v1_load_g(urow, g);
        nvfp4_grp_regs u1 = nvfp4_v1_load_g(urow, g + 32);
        nvfp4_grp_regs u2 = nvfp4_v1_load_g(urow, g + 64);
        nvfp4_grp_regs u3 = nvfp4_v1_load_g(urow, g + 96);
        const int* a0 = (const int*)(arow + (size_t)g * 32);
        const int* a1 = (const int*)(arow + (size_t)(g + 32) * 32);
        const int* a2 = (const int*)(arow + (size_t)(g + 64) * 32);
        const int* a3 = (const int*)(arow + (size_t)(g + 96) * 32);
        float d80 = adrow[g], d81 = adrow[g + 32], d82 = adrow[g + 64], d83 = adrow[g + 96];
        accg += nvfp4_regs_dot(g0, a0, d80);
        accu += nvfp4_regs_dot(u0, a0, d80);
        accg += nvfp4_regs_dot(g1, a1, d81);
        accu += nvfp4_regs_dot(u1, a1, d81);
        accg += nvfp4_regs_dot(g2, a2, d82);
        accu += nvfp4_regs_dot(u2, a2, d82);
        accg += nvfp4_regs_dot(g3, a3, d83);
        accu += nvfp4_regs_dot(u3, a3, d83);
    }
    for (; g + 32 < nsb; g += 64) {
        nvfp4_grp_regs g0 = nvfp4_v1_load_g(grow, g);
        nvfp4_grp_regs g1 = nvfp4_v1_load_g(grow, g + 32);
        nvfp4_grp_regs u0 = nvfp4_v1_load_g(urow, g);
        nvfp4_grp_regs u1 = nvfp4_v1_load_g(urow, g + 32);
        const int* a0 = (const int*)(arow + (size_t)g * 32);
        const int* a1 = (const int*)(arow + (size_t)(g + 32) * 32);
        float d80 = adrow[g], d81 = adrow[g + 32];
        accg += nvfp4_regs_dot(g0, a0, d80);
        accu += nvfp4_regs_dot(u0, a0, d80);
        accg += nvfp4_regs_dot(g1, a1, d81);
        accu += nvfp4_regs_dot(u1, a1, d81);
    }
    for (; g < nsb; g += 32) {
        const signed char* aqb = arow + (size_t)g * 32;
        float d8 = adrow[g];
        accg += expert_dot_nvfp4_g(grow, g, aqb, d8);
        accu += expert_dot_nvfp4_g(urow, g, aqb, d8);
    }
}
// down per-warp body: one plane, same hoisting; returns the un-reduced lane partial.
__device__ __forceinline__ float moe_down_nvfp4_rows_ilp_body(
        const unsigned char* __restrict__ wrow, const signed char* __restrict__ arow,
        const float* __restrict__ adrow, int nsb, int lane) {
    float acc = 0.0f;
    int g = lane;
    for (; g + 96 < nsb; g += 128) {
        nvfp4_grp_regs w0 = nvfp4_v1_load_g(wrow, g);
        nvfp4_grp_regs w1 = nvfp4_v1_load_g(wrow, g + 32);
        nvfp4_grp_regs w2 = nvfp4_v1_load_g(wrow, g + 64);
        nvfp4_grp_regs w3 = nvfp4_v1_load_g(wrow, g + 96);
        acc += nvfp4_regs_dot(w0, (const int*)(arow + (size_t)g * 32), adrow[g]);
        acc += nvfp4_regs_dot(w1, (const int*)(arow + (size_t)(g + 32) * 32), adrow[g + 32]);
        acc += nvfp4_regs_dot(w2, (const int*)(arow + (size_t)(g + 64) * 32), adrow[g + 64]);
        acc += nvfp4_regs_dot(w3, (const int*)(arow + (size_t)(g + 96) * 32), adrow[g + 96]);
    }
    for (; g + 32 < nsb; g += 64) {
        nvfp4_grp_regs w0 = nvfp4_v1_load_g(wrow, g);
        nvfp4_grp_regs w1 = nvfp4_v1_load_g(wrow, g + 32);
        acc += nvfp4_regs_dot(w0, (const int*)(arow + (size_t)g * 32), adrow[g]);
        acc += nvfp4_regs_dot(w1, (const int*)(arow + (size_t)(g + 32) * 32), adrow[g + 32]);
    }
    for (; g < nsb; g += 32)
        acc += expert_dot_nvfp4_g(wrow, g, arow + (size_t)g * 32, adrow[g]);
    return acc;
}
__device__ __forceinline__ void moe_gate_up_rows_ilp_warp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs, int qt_g,
        int qt_u, long rb_g, long rb_u, int o, int pr, int lane) {
    int nsb = in_f >> 5;
    int tok = pr / n_used;
    const unsigned char* grow = (const unsigned char*)ptrs[pr] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)ptrs[n_pairs + pr] + (long)o * rb_u;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float* adrow = ad + (size_t)tok * nsb;
    float accg = 0.0f, accu = 0.0f;
    if (qt_g == QT_NVFP4 && qt_u == QT_NVFP4) {
        moe_gate_up_nvfp4_rows_ilp_body(grow, urow, arow, adrow, nsb, lane, accg, accu);
    } else {
        accg = accu = __int_as_float(0x7fc00000);  // wiring error: poison, never a silent zero
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * scl[n_pairs + pr], limit), -limit);
        float x = fminf(accg * scl[pr], limit);
        act[(size_t)pr * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
extern "C" __global__ void moe_gate_up_preclamp8_q8_rows_ilp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs,
        int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int pr = blockIdx.y;
    if (o >= n_ff || pr >= n_pairs) return;
    moe_gate_up_rows_ilp_warp(ptrs, scl, aq, ad, limit, act, in_f, n_ff, n_used, n_pairs, qt_g,
                              qt_u, rb_g, rb_u, o, pr, threadIdx.x);
}
extern "C" __global__ void moe_gate_up_preclamp8_q8_rows_w4_ilp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs,
        int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int pr = blockIdx.y;
    if (o >= n_ff || pr >= n_pairs) return;
    moe_gate_up_rows_ilp_warp(ptrs, scl, aq, ad, limit, act, in_f, n_ff, n_used, n_pairs, qt_g,
                              qt_u, rb_g, rb_u, o, pr, threadIdx.x);
}
__device__ __forceinline__ void moe_down_rows_ilp_warp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb,
        int o, int tok, int lane) {
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int pr = tok * n_used + j;
        const unsigned char* wrow = (const unsigned char*)ptrs[2 * n_pairs + pr] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)pr * in_f;
        const float* adrow = ad2 + (size_t)pr * nsb;
        float acc = (qt == QT_NVFP4) ? moe_down_nvfp4_rows_ilp_body(wrow, arow, adrow, nsb, lane)
                                     : __int_as_float(0x7fc00000);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(scl[2 * n_pairs + pr], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}
extern "C" __global__ void moe_down8_fma_q8_rows_ilp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb) {
    int o = blockIdx.x;
    int tok = blockIdx.y;
    if (o >= out_f) return;
    moe_down_rows_ilp_warp(ptrs, scl, aq2, ad2, dst, in_f, out_f, n_used, n_pairs, qt, rb, o, tok,
                           threadIdx.x);
}
extern "C" __global__ void moe_down8_fma_q8_rows_w4_ilp(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int tok = blockIdx.y;
    if (o >= out_f) return;
    moe_down_rows_ilp_warp(ptrs, scl, aq2, ad2, dst, in_f, out_f, n_used, n_pairs, qt, rb, o, tok,
                           threadIdx.x);
}

// ---- DEDUP-SCHEDULE twins of the verify-rows MoE pair (lane/glm5-dedup, 2026-08-31) ----
//
// WHY: the struct-battery instrument measured **21.96% cumulative repeat fraction** across the
// pair's expert visits (2.55M visits / 99,751 layer-calls, mode-stable 22.27% greedy / 21.53%
// vendor-sampled — struct-battery WINDOW.md cell 2), i.e. 6.9x the 3.21% independent-routing
// bound: about a fifth of the (row, expert) visits re-read a slab another verify row in the SAME
// layer-call already read. The pair runs at 90.2% / 89.9% of this card class's theoretical DRAM
// peak (moe-loc LANE.md §1.3) so there is no efficiency left to win — the ONLY remaining lever is
// reading less, and a repeat read is only avoided if it is SCHEDULED inside the reuse window.
//
// The mechanism is a pure VISIT-ORDER change: which block computes which (row, pair) output.
//
//   * `_ord` (gate/up): the grid is TRANSPOSED so the pair index is the FASTEST dimension
//     (grid = (n_pairs, n_ff) instead of (n_ff, n_pairs)), and the pair walked by block
//     `blockIdx.x` is read from an EXPERT-MAJOR order plane (`ptrs[3*n_pairs + q]`, built by
//     `moe_vrows_order_from_sel` on device / the host arm's stable sort). Consequence: the whole
//     pair union for ONE expert-FFN row `o` is co-resident, and two pairs sharing an expert are
//     ADJACENT blocks reading the IDENTICAL gate row and up row. The reuse distance collapses
//     from "one 9.44 MB slab pass must survive in L2" (the shipped o-fastest schedule: 2048
//     blocks) to ~1 block over ~2 x 2.3 KB — an L1-class hit instead of an L2 gamble. The
//     charter's slab-residency argument (4.72 MB slab vs ~128 MB L2) is the FALLBACK argument
//     here, not the operative one.
//   * `_tmaj` (down): same idea where the accumulation forbids a permutation. The down kernel's
//     block owns one (token, out-row) and MUST walk j = 0..n_used-1 in SLOT order (that
//     `__fmaf_rn` chain is the vrest gate-4 bit bar), so the pair order inside a block is
//     untouchable. Only the GRID is transposed — token fastest, grid = (t, out_f) — so the t
//     blocks at the same out-row `o` are adjacent and a repeated expert's down row (1152 B at
//     the serving shape) is read once for all the tokens that share it.
//
// BIT IDENTITY, and why it is structural rather than measured: in both kernels every output is a
// pure function of its (o, pr) / (o, tok) coordinate — `ptrs[pr]`, `scl[pr]`, `tok = pr / n_used`,
// `act[pr*n_ff + o]`, `dst[tok*out_f + o]` — and the bodies below are their twins' character for
// character (same `expert_dot_g` g-strided chain, same `warp_reduce_sum`, same
// `swiglu_preclamped_mul_scaled_f32` expression, same slot-ordered `__fmaf_rn` chain). Neither
// kernel has a `__syncthreads`, shared memory, or any cross-block communication, so RE-INDEXING
// WHICH BLOCK COMPUTES WHICH OUTPUT MOVES NO BITS. The order plane is a permutation, so every
// pair is still computed exactly once. Gated in `glm5_dedup_sched_gpu`: bit identity vs the
// shipped pair at t=2..8 x {live macros, none} x both table provenances, a valid-shuffle arm that
// must stay bit-INERT, and a non-permutation order plane that must BITE.
//
// The win, by contrast, is a SCHEDULING property (CUDA dispatches blocks in increasing linear id,
// x fastest — the ordering every locality door in this family already rides), so it is unpriceable
// on an exactness-only rig: both doors ship default OFF and the box prices the wall.
extern "C" __global__ void moe_gate_up_preclamp8_q8_rows_ord(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float limit,
        float* __restrict__ act, int in_f, int n_ff, int n_used, int n_pairs,
        int qt_g, int qt_u, long rb_g, long rb_u) {
    // Guard BEFORE the order-plane load: the plane is [3*n_pairs .. 4*n_pairs).
    if (blockIdx.x >= (unsigned)n_pairs || blockIdx.y >= (unsigned)n_ff) return;
    int pr = (int)ptrs[3 * n_pairs + blockIdx.x];   // expert-major visit order
    int o = blockIdx.y;                             // expert-FFN row (now the SLOW dimension)
    int lane = threadIdx.x;          // 32 lanes, one warp per (o,pr)
    int nsb = in_f >> 5;
    int tok = pr / n_used;
    const unsigned char* grow = (const unsigned char*)ptrs[pr] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)ptrs[n_pairs + pr] + (long)o * rb_u;
    const signed char* arow = aq + (size_t)tok * in_f;
    const float* adrow = ad + (size_t)tok * nsb;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = arow + (size_t)g * 32;
        float d8 = adrow[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * scl[n_pairs + pr], limit), -limit);
        float x = fminf(accg * scl[pr], limit);
        act[(size_t)pr * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}
// TOKEN-MAJOR grid twin of moe_down8_fma_q8_rows: grid = (t, out_f), token fastest. The j loop —
// the slot-ordered __fmaf_rn chain that IS the accumulation order — is verbatim and keeps its
// ORIGINAL slot order; only which block runs when changes.
extern "C" __global__ void moe_down8_fma_q8_rows_tmaj(
        const unsigned long long* __restrict__ ptrs, const float* __restrict__ scl,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int n_pairs, int qt, long rb) {
    int tok = blockIdx.x;            // verify row (now the FAST dimension)
    int o = blockIdx.y;              // output row
    if (o >= out_f) return;          // grid.x == t exactly, as the shipped twin's grid.y was
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int pr = tok * n_used + j;
        const unsigned char* wrow = (const unsigned char*)ptrs[2 * n_pairs + pr] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)pr * in_f;
        const float* adrow = ad2 + (size_t)pr * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(scl[2 * n_pairs + pr], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}

// EXPERT-MAJOR ORDER PLANE for the `_ord` gate/up twin (lane/glm5-dedup door E). Writes the
// permutation into the pointer table's fourth plane, `ptrs[3*n_pairs .. 4*n_pairs)`, so the door
// costs the device-tables arm ONE extra launch and the host-tables arm NOTHING (the host appends
// the plane to the vector it already uploads in one `htod_u64_into`).
//
// STABLE COUNTING RANK, chosen so the device build is bit-identical to the host's stable sort by
// (expert id, pair index) with no scratch, no scan and no order-of-execution dependence: thread p
// counts how many pairs sort strictly before it and stores itself at that rank. Ties break on the
// pair index, so the permutation is a total order and per-expert runs keep ascending slot order.
// O(n_pairs^2) with n_pairs = t*n_used <= 64 on every serving shape (4096 comparisons total),
// one 128-thread block: cheaper than any scan, and there is no correctness cliff to grow into.
extern "C" __global__ void moe_vrows_order_from_sel(
        const int* __restrict__ sel, unsigned long long* __restrict__ ptrs, int n_pairs) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    int ep = sel[p];
    int rank = 0;
    for (int q = 0; q < n_pairs; q++) {
        int eq = sel[q];
        if (eq < ep || (eq == ep && q < p)) rank++;
    }
    ptrs[3 * n_pairs + rank] = (unsigned long long)p;
}

// DEVICE-SIDE POINTER/SCALE TABLE BUILD for the verify-rows pair (lane/glm5-moe-loc door D,
// MEMRA_MOE_VROWS_DEV_TABLES). The pair's `ptrs`/`scl` tables were built on the HOST, which
// forced the router's selection back across the bus: `moe_router_sigmoid_topk_host` does 2
// DtoH into a pinned stage plus a FULL `cuStreamSynchronize` per MoE layer-call purely so the
// host can evaluate `slab_base + ex*expert_stride` and three `macro_scale(ex)` lookups. That
// is 42 device-wide drains + 84 DtoH + 84 pageable HtoD per ship round (the decode-gap
// attribution's "43 cuStreamSynchronize/token ... the per-layer router-admission sync
// structure", and 44.6% of the unattributed 71.6 HtoD calls/token). This kernel evaluates the
// SAME arithmetic where the selection already lives.
//
// BIT IDENTITY, term by term vs the host loop (hybrid_forward.rs `moe_vrows_pairs_q8`):
//   * ptrs: `base + ex*stride` is exact integer arithmetic on the same base and the same
//     stride -> identical addresses. `sel` here is the router's own device i32 output, which
//     is what the pinned DtoH copied verbatim, so `ex` is the same expert id.
//   * scl gate/up: a table lookup of the same f32 macro plane at the same index -> identical.
//   * scl down: `selw[p] * md` is ONE IEEE-754 single multiply of the same two operands in the
//     same order as the host's `w * m.down_exps.macro_scale(ex)` -> identical bits (both
//     round-to-nearest-even; no FMA contraction is possible in a bare product).
// Absent macro planes (`have_macros == 0`) take 1.0f exactly as `macro_scale` returns 1.0 for a
// non-macro bank. One thread per pair; n_pairs = t*n_used <= 128 on every serving shape.
extern "C" __global__ void moe_vrows_tables_from_sel(
        const int* __restrict__ sel, const float* __restrict__ selw,
        const float* __restrict__ mac_g, const float* __restrict__ mac_u,
        const float* __restrict__ mac_d,
        unsigned long long* __restrict__ ptrs, float* __restrict__ scl,
        unsigned long long pg, unsigned long long pu, unsigned long long pd,
        long sg, long su, long sd, int n_pairs, int have_macros) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    long ex = (long)sel[p];
    ptrs[p] = pg + (unsigned long long)(ex * sg);
    ptrs[n_pairs + p] = pu + (unsigned long long)(ex * su);
    ptrs[2 * n_pairs + p] = pd + (unsigned long long)(ex * sd);
    float mg = have_macros ? mac_g[ex] : 1.0f;
    float mu = have_macros ? mac_u[ex] : 1.0f;
    float md = have_macros ? mac_d[ex] : 1.0f;
    scl[p] = mg;
    scl[n_pairs + p] = mu;
    // down-proj macro folds into the accumulate weight — the axpy_into fold, verbatim.
    scl[2 * n_pairs + p] = selw[p] * md;
}

// One MoE layer's down-proj + weighted accumulation for all 8 routed experts in ONE launch.
// dst[o] = fma(w[7], y_7[o], ... fma(w[0], y_0[o], 0.0f)) where y_j = W_down_j @ act_j.
// Reproduces zeros(moe_out) + 8 sequential axpy_f32 in slot order. grid: (out_f); block: 256.
extern "C" __global__ void moe_down8_fma_f32(
        wptr8_t dp, f32x8_t w, const float* __restrict__ act, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s[32];
    float chain = 0.0f;              // tid 0's slot-ordered accumulator (other threads' unused)
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const float* xrow = act + (size_t)j * in_f;
        float acc = 0.0f;
        for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt, wrow, i) * xrow[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            // slot-ordered FMA chain == the sequential axpy_f32 accumulation (see header).
            if (tid == 0) chain = __fmaf_rn(w.v[j], v, chain);
        }
        __syncthreads();             // s[] reused next iteration
    }
    if (tid == 0) dst[o] = chain;
}

// ================================================================================================
// LAUNCH-STRUCTURE STAGE 3 (2026-07-05): DEVICE-SIDE ROUTED DISPATCH for fully-resident layers.
//
// The stage-1 router still pays ONE DtoH + stream sync per MoE layer per token (~36us of host
// stall x 40 layers = the largest non-kernel slice of the 35B decode wall after stage 2). When
// EVERY block of a layer is SLRU-resident (prewarmed or organically), the host does not need
// sel/w at all: these twins read the router's device sel/w output directly and fetch the 8
// expert weight pointers from a per-layer device table [3, n_expert] of slot base addresses
// (gate row, up row, down row — fixed addresses for the cache's lifetime).
//
// BIT-IDENTITY vs moe_gate_up_silu8_f32/moe_down8_fma_f32: the ONLY change is where the weight
// pointer and the w scalar come from (device loads instead of kernel params). Same grid/block,
// same dot reduction order, same SiLU expression, same slot-ordered __fmaf_rn chain. The sel/w
// VALUES are the same bits either way (both paths consume moe_router_topk_f32's output).
// ================================================================================================
// q8 dp4a twins of the _dev pair (resident-experts arc, 2026-07-06): device sel/w + pointer
// table (like _dev) + int dp4a dots vs a q8_1 activation (like the _q8 pair). One warp per
// (row, slot); same silu expression / slot-ordered FMA chain as every twin in this family.
// Per-expert macro-scale fold for the DOWN projection: w[i] *= macros[2*n_expert + sel[i]],
// applied to the device router weights once per layer (compressed-tensors NVFP4 artifacts
// carry per-expert global scales). Every down twin consumes w verbatim, so this one launch
// macro-folds all of them. Launched ONLY when the layer carries non-trivial macros.
extern "C" __global__ void moe_w_scale_by_expert(
        float* __restrict__ w, const int* __restrict__ sel,
        const float* __restrict__ macros, int n_expert, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= __ldg(&macros[2 * n_expert + sel[i]]);
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}
// gemma4 GELU twin of moe_gate_up_silu8_dev_q8: identical dots/reduce, gelu_tanh epilogue
// (the gelu_tanh_mul_f32 expression exactly).
extern "C" __global__ void moe_gate_up_gelu8_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float x = accg;
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[(size_t)j * n_ff + o] = 0.5f * x * (1.0f + th) * accu;
    }
}

// gemma4 R3: fold the per-expert OUTPUT scale into the routing weights on device:
// w[i] *= s[sel[i]] (post-renorm — associative with the down accumulate's w*dot).
extern "C" __global__ void moe_w_exscale(float* __restrict__ w, const int* __restrict__ sel,
                                         const float* __restrict__ s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= s[sel[i]];
}

extern "C" __global__ void moe_down8_fma_dev_q8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int ex = sel[j];
        const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(w[j], acc, chain);
    }
    if (lane == 0) dst[o] = chain;
}

// ---- dev_q8 LAUNCH-GEOMETRY VARIANTS (multirow/occupancy arc 2026-07-05, rtx6000 lane) ----
// Baseline geometry is warp-starved on 188 SMs: gate_up = 4096 one-warp blocks (n_ff x n_used),
// down = 2048 one-warp blocks with an n_used=8 SERIAL slot loop AND nsb=16 (in_f=512) leaving
// lanes 16..31 idle in every dot. These variants change ONLY launch geometry; the per-(row,slot)
// accumulation is expert_dot_g in the SAME g order + the SAME warp_reduce_sum tree, and the down
// FMA chain stays slot-ordered serial -> outputs BIT-IDENTICAL to the base pair.
//
//   gu_geom<RPW>: each warp computes RPW consecutive rows of ONE slot; the activation group
//   (aqb/d8) is read once per g and reused across the RPW gate+up dots (RPW weight streams in
//   flight hide load latency — the q4k/q5k mmvq multirow recipe). blockDim.y packs several
//   row-tiles per block for scheduler occupancy; grid = (ceil(n_ff/(RPW*wpb)), n_used).
//
//   down_w8<RPW>: block = (32, n_used) — warp j computes slot j's dot for RPW consecutive rows
//   (identical 32-lane tree per (row,slot)), partials land in smem, then warp 0 lane 0 replays
//   the slot-ordered __fmaf_rn chain per row (8 sequential FMAs — cheap). The n_used loop
//   parallelizes; ONLY the chain stays serial (bit-identity contract).
template<int RPW>
__device__ __forceinline__ void moe_gu_dev_q8_geom(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o0 = ((int)blockIdx.x * (int)blockDim.y + (int)threadIdx.y) * RPW;
    int j = blockIdx.y;
    if (o0 >= n_ff) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* gbase = (const unsigned char*)table[ex];
    const unsigned char* ubase = (const unsigned char*)table[n_expert + ex];
    float accg[RPW], accu[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { accg[r] = 0.0f; accu[r] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= n_ff) break;
            accg[r] += expert_dot_g(qt_g, gbase + (long)o * rb_g, g, aqb, d8, nsb);
            accu[r] += expert_dot_g(qt_u, ubase + (long)o * rb_u, g, aqb, d8, nsb);
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= n_ff) break;
        float ag = warp_reduce_sum(accg[r]) * __ldg(&macros[ex]);
        float au = warp_reduce_sum(accu[r]) * __ldg(&macros[n_expert + ex]);
        if (lane == 0) act[(size_t)j * n_ff + o] = (ag / (1.0f + expf(-ag))) * au;
    }
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r1(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    moe_gu_dev_q8_geom<1>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u, macros);
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    moe_gu_dev_q8_geom<2>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u, macros);
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_r4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    moe_gu_dev_q8_geom<4>(table, sel, aq, ad, act, in_f, n_ff, n_expert, qt_g, qt_u, rb_g, rb_u, macros);
}
template<int RPW>
__device__ __forceinline__ void moe_down8_dev_q8_w8_geom(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * RPW;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    int nsb = in_f >> 5;
    __shared__ float s[RPW][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc[RPW];
        #pragma unroll
        for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
        for (int g = lane; g < nsb; g += 32) {
            const signed char* aqb = arow + (size_t)g * 32;
            float d8 = adrow[g];
            #pragma unroll
            for (int r = 0; r < RPW; r++) {
                int o = o0 + r;
                if (o >= out_f) break;
                acc[r] += expert_dot_g(qt, wbase + (long)o * rb, g, aqb, d8, nsb);
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            float a = warp_reduce_sum(acc[r]);
            if (lane == 0 && o0 + r < out_f) s[r][j] = a;
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;          // slot-ordered serial chain == base kernel's exact FP order
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r1(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<1>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<2>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8r4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    moe_down8_dev_q8_w8_geom<4>(table, sel, w, aq2, ad2, dst, in_f, out_f, n_used, n_expert, qt, rb);
}

// ---- ROUND 2 geometry variants (same arc): idle-lane fix + warp-split ----
//
// down HALF-WARP DUAL-ROW (nsb==16 ONLY, i.e. in_f==512 — the 35B expert down shape): the base
// dot loop `for (g = lane; g < nsb; g += 32)` leaves lanes 16..31 IDLE when nsb=16. Here lanes
// 0..15 compute row o0 and lanes 16..31 compute row o0+1 (same g = lane&15 per half — exactly
// the base kernel's per-lane group assignment, single iteration so no accumulation-order change).
// BIT-IDENTITY of the reduce: the base 32-lane tree runs with lanes 16..31 holding 0.0f; row A
// reproduces that exactly by masking the upper half to 0.0f; row B's partials are shifted down
// 16 lanes first (so group g sits at lane g, like base) then upper half masked — SAME tree, SAME
// bits. The FMA chain stays slot-ordered serial on warp 0.
//   _h2:   block (32,1)  grid (out_f/2)      — serial n_used loop, 2 rows/warp
//   _w8h2: block (32,8)  grid (out_f/2)      — warp j = slot j, 2 rows/warp, smem chain replay
__device__ __forceinline__ float2 down_h2_dot(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, int qt, long rb, int lane) {
    int nsb = in_f >> 5;                       // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                              + (long)(o0 + half) * rb;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    // one group per lane (nsb==16): identical expert_dot_g call to the base kernel's lane l16.
    float acc = expert_dot_g_v(qt, wrow, l16, arow + (size_t)l16 * 32, adrow[l16], nsb);
    // row A (o0): lanes 0..15 partials, upper half 0 — the base tree layout verbatim.
    float accA = (half == 0) ? acc : 0.0f;
    float a0 = warp_reduce_sum(accA);
    // row B (o0+1): shift partials down 16 so group g sits at lane g, mask upper half.
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f;
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void moe_down8_fma_dev_q8_h2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    float chain0 = 0.0f, chain1 = 0.0f;
    for (int j = 0; j < n_used; j++) {
        float2 a = down_h2_dot(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) {
            chain0 = __fmaf_rn(w[j], a.x, chain0);
            chain1 = __fmaf_rn(w[j], a.y, chain1);
        }
    }
    if (lane == 0) {
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}

// w8h2 x mr2: each half-warp computes TWO serial rows (activation group regs reused across the
// row pair — the mr2 recipe stacked on h2). 4 rows/block, block (32,8), grid (out_f/4).
// BIT-IDENTITY per row: same single-group expert_dot_g call, same masked 32-lane tree as h2.
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2r2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 4;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    int nsb = in_f >> 5;                 // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    __shared__ float s[4][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* aqb = aq2 + (size_t)j * in_f + (size_t)l16 * 32;
        float d8 = ad2[(size_t)j * nsb + l16];
        #pragma unroll
        for (int r = 0; r < 2; r++) {    // two row-pairs, activation regs (aqb/d8) reused
            int o = o0 + 2 * r + half;
            float acc = (o < out_f)
                ? expert_dot_g(qt, wbase + (long)o * rb, l16, aqb, d8, nsb) : 0.0f;
            float accA = (half == 0) ? acc : 0.0f;
            float a0 = warp_reduce_sum(accA);
            float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
            float accB = (lane < 16) ? shifted : 0.0f;
            float a1 = warp_reduce_sum(accB);
            if (lane == 0) { s[2 * r][j] = a0; s[2 * r + 1][j] = a1; }
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}

// ---- WIDE-LOAD (_v) twins (down8 lane 2026-07-08) ----
// The w8h2/w8h2r2/base-gate_up bodies VERBATIM with expert_dot_g swapped for expert_dot_g_v:
// same geometry, same g order, same masked 32-lane tree, same slot-ordered __fmaf_rn chain,
// same SiLU expression. Only the IQ4_XS group-dot internals change (value-identical wide loads,
// see expert_dot_iq4xs_g_v) -> outputs BIT-IDENTICAL to their scalar twins.
//   _w8h2v:   MEMRA_MOE_DEVQ8_DOWN=w8h2v   (w8h2 geometry — the current 35B auto winner)
//   _w8h2r2v: MEMRA_MOE_DEVQ8_DOWN=w8h2r2v (r2 re-test: activation-reg reuse may pay once the
//             decode is cheap — the tradeoff that lost by 1% at scalar decode cost)
//   gate_up _v: MEMRA_MOE_DEVQ8_GU=v (same dot body feeds the 69%-eff gate_up twin, 15.1us x
//             40/tok — bigger absolute slice than down; base geometry)
__device__ __forceinline__ float2 down_h2_dot_v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        int j, int o0, int in_f, int n_expert, int qt, long rb, int lane) {
    int nsb = in_f >> 5;                       // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    int ex = sel[j];
    const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                              + (long)(o0 + half) * rb;
    const signed char* arow = aq2 + (size_t)j * in_f;
    const float* adrow = ad2 + (size_t)j * nsb;
    float acc = expert_dot_g_v(qt, wrow, l16, arow + (size_t)l16 * 32, adrow[l16], nsb);
    float accA = (half == 0) ? acc : 0.0f;     // row A: base tree layout verbatim
    float a0 = warp_reduce_sum(accA);
    float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
    float accB = (lane < 16) ? shifted : 0.0f; // row B: shift-down-16 then mask, same tree
    float a1 = warp_reduce_sum(accB);
    return make_float2(a0, a1);
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_v(table, sel, aq2, ad2, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(w[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(w[jj], s[1][jj], chain1);
        }
        dst[o0] = chain0;
        if (o0 + 1 < out_f) dst[o0 + 1] = chain1;
    }
}
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2r2v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o0 = (int)blockIdx.x * 4;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    int nsb = in_f >> 5;                 // == 16 (dispatch-gated)
    int half = lane >> 4, l16 = lane & 15;
    __shared__ float s[4][8];
    if (j < n_used) {
        int ex = sel[j];
        const unsigned char* wbase = (const unsigned char*)table[2 * n_expert + ex];
        const signed char* aqb = aq2 + (size_t)j * in_f + (size_t)l16 * 32;
        float d8 = ad2[(size_t)j * nsb + l16];
        #pragma unroll
        for (int r = 0; r < 2; r++) {    // two row-pairs, activation regs (aqb/d8) reused
            int o = o0 + 2 * r + half;
            float acc = (o < out_f)
                ? expert_dot_g_v(qt, wbase + (long)o * rb, l16, aqb, d8, nsb) : 0.0f;
            float accA = (half == 0) ? acc : 0.0f;
            float a0 = warp_reduce_sum(accA);
            float shifted = __shfl_down_sync(0xffffffffu, acc, 16);
            float accB = (lane < 16) ? shifted : 0.0f;
            float a1 = warp_reduce_sum(accB);
            if (lane == 0) { s[2 * r][j] = a0; s[2 * r + 1][j] = a1; }
        }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            float chain = 0.0f;
            for (int jj = 0; jj < n_used; jj++) chain = __fmaf_rn(w[jj], s[r][jj], chain);
            dst[o] = chain;
        }
    }
}
extern "C" __global__ void moe_gate_up_silu8_dev_q8_v(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}

// ---- WALL-GAP ARC (2026-07-10, owner: "94% of wall is not 100%"): cp.async ROW-STAGED
// gate_up twin. The _v dot issues ~24 scattered synchronous byte-loads per lane per iteration
// (IQ3_S superblock: qs/qh/signs/scales all separate) — measured 482GB/s = 56% of wall, the
// b4-tier long_scoreboard signature. This twin bulk-stages BOTH expert rows to shared memory
// with cp.async 16B chunks (one commit/wait, no ring), then runs the dot bodies VERBATIM from
// smem — same bytes, same order, byte-identical outputs. Rows are 16B-aligned by construction
// (IQ3_S rb = in_f/256*110: 880B at in_f 2048; slab bases 256B-aligned).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_vsm(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    extern __shared__ unsigned char srow_vsm[];          // [rb_g + rb_u]
    for (int off = lane * 16; off < (int)rb_g; off += 32 * 16)
        cp_async16_g(srow_vsm + off, grow + off);
    for (int off = lane * 16; off < (int)rb_u; off += 32 * 16)
        cp_async16_g(srow_vsm + rb_g + off, urow + off);
    cp_async_commit();
    cp_async_wait<0>();
    __syncwarp();
    const unsigned char* gsm = srow_vsm;
    const unsigned char* usm = srow_vsm + rb_g;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}

// vsm2: 2-stage pipelined variant — rows split in half along superblocks; half h+1's cp.async
// is in flight while half h computes. Same dot bodies from smem (byte-identical values); the
// per-lane group order is UNCHANGED?? NO — it is: lane processes g = lane, lane+32 which spans
// both halves (g=lane in half0 for lane<32 when nsb=64: g=lane -> superblock g/8 -> halves by
// g<nsb/2). Loop restructured to walk halves outer, g-within-half inner: per lane the two
// g-values (lane, lane+32) land one in EACH half at nsb=64 -> same two groups, same ORDER
// (lane < lane+32 == half0 then half1). accg accumulation order preserved -> bit-identical.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_vsm2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    extern __shared__ unsigned char srow2[];             // [rb_g + rb_u]
    int hg = (int)rb_g / 2, hu = (int)rb_u / 2;          // halves are 16B-aligned (880/2=440.. NOT 16-aligned!)
    // 440 % 16 != 0 -> split at superblock granularity instead: half0 = first (nsb/2) groups'
    // superblocks. IQ3_S: 8 groups per 110B superblock; nsb=64 -> 8 superblocks -> half = 4
    // superblocks = 440B. cp.async 16B needs 16B alignment: 440 % 16 = 8 -> VIOLATION.
    // Fallback: stage half0 = ceil-to-16B prefix; the boundary superblock loads land in stage 0.
    int h0g = (hg + 15) & ~15;
    int h0u = (hu + 15) & ~15;
    if (h0g > (int)rb_g) h0g = (int)rb_g;
    if (h0u > (int)rb_u) h0u = (int)rb_u;
    // stage 0: first halves of both rows
    for (int off = lane * 16; off < h0g; off += 512) cp_async16_g(srow2 + off, grow + off);
    for (int off = lane * 16; off < h0u; off += 512) cp_async16_g(srow2 + rb_g + off, urow + off);
    cp_async_commit();
    // stage 1: second halves (issued now, awaited after half-0 compute)
    for (int off = h0g + lane * 16; off < (int)rb_g; off += 512) cp_async16_g(srow2 + off, grow + off);
    for (int off = h0u + lane * 16; off < (int)rb_u; off += 512) cp_async16_g(srow2 + rb_g + off, urow + off);
    cp_async_commit();
    const unsigned char* gsm = srow2;
    const unsigned char* usm = srow2 + rb_g;
    float accg = 0.0f, accu = 0.0f;
    int half_nsb = nsb / 2;
    cp_async_wait<1>();                                   // half 0 resident
    __syncwarp();
    for (int g = lane; g < half_nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8, nsb);
    }
    cp_async_wait<0>();                                   // half 1 resident
    __syncwarp();
    for (int g = half_nsb + lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_v(qt_g, gsm, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, usm, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}

// ---- SMALL-M VERIFY ROWS TWINS (MEMRA_SPEC_M2, lane/spec-m2 2026-07-08) ----
// grid.z = token: the spec verify's MoE dev token loop (t = 2..K+2) ran one launch-pair per
// token (plus per-token quantizes: 4t launches/layer). These twins run the serial loop's
// per-token program on a z axis of tokens — every pointer is offset by tok exactly as the host
// loop sliced it (sel/w + tok*n_used; aq/ad + token activation rows; act/dst + token output
// rows). Per (token, row, slot) the body is the _v / w8h2v kernel VERBATIM: same dot order,
// same warp tree, same slot-ordered __fmaf_rn chain -> outputs BIT-IDENTICAL to the serial
// loop. n_used rides a kernel param here (the non-rows gate_up encodes it as gridDim.y, which
// the z-twin keeps; down needs it for the activation-row stride).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_v_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used,
        const float* __restrict__ macros) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[tok * n_used + j];
    const signed char* aqt = aq + (size_t)tok * in_f;
    const float* adt = ad + (size_t)tok * nsb;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aqt + (size_t)g * 32;
        float d8 = adt[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[((size_t)tok * n_used + j) * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}
// gemma4 GELU rows twin (verify t=2..K+2, one launch for all tokens): per (token, row, slot)
// the body is moe_gate_up_gelu8_dev_q8 VERBATIM (expert_dot_g order, warp tree, gelu epilogue).
extern "C" __global__ void moe_gate_up_gelu8_dev_q8_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[tok * n_used + j];
    const signed char* aqt = aq + (size_t)tok * in_f;
    const float* adt = ad + (size_t)tok * nsb;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aqt + (size_t)g * 32;
        float d8 = adt[g];
        accg += expert_dot_g_v(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g_v(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float x = accg;
        float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
        act[((size_t)tok * n_used + j) * n_ff + o] = 0.5f * x * (1.0f + th) * accu;
    }
}

// gemma4 generic down rows twin: grid.z = token; per row the base moe_down8_fma_dev_q8 body
// VERBATIM (serial slot-ordered __fmaf_rn chain).
extern "C" __global__ void moe_down8_fma_dev_q8_rows_g(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const int* selt = sel + tok * n_used;
    const float* wt = w + tok * n_used;
    float chain = 0.0f;
    for (int j = 0; j < n_used; j++) {
        int ex = selt[j];
        const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex] + (long)o * rb;
        const signed char* arow = aq2 + ((size_t)tok * n_used + j) * in_f;
        const float* adrow = ad2 + ((size_t)tok * n_used + j) * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g_v(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) chain = __fmaf_rn(wt[j], acc, chain);
    }
    if (lane == 0) dst[(size_t)tok * out_f + o] = chain;
}

// Step-3.7 B=1 down twin: the base rows kernel launches one 32-thread block per
// output row, so sm_120's 24-block/SM limit caps it at 24 of 48 resident warps.
// Warp j computes exactly the base kernel's dot for slot j (same g assignment,
// expert_dot_g_v body, and warp reduction tree); warp 0 lane 0 then replays the
// original slot-ordered __fmaf_rn chain. Only the independent slot dots move
// from serial to parallel, so output bits and weight bytes are unchanged.
extern "C" __global__ void moe_down8_fma_dev_q8_rows_w8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w,
        const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int tok = blockIdx.z;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int j = threadIdx.y;
    int nsb = in_f >> 5;
    const int* selt = sel + tok * n_used;
    const float* wt = w + tok * n_used;
    __shared__ float partial[8];
    if (j < n_used) {
        int ex = selt[j];
        const unsigned char* wrow = (const unsigned char*)table[2 * n_expert + ex]
                                  + (long)o * rb;
        const signed char* arow = aq2 + ((size_t)tok * n_used + j) * in_f;
        const float* adrow = ad2 + ((size_t)tok * n_used + j) * nsb;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32)
            acc += expert_dot_g_v(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) partial[j] = acc;
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain = 0.0f;
        for (int jj = 0; jj < n_used; jj++)
            chain = __fmaf_rn(wt[jj], partial[jj], chain);
        dst[(size_t)tok * out_f + o] = chain;
    }
}

// down rows twin of w8h2v (the AUTO winner for the 35B shape, in_f==512 && n_used<=8 —
// dispatch-gated by the host). Same down_h2_dot_v body per (token, row-pair, slot).
extern "C" __global__ void moe_down8_fma_dev_q8_w8h2v_rows(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const float* __restrict__ w, const signed char* __restrict__ aq2,
        const float* __restrict__ ad2, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int tok = blockIdx.z;
    int o0 = (int)blockIdx.x * 2;
    if (o0 >= out_f) return;
    int lane = threadIdx.x;
    int j = threadIdx.y;                 // slot; blockDim.y == n_used (max 8)
    const int* selt = sel + tok * n_used;
    const float* wt = w + tok * n_used;
    const signed char* aq2t = aq2 + (size_t)tok * n_used * in_f;
    const float* ad2t = ad2 + (size_t)tok * n_used * (in_f >> 5);
    float* dstt = dst + (size_t)tok * out_f;
    __shared__ float s[2][8];
    if (j < n_used) {
        float2 a = down_h2_dot_v(table, selt, aq2t, ad2t, j, o0, in_f, n_expert, qt, rb, lane);
        if (lane == 0) { s[0][j] = a.x; s[1][j] = a.y; }
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain0 = 0.0f, chain1 = 0.0f;
        for (int jj = 0; jj < n_used; jj++) {   // slot-ordered serial == base FP order
            chain0 = __fmaf_rn(wt[jj], s[0][jj], chain0);
            chain1 = __fmaf_rn(wt[jj], s[1][jj], chain1);
        }
        dstt[o0] = chain0;
        if (o0 + 1 < out_f) dstt[o0 + 1] = chain1;
    }
}

// gate_up SLOT-PACKED blocks: block (32, n_used), warp j = slot j for the SAME row o — one block
// per row, 8x fewer blocks, same warp count; the 8 warps share the row's activation groups via
// L1. Each warp's body is the base kernel VERBATIM (same loop, same tree) -> bit-identical.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_j8(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = threadIdx.y;                 // slot from block y-dim; blockDim.y == n_used
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}

// ---- IQ3_S SMEM-GRID twins (2026-07-06 rtx6000 lane) ----
// The iq3s_grid LUT moved __constant__ -> __device__ (+11.8% decode: constant-cache divergent-read
// serialization). It still rides L1 though, CONTENDING with the weight stream (each gate_up warp
// does ~32 divergent grid lookups per lane per launch on the 35B shape). These twins copy the 2KB
// grid (512 u32) into SHARED memory once per block and look up from smem: banked, divergent-
// friendly, zero L1 contention. VALUES are the same table bytes -> outputs BIT-IDENTICAL (same
// expert_dot_iq3s_g expression, same g order, same warp tree).
__device__ __forceinline__ float expert_dot_iq3s_g_sm(const unsigned char* wrow, int g,
                                                      const signed char* aqb, float d8,
                                                      const unsigned int* gsm) {
    int sblk = g >> 3, ib32 = g & 7;
    const unsigned char* b = wrow + (long)sblk * 110;
    float d = half_to_float(*(const unsigned short*)b);
    const unsigned char* qs    = b + 2  + ib32 * 8;
    unsigned char qh           = b[66 + ib32];
    const unsigned char* signs = b + 74 + ib32 * 4;
    const unsigned char* scales= b + 106;
    int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
    float db = d * (1.0f + 2.0f * (float)sc_nib);
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int gl = gsm[qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100)];
        int gh = gsm[qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100)];
        unsigned char sb = signs[l0 / 2];
        int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
        int grid_l = __vsub4(gl ^ signs0, signs0);
        int grid_h = __vsub4(gh ^ signs1, signs1);
        sumi = dp4a(grid_l, aq4[l0 + 0], sumi);
        sumi = dp4a(grid_h, aq4[l0 + 1], sumi);
    }
    return db * (float)sumi * d8;
}
// smem-or-L1 dot: IQ3_S goes through the smem grid; every other qtype = expert_dot_g verbatim.
__device__ __forceinline__ float expert_dot_g_sm(int qtype, const unsigned char* wrow, int g,
                                                 const signed char* aqb, float d8,
                                                 const unsigned int* gsm, int nsb) {
    if (qtype == QT_IQ3_S) return expert_dot_iq3s_g_sm(wrow, g, aqb, d8, gsm);
    return expert_dot_g(qtype, wrow, g, aqb, d8, nsb);
}
// base-geometry twin: grid (n_ff, n_used), block (32,1) — ONE warp both copies the 2KB grid
// (16 coalesced u32 loads/lane) and runs the base dot loop.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_sg(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    __shared__ unsigned int gsm[512];
    int lane = threadIdx.x;
    #pragma unroll
    for (int i = lane; i < 512; i += 32) gsm[i] = iq3s_grid_const[i];
    __syncwarp();
    int o = blockIdx.x;
    int j = blockIdx.y;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_sm(qt_g, grow, g, aqb, d8, gsm, nsb);
        accu += expert_dot_g_sm(qt_u, urow, g, aqb, d8, gsm, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}
// j8-geometry twin: block (32, n_used) — ONE 2KB copy (spread over all 32*n_used threads)
// serves n_used warps; 8x fewer blocks = 8x less copy traffic than _sg.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_j8sg(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    __shared__ unsigned int gsm[512];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int nth = blockDim.y * 32;
    for (int i = tid; i < 512; i += nth) gsm[i] = iq3s_grid_const[i];
    __syncthreads();
    int o = blockIdx.x;
    int j = threadIdx.y;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g_sm(qt_g, grow, g, aqb, d8, gsm, nsb);
        accu += expert_dot_g_sm(qt_u, urow, g, aqb, d8, gsm, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}

// gate_up 2-WARP SPLIT: the RPW multirow direction REDUCED warp count and lost; this DOUBLES it.
// block (32,2): warp 0 computes the gate dot, warp 1 the up dot — each with the base kernel's
// exact per-warp g order + 32-lane tree (bit-identical partials); warp 0 lane 0 applies the same
// silu expression after the smem exchange. grid unchanged (n_ff, n_used) -> 2x warps in flight
// on the same latency-bound weight streams, zero extra launches, zero numeric change.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_s2(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int which = threadIdx.y;             // 0 = gate, 1 = up
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* wrow = (which == 0)
        ? (const unsigned char*)table[ex] + (long)o * rb_g
        : (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    long qt = (which == 0) ? qt_g : qt_u;
    __shared__ float sg, su;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g((int)qt, wrow, g, aq + (size_t)g * 32, ad[g], nsb);
    acc = warp_reduce_sum(acc);
    if (lane == 0) { if (which == 0) sg = acc; else su = acc; }
    __syncthreads();
    if (which == 0 && lane == 0) {
        float g = sg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (su * __ldg(&macros[n_expert + ex]));
    }
}
// gate_up 4-WARP G-SPLIT (nsb==64 ONLY, i.e. in_f==2048 — the 35B expert gate/up shape): block
// (32,4), warp y: 0=gate-low 1=gate-high 2=up-low 3=up-high. "low" computes group g=lane, "high"
// g=lane+32 — TOGETHER exactly the base warp's two serial iterations. BIT-IDENTITY: base per-lane
// acc = (0 + d(g=l)) + d(g=l+32); here low's d(l) (0+x==x) merges with high's d(l+32) via smem in
// that same order, then the SAME 32-lane tree runs in the low warp. Halves each warp's serial
// group count AND 4x the warps in flight; the up-warp's silu operand crosses via smem.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_gs4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int nsb = in_f >> 5;  // slot count for the slot-major (QT_NVFP4_V2) scale tail
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int wy = threadIdx.y;                // 0..3: 0=gate-low 1=gate-high 2=up-low 3=up-high
    int is_up = wy >> 1, is_hi = wy & 1;
    int ex = sel[j];
    const unsigned char* wrow = is_up
        ? (const unsigned char*)table[n_expert + ex] + (long)o * rb_u
        : (const unsigned char*)table[ex] + (long)o * rb_g;
    int qt = is_up ? qt_u : qt_g;
    int g = lane + (is_hi << 5);
    float d = expert_dot_g(qt, wrow, g, aq + (size_t)g * 32, ad[g], nsb);
    __shared__ float hi[2][32];
    __shared__ float gu[2];
    if (is_hi) hi[is_up][lane] = d;
    __syncthreads();
    if (!is_hi) {
        float acc = d + hi[is_up][lane]; // base per-lane serial order verbatim
        acc = warp_reduce_sum(acc);      // base 32-lane tree
        if (lane == 0) gu[is_up] = acc;
    }
    __syncthreads();
    if (wy == 0 && lane == 0) {
        float gg = gu[0] * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (gg / (1.0f + expf(-gg))) * (gu[1] * __ldg(&macros[n_expert + ex]));
    }
}
// gate_up nsb==64 UNROLLED twin (in_f==2048 — the 35B expert gate/up shape): the base loop's two
// g-iterations (g=lane, g=lane+32) are issued as INDEPENDENT expressions so all 4 dot bodies'
// loads pipeline (base: accg/accu serialize each warp's second iteration behind the first).
// BIT-IDENTITY: accg = (0 + dg(l)) + dg(l+32) — the base loop's exact accumulation order — then
// the same warp tree + silu expression. Geometry unchanged (one warp per (row,slot)).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_u64(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int nsb = in_f >> 5;  // slot count for the slot-major (QT_NVFP4_V2) scale tail
    int o = blockIdx.x;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int ex = sel[j];
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    int g0 = lane, g1 = lane + 32;       // nsb==64 exactly (dispatch-gated)
    const signed char* a0 = aq + (size_t)g0 * 32;
    const signed char* a1 = aq + (size_t)g1 * 32;
    float d80 = ad[g0], d81 = ad[g1];
    float g_lo = expert_dot_g(qt_g, grow, g0, a0, d80, nsb);
    float g_hi = expert_dot_g(qt_g, grow, g1, a1, d81, nsb);
    float u_lo = expert_dot_g(qt_u, urow, g0, a0, d80, nsb);
    float u_hi = expert_dot_g(qt_u, urow, g1, a1, d81, nsb);
    float accg = (0.0f + g_lo) + g_hi;   // base loop's accumulation order verbatim
    float accu = (0.0f + u_lo) + u_hi;
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float g = accg * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (accu * __ldg(&macros[n_expert + ex]));
    }
}
// s2 with ROWS packed per block for scheduler density: block (32,2,rz), grid (n_ff/rz, n_used).
extern "C" __global__ void moe_gate_up_silu8_dev_q8_s2z(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = (int)blockIdx.x * (int)blockDim.z + (int)threadIdx.z;
    if (o >= n_ff) return;
    int j = blockIdx.y;
    int lane = threadIdx.x;
    int which = threadIdx.y;
    int nsb = in_f >> 5;
    int ex = sel[j];
    const unsigned char* wrow = (which == 0)
        ? (const unsigned char*)table[ex] + (long)o * rb_g
        : (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    long qt = (which == 0) ? qt_g : qt_u;
    __shared__ float sgu[16][2];         // [z][which]
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32)
        acc += expert_dot_g((int)qt, wrow, g, aq + (size_t)g * 32, ad[g], nsb);
    acc = warp_reduce_sum(acc);
    if (lane == 0) sgu[threadIdx.z][which] = acc;
    __syncthreads();
    if (which == 0 && lane == 0) {
        float g = sgu[threadIdx.z][0] * __ldg(&macros[ex]);
        act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (sgu[threadIdx.z][1] * __ldg(&macros[n_expert + ex]));
    }
}

extern "C" __global__ void moe_gate_up_silu8_dev(
        const unsigned long long* __restrict__ table,  // [3, n_expert] slot base addresses
        const int* __restrict__ sel,                   // [n_used] this token's expert ids (device)
        const float* __restrict__ x, float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        const float* __restrict__ macros) {
    int o = blockIdx.x;              // expert-FFN row 0..n_ff-1
    int j = blockIdx.y;              // routed-expert slot 0..n_used-1
    int tid = threadIdx.x;
    __shared__ float s[32];
    __shared__ float g_final;
    const int ex = sel[j];           // broadcast load
    // ---- gate dot: EXACT qmatvec_f32 structure ----
    const unsigned char* grow = (const unsigned char*)(uintptr_t)table[ex] + (long)o * rb_g;
    float acc = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt_g, grow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) g_final = v;
    }
    __syncthreads();                 // s + g_final ready; s reused below
    // ---- up dot: same structure ----
    const unsigned char* urow = (const unsigned char*)(uintptr_t)table[n_expert + ex] + (long)o * rb_u;
    float acc2 = 0.0f;
    for (int i = tid; i < in_f; i += blockDim.x) acc2 += deq(qt_u, urow, i) * x[i];
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc2;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) {
            float g = g_final * __ldg(&macros[ex]);
            // silu_mul_f32's exact expression on the exact dot values.
            act[(size_t)j * n_ff + o] = (g / (1.0f + expf(-g))) * (v * __ldg(&macros[n_expert + ex]));
        }
    }
}

extern "C" __global__ void moe_down8_fma_dev(
        const unsigned long long* __restrict__ table,  // [3, n_expert]; down row at 2*n_expert
        const int* __restrict__ sel,                   // [n_used] (device)
        const float* __restrict__ w,                   // [n_used] renormalized weights (device)
        const float* __restrict__ act, float* __restrict__ dst,
        int in_f, int out_f, int n_used, int n_expert, int qt, long rb) {
    int o = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s[32];
    float chain = 0.0f;              // tid 0's slot-ordered accumulator (other threads' unused)
    for (int j = 0; j < n_used; j++) {
        const unsigned char* wrow =
            (const unsigned char*)(uintptr_t)table[2 * n_expert + sel[j]] + (long)o * rb;
        const float* xrow = act + (size_t)j * in_f;
        float acc = 0.0f;
        for (int i = tid; i < in_f; i += blockDim.x) acc += deq(qt, wrow, i) * xrow[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            // slot-ordered FMA chain == the sequential axpy_f32 accumulation (see header).
            if (tid == 0) chain = __fmaf_rn(w[j], v, chain);
        }
        __syncthreads();             // s[] reused next iteration
    }
    if (tid == 0) dst[o] = chain;
}

// ---- CSR EXPERT-DEDUP VERIFY TWINS (verify-cost target #1, 2026-07-10) ----
// The _rows twins re-stream + re-decode each selected expert's full gate/up/down rows once per
// (token, slot) pair. Measured cross-token overlap at verify (MEMRA_MOE_OVERLAP, 35B K=3 p2,
// t=4): unique/pairs = 0.60-0.62 — 38-40% of the expert weight traffic AND nibble decode is
// duplicated. These twins group pairs by expert (CSR built ON DEVICE — ZERO-DtoH preserved),
// hoist the IQ4_XS/IQ3_S group decode into registers once per (expert, row, group), and replay only
// the per-pair dp4a chain per token.
// BIT-IDENTITY CONTRACT: exp_decode_g_cached + exp_dot_cached replay expert_dot_iq4xs_g /
// expert_dot_iq3s_g VERBATIM (same packing, same dp4a order, same scalar expression);
// per pair the lane-strided g order and the warp tree match the _v_rows / w8h2v bodies; the
// down combine replays the slot-ordered __fmaf_rn chain. Outputs bit-identical to the _rows
// twins. Host dispatch-gates each projection qtype to {IQ4_XS, IQ3_S} (the k-quant tail
// layers keep the _rows twins).
// Cached group decode: 8 dp4a weight words + (d, scale) such that the dot below replays the
// expert_dot_*_g expression bit-for-bit. IQ4_XS: w[k]=wlo[k], w[4+k]=whi[k], expression
// d_sb*(float)(scale*sumi)*d8. IQ3_S: w[k] = signed grid ints in linear aq4 order, scale=1
// (so (float)(1*sumi) == (float)sumi), expression db*(float)sumi*d8. The 35B UD mix runs
// gate/up = IQ3_S, down = IQ4_XS.
struct expg { int w[8]; float d; int scale; };
__device__ __forceinline__ expg exp_decode_g_cached(int qt, const unsigned char* wrow, int g) {
    expg r;
    if (qt == 5) {                              // QT_IQ4_XS
        int sblk = g >> 3, ib = g & 7;
        const unsigned char* b = wrow + (long)sblk * 136;
        r.d = half_to_float(*(const unsigned short*)b);
        unsigned short sh = *(const unsigned short*)(b + 2);
        const unsigned char* sl = b + 4;
        const unsigned char* qs = b + 8 + ib * 16;
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        r.scale = ls - 32;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            r.w[k]   = (kvalues_iq4nl_d[qs[k*4+0]&0xf]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]&0xf]&0xff)<<8)
                     | ((kvalues_iq4nl_d[qs[k*4+2]&0xf]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]&0xf]&0xff)<<24);
            r.w[4+k] = (kvalues_iq4nl_d[qs[k*4+0]>>4]&0xff) | ((kvalues_iq4nl_d[qs[k*4+1]>>4]&0xff)<<8)
                     | ((kvalues_iq4nl_d[qs[k*4+2]>>4]&0xff)<<16) | ((kvalues_iq4nl_d[qs[k*4+3]>>4]&0xff)<<24);
        }
    } else if (qt == 12) {                      // QT_Q4_0: -8 folded into the ints (gemma)
        const unsigned char* b = wrow + (long)g * 18;
        r.d = half_to_float(*(const unsigned short*)b);
        r.scale = 1;
        const unsigned char* qs = b + 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
            r.w[k]     = __vsub4((int)(raw & 0x0F0F0F0Fu), 0x08080808);
            r.w[4 + k] = __vsub4((int)((raw >> 4) & 0x0F0F0F0Fu), 0x08080808);
        }
    } else {                                    // QT_IQ3_S (host-gated to {5, 6, 12})
        int sblk = g >> 3, ib32 = g & 7;
        const unsigned char* b = wrow + (long)sblk * 110;
        float d = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs    = b + 2  + ib32 * 8;
        unsigned char qh           = b[66 + ib32];
        const unsigned char* signs = b + 74 + ib32 * 4;
        const unsigned char* scales= b + 106;
        int sc_nib = (ib32 & 1) ? (scales[ib32 / 2] >> 4) : (scales[ib32 / 2] & 0xf);
        r.d = d * (1.0f + 2.0f * (float)sc_nib);
        r.scale = 1;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            int gl = iq3s_grid_d(qs[l0 + 0] | (((int)qh << (8 - l0)) & 0x100));
            int gh = iq3s_grid_d(qs[l0 + 1] | (((int)qh << (7 - l0)) & 0x100));
            unsigned char sb = signs[l0 / 2];
            int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
            int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);
            r.w[l0 + 0] = __vsub4(gl ^ signs0, signs0);
            r.w[l0 + 1] = __vsub4(gh ^ signs1, signs1);
        }
    }
    return r;
}
// dp4a is exact integer math — dot ORDER is bit-irrelevant; only the closing FLOAT ops care.
// CODEGEN CONTRACT (ULP lesson, 2026-07-10): the _rows twins compile `acc += d*(float)(s*sumi)*d8`
// with nvcc's default fmad contraction — the final x*d8 fuses into the accumulate as
// fma(d*(float)(s*sumi), d8, acc). A structurally different kernel contracts DIFFERENTLY and
// drifts last-ULP (measured: 35% of ACT elements). So the accumulate is written as EXPLICIT
// intrinsics here — __fmaf_rn(__fmul_rn(d,(float)(s*sumi)), d8, acc) — pinning the exact
// rounding sequence instead of trusting the optimizer to match.
__device__ __forceinline__ int exp_sumi_cached(int qt, const expg& e, const signed char* aqb) {
    const int* aq4 = (const int*)aqb;
    int sumi = 0;
    if (qt == 5 || qt == 12) {                   // IQ4_XS / Q4_0: (w[k],a[k]),(w[4+k],a[4+k])
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            sumi = dp4a(e.w[k],     aq4[k],     sumi);
            sumi = dp4a(e.w[4 + k], aq4[4 + k], sumi);
        }
    } else {                                     // IQ3_S linear order
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(e.w[k], aq4[k], sumi);
    }
    return sumi;
}
__device__ __forceinline__ float exp_dot_acc_cached(int qt, const expg& e,
                                                    const signed char* aqb, float d8, float acc) {
    int sumi = exp_sumi_cached(qt, e, aqb);
    return __fadd_rn(acc, __fmul_rn(__fmul_rn(e.d, (float)(e.scale * sumi)), d8));
}
// single-group form (down: nsb==16, one group per lane, NO accumulate) — pure rounded muls.
__device__ __forceinline__ float exp_dot_cached(int qt, const expg& e, const signed char* aqb,
                                                float d8) {
    int sumi = exp_sumi_cached(qt, e, aqb);
    return __fmul_rn(__fmul_rn(e.d, (float)(e.scale * sumi)), d8);
}

#define CSR_MAXP 10   // pairs per expert <= t <= 10 (verify t = 2..K+2, K <= 8); host-gated
// OWNER-SCAN dedup (v3): no separate CSR build — grid.y = pair index; the block whose pair is
// the FIRST occurrence of its expert OWNS the expert and serves every pair that selected it;
// duplicate blocks exit after an n_pairs-long L1 scan (~24-80 loads). v2's one-thread build
// kernel measured 18.2us/launch (5.5% of the round loop) and its parallel fix still cost a
// launch + 4 allocs per layer — inlining the scan makes the dedup's fixed cost ~0.
// gemma4 GELU CSR twin (verify dedup): owner-scan body of _csr_iq4 with the gelu epilogue.
extern "C" __global__ void moe_gate_up_gelu8_dev_q8_csr(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used, int n_pairs) {
    int pself = blockIdx.y;
    int ex = sel[pself];
    for (int q = 0; q < pself; q++) if (sel[q] == ex) return;
    int plist[CSR_MAXP];
    int np = 0;
    for (int q = pself; q < n_pairs; q++) if (sel[q] == ex && np < CSR_MAXP) plist[np++] = q;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg[CSR_MAXP], accu[CSR_MAXP];
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        expg wg = exp_decode_g_cached(qt_g, grow, g);
        expg wu = exp_decode_g_cached(qt_u, urow, g);
        #pragma unroll
        for (int i = 0; i < CSR_MAXP; i++) {
            if (i < np) {
                int tok = plist[i] / n_used;
                const signed char* aqb = aq + (size_t)tok * in_f + (size_t)g * 32;
                float d8 = ad[(size_t)tok * nsb + g];
                accg[i] = exp_dot_acc_cached(qt_g, wg, aqb, d8, accg[i]);
                accu[i] = exp_dot_acc_cached(qt_u, wu, aqb, d8, accu[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) {
        if (i < np) {
            float sg = warp_reduce_sum(accg[i]);
            float su = warp_reduce_sum(accu[i]);
            if (lane == 0) {
                float x = sg;
                float th = tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x));
                act[(size_t)plist[i] * n_ff + o] = 0.5f * x * (1.0f + th) * su;
            }
        }
    }
}

// NVFP4 cached-weight dot: EXACTLY expert_dot_nvfp4_g's body — same partial accumulation,
// same d8-close as a RETURNED product consumed by `acc += <ret>` — with the weight decode
// (lookup ints + UE4M3 sub-scales) supplied from registers instead of re-derived per call.
// STRUCTURE MATTERS (gate2 defect, 2026-08-21): the first CSR-NVFP4 kernel folded the
// d8-close into the accumulate in-body (`acc += d8*pg` -> fmaf) while the rows program
// materializes the helper's return value (mul, then add) — last-ULP drift on 11041/32768
// ACT elements at t=8, which decode-batch-gate2 caught as a batch-composition dependence.
// Keeping the helper-call shape (`acc += nvfp4_dot_cached(..)`) reproduces the rows
// codegen; MEMRA_MOE_CSR=2 byte-compares to enforce it.
__device__ __forceinline__ float nvfp4_dot_cached(const int* __restrict__ wv,
                                                  const float* __restrict__ sf,
                                                  const signed char* __restrict__ aqb,
                                                  float d8) {
    const int* aq4 = (const int*)aqb;
    float partial = 0.0f;
    #pragma unroll
    for (int sl = 0; sl < 2; sl++) {
        int base = sl * 4;
        int sumi = 0;
        sumi = dp4a(wv[base + 0], aq4[base + 0], sumi);
        sumi = dp4a(wv[base + 1], aq4[base + 1], sumi);
        sumi = dp4a(wv[base + 2], aq4[base + 2], sumi);
        sumi = dp4a(wv[base + 3], aq4[base + 3], sumi);
        partial += sf[sl] * (float)sumi;
    }
    return d8 * partial;
}

// NVFP4 CSR owner-scan twin (lane/moebatch-q35moe, 2026-08-21): same dedup skeleton as
// _csr_iq4, weight decode cached per (expert, o, group) and reused across the expert's
// (token, slot) pairs. Per-pair arithmetic = nvfp4_dot_cached (expert_dot_nvfp4_g's exact
// body over the cached decode); MEMRA_MOE_CSR=2 byte-compares to enforce bit-identity.
extern "C" __global__ void moe_gate_up_silu8_dev_q8_csr_nvfp4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used, int n_pairs) {
    int pself = blockIdx.y;
    int ex = sel[pself];
    for (int q = 0; q < pself; q++) if (sel[q] == ex) return;   // duplicate: owner is earlier
    int plist[CSR_MAXP];
    int np = 0;
    for (int q = pself; q < n_pairs; q++) if (sel[q] == ex && np < CSR_MAXP) plist[np++] = q;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg[CSR_MAXP], accu[CSR_MAXP];
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        // cached NVFP4 decode: group g = half of a 64-elem/36B block; two 16-elem
        // sub-blocks, each 4 packed lookup ints + one UE4M3 sub-scale.
        int sblk = g >> 1, s0 = (g & 1) * 2;
        const unsigned char* bg = grow + (long)sblk * 36;
        const unsigned char* bu = urow + (long)sblk * 36;
        int wgv[8], wuv[8];
        float sgf[2], suf[2];
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qsg = bg + 4 + s * 8;
            const unsigned char* qsu = bu + 4 + s * 8;
            int2 vga = get_int_from_table_16_d(get_int_b4(qsg), kvalues_mxfp4_d);
            int2 vgb = get_int_from_table_16_d(get_int_b4(qsg + 4), kvalues_mxfp4_d);
            int2 vua = get_int_from_table_16_d(get_int_b4(qsu), kvalues_mxfp4_d);
            int2 vub = get_int_from_table_16_d(get_int_b4(qsu + 4), kvalues_mxfp4_d);
            wgv[sl * 4 + 0] = vga.x; wgv[sl * 4 + 1] = vgb.x;
            wgv[sl * 4 + 2] = vga.y; wgv[sl * 4 + 3] = vgb.y;
            wuv[sl * 4 + 0] = vua.x; wuv[sl * 4 + 1] = vub.x;
            wuv[sl * 4 + 2] = vua.y; wuv[sl * 4 + 3] = vub.y;
            sgf[sl] = ue4m3_to_f32_d(bg[s]);
            suf[sl] = ue4m3_to_f32_d(bu[s]);
        }
        #pragma unroll
        for (int i = 0; i < CSR_MAXP; i++) {
            if (i < np) {
                int tok = plist[i] / n_used;
                const signed char* aqb = aq + (size_t)tok * in_f + (size_t)g * 32;
                float d8 = ad[(size_t)tok * nsb + g];
                accg[i] += nvfp4_dot_cached(wgv, sgf, aqb, d8);
                accu[i] += nvfp4_dot_cached(wuv, suf, aqb, d8);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) {
        if (i < np) {
            float sg = warp_reduce_sum(accg[i]);
            float su = warp_reduce_sum(accu[i]);
            if (lane == 0)
                act[(size_t)plist[i] * n_ff + o] = (sg / (1.0f + expf(-sg))) * su;
        }
    }
}

extern "C" __global__ void moe_gate_up_silu8_dev_q8_csr_iq4(
        const unsigned long long* __restrict__ table, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ act,
        int in_f, int n_ff, int n_expert, int qt_g, int qt_u, long rb_g, long rb_u,
        int n_used, int n_pairs) {
    int pself = blockIdx.y;
    int ex = sel[pself];
    for (int q = 0; q < pself; q++) if (sel[q] == ex) return;   // duplicate: owner is earlier
    int plist[CSR_MAXP];
    int np = 0;
    for (int q = pself; q < n_pairs; q++) if (sel[q] == ex && np < CSR_MAXP) plist[np++] = q;
    int o = blockIdx.x;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = (const unsigned char*)table[ex] + (long)o * rb_g;
    const unsigned char* urow = (const unsigned char*)table[n_expert + ex] + (long)o * rb_u;
    float accg[CSR_MAXP], accu[CSR_MAXP];
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int g = lane; g < nsb; g += 32) {
        expg wg = exp_decode_g_cached(qt_g, grow, g);
        expg wu = exp_decode_g_cached(qt_u, urow, g);
        #pragma unroll
        for (int i = 0; i < CSR_MAXP; i++) {
            if (i < np) {
                int tok = plist[i] / n_used;
                const signed char* aqb = aq + (size_t)tok * in_f + (size_t)g * 32;
                float d8 = ad[(size_t)tok * nsb + g];
                accg[i] = exp_dot_acc_cached(qt_g, wg, aqb, d8, accg[i]);
                accu[i] = exp_dot_acc_cached(qt_u, wu, aqb, d8, accu[i]);
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < CSR_MAXP; i++) {
        if (i < np) {
            float sg = warp_reduce_sum(accg[i]);
            float su = warp_reduce_sum(accu[i]);
            if (lane == 0)
                act[(size_t)plist[i] * n_ff + o] = (sg / (1.0f + expf(-sg))) * su;
        }
    }
}


// ===== Q4_0 decode MMVQ (gemma-4 QAT GGUF, 2026-07-10). Block = 18B per 32 elems: fp16 d +
// 16B nibbles (elem i = low nibble of byte i for i<16, high nibble of byte i-16 for i>=16).
// value = d*(q-8); with per-32 q8_1 activations (aq int8 + ad group scale):
//   dot_g = d * (sumi - 8*sums) * d8, sumi = dp4a(q, a), sums = dp4a(1, a) — exact ints,
// one float expression per group (the q4_K vendoring pattern; llama vec_dot_q4_0_q8_1 math).
// Q4_0 mr2 (gemma trunk lane): 2 rows/warp — the activation int4 loads AND the row-independent
// ones-sum (sums) are computed ONCE per group and reused across both rows' dp4a chains.
// Per-row accumulation chain identical to qmatvec_q4_0_mmvq (bit-identical per row).
__device__ __forceinline__ void q4_0_mmvq_row2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f, acc1 = 0.0f;
    bool two = (o0 + 1) < out_f;
    const unsigned char* w0 = W + (long)o0 * row_bytes;
    const unsigned char* w1 = W + (long)(o0 + 1) * row_bytes;
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            const unsigned char* b = w0 + (long)g * 18;
            float d4 = half_to_float(*(const unsigned short*)b);
            const unsigned char* qs = b + 2;
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                sumi = dp4a((int)(raw & 0x0F0F0F0Fu), aq4[k], sumi);
                sumi = dp4a((int)((raw >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
        if (two) {
            const unsigned char* b = w1 + (long)g * 18;
            float d4 = half_to_float(*(const unsigned short*)b);
            const unsigned char* qs = b + 2;
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t raw; memcpy(&raw, qs + 4 * k, 4);
                sumi = dp4a((int)(raw & 0x0F0F0F0Fu), aq4[k], sumi);
                sumi = dp4a((int)((raw >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc1 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (two) acc1 = warp_reduce_sum(acc1);
    if (lane == 0) {
        y[(size_t)t * out_f + o0] = acc0;
        if (two) y[(size_t)t * out_f + o0 + 1] = acc1;
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_mr2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_row2(W, aq, ad, y, in_f, out_f, m, row_bytes,
                   (blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2, blockIdx.y);
}
// ----- FUSED Q4_0 m=1 PAIR/TRIPLE (gemma: gate+up / wq+wk+wv share the quantized input).
// Block-offset partition over the mr2 row pairs; per (tensor,row) chain = mr2 VERBATIM. -----
extern "C" __global__ void qmatvec_q4_0_mmvq_fused2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long rb0, long rb1) {
    int pairs0 = (out0 + 1) / 2;
    int nb0 = (pairs0 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    if (b < nb0) {
        q4_0_mmvq_row2(W0, aq, ad, y0, in_f, out0, 1, rb0,
                       (b * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
    } else {
        b -= nb0;
        q4_0_mmvq_row2(W1, aq, ad, y1, in_f, out1, 1, rb1,
                       (b * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused3(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long rb0, long rb1, long rb2) {
    int nb0 = ((out0 + 1) / 2 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int nb1 = ((out1 + 1) / 2 + MEMRA_MMVQ_ROWS - 1) / MEMRA_MMVQ_ROWS;
    int b = blockIdx.x;
    const unsigned char* W; float* y; int out_f; long rb;
    if (b < nb0)            { W = W0; y = y0; out_f = out0; rb = rb0; }
    else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; rb = rb1; b -= nb0; }
    else                    { W = W2; y = y2; out_f = out2; rb = rb2; b -= nb0 + nb1; }
    q4_0_mmvq_row2(W, aq, ad, y, in_f, out_f, 1, rb,
                   (b * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2, 0);
}

// ----- Q4_0 SPLIT-PLANE (rp) MIRROR twins (2026-07-10, the verify-trunk/decode-trunk 18B-
// straggle cure — A6's NVFP4 layout applied to Q4_0). Mirror layout in ONE buffer:
// qs plane [out_f x nblk x 16B] (16B-aligned -> ONE LDG.128/block) then d plane
// [out_f x nblk x 2B] (dense u16). Raw GGUF bytes stay resident for prefill/gemm/Stage-A;
// decode-class kernels read the mirror. Every twin's per (token,row) float chain
// (d4*(sumi-8*sums)*d8 in ascending-g order) is VERBATIM its block-layout source kernel —
// the standing batched==mr2 bit-identity contract extends to the rp family (kernel gates +
// VERIFY-GATE + run-spec battery arbitrate). Microprobe: m=1 1.34x, m=3 1.17x, m=4 1.13x,
// bitwise-exact (rp_q4_probe). -----
// ===== Q8_0 split-plane (rp) decode twins — the H100 coalescing fix (2026-07-26 ncu:
// the 34B-stride GGUF layout holds Max Bandwidth at 41-46% with Mem Busy 66-76% —
// misaligned 4B weight loads waste sectors; split planes make every weight load an
// aligned 16B ldcs). Per (row, block) the dp4a int inputs are the SAME BYTES as
// get_int_b2 on the GGUF layout, same k order, same accumulate -> BIT-IDENTICAL. =====
__device__ __forceinline__ void q8_0_rp_planes(const unsigned char* W, int out_f,
                                               int o, int nblk,
                                               const unsigned char** wq,
                                               const unsigned short** wd) {
    // qs plane = out_f*nblk*32 bytes, then the half d plane (the q4_0/NVFP4 rp convention).
    *wq = W + ((size_t)o * nblk) * 32;
    *wd = (const unsigned short*)(W + (size_t)out_f * nblk * 32) + (size_t)o * nblk;
}
// device-side build: one thread per q8_0 block, pure byte permutation.
extern "C" __global__ void q8_0_split_rp_build(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int out_f, int nblk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_f * nblk) return;
    const unsigned char* b = src + (size_t)i * 34;
    long qplane = (long)out_f * nblk * 32;
    unsigned char* q = dst + (size_t)i * 32;
    #pragma unroll
    for (int k = 0; k < 32; k++) q[k] = b[2 + k];
    dst[qplane + (size_t)i * 2 + 0] = b[0];
    dst[qplane + (size_t)i * 2 + 1] = b[1];
}
// rp twin of qmatvec_q8_0_mmvq: same grid/warp mapping, aligned int4 weight loads.
extern "C" __global__ void qmatvec_q8_0_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    MEMRA_PDL_ENTRY();
    (void)row_bytes;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
// Small-shape rp twin: 2 warps/block (64 threads) doubles the grid so sub-wave shapes
// (attn qkv out_f=2048: 512 blocks / 132 SMs ~ 0.97 waves at the 4-warp block) fill the
// machine. Per-row program IDENTICAL to qmatvec_q8_0_mmvq_rp (one warp per row, same
// block walk) -> bit-identical; only the block geometry changes. Dispatch: rp && the
// 4-warp grid would be sub-wave (out_f/4 < 4*SMs).
extern "C" __global__ void qmatvec_q8_0_mmvq_rp_g2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o = blockIdx.x * 2 + threadIdx.y;      // 2 rows per block
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// ===== Q8_0 rp + cp.async ring (rpca recipe, H100 m=1 latency fix — 2026-07-26 ncu:
// issue every 6-9 cycles, 0.16-0.29 eligible warps, long-scoreboard on the weight
// stream). Window per 32-block iteration: 1024B quants + 64B scales, staged through a
// per-warp smem ring so the dp4a loop reads smem while cp.async prefetches STAGES-1
// windows ahead. Accumulation order per (row, block) IDENTICAL to qmatvec_q8_0_mmvq_rp
// -> bit-identical (k-split stays banned; this hides latency without reordering). =====
__device__ __forceinline__ void ca_issue_window_q8(unsigned char* dst,
        const unsigned char* qsrc, const unsigned char* ssrc, int lane) {
    cp_async16_g(dst + lane * 16, qsrc + lane * 16);                     // quant lo: 512B
    cp_async16_g(dst + 512 + lane * 16, qsrc + 512 + lane * 16);        // quant hi: 512B
    if (lane < 4) cp_async16_g(dst + 1024 + lane * 16, ssrc + lane * 16); // scales: 64B
}
extern "C" __global__ void qmatvec_q8_0_mmvq_rpca(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    constexpr int STAGES = 3;
    constexpr int WIN = 1024 + 64;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    int niter = nblk >> 5;                     // dispatch gate: nblk % 32 == 0
    const unsigned char* qplane = W + ((size_t)o * nblk) * 32;
    const unsigned char* splane = W + (size_t)out_f * nblk * 32 + (size_t)o * nblk * 2;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    __shared__ __align__(16) unsigned char smw[MEMRA_MMVQ_ROWS][STAGES][WIN];
    unsigned char (*ring)[WIN] = smw[threadIdx.y];
    float acc = 0.0f;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < niter) {
            ca_issue_window_q8(ring[s], qplane + (size_t)s * 1024, splane + (size_t)s * 64, lane);
        }
        cp_async_commit();
    }
    for (int it = 0; it < niter; it++) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        const unsigned char* wnd = ring[it % STAGES];
        int blk = it * 32 + lane;
        const int4* wq16 = (const int4*)(wnd + lane * 32);
        int4 w01 = wq16[0], w23 = wq16[1];
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(*(const unsigned short*)(wnd + 1024 + lane * 2));
        const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
        int itn = it + STAGES - 1;
        if (itn < niter) {
            ca_issue_window_q8(ring[itn % STAGES], qplane + (size_t)itn * 1024,
                               splane + (size_t)itn * 64, lane);
        }
        cp_async_commit();
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// m=1 two-rows-per-warp rp twin (the q4_0 mr2 recipe: doubles per-warp bytes in flight —
// the m=1 latency lever; same per-row dp4a order as qmatvec_q8_0_mmvq_rp -> bit-identical).
extern "C" __global__ void qmatvec_q8_0_mmvq_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    MEMRA_PDL_ENTRY();
    (void)row_bytes;
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y) * 2;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    bool two = (o0 + 1) < out_f;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q8_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q8_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nblk;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[blk];
        {
            int4 w01 = __ldcs((const int4*)(wq0 + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq0 + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc0 += half_to_float(wd0[blk]) * d8 * (float)sumi;
        }
        if (two) {
            int4 w01 = __ldcs((const int4*)(wq1 + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq1 + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc1 += half_to_float(wd1[blk]) * d8 * (float)sumi;
        }
    }
    float v0 = warp_reduce_sum(acc0);
    if (lane == 0) y[(size_t)t * out_f + o0] = v0;
    if (two) {
        float v1 = warp_reduce_sum(acc1);
        if (lane == 0) y[(size_t)t * out_f + o0 + 1] = v1;
    }
}

// batched rp row body + wrappers (mirror of q8_0_mmvq_batched_row with plane loads).
template<int MCOLS>
__device__ __forceinline__ void q8_0_mmvq_batched_row_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, int o) {
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * (in_f / 32) + blk] * (float)sumi;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q8_0_mmvq_batched_row_rp<2>(W, aq, ad, y, in_f, out_f, m,
                                blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q8_0_mmvq_batched_row_rp<4>(W, aq, ad, y, in_f, out_f, m,
                                blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q8_0_mmvq_batched_row_rp<8>(W, aq, ad, y, in_f, out_f, m,
                                blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}
extern "C" __global__ void qmatvec_q8_0_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q8_0_mmvq_batched_row_rp<16>(W, aq, ad, y, in_f, out_f, m,
                                 blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y);
}

__device__ __forceinline__ void q4_0_rp_planes(const unsigned char* W, int out_f,
                                               int o, int nblk,
                                               const unsigned char** wq,
                                               const unsigned short** wd) {
    // planes derived from shape (the NVFP4 rp convention): qs plane is out_f*nblk*16 bytes.
    *wq = W + ((size_t)o * nblk) * 16;
    *wd = (const unsigned short*)(W + (size_t)out_f * nblk * 16) + (size_t)o * nblk;
}
// device-side mirror build: one thread per q4_0 block, pure byte permutation.
extern "C" __global__ void q4_0_split_rp_build(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int out_f, int nblk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_f * nblk) return;
    const unsigned char* b = src + (size_t)i * 18;
    long qplane = (long)out_f * nblk * 16;
    unsigned char* q = dst + (size_t)i * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) q[k] = b[2 + k];
    dst[qplane + (size_t)i * 2 + 0] = b[0];
    dst[qplane + (size_t)i * 2 + 1] = b[1];
}
// m=1 two-rows-per-warp twin (the mr2 body with split loads).
__device__ __forceinline__ void q4_0_mmvq_row2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    (void)row_bytes;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f, acc1 = 0.0f;
    bool two = (o0 + 1) < out_f;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nsb, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nsb, &wq1, &wd1);
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)g * 16));
            float d4 = half_to_float(wd0[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
        if (two) {
            int4 qv = __ldcs((const int4*)(wq1 + (size_t)g * 16));
            float d4 = half_to_float(wd1[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc1 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (two) acc1 = warp_reduce_sum(acc1);
    if (lane == 0) {
        y[(size_t)t * out_f + o0] = acc0;
        if (two) y[(size_t)t * out_f + o0 + 1] = acc1;
    }
}
// mr1 split-plane twin (E4B mr2-efficiency probe, 2026-07-13): ONE row per warp — 2x the
// blocks of mr2 for tall-input/short-output shapes (ffn_down 10240->2560 runs 69% of the
// byte floor under mr2's 4-wave grid; more blocks = more latency hiding + finer tail).
// Per-row dot = q4_0_mmvq_row2_rp's acc0 path VERBATIM (bit-identical per row).
__device__ __forceinline__ void q4_0_mmvq_row1_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes, int o0, int t) {
    (void)row_bytes;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const signed char* arow = aq + (size_t)t * in_f;
    const float* adrow = ad + (size_t)t * nsb;
    float acc0 = 0.0f;
    const unsigned char* wq0; const unsigned short* wd0;
    q4_0_rp_planes(W, out_f, o0, nsb, &wq0, &wd0);
    for (int g = lane; g < nsb; g += 32) {
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float d8 = adrow[g];
        int sums = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, aq4[k], sums);
        {
            // W plane rides STREAMING loads (__ldcs = evict-first, 2026-07-14 duty arc):
            // decode weights are single-use per token — evict-normal W lines thrash the
            // L2 share the activation re-reads (every warp) and KV live on. Same bytes,
            // same values — bit-identical.
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)g * 16));
            float d4 = half_to_float(wd0[g]);
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(qk[k] & 0x0F0F0F0F, aq4[k], sumi);
                sumi = dp4a((int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu), aq4[4 + k], sumi);
            }
            acc0 += d4 * (float)(sumi - 8 * sums) * d8;
        }
    }
    acc0 = warp_reduce_sum(acc0);
    if (lane == 0) y[(size_t)t * out_f + o0] = acc0;
}
extern "C" __global__ void qmatvec_q4_0_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    MEMRA_PDL_ENTRY();
    q4_0_mmvq_row1_rp(W, aq, ad, y, in_f, out_f, m, row_bytes,
                      blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y, blockIdx.y);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_row2_rp(W, aq, ad, y, in_f, out_f, m, row_bytes,
                      (blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2, blockIdx.y);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long rb0, long rb1) {
    // GRID-STRIDE (2026-07-12, 31B wave-quantization fix): the flat launch ran 5.46 waves/SM
    // — the 0.46 tail wave idled ~54% of the card for ~8% of the duration (ncu, DRAM capped
    // at 92.9%). Striding distributes the tail one iteration wide across every SM. Per-row
    // math and each warp's row assignment order are untouched — bit-identical outputs.
    const int rpb = (int)blockDim.y;   // host MEMRA_FUSED_RPB knob (tail-wave granularity)
    int pairs0 = (out0 + 1) / 2;
    int nb0 = (pairs0 + rpb - 1) / rpb;
    int nb1 = ((out1 + 1) / 2 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1; vb += gridDim.x) {
        int b = vb;
        if (b < nb0) {
            q4_0_mmvq_row2_rp(W0, aq, ad, y0, in_f, out0, 1, rb0,
                              (b * rpb + (int)threadIdx.y) * 2, 0);
        } else {
            b -= nb0;
            q4_0_mmvq_row2_rp(W1, aq, ad, y1, in_f, out1, 1, rb1,
                              (b * rpb + (int)threadIdx.y) * 2, 0);
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long rb0, long rb1, long rb2) {
    const int rpb = (int)blockDim.y;   // host MEMRA_FUSED_RPB knob
    int nb0 = ((out0 + 1) / 2 + rpb - 1) / rpb;
    int nb1 = ((out1 + 1) / 2 + rpb - 1) / rpb;
    int nb2 = ((out2 + 1) / 2 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1 + nb2; vb += gridDim.x) {
        int b = vb;
        const unsigned char* W; float* y; int out_f; long rb;
        if (b < nb0)            { W = W0; y = y0; out_f = out0; rb = rb0; }
        else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; rb = rb1; b -= nb0; }
        else                    { W = W2; y = y2; out_f = out2; rb = rb2; b -= nb0 + nb1; }
        q4_0_mmvq_row2_rp(W, aq, ad, y, in_f, out_f, 1, rb,
                          (b * rpb + (int)threadIdx.y) * 2, 0);
    }
}
// batched (weight-read-once, m<=MCOLS) twin — body mirrors q4_0_mmvq_batched.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q4_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 qv = __ldcs((const int4*)(wq + (size_t)blk * 16));
        float d4 = half_to_float(wd[blk]);
        int lo[4], hi[4];
        const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            lo[k] = qk[k] & 0x0F0F0F0F;
            hi[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13, the L1TEX fix): same values, same dp4a order.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            const int al[4] = { a01.x, a01.y, a01.z, a01.w };
            const int ah[4] = { a23.x, a23.y, a23.z, a23.w };
            int sumi = 0, sums = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = dp4a(lo[k], al[k], sumi);
                sumi = dp4a(hi[k], ah[k], sumi);
                sums = dp4a(0x01010101, al[k], sums);
                sums = dp4a(0x01010101, ah[k], sums);
            }
            acc[c] += d4 * (float)(sumi - 8 * sums) * ad[(size_t)c * nblk + blk];
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// batched 2-rows-per-warp twin — body mirrors q4_0_mmvq_batched_mr2 (row-shared activation
// int4 loads + ones-sums, per-row chains in the same order).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_rp_bx(
        int bx, const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o0 = (bx * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = __ldcs((const int4*)(wq1 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            // int4-vectorized (2026-07-13): the 8 scalar int loads were 4x the L1TEX
            // transactions of the t=1 walk's two 16B loads — L1TEX measured 90% saturated
            // (the b-tier limiter). Same bytes, same order per k — bit-identical.
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)c * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0 = dp4a(lo0[k], a[k], s0);
                s0 = dp4a(hi0[k], a[4 + k], s0);
            }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(blockIdx.x, W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- BATCHED FUSED2 (2026-07-13, the megakernel tail-fill mechanism in microcosm):
// ONE launch covers gate rows THEN up rows via grid segmentation — the up segment's
// blocks schedule onto SMs as the gate segment drains, filling the per-launch tail
// waves that the 6-falsification b-tier plateau evidence points at. Bit-identical per
// row (the exact mr2_rp program; only the block->work mapping changes).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_f2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    int nb0 = (out0 + 2 * MEMRA_MMVQ_ROWS - 1) / (2 * MEMRA_MMVQ_ROWS);
    if ((int)blockIdx.x < nb0) {
        q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(blockIdx.x, W0, aq, ad, y0, in_f, out0, m, row_bytes);
    } else {
        q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(blockIdx.x - nb0, W1, aq, ad, y1, in_f, out1, m, row_bytes);
    }
}
// f3 twin: three segments (the verify qkv triple on swa layers; globals fuse q,k via f2).
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_f3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    int nb0 = (out0 + 2 * MEMRA_MMVQ_ROWS - 1) / (2 * MEMRA_MMVQ_ROWS);
    int nb1 = (out1 + 2 * MEMRA_MMVQ_ROWS - 1) / (2 * MEMRA_MMVQ_ROWS);
    int bx = blockIdx.x;
    if (bx < nb0) {
        q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(bx, W0, aq, ad, y0, in_f, out0, m, row_bytes);
    } else if (bx < nb0 + nb1) {
        q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(bx - nb0, W1, aq, ad, y1, in_f, out1, m, row_bytes);
    } else {
        q4_0_mmvq_batched_mr2_rp_bx<MCOLS>(bx - nb0 - nb1, W2, aq, ad, y2, in_f, out2, m, row_bytes);
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_f3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q4_0_mmvq_batched_f3_rp<2>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_f3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q4_0_mmvq_batched_f3_rp<4>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_f3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, int m, long row_bytes) {
    q4_0_mmvq_batched_f3_rp<8>(W0, W1, W2, aq, ad, y0, y1, y2, in_f, out0, out1, out2, m, row_bytes);
}

extern "C" __global__ void qmatvec_q4_0_mmvq_b2_f2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q4_0_mmvq_batched_f2_rp<2>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_f2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q4_0_mmvq_batched_f2_rp<4>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_f2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, long row_bytes) {
    q4_0_mmvq_batched_f2_rp<8>(W0, W1, aq, ad, y0, y1, in_f, out0, out1, m, row_bytes);
}

extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}

// ---- Q4_0 M-SPLIT r2 twin (2026-07-13, the 31B depth-verify occupancy fix): the b8_r2_rp
// kernel is REGISTER-CHOKED (ncu: 72 regs, occupancy capped at 7 blocks, warps 47-55%,
// DRAM 25-45% of wall) — the acc[2][MCOLS] array is the pressure. The rpms pattern (NVFP4,
// 2026-07-06) splits the M columns across a warp PAIR: both warps walk the FULL k-range of
// the SAME 2 rows, each owning half the columns — acc drops to [2][MCOLS/2], grid.x doubles
// (block (32,4) = 2 pairs x 2 rows), and every (token,row) dot keeps the reference per-lane
// serial chain + warp_reduce_sum -> BIT-IDENTICAL to _r2_rp (column partition, not k-order;
// the rpks k-order lesson). The twin warp re-reads the same weight int4s in near-lockstep ->
// L1 serves the second copy.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    constexpr int CH = MCOLS / 2;           // columns per warp
    int pair = (int)threadIdx.y >> 1;       // 0..1: which 2-row group of the block
    int kc   = (int)threadIdx.y & 1;        // 0..1: which column half
    int o0 = (blockIdx.x * 2 + pair) * 2;
    if (o0 >= out_f) return;
    int c0 = kc * CH;
    if (c0 >= m) return;                    // whole column half masked
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[CH], acc1[CH];
    #pragma unroll
    for (int c = 0; c < CH; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = __ldcs((const int4*)(wq1 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < CH; c++) {
            int col = c0 + c;
            if (col >= m) break;
            const signed char* arow = aq + (size_t)col * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int a[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = ad[(size_t)col * nblk + blk];
            int s0 = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0 = dp4a(lo0[k], a[k], s0);
                s0 = dp4a(hi0[k], a[4 + k], s0);
            }
            acc0[c] += d40 * (float)(s0 - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < CH; c++) {
        int col = c0 + c;
        if (col >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)col * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)col * out_f + o0 + 1] = a1;
        }
    }
}
// ---- Q4_0 LOAD-AHEAD r2 twin (2026-07-13, same target as the smem-slab probe): the
// c-loop's activation loads are serial 32B L2 dependency chains (long_scoreboard 42.5%).
// This twin double-buffers the per-column activation int4s in registers: column c+1's
// loads are ISSUED before column c's dp4a chain executes, so the load latency overlaps
// compute instead of stalling it. No smem, no syncs. Math order per (row,col) unchanged
// (bit-identical); +8 int registers.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2;
    if (o0 >= out_f) return;
    bool two = (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int nblk = in_f / 32;
    const unsigned char* wq0; const unsigned short* wd0;
    const unsigned char* wq1; const unsigned short* wd1;
    q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
    q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int blk = lane; blk < nblk; blk += 32) {
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = __ldcs((const int4*)(wq1 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        // prime column 0's loads
        int a_nxt[8]; float d8_nxt = 0.0f;
        {
            const int* aq4 = (const int*)(aq + (size_t)0 * in_f + (size_t)blk * 32);
            #pragma unroll
            for (int k = 0; k < 8; k++) a_nxt[k] = aq4[k];
            d8_nxt = ad[(size_t)0 * nblk + blk];
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            int a[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) a[k] = a_nxt[k];
            float d8 = d8_nxt;
            if (c + 1 < MCOLS && c + 1 < m) {   // issue c+1's loads BEFORE computing c
                const int* aq4 = (const int*)(aq + (size_t)(c + 1) * in_f + (size_t)blk * 32);
                #pragma unroll
                for (int k = 0; k < 8; k++) a_nxt[k] = aq4[k];
                d8_nxt = ad[(size_t)(c + 1) * nblk + blk];
            }
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            int s0i = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0i = dp4a(lo0[k], a[k], s0i);
                s0i = dp4a(hi0[k], a[4 + k], s0i);
            }
            acc0[c] += d40 * (float)(s0i - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2la_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_la_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
// ---- Q4_0 SMEM-SLAB r2 twin (2026-07-13, the 31B depth-verify latency fix): the r2 b-tiers
// are LATENCY-bound (ncu: long_scoreboard 42.5% of stalls, DRAM 22-45%, L2 hit 39%) — the
// c-loop's per-column activation loads are 8 serial 32B L2 dependency chains per warp
// iteration. This twin stages a 32-k-block SLAB of ALL columns' activations (+ q8 scales)
// into shared memory cooperatively (one coalesced pass per block instead of per-warp chains),
// then each lane consumes ITS k-block of the slab from smem — the inner loop's only global
// stream left is the weight planes. BIT-IDENTITY: per-(row,col) dot chain, lane->k-block
// mapping, and warp_reduce order are the r2_rp body's verbatim; only the activation LOAD
// SOURCE changes (same bytes). smem = MCOLS*(1KB act + 128B scales) (+pad), single-buffered.
template<int MCOLS>
__device__ __forceinline__ void q4_0_mmvq_batched_mr2_sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o0 = (blockIdx.x * MEMRA_MMVQ_ROWS + (int)threadIdx.y) * 2;
    bool row_ok = o0 < out_f;
    bool two = row_ok && (o0 + 1) < out_f;
    int lane = threadIdx.x;
    int tid = (int)threadIdx.y * 32 + lane;
    int nblk = in_f / 32;
    const unsigned char* wq0 = nullptr; const unsigned short* wd0 = nullptr;
    const unsigned char* wq1 = nullptr; const unsigned short* wd1 = nullptr;
    if (row_ok) {
        q4_0_rp_planes(W, out_f, o0, nblk, &wq0, &wd0);
        q4_0_rp_planes(W, out_f, o0 + 1, nblk, &wq1, &wd1);
    }
    extern __shared__ int sm_slab[];                    // [MCOLS][32 blk][9 int] (8 + pad —
                                                        // stride 8 = 4-way bank conflicts)
    float* sm_d8 = (float*)(sm_slab + MCOLS * 32 * 9);  // [MCOLS][32] scales
    float acc0[MCOLS], acc1[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) { acc0[c] = 0.0f; acc1[c] = 0.0f; }
    for (int s0 = 0; s0 < nblk; s0 += 32) {
        int slab = min(32, nblk - s0);
        __syncthreads();
        // cooperative stage: MCOLS*slab*8 ints + MCOLS*slab scales, 128 threads.
        for (int i = tid; i < MCOLS * slab * 8; i += 128) {
            int c = i / (slab * 8);
            int r = i - c * (slab * 8);        // blk_in_slab*8 + word
            if (c < m) {
                const int* arow = (const int*)(aq + (size_t)c * in_f + (size_t)s0 * 32);
                sm_slab[(c * 32 + r / 8) * 9 + (r & 7)] = arow[r];
            }
        }
        for (int i = tid; i < MCOLS * slab; i += 128) {
            int c = i / slab;
            int b = i - c * slab;
            if (c < m) sm_d8[c * 32 + b] = ad[(size_t)c * nblk + s0 + b];
        }
        __syncthreads();
        int blk = s0 + lane;
        if (!row_ok || lane >= slab) continue;
        int lo0[4], hi0[4], lo1[4], hi1[4];
        {
            int4 qv = __ldcs((const int4*)(wq0 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo0[k] = qk[k] & 0x0F0F0F0F;
                hi0[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        if (two) {
            int4 qv = __ldcs((const int4*)(wq1 + (size_t)blk * 16));
            const int qk[4] = { qv.x, qv.y, qv.z, qv.w };
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                lo1[k] = qk[k] & 0x0F0F0F0F;
                hi1[k] = (int)(((uint32_t)qk[k] >> 4) & 0x0F0F0F0Fu);
            }
        }
        float d40 = half_to_float(wd0[blk]);
        float d41 = two ? half_to_float(wd1[blk]) : 0.0f;
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const int* aq4 = &sm_slab[(c * 32 + lane) * 9];
            int a[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) a[k] = aq4[k];
            int sums = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sums = dp4a(0x01010101, a[k], sums);
            float d8 = sm_d8[c * 32 + lane];
            int s0i = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s0i = dp4a(lo0[k], a[k], s0i);
                s0i = dp4a(hi0[k], a[4 + k], s0i);
            }
            acc0[c] += d40 * (float)(s0i - 8 * sums) * d8;
            if (two) {
                int s1 = 0;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    s1 = dp4a(lo1[k], a[k], s1);
                    s1 = dp4a(hi1[k], a[4 + k], s1);
                }
                acc1[c] += d41 * (float)(s1 - 8 * sums) * d8;
            }
        }
    }
    if (!row_ok) return;
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a0 = warp_reduce_sum(acc0[c]);
        if (lane == 0) y[(size_t)c * out_f + o0] = a0;
        if (two) {
            float a1 = warp_reduce_sum(acc1[c]);
            if (lane == 0) y[(size_t)c * out_f + o0 + 1] = a1;
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2sm_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ ad_q,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_sm_rp<8>(W, ad_q, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b2_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<2>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b4_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<4>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b8_r2ms_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_ms_rp<8>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_rp<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}
extern "C" __global__ void qmatvec_q4_0_mmvq_b16_r2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    q4_0_mmvq_batched_mr2_rp<16>(W, aq, ad, y, in_f, out_f, m, row_bytes);
}


extern "C" __global__ void qmatvec_q4_0_mmvq(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        const unsigned char* b = wrow + (long)g * 18;
        float d4 = half_to_float(*(const unsigned short*)b);
        const unsigned char* qs = b + 2;
        const int* aq4 = (const int*)(arow + (size_t)g * 32);
        int sumi = 0, sums = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            uint32_t raw;
            memcpy(&raw, qs + 4 * k, 4);
            int lo = (int)(raw & 0x0F0F0F0Fu);
            int hi = (int)((raw >> 4) & 0x0F0F0F0Fu);
            int a_lo = aq4[k];
            int a_hi = aq4[4 + k];
            sumi = dp4a(lo, a_lo, sumi);
            sumi = dp4a(hi, a_hi, sumi);
            sums = dp4a(0x01010101, a_lo, sums);
            sums = dp4a(0x01010101, a_hi, sums);
        }
        acc += d4 * (float)(sumi - 8 * sums) * adrow[g];
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// mr1 twins of the t=1 fused pair/triple (2026-07-14): the singles took the mr1
// one-row-per-warp upgrade 2026-07-13 (+3.75% E4B / +0.9% 31B) but the fused t=1 kernels
// stayed on the mr2 two-serial-rows walk — the DRAM-duty map reads fused3 at 57% and the
// singles at 75% while fused2's bigger grid reads 86%. Per-row dot = q4_0_mmvq_row1_rp
// VERBATIM (bit-identical per row); grid-stride retained.
extern "C" __global__ void qmatvec_q4_0_mmvq_fused2_mr1_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, long rb0, long rb1) {
    MEMRA_PDL_ENTRY();
    const int rpb = (int)blockDim.y;
    int nb0 = (out0 + rpb - 1) / rpb;
    int nb1 = (out1 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1; vb += gridDim.x) {
        int b = vb;
        if (b < nb0) {
            q4_0_mmvq_row1_rp(W0, aq, ad, y0, in_f, out0, 1, rb0,
                              b * rpb + (int)threadIdx.y, 0);
        } else {
            b -= nb0;
            q4_0_mmvq_row1_rp(W1, aq, ad, y1, in_f, out1, 1, rb1,
                              b * rpb + (int)threadIdx.y, 0);
        }
    }
}
extern "C" __global__ void qmatvec_q4_0_mmvq_fused3_mr1_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int in_f, int out0, int out1, int out2, long rb0, long rb1, long rb2) {
    MEMRA_PDL_ENTRY();
    const int rpb = (int)blockDim.y;
    int nb0 = (out0 + rpb - 1) / rpb;
    int nb1 = (out1 + rpb - 1) / rpb;
    int nb2 = (out2 + rpb - 1) / rpb;
    for (int vb = blockIdx.x; vb < nb0 + nb1 + nb2; vb += gridDim.x) {
        int b = vb;
        const unsigned char* W; float* y; int out_f; long rb;
        if (b < nb0)            { W = W0; y = y0; out_f = out0; rb = rb0; }
        else if (b < nb0 + nb1) { W = W1; y = y1; out_f = out1; rb = rb1; b -= nb0; }
        else                    { W = W2; y = y2; out_f = out2; rb = rb2; b -= nb0 + nb1; }
        q4_0_mmvq_row1_rp(W, aq, ad, y, in_f, out_f, 1, rb,
                          b * rpb + (int)threadIdx.y, 0);
    }
}

// ===== K-QUANT split-plane (rp) decode twins — the H100 K-quant coalescing fix
// (2026-08-01 ncu, q27 Q4_K_M decode: qmatvec_q4_K_mmvq holds DRAM at 41-54% with
// "uncoalesced global accesses" = 65% excessive sectors; qmatvec_q6_K_mmvq 40% DRAM
// at 78% excessive sectors — the 144B/210B GGUF superblock strides land every 4B
// weight load off-sector, the exact Q8_0 disease of 2026-07-26). Mirror layout
// (same total bytes as the source tensor, planes):
//   q4_K: [qs: out_f*nsbk*128] ++ [meta: 16B/sblk = d(2) dmin(2) scales(12), the
//          GGUF header bytes verbatim]
//   q6_K: [ql: out_f*nsbk*128] ++ [qh: *64] ++ [scales: *16] ++ [d: *2]
// (nsbk = in_f/256 superblocks per row). Every quant fetch becomes an aligned 16B
// __ldcs and the header/scale fetches land on sector-contiguous planes. Per
// (token,row,g) the dp4a int inputs are the SAME BYTES in the same k order and the
// float chain is VERBATIM the GGUF-layout kernel -> BIT-IDENTICAL (kernel-check rp
// gates + run-gen argmax + run-spec K=1..8 arbitrate).
//
// v2 LANE->CHUNK REMAP (2026-08-01, the ledgered residue fix): the v1 mirror kept the
// GGUF intra-superblock byte order, so paired lanes (q4_K: grp 2c/2c+1; q6_K: runs
// sharing a ql window, 4 runs sharing a qh window) issued 16B loads at 32B stride over
// the SAME chunk — every qs sector requested twice and only half a sector utilized per
// instruction, plus byte-granular d/dmin/scale reads (ncu on the hot 4352-grid q4_K
// shape: 32% excessive sectors, 70% of stalls L1TEX scoreboard). v2 re-packs the quant
// planes so each grp g (0..7) of a superblock owns ONE contiguous 16B chunk:
//   q4_K qs / q6_K ql chunk byte 4k+b = nib(4k+b) | nib(4(k+4)+b) << 4  (nib = that
//     lane's original low-or-high source nibble) -> the kernel's k-th dp4a int is
//     (k<4 ? wv[k] & 0x0F0F0F0F : (wv[k-4] >> 4) & 0x0F0F0F0F), byte-equal to v1;
//   q6_K qh chunk (8B/grp): byte 4i+b packs crumbs j=0..3 of source bytes 16i+4j+b at
//     the lane's 2-bit position -> qhn[k] = (hv[k>>2] >> 2*(k&3)) & 0x03030303;
//   q4_K meta stays the 16B GGUF header (read as ONE int4, fields register-extracted);
//   q6_K scales stay GGUF order (is0/is1 are plane-adjacent: one aligned 2B load).
// Warp addresses per g-iter become wqs+g*16 / wql+g*16 / wqh+g*8 — dense contiguous
// 512B/512B/256B windows, one load each. Plane offsets and total bytes are unchanged;
// only intra-superblock byte order moved, so q4k/q6k_rp_planes and the Rust builder
// (build_kq_rp4_raw) are untouched. Same values, same fold order — only addresses
// change; the KQRP bit-bad=0 gates still arbitrate. =====
__device__ __forceinline__ void q4k_rp_planes(const unsigned char* W, int out_f, int o,
                                              int nsbk,
                                              const unsigned char** wqs,
                                              const unsigned char** wmeta) {
    *wqs   = W + ((size_t)o * nsbk) * 128;
    *wmeta = W + (size_t)out_f * nsbk * 128 + ((size_t)o * nsbk) * 16;
}
// device-side mirror build: one thread per q4_K superblock, pure nibble/byte permutation.
// v2 chunked layout: grp g owns qs bytes [g*16, g*16+16); chunk byte 4k+b holds that
// lane's source nibble of GGUF qs[(g>>1)*32 + 4k + b] (low) and ...+16... (high).
extern "C" __global__ void q4_K_split_rp_build(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int out_f, int nsbk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_f * nsbk) return;
    const unsigned char* b = src + (size_t)i * 144;
    size_t qplane = (size_t)out_f * nsbk * 128;
    unsigned char* q = dst + (size_t)i * 128;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        const unsigned char* s = b + 16 + (g >> 1) * 32;   // the grp's GGUF qs window
        int sh = (g & 1) * 4;                              // odd grp = high nibbles
        #pragma unroll
        for (int j = 0; j < 16; j++)
            q[g * 16 + j] = (unsigned char)(((s[j] >> sh) & 0xF)
                                          | (((s[j + 16] >> sh) & 0xF) << 4));
    }
    unsigned char* mt = dst + qplane + (size_t)i * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) mt[k] = b[k];
}
// rp twin of qmatvec_q4_K_mmvq: same grid/warp mapping, aligned int4 weight loads.
// (A dense-a-window + warp-shuffle exchange for the activation pair was MEASURED
// NEGATIVE here 2026-08-01: hot 4352-grid shape 23.7 -> 27.4us, DRAM 66 -> 57% — the
// 16 SHFL/iter exceed what the a-side coalescing saves. Direct a-loads stay.)
extern "C" __global__ void qmatvec_q4_K_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsbk = in_f >> 8;
    const unsigned char* wqs; const unsigned char* wmeta;
    q4k_rp_planes(W, out_f, o, nsbk, &wqs, &wmeta);
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        // ONE int4 meta load (16B GGUF header), fields extracted from registers —
        // same bytes as the v1 short/byte loads.
        const int4 mv = *(const int4*)(wmeta + (size_t)sblk * 16);
        float d_sb    = half_to_float((unsigned short)((unsigned)mv.x & 0xFFFFu));
        float dmin_sb = half_to_float((unsigned short)((unsigned)mv.x >> 16));
        // scales[12] = meta bytes 4..15: mv.y = scales[0..3], mv.z = [4..7], mv.w = [8..11].
        unsigned char sc, mn;
        if (grp < 4) {
            unsigned sg  = ((unsigned)mv.y >> (grp * 8)) & 0xFFu;   // scales[grp]
            unsigned sg4 = ((unsigned)mv.z >> (grp * 8)) & 0xFFu;   // scales[grp+4]
            sc = sg & 63; mn = sg4 & 63;
        } else {
            int g4 = grp - 4;
            unsigned s8 = ((unsigned)mv.w >> (g4 * 8)) & 0xFFu;     // scales[grp+4]
            unsigned s0 = ((unsigned)mv.y >> (g4 * 8)) & 0xFFu;     // scales[grp-4]
            unsigned s4 = ((unsigned)mv.z >> (g4 * 8)) & 0xFFu;     // scales[grp]
            sc = (s8 & 0xF) | ((s0 >> 6) << 4);
            mn = (s8 >> 4) | ((s4 >> 6) << 4);
        }
        // ONE 16B qs load from the grp's chunk (warp: dense contiguous 512B/iter).
        int4 wv = __ldcs((const int4*)(wqs + (size_t)sblk * 128 + grp * 16));
        int q4v[4] = { wv.x, wv.y, wv.z, wv.w };
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi_d = 0, sumi_sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wpack = (k < 4) ? (q4v[k] & 0x0F0F0F0F) : ((q4v[k - 4] >> 4) & 0x0F0F0F0F);
            int a = aq4[k];
            sumi_d   = dp4a(wpack, a, sumi_d);
            sumi_sum = dp4a(0x01010101, a, sumi_sum);
        }
        float d8 = adrow[g];
        acc += d_sb   * (float)((int)sc * sumi_d) * d8
             - dmin_sb * (float)((int)mn * sumi_sum) * d8;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
__device__ __forceinline__ void q6k_rp_planes(const unsigned char* W, int out_f, int o,
                                              int nsbk,
                                              const unsigned char** wql,
                                              const unsigned char** wqh,
                                              const signed char** wsc,
                                              const unsigned short** wd) {
    size_t nsbt = (size_t)out_f * nsbk;
    *wql = W + ((size_t)o * nsbk) * 128;
    *wqh = W + nsbt * 128 + ((size_t)o * nsbk) * 64;
    *wsc = (const signed char*)(W + nsbt * 192 + ((size_t)o * nsbk) * 16);
    *wd  = (const unsigned short*)(W + nsbt * 208) + (size_t)o * nsbk;
}
// device-side mirror build: one thread per q6_K superblock, pure nibble/crumb permutation.
// v2 chunked layout: grp g = (n = g>>2, run = g&3) owns ql bytes [g*16, +16) (nibble-packed
// like q4_K from its GGUF window b[n*64 + (run&1)*32 ..], high nibbles when run>=2) and qh
// bytes [g*8, +8) (byte 4i+b packs crumbs j=0..3 of GGUF qh[n*32 + 16i + 4j + b] at the
// lane's 2-bit position 2*run). scales/d keep GGUF order.
extern "C" __global__ void q6_K_split_rp_build(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int out_f, int nsbk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_f * nsbk) return;
    const unsigned char* b = src + (size_t)i * 210;
    size_t nsbt = (size_t)out_f * nsbk;
    unsigned char* ql = dst + (size_t)i * 128;
    unsigned char* qh = dst + nsbt * 128 + (size_t)i * 64;
    unsigned char* sc = dst + nsbt * 192 + (size_t)i * 16;
    unsigned char* dh = dst + nsbt * 208 + (size_t)i * 2;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        int n = g >> 2, run = g & 3;
        const unsigned char* s = b + n * 64 + (run & 1) * 32;   // the grp's GGUF ql window
        int sh = (run >> 1) * 4;                                // run>=2 = high nibbles
        #pragma unroll
        for (int j = 0; j < 16; j++)
            ql[g * 16 + j] = (unsigned char)(((s[j] >> sh) & 0xF)
                                           | (((s[j + 16] >> sh) & 0xF) << 4));
        const unsigned char* h = b + 128 + n * 32;              // the grp's GGUF qh window
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int i2 = (t >> 2) * 16, b2 = t & 3;
            unsigned v = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++)
                v |= ((unsigned)(h[i2 + 4 * j + b2] >> (2 * run)) & 3u) << (2 * j);
            qh[g * 8 + t] = (unsigned char)v;
        }
    }
    #pragma unroll
    for (int k = 0; k < 16; k++) sc[k] = b[192 + k];
    dh[0] = b[208]; dh[1] = b[209];
}
// rp twin of qmatvec_q6_K_mmvq: same grid/warp mapping, aligned int4 ql/qh loads.
// Carries MEMRA_PDL_ENTRY like its GGUF-layout source (PDL wave-A, 2026-07-23) —
// the dispatch marked-name list admits it to the programmatic-serialization launch.
// (Dense-a-window shuffle exchange measured negative here too: 23.7 -> 26.6us.)
extern "C" __global__ void qmatvec_q6_K_mmvq_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    MEMRA_PDL_ENTRY();
    (void)row_bytes;
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    int t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsbk = in_f >> 8;
    const unsigned char* wql; const unsigned char* wqh;
    const signed char* wsc; const unsigned short* wd6;
    q6k_rp_planes(W, out_f, o, nsbk, &wql, &wqh, &wsc, &wd6);
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        float d = half_to_float(wd6[sblk]);
        // is0/is1 are plane-adjacent bytes (GGUF scale order kept): ONE aligned 2B load;
        // sign-extension matches the v1 signed-char reads.
        unsigned short sv = *(const unsigned short*)((const unsigned char*)wsc
                                                     + (size_t)sblk * 16 + grp * 2);
        int sc0 = (int)(signed char)(sv & 0xFF);
        int sc1 = (int)(signed char)(sv >> 8);
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        // ONE 16B ql chunk + ONE 8B qh chunk (warp: dense 512B + 256B windows/iter);
        // the k-th qln/qhn ints below are byte-equal to the v1 window extraction.
        int4 lv = __ldcs((const int4*)(wql + (size_t)sblk * 128 + grp * 16));
        int qlv[4] = { lv.x, lv.y, lv.z, lv.w };
        uint2 hv = __ldcs((const uint2*)(wqh + (size_t)sblk * 64 + grp * 8));
        unsigned qhv[2] = { hv.x, hv.y };
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qln = (k < 4) ? (qlv[k] & 0x0F0F0F0F) : ((qlv[k - 4] >> 4) & 0x0F0F0F0F);
            int qhn = (int)((qhv[k >> 2] >> (2 * (k & 3))) & 0x03030303u);
            int vpack = qln | (qhn << 4);
            int wpack = __vsubss4(vpack, 0x20202020);
            int a = aq4[k];
            if (k < 4) sumi0 = dp4a(wpack, a, sumi0);
            else       sumi1 = dp4a(wpack, a, sumi1);
        }
        float d8 = adrow[g];
        acc += d * d8 * ( (float)(sumi0 * sc0) + (float)(sumi1 * sc1) );
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}
// batched rp row bodies + wrappers (mirror of q4k/q6k_mmvq_batched with plane loads;
// the spec-verify m=2..16 tiers ride these when the weight carries an rp4 mirror).
template<int MCOLS>
__device__ __forceinline__ void q4k_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsbk = in_f >> 8;
    const unsigned char* wqs; const unsigned char* wmeta;
    q4k_rp_planes(W, out_f, o, nsbk, &wqs, &wmeta);
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        // ONE int4 meta load, register-extracted (same bytes as the v1 short/byte loads).
        const int4 mv = *(const int4*)(wmeta + (size_t)sblk * 16);
        float d_sb    = half_to_float((unsigned short)((unsigned)mv.x & 0xFFFFu));
        float dmin_sb = half_to_float((unsigned short)((unsigned)mv.x >> 16));
        unsigned char sc, mn;
        if (grp < 4) {
            unsigned sg  = ((unsigned)mv.y >> (grp * 8)) & 0xFFu;   // scales[grp]
            unsigned sg4 = ((unsigned)mv.z >> (grp * 8)) & 0xFFu;   // scales[grp+4]
            sc = sg & 63; mn = sg4 & 63;
        } else {
            int g4 = grp - 4;
            unsigned s8 = ((unsigned)mv.w >> (g4 * 8)) & 0xFFu;     // scales[grp+4]
            unsigned s0 = ((unsigned)mv.y >> (g4 * 8)) & 0xFFu;     // scales[grp-4]
            unsigned s4 = ((unsigned)mv.z >> (g4 * 8)) & 0xFFu;     // scales[grp]
            sc = (s8 & 0xF) | ((s0 >> 6) << 4);
            mn = (s8 >> 4) | ((s4 >> 6) << 4);
        }
        // ONE 16B qs load from the grp's chunk (warp: dense contiguous 512B/iter).
        int4 wv = __ldcs((const int4*)(wqs + (size_t)sblk * 128 + grp * 16));
        int q4v[4] = { wv.x, wv.y, wv.z, wv.w };
        int wpack[8];                            // decode the 4-bit weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++)
            wpack[k] = (k < 4) ? (q4v[k] & 0x0F0F0F0F) : ((q4v[k - 4] >> 4) & 0x0F0F0F0F);
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi_d = 0, sumi_sum = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                sumi_d   = dp4a(wpack[k], aq4[k], sumi_d);
                sumi_sum = dp4a(0x01010101, aq4[k], sumi_sum);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d_sb   * (float)((int)sc * sumi_d) * d8
                    - dmin_sb * (float)((int)mn * sumi_sum) * d8;
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q4k_mmvq_batched_rp<2>(W, aq, ad, y, in_f, out_f, m);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q4k_mmvq_batched_rp<4>(W, aq, ad, y, in_f, out_f, m);
}
extern "C" __global__ void qmatvec_q4_K_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q4k_mmvq_batched_rp<8>(W, aq, ad, y, in_f, out_f, m);
}
// mcols=16 split-plane twin (lane/rp-on-st, 2026-08-06). LAYOUT law, same as NVFP4's: the b16
// dispatch pins variant="rp" whenever the weight is a split-plane mirror, so a Q4_K b16 without
// this twin would miss the symbol on any mirrored k-quant trunk — and routing split-plane bytes
// through the GGUF-layout b16 would decode garbage. Body = the q4k_mmvq_batched_rp template the
// b2/b4/b8 rp kernels run -> bit-identical per (token,row).
extern "C" __global__ void qmatvec_q4_K_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q4k_mmvq_batched_rp<16>(W, aq, ad, y, in_f, out_f, m);
}
template<int MCOLS>
__device__ __forceinline__ void q6k_mmvq_batched_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m) {
    int o = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (o >= out_f) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsbk = in_f >> 8;
    const unsigned char* wql; const unsigned char* wqh;
    const signed char* wsc; const unsigned short* wd6;
    q6k_rp_planes(W, out_f, o, nsbk, &wql, &wqh, &wsc, &wd6);
    float acc[MCOLS];
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) acc[c] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 3;
        int grp  = g & 7;
        float d = half_to_float(wd6[sblk]);
        // is0/is1 plane-adjacent: ONE aligned 2B load, sign-extension as the v1 reads.
        unsigned short sv = *(const unsigned short*)((const unsigned char*)wsc
                                                     + (size_t)sblk * 16 + grp * 2);
        int sc0 = (int)(signed char)(sv & 0xFF);
        int sc1 = (int)(signed char)(sv >> 8);
        // ONE 16B ql chunk + ONE 8B qh chunk (warp: dense 512B + 256B windows/iter).
        int4 lv = __ldcs((const int4*)(wql + (size_t)sblk * 128 + grp * 16));
        int qlv[4] = { lv.x, lv.y, lv.z, lv.w };
        uint2 hv = __ldcs((const uint2*)(wqh + (size_t)sblk * 64 + grp * 8));
        unsigned qhv[2] = { hv.x, hv.y };
        int wpack[8];                            // decode the 6-bit signed weights ONCE for this group
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qln = (k < 4) ? (qlv[k] & 0x0F0F0F0F) : ((qlv[k - 4] >> 4) & 0x0F0F0F0F);
            int qhn = (int)((qhv[k >> 2] >> (2 * (k & 3))) & 0x03030303u);
            int vpack = qln | (qhn << 4);
            wpack[k] = __vsubss4(vpack, 0x20202020);
        }
        #pragma unroll
        for (int c = 0; c < MCOLS; c++) {
            if (c >= m) break;
            const signed char* arow = aq + (size_t)c * in_f;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi0 = 0, sumi1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                if (k < 4) sumi0 = dp4a(wpack[k], aq4[k], sumi0);
                else       sumi1 = dp4a(wpack[k], aq4[k], sumi1);
            }
            float d8 = ad[(size_t)c * nsb + g];
            acc[c] += d * d8 * ( (float)(sumi0 * sc0) + (float)(sumi1 * sc1) );
        }
    }
    #pragma unroll
    for (int c = 0; c < MCOLS; c++) {
        if (c >= m) break;
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + o] = a;
    }
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b2_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q6k_mmvq_batched_rp<2>(W, aq, ad, y, in_f, out_f, m);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b4_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q6k_mmvq_batched_rp<4>(W, aq, ad, y, in_f, out_f, m);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b8_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q6k_mmvq_batched_rp<8>(W, aq, ad, y, in_f, out_f, m);
}
extern "C" __global__ void qmatvec_q6_K_mmvq_b16_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    q6k_mmvq_batched_rp<16>(W, aq, ad, y, in_f, out_f, m);
}

// ================== NVFP4 fused3: QKV (unequal out_f) in ONE launch ==================
// The q8_0_mmvq_fused2 block-offset recipe applied to the nvfp4 rp multirow body: blocks
// [0,nb0) walk W0 (wq), [nb0,nb0+nb1) walk W1 (wk), rest walk W2 (wv). Per (tensor,row,t)
// the body is nvfp4_mmvq_multirow_rp VERBATIM (same dequant, dp4a order, warp reduce,
// yscale fold) -> BIT-IDENTICAL to three separate launches. grid.y = m (verify t rows).
// Motivation (RIG-NATIVE-DECODE.md): mmvq efficiency scales with transfer size — three
// 3-17 MB launches at 57% become one ~22 MB launch (16 attn layers x every decode step).
template<int RPW>
__device__ __forceinline__ void nvfp4_mmvq_fused_seg_rp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, int seg_block0, float yscale) {
    int o0 = (((int)blockIdx.x - seg_block0) * MEMRA_MMVQ_ROWS + threadIdx.y) * RPW;
    int t = blockIdx.y;
    if (o0 >= out_f || t >= m) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    int nsb64 = in_f >> 6;
    const unsigned char* qplane = W;
    const unsigned char* splane = W + (size_t)out_f * nsb64 * 32;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.0f;
    for (int g = lane; g < nsb; g += 32) {
        int sblk = g >> 1;
        int s0 = (g & 1) * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            int o = o0 + r;
            if (o >= out_f) break;
            int4 qw = *(const int4*)(qplane + ((size_t)o * nsb64 + sblk) * 32 + (size_t)(g & 1) * 16);
            int cscw = *(const int*)(splane + ((size_t)o * nsb64 + sblk) * 4);
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int q4a = (sl == 0) ? qw.x : qw.z;
                int q4b = (sl == 0) ? qw.y : qw.w;
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + sl))) & 0xFF)) * (float)sumi;
            }
            acc[r] += adg * partial;
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        int o = o0 + r;
        if (o >= out_f) break;
        float a = warp_reduce_sum(acc[r]);
        if (lane == 0) y[(size_t)t * out_f + o] = a * yscale;
    }
}

// fused4 twin: the Linear-mixer projection quartet (wqkv + wqkv_gate + ssm_beta +
// ssm_alpha, 48 GDN layers x every decode step) in ONE launch — rig-native increment 2.
// Same seg body VERBATIM per (tensor,row,t) -> bit-identical to four separate launches.
extern "C" __global__ void qmatvec_nvfp4_mmvq_fused4_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, int in_f, int out0, int out1, int out2, int out3, int m,
        float s0, float s1, float s2, float s3) {
    const int rows_pb = MEMRA_MMVQ_ROWS * 2; // RPW=2
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    const int nb1 = (out1 + rows_pb - 1) / rows_pb;
    const int nb2 = (out2 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_fused_seg_rp<2>(W0, aq, ad, y0, in_f, out0, m, 0, s0);
    } else if ((int)blockIdx.x < nb0 + nb1) {
        nvfp4_mmvq_fused_seg_rp<2>(W1, aq, ad, y1, in_f, out1, m, nb0, s1);
    } else if ((int)blockIdx.x < nb0 + nb1 + nb2) {
        nvfp4_mmvq_fused_seg_rp<2>(W2, aq, ad, y2, in_f, out2, m, nb0 + nb1, s2);
    } else {
        nvfp4_mmvq_fused_seg_rp<2>(W3, aq, ad, y3, in_f, out3, m, nb0 + nb1 + nb2, s3);
    }
}

// fused2 twin of the fused3 dispatcher below, for MIXED-type trios (gemma4 dense
// NVFP4mix keeps attn_v / ffn_down at Q8_0, so an all-NVFP4 fused3 can never match
// there). Two weights, two output segments; per (tensor,row) the seg body is
// nvfp4_mmvq_fused_seg_rp VERBATIM — bit-identical to two separate launches.
extern "C" __global__ void qmatvec_nvfp4_mmvq_fused2_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, float s0, float s1) {
    MEMRA_PDL_ENTRY();
    const int rows_pb = MEMRA_MMVQ_ROWS * 2; // RPW=2
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_fused_seg_rp<2>(W0, aq, ad, y0, in_f, out0, m, 0, s0);
    } else {
        nvfp4_mmvq_fused_seg_rp<2>(W1, aq, ad, y1, in_f, out1, m, nb0, s1);
    }
}

// SUB-WAVE GRID-FILL twin of qmatvec_nvfp4_mmvq_fused2_rp (MEMRA_B200_MATVEC_ARM occupancy arm,
// 2026-09-02): same dispatcher shape, but instantiates nvfp4_mmvq_fused_seg_rp<1> (one row per
// warp) instead of <2>. With RPW halved, MEMRA_MMVQ_ROWS(4) warps/block still hold, but the grid
// (nb0+nb1) DOUBLES for the same out0/out1 -> twice the resident warps for the same total output
// rows, at the cost of each warp re-reading its activation row once more (already resident in
// L1/L2, per the reuse argument the vrest dedup-schedule lane already made for this family).
// Per (tensor,row) the seg body is nvfp4_mmvq_fused_seg_rp VERBATIM regardless of RPW -> the r=0
// iteration of the <2> body and the whole-warp <1> body compute IDENTICAL bits for a given row
// (same qplane/splane addressing, same dp4a chain, same warp_reduce_sum). Dispatched only when
// the <2> grid would be sub-wave on this device's SM count (host policy, lib.rs); default OFF.
extern "C" __global__ void qmatvec_nvfp4_mmvq_fused2_rp_g2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y0, float* __restrict__ y1,
        int in_f, int out0, int out1, int m, float s0, float s1) {
    MEMRA_PDL_ENTRY();
    const int rows_pb = MEMRA_MMVQ_ROWS * 1;
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_fused_seg_rp<1>(W0, aq, ad, y0, in_f, out0, m, 0, s0);
    } else {
        nvfp4_mmvq_fused_seg_rp<1>(W1, aq, ad, y1, in_f, out1, m, nb0, s1);
    }
}

extern "C" __global__ void qmatvec_nvfp4_mmvq_fused3_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y0, float* __restrict__ y1,
        float* __restrict__ y2, int in_f, int out0, int out1, int out2, int m,
        float s0, float s1, float s2) {
    const int rows_pb = MEMRA_MMVQ_ROWS * 2; // RPW=2
    const int nb0 = (out0 + rows_pb - 1) / rows_pb;
    const int nb1 = (out1 + rows_pb - 1) / rows_pb;
    if ((int)blockIdx.x < nb0) {
        nvfp4_mmvq_fused_seg_rp<2>(W0, aq, ad, y0, in_f, out0, m, 0, s0);
    } else if ((int)blockIdx.x < nb0 + nb1) {
        nvfp4_mmvq_fused_seg_rp<2>(W1, aq, ad, y1, in_f, out1, m, nb0, s1);
    } else {
        nvfp4_mmvq_fused_seg_rp<2>(W2, aq, ad, y2, in_f, out2, m, nb0 + nb1, s2);
    }
}

// ---- BF16-resident matvec (decode m=1, MEMRA_BF16_MMV): y[row] = sum_j bf16(w[row,j]) * x[j],
// f32 accumulate. One block per output row, 16-byte weight loads (8 x bf16), deterministic
// shared-memory tree reduce. Removes the per-call bf16->f32 expansion class (alloc + convert
// kernel + f32 cuBLASLt = ~5x weight traffic) for FULL_PREC bf16-resident projections at t=1.
// bf16->f32 is the same bits<<16 contract as deq()'s QT_BF16 arm. The launcher requires
// in_f % 8 == 0 and falls back to the expansion path otherwise.
extern "C" __global__ void matvec_bf16_f32acc(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f) {
    const size_t row = blockIdx.x;
    const unsigned short* wr = w + row * (size_t)in_f;
    float acc = 0.0f;
    const int stride = blockDim.x * 8;
    // 4x unroll: the 4096-wide rows give each 128-thread block only 4 iterations —
    // too shallow to hide DRAM latency (qkvg measured 1.09 TB/s vs sel_v2's 1.5).
    // A single sequential accumulator keeps the FP order IDENTICAL at any unroll.
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
        acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
        acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
        acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
        acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
        acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
        acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
        acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
        acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = red[0];
}

// ---- Selected-experts BATCHED NVFP4 dp4a matvec (device routes program, t=1 decode). ----
// Body identical to qmatvec_nvfp4_dp4a; the only change is operand addressing: blockIdx.y is
// the SELECTION index t, the weight row comes from a CONTIGUOUS per-rank expert bank at
// sel[t]*expert_stride, and the activation row/scales advance by act_row_stride/ad_row_stride
// elements per selection (0 for the shared gate/up input, per-expert rows for down). One launch
// covers every selected expert of a layer — the per-expert launch loop was pure host latency
// (~100 sequential launches/layer measured 291us with ~35us of arithmetic). Per (expert, row)
// the dot and reduction order are BIT-IDENTICAL to the per-expert kernel.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= n_sel) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = Wbank + (long)sel[t] * expert_stride + (long)o * row_bytes;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        int sblk = g >> 1;
        int whichHalf = g & 1;
        const unsigned char* b = wrow + (long)sblk * 36;
        const unsigned char* d_bytes = b;
        const unsigned char* qs = b + 4;
        int s0 = whichHalf * 2;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int s = s0 + sl;
            const unsigned char* qss = qs + s * 8;
            int q4a = get_int_b4(qss);
            int q4b = get_int_b4(qss + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(d_bytes[s]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// W4A16 selected-expert gate+up pair. The activation is checkpoint BF16, expanded exactly as
// bits<<16 inside the qmatvec_f32 reduction. Each selected id is LOCAL to this rank's contiguous
// expert bank. Per projection row this is bit-identical to qmatvec_f32 over a BF16-rounded input;
// combining the two projections only removes one launch.
extern "C" __global__ void qmatvec_nvfp4_bf16_sel_dual_rows(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel, const int* __restrict__ token_rows,
        const unsigned short* __restrict__ x,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel, long row_bytes, long expert_stride) {
    int o2 = blockIdx.x, t = blockIdx.y;
    if (o2 >= 2 * out_f || t >= n_sel) return;
    const unsigned char* bank = o2 < out_f ? Wg : Wu;
    float* y = o2 < out_f ? yg : yu;
    int o = o2 < out_f ? o2 : o2 - out_f;
    const unsigned char* wrow =
        bank + (long)sel[t] * expert_stride + (long)o * row_bytes;
    const unsigned short* xrow = x + (size_t)token_rows[t] * (size_t)in_f;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) {
        float xv = __uint_as_float((unsigned)xrow[i] << 16);
        acc += deq_nvfp4(wrow, i) * xv;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x + 31) / 32 ? s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[(size_t)t * out_f + o] = v;
    }
}

// Multi-token selected-expert gate+up pair. One CTA computes two adjacent rows from both banks;
// every output accumulator keeps the scalar kernel's element order and reduction tree.
extern "C" __global__ void qmatvec_nvfp4_bf16_sel_quad_rows(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel, const int* __restrict__ token_rows,
        const unsigned short* __restrict__ x,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel, long row_bytes, long expert_stride) {
    const int o0 = 2 * blockIdx.x;
    const int t = blockIdx.y;
    if (o0 >= out_f || t >= n_sel) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    const unsigned char* gate_row0 =
        Wg + (long)sel[t] * expert_stride + (long)o0 * row_bytes;
    const unsigned char* up_row0 =
        Wu + (long)sel[t] * expert_stride + (long)o0 * row_bytes;
    const unsigned char* gate_row1 =
        has_o1 ? gate_row0 + row_bytes : gate_row0;
    const unsigned char* up_row1 =
        has_o1 ? up_row0 + row_bytes : up_row0;
    const unsigned short* xrow = x + (size_t)token_rows[t] * (size_t)in_f;
    const float4 initial =
        dot_nvfp4_bf16_quad_row_256(
            gate_row0, up_row0, gate_row1, up_row1, xrow, in_f);
    float4 acc = has_o1
        ? initial
        : make_float4(initial.x, initial.y, 0.0f, 0.0f);
    __shared__ float gate0_s[32], up0_s[32], gate1_s[32], up1_s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.z += __shfl_down_sync(0xffffffff, acc.z, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.w += __shfl_down_sync(0xffffffff, acc.w, off);
    if ((threadIdx.x & 31) == 0) {
        gate0_s[threadIdx.x >> 5] = acc.x;
        up0_s[threadIdx.x >> 5] = acc.y;
        gate1_s[threadIdx.x >> 5] = acc.z;
        up1_s[threadIdx.x >> 5] = acc.w;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float gate0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? gate0_s[threadIdx.x] : 0.0f;
        float up0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? up0_s[threadIdx.x] : 0.0f;
        float gate1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? gate1_s[threadIdx.x] : 0.0f;
        float up1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? up1_s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            gate0_v += __shfl_down_sync(0xffffffff, gate0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            up0_v += __shfl_down_sync(0xffffffff, up0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            gate1_v += __shfl_down_sync(0xffffffff, gate1_v, off);
        for (int off = 16; off > 0; off >>= 1)
            up1_v += __shfl_down_sync(0xffffffff, up1_v, off);
        if (threadIdx.x == 0) {
            const size_t base = (size_t)t * out_f;
            yg[base + o0] = gate0_v;
            yu[base + o0] = up0_v;
            if (has_o1) {
                yg[base + o1] = gate1_v;
                yu[base + o1] = up1_v;
            }
        }
    }
}

// Device-routed W4A16 gate+up over fixed token/slot rows. `sel` carries GLOBAL expert ids;
// each rank rejects slots outside its contiguous owner range and converts owned ids to local bank
// rows. `pair/top_k` is the source token row. No host partition/count is required.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_dual_slots(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel, const unsigned short* __restrict__ x,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_pairs, int top_k,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    int o2 = blockIdx.x;
    if (o2 >= 2 * out_f) return;
    const unsigned char* bank = o2 < out_f ? Wg : Wu;
    float* y = o2 < out_f ? yg : yu;
    int o = o2 < out_f ? o2 : o2 - out_f;
    __shared__ float s[32];
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const int token = pair / top_k;
        const unsigned char* wrow =
            bank + (long)expert * expert_stride + (long)o * row_bytes;
        const unsigned short* xrow = x + (size_t)token * (size_t)in_f;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < in_f; i += blockDim.x) {
            float xv = __uint_as_float((unsigned)xrow[i] << 16);
            acc += deq_nvfp4(wrow, i) * xv;
        }
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = threadIdx.x < (blockDim.x + 31) / 32 ? s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) y[(size_t)pair * out_f + o] = v;
        }
        __syncthreads();
    }
}

// Optional A8 expert-compute twin for t=1. The external routed-expert boundary remains BF16;
// this kernel consumes one rank-local q8_1 quantization of that row and uses the established
// interleaved-NVFP4 dp4a dot. Global expert ids are rejected outside the contiguous owner range.
extern "C" __global__ void qmatvec_nvfp4_q8_ep_dual_slots(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_pairs, int top_k,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    int o2 = blockIdx.x;
    if (o2 >= 2 * out_f) return;
    const unsigned char* bank = o2 < out_f ? Wg : Wu;
    float* y = o2 < out_f ? yg : yu;
    int o = o2 < out_f ? o2 : o2 - out_f;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    __shared__ float s[32];
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const int token = pair / top_k;
        const unsigned char* wrow =
            bank + (long)expert * expert_stride + (long)o * row_bytes;
        const signed char* arow = aq + (size_t)token * (size_t)in_f;
        const float* adrow = ad + (size_t)token * (size_t)nsb;
        float acc = 0.0f;
        for (int g = tid; g < nsb; g += blockDim.x) {
            int sblk = g >> 1;
            int which_half = g & 1;
            const unsigned char* b = wrow + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            int s0 = which_half * 2;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = {a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w};
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; ++sl) {
                int sub = s0 + sl;
                const unsigned char* qss = qs + sub * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[sub]) * (float)sumi;
            }
            acc += adrow[g] * partial;
        }
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = tid < (blockDim.x + 31) / 32 ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) y[(size_t)pair * out_f + o] = v;
        }
        __syncthreads();
    }
}

// Known-good HY3 gate+up schedule from the admitted W4A8 receipt: one CTA owns one output row
// in both banks. The activation bytes are shared, while gate and up retain independent dot and
// reduction chains. Kept beside the separate-CTA schedule for a single-binary A/B.
extern "C" __global__ void qmatvec_nvfp4_q8_ep_paired_slots(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_pairs, int top_k,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    const int o = blockIdx.x;
    if (o >= out_f) return;
    const int tid = threadIdx.x;
    const int nsb = in_f >> 5;
    __shared__ float gate_s[32], up_s[32];
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const int token = pair / top_k;
        const unsigned char* gate_row =
            Wg + (long)expert * expert_stride + (long)o * row_bytes;
        const unsigned char* up_row =
            Wu + (long)expert * expert_stride + (long)o * row_bytes;
        const signed char* arow = aq + (size_t)token * (size_t)in_f;
        const float* adrow = ad + (size_t)token * (size_t)nsb;
        float2 acc = dot_nvfp4_q8_dual_row(gate_row, up_row, arow, adrow, in_f);
        for (int off = 16; off > 0; off >>= 1)
            acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
        for (int off = 16; off > 0; off >>= 1)
            acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
        if ((tid & 31) == 0) {
            gate_s[tid >> 5] = acc.x;
            up_s[tid >> 5] = acc.y;
        }
        __syncthreads();
        if (tid < 32) {
            float gate_v = tid < (blockDim.x + 31) / 32 ? gate_s[tid] : 0.0f;
            float up_v = tid < (blockDim.x + 31) / 32 ? up_s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                gate_v += __shfl_down_sync(0xffffffff, gate_v, off);
            for (int off = 16; off > 0; off >>= 1)
                up_v += __shfl_down_sync(0xffffffff, up_v, off);
            if (tid == 0) {
                yg[(size_t)pair * out_f + o] = gate_v;
                yu[(size_t)pair * out_f + o] = up_v;
            }
        }
        __syncthreads();
    }
}

// Multi-token twin of the fixed-slot gate/up kernel. One CTA owns one canonical token/slot row
// instead of serially walking every pair. Dots and reductions are unchanged; only independent
// pair rows execute concurrently. Non-owner CTAs return before reading expert weights.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_dual_pairs(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel, const unsigned short* __restrict__ x,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_pairs, int top_k,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    int o2 = blockIdx.x;
    int pair = blockIdx.y;
    if (o2 >= 2 * out_f || pair >= n_pairs) return;
    const int global_expert = sel[pair];
    if (global_expert < owner_start || global_expert >= owner_end) return;
    const unsigned char* bank = o2 < out_f ? Wg : Wu;
    float* y = o2 < out_f ? yg : yu;
    int o = o2 < out_f ? o2 : o2 - out_f;
    const int expert = global_expert - owner_start;
    const int token = pair / top_k;
    const unsigned char* wrow =
        bank + (long)expert * expert_stride + (long)o * row_bytes;
    const unsigned short* xrow = x + (size_t)token * (size_t)in_f;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < in_f; i += blockDim.x) {
        float xv = __uint_as_float((unsigned)xrow[i] << 16);
        acc += deq_nvfp4(wrow, i) * xv;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x + 31) / 32 ? s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[(size_t)pair * out_f + o] = v;
    }
}

// Adjacent-row multi-token twin. One CTA owns one canonical token/slot row and two output rows;
// non-owner CTAs return before reading expert weights.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_quad_pairs(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel, const unsigned short* __restrict__ x,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_pairs, int top_k,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    const int o0 = 2 * blockIdx.x;
    const int pair = blockIdx.y;
    if (o0 >= out_f || pair >= n_pairs) return;
    const int global_expert = sel[pair];
    if (global_expert < owner_start || global_expert >= owner_end) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    const int expert = global_expert - owner_start;
    const int token = pair / top_k;
    const unsigned char* gate_row0 =
        Wg + (long)expert * expert_stride + (long)o0 * row_bytes;
    const unsigned char* up_row0 =
        Wu + (long)expert * expert_stride + (long)o0 * row_bytes;
    const unsigned char* gate_row1 =
        has_o1 ? gate_row0 + row_bytes : gate_row0;
    const unsigned char* up_row1 =
        has_o1 ? up_row0 + row_bytes : up_row0;
    const unsigned short* xrow = x + (size_t)token * (size_t)in_f;
    const float4 initial =
        dot_nvfp4_bf16_quad_row_256(
            gate_row0, up_row0, gate_row1, up_row1, xrow, in_f);
    float4 acc = has_o1
        ? initial
        : make_float4(initial.x, initial.y, 0.0f, 0.0f);
    __shared__ float gate0_s[32], up0_s[32], gate1_s[32], up1_s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.z += __shfl_down_sync(0xffffffff, acc.z, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.w += __shfl_down_sync(0xffffffff, acc.w, off);
    if ((threadIdx.x & 31) == 0) {
        gate0_s[threadIdx.x >> 5] = acc.x;
        up0_s[threadIdx.x >> 5] = acc.y;
        gate1_s[threadIdx.x >> 5] = acc.z;
        up1_s[threadIdx.x >> 5] = acc.w;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float gate0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? gate0_s[threadIdx.x] : 0.0f;
        float up0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? up0_s[threadIdx.x] : 0.0f;
        float gate1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? gate1_s[threadIdx.x] : 0.0f;
        float up1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? up1_s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            gate0_v += __shfl_down_sync(0xffffffff, gate0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            up0_v += __shfl_down_sync(0xffffffff, up0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            gate1_v += __shfl_down_sync(0xffffffff, gate1_v, off);
        for (int off = 16; off > 0; off >>= 1)
            up1_v += __shfl_down_sync(0xffffffff, up1_v, off);
        if (threadIdx.x == 0) {
            const size_t base = (size_t)pair * out_f;
            yg[base + o0] = gate0_v;
            yu[base + o0] = up0_v;
            if (has_o1) {
                yg[base + o1] = gate1_v;
                yu[base + o1] = up1_v;
            }
        }
    }
}

// Batched W4A16 down rows. `global_pairs[j]` is the canonical token-major route slot for this
// owner-local pair; every rank writes disjoint rows into the root device's peer-accessible slot
// slab. The root then reduces slots in original token/slot order, so owner assignment does not
// change the numeric parenthesization.
extern "C" __global__ void qmatvec_nvfp4_bf16_sel_down_rows(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const int* __restrict__ global_pairs, const unsigned short* __restrict__ x,
        const float* __restrict__ macros_down, float* __restrict__ dst,
        int in_f, int out_f, int n_sel, long row_bytes, long expert_stride) {
    int o0 = 2 * blockIdx.x, j = blockIdx.y;
    if (o0 >= out_f || j >= n_sel) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    const int expert = sel[j];
    const unsigned char* row0 =
        Wbank + (long)expert * expert_stride + (long)o0 * row_bytes;
    const unsigned char* row1 = has_o1 ? row0 + row_bytes : row0;
    const unsigned short* xrow = x + (size_t)j * (size_t)in_f;
    float2 acc = has_o1
        ? dot_nvfp4_bf16_dual_row_256(row0, row1, xrow, in_f)
        : make_float2(dot_nvfp4_bf16_row_256(row0, xrow, in_f), 0.0f);
    __shared__ float row0_s[32], row1_s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
    if ((threadIdx.x & 31) == 0) {
        row0_s[threadIdx.x >> 5] = acc.x;
        row1_s[threadIdx.x >> 5] = acc.y;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float row0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? row0_s[threadIdx.x] : 0.0f;
        float row1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? row1_s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            row0_v += __shfl_down_sync(0xffffffff, row0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            row1_v += __shfl_down_sync(0xffffffff, row1_v, off);
        if (threadIdx.x == 0) {
            const size_t base = (size_t)global_pairs[j] * out_f;
            dst[base + o0] = __fmul_rn(macros_down[expert], row0_v);
            if (has_o1) {
                dst[base + o1] = __fmul_rn(macros_down[expert], row1_v);
            }
        }
    }
}

// Device-routed fixed-slot down rows. Exactly one rank owns each global expert and writes that
// pair's root-resident row; non-owners return without touching the slot.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_down_slots(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const unsigned short* __restrict__ x, const float* __restrict__ macros_down,
        float* __restrict__ dst, int in_f, int out_f, int n_pairs,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    int o0 = 2 * blockIdx.x;
    if (o0 >= out_f) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    __shared__ float row0_s[32], row1_s[32];
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const unsigned char* row0 =
            Wbank + (long)expert * expert_stride + (long)o0 * row_bytes;
        const unsigned char* row1 = has_o1 ? row0 + row_bytes : row0;
        const unsigned short* xrow = x + (size_t)pair * (size_t)in_f;
        float2 acc = has_o1
            ? dot_nvfp4_bf16_dual_row_256(row0, row1, xrow, in_f)
            : make_float2(dot_nvfp4_bf16_row_256(row0, xrow, in_f), 0.0f);
        for (int off = 16; off > 0; off >>= 1)
            acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
        for (int off = 16; off > 0; off >>= 1)
            acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
        if ((threadIdx.x & 31) == 0) {
            row0_s[threadIdx.x >> 5] = acc.x;
            row1_s[threadIdx.x >> 5] = acc.y;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            float row0_v =
                threadIdx.x < (blockDim.x + 31) / 32 ? row0_s[threadIdx.x] : 0.0f;
            float row1_v =
                threadIdx.x < (blockDim.x + 31) / 32 ? row1_s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                row0_v += __shfl_down_sync(0xffffffff, row0_v, off);
            for (int off = 16; off > 0; off >>= 1)
                row1_v += __shfl_down_sync(0xffffffff, row1_v, off);
            if (threadIdx.x == 0) {
                const size_t base = (size_t)pair * out_f;
                dst[base + o0] = __fmul_rn(macros_down[expert], row0_v);
                if (has_o1) {
                    dst[base + o1] = __fmul_rn(macros_down[expert], row1_v);
                }
            }
        }
        __syncthreads();
    }
}

// Multi-token twin of the fixed-slot down kernel. Exactly one owner rank writes each canonical
// pair row, and every dot preserves the scalar kernel's reduction order.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_down_pairs(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const unsigned short* __restrict__ x, const float* __restrict__ macros_down,
        float* __restrict__ dst, int in_f, int out_f, int n_pairs,
        int owner_start, int owner_end, long row_bytes, long expert_stride) {
    int o0 = 2 * blockIdx.x;
    int pair = blockIdx.y;
    if (o0 >= out_f || pair >= n_pairs) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    const int global_expert = sel[pair];
    if (global_expert < owner_start || global_expert >= owner_end) return;
    const int expert = global_expert - owner_start;
    const unsigned char* row0 =
        Wbank + (long)expert * expert_stride + (long)o0 * row_bytes;
    const unsigned char* row1 = has_o1 ? row0 + row_bytes : row0;
    const unsigned short* xrow = x + (size_t)pair * (size_t)in_f;
    float2 acc = has_o1
        ? dot_nvfp4_bf16_dual_row_256(row0, row1, xrow, in_f)
        : make_float2(dot_nvfp4_bf16_row_256(row0, xrow, in_f), 0.0f);
    __shared__ float row0_s[32], row1_s[32];
    for (int off = 16; off > 0; off >>= 1)
        acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
    for (int off = 16; off > 0; off >>= 1)
        acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
    if ((threadIdx.x & 31) == 0) {
        row0_s[threadIdx.x >> 5] = acc.x;
        row1_s[threadIdx.x >> 5] = acc.y;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float row0_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? row0_s[threadIdx.x] : 0.0f;
        float row1_v =
            threadIdx.x < (blockDim.x + 31) / 32 ? row1_s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            row0_v += __shfl_down_sync(0xffffffff, row0_v, off);
        for (int off = 16; off > 0; off >>= 1)
            row1_v += __shfl_down_sync(0xffffffff, row1_v, off);
        if (threadIdx.x == 0) {
            const size_t base = (size_t)pair * out_f;
            dst[base + o0] = __fmul_rn(macros_down[expert], row0_v);
            if (has_o1) {
                dst[base + o1] = __fmul_rn(macros_down[expert], row1_v);
            }
        }
    }
}

// W4A16 selected-expert down + owner-local weighted combine. Every down input row is already
// BF16-rounded by the host-expf SwiGLU kernel. Each dot replays qmatvec_f32's 256-thread reduction;
// the owner then accumulates its selected slots in their original route order. The final EP join
// adds owner rows in rank order, so this is a separately gated numeric class rather than a
// bit-identity claim against the single-device slot chain.
extern "C" __global__ void qmatvec_nvfp4_bf16_sel_down_fma(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const unsigned short* __restrict__ x, const float* __restrict__ route_w,
        const float* __restrict__ macros_down, float* __restrict__ dst,
        int in_f, int out_f, int n_sel, long row_bytes, long expert_stride) {
    int o = blockIdx.x;
    if (o >= out_f) return;
    __shared__ float s[32];
    float chain = 0.0f;
    for (int j = 0; j < n_sel; ++j) {
        const int expert = sel[j];
        const unsigned char* wrow =
            Wbank + (long)expert * expert_stride + (long)o * row_bytes;
        const unsigned short* xrow = x + (size_t)j * (size_t)in_f;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < in_f; i += blockDim.x) {
            float xv = __uint_as_float((unsigned)xrow[i] << 16);
            acc += deq_nvfp4(wrow, i) * xv;
        }
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = threadIdx.x < (blockDim.x + 31) / 32 ? s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) {
                const float scaled = __fmul_rn(macros_down[expert], v);
                chain = __fadd_rn(chain, __fmul_rn(route_w[j], scaled));
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[o] = chain;
}

// Device-routed t=1 owner-local combine. The fixed top-k slot order is identical to the
// host-partitioned owner list after skipping non-owned slots, so the existing rank-order join keeps
// the same numerical program without a CPU route partition.
extern "C" __global__ void qmatvec_nvfp4_bf16_ep_down_fma(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const unsigned short* __restrict__ x, const float* __restrict__ route_w,
        const float* __restrict__ macros_down, float* __restrict__ dst,
        int in_f, int out_f, int n_pairs, int owner_start, int owner_end,
        long row_bytes, long expert_stride) {
    int o = blockIdx.x;
    if (o >= out_f) return;
    __shared__ float s[32];
    float chain = 0.0f;
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const unsigned char* wrow =
            Wbank + (long)expert * expert_stride + (long)o * row_bytes;
        const unsigned short* xrow = x + (size_t)pair * (size_t)in_f;
        float acc = 0.0f;
        for (int i = threadIdx.x; i < in_f; i += blockDim.x) {
            float xv = __uint_as_float((unsigned)xrow[i] << 16);
            acc += deq_nvfp4(wrow, i) * xv;
        }
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = threadIdx.x < (blockDim.x + 31) / 32 ? s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) {
                const float scaled = __fmul_rn(macros_down[expert], v);
                chain = __fadd_rn(chain, __fmul_rn(route_w[pair], scaled));
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[o] = chain;
}

// Optional A8 fixed-slot down rows. Exactly one rank owns each global expert and writes that
// pair's root-resident row; the root applies route weights later in canonical token/slot order.
extern "C" __global__ void qmatvec_nvfp4_q8_ep_down_slots(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ macros_down, float* __restrict__ dst,
        int in_f, int out_f, int n_pairs, int owner_start, int owner_end,
        long row_bytes, long expert_stride) {
    const int o0 = 2 * blockIdx.x;
    if (o0 >= out_f) return;
    const int o1 = o0 + 1;
    const bool has_o1 = o1 < out_f;
    const int tid = threadIdx.x;
    const int nsb = in_f >> 5;
    __shared__ float row0_s[32], row1_s[32];
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const unsigned char* row0 =
            Wbank + (long)expert * expert_stride + (long)o0 * row_bytes;
        const unsigned char* row1 = has_o1 ? row0 + row_bytes : row0;
        const signed char* arow = aq + (size_t)pair * (size_t)in_f;
        const float* adrow = ad + (size_t)pair * (size_t)nsb;
        const float2 initial =
            dot_nvfp4_q8_dual_row(row0, row1, arow, adrow, in_f);
        float2 acc = has_o1 ? initial : make_float2(initial.x, 0.0f);
        for (int off = 16; off > 0; off >>= 1)
            acc.x += __shfl_down_sync(0xffffffff, acc.x, off);
        for (int off = 16; off > 0; off >>= 1)
            acc.y += __shfl_down_sync(0xffffffff, acc.y, off);
        if ((tid & 31) == 0) {
            row0_s[tid >> 5] = acc.x;
            row1_s[tid >> 5] = acc.y;
        }
        __syncthreads();
        if (tid < 32) {
            float row0_v = tid < (blockDim.x + 31) / 32 ? row0_s[tid] : 0.0f;
            float row1_v = tid < (blockDim.x + 31) / 32 ? row1_s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                row0_v += __shfl_down_sync(0xffffffff, row0_v, off);
            for (int off = 16; off > 0; off >>= 1)
                row1_v += __shfl_down_sync(0xffffffff, row1_v, off);
            if (tid == 0) {
                const size_t base = (size_t)pair * out_f;
                dst[base + o0] = __fmul_rn(macros_down[expert], row0_v);
                if (has_o1) {
                    dst[base + o1] = __fmul_rn(macros_down[expert], row1_v);
                }
            }
        }
        __syncthreads();
    }
}

// Optional A8 t=1 down + owner-local weighted combine. The dot body matches the established
// interleaved-NVFP4 q8_1 kernel; route accumulation remains slot-ordered within each owner.
extern "C" __global__ void qmatvec_nvfp4_q8_ep_down_fma(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ route_w, const float* __restrict__ macros_down,
        float* __restrict__ dst,
        int in_f, int out_f, int n_pairs, int owner_start, int owner_end,
        long row_bytes, long expert_stride) {
    int o = blockIdx.x;
    if (o >= out_f) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    __shared__ float s[32];
    float chain = 0.0f;
    for (int pair = 0; pair < n_pairs; ++pair) {
        const int global_expert = sel[pair];
        if (global_expert < owner_start || global_expert >= owner_end) continue;
        const int expert = global_expert - owner_start;
        const unsigned char* wrow =
            Wbank + (long)expert * expert_stride + (long)o * row_bytes;
        const signed char* arow = aq + (size_t)pair * (size_t)in_f;
        const float* adrow = ad + (size_t)pair * (size_t)nsb;
        float acc = 0.0f;
        for (int g = tid; g < nsb; g += blockDim.x) {
            int sblk = g >> 1;
            int which_half = g & 1;
            const unsigned char* b = wrow + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            int s0 = which_half * 2;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = {a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w};
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; ++sl) {
                int sub = s0 + sl;
                const unsigned char* qss = qs + sub * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[sub]) * (float)sumi;
            }
            acc += adrow[g] * partial;
        }
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = tid < (blockDim.x + 31) / 32 ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) {
                const float scaled = __fmul_rn(macros_down[expert], v);
                chain = __fadd_rn(chain, __fmul_rn(route_w[pair], scaled));
            }
        }
        __syncthreads();
    }
    if (tid == 0) dst[o] = chain;
}

// ═══ SLOT-MAJOR row layout (QT_NVFP4_V2) and the readers that consume it ═══════════════════
//
// Per row the 16 qs bytes of slot g sit CONTIGUOUSLY at g*16 (a warp reads 512B in one
// coalesced wave) and the two UE4M3 scale bytes at nslots*16 + g*2. The bytes are a pure
// permutation of the block_nvfp4 row and the per-slot dp4a/scale order is unchanged —
// BIT-IDENTICAL per row to the v1 kernels, and that claim is now a device-side gate
// (`nvfp4-bank-oracle`, unpack(v2) vs unpack(v1) through the same entry) rather than a
// comment. It was only ever a comment before 2026-09-01, which is how the corruption below
// shipped.
//
// LAYOUT GEOMETRY IS A PROPERTY OF THE BANK, NEVER OF THE ENVIRONMENT. Every producer records
// `slot_major` on the resident bank (tp.rs `ResidentNvfp4{Column,Row}BankRank`) and every
// reader branches on that stored field. Two producers exist: the always-slot-major EP2
// whole-expert banks (`MEMRA_STEP_NVFP4_EP2`) and the TP shard banks under
// `MEMRA_NVFP4_BANK_SM` (default ON since 2026-09-01). A reader that re-derives the layout
// from an env door,
// or takes a defaultable geometry scalar, is the exact hole that produced the 2026-08-29
// step37 text corruption: `kq_fetch(..., int in_f = 0)` in moe_f16_grouped.cu had two callers
// omit `in_f`, so the QT_NVFP4_V2 branch fetched the scale byte from inside the packed-codes
// region — right codes, wrong per-16 scale, on every k-block but kb=0. Diagnosis:
// research/step37-bankv3-20260901/DIAGNOSIS.md. Fixed compiler-enforced (no default) at
// 1b18a61e8; keep it that way.
extern "C" __global__ void qmatvec_nvfp4_dp4a_v2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= m) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = W + (long)o * row_bytes;
    const unsigned char* drow = wrow + (size_t)nsb * 16;
    const signed char*   arow = aq + (size_t)t * in_f;
    const float*         adrow = ad + (size_t)t * nsb;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        const unsigned char* qsg = wrow + (size_t)g * 16;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = get_int_b4(qsg + sl * 8);
            int q4b = get_int_b4(qsg + sl * 8 + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// ===================== EP2 OWNER-GUARDED SWEEPS (MEMRA_STEP_NVFP4_EP2=1) =====================
// Whole-expert layout at 2 ranks: expert e lives ENTIRE on rank (e & 1), at bank slot (e >> 1).
// Each rank sweeps only the routed pairs it owns, at FULL width (gate/up out=expert_width,
// down in=expert_width) — the 2x-wider rows this session's latency receipts favor — and
// accumulates its owned slots IN SLOT ORDER into a per-rank partial; the cross-rank join adds
// the two partials. NUMERIC-CLASS door: the slot chain regroups from (s0+s1+...+s7) into
// (owned0-sum + owned1-sum) — argmax gate + battery acceptance (the DEV_ROUTES/QKV_FUSED
// class). Per-pair dot programs are the _sel_v2 bodies verbatim.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_gu_ep(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride,
        int owner) {
    int o2 = blockIdx.x, t = blockIdx.y;
    if (o2 >= 2 * out_f || t >= n_sel) return;
    const int ex = sel[t];
    if ((ex & 1) != owner) return;             // not this rank's expert
    const unsigned char* Wbank = (o2 < out_f) ? Wg : Wu;
    float* y = (o2 < out_f) ? yg : yu;
    int o = (o2 < out_f) ? o2 : o2 - out_f;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = Wbank + (long)(ex >> 1) * expert_stride + (long)o * row_bytes;
    const unsigned char* drow = wrow + (size_t)nsb * 16;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        const unsigned char* qsg = wrow + (size_t)g * 16;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = get_int_b4(qsg + sl * 8);
            int q4b = get_int_b4(qsg + sl * 8 + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// Owner-guarded SwiGLU: exactly silu_mul_scaled_q8_1_sel(+clamp) but unowned pairs are
// skipped entirely (their gate/up rows were never written).
extern "C" __global__ void silu_mul_scaled_q8_1_sel_ep(
        const float* __restrict__ gate, const float* __restrict__ up,
        const float* __restrict__ gmac, const float* __restrict__ umac,
        const int* __restrict__ sel, float limit, int has_limit,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_per, int n_sel,
        int owner) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nblk_per = n_per / 32;
    if (warp >= nblk_per * n_sel) return;
    int t = warp / nblk_per;
    int e = sel[t];
    if ((e & 1) != owner) return;
    float gs = gmac[e], us = umac[e];
    int i = warp * 32 + lane;
    float g = gate[i] * gs;
    float silu = g / (1.0f + expf(-g));
    float u = up[i] * us;
    if (has_limit) {
        silu = silu > limit ? limit : silu;
        u = fmaxf(fminf(u, limit), -limit);
    }
    float v = silu * u;
    float amax = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[i] = (signed char)__float2int_rn(v * id);
    if (lane == 0) out_d[warp] = d;
}

// Owner-guarded down + slot-ordered owned combine in ONE launch (the down8 w8 shape).
// dst[o] = sum over OWNED slots j ascending of fma(w[j]*md[sel[j]], dot_j[o], .) — this
// rank's half of the regrouped chain; the join adds the two ranks' halves.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_down8_ep(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ route_w, const float* __restrict__ md,
        float* __restrict__ dst,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride,
        int owner) {
    const int o = blockIdx.x;
    if (o >= out_f) return;
    const int lane = threadIdx.x;
    const int j = threadIdx.y;
    const int nsb = in_f >> 5;
    __shared__ float s_dot[8];
    bool owned = false;
    if (j < n_sel) {
        const int ex = sel[j];
        owned = (ex & 1) == owner;
        if (owned) {
            const unsigned char* wrow =
                Wbank + (long)(ex >> 1) * expert_stride + (long)o * row_bytes;
            const unsigned char* drow = wrow + (size_t)nsb * 16;
            const signed char*   arow  = aq + (size_t)j * (size_t)act_row_stride;
            const float*         adrow = ad + (size_t)j * (size_t)ad_row_stride;
            float acc = 0.0f;
            for (int g = lane; g < nsb; g += 32) {
                const unsigned char* qsg = wrow + (size_t)g * 16;
                const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
                int4 a01 = aq16[0], a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int q4a = get_int_b4(qsg + sl * 8);
                    int q4b = get_int_b4(qsg + sl * 8 + 4);
                    int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                    int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                    int base = sl * 4;
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
                }
                acc += adrow[g] * partial;
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) s_dot[j] = acc;
        }
    }
    // ownership mask to warp 0 via shared (bool per slot)
    __shared__ unsigned char s_own[8];
    if (lane == 0 && j < n_sel) s_own[j] = owned ? 1 : 0;
    __syncthreads();
    if (j == 0 && lane == 0) {
        float out = 0.0f;
        for (int p = 0; p < n_sel; ++p) {
            if (!s_own[p]) continue;           // skip, never add an exact zero
            float w = route_w[p] * md[sel[p]];
            out += w * s_dot[p];
        }
        dst[o] = out;
    }
}

// NVFP4 sel GATE+UP WARP-PER-ROW (MEMRA_NVFP4_SEL_GU_WPR=1, sub-door of MEMRA_NVFP4_SEL_GU,
// default OFF and UNPRICED). NUMERIC-CLASS door, same class and
// acceptance as MEMRA_STEP_TP_QKV_FUSED / MEMRA_BF16_MMV / MEMRA_RMS_BLOCK: the per-row
// REDUCTION ORDER changes, values move by ULPs, and the gate is run-gen's prefill/decode
// argmax MATCH plus the boot battery — not a bit-tape.
//
// Why: the base _gu kernel gives one row to a 128-thread block, and in_f=4096 means nsb=128,
// so each thread owns exactly ONE 32-element slot — one vectorized load — and then the block
// pays a two-stage reduce (5 shfl + shared + __syncthreads + 5 more shfl) to combine 128
// partials. nsys: 23.6 MB per call in 29.8us = 792 GB/s, 44% of this pair's GDDR7 peak, while
// the wide-row members of the same family (lm_head 1.62 TB/s, dense FFN 1.51) sit at roofline.
// The reduce, not the load, is the cost. Here ONE WARP owns a row: 4 slots per lane, a
// warp-only reduce (no shared memory, no barrier), and blockDim.y packs 4 rows per block so
// the launch geometry stays comparable.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_gu_wpr(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int o2 = (int)blockIdx.x * (int)blockDim.y + (int)threadIdx.y;
    const int t  = blockIdx.y;
    if (o2 >= 2 * out_f || t >= n_sel) return;
    const unsigned char* Wbank = (o2 < out_f) ? Wg : Wu;
    float* y = (o2 < out_f) ? yg : yu;
    const int o = (o2 < out_f) ? o2 : o2 - out_f;
    const int lane = threadIdx.x;
    const int nsb = in_f >> 5;
    const unsigned char* wrow = Wbank + (long)sel[t] * expert_stride + (long)o * row_bytes;
    const unsigned char* drow = wrow + (size_t)nsb * 16;
    const signed char*   arow  = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float acc = 0.0f;
    for (int g = lane; g < nsb; g += 32) {   // 4 slots per lane at in_f=4096
        const unsigned char* qsg = wrow + (size_t)g * 16;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = get_int_b4(qsg + sl * 8);
            int q4b = get_int_b4(qsg + sl * 8 + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[(size_t)t * out_f + o] = acc;
}

// NVFP4 sel GATE+UP MULTIROW (MEMRA_NVFP4_SEL_GU_RPW=2|4, sub-door of MEMRA_NVFP4_SEL_GU,
// default OFF and UNPRICED) — the q8 `gu_geom<RPW>` arm ported to
// the NVFP4 banks. The base _gu kernel runs one block per (row, slot) and each block re-reads
// its slot's WHOLE activation row: 10240 blocks x 4.6 KB = 47 MB of L2 traffic per layer per
// rank against only 23.6 MB of weights. Here one block covers RPW consecutive output rows of
// one slot, so the activation group each thread holds is read ONCE and reused across the RPW
// gate+up dots (the mmvq multirow recipe), and RPW independent weight streams stay in flight.
// Per-row accumulation order and the reduce tree are the base kernel's -> BIT-IDENTICAL.
template<int RPW>
__device__ __forceinline__ void nvfp4_sel_v2_gu_rpw_body(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int o0 = (int)blockIdx.x * RPW;
    const int t  = blockIdx.y;
    if (o0 >= out_f || t >= n_sel) return;
    const int tid = threadIdx.x;
    const int nsb = in_f >> 5;
    const unsigned char* gexp = Wg + (long)sel[t] * expert_stride;
    const unsigned char* uexp = Wu + (long)sel[t] * expert_stride;
    const signed char*   arow  = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float accg[RPW], accu[RPW];
    #pragma unroll
    for (int r = 0; r < RPW; r++) { accg[r] = 0.0f; accu[r] = 0.0f; }
    for (int g = tid; g < nsb; g += blockDim.x) {
        // ONE activation group per thread per g, reused by all 2*RPW dots below.
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        const float adg = adrow[g];
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            const int o = o0 + r;
            if (o >= out_f) break;
            #pragma unroll
            for (int side = 0; side < 2; side++) {
                const unsigned char* wrow =
                    (side == 0 ? gexp : uexp) + (long)o * row_bytes;
                const unsigned char* drow = wrow + (size_t)nsb * 16;
                const unsigned char* qsg = wrow + (size_t)g * 16;
                float partial = 0.0f;
                #pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    int q4a = get_int_b4(qsg + sl * 8);
                    int q4b = get_int_b4(qsg + sl * 8 + 4);
                    int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                    int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                    int base = sl * 4;
                    int sumi = 0;
                    sumi = dp4a(va.x, aq4[base + 0], sumi);
                    sumi = dp4a(vb.x, aq4[base + 1], sumi);
                    sumi = dp4a(va.y, aq4[base + 2], sumi);
                    sumi = dp4a(vb.y, aq4[base + 3], sumi);
                    partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
                }
                if (side == 0) accg[r] += adg * partial;
                else           accu[r] += adg * partial;
            }
        }
    }
    // Per-row two-stage reduce, the base kernel's tree, one row at a time.
    __shared__ float s[32];
    #pragma unroll
    for (int r = 0; r < RPW; r++) {
        const int o = o0 + r;
        if (o >= out_f) break;
        #pragma unroll
        for (int side = 0; side < 2; side++) {
            float acc = (side == 0) ? accg[r] : accu[r];
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if ((tid & 31) == 0) s[tid >> 5] = acc;
            __syncthreads();
            if (tid < 32) {
                float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
                if (tid == 0) {
                    float* y = (side == 0) ? yg : yu;
                    y[(size_t)t * out_f + o] = v;
                }
            }
            __syncthreads();   // s[] reused by the next (row, side)
        }
    }
}
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_gu_r2(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    nvfp4_sel_v2_gu_rpw_body<2>(Wg, Wu, sel, aq, ad, yg, yu, in_f, out_f, n_sel,
                                row_bytes, expert_stride, act_row_stride, ad_row_stride);
}
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_gu_r4(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    nvfp4_sel_v2_gu_rpw_body<4>(Wg, Wu, sel, aq, ad, yg, yu, in_f, out_f, n_sel,
                                row_bytes, expert_stride, act_row_stride, ad_row_stride);
}

// NVFP4 sel DOWN + COMBINE in ONE launch (MEMRA_NVFP4_SEL_DOWN8, default ON since
// 2026-09-01; rollback MEMRA_NVFP4_SEL_DOWN8=0) — the q8
// `down8 w8` arm (cx-downkernel, waves/SM
// 0.91 -> 4.36, +8.9% on that kernel) ported to the NVFP4 bank family, which never took it:
// the sel path still launches (out_f, n_sel) ONE-WARP blocks (the q8 BASE shape) and then a
// separate axpy_rows_seq_md over an n_sel x out_f partial buffer.
//
// block = (32, n_sel): warp j computes slot j's dot for output row blockIdx.x with the EXACT
// _sel_v2 per-slot FP chain and 32-lane reduce tree; partials land in smem; then warp 0 lane 0
// replays the slot-ordered accumulation of axpy_rows_seq_md_f32 in its own expression shape
// (w = route_w[j] * md[sel[j]]; out += w * dot_j) so the same compiler contraction applies.
// -> BIT-IDENTICAL to the two-launch pair (same dot program, same reduce, same chain order),
// with 8 warps per block instead of 1 and the partial-buffer round trip gone.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_down8(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ route_w, const float* __restrict__ md,
        float* __restrict__ dst,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int o = blockIdx.x;
    if (o >= out_f) return;
    const int lane = threadIdx.x;
    const int j = threadIdx.y;               // slot; blockDim.y == n_sel (<= 8)
    const int nsb = in_f >> 5;
    __shared__ float s_dot[8];
    if (j < n_sel) {
        const unsigned char* wrow = Wbank + (long)sel[j] * expert_stride + (long)o * row_bytes;
        const unsigned char* drow = wrow + (size_t)nsb * 16;
        const signed char*   arow  = aq + (size_t)j * (size_t)act_row_stride;
        const float*         adrow = ad + (size_t)j * (size_t)ad_row_stride;
        float acc = 0.0f;
        for (int g = lane; g < nsb; g += 32) {      // _sel_v2's loop (blockDim.x == 32 in both)
            const unsigned char* qsg = wrow + (size_t)g * 16;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int q4a = get_int_b4(qsg + sl * 8);
                int q4b = get_int_b4(qsg + sl * 8 + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
            }
            acc += adrow[g] * partial;
        }
        // _sel_v2's reduce at blockDim.x == 32 is the warp tree plus a shared pass over ONE
        // warp (lane 0 reads s[0], the rest add exact +0.0) — the warp tree alone here.
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) s_dot[j] = acc;
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float out = 0.0f;
        for (int p = 0; p < n_sel; ++p) {
            float w = route_w[p] * md[sel[p]];   // axpy_rows_seq_md_f32, verbatim shape
            out += w * s_dot[p];
        }
        dst[o] = out;
    }
}

// Slot-major STREAMING twin (MEMRA_NVFP4_SEL_SM_STREAM=1, sub-door of MEMRA_NVFP4_BANK_SM,
// default OFF and UNPRICED): 8 consecutive output rows per block, each row ONE contiguous
// stream (16B/thread coalesced qs + packed scale tail) with next-row register prefetch hiding
// DRAM latency behind the reduction barriers. Needs 16B-aligned rows and one slot per thread,
// so the launcher gates it on row_bytes % 16 == 0 && in_f <= 4096 (step37 gate/up 2304B yes,
// down 360B no). Per-row program identical to _sel_v2 -> BIT-IDENTICAL. The v1-layout
// streaming attempt LOST on scattered 36B reads; this is that shape on coalesced rows.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2s(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int ROWS = 8;
    int t = blockIdx.y;
    if (t >= n_sel) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    bool active = tid < nsb;
    const unsigned char* wexp = Wbank + (long)sel[t] * expert_stride;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    int aq4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float adg = 0.0f;
    if (active) {
        const int4* aq16 = (const int4*)(arow + (size_t)tid * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        aq4[0] = a01.x; aq4[1] = a01.y; aq4[2] = a01.z; aq4[3] = a01.w;
        aq4[4] = a23.x; aq4[5] = a23.y; aq4[6] = a23.z; aq4[7] = a23.w;
        adg = adrow[tid];
    }
    __shared__ float s[32];
    int o0 = blockIdx.x * ROWS;
    int4 q = make_int4(0, 0, 0, 0);
    unsigned char d0 = 0, d1 = 0;
    if (active && o0 < out_f) {
        const unsigned char* wrow = wexp + (long)o0 * row_bytes;
        q = *(const int4*)(wrow + (size_t)tid * 16);
        const unsigned char* dr = wrow + (size_t)nsb * 16 + tid * 2;
        d0 = dr[0];
        d1 = dr[1];
    }
    for (int r = 0; r < ROWS; r++) {
        int o = o0 + r;
        if (o >= out_f) return;
        int4 cq = q;
        unsigned char cd0 = d0, cd1 = d1;
        if (active && r + 1 < ROWS && o + 1 < out_f) {
            const unsigned char* wn = wexp + (long)(o + 1) * row_bytes;
            q = *(const int4*)(wn + (size_t)tid * 16);
            const unsigned char* dn = wn + (size_t)nsb * 16 + tid * 2;
            d0 = dn[0];
            d1 = dn[1];
        }
        float partial = 0.0f;
        {
            int2 va = get_int_from_table_16_d(cq.x, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(cq.y, kvalues_mxfp4_d);
            int sumi = 0;
            sumi = dp4a(va.x, aq4[0], sumi);
            sumi = dp4a(vb.x, aq4[1], sumi);
            sumi = dp4a(va.y, aq4[2], sumi);
            sumi = dp4a(vb.y, aq4[3], sumi);
            partial += ue4m3_to_f32_d(cd0) * (float)sumi;
        }
        {
            int2 va = get_int_from_table_16_d(cq.z, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(cq.w, kvalues_mxfp4_d);
            int sumi = 0;
            sumi = dp4a(va.x, aq4[4], sumi);
            sumi = dp4a(vb.x, aq4[5], sumi);
            sumi = dp4a(va.y, aq4[6], sumi);
            sumi = dp4a(vb.y, aq4[7], sumi);
            partial += ue4m3_to_f32_d(cd1) * (float)sumi;
        }
        float acc = adg * partial;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) y[(size_t)t * out_f + o] = v;
        }
        __syncthreads();
    }
}

// SELECTED-EXPERTS sweep over slot-major banks (MEMRA_NVFP4_BANK_SM, default ON since
// 2026-09-01): the
// _sel body with the slot-major byte map — one coalesced 512B warp wave per slot group
// instead of the 36B-superblock scatter. Weights at sel[t]*expert_stride into the contiguous
// per-rank bank; same dp4a order, same per-slot scale multiply, same shfl+shared reduce tree
// as qmatvec_nvfp4_dp4a_sel -> BIT-IDENTICAL per (expert, row). Gated by nvfp4-bank-oracle.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    int o = blockIdx.x, t = blockIdx.y;
    if (o >= out_f || t >= n_sel) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = Wbank + (long)sel[t] * expert_stride + (long)o * row_bytes;
    const unsigned char* drow = wrow + (size_t)nsb * 16;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        const unsigned char* qsg = wrow + (size_t)g * 16;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = get_int_b4(qsg + sl * 8);
            int q4b = get_int_b4(qsg + sl * 8 + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// FUSION #2a (MEMRA_NVFP4_SEL_GU=1, default OFF): gate+up selected-expert sweeps in ONE
// launch over slot-major banks. The two
// sweeps share sel/aq/ad and have identical geometry; blocks [0,out_f) run the exact
// _sel_v2 body on the GATE bank, [out_f,2*out_f) on the UP bank — per-row bit-identical,
// halves the sweep launch count and doubles grid fill.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_v2_gu(
        const unsigned char* __restrict__ Wg, const unsigned char* __restrict__ Wu,
        const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    int o2 = blockIdx.x, t = blockIdx.y;
    if (o2 >= 2 * out_f || t >= n_sel) return;
    const unsigned char* Wbank = (o2 < out_f) ? Wg : Wu;
    float* y = (o2 < out_f) ? yg : yu;
    int o = (o2 < out_f) ? o2 : o2 - out_f;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* wrow = Wbank + (long)sel[t] * expert_stride + (long)o * row_bytes;
    const unsigned char* drow = wrow + (size_t)nsb * 16;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    float acc = 0.0f;
    for (int g = tid; g < nsb; g += blockDim.x) {
        const unsigned char* qsg = wrow + (size_t)g * 16;
        const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        float partial = 0.0f;
        #pragma unroll
        for (int sl = 0; sl < 2; sl++) {
            int q4a = get_int_b4(qsg + sl * 8);
            int q4b = get_int_b4(qsg + sl * 8 + 4);
            int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
            int base = sl * 4;
            int sumi = 0;
            sumi = dp4a(va.x, aq4[base + 0], sumi);
            sumi = dp4a(vb.x, aq4[base + 1], sumi);
            sumi = dp4a(va.y, aq4[base + 2], sumi);
            sumi = dp4a(vb.y, aq4[base + 3], sumi);
            partial += ue4m3_to_f32_d(drow[g * 2 + sl]) * (float)sumi;
        }
        acc += adrow[g] * partial;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((tid & 31) == 0) s[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) y[(size_t)t * out_f + o] = v;
    }
}

// STREAMING twin: ROWS consecutive rows per block with NEXT-ROW REGISTER PREFETCH — the
// per-row program keeps the exact 128-slot arrangement and shfl/shared reduce of the
// single-row kernel (bit-identical per row); the prefetch hides the next row's DRAM latency
// behind the current row's reduction barriers. Targets the measured 570GB/s (36% of peak)
// of the one-row-per-block form on the 188-SM card.
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_stream(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int ROWS = 16;
    int t = blockIdx.y;
    if (t >= n_sel) return;
    int tid = threadIdx.x;
    int nsb = in_f >> 5;
    // One group per thread (launcher refuses nsb > blockDim); inactive threads still walk
    // the loop for the block-wide barriers, contributing zero.
    bool active = tid < nsb;
    const unsigned char* wexp = Wbank + (long)sel[t] * expert_stride;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    // Activations are row-invariant: load once.
    int sblk = tid >> 1;
    int whichHalf = tid & 1;
    int s0 = whichHalf * 2;
    int aq4[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float adg = 0.0f;
    if (active) {
        const int4* aq16 = (const int4*)(arow + (size_t)tid * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        aq4[0] = a01.x; aq4[1] = a01.y; aq4[2] = a01.z; aq4[3] = a01.w;
        aq4[4] = a23.x; aq4[5] = a23.y; aq4[6] = a23.z; aq4[7] = a23.w;
        adg = adrow[tid];
    }
    __shared__ float s[32];
    int o0 = blockIdx.x * ROWS;
    // Prefetch row 0's weights.
    int q4a0 = 0, q4b0 = 0, q4a1 = 0, q4b1 = 0;
    unsigned char d0 = 0, d1 = 0;
    if (active) {
        const unsigned char* b = wexp + (long)o0 * row_bytes + (long)sblk * 36;
        q4a0 = get_int_b4(b + 4 + s0 * 8);
        q4b0 = get_int_b4(b + 4 + s0 * 8 + 4);
        q4a1 = get_int_b4(b + 4 + (s0 + 1) * 8);
        q4b1 = get_int_b4(b + 4 + (s0 + 1) * 8 + 4);
        d0 = b[s0];
        d1 = b[s0 + 1];
    }
    for (int r = 0; r < ROWS; r++) {
        int o = o0 + r;
        if (o >= out_f) return;
        int cq4a0 = q4a0, cq4b0 = q4b0, cq4a1 = q4a1, cq4b1 = q4b1;
        unsigned char cd0 = d0, cd1 = d1;
        if (active && r + 1 < ROWS && o + 1 < out_f) {
            const unsigned char* bn = wexp + (long)(o + 1) * row_bytes + (long)sblk * 36;
            q4a0 = get_int_b4(bn + 4 + s0 * 8);
            q4b0 = get_int_b4(bn + 4 + s0 * 8 + 4);
            q4a1 = get_int_b4(bn + 4 + (s0 + 1) * 8);
            q4b1 = get_int_b4(bn + 4 + (s0 + 1) * 8 + 4);
            d0 = bn[s0];
            d1 = bn[s0 + 1];
        }
        float partial = 0.0f;
        {
            int2 va = get_int_from_table_16_d(cq4a0, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(cq4b0, kvalues_mxfp4_d);
            int sumi = 0;
            sumi = dp4a(va.x, aq4[0], sumi);
            sumi = dp4a(vb.x, aq4[1], sumi);
            sumi = dp4a(va.y, aq4[2], sumi);
            sumi = dp4a(vb.y, aq4[3], sumi);
            partial += ue4m3_to_f32_d(cd0) * (float)sumi;
        }
        {
            int2 va = get_int_from_table_16_d(cq4a1, kvalues_mxfp4_d);
            int2 vb = get_int_from_table_16_d(cq4b1, kvalues_mxfp4_d);
            int sumi = 0;
            sumi = dp4a(va.x, aq4[4], sumi);
            sumi = dp4a(vb.x, aq4[5], sumi);
            sumi = dp4a(va.y, aq4[6], sumi);
            sumi = dp4a(vb.y, aq4[7], sumi);
            partial += ue4m3_to_f32_d(cd1) * (float)sumi;
        }
        float acc = adg * partial;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) y[(size_t)t * out_f + o] = v;
        }
        __syncthreads();
    }
}

// MULTI-ROW twin of qmatvec_nvfp4_dp4a_sel: each block computes ROWS consecutive output rows
// of the SAME (expert, activation) pair, running the exact single-row program per row (same g
// striding, same shfl/shared reduction) — per row BIT-IDENTICAL to the single-row kernel. The
// win is launch-tail amortization + warm L1 activation reloads across rows (the single-row
// form measured ~570GB/s of weight traffic on the 188-SM card: one 2304B row per block).
extern "C" __global__ void qmatvec_nvfp4_dp4a_sel_mr4(
        const unsigned char* __restrict__ Wbank, const int* __restrict__ sel,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int n_sel,
        long row_bytes, long expert_stride, long act_row_stride, long ad_row_stride) {
    const int ROWS = 4;
    const int ROW_THREADS = 128;
    int t = blockIdx.y;
    if (t >= n_sel) return;
    int r = threadIdx.x / ROW_THREADS;
    int tid = threadIdx.x % ROW_THREADS;
    int nsb = in_f >> 5;
    const unsigned char* wexp = Wbank + (long)sel[t] * expert_stride;
    const signed char*   arow = aq + (size_t)t * (size_t)act_row_stride;
    const float*         adrow = ad + (size_t)t * (size_t)ad_row_stride;
    __shared__ float s[ROWS][32];
    int o = blockIdx.x * ROWS + r;
    if (o < out_f) {
        const unsigned char* wrow = wexp + (long)o * row_bytes;
        float acc = 0.0f;
        for (int g = tid; g < nsb; g += ROW_THREADS) {
            int sblk = g >> 1;
            int whichHalf = g & 1;
            const unsigned char* b = wrow + (long)sblk * 36;
            const unsigned char* d_bytes = b;
            const unsigned char* qs = b + 4;
            int s0 = whichHalf * 2;
            const int4* aq16 = (const int4*)(arow + (size_t)g * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            float partial = 0.0f;
            #pragma unroll
            for (int sl = 0; sl < 2; sl++) {
                int ss = s0 + sl;
                const unsigned char* qss = qs + ss * 8;
                int q4a = get_int_b4(qss);
                int q4b = get_int_b4(qss + 4);
                int2 va = get_int_from_table_16_d(q4a, kvalues_mxfp4_d);
                int2 vb = get_int_from_table_16_d(q4b, kvalues_mxfp4_d);
                int base = sl * 4;
                int sumi = 0;
                sumi = dp4a(va.x, aq4[base + 0], sumi);
                sumi = dp4a(vb.x, aq4[base + 1], sumi);
                sumi = dp4a(va.y, aq4[base + 2], sumi);
                sumi = dp4a(vb.y, aq4[base + 3], sumi);
                partial += ue4m3_to_f32_d(d_bytes[ss]) * (float)sumi;
            }
            acc += adrow[g] * partial;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[r][tid >> 5] = acc;
        __syncthreads();
        if (tid < 32) {
            float v = (tid < (ROW_THREADS + 31) / 32) ? s[r][tid] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (tid == 0) y[(size_t)t * out_f + o] = v;
        }
    } else {
        __syncthreads();
    }
}

// Selected-experts batched twin of silu_mul_scaled_q8_1: gate/up hold [n_sel, n_per] rows, the
// per-expert macro-scales come from device arrays indexed via sel. Same warp-per-32-block
// program, same amax/127 rounding — per expert row the q8_1 output is BIT-IDENTICAL to the
// scalar-macro kernel. n_per must be a multiple of 32.
extern "C" __global__ void silu_mul_scaled_q8_1_sel(
        const float* __restrict__ gate, const float* __restrict__ up,
        const float* __restrict__ gmac, const float* __restrict__ umac,
        const int* __restrict__ sel,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_per, int n_sel) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nblk_per = n_per / 32;
    if (warp >= nblk_per * n_sel) return;
    int t = warp / nblk_per;
    int e = sel[t];
    float gs = gmac[e], us = umac[e];
    int i = warp * 32 + lane;
    float g = gate[i] * gs;
    float r = (g / (1.0f + expf(-g))) * (up[i] * us);
    float amax = fabsf(r);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[i] = (signed char)__float2int_rn(r * id);
    if (lane == 0) out_d[warp] = d;
}

// FUSION #2e: shexp down matvec + scaled accumulate in ONE launch (t=1). Block r runs the
// exact matvec_bf16_f32acc per-row program, then thread 0 applies the exact add_scaled_rows
// expression (dst[r] += y_r * scale[0]) — the register value equals the memory roundtrip the
// split path took, so the accumulate is bit-identical. Also removes the per-layer ownership
// alloc + 16KB copy the split path needed.
extern "C" __global__ void matvec_bf16_down_addscale(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        const float* __restrict__ scale, float* __restrict__ dst, int in_f) {
    const size_t row = blockIdx.x;
    const unsigned short* wr = w + row * (size_t)in_f;
    float acc = 0.0f;
    const int stride = blockDim.x * 8;
    // 4x unroll: the 4096-wide rows give each 128-thread block only 4 iterations —
    // too shallow to hide DRAM latency (qkvg measured 1.09 TB/s vs sel_v2's 1.5).
    // A single sequential accumulator keeps the FP order IDENTICAL at any unroll.
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
        acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
        acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
        acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
        acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
        acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
        acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
        acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
        acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
    }
    __shared__ float red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) dst[row] += red[0] * scale[0];
}

// FUSION #2b: shexp gate/up bf16 matvecs + SwiGLU act in ONE launch (t=1). Block r runs
// the exact matvec_bf16_f32acc per-row program on the GATE row, then the UP row (two
// sequential dots, identical FP chains to the dual kernel's separate blocks), then thread 0
// applies the exact silu (or step35 clamped) expression — bit-identical to
// matvec_bf16_dual_into + ffn_act_lim. limit <= 0 selects the plain silu form.
// ======================= T-COLUMN VERIFY TWINS (step37 MTP spec) =======================
// Weight-amortized small-t matvecs: each block loads its row's weights ONCE and accumulates
// against T input columns (T <= 8). Per COLUMN the FP sequence is IDENTICAL to the t=1
// kernel (same pack order, same per-column accumulator, same reduce tree), so every
// column's output is bit-equal to running the t=1 kernel on that column alone.

extern "C" __global__ void matvec_bf16_f32acc_tcol(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int t) {
    const size_t row = blockIdx.x;
    const unsigned short* wr = w + row * (size_t)in_f;
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c] = 0.0f;
    const int stride = blockDim.x * 8;
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float wv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) wv[j] = __uint_as_float((unsigned)wp[j] << 16);
        for (int c = 0; c < t; c++) {
            const float* xc = x + (size_t)c * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xc + i);
            float4 x1 = *reinterpret_cast<const float4*>(xc + i + 4);
            acc[c] += wv[0] * x0.x;
            acc[c] += wv[1] * x0.y;
            acc[c] += wv[2] * x0.z;
            acc[c] += wv[3] * x0.w;
            acc[c] += wv[4] * x1.x;
            acc[c] += wv[5] * x1.y;
            acc[c] += wv[6] * x1.z;
            acc[c] += wv[7] * x1.w;
        }
    }
    __shared__ float red[256];
    for (int c = 0; c < t; c++) {
        red[threadIdx.x] = acc[c];
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(size_t)c * gridDim.x + row] = red[0];
        __syncthreads();
    }
}

extern "C" __global__ void matvec_bf16_qkvg_tcol(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g, int t) {
    int r = blockIdx.x;
    const unsigned short* w;
    float* y;
    int row;
    int out_stride;
    if (r < out_q) {
        w = wq; y = yq; row = r; out_stride = out_q;
    } else if (r < out_q + out_kv) {
        w = wk; y = yk; row = r - out_q; out_stride = out_kv;
    } else if (r < out_q + 2 * out_kv) {
        w = wv; y = yv; row = r - out_q - out_kv; out_stride = out_kv;
    } else {
        w = wg; y = yg; row = r - out_q - 2 * out_kv; out_stride = out_g;
    }
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c] = 0.0f;
    const int stride = blockDim.x * 8;
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float wv8[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) wv8[j] = __uint_as_float((unsigned)wp[j] << 16);
        for (int c = 0; c < t; c++) {
            const float* xc = x + (size_t)c * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xc + i);
            float4 x1 = *reinterpret_cast<const float4*>(xc + i + 4);
            acc[c] += wv8[0] * x0.x;
            acc[c] += wv8[1] * x0.y;
            acc[c] += wv8[2] * x0.z;
            acc[c] += wv8[3] * x0.w;
            acc[c] += wv8[4] * x1.x;
            acc[c] += wv8[5] * x1.y;
            acc[c] += wv8[6] * x1.z;
            acc[c] += wv8[7] * x1.w;
        }
    }
    __shared__ float s[32];
    for (int c = 0; c < t; c++) {
        float a = acc[c];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = a;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) y[(size_t)c * out_stride + row] = v;
        }
        __syncthreads();
    }
}

// COMPILE-TIME-T twin of matvec_bf16_qkvg_tcol (same local-spill disease as b4_tcol).
template <int T>
static __device__ __forceinline__ void qkvg_tcol_body(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    int r = blockIdx.x;
    const unsigned short* w;
    float* y;
    int row;
    int out_stride;
    if (r < out_q) {
        w = wq; y = yq; row = r; out_stride = out_q;
    } else if (r < out_q + out_kv) {
        w = wk; y = yk; row = r - out_q; out_stride = out_kv;
    } else if (r < out_q + 2 * out_kv) {
        w = wv; y = yv; row = r - out_q - out_kv; out_stride = out_kv;
    } else {
        w = wg; y = yg; row = r - out_q - 2 * out_kv; out_stride = out_g;
    }
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc[T];
    #pragma unroll
    for (int c = 0; c < T; c++) acc[c] = 0.0f;
    const int stride = blockDim.x * 8;
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float wv8[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) wv8[j] = __uint_as_float((unsigned)wp[j] << 16);
        #pragma unroll
        for (int c = 0; c < T; c++) {
            const float* xc = x + (size_t)c * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xc + i);
            float4 x1 = *reinterpret_cast<const float4*>(xc + i + 4);
            acc[c] += wv8[0] * x0.x;
            acc[c] += wv8[1] * x0.y;
            acc[c] += wv8[2] * x0.z;
            acc[c] += wv8[3] * x0.w;
            acc[c] += wv8[4] * x1.x;
            acc[c] += wv8[5] * x1.y;
            acc[c] += wv8[6] * x1.z;
            acc[c] += wv8[7] * x1.w;
        }
    }
    __shared__ float s[32];
    #pragma unroll
    for (int c = 0; c < T; c++) {
        float a = acc[c];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = a;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) y[(size_t)c * out_stride + row] = v;
        }
        __syncthreads();
    }
}
extern "C" __global__ void matvec_bf16_qkvg_tcol_t2(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    qkvg_tcol_body<2>(wq, wk, wv, wg, x, yq, yk, yv, yg, in_f, out_q, out_kv, out_g);
}
extern "C" __global__ void matvec_bf16_qkvg_tcol_t4(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    qkvg_tcol_body<4>(wq, wk, wv, wg, x, yq, yk, yv, yg, in_f, out_q, out_kv, out_g);
}
extern "C" __global__ void matvec_bf16_qkvg_tcol_t8(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    qkvg_tcol_body<8>(wq, wk, wv, wg, x, yq, yk, yv, yg, in_f, out_q, out_kv, out_g);
}

// COMPILE-TIME-T twins of matvec_bf16_b4_tcol: the runtime-t inner loops index acc[c]
// dynamically, which spills the accumulators to local memory and turns the weight walk
// latency-bound (283us vs the t=1 kernel's 33 at t=8). Full unroll keeps the IDENTICAL
// per-(b, c) FP chain — bit-identical per column — with the accumulators in registers.
template <int T>
static __device__ __forceinline__ void b4_tcol_body(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    int tid = threadIdx.x;
    const unsigned short* ws4[4] = { w0, w1, w2, w3 };
    __shared__ float s[32];
    float total[T];
    #pragma unroll
    for (int c = 0; c < T; c++) total[c] = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned short* wr = ws4[b] + (size_t)r * block_cols;
        float acc[T];
        #pragma unroll
        for (int c = 0; c < T; c++) acc[c] = 0.0f;
#pragma unroll 4
        for (int i = tid * 8; i < block_cols; i += blockDim.x * 8) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float wv8[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) wv8[j] = __uint_as_float((unsigned)wp[j] << 16);
            #pragma unroll
            for (int c = 0; c < T; c++) {
                const float* xb = x + ((size_t)c * 4 + b) * block_cols;
                float4 x0 = *reinterpret_cast<const float4*>(xb + i);
                float4 x1 = *reinterpret_cast<const float4*>(xb + i + 4);
                acc[c] += wv8[0] * x0.x;
                acc[c] += wv8[1] * x0.y;
                acc[c] += wv8[2] * x0.z;
                acc[c] += wv8[3] * x0.w;
                acc[c] += wv8[4] * x1.x;
                acc[c] += wv8[5] * x1.y;
                acc[c] += wv8[6] * x1.z;
                acc[c] += wv8[7] * x1.w;
            }
        }
        #pragma unroll
        for (int c = 0; c < T; c++) {
            float a = acc[c];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
            if ((tid & 31) == 0) s[tid >> 5] = a;
            __syncthreads();
            if (tid == 0) {
                float v = 0.0f;
                for (int wi = 0; wi < (blockDim.x + 31) / 32; wi++) v += s[wi];
                total[c] += v;
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        #pragma unroll
        for (int c = 0; c < T; c++) y[(size_t)c * out_f + r] = total[c];
    }
}
extern "C" __global__ void matvec_bf16_b4_tcol_t2(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    b4_tcol_body<2>(w0, w1, w2, w3, x, y, block_cols, out_f);
}
extern "C" __global__ void matvec_bf16_b4_tcol_t4(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    b4_tcol_body<4>(w0, w1, w2, w3, x, y, block_cols, out_f);
}
extern "C" __global__ void matvec_bf16_b4_tcol_t8(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    b4_tcol_body<8>(w0, w1, w2, w3, x, y, block_cols, out_f);
}

extern "C" __global__ void matvec_bf16_b4_tcol(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f,
        int t) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    int tid = threadIdx.x;
    const unsigned short* ws4[4] = { w0, w1, w2, w3 };
    __shared__ float s[32];
    float total[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) total[c] = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned short* wr = ws4[b] + (size_t)r * block_cols;
        float acc[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) acc[c] = 0.0f;
#pragma unroll 4
        for (int i = tid * 8; i < block_cols; i += blockDim.x * 8) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float wv8[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) wv8[j] = __uint_as_float((unsigned)wp[j] << 16);
            for (int c = 0; c < t; c++) {
                const float* xb = x + ((size_t)c * 4 + b) * block_cols;
                float4 x0 = *reinterpret_cast<const float4*>(xb + i);
                float4 x1 = *reinterpret_cast<const float4*>(xb + i + 4);
                acc[c] += wv8[0] * x0.x;
                acc[c] += wv8[1] * x0.y;
                acc[c] += wv8[2] * x0.z;
                acc[c] += wv8[3] * x0.w;
                acc[c] += wv8[4] * x1.x;
                acc[c] += wv8[5] * x1.y;
                acc[c] += wv8[6] * x1.z;
                acc[c] += wv8[7] * x1.w;
            }
        }
        for (int c = 0; c < t; c++) {
            float a = acc[c];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
            if ((tid & 31) == 0) s[tid >> 5] = a;
            __syncthreads();
            if (tid == 0) {
                float v = 0.0f;
                for (int wi = 0; wi < (blockDim.x + 31) / 32; wi++) v += s[wi];
                total[c] += v;
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        for (int c = 0; c < t; c++) y[(size_t)c * out_f + r] = total[c];
    }
}

// DUAL-SILU INTERLEAVED twin (MEMRA_DUAL_ILV=1): ONE main loop loading gate and up
// packs together — two independent accumulators keep each row's FP add order IDENTICAL
// to the sequential twin, so values are bit-exact; the loads co-issue (2x ILP) and one
// __syncthreads round drops. Reduce phases run per-accumulator with the same programs.
extern "C" __global__ void matvec_bf16_dual_silu_ilv(
        const unsigned short* __restrict__ wg, const unsigned short* __restrict__ wu,
        const float* __restrict__ x, float* __restrict__ act,
        int in_f, int out_f, float limit) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    __shared__ float red[32];
    __shared__ float vals_sh[2];
    const unsigned short* wgr = wg + (size_t)r * (size_t)in_f;
    const unsigned short* wur = wu + (size_t)r * (size_t)in_f;
    float accg = 0.0f;
    float accu = 0.0f;
    const int stride = blockDim.x * 8;
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pg = *reinterpret_cast<const uint4*>(wgr + i);
        uint4 pu = *reinterpret_cast<const uint4*>(wur + i);
        const unsigned short* wpg = reinterpret_cast<const unsigned short*>(&pg);
        const unsigned short* wpu = reinterpret_cast<const unsigned short*>(&pu);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
        accg += __uint_as_float((unsigned)wpg[0] << 16) * x0.x;
        accg += __uint_as_float((unsigned)wpg[1] << 16) * x0.y;
        accg += __uint_as_float((unsigned)wpg[2] << 16) * x0.z;
        accg += __uint_as_float((unsigned)wpg[3] << 16) * x0.w;
        accg += __uint_as_float((unsigned)wpg[4] << 16) * x1.x;
        accg += __uint_as_float((unsigned)wpg[5] << 16) * x1.y;
        accg += __uint_as_float((unsigned)wpg[6] << 16) * x1.z;
        accg += __uint_as_float((unsigned)wpg[7] << 16) * x1.w;
        accu += __uint_as_float((unsigned)wpu[0] << 16) * x0.x;
        accu += __uint_as_float((unsigned)wpu[1] << 16) * x0.y;
        accu += __uint_as_float((unsigned)wpu[2] << 16) * x0.z;
        accu += __uint_as_float((unsigned)wpu[3] << 16) * x0.w;
        accu += __uint_as_float((unsigned)wpu[4] << 16) * x1.x;
        accu += __uint_as_float((unsigned)wpu[5] << 16) * x1.y;
        accu += __uint_as_float((unsigned)wpu[6] << 16) * x1.z;
        accu += __uint_as_float((unsigned)wpu[7] << 16) * x1.w;
    }
    #pragma unroll
    for (int which = 0; which < 2; which++) {
        float acc = which == 0 ? accg : accu;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) vals_sh[which] = v;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float g = vals_sh[0];
        float u = vals_sh[1];
        if (limit > 0.0f) {
            float uc = fmaxf(fminf(u, limit), -limit);
            float sl = g / (1.0f + expf(-g));
            act[r] = fminf(sl, limit) * uc;
        } else {
            act[r] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// F32ACC X4 twin (MEMRA_DOWN_X4=1, SHORT-ROW shapes): in_f<=2048 gives each 128-thread
// block barely one loop iteration — latency-starved (shexp down measured 420GB/s).
// Four sequential rows per block, each running the EXACT f32acc per-row program.
extern "C" __global__ void matvec_bf16_f32acc_x4(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f) {
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
#pragma unroll 4
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(x + i);
            float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[row] = red[0];
        __syncthreads();
    }
}

extern "C" __global__ void matvec_bf16_dual_silu(
        const unsigned short* __restrict__ wg, const unsigned short* __restrict__ wu,
        const float* __restrict__ x, float* __restrict__ act,
        int in_f, int out_f, float limit) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    __shared__ float red[32];
    __shared__ float vals_sh[2];
    float vals[2];
    (void)vals_sh;
    #pragma unroll
    for (int which = 0; which < 2; which++) {
        const unsigned short* wr =
            (which == 0 ? wg : wu) + (size_t)r * (size_t)in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
#pragma unroll 4
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(x + i);
            float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        // EXACT dual-kernel reduce (shfl tree + shared[32] + warp-0 tree).
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) vals[which] = v;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float g = vals[0];
        float u = vals[1];
        if (limit > 0.0f) {
            // step35 clamped form (swiglu_clamped_mul_scaled_f32 at gs=us=1).
            float uc = fmaxf(fminf(u, limit), -limit);
            float sl = g / (1.0f + expf(-g));
            act[r] = fminf(sl, limit) * uc;
        } else {
            act[r] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// T-ROW twin of matvec_bf16_dual_silu (spec verify / batched serving shexp): blockIdx.y
// is the token row; x and act advance by full rows. The per-(row, output) program is the
// t=1 kernel byte-for-byte — bit-identical per row.
extern "C" __global__ void matvec_bf16_dual_silu_rows(
        const unsigned short* __restrict__ wg, const unsigned short* __restrict__ wu,
        const float* __restrict__ x, float* __restrict__ act,
        int in_f, int out_f, float limit) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    const int trow = blockIdx.y;
    x += (size_t)trow * in_f;
    act += (size_t)trow * out_f;
    __shared__ float red[32];
    float vals[2];
    #pragma unroll
    for (int which = 0; which < 2; which++) {
        const unsigned short* wr =
            (which == 0 ? wg : wu) + (size_t)r * (size_t)in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
#pragma unroll 4
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(x + i);
            float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (threadIdx.x == 0) vals[which] = v;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float g = vals[0];
        float u = vals[1];
        if (limit > 0.0f) {
            float uc = fmaxf(fminf(u, limit), -limit);
            float sl = g / (1.0f + expf(-g));
            act[r] = fminf(sl, limit) * uc;
        } else {
            act[r] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

// T-ROW twin of matvec_bf16_f32acc_x4: blockIdx.y is the token row.
extern "C" __global__ void matvec_bf16_f32acc_x4_rows(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f) {
    const int trow = blockIdx.y;
    x += (size_t)trow * in_f;
    y += (size_t)trow * out_f;
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
#pragma unroll 4
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(x + i);
            float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[row] = red[0];
        __syncthreads();
    }
}

// PREFETCH twin of matvec_bf16_f32acc_x4_rows (MEMRA_B200_MATVEC_ARM occupancy arm,
// lane/b200-matvec-occupancy, 2026-09-02). B200 census: this pair was 17.0% of GPU time at
// 24.1us avg over the KDA [8192x4096]/[4096x8192] bf16 projections; roofline for one such
// projection is ~8us of HBM3e traffic (64 MB / 8 TB/s), so the measured cost is ~3x its byte
// budget — a latency, not bandwidth, signature on B200's narrower 148-SM/8-TB/s shape.
//
// Same grid/block mapping, same 4-row sequential loop, same red[] tree reduction as the
// shipped kernel; the ONLY change is software-pipelined (double-buffered) K-loop loads: the
// NEXT iteration's weight/activation reads issue before the CURRENT iteration's 8-fma chain
// runs, so the load latency for i+stride overlaps the compute for i instead of stalling behind
// it. The accumulation itself is untouched — same 8 sequential `acc +=` fmas, in the same
// per-thread i order, for the same i -> bit-identical per (row, token) to the shipped kernel.
// This is the same class of change as the qmatvec mmvq family's `pf` weight-prefetch variant
// (mmv-bv doc comment above, `sm_count`): load-issue timing changes, arithmetic order does not.
extern "C" __global__ void matvec_bf16_f32acc_x4_rows_pf(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f) {
    const int trow = blockIdx.y;
    x += (size_t)trow * in_f;
    y += (size_t)trow * out_f;
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        const int stride = blockDim.x * 8;
        int i = threadIdx.x * 8;
        bool have = i < in_f;
        uint4 pack = have ? *reinterpret_cast<const uint4*>(wr + i) : make_uint4(0, 0, 0, 0);
        float4 x0 = have ? *reinterpret_cast<const float4*>(x + i) : make_float4(0, 0, 0, 0);
        float4 x1 = have ? *reinterpret_cast<const float4*>(x + i + 4) : make_float4(0, 0, 0, 0);
        float acc = 0.0f;
        for (; i < in_f; i += stride) {
            int ni = i + stride;
            bool have_next = ni < in_f;
            uint4 npack;
            float4 nx0, nx1;
            if (have_next) {
                npack = *reinterpret_cast<const uint4*>(wr + ni);
                nx0 = *reinterpret_cast<const float4*>(x + ni);
                nx1 = *reinterpret_cast<const float4*>(x + ni + 4);
            }
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            if (have_next) {
                pack = npack;
                x0 = nx0;
                x1 = nx1;
            }
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[row] = red[0];
        __syncthreads();
    }
}

// TP2 row-parallel decode partial: identical BF16 expansion, multiply/add order and
// red[256] tree as matvec_bf16_f32acc_x4_rows, restricted to one aligned contiguous K range.
// Each rank computes every output row over its K half; Tp2ReplicatedRowJoin then adds
// (rank0, rank1) in the same order on both cards. The cross-rank association is a deliberate
// TP numeric class and is gated against the unsplit full-row program.
extern "C" __global__ void matvec_bf16_f32acc_x4_range(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int k_start, int k_len) {
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc = 0.0f;
        const int stride = blockDim.x * 8;
        const int k_end = k_start + k_len;
#pragma unroll 4
        for (int i = k_start + threadIdx.x * 8; i < k_end; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            const int local_i = i - k_start;
            float4 x0 = *reinterpret_cast<const float4*>(x + local_i);
            float4 x1 = *reinterpret_cast<const float4*>(x + local_i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[row] = red[0];
        __syncthreads();
    }
}

// T-COLUMN twin of matvec_bf16_f32acc_x4 (lane/glm5-verify-batch, the varlen-batched-cores
// pattern): one block owns 4 output rows for ALL t tokens, so the weight pack is loaded ONCE
// per (row, K-step) and reused across tokens — vs the _rows twin's grid.y=t per-token weight
// re-read. EXACTNESS BY CONSTRUCTION (LAW:vl-bit-identity-order-pinning): each token keeps
// its OWN single-chain f32 accumulator fed in the exact per-token add order of the t=1
// kernel (the 8 pack lanes ascending, K ascending), and the shared-tree reduce runs once per
// token with the identical red[256] strided tree — so output (row, token) is bit-identical
// to the t=1 launch for that token. Gated by glm5_verify_batch_gpu's tcols bit-gate.
// T is a runtime arg bounded by MEMRA_BF16_TCOLS_MAX; the host launcher enforces it.
#define MEMRA_BF16_TCOLS_MAX 8
extern "C" __global__ void matvec_bf16_f32acc_x4_tcols(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc[MEMRA_BF16_TCOLS_MAX];
        for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
        const int stride = blockDim.x * 8;
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            for (int tt = 0; tt < t; tt++) {
                const float* xr = x + (size_t)tt * in_f;
                float4 x0 = *reinterpret_cast<const float4*>(xr + i);
                float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
                float a = acc[tt];
                a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
                acc[tt] = a;
            }
        }
        for (int tt = 0; tt < t; tt++) {
            red[threadIdx.x] = acc[tt];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) y[(size_t)tt * out_f + row] = red[0];
            __syncthreads();
        }
    }
}

// WIDE-T TWIN of matvec_bf16_f32acc_x4_tcols (lane/glm5-matvec, MEMRA_BF16_TCOLS_WIDE):
// t = 9..=16 — the DFlash2 drafter reuses the TARGET's lm head over its block's nd = 15
// mask-fill rows, which the t<=8 tcols launcher refuses, so the ship shape re-read the
// 1.27 GB head 15x per round through the _rows grid.y=t kernel (diet-battery c8-ship
// census: 60 calls x 5.31 ms = 11.7% of capture GPU). A SEPARATE kernel (not a raised
// MEMRA_BF16_TCOLS_MAX) so the priced t<=8 class keeps its acc[8] register footprint and
// SASS untouched — the qmatvec _tw32 lesson: sizing acc[] for the widest launch cost the
// t=2 verify ~10%. Body otherwise VERBATIM: per-token order-pinned single-chain
// accumulators (8 pack lanes ascending, K ascending), identical red[256] strided tree per
// token — bit-identical per (row, token) to the t=1 program, gated alongside the t<=8
// twin (glm5_matvec_doors_gpu).
#define MEMRA_BF16_TCOLS16_MAX 16
extern "C" __global__ void matvec_bf16_f32acc_x4_tcols16(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    __shared__ float red[256];
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc[MEMRA_BF16_TCOLS16_MAX];
        for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
        const int stride = blockDim.x * 8;
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            for (int tt = 0; tt < t; tt++) {
                const float* xr = x + (size_t)tt * in_f;
                float4 x0 = *reinterpret_cast<const float4*>(xr + i);
                float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
                float a = acc[tt];
                a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
                acc[tt] = a;
            }
        }
        for (int tt = 0; tt < t; tt++) {
            red[threadIdx.x] = acc[tt];
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) y[(size_t)tt * out_f + row] = red[0];
            __syncthreads();
        }
    }
}

// X1-GRID TWIN of matvec_bf16_f32acc_x4_tcols (lane/glm5-matvec, MEMRA_BF16_TCOLS_X1):
// ONE output row per block (grid.x = out_f), the p-loop dropped. WHY: the trunk kda
// shapes launch out_f/4 = 512..2048 blocks of 128 threads — about one resident wave on
// this card class, so every block's bit-pinned tree-reduce phases align and DRAM idles
// between them (c8-ship census: 1.05 TB/s = 59% of peak on the 67.1 MB kda calls, while
// the SAME x4 kernel at the lm head's 38720-block grid runs 1.43 TB/s = 80%). 4x the
// blocks re-creates the cross-block load/reduce overlap. Per-row body and red[256] tree
// VERBATIM — each row's float program is unchanged, so output (row, token) stays
// bit-identical to the x4 twin and to the t=1 program.
extern "C" __global__ void matvec_bf16_f32acc_x1_tcols(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    __shared__ float red[256];
    int row = blockIdx.x;
    if (row >= out_f) return;
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc[MEMRA_BF16_TCOLS_MAX];
    for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
    const int stride = blockDim.x * 8;
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        for (int tt = 0; tt < t; tt++) {
            const float* xr = x + (size_t)tt * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xr + i);
            float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
            float a = acc[tt];
            a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            acc[tt] = a;
        }
    }
    for (int tt = 0; tt < t; tt++) {
        red[threadIdx.x] = acc[tt];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) y[(size_t)tt * out_f + row] = red[0];
        __syncthreads();
    }
}

// ---- Door R (lane/glm5-door-r, MEMRA_BF16_TCOLS_RED_FUSED): fused-t reduce tails. ----
//
// WHY (moe-loc LANE.md §2.2): after door X the kda trunk's tcols calls sit at 67.0% of peak
// because the reduce tail runs t SEPARATE 7-level strided trees, 9 block-wide barriers per
// token column (~30 at t=3.34, 135 at the drafter head's t=15) against a 4-iteration main
// loop — the kernel is barrier/tail-bound, not DRAM-bound. The `_rf` twins restructure ONLY
// the tail: (a) one shared region per token column (`red[t*blockDim.x]`, dynamic) so the t
// trees share ONE barrier sequence — a single store barrier plus one barrier per block-wide
// level down to s=32; (b) levels s<=16 are intra-warp (after the s=32 level only indices
// 0..31 of each column hold live partials), so they become a `__shfl_down_sync` chain with
// the IDENTICAL pairing and the IDENTICAL `v = v + v[i+off]` operand order as the strided
// tree, zero barriers, one column per warp. Barriers per block: 9t -> 3 (x1 form at 128
// threads). BIT-IDENTICAL by pairing preservation: at every level the same index pairs are
// added in the same operand order as `red[i] += red[i+s]` (induction: after level s, lanes
// i<s hold exactly the tree's red[i]); the main loop is VERBATIM, so output (row, token)
// matches the standing twins and the t=1 program bit-for-bit. The gate's shifted-pairing
// RED (`..._rf_redshift` below) proves the bar can see an association change.
// CONTRACT: blockDim.x must be a POWER OF TWO >= 32 (the block-wide loop must pass exactly
// through s=32); the host launcher refuses the door for any other MEMRA_MMV_BLOCK. Dynamic
// shared = t * blockDim.x * 4 bytes, passed at launch.
extern "C" __global__ void matvec_bf16_f32acc_x1_tcols_rf(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    extern __shared__ float red[]; // t * blockDim.x floats
    int row = blockIdx.x;
    if (row >= out_f) return;
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc[MEMRA_BF16_TCOLS_MAX];
    for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
    const int stride = blockDim.x * 8;
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        for (int tt = 0; tt < t; tt++) {
            const float* xr = x + (size_t)tt * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xr + i);
            float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
            float a = acc[tt];
            a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            acc[tt] = a;
        }
    }
    for (int tt = 0; tt < t; tt++) red[tt * blockDim.x + threadIdx.x] = acc[tt];
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            for (int tt = 0; tt < t; tt++)
                red[tt * blockDim.x + threadIdx.x] += red[tt * blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = (int)blockDim.x >> 5;
    for (int tt = wid; tt < t; tt += nw) {
        float v = red[tt * blockDim.x + lane];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[(size_t)tt * out_f + row] = v;
    }
}

// Fused-tail twin of matvec_bf16_f32acc_x4_tcols (door R): p-loop body VERBATIM, tail as
// above; one trailing barrier per p iteration protects the shared region before the next
// row's stores (the standing kernel's own trailing sync, once per p instead of once per t).
extern "C" __global__ void matvec_bf16_f32acc_x4_tcols_rf(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    extern __shared__ float red[]; // t * blockDim.x floats
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc[MEMRA_BF16_TCOLS_MAX];
        for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
        const int stride = blockDim.x * 8;
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            for (int tt = 0; tt < t; tt++) {
                const float* xr = x + (size_t)tt * in_f;
                float4 x0 = *reinterpret_cast<const float4*>(xr + i);
                float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
                float a = acc[tt];
                a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
                acc[tt] = a;
            }
        }
        for (int tt = 0; tt < t; tt++) red[tt * blockDim.x + threadIdx.x] = acc[tt];
        __syncthreads();
        for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
            if (threadIdx.x < s) {
                for (int tt = 0; tt < t; tt++)
                    red[tt * blockDim.x + threadIdx.x] +=
                        red[tt * blockDim.x + threadIdx.x + s];
            }
            __syncthreads();
        }
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        const int nw = (int)blockDim.x >> 5;
        for (int tt = wid; tt < t; tt += nw) {
            float v = red[tt * blockDim.x + lane];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) y[(size_t)tt * out_f + row] = v;
        }
        __syncthreads();
    }
}

// Fused-tail twin of matvec_bf16_f32acc_x4_tcols16 (door R): the drafter head's t=15 case
// is the extreme win (135 barriers -> 6 per block at 128 threads); acc[16] stays SEPARATE
// from the priced t<=8 class (the `_tw32` acc-sizing lesson, same as the standing twin).
extern "C" __global__ void matvec_bf16_f32acc_x4_tcols16_rf(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    extern __shared__ float red[]; // t * blockDim.x floats
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int row = blockIdx.x * 4 + p;
        if (row >= out_f) return;
        const unsigned short* wr = w + (size_t)row * in_f;
        float acc[MEMRA_BF16_TCOLS16_MAX];
        for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
        const int stride = blockDim.x * 8;
        for (int i = threadIdx.x * 8; i < in_f; i += stride) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            for (int tt = 0; tt < t; tt++) {
                const float* xr = x + (size_t)tt * in_f;
                float4 x0 = *reinterpret_cast<const float4*>(xr + i);
                float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
                float a = acc[tt];
                a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
                acc[tt] = a;
            }
        }
        for (int tt = 0; tt < t; tt++) red[tt * blockDim.x + threadIdx.x] = acc[tt];
        __syncthreads();
        for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
            if (threadIdx.x < s) {
                for (int tt = 0; tt < t; tt++)
                    red[tt * blockDim.x + threadIdx.x] +=
                        red[tt * blockDim.x + threadIdx.x + s];
            }
            __syncthreads();
        }
        const int lane = threadIdx.x & 31;
        const int wid = threadIdx.x >> 5;
        const int nw = (int)blockDim.x >> 5;
        for (int tt = wid; tt < t; tt += nw) {
            float v = red[tt * blockDim.x + lane];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) y[(size_t)tt * out_f + row] = v;
        }
        __syncthreads();
    }
}

// GATE-ONLY shifted-pairing RED twin of matvec_bf16_f32acc_x1_tcols_rf: the warp phase runs
// the shuffle chain with ASCENDING offsets (1,2,4,8,16) — the same 32 partials summed under
// a DIFFERENT association (adjacent-pair tree instead of the halving tree). Mathematically
// the same sum; in f32 rounding it is not, so the door-R bit bar must see it. Never
// dispatched by any route — it exists so glm5_matvec_doors_gpu can prove the pairing bar
// bites (the `_vl` twin discipline's red arm for a REDUCTION restructure).
extern "C" __global__ void matvec_bf16_f32acc_x1_tcols_rf_redshift(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int t) {
    extern __shared__ float red[]; // t * blockDim.x floats
    int row = blockIdx.x;
    if (row >= out_f) return;
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc[MEMRA_BF16_TCOLS_MAX];
    for (int tt = 0; tt < t; tt++) acc[tt] = 0.0f;
    const int stride = blockDim.x * 8;
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        for (int tt = 0; tt < t; tt++) {
            const float* xr = x + (size_t)tt * in_f;
            float4 x0 = *reinterpret_cast<const float4*>(xr + i);
            float4 x1 = *reinterpret_cast<const float4*>(xr + i + 4);
            float a = acc[tt];
            a += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            a += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            a += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            a += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            a += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            a += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            a += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            a += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            acc[tt] = a;
        }
    }
    for (int tt = 0; tt < t; tt++) red[tt * blockDim.x + threadIdx.x] = acc[tt];
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            for (int tt = 0; tt < t; tt++)
                red[tt * blockDim.x + threadIdx.x] += red[tt * blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = (int)blockDim.x >> 5;
    for (int tt = wid; tt < t; tt += nw) {
        float v = red[tt * blockDim.x + lane];
        // THE SHIFT: ascending offsets — a different association of the same 32 partials.
        for (int off = 1; off <= 16; off <<= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[(size_t)tt * out_f + row] = v;
    }
}

// ---- Fused QKV F32 matvec (step TP decode v2, MEMRA_STEP_TP_QKV_FUSED). ----
// One launch computes all three rank-local projections from the shared input: block b maps to
// (weight, output, row) by range — rows [0, out_q) are Q, [out_q, out_q+out_kv) are K, the rest
// V. Reads the load-time F32 mirror (same values the chunked cuBLASLt program reads); f32
// accumulate, per-row deterministic two-level tree reduce. NUMERIC CLASS CHANGE vs the chunked
// cuBLASLt program (different per-row reduction order) — the door is default OFF and gated by
// the run-gen argmax gate + boot battery (DEV_ROUTES acceptance class).
extern "C" __global__ void matvec_f32_qkv(
        const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ wv,
        const float* __restrict__ wg, const float* __restrict__ x, float* __restrict__ yq,
        float* __restrict__ yk, float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    int r = blockIdx.x;
    const float* w;
    float* y;
    int row;
    if (r < out_q) {
        w = wq; y = yq; row = r;
    } else if (r < out_q + out_kv) {
        w = wk; y = yk; row = r - out_q;
    } else if (r < out_q + 2 * out_kv) {
        w = wv; y = yv; row = r - out_q - out_kv;
    } else {
        w = wg; y = yg; row = r - out_q - 2 * out_kv;
    }
    const float* wr = w + (size_t)row * in_f;
    float acc = 0.0f;
    const int stride = blockDim.x * 4;
#pragma unroll 4
    for (int i = threadIdx.x * 4; i < in_f; i += stride) {
        float4 wv4 = *reinterpret_cast<const float4*>(wr + i);
        float4 xv4 = *reinterpret_cast<const float4*>(x + i);
        acc += wv4.x * xv4.x + wv4.y * xv4.y + wv4.z * xv4.z + wv4.w * xv4.w;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}

// ---- Dual BF16 matvec (shared-expert gate+up, decode m=1, MEMRA_BF16_MMV). ----
// One launch computes both same-shape projections of the shared input: rows [0, out) are the
// gate, [out, 2*out) the up. Per row the body is identical to matvec_bf16_f32acc (same 16-byte
// loads, same bits<<16 contract, same tree reduce) — bit-identical to two separate launches.
extern "C" __global__ void matvec_bf16_dual(
        const unsigned short* __restrict__ wg, const unsigned short* __restrict__ wu,
        const float* __restrict__ x, float* __restrict__ yg, float* __restrict__ yu,
        int in_f, int out_f) {
    int r = blockIdx.x;
    const unsigned short* w;
    float* y;
    int row;
    if (r < out_f) {
        w = wg; y = yg; row = r;
    } else {
        w = wu; y = yu; row = r - out_f;
    }
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc = 0.0f;
    const int stride = blockDim.x * 8;
    // 4x unroll: the 4096-wide rows give each 128-thread block only 4 iterations —
    // too shallow to hide DRAM latency (qkvg measured 1.09 TB/s vs sel_v2's 1.5).
    // A single sequential accumulator keeps the FP order IDENTICAL at any unroll.
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
        acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
        acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
        acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
        acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
        acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
        acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
        acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
        acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}

// Row-offset twin of axpy_rows_seq_md_f32 (spec verify t-column combine): accumulate rows
// [row0, row0+n_rows) of a taller partial slab — the exact sequential per-pair FP chain of
// the base kernel over that window, so a column's combine over its own 8 pairs is bit-equal
// to its t=1 combine.
extern "C" __global__ void axpy_rows_seq_md_off_f32(
        const float* __restrict__ x, const float* __restrict__ w,
        const float* __restrict__ md, const int* __restrict__ sel, float* __restrict__ y,
        int width, int n_rows, int row0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float acc = 0.0f;
    for (int p = 0; p < n_rows; p++) {
        int pp = row0 + p;
        acc += w[pp] * md[sel[pp]] * x[(size_t)pp * width + i];
    }
    y[i] = acc;
}

extern "C" __global__ void qk_norm_rope_f32(
        const float* __restrict__ q_raw, const float* __restrict__ k_raw,
        const float* __restrict__ qw, const float* __restrict__ kw,
        float* __restrict__ q_out, float* __restrict__ k_out,
        const int* __restrict__ pos,
        int head_dim, int n_dims, int nh_q, float eps, float theta_scale, float freq_scale,
        const float* __restrict__ ff) {
    int h = blockIdx.x;
    const float* xr;
    const float* w;
    float* dr;
    if (h < nh_q) {
        xr = q_raw + (size_t)h * head_dim;
        w = qw;
        dr = q_out + (size_t)h * head_dim;
    } else {
        int r = h - nh_q;
        xr = k_raw + (size_t)r * head_dim;
        w = kw;
        dr = k_out + (size_t)r * head_dim;
    }
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) { float v = xr[i]; sum += v * v; }
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
    float scale = rsqrtf(s[0] / head_dim + eps);
    __shared__ float row[512]; // head_dim <= 512 (launcher-enforced)
    for (int i = tid; i < head_dim; i += blockDim.x) row[i] = xr[i] * scale * w[i];
    __syncthreads();
    int half = n_dims / 2;
    for (int j = tid; j < head_dim; j += blockDim.x) {
        if (j < half) {
            float theta = (float)pos[0] * powf(theta_scale, (float)j) * freq_scale;
            if (ff) theta = (float)pos[0] * powf(theta_scale, (float)j) / ff[j] * freq_scale;
            float c = cosf(theta), sn = sinf(theta);
            float x0 = row[j];
            float x1 = row[j + half];
            dr[j] = x0 * c - x1 * sn;
            dr[j + half] = x0 * sn + x1 * c;
        } else if (j >= n_dims) {
            dr[j] = row[j];
        }
    }
}

// ---- Four-block F32 matvec with in-order block accumulation (step TP O partial, t=1). ----
// One launch computes a rank's whole O partial: per output row, the four canonical input-column
// blocks accumulate SEQUENTIALLY (b0 then b1 then b2 then b3 — v1's add-chain order per
// element), each block dot via the shared tree reduce. Replaces 4 cuBLASLt launches + the
// root-side 4-copy/8-add chain with one launch + one peer copy + one add. Same numeric-class
// door as the fused QKV projection.
extern "C" __global__ void matvec_f32_b4(
        const float* __restrict__ w0, const float* __restrict__ w1,
        const float* __restrict__ w2, const float* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    int tid = threadIdx.x;
    const float* ws[4] = { w0, w1, w2, w3 };
    float total = 0.0f;
    __shared__ float s[32];
    for (int b = 0; b < 4; b++) {
        const float* wr = ws[b] + (size_t)r * block_cols;
        const float* xb = x + (size_t)b * block_cols;
        float acc = 0.0f;
        for (int i = tid * 4; i < block_cols; i += blockDim.x * 4) {
            float4 wv4 = *reinterpret_cast<const float4*>(wr + i);
            float4 xv4 = *reinterpret_cast<const float4*>(xb + i);
            acc += wv4.x * xv4.x + wv4.y * xv4.y + wv4.z * xv4.z + wv4.w * xv4.w;
        }
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid == 0) {
            float v = 0.0f;
            for (int wi = 0; wi < (blockDim.x + 31) / 32; wi++) v += s[wi];
            total += v;
        }
        __syncthreads();
    }
    if (tid == 0) y[r] = total;
}

// ---- Sequential weighted row-sum (device routes combine, t=1). ----
// y[i] = sum_p w[p] * x[p*width + i], accumulated in row order p=0..n_rows — the exact
// per-element FP chain of the reset + n_rows sequential axpy launches it replaces
// (0.0f + w0*x0 == w0*x0 bitwise; each subsequent add identical). One launch, no reset.
extern "C" __global__ void axpy_rows_seq_f32(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
        int width, int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float acc = 0.0f;
    for (int p = 0; p < n_rows; p++) acc += w[p] * x[(size_t)p * width + i];
    y[i] = acc;
}

// Token-major twin: each output token reduces its own contiguous `slots` route rows in canonical
// slot order. Used by batched EP decode after owner ranks scatter down rows into the root slab.
extern "C" __global__ void axpy_rows_seq_tokens_f32(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
        int width, int slots, int tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y;
    if (i >= width || token >= tokens) return;
    float acc = 0.0f;
    int row0 = token * slots;
    for (int slot = 0; slot < slots; ++slot) {
        int row = row0 + slot;
        acc += w[row] * x[(size_t)row * width + i];
    }
    y[(size_t)token * width + i] = acc;
}

// ---- Clamped selected-experts SwiGLU (step35 routed clamp, device routes program). ----
// step35 clamp semantics (llama-graph.cpp:2146, tp.rs step_expert_activation_host): the SiLU
// arm clamps ABOVE only (min(silu, limit)), the linear arm symmetrically (clamp(up, +-limit)).
// Same warp-per-32-block q8_1 emission as silu_mul_scaled_q8_1_sel. Dispatch only with
// limit > 1e-6 (the upstream eps gate).
extern "C" __global__ void silu_mul_scaled_q8_1_sel_clamp(
        const float* __restrict__ gate, const float* __restrict__ up,
        const float* __restrict__ gmac, const float* __restrict__ umac,
        const int* __restrict__ sel, float limit,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_per, int n_sel) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nblk_per = n_per / 32;
    if (warp >= nblk_per * n_sel) return;
    int t = warp / nblk_per;
    int e = sel[t];
    float gs = gmac[e], us = umac[e];
    int i = warp * 32 + lane;
    float g = gate[i] * gs;
    float silu = g / (1.0f + expf(-g));
    float u = fmaxf(fminf(up[i] * us, limit), -limit);
    float r = fminf(silu, limit) * u;
    float amax = fabsf(r);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float d = amax / 127.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    out_q[i] = (signed char)__float2int_rn(r * id);
    if (lane == 0) out_d[warp] = d;
}

// ---- Device-routed combine: axpy_rows_seq with the weight fold in-kernel. ----
// w[p] = w_route[p] * md[sel[p]] — the same single f32 multiply the host fold performs —
// then the identical sequential row accumulation of axpy_rows_seq_f32.
extern "C" __global__ void axpy_rows_seq_md_f32(
        const float* __restrict__ x, const float* __restrict__ w_route,
        const float* __restrict__ md, const int* __restrict__ sel, float* __restrict__ y,
        int width, int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    float acc = 0.0f;
    for (int p = 0; p < n_rows; p++) {
        float w = w_route[p] * md[sel[p]];
        acc += w * x[(size_t)p * width + i];
    }
    y[i] = acc;
}

// ---- BF16 twins of the fused v2 attention kernels (MEMRA_STEP_TP_QKV_FUSED without the F32
// mirror). Weights read as raw checkpoint bf16 (bits<<16 exact expansion), f32 accumulate —
// same VALUES as the F32-mirror kernels, 8-elems-per-thread-iteration accumulation (the
// matvec_bf16_f32acc grouping), so a different per-row FP order: same numeric-class door.
// Halves the attention weight read traffic (218 -> 109 MB per rank per layer on step37) and
// frees the mirror's VRAM.
extern "C" __global__ void matvec_bf16_qkvg(
        const unsigned short* __restrict__ wq, const unsigned short* __restrict__ wk,
        const unsigned short* __restrict__ wv, const unsigned short* __restrict__ wg,
        const float* __restrict__ x, float* __restrict__ yq, float* __restrict__ yk,
        float* __restrict__ yv, float* __restrict__ yg,
        int in_f, int out_q, int out_kv, int out_g) {
    int r = blockIdx.x;
    const unsigned short* w;
    float* y;
    int row;
    if (r < out_q) {
        w = wq; y = yq; row = r;
    } else if (r < out_q + out_kv) {
        w = wk; y = yk; row = r - out_q;
    } else if (r < out_q + 2 * out_kv) {
        w = wv; y = yv; row = r - out_q - out_kv;
    } else {
        w = wg; y = yg; row = r - out_q - 2 * out_kv;
    }
    const unsigned short* wr = w + (size_t)row * in_f;
    float acc = 0.0f;
    const int stride = blockDim.x * 8;
    // 4x unroll: the 4096-wide rows give each 128-thread block only 4 iterations —
    // too shallow to hide DRAM latency (qkvg measured 1.09 TB/s vs sel_v2's 1.5).
    // A single sequential accumulator keeps the FP order IDENTICAL at any unroll.
#pragma unroll 4
    for (int i = threadIdx.x * 8; i < in_f; i += stride) {
        uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4);
        acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
        acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
        acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
        acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
        acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
        acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
        acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
        acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
    }
    __shared__ float s[32];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (threadIdx.x == 0) y[row] = v;
    }
}

extern "C" __global__ void matvec_bf16_b4(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    int r = blockIdx.x;
    if (r >= out_f) return;
    int tid = threadIdx.x;
    const unsigned short* ws4[4] = { w0, w1, w2, w3 };
    float total = 0.0f;
    __shared__ float s[32];
    for (int b = 0; b < 4; b++) {
        const unsigned short* wr = ws4[b] + (size_t)r * block_cols;
        const float* xb = x + (size_t)b * block_cols;
        float acc = 0.0f;
#pragma unroll 4
        for (int i = tid * 8; i < block_cols; i += blockDim.x * 8) {
            uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
            float4 x0 = *reinterpret_cast<const float4*>(xb + i);
            float4 x1 = *reinterpret_cast<const float4*>(xb + i + 4);
            acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
            acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
            acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
            acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
            acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
            acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
            acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
            acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((tid & 31) == 0) s[tid >> 5] = acc;
        __syncthreads();
        if (tid == 0) {
            float v = 0.0f;
            for (int wi = 0; wi < (blockDim.x + 31) / 32; wi++) v += s[wi];
            total += v;
        }
        __syncthreads();
    }
    if (tid == 0) y[r] = total;
}

// B4 x2 twin (the #2b grid-halving receipt): half the blocks, each running the EXACT
// matvec_bf16_b4 per-row program on rows r and r+half sequentially — the second row's
// stream hides the first row's reduce tail. Bit-identical per row (same sub-dot order,
// same shfl+s[32] reduce, same thread-0 accumulation).
extern "C" __global__ void matvec_bf16_b4_x2(
        const unsigned short* __restrict__ w0, const unsigned short* __restrict__ w1,
        const unsigned short* __restrict__ w2, const unsigned short* __restrict__ w3,
        const float* __restrict__ x, float* __restrict__ y, int block_cols, int out_f) {
    int tid = threadIdx.x;
    const unsigned short* ws4[4] = { w0, w1, w2, w3 };
    __shared__ float s[32];
    const int half = (out_f + 1) >> 1;
    for (int p = 0; p < 2; p++) {
        int r = blockIdx.x + p * half;
        if (r >= out_f) return;
        float total = 0.0f;
        for (int b = 0; b < 4; b++) {
            const unsigned short* wr = ws4[b] + (size_t)r * block_cols;
            const float* xb = x + (size_t)b * block_cols;
            float acc = 0.0f;
    #pragma unroll 4
        for (int i = tid * 8; i < block_cols; i += blockDim.x * 8) {
                uint4 pack = *reinterpret_cast<const uint4*>(wr + i);
                const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pack);
                float4 x0 = *reinterpret_cast<const float4*>(xb + i);
                float4 x1 = *reinterpret_cast<const float4*>(xb + i + 4);
                acc += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                acc += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                acc += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                acc += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                acc += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                acc += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                acc += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                acc += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if ((tid & 31) == 0) s[tid >> 5] = acc;
            __syncthreads();
            if (tid == 0) {
                float v = 0.0f;
                for (int wi = 0; wi < (blockDim.x + 31) / 32; wi++) v += s[wi];
                total += v;
            }
            __syncthreads();
        }
        if (tid == 0) y[r] = total;
    }
}

// Token-graph chunk loop (step37 F-lite): append the freshly argmax'd token id into a device
// history ring at a device-counter index — the same replayed graph writes slot 0,1,2,... as
// the chunk advances, and the host reads the whole ring back once per chunk.
extern "C" __global__ void u32_hist_append(
        const unsigned int* __restrict__ tok, unsigned int* __restrict__ hist,
        int* __restrict__ idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int i = *idx;
        hist[i] = *tok;
        *idx = i + 1;
    }
}

// ---------------------------------------------------------------------------
// BF16 -> q8_0 WEIGHT ENCODER (MEMRA_STEP_TP_W8, 2026-08-25).
//
// The step37 attention projections ship bf16 and are the single largest byte
// class this box streams per decode token (~3.4 GB of ~6.5 GB per card). The
// q8 mmvq path measures 14.0 us at the fused qkv shape against bf16's 23.0 and
// 11.7 against o_proj's 24.2, so a q8_0 mirror of those weights is worth
// ~-1.0 ms/token. This kernel builds that mirror once at load.
//
// Output is ggml q8_0: per 32 weights, [half d][32 x int8], 34 bytes, d =
// amax/127 and q = lrintf(x/d) clamped to +-127 — the SAME block program
// quant_K_block writes for the KV cache, so one format description covers both.
// One warp owns one 32-block; grid is (in_f/32, out_f).
extern "C" __global__ void encode_q8_0_rows_from_bf16(
        const unsigned short* __restrict__ w, unsigned char* __restrict__ out,
        int in_f, int out_f) {
    // FLAT 1D grid over (row, 32-block) pairs. Rows on blockIdx.y would cap the kernel at
    // 65535 output rows, and the LM head has 128896 — that launch returned
    // CUDA_ERROR_INVALID_VALUE the first time the head reached this encoder.
    const int nblk_row = in_f / 32;
    const long long pair = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    const int row = (int)(pair / nblk_row);
    const int blk = (int)(pair % nblk_row);
    const int lane = threadIdx.x;            // 0..31
    if (row >= out_f || blk * 32 >= in_f) return;
    const int eidx = blk * 32 + lane;
    const float x = (eidx < in_f)
        ? __uint_as_float((unsigned)w[(size_t)row * in_f + eidx] << 16)
        : 0.0f;
    float amax = fabsf(x);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int q = (int)lrintf(x * id);
    q = max(-127, min(127, q));
    const int nblk = in_f / 32;
    unsigned char* dst = out + ((size_t)row * nblk + blk) * 34;
    if (lane == 0) *(half*)dst = __float2half(d);
    ((signed char*)(dst + 2))[lane] = (signed char)q;
}

// FUSED q8_0 QKV (MEMRA_STEP_TP_W8, 2026-08-25). The first W8 wiring issued one mmvq per
// projection and MEASURED SLOWER than the bf16 fused kernel (79.52 vs 80.72 tok/s): three
// launches plus the activation quantize cost more than the halved weight bytes saved. This
// kernel restores the single launch — one warp per output row across the stacked
// q/k/v rows, each row running the EXACT qmatvec_q8_0_mmvq_rp per-row program (same block
// walk, same dp4a order, same warp reduce), so it is bit-identical to the per-matrix calls
// it replaces. Gate rows stay bf16 and are not part of this kernel.
extern "C" __global__ void qmatvec_q8_0_qkv_rp(
        const unsigned char* __restrict__ Wq, const unsigned char* __restrict__ Wk,
        const unsigned char* __restrict__ Wv,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
        int in_f, int out_q, int out_kv) {
    MEMRA_PDL_ENTRY();
    const int r = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    const int rows = out_q + 2 * out_kv;
    if (r >= rows) return;
    const unsigned char* W;
    float* y;
    int row, plane_rows;
    if (r < out_q) {
        W = Wq; y = yq; row = r; plane_rows = out_q;
    } else if (r < out_q + out_kv) {
        W = Wk; y = yk; row = r - out_q; plane_rows = out_kv;
    } else {
        W = Wv; y = yv; row = r - out_q - out_kv; plane_rows = out_kv;
    }
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, plane_rows, row, nblk, &wq, &wd);
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(aq + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * ad[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = acc;
}

// FUSED q8_0 O-PROJECTION over the HEAD_SPLIT blocks (MEMRA_STEP_TP_W8, 2026-08-25). The
// bf16 twin (matvec_bf16_b4) is 24.2 us/layer at 1.38 TB/s and the q8 mmvq path runs the
// same shape in 11.7 at 1.52, the largest single decode line left after the QKV arm landed
// +2.9%. One warp per output row; per block b the warp walks that block's planar q8_0
// mirror with its own segment of the q8_1 activation, warp-reduces, and adds into the row
// total — the same per-block-then-add shape matvec_bf16_b4 uses.
extern "C" __global__ void qmatvec_q8_0_b4_rp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int block_cols, int out_f) {
    MEMRA_PDL_ENTRY();
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = block_cols / 32;               // 32-blocks per HEAD_SPLIT block
    const unsigned char* planes[4] = { W0, W1, W2, W3 };
    float total = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned char* wq; const unsigned short* wd;
        q8_0_rp_planes(planes[b], out_f, row, nblk, &wq, &wd);
        const signed char* arow = aq + (size_t)b * block_cols;
        const float* adrow = ad + (size_t)b * nblk;
        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) {
            int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            float dw = half_to_float(wd[blk]);
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc += dw * adrow[blk] * (float)sumi;
        }
        total += warp_reduce_sum(acc);
    }
    if (lane == 0) y[row] = total;
}

// T-COLUMN q8_0 TWINS OF THE VERIFY WALK'S TWO HOT KERNELS (MEMRA_STEP_TP_W8, 2026-08-26).
// nsys on a K=1 spec window put matvec_bf16_b4_tcol at 24.8% of GPU time and
// matvec_bf16_qkvg_tcol at 12.3% — 37% of the verify in two BF16 kernels, because the W8 door
// had only replaced the DECODE twins. These read the same planar q8_0 mirrors the decode arm
// uses, one warp per (row, column): per-row program identical to qmatvec_q8_0_qkv_rp /
// qmatvec_q8_0_b4_rp, so a t-column call is bit-identical to t separate single-row calls.
// blockIdx.y is the verify column; the activation is q8_1 per column (aq/ad strided by column).
extern "C" __global__ void qmatvec_q8_0_qkv_rp_t(
        const unsigned char* __restrict__ Wq, const unsigned char* __restrict__ Wk,
        const unsigned char* __restrict__ Wv,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
        int in_f, int out_q, int out_kv) {
    MEMRA_PDL_ENTRY();
    const int col = blockIdx.y;
    const int r = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    const int rows = out_q + 2 * out_kv;
    if (r >= rows) return;
    const unsigned char* W;
    float* y;
    int row, plane_rows, ystride;
    if (r < out_q) {
        W = Wq; y = yq; row = r; plane_rows = out_q; ystride = out_q;
    } else if (r < out_q + out_kv) {
        W = Wk; y = yk; row = r - out_q; plane_rows = out_kv; ystride = out_kv;
    } else {
        W = Wv; y = yv; row = r - out_q - out_kv; plane_rows = out_kv; ystride = out_kv;
    }
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, plane_rows, row, nblk, &wq, &wd);
    const signed char* arow = aq + (size_t)col * in_f;
    const float* adrow = ad + (size_t)col * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)col * ystride + row] = acc;
}

// T-column q8_0 o_proj over the four HEAD_SPLIT blocks; blockIdx.y is the verify column.
extern "C" __global__ void qmatvec_q8_0_b4_rp_t(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int block_cols, int out_f) {
    MEMRA_PDL_ENTRY();
    const int col = blockIdx.y;
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = block_cols / 32;
    const unsigned char* planes[4] = { W0, W1, W2, W3 };
    const signed char* acol = aq + (size_t)col * 4 * block_cols;
    const float* adcol = ad + (size_t)col * 4 * nblk;
    float total = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned char* wq; const unsigned short* wd;
        q8_0_rp_planes(planes[b], out_f, row, nblk, &wq, &wd);
        const signed char* arow = acol + (size_t)b * block_cols;
        const float* adrow = adcol + (size_t)b * nblk;
        float acc = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) {
            int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            float dw = half_to_float(wd[blk]);
            const int4* aq16 = (const int4*)(arow + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc += dw * adrow[blk] * (float)sumi;
        }
        total += warp_reduce_sum(acc);
    }
    if (lane == 0) y[(size_t)col * out_f + row] = total;
}

// T-COLUMN q8_0 SINGLE-MATRIX GEMV (MEMRA_STEP_TP_W8 + MEMRA_W8_HYBRID, 2026-08-26). The verify
// walk's shexp/dense rows still ran bf16 (`matvec_bf16_f32acc_x4_rows`: 78 launches/round at
// 56.5 us = ~162 ms of GPU in a 37-round capture) because the mirror routing only fired at t==1.
// One warp per (row, column); per-row program identical to qmatvec_q8_0_mmvq_rp, so a t-column
// call is bit-identical to t single-row calls.
extern "C" __global__ void qmatvec_q8_0_rows_t(
        const unsigned char* __restrict__ W,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f) {
    MEMRA_PDL_ENTRY();
    const int col = blockIdx.y;
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, row, nblk, &wq, &wd);
    const signed char* arow = aq + (size_t)col * in_f;
    const float* adrow = ad + (size_t)col * nblk;
    float acc = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* aq16 = (const int4*)(arow + blk * 32);
        int4 a01 = aq16[0], a23 = aq16[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * adrow[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[(size_t)col * out_f + row] = acc;
}

// WEIGHT-ONCE T-COLUMN TWINS (`_tw`, MEMRA_Q8T_WONCE, 2026-08-27). The `_t` forms above put the
// verify column on blockIdx.y — a full row-grid copy per column — and load weights with __ldcs,
// the STREAMING evict-first modifier. At t=1 that pairing is right (weights have no reuse); at
// t>1 it is exactly wrong: the columns share every weight byte and the second column re-reads
// them all from DRAM through a path that refuses cache retention. nsys (2026-08-27, K=1 doors):
// qkv_rp_t 36.5 us vs qkv_rp 21.8 (1.67x for 2 columns), b4_rp_t 29.0 vs 20.3 (1.43x), where a
// weight-bound kernel should scale ~1.1x. These twins drop the column grid axis, load each weight
// int4 ONCE, and run `t` independent accumulator chains against it.
//
// EXACTNESS (LAW:vl-bit-identity-order-pinning): each column keeps its own single-dependence
// accumulator chain over the SAME lane-strided blk order as the `_t` form, reduced by the same
// warp_reduce_sum — the per-column float program is unchanged, only the weight-load schedule
// moves. ptxas must not be given a reason to merge chains: the accumulators live in a plain
// array indexed by a #pragma unroll'd loop, never summed across columns.
#define MEMRA_Q8T_TMAX 8
// Wide twins (_tw32) carry their own accumulator bound so the t<=8 kernels keep their
// register footprint: acc[32] in every launch sized registers for 32 columns and cost the
// t=2 decode verify ~10% (naked-spec 83.6 vs 93.2, caught by the deploy gate 2026-08-27).
#define MEMRA_Q8T_TMAX_WIDE 32

extern "C" __global__ void qmatvec_q8_0_qkv_rp_tw(
        const unsigned char* __restrict__ Wq, const unsigned char* __restrict__ Wk,
        const unsigned char* __restrict__ Wv,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
        int in_f, int out_q, int out_kv, int t) {
    MEMRA_PDL_ENTRY();
    const int r = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    const int rows = out_q + 2 * out_kv;
    if (r >= rows) return;
    const unsigned char* W;
    float* y;
    int row, plane_rows, ystride;
    if (r < out_q) {
        W = Wq; y = yq; row = r; plane_rows = out_q; ystride = out_q;
    } else if (r < out_q + out_kv) {
        W = Wk; y = yk; row = r - out_q; plane_rows = out_kv; ystride = out_kv;
    } else {
        W = Wv; y = yv; row = r - out_q - out_kv; plane_rows = out_kv; ystride = out_kv;
    }
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, plane_rows, row, nblk, &wq, &wd);
    float acc[MEMRA_Q8T_TMAX];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        for (int c = 0; c < t; c++) {
            const int4* aq16 = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    for (int c = 0; c < t; c++) {
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * ystride + row] = a;
    }
}

extern "C" __global__ void qmatvec_q8_0_qkv_rp_tw32(
        const unsigned char* __restrict__ Wq, const unsigned char* __restrict__ Wk,
        const unsigned char* __restrict__ Wv,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ yq, float* __restrict__ yk, float* __restrict__ yv,
        int in_f, int out_q, int out_kv, int t) {
    MEMRA_PDL_ENTRY();
    const int r = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    const int rows = out_q + 2 * out_kv;
    if (r >= rows) return;
    const unsigned char* W;
    float* y;
    int row, plane_rows, ystride;
    if (r < out_q) {
        W = Wq; y = yq; row = r; plane_rows = out_q; ystride = out_q;
    } else if (r < out_q + out_kv) {
        W = Wk; y = yk; row = r - out_q; plane_rows = out_kv; ystride = out_kv;
    } else {
        W = Wv; y = yv; row = r - out_q - out_kv; plane_rows = out_kv; ystride = out_kv;
    }
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, plane_rows, row, nblk, &wq, &wd);
    float acc[MEMRA_Q8T_TMAX_WIDE];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX_WIDE; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        for (int c = 0; c < t; c++) {
            const int4* aq16 = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    for (int c = 0; c < t; c++) {
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * ystride + row] = a;
    }
}

extern "C" __global__ void qmatvec_q8_0_b4_rp_tw(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int block_cols, int out_f, int t) {
    MEMRA_PDL_ENTRY();
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = block_cols / 32;
    const unsigned char* planes[4] = { W0, W1, W2, W3 };
    float total[MEMRA_Q8T_TMAX];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX; c++) total[c] = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned char* wq; const unsigned short* wd;
        q8_0_rp_planes(planes[b], out_f, row, nblk, &wq, &wd);
        float acc[MEMRA_Q8T_TMAX];
        #pragma unroll
        for (int c = 0; c < MEMRA_Q8T_TMAX; c++) acc[c] = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) {
            int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            float dw = half_to_float(wd[blk]);
            for (int c = 0; c < t; c++) {
                const signed char* arow = aq + (size_t)c * 4 * block_cols + (size_t)b * block_cols;
                const float* adrow = ad + (size_t)c * 4 * nblk + (size_t)b * nblk;
                const int4* aq16 = (const int4*)(arow + blk * 32);
                int4 a01 = aq16[0], a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                int sumi = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
                acc[c] += dw * adrow[blk] * (float)sumi;
            }
        }
        // Same per-plane reduce-then-add order as the `_t` form: total_c += reduce(acc_c) per b.
        for (int c = 0; c < t; c++) total[c] += warp_reduce_sum(acc[c]);
    }
    if (lane == 0)
        for (int c = 0; c < t; c++) y[(size_t)c * out_f + row] = total[c];
}

extern "C" __global__ void qmatvec_q8_0_b4_rp_tw32(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2, const unsigned char* __restrict__ W3,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int block_cols, int out_f, int t) {
    MEMRA_PDL_ENTRY();
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = block_cols / 32;
    const unsigned char* planes[4] = { W0, W1, W2, W3 };
    float total[MEMRA_Q8T_TMAX_WIDE];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX_WIDE; c++) total[c] = 0.0f;
    for (int b = 0; b < 4; b++) {
        const unsigned char* wq; const unsigned short* wd;
        q8_0_rp_planes(planes[b], out_f, row, nblk, &wq, &wd);
        float acc[MEMRA_Q8T_TMAX_WIDE];
        #pragma unroll
        for (int c = 0; c < MEMRA_Q8T_TMAX_WIDE; c++) acc[c] = 0.0f;
        for (int blk = lane; blk < nblk; blk += 32) {
            int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
            int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
            int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
            float dw = half_to_float(wd[blk]);
            for (int c = 0; c < t; c++) {
                const signed char* arow = aq + (size_t)c * 4 * block_cols + (size_t)b * block_cols;
                const float* adrow = ad + (size_t)c * 4 * nblk + (size_t)b * nblk;
                const int4* aq16 = (const int4*)(arow + blk * 32);
                int4 a01 = aq16[0], a23 = aq16[1];
                int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
                int sumi = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
                acc[c] += dw * adrow[blk] * (float)sumi;
            }
        }
        // Same per-plane reduce-then-add order as the `_t` form: total_c += reduce(acc_c) per b.
        for (int c = 0; c < t; c++) total[c] += warp_reduce_sum(acc[c]);
    }
    if (lane == 0)
        for (int c = 0; c < t; c++) y[(size_t)c * out_f + row] = total[c];
}

extern "C" __global__ void qmatvec_q8_0_rows_tw(
        const unsigned char* __restrict__ W,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int t) {
    MEMRA_PDL_ENTRY();
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, row, nblk, &wq, &wd);
    float acc[MEMRA_Q8T_TMAX];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        for (int c = 0; c < t; c++) {
            const int4* aq16 = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    for (int c = 0; c < t; c++) {
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + row] = a;
    }
}

extern "C" __global__ void qmatvec_q8_0_rows_tw32(
        const unsigned char* __restrict__ W,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int t) {
    MEMRA_PDL_ENTRY();
    const int row = blockIdx.x * MEMRA_MMVQ_ROWS + threadIdx.y;
    if (row >= out_f) return;
    const int lane = threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq; const unsigned short* wd;
    q8_0_rp_planes(W, out_f, row, nblk, &wq, &wd);
    float acc[MEMRA_Q8T_TMAX_WIDE];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX_WIDE; c++) acc[c] = 0.0f;
    for (int blk = lane; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        for (int c = 0; c < t; c++) {
            const int4* aq16 = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    for (int c = 0; c < t; c++) {
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + row] = a;
    }
}

// =====================================================================================
// B200 HBM-SPEED DECODE MATVECS (MEMRA_B200_GEMV_V2, lane/b200-gemv-hbm-20260902)
// =====================================================================================
//
// THE ROOFLINE THIS SET IS DESIGNED AGAINST. nsys, 2x B200 SXM (8 TB/s HBM3e, 148 SMs,
// 228 KB smem/SM), sm_100a build, GLM-5.3-Flash NVFP4 W4A16 mint, resident PP2, plain
// decode t=1, both devices summed, every occupancy door ON:
//
//   qmatvec_kda6_bf16f32           93.8us / ~200 MB  = 2.1 TB/s   (26% of HBM)
//   moe_gate_up_preclamp8_q8_w4    52.4us / ~50 MB   = 1.0 TB/s   (12%)
//   moe_down8_fma_q8_w4            28.2us / ~25 MB   = 0.9 TB/s   (11%)
//   matvec_bf16_f32acc_x4_rows_pf  23.6us / 64 MB    = 2.7 TB/s   (34%)
//
// On the RTX PRO 6000 (1.8 TB/s GDDR7, 188 SMs) the same kernels sit near their DRAM wall.
// On B200 they are 3 to 9x off it, and the previous lane's warp-packing/prefetch arms bought
// only ~5% — so the remaining gap is NOT block-slot occupancy. It is BYTES IN FLIGHT PER SM.
// Little's law at 8 TB/s and a ~700 ns HBM3e round trip needs ~5.6 MB of reads outstanding
// ACROSS THE DIE, i.e. ~38 KB per SM, at all times. The shipped kernels do not get there:
//
//   * matvec_bf16_f32acc_x4_rows walks its FOUR rows SEQUENTIALLY, each row ending in a full
//     red[] tree with a __syncthreads per step (28 barriers per block), and it re-reads the
//     f32 activation ONCE PER ROW. Per K step a thread has 1 weight load and 2 activation
//     loads in flight and then stalls on its own 8-fma chain; the activation traffic is 2x
//     the weight traffic it is trying to stream.
//   * the NVFP4 expert pair runs one un-unrolled g loop, so a lane holds one group's loads.
//   * moe_down8_fma_q8 walks its 8 experts SEQUENTIALLY inside ONE warp, so the whole kernel
//     is out_f warps wide (4096 warps = 0.43 of a full-occupancy B200 wave).
//
// Every kernel below is the SAME arithmetic, rescheduled: more independent loads in flight
// per thread, activations loaded once and reused across the rows that share them, reductions
// batched so R rows pay ONE barrier chain instead of R, and grids that cover the die. All of
// them are BIT-IDENTICAL to their shipped twin per output element (the split-K arm at the end
// is the one exception and is a NAMED numeric class). Door MEMRA_B200_GEMV_V2, default OFF.

// Rows per block for the v2 bf16 GEMV. 8 (not the shipped 4) because the activation is loaded
// ONCE per K step and reused across all R rows: at R=8 a block reads 8*in_f*2 B of weight
// against in_f*4 B of activation (4:1) where the shipped kernel reads 4*in_f*2 against
// 4*in_f*4 (1:2). It also keeps the grid honest: out_f=8192 -> 1024 CTAs over 148 SMs.
#define MEMRA_GEMV_V2_ROWS 8

// One dynamic-shared-memory window for the whole v2 family (R * blockDim.x floats). Declared
// once at file scope: a per-function `extern __shared__` of the same name is a linkage-conflict
// warning, and every kernel that takes dynamic smem aliases the same window anyway.
extern __shared__ float gemv_v2_red[];

// Per-thread K walk for R rows at once. `wr[]` are the R row bases (clamped to a live row in
// the tail block; those lanes' results are discarded, never written). Two-stage software
// pipeline: stage B's R weight loads plus the 2 activation loads issue BEFORE stage A's R x 8
// fma chains run, so at R=8 a thread has 10 independent loads (8 x 16 B weight + 2 x 16 B
// activation = 160 B) outstanding while it computes. At 128 threads/block and the ~5 resident
// blocks this register budget admits, that is ~100 KB in flight per SM, comfortably past the
// ~38 KB Little's-law floor.
//
// BIT-IDENTITY. For a given row, thread `tid` accumulates exactly the shipped kernel's subset
// (i = tid*8, +stride, ...) in exactly the shipped order, with the same 8 `acc +=` fma
// expressions on the same operands. Only the ISSUE order of loads belonging to DIFFERENT rows
// changes, and the R rows never share an accumulator. `ld.global.nc` (__ldg) changes the cache
// path, not the value.
template <int R>
__device__ __forceinline__ void gemv_v2_walk_bf16(
        const unsigned short* __restrict__ w, int in_f, int row0, int nrow,
        const float* __restrict__ x, float (&acc)[R], int k0, int k_end) {
    const unsigned short* wr[R];
#pragma unroll
    for (int p = 0; p < R; p++) wr[p] = w + (size_t)(row0 + min(p, nrow - 1)) * (size_t)in_f;
    const int stride = (int)blockDim.x * 8;
    int i = k0 + (int)threadIdx.x * 8;
    bool have = i < k_end;
    float4 xa0 = make_float4(0.f, 0.f, 0.f, 0.f), xa1 = xa0;
    uint4 pa[R];
#pragma unroll
    for (int p = 0; p < R; p++) pa[p] = make_uint4(0u, 0u, 0u, 0u);
    if (have) {
        xa0 = __ldg(reinterpret_cast<const float4*>(x + i));
        xa1 = __ldg(reinterpret_cast<const float4*>(x + i + 4));
#pragma unroll
        for (int p = 0; p < R; p++) pa[p] = __ldg(reinterpret_cast<const uint4*>(wr[p] + i));
    }
    while (have) {
        const int ni = i + stride;
        const bool have_next = ni < k_end;
        float4 xb0 = make_float4(0.f, 0.f, 0.f, 0.f), xb1 = xb0;
        uint4 pb[R];
#pragma unroll
        for (int p = 0; p < R; p++) pb[p] = make_uint4(0u, 0u, 0u, 0u);
        if (have_next) {
            xb0 = __ldg(reinterpret_cast<const float4*>(x + ni));
            xb1 = __ldg(reinterpret_cast<const float4*>(x + ni + 4));
#pragma unroll
            for (int p = 0; p < R; p++) pb[p] = __ldg(reinterpret_cast<const uint4*>(wr[p] + ni));
        }
#pragma unroll
        for (int p = 0; p < R; p++) {
            const unsigned short* wp = reinterpret_cast<const unsigned short*>(&pa[p]);
            acc[p] += __uint_as_float((unsigned)wp[0] << 16) * xa0.x;
            acc[p] += __uint_as_float((unsigned)wp[1] << 16) * xa0.y;
            acc[p] += __uint_as_float((unsigned)wp[2] << 16) * xa0.z;
            acc[p] += __uint_as_float((unsigned)wp[3] << 16) * xa0.w;
            acc[p] += __uint_as_float((unsigned)wp[4] << 16) * xa1.x;
            acc[p] += __uint_as_float((unsigned)wp[5] << 16) * xa1.y;
            acc[p] += __uint_as_float((unsigned)wp[6] << 16) * xa1.z;
            acc[p] += __uint_as_float((unsigned)wp[7] << 16) * xa1.w;
        }
        i = ni;
        have = have_next;
        xa0 = xb0;
        xa1 = xb1;
#pragma unroll
        for (int p = 0; p < R; p++) pa[p] = pb[p];
    }
}

// Batched block reduction for the R rows the walk above produced, in the SHIPPED tree's exact
// order: `red[t] += red[t + s]` for s = blockDim.x/2 down to 1. The R rows run in LOCKSTEP so
// the block pays ONE barrier per s instead of R chains of them (7 barriers instead of 28 at
// blockDim=128). Once s reaches 16 the remaining steps live inside one warp, and
// `__shfl_down_sync` by 16, 8, 4, 2, 1 pairs the SAME lanes in the SAME order as the smem
// tree does, so the warp tail is the tree, not an approximation of it — taken only when
// blockDim is a power of two (the shipped `s >>= 1` walk lands on 16 only then; mmv_block()
// admits 96/160/224 too, and those keep the smem loop verbatim, bug-compatible and all).
// `red` is R * blockDim.x floats of DYNAMIC shared memory (4 KB at R=8, blockDim=128).
template <int R>
__device__ __forceinline__ void gemv_v2_reduce_bf16(
        float* red, const float (&acc)[R], float* __restrict__ y, int row0, int nrow) {
    const int nb = (int)blockDim.x;
    const int tid = (int)threadIdx.x;
#pragma unroll
    for (int p = 0; p < R; p++) red[p * nb + tid] = acc[p];
    __syncthreads();
    int s = nb >> 1;
    for (; s >= 32; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int p = 0; p < R; p++) red[p * nb + tid] += red[p * nb + tid + s];
        }
        __syncthreads();
    }
    if ((nb & (nb - 1)) == 0) {
        const int warp = tid >> 5, lane = tid & 31, nwarps = nb >> 5;
        for (int p = warp; p < nrow; p += nwarps) {
            float v = red[p * nb + lane];
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) y[row0 + p] = v;
        }
        return;
    }
    for (; s > 0; s >>= 1) {
        if (tid < s) {
#pragma unroll
            for (int p = 0; p < R; p++) red[p * nb + tid] += red[p * nb + tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int p = 0; p < nrow; p++) y[row0 + p] = red[p * nb];
    }
}

// v2 BF16 GEMV, BIT-IDENTICAL twin of matvec_bf16_f32acc_x4_rows. grid = (out_f/8, t, 1),
// block = mmv_block() (the blockDim is part of the bit-identity claim: the reduction tree's
// shape is a function of it), dynamic smem = 8 * blockDim.x * 4 B.
extern "C" __global__ __launch_bounds__(256) void matvec_bf16_v2(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f) {
    const int R = MEMRA_GEMV_V2_ROWS;
    const int trow = blockIdx.y;
    x += (size_t)trow * in_f;
    y += (size_t)trow * out_f;
    const int row0 = blockIdx.x * R;
    const int nrow = min(R, out_f - row0);
    if (nrow <= 0) return;
    float acc[R];
#pragma unroll
    for (int p = 0; p < R; p++) acc[p] = 0.0f;
    gemv_v2_walk_bf16<R>(w, in_f, row0, nrow, x, acc, 0, in_f);
    gemv_v2_reduce_bf16<R>(gemv_v2_red, acc, y, row0, nrow);
}

// SPLIT-K arm. NAMED NUMERIC CLASS `bf16_gemv_v2_splitk` — NOT bit-identical: a row's K sum is
// split into `ksplit` contiguous chunks, each reduced independently, and the chunk partials are
// added by the combine kernel below. The class exists for shapes whose row grid cannot cover
// two waves of CTAs on this die (out_f/8 * t < 2 * SM count); the shipped GLM-5.3 KDA decode
// shapes (out_f 4096/8192 at t=1 -> 512/1024 CTAs vs 296) never reach it, which is why the
// door's dispatch is bit-identical in practice. The combine order is FIXED and ascending in k
// (never atomics, never a reduction whose order depends on scheduling), so the class is
// deterministic: the same input always produces the same bytes.
// Chunks are multiples of 8 elements so every thread's 16 B loads stay aligned.
extern "C" __global__ __launch_bounds__(256) void matvec_bf16_v2_sk(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ part, int in_f, int out_f, int ksplit) {
    const int R = MEMRA_GEMV_V2_ROWS;
    const int trow = blockIdx.y;
    const int ks = blockIdx.z;
    const int chunk = ((in_f / 8 + ksplit - 1) / ksplit) * 8;
    const int k0 = ks * chunk;
    if (k0 >= in_f) return;
    const int k_end = min(in_f, k0 + chunk);
    const float* xr = x + (size_t)trow * in_f;
    // partial plane ks, token row trow: part[((ks * gridDim.y) + trow) * out_f + row]
    float* pr = part + ((size_t)ks * gridDim.y + trow) * out_f;
    const int row0 = blockIdx.x * R;
    const int nrow = min(R, out_f - row0);
    if (nrow <= 0) return;
    float acc[R];
#pragma unroll
    for (int p = 0; p < R; p++) acc[p] = 0.0f;
    gemv_v2_walk_bf16<R>(w, in_f, row0, nrow, xr, acc, k0, k_end);
    gemv_v2_reduce_bf16<R>(gemv_v2_red, acc, pr, row0, nrow);
}

// FIXED-ORDER split-K combine: y[row] = sum over ks ASCENDING of part[ks][row]. One thread per
// (token, row); no atomics, no scheduling-dependent order.
extern "C" __global__ void matvec_bf16_v2_sk_combine(
        const float* __restrict__ part, float* __restrict__ y, int out_f, int t, int ksplit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_f * t) return;
    const int trow = idx / out_f;
    const int row = idx - trow * out_f;
    float s = 0.0f;
    for (int ks = 0; ks < ksplit; ks++) s += part[((size_t)ks * t + trow) * out_f + row];
    y[(size_t)trow * out_f + row] = s;
}

// v2 twin of the fused KDA six-projection BF16 kernel (qmatvec_kda6_bf16f32). Same six ranges
// in the same block order, same f32 rows through f32_mmvq_row1; the three BF16 ranges take the
// v2 walk at 8 rows/block instead of kda6_bf16_rows4's four sequential rows, so the fused
// launch inherits the whole activation-reuse / loads-in-flight / one-barrier-chain design.
// BIT-IDENTICAL per row to qmatvec_kda6_bf16f32 (which is itself bit-identical per row to
// matvec_bf16_f32acc_x4_rows). Block = mmv_block(); dynamic smem = 8 * blockDim.x * 4 B.
__device__ __forceinline__ void kda6_bf16_rows8_v2(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int blk) {
    const int R = MEMRA_GEMV_V2_ROWS;
    const int row0 = blk * R;
    const int nrow = min(R, out_f - row0);
    if (nrow <= 0) return;
    float acc[R];
#pragma unroll
    for (int p = 0; p < R; p++) acc[p] = 0.0f;
    gemv_v2_walk_bf16<R>(w, in_f, row0, nrow, x, acc, 0, in_f);
    gemv_v2_reduce_bf16<R>(gemv_v2_red, acc, y, row0, nrow);
}

extern "C" __global__ __launch_bounds__(256) void qmatvec_kda6_bf16f32_v2(
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,
        const unsigned short* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    const int R = MEMRA_GEMV_V2_ROWS;
    int t = blockIdx.y;
    if (t >= m) return;
    const float* xrow = x + (size_t)t * in_f;
    int b = blockIdx.x;
    int nb;
    nb = (out0 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v2(W0, xrow, y0 + (size_t)t * out0, in_f, out0, b);
        return;
    }
    b -= nb;
    nb = (out1 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v2(W1, xrow, y1 + (size_t)t * out1, in_f, out1, b);
        return;
    }
    b -= nb;
    nb = (out2 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v2(W2, xrow, y2 + (size_t)t * out2, in_f, out2, b);
        return;
    }
    b -= nb;
    nb = (out3 + R - 1) / R;
    if (b < nb) {
        for (int p = 0; p < R; p++)
            f32_mmvq_row1(W3, xrow, y3 + (size_t)t * out3, in_f, out3, b * R + p);
        return;
    }
    b -= nb;
    nb = (out4 + R - 1) / R;
    if (b < nb) {
        for (int p = 0; p < R; p++)
            f32_mmvq_row1(W4, xrow, y4 + (size_t)t * out4, in_f, out4, b * R + p);
        return;
    }
    // BUG FIX (box receipt 2026-09-02, gate-gemv-bench): this `b -= nb` was dropped in the
    // first cut. The shipped kernel has it; without it the last range's block index still
    // carries range 4's offset, every `b * R + p` lands past `out5`, `f32_mmvq_row1` returns
    // for all of them, and y5 is NEVER WRITTEN. On the bench dims that is exactly the observed
    // failure: MISMATCH n=64 (= out5) max_abs_diff=2.442e2 (= max |y5|), with the other five
    // ranges bit-identical. Range 5 is the ONLY range with no `if (b < nb)` guard behind it, so
    // nothing else caught the missing decrement.
    b -= nb;
    for (int p = 0; p < R; p++)
        f32_mmvq_row1(W5, xrow, y5 + (size_t)t * out5, in_f, out5, b * R + p);
}

// ---- v2 twins of the plain-decode glm5_next NVFP4 W4A16 expert pair -------------------------
//
// The 36-byte NVFP4 block layout forbids wider loads (a row base plus `sblk*36 + 4 + s*8` is
// only 4 B aligned, so `uint2`/`uint4` are illegal here and the four `get_int_b4` reads per
// group are not a missed vectorization). The lever that IS available is depth: the shipped
// g loop holds ONE group's loads per row, and at in_f=4096 a lane runs only nsb/32 = 4
// iterations, so it spends most of its life on a dependent dp4a chain with a single 18 B group
// outstanding. These twins unroll the g walk by two, issuing BOTH groups' weight, scale and
// activation loads before either dp4a chain runs, and pack 8 warps per block so the SM's
// resident-block limit never binds before its warp limit.
//
// BIT-IDENTITY: `accg += dot(g); accu += dot(g); accg += dot(g+32); accu += dot(g+32)` is the
// shipped per-accumulator order, unrolled; `expert_dot_g` is called with the same arguments and
// is a pure function of them. The epilogue and the slot-ordered __fmaf_rn down chain are
// verbatim. Only which warp runs a given (o, j) and when its loads issue changes.
extern "C" __global__ __launch_bounds__(256) void moe_gate_up_preclamp8_q8_v2(
        wptr8_t gp, wptr8_t up, const signed char* __restrict__ aq, const float* __restrict__ ad,
        f32x8_t gs, f32x8_t us, float limit,
        float* __restrict__ act, int in_f, int n_ff, int qt_g, int qt_u, long rb_g, long rb_u) {
    int o = blockIdx.x * (int)blockDim.y + (int)threadIdx.y;   // expert-FFN row (packed)
    int j = blockIdx.y;                                        // routed slot
    if (o >= n_ff) return;
    int lane = threadIdx.x;
    int nsb = in_f >> 5;
    const unsigned char* grow = gp.p[j] + (long)o * rb_g;
    const unsigned char* urow = up.p[j] + (long)o * rb_u;
    float accg = 0.0f, accu = 0.0f;
    int g = lane;
    for (; g + 32 < nsb; g += 64) {
        const signed char* a0 = aq + (size_t)g * 32;
        const signed char* a1 = aq + (size_t)(g + 32) * 32;
        float d80 = ad[g];
        float d81 = ad[g + 32];
        float g0 = expert_dot_g(qt_g, grow, g, a0, d80, nsb);
        float u0 = expert_dot_g(qt_u, urow, g, a0, d80, nsb);
        float g1 = expert_dot_g(qt_g, grow, g + 32, a1, d81, nsb);
        float u1 = expert_dot_g(qt_u, urow, g + 32, a1, d81, nsb);
        accg += g0;
        accu += u0;
        accg += g1;
        accu += u1;
    }
    for (; g < nsb; g += 32) {
        const signed char* aqb = aq + (size_t)g * 32;
        float d8 = ad[g];
        accg += expert_dot_g(qt_g, grow, g, aqb, d8, nsb);
        accu += expert_dot_g(qt_u, urow, g, aqb, d8, nsb);
    }
    accg = warp_reduce_sum(accg);
    accu = warp_reduce_sum(accu);
    if (lane == 0) {
        float u = fmaxf(fminf(accu * us.v[j], limit), -limit);
        float x = fminf(accg * gs.v[j], limit);
        act[(size_t)j * n_ff + o] = (x / (1.0f + expf(-x))) * u;
    }
}

// v2 down projection. The shipped kernel walks its 8 experts SEQUENTIALLY inside ONE warp, so
// the whole launch is out_f warps wide: 4096 warps for the GLM-5.3 down shape, 0.43 of a
// full-occupancy B200 wave, and every expert's DRAM round trip is serialized behind the last.
// Here ONE BLOCK owns one output row and warp j owns expert slot j, so the launch is
// out_f * n_used warps wide (8x) and the eight experts' bytes are in flight together.
//
// STILL BIT-IDENTICAL: each expert's partial is the shipped per-expert g-strided chain plus the
// same warp_reduce_sum, and the final `chain = __fmaf_rn(w.v[k], part[k], chain)` runs in the
// SAME ascending slot order on ONE thread. Parallelizing the experts moved no bits because the
// slot chain was never the parallel part.
extern "C" __global__ __launch_bounds__(256) void moe_down8_fma_q8_v2(
        wptr8_t dp, f32x8_t w, const signed char* __restrict__ aq2, const float* __restrict__ ad2,
        float* __restrict__ dst, int in_f, int out_f, int n_used, int qt, long rb) {
    int o = blockIdx.x;
    if (o >= out_f) return;
    __shared__ float parts[8];
    int j = (int)threadIdx.y;
    int lane = (int)threadIdx.x;
    int nsb = in_f >> 5;
    if (j < n_used) {
        const unsigned char* wrow = dp.p[j] + (long)o * rb;
        const signed char* arow = aq2 + (size_t)j * in_f;
        const float* adrow = ad2 + (size_t)j * nsb;
        float acc = 0.0f;
        int g = lane;
        for (; g + 32 < nsb; g += 64) {
            float p0 = expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
            float p1 = expert_dot_g(qt, wrow, g + 32, arow + (size_t)(g + 32) * 32, adrow[g + 32], nsb);
            acc += p0;
            acc += p1;
        }
        for (; g < nsb; g += 32)
            acc += expert_dot_g(qt, wrow, g, arow + (size_t)g * 32, adrow[g], nsb);
        acc = warp_reduce_sum(acc);
        if (lane == 0) parts[j] = acc;
    }
    __syncthreads();
    if (j == 0 && lane == 0) {
        float chain = 0.0f;
        for (int k = 0; k < n_used; k++) chain = __fmaf_rn(w.v[k], parts[k], chain);
        dst[o] = chain;
    }
}


// =====================================================================================
// v3: cp.async-STAGED bf16 GEMV (MEMRA_B200_GEMV_V2=2, lane/b200-gemv-hbm-20260902)
// =====================================================================================
//
// WHY A THIRD FORM. Box receipt 2026-09-02 (B200 dev 0, b200_matvec_bench 5 3): v2 took the
// bf16 rows from 33.4/30.65 us to 23.7/24.3 us (1.41x / 1.26x) and the fused kda6 group from
// 100.8 to 65.0 us (1.55x, 3.18 TB/s). 3.18 TB/s is 40% of the 8 TB/s wall — real, and short of
// the 60% target. The v2 in-flight budget is REGISTER-BOUND and that is what caps it:
//
//   v2 kda6  : 96 registers -> 65536/96 = 682 threads/SM, i.e. 5 CTAs of 128 -> 640 threads,
//              each holding 10 x 16 B = 160 B  =>  ~102 KB per SM outstanding.
//
// Of the three levers this lane's open items named, only one changes that number:
//
//   persistent CTA        : same loop, fewer launches. In-flight per SM UNCHANGED (~102 KB).
//                           It attacks launch count, which is the graph lane's axis, not this one.
//   2-CTA cluster + DSMEM : shares the activation between two CTAs. The activation is already
//                           only 1/4 of v2's traffic, so the ceiling on the win is ~12% of load
//                           instructions and in-flight bytes are UNCHANGED.
//   cp.async / TMA staging: moves the in-flight bytes OUT of registers into shared memory, so
//                           the budget stops being register-bound entirely.
//
//   v3       : 2 stages x 8 rows x (blockDim*8) elements x 2 B = 32 KB in flight per CTA, plus
//              4 KB of reduction window = 36 KB of smem per CTA. 228 KB/SM / 36 KB = 6 CTAs
//              =>  ~192 KB per SM outstanding, 1.9x v2, at a register count low enough that
//              registers stop binding at all. 36 KB also stays under the 48 KB default dynamic
//              smem cap, so no cudaFuncSetAttribute opt-in is needed.
//
// cp.async rather than `cp.async.bulk`/TMA: same arithmetic (both land bytes in smem without
// occupying registers), a fraction of the machinery, and no mbarrier protocol to get wrong on a
// part this lane cannot run. TMA remains the follow-up if v3's shape is right and its issue rate
// is the next wall.
//
// THE CHUNK SIZE IS NOT A TUNING KNOB. It is pinned to `blockDim.x * 8`, which is exactly the
// shipped kernel's per-thread stride, so chunk `c` hands thread `tid` exactly one index
// `i = c*kch + tid*8` and walking chunks ascending reproduces the shipped `i` sequence
// EXACTLY. Any other chunk size would reorder a row's accumulation and cost the bit-identity.
//
// NO BARRIERS ARE ADDED, and that is load-bearing: thread `tid` issues the copy for
// `dst + p*kch + tid*8` and later reads THAT SAME ADDRESS, for every one of the R rows. Each
// thread therefore consumes only bytes it copied itself, `__pipeline_commit`/`wait_prior` are
// per-thread, and no `__syncthreads` is required between stages -- so v3 keeps v2's
// one-barrier-chain-per-block property (9.4 KB of DRAM traffic per barrier) while doubling the
// bytes outstanding. Reissuing a slot overwrites only the issuing thread's own lane, after that
// thread has read it in program order.
#define MEMRA_GEMV_V3_STAGES 2

// Elements of `unsigned short` in one stage slot, and the float offset at which the reduction
// window starts. Both are recomputed identically by the host launcher (`gemv_v3_smem_bytes`).
#define MEMRA_GEMV_V3_SLOT(R, kch) ((R) * (kch))
#define MEMRA_GEMV_V3_REDOFF(R, kch) ((MEMRA_GEMV_V3_STAGES * (R) * (kch)) / 2)

template <int R>
__device__ __forceinline__ void gemv_v3_stage_issue(
        const unsigned short* const* wr, unsigned short* stage, int in_f, int kch, int c) {
    const int k0 = c * kch;
    const int len = min(kch, in_f - k0);
    unsigned short* dst = stage + (c % MEMRA_GEMV_V3_STAGES) * MEMRA_GEMV_V3_SLOT(R, kch);
    const int j = (int)threadIdx.x * 8;
    if (j < len) {
#pragma unroll
        for (int p = 0; p < R; p++)
            __pipeline_memcpy_async(dst + p * kch + j, wr[p] + k0 + j, 16);
    }
}

template <int R>
__device__ __forceinline__ void gemv_v3_walk_bf16(
        const unsigned short* __restrict__ w, int in_f, int row0, int nrow,
        const float* __restrict__ x, float (&acc)[R], unsigned short* stage) {
    const int kch = (int)blockDim.x * 8;      // == the shipped per-thread stride, pinned
    const int nch = (in_f + kch - 1) / kch;
    const unsigned short* wr[R];
#pragma unroll
    for (int p = 0; p < R; p++) wr[p] = w + (size_t)(row0 + min(p, nrow - 1)) * (size_t)in_f;

    for (int s = 0; s < MEMRA_GEMV_V3_STAGES && s < nch; s++) {
        gemv_v3_stage_issue<R>(wr, stage, in_f, kch, s);
        __pipeline_commit();
    }
    for (int c = 0; c < nch; c++) {
        // nch == 1 (in_f <= blockDim.x * 8, i.e. <= 1024 at the default block): the prologue
        // committed exactly one group, so "all but the newest STAGES-1" waits for nothing and
        // the read below would race this thread's own cp.async. Wait for everything instead.
        __pipeline_wait_prior(nch == 1 ? 0 : MEMRA_GEMV_V3_STAGES - 1);
        const int i = c * kch + (int)threadIdx.x * 8;
        if (i < in_f) {
            const unsigned short* sb = stage
                    + (c % MEMRA_GEMV_V3_STAGES) * MEMRA_GEMV_V3_SLOT(R, kch)
                    + (int)threadIdx.x * 8;
            const float4 x0 = __ldg(reinterpret_cast<const float4*>(x + i));
            const float4 x1 = __ldg(reinterpret_cast<const float4*>(x + i + 4));
#pragma unroll
            for (int p = 0; p < R; p++) {
                const unsigned short* wp = sb + p * kch;
                acc[p] += __uint_as_float((unsigned)wp[0] << 16) * x0.x;
                acc[p] += __uint_as_float((unsigned)wp[1] << 16) * x0.y;
                acc[p] += __uint_as_float((unsigned)wp[2] << 16) * x0.z;
                acc[p] += __uint_as_float((unsigned)wp[3] << 16) * x0.w;
                acc[p] += __uint_as_float((unsigned)wp[4] << 16) * x1.x;
                acc[p] += __uint_as_float((unsigned)wp[5] << 16) * x1.y;
                acc[p] += __uint_as_float((unsigned)wp[6] << 16) * x1.z;
                acc[p] += __uint_as_float((unsigned)wp[7] << 16) * x1.w;
            }
        }
        if (c + MEMRA_GEMV_V3_STAGES < nch)
            gemv_v3_stage_issue<R>(wr, stage, in_f, kch, c + MEMRA_GEMV_V3_STAGES);
        __pipeline_commit();
    }
}

// v3 BF16 GEMV. Same grid/block/reduction as `matvec_bf16_v2`, same per-row accumulation order
// as `matvec_bf16_f32acc_x4_rows` -> BIT-IDENTICAL to both. Dynamic smem =
// STAGES*R*(blockDim.x*8)*2 + R*blockDim.x*4 bytes.
extern "C" __global__ __launch_bounds__(256) void matvec_bf16_v3(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f) {
    const int R = MEMRA_GEMV_V2_ROWS;
    const int trow = blockIdx.y;
    x += (size_t)trow * in_f;
    y += (size_t)trow * out_f;
    const int row0 = blockIdx.x * R;
    const int nrow = min(R, out_f - row0);
    if (nrow <= 0) return;
    const int kch = (int)blockDim.x * 8;
    unsigned short* stage = reinterpret_cast<unsigned short*>(gemv_v2_red);
    float* red = gemv_v2_red + MEMRA_GEMV_V3_REDOFF(R, kch);
    float acc[R];
#pragma unroll
    for (int p = 0; p < R; p++) acc[p] = 0.0f;
    gemv_v3_walk_bf16<R>(w, in_f, row0, nrow, x, acc, stage);
    gemv_v2_reduce_bf16<R>(red, acc, y, row0, nrow);
}

__device__ __forceinline__ void kda6_bf16_rows8_v3(
        const unsigned short* __restrict__ w, const float* __restrict__ x,
        float* __restrict__ y, int in_f, int out_f, int blk) {
    const int R = MEMRA_GEMV_V2_ROWS;
    const int row0 = blk * R;
    const int nrow = min(R, out_f - row0);
    if (nrow <= 0) return;
    const int kch = (int)blockDim.x * 8;
    unsigned short* stage = reinterpret_cast<unsigned short*>(gemv_v2_red);
    float* red = gemv_v2_red + MEMRA_GEMV_V3_REDOFF(R, kch);
    float acc[R];
#pragma unroll
    for (int p = 0; p < R; p++) acc[p] = 0.0f;
    gemv_v3_walk_bf16<R>(w, in_f, row0, nrow, x, acc, stage);
    gemv_v2_reduce_bf16<R>(red, acc, y, row0, nrow);
}

// v3 twin of the fused KDA six-projection kernel. Range partition, block order and the three
// f32 ranges are `qmatvec_kda6_bf16f32_v2`'s verbatim (including the `b -= nb` before range 5
// whose absence was the first cut's y5 bug); only the three BF16 ranges change walk.
extern "C" __global__ __launch_bounds__(256) void qmatvec_kda6_bf16f32_v3(
        const unsigned short* __restrict__ W0, const unsigned short* __restrict__ W1,
        const unsigned short* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    const int R = MEMRA_GEMV_V2_ROWS;
    int t = blockIdx.y;
    if (t >= m) return;
    const float* xrow = x + (size_t)t * in_f;
    int b = blockIdx.x;
    int nb;
    nb = (out0 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v3(W0, xrow, y0 + (size_t)t * out0, in_f, out0, b);
        return;
    }
    b -= nb;
    nb = (out1 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v3(W1, xrow, y1 + (size_t)t * out1, in_f, out1, b);
        return;
    }
    b -= nb;
    nb = (out2 + R - 1) / R;
    if (b < nb) {
        kda6_bf16_rows8_v3(W2, xrow, y2 + (size_t)t * out2, in_f, out2, b);
        return;
    }
    b -= nb;
    nb = (out3 + R - 1) / R;
    if (b < nb) {
        for (int p = 0; p < R; p++)
            f32_mmvq_row1(W3, xrow, y3 + (size_t)t * out3, in_f, out3, b * R + p);
        return;
    }
    b -= nb;
    nb = (out4 + R - 1) / R;
    if (b < nb) {
        for (int p = 0; p < R; p++)
            f32_mmvq_row1(W4, xrow, y4 + (size_t)t * out4, in_f, out4, b * R + p);
        return;
    }
    b -= nb;
    for (int p = 0; p < R; p++)
        f32_mmvq_row1(W5, xrow, y5 + (size_t)t * out5, in_f, out5, b * R + p);
}

// =====================================================================================
// W8-POSTURE q8_0 DECODE MATVECS (MEMRA_B200_GEMV_V2=1, lane/b200-gemv-hbm-20260902 round 3)
// =====================================================================================
//
// WHY THESE AND NOT THE BF16 ONES. Serving A/B pair 1 on the pair (2026-09-02) moved NOTHING:
// 49.2 -> 49.3 tok/s code, 48.8 -> 48.7 prose, and the boot log printed no gemv engagement line
// at all. The door had nothing to dispatch, because the serving base now runs the q8_0 trunk
// mirror (MEMRA_GLM5_W8, PR #86, +10% plain): `matvec_bf16_rows_into` reroutes every bf16-
// resident KDA/MLA projection through `matvec_bf16_via_q8_mirror(_t)` BEFORE the bf16 arms are
// reached, so `matvec_bf16_f32acc_x4_rows` and `qmatvec_kda6_bf16f32` are simply not on the
// t=1 decode path in the posture we serve. The bf16 v2/v3 arms stay for the non-W8 posture.
//
// THE KERNELS THAT ACTUALLY SERVE THE W8 TRUNK (read off lib.rs, not guessed):
//   t == 1        `matvec_bf16_via_q8_mirror` -> `qmatvec_mmvq_into(.., QT_Q8_0, rp=true)`
//                 -> `qmatvec_q8_0_mmvq_rp` (4 warps/block, 1 row/warp) for the glm5 shapes;
//                    `qmatvec_q8_0_mmvq_rp_g2` only when out_f/4 < 4*SMs (out_f < 2368 on
//                    B200, so not the 4096/8192 KDA rows); `_rpca`/`_mr2_rp` are off by
//                    measurement (mr2 lost on H100: halving the grid costs more than 2-row ILP).
//   t in 2..=32   `matvec_bf16_via_q8_mirror_t` -> `qmatvec_q8_0_rows_tw` (t<=8, the verify
//                 width) / `_tw32` (t<=32) under MEMRA_Q8T_WONCE, else `qmatvec_q8_0_rows_t`.
//
// THE IN-FLIGHT PROBLEM, same shape as the bf16 one and arrived at the same way. The q8_0 rp
// mirror is 32 B of quants + a 2 B scale per 32-element block, split into a quant plane and a
// scale plane, so unlike NVFP4's 36 B blocks every weight fetch is ALREADY an aligned 16 B
// `__ldcs` — there is no load-width lever here. What there is:
//
//   * `qmatvec_q8_0_mmvq_rp` reads 36 B of ACTIVATION (32 B `aq` + a 4 B `ad` scale) for every
//     34 B of WEIGHT, per lane, per block-iteration. The activation is identical for every one
//     of the 8192 rows in the launch, so those bytes are L1/L2 hits — but they are 50% of the
//     kernel's load INSTRUCTIONS and they sit in the dependency chain ahead of every dp4a. This
//     is the same 2:1 activation:weight ratio that cost `matvec_bf16_f32acc_x4_rows` its
//     bandwidth, in a different dtype.
//   * at in_f=4096 a lane walks nblk/32 = 4 block-iterations, un-unrolled, so it holds ONE
//     block's loads at a time.
//
// The v2 twins stage the q8_1 activation into shared memory ONCE PER CTA (in_f + nblk*4 bytes,
// 4.6 KB at in_f=4096 — one barrier, then every warp reads it from smem), pack 8 warps per
// block instead of 4, and unroll the block walk by two so BOTH iterations' weight and scale
// loads issue before either dp4a chain runs.
//
// BIT-IDENTICAL per output row: the per-row program is untouched — same warp-per-row mapping,
// same `blk = lane, lane+32, ...` walk, same eight `dp4a` in the same order, same
// `acc += dw * ad[blk] * (float)sumi` association, same `warp_reduce_sum`, same lane-0 store.
// Staging copies bytes without changing them; unrolling reorders LOAD ISSUE, not accumulation;
// packing only changes which warp owns a row.

// Warps per block for the v2 q8_0 twins (the shipped kernels use MEMRA_MMVQ_ROWS = 4).
#define MEMRA_Q8_V2_ROWS 8

// One CTA-wide copy of this token row's q8_1 activation into shared memory. `saq` must be 16 B
// aligned (it is: dynamic smem base) and `in_f % 32 == 0`, so `sad` lands 16 B aligned too.
// Ends in a __syncthreads: the ONLY barrier these kernels have.
__device__ __forceinline__ void q8_0_stage_act(
        const signed char* __restrict__ arow, const float* __restrict__ adrow,
        signed char* saq, float* sad, int in_f, int nblk) {
    const int tid = (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
    const int nthr = (int)blockDim.x * (int)blockDim.y;
    int4* d4 = reinterpret_cast<int4*>(saq);
    const int4* s4 = reinterpret_cast<const int4*>(arow);
    for (int i = tid; i < in_f / 16; i += nthr) d4[i] = __ldg(s4 + i);
    for (int i = tid; i < nblk; i += nthr) sad[i] = __ldg(adrow + i);
    __syncthreads();
}

// The shipped `qmatvec_q8_0_mmvq_rp` per-row body, reading the staged activation and walking
// `blk` two iterations at a time. `y` is already offset to this token row.
__device__ __forceinline__ void q8_0_mmvq_row1_rp_v2(
        const unsigned char* __restrict__ W, int out_f, int o, int nblk,
        const signed char* saq, const float* sad, float* __restrict__ y) {
    if (o >= out_f) return;
    const int lane = (int)threadIdx.x;
    const unsigned char* wq;
    const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    float acc = 0.0f;
    int blk = lane;
    for (; blk + 32 < nblk; blk += 64) {
        const int nb2 = blk + 32;
        // Both iterations' weight and scale loads issue before either dp4a chain.
        int4 wa0 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 wa1 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int4 wb0 = __ldcs((const int4*)(wq + (size_t)nb2 * 32));
        int4 wb1 = __ldcs((const int4*)(wq + (size_t)nb2 * 32 + 16));
        float dwa = half_to_float(wd[blk]);
        float dwb = half_to_float(wd[nb2]);
        const int4* a4a = (const int4*)(saq + blk * 32);
        const int4* a4b = (const int4*)(saq + nb2 * 32);
        int4 aa0 = a4a[0], aa1 = a4a[1];
        int4 ab0 = a4b[0], ab1 = a4b[1];
        int wia[8] = { wa0.x, wa0.y, wa0.z, wa0.w, wa1.x, wa1.y, wa1.z, wa1.w };
        int aqa[8] = { aa0.x, aa0.y, aa0.z, aa0.w, aa1.x, aa1.y, aa1.z, aa1.w };
        int wib[8] = { wb0.x, wb0.y, wb0.z, wb0.w, wb1.x, wb1.y, wb1.z, wb1.w };
        int aqb[8] = { ab0.x, ab0.y, ab0.z, ab0.w, ab1.x, ab1.y, ab1.z, ab1.w };
        int sa = 0, sb = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sa = dp4a(wia[k], aqa[k], sa);
        #pragma unroll
        for (int k = 0; k < 8; k++) sb = dp4a(wib[k], aqb[k], sb);
        acc += dwa * sad[blk] * (float)sa;
        acc += dwb * sad[nb2] * (float)sb;
    }
    for (; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* a4 = (const int4*)(saq + blk * 32);
        int4 a01 = a4[0], a23 = a4[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * sad[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc;
}

// ---- ILP twin of q8_0_mmvq_row1_rp_v2 (lane/glm5-q8-row-ilp-20260904, door MEMRA_Q8_ROW_ILP).
// WHY. The two kernels this body serves are 1.8 ms of a plain glm5_next t=1 token on the
// 2x B200 pair (door-ON census 2026-09-04: qmatvec_kda6_q8f32_rp_v2 35/token x 36.2 us =
// 1.27 ms at ~49% of the part's 8 TB/s for its ~143 MB of mirror bytes, qmatvec_q8_0_mmvq_rp_v2
// 47/token x 11.5 us). Root ncu on the rig at the served geometry (darklanes
// research/glm5-b200-20260902/ncu-rig, kda6.csv) reads the row body at 62-72% occupancy,
// DRAM 46-49% of peak, long-scoreboard stalls 71-76% of warp-active cycles, issue-active
// 7-12%: latency-bound on global loads. The shipped body walks two blocks per round and its
// four loads wait on the previous round's dependent chain; this twin issues FOUR blocks' loads
// per round (eight int4 + four scales per lane) before any dp4a, then the shipped two-deep
// round, then the shipped tail. EXACTNESS by construction: the per-lane accumulation order is
// the shipped one (acc += term(blk), term(blk+32), term(blk+64), term(blk+96), ... in the same
// `dw * sad[blk] * (float)sumi` statement form), the warp tree is verbatim, and the loads are
// the exact bytes the shipped body reads. Gate: b200_matvec_bench families 7 and 9 print the
// shipped-vs-ilp bitwise compare; the box greedy tape holds it at model scale.
__device__ __forceinline__ void q8_0_mmvq_row1_rp_v2_ilp(
        const unsigned char* __restrict__ W, int out_f, int o, int nblk,
        const signed char* saq, const float* sad, float* __restrict__ y) {
    if (o >= out_f) return;
    const int lane = (int)threadIdx.x;
    const unsigned char* wq;
    const unsigned short* wd;
    q8_0_rp_planes(W, out_f, o, nblk, &wq, &wd);
    float acc = 0.0f;
    int blk = lane;
    for (; blk + 96 < nblk; blk += 128) {
        const int b1 = blk + 32, b2 = blk + 64, b3 = blk + 96;
        // Four blocks' weight and scale loads issue before any dp4a chain.
        int4 w00 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int4 w10 = __ldcs((const int4*)(wq + (size_t)b1 * 32));
        int4 w11 = __ldcs((const int4*)(wq + (size_t)b1 * 32 + 16));
        int4 w20 = __ldcs((const int4*)(wq + (size_t)b2 * 32));
        int4 w21 = __ldcs((const int4*)(wq + (size_t)b2 * 32 + 16));
        int4 w30 = __ldcs((const int4*)(wq + (size_t)b3 * 32));
        int4 w31 = __ldcs((const int4*)(wq + (size_t)b3 * 32 + 16));
        float dw0 = half_to_float(wd[blk]);
        float dw1 = half_to_float(wd[b1]);
        float dw2 = half_to_float(wd[b2]);
        float dw3 = half_to_float(wd[b3]);
        const int4* a4_0 = (const int4*)(saq + blk * 32);
        const int4* a4_1 = (const int4*)(saq + b1 * 32);
        const int4* a4_2 = (const int4*)(saq + b2 * 32);
        const int4* a4_3 = (const int4*)(saq + b3 * 32);
        int4 a00 = a4_0[0], a01 = a4_0[1];
        int4 a10 = a4_1[0], a11 = a4_1[1];
        int4 a20 = a4_2[0], a21 = a4_2[1];
        int4 a30 = a4_3[0], a31 = a4_3[1];
        int wi0[8] = { w00.x, w00.y, w00.z, w00.w, w01.x, w01.y, w01.z, w01.w };
        int wi1[8] = { w10.x, w10.y, w10.z, w10.w, w11.x, w11.y, w11.z, w11.w };
        int wi2[8] = { w20.x, w20.y, w20.z, w20.w, w21.x, w21.y, w21.z, w21.w };
        int wi3[8] = { w30.x, w30.y, w30.z, w30.w, w31.x, w31.y, w31.z, w31.w };
        int aq0[8] = { a00.x, a00.y, a00.z, a00.w, a01.x, a01.y, a01.z, a01.w };
        int aq1[8] = { a10.x, a10.y, a10.z, a10.w, a11.x, a11.y, a11.z, a11.w };
        int aq2[8] = { a20.x, a20.y, a20.z, a20.w, a21.x, a21.y, a21.z, a21.w };
        int aq3[8] = { a30.x, a30.y, a30.z, a30.w, a31.x, a31.y, a31.z, a31.w };
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) s0 = dp4a(wi0[k], aq0[k], s0);
        #pragma unroll
        for (int k = 0; k < 8; k++) s1 = dp4a(wi1[k], aq1[k], s1);
        #pragma unroll
        for (int k = 0; k < 8; k++) s2 = dp4a(wi2[k], aq2[k], s2);
        #pragma unroll
        for (int k = 0; k < 8; k++) s3 = dp4a(wi3[k], aq3[k], s3);
        acc += dw0 * sad[blk] * (float)s0;
        acc += dw1 * sad[b1] * (float)s1;
        acc += dw2 * sad[b2] * (float)s2;
        acc += dw3 * sad[b3] * (float)s3;
    }
    for (; blk + 32 < nblk; blk += 64) {
        const int nb2 = blk + 32;
        int4 wa0 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 wa1 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int4 wb0 = __ldcs((const int4*)(wq + (size_t)nb2 * 32));
        int4 wb1 = __ldcs((const int4*)(wq + (size_t)nb2 * 32 + 16));
        float dwa = half_to_float(wd[blk]);
        float dwb = half_to_float(wd[nb2]);
        const int4* a4a = (const int4*)(saq + blk * 32);
        const int4* a4b = (const int4*)(saq + nb2 * 32);
        int4 aa0 = a4a[0], aa1 = a4a[1];
        int4 ab0 = a4b[0], ab1 = a4b[1];
        int wia[8] = { wa0.x, wa0.y, wa0.z, wa0.w, wa1.x, wa1.y, wa1.z, wa1.w };
        int aqa[8] = { aa0.x, aa0.y, aa0.z, aa0.w, aa1.x, aa1.y, aa1.z, aa1.w };
        int wib[8] = { wb0.x, wb0.y, wb0.z, wb0.w, wb1.x, wb1.y, wb1.z, wb1.w };
        int aqb[8] = { ab0.x, ab0.y, ab0.z, ab0.w, ab1.x, ab1.y, ab1.z, ab1.w };
        int sa = 0, sb = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sa = dp4a(wia[k], aqa[k], sa);
        #pragma unroll
        for (int k = 0; k < 8; k++) sb = dp4a(wib[k], aqb[k], sb);
        acc += dwa * sad[blk] * (float)sa;
        acc += dwb * sad[nb2] * (float)sb;
    }
    for (; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        const int4* a4 = (const int4*)(saq + blk * 32);
        int4 a01 = a4[0], a23 = a4[1];
        int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
        acc += dw * sad[blk] * (float)sumi;
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[o] = acc;
}
template <bool ILP>
__device__ __forceinline__ void q8_0_row_v2_dispatch(
        const unsigned char* __restrict__ W, int out_f, int o, int nblk,
        const signed char* saq, const float* sad, float* __restrict__ y) {
    if (ILP) q8_0_mmvq_row1_rp_v2_ilp(W, out_f, o, nblk, saq, sad, y);
    else q8_0_mmvq_row1_rp_v2(W, out_f, o, nblk, saq, sad, y);
}

// v2 twin of `qmatvec_q8_0_mmvq_rp` (the t=1 W8 trunk kernel). grid = (out_f/8, m, 1),
// block = (32, 8, 1), dynamic smem = in_f + nblk*4 bytes. BIT-IDENTICAL per output.
extern "C" __global__ __launch_bounds__(256) void qmatvec_q8_0_mmvq_rp_v2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    const int t = blockIdx.y;
    if (t >= m) return;
    const int nblk = in_f / 32;
    extern __shared__ float gemv_v2_red[];
    signed char* saq = reinterpret_cast<signed char*>(gemv_v2_red);
    float* sad = reinterpret_cast<float*>(saq + in_f);
    q8_0_stage_act(aq + (size_t)t * in_f, ad + (size_t)t * nblk, saq, sad, in_f, nblk);
    const int o = blockIdx.x * MEMRA_Q8_V2_ROWS + (int)threadIdx.y;
    q8_0_mmvq_row1_rp_v2(W, out_f, o, nblk, saq, sad, y + (size_t)t * out_f);
}
// The MEMRA_Q8_ROW_ILP twin of the kernel above: same staging, same grid, the four-deep row body.
extern "C" __global__ __launch_bounds__(256) void qmatvec_q8_0_mmvq_rp_v2_ilp(
        const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
        const float* __restrict__ ad, float* __restrict__ y,
        int in_f, int out_f, int m, long row_bytes) {
    (void)row_bytes;
    const int t = blockIdx.y;
    if (t >= m) return;
    const int nblk = in_f / 32;
    extern __shared__ float gemv_v2_red[];
    signed char* saq = reinterpret_cast<signed char*>(gemv_v2_red);
    float* sad = reinterpret_cast<float*>(saq + in_f);
    q8_0_stage_act(aq + (size_t)t * in_f, ad + (size_t)t * nblk, saq, sad, in_f, nblk);
    const int o = blockIdx.x * MEMRA_Q8_V2_ROWS + (int)threadIdx.y;
    q8_0_mmvq_row1_rp_v2_ilp(W, out_f, o, nblk, saq, sad, y + (size_t)t * out_f);
}

// v2 twin of `qmatvec_q8_0_rows_tw` (the VERIFY-width t<=MEMRA_Q8T_TMAX W8 kernel). The
// weight-once t-column structure is the shipped kernel's verbatim — one weight fetch feeds all
// t columns — so the activation is NOT staged here (t*in_f would be 32 KB at t=8); the levers
// are the 8-warp packing and the block walk unrolled by two. BIT-IDENTICAL per (row, column):
// same `acc[c] += dw * ad[c*nblk + blk] * (float)sumi` in the same blk order.
extern "C" __global__ __launch_bounds__(256) void qmatvec_q8_0_rows_tw_v2(
        const unsigned char* __restrict__ W,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        float* __restrict__ y, int in_f, int out_f, int t) {
    const int row = blockIdx.x * MEMRA_Q8_V2_ROWS + (int)threadIdx.y;
    if (row >= out_f) return;
    const int lane = (int)threadIdx.x;
    const int nblk = in_f / 32;
    const unsigned char* wq;
    const unsigned short* wd;
    q8_0_rp_planes(W, out_f, row, nblk, &wq, &wd);
    float acc[MEMRA_Q8T_TMAX];
    #pragma unroll
    for (int c = 0; c < MEMRA_Q8T_TMAX; c++) acc[c] = 0.0f;
    int blk = lane;
    for (; blk + 32 < nblk; blk += 64) {
        const int nb2 = blk + 32;
        int4 wa0 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 wa1 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int4 wb0 = __ldcs((const int4*)(wq + (size_t)nb2 * 32));
        int4 wb1 = __ldcs((const int4*)(wq + (size_t)nb2 * 32 + 16));
        float dwa = half_to_float(wd[blk]);
        float dwb = half_to_float(wd[nb2]);
        int wia[8] = { wa0.x, wa0.y, wa0.z, wa0.w, wa1.x, wa1.y, wa1.z, wa1.w };
        int wib[8] = { wb0.x, wb0.y, wb0.z, wb0.w, wb1.x, wb1.y, wb1.z, wb1.w };
        for (int c = 0; c < t; c++) {
            const int4* a4a = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            const int4* a4b = (const int4*)(aq + (size_t)c * in_f + nb2 * 32);
            int4 aa0 = a4a[0], aa1 = a4a[1];
            int4 ab0 = a4b[0], ab1 = a4b[1];
            int aqa[8] = { aa0.x, aa0.y, aa0.z, aa0.w, aa1.x, aa1.y, aa1.z, aa1.w };
            int aqb[8] = { ab0.x, ab0.y, ab0.z, ab0.w, ab1.x, ab1.y, ab1.z, ab1.w };
            int sa = 0, sb = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sa = dp4a(wia[k], aqa[k], sa);
            #pragma unroll
            for (int k = 0; k < 8; k++) sb = dp4a(wib[k], aqb[k], sb);
            acc[c] += dwa * ad[(size_t)c * nblk + blk] * (float)sa;
            acc[c] += dwb * ad[(size_t)c * nblk + nb2] * (float)sb;
        }
    }
    for (; blk < nblk; blk += 32) {
        int4 w01 = __ldcs((const int4*)(wq + (size_t)blk * 32));
        int4 w23 = __ldcs((const int4*)(wq + (size_t)blk * 32 + 16));
        int wi[8] = { w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w };
        float dw = half_to_float(wd[blk]);
        for (int c = 0; c < t; c++) {
            const int4* aq16 = (const int4*)(aq + (size_t)c * in_f + blk * 32);
            int4 a01 = aq16[0], a23 = aq16[1];
            int aq4[8] = { a01.x, a01.y, a01.z, a01.w, a23.x, a23.y, a23.z, a23.w };
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = dp4a(wi[k], aq4[k], sumi);
            acc[c] += dw * ad[(size_t)c * nblk + blk] * (float)sumi;
        }
    }
    for (int c = 0; c < t; c++) {
        float a = warp_reduce_sum(acc[c]);
        if (lane == 0) y[(size_t)c * out_f + row] = a;
    }
}

// FUSED six-projection KDA group for the W8 posture. The W8 path had NO fused twin: the
// existing `qmatvec_kda6_q8f32_mmvq` addresses INTERLEAVED 34 B blocks (a resident plain-layout
// Q8_0 tensor), while `MEMRA_GLM5_W8`'s mirror is the SPLIT-PLANE rp4 form, and
// `MEMRA_KDA_FUSED_PROJ`'s bf16 arm declines outright whenever the W8 door is on. So the six
// projections run as six separate launches today. This is the fusion: same six-unequal-range
// block split as `qmatvec_kda6_bf16f32`, the three mirrored ranges on the rp v2 body, the three
// f32 low-rank/beta ranges on `f32_mmvq_row1`.
//
// NUMERIC CLASSES, unchanged from the sibling fused kernels: the three q8_0 ranges are
// BIT-IDENTICAL to `qmatvec_q8_0_mmvq_rp` per row; the three f32 ranges replace cuBLASLt with
// the same deterministic warp tree the q8 arm of `MEMRA_KDA_FUSED_PROJ` already ships and has
// pinned (a reduction-order class, not a new one).
// The body of the fused six-projection W8 kernel, templated on the row walk so the shipped
// kernel and its MEMRA_Q8_ROW_ILP twin share one program (lane/glm5-q8-row-ilp-20260904).
template <bool ILP>
__device__ __forceinline__ void kda6_q8f32_rp_v2_body(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    const int R = MEMRA_Q8_V2_ROWS;
    const int t = blockIdx.y;
    if (t >= m) return;
    const int nblk = in_f / 32;
    extern __shared__ float gemv_v2_red[];
    signed char* saq = reinterpret_cast<signed char*>(gemv_v2_red);
    float* sad = reinterpret_cast<float*>(saq + in_f);
    // Staged BEFORE the range branch so the barrier inside is block-uniform.
    q8_0_stage_act(aq + (size_t)t * in_f, ad + (size_t)t * nblk, saq, sad, in_f, nblk);
    const float* xrow = x + (size_t)t * in_f;
    int b = blockIdx.x;
    int nb;
    nb = (out0 + R - 1) / R;
    if (b < nb) {
        q8_0_row_v2_dispatch<ILP>(W0, out0, b * R + (int)threadIdx.y, nblk, saq, sad,
                             y0 + (size_t)t * out0);
        return;
    }
    b -= nb;
    nb = (out1 + R - 1) / R;
    if (b < nb) {
        q8_0_row_v2_dispatch<ILP>(W1, out1, b * R + (int)threadIdx.y, nblk, saq, sad,
                             y1 + (size_t)t * out1);
        return;
    }
    b -= nb;
    nb = (out2 + R - 1) / R;
    if (b < nb) {
        q8_0_row_v2_dispatch<ILP>(W2, out2, b * R + (int)threadIdx.y, nblk, saq, sad,
                             y2 + (size_t)t * out2);
        return;
    }
    b -= nb;
    nb = (out3 + R - 1) / R;
    if (b < nb) {
        f32_mmvq_row1(W3, xrow, y3 + (size_t)t * out3, in_f, out3, b * R + (int)threadIdx.y);
        return;
    }
    b -= nb;
    nb = (out4 + R - 1) / R;
    if (b < nb) {
        f32_mmvq_row1(W4, xrow, y4 + (size_t)t * out4, in_f, out4, b * R + (int)threadIdx.y);
        return;
    }
    b -= nb;
    f32_mmvq_row1(W5, xrow, y5 + (size_t)t * out5, in_f, out5, b * R + (int)threadIdx.y);
}
extern "C" __global__ __launch_bounds__(256) void qmatvec_kda6_q8f32_rp_v2(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    kda6_q8f32_rp_v2_body<false>(W0, W1, W2, W3, W4, W5, aq, ad, x, y0, y1, y2, y3, y4, y5, in_f, out0, out1, out2, out3, out4, out5, m);
}
extern "C" __global__ __launch_bounds__(256) void qmatvec_kda6_q8f32_rp_v2_ilp(
        const unsigned char* __restrict__ W0, const unsigned char* __restrict__ W1,
        const unsigned char* __restrict__ W2,
        const float* __restrict__ W3, const float* __restrict__ W4,
        const float* __restrict__ W5,
        const signed char* __restrict__ aq, const float* __restrict__ ad,
        const float* __restrict__ x,
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        float* __restrict__ y3, float* __restrict__ y4, float* __restrict__ y5,
        int in_f, int out0, int out1, int out2, int out3, int out4, int out5,
        int m) {
    kda6_q8f32_rp_v2_body<true>(W0, W1, W2, W3, W4, W5, aq, ad, x, y0, y1, y2, y3, y4, y5, in_f, out0, out1, out2, out3, out4, out5, m);
}

// ---- NVFP4 expert slab SLOT-MAJOR repack, device side, once per resident slab at upload
// (lane/moe-expert-rp, memra#147). Per ROW (the QT_NVFP4_V2 form every reader already has: the
// expert dots via expert_dot_nvfp4_v2_g, the grouped prefill via dequant_nvfp4v2 / kq_fetch<V2>):
// block (o, s) at src row o + s*36 ([4B scales][32B quants]) goes to quants at row o + s*32 and
// scales at row o + nsb64*32 + s*4; row stride stays nsb64*36. The source is read as 9 aligned
// u32 words (36B stride from a 256B-aligned slab is 4B-aligned). Same bytes as
// tp.rs nvfp4_matrix_v2_permute.
extern "C" __global__ void nvfp4_expert_split_repack(
        const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
        int n_expert, int rows, int nsb64) {
    size_t nblk = (size_t)n_expert * rows * nsb64;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk) return;
    size_t row = i / nsb64;                                 // global row index (ex*rows + o)
    int s = (int)(i - row * nsb64);
    const unsigned int* sb = (const unsigned int*)(src + i * 36);
    size_t rbase = row * (size_t)nsb64 * 36;
    unsigned int* q = (unsigned int*)(dst + rbase + (size_t)s * 32);
    unsigned int* sc = (unsigned int*)(dst + rbase + (size_t)nsb64 * 32 + (size_t)s * 4);
    sc[0] = sb[0];
    #pragma unroll
    for (int k = 0; k < 8; k++) q[k] = sb[1 + k];
}
