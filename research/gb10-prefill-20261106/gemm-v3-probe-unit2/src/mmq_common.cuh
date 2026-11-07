// mmq_common.cuh — shared MMQ preamble (memra-owned), extracted byte-identical from the four
// int8-MMA MMQ vendor TUs: mmq_q8_0.cu / mmq_q4_0.cu / mmq_q45k.cu / mmq_nvfp4_w4a8.cu
// (lane/kernel-dedup-20260821, SASS-identical extraction — see research/kernel-dedup-20260821/).
//
// SCOPE: increment 1 took only lines byte-identical across ALL FOUR TUs. Increment 2 (relaxed
// rule: identical CODE, comments may differ — the SASS gate stays the arbiter) added QI8_1,
// MMQ_TILE_Y_K, and get_int_b2 (generic comments here; qtype-specific comment facts stay in the
// TUs at the point of use). Increment 3 added struct block_q8_1_mmq (MMQ_BLOCK_Q8_1_MMQ_LOCAL
// opt-out below). The int8 mma tile machinery lives in mmq_mma_i8.cuh. Everything
// qtype-specific (QK*/QI* per format, MMQ_ITER_K, MMQ_MMA_TILE_X_K_*, weight block structs, the
// ggml_cuda_mma tile machinery — q45k/w4a8 variants differ by design — load_tiles/vec_dot/
// write_back/quantize/launchers) stays local to its TU. Header-inlined statics only: still
// no cross-TU linkage, no external deps, no ggml headers.
//
// MMQ_X stays #ifndef-guarded exactly as in the TUs: build.rs tune seams override it via
// -DMMQ_X (and -DMMQ_Y, which remains per-TU — the TUs' MMQ_Y guards differ by design).

#pragma once

#define WARP_SIZE 32
#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))

#define QK8_1 32
#define QI8_1 8                  // QK8_1 / (4 * QR8_1), QR8_1 == 1
#define MATRIX_ROW_PADDING 512

#define MMQ_TILE_NE_K 32
// y-tile stride in ints: MMQ_TILE_NE_K int8-as-int quants + MMQ_TILE_NE_K/QI8_1 scale ints.
// NOTE: mmq_nvfp4_w4a8.cu redefines QI8_1/MMQ_TILE_Y_K locally with token-identical bodies
// (legal identical redefinition) — untouched in increment 2, folds in when that TU is next open.
#define MMQ_TILE_Y_K (MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1)                        // 36

#define MMQ_WARP_SIZE 32
#ifndef MMQ_X
#define MMQ_X         128
#endif

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

// Turing+ granularity (mmq_get_granularity_device): mmq_x>=48 -> 16.
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
    return mmq_x >= 48 ? 16 : 8;
}

// get_int_b2 (common.cuh): read an int from a buffer with only >=2-byte alignment guaranteed
// (block qs arrays that start at +2 after the fp16 scale). Qtype-specific alignment facts stay
// in each adopting TU at the point of use.
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int x32  = x16[2 * i32 + 0] <<  0;
    x32     |= x16[2 * i32 + 1] << 16;
    return x32;
}

// block_q8_1_mmq (mmq.cuh): quantized-activation y block — 4x 4-byte scale slots + 128 int8
// quants (sizeof == one MMQ_TILE_Y_K stride in ints). The scale union is layout-agnostic; which
// member is live is a per-TU fact of the quantize kernel's DS layout template arg (D4: float d,
// no sum term; DS4: half2 (d, partial sum)) and stays commented in each adopting TU. A TU that
// keeps its own definition (mmq_q45k.cu) defines MMQ_BLOCK_Q8_1_MMQ_LOCAL before including this
// header — a second in-TU definition is a C++ redefinition error even when byte-identical.
#ifndef MMQ_BLOCK_Q8_1_MMQ_LOCAL
struct block_q8_1_mmq {
    union {
        float d4[4];
        half2 ds4[4];
        half  d2s6[8];
    };
    int8_t qs[4 * QK8_1];           // 128 values
};
static_assert(sizeof(block_q8_1_mmq) == 4 * MMQ_TILE_Y_K, "block_q8_1_mmq != MMQ_TILE_Y_K ints");
#endif // MMQ_BLOCK_Q8_1_MMQ_LOCAL
