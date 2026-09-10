// Dense wide-prefill TILE WIDTH door (memra #463). Included by dsv4_gpu.cu,
// inherits its -fmad=false build. No kernel body is touched by this file.
//
// What it controls. Above DSV4_TMAX the four dense entry points
// (memra_dsv4_gemv_bf16_m, memra_dsv4_gemv_fp8_m, memra_dsv4_dots_f32_mrow,
// memra_dsv4_dots_f32acc_mrow) decompose a wide transaction into tiles and
// relaunch the registered-row kernel per tile. The tile width sets the launch
// count of the largest kernel family in a prefill: 40.6% of GPU-busy time,
// 423,660 launches at 114.9 per row, more than half of every launch a prefill
// issues (darklanes #596 section 28, nsys on the served program).
//
// The shipped width is 8. At the served chunk width of 64 that is 8 launches
// per dense call and every dense weight is streamed 8 times per chunk. 32 is
// DSV4_TMAX itself, the widest instantiation the dispatch switches already
// carry, so tiling 64 as 2 x 32 is a 4x launch reduction with NO new kernel.
//
// Why the tile width is not an arithmetic choice. In every one of these kernels
// the per-row accumulator part[t] is advanced over the same i0 sequence, in the
// same order, with the same 128-leaf halving tree, whatever M the instantiation
// carries: M only decides how many independent accumulators one block keeps in
// registers, never the order of one accumulator's adds. A row's value therefore
// does not depend on which tile carried it. That is an argument, not a receipt,
// and the compiler still gets to schedule each M instantiation on its own, so
// dsv4_dense_tile_gate compares the two arms bit for bit over logits, live
// cache, DSpark rings and sampled speculative output and refuses on the first
// differing bit.
//
// Door hygiene: default OFF (8, what ships), gate-only selection plus an
// environment arm, both arms counted so a receipt can prove which one ran.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define DSV4_DENSE_TILE_DEFAULT 8

// Only "8" and "32" are legal. Anything else is a typo an operator must see:
// silently resolving an unparsable width to the default is exactly the silence
// memra #454 was filed about. Refusal is carried as 0 and surfaced as 40077 by
// the entry points, which is a returned error code, not an abort.
static int dsv4_dense_tile_environment_default() {
    const char* value = std::getenv("MEMRA_DSV4_DENSE_TILE");
    if (!value) return DSV4_DENSE_TILE_DEFAULT;
    if (std::strcmp(value, "8") == 0) return 8;
    if (std::strcmp(value, "32") == 0) return 32;
    return 0;
}

// Host-thread-local and read before every tiled decomposition. Never read by a
// device kernel, and a captured graph freezes the launches it captured, so a
// later switch cannot mutate a graph. Gate callers drain before switching.
static thread_local int dsv4_dense_tile_width_state = dsv4_dense_tile_environment_default();
// [0] = tiled launches issued at width 8, [1] = at width 32. Engagement receipt.
static thread_local uint64_t dsv4_dense_tile_launches[2] = {};

static inline int dsv4_dense_tile_width() { return dsv4_dense_tile_width_state; }

static inline void dsv4_dense_tile_count(int width, int launches) {
    dsv4_dense_tile_launches[width == 32 ? 1 : 0] += (uint64_t)launches;
}

extern "C" int memra_dsv4_dense_tile_set_for_gate(int width) {
    if (width != 8 && width != 32) return 40077;
    dsv4_dense_tile_width_state = width;
    return 0;
}
extern "C" int memra_dsv4_dense_tile_width_for_gate() { return dsv4_dense_tile_width_state; }
extern "C" int memra_dsv4_dense_tile_restore_default_for_gate() {
    dsv4_dense_tile_width_state = dsv4_dense_tile_environment_default();
    return dsv4_dense_tile_width_state;
}
extern "C" int memra_dsv4_dense_tile_counts_for_gate(uint64_t* tile8, uint64_t* tile32) {
    if (!tile8 || !tile32) return 40077;
    *tile8 = dsv4_dense_tile_launches[0];
    *tile32 = dsv4_dense_tile_launches[1];
    return 0;
}
