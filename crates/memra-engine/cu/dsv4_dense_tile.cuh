// Dense wide-prefill tiling instrument. Included by dsv4_gpu.cu, inherits its
// -fmad=false build. Holds NO policy: the tile width is `DSV4_TMAX`, in the code,
// with no environment variable and no runtime selection.
//
// History, because the constant here used to be a door. Above `DSV4_TMAX` the
// four dense entry points decompose a wide transaction into tiles and relaunch
// the registered-row kernel per tile. That width was a hard-coded 8, unrelated to
// the `M = 1..32` instantiations the dispatch switches actually carry, so a
// served 64-row prefill chunk issued eight launches per dense call where two
// would do. Tiling at `DSV4_TMAX` is bit-identical (M decides only how many
// independent accumulators a block keeps in registers, never one accumulator's
// add order or its 128-leaf tree) and measured +3.94% / +4.10% on the served
// program, so it is the code rather than a door (memra #463, #468, #470;
// darklanes `research/dsv4f-dense-tile-20260910/`).
//
// What survives is this counter, which is an instrument and not a switch: it is
// how `dsv4_dense_tile_gate` proves a transaction reached the tiled path at all,
// so its byte-identity assertions cannot pass vacuously on an untiled walk.
#pragma once
#include <cstdint>

// Tiled decompositions issued on this host thread, and the tiles they produced.
static thread_local uint64_t dsv4_dense_tile_decompositions = 0;
static thread_local uint64_t dsv4_dense_tile_launches = 0;

static inline void dsv4_dense_tile_count(int launches) {
    dsv4_dense_tile_decompositions += 1;
    dsv4_dense_tile_launches += (uint64_t)launches;
}

extern "C" int memra_dsv4_dense_tile_counts_for_gate(uint64_t* decompositions,
                                                     uint64_t* launches) {
    if (!decompositions || !launches) return 40077;
    *decompositions = dsv4_dense_tile_decompositions;
    *launches = dsv4_dense_tile_launches;
    return 0;
}
