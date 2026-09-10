// Dense entry-point SHAPE CENSUS. An instrument, not a door: no environment
// read, no setter that changes what any kernel does, and every counter is
// host-thread-local. Included by dsv4_gpu.cu.
//
// Why it exists (memra #472). The dense GEMV/dots family is 40.6% of a served
// prefill's GPU-busy time, and tiling it wider is spent: memra #471 cut the
// launch count 4x for about a tenth of the family, which says roughly seven
// eighths of a dense launch is PER-ROW work that grouping rows does not touch.
// Naming that work needs the shapes, and the shapes are not in any doc: the
// four entry points are called from many sites with strides that make the
// visible tensor dims a poor guide. So the engine counts its own calls.
//
// What a row here buys, stated so the arithmetic below cannot be mistaken for a
// measurement. Each kernel's grid is `n` blocks, one per OUTPUT row, and every
// block reads:
//   * its own weight row: `k` elements once per launch, so `n * k` per launch;
//   * the WHOLE activation row for each of its `M` accumulators: `M * k`
//     elements PER BLOCK, so `n * M * k` per launch.
// The second term is multiplied by `n` and the first is not. That asymmetry is
// the hypothesis this census exists to price, and it is a hypothesis until the
// numbers land beside a measured duration.
#pragma once
#include <cstdint>
#include <cstring>

// Four entry points, indexed in call order below.
#define DSV4_DENSE_ENTRY_GEMV_BF16 0
#define DSV4_DENSE_ENTRY_GEMV_FP8 1
#define DSV4_DENSE_ENTRY_DOTS_F32 2
#define DSV4_DENSE_ENTRY_DOTS_F32ACC 3
#define DSV4_DENSE_CENSUS_SLOTS 256

struct Dsv4DenseCensusRow {
    int32_t entry;
    int32_t m;
    int32_t n;
    int32_t k;
    uint64_t calls;
};

static thread_local Dsv4DenseCensusRow dsv4_dense_census[DSV4_DENSE_CENSUS_SLOTS];
static thread_local int dsv4_dense_census_used = 0;
static thread_local uint64_t dsv4_dense_census_overflow = 0;
static thread_local int dsv4_dense_census_on = 0;

// Recording is OFF unless a census binary turns it on, so a serving process pays
// nothing but a predictable branch. It cannot change dispatch: the only state it
// touches is this table.
extern "C" int memra_dsv4_dense_census_set_for_gate(int on) {
    if (on != 0 && on != 1) return 40078;
    dsv4_dense_census_on = on;
    return 0;
}
extern "C" int memra_dsv4_dense_census_reset_for_gate() {
    std::memset(dsv4_dense_census, 0, sizeof(dsv4_dense_census));
    dsv4_dense_census_used = 0;
    dsv4_dense_census_overflow = 0;
    return 0;
}
static inline void dsv4_dense_census_note(int entry, int m, int n, int k) {
    if (!dsv4_dense_census_on) return;
    for (int i = 0; i < dsv4_dense_census_used; i++) {
        Dsv4DenseCensusRow& row = dsv4_dense_census[i];
        if (row.entry == entry && row.m == m && row.n == n && row.k == k) {
            row.calls++;
            return;
        }
    }
    if (dsv4_dense_census_used >= DSV4_DENSE_CENSUS_SLOTS) {
        dsv4_dense_census_overflow++;
        return;
    }
    Dsv4DenseCensusRow& row = dsv4_dense_census[dsv4_dense_census_used++];
    row.entry = entry;
    row.m = m;
    row.n = n;
    row.k = k;
    row.calls = 1;
}
// Copies at most `cap` rows out and returns the number used. `overflow` counts
// calls that found no free slot, so a truncated census says so instead of
// looking like a complete one.
extern "C" int memra_dsv4_dense_census_read_for_gate(Dsv4DenseCensusRow* out, int cap,
                                                     int* used, uint64_t* overflow) {
    if (!out || !used || !overflow || cap <= 0) return 40078;
    int count = dsv4_dense_census_used < cap ? dsv4_dense_census_used : cap;
    for (int i = 0; i < count; i++) out[i] = dsv4_dense_census[i];
    *used = dsv4_dense_census_used;
    *overflow = dsv4_dense_census_overflow;
    return 0;
}
