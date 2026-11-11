// Host-only check (no GPU): CUTLASS 4.8.1's block-scaled scale-factor layouts for the sm_120 grouped kernels memra
// instantiates. (1) SFA of (rows, K) == SFB of (rows, K) as functions of (row, k-block), for the normal form (rows on
// A, tile 128x128x128) and the swapped forms (rows on B, tileN 8 and 16); (2) every one equals the closed form memra
// writes without CUTLASS headers:
//   off(r, kb) = (r / 128) * (ceil(KB / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4
// and its size is ceil(rows / 128) * 128 * ceil(KB / 4) * 4 bytes.
#include <stdio.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

template <class EA, class EB, int AlignA, int AlignB, class TileN, class LayoutD>
struct Cfg {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using Arch = cutlass::arch::Sm120;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<_128, TileN, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*, 8,
        cutlass::bfloat16_t, LayoutD*, 8, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OpClass, EA, cutlass::layout::RowMajor*, AlignA, EB, cutlass::layout::ColumnMajor*, AlignB, float, Tile,
        Cluster, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};
using F8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using F4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using Normal = Cfg<F8, F4, 16, 128, _128, cutlass::layout::RowMajor>;
using Dec8 = Cfg<F4, F8, 128, 16, _8, cutlass::layout::ColumnMajor>;
using Dec16 = Cfg<F4, F8, 128, 16, _16, cutlass::layout::ColumnMajor>;
using NormalF8 = Cfg<F8, F8, 16, 16, _128, cutlass::layout::RowMajor>;
using Dec8F8 = Cfg<F8, F8, 16, 16, _8, cutlass::layout::ColumnMajor>;

static long closed(long r, long kb, long KB) {
    long kbt = (KB + 3) / 4;
    return (r / 128) * (kbt * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}

template <class C>
static long check(const char* name, int rows, int other, int K) {
    int KB = K / 32;
    long bad = 0;
    // rows on A (SFA of (rows, other, K)) and rows on B (SFB of (other, rows, K))
    auto la = C::SfCfg::tile_atom_to_shape_SFA(make_shape(rows, other, K, 1));
    auto lb = C::SfCfg::tile_atom_to_shape_SFB(make_shape(other, rows, K, 1));
    long sa = size(filter_zeros(la)), sb = size(filter_zeros(lb));
    long want = (long)((rows + 127) / 128) * 128 * ((KB + 3) / 4) * 4;
    if (sa != want || sb != want) { printf("%s rows %d K %d: size A %ld B %ld closed %ld\n", name, rows, K, sa, sb, want); ++bad; }
    for (int r = 0; r < rows; ++r)
        for (int kb = 0; kb < KB; ++kb) {
            long a = la(r, kb * 32, 0), b = lb(r, kb * 32, 0), c = closed(r, kb, KB);
            // every element of the 32-wide block reads the same byte
            long a2 = la(r, kb * 32 + 31, 0);
            if (a != b || a != c || a2 != a) {
                if (bad < 5) printf("%s rows %d K %d: (%d, %d) SFA %ld SFB %ld closed %ld\n", name, rows, K, r, kb, a, b, c);
                ++bad;
            }
        }
    return bad;
}

int main() {
    long bad = 0, n = 0;
    for (int K : {1024, 2048, 4096, 5120, 6144, 8192, 2304, 1536, 512})
        for (int rows : {1, 2, 7, 8, 16, 31, 32, 64, 100, 128, 129, 300, 512, 1000}) {
            bad += check<Normal>("normal f4", rows, 2304, K);
            bad += check<Dec8>("swapped8 f4", rows, 2304, K);
            bad += check<Dec16>("swapped16 f4", rows, 2304, K);
            bad += check<NormalF8>("normal f8", rows, 1280, K);
            bad += check<Dec8F8>("swapped8 f8", rows, 1280, K);
            n += 5;
        }
    printf("SF-LAYOUT %s: %ld shape checks, %ld mismatches\n", bad ? "FAIL" : "PASS", n, bad);
    return bad != 0;
}
