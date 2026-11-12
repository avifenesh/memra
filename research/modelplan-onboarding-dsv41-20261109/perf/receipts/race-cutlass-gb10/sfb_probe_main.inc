// Race hunt probe (host only): the mainloop's SFB smem sizing against the SFB TMA box, per grouped/plain form.
// Included at the end of a copy of cu/dsv41_cutlass.cu (namespace v41c in scope).
#include <cstdio>
template <class C>
static void show(const char* name) {
    using M = typename C::Main;
    using LSFB = typename M::SmemLayoutSFB;
    using LSFA = typename M::SmemLayoutSFA;
    using LB = typename M::SmemLayoutB;
    using TS = typename M::TileShapeSFB;
    constexpr int stages = M::DispatchPolicy::Stages;
    long sfb_cosize = cute::cosize_v<LSFB>;
    long sfb_stage = cute::cosize(cute::take<0, 2>(LSFB{}));
    long sfa_cosize = cute::cosize_v<LSFA>;
    long b_cosize = cute::cosize_v<LB>;
    long box_rows = cute::size<1>(TS{});
    long box_k = cute::size<2>(TS{});
    printf("%-8s stages %d | SFB smem cosize %ld (per stage by take<0,2> %ld) | SFB TMA box %ld rows x %ld k -> %ld SF codes "
           "| tx MK %u NK %u | SFA cosize %ld | B cosize %ld | kernel smem %zu\n",
           name, stages, sfb_cosize, sfb_stage, box_rows, box_k, box_rows * box_k / 32, (unsigned)M::TmaTransactionBytesMK,
           (unsigned)M::TmaTransactionBytesNK, sfa_cosize, b_cosize, sizeof(typename C::Kernel::SharedStorage));
    printf("         SmemLayoutSFB: ");
    cute::print(LSFB{});
    printf("\n");
}
int main() {
    show<GS4x16>("GS4x16");
    show<GS4x8>("GS4x8");
    show<GN4>("GN4");
    show<PS4x16>("PS4x16");
    show<PS4x8>("PS4x8");
    show<PN4>("PN4");
    return 0;
}
