// Fail-closed twin of cu/dsv41_cutlass.cu for targets outside the 120a family (CUTLASS's sm_120 block-scaled
// kernels exist only there): the launchers refuse, the layout helpers stay exact.
#include <cuda_runtime.h>

extern "C" long memra_v41_sf_off(long r, long kb, long kbn) {
    return (r / 128) * (((kbn + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}
extern "C" long memra_v41_sf_bytes(long rows, long k) { return (rows + 127) / 128 * 128 * (((k / 32) + 3) / 4) * 4; }
extern "C" int memra_v41_cutlass_gemm(const void*, long, const void*, const void*, long, const void*, int, void*, long,
                                      int, int, int, int, void*) {
    return 41049;
}
extern "C" int memra_v41_cutlass_sfrow(const int*, int, int, int, int*, void*) { return 41049; }
extern "C" long memra_v41_cutlass_moe_bytes(int) { return 0; }
extern "C" int memra_v41_cutlass_moe(const int*, int, int, int, int, int, const void*, const void*, const void*, void*,
                                     void*, long, int, int, long, int, int, void*, long, void*) {
    return 41049;
}
