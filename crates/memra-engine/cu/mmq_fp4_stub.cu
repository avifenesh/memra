// Portable (sm_89 / sm_90a) fail-closed ABI stubs for the W4A4 MMQ path.
// build.rs compiles the real mmq_fp4.cu on both Blackwell targets (sm_120a warp
// MMA, sm_100a tcgen05/TMEM twin) and swaps this file in on every other arch.
// These symbols exist only so the shared Rust FFI surface links without making
// an unsupported block-scaled MMA instruction available.
#include <cstddef>

extern "C" size_t memra_mmq_nvfp4_act_bytes(int, int) {
    return 0;
}

extern "C" int memra_mmq_nvfp4_ex2(
        const void *, const float *, float *, int, int, int, void *, void *, float, int, int) {
    return 2901;
}

extern "C" int memra_a4_stats_accumulate(const float *, long long, void *, void *) {
    return 2901;
}

extern "C" size_t memra_mmq_nvfp4_calibrated_prefill_act_bytes(int, int) {
    return 0;
}

extern "C" int memra_mmq_nvfp4_calibrated_prefill(
        const void *, const float *, float *, int, int, int, void *, void *, float, float, int,
        void *) {
    return 2901;
}
