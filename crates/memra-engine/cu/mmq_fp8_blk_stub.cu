// Portable Ada (sm_89) fail-closed ABI stubs for the per-block FP8 MMQ launcher.
// mmq_fp8_blk.cu's tile MMA is .kind::f8f6f4 (sm_100a+); these symbols exist only so the shared
// Rust FFI surface links on sm_89 without making an unsupported MMA instruction available.
//
// THIS FILE IS AN ABI MIRROR. build.rs swaps it in for cu/mmq_fp8_blk.cu on the `portable`
// arches (89, 90a), and src/mmq_ffi.rs calls the whole surface unconditionally — the arch is a
// build-time env var, so there is no #[cfg] that could gate the Rust side. Every `extern "C"`
// entry point the real file exports must therefore exist here too, or the portable arches do
// not LINK. That is not hypothetical: adding `memra_mmq_fp8_blk_quantize_act` and
// `memra_mmq_fp8_blk_grouped` to the real file (58ce746ad3) without adding them here made main
// unreleasable from 2026-08-22, and killed the v0.105.0 and v0.106.0 release runs with
// `rust-lld: error: undefined symbol` in the sm_89 matrix cell. CI compiled sm_120a only, so
// nothing saw it until a tag existed. tools/stub-abi-census.py now refuses the drift, and
// ci.yml compiles both portable arches. Never make a body here silently succeed: a stub that
// returns wrong numbers is worse than one that refuses.
#include <cstddef>

extern "C" size_t memra_mmq_nvfp4_f8f4_act_bytes(int, int);

extern "C" size_t memra_mmq_fp8_blk_act_bytes(int in_f, int n_tokens) {
    return memra_mmq_nvfp4_f8f4_act_bytes(in_f, n_tokens);
}

extern "C" int memra_mmq_fp8_blk_scale_rows(int out_f) { return (out_f + 127) / 128; }
extern "C" int memra_mmq_fp8_blk_scale_cols(int in_f)  { return (in_f  + 127) / 128; }

extern "C" int memra_mmq_fp8_blk(
        const void *, const float *, const float *, float *, int, int, int, void *, void *, float) {
    return 2904;
}

extern "C" int memra_fp8_blk_count_nan(const void *, size_t, unsigned int *, void *) {
    return 2905;
}

// Stage 1 of the reusable grouped API (mmq_fp8_blk.cu:714). The real body launches
// quantize_mmq_e4m3_d128_kernel; there is no portable equivalent, so this refuses.
extern "C" int memra_mmq_fp8_blk_quantize_act(
        const float *, void *, int, int, void *) {
    return 2906;
}

// Stage 2 of the reusable grouped API (mmq_fp8_blk.cu:732). Same .kind::f8f6f4 tile MMA as
// memra_mmq_fp8_blk above, so the same refusal class.
extern "C" int memra_mmq_fp8_blk_grouped(
        const void *, const float *, const int *, const int *, const int *, const int *,
        const void *, float *, int, int, int, int, int, int, size_t, size_t, void *, float) {
    return 2907;
}
