// Portable Ada (sm_89) fail-closed ABI stubs for the NVFP4 W4A8/F8F4 MMQ launchers.
// Their .kind::f8f6f4 tile MMA needs sm_100a+; the quantizer/act-bytes helpers in
// mmq_nvfp4_f8f4.cu still compile on sm_89. These symbols exist only so the shared
// Rust FFI surface links without making an unsupported MMA instruction available.
#include <cstddef>

extern "C" size_t memra_mmq_nvfp4_w4a8_act_bytes(int, int) {
    return 0;
}

extern "C" int memra_mmq_nvfp4_w4a8(
        const void *, const float *, float *, int, int, int, void *, void *, float, int) {
    return 2902;
}

extern "C" int memra_mmq_nvfp4_f8f4(
        const void *, const float *, float *, int, int, int, void *, void *, float, int) {
    return 2903;
}

extern "C" int memra_mmq_nvfp4_f8f4_preq(
        const void *, const void *, float *, int, int, int, void *, float, int) {
    return 2904;
}

extern "C" size_t memra_mmq_nvfp4_f8f4_v2_img_bytes(int, int) {
    return 0;
}

extern "C" int memra_mmq_nvfp4_f8f4_v2_preq(
        const void *, const void *, float *, int, int, int, void *, void *, float, int) {
    return 2905;
}

extern "C" int memra_mmq_nvfp4_f8f4_v2_gateup_silu(
        const void *, const void *, const void *, void *, int, int, int, void *, void *, float, float,
        int) {
    return 2905;
}
