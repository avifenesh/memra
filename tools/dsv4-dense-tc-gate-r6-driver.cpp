// CUDA 13.1 driver-only R6 shim.
// Loads the CUDA 13.3 cubin, checks the three-argument ABI, and validates raw
// plus engine-normalized conversion over every packed E4M3x2 byte pair.

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

static std::string cu_error(CUresult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    return std::string(name ? name : "CUDA_UNKNOWN") + ": "
        + (message ? message : "unknown");
}

static void cu_check(CUresult result, const char* where) {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "%s: %s\n", where, cu_error(result).c_str());
        std::exit(2);
    }
}

static float e4m3_finite(uint8_t code) {
    const unsigned mag = code & 0x7fu;
    if (mag == 0u) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float value;
    if (exp == 0u) {
        value = static_cast<float>(man) * 0x1p-9f;
    } else {
        const uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&value, &bits, sizeof(value));
    }
    return (code & 0x80u) ? -value : value;
}

static uint16_t bf16_bits_finite(uint8_t code) {
    if ((code & 0x7fu) == 0u) return (code & 0x80u) ? 0x8000u : 0u;
    uint32_t bits = 0;
    const float value = e4m3_finite(code);
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0xffffu) != 0u) {
        std::fprintf(stderr, "non-BF16-exact finite E4M3 code 0x%02x\n", code);
        std::exit(3);
    }
    return static_cast<uint16_t>(bits >> 16);
}

static uint8_t engine_normalize(uint8_t code) {
    const uint8_t mag = code & 0x7fu;
    return (mag == 0u || mag == 0x7fu) ? 0u : code;
}

static bool is_bf16_nan(uint16_t bits) {
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0u;
}

static void check_abi(CUfunction function, const char* name) {
    constexpr std::size_t kCount = 3;
    constexpr std::size_t kOffsets[kCount] = {0, 8, 16};
    constexpr std::size_t kSizes[kCount] = {8, 8, 4};
    for (std::size_t i = 0; i < kCount; ++i) {
        std::size_t offset = 0;
        std::size_t size = 0;
        cu_check(cuFuncGetParamInfo(function, i, &offset, &size), "cuFuncGetParamInfo");
        std::printf("abi kernel=%s param[%zu] offset=%zu size=%zu\n",
                    name, i, offset, size);
        if (offset != kOffsets[i] || size != kSizes[i]) {
            std::fprintf(stderr, "FAIL ABI kernel=%s param[%zu] expected offset=%zu size=%zu\n",
                         name, i, kOffsets[i], kSizes[i]);
            std::exit(7);
        }
    }
}

static std::vector<uint32_t> all_pairs() {
    std::vector<uint32_t> values(65536);
    for (uint32_t i = 0; i < values.size(); ++i) values[i] = i;
    return values;
}

static std::vector<uint32_t> run_kernel(CUfunction function,
                                        CUdeviceptr d_input, int n) {
    CUdeviceptr d_output = 0;
    cu_check(cuMemAlloc(&d_output, static_cast<std::size_t>(n) * sizeof(uint32_t)),
             "cuMemAlloc output");
    void* args[] = {&d_input, &d_output, &n};
    cu_check(cuLaunchKernel(function, (n + 255) / 256, 1, 1, 256, 1, 1, 0,
                            nullptr, args, nullptr), "cuLaunchKernel R6");
    cu_check(cuCtxSynchronize(), "cuCtxSynchronize R6");
    std::vector<uint32_t> output(static_cast<std::size_t>(n));
    cu_check(cuMemcpyDtoH(output.data(), d_output,
                          output.size() * sizeof(uint32_t)), "cuMemcpyDtoH R6");
    cu_check(cuMemFree(d_output), "cuMemFree output");
    return output;
}

static void validate_raw(const std::vector<uint32_t>& inputs,
                         const std::vector<uint32_t>& output) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t lo = static_cast<uint8_t>(inputs[i]);
        const uint8_t hi = static_cast<uint8_t>(inputs[i] >> 8);
        const uint16_t got_lo = static_cast<uint16_t>(output[i]);
        const uint16_t got_hi = static_cast<uint16_t>(output[i] >> 16);
        const bool lo_nan = (lo & 0x7fu) == 0x7fu;
        const bool hi_nan = (hi & 0x7fu) == 0x7fu;
        if (lo_nan ? !is_bf16_nan(got_lo) : got_lo != bf16_bits_finite(lo)) {
            std::fprintf(stderr, "FAIL raw low pair=0x%04x got=0x%04x code=0x%02x\n",
                         static_cast<unsigned>(inputs[i]), got_lo, lo);
            std::exit(7);
        }
        if (hi_nan ? !is_bf16_nan(got_hi) : got_hi != bf16_bits_finite(hi)) {
            std::fprintf(stderr, "FAIL raw high pair=0x%04x got=0x%04x code=0x%02x\n",
                         static_cast<unsigned>(inputs[i]), got_hi, hi);
            std::exit(7);
        }
    }
    std::puts("PASS raw all65536 finite/signed-zero exact; NaN policy=any BF16 NaN");
}

static void validate_engine(const std::vector<uint32_t>& inputs,
                            const std::vector<uint32_t>& output) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const uint8_t lo = engine_normalize(static_cast<uint8_t>(inputs[i]));
        const uint8_t hi = engine_normalize(static_cast<uint8_t>(inputs[i] >> 8));
        const uint32_t expected = static_cast<uint32_t>(bf16_bits_finite(hi)) << 16
                                | bf16_bits_finite(lo);
        if (output[i] != expected) {
            std::fprintf(stderr, "FAIL engine pair=0x%04x got=0x%08x want=0x%08x\n",
                         static_cast<unsigned>(inputs[i]), output[i], expected);
            std::exit(7);
        }
    }
    std::puts("PASS engine-normalized all65536 exact; NaN/signed-zero -> +0 policy");
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[1], "--load-only") != 0)) {
        std::fprintf(stderr, "usage: %s [--load-only] <r6-sm120f-or-sm120a.cubin>\n", argv[0]);
        return 2;
    }
    const bool load_only = argc == 3;
    const char* cubin = load_only ? argv[2] : argv[1];
    cu_check(cuInit(0), "cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    cu_check(cuDeviceGet(&device, 0), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule module = nullptr;
    cu_check(cuModuleLoad(&module, cubin), "cuModuleLoad R6 cubin");
    CUfunction raw = nullptr;
    CUfunction engine = nullptr;
    cu_check(cuModuleGetFunction(&raw, module, "r6_raw_packed_cvt_basis"),
             "lookup R6 raw kernel");
    cu_check(cuModuleGetFunction(&engine, module, "r6_engine_packed_cvt_basis"),
             "lookup R6 engine kernel");
    check_abi(raw, "r6_raw_packed_cvt_basis");
    check_abi(engine, "r6_engine_packed_cvt_basis");
    std::printf("module=%s kernels=raw+engine pairs=65536\n", cubin);
    if (load_only) {
        cu_check(cuModuleUnload(module), "cuModuleUnload R6");
        cu_check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
        std::puts("PASS R6 cubin load/function lookup/ABI; no kernel launch requested");
        return 0;
    }
    const std::vector<uint32_t> inputs = all_pairs();
    CUdeviceptr d_input = 0;
    cu_check(cuMemAlloc(&d_input, inputs.size() * sizeof(uint32_t)), "cuMemAlloc input");
    cu_check(cuMemcpyHtoD(d_input, inputs.data(), inputs.size() * sizeof(uint32_t)),
             "cuMemcpyHtoD input");
    const std::vector<uint32_t> raw_output = run_kernel(raw, d_input, 65536);
    const std::vector<uint32_t> engine_output = run_kernel(engine, d_input, 65536);
    validate_raw(inputs, raw_output);
    validate_engine(inputs, engine_output);
    cu_check(cuMemFree(d_input), "cuMemFree input");
    cu_check(cuModuleUnload(module), "cuModuleUnload R6");
    cu_check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    std::puts("PASS R6 raw characterization + engine-normalized all-pairs gate");
    return 0;
}
