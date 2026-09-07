// CUDA 13.1 driver-API loader/basis shim for the CUDA 13.3 R5 cubin.
//
// This is deliberately plain C++: it links libcuda only, not CUDA 13.3
// cudart, and loads a precompiled cubin with cuModuleLoad. A cubin path has no
// PTX for the driver to JIT. The ABI is the stable extern-C kernel:
//   r5_packed_cvt_basis(const uint32_t*, uint32_t*, int)

#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

static float e4m3(uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float value;
    if (exp == 0u) {
        value = static_cast<float>(man) * 0x1p-9f;
    } else {
        const uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&value, &bits, sizeof(value));
    }
    return (x & 0x80u) ? -value : value;
}

static uint16_t bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0xffffu) != 0u) {
        std::fprintf(stderr, "non-BF16-exact basis value %.9g\n", value);
        std::exit(3);
    }
    return static_cast<uint16_t>(bits >> 16);
}

static std::vector<uint32_t> basis_inputs() {
    return {
        0x00003838u, 0x0000bf80u, 0x00007e38u, 0x00007f00u,
        0x00003f38u, 0x00007e7eu, 0x00000000u, 0x0000ff38u,
    };
}

static std::vector<uint32_t> basis_expected(const std::vector<uint32_t>& inputs) {
    std::vector<uint32_t> expected(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const uint16_t lo = bf16_bits(e4m3(static_cast<uint8_t>(inputs[i] & 0xffu)));
        const uint16_t hi = bf16_bits(e4m3(static_cast<uint8_t>((inputs[i] >> 8) & 0xffu)));
        expected[i] = static_cast<uint32_t>(hi) << 16 | lo;
    }
    return expected;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[1], "--load-only") != 0)) {
        std::fprintf(stderr, "usage: %s [--load-only] <r5-sm120f.cubin>\n", argv[0]);
        return 2;
    }
    const bool load_only = argc == 3;
    const char* cubin = load_only ? argv[2] : argv[1];
    const std::vector<uint32_t> inputs = basis_inputs();
    const std::vector<uint32_t> expected = basis_expected(inputs);

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUdeviceptr d_input = 0;
    CUdeviceptr d_output = 0;
    cu_check(cuInit(0), "cuInit");
    cu_check(cuDeviceGet(&device, 0), "cuDeviceGet");
    cu_check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    cu_check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    cu_check(cuModuleLoad(&module, cubin), "cuModuleLoad cubin");
    cu_check(cuModuleGetFunction(&function, module, "r5_packed_cvt_basis"),
             "cuModuleGetFunction r5_packed_cvt_basis");
    int regs = 0;
    int shared = 0;
    cu_check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, function),
             "query native conversion registers");
    cu_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function),
             "query native conversion shared");
    std::printf("module=%s kernel=r5_packed_cvt_basis regs=%d shared=%d basis_vectors=%zu\n",
                cubin, regs, shared, inputs.size());
    if (load_only) {
        cu_check(cuModuleUnload(module), "cuModuleUnload");
        cu_check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
        std::puts("PASS cubin load/function lookup; no kernel launch requested");
        return 0;
    }

    cu_check(cuMemAlloc(&d_input, inputs.size() * sizeof(uint32_t)), "cuMemAlloc input");
    cu_check(cuMemAlloc(&d_output, expected.size() * sizeof(uint32_t)), "cuMemAlloc output");
    cu_check(cuMemcpyHtoD(d_input, inputs.data(), inputs.size() * sizeof(uint32_t)),
             "cuMemcpyHtoD input");
    int n = static_cast<int>(inputs.size());
    void* args[] = {&d_input, &d_output, &n};
    cu_check(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr),
             "cuLaunchKernel r5_packed_cvt_basis");
    cu_check(cuCtxSynchronize(), "cuCtxSynchronize basis");
    std::vector<uint32_t> actual(expected.size());
    cu_check(cuMemcpyDtoH(actual.data(), d_output, actual.size() * sizeof(uint32_t)),
             "cuMemcpyDtoH output");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL basis[%zu] input=0x%08x got=0x%08x want=0x%08x\n",
                         i, inputs[i], actual[i], expected[i]);
            cuMemFree(d_input);
            cuMemFree(d_output);
            cuModuleUnload(module);
            cuDevicePrimaryCtxRelease(device);
            return 7;
        }
    }
    cu_check(cuMemFree(d_input), "cuMemFree input");
    cu_check(cuMemFree(d_output), "cuMemFree output");
    cu_check(cuModuleUnload(module), "cuModuleUnload");
    cu_check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    std::printf("PASS packed_basis_vectors=%zu native=cvt.rn.bf16x2.e4m3x2 driver_api_only=true\n",
                expected.size());
    return 0;
}
