// CUDA 13.1 current-engine control plus driver-module launcher for the CUDA
// 13.3 R7 native-conversion cubin. R3 is included only for the actual current
// memra_dsv4_gemv_fp8_m control and receipt helpers. The external candidate is
// a cubin, so the launcher does not use CUDA 13.3 cudart or PTX JIT.

#define main dsv4_dense_tc_gate_r3_unused_main
#include "dsv4-dense-tc-gate.cu"
#undef main

#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4_dense_tc_gate_r7 {

using namespace dsv4_dense_tc_gate;
using std::int8_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::size_t;

constexpr int kGuardFloats = 16;
constexpr uint32_t kGuardWord = 0xa5a5a5a5u;
constexpr const char* kKernelName = "r7_dense_native";

static std::string cu_error(CUresult result) {
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    return std::string(name ? name : "CUDA_UNKNOWN") + ": "
        + (message ? message : "unknown");
}

static void cu_check(CUresult result, const char* where) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(where) + ": " + cu_error(result));
}

static int scale_exponent(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffu) != 0u || (bits >> 31) != 0u || value == 0.0f)
        throw std::runtime_error("R7 requires positive exact power-of-two scales");
    return static_cast<int>((bits >> 23) & 0xffu) - 127;
}

static std::vector<int8_t> make_scale_exponents(const HostData& h) {
    std::vector<int8_t> out(h.scales.size());
    for (size_t i = 0; i < h.scales.size(); ++i)
        out[i] = static_cast<int8_t>(scale_exponent(h.scales[i]));
    for (size_t i = 0; i < h.codes.size(); ++i) {
        const int row = static_cast<int>(i / h.k);
        const int col = static_cast<int>(i % h.k);
        const float value = e4m3(h.codes[i])
            * h.scales[(row / 128) * h.sc_cols + col / 128];
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0xffffu) != 0u || static_cast<uint16_t>(bits >> 16) != h.shadow[i])
            throw std::runtime_error("R7 host BF16 shadow/exponent proof failed");
    }
    return out;
}

struct ModuleInfo {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

static void check_abi(CUfunction function) {
    constexpr size_t kCount = 7;
    constexpr size_t kOffsets[kCount] = {0, 8, 16, 24, 32, 36, 40};
    constexpr size_t kSizes[kCount] = {8, 8, 8, 8, 4, 4, 4};
    for (size_t i = 0; i < kCount; ++i) {
        size_t offset = 0, size = 0;
        cu_check(cuFuncGetParamInfo(function, i, &offset, &size), "query R7 ABI");
        std::printf("abi kernel=%s param[%zu] offset=%zu size=%zu\n",
                    kKernelName, i, offset, size);
        if (offset != kOffsets[i] || size != kSizes[i])
            throw std::runtime_error("R7 kernel ABI mismatch at parameter "
                                     + std::to_string(i));
    }
}

static ModuleInfo load_module(const char* cubin) {
    check(cudaSetDevice(0), "set device");
    check(cudaFree(nullptr), "create primary context");
    cu_check(cuInit(0), "cuInit");
    CUcontext context = nullptr;
    cu_check(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (!context) throw std::runtime_error("no current CUDA context");
    ModuleInfo info;
    cu_check(cuModuleLoad(&info.module, cubin), "load R7 cubin");
    cu_check(cuModuleGetFunction(&info.function, info.module, kKernelName),
             "lookup R7 kernel");
    check_abi(info.function);
    int regs = 0, shared = 0;
    cu_check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, info.function),
             "query R7 registers");
    cu_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, info.function),
             "query R7 shared");
    std::printf("module=%s kernel=%s regs=%d shared=%d\n", cubin, kKernelName, regs, shared);
    return info;
}

static void arm_guard(float* output, int rows) {
    check(cudaMemset(output + rows, 0xa5, kGuardFloats * sizeof(float)),
               "arm output guard");
}

static void check_guard(float* output, int rows, const char* arm) {
    uint32_t got[kGuardFloats] = {};
    check(cudaMemcpy(got, output + rows, sizeof(got), cudaMemcpyDeviceToHost),
               "read output guard");
    for (int i = 0; i < kGuardFloats; ++i) {
        if (got[i] != kGuardWord)
            throw std::runtime_error(std::string(arm) + " wrote past output guard");
    }
}

static double run_candidate(const HostData& h, CUfunction function,
                            uint8_t* d_codes, int8_t* d_exp, uint16_t* d_input,
                            float* d_output, uint32_t* d_flush, size_t flush_words,
                            cudaStream_t runtime_stream) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "R7 create start");
    check(cudaEventCreate(&stop), "R7 create stop");
    flush(runtime_stream, d_flush, flush_words);
    check(cudaEventRecord(start, runtime_stream), "R7 candidate start");
    uint8_t* codes = d_codes;
    int8_t* exp = d_exp;
    uint16_t* input = d_input;
    float* output = d_output;
    int rows = h.rows, k = h.k, sc_cols = h.sc_cols;
    void* args[] = {&codes, &exp, &input, &output, &rows, &k, &sc_cols};
    const unsigned blocks = static_cast<unsigned>((h.rows + 63) / 64);
    cu_check(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0,
                            reinterpret_cast<CUstream>(runtime_stream), args, nullptr),
             "launch R7 candidate");
    check(cudaEventRecord(stop, runtime_stream), "R7 candidate stop");
    check(cudaEventSynchronize(stop), "R7 candidate sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "R7 candidate elapsed");
    check(cudaEventDestroy(start), "R7 destroy start");
    check(cudaEventDestroy(stop), "R7 destroy stop");
    return ms;
}

static double run_control(const HostData& h, uint8_t* d_codes, float* d_scales,
                          uint16_t* d_input, float* d_output, uint32_t* d_flush,
                          size_t flush_words, cudaStream_t stream) {
    return run_current(h, d_codes, d_scales, d_input, d_output,
                       d_flush, flush_words, stream);
}

static int run_case(CUfunction function, int rows, int k, bool full) {
    HostData h = make_data(rows, k);
    const std::vector<int8_t> exp = make_scale_exponents(h);
    uint8_t* d_codes = nullptr;
    float* d_scales = nullptr;
    int8_t* d_exp = nullptr;
    uint16_t* d_input = nullptr;
    float* d_control = nullptr;
    float* d_candidate = nullptr;
    uint32_t* d_flush = nullptr;
    int l2_bytes = 0;
    check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0), "query L2");
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(l2_bytes), 1u << 20);
    const size_t flush_words = (flush_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    check(cudaMalloc(&d_codes, h.codes.size()), "R7 codes alloc");
    check(cudaMalloc(&d_scales, h.scales.size() * sizeof(float)), "R7 scales alloc");
    check(cudaMalloc(&d_exp, exp.size() * sizeof(int8_t)), "R7 exponent alloc");
    check(cudaMalloc(&d_input, h.input.size() * sizeof(uint16_t)), "R7 input alloc");
    check(cudaMalloc(&d_control, (rows + kGuardFloats) * sizeof(float)), "R7 control alloc");
    check(cudaMalloc(&d_candidate, (rows + kGuardFloats) * sizeof(float)), "R7 candidate alloc");
    check(cudaMalloc(&d_flush, flush_words * sizeof(uint32_t)), "R7 flush alloc");
    check(cudaMemset(d_flush, 0, flush_words * sizeof(uint32_t)), "R7 flush init");
    check(cudaMemcpy(d_codes, h.codes.data(), h.codes.size(), cudaMemcpyHostToDevice), "R7 codes copy");
    check(cudaMemcpy(d_scales, h.scales.data(), h.scales.size() * sizeof(float), cudaMemcpyHostToDevice), "R7 scales copy");
    check(cudaMemcpy(d_exp, exp.data(), exp.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "R7 exponent copy");
    check(cudaMemcpy(d_input, h.input.data(), h.input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "R7 input copy");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "R7 stream create");

    auto control_arm = [&](const char* label) {
        arm_guard(d_control, rows);
        const double ms = run_control(h, d_codes, d_scales, d_input, d_control,
                                      d_flush, flush_words, stream);
        check_guard(d_control, rows, label);
        return ms;
    };
    auto candidate_arm = [&](const char* label) {
        arm_guard(d_candidate, rows);
        const double ms = run_candidate(h, function, d_codes, d_exp, d_input,
                                        d_candidate, d_flush, flush_words, stream);
        check_guard(d_candidate, rows, label);
        return ms;
    };

    const double warm_control = control_arm("control warmup");
    const double warm_candidate = candidate_arm("candidate warmup");
    std::printf("WARMUP rows=%d k=%d control_ms=%.6f candidate_ms=%.6f\n",
                rows, k, warm_control, warm_candidate);
    std::vector<float> control_first(rows), candidate_first(rows);
    bool have_control = false, have_candidate = false;
    std::vector<double> control_samples, candidate_samples;
    const int cycles = full ? 3 : 1;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        control_samples.push_back(control_arm("control scored"));
        record_scored_output(d_control, h.control, control_first, have_control, rows, "control");
        candidate_samples.push_back(candidate_arm("candidate scored"));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate, rows, "candidate");
        float max_abs = 0.0f, max_rel = 0.0f;
        validate_numeric_band(h.control, h.candidate, cycle, kDefaultAbsTol, kDefaultRelTol,
                              max_abs, max_rel);
        candidate_samples.push_back(candidate_arm("candidate scored repeat"));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate, rows, "candidate");
        validate_numeric_band(h.control, h.candidate, cycle, kDefaultAbsTol, kDefaultRelTol,
                              max_abs, max_rel);
        control_samples.push_back(control_arm("control scored repeat"));
        record_scored_output(d_control, h.control, control_first, have_control, rows, "control");
    }
    auto mean = [](const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    };
    std::printf("RESULT rows=%d k=%d full=%s control_ms=%.6f candidate_ms=%.6f ratio=%.6f\n",
                rows, k, full ? "true" : "false", mean(control_samples), mean(candidate_samples),
                mean(candidate_samples) / mean(control_samples));
    check(cudaStreamDestroy(stream), "R7 stream destroy");
    cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_exp); cudaFree(d_input);
    cudaFree(d_control); cudaFree(d_candidate); cudaFree(d_flush);
    return 0;
}

} // namespace dsv4_dense_tc_gate_r7

int main(int argc, char** argv) {
    using namespace dsv4_dense_tc_gate_r7;
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s [--basis] <r7.cubin> [rows k]\n", argv[0]);
        return 2;
    }
    const bool basis = std::strcmp(argv[1], "--basis") == 0;
    const char* cubin = basis ? (argc > 2 ? argv[2] : nullptr) : argv[1];
    if (!cubin) return 2;
    try {
        ModuleInfo info = load_module(cubin);
        int rc = 0;
        if (basis) {
            for (int rows : {1, 64, 65, 129})
                for (int k : {128, 256}) rc |= run_case(info.function, rows, k, false);
        } else {
            const int rows = argc > 2 ? std::atoi(argv[2]) : kDefaultRows;
            const int k = argc > 3 ? std::atoi(argv[3]) : kDefaultK;
            if (rows <= 0 || k <= 0 || k % 128 != 0) {
                std::fprintf(stderr, "rows must be positive and k must be a positive multiple of 128\n");
                return 2;
            }
            rc = run_case(info.function, rows, k, true);
        }
        cu_check(cuModuleUnload(info.module), "unload R7 cubin");
        if (rc == 0) std::puts("PASS R7 native dense projection gate");
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL R7 %s\n", e.what());
        return 1;
    }
}
