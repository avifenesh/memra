// CUDA 13.1 current-engine control plus driver-module launcher for R8.
// The external R8 cubin is the original F32 GEMV order with native FP8 pair
// decode. This launcher owns the exact-output, edge, guard, warmup, L2 flush,
// and ABBA checks without using CUDA 13.3 cudart or PTX JIT.

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

namespace dsv4_dense_tc_gate_r8 {

using namespace dsv4_dense_tc_gate;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::size_t;

constexpr int kGuardFloats = 16;
constexpr uint32_t kGuardWord = 0xa5a5a5a5u;
constexpr const char* kKernelName = "r8_gemv_fp8_native";

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

struct ModuleInfo { CUmodule module = nullptr; CUfunction function = nullptr; };

static void check_abi(CUfunction function) {
    constexpr size_t kCount = 7;
    constexpr size_t kOffsets[kCount] = {0, 8, 16, 24, 32, 40, 44};
    constexpr size_t kSizes[kCount] = {8, 8, 4, 8, 8, 4, 4};
    for (size_t i = 0; i < kCount; ++i) {
        size_t offset = 0, size = 0;
        cu_check(cuFuncGetParamInfo(function, i, &offset, &size), "query R8 ABI");
        std::printf("abi kernel=%s param[%zu] offset=%zu size=%zu\n",
                    kKernelName, i, offset, size);
        if (offset != kOffsets[i] || size != kSizes[i])
            throw std::runtime_error("R8 ABI mismatch at parameter " + std::to_string(i));
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
    cu_check(cuModuleLoad(&info.module, cubin), "load R8 cubin");
    cu_check(cuModuleGetFunction(&info.function, info.module, kKernelName),
             "lookup R8 kernel");
    check_abi(info.function);
    int regs = 0, shared = 0;
    cu_check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, info.function),
             "query R8 registers");
    cu_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, info.function),
             "query R8 shared");
    std::printf("module=%s kernel=%s regs=%d shared=%d\n", cubin, kKernelName, regs, shared);
    return info;
}

static HostData make_edge_data() {
    const int rows = 4, k = 256, sc_cols = 2;
    HostData h{rows, k, sc_cols};
    h.codes.resize(static_cast<size_t>(rows) * k);
    h.scales = {std::ldexp(1.0f, -4), std::ldexp(1.0f, 4)};
    h.input.resize(k);
    h.control.resize(rows);
    h.candidate.resize(rows);
    const uint8_t pattern[] = {0x00u, 0x80u, 0x7fu, 0xffu, 0x38u, 0xb8u, 0x7eu, 0xfeu};
    const uint16_t xpattern[] = {0x0000u, 0x8000u, 0x3f80u, 0xbf80u, 0x3f00u, 0xbf00u};
    for (int i = 0; i < k; ++i) h.input[i] = xpattern[i % 6];
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < k; ++col)
            h.codes[static_cast<size_t>(row) * k + col] = pattern[(col + row) % 8];
    return h;
}

static float f32_add(float lhs, float rhs) {
    volatile float result = lhs + rhs;
    return result;
}

// mode 0 is the shipped all-A-then-all-B order; mode 1 is the old pair
// interleave (A0,A1,B0,B1,...); mode 2 interleaves individual products.
static float cancellation_anchor(int k, int mode) {
    float thread_part = 0.0f;
    const float large = std::ldexp(1.0f, 30);
    const float a[8] = {large, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float b[8] = {-large, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    for (int segment = 0; segment < k / 2048; ++segment) {
        if (mode == 0) {
            for (int j = 0; j < 8; ++j) thread_part = f32_add(thread_part, a[j]);
            for (int j = 0; j < 8; ++j) thread_part = f32_add(thread_part, b[j]);
        } else if (mode == 1) {
            for (int j = 0; j < 8; j += 2) {
                thread_part = f32_add(thread_part, a[j]);
                thread_part = f32_add(thread_part, a[j + 1]);
                thread_part = f32_add(thread_part, b[j]);
                thread_part = f32_add(thread_part, b[j + 1]);
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                thread_part = f32_add(thread_part, a[j]);
                thread_part = f32_add(thread_part, b[j]);
            }
        }
    }
    float reduction = 0.0f;
    for (int tid = 0; tid < 128; ++tid) reduction = f32_add(reduction, thread_part);
    return reduction;
}

static HostData make_cancellation_data(int k) {
    const int rows = 1;
    HostData h{rows, k, (k + 127) / 128};
    h.codes.assign(static_cast<size_t>(rows) * k, 0x38u);
    h.scales.assign(h.sc_cols, 1.0f);
    h.input.assign(k, 0x3f80u); // BF16 +1
    h.control.resize(rows);
    h.candidate.resize(rows);
    // Each 2048-wide segment has, per thread, eight A products
    // [2^30,1,1,...] followed by eight B products [-2^30,1,1,...].
    // The shipped order yields 7 per thread. The preflight computes the old
    // pair-interleaved and element-interleaved alternatives explicitly.
    for (int base = 0; base < k; base += 2048) {
        for (int tid = 0; tid < 128; ++tid) {
            const int a = base + tid * 8;
            const int b = a + 1024;
            if (b + 7 >= k) continue;
            h.codes[a] = 0x38u;
            h.input[a] = 0x4e80u; // BF16 2^30
            h.codes[b] = 0xb8u;
            h.input[b] = 0x4e80u; // BF16 2^30
        }
    }
    return h;
}

static void arm_guard(float* output, int rows) {
    check(cudaMemset(output + rows, 0xa5, kGuardFloats * sizeof(float)), "arm R8 guard");
}

static void check_guard(float* output, int rows, const char* arm) {
    uint32_t got[kGuardFloats] = {};
    check(cudaMemcpy(got, output + rows, sizeof(got), cudaMemcpyDeviceToHost), "read R8 guard");
    for (int i = 0; i < kGuardFloats; ++i)
        if (got[i] != kGuardWord) throw std::runtime_error(std::string(arm) + " wrote past guard");
}

static double run_candidate(const HostData& h, CUfunction function, uint8_t* d_codes,
                            float* d_scales, uint16_t* d_input, float* d_output,
                            uint32_t* d_flush, size_t flush_words, cudaStream_t stream) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "R8 create start");
    check(cudaEventCreate(&stop), "R8 create stop");
    flush(stream, d_flush, flush_words);
    check(cudaEventRecord(start, stream), "R8 candidate start");
    uint8_t* w = d_codes;
    float* sc = d_scales;
    int sc_cols = h.sc_cols;
    uint16_t* x = d_input;
    float* y = d_output;
    int n = h.rows, k = h.k;
    void* args[] = {&w, &sc, &sc_cols, &x, &y, &n, &k};
    cu_check(cuLaunchKernel(function, static_cast<unsigned>(h.rows), 1, 1,
                            128, 1, 1, 0, reinterpret_cast<CUstream>(stream), args, nullptr),
             "launch R8 candidate");
    check(cudaEventRecord(stop, stream), "R8 candidate stop");
    check(cudaEventSynchronize(stop), "R8 candidate sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "R8 candidate elapsed");
    check(cudaEventDestroy(start), "R8 destroy start");
    check(cudaEventDestroy(stop), "R8 destroy stop");
    return ms;
}

static void require_bit_identity(const HostData& h, int row_count, const char* arm) {
    if (std::memcmp(h.control.data(), h.candidate.data(),
                    static_cast<size_t>(row_count) * sizeof(float)) != 0) {
        for (int row = 0; row < row_count; ++row) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &h.control[row], sizeof(a));
            std::memcpy(&b, &h.candidate[row], sizeof(b));
            if (a != b) {
                std::fprintf(stderr, "FAIL %s bit mismatch row=%d control=0x%08x candidate=0x%08x\n",
                             arm, row, a, b);
                break;
            }
        }
        throw std::runtime_error("R8 candidate is not bit-identical to current control");
    }
}

static int run_case(CUfunction function, HostData h, bool full) {
    const int rows = h.rows, k = h.k;
    if (k <= 0 || k % 8 != 0) throw std::runtime_error("R8 k must be a positive multiple of 8");
    uint8_t* d_codes = nullptr;
    float* d_scales = nullptr;
    uint16_t* d_input = nullptr;
    float* d_control = nullptr;
    float* d_candidate = nullptr;
    uint32_t* d_flush = nullptr;
    int l2_bytes = 0;
    check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0), "query L2");
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(l2_bytes), 1u << 20);
    const size_t flush_words = (flush_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    check(cudaMalloc(&d_codes, h.codes.size()), "R8 codes alloc");
    check(cudaMalloc(&d_scales, h.scales.size() * sizeof(float)), "R8 scales alloc");
    check(cudaMalloc(&d_input, h.input.size() * sizeof(uint16_t)), "R8 input alloc");
    check(cudaMalloc(&d_control, (rows + kGuardFloats) * sizeof(float)), "R8 control alloc");
    check(cudaMalloc(&d_candidate, (rows + kGuardFloats) * sizeof(float)), "R8 candidate alloc");
    check(cudaMalloc(&d_flush, flush_words * sizeof(uint32_t)), "R8 flush alloc");
    check(cudaMemset(d_flush, 0, flush_words * sizeof(uint32_t)), "R8 flush init");
    check(cudaMemcpy(d_codes, h.codes.data(), h.codes.size(), cudaMemcpyHostToDevice), "R8 codes copy");
    check(cudaMemcpy(d_scales, h.scales.data(), h.scales.size() * sizeof(float), cudaMemcpyHostToDevice), "R8 scales copy");
    check(cudaMemcpy(d_input, h.input.data(), h.input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "R8 input copy");
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "R8 stream create");
    auto control_arm = [&](const char* label) {
        arm_guard(d_control, rows);
        const double ms = run_current(h, d_codes, d_scales, d_input, d_control,
                                      d_flush, flush_words, stream);
        check_guard(d_control, rows, label);
        return ms;
    };
    auto candidate_arm = [&](const char* label) {
        arm_guard(d_candidate, rows);
        const double ms = run_candidate(h, function, d_codes, d_scales, d_input,
                                        d_candidate, d_flush, flush_words, stream);
        check_guard(d_candidate, rows, label);
        return ms;
    };
    const double warm_control = control_arm("R8 control warmup");
    const double warm_candidate = candidate_arm("R8 candidate warmup");
    std::printf("WARMUP rows=%d k=%d control_ms=%.6f candidate_ms=%.6f\n",
                rows, k, warm_control, warm_candidate);
    std::vector<float> control_first(rows), candidate_first(rows);
    bool have_control = false, have_candidate = false;
    std::vector<double> control_samples, candidate_samples;
    const int cycles = full ? 3 : 1;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        control_samples.push_back(control_arm("R8 control scored"));
        record_scored_output(d_control, h.control, control_first, have_control, rows, "control");
        candidate_samples.push_back(candidate_arm("R8 candidate scored"));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate, rows, "candidate");
        require_bit_identity(h, rows, "R8 candidate");
        candidate_samples.push_back(candidate_arm("R8 candidate scored repeat"));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate, rows, "candidate");
        require_bit_identity(h, rows, "R8 candidate repeat");
        control_samples.push_back(control_arm("R8 control scored repeat"));
        record_scored_output(d_control, h.control, control_first, have_control, rows, "control");
    }
    auto mean = [](const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    };
    std::printf("RESULT rows=%d k=%d full=%s control_ms=%.6f candidate_ms=%.6f ratio=%.6f bit_identity=true\n",
                rows, k, full ? "true" : "false", mean(control_samples), mean(candidate_samples),
                mean(candidate_samples) / mean(control_samples));
    check(cudaStreamDestroy(stream), "R8 stream destroy");
    cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_input); cudaFree(d_control);
    cudaFree(d_candidate); cudaFree(d_flush);
    return 0;
}

} // namespace dsv4_dense_tc_gate_r8

int main(int argc, char** argv) {
    dsv4_dense_tc_gate::pin_control_policy();
    if (argc == 2 && std::strcmp(argv[1], "--check-controls") == 0) {
        std::puts("PASS dense_tc_control dense_fast=0 cpu_policy_only=true"); return 0;
    }
    using namespace dsv4_dense_tc_gate_r8;
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s [--basis] <r8.cubin> [rows k]\n", argv[0]);
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
            for (int k : {128, 256, 2048, 4096, 8192})
                    rc |= run_case(info.function, make_data(rows, k), false);
            for (int k : {2048, 4096, 8192}) {
                std::printf("CANCELLATION_ANCHOR k=%d ordered=%.9g pair_interleaved=%.9g element_interleaved=%.9g\n",
                            k, cancellation_anchor(k, 0), cancellation_anchor(k, 1),
                            cancellation_anchor(k, 2));
                rc |= run_case(info.function, make_cancellation_data(k), false);
            }
            rc |= run_case(info.function, make_edge_data(), false);
        } else {
            const int rows = argc > 2 ? std::atoi(argv[2]) : kDefaultRows;
            const int k = argc > 3 ? std::atoi(argv[3]) : kDefaultK;
            if (rows <= 0 || k <= 0 || k % 8 != 0) {
                std::fprintf(stderr, "rows must be positive and k a positive multiple of 8\n");
                return 2;
            }
            rc = run_case(info.function, make_data(rows, k), true);
        }
        cu_check(cuModuleUnload(info.module), "unload R8 cubin");
        if (rc == 0) std::puts("PASS R8 original-F32-order native decode gate");
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL R8 %s\n", e.what());
        return 1;
    }
}
