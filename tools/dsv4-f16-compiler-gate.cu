// Driver-loader gate for the CUDA 13.3 mma_throughput cubin versus the current
// CUDA 13.1-compiled engine kernel.  The included matched harness supplies the
// current sparse-bank fixture/data setup and current moe_f16_grouped.cu symbol;
// this file adds only the driver-module load, exact function lookup, occupancy,
// and optional same-buffer ABBA runner.
//
// Build with the current engine-side compiler/runtime, not CUDA 13.3 cudart:
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     -gencode arch=compute_120a,code=sm_120a -lcublas -lcuda \
//     -o target/dsv4-f16-compiler-gate tools/dsv4-f16-compiler-gate.cu
//
// Run only under the owning GPU lock:
//   target/dsv4-f16-compiler-gate target/mma-throughput-ptx/13.3-v93/moe.pragma.cubin
//   target/dsv4-f16-compiler-gate --load-only <cubin>
//
// The external module is a cubin, not PTX, so this path performs no PTX JIT.

#define main dsv4_matched_harness_unused_main
#include "dsv4-mixed-vs-f16-gate.cu"
#undef main

#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4_f16_compiler_gate {

using namespace dsv4_mixed_vs_f16;

static constexpr const char* kKernelName =
    "_Z20moe_kq_sktail_kernelILi108ELb1ELb1EEvPKyiiPKilPK6__halfPfPKfS3_iiiiii";

static std::string cu_error(const CUresult r) {
    const char* name = nullptr;
    const char* msg = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &msg);
    return std::string(name ? name : "CU_UNKNOWN") + ": " + (msg ? msg : "unknown");
}

static void cu_check(const CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) throw std::runtime_error(std::string(where) + ": " + cu_error(r));
}

static void build_sparse_banks(
        std::vector<std::uint8_t>& weights, std::vector<std::uint8_t>& scales,
        std::vector<HostExpert>& experts, DeviceBuffer<std::uint8_t>& d_weights,
        DeviceBuffer<std::uint8_t>& d_scales, DeviceBuffer<unsigned long long>& d_table) {
    weights.resize(static_cast<std::size_t>(kExperts) * kOutF * kRowBytes);
    scales.resize(static_cast<std::size_t>(kExperts) * kOutF * kWeightScaleGroups);
    experts.reserve(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        experts.push_back(make_host_expert(e));
        for (int row = 0; row < kOutF; ++row) {
            for (int k = 0; k < kInF; k += 2) {
                const std::uint8_t lo = weight_code(e, row, k);
                const std::uint8_t hi = weight_code(e, row, k + 1);
                weights[static_cast<std::size_t>(e) * kOutF * kRowBytes +
                        static_cast<std::size_t>(row) * kRowBytes + (k >> 1)] =
                    static_cast<std::uint8_t>(lo | (hi << 4));
            }
            for (int g = 0; g < kWeightScaleGroups; ++g) {
                scales[static_cast<std::size_t>(e) * kOutF * kWeightScaleGroups +
                       static_cast<std::size_t>(row) * kWeightScaleGroups + g] =
                    weight_scale_code(e, row, g);
            }
        }
    }
    cuda_check(cudaMemcpy(d_weights.p, weights.data(), weights.size(), cudaMemcpyHostToDevice),
               "upload sparse weights");
    cuda_check(cudaMemcpy(d_scales.p, scales.data(), scales.size(), cudaMemcpyHostToDevice),
               "upload sparse scales");

    const std::size_t weight_stride = static_cast<std::size_t>(kOutF) * kRowBytes;
    const std::size_t scale_stride = static_cast<std::size_t>(kOutF) * kWeightScaleGroups;
    std::vector<unsigned long long> table(static_cast<std::size_t>(6) * kBankExperts, 0);
    for (int group = 0; group < kExperts; ++group) {
        const int bank_id = 5 + 21 * group;
        const int payload = kTop6Ids[group];
        for (int projection = 0; projection < 3; ++projection) {
            table[2 * projection * kBankExperts + bank_id] =
                reinterpret_cast<std::uintptr_t>(d_weights.p + payload * weight_stride);
            table[(2 * projection + 1) * kBankExperts + bank_id] =
                reinterpret_cast<std::uintptr_t>(d_scales.p + payload * scale_stride);
        }
    }
    cuda_check(cudaMemcpy(d_table.p, table.data(), table.size() * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice), "upload sparse table");
}

struct ModuleInfo {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    int regs = 0;
    int static_shared = 0;
    int max_threads = 0;
    int ext_occupancy = 0;
    int current_occupancy = 0;
    int sms = 0;
};

static void validate_param_abi(CUfunction function) {
    // moe_kq_sktail_kernel's actual ABI is:
    // table, proj, n_expert, ex_ids, row_bytes, A, Y, row_scale, ex_off,
    // n_active, in_f, out_f, mlo, mhi, total_tiles.
    static constexpr std::size_t kParamSizes[] = {
        8, 4, 4, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4,
    };
    static constexpr std::size_t kParamOffsets[] = {
        0, 8, 12, 16, 24, 32, 40, 48, 56, 64, 68, 72, 76, 80, 84,
    };
    constexpr std::size_t kParamCount = sizeof(kParamSizes) / sizeof(kParamSizes[0]);
    for (std::size_t i = 0; i < kParamCount; ++i) {
        std::size_t offset = 0, size = 0;
        cu_check(cuFuncGetParamInfo(function, i, &offset, &size), "query kernel parameter ABI");
        std::printf("param[%zu] offset=%zu size=%zu\n", i, offset, size);
        if (offset != kParamOffsets[i] || size != kParamSizes[i]) {
            throw std::runtime_error("external kernel parameter ABI mismatch at index " +
                                     std::to_string(i));
        }
    }
}

static ModuleInfo load_module_and_query(const char* cubin) {
    cuda_check(cudaSetDevice(0), "set device");
    cuda_check(cudaFree(nullptr), "create primary context");
    cu_check(cuInit(0), "cuInit");
    CUcontext context = nullptr;
    cu_check(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
    if (!context) throw std::runtime_error("no current CUDA context after cudaFree(0)");

    ModuleInfo info;
    cu_check(cuModuleLoad(&info.module, cubin), "cuModuleLoad external cubin");
    cu_check(cuModuleGetFunction(&info.function, info.module, kKernelName),
             "cuModuleGetFunction M1 half2 kernel");
    validate_param_abi(info.function);
    cu_check(cuFuncGetAttribute(&info.regs, CU_FUNC_ATTRIBUTE_NUM_REGS, info.function),
             "query external register count");
    cu_check(cuFuncGetAttribute(&info.static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                info.function), "query external static shared");
    cu_check(cuFuncGetAttribute(&info.max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                info.function), "query external max threads");
    cu_check(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                 &info.ext_occupancy, info.function, 128, 1024),
             "query external occupancy");
    cuda_check(cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, 0),
               "query SM count");
    int current_occ = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &current_occ, ::moe_kq_sktail_kernel<QT_NVFP4_MODELOPT, true, true>,
                   128, 1024), "query current launcher occupancy");
    info.current_occupancy = current_occ;
    if (info.ext_occupancy < 1 || info.current_occupancy < 1)
        throw std::runtime_error("external/current occupancy returned zero");
    std::printf("module=%.256s kernel=%s regs=%d static_shared=%d max_threads=%d "
                "occupancy_external=%d occupancy_current=%d sms=%d grid_external=%d grid_current=%d\n",
                cubin, kKernelName, info.regs, info.static_shared, info.max_threads,
                info.ext_occupancy, info.current_occupancy, info.sms,
                info.ext_occupancy * info.sms, info.current_occupancy * info.sms);
    return info;
}

static void launch_external(
        CUfunction function, CUstream stream, const int grid,
        DeviceBuffer<unsigned long long>& d_table, DeviceBuffer<int>& d_ids,
        DeviceBuffer<__half>& d_act, DeviceBuffer<float>& d_out,
        DeviceBuffer<float>& d_row_scale, DeviceBuffer<int>& d_offsets) {
    unsigned long long* table = d_table.p;
    int proj = 1;
    int n_expert = kBankExperts;
    int* ex_ids = d_ids.p;
    long row_bytes = kRowBytes;
    __half* act = d_act.p;
    float* out = d_out.p;
    float* row_scale = d_row_scale.p;
    int* ex_off = d_offsets.p;
    int n_active = kBankExperts;
    int in_f = kInF;
    int out_f = kOutF;
    int mlo = 1;
    int mhi = 2;
    int total_tiles = -1;
    void* args[15] = {&table, &proj, &n_expert, &ex_ids, &row_bytes, &act, &out,
                    &row_scale, &ex_off, &n_active, &in_f, &out_f,
                    &mlo, &mhi, &total_tiles};
    cu_check(cuLaunchKernel(function, grid, 1, 1, 32, 4, 1, 1024, stream,
                            args, nullptr), "launch external M1 half2 cubin");
}

int run(const char* cubin, const bool load_only) {
    ModuleInfo info = load_module_and_query(cubin);
    if (load_only) {
        cu_check(cuModuleUnload(info.module), "unload external cubin");
        std::puts("PASS module load/function lookup/occupancy; no kernel launch requested");
        return 0;
    }

    std::vector<std::uint8_t> h_weights, h_scales;
    std::vector<HostExpert> experts;
    DeviceBuffer<std::uint8_t> d_weights(
        static_cast<std::size_t>(kExperts) * kOutF * kRowBytes);
    DeviceBuffer<std::uint8_t> d_scales(
        static_cast<std::size_t>(kExperts) * kOutF * kWeightScaleGroups);
    DeviceBuffer<unsigned long long> d_table(static_cast<std::size_t>(6) * kBankExperts);
    build_sparse_banks(h_weights, h_scales, experts, d_weights, d_scales, d_table);
    DeviceBuffer<int> d_ids(kExperts), d_ref_ids(kBankExperts), d_offsets(kBankExperts + 1);
    DeviceBuffer<std::uint8_t> d_act_codes(static_cast<std::size_t>(kExperts) * kInF);
    DeviceBuffer<float> d_act_scales(static_cast<std::size_t>(kExperts) * kActScaleGroups);
    DeviceBuffer<__half> d_act_f16(static_cast<std::size_t>(kExperts) * kInF);
    DeviceBuffer<float> d_row_scale(kExperts);
    DeviceBuffer<float> d_current(static_cast<std::size_t>(kBankExperts) * kOutF);
    DeviceBuffer<float> d_external(static_cast<std::size_t>(kBankExperts) * kOutF);
    int l2_bytes = 0;
    cuda_check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0), "query L2");
    DeviceBuffer<std::uint32_t> d_flush(
        (std::max<std::size_t>(2 * static_cast<std::size_t>(l2_bytes), 1u << 20) + 3) / 4);
    DeviceBuffer<std::uint32_t> d_sink(256 * 8);
    cuda_check(cudaMemset(d_flush.p, 0x5a, d_flush.n * sizeof(std::uint32_t)), "init flush");
    cudaStream_t runtime_stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&runtime_stream, cudaStreamNonBlocking), "create stream");
    CUstream driver_stream = reinterpret_cast<CUstream>(runtime_stream);
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "create start");
    cuda_check(cudaEventCreate(&stop), "create stop");

    const std::size_t output_bytes = static_cast<std::size_t>(kBankExperts) * kOutF * sizeof(float);
    const int current_grid = info.sms * info.current_occupancy;
    const int external_grid = info.sms * info.ext_occupancy;
    for (int active : {1, 3, 4, 6}) {
        std::vector<int> ids(kTop6Ids, kTop6Ids + active);
        upload_active(active, ids, experts, d_ids, d_ref_ids, d_offsets,
                      d_act_codes, d_act_scales, d_act_f16, d_row_scale);
        auto run_current = [&](std::vector<float>& host, float* ms) {
            flush_l2(d_flush, d_sink, 256, runtime_stream);
            cuda_check(cudaMemsetAsync(d_current.p, 0xa5, output_bytes, runtime_stream), "clear current");
            cuda_check(cudaEventRecord(start, runtime_stream), "current start");
            const int rc = memra_moe_kq_gemm_sk_m1_half2(
                d_table.p, kBankExperts, d_ref_ids.p, d_act_f16.p, d_current.p,
                d_row_scale.p, d_offsets.p, kBankExperts, kInF, kOutF, kRowBytes,
                runtime_stream);
            if (rc) throw std::runtime_error("current half2 rc=" + std::to_string(rc));
            cuda_check(cudaEventRecord(stop, runtime_stream), "current stop");
            cuda_check(cudaEventSynchronize(stop), "current sync");
            cuda_check(cudaEventElapsedTime(ms, start, stop), "current elapsed");
            host.resize(static_cast<std::size_t>(kBankExperts) * kOutF);
            cuda_check(cudaMemcpy(host.data(), d_current.p, output_bytes, cudaMemcpyDeviceToHost),
                       "copy current output");
        };
        auto run_external = [&](std::vector<float>& host, float* ms) {
            flush_l2(d_flush, d_sink, 256, runtime_stream);
            cuda_check(cudaMemsetAsync(d_external.p, 0xa5, output_bytes, runtime_stream), "clear external");
            cuda_check(cudaEventRecord(start, runtime_stream), "external start");
            launch_external(info.function, driver_stream, external_grid,
                            d_table, d_ref_ids, d_act_f16, d_external,
                            d_row_scale, d_offsets);
            cuda_check(cudaEventRecord(stop, runtime_stream), "external stop");
            cuda_check(cudaEventSynchronize(stop), "external sync");
            cuda_check(cudaEventElapsedTime(ms, start, stop), "external elapsed");
            host.resize(static_cast<std::size_t>(kBankExperts) * kOutF);
            cuda_check(cudaMemcpy(host.data(), d_external.p, output_bytes, cudaMemcpyDeviceToHost),
                       "copy external output");
        };
        std::vector<float> warm_current, warm_external;
        const unsigned long long warm_before =
            memra_moe_kq_gemm_sk_m1_half2_dispatches();
        float warm_current_ms = 0.0f, warm_external_ms = 0.0f;
        run_current(warm_current, &warm_current_ms);
        const unsigned long long warm_after =
            memra_moe_kq_gemm_sk_m1_half2_dispatches();
        if (warm_after - warm_before != 1)
            throw std::runtime_error("current half2 warmup enqueue counter delta was not 1");
        run_external(warm_external, &warm_external_ms);
        if (std::memcmp(warm_current.data(), warm_external.data(), output_bytes) != 0)
            throw std::runtime_error("warm current/external output mismatch");

        const unsigned long long timed_before =
            memra_moe_kq_gemm_sk_m1_half2_dispatches();
        for (int cycle = 0; cycle < 3; ++cycle) {
            std::vector<float> current_a1, external_b1, external_b2, current_a2;
            float current_ms1 = 0.0f, external_ms1 = 0.0f;
            float external_ms2 = 0.0f, current_ms2 = 0.0f;
            run_current(current_a1, &current_ms1);
            run_external(external_b1, &external_ms1);
            run_external(external_b2, &external_ms2);
            run_current(current_a2, &current_ms2);
            if (std::memcmp(current_a1.data(), external_b1.data(), output_bytes) != 0
                || std::memcmp(current_a1.data(), external_b2.data(), output_bytes) != 0
                || std::memcmp(current_a1.data(), current_a2.data(), output_bytes) != 0) {
                throw std::runtime_error("external pragma cubin or repeated current output differs");
            }
            std::printf("active=%d cycle=%d exact=PASS current_ms=(%.3f,%.3f) "
                        "external_ms=(%.3f,%.3f) grid_current=%d grid_external=%d\n",
                        active, cycle + 1, current_ms1, current_ms2,
                        external_ms1, external_ms2, current_grid, external_grid);
        }
        const unsigned long long timed_after =
            memra_moe_kq_gemm_sk_m1_half2_dispatches();
        if (timed_after - timed_before != 6)
            throw std::runtime_error("current half2 timed enqueue counter delta was not 6");
    }
    cuda_check(cudaEventDestroy(start), "destroy start");
    cuda_check(cudaEventDestroy(stop), "destroy stop");
    cuda_check(cudaStreamDestroy(runtime_stream), "destroy stream");
    cu_check(cuModuleUnload(info.module), "unload module");
    std::puts("PASS CUDA 13.3 pragma cubin exact-output/ABBA loader gate");
    return 0;
}

} // namespace dsv4_f16_compiler_gate

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s [--load-only] <cubin> | --pair <cuda13.3-base> <cuda13.3-pragma>\n", argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "--pair") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "--pair requires baseline and pragma cubin paths\n");
            return 2;
        }
        try {
            std::printf("pair arm=13.3-baseline compared against current CUDA13.1 engine\n");
            const int base_rc = dsv4_f16_compiler_gate::run(argv[2], false);
            if (base_rc) return base_rc;
            std::printf("pair arm=13.3-mma_throughput compared against current CUDA13.1 engine\n");
            return dsv4_f16_compiler_gate::run(argv[3], false);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL %s\n", e.what());
            return 1;
        }
    }
    const bool load_only = std::strcmp(argv[1], "--load-only") == 0;
    const char* cubin = load_only ? (argc > 2 ? argv[2] : nullptr) : argv[1];
    if (!cubin) {
        std::fprintf(stderr, "missing cubin path\n");
        return 2;
    }
    try {
        return dsv4_f16_compiler_gate::run(cubin, load_only);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
