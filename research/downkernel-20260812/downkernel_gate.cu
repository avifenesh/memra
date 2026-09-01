// Exact-geometry gate and microbenchmark for Step-3.7 B=1 expert-down.
//
// Baseline-only NCU build (before the candidate symbol exists):
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 \
//     -o /tmp/downkernel-gate research/downkernel-20260812/downkernel_gate.cu
//   /tmp/downkernel-gate ncu-baseline
//
// Candidate build (after moe_down8_fma_dev_q8_rows_w8 is present):
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 \
//     -DDOWNKERNEL_CANDIDATE_AVAILABLE=1 -o /tmp/downkernel-gate \
//     research/downkernel-20260812/downkernel_gate.cu
//   /tmp/downkernel-gate check BASE.bin CANDIDATE.bin
//   /tmp/downkernel-gate bench REPETITIONS
//
// The benchmark walks 40 physically distinct expert-layer banks per synthetic token,
// matching the two-device Step decode semantic count while running serially on one GPU.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../../crates/memra-engine/cu/qmatvec.cu"

#define CUDA_CHECK(expr) do {                                                    \
    cudaError_t err_ = (expr);                                                   \
    if (err_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(err_),    \
                     __FILE__, __LINE__);                                        \
        std::exit(1);                                                            \
    }                                                                            \
} while (0)

static constexpr int kIn = 1280;
static constexpr int kOut = 4096;
static constexpr int kUsed = 8;
static constexpr int kExperts = 288;
static constexpr int kQtypeIq4Xs = 5;
static constexpr int kRowBytes = (kIn / 256) * 136;
static constexpr int kGroups = kIn / 32;
static constexpr int kLayers = 40;
static constexpr int kSelected[kUsed] = {3, 17, 41, 79, 113, 157, 211, 283};

static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static std::uint64_t hash_bytes(const void* data, std::size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<unsigned char> make_weight_bank() {
    const std::size_t matrix_bytes = static_cast<std::size_t>(kOut) * kRowBytes;
    std::vector<unsigned char> weights(kUsed * matrix_bytes);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<unsigned char>(mix32(static_cast<std::uint32_t>(i) ^ 0xd04e1234u));
    }
    // IQ4_XS stores one fp16 super-block scale in the first two bytes of each
    // 136-byte block. Keep it finite and varied; all other metadata/quant bytes
    // remain deterministic noise.
    for (int slot = 0; slot < kUsed; ++slot) {
        for (int row = 0; row < kOut; ++row) {
            for (int block = 0; block < kIn / 256; ++block) {
                const std::uint32_t r = mix32(static_cast<std::uint32_t>(
                    slot * 0x100000 + row * 17 + block));
                const std::uint16_t h = static_cast<std::uint16_t>(
                    ((r >> 16) & 0x8000u) | ((10u + ((r >> 10) & 7u)) << 10) |
                    (r & 0x03ffu));
                const std::size_t off = static_cast<std::size_t>(slot) * matrix_bytes +
                    static_cast<std::size_t>(row) * kRowBytes + block * 136;
                weights[off] = static_cast<unsigned char>(h & 0xffu);
                weights[off + 1] = static_cast<unsigned char>(h >> 8);
            }
        }
    }
    return weights;
}

struct SharedInputs {
    int* sel = nullptr;
    float* route = nullptr;
    signed char* aq = nullptr;
    float* ad = nullptr;
    float* dst = nullptr;
};

static SharedInputs make_shared_inputs() {
    SharedInputs inputs;
    std::vector<int> sel(kSelected, kSelected + kUsed);
    std::vector<float> route(kUsed);
    std::vector<signed char> aq(kUsed * kIn);
    std::vector<float> ad(kUsed * kGroups);
    for (int j = 0; j < kUsed; ++j) {
        route[j] = static_cast<float>(j + 1) / 36.0f;
    }
    for (std::size_t i = 0; i < aq.size(); ++i) {
        aq[i] = static_cast<signed char>(static_cast<int>(mix32(
            static_cast<std::uint32_t>(i) ^ 0xa511e9b3u) % 255u) - 127);
    }
    for (std::size_t i = 0; i < ad.size(); ++i) {
        ad[i] = static_cast<float>(1 + mix32(static_cast<std::uint32_t>(i) ^
            0x63d83595u) % 31u) / 4096.0f;
    }
    CUDA_CHECK(cudaMalloc(&inputs.sel, sel.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&inputs.route, route.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&inputs.aq, aq.size()));
    CUDA_CHECK(cudaMalloc(&inputs.ad, ad.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&inputs.dst, kOut * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(inputs.sel, sel.data(), sel.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputs.route, route.data(), route.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputs.aq, aq.data(), aq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(inputs.ad, ad.data(), ad.size() * sizeof(float), cudaMemcpyHostToDevice));
    return inputs;
}

static void free_shared_inputs(SharedInputs& inputs) {
    CUDA_CHECK(cudaFree(inputs.dst));
    CUDA_CHECK(cudaFree(inputs.ad));
    CUDA_CHECK(cudaFree(inputs.aq));
    CUDA_CHECK(cudaFree(inputs.route));
    CUDA_CHECK(cudaFree(inputs.sel));
}

struct LayerBank {
    unsigned char* weights = nullptr;
    unsigned long long* table = nullptr;
};

static LayerBank make_layer_bank(const std::vector<unsigned char>& host_weights) {
    LayerBank layer;
    const std::size_t matrix_bytes = static_cast<std::size_t>(kOut) * kRowBytes;
    CUDA_CHECK(cudaMalloc(&layer.weights, host_weights.size()));
    CUDA_CHECK(cudaMemcpy(layer.weights, host_weights.data(), host_weights.size(),
                          cudaMemcpyHostToDevice));
    std::vector<unsigned long long> table(3 * kExperts, 0);
    for (int j = 0; j < kUsed; ++j) {
        table[2 * kExperts + kSelected[j]] = reinterpret_cast<unsigned long long>(
            layer.weights + static_cast<std::size_t>(j) * matrix_bytes);
    }
    CUDA_CHECK(cudaMalloc(&layer.table, table.size() * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(layer.table, table.data(), table.size() * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    return layer;
}

static void free_layer_bank(LayerBank& layer) {
    CUDA_CHECK(cudaFree(layer.table));
    CUDA_CHECK(cudaFree(layer.weights));
}

static void launch_baseline(const LayerBank& layer, const SharedInputs& inputs) {
    moe_down8_fma_dev_q8_rows_g<<<dim3(kOut, 1, 1), dim3(32, 1, 1)>>>(
        layer.table, inputs.sel, inputs.route, inputs.aq, inputs.ad, inputs.dst,
        kIn, kOut, kUsed, kExperts, kQtypeIq4Xs, kRowBytes);
}

#ifdef DOWNKERNEL_CANDIDATE_AVAILABLE
static void launch_candidate(const LayerBank& layer, const SharedInputs& inputs) {
    moe_down8_fma_dev_q8_rows_w8<<<dim3(kOut, 1, 1), dim3(32, kUsed, 1)>>>(
        layer.table, inputs.sel, inputs.route, inputs.aq, inputs.ad, inputs.dst,
        kIn, kOut, kUsed, kExperts, kQtypeIq4Xs, kRowBytes);
}
#endif

static int ncu_baseline() {
    const std::vector<unsigned char> weights = make_weight_bank();
    LayerBank layer = make_layer_bank(weights);
    SharedInputs inputs = make_shared_inputs();
    for (int i = 0; i < 3; ++i) launch_baseline(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    launch_baseline(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    free_shared_inputs(inputs);
    free_layer_bank(layer);
    return 0;
}

#ifdef DOWNKERNEL_CANDIDATE_AVAILABLE
static std::vector<float> read_output(const SharedInputs& inputs) {
    std::vector<float> output(kOut);
    CUDA_CHECK(cudaMemcpy(output.data(), inputs.dst, output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return output;
}

static bool write_output(const char* path, const std::vector<float>& output) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::perror(path);
        return false;
    }
    const bool ok = std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size()
        && std::fclose(file) == 0;
    if (!ok) std::perror(path);
    return ok;
}

static int check_outputs(const char* baseline_path, const char* candidate_path) {
    const std::vector<unsigned char> weights = make_weight_bank();
    LayerBank layer = make_layer_bank(weights);
    SharedInputs inputs = make_shared_inputs();
    launch_baseline(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> baseline = read_output(inputs);
    launch_candidate(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> candidate = read_output(inputs);

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        mismatches += std::memcmp(&baseline[i], &candidate[i], sizeof(float)) != 0;
    }
    const std::uint64_t base_hash = hash_bytes(baseline.data(), baseline.size() * sizeof(float));
    const std::uint64_t cand_hash = hash_bytes(candidate.data(), candidate.size() * sizeof(float));
    std::printf("shape in=%d out=%d used=%d experts=%d rb=%d baseline=%016llx candidate=%016llx mismatches=%zu %s\n",
                kIn, kOut, kUsed, kExperts, kRowBytes,
                static_cast<unsigned long long>(base_hash),
                static_cast<unsigned long long>(cand_hash), mismatches,
                mismatches == 0 ? "BIT-IDENTICAL" : "FAIL");
    const bool wrote = write_output(baseline_path, baseline) &&
        write_output(candidate_path, candidate);
    free_shared_inputs(inputs);
    free_layer_bank(layer);
    return mismatches == 0 && wrote ? 0 : 1;
}

enum class Arm { Baseline, Candidate };

static double time_arm(Arm arm, const std::vector<LayerBank>& layers,
                       const SharedInputs& inputs, int repetitions) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < repetitions; ++rep) {
        for (const LayerBank& layer : layers) {
            if (arm == Arm::Baseline) launch_baseline(layer, inputs);
            else launch_candidate(layer, inputs);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    return elapsed_ms / repetitions;
}

static int bench(int repetitions) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("device=%s sms=%d free_mib=%.1f repetitions=%d layers=%d\n",
                prop.name, prop.multiProcessorCount, free_bytes / 1048576.0,
                repetitions, kLayers);
    const std::vector<unsigned char> weights = make_weight_bank();
    std::vector<LayerBank> layers;
    layers.reserve(kLayers);
    for (int i = 0; i < kLayers; ++i) layers.push_back(make_layer_bank(weights));
    SharedInputs inputs = make_shared_inputs();

    // Warm both arms over the complete physical working set before sampling.
    time_arm(Arm::Baseline, layers, inputs, 2);
    time_arm(Arm::Candidate, layers, inputs, 2);
    const Arm schedule[] = {
        Arm::Baseline, Arm::Candidate, Arm::Candidate, Arm::Baseline,
        Arm::Baseline, Arm::Candidate, Arm::Candidate, Arm::Baseline,
        Arm::Baseline, Arm::Candidate, Arm::Candidate, Arm::Baseline,
        Arm::Baseline, Arm::Candidate, Arm::Candidate, Arm::Baseline,
    };
    const std::size_t bytes_per_layer = static_cast<std::size_t>(kUsed) * kOut * kRowBytes;
    const std::size_t bytes_per_token = kLayers * bytes_per_layer;
    int base_n = 0;
    int cand_n = 0;
    for (int position = 0; position < 16; ++position) {
        const Arm arm = schedule[position];
        const double token_ms = time_arm(arm, layers, inputs, repetitions);
        const double logical_gbs = bytes_per_token / (token_ms * 1.0e6);
        const int sample = arm == Arm::Baseline ? ++base_n : ++cand_n;
        std::printf("sample=%d position=%d arm=%s repetitions=%d token_ms=%.6f logical_weight_gbs=%.3f\n",
                    sample, position + 1,
                    arm == Arm::Baseline ? "baseline" : "candidate",
                    repetitions, token_ms, logical_gbs);
    }
    std::printf("semantic_mix launches_per_token=%d weight_bytes_per_token=%zu N=%d/arm schedule=ABBAx4\n",
                kLayers, bytes_per_token, base_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    free_shared_inputs(inputs);
    for (LayerBank& layer : layers) free_layer_bank(layer);
    return base_n == 8 && cand_n == 8 ? 0 : 1;
}

static int ncu_candidate() {
    const std::vector<unsigned char> weights = make_weight_bank();
    LayerBank layer = make_layer_bank(weights);
    SharedInputs inputs = make_shared_inputs();
    for (int i = 0; i < 3; ++i) launch_candidate(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    launch_candidate(layer, inputs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    free_shared_inputs(inputs);
    free_layer_bank(layer);
    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "ncu-baseline") == 0) return ncu_baseline();
#ifdef DOWNKERNEL_CANDIDATE_AVAILABLE
    if (argc == 2 && std::strcmp(argv[1], "ncu-candidate") == 0) return ncu_candidate();
    if (argc == 4 && std::strcmp(argv[1], "check") == 0) {
        return check_outputs(argv[2], argv[3]);
    }
    if (argc == 3 && std::strcmp(argv[1], "bench") == 0) {
        const int repetitions = std::atoi(argv[2]);
        if (repetitions <= 0) {
            std::fprintf(stderr, "bench repetitions must be positive\n");
            return 2;
        }
        return bench(repetitions);
    }
#endif
    std::fprintf(stderr, "usage: %s ncu-baseline"
#ifdef DOWNKERNEL_CANDIDATE_AVAILABLE
                 " | ncu-candidate | check BASE.bin CANDIDATE.bin | bench REPETITIONS"
#endif
                 "\n", argv[0]);
    return 2;
}
