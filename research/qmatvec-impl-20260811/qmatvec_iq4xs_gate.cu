// Arm-1 exactness and directional-timing harness for qmatvec_iq4_XS_dp4a.
//
// Build from the repository root with the same architecture and optimization
// class as the production qmatvec fatbin:
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 \
//     -o /tmp/qmatvec-iq4xs-gate \
//     research/qmatvec-impl-20260811/qmatvec_iq4xs_gate.cu
//
// `check FILE` writes deterministic raw f32 outputs for every observed Step
// shape, first aligned and then at base+4. Compile/run once before and once
// after the rewrite; byte-compare the two FILEs. `bench N` times the exact
// 315-launch semantic mix from the feasibility study with unique matrix bytes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../../crates/memra-engine/cu/qmatvec.cu"

#define CUDA_CHECK(expr) do {                                                   \
    cudaError_t err_ = (expr);                                                  \
    if (err_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(err_),   \
                     __FILE__, __LINE__);                                       \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

struct Shape {
    int out_f;
    int in_f;
    int launches;
    const char* label;
};

// The 11 unique (out_f, in_f) pairs and their combined two-device launch
// counts in one Step-3.7 token. Sum(launches) == 315.
static const Shape kShapes[] = {
    {8192,  4096, 12, "attention-q-swa"},
    {12288, 4096, 33, "attention-q-full"},
    {1024,  4096, 45, "attention-k"},
    {64,    4096, 12, "head-gate-swa"},
    {96,    4096, 33, "head-gate-full"},
    {4096,  8192, 12, "attention-out-swa"},
    {4096, 12288, 33, "attention-out-full"},
    {11264, 4096,  6, "dense-gate-up"},
    {4096, 11264,  3, "dense-down"},
    {1280,  4096, 84, "shared-gate-up"},
    {4096,  1280, 42, "shared-down"},
};

static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static std::vector<unsigned char> make_weights(const Shape& shape) {
    const std::size_t row_bytes = (std::size_t)(shape.in_f / 256) * 136;
    std::vector<unsigned char> weights((std::size_t)shape.out_f * row_bytes);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = (unsigned char)mix32((std::uint32_t)i ^
                                          (std::uint32_t)shape.in_f ^
                                          ((std::uint32_t)shape.out_f << 12));
    }
    // Keep every super-block scale finite and normal while retaining varied
    // signs, exponents, and mantissas. Other header and quant bytes stay noisy.
    for (int row = 0; row < shape.out_f; ++row) {
        for (std::size_t off = 0; off < row_bytes; off += 136) {
            const std::uint32_t r = mix32((std::uint32_t)row ^
                                           (std::uint32_t)off ^
                                           (std::uint32_t)shape.in_f);
            const std::uint16_t h = (std::uint16_t)(((r >> 16) & 0x8000u) |
                ((10u + ((r >> 10) & 7u)) << 10) | (r & 0x03ffu));
            const std::size_t base = (std::size_t)row * row_bytes + off;
            weights[base] = (unsigned char)(h & 0xffu);
            weights[base + 1] = (unsigned char)(h >> 8);
        }
    }
    return weights;
}

static std::uint64_t hash_bytes(const void* data, std::size_t n) {
    const unsigned char* p = (const unsigned char*)data;
    std::uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<float> run_shape(const Shape& shape, bool misaligned) {
    const int nsb = shape.in_f / 32;
    const std::size_t row_bytes = (std::size_t)(shape.in_f / 256) * 136;
    const std::size_t weight_bytes = (std::size_t)shape.out_f * row_bytes;
    std::vector<unsigned char> weights = make_weights(shape);
    std::vector<signed char> aq(shape.in_f);
    std::vector<float> ad(nsb);
    for (int i = 0; i < shape.in_f; ++i) {
        aq[i] = (signed char)((int)(mix32((std::uint32_t)i ^ 0xa511e9b3u) % 255u) - 127);
    }
    for (int g = 0; g < nsb; ++g) {
        ad[g] = (float)(1 + (mix32((std::uint32_t)g ^ 0x63d83595u) % 31u)) / 4096.0f;
    }

    unsigned char* allocation = nullptr;
    signed char* aq_d = nullptr;
    float* ad_d = nullptr;
    float* y_d = nullptr;
    CUDA_CHECK(cudaMalloc(&allocation, weight_bytes + 8));
    unsigned char* w_d = allocation + (misaligned ? 4 : 0);
    CUDA_CHECK(cudaMalloc(&aq_d, aq.size()));
    CUDA_CHECK(cudaMalloc(&ad_d, ad.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, (std::size_t)shape.out_f * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(w_d, weights.data(), weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(aq_d, aq.data(), aq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ad_d, ad.data(), ad.size() * sizeof(float), cudaMemcpyHostToDevice));

    qmatvec_iq4_XS_dp4a<<<dim3(shape.out_f, 1, 1), dim3(128, 1, 1)>>>(
        w_d, aq_d, ad_d, y_d, shape.in_f, shape.out_f, 1, (long)row_bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> y(shape.out_f);
    CUDA_CHECK(cudaMemcpy(y.data(), y_d, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(ad_d));
    CUDA_CHECK(cudaFree(aq_d));
    CUDA_CHECK(cudaFree(allocation));
    return y;
}

static int check_outputs(const char* output_path) {
    std::FILE* output = std::fopen(output_path, "wb");
    if (output == nullptr) {
        std::perror(output_path);
        return 1;
    }
    int failures = 0;
    for (const Shape& shape : kShapes) {
        const std::vector<float> aligned = run_shape(shape, false);
        const std::vector<float> fallback = run_shape(shape, true);
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < aligned.size(); ++i) {
            std::uint32_t a;
            std::uint32_t b;
            std::memcpy(&a, &aligned[i], sizeof(a));
            std::memcpy(&b, &fallback[i], sizeof(b));
            mismatches += a != b;
        }
        const std::uint64_t aligned_hash = hash_bytes(aligned.data(), aligned.size() * sizeof(float));
        const std::uint64_t fallback_hash = hash_bytes(fallback.data(), fallback.size() * sizeof(float));
        std::printf("shape out=%5d in=%5d %-18s aligned=%016llx fallback=%016llx mismatches=%zu %s\n",
                    shape.out_f, shape.in_f, shape.label,
                    (unsigned long long)aligned_hash, (unsigned long long)fallback_hash,
                    mismatches, mismatches == 0 ? "OK" : "FAIL");
        const std::uint32_t meta[] = {(std::uint32_t)shape.out_f, (std::uint32_t)shape.in_f, 0};
        std::fwrite(meta, sizeof(meta), 1, output);
        std::fwrite(aligned.data(), sizeof(float), aligned.size(), output);
        const std::uint32_t fallback_meta[] = {
            (std::uint32_t)shape.out_f, (std::uint32_t)shape.in_f, 4};
        std::fwrite(fallback_meta, sizeof(fallback_meta), 1, output);
        std::fwrite(fallback.data(), sizeof(float), fallback.size(), output);
        failures += mismatches != 0;
    }
    if (std::fclose(output) != 0) {
        std::perror(output_path);
        return 1;
    }
    std::printf("exactness layouts: %s\n", failures == 0 ? "ALL GREEN" : "FAIL");
    return failures == 0 ? 0 : 1;
}

struct BenchAllocation {
    const Shape* shape;
    unsigned char* weights;
    std::size_t matrix_bytes;
    std::size_t row_bytes;
};

static int bench_semantic_mix(int repetitions) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("device=%s sms=%d free_mib=%.1f repetitions=%d\n", prop.name,
                prop.multiProcessorCount, free_bytes / 1048576.0, repetitions);

    std::vector<BenchAllocation> allocations;
    std::size_t weight_bytes_per_sweep = 0;
    std::size_t logical_bytes_per_sweep = 0;
    int launches_per_sweep = 0;
    for (const Shape& shape : kShapes) {
        const std::size_t row_bytes = (std::size_t)(shape.in_f / 256) * 136;
        const std::size_t matrix_bytes = (std::size_t)shape.out_f * row_bytes;
        unsigned char* weights = nullptr;
        CUDA_CHECK(cudaMalloc(&weights, matrix_bytes * shape.launches));
        CUDA_CHECK(cudaMemset(weights, 1, matrix_bytes * shape.launches));
        allocations.push_back({&shape, weights, matrix_bytes, row_bytes});
        weight_bytes_per_sweep += matrix_bytes * shape.launches;
        logical_bytes_per_sweep += (matrix_bytes + (std::size_t)shape.in_f +
            (std::size_t)(shape.in_f / 32) * sizeof(float) +
            (std::size_t)shape.out_f * sizeof(float)) * shape.launches;
        launches_per_sweep += shape.launches;
    }

    signed char* aq = nullptr;
    float* ad = nullptr;
    float* y = nullptr;
    CUDA_CHECK(cudaMalloc(&aq, 12288));
    CUDA_CHECK(cudaMalloc(&ad, (12288 / 32) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, 12288 * sizeof(float)));
    CUDA_CHECK(cudaMemset(aq, 1, 12288));
    std::vector<float> ad_h(12288 / 32, 1.0f / 4096.0f);
    CUDA_CHECK(cudaMemcpy(ad, ad_h.data(), ad_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto launch_sweep = [&]() {
        for (const BenchAllocation& allocation : allocations) {
            const Shape& shape = *allocation.shape;
            for (int i = 0; i < shape.launches; ++i) {
                const unsigned char* weights = allocation.weights +
                    (std::size_t)i * allocation.matrix_bytes;
                qmatvec_iq4_XS_dp4a<<<dim3(shape.out_f, 1, 1), dim3(128, 1, 1)>>>(
                    weights, aq, ad, y, shape.in_f, shape.out_f, 1,
                    (long)allocation.row_bytes);
            }
        }
    };

    launch_sweep();
    launch_sweep();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int rep = 0; rep < repetitions; ++rep) {
        launch_sweep();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double token_ms = elapsed_ms / repetitions;
    const double logical_gbs = logical_bytes_per_sweep / (token_ms * 1.0e6);
    std::printf("semantic_mix launches=%d weight_bytes=%zu logical_bytes=%zu token_ms=%.6f logical_gbs=%.3f\n",
                launches_per_sweep, weight_bytes_per_sweep, logical_bytes_per_sweep,
                token_ms, logical_gbs);

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(ad));
    CUDA_CHECK(cudaFree(aq));
    for (const BenchAllocation& allocation : allocations) {
        CUDA_CHECK(cudaFree(allocation.weights));
    }
    return launches_per_sweep == 315 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "check") == 0) {
        return check_outputs(argv[2]);
    }
    if (argc == 3 && std::strcmp(argv[1], "bench") == 0) {
        const int repetitions = std::atoi(argv[2]);
        if (repetitions <= 0) {
            std::fprintf(stderr, "bench repetitions must be positive\n");
            return 2;
        }
        return bench_semantic_mix(repetitions);
    }
    std::fprintf(stderr, "usage: %s check OUTPUT.bin | bench REPETITIONS\n", argv[0]);
    return 2;
}
