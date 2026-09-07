// Standalone exact top-k selector gate for the DSV4 indexer.
//
// This tool includes dsv4_gpu.cu and calls its actual memra_dsv4_topk_idx_numeric wrapper/kernel
// as the 512-thread shared-memory bitonic control, plus a three-byte MSD radix-cut candidate.
// Both construct the exact numeric-zero-normalized key:
//   (~orderable_f32(score) << 32) | original_index
// The production wrapper has no finite-score check. The nonfinite corpus below therefore calls
// BOTH actual wrappers and compares their raw-key output against an independent host key oracle;
// no host-side finite filter is used to hide a policy mismatch.
//
// Compile only after the owning lane authorizes a CUDA window:
//   nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/dsv4-index-topk-gate \
//     tools/dsv4-index-topk-gate.cu
//
// Timing labels are device-kernel event timings only. Allocation and H2D/D2H copies are excluded;
// `cold_kernel_ms` is the first launch after setup and `warm_kernel_ms_per_launch` is the mean of
// the repeated launches. No timing result is a serving claim.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "../crates/memra-engine/cu/dsv4_gpu.cu"

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t _e = (call);                                                       \
        if (_e != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(_e));                                      \
            std::exit(2);                                                              \
        }                                                                               \
    } while (0)

static constexpr int DEFAULT_ABBA_CYCLES = 3;

static void check_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}

static uint64_t host_numeric_key(float value, unsigned index) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7FFFFFFFu) == 0) bits = 0;
    const uint32_t orderable = bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
    return (uint64_t)(~orderable) << 32 | index;
}

static std::vector<int> host_oracle(const std::vector<float>& scores, int k, int win) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return host_numeric_key(scores[a], (unsigned)a)
            < host_numeric_key(scores[b], (unsigned)b);
    });
    order.resize(k);
    for (int& index : order) index += win;
    return order;
}

static float f32_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static std::vector<float> make_scores(int n, const std::string& kind) {
    std::vector<float> scores(n);
    for (int i = 0; i < n; ++i) {
        if (kind == "all-equal") {
            scores[i] = 3.25f;
        } else if (kind == "signed-zero") {
            scores[i] = (i & 1) ? -0.0f : 0.0f;
        } else if (kind == "ties") {
            scores[i] = ((i * 17 + i / 13) % 19 - 9) * 0.25f;
            if (i % 23 == 0) scores[i] = (i & 1) ? -0.0f : 0.0f;
        } else {
            const unsigned h = (unsigned)i * 2654435761u + 1013904223u;
            scores[i] = ((int)(h % 20001u) - 10000) / 317.0f;
        }
    }
    return scores;
}

static float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0.0f;
    check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    return milliseconds;
}

static void run_valid_case(const char* name, const char* data_kind,
                           const std::vector<float>& scores, int k, int win, int abba_cycles) {
    const int n = (int)scores.size();
    float* d_scores = nullptr;
    int* d_baseline = nullptr;
    int* d_candidate = nullptr;
    unsigned long long* d_keys = nullptr;
    unsigned long long* d_candidates = nullptr;
    check_cuda(cudaMalloc(&d_scores, (size_t)n * sizeof(float)), "cudaMalloc scores");
    check_cuda(cudaMalloc(&d_baseline, (size_t)k * sizeof(int)), "cudaMalloc baseline");
    check_cuda(cudaMalloc(&d_candidate, (size_t)k * sizeof(int)), "cudaMalloc candidate");
    check_cuda(cudaMalloc(&d_keys, (size_t)n * sizeof(unsigned long long)), "cudaMalloc keys");
    check_cuda(cudaMalloc(&d_candidates, (size_t)n * sizeof(unsigned long long)),
               "cudaMalloc candidates");
    check_cuda(cudaMemcpy(d_scores, scores.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy scores");

    const auto launch_baseline = [&]() {
        const int rc = memra_dsv4_topk_idx_numeric(d_scores, n, k, win, d_baseline, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "baseline current wrapper refused %s rc=%d\n", name, rc);
            std::exit(1);
        }
    };
    const auto launch_candidate = [&]() {
        const int rc = memra_dsv4_topk_idx_radix_m1(
            d_scores, n, k, win, d_candidate, d_keys, d_candidates, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "candidate production wrapper refused %s rc=%d\n", name, rc);
            std::exit(1);
        }
    };

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaDeviceSynchronize(), "setup synchronize");
    check_cuda(cudaEventRecord(start), "baseline first start");
    launch_baseline();
    check_cuda(cudaEventRecord(stop), "baseline first stop");
    check_cuda(cudaEventSynchronize(stop), "baseline first sync");
    const float baseline_first = event_ms(start, stop);
    check_cuda(cudaEventRecord(start), "candidate first start");
    launch_candidate();
    check_cuda(cudaEventRecord(stop), "candidate first stop");
    check_cuda(cudaEventSynchronize(stop), "candidate first sync");
    const float candidate_first = event_ms(start, stop);

    struct EventPair {
        cudaEvent_t start;
        cudaEvent_t stop;
        bool baseline;
    };
    std::vector<EventPair> abba;
    abba.reserve((size_t)abba_cycles * 4);
    for (int cycle = 0; cycle < abba_cycles; ++cycle) {
        // A B B A: the same arm is never allowed to own one contiguous warm block.
        for (bool baseline : {true, false, false, true}) {
            EventPair pair{nullptr, nullptr, baseline};
            check_cuda(cudaEventCreate(&pair.start), "cudaEventCreate ABBA start");
            check_cuda(cudaEventCreate(&pair.stop), "cudaEventCreate ABBA stop");
            check_cuda(cudaEventRecord(pair.start), "cudaEventRecord ABBA start");
            if (baseline) launch_baseline();
            else launch_candidate();
            check_cuda(cudaEventRecord(pair.stop), "cudaEventRecord ABBA stop");
            abba.push_back(pair);
        }
    }
    check_cuda(cudaEventSynchronize(abba.back().stop), "ABBA sync");
    float baseline_warm_sum = 0.0f;
    float candidate_warm_sum = 0.0f;
    int baseline_warm_count = 0;
    int candidate_warm_count = 0;
    for (const EventPair& pair : abba) {
        const float milliseconds = event_ms(pair.start, pair.stop);
        if (pair.baseline) {
            baseline_warm_sum += milliseconds;
            baseline_warm_count++;
        } else {
            candidate_warm_sum += milliseconds;
            candidate_warm_count++;
        }
        check_cuda(cudaEventDestroy(pair.start), "cudaEventDestroy ABBA start");
        check_cuda(cudaEventDestroy(pair.stop), "cudaEventDestroy ABBA stop");
    }
    const float baseline_warm = baseline_warm_sum / baseline_warm_count;
    const float candidate_warm = candidate_warm_sum / candidate_warm_count;

    std::vector<int> baseline(k), candidate(k), expected = host_oracle(scores, k, win);
    check_cuda(cudaMemcpy(baseline.data(), d_baseline, (size_t)k * sizeof(int),
                          cudaMemcpyDeviceToHost), "cudaMemcpy baseline");
    check_cuda(cudaMemcpy(candidate.data(), d_candidate, (size_t)k * sizeof(int),
                          cudaMemcpyDeviceToHost), "cudaMemcpy candidate");
    if (baseline != expected || candidate != expected || baseline != candidate) {
        std::fprintf(stderr, "FAIL %s exact ordering\n", name);
        std::exit(1);
    }
    std::printf(
        "CASE name=%s data_kind=%s n=%d k=%d win=%d exact=true baseline_first_launch_kernel_ms=%.6f "
        "candidate_first_launch_kernel_ms=%.6f baseline_warm_kernel_ms_per_launch=%.6f "
        "candidate_warm_kernel_ms_per_launch=%.6f abba_cycles=%d timing_scope=device_kernel_only\n",
        name, data_kind, n, k, win, baseline_first, candidate_first, baseline_warm,
        candidate_warm, abba_cycles);
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check_cuda(cudaFree(d_scores), "cudaFree scores");
    check_cuda(cudaFree(d_baseline), "cudaFree baseline");
    check_cuda(cudaFree(d_candidate), "cudaFree candidate");
    check_cuda(cudaFree(d_keys), "cudaFree keys");
    check_cuda(cudaFree(d_candidates), "cudaFree candidates");
}

static void run_nonfinite_case(const char* name, const std::vector<float>& scores, int win) {
    const int n = (int)scores.size();
    const int k = 512;
    float* d_scores = nullptr;
    int* d_baseline = nullptr;
    int* d_candidate = nullptr;
    unsigned long long* d_keys = nullptr;
    unsigned long long* d_candidates = nullptr;
    check_cuda(cudaMalloc(&d_scores, (size_t)n * sizeof(float)), "cudaMalloc nonfinite scores");
    check_cuda(cudaMalloc(&d_baseline, (size_t)k * sizeof(int)), "cudaMalloc nonfinite baseline");
    check_cuda(cudaMalloc(&d_candidate, (size_t)k * sizeof(int)), "cudaMalloc nonfinite candidate");
    check_cuda(cudaMalloc(&d_keys, (size_t)n * sizeof(unsigned long long)),
               "cudaMalloc nonfinite keys");
    check_cuda(cudaMalloc(&d_candidates, (size_t)n * sizeof(unsigned long long)),
               "cudaMalloc nonfinite candidates");
    check_cuda(cudaMemcpy(d_scores, scores.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy nonfinite scores");
    const int baseline_rc = memra_dsv4_topk_idx_numeric(
        d_scores, n, k, win, d_baseline, nullptr);
    const int candidate_rc = memra_dsv4_topk_idx_radix_m1(
        d_scores, n, k, win, d_candidate, d_keys, d_candidates, nullptr);
    if (baseline_rc != 0 || candidate_rc != 0) {
        std::fprintf(stderr, "FAIL nonfinite raw-key case %s baseline=%d candidate=%d\n",
                     name, baseline_rc, candidate_rc);
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "nonfinite synchronize");
    std::vector<int> baseline(k), candidate(k), expected = host_oracle(scores, k, win);
    check_cuda(cudaMemcpy(baseline.data(), d_baseline, (size_t)k * sizeof(int),
                          cudaMemcpyDeviceToHost), "cudaMemcpy nonfinite baseline");
    check_cuda(cudaMemcpy(candidate.data(), d_candidate, (size_t)k * sizeof(int),
                          cudaMemcpyDeviceToHost), "cudaMemcpy nonfinite candidate");
    if (baseline != expected || candidate != expected || baseline != candidate) {
        std::fprintf(stderr, "FAIL nonfinite raw-key ordering %s\n", name);
        std::exit(1);
    }
    std::printf("CASE name=%s data_kind=%s n=%d k=%d win=%d exact=true "
                "production_nonfinite_policy=raw-key host_finite_filter=false\n",
                name, name, n, k, win);
    check_cuda(cudaFree(d_scores), "cudaFree nonfinite scores");
    check_cuda(cudaFree(d_baseline), "cudaFree nonfinite baseline");
    check_cuda(cudaFree(d_candidate), "cudaFree nonfinite candidate");
    check_cuda(cudaFree(d_keys), "cudaFree nonfinite keys");
    check_cuda(cudaFree(d_candidates), "cudaFree nonfinite candidates");
}

static void run_candidate_refusal(const char* name, int n, int win) {
    const int k = 512;
    const auto scores = make_scores(n, "random");
    float* d_scores = nullptr;
    int* d_out = nullptr;
    unsigned long long* d_keys = nullptr;
    unsigned long long* d_candidates = nullptr;
    check_cuda(cudaMalloc(&d_scores, (size_t)n * sizeof(float)), "cudaMalloc refusal scores");
    check_cuda(cudaMalloc(&d_out, (size_t)k * sizeof(int)), "cudaMalloc refusal output");
    check_cuda(cudaMalloc(&d_keys, (size_t)n * sizeof(unsigned long long)),
               "cudaMalloc refusal keys");
    check_cuda(cudaMalloc(&d_candidates, (size_t)n * sizeof(unsigned long long)),
               "cudaMalloc refusal candidates");
    check_cuda(cudaMemcpy(d_scores, scores.data(), (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice), "cudaMemcpy refusal scores");
    const int rc = memra_dsv4_topk_idx_radix_m1(
        d_scores, n, k, win, d_out, d_keys, d_candidates, nullptr);
    if (rc != 40008) {
        std::fprintf(stderr, "FAIL candidate boundary %s rc=%d\n", name, rc);
        std::exit(1);
    }
    std::printf("CASE name=%s data_kind=random n=%d k=%d win=%d valid_buffers=true "
                "refused=true rc=%d\n", name, n, k, win, rc);
    check_cuda(cudaFree(d_scores), "cudaFree refusal scores");
    check_cuda(cudaFree(d_out), "cudaFree refusal output");
    check_cuda(cudaFree(d_keys), "cudaFree refusal keys");
    check_cuda(cudaFree(d_candidates), "cudaFree refusal candidates");
}

int main(int argc, char** argv) {
    int abba_cycles = DEFAULT_ABBA_CYCLES;
    if (argc == 3 && std::string(argv[1]) == "--repeats") {
        abba_cycles = std::atoi(argv[2]);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: dsv4-index-topk-gate [--repeats N]\n");
        return 2;
    }
    if (abba_cycles < 1) {
        std::fprintf(stderr, "--repeats/ABBA cycles must be positive\n");
        return 2;
    }

    std::printf("CONTROL current_numeric_wrapper_max_nb=4096 nonfinite_check=false\n");
    const int win = 8192;
    run_valid_case("finite-random-n2048", "random", make_scores(2048, "random"), 512, win,
                   abba_cycles);
    run_valid_case("compressed-8224-n2056", "random", make_scores(2056, "random"), 512, win,
                   abba_cycles);
    run_valid_case("compressed-8256-n2064", "ties", make_scores(2064, "ties"), 512, win,
                   abba_cycles);
    run_valid_case("compressed-8256-signed-zero-n2064", "signed-zero", make_scores(2064, "signed-zero"),
                   512, win, abba_cycles);
    run_valid_case("compressed-8256-all-equal-n2064", "all-equal", make_scores(2064, "all-equal"),
                   512, win, abba_cycles);
    run_valid_case("compressed-8448-n2112", "random", make_scores(2112, "random"), 512, win,
                   abba_cycles);
    run_valid_case("padded-all-equal-n4096", "all-equal", make_scores(4096, "all-equal"), 512,
                   win, abba_cycles);
    run_valid_case("finite-random-n4096", "random", make_scores(4096, "random"), 512, win,
                   abba_cycles);
    // Real non-null buffers prove the wrapper's max-N contract rather than its null-pointer guard.
    const auto boundary_scores = make_scores(4097, "random");
    float* d_boundary_score = nullptr;
    int* d_boundary_idx = nullptr;
    check_cuda(cudaMalloc(&d_boundary_score, boundary_scores.size() * sizeof(float)),
               "cudaMalloc boundary score");
    check_cuda(cudaMalloc(&d_boundary_idx, 512 * sizeof(int)), "cudaMalloc boundary index");
    check_cuda(cudaMemcpy(d_boundary_score, boundary_scores.data(),
                          boundary_scores.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy boundary score");
    const int boundary_rc = memra_dsv4_topk_idx_numeric(
        d_boundary_score, 4097, 512, win, d_boundary_idx, nullptr);
    check_cuda(cudaFree(d_boundary_score), "cudaFree boundary score");
    check_cuda(cudaFree(d_boundary_idx), "cudaFree boundary index");
    if (boundary_rc != 40008) {
        std::fprintf(stderr, "FAIL current wrapper max-n boundary rc=%d\n", boundary_rc);
        return 1;
    }
    std::printf("CASE name=current-wrapper-refusal-n4097 data_kind=random n=4097 k=512 "
                "valid_buffers=true refused=true rc=%d\n", boundary_rc);

    run_candidate_refusal("candidate-refusal-n2047", 2047, win);
    run_candidate_refusal("candidate-refusal-n4097", 4097, win);

    auto nan_positive = make_scores(2048, "random");
    nan_positive[17] = f32_bits(0x7FC00001u);
    run_nonfinite_case("nonfinite-nan-positive-payload", nan_positive, win);
    auto nan_negative = make_scores(2048, "random");
    nan_negative[31] = f32_bits(0xFFC00001u);
    run_nonfinite_case("nonfinite-nan-negative-payload", nan_negative, win);
    auto positive_inf = make_scores(2048, "random");
    positive_inf[47] = f32_bits(0x7F800000u);
    run_nonfinite_case("nonfinite-positive-infinity", positive_inf, win);
    auto negative_inf = make_scores(2048, "random");
    negative_inf[53] = f32_bits(0xFF800000u);
    run_nonfinite_case("nonfinite-negative-infinity", negative_inf, win);
    std::printf("PASS DSV4 exact top-k gate; no serving/performance verdict\n");
    return 0;
}
