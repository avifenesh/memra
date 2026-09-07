// Standalone DSV4 GU geometry/component gate.
//
// This tool includes the production moe_f16_grouped.cu and exercises:
//   1. the actual memra_moe_kq_gemm_sk_gu_half2 launcher;
//   2. the same <108,false,true> kernel with a deployable worst-case grid cap;
//   3. the existing <108,true,true> GU-M1+half2 template.
//
// The fixture is one-token, bank=128, hidden=4096, intermediate=2048, and active-group counts
// {1,3,4,6}. The grid-cap arm never uses a host active-group count to choose work: it passes
// total_tiles=-1, so sk_tile_prefix remains authoritative on the device. The only host-derived
// bound is the safe top-k-6 worst case: 6 * ceil(2048 / 64) = 192 tile slots.
//
// Compile (no execution implied):
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr \
//     -gencode arch=compute_120a,code=sm_120a \
//     -o target/dsv4-gu-geometry-gate-r5 tools/dsv4-gu-geometry-gate.cu \
//     -lcublas -lcublasLt

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../crates/memra-engine/cu/moe_f16_grouped.cu"

static constexpr int NE = 128;
static constexpr int HIDDEN = 4096;
static constexpr int INTER = 2048;
static constexpr int TOPK_MAX = 6;
// Keep the bound tied to the production visitor's BN, rather than duplicating a tile width
// in the harness.  The included source defines SK_BN=64 for this GU template.
static constexpr int TILE_N = SK_BN;
static_assert(TILE_N == 64, "GU geometry gate expects the production 64-column visitor tile");
static constexpr int GU_TILE_BOUND = TOPK_MAX * ((INTER + TILE_N - 1) / TILE_N);
static constexpr size_t H2_SMEM_BYTES = 256 * sizeof(uint32_t);
static constexpr uint32_t GUARD_BITS = 0x7fc01234u;

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t _e = (call);                                                       \
        if (_e != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(_e));                                      \
            std::exit(2);                                                              \
        }                                                                               \
    } while (0)

static void check_rc(const char* what, int rc) {
    if (rc != 0) {
        std::fprintf(stderr, "%s rc=%d\n", what, rc);
        std::exit(1);
    }
}

static __half host_half(float value) {
    return __float2half(value);
}

static uint8_t packed_code(size_t i) {
    return static_cast<uint8_t>((i * 37u + i / 17u * 11u + 0x23u) & 0xFFu);
}

static uint8_t scale_code(size_t i) {
    uint8_t code = static_cast<uint8_t>((0x28u + i * 7u + i / 31u * 3u) & 0x7Eu);
    if ((code & 0x7Fu) == 0x7Fu) code = 0x7Eu;
    return code;
}

struct BankBuffers {
    int active;
    cudaStream_t stream;
    unsigned long long* table;
    int* ex_ids;
    int* ex_off;
    __half* act;
    float* row_scale;
    float* macro_g;
    float* macro_u;
    float* route_w;
    uint8_t* gate_codes;
    uint8_t* gate_scales;
    uint8_t* down_codes;
    uint8_t* down_scales;
    uint8_t* up_codes;
    uint8_t* up_scales;
    float* h_control;
    float* h_grid;
    float* h_m1;
    __half* hhalf_control;
    __half* hhalf_grid;
    __half* hhalf_m1;
    float* contrib_control;
    float* contrib_grid;
    float* contrib_m1;
    size_t h_len;
    size_t contrib_len;
};

static void alloc_bank(BankBuffers& b, int active) {
    b.active = active;
    const size_t gate_weight_bytes = static_cast<size_t>(INTER) * (HIDDEN / 2);
    const size_t gate_scale_bytes = static_cast<size_t>(INTER) * (HIDDEN / 16);
    const size_t down_weight_bytes = static_cast<size_t>(HIDDEN) * (INTER / 2);
    const size_t down_scale_bytes = static_cast<size_t>(HIDDEN) * (INTER / 16);
    const size_t h_len = static_cast<size_t>(active) * INTER + 16;
    const size_t contrib_len = static_cast<size_t>(active) * HIDDEN + 16;
    const size_t act_len = static_cast<size_t>(active) * HIDDEN;

    CUDA_OK(cudaMalloc(&b.table, 6 * NE * sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&b.ex_ids, NE * sizeof(int)));
    CUDA_OK(cudaMalloc(&b.ex_off, (NE + 1) * sizeof(int)));
    CUDA_OK(cudaMalloc(&b.act, act_len * sizeof(__half)));
    CUDA_OK(cudaMalloc(&b.row_scale, active * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.macro_g, active * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.macro_u, active * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.route_w, active * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.gate_codes, static_cast<size_t>(active) * gate_weight_bytes));
    CUDA_OK(cudaMalloc(&b.gate_scales, static_cast<size_t>(active) * gate_scale_bytes));
    CUDA_OK(cudaMalloc(&b.down_codes, static_cast<size_t>(active) * down_weight_bytes));
    CUDA_OK(cudaMalloc(&b.down_scales, static_cast<size_t>(active) * down_scale_bytes));
    CUDA_OK(cudaMalloc(&b.up_codes, static_cast<size_t>(active) * gate_weight_bytes));
    CUDA_OK(cudaMalloc(&b.up_scales, static_cast<size_t>(active) * gate_scale_bytes));
    CUDA_OK(cudaMalloc(&b.h_control, h_len * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.h_grid, h_len * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.h_m1, h_len * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.hhalf_control, h_len * sizeof(__half)));
    CUDA_OK(cudaMalloc(&b.hhalf_grid, h_len * sizeof(__half)));
    CUDA_OK(cudaMalloc(&b.hhalf_m1, h_len * sizeof(__half)));
    CUDA_OK(cudaMalloc(&b.contrib_control, contrib_len * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.contrib_grid, contrib_len * sizeof(float)));
    CUDA_OK(cudaMalloc(&b.contrib_m1, contrib_len * sizeof(float)));
    b.h_len = h_len;
    b.contrib_len = contrib_len;

    std::vector<int> ids(NE);
    std::vector<int> offsets(NE + 1, active);
    const int selected[TOPK_MAX] = {0, 17, 34, 51, 68, 85};
    std::vector<int> selected_flag(NE, 0);
    for (int i = 0; i < active; ++i) selected_flag[selected[i]] = 1;
    int live = 0;
    for (int e = 0; e < NE; ++e) {
        ids[e] = e;
        offsets[e] = live;
        live += selected_flag[e];
    }
    offsets[NE] = live;
    CUDA_OK(cudaMemcpy(b.ex_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.ex_off, offsets.data(), offsets.size() * sizeof(int),
                       cudaMemcpyHostToDevice));

    std::vector<__half> act_host(act_len);
    for (int group = 0; group < active; ++group) {
        for (int k = 0; k < HIDDEN; ++k) {
            float value = (group + 1) * 0.03125f + ((k * 13 + group * 17) % 97) / 64.0f - 0.75f;
            act_host[static_cast<size_t>(group) * HIDDEN + k] = host_half(value);
        }
    }
    std::vector<float> row_scale(active), macro_g(active), macro_u(active), route_w(active);
    for (int i = 0; i < active; ++i) {
        row_scale[i] = 0.75f + 0.125f * (i + 1);
        macro_g[i] = 0.5f + 0.0625f * (i + 2);
        macro_u[i] = 0.625f + 0.09375f * (i + 1);
        route_w[i] = 0.125f + 0.0625f * i;
    }
    CUDA_OK(cudaMemcpy(b.act, act_host.data(), act_host.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.row_scale, row_scale.data(), row_scale.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.macro_g, macro_g.data(), macro_g.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.macro_u, macro_u.data(), macro_u.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.route_w, route_w.data(), route_w.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<uint8_t> gate_codes(static_cast<size_t>(active) * gate_weight_bytes);
    std::vector<uint8_t> up_codes(static_cast<size_t>(active) * gate_weight_bytes);
    std::vector<uint8_t> down_codes(static_cast<size_t>(active) * down_weight_bytes);
    std::vector<uint8_t> gate_scales(static_cast<size_t>(active) * gate_scale_bytes);
    std::vector<uint8_t> up_scales(static_cast<size_t>(active) * gate_scale_bytes);
    std::vector<uint8_t> down_scales(static_cast<size_t>(active) * down_scale_bytes);
    for (size_t i = 0; i < gate_codes.size(); ++i) {
        gate_codes[i] = packed_code(i + 3);
        up_codes[i] = packed_code(i + 1009);
    }
    for (size_t i = 0; i < down_codes.size(); ++i) down_codes[i] = packed_code(i + 2003);
    for (size_t i = 0; i < gate_scales.size(); ++i) {
        gate_scales[i] = scale_code(i + 5);
        up_scales[i] = scale_code(i + 7001);
    }
    for (size_t i = 0; i < down_scales.size(); ++i) down_scales[i] = scale_code(i + 11003);
    CUDA_OK(cudaMemcpy(b.gate_codes, gate_codes.data(), gate_codes.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.up_codes, up_codes.data(), up_codes.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.down_codes, down_codes.data(), down_codes.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.gate_scales, gate_scales.data(), gate_scales.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.up_scales, up_scales.data(), up_scales.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.down_scales, down_scales.data(), down_scales.size(), cudaMemcpyHostToDevice));

    std::vector<uint64_t> table(6 * NE);
    for (int e = 0; e < NE; ++e) {
        int local = 0;
        while (local < active && selected[local] != e) ++local;
        if (local == active) local = 0;
        table[e] = reinterpret_cast<uint64_t>(b.gate_codes + static_cast<size_t>(local) * gate_weight_bytes);
        table[NE + e] = reinterpret_cast<uint64_t>(b.gate_scales + static_cast<size_t>(local) * gate_scale_bytes);
        table[2 * NE + e] = reinterpret_cast<uint64_t>(b.down_codes + static_cast<size_t>(local) * down_weight_bytes);
        table[3 * NE + e] = reinterpret_cast<uint64_t>(b.down_scales + static_cast<size_t>(local) * down_scale_bytes);
        table[4 * NE + e] = reinterpret_cast<uint64_t>(b.up_codes + static_cast<size_t>(local) * gate_weight_bytes);
        table[5 * NE + e] = reinterpret_cast<uint64_t>(b.up_scales + static_cast<size_t>(local) * gate_scale_bytes);
    }
    CUDA_OK(cudaMemcpy(b.table, table.data(), table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    // Fill the tail with a bit-pattern guard, rather than cudaMemset(0xFF): the latter
    // produces 0xffffffff and would not distinguish an accidental write from an ordinary
    // NaN. The live rows are overwritten by every arm; only these trailing elements are
    // used as OOB sentinels.
    const std::vector<uint32_t> h_guard(b.h_len, GUARD_BITS);
    const std::vector<uint32_t> c_guard(b.contrib_len, GUARD_BITS);
    CUDA_OK(cudaMemcpy(b.h_control, h_guard.data(), h_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.h_grid, h_guard.data(), h_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.h_m1, h_guard.data(), h_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.contrib_control, c_guard.data(), c_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.contrib_grid, c_guard.data(), c_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b.contrib_m1, c_guard.data(), c_guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
}

static void free_bank(BankBuffers& b) {
    CUDA_OK(cudaFree(b.table)); CUDA_OK(cudaFree(b.ex_ids)); CUDA_OK(cudaFree(b.ex_off));
    CUDA_OK(cudaFree(b.act)); CUDA_OK(cudaFree(b.row_scale)); CUDA_OK(cudaFree(b.macro_g));
    CUDA_OK(cudaFree(b.macro_u)); CUDA_OK(cudaFree(b.route_w)); CUDA_OK(cudaFree(b.gate_codes));
    CUDA_OK(cudaFree(b.gate_scales)); CUDA_OK(cudaFree(b.down_codes)); CUDA_OK(cudaFree(b.down_scales));
    CUDA_OK(cudaFree(b.up_codes)); CUDA_OK(cudaFree(b.up_scales)); CUDA_OK(cudaFree(b.h_control));
    CUDA_OK(cudaFree(b.h_grid)); CUDA_OK(cudaFree(b.h_m1)); CUDA_OK(cudaFree(b.hhalf_control));
    CUDA_OK(cudaFree(b.hhalf_grid)); CUDA_OK(cudaFree(b.hhalf_m1)); CUDA_OK(cudaFree(b.contrib_control));
    CUDA_OK(cudaFree(b.contrib_grid)); CUDA_OK(cudaFree(b.contrib_m1));
}

__global__ static void f32_to_half_rows(const float* src, __half* dst, size_t n) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        dst[i] = __float2half(src[i]);
}

struct LaunchGeometry {
    int sms;
    int control_occ;
    int control_full_grid;
    int capped_grid;
    int m1_occ;
    int m1_full_grid;
};

static LaunchGeometry geometry() {
    int dev = 0, sms = 1, control_occ = 1, m1_occ = 1;
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &control_occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, false, true>, 128,
        H2_SMEM_BYTES));
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &m1_occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true, true>, 128,
        H2_SMEM_BYTES));
    return {sms, control_occ, sms * control_occ, std::min(sms * control_occ, GU_TILE_BOUND),
            m1_occ, sms * m1_occ};
}

static void launch_grid_cap(const BankBuffers& b, const LaunchGeometry& g) {
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, false, true>
        <<<g.capped_grid, dim3(32, 4, 1), H2_SMEM_BYTES, b.stream>>>(
            b.table, NE, b.ex_ids, HIDDEN / 2, b.act, b.h_grid, b.row_scale,
            b.macro_g, b.macro_u, b.route_w, b.ex_off, NE, HIDDEN, INTER, -1, 6.0f);
    CUDA_OK(cudaGetLastError());
}

static void launch_m1_half2(const BankBuffers& b, const LaunchGeometry& g) {
    moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, true, true>
        <<<g.m1_full_grid, dim3(32, 4, 1), H2_SMEM_BYTES, b.stream>>>(
            b.table, NE, b.ex_ids, HIDDEN / 2, b.act, b.h_m1, b.row_scale,
            b.macro_g, b.macro_u, b.route_w, b.ex_off, NE, HIDDEN, INTER, -1, 6.0f);
    CUDA_OK(cudaGetLastError());
}

static void half_from(float* src, __half* dst, size_t n, cudaStream_t stream) {
    f32_to_half_rows<<<128, 256, 0, stream>>>(src, dst, n);
    CUDA_OK(cudaGetLastError());
}

static void launch_down(const BankBuffers& b, __half* hhalf, float* contribution) {
    check_rc("down half2", memra_moe_kq_gemm_sk_m1_half2(
        b.table, NE, b.ex_ids, hhalf, contribution, b.row_scale, b.ex_off, NE,
        INTER, HIDDEN, INTER / 2, reinterpret_cast<void*>(b.stream)));
}

static std::vector<uint32_t> dtoh_bits(const float* device, size_t n, cudaStream_t stream) {
    std::vector<float> values(n);
    CUDA_OK(cudaMemcpyAsync(values.data(), device, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<uint32_t> bits(n);
    for (size_t i = 0; i < n; ++i) std::memcpy(&bits[i], &values[i], sizeof(uint32_t));
    return bits;
}

static void assert_rows_and_guards(const char* arm, const std::vector<uint32_t>& reference,
                                   const std::vector<uint32_t>& actual, int active,
                                   int width, const char* plane) {
    for (int group = 0; group < active; ++group) {
        const size_t begin = static_cast<size_t>(group) * width;
        for (int col = 0; col < width; ++col) {
            if (reference[begin + col] != actual[begin + col]) {
                std::fprintf(stderr, "FAIL %s %s group=%d col=%d ref=%08x got=%08x\n",
                             arm, plane, group, col, reference[begin + col], actual[begin + col]);
                std::exit(1);
            }
        }
    }
    for (size_t i = static_cast<size_t>(active) * width; i < actual.size(); ++i) {
        if (actual[i] != GUARD_BITS) {
            std::fprintf(stderr, "FAIL %s %s guard index=%zu got=%08x\n", arm, plane, i, actual[i]);
            std::exit(1);
        }
    }
}

static void assert_h_bits_finite_guards(const char* arm, const std::vector<uint32_t>* reference,
                                        const std::vector<uint32_t>& actual, int active,
                                        int width) {
    if (reference != nullptr)
        assert_rows_and_guards(arm, *reference, actual, active, width, "H");
    const size_t live = static_cast<size_t>(active) * width;
    for (size_t i = 0; i < live; ++i) {
        float value = 0.0f;
        std::memcpy(&value, &actual[i], sizeof(value));
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL %s H nonfinite index=%zu bits=%08x\n", arm, i,
                         actual[i]);
            std::exit(1);
        }
    }
    for (size_t i = live; i < actual.size(); ++i) {
        if (actual[i] != GUARD_BITS) {
            std::fprintf(stderr, "FAIL %s H guard index=%zu got=%08x\n", arm, i, actual[i]);
            std::exit(1);
        }
    }
}

static void flush_l2(cudaStream_t stream, uint8_t* buffer, size_t bytes) {
    CUDA_OK(cudaMemsetAsync(buffer, 0xA5, bytes, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
}

enum class ScoredArm { Current, GridCap, M1Half2 };

static const char* scored_arm_name(ScoredArm arm) {
    switch (arm) {
        case ScoredArm::Current: return "current";
        case ScoredArm::GridCap: return "grid-cap";
        case ScoredArm::M1Half2: return "m1-half2";
    }
    return "unknown";
}

static float score_launch(ScoredArm arm, const BankBuffers& b, const LaunchGeometry& g,
                          uint8_t* flush_buffer, size_t flush_bytes,
                          const std::vector<uint32_t>* reference,
                          std::vector<uint32_t>& observed) {
    // flush_bytes is exactly 2x the device L2 size. It is scrubbed before EVERY scored
    // launch, so no arm gets a cache-warm advantage from its predecessor in the sequence.
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    const unsigned long long before = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    CUDA_OK(cudaEventRecord(start, b.stream));
    switch (arm) {
        case ScoredArm::Current:
            check_rc("GU half2 current", memra_moe_kq_gemm_sk_gu_half2(
                b.table, NE, b.ex_ids, b.act, b.h_control, b.row_scale, b.macro_g, b.macro_u,
                b.route_w, b.ex_off, NE, HIDDEN, INTER, 6.0f, HIDDEN / 2, b.stream));
            break;
        case ScoredArm::GridCap:
            launch_grid_cap(b, g);
            break;
        case ScoredArm::M1Half2:
            launch_m1_half2(b, g);
            break;
    }
    CUDA_OK(cudaEventRecord(stop, b.stream));
    CUDA_OK(cudaEventSynchronize(stop));
    const unsigned long long after = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    const unsigned long long delta = after - before;
    if (arm == ScoredArm::Current && delta != 1) {
        std::fprintf(stderr, "FAIL scored arm=%s control enqueue delta=%llu\n",
                     scored_arm_name(arm), delta);
        std::exit(1);
    }
    if (arm != ScoredArm::Current && delta != 0) {
        std::fprintf(stderr, "FAIL scored arm=%s unexpectedly changed control counter by %llu\n",
                     scored_arm_name(arm), delta);
        std::exit(1);
    }
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));

    const float* output = arm == ScoredArm::Current
        ? b.h_control : (arm == ScoredArm::GridCap ? b.h_grid : b.h_m1);
    observed = dtoh_bits(output, b.h_len, b.stream);
    assert_h_bits_finite_guards(scored_arm_name(arm), reference, observed, b.active, INTER);
    return ms;
}

static void run_comparison(const char* sequence, ScoredArm candidate, int active,
                           const BankBuffers& b, const LaunchGeometry& g,
                           uint8_t* flush_buffer, size_t flush_bytes,
                           std::vector<uint32_t>& h_reference) {
    const unsigned long long control_before = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    double current_sum = 0.0;
    double candidate_sum = 0.0;
    int current_count = 0;
    int candidate_count = 0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        // True A/B/B/A, repeated three times. The first current row establishes the
        // reference; every subsequent timed row is compared against it immediately.
        std::vector<uint32_t> observed;
        const std::vector<uint32_t>* reference = h_reference.empty() ? nullptr : &h_reference;
        current_sum += score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes,
                                    reference, observed);
        ++current_count;
        if (h_reference.empty()) h_reference = observed;

        observed.clear();
        candidate_sum += score_launch(candidate, b, g, flush_buffer, flush_bytes,
                                      &h_reference, observed);
        ++candidate_count;
        observed.clear();
        candidate_sum += score_launch(candidate, b, g, flush_buffer, flush_bytes,
                                      &h_reference, observed);
        ++candidate_count;

        observed.clear();
        current_sum += score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes,
                                    &h_reference, observed);
        ++current_count;
    }
    const unsigned long long control_after = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    if (control_after - control_before != 6) {
        std::fprintf(stderr, "FAIL active=%d sequence=%s control dispatch delta=%llu expected=6\n",
                     active, sequence, control_after - control_before);
        std::exit(1);
    }
    std::printf("COMPARE active_groups=%d sequence=%s cycles=3 current_samples=%d candidate=%s "
                "candidate_samples=%d control_dispatch_delta=6 "
                "current_mean_ms=%.6f candidate_mean_ms=%.6f "
                "flush_2x_l2_before_every_scored=true h_bit_exact_finite_guards_every_row=true "
                "within_arm_repeat=true\n",
                active, sequence, current_count, scored_arm_name(candidate), candidate_count,
                current_sum / current_count, candidate_sum / candidate_count);
}

static void run_case(int active, uint8_t* flush_buffer, size_t flush_bytes) {
    BankBuffers b{};
    b.stream = nullptr;
    alloc_bank(b, active);
    const LaunchGeometry g = geometry();
    // Warm every concrete kernel before collecting scored events. Runtime/module first-use
    // cost is not a kernel improvement. These observations are checked but never aggregated.
    std::vector<uint32_t> h_reference;
    (void)score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes,
                       nullptr, h_reference);
    for (ScoredArm candidate : {ScoredArm::GridCap, ScoredArm::M1Half2}) {
        std::vector<uint32_t> observed;
        (void)score_launch(candidate, b, g, flush_buffer, flush_bytes,
                           &h_reference, observed);
    }
    std::printf("WARMUP active_groups=%d arms=current,grid-cap,m1-half2 checked=true scored=false\n", active);
    // The real control participates in both complete ABBA comparisons. All timed rows
    // are compared against the checked warm reference after their device event.
    run_comparison("current/grid/grid/current", ScoredArm::GridCap, active, b, g,
                   flush_buffer, flush_bytes, h_reference);
    run_comparison("current/m1/m1/current", ScoredArm::M1Half2, active, b, g,
                   flush_buffer, flush_bytes, h_reference);

    // The down result below is a deliberately labeled diagnostic: this standalone fixture
    // converts GU f32 output with f32->half and then uses the existing m1-half2 down launcher.
    // It does not reproduce the production FP8-QAT normalization/transport, so it is not a
    // full-chain serving proof. The GU H identity above is the production-relevant gate.
    const auto h_control = dtoh_bits(b.h_control, b.h_len, b.stream);
    const auto h_grid = dtoh_bits(b.h_grid, b.h_len, b.stream);
    const auto h_m1 = dtoh_bits(b.h_m1, b.h_len, b.stream);
    assert_h_bits_finite_guards("down-control-input", &h_reference, h_control, active, INTER);
    assert_h_bits_finite_guards("down-grid-input", &h_reference, h_grid, active, INTER);
    assert_h_bits_finite_guards("down-m1-input", &h_reference, h_m1, active, INTER);
    half_from(b.h_control, b.hhalf_control, static_cast<size_t>(active) * INTER, b.stream);
    half_from(b.h_grid, b.hhalf_grid, static_cast<size_t>(active) * INTER, b.stream);
    half_from(b.h_m1, b.hhalf_m1, static_cast<size_t>(active) * INTER, b.stream);
    launch_down(b, b.hhalf_control, b.contrib_control);
    launch_down(b, b.hhalf_grid, b.contrib_grid);
    launch_down(b, b.hhalf_m1, b.contrib_m1);
    CUDA_OK(cudaStreamSynchronize(b.stream));
    const auto c_control = dtoh_bits(b.contrib_control, b.contrib_len, b.stream);
    const auto c_grid = dtoh_bits(b.contrib_grid, b.contrib_len, b.stream);
    const auto c_m1 = dtoh_bits(b.contrib_m1, b.contrib_len, b.stream);
    assert_rows_and_guards("grid-cap", c_control, c_grid, active, HIDDEN, "contribution");
    assert_rows_and_guards("m1-half2", c_control, c_m1, active, HIDDEN, "contribution");
    const int exact_active_tiles_diagnostic = active * ((INTER + TILE_N - 1) / TILE_N);
    std::printf("CASE active_groups=%d bank=128 hidden=4096 inter=2048 max_tiles=%d "
                "exact_active_tiles_diagnostic_only=%d full_grid=%d capped_grid=%d "
                "sms=%d control_occ=%d m1_full_grid=%d m1_occ=%d "
                "grid_total_tiles_source=device_prefix "
                "h_bit_exact_finite_guards=true within_arm_repeat=true "
                "down_contribution_diagnostic=true down_transport=f32_to_half_then_m1_half2\n",
                active, GU_TILE_BOUND, exact_active_tiles_diagnostic, g.control_full_grid,
                g.capped_grid, g.sms, g.control_occ, g.m1_full_grid, g.m1_occ);
    free_bank(b);
}

int main() {
    int dev = 0; CUDA_OK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(prop.l2CacheSize), 1 << 20);
    uint8_t* flush_buffer = nullptr; CUDA_OK(cudaMalloc(&flush_buffer, flush_bytes));
    std::printf("CONTROL bank=128 topk_max=6 tile_n=64 deployable_tile_bound=%d l2_flush_bytes=%zu "
                "active_count_grid_choice=device_prefix_authoritative\n", GU_TILE_BOUND, flush_bytes);
    for (int active : {1, 3, 4, 6}) run_case(active, flush_buffer, flush_bytes);
    CUDA_OK(cudaFree(flush_buffer));
    std::printf("PASS GU geometry component source gate; no production/throughput verdict\n");
    return 0;
}
