// Matched standalone DSV4 mixed-FP8xFP4 versus the current f16 mirror.
//
// The f16 arm below is not a reimplementation: this TU includes the current
// moe_f16_grouped.cu and calls its exact exported
// memra_moe_kq_gemm_sk_m1_half2 symbol.  Both arms consume the same raw
// ModelOpt code/scales and the same selected expert IDs.  The f16 arm receives
// the CPU-built equivalent of dsv4_fp8_gather_half's normalized-half rows and
// row_scale=max(per-K128-scale)/128.  The mixed arm consumes the raw FP8 bytes
// and per-K128 scales directly, then applies signed per-K16 weight scales in
// f32.  This is a named numeric class, not a bit-identity claim.
//
// Build (compile/link only; do not infer a result without the owning GPU cell):
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     -gencode arch=compute_120a,code=sm_120a \
//     -lcublas -o target/dsv4-mixed-vs-f16-gate \
//     tools/dsv4-mixed-vs-f16-gate.cu

#include "../crates/memra-engine/cu/moe_f16_grouped.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

namespace dsv4_mixed_vs_f16 {

constexpr int kExperts = 6;
constexpr int kBankExperts = 128;
constexpr int kInF = 2048;
constexpr int kOutF = 4096;
constexpr int kRowBytes = kInF / 2;
constexpr int kWeightScaleGroups = kInF / 16;
constexpr int kActScaleGroups = kInF / 128;
constexpr int kTop6Ids[kExperts] = {5, 2, 4, 1, 3, 0};
constexpr float kCandidateAbsTol = 0.0625f;
constexpr float kCandidateRelTol = 1.0e-5f;

static __device__ __forceinline__ float mixed_e4m3_raw(std::uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v;
    if (exp == 0u) v = static_cast<float>(man) * 0x1p-9f;
    else v = __uint_as_float(((exp + 120u) << 23) | (man << 20));
    return (x & 0x80u) ? -v : v;
}

// Raw E2M1 is in the middle four bits of an 8-bit f8f6f4 container.  The
// identity UE8M0 operands make the block-scale MMA an unscaled product; all
// ModelOpt signed scale bytes are applied after the partial in f32.
static __device__ __forceinline__ void mixed_mma_w4a8(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    constexpr unsigned one = 0x7f7f7f7fu;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{0,0},{%11},{0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(one), "r"(one));
}

// Candidate: one m1 output-neuron tile per block, one active expert group per
// grid.y.  This is the same register-fed K16 program gated in the preceding
// standalone lane, but the table is expert-major and the activation rows are
// compact active-group rows, exactly like the f16 ABI.
extern "C" __global__ void mixed_m1_direct_grouped(
        float* __restrict__ out, const std::uint8_t* __restrict__ weights,
        const std::uint8_t* __restrict__ weight_scales,
        const int* __restrict__ ex_ids, const std::uint8_t* __restrict__ act_codes,
        const float* __restrict__ act_scales, int active, int in_f, int out_f,
        int weight_stride, int scale_stride) {
    const int group = static_cast<int>(blockIdx.y);
    if (group >= active) return;
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int row = lane / 4;
    const int chunk = lane & 3;
    const int row_base = static_cast<int>(blockIdx.x) * 16;
    const int row_bytes = in_f / 2;
    const int scale_groups = in_f / 16;
    const int expert = ex_ids[group];
    const std::size_t wbase = static_cast<std::size_t>(expert) * weight_stride;
    const std::size_t sbase = static_cast<std::size_t>(expert) * scale_stride;
    const std::uint8_t* wbank = weights + wbase;
    const std::uint8_t* sbank = weight_scales + sbase;
    const std::uint8_t* arow = act_codes + static_cast<std::size_t>(group) * in_f;
    const float* ascale = act_scales + static_cast<std::size_t>(group) * (in_f / 128);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < (in_f + 31) / 32; ++kb) {
        for (int part = 0; part < 2; ++part) {
            unsigned a[4] = {0u, 0u, 0u, 0u};
            unsigned b[2] = {0u, 0u};
            const int start = kb * 32 + part * 16;
            for (int j = 0; j < 4; ++j) {
                const int k = start + chunk * 4 + j;
                if (k >= in_f) continue;
                const std::uint8_t p0 = wbank[
                    static_cast<std::size_t>(row_base + row) * row_bytes + (k >> 1)];
                const std::uint8_t p8 = wbank[
                    static_cast<std::size_t>(row_base + row + 8) * row_bytes + (k >> 1)];
                const unsigned c0 = (k & 1) ? (p0 >> 4) : (p0 & 0xfu);
                const unsigned c8 = (k & 1) ? (p8 >> 4) : (p8 & 0xfu);
                const unsigned shift = static_cast<unsigned>(j * 8);
                const unsigned char v0 = static_cast<unsigned char>(c0 << 2);
                const unsigned char v8 = static_cast<unsigned char>(c8 << 2);
                const unsigned char av = arow[k];
                if (part == 0) {
                    a[0] |= static_cast<unsigned>(v0) << shift;
                    a[1] |= static_cast<unsigned>(v8) << shift;
                    b[0] |= static_cast<unsigned>(av) << shift;
                } else {
                    a[2] |= static_cast<unsigned>(v0) << shift;
                    a[3] |= static_cast<unsigned>(v8) << shift;
                    b[1] |= static_cast<unsigned>(av) << shift;
                }
            }
            float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mixed_mma_w4a8(partial, a, b);
            for (int l = 0; l < 4; ++l) {
                const int output_row = (l / 2) * 8 + row;
                const int group16 = kb * 2 + part;
                const float ws = (start < in_f)
                    ? mixed_e4m3_raw(sbank[
                        static_cast<std::size_t>(row_base + output_row) * scale_groups + group16])
                    : 0.0f;
                const float xs = (kb * 32 < in_f) ? ascale[kb >> 2] : 0.0f;
                const float weighted = __fmul_rn(partial[l], ws);
                acc[l] = __fadd_rn(acc[l], __fmul_rn(weighted, xs));
            }
        }
    }
    for (int l = 0; l < 4; ++l) {
        const int output_row = (l / 2) * 8 + row;
        // The f16 baseline is N=1.  The MMA tile has eight replicated input
        // columns; the lane with chunk=0 owns column 0 for each row.
        if (chunk == 0 && (l & 1) == 0) {
            out[static_cast<std::size_t>(group) * out_f + row_base + output_row] = acc[l];
        }
    }
}

extern "C" __global__ void matched_l2_flush(
        const std::uint32_t* __restrict__ src, std::uint32_t* __restrict__ sink,
        std::size_t words) {
    std::uint32_t x = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        x ^= src[i] + static_cast<std::uint32_t>(i * 0x9e3779b9u);
    }
    for (int off = 16; off > 0; off >>= 1) x ^= __shfl_down_sync(0xffffffffu, x, off);
    if ((threadIdx.x & 31) == 0) sink[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = x;
}

struct HostExpert {
    std::vector<std::uint8_t> act;
    std::vector<float> act_scales;
    std::vector<__half> act_f16;
    float row_scale = 0.0f;
};

static float host_e4m3_raw(const std::uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v;
    if (exp == 0u) v = static_cast<float>(man) * 0x1p-9f;
    else {
        const std::uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&v, &bits, sizeof(v));
    }
    return (x & 0x80u) ? -v : v;
}

static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static std::uint8_t weight_code(const int expert, const int row, const int k) {
    return static_cast<std::uint8_t>(mix32(
        0x13579bdu + static_cast<std::uint32_t>(expert) * 0x9e3779b9u +
        static_cast<std::uint32_t>(row) * 0x85ebca6bu + static_cast<std::uint32_t>(k)) & 0xfu);
}

static std::uint8_t weight_scale_code(const int expert, const int row, const int group) {
    static constexpr std::uint8_t p[] = {0x40, 0x44, 0x48, 0x4c, 0x50, 0x54, 0x38, 0x3c};
    return p[mix32(static_cast<std::uint32_t>(expert * 131 + row * 17 + group * 7)) & 7u];
}

static std::uint8_t activation_code(const int expert, const int k) {
    static constexpr std::uint8_t p[] = {
        0x01, 0x77, 0x36, 0xc4, 0x49, 0x0f, 0xa8, 0x51,
        0x18, 0xd1, 0x03, 0x4a, 0xb5, 0x27, 0x70, 0xc1,
    };
    return p[mix32(static_cast<std::uint32_t>(expert * 0x9e37 + k * 0x85eb + (k >> 4))) & 15u];
}

static std::vector<float> activation_scale_row(const int expert) {
    static constexpr float p[] = {1.0f, 0.5f, 2.0f, 0.25f, 4.0f, 0.125f, 1.0f, 0.5f,
                                  2.0f, 0.25f, 4.0f, 0.125f, 1.0f, 0.5f, 2.0f, 0.25f};
    std::vector<float> out(kActScaleGroups);
    for (int g = 0; g < kActScaleGroups; ++g) out[g] = p[(expert * 5 + g * 3) & 15];
    return out;
}

static HostExpert make_host_expert(const int expert) {
    HostExpert h;
    h.act.resize(kInF);
    h.act_scales = activation_scale_row(expert);
    h.act_f16.resize(kInF);
    float max_scale = 0.0f;
    for (float s : h.act_scales) max_scale = std::max(max_scale, s);
    h.row_scale = std::ldexp(max_scale, -7);
    for (int k = 0; k < kInF; ++k) {
        h.act[k] = activation_code(expert, k);
        const float value = host_e4m3_raw(h.act[k]) * h.act_scales[k >> 7];
        h.act_f16[k] = __float2half_rn(value / h.row_scale);
        const float back = __half2float(h.act_f16[k]) * h.row_scale;
        if (std::memcmp(&back, &value, sizeof(back)) != 0) {
            throw std::runtime_error("FP8->normalized-half preflight is not lossless");
        }
    }
    return h;
}

static float host_mul(const float a, const float b);
static float host_add(const float a, const float b);

static std::vector<float> cpu_mixed_oracle(
        const std::vector<std::uint8_t>& weights, const std::vector<std::uint8_t>& scales,
        const HostExpert& act, const int expert, const int out_f) {
    std::vector<float> out(out_f, 0.0f);
    for (int row = 0; row < out_f; ++row) {
        float sum = 0.0f;
        for (int base = 0; base < kInF; base += 32) {
            for (int part = 0; part < 2; ++part) {
                float partial = 0.0f;
                const int start = base + part * 16;
                for (int j = 0; j < 16; ++j) {
                    const int k = start + j;
                    const std::uint8_t packed = weights[
                        static_cast<std::size_t>(expert) * kOutF * kRowBytes +
                        static_cast<std::size_t>(row) * kRowBytes + (k >> 1)];
                    const std::uint8_t code = (k & 1) ? (packed >> 4) : (packed & 0xfu);
                    static constexpr float fp4[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
                    const float product = host_mul(fp4[code], host_e4m3_raw(act.act[k]));
                    partial = host_add(partial, product);
                }
                const std::uint8_t sb = scales[
                    static_cast<std::size_t>(expert) * kOutF * kWeightScaleGroups +
                    static_cast<std::size_t>(row) * kWeightScaleGroups + (start >> 4)];
                const float weighted = host_mul(partial, host_e4m3_raw(sb));
                sum = host_add(sum, host_mul(weighted, act.act_scales[base >> 7]));
            }
        }
        out[row] = sum;
    }
    return out;
}

static float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    float r = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) r = std::max(r, std::fabs(a[i] - b[i]));
    return r;
}

static float max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    float r = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
        r = std::max(r, std::fabs(a[i] - b[i]) / std::max(1.0f, std::fabs(b[i])));
    return r;
}

static bool finite_vector(const std::vector<float>& v) {
    for (const float x : v) if (!std::isfinite(x)) return false;
    return true;
}

static bool candidate_accepts(const std::vector<float>& got,
                              const std::vector<float>& expected,
                              float* abs_out, float* rel_out) {
    if (got.size() != expected.size() || !finite_vector(got) || !finite_vector(expected)) {
        *abs_out = std::numeric_limits<float>::infinity();
        *rel_out = std::numeric_limits<float>::infinity();
        return false;
    }
    *abs_out = max_abs(got, expected);
    *rel_out = max_rel(got, expected);
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        if (diff > kCandidateAbsTol + kCandidateRelTol * std::fabs(expected[i])) return false;
    }
    return true;
}

static float host_mul(const float a, const float b) {
    volatile float r = a * b;
    return r;
}

static float host_add(const float a, const float b) {
    volatile float r = a + b;
    return r;
}

static void cuda_check(const cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct DeviceBuffer {
    T* p = nullptr;
    std::size_t n = 0;
    explicit DeviceBuffer(const std::size_t count = 0) : n(count) {
        if (n) cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};

static void upload_active(const int active, const std::vector<int>& ids,
                          const std::vector<HostExpert>& experts,
                          DeviceBuffer<int>& d_ids, DeviceBuffer<int>& d_ref_ids,
                          DeviceBuffer<int>& d_off,
                          DeviceBuffer<std::uint8_t>& d_codes,
                          DeviceBuffer<float>& d_scales,
                          DeviceBuffer<__half>& d_f16,
                          DeviceBuffer<float>& d_row_scale) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(active) * kInF);
    std::vector<float> scales(static_cast<std::size_t>(active) * kActScaleGroups);
    std::vector<__half> f16(static_cast<std::size_t>(active) * kInF);
    std::vector<float> row_scale(active);
    std::vector<int> offsets(kBankExperts + 1), selected(active), reference_ids(kBankExperts);
    for (int g = 0; g < active; ++g) {
        const HostExpert& h = experts[ids[g]];
        selected[g] = ids[g];
        std::memcpy(codes.data() + static_cast<std::size_t>(g) * kInF,
                    h.act.data(), kInF);
        std::memcpy(scales.data() + static_cast<std::size_t>(g) * kActScaleGroups,
                    h.act_scales.data(), kActScaleGroups * sizeof(float));
        std::memcpy(f16.data() + static_cast<std::size_t>(g) * kInF,
                    h.act_f16.data(), kInF * sizeof(__half));
        row_scale[g] = h.row_scale;
    }
    for (int e = 0; e < kBankExperts; ++e) {
        reference_ids[e] = e;
        const bool live = e >= 5 && (e - 5) % 21 == 0 && (e - 5) / 21 < active;
        offsets[e + 1] = offsets[e] + static_cast<int>(live);
    }
    cuda_check(cudaMemcpy(d_ids.p, selected.data(), active * sizeof(int), cudaMemcpyHostToDevice), "upload ids");
    cuda_check(cudaMemcpy(d_ref_ids.p, reference_ids.data(), kBankExperts * sizeof(int), cudaMemcpyHostToDevice), "upload reference bank ids");
    cuda_check(cudaMemcpy(d_off.p, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "upload full rank offsets");
    cuda_check(cudaMemcpy(d_codes.p, codes.data(), codes.size(), cudaMemcpyHostToDevice), "upload FP8 codes");
    cuda_check(cudaMemcpy(d_scales.p, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice), "upload FP8 scales");
    cuda_check(cudaMemcpy(d_f16.p, f16.data(), f16.size() * sizeof(__half), cudaMemcpyHostToDevice), "upload normalized half");
    cuda_check(cudaMemcpy(d_row_scale.p, row_scale.data(), row_scale.size() * sizeof(float), cudaMemcpyHostToDevice), "upload row scales");
}

static void flush_l2(DeviceBuffer<std::uint32_t>& src, DeviceBuffer<std::uint32_t>& sink,
                     const int blocks, cudaStream_t stream) {
    matched_l2_flush<<<blocks, 256, 0, stream>>>(src.p, sink.p, src.n);
    cuda_check(cudaGetLastError(), "L2 flush launch");
}

int run_matched_gate() {
    try {
        const int active_counts[] = {1, 3, 4, 6};
        std::vector<std::uint8_t> h_weights(
            static_cast<std::size_t>(kExperts) * kOutF * kRowBytes);
        std::vector<std::uint8_t> h_weight_scales(
            static_cast<std::size_t>(kExperts) * kOutF * kWeightScaleGroups);
        std::vector<HostExpert> experts;
        experts.reserve(kExperts);
        for (int e = 0; e < kExperts; ++e) {
            experts.push_back(make_host_expert(e));
            for (int row = 0; row < kOutF; ++row) {
                for (int k = 0; k < kInF; k += 2) {
                    const std::uint8_t lo = weight_code(e, row, k);
                    const std::uint8_t hi = weight_code(e, row, k + 1);
                    h_weights[static_cast<std::size_t>(e) * kOutF * kRowBytes +
                              static_cast<std::size_t>(row) * kRowBytes + (k >> 1)] =
                        static_cast<std::uint8_t>(lo | (hi << 4));
                }
                for (int g = 0; g < kWeightScaleGroups; ++g) {
                    h_weight_scales[static_cast<std::size_t>(e) * kOutF * kWeightScaleGroups +
                                    static_cast<std::size_t>(row) * kWeightScaleGroups + g] =
                        weight_scale_code(e, row, g);
                }
            }
        }

        DeviceBuffer<std::uint8_t> d_weights(h_weights.size());
        DeviceBuffer<std::uint8_t> d_weight_scales(h_weight_scales.size());
        cuda_check(cudaMemcpy(d_weights.p, h_weights.data(), h_weights.size(), cudaMemcpyHostToDevice), "upload weights");
        cuda_check(cudaMemcpy(d_weight_scales.p, h_weight_scales.data(), h_weight_scales.size(), cudaMemcpyHostToDevice), "upload weight scales");
        DeviceBuffer<unsigned long long> d_table(static_cast<std::size_t>(6) * kBankExperts);
        std::vector<unsigned long long> table(static_cast<std::size_t>(6) * kBankExperts, 0);
        const std::size_t weight_stride = static_cast<std::size_t>(kOutF) * kRowBytes;
        const std::size_t scale_stride = static_cast<std::size_t>(kOutF) * kWeightScaleGroups;
        for (int group = 0; group < kExperts; ++group) {
            const int bank_id = 5 + 21 * group;
            const int payload = kTop6Ids[group];
            for (int projection = 0; projection < 3; ++projection) {
                table[2 * projection * kBankExperts + bank_id] =
                    reinterpret_cast<std::uintptr_t>(d_weights.p + payload * weight_stride);
                table[(2 * projection + 1) * kBankExperts + bank_id] =
                    reinterpret_cast<std::uintptr_t>(d_weight_scales.p + payload * scale_stride);
            }
        }
        cuda_check(cudaMemcpy(d_table.p, table.data(), table.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice), "upload ModelOpt table");

        DeviceBuffer<int> d_ids(kExperts), d_ref_ids(kBankExperts), d_offsets(kBankExperts + 1);
        DeviceBuffer<std::uint8_t> d_act_codes(static_cast<std::size_t>(kExperts) * kInF);
        DeviceBuffer<float> d_act_scales(static_cast<std::size_t>(kExperts) * kActScaleGroups);
        DeviceBuffer<__half> d_act_f16(static_cast<std::size_t>(kExperts) * kInF);
        DeviceBuffer<float> d_row_scale(kExperts);
        DeviceBuffer<float> d_candidate(static_cast<std::size_t>(kExperts) * kOutF);
        DeviceBuffer<float> d_f16(static_cast<std::size_t>(kExperts) * kOutF);

        int l2_bytes = 0;
        cuda_check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0), "query L2 size");
        const std::size_t flush_bytes = std::max<std::size_t>(2 * static_cast<std::size_t>(l2_bytes), 1u << 20);
        DeviceBuffer<std::uint32_t> d_flush((flush_bytes + 3) / 4);
        const int flush_blocks = 256;
        DeviceBuffer<std::uint32_t> d_flush_sink(flush_blocks * 8);
        cuda_check(cudaMemset(d_flush.p, 0x5a, d_flush.n * sizeof(std::uint32_t)), "init L2 flush");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        cudaEvent_t start = nullptr, stop = nullptr;
        cuda_check(cudaEventCreate(&start), "create start event");
        cuda_check(cudaEventCreate(&stop), "create stop event");
        std::printf("protocol compiler=/usr/local/cuda-13.1/bin/nvcc arch=compute_120a,code=sm_120a "
                    "opt=-O3 source=crates/memra-engine/cu/moe_f16_grouped.cu "
                    "symbol=memra_moe_kq_gemm_sk_m1_half2 flags_match=build.rs:gencode+O3 "
                    "extra=std-c++17,lcublas no-fmad-override candidate_tol_abs=%.9g candidate_tol_rel=%.9g\n",
                    kCandidateAbsTol, kCandidateRelTol);
        std::printf("matched gate L2_bytes=%d flush_bytes=%zu weights=%zu scales=%zu rank_experts=%d sparse_metadata=true\n",
                    l2_bytes, flush_bytes, h_weights.size(), h_weight_scales.size(), kBankExperts);

        for (const int active : active_counts) {
            std::vector<int> ids(kTop6Ids, kTop6Ids + active);
            upload_active(active, ids, experts, d_ids, d_ref_ids, d_offsets, d_act_codes,
                          d_act_scales, d_act_f16, d_row_scale);
            const std::vector<float> expected0 = [&] {
                std::vector<float> out(static_cast<std::size_t>(active) * kOutF);
                for (int g = 0; g < active; ++g) {
                    const auto one = cpu_mixed_oracle(h_weights, h_weight_scales,
                                                      experts[ids[g]], ids[g], kOutF);
                    std::memcpy(out.data() + static_cast<std::size_t>(g) * kOutF,
                                one.data(), kOutF * sizeof(float));
                }
                return out;
            }();

            auto run_arm = [&](const bool candidate, std::vector<float>& host_out) {
                flush_l2(d_flush, d_flush_sink, flush_blocks, stream);
                cuda_check(cudaEventRecord(start, stream), "record start");
                if (candidate) {
                    mixed_m1_direct_grouped<<<dim3(kOutF / 16, active, 1), 32, 0, stream>>>(
                        d_candidate.p, d_weights.p, d_weight_scales.p, d_ids.p,
                        d_act_codes.p, d_act_scales.p, active, kInF, kOutF,
                        static_cast<int>(weight_stride), static_cast<int>(scale_stride));
                    cuda_check(cudaGetLastError(), "launch mixed candidate");
                } else {
                    const int rc = memra_moe_kq_gemm_sk_m1_half2(
                        d_table.p, kBankExperts, d_ref_ids.p, d_act_f16.p, d_f16.p, d_row_scale.p,
                        d_offsets.p, kBankExperts, kInF, kOutF, kRowBytes, stream);
                    if (rc != 0) throw std::runtime_error("memra_moe_kq_gemm_sk_m1_half2 rc=" + std::to_string(rc));
                }
                cuda_check(cudaEventRecord(stop, stream), "record stop");
                cuda_check(cudaEventSynchronize(stop), "sync stop");
                float ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
                host_out.resize(static_cast<std::size_t>(active) * kOutF);
                cuda_check(cudaMemcpy(host_out.data(), candidate ? d_candidate.p : d_f16.p,
                                      host_out.size() * sizeof(float), cudaMemcpyDeviceToHost),
                            "copy arm output");
                return ms;
            };

            // Warm each actual arm once, then measure CUDA-event ABBA. Flush
            // occurs before every arm and is excluded from the event interval.
            std::vector<float> warm_a, warm_b;
            const unsigned long long warm_dispatch_before =
                memra_moe_kq_gemm_sk_m1_half2_dispatches();
            (void)run_arm(false, warm_a);
            const unsigned long long warm_dispatch_after =
                memra_moe_kq_gemm_sk_m1_half2_dispatches();
            if (warm_dispatch_after - warm_dispatch_before != 1) {
                throw std::runtime_error("warm f16 arm did not record one half2 enqueue");
            }
            (void)run_arm(true, warm_b);
            unsigned long long timed_f16_before =
                memra_moe_kq_gemm_sk_m1_half2_dispatches();
            for (int cycle = 0; cycle < 3; ++cycle) {
                std::vector<float> f16_a1, mixed_b1, mixed_b2, f16_a2;
                const float f16_ms1 = run_arm(false, f16_a1);
                const float mixed_ms1 = run_arm(true, mixed_b1);
                const float mixed_ms2 = run_arm(true, mixed_b2);
                const float f16_ms2 = run_arm(false, f16_a2);
                float mixed_abs1 = 0.0f, mixed_rel1 = 0.0f;
                float mixed_abs2 = 0.0f, mixed_rel2 = 0.0f;
                const bool mixed_ok1 = candidate_accepts(
                    mixed_b1, expected0, &mixed_abs1, &mixed_rel1);
                const bool mixed_ok2 = candidate_accepts(
                    mixed_b2, expected0, &mixed_abs2, &mixed_rel2);
                if (!mixed_ok1 || !mixed_ok2) {
                    throw std::runtime_error("candidate-vs-own-oracle tolerance/finite gate failed");
                }
                if (!finite_vector(f16_a1) || !finite_vector(f16_a2)
                    || !finite_vector(mixed_b1) || !finite_vector(mixed_b2)
                    || f16_a1.size() != expected0.size() || f16_a2.size() != expected0.size()) {
                    throw std::runtime_error("matched arm shape or finite gate failed");
                }
                const float f16_abs = max_abs(f16_a1, expected0);
                const float f16_rel = max_rel(f16_a1, expected0);
                const float f16_abs2 = max_abs(f16_a2, expected0);
                const float f16_rel2 = max_rel(f16_a2, expected0);
                std::printf("active=%d cycle=%d candidate-own=(%.9g,%.9g)/(%.9g,%.9g) "
                            "f16-own=(%.9g,%.9g)/(%.9g,%.9g) "
                            "candidate-f16=(%.9g,%.9g)/(%.9g,%.9g) "
                            "ABBA f16_ms=(%.3f,%.3f) mixed_ms=(%.3f,%.3f)\n",
                            active, cycle + 1, mixed_abs1, mixed_rel1, mixed_abs2, mixed_rel2,
                            f16_abs, f16_rel, f16_abs2, f16_rel2,
                            max_abs(mixed_b1, f16_a1), max_rel(mixed_b1, f16_a1),
                            max_abs(mixed_b2, f16_a2), max_rel(mixed_b2, f16_a2),
                            f16_ms1, f16_ms2, mixed_ms1, mixed_ms2);
                if (std::memcmp(mixed_b1.data(), mixed_b2.data(), mixed_b1.size() * sizeof(float)) != 0
                    || std::memcmp(f16_a1.data(), f16_a2.data(), f16_a1.size() * sizeof(float)) != 0) {
                    throw std::runtime_error("ABBA arm output is not bit-stable");
                }
            }
            const unsigned long long timed_f16_after =
                memra_moe_kq_gemm_sk_m1_half2_dispatches();
            if (timed_f16_after - timed_f16_before != 6) {
                throw std::runtime_error("timed f16 arm enqueue counter delta was not 6");
            }
        }
        cuda_check(cudaEventDestroy(start), "destroy start event");
        cuda_check(cudaEventDestroy(stop), "destroy stop event");
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
        std::puts("PASS matched mixed-vs-current-memra_moe_kq_gemm_sk_m1_half2 harness; "
                  "all active counts, 3 ABBA cycles, oracle tolerance, finite/shape checks, "
                  "and half2 enqueue receipts passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}

} // namespace dsv4_mixed_vs_f16

int main() {
    return dsv4_mixed_vs_f16::run_matched_gate();
}
