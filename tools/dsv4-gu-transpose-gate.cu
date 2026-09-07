// Standalone DSV4 GU output-neuron transpose probe.
//
// This is intentionally not wired into the engine.  It compares the current
// ModelOpt GU half2 F16-MMA launcher against a transpose candidate:
//
//   A = exact dequantized FP16 weight rows [32 output rows, K]
//   B = the existing normalized FP16 activation row replicated across N=8
//       columns, so each m16n8k16 result contains 16 useful output neurons
//       and eight duplicate activation columns.
//
// The candidate processes G and U sequentially through one shared weight-A /
// activation-B tile.  Each projection keeps its own f32 accumulator and the
// exact ascending K64/K16 chain.  This avoids allocating simultaneous G/U
// weight banks while preserving the per-projection reduction order.
//
// Build (run only under the owning non-serving GPU lock; no GPU result is implied
// by compilation):
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a \
//     tools/dsv4-gu-transpose-gate.cu -o target/dsv4-gu-transpose-gate \
//     -lcublas -lcublasLt

#include "../crates/memra-engine/cu/moe_f16_grouped.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4_gu_transpose {

constexpr int kBankExperts = 128;
constexpr int kExperts = 6;
constexpr int kHidden = 4096;
constexpr int kInter = 2048;
constexpr int kRowBytes = kHidden / 2;
constexpr int kScaleGroups = kHidden / 16;
constexpr int kTransposeBm = 32;
constexpr int kTransposeBRows = 16;
constexpr int kSelected[kExperts] = {0, 17, 34, 51, 68, 85};
constexpr int kPayload[kExperts] = {5, 2, 4, 1, 3, 0};
constexpr size_t kPackedLutBytes = 256 * sizeof(std::uint32_t);
constexpr std::uint32_t kGuardBits = 0x7fc01234u;

static __device__ __forceinline__ float transpose_gu_epilogue(
        float gate, float up, float row_scale, float macro_gate,
        float macro_up, float route_weight, float limit) {
    // This is the production KQGU_STORE DAG in the same order.  Do not turn
    // this into a fused expression: the exact F16-MMA comparison depends on
    // the named round-to-nearest products and clamp points.
    float g = __fmul_rn(__fmul_rn(gate, row_scale), macro_gate);
    float u = __fmul_rn(__fmul_rn(up, row_scale), macro_up);
    u = fminf(fmaxf(u, -limit), limit);
    g = fminf(g, limit);
    float h = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u);
    return __fmul_rn(h, route_weight);
}

template<bool PackedStore>
static __global__ void __launch_bounds__(64)
dsv4_gu_transpose_kernel(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ H, const float* __restrict__ row_scale,
        const float* __restrict__ macro_g, const float* __restrict__ macro_u,
        const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit) {
    if (in_f != kHidden || out_f != kInter) return;

    __shared__ int s_pre[SK_MAX_G + 1];
    // One weight-A tile is reused for G then U.  B is one activation row
    // copied once and replicated into 16 shared rows for the ldmatrix B
    // fragment.  This is intentionally not two simultaneous G/U banks.
    __shared__ __align__(16) __half As[SKT_STAGES][kTransposeBm][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SKT_STAGES][kTransposeBRows][SKT_STRIDE];
    __shared__ std::uint32_t s_cb[KQ_CB_WORDS(QT_NVFP4_MODELOPT)];
    extern __shared__ std::uint32_t packed_h2[];

    const int ntx = (out_f + kTransposeBm - 1) / kTransposeBm;
    kq_stage_codebook<QT_NVFP4_MODELOPT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    // m_e=1 only.  The prefix remains device-authoritative and the host
    // launch passes total_tiles=-1, matching the production visitor contract.
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, kTransposeBm, 1, 2);
    if (total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int nkb = in_f / SKT_BK;
    const int wm = warp * 16;

    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int g = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g];
        const int m_e = ex_off[g + 1] - lo;
        if (m_e != 1) continue;
        const int local = t - s_pre[g];
        const int n0 = (local % ntx) * kTransposeBm;
        const __half* Ag = A + static_cast<size_t>(lo) * in_f;
        const int eid = ex_ids[g];

        float gate_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float up_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Projection 0 is gate, projection 1 is up.  Sequential reuse is
        // deliberate: each projection gets a fresh accumulator and the same
        // K-block order as the current interleaved GU control.
        for (int projection = 0; projection < 2; ++projection) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int qplane = projection == 0 ? 0 : 4;
            const int splane = projection == 0 ? 1 : 5;
            const std::uint8_t* W = reinterpret_cast<const std::uint8_t*>(
                table[static_cast<size_t>(qplane) * n_expert + eid]);
            const std::uint8_t* S = reinterpret_cast<const std::uint8_t*>(
                table[static_cast<size_t>(splane) * n_expert + eid]);

#define DSV4_TRANSPOSE_LOAD(STAGE, K0)                                                \
    do {                                                                               \
        const int output_row = tid >> 1;                                              \
        if (output_row < kTransposeBm) {                                               \
            const int row_part = (tid & 1) * 2;                                       \
            const int row = min(n0 + output_row, out_f - 1);                          \
            const std::uint8_t* wrow = W + static_cast<size_t>(row) * row_bytes;      \
            const std::uint8_t* srow = S + static_cast<size_t>(row) * (in_f / 16);    \
            for (int part = 0; part < 2; ++part) {                                    \
                const int chunk = row_part + part;                                    \
                const KqRaw raw = kq_fetch<QT_NVFP4_MODELOPT>(                        \
                    wrow, srow, (K0) + chunk * 16, s_cb, in_f);                       \
                kq_store_variant<QT_NVFP4_MODELOPT, PackedStore>(                     \
                    raw, &As[STAGE][output_row][chunk * 16], s_cb, packed_h2);         \
            }                                                                          \
        }                                                                              \
        if (tid < 8) {                                                                 \
            sk_cp16(&Bs[STAGE][0][tid * 8], Ag + (K0) + tid * 8);                      \
        }                                                                              \
        asm volatile("cp.async.commit_group;");                                       \
    } while (0)

            // sk_cp16 copies 16 bytes = 8 half values; eight copies cover
            // the complete K64 activation tile.
            DSV4_TRANSPOSE_LOAD(0, 0);
            if (nkb > 1) DSV4_TRANSPOSE_LOAD(1, SKT_BK);

            for (int kb = 0; kb < nkb; ++kb) {
                const int cur = kb % SKT_STAGES;
                if (kb + 2 < nkb) {
                    DSV4_TRANSPOSE_LOAD((kb + 2) % SKT_STAGES,
                                        (kb + 2) * SKT_BK);
                    asm volatile("cp.async.wait_group 2;");
                } else if (kb + 1 < nkb) {
                    asm volatile("cp.async.wait_group 1;");
                } else {
                    asm volatile("cp.async.wait_group 0;");
                }
                __syncthreads();

                // One global activation load, then exact half-bit replication
                // into the 16 B rows consumed by ldmatrix.  No activation
                // requantization or f32 reconstruction occurs here.
                // Row 0 is the cp.async source.  Never rewrite it while other
                // threads are reading it; only populate replicated rows 1..15.
                for (int i = tid; i < (kTransposeBRows - 1) * SKT_BK; i += 64) {
                    const int row = 1 + i / SKT_BK;
                    const int col = i % SKT_BK;
                    Bs[cur][row][col] = Bs[cur][0][col];
                }
                __syncthreads();

                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    unsigned a[4], b0[4];
                    sk_ldm16x16(a, &As[cur][wm][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[cur][0][kk * 16], SKT_STRIDE);
                    // N=8 only. Every B column is the same activation row;
                    // b0[0]/b0[2] is the first output-column fragment.
                    sk_mma(acc, a, b0[0], b0[2]);
                }
                __syncthreads();
            }
#undef DSV4_TRANSPOSE_LOAD
            for (int i = 0; i < 4; ++i) {
                if (projection == 0) gate_acc[i] = acc[i];
                else up_acc[i] = acc[i];
            }
        }

        if ((lane & 3) == 0) {
            const int row = n0 + wm + lane / 4;
            const int pair = lo;
            const float rs = row_scale[pair];
            const float mg = macro_g[pair];
            const float mu = macro_u[pair];
            const float rw = route_w[pair];
            float* hrow = H + static_cast<size_t>(pair) * out_f;
            if (row < out_f) {
                hrow[row] = transpose_gu_epilogue(
                    gate_acc[0], up_acc[0], rs, mg, mu, rw, limit);
            }
            if (row + 8 < out_f) {
                hrow[row + 8] = transpose_gu_epilogue(
                    gate_acc[2], up_acc[2], rs, mg, mu, rw, limit);
            }
        }
    }
}

} // namespace dsv4_gu_transpose

namespace dsv4_gu_transpose {

static void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

static void check_rc(const char* what, int rc) {
    if (rc != 0) throw std::runtime_error(std::string(what) + " rc=" + std::to_string(rc));
}

static std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

static bool anchor_row(int row) {
    static constexpr int rows[] = {
        0, 1, 7, 8, 15, 16, 23, 31, 32, 47, 63, 64,
        127, 255, 511, 1023, 1535, 2031, 2032, 2047,
    };
    for (const int candidate : rows) if (candidate == row) return true;
    return false;
}

static std::uint8_t weight_code(int expert, int row, int k, bool anchors) {
    std::uint32_t v = mix32(0x13579bdu
        + static_cast<std::uint32_t>(expert) * 0x9e3779b9u
        + static_cast<std::uint32_t>(row) * 0x85ebca6bu
        + static_cast<std::uint32_t>(k));
    if (anchors && anchor_row(row) && k >= kHidden - 64) {
        v = mix32(v ^ (static_cast<std::uint32_t>(row) << 11)
                  ^ static_cast<std::uint32_t>(k >> 4));
    }
    return static_cast<std::uint8_t>(v & 0xfu);
}

static std::uint8_t weight_scale_code(int expert, int row, int group, bool anchors) {
    static constexpr std::uint8_t values[] = {
        0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60,
        0x34, 0x44, 0x4c, 0x54, 0x5c, 0x64, 0x3c, 0x6c,
    };
    std::uint32_t index = mix32(static_cast<std::uint32_t>(expert * 131
        + row * 17 + group * 7));
    if (anchors && anchor_row(row) && group >= kScaleGroups - 4) {
        index = static_cast<std::uint32_t>(row * 3 + group * 11 + expert);
    }
    return values[index & 15u];
}

struct Fixture {
    bool anchors = false;
    int active = 0;
    std::vector<std::uint8_t> gate_codes;
    std::vector<std::uint8_t> gate_scales;
    std::vector<std::uint8_t> up_codes;
    std::vector<std::uint8_t> up_scales;
    std::vector<__half> act;
    std::vector<float> row_scale;
    std::vector<float> macro_g;
    std::vector<float> macro_u;
    std::vector<float> route_w;

    Fixture(int active_count, bool anchor_fixture) : anchors(anchor_fixture), active(active_count) {
        const size_t weight_stride = static_cast<size_t>(kInter) * kRowBytes;
        const size_t scale_stride = static_cast<size_t>(kInter) * kScaleGroups;
        gate_codes.resize(static_cast<size_t>(kExperts) * weight_stride);
        gate_scales.resize(static_cast<size_t>(kExperts) * scale_stride);
        up_codes.resize(static_cast<size_t>(kExperts) * weight_stride);
        up_scales.resize(static_cast<size_t>(kExperts) * scale_stride);
        for (int expert = 0; expert < kExperts; ++expert) {
            for (int row = 0; row < kInter; ++row) {
                for (int k = 0; k < kHidden; k += 2) {
                    const std::uint8_t lo = weight_code(expert, row, k, anchors);
                    const std::uint8_t hi = weight_code(expert, row, k + 1, anchors);
                    gate_codes[static_cast<size_t>(expert) * weight_stride
                               + static_cast<size_t>(row) * kRowBytes + (k >> 1)]
                        = static_cast<std::uint8_t>(lo | (hi << 4));
                    const std::uint8_t ulo = weight_code(expert + 11, row, k + 3, anchors);
                    const std::uint8_t uhi = weight_code(expert + 11, row, k + 4, anchors);
                    up_codes[static_cast<size_t>(expert) * weight_stride
                             + static_cast<size_t>(row) * kRowBytes + (k >> 1)]
                        = static_cast<std::uint8_t>(ulo | (uhi << 4));
                }
                for (int group = 0; group < kScaleGroups; ++group) {
                    gate_scales[static_cast<size_t>(expert) * scale_stride
                                + static_cast<size_t>(row) * kScaleGroups + group]
                        = weight_scale_code(expert, row, group, anchors);
                    up_scales[static_cast<size_t>(expert) * scale_stride
                              + static_cast<size_t>(row) * kScaleGroups + group]
                        = weight_scale_code(expert + 7, row, group, anchors);
                }
            }
        }

        act.resize(static_cast<size_t>(active) * kHidden);
        row_scale.resize(active);
        macro_g.resize(active);
        macro_u.resize(active);
        route_w.resize(active);
        for (int group = 0; group < active; ++group) {
            row_scale[group] = 0.75f + 0.125f * static_cast<float>(group + 1);
            macro_g[group] = 0.5f + 0.0625f * static_cast<float>(group + 2);
            macro_u[group] = 0.625f + 0.09375f * static_cast<float>(group + 1);
            route_w[group] = 0.125f + 0.0625f * static_cast<float>(group);
            for (int k = 0; k < kHidden; ++k) {
                float value = static_cast<float>(group + 1) * 0.03125f
                    + static_cast<float>((k * 13 + group * 17) % 97) / 64.0f - 0.75f;
                if (anchors && k < 8) {
                    // Explicit first 8-half block anchor: catches a short
                    // cp.async coverage bug immediately.
                    value = static_cast<float>((group + 1) * (k + 3)) / 19.0f - 0.875f;
                } else if (anchors && k >= kHidden - 8) {
                    // Explicit last 8-half block anchor: catches a K-tail
                    // offset or partial-copy bug that first-block tests miss.
                    value = static_cast<float>((group + 3) * (kHidden - k + 5)) / 23.0f - 1.125f;
                } else if (anchors && (k < 16 || k >= kHidden - 16)) {
                    value = static_cast<float>((group + 1) * (k + 3)) / 37.0f - 1.25f;
                }
                act[static_cast<size_t>(group) * kHidden + k] = __float2half(value);
            }
        }
    }
};

struct Buffers {
    int active = 0;
    cudaStream_t stream = nullptr;
    unsigned long long* table = nullptr;
    int* ex_ids = nullptr;
    int* ex_off = nullptr;
    __half* act = nullptr;
    float* row_scale = nullptr;
    float* macro_g = nullptr;
    float* macro_u = nullptr;
    float* route_w = nullptr;
    std::uint8_t* gate_codes = nullptr;
    std::uint8_t* gate_scales = nullptr;
    std::uint8_t* up_codes = nullptr;
    std::uint8_t* up_scales = nullptr;
    float* h_control = nullptr;
    float* h_transpose = nullptr;
    std::size_t h_len = 0;
};

template<class T>
static T* device_alloc(std::size_t count, const char* what) {
    T* ptr = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), what);
    return ptr;
}

static void free_buffers(Buffers& b) {
    auto release = [](auto*& ptr) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
    };
    release(b.table); release(b.ex_ids); release(b.ex_off); release(b.act);
    release(b.row_scale); release(b.macro_g); release(b.macro_u); release(b.route_w);
    release(b.gate_codes); release(b.gate_scales); release(b.up_codes); release(b.up_scales);
    release(b.h_control); release(b.h_transpose);
    if (b.stream) cudaStreamDestroy(b.stream);
    b.stream = nullptr;
}

static void upload_buffers(Buffers& b, const Fixture& f) {
    b.active = f.active;
    b.h_len = static_cast<std::size_t>(f.active) * kInter + 16;
    const size_t weight_stride = static_cast<size_t>(kInter) * kRowBytes;
    const size_t scale_stride = static_cast<size_t>(kInter) * kScaleGroups;
    b.table = device_alloc<unsigned long long>(6 * kBankExperts, "table");
    b.ex_ids = device_alloc<int>(kBankExperts, "expert ids");
    b.ex_off = device_alloc<int>(kBankExperts + 1, "expert offsets");
    b.act = device_alloc<__half>(static_cast<size_t>(f.active) * kHidden, "activation");
    b.row_scale = device_alloc<float>(f.active, "row scale");
    b.macro_g = device_alloc<float>(f.active, "macro gate");
    b.macro_u = device_alloc<float>(f.active, "macro up");
    b.route_w = device_alloc<float>(f.active, "route weight");
    b.gate_codes = device_alloc<std::uint8_t>(f.gate_codes.size(), "gate codes");
    b.gate_scales = device_alloc<std::uint8_t>(f.gate_scales.size(), "gate scales");
    b.up_codes = device_alloc<std::uint8_t>(f.up_codes.size(), "up codes");
    b.up_scales = device_alloc<std::uint8_t>(f.up_scales.size(), "up scales");
    b.h_control = device_alloc<float>(b.h_len, "control H");
    b.h_transpose = device_alloc<float>(b.h_len, "transpose H");
    cuda_check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), "stream");

    std::vector<int> ids(kBankExperts), offsets(kBankExperts + 1, 0);
    std::vector<int> selected_flag(kBankExperts, 0);
    for (int i = 0; i < f.active; ++i) selected_flag[kSelected[i]] = 1;
    int live = 0;
    for (int e = 0; e < kBankExperts; ++e) {
        ids[e] = e;
        offsets[e] = live;
        live += selected_flag[e];
    }
    offsets[kBankExperts] = live;
    cuda_check(cudaMemcpy(b.ex_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice),
               "expert ids upload");
    cuda_check(cudaMemcpy(b.ex_off, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice),
               "expert offsets upload");

    std::vector<unsigned long long> table(6 * kBankExperts, 0);
    for (int e = 0; e < kBankExperts; ++e) {
        int selected = 0;
        while (selected < f.active && kSelected[selected] != e) ++selected;
        if (selected == f.active) selected = 0;
        const int payload = kPayload[selected];
        table[e] = reinterpret_cast<unsigned long long>(
            b.gate_codes + static_cast<size_t>(payload) * weight_stride);
        table[kBankExperts + e] = reinterpret_cast<unsigned long long>(
            b.gate_scales + static_cast<size_t>(payload) * scale_stride);
        table[4 * kBankExperts + e] = reinterpret_cast<unsigned long long>(
            b.up_codes + static_cast<size_t>(payload) * weight_stride);
        table[5 * kBankExperts + e] = reinterpret_cast<unsigned long long>(
            b.up_scales + static_cast<size_t>(payload) * scale_stride);
    }
    cuda_check(cudaMemcpy(b.table, table.data(), table.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice),
               "table upload");
    cuda_check(cudaMemcpy(b.act, f.act.data(), f.act.size() * sizeof(__half), cudaMemcpyHostToDevice),
               "activation upload");
    cuda_check(cudaMemcpy(b.row_scale, f.row_scale.data(), f.row_scale.size() * sizeof(float), cudaMemcpyHostToDevice),
               "row scale upload");
    cuda_check(cudaMemcpy(b.macro_g, f.macro_g.data(), f.macro_g.size() * sizeof(float), cudaMemcpyHostToDevice),
               "macro gate upload");
    cuda_check(cudaMemcpy(b.macro_u, f.macro_u.data(), f.macro_u.size() * sizeof(float), cudaMemcpyHostToDevice),
               "macro up upload");
    cuda_check(cudaMemcpy(b.route_w, f.route_w.data(), f.route_w.size() * sizeof(float), cudaMemcpyHostToDevice),
               "route upload");
    cuda_check(cudaMemcpy(b.gate_codes, f.gate_codes.data(), f.gate_codes.size(), cudaMemcpyHostToDevice),
               "gate codes upload");
    cuda_check(cudaMemcpy(b.gate_scales, f.gate_scales.data(), f.gate_scales.size(), cudaMemcpyHostToDevice),
               "gate scales upload");
    cuda_check(cudaMemcpy(b.up_codes, f.up_codes.data(), f.up_codes.size(), cudaMemcpyHostToDevice),
               "up codes upload");
    cuda_check(cudaMemcpy(b.up_scales, f.up_scales.data(), f.up_scales.size(), cudaMemcpyHostToDevice),
               "up scales upload");
}

static void reset_output(float* output, std::size_t len, cudaStream_t stream) {
    std::vector<std::uint32_t> guard(len, kGuardBits);
    cuda_check(cudaMemcpyAsync(output, guard.data(), guard.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, stream), "H guard reset");
    cuda_check(cudaStreamSynchronize(stream), "H guard reset sync");
}

static std::vector<std::uint32_t> read_bits(const float* output, std::size_t len,
                                            cudaStream_t stream) {
    std::vector<float> values(len);
    cuda_check(cudaMemcpyAsync(values.data(), output, len * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "H readback");
    cuda_check(cudaStreamSynchronize(stream), "H readback sync");
    std::vector<std::uint32_t> bits(len);
    for (std::size_t i = 0; i < len; ++i)
        std::memcpy(&bits[i], &values[i], sizeof(std::uint32_t));
    return bits;
}

static void check_finite_and_guard(const char* arm, const std::vector<std::uint32_t>& bits,
                                   int active) {
    const std::size_t live = static_cast<std::size_t>(active) * kInter;
    for (std::size_t i = 0; i < live; ++i) {
        float value = 0.0f;
        std::memcpy(&value, &bits[i], sizeof(value));
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(arm) + " nonfinite H at " + std::to_string(i));
        }
    }
    for (std::size_t i = live; i < bits.size(); ++i) {
        if (bits[i] != kGuardBits) {
            throw std::runtime_error(std::string(arm) + " H guard changed at " + std::to_string(i));
        }
    }
}

static void require_equal(const char* arm, const std::vector<std::uint32_t>& reference,
                          const std::vector<std::uint32_t>& actual, int active) {
    if (reference.size() != actual.size()) throw std::runtime_error("H size mismatch");
    const std::size_t live = static_cast<std::size_t>(active) * kInter;
    for (std::size_t i = 0; i < live; ++i) {
        if (reference[i] != actual[i]) {
            throw std::runtime_error(std::string(arm) + " H bit mismatch at "
                                     + std::to_string(i) + " ref="
                                     + std::to_string(reference[i]) + " got="
                                     + std::to_string(actual[i]));
        }
    }
    check_finite_and_guard(arm, actual, active);
}

struct Geometry {
    int sms = 1;
    int control_occ = 1;
    int transpose_occ = 1;
    int control_grid = 1;
    int transpose_grid = 1;
};

static Geometry geometry() {
    Geometry g;
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get device");
    cuda_check(cudaDeviceGetAttribute(&g.sms, cudaDevAttrMultiProcessorCount, device), "SM count");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &g.control_occ, moe_kq_sktail_gu_kernel<QT_NVFP4_MODELOPT, false, true>,
                   128, kPackedLutBytes), "control occupancy");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &g.transpose_occ, dsv4_gu_transpose_kernel<true>, 64,
                   kPackedLutBytes), "transpose occupancy");
    g.control_grid = g.sms * std::max(g.control_occ, 1);
    g.transpose_grid = g.sms * std::max(g.transpose_occ, 1);
    return g;
}

enum class Arm { Control, Transpose };

static const char* arm_name(Arm arm) {
    return arm == Arm::Control ? "current-gu-half2" : "transpose";
}

static void flush_l2(cudaStream_t stream, std::uint8_t* buffer, std::size_t bytes) {
    cuda_check(cudaMemsetAsync(buffer, 0xA5, bytes, stream), "L2 flush");
    cuda_check(cudaStreamSynchronize(stream), "L2 flush sync");
}

struct TimedOutput {
    float ms = 0.0f;
    std::vector<std::uint32_t> bits;
};

static TimedOutput run_arm(Buffers& b, const Geometry& g, Arm arm,
                           std::uint8_t* flush_buffer, std::size_t flush_bytes) {
    float* output = arm == Arm::Control ? b.h_control : b.h_transpose;
    reset_output(output, b.h_len, b.stream);
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event start");
    cuda_check(cudaEventCreate(&stop), "event stop");
    const unsigned long long before = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    cuda_check(cudaEventRecord(start, b.stream), "event record start");
    if (arm == Arm::Control) {
        check_rc("current GU half2", memra_moe_kq_gemm_sk_gu_half2(
            b.table, kBankExperts, b.ex_ids, b.act, b.h_control, b.row_scale,
            b.macro_g, b.macro_u, b.route_w, b.ex_off, kBankExperts,
            kHidden, kInter, 6.0f, kRowBytes, reinterpret_cast<void*>(b.stream)));
    } else {
        dsv4_gu_transpose_kernel<true><<<g.transpose_grid, dim3(32, 2, 1),
                                          kPackedLutBytes, b.stream>>>(
            b.table, kBankExperts, b.ex_ids, kRowBytes, b.act, b.h_transpose,
            b.row_scale, b.macro_g, b.macro_u, b.route_w, b.ex_off,
            kBankExperts, kHidden, kInter, -1, 6.0f);
        cuda_check(cudaGetLastError(), "transpose launch");
    }
    cuda_check(cudaEventRecord(stop, b.stream), "event record stop");
    cuda_check(cudaEventSynchronize(stop), "event stop sync");
    const unsigned long long after = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    const unsigned long long delta = after - before;
    if (arm == Arm::Control && delta != 1) {
        throw std::runtime_error("current control enqueue delta was not one");
    }
    if (arm == Arm::Transpose && delta != 0) {
        throw std::runtime_error("transpose changed current control enqueue counter");
    }
    TimedOutput result;
    cuda_check(cudaEventElapsedTime(&result.ms, start, stop), "event elapsed");
    result.bits = read_bits(output, b.h_len, b.stream);
    check_finite_and_guard(arm_name(arm), result.bits, b.active);
    cuda_check(cudaEventDestroy(start), "event start destroy");
    cuda_check(cudaEventDestroy(stop), "event stop destroy");
    return result;
}

static void run_case(int active, bool anchors, std::uint8_t* flush_buffer,
                     std::size_t flush_bytes) {
    Fixture fixture(active, anchors);
    Buffers buffers;
    buffers.active = active;
    try {
        upload_buffers(buffers, fixture);
        const Geometry g = geometry();
        const std::size_t prefix_bytes = sizeof(int) * (SK_MAX_G + 1);
        const std::size_t dynamic_lut_bytes = kPackedLutBytes;
        // Current GU has one Bs[SK_BN][SKT_STRIDE] bank, not three staged B
        // banks.  The transpose candidate has three Bs[kTransposeBRows]
        // stages because it sequentially reuses one G/U weight-A bank.
        const std::size_t control_declared_shared =
            prefix_bytes + sizeof(__half) *
                (SKT_STAGES * SK_BM * SKT_STRIDE + SK_BN * SKT_STRIDE);
        const std::size_t transpose_declared_shared =
            prefix_bytes + sizeof(__half) *
                (SKT_STAGES * kTransposeBm * SKT_STRIDE
                 + SKT_STAGES * kTransposeBRows * SKT_STRIDE);
        std::printf("CASE active=%d fixture=%s control_grid=%d transpose_grid=%d "
                    "control_occ=%d transpose_occ=%d control_static_prefix=%zu "
                    "transpose_static_prefix=%zu dynamic_lut_bytes=%zu "
                    "control_declared_shared=%zu transpose_declared_shared=%zu\n",
                    active, anchors ? "anchors" : "random", g.control_grid,
                    g.transpose_grid, g.control_occ, g.transpose_occ,
                    control_declared_shared, transpose_declared_shared,
                    dynamic_lut_bytes, control_declared_shared + dynamic_lut_bytes,
                    transpose_declared_shared + dynamic_lut_bytes);

        // Warm both real arms.  Warm output is still checked, but it is not
        // included in scored timing.
        TimedOutput warm_control = run_arm(buffers, g, Arm::Control, flush_buffer, flush_bytes);
        TimedOutput warm_transpose = run_arm(buffers, g, Arm::Transpose, flush_buffer, flush_bytes);
        require_equal("warm transpose", warm_control.bits, warm_transpose.bits, active);
        std::printf("WARMUP active=%d fixture=%s checked=true scored=false\n",
                    active, anchors ? "anchors" : "random");

        const unsigned long long control_before = memra_moe_kq_gemm_sk_gu_half2_dispatches();
        double control_sum = 0.0;
        double transpose_sum = 0.0;
        for (int cycle = 0; cycle < 3; ++cycle) {
            TimedOutput control_a = run_arm(buffers, g, Arm::Control, flush_buffer, flush_bytes);
            require_equal("current repeat", warm_control.bits, control_a.bits, active);
            control_sum += control_a.ms;

            TimedOutput transpose_b = run_arm(buffers, g, Arm::Transpose,
                                              flush_buffer, flush_bytes);
            require_equal("transpose", warm_control.bits, transpose_b.bits, active);
            transpose_sum += transpose_b.ms;

            TimedOutput transpose_b2 = run_arm(buffers, g, Arm::Transpose,
                                               flush_buffer, flush_bytes);
            require_equal("transpose repeat", warm_control.bits, transpose_b2.bits, active);
            transpose_sum += transpose_b2.ms;

            TimedOutput control_a2 = run_arm(buffers, g, Arm::Control, flush_buffer, flush_bytes);
            require_equal("current repeat 2", warm_control.bits, control_a2.bits, active);
            control_sum += control_a2.ms;
        }
        const unsigned long long control_after = memra_moe_kq_gemm_sk_gu_half2_dispatches();
        if (control_after - control_before != 6) {
            throw std::runtime_error("current control dispatch delta was not six");
        }
        std::printf("ABBA active=%d fixture=%s cycles=3 current_samples=6 transpose_samples=6 "
                    "current_mean_ms=%.6f transpose_mean_ms=%.6f "
                    "flush_2x_l2_before_every_scored=true h_bit_exact_finite_guards_every_row=true "
                    "within_arm_repeat=true\n",
                    active, anchors ? "anchors" : "random",
                    control_sum / 6.0, transpose_sum / 6.0);
        free_buffers(buffers);
    } catch (...) {
        free_buffers(buffers);
        throw;
    }
}

} // namespace dsv4_gu_transpose

int main() {
    try {
        int device = 0;
        dsv4_gu_transpose::cuda_check(
            cudaGetDevice(&device), "get device");
        int l2 = 0;
        dsv4_gu_transpose::cuda_check(
            cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device), "L2 size");
        const std::size_t flush_bytes = std::max<std::size_t>(
            2 * static_cast<std::size_t>(l2), 1u << 20);
        std::uint8_t* flush_buffer = nullptr;
        dsv4_gu_transpose::cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&flush_buffer), flush_bytes), "flush alloc");
        std::printf("protocol compiler=/usr/local/cuda-13.1/bin/nvcc arch=compute_120a,code=sm_120a "
                    "opt=-O3 source=tools/dsv4-gu-transpose-gate.cu "
                    "control=memra_moe_kq_gemm_sk_gu_half2 "
                    "candidate=dsv4_gu_transpose_kernel<108,true> "
                    "fixture=bank128_hidden4096_inter2048 active=1,3,4,6 "
                    "flush_2x_l2=%zu ABBA_cycles=3\n", flush_bytes);
        for (const int active : {1, 3, 4, 6}) {
            dsv4_gu_transpose::run_case(active, true, flush_buffer, flush_bytes);
        }
        std::puts("ANCHOR_PREFLIGHT_PASS all active counts exact H bits");
        for (const int active : {1, 3, 4, 6}) {
            dsv4_gu_transpose::run_case(active, false, flush_buffer, flush_bytes);
        }
        dsv4_gu_transpose::cuda_check(cudaFree(flush_buffer), "flush free");
        std::puts("PASS GU transpose matched F16-MMA control; anchor preflight, random H-bit "
                  "identity, finite/guard checks and 3-cycle cold ABBA completed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
