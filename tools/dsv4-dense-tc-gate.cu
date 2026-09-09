// Standalone DSV4 dense FP8-control versus BF16-MMA output-neuron probe.
//
// Numeric classes:
//   dsv4_gemv_fp8_m_f32acc        actual current engine C-ABI control entry
//   dsv4_dense_bf16_mma_f32acc_m1 FP8->BF16 tile decode, BF16 input, MMA f32 accumulate
//
// The candidate is intentionally NOT a bit-identity arm: the tensor-core reduction
// order differs from the control's per-thread serial chain plus 128-leaf tree. FP8
// codes/scales remain resident; the candidate decodes each BF16 A tile on the fly.
// The loader proof is separate: finite E4M3 * exact power-of-two scale must be
// exactly representable in BF16 before the candidate is admitted.
//
// Build only in this lane:
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 -fmad=false \
//     -Xcompiler=-ffp-contract=off \
//     -gencode arch=compute_120a,code=sm_120a \
//     -lcublasLt -lcublas \
//     -o target/dsv4-dense-tc-gate-r3 tools/dsv4-dense-tc-gate.cu
//
// No engine runtime/FFI integration; the current CUDA TU is included only as the
// control source. The default geometry is the matched q_b receipt (rows=32768,
// k=1024). Optional arguments are rows k flush_mb abs_tol rel_tol; --basis
// runs independent K=16/K=32 operand/layout checks.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "../crates/memra-engine/cu/mma_tile.cuh"
#include "../crates/memra-engine/cu/dsv4_gpu.cu"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dsv4_dense_tc_gate {

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::size_t;

constexpr int kDefaultRows = 32768;
constexpr int kDefaultK = 1024;
constexpr float kDefaultAbsTol = 0.25f;
constexpr float kDefaultRelTol = 1.0e-3f;

static void check(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(rc));
        std::exit(2);
    }
}

static void check_rc(int rc, const char* what) {
    if (rc != 0) {
        std::fprintf(stderr, "%s: rc=%d\n", what, rc);
        std::exit(2);
    }
}

// Shared by this probe and the R4/R7/R8/R9 include-based drivers. Pin once
// before any CUDA call/timing, so the control's measured envelope is unchanged.
static void pin_control_policy() {
    check_rc(::memra_dsv4_dense_fast_set_for_gate(0), "dense-fast control OFF");
    if (::memra_dsv4_dense_fast_enabled_for_gate() != 0) {
        std::fprintf(stderr, "dense-fast control override did not hold\n");
        std::exit(2);
    }
}

static float e4m3(uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v;
    if (exp == 0u) {
        v = static_cast<float>(man) * 0x1p-9f;
    } else {
        const uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&v, &bits, sizeof(v));
    }
    return (x & 0x80u) ? -v : v;
}

// The control is the actual current engine entry point, included above rather
// than reimplemented here. This keeps the comparison tied to the live source.
extern "C" int memra_dsv4_gemv_fp8_m(const void*, const float*, int, const void*, float*,
                                     int, int, int, int, int, void*);

static int launch_current_fp8(const uint8_t* w, const float* sc, int sc_cols,
                              const uint16_t* x, float* y, int rows, int k,
                              cudaStream_t stream) {
    return ::memra_dsv4_gemv_fp8_m(
        w, sc, sc_cols, x, y, 1, rows, k, k, rows, stream);
}

// BF16 output-neuron candidate. Each warp computes one 16x8 tile where all eight
// B columns are the same input vector. This is intentionally a named reduction-order
// class: output column zero is the result, columns 1..7 are duplicated work.
static __device__ __forceinline__ void load_a_tile(
        bw::ATile_m16k16_bf16& tile, const __nv_bfloat16* base, int stride_bf16) {
    // Use the in-tree validated helper and pass its physical u32-pair stride.
    bw::load_ldmatrix_A(tile, base, stride_bf16 / 2);
}

static __device__ __forceinline__ void load_b_tile(
        bw::BTile_n8k16_bf16& tile, const __nv_bfloat16* base, int stride_bf16) {
    bw::ATile_m16k16_bf16 trans;
    // The x4.trans helper returns a 16x16 transposed fragment. As in the
    // validated attention path, x0/x2 are the first n=8 half; x1/x3 are the
    // second n=8 half. The candidate uses only the first half.
    bw::load_ldmatrix_A_trans(trans, base, stride_bf16 / 2);
    tile.x[0] = trans.x[0];
    tile.x[1] = trans.x[2];
}

__global__ void bf16_mma_output_neuron_m1(const uint8_t* __restrict__ codes,
                                          const float* __restrict__ scales, int sc_cols,
                                          const __nv_bfloat16* __restrict__ x,
                                          float* __restrict__ y, int rows, int k) {
    __shared__ __nv_bfloat16 a_s[16 * 16];
    __shared__ __nv_bfloat16 b_s[16 * 16];
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 16;
    bw::CTile_m16n8_f32 acc{};
    acc.x[0] = acc.x[1] = acc.x[2] = acc.x[3] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int index = lane; index < 16 * 16; index += 32) {
            const int r = index / 16;
            const int kk = index % 16;
            const int row = row_base + r;
            if (row < rows) {
                const int col = k0 + kk;
                const uint8_t code = codes[static_cast<long>(row) * k + col];
                const float scale = scales[(row >> 7) * sc_cols + (col >> 7)];
                a_s[index] = __float2bfloat16(dsv4_e4m3(code) * scale);
            } else {
                a_s[index] = __float2bfloat16(0.0f);
            }
        }
        // ldmatrix.x4.trans reads a full 16x16 source tile even though this
        // output-neuron candidate consumes only its first n=8 half. Initialize
        // every row so no lane observes uninitialized shared memory.
        for (int index = lane; index < 16 * 16; index += 32) {
            const int kk = index / 16;
            const int col = index % 16;
            // The transpose loader follows the validated attention contract:
            // source rows are K and source columns are N. Replicate x[k] over
            // the N columns, rather than storing the transposed [N][K] view.
            b_s[kk * 16 + col] = x[k0 + kk];
        }
        __syncwarp();
        bw::ATile_m16k16_bf16 a;
        bw::BTile_n8k16_bf16 b;
        load_a_tile(a, a_s, 16);
        load_b_tile(b, b_s, 16);
        bw::mma_m16n8k16_bf16(acc, a, b);
    }
    // The fragment-to-lane map is fixed by the same in-tree tile port. Store
    // only the first duplicated N column; every N column has the same input.
    for (int l = 0; l < 4; ++l) {
        if (bw::CTile_m16n8_f32::get_j(l) == 0) {
            const int row = row_base + bw::CTile_m16n8_f32::get_i(l);
            if (row < rows) y[row] = acc.x[l];
        }
    }
}

extern "C" int launch_bf16_mma_output_neuron_m1(
        const uint8_t* codes, const float* scales, int sc_cols,
        const __nv_bfloat16* x, float* y, int rows, int k, cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || k % 16 != 0 || sc_cols <= 0) return 40020;
    const unsigned blocks = static_cast<unsigned>((rows + 15) / 16);
    bf16_mma_output_neuron_m1<<<blocks, 32, 0, stream>>>(
        codes, scales, sc_cols, x, y, rows, k);
    return static_cast<int>(cudaGetLastError());
}

// Small independent operand/layout diagnostic. A and B are supplied in the
// natural row-major source forms [M=16][K] and [K][N=16], respectively. The
// kernel stores every C element, so this catches A/B fragment orientation and
// the CTile lane map before the output-neuron reduction is admitted.
__global__ void bf16_mma_basis(const __nv_bfloat16* a_src,
                               const __nv_bfloat16* b_src, float* out, int k) {
    __shared__ __nv_bfloat16 a_s[16 * 16];
    __shared__ __nv_bfloat16 b_s[16 * 16];
    const int lane = threadIdx.x & 31;
    bw::CTile_m16n8_f32 acc{};
    acc.x[0] = acc.x[1] = acc.x[2] = acc.x[3] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 16) {
        for (int index = lane; index < 16 * 16; index += 32) {
            const int row = index / 16;
            const int kk = index % 16;
            a_s[index] = (k0 + kk < k) ? a_src[row * k + k0 + kk]
                                       : __float2bfloat16(0.0f);
        }
        for (int index = lane; index < 16 * 16; index += 32) {
            const int kk = index / 16;
            const int col = index % 16;
            b_s[index] = (k0 + kk < k) ? b_src[(k0 + kk) * 16 + col]
                                       : __float2bfloat16(0.0f);
        }
        __syncwarp();
        bw::ATile_m16k16_bf16 a;
        bw::BTile_n8k16_bf16 b;
        load_a_tile(a, a_s, 16);
        load_b_tile(b, b_s, 16);
        bw::mma_m16n8k16_bf16(acc, a, b);
    }
    for (int l = 0; l < 4; ++l) {
        const int row = bw::CTile_m16n8_f32::get_i(l);
        const int col = bw::CTile_m16n8_f32::get_j(l);
        if (row < 16 && col < 8) out[row * 8 + col] = acc.x[l];
    }
}

__global__ void flush_words(uint32_t* data, size_t words) {
    uint32_t v = 0x9e3779b9u + static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        v ^= data[i] + static_cast<uint32_t>(i * 0x85ebca6bu);
        data[i] = v;
    }
}

struct HostData {
    int rows;
    int k;
    int sc_cols;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<uint16_t> input;
    std::vector<uint16_t> shadow;
    std::vector<float> control;
    std::vector<float> candidate;
};

static HostData make_data(int rows, int k) {
    HostData d{rows, k, (k + 127) / 128};
    const int scale_rows = (rows + 127) / 128;
    d.codes.resize(static_cast<size_t>(rows) * k);
    d.scales.resize(static_cast<size_t>(scale_rows) * d.sc_cols);
    d.input.resize(k);
    d.shadow.resize(static_cast<size_t>(rows) * k);
    d.control.resize(rows);
    d.candidate.resize(rows);
    for (size_t i = 0; i < d.codes.size(); ++i) {
        uint8_t code = static_cast<uint8_t>(1u + ((i * 37u + i / 257u * 11u) % 0x7eu));
        if ((code & 0x7fu) == 0x7fu) code = 0x7eu;
        if (i & 1u) code |= 0x80u;
        d.codes[i] = code;
    }
    for (size_t i = 0; i < d.scales.size(); ++i)
        d.scales[i] = std::ldexp(1.0f, static_cast<int>(i % 9u) - 4);
    for (int i = 0; i < k; ++i) {
        const uint16_t bits = static_cast<uint16_t>(0x3f00u + ((i * 29u) & 0x7fu));
        d.input[i] = bits;
    }
    size_t exact = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < d.codes.size(); ++i) {
        const int row = static_cast<int>(i / k);
        const int col = static_cast<int>(i % k);
        const float value = e4m3(d.codes[i]) * d.scales[(row / 128) * d.sc_cols + col / 128];
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0xffffu) != 0u) {
            std::fprintf(stderr, "FP8 shadow is not BF16-exact at [%d,%d]\n", row, col);
            std::exit(3);
        }
        d.shadow[i] = static_cast<uint16_t>(bits >> 16);
        max_abs = std::max(max_abs, std::fabs(value));
        ++exact;
    }
    std::printf("SHADOW exact_bf16=%s elements=%zu max_abs=%.9g\n",
                exact == d.codes.size() ? "true" : "false", exact, max_abs);
    return d;
}

static float bf16_to_float(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

static int run_basis() {
    constexpr float kBasisAbsTol = 1.0e-3f;
    for (const int k : {16, 32}) {
        std::vector<uint16_t> a(static_cast<size_t>(16) * k);
        std::vector<uint16_t> b(static_cast<size_t>(k) * 16);
        std::vector<float> expected(16 * 8, 0.0f);
        std::vector<float> actual(16 * 8, 0.0f);
        for (int row = 0; row < 16; ++row) {
            for (int kk = 0; kk < k; ++kk) {
                a[static_cast<size_t>(row) * k + kk] = static_cast<uint16_t>(
                    0x3f80u + ((row * 7 + kk * 3 + 1) & 0x7fu));
            }
        }
        for (int kk = 0; kk < k; ++kk) {
            for (int col = 0; col < 16; ++col) {
                b[static_cast<size_t>(kk) * 16 + col] = static_cast<uint16_t>(
                    0x3e00u + ((kk * 5 + col * 3 + 7) & 0x7fu));
            }
        }
        for (int row = 0; row < 16; ++row) {
            for (int col = 0; col < 8; ++col) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; ++kk) {
                    sum += bf16_to_float(a[static_cast<size_t>(row) * k + kk])
                        * bf16_to_float(b[static_cast<size_t>(kk) * 16 + col]);
                }
                expected[row * 8 + col] = sum;
            }
        }

        __nv_bfloat16* d_a = nullptr;
        __nv_bfloat16* d_b = nullptr;
        float* d_out = nullptr;
        check(cudaMalloc(&d_a, a.size() * sizeof(uint16_t)), "basis A alloc");
        check(cudaMalloc(&d_b, b.size() * sizeof(uint16_t)), "basis B alloc");
        check(cudaMalloc(&d_out, actual.size() * sizeof(float)), "basis output alloc");
        check(cudaMemcpy(d_a, a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
              "basis A copy");
        check(cudaMemcpy(d_b, b.data(), b.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
              "basis B copy");
        cudaStream_t stream;
        check(cudaStreamCreate(&stream), "basis stream create");
        bf16_mma_basis<<<1, 32, 0, stream>>>(d_a, d_b, d_out, k);
        check(cudaGetLastError(), "basis launch");
        check(cudaStreamSynchronize(stream), "basis sync");
        check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "basis output copy");
        float max_abs = 0.0f;
        int worst = 0;
        bool pass = true;
        for (int i = 0; i < 16 * 8; ++i) {
            const float abs_err = std::fabs(actual[i] - expected[i]);
            if (!std::isfinite(actual[i]) || abs_err > max_abs) {
                max_abs = abs_err;
                worst = i;
            }
            if (!std::isfinite(actual[i]) || abs_err > kBasisAbsTol) pass = false;
        }
        std::printf("BASIS k=%d pass=%s max_abs=%.9g worst_row=%d worst_col=%d\n",
                    k, pass ? "true" : "false", max_abs, worst / 8, worst % 8);
        if (!pass) {
            std::fprintf(stderr,
                         "FAIL basis k=%d row=%d col=%d expected=%.9g actual=%.9g\n",
                         k, worst / 8, worst % 8, expected[worst], actual[worst]);
        }
        check(cudaStreamDestroy(stream), "basis stream destroy");
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_out);
        if (!pass) return 7;
    }
    std::printf("BASIS PASS k=16,32 independent_row_oracle=true\n");
    return 0;
}

static void flush(cudaStream_t stream, uint32_t* data, size_t words) {
    const unsigned blocks = 4096;
    flush_words<<<blocks, 256, 0, stream>>>(data, words);
    check(cudaGetLastError(), "flush launch");
    check(cudaStreamSynchronize(stream), "flush sync");
}

static double run_current(const HostData& h, uint8_t* w, float* sc, uint16_t* x,
                          float* y, uint32_t* flush_buf, size_t flush_words_n,
                          cudaStream_t stream) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    flush(stream, flush_buf, flush_words_n);
    check(cudaEventRecord(start, stream), "control start");
    check_rc(launch_current_fp8(w, sc, h.sc_cols, x, y, h.rows, h.k, stream), "control launch");
    check(cudaEventRecord(stop, stream), "control stop");
    check(cudaEventSynchronize(stop), "control sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "control elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static double run_candidate(const HostData& h, uint8_t* codes, float* scales,
                            __nv_bfloat16* x, float* y, uint32_t* flush_buf,
                            size_t flush_words_n, cudaStream_t stream) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    flush(stream, flush_buf, flush_words_n);
    check(cudaEventRecord(start, stream), "candidate start");
    check_rc(launch_bf16_mma_output_neuron_m1(
                  codes, scales, h.sc_cols, x, y, h.rows, h.k, stream),
              "candidate launch");
    check(cudaEventRecord(stop, stream), "candidate stop");
    check(cudaEventSynchronize(stop), "candidate sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "candidate elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static void record_scored_output(float* d_output, std::vector<float>& host,
                                 std::vector<float>& first, bool& have_first,
                                 int rows, const char* arm) {
    check(cudaMemcpy(host.data(), d_output, static_cast<size_t>(rows) * sizeof(float),
                     cudaMemcpyDeviceToHost), "scored output copy");
    for (int row = 0; row < rows; ++row) {
        if (!std::isfinite(host[row])) {
            std::fprintf(stderr, "FAIL nonfinite %s output at row %d\n", arm, row);
            std::exit(4);
        }
    }
    if (!have_first) {
        first = host;
        have_first = true;
        return;
    }
    if (std::memcmp(host.data(), first.data(), static_cast<size_t>(rows) * sizeof(float)) != 0) {
        for (int row = 0; row < rows; ++row) {
            uint32_t got_bits = 0, first_bits = 0;
            std::memcpy(&got_bits, &host[row], sizeof(got_bits));
            std::memcpy(&first_bits, &first[row], sizeof(first_bits));
            if (got_bits != first_bits) {
                std::fprintf(stderr,
                             "FAIL %s arm is not bit-repeatable at row %d: first=0x%08x got=0x%08x\n",
                             arm, row, first_bits, got_bits);
                break;
            }
        }
        std::exit(6);
    }
}

static void validate_numeric_band(const std::vector<float>& control,
                                  const std::vector<float>& candidate,
                                  int cycle, float abs_tol, float rel_tol,
                                  float& max_abs, float& max_rel) {
    for (size_t row = 0; row < control.size(); ++row) {
        const float abs_err = std::fabs(control[row] - candidate[row]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(control[row]));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > abs_tol || rel_err > rel_tol) {
            std::fprintf(stderr,
                         "FAIL declared numerical band at cycle %d row %zu: abs=%.9g rel=%.9g\n",
                         cycle, row, abs_err, rel_err);
            std::exit(5);
        }
    }
}

} // namespace dsv4_dense_tc_gate

int main(int argc, char** argv) {
    dsv4_dense_tc_gate::pin_control_policy();
    if (argc == 2 && std::strcmp(argv[1], "--check-controls") == 0) {
        std::puts("PASS dense_tc_control dense_fast=0 cpu_policy_only=true"); return 0;
    }
    using namespace dsv4_dense_tc_gate;
    if (argc > 1 && std::strcmp(argv[1], "--basis") == 0) return run_basis();
    const int rows = argc > 1 ? std::atoi(argv[1]) : kDefaultRows;
    const int k = argc > 2 ? std::atoi(argv[2]) : kDefaultK;
    const int flush_mb_arg = argc > 3 ? std::atoi(argv[3]) : 0;
    const float abs_tol = argc > 4 ? std::strtof(argv[4], nullptr) : kDefaultAbsTol;
    const float rel_tol = argc > 5 ? std::strtof(argv[5], nullptr) : kDefaultRelTol;
    if (rows <= 0 || k <= 0 || k % 16 != 0 || flush_mb_arg < 0 ||
        !std::isfinite(abs_tol) || !std::isfinite(rel_tol) || abs_tol < 0.0f || rel_tol < 0.0f) {
        std::fprintf(stderr, "usage: dsv4-dense-tc-gate [rows] [k] [flush_mb] [abs_tol] [rel_tol]\n");
        return 2;
    }
    int device = 0;
    int l2_bytes = 0;
    check(cudaGetDevice(&device), "get device");
    check(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device), "query L2");
    if (l2_bytes <= 0) {
        std::fprintf(stderr, "device reported no L2 cache size\n");
        return 2;
    }
    const size_t flush_bytes = flush_mb_arg > 0
        ? static_cast<size_t>(flush_mb_arg) * 1024 * 1024
        : static_cast<size_t>(l2_bytes) * 2;
    HostData h = make_data(rows, k);

    uint8_t* d_codes = nullptr;
    float* d_scales = nullptr;
    uint16_t* d_input = nullptr;
    float* d_control = nullptr;
    float* d_candidate = nullptr;
    uint32_t* d_flush = nullptr;
    const size_t flush_words_n = (flush_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    check(cudaMalloc(&d_codes, h.codes.size()), "codes alloc");
    check(cudaMalloc(&d_scales, h.scales.size() * sizeof(float)), "scales alloc");
    check(cudaMalloc(&d_input, h.input.size() * sizeof(uint16_t)), "input alloc");
    check(cudaMalloc(&d_control, h.control.size() * sizeof(float)), "control alloc");
    check(cudaMalloc(&d_candidate, h.candidate.size() * sizeof(float)), "candidate alloc");
    check(cudaMalloc(&d_flush, flush_words_n * sizeof(uint32_t)), "flush alloc");
    check(cudaMemset(d_flush, 0, flush_words_n * sizeof(uint32_t)), "flush init");
    check(cudaMemcpy(d_codes, h.codes.data(), h.codes.size(), cudaMemcpyHostToDevice), "codes copy");
    check(cudaMemcpy(d_scales, h.scales.data(), h.scales.size() * sizeof(float), cudaMemcpyHostToDevice), "scales copy");
    check(cudaMemcpy(d_input, h.input.data(), h.input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "input copy");
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream create");

    std::vector<double> control_samples;
    std::vector<double> candidate_samples;
    control_samples.reserve(6);
    candidate_samples.reserve(6);
    const double warm_control = run_current(
        h, d_codes, d_scales, d_input, d_control, d_flush, flush_words_n, stream);
    const double warm_candidate = run_candidate(
        h, d_codes, d_scales, reinterpret_cast<__nv_bfloat16*>(d_input), d_candidate,
        d_flush, flush_words_n, stream);
    std::printf("WARMUP control_ms=%.6f candidate_ms=%.6f\n", warm_control, warm_candidate);

    std::vector<float> control_first(rows);
    std::vector<float> candidate_first(rows);
    bool have_control_first = false;
    bool have_candidate_first = false;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    // Three ABBA cycles, with the same cold-cache flush before every arm.
    for (int cycle = 0; cycle < 3; ++cycle) {
        control_samples.push_back(run_current(
            h, d_codes, d_scales, d_input, d_control, d_flush, flush_words_n, stream));
        record_scored_output(d_control, h.control, control_first, have_control_first,
                             rows, "control");
        candidate_samples.push_back(run_candidate(
            h, d_codes, d_scales, reinterpret_cast<__nv_bfloat16*>(d_input), d_candidate,
            d_flush, flush_words_n, stream));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate_first,
                             rows, "candidate");
        validate_numeric_band(h.control, h.candidate, cycle, abs_tol, rel_tol,
                              max_abs, max_rel);
        candidate_samples.push_back(run_candidate(
            h, d_codes, d_scales, reinterpret_cast<__nv_bfloat16*>(d_input), d_candidate,
            d_flush, flush_words_n, stream));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate_first,
                             rows, "candidate");
        validate_numeric_band(h.control, h.candidate, cycle, abs_tol, rel_tol,
                              max_abs, max_rel);
        control_samples.push_back(run_current(
            h, d_codes, d_scales, d_input, d_control, d_flush, flush_words_n, stream));
        record_scored_output(d_control, h.control, control_first, have_control_first,
                             rows, "control");
    }
    auto mean = [](const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / static_cast<double>(values.size());
    };
    const double control_ms = mean(control_samples);
    const double candidate_ms = mean(candidate_samples);
    std::printf("CLASS control=dsv4_gemv_fp8_m_f32acc candidate=dsv4_dense_bf16_mma_f32acc_m1 rows=%d k=%d abba_cycles=3 l2_bytes=%d flush_bytes=%zu abs_tol=%.9g rel_tol=%.9g\n",
                rows, k, l2_bytes, flush_bytes, abs_tol, rel_tol);
    std::printf("RESULT control_ms=%.6f candidate_ms=%.6f ratio=%.6f max_abs=%.9g max_rel=%.9g\n",
                control_ms, candidate_ms, candidate_ms / control_ms, max_abs, max_rel);
    if (max_abs > abs_tol || max_rel > rel_tol) {
        std::fprintf(stderr, "FAIL declared numerical band: max_abs=%.9g max_rel=%.9g\n",
                     max_abs, max_rel);
        return 5;
    }
    check(cudaStreamDestroy(stream), "stream destroy");
    cudaFree(d_codes);
    cudaFree(d_scales);
    cudaFree(d_input);
    cudaFree(d_control);
    cudaFree(d_candidate);
    cudaFree(d_flush);
    std::printf("PASS scored_rows=%d control_runs=%zu candidate_runs=%zu bit_repeatability=true numeric_band=abs<=%.9g,rel<=%.9g\n",
                rows, control_samples.size(), candidate_samples.size(), abs_tol, rel_tol);
    return 0;
}
