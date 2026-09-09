// R4 standalone dense FP8 -> BF16 MMA gate.
//
// R3 proved the x4.trans orientation but lost badly on the target pair because
// it decoded one 16x16 tile with scalar FP8->f32->BF16 work and a barrier on
// every K=16 step. R4 keeps the same resident FP8 codes/scales and BF16
// f32-accumulate MMA numeric class, but changes the mechanism:
//
//   * four warps own 64 output rows per block;
//   * each warp stages a K=128 weight tile with vectorized uint2 code loads;
//   * FP8 values are converted through a shared BF16-bit LUT, then the exact
//     power-of-two scale exponent is applied to the BF16 exponent field;
//   * one block barrier protects each K=128 stage, followed by eight K=16
//     MMA steps with no per-step global decode or barrier.
//
// This file includes the frozen R3 gate for the actual current control entry
// and shared receipt helpers, while keeping the R3 source unchanged. It is a
// standalone compile/run gate only. No engine integration or BF16 weight
// residency is introduced.

#define main dsv4_dense_tc_gate_r3_main
#include "dsv4-dense-tc-gate.cu"
#undef main

namespace dsv4_dense_tc_gate_r4 {

using namespace dsv4_dense_tc_gate;
using std::int8_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::size_t;

constexpr int kWarps = 4;
constexpr int kRowsPerWarp = 16;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kTileK = 128;
constexpr int kTileN = 16;

static __device__ __forceinline__ uint16_t scaled_bf16_bits(uint16_t base,
                                                              int scale_exp) {
    if ((base & 0x7fffu) == 0u) return 0u;
    // E4M3 values in this gate are finite BF16-normal values. The model scale
    // is E8M0, so shifting the BF16 exponent is exactly the same BF16 byte
    // result as f32 multiply followed by __float2bfloat16.
    return static_cast<uint16_t>(static_cast<int>(base) + scale_exp * 128);
}

static __device__ __forceinline__ void load_a_r4(
        bw::ATile_m16k16_bf16& tile, const __nv_bfloat16* base,
        int stride_bf16, int lane) {
    int* regs = reinterpret_cast<int*>(tile.x);
    const uint32_t* row = reinterpret_cast<const uint32_t*>(base)
        + (lane % 16) * (stride_bf16 / 2) + (lane / 16) * 4;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

static __device__ __forceinline__ void load_b_r4(
        bw::BTile_n8k16_bf16& tile, const __nv_bfloat16* base,
        int stride_bf16, int lane) {
    bw::ATile_m16k16_bf16 trans;
    int* regs = reinterpret_cast<int*>(trans.x);
    const uint32_t* row = reinterpret_cast<const uint32_t*>(base)
        + (lane % 16) * (stride_bf16 / 2) + (lane / 16) * 4;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(regs[0]), "=r"(regs[2]), "=r"(regs[1]), "=r"(regs[3])
        : "r"(addr));
    tile.x[0] = trans.x[0];
    tile.x[1] = trans.x[2];
}

static __device__ __forceinline__ int c_row(int lane, int l) {
    return ((l / 2) * 8) + (lane / 4);
}

static __device__ __forceinline__ int c_col(int lane, int l) {
    return ((lane % 4) * 2) + (l % 2);
}

__global__ void bf16_mma_output_neuron_k128_packed(
        const uint8_t* __restrict__ codes,
        const int8_t* __restrict__ scale_exp,
        const uint16_t* __restrict__ fp8_bf16_lut,
        const __nv_bfloat16* __restrict__ x,
        float* __restrict__ y, int rows, int k, int sc_cols) {
    __shared__ uint16_t lut[256];
    __shared__ __nv_bfloat16 a_s[kWarps][kRowsPerWarp * kTileK];
    __shared__ __nv_bfloat16 b_s[kTileK * kTileN];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;

    for (int i = tid; i < 256; i += blockDim.x) lut[i] = fp8_bf16_lut[i];
    __syncthreads();

    bw::CTile_m16n8_f32 acc{};
    acc.x[0] = acc.x[1] = acc.x[2] = acc.x[3] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += kTileK) {
        // B source is natural row-major [K][N], matching the validated
        // x4.trans contract. Every K value is replicated across N=16.
        if (tid < kTileK) {
            const int kk = tid;
            const __nv_bfloat16 value = x[k0 + kk];
#pragma unroll
            for (int col = 0; col < kTileN; ++col)
                b_s[kk * kTileN + col] = value;
        }

        // Two lanes own each output row. Each lane issues eight 8-byte code
        // loads, covering the row's 128 codes without scalar global loads.
        const int row_local = lane / 2;
        const int half = lane & 1;
        const int row = row_base + warp * kRowsPerWarp + row_local;
        const int scale_col = k0 >> 7;
        const int exponent = (row < rows)
            ? static_cast<int>(scale_exp[(row >> 7) * sc_cols + scale_col]) : 0;
        for (int chunk = half * 8; chunk < kTileK; chunk += 16) {
            uint2 packed = make_uint2(0u, 0u);
            if (row < rows) {
                packed = *reinterpret_cast<const uint2*>(
                    codes + static_cast<long>(row) * k + k0 + chunk);
            }
            const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint8_t code = static_cast<uint8_t>(
                    (words[j >> 2] >> ((j & 3) * 8)) & 0xffu);
                const uint16_t bits = scaled_bf16_bits(lut[code], exponent);
                a_s[warp][row_local * kTileK + chunk + j] =
                    __ushort_as_bfloat16(bits);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            bw::ATile_m16k16_bf16 a;
            bw::BTile_n8k16_bf16 b;
            load_a_r4(a, &a_s[warp][0] + kk, kTileK, lane);
            load_b_r4(b, &b_s[kk * kTileN], kTileN, lane);
            bw::mma_m16n8k16_bf16(acc, a, b);
        }
        // Keep the staged tile reusable only after every warp has completed
        // its MMA chain. The output is stored once, after all K=128 tiles.
        __syncthreads();
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int col = c_col(lane, l);
        if (col == 0) {
            const int row_out = row_base + warp * kRowsPerWarp + c_row(lane, l);
            if (row_out < rows) y[row_out] = acc.x[l];
        }
    }
}

extern "C" int launch_bf16_mma_output_neuron_k128_packed(
        const uint8_t* codes, const int8_t* scale_exp, const uint16_t* fp8_bf16_lut,
        const __nv_bfloat16* x, float* y, int rows, int k, int sc_cols,
        cudaStream_t stream) {
    if (rows <= 0 || k <= 0 || k % kTileK != 0 || sc_cols <= 0) return 40020;
    const unsigned blocks = static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    bf16_mma_output_neuron_k128_packed<<<blocks, kWarps * 32, 0, stream>>>(
        codes, scale_exp, fp8_bf16_lut, x, y, rows, k, sc_cols);
    return static_cast<int>(cudaGetLastError());
}

static std::vector<uint16_t> make_fp8_bf16_lut() {
    std::vector<uint16_t> lut(256);
    for (int code = 0; code < 256; ++code) {
        const float value = e4m3(static_cast<uint8_t>(code));
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0xffffu) != 0u) {
            std::fprintf(stderr, "FP8 LUT is not BF16-exact at code 0x%02x\n", code);
            std::exit(3);
        }
        lut[code] = static_cast<uint16_t>(bits >> 16);
    }
    return lut;
}

static int scale_exponent(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffu) != 0u || (bits >> 31) != 0u || value == 0.0f) {
        std::fprintf(stderr, "R4 requires positive exact power-of-two scale, got %.9g\n", value);
        std::exit(3);
    }
    return static_cast<int>((bits >> 23) & 0xffu) - 127;
}

static std::vector<int8_t> make_scale_exponents(const HostData& h,
                                                 const std::vector<uint16_t>& lut) {
    std::vector<int8_t> exponents(h.scales.size());
    for (size_t i = 0; i < h.scales.size(); ++i)
        exponents[i] = static_cast<int8_t>(scale_exponent(h.scales[i]));
    for (size_t i = 0; i < h.codes.size(); ++i) {
        const int row = static_cast<int>(i / h.k);
        const int col = static_cast<int>(i % h.k);
        const uint16_t base = lut[h.codes[i]];
        const uint16_t got = (base & 0x7fffu) == 0u
            ? 0u
            : static_cast<uint16_t>(
                  static_cast<int>(base)
                  + static_cast<int>(exponents[(row / 128) * h.sc_cols + col / 128]) * 128);
        if (got != h.shadow[i]) {
            std::fprintf(stderr, "R4 BF16 byte proof failed at [%d,%d]: got=0x%04x want=0x%04x\n",
                         row, col, got, h.shadow[i]);
            std::exit(3);
        }
    }
    return exponents;
}

static double run_candidate_r4(const HostData& h, uint8_t* codes, int8_t* scale_exp,
                               uint16_t* lut, __nv_bfloat16* x, float* y,
                               uint32_t* flush_buf, size_t flush_words_n,
                               cudaStream_t stream) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "R4 event create");
    check(cudaEventCreate(&stop), "R4 event create");
    flush(stream, flush_buf, flush_words_n);
    check(cudaEventRecord(start, stream), "R4 candidate start");
    check_rc(launch_bf16_mma_output_neuron_k128_packed(
                  codes, scale_exp, lut, x, y, h.rows, h.k, h.sc_cols, stream),
              "R4 candidate launch");
    check(cudaEventRecord(stop, stream), "R4 candidate stop");
    check(cudaEventSynchronize(stop), "R4 candidate sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "R4 candidate elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static int run_basis_r4() {
    constexpr uint8_t kOneFp8 = 0x38u; // E4M3 +1.0
    constexpr uint16_t kOneBf16 = 0x3f80u;
    constexpr float kAbsTol = 0.25f;
    const std::vector<uint16_t> lut = make_fp8_bf16_lut();
    for (const int rows : {1, 64, 65, 129}) {
        for (const int k : {128, 256}) {
            const int sc_cols = (k + 127) / 128;
            const int scale_rows = (rows + 127) / 128;
            std::vector<uint8_t> codes(static_cast<size_t>(rows) * k, 0u);
            std::vector<uint16_t> input(k, kOneBf16);
            std::vector<int8_t> scale_exp(static_cast<size_t>(scale_rows) * sc_cols, 0);
            std::vector<float> expected(rows, 0.0f);
            std::vector<float> last_only(rows, 0.0f);
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < k; ++col) {
                    // Only the first K=128 tile contributes. For K=256 this
                    // makes the last-tile-only bug a zero-output control.
                    codes[static_cast<size_t>(row) * k + col] =
                        col < 128 ? kOneFp8 : 0u;
                    const float term = e4m3(codes[static_cast<size_t>(row) * k + col])
                        * bf16_to_float(input[col]);
                    expected[row] += term;
                    if (col >= k - 128) last_only[row] += term;
                }
            }

            uint8_t* d_codes = nullptr;
            int8_t* d_scale_exp = nullptr;
            uint16_t* d_lut = nullptr;
            uint16_t* d_input = nullptr;
            float* d_output = nullptr;
            check(cudaMalloc(&d_codes, codes.size()), "R4 basis codes alloc");
            check(cudaMalloc(&d_scale_exp, scale_exp.size() * sizeof(int8_t)),
                  "R4 basis scale alloc");
            check(cudaMalloc(&d_lut, lut.size() * sizeof(uint16_t)), "R4 basis LUT alloc");
            check(cudaMalloc(&d_input, input.size() * sizeof(uint16_t)), "R4 basis input alloc");
            check(cudaMalloc(&d_output, expected.size() * sizeof(float)),
                  "R4 basis output alloc");
            check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice),
                  "R4 basis codes copy");
            check(cudaMemcpy(d_scale_exp, scale_exp.data(), scale_exp.size() * sizeof(int8_t),
                             cudaMemcpyHostToDevice), "R4 basis scale copy");
            check(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice), "R4 basis LUT copy");
            check(cudaMemcpy(d_input, input.data(), input.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice), "R4 basis input copy");
            check(cudaMemset(d_output, 0, expected.size() * sizeof(float)),
                  "R4 basis output init");
            cudaStream_t stream;
            check(cudaStreamCreate(&stream), "R4 basis stream create");
            check_rc(launch_bf16_mma_output_neuron_k128_packed(
                          d_codes, d_scale_exp, d_lut,
                          reinterpret_cast<__nv_bfloat16*>(d_input), d_output,
                          rows, k, sc_cols, stream), "R4 basis launch");
            check(cudaStreamSynchronize(stream), "R4 basis sync");
            std::vector<float> actual(rows);
            check(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "R4 basis output copy");

            float max_abs = 0.0f;
            float last_err = 0.0f;
            int worst = 0;
            bool pass = true;
            for (int row = 0; row < rows; ++row) {
                const float full_err = std::fabs(actual[row] - expected[row]);
                const float only_last_err = std::fabs(actual[row] - last_only[row]);
                if (!std::isfinite(actual[row]) || full_err > max_abs) {
                    max_abs = full_err;
                    worst = row;
                }
                last_err = std::max(last_err, only_last_err);
                if (!std::isfinite(actual[row]) || full_err > kAbsTol) pass = false;
            }
            const float anchor = std::fabs(expected[0] - last_only[0]);
            if (k == 256) {
                if (anchor < 64.0f ||
                    std::fabs(actual[0] - last_only[0]) <=
                        std::fabs(actual[0] - expected[0])) {
                    std::fprintf(stderr,
                                 "FAIL R4 first-tile anchor rows=%d k=%d actual=%.9g full=%.9g last_only=%.9g\n",
                                 rows, k, actual[0], expected[0], last_only[0]);
                    pass = false;
                }
            }
            std::printf("R4_BASIS rows=%d k=%d pass=%s max_abs=%.9g last_only_err=%.9g anchor=%.9g worst_row=%d\n",
                        rows, k, pass ? "true" : "false", max_abs, last_err,
                        anchor, worst);
            if (!pass) {
                std::fprintf(stderr,
                             "FAIL R4 basis rows=%d k=%d row=%d expected=%.9g actual=%.9g\n",
                             rows, k, worst, expected[worst], actual[worst]);
            }
            check(cudaStreamDestroy(stream), "R4 basis stream destroy");
            cudaFree(d_codes);
            cudaFree(d_scale_exp);
            cudaFree(d_lut);
            cudaFree(d_input);
            cudaFree(d_output);
            if (!pass) return 7;
        }
    }
    std::printf("R4_BASIS PASS k=128,256 rows=1,64,65,129 independent_oracle=true\n");
    return 0;
}

} // namespace dsv4_dense_tc_gate_r4

int main(int argc, char** argv) {
    dsv4_dense_tc_gate::pin_control_policy();
    if (argc == 2 && std::strcmp(argv[1], "--check-controls") == 0) {
        std::puts("PASS dense_tc_control dense_fast=0 cpu_policy_only=true"); return 0;
    }
    using namespace dsv4_dense_tc_gate;
    using namespace dsv4_dense_tc_gate_r4;
    if (argc > 1 && std::strcmp(argv[1], "--basis") == 0) return run_basis_r4();
    const int rows = argc > 1 ? std::atoi(argv[1]) : kDefaultRows;
    const int k = argc > 2 ? std::atoi(argv[2]) : kDefaultK;
    const int flush_mb_arg = argc > 3 ? std::atoi(argv[3]) : 0;
    const float abs_tol = argc > 4 ? std::strtof(argv[4], nullptr) : kDefaultAbsTol;
    const float rel_tol = argc > 5 ? std::strtof(argv[5], nullptr) : kDefaultRelTol;
    if (rows <= 0 || k <= 0 || k % kTileK != 0 || flush_mb_arg < 0 ||
        !std::isfinite(abs_tol) || !std::isfinite(rel_tol) || abs_tol < 0.0f || rel_tol < 0.0f) {
        std::fprintf(stderr, "usage: dsv4-dense-tc-gate-r4 [rows] [k] [flush_mb] [abs_tol] [rel_tol]\n");
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
    const std::vector<uint16_t> fp8_lut = make_fp8_bf16_lut();
    const std::vector<int8_t> scale_exp = make_scale_exponents(h, fp8_lut);

    uint8_t* d_codes = nullptr;
    float* d_scales = nullptr;
    int8_t* d_scale_exp = nullptr;
    uint16_t* d_lut = nullptr;
    uint16_t* d_input = nullptr;
    float* d_control = nullptr;
    float* d_candidate = nullptr;
    uint32_t* d_flush = nullptr;
    const size_t flush_words_n = (flush_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    check(cudaMalloc(&d_codes, h.codes.size()), "codes alloc");
    check(cudaMalloc(&d_scales, h.scales.size() * sizeof(float)), "scales alloc");
    check(cudaMalloc(&d_scale_exp, scale_exp.size() * sizeof(int8_t)), "scale exp alloc");
    check(cudaMalloc(&d_lut, fp8_lut.size() * sizeof(uint16_t)), "FP8 LUT alloc");
    check(cudaMalloc(&d_input, h.input.size() * sizeof(uint16_t)), "input alloc");
    check(cudaMalloc(&d_control, h.control.size() * sizeof(float)), "control alloc");
    check(cudaMalloc(&d_candidate, h.candidate.size() * sizeof(float)), "candidate alloc");
    check(cudaMalloc(&d_flush, flush_words_n * sizeof(uint32_t)), "flush alloc");
    check(cudaMemset(d_flush, 0, flush_words_n * sizeof(uint32_t)), "flush init");
    check(cudaMemcpy(d_codes, h.codes.data(), h.codes.size(), cudaMemcpyHostToDevice), "codes copy");
    check(cudaMemcpy(d_scales, h.scales.data(), h.scales.size() * sizeof(float), cudaMemcpyHostToDevice), "scales copy");
    check(cudaMemcpy(d_scale_exp, scale_exp.data(), scale_exp.size() * sizeof(int8_t), cudaMemcpyHostToDevice), "scale exp copy");
    check(cudaMemcpy(d_lut, fp8_lut.data(), fp8_lut.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "FP8 LUT copy");
    check(cudaMemcpy(d_input, h.input.data(), h.input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "input copy");
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "stream create");

    std::vector<double> control_samples;
    std::vector<double> candidate_samples;
    control_samples.reserve(6);
    candidate_samples.reserve(6);
    const double warm_control = run_current(
        h, d_codes, d_scales, d_input, d_control, d_flush, flush_words_n, stream);
    const double warm_candidate = run_candidate_r4(
        h, d_codes, d_scale_exp, d_lut, reinterpret_cast<__nv_bfloat16*>(d_input),
        d_candidate, d_flush, flush_words_n, stream);
    std::printf("WARMUP control_ms=%.6f candidate_ms=%.6f\n", warm_control, warm_candidate);

    std::vector<float> control_first(rows);
    std::vector<float> candidate_first(rows);
    bool have_control_first = false;
    bool have_candidate_first = false;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int cycle = 0; cycle < 3; ++cycle) {
        control_samples.push_back(run_current(
            h, d_codes, d_scales, d_input, d_control, d_flush, flush_words_n, stream));
        record_scored_output(d_control, h.control, control_first, have_control_first,
                             rows, "control");
        candidate_samples.push_back(run_candidate_r4(
            h, d_codes, d_scale_exp, d_lut, reinterpret_cast<__nv_bfloat16*>(d_input),
            d_candidate, d_flush, flush_words_n, stream));
        record_scored_output(d_candidate, h.candidate, candidate_first, have_candidate_first,
                             rows, "candidate");
        validate_numeric_band(h.control, h.candidate, cycle, abs_tol, rel_tol,
                              max_abs, max_rel);
        candidate_samples.push_back(run_candidate_r4(
            h, d_codes, d_scale_exp, d_lut, reinterpret_cast<__nv_bfloat16*>(d_input),
            d_candidate, d_flush, flush_words_n, stream));
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
    std::printf("CLASS control=dsv4_gemv_fp8_m_f32acc candidate=dsv4_dense_bf16_mma_f32acc_k128_packed rows=%d k=%d warps=%d tile_k=%d abba_cycles=3 l2_bytes=%d flush_bytes=%zu abs_tol=%.9g rel_tol=%.9g\n",
                rows, k, kWarps, kTileK, l2_bytes, flush_bytes, abs_tol, rel_tol);
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
    cudaFree(d_scale_exp);
    cudaFree(d_lut);
    cudaFree(d_input);
    cudaFree(d_control);
    cudaFree(d_candidate);
    cudaFree(d_flush);
    std::printf("PASS scored_rows=%d control_runs=%zu candidate_runs=%zu bit_repeatability=true numeric_band=abs<=%.9g,rel<=%.9g\n",
                rows, control_samples.size(), candidate_samples.size(), abs_tol, rel_tol);
    return 0;
}
