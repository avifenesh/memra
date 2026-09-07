// CUDA 13.1 control/basis/driver launcher for the R9 CUDA 13.3 cubin.
// R9 preserves dsv4_e4m3 negative zero and normalizes only mag==0x7f NaNs.

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

namespace dsv4_dense_tc_gate_r9 {

using namespace dsv4_dense_tc_gate;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::size_t;

constexpr int kGuardFloats = 16;
constexpr uint32_t kGuardWord = 0xa5a5a5a5u;
constexpr const char* kBasisName = "r9_control_e4m3x2_basis";

extern "C" __global__ void r9_control_e4m3x2_basis(
        const uint32_t* packed, uint32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t value = static_cast<uint16_t>(packed[i]);
    const __nv_bfloat16 lo = __float2bfloat16(
        dsv4_e4m3(static_cast<uint8_t>(value)));
    const __nv_bfloat16 hi = __float2bfloat16(
        dsv4_e4m3(static_cast<uint8_t>(value >> 8)));
    out[i] = static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16
           | __bfloat16_as_ushort(lo);
}

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

static void check_abi(CUfunction function, const char* name,
                      const size_t* offsets, const size_t* sizes, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        size_t offset = 0, size = 0;
        cu_check(cuFuncGetParamInfo(function, i, &offset, &size), "query R9 ABI");
        std::printf("abi kernel=%s param[%zu] offset=%zu size=%zu\n",
                    name, i, offset, size);
        if (offset != offsets[i] || size != sizes[i])
            throw std::runtime_error(std::string(name) + " ABI mismatch");
    }
}

static float cpu_e4m3(uint8_t code) {
    const unsigned mag = code & 0x7fu;
    if (mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    const float raw = exp == 0u
        ? static_cast<float>(man) * 0x1p-9f
        : std::ldexp(1.0f + static_cast<float>(man) / 8.0f,
                     static_cast<int>(exp) - 7);
    return (code & 0x80u) ? -raw : raw;
}

static uint16_t cpu_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0xffffu) != 0u)
        throw std::runtime_error("R9 CPU oracle encountered non-BF16-exact value");
    return static_cast<uint16_t>(bits >> 16);
}

static std::vector<uint32_t> cpu_basis_expected() {
    std::vector<uint32_t> expected(65536);
    for (uint32_t pair = 0; pair < expected.size(); ++pair) {
        const uint16_t value = static_cast<uint16_t>(pair);
        const uint16_t lo = cpu_bf16(cpu_e4m3(static_cast<uint8_t>(value)));
        const uint16_t hi = cpu_bf16(cpu_e4m3(static_cast<uint8_t>(value >> 8)));
        expected[pair] = static_cast<uint32_t>(hi) << 16 | lo;
    }
    return expected;
}

static void run_conversion_basis(CUfunction basis_function) {
    const std::vector<uint32_t> inputs = [] {
        std::vector<uint32_t> values(65536);
        for (uint32_t i = 0; i < values.size(); ++i) values[i] = i;
        return values;
    }();
    const std::vector<uint32_t> expected = cpu_basis_expected();
    constexpr size_t kBasisOffsets[3] = {0, 8, 16};
    constexpr size_t kBasisSizes[3] = {8, 8, 4};
    // The control basis is compiled in this CUDA 13.1 TU and calls the actual
    // included dsv4_e4m3 helper, not this CPU formula.
    uint32_t* d_input = nullptr;
    uint32_t* d_native = nullptr;
    uint32_t* d_control = nullptr;
    check(cudaMalloc(&d_input, inputs.size() * sizeof(uint32_t)), "R9 basis input alloc");
    check(cudaMalloc(&d_native, inputs.size() * sizeof(uint32_t)), "R9 basis native alloc");
    check(cudaMalloc(&d_control, inputs.size() * sizeof(uint32_t)), "R9 basis control alloc");
    check(cudaMemcpy(d_input, inputs.data(), inputs.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "R9 basis input copy");
    int n = static_cast<int>(inputs.size());
    uint32_t* in = d_input;
    uint32_t* out = d_native;
    void* native_args[] = {&in, &out, &n};
    cu_check(cuLaunchKernel(basis_function, (n + 255) / 256, 1, 1, 256, 1, 1,
                            0, nullptr, native_args, nullptr), "R9 basis native launch");
    r9_control_e4m3x2_basis<<<(n + 255) / 256, 256>>>(d_input, d_control, n);
    check(cudaGetLastError(), "R9 basis control launch");
    check(cudaDeviceSynchronize(), "R9 basis sync");
    std::vector<uint32_t> native(inputs.size()), control(inputs.size());
    check(cudaMemcpy(native.data(), d_native, native.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
          "R9 basis native copy");
    check(cudaMemcpy(control.data(), d_control, control.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
          "R9 basis control copy");
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (native[i] != control[i] || native[i] != expected[i]) {
            std::fprintf(stderr, "FAIL R9 basis pair=0x%04x native=0x%08x control=0x%08x cpu=0x%08x\n",
                         static_cast<unsigned>(inputs[i]), native[i], control[i], expected[i]);
            std::exit(7);
        }
    }
    // Keep the ABI contract visible in the receipt even though the control
    // basis is a runtime kernel in this TU.
    std::printf("R9_BASIS_ABI %s expected_offsets=%zu,%zu,%zu sizes=%zu,%zu,%zu\n",
                kBasisName, kBasisOffsets[0], kBasisOffsets[1], kBasisOffsets[2],
                kBasisSizes[0], kBasisSizes[1], kBasisSizes[2]);
    std::puts("PASS R9 all65536 native/control/CPU pairs; negative-zero preserved; NaN -> +0");
    cudaFree(d_input); cudaFree(d_native); cudaFree(d_control);
}

struct ModuleInfo { CUmodule module = nullptr; CUfunction function = nullptr; CUfunction basis = nullptr; };

static ModuleInfo load_module(const char* cubin) {
    check(cudaSetDevice(0), "R9 set device");
    check(cudaFree(nullptr), "R9 primary context");
    cu_check(cuInit(0), "R9 cuInit");
    CUcontext context = nullptr;
    cu_check(cuCtxGetCurrent(&context), "R9 current context");
    if (!context) throw std::runtime_error("R9 no CUDA context");
    ModuleInfo info;
    cu_check(cuModuleLoad(&info.module, cubin), "R9 load cubin");
    cu_check(cuModuleGetFunction(&info.function, info.module, "r9_gemv_fp8_native"),
             "R9 lookup GEMV");
    cu_check(cuModuleGetFunction(&info.basis, info.module, "r9_packed_cvt_basis"),
             "R9 lookup packed basis");
    constexpr size_t offsets[7] = {0, 8, 16, 24, 32, 40, 44};
    constexpr size_t sizes[7] = {8, 8, 4, 8, 8, 4, 4};
    check_abi(info.function, "r9_gemv_fp8_native", offsets, sizes, 7);
    constexpr size_t basis_offsets[3] = {0, 8, 16};
    constexpr size_t basis_sizes[3] = {8, 8, 4};
    check_abi(info.basis, "r9_packed_cvt_basis", basis_offsets, basis_sizes, 3);
    int regs = 0, shared = 0;
    cu_check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, info.function), "R9 regs");
    cu_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, info.function), "R9 shared");
    std::printf("module=%s kernel=r9_gemv_fp8_native regs=%d shared=%d\n", cubin, regs, shared);
    return info;
}

static HostData make_edge_data() {
    HostData h{4, 256, 2};
    h.codes.resize(4u * 256u);
    h.scales = {std::ldexp(1.0f, -4), std::ldexp(1.0f, 4)};
    h.input.resize(256);
    h.control.resize(4); h.candidate.resize(4);
    const uint8_t codes[] = {0x00u, 0x80u, 0x7fu, 0xffu, 0x38u, 0xb8u, 0x7eu, 0xfeu};
    const uint16_t input[] = {0x0000u, 0x8000u, 0x3f80u, 0xbf80u, 0x3f00u, 0xbf00u};
    for (int i = 0; i < 256; ++i) h.input[i] = input[i % 6];
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 256; ++col)
            h.codes[static_cast<size_t>(row) * 256 + col] = codes[(row + col) % 8];
    return h;
}

static float f32_add(float lhs, float rhs) { volatile float out = lhs + rhs; return out; }

static float cancellation_anchor(int k, int mode) {
    float thread = 0.0f;
    const float l = std::ldexp(1.0f, 30);
    const float a[8] = {l, 1, 1, 1, 1, 1, 1, 1};
    const float b[8] = {-l, 1, 1, 1, 1, 1, 1, 1};
    for (int seg = 0; seg < k / 2048; ++seg) {
        if (mode == 0) {
            for (int j = 0; j < 8; ++j) thread = f32_add(thread, a[j]);
            for (int j = 0; j < 8; ++j) thread = f32_add(thread, b[j]);
        } else if (mode == 1) {
            for (int j = 0; j < 8; j += 2) {
                thread = f32_add(thread, a[j]); thread = f32_add(thread, a[j + 1]);
                thread = f32_add(thread, b[j]); thread = f32_add(thread, b[j + 1]);
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                thread = f32_add(thread, a[j]); thread = f32_add(thread, b[j]);
            }
        }
    }
    float total = 0.0f;
    for (int tid = 0; tid < 128; ++tid) total = f32_add(total, thread);
    return total;
}

static HostData make_cancellation_data(int k) {
    HostData h{1, k, (k + 127) / 128};
    h.codes.assign(k, 0x38u); h.scales.assign(h.sc_cols, 1.0f);
    h.input.assign(k, 0x3f80u); h.control.resize(1); h.candidate.resize(1);
    for (int base = 0; base < k; base += 2048)
        for (int tid = 0; tid < 128; ++tid) {
            const int a = base + tid * 8, b = a + 1024;
            if (b + 7 >= k) continue;
            h.codes[a] = 0x38u; h.input[a] = 0x4e80u;
            h.codes[b] = 0xb8u; h.input[b] = 0x4e80u;
        }
    return h;
}

static void arm_guard(float* output, int rows) { check(cudaMemset(output + rows, 0xa5, kGuardFloats * sizeof(float)), "R9 arm guard"); }
static void check_guard(float* output, int rows, const char* arm) {
    uint32_t got[kGuardFloats] = {};
    check(cudaMemcpy(got, output + rows, sizeof(got), cudaMemcpyDeviceToHost), "R9 guard copy");
    for (uint32_t word : got) if (word != kGuardWord) throw std::runtime_error(std::string(arm) + " wrote past guard");
}

static double run_candidate(const HostData& h, CUfunction fn, uint8_t* d_codes, float* d_scales,
                            uint16_t* d_input, float* d_output, uint32_t* d_flush,
                            size_t flush_words, cudaStream_t stream) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "R9 event start"); check(cudaEventCreate(&stop), "R9 event stop");
    flush(stream, d_flush, flush_words); check(cudaEventRecord(start, stream), "R9 start");
    uint8_t* w = d_codes; float* sc = d_scales; int sc_cols = h.sc_cols;
    uint16_t* x = d_input; float* y = d_output; int n = h.rows, k = h.k;
    void* args[] = {&w, &sc, &sc_cols, &x, &y, &n, &k};
    cu_check(cuLaunchKernel(fn, static_cast<unsigned>(h.rows), 1, 1, 128, 1, 1, 0,
                            reinterpret_cast<CUstream>(stream), args, nullptr), "R9 launch");
    check(cudaEventRecord(stop, stream), "R9 stop"); check(cudaEventSynchronize(stop), "R9 sync");
    float ms = 0.0f; check(cudaEventElapsedTime(&ms, start, stop), "R9 elapsed");
    check(cudaEventDestroy(start), "R9 destroy start"); check(cudaEventDestroy(stop), "R9 destroy stop");
    return ms;
}

static int run_case(CUfunction fn, HostData h, bool full) {
    const int rows = h.rows;
    uint8_t* d_codes = nullptr; float* d_scales = nullptr; uint16_t* d_input = nullptr;
    float* d_control = nullptr; float* d_candidate = nullptr; uint32_t* d_flush = nullptr;
    int l2 = 0; check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "R9 L2");
    const size_t flush_words = (std::max<size_t>(2u * static_cast<size_t>(l2), 1u << 20) + 3) / 4;
    check(cudaMalloc(&d_codes, h.codes.size()), "R9 codes"); check(cudaMalloc(&d_scales, h.scales.size() * sizeof(float)), "R9 scales");
    check(cudaMalloc(&d_input, h.input.size() * sizeof(uint16_t)), "R9 input");
    check(cudaMalloc(&d_control, (rows + kGuardFloats) * sizeof(float)), "R9 control");
    check(cudaMalloc(&d_candidate, (rows + kGuardFloats) * sizeof(float)), "R9 candidate");
    check(cudaMalloc(&d_flush, flush_words * sizeof(uint32_t)), "R9 flush"); check(cudaMemset(d_flush, 0, flush_words * 4), "R9 flush init");
    check(cudaMemcpy(d_codes, h.codes.data(), h.codes.size(), cudaMemcpyHostToDevice), "R9 codes copy");
    check(cudaMemcpy(d_scales, h.scales.data(), h.scales.size() * sizeof(float), cudaMemcpyHostToDevice), "R9 scales copy");
    check(cudaMemcpy(d_input, h.input.data(), h.input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "R9 input copy");
    cudaStream_t stream = nullptr; check(cudaStreamCreate(&stream), "R9 stream");
    auto control = [&](const char* label) { arm_guard(d_control, rows); double ms = run_current(h, d_codes, d_scales, d_input, d_control, d_flush, flush_words, stream); check_guard(d_control, rows, label); return ms; };
    auto candidate = [&](const char* label) { arm_guard(d_candidate, rows); double ms = run_candidate(h, fn, d_codes, d_scales, d_input, d_candidate, d_flush, flush_words, stream); check_guard(d_candidate, rows, label); return ms; };
    std::printf("WARMUP rows=%d k=%d control_ms=%.6f candidate_ms=%.6f\n", rows, h.k, control("R9 control warmup"), candidate("R9 candidate warmup"));
    std::vector<float> cf(rows), nf(rows); bool hc = false, hn = false; std::vector<double> cm, nm;
    for (int cycle = 0; cycle < (full ? 3 : 1); ++cycle) {
        cm.push_back(control("R9 control scored")); record_scored_output(d_control, h.control, cf, hc, rows, "R9 control");
        nm.push_back(candidate("R9 candidate scored")); record_scored_output(d_candidate, h.candidate, nf, hn, rows, "R9 candidate");
        if (std::memcmp(h.control.data(), h.candidate.data(), rows * sizeof(float)) != 0) throw std::runtime_error("R9 candidate not bit-identical");
        nm.push_back(candidate("R9 candidate repeat")); record_scored_output(d_candidate, h.candidate, nf, hn, rows, "R9 candidate");
        if (std::memcmp(h.control.data(), h.candidate.data(), rows * sizeof(float)) != 0) throw std::runtime_error("R9 repeat not bit-identical");
        cm.push_back(control("R9 control repeat")); record_scored_output(d_control, h.control, cf, hc, rows, "R9 control");
    }
    auto mean=[](const std::vector<double>& v){double s=0;for(double x:v)s+=x;return s/v.size();};
    std::printf("RESULT rows=%d k=%d full=%s control_ms=%.6f candidate_ms=%.6f ratio=%.6f bit_identity=true\n", rows, h.k, full?"true":"false", mean(cm), mean(nm), mean(nm)/mean(cm));
    check(cudaStreamDestroy(stream), "R9 stream destroy"); cudaFree(d_codes); cudaFree(d_scales); cudaFree(d_input); cudaFree(d_control); cudaFree(d_candidate); cudaFree(d_flush); return 0;
}

} // namespace dsv4_dense_tc_gate_r9

int main(int argc, char** argv) {
    using namespace dsv4_dense_tc_gate_r9;
    if (argc < 2 || argc > 4) { std::fprintf(stderr, "usage: %s [--basis] <r9.cubin> [rows k]\n", argv[0]); return 2; }
    const bool basis = std::strcmp(argv[1], "--basis") == 0; const char* cubin = basis ? (argc > 2 ? argv[2] : nullptr) : argv[1]; if (!cubin) return 2;
    try {
        ModuleInfo info = load_module(cubin);
        run_conversion_basis(info.basis);
        int rc = 0;
        if (basis) {
            for (int rows : {1,64,65,129}) for (int k : {128,256,2048,4096,8192}) rc |= run_case(info.function, make_data(rows,k), false);
            for (int k : {2048,4096,8192}) { std::printf("CANCELLATION_ANCHOR k=%d ordered=%.9g pair_interleaved=%.9g element_interleaved=%.9g\n", k, cancellation_anchor(k,0), cancellation_anchor(k,1), cancellation_anchor(k,2)); rc |= run_case(info.function, make_cancellation_data(k), false); }
            rc |= run_case(info.function, make_edge_data(), false);
        } else {
            const int rows = argc > 2 ? std::atoi(argv[2]) : kDefaultRows; const int k = argc > 3 ? std::atoi(argv[3]) : kDefaultK;
            if (rows <= 0 || k <= 0 || k % 8) return 2;
            rc = run_case(info.function, make_data(rows,k), true);
        }
        cu_check(cuModuleUnload(info.module), "R9 unload"); if (!rc) std::puts("PASS R9 negative-zero-preserving native GEMV gate"); return rc;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL R9 %s\n", e.what()); return 1; }
}
