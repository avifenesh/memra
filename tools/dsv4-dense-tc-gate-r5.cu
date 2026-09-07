// Standalone R5 preflight for native packed E4M3x2 -> BF16x2 conversion.
//
// PTX ISA 9.3 spells the candidate instruction as:
//   cvt.rn.bf16x2.e4m3x2 d, a;
// The current DSV4 engine target is sm_120a. This translation unit is a
// compile-only capability gate before attempting to replace R4's shared BF16
// LUT. If ptxas accepts the instruction for sm_120a, --basis can execute an
// independent packed bit oracle on a GPU; this lane does not run it.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using std::uint16_t;
using std::uint32_t;

static void check(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(rc));
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

static uint16_t bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0xffffu) != 0u) {
        std::fprintf(stderr, "basis value is not BF16-exact: %.9g\n", value);
        std::exit(3);
    }
    return static_cast<uint16_t>(bits >> 16);
}

static uint16_t scale_bf16_bits(uint16_t base, int scale_exp) {
    if ((base & 0x7fffu) == 0u) return 0u;
    return static_cast<uint16_t>(static_cast<int>(base) + scale_exp * 128);
}

// This is intentionally the exact PTX candidate under test. If sm_120a does
// not admit the family-specific conversion, nvcc/ptxas must fail the build.
__device__ __forceinline__ uint32_t native_e4m3x2_to_bf16x2(uint16_t packed) {
    uint32_t out = 0;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out) : "h"(packed));
    return out;
}

extern "C" __global__ void r5_packed_cvt_basis(
        const uint32_t* packed, uint32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = native_e4m3x2_to_bf16x2(static_cast<uint16_t>(packed[i]));
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--basis") != 0) {
        std::fprintf(stderr, "usage: dsv4-dense-tc-gate-r5 [--basis]\n");
        return 2;
    }
    // Host construction is independent of CUDA's native conversion. Include
    // signed/zero/normal E4M3 pairs and exact scale exponents for the packed
    // post-conversion arithmetic that R5 would use.
    const std::vector<uint32_t> packed = {
        0x00003838u, 0x0000bf80u, 0x00007e38u, 0x00007f00u,
        0x00003f38u, 0x00007e7eu, 0x00000000u, 0x0000ff38u,
    };
    std::vector<uint32_t> expected(packed.size());
    for (size_t i = 0; i < packed.size(); ++i) {
        const uint16_t lo = bf16_bits(e4m3(static_cast<uint8_t>(packed[i] & 0xffu)));
        const uint16_t hi = bf16_bits(e4m3(static_cast<uint8_t>((packed[i] >> 8) & 0xffu)));
        // These scales are part of the independent packing oracle, not the
        // native conversion instruction. The basis uses E8M0 exponents 0.
        expected[i] = static_cast<uint32_t>(scale_bf16_bits(hi, 0)) << 16
                    | scale_bf16_bits(lo, 0);
    }
    if (argc <= 1) {
        std::printf("COMPILE_ONLY native=cvt.rn.bf16x2.e4m3x2 basis_vectors=%zu\n",
                    packed.size());
        return 0;
    }

    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    check(cudaMalloc(&d_in, packed.size() * sizeof(uint32_t)), "input alloc");
    check(cudaMalloc(&d_out, expected.size() * sizeof(uint32_t)), "output alloc");
    check(cudaMemcpy(d_in, packed.data(), packed.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "input copy");
    r5_packed_cvt_basis<<<1, 32>>>(d_in, d_out, static_cast<int>(packed.size()));
    check(cudaGetLastError(), "basis launch");
    check(cudaDeviceSynchronize(), "basis sync");
    std::vector<uint32_t> actual(expected.size());
    check(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), "output copy");
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "FAIL packed basis[%zu]: got=0x%08x want=0x%08x\n",
                         i, actual[i], expected[i]);
            cudaFree(d_in);
            cudaFree(d_out);
            return 7;
        }
    }
    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("PASS packed_basis_vectors=%zu native=cvt.rn.bf16x2.e4m3x2\n",
                expected.size());
    return 0;
}
