// Standalone ModelOpt NVFP4 kq_store half2 gate.
//
// This tool does not alter the engine. It compares the current scalar
//   half(float(scale_e4m3fn) * float(E2M1_doubled_code))
// materialization with a packed half2 path for every finite signed E4M3FN
// scale byte and every packed pair of E2M1 nibbles.  0x7f/0xff are treated as
// nonfinite/refused contract bytes and are counted, not silently admitted.
//
// Compile only after the owning lane has approved the CUDA window:
//   nvcc -O3 -std=c++17 -arch=sm_120 -o /tmp/dsv4-modelopt-half2-gate \
//     tools/dsv4-modelopt-half2-gate.cu

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_OK(call)                                                                  \
    do {                                                                               \
        cudaError_t _e = (call);                                                       \
        if (_e != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(_e));                                      \
            std::exit(2);                                                              \
        }                                                                              \
    } while (0)

__constant__ int8_t D_CODE[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                  0, -1, -2, -3, -4, -6, -8, -12};
__constant__ uint32_t D_CODE_HALF2[256];
static constexpr int8_t H_CODE[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                      0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float modelopt_e4m3fn(uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0x7fu) return 0.0f;  // nonfinite/refused contract byte
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v = exp ? __uint_as_float((exp + 120u) << 23 | (man << 20))
                  : static_cast<float>(man) * 0x1p-9f;
    if (x & 0x80u) v = -v;
    return v * 0.5f;
}

__global__ void exhaustive_compare(uint16_t* ref, uint16_t* cand,
                                   unsigned* mismatches, unsigned* nonfinite,
                                   unsigned* signed_zero_ref, unsigned* signed_zero_cand) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 256u * 256u) return;
    const uint8_t scale_byte = static_cast<uint8_t>(id >> 8);
    const uint8_t packed = static_cast<uint8_t>(id);
    const unsigned out = id * 2u;
    const unsigned lo = packed & 0xfu;
    const unsigned hi = packed >> 4;
    if ((scale_byte & 0x7fu) == 0x7fu) {
        atomicAdd(nonfinite, 1u);
        return;
    }
    const float scale = modelopt_e4m3fn(scale_byte);
    const uint16_t r0 = __half_as_ushort(__float2half_rn(scale * static_cast<float>(D_CODE[lo])));
    const uint16_t r1 = __half_as_ushort(__float2half_rn(scale * static_cast<float>(D_CODE[hi])));
    ref[out + 0] = r0;
    ref[out + 1] = r1;

    const __half2 hscale = __float2half2_rn(scale);
    const uint32_t code_bits = D_CODE_HALF2[packed];
    const __half2 hcodes = *reinterpret_cast<const __half2*>(&code_bits);
    const __half2 product = __hmul2(hscale, hcodes);
    const uint32_t product_bits = *reinterpret_cast<const uint32_t*>(&product);
    cand[out + 0] = static_cast<uint16_t>(product_bits & 0xffffu);
    cand[out + 1] = static_cast<uint16_t>(product_bits >> 16);
    if (cand[out + 0] != ref[out + 0] || cand[out + 1] != ref[out + 1])
        atomicAdd(mismatches, 1u);
    if (r0 == 0x8000u || r1 == 0x8000u) atomicAdd(signed_zero_ref, 1u);
    if (cand[out + 0] == 0x8000u || cand[out + 1] == 0x8000u)
        atomicAdd(signed_zero_cand, 1u);
}

// One real 64-value K block is four 16-value kq_store calls. Each block owns
// one B tile and each thread writes 16 half values in the scalar arm or eight
// half2 values in the candidate arm.
__global__ void store_scalar_tiles(const uint8_t* codes, const uint8_t* scales,
                                   uint16_t* out, unsigned tiles) {
    const unsigned tile = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (tile >= tiles) return;
    for (unsigned rep = 0; rep < 4; ++rep) {
        const unsigned in = (tile * 4u + rep) * 9u;
        const float scale = modelopt_e4m3fn(scales[tile * 4u + rep]);
        const unsigned out_base = (tile * 4u + rep) * 2048u + tid * 16u;
        #pragma unroll
        for (unsigned j = 0; j < 16; ++j) {
            const uint8_t byte = codes[in + (j >> 1)];
            const unsigned code = (j & 1u) ? (byte >> 4) : (byte & 0xfu);
            out[out_base + j] = __half_as_ushort(
                __float2half_rn(scale * static_cast<float>(D_CODE[code])));
        }
    }
}

__global__ void store_half2_tiles(const uint8_t* codes, const uint8_t* scales,
                                  uint16_t* out, unsigned tiles) {
    const unsigned tile = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (tile >= tiles) return;
    for (unsigned rep = 0; rep < 4; ++rep) {
        const unsigned in = (tile * 4u + rep) * 9u;
        const float scale = modelopt_e4m3fn(scales[tile * 4u + rep]);
        const __half2 hscale = __float2half2_rn(scale);
        const unsigned out_base = (tile * 4u + rep) * 1024u + tid * 8u;
        #pragma unroll
        for (unsigned pair = 0; pair < 8; ++pair) {
            const uint32_t code_bits = D_CODE_HALF2[codes[in + pair]];
            const __half2 hcodes = *reinterpret_cast<const __half2*>(&code_bits);
            const __half2 product = __hmul2(hscale, hcodes);
            reinterpret_cast<uint32_t*>(out)[out_base + pair] =
                *reinterpret_cast<const uint32_t*>(&product);
        }
    }
}

static uint16_t exact_half_int(int value) {
    if (value == 0) return 0;
    const unsigned sign = value < 0 ? 0x8000u : 0u;
    unsigned magnitude = static_cast<unsigned>(value < 0 ? -value : value);
    unsigned exponent = 0;
    while ((1u << (exponent + 1u)) <= magnitude) ++exponent;
    const unsigned exp_bits = exponent + 15u;
    const unsigned mantissa = (magnitude - (1u << exponent)) << (10u - exponent);
    return static_cast<uint16_t>(sign | (exp_bits << 10u) | mantissa);
}

int main(int argc, char** argv) {
    const bool exhaustive_only = argc == 2 && std::strcmp(argv[1], "--exhaustive-only") == 0;
    if (argc != 1 && !exhaustive_only) {
        std::fprintf(stderr, "usage: dsv4-modelopt-half2-gate [--exhaustive-only]\n");
        return 2;
    }
    std::vector<uint32_t> code_half2(256);
    for (unsigned packed = 0; packed < 256; ++packed) {
        const uint16_t lo = exact_half_int(H_CODE[packed & 0xfu]);
        const uint16_t hi = exact_half_int(H_CODE[packed >> 4]);
        code_half2[packed] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    }
    CUDA_OK(cudaMemcpyToSymbol(D_CODE_HALF2, code_half2.data(), code_half2.size() * sizeof(uint32_t)));

    uint16_t* ref = nullptr;
    uint16_t* cand = nullptr;
    unsigned *mismatches = nullptr, *nonfinite = nullptr, *signed_ref = nullptr, *signed_cand = nullptr;
    CUDA_OK(cudaMalloc(&ref, 256u * 256u * 2u * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&cand, 256u * 256u * 2u * sizeof(uint16_t)));
    CUDA_OK(cudaMallocManaged(&mismatches, sizeof(unsigned)));
    CUDA_OK(cudaMallocManaged(&nonfinite, sizeof(unsigned)));
    CUDA_OK(cudaMallocManaged(&signed_ref, sizeof(unsigned)));
    CUDA_OK(cudaMallocManaged(&signed_cand, sizeof(unsigned)));
    *mismatches = *nonfinite = *signed_ref = *signed_cand = 0;
    exhaustive_compare<<<256, 256>>>(ref, cand, mismatches, nonfinite, signed_ref, signed_cand);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    const unsigned expected_nonfinite = 2u * 256u;
    std::printf("EXHAUSTIVE finite_scales=%u packed_codes=256 mismatches=%u nonfinite_refused=%u/%u signed_zero_ref=%u signed_zero_candidate=%u\n",
                254u, *mismatches, *nonfinite, expected_nonfinite, *signed_ref, *signed_cand);
    if (*mismatches != 0 || *nonfinite != expected_nonfinite || *signed_ref != *signed_cand)
        return 1;
    if (exhaustive_only) {
        CUDA_OK(cudaFree(ref));
        CUDA_OK(cudaFree(cand));
        CUDA_OK(cudaFree(mismatches));
        CUDA_OK(cudaFree(nonfinite));
        CUDA_OK(cudaFree(signed_ref));
        CUDA_OK(cudaFree(signed_cand));
        std::printf("PASS exhaustive ModelOpt half2 finite-scale/code identity; no throughput measurement\n");
        return 0;
    }

    constexpr unsigned tiles = 8192;
    const size_t code_bytes = static_cast<size_t>(tiles) * 4u * 9u;
    std::vector<uint8_t> h_codes(code_bytes), h_scales(static_cast<size_t>(tiles) * 4u);
    for (unsigned tile = 0; tile < tiles; ++tile) {
        for (unsigned rep = 0; rep < 4; ++rep) {
            h_scales[tile * 4u + rep] = static_cast<uint8_t>((tile * 17u + rep * 29u) & 0x7eu);
            for (unsigned j = 0; j < 8; ++j)
                h_codes[(tile * 4u + rep) * 9u + j] = static_cast<uint8_t>(tile * 13u + rep * 7u + j * 3u);
        }
    }
    uint8_t *d_codes = nullptr, *d_scales = nullptr;
    uint16_t *d_ref = nullptr, *d_cand = nullptr;
    CUDA_OK(cudaMalloc(&d_codes, h_codes.size()));
    CUDA_OK(cudaMalloc(&d_scales, h_scales.size()));
    CUDA_OK(cudaMalloc(&d_ref, static_cast<size_t>(tiles) * 4u * 2048u * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_cand, static_cast<size_t>(tiles) * 4u * 2048u * sizeof(uint16_t)));
    CUDA_OK(cudaMemcpy(d_codes, h_codes.data(), h_codes.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size(), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    store_scalar_tiles<<<tiles, 128>>>(d_codes, d_scales, d_ref, tiles);
    store_half2_tiles<<<tiles, 128>>>(d_codes, d_scales, d_cand, tiles);
    CUDA_OK(cudaDeviceSynchronize());
    constexpr int repeats = 20;
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) store_scalar_tiles<<<tiles, 128>>>(d_codes, d_scales, d_ref, tiles);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float scalar_ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&scalar_ms, start, stop));
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) store_half2_tiles<<<tiles, 128>>>(d_codes, d_scales, d_cand, tiles);
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float half2_ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&half2_ms, start, stop));
    std::printf("DIAGNOSTIC warm_replicated_input_store_micro tiles=%u stores_per_tile=4 repeats=%d scalar_ms=%.3f half2_ms=%.3f ratio=%.4fx not_engine_timing=true\n",
                tiles, repeats, scalar_ms, half2_ms, scalar_ms / half2_ms);
    return 0;
}
