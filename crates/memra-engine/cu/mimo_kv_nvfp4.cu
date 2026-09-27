// MiMo KV rows in memra's GGUF NVFP4 layout: 36 bytes per 64 f32 values.
// Each block stores four UE4M3 scales, then four groups of eight packed E2M1
// bytes. The low nibble holds element j and the high nibble element j + 8.
//
// This encoder follows memra_gguf::nvfp4_repack::f32_to_nvfp4, including its
// positive-finite scale search over codes 1..126, lower-code tie rule, zero
// handling, and IEEE f32 intermediate rounding. It does not use tensor cores.

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float raw_ue4m3(uint8_t code) {
    const int exp = (code >> 3) & 15;
    const int man = code & 7;
    if (exp == 0) {
        return __fmul_rn(static_cast<float>(man), 0x1p-9f);
    }
    // (8 + man) * 2^(exp - 10) is exact for every UE4M3 code.
    const float power = __uint_as_float(static_cast<unsigned int>(exp + 117) << 23);
    return __fmul_rn(static_cast<float>(8 + man), power);
}

__device__ __forceinline__ float gguf_ue4m3(uint8_t code) {
    // The GGUF scale decoder halves the raw UE4M3 value. The doubled E2M1
    // table below cancels that half on dequantization.
    if (code == 0 || code == 0x7f) return 0.0f;
    return __fmul_rn(raw_ue4m3(code), 0.5f);
}

__device__ __forceinline__ uint8_t nearest_scale(float x) {
    if (!(x > 0.0f)) return 0;
    uint8_t best = 0;
    float best_distance = __uint_as_float(0x7f800000u);
    // The Rust reference compares every positive finite code in ascending
    // order. Strict < keeps the smaller code on a tie. In particular, +inf
    // returns code 0: all candidate distances are +inf.
#pragma unroll 1
    for (int c = 1; c < 0x7f; ++c) {
        const float distance = fabsf(__fsub_rn(raw_ue4m3(static_cast<uint8_t>(c)), x));
        if (distance < best_distance) {
            best_distance = distance;
            best = static_cast<uint8_t>(c);
        }
    }
    return best;
}

__device__ __forceinline__ uint8_t nearest_e2m1(float x) {
    constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                    2.0f, 3.0f, 4.0f, 6.0f};
    const float ax = fabsf(x);
    int best = 0;
    float best_distance = __uint_as_float(0x7f800000u);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const float distance = fabsf(__fsub_rn(ax, magnitude[c]));
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    if (best == 0) return 0;  // Also maps NaN and either signed zero to +0.
    return static_cast<uint8_t>(best | (x < 0.0f ? 8 : 0));
}

__global__ void encode_rows(const float* input, uint8_t* output,
                            size_t rows, int width) {
    const size_t blocks_per_row = static_cast<size_t>(width) / 64;
    const size_t total_blocks = rows * blocks_per_row;
    const int lane = static_cast<int>(threadIdx.x);
    const int sub = lane / 16;
    const int in_sub = lane & 15;
    for (size_t b = blockIdx.x; b < total_blocks; b += gridDim.x) {
        const size_t row = b / blocks_per_row;
        const size_t row_block = b % blocks_per_row;
        const float value = input[row * static_cast<size_t>(width) + row_block * 64 + lane];
        const float absolute = fabsf(value);
        // Rust f32::max ignores NaN when folding from 0.0.
        float maximum = isnan(absolute) ? 0.0f : absolute;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, maximum, offset, 16);
            if (other > maximum) maximum = other;
        }

        unsigned int scale_code = in_sub == 0
            ? nearest_scale(__fdiv_rn(maximum, 6.0f))
            : 0u;
        scale_code = __shfl_sync(0xffffffffu, scale_code, lane & 16, 32);
        const float decoded_scale =
            __fmul_rn(gguf_ue4m3(static_cast<uint8_t>(scale_code)), 2.0f);
        const float inverse = decoded_scale > 0.0f
            ? __fdiv_rn(1.0f, decoded_scale) : 0.0f;
        const uint8_t code = nearest_e2m1(__fmul_rn(value, inverse));
        uint8_t* block = output + b * 36;
        if (in_sub == 0) block[sub] = static_cast<uint8_t>(scale_code);
        // The source lanes (in_sub 8..15) must participate in this shuffle.
        // Keeping it inside the low-lane write branch reads inactive lanes.
        const unsigned int high =
            __shfl_sync(0xffffffffu, static_cast<unsigned int>(code),
                        (lane & 16) + (in_sub & 7) + 8, 32);
        if (in_sub < 8) {
            block[4 + sub * 8 + in_sub] =
                static_cast<uint8_t>(code | (high << 4));
        }
    }
}

__global__ void decode_rows(const uint8_t* input, float* output,
                            size_t rows, int width) {
    constexpr int doubled_e2m1[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    const size_t blocks_per_row = static_cast<size_t>(width) / 64;
    const size_t total_blocks = rows * blocks_per_row;
    const int lane = static_cast<int>(threadIdx.x);
    const int sub = lane / 16;
    const int in_sub = lane & 15;
    for (size_t b = blockIdx.x; b < total_blocks; b += gridDim.x) {
        const uint8_t* block = input + b * 36;
        const float scale = gguf_ue4m3(block[sub]);
        const uint8_t packed = block[4 + sub * 8 + (in_sub & 7)];
        const int code = in_sub < 8 ? (packed & 15) : (packed >> 4);
        const size_t row = b / blocks_per_row;
        const size_t row_block = b % blocks_per_row;
        output[row * static_cast<size_t>(width) + row_block * 64 + lane] =
            __fmul_rn(static_cast<float>(doubled_e2m1[code]), scale);
    }
}

// Return values: 0 = launch enqueued; 1 = null buffer; 2 = unsupported width;
// 3 = zero rows; 4 = arithmetic/address overflow; 5 = insufficient extent;
// 6 = null stream; 7 = unaligned f32 buffer; 8 = overlapping buffers;
// 10000 + cudaError_t = CUDA runtime/launch error.
static int validate(const float* f32, size_t f32_elements,
                    const uint8_t* packed, size_t packed_bytes,
                    size_t rows, int width, cudaStream_t stream) {
    if (f32 == nullptr || packed == nullptr) return 1;
    if (width != 128 && width != 192) return 2;
    if (rows == 0) return 3;
    const size_t f32_row_bytes = static_cast<size_t>(width) * sizeof(float);
    const size_t packed_row_bytes = static_cast<size_t>(width / 64) * 36;
    if (rows > SIZE_MAX / f32_row_bytes ||
        rows > SIZE_MAX / packed_row_bytes) return 4;
    const size_t f32_needed = rows * static_cast<size_t>(width);
    const size_t packed_needed = rows * packed_row_bytes;
    if (f32_elements < f32_needed || packed_bytes < packed_needed) return 5;
    if (stream == nullptr) return 6;
    const uintptr_t f32_address = reinterpret_cast<uintptr_t>(f32);
    const uintptr_t packed_address = reinterpret_cast<uintptr_t>(packed);
    if (f32_address % alignof(float) != 0) return 7;
    const size_t f32_needed_bytes = rows * f32_row_bytes;
    if (f32_address > UINTPTR_MAX - f32_needed_bytes ||
        packed_address > UINTPTR_MAX - packed_needed) return 4;
    if (f32_address < packed_address + packed_needed &&
        packed_address < f32_address + f32_needed_bytes) return 8;
    return 0;
}

}  // namespace

// Buffers are contiguous device-accessible rows. Extents are capacities, in
// f32 elements for f32 buffers and bytes for packed buffers. Both launches are
// asynchronous on the supplied non-default stream; the caller owns lifetime
// and checks completion/error on that stream.
extern "C" int memra_mimo_kv_nvfp4_encode_f32(
    const float* input, size_t input_elements,
    uint8_t* output, size_t output_bytes,
    size_t rows, int width, cudaStream_t stream) {
    const int refusal = validate(input, input_elements, output, output_bytes,
                                 rows, width, stream);
    if (refusal != 0) return refusal;
    const size_t total_blocks = rows * static_cast<size_t>(width / 64);
    const unsigned int grid = static_cast<unsigned int>(
        total_blocks < 65535 ? total_blocks : 65535);
    void* arguments[] = {&input, &output, &rows, &width};
    const cudaError_t error = cudaLaunchKernel(
        reinterpret_cast<const void*>(encode_rows), dim3(grid), dim3(64),
        arguments, 0, stream);
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_kv_nvfp4_decode_f32(
    const uint8_t* input, size_t input_bytes,
    float* output, size_t output_elements,
    size_t rows, int width, cudaStream_t stream) {
    const int refusal = validate(output, output_elements, input, input_bytes,
                                 rows, width, stream);
    if (refusal != 0) return refusal;
    const size_t total_blocks = rows * static_cast<size_t>(width / 64);
    const unsigned int grid = static_cast<unsigned int>(
        total_blocks < 65535 ? total_blocks : 65535);
    void* arguments[] = {&input, &output, &rows, &width};
    const cudaError_t error = cudaLaunchKernel(
        reinterpret_cast<const void*>(decode_rows), dim3(grid), dim3(64),
        arguments, 0, stream);
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
