// Standalone MiMo V component codec. Each 16-value group is a positive UE4M3
// scale byte followed by 16 signed 5-bit two's-complement codes, packed from
// the least significant bit first (11 bytes total). This has no KV dispatch.
//
// The arithmetic follows value_codec_quality_probe.rs at ece94bf4: separate
// positive/negative maxima, nearest positive scale with lower-code ties, and
// code rounding to nearest with halfway cases away from zero.

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace {

constexpr size_t kWidth = 128;
constexpr size_t kGroupValues = 16;
constexpr size_t kGroupBytes = 11;
constexpr size_t kGroupsPerRow = kWidth / kGroupValues;
constexpr size_t kMaxRows = 1024;

__device__ __forceinline__ float raw_ue4m3(uint8_t code) {
    if (code == 0 || code == 0x7f) return 0.0f;
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exponent == 0) {
        return __fmul_rn(static_cast<float>(mantissa), 0x1p-9f);
    }
    // (8 + mantissa) * 2^(exponent - 10) is exact in f32.
    const float power =
        __uint_as_float(static_cast<unsigned int>(exponent + 117) << 23);
    return __fmul_rn(static_cast<float>(8 + mantissa), power);
}

__device__ __forceinline__ uint8_t nearest_scale(float target) {
    if (!isfinite(target) || !(target > 0.0f)) return 0;
    uint8_t best = 0;
    float distance = __uint_as_float(0x7f800000u);
#pragma unroll 1
    for (int candidate = 1; candidate < 0x7f; ++candidate) {
        const float error = fabsf(
            __fsub_rn(raw_ue4m3(static_cast<uint8_t>(candidate)), target));
        if (error < distance) {
            best = static_cast<uint8_t>(candidate);
            distance = error;
        }
    }
    return best;
}

__global__ void encode_groups(const float* input, uint8_t* output,
                              size_t groups) {
    for (size_t group = blockIdx.x * static_cast<size_t>(blockDim.x) +
                        threadIdx.x;
         group < groups;
         group += gridDim.x * static_cast<size_t>(blockDim.x)) {
        const float* values = input + group * kGroupValues;
        uint8_t* bytes = output + group * kGroupBytes;
        float positive = 0.0f;
        float negative = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float value = values[i];
            // f32::max from a zero seed ignores NaNs, matching the probe.
            if (value > positive) positive = value;
            if (-value > negative) negative = -value;
        }
        const float positive_scale = __fdiv_rn(positive, 15.0f);
        const float negative_scale = __fdiv_rn(negative, 16.0f);
        const float target = positive_scale > negative_scale
            ? positive_scale : negative_scale;
        const uint8_t scale_byte = nearest_scale(target);
        const float scale = raw_ue4m3(scale_byte);
        bytes[0] = scale_byte;
#pragma unroll
        for (int i = 1; i < 11; ++i) bytes[i] = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int code = 0;
            if (scale > 0.0f) {
                const float quotient = __fdiv_rn(values[i], scale);
                const float rounded = roundf(quotient);
                if (!isnan(rounded)) {
                    code = rounded < -16.0f ? -16
                         : rounded > 15.0f ? 15
                         : static_cast<int>(rounded);
                }
            }
            const uint8_t bits = static_cast<uint8_t>(code) & 31;
            const int bit = i * 5;
            const int byte = 1 + bit / 8;
            const int shift = bit % 8;
            bytes[byte] |= static_cast<uint8_t>(bits << shift);
            if (shift > 3) {
                bytes[byte + 1] |= static_cast<uint8_t>(bits >> (8 - shift));
            }
        }
    }
}

__global__ void decode_groups(const uint8_t* input, float* output,
                              size_t groups) {
    for (size_t group = blockIdx.x * static_cast<size_t>(blockDim.x) +
                        threadIdx.x;
         group < groups;
         group += gridDim.x * static_cast<size_t>(blockDim.x)) {
        const uint8_t* bytes = input + group * kGroupBytes;
        float* values = output + group * kGroupValues;
        const float scale = raw_ue4m3(bytes[0]);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int bit = i * 5;
            const int byte = 1 + bit / 8;
            const int shift = bit % 8;
            uint16_t packed = bytes[byte];
            if (shift > 3) {
                packed |= static_cast<uint16_t>(bytes[byte + 1]) << 8;
            }
            const int bits = (packed >> shift) & 31;
            const int code = bits >= 16 ? bits - 32 : bits;
            values[i] = __fmul_rn(static_cast<float>(code), scale);
        }
    }
}

// Return 0 on enqueued launch, 1 null pointer, 2 row bound, 3 insufficient
// extent, 4 null stream, 5 f32 misalignment, 6 overlap, 7 address overflow,
// or 10000 + cudaError_t on a CUDA launch failure.
static int validate(const float* values, size_t value_elements,
                    const uint8_t* bytes, size_t byte_elements,
                    size_t rows, cudaStream_t stream) {
    if (values == nullptr || bytes == nullptr) return 1;
    if (rows == 0 || rows > kMaxRows) return 2;
    const size_t needed_values = rows * kWidth;
    const size_t needed_bytes = rows * kGroupsPerRow * kGroupBytes;
    if (value_elements < needed_values || byte_elements < needed_bytes) return 3;
    if (stream == nullptr) return 4;
    const uintptr_t value_address = reinterpret_cast<uintptr_t>(values);
    const uintptr_t byte_address = reinterpret_cast<uintptr_t>(bytes);
    if (value_address % alignof(float) != 0) return 5;
    const size_t value_bytes = needed_values * sizeof(float);
    if (value_address > UINTPTR_MAX - value_bytes ||
        byte_address > UINTPTR_MAX - needed_bytes) return 7;
    if (value_address < byte_address + needed_bytes &&
        byte_address < value_address + value_bytes) return 6;
    return 0;
}

}  // namespace

// Buffers contain contiguous [128] head rows. The caller owns both buffers
// until stream completion. This component accepts 1..1024 rows per call.
extern "C" int memra_mimo_s5_g16_encode_f32(
    const float* input, size_t input_elements, uint8_t* output,
    size_t output_bytes, size_t rows, cudaStream_t stream) {
    const int refusal = validate(input, input_elements, output, output_bytes,
                                 rows, stream);
    if (refusal != 0) return refusal;
    size_t groups = rows * kGroupsPerRow;
    const unsigned int grid =
        static_cast<unsigned int>((groups + 127) / 128);
    void* args[] = {&input, &output, &groups};
    const cudaError_t error = cudaLaunchKernel(
        reinterpret_cast<const void*>(encode_groups), dim3(grid), dim3(128),
        args, 0, stream);
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_s5_g16_decode_f32(
    const uint8_t* input, size_t input_bytes, float* output,
    size_t output_elements, size_t rows, cudaStream_t stream) {
    const int refusal = validate(output, output_elements, input, input_bytes,
                                 rows, stream);
    if (refusal != 0) return refusal;
    size_t groups = rows * kGroupsPerRow;
    const unsigned int grid =
        static_cast<unsigned int>((groups + 127) / 128);
    void* args[] = {&input, &output, &groups};
    const cudaError_t error = cudaLaunchKernel(
        reinterpret_cast<const void*>(decode_groups), dim3(grid), dim3(128),
        args, 0, stream);
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
