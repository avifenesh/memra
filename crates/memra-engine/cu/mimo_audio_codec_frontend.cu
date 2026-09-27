// Pinned MiMo bundled audio tokenizer Conv1D frontend.
// Input and output are frame-major f32 carriers of BF16 values. Resident
// weights and biases remain the original little-endian BF16 checkpoint bytes.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float weight_bf16(const uint16_t* values, size_t index) {
    return __uint_as_float(static_cast<uint32_t>(values[index]) << 16);
}

__device__ __forceinline__ float conv_gelu(float sum, float bias) {
    const float conv = bf16(sum + bias);
    return bf16(0.5f * conv * (1.0f + erff(conv * 0.7071067811865475f)));
}

__global__ void im2col_bf16(const float* input, float* columns,
                            int frames, int channels, int stride, int rows) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t extent = static_cast<size_t>(rows) * channels * 3;
    if (index >= extent) return;
    const int tap = static_cast<int>(index % 3);
    const int channel = static_cast<int>((index / 3) % channels);
    const int row = static_cast<int>(index / (static_cast<size_t>(channels) * 3));
    const int source = row * stride + tap - 1;
    columns[index] = source < 0 || source >= frames
        ? 0.0f
        : bf16(input[static_cast<size_t>(source) * channels + channel]);
}

__global__ void conv_epilogue(const float* projected, const uint16_t* bias,
                              float* output, int rows, int out_channels) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t extent = static_cast<size_t>(rows) * out_channels;
    if (index >= extent) return;
    const int channel = static_cast<int>(index % out_channels);
    output[index] = conv_gelu(projected[index], weight_bf16(bias, channel));
}

// Correct fallback for shapes that cuBLASLt declines. It is bounded and
// deliberately simple; the im2col plus tensor-core GEMM path serves normal
// admitted shapes.
__global__ void conv_direct(const float* input, const uint16_t* weight,
                            const uint16_t* bias, float* output, int frames,
                            int in_channels, int out_channels, int stride, int rows) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t extent = static_cast<size_t>(rows) * out_channels;
    if (index >= extent) return;
    const int channel = static_cast<int>(index % out_channels);
    const int row = static_cast<int>(index / out_channels);
    float sum = 0.0f;
    for (int input_channel = 0; input_channel < in_channels; ++input_channel) {
        for (int tap = 0; tap < 3; ++tap) {
            const int source = row * stride + tap - 1;
            if (source < 0 || source >= frames) continue;
            const float sample =
                bf16(input[static_cast<size_t>(source) * in_channels + input_channel]);
            const size_t weight_index =
                (static_cast<size_t>(channel) * in_channels + input_channel) * 3 + tap;
            sum = fmaf(sample, weight_bf16(weight, weight_index), sum);
        }
    }
    output[index] = conv_gelu(sum, weight_bf16(bias, channel));
}

__global__ void check_finite(const float* values, int elements, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements && !isfinite(values[index])) atomicOr(fault, 1);
}

bool valid_conv(int frames, int in_channels, int out_channels, int stride, int rows) {
    return frames >= 1 && frames <= 1024 &&
           in_channels >= 1 && in_channels <= 1024 &&
           out_channels >= 1 && out_channels <= 1024 &&
           (stride == 1 || stride == 2) &&
           rows == (frames + stride - 1) / stride;
}

int launch_status() {
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

}  // namespace

extern "C" int memra_mimo_codec_im2col_bf16(
    const float* input, float* columns, int frames, int channels,
    int stride, int rows, void* stream_v) {
    if (!input || !columns || !stream_v ||
        !valid_conv(frames, channels, 1, stride, rows)) return 40001;
    const size_t elements = static_cast<size_t>(rows) * channels * 3;
    im2col_bf16<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                   static_cast<cudaStream_t>(stream_v)>>>(
        input, columns, frames, channels, stride, rows);
    return launch_status();
}

extern "C" int memra_mimo_codec_conv_epilogue(
    const float* projected, const uint16_t* bias, float* output,
    int rows, int out_channels, void* stream_v) {
    if (!projected || !bias || !output || !stream_v ||
        rows < 1 || rows > 1024 || out_channels < 1 || out_channels > 1024) return 40001;
    const size_t elements = static_cast<size_t>(rows) * out_channels;
    conv_epilogue<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                    static_cast<cudaStream_t>(stream_v)>>>(
        projected, bias, output, rows, out_channels);
    return launch_status();
}

extern "C" int memra_mimo_codec_conv_direct(
    const float* input, const uint16_t* weight, const uint16_t* bias,
    float* output, int frames, int in_channels, int out_channels,
    int stride, int rows, void* stream_v) {
    if (!input || !weight || !bias || !output || !stream_v ||
        !valid_conv(frames, in_channels, out_channels, stride, rows)) return 40001;
    const size_t elements = static_cast<size_t>(rows) * out_channels;
    conv_direct<<<static_cast<unsigned>((elements + 255) / 256), 256, 0,
                  static_cast<cudaStream_t>(stream_v)>>>(
        input, weight, bias, output, frames, in_channels, out_channels, stride, rows);
    return launch_status();
}

extern "C" int memra_mimo_codec_check_finite(
    const float* values, int elements, int* fault, void* stream_v) {
    if (!values || !fault || !stream_v || elements < 1 || elements > 1024 * 1024)
        return 40001;
    check_finite<<<(elements + 255) / 256, 256, 0,
                   static_cast<cudaStream_t>(stream_v)>>>(values, elements, fault);
    return launch_status();
}
