#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ inline float bf16_to_f32(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ inline uint16_t f32_to_bf16_rne(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

__global__ void mimo_audio_embed_sum_kernel(
    const uint16_t* codes,
    const uint64_t* table_ptrs,
    float* output,
    int positions,
    int width,
    int vocab
) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = blockIdx.y;
    if (element >= width || position >= positions) return;
    float sum = 0.0f;
    for (int channel = 0; channel < 20; ++channel) {
        const uint16_t code = codes[position * 20 + channel];
        if (code >= vocab) {
            output[position * width + element] = __int_as_float(0x7fc00000);
            return;
        }
        const auto table = reinterpret_cast<const uint16_t*>(
            static_cast<uintptr_t>(table_ptrs[channel])
        );
        const uint16_t weight = table[static_cast<size_t>(code) * width + element];
        sum = bf16_to_f32(f32_to_bf16_rne(sum + bf16_to_f32(weight)));
    }
    output[position * width + element] = sum;
}

}  // namespace

extern "C" int memra_mimo_audio_embed_sum_bf16(
    const uint16_t* codes,
    const uint64_t* table_ptrs,
    float* output,
    int positions,
    int channels,
    int width,
    int vocab,
    void* stream
) {
    if (!codes || !table_ptrs || !output || !stream ||
        positions <= 0 || positions > 6000 ||
        channels != 20 || width != 1024 || vocab != 1280) {
        return 40001;
    }
    const dim3 threads(256);
    const dim3 blocks((width + threads.x - 1) / threads.x, positions);
    mimo_audio_embed_sum_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        codes, table_ptrs, output, positions, width, vocab
    );
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
