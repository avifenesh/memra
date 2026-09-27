// BF16 activation boundaries for one pinned MiMo ViT block. Matrix products
// remain in Memra's f32-accumulator path; the source stores BF16 activations.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__device__ __forceinline__ float bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void epilogue(const float* a, const float* b, float* output,
                         int elements, int width, int operation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float x = a[index];
    float value = x;
    if (operation == 1) {
        value += b[index % width];  // biased nn.Linear
    } else if (operation == 2) {
        value += b[index];  // residual
    } else if (operation == 3) {
        // Source BF16 SiLU output is stored before multiplication by BF16 up.
        value = bf16(x / (1.0f + expf(-x))) * b[index];
    }
    output[index] = bf16(value);
}

__global__ void split_qkv(const float* fused, float* q, float* k, float* v,
                          int patches) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= patches * 3072) return;
    const int patch = index / 3072;
    const int feature = index % 3072;
    if (feature < 2048) {
        q[patch * 2048 + feature] = fused[index];
    } else if (feature < 2560) {
        k[patch * 512 + feature - 2048] = fused[index];
    } else {
        v[patch * 512 + feature - 2560] = fused[index];
    }
}

__global__ void check_bf16(const float* values, int elements, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = values[index];
    if (!isfinite(value)) {
        atomicOr(fault, 1);
    } else if ((__float_as_uint(value) & 0xffffu) != 0) {
        atomicOr(fault, 2);
    }
}

}  // namespace

extern "C" int memra_mimo_vision_block_epilogue(
    const float* a, const float* b, float* output, int elements,
    int width, int operation, void* stream_v) {
    if (!a || !output || !stream_v || elements <= 0 || width <= 0 ||
        (elements % width) != 0 || operation < 0 || operation > 3 ||
        (operation != 0 && !b)) return 40001;
    epilogue<<<(elements + 255) / 256, 256, 0,
               static_cast<cudaStream_t>(stream_v)>>>(
        a, b, output, elements, width, operation);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_vision_block_split_qkv(
    const float* fused, float* q, float* k, float* v, int patches,
    void* stream_v) {
    if (!fused || !q || !k || !v || !stream_v ||
        patches <= 0 || patches > 1024) return 40001;
    const int elements = patches * 3072;
    split_qkv<<<(elements + 255) / 256, 256, 0,
                static_cast<cudaStream_t>(stream_v)>>>(
        fused, q, k, v, patches);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_vision_block_check_bf16(
    const float* values, int elements, int* fault, void* stream_v) {
    if (!values || !fault || !stream_v || elements <= 0 ||
        elements > 1024 * 4608) return 40001;
    check_bf16<<<(elements + 255) / 256, 256, 0,
                 static_cast<cudaStream_t>(stream_v)>>>(
        values, elements, fault);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
