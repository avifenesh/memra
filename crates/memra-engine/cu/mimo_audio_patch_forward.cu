// BF16 activation boundaries for the pinned MiMo audio patch transformer.
// Matrix products accumulate in f32 in Memra. The source Qwen2Model stores
// BF16 activations between the operations below.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace {

__device__ __forceinline__ float bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void epilogue(const float* a, const float* b, float* output,
                         int elements, int width, int operation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float x = a[index];
    float value = 0.0f;
    switch (operation) {
        case 0: value = x; break;  // plain BF16 store
        case 1: value = x + b[index % width]; break;  // Linear bias
        case 2: value = x + b[index]; break;  // residual
        case 3: {  // Qwen2 SiLU, then elementwise multiply
            const float activated = bf16(x / (1.0f + expf(-x)));
            value = bf16(activated * b[index]);
            break;
        }
        case 4:  // nn.GELU default, erf form
            value = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
            break;
        case 5:  // Qwen2RMSNorm casts to BF16 before multiplying its weight
            value = bf16(x) * b[index % width];
            break;
        default: return;
    }
    output[index] = bf16(value);
}

__global__ void rotary_bf16(float* q, float* k, int groups) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = groups * 4 * 16 * 32;
    if (index >= pairs) return;
    const int dim = index % 32;
    const int head_token = index / 32;
    const int token = head_token / 16;
    const int base = head_token * 64;
    const float inv_frequency =
        1.0f / powf(640000.0f, static_cast<float>(2 * dim) / 64.0f);
    const float angle = static_cast<float>(token % 4) * inv_frequency;
    const float cosine = bf16(cosf(angle));
    const float sine = bf16(sinf(angle));
    const float q0 = q[base + dim];
    const float q1 = q[base + dim + 32];
    const float k0 = k[base + dim];
    const float k1 = k[base + dim + 32];
    // Torch BF16 elementwise products store BF16 before the add/subtract.
    q[base + dim] = bf16(bf16(q0 * cosine) - bf16(q1 * sine));
    q[base + dim + 32] = bf16(bf16(q1 * cosine) + bf16(q0 * sine));
    k[base + dim] = bf16(bf16(k0 * cosine) - bf16(k1 * sine));
    k[base + dim + 32] = bf16(bf16(k1 * cosine) + bf16(k0 * sine));
}

__global__ void check_finite(const float* input, int elements, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements && !isfinite(input[index])) atomicOr(fault, 1);
}

}  // namespace

// operation 0: round, 1: bias, 2: residual add, 3: SiLU * up,
// 4: GELU(erf), 5: Qwen2RMSNorm post-normalization weight.
extern "C" int memra_mimo_audio_patch_epilogue(
    const float* a, const float* b, float* output, int elements,
    int width, int operation, void* stream_v) {
    if (!a || !output || !stream_v || elements <= 0 || width <= 0 ||
        operation < 0 || operation > 5 || (operation != 0 && operation != 4 && !b) ||
        ((operation == 1 || operation == 5) && elements % width != 0)) {
        return 40001;
    }
    epilogue<<<(elements + 255) / 256, 256, 0,
               static_cast<cudaStream_t>(stream_v)>>>(
        a, b, output, elements, width, operation);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_audio_patch_rope_bf16(
    float* q, float* k, int groups, void* stream_v) {
    if (!q || !k || !stream_v || groups <= 0 || groups > 1500) return 40001;
    const int pairs = groups * 4 * 16 * 32;
    rotary_bf16<<<(pairs + 255) / 256, 256, 0,
                  static_cast<cudaStream_t>(stream_v)>>>(q, k, groups);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

extern "C" int memra_mimo_audio_patch_check_finite(
    const float* input, int elements, int* fault, void* stream_v) {
    if (!input || !fault || !stream_v || elements <= 0) return 40001;
    check_finite<<<(elements + 255) / 256, 256, 0,
                   static_cast<cudaStream_t>(stream_v)>>>(input, elements, fault);
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
