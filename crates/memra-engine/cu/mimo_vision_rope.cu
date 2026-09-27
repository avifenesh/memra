// MiMo vision two-axis RoPE after BF16 QKV projection. Each patch has 32
// height/width phase pairs, reused for the second half of every 64-d head.
// Q [patch,32,64], K [patch,8,64], phase [patch,32].

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kQueryHeads = 32;
constexpr int kKeyHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kHalf = 32;
constexpr int kMaxPatches = 1024;

bool on_current_device(const void* pointer, int device) {
    cudaPointerAttributes attributes{};
    return cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
           attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
}

__global__ void rotate_axial(const float* __restrict__ q,
                             const float* __restrict__ k,
                             const float* __restrict__ cos,
                             const float* __restrict__ sin,
                             float* __restrict__ out_q,
                             float* __restrict__ out_k,
                             size_t pairs) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    const size_t patch = index / (kQueryHeads * kHalf);
    const int head = static_cast<int>((index / kHalf) % kQueryHeads);
    const int dim = static_cast<int>(index % kHalf);
    const size_t phase = patch * kHalf + dim;
    const float c = cos[phase];
    const float s = sin[phase];

    const size_t q_base = (patch * kQueryHeads + head) * kHeadDim + dim;
    const float q0 = q[q_base];
    const float q1 = q[q_base + kHalf];
    out_q[q_base] = __bfloat162float(__float2bfloat16_rn(q0 * c - q1 * s));
    out_q[q_base + kHalf] =
        __bfloat162float(__float2bfloat16_rn(q1 * c + q0 * s));

    if (head < kKeyHeads) {
        const size_t k_base = (patch * kKeyHeads + head) * kHeadDim + dim;
        const float k0 = k[k_base];
        const float k1 = k[k_base + kHalf];
        out_k[k_base] = __bfloat162float(__float2bfloat16_rn(k0 * c - k1 * s));
        out_k[k_base + kHalf] =
            __bfloat162float(__float2bfloat16_rn(k1 * c + k0 * s));
    }
}

}  // namespace

// 0 = enqueued. 40001 = geometry refusal, 40002 = pointer/device refusal,
// 10000 + cudaError_t = CUDA failure. Caller synchronizes its stream.
extern "C" int memra_mimo_vision_rope_f32(
    const float* q, const float* k, const float* cos, const float* sin,
    float* out_q, float* out_k, int patches, int query_heads, int key_heads,
    int head_dim, void* stream_v) {
    if (patches <= 0 || patches > kMaxPatches || query_heads != kQueryHeads ||
        key_heads != kKeyHeads || head_dim != kHeadDim) return 40001;
    if (q == nullptr || k == nullptr || cos == nullptr || sin == nullptr ||
        out_q == nullptr || out_k == nullptr || stream_v == nullptr) return 40002;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 10000 + static_cast<int>(prior);
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return 10000 + static_cast<int>(error);
    if (!on_current_device(q, device) || !on_current_device(k, device) ||
        !on_current_device(cos, device) || !on_current_device(sin, device) ||
        !on_current_device(out_q, device) || !on_current_device(out_k, device)) {
        return 40002;
    }
    const size_t pairs = static_cast<size_t>(patches) * kQueryHeads * kHalf;
    rotate_axial<<<static_cast<unsigned>((pairs + 255) / 256), 256, 0,
                   static_cast<cudaStream_t>(stream_v)>>>(
        q, k, cos, sin, out_q, out_k, pairs);
    error = cudaGetLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}
