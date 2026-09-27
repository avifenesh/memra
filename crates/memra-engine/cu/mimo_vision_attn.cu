// Bounded preprojected MiMo ViT attention component. Not a serving path.
//
// Q [patch, 32, 64], K/V [patch, 8, 64], output [patch, 32, 64].
// Q/K have already received axial RoPE. Each length is one independent image
// or frame sequence in the current patch order. Local layers see +/-64 keys;
// only a visible first key in each sequence receives that head's learned bias.
// This is a score bias, not the text attention sink denominator.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kQueryHeads = 32;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kMaxSequencePatches = 256;
constexpr int kMaxTotalPatches = 1024;
constexpr int kMaxSequences = 32;
constexpr int kFiniteThreads = 256;

__global__ void finite_inputs(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              const float* __restrict__ sink, int patches,
                              int* __restrict__ fault) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_count = static_cast<size_t>(patches) * kQueryHeads * kHeadDim;
    const size_t kv_count = static_cast<size_t>(patches) * kKvHeads * kHeadDim;
    if (index < q_count && !isfinite(q[index])) atomicOr(fault, 1);
    if (index < kv_count && (!isfinite(k[index]) || !isfinite(v[index]))) {
        atomicOr(fault, 1);
    }
    if (index < kQueryHeads && sink != nullptr && !isfinite(sink[index])) {
        atomicOr(fault, 1);
    }
}

__device__ float warp_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

// One warp owns a (query patch, query head). Each lane owns two dimensions.
// Online softmax avoids a scores buffer. The preflight and result flag make
// finite-input and finite-output refusal observable at the host API boundary.
__global__ void preprojected_attention(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ sink,
                                       float* __restrict__ output, int offset,
                                       int length, int window,
                                       int* __restrict__ fault) {
    const int query = blockIdx.x / kQueryHeads;
    const int head = blockIdx.x % kQueryHeads;
    const int lane = threadIdx.x;
    const int kv_head = head / (kQueryHeads / kKvHeads);
    const int first = window == 0 || query <= window ? 0 : query - window;
    const int last = window == 0 || query + window + 1 >= length
                         ? length
                         : query + window + 1;
    const size_t q_base =
        (static_cast<size_t>(offset + query) * kQueryHeads + head) * kHeadDim;
    float max_score = -CUDART_INF_F;
    float denominator = 0.0f;
    float value_lo = 0.0f;
    float value_hi = 0.0f;

    for (int key = first; key < last; ++key) {
        const size_t kv_base =
            (static_cast<size_t>(offset + key) * kKvHeads + kv_head) * kHeadDim;
        const float dot = q[q_base + lane] * k[kv_base + lane] +
                          q[q_base + lane + 32] * k[kv_base + lane + 32];
        float score = warp_sum(dot) * 0.125f;
        if (key == 0 && sink != nullptr) score += sink[head];
        if (!isfinite(score)) {
            if (lane == 0) atomicOr(fault, 2);
            return;
        }
        const float next_max = fmaxf(max_score, score);
        const float alpha = expf(max_score - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        max_score = next_max;
        value_lo = value_lo * alpha + beta * v[kv_base + lane];
        value_hi = value_hi * alpha + beta * v[kv_base + lane + 32];
    }
    const float out_lo = value_lo / denominator;
    const float out_hi = value_hi / denominator;
    if (!isfinite(denominator) || denominator <= 0.0f ||
        !isfinite(out_lo) || !isfinite(out_hi)) {
        atomicOr(fault, 2);
        return;
    }
    output[q_base + lane] = out_lo;
    output[q_base + lane + 32] = out_hi;
}

int cuda_failure(cudaError_t error) {
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

}  // namespace

// 0: completed; 40001..40005: geometry/sequence/pointer refusal;
// 40007: nonfinite input; 40008: nonfinite arithmetic/output;
// 10000+cudaError_t: CUDA failure. The call synchronizes its stream.
extern "C" int memra_mimo_vision_preprojected_f32(
    const float* q, const float* k, const float* v, const float* sink,
    float* output, const int* lengths, int sequence_count, int patches,
    int query_heads, int kv_heads, int head_dim, int window, void* stream_v) {
    if (query_heads != kQueryHeads || kv_heads != kKvHeads ||
        head_dim != kHeadDim || (window != 0 && window != 64) ||
        (window == 0) != (sink == nullptr)) {
        return 40001;
    }
    if (sequence_count <= 0 || sequence_count > kMaxSequences ||
        patches <= 0 || patches > kMaxTotalPatches) {
        return 40002;
    }
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        lengths == nullptr || stream_v == nullptr) {
        return 40003;
    }
    int sum = 0;
    for (int index = 0; index < sequence_count; ++index) {
        if (lengths[index] <= 0 || lengths[index] > kMaxSequencePatches) {
            return 40004;
        }
        sum += lengths[index];
    }
    if (sum != patches) return 40005;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return cuda_failure(prior);

    const cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    int* fault = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&fault), sizeof(int));
    if (error != cudaSuccess) return cuda_failure(error);
    auto finish = [&](int result) {
        const cudaError_t freed = cudaFree(fault);
        return result == 0 ? cuda_failure(freed) : result;
    };
    error = cudaMemsetAsync(fault, 0, sizeof(int), stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    const size_t q_count = static_cast<size_t>(patches) * kQueryHeads * kHeadDim;
    const int finite_blocks = static_cast<int>(
        (q_count + kFiniteThreads - 1) / kFiniteThreads);
    finite_inputs<<<finite_blocks, kFiniteThreads, 0, stream>>>(
        q, k, v, sink, patches, fault);
    error = cudaGetLastError();
    if (error != cudaSuccess) return finish(cuda_failure(error));
    int host_fault = 0;
    error = cudaMemcpyAsync(&host_fault, fault, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    if (host_fault != 0) return finish(40007);

    int offset = 0;
    for (int index = 0; index < sequence_count; ++index) {
        const int length = lengths[index];
        preprojected_attention<<<length * kQueryHeads, 32, 0, stream>>>(
            q, k, v, sink, output, offset, length, window, fault);
        error = cudaGetLastError();
        if (error != cudaSuccess) return finish(cuda_failure(error));
        offset += length;
    }
    host_fault = 0;
    error = cudaMemcpyAsync(&host_fault, fault, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    return finish(host_fault == 0 ? 0 : 40008);
}
