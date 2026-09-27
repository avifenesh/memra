// Fresh MiMo text chunk attention, Q/K post-RoPE and V pre-scaled by 0.707.
// Q [chunk,64,192], K [chunk,kv_heads,192], V [chunk,kv_heads,128].
// One warp owns one (query position, head). All query positions launch together.
// There is no preceding cache and no persistent KV update in this component.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kHeads = 64;
constexpr int kQk = 192;
constexpr int kValue = 128;
constexpr int kMaxChunk = 256;
constexpr int kFiniteThreads = 256;

__global__ void finite_inputs(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              const float* __restrict__ sink,
                              int chunk, int kv_heads,
                              int* __restrict__ fault) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_count = static_cast<size_t>(chunk) * kHeads * kQk;
    const size_t k_count = static_cast<size_t>(chunk) * kv_heads * kQk;
    const size_t v_count = static_cast<size_t>(chunk) * kv_heads * kValue;
    if (index < q_count && !isfinite(q[index])) atomicOr(fault, 1);
    if (index < k_count && !isfinite(k[index])) atomicOr(fault, 1);
    if (index < v_count && !isfinite(v[index])) atomicOr(fault, 1);
    if (index < kHeads && sink != nullptr && !isfinite(sink[index])) {
        atomicOr(fault, 1);
    }
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, delta);
    }
    return __shfl_sync(0xffffffff, x, 0);
}

__global__ void chunk_attention(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const float* __restrict__ sink,
                                float* __restrict__ output,
                                int kv_heads, int window,
                                const int* __restrict__ input_fault,
                                int* __restrict__ result_fault) {
    if (*input_fault != 0) return;

    const int query = blockIdx.x / kHeads;
    const int head = blockIdx.x % kHeads;
    const int lane = threadIdx.x;
    const int kv_head = head / (kHeads / kv_heads);
    const int first = window == 0 || query + 1 <= window
                          ? 0 : query + 1 - window;
    const size_t q_base = (static_cast<size_t>(query) * kHeads + head) * kQk;
    float query_regs[6];
#pragma unroll
    for (int row = 0; row < 6; ++row) {
        query_regs[row] = q[q_base + row * 32 + lane];
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(kQk));
    float maximum = sink == nullptr ? -CUDART_INF_F : sink[head];
    float denominator = sink == nullptr ? 0.0f : 1.0f;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int token = first; token <= query; ++token) {
        const size_t k_base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * kQk;
        float dot = 0.0f;
#pragma unroll
        for (int row = 0; row < 6; ++row) {
            dot += query_regs[row] * k[k_base + row * 32 + lane];
        }
        const float score = warp_sum(dot) * scale;
        if (!isfinite(score)) {
            if (lane == 0) atomicOr(result_fault, 1);
            return;
        }
        const float next_max = fmaxf(maximum, score);
        const float alpha = expf(maximum - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        maximum = next_max;
        const size_t v_base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * kValue;
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            accum[row] = accum[row] * alpha +
                         beta * v[v_base + row * 32 + lane];
        }
    }
    if (!isfinite(denominator) || denominator <= 0.0f) {
        if (lane == 0) atomicOr(result_fault, 1);
        return;
    }
    const size_t out_base =
        (static_cast<size_t>(query) * kHeads + head) * kValue;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        const float result = accum[row] / denominator;
        if (!isfinite(result)) {
            atomicOr(result_fault, 1);
            return;
        }
        output[out_base + row * 32 + lane] = result;
    }
}

int cuda_failure(cudaError_t error) {
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

}  // namespace

// 0: completed. 40001..40003: contract error. 40007: nonfinite input.
// 40008: nonfinite score/output. 10000+cudaError_t: CUDA failure.
// Synchronizes the supplied stream before returning.
extern "C" int memra_mimo_text_chunk_attention_f32(
    const float* q, const float* k, const float* v, const float* sink,
    float* output, int chunk, int heads, int kv_heads, int qk_dim,
    int value_dim, int window, void* stream_v) {
    if (heads != kHeads || qk_dim != kQk || value_dim != kValue ||
        !((kv_heads == 4 && window == 0 && sink == nullptr) ||
          (kv_heads == 8 && window == 128 && sink != nullptr))) {
        return 40001;
    }
    if (chunk < 1 || chunk > kMaxChunk) return 40002;
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        stream_v == nullptr) return 40003;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return cuda_failure(prior);

    const cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    int* fault = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&fault), 2 * sizeof(int));
    if (error != cudaSuccess) return cuda_failure(error);
    auto finish = [&](int result) {
        const cudaError_t freed = cudaFree(fault);
        return result == 0 ? cuda_failure(freed) : result;
    };
    error = cudaMemsetAsync(fault, 0, 2 * sizeof(int), stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    const size_t q_count = static_cast<size_t>(chunk) * kHeads * kQk;
    const int blocks = static_cast<int>((q_count + kFiniteThreads - 1) /
                                        kFiniteThreads);
    finite_inputs<<<blocks, kFiniteThreads, 0, stream>>>(
        q, k, v, sink, chunk, kv_heads, fault);
    error = cudaGetLastError();
    if (error != cudaSuccess) return finish(cuda_failure(error));

    chunk_attention<<<chunk * kHeads, 32, 0, stream>>>(
        q, k, v, sink, output, kv_heads, window, fault, fault + 1);
    error = cudaGetLastError();
    if (error != cudaSuccess) return finish(cuda_failure(error));
    int host_fault[2] = {0, 0};
    error = cudaMemcpyAsync(host_fault, fault, 2 * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    if (host_fault[0] != 0) return finish(40007);
    return finish(host_fault[1] != 0 ? 40008 : 0);
}
