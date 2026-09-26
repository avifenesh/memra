// Bounded f32 MiMo decode attention component. Not a serving path.
//
// Q      [64, 192]
// K      [seq, kv_heads, 192]
// V      [seq, kv_heads, 128]
// sink   [64], optional additional softmax logit with no value row
// output [64, 128]
//
// All arrays are contiguous device f32 arrays. The cache contains only causal
// keys through the current token. window=0 uses all seq keys; window=128 uses
// the last min(seq, 128) keys. Inputs are expected to contain finite values.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>
#include <cmath>

namespace {

constexpr int kHeads = 64;
constexpr int kQkDim = 192;
constexpr int kVDim = 128;
constexpr int kThreads = 256;

__global__ void mimo_sink_attn_decode_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ sink,
    float* __restrict__ output, int seq, int kv_heads, int window) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int kv_head = head / (kHeads / kv_heads);
    const int start = window == 0 || seq <= window ? 0 : seq - window;
    const float scale = 1.0f / sqrtf(static_cast<float>(kQkDim));

    __shared__ float partial[kThreads];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float normalizer;

    float max_logit = -CUDART_INF_F;
    float denominator = 0.0f;
    if (lane == 0 && sink != nullptr) {
        max_logit = sink[head];
        denominator = 1.0f;
    }
    float value_sum = 0.0f;

    for (int token = start; token < seq; ++token) {
        const size_t k_base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * kQkDim;
        partial[lane] =
            lane < kQkDim ? q[head * kQkDim + lane] * k[k_base + lane] : 0.0f;
        __syncthreads();

        for (int stride = kThreads / 2; stride > 0; stride /= 2) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }

        if (lane == 0) {
            const float score = partial[0] * scale;
            const float next_max = fmaxf(max_logit, score);
            alpha = expf(max_logit - next_max);
            beta = expf(score - next_max);
            denominator = denominator * alpha + beta;
            max_logit = next_max;
            normalizer = denominator;
        }
        __syncthreads();

        if (lane < kVDim) {
            const size_t v_base =
                (static_cast<size_t>(token) * kv_heads + kv_head) * kVDim;
            value_sum = value_sum * alpha + beta * v[v_base + lane];
        }
        __syncthreads();
    }

    if (lane < kVDim) {
        output[head * kVDim + lane] = value_sum / normalizer;
    }
}

}  // namespace

// Return 0 on accepted launch, 40001..40005 for refused contracts, or
// 10000 + cudaError_t for a CUDA runtime/launch error. Execution is asynchronous;
// a later stream synchronization must detect device execution failures.
extern "C" int memra_mimo_sink_attn_decode_f32(
    const float* q, const float* k, const float* v, const float* sink,
    float* output, int seq, int heads, int kv_heads, int qk_dim, int v_dim,
    int window, void* stream_v) {
    if (heads != kHeads || (kv_heads != 4 && kv_heads != 8) ||
        qk_dim != kQkDim || v_dim != kVDim) {
        return 40001;
    }
    if (seq <= 0 || seq > 4096) return 40002;
    if (window != 0 && window != 128) return 40003;
    if ((kv_heads == 4 && (window != 0 || sink != nullptr)) ||
        (kv_heads == 8 && (window != 128 || sink == nullptr))) {
        return 40004;
    }
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        stream_v == nullptr) return 40005;

    // A prior asynchronous error is a refusal, not evidence that this launch
    // failed. Peek before launching so the error state remains available.
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 10000 + static_cast<int>(prior);

    mimo_sink_attn_decode_kernel<<<kHeads, kThreads, 0,
                                   static_cast<cudaStream_t>(stream_v)>>>(
        q, k, v, sink, output, seq, kv_heads, window);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return 10000 + static_cast<int>(launch);
    return 0;
}
