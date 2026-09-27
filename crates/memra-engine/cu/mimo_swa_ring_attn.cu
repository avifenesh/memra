// MiMo local decode over a device-resident 128-token f32 KV ring.
// Q [64,192], K ring [128,8,192], V ring [128,8,128], sink [64].
// K is post-RoPE and V already includes the source's 0.707 scale.
// This is a model component, not a serving path.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kHeads = 64;
constexpr int kKvHeads = 8;
constexpr int kQk = 192;
constexpr int kValue = 128;
constexpr int kWindow = 128;
constexpr int kThreads = 256;

template <bool kBatch>
__global__ void decode_ring(const float* __restrict__ q,
                            const float* __restrict__ k,
                            const float* __restrict__ v,
                            const float* __restrict__ sink,
                            float* __restrict__ output, int position) {
    const int query = kBatch ? blockIdx.y : 0;
    const int at = kBatch ? query : position;
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int kv_head = head / (kHeads / kKvHeads);
    const int count = min(at + 1, kWindow);
    const int first = at + 1 - count;
    const float scale = 1.0f / sqrtf(static_cast<float>(kQk));

    __shared__ float partial[kThreads];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float normalizer;

    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    if (lane == 0) {
        maximum = sink[head];
        denominator = 1.0f;
    }
    float value_sum = 0.0f;

    for (int token = first; token <= at; ++token) {
        const int slot = token & (kWindow - 1);
        const size_t k_base =
            (static_cast<size_t>(slot) * kKvHeads + kv_head) * kQk;
        partial[lane] =
            lane < kQk ? q[(static_cast<size_t>(query) * kHeads + head) * kQk + lane] *
                             k[k_base + lane] : 0.0f;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride /= 2) {
            if (lane < stride) partial[lane] += partial[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            const float score = partial[0] * scale;
            const float next_max = fmaxf(maximum, score);
            alpha = expf(maximum - next_max);
            beta = expf(score - next_max);
            denominator = denominator * alpha + beta;
            maximum = next_max;
            normalizer = denominator;
        }
        __syncthreads();
        if (lane < kValue) {
            const size_t v_base =
                (static_cast<size_t>(slot) * kKvHeads + kv_head) * kValue;
            value_sum = value_sum * alpha + beta * v[v_base + lane];
        }
        __syncthreads();
    }
    if (lane < kValue) {
        output[(static_cast<size_t>(query) * kHeads + head) * kValue + lane] =
            value_sum / normalizer;
    }
}

}  // namespace

// 0 = enqueued, 40001..40003 = invalid arguments, 10000 + cudaError_t =
// prior asynchronous error or launch error. Caller synchronizes the stream.
extern "C" int memra_mimo_swa_ring_decode_f32(
    const float* q, const float* k, const float* v, const float* sink,
    float* output, int position, int heads, int kv_heads, int qk_dim,
    int v_dim, int window, void* stream_v) {
    if (heads != kHeads || kv_heads != kKvHeads ||
        qk_dim != kQk || v_dim != kValue || window != kWindow) {
        return 40001;
    }
    if (position < 0 || position >= 1048576) return 40002;
    if (q == nullptr || k == nullptr || v == nullptr || sink == nullptr ||
        output == nullptr || stream_v == nullptr) return 40003;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 10000 + static_cast<int>(prior);
    decode_ring<false><<<kHeads, kThreads, 0, static_cast<cudaStream_t>(stream_v)>>>(
        q, k, v, sink, output, position);
    const cudaError_t launch = cudaGetLastError();
    return launch == cudaSuccess ? 0 : 10000 + static_cast<int>(launch);
}

// Fresh 1..128 row component, with all K/V already in the local ring.
// Each (query, head) CTA executes the same arithmetic and reduction order as
// memra_mimo_swa_ring_decode_f32 at that query's absolute position.
extern "C" int memra_mimo_swa_ring_first_chunk_f32(
    const float* q, const float* k, const float* v, const float* sink,
    float* output, int queries, int heads, int kv_heads, int qk_dim,
    int v_dim, int window, void* stream_v) {
    if (heads != kHeads || kv_heads != kKvHeads ||
        qk_dim != kQk || v_dim != kValue || window != kWindow) {
        return 40001;
    }
    if (queries < 1 || queries > kWindow) return 40002;
    if (q == nullptr || k == nullptr || v == nullptr || sink == nullptr ||
        output == nullptr || stream_v == nullptr) return 40003;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 10000 + static_cast<int>(prior);
    decode_ring<true><<<dim3(kHeads, queries), kThreads, 0,
                         static_cast<cudaStream_t>(stream_v)>>>(
        q, k, v, sink, output, 0);
    const cudaError_t launch = cudaGetLastError();
    return launch == cudaSuccess ? 0 : 10000 + static_cast<int>(launch);
}
