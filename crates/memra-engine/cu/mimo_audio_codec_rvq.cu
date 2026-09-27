// One source-order F32 Euclidean codebook selection and residual subtraction.
// Post-downsample rows are [tokens,1024]; embeds are [bins,1024].
// This scalar kernel keeps x norm, (2*x) dot embed, embed norm, and the
// parenthesized subtraction. The TU is compiled with -fmad=false.

#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstddef>

namespace {
constexpr int kWidth = 1024;
constexpr int kThreads = 256;

__global__ void finite_values(const float* values, int count, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(values[index])) atomicOr(fault, 1);
}

__global__ void select_code(const float* residual, const float* embed,
                            int* code_ids, int bins, int* fault) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    const float* x = residual + static_cast<size_t>(token) * kWidth;
    __shared__ float x_norm;
    __shared__ float scores[kThreads];
    __shared__ int indices[kThreads];

    if (lane == 0) {
        float sum = 0.0f;
        for (int col = 0; col < kWidth; ++col) {
            const float value = x[col];
            sum += value * value;
        }
        x_norm = sum;
        if (!isfinite(sum)) atomicOr(fault, 2);
    }
    __syncthreads();

    float best = -INFINITY;
    int best_index = INT_MAX;
    for (int code = lane; code < bins; code += kThreads) {
        const float* row = embed + static_cast<size_t>(code) * kWidth;
        float dot = 0.0f;
        float embed_norm = 0.0f;
        for (int col = 0; col < kWidth; ++col) {
            const float value = row[col];
            dot += (2.0f * x[col]) * value;
            embed_norm += value * value;
        }
        const float score = -((x_norm - dot) + embed_norm);
        if (!isfinite(score)) {
            atomicOr(fault, 4);
            continue;
        }
        if (score > best || (score == best && code < best_index)) {
            best = score;
            best_index = code;
        }
    }
    scores[lane] = best;
    indices[lane] = best_index;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float other = scores[lane + stride];
            const int other_index = indices[lane + stride];
            if (other > scores[lane] ||
                (other == scores[lane] && other_index < indices[lane])) {
                scores[lane] = other;
                indices[lane] = other_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) code_ids[token] = indices[0];
}

__global__ void subtract_embed(const float* residual, const float* embed,
                               const int* code_ids, float* next, int tokens,
                               int bins, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * kWidth) return;
    const int token = index / kWidth;
    const int code = code_ids[token];
    if (code < 0 || code >= bins) {
        atomicOr(fault, 8);
        next[index] = 0.0f;
        return;
    }
    next[index] = residual[index] -
                  embed[static_cast<size_t>(code) * kWidth + index % kWidth];
}

int launch_status() {
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

bool valid(int tokens, int bins) {
    return tokens >= 1 && tokens <= 256 &&
           (bins == 128 || bins == 256 || bins == 1024);
}
}  // namespace

extern "C" int memra_mimo_codec_rvq_check_finite(
    const float* values, int count, int* fault, void* stream_v) {
    if (!values || !fault || !stream_v || count < 1 || count > 1024 * 1024)
        return 40001;
    finite_values<<<(count + 255) / 256, 256, 0,
                    static_cast<cudaStream_t>(stream_v)>>>(values, count, fault);
    return launch_status();
}

extern "C" int memra_mimo_codec_rvq_select(
    const float* residual, const float* embed, int* code_ids,
    int tokens, int bins, int* fault, void* stream_v) {
    if (!residual || !embed || !code_ids || !fault || !stream_v ||
        !valid(tokens, bins)) return 40001;
    select_code<<<tokens, kThreads, 0, static_cast<cudaStream_t>(stream_v)>>>(
        residual, embed, code_ids, bins, fault);
    return launch_status();
}

extern "C" int memra_mimo_codec_rvq_subtract(
    const float* residual, const float* embed, const int* code_ids,
    float* next, int tokens, int bins, int* fault, void* stream_v) {
    if (!residual || !embed || !code_ids || !next || !fault || !stream_v ||
        !valid(tokens, bins)) return 40001;
    subtract_embed<<<(tokens * kWidth + 255) / 256, 256, 0,
                     static_cast<cudaStream_t>(stream_v)>>>(
        residual, embed, code_ids, next, tokens, bins, fault);
    return launch_status();
}
