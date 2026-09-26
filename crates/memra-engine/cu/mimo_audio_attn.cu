// MiMo audio patch encoder attention over already projected f32 Q/K/V.
// Q/K already include RoPE. Layout is [groups, 4, 16, 64]; every query
// sees all four keys in its group, with no sink, cache, or output projection.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>

namespace {

constexpr int kMaxGroups = 1500;
constexpr int kGroupSize = 4;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;
constexpr int kLayers = 6;
constexpr int kFiniteThreads = 256;

__device__ float warp_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, delta);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__global__ void finite_inputs(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              size_t count, int* __restrict__ fault) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count &&
        (!isfinite(q[index]) || !isfinite(k[index]) || !isfinite(v[index]))) {
        atomicOr(fault, 1);
    }
}

// Four warps per block, one per query token. A warp handles one group/head,
// with each lane holding two of the 64 dimensions. Each warp computes all
// four scores before normalizing and writing its own 64-value output row.
__global__ void preprojected_attention(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       float* __restrict__ output,
                                       int* __restrict__ fault) {
    const int group = blockIdx.x / kHeads;
    const int head = blockIdx.x % kHeads;
    const int query = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const size_t q_base =
        (static_cast<size_t>(group * kGroupSize + query) * kHeads + head) *
        kHeadDim;

    float scores[kGroupSize];
    for (int key = 0; key < kGroupSize; ++key) {
        const size_t k_base =
            (static_cast<size_t>(group * kGroupSize + key) * kHeads + head) *
            kHeadDim;
        const float partial = q[q_base + lane] * k[k_base + lane] +
                              q[q_base + lane + 32] * k[k_base + lane + 32];
        scores[key] = warp_sum(partial) * 0.125f;
    }
    for (float score : scores) {
        if (!isfinite(score)) {
            if (lane == 0) atomicOr(fault, 2);
            return;
        }
    }

    float weights[kGroupSize] = {};
    int valid = 1;
    if (lane == 0) {
        float maximum = scores[0];
        for (int key = 1; key < kGroupSize; ++key) {
            maximum = fmaxf(maximum, scores[key]);
        }
        float denominator = 0.0f;
        for (int key = 0; key < kGroupSize; ++key) {
            weights[key] = expf(scores[key] - maximum);
            denominator += weights[key];
        }
        valid = isfinite(denominator) && denominator > 0.0f;
        if (valid) {
            for (float& weight : weights) weight /= denominator;
        } else {
            atomicOr(fault, 2);
        }
    }
    valid = __shfl_sync(0xffffffff, valid, 0);
    if (!valid) return;
    for (float& weight : weights) {
        weight = __shfl_sync(0xffffffff, weight, 0);
    }

    float value_lo = 0.0f;
    float value_hi = 0.0f;
    for (int key = 0; key < kGroupSize; ++key) {
        const size_t v_base =
            (static_cast<size_t>(group * kGroupSize + key) * kHeads + head) *
            kHeadDim;
        value_lo += weights[key] * v[v_base + lane];
        value_hi += weights[key] * v[v_base + lane + 32];
    }
    if (!isfinite(value_lo) || !isfinite(value_hi)) {
        atomicOr(fault, 2);
        return;
    }
    output[q_base + lane] = value_lo;
    output[q_base + lane + 32] = value_hi;
}

int cuda_failure(cudaError_t error) {
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

bool on_current_device(const float* ptr, int device) {
    cudaPointerAttributes attributes{};
    return cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess &&
           attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
}

}  // namespace

// 0: completed; 40001: pinned geometry refusal; 40002: group refusal;
// 40003: pointer/device refusal; 40007: nonfinite input;
// 40008: nonfinite arithmetic/output; 10000+cudaError_t: CUDA failure.
// The call synchronizes its stream so the finite checks are observable.
extern "C" int memra_mimo_audio_preprojected_f32(
    const float* q, const float* k, const float* v, float* output,
    int layer, int groups, int group_size, int heads, int head_dim,
    int full_attention, void* stream_v) {
    if (layer < 0 || layer >= kLayers || group_size != kGroupSize ||
        heads != kHeads || head_dim != kHeadDim || full_attention != 1) {
        return 40001;
    }
    if (groups <= 0 || groups > kMaxGroups) return 40002;
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        stream_v == nullptr) {
        return 40003;
    }
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return cuda_failure(prior);
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return cuda_failure(error);
    if (!on_current_device(q, device) || !on_current_device(k, device) ||
        !on_current_device(v, device) || !on_current_device(output, device)) {
        return 40003;
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    int* fault = nullptr;
    error = cudaMalloc(reinterpret_cast<void**>(&fault), sizeof(int));
    if (error != cudaSuccess) return cuda_failure(error);
    auto finish = [&](int result) {
        const cudaError_t freed = cudaFree(fault);
        return result == 0 ? cuda_failure(freed) : result;
    };
    error = cudaMemsetAsync(fault, 0, sizeof(int), stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    const size_t count =
        static_cast<size_t>(groups) * kGroupSize * kHeads * kHeadDim;
    const int blocks =
        static_cast<int>((count + kFiniteThreads - 1) / kFiniteThreads);
    finite_inputs<<<blocks, kFiniteThreads, 0, stream>>>(q, k, v, count, fault);
    error = cudaGetLastError();
    if (error != cudaSuccess) return finish(cuda_failure(error));
    int host_fault = 0;
    error = cudaMemcpyAsync(&host_fault, fault, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    if (host_fault != 0) return finish(40007);

    preprojected_attention<<<groups * kHeads, 128, 0, stream>>>(
        q, k, v, output, fault);
    error = cudaGetLastError();
    if (error != cudaSuccess) return finish(cuda_failure(error));
    host_fault = 0;
    error = cudaMemcpyAsync(&host_fault, fault, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) return finish(cuda_failure(error));
    return finish(host_fault == 0 ? 0 : 40008);
}
