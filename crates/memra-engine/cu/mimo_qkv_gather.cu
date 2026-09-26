// Device-side gather of MiMo V2.6's four checkpoint QKV shard projections.
// Each shard output is [token, Q_shard | K_shard | V_shard]. The typed plan
// carries the four-shard declaration; this kernel only accepts the pinned
// global or sliding text geometry. V is scaled before the KV write.

#include <cuda_runtime.h>
#include <stdint.h>

__global__ void mimo_qkv_gather_kernel(
    const float* __restrict__ s0,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ s3,
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    int q_per_shard,
    int k_per_shard,
    int v_per_shard) {
    const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int token = (int)blockIdx.y;
    const int shard = (int)blockIdx.z / 3;
    const int segment = (int)blockIdx.z % 3;
    const int shard_width = q_per_shard + k_per_shard + v_per_shard;
    const float* source = shard == 0 ? s0 : shard == 1 ? s1 : shard == 2 ? s2 : s3;
    source += (size_t)token * shard_width;

    if (segment == 0 && col < q_per_shard) {
        q[(size_t)token * 4 * q_per_shard + shard * q_per_shard + col] = source[col];
    } else if (segment == 1 && col < k_per_shard) {
        k[(size_t)token * 4 * k_per_shard + shard * k_per_shard + col] =
            source[q_per_shard + col];
    } else if (segment == 2 && col < v_per_shard) {
        v[(size_t)token * 4 * v_per_shard + shard * v_per_shard + col] =
            source[q_per_shard + k_per_shard + col] * 0.707f;
    }
}

extern "C" int memra_mimo_qkv_gather_f32(
    const float* s0, const float* s1, const float* s2, const float* s3,
    float* q, float* k, float* v,
    int tokens, int q_per_shard, int k_per_shard, int v_per_shard,
    void* stream_v) {
    if (!s0 || !s1 || !s2 || !s3 || !q || !k || !v || !stream_v ||
        tokens < 1 || tokens > 128 || q_per_shard != 3072 ||
        !((k_per_shard == 192 && v_per_shard == 128) ||
          (k_per_shard == 384 && v_per_shard == 256))) {
        return 47001;
    }
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 1000 + (int)prior;
    dim3 block(256);
    dim3 grid((q_per_shard + 255) / 256, (unsigned)tokens, 12);
    mimo_qkv_gather_kernel<<<grid, block, 0, (cudaStream_t)stream_v>>>(
        s0, s1, s2, s3, q, k, v, q_per_shard, k_per_shard, v_per_shard);
    cudaError_t error = cudaGetLastError();
    return error == cudaSuccess ? 0 : 1000 + (int)error;
}
