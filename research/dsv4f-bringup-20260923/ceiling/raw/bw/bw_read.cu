// bw_read.cu: practical device-memory read bandwidth. A grid-stride kernel reads a buffer with
// 128-bit loads and folds it into one word per thread (so the loads cannot be elided), over
// buffer sizes from 64 MiB to 4 GiB. Prints the best of 20 timed passes per size (cudaEvent).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA_ERROR %s line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
__global__ void rd(const uint4* __restrict__ p, size_t n, uint32_t* out) {
    uint32_t acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
int main() {
    int dev = 0, sms = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    printf("DEVICE %s sms=%d\n", pr.name, sms);
    const size_t maxb = 4ull << 30;
    uint4* p; uint32_t* o; CK(cudaMalloc(&p, maxb)); CK(cudaMalloc(&o, 4)); CK(cudaMemset(p, 1, maxb));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (size_t bytes = 64ull << 20; bytes <= maxb; bytes *= 4) {
        size_t n = bytes / 16; float best = 1e30f;
        for (int blocksPerSm : {4, 8, 16}) {
            for (int r = 0; r < 21; r++) {
                CK(cudaEventRecord(a));
                rd<<<sms * blocksPerSm, 512>>>(p, n, o);
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                float ms; CK(cudaEventElapsedTime(&ms, a, b));
                if (r > 0 && ms < best) best = ms;
            }
        }
        printf("READ bytes=%zu best_ms=%.4f GBps=%.1f\n", bytes, best, bytes / (best * 1e-3) / 1e9);
    }
    return 0;
}
