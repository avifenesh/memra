
// Streaming-read roofline at the expert GEMM's per-call sizes, measured the way the bench measures:
// back-to-back launches, each reading a different slice of a 3 GiB pool (so the bytes come from DRAM),
// host wall time over the calls, nothing subtracted. Kernel = ceiling/bw.cu k_read (16-byte loads,
// 4 in flight per thread, grid 82 SMs x 8 blocks x 256 threads).
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda %s\n", cudaGetErrorString(e_)); exit(4);} } while (0)
__global__ void k_read(const uint4* __restrict__ p, size_t n, uint32_t* __restrict__ sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a = p[i], b = p[i + stride], c = p[i + 2 * stride], d = p[i + 3 * stride];
        acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n; i += stride) { uint4 a = p[i]; acc ^= a.x ^ a.y ^ a.z ^ a.w; }
    if (acc == 0x9e3779b9u) sink[0] = acc;
}
int main(int argc, char** argv) {
    const size_t pool = 3ull << 30;
    uint8_t* p; CK(cudaMalloc(&p, pool)); CK(cudaMemset(p, 1, pool));
    uint32_t* sink; CK(cudaMalloc(&sink, 64));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int grid = pr.multiProcessorCount * 8;
    const int reps = 200;
    for (int a = 1; a < argc; ++a) {
        const size_t s = (size_t)atoll(argv[a]) & ~(size_t)255;
        const size_t slots = pool / s;
        double best = 1e30;
        for (int t = 0; t < 5; ++t) {
            k_read<<<grid, 256>>>((const uint4*)p, s / 16, sink);
            CK(cudaDeviceSynchronize());
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) k_read<<<grid, 256>>>((const uint4*)(p + (i % slots) * s), s / 16, sink);
            CK(cudaDeviceSynchronize());
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            if (us < best) best = us;
        }
        printf("bwslice bytes=%zu best %.1f us/call %.1f GB/s (best of 5 x %d calls, rotating over %zu slices)\n", s, best, s / best / 1e3, reps, slots);
    }
    return 0;
}
