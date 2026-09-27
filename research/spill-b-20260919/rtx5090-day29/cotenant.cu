// Day 29 fault arm: a self-owned co-tenant that holds N MiB of device memory and sleeps until SIGTERM.
// usage: cotenant <mib>   (spawned and stopped by cotenant-cell.sh, inside the collector's lock hold)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char** argv) {
    size_t mib = argc > 1 ? strtoull(argv[1], nullptr, 10) : 1390;
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, mib << 20);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc(%zu MiB) failed: %s\n", mib, cudaGetErrorString(e)); return 1; }
    cudaMemset(p, 0, mib << 20); cudaDeviceSynchronize();
    printf("cotenant holding %zu MiB pid=%d\n", mib, getpid()); fflush(stdout);
    for (;;) pause();
}
