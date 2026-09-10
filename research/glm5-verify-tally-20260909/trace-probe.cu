#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
__global__ void verify_tally_trace_probe(float* x) { x[threadIdx.x] = threadIdx.x; }
int main() {
    float* x;
    if (cudaMalloc(&x, 1024*sizeof(float)) != cudaSuccess) return 1;
    cudaProfilerStart();
    nvtxRangePushA("glm5-tally-probe");
    for (int i=0; i<10; ++i) verify_tally_trace_probe<<<1,1024>>>(x);
    cudaDeviceSynchronize();
    nvtxRangePop();
    cudaProfilerStop();
    cudaFree(x);
    return 0;
}
