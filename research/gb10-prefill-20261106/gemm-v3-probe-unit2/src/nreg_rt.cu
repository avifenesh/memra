// setmaxnreg at runtime on GB10: 4 warpgroups (512 threads, 128 registers at entry); warpgroup 0
// deallocates to 32, warpgroups 1-3 allocate 160 (3 x 128 x 160 + 128 x 32 = 65,536). Each consumer
// thread keeps 112 live f32 accumulators; the result is checked against the host.
#include <cstdio>
#include <cuda_runtime.h>
#define NACC 112
__global__ void __launch_bounds__(512, 1) k(float* out, const float* in, int iters) {
    const int wg = threadIdx.x / 128;
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n");
        out[blockIdx.x * 512 + threadIdx.x] = -1.0f;
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = in[i];
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = fmaf(acc[i], 0.5f, (float) i);
    }
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; ++i) t += acc[i];
    out[blockIdx.x * 512 + threadIdx.x] = t;
}
int main() {
    float *out, *in;
    cudaMalloc(&out, 48 * 512 * 4); cudaMalloc(&in, NACC * 4);
    float h[NACC]; for (int i = 0; i < NACC; ++i) h[i] = (float) i;
    cudaMemcpy(in, h, sizeof h, cudaMemcpyHostToDevice);
    k<<<48, 512>>>(out, in, 64);
    cudaError_t e = cudaDeviceSynchronize();
    float r[512]; cudaMemcpy(r, out, sizeof r, cudaMemcpyDeviceToHost);
    // host reference for one consumer thread
    double t = 0; for (int i = 0; i < NACC; ++i) { float a = h[i]; for (int it = 0; it < 64; ++it) a = fmaf(a, 0.5f, (float) i); t += a; }
    float tf = 0.0f; for (int i = 0; i < NACC; ++i) { float a = h[i]; for (int it = 0; it < 64; ++it) a = fmaf(a, 0.5f, (float) i); tf += a; }
    printf("setmaxnreg runtime: %s, producer %.1f consumer %.4f host %.4f %s\n", cudaGetErrorString(e), r[0], r[200], tf,
           (r[0] == -1.0f && r[200] == tf) ? "OK" : "MISMATCH");
    return 0;
}
