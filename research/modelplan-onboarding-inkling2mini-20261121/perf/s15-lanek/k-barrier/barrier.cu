// grid barrier cost on this card: N barriers in one kernel vs N empty graph nodes
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("err %s line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)
__device__ __forceinline__ void gbar(unsigned long long* ctr, unsigned nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned long long a = atomicAdd(ctr, 1ull);
    const unsigned long long target = (a / nb + 1) * nb;
    while (true) {
      unsigned long long v;
      asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(ctr));
      if (v >= target) break;
    }
  }
  __syncthreads();
}
__global__ void bars(unsigned long long* ctr, int n, float* sink) {
  float a = threadIdx.x;
  for (int i = 0; i < n; i++) { a = a * 1.0001f + 1.0f; gbar(ctr, gridDim.x); }
  if (a == 12345.0f) sink[0] = a;
}
__global__ void empty_k(float* sink) { if (threadIdx.x == 1234567) sink[0] = 1; }
int main() {
  unsigned long long* ctr; float* sink;
  CK(cudaMalloc(&ctr, 8)); CK(cudaMemset(ctr, 0, 8)); CK(cudaMalloc(&sink, 4));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int blocks : {sms, 2 * sms, 4 * sms}) for (int thr : {128, 256}) {
    const int N = 2000;
    bars<<<blocks, thr, 0, st>>>(ctr, 10, sink); CK(cudaStreamSynchronize(st));
    float best = 1e30f;
    for (int t = 0; t < 5; t++) {
      cudaEventRecord(e0, st); bars<<<blocks, thr, 0, st>>>(ctr, N, sink); cudaEventRecord(e1, st);
      CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    printf("barrier blocks=%d threads=%d: %.3f us per barrier\n", blocks, thr, best * 1000.0f / N);
  }
  // graph of K tiny kernel nodes (one block), per node
  for (int g : {1, sms}) {
    cudaGraph_t gr; cudaGraphExec_t ex; const int K = 1000;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < K; i++) empty_k<<<g, 256, 0, st>>>(sink);
    CK(cudaStreamEndCapture(st, &gr)); CK(cudaGraphInstantiate(&ex, gr, 0));
    cudaGraphLaunch(ex, st); CK(cudaStreamSynchronize(st));
    float best = 1e30f;
    for (int t = 0; t < 5; t++) {
      cudaEventRecord(e0, st); cudaGraphLaunch(ex, st); cudaEventRecord(e1, st);
      CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    printf("graph node grid=%d x256: %.3f us per node\n", g, best * 1000.0f / K);
  }
  return 0;
}
