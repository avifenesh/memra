#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s line %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)
__global__ void pingpong(volatile int* mine, volatile int* peer, int iters, int who){
  for(int i=0;i<iters;i++){
    if(who==0){ *peer = 2*i+1; while(*mine != 2*i+1){} }
    else { while(*mine != 2*i+1){} *peer = 2*i+1; }
  }
}
__global__ void peerstore(float4* dst, const float4* src, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=src[i]; }
int main(){
  int can01=0,can10=0; CK(cudaDeviceCanAccessPeer(&can01,0,1)); CK(cudaDeviceCanAccessPeer(&can10,1,0));
  printf("canAccessPeer 0->1 %d 1->0 %d\n",can01,can10);
  CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1,0));
  CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0,0));
  size_t big=256<<20; void *a,*b; CK(cudaSetDevice(0)); CK(cudaMalloc(&a,big)); CK(cudaSetDevice(1)); CK(cudaMalloc(&b,big));
  cudaStream_t s0; CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0));
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  size_t sizes[]={8192,32768,1<<20,big};
  for(size_t sz: sizes){ int reps = sz>=big?10:1000;
    CK(cudaMemcpyPeerAsync(b,1,a,0,sz,s0)); CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0,s0)); for(int r=0;r<reps;r++) CK(cudaMemcpyPeerAsync(b,1,a,0,sz,s0)); CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); printf("memcpyPeer 0->1 %zu B: %.2f us/op %.2f GB/s\n",sz,ms*1000/reps,(double)sz*reps/ms/1e6);
  }
  // kernel peer store 8KB and 32MB
  for(size_t sz: {(size_t)8192,(size_t)(32<<20)}){ int n=sz/16; int reps=sz<100000?1000:20;
    peerstore<<<(n+255)/256,256,0,s0>>>((float4*)b,(const float4*)a,n); CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0,s0)); for(int r=0;r<reps;r++) peerstore<<<(n+255)/256,256,0,s0>>>((float4*)b,(const float4*)a,n); CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); printf("kernel peer store %zu B: %.2f us/op %.2f GB/s\n",sz,ms*1000/reps,(double)sz*reps/ms/1e6);
    peerstore<<<(n+255)/256,256,0,s0>>>((float4*)a,(const float4*)a,n); CK(cudaStreamSynchronize(s0));
  }
  // flag ping-pong latency
  int *f0,*f1; CK(cudaSetDevice(0)); CK(cudaMalloc(&f0,4)); CK(cudaMemset(f0,0,4)); CK(cudaSetDevice(1)); CK(cudaMalloc(&f1,4)); CK(cudaMemset(f1,0,4));
  cudaStream_t s1; CK(cudaStreamCreate(&s1)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
  int iters=10000; auto t0=std::chrono::steady_clock::now();
  CK(cudaSetDevice(1)); pingpong<<<1,1,0,s1>>>(f1,f0,iters,1);
  CK(cudaSetDevice(0)); pingpong<<<1,1,0,s0>>>(f0,f1,iters,0);
  CK(cudaStreamSynchronize(s0)); CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1));
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
  printf("flag pingpong round trip: %.2f us\n", us/iters);
  // launch latency of empty kernel
  CK(cudaSetDevice(0)); t0=std::chrono::steady_clock::now(); for(int r=0;r<10000;r++) peerstore<<<1,32,0,s0>>>((float4*)a,(const float4*)a,0); CK(cudaStreamSynchronize(s0));
  printf("empty launch host rate: %.2f us/launch\n", std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/10000);
  return 0;
}
