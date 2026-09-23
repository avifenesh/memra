#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const float4* __restrict__ a, float* out, size_t n){ float s=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x){float4 v=__ldg(a+i); s+=v.x+v.y+v.z+v.w;} if(s==12345.f) *out=s; }
int main(){ for(int d=0; d<2; d++){ cudaSetDevice(d); size_t sz=2ull<<30; void *a,*b; cudaMalloc(&a,sz); cudaMalloc(&b,sz); cudaMemset(a,0,sz);
 cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
 cudaMemcpy(b,a,sz,cudaMemcpyDeviceToDevice); cudaEventRecord(e0); for(int r=0;r<20;r++) cudaMemcpy(b,a,sz,cudaMemcpyDeviceToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
 printf("dev%d D2D copy %.1f GB/s (r+w)\n",d,2.0*sz*20/ms/1e6);
 rd<<<188*8,512>>>((float4*)a,(float*)b,sz/16); cudaDeviceSynchronize(); cudaEventRecord(e0); for(int r=0;r<50;r++) rd<<<188*8,512>>>((float4*)a,(float*)b,sz/16); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
 printf("dev%d read kernel %.1f GB/s\n",d,(double)sz*50/ms/1e6);
 size_t s2=26700000; cudaEventRecord(e0); for(int r=0;r<500;r++) rd<<<188*4,256>>>((float4*)a,(float*)b,s2/16); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
 printf("dev%d read 26.7MB: %.1f us %.1f GB/s\n",d,ms*1000/500,(double)s2*500/ms/1e6);
 cudaFree(a); cudaFree(b);} }
