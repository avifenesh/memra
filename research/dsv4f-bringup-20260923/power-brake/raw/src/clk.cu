#include <cstdio>
#include <cuda_runtime.h>
__global__ void spin(unsigned long long* out, long iters){
  unsigned long long c0=clock64(), t0; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t0));
  float a=threadIdx.x, b=1.0001f; for(long i=0;i<iters;i++){ a=fmaf(a,b,0.5f); b=fmaf(b,a,-0.5f);} 
  unsigned long long c1=clock64(), t1; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t1));
  if(threadIdx.x==0 && blockIdx.x==0){ out[0]=c1-c0; out[1]=t1-t0; } if(a==123.f) out[2]=b; }
int main(){ for(int d=0; d<2; d++){ cudaSetDevice(d); unsigned long long* o; cudaMallocManaged(&o,24);
  for(int rep=0; rep<4; rep++){ spin<<<188*4,256>>>(o, 4000000); cudaDeviceSynchronize(); printf("dev%d rep%d effective SM clock %.0f MHz over %.1f ms\n",d,rep,(double)o[0]/o[1]*1000,o[1]/1e6);} } }
