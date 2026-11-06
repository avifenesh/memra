// ceil_probe.cu: per-part ceilings on GB10 (sm_121a) for the Qwen3.8-27B prefill parts.
//   bf16   : mma.sync m16n8k16 bf16 -> f32, 8 independent accumulators (attention class)
//   f8bs   : mma.sync m16n8k32 kind::mxf8f6f4.block_scale.scale_vec::1X e4m3 x e4m3, ue8m0 = 1
//            (the form memra's F8F4 tile issues), 8 independent accumulators
//   f8resc : the F8F4 program's per-k32 form: fresh C = 0 per MMA, then sum += (dA*dB) * C,
//            8 independent output tiles (issue-bound ceiling of that program class)
//   dram   : float4 read and read+write copy over 2 GiB
// build: nvcc -arch=sm_121a -O3 -o ceil_probe ceil_probe.cu
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{auto e=(x);if(e){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

#define MMA_BF16(c,a,b) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]))
#define MMA_F8BS(c,a,b) asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 " \
  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(0x7F7F7F7Fu),"r"(0x7F7F7F7Fu))

template<int IT> __global__ void k_bf16(float* s){
  unsigned a[4]={0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u}, b[2]={0x3f803f80u,0x3f803f80u};
  float c[8][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) MMA_BF16(c[j],a,b);
  }
  float t=0; for(int j=0;j<8;j++) t+=c[j][0];
  if(t==-1.0f) s[0]=t;
}
template<int IT> __global__ void k_f8bs(float* s){
  unsigned a[4]={0x38383838u,0x38383838u,0x38383838u,0x38383838u}, b[2]={0x38383838u,0x38383838u};
  float c[8][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) MMA_F8BS(c[j],a,b);
  }
  float t=0; for(int j=0;j<8;j++) t+=c[j][0];
  if(t==-1.0f) s[0]=t;
}
template<int IT> __global__ void k_f8resc(float* s, float dA, float dB){
  unsigned a[4]={0x30303030u,0x30303030u,0x30303030u,0x30303030u}, b[2]={0x30303030u,0x30303030u};
  float sum[8][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<8;j++){
      float c[4]={0.f,0.f,0.f,0.f};
      a[j & 3] += 0x01010101u;   // operands change every MMA: no two issues can be merged
      MMA_F8BS(c,a,b);
      #pragma unroll
      for(int l=0;l<4;l++) sum[j][l] += dA * dB * c[l];
    }
    dA = dA * 1.0000001f;   // keep the product live per iteration, like per-k32 scales
  }
  float t=0; for(int j=0;j<8;j++) for(int l=0;l<4;l++) t+=sum[j][l];
  if(t==-1.0f) s[0]=t;
}
// The F8F4 tile's exact epilogue shape: per MMA, four distinct (dA[row] * dB[col]) products
// (two rows x two columns per thread) and four FFMAs; scales change every k32 step so nothing
// can be hoisted. 8 independent output tiles per warp, like the tile's j0 x n unroll.
template<int IT> __global__ void k_f8tile(float* s, const float* sc){
  unsigned a[4]={0x30303030u,0x30303030u,0x30303030u,0x30303030u}, b[2]={0x30303030u,0x30303030u};
  float sum[8][4]={};
  float dA0=sc[threadIdx.x&7], dA1=sc[8+(threadIdx.x&7)];
  float dB[8][2];
  #pragma unroll
  for(int j=0;j<8;j++){ dB[j][0]=sc[16+j]; dB[j][1]=sc[24+j]; }
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<8;j++){
      float c[4]={0.f,0.f,0.f,0.f};
      a[j & 3] += 0x01010101u;
      MMA_F8BS(c,a,b);
      sum[j][0] += dA0 * dB[j][0] * c[0];
      sum[j][1] += dA0 * dB[j][1] * c[1];
      sum[j][2] += dA1 * dB[j][0] * c[2];
      sum[j][3] += dA1 * dB[j][1] * c[3];
    }
    dA0 = __int_as_float(__float_as_int(dA0) ^ (i & 1)); dA1 = __int_as_float(__float_as_int(dA1) ^ (i & 1));
    #pragma unroll
    for(int j=0;j<8;j++){ dB[j][0] = __int_as_float(__float_as_int(dB[j][0]) ^ (i & 1)); dB[j][1] = __int_as_float(__float_as_int(dB[j][1]) ^ (i & 1)); }
  }
  float t=0; for(int j=0;j<8;j++) for(int l=0;l<4;l++) t+=sum[j][l];
  if(t==-1.0f) s[0]=t;
}
__global__ void k_read(const float4* __restrict__ in, float4* out, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  float4 acc=make_float4(0,0,0,0);
  for(;i<n;i+=st){ float4 v=in[i]; acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w; }
  if(acc.x==-1.0f) out[0]=acc;
}
__global__ void k_copy(const float4* __restrict__ in, float4* __restrict__ out, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, st=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=st) out[i]=in[i];
}

template<class F> float timeit(F f, int reps){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int r=0;r<reps;r++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); return ms/reps;
}
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int sms=p.multiProcessorCount;
  printf("device %s sms %d l2 %d bytes smem/SM %zu smem/block optin %zu regs/SM %d\n", p.name, sms, p.l2CacheSize, p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.regsPerMultiprocessor);
  float* sc; CK(cudaMalloc(&sc,64*sizeof(float))); { float h[64]; for(int i=0;i<64;i++) h[i]=1.0f+i*1e-3f; CK(cudaMemcpy(sc,h,sizeof(h),cudaMemcpyHostToDevice)); }
  float* s; CK(cudaMalloc(&s,64));
  constexpr int IT=4096;
  for(int wpb: {4,8,16}){
    int blocks=sms*2, thr=32*wpb;
    double nw=(double)blocks*wpb;
    float ms=timeit([&]{k_bf16<IT><<<blocks,thr>>>(s);},5);
    double fl=nw*IT*8*16.0*8*16*2; printf("bf16   m16n8k16  warps/blk %2d: %.1f TFLOPS\n",wpb,fl/ms/1e9);
    ms=timeit([&]{k_f8bs<IT><<<blocks,thr>>>(s);},5);
    fl=nw*IT*8*16.0*8*32*2; printf("f8bs   m16n8k32  warps/blk %2d: %.1f TFLOPS\n",wpb,fl/ms/1e9);
    ms=timeit([&]{k_f8resc<IT><<<blocks,thr>>>(s,1.f,1.f);},5);
    printf("f8resc m16n8k32  warps/blk %2d: %.1f TFLOPS (per-k32 C=0 + sum += dA*dB*C)\n",wpb,fl/ms/1e9);
    ms=timeit([&]{k_f8tile<IT><<<blocks,thr>>>(s,sc);},5);
    printf("f8tile m16n8k32  warps/blk %2d: %.1f TFLOPS (tile epilogue: 4 FMUL + 4 FFMA per MMA)\n",wpb,fl/ms/1e9);
  }
  CK(cudaGetLastError());
  size_t bytes=2ull<<30, n=bytes/16;
  float4 *in,*out; CK(cudaMalloc(&in,bytes)); CK(cudaMalloc(&out,bytes));
  CK(cudaMemset(in,0,bytes)); CK(cudaMemset(out,0,bytes));
  float ms=timeit([&]{k_read<<<sms*16,256>>>(in,out,n);},10);
  printf("dram read  %.1f GB/s\n", bytes/ms/1e6);
  ms=timeit([&]{k_copy<<<sms*16,256>>>(in,out,n);},10);
  printf("dram copy  %.1f GB/s (read+write bytes)\n", 2.0*bytes/ms/1e6);
  CK(cudaGetLastError());
  return 0;
}
