// ceil_attn.cu: tensor-pipe ceilings on GB10 (sm_121a) for the forms the prime attention issues.
// All forms run mma.sync m16n8k16 bf16 -> f32 with operands in registers or shared memory, one CTA
// per SM, W warps per CTA, no global memory in the timed loop.
//   peak8   8 independent accumulators per warp (the GEMM lane's bf16 peak form)
//   chain4  4 accumulators per warp, each MMA depending on the previous MMA of its chain
//           (the QK shape: four 16-deep score chains per 32-key tile)
//   chain2  2 accumulators per warp (QK with the key groups not interleaved)
//   pv2     PV shape: 32 independent two-MMA chains from zero, each folded into an f32 accumulator
//           by FFMA with a scale (O = O*alpha + C)
//   qkldsm  chain4 with the B operand ldmatrix'd from a swizzled smem tile before every MMA pair
//           (QK as the kernel issues it, Q fragments in registers)
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a ceil_attn.cu -o ceil_attn
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdint>
#define CK(x) do{auto e=(x);if(e){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

#define MMA(c,a,b) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" \
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]))
#define MMAZ(c,a,b) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};" \
  :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(0.0f))

template<int IT> __global__ void k_peak8(float* s){
  unsigned a[4]={0x3f803f80u,0x3f803f81u,0x3f803f82u,0x3f803f83u}, b[2]={0x3f803f84u,0x3f803f85u};
  float c[8][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<8;j++) MMA(c[j],a,b);
  }
  float t=0; for(int j=0;j<8;j++) t+=c[j][0]+c[j][1]+c[j][2]+c[j][3];
  if(t==-1.0f) s[0]=t;
}
template<int IT, int NC> __global__ void k_chain(float* s){
  unsigned a[4]={0x3f803f80u,0x3f803f81u,0x3f803f82u,0x3f803f83u}, b[2]={0x3f803f84u,0x3f803f85u};
  float c[NC][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    // NC chains, 16 steps each, chain order kept: step k of every chain, then step k+1
    #pragma unroll
    for(int k=0;k<16;k++){
      #pragma unroll
      for(int j=0;j<NC;j++) MMA(c[j],a,b);
    }
  }
  float t=0; for(int j=0;j<NC;j++) t+=c[j][0]+c[j][1]+c[j][2]+c[j][3];
  if(t==-1.0f) s[0]=t;
}
template<int IT> __global__ void k_pv2(float* s, float alpha){
  unsigned a[4]={0x3f803f80u,0x3f803f81u,0x3f803f82u,0x3f803f83u}, b[2]={0x3f803f84u,0x3f803f85u};
  float o[32][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int j=0;j<32;j+=2){
      float c0[4], c1[4];
      MMAZ(c0,a,b); MMAZ(c1,a,b);
      MMA(c0,a,b); MMA(c1,a,b);
      #pragma unroll
      for(int l=0;l<4;l++){ o[j][l]=o[j][l]*alpha+c0[l]; o[j+1][l]=o[j+1][l]*alpha+c1[l]; }
    }
  }
  float t=0; for(int j=0;j<32;j++) t+=o[j][0]+o[j][1]+o[j][2]+o[j][3];
  if(t==-1.0f) s[0]=t;
}
template<int IT> __global__ void k_qkldsm(float* s){
  extern __shared__ __align__(16) unsigned char sm[];
  // 32 keys x 256 dims bf16, 16-byte chunks XOR-swizzled by (row & 7), as the K plane
  for(int i=threadIdx.x; i<32*256*2/16; i+=blockDim.x) ((uint4*)sm)[i]=make_uint4(0x3f803f80u+i,0x3f803f81u,0x3f803f82u,0x3f803f83u);
  __syncthreads();
  unsigned q[16][4];
  #pragma unroll
  for(int k=0;k<16;k++){ q[k][0]=0x3f803f80u+k; q[k][1]=0x3f803f81u; q[k][2]=0x3f803f82u; q[k][3]=0x3f803f83u; }
  const int lane=threadIdx.x&31;
  const uint32_t base=(uint32_t)__cvta_generic_to_shared(sm);
  float c[4][4]={};
  #pragma unroll 1
  for(int i=0;i<IT;i++){
    #pragma unroll
    for(int kg=0;kg<2;kg++){
      #pragma unroll
      for(int kt=0;kt<16;kt++){
        const int r=kg*16+(lane%16), ch=kt*2+lane/16;
        const uint32_t addr=base+r*512+((ch^(r&7))*16);
        unsigned d[4];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];":"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(addr));
        unsigned blo[2]={d[0],d[2]}, bhi[2]={d[1],d[3]};
        MMA(c[kg*2],q[kt],blo); MMA(c[kg*2+1],q[kt],bhi);
      }
    }
  }
  float t=0; for(int j=0;j<4;j++) t+=c[j][0]+c[j][1]+c[j][2]+c[j][3];
  if(t==-1.0f) s[0]=t;
}

template<class F> float timeit(F f, int reps){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int r=0;r<reps;r++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); return ms/reps;
}
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  const int sms=p.multiProcessorCount;
  printf("device %s sms %d\n", p.name, sms);
  float* s; CK(cudaMalloc(&s,64));
  constexpr int IT=2048;
  const double mf=16.0*8*16*2;   // FLOP per m16n8k16
  CK(cudaFuncSetAttribute(k_qkldsm<IT>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384));
  for(int w: {4,8}){
    const int blocks=sms, thr=32*w; const double nw=(double)blocks*w;
    float ms=timeit([&]{k_peak8<IT><<<blocks,thr>>>(s);},5);
    printf("peak8  warps/SM %d: %.1f TFLOPS\n", w, nw*IT*8*mf/ms/1e9);
    ms=timeit([&]{k_chain<IT,4><<<blocks,thr>>>(s);},5);
    printf("chain4 warps/SM %d: %.1f TFLOPS\n", w, nw*IT*16*4*mf/ms/1e9);
    ms=timeit([&]{k_chain<IT,2><<<blocks,thr>>>(s);},5);
    printf("chain2 warps/SM %d: %.1f TFLOPS\n", w, nw*IT*16*2*mf/ms/1e9);
    ms=timeit([&]{k_pv2<IT><<<blocks,thr>>>(s,0.999f);},5);
    printf("pv2    warps/SM %d: %.1f TFLOPS (two-MMA chains from zero + FFMA fold)\n", w, nw*IT*64*mf/ms/1e9);
    ms=timeit([&]{k_qkldsm<IT><<<blocks,thr,16384>>>(s);},5);
    printf("qkldsm warps/SM %d: %.1f TFLOPS (chain4, B by ldmatrix per MMA pair)\n", w, nw*IT*64*mf/ms/1e9);
  }
  CK(cudaGetLastError());
  return 0;
}
