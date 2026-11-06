// ceil_probe3.cu: the F8F4 tile epilogue ceiling on GB10 with scalar FP32 (4 FMUL + 4 FFMA per
// MMA, ceil_probe2's f8tile row) against the packed f32x2 form (2 mul.rn.f32x2 + 2 fma.rn.f32x2
// per MMA: the same IEEE round-to-nearest product and fused add per lane, half the instructions).
// Also checks that the packed form returns the scalar form's bits on live operands.
// build: nvcc -gencode arch=compute_121a,code=sm_121a -O3 -o ceil_probe3 ceil_probe3.cu
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do{auto e=(x);if(e){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

#define MMA_F8BS(c,a,b) asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 " \
  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};" \
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(0x7F7F7F7Fu),"r"(0x7F7F7F7Fu))

static __device__ __forceinline__ unsigned long long pk(float lo, float hi) {
  unsigned long long r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi)); return r;
}
static __device__ __forceinline__ void upk(unsigned long long v, float & lo, float & hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(v));
}
static __device__ __forceinline__ unsigned long long mul2(unsigned long long a, unsigned long long b) {
  unsigned long long d; asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d;
}
static __device__ __forceinline__ unsigned long long fma2(unsigned long long a, unsigned long long b, unsigned long long c) {
  unsigned long long d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d;
}

template<int IT, bool PACKED> __global__ void k_f8tile(float* s, const float* sc){
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
      if constexpr (PACKED) {
        const unsigned long long d2 = pk(dB[j][0], dB[j][1]);
        const unsigned long long p0 = mul2(pk(dA0, dA0), d2), p1 = mul2(pk(dA1, dA1), d2);
        unsigned long long s01 = fma2(p0, pk(c[0], c[1]), pk(sum[j][0], sum[j][1]));
        unsigned long long s23 = fma2(p1, pk(c[2], c[3]), pk(sum[j][2], sum[j][3]));
        upk(s01, sum[j][0], sum[j][1]); upk(s23, sum[j][2], sum[j][3]);
      } else {
        sum[j][0] += dA0 * dB[j][0] * c[0];
        sum[j][1] += dA0 * dB[j][1] * c[1];
        sum[j][2] += dA1 * dB[j][0] * c[2];
        sum[j][3] += dA1 * dB[j][1] * c[3];
      }
    }
    dA0 = __int_as_float(__float_as_int(dA0) ^ (i & 1)); dA1 = __int_as_float(__float_as_int(dA1) ^ (i & 1));
    #pragma unroll
    for(int j=0;j<8;j++){ dB[j][0] = __int_as_float(__float_as_int(dB[j][0]) ^ (i & 1)); dB[j][1] = __int_as_float(__float_as_int(dB[j][1]) ^ (i & 1)); }
  }
  float t=0; for(int j=0;j<8;j++) for(int l=0;l<4;l++) t+=sum[j][l];
  if(t==-1.0f) s[0]=t;
}

// bit check: scalar (a*b)*c + s (contracted to mul + fma) against mul.rn.f32x2 + fma.rn.f32x2
__global__ void k_bits(const float* x, unsigned* bad, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  float a = x[4*i], b = x[4*i+1], c = x[4*i+2], s = x[4*i+3];
  float r1 = s; r1 += a * b * c;
  float lo, hi;
  upk(fma2(mul2(pk(a, b), pk(b, a)), pk(c, c), pk(s, s)), lo, hi);
  if (__float_as_uint(lo) != __float_as_uint(r1) || __float_as_uint(hi) != __float_as_uint(r1)) atomicAdd(bad, 1u);
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
  printf("device %s sms %d\n", p.name, sms);
  float* sc; CK(cudaMalloc(&sc,64*sizeof(float))); { float h[64]; for(int i=0;i<64;i++) h[i]=1.0f+i*1e-3f; CK(cudaMemcpy(sc,h,sizeof(h),cudaMemcpyHostToDevice)); }
  float* s; CK(cudaMalloc(&s,64));
  constexpr int IT=4096;
  for(int wpb: {4,8,16}){
    int blocks=sms*2, thr=32*wpb;
    double nw=(double)blocks*wpb, fl=nw*IT*8*16.0*8*32*2;
    float ms=timeit([&]{k_f8tile<IT,false><<<blocks,thr>>>(s,sc);},5);
    printf("f8tile scalar  warps/SM %2d: %.1f TFLOPS (4 FMUL + 4 FFMA per MMA)\n",2*wpb,fl/ms/1e9);
    ms=timeit([&]{k_f8tile<IT,true><<<blocks,thr>>>(s,sc);},5);
    printf("f8tile f32x2   warps/SM %2d: %.1f TFLOPS (2 mul.f32x2 + 2 fma.f32x2 per MMA)\n",2*wpb,fl/ms/1e9);
  }
  // bits on 16M random live operands (incl. subnormal, huge, negative)
  const int n = 1 << 24;
  float* hx = new float[4*(size_t)n]; unsigned st = 12345u;
  for (size_t i = 0; i < 4*(size_t)n; ++i) { st = st*1664525u + 1013904223u; unsigned u = st; if ((i & 7) == 0) u &= 0x80FFFFFFu; std::memcpy(&hx[i], &u, 4); if (hx[i] != hx[i]) hx[i] = 1.5f; }
  float* dx; unsigned* dbad; CK(cudaMalloc(&dx, 16*(size_t)n)); CK(cudaMalloc(&dbad, 4)); CK(cudaMemset(dbad, 0, 4));
  CK(cudaMemcpy(dx, hx, 16*(size_t)n, cudaMemcpyHostToDevice));
  k_bits<<<n/256, 256>>>(dx, dbad, n); unsigned hb; CK(cudaMemcpy(&hb, dbad, 4, cudaMemcpyDeviceToHost));
  printf("f32x2 vs scalar mul+fma bit mismatches: %u / %d\n", hb, 2*n);
  CK(cudaGetLastError());
  return 0;
}
