// tcgen05 rung 5 — throughput: block-scale tcgen05 GEMM mainloop vs shipped plain mma.sync arm
// (memra sm_100a lane, 2026-08-15). Direction cell (co-tenant box): per-SM + chip FLOP rates.
//   T1: dependent-chain tcgen05.mma kind::mxf8f6f4.block_scale (M=128,N=256,K=32/instr),
//       CHUNK uncommitted per commit/wait — the CUTLASS mainloop shape.
//   T2: plain mma.sync.m16n8k32.kind::f8f6f4 (the -DMEMRA_FP8BLK_PLAIN_MMA arm), 16 warps.
// Usage: bench5 <mode 1|2> [ctas] [iters]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

constexpr int M = 128, N = 256, KB = 32;

__device__ static inline uint64_t sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo   & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo   & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)0b001 << 46;
  return d;
}
__device__ static inline uint32_t idesc_mxf8f6f4(int m, int n) {
  uint32_t d = 0;
  d |= ((uint32_t)(n >> 3) & 0x3F) << 17;
  d |= 1u << 23;
  d |= ((uint32_t)(m >> 7) & 0x3) << 27;
  return d;
}

// T1: tcgen05 block-scale mainloop. iters chunks x CHUNK MMAs, one commit/wait per chunk.
constexpr int CHUNK = 128;
extern "C" __global__ void t1(const uint8_t* A, const uint8_t* B, float* D, int iters) {
  __shared__ __align__(128) uint8_t sA[M * KB];
  __shared__ __align__(128) uint8_t sB[N * KB];
  __shared__ __align__(128) uint8_t sSF[512];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];
  int tid = threadIdx.x;
  for (int i = tid; i < M * KB; i += blockDim.x) sA[i] = A[i];
  for (int i = tid; i < N * KB; i += blockDim.x) sB[i] = B[i];
  for (int i = tid; i < 512; i += blockDim.x) sSF[i] = 0x7F;
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();
  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr, sfa_tmem = taddr + 256, sfb_tmem = taddr + 264;
  if (tid == 0) {
    uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA), 128, 256);
    uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB), 128, 256);
    uint64_t sfd = sdesc((uint32_t)__cvta_generic_to_shared(sSF), 16, 128);
    uint32_t idesc = idesc_mxf8f6f4(M, N);
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfd));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfd));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem + 4), "l"(sfd));
    for (int it = 0; it < iters; it++) {
      for (int c = 0; c < CHUNK; c++) {
        asm volatile(
          "{.reg .pred p; setp.ne.u32 p, %6, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
          "[%0], %1, %2, %3, [%4], [%5], p;}\n\t"
          :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc),
             "r"(sfa_tmem), "r"(sfb_tmem), "r"((uint32_t)(it + c)));
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mb));
      asm volatile("{.reg .pred p;\n\tW1: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W1;}\n\t"
                   :: "r"(mb), "r"((uint32_t)(it & 1)));
    }
  }
  asm volatile("tcgen05.fence::after_thread_sync;");
  __syncthreads();
  int warp = tid / 32, lane = tid % 32;
  uint32_t q = d_tmem + ((uint32_t)(32 * warp) << 16);
  uint32_t r0;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(q));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  if (lane == 0) D[blockIdx.x * 4 + warp] = __uint_as_float(r0);
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

// T2: plain mma.sync m16n8k32 kind::f8f6f4 dependent-accumulator loop, 16 warps.
extern "C" __global__ void t2(const uint8_t* A, float* D, int iters) {
  uint32_t a0 = A[threadIdx.x] * 0x01010101u, a1 = a0 ^ 0x33221100u,
           a2 = a0 + 7, a3 = a1 + 13, b0 = a2 ^ a1, b1 = a0 + a3;
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  for (int it = 0; it < iters; it++) {
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  if ((threadIdx.x & 31) == 0) D[blockIdx.x * 32 + threadIdx.x / 32] = d0 + d1 + d2 + d3;
}

int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 1;
  int ctas = argc > 2 ? atoi(argv[2]) : 148;
  int iters = argc > 3 ? atoi(argv[3]) : (mode == 1 ? 512 : 262144);
  static uint8_t hA[N * KB];
  srand(7);
  for (auto& v : hA) v = (uint8_t)(rand() & 0x7F) & 0x77;
  uint8_t* dA; float* dD;
  CK(cudaMalloc(&dA, sizeof hA)); CK(cudaMalloc(&dD, ctas * 512 * 4));
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double flops;
  // warmup
  if (mode == 1) { t1<<<ctas, 128>>>(dA, dA, dD, 8); }
  else           { t2<<<ctas, 512>>>(dA, dD, 4096); }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(e0));
  if (mode == 1) {
    t1<<<ctas, 128>>>(dA, dA, dD, iters);
    flops = (double)ctas * iters * CHUNK * 2.0 * M * N * KB;
  } else {
    t2<<<ctas, 512>>>(dA, dD, iters);
    flops = (double)ctas * 16.0 * iters * 2.0 * 16 * 8 * 32;
  }
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
  double tf = flops / (ms * 1e-3) / 1e12;
  printf("mode=%d ctas=%d iters=%d elapsed=%.2fms rate=%.1f TFLOP/s (%.1f%% of 4500 dense-fp8 peak)\n",
         mode, ctas, iters, ms, tf, 100.0 * tf / 4500.0);
  return 0;
}
