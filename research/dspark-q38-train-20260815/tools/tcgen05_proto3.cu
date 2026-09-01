// tcgen05 rung 3 — K-loop accumulation with enable-input-d chaining (memra sm_100a lane, 2026-08-15).
// D[128x8](f32) = sum over 4 K-blocks: (A_kb[128x32](e4m3) * 2^(SFA[kb][m]-127))
//                                    x (B_kb[8x32](e4m3)  * 2^(SFB[kb][n]-127))
// This is the mmq_fp8_blk shape: block size 32 along K, per-(row,block) scales.
// Chain: 4 tcgen05.mma ops on the same accumulator, enable-input-d=0 for kb=0, =1 after;
// single commit at the end. cps and MMAs are same-CTA async ops -> issue-order execution.
// SF atom per rung 2: 32x16B warpx4 tile, scale for row 32q+l at byte l*16 + q*4.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

static float e4m3_to_f32(uint8_t v) {
  int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  if (e == 0xF && m == 7) return s ? -__builtin_nanf("") : __builtin_nanf("");
  float f;
  if (e == 0) f = (m / 8.0f) * (1.0f / 64.0f);
  else        f = (1.0f + m / 8.0f) * __builtin_powf(2.0f, (float)(e - 7));
  return s ? -f : f;
}
static float ue8m0_to_f32(uint8_t v) { return __builtin_powf(2.0f, (float)v - 127.0f); }

constexpr int M = 128, N = 8, KB = 32, NB = 4, K = KB * NB;

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
  d |= 1u << 23;                       // UE8M0
  d |= ((uint32_t)(m >> 7) & 0x3) << 27;
  return d;
}

extern "C" __global__ void proto3(const uint8_t* A, const uint8_t* B, float* D,
                                  const uint8_t* SFA, const uint8_t* SFB) {
  // per-block contiguous tiles, each staged in the rung-1 core-tiled layout
  __shared__ __align__(128) uint8_t sA[NB * M * KB];    // 4 x 4096B
  __shared__ __align__(128) uint8_t sB[NB * 256];       // 4 x 256B
  __shared__ __align__(128) uint8_t sSFA[NB * 512];     // 4 x (32x16B warpx4 tile)
  __shared__ __align__(128) uint8_t sSFB[NB * 512];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];
  int tid = threadIdx.x;

  // A[m][k] global K-major; block kb tile holds columns kb*32..kb*32+31
  for (int i = tid; i < M * K; i += blockDim.x) {
    int r = i / K, c = i % K, kb = c / KB, ck = c % KB;
    sA[kb * (M * KB) + (r / 8) * 256 + (ck / 16) * 128 + (r % 8) * 16 + (ck % 16)] = A[i];
  }
  for (int i = tid; i < N * K; i += blockDim.x) {
    int r = i / K, c = i % K, kb = c / KB, ck = c % KB;
    sB[kb * 256 + (r / 8) * 256 + (ck / 16) * 128 + (r % 8) * 16 + (ck % 16)] = B[i];
  }
  for (int i = tid; i < NB * 512; i += blockDim.x) { sSFA[i] = 0; sSFB[i] = 0; }
  __syncthreads();
  // SFA[kb*M + m], SFB[kb*N + n]; rung-2 atom placement per block tile
  if (tid < M) {
    int q = tid / 32, l = tid % 32;
    for (int kb = 0; kb < NB; kb++)
      sSFA[kb * 512 + l * 16 + q * 4] = SFA[kb * M + tid];
  }
  if (tid < N)
    for (int kb = 0; kb < NB; kb++)
      sSFB[kb * 512 + tid * 16] = SFB[kb * N + tid];
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();

  // 64 tmem cols: D 0-7, SFA blocks at 8/12/16/20, SFB blocks at 24/28/32/36
  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr;

  if (tid == 0) {
    uint32_t idesc = idesc_mxf8f6f4(M, N);
    for (int kb = 0; kb < NB; kb++) {
      uint32_t sfa_tmem = taddr + 8 + 4 * kb, sfb_tmem = taddr + 24 + 4 * kb;
      uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA) + kb * 512, 16, 128);
      uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB) + kb * 512, 16, 128);
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfad));
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfbd));
    }
    for (int kb = 0; kb < NB; kb++) {
      uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA) + kb * (M * KB), 128, 256);
      uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB) + kb * 256, 128, 256);
      uint32_t sfa_tmem = taddr + 8 + 4 * kb, sfb_tmem = taddr + 24 + 4 * kb;
      asm volatile(
        "{.reg .pred p; setp.ne.u32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;}\n\t"
        :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc),
           "r"(sfa_tmem), "r"(sfb_tmem), "r"((uint32_t)kb));
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  {
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("{.reg .pred p;\n\tWAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra WAIT;}\n\t" :: "r"(mb));
  }
  asm volatile("tcgen05.fence::after_thread_sync;");
  __syncthreads();

  int warp = tid / 32, lane = tid % 32;
  uint32_t q = d_tmem + ((uint32_t)(32 * warp) << 16);
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                 "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(q));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  float* out = D + (warp * 32 + lane) * N;
  for (int c = 0; c < 8; c++) memcpy(&out[c], &r[c], 4);
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

int main() {
  static uint8_t hA[M * K], hB[N * K], hSFA[NB * M], hSFB[NB * N];
  srand(7);
  for (auto& v : hA) v = (uint8_t)(rand() & 0x7F) & 0x77;
  for (auto& v : hB) v = (uint8_t)(rand() & 0x7F) & 0x77;
  for (auto& v : hSFA) v = (uint8_t)(124 + rand() % 7);   // 2^-3 .. 2^3
  for (auto& v : hSFB) v = (uint8_t)(124 + rand() % 7);
  uint8_t *dA, *dB, *dSFA, *dSFB; float* dD;
  CK(cudaMalloc(&dA, sizeof hA)); CK(cudaMalloc(&dB, sizeof hB));
  CK(cudaMalloc(&dSFA, sizeof hSFA)); CK(cudaMalloc(&dSFB, sizeof hSFB));
  CK(cudaMalloc(&dD, M * N * 4));
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFA, hSFA, sizeof hSFA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFB, hSFB, sizeof hSFB, cudaMemcpyHostToDevice));
  proto3<<<1, 128>>>(dA, dB, dD, dSFA, dSFB);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float hD[M * N];
  CK(cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int kb = 0; kb < NB; kb++) {
      float b = 0;
      for (int k = 0; k < KB; k++)
        b += e4m3_to_f32(hA[m * K + kb * KB + k]) * e4m3_to_f32(hB[n * K + kb * KB + k]);
      acc += b * ue8m0_to_f32(hSFA[kb * M + m]) * ue8m0_to_f32(hSFB[kb * N + n]);
    }
    float got = hD[m * N + n];
    if (fabsf(got - acc) > 1e-4f * fabsf(acc) + 1e-3f && ++bad <= 6)
      printf("MISMATCH m=%d n=%d want %.4f got %.4f ratio %.4f\n", m, n, acc, got, got / acc);
  }
  printf(bad ? "FAIL %d/1024\n" : "PROTO3-OK 1024/1024 K-loop x4 blocks, per-block scales\n", bad);
  return bad != 0;
}
