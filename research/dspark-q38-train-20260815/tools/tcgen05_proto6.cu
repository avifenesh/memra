// tcgen05 rung 6 — TMA gmem->smem staging feeding block-scale MMAs (memra sm_100a lane, 2026-08-15).
// D[128x256] = sum_kb (A_kb * SFA) x (B_kb * SFB), K=128 as 4 blocks; A and B arrive via
// cp.async.bulk.tensor.2d (TMA), NO swizzle: k-core is the OUTER smem dimension, so each
// {16B x rows} TMA box lands exactly core-tiled. Descs: LBO = k-core stride, SBO = 128.
//   A smem: (c/16)*2048 + (r/8)*128 + (r%8)*16 + c%16   (128 rows -> k-core slab 2048B)
//   B smem: (c/16)*4096 + (r/8)*128 + (r%8)*16 + c%16   (256 rows -> k-core slab 4096B)
// 8 A-boxes {16,128} + 8 B-boxes {16,256}, one mbarrier expect_tx, then rung-4 MMA chain.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cuda.h>
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

constexpr int M = 128, N = 256, KB = 32, NB = 4, K = KB * NB;
constexpr int SA_BYTES = M * K, SB_BYTES = N * K;                 // 16K, 32K
constexpr int OFF_B = SA_BYTES, OFF_SFA = OFF_B + SB_BYTES;       // SFA 4x512
constexpr int OFF_SFB = OFF_SFA + NB * 512;                       // SFB 4x1024
constexpr int OFF_MBAR = OFF_SFB + NB * 1024;
constexpr int OFF_TADDR = OFF_MBAR + 16;
constexpr int SMEM_TOTAL = OFF_TADDR + 16;

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

extern "C" __global__ void proto6(const __grid_constant__ CUtensorMap mapA,
                                  const __grid_constant__ CUtensorMap mapB,
                                  float* D, const uint8_t* SFA, const uint8_t* SFB) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sA = smem;
  uint8_t* sB = smem + OFF_B;
  uint8_t* sSFA = smem + OFF_SFA;
  uint8_t* sSFB = smem + OFF_SFB;
  uint64_t* mbar = (uint64_t*)(smem + OFF_MBAR);
  uint32_t* taddr_slot = (uint32_t*)(smem + OFF_TADDR);
  int tid = threadIdx.x;
  uint32_t mb = (uint32_t)__cvta_generic_to_shared(mbar);
  uint32_t mb2 = mb + 8;

  for (int i = tid; i < NB * 512; i += blockDim.x) sSFA[i] = 0;
  for (int i = tid; i < NB * 1024; i += blockDim.x) sSFB[i] = 0;
  __syncthreads();
  if (tid < M) {
    int q = tid / 32, l = tid % 32;
    for (int kb = 0; kb < NB; kb++)
      sSFA[kb * 512 + l * 16 + q * 4] = SFA[kb * M + tid];
  }
  for (int n = tid; n < N; n += blockDim.x) {
    int half = n / 128, r = n % 128, q = r / 32, l = r % 32;
    for (int kb = 0; kb < NB; kb++)
      sSFB[kb * 1024 + half * 512 + l * 16 + q * 4] = SFB[kb * N + n];
  }
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb2));
  }
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();

  if (tid == 0) {
    // expect all TMA bytes on mbar, then issue 8 A-boxes + 8 B-boxes
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mb), "r"((uint32_t)(SA_BYTES + SB_BYTES)));
    for (int kc = 0; kc < K / 16; kc++) {
      uint32_t dstA = (uint32_t)__cvta_generic_to_shared(sA) + kc * 2048;
      asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :: "r"(dstA), "l"(&mapA), "r"(kc * 16), "r"(0), "r"(mb) : "memory");
      uint32_t dstB = (uint32_t)__cvta_generic_to_shared(sB) + kc * 4096;
      asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                   " [%0], [%1, {%2, %3}], [%4];"
                   :: "r"(dstB), "l"(&mapB), "r"(kc * 16), "r"(0), "r"(mb) : "memory");
    }
  }
  {
    asm volatile("{.reg .pred p;\n\tW0: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W0;}\n\t" :: "r"(mb));
  }
  __syncthreads();

  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr;

  if (tid == 0) {
    uint32_t idesc = idesc_mxf8f6f4(M, N);
    for (int kb = 0; kb < NB; kb++) {
      uint32_t sfa_tmem = taddr + 256 + 4 * kb;
      uint32_t sfb_tmem = taddr + 272 + 8 * kb;
      uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA) + kb * 512, 16, 128);
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfad));
      for (int half = 0; half < 2; half++) {
        uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB) + kb * 1024 + half * 512, 16, 128);
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem + 4 * half), "l"(sfbd));
      }
    }
    for (int kb = 0; kb < NB; kb++) {
      // block kb = k-cores 2kb, 2kb+1; k-core-outer layout: LBO = slab stride, SBO = 128
      uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA) + kb * 2 * 2048, 2048, 128);
      uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB) + kb * 2 * 4096, 4096, 128);
      uint32_t sfa_tmem = taddr + 256 + 4 * kb;
      uint32_t sfb_tmem = taddr + 272 + 8 * kb;
      asm volatile(
        "{.reg .pred p; setp.ne.u32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
        "[%0], %1, %2, %3, [%4], [%5], p;}\n\t"
        :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc),
           "r"(sfa_tmem), "r"(sfb_tmem), "r"((uint32_t)kb));
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mb2));
  }
  {
    asm volatile("{.reg .pred p;\n\tW1: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W1;}\n\t" :: "r"(mb2));
  }
  asm volatile("tcgen05.fence::after_thread_sync;");
  __syncthreads();

  int warp = tid / 32, lane = tid % 32;
  float* out = D + (warp * 32 + lane) * N;
  for (int ch = 0; ch < 32; ch++) {
    uint32_t q = d_tmem + ch * 8 + ((uint32_t)(32 * warp) << 16);
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(q));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    for (int c = 0; c < 8; c++) memcpy(&out[ch * 8 + c], &r[c], 4);
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

int main() {
  static uint8_t hA[M * K], hB[N * K], hSFA[NB * M], hSFB[NB * N];
  static float hD[M * N];
  srand(7);
  for (auto& v : hA) v = (uint8_t)(rand() & 0x7F) & 0x77;
  for (auto& v : hB) v = (uint8_t)(rand() & 0x7F) & 0x77;
  for (auto& v : hSFA) v = (uint8_t)(124 + rand() % 7);
  for (auto& v : hSFB) v = (uint8_t)(124 + rand() % 7);
  uint8_t *dA, *dB, *dSFA, *dSFB; float* dD;
  CK(cudaMalloc(&dA, sizeof hA)); CK(cudaMalloc(&dB, sizeof hB));
  CK(cudaMalloc(&dSFA, sizeof hSFA)); CK(cudaMalloc(&dSFB, sizeof hSFB));
  CK(cudaMalloc(&dD, M * N * 4));
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFA, hSFA, sizeof hSFA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFB, hSFB, sizeof hSFB, cudaMemcpyHostToDevice));

  // tensor maps: dim0 = K bytes (fastest), dim1 = rows; box {16, rows}
  CUtensorMap mapA, mapB;
  cuuint64_t gdimA[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gdimB[2] = {(cuuint64_t)K, (cuuint64_t)N};
  cuuint64_t gstride[1] = {(cuuint64_t)K};            // stride of dim1 in bytes
  cuuint32_t boxA[2] = {16, M}, boxB[2] = {16, N};
  cuuint32_t estride[2] = {1, 1};
  CUresult r1 = cuTensorMapEncodeTiled(&mapA, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dA,
      gdimA, gstride, boxA, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CUresult r2 = cuTensorMapEncodeTiled(&mapB, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, dB,
      gdimB, gstride, boxB, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r1 != CUDA_SUCCESS || r2 != CUDA_SUCCESS) { printf("tensormap encode fail %d %d\n", r1, r2); return 1; }

  CK(cudaFuncSetAttribute(proto6, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL));
  proto6<<<1, 128, SMEM_TOTAL>>>(mapA, mapB, dD, dSFA, dSFB);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
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
  printf(bad ? "FAIL %d/32768\n" : "PROTO6-OK 32768/32768 TMA-staged, k-core-outer descs\n", bad);
  return bad != 0;
}
