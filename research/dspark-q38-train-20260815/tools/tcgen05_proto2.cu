// tcgen05 rung 2 — REAL ue8m0 block scales (memra sm_100a lane, 2026-08-15).
// D[128x8](f32) = (A[128x32](e4m3) * 2^(SFA[m]-127)) x (B[8x32](e4m3) * 2^(SFB[n]-127))
// kind::mxf8f6f4.block_scale.scale_vec::1X, K=32 = one scale block.
// SF staging hypothesis: one tcgen05.cp.128x128b per SF tensor; smem core-tiled
// (8-row x 16B cores); scale byte for lane m at core-tiled offset, byte 0 of its 16B row
// (SFA_ID=0 selects byte sub-column 0 of the tmem cell).
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

constexpr int M = 128, N = 8, K = 32;

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

extern "C" __global__ void proto2(const uint8_t* A, const uint8_t* B, float* D,
                                  const uint8_t* SFA, const uint8_t* SFB, uint32_t sfmode) {
  __shared__ __align__(128) uint8_t sA[M * K];
  __shared__ __align__(128) uint8_t sB[256];
  __shared__ __align__(128) uint8_t sSFA[512];    // 32 rows x 16B tile, warpx4-dup'd
  __shared__ __align__(128) uint8_t sSFB[512];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];
  int tid = threadIdx.x;

  // core-tiled operand staging (8x16B cores; groups of 2 k-cores)
  for (int i = tid; i < M * K; i += blockDim.x) {
    int r = i / K, c = i % K;
    sA[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = A[i];
  }
  for (int i = tid; i < N * K; i += blockDim.x) {
    int r = i / K, c = i % K;
    sB[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = B[i];
  }
  // SF tile 32x16B (core-tiled == row-major for single-core-column tensor).
  // Rung-2 finding: quarter q of the MMA reads its scale from byte-column q of the
  // warpx4-dup'd row. CUTLASS atom: SF for row 32q+l at byte l*16 + q*stride.
  // sfmode 0: stride 4 (byte 4q). sfmode 1: stride 1 (byte q).
  for (int i = tid; i < 512; i += blockDim.x) { sSFA[i] = 0; sSFB[i] = 0; }
  __syncthreads();
  if (tid < M) {
    int q = tid / 32, l = tid % 32;
    sSFA[l * 16 + q * (sfmode == 0 ? 4 : 1)] = SFA[tid];
  }
  if (tid < N) sSFB[tid * 16] = SFB[tid];
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();

  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr, sfa_tmem = taddr + 8, sfb_tmem = taddr + 12;

  if (tid == 0) {
    uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA), 128, 256);
    uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB), 128, 256);
    uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA), 16, 128);
    uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB), 16, 128);
    uint32_t idesc = idesc_mxf8f6f4(M, N);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfad));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfbd));
    asm volatile(
      "{.reg .pred p; setp.eq.u32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], %1, %2, %3, [%4], [%5], p;}\n\t"
      :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem));
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
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

int main(int argc, char** argv) {
  uint32_t sfmode = argc > 1 ? (uint32_t)atoi(argv[1]) : 0;
  uint8_t hA[M * K], hB[N * K], hSFA[M], hSFB[N];
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
  proto2<<<1, 128>>>(dA, dB, dD, dSFA, dSFB, sfmode);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float hD[M * N];
  CK(cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int k = 0; k < K; k++) acc += e4m3_to_f32(hA[m * K + k]) * e4m3_to_f32(hB[n * K + k]);
    acc *= ue8m0_to_f32(hSFA[m]) * ue8m0_to_f32(hSFB[n]);
    float got = hD[m * N + n];
    if (fabsf(got - acc) > 1e-4f * fabsf(acc) + 1e-3f && ++bad <= 6)
      printf("MISMATCH m=%d n=%d want %.4f got %.4f ratio %.4f\n", m, n, acc, got, got / acc);
  }
  printf(bad ? "sfmode=%u FAIL %d/1024\n" : "sfmode=%u PROTO2-OK 1024/1024 with real scales\n", sfmode, bad);
  return bad != 0;
}
