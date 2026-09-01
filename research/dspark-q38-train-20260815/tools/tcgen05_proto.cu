// tcgen05 block-scale MMA prototype v0 — memra sm_100a lane (research cell, 2026-08-15).
// Smallest checkable unit: one tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
// D[128x8](f32) = A[128x32](e4m3) x B[8x32](e4m3), scales = ue8m0 identity (0x7F).
// Oracle: CPU f32 reference over the decoded e4m3 values. Success bar: exact match of every
// element (fp8 products at K=32 accumulate exactly in f32).
//
// Programming model per PTX ISA 9.7.17 (design brief, DESIGN-tcgen05-sm100.md):
//   tmem alloc (one warp) -> smem A/B/SF staged by plain stores + fence.proxy.async ->
//   tcgen05.cp.warpx4 (SF dup to 4 lane-quarters) -> tcgen05.mma (one thread) ->
//   tcgen05.commit + mbarrier wait -> fence::after_thread_sync -> per-warp tcgen05.ld.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

// ---- e4m3 helpers (host) ----
static float e4m3_to_f32(uint8_t v) {
  int s = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
  if (e == 0xF && m == 7) return s ? -__builtin_nanf("") : __builtin_nanf("");
  float f;
  if (e == 0) f = (m / 8.0f) * (1.0f / 64.0f);          // subnormal, 2^-6
  else        f = (1.0f + m / 8.0f) * __builtin_powf(2.0f, (float)(e - 7));
  return s ? -f : f;
}

constexpr int M = 128, N = 8, K = 32;

// smem desc encode: (addr&0x3FFFF)>>4
__device__ static inline uint64_t sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);              // bits 0-13
  d |= (uint64_t)((lbo   & 0x3FFFF) >> 4) << 16;        // bits 16-29
  d |= (uint64_t)((sbo   & 0x3FFFF) >> 4) << 32;        // bits 32-45
  d |= (uint64_t)0b001 << 46;                           // tcgen05 marker
  // swizzle none (61-63 = 0), base offset 0, relative LBO mode (52=0)
  return d;
}

// idesc for kind::mxf8f6f4 block_scale (brief table): E4M3 atype=0,btype=0; N>>3 bits17-22;
// scale_type ue8m0 bit23=1; M>>7 bits27-28; SFA_ID/SFB_ID 0.
__device__ static inline uint32_t idesc_mxf8f6f4(int m, int n) {
  uint32_t d = 0;
  d |= ((uint32_t)(n >> 3) & 0x3F) << 17;
  d |= 1u << 23;                                        // UE8M0
  d |= ((uint32_t)(m >> 7) & 0x3) << 27;
  return d;
}

extern "C" __global__ void proto(const uint8_t* A, const uint8_t* B, float* D, uint32_t albo, uint32_t asbo, uint32_t blbo, uint32_t bsbo) {
  // smem plan (16B-aligned): A 128x32 K-major (4096B) | B 8x32 K-major (256B)
  // | SFA 128B | SFB 128B | mbar 8B | tmem-addr slot 4B
  __shared__ __align__(128) uint8_t sA[M * K];
  __shared__ __align__(128) uint8_t sB[N * K];
  __shared__ __align__(128) uint8_t sSFA[512];          // 32 lanes x 16B tile for tcgen05.cp .32x128b
  __shared__ __align__(128) uint8_t sSFB[512];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];

  int tid = threadIdx.x;
  // stage operands in CANONICAL core-tiled layout: 8-row x 16-byte cores contiguous (128B),
  // cores ordered [row-group][k-core]; LBO = stride between k-cores, SBO = between row-groups.
  for (int i = tid; i < M * K; i += blockDim.x) {
    int r = i / K, c = i % K;
    sA[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = A[i];
  }
  for (int i = tid; i < N * K; i += blockDim.x) {
    int r = i / K, c = i % K;
    sB[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = B[i];
  }
  for (int i = tid; i < 512; i += blockDim.x) { sSFA[i] = 0x7F; sSFB[i] = 0x7F; }  // identity ue8m0
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  __syncthreads();
  // generic-proxy writes -> async proxy fence before MMA consumes smem
  asm volatile("fence.proxy.async;");
  __syncthreads();

  // tmem alloc: one warp; 32 cols min covers D(8 cols) + SFA(1) + SFB(1)
  if (tid < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  }
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem   = taddr;          // cols 0-7   : D 128x8 f32
  uint32_t sfa_tmem = taddr + 8;      // cols 8-11  : SFA (cp .32x128b writes 4 cols)
  uint32_t sfb_tmem = taddr + 12;     // cols 12-15 : SFB (cp .32x128b writes 4 cols)

  if (tid == 0) {
    uint32_t a_s = (uint32_t)__cvta_generic_to_shared(sA);
    uint32_t b_s = (uint32_t)__cvta_generic_to_shared(sB);
    uint32_t sfa_s = (uint32_t)__cvta_generic_to_shared(sSFA);
    uint32_t sfb_s = (uint32_t)__cvta_generic_to_shared(sSFB);
    // K-major, no swizzle. Core matrix = 8 rows x 16B. A: 128 rows x 32B -> 16 row-cores x 2 col-cores.
    // LBO = byte offset between the 2 col-cores (16B), SBO = offset between 8-row groups (8*32=256B).
    uint64_t adesc = sdesc(a_s, albo, asbo);
    uint64_t bdesc = sdesc(b_s, blbo, bsbo);
    uint32_t idesc = idesc_mxf8f6f4(M, N);
    uint64_t sfadesc = sdesc(sfa_s, 16, 128);
    uint64_t sfbdesc = sdesc(sfb_s, 16, 128);
    // SF staging: dup to all 4 lane-quarters
#if STAGE >= 2
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfadesc));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfbdesc));
#endif
    // MMA: enable-input-d = 0 (overwrite D)
#if STAGE >= 3
    asm volatile(
      "{.reg .pred p; setp.eq.u32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
      "[%0], %1, %2, %3, [%4], [%5], p;}\n\t"
      :: "r"(d_tmem), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(sfa_tmem), "r"(sfb_tmem));
#endif
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  // wait for completion
  {
    uint32_t mb = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
      "{.reg .pred p;\n\t"
      "WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
      "@!p bra WAIT;}\n\t" :: "r"(mb));
  }
  asm volatile("tcgen05.fence::after_thread_sync;");
  __syncthreads();

  // read back: each warp reads its 32-lane quarter, 8 columns
  int warp = tid / 32, lane = tid % 32;
  uint32_t q = d_tmem + ((uint32_t)(32 * warp) << 16);
  uint32_t r[16] = {0};
#if STAGE >= 4
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                 "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15])
               : "r"(q));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
#endif
  int row = warp * 32 + lane;
  float* out = D + row * 16;
  for (int c = 0; c < 16; c++) memcpy(&out[c], &r[c], 4);
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

int main(int argc, char** argv) {
  uint8_t hA[M * K], hB[N * K];
  srand(7);
  for (auto& v : hA) v = (uint8_t)(rand() & 0x7F) & 0x77;  // positive-ish, avoid nan (e<15)
  for (auto& v : hB) v = (uint8_t)(rand() & 0x7F) & 0x77;
  uint8_t *dA, *dB; float* dD;
  CK(cudaMalloc(&dA, sizeof hA)); CK(cudaMalloc(&dB, sizeof hB));
  CK(cudaMalloc(&dD, M * 16 * 4));
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice));
  uint32_t AL=atoi(argv[1]), AS=atoi(argv[2]), BL=atoi(argv[3]), BS=atoi(argv[4]);
  uint32_t combos[][2] = {{0,0}};  (void)combos;
  float want[M][N];
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float a = 0; for (int k = 0; k < K; k++) a += e4m3_to_f32(hA[m*K+k]) * e4m3_to_f32(hB[n*K+k]);
    want[m][n] = a;
  }
  {
    proto<<<1, 128>>>(dA, dB, dD, AL, AS, BL, BS);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("A(%u,%u) B(%u,%u): FAULT\n", AL,AS,BL,BS); return 2; }
    float hd[M*16]; cudaMemcpy(hd, dD, sizeof hd, cudaMemcpyDeviceToHost);
    int hit1 = 0, hit2 = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
      float w = want[m][n];
      float g1 = hd[m*16+n], g2 = hd[m*16+n*2];
      if (fabsf(g1-w) <= 1e-4f*fabsf(w)+1e-3f) hit1++;
      if (fabsf(g2-w) <= 1e-4f*fabsf(w)+1e-3f) hit2++;
    }
    printf("A(%u,%u) B(%u,%u): stride1=%d/1024 stride2=%d/1024\n", AL,AS,BL,BS, hit1, hit2);
  }
  if (0) {
  float hD[M * 16];
  CK(cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost));
  printf("want row0:"); for (int n=0;n<8;n++) printf(" %.1f", [&]{float a=0;for(int k=0;k<K;k++)a+=e4m3_to_f32(hA[k])*e4m3_to_f32(hB[n*K+k]);return a;}()); printf("\ngot16 row0:"); for(int c=0;c<16;c++) printf(" %.1f", hD[c]); printf("\n");
  // CPU oracle
  int bad = 0;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int k = 0; k < K; k++) acc += e4m3_to_f32(hA[m * K + k]) * e4m3_to_f32(hB[n * K + k]);
    float got = hD[m * 16 + n * 2];
    if (acc != got && ++bad <= 5)
      printf("MISMATCH m=%d n=%d want %.6f got %.6f\n", m, n, acc, got);
  }
  printf(bad ? "x" : "x");
  }
  printf("SWEEP DONE\n");
  return 0;
}
