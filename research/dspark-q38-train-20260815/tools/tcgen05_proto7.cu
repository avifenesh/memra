// tcgen05 rung 7 — NVFP4 exact program (memra sm_100a lane, 2026-08-15).
// D[128x8](f32) = (A[128x64](e2m1) * SFA(ue4m3 per 16)) x (B[8x64](e2m1) * SFB)
// kind::mxf4nvf4.block_scale.scale_vec::4X — K=64 values = 32 bytes/row (2 k-cores), 4 scale
// blocks of 16 per row per MMA.
// SF-4X layout hypothesis (extension of the rung-2 atom): row 32q+l's FOUR scale bytes occupy
// the full 4-byte quad at tile[l*16 + q*4 .. +3] of the warpx4-dup'd 32x16B tile.
// sfmode 0: quad layout above. sfmode 1: strided (block j at tile[l*16 + j*4 + q]) — refuter.
// Owner context (2026-08-15): "if b200 is bw nvfp is plausible" + "we already support st nvfp".
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

static float e2m1_to_f32(uint8_t nib) {
  static const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float v = lut[nib & 7];
  return (nib & 8) ? -v : v;
}
static float ue4m3_to_f32(uint8_t v) {
  int e = (v >> 3) & 0xF, m = v & 7;
  if (e == 0) return (m / 8.0f) * (1.0f / 64.0f);
  return (1.0f + m / 8.0f) * __builtin_powf(2.0f, (float)(e - 7));
}

constexpr int M = 128, N = 8, K = 64;          // values; bytes per row = K/2 = 32

__device__ static inline uint64_t sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)((lbo   & 0x3FFFF) >> 4) << 16;
  d |= (uint64_t)((sbo   & 0x3FFFF) >> 4) << 32;
  d |= (uint64_t)0b001 << 46;
  return d;
}
__device__ static inline uint32_t idesc_mxf4nvf4(int m, int n, uint32_t sftype, uint32_t tcode) {
  uint32_t d = 0;
  d |= (tcode & 7u) << 7;                      // atype (sweep)
  d |= (tcode & 7u) << 10;                     // btype (sweep)
  d |= ((uint32_t)(n >> 3) & 0x3F) << 17;
  d |= (sftype & 1u) << 23;
  d |= ((uint32_t)(m >> 7) & 0x3) << 27;
  return d;
}

extern "C" __global__ void proto7(const uint8_t* A, const uint8_t* B, float* D,
                                  const uint8_t* SFA, const uint8_t* SFB,
                                  uint32_t sfmode, uint32_t sftype, uint32_t tcode) {
  __shared__ __align__(128) uint8_t sA[M * (K / 2)];   // 4096B: 2 k-cores per 8-row group
  __shared__ __align__(128) uint8_t sB[256];
  __shared__ __align__(128) uint8_t sSFA[2048];
  __shared__ __align__(128) uint8_t sSFB[2048];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];
  int tid = threadIdx.x;

  for (int i = tid; i < M * (K / 2); i += blockDim.x) {
    int r = i / (K / 2), c = i % (K / 2);
    sA[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = A[i];
  }
  for (int i = tid; i < N * (K / 2); i += blockDim.x) {
    int r = i / (K / 2), c = i % (K / 2);
    sB[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = B[i];
  }
  for (int i = tid; i < 2048; i += blockDim.x) { sSFA[i] = 0; sSFB[i] = 0; }
  __syncthreads();
  // SFA[row*4 + j] = scale for row, 16-value block j
  if (tid < M) {
    int q = tid / 32, l = tid % 32;
    for (int j = 0; j < 4; j++) {
      int off = (sfmode == 0) ? l * 16 + q * 4 + j
              : (sfmode == 1) ? l * 16 + j * 4 + q
              : j * 512 + l * 16 + q * 4;          // sfmode 2: one tile per 16-value block
      sSFA[off] = SFA[tid * 4 + j];
    }
  }
  if (tid < N)
    for (int j = 0; j < 4; j++) {
      int off = (sfmode == 2) ? j * 512 + tid * 16 : tid * 16 + ((sfmode == 0) ? j : j * 4);
      sSFB[off] = SFB[tid * 4 + j];
    }
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();

  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr;
  uint32_t sfa_tmem = (sfmode == 2) ? taddr + 16 : taddr + 8;
  uint32_t sfb_tmem = (sfmode == 2) ? taddr + 32 : taddr + 12;

  if (tid == 0) {
    uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA), 128, 256);
    uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB), 128, 256);
    uint32_t idesc = idesc_mxf4nvf4(M, N, sftype, tcode);
    if (sfmode == 2) {
      for (int j = 0; j < 4; j++) {
        uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA) + j * 512, 16, 128);
        uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB) + j * 512, 16, 128);
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem + 4 * j), "l"(sfad));
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem + 4 * j), "l"(sfbd));
      }
    } else {
      uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA), 16, 128);
      uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB), 16, 128);
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfad));
      asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfbd));
    }
    asm volatile(
      "{.reg .pred p; setp.eq.u32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
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
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

extern "C" __global__ void proto7_mxf4(const uint8_t* A, const uint8_t* B, float* D,
                                  const uint8_t* SFA, const uint8_t* SFB,
                                  uint32_t sfmode, uint32_t sftype, uint32_t tcode) {
  __shared__ __align__(128) uint8_t sA[M * (K / 2)];   // 4096B: 2 k-cores per 8-row group
  __shared__ __align__(128) uint8_t sB[256];
  __shared__ __align__(128) uint8_t sSFA[2048];
  __shared__ __align__(128) uint8_t sSFB[2048];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t taddr_slot[1];
  int tid = threadIdx.x;

  for (int i = tid; i < M * (K / 2); i += blockDim.x) {
    int r = i / (K / 2), c = i % (K / 2);
    sA[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = A[i];
  }
  for (int i = tid; i < N * (K / 2); i += blockDim.x) {
    int r = i / (K / 2), c = i % (K / 2);
    sB[(r / 8) * 256 + (c / 16) * 128 + (r % 8) * 16 + (c % 16)] = B[i];
  }
  for (int i = tid; i < 2048; i += blockDim.x) { sSFA[i] = 0; sSFB[i] = 0; }
  __syncthreads();
  // SFA[row*4 + j] = scale for row, 16-value block j
  if (tid < M) {
    int q = tid / 32, l = tid % 32;
    for (int j = 0; j < 4; j++) {
      int off = (sfmode == 0) ? l * 16 + q * 4 + j : l * 16 + j * 4 + q;
      sSFA[off] = 0x7F;
    }
  }
  if (tid < N)
    for (int j = 0; j < 4; j++)
      sSFB[tid * 16 + ((sfmode == 0) ? j : j * 4)] = 0x7F;
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();

  if (tid < 32)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(taddr_slot)));
  __syncthreads();
  uint32_t taddr = taddr_slot[0];
  uint32_t d_tmem = taddr, sfa_tmem = taddr + 8, sfb_tmem = taddr + 12;

  if (tid == 0) {
    uint64_t adesc = sdesc((uint32_t)__cvta_generic_to_shared(sA), 128, 256);
    uint64_t bdesc = sdesc((uint32_t)__cvta_generic_to_shared(sB), 128, 256);
    uint64_t sfad = sdesc((uint32_t)__cvta_generic_to_shared(sSFA), 16, 128);
    uint64_t sfbd = sdesc((uint32_t)__cvta_generic_to_shared(sSFB), 16, 128);
    uint32_t idesc = idesc_mxf4nvf4(M, N, sftype, tcode);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_tmem), "l"(sfad));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_tmem), "l"(sfbd));
    asm volatile(
      "{.reg .pred p; setp.eq.u32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
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
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(taddr));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}


int main(int argc, char** argv) {
  uint32_t sfmode = argc > 1 ? (uint32_t)atoi(argv[1]) : 0;
  uint32_t sftype = argc > 2 ? (uint32_t)atoi(argv[2]) : 1;
  uint32_t tcode = argc > 3 ? (uint32_t)atoi(argv[3]) : 0;
  uint32_t kmode_pre = argc > 4 ? (uint32_t)atoi(argv[4]) : 0;
  static uint8_t hA[M * K / 2], hB[N * K / 2], hSFA[M * 4], hSFB[N * 4];
  srand(7);
  for (auto& v : hA) v = (uint8_t)(rand() & 0x77);        // both nibbles positive e2m1
  for (auto& v : hB) v = (uint8_t)(rand() & 0x77);
  for (auto& v : hSFA) v = (uint8_t)(0x30 + rand() % 32); // ue4m3 exponents around 2^-1..2^0
  for (auto& v : hSFB) v = (uint8_t)(0x30 + rand() % 32);
  uint8_t *dA, *dB, *dSFA, *dSFB; float* dD;
  CK(cudaMalloc(&dA, sizeof hA)); CK(cudaMalloc(&dB, sizeof hB));
  CK(cudaMalloc(&dSFA, sizeof hSFA)); CK(cudaMalloc(&dSFB, sizeof hSFB));
  CK(cudaMalloc(&dD, M * N * 4));
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFA, hSFA, sizeof hSFA, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dSFB, hSFB, sizeof hSFB, cudaMemcpyHostToDevice));
  uint32_t kmode = kmode_pre;
  if (kmode == 1) proto7_mxf4<<<1, 128>>>(dA, dB, dD, dSFA, dSFB, sfmode, sftype, tcode);
  else proto7<<<1, 128>>>(dA, dB, dD, dSFA, dSFB, sfmode, sftype, tcode);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float hD[M * N];
  CK(cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int j = 0; j < 4; j++) {                        // four 16-value scale blocks
      float b = 0;
      for (int k = 0; k < 16; k++) {
        int kv = j * 16 + k;
        uint8_t ab = hA[m * (K / 2) + kv / 2], bb = hB[n * (K / 2) + kv / 2];
        float av = e2m1_to_f32((kv & 1) ? (ab >> 4) : (ab & 0xF));
        float bv = e2m1_to_f32((kv & 1) ? (bb >> 4) : (bb & 0xF));
        b += av * bv;
      }
      float sa = ue4m3_to_f32(hSFA[m * 4 + j]), sb = ue4m3_to_f32(hSFB[n * 4 + j]);
      if (kmode == 1) { sa = 1.0f; sb = 1.0f; }
      acc += b * sa * sb;
    }
    float got = hD[m * N + n];
    if (fabsf(got - acc) > 1e-4f * fabsf(acc) + 1e-3f && ++bad <= 6)
      printf("MISMATCH m=%d n=%d want %.4f got %.4f ratio %.4f\n", m, n, acc, got, got / acc);
  }
  printf(bad ? "sfmode=%u sftype=%u tc FAIL %d/1024\n" : "sfmode=%u sftype=%u PROTO7-OK 1024/1024 NVFP4 mxf4nvf4 4X\n",
         sfmode, sftype, bad);
  return bad != 0;
}
