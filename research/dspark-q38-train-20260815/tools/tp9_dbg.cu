// Probe: does the mxf4nvf4 MMA write D at all? Identity-ish: all SF bytes 0x7F would be huge for
// ue4m3 (2^8); use 0x38=2^0 as ue4m3 identity. All A/B nibbles = 0x2 (1.0). Expect D = K * 1 = 64.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)
__device__ static inline uint64_t sdesc(uint32_t s, uint32_t l, uint32_t b) {
  return (uint64_t)((s & 0x3FFFF) >> 4) | ((uint64_t)((l & 0x3FFFF) >> 4) << 16)
       | ((uint64_t)((b & 0x3FFFF) >> 4) << 32) | ((uint64_t)0b001 << 46);
}
extern "C" __global__ void probe(float* D, uint32_t idesc, uint32_t sfbyte, uint32_t abyte) {
  __shared__ __align__(128) uint8_t sA[4096]; // 128x32B
  __shared__ __align__(128) uint8_t sB[256];
  __shared__ __align__(128) uint8_t sSF[512];
  __shared__ __align__(16) uint64_t mbar[1];
  __shared__ __align__(16) uint32_t ta[1];
  int tid = threadIdx.x;
  for (int i = tid; i < 4096; i += 128) sA[i] = (uint8_t)abyte;   // two e2m1 "1.0" per byte
  for (int i = tid; i < 256; i += 128) sB[i] = 0x38;
  for (int i = tid; i < 512; i += 128) sSF[i] = (uint8_t)sfbyte;
  if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  __syncthreads();
  asm volatile("fence.proxy.async;");
  __syncthreads();
  if (tid < 32) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"((uint32_t)__cvta_generic_to_shared(ta)));
  __syncthreads();
  uint32_t t = ta[0];
  if (tid == 0) {
    uint64_t ad = sdesc((uint32_t)__cvta_generic_to_shared(sA), 128, 256);
    uint64_t bd = sdesc((uint32_t)__cvta_generic_to_shared(sB), 128, 256);
    uint64_t sd = sdesc((uint32_t)__cvta_generic_to_shared(sSF), 16, 128);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(t + 8), "l"(sd));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(t + 12), "l"(sd));
    asm volatile("{.reg .pred p; setp.eq.u32 p, 1, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;}\n\t"
      :: "r"(t), "l"(ad), "l"(bd), "r"(idesc), "r"(t + 8), "r"(t + 12));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  }
  asm volatile("{.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra W;}\n" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
  asm volatile("tcgen05.fence::after_thread_sync;");
  __syncthreads();
  int warp = tid / 32, lane = tid % 32;
  uint32_t q = t + ((uint32_t)(32 * warp) << 16);
  uint32_t r0;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r0) : "r"(q));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  D[warp * 32 + lane] = __uint_as_float(r0);
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(t));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}
int main(int argc, char** argv) {
  uint32_t idesc = (uint32_t)strtoul(argv[1], 0, 0), sfbyte = (uint32_t)strtoul(argv[2], 0, 0);
  float* dD; CK(cudaMalloc(&dD, 128 * 4));
  CK(cudaMemset(dD, 0xEE, 128 * 4));      // poison: distinguishes "not written" from "wrote 0"
  uint32_t abyte = (uint32_t)strtoul(argv[3], 0, 0);
  probe<<<1, 128>>>(dD, idesc, sfbyte, abyte);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float h[128]; CK(cudaMemcpy(h, dD, sizeof h, cudaMemcpyDeviceToHost));
  printf("idesc=0x%08x sf=0x%02x D[0]=%g D[1]=%g D[64]=%g D[127]=%g\n", idesc, sfbyte, h[0], h[1], h[64], h[127]);
  return 0;
}
