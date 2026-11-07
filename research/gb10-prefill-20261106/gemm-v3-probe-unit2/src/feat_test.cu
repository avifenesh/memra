// Which sm_121a PTX features a warp-specialized TMA tile can use: compile-only checks, one kernel each.
#include <cstdint>
#include <cuda.h>

__global__ void k_setmaxnreg() {
#if defined(T_SETMAXNREG)
    if (threadIdx.x < 128) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    }
#endif
}

__global__ void k_mbarrier(int* out) {
    __shared__ __align__(8) uint64_t bar;
    const unsigned b = (unsigned) __cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(0));
    }
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(b), "r"(0));
    }
    out[threadIdx.x] = (int) done;
}

__global__ void k_bulk(const int4* g, int* out) {
    __shared__ __align__(128) int4 buf[64];
    __shared__ __align__(8) uint64_t bar;
    const unsigned b = (unsigned) __cvta_generic_to_shared(&bar);
    const unsigned s = (unsigned) __cvta_generic_to_shared(buf);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;\n");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(1024));
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(s), "l"(g), "r"(1024), "r"(b) : "memory");
    }
    __syncthreads();
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(b), "r"(0));
    }
    out[threadIdx.x] = buf[threadIdx.x & 63].x;
}

__global__ void k_tma(const __grid_constant__ CUtensorMap map, int* out) {
    __shared__ __align__(128) uint8_t buf[256 * 64];
    __shared__ __align__(8) uint64_t bar;
    const unsigned b = (unsigned) __cvta_generic_to_shared(&bar);
    const unsigned s = (unsigned) __cvta_generic_to_shared(buf);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(1));
        asm volatile("fence.proxy.async.shared::cta;\n");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(256 * 64));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                     :: "r"(s), "l"(&map), "r"(0), "r"(0), "r"(b) : "memory");
    }
    __syncthreads();
    unsigned done = 0;
    while (!done) {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(b), "r"(0));
    }
    out[threadIdx.x] = buf[threadIdx.x];
}

__global__ void k_tma_multicast(const __grid_constant__ CUtensorMap map, int* out) {
#if defined(T_MULTICAST)
    __shared__ __align__(128) uint8_t buf[256 * 64];
    __shared__ __align__(8) uint64_t bar;
    const unsigned b = (unsigned) __cvta_generic_to_shared(&bar);
    const unsigned s = (unsigned) __cvta_generic_to_shared(buf);
    const uint16_t mask = 0x3;
    if (threadIdx.x == 0) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;\n"
                     :: "r"(s), "l"(&map), "r"(0), "r"(0), "r"(b), "h"(mask) : "memory");
    }
    out[threadIdx.x] = buf[threadIdx.x];
#endif
}

__global__ void k_elect(int* out) {
    unsigned is = 0;
    asm volatile("{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(is));
    out[threadIdx.x] = (int) is;
}
