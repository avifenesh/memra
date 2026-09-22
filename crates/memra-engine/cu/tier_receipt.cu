// tier_receipt.cu: the D2D receipt kernels of the host-tier contracts door (WP-A day 22, memra#536
// Move 2 slice 3, `memra_tier::conformance::d2d_receipt_witnessed`). Receipts over KV BYTES, never a
// numeric program over tokens: nothing here reads a token, an activation or a KV encoding.
//
// d2d_receipt_digest: the copy-stream reduction whose CPU oracle is
// `memra_tier::conformance::receipt_digest`. The span is read as little-endian u64 words (the last
// partial word zero-padded); lane l accumulates mix64(w_j + (j + 1) * C_l) in wrapping u64 arithmetic
// for l = 0..4; the host folds the byte count (`receipt_digest_from_lanes`). Wrapping sums are
// order-independent, so the block and atomic order cannot move the value: the device answer IS the
// oracle's. `out` holds the four lane sums and must be zero on entry. Any alignment: the aligned
// fast path reads u64 words, the unaligned or tail path assembles bytes.
//
// tier_delay_spin: the `MEMRA_KV_HOST_FAULT=d2d-delay-*` fault's delay, one thread spinning on
// %globaltimer for `ns` nanoseconds on the copy stream ahead of the copy. Diagnostics only.
#include <stdint.h>

extern "C" {

__device__ __forceinline__ unsigned long long memra_receipt_mix64(unsigned long long z) {
    z ^= z >> 30;
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27;
    z *= 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z;
}

__global__ void d2d_receipt_digest(const unsigned char* __restrict__ p,
                                   unsigned long long n,
                                   unsigned long long* __restrict__ out) {
    const unsigned long long C0 = 0x9E3779B97F4A7C15ULL;
    const unsigned long long C1 = 0xC2B2AE3D27D4EB4FULL;
    const unsigned long long C2 = 0x165667B19E3779F9ULL;
    const unsigned long long C3 = 0x27D4EB2F165667C5ULL;
    const unsigned long long words = (n + 7ULL) / 8ULL;
    const unsigned long long full = n / 8ULL;
    const bool aligned = ((uintptr_t)p & 7ULL) == 0ULL;
    unsigned long long a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long w = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; w < words;
         w += stride) {
        unsigned long long v;
        if (aligned && w < full) {
            v = *reinterpret_cast<const unsigned long long*>(p + w * 8ULL);
        } else {
            v = 0ULL;
            const unsigned long long base = w * 8ULL;
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const unsigned long long i = base + (unsigned long long)b;
                if (i < n) v |= (unsigned long long)p[i] << (8 * b);
            }
        }
        const unsigned long long j1 = w + 1ULL;
        a0 += memra_receipt_mix64(v + j1 * C0);
        a1 += memra_receipt_mix64(v + j1 * C1);
        a2 += memra_receipt_mix64(v + j1 * C2);
        a3 += memra_receipt_mix64(v + j1 * C3);
    }
    // Block reduction (wrapping sums: any order), then one atomic per lane per block.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 += __shfl_down_sync(0xffffffffu, a0, o);
        a1 += __shfl_down_sync(0xffffffffu, a1, o);
        a2 += __shfl_down_sync(0xffffffffu, a2, o);
        a3 += __shfl_down_sync(0xffffffffu, a3, o);
    }
    __shared__ unsigned long long s[4][32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { s[0][warp] = a0; s[1][warp] = a1; s[2][warp] = a2; s[3][warp] = a3; }
    __syncthreads();
    if (warp == 0) {
        const int warps = (blockDim.x + 31) >> 5;
        unsigned long long b0 = lane < warps ? s[0][lane] : 0ULL;
        unsigned long long b1 = lane < warps ? s[1][lane] : 0ULL;
        unsigned long long b2 = lane < warps ? s[2][lane] : 0ULL;
        unsigned long long b3 = lane < warps ? s[3][lane] : 0ULL;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            b0 += __shfl_down_sync(0xffffffffu, b0, o);
            b1 += __shfl_down_sync(0xffffffffu, b1, o);
            b2 += __shfl_down_sync(0xffffffffu, b2, o);
            b3 += __shfl_down_sync(0xffffffffu, b3, o);
        }
        if (lane == 0) {
            atomicAdd(&out[0], b0);
            atomicAdd(&out[1], b1);
            atomicAdd(&out[2], b2);
            atomicAdd(&out[3], b3);
        }
    }
}

__global__ void tier_delay_spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

}  // extern "C"
