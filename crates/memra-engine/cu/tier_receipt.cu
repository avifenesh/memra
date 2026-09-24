// tier_receipt.cu: the receipt kernels of the host-tier contracts door (the D2D digest since WP-A day 22, memra#536
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

// d2h_receipt_sha256 (WP-A day 38, `DAY38.md` design G, `memra_tier::conformance::d2h_device_receipt`):
// the receipt program `memra_tier::contracts::checksum` on the device, byte for byte: SHA-256 over the
// frame "memra-tier\0v1\0" (14 bytes) || le64(11) || "valid-bytes" (11 bytes) || le64(len) || payload,
// one thread per item (a sequential chain per item; the items of a batch run in SIMT lockstep), up to
// RECEIPT_ITEMS items per launch passed BY VALUE (count, device pointers, lengths), 32 bytes out per item
// (the digest in its byte order). The 41-byte prefix puts payload byte j at stream offset 41 + j; after
// the first block every block that lies inside the payload is 64 bytes read as 16 aligned 32-bit words
// plus one more when the window is not word aligned, funnelled into 16 big-endian words; the first and
// the last blocks are assembled byte by byte. An item of length 0 or pointer 0 hashes the empty payload.
#define RECEIPT_ITEMS 64
struct ReceiptItems {
    unsigned long long n;
    unsigned long long ptr[RECEIPT_ITEMS];
    unsigned long long len[RECEIPT_ITEMS];
};
__constant__ unsigned int memra_receipt_k256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
__device__ __forceinline__ unsigned int memra_receipt_rotr(unsigned int x, unsigned int n) {
    return __funnelshift_r(x, x, n);
}
__device__ __forceinline__ void memra_receipt_compress(unsigned int* h, const unsigned int* m) {
    unsigned int w[16];
#pragma unroll
    for (int t = 0; t < 16; t++) w[t] = m[t];
    unsigned int a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
    for (int t = 0; t < 64; t++) {
        unsigned int wt;
        if (t < 16) {
            wt = w[t];
        } else {
            unsigned int w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
            unsigned int s0 = memra_receipt_rotr(w15, 7) ^ memra_receipt_rotr(w15, 18) ^ (w15 >> 3);
            unsigned int s1 = memra_receipt_rotr(w2, 17) ^ memra_receipt_rotr(w2, 19) ^ (w2 >> 10);
            wt = w[t & 15] + s0 + w[(t + 9) & 15] + s1;
            w[t & 15] = wt;
        }
        unsigned int S1 = memra_receipt_rotr(e, 6) ^ memra_receipt_rotr(e, 11) ^ memra_receipt_rotr(e, 25);
        unsigned int ch = (e & f) ^ (~e & g);
        unsigned int t1 = hh + S1 + ch + memra_receipt_k256[t] + wt;
        unsigned int S0 = memra_receipt_rotr(a, 2) ^ memra_receipt_rotr(a, 13) ^ memra_receipt_rotr(a, 22);
        unsigned int mj = (a & b) ^ (a & c) ^ (b & c);
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + S0 + mj;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
// Byte k of an item's framed, padded stream (prefix, payload, 0x80, zeros, the big-endian bit length).
__device__ __forceinline__ unsigned char memra_receipt_stream_byte(
    const unsigned char* p, unsigned long long len, unsigned long long k, unsigned long long padded) {
    const unsigned char pre[14] = {'m', 'e', 'm', 'r', 'a', '-', 't', 'i', 'e', 'r', 0, 'v', '1', 0};
    const unsigned char dom[11] = {'v', 'a', 'l', 'i', 'd', '-', 'b', 'y', 't', 'e', 's'};
    if (k < 14) return pre[k];
    if (k < 22) return (k == 14) ? 11 : 0;
    if (k < 33) return dom[k - 22];
    if (k < 41) return (unsigned char)((len >> (8 * (k - 33))) & 0xff);
    unsigned long long j = k - 41;
    if (j < len) return p[j];
    if (j == len) return 0x80;
    unsigned long long bits = (41 + len) * 8;
    if (k >= padded - 8) return (unsigned char)((bits >> (8 * (padded - 1 - k))) & 0xff);
    return 0;
}
__global__ void d2h_receipt_sha256(ReceiptItems items, unsigned char* __restrict__ out) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= items.n) return;
    const unsigned char* p = (const unsigned char*)items.ptr[i];
    const unsigned long long len = p ? items.len[i] : 0ULL;
    const unsigned long long total = 41 + len;
    const unsigned long long padded = ((total + 8) / 64 + 1) * 64;
    unsigned int h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                         0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    unsigned int m[16];
    for (unsigned long long b = 0; b < padded / 64; b++) {
        const unsigned long long s = 64 * b;
        if (s >= 41 && s + 64 <= total) {
            const unsigned long long addr = (unsigned long long)(p + (s - 41));
            const unsigned int* base = (const unsigned int*)(addr & ~3ULL);
            const unsigned int off = (unsigned int)(addr & 3ULL);
            unsigned int wv[17];
#pragma unroll
            for (int t = 0; t < 16; t++) wv[t] = base[t];
            // The 17th aligned word holds window bytes only when the window is not word aligned;
            // reading it otherwise could pass the payload's end.
            wv[16] = off ? base[16] : 0u;
            const unsigned int sel = (off + 3) | ((off + 2) << 4) | ((off + 1) << 8) | (off << 12);
#pragma unroll
            for (int t = 0; t < 16; t++) m[t] = __byte_perm(wv[t], wv[t + 1], sel);
        } else {
#pragma unroll 1
            for (int t = 0; t < 16; t++) {
                unsigned int w = 0;
                for (int k = 0; k < 4; k++)
                    w = (w << 8) | memra_receipt_stream_byte(p, len, s + 4 * t + k, padded);
                m[t] = w;
            }
        }
        memra_receipt_compress(h, m);
    }
    for (int t = 0; t < 8; t++) {
        out[32 * i + 4 * t + 0] = (unsigned char)(h[t] >> 24);
        out[32 * i + 4 * t + 1] = (unsigned char)(h[t] >> 16);
        out[32 * i + 4 * t + 2] = (unsigned char)(h[t] >> 8);
        out[32 * i + 4 * t + 3] = (unsigned char)(h[t]);
    }
}

// tier_flip_byte (WP-A day 38, the `d2h-source-flip` fault's red arm): XOR one byte at `p` with 0x40, one
// thread, queued on the copy stream after the receipt digest and before the copy. Diagnostics only.
__global__ void tier_flip_byte(unsigned char* p) { p[0] ^= 0x40; }

__global__ void tier_delay_spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

}  // extern "C"
