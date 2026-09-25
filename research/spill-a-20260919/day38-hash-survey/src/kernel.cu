// WP-A day 38 survey: the receipt program (memra_tier::contracts::checksum) on the device. SHA-256 over the frame
// "memra-tier\0v1\0" (14 bytes) || le64(11) || "valid-bytes" (11 bytes) || le64(len) || payload, one thread per item.
// The 41-byte prefix puts payload byte j at stream offset 41 + j; after the first block every 64-byte block is payload
// bytes [64b - 41, 64b + 23), read as 17 aligned 32-bit words funnelled into 16 big-endian words.
typedef unsigned int u32;
typedef unsigned long long u64;
__constant__ u32 K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
__device__ __forceinline__ u32 rotr(u32 x, u32 n) { return __funnelshift_r(x, x, n); }
__device__ __forceinline__ void compress(u32* h, const u32* m) {
    u32 w[16];
#pragma unroll
    for (int t = 0; t < 16; t++) w[t] = m[t];
    u32 a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
#pragma unroll
    for (int t = 0; t < 64; t++) {
        u32 wt;
        if (t < 16) {
            wt = w[t];
        } else {
            u32 w15 = w[(t + 1) & 15], w2 = w[(t + 14) & 15];
            u32 s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >> 3);
            u32 s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >> 10);
            wt = w[t & 15] + s0 + w[(t + 9) & 15] + s1;
            w[t & 15] = wt;
        }
        u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        u32 ch = (e & f) ^ (~e & g);
        u32 t1 = hh + S1 + ch + K256[t] + wt;
        u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        u32 mj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
// Byte k of the framed stream for an item of `len` payload bytes (prefix, payload, padding), k < total padded length.
__device__ __forceinline__ unsigned char stream_byte(const unsigned char* p, u64 len, u64 k, u64 padded) {
    const unsigned char pre[14] = {'m','e','m','r','a','-','t','i','e','r',0,'v','1',0};
    const unsigned char dom[11] = {'v','a','l','i','d','-','b','y','t','e','s'};
    if (k < 14) return pre[k];
    if (k < 22) return (k == 14) ? 11 : 0;
    if (k < 33) return dom[k - 22];
    if (k < 41) return (unsigned char)((len >> (8 * (k - 33))) & 0xff);
    u64 j = k - 41;
    if (j < len) return p[j];
    if (j == len) return 0x80;
    u64 bits = (41 + len) * 8;
    if (k >= padded - 8) return (unsigned char)((bits >> (8 * (padded - 1 - k))) & 0xff);
    return 0;
}
extern "C" __global__ void framed_sha256(const u64* ptrs, const u64* lens, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned char* p = (const unsigned char*)ptrs[i];
    u64 len = lens[i];
    u64 total = 41 + len;
    u64 padded = ((total + 8) / 64 + 1) * 64;
    u32 h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    u64 blocks = padded / 64;
    // Blocks fully inside the payload: stream [64b, 64b + 64) with 64b >= 41 and 64b + 64 <= 41 + len.
    u64 b = 0;
    u32 m[16];
    for (; b < blocks; b++) {
        u64 s = 64 * b;
        bool fast = (s >= 41) && (s + 64 <= total);
        if (fast) {
            const unsigned char* q = p + (s - 41);
            u64 addr = (u64)q;
            const u32* base = (const u32*)(addr & ~3ull);
            u32 off = (u32)(addr & 3);
            u32 wv[17];
#pragma unroll
            for (int t = 0; t < 16; t++) wv[t] = base[t];
            // The 17th aligned word holds window bytes only when the window is not word aligned; reading it
            // otherwise could pass the payload's end.
            wv[16] = off ? base[16] : 0u;
            // Big-endian word t = stream bytes q[4t .. 4t+4). From aligned words a, b = wv[t], wv[t+1] and offset off:
            // byte k of the window (little-endian index) is __byte_perm(a, b, ...) selector off + k.
            u32 sel = (off + 3) | ((off + 2) << 4) | ((off + 1) << 8) | (off << 12);
#pragma unroll
            for (int t = 0; t < 16; t++) m[t] = __byte_perm(wv[t], wv[t + 1], sel);
        } else {
#pragma unroll 1
            for (int t = 0; t < 16; t++) {
                u32 w = 0;
                for (int k = 0; k < 4; k++) w = (w << 8) | stream_byte(p, len, s + 4 * t + k, padded);
                m[t] = w;
            }
        }
        compress(h, m);
    }
    for (int t = 0; t < 8; t++) {
        out[32 * i + 4 * t + 0] = (unsigned char)(h[t] >> 24);
        out[32 * i + 4 * t + 1] = (unsigned char)(h[t] >> 16);
        out[32 * i + 4 * t + 2] = (unsigned char)(h[t] >> 8);
        out[32 * i + 4 * t + 3] = (unsigned char)(h[t]);
    }
}
