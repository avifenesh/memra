// Standalone probe of mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
// on the card: (1) fragment layout and scale selectors (which lane / byte / thread-id gives which
// row's and column's scale), (2) unit scales from zero == the plain e4m3 MMA, (3) a K chain with the
// operands exchanged (weights on the m16 side) gives the same bytes as the plain roles, (4) e2m1 in an
// 8-bit container gives the same bytes as the same values expanded to e4m3.
// Build: nvcc -O2 -arch=sm_121a (GB10) or sm_120a; run with no arguments; exit 0 = every check passed.
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                         \
    do {                                                                              \
        cudaError_t e_ = (x);                                                         \
        if (e_ != cudaSuccess) {                                                      \
            printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

template <int BA, int TA, int BB, int TB>
__device__ __forceinline__ void mma_bs_ee(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1, const float c[4],
                                          uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {%16, %17}, %15, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sa), "r"(sb), "n"(BA), "n"(TA), "n"(BB), "n"(TB));
}
// A e4m3, B e2m1 (container) and A e2m1, B e4m3
__device__ __forceinline__ void mma_bs_e4_f4(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                             const float c[4], uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_bs_f4_e4(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                             const float c[4], uint32_t sa, uint32_t sb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sa), "r"(sb));
}
__device__ __forceinline__ void mma_plain(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

__device__ __forceinline__ uint32_t ld4(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// ---- (1) selectors: A [16][32], B [8][32] (column-major: B[col][k]), scales sa[16], sb[8], C [16][8].
// Hypothesis: thread-id-a = TA -> lane t = 2TA gives row g's scale, lane t = 2TA + 1 row g + 8's, at byte
// BA; thread-id-b = TB -> lane t = TB gives column g's scale, at byte BB. Every other byte of every lane
// holds a decoy code, so a wrong hypothesis shows as a wrong result.
template <int BA, int TA, int BB, int TB>
__global__ void k_sel(const uint8_t* A, const uint8_t* B, const uint8_t* sa, const uint8_t* sb, const float* C,
                      float* D) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t af[4] = {ld4(A + g * 32 + 4 * t), ld4(A + (g + 8) * 32 + 4 * t), ld4(A + g * 32 + 16 + 4 * t),
                      ld4(A + (g + 8) * 32 + 16 + 4 * t)};
    uint32_t b0 = ld4(B + g * 32 + 4 * t), b1 = ld4(B + g * 32 + 16 + 4 * t);
    float c[4] = {C[g * 8 + 2 * t], C[g * 8 + 2 * t + 1], C[(g + 8) * 8 + 2 * t], C[(g + 8) * 8 + 2 * t + 1]};
    uint32_t va = 0x6A6A6A6Au + lane, vb = 0x8F8F8F8Fu - lane;  // decoys
    if (t == 2 * TA) va = (va & ~(0xFFu << (8 * BA))) | ((uint32_t)sa[g] << (8 * BA));
    if (t == 2 * TA + 1) va = (va & ~(0xFFu << (8 * BA))) | ((uint32_t)sa[g + 8] << (8 * BA));
    if (t == TB) vb = (vb & ~(0xFFu << (8 * BB))) | ((uint32_t)sb[g] << (8 * BB));
    float d[4];
    mma_bs_ee<BA, TA, BB, TB>(d, af, b0, b1, c, va, vb);
    D[g * 8 + 2 * t] = d[0];
    D[g * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t] = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

// ---- (2) unit scales from zero against the plain MMA (one K block), outputs P (plain) and Q (scaled)
__global__ void k_unit(const uint8_t* A, const uint8_t* B, float* P, float* Q) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    uint32_t af[4] = {ld4(A + g * 32 + 4 * t), ld4(A + (g + 8) * 32 + 4 * t), ld4(A + g * 32 + 16 + 4 * t),
                      ld4(A + (g + 8) * 32 + 16 + 4 * t)};
    uint32_t b0 = ld4(B + g * 32 + 4 * t), b1 = ld4(B + g * 32 + 16 + 4 * t);
    float z[4] = {0, 0, 0, 0}, d[4], e[4];
    mma_plain(d, af, b0, b1);
    mma_bs_ee<0, 0, 0, 0>(e, af, b0, b1, z, 0x7F7F7F7Fu, 0x7F7F7F7Fu);
    for (int i = 0; i < 4; ++i) {
        int r = g + (i >= 2 ? 8 : 0), c = 2 * t + (i & 1);
        P[r * 8 + c] = d[i];
        Q[r * 8 + c] = e[i];
    }
}

// ---- (3)/(4) K chains. X: activations [16][K] (rows) e4m3 + scales [16][KB]; W: weights [16][K] e4m3 codes
// (the fp4 values expanded) and Wc: the same values as e2m1 containers (nibble << 2); weight scales [16][KB].
// Plain roles: A = activations rows 0..15, B = weight rows 0..7 as columns: out1[r][c] (16 x 8).
// Exchanged: A = weight rows 0..15, B = activation rows 0..7 as columns: out2[w][r] (16 x 8).
// mode 0: e4m3 both; mode 1: weights as e2m1 containers.
// Scale lane mapping per the selector probe's winning hypothesis (byte 0, thread-id 0).
__global__ void k_chain(const uint8_t* X, const uint8_t* Xs, const uint8_t* W, const uint8_t* Wc, const uint8_t* Ws,
                        int K, float* out1, float* out2, float* out3, float* out4, float* out5) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3, KB = K / 32;
    float a1[4] = {0, 0, 0, 0}, a2[4] = {0, 0, 0, 0}, a3[4] = {0, 0, 0, 0}, a4[4] = {0, 0, 0, 0}, a5[4] = {0, 0, 0, 0};
    for (int kb = 0; kb < KB; ++kb) {
        const uint8_t* x0 = X + (long)g * K + kb * 32;
        const uint8_t* x8 = X + (long)(g + 8) * K + kb * 32;
        const uint8_t* w0 = W + (long)g * K + kb * 32;
        const uint8_t* w8 = W + (long)(g + 8) * K + kb * 32;
        const uint8_t* c0 = Wc + (long)g * K + kb * 32;
        const uint8_t* c8 = Wc + (long)(g + 8) * K + kb * 32;
        uint32_t ax[4] = {ld4(x0 + 4 * t), ld4(x8 + 4 * t), ld4(x0 + 16 + 4 * t), ld4(x8 + 16 + 4 * t)};
        uint32_t aw[4] = {ld4(w0 + 4 * t), ld4(w8 + 4 * t), ld4(w0 + 16 + 4 * t), ld4(w8 + 16 + 4 * t)};
        uint32_t ac[4] = {ld4(c0 + 4 * t), ld4(c8 + 4 * t), ld4(c0 + 16 + 4 * t), ld4(c8 + 16 + 4 * t)};
        uint32_t bw0 = ld4(w0 + 4 * t), bw1 = ld4(w0 + 16 + 4 * t);  // weight row g as column g
        uint32_t bc0 = ld4(c0 + 4 * t), bc1 = ld4(c0 + 16 + 4 * t);
        uint32_t bx0 = ld4(x0 + 4 * t), bx1 = ld4(x0 + 16 + 4 * t);  // activation row g as column g
        // scales: lane t = 0 row g, t = 1 row g + 8 (A side); lane t = 0 column g (B side)
        uint32_t sxa = t == 0 ? Xs[g * KB + kb] : Xs[(g + 8) * KB + kb];
        uint32_t swa = t == 0 ? Ws[g * KB + kb] : Ws[(g + 8) * KB + kb];
        uint32_t swb = Ws[g * KB + kb], sxb = Xs[g * KB + kb];
        float d[4];
        mma_bs_ee<0, 0, 0, 0>(d, ax, bw0, bw1, a1, sxa, swb);
        for (int i = 0; i < 4; ++i) a1[i] = d[i];
        mma_bs_ee<0, 0, 0, 0>(d, aw, bx0, bx1, a2, swa, sxb);
        for (int i = 0; i < 4; ++i) a2[i] = d[i];
        mma_bs_e4_f4(d, ax, bc0, bc1, a3, sxa, swb);
        for (int i = 0; i < 4; ++i) a3[i] = d[i];
        mma_bs_f4_e4(d, ac, bx0, bx1, a4, swa, sxb);
        for (int i = 0; i < 4; ++i) a4[i] = d[i];
        // the old program: the block from zero, then acc + (d * sa) * sb on CUDA cores
        mma_plain(d, ax, bw0, bw1);
        for (int i = 0; i < 4; ++i) {
            int r = g + (i >= 2 ? 8 : 0), c = 2 * t + (i & 1);
            float sa = ldexpf(1.f, (int)Xs[r * KB + kb] - 127), sb = ldexpf(1.f, (int)Ws[c * KB + kb] - 127);
            a5[i] = __fadd_rn(a5[i], __fmul_rn(__fmul_rn(d[i], sa), sb));
        }
    }
    for (int i = 0; i < 4; ++i) {
        int r = g + (i >= 2 ? 8 : 0), c = 2 * t + (i & 1);
        out1[r * 8 + c] = a1[i];
        out2[r * 8 + c] = a2[i];
        out3[r * 8 + c] = a3[i];
        out4[r * 8 + c] = a4[i];
        out5[r * 8 + c] = a5[i];
    }
}

// ---- host
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() {
    rs = rs * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rs >> 33);
}
static double e4m3(uint8_t c) {
    int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    double v = e == 0 ? ldexp(m / 8.0, -6) : ldexp(1 + m / 8.0, e - 7);
    return s ? -v : v;
}
static double ue8(uint8_t c) { return ldexp(1.0, (int)c - 127); }
static uint8_t rnd_e4m3() {
    uint8_t b = rnd() & 0xFF;
    return (b & 0x7F) == 0x7F ? b ^ 1 : b;
}
static const uint8_t F4E4[16] = {0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
                                 0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC};

template <int BA, int TA, int BB, int TB>
static double run_sel(uint8_t* dA, uint8_t* dB, uint8_t* dsa, uint8_t* dsb, float* dC, float* dD, const double* ref) {
    k_sel<BA, TA, BB, TB><<<1, 32>>>(dA, dB, dsa, dsb, dC, dD);
    CK(cudaGetLastError());
    float h[128];
    CK(cudaMemcpy(h, dD, sizeof h, cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int i = 0; i < 128; ++i) {
        double r = fabs(h[i] - ref[i]) / fmax(fabs(ref[i]), 1e-30);
        if (!(r == r)) r = 1e30;
        worst = fmax(worst, r);
    }
    return worst;
}

int main() {
    int fails = 0;
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s sm_%d%d\n", pr.name, pr.major, pr.minor);
    // ---------- (1) selectors
    uint8_t A[16 * 32], B[8 * 32], sa[16], sb[8];
    float C[128];
    for (int i = 0; i < 16 * 32; ++i) A[i] = rnd_e4m3();
    for (int i = 0; i < 8 * 32; ++i) B[i] = rnd_e4m3();
    for (int i = 0; i < 16; ++i) sa[i] = 120 + i;       // distinct per row: 2^-7 .. 2^8
    for (int i = 0; i < 8; ++i) sb[i] = 112 + 2 * i;     // distinct per column
    for (int i = 0; i < 128; ++i) C[i] = (float)((int)(rnd() % 2001) - 1000) / 64.f;
    double ref[128];
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c) {
            double s = 0;
            for (int k = 0; k < 32; ++k) s += e4m3(A[r * 32 + k]) * e4m3(B[c * 32 + k]);
            ref[r * 8 + c] = s * ue8(sa[r]) * ue8(sb[c]) + C[r * 8 + c];
        }
    uint8_t *dA, *dB, *dsa, *dsb;
    float *dC, *dD;
    CK(cudaMalloc(&dA, sizeof A));
    CK(cudaMalloc(&dB, sizeof B));
    CK(cudaMalloc(&dsa, 16));
    CK(cudaMalloc(&dsb, 8));
    CK(cudaMalloc(&dC, sizeof C));
    CK(cudaMalloc(&dD, sizeof C));
    CK(cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsa, sa, 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsb, sb, 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, C, sizeof C, cudaMemcpyHostToDevice));
    double w;
#define SEL(BA, TA, BB, TB)                                                                    \
    w = run_sel<BA, TA, BB, TB>(dA, dB, dsa, dsb, dC, dD, ref);                                \
    printf("selector byte-a %d thread-a %d byte-b %d thread-b %d: max rel err %.3e %s\n", BA, TA, BB, TB, w, \
           w < 1e-4 ? "OK" : "FAIL");                                                          \
    if (!(w < 1e-4)) ++fails;
    SEL(0, 0, 0, 0) SEL(1, 0, 0, 0) SEL(2, 0, 0, 0) SEL(3, 0, 0, 0)
    SEL(0, 1, 0, 0) SEL(1, 1, 0, 0) SEL(2, 1, 0, 0) SEL(3, 1, 0, 0)
    SEL(0, 0, 1, 0) SEL(0, 0, 2, 0) SEL(0, 0, 3, 0)
    SEL(0, 0, 0, 1) SEL(0, 0, 0, 2) SEL(0, 0, 0, 3)
    SEL(3, 1, 3, 3) SEL(2, 1, 1, 2)
    // ---------- (2) unit scales from zero == plain MMA
    {
        float *dP, *dQ;
        CK(cudaMalloc(&dP, 512));
        CK(cudaMalloc(&dQ, 512));
        int diff = 0;
        for (int rep = 0; rep < 64; ++rep) {
            for (int i = 0; i < 16 * 32; ++i) A[i] = rnd_e4m3();
            for (int i = 0; i < 8 * 32; ++i) B[i] = rnd_e4m3();
            CK(cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice));
            k_unit<<<1, 32>>>(dA, dB, dP, dQ);
            CK(cudaGetLastError());
            uint32_t p[128], q[128];
            CK(cudaMemcpy(p, dP, 512, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(q, dQ, 512, cudaMemcpyDeviceToHost));
            for (int i = 0; i < 128; ++i) diff += p[i] != q[i];
        }
        printf("unit scales from zero vs plain e4m3 MMA: %d of %d outputs differ %s\n", diff, 64 * 128,
               diff ? "(INFO)" : "(identical)");
    }
    // ---------- (3)/(4) chains
    {
        const int K = 5120, KB = K / 32;
        size_t n = 16 * (size_t)K;
        uint8_t* X = (uint8_t*)malloc(n);
        uint8_t* W = (uint8_t*)malloc(n);
        uint8_t* Wc = (uint8_t*)malloc(n);
        uint8_t* Xs = (uint8_t*)malloc(16 * KB);
        uint8_t* Ws = (uint8_t*)malloc(16 * KB);
        uint8_t *dX, *dW, *dWc, *dXs, *dWs;
        float* dO;
        CK(cudaMalloc(&dX, n));
        CK(cudaMalloc(&dW, n));
        CK(cudaMalloc(&dWc, n));
        CK(cudaMalloc(&dXs, 16 * KB));
        CK(cudaMalloc(&dWs, 16 * KB));
        CK(cudaMalloc(&dO, 4 * 640));
        int d12 = 0, d13 = 0, d24 = 0, d14 = 0, tot = 0;
        double maxrel = 0, en_new = 0, en_old = 0, mx_new = 0, mx_old = 0; int nerr = 0;
        for (int rep = 0; rep < 32; ++rep) {
            for (size_t i = 0; i < n; ++i) {
                X[i] = rnd_e4m3();
                uint8_t nb = rnd() & 15;
                W[i] = F4E4[nb];
                Wc[i] = (uint8_t)(nb << 2);
            }
            // realistic scale codes (activations near 2^-6..2^2, weights near 2^-10..2^-4); rep 0..3 wide
            for (int i = 0; i < 16 * KB; ++i) {
                Xs[i] = rep < 4 ? (uint8_t)(60 + rnd() % 130) : (uint8_t)(121 + rnd() % 8);
                Ws[i] = rep < 4 ? (uint8_t)(60 + rnd() % 130) : (uint8_t)(117 + rnd() % 7);
            }
            CK(cudaMemcpy(dX, X, n, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dW, W, n, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWc, Wc, n, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dXs, Xs, 16 * KB, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dWs, Ws, 16 * KB, cudaMemcpyHostToDevice));
            k_chain<<<1, 32>>>(dX, dXs, dW, dWc, dWs, K, dO, dO + 128, dO + 256, dO + 384, dO + 512);
            CK(cudaGetLastError());
            float o[640];
            CK(cudaMemcpy(o, dO, sizeof o, cudaMemcpyDeviceToHost));
            uint32_t ob[640];
            memcpy(ob, o, sizeof o);
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c) {
                    // out1[r][c]: act r x weight c; out2[c][r]: weight c x act r (exchanged)
                    uint32_t v1 = ob[r * 8 + c], v2 = ob[128 + c * 8 + r], v3 = ob[256 + r * 8 + c],
                             v4 = ob[384 + c * 8 + r];
                    d12 += v1 != v2;
                    d13 += v1 != v3;
                    d24 += v2 != v4;
                    d14 += v1 != v4;
                    ++tot;
                    if (rep >= 4) {
                        double s = 0, sab = 0;
                        for (int kb = 0; kb < KB; ++kb) {
                            double p = 0, pa = 0;
                            for (int k = 0; k < 32; ++k) {
                                double q = e4m3(X[(size_t)r * K + kb * 32 + k]) * e4m3(W[(size_t)c * K + kb * 32 + k]);
                                p += q; pa += fabs(q);
                            }
                            double sc = ue8(Xs[r * KB + kb]) * ue8(Ws[c * KB + kb]);
                            s += p * sc; sab += pa * sc;
                        }
                        double rel = fabs(o[r * 8 + c] - s) / fmax(fabs(s), 1e-30);
                        maxrel = fmax(maxrel, rel);
                        // error against the sum of |terms| (independent of cancellation)
                        double en = fabs(o[r * 8 + c] - s) / sab, eo = fabs(o[512 + r * 8 + c] - s) / sab;
                        en_new += en; en_old += eo; mx_new = fmax(mx_new, en); mx_old = fmax(mx_old, eo); ++nerr;
                    }
                }
        }
        printf("chain K=%d: plain vs exchanged %d/%d differ %s\n", K, d12, tot, d12 ? "FAIL" : "OK");
        printf("chain K=%d: e4m3 weights vs e2m1 containers (plain roles) %d/%d differ %s\n", K, d13, tot,
               d13 ? "(INFO)" : "(identical)");
        printf("chain K=%d: e4m3 vs e2m1 (exchanged roles) %d/%d differ %s\n", K, d24, tot, d24 ? "(INFO)" : "(identical)");
        printf("chain K=%d: plain e4m3 vs exchanged e2m1 %d/%d differ\n", K, d14, tot);
        printf("chain vs exact (double) sum, realistic scales: max rel err %.3e (INFO: cancellation-sensitive)\n", maxrel);
        printf("error / sum|terms| over %d outputs, K=%d: chain mean %.3e max %.3e | old program (block from zero, fp32 promotion) mean %.3e max %.3e (INFO)\n",
               nerr, K, en_new / nerr, mx_new, en_old / nerr, mx_old);
        if (d12) ++fails;
    }
    printf(fails ? "PROBE FAIL (%d)\n" : "PROBE PASS\n", fails);
    return fails ? 1 : 0;
}
