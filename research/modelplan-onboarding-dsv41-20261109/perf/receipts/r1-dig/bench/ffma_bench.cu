// Microbench of memra's fp32 FMA family on narrow outputs (the router's gate, the compressor's pair), K = 5120:
//   kpc  - v41_gemv_ffma_narrow_kernel / _parts (one fma chain an output over K, whole K landed first)
//   r1   - v41_gemv_ffma_seg_kernel (the K-segment order: 8 chains an output side by side, then the tree)
//   r2   - the one-chain program (kpc's bytes) with K streamed through a cp.async ring and the chain's
//          shared-memory reads one step ahead; a block = one warp of 4 columns x up to 8 rows; the compressor's
//          two outputs (wkv, wgate) in one launch
// Checks: r2 bytes == kpc bytes for every shape; r1 against kpc reported. Timing: CUDA graphs of R launches
// rotating over L weight copies (bigger than L2, as a layer's weights arrive from memory), median of replays.
// usage: ffma_bench [reps]
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

typedef uint16_t v41_bf;
#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

__device__ __forceinline__ float v41_bf2f(v41_bf u) { return __uint_as_float(((uint32_t)u) << 16); }
__device__ __forceinline__ void v41_cp16(void* smem, const void* gmem, bool pred) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    int n = pred ? 16 : 0;  // 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void v41_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void v41_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
#define V41_KSEG 8
__device__ __forceinline__ float v41_ktree(float s0, float s1, float s2, float s3, float s4, float s5, float s6,
                                           float s7) {
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}
#define V41_SMF_NC 4

// ---- kpc (verbatim from 761548cde9) ----
#include "kpc_narrow.inc"
// ---- r1 (verbatim from 5a4e78bb32) ----
#include "r1_seg.inc"

// ---- r2 ----
#define R2_NS 4
struct R2Out {
    const v41_bf* B;
    long ldb;
    float* C;
    long ldc;
    int N;
    int nblk;
};
template <int NS, int R2_KC, bool WORD = false>
__global__ void __launch_bounds__(32) r2_chain_kernel(const v41_bf* __restrict__ A, long lda, R2Out o0, R2Out o1,
                                                      int M, int K) {
    constexpr int R2_PITCH = R2_KC + 8;
    extern __shared__ __align__(16) uint8_t r2_sm[];
    v41_bf* sm = (v41_bf*)r2_sm;
    int b = blockIdx.x;
    const bool second = b >= o0.nblk;
    const v41_bf* __restrict__ B = second ? o1.B : o0.B;
    const long ldb = second ? o1.ldb : o0.ldb;
    float* __restrict__ C = second ? o1.C : o0.C;
    const long ldc = second ? o1.ldc : o0.ldc;
    const int N = second ? o1.N : o0.N;
    if (second) b -= o0.nblk;
    const int n0 = b * 4, rows = 4 + M, lane = threadIdx.x, nk = K / R2_KC;
    // a stage: rows 0..3 the 4 columns of B, then the M rows of A, R2_KC elements each; lane l lands the l-th
    // 16 bytes of every row
    auto load = [&](int t) {
        v41_bf* base = sm + (t % NS) * rows * R2_PITCH;
#pragma unroll
        for (int h = 0; h < R2_KC / 256; ++h) {
            long k0 = (long)t * R2_KC + h * 256 + lane * 8;
            int e = h * 256 + lane * 8;
#pragma unroll
            for (int row = 0; row < 4; ++row) {
                bool ok = n0 + row < N;
                v41_cp16(base + row * R2_PITCH + e, B + (long)(ok ? n0 + row : 0) * ldb + k0, ok);
            }
            for (int row = 0; row < M; ++row)
                v41_cp16(base + (4 + row) * R2_PITCH + e, A + (long)row * lda + k0, true);
        }
    };
#pragma unroll
    for (int t = 0; t < NS - 1; ++t) {
        if (t < nk) load(t);
        v41_cp_commit();
    }
    const int r = lane >> 2, cc = lane & 3;
    const bool chain = r < M && n0 + cc < N;
    float acc = 0.f;
    for (int t = 0; t < nk; ++t) {
        v41_cp_wait<NS - 2>();
        __syncwarp();  // stage t landed by every lane; every lane is done with stage t - 1's slot
        if (t + NS - 1 < nk) load(t + NS - 1);
        v41_cp_commit();
        if (chain) {
            const v41_bf* base = sm + (t % NS) * rows * R2_PITCH;
            const v41_bf* ar = base + (4 + r) * R2_PITCH;
            const v41_bf* br = base + cc * R2_PITCH;
            uint4 av = *(const uint4*)ar, bv = *(const uint4*)br;
#pragma unroll
            for (int k = 0; k < R2_KC; k += 8) {
                uint4 an = av, bn = bv;
                if (k + 8 < R2_KC) {
                    an = *(const uint4*)(ar + k + 8);
                    bn = *(const uint4*)(br + k + 8);
                }
                if (WORD) {
                    const uint32_t aw[4] = {av.x, av.y, av.z, av.w}, bw[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        acc = __fmaf_rn(__uint_as_float(aw[u] << 16), __uint_as_float(bw[u] << 16), acc);
                        acc = __fmaf_rn(__uint_as_float(aw[u] & 0xffff0000u), __uint_as_float(bw[u] & 0xffff0000u), acc);
                    }
                } else {
                    const v41_bf* ae = (const v41_bf*)&av;
                    const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
                    for (int u = 0; u < 8; ++u) acc = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), acc);
                }
                av = an;
                bv = bn;
            }
        }
    }
    if (chain) C[(long)r * ldc + n0 + cc] = acc;
}

// ---- host ----
static void kpc_launch(const v41_bf* a, const v41_bf* b, float* c, int m, int n, int k, cudaStream_t st) {
    if (m > 8) {
        for (int parts = 2; parts <= 8; parts *= 2) {
            int kp = k / parts;
            int smem = (V41_SMF_NC + m) * (kp + 8) * 2;
            if (k % parts == 0 && kp % 8 == 0 && smem <= 99 * 1024) {
                v41_gemv_ffma_narrow_parts_kernel<<<(n + 3) / 4, 128, smem, st>>>(a, k, b, k, c, n, m, n, k, kp);
                return;
            }
        }
        fprintf(stderr, "kpc: no fit\n");
        exit(1);
    }
    int kp = k;
    if ((V41_SMF_NC + m) * (kp + 8) * 2 > 99 * 1024) kp = k / 2;
    int smem = (V41_SMF_NC + m) * (kp + 8) * 2;
    if (kp == k)
        v41_gemv_ffma_narrow_kernel<<<(n + 3) / 4, 128, smem, st>>>(a, k, b, k, c, n, m, n, k);
    else
        v41_gemv_ffma_narrow_parts_kernel<<<(n + 3) / 4, 128, smem, st>>>(a, k, b, k, c, n, m, n, k, kp);
}
static void r1_launch(const v41_bf* a, const v41_bf* b, float* c, int m, int n, int k, cudaStream_t st) {
    for (int parts = 1; parts <= V41_KSEG; parts *= 2) {
        int kp = k / parts;
        long smem = v41_ffma_seg_smem(m, k, kp);
        if (smem <= 99 * 1024) {
            v41_gemv_ffma_seg_kernel<<<(n + 3) / 4, 256, smem, st>>>(a, k, b, k, c, n, m, n, k, kp);
            return;
        }
    }
    fprintf(stderr, "r1: no fit\n");
    exit(1);
}
template <int NS, int KC = 256, bool WORD = false>
static void r2_launch(const v41_bf* a, const v41_bf* b0, float* c0, int n0, const v41_bf* b1, float* c1, int n1, int m,
                      int k, cudaStream_t st) {
    R2Out o0{b0, k, c0, n0, n0, (n0 + 3) / 4}, o1{b1, k, c1, n1, n1, b1 ? (n1 + 3) / 4 : 0};
    int smem = NS * (4 + m) * (KC + 8) * 2;
    r2_chain_kernel<NS, KC, WORD><<<o0.nblk + o1.nblk, 32, smem, st>>>(a, k, o0, o1, m, k);
}

// r2w: the same chain, two warps a block: warp 1 lands K through a cp.async ring of bf16 stages and widens each stage
// to fp32 in one of two buffers; warp 0 runs the chains on the fp32 buffer (4 shared loads + 8 fmas an 8-element step,
// no conversion on the chain's warp). Named barriers: FULL(b) = 1 + b, EMPTY(b) = 3 + b.
#define W_KC 256
#define W_NSR 4
#define W_RP (W_KC + 8)   // bf16 elements a raw row
#define W_FP (W_KC + 4)   // floats an fp32 row
__device__ __forceinline__ void w_bar_sync(int id) { asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory"); }
__device__ __forceinline__ void w_bar_arrive(int id) { asm volatile("bar.arrive %0, 64;\n" ::"r"(id) : "memory"); }
__global__ void __launch_bounds__(64) r2w_chain_kernel(const v41_bf* __restrict__ A, long lda, R2Out o0, R2Out o1,
                                                       int M, int K) {
    extern __shared__ __align__(16) uint8_t w_sm[];
    const int mrows = 4 + min(8, M);                              // the smem layout: the largest row group
    v41_bf* raw = (v41_bf*)w_sm;                                  // [W_NSR][mrows][W_RP]
    float* f32 = (float*)(w_sm + W_NSR * mrows * W_RP * 2);       // [2][mrows][W_FP]
    int b = blockIdx.x;
    const bool second = b >= o0.nblk;
    const v41_bf* __restrict__ B = second ? o1.B : o0.B;
    const long ldb = second ? o1.ldb : o0.ldb;
    float* __restrict__ C = second ? o1.C : o0.C;
    const long ldc = second ? o1.ldc : o0.ldc;
    const int N = second ? o1.N : o0.N;
    if (second) b -= o0.nblk;
    // a block takes 8 of the call's rows (blockIdx.y), the chains of every row group running side by side
    {
        const int r0 = blockIdx.y * 8;
        A += (long)r0 * lda;
        C += (long)r0 * ldc;
        M = min(8, M - r0);
    }
    const int rows = 4 + M;
    const int n0 = b * 4, nk = K / W_KC, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp == 1) {
        auto land = [&](int t) {
            v41_bf* base = raw + (t % W_NSR) * rows * W_RP;
            long k0 = (long)t * W_KC + lane * 8;
#pragma unroll
            for (int row = 0; row < 4; ++row) {
                bool ok = n0 + row < N;
                v41_cp16(base + row * W_RP + lane * 8, B + (long)(ok ? n0 + row : 0) * ldb + k0, ok);
            }
            for (int row = 0; row < M; ++row)
                v41_cp16(base + (4 + row) * W_RP + lane * 8, A + (long)row * lda + k0, true);
        };
#pragma unroll
        for (int t = 0; t < W_NSR - 1; ++t) {
            if (t < nk) land(t);
            v41_cp_commit();
        }
        for (int t = 0; t < nk; ++t) {
            v41_cp_wait<W_NSR - 2>();
            __syncwarp();
            if (t + W_NSR - 1 < nk) land(t + W_NSR - 1);
            v41_cp_commit();
            if (t >= 2) w_bar_sync(3 + (t & 1));  // the chains are done with buffer t % 2 (stage t - 2)
            const v41_bf* src = raw + (t % W_NSR) * rows * W_RP + lane * 8;
            float* dst = f32 + (t & 1) * rows * W_FP + lane * 8;
            for (int row = 0; row < rows; ++row) {
                uint4 v = *(const uint4*)(src + row * W_RP);
                float4 lo = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                        __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
                float4 hi = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                        __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
                *(float4*)(dst + row * W_FP) = lo;
                *(float4*)(dst + row * W_FP + 4) = hi;
            }
            w_bar_arrive(1 + (t & 1));  // buffer t % 2 holds stage t
        }
    } else {
        const int r = lane >> 2, cc = lane & 3;
        const bool chain = r < M && n0 + cc < N;
        float acc = 0.f;
        for (int t = 0; t < nk; ++t) {
            w_bar_sync(1 + (t & 1));
            if (chain) {
                const float* base = f32 + (t & 1) * rows * W_FP;
                const float* ar = base + (4 + r) * W_FP;
                const float* br = base + cc * W_FP;
                float4 a0 = *(const float4*)ar, a1 = *(const float4*)(ar + 4);
                float4 b0 = *(const float4*)br, b1 = *(const float4*)(br + 4);
#pragma unroll
                for (int k = 0; k < W_KC; k += 8) {
                    float4 na0 = a0, na1 = a1, nb0 = b0, nb1 = b1;
                    if (k + 8 < W_KC) {
                        na0 = *(const float4*)(ar + k + 8);
                        na1 = *(const float4*)(ar + k + 12);
                        nb0 = *(const float4*)(br + k + 8);
                        nb1 = *(const float4*)(br + k + 12);
                    }
                    acc = __fmaf_rn(a0.x, b0.x, acc);
                    acc = __fmaf_rn(a0.y, b0.y, acc);
                    acc = __fmaf_rn(a0.z, b0.z, acc);
                    acc = __fmaf_rn(a0.w, b0.w, acc);
                    acc = __fmaf_rn(a1.x, b1.x, acc);
                    acc = __fmaf_rn(a1.y, b1.y, acc);
                    acc = __fmaf_rn(a1.z, b1.z, acc);
                    acc = __fmaf_rn(a1.w, b1.w, acc);
                    a0 = na0; a1 = na1; b0 = nb0; b1 = nb1;
                }
            }
            if (t < nk - 2) w_bar_arrive(3 + (t & 1));  // buffer t % 2 is free again
        }
        if (chain) C[(long)r * ldc + n0 + cc] = acc;
    }
}
static void r2w_launch(const v41_bf* a, const v41_bf* b0, float* c0, int n0, const v41_bf* b1, float* c1, int n1, int m,
                       int k, cudaStream_t st) {
    R2Out o0{b0, k, c0, n0, n0, (n0 + 3) / 4}, o1{b1, k, c1, n1, n1, b1 ? (n1 + 3) / 4 : 0};
    int mg = m < 8 ? m : 8;
    int smem = W_NSR * (4 + mg) * W_RP * 2 + 2 * (4 + mg) * W_FP * 4;
    r2w_chain_kernel<<<dim3(o0.nblk + o1.nblk, (m + 7) / 8), 64, smem, st>>>(a, k, o0, o1, m, k);
}
// the floor: one dependent fma chain of K steps a thread, operands in registers
__global__ void chain_floor_kernel(float* out, int K) {
    float acc = threadIdx.x * 1e-3f;
    for (int k = 0; k < K; k += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u) acc = __fmaf_rn(acc, 1.0000001f, 1e-7f);
    }
    out[blockIdx.x * 32 + threadIdx.x] = acc;
}
__global__ void empty_kernel() {}

static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float randn(uint64_t& s) {
    // Box-Muller on a 64-bit LCG
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    double u1 = ((s >> 11) + 1.0) / 9007199254740993.0;
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    double u2 = (s >> 11) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 21;
    const int K = 5120, R = 96, NMAX = 512, MMAX = 32;
    const int L = argc > 2 ? atoi(argv[2]) : 48;
    CK(cudaFuncSetAttribute(v41_gemv_ffma_narrow_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(v41_gemv_ffma_narrow_parts_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            99 * 1024));
    CK(cudaFuncSetAttribute(v41_gemv_ffma_seg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<4, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<4, 256, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<3, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<6, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<3, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2_chain_kernel<4, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(r2w_chain_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device %s, %d SMs, L2 %d MB, %d weight copies rotated\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20, L);
    uint64_t seed = 12345;
    std::vector<uint16_t> ha((size_t)MMAX * K), hb((size_t)L * 2 * NMAX * K);
    for (auto& x : ha) x = f2bf(randn(seed));
    for (auto& x : hb) x = f2bf(0.02f * randn(seed));
    v41_bf *da, *db;
    float *dc;
    CK(cudaMalloc(&da, ha.size() * 2));
    CK(cudaMalloc(&db, hb.size() * 2));
    CK(cudaMalloc(&dc, (size_t)8 * 2 * MMAX * NMAX * 4));
    CK(cudaMemcpy(da, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto wcopy = [&](int l, int which) { return db + ((size_t)l * 2 + which) * NMAX * K; };
    float* cK = dc;
    float* cR1 = dc + 2 * MMAX * NMAX;
    float* cR2 = dc + 4 * MMAX * NMAX;
    float* cR2b = dc + 6 * MMAX * NMAX;
    // time a variant: a graph of R launches rotating over the L weight copies
    auto timeit = [&](auto&& body) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < R; ++i) body(i % L);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, st));
        std::vector<float> ts;
        for (int i = 0; i < reps; ++i) {
            CK(cudaEventRecord(e0, st));
            CK(cudaGraphLaunch(ge, st));
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            ts.push_back(ms * 1000.f / R);
        }
        std::sort(ts.begin(), ts.end());
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(g));
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
        return std::make_pair(ts[ts.size() / 2], std::make_pair(ts.front(), ts.back()));
    };
    auto pr = [](const char* name, std::pair<float, std::pair<float, float>> t, float gap) {
        printf("  %-26s %8.2f us [%7.2f, %7.2f]  net %8.2f\n", name, t.first, t.second.first, t.second.second,
               t.first - gap);
    };
    auto gap = timeit([&](int) { empty_kernel<<<1, 32, 0, st>>>(); });
    printf("empty-kernel graph step: %.2f us [%.2f, %.2f]\n", gap.first, gap.second.first, gap.second.second);
    pr("floor: 5120 dependent fma", timeit([&](int) { chain_floor_kernel<<<96, 32, 0, st>>>(dc, K); }), gap.first);
    int fails = 0;
    for (int m : {1, 2, 4, 8, 16, 24, 32}) {
        for (int n : {384, 512}) {
            // bytes: one copy, both outputs
            CK(cudaMemset(dc, 0xff, (size_t)8 * 2 * MMAX * NMAX * 4));
            kpc_launch(da, wcopy(0, 0), cK, m, n, K, st);
            r1_launch(da, wcopy(0, 0), cR1, m, n, K, st);
            if (m <= 8) r2_launch<R2_NS>(da, wcopy(0, 0), cR2, n, nullptr, nullptr, 0, m, K, st);
            else r2w_launch(da, wcopy(0, 0), cR2, n, nullptr, nullptr, 0, m, K, st);
            // the pair: second output on copy 0's other matrix
            kpc_launch(da, wcopy(0, 1), cK + MMAX * NMAX, m, n, K, st);
            if (m <= 8) r2_launch<R2_NS>(da, wcopy(0, 0), cR2b, n, wcopy(0, 1), cR2b + MMAX * NMAX, n, m, K, st);
            else r2w_launch(da, wcopy(0, 0), cR2b, n, wcopy(0, 1), cR2b + MMAX * NMAX, n, m, K, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            std::vector<float> hk(2 * MMAX * NMAX), h1(MMAX * NMAX), h2(MMAX * NMAX), h2b(2 * MMAX * NMAX);
            CK(cudaMemcpy(hk.data(), cK, hk.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h1.data(), cR1, h1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(), cR2, h2.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2b.data(), cR2b, h2b.size() * 4, cudaMemcpyDeviceToHost));
            int cnt = m * n, d2 = 0, d2b = 0, d1 = 0;
            double maxrel1 = 0;
            for (int i = 0; i < cnt; ++i) {
                int r = i / n, c = i % n;
                long o = (long)r * n + c;
                if (memcmp(&hk[o], &h2[o], 4)) ++d2;
                if (memcmp(&hk[o], &h2b[o], 4) || memcmp(&hk[MMAX * NMAX + o], &h2b[MMAX * NMAX + o], 4)) ++d2b;
                if (memcmp(&hk[o], &h1[o], 4)) {
                    ++d1;
                    maxrel1 = std::max(maxrel1, (double)fabsf(hk[o] - h1[o]) / (fabsf(hk[o]) + 1e-30));
                }
            }
            if (d2 || d2b) ++fails;
            printf("M=%d N=%d K=%d: r2 vs kpc %d of %d differ, r2 pair vs kpc %d, r1 vs kpc %d differ (max rel %.2e)\n",
                   m, n, K, d2, cnt, d2b, d1, maxrel1);
            pr("kpc", timeit([&](int l) { kpc_launch(da, wcopy(l, 0), cK, m, n, K, st); }), gap.first);
            pr("r1 (seg)", timeit([&](int l) { r1_launch(da, wcopy(l, 0), cR1, m, n, K, st); }), gap.first);
            if (m <= 8)
                pr("r2 (stream chain)",
                   timeit([&](int l) { r2_launch<R2_NS>(da, wcopy(l, 0), cR2, n, nullptr, nullptr, 0, m, K, st); }),
                   gap.first);
            if (m <= 8) {
                pr("r2x (r2, word widening)",
                   timeit([&](int l) { r2_launch<4, 256, true>(da, wcopy(l, 0), cR2, n, nullptr, nullptr, 0, m, K, st); }),
                   gap.first);
                CK(cudaMemset(cR2, 0xff, 2 * MMAX * NMAX * 4));
                r2_launch<4, 256, true>(da, wcopy(0, 0), cR2, n, wcopy(0, 1), cR2 + MMAX * NMAX, n, m, K, st);
                CK(cudaStreamSynchronize(st));
                CK(cudaGetLastError());
                std::vector<float> hx(2 * MMAX * NMAX);
                CK(cudaMemcpy(hx.data(), cR2, hx.size() * 4, cudaMemcpyDeviceToHost));
                int dx = 0;
                for (int i = 0; i < m * n; ++i)
                    dx += (memcmp(&hx[i], &hk[i], 4) != 0) + (memcmp(&hx[MMAX * NMAX + i], &hk[MMAX * NMAX + i], 4) != 0);
                printf("  r2x (pair) vs kpc: %d of %d differ\n", dx, 2 * m * n);
                if (dx) ++fails;
            }
            pr("r2w (2 warps, fp32 stage)",
               timeit([&](int l) { r2w_launch(da, wcopy(l, 0), cR2, n, nullptr, nullptr, 0, m, K, st); }), gap.first);
            {
                CK(cudaMemset(cR2, 0xff, 2 * MMAX * NMAX * 4));
                r2w_launch(da, wcopy(0, 0), cR2, n, wcopy(0, 1), cR2 + MMAX * NMAX, n, m, K, st);
                CK(cudaStreamSynchronize(st));
                CK(cudaGetLastError());
                std::vector<float> hv(2 * MMAX * NMAX);
                CK(cudaMemcpy(hv.data(), cR2, hv.size() * 4, cudaMemcpyDeviceToHost));
                int dv = 0;
                for (int i = 0; i < m * n; ++i)
                    dv += (memcmp(&hv[i], &hk[i], 4) != 0) + (memcmp(&hv[MMAX * NMAX + i], &hk[MMAX * NMAX + i], 4) != 0);
                printf("  r2w (pair) vs kpc: %d of %d differ\n", dv, 2 * m * n);
                if (dv) ++fails;
            }
            if (n == 512) {
                pr("kpc pair (2 launches)", timeit([&](int l) {
                       kpc_launch(da, wcopy(l, 0), cK, m, n, K, st);
                       kpc_launch(da, wcopy(l, 1), cK + MMAX * NMAX, m, n, K, st);
                   }),
                   2 * gap.first);
                pr("r1 pair (2 launches)", timeit([&](int l) {
                       r1_launch(da, wcopy(l, 0), cR1, m, n, K, st);
                       r1_launch(da, wcopy(l, 1), cR1 + MMAX * NMAX, m, n, K, st);
                   }),
                   2 * gap.first);
                if (m <= 8)
                    pr("r2 pair (1 launch)", timeit([&](int l) {
                           r2_launch<R2_NS>(da, wcopy(l, 0), cR2b, n, wcopy(l, 1), cR2b + MMAX * NMAX, n, m, K, st);
                       }),
                       gap.first);
                pr("r2w pair (1 launch)", timeit([&](int l) {
                       r2w_launch(da, wcopy(l, 0), cR2b, n, wcopy(l, 1), cR2b + MMAX * NMAX, n, m, K, st);
                   }),
                   gap.first);
            }
        }
    }
    printf(fails ? "BYTES-FAIL %d shapes\n" : "BYTES-OK: r2 == kpc on every shape\n", fails);
    return fails ? 1 : 0;
}
