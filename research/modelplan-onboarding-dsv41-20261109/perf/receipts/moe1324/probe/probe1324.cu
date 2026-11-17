// memra#1324 probe: the routed-expert GEMMs of a four-rank split MoE share (w1 + w3: n 576, k 5120, two
// projections; w2: n 1280, k 2304) at 1 to 16 rows on one card, on the device plan's tile list.
//   main build (default): main's memra_v41_moe_small against launch-shape variants of the same body (ring depth,
//     K blocks a stage, warp columns, warps a block), every variant's output compared byte for byte with main's on
//     every routing, then timed in rounds (forward, reverse order alternating);
//   -DBASE: the pre-CUTLASS build's swiglu_tiles (w1 + w3 + SwiGLU) and gemm_f8_tiles (w2) on the same shapes.
// Weights: 384 experts (1.7 GB), 64 routings of distinct experts a token; a graph of 64 calls rotates the routings,
// so every call reads its experts from DRAM. One line a cell: median graph us a call over the rounds.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

extern "C" {
int memra_v41_moe_plan(const int* sel, const float* selw, int rows, int topk, int ne, int bm, int e0, int e1,
                       int* counts, int* off, int* ptok, float* pw, int* order, void* tiles, int* ntiles, int* err,
                       int* flag, void* stream);
#ifdef BASE
int memra_v41_swiglu_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles, const int* ntiles,
                           int max_tiles, const void* ew, long ldb, long ldbs, int wfmt, const float* wrow,
                           float limit, void* h, long ldh, int n, int k, int bm, void* stream);
int memra_v41_gemm_f8_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles, const int* ntiles,
                            int max_tiles, const void* ew, int proj, long ldb, long ldbs, int wfmt, void* c, long ldc,
                            int out_f32, int n, int k, int bm, void* stream);
#else
int memra_v41_moe_small(const void* a, long lda, const void* asf, const int* sfrow, const void* tiles,
                        const int* ntiles, int max_tiles, const void* ew, int nproj, int p0, int p1, long ldb,
                        void* d0, void* d1, long ldc, int n, int k, int bm, void* stream);
int memra_v41_moe_small_fits(int max_tiles, int n, int nproj, int bm);
#ifdef HEAD
void memra_v41_moe_small_ring(int ring);
#endif
int memra_v41_swiglu(const void* gate, const void* up, long ld, const float* wrow, float limit, void* out, long ldo,
                     int rows, int inter, void* stream);
#endif
}

#define CK(x)                                                                                          \
    do {                                                                                               \
        cudaError_t e_ = (x);                                                                          \
        if (e_ != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                                   \
        }                                                                                              \
    } while (0)
#define RC(x)                                                                \
    do {                                                                     \
        int r_ = (x);                                                        \
        if (r_) {                                                            \
            fprintf(stderr, "rc %d from %s at %s:%d\n", r_, #x, __FILE__, __LINE__); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static cudaStream_t g_st;
static int g_sms = 0, g_optin = 0;

__device__ __forceinline__ uint32_t hash32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}
// kind 0: e4m3 codes without NaN; 1: any byte (e2m1 pairs); 2: ue8m0 scale in [lo, lo + 12]
__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int kind, int lo) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t h = hash32(i * 0x9E3779B97F4A7C15ULL ^ seed);
        uint8_t v;
        if (kind == 0) v = (uint8_t)(h & 0xF7);
        else if (kind == 1) v = (uint8_t)h;
        else v = (uint8_t)(lo + (h % 13));
        p[i] = v;
    }
}
static void* dalloc(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, n < 256 ? 256 : n));
    return p;
}
static void fill(void* p, size_t n, int kind, uint32_t seed, int lo = 118) {
    fill_kernel<<<1024, 256, 0, g_st>>>((uint8_t*)p, n, seed, kind, lo);
    CK(cudaGetLastError());
}
static long al256(long x) { return (x + 255) & ~255L; }
// CUTLASS's scale layout bytes (memra_v41_sf_bytes)
static long sf_bytes(long rows, long k) { return (rows + 127) / 128 * 128 * (((k / 32) + 3) / 4) * 4; }

struct Tile {
    int row0, rows, m0, expert;
};
struct ExpW {
    const uint8_t* w[3];
    const uint8_t* s[3];
    float scale2[3];
    int pad;
};

#ifndef BASE
// ------------------------------------------------------------------------------------------------
// The variant body: v41_moe_small_kernel with its launch shape as template parameters. CW weight rows a warp, PK K
// blocks a stage (a multiple of 4; the last stage may be shorter), PS stages a warp's ring, NW warps a block. One
// output element is the same chain as main's: the block-scaled MMA on the e2m1 container over the K blocks in
// ascending order, one fp32 accumulator, bf16 out.
namespace px {
__device__ __forceinline__ void cp16(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void cp4(void* smem, const void* g) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ uint32_t cont4(uint32_t nib16) {
    return ((nib16 & 0xFu) << 2) | (((nib16 >> 4) & 0xFu) << 10) | (((nib16 >> 8) & 0xFu) << 18) |
           (((nib16 >> 12) & 0xFu) << 26);
}
template <int BA, int BB>
__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                    uint32_t sb) {
    asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0), "n"(BB),
          "n"(0));
}
__device__ __forceinline__ void mma_u(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                      uint32_t sb, int u) {
    switch (u) {
        case 0: mma<0, 0>(c, a, b0, b1, sa, sb); break;
        case 1: mma<1, 1>(c, a, b0, b1, sa, sb); break;
        case 2: mma<2, 2>(c, a, b0, b1, sa, sb); break;
        default: mma<3, 3>(c, a, b0, b1, sa, sb); break;
    }
}
__device__ __forceinline__ long sf_off(long r, long kb, long kbn) {
    return (r / 128) * (((kbn + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

// ACT 0: each warp stages the tile's activation rows in its own ring (main's layout); ACT 1: the block stages them
// once a stage for all its warps (a block barrier a stage).
template <int CW, int PK, int PS, int NW, int ACT>
struct Sx {
    static constexpr int WR = PK * 16 + 16;
    static constexpr int AR = PK * 32 + 16;
    static constexpr int W_B = CW * WR;
    static constexpr int WS_B = (CW * PK + 15) & ~15;
    static constexpr int WST = W_B + WS_B;  // a warp's weights of a stage
    __host__ __device__ static constexpr int ast(int ra) { return ra * AR + ((ra * PK + 15) & ~15); }
    // ACT 0: a warp's stage; ACT 1: the block's stage
    __host__ __device__ static constexpr int stage(int ra) { return ACT ? NW * WST + ast(ra) : WST + ast(ra); }
    __host__ __device__ static constexpr int block(int ra) { return ACT ? PS * stage(ra) : NW * PS * stage(ra); }
};

template <int CW, int PK, int PS, int NW, int ACT>
__global__ void __launch_bounds__(NW * 32) xk(const uint8_t* __restrict__ A, long lda, const uint8_t* __restrict__ asf,
                                              const int* __restrict__ sfrow, const Tile* __restrict__ tiles,
                                              const int* __restrict__ ntiles, const ExpW* __restrict__ ew, int p0,
                                              int p1, long ldb, uint16_t* __restrict__ d0, uint16_t* __restrict__ d1,
                                              long ldc, int N, int K, int BM) {
    using P = Sx<CW, PK, PS, NW, ACT>;
    constexpr int PW = PK / 4;  // scale words a row a stage
    extern __shared__ __align__(16) uint8_t xsm[];
    const Tile tl = tiles[blockIdx.y];
    if ((int)blockIdx.y >= *ntiles) return;
    const int proj = blockIdx.z ? p1 : p0;
    uint16_t* C = blockIdx.z ? d1 : d0;
    const ExpW& w = ew[tl.expert];
    const uint8_t* __restrict__ W = w.w[proj];
    const uint8_t* __restrict__ WS = w.s[proj];
    const long t0 = (long)tl.row0 + tl.m0;
    const int M = min(tl.rows - tl.m0, BM);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rw0 = (blockIdx.x * NW + warp) * CW;
    const int nw = max(0, min(CW, N - rw0));
    if (!ACT && nw <= 0) return;
    const int KB = K / 32, NS = (KB + PK - 1) / PK;
    const int STAGE = P::stage(BM);
    // a stage's parts: this warp's weight rows and their scale words, the activation rows and their scale words
    auto wb = [&](int i) -> uint8_t* {
        return ACT ? xsm + (i % PS) * STAGE + warp * P::WST : xsm + warp * PS * STAGE + (i % PS) * STAGE;
    };
    auto ab = [&](int i) -> uint8_t* {
        return ACT ? xsm + (i % PS) * STAGE + NW * P::WST : xsm + warp * PS * STAGE + (i % PS) * STAGE + P::WST;
    };
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    auto issue = [&](int i) {
        uint8_t* sw = wb(i);
        uint8_t* sa = ab(i);
        const int kb0 = i * PK, pk = min(PK, KB - kb0), pw = pk / 4;
        for (int c = lane; c < nw * PK; c += 32) {
            int r = c / PK, q = c % PK;
            if (q < pk) cp16(sw + r * P::WR + q * 16, W + (long)(rw0 + r) * ldb + kb0 * 16 + q * 16);
        }
        for (int c = lane; c < nw * PW; c += 32) {
            int r = c / PW, j = c % PW;
            if (j < pw) cp4(sw + P::W_B + c * 4, WS + sf_off(rw0 + r, kb0 + 4 * j, KB));
        }
        const int at = ACT ? threadIdx.x : lane, an = ACT ? NW * 32 : 32;
        for (int c = at; c < M * 2 * PK; c += an) {
            int r = c / (2 * PK), q = c % (2 * PK);
            if (q < 2 * pk) cp16(sa + r * P::AR + q * 16, A + (t0 + r) * lda + kb0 * 32 + q * 16);
        }
        for (int c = at; c < M * PW; c += an) {
            int r = c / PW, j = c % PW;
            int sr = sfrow ? sfrow[t0 + r] : (int)(t0 + r);
            if (j < pw) cp4(sa + BM * P::AR + c * 4, asf + sf_off(sr, kb0 + 4 * j, KB));
        }
    };
    for (int s = 0; s < PS - 1; ++s) {
        if (s < NS) issue(s);
        commit();
    }
    const bool w0 = g < nw, w1 = g + 8 < nw;
    const int wsr = g + 8 * (t & 1);
    const bool wsok = wsr < nw;
    for (int i = 0; i < NS; ++i) {
        wait<PS - 2>();
        if (ACT) __syncthreads();
        else __syncwarp();
        if (i + PS - 1 < NS) issue(i + PS - 1);
        commit();
        if (ACT && nw == 0) continue;
        const uint8_t* st = wb(i);
        const uint8_t* sa = ab(i);
        const int pw = min(PK, KB - i * PK) / 4;
#pragma unroll
        for (int j = 0; j < PW; ++j) {
            if (j < pw) {
                const uint32_t wsc = wsok ? *(const uint32_t*)(st + P::W_B + (wsr * PW + j) * 4) : 0u;
                const uint32_t asc = g < M ? *(const uint32_t*)(sa + BM * P::AR + (g * PW + j) * 4) : 0u;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kk = 4 * j + u;
                    uint32_t af[4] = {0u, 0u, 0u, 0u};
                    if (w0) {
                        af[0] = cont4(*(const uint16_t*)(st + g * P::WR + kk * 16 + 2 * t));
                        af[2] = cont4(*(const uint16_t*)(st + g * P::WR + kk * 16 + 8 + 2 * t));
                    }
                    if (w1) {
                        af[1] = cont4(*(const uint16_t*)(st + (g + 8) * P::WR + kk * 16 + 2 * t));
                        af[3] = cont4(*(const uint16_t*)(st + (g + 8) * P::WR + kk * 16 + 8 + 2 * t));
                    }
                    uint32_t b0 = 0u, b1 = 0u;
                    if (g < M) {
                        const uint8_t* ar = sa + g * P::AR + kk * 32 + 4 * t;
                        b0 = *(const uint32_t*)ar;
                        b1 = *(const uint32_t*)(ar + 16);
                    }
                    mma_u(acc, af, b0, b1, wsc, asc, u);
                }
            }
        }
    }
    wait<0>();
    if (nw == 0) return;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        int wr = g + (e >= 2 ? 8 : 0), row = 2 * t + (e & 1), col = rw0 + wr;
        if (wr < CW && row < M && col < N) C[(t0 + row) * ldc + col] = f2bf(acc[e]);
    }
}

struct Args {
    const void *a, *asf;
    const int* sfrow;
    const void* tiles;
    const int* ntiles;
    int max_tiles;
    const void* ew;
    int nproj, p0, p1;
    long ldb;
    void *d0, *d1;
    long ldc;
    int n, k, bm;
};
// 0 ok, 1 the shape does not fit this variant (shared memory), else a CUDA error
template <int CW, int PK, int PS, int NW, int ACT>
int launch(const Args& x) {
    using P = Sx<CW, PK, PS, NW, ACT>;
    int sm = P::block(x.bm);
    if (sm > g_optin) return 1;
    static int set = 0;
    if (!set) {
        CK(cudaFuncSetAttribute(xk<CW, PK, PS, NW, ACT>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_optin));
        set = 1;
    }
    dim3 gr((x.n + NW * CW - 1) / (NW * CW), x.max_tiles, x.nproj);
    xk<CW, PK, PS, NW, ACT><<<gr, NW * 32, sm, g_st>>>((const uint8_t*)x.a, x.k, (const uint8_t*)x.asf, x.sfrow,
                                                       (const Tile*)x.tiles, x.ntiles, (const ExpW*)x.ew, x.p0, x.p1,
                                                       x.ldb, (uint16_t*)x.d0, (uint16_t*)x.d1, x.ldc, x.n, x.k, x.bm);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : 100 + (int)e;
}
template <int CW, int PK, int PS, int NW, int ACT>
int occ(int bm) {  // resident blocks an SM, then registers a thread * 1000
    int nb = 0;
    using P = Sx<CW, PK, PS, NW, ACT>;
    if (P::block(bm) > g_optin) return 0;
    CK(cudaFuncSetAttribute(xk<CW, PK, PS, NW, ACT>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_optin));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, xk<CW, PK, PS, NW, ACT>, NW * 32, P::block(bm)));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, xk<CW, PK, PS, NW, ACT>));
    return nb + 1000 * fa.numRegs;
}
struct Var {
    std::string name;
    int (*run)(const Args&);
    int (*occ)(int);
};
#define PXV(CW, PK, PS, NW, ACT)                                                                      \
    Var{"cw" #CW "_pk" #PK "_ps" #PS "_nw" #NW "_a" #ACT, launch<CW, PK, PS, NW, ACT>, occ<CW, PK, PS, NW, ACT>}
#ifdef HEAD
template <int RING>
int eng(const Args& x) {
    memra_v41_moe_small_ring(RING);
    int rc = memra_v41_moe_small(x.a, x.k, x.asf, x.sfrow, x.tiles, x.ntiles, x.max_tiles, x.ew, x.nproj, x.p0, x.p1,
                                 x.ldb, x.d0, x.d1, x.ldc, x.n, x.k, x.bm, g_st);
    memra_v41_moe_small_ring(1);
    return rc ? 100 + rc : 0;
}
int occ_none(int) { return 0; }
#endif
static std::vector<Var> variants() {
    return {
#ifdef HEAD
        // the engine's launcher: ring 2 (the block ring) and ring 0 (by the card)
        Var{"eng_ring2", eng<2>, occ_none}, Var{"eng_ring0", eng<0>, occ_none},
#endif
        // main's shapes (cw by main's rule: 16 at 4 rows and wider, 4 at one row)
        PXV(16, 4, 6, 4, 0), PXV(4, 4, 6, 4, 0),
        // block-shared activations, two stages: 2 KB of weights a warp a stage
        PXV(16, 8, 2, 4, 1), PXV(8, 16, 2, 4, 1), PXV(4, 32, 2, 4, 1),
        // neighbours: 1 KB and 4 KB a warp a stage, 2 and 8 warps, three stages
        PXV(8, 8, 2, 4, 1), PXV(16, 16, 2, 4, 1), PXV(4, 16, 2, 4, 1),
        PXV(16, 8, 2, 2, 1), PXV(8, 16, 2, 2, 1), PXV(16, 8, 2, 8, 1), PXV(8, 16, 2, 8, 1),
        PXV(16, 8, 3, 4, 1), PXV(8, 16, 3, 4, 1),
        // per-warp activations, two stages
        PXV(16, 8, 2, 4, 0), PXV(8, 16, 2, 2, 0),
    };
}
}  // namespace px
#endif

// ------------------------------------------------------------------------------------------------
struct Shape {
    const char* name;
    int n, k, nproj, p0, p1;
};

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// us a call of f over a graph of NC calls (f(i) uses routing i % S), median of `reps` launches
static double graph_us(int NC, int reps, const std::function<void(int)>& f) {
    cudaGraph_t gr;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeRelaxed));
    for (int i = 0; i < NC; i++) f(i);
    CK(cudaStreamEndCapture(g_st, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    CK(cudaGraphLaunch(ge, g_st));
    CK(cudaStreamSynchronize(g_st));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(a, g_st));
        CK(cudaGraphLaunch(ge, g_st));
        CK(cudaEventRecord(b, g_st));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1e3 / NC);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(gr));
    return median(t);
}

int main(int argc, char** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    int rounds = argc > 2 ? atoi(argv[2]) : 6;
    std::vector<int> Rs = {1, 2, 4, 8, 9, 10, 12, 16};
    if (argc > 3) {
        Rs.clear();
        for (char* p = strtok(argv[3], ","); p; p = strtok(nullptr, ",")) Rs.push_back(atoi(p));
    }
    const char* only = argc > 4 ? argv[4] : nullptr;  // comma list of variant names to keep (main build)
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    g_sms = p.multiProcessorCount;
    CK(cudaDeviceGetAttribute(&g_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
#ifdef BASE
    const char* build = "base";
#else
    const char* build = "main";
#endif
    printf("# build %s device %d %s SMs %d L2 %d MB smem optin %d\n", build, dev, p.name, g_sms, p.l2CacheSize >> 20,
           g_optin);
    const int ne = 384, topk = 6, d = 5120, inter = 2304, hn = 576, dn = 1280;
    // experts: w1, w3 [hn][d / 2], w2 [dn][inter / 2]; scales in each build's layout
#ifdef BASE
    long s13b = (long)hn * (d / 32), s2b = (long)dn * (inter / 32);
#else
    long s13b = sf_bytes(hn, d), s2b = sf_bytes(dn, inter);
#endif
    long w13b = (long)hn * d / 2, w2b = (long)dn * inter / 2;
    long per = al256(w13b) * 2 + al256(w2b) + al256(s13b) * 2 + al256(s2b);
    uint8_t* pool = (uint8_t*)dalloc((size_t)per * ne);
    std::vector<ExpW> tab(ne);
    for (int e = 0; e < ne; e++) {
        uint8_t* q = pool + (size_t)e * per;
        uint8_t *w1 = q, *w3 = w1 + al256(w13b), *w2 = w3 + al256(w13b);
        uint8_t *s1 = w2 + al256(w2b), *s3 = s1 + al256(s13b), *s2 = s3 + al256(s13b);
        fill(w1, w13b, 1, 10 + e);
        fill(w3, w13b, 1, 20000 + e);
        fill(w2, w2b, 1, 40000 + e);
        fill(s1, s13b, 2, 50000 + e, 120);
        fill(s3, s13b, 2, 60000 + e, 120);
        fill(s2, s2b, 2, 70000 + e, 120);
        tab[e] = ExpW{{w1, w2, w3}, {s1, s2, s3}, {1.f, 1.f, 1.f}, 0};
    }
    ExpW* dtab = (ExpW*)dalloc(sizeof(ExpW) * ne);
    CK(cudaMemcpy(dtab, tab.data(), sizeof(ExpW) * ne, cudaMemcpyHostToDevice));
    const int S = 64, NC = 64;
    for (int R : Rs) {
        const int np = R * topk;
        const int bm = R <= 8 ? R : 8;  // moe_bm: rows <= 8 -> rows, else 8-row tiles
        const int max_tiles = bm >= R ? std::min(ne, np) : (np + bm - 1) / bm + std::min(ne, np);
        std::vector<int> sel((size_t)S * np);
        std::vector<float> selw((size_t)S * np);
        uint32_t lcg = 12345u + 977u * R;
        auto rnd = [&]() {
            lcg = lcg * 1664525u + 1013904223u;
            return lcg >> 8;
        };
        // PROBE_POOL=1: a call's tokens draw their experts from a pool of P experts (the trace's prompt rows share
        // experts: the base's bandwidth-bound times put about 17 distinct experts at 4 rows and 28 at 9 and 10)
        const bool pool = getenv("PROBE_POOL") && atoi(getenv("PROBE_POOL")) == 1;
        auto pool_of = [](int r) {
            switch (r) {
                case 1: return 6;
                case 2: return 10;
                case 4: return 27;
                case 8: return 31;
                case 9: return 33;
                case 10: return 33;
                case 12: return 36;
                case 16: return 42;
                default: return std::min(384, 6 * r);
            }
        };
        std::vector<int> pl;
        for (int s = 0; s < S; s++) {
            if (pool) {
                pl.clear();
                while ((int)pl.size() < pool_of(R)) {
                    int e = rnd() % ne;
                    if (std::find(pl.begin(), pl.end(), e) == pl.end()) pl.push_back(e);
                }
            }
            for (int t = 0; t < R; t++) {
                int* row = &sel[(size_t)s * np + t * topk];
                for (int j = 0; j < topk; j++) {
                    int e;
                    bool dup;
                    do {
                        e = pool ? pl[rnd() % pl.size()] : rnd() % ne;
                        dup = false;
                        for (int u = 0; u < j; u++) dup |= row[u] == e;
                    } while (dup);
                    row[j] = e;
                    selw[(size_t)s * np + t * topk + j] = 0.1f + (rnd() % 1000) * 1e-4f;
                }
            }
        }
        int* dsel = (int*)dalloc(sel.size() * 4);
        float* dselw = (float*)dalloc(selw.size() * 4);
        CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dselw, selw.data(), selw.size() * 4, cudaMemcpyHostToDevice));
        int* counts = (int*)dalloc((size_t)ne * 16 * 4);
        int* flag = (int*)dalloc(4);
        CK(cudaMemset(flag, 0, 4));
        std::vector<int*> off(S), ptok(S), order(S), ntiles(S), err(S);
        std::vector<float*> pw(S);
        std::vector<Tile*> tiles(S);
        long tile_sum = 0, tile_max = 0, multi = 0;
        for (int s = 0; s < S; s++) {
            off[s] = (int*)dalloc((ne + 1) * 4);
            ptok[s] = (int*)dalloc(np * 4);
            pw[s] = (float*)dalloc(np * 4);
            order[s] = (int*)dalloc(np * 4);
            tiles[s] = (Tile*)dalloc(max_tiles * 16);
            ntiles[s] = (int*)dalloc(4);
            err[s] = (int*)dalloc(4);
            CK(cudaMemset(err[s], 0, 4));
            CK(cudaMemset(ptok[s], 0, np * 4));
            CK(cudaMemset(pw[s], 0, np * 4));
            RC(memra_v41_moe_plan(dsel + (size_t)s * np, dselw + (size_t)s * np, R, topk, ne, bm, 0, ne, counts,
                                  off[s], ptok[s], pw[s], order[s], tiles[s], ntiles[s], err[s], flag, g_st));
            CK(cudaStreamSynchronize(g_st));
            int nt = 0;
            CK(cudaMemcpy(&nt, ntiles[s], 4, cudaMemcpyDeviceToHost));
            std::vector<Tile> ht(nt);
            CK(cudaMemcpy(ht.data(), tiles[s], (size_t)nt * 16, cudaMemcpyDeviceToHost));
            tile_sum += nt;
            tile_max = std::max<long>(tile_max, nt);
            for (auto& x : ht) multi += x.rows - x.m0 > 1;
        }
        printf("# routing %s rows %d pairs %d bm %d max_tiles %d tiles mean %.1f max %ld, tiles of 2+ rows %.2f a routing\n", pool ? "pool" : "random", R,
               np, bm, max_tiles, (double)tile_sum / S, tile_max, (double)multi / S);
        // activations: the pairs' e4m3 codes and scales
        void* aq = dalloc((size_t)np * d);
        void* q2 = dalloc((size_t)np * inter);
        fill(aq, (size_t)np * d, 0, 77 + R);
        fill(q2, (size_t)np * inter, 0, 88 + R);
#ifdef BASE
        void* as1 = dalloc((size_t)np * d / 32);
        void* as2 = dalloc((size_t)np * inter / 32);
        fill(as1, (size_t)np * d / 32, 2, 91 + R, 118);
        fill(as2, (size_t)np * inter / 32, 2, 92 + R, 118);
        void* hb = dalloc((size_t)np * inter * 2);
        void* o = dalloc((size_t)np * dn * 2);
        CK(cudaStreamSynchronize(g_st));
        std::vector<double> th, to;
        for (int r = 0; r < rounds; r++) {
            th.push_back(graph_us(NC, 5, [&](int i) {
                int s = i % S;
                RC(memra_v41_swiglu_tiles(aq, d, as1, d / 32, tiles[s], ntiles[s], max_tiles, dtab, d / 2, d / 32, 1,
                                          pw[s], 10.0f, hb, inter, hn, d, bm, g_st));
            }));
            to.push_back(graph_us(NC, 5, [&](int i) {
                int s = i % S;
                RC(memra_v41_gemm_f8_tiles(q2, inter, as2, inter / 32, tiles[s], ntiles[s], max_tiles, dtab, 1,
                                           inter / 2, inter / 32, 1, o, dn, 0, dn, inter, bm, g_st));
            }));
        }
        printf("cell base rows %2d w13+swiglu %8.2f us w2 %8.2f us sum %8.2f us\n", R, median(th), median(to),
               median(th) + median(to));
        fflush(stdout);
#else
        void* asf1 = dalloc(sf_bytes(np, d));
        void* asf2 = dalloc(sf_bytes(np, inter));
        fill(asf1, sf_bytes(np, d), 2, 91 + R, 118);
        fill(asf2, sf_bytes(np, inter), 2, 92 + R, 118);
        const Shape shapes[2] = {{"w1w3", hn, d, 2, 0, 2}, {"w2", dn, inter, 1, 1, 1}};
        void* ref0 = dalloc((size_t)np * dn * 2);
        void* ref1 = dalloc((size_t)np * dn * 2);
        void* got0 = dalloc((size_t)np * dn * 2);
        void* got1 = dalloc((size_t)np * dn * 2);
        void* hb = dalloc((size_t)np * inter * 2);
        std::vector<uint16_t> h1((size_t)np * dn), h2((size_t)np * dn);
        auto vars = px::variants();
        if (only) {
            std::vector<px::Var> keep;
            std::string o = std::string(",") + only + ",";
            for (auto& v : vars)
                if (o.find("," + v.name + ",") != std::string::npos) keep.push_back(v);
            vars = keep;
        }
        for (const auto& sh : shapes) {
            auto args = [&](int s, void* d0, void* d1) {
                px::Args x;
                x.a = sh.nproj == 2 ? aq : q2;
                x.asf = sh.nproj == 2 ? asf1 : asf2;
                x.sfrow = nullptr;
                x.tiles = tiles[s];
                x.ntiles = ntiles[s];
                x.max_tiles = max_tiles;
                x.ew = dtab;
                x.nproj = sh.nproj;
                x.p0 = sh.p0;
                x.p1 = sh.p1;
                x.ldb = sh.k / 2;
                x.d0 = d0;
                x.d1 = d1;
                x.ldc = sh.n;
                x.n = sh.n;
                x.k = sh.k;
                x.bm = bm;
                return x;
            };
            auto main_call = [&](int s, void* d0, void* d1) {
                const px::Args x = args(s, d0, d1);
#ifdef HEAD
                memra_v41_moe_small_ring(1);  // the reference: the six-stage warp ring (main's body)
#endif
                RC(memra_v41_moe_small(x.a, x.k, x.asf, nullptr, x.tiles, x.ntiles, x.max_tiles, x.ew, x.nproj, x.p0,
                                       x.p1, x.ldb, x.d0, x.d1, x.ldc, x.n, x.k, x.bm, g_st));
            };
            int fits = memra_v41_moe_small_fits(max_tiles, sh.n, sh.nproj, bm);
            // bytes: every variant against main on every routing (both outputs, the untouched bytes included)
            const size_t ob = (size_t)np * sh.n * 2;
            std::vector<long> diff(vars.size(), 0);
            std::vector<int> ok(vars.size(), 1);
            for (int s = 0; s < S; s++) {
                CK(cudaMemsetAsync(ref0, 0xEE, ob, g_st));
                CK(cudaMemsetAsync(ref1, 0xEE, ob, g_st));
                main_call(s, ref0, ref1);
                CK(cudaMemcpyAsync(h1.data(), ref0, ob, cudaMemcpyDeviceToHost, g_st));
                std::vector<uint16_t> h1b((size_t)np * sh.n);
                CK(cudaMemcpyAsync(h1b.data(), ref1, ob, cudaMemcpyDeviceToHost, g_st));
                CK(cudaStreamSynchronize(g_st));
                for (size_t vi = 0; vi < vars.size(); vi++) {
                    CK(cudaMemsetAsync(got0, 0xEE, ob, g_st));
                    CK(cudaMemsetAsync(got1, 0xEE, ob, g_st));
                    int rc = vars[vi].run(args(s, got0, got1));
                    if (rc == 1) {
                        ok[vi] = 0;
                        continue;
                    }
                    RC(rc);
                    CK(cudaMemcpyAsync(h2.data(), got0, ob, cudaMemcpyDeviceToHost, g_st));
                    std::vector<uint16_t> h2b((size_t)np * sh.n);
                    CK(cudaMemcpyAsync(h2b.data(), got1, ob, cudaMemcpyDeviceToHost, g_st));
                    CK(cudaStreamSynchronize(g_st));
                    for (size_t e = 0; e < (size_t)np * sh.n; e++) diff[vi] += (h1[e] != h2[e]) + (h1b[e] != h2b[e]);
                }
            }
            // time: main and every variant, rounds alternating forward / reverse
            std::vector<std::vector<double>> tv(vars.size());
            std::vector<double> tm;
            std::vector<double> tsw;
            for (int r = 0; r < rounds; r++) {
                auto tmain = [&]() { tm.push_back(graph_us(NC, 5, [&](int i) { main_call(i % S, got0, got1); })); };
                if (r % 2 == 0) tmain();
                for (size_t j = 0; j < vars.size(); j++) {
                    size_t vi = r % 2 == 0 ? j : vars.size() - 1 - j;
                    if (!ok[vi]) continue;
                    tv[vi].push_back(graph_us(NC, 5, [&](int i) { RC(vars[vi].run(args(i % S, got0, got1))); }));
                }
                if (r % 2 == 1) tmain();
                if (sh.nproj == 2)
                    tsw.push_back(graph_us(NC, 5, [&](int i) {
                        int s = i % S;
                        RC(memra_v41_swiglu(got0, got1, hn, pw[s], 10.0f, hb, inter, np, hn, g_st));
                    }));
            }
            double mm = median(tm);
            printf("cell main rows %2d %-5s n %5d k %5d main %8.2f us (rule takes small %d)%s\n", R, sh.name, sh.n,
                   sh.k, mm, fits, sh.nproj == 2 ? "" : "");
            if (sh.nproj == 2) printf("cell main rows %2d swiglu %8.2f us\n", R, median(tsw));
            for (size_t vi = 0; vi < vars.size(); vi++) {
                if (!ok[vi]) {
                    printf("var rows %2d %-5s %-20s does not fit\n", R, sh.name, vars[vi].name.c_str());
                    continue;
                }
                double mv = median(tv[vi]);
                printf("var rows %2d %-5s %-20s %8.2f us  /main %.3f  occ %d  regs %d  bytes differ %ld\n", R, sh.name,
                       vars[vi].name.c_str(), mv, mv / mm, vars[vi].occ(bm) % 1000, vars[vi].occ(bm) / 1000, diff[vi]);
            }
            fflush(stdout);
        }
#endif
        CK(cudaDeviceSynchronize());
    }
    printf("# done\n");
    return 0;
}
