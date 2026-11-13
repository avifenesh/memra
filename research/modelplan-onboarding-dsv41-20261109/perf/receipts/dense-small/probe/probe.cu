// Decode-shaped timing of the DeepSeek-V4.1 FP8/FP4 GEMM calls, one memra build per binary (k3, bs1, cutlass):
// the dense linears (act_quant + GEMM, as `Ops::linear_q`) at the four-rank share and the one-card shapes, and
// the split MoE of one layer on one rank (plan, w1+w3+SwiGLU, w2, the shared expert's w2, combine) exactly as
// each build's tp.rs issues it. Weights rotate through copies larger than L2, so every call reads DRAM.
// Built with -DCUT for the CUTLASS build. Prints one line a cell: eager GPU us a call (events over N calls), host
// us a call (enqueue loop), graph GPU us a call (the same N calls captured, replayed).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

extern "C" {
int memra_v41_act_quant(const void* x, long ldx, const int* rowidx, void* q, long ldq, void* s, long lds, void* y,
                        long ldy, int rows, int k, void* stream);
int memra_v41_moe_plan(const int* sel, const float* selw, int rows, int topk, int ne, int bm, int e0, int e1,
                       int* counts, int* off, int* ptok, float* pw, int* order, void* tiles, int* ntiles, int* err,
                       void* stream);
int memra_v41_moe_combine(const void* slots, long lds, const int* order, int topk, const void* shared, long ldsh,
                          void* out, long ldo, int rows, int d, void* stream);
#ifdef CUT
int memra_v41_act_quant_sf(const void* x, long ldx, const int* rowidx, void* q, long ldq, void* sf, const int* sfrow,
                           int rows, int k, void* stream);
int memra_v41_sf_place(const void* s, long lds, const int* sfrow, void* sf, int rows, int k, void* stream);
int memra_v41_cutlass_gemm(const void* a, long lda, const void* asf, const void* w, long ldw, const void* wsf,
                           int wfmt, void* c, long ldc, int m, int n, int k, int form, void* stream);
int memra_v41_cutlass_sfrow(const int* off, int e0, int ng, int np, int* sfrow, void* stream);
long memra_v41_sf_bytes(long rows, long k);
long memra_v41_cutlass_moe_bytes(int groups);
int memra_v41_cutlass_moe(const int* off, int e0, int ng, int nproj, int proj0, int proj1, const void* ex,
                          const void* a, const void* asf, void* d0, void* d1, long ldc, int n, int k, long ldw,
                          int rows, int form, void* blk, long blk_bytes, void* stream);
int memra_v41_moe_small_fits(int max_tiles, int n, int nproj, int bm);
int memra_v41_moe_small(const void* a, long lda, const void* asf, const int* sfrow, const void* tiles,
                        const int* ntiles, int max_tiles, const void* ew, int nproj, int p0, int p1, long ldb,
                        void* d0, void* d1, long ldc, int n, int k, int bm, void* stream);
int memra_v41_swiglu(const void* gate, const void* up, long ld, const float* wrow, float limit, void* out, long ldo,
                     int rows, int inter, void* stream);
#ifdef NEW
int memra_v41_dense_small(const void* a, long lda, const void* asf, const void* w, long ldw, const void* wsf, int wfmt,
                          void* c, long ldc, int m, int n, int k, void* stream);
int memra_v41_dense_small_fits(int m, int n, int k);
int memra_v41_act_quant_sf_split(const void* x, long ldx, void* q, long ldq, void* sf, int rows, int split, int k,
                                 void* stream);
#endif
#else
int memra_v41_gemm_f8(const void* a, long lda, const void* as, long ldas, const void* b, long ldb, const void* bs,
                      long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32, int m, int n, int k,
                      void* stream);
int memra_v41_gemm_f8_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                            const int* ntiles, int max_tiles, const void* ew, int proj, long ldb, long ldbs,
                            int wfmt, void* c, long ldc, int out_f32, int n, int k, int bm, void* stream);
int memra_v41_swiglu_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                           const int* ntiles, int max_tiles, const void* ew, long ldb, long ldbs, int wfmt,
                           const float* wrow, float limit, void* h, long ldh, int n, int k, int bm, void* stream);
#endif
}

#if defined(NEW) && defined(BUILD_NAME5)
static const char* kBuild = "new5";
#elif defined(NEW)
static const char* kBuild = "new";
#elif defined(CUT)
static const char* kBuild = "cutlass";
#else
static const char* kBuild = BUILD_NAME;
#endif

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#define RC(x)                                                                    \
    do {                                                                         \
        int r_ = (x);                                                            \
        if (r_ != 0) {                                                           \
            fprintf(stderr, "rc %d from %s at %s:%d\n", r_, #x, __FILE__, __LINE__); \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

__device__ __forceinline__ uint32_t hash32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}
// kind 0: e4m3 codes without NaN; 1: any byte (e2m1 pairs); 2: ue8m0 scale in [lo, lo + 12]; 3: bf16 near 1
__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed, int kind, int lo) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        uint32_t h = hash32(i * 0x9E3779B97F4A7C15ULL ^ seed);
        uint8_t v;
        if (kind == 0) v = (uint8_t)(h & 0xF7);
        else if (kind == 1) v = (uint8_t)h;
        else if (kind == 2) v = (uint8_t)(lo + (h % 13));
        else v = (i & 1) ? (uint8_t)(0x3E + ((h >> 8) & 1) + ((h >> 9) & 0x80)) : (uint8_t)h;
        p[i] = v;
    }
}

static cudaStream_t g_st;
// a marker launch before each cell: nsys shows mark_kernel with gridDim.x = the cell id + 1
__global__ void mark_kernel() {}
static int g_mark = 0;
static void nvtxRangePushA(const char* name) {
    g_mark++;
    printf("# mark %d %s\n", g_mark, name);
    mark_kernel<<<g_mark, 1, 0, g_st>>>();
}
static void nvtxRangePop() {}
static int g_sms = 0;
static long g_l2 = 0;

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

// scale bytes of `rows` rows of `k` inputs in each build's layout
static long sf_rows_bytes(long rows, long k) {
#ifdef CUT
    return memra_v41_sf_bytes(rows, k);
#else
    return rows * (k / 32);
#endif
}

struct Res {
    double eager_us, host_us, graph_us;
};

// N calls of f(i): eager (events + host enqueue clock) and as one captured graph
static Res time_calls(int N, const std::function<void(int)>& f, bool graph = true) {
    for (int i = 0; i < 4; i++) f(i);  // warm
    CK(cudaStreamSynchronize(g_st));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    Res r{0, 0, -1};
    double best_e = 1e30, best_h = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        CK(cudaEventRecord(a, g_st));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; i++) f(i);
        auto t1 = std::chrono::steady_clock::now();
        CK(cudaEventRecord(b, g_st));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        best_e = std::min(best_e, ms * 1e3 / N);
        best_h = std::min(best_h, std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
    }
    r.eager_us = best_e;
    r.host_us = best_h;
    if (graph) {
        cudaGraph_t gr;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeRelaxed));
        for (int i = 0; i < N; i++) f(i);
        CK(cudaStreamEndCapture(g_st, &gr));
        CK(cudaGraphInstantiate(&ge, gr, 0));
        CK(cudaGraphLaunch(ge, g_st));
        CK(cudaStreamSynchronize(g_st));
        double best_g = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            CK(cudaEventRecord(a, g_st));
            CK(cudaGraphLaunch(ge, g_st));
            CK(cudaEventRecord(b, g_st));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            best_g = std::min(best_g, ms * 1e3 / N);
        }
        r.graph_us = best_g;
        CK(cudaGraphExecDestroy(ge));
        CK(cudaGraphDestroy(gr));
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    return r;
}

static void report(const char* what, const char* name, int n, int k, int m, const Res& r, int calls_per) {
    printf("%-8s %-6s %-12s n %6d k %5d rows %3d  eager %8.2f us  host %7.2f us  graph %8.2f us  (a cell = %d launches)\n",
           kBuild, what, name, n, k, m, r.eager_us, r.host_us, r.graph_us, calls_per);
    fflush(stdout);
}

// ---- dense linears
struct DenseShape {
    const char* name;
    int n, k;
};

static void dense(int m_list_n, const int* m_list, int N) {
    const DenseShape shapes[] = {
        // the four-rank share (one rank's call)
        {"wq_a", 1280, 5120},      {"wkv", 512, 5120},     {"idx_wq_b", 4096, 1280},
        {"wq_b/4", 8192, 1280},    {"wo_b/4", 1280, 8192}, {"sh_w13/4", 576, 5120},
        {"sh_w2/4", 1280, 2304},   {"wqkv", 1792, 5120},   {"sh_w13f/4", 1152, 5120},
        // one card
        {"wq_b", 32768, 1280},     {"wo_b", 5120, 8192},   {"sh_w13", 2304, 5120},
        {"sh_w2", 5120, 2304},     {"sh_w13f", 4608, 5120},
        // a PRO 6000's four-rank share per SM on this card (n x 82 / 188): CUTLASS's CTAs a card and the body's
        // rows an SM as there
        {"P:wq_a", 576, 5120},     {"P:wkv", 224, 5120},   {"P:wqkv", 768, 5120},   {"P:idx_wq_b", 1792, 1280},
        {"P:wq_b/4", 3584, 1280},  {"P:wo_b/4", 576, 8192}, {"P:sh_w13/4", 256, 5120}, {"P:sh_w13f/4", 512, 5120},
        {"P:sh_w2/4", 576, 2304},
    };
    const int maxm = 16;
    void* x = dalloc((size_t)maxm * 32768 * 2);
    fill(x, (size_t)maxm * 32768 * 2, 3, 7);
    void* q = dalloc((size_t)maxm * 32768);
    void* s = dalloc((size_t)sf_rows_bytes(maxm, 32768) + 4096);
    void* c = dalloc((size_t)maxm * 32768 * 2);
    for (const auto& sh : shapes) {
        long wb = (long)sh.n * sh.k, sb = sf_rows_bytes(sh.n, sh.k);
        long per = al256(wb) + al256(sb);
        // copies: four times L2 at least, at least 8
        int copies = (int)std::max<long>(8, (4 * g_l2 + per - 1) / per);
        uint8_t* pool = (uint8_t*)dalloc((size_t)per * copies);
        for (int i = 0; i < copies; i++) {
            fill(pool + (size_t)i * per, wb, 0, 1000 + i);
            fill(pool + (size_t)i * per + al256(wb), sb, 2, 2000 + i, 120);
        }
        CK(cudaStreamSynchronize(g_st));
        for (int mi = 0; mi < m_list_n; mi++) {
            int m = m_list[mi];
            auto lin = [&](int i, bool aq) {
                uint8_t* w = pool + (size_t)(i % copies) * per;
                uint8_t* ws = w + al256(wb);
#ifdef CUT
                if (aq) RC(memra_v41_act_quant_sf(x, sh.k, nullptr, q, sh.k, s, nullptr, m, sh.k, g_st));
                RC(memra_v41_cutlass_gemm(q, sh.k, s, w, sh.k, ws, 0, c, sh.n, m, sh.n, sh.k, 0, g_st));
#else
                if (aq) RC(memra_v41_act_quant(x, sh.k, nullptr, q, sh.k, s, sh.k / 32, nullptr, 0, m, sh.k, g_st));
                RC(memra_v41_gemm_f8(q, sh.k, s, sh.k / 32, w, sh.k, ws, sh.k / 32, 1.0f, 0, c, sh.n, 0, m, sh.n,
                                     sh.k, g_st));
#endif
            };
            std::string tag = std::string("dense ") + sh.name + " m" + std::to_string(m);
            nvtxRangePushA((tag + " linear").c_str());
            Res r1 = time_calls(N, [&](int i) { lin(i, true); });
            nvtxRangePop();
            report("linear", sh.name, sh.n, sh.k, m, r1, 2);
            nvtxRangePushA((tag + " gemm").c_str());
            Res r2 = time_calls(N, [&](int i) { lin(i, false); });
            nvtxRangePop();
            report("gemm", sh.name, sh.n, sh.k, m, r2, 1);
        }
        CK(cudaFree(pool));
    }
    CK(cudaFree(x));
    CK(cudaFree(q));
    CK(cudaFree(s));
    CK(cudaFree(c));
}

// ---- one four-rank decode layer's dense FP8 linears on one rank, as each build issues them (a linear = its
// act_quant + the GEMM; the shared expert's w1 and w3 read one quantization), weights rotating through copies
static void dense_set(int m, int N, bool scaled) {
    struct L {
        const char* name;
        int n, k, quant;
    };
#ifdef NEW
    const L ls0[] = {{"wqkv", 1792, 5120, 1},   {"wq_b/4", 8192, 1280, 1}, {"idx_wq_b", 4096, 1280, 1},
                    {"wo_b/4", 1280, 8192, 1}, {"sh_w13f/4", 1152, 5120, 1}};
#else
    const L ls0[] = {{"wq_a", 1280, 5120, 1},   {"wkv", 512, 5120, 1},     {"wq_b/4", 8192, 1280, 1},
                    {"idx_wq_b", 4096, 1280, 1}, {"wo_b/4", 1280, 8192, 1}, {"sh_w1/4", 576, 5120, 1},
                    {"sh_w3/4", 576, 5120, 0}};
#endif
    const int nl0 = sizeof(ls0) / sizeof(ls0[0]);
    // scaled: each n x 82 / 188 rounded to 32 (a PRO 6000's four-rank share per SM on the laptop's card)
    std::vector<L> ls(ls0, ls0 + nl0);
    if (scaled)
        for (auto& l : ls) l.n = std::max(32, (int)((long)l.n * 82 / 188 + 16) / 32 * 32);
    const int nl = nl0;
    long per = 0;
    std::vector<long> wo(nl), so(nl);
    for (int j = 0; j < nl; j++) {
        wo[j] = per;
        per += al256((long)ls[j].n * ls[j].k);
        so[j] = per;
        per += al256(sf_rows_bytes(ls[j].n, ls[j].k));
    }
    int copies = (int)std::max<long>(4, (4 * g_l2 + per - 1) / per);
    uint8_t* pool = (uint8_t*)dalloc((size_t)per * copies);
    for (int i = 0; i < copies; i++)
        for (int j = 0; j < nl; j++) {
            fill(pool + (size_t)i * per + wo[j], (size_t)ls[j].n * ls[j].k, 0, 3000 + 16 * i + j);
            fill(pool + (size_t)i * per + so[j], sf_rows_bytes(ls[j].n, ls[j].k), 2, 4000 + 16 * i + j, 120);
        }
    void* x = dalloc((size_t)16 * 8192 * 2);
    fill(x, (size_t)16 * 8192 * 2, 3, 9);
    void* q = dalloc((size_t)16 * 8192);
    void* sc = dalloc((size_t)sf_rows_bytes(16, 8192) + 4096);
    void* c = dalloc((size_t)16 * 8192 * 2);
    CK(cudaStreamSynchronize(g_st));
    auto layer = [&](int i) {
        uint8_t* base = pool + (size_t)(i % copies) * per;
        for (int j = 0; j < nl; j++) {
            const L& l = ls[j];
            uint8_t* w = base + wo[j];
            uint8_t* ws = base + so[j];
#ifdef CUT
            if (l.quant) RC(memra_v41_act_quant_sf(x, l.k, nullptr, q, l.k, sc, nullptr, m, l.k, g_st));
            RC(memra_v41_cutlass_gemm(q, l.k, sc, w, l.k, ws, 0, c, l.n, m, l.n, l.k, 0, g_st));
#else
            if (l.quant) RC(memra_v41_act_quant(x, l.k, nullptr, q, l.k, sc, l.k / 32, nullptr, 0, m, l.k, g_st));
            RC(memra_v41_gemm_f8(q, l.k, sc, l.k / 32, w, l.k, ws, l.k / 32, 1.0f, 0, c, l.n, 0, m, l.n, l.k, g_st));
#endif
        }
    };
    std::string tag = "set m" + std::to_string(m);
    nvtxRangePushA(tag.c_str());
    Res r = time_calls(N, layer);
    nvtxRangePop();
    int launches = 0;
    for (int j = 0; j < nl; j++) launches += 1 + ls[j].quant;
    report("set", scaled ? "P:dense4" : "dense4", nl, 0, m, r, launches);
    CK(cudaFree(pool));
    CK(cudaFree(x));
    CK(cudaFree(q));
    CK(cudaFree(sc));
    CK(cudaFree(c));
}


#ifdef NEW
// ---- the small-row body at 9 to 16 rows (form 4) against CUTLASS's tileN-16 form (3), by launch shape (env V41D_RPB:
// rows a block at least; set by the caller), weights rotating through copies
static void wide(int N) {
    const DenseShape shapes[] = {{"wq_a", 1280, 5120},    {"wqkv", 1792, 5120},    {"wo_b/4", 1280, 8192},
                                 {"sh_w13f/4", 1152, 5120}, {"sh_w2/4", 1280, 2304}, {"P:wqkv", 768, 5120},
                                 {"P:wo_b/4", 576, 8192},   {"P:sh_w13f/4", 512, 5120}, {"P:sh_w2/4", 576, 2304},
                                 {"P:wkv", 224, 5120},      {"P:wq_b/4", 3584, 1280}};
    void* x = dalloc((size_t)16 * 8192);
    fill(x, (size_t)16 * 8192, 0, 7);
    void* s = dalloc((size_t)sf_rows_bytes(16, 8192) + 4096);
    fill(s, sf_rows_bytes(16, 8192), 2, 8, 118);
    void* c = dalloc((size_t)16 * 32768 * 2);
    const char* rpb = getenv("V41D_RPB");
    for (const auto& sh : shapes) {
        long wb = (long)sh.n * sh.k, sb = sf_rows_bytes(sh.n, sh.k);
        long per = al256(wb) + al256(sb);
        int copies = (int)std::max<long>(8, (4 * g_l2 + per - 1) / per);
        uint8_t* pool = (uint8_t*)dalloc((size_t)per * copies);
        for (int i = 0; i < copies; i++) {
            fill(pool + (size_t)i * per, wb, 0, 1000 + i);
            fill(pool + (size_t)i * per + al256(wb), sb, 2, 2000 + i, 120);
        }
        for (int m : {8, 9, 12, 16}) {
            auto g = [&](int i, int form) {
                uint8_t* w = pool + (size_t)(i % copies) * per;
                RC(memra_v41_cutlass_gemm(x, sh.k, s, w, sh.k, w + al256(wb), 0, c, sh.n, m, sh.n, sh.k, form, g_st));
            };
            Res rc = time_calls(N, [&](int i) { g(i, m <= 8 ? 2 : 3); });
            Res rs = time_calls(N, [&](int i) { g(i, 4); });
            printf("wide %-12s m %2d rpb %3s cutlass %8.2f small %8.2f ratio %.3f\n", sh.name, m, rpb ? rpb : "-",
                   rc.graph_us, rs.graph_us, rs.graph_us / rc.graph_us);
            fflush(stdout);
        }
        CK(cudaFree(pool));
    }
}
#endif

#ifdef NEW
// ---- byte check: the small-row body (form 4) and form 0 against CUTLASS's normal form (1) and its swapped forms
static void check() {
    struct S {
        int n, k;
    };
    const S shapes[] = {{1280, 5120}, {512, 5120},  {1792, 5120}, {4096, 1280}, {8192, 1280},
                        {1280, 8192}, {576, 5120},  {1152, 5120}, {1280, 2304}, {32768, 1280},
                        {5120, 8192}, {2304, 5120}, {4608, 5120}, {5120, 2304}};
    const int maxm = 16;
    void* x = dalloc((size_t)maxm * 8192);
    void* xs = dalloc((size_t)sf_rows_bytes(maxm, 8192) + 4096);
    void* ref = dalloc((size_t)maxm * 32768 * 2);
    void* got = dalloc((size_t)maxm * 32768 * 2);
    std::vector<uint16_t> h1((size_t)maxm * 32768), h2((size_t)maxm * 32768);
    long checks = 0, bad = 0;
    int si = 0;
    for (const auto& sh : shapes) {
        si++;
        for (int wf : {0, 1}) {
            long ldw = wf == 0 ? sh.k : sh.k / 2;
            uint8_t* w = (uint8_t*)dalloc((size_t)sh.n * ldw);
            uint8_t* ws = (uint8_t*)dalloc((size_t)sf_rows_bytes(sh.n, sh.k));
            fill(w, (size_t)sh.n * ldw, wf == 0 ? 0 : 1, 100 * si + wf);
            fill(ws, sf_rows_bytes(sh.n, sh.k), 2, 200 * si + wf, 112);
            for (int m = 1; m <= maxm; m++) {
                fill(x, (size_t)m * sh.k, 0, 300 * si + 7 * m + wf);
                fill(xs, sf_rows_bytes(m, sh.k), 2, 400 * si + 7 * m + wf, 118);
                CK(cudaMemsetAsync(ref, 0xEE, (size_t)m * sh.n * 2, g_st));
                RC(memra_v41_cutlass_gemm(x, sh.k, xs, w, ldw, ws, wf, ref, sh.n, m, sh.n, sh.k, 1, g_st));
                CK(cudaMemcpyAsync(h1.data(), ref, (size_t)m * sh.n * 2, cudaMemcpyDeviceToHost, g_st));
                CK(cudaStreamSynchronize(g_st));
                char line[256];
                int o = snprintf(line, sizeof line, "check n %5d k %5d wf %d m %2d:", sh.n, sh.k, wf, m);
                for (int form : {0, 2, 3, 4}) {
                    CK(cudaMemsetAsync(got, 0xDD, (size_t)m * sh.n * 2, g_st));
                    RC(memra_v41_cutlass_gemm(x, sh.k, xs, w, ldw, ws, wf, got, sh.n, m, sh.n, sh.k, form, g_st));
                    CK(cudaMemcpyAsync(h2.data(), got, (size_t)m * sh.n * 2, cudaMemcpyDeviceToHost, g_st));
                    CK(cudaStreamSynchronize(g_st));
                    long d = 0;
                    for (long e = 0; e < (long)m * sh.n; e++) d += h1[e] != h2[e];
                    checks++;
                    bad += d != 0;
                    o += snprintf(line + o, sizeof line - o, " form%d %ld", form, d);
                }
                printf("%s\n", line);
            }
            CK(cudaFree(w));
            CK(cudaFree(ws));
        }
    }
    printf("# check: %ld comparisons, %ld differ\n", checks, bad);
    fflush(stdout);
}
#endif

// ---- the split MoE of one layer on one rank (four ranks: 576 hidden columns, 1280 output columns a rank)
struct ExpW {
    const uint8_t* w[3];
    const uint8_t* s[3];
    float scale2[3];
    int pad;
};
struct Tile {
    int row0, rows, m0, expert;
};

static void moe(int R, int N) {
    const int ne = 384, topk = 6, d = 5120, inter = 2304, hn = 576, dn = 1280, h0 = 0, dim0 = 0;
    const int np = R * topk;
    const int bm = R <= 8 ? R : 8;  // moe_bm: rows <= 8 -> rows, else 8-row tiles
    const int max_tiles = R <= 8 ? std::min(ne, np) : (np + bm - 1) / bm + std::min(ne, np);
    // the experts: w1, w3 [hn][d/2], w2 [dn][inter/2], scales in the build's layout
    long w13b = (long)hn * d / 2, w2b = (long)dn * inter / 2;
    long s13b = sf_rows_bytes(hn, d), s2b = sf_rows_bytes(dn, inter);
    long per = al256(w13b) * 2 + al256(w2b) + al256(s13b) * 2 + al256(s2b);
    uint8_t* pool = (uint8_t*)dalloc((size_t)per * ne);
    std::vector<ExpW> tab(ne);
    for (int e = 0; e < ne; e++) {
        uint8_t* p = pool + (size_t)e * per;
        uint8_t *w1 = p, *w3 = w1 + al256(w13b), *w2 = w3 + al256(w13b);
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
    // the shared expert's w2 slice (FP8 dense [dn][inter]), 8 copies
    long sw2b = (long)dn * inter, ssb = sf_rows_bytes(dn, inter), sper = al256(sw2b) + al256(ssb);
    uint8_t* spool = (uint8_t*)dalloc((size_t)sper * 8);
    for (int i = 0; i < 8; i++) {
        fill(spool + (size_t)i * sper, sw2b, 0, 80000 + i);
        fill(spool + (size_t)i * sper + al256(sw2b), ssb, 2, 90000 + i, 120);
    }
    // S routings: rows x topk distinct experts each
    const int S = 64;
    std::vector<int> sel((size_t)S * np);
    std::vector<float> selw((size_t)S * np);
    uint32_t lcg = 12345;
    auto rnd = [&]() { lcg = lcg * 1664525u + 1013904223u; return lcg >> 8; };
    for (int s = 0; s < S; s++)
        for (int t = 0; t < R; t++) {
            int* row = &sel[(size_t)s * np + t * topk];
            for (int j = 0; j < topk; j++) {
                int e;
                bool dup;
                do {
                    e = rnd() % ne;
                    dup = false;
                    for (int u = 0; u < j; u++) dup |= row[u] == e;
                } while (dup);
                row[j] = e;
                selw[(size_t)s * np + t * topk + j] = 0.1f + (rnd() % 1000) * 1e-4f;
            }
        }
    int* dsel = (int*)dalloc(sel.size() * 4);
    float* dselw = (float*)dalloc(selw.size() * 4);
    CK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dselw, selw.data(), selw.size() * 4, cudaMemcpyHostToDevice));
    // plan buffers, one set a routing (a captured graph replays every call with its own plan)
    int* counts = (int*)dalloc(4 * ne * 4);
    std::vector<int*> off(S), ptok(S), order(S), ntiles(S), err(S);
    std::vector<float*> pw(S);
    std::vector<Tile*> tiles(S);
    for (int s = 0; s < S; s++) {
        off[s] = (int*)dalloc((ne + 1) * 4);
        ptok[s] = (int*)dalloc(np * 4);
        pw[s] = (float*)dalloc(np * 4);
        order[s] = (int*)dalloc(np * 4);
        tiles[s] = (Tile*)dalloc(max_tiles * 16);
        ntiles[s] = (int*)dalloc(4);
        err[s] = (int*)dalloc(4);
        CK(cudaMemset(err[s], 0, 4));
    }
    void* x = dalloc((size_t)R * d * 2);
    fill(x, (size_t)R * d * 2, 3, 77);
    void* aq = dalloc((size_t)np * d);
    void* hb = dalloc((size_t)(np + R) * inter * 2);
    fill(hb, (size_t)(np + R) * inter * 2, 3, 78);
    void* q2 = dalloc((size_t)(np + R) * inter);
    void* sc2 = dalloc((size_t)(np + R) * inter / 32 + 256);
    void* o = dalloc((size_t)(np + R) * dn * 2);
    void* out = dalloc((size_t)R * d * 2);
#ifdef CUT
    long pad_rows = np + 127L * std::min(ne, np);
    void* asf = dalloc(memra_v41_sf_bytes(pad_rows, d));
    void* asf2 = dalloc(memra_v41_sf_bytes(pad_rows, inter));
    void* ssf = dalloc(memra_v41_sf_bytes(R, inter));
    int* sfrow = (int*)dalloc(np * 4);
    int* sfrow2 = (int*)dalloc(np * 4);
    void* gb = dalloc((size_t)np * hn * 2);
    void* ub = dalloc((size_t)np * hn * 2);
    long blk13 = memra_v41_cutlass_moe_bytes(ne * 2), blk2 = memra_v41_cutlass_moe_bytes(ne);
    void* blk_a = dalloc(blk13);
    void* blk_b = dalloc(blk2);
    int small13 = memra_v41_moe_small_fits(max_tiles, hn, 2, bm), small2 = memra_v41_moe_small_fits(max_tiles, dn, 1, bm);
#ifdef NEW
    void* asf_np = dalloc(memra_v41_sf_bytes(np, d));
    void* asf_o = dalloc(memra_v41_sf_bytes(np, inter) + memra_v41_sf_bytes(R, inter));
#endif
#else
    void* as_ = dalloc((size_t)np * d / 32 + 256);
#endif
    CK(cudaStreamSynchronize(g_st));
    auto plan = [&](int i) {
        int s = i % S;
        RC(memra_v41_moe_plan(dsel + (size_t)s * np, dselw + (size_t)s * np, R, topk, ne, bm, 0, ne, counts, off[s],
                              ptok[s], pw[s], order[s], tiles[s], ntiles[s], err[s], g_st));
    };
    auto hidden_p = [&](int i, bool fresh) {
        (void)fresh;
        int s = i % S;
#ifdef NEW
        if (fresh && small13) {  // the pairs' rows quantized straight into a layout of np rows (no sfrow kernel)
            RC(memra_v41_act_quant_sf(x, d, ptok[s], aq, d, asf_np, nullptr, np, d, g_st));
            RC(memra_v41_moe_small(aq, d, asf_np, nullptr, tiles[s], ntiles[s], max_tiles, dtab, 2, 0, 2, d / 2, gb, ub,
                                   hn, hn, d, bm, g_st));
            RC(memra_v41_swiglu(gb, ub, hn, pw[s], 10.0f, (uint8_t*)hb + h0 * 2, inter, np, hn, g_st));
            return;
        }
#endif
#ifdef CUT
        RC(memra_v41_cutlass_sfrow(off[s], 0, ne, np, sfrow, g_st));
        RC(memra_v41_act_quant_sf(x, d, ptok[s], aq, d, asf, sfrow, np, d, g_st));
        if (small13)
            RC(memra_v41_moe_small(aq, d, asf, sfrow, tiles[s], ntiles[s], max_tiles, dtab, 2, 0, 2, d / 2, gb, ub, hn,
                                   hn, d, bm, g_st));
        else
            RC(memra_v41_cutlass_moe(off[s], 0, ne, 2, 0, 2, dtab, aq, asf, gb, ub, hn, hn, d, d / 2, R, 0, blk_a, blk13,
                                     g_st));
        RC(memra_v41_swiglu(gb, ub, hn, pw[s], 10.0f, (uint8_t*)hb + h0 * 2, inter, np, hn, g_st));
#else
        RC(memra_v41_act_quant(x, d, ptok[s], aq, d, as_, d / 32, nullptr, 0, np, d, g_st));
        RC(memra_v41_swiglu_tiles(aq, d, as_, d / 32, tiles[s], ntiles[s], max_tiles, dtab, d / 2, d / 32, 1, pw[s],
                                  10.0f, (uint8_t*)hb + h0 * 2, inter, hn, d, bm, g_st));
#endif
    };
    auto outp_p = [&](int i, bool fresh) {
        (void)fresh;
        int s = i % S;
        uint8_t* sw2 = spool + (size_t)(i % 8) * sper;
        uint8_t* sws = sw2 + al256(sw2b);
#ifdef NEW
        if (fresh && small2) {  // the pairs' and the shared expert's hidden rows quantized straight into two layouts
            RC(memra_v41_act_quant_sf_split(hb, inter, q2, inter, asf_o, np + R, np, inter, g_st));
            RC(memra_v41_moe_small(q2, inter, asf_o, nullptr, tiles[s], ntiles[s], max_tiles, dtab, 1, 1, 1, inter / 2,
                                   o, nullptr, dn, dn, inter, bm, g_st));
            RC(memra_v41_cutlass_gemm((uint8_t*)q2 + (size_t)np * inter, inter,
                                      (uint8_t*)asf_o + memra_v41_sf_bytes(np, inter), sw2, inter, sws, 0,
                                      (uint8_t*)o + (size_t)np * dn * 2, dn, R, dn, inter, 0, g_st));
            RC(memra_v41_moe_combine(o, dn, order[s], topk, (uint8_t*)o + (size_t)np * dn * 2, dn,
                                     (uint8_t*)out + dim0 * 2, d, R, dn, g_st));
            return;
        }
#endif
        RC(memra_v41_act_quant(hb, inter, nullptr, q2, inter, sc2, inter / 32, nullptr, 0, np + R, inter, g_st));
#ifdef CUT
        RC(memra_v41_cutlass_sfrow(off[s], 0, ne, np, sfrow2, g_st));
        RC(memra_v41_sf_place(sc2, inter / 32, sfrow2, asf2, np, inter, g_st));
        RC(memra_v41_sf_place((uint8_t*)sc2 + (size_t)np * inter / 32, inter / 32, nullptr, ssf, R, inter, g_st));
        if (small2)
            RC(memra_v41_moe_small(q2, inter, asf2, sfrow2, tiles[s], ntiles[s], max_tiles, dtab, 1, 1, 1, inter / 2, o,
                                   nullptr, dn, dn, inter, bm, g_st));
        else
            RC(memra_v41_cutlass_moe(off[s], 0, ne, 1, 1, 1, dtab, q2, asf2, o, nullptr, dn, dn, inter, inter / 2, R, 0,
                                     blk_b, blk2, g_st));
        RC(memra_v41_cutlass_gemm((uint8_t*)q2 + (size_t)np * inter, inter, ssf, sw2, inter, sws, 0,
                                  (uint8_t*)o + (size_t)np * dn * 2, dn, R, dn, inter, 0, g_st));
#else
        RC(memra_v41_gemm_f8_tiles(q2, inter, sc2, inter / 32, tiles[s], ntiles[s], max_tiles, dtab, 1, inter / 2,
                                   inter / 32, 1, o, dn, 0, dn, inter, bm, g_st));
        RC(memra_v41_gemm_f8((uint8_t*)q2 + (size_t)np * inter, inter, (uint8_t*)sc2 + (size_t)np * inter / 32,
                             inter / 32, sw2, inter, sws, inter / 32, 1.0f, 0, (uint8_t*)o + (size_t)np * dn * 2, dn, 0,
                             R, dn, inter, g_st));
#endif
        RC(memra_v41_moe_combine(o, dn, order[s], topk, (uint8_t*)o + (size_t)np * dn * 2, dn,
                                 (uint8_t*)out + dim0 * 2, d, R, dn, g_st));
    };
    auto hidden = [&](int i) { hidden_p(i, true); };
    auto outp = [&](int i) { outp_p(i, true); };
#ifdef NEW
    {  // the new path's bytes against the old path's on a few routings: hidden rows, then the layer's output columns
        size_t hbn = (size_t)(np + R) * inter * 2, on = (size_t)R * d * 2;
        std::vector<uint8_t> a1(hbn), a2(hbn), b1(on), b2(on);
        long dh = 0, dd = 0;
        for (int i = 0; i < 6; i++) {
            plan(i);
            fill(hb, hbn, 3, 500 + i);
            hidden_p(i, false);
            CK(cudaMemcpyAsync(a1.data(), hb, hbn, cudaMemcpyDeviceToHost, g_st));
            fill(hb, hbn, 3, 500 + i);
            hidden_p(i, true);
            CK(cudaMemcpyAsync(a2.data(), hb, hbn, cudaMemcpyDeviceToHost, g_st));
            CK(cudaStreamSynchronize(g_st));
            for (size_t e = 0; e < hbn; e++) dh += a1[e] != a2[e];
            CK(cudaMemsetAsync(out, 0, on, g_st));
            outp_p(i, false);
            CK(cudaMemcpyAsync(b1.data(), out, on, cudaMemcpyDeviceToHost, g_st));
            CK(cudaMemsetAsync(out, 0, on, g_st));
            outp_p(i, true);
            CK(cudaMemcpyAsync(b2.data(), out, on, cudaMemcpyDeviceToHost, g_st));
            CK(cudaStreamSynchronize(g_st));
            for (size_t e = 0; e < on; e++) dd += b1[e] != b2[e];
        }
        printf("moecheck rows %d small13 %d small2 %d: hidden bytes differ %ld, output bytes differ %ld\n", R, small13,
               small2, dh, dd);
    }
#endif
    std::string tag = "moe rows" + std::to_string(R);
#if defined(NEW)
    int lh = small13 ? 3 : 5, lo = small2 ? 4 : 8;
#elif defined(CUT)
    int lh = small13 ? 4 : 5, lo = small2 ? 7 : 8;
#else
    int lh = 2, lo = 4;
#endif
    nvtxRangePushA((tag + " plan").c_str());
    Res rp = time_calls(N, plan);
    nvtxRangePop();
    report("moe", "plan", ne, 0, R, rp, 1);
    nvtxRangePushA((tag + " hidden").c_str());
    Res rh = time_calls(N, [&](int i) { plan(i); hidden(i); });
    nvtxRangePop();
    report("moe", "plan+hid", hn, d, R, rh, 1 + lh);
    nvtxRangePushA((tag + " out").c_str());
    Res ro = time_calls(N, [&](int i) { plan(i); outp(i); });
    nvtxRangePop();
    report("moe", "plan+out", dn, inter, R, ro, 1 + lo);
    nvtxRangePushA((tag + " all").c_str());
    Res ra = time_calls(N, [&](int i) { plan(i); hidden(i); outp(i); });
    nvtxRangePop();
    report("moe", "plan+all", dn, inter, R, ra, 1 + lh + lo);
    CK(cudaStreamSynchronize(g_st));
    int herr[1];
    CK(cudaMemcpy(herr, err[0], 4, cudaMemcpyDeviceToHost));
    if (herr[0]) printf("plan err %d\n", herr[0]);
    CK(cudaDeviceSynchronize());
    // freed with the process
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    g_sms = p.multiProcessorCount;
    g_l2 = p.l2CacheSize;
    printf("# build %s device %s SMs %d L2 %ld MB\n", kBuild, p.name, g_sms, g_l2 >> 20);
    const char* what = argc > 1 ? argv[1] : "all";
    int N = argc > 2 ? atoi(argv[2]) : 64;
    if (!strcmp(what, "all") || !strcmp(what, "dense")) {
        int ms[] = {1, 4, 8, 16};
        dense(4, ms, N);
    }
    if (!strcmp(what, "all") || !strcmp(what, "set")) {
        for (int m : {1, 4, 8, 16}) dense_set(m, N, false);
        for (int m : {1, 4, 8, 16}) dense_set(m, N, true);
    }
    if (!strcmp(what, "all") || !strcmp(what, "moe")) {
        for (int R : {1, 4, 8}) moe(R, N);
    }
    if (!strcmp(what, "step")) {  // the step model's rows: a layer's dense set and its split MoE
        for (int m : {1, 8, 9, 12, 16}) dense_set(m, N, false);
        for (int m : {1, 8, 9, 12, 16}) dense_set(m, N, true);
        for (int R : {1, 8, 9, 12, 16}) moe(R, N);
    }
#ifdef NEW
    if (!strcmp(what, "check")) check();
    if (!strcmp(what, "wide")) wide(N);
#endif
    CK(cudaDeviceSynchronize());
    printf("# done\n");
    return 0;
}
