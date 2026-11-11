// Probe (scratch, not memra): CUTLASS 4.8.1's sm_120 block-scaled grouped GEMM against memra's dsv41 GEMM on the same
// inputs. A: e4m3 activations [rows][K] + one ue8m0 per row and 32 inputs; B: the experts' MXFP4 weights [N][K/2]
// (low nibble first) + one ue8m0 per row and 32 inputs; D: bf16. Shapes: w13 (N 2304, K 5120) at R rows an expert
// for E experts (one group an expert), and wq_a (N 1280, K 5120, e4m3 weights, 32 x 32 block scales) at 4,096 rows.
// For each: CUTLASS's time and output bytes, memra's dispatched kernel's time, and both against memra's reference
// tile byte for byte. usage: cutlass_probe [E R]... (default: 96 64, 96 512)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/version.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);           \
            exit(2);                                                                            \
        }                                                                                       \
    } while (0)

// ---- memra's entry points (cu/dsv41_gpu.cu, linked in)
extern "C" int memra_v41_gemm_f8_ref(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                     const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                     int m, int n, int k, void* stream);
extern "C" int memra_v41_gemm_f8(const void* a, long lda, const void* as, long ldas, const void* b, long ldb,
                                 const void* bs, long ldbs, float bscale2, int wfmt, void* c, long ldc, int out_f32,
                                 int m, int n, int k, void* stream);
extern "C" int memra_v41_gemm_f8_tiles(const void* a, long lda, const void* as, long ldas, const void* tiles,
                                       const int* ntiles, int max_tiles, const void* ew, int proj, long ldb, long ldbs,
                                       int wfmt, void* c, long ldc, int out_f32, int n, int k, int bm, void* stream);
struct V41Tile { int row0, rows, m0, expert; };
struct V41ExpW { const uint8_t* w[3]; const uint8_t* s[3]; float scale2[3]; int pad; };

// ---- the CUTLASS kernel (two operand-type variants)
template <class EA, class EB, int AlignB, class ED = cutlass::bfloat16_t>
struct Cfg {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using ElementA = EA;
    using ElementB = EB;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementD = ED;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int AlignA = 16, AlignD = 128 / cutlass::sizeof_bits<ED>::value;
    using Arch = cutlass::arch::Sm120;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<_128, _128, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*,
        AlignD, ElementD, LayoutD*, AlignD, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OpClass, ElementA, LayoutA*, AlignA, ElementB, LayoutB*, AlignB, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
    using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};
using CfgF4 = Cfg<cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::mx_float4_t<cutlass::float_e2m1_t>, 128>;
using CfgF8 = Cfg<cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::mx_float8_t<cutlass::float_e4m3_t>, 16>;
// decode widths: weights on A (rows = output columns), activations on B (columns = the call's rows), D column-major
template <class TileN, class ED = cutlass::bfloat16_t>
struct CfgDec {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementD = ED;
    using LayoutD = cutlass::layout::ColumnMajor;
    using Arch = cutlass::arch::Sm120;
    using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<_128, TileN, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutD*,
        128 / cutlass::sizeof_bits<ED>::value, ElementD, LayoutD*, 128 / cutlass::sizeof_bits<ED>::value,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OpClass, ElementA, LayoutA*, 128, ElementB, LayoutB*, 16, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::InternalStrideA;
    using StrideB = typename Kernel::InternalStrideB;
    using StrideD = typename Kernel::InternalStrideD;
    using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
    using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

using CfgF4f = Cfg<cutlass::mx_float8_t<cutlass::float_e4m3_t>, cutlass::mx_float4_t<cutlass::float_e2m1_t>, 128, float>;

static uint64_t rs = 0xd5415c4e;
static std::vector<uint32_t> g_last;  // the last fp32 CUTLASS output (row-major [rows][N])
static uint32_t rnd() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(rs >> 33); }
static void fill_e4m3(std::vector<uint8_t>& v) { for (auto& b : v) { b = rnd() & 0xFF; if ((b & 0x7F) == 0x7F) b ^= 1; } }
static void fill_bytes(std::vector<uint8_t>& v) { for (auto& b : v) b = rnd() & 0xFF; }
static void fill_range(std::vector<uint8_t>& v, int lo, int hi) { for (auto& b : v) b = lo + rnd() % (hi - lo + 1); }
template <class T> static T* up(const std::vector<T>& h) { T* d; CK(cudaMalloc(&d, h.size() * sizeof(T) + 256)); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }

// One grouped run: G groups, group g = rows [g*R, g*R+R) of A against weight g; WF 1 (FP4) or 0 (FP8 32x32 blocks).
template <class C>
static void run(const char* name, int G, int R, int N, int K, int wf, int reps) {
    rs = 0xd5415c4e;
    int KB = K / 32, M = G * R;
    long wrow = wf ? K / 2 : K;  // bytes a weight row
    std::vector<uint8_t> A((size_t)M * K), As((size_t)M * KB), W((size_t)G * N * wrow);
    std::vector<uint8_t> Ws(wf ? (size_t)G * N * KB : (size_t)G * ((N + 31) / 32) * KB);
    fill_e4m3(A); fill_range(As, 118, 130);
    if (wf) { fill_bytes(W); fill_range(Ws, 112, 124); } else { fill_e4m3(W); fill_range(Ws, 100, 127); }
    uint8_t *dA = up(A), *dAs = up(As), *dW = up(W), *dWs = up(Ws);
    constexpr int EB_ = sizeof(typename C::ElementD);  // 2: bf16 out, 4: the fp32 accumulator
    const int OF = EB_ == 4;
    size_t outb = (size_t)M * N * EB_;
    uint8_t *dRef, *dOurs, *dCut;
    CK(cudaMalloc(&dRef, outb)); CK(cudaMalloc(&dOurs, outb)); CK(cudaMalloc(&dCut, outb));
    long wsl = wf ? (long)N * KB : (long)((N + 31) / 32) * KB;  // scale bytes a weight
    // the reference tile, group by group
    for (int g = 0; g < G; ++g) {
        int rc = memra_v41_gemm_f8_ref(dA + (size_t)g * R * K, K, dAs + (size_t)g * R * KB, KB, dW + (size_t)g * N * wrow,
                                       wrow, dWs + (size_t)g * wsl, KB, 1.f, wf, dRef + (size_t)g * R * N * EB_, N, OF, R, N, K, 0);
        if (rc) { printf("ref rc %d\n", rc); exit(3); }
    }
    // memra's dispatched kernel: the tile list (FP4) or the dense call (one group)
    std::vector<V41Tile> tl; std::vector<V41ExpW> ew(G);
    int bm = R >= 128 ? 128 : 64;
    for (int g = 0; g < G; ++g) {
        for (int m0 = 0; m0 < R; m0 += bm) tl.push_back({g * R, R, m0, g});
        for (int p = 0; p < 3; ++p) { ew[g].w[p] = dW + (size_t)g * N * wrow; ew[g].s[p] = dWs + (size_t)g * wsl; ew[g].scale2[p] = 1.f; }
        ew[g].pad = 0;
    }
    int nt = (int)tl.size();
    V41Tile* dT = up(tl); V41ExpW* dE = up(ew); int* dN = up(std::vector<int>{nt});
    auto ours = [&]() {
        int rc = wf ? memra_v41_gemm_f8_tiles(dA, K, dAs, KB, dT, dN, nt, dE, 0, wrow, KB, 1, dOurs, N, OF, N, K, bm, 0)
                    : memra_v41_gemm_f8(dA, K, dAs, KB, dW, wrow, dWs, KB, 1.f, 0, dOurs, N, OF, M, N, K, 0);
        if (rc) { printf("ours rc %d\n", rc); exit(3); }
    };
    // CUTLASS: scale factors repacked into its layout; FP8 weights take their block's code on every column
    using G_ = typename C::Gemm;
    using PS = typename C::ProblemShape::UnderlyingProblemShape;
    std::vector<PS> ps; std::vector<typename C::StrideA> sa; std::vector<typename C::StrideB> sb; std::vector<typename C::StrideD> sd;
    std::vector<typename C::LayoutSFA> lsa; std::vector<typename C::LayoutSFB> lsb;
    std::vector<const void*> pa, pb, psa, psb; std::vector<void*> pd;
    using SFT = cutlass::float_ue8m0_t;
    std::vector<uint8_t*> keep;
    for (int g = 0; g < G; ++g) {
        ps.push_back({R, N, K});
        sa.push_back(cutlass::make_cute_packed_stride(typename C::StrideA{}, {R, K, 1}));
        sb.push_back(cutlass::make_cute_packed_stride(typename C::StrideB{}, {N, K, 1}));
        sd.push_back(cutlass::make_cute_packed_stride(typename C::StrideD{}, {R, N, 1}));
        auto la = C::SfCfg::tile_atom_to_shape_SFA(make_shape(R, N, K, 1));
        auto lb = C::SfCfg::tile_atom_to_shape_SFB(make_shape(R, N, K, 1));
        lsa.push_back(la); lsb.push_back(lb);
        std::vector<uint8_t> hsa(size(filter_zeros(la)), 0), hsb(size(filter_zeros(lb)), 0);
        for (int r = 0; r < R; ++r) for (int kb = 0; kb < KB; ++kb) hsa[la(r, kb * 32, 0)] = As[((size_t)g * R + r) * KB + kb];
        for (int n = 0; n < N; ++n) for (int kb = 0; kb < KB; ++kb)
            hsb[lb(n, kb * 32, 0)] = wf ? Ws[((size_t)g * N + n) * KB + kb] : Ws[(size_t)g * wsl + (n / 32) * KB + kb];
        uint8_t *x = up(hsa), *y = up(hsb); keep.push_back(x); keep.push_back(y);
        pa.push_back(dA + (size_t)g * R * K); pb.push_back(dW + (size_t)g * N * wrow);
        psa.push_back(x); psb.push_back(y); pd.push_back(dCut + (size_t)g * R * N * EB_);
    }
    auto dps = up(ps); auto dsa = up(sa); auto dsb = up(sb); auto dsd = up(sd); auto dlsa = up(lsa); auto dlsb = up(lsb);
    auto dpa = up(pa); auto dpb = up(pb); auto dpsa = up(psa); auto dpsb = up(psb); auto dpd = up(pd);
    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename G_::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {G, dps, ps.data()},
                                {(const typename C::Main::ElementA**)dpa, dsa, (const typename C::Main::ElementB**)dpb, dsb,
                                 (const SFT**)dpsa, dlsa, (const SFT**)dpsb, dlsb},
                                {{}, nullptr, dsd, (typename C::ElementD**)dpd, dsd},
                                hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    G_ gemm;
    size_t wsz = G_::get_workspace_size(args);
    void* ws = nullptr; if (wsz) CK(cudaMalloc(&ws, wsz));
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("%s: CUTLASS cannot implement\n", name); return; }
    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) { printf("%s: CUTLASS initialize failed\n", name); return; }
    auto cut = [&]() { if (gemm.run() != cutlass::Status::kSuccess) { printf("CUTLASS run failed\n"); exit(4); } };
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto f) {
        for (int i = 0; i < 3; ++i) f();
        CK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (int i = 0; i < reps; ++i) { CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms); }
        std::sort(t.begin(), t.end()); return t[t.size() / 2];
    };
    float tc = timeit(cut), to = timeit(ours);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hr(outb), ho(outb), hc(outb);
    CK(cudaMemcpy(hr.data(), dRef, outb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho.data(), dOurs, outb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), dCut, outb, cudaMemcpyDeviceToHost));
    if (EB_ == 4) { g_last.assign((uint32_t*)hc.data(), (uint32_t*)hc.data() + outb / 4); }
    size_t dco = 0, dcr = 0, n2 = outb / EB_; double maxrel = 0; int shown = 0;
    for (size_t i = 0; i < n2; ++i) {
        uint32_t r, o, c;
        if (EB_ == 2) { r = (uint32_t)((uint16_t*)hr.data())[i] << 16; o = (uint32_t)((uint16_t*)ho.data())[i] << 16; c = (uint32_t)((uint16_t*)hc.data())[i] << 16; }
        else { r = ((uint32_t*)hr.data())[i]; o = ((uint32_t*)ho.data())[i]; c = ((uint32_t*)hc.data())[i]; }
        dco += o != r; dcr += c != r;
        if (c != r) {
            float a, b; memcpy(&a, &r, 4); memcpy(&b, &c, 4);
            double rel = fabs((double)a - b) / fmax(fabs((double)a), 1e-30); if (rel > maxrel) maxrel = rel;
            if (shown++ < 8) {
                size_t row = i / N, col = i % N;
                printf("  differs: group %zu row %zu col %zu: reference %08x (%.9g) CUTLASS %08x (%.9g)\n", row / R, row % R, col, r, a, c, b);
            }
        }
    }
    double fl = 2.0 * M * N * (double)K;
    printf("%s%s G %d R %d N %d K %d: CUTLASS %.4f ms (%.1f TFLOPS), memra %.4f ms (%.1f TFLOPS), CUTLASS/memra %.3f | "
           "outputs differing from the reference tile: memra %zu, CUTLASS %zu of %zu (CUTLASS max rel %.3e)\n",
           name, OF ? " (fp32 out)" : "", G, R, N, K, tc, fl / tc / 1e9, to, fl / to / 1e9, tc / to, dco, dcr, n2, maxrel);
    for (auto p : keep) cudaFree(p);
    cudaFree(dA); cudaFree(dAs); cudaFree(dW); cudaFree(dWs); cudaFree(dRef); cudaFree(dOurs); cudaFree(dCut);
    if (ws) cudaFree(ws);
}

// Decode widths: G experts with R rows each (R <= 8), MXFP4 weights; CUTLASS with the operands swapped against memra's
// small-row tile list and the reference tile.
template <class C>
static void run_dec(const char* name, int G, int R, int N, int K, int reps) {
    rs = 0xd5415c4e;
    int KB = K / 32, M = G * R;
    long wrow = K / 2;
    std::vector<uint8_t> A((size_t)M * K), As((size_t)M * KB), W((size_t)G * N * wrow), Ws((size_t)G * N * KB);
    fill_e4m3(A); fill_range(As, 118, 130); fill_bytes(W); fill_range(Ws, 112, 124);
    uint8_t *dA = up(A), *dAs = up(As), *dW = up(W), *dWs = up(Ws);
    constexpr int EB_ = sizeof(typename C::ElementD);
    const int OF = EB_ == 4;
    size_t outb = (size_t)M * N * EB_;
    uint8_t *dRef, *dOurs, *dCut;
    CK(cudaMalloc(&dRef, outb)); CK(cudaMalloc(&dOurs, outb)); CK(cudaMalloc(&dCut, outb));
    for (int g = 0; g < G; ++g) {
        int rc = memra_v41_gemm_f8_ref(dA + (size_t)g * R * K, K, dAs + (size_t)g * R * KB, KB, dW + (size_t)g * N * wrow,
                                       wrow, dWs + (size_t)g * N * KB, KB, 1.f, 1, dRef + (size_t)g * R * N * EB_, N, OF, R, N, K, 0);
        if (rc) { printf("ref rc %d\n", rc); exit(3); }
    }
    std::vector<V41Tile> tl; std::vector<V41ExpW> ew(G);
    // the engine's plan for these widths: tiles of at most 8 rows (V41_SM_MAX), several a group past 8 rows
    int bmd = R < 8 ? R : 8;
    for (int g = 0; g < G; ++g) {
        for (int m0 = 0; m0 < R; m0 += bmd) tl.push_back({g * R, R, m0, g});
        for (int p = 0; p < 3; ++p) { ew[g].w[p] = dW + (size_t)g * N * wrow; ew[g].s[p] = dWs + (size_t)g * N * KB; ew[g].scale2[p] = 1.f; }
        ew[g].pad = 0;
    }
    int nt = (int)tl.size();
    V41Tile* dT = up(tl); V41ExpW* dE = up(ew); int* dN = up(std::vector<int>{nt});
    auto ours = [&]() {
        int rc = memra_v41_gemm_f8_tiles(dA, K, dAs, KB, dT, dN, nt, dE, 0, wrow, KB, 1, dOurs, N, OF, N, K, bmd, 0);
        if (rc) { printf("ours rc %d\n", rc); exit(3); }
    };
    using G_ = typename C::Gemm;
    using PS = typename C::ProblemShape::UnderlyingProblemShape;
    std::vector<PS> ps; std::vector<typename C::StrideA> sa; std::vector<typename C::StrideB> sb; std::vector<typename C::StrideD> sd;
    std::vector<typename C::LayoutSFA> lsa; std::vector<typename C::LayoutSFB> lsb;
    std::vector<const void*> pa, pb, psa, psb; std::vector<void*> pd; std::vector<uint8_t*> keep;
    using SFT = cutlass::float_ue8m0_t;
    for (int g = 0; g < G; ++g) {
        ps.push_back({N, R, K});  // M = output columns (weight rows), N = the call's rows
        sa.push_back(cutlass::make_cute_packed_stride(typename C::StrideA{}, {N, K, 1}));
        sb.push_back(cutlass::make_cute_packed_stride(typename C::StrideB{}, {R, K, 1}));
        sd.push_back(cutlass::make_cute_packed_stride(typename C::StrideD{}, {N, R, 1}));
        auto la = C::SfCfg::tile_atom_to_shape_SFA(make_shape(N, R, K, 1));
        auto lb = C::SfCfg::tile_atom_to_shape_SFB(make_shape(N, R, K, 1));
        lsa.push_back(la); lsb.push_back(lb);
        std::vector<uint8_t> hsa(size(filter_zeros(la)), 0), hsb(size(filter_zeros(lb)), 0);
        for (int n = 0; n < N; ++n) for (int kb = 0; kb < KB; ++kb) hsa[la(n, kb * 32, 0)] = Ws[((size_t)g * N + n) * KB + kb];
        for (int r = 0; r < R; ++r) for (int kb = 0; kb < KB; ++kb) hsb[lb(r, kb * 32, 0)] = As[((size_t)g * R + r) * KB + kb];
        uint8_t *x = up(hsa), *y = up(hsb); keep.push_back(x); keep.push_back(y);
        pa.push_back(dW + (size_t)g * N * wrow); pb.push_back(dA + (size_t)g * R * K);
        psa.push_back(x); psb.push_back(y); pd.push_back(dCut + (size_t)g * R * N * EB_);
    }
    auto dps = up(ps); auto dsa = up(sa); auto dsb = up(sb); auto dsd = up(sd); auto dlsa = up(lsa); auto dlsb = up(lsb);
    auto dpa = up(pa); auto dpb = up(pb); auto dpsa = up(psa); auto dpsb = up(psb); auto dpd = up(pd);
    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename G_::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                {G, dps, ps.data()},
                                {(const typename C::Main::ElementA**)dpa, dsa, (const typename C::Main::ElementB**)dpb, dsb,
                                 (const SFT**)dpsa, dlsa, (const SFT**)dpsb, dlsb},
                                {{}, nullptr, dsd, (typename C::ElementD**)dpd, dsd},
                                hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    G_ gemm;
    size_t wsz = G_::get_workspace_size(args);
    void* ws = nullptr; if (wsz) CK(cudaMalloc(&ws, wsz));
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("%s: CUTLASS cannot implement\n", name); return; }
    if (gemm.initialize(args, ws) != cutlass::Status::kSuccess) { printf("%s: CUTLASS initialize failed\n", name); return; }
    auto cut = [&]() { if (gemm.run() != cutlass::Status::kSuccess) { printf("CUTLASS run failed\n"); exit(4); } };
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto f) {
        for (int i = 0; i < 3; ++i) f();
        CK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (int i = 0; i < reps; ++i) { CK(cudaEventRecord(e0)); f(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms); }
        std::sort(t.begin(), t.end()); return t[t.size() / 2];
    };
    float tc = timeit(cut), to = timeit(ours);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<uint8_t> hr8(outb), ho8(outb), hc8(outb);
    CK(cudaMemcpy(hr8.data(), dRef, outb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho8.data(), dOurs, outb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc8.data(), dCut, outb, cudaMemcpyDeviceToHost));
    size_t dco = 0, dcr = 0, ne = outb / EB_;
    auto at = [&](std::vector<uint8_t>& v, size_t i) -> uint32_t { return EB_ == 4 ? ((uint32_t*)v.data())[i] : ((uint16_t*)v.data())[i]; };
    for (size_t i = 0; i < ne; ++i) { dco += at(ho8, i) != at(hr8, i); dcr += at(hc8, i) != at(hr8, i); }
    if (EB_ == 4 && !g_last.empty()) {  // against the normal configuration's fp32 output on the same inputs
        size_t dx = 0; for (size_t i = 0; i < ne && i < g_last.size(); ++i) dx += at(hc8, i) != g_last[i];
        printf("  CUTLASS swapped (this) vs CUTLASS normal (the run before), fp32 accumulators: %zu of %zu differ\n", dx, ne);
    }
    double bytes = (double)G * N * wrow;
    printf("%s G %d R %d N %d K %d: CUTLASS %.4f ms (%.0f GB/s weights), memra %.4f ms (%.0f GB/s), CUTLASS/memra %.3f | "
           "outputs differing from the reference tile: memra %zu, CUTLASS %zu of %zu%s\n",
           name, G, R, N, K, tc, bytes / tc / 1e6, to, bytes / to / 1e6, tc / to, dco, dcr, ne, EB_ == 4 ? " (fp32)" : "");
    for (auto p : keep) cudaFree(p);
    cudaFree(dA); cudaFree(dAs); cudaFree(dW); cudaFree(dWs); cudaFree(dRef); cudaFree(dOurs); cudaFree(dCut);
    if (ws) cudaFree(ws);
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device %s sm_%d%d, CUTLASS %d.%d.%d\n", p.name, p.major, p.minor, CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
    if (argc > 1 && !strcmp(argv[1], "cmp")) {
        // tile == small-row inside CUTLASS: the normal configuration (rows on A, 128 x 128) and the swapped small-N one
        // on the same inputs, fp32 out, for 1, 2, 8, 16, 32 and 64 rows an expert
        for (int r : {16, 32, 64}) {
            g_last.clear();
            run<CfgF4f>("normal fp32", 6, r, 2304, 5120, 1, 3);
            run_dec<CfgDec<_8, float>>("swapped tileN 8 fp32", 6, r, 2304, 5120, 3);
            run<CfgF4f>("normal fp32", 6, r, 2304, 5120, 1, 3);
            run_dec<CfgDec<_16, float>>("swapped tileN 16 fp32", 6, r, 2304, 5120, 3);
        }
        printf("PROBE-DONE\n");
        return 0;
    }
    std::vector<std::pair<int, int>> er;
    for (int i = 1; i + 1 < argc; i += 2) er.push_back({atoi(argv[i]), atoi(argv[i + 1])});
    if (er.empty()) er = {{96, 64}, {96, 512}};
    for (auto [e, r] : er) run<CfgF4>("w13 mxfp4", e, r, 2304, 5120, 1, 20);
    // the fp32 accumulator itself, both sides (the 512-row case had one bf16 output apart)
    for (auto [e, r] : er) run<CfgF4f>("w13 mxfp4", e, r, 2304, 5120, 1, 10);
    run<CfgF8>("wq_a fp8", 1, 4096, 1280, 5120, 0, 20);
    // decode widths (the engine's plan at one token: a tile an expert, its rows)
    for (int r : {1, 2, 8}) {
        run_dec<CfgDec<_8>>("decode w13 tileN 8", 6, r, 2304, 5120, 20);
        run_dec<CfgDec<_16>>("decode w13 tileN 16", 6, r, 2304, 5120, 20);
    }
    run_dec<CfgDec<_8>>("decode w13 tileN 8", 48, 1, 2304, 5120, 20);
    printf("PROBE-DONE\n");
    return 0;
}
