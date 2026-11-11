// Probe (scratch, not memra) for the CUTLASS integration's first risks, CUTLASS 4.8.1 sm_120 block-scaled kernels:
//  T1 grouped ptr-array GEMM (MXFP4 experts, e4m3 activations, ue8m0 at 32): arguments written ON THE CARD by one
//     setup kernel from an expert-offset array (problem shapes, pointer arrays, strides, scale layouts, each pair's
//     scale row), activation scales written in the closed-form layout from that row map; host problem shapes nullptr;
//     groups with zero rows. Checked against host-built arguments (the rev-6 probe's way) byte for byte, for the
//     normal (rows on A, 128x128) and swapped (tileN 8, 16) forms, and the forms against each other.
//  T2 CUDA-graph capture of [setup kernel + grouped launch] and replays after the offsets change (new routing, same
//     group count): each replay == an eager run of that routing.
//  T3 the non-grouped (plain) kernel for dense FP8 (32x32 block weight scales expanded per row): plain normal ==
//     plain swapped tileN 8/16 == grouped normal (G = 1), 1..64 and 4096 rows; plain under graph capture == eager.
//  T4 timing: the setup kernel, a grouped launch eager and replayed.
// usage: graph_probe            (prints PROBE-PASS / PROBE-FAIL lines, PROBE-DONE)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
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

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);          \
            exit(2);                                                                           \
        }                                                                                      \
    } while (0)

using F8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using F4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using SFT = cutlass::float_ue8m0_t;
using BF = cutlass::bfloat16_t;

// grouped: EA on A (RowMajor), EB on B (ColumnMajor), D RowMajor (normal) or ColumnMajor (swapped)
template <class EA, class EB, int AlA, int AlB, class TileN, class LayD>
struct GCfg {
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<_128, TileN, _128>;
    using Cl = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, Op, Tile, Cl, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayD*, 8, BF, LayD*, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, EA, cutlass::layout::RowMajor*, AlA, EB, cutlass::layout::ColumnMajor*, AlB, float, Tile, Cl,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using PS = typename ProblemShape::UnderlyingProblemShape;
    using SA = typename Kernel::InternalStrideA;
    using SB = typename Kernel::InternalStrideB;
    using SD = typename Kernel::InternalStrideD;
    using LSA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
    using LSB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
    using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using EAe = typename Main::ElementA;
    using EBe = typename Main::ElementB;
    static constexpr bool kSwap = std::is_same_v<LayD, cutlass::layout::ColumnMajor>;
};
// plain (one problem)
template <class EA, class EB, int AlA, int AlB, class TileN, class LayD>
struct PCfg {
    using ProblemShape = Shape<int, int, int, int>;
    using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<_128, TileN, _128>;
    using Cl = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, Op, Tile, Cl, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayD, 8, BF, LayD, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, EA, cutlass::layout::RowMajor, AlA, EB, cutlass::layout::ColumnMajor, AlB, float, Tile, Cl,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using SA = typename Kernel::StrideA;
    using SB = typename Kernel::StrideB;
    using SD = typename Kernel::StrideD;
    using LSA = typename Kernel::CollectiveMainloop::LayoutSFA;
    using LSB = typename Kernel::CollectiveMainloop::LayoutSFB;
    using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using EAe = typename Main::ElementA;
    using EBe = typename Main::ElementB;
    static constexpr bool kSwap = std::is_same_v<LayD, cutlass::layout::ColumnMajor>;
};

using GN = GCfg<F8, F4, 16, 128, _128, cutlass::layout::RowMajor>;      // experts, rows on A
using GS8 = GCfg<F4, F8, 128, 16, _8, cutlass::layout::ColumnMajor>;    // experts, swapped tileN 8
using GS16 = GCfg<F4, F8, 128, 16, _16, cutlass::layout::ColumnMajor>;  // experts, swapped tileN 16
using GN8 = GCfg<F8, F8, 16, 16, _128, cutlass::layout::RowMajor>;      // dense as one group
using PN8 = PCfg<F8, F8, 16, 16, _128, cutlass::layout::RowMajor>;      // dense plain, rows on A
using PS8 = PCfg<F8, F8, 16, 16, _8, cutlass::layout::ColumnMajor>;     // dense plain swapped tileN 8
using PS16 = PCfg<F8, F8, 16, 16, _16, cutlass::layout::ColumnMajor>;   // dense plain swapped tileN 16

// the closed-form scale layout (sf_check.cu proves it equals tile_atom_to_shape_SF{A,B})
__host__ __device__ inline long sf_off(long r, long kb, long KB) {
    return (r / 128) * (((KB + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}
static long sf_bytes(long rows, long KB) { return (rows + 127) / 128 * 128 * ((KB + 3) / 4) * 4; }

static uint64_t rs = 0xd5415c4e;
static uint32_t rnd() { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(rs >> 33); }
static void fill_e4m3(std::vector<uint8_t>& v) { for (auto& b : v) { b = rnd() & 0xFF; if ((b & 0x7F) == 0x7F) b ^= 1; } }
static void fill_range(std::vector<uint8_t>& v, int lo, int hi) { for (auto& b : v) b = lo + rnd() % (hi - lo + 1); }
template <class T> static T* up(const std::vector<T>& h) { T* d; CK(cudaMalloc(&d, h.size() * sizeof(T) + 256)); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <class T> static T* dalloc(size_t n) { T* d; CK(cudaMalloc(&d, n * sizeof(T) + 256)); return d; }
static int g_fail = 0;
static void verdict(const char* what, size_t diff, size_t n) {
    printf("%s %s: %zu of %zu differ\n", diff ? "PROBE-FAIL" : "PROBE-PASS", what, diff, n);
    if (diff) g_fail = 1;
}
static size_t ndiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i]; return d;
}

// ---- T1/T2: the grouped GEMM with arguments built on the card
struct Exp {  // the expert table: weight and scale pointers per group (scales already in the closed form)
    const uint8_t* w;
    const uint8_t* s;
};
// one block: group g's rows are [off[g], off[g+1]); its scale rows start at a multiple of 128 (the padded prefix)
template <class C>
__global__ void setup_kernel(const int* __restrict__ off, int G, int N, int K, const uint8_t* A, const uint8_t* asf,
                             const Exp* __restrict__ ex, BF* D, long ldc, typename C::PS* ps, const typename C::EAe** pa,
                             const typename C::EBe** pb, const SFT** psa, const SFT** psb, BF** pd, typename C::SA* sa,
                             typename C::SB* sb, typename C::SD* sd, typename C::LSA* lsa, typename C::LSB* lsb,
                             int* sfrow) {
    __shared__ int pad[1025];
    int t = threadIdx.x, KB = K / 32;
    long kbt4 = ((KB + 3) / 4) * 4;
    if (t == 0) {
        int p = 0;
        for (int g = 0; g < G; ++g) { pad[g] = p; p += (off[g + 1] - off[g] + 127) / 128 * 128; }
        pad[G] = p;
    }
    __syncthreads();
    for (int g = t; g < G; g += blockDim.x) {
        int M = off[g + 1] - off[g];
        const uint8_t* act = A + (long)off[g] * K;
        const uint8_t* actsf = asf + (long)pad[g] * kbt4;
        pd[g] = D + (long)off[g] * ldc;
        if constexpr (!C::kSwap) {
            ps[g] = typename C::PS{M, N, K};
            pa[g] = (const typename C::EAe*)act;
            pb[g] = (const typename C::EBe*)ex[g].w;
            psa[g] = (const SFT*)actsf;
            psb[g] = (const SFT*)ex[g].s;
            sa[g] = cutlass::make_cute_packed_stride(typename C::SA{}, {M, K, 1});
            sb[g] = cutlass::make_cute_packed_stride(typename C::SB{}, {N, K, 1});
            sd[g] = cutlass::make_cute_packed_stride(typename C::SD{}, {M, (int)ldc, 1});
            lsa[g] = C::SfCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
            lsb[g] = C::SfCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        } else {
            ps[g] = typename C::PS{N, M, K};
            pa[g] = (const typename C::EAe*)ex[g].w;
            pb[g] = (const typename C::EBe*)act;
            psa[g] = (const SFT*)ex[g].s;
            psb[g] = (const SFT*)actsf;
            sa[g] = cutlass::make_cute_packed_stride(typename C::SA{}, {N, K, 1});
            sb[g] = cutlass::make_cute_packed_stride(typename C::SB{}, {M, K, 1});
            sd[g] = cutlass::make_cute_packed_stride(typename C::SD{}, {(int)ldc, M, 1});
            lsa[g] = C::SfCfg::tile_atom_to_shape_SFA(make_shape(N, M, K, 1));
            lsb[g] = C::SfCfg::tile_atom_to_shape_SFB(make_shape(N, M, K, 1));
        }
    }
    // each pair's scale row (what act_quant reads to place a row's scales)
    for (int g = 0; g < G; ++g)
        for (int p = off[g] + t; p < off[g + 1]; p += blockDim.x) sfrow[p] = pad[g] + (p - off[g]);
}
// activation scales [rows][KB] row-major -> the closed form at each row's sfrow (act_quant's job in the engine)
__global__ void place_sf(const uint8_t* src, int rows, int KB, const int* sfrow, uint8_t* dst) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)rows * KB) return;
    int r = i / KB, kb = i % KB;
    dst[sf_off(sfrow[r], kb, KB)] = src[i];
}

template <class C>
struct Grouped {
    int G, N, K;
    typename C::PS* ps; const typename C::EAe** pa; const typename C::EBe** pb; const SFT** psa; const SFT** psb;
    BF** pd; typename C::SA* sa; typename C::SB* sb; typename C::SD* sd; typename C::LSA* lsa; typename C::LSB* lsb;
    typename C::Gemm gemm; void* ws = nullptr;
    void alloc(int G_, int N_, int K_, cudaStream_t st) {
        G = G_; N = N_; K = K_;
        ps = dalloc<typename C::PS>(G); pa = dalloc<const typename C::EAe*>(G); pb = dalloc<const typename C::EBe*>(G);
        psa = dalloc<const SFT*>(G); psb = dalloc<const SFT*>(G); pd = dalloc<BF*>(G); sa = dalloc<typename C::SA>(G);
        sb = dalloc<typename C::SB>(G); sd = dalloc<typename C::SD>(G); lsa = dalloc<typename C::LSA>(G); lsb = dalloc<typename C::LSB>(G);
        cutlass::KernelHardwareInfo hw; hw.device_id = 0;
        hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
        typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                         {G, ps, nullptr},
                                         {pa, sa, pb, sb, psa, lsa, psb, lsb},
                                         {{}, nullptr, sd, pd, sd},
                                         hw};
        args.epilogue.thread.alpha = 1.0f;
        args.epilogue.thread.beta = 0.0f;
        size_t wsz = C::Gemm::get_workspace_size(args);
        if (wsz) CK(cudaMalloc(&ws, wsz));
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("grouped: cannot implement\n"); exit(4); }
        if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) { printf("grouped: initialize failed\n"); exit(4); }
        printf("grouped %s G %d: workspace %zu bytes\n", C::kSwap ? "swapped" : "normal", G, wsz);
    }
    void setup(const int* off, const uint8_t* A, const uint8_t* asf, const Exp* ex, BF* D, long ldc, int* sfrow, cudaStream_t st) {
        setup_kernel<C><<<1, 1024, 0, st>>>(off, G, N, K, A, asf, ex, D, ldc, ps, pa, pb, psa, psb, pd, sa, sb, sd, lsa, lsb, sfrow);
    }
    void run(cudaStream_t st) { if (gemm.run(st) != cutlass::Status::kSuccess) { printf("grouped run failed\n"); exit(4); } }
};

// host-built arguments (the proven rev-6 way: host repack through CUTLASS's own layout functions)
template <class C>
static std::vector<uint16_t> grouped_host(const std::vector<int>& off, int N, int K, const uint8_t* dA,
                                          const std::vector<uint8_t>& As, const std::vector<const uint8_t*>& dW,
                                          const std::vector<std::vector<uint8_t>>& Ws, cudaStream_t st) {
    int G = (int)off.size() - 1, KB = K / 32, M = off[G];
    BF* dD = dalloc<BF>((size_t)M * N);
    CK(cudaMemset(dD, 0xAB, (size_t)M * N * 2));
    std::vector<typename C::PS> ps; std::vector<typename C::SA> sa; std::vector<typename C::SB> sb; std::vector<typename C::SD> sd;
    std::vector<typename C::LSA> lsa; std::vector<typename C::LSB> lsb;
    std::vector<const void*> pa, pb, psa, psb; std::vector<void*> pd; std::vector<uint8_t*> keep;
    for (int g = 0; g < G; ++g) {
        int R = off[g + 1] - off[g];
        int Mx = C::kSwap ? N : R, Nx = C::kSwap ? R : N;
        ps.push_back({Mx, Nx, K});
        sa.push_back(cutlass::make_cute_packed_stride(typename C::SA{}, {Mx, K, 1}));
        sb.push_back(cutlass::make_cute_packed_stride(typename C::SB{}, {Nx, K, 1}));
        sd.push_back(cutlass::make_cute_packed_stride(typename C::SD{}, {Mx, Nx, 1}));
        auto la = C::SfCfg::tile_atom_to_shape_SFA(make_shape(Mx, Nx, K, 1));
        auto lb = C::SfCfg::tile_atom_to_shape_SFB(make_shape(Mx, Nx, K, 1));
        lsa.push_back(la); lsb.push_back(lb);
        std::vector<uint8_t> act(std::max<long>(1, size(filter_zeros(C::kSwap ? lb : la))), 0), wsf(size(filter_zeros(C::kSwap ? la : lb)), 0);
        for (int r = 0; r < R; ++r) for (int kb = 0; kb < KB; ++kb) {
            uint8_t v = As[(size_t)(off[g] + r) * KB + kb];
            if (C::kSwap) act[lb(r, kb * 32, 0)] = v; else act[la(r, kb * 32, 0)] = v;
        }
        for (int n = 0; n < N; ++n) for (int kb = 0; kb < KB; ++kb) {
            uint8_t v = Ws[g][(size_t)n * KB + kb];
            if (C::kSwap) wsf[la(n, kb * 32, 0)] = v; else wsf[lb(n, kb * 32, 0)] = v;
        }
        uint8_t *x = up(act), *y = up(wsf); keep.push_back(x); keep.push_back(y);
        const uint8_t* actp = dA + (size_t)off[g] * K;
        pa.push_back(C::kSwap ? (const void*)dW[g] : (const void*)actp);
        pb.push_back(C::kSwap ? (const void*)actp : (const void*)dW[g]);
        psa.push_back(C::kSwap ? y : x); psb.push_back(C::kSwap ? x : y);
        pd.push_back(dD + (size_t)off[g] * N);
    }
    auto dps = up(ps); auto dsa = up(sa); auto dsb = up(sb); auto dsd = up(sd); auto dlsa = up(lsa); auto dlsb = up(lsb);
    auto dpa = up(pa); auto dpb = up(pb); auto dpsa = up(psa); auto dpsb = up(psb); auto dpd = up(pd);
    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped, {G, dps, ps.data()},
                                     {(const typename C::EAe**)dpa, dsa, (const typename C::EBe**)dpb, dsb, (const SFT**)dpsa, dlsa, (const SFT**)dpsb, dlsb},
                                     {{}, nullptr, dsd, (BF**)dpd, dsd}, hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    typename C::Gemm gemm;
    void* ws = nullptr; size_t wsz = C::Gemm::get_workspace_size(args); if (wsz) CK(cudaMalloc(&ws, wsz));
    if (gemm.can_implement(args) != cutlass::Status::kSuccess || gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) { printf("host grouped init failed\n"); exit(4); }
    if (gemm.run(st) != cutlass::Status::kSuccess) { printf("host grouped run failed\n"); exit(4); }
    CK(cudaStreamSynchronize(st));
    std::vector<uint16_t> h((size_t)M * N);
    CK(cudaMemcpy(h.data(), dD, h.size() * 2, cudaMemcpyDeviceToHost));
    for (auto p : keep) cudaFree(p);
    cudaFree(dD); if (ws) cudaFree(ws);
    cudaFree(dps); cudaFree(dsa); cudaFree(dsb); cudaFree(dsd); cudaFree(dlsa); cudaFree(dlsb);
    cudaFree(dpa); cudaFree(dpb); cudaFree(dpsa); cudaFree(dpsb); cudaFree(dpd);
    return h;
}

static std::vector<int> draw_off(int G, int cap) {
    static const int sizes[] = {0, 0, 0, 1, 2, 3, 7, 8, 9, 16, 33, 64, 129};
    std::vector<int> off(G + 1, 0);
    for (int g = 0; g < G; ++g) {
        int r = sizes[rnd() % 13];
        if (off[g] + r > cap) r = 0;
        off[g + 1] = off[g] + r;
    }
    return off;
}

template <class C>
static void t1_t2(const char* name, int G, int N, int K, int cap, const uint8_t* dA, const uint8_t* dAs,
                  const std::vector<uint8_t>& As, const std::vector<const uint8_t*>& dW, const std::vector<std::vector<uint8_t>>& Ws,
                  const Exp* dEx, std::vector<std::vector<uint16_t>>* keep_out) {
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    int KB = K / 32;
    BF* dD = dalloc<BF>((size_t)cap * N);
    uint8_t* dsf = dalloc<uint8_t>(sf_bytes(cap + G * 127L, KB));
    int* dsfrow = dalloc<int>(cap);
    int* doff = dalloc<int>(G + 1);
    Grouped<C> gr; gr.alloc(G, N, K, st);
    auto eager = [&](const std::vector<int>& off) {
        CK(cudaMemcpyAsync(doff, off.data(), (G + 1) * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemsetAsync(dD, 0xAB, (size_t)cap * N * 2, st));
        gr.setup(doff, dA, dsf, dEx, dD, N, dsfrow, st);
        int M = off[G];
        if (M) place_sf<<<((long)M * KB + 255) / 256, 256, 0, st>>>(dAs, M, KB, dsfrow, dsf);
        gr.run(st);
        CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
        std::vector<uint16_t> h((size_t)M * N);
        CK(cudaMemcpy(h.data(), dD, h.size() * 2, cudaMemcpyDeviceToHost));
        return h;
    };
    // T1: device-built arguments == host-built arguments, three routings
    for (int it = 0; it < 3; ++it) {
        auto off = draw_off(G, cap);
        auto hd = eager(off);
        auto hh = grouped_host<C>(off, N, K, dA, As, dW, Ws, st);
        int empty = 0; for (int g = 0; g < G; ++g) empty += off[g + 1] == off[g];
        char what[256]; snprintf(what, sizeof what, "T1 %s routing %d (%d rows, %d of %d groups empty): device-built vs host-built", name, it, off[G], empty, G);
        verdict(what, ndiff(hd, hh), hd.size());
        if (it == 0 && keep_out) keep_out->push_back(hd);
    }
    // T2: capture [offsets copy is outside] setup + scale placement + grouped launch; replay over new routings
    {
        // the scale placement in the graph covers the first routing's total rows (a fixed launch); every replay's
        // routing has that same total
        cudaGraph_t graph; cudaGraphExec_t exec;
        auto off0 = draw_off(G, cap);
        CK(cudaMemcpyAsync(doff, off0.data(), (G + 1) * 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        gr.setup(doff, dA, dsf, dEx, dD, N, dsfrow, st);
        place_sf<<<((long)off0[G] * KB + 255) / 256, 256, 0, st>>>(dAs, off0[G], KB, dsfrow, dsf);
        gr.run(st);
        cudaError_t ce = cudaStreamEndCapture(st, &graph);
        if (ce != cudaSuccess) { printf("PROBE-FAIL T2 %s: capture failed: %s\n", name, cudaGetErrorString(ce)); g_fail = 1; cudaGetLastError(); return; }
        CK(cudaGraphInstantiate(&exec, graph, 0));
        for (int it = 0; it < 3; ++it) {
            // a new routing with the SAME total rows (the captured placement launch covers off0[G] rows)
            std::vector<int> off;
            do { off = draw_off(G, cap); } while (off[G] < off0[G]);
            // trim the last groups so the total equals off0[G]
            for (int g = G - 1; g >= 0 && off[G] > off0[G]; --g) {
                int extra = off[G] - off0[G], r = off[g + 1] - off[g], cut = std::min(extra, r);
                for (int h = g + 1; h <= G; ++h) off[h] -= cut;
            }
            auto he = eager(off);
            CK(cudaMemcpyAsync(doff, off.data(), (G + 1) * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemsetAsync(dD, 0xAB, (size_t)cap * N * 2, st));
            CK(cudaGraphLaunch(exec, st));
            CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
            std::vector<uint16_t> hg((size_t)off[G] * N);
            CK(cudaMemcpy(hg.data(), dD, hg.size() * 2, cudaMemcpyDeviceToHost));
            char what[256]; snprintf(what, sizeof what, "T2 %s replay %d (%d rows): graph replay vs eager", name, it, off[G]);
            verdict(what, ndiff(hg, he), he.size());
        }
        // T4 timing: setup kernel alone, eager [setup + place + gemm], graph replay
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        auto med = [&](auto f, int reps) {
            for (int i = 0; i < 5; ++i) f();
            CK(cudaStreamSynchronize(st));
            std::vector<float> t;
            for (int i = 0; i < reps; ++i) { CK(cudaEventRecord(e0, st)); f(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms); }
            std::sort(t.begin(), t.end()); return t[t.size() / 2] * 1000.f;
        };
        float tset = med([&] { gr.setup(doff, dA, dsf, dEx, dD, N, dsfrow, st); }, 50);
        float tgemm = med([&] { gr.run(st); }, 20);
        float tgraph = med([&] { CK(cudaGraphLaunch(exec, st)); }, 20);
        printf("T4 %s G %d rows %d: setup kernel %.1f us, grouped GEMM %.1f us, graph [setup+place+GEMM] %.1f us\n", name, G, off0[G], tset, tgemm, tgraph);
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    cudaFree(dD); cudaFree(dsf); cudaFree(dsfrow); cudaFree(doff);
    cudaStreamDestroy(st);
}

// ---- T3: plain kernels for dense FP8 (32x32 block weight codes expanded per row)
template <class C>
static std::vector<uint16_t> plain(int M, int N, int K, const uint8_t* dA, const uint8_t* dAsf, const uint8_t* dW,
                                   const uint8_t* dWsf, cudaStream_t st, bool graph) {
    constexpr bool swap = C::kSwap;
    int Mx = swap ? N : M, Nx = swap ? M : N;
    BF* dD = dalloc<BF>((size_t)M * N);
    CK(cudaMemset(dD, 0xAB, (size_t)M * N * 2));
    typename C::SA sa = cutlass::make_cute_packed_stride(typename C::SA{}, {Mx, K, 1});
    typename C::SB sb = cutlass::make_cute_packed_stride(typename C::SB{}, {Nx, K, 1});
    typename C::SD sd = cutlass::make_cute_packed_stride(typename C::SD{}, {Mx, Nx, 1});
    auto la = C::SfCfg::tile_atom_to_shape_SFA(make_shape(Mx, Nx, K, 1));
    auto lb = C::SfCfg::tile_atom_to_shape_SFB(make_shape(Mx, Nx, K, 1));
    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {Mx, Nx, K, 1},
                                     {(const typename C::EAe*)(swap ? dW : dA), sa, (const typename C::EBe*)(swap ? dA : dW), sb,
                                      (const SFT*)(swap ? dWsf : dAsf), la, (const SFT*)(swap ? dAsf : dWsf), lb},
                                     {{}, nullptr, sd, dD, sd}, hw};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    typename C::Gemm gemm;
    size_t wsz = C::Gemm::get_workspace_size(args);
    void* ws = nullptr; if (wsz) CK(cudaMalloc(&ws, wsz));
    if (gemm.can_implement(args) != cutlass::Status::kSuccess || gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) { printf("plain init failed\n"); exit(4); }
    if (graph) {
        cudaGraph_t g; cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        if (gemm.run(st) != cutlass::Status::kSuccess) { printf("plain run (capture) failed\n"); exit(4); }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ex, g, 0));
        CK(cudaGraphLaunch(ex, st));
        CK(cudaStreamSynchronize(st));
        cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
    } else if (gemm.run(st) != cutlass::Status::kSuccess) { printf("plain run failed\n"); exit(4); }
    CK(cudaStreamSynchronize(st)); CK(cudaGetLastError());
    std::vector<uint16_t> h((size_t)M * N);
    CK(cudaMemcpy(h.data(), dD, h.size() * 2, cudaMemcpyDeviceToHost));
    cudaFree(dD); if (ws) cudaFree(ws);
    if (!graph) printf("  plain %s M %d: workspace %zu\n", swap ? "swapped" : "normal", M, wsz);
    return h;
}

// T5: concurrency. S streams, each with its own argument block, workspace and output, launch [setup + grouped GEMM]
// (and the plain dense forms) R times back to back with no sync between streams; every output is compared with a
// single-stream reference. A difference means the launch is not safe to co-schedule with itself.
template <class C>
static void t5(const char* name, int G, int N, int K, int cap, const uint8_t* dA, const uint8_t* dAs, const Exp* dEx,
               int S, int R) {
    int KB = K / 32;
    auto off = draw_off(G, cap);
    int M = off[G];
    std::vector<cudaStream_t> st(S);
    for (auto& x : st) CK(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking));
    std::vector<Grouped<C>> gr(S);
    std::vector<BF*> dD(S);
    std::vector<uint8_t*> dsf(S);
    std::vector<int*> dsfrow(S), doff(S);
    for (int i = 0; i < S; ++i) {
        gr[i].alloc(G, N, K, st[i]);
        dD[i] = dalloc<BF>((size_t)cap * N);
        dsf[i] = dalloc<uint8_t>(sf_bytes(cap + G * 127L, KB));
        dsfrow[i] = dalloc<int>(cap);
        doff[i] = dalloc<int>(G + 1);
        CK(cudaMemcpy(doff[i], off.data(), (G + 1) * 4, cudaMemcpyHostToDevice));
    }
    auto launch = [&](int i) {
        gr[i].setup(doff[i], dA, dsf[i], dEx, dD[i], N, dsfrow[i], st[i]);
        place_sf<<<((long)M * KB + 255) / 256, 256, 0, st[i]>>>(dAs, M, KB, dsfrow[i], dsf[i]);
        gr[i].run(st[i]);
    };
    launch(0);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> ref((size_t)M * N), got((size_t)M * N);
    CK(cudaMemcpy(ref.data(), dD[0], ref.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad_runs = 0, bad_elems = 0;
    for (int r = 0; r < R; ++r) {
        for (int i = 0; i < S; ++i) CK(cudaMemsetAsync(dD[i], 0xAB, (size_t)cap * N * 2, st[i]));
        for (int i = 0; i < S; ++i) launch(i);
        CK(cudaDeviceSynchronize());
        for (int i = 0; i < S; ++i) {
            CK(cudaMemcpy(got.data(), dD[i], got.size() * 2, cudaMemcpyDeviceToHost));
            size_t d = ndiff(got, ref);
            if (d) { ++bad_runs; bad_elems += d; if (bad_runs <= 3) printf("  %s run %d stream %d: %zu of %zu differ\n", name, r, i, d, got.size()); }
        }
    }
    char what[200];
    snprintf(what, sizeof what, "T5 %s: %d streams x %d rounds (%d rows, G %d), outputs differing from the single-stream run", name, S, R, M, G);
    verdict(what, bad_runs, (size_t)S * R);
    for (int i = 0; i < S; ++i) { cudaFree(dD[i]); cudaFree(dsf[i]); cudaFree(dsfrow[i]); cudaFree(doff[i]); cudaStreamDestroy(st[i]); }
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device %s sm_%d%d, %d SMs, CUTLASS %d.%d.%d\n", p.name, p.major, p.minor, p.multiProcessorCount, CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
    int R = argc > 1 ? atoi(argv[1]) : 50;
    const int G = 96, N = 2304, K = 5120, KB = K / 32, cap = 256;
    std::vector<uint8_t> A((size_t)cap * K), As((size_t)cap * KB);
    fill_e4m3(A); fill_range(As, 118, 130);
    uint8_t *dA = up(A), *dAs = up(As);
    std::vector<Exp> ex(G);
    for (int g = 0; g < G; ++g) {
        std::vector<uint8_t> w((size_t)N * K / 2); for (auto& b : w) b = rnd() & 0xFF;
        std::vector<uint8_t> raw((size_t)N * KB); fill_range(raw, 112, 124);
        std::vector<uint8_t> sfc(sf_bytes(N, KB), 0);
        for (int n = 0; n < N; ++n) for (int kb = 0; kb < KB; ++kb) sfc[sf_off(n, kb, KB)] = raw[(size_t)n * KB + kb];
        ex[g] = {up(w), up(sfc)};
    }
    Exp* dEx = up(ex);
    for (int S : {2, 4}) {
        t5<GS16>("grouped swapped16", G, N, K, cap, dA, dAs, dEx, S, R);
        t5<GS8>("grouped swapped8", G, N, K, cap, dA, dAs, dEx, S, R);
        t5<GN>("grouped normal", G, N, K, cap, dA, dAs, dEx, S, R);
    }
    printf("%s\nPROBE-DONE\n", g_fail ? "PROBE-RESULT FAIL" : "PROBE-RESULT PASS");
    return g_fail;
}
