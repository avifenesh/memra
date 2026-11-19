// The CUTLASS forms of the block-scaled GEMMs, shared like the small-batch skeleton (cu/bs_small.cuh): one
// configuration template over CUTLASS's sm_120 block-scaled kernels (vendored CUTLASS v4.8.1, third_party/cutlass),
// the dense launcher and the device-side setup of a grouped (ptr-array) call. Header-only, included by each TU that
// instantiates it; the TU names its element types and forms. First caller: cu/dsv41_cutlass.cu.
//
// Two forms of one program: normal (rows on A, 128 x 128 x 128 tiles) and swapped (weights on A, so output columns
// are the MMA's M, the call's rows on B, tileN 8 or 16, D written column-major, which is the row-major [rows][n]
// output). The scaling and the fp32 accumulation are inside the tensor core; one accumulator is chained over the K
// tiles in ascending order. Scale factors are in CUTLASS's interleaved layout (Sm1xxBlockScaledConfig<32>, the
// same function for A and B; `bs::sf32_off`). Grouped calls take their arguments on the card: one setup kernel
// writes the problem shapes, pointer arrays, strides and scale layouts from the device MoE plan's expert offsets, so
// nothing is read back to the host and a CUDA graph replays the call with whatever routing the plan holds. Groups
// with no rows have no tiles.
//
// Status codes (`Status`): 0 ok, else the step that refused; the caller maps them to its own codes.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "bs_small.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

namespace bs {
namespace cut {

using cute::_1;
using cute::Shape;

using SFT = cutlass::float_ue8m0_t;
using BF = cutlass::bfloat16_t;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

enum Status { kOk = 0, kWorkspace = 3, kCannotImplement = 4, kInitialize = 5, kRun = 6, kLaunch = 7, kBlockBytes = 8 };

// A kernel's stride and scale-layout types: the ptr-array kernels name them Internal*.
template <class K, class M, bool Grouped>
struct Types;
template <class K, class M>
struct Types<K, M, true> {
    using SA = typename K::InternalStrideA;
    using SB = typename K::InternalStrideB;
    using SD = typename K::InternalStrideD;
    using LSA = typename M::InternalLayoutSFA;
    using LSB = typename M::InternalLayoutSFB;
};
template <class K, class M>
struct Types<K, M, false> {
    using SA = typename K::StrideA;
    using SB = typename K::StrideB;
    using SD = typename K::StrideD;
    using LSA = typename M::LayoutSFA;
    using LSB = typename M::LayoutSFB;
};

// One configuration. EAct: the activations' type (CUTLASS's mx_float* wrapper); EW: the weights'; AlW: the weights'
// alignment in elements; Swap: weights on A; Grouped: the ptr-array grouped kernel. bf16 out, fp32 accumulation.
template <class EAct, class EW, int AlW, class TileN, bool Swap, bool Grouped>
struct Cfg {
    using EA = cute::conditional_t<Swap, EW, EAct>;
    using EB = cute::conditional_t<Swap, EAct, EW>;
    static constexpr int AlA = Swap ? AlW : 16;
    static constexpr int AlB = Swap ? 16 : AlW;
    using LayD = cute::conditional_t<Swap, ColM, RowM>;
    using LA = cute::conditional_t<Grouped, RowM*, RowM>;
    using LB = cute::conditional_t<Grouped, ColM*, ColM>;
    using LD = cute::conditional_t<Grouped, LayD*, LayD>;
    using ProblemShape = cute::conditional_t<Grouped, cutlass::gemm::GroupProblemShape<Shape<int, int, int>>,
                                             Shape<int, int, int, int>>;
    using Sched = cute::conditional_t<Grouped, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative,
                                      cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using Tile = Shape<cute::_128, TileN, cute::_128>;
    using Cl = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, Op, Tile, Cl, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LD, 8, BF, LD, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, EA, LA, AlA, EB, LB, AlB, float, Tile, Cl,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        Sched>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using SA = typename Types<Kernel, Main, Grouped>::SA;
    using SB = typename Types<Kernel, Main, Grouped>::SB;
    using SD = typename Types<Kernel, Main, Grouped>::SD;
    using LSA = typename Types<Kernel, Main, Grouped>::LSA;
    using LSB = typename Types<Kernel, Main, Grouped>::LSB;
    using SfCfg = typename Main::Sm1xxBlkScaledConfig;
    using EAe = typename Main::ElementA;
    using EBe = typename Main::ElementB;
    static constexpr bool kSwap = Swap;
};

static inline cutlass::KernelHardwareInfo hw() {
    cutlass::KernelHardwareInfo h;
    cudaGetDevice(&h.device_id);
    h.sm_count = sms();
    return h;
}

// a row-major (K-contiguous) operand's stride with row pitch `ld` elements
template <class S>
CUTE_HOST_DEVICE S kstride(long ld) {
    S s{};
    cute::get<0>(s) = ld;
    return s;
}
// D's stride: row-major [rows][ldc] (normal) or column-major with column pitch ldc (swapped: the same memory)
template <class C>
CUTE_HOST_DEVICE typename C::SD dstride(long ldc) {
    typename C::SD s{};
    if constexpr (C::kSwap) cute::get<1>(s) = ldc;
    else cute::get<0>(s) = ldc;
    return s;
}

// ---- one dense call: c[m][n] = a . w^T (bf16, row pitch ldc); ldw_el: a weight row's elements
template <class C>
static int plain(const void* a, long lda, const void* asf, const void* w, long ldw_el, const void* wsf, void* c,
                 long ldc, int m, int n, int k, cudaStream_t st) {
    using cute::make_shape;
    int M = C::kSwap ? n : m, N = C::kSwap ? m : n;
    const void* pa = C::kSwap ? w : a;
    const void* pb = C::kSwap ? a : w;
    const void* sa = C::kSwap ? wsf : asf;
    const void* sb = C::kSwap ? asf : wsf;
    long ldA = C::kSwap ? ldw_el : lda, ldB = C::kSwap ? lda : ldw_el;
    typename C::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, k, 1},
        {(const typename C::EAe*)pa, kstride<typename C::SA>(ldA), (const typename C::EBe*)pb,
         kstride<typename C::SB>(ldB), (const SFT*)sa, C::SfCfg::tile_atom_to_shape_SFA(make_shape(M, N, k, 1)),
         (const SFT*)sb, C::SfCfg::tile_atom_to_shape_SFB(make_shape(M, N, k, 1))},
        {{}, nullptr, dstride<C>(ldc), (BF*)c, dstride<C>(ldc)},
        hw()};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    typename C::Gemm gemm;
    if (C::Gemm::get_workspace_size(args) != 0) return kWorkspace;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return kCannotImplement;
    if (gemm.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return kInitialize;
    if (gemm.run(st) != cutlass::Status::kSuccess) return kRun;
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : kLaunch;
}

// ---- grouped calls

__host__ __device__ inline long al256(long x) { return (x + 255) & ~255L; }

// the argument arrays of `groups` groups of configuration C inside one block of device memory
template <class C>
struct Arrays {
    typename C::ProblemShape::UnderlyingProblemShape* ps;
    const typename C::EAe** pa;
    const typename C::EBe** pb;
    const SFT** psa;
    const SFT** psb;
    BF** pd;
    typename C::SA* sa;
    typename C::SB* sb;
    typename C::SD* sd;
    typename C::LSA* lsa;
    typename C::LSB* lsb;
    __host__ __device__ static long bytes(int g) {
        return al256(g * sizeof(*ps)) + 5 * al256(g * 8L) + al256(g * sizeof(*sa)) + al256(g * sizeof(*sb)) +
               al256(g * sizeof(*sd)) + al256(g * sizeof(*lsa)) + al256(g * sizeof(*lsb));
    }
    __host__ __device__ Arrays(uint8_t* base, int g) {
        uint8_t* p = base;
        ps = (decltype(ps))p; p += al256(g * sizeof(*ps));
        pa = (decltype(pa))p; p += al256(g * 8L);
        pb = (decltype(pb))p; p += al256(g * 8L);
        psa = (decltype(psa))p; p += al256(g * 8L);
        psb = (decltype(psb))p; p += al256(g * 8L);
        pd = (decltype(pd))p; p += al256(g * 8L);
        sa = (decltype(sa))p; p += al256(g * sizeof(*sa));
        sb = (decltype(sb))p; p += al256(g * sizeof(*sb));
        sd = (decltype(sd))p; p += al256(g * sizeof(*sd));
        lsa = (decltype(lsa))p; p += al256(g * sizeof(*lsa));
        lsb = (decltype(lsb))p;
    }
};

// The padded prefix of the groups' rows (multiples of 128) in shared memory; G <= 1024.
__device__ inline void pad_prefix(const int* off, int G, int* pad) {
    int t = threadIdx.x;
    int v = t < G ? (off[t + 1] - off[t] + 127) / 128 * 128 : 0;
    pad[t] = v;
    __syncthreads();
    for (int s = 1; s < 1024; s <<= 1) {  // inclusive Hillis-Steele scan
        int x = t >= s ? pad[t - s] : 0;
        __syncthreads();
        pad[t] += x;
        __syncthreads();
    }
    int inc = pad[t];
    __syncthreads();
    pad[t + 1] = inc;
    if (t == 0) pad[0] = 0;
    __syncthreads();
}

// The setup of one grouped call: nproj (1 or 2) projections of every local expert, groups [j * G, (j + 1) * G)
// for projection proj[j] into D base d[j]. a: the pairs' codes [np][k] (pair p at row p); asf: their scales over
// the padded rows (the groups each start their scale rows at a multiple of 128).
template <class C>
__global__ void __launch_bounds__(1024) bs_cutlass_setup_kernel(const int* __restrict__ off, int G, int nproj, int p0,
                                                                int p1, const ExpW* __restrict__ ex, const uint8_t* a,
                                                                const uint8_t* asf, BF* d0, BF* d1, long ldc, int n,
                                                                int k, long ldw_el, uint8_t* blk) {
    using cute::make_shape;
    __shared__ int pad[1025];
    pad_prefix(off, G, pad);
    Arrays<C> ar(blk, G * nproj);
    long kbt4 = (long)((k / 32 + 3) / 4) * 4;
    static_assert(!C::kSwap, "grouped calls take the normal form only");
    for (int i = threadIdx.x; i < G * nproj; i += blockDim.x) {
        int g = i % G, j = i / G;
        int proj = j == 0 ? p0 : p1;
        int rows = off[g + 1] - off[g];
        const uint8_t* act = a + (long)off[g] * k;
        const uint8_t* actsf = asf + (long)pad[g] * kbt4;
        BF* dd = (j == 0 ? d0 : d1) + (long)off[g] * ldc;
        ar.pd[i] = dd;
        ar.sd[i] = dstride<C>(ldc);
        ar.ps[i] = {rows, n, k};
        ar.pa[i] = (const typename C::EAe*)act;
        ar.pb[i] = (const typename C::EBe*)ex[g].w[proj];
        ar.psa[i] = (const SFT*)actsf;
        ar.psb[i] = (const SFT*)ex[g].s[proj];
        ar.sa[i] = kstride<typename C::SA>(k);
        ar.sb[i] = kstride<typename C::SB>(ldw_el);
        ar.lsa[i] = C::SfCfg::tile_atom_to_shape_SFA(make_shape(rows, n, k, 1));
        ar.lsb[i] = C::SfCfg::tile_atom_to_shape_SFB(make_shape(rows, n, k, 1));
    }
}

template <class C>
static long ws_bytes(int groups) {
    typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {groups, nullptr, nullptr},
                                     {},
                                     {},
                                     hw()};
    return (long)C::Gemm::get_workspace_size(args);
}

// Bytes of the argument block of a grouped call of `groups` groups (arrays and CUTLASS's workspace).
template <class C>
static long grouped_bytes(int groups) {
    return al256(Arrays<C>::bytes(groups)) + al256(ws_bytes<C>(groups));
}

// One grouped call over G groups (local experts) from the expert offsets `off` [G + 1] (pairs ordered by group);
// blk: grouped_bytes<C>(G * nproj) bytes of device memory that live until the launch has run.
template <class C>
static int grouped(const int* off, int G, int nproj, int p0, int p1, const void* ex, const void* a, const void* asf,
                   void* d0, void* d1, long ldc, int n, int k, long ldw_el, void* blk, long blk_bytes,
                   cudaStream_t st) {
    int groups = G * nproj;
    long need = grouped_bytes<C>(groups);
    if (blk_bytes < need) return kBlockBytes;
    bs_cutlass_setup_kernel<C><<<1, 1024, 0, st>>>(off, G, nproj, p0, p1, (const ExpW*)ex, (const uint8_t*)a,
                                                   (const uint8_t*)asf, (BF*)d0, (BF*)d1, ldc, n, k, ldw_el,
                                                   (uint8_t*)blk);
    Arrays<C> ar((uint8_t*)blk, groups);
    void* ws = (uint8_t*)blk + al256(Arrays<C>::bytes(groups));
    typename C::Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {groups, ar.ps, nullptr},
                                     {ar.pa, ar.sa, ar.pb, ar.sb, ar.psa, ar.lsa, ar.psb, ar.lsb},
                                     {{}, nullptr, ar.sd, ar.pd, ar.sd},
                                     hw()};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    typename C::Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return kCannotImplement;
    if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) return kInitialize;
    if (gemm.run(st) != cutlass::Status::kSuccess) return kRun;
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : kLaunch;
}

}  // namespace cut
}  // namespace bs
