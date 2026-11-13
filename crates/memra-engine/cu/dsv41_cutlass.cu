// DeepSeek-V4.1 FP8/FP4 GEMMs on CUTLASS's sm_120 block-scaled kernels (vendored CUTLASS v4.8.1,
// third_party/cutlass). Every routed-expert and dense FP8 linear of the V4.1 route runs here, at every width.
//
// The program: e4m3 activations (one ue8m0 code per row and 32 inputs) against e4m3 (dense) or MXFP4 e2m1
// (experts) weights (one ue8m0 code per output row and 32 inputs), the scaling and the fp32 accumulation inside
// the tensor core (kind::mxf8f6f4, block_scale, scale_vec::1X), one accumulator chained over the K tiles in
// ascending order, bf16 out. Three forms of the same program:
//   normal   rows on A, 128 x 128 x 128 tiles (prefill, wide batches)
//   swapped  weights on A (output columns are the MMA's M), the call's rows on B, tileN 8 or 16, D written
//            column-major, which is the row-major [rows][n] output (decode, verify, small batches)
// On GB10 (bsmma probe revs 4-6) the forms gave bit-identical fp32 accumulators at 1-64 rows an expert, for the
// experts and the dense FP8 shapes; `dsv41-gate kernels` checks every form against the normal one byte for byte.
// The choice of form is a speed choice only.
//
// Scale factors are in CUTLASS's interleaved layout (Sm1xxBlockScaledConfig<32>, the same function for A and B):
//   off(r, kb) = (r / 128) * (ceil(KB / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4
// over ceil(rows / 128) * 128 rows. Weights are repacked into it once at load (dsv41_gpu.rs `sf_pack`), and
// act_quant writes the activations' codes there (`memra_v41_act_quant_sf`). A grouped call's groups each start
// their scale rows at a multiple of 128 (the padded prefix of the groups' row counts); `memra_v41_cutlass_sfrow`
// gives each pair its scale row.
//
// Grouped calls (one group a local expert, two per expert for w1 + w3 in one launch) take their arguments on
// the card: one setup kernel writes the problem shapes, pointer arrays, strides and scale layouts from the
// device MoE plan's expert offsets, so nothing is read back to the host and a CUDA graph replays the call with
// whatever routing the plan holds. Host problem shapes are not passed (CUTLASS launches one CTA an SM and its
// scheduler walks the device shapes); groups with no rows have no tiles.
//
// Grouped calls take the normal form only, at every width. CUTLASS 4.8.1's ptr-array kernel in the swapped forms
// (tileN 8 and 16) is not deterministic on GB10 when other work runs on the card beside it: now and then one warp's
// slice of a tile (its 2 x 16 output columns of one row) comes out of one wrong K step, with the same inputs
// (2 ranks on one card, peer-pull, 40 rounds of 12/16/32-row calls a run: tileN 16 22-28 corrupt outputs a run,
// tileN 8 forced at every width 174-194, the normal form 0 in every run; receipts research/
// modelplan-onboarding-dsv41-20261109/perf/receipts/race-cutlass-gb10). The dense (non-grouped) swapped forms
// showed no such slice in the same runs and stay.
// Grouped calls of a few rows an expert (a plan of tiles of at most 8 rows) do not come here: they run the
// small-row body of cu/dsv41_gpu.cu (`memra_v41_moe_small`), which is this program element for element (the same
// MMA on e2m1 weights, one chain over K; equal to the normal form in fp32 on GB10).
#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

namespace v41c {

using F8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using F4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using SFT = cutlass::float_ue8m0_t;
using BF = cutlass::bfloat16_t;
using RowM = cutlass::layout::RowMajor;
using ColM = cutlass::layout::ColumnMajor;

// the forms
enum Form { kAuto = 0, kNormal = 1, kSwap8 = 2, kSwap16 = 3, kSmall = 4 };

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

// One configuration. EW: the weights' type; Swap: weights on A; Grouped: ptr-array grouped kernel.
template <class EW, int AlW, class TileN, bool Swap, bool Grouped>
struct Cfg {
    using EA = cute::conditional_t<Swap, EW, F8>;
    using EB = cute::conditional_t<Swap, F8, EW>;
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
    using Tile = Shape<_128, TileN, _128>;
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

// SMs of the current device (cached a device)
static int sms() {
    static int n[64] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!n[dev & 63]) cudaDeviceGetAttribute(&n[dev & 63], cudaDevAttrMultiProcessorCount, dev);
    return n[dev & 63] > 0 ? n[dev & 63] : 1;
}
static cutlass::KernelHardwareInfo hw() {
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

// ---- one dense call: c[m][n] = a . w^T
template <class C>
static int plain(const void* a, long lda, const void* asf, const void* w, long ldw_el, const void* wsf, void* c,
                 long ldc, int m, int n, int k, cudaStream_t st) {
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
    if (C::Gemm::get_workspace_size(args) != 0) return 41043;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 41044;
    if (gemm.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return 41045;
    if (gemm.run(st) != cutlass::Status::kSuccess) return 41046;
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : 41047;
}

// ---- grouped calls
struct ExpW {  // the V41ExpW record of dsv41_gpu.cu: w1, w2, w3 codes, their scales, second-level scales
    const uint8_t* w[3];
    const uint8_t* s[3];
    float scale2[3];
    int pad;
};

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

// Each pair's scale row: pairs of local expert g sit at pad[g] + (p - off[g]); other pairs -1.
__global__ void __launch_bounds__(1024) sfrow_kernel(const int* __restrict__ off, int G, int np, int* __restrict__ sfrow) {
    __shared__ int pad[1025];
    pad_prefix(off, G, pad);
    int lo = off[0], hi = off[G];
    for (int p = threadIdx.x; p < np; p += blockDim.x) {
        int r = -1;
        if (p >= lo && p < hi) {
            int a = 0, b = G - 1;  // the group holding p: the last g with off[g] <= p
            while (a < b) {
                int mid = (a + b + 1) / 2;
                if (off[mid] <= p) a = mid; else b = mid - 1;
            }
            r = pad[a] + (p - off[a]);
        }
        sfrow[p] = r;
    }
}

// The setup of one grouped call: nproj (1 or 2) projections of every local expert, groups [j * G, (j + 1) * G)
// for projection proj[j] into D base d[j]. a: the pairs' codes [np][k] (pair p at row p); asf: their scales over
// the padded rows.
template <class C>
__global__ void __launch_bounds__(1024) setup_kernel(const int* __restrict__ off, int G, int nproj, int p0, int p1,
                                                     const ExpW* __restrict__ ex, const uint8_t* a, const uint8_t* asf,
                                                     BF* d0, BF* d1, long ldc, int n, int k, long ldw_el, uint8_t* blk) {
    __shared__ int pad[1025];
    pad_prefix(off, G, pad);
    Arrays<C> ar(blk, G * nproj);
    long kbt4 = (long)((k / 32 + 3) / 4) * 4;
    static_assert(!C::kSwap, "grouped calls take the normal form only (see the top of this file)");
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

template <class C>
static long grouped_bytes(int groups) {
    return al256(Arrays<C>::bytes(groups)) + al256(ws_bytes<C>(groups));
}

template <class C>
static int grouped(const int* off, int G, int nproj, int p0, int p1, const void* ex, const void* a, const void* asf,
                   void* d0, void* d1, long ldc, int n, int k, long ldw_el, void* blk, long blk_bytes,
                   cudaStream_t st) {
    int groups = G * nproj;
    long need = grouped_bytes<C>(groups);
    if (blk_bytes < need) return 41048;
    setup_kernel<C><<<1, 1024, 0, st>>>(off, G, nproj, p0, p1, (const ExpW*)ex, (const uint8_t*)a,
                                        (const uint8_t*)asf, (BF*)d0, (BF*)d1, ldc, n, k, ldw_el, (uint8_t*)blk);
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
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 41044;
    if (gemm.initialize(args, ws, st) != cutlass::Status::kSuccess) return 41045;
    if (gemm.run(st) != cutlass::Status::kSuccess) return 41046;
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : 41047;
}

using PN8 = Cfg<F8, 16, _128, false, false>;
using PS8x8 = Cfg<F8, 16, _8, true, false>;
using PS8x16 = Cfg<F8, 16, _16, true, false>;
using PN4 = Cfg<F4, 128, _128, false, false>;
using PS4x8 = Cfg<F4, 128, _8, true, false>;
using PS4x16 = Cfg<F4, 128, _16, true, false>;
using GN4 = Cfg<F4, 128, _128, false, true>;

// the form for a call of at most `rows` rows a group (a token picks an expert once: a group's rows <= the call's)
static int pick(int form, int rows) {
    if (form != kAuto) return form;
    if (rows <= 8) return kSwap8;
    if (rows <= 16) return kSwap16;
    return kNormal;
}

}  // namespace v41c

using namespace v41c;

// The scale-layout offset (the closed form above), for the gate and the host repack tests.
extern "C" long memra_v41_sf_off(long r, long kb, long kbn) {
    return (r / 128) * (((kbn + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}
// Bytes of the scale layout of `rows` rows of `k` inputs.
extern "C" long memra_v41_sf_bytes(long rows, long k) { return (rows + 127) / 128 * 128 * (((k / 32) + 3) / 4) * 4; }

// The small-row body of the dense GEMMs (cu/dsv41_gpu.cu): this program element for element, for calls of up to
// 16 rows, and the rule that picks it for form 0.
extern "C" int memra_v41_dense_small(const void* a, long lda, const void* asf, const void* w, long ldw,
                                     const void* wsf, int wfmt, void* c, long ldc, int m, int n, int k, void* stream);
extern "C" int memra_v41_dense_small_fits(int m, int n, int k);

// c[m][n] (bf16, row pitch ldc) = a . w^T. a: e4m3 codes [m][lda]; asf: their scales in the layout of m rows;
// w: weight rows [n][ldw bytes] (wfmt 0: e4m3; 1 and 2: e2m1 pairs, low nibble first; a minted NVFP4 weight
// carries its ue8m0 codes back, see dsv41_gpu.rs); wsf: their scales in the layout of n rows. form: 0 by rows
// (the small-row body where `memra_v41_dense_small_fits` takes the call, else the CUTLASS form by rows), 1 normal,
// 2 swapped tileN 8, 3 swapped tileN 16, 4 the small-row body for every call of at most 16 rows and the CUTLASS
// form by rows above (the gates' seam to run it at every width it serves). The bytes do not depend on it.
extern "C" int memra_v41_cutlass_gemm(const void* a, long lda, const void* asf, const void* w, long ldw,
                                      const void* wsf, int wfmt, void* c, long ldc, int m, int n, int k, int form,
                                      void* stream) {
    if (m < 0 || n <= 0 || k <= 0 || k % 128 || lda % 16 || ldw % 16 || ldc % 8 || wfmt < 0 || wfmt > 2) return 41040;
    if (form < 0 || form > 4) return 41040;
    // TMA (and the small-row body's cp.async) moves 16 bytes at a time: an unaligned base is refused, not read wrong
    if (((uintptr_t)a | (uintptr_t)asf | (uintptr_t)w | (uintptr_t)wsf | (uintptr_t)c) % 16) return 41040;
    if (m == 0) return 0;
    if ((form == kSmall && m <= 16) || (form == kAuto && memra_v41_dense_small_fits(m, n, k)))
        return memra_v41_dense_small(a, lda, asf, w, ldw, wsf, wfmt, c, ldc, m, n, k, stream);
    if (form == kSmall) form = kAuto;
    cudaStream_t st = (cudaStream_t)stream;
    long ldw_el = wfmt == 0 ? ldw : ldw * 2;
    switch (pick(form, m)) {
        case kNormal:
            return wfmt == 0 ? plain<PN8>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st)
                             : plain<PN4>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st);
        case kSwap8:
            return wfmt == 0 ? plain<PS8x8>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st)
                             : plain<PS4x8>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st);
        default:
            return wfmt == 0 ? plain<PS8x16>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st)
                             : plain<PS4x16>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st);
    }
}

// Each pair's scale row for the grouped calls over local experts [e0, e0 + ng) of a plan (off: i32 [ne + 1]
// expert offsets, pairs ordered by expert): sfrow i32 [np], -1 for pairs of other experts.
extern "C" int memra_v41_cutlass_sfrow(const int* off, int e0, int ng, int np, int* sfrow, void* stream) {
    if (ng <= 0 || ng > 1024 || np < 0 || e0 < 0) return 41041;
    if (np == 0) return 0;
    sfrow_kernel<<<1, 1024, 0, (cudaStream_t)stream>>>(off + e0, ng, np, sfrow);
    return cudaGetLastError() == cudaSuccess ? 0 : 41047;
}

// Bytes of the argument block of a grouped call of `groups` groups (arrays and CUTLASS's workspace).
extern "C" long memra_v41_cutlass_moe_bytes(int groups) { return grouped_bytes<GN4>(groups); }

// One grouped call over the local experts [e0, e0 + ng) of a plan: projection proj0 of every expert into d0
// and, when nproj is 2, projection proj1 into d1 (w1 and w3 in one launch). Projections index the expert table
// records (`ExpW`, the rank's table: record g is local expert g): 0 w1, 1 w2, 2 w3. a: e4m3 codes [np][k] with
// pair p at row p; asf: their scales placed through `memra_v41_cutlass_sfrow`; ldw: a weight row's bytes (e2m1
// pairs: k / 2); n: output columns; d rows at pair index with pitch ldc. rows: the call's tokens (a group's
// most rows; none: nothing to do); form: 0 (by rows) or 1, both the normal form (the swapped grouped forms are
// gone, see the top of this file; 2 and 3 are refused); blk: `memra_v41_cutlass_moe_bytes(ng * nproj)` bytes of
// device memory that live until the launch has run (a captured graph replays from the same block).
extern "C" int memra_v41_cutlass_moe(const int* off, int e0, int ng, int nproj, int proj0, int proj1, const void* ex,
                                     const void* a, const void* asf, void* d0, void* d1, long ldc, int n, int k,
                                     long ldw, int rows, int form, void* blk, long blk_bytes, void* stream) {
    if (ng <= 0 || ng > 1024 || nproj < 1 || nproj > 2 || e0 < 0 || k % 128 || ldc % 8 || ldw % 16) return 41042;
    if (proj0 < 0 || proj0 > 2 || proj1 < 0 || proj1 > 2 || form < 0 || form > 1 || n <= 0) return 41042;
    // the setup kernel offsets these bases by whole rows and scale blocks; TMA needs each 16-byte aligned
    if (((uintptr_t)a | (uintptr_t)asf | (uintptr_t)d0 | (uintptr_t)(nproj == 2 ? d1 : d0)) % 16) return 41042;
    if (rows <= 0) return 0;
    return grouped<GN4>(off + e0, ng, nproj, proj0, proj1, ex, a, asf, d0, d1, ldc, n, k, ldw * 2, blk, blk_bytes,
                        (cudaStream_t)stream);
}
