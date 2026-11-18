// DeepSeek-V4.1 FP8/FP4 GEMMs on CUTLASS's sm_120 block-scaled kernels (vendored CUTLASS v4.8.1,
// third_party/cutlass). Every routed-expert and dense FP8 linear of the V4.1 route runs here, at every width. The
// configuration template, the dense launcher and the grouped call's device setup are the shared
// cu/bs_cutlass.cuh; this unit names the route's element types and forms, its form rule and its FFI.
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

#include "bs_cutlass.cuh"

using namespace cute;

namespace v41c {

using bs::cut::Cfg;

using F8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using F4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;

// the forms
enum Form { kAuto = 0, kNormal = 1, kSwap8 = 2, kSwap16 = 3, kSmall = 4 };

// The configurations (cu/bs_cutlass.cuh): e4m3 activations against e4m3 (dense FP8) or e2m1 (MXFP4) weights.
using PN8 = Cfg<F8, F8, 16, _128, false, false>;
using PS8x8 = Cfg<F8, F8, 16, _8, true, false>;
using PS8x16 = Cfg<F8, F8, 16, _16, true, false>;
using PN4 = Cfg<F8, F4, 128, _128, false, false>;
using PS4x8 = Cfg<F8, F4, 128, _8, true, false>;
using PS4x16 = Cfg<F8, F4, 128, _16, true, false>;
using GN4 = Cfg<F8, F4, 128, _128, false, true>;

// the skeleton's status as this unit's return code (41043 to 41048; 0 ok)
static inline int code(int s) { return s == bs::cut::kOk ? 0 : 41040 + s; }

template <class C>
static int plain(const void* a, long lda, const void* asf, const void* w, long ldw_el, const void* wsf, void* c,
                 long ldc, int m, int n, int k, cudaStream_t st) {
    return code(bs::cut::plain<C>(a, lda, asf, w, ldw_el, wsf, c, ldc, m, n, k, st));
}

// Each pair's scale row: pairs of local expert g sit at pad[g] + (p - off[g]); other pairs -1.
__global__ void __launch_bounds__(1024) sfrow_kernel(const int* __restrict__ off, int G, int np, int* __restrict__ sfrow) {
    __shared__ int pad[1025];
    bs::cut::pad_prefix(off, G, pad);
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
extern "C" long memra_v41_sf_off(long r, long kb, long kbn) { return bs::sf32_off(r, kb, kbn); }
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
extern "C" long memra_v41_cutlass_moe_bytes(int groups) { return bs::cut::grouped_bytes<GN4>(groups); }

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
    return code(bs::cut::grouped<GN4>(off + e0, ng, nproj, proj0, proj1, ex, a, asf, d0, d1, ldc, n, k, ldw * 2, blk,
                                      blk_bytes, (cudaStream_t)stream));
}
