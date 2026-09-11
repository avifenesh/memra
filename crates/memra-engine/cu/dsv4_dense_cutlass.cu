// DSV4 dense projections on tensor cores (memra #472).
//
// Replaces the scalar per-output-row GEMV for the prefill's m = DSV4_TMAX
// transactions. Compiled only under MEMRA_CUTLASS; when it is not built, or when
// a shape is not admitted, the caller keeps the scalar kernel and nothing here
// runs. There is no environment variable and no door: this is either the code or
// it is deleted (owner ruling 2026-09-10).
//
// WHY THIS SHAPE, in one paragraph, because the design took five passes to find:
// the dense family is 40.6% of a served prefill and is bound by INSTRUCTION
// ISSUE, not bytes (a 2-D register tile cutting reads 5.4x returned 1.29x). The
// fix is tensor cores. The obstacle was the per-128 block scale, which cannot be
// folded into a bf16 operand: doing so costs 3,236x the shipped kernel's dot
// error and, being a value 128 weights SHARE, the error is coherent and does not
// shrink with k. So the operand stays exact and the scale multiplies the block's
// f32 partial sum. And that is the whole trick here: a 128-wide k-block IS a
// split-K slice, so the scale placement the numerics demand costs no custom
// mainloop at all -- only a custom REDUCTION, with CUTLASS doing the tensor-core
// work unmodified.
//
// Split-K at that granularity is also FASTER than one slice (1.426x), because
// m = 32 is so skinny that a single-slice GEMM at n = 1024 launches 16
// threadblocks onto 188 SMs while 32 slices give 512. The constraint that was
// supposed to be the price supplies the occupancy the shape was missing.
//
// Receipts: darklanes research/dsv4f-dense-perrow-20260910/.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/numeric_types.h"

// The scale granularity along k, and therefore the split-K slice width. These are
// the same number by construction and the code says so rather than repeating 128.
#define DSV4_DENSE_SCALE_BLOCK 128

using DsvGemm = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

// ---------------------------------------------------------------------------
// Lazy bf16 mirror of the e4m3 dense weights.
//
// The conversion is LOSSLESS (e4m3 carries 3 mantissa bits, bf16 carries 7), so
// this costs memory and nothing else: about 3.39 GB per card for the whole dense
// trunk, against roughly 11.8 GB free with the model loaded. It is keyed on the
// weight pointer and filled on first use rather than at load, so no loader change
// is needed; the steady state is identical either way.
//
// The scale is deliberately NOT folded in here. That is the thing the numerics
// forbid, and mirroring the codes alone is what keeps the operand exact.
// ---------------------------------------------------------------------------
namespace {
// Announce the first admitted call AND the first declined one. The first served
// A/B of this path measured two identical arms because the symbol was
// weak-undefined and nothing said so; a path that is inert has to be able to
// report that it is inert.
void announce_decline(const char* why, int m, int n, int k, int xstride, int sc_cols) {
    static bool said = false;
    if (said) return;
    said = true;
    fprintf(stderr,
            "[dsv4-dense-cutlass] DECLINED (%s): m=%d n=%d k=%d xstride=%d sc_cols=%d; the scalar "
            "kernel runs\n",
            why, m, n, k, xstride, sc_cols);
}

struct Mirror {
    __nv_bfloat16* ptr = nullptr;
    size_t elems = 0;
};
std::mutex g_mirror_lock;
std::map<const void*, Mirror> g_mirrors;
size_t g_mirror_bytes = 0;
uint64_t g_mirror_misses = 0;

__global__ void e4m3_to_bf16_kernel(const __nv_fp8_e4m3* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst, long count) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    dst[i] = __float2bfloat16(__half2float(__nv_cvt_fp8_to_halfraw(src[i].__x, __NV_E4M3)));
}

// partial[b] is column-major [m][n] with ld = n. sc is [row_block][sc_cols].
// master += partial * scale, all in f32: the scale lands BEFORE accumulation into
// the master sum, which is the contract the whole design rests on.
__global__ void scaled_reduce_kernel(const float* __restrict__ partial,
                                     const float* __restrict__ sc, int sc_cols,
                                     float* __restrict__ y, int n, int m, int blocks,
                                     int ystride) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * m) return;
    int row = (int)(idx % n);
    int t = (int)(idx / n);
    const float* srow = sc + (long)(row >> 7) * sc_cols;
    float master = 0.f;
    for (int b = 0; b < blocks; b++)
        master += partial[(long)b * m * n + (long)t * n + row] * srow[b];
    y[(long)t * ystride + row] = master;
}

// One partial workspace per host thread, grown on demand. Peak among the shapes
// the split-K arm wins is 8.4 MB; the 529 MB case is n = 129280, which takes the
// single-slice arm and never allocates here.
thread_local float* g_ws = nullptr;
thread_local size_t g_ws_elems = 0;

float* workspace(size_t elems) {
    if (elems <= g_ws_elems) return g_ws;
    if (g_ws) cudaFree(g_ws);
    g_ws = nullptr;
    if (cudaMalloc(&g_ws, elems * sizeof(float)) != cudaSuccess) {
        g_ws_elems = 0;
        return nullptr;
    }
    g_ws_elems = elems;
    return g_ws;
}

const __nv_bfloat16* mirror_for(const void* w_codes, int n, int k, cudaStream_t stream) {
    size_t elems = (size_t)n * k;
    {
        std::lock_guard<std::mutex> guard(g_mirror_lock);
        auto it = g_mirrors.find(w_codes);
        if (it != g_mirrors.end() && it->second.elems == elems) return it->second.ptr;
    }
    __nv_bfloat16* dst = nullptr;
    if (cudaMalloc(&dst, elems * sizeof(__nv_bfloat16)) != cudaSuccess) return nullptr;
    e4m3_to_bf16_kernel<<<(elems + 255) / 256, 256, 0, stream>>>(
        (const __nv_fp8_e4m3*)w_codes, dst, (long)elems);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(dst);
        return nullptr;
    }
    // The mirror must be complete before any GEMM reads it, and it is filled once.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaFree(dst);
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(g_mirror_lock);
    auto& slot = g_mirrors[w_codes];
    if (slot.ptr && slot.elems == elems) {
        // Another thread won the race; keep one mirror and free ours.
        cudaFree(dst);
        return slot.ptr;
    }
    if (slot.ptr) cudaFree(slot.ptr);
    slot.ptr = dst;
    slot.elems = elems;
    g_mirror_bytes += elems * sizeof(__nv_bfloat16);
    g_mirror_misses++;
    return dst;
}

// NOTE, and it corrects a number this lane published: there is no per-shape
// choice between split-K and a single slice here, because a SINGLE SLICE CANNOT
// CARRY THE SCALE. One slice produces one partial covering all of k, and the
// scale changes every 128 along k, so there is nothing to multiply it by. The
// "best of two per shape" that measured 6.377x was on the UNSCALED gemm and is
// not reachable once the numerics are honoured; split-K at the scale granularity
// is mandatory rather than preferred. The real dispatch is split-K against the
// SCALAR kernel, and split-K beats it at every census shape.
uint64_t g_calls_splitk = 0;
uint64_t g_calls_declined = 0;
}  // namespace

extern "C" int memra_dsv4_dense_cutlass_counts_for_gate(uint64_t* splitk, uint64_t* declined,
                                                        uint64_t* mirror_bytes) {
    if (!splitk || !declined || !mirror_bytes) return 40079;
    *splitk = g_calls_splitk;
    *declined = g_calls_declined;
    *mirror_bytes = g_mirror_bytes;
    return 0;
}

// Returns 0 when it ran, 40080 when the shape is not admitted (the caller keeps
// the scalar kernel), and a CUDA/CUTLASS code on a real failure. Declining is
// NOT an error: it is how the skinny and strided shapes stay on the path that
// suits them.
extern "C" int memra_dsv4_dense_cutlass_fp8(const void* w_codes, const float* sc_f32, int sc_cols,
                                            const void* x_bf16, float* y, int m, int n, int k,
                                            int xstride, int ystride, void* stream_v) {
    // Admission, stated positively so a reader can see the whole domain at once.
    if (m != 32) { announce_decline("m!=32", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }
    if (k % DSV4_DENSE_SCALE_BLOCK != 0) { announce_decline("k%128", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }
    if (xstride != k) { announce_decline("xstride!=k", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }
    if (n <= 0 || k <= 0) { announce_decline("bad n/k", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }
    const int blocks = k / DSV4_DENSE_SCALE_BLOCK;
    if (sc_cols < blocks) { announce_decline("sc_cols<blocks", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }
    if (ystride <= 0) ystride = n;

    cudaStream_t stream = (cudaStream_t)stream_v;
    const __nv_bfloat16* w = mirror_for(w_codes, n, k, stream);
    if (!w) { announce_decline("no mirror", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }

    float* part = workspace((size_t)blocks * m * n);
    if (!part) { announce_decline("no workspace", m, n, k, xstride, sc_cols); g_calls_declined++; return 40080; }

    DsvGemm gemm;
    typename DsvGemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kBatched, {n, m, DSV4_DENSE_SCALE_BLOCK}, blocks,
        {1.f, 0.f}, (cutlass::bfloat16_t const*)w, (cutlass::bfloat16_t const*)x_bf16, part, part,
        (int64_t)DSV4_DENSE_SCALE_BLOCK, (int64_t)DSV4_DENSE_SCALE_BLOCK, (int64_t)m * n,
        (int64_t)m * n, k, k, n, n);
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        announce_decline("cutlass cannot_implement", m, n, k, xstride, sc_cols);
        g_calls_declined++;
        return 40080;
    }
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return 40081;
    if (gemm(stream) != cutlass::Status::kSuccess) return 40082;
    scaled_reduce_kernel<<<((long)n * m + 255) / 256, 256, 0, stream>>>(part, sc_f32, sc_cols, y, n,
                                                                       m, blocks, ystride);
    if (cudaGetLastError() != cudaSuccess) return 40083;
    // Engagement receipt, once per process. Without it a served A/B can only
    // infer that this path ran from the fact that the numbers moved, and "the
    // numbers moved" is exactly what the A/B is trying to establish.
    if (g_calls_splitk == 0)
        fprintf(stderr,
                "[dsv4-dense-cutlass] engaged: split-K at the %d-wide scale block, m=%d n=%d k=%d, "
                "blocks=%d\n",
                DSV4_DENSE_SCALE_BLOCK, m, n, k, blocks);
    g_calls_splitk++;
    return 0;
}
