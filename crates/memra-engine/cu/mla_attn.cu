// mla_attn.cu — MLA (multi-head latent attention) CUDA forward: the glm-dsa / glm5_next
// attention core (increment 4 of research/mla-bringup-20260801/DESIGN.md §4).
//
// NUMERIC CONTRACT: the permanent CPU f32 oracle is `crates/memra-engine/src/mla.rs`
// (`mla_attend_naive` == `mla_attend_absorbed`, proven there across shapes incl. full
// `MlaDims::GLM52` and `MlaDims::GLM5_NEXT`). Every kernel here mirrors that module's math:
// f32 accumulation throughout, softmax scale 1/sqrt(d_nope + d_rope) — the ORIGINAL qk head
// dim, NEVER the absorbed width (kv_rank + d_rope) — and the causal horizon
// `visible(i) = t_kv - t_q + i + 1` (queries are a suffix of the cache).
//
// FORM CHOICE — ABSORBED for BOTH prefill and decode.
// The oracle offers two provably equal forms. This increment runs the absorbed one everywhere:
//   * the latent cache is the ONLY KV state, so no arm ever needs the expanded per-head K/V
//     (the naive form would decompress kv_rank -> n_head*(d_nope+d_v) for every cached token,
//     i.e. materialize the very tensor MLA exists to avoid);
//   * one code path serves prefill, chunked prefill and decode, so the decode gate and the
//     prefill gate test the same arithmetic;
//   * it is what `mla_attend_absorbed` does line for line, which makes the maxdiff gate a
//     direct comparison rather than an equivalence argument.
// COST accepted deliberately: absorbed scores are kv_rank+d_rope wide (576 for GLM-5.2, 512
// for glm5_next) against the expanded form's d_nope+d_rope (256) — ~2.2x the score FLOPs at
// prefill shapes, where the expanded form is the cheaper one. Correctness first; the fused
// GEMM-shaped decode kernel and an expanded-form prefill arm are DESIGN.md increment 5.
//
// SPARSE-INDEX SEAM (DSA / glm5_next `SparseIndexPlan::Own` + kpool). The DENSE core below takes
// its cache horizon from `visible`, computed once per (query, head) block. The sparse arm takes
// that seam: `memra_mla_attn_gathered_kernel` (section "DSA k-pool indexer", end of this file)
// replaces the contiguous `0..visible` walk with a gathered position list and leaves the
// score/softmax/accumulate body unchanged — nothing here assumes cache rows are adjacent except
// the loads themselves. The selection that produces that list — pool collapse, scoring, causal
// pool validity, top-k, expansion, tail append — lives in the same section.
//
// Existing-kernel survey (reuse-or-new law):
//   - rope_neox (kernels.cu) pairs (j, j+half); GLM-5.2 rope is INTERLEAVED (2j, 2j+1). memra
//     could serve it by permuting the rope rows of wq_b/wkv_a at load (mla.rs
//     `norm_to_neox_perm`, equivalence pinned by `rope_norm_equals_permuted_neox`). This file
//     applies the interleaved rotation DIRECTLY instead, so the loader keeps checkpoint bytes
//     unmutated and the fixture's CPU projection chain (which ropes interleaved) is compared
//     against the same rotation it computes. The permutation remains available as a load-time
//     optimization; both are proven equal for dot-product consumers, which is all attention is.
//   - dsv4_rope (dsv4_gpu.cu) IS an interleaved rope, but it is table-driven (a precomputed
//     [pos][rd] cos/sin plane) and lives in the DeepSeek-V4 door's TU, which is compiled
//     -fmad=false for that lane's bit-parity contract. Coupling the MLA serving path to it
//     would import both the table plumbing and that flag's baggage.
//   - fa_prefill / fa_decode_vec_q (flash_attn.cu) assume dk == dv <= 256, per-head K and V
//     planes, and a GQA vec-lane shape. MLA is n_head_kv == 1, dk != dv, V a PREFIX VIEW of K
//     (DESIGN.md §2.3) — no variant of those kernels fits.
//
// House pattern: static-lib TU with extern "C" host launchers (mmq_ffi / dsv4_gpu kind),
// errors 0 ok / 10000+cudaError / 40000+contract, stream passed as void*.

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

#define MLA_ERR()                                              \
    do {                                                       \
        cudaError_t ce_ = cudaGetLastError();                  \
        if (ce_ != cudaSuccess) return 10000 + (int)ce_;       \
    } while (0)

// Shared-memory ceiling for the rank-space vectors held per block (q_lat + accumulator).
// GLM-5.2 and glm5_next both sit at kv_rank 512; the guard exists so a wider checkpoint
// fails LOUDLY at launch instead of silently overrunning the static shared arrays.
#define MLA_MAX_RANK 1024
#define MLA_MAX_ROPE 256
// One timestep scored per warp, so the tile depth IS the warp count of the block.
#define MLA_THREADS 256
#define MLA_WARPS (MLA_THREADS / 32)

// ------------------------------------------------------------------ rope (interleaved/NORM)

// Interleaved ("NORM") rope over each contiguous `d_rope`-wide vector of `x`, laid out
// [n_pos][n_vec][d_rope]. Mirrors mla.rs `rope_interleaved`: pair (x[2j], x[2j+1]) rotated by
// theta_j = pos * base^(-2j/d_rope).
//
// The oracle walks j with a running `theta *= theta_scale` recurrence; each lane here evaluates
// the same CLOSED FORM the oracle documents, independently (a per-lane recurrence would
// serialize the kernel). The two differ only by f32 rounding of the angle and are covered by
// the maxdiff gate, not assumed equal.
extern "C" __global__ void memra_mla_rope_interleaved_kernel(float* __restrict__ x, int n_pos,
                                                             int n_vec, int d_rope,
                                                             const int* __restrict__ positions,
                                                             float base) {
    int half = d_rope / 2;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long tot = (long)n_pos * n_vec * half;
    if (i >= tot) return;
    int j = (int)(i % half);
    long pv = i / half;
    int v = (int)(pv % n_vec);
    int p = (int)(pv / n_vec);
    float theta = (float)positions[p] * powf(base, -2.0f * (float)j / (float)d_rope);
    float s, c;
    sincosf(theta, &s, &c);
    long b = ((long)p * n_vec + v) * d_rope + 2 * j;
    float a0 = x[b], a1 = x[b + 1];
    x[b] = a0 * c - a1 * s;
    x[b + 1] = a0 * s + a1 * c;
}

extern "C" int memra_mla_rope_interleaved_f32(float* x, int n_pos, int n_vec, int d_rope,
                                              const int* positions, float base,
                                              void* stream_v) {
    if (d_rope == 0) return 0; // NoPE: no rope plane at all (glm5_next)
    if (d_rope % 2 != 0) return 40001;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)n_pos * n_vec * (d_rope / 2);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_rope_interleaved_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        x, n_pos, n_vec, d_rope, positions, base);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------------------ latent-row split / append

// Split the wkv_a output rows [t][kv_rank + d_rope] into c_kv [t][kv_rank] and k_pe [t][d_rope].
// (The engine's rms_norm and the rope kernel above both want a contiguous plane; the projection
// emits the two concatenated.)
extern "C" __global__ void memra_mla_split_latent_kernel(const float* __restrict__ kv,
                                                         float* __restrict__ c_kv,
                                                         float* __restrict__ k_pe, int t,
                                                         int kv_rank, int d_rope) {
    int width = kv_rank + d_rope;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)t * width) return;
    int row = (int)(i / width), col = (int)(i % width);
    if (col < kv_rank) {
        c_kv[(long)row * kv_rank + col] = kv[i];
    } else {
        k_pe[(long)row * d_rope + (col - kv_rank)] = kv[i];
    }
}

extern "C" int memra_mla_split_latent_f32(const float* kv, float* c_kv, float* k_pe, int t,
                                          int kv_rank, int d_rope, void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)t * (kv_rank + d_rope);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_split_latent_kernel<<<(unsigned)blocks, threads, 0, stream>>>(kv, c_kv, k_pe, t,
                                                                           kv_rank, d_rope);
    MLA_ERR();
    return 0;
}

// Append `t` latent rows [ c_kv | k_pe ] into the cache plane at row `slot`.
// The cache row IS the concatenation — V is the first kv_rank elements of the same row.
extern "C" __global__ void memra_mla_append_latent_kernel(float* __restrict__ cache,
                                                          const float* __restrict__ c_kv,
                                                          const float* __restrict__ k_pe,
                                                          int slot, int t, int kv_rank,
                                                          int d_rope) {
    int width = kv_rank + d_rope;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)t * width) return;
    int row = (int)(i / width), col = (int)(i % width);
    float v = (col < kv_rank) ? c_kv[(long)row * kv_rank + col]
                              : k_pe[(long)row * d_rope + (col - kv_rank)];
    cache[(long)(slot + row) * width + col] = v;
}

extern "C" int memra_mla_append_latent_f32(float* cache, const float* c_kv, const float* k_pe,
                                           int slot, int t, int kv_rank, int d_rope,
                                           void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)t * (kv_rank + d_rope);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_append_latent_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        cache, c_kv, k_pe, slot, t, kv_rank, d_rope);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------------------ absorb / decompress GEMMs

// q_lat[i][h][l] = sum_p q_nope[i][h][p] * w_uk[h][p][l]   (mla.rs "absorb")
//
// `wk_b` is the CONVERSION-SPLIT tensor `attn_k_b`, ne {d_nope, kv_rank, n_head}: element
// (h, l, p) lives at h*kv_rank*d_nope + l*d_nope + p — i.e. CONTIGUOUS IN p, which is exactly
// the reduction axis, so each output element is one coalesced-ish dot. This is the transposed
// layout the loader asserts at load; a non-transposed checkpoint would produce silent garbage,
// which is why that assert exists in `MlaAttnLayer::load`.
extern "C" __global__ void memra_mla_absorb_q_kernel(const float* __restrict__ q_nope,
                                                     const float* __restrict__ wk_b,
                                                     float* __restrict__ q_lat, int n_head,
                                                     int d_nope, int kv_rank) {
    extern __shared__ float smem[];
    int blk = blockIdx.x;      // i * n_head + h
    int h = blk % n_head;
    const float* qn = q_nope + (long)blk * d_nope;
    for (int p = threadIdx.x; p < d_nope; p += blockDim.x) smem[p] = qn[p];
    __syncthreads();
    const float* w = wk_b + (long)h * kv_rank * d_nope;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
        const float* row = w + (long)l * d_nope;
        float acc = 0.0f;
        for (int p = 0; p < d_nope; ++p) acc += smem[p] * row[p];
        q_lat[(long)blk * kv_rank + l] = acc;
    }
}

extern "C" int memra_mla_absorb_q_f32(const float* q_nope, const float* wk_b, float* q_lat,
                                      int t_q, int n_head, int d_nope, int kv_rank,
                                      void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_absorb_q_kernel<<<(unsigned)blocks, MLA_THREADS, d_nope * sizeof(float), stream>>>(
        q_nope, wk_b, q_lat, n_head, d_nope, kv_rank);
    MLA_ERR();
    return 0;
}

// out[i][h][j] = sum_l w_uv[h][j][l] * o_lat[i][h][l]   (mla.rs "decompress once")
//
// `wv_b` is `attn_v_b`, ne {kv_rank, d_v, n_head}: element (h, j, l) at
// h*d_v*kv_rank + j*kv_rank + l — contiguous in l, the reduction axis.
extern "C" __global__ void memra_mla_decompress_v_kernel(const float* __restrict__ o_lat,
                                                         const float* __restrict__ wv_b,
                                                         float* __restrict__ out, int n_head,
                                                         int d_v, int kv_rank) {
    extern __shared__ float smem[];
    int blk = blockIdx.x; // i * n_head + h
    int h = blk % n_head;
    const float* ol = o_lat + (long)blk * kv_rank;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) smem[l] = ol[l];
    __syncthreads();
    const float* w = wv_b + (long)h * d_v * kv_rank;
    for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
        const float* row = w + (long)j * kv_rank;
        float acc = 0.0f;
        for (int l = 0; l < kv_rank; ++l) acc += row[l] * smem[l];
        out[(long)blk * d_v + j] = acc;
    }
}

extern "C" int memra_mla_decompress_v_f32(const float* o_lat, const float* wv_b, float* out,
                                          int t_q, int n_head, int d_v, int kv_rank,
                                          void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_kernel<<<(unsigned)blocks, MLA_THREADS, kv_rank * sizeof(float),
                                    stream>>>(o_lat, wv_b, out, n_head, d_v, kv_rank);
    MLA_ERR();
    return 0;
}

// ---------------------------------------------------- decode-split twins (lane/glm5-decode-diet)
//
// PURE LAUNCH-GEOMETRY RESTRUCTURE, BIT-GATED (lever 4 of the decode diet, 2026-08-31). At
// decode widths the absorb/decompress launchers above put t_q*n_head blocks on the grid — 64
// blocks at t=1 on the glm5 geometry, single-digit-percent occupancy on a ~170-SM card, and
// the census priced the pair at ~211 us/layer (104 absorb + 107 decompress; weights are only
// ~0.7 ms of the whole 4.76 ms family). The twins below split each (token, head) block's
// OUTPUT RANGE across `split` blocks: every output element is still ONE thread's serial
// ascending-index dot — the same expression, the same order, the same bits — only WHICH block
// computes it changes, so bit identity to the unsplit kernels is by construction and asserted
// bytewise by crates/memra-engine/tests/mla_decode_split_gpu.rs (including split values that
// do not divide the output width). The smem stage is the same loads with the same values.
// Host seam MEMRA_MLA_DECODE_SPLIT (default OFF, read per call in mla_ffi.rs).

extern "C" __global__ void memra_mla_absorb_q_split_kernel(const float* __restrict__ q_nope,
                                                           const float* __restrict__ wk_b,
                                                           float* __restrict__ q_lat,
                                                           int n_head, int d_nope, int kv_rank,
                                                           int split) {
    extern __shared__ float smem[];
    int blk = blockIdx.x / split;   // i * n_head + h
    int chunk = blockIdx.x % split; // this block's output slice
    int h = blk % n_head;
    const float* qn = q_nope + (long)blk * d_nope;
    for (int p = threadIdx.x; p < d_nope; p += blockDim.x) smem[p] = qn[p];
    __syncthreads();
    const float* w = wk_b + (long)h * kv_rank * d_nope;
    int per = (kv_rank + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < kv_rank ? lo + per : kv_rank;
    for (int l = lo + threadIdx.x; l < hi; l += blockDim.x) {
        const float* row = w + (long)l * d_nope;
        float acc = 0.0f;
        for (int p = 0; p < d_nope; ++p) acc += smem[p] * row[p];
        q_lat[(long)blk * kv_rank + l] = acc;
    }
}

extern "C" int memra_mla_absorb_q_split_f32(const float* q_nope, const float* wk_b,
                                            float* q_lat, int t_q, int n_head, int d_nope,
                                            int kv_rank, int split, void* stream_v) {
    if (split < 1 || split > kv_rank) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_absorb_q_split_kernel<<<(unsigned)blocks, MLA_THREADS, d_nope * sizeof(float),
                                      stream>>>(q_nope, wk_b, q_lat, n_head, d_nope, kv_rank,
                                                split);
    MLA_ERR();
    return 0;
}

extern "C" __global__ void memra_mla_decompress_v_split_kernel(const float* __restrict__ o_lat,
                                                               const float* __restrict__ wv_b,
                                                               float* __restrict__ out,
                                                               int n_head, int d_v, int kv_rank,
                                                               int split) {
    extern __shared__ float smem[];
    int blk = blockIdx.x / split;
    int chunk = blockIdx.x % split;
    int h = blk % n_head;
    const float* ol = o_lat + (long)blk * kv_rank;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) smem[l] = ol[l];
    __syncthreads();
    const float* w = wv_b + (long)h * d_v * kv_rank;
    int per = (d_v + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < d_v ? lo + per : d_v;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
        const float* row = w + (long)j * kv_rank;
        float acc = 0.0f;
        for (int l = 0; l < kv_rank; ++l) acc += row[l] * smem[l];
        out[(long)blk * d_v + j] = acc;
    }
}

extern "C" int memra_mla_decompress_v_split_f32(const float* o_lat, const float* wv_b,
                                                float* out, int t_q, int n_head, int d_v,
                                                int kv_rank, int split, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (split < 1 || split > d_v) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_split_kernel<<<(unsigned)blocks, MLA_THREADS,
                                          kv_rank * sizeof(float), stream>>>(
        o_lat, wv_b, out, n_head, d_v, kv_rank, split);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------------------ absorbed MQA attention

// One block per (query i, head h). Streams the latent cache one timestep per warp per tile, with an
// online (flash-style) softmax: running max `m` and denominator `d`, accumulator held in shared
// rank space. Serves prefill, chunked prefill and decode identically — decode is simply t_q == 1.
//
// The accumulation ORDER differs from the CPU oracle's (tiled + rescaled vs a single left-to-right
// pass), so parity is a maxdiff bound, never bit-identity. That is the documented bar for this
// kernel family (DESIGN.md increment 4: "per-layer maxdiff vs mla.rs").
extern "C" __global__ void memra_mla_attn_absorbed_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, float* __restrict__ o_lat, int n_head, int kv_rank,
    int d_rope, int t_q, int t_kv, float scale) {
    __shared__ float s_q[MLA_MAX_RANK];
    __shared__ float s_qp[MLA_MAX_ROPE];
    __shared__ float s_acc[MLA_MAX_RANK];
    __shared__ float s_score[MLA_WARPS];

    int blk = blockIdx.x; // i * n_head + h
    int i = blk / n_head;
    int width = kv_rank + d_rope;
    // Causal horizon, mla.rs convention: the queries occupy the LAST t_q cache rows.
    int visible = t_kv - t_q + i + 1;

    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
        s_q[l] = q_lat[(long)blk * kv_rank + l];
        s_acc[l] = 0.0f;
    }
    for (int p = threadIdx.x; p < d_rope; p += blockDim.x)
        s_qp[p] = q_pe[(long)blk * d_rope + p];
    __syncthreads();

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float m = -FLT_MAX;
    float dsum = 0.0f;

    for (int t0 = 0; t0 < visible; t0 += MLA_WARPS) {
        int t = t0 + warp;
        float part = 0.0f;
        if (t < visible) {
            const float* row = cache + (long)t * width;
            for (int l = lane; l < kv_rank; l += 32) part += s_q[l] * row[l];
            for (int p = lane; p < d_rope; p += 32) part += s_qp[p] * row[kv_rank + p];
        }
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
        if (lane == 0) s_score[warp] = (t < visible) ? part * scale : -FLT_MAX;
        __syncthreads();

        // Every thread recomputes the tile's softmax bookkeeping from the SAME shared scores,
        // so m/dsum stay identical across the block without a second reduction.
        float tmax = -FLT_MAX;
        for (int w = 0; w < MLA_WARPS; ++w) tmax = fmaxf(tmax, s_score[w]);
        float mnew = fmaxf(m, tmax);
        float rescale = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
        float tsum = 0.0f;
        for (int w = 0; w < MLA_WARPS; ++w)
            if (t0 + w < visible) tsum += expf(s_score[w] - mnew);
        dsum = dsum * rescale + tsum;

        for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
            float a = s_acc[l] * rescale;
            for (int w = 0; w < MLA_WARPS; ++w) {
                int tt = t0 + w;
                if (tt >= visible) break;
                a += expf(s_score[w] - mnew) * cache[(long)tt * width + l];
            }
            s_acc[l] = a;
        }
        m = mnew;
        __syncthreads();
    }

    float inv = 1.0f / dsum;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x)
        o_lat[(long)blk * kv_rank + l] = s_acc[l] * inv;
}

extern "C" int memra_mla_attn_absorbed_f32(const float* q_lat, const float* q_pe,
                                           const float* cache, float* o_lat, int n_head,
                                           int kv_rank, int d_rope, int t_q, int t_kv,
                                           float scale, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (t_q > t_kv) return 40004; // queries must be a suffix of the cache (mla.rs contract)
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_attn_absorbed_kernel<<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, o_lat, n_head, kv_rank, d_rope, t_q, t_kv, scale);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------------------ DSA k-pool indexer
//
// The SPARSE ARM the file header reserves. glm5_next's 11 MLA layers (+1 MTP) each run a
// DeepSeek-Sparse-Attention indexer that picks at most `index_topk` cache positions per query;
// below that budget it selects everything and coincides with the dense core above, ABOVE it the
// two are different functions. The oracle is `memra_reference::kpool_allowed_tokens`
// (crates/memra-reference/src/lib.rs), itself a transcription of `Glm5NextTextIndexer.forward`
// (research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py:771).
//
// SCOPE, and it is load-bearing: single sequence, no padding, pooling starts at cache row 0.
// That is the scope the Rust oracle documents for itself and the shape this engine's per-session
// cache has. The reference's packed per-token validity channel is therefore absent from the state
// plane below; a batched/padded arm needs it back, and its own gate.
//
// Prior art followed: `dsv4_indexer_score_kernel` / `dsv4_sink_attn_kernel` (cu/dsv4_gpu.cu) —
// block per (query, candidate) with one thread per head for scoring, and a gathered index list
// with -1 = masked for the attention walk. The SELECTION body differs (pools collapsed by a
// learned softmax, not raw per-token keys), the machinery does not.

#define MLA_MAX_POOL 16

// TAIL RING. The indexer state plane is read EXACTLY ONCE per row — by the pool-key build of the
// pool that row belongs to (below) — so every row under `pools_ready * pool` is dead and the plane
// only has to hold the incomplete tail plus the slice of a call currently in flight (the Rust
// side DRAINS the ring inside one call, so `rows` bounds no `t`). `rows` is the physical ring size,
// ALWAYS a multiple of `pool` (the Rust side rounds the allocation down), which keeps a pool's
// `pool` members contiguous mod `rows` and is why both the append and the read can be a plain
// `% rows` with no wrap-splitting anywhere. `rows == 0` is the FLAT plane, absolute addressing —
// bit-identical output, identical loads, one integer modulo apart.
//
// Deliberately a SEPARATE symbol from `memra_mla_append_latent_f32`: the LATENT plane is not a
// ring (its rows are re-read by every later query through the gathered attention walk), so the
// two planes must not share a row-addressing contract even though they share a row SHAPE.
extern "C" __global__ void memra_mla_index_append_ring_kernel(float* __restrict__ plane,
                                                              const float* __restrict__ a,
                                                              const float* __restrict__ b,
                                                              int slot, int t, int wa, int wb,
                                                              int rows) {
    int width = wa + wb;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)t * width) return;
    int row = (int)(i / width), col = (int)(i % width);
    float v = (col < wa) ? a[(long)row * wa + col] : b[(long)row * wb + (col - wa)];
    int dst = slot + row;
    if (rows > 0) dst %= rows;
    plane[(long)dst * width + col] = v;
}

extern "C" int memra_mla_index_append_ring_f32(float* plane, const float* a, const float* b,
                                               int slot, int t, int wa, int wb, int rows,
                                               void* stream_v) {
    if (slot < 0 || t < 0 || wa < 0 || wb < 0 || rows < 0) return 40017;
    // A call wider than the ring would overwrite its own rows mid-append. The Rust caller proves
    // the stronger liveness bound before it gets here; this is the kernel's own floor.
    if (rows > 0 && t > rows) return 40018;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)t * (wa + wb);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_index_append_ring_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        plane, a, b, slot, t, wa, wb, rows);
    MLA_ERR();
    return 0;
}

// pool_keys[p][c] = sum_s softmax_s(gate[p*pool+s][c] + ape[s][c]) * k[p*pool+s][c]
//
// `state` rows are [ k_norm(wk(x)) : d | index_kpool_compress_gate(x) : d ], the layout of the
// indexer plane in `LatentKvLayer::index_rows`. The softmax is PER CHANNEL over the pool's
// members — not over channels — which is why the reduction axis here is `s` and every (p, c)
// pair is independent.
//
// Only COMPLETE pools exist: `n_pools = t_kv / pool`. The incomplete tail is never collapsed and
// never scored; it enters the selection as raw indices through the tail append below.
//
// INCREMENTAL BUILD (`pool_begin`). A pool's key is a function of the `pool` state rows it owns
// and the constant `ape` — nothing else. Those rows are append-only and never rewritten, so the
// key is FINAL the moment the pool's last row lands and is IMMUTABLE forever after. The launcher
// therefore builds only pools `[pool_begin, n_pools)`; pools below `pool_begin` are already
// resident in `pool_keys` from an earlier call, bit-identical to what a rebuild would produce
// (same kernel, same inputs, same arithmetic). `pool_begin == 0` is the full rebuild.
//
// The one non-immutable object in this scheme is the INCOMPLETE TAIL, and it is deliberately not
// a pool: rows `[n_pools * pool, t_kv)` never collapse into a key at all. They reach the query
// through `always_tail` in the selection kernel, recomputed from `t_kv` every call.
extern "C" __global__ void memra_mla_kpool_pool_keys_kernel(const float* __restrict__ state,
                                                            const float* __restrict__ ape,
                                                            float* __restrict__ pool_keys,
                                                            int pool_begin, int n_pools, int pool,
                                                            int d, int state_rows) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long span = (long)(n_pools - pool_begin) * d;
    if (i >= span) return;
    i += (long)pool_begin * d;
    int p = (int)(i / d), c = (int)(i % d);
    int stride = 2 * d;
    // TAIL RING (`state_rows > 0`): the state plane holds only the live tail, so a pool's rows
    // live at `(p * pool + s) % state_rows`. `state_rows` is a multiple of `pool`, so the pool's
    // members stay CONTIGUOUS in the ring and the reduction below is the same walk it always was
    // — same values, same order, same arithmetic. `state_rows == 0` is the flat plane.
    long base = (long)p * pool;
    if (state_rows > 0) base %= state_rows;
    float logits[MLA_MAX_POOL];
    float m = -FLT_MAX;
    for (int s = 0; s < pool; ++s) {
        float l = state[(base + s) * stride + d + c] + ape[(long)s * d + c];
        logits[s] = l;
        m = fmaxf(m, l);
    }
    float sum = 0.0f;
    for (int s = 0; s < pool; ++s) {
        logits[s] = expf(logits[s] - m);
        sum += logits[s];
    }
    float acc = 0.0f;
    for (int s = 0; s < pool; ++s) acc += (logits[s] / sum) * state[(base + s) * stride + c];
    pool_keys[i] = acc;
}

extern "C" int memra_mla_kpool_pool_keys_f32(const float* state, const float* ape,
                                             float* pool_keys, int pool_begin, int n_pools,
                                             int pool, int d, int state_rows, void* stream_v) {
    if (pool <= 0 || pool > MLA_MAX_POOL) return 40010;
    // A resident plane that claims MORE finished pools than the cache holds is a stale-key bug,
    // not a no-op: it would leave pools built over rows that were later overwritten. Refuse.
    if (pool_begin < 0 || pool_begin > n_pools) return 40016;
    // A ring that is not a whole number of pools would split a pool across the wrap, and the
    // contiguous `base + s` walk above would silently read the wrong rows. Refuse.
    if (state_rows < 0 || (state_rows > 0 && state_rows % pool != 0)) return 40019;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)(n_pools - pool_begin) * d;
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_kpool_pool_keys_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        state, ape, pool_keys, pool_begin, n_pools, pool, d, state_rows);
    MLA_ERR();
    return 0;
}

// score[i][p] = sum_h relu(dot(q[i][h], pool_key[p]) * qk_scale) * (hw[i][h] * head_scale),
// -INFINITY where pool `p` is not causally visible to query `i`.
//
// A pool is selectable only when its LAST token is visible, i.e. p < (pos_i + 1) / pool with
// pos_i the query's ABSOLUTE cache row (`first_pos + i`) — queries are the last t_q rows.
//
// TWO IMPLEMENTATIONS, ONE ANSWER — the same arrangement the selection below carries.
//   * `..._score_ref_kernel` — the correctness-grade original: block per (query, candidate), one
//     thread per head, head sum walked SEQUENTIALLY in h order by thread 0 so the accumulation
//     order is the oracle's. dsv4's shape. RETAINED as the in-tree definition of the arithmetic.
//   * `..._score_tiled_kernel` — the SHIPPED one, a register-tiled fused GEMM+reduce.
// They are gated BIT-IDENTICAL against each other
// (`gpu_kpool_scoring_is_byte_identical_to_the_reference_kernel`), which is why the fast one
// carries no flag: it is not a second scoring program, it is the same one computed faster.
//
// THE ROUNDING SEQUENCE IS THE CONTRACT. Read off this kernel's PTX (CUDA 13.1, -O3, and this
// TU is deliberately NOT compiled -fmad=false, see build.rs) it is, per (query, head, pool):
//     dot   = fma.rn(q[c], k[c], dot)   sequentially, c ascending, from +0.0f
//     m1    = mul.rn(dot, qk_scale)
//     r     = max.f32(m1, +0.0f)        <- fmaxf(x, 0.0f), operand order load-bearing for -0.0
//     w     = mul.rn(hw[t][h], head_scale)
//     s     = mul.rn(r, w)
//     acc   = add.rn(acc, s)            sequentially, h ascending, from +0.0f
// Six separately-rounded steps. The tiled kernel spells every one of them with an explicit
// `__fmaf_rn` / `__fmul_rn` / `__fadd_rn` intrinsic so no contraction decision, no unroll shape
// and no compiler version can fork it: `sc += relu * w` alone would contract to one FMA and
// round ONCE where the reference rounds twice. Both accumulators start at +0.0f rather than at
// the first term, because `(+0.0) + (-0.0)` is `+0.0` while `-0.0` alone is not.
//
// WHY BIT-IDENTITY AND NOT A TOLERANCE. The selection downstream sorts on these values with a
// tie-break (score descending, pool index ascending). ReLU makes exact 0.0 ties ORDINARY, and a
// last-ulp difference either side of zero moves a pool in or out of the budget. A faster scorer
// that changed a score bit would be a different selection program, not a faster one.
//
// WHY NOT cuBLASLt. Three reasons, all fatal: (1) true f32 on Blackwell is SIMT FFMA — the
// tensor cores only serve TF32/BF16 — so a real-f32 GEMM buys no peak this kernel cannot reach,
// and the TF32 path that WOULD be faster is exactly what moves scores; (2) a GEMM's reduction
// order is reassociated and split-K'd, so `relu(±epsilon)` flips and the tie set moves —
// bit-identity stops being a construction and becomes an empirical hope; (3) the per-head score
// plane `[t_q * heads, n_pools]` is 17 GB at the shipped 1M/512 shape, so the head mix cannot be
// a separate pass without chunking the pool axis and paying ~32 round-trips of a 536 MB
// intermediate. Fusing the head reduction into the accumulator is the whole point.
extern "C" __global__ void memra_mla_kpool_score_ref_kernel(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int heads, int d, int n_pools,
    int pool, int first_pos, float qk_scale, float head_scale) {
    extern __shared__ float sh[];
    long i = blockIdx.x;
    int t = (int)(i / n_pools);
    int p = (int)(i % n_pools);
    int visible_pools = (first_pos + t + 1) / pool;
    if (visible_pools > n_pools) visible_pools = n_pools;
    if (p >= visible_pools) {
        if (threadIdx.x == 0) score[i] = -INFINITY;
        return;
    }
    int h = threadIdx.x;
    if (h < heads) {
        const float* qr = q + ((long)t * heads + h) * d;
        const float* kr = pool_keys + (long)p * d;
        float dot = 0.0f;
        for (int c = 0; c < d; ++c) dot += qr[c] * kr[c];
        sh[h] = fmaxf(dot * qk_scale, 0.0f) * (hw[(long)t * heads + h] * head_scale);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int hh = 0; hh < heads; ++hh) acc += sh[hh];
        score[i] = acc;
    }
}

extern "C" int memra_mla_kpool_score_ref_f32(const float* q, const float* pool_keys,
                                             const float* hw, float* score, int t_q, int heads,
                                             int d, int n_pools, int pool, int first_pos,
                                             float qk_scale, float head_scale, void* stream_v) {
    if (heads <= 0 || heads > 1024) return 40011;
    if (pool <= 0) return 40010;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)t_q * n_pools;
    if (n == 0) return 0;
    if (n > 2147483647L) return 40012; // grid.x contract
    memra_mla_kpool_score_ref_kernel<<<(unsigned)n, heads, (size_t)heads * sizeof(float),
                                       stream>>>(q, pool_keys, hw, score, heads, d, n_pools, pool,
                                                 first_pos, qk_scale, head_scale);
    MLA_ERR();
    return 0;
}

// ---------------------------------------------------------------------------- scoring, tiled
//
// THE SHIPPED SCORER. Same six-step rounding sequence as the reference kernel above, same
// visibility rule, same -INFINITY marks — rearranged so the machine can actually feed it.
//
// WHAT THE REFERENCE KERNEL SPENDS AND WHERE. Per (query, pool) it launches ONE block of `heads`
// threads, and each of those threads walks the full `d`-long dot alone:
//   * `pool_keys[p]` is re-read once per QUERY (134 MB x 512 queries = 68 GB per layer at the
//     1M/512 prefill shape) and `q[t][h]` is read with a `heads*d` stride between neighbouring
//     threads, so neither operand is reused in registers and neither load coalesces;
//   * the head mix is 32 serial adds executed by thread 0 with 31 threads parked;
//   * the grid is `t_q * n_pools` = 134M blocks of 32 threads, so per-block launch and
//     __syncthreads overhead is paid 134M times for 4096 FMAs of real work each.
// MEASURED (2x RTX PRO 6000 Blackwell Server, release, kpool-bench-frankfurt.txt): 1294.5 ms
// per MLA layer for one 512-token prefill chunk at 1M context — ~425 GFLOP/s of a card that
// carries ~100 TFLOP/s of f32 FMA. x12 MLA layers that is 15.5 s per chunk, and scoring is the
// dominant stage at EVERY shape on the ladder.
//
// THE ARRANGEMENT. One block owns a `BT x BP` tile of the (query, pool) score plane:
//   * the pool-key tile `[d][BP]` is loaded ONCE, transposed, and stays resident in shared
//     memory for the whole head loop — this is what turns the 32x-per-head re-read into one
//     read, and it is why `d` is bounded (see the smem guard in the launcher);
//   * the q tile for the CURRENT head streams through in `KC`-column slabs, also transposed;
//   * each thread keeps `RT x RP` dot accumulators and `RT x RP` head-mixed accumulators in
//     registers, so the inner step is `RT + RP` shared loads for `RT * RP` FMAs;
//   * thread ownership is STRIDED (`t0 + ty + rt*TY`, `p0 + tx + rp*TX`), which makes every
//     shared read either a broadcast or 32 consecutive banks — no conflicts, no alignment
//     requirement on the padded row strides.
// Traffic at 1M/512 with the BIG config: pool keys 4 x 134 MB (one pass per query tile) instead
// of 68 GB, q ~34 GB but from an 8 MB working set that lives in L2, score written once (536 MB).
// The kernel is then FMA-bound at 5.5e11 FMAs per layer.
//
// EDGE RULES that keep the arithmetic identical rather than merely close:
//   * the c loop is bounded by `kc_len`, never zero-padded to `KC`. A padded `fma(0,0,dot)` is
//     not a no-op: it turns a `-0.0` dot into `+0.0`, which changes the sign of a zero score.
//   * out-of-range queries/pools are zero-filled on load and DROPPED on store, never stored.
//   * the whole-block "nothing here is visible" early-out is decided on blockIdx before the
//     first __syncthreads, so every barrier stays block-uniform.
template <int TX, int TY, int RT, int RP, int KC>
__global__ __launch_bounds__(TX* TY) void memra_mla_kpool_score_tiled_kernel(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int t_q, int heads, int d,
    int n_pools, int pool, int first_pos, float qk_scale, float head_scale) {
    constexpr int BT = TY * RT;   // queries per block
    constexpr int BP = TX * RP;   // pools per block
    constexpr int NT = TX * TY;   // threads per block
    constexpr int SBP = BP + 1;   // +1: transposed stores walk this stride, 1 mod 32 = no conflict
    constexpr int SBT = BT + 1;

    const int tid = (int)threadIdx.x;
    const int tx = tid % TX;
    const int ty = tid / TX;
    const int p0 = (int)blockIdx.x * BP;
    const int t0 = (int)blockIdx.y * BT;

    // Causal horizon: pool `p` is selectable by query `t` only once its LAST token is visible,
    // i.e. p < (first_pos + t + 1) / pool. The block's most permissive query bounds the tile.
    int t_last = t0 + BT - 1;
    if (t_last > t_q - 1) t_last = t_q - 1;
    int vis_max = (first_pos + t_last + 1) / pool;
    if (vis_max > n_pools) vis_max = n_pools;
    if (p0 >= vis_max) {
#pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
            int t = t0 + ty + rt * TY;
            if (t >= t_q) continue;
#pragma unroll
            for (int rp = 0; rp < RP; ++rp) {
                int p = p0 + tx + rp * TX;
                if (p < n_pools) score[(long)t * n_pools + p] = -INFINITY;
            }
        }
        return;
    }

    extern __shared__ float kpool_sh[];
    float* ksh = kpool_sh;                 // [d][SBP], resident across the head loop
    float* qsh = kpool_sh + (long)d * SBP; // [min(KC,d)][SBT], one head's slab at a time

    for (int i = tid; i < d * BP; i += NT) {
        int p = i / d, c = i - p * d;
        int gp = p0 + p;
        ksh[(long)c * SBP + p] = (gp < n_pools) ? pool_keys[(long)gp * d + c] : 0.0f;
    }

    float sc[RT][RP];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int rp = 0; rp < RP; ++rp) sc[rt][rp] = 0.0f;

    for (int h = 0; h < heads; ++h) {
        float dot[RT][RP];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int rp = 0; rp < RP; ++rp) dot[rt][rp] = 0.0f;

        for (int kc0 = 0; kc0 < d; kc0 += KC) {
            const int kc_len = (d - kc0 < KC) ? (d - kc0) : KC;
            __syncthreads(); // the previous slab's readers are done with qsh
            for (int i = tid; i < kc_len * BT; i += NT) {
                int t = i / kc_len, c = i - t * kc_len;
                int gt = t0 + t;
                qsh[(long)c * SBT + t] =
                    (gt < t_q) ? q[((long)gt * heads + h) * d + kc0 + c] : 0.0f;
            }
            __syncthreads();
#pragma unroll 4
            for (int c = 0; c < kc_len; ++c) {
                float a[RT], b[RP];
#pragma unroll
                for (int rt = 0; rt < RT; ++rt) a[rt] = qsh[(long)c * SBT + ty + rt * TY];
#pragma unroll
                for (int rp = 0; rp < RP; ++rp)
                    b[rp] = ksh[(long)(kc0 + c) * SBP + tx + rp * TX];
#pragma unroll
                for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                    for (int rp = 0; rp < RP; ++rp)
                        dot[rt][rp] = __fmaf_rn(a[rt], b[rp], dot[rt][rp]);
            }
        }

        // Head mix, spelled to the reference kernel's rounding sequence exactly. Every operation
        // is an explicit intrinsic: `sc[rt][rp] += relu * w` would contract to a single FMA and
        // round once where the reference rounds twice.
#pragma unroll
        for (int rt = 0; rt < RT; ++rt) {
            int t = t0 + ty + rt * TY;
            float w = (t < t_q) ? __fmul_rn(hw[(long)t * heads + h], head_scale) : 0.0f;
#pragma unroll
            for (int rp = 0; rp < RP; ++rp) {
                float r = fmaxf(__fmul_rn(dot[rt][rp], qk_scale), 0.0f);
                sc[rt][rp] = __fadd_rn(sc[rt][rp], __fmul_rn(r, w));
            }
        }
    }

#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
        int t = t0 + ty + rt * TY;
        if (t >= t_q) continue;
        int vis = (first_pos + t + 1) / pool;
        if (vis > n_pools) vis = n_pools;
#pragma unroll
        for (int rp = 0; rp < RP; ++rp) {
            int p = p0 + tx + rp * TX;
            if (p >= n_pools) continue;
            score[(long)t * n_pools + p] = (p < vis) ? sc[rt][rp] : -INFINITY;
        }
    }
}

/// Shared-memory bytes one tile configuration needs, and the launch when it fits. Returns 1 when
/// the tile does not fit the 48 KB static ceiling (the caller falls back to the reference
/// kernel — same answer, slow path, and it is what a `d` wider than any shipped indexer takes).
template <int TX, int TY, int RT, int RP, int KC>
static int memra_kpool_score_launch(const float* q, const float* pool_keys, const float* hw,
                                    float* score, int t_q, int heads, int d, int n_pools,
                                    int pool, int first_pos, float qk_scale, float head_scale,
                                    cudaStream_t stream) {
    constexpr int BT = TY * RT;
    constexpr int BP = TX * RP;
    const int kc = (d < KC) ? d : KC;
    const size_t smem = ((size_t)d * (BP + 1) + (size_t)kc * (BT + 1)) * sizeof(float);
    if (smem > 48u * 1024u) return 1;
    // OCCUPANCY, and it is not cosmetic: the BIG tile wants ~41 KB, so two blocks per SM need a
    // shared carveout above the default. At one block per SM every warp in the SM sits on the
    // same __syncthreads pair (8 per head, 32 heads) and the SM idles through each q-slab's
    // global load latency; two blocks interleave those bubbles. The kernel's own reuse is all in
    // registers and shared memory, so the L1 it gives up buys nothing back.
    static bool carveout_set = false; // benign race: the call is idempotent
    if (!carveout_set) {
        cudaFuncSetAttribute(memra_mla_kpool_score_tiled_kernel<TX, TY, RT, RP, KC>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        carveout_set = true;
    }
    dim3 grid((unsigned)((n_pools + BP - 1) / BP), (unsigned)((t_q + BT - 1) / BT));
    memra_mla_kpool_score_tiled_kernel<TX, TY, RT, RP, KC><<<grid, TX * TY, smem, stream>>>(
        q, pool_keys, hw, score, t_q, heads, d, n_pools, pool, first_pos, qk_scale, head_scale);
    return 0;
}

// Tile choice is a function of `t_q` only, and `t_q` is a serving shape, not a tuning knob:
//   BIG   (BT 128) — a full prefill chunk (512 on the shipped path). 8x8 register tile, 16 shared
//                    loads per 64 FMAs, and only 4 passes over the 134 MB pool-key plane.
//   MID   (BT 32)  — the LAST chunk of any prompt is an arbitrary t_q < chunk, so this is a real
//                    serving shape and not a corner case. Same block, 2x8 tile.
//   SMALL (BT 1)   — decode. One query, so there is no query axis to tile over and every thread
//                    owns one pool; the stage is bandwidth-bound on the pool-key plane there
//                    (134 MB at 1M context), not FMA-bound.
// Thresholds sit at the tile heights so a shape never runs a tile it cannot half-fill.
//
// SMALL-TILE CROSSOVER, measured (2x RTX PRO 6000 Blackwell Server, release, 11 trials,
// research/glm53-flash-bringup-20260827/kpool-bench-frankfurt-tiled.txt): at t_q=1 the
// tiled path is SLOWER than the reference kernel below ~16k pools -- 0.140 vs 0.021 ms at
// 1024 pools (6.5x), 0.141 vs 0.052 ms at 4096 (2.7x) -- and only wins from ~16k pools up
// (0.151 vs 0.171, then 1.560 vs 2.537 at 262144). The tile's fixed setup dominates when
// there is one query and few pools, which is short-context DECODE: the most common serving
// shape, not a corner case. So decode dispatches on pool count, to the kernel measured
// faster at that size. Both kernels are bit-identical (gate 12), so this picks speed only.
#define MLA_KPOOL_SMALL_TILE_MIN_POOLS 16384
extern "C" int memra_mla_kpool_score_f32(const float* q, const float* pool_keys, const float* hw,
                                         float* score, int t_q, int heads, int d, int n_pools,
                                         int pool, int first_pos, float qk_scale,
                                         float head_scale, void* stream_v) {
    if (heads <= 0 || heads > 1024) return 40011;
    if (pool <= 0) return 40010;
    if (d <= 0) return 40017;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long n = (long)t_q * n_pools;
    if (n == 0) return 0;
    if (n > 2147483647L) return 40012; // grid.x contract, retained for the reference fallback
    int rc;
    if (t_q >= 128) {
        rc = memra_kpool_score_launch<8, 16, 8, 8, 16>(q, pool_keys, hw, score, t_q, heads, d,
                                                       n_pools, pool, first_pos, qk_scale,
                                                       head_scale, stream);
    } else if (t_q >= 8) {
        rc = memra_kpool_score_launch<8, 16, 2, 8, 16>(q, pool_keys, hw, score, t_q, heads, d,
                                                       n_pools, pool, first_pos, qk_scale,
                                                       head_scale, stream);
    } else if (n_pools >= MLA_KPOOL_SMALL_TILE_MIN_POOLS) {
        rc = memra_kpool_score_launch<64, 1, 1, 1, 16>(q, pool_keys, hw, score, t_q, heads, d,
                                                       n_pools, pool, first_pos, qk_scale,
                                                       head_scale, stream);
    } else {
        // Measured crossover, see the note above: the reference kernel wins this size.
        return memra_mla_kpool_score_ref_f32(q, pool_keys, hw, score, t_q, heads, d, n_pools,
                                             pool, first_pos, qk_scale, head_scale, stream_v);
    }
    if (rc != 0) {
        return memra_mla_kpool_score_ref_f32(q, pool_keys, hw, score, t_q, heads, d, n_pools,
                                             pool, first_pos, qk_scale, head_scale, stream_v);
    }
    MLA_ERR();
    return 0;
}

// ---------------------------------------------------------------- selection: the ORDER contract
//
// Top-`select_k` pools per query, expanded to raw cache rows, plus the always-selected tail.
// Emits `idx[i][0..width]`, ASCENDING, -1 padded.
//
// TIE-BREAK IS PART OF THE PROGRAM, not an implementation detail. ReLU zeroes every head whose
// query-pool dot is non-positive, so exact 0.0 ties across pools are COMMON, not rare. The oracle
// (`memra_reference::kpool_allowed_tokens`) sorts score-DESCENDING then pool-index-ASCENDING and
// takes the first `select_k`. Both kernels below reproduce that total order exactly, or the
// selection-parity gate flaps on which zero-scoring pools happen to be picked.
//
// TWO IMPLEMENTATIONS, ONE ANSWER:
//   * `..._select_ref_kernel` — the correctness-grade original, `select_k` rounds of block-wide
//     "largest candidate strictly BELOW the previous pick". O(select_k * n_pools / threads).
//   * `..._select_kernel` — the SHIPPED one, a radix select on a 64-bit order key.
//     O(8 * n_pools / threads).
// They are gated byte-identical against each other at serving-scale shapes
// (`gpu_kpool_radix_selection_is_byte_identical_to_the_reference_kernel`), which is why the fast
// one carries no flag: it is not a second selection program, it is the same one computed faster.
//
// A 64-BIT ORDER KEY, and why the tie-break survives.
//   key(p) = (desc32(score[p]) << 32) | (uint32)p
// where `desc32` is the standard IEEE monotone bit map (`u ^= (u >> 31) ? ~0u : 0x80000000u`)
// composed with a bitwise NOT, so it is a strictly DECREASING injection from finite f32 to u32.
// Then, for finite scores:
//   key(p) < key(q)  <=>  score[p] > score[q], or (score[p] == score[q] and p < q)
// which is the oracle's comparator verbatim. The `select_k` smallest keys are therefore exactly
// the oracle's `select_k` selected pools, ties included; and because the low 32 bits are the pool
// index, every key is DISTINCT, so "the select_k-th smallest key" is unambiguous and radix select
// — an EXACT selection algorithm, not an approximation — returns it. Pool indices are unique per
// row, so the answer does not depend on thread scheduling or histogram atomic order.
//
// -0.0: `desc32` canonicalizes it to +0.0 first. The two compare EQUAL as floats (which is the
// comparator the oracle and the ref kernel use) but have different bit patterns, so keying them
// apart would order a -0.0 pool ahead of a +0.0 pool that the oracle calls tied.
//
// NON-FINITE scores are SKIPPED, in both kernels: -INFINITY is how the score kernel marks a pool
// the query cannot see. NaN is skipped too — that is the shipped ref kernel's behaviour and it is
// preserved deliberately (the oracle's `partial_cmp(..).unwrap_or(Equal)` would instead treat NaN
// as tied with everything; no gated fixture reaches it, and matching the ref kernel is what keeps
// the parity gates meaningful).
__device__ __forceinline__ unsigned memra_kpool_desc32(float s) {
    if (s == 0.0f) s = 0.0f; // canonicalize -0.0; true for both zeros, false for everything else
    unsigned u = __float_as_uint(s);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); // ascending float order
    return ~u;                                     // descending float order
}

__device__ __forceinline__ unsigned long long memra_kpool_key(float s, int p) {
    return ((unsigned long long)memra_kpool_desc32(s) << 32) | (unsigned long long)(unsigned)p;
}

// Two passes rather than a destructive extraction:
//   1. `select_k` rounds of block-wide "largest candidate strictly BELOW the previous pick" in
//      that same total order, which leaves the score row intact and lands on the select_k-th
//      element (`thr`). Fewer valid pools than the budget simply exhausts the rounds early.
//   2. A pool is selected iff it is finite and (score > thr.s) or (score == thr.s and p <= thr.p)
//      — the membership test that total order induces. Threads own CONTIGUOUS pool ranges, so a
//      count + exclusive scan writes the survivors in ascending pool order without a sort.
//
// COST: O(select_k * n_pools / threads) per query. At the shipped budget (top_k 2048, pool 4 ->
// select_k 512) and a 1M context (n_pools 262144) that is ~0.5M block iterations per query — far
// too slow to serve. RETAINED as the in-tree oracle for the radix kernel: it is the definition of
// the order the fast path has to reproduce, and it is itself gated against the Rust reference.
extern "C" __global__ void memra_mla_kpool_select_ref_kernel(const float* __restrict__ score,
                                                             int* __restrict__ idx, int n_pools,
                                                             int pool, int select_k, int width,
                                                             int first_pos, int always_tail) {
    __shared__ float sh_s[MLA_THREADS];
    __shared__ int sh_p[MLA_THREADS];
    __shared__ int sh_n[MLA_THREADS];
    __shared__ float thr_s;
    __shared__ int thr_p;
    __shared__ int thr_live;
    __shared__ int sh_total;

    int t = blockIdx.x;
    const float* row = score + (long)t * n_pools;
    int* out = idx + (long)t * width;
    int tid = threadIdx.x;

    if (tid == 0) {
        thr_s = INFINITY;
        thr_p = -1;
        thr_live = 0;
    }
    __syncthreads();

    for (int round = 0; round < select_k; ++round) {
        float prev_s = thr_s;
        int prev_p = thr_p;
        float best_s = -INFINITY;
        int best_p = -1;
        for (int p = tid; p < n_pools; p += MLA_THREADS) {
            float s = row[p];
            if (!isfinite(s)) continue;
            // strictly below (prev_s, prev_p) in the order "score desc, index asc"
            if (!(s < prev_s || (s == prev_s && p > prev_p))) continue;
            if (best_p < 0 || s > best_s || (s == best_s && p < best_p)) {
                best_s = s;
                best_p = p;
            }
        }
        sh_s[tid] = best_s;
        sh_p[tid] = best_p;
        __syncthreads();
        for (int step = MLA_THREADS / 2; step > 0; step >>= 1) {
            if (tid < step) {
                int op = sh_p[tid + step];
                if (op >= 0) {
                    int mp = sh_p[tid];
                    float os = sh_s[tid + step], ms = sh_s[tid];
                    if (mp < 0 || os > ms || (os == ms && op < mp)) {
                        sh_s[tid] = os;
                        sh_p[tid] = op;
                    }
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            if (sh_p[0] >= 0) {
                thr_s = sh_s[0];
                thr_p = sh_p[0];
                thr_live = 1;
            }
        }
        __syncthreads();
        if (sh_p[0] < 0) break; // fewer valid pools than the budget
    }

    // Pass 2: membership + ascending emit over CONTIGUOUS per-thread pool ranges.
    int chunk = (n_pools + MLA_THREADS - 1) / MLA_THREADS;
    int lo = tid * chunk;
    int hi = lo + chunk;
    if (lo > n_pools) lo = n_pools;
    if (hi > n_pools) hi = n_pools;
    float ts = thr_s;
    int tp = thr_p;
    int live = thr_live;
    int mine = 0;
    for (int p = lo; p < hi; ++p) {
        float s = row[p];
        if (!isfinite(s) || !live) continue;
        if (s > ts || (s == ts && p <= tp)) ++mine;
    }
    sh_n[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int j = 0; j < MLA_THREADS; ++j) {
            int c = sh_n[j];
            sh_n[j] = run;
            run += c;
        }
        sh_total = run;
    }
    __syncthreads();
    int slot = sh_n[tid];
    for (int p = lo; p < hi; ++p) {
        float s = row[p];
        if (!isfinite(s) || !live) continue;
        if (!(s > ts || (s == ts && p <= tp))) continue;
        for (int j = 0; j < pool; ++j) out[slot * pool + j] = p * pool + j;
        ++slot;
    }
    __syncthreads();

    int filled = sh_total * pool;
    if (always_tail) {
        int visible = first_pos + t + 1;
        int tail = visible % pool;
        for (int j = tid; j < tail; j += MLA_THREADS) out[filled + j] = visible - tail + j;
        filled += tail;
    }
    for (int j = filled + tid; j < width; j += MLA_THREADS) out[j] = -1;
}

// RADIX SELECT on the 64-bit order key documented above. Structure mirrors the ref kernel: a
// threshold phase, then the SAME contiguous-range membership + ascending emit + tail + pad.
//
// Threshold phase — 8 MSB-first passes of 8 bits over the key:
//   pass b builds a 256-bin shared histogram of byte `b` across the finite pools whose higher
//   bytes already match the resolved prefix, then walks the bins in ascending order to find the
//   one holding rank `k`; the prefix gains that byte and `k` drops by the bins before it. After
//   byte 0 the prefix IS the select_k-th smallest key. When a chosen bin holds exactly ONE pool
//   the descent stops early and one scan pass reads that pool's full key.
//   Pass 7 also totals the finite pools: `n_fin == 0` selects nothing (only the tail follows),
//   and `n_fin < select_k` clamps the rank so every visible pool is selected — the same two
//   degenerate answers the ref kernel's round exhaustion gives.
//
// Membership is then `key(p) <= thr`, which is `(score > thr_s) || (score == thr_s && p <=
// thr_p)` under the order isomorphism — the ref kernel's test, in key space.
//
// COST: O(8 * n_pools / threads) per query, INDEPENDENT of select_k. At the shipped budget and a
// 1M context that is 64x fewer block iterations than the ref kernel (8 rather than 512 sweeps).
extern "C" __global__ void memra_mla_kpool_select_kernel(const float* __restrict__ score,
                                                         int* __restrict__ idx, int n_pools,
                                                         int pool, int select_k, int width,
                                                         int first_pos, int always_tail) {
    __shared__ unsigned sh_hist[256];
    __shared__ int sh_n[MLA_THREADS];
    __shared__ unsigned long long sh_prefix;
    __shared__ int sh_k;      // remaining 1-based rank inside the resolved prefix
    __shared__ int sh_live;   // 0 = no finite candidate at all
    __shared__ int sh_unique;              // the chosen bin holds one pool: finish with a scan
    __shared__ unsigned long long sh_found; // that pool's FULL key, from the scan
    __shared__ int sh_total;

    int t = blockIdx.x;
    const float* row = score + (long)t * n_pools;
    int* out = idx + (long)t * width;
    int tid = threadIdx.x;

    if (tid == 0) {
        sh_prefix = 0ull;
        sh_k = select_k;
        sh_live = (select_k > 0 && n_pools > 0) ? 1 : 0;
        sh_unique = 0;
    }
    __syncthreads();

    for (int b = 7; b >= 0 && sh_live && !sh_unique; --b) {
        for (int j = tid; j < 256; j += MLA_THREADS) sh_hist[j] = 0u;
        unsigned long long pre = sh_prefix;
        int shift = 8 * b;
        // Bytes ABOVE `b`, i.e. the part of the key already resolved. `~0ull << 64` is undefined,
        // so the first pass (nothing resolved yet) takes an all-zero mask explicitly.
        unsigned long long mask = (b == 7) ? 0ull : (~0ull << (shift + 8));
        __syncthreads();
        for (int p = tid; p < n_pools; p += MLA_THREADS) {
            float s = row[p];
            if (!isfinite(s)) continue;
            unsigned long long key = memra_kpool_key(s, p);
            if ((key & mask) != pre) continue;
            atomicAdd(&sh_hist[(unsigned)((key >> shift) & 0xffull)], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            if (b == 7) {
                unsigned n_fin = 0;
                for (int j = 0; j < 256; ++j) n_fin += sh_hist[j];
                if (n_fin == 0u) sh_live = 0;                      // nothing causally visible
                else if ((unsigned)sh_k > n_fin) sh_k = (int)n_fin; // budget exceeds the candidates
            }
            if (sh_live) {
                unsigned run = 0, chosen = 0;
                int bin = 0;
                for (int j = 0; j < 256; ++j) {
                    unsigned c = sh_hist[j];
                    if (c == 0u) continue;
                    if (run + c >= (unsigned)sh_k) {
                        bin = j;
                        chosen = c;
                        break;
                    }
                    run += c;
                }
                sh_k -= (int)run;
                sh_prefix = pre | ((unsigned long long)(unsigned)bin << shift);
                sh_unique = (chosen == 1u && b > 0) ? 1 : 0;
            }
        }
        __syncthreads();
        if (sh_live && sh_unique) {
            // Exactly one pool matches the prefix down to byte `b`, so exactly one thread stores
            // and the write is race-free. `shift >= 8` here, so the shift is always defined.
            // The store lands in `sh_found`, NOT in `sh_prefix`: threads read `sh_prefix` into
            // `pre2` here with no barrier between them, so writing it back would let a fast warp
            // corrupt the comparand a slow warp has not loaded yet.
            unsigned long long pre2 = sh_prefix;
            unsigned long long mask2 = ~0ull << shift;
            for (int p = tid; p < n_pools; p += MLA_THREADS) {
                float s = row[p];
                if (!isfinite(s)) continue;
                unsigned long long key = memra_kpool_key(s, p);
                if ((key & mask2) == pre2) sh_found = key;
            }
            __syncthreads();
            if (tid == 0) sh_prefix = sh_found;
            __syncthreads();
        }
    }

    unsigned long long thr = sh_prefix;
    int live = sh_live;

    // Membership + ascending emit over CONTIGUOUS per-thread pool ranges — identical in structure
    // to the ref kernel, with `key(p) <= thr` standing in for its float comparator pair.
    int chunk = (n_pools + MLA_THREADS - 1) / MLA_THREADS;
    int lo = tid * chunk;
    int hi = lo + chunk;
    if (lo > n_pools) lo = n_pools;
    if (hi > n_pools) hi = n_pools;
    int mine = 0;
    for (int p = lo; p < hi; ++p) {
        float s = row[p];
        if (!isfinite(s) || !live) continue;
        if (memra_kpool_key(s, p) <= thr) ++mine;
    }
    sh_n[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int j = 0; j < MLA_THREADS; ++j) {
            int c = sh_n[j];
            sh_n[j] = run;
            run += c;
        }
        sh_total = run;
    }
    __syncthreads();
    int slot = sh_n[tid];
    for (int p = lo; p < hi; ++p) {
        float s = row[p];
        if (!isfinite(s) || !live) continue;
        if (memra_kpool_key(s, p) > thr) continue;
        for (int j = 0; j < pool; ++j) out[slot * pool + j] = p * pool + j;
        ++slot;
    }
    __syncthreads();

    int filled = sh_total * pool;
    if (always_tail) {
        int visible = first_pos + t + 1;
        int tail = visible % pool;
        for (int j = tid; j < tail; j += MLA_THREADS) out[filled + j] = visible - tail + j;
        filled += tail;
    }
    for (int j = filled + tid; j < width; j += MLA_THREADS) out[j] = -1;
}

// Shared argument audit for both selection launchers.
static inline int memra_mla_kpool_select_check(int pool, int select_k, int width,
                                               int always_tail) {
    if (pool <= 0 || pool > MLA_MAX_POOL) return 40010;
    // A no-tail indexer leaves every query before the first complete pool with an EMPTY candidate
    // set, which is a division by a zero softmax denominator here and an explicit refusal in the
    // oracle. No shipped glm5_next config sets it (`index_kpool_always_select_tail: true`), so
    // this fails loudly rather than carrying a NaN arm nothing gates.
    if (!always_tail) return 40013;
    if (width < select_k * pool + (always_tail ? pool - 1 : 0)) return 40014;
    return 0;
}

extern "C" int memra_mla_kpool_select_f32(const float* score, int* idx, int t_q, int n_pools,
                                          int pool, int select_k, int width, int first_pos,
                                          int always_tail, void* stream_v) {
    int bad = memra_mla_kpool_select_check(pool, select_k, width, always_tail);
    if (bad) return bad;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (t_q == 0) return 0;
    memra_mla_kpool_select_kernel<<<(unsigned)t_q, MLA_THREADS, 0, stream>>>(
        score, idx, n_pools, pool, select_k, width, first_pos, always_tail);
    MLA_ERR();
    return 0;
}

// The correctness-grade twin. NOT on any serving path: it exists so the gate can prove the radix
// kernel byte-identical to the order definition at shapes the micro fixture cannot reach.
extern "C" int memra_mla_kpool_select_ref_f32(const float* score, int* idx, int t_q, int n_pools,
                                              int pool, int select_k, int width, int first_pos,
                                              int always_tail, void* stream_v) {
    int bad = memra_mla_kpool_select_check(pool, select_k, width, always_tail);
    if (bad) return bad;
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (t_q == 0) return 0;
    memra_mla_kpool_select_ref_kernel<<<(unsigned)t_q, MLA_THREADS, 0, stream>>>(
        score, idx, n_pools, pool, select_k, width, first_pos, always_tail);
    MLA_ERR();
    return 0;
}

// The gathered twin of `memra_mla_attn_absorbed_kernel`: identical score/softmax/accumulate body,
// with the contiguous `0..visible` cache walk replaced by the indexer's per-query index list
// (-1 = empty slot). The list is shared across heads because the indexer mixes heads BEFORE
// selecting — one selection per query, not per (query, head).
extern "C" __global__ void memra_mla_attn_gathered_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ o_lat,
    int n_head, int kv_rank, int d_rope, int n_slots, float scale) {
    __shared__ float s_q[MLA_MAX_RANK];
    __shared__ float s_qp[MLA_MAX_ROPE];
    __shared__ float s_acc[MLA_MAX_RANK];
    __shared__ float s_score[MLA_WARPS];
    __shared__ int s_row[MLA_WARPS];

    int blk = blockIdx.x; // i * n_head + h
    int i = blk / n_head;
    int width = kv_rank + d_rope;
    const int* row_idx = idx + (long)i * n_slots;

    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
        s_q[l] = q_lat[(long)blk * kv_rank + l];
        s_acc[l] = 0.0f;
    }
    for (int p = threadIdx.x; p < d_rope; p += blockDim.x)
        s_qp[p] = q_pe[(long)blk * d_rope + p];
    __syncthreads();

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float m = -FLT_MAX;
    float dsum = 0.0f;

    for (int s0 = 0; s0 < n_slots; s0 += MLA_WARPS) {
        int s = s0 + warp;
        int t = (s < n_slots) ? row_idx[s] : -1;
        float part = 0.0f;
        if (t >= 0) {
            const float* row = cache + (long)t * width;
            for (int l = lane; l < kv_rank; l += 32) part += s_q[l] * row[l];
            for (int p = lane; p < d_rope; p += 32) part += s_qp[p] * row[kv_rank + p];
        }
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffffu, part, off);
        if (lane == 0) {
            s_score[warp] = (t >= 0) ? part * scale : -FLT_MAX;
            s_row[warp] = t;
        }
        __syncthreads();

        float tmax = -FLT_MAX;
        for (int w = 0; w < MLA_WARPS; ++w)
            if (s_row[w] >= 0) tmax = fmaxf(tmax, s_score[w]);
        float mnew = fmaxf(m, tmax);
        float rescale = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
        float tsum = 0.0f;
        if (mnew > -FLT_MAX)
            for (int w = 0; w < MLA_WARPS; ++w)
                if (s_row[w] >= 0) tsum += expf(s_score[w] - mnew);
        dsum = dsum * rescale + tsum;

        for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
            float a = s_acc[l] * rescale;
            if (mnew > -FLT_MAX)
                for (int w = 0; w < MLA_WARPS; ++w) {
                    int tt = s_row[w];
                    if (tt < 0) continue;
                    a += expf(s_score[w] - mnew) * cache[(long)tt * width + l];
                }
            s_acc[l] = a;
        }
        m = mnew;
        __syncthreads();
    }

    float inv = 1.0f / dsum;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x)
        o_lat[(long)blk * kv_rank + l] = s_acc[l] * inv;
}

extern "C" int memra_mla_attn_gathered_f32(const float* q_lat, const float* q_pe,
                                           const float* cache, const int* idx, float* o_lat,
                                           int n_head, int kv_rank, int d_rope, int t_q,
                                           int n_slots, float scale, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (n_slots <= 0) return 40015; // an empty candidate list is a zero softmax denominator
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_attn_gathered_kernel<<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope, n_slots, scale);
    MLA_ERR();
    return 0;
}
