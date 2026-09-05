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
#include <cuda_bf16.h>
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

// Live-slot twin (lane/mla-live-len-20260905): the row offset comes from the decode-graph door's
// device position word (`pos_d[0]`, the same word the KDA runs read) instead of a launch scalar,
// so the append can sit inside a captured graph and replay at the next slot. Same element
// mapping, same store: bit-identical to the scalar kernel at slot == pos_d[0].
extern "C" __global__ void memra_mla_append_latent_live_kernel(float* __restrict__ cache,
                                                               const float* __restrict__ c_kv,
                                                               const float* __restrict__ k_pe,
                                                               const int* __restrict__ pos_d,
                                                               int t, int kv_rank, int d_rope) {
    int slot = pos_d[0];
    int width = kv_rank + d_rope;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)t * width) return;
    int row = (int)(i / width), col = (int)(i % width);
    float v = (col < kv_rank) ? c_kv[(long)row * kv_rank + col]
                              : k_pe[(long)row * d_rope + (col - kv_rank)];
    cache[(long)(slot + row) * width + col] = v;
}
extern "C" int memra_mla_append_latent_live_f32(float* cache, const float* c_kv, const float* k_pe,
                                                const int* pos_d, int t, int kv_rank, int d_rope,
                                                void* stream_v) {
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)t * (kv_rank + d_rope);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_append_latent_live_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        cache, c_kv, k_pe, pos_d, t, kv_rank, d_rope);
    MLA_ERR();
    return 0;
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

// ------------------------------------------- coalesced warp-per-row twins (lane/mla-coalesce)
//
// THE DEFECT, and it is orthogonal to the decode-split twins below. `memra_mla_absorb_q_kernel`
// gives output row `l` to THREAD `l` and has that thread walk its row serially:
//
//     for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
//         const float* row = w + (long)l * d_nope;
//         for (int p = 0; p < d_nope; ++p) acc += smem[p] * row[p];
//
// At any step `p` the 32 lanes of a warp read `w[l*d_nope + p]` for 32 consecutive `l`, which
// are `d_nope` floats apart = 1 KB on the served artifact (qk_nope_head_dim 256). Every lane's
// load lands in its own sector: a warp that
// could pull ONE 128-byte transaction pulls 32. `memra_mla_decompress_v_kernel` is the same
// loop with the roles swapped and a `kv_rank` float = 2 KB stride, which is worse.
//
// Measured on 2x B200 SXM (GLM-5.3-Flash, current best posture, nsys 2026-09-03): the pair is
// 70.8 + 70.6 us per launch x 11 MLA layers = 1.56 ms of an 18.44 ms token. On the served
// geometry (num_attention_heads 64, kv_lora_rank 512, qk_nope_head_dim 256, v_head_dim 256) it
// reads 33.6 MB of f32 wk_b and 33.6 MB of wv_b PER LAYER, so 738 MB per token. At 8 TB/s that
// is 0.092 ms. It takes 1.56 ms: 5.9% of roofline, a 17x gap, and this access pattern is the
// mechanism rather than another sighting of the 11-34% band.
//
// THE FIX: a WARP owns an output row instead of a thread. Lane k reads `row[p]` at
// p = k, k+32, ..., so the warp's 32 loads are 32 CONSECUTIVE floats = one 128 B transaction,
// and the row finishes with a shuffle reduction.
//
// NUMERIC CLASS `mla_warp_row_reduce`, named rather than hidden: the per-output sum is no
// longer one thread's serial ascending-index dot. It is 32 lane-partial sums (each ascending,
// stride 32) combined by a shuffle tree. Same terms, same values, different association — so
// this is NOT bit-identical to the shipped kernels and it is NOT the decode-split twins'
// contract, which deliberately kept the serial dot so it could claim bit identity. Door
// MEMRA_MLA_COALESCE, default OFF, and it needs a greedy tape plus an argmax gate.
//
// COMPOSES WITH THE SPLIT rather than competing with it: the two fix different halves. The
// split raises the GRID (t_q*n_head blocks -> ~1024) and leaves the loads uncoalesced; this
// fixes the LOADS and leaves the grid alone. So these kernels take the same `split`/`chunk`
// output-range partition, and `split == 1` is the unsplit case.

__device__ __forceinline__ float mla_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

extern "C" __global__ void memra_mla_absorb_q_wp_kernel(const float* __restrict__ q_nope,
                                                        const float* __restrict__ wk_b,
                                                        float* __restrict__ q_lat, int n_head,
                                                        int d_nope, int kv_rank, int split) {
    extern __shared__ float smem[];
    int blk = blockIdx.x / split;
    int chunk = blockIdx.x % split;
    int h = blk % n_head;
    const float* qn = q_nope + (long)blk * d_nope;
    for (int p = threadIdx.x; p < d_nope; p += blockDim.x) smem[p] = qn[p];
    __syncthreads();
    const float* w = wk_b + (long)h * kv_rank * d_nope;
    int per = (kv_rank + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < kv_rank ? lo + per : kv_rank;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    for (int l = lo + warp; l < hi; l += nwarp) {
        const float* row = w + (long)l * d_nope;
        float acc = 0.0f;
        for (int p = lane; p < d_nope; p += 32) acc += smem[p] * row[p];
        acc = mla_warp_sum(acc);
        if (lane == 0) q_lat[(long)blk * kv_rank + l] = acc;
    }
}

extern "C" __global__ void memra_mla_decompress_v_wp_kernel(const float* __restrict__ o_lat,
                                                            const float* __restrict__ wv_b,
                                                            float* __restrict__ out, int n_head,
                                                            int d_v, int kv_rank, int split) {
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
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    for (int j = lo + warp; j < hi; j += nwarp) {
        const float* row = w + (long)j * kv_rank;
        float acc = 0.0f;
        for (int l = lane; l < kv_rank; l += 32) acc += row[l] * smem[l];
        acc = mla_warp_sum(acc);
        if (lane == 0) out[(long)blk * d_v + j] = acc;
    }
}

// ---------------------------------------------------------------- decompress_v + q8_1 epilogue
// (lane/mla-wo-zq8-20260905) The `_wp` decompress kernels with `wo`'s q8_1 pair emitted beside the
// f32 output. Each block owns `per = d_v / split` consecutive outputs of one (token, head) row; with
// per % 32 == 0 those are whole q8 blocks of the token's [n_head * d_v] wo input (block index
// (h*d_v + lo)/32 + b), so after the warps' dots land in shared memory one warp per q8 block runs
// quantize_q8_1's arithmetic (cu/qmatvec.cu:589: per-32 amax via shfl_xor, d = amax/127, id = 1/d,
// rint(v*id)) over the same f32 values the plain kernel writes. The dot itself is the `_wp` body
// verbatim (same per-lane order, same warp sum), so `out` is bit-identical to the plain kernel and
// the pair is bit-identical to quantize_q8_1 over it. Launchers refuse per % 32 != 0 / per > 256.
template <typename WT>
__device__ __forceinline__ float mla_wp_widen(WT v);
template <> __device__ __forceinline__ float mla_wp_widen<float>(float v) { return v; }
template <> __device__ __forceinline__ float mla_wp_widen<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename WT>
__device__ __forceinline__ void memra_mla_decompress_v_wp_zq8_body(
        const float* __restrict__ o_lat, const WT* __restrict__ wv_b, float* __restrict__ out,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_head, int d_v,
        int kv_rank, int split) {
    extern __shared__ float smem[];
    __shared__ float so[256];
    int blk = blockIdx.x / split;
    int chunk = blockIdx.x % split;
    int h = blk % n_head;
    const float* ol = o_lat + (long)blk * kv_rank;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) smem[l] = ol[l];
    __syncthreads();
    const WT* w = wv_b + (long)h * d_v * kv_rank;
    int per = (d_v + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < d_v ? lo + per : d_v;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    for (int j = lo + warp; j < hi; j += nwarp) {
        const WT* row = w + (long)j * kv_rank;
        float acc = 0.0f;
        for (int l = lane; l < kv_rank; l += 32) acc += mla_wp_widen<WT>(row[l]) * smem[l];
        acc = mla_warp_sum(acc);
        if (lane == 0) { out[(long)blk * d_v + j] = acc; so[j - lo] = acc; }
    }
    __syncthreads();
    int nq = (hi - lo) >> 5;   // whole q8 blocks in this block's range (launcher guarantees exact)
    for (int b = warp; b < nq; b += nwarp) {
        float v = so[b * 32 + lane];
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        long e = (long)blk * d_v + lo + b * 32;
        out_q[e + lane] = (signed char)__float2int_rn(v * id);
        if (lane == 0) out_d[e >> 5] = d;
    }
}
extern "C" __global__ void memra_mla_decompress_v_wp_zq8_kernel(
        const float* __restrict__ o_lat, const float* __restrict__ wv_b, float* __restrict__ out,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_head, int d_v,
        int kv_rank, int split) {
    memra_mla_decompress_v_wp_zq8_body<float>(o_lat, wv_b, out, out_q, out_d, n_head, d_v, kv_rank, split);
}
extern "C" __global__ void memra_mla_decompress_v_wp_bf16_zq8_kernel(
        const float* __restrict__ o_lat, const __nv_bfloat16* __restrict__ wv_b, float* __restrict__ out,
        signed char* __restrict__ out_q, float* __restrict__ out_d, int n_head, int d_v,
        int kv_rank, int split) {
    memra_mla_decompress_v_wp_zq8_body<__nv_bfloat16>(o_lat, wv_b, out, out_q, out_d, n_head, d_v, kv_rank, split);
}
static inline int mla_wp_zq8_shape_ok(int d_v, int split) {
    if (split < 1 || split > d_v) return 0;
    if (d_v % split != 0) return 0;
    int per = d_v / split;
    return (per % 32 == 0 && per <= 256) ? 1 : 0;
}
extern "C" int memra_mla_decompress_v_wp_zq8_f32(const float* o_lat, const float* wv_b, float* out,
                                                 signed char* out_q, float* out_d, int t_q,
                                                 int n_head, int d_v, int kv_rank, int split,
                                                 void* stream_v) {
    if (!mla_wp_zq8_shape_ok(d_v, split)) return 40004;
    if (kv_rank > MLA_MAX_RANK) return 40002;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_wp_zq8_kernel<<<(unsigned)blocks, MLA_THREADS, kv_rank * sizeof(float),
                                           stream>>>(o_lat, wv_b, out, out_q, out_d, n_head, d_v,
                                                     kv_rank, split);
    MLA_ERR();
    return 0;
}
extern "C" int memra_mla_decompress_v_wp_bf16_zq8(const float* o_lat, const unsigned short* wv_b,
                                                  float* out, signed char* out_q, float* out_d,
                                                  int t_q, int n_head, int d_v, int kv_rank,
                                                  int split, void* stream_v) {
    if (!mla_wp_zq8_shape_ok(d_v, split)) return 40004;
    if (kv_rank > MLA_MAX_RANK) return 40002;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_wp_bf16_zq8_kernel<<<(unsigned)blocks, MLA_THREADS,
                                                kv_rank * sizeof(float), stream>>>(
        o_lat, (const __nv_bfloat16*)wv_b, out, out_q, out_d, n_head, d_v, kv_rank, split);
    MLA_ERR();
    return 0;
}

// ---------------------------------------------------------------- BF16 absorb planes
// The `_wp` decode kernels with the weight plane read as BF16 and widened per element
// (lane/mla-absorb-bf16-20260905). Same per-lane element order and the same f32 products and
// adds as the f32 twins: where the resident f32 plane is itself a widening of BF16 (the B200
// hybrid mint ships kv_b_proj in BF16), the results are bit-identical at half the bytes.
extern "C" __global__ void memra_mla_absorb_q_wp_bf16_kernel(const float* __restrict__ q_nope,
                                                        const __nv_bfloat16* __restrict__ wk_b,
                                                        float* __restrict__ q_lat, int n_head,
                                                        int d_nope, int kv_rank, int split) {
    extern __shared__ float smem[];
    int blk = blockIdx.x / split;
    int chunk = blockIdx.x % split;
    int h = blk % n_head;
    const float* qn = q_nope + (long)blk * d_nope;
    for (int p = threadIdx.x; p < d_nope; p += blockDim.x) smem[p] = qn[p];
    __syncthreads();
    const __nv_bfloat16* w = wk_b + (long)h * kv_rank * d_nope;
    int per = (kv_rank + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < kv_rank ? lo + per : kv_rank;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    for (int l = lo + warp; l < hi; l += nwarp) {
        const __nv_bfloat16* row = w + (long)l * d_nope;
        float acc = 0.0f;
        for (int p = lane; p < d_nope; p += 32) acc += smem[p] * __bfloat162float(row[p]);
        acc = mla_warp_sum(acc);
        if (lane == 0) q_lat[(long)blk * kv_rank + l] = acc;
    }
}

extern "C" __global__ void memra_mla_decompress_v_wp_bf16_kernel(const float* __restrict__ o_lat,
                                                            const __nv_bfloat16* __restrict__ wv_b,
                                                            float* __restrict__ out, int n_head,
                                                            int d_v, int kv_rank, int split) {
    extern __shared__ float smem[];
    int blk = blockIdx.x / split;
    int chunk = blockIdx.x % split;
    int h = blk % n_head;
    const float* ol = o_lat + (long)blk * kv_rank;
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) smem[l] = ol[l];
    __syncthreads();
    const __nv_bfloat16* w = wv_b + (long)h * d_v * kv_rank;
    int per = (d_v + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < d_v ? lo + per : d_v;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int nwarp = blockDim.x >> 5;
    for (int j = lo + warp; j < hi; j += nwarp) {
        const __nv_bfloat16* row = w + (long)j * kv_rank;
        float acc = 0.0f;
        for (int l = lane; l < kv_rank; l += 32) acc += __bfloat162float(row[l]) * smem[l];
        acc = mla_warp_sum(acc);
        if (lane == 0) out[(long)blk * d_v + j] = acc;
    }
}


extern "C" int memra_mla_absorb_q_wp_bf16(const float* q_nope, const unsigned short* wk_b,
                                          float* q_lat, int t_q, int n_head, int d_nope,
                                          int kv_rank, int split, void* stream_v) {
    if (split < 1 || split > kv_rank) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_absorb_q_wp_bf16_kernel<<<(unsigned)blocks, MLA_THREADS, d_nope * sizeof(float),
                                        stream>>>(q_nope, (const __nv_bfloat16*)wk_b, q_lat,
                                                  n_head, d_nope, kv_rank, split);
    MLA_ERR();
    return 0;
}

extern "C" int memra_mla_decompress_v_wp_bf16(const float* o_lat, const unsigned short* wv_b,
                                              float* out, int t_q, int n_head, int d_v,
                                              int kv_rank, int split, void* stream_v) {
    if (split < 1 || split > d_v) return 40003;
    if (kv_rank > MLA_MAX_RANK) return 40002;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_wp_bf16_kernel<<<(unsigned)blocks, MLA_THREADS,
                                            kv_rank * sizeof(float), stream>>>(
        o_lat, (const __nv_bfloat16*)wv_b, out, n_head, d_v, kv_rank, split);
    MLA_ERR();
    return 0;
}

extern "C" int memra_mla_absorb_q_wp_f32(const float* q_nope, const float* wk_b, float* q_lat,
                                         int t_q, int n_head, int d_nope, int kv_rank, int split,
                                         void* stream_v) {
    if (split < 1 || split > kv_rank) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_absorb_q_wp_kernel<<<(unsigned)blocks, MLA_THREADS, d_nope * sizeof(float),
                                   stream>>>(q_nope, wk_b, q_lat, n_head, d_nope, kv_rank, split);
    MLA_ERR();
    return 0;
}

extern "C" int memra_mla_decompress_v_wp_f32(const float* o_lat, const float* wv_b, float* out,
                                             int t_q, int n_head, int d_v, int kv_rank, int split,
                                             void* stream_v) {
    if (split < 1 || split > d_v) return 40003;
    if (kv_rank > MLA_MAX_RANK) return 40002;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_decompress_v_wp_kernel<<<(unsigned)blocks, MLA_THREADS, kv_rank * sizeof(float),
                                       stream>>>(o_lat, wv_b, out, n_head, d_v, kv_rank, split);
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
// Body of the absorbed decode core, shared by the scalar-t_kv entry below and the live-length
// twin (lane/mla-live-len-20260905) that reads t_kv from the door's device position word. One
// body, two entries: the live entry is bit-identical to the scalar one at t_kv == pos_d[0] + t_q.
__device__ __forceinline__ void memra_mla_attn_absorbed_body(
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
extern "C" __global__ void memra_mla_attn_absorbed_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, float* __restrict__ o_lat, int n_head, int kv_rank,
    int d_rope, int t_q, int t_kv, float scale) {
    memra_mla_attn_absorbed_body(q_lat, q_pe, cache, o_lat, n_head, kv_rank, d_rope, t_q, t_kv, scale);
}
extern "C" __global__ void memra_mla_attn_absorbed_live_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, float* __restrict__ o_lat, int n_head, int kv_rank,
    int d_rope, int t_q, const int* __restrict__ pos_d, float scale) {
    int t_kv = pos_d[0] + t_q;
    memra_mla_attn_absorbed_body(q_lat, q_pe, cache, o_lat, n_head, kv_rank, d_rope, t_q, t_kv, scale);
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
// Live-length twin: t_kv = pos_d[0] + t_q on the device, so the launch geometry (t_q * n_head
// blocks) is fixed and the kernel can sit inside a captured graph. The host cannot check
// t_q <= t_kv here; the door owns that invariant (pos_d is the slot the queries append at).
extern "C" int memra_mla_attn_absorbed_live_f32(const float* q_lat, const float* q_pe,
                                                const float* cache, float* o_lat, int n_head,
                                                int kv_rank, int d_rope, int t_q,
                                                const int* pos_d, float scale, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_attn_absorbed_live_kernel<<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, o_lat, n_head, kv_rank, d_rope, t_q, pos_d, scale);
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

// Live twin (t = 1): the destination row is the door's device position word (`slot == pos`,
// the latent plane's own invariant that `memra_mla_append_latent_live_kernel` already rests on).
extern "C" __global__ void memra_mla_index_append_ring_live_kernel(
    float* __restrict__ plane, const float* __restrict__ a, const float* __restrict__ b,
    const int* __restrict__ pos_d, int wa, int wb, int rows) {
    int width = wa + wb;
    int col = (int)((long)blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= width) return;
    float v = (col < wa) ? a[col] : b[col - wa];
    int dst = pos_d[0];
    if (rows > 0) dst %= rows;
    plane[(long)dst * width + col] = v;
}

extern "C" int memra_mla_index_append_ring_live_f32(float* plane, const float* a, const float* b,
                                                    const int* pos_d, int t, int wa, int wb,
                                                    int rows, void* stream_v) {
    if (t < 0 || wa < 0 || wb < 0 || rows < 0) return 40017;
    if (t != 1) return 40030; // live twin: one row at pos_d[0]
    cudaStream_t stream = (cudaStream_t)stream_v;
    long tot = (long)(wa + wb);
    if (tot == 0) return 0;
    int threads = 256;
    long blocks = (tot + threads - 1) / threads;
    memra_mla_index_append_ring_live_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        plane, a, b, pos_d, wa, wb, rows);
    MLA_ERR();
    return 0;
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
__device__ __forceinline__ void memra_mla_kpool_pool_key_cell(const float* __restrict__ state,
                                                              const float* __restrict__ ape,
                                                              float* __restrict__ pool_keys,
                                                              long i, int pool, int d,
                                                              int state_rows) {
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

extern "C" __global__ void memra_mla_kpool_pool_keys_kernel(const float* __restrict__ state,
                                                            const float* __restrict__ ape,
                                                            float* __restrict__ pool_keys,
                                                            int pool_begin, int n_pools, int pool,
                                                            int d, int state_rows) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long span = (long)(n_pools - pool_begin) * d;
    if (i >= span) return;
    i += (long)pool_begin * d;
    memra_mla_kpool_pool_key_cell(state, ape, pool_keys, i, pool, d, state_rows);
}

// Live twin (t = 1, the state row at pos_d[0] already appended): builds the ONE pool this token
// completes, if any. `t_kv = pos + 1`; the pool `t_kv / pool - 1` is complete exactly when
// `t_kv % pool == 0`, and it is the same cell arithmetic the incremental host build runs for
// `[pools_ready, ready_now)` at t = 1 (a span of one pool or none). Fixed grid over `d`.
extern "C" __global__ void memra_mla_kpool_pool_keys_live_kernel(
    const float* __restrict__ state, const float* __restrict__ ape, float* __restrict__ pool_keys,
    const int* __restrict__ pos_d, int pool, int d, int state_rows) {
    int t_kv = pos_d[0] + 1;
    if (t_kv % pool != 0) return;
    int c = (int)((long)blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= d) return;
    long p = (long)(t_kv / pool) - 1;
    memra_mla_kpool_pool_key_cell(state, ape, pool_keys, p * d + c, pool, d, state_rows);
}

extern "C" int memra_mla_kpool_pool_keys_live_f32(const float* state, const float* ape,
                                                  float* pool_keys, const int* pos_d, int pool,
                                                  int d, int state_rows, void* stream_v) {
    if (pool <= 0 || pool > MLA_MAX_POOL) return 40010;
    if (d <= 0) return 40017;
    if (state_rows < 0 || (state_rows > 0 && state_rows % pool != 0)) return 40019;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int threads = 256;
    long blocks = ((long)d + threads - 1) / threads;
    memra_mla_kpool_pool_keys_live_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        state, ape, pool_keys, pos_d, pool, d, state_rows);
    MLA_ERR();
    return 0;
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

// Live twin of the reference scorer (t_q = 1): `first_pos` and `n_pools` from the door's
// position word; grid over the CAPACITY pool count, CTAs at or past the live count exit. Same
// per-pool arithmetic, so it is bit-identical to `memra_mla_kpool_score_ref_kernel` (and
// therefore to the tiled and head-blocked kernels, which are bit-identical to it) on
// `[0, n_pools)`. Serves the geometries the head-blocked live twin has no instantiation for.
extern "C" __global__ void memra_mla_kpool_score_ref_live_kernel(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int heads, int d,
    const int* __restrict__ pos_d, int pool, float qk_scale, float head_scale) {
    extern __shared__ float sh[];
    int first_pos = pos_d[0];
    int n_pools = (first_pos + 1) / pool;
    int p = (int)blockIdx.x;
    if (p >= n_pools) return;
    // t = 0: `visible_pools = (first_pos + 1) / pool = n_pools`, so every pool below n_pools is
    // visible and the reference kernel's -INFINITY arm never fires at t_q = 1.
    int h = threadIdx.x;
    if (h < heads) {
        const float* qr = q + (long)h * d;
        const float* kr = pool_keys + (long)p * d;
        float dot = 0.0f;
        for (int c = 0; c < d; ++c) dot += qr[c] * kr[c];
        sh[h] = fmaxf(dot * qk_scale, 0.0f) * (hw[h] * head_scale);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int hh = 0; hh < heads; ++hh) acc += sh[hh];
        score[p] = acc;
    }
}

extern "C" int memra_mla_kpool_score_ref_live_f32(const float* q, const float* pool_keys,
                                                  const float* hw, float* score, int t_q,
                                                  int heads, int d, const int* pos_d,
                                                  int n_pools_cap, int pool, float qk_scale,
                                                  float head_scale, void* stream_v) {
    if (heads <= 0 || heads > 1024) return 40011;
    if (pool <= 0) return 40010;
    if (t_q != 1) return 40030; // live twin: one query row
    cudaStream_t stream = (cudaStream_t)stream_v;
    if (n_pools_cap <= 0) return 0;
    if ((long)n_pools_cap > 2147483647L) return 40012;
    memra_mla_kpool_score_ref_live_kernel<<<(unsigned)n_pools_cap, heads,
                                            (size_t)heads * sizeof(float), stream>>>(
        q, pool_keys, hw, score, heads, d, pos_d, pool, qk_scale, head_scale);
    MLA_ERR();
    return 0;
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
// Body of the single-CTA selector, shared by the scalar-n_pools entry and the live-count twin
// (lane/mla-kpool-live-20260905): the grid is t_q blocks whatever n_pools is, so reading n_pools
// from the door's device word makes the launch capturable; same scan, same keys, same emit.
__device__ __forceinline__ void memra_mla_kpool_select_body(const float* __restrict__ score,
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
extern "C" __global__ void memra_mla_kpool_select_kernel(const float* __restrict__ score,
                                                         int* __restrict__ idx, int n_pools,
                                                         int pool, int select_k, int width,
                                                         int first_pos, int always_tail) {
    memra_mla_kpool_select_body(score, idx, n_pools, pool, select_k, width, first_pos, always_tail);
}
extern "C" __global__ void memra_mla_kpool_select_live_kernel(const float* __restrict__ score,
                                                              int* __restrict__ idx,
                                                              int* __restrict__ width_d,
                                                              const int* __restrict__ pos_d,
                                                              int pool, int select_k_cap,
                                                              int width_cap, int always_tail) {
    // t_q = 1: first_pos = pos_d[0], n_pools = (pos + 1) / pool, select_k = min(cap, n_pools)
    // (the host's IndexGeom::select_k), and the row is laid out at the CAPACITY width: the body
    // sentinel-fills [filled, width) with -1 and the gathered attention masks -1 rows, so the
    // first index_width(n_pools) entries are the scalar launch's and the rest are inert.
    int first_pos = pos_d[0];
    int n_pools = (first_pos + 1) / pool;
    int select_k = select_k_cap < n_pools ? select_k_cap : n_pools;
    // The token's TRUE index_width, published for the live attention twins: they walk exactly
    // this many slots, so their chunking and reduction order match the scalar launch at this
    // width rather than the capacity stride.
    if (threadIdx.x == 0) width_d[0] = select_k * pool + (always_tail ? pool - 1 : 0);
    memra_mla_kpool_select_body(score, idx, n_pools, pool, select_k, width_cap, first_pos,
                                always_tail);
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
extern "C" int memra_mla_kpool_select_live_f32(const float* score, int* idx, int* width_d,
                                               int t_q, const int* pos_d, int pool,
                                               int select_k_cap, int width_cap, int always_tail,
                                               void* stream_v) {
    int bad = memra_mla_kpool_select_check(pool, select_k_cap, width_cap, always_tail);
    if (bad) return bad;
    if (t_q != 1) return 40030; // live twin: one query row at pos_d[0]
    cudaStream_t stream = (cudaStream_t)stream_v;
    memra_mla_kpool_select_live_kernel<<<(unsigned)t_q, MLA_THREADS, 0, stream>>>(
        score, idx, width_d, pos_d, pool, select_k_cap, width_cap, always_tail);
    MLA_ERR();
    return 0;
}

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
__device__ __forceinline__ void memra_mla_attn_gathered_body(
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

extern "C" __global__ void memra_mla_attn_gathered_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ o_lat,
    int n_head, int kv_rank, int d_rope, int n_slots, float scale) {
    memra_mla_attn_gathered_body(q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope, n_slots,
                                 scale);
}

// Live-width twin (t_q = 1): `n_slots` is the token's true index_width published by the live
// selector (`width_d`), so the slot walk and its reduction order are the scalar launch's at that
// width; the capacity stride past it is never read.
extern "C" __global__ void memra_mla_attn_gathered_live_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ o_lat,
    int n_head, int kv_rank, int d_rope, const int* __restrict__ n_slots_d, float scale) {
    memra_mla_attn_gathered_body(q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope,
                                 n_slots_d[0], scale);
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

extern "C" int memra_mla_attn_gathered_live_f32(const float* q_lat, const float* q_pe,
                                                const float* cache, const int* idx, float* o_lat,
                                                int n_head, int kv_rank, int d_rope, int t_q,
                                                const int* n_slots_d, float scale,
                                                void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (t_q != 1) return 40030; // live twin: one query row (the idx stride is the live width)
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_attn_gathered_live_kernel<<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope, n_slots_d, scale);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------- B200 (sm_100a) t<=8 decode arm
//
// MEMRA_B200_MLA_DECODE_ARM (host seam mla_ffi.rs, default OFF). Census motivation
// (nsys, 2x B200 SXM sm_100a, GLM-5.3-Flash NVFP4, plain decode t=1): the 11 MLA/DSA sparse
// layers' `memra_mla_attn_gathered_kernel` costs 16 x 42.3 us/token. At t_q=1 the grid is
// `t_q * n_head` = 64 blocks (n_head=64 on the glm5_next/GLM-5.2 geometry) — head-parallel by
// construction (`blk = i * n_head + h`, ALREADY one CTA per head), but 64 CTAs on a B200 die
// (~132-148 SMs) is well under one wave, and per-block work is 256 threads running a serial,
// syncthreads-per-tile online-softmax fold with no float4/uint4 vectorization — a latency,
// not a throughput, bound.
//
// SPLIT DESIGN, output-range only (mirrors the absorb_q_split / decompress_v_split siblings
// above, decode-diet lever 4, MEMRA_MLA_DECODE_SPLIT). Unlike those two
// pure independent-output matvecs, attn_gathered's OUTPUT elements (o_lat[blk][l], l in
// 0..kv_rank) all share ONE softmax normalizer (m, dsum) computed by walking every tile of the
// SAME gathered slot list. Splitting the walk itself across CTAs (segment the slots, combine
// partials) would change the number and order of the online-softmax rescale ops relative to the
// unsplit kernel's single sequential fold — a DIFFERENT floating-point program, not a
// launch-geometry change, so that path is refused here on the bit-identity bar this lane
// requires (see mla-b200-decode-20260902 LANE.md "why not slot-split").
//
// What IS bit-identical by construction: splitting the OUTPUT WRITE RANGE the same way
// absorb_q_split / decompress_v_split do. Every split block runs the score/softmax tile loop
// IN FULL (full slot walk, same m/dsum sequence, same rounding — the exact per-tile combine
// code of `memra_mla_attn_gathered_kernel`, unmodified) so the shared normalizer is identical
// across every chunk; only the final per-l accumulate-and-write loop is restricted to
// `[lo, hi)`. Each kept output element's accumulate chain (rescale, then w-ascending
// `expf(...)*cache[...][l]` adds) is therefore the SAME sequence of operations as the unsplit
// kernel computes for that same l — WHICH block runs it changes, not the arithmetic — so this
// is bit-identical by the same argument as the absorb/decompress splits (asserted in
// `mla_decode_arm_gate.rs`).
//
// TRADE-OFF, stated plainly: this trades REDUNDANT tile-walk compute (the dominant cost, ~42.3
// us of the kernel) for more CTAs — every split factor `> 1` repeats the full slot walk that
// many times. Unlike the absorb/decompress splits (near-zero marginal cost per split), this is
// only a net win if the box is latency/occupancy-bound, which is a hardware question, not a
// code one, and the 2x B200 pair answered it 2026-09-02 (`mla-decode-arm-gate` device 0, N=5,
// bit-identical): split=2 wins at t_q=1 (564.6 -> 516.4 us) and LOSES at t_q=4 (665.3 ->
// 822.7 us), the DFlash2 spec-verify shape. The host policy (`mla_b200_split_for` in
// mla_ffi.rs) therefore reads the factor from the t_q-keyed table
// `MLA_B200_ATTN_GATHERED_SPLIT`, which splits at t_q=1 only and ships the unsplit kernel at
// every other width; the gate fails (`REGRESSION`) if a table cell is slower than shipped by
// more than 5% on a later run.
extern "C" __global__ void memra_mla_attn_gathered_split_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ o_lat,
    int n_head, int kv_rank, int d_rope, int n_slots, float scale, int split) {
    __shared__ float s_q[MLA_MAX_RANK];
    __shared__ float s_qp[MLA_MAX_ROPE];
    __shared__ float s_acc[MLA_MAX_RANK];
    __shared__ float s_score[MLA_WARPS];
    __shared__ int s_row[MLA_WARPS];

    int blk = blockIdx.x / split;   // i * n_head + h
    int chunk = blockIdx.x % split; // this block's OUTPUT slice of [0, kv_rank)
    int i = blk / n_head;
    int width = kv_rank + d_rope;
    const int* row_idx = idx + (long)i * n_slots;

    int per = (kv_rank + split - 1) / split;
    int lo = chunk * per;
    int hi = lo + per < kv_rank ? lo + per : kv_rank;

    // Full q load: the score dot needs the WHOLE kv_rank + d_rope vector regardless of which
    // output range this block owns.
    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) s_q[l] = q_lat[(long)blk * kv_rank + l];
    for (int p = threadIdx.x; p < d_rope; p += blockDim.x)
        s_qp[p] = q_pe[(long)blk * d_rope + p];
    // Accumulator only needs the owned range.
    for (int l = lo + threadIdx.x; l < hi; l += blockDim.x) s_acc[l] = 0.0f;
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

        for (int l = lo + threadIdx.x; l < hi; l += blockDim.x) {
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
    for (int l = lo + threadIdx.x; l < hi; l += blockDim.x)
        o_lat[(long)blk * kv_rank + l] = s_acc[l] * inv;
}

extern "C" int memra_mla_attn_gathered_split_f32(const float* q_lat, const float* q_pe,
                                                  const float* cache, const int* idx,
                                                  float* o_lat, int n_head, int kv_rank,
                                                  int d_rope, int t_q, int n_slots, float scale,
                                                  int split, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (n_slots <= 0) return 40015;
    if (split < 1 || split > kv_rank) return 40003;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head * split;
    if (blocks == 0) return 0;
    memra_mla_attn_gathered_split_kernel<<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope, n_slots, scale, split);
    MLA_ERR();
    return 0;
}

// ============================================================ B200 DSA decode door (sm_100a)
//
// MEMRA_B200_DSA_DECODE (host seam mla_ffi.rs, default OFF; docs/FLAGS.md row). Owner target
// 2026-09-02: 230 tok/s plain with the 1M window as the product. The roofline this door was
// built from is research/b200-dsa-decode-20260902/ROOFLINE.md; the numbers below are quoted
// from it rather than re-derived, and the geometry is GLM-5.3-Flash NVFP4 on 2x B200 SXM
// (148 SMs, 8 TB/s HBM3e, 70.5 TFLOP/s f32 FFMA -- there is NO tensor-core path for true f32
// on Blackwell, so SIMT FFMA is the ceiling every kernel here is measured against).
//
// WHAT THE ROOFLINE FOUND, in one paragraph. The latent cache is f32 in this checkout, and the
// DSA index list is shared across heads (the indexer mixes heads BEFORE selecting), so the
// gathered set is 2048 x 512 x 4 B = 4.00 MiB per layer per token -- L2-resident on this die,
// not an HBM problem. `memra_mla_attn_gathered_kernel` costs 726.2 us/layer against a 3.81 us
// FFMA floor (190x) and a 0.52 us HBM floor (1390x), and it is depth-FLAT because n_slots is
// pinned at the DSA top-k budget. The cycles go to: 24 `expf` per thread per 8-slot tile where
// 8 distinct values exist (98.4% redundant, 1.57M per CTA per layer); two barriers per tile
// with 8 warps and one CTA per SM to hide them; a PV accumulate that re-reads every gathered
// row from GLOBAL, scalar, after the score pass already read it. Head-batching ("read each row
// once, serve all 64 heads") is the WRONG fix at t_q=1: it would divide the only 64 CTAs the
// kernel has to save L2 traffic that is not the bottleneck.
//
// So this door does not re-shape the head axis. It (1) stages each tile's KV rows into shared
// memory once, per warp, with float4 loads, and serves BOTH the score dot and the PV accumulate
// from that staging; (2) hoists the 8 tile exponentials into registers so `tsum` and the
// accumulate share them; (3) offers a slot-split arm for the occupancy the head axis cannot
// give; and (4) replaces the decode pool scorer with one that uses the reuse axis the shipped
// tile ignores at t_q=1 -- heads.

// ---------------------------------------------------------------- 1. gathered attention, fast
//
// `memra_mla_attn_gathered_dsa_kernel` -- BIT-IDENTICAL to `memra_mla_attn_gathered_kernel`.
//
// Same grid (`t_q * n_head`, one CTA per (token, head)), same MLA_WARPS-wide slot tiles, same
// warp-per-slot lane stride, same 5-step `__shfl_down_sync` tree, same `tmax`/`rescale`/`tsum`
// online-softmax combine, same ascending-`w` accumulate. Every floating-point operation that
// produces an output element is the same operation on the same operands in the same order, so
// bit identity is a CONSTRUCTION, not a tolerance -- and `dsa-decode-gate` asserts it bytewise
// anyway rather than trusting the argument.
//
// The three things that change, none of them arithmetic:
//   * STAGING. Warp `w` copies its own slot's cache row into `s_kv[w]` with `float4` loads
//     (16 B/thread/instruction, fully coalesced inside the warp), then `__syncwarp()` and reads
//     it back with the SAME lane stride the shipped kernel uses against global. No
//     __syncthreads is added: the producer and consumer of `s_kv[w]` are the same warp.
//   * ONE READ PER ROW PER CTA. The PV accumulate reads `s_kv[w * width + l]` instead of a
//     second global trip through `cache[tt * width + l]`, so the row crosses the L2/SM boundary
//     ONCE per head instead of twice. The trailing `__syncthreads()` that already guarded
//     `s_acc`/`s_score` guards `s_kv` too; the barrier count per tile is unchanged at 2.
//   * EXP HOISTING. `pw[w] = expf(s_score[w] - mnew)` is evaluated ONCE per thread per tile into
//     registers and consumed by both the `tsum` fold and every iteration of the `l` loop. The
//     shipped kernel evaluates it 1 + kv_rank/blockDim.x times (3 at the glm5 shape) -- 24 per
//     thread per tile against 8 here. Identical values by determinism, so identical bits.
//
// SHARED MEMORY: static `s_q + s_qp + s_acc` = 9.2 KB, plus dynamic `s_kv` = MLA_WARPS * width *
// 4 B (16 KB at the glm5 width 512). The launcher refuses any geometry whose staging exceeds
// MLA_DSA_KV_SMEM_MAX so this never needs a dynamic-smem opt-in and never silently spills; a
// refused geometry falls back to the shipped kernel, which is what the host door does with a
// non-zero return.
#define MLA_DSA_KV_SMEM_MAX (32 * 1024)
// Slot-chunk ceiling for the warp-online arm: bounds `s_w[]` in the combine kernel.
#define MLA_DSA_MAX_CHUNKS 64

extern "C" __global__ void memra_mla_attn_gathered_dsa_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ o_lat,
    int n_head, int kv_rank, int d_rope, int n_slots, float scale) {
    __shared__ float s_q[MLA_MAX_RANK];
    __shared__ float s_qp[MLA_MAX_ROPE];
    __shared__ float s_acc[MLA_MAX_RANK];
    __shared__ float s_score[MLA_WARPS];
    __shared__ int s_row[MLA_WARPS];
    // __align__(16): this array is accessed through `float4` below, and CUDA
    // guarantees only the element type's alignment for a plain `extern __shared__`
    // declaration. Same convention as hybrid.cu's `__align__(16)` gdn_k2_smem and
    // mmq_fp8_blk.cu's `__align__(128)`: a construction, not a reliance on the
    // base alignment a given toolkit happens to hand out.
    extern __shared__ __align__(16) float s_kv[]; // [MLA_WARPS][width]

    int blk = blockIdx.x; // i * n_head + h
    int i = blk / n_head;
    int width = kv_rank + d_rope;
    int width4 = width >> 2; // the launcher guarantees width % 4 == 0
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
        float* krow = s_kv + (long)warp * width;
        if (t >= 0) {
            const float4* src = (const float4*)(cache + (long)t * width);
            float4* dst = (float4*)krow;
            for (int c = lane; c < width4; c += 32) dst[c] = src[c];
        }
        __syncwarp();
        float part = 0.0f;
        if (t >= 0) {
            const float* row = krow;
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
#pragma unroll
        for (int w = 0; w < MLA_WARPS; ++w)
            if (s_row[w] >= 0) tmax = fmaxf(tmax, s_score[w]);
        float mnew = fmaxf(m, tmax);
        float rescale = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
        // The whole point: 8 exponentials per thread per tile, not 8 + 8 * (kv_rank/blockDim.x).
        float pw[MLA_WARPS];
        float tsum = 0.0f;
        if (mnew > -FLT_MAX) {
#pragma unroll
            for (int w = 0; w < MLA_WARPS; ++w) {
                if (s_row[w] < 0) {
                    pw[w] = 0.0f;
                    continue;
                }
                pw[w] = expf(s_score[w] - mnew);
                tsum += pw[w];
            }
        }
        dsum = dsum * rescale + tsum;

        for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
            float a = s_acc[l] * rescale;
            if (mnew > -FLT_MAX)
#pragma unroll
                for (int w = 0; w < MLA_WARPS; ++w) {
                    if (s_row[w] < 0) continue;
                    a += pw[w] * s_kv[(long)w * width + l];
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

extern "C" int memra_mla_attn_gathered_dsa_f32(const float* q_lat, const float* q_pe,
                                               const float* cache, const int* idx, float* o_lat,
                                               int n_head, int kv_rank, int d_rope, int t_q,
                                               int n_slots, float scale, void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (n_slots <= 0) return 40015;
    int width = kv_rank + d_rope;
    // float4 staging: the row stride and every row start must be 16 B aligned. `cache` is a
    // device allocation base (256 B aligned), so `width % 4 == 0` is the whole condition.
    if (width % 4 != 0) return 40020;
    size_t smem = (size_t)MLA_WARPS * (size_t)width * sizeof(float);
    if (smem > (size_t)MLA_DSA_KV_SMEM_MAX) return 40021;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long blocks = (long)t_q * n_head;
    if (blocks == 0) return 0;
    memra_mla_attn_gathered_dsa_kernel<<<(unsigned)blocks, MLA_THREADS, smem, stream>>>(
        q_lat, q_pe, cache, idx, o_lat, n_head, kv_rank, d_rope, n_slots, scale);
    MLA_ERR();
    return 0;
}

// ------------------------------------------ 2. gathered attention, warp-online (NUMERIC CLASS)
//
// NUMERIC CLASS `dsa-warp-online-f32`. NOT bit-identical, and never claimed to be.
//
// WHAT THE 5090 CORRECTNESS RUN TAUGHT THIS LANE, recorded because it killed the first design.
// The single-pass kernel above is bit-identical and, measured, a small LOSS: 446 -> 482 us at
// t_q=1 and 846 -> 1125 us at t_q=4 (RTX 5090, release, N=3 interleaved, correctness rig, timing
// diagnostic only per the rig law). The reason is that its two claimed savings were already
// gone: `expf(s_score[w] - mnew)` is loop-invariant in the `l` loop, so nvcc had ALREADY hoisted
// it, and the second `cache[tt * width + l]` pass hits L1/L2, so staging the row through shared
// memory ADDS a full smem write and read per row and buys back only L1 hits. Bit identity, on
// this kernel, is the binding constraint: the shipped fold is already at a local optimum inside
// it. So the depth win has to come from a different PROGRAM, named as such.
//
// THE PROGRAM. One WARP owns one (token, head, slot-chunk) and holds the whole kv_rank-wide
// accumulator in REGISTERS, `J = kv_rank / 32` floats per lane. Per slot it loads the cache row
// ONCE into registers (float4, coalesced, `J/4` instructions per lane), uses those SAME
// registers for both the QK dot and the PV accumulate, and folds the online softmax
// warp-locally. Consequences, all of them the roofline's asks:
//   * EVERY KV ELEMENT IS READ FROM MEMORY EXACTLY ONCE and consumed twice from registers. The
//     shipped kernel reads it twice per head; the single-pass kernel reads it once and then
//     round-trips it through shared memory.
//   * ZERO `__syncthreads`. The shipped kernel pays two barriers per 8-slot tile, 512 per CTA,
//     with 8 warps and one CTA per SM to hide them. There is no barrier here at all: the fold is
//     warp-local and the cross-lane reduction is a 5-step `__shfl_xor_sync` butterfly (every
//     lane ends with the sum, so no broadcast either).
//   * TRANSCENDENTALS COLLAPSE. Two `expf` per slot per warp (the rescale and the new weight)
//     against the shipped kernel's ~196k per warp per layer: ~48x fewer.
//   * The slot chunk is the occupancy knob the head axis cannot provide: at t_q=1 there are 64
//     independent (token, head) outputs for 148 SMs, and `chunks` multiplies that directly.
//
// WHY IT IS NOT BIT-IDENTICAL, stated exactly. The shipped kernel folds in MLA_WARPS-wide tiles
// (one max and one rescale per 8 slots); this folds per SLOT (a max and a rescale each), and the
// combine then merges `chunks` partials. Both are the same sum in real arithmetic and neither is
// the other's rounding. It therefore ships under a NAMED class with an ARGMAX gate over every
// (token, head) latent row plus a reported maxdiff/max-relative bound (`dsa-decode-gate`), and
// behind `MEMRA_B200_DSA_DECODE=2`. Level 1 engages only bit-identical arms.
//
// `J` and `JP` are TEMPLATE parameters because `acc[J]`, `qv[J]`, `kv[J]` must live in
// registers: a runtime bound would put them in local memory and lose the entire kernel. A
// geometry with no instantiation returns 40023 and the host falls through to the shipped path.

template <int J, int JP>
__device__ __forceinline__ void memra_mla_dsa_attn_warp_body(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ part_m,
    float* __restrict__ part_d, float* __restrict__ part_acc, int n_head, int kv_rank,
    int d_rope, int n_slots, int chunks, int per, int pairs, float scale) {
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x % 32;
    const long g = (long)blockIdx.x * MLA_WARPS + warp; // one warp per (pair, chunk)
    if (g >= (long)pairs * chunks) return;
    const int blk = (int)(g / chunks);
    const int chunk = (int)(g % chunks);
    const int i = blk / n_head;
    const int width = kv_rank + d_rope;
    const int* row_idx = idx + (long)i * n_slots;

    int lo = chunk * per;
    int hi = lo + per < n_slots ? lo + per : n_slots;

    float qv[J];
#pragma unroll
    for (int j = 0; j < J; ++j) qv[j] = q_lat[(long)blk * kv_rank + lane + 32 * j];
    float qp[JP > 0 ? JP : 1];
#pragma unroll
    for (int j = 0; j < JP; ++j) {
        int p = lane + 32 * j;
        qp[j] = (p < d_rope) ? q_pe[(long)blk * d_rope + p] : 0.0f;
    }
    float acc[J];
#pragma unroll
    for (int j = 0; j < J; ++j) acc[j] = 0.0f;

    float m = -FLT_MAX;
    float dsum = 0.0f;

    for (int s = lo; s < hi; ++s) {
        int t = row_idx[s];
        if (t < 0) continue;
        const float* row = cache + (long)t * width;
        float kv[J];
#pragma unroll
        for (int j = 0; j < J; ++j) kv[j] = row[lane + 32 * j];
        float kp[JP > 0 ? JP : 1];
#pragma unroll
        for (int j = 0; j < JP; ++j) {
            int p = lane + 32 * j;
            kp[j] = (p < d_rope) ? row[kv_rank + p] : 0.0f;
        }
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < J; ++j) part += qv[j] * kv[j];
#pragma unroll
        for (int j = 0; j < JP; ++j) part += qp[j] * kp[j];
        // Butterfly, not a down-shift: every lane ends holding the full sum, so the accumulate
        // below needs no broadcast and no shared memory.
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);

        float sc = part * scale;
        float mnew = fmaxf(m, sc);
        float rescale = (m == -FLT_MAX) ? 0.0f : expf(m - mnew);
        float pwt = expf(sc - mnew);
        dsum = dsum * rescale + pwt;
#pragma unroll
        for (int j = 0; j < J; ++j) acc[j] = acc[j] * rescale + pwt * kv[j];
        m = mnew;
    }

    // UNNORMALIZED partials: the combine owns the single division, exactly as the shipped
    // kernel's one `1.0f / dsum` does.
    long pbase = (long)blk * chunks + chunk;
    if (lane == 0) {
        part_m[pbase] = m;
        part_d[pbase] = dsum;
    }
#pragma unroll
    for (int j = 0; j < J; ++j) part_acc[pbase * kv_rank + lane + 32 * j] = acc[j];
}

template <int J, int JP>
__global__ __launch_bounds__(MLA_THREADS) void memra_mla_dsa_attn_warp_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ part_m,
    float* __restrict__ part_d, float* __restrict__ part_acc, int n_head, int kv_rank,
    int d_rope, int n_slots, int chunks, int per, int pairs, float scale) {
    memra_mla_dsa_attn_warp_body<J, JP>(q_lat, q_pe, cache, idx, part_m, part_d, part_acc, n_head,
                                        kv_rank, d_rope, n_slots, chunks, per, pairs, scale);
}

// Live-width twin (t_q = 1): the slot count comes from the live selector's `width_d`, and the
// chunk span is derived from it ON THE DEVICE with the same `memra_mla_dsa_attn_chunk_span`
// arithmetic, so every chunk holds exactly the slots the scalar launch gives it at that width
// and the ascending-chunk combine merges the same partials. The grid is `pairs * chunks`, which
// does not depend on the width, so the launch is fixed-geometry (capturable).
template <int J, int JP>
__global__ __launch_bounds__(MLA_THREADS) void memra_mla_dsa_attn_warp_live_kernel(
    const float* __restrict__ q_lat, const float* __restrict__ q_pe,
    const float* __restrict__ cache, const int* __restrict__ idx, float* __restrict__ part_m,
    float* __restrict__ part_d, float* __restrict__ part_acc, int n_head, int kv_rank,
    int d_rope, const int* __restrict__ n_slots_d, int chunks, int pairs, float scale) {
    const int n_slots = n_slots_d[0];
    const int per = (n_slots + chunks - 1) / chunks;
    memra_mla_dsa_attn_warp_body<J, JP>(q_lat, q_pe, cache, idx, part_m, part_d, part_acc, n_head,
                                        kv_rank, d_rope, n_slots, chunks, per, pairs, scale);
}

// Merge `chunks` partials per (token, head), in ASCENDING chunk order so the class is
// deterministic across runs.
extern "C" __global__ void memra_mla_dsa_attn_combine_kernel(const float* __restrict__ part_m,
                                                             const float* __restrict__ part_d,
                                                             const float* __restrict__ part_acc,
                                                             float* __restrict__ o_lat,
                                                             int kv_rank, int chunks) {
    int blk = blockIdx.x;
    long pbase = (long)blk * chunks;

    __shared__ float s_w[MLA_DSA_MAX_CHUNKS]; // chunk weight exp(m_c - gm)
    __shared__ float s_inv;

    if (threadIdx.x == 0) {
        float gm = -FLT_MAX;
        for (int c = 0; c < chunks; ++c) {
            float mc = part_m[pbase + c];
            if (mc > -FLT_MAX) gm = fmaxf(gm, mc);
        }
        float den = 0.0f;
        for (int c = 0; c < chunks; ++c) {
            float mc = part_m[pbase + c];
            float w = (mc > -FLT_MAX && gm > -FLT_MAX) ? expf(mc - gm) : 0.0f;
            s_w[c] = w;
            den += part_d[pbase + c] * w;
        }
        s_inv = 1.0f / den;
    }
    __syncthreads();

    for (int l = threadIdx.x; l < kv_rank; l += blockDim.x) {
        float a = 0.0f;
        for (int c = 0; c < chunks; ++c) a += part_acc[(pbase + c) * kv_rank + l] * s_w[c];
        o_lat[(long)blk * kv_rank + l] = a * s_inv;
    }
}

/// Slot count per chunk the host must use to size the workspace and to launch. Exposed so the
/// Rust side cannot compute a different partition than the kernels walk.
extern "C" int memra_mla_dsa_attn_chunk_span(int n_slots, int chunks) {
    if (chunks <= 0) return 0;
    return (n_slots + chunks - 1) / chunks;
}

template <int J, int JP>
static void memra_dsa_warp_launch(const float* q_lat, const float* q_pe, const float* cache,
                                  const int* idx, float* part_m, float* part_d, float* part_acc,
                                  int n_head, int kv_rank, int d_rope, int n_slots, int chunks,
                                  int per, long pairs, float scale, cudaStream_t stream) {
    long warps = pairs * chunks;
    long blocks = (warps + MLA_WARPS - 1) / MLA_WARPS;
    memra_mla_dsa_attn_warp_kernel<J, JP><<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, idx, part_m, part_d, part_acc, n_head, kv_rank, d_rope, n_slots,
        chunks, per, (int)pairs, scale);
}

extern "C" int memra_mla_dsa_attn_split_f32(const float* q_lat, const float* q_pe,
                                            const float* cache, const int* idx, float* o_lat,
                                            float* part_m, float* part_d, float* part_acc,
                                            int n_head, int kv_rank, int d_rope, int t_q,
                                            int n_slots, int chunks, float scale,
                                            void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (n_slots <= 0) return 40015;
    if (chunks < 1 || chunks > MLA_DSA_MAX_CHUNKS) return 40022;
    if (kv_rank % 32 != 0) return 40023;
    int per = memra_mla_dsa_attn_chunk_span(n_slots, chunks);
    if (per <= 0) return 40022;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long pairs = (long)t_q * n_head;
    if (pairs == 0) return 0;

    const int j = kv_rank / 32;
    const int jp = (d_rope + 31) / 32;
    // Instantiations for the shipped geometries: kv_rank 512 (glm5_next, GLM-5.2) and 1024,
    // crossed with d_rope 0 (NoPE) and 64 (GLM-5.2 rope). Anything else takes the shipped path.
#define MLA_DSA_WARP_CASE(JJ, JPP)                                                             \
    if (j == (JJ) && jp == (JPP)) {                                                            \
        memra_dsa_warp_launch<JJ, JPP>(q_lat, q_pe, cache, idx, part_m, part_d, part_acc,       \
                                       n_head, kv_rank, d_rope, n_slots, chunks, per, pairs,   \
                                       scale, stream);                                         \
    } else
    MLA_DSA_WARP_CASE(16, 0)
    MLA_DSA_WARP_CASE(16, 2)
    MLA_DSA_WARP_CASE(32, 0)
    MLA_DSA_WARP_CASE(32, 2) {
        return 40023;
    }
#undef MLA_DSA_WARP_CASE
    MLA_ERR();
    memra_mla_dsa_attn_combine_kernel<<<(unsigned)pairs, MLA_THREADS, 0, stream>>>(
        part_m, part_d, part_acc, o_lat, kv_rank, chunks);
    MLA_ERR();
    return 0;
}

template <int J, int JP>
static void memra_dsa_warp_live_launch(const float* q_lat, const float* q_pe, const float* cache,
                                       const int* idx, float* part_m, float* part_d,
                                       float* part_acc, int n_head, int kv_rank, int d_rope,
                                       const int* n_slots_d, int chunks, long pairs, float scale,
                                       cudaStream_t stream) {
    long warps = pairs * chunks;
    long blocks = (warps + MLA_WARPS - 1) / MLA_WARPS;
    memra_mla_dsa_attn_warp_live_kernel<J, JP><<<(unsigned)blocks, MLA_THREADS, 0, stream>>>(
        q_lat, q_pe, cache, idx, part_m, part_d, part_acc, n_head, kv_rank, d_rope, n_slots_d,
        chunks, (int)pairs, scale);
}

extern "C" int memra_mla_dsa_attn_split_live_f32(const float* q_lat, const float* q_pe,
                                            const float* cache, const int* idx, float* o_lat,
                                            float* part_m, float* part_d, float* part_acc,
                                            int n_head, int kv_rank, int d_rope, int t_q,
                                            const int* n_slots_d, int chunks, float scale,
                                            void* stream_v) {
    if (kv_rank > MLA_MAX_RANK) return 40002;
    if (d_rope > MLA_MAX_ROPE) return 40003;
    if (t_q != 1) return 40030; // live twin: one query row
    if (chunks < 1 || chunks > MLA_DSA_MAX_CHUNKS) return 40022;
    if (kv_rank % 32 != 0) return 40023;
    cudaStream_t stream = (cudaStream_t)stream_v;
    long pairs = (long)t_q * n_head;
    if (pairs == 0) return 0;

    const int j = kv_rank / 32;
    const int jp = (d_rope + 31) / 32;
    // Instantiations for the shipped geometries: kv_rank 512 (glm5_next, GLM-5.2) and 1024,
    // crossed with d_rope 0 (NoPE) and 64 (GLM-5.2 rope). Anything else takes the shipped path.
#define MLA_DSA_WARP_LIVE_CASE(JJ, JPP)                                                             \
    if (j == (JJ) && jp == (JPP)) {                                                            \
        memra_dsa_warp_live_launch<JJ, JPP>(q_lat, q_pe, cache, idx, part_m, part_d, part_acc, \
                                            n_head, kv_rank, d_rope, n_slots_d, chunks, pairs,  \
                                            scale, stream);                                         \
    } else
    MLA_DSA_WARP_LIVE_CASE(16, 0)
    MLA_DSA_WARP_LIVE_CASE(16, 2)
    MLA_DSA_WARP_LIVE_CASE(32, 0)
    MLA_DSA_WARP_LIVE_CASE(32, 2) {
        return 40023;
    }
#undef MLA_DSA_WARP_LIVE_CASE
    MLA_ERR();
    memra_mla_dsa_attn_combine_kernel<<<(unsigned)pairs, MLA_THREADS, 0, stream>>>(
        part_m, part_d, part_acc, o_lat, kv_rank, chunks);
    MLA_ERR();
    return 0;
}

// ------------------------------------------------------- 3. decode pool scoring, head-blocked
//
// `memra_mla_kpool_score_dsa_kernel<H, RP, KC>` -- BIT-IDENTICAL to
// `memra_mla_kpool_score_ref_kernel` (and therefore to the shipped tiled kernel, which is
// itself gated bit-identical to the reference).
//
// WHY THE SHIPPED DECODE PATH IS 44x OFF ITS FLOOR. `n_pools = t_kv / pool`, and the score is
// `f(q_t, k_p)` with a brand-new `q_t` every token, so NO score survives a decode step: there
// is no scored-cache formulation of exact DSA top-k and the full scan is required. That makes
// the stage depth-LINEAR by construction and it is the 31.1 -> 22.7 tok/s slide from 256k to
// 1M. What is NOT required is running it at 44x the floor. Decode dispatches
// `memra_kpool_score_tiled_kernel<64, 1, 1, 1, 16>`: BT = TY*RT = 1 query, BP = TX*RP = 64
// pools, ONE accumulator per thread, so the inner step is RT + RP = 2 shared loads for
// RT * RP = 1 FFMA. The register blocking that makes the BT=128 prefill tile pay for itself is
// simply absent at BT=1, and the grid is n_pools/64 blocks of 64 threads (32768 threads at
// 128k on a die that wants ~300k).
//
// THE AXIS THE DECODE SHAPE ACTUALLY HAS IS HEADS. At t_q=1 there is one query but `heads`
// (32 on the glm5 indexer) independent dots against every pool key, and every one of them
// reuses the same pool key. This kernel blocks on (head, pool) instead of (query, pool):
//   * one thread owns RP pools and ALL H heads, holding `dot[H][RP]` in registers;
//   * pool keys stream through shared memory in KC-wide slabs, staged with coalesced global
//     reads and stored TRANSPOSED (`ksh[cc][p_local]`, row stride BP+1) so the compute loop's
//     per-thread read is conflict-free;
//   * the q slab is `qsh[cc][h]`, read as a `float4` over FOUR HEADS AT A TIME -- every thread
//     in the block reads the same address, so it is a broadcast, and the load count drops from
//     H per `c` to H/4. Shared loads per FFMA fall from 2.0 to (RP + H/4) / (H*RP) = 0.156 at
//     H=32, RP=2, i.e. from ~1/3 of FFMA peak to ~87% of it.
//   * The q staging read is strided by `d` in global (h is the fast axis in smem, not in
//     global). Deliberate: the whole q plane is H*d*4 B = 16 KB at the glm5 shape and is
//     L1/L2-resident, while the conflict-free smem layout it buys is on the hot path.
//
// BIT-IDENTITY IS A CONSTRUCTION, and it is the requirement, not a nicety: the selection
// downstream sorts these scores with a (score DESC, pool index ASC) tie-break, ReLU makes exact
// 0.0 ties ORDINARY, and a last-ulp move either side of zero moves a pool in or out of the
// budget. Both invariants the reference kernel's rounding sequence needs are preserved here:
//   * each `dot[h][r]` accumulates over `c` STRICTLY ASCENDING from +0.0f (the slab loop is
//     ascending in `c0` and the inner loop ascending in `cc`), one FFMA per term;
//   * the head mix runs `h` STRICTLY ASCENDING from +0.0f inside ONE thread -- no cross-thread
//     or cross-warp reassociation anywhere -- and spells all six rounding steps with explicit
//     `__fmaf_rn` / `__fmul_rn` / `__fadd_rn` intrinsics, so no contraction decision and no
//     compiler version can fork it. `acc += relu * w` alone would contract to one FMA and round
//     ONCE where the reference rounds twice.
// Both accumulators start at +0.0f rather than at the first term, because `(+0.0) + (-0.0)` is
// `+0.0` while `-0.0` alone is not. `-INFINITY` visibility marks and the out-of-range DROP (not
// store) rule are copied from the tiled kernel verbatim.
#define MLA_DSA_SCORE_TPB 128

// Body of the head-blocked decode scorer, shared by the scalar-n_pools entry and the live-count
// twin (lane/mla-kpool-live-20260905): the twin reads n_pools from the door's device word and
// runs on a capacity grid; every block masks `p < n_pools` and `p < vis` per element, so at
// t_q = 1 (one score row, no stride dependence) it is bit-identical to the scalar launch.
template <int H, int RP, int KC>
__device__ __forceinline__ void memra_mla_kpool_score_dsa_body(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int d, int n_pools, int pool,
    int first_pos, float qk_scale, float head_scale) {
    constexpr int TPB = MLA_DSA_SCORE_TPB;
    constexpr int BP = TPB * RP;  // pools per block
    constexpr int SBP = BP + 1;   // +1: transposed stores walk this stride, 1 mod 32 = no conflict

    const int tid = (int)threadIdx.x;
    const int p0 = (int)blockIdx.x * BP;
    const int t = (int)blockIdx.y;

    int vis = (first_pos + t + 1) / pool;
    if (vis > n_pools) vis = n_pools;
    // Block-uniform (vis depends only on blockIdx.y), decided before the first barrier.
    if (p0 >= vis) {
#pragma unroll
        for (int r = 0; r < RP; ++r) {
            int p = p0 + tid + r * TPB;
            if (p < n_pools) score[(long)t * n_pools + p] = -INFINITY;
        }
        return;
    }

    // __align__(16): `qsh` is read through `float4` in the inner loop. See the note
    // on `s_kv` above.
    extern __shared__ __align__(16) float dsa_score_sh[];
    float* qsh = dsa_score_sh;          // [KC][H], float4-read over the head axis
    float* ksh = dsa_score_sh + KC * H; // [KC][SBP], transposed

    float dot[H][RP];
#pragma unroll
    for (int h = 0; h < H; ++h)
#pragma unroll
        for (int r = 0; r < RP; ++r) dot[h][r] = 0.0f;

    for (int c0 = 0; c0 < d; c0 += KC) {
        __syncthreads(); // previous slab's readers are done with qsh/ksh
        for (int e = tid; e < KC * H; e += TPB) {
            int cc = e / H, h = e - cc * H;
            qsh[cc * H + h] = q[((long)t * H + h) * d + c0 + cc];
        }
        for (int e = tid; e < BP * KC; e += TPB) {
            int pl = e / KC, cc = e - pl * KC;
            int gp = p0 + pl;
            ksh[cc * SBP + pl] = (gp < n_pools) ? pool_keys[(long)gp * d + c0 + cc] : 0.0f;
        }
        __syncthreads();

#pragma unroll 4
        for (int cc = 0; cc < KC; ++cc) {
            float kv[RP];
#pragma unroll
            for (int r = 0; r < RP; ++r) kv[r] = ksh[cc * SBP + tid + r * TPB];
#pragma unroll
            for (int h = 0; h < H; h += 4) {
                const float4 q4 = *(const float4*)(qsh + cc * H + h);
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    dot[h + 0][r] = __fmaf_rn(q4.x, kv[r], dot[h + 0][r]);
                    dot[h + 1][r] = __fmaf_rn(q4.y, kv[r], dot[h + 1][r]);
                    dot[h + 2][r] = __fmaf_rn(q4.z, kv[r], dot[h + 2][r]);
                    dot[h + 3][r] = __fmaf_rn(q4.w, kv[r], dot[h + 3][r]);
                }
            }
        }
    }

    // Head mix, h ASCENDING inside one thread, six rounding steps spelled out.
    float acc[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
        float w = __fmul_rn(hw[(long)t * H + h], head_scale);
#pragma unroll
        for (int r = 0; r < RP; ++r) {
            float rl = fmaxf(__fmul_rn(dot[h][r], qk_scale), 0.0f);
            acc[r] = __fadd_rn(acc[r], __fmul_rn(rl, w));
        }
    }

#pragma unroll
    for (int r = 0; r < RP; ++r) {
        int p = p0 + tid + r * TPB;
        if (p >= n_pools) continue;
        score[(long)t * n_pools + p] = (p < vis) ? acc[r] : -INFINITY;
    }
}
template <int H, int RP, int KC>
__global__ __launch_bounds__(MLA_DSA_SCORE_TPB) void memra_mla_kpool_score_dsa_kernel(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int d, int n_pools, int pool,
    int first_pos, float qk_scale, float head_scale) {
    memra_mla_kpool_score_dsa_body<H, RP, KC>(q, pool_keys, hw, score, d, n_pools, pool, first_pos,
                                              qk_scale, head_scale);
}
template <int H, int RP, int KC>
__global__ __launch_bounds__(MLA_DSA_SCORE_TPB) void memra_mla_kpool_score_dsa_live_kernel(
    const float* __restrict__ q, const float* __restrict__ pool_keys,
    const float* __restrict__ hw, float* __restrict__ score, int d,
    const int* __restrict__ pos_d, int pool, float qk_scale, float head_scale) {
    // t_q = 1: the query sits at pos_d[0]; the host's `first_pos = slot` and
    // `n_pools = (slot + 1) / pool` derived on the device.
    int first_pos = pos_d[0];
    int n_pools = (first_pos + 1) / pool;
    memra_mla_kpool_score_dsa_body<H, RP, KC>(q, pool_keys, hw, score, d, n_pools, pool, first_pos,
                                              qk_scale, head_scale);
}

template <int H, int RP, int KC>
static int memra_kpool_score_dsa_launch(const float* q, const float* pool_keys, const float* hw,
                                        float* score, int t_q, int d, int n_pools, int pool,
                                        int first_pos, float qk_scale, float head_scale,
                                        cudaStream_t stream) {
    constexpr int BP = MLA_DSA_SCORE_TPB * RP;
    const size_t smem = ((size_t)KC * H + (size_t)KC * (BP + 1)) * sizeof(float);
    if (smem > 48u * 1024u) return 1;
    dim3 grid((unsigned)((n_pools + BP - 1) / BP), (unsigned)t_q);
    memra_mla_kpool_score_dsa_kernel<H, RP, KC>
        <<<grid, MLA_DSA_SCORE_TPB, smem, stream>>>(q, pool_keys, hw, score, d, n_pools, pool,
                                                    first_pos, qk_scale, head_scale);
    return 0;
}

template <int H, int RP, int KC>
static int memra_kpool_score_dsa_live_launch(const float* q, const float* pool_keys, const float* hw,
                                        float* score, int t_q, int d, const int* pos_d, int n_pools_cap, int pool,
                                        float qk_scale, float head_scale,
                                        cudaStream_t stream) {
    constexpr int BP = MLA_DSA_SCORE_TPB * RP;
    const size_t smem = ((size_t)KC * H + (size_t)KC * (BP + 1)) * sizeof(float);
    if (smem > 48u * 1024u) return 1;
    dim3 grid((unsigned)((n_pools_cap + BP - 1) / BP), (unsigned)t_q);
    memra_mla_kpool_score_dsa_live_kernel<H, RP, KC>
        <<<grid, MLA_DSA_SCORE_TPB, smem, stream>>>(q, pool_keys, hw, score, d, pos_d, pool,
                                                    qk_scale, head_scale);
    return 0;
}

/// Decode-shaped scorer. Returns 40023 when this geometry has no instantiation (the host door
/// then falls through to `memra_mla_kpool_score_f32`, the shipped dispatch, unchanged). The
/// head count is a TEMPLATE parameter because `dot[H][RP]` must live in registers: a runtime
/// bound would put it in local memory and lose the whole point.
extern "C" int memra_mla_kpool_score_dsa_f32(const float* q, const float* pool_keys,
                                             const float* hw, float* score, int t_q, int heads,
                                             int d, int n_pools, int pool, int first_pos,
                                             float qk_scale, float head_scale, void* stream_v) {
    if (pool <= 0) return 40010;
    if (d <= 0) return 40017;
    if (t_q <= 0 || n_pools <= 0) return 0;
    constexpr int KC = 32;
    if (d % KC != 0) return 40023; // the slab loop is exact, never zero-padded (see edge rules)
    cudaStream_t stream = (cudaStream_t)stream_v;
    int rc;
    switch (heads) {
        case 16:
            rc = memra_kpool_score_dsa_launch<16, 2, KC>(q, pool_keys, hw, score, t_q, d, n_pools,
                                                         pool, first_pos, qk_scale, head_scale,
                                                         stream);
            break;
        case 32:
            rc = memra_kpool_score_dsa_launch<32, 2, KC>(q, pool_keys, hw, score, t_q, d, n_pools,
                                                         pool, first_pos, qk_scale, head_scale,
                                                         stream);
            break;
        case 64:
            rc = memra_kpool_score_dsa_launch<64, 2, KC>(q, pool_keys, hw, score, t_q, d, n_pools,
                                                         pool, first_pos, qk_scale, head_scale,
                                                         stream);
            break;
        default:
            return 40023;
    }
    if (rc != 0) return 40023;
    MLA_ERR();
    return 0;
}

extern "C" int memra_mla_kpool_score_dsa_live_f32(const float* q, const float* pool_keys,
                                             const float* hw, float* score, int t_q, int heads,
                                             int d, const int* pos_d, int n_pools_cap, int pool,
                                             float qk_scale, float head_scale, void* stream_v) {
    if (pool <= 0) return 40010;
    if (d <= 0) return 40017;
    if (t_q != 1) return 40030; // live twin: one score row (no stride dependence)
    if (n_pools_cap <= 0) return 0;
    constexpr int KC = 32;
    if (d % KC != 0) return 40023; // the slab loop is exact, never zero-padded (see edge rules)
    cudaStream_t stream = (cudaStream_t)stream_v;
    int rc;
    switch (heads) {
        case 16:
            rc = memra_kpool_score_dsa_live_launch<16, 2, KC>(q, pool_keys, hw, score, t_q, d, pos_d, n_pools_cap,
                                                         pool, qk_scale, head_scale,
                                                         stream);
            break;
        case 32:
            rc = memra_kpool_score_dsa_live_launch<32, 2, KC>(q, pool_keys, hw, score, t_q, d, pos_d, n_pools_cap,
                                                         pool, qk_scale, head_scale,
                                                         stream);
            break;
        case 64:
            rc = memra_kpool_score_dsa_live_launch<64, 2, KC>(q, pool_keys, hw, score, t_q, d, pos_d, n_pools_cap,
                                                         pool, qk_scale, head_scale,
                                                         stream);
            break;
        default:
            return 40023;
    }
    if (rc != 0) return 40023;
    MLA_ERR();
    return 0;
}

// ================================================== B200 DSA k-pool SELECT door (sm_100a)
//
// MEMRA_B200_DSA_SELECT (host seam mla_ffi.rs, default OFF; docs/FLAGS.md row). The lane that
// shipped MEMRA_B200_DSA_DECODE took `attn_gathered` and `kpool_score` down ~10x and 4-7.6x on
// the pair and named THIS kernel as what the scorer had been hiding:
// `memra_mla_kpool_select_kernel` grids `t_q` blocks, so at plain decode it is ONE CTA -- 0.68%
// of a 148-SM die -- walking `n_pools` up to ~10 times (8 MSB-first radix passes, an optional
// unique-resolution scan, then the membership count and the emit). It is depth-LINEAR in
// `n_pools = t_kv / pool`, and its byte floor is `n_pools * 4 B` per sweep, which at 1M is
// 1 MB -> ~0.13 us at 8 TB/s against a measured ~170 us. The gap is parallelism, nothing else.
//
// EXACT, NOT BANDED. The output is a pure function of ONE 64-bit number: the shipped kernel
// picks `thr` = the `select_k`-th smallest order key, then emits every pool with
// `key(p) <= thr`. `memra_kpool_key` is a strictly decreasing injection composed with the pool
// index, so keys are DISTINCT and "the select_k-th smallest" is unambiguous. Reproducing `thr`
// bit-for-bit therefore reproduces the selection bit-for-bit -- the parallel kernels below
// compute the SAME `thr` and run the SAME membership test, so this is a launch-geometry change
// with an exact answer, not a tolerance. `dsa-select-gate` asserts the emitted `idx` plane
// byte-identical to the shipped kernel's, and carries a RED ARM (a deliberately wrong
// threshold) that must fail before the real kernel is allowed to pass.
//
// WHY THE HIGH WORD, THEN THE INDEX -- and why that is 2 radix passes, not 8. The key is
// `(desc32(score) << 32) | pool_index`. The LOW word is the pool index, which is unique per
// row, so it never needs a radix descent: once the high word (the score) is resolved, the
// threshold's index is simply the `r`-th smallest index among the pools that TIE at that score,
// and a rank-`r` selection over an ascending index range is a count plus a scan. So the descent
// only has to resolve 32 bits, and two 16-bit passes do it. Ties are the common case here, not
// the corner: ReLU zeroes every head whose query-pool dot is non-positive, so exact 0.0 ties
// across pools are ORDINARY (the same fact that makes the scorer's bit-identity load-bearing).
//
// SIX LAUNCHES, NO GRID BARRIER, NO SPIN. Each multi-CTA pass ends with the LAST CTA to arrive
// running the epilogue (`__syncthreads()`, then `__threadfence()`, then `atomicAdd` on a done
// counter -- the full two-phase-reduction idiom; the leading barrier is load-bearing and its
// absence was a real race, see `memra_sel_last_arrival`), so there is no cooperative launch, no
// persistent-grid residency assumption and no spin-wait that could hang a serving box if
// occupancy ever changed. The launches are a one-off `clear` (the workspace arrives
// uninitialised) followed by five passes: (1) histogram of the high word's top 16 bits + finite
// count + descent;
// (2) histogram of its low 16 bits within the chosen prefix + descent, which fixes `thr_s`;
// (3) per-CTA count of the tie group + locate the rank-`r` index, which fixes `thr_p`;
// (4) per-CTA count of members; (5) exclusive-scan the CTA counts and emit. Passes 3 and 4 both
// end in a last-CTA epilogue; pass 5 reads the scan the pass-4 epilogue wrote.
//
// DEGENERATE ANSWERS ARE COPIED, NOT REDERIVED: `n_fin == 0` selects nothing and leaves only the
// tail, and `n_fin < select_k` clamps the rank so every visible pool is selected. Those are the
// two answers the shipped kernel's round exhaustion gives, and the gate covers both.

#define MLA_SEL_BINS 65536          // 16-bit radix; the bins live in the global workspace
#define MLA_SEL_THREADS 256
#define MLA_SEL_MAX_CTAS 1024
// Workspace control words, per query. Kept in one cache line's worth of slots for clarity.
#define MLA_SEL_CTRL_LIVE 0
#define MLA_SEL_CTRL_K 1
#define MLA_SEL_CTRL_HI 2      // resolved high word of the threshold key (desc32(score))
#define MLA_SEL_CTRL_TP 3      // threshold pool index
#define MLA_SEL_CTRL_DONE 4    // last-CTA arrival counter
#define MLA_SEL_CTRL_TOTAL 5   // total selected pools
#define MLA_SEL_CTRL_RANK 6    // rank inside the tie group
#define MLA_SEL_CTRL_WORDS 8

/// Workspace ints per query: the 16-bit histogram, the control words, and one count per CTA
/// (used twice: the tie-group counts and the membership counts).
extern "C" long memra_mla_kpool_select_ws_ints(int n_ctas) {
    if (n_ctas < 1) n_ctas = 1;
    return (long)MLA_SEL_BINS + MLA_SEL_CTRL_WORDS + (long)n_ctas;
}

/// Grid width the launcher uses. Exposed so the host sizes the workspace from the SAME number
/// the kernels index with; a mismatch here would be an out-of-bounds write, not a slow path.
extern "C" int memra_mla_kpool_select_ctas(int n_pools) {
    long want = ((long)n_pools + MLA_SEL_THREADS - 1) / MLA_SEL_THREADS;
    if (want < 1) want = 1;
    if (want > MLA_SEL_MAX_CTAS) want = MLA_SEL_MAX_CTAS;
    return (int)want;
}

__device__ __forceinline__ unsigned memra_sel_hi(float s, int p) {
    return (unsigned)(memra_kpool_key(s, p) >> 32);
}

/// True when this CTA is the last of `n_ctas` to arrive, and the point at which THIS CTA's
/// global writes become visible to whichever CTA arrives last. The CUDA Programming Guide's
/// `threadFenceReduction` idiom, spelled with BOTH barriers it needs:
///
///   `__syncthreads()` FIRST, then `__threadfence()`, then the counter increment by thread 0.
///
/// THE LEADING BARRIER IS LOAD-BEARING and its absence was a real, silent race (caught in review
/// on PR #115, never by a gate -- see below). `__threadfence()` orders only the CALLING THREAD's
/// prior writes. The histogram pass's data is written by all 256 threads (`atomicAdd(&hist[..])`)
/// and by eight warp leaders (the finite count), with only warp-scoped `__shfl_down_sync` between
/// those writes and this call. Without a block barrier, thread 0 could publish this CTA's arrival
/// while warp 7 had not yet issued its histogram atomics; the CTA that then observes the full
/// count -- on another SM, having synchronised only through the counter -- would run the descent
/// on a histogram missing those bins and an undercounted `n_fin`, producing a wrong `ctrl[HI]`
/// and a wrong rank clamp. That is a non-deterministic break of the exact byte-identity this
/// whole door exists to provide. The barrier makes thread 0's device-scope fence publish the
/// WHOLE CTA's writes, which is what the idiom requires and what the sibling passes were already
/// getting for free by reducing through shared memory and letting thread 0 be the sole writer.
///
/// NOTHING IN THE TREE CAN CATCH THIS, which is why it is written down rather than left to a
/// gate: `compute-sanitizer racecheck` is shared-memory only, `synccheck` looks for barrier
/// divergence, and `dsa-select-gate` runs each exactness cell on an idle device where the window
/// is vanishingly small. The 40/40 EXACT receipt did not speak to it and a re-run does not
/// either -- the argument above is the evidence, and the gate only shows the fix costs nothing.
///
/// Every call site invokes this from ALL threads of the block unconditionally, which the leading
/// `__syncthreads()` requires; keep it that way.
__device__ __forceinline__ bool memra_sel_last_arrival(int* ctrl, int n_ctas) {
    __shared__ int s_last;
    __syncthreads(); // publish every thread's writes to the block before thread 0 fences them
    __threadfence();
    if (threadIdx.x == 0) {
        int prev = atomicAdd(&ctrl[MLA_SEL_CTRL_DONE], 1);
        s_last = (prev == n_ctas - 1) ? 1 : 0;
    }
    __syncthreads();
    return s_last != 0;
}

// ---- pass 1 and 2: 16-bit histogram of the high word + descent by the last CTA ------------
//
// `pass` 0 histograms bits [31:16] of the high word with no prefix; `pass` 1 histograms bits
// [15:0] restricted to the resolved top half. Pass 0 also totals the finite pools, which is
// where the two degenerate answers are decided.
extern "C" __global__ void memra_mla_kpool_select_hist_kernel(const float* __restrict__ score,
                                                              int* __restrict__ ws, int n_pools,
                                                              int select_k, int ws_stride,
                                                              int n_ctas, int pass) {
    int t = blockIdx.y;
    const float* row = score + (long)t * n_pools;
    int* base = ws + (long)t * ws_stride;
    unsigned* hist = (unsigned*)base;
    int* ctrl = base + MLA_SEL_BINS;

    // Zero the histogram cooperatively across the whole grid before anyone accumulates: the
    // launcher issues a separate clear kernel, so nothing here races the zeroing.
    int shift = pass == 0 ? 16 : 0;
    unsigned pre = pass == 0 ? 0u : (unsigned)ctrl[MLA_SEL_CTRL_HI];
    unsigned mask = pass == 0 ? 0u : 0xffff0000u;
    int live = pass == 0 ? 1 : ctrl[MLA_SEL_CTRL_LIVE];

    if (live) {
        unsigned local_fin = 0;
        for (long p = (long)blockIdx.x * MLA_SEL_THREADS + threadIdx.x; p < n_pools;
             p += (long)n_ctas * MLA_SEL_THREADS) {
            float s = row[p];
            if (!isfinite(s)) continue;
            unsigned hi = memra_sel_hi(s, (int)p);
            if ((hi & mask) != pre) continue;
            ++local_fin;
            atomicAdd(&hist[(hi >> shift) & 0xffffu], 1u);
        }
        if (pass == 0) {
            // Warp-then-block reduce the finite count so pass 0 needs no second sweep.
            for (int off = 16; off > 0; off >>= 1)
                local_fin += __shfl_down_sync(0xffffffffu, local_fin, off);
            if ((threadIdx.x & 31) == 0) atomicAdd((unsigned*)&ctrl[MLA_SEL_CTRL_RANK], local_fin);
        }
    }

    if (!memra_sel_last_arrival(ctrl, n_ctas)) return;

    // DESCENT, run by the last-arriving CTA with ALL 256 threads. A single-threaded walk of
    // 65536 bins would be ~131 us on its own -- more than the whole kernel this door replaces --
    // so the bin search is two-level: every thread sums its own contiguous slice of
    // MLA_SEL_BINS/MLA_SEL_THREADS bins, thread 0 exclusive-scans the 256 slice sums and names
    // the slice holding rank k, and that one thread re-walks its 256 bins to name the bin and
    // the running count before it. Same answer as the ascending single-threaded walk, because
    // both levels are contiguous and ascending.
    __shared__ unsigned s_slice[MLA_SEL_THREADS];
    __shared__ int s_owner;
    __shared__ unsigned s_before;
    __shared__ int s_go;

    if (threadIdx.x == 0) {
        ctrl[MLA_SEL_CTRL_DONE] = 0; // rearm for the next pass
        if (pass == 0) {
            unsigned n_fin = (unsigned)ctrl[MLA_SEL_CTRL_RANK];
            if (n_fin == 0u || select_k <= 0 || n_pools <= 0) {
                ctrl[MLA_SEL_CTRL_LIVE] = 0;
            } else {
                ctrl[MLA_SEL_CTRL_LIVE] = 1;
                ctrl[MLA_SEL_CTRL_K] = (unsigned)select_k > n_fin ? (int)n_fin : select_k;
            }
        }
        s_go = ctrl[MLA_SEL_CTRL_LIVE];
    }
    __syncthreads();
    if (!s_go) return;

    constexpr int SLICE = MLA_SEL_BINS / MLA_SEL_THREADS;
    unsigned k = (unsigned)ctrl[MLA_SEL_CTRL_K];
    unsigned mysum = 0;
    for (int j = threadIdx.x * SLICE; j < (int)threadIdx.x * SLICE + SLICE; ++j) mysum += hist[j];
    s_slice[threadIdx.x] = mysum;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned run = 0;
        int owner = MLA_SEL_THREADS - 1;
        for (int j = 0; j < MLA_SEL_THREADS; ++j) {
            unsigned c = s_slice[j];
            if (run + c >= k) {
                owner = j;
                break;
            }
            run += c;
        }
        s_owner = owner;
        s_before = run;
    }
    __syncthreads();
    if ((int)threadIdx.x == s_owner) {
        unsigned run = s_before;
        int bin = s_owner * SLICE;
        for (int j = s_owner * SLICE; j < s_owner * SLICE + SLICE; ++j) {
            unsigned c = hist[j];
            if (c == 0u) continue;
            if (run + c >= k) {
                bin = j;
                break;
            }
            run += c;
        }
        ctrl[MLA_SEL_CTRL_K] = (int)(k - run);
        ctrl[MLA_SEL_CTRL_HI] = (int)(pre | ((unsigned)bin << shift));
    }
    // Pass 0 leaves the bins ZEROED for pass 1, which is what lets the launcher issue ONE clear
    // kernel instead of one per pass. Safe here and only here: this is the last-arriving CTA, so
    // every histogram write has landed, and the descent above has already read what it needs
    // (the barrier orders the owner thread's reads before these writes).
    __syncthreads();
    if (pass == 0)
        for (int j = threadIdx.x; j < MLA_SEL_BINS; j += MLA_SEL_THREADS) hist[j] = 0u;
}

/// Zero the histogram and the control words once, before pass 0. The workspace comes from an
/// UNINITIALISED device allocation, so this is what makes the first pass well-defined; pass 0's
/// own epilogue re-zeroes the bins for pass 1, so this runs once per call, not once per pass.
extern "C" __global__ void memra_mla_kpool_select_clear_kernel(int* __restrict__ ws,
                                                               int ws_stride, int n_ctas) {
    int t = blockIdx.y;
    int* base = ws + (long)t * ws_stride;
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < MLA_SEL_BINS;
         j += (long)gridDim.x * blockDim.x)
        base[j] = 0;
    int* ctrl = base + MLA_SEL_BINS;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctrl[MLA_SEL_CTRL_DONE] = 0;
        ctrl[MLA_SEL_CTRL_RANK] = 0;
        ctrl[MLA_SEL_CTRL_HI] = 0;
        ctrl[MLA_SEL_CTRL_TP] = -1;
        ctrl[MLA_SEL_CTRL_TOTAL] = 0;
        ctrl[MLA_SEL_CTRL_LIVE] = 1;
        ctrl[MLA_SEL_CTRL_K] = 0;
    }
    // The per-CTA count slots are written before they are read in every pass, but zero them
    // anyway so a short grid can never leave a stale word inside the scanned range.
    for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < n_ctas;
         j += (long)gridDim.x * blockDim.x)
        ctrl[MLA_SEL_CTRL_WORDS + j] = 0;
}

// ---- pass 3: fix `thr_p`, the rank-r smallest INDEX among pools tying at the threshold score
//
// After the two histogram passes `ctrl[HI]` is the threshold's high word exactly and `ctrl[K]`
// is the 1-based rank WITHIN that tie group. Each CTA owns a contiguous pool range and counts
// its tie members; the last CTA exclusive-scans those counts, finds the CTA holding rank r, and
// -- because a contiguous range walked ascending yields ascending indices -- re-walks that one
// range to name the index. The re-walk is one CTA over `n_pools / n_ctas` pools, not over
// `n_pools`.
extern "C" __global__ void memra_mla_kpool_select_tie_kernel(const float* __restrict__ score,
                                                             int* __restrict__ ws, int n_pools,
                                                             int ws_stride, int n_ctas) {
    __shared__ int s_cnt;
    int t = blockIdx.y;
    const float* row = score + (long)t * n_pools;
    int* base = ws + (long)t * ws_stride;
    int* ctrl = base + MLA_SEL_BINS;
    int* cta = base + MLA_SEL_BINS + MLA_SEL_CTRL_WORDS;

    long chunk = ((long)n_pools + n_ctas - 1) / n_ctas;
    long lo = (long)blockIdx.x * chunk;
    long hi_end = lo + chunk;
    if (lo > n_pools) lo = n_pools;
    if (hi_end > n_pools) hi_end = n_pools;
    unsigned target = (unsigned)ctrl[MLA_SEL_CTRL_HI];
    int live = ctrl[MLA_SEL_CTRL_LIVE];

    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    if (live) {
        int mine = 0;
        for (long p = lo + threadIdx.x; p < hi_end; p += MLA_SEL_THREADS) {
            float s = row[p];
            if (!isfinite(s)) continue;
            if (memra_sel_hi(s, (int)p) == target) ++mine;
        }
        atomicAdd(&s_cnt, mine);
    }
    __syncthreads();
    if (threadIdx.x == 0) cta[blockIdx.x] = s_cnt;

    if (!memra_sel_last_arrival(ctrl, n_ctas)) return;
    if (threadIdx.x != 0) return;
    ctrl[MLA_SEL_CTRL_DONE] = 0;
    if (!live) return;
    int r = ctrl[MLA_SEL_CTRL_K];
    int run = 0, owner = 0;
    for (int j = 0; j < n_ctas; ++j) {
        int c = cta[j];
        if (run + c >= r) {
            owner = j;
            break;
        }
        run += c;
    }
    int need = r - run; // 1-based rank inside the owning CTA's range
    long olo = (long)owner * chunk;
    long ohi = olo + chunk;
    if (olo > n_pools) olo = n_pools;
    if (ohi > n_pools) ohi = n_pools;
    int seen = 0;
    for (long p = olo; p < ohi; ++p) {
        float s = row[p];
        if (!isfinite(s)) continue;
        if (memra_sel_hi(s, (int)p) != target) continue;
        if (++seen == need) {
            ctrl[MLA_SEL_CTRL_TP] = (int)p;
            return;
        }
    }
}

// ---- pass 4: per-CTA membership counts, then the last CTA exclusive-scans them in place ----
extern "C" __global__ void memra_mla_kpool_select_count_kernel(const float* __restrict__ score,
                                                               int* __restrict__ ws, int n_pools,
                                                               int ws_stride, int n_ctas) {
    __shared__ int s_cnt;
    int t = blockIdx.y;
    const float* row = score + (long)t * n_pools;
    int* base = ws + (long)t * ws_stride;
    int* ctrl = base + MLA_SEL_BINS;
    int* cta = base + MLA_SEL_BINS + MLA_SEL_CTRL_WORDS;

    long chunk = ((long)n_pools + n_ctas - 1) / n_ctas;
    long lo = (long)blockIdx.x * chunk;
    long hi_end = lo + chunk;
    if (lo > n_pools) lo = n_pools;
    if (hi_end > n_pools) hi_end = n_pools;
    int live = ctrl[MLA_SEL_CTRL_LIVE];
    unsigned long long thr =
        ((unsigned long long)(unsigned)ctrl[MLA_SEL_CTRL_HI] << 32) |
        (unsigned long long)(unsigned)ctrl[MLA_SEL_CTRL_TP];

    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    if (live) {
        int mine = 0;
        for (long p = lo + threadIdx.x; p < hi_end; p += MLA_SEL_THREADS) {
            float s = row[p];
            if (!isfinite(s)) continue;
            if (memra_kpool_key(s, (int)p) <= thr) ++mine;
        }
        atomicAdd(&s_cnt, mine);
    }
    __syncthreads();
    if (threadIdx.x == 0) cta[blockIdx.x] = s_cnt;

    if (!memra_sel_last_arrival(ctrl, n_ctas)) return;
    if (threadIdx.x != 0) return;
    ctrl[MLA_SEL_CTRL_DONE] = 0;
    int run = 0;
    for (int j = 0; j < n_ctas; ++j) {
        int c = cta[j];
        cta[j] = run;
        run += c;
    }
    ctrl[MLA_SEL_CTRL_TOTAL] = run;
}

// ---- pass 5: emit, ascending, then the tail and the -1 pad -------------------------------
//
// Same emit contract as the shipped kernel: CONTIGUOUS ranges walked ascending, each selected
// pool expanded to its `pool` raw cache rows, then `always_tail`'s incomplete tail, then -1 to
// `width`. Two levels of contiguous range (CTA, then thread inside it) instead of one, which
// preserves ascending order because both levels are contiguous and both are walked ascending.
extern "C" __global__ void memra_mla_kpool_select_emit_kernel(const float* __restrict__ score,
                                                              const int* __restrict__ ws,
                                                              int* __restrict__ idx, int n_pools,
                                                              int pool, int width, int first_pos,
                                                              int always_tail, int ws_stride,
                                                              int n_ctas) {
    __shared__ int sh_n[MLA_SEL_THREADS];
    int t = blockIdx.y;
    const float* row = score + (long)t * n_pools;
    int* out = idx + (long)t * width;
    const int* base = ws + (long)t * ws_stride;
    const int* ctrl = base + MLA_SEL_BINS;
    const int* cta = base + MLA_SEL_BINS + MLA_SEL_CTRL_WORDS;
    int tid = threadIdx.x;

    int live = ctrl[MLA_SEL_CTRL_LIVE];
    int total = ctrl[MLA_SEL_CTRL_TOTAL];
    unsigned long long thr =
        ((unsigned long long)(unsigned)ctrl[MLA_SEL_CTRL_HI] << 32) |
        (unsigned long long)(unsigned)ctrl[MLA_SEL_CTRL_TP];

    long chunk = ((long)n_pools + n_ctas - 1) / n_ctas;
    long clo = (long)blockIdx.x * chunk;
    long chi = clo + chunk;
    if (clo > n_pools) clo = n_pools;
    if (chi > n_pools) chi = n_pools;

    // Per-thread contiguous sub-range of this CTA's range.
    long span = chi - clo;
    long tchunk = (span + MLA_SEL_THREADS - 1) / MLA_SEL_THREADS;
    long lo = clo + (long)tid * tchunk;
    long hi_end = lo + tchunk;
    if (lo > chi) lo = chi;
    if (hi_end > chi) hi_end = chi;

    int mine = 0;
    if (live) {
        for (long p = lo; p < hi_end; ++p) {
            float s = row[p];
            if (!isfinite(s)) continue;
            if (memra_kpool_key(s, (int)p) <= thr) ++mine;
        }
    }
    sh_n[tid] = mine;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int j = 0; j < MLA_SEL_THREADS; ++j) {
            int c = sh_n[j];
            sh_n[j] = run;
            run += c;
        }
    }
    __syncthreads();
    int slot = cta[blockIdx.x] + sh_n[tid];
    if (live) {
        for (long p = lo; p < hi_end; ++p) {
            float s = row[p];
            if (!isfinite(s)) continue;
            if (memra_kpool_key(s, (int)p) > thr) continue;
            for (int j = 0; j < pool; ++j) out[(long)slot * pool + j] = (int)p * pool + j;
            ++slot;
        }
    }

    // The tail and the pad are written once, by CTA 0, exactly as the shipped kernel does.
    if (blockIdx.x != 0) return;
    int filled = total * pool;
    if (always_tail) {
        int visible = first_pos + t + 1;
        int tail = visible % pool;
        for (int j = tid; j < tail; j += MLA_SEL_THREADS) out[filled + j] = visible - tail + j;
        filled += tail;
    }
    for (int j = filled + tid; j < width; j += MLA_SEL_THREADS) out[j] = -1;
}

/// Exact multi-CTA k-pool selection (`MEMRA_B200_DSA_SELECT`). Byte-identical output to
/// `memra_mla_kpool_select_f32`: it computes the SAME `select_k`-th smallest order key and runs
/// the SAME `key(p) <= thr` membership test, only with the work spread over `n_ctas` CTAs per
/// query instead of one. Six launches per call (clear, two histogram descents, tie locate,
/// membership count, emit), each ending in a last-CTA epilogue rather than a grid barrier.
///
/// `ws` must hold `t_q * memra_mla_kpool_select_ws_ints(memra_mla_kpool_select_ctas(n_pools))`
/// ints; the host reads both numbers from these entry points so the two cannot disagree about
/// the layout the kernels index.
extern "C" int memra_mla_kpool_select_dsa_f32(const float* score, int* idx, int* ws, int t_q,
                                              int n_pools, int pool, int select_k, int width,
                                              int first_pos, int always_tail, void* stream_v) {
    int bad = memra_mla_kpool_select_check(pool, select_k, width, always_tail);
    if (bad) return bad;
    if (t_q == 0) return 0;
    if (n_pools < 0) return 40012;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int n_ctas = memra_mla_kpool_select_ctas(n_pools);
    int ws_stride = (int)memra_mla_kpool_select_ws_ints(n_ctas);
    dim3 grid((unsigned)n_ctas, (unsigned)t_q);

    dim3 cgrid((unsigned)((MLA_SEL_BINS + MLA_SEL_THREADS - 1) / MLA_SEL_THREADS), (unsigned)t_q);
    memra_mla_kpool_select_clear_kernel<<<cgrid, MLA_SEL_THREADS, 0, stream>>>(ws, ws_stride,
                                                                              n_ctas);
    MLA_ERR();
    memra_mla_kpool_select_hist_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, n_pools, select_k, ws_stride, n_ctas, 0);
    MLA_ERR();
    memra_mla_kpool_select_hist_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, n_pools, select_k, ws_stride, n_ctas, 1);
    MLA_ERR();
    memra_mla_kpool_select_tie_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(score, ws, n_pools,
                                                                           ws_stride, n_ctas);
    MLA_ERR();
    memra_mla_kpool_select_count_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(score, ws, n_pools,
                                                                             ws_stride, n_ctas);
    MLA_ERR();
    memra_mla_kpool_select_emit_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, idx, n_pools, pool, width, first_pos, always_tail, ws_stride, n_ctas);
    MLA_ERR();
    return 0;
}

/// RED-ARM hook for `dsa-select-gate`, never on a serving path. Runs the exact pipeline and then
/// perturbs the resolved threshold by `bump` before the membership count. The gate calls it with
/// `bump = -1` and REQUIRES a mismatch: a gate that cannot fail is not a gate, and this is what
/// proves the byte comparison it runs on the real kernel is actually looking at the selection.
///
/// WHY -1 AND NOT +1, learned the hard way on the first run: membership is `key(p) <= thr`, so
/// RAISING the threshold by one admits pool `thr_p + 1` only if that pool TIES at the threshold
/// score -- otherwise its key differs in the high word and the set is unchanged, and the red arm
/// silently matches. LOWERING it by one always drops exactly the threshold pool itself, because
/// `key(thr_p) == thr` by construction, so the perturbation is guaranteed to move the plane by
/// one pool (`pool` slots) at every shape. A red arm that can no-op is not a red arm.
extern "C" __global__ void memra_mla_kpool_select_perturb_kernel(int* __restrict__ ws,
                                                                 int ws_stride, int bump) {
    int t = blockIdx.y;
    int* ctrl = ws + (long)t * ws_stride + MLA_SEL_BINS;
    if (threadIdx.x == 0) ctrl[MLA_SEL_CTRL_TP] = ctrl[MLA_SEL_CTRL_TP] + bump;
}

extern "C" int memra_mla_kpool_select_dsa_redarm_f32(const float* score, int* idx, int* ws,
                                                     int t_q, int n_pools, int pool,
                                                     int select_k, int width, int first_pos,
                                                     int always_tail, int bump, void* stream_v) {
    int bad = memra_mla_kpool_select_check(pool, select_k, width, always_tail);
    if (bad) return bad;
    if (t_q == 0) return 0;
    cudaStream_t stream = (cudaStream_t)stream_v;
    int n_ctas = memra_mla_kpool_select_ctas(n_pools);
    int ws_stride = (int)memra_mla_kpool_select_ws_ints(n_ctas);
    dim3 grid((unsigned)n_ctas, (unsigned)t_q);
    dim3 cgrid((unsigned)((MLA_SEL_BINS + MLA_SEL_THREADS - 1) / MLA_SEL_THREADS), (unsigned)t_q);

    memra_mla_kpool_select_clear_kernel<<<cgrid, MLA_SEL_THREADS, 0, stream>>>(ws, ws_stride,
                                                                              n_ctas);
    memra_mla_kpool_select_hist_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, n_pools, select_k, ws_stride, n_ctas, 0);
    memra_mla_kpool_select_hist_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, n_pools, select_k, ws_stride, n_ctas, 1);
    memra_mla_kpool_select_tie_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(score, ws, n_pools,
                                                                           ws_stride, n_ctas);
    dim3 one(1u, (unsigned)t_q);
    memra_mla_kpool_select_perturb_kernel<<<one, 32, 0, stream>>>(ws, ws_stride, bump);
    memra_mla_kpool_select_count_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(score, ws, n_pools,
                                                                             ws_stride, n_ctas);
    MLA_ERR();
    memra_mla_kpool_select_emit_kernel<<<grid, MLA_SEL_THREADS, 0, stream>>>(
        score, ws, idx, n_pools, pool, width, first_pos, always_tail, ws_stride, n_ctas);
    MLA_ERR();
    return 0;
}
