// Kimi Delta Attention (KDA) kernels — the glm5_next (GLM-5.3-Flash) linear-attention mixer.
// All f32, no tensor cores -> sm_120-native, same class as cu/hybrid.cu's GDN kernels.
//
// KDA vs the Gated DeltaNet kernels next door (cu/hybrid.cu gdn_*): same delta-rule family,
// four semantic differences that make these SEPARATE kernels rather than GDN parameters —
//   1. decay is PER CHANNEL (width heads*head_dim), not per head. The state decays along its
//      K dimension with a different factor per row, so the scalar `g_val * s` fold GDN uses
//      (and every scalar-decay identity the chunked WY transform rests on) does not hold.
//   2. beta is a per-head sigmoid of its own projection, not a shared alpha/dt program.
//   3. q and k are L2-normalized (eps 1e-6 INSIDE the sqrt, independent of the layer eps).
//   4. the output norm is SIGMOID-gated, where GDN's gated_rmsnorm_f32 gates with SiLU.
//
// LAYOUTS. KDA is symmetric (q, k and v are all heads*head_dim wide) and has no GQA head
// repeat, so channel c == h*head_dim + i IS the (head, dim) pair and every per-token tensor
// stays token-major with no repack:
//   q,k,v,g,gate,core: [T, qkv]   element (t,h,i) at (t*H + h)*D + i
//   beta:              [T, H]     element (t,h)   at  t*H + h
//   conv ring:         [3*qkv, K-1] channel-major, RAW pre-conv values, plane p at p*qkv rows
//   conv weight:       [3*qkv, K]   channel-major, plane p at p*qkv rows (fused at load)
//   state:             [H, D, D]  TRANSPOSED M[col][i] = S[i][col], at (h*D + col)*D + i
// The state transpose is GDN's (cu/hybrid.cu gdn_scan_kernel) and is what lets one warp own
// one output column with its 32 lanes sharding the K dimension.

#include <cuda_runtime.h>

__device__ __forceinline__ float memra_kda_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float memra_kda_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// All-lane sum via XOR butterfly (cu/hybrid.cu warp_reduce_sum, same form).
template <int WARP>
__device__ __forceinline__ float memra_kda_warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
// Lane-0-valid sum: saves the broadcast shuffle where the consumer is lane-0-gated.
template <int WARP>
__device__ __forceinline__ float memra_kda_warp_sum_down(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// ---- Per-plane causal short conv + SiLU, token-major in AND out. ----
// The checkpoint ships three per-plane conv weights (q/k/v); they are concatenated at load into
// one [3*qkv, K] buffer, so `plane` selects the row block — the same arithmetic that selects the
// plane's rows in the fused conv ring. Applying each plane's taps to its own plane IS the fused
// grouped conv over 3*qkv channels (memra-reference kimi_delta_net says so in-line).
// `ring` is always a real buffer — a fresh prefill passes a ZEROED one, which IS the reference's
// zero left pad. Rows before the window start read it; slot j holds the raw value for position
// -(K-1)+j, the same slot order memra-reference writes its conv_state in.
// Launch: grid=(ceil(qkv/256), T), block=256.
extern "C" __global__ void memra_kda_conv_silu_f32(
        const float* __restrict__ x_tm,     // [T, qkv] token-major projection output
        const float* __restrict__ w,        // [3*qkv, K] fused, channel-major
        const float* __restrict__ ring,     // [3*qkv, K-1] fused (zeroed = fresh prefill)
        float* __restrict__ y_tm,           // [T, qkv] token-major, SiLU applied
        int qkv, int T, int K, int plane) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= qkv || t >= T) return;
    const int pad = K - 1;
    const int fused = plane * qkv + c;
    const float* wc = w + (size_t)fused * K;
    const float* rc = ring + (size_t)fused * pad;
    float acc = 0.0f;
    // Ascending tap order — the reference accumulates taps ascending too.
    for (int j = 0; j < K; j++) {
        int tt = t - pad + j;
        float xv = (tt >= 0) ? x_tm[(size_t)tt * qkv + c] : rc[pad + tt];
        acc += xv * wc[j];
    }
    y_tm[(size_t)t * qkv + c] = memra_kda_silu(acc);
}

// ---- Roll the carried conv ring forward over a T-token prefill. ----
// Slot idx must end holding the RAW pre-conv value of absolute position (end - pad + idx). Values
// that predate this chunk come from the ring's own older slots, so every slot is read into a
// register BEFORE any store — one thread per channel makes that in-place update safe.
// Launch: grid=ceil(qkv/256), block=256.
extern "C" __global__ void memra_kda_conv_ring_roll_f32(
        const float* __restrict__ x_tm,     // [T, qkv] token-major, RAW (pre-conv) projection
        float* __restrict__ ring,           // [3*qkv, K-1] fused, updated in place
        int qkv, int T, int K, int plane) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv) return;
    const int pad = K - 1;
    float* rc = ring + (size_t)(plane * qkv + c) * pad;
    float old[8];
    for (int j = 0; j < pad; j++) old[j] = rc[j];
    for (int idx = 0; idx < pad; idx++) {
        int src = T - pad + idx;
        rc[idx] = (src >= 0) ? x_tm[(size_t)src * qkv + c] : old[pad + src];
    }
}

// ---- T=1 decode: assemble [ring | new], conv + SiLU, and roll the ring — one launch. ----
// The ssm_conv1d_fused_decode_f32 pattern from cu/hybrid.cu, re-derived for the fused KDA ring
// and one plane. Never materializes the assembled window to HBM.
// Launch: grid=ceil(qkv/256), block=256.
extern "C" __global__ void memra_kda_conv_silu_decode_f32(
        const float* __restrict__ x_new,    // [qkv] this step's RAW projection row
        float* __restrict__ ring,           // [3*qkv, K-1] fused, updated in place
        const float* __restrict__ w,        // [3*qkv, K] fused
        float* __restrict__ y,              // [qkv] SiLU applied
        int qkv, int K, int plane) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv) return;
    const int pad = K - 1;
    const int fused = plane * qkv + c;
    float* rc = ring + (size_t)fused * pad;
    const float* wc = w + (size_t)fused * K;
    float win[8];
    for (int j = 0; j < pad; j++) win[j] = rc[j];
    float xv = x_new[c];
    float acc = 0.0f;
    for (int j = 0; j < pad; j++) acc += win[j] * wc[j];
    acc += xv * wc[pad];
    y[c] = memra_kda_silu(acc);
    for (int j = 0; j + 1 < pad; j++) rc[j] = win[j + 1];
    if (pad > 0) rc[pad - 1] = xv;
}

// ---- Per-channel forget gate. ----
// g = gate_lower_bound * sigmoid(exp(A_log[head]) * (f_b(f_a(x)) + dt_bias)), dt_bias per CHANNEL.
// Emitted as the RAW log-gate: the scan applies expf, matching the GDN convention and keeping
// the decay's exp on the consumer side where the recurrence needs it.
// Launch: grid=(ceil(qkv/256), T), block=256.
extern "C" __global__ void memra_kda_gate_f32(
        const float* __restrict__ f,        // [T, qkv] f_b(f_a(x))
        const float* __restrict__ dt_bias,  // [qkv]
        const float* __restrict__ a_log,    // [H]
        float* __restrict__ g,              // [T, qkv] raw log-gate
        int qkv, int T, int head_dim, float lower_bound) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= qkv || t >= T) return;
    float raw = f[(size_t)t * qkv + c] + dt_bias[c];
    float rate = expf(a_log[c / head_dim]);
    g[(size_t)t * qkv + c] = lower_bound * memra_kda_sigmoid(rate * raw);
}

// ---- The delta-rule recurrence with PER-CHANNEL decay. ----
// Per token, per head, in exactly memra-reference kimi_delta_net's order:
//   S[i][col] *= exp(g[i])                    (decay first, per key index)
//   memory[col] = sum_i S[i][col] * k[i]      (from the DECAYED state)
//   delta[col]  = (v[col] - memory[col]) * beta
//   S[i][col]  += k[i] * delta[col]
//   out[col]    = sum_i S[i][col] * q[i]
// GDN folds its decay into the memory reduction instead; that identity is scalar-only, so it is
// deliberately not ported. `scale` carries head_dim^-0.5: the reference scales q after the L2
// norm, and q feeds only the readout (never the state), so scaling the readout is exact.
// Grid: (H, 1, D/cols_per_block); block: (WARP, cols_per_block). One warp owns one output
// column; its lanes shard the D-long K dimension.
template <int D, int WARP>
__device__ void memra_kda_scan_kernel(
        const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
        const float* __restrict__ g, const float* __restrict__ beta,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        float* __restrict__ o, int H, int T, float scale) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= D) return;
    constexpr int rows_per_lane = D / WARP;

    const float* st = state_in + ((size_t)h * D + col) * D;
    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) s_shard[r] = st[r * WARP + lane];

    for (int t = 0; t < T; t++) {
        const size_t base = ((size_t)t * H + h) * D;
        const float beta_val = beta[(size_t)t * H + h];
        float k_reg[rows_per_lane], q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            int i = r * WARP + lane;
            s_shard[r] *= expf(g[base + i]);
            k_reg[r] = k[base + i];
            q_reg[r] = q[base + i];
        }
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) kv_shard += s_shard[r] * k_reg[r];
        const float kv_col = memra_kda_warp_sum<WARP>(kv_shard);
        const float delta_col = (v[base + col] - kv_col) * beta_val;
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] += k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = memra_kda_warp_sum_down<WARP>(attn_partial);
        if (lane == 0) o[base + col] = attn_col * scale;
    }

    float* so = state_out + ((size_t)h * D + col) * D;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) so[r * WARP + lane] = s_shard[r];
}

// head_dim 128 is the only KDA geometry glm5_next ships (linear_attn_config.head_dim = 128,
// research/glm53-flash-bringup-20260827/CENSUS.md). The loader refuses every other width rather
// than silently running an uninstantiated shape.
extern "C" __global__ void memra_kda_scan_s128(
        const float* q, const float* k, const float* v, const float* g, const float* beta,
        const float* state_in, float* state_out, float* o, int H, int T, float scale) {
    memra_kda_scan_kernel<128, 32>(q, k, v, g, beta, state_in, state_out, o, H, T, scale);
}

// =====================================================================================
// CHUNKED KDA PREFILL SCAN (the WY / per-channel-Gcum form) — MEMRA_KDA_CHUNKED seam.
//
// Derivation is the banked `chunk_kimi_delta_attention` reference
// (research/glm53-flash-bringup-20260827/modular_glm5_next-ref.py), NOT a transcription of the
// GDN K1-K5 chain in cu/hybrid.cu: KDA's decay is PER CHANNEL, so the cumulative gate is a
// [T, qkv] tensor Gcum (inclusive per-chunk cumsum of the raw log gate g), the pair matrices
// carry the exp INSIDE the d-reduction, and the beta convention differs (KDA's A carries the
// ROW token's beta; GDN's carries the column's).
//
// Per chunk of C tokens (local j, channels d, head h; G_j := Gcum row j):
//   A[j][i] = beta_j sum_d k_j[d] k_i[d] exp(G_j[d]-G_i[d])          (i < j, stored POSITIVE;
//             the reference's attn matrix is -A and its UT transform equals K3's subtract form:
//             T = (I + A)^{-1})
//   P[j][i] = sum_d q_j[d] k_i[d] exp(G_j[d]-G_i[d])                 (i <= j; upper tri ZERO)
//   U = T (v . beta_row),  W = T (k . beta_row . exp(G))             (both [C, D] per head)
//   sequential over chunks, S in the gdn transposed M[col][i] layout:
//     Y[j][col]  = U[j][col] - sum_i W[j][i] M[col][i]               (the delta rows)
//     o[j][col]  = sum_i q_j[i] exp(G_j[i]) S_start[i][col] + sum_{i<=j} P[j][i] Y[i][col]
//     M[col][i] <- exp(GC[i]) M[col][i] + sum_j k_j[i] exp(GC[i]-G_j[i]) Y[j][col]
//   where GC = Gcum at the chunk's LAST token, per channel.
// Every exp() argument above is <= 0 (g <= 0, so Gcum is non-increasing and j >= i keeps
// G_j - G_i <= 0), so every exp() is in (0,1] — the same no-overflow property the GDN chain's
// header states. Verified symbolically vs the sequential recurrence at C=1 and on the C=2
// cross terms; the fixture and chunk-boundary gates in tests/kda_chunked_gpu.rs are the
// numeric authority.
//
// NOT bit-identical to memra_kda_scan_s128 (chunked FP accumulation order — the GDN A4
// precedent); the acceptance bar is the scale-relative reference band, stated in the gate.
// Split-invariance: two calls split at a multiple of C are BIT-identical to one call (chunk
// grids realign; the K4 smem state round-trips through f32 global exactly), gated in tests.
//
// Kernel split (5 launches per layer call; K1-K3+K5 chunk-parallel, K4 sequential in chunks):
//   K1 memra_kda_chunk_cumgate: per-chunk per-CHANNEL inclusive cumsum of g.  [T,qkv]
//   K2 memra_kda_chunk_attn:    A (strictly-lower, positive form) and P (inclusive).
//   K3 memra_kda_chunk_solve:   forward substitution of (I+A)^{-1} on both RHS at once.
//   K4 memra_kda_chunk_state:   inter-chunk state pass; snapshots chunk-start state for K5.
//   K5 memra_kda_chunk_output:  o = inter (vs snapshot) + P @ Y, scaled.
// Layouts: q,k,v,g,gcum,o [T, qkv] token-major ((t*H+h)*D+d); beta [T,H]; A,P [NC,H,C,C]
// (((c*H+h)*C+j)*C+i); U,W,Y [NC,H,C,D]; Ssnap [NC,H,D,D] TRANSPOSED St[i][col] (K5 reads
// coalesce). C is a multiple of 32 in [32,128] (default 64).
// =====================================================================================

#define KDA_CHUNK_D 128

// K1: per-chunk per-channel inclusive cumsum of the raw log gate. grid (NC, H), block 128:
// thread d owns channel h*128+d and scans ascending t — deterministic, coalesced over d.
extern "C" __global__ void memra_kda_chunk_cumgate_f32(
        const float* __restrict__ g, float* __restrict__ gcum, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int d = threadIdx.x;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    float acc = 0.0f;
    for (int j = 0; j < Cc; j++) {
        const size_t idx = (size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D + d;
        acc += g[idx];
        gcum[idx] = acc;
    }
}

// K2: the pair matrices, with the per-channel decay INSIDE the d-reduction (the factored
// k*exp(-Gcum) form GDN uses for its scalar gate would overflow: Gcum reaches -5*C per
// channel and exp(+5*C) is inf at C >= 18). Warp-per-j-row butterfly dots over 32-row i
// sub-tiles of k and Gcum staged in smem (the GDN generic-K2 shape; the register-tiled 2x2
// twin is a tuning follow-up — this increment's authority is the band gate, and the flag
// ships OFF). grid (NC, H), block (32, 8).
extern "C" __global__ void memra_kda_chunk_attn_f32(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ gcum, const float* __restrict__ beta,
        float* __restrict__ A, float* __restrict__ P, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int lane = threadIdx.x, w = threadIdx.y;
    const int tid = w * 32 + lane;
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    __shared__ float kt[32][KDA_CHUNK_D];   // i-row k sub-tile
    __shared__ float gt[32][KDA_CHUNK_D];   // i-row Gcum sub-tile
    __shared__ float bt[128];               // chunk beta (Cc <= 128)
    if (tid < Cc) bt[tid] = beta[(size_t)(t0 + tid) * H + h];
    for (int it0 = 0; it0 < Cc; it0 += 32) {
        const int itn = min(32, Cc - it0);
        __syncthreads();
        for (int idx = tid; idx < itn * KDA_CHUNK_D; idx += 256) {
            int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
            const size_t src = (size_t)(t0 + it0 + r) * qkv + (size_t)h * KDA_CHUNK_D + d;
            kt[r][d] = k[src];
            gt[r][d] = gcum[src];
        }
        __syncthreads();
        for (int j = w; j < Cc; j += 8) {
            if (j < it0) continue;                    // pairs need i <= j
            const size_t jrow = (size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D;
            float kjr[4], qjr[4], gjr[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                kjr[r] = k[jrow + r * 32 + lane];
                qjr[r] = q[jrow + r * 32 + lane];
                gjr[r] = gcum[jrow + r * 32 + lane];
            }
            float* Arow = A + (((size_t)c * H + h) * C + j) * C;
            float* Prow = P + (((size_t)c * H + h) * C + j) * C;
            const int iend = min(itn, j - it0 + 1);   // i in [it0, min(j, it0+itn-1)]
            for (int ii = 0; ii < iend; ii++) {
                float dk = 0.0f, dq = 0.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    const int d = r * 32 + lane;
                    const float dec = expf(gjr[r] - gt[ii][d]);   // <= 1: j >= i
                    const float kv = kt[ii][d] * dec;
                    dk += kjr[r] * kv;
                    dq += qjr[r] * kv;
                }
                #pragma unroll
                for (int o2 = 16; o2 > 0; o2 >>= 1) {
                    dk += __shfl_xor_sync(0xffffffff, dk, o2);
                    dq += __shfl_xor_sync(0xffffffff, dq, o2);
                }
                if (lane == 0) {
                    const int i = it0 + ii;
                    if (i < j) Arow[i] = bt[j] * dk;  // POSITIVE form; K3 subtracts
                    Prow[i] = dq;                      // inclusive diagonal, no beta
                }
            }
        }
    }
    __syncthreads();
    // zero-fill P's upper triangle (K5's rectangular inner loop relies on it)
    for (int j = w; j < Cc; j += 8) {
        float* Prow = P + (((size_t)c * H + h) * C + j) * C;
        for (int i = j + 1 + lane; i < Cc; i += 32) Prow[i] = 0.0f;
    }
}

// K3: forward substitution R_j = RHS_j - sum_{i<j} A[j,i] R_i for both RHS at once.
// RHS_u = beta_j v_j[col]; RHS_w = beta_j k_j[col] exp(Gcum_j[col]) — KDA folds beta into
// BOTH right-hand sides (the reference's v_beta / k_beta), where GDN's convention kept beta
// in A and the K4 rank update. Same structure as gdn_chunk_solve: threads 0..127 solve U,
// 128..255 solve W; each column's history is thread-private, no __syncthreads in the solve.
// grid (NC, H), block 256.
template <int CT>
__device__ void memra_kda_chunk_solve_kernel(
        const float* __restrict__ v, const float* __restrict__ k,
        const float* __restrict__ A, const float* __restrict__ gcum,
        const float* __restrict__ beta,
        float* __restrict__ U, float* __restrict__ W, int H, int T, int c) {
    const int h = blockIdx.y;
    const int t0 = c * CT;
    const int Cc = min(CT, T - t0);
    const int tid = threadIdx.x;
    const int col = tid & (KDA_CHUNK_D - 1);
    const bool is_w = tid >= KDA_CHUNK_D;
    float* R = is_w ? W : U;
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    __shared__ float As[CT][CT];
    __shared__ float bt[CT];
    if (tid < Cc) bt[tid] = beta[(size_t)(t0 + tid) * H + h];
    for (int idx = tid; idx < Cc * CT; idx += 256) {
        int j = idx / CT, i = idx % CT;
        if (i < j) As[j][i] = A[(((size_t)c * H + h) * CT + j) * CT + i];
    }
    __syncthreads();
    const size_t rbase = ((size_t)c * H + h) * (size_t)CT * KDA_CHUNK_D;
    float hist[CT];
    if (Cc == CT) {
        // full chunk: compile-time bounds keep the history in REGISTERS (the gdn_chunk_solve
        // 3.6x-over-local-memory form)
        #pragma unroll
        for (int j = 0; j < CT; j++) {
            const size_t src = (size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D + col;
            float acc = is_w ? bt[j] * k[src] * expf(gcum[src]) : bt[j] * v[src];
            #pragma unroll
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * KDA_CHUNK_D + col] = acc;
        }
    } else {
        for (int j = 0; j < Cc; j++) {          // tail chunk: dynamic bound
            const size_t src = (size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D + col;
            float acc = is_w ? bt[j] * k[src] * expf(gcum[src]) : bt[j] * v[src];
            for (int i = 0; i < j; i++) acc -= As[j][i] * hist[i];
            hist[j] = acc;
            R[rbase + (size_t)j * KDA_CHUNK_D + col] = acc;
        }
    }
}
extern "C" __global__ void memra_kda_chunk_solve32_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        const float* beta, float* U, float* W, int H, int T) {
    memra_kda_chunk_solve_kernel<32>(v, k, A, gcum, beta, U, W, H, T, blockIdx.x);
}
extern "C" __global__ void memra_kda_chunk_solve64_f32(
        const float* v, const float* k, const float* A, const float* gcum,
        const float* beta, float* U, float* W, int H, int T) {
    memra_kda_chunk_solve_kernel<64>(v, k, A, gcum, beta, U, W, H, T, blockIdx.x);
}
// Generic (any C <= 128): thread-private history in local memory.
extern "C" __global__ void memra_kda_chunk_solve_f32(
        const float* __restrict__ v, const float* __restrict__ k,
        const float* __restrict__ A, const float* __restrict__ gcum,
        const float* __restrict__ beta,
        float* __restrict__ U, float* __restrict__ W, int H, int T, int C) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int tid = threadIdx.x;
    const int col = tid & (KDA_CHUNK_D - 1);
    const bool is_w = tid >= KDA_CHUNK_D;
    float* R = is_w ? W : U;
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    const float* Abase = A + ((size_t)c * H + h) * C * C;
    const size_t rbase = ((size_t)c * H + h) * (size_t)C * KDA_CHUNK_D;
    float hist[128];
    for (int j = 0; j < Cc; j++) {
        const size_t src = (size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D + col;
        const float bj = beta[(size_t)(t0 + j) * H + h];
        float acc = is_w ? bj * k[src] * expf(gcum[src]) : bj * v[src];
        const float* Aj = Abase + (size_t)j * C;
        for (int i = 0; i < j; i++) acc -= Aj[i] * hist[i];
        hist[j] = acc;
        R[rbase + (size_t)j * KDA_CHUNK_D + col] = acc;
    }
}

#define KDA_NSPLIT 4   // K4 state col-split (blocks per head); 128/4 = 32 cols/block

// K4: sequential inter-chunk state pass, the gdn_chunk_state_f32 shape with per-channel
// decay: the k rank-update rows are staged pre-decayed (k_j[d] * exp(GC[d]-G_j[d]), one exp
// per element — GDN's scalar gk[j] vector does not exist here), and the end-of-chunk state
// decay is per channel i (exp(GC[i])). No beta anywhere: KDA's Y already carries it (K3).
// grid (H, KDA_NSPLIT), block 256; blocks col-partition the state, fully independent.
// All accumulations ascending serial per thread — deterministic run-to-run.
extern "C" __global__ void memra_kda_chunk_state_f32(
        const float* __restrict__ k, const float* __restrict__ gcum,
        const float* __restrict__ U, const float* __restrict__ W,
        float* __restrict__ Y, float* __restrict__ Ssnap,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        int H, int T, int C) {
    constexpr int COLS = KDA_CHUNK_D / KDA_NSPLIT;   // 32
    const int h = blockIdx.x;
    const int col0 = blockIdx.y * COLS;
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    __shared__ float Ms[COLS][KDA_CHUNK_D + 4];
    __shared__ float wt[32][KDA_CHUNK_D];   // W sub-tile; step B reuses it for decayed k
    __shared__ float ys[32][COLS + 1];
    __shared__ float gcs[KDA_CHUNK_D];      // GC: Gcum at the chunk's last token, per channel
    const int tid = threadIdx.x;
    for (int idx = tid; idx < COLS * KDA_CHUNK_D; idx += 256) {
        int cl2 = idx / KDA_CHUNK_D, i = idx % KDA_CHUNK_D;
        Ms[cl2][i] = state_in[((size_t)h * KDA_CHUNK_D + col0 + cl2) * KDA_CHUNK_D + i];
    }
    __syncthreads();
    const int NC = (T + C - 1) / C;
    const int cl = tid % COLS, jr = tid / COLS;
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        if (tid < KDA_CHUNK_D) {
            gcs[tid] = gcum[(size_t)(t0 + Cc - 1) * qkv + (size_t)h * KDA_CHUNK_D + tid];
        }
        // snapshot the chunk-START state for K5's inter-chunk output term (TRANSPOSED to
        // St[i][col] so K5 reads coalesce — the gdn discipline).
        float* sc_out = Ssnap + ((size_t)c * H + h) * KDA_CHUNK_D * KDA_CHUNK_D;
        for (int idx = tid; idx < COLS * KDA_CHUNK_D; idx += 256) {
            int i = idx / COLS, cl2 = idx % COLS;
            sc_out[(size_t)i * KDA_CHUNK_D + col0 + cl2] = Ms[cl2][i];
        }
        float acc[KDA_CHUNK_D / 8];
        #pragma unroll
        for (int r = 0; r < KDA_CHUNK_D / 8; r++) acc[r] = 0.0f;
        for (int jt = 0; jt < Cc; jt += 32) {
            const int jn = min(32, Cc - jt);
            __syncthreads();
            // step A staging: W rows
            for (int idx = tid; idx < 32 * KDA_CHUNK_D; idx += 256) {
                int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
                wt[r][d] = (r < jn)
                    ? W[(((size_t)c * H + h) * C + jt + r) * KDA_CHUNK_D + d]
                    : 0.0f;
            }
            __syncthreads();
            {
                const size_t yb = (((size_t)c * H + h) * C + jt) * KDA_CHUNK_D + col0 + cl;
                const float u0 = (jr      < jn) ? U[yb + (size_t)jr * KDA_CHUNK_D] : 0.0f;
                const float u1 = (jr + 8  < jn) ? U[yb + (size_t)(jr + 8) * KDA_CHUNK_D] : 0.0f;
                const float u2 = (jr + 16 < jn) ? U[yb + (size_t)(jr + 16) * KDA_CHUNK_D] : 0.0f;
                const float u3 = (jr + 24 < jn) ? U[yb + (size_t)(jr + 24) * KDA_CHUNK_D] : 0.0f;
                float pw0 = 0.0f, pw1 = 0.0f, pw2 = 0.0f, pw3 = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < KDA_CHUNK_D; i += 4) {
                    const float4 m = *reinterpret_cast<const float4*>(&Ms[cl][i]);
                    const float4 w0 = *reinterpret_cast<const float4*>(&wt[jr][i]);
                    const float4 w1 = *reinterpret_cast<const float4*>(&wt[jr + 8][i]);
                    const float4 w2 = *reinterpret_cast<const float4*>(&wt[jr + 16][i]);
                    const float4 w3 = *reinterpret_cast<const float4*>(&wt[jr + 24][i]);
                    pw0 += w0.x * m.x + w0.y * m.y + w0.z * m.z + w0.w * m.w;
                    pw1 += w1.x * m.x + w1.y * m.y + w1.z * m.z + w1.w * m.w;
                    pw2 += w2.x * m.x + w2.y * m.y + w2.z * m.z + w2.w * m.w;
                    pw3 += w3.x * m.x + w3.y * m.y + w3.z * m.z + w3.w * m.w;
                }
                const float y0 = u0 - pw0, y1 = u1 - pw1, y2 = u2 - pw2, y3 = u3 - pw3;
                if (jr      < jn) { Y[yb + (size_t)jr * KDA_CHUNK_D] = y0;        ys[jr][cl] = y0; }
                if (jr + 8  < jn) { Y[yb + (size_t)(jr + 8) * KDA_CHUNK_D] = y1;  ys[jr + 8][cl] = y1; }
                if (jr + 16 < jn) { Y[yb + (size_t)(jr + 16) * KDA_CHUNK_D] = y2; ys[jr + 16][cl] = y2; }
                if (jr + 24 < jn) { Y[yb + (size_t)(jr + 24) * KDA_CHUNK_D] = y3; ys[jr + 24][cl] = y3; }
            }
            __syncthreads();
            // step B staging: k rows PRE-DECAYED per channel (this is where KDA departs
            // from GDN's scalar gk vector)
            for (int idx = tid; idx < 32 * KDA_CHUNK_D; idx += 256) {
                int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
                if (r < jn) {
                    const size_t src = (size_t)(t0 + jt + r) * qkv + (size_t)h * KDA_CHUNK_D + d;
                    wt[r][d] = k[src] * expf(gcs[d] - gcum[src]);   // <= 1: GC is the chunk min
                } else {
                    wt[r][d] = 0.0f;
                }
            }
            __syncthreads();
            for (int jj = 0; jj < jn; jj++) {
                const float yv = ys[jj][cl];
                #pragma unroll
                for (int r = 0; r < KDA_CHUNK_D / 8; r++)
                    acc[r] += wt[jj][jr * (KDA_CHUNK_D / 8) + r] * yv;
            }
        }
        #pragma unroll
        for (int r = 0; r < KDA_CHUNK_D / 8; r++) {
            const int i = jr * (KDA_CHUNK_D / 8) + r;
            Ms[cl][i] = expf(gcs[i]) * Ms[cl][i] + acc[r];
        }
        __syncthreads();   // Ms/gcs stable before the next chunk rewrites them
    }
    for (int idx = tid; idx < COLS * KDA_CHUNK_D; idx += 256) {
        int cl2 = idx / KDA_CHUNK_D, i = idx % KDA_CHUNK_D;
        state_out[((size_t)h * KDA_CHUNK_D + col0 + cl2) * KDA_CHUNK_D + i] = Ms[cl2][i];
    }
}

// K5: full output assembly, chunk-parallel:
//   o[j,col] = scale ( sum_i q_j[i] exp(G_j[i]) S_c[i][col]  +  sum_{i<=j} P[j,i] Y[i,col] )
// The gdn_chunk_output_f32 shape; the per-channel inter-chunk gate folds into the q staging
// (qs holds q .* exp(Gcum) — GDN's post-hoc scalar b_j block does not exist here).
// grid (NC, H, ceil(C/32)), block 256.
extern "C" __global__ void memra_kda_chunk_output_f32(
        const float* __restrict__ q, const float* __restrict__ gcum,
        const float* __restrict__ P, const float* __restrict__ Y,
        const float* __restrict__ Ssnap, float* __restrict__ o,
        int H, int T, int C, float scale) {
    const int c = blockIdx.x, h = blockIdx.y;
    const int t0 = c * C;
    const int Cc = min(C, T - t0);
    const int j0 = blockIdx.z * 32;
    if (j0 >= Cc) return;                      // uniform per block (tail chunk)
    const size_t qkv = (size_t)H * KDA_CHUNK_D;
    __shared__ float ts[32][KDA_CHUNK_D];      // phase 1: St sub-tile; phase 2: Y sub-tile
    __shared__ float qs[32][KDA_CHUNK_D];      // the block's gated q rows (zero-padded tail)
    const int tid = threadIdx.x;
    const int cg = tid % 32, rg = tid / 32;
    const int c0 = cg * 4, r0 = rg * 4;
    const int jend = min(j0 + 32, Cc);
    const int jn = jend - j0;
    for (int idx = tid; idx < 32 * KDA_CHUNK_D; idx += 256) {
        int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
        if (r < jn) {
            const size_t src = (size_t)(t0 + j0 + r) * qkv + (size_t)h * KDA_CHUNK_D + d;
            qs[r][d] = q[src] * expf(gcum[src]);   // <= 1: Gcum is non-positive
        } else {
            qs[r][d] = 0.0f;
        }
    }
    float acc[4][4];
    #pragma unroll
    for (int rr = 0; rr < 4; rr++)
        #pragma unroll
        for (int cc = 0; cc < 4; cc++) acc[rr][cc] = 0.0f;
    // phase 1: inter-chunk term (q .* exp(G)) . S_c[:,col]
    const float* st = Ssnap + ((size_t)c * H + h) * KDA_CHUNK_D * KDA_CHUNK_D;
    for (int it0 = 0; it0 < KDA_CHUNK_D; it0 += 32) {
        __syncthreads();
        for (int idx = tid; idx < 32 * KDA_CHUNK_D; idx += 256) {
            int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
            ts[r][d] = st[(size_t)(it0 + r) * KDA_CHUNK_D + d];
        }
        __syncthreads();
        #pragma unroll 4
        for (int ii = 0; ii < 32; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float qv = qs[r0 + rr][it0 + ii];
                acc[rr][0] += qv * tv.x; acc[rr][1] += qv * tv.y;
                acc[rr][2] += qv * tv.z; acc[rr][3] += qv * tv.w;
            }
        }
    }
    // phase 2: intra-chunk term P @ Y (rectangular: P upper triangle is ZERO by the K2
    // contract, so no per-row bounds in the inner loop)
    for (int it0 = 0; it0 < jend; it0 += 32) {
        const int itn = min(32, jend - it0);
        __syncthreads();
        for (int idx = tid; idx < 32 * KDA_CHUNK_D; idx += 256) {
            int r = idx / KDA_CHUNK_D, d = idx % KDA_CHUNK_D;
            ts[r][d] = (r < itn)
                ? Y[(((size_t)c * H + h) * C + it0 + r) * KDA_CHUNK_D + d]
                : 0.0f;
        }
        __syncthreads();
        const float* P0 = P + (((size_t)c * H + h) * C + j0 + r0) * C + it0;
        for (int ii = 0; ii < itn; ii++) {
            const float4 tv = *reinterpret_cast<const float4*>(&ts[ii][c0]);
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                const float pv = (r0 + rr < jn) ? P0[(size_t)rr * C + ii] : 0.0f;
                acc[rr][0] += pv * tv.x; acc[rr][1] += pv * tv.y;
                acc[rr][2] += pv * tv.z; acc[rr][3] += pv * tv.w;
            }
        }
    }
    #pragma unroll
    for (int rr = 0; rr < 4; rr++) {
        const int j = j0 + r0 + rr;
        if (j < jend) {
            const float4 ov = make_float4(scale * acc[rr][0], scale * acc[rr][1],
                                          scale * acc[rr][2], scale * acc[rr][3]);
            *reinterpret_cast<float4*>(
                &o[(size_t)(t0 + j) * qkv + (size_t)h * KDA_CHUNK_D + c0]) = ov;
        }
    }
}

// ---- Sigmoid-gated fp32 RMSNorm over head_dim. ----
// dst = RMSNorm(core, w, eps) * sigmoid(gate). cu/hybrid.cu's gated_rmsnorm_f32 is the same
// reduction with a SiLU gate; KDA's Glm5NextTextRMSNormGated hardcodes sigmoid, so the gate
// activation is the whole difference and it is not a flag.
// One block per row (row = t*H + h), ncols = head_dim.
extern "C" __global__ void memra_kda_gated_rmsnorm_f32(
        const float* __restrict__ core, const float* __restrict__ w,
        const float* __restrict__ gate, float* __restrict__ dst,
        int ncols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* crow = core + (size_t)row * ncols;
    const float* grow = gate + (size_t)row * ncols;
    float* drow = dst + (size_t)row * ncols;
    float sum = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) { float v = crow[i]; sum += v * v; }
    __shared__ float s[32];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((tid & 31) == 0) s[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s[tid] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (tid == 0) s[0] = v;
    }
    __syncthreads();
    float scale = rsqrtf(s[0] / ncols + eps);
    for (int i = tid; i < ncols; i += blockDim.x) {
        drow[i] = (crow[i] * scale * w[i]) * memra_kda_sigmoid(grow[i]);
    }
}
