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

// ---- T=1 decode, ALL THREE PLANES in one launch (lane/glm5-kda-conv3-20260904, door
// MEMRA_KDA_CONV3). The per-plane kernel above is 105 launches per plain glm5_next token on the
// 2x B200 pair (door-ON census 2026-09-04: 3 per KDA layer x 34 layers, 2.9 us mean each,
// 32 blocks of 256 threads apiece), three independent launches whose only relation is the
// plane index. This kernel is that body with `plane = blockIdx.y`: per channel the SAME
// window read, the SAME `acc` chain in the SAME order, the SAME SiLU, the SAME ring roll on
// the SAME `fused` row, so every output byte and every ring byte is what the three launches
// write. Launch: grid=(ceil(qkv/256), 3), block=256.
extern "C" __global__ void memra_kda_conv_silu_decode3_f32(
        const float* __restrict__ x0, const float* __restrict__ x1,
        const float* __restrict__ x2,
        float* __restrict__ ring,           // [3*qkv, K-1] fused, updated in place
        const float* __restrict__ w,        // [3*qkv, K] fused
        float* __restrict__ y0, float* __restrict__ y1, float* __restrict__ y2,
        int qkv, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= qkv) return;
    const int plane = blockIdx.y;
    const float* x_new = plane == 0 ? x0 : (plane == 1 ? x1 : x2);
    float* y = plane == 0 ? y0 : (plane == 1 ? y1 : y2);
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

// memra_kda_gate_f32 and sigmoid_f32(beta_raw) in ONE launch (lane/launch-collapse-20260906):
// the two gates read different tensors and were two launches per layer (34 pairs per token on
// GLM-5.3-Flash, in-graph). Threads c < qkv run the forget-gate body above verbatim; threads
// c < heads (heads = qkv / head_dim <= qkv) also run sigmoid_f32's expression verbatim on
// beta_raw[t, c]. Same grid as memra_kda_gate_f32. Gate: tests/kda_small_folds_gpu.rs.
extern "C" __global__ void memra_kda_gate_beta_f32(
        const float* __restrict__ f,         // [T, qkv] f_b(f_a(x))
        const float* __restrict__ dt_bias,   // [qkv]
        const float* __restrict__ a_log,     // [H]
        float* __restrict__ g,               // [T, qkv] raw log-gate
        const float* __restrict__ beta_raw,  // [T, H]
        float* __restrict__ beta,            // [T, H] sigmoid(beta_raw)
        int qkv, int T, int head_dim, int heads, float lower_bound) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= T) return;
    if (c < heads) {
        float x = beta_raw[(size_t)t * heads + c];
        beta[(size_t)t * heads + c] = 1.0f / (1.0f + expf(-x));
    }
    if (c >= qkv) return;
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

// ---- The same gated norm emitting its q8_1 quantization beside the f32 row (lane/kda-onorm-zq8-
// 20260905): dst (f32, byte-identical to the kernel above: same reduction, same
// `(c*scale*w)*sigmoid(g)` expression per element) plus `out_q` / `out_d` in `quantize_q8_1`'s
// layout and arithmetic (per 32-block amax via shfl_xor, d = amax/127, id = 1/d, rint(v*id);
// cu/qmatvec.cu:589), so the pair the `wo` MMVQ consumes is bit-identical to running
// quantize_q8_1 over dst. One block per (t, h) row of head_dim: with ncols % 32 == 0 every
// 32-lane warp round holds one contiguous q8 block, and because the token's wo input is the
// heads' rows laid end to end, block (t*H+h, blk) IS token block (t, h*ncols/32 + blk) of the
// [t, H*ncols] activation: the same bytes at the same offsets. Saves the standalone quantize
// launch per KDA layer (34 per token on GLM-5.3-Flash). Requires ncols % 32 == 0.
extern "C" __global__ void memra_kda_gated_rmsnorm_zq8_f32(
        const float* __restrict__ core, const float* __restrict__ w,
        const float* __restrict__ gate, float* __restrict__ dst,
        signed char* __restrict__ out_q, float* __restrict__ out_d,
        int ncols, float eps) {
    int row = blockIdx.x, tid = threadIdx.x;
    const float* crow = core + (size_t)row * ncols;
    const float* grow = gate + (size_t)row * ncols;
    float* drow = dst + (size_t)row * ncols;
    signed char* qrow = out_q + (size_t)row * ncols;
    float* dd = out_d + (size_t)row * (ncols >> 5);
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
    int lane = tid & 31;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float v = (crow[i] * scale * w[i]) * memra_kda_sigmoid(grow[i]);
        drow[i] = v;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        float d = amax / 127.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        qrow[i] = (signed char)__float2int_rn(v * id);
        if (lane == 0) dd[i >> 5] = d;
    }
}
