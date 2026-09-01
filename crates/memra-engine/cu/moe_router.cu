// EDGE-1 §A: fused MoE router. Replaces the host dtoh + softmax-256 + stable DESC top-8 sort +
// renorm in hybrid_forward.rs (~281-298). One CTA per token row, blockDim = n_expert (256) =
// one thread per expert. Reproduces the Stage-1 host numerics EXACTLY so the selected experts +
// renormalized weights are bit-identical (the argmax-1178 gate depends on this).
//
// Host path (the oracle this matches):
//   maxl = max over 256 logits
//   probs[i] = exp(logit[i] - maxl);  den = sum;  probs[i] /= den          (softmax over 256)
//   sort idx DESC by (probs[b].total_cmp(probs[a]).then(a.cmp(b)))         (prob DESC, idx ASC)
//   sel = idx[..8]
//   w[j] = probs[sel[j]];  ws = sum(w);  ws = max(ws, 6.103515625e-5);  w[j] /= ws
//
// Tie handling: iterative argmax over n_used rounds. Each round picks the expert with the
// largest prob; ties broken by SMALLEST index (matches the host `.then(a.cmp(b))`). The chosen
// expert is masked to -INF for the next round. This reproduces the stable DESC sort's top-k.
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Block reduce to find argmax of `val` with smallest-index tiebreak. Each thread brings (val, idx).
// Returns the winning (val, idx) to ALL threads via shared memory.
// We encode the comparison as: a beats b iff (a.val > b.val) || (a.val == b.val && a.idx < b.idx).
extern "C" __global__ void moe_router_topk_f32(
    const float* __restrict__ logits,   // [t, n_expert]
    int*   __restrict__ sel_idx,         // [t, n_used]  (out)
    float* __restrict__ sel_w,           // [t, n_used]  (out)
    int n_expert,                        // 256
    int n_used)                          // 8
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;         // one thread per expert, tid in [0, n_expert)
    const float* lg = logits + (size_t)row * n_expert;

    // shared scratch: per-warp partials for reductions + the running prob array.
    extern __shared__ float smem[];      // unused (we use static below)
    (void)smem;
    __shared__ float s_val[32];          // per-warp reduce scratch (max 32 warps = 1024 threads)
    __shared__ int   s_idx[32];
    __shared__ float s_max;              // block max logit
    __shared__ float s_den;              // softmax denominator
    __shared__ float s_pick_val;         // winning prob this round
    __shared__ int   s_pick_idx;         // winning expert this round
    __shared__ float s_wsum;             // accumulated weight sum over picked experts

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (n_expert + 31) >> 5;
    const unsigned FULL = 0xffffffffu;

    // thread's own logit (threads with tid < n_expert only; blockDim == n_expert so all valid).
    float my_logit = (tid < n_expert) ? lg[tid] : -FLT_MAX;

    // ---- 1. block-max reduce over 256 (matches row.iter().fold(NEG_INF, max)) ----
    float v = my_logit;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    if (lane == 0) s_val[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(FULL, t, o));
        if (lane == 0) s_max = t;
    }
    __syncthreads();
    const float maxl = s_max;

    // ---- 2. exp(l - max), block-sum denom ----
    float my_exp = (tid < n_expert) ? expf(my_logit - maxl) : 0.0f;
    float sden = my_exp;
    for (int o = 16; o > 0; o >>= 1) sden += __shfl_down_sync(FULL, sden, o);
    if (lane == 0) s_val[warp] = sden;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(FULL, t, o);
        if (lane == 0) s_den = t;
    }
    __syncthreads();
    const float den = s_den;

    // my probability (the unbiased softmax prob used both as the top-k key AND the weight).
    float my_prob = my_exp / den;     // tid >= n_expert -> 0 (never picked: prob 0, masked below)
    // masked working copy for iterative argmax (winner -> -INF so it can't be re-picked).
    float work = (tid < n_expert) ? my_prob : -FLT_MAX;

    if (tid == 0) s_wsum = 0.0f;
    __syncthreads();

    // ---- 3. iterative argmax: n_used rounds, prob DESC, smallest-index tiebreak ----
    for (int j = 0; j < n_used; ++j) {
        // warp-level argmax with smallest-index tiebreak
        float bv = work;
        int   bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(FULL, bv, o);
            int   oi = __shfl_down_sync(FULL, bi, o);
            // pick other if its val is strictly greater, OR equal val with smaller index
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float t  = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
            int   ti = (lane < nwarps) ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_down_sync(FULL, t, o);
                int   oi = __shfl_down_sync(FULL, ti, o);
                if (ov > t || (ov == t && oi < ti)) { t = ov; ti = oi; }
            }
            if (lane == 0) { s_pick_val = t; s_pick_idx = ti; }
        }
        __syncthreads();

        int   pick_idx = s_pick_idx;
        // gather the WINNER's unbiased prob (== work value before masking == my_prob at pick_idx)
        // s_pick_val is exactly that (work==my_prob for picked, not yet masked this round).
        float pick_prob = s_pick_val;

        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            sel_w[(size_t)row * n_used + j]   = pick_prob;   // raw prob; renormalized below
            s_wsum += pick_prob;
        }
        // mask the winner for the next round
        if (tid == pick_idx) work = -FLT_MAX;
        __syncthreads();
    }

    // ---- 4. renorm: ws = max(sum, F16_MIN_NORMAL) BEFORE divide ----
    if (tid == 0) {
        float ws = fmaxf(s_wsum, 6.103515625e-5f);   // F16 smallest normal, clamp before divide
        for (int j = 0; j < n_used; ++j) {
            sel_w[(size_t)row * n_used + j] /= ws;
        }
    }
}

/*
 * The expf core below is adapted for CUDA from Arm Optimized Routines v21.02:
 * https://github.com/ARM-software/optimized-routines/blob/v21.02/math/expf.c
 * https://github.com/ARM-software/optimized-routines/blob/v21.02/math/exp2f_data.c
 *
 * Copyright (c) 2017-2019, Arm Limited.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
__device__ __constant__ unsigned long long SIG_EXP2_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};

// CUDA transcription of the scalar expf evaluation used by the x86_64 glibc host oracle. Explicit
// RN and FMA operations align the frozen router corpus and production golden; this is not a claim
// of universal host/device libm bit parity. Router logits are finite in a valid model; the special
// cases retain sensible diagnostic behavior outside the measured corpus.
static __device__ __forceinline__ float sigmoid_host_expf(float x) {
    const float inf = __int_as_float(0x7f800000);
    if (isnan(x)) return x + x;
    if (x == inf) return inf;
    if (x == -inf) return 0.0f;
    if (x > 0x1.62e42ep6f) return inf;
    if (x < -0x1.9fe368p6f) return 0.0f;

    const double inv_ln2_n = 0x1.71547652b82fep+5;
    const double shift = 0x1.8p+52;
    const double c0 = 0x1.c6af84b912394p-20;
    const double c1 = 0x1.ebfce50fac4f3p-13;
    const double c2 = 0x1.62e42ff0c52d6p-6;
    const double xd = (double)x;
    const double scaled = __dmul_rn(inv_ln2_n, xd);
    double kd = __dadd_rn(scaled, shift);
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = __dsub_rn(kd, shift);
    const double r = __dsub_rn(scaled, kd);
    unsigned long long bits = SIG_EXP2_TAB[ki & 31ULL];
    bits += ki << 47;
    const double s = __longlong_as_double((long long)bits);
    const double z = fma(c0, r, c1);
    const double r2 = __dmul_rn(r, r);
    double y = fma(c2, r, 1.0);
    y = fma(z, r2, y);
    y = __dmul_rn(y, s);
    return (float)y;
}

// SwiGLU epilogue matching the host oracle's scalar operation order:
// gate / (1 + exp(-gate)) * up. The exp implementation above is qualified only
// against Memra's frozen x86_64 glibc corpus; this kernel is not a universal
// cross-libm parity claim.
extern "C" __global__ void silu_mul_host_expf_f32(
    const float* gate,
    const float* up,
    float* dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float denominator = __fadd_rn(1.0f, sigmoid_host_expf(-g));
        dst[i] = __fmul_rn(__fdiv_rn(g, denominator), up[i]);
    }
}

// Step-3.7's final routed layers clamp the two SwiGLU operands differently:
//   min(silu(gate), limit) * clamp(up, -limit, limit).
// Keep the host-oracle expf transcription and explicit RN operation order from the
// unclamped kernel above. Inputs produced by a valid expert matmul are finite.
extern "C" __global__ void silu_clamped_mul_host_expf_f32(
    const float* gate,
    const float* up,
    float limit,
    float* dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        const float denominator = __fadd_rn(1.0f, sigmoid_host_expf(-g));
        float silu = __fdiv_rn(g, denominator);
        silu = silu > limit ? limit : silu;
        float linear = up[i];
        linear = linear > limit ? limit : linear;
        linear = linear < -limit ? -limit : linear;
        dst[i] = __fmul_rn(silu, linear);
    }
}

// Step-3.7 / DeepSeek-V3-class sigmoid router. The host oracle contract is:
//   score[i] = 1 / (1 + exp(-logit[i]))
//   selection key = score[i] + correction_bias[i]
//   inactive original ids are removed before top-k
//   key DESC, original id ASC on exact ties
//   weight = un-biased score; optional slot-order normalization; then scale
//
// `correction_bias` is a resident zero row when the checkpoint has no bias, and `active` is a
// resident all-one row when there is no pruning overlay. The launcher rounds blockDim up to a
// whole warp, so FULL shuffle masks remain valid for expert counts such as Step's 288.
// DEVICE-EXPF twin (MEMRA_SIG_EXPF_DEV=1): identical round-robin selection structure, but
// score = 1/(1+expf(-logit)) with the DEVICE libm expf instead of the double-precision
// host-glibc transcription — sigmoid_host_expf runs ~6 FP64 ops per expert and FP64 is
// 1/64-rate on the RTX PRO 6000 class, which is the whole 18us wall of the t=1 router.
// NUMERIC-CLASS door: keys/weights shift by ULPs, near-tie selections can flip -> new tape
// + battery arbitrate, exactly the QKV_FUSED/DEV_ROUTER acceptance class.
extern "C" __global__ void moe_router_sigmoid_topk_f32_dexp(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    const unsigned char* __restrict__ active,
    int*   __restrict__ sel_idx,
    float* __restrict__ sel_w,
    int n_expert,
    int n_used,
    float scaling_factor,
    int route_norm)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned FULL = 0xffffffffu;
    const float* lg = logits + (size_t)row * n_expert;

    __shared__ float s_val[32];
    __shared__ int   s_idx[32];
    __shared__ int   s_pick_idx;
    __shared__ float s_pick_w[32];   // n_used <= n_expert <= 1024, launcher caps n_used at 32
    __shared__ float s_score[1024];

    const bool live = tid < n_expert && active[tid] != 0;
    const float score = live ? 1.0f / (1.0f + expf(-lg[tid])) : 0.0f;
    s_score[tid] = score;
    float work = live ? score + correction_bias[tid] : -FLT_MAX;
    __syncthreads();

    for (int j = 0; j < n_used; ++j) {
        float bv = work;
        int bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(FULL, bv, o);
            const int oi = __shfl_down_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float v = lane < nwarps ? s_val[lane] : -FLT_MAX;
            int i = lane < nwarps ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(FULL, v, o);
                const int oi = __shfl_down_sync(FULL, i, o);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane == 0) { s_pick_idx = i; }
        }
        __syncthreads();

        const int pick_idx = s_pick_idx;
        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            // SHARED PICK CACHE (2026-08-23): the tail below used to re-READ these weights
            // from global sel_w (8 dependent loads for the sum, then 8 more for the
            // normalize) on a single thread — nsys measured this ONE-BLOCK kernel at 16.3us
            // per MoE layer (0.68 ms/token) for ~1us of arithmetic. Keeping the picks in
            // shared costs nothing and the tail's FP expressions are unchanged, so the
            // outputs are bit-identical.
            s_pick_w[j] = s_score[pick_idx];
            sel_w[(size_t)row * n_used + j] = s_score[pick_idx];
        }
        if (tid == pick_idx) { work = -FLT_MAX; }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < n_used; ++j) {
            sum += s_pick_w[j];
        }
        if (route_norm) {
            const float den = fmaxf(sum, 1e-20f);
            for (int j = 0; j < n_used; ++j) {
                sel_w[(size_t)row * n_used + j] = s_pick_w[j] / den * scaling_factor;
            }
        } else {
            for (int j = 0; j < n_used; ++j) {
                sel_w[(size_t)row * n_used + j] = s_pick_w[j] * scaling_factor;
            }
        }
    }
}

// BARRIER-LEAN twin (MEMRA_TOPK_FAST=1): the round-robin kernel above pays n_used rounds x
// 3 __syncthreads (24 barriers for Step's top-8 = its 18us wall at t=1). This twin computes
// each WARP's local top-8 barrier-free (8 masked warp-argmax passes), then ONE barrier and a
// warp-0 merge over the <=nwarps*8 candidates. Selection is EXACT (key DESC, id ASC ties);
// score/weight/normalization arithmetic is byte-for-byte the round-robin kernel's, so the
// outputs are identical — a latency twin, not a numeric door.
// PEER DOORBELL RING (MEMRA_FENCE_RANK1=1). The TP joins wait on a CROSS-DEVICE event: rank1
// records, e (dev0) waits. nsys puts 26 waits/token >20us on dev0 totalling ~1.15 ms/token of
// idle while rank1's work is already smaller than dev0's — it is event latency, not work. The
// memops receipt (2026-08-23) established the legal directions: a peer stream MEMOP is rejected
// over PCIe P2P, but a peer KERNEL STORE into root memory is exactly what the direct join
// already relies on. So rank1 rings a flag in ROOT memory with this kernel and e waits it with
// a SAME-DEVICE cuStreamWaitValue32 (GEQ). Ordering only — no value changes anywhere.
// __threadfence_system before the store; PCIe posted writes from one device land in order, so
// rank1's accumulator/partial stores are visible before the flag write arrives.
extern "C" __global__ void memra_ring_flag(unsigned long long p, unsigned int v) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        *reinterpret_cast<volatile unsigned int*>(p) = v;
    }
}

// SELECTION MIRROR (MEMRA_SEL_MIRROR=1): sel (n int32) + route weights (n f32) copied in ONE
// launch. It replaces two 32-BYTE cuMemcpyAsync D2D copies per rank per MoE layer — nsys on the
// turn8-context decode measured 167 such copies per token at 3.79us each (0.63 ms/token of
// copy-engine dispatch on the serialized rank stream) for 32 bytes of payload, against ~2us for
// a launch. Byte-for-byte identical values.
extern "C" __global__ void moe_sel_w_mirror(
        const int* __restrict__ sel_src, const float* __restrict__ w_src,
        int* __restrict__ sel_dst, float* __restrict__ w_dst, int n) {
    const int i = threadIdx.x;
    if (i < n) {
        sel_dst[i] = sel_src[i];
        w_dst[i]   = w_src[i];
    }
}

// DEXP + BARRIER-LEAN twin (MEMRA_SIG_EXPF_DEV=1 + MEMRA_TOPK_FAST=1): the _fast
// twin's warp-local top-k + single-merge structure over the _dexp scoring class
// (device-libm expf sigmoid). With dexp the 288 sigmoids are cheap ALU and the
// round-robin kernel's remaining wall is its n_used x 3 barriers — exactly what the
// _fast structure removes. Selection is EXACT (key DESC, id ASC) and the weight tail
// is byte-for-byte the _dexp kernel's: outputs identical to the serving _dexp arm —
// a latency twin within the already-qualified dexp numeric class, not a new door.
extern "C" __global__ void moe_router_sigmoid_topk_f32_dexp_fast(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    const unsigned char* __restrict__ active,
    int*   __restrict__ sel_idx,
    float* __restrict__ sel_w,
    int n_expert,
    int n_used,
    float scaling_factor,
    int route_norm)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned FULL = 0xffffffffu;
    const float* lg = logits + (size_t)row * n_expert;

    __shared__ float s_score[1024];
    __shared__ float c_val[32 * 8];
    __shared__ int   c_idx[32 * 8];

    const bool live = tid < n_expert && active[tid] != 0;
    const float score = live ? 1.0f / (1.0f + expf(-lg[tid])) : 0.0f;
    if (tid < n_expert) s_score[tid] = score;
    float work = live ? score + correction_bias[tid] : -FLT_MAX;

    // Warp-local top-n_used: n_used masked argmax passes, no barriers.
    for (int j = 0; j < n_used; ++j) {
        float bv = work;
        int bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(FULL, bv, o);
            const int oi = __shfl_down_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        bv = __shfl_sync(FULL, bv, 0);
        bi = __shfl_sync(FULL, bi, 0);
        if (lane == 0) { c_val[warp * 8 + j] = bv; c_idx[warp * 8 + j] = bi; }
        if (tid == bi) { work = -FLT_MAX; }
    }
    __syncthreads();

    // Warp 0 merges the nwarps*n_used candidates with the same key-DESC id-ASC order.
    if (warp == 0) {
        const int ncand = nwarps * n_used;
        float mv[8];
        int   mi[8];
        int   mine = 0;
        for (int c = lane; c < ncand; c += 32) {
            mv[mine] = c_val[c];
            mi[mine] = c_idx[c];
            mine++;
        }
        for (int j = 0; j < n_used; ++j) {
            float bv = -FLT_MAX;
            int bi = 0x7fffffff;
            for (int k = 0; k < mine; ++k) {
                if (mv[k] > bv || (mv[k] == bv && mi[k] < bi)) { bv = mv[k]; bi = mi[k]; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(FULL, bv, o);
                const int oi = __shfl_down_sync(FULL, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            bv = __shfl_sync(FULL, bv, 0);
            bi = __shfl_sync(FULL, bi, 0);
            if (lane == 0) {
                sel_idx[(size_t)row * n_used + j] = bi;
                sel_w[(size_t)row * n_used + j] = s_score[bi];
            }
            for (int k = 0; k < mine; ++k) {
                if (mi[k] == bi && mv[k] == bv) { mv[k] = -FLT_MAX; mi[k] = 0x7fffffff; }
            }
        }
        if (lane == 0) {
            float sum = 0.0f;
            for (int j = 0; j < n_used; ++j) {
                sum += sel_w[(size_t)row * n_used + j];
            }
            if (route_norm) {
                const float den = fmaxf(sum, 1e-20f);
                for (int j = 0; j < n_used; ++j) {
                    sel_w[(size_t)row * n_used + j] =
                        sel_w[(size_t)row * n_used + j] / den * scaling_factor;
                }
            } else {
                for (int j = 0; j < n_used; ++j) {
                    sel_w[(size_t)row * n_used + j] *= scaling_factor;
                }
            }
        }
    }
}

extern "C" __global__ void moe_router_sigmoid_topk_f32_fast(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    const unsigned char* __restrict__ active,
    int*   __restrict__ sel_idx,
    float* __restrict__ sel_w,
    int n_expert,
    int n_used,
    float scaling_factor,
    int route_norm)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned FULL = 0xffffffffu;
    const float* lg = logits + (size_t)row * n_expert;

    __shared__ float s_score[1024];
    __shared__ float c_val[32 * 8];
    __shared__ int   c_idx[32 * 8];

    const bool live = tid < n_expert && active[tid] != 0;
    const float exp_neg = live ? sigmoid_host_expf(-lg[tid]) : 0.0f;
    const float score = live ? 1.0f / (1.0f + exp_neg) : 0.0f;
    if (tid < n_expert) s_score[tid] = score;
    float work = live ? score + correction_bias[tid] : -FLT_MAX;

    // Warp-local top-n_used: n_used masked argmax passes, no barriers.
    for (int j = 0; j < n_used; ++j) {
        float bv = work;
        int bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(FULL, bv, o);
            const int oi = __shfl_down_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        bv = __shfl_sync(FULL, bv, 0);
        bi = __shfl_sync(FULL, bi, 0);
        if (lane == 0) { c_val[warp * 8 + j] = bv; c_idx[warp * 8 + j] = bi; }
        if (tid == bi) { work = -FLT_MAX; }
    }
    __syncthreads();

    // Warp 0 merges the nwarps*n_used candidates with the same key-DESC id-ASC order.
    if (warp == 0) {
        const int ncand = nwarps * n_used;
        // Each lane owns candidates lane, lane+32, ... (<= 8 for 288 experts / 9 warps).
        float mv[8];
        int   mi[8];
        int   mine = 0;
        for (int c = lane; c < ncand; c += 32) {
            mv[mine] = c_val[c];
            mi[mine] = c_idx[c];
            mine++;
        }
        for (int j = 0; j < n_used; ++j) {
            float bv = -FLT_MAX;
            int bi = 0x7fffffff;
            for (int k = 0; k < mine; ++k) {
                if (mv[k] > bv || (mv[k] == bv && mi[k] < bi)) { bv = mv[k]; bi = mi[k]; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(FULL, bv, o);
                const int oi = __shfl_down_sync(FULL, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            bv = __shfl_sync(FULL, bv, 0);
            bi = __shfl_sync(FULL, bi, 0);
            if (lane == 0) {
                sel_idx[(size_t)row * n_used + j] = bi;
                sel_w[(size_t)row * n_used + j] = s_score[bi];
            }
            // Retire the picked candidate from its owning lane.
            for (int k = 0; k < mine; ++k) {
                if (mi[k] == bi && mv[k] == bv) { mv[k] = -FLT_MAX; mi[k] = 0x7fffffff; }
            }
        }
        // Match the host's selected-slot accumulation and expression order on one thread
        // (byte-for-byte the round-robin kernel's tail).
        if (lane == 0) {
            float sum = 0.0f;
            for (int j = 0; j < n_used; ++j) {
                sum += sel_w[(size_t)row * n_used + j];
            }
            if (route_norm) {
                const float den = fmaxf(sum, 1e-20f);
                for (int j = 0; j < n_used; ++j) {
                    sel_w[(size_t)row * n_used + j] =
                        sel_w[(size_t)row * n_used + j] / den * scaling_factor;
                }
            } else {
                for (int j = 0; j < n_used; ++j) {
                    sel_w[(size_t)row * n_used + j] *= scaling_factor;
                }
            }
        }
    }
}

extern "C" __global__ void moe_router_sigmoid_topk_f32(
    const float* __restrict__ logits,             // [t, n_expert]
    const float* __restrict__ correction_bias,    // [n_expert]
    const unsigned char* __restrict__ active,     // [n_expert], 0 = masked
    int*   __restrict__ sel_idx,                  // [t, n_used] (out)
    float* __restrict__ sel_w,                    // [t, n_used] (out)
    int n_expert,
    int n_used,
    float scaling_factor,
    int route_norm)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const unsigned FULL = 0xffffffffu;
    const float* lg = logits + (size_t)row * n_expert;

    __shared__ float s_val[32];
    __shared__ int   s_idx[32];
    __shared__ int   s_pick_idx;
    __shared__ float s_pick_w[32];   // n_used <= n_expert <= 1024, launcher caps n_used at 32
    __shared__ float s_score[1024];

    const bool live = tid < n_expert && active[tid] != 0;
    const float exp_neg = live ? sigmoid_host_expf(-lg[tid]) : 0.0f;
    const float score = live ? 1.0f / (1.0f + exp_neg) : 0.0f;
    s_score[tid] = score;
    float work = live ? score + correction_bias[tid] : -FLT_MAX;
    __syncthreads();

    for (int j = 0; j < n_used; ++j) {
        float bv = work;
        int bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(FULL, bv, o);
            const int oi = __shfl_down_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float v = lane < nwarps ? s_val[lane] : -FLT_MAX;
            int i = lane < nwarps ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(FULL, v, o);
                const int oi = __shfl_down_sync(FULL, i, o);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane == 0) { s_pick_idx = i; }
        }
        __syncthreads();

        const int pick_idx = s_pick_idx;
        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            // SHARED PICK CACHE (2026-08-23): the tail below used to re-READ these weights
            // from global sel_w (8 dependent loads for the sum, then 8 more for the
            // normalize) on a single thread — nsys measured this ONE-BLOCK kernel at 16.3us
            // per MoE layer (0.68 ms/token) for ~1us of arithmetic. Keeping the picks in
            // shared costs nothing and the tail's FP expressions are unchanged, so the
            // outputs are bit-identical.
            s_pick_w[j] = s_score[pick_idx];
            sel_w[(size_t)row * n_used + j] = s_score[pick_idx];
        }
        if (tid == pick_idx) { work = -FLT_MAX; }
        __syncthreads();
    }

    // Match the host's selected-slot accumulation and expression order on one thread.
    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < n_used; ++j) {
            sum += s_pick_w[j];
        }
        if (route_norm) {
            const float den = fmaxf(sum, 1e-20f);
            for (int j = 0; j < n_used; ++j) {
                sel_w[(size_t)row * n_used + j] = s_pick_w[j] / den * scaling_factor;
            }
        } else {
            for (int j = 0; j < n_used; ++j) {
                sel_w[(size_t)row * n_used + j] = s_pick_w[j] * scaling_factor;
            }
        }
    }
}

// gemma4 twin: per-expert output scale folded into the renorm write.
extern "C" __global__ void moe_router_topk_scaled_f32(
    const float* __restrict__ logits,   // [t, n_expert]
    int*   __restrict__ sel_idx,         // [t, n_used]  (out)
    float* __restrict__ sel_w,           // [t, n_used]  (out)
    int n_expert,                        // 256
    int n_used,                          // 8
    const float* __restrict__ ex_scale)  // [n_expert] gemma4 per-expert output scale
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;         // one thread per expert, tid in [0, n_expert)
    const float* lg = logits + (size_t)row * n_expert;

    // shared scratch: per-warp partials for reductions + the running prob array.
    extern __shared__ float smem[];      // unused (we use static below)
    (void)smem;
    __shared__ float s_val[32];          // per-warp reduce scratch (max 32 warps = 1024 threads)
    __shared__ int   s_idx[32];
    __shared__ float s_max;              // block max logit
    __shared__ float s_den;              // softmax denominator
    __shared__ float s_pick_val;         // winning prob this round
    __shared__ int   s_pick_idx;
    __shared__ float s_pick_w[32];   // n_used <= n_expert <= 1024, launcher caps n_used at 32         // winning expert this round
    __shared__ float s_wsum;             // accumulated weight sum over picked experts

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = (n_expert + 31) >> 5;
    const unsigned FULL = 0xffffffffu;

    // thread's own logit (threads with tid < n_expert only; blockDim == n_expert so all valid).
    float my_logit = (tid < n_expert) ? lg[tid] : -FLT_MAX;

    // ---- 1. block-max reduce over 256 (matches row.iter().fold(NEG_INF, max)) ----
    float v = my_logit;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    if (lane == 0) s_val[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
        for (int o = 16; o > 0; o >>= 1) t = fmaxf(t, __shfl_down_sync(FULL, t, o));
        if (lane == 0) s_max = t;
    }
    __syncthreads();
    const float maxl = s_max;

    // ---- 2. exp(l - max), block-sum denom ----
    float my_exp = (tid < n_expert) ? expf(my_logit - maxl) : 0.0f;
    float sden = my_exp;
    for (int o = 16; o > 0; o >>= 1) sden += __shfl_down_sync(FULL, sden, o);
    if (lane == 0) s_val[warp] = sden;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarps) ? s_val[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(FULL, t, o);
        if (lane == 0) s_den = t;
    }
    __syncthreads();
    const float den = s_den;

    // my probability (the unbiased softmax prob used both as the top-k key AND the weight).
    float my_prob = my_exp / den;     // tid >= n_expert -> 0 (never picked: prob 0, masked below)
    // masked working copy for iterative argmax (winner -> -INF so it can't be re-picked).
    float work = (tid < n_expert) ? my_prob : -FLT_MAX;

    if (tid == 0) s_wsum = 0.0f;
    __syncthreads();

    // ---- 3. iterative argmax: n_used rounds, prob DESC, smallest-index tiebreak ----
    for (int j = 0; j < n_used; ++j) {
        // warp-level argmax with smallest-index tiebreak
        float bv = work;
        int   bi = tid;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_down_sync(FULL, bv, o);
            int   oi = __shfl_down_sync(FULL, bi, o);
            // pick other if its val is strictly greater, OR equal val with smaller index
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { s_val[warp] = bv; s_idx[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            float t  = (lane < nwarps) ? s_val[lane] : -FLT_MAX;
            int   ti = (lane < nwarps) ? s_idx[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_down_sync(FULL, t, o);
                int   oi = __shfl_down_sync(FULL, ti, o);
                if (ov > t || (ov == t && oi < ti)) { t = ov; ti = oi; }
            }
            if (lane == 0) { s_pick_val = t; s_pick_idx = ti; }
        }
        __syncthreads();

        int   pick_idx = s_pick_idx;
        // gather the WINNER's unbiased prob (== work value before masking == my_prob at pick_idx)
        // s_pick_val is exactly that (work==my_prob for picked, not yet masked this round).
        float pick_prob = s_pick_val;

        if (tid == 0) {
            sel_idx[(size_t)row * n_used + j] = pick_idx;
            sel_w[(size_t)row * n_used + j]   = pick_prob;   // raw prob; renormalized below
            s_wsum += pick_prob;
        }
        // mask the winner for the next round
        if (tid == pick_idx) work = -FLT_MAX;
        __syncthreads();
    }

    // ---- 4. renorm: ws = max(sum, F16_MIN_NORMAL) BEFORE divide ----
    if (tid == 0) {
        float ws = fmaxf(s_wsum, 6.103515625e-5f);   // F16 smallest normal, clamp before divide
        for (int j = 0; j < n_used; ++j) {
            // gemma4 R3 fold: (w / ws) * ex_scale[sel] — the moe_w_exscale chain verbatim.
            sel_w[(size_t)row * n_used + j] = sel_w[(size_t)row * n_used + j] / ws
                * ex_scale[sel_idx[(size_t)row * n_used + j]];
        }
    }
}
