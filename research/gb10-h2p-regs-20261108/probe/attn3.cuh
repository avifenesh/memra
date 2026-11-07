// attn3.cuh: two register levers on the GB10 h2p prime attention, as probe candidates.
//
// Included after flash_attn.cu (uses its tiles, mma_bf16, load_q_frags, row_max4/row_sum4 and the
// h2p_* mbarrier / ldmatrix helpers). Every candidate keeps h2p's per-row program: the same
// tile sequence, the same 16-deep QK chains, the same softmax expressions, the same bf16 P
// rounding, the same rescale-and-fold of a fresh two-MMA PV chain (fma(O, alpha, C)) and the same
// per-lane row-sum order, so O must equal h2p bit for bit.
//
// Ring "K3V2": three K planes and two V planes (80 KB) with per-plane full/empty mbarriers. Tile t's
// K sits in K slot t % 3, its V in V slot t % 2. A warp issues K(t + 2) right after its QK(t) (the
// slot held K(t - 1), released by every warp's QK(t - 1)) and V(t + 1) right after its softmax(t)
// (the slot held V(t - 1), released by every warp's PV(t - 1)). That frees 16 KB against h2p's
// three K+V stages for Q fragments or the pair exchange.
//
// (a) qa<QS, WIDE>: h2p on the K3V2 ring with Q k-tiles QS..15 kept in shared memory as per-lane
//     fragments (one LDS.128 per k-tile), freeing 4 * (16 - QS) registers. WIDE runs the two
//     16-key halves of the QK in one k loop (four live accumulators) instead of one after the
//     other; each accumulator still sees k-tiles 0..15 in order.
// (b) pw<SKEW, QS>: a warp-pair split in a 12-warp CTA (2 query heads x 3 pairs x 16 rows, BQ 48).
//     Warp 2p computes S for keys 0..15 of the tile and PV for O columns 0..127, warp 2p + 1 keys
//     16..31 and columns 128..255. They exchange the row max (order-free), P (both 16-key halves,
//     bf16, as A fragments) and the first warp's per-lane partial row sum, which the second warp
//     continues in h2p's order; the second warp keeps l and hands it over at the end. About half of
//     h2p's per-warp registers, 3 warps per scheduler instead of 2.
#pragma once

// Non-blocking phase test: true once the barrier's phase with this parity has completed.
static __device__ __forceinline__ bool a3_mbar_test(uint64_t* bar, uint32_t parity) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar), ok;
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(a), "r"(parity) : "memory");
    return ok != 0;
}
static __device__ __forceinline__ void a3_ld_frag(ATile& t, const uint4* p) {
    const uint4 v = *p;
    int* xi = (int*)t.x;
    xi[0] = (int)v.x; xi[1] = (int)v.y; xi[2] = (int)v.z; xi[3] = (int)v.w;
}
static __device__ __forceinline__ void a3_st_frag(uint4* p, const ATile& t) {
    const int* xi = (const int*)t.x;
    *p = make_uint4((unsigned)xi[0], (unsigned)xi[1], (unsigned)xi[2], (unsigned)xi[3]);
}

// ----------------------------------------------------------------------------------------------
// (a) Q k-tiles in shared memory on the K3V2 ring
// ----------------------------------------------------------------------------------------------
template<int QS, bool WIDE>
struct QaLayout {
    static constexpr uint32_t PLANE_B = BK * 256 * 2;                      // 16 KB
    static constexpr uint32_t V0 = 3 * PLANE_B;                            // V planes after 3 K
    static constexpr uint32_t QOFF = 5 * PLANE_B;                          // Q fragments
    static constexpr uint32_t NQS = 16 - QS;
    static constexpr uint32_t BAROFF = QOFF + 8 * NQS * 512;
    static constexpr uint32_t SMEM = BAROFF + 10 * 8;
};

// VM: when V(t + 1) is issued. 0 after softmax(t) (blocking), 1 after PV(t), 2 after softmax(t) if
// its slot is already free, else after PV(t).
// PROD 1: K(t + 2) after QK(t) and V(t + 1) per VM; PROD 0: both at the end of tile t, V first (as
// h2p, where the producer's temporaries meet the lowest register pressure).
template<int QS, bool WIDE, int VM = 0, int PROD = 1>
static __device__ __forceinline__ void qa_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    using L = QaLayout<QS, WIDE>;
    constexpr int HD = 256, HEAD_DIM = HD, HPC = 2, BQ = BLOCK_Q;
    constexpr int HD_KTILES = HD / K_STEP, O_NBLK = HD / N_KEYS;
    constexpr int NT = HPC * N_WARPS * WARP_SZ;            // 256 threads
    constexpr uint32_t PLANE_B = L::PLANE_B;
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int wq = warp % N_WARPS;
    const int head = (int)blockIdx.y * HPC + warp / N_WARPS;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base = blockIdx.x * BQ;
    const int qrow_base = q_base + wq * M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* const s0 = (__nv_bfloat16*)smem_raw;
    uint64_t* const fullK = (uint64_t*)(smem_raw + L::BAROFF);
    uint64_t* const emptyK = fullK + 3;
    uint64_t* const fullV = fullK + 6;
    uint64_t* const emptyV = fullK + 8;
    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int b = 0; b < 3; ++b) { h2p_mbar_init(&fullK[b], NT); h2p_mbar_init(&emptyK[b], 2 * N_WARPS); }
        #pragma unroll
        for (int b = 0; b < 2; ++b) { h2p_mbar_init(&fullV[b], NT); h2p_mbar_init(&emptyV[b], 2 * N_WARPS); }
    }

    const int causal_i = causal;
    const int q_pos0w = (T_kv - T) + qrow_base;
    ATile Qf[HD_KTILES];
    // Q staging uses the first 64 KB (K planes 0-2, V plane 0) before any copy is issued.
    load_q_frags<HD, true>(Qf, Q, s0 + warp * M_ROWS * HEAD_DIM, qrow_base, nqw, head, n_head, head_dim, lane);
    uint4* const qsm = (uint4*)(smem_raw + L::QOFF) + warp * L::NQS * 32 + lane;
    #pragma unroll
    for (int kt = QS; kt < HD_KTILES; ++kt) a3_st_frag(qsm + (kt - QS) * 32, Qf[kt]);
    __syncthreads();   // staging reads done; the barrier inits are visible

    CTile O_acc[O_NBLK];
    #pragma unroll
    for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
    float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
    const int r_lo = lane / 4, r_hi = r_lo + 8, c0 = (lane % 4) * 2;
    const size_t kv_off = (size_t)kv_head * head_dim;
    const int q_pos_max = (T_kv - T) + q_base + (BQ - 1);
    int nt = (T_kv + BK - 1) / BK;
    if (causal_i) { int ntc = q_pos_max / BK + 1; nt = min(nt, ntc); }

    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const uint32_t pdst = sbase + (uint32_t)(warp * HEAD_DIM * 2 + ((lane ^ warp) << 4));
    const __nv_bfloat16* const pk = Kw + (size_t)warp * kv_dim_k + kv_off + lane * 8;
    const __nv_bfloat16* const pv = Vw + (size_t)warp * kv_dim_v + kv_off + lane * 8;
    auto produceK = [&](int tn) {
        const int s = tn % 3;
        if (tn >= 3) h2p_mbar_wait(&emptyK[s], (uint32_t)((tn / 3 - 1) & 1));   // K(tn-3) read
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool ok = warp + 8 * j < nk1;
            const uint32_t d = pdst + (uint32_t)s * PLANE_B + (uint32_t)(j * 8 * HEAD_DIM * 2);
            h2p_cp_async_16(d, ok ? (const void*)(pk + (size_t)(k1 + 8 * j) * kv_dim_k) : (const void*)Kw, ok ? 16u : 0u);
        }
        h2p_mbar_cp_arrive(&fullK[s]);
    };
    auto produceV = [&](int tn) {
        const int s = tn & 1;
        if (tn >= 2) h2p_mbar_wait(&emptyV[s], (uint32_t)((tn / 2 - 1) & 1));   // V(tn-2) read
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool ok = warp + 8 * j < nk1;
            const uint32_t d = pdst + L::V0 + (uint32_t)s * PLANE_B + (uint32_t)(j * 8 * HEAD_DIM * 2);
            h2p_cp_async_16(d, ok ? (const void*)(pv + (size_t)(k1 + 8 * j) * kv_dim_v) : (const void*)Vw, ok ? 16u : 0u);
        }
        h2p_mbar_cp_arrive(&fullV[s]);
    };

    uint32_t kofs[4], vofs[4];
    {
        const int x = lane & 7;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            kofs[m] = sbase + (uint32_t)(((lane & 7) + 8 * (lane >> 4)) * (HEAD_DIM * 2)
                                         + 16 * ((2 * m + ((lane >> 3) & 1)) ^ x));
            vofs[m] = sbase + (uint32_t)((lane & 15) * (HEAD_DIM * 2) + 16 * ((2 * m + (lane >> 4)) ^ x));
        }
    }

    if (nt > 0) { produceK(0); produceV(0); }
    if (nt > 1) produceK(1);
    if (nt > 0 && warp >= N_WARPS) h2p_bar_sync(1, NT);   // head B starts after head A's first QK

    for (int ti = 0; ti < nt; ++ti) {
        const int k0 = ti * BK;
        const int nk = min(BK, T_kv - k0);
        const uint32_t kpl = (uint32_t)(ti % 3) * PLANE_B, vpl = L::V0 + (uint32_t)(ti & 1) * PLANE_B;
        h2p_mbar_wait(&fullK[ti % 3], (uint32_t)((ti / 3) & 1));

        CTile Sc[BK / N_KEYS];
        if constexpr (WIDE) {
            CTile C0, C1, C2, C3;
            C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f; C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
            C2.x[0]=C2.x[1]=C2.x[2]=C2.x[3]=0.0f; C3.x[0]=C3.x[1]=C3.x[2]=C3.x[3]=0.0f;
            #pragma unroll
            for (int kt = 0; kt < HD_KTILES; ++kt) {
                ATile Qt;
                if (kt < QS) Qt = Qf[kt]; else a3_ld_frag(Qt, qsm + (kt - QS) * 32);
                ATile Ka, Kb;
                h2p_ldsm_k(Ka, kofs[kt & 3] + kpl + (uint32_t)(128 * (kt >> 2)));
                h2p_ldsm_k(Kb, kofs[kt & 3] + kpl + (uint32_t)(16 * HEAD_DIM * 2 + 128 * (kt >> 2)));
                BTile B0; B0.x[0]=Ka.x[0]; B0.x[1]=Ka.x[2];
                BTile B1; B1.x[0]=Ka.x[1]; B1.x[1]=Ka.x[3];
                BTile B2; B2.x[0]=Kb.x[0]; B2.x[1]=Kb.x[2];
                BTile B3; B3.x[0]=Kb.x[1]; B3.x[1]=Kb.x[3];
                mma_bf16(C0, Qt, B0); mma_bf16(C1, Qt, B1);
                mma_bf16(C2, Qt, B2); mma_bf16(C3, Qt, B3);
            }
            Sc[0] = C0; Sc[1] = C1; Sc[2] = C2; Sc[3] = C3;
        } else {
            for (int kg = 0; kg < BK; kg += 2 * N_KEYS) {
                CTile C0, C1;
                C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Qt;
                    if (kt < QS) Qt = Qf[kt]; else a3_ld_frag(Qt, qsm + (kt - QS) * 32);
                    ATile Kt;
                    h2p_ldsm_k(Kt, kofs[kt & 3] + kpl + (uint32_t)(kg * HEAD_DIM * 2 + 128 * (kt >> 2)));
                    BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                    BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                    mma_bf16(C0, Qt, Blo);
                    mma_bf16(C1, Qt, Bhi);
                }
                Sc[kg / N_KEYS + 0] = C0;
                Sc[kg / N_KEYS + 1] = C1;
            }
        }
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyK[ti % 3]);   // this warp is done with K(ti)
        if (ti == 0 && warp < N_WARPS) h2p_bar_arrive(1, NT);
        if (PROD == 1 && ti + 2 < nt) produceK(ti + 2);

        float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
        #pragma unroll
        for (int g = 0; g < BK / N_KEYS; ++g) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int col = g * N_KEYS + c0 + (l & 1);
                int row = (l < 2) ? r_lo : r_hi;
                int q_pos = q_pos0w + row;
                float s = Sc[g].x[l] * scale;
                if (col >= nk) s = NEG_INF;
                if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                Sc[g].x[l] = s;
                if (l < 2) s_tile_max_lo = fmaxf(s_tile_max_lo, s);
                else       s_tile_max_hi = fmaxf(s_tile_max_hi, s);
            }
        }
        s_tile_max_lo = row_max4(s_tile_max_lo);
        s_tile_max_hi = row_max4(s_tile_max_hi);
        float m_prev_lo = m_lo, m_prev_hi = m_hi;
        float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
        float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
        float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
        float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);
        float l_part_lo = 0.0f, l_part_hi = 0.0f;
        #pragma unroll
        for (int g = 0; g < BK / N_KEYS; ++g) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                float mn = (l < 2) ? m_new_lo : m_new_hi;
                float s  = Sc[g].x[l];
                float p  = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
                Sc[g].x[l] = p;
                if (l < 2) l_part_lo += p; else l_part_hi += p;
            }
        }
        l_part_lo = row_sum4(l_part_lo);
        l_part_hi = row_sum4(l_part_hi);
        l_lo = __fmaf_rn(l_lo, alpha_lo, l_part_lo);
        l_hi = __fmaf_rn(l_hi, alpha_hi, l_part_hi);
        m_lo = m_new_lo; m_hi = m_new_hi;

        ATile Pf[BK / K_STEP];
        #pragma unroll
        for (int kk = 0; kk < BK; kk += K_STEP) {
            const int g = kk / N_KEYS;
            Pf[kk / K_STEP].x[0] = __floats2bfloat162_rn(Sc[g].x[0], Sc[g].x[1]);
            Pf[kk / K_STEP].x[1] = __floats2bfloat162_rn(Sc[g].x[2], Sc[g].x[3]);
            Pf[kk / K_STEP].x[2] = __floats2bfloat162_rn(Sc[g + 1].x[0], Sc[g + 1].x[1]);
            Pf[kk / K_STEP].x[3] = __floats2bfloat162_rn(Sc[g + 1].x[2], Sc[g + 1].x[3]);
        }
        bool vdone = !(ti + 1 < nt);
        if (PROD == 1 && !vdone && (VM == 0 || (VM == 2 && (ti + 1 < 2 || a3_mbar_test(&emptyV[(ti + 1) & 1], (uint32_t)(((ti + 1) / 2 - 1) & 1)))))) {
            produceV(ti + 1); vdone = true;
        }
        h2p_mbar_wait(&fullV[ti & 1], (uint32_t)((ti / 2) & 1));
        #pragma unroll
        for (int d0 = 0; d0 < HEAD_DIM; d0 += 2 * N_KEYS) {
            CTile Clo, Chi;
            Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
            Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < BK; kk += K_STEP) {
                ATile Bt;
                h2p_ldsm_vt(Bt, vofs[(d0 / 16) & 3] + vpl + (uint32_t)(kk * HEAD_DIM * 2 + 128 * ((d0 / 16) >> 2)));
                BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                mma_bf16(Clo, Pf[kk / K_STEP], Blo);
                mma_bf16(Chi, Pf[kk / K_STEP], Bhi);
            }
            CTile& Olo = O_acc[(d0 / N_KEYS) + 0];
            CTile& Ohi = O_acc[(d0 / N_KEYS) + 1];
            Olo.x[0] = __fmaf_rn(Olo.x[0], alpha_lo, Clo.x[0]); Olo.x[1] = __fmaf_rn(Olo.x[1], alpha_lo, Clo.x[1]);
            Olo.x[2] = __fmaf_rn(Olo.x[2], alpha_hi, Clo.x[2]); Olo.x[3] = __fmaf_rn(Olo.x[3], alpha_hi, Clo.x[3]);
            Ohi.x[0] = __fmaf_rn(Ohi.x[0], alpha_lo, Chi.x[0]); Ohi.x[1] = __fmaf_rn(Ohi.x[1], alpha_lo, Chi.x[1]);
            Ohi.x[2] = __fmaf_rn(Ohi.x[2], alpha_hi, Chi.x[2]); Ohi.x[3] = __fmaf_rn(Ohi.x[3], alpha_hi, Chi.x[3]);
        }
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyV[ti & 1]);   // this warp is done with V(ti)
        if (!vdone) produceV(ti + 1);
        if (PROD == 0 && ti + 2 < nt) produceK(ti + 2);
    }

    #pragma unroll
    for (int c = 0; c < O_NBLK; ++c) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int r = CTile::get_i(l);
            int d = c * N_KEYS + CTile::get_j(l);
            if (r < nqw) {
                const float lv = (l < 2) ? l_lo : l_hi;
                float linv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
            }
        }
    }
}

#define QA_KERNEL(NAME, QS, WIDE, ...)                                                                    \
extern "C" __global__ void __launch_bounds__(2 * N_WARPS * WARP_SZ, 1) NAME(                         \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    qa_body<QS, WIDE, ##__VA_ARGS__>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal,             \
                      kv_dim_k, kv_dim_v);                                                           \
}
QA_KERNEL(cand_r32, 16, false)   // control: h2p on the K3V2 ring, all of Q in registers
QA_KERNEL(cand_qa12, 12, false)  // (a): Q k-tiles 12..15 in shared memory
QA_KERNEL(cand_qa12w, 12, true)  // (a) + both QK halves in one k loop
QA_KERNEL(cand_r32w, 16, true)   // ring + wide QK, all of Q in registers
QA_KERNEL(cand_qa12v1, 12, false, 1)   // (a), V(t+1) after PV(t)
QA_KERNEL(cand_qa12v2, 12, false, 2)   // (a), V(t+1) after softmax(t) when free
QA_KERNEL(cand_r32e, 16, false, 0, 0)  // ring control, both refills at the end of the tile
QA_KERNEL(cand_qa12e, 12, false, 0, 0) // (a), both refills at the end of the tile
QA_KERNEL(cand_qa13e, 13, false, 0, 0) // Q k-tiles 13..15 in shared memory
QA_KERNEL(cand_qa14e, 14, false, 0, 0) // Q k-tiles 14..15 in shared memory

// ----------------------------------------------------------------------------------------------
// (b) warp-pair split, 12 warps
// ----------------------------------------------------------------------------------------------
template<int QS>
struct PwLayout {
    static constexpr uint32_t PLANE_B = BK * 256 * 2;                      // 16 KB
    static constexpr uint32_t V0 = 3 * PLANE_B;
    static constexpr uint32_t XOFF = 5 * PLANE_B;                          // pair exchange
    static constexpr uint32_t XPAIR = 512 + 256 + 1024;                    // max, sum, P
    static constexpr uint32_t QOFF = XOFF + 6 * XPAIR;
    static constexpr uint32_t NQS = 16 - QS;
    static constexpr uint32_t BAROFF = QOFF + 12 * NQS * 512;
    static constexpr uint32_t SMEM = BAROFF + 10 * 8;
};

// SKEW: 0 none; 1 warps 4-7 start after warps 0-3 finish QK(0) and warps 8-11 after warps 4-7 do;
// 2 warps 4-7 after warps 0-3's QK(0), warps 8-11 after warps 0-3's softmax(0).
template<int SKEW, int QS, int VM = 0, int PROD = 1>
static __device__ __forceinline__ void pw_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    using L = PwLayout<QS>;
    constexpr int HD = 256, HEAD_DIM = HD, HPC = 2, PPH = 3, BQ = PPH * M_ROWS;   // 48 rows per head
    constexpr int NW = 2 * HPC * PPH;                       // 12 warps
    constexpr int NT = NW * WARP_SZ;                        // 384 threads
    constexpr int HD_KTILES = HD / K_STEP, O_HALF = HD / N_KEYS / 2;   // 16 O CTiles per warp
    constexpr uint32_t PLANE_B = L::PLANE_B;
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int pr = warp >> 1, role = warp & 1;              // pair, 0 = keys 0-15 / cols 0-127
    const int grp = warp >> 2;                              // skew group (one warp per scheduler)
    const int head = (int)blockIdx.y * HPC + pr / PPH;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base = blockIdx.x * BQ;
    const int qrow_base = q_base + (pr % PPH) * M_ROWS;
    if (blockIdx.y * HPC >= n_head || q_base >= T) return;  // CTA-uniform
    const int nqw = min(M_ROWS, T - qrow_base);            // <= 0: this pair only helps

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* const s0 = (__nv_bfloat16*)smem_raw;
    uint64_t* const fullK = (uint64_t*)(smem_raw + L::BAROFF);
    uint64_t* const emptyK = fullK + 3;
    uint64_t* const fullV = fullK + 6;
    uint64_t* const emptyV = fullK + 8;
    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int b = 0; b < 3; ++b) { h2p_mbar_init(&fullK[b], NT); h2p_mbar_init(&emptyK[b], NW); }
        #pragma unroll
        for (int b = 0; b < 2; ++b) { h2p_mbar_init(&fullV[b], NT); h2p_mbar_init(&emptyV[b], NW); }
    }
    char* const xp = smem_raw + L::XOFF + pr * L::XPAIR;
    float* const xmax_me = (float*)xp + role * 64;          // [2][32] per role
    const float* const xmax_ot = (const float*)xp + (role ^ 1) * 64;
    float* const xsum = (float*)(xp + 512);                 // [2][32], role 0 writes
    uint4* const xP_me = (uint4*)(xp + 768) + role * 32 + lane;
    const uint4* const xP_ot = (const uint4*)(xp + 768) + (role ^ 1) * 32 + lane;
    const int pbar = 4 + pr;

    const int causal_i = causal;
    const int q_pos0w = (T_kv - T) + qrow_base;
    ATile Qf[HD_KTILES];
    // staging: 12 warps x 8 KB = 96 KB would overrun the 80 KB ring; stage in two rounds
    if (warp < 8) load_q_frags<HD, true>(Qf, Q, s0 + warp * M_ROWS * HEAD_DIM, qrow_base, nqw, head, n_head, head_dim, lane);
    __syncthreads();
    if (warp >= 8) load_q_frags<HD, true>(Qf, Q, s0 + (warp - 8) * M_ROWS * HEAD_DIM, qrow_base, nqw, head, n_head, head_dim, lane);
    uint4* const qsm = (uint4*)(smem_raw + L::QOFF) + warp * L::NQS * 32 + lane;
    #pragma unroll
    for (int kt = QS; kt < HD_KTILES; ++kt) a3_st_frag(qsm + (kt - QS) * 32, Qf[kt]);
    __syncthreads();

    CTile O_acc[O_HALF];
    #pragma unroll
    for (int c = 0; c < O_HALF; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
    float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
    const int r_lo = lane / 4, r_hi = r_lo + 8, c0 = (lane % 4) * 2;
    const size_t kv_off = (size_t)kv_head * head_dim;
    const int q_pos_max = (T_kv - T) + q_base + (BQ - 1);
    int nt = (T_kv + BK - 1) / BK;
    if (causal_i) { int ntc = q_pos_max / BK + 1; nt = min(nt, ntc); }

    // producer: thread (warp, lane) copies rows warp + 12j (j < 3, row < 32), chunk lane
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const int nrow = warp < 8 ? 3 : 2;
    auto produce = [&](int tn, uint32_t plane, const __nv_bfloat16* src, int kv_dim, uint64_t* bar) {
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (j < nrow) {
                const int row = warp + 12 * j;
                const bool ok = row < nk1;
                const uint32_t d = sbase + plane + (uint32_t)(row * HEAD_DIM * 2 + ((lane ^ (row & 7)) << 4));
                h2p_cp_async_16(d, ok ? (const void*)(src + (size_t)(k1 + row) * kv_dim + kv_off + lane * 8) : (const void*)src,
                                ok ? 16u : 0u);
            }
        }
        h2p_mbar_cp_arrive(bar);
    };
    auto produceK = [&](int tn) {
        const int s = tn % 3;
        if (tn >= 3) h2p_mbar_wait(&emptyK[s], (uint32_t)((tn / 3 - 1) & 1));
        produce(tn, (uint32_t)s * PLANE_B, Kw, kv_dim_k, &fullK[s]);
    };
    auto produceV = [&](int tn) {
        const int s = tn & 1;
        if (tn >= 2) h2p_mbar_wait(&emptyV[s], (uint32_t)((tn / 2 - 1) & 1));
        produce(tn, L::V0 + (uint32_t)s * PLANE_B, Vw, kv_dim_v, &fullV[s]);
    };

    uint32_t kofs[4], vofs[4];
    {
        const int x = lane & 7;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            kofs[m] = sbase + (uint32_t)((role * 16 + (lane & 7) + 8 * (lane >> 4)) * (HEAD_DIM * 2)
                                         + 16 * ((2 * m + ((lane >> 3) & 1)) ^ x));
            vofs[m] = sbase + (uint32_t)((lane & 15) * (HEAD_DIM * 2) + 16 * ((2 * m + (lane >> 4)) ^ x));
        }
    }
    const uint32_t vcol = (uint32_t)(256 * role);           // 16-dim block 8*role: +128*(8 role >> 2)

    if (nt > 0) { produceK(0); produceV(0); }
    if (nt > 1) produceK(1);
    if constexpr (SKEW >= 1) {
        if (nt > 0 && grp >= 1) h2p_bar_sync(grp, 2 * 4 * WARP_SZ);
    }

    for (int ti = 0; ti < nt; ++ti) {
        const int k0 = ti * BK;
        const int nk = min(BK, T_kv - k0);
        const uint32_t kpl = (uint32_t)(ti % 3) * PLANE_B, vpl = L::V0 + (uint32_t)(ti & 1) * PLANE_B;
        h2p_mbar_wait(&fullK[ti % 3], (uint32_t)((ti / 3) & 1));

        // ---- QK for this warp's 16 keys: h2p's kg = 16 * role chains ----
        CTile Sc[2];
        {
            CTile C0, C1;
            C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
            C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
            #pragma unroll
            for (int kt = 0; kt < HD_KTILES; ++kt) {
                ATile Qt;
                if (kt < QS) Qt = Qf[kt]; else a3_ld_frag(Qt, qsm + (kt - QS) * 32);
                ATile Kt;
                h2p_ldsm_k(Kt, kofs[kt & 3] + kpl + (uint32_t)(128 * (kt >> 2)));
                BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                mma_bf16(C0, Qt, Blo);
                mma_bf16(C1, Qt, Bhi);
            }
            Sc[0] = C0; Sc[1] = C1;
        }
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyK[ti % 3]);
        if constexpr (SKEW == 1) { if (ti == 0 && grp < 2) h2p_bar_arrive(grp + 1, 2 * 4 * WARP_SZ); }
        if constexpr (SKEW == 2) { if (ti == 0 && grp == 0) h2p_bar_arrive(1, 2 * 4 * WARP_SZ); }
        if (PROD == 1 && ti + 2 < nt) produceK(ti + 2);

        // ---- scale, mask, this half's row max ----
        float mx_lo = NEG_INF, mx_hi = NEG_INF;
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int col = (2 * role + g) * N_KEYS + c0 + (l & 1);
                int row = (l < 2) ? r_lo : r_hi;
                int q_pos = q_pos0w + row;
                float s = Sc[g].x[l] * scale;
                if (col >= nk) s = NEG_INF;
                if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                Sc[g].x[l] = s;
                if (l < 2) mx_lo = fmaxf(mx_lo, s);
                else       mx_hi = fmaxf(mx_hi, s);
            }
        }
        mx_lo = row_max4(mx_lo);
        mx_hi = row_max4(mx_hi);
        xmax_me[lane] = mx_lo; xmax_me[32 + lane] = mx_hi;
        h2p_bar_sync(pbar, 2 * WARP_SZ);
        const float s_tile_max_lo = fmaxf(mx_lo, xmax_ot[lane]);   // max is order-free
        const float s_tile_max_hi = fmaxf(mx_hi, xmax_ot[32 + lane]);
        float m_prev_lo = m_lo, m_prev_hi = m_hi;
        float m_new_lo = fmaxf(m_prev_lo, s_tile_max_lo);
        float m_new_hi = fmaxf(m_prev_hi, s_tile_max_hi);
        float alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
        float alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                float mn = (l < 2) ? m_new_lo : m_new_hi;
                float s  = Sc[g].x[l];
                Sc[g].x[l] = (s == NEG_INF) ? 0.0f : exp2f((s - mn) * LOG2E);
            }
        }
        if (role == 0) {
            // keys 0..15 open h2p's per-lane sum (g = 0, 1)
            float pl = 0.0f, ph = 0.0f;
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) { if (l < 2) pl += Sc[g].x[l]; else ph += Sc[g].x[l]; }
            }
            xsum[lane] = pl; xsum[32 + lane] = ph;
        }
        ATile Pm;
        Pm.x[0] = __floats2bfloat162_rn(Sc[0].x[0], Sc[0].x[1]);
        Pm.x[1] = __floats2bfloat162_rn(Sc[0].x[2], Sc[0].x[3]);
        Pm.x[2] = __floats2bfloat162_rn(Sc[1].x[0], Sc[1].x[1]);
        Pm.x[3] = __floats2bfloat162_rn(Sc[1].x[2], Sc[1].x[3]);
        a3_st_frag(xP_me, Pm);
        h2p_bar_sync(pbar, 2 * WARP_SZ);
        if (role == 1) {
            // keys 16..31 continue it (g = 2, 3), then the 4-lane reduce
            float pl = xsum[lane], ph = xsum[32 + lane];
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) { if (l < 2) pl += Sc[g].x[l]; else ph += Sc[g].x[l]; }
            }
            pl = row_sum4(pl);
            ph = row_sum4(ph);
            l_lo = __fmaf_rn(l_lo, alpha_lo, pl);
            l_hi = __fmaf_rn(l_hi, alpha_hi, ph);
        }
        m_lo = m_new_lo; m_hi = m_new_hi;
        ATile Po; a3_ld_frag(Po, xP_ot);
        if constexpr (SKEW == 2) { if (ti == 0 && grp == 0) h2p_bar_arrive(2, 2 * 4 * WARP_SZ); }
        bool vdone = !(ti + 1 < nt);
        if (PROD == 1 && !vdone && (VM == 0 || (VM == 2 && (ti + 1 < 2 || a3_mbar_test(&emptyV[(ti + 1) & 1], (uint32_t)(((ti + 1) / 2 - 1) & 1)))))) {
            produceV(ti + 1); vdone = true;
        }
        h2p_mbar_wait(&fullV[ti & 1], (uint32_t)((ti / 2) & 1));

        // ---- PV for this warp's 128 O columns ----
        const ATile& P0 = role ? Po : Pm;
        const ATile& P1 = role ? Pm : Po;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            CTile Clo, Chi;
            Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
            Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                ATile Bt;
                h2p_ldsm_vt(Bt, vofs[i & 3] + vpl + vcol + (uint32_t)(kk * K_STEP * HEAD_DIM * 2 + 128 * (i >> 2)));
                BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                mma_bf16(Clo, kk ? P1 : P0, Blo);
                mma_bf16(Chi, kk ? P1 : P0, Bhi);
            }
            CTile& Olo = O_acc[2 * i + 0];
            CTile& Ohi = O_acc[2 * i + 1];
            Olo.x[0] = __fmaf_rn(Olo.x[0], alpha_lo, Clo.x[0]); Olo.x[1] = __fmaf_rn(Olo.x[1], alpha_lo, Clo.x[1]);
            Olo.x[2] = __fmaf_rn(Olo.x[2], alpha_hi, Clo.x[2]); Olo.x[3] = __fmaf_rn(Olo.x[3], alpha_hi, Clo.x[3]);
            Ohi.x[0] = __fmaf_rn(Ohi.x[0], alpha_lo, Chi.x[0]); Ohi.x[1] = __fmaf_rn(Ohi.x[1], alpha_lo, Chi.x[1]);
            Ohi.x[2] = __fmaf_rn(Ohi.x[2], alpha_hi, Chi.x[2]); Ohi.x[3] = __fmaf_rn(Ohi.x[3], alpha_hi, Chi.x[3]);
        }
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyV[ti & 1]);
        if (!vdone) produceV(ti + 1);
        if (PROD == 0 && ti + 2 < nt) produceK(ti + 2);
    }

    // the second warp of the pair holds l; hand it to the first
    if (role == 1) { xsum[lane] = l_lo; xsum[32 + lane] = l_hi; }
    h2p_bar_sync(pbar, 2 * WARP_SZ);
    if (role == 0) { l_lo = xsum[lane]; l_hi = xsum[32 + lane]; }
    #pragma unroll
    for (int c = 0; c < O_HALF; ++c) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int r = CTile::get_i(l);
            int d = 128 * role + c * N_KEYS + CTile::get_j(l);
            if (r < nqw) {
                const float lv = (l < 2) ? l_lo : l_hi;
                float linv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
            }
        }
    }
}

#define PW_KERNEL(NAME, SKEW, QS, ...)                                                                    \
extern "C" __global__ void __launch_bounds__(12 * WARP_SZ, 1) NAME(                                  \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    pw_body<SKEW, QS, ##__VA_ARGS__>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal,             \
                      kv_dim_k, kv_dim_v);                                                           \
}
PW_KERNEL(cand_pw0, 0, 16)   // (b), no skew
PW_KERNEL(cand_pw1, 1, 16)   // (b), groups one QK apart
PW_KERNEL(cand_pw2, 2, 16)   // (b), group 2 after group 0's softmax
PW_KERNEL(cand_pw1q15, 1, 15)   // (b) + Q k-tile 15 in shared memory
PW_KERNEL(cand_pw0v2, 0, 16, 2)
PW_KERNEL(cand_pw1v2, 1, 16, 2)
PW_KERNEL(cand_pw2v2, 2, 16, 2)
PW_KERNEL(cand_pw1v1, 1, 16, 1)
PW_KERNEL(cand_pw0e, 0, 16, 0, 0)
PW_KERNEL(cand_pw1e, 1, 16, 0, 0)
PW_KERNEL(cand_pw2e, 2, 16, 0, 0)
PW_KERNEL(cand_pw1q15e, 1, 15, 0, 0)
