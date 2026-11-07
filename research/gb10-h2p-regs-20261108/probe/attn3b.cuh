// attn3b.cuh: both register levers on the "K2V3" ring (included after attn3.cuh).
//
// K2V3: two K planes and three V planes (80 KB). Tile t's K sits in K slot t % 2, its V in V slot
// t % 3. One full mbarrier per tile covers both planes: a thread issues V(t) at the end of tile t - 2
// and K(t) at the top of tile t - 1, and its `cp.async.mbarrier.arrive.noinc` after K(t) fires once
// all of its earlier copies landed, V(t) included. So a warp waits once, at the top of a tile, as in
// h2p; it issues K(t + 1) right after that wait (the slot held K(t - 1), released by every warp's
// QK(t - 1)) and V(t + 2) after its PV(t) (the slot held V(t - 1), released by every warp's PV(t - 1)).
// Both copy points sit where h2p's producer sits, at the lowest register pressure, and each copy has
// about a tile to land. The 16 KB this frees against h2p's 96 KB ring holds Q fragments (a) or the
// pair exchange (b).
#pragma once

struct K2V3 {
    static constexpr uint32_t PLANE_B = BK * 256 * 2;      // 16 KB
    static constexpr uint32_t V0 = 2 * PLANE_B;            // V planes after the 2 K planes
    static constexpr uint32_t END = 5 * PLANE_B;           // 80 KB
    static constexpr uint32_t NBAR = 7;                    // full[2], emptyK[2], emptyV[3]
};

// ----------------------------------------------------------------------------------------------
// (a) qb<QS>: h2p with Q k-tiles QS..15 as per-lane fragments in shared memory
// ----------------------------------------------------------------------------------------------
template<int QS>
struct QbLayout {
    static constexpr uint32_t NQS = 16 - QS;
    static constexpr uint32_t QOFF = K2V3::END;
    static constexpr uint32_t BAROFF = QOFF + 8 * NQS * 512;
    static constexpr uint32_t SMEM = BAROFF + K2V3::NBAR * 8;
};

template<int QS>
static __device__ __forceinline__ void qb_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    using L = QbLayout<QS>;
    constexpr int HD = 256, HEAD_DIM = HD, HPC = 2, BQ = BLOCK_Q;
    constexpr int HD_KTILES = HD / K_STEP, O_NBLK = HD / N_KEYS;
    constexpr int NT = HPC * N_WARPS * WARP_SZ;            // 256 threads
    constexpr uint32_t PLANE_B = K2V3::PLANE_B;
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
    uint64_t* const full = (uint64_t*)(smem_raw + L::BAROFF);
    uint64_t* const emptyK = full + 2;
    uint64_t* const emptyV = full + 4;
    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) { h2p_mbar_init(&full[b], NT); h2p_mbar_init(&emptyK[b], 2 * N_WARPS); }
        #pragma unroll
        for (int b = 0; b < 3; ++b) h2p_mbar_init(&emptyV[b], 2 * N_WARPS);
    }

    const int causal_i = causal;
    const int q_pos0w = (T_kv - T) + qrow_base;
    ATile Qf[HD_KTILES];
    // Q staging uses the first 64 KB before any copy is issued.
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
    auto copyK = [&](int tn) {   // K(tn) into K slot tn % 2, then the tile's full arrival
        const int s = tn & 1;
        if (tn >= 2) h2p_mbar_wait(&emptyK[s], (uint32_t)(((tn >> 1) - 1) & 1));   // K(tn - 2) read
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool ok = warp + 8 * j < nk1;
            const uint32_t d = pdst + (uint32_t)s * PLANE_B + (uint32_t)(j * 8 * HEAD_DIM * 2);
            h2p_cp_async_16(d, ok ? (const void*)(pk + (size_t)(k1 + 8 * j) * kv_dim_k) : (const void*)Kw, ok ? 16u : 0u);
        }
        h2p_mbar_cp_arrive(&full[s]);
    };
    auto copyV = [&](int tn) {   // V(tn) into V slot tn % 3; tile tn's full arrival comes with K(tn)
        const int s = tn % 3;
        if (tn >= 3) h2p_mbar_wait(&emptyV[s], (uint32_t)((tn / 3 - 1) & 1));      // V(tn - 3) read
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool ok = warp + 8 * j < nk1;
            const uint32_t d = pdst + K2V3::V0 + (uint32_t)s * PLANE_B + (uint32_t)(j * 8 * HEAD_DIM * 2);
            h2p_cp_async_16(d, ok ? (const void*)(pv + (size_t)(k1 + 8 * j) * kv_dim_v) : (const void*)Vw, ok ? 16u : 0u);
        }
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

    if (nt > 0) copyV(0);
    if (nt > 1) copyV(1);
    if (nt > 0) copyK(0);
    if (nt > 0 && warp >= N_WARPS) h2p_bar_sync(1, NT);   // head B starts after head A's first QK

    for (int ti = 0; ti < nt; ++ti) {
        const int k0 = ti * BK;
        const int nk = min(BK, T_kv - k0);
        const uint32_t kpl = (uint32_t)(ti & 1) * PLANE_B, vpl = K2V3::V0 + (uint32_t)(ti % 3) * PLANE_B;
        h2p_mbar_wait(&full[ti & 1], (uint32_t)((ti >> 1) & 1));
        if (ti + 1 < nt) copyK(ti + 1);

        CTile Sc[BK / N_KEYS];
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
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyK[ti & 1]);   // this warp is done with K(ti)
        if (ti == 0 && warp < N_WARPS) h2p_bar_arrive(1, NT);

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
        l_lo = l_lo * alpha_lo + l_part_lo;
        l_hi = l_hi * alpha_hi + l_part_hi;
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
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
            O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
        }
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
            O_acc[(d0 / N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0 / N_KEYS) + 0].x[1] += Clo.x[1];
            O_acc[(d0 / N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0 / N_KEYS) + 0].x[3] += Clo.x[3];
            O_acc[(d0 / N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0 / N_KEYS) + 1].x[1] += Chi.x[1];
            O_acc[(d0 / N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0 / N_KEYS) + 1].x[3] += Chi.x[3];
        }
        __syncwarp();
        if (lane == 0) h2p_mbar_arrive(&emptyV[ti % 3]);   // this warp is done with V(ti)
        if (ti + 2 < nt) copyV(ti + 2);
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

#define QB_KERNEL(NAME, QS)                                                                          \
extern "C" __global__ void __launch_bounds__(2 * N_WARPS * WARP_SZ, 1) NAME(                         \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    qb_body<QS>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal, kv_dim_k, kv_dim_v);\
}
QB_KERNEL(cand_rb, 16)     // K2V3 ring control, all of Q in registers
QB_KERNEL(cand_qb12, 12)   // (a): Q k-tiles 12..15 in shared memory
QB_KERNEL(cand_qb13, 13)
QB_KERNEL(cand_qb14, 14)

// ----------------------------------------------------------------------------------------------
// (b) pb<SKEW, QS>: the warp-pair split on the K2V3 ring, 12 warps (2 heads x 3 pairs x 16 rows)
// ----------------------------------------------------------------------------------------------
template<int QS>
struct PbLayout {
    static constexpr uint32_t XOFF = K2V3::END;
    static constexpr uint32_t XPAIR = 512 + 256 + 1024;    // max [2 roles][2][32], sum [2][32], P [2][32]
    static constexpr uint32_t NQS = 16 - QS;
    static constexpr uint32_t QOFF = XOFF + 6 * XPAIR;
    static constexpr uint32_t BAROFF = QOFF + 12 * NQS * 512;
    static constexpr uint32_t SMEM = BAROFF + K2V3::NBAR * 8;
};

// SKEW: 0 none; 1 warps 4-7 start after warps 0-3 finish QK(0), warps 8-11 after warps 4-7 do;
// 2 warps 4-7 after warps 0-3's QK(0), warps 8-11 after warps 0-3's softmax(0).
template<int SKEW, int QS>
static __device__ __forceinline__ void pb_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    using L = PbLayout<QS>;
    constexpr int HD = 256, HEAD_DIM = HD, HPC = 2, PPH = 3, BQ = PPH * M_ROWS;   // 48 rows per head
    constexpr int NW = 2 * HPC * PPH;                       // 12 warps
    constexpr int NT = NW * WARP_SZ;                        // 384 threads
    constexpr int HD_KTILES = HD / K_STEP, O_HALF = HD / N_KEYS / 2;   // 16 O CTiles per warp
    constexpr uint32_t PLANE_B = K2V3::PLANE_B;
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int pr = warp >> 1, role = warp & 1;              // pair; role 0 = keys 0-15, columns 0-127
    const int grp = warp >> 2;                              // skew group: one warp per scheduler
    const int head = (int)blockIdx.y * HPC + pr / PPH;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base = blockIdx.x * BQ;
    const int qrow_base = q_base + (pr % PPH) * M_ROWS;
    if ((int)blockIdx.y * HPC >= n_head || q_base >= T) return;   // CTA-uniform
    const int nqw = min(M_ROWS, T - qrow_base);            // <= 0: this pair computes on zero rows

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* const s0 = (__nv_bfloat16*)smem_raw;
    uint64_t* const full = (uint64_t*)(smem_raw + L::BAROFF);
    uint64_t* const emptyK = full + 2;
    uint64_t* const emptyV = full + 4;
    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) { h2p_mbar_init(&full[b], NT); h2p_mbar_init(&emptyK[b], NW); }
        #pragma unroll
        for (int b = 0; b < 3; ++b) h2p_mbar_init(&emptyV[b], NW);
    }
    char* const xp = smem_raw + L::XOFF + pr * L::XPAIR;
    float* const xmax_me = (float*)xp + role * 64;
    const float* const xmax_ot = (const float*)xp + (role ^ 1) * 64;
    float* const xsum = (float*)(xp + 512);
    uint4* const xP_me = (uint4*)(xp + 768) + role * 32 + lane;
    const uint4* const xP_ot = (const uint4*)(xp + 768) + (role ^ 1) * 32 + lane;
    const int pbar = 4 + pr;

    const int causal_i = causal;
    const int q_pos0w = (T_kv - T) + qrow_base;
    ATile Qf[HD_KTILES];
    // 12 warps x 8 KB of staging would overrun the 80 KB ring: two rounds
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
    auto copy_rows = [&](int tn, uint32_t plane, const __nv_bfloat16* src, int kv_dim) {
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            if (j < nrow) {
                const int row = warp + 12 * j;
                const bool ok = row < nk1;
                const uint32_t d = sbase + plane + (uint32_t)(row * HEAD_DIM * 2 + ((lane ^ (row & 7)) << 4));
                h2p_cp_async_16(d, ok ? (const void*)(src + (size_t)(k1 + row) * kv_dim + kv_off + lane * 8)
                                      : (const void*)src, ok ? 16u : 0u);
            }
        }
    };
    auto copyK = [&](int tn) {
        const int s = tn & 1;
        if (tn >= 2) h2p_mbar_wait(&emptyK[s], (uint32_t)(((tn >> 1) - 1) & 1));
        copy_rows(tn, (uint32_t)s * PLANE_B, Kw, kv_dim_k);
        h2p_mbar_cp_arrive(&full[s]);
    };
    auto copyV = [&](int tn) {
        const int s = tn % 3;
        if (tn >= 3) h2p_mbar_wait(&emptyV[s], (uint32_t)((tn / 3 - 1) & 1));
        copy_rows(tn, K2V3::V0 + (uint32_t)s * PLANE_B, Vw, kv_dim_v);
    };

    uint32_t kofs[4], vofs[4];
    {
        const int x = lane & 7;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            kofs[m] = sbase + (uint32_t)((role * 16 + (lane & 7) + 8 * (lane >> 4)) * (HEAD_DIM * 2)
                                         + 16 * ((2 * m + ((lane >> 3) & 1)) ^ x));
            vofs[m] = sbase + (uint32_t)((lane & 15) * (HEAD_DIM * 2) + 16 * ((2 * m + (lane >> 4)) ^ x)
                                         + 256 * role);    // 16-dim blocks 8 role .. 8 role + 7
        }
    }

    if (nt > 0) copyV(0);
    if (nt > 1) copyV(1);
    if (nt > 0) copyK(0);
    if constexpr (SKEW >= 1) {
        if (nt > 0 && grp >= 1) h2p_bar_sync(grp, 2 * 4 * WARP_SZ);
    }

    for (int ti = 0; ti < nt; ++ti) {
        const int k0 = ti * BK;
        const int nk = min(BK, T_kv - k0);
        const uint32_t kpl = (uint32_t)(ti & 1) * PLANE_B, vpl = K2V3::V0 + (uint32_t)(ti % 3) * PLANE_B;
        h2p_mbar_wait(&full[ti & 1], (uint32_t)((ti >> 1) & 1));
        if (ti + 1 < nt) copyK(ti + 1);

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
        if (lane == 0) h2p_mbar_arrive(&emptyK[ti & 1]);
        if constexpr (SKEW == 1) { if (ti == 0 && grp < 2) h2p_bar_arrive(grp + 1, 2 * 4 * WARP_SZ); }
        if constexpr (SKEW == 2) { if (ti == 0 && grp == 0) h2p_bar_arrive(1, 2 * 4 * WARP_SZ); }

        // ---- scale, mask, this half's row max; the pair's max ----
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
                h2p_ldsm_vt(Bt, vofs[i & 3] + vpl + (uint32_t)(kk * K_STEP * HEAD_DIM * 2 + 128 * (i >> 2)));
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
        if (lane == 0) h2p_mbar_arrive(&emptyV[ti % 3]);
        if (ti + 2 < nt) copyV(ti + 2);
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

#define PB_KERNEL(NAME, SKEW, QS)                                                                    \
extern "C" __global__ void __launch_bounds__(12 * WARP_SZ, 1) NAME(                                  \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    pb_body<SKEW, QS>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal,             \
                      kv_dim_k, kv_dim_v);                                                           \
}
PB_KERNEL(cand_pb0, 0, 16)
PB_KERNEL(cand_pb1, 1, 16)
PB_KERNEL(cand_pb2, 2, 16)
PB_KERNEL(cand_pb1q15, 1, 15)
PB_KERNEL(cand_pb0q15, 0, 15)
