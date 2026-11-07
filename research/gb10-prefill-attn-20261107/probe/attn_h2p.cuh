// h2p: the h2 prime-attention row program on a three-stage K/V ring with per-stage mbarriers and a
// one-QK skew between the two query heads of a CTA (probe copy).
//
// h2 runs two query heads of one KV head in an 8-warp CTA; warps w and w + 4 (one per head) share
// a scheduler. h2's two CTA barriers per 32-key tile keep all eight warps in phase, so both warps of
// a scheduler reach their softmax at the same time and the tensor pipe idles. Here the stages are
// handed over by mbarriers (full: every thread's cp.async landed; empty: all eight warps read it),
// so the head-B warps can run one QK behind the head-A warps: one warp's softmax overlaps the
// other's MMAs. Each row still sees h2's tile sequence and the same QK chains, softmax expressions,
// bf16 P rounding, O rescale and PV chains, so O is bit-identical to h2.
//
// Knobs: AD = ldmatrix addressing (per-lane 32-bit offsets, K lanes ordered so each B operand lands
// in consecutive registers); MF = interior tiles skip the causal/tail mask compares (the mask is a
// no-op there: every key of the tile is at or before every row of the warp and the tile is full).
#pragma once
#include <type_traits>

static __device__ __forceinline__ void cp_async_16_sz(uint32_t dst, const void* src, uint32_t sz) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}

// D = A*B + 0: the same operation as mma_bf16 on a zeroed D, with the zero passed as the C operand
// so no MOV materializes it
static __device__ __forceinline__ void mma_bf16_z(CTile& D, const ATile& A, const BTile& B){
    const int* Ax=(const int*)A.x; const int* Bx=(const int*)B.x; float* Dx=D.x;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
        : "=f"(Dx[0]),"=f"(Dx[1]),"=f"(Dx[2]),"=f"(Dx[3])
        : "r"(Ax[0]),"r"(Ax[1]),"r"(Ax[2]),"r"(Ax[3]),"r"(Bx[0]),"r"(Bx[1]),"f"(0.0f));
}

template<bool AD, bool MF, bool ZC = false>
static __device__ __forceinline__ void h2p_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    constexpr int HD = 256, HEAD_DIM = HD, HPC = 2, BQ = BLOCK_Q;
    constexpr int HD_KTILES = HD / K_STEP, O_NBLK = HD / N_KEYS;
    constexpr int NT = HPC * N_WARPS * WARP_SZ;            // 256 threads
    constexpr int ST = 3;                                  // ring stages
    constexpr uint32_t PLANE_B = BK * HD * 2;              // 16 KB: one K or V plane
    constexpr uint32_t STAGE_B = 2 * PLANE_B;              // K plane then V plane
    const int warp = threadIdx.y, lane = threadIdx.x;
    const int wq = warp % N_WARPS;
    const int head = (int)blockIdx.y * HPC + warp / N_WARPS;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base = blockIdx.x * BQ;
    const int qrow_base = q_base + wq * M_ROWS;
    if (head >= n_head || q_base >= T) return;             // CTA-uniform (n_head even)
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* const s0 = (__nv_bfloat16*)smem_raw;
    uint64_t* const full = (uint64_t*)(smem_raw + ST * STAGE_B);
    uint64_t* const empty = full + ST;
    if (warp == 0 && lane == 0) {
        #pragma unroll
        for (int b = 0; b < ST; ++b) { mbar_init(&full[b], NT); mbar_init(&empty[b], 2 * N_WARPS); }
    }

    const int causal_i = causal;
    const int q_pos0w = (T_kv - T) + qrow_base;
    ATile Qf[HD_KTILES];
    // Q staging uses stages 0-1 before any copy is issued
    load_q_frags<HD, true>(Qf, Q, s0 + warp * M_ROWS * HEAD_DIM, qrow_base, nqw, head, n_head, head_dim, lane);
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

    // producer: thread (warp, lane) copies rows warp + 8j (j = 0..3), 16-byte chunk `lane`, of the
    // K and V planes; the plane's XOR swizzle (chunk ^ (row & 7)) is chunk ^ warp for all four rows.
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const uint32_t pdst = sbase + (uint32_t)(warp * HEAD_DIM * 2 + ((lane ^ warp) << 4));
    const __nv_bfloat16* const pk = Kw + (size_t)warp * kv_dim_k + kv_off + lane * 8;
    const __nv_bfloat16* const pv = Vw + (size_t)warp * kv_dim_v + kv_off + lane * 8;
    auto produce = [&](int tn) {
        const int b = tn % ST;
        if (tn >= ST) mbar_wait(&empty[b], (uint32_t)((tn / ST - 1) & 1));
        const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const bool ok = warp + 8 * j < nk1;   // rows past the tail land as zeros
            const uint32_t d = pdst + (uint32_t)b * STAGE_B + (uint32_t)(j * 8 * HEAD_DIM * 2);
            cp_async_16_sz(d, ok ? (const void*)(pk + (size_t)(k1 + 8 * j) * kv_dim_k) : (const void*)Kw, ok ? 16u : 0u);
            cp_async_16_sz(d + PLANE_B, ok ? (const void*)(pv + (size_t)(k1 + 8 * j) * kv_dim_v) : (const void*)Vw, ok ? 16u : 0u);
        }
        mbar_cp_arrive(&full[b]);
    };

    // AD: byte offset of (row block, 16-dim block j) = ofs[j & 3] + 128*(j >> 2) + rows*512
    uint32_t kofs[4], vofs[4];
    if constexpr (AD) {
        const int x = lane & 7;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            kofs[m] = sbase + (uint32_t)(((lane & 7) + 8 * (lane >> 4)) * (HEAD_DIM * 2)
                                         + 16 * ((2 * m + ((lane >> 3) & 1)) ^ x));
            vofs[m] = sbase + (uint32_t)((lane & 15) * (HEAD_DIM * 2) + 16 * ((2 * m + (lane >> 4)) ^ x));
        }
    }

    if (nt > 0) produce(0);
    if (nt > 1) produce(1);
    if (nt > 0 && warp >= N_WARPS) named_bar_sync(1, NT);   // head B starts one QK later

    for (int ti = 0; ti < nt; ++ti) {
        const int b = ti % ST;
        const int k0 = ti * BK;
        const int nk = min(BK, T_kv - k0);
        __nv_bfloat16* const sK = s0 + b * (STAGE_B / 2);
        __nv_bfloat16* const sV = sK + BK * HEAD_DIM;
        const uint32_t kpl = (uint32_t)b * STAGE_B, vpl = kpl + PLANE_B;
        mbar_wait(&full[b], (uint32_t)((ti / ST) & 1));

        // ---- QK^T -> 4 score CTiles in registers ----
        CTile Sc[BK / N_KEYS];
        #pragma unroll
        for (int g = 0; g < BK / N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
        for (int kg = 0; kg < BK; kg += 2 * N_KEYS) {
            CTile C0, C1;
            C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
            C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
            #pragma unroll
            for (int kt = 0; kt < HD_KTILES; ++kt) {
                ATile Kt;
                if constexpr (AD) ldsm_k_ad(Kt, kofs[kt & 3] + kpl + (uint32_t)(kg * HEAD_DIM * 2 + 128 * (kt >> 2)));
                else ld_A_sw(Kt, sK, kg, kt * 2, HEAD_DIM / 8);
                BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                if (ZC && kt == 0) { mma_bf16_z(C0, Qf[kt], Blo); mma_bf16_z(C1, Qf[kt], Bhi); }
                else { mma_bf16(C0, Qf[kt], Blo); mma_bf16(C1, Qf[kt], Bhi); }
            }
            Sc[kg / N_KEYS + 0] = C0;
            Sc[kg / N_KEYS + 1] = C1;
        }
        if (ti == 0 && warp < N_WARPS) named_bar_arrive(1, NT);

        // ---- softmax on registers (scale + causal/tail mask + 4-lane reduce) ----
        float alpha_lo, alpha_hi;
        auto softmax = [&](auto maskc) {
            constexpr bool MASK = decltype(maskc)::value;
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK / N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float s = Sc[g].x[l] * scale;
                    if constexpr (MASK) {
                        int col = g * N_KEYS + c0 + (l & 1);
                        int row = (l < 2) ? r_lo : r_hi;
                        int q_pos = q_pos0w + row;
                        if (col >= nk) s = NEG_INF;
                        if (causal_i && (k0 + col) > q_pos) s = NEG_INF;
                    }
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
            alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);
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
        };
        if constexpr (MF) {
            // warp-uniform: every key of the tile is at or before the warp's first row, tile full
            const bool interior = nk == BK && (!causal_i || k0 + BK - 1 <= q_pos0w);
            if (interior) softmax(std::false_type{}); else softmax(std::true_type{});
        } else {
            softmax(std::true_type{});
        }

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
                ATile A; ATile Bt;
                A = Pf[kk / K_STEP];
                if constexpr (AD) ldsm_vt_ad(Bt, vofs[(d0 / 16) & 3] + vpl + (uint32_t)(kk * HEAD_DIM * 2 + 128 * ((d0 / 16) >> 2)));
                else ld_A_trans_sw(Bt, sV, kk, d0 / 8, HEAD_DIM / 8);
                BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                if (ZC && kk == 0) { mma_bf16_z(Clo, A, Blo); mma_bf16_z(Chi, A, Bhi); }
                else { mma_bf16(Clo, A, Blo); mma_bf16(Chi, A, Bhi); }
            }
            O_acc[(d0 / N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0 / N_KEYS) + 0].x[1] += Clo.x[1];
            O_acc[(d0 / N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0 / N_KEYS) + 0].x[3] += Clo.x[3];
            O_acc[(d0 / N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0 / N_KEYS) + 1].x[1] += Chi.x[1];
            O_acc[(d0 / N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0 / N_KEYS) + 1].x[3] += Chi.x[3];
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[b]);   // this warp is done with stage b
        if (ti + 2 < nt) produce(ti + 2);
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

#define H2P_KERNEL(NAME, AD, MF, ...)                                                                     \
extern "C" __global__ void __launch_bounds__(2 * N_WARPS * WARP_SZ, 1) NAME(                         \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    h2p_body<AD, MF, ##__VA_ARGS__>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv, T, T_kv, scale, causal,\
                     kv_dim_k, kv_dim_v);                                                            \
}
H2P_KERNEL(cand_pp2, false, false)     // lean producer, h2 addressing
H2P_KERNEL(cand_ppad, true, false)     // + ldmatrix addressing
H2P_KERNEL(cand_ppadm, true, true)     // + mask-free interior tiles
H2P_KERNEL(cand_ppm, false, true)      // lean producer + mask-free interior, h2 addressing
H2P_KERNEL(cand_ppadz, true, false, true)  // ppad + zero C operand on each chain's first MMA
H2P_KERNEL(cand_ppadmz, true, true, true)  // ppadm + zero C operand
