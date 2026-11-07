// Candidate prime-attention bodies for the GB10 attention lane (probe copy; the landed kernel moves
// into flash_attn.cu). Each candidate is fa_prefill_qw_db_body<256, false, 2, true> (h2) copied
// statement for statement; only the staging schedule, barrier placement and where the Q fragments
// live change. Every row keeps h2's 32-key tile sequence, QK chains, softmax expressions, bf16 P
// rounding, O rescale and PV chains, so O is bit-identical by construction (the probe checks it).
//
// Knobs:
//   SB      one barrier per tile (wait, barrier, then prefetch the next tile into the stage the
//           barrier just released) instead of h2's prefetch-wait-barrier-compute-barrier
//   QREG    Q k-tiles held in registers (16 = all, as h2); k-tiles QREG..15 are ldmatrix'd each
//           tile from a persistent smem copy placed after the K/V stages
//   FA2S    FA2's staging schedule: one K plane and one V plane (V(i) lands during QK(i), K(i+1)
//           during softmax+PV(i), two barriers per tile), all of Q in smem (QREG must be 0)
//   PP      three-stage K/V ring with per-stage full/empty mbarriers instead of CTA barriers, and
//           the second head's warps start one QK later, so the two warps that share a scheduler
//           (warp w and w + 4) run out of phase: one's softmax overlaps the other's MMAs
//   AD      ldmatrix addressing: per-lane 32-bit shared offsets precomputed once (the XOR swizzle
//           folds into four offsets per lane plus immediates), and the K loads take their lanes
//           in the order that lands each B operand in two consecutive registers (no MOV pairs).
//           Same bytes reach the same HMMA operand slots.
//   NOSTAGE diagnostic: stage tile 0 once, no copies and no barriers in the loop
//   NOSM    diagnostic (with NOSTAGE): no softmax, P = bf16(S), alpha = 1
#pragma once

// K tile as two B operands: lanes 0-7 rows 0-7 chunk c, 8-15 rows 0-7 chunk c+1, 16-23 rows 8-15
// chunk c, 24-31 rows 8-15 chunk c+1; outputs land so x[0],x[2] / x[1],x[3] are the ld_A_sw values.
static __device__ __forceinline__ void ldsm_k_ad(ATile& t, uint32_t addr) {
    int* xi = (int*)t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[2]),"=r"(xi[1]),"=r"(xi[3]) : "r"(addr));
}
// ld_A_trans_sw's lane map and output order, address supplied
static __device__ __forceinline__ void ldsm_vt_ad(ATile& t, uint32_t addr) {
    int* xi = (int*)t.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(xi[0]),"=r"(xi[2]),"=r"(xi[1]),"=r"(xi[3]) : "r"(addr));
}

static __device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count) : "memory");
}
static __device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" :: "r"(a) : "memory");
}
// arrives once this thread's earlier cp.async copies have landed (count fixed at init: noinc)
static __device__ __forceinline__ void mbar_cp_arrive(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(a) : "memory");
}
static __device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT_%=; }"
                 :: "r"(a), "r"(parity) : "memory");
}
// 16-byte cp.async that zero-fills (src-size 0 reads nothing)
static __device__ __forceinline__ void cp_async_16_zero(__nv_bfloat16* smem_dst, const __nv_bfloat16* gsrc) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" :: "r"(d), "l"(gsrc));
}
static __device__ __forceinline__ void named_bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
static __device__ __forceinline__ void named_bar_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory");
}

template<bool SB, int QREG = 16, bool NOSTAGE = false, bool NOSM = false, bool FA2S = false, bool PP = false, bool AD = false>
static __device__ __forceinline__ void cand_body(
        const float* __restrict__ Q, const __nv_bfloat16* __restrict__ Kw,
        const __nv_bfloat16* __restrict__ Vw, float* __restrict__ O,
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,
        float scale, int causal, int kv_dim_k, int kv_dim_v)
{
    constexpr int HD = 256, HPC = 2;
    constexpr int HEAD_DIM  = HD;
    constexpr int BQ = BLOCK_Q;
    constexpr int HD_KTILES = HD / K_STEP;
    constexpr int O_NBLK    = HD / N_KEYS;
    constexpr int NT = HPC * N_WARPS * WARP_SZ;
    static_assert(QREG == HD_KTILES || QREG == 8 || QREG == 12 || QREG == 0, "swizzle needs 8k chunks");
    static_assert(!FA2S || (QREG == 0 && !SB && !NOSTAGE), "FA2S keeps all of Q in smem");
    static_assert(!PP || (QREG == HD_KTILES && !SB && !NOSTAGE && !FA2S), "PP: three stages, Q in registers");
    constexpr int PPST = 3;   // PP ring stages
    static_assert(!AD || (QREG == HD_KTILES && !FA2S && !PP), "AD: the four-plane layout");
    constexpr int QS = (HD_KTILES - QREG) * K_STEP;     // dims held in smem
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int wq = warp % N_WARPS;
    const int head    = (int)blockIdx.y * HPC + warp / N_WARPS;
    const int kv_head = head / (n_head / n_head_kv);
    const int q_base  = blockIdx.x * BQ;
    const int qrow_base = q_base + wq*M_ROWS;
    if (head >= n_head || q_base >= T) return;
    const int nqw = min(M_ROWS, T - qrow_base);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK0 = (__nv_bfloat16*)smem_raw;                // BK*HEAD_DIM
    __nv_bfloat16* sK1 = sK0 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sV0 = sK1 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sV1 = sV0 + BK*HEAD_DIM;                      // BK*HEAD_DIM
    __nv_bfloat16* sQstage = sK0 + warp*M_ROWS*HEAD_DIM;
    // persistent Q tail [warp][16][QS], 16-byte chunks XOR-swizzled by (row & 7). FA2S uses only
    // sK0 (K) and sK1 (V) and puts Q right after them.
    __nv_bfloat16* sQp = (FA2S ? sV0 : sV1 + BK*HEAD_DIM) + warp*M_ROWS*(QS > 0 ? QS : 8);

    // PP: stage b = K plane at sK0 + b*2*BK*HD, V plane after it; full[3] + empty[3] after the ring
    uint64_t* pp_full = (uint64_t*)(smem_raw + PPST * 2 * BK * HEAD_DIM * 2);
    uint64_t* pp_empty = pp_full + PPST;
    if constexpr (PP) {
        if (warp == 0 && lane == 0) {
            for (int b = 0; b < PPST; ++b) { mbar_init(&pp_full[b], NT); mbar_init(&pp_empty[b], 2 * N_WARPS); }
        }
    }

    const int causal_i = causal;
    {
        const int q_pos0w = (T_kv - T) + qrow_base;

        ATile Qf[QREG > 0 ? QREG : 1];
        if constexpr (QREG == HD_KTILES) {
            load_q_frags<HD, true>(Qf, Q, sQstage, qrow_base, nqw, head, n_head, head_dim, lane);
        } else {
            // the same __float2bfloat16 per element as load_q_frags
            for (int i = lane; i < M_ROWS*HEAD_DIM; i += WARP_SZ) {
                const int r = i / HEAD_DIM, d = i % HEAD_DIM;
                const float qv = (r < nqw) ? Q[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] : 0.0f;
                const __nv_bfloat16 b = __float2bfloat16(qv);
                if (d < QREG*K_STEP) sQstage[r*HEAD_DIM + ((d / 8) ^ (r & 7))*8 + (d & 7)] = b;
                else { const int e = d - QREG*K_STEP; sQp[r*QS + ((e / 8) ^ (r & 7))*8 + (e & 7)] = b; }
            }
            __syncwarp();
            #pragma unroll
            for (int kt = 0; kt < QREG; ++kt) ld_A_sw(Qf[kt], sQstage, 0, kt*2, HEAD_DIM/8);
            __syncwarp();
        }
        __syncthreads();   // all warps done with sK0∪sK1 before prefetch overwrites

        CTile O_acc[O_NBLK];
        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) { O_acc[c].x[0]=O_acc[c].x[1]=O_acc[c].x[2]=O_acc[c].x[3]=0.0f; }
        float m_lo = NEG_INF, m_hi = NEG_INF, l_lo = 0.0f, l_hi = 0.0f;
        const int r_lo = lane / 4;
        const int r_hi = r_lo + 8;
        const int c0   = (lane % 4) * 2;

        const size_t kv_off = (size_t)kv_head * head_dim;
        const int bt = warp*WARP_SZ + lane;
        // AD: byte offset of (row block, 16-dim block j) = ofs[j & 3] + 128*(j >> 2) + rows*512,
        // since (2j + h) ^ x = 8*(j >> 2) + ((2*(j & 3) + h) ^ x) for x = row & 7 < 8
        uint32_t kofs[4], vofs[4];
        if constexpr (AD) {
            const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
            const int x = lane & 7;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                kofs[m] = sb + (uint32_t)(((lane & 7) + 8 * (lane >> 4)) * (HEAD_DIM * 2)
                                          + 16 * ((2 * m + ((lane >> 3) & 1)) ^ x));
                vofs[m] = sb + (uint32_t)((lane & 15) * (HEAD_DIM * 2) + 16 * ((2 * m + (lane >> 4)) ^ x));
            }
        }

        const int q_pos_max = (T_kv - T) + q_base + (BQ - 1);
        int nt = (T_kv + BK - 1) / BK;
        if (causal_i) { int ntc = q_pos_max / BK + 1; nt = min(nt, ntc); }

        // PP producer: every thread copies its share of tile tn into stage tn % 3 once all warps
        // released the tile that stage held (tn - 3), then arrives on the stage's full barrier.
        auto pp_produce = [&](int tn) {
            const int b = tn % PPST;
            if (tn >= PPST) mbar_wait(&pp_empty[b], (uint32_t)((tn / PPST - 1) & 1));
            const int k1 = tn * BK, nk1 = min(BK, T_kv - k1);
            __nv_bfloat16* dK = sK0 + b * 2 * BK * HEAD_DIM;
            __nv_bfloat16* dV = dK + BK * HEAD_DIM;
            for (int i = bt; i < BK * (HEAD_DIM / 8); i += NT) {
                const int kk = i / (HEAD_DIM / 8), dv = i % (HEAD_DIM / 8), ds = dv ^ (kk & 7);
                if (kk < nk1) {
                    cp_async_16(dK + kk * HEAD_DIM + ds * 8, Kw + (size_t)(k1 + kk) * kv_dim_k + kv_off + dv * 8);
                    cp_async_16(dV + kk * HEAD_DIM + ds * 8, Vw + (size_t)(k1 + kk) * kv_dim_v + kv_off + dv * 8);
                } else {
                    cp_async_16_zero(dK + kk * HEAD_DIM + ds * 8, Kw);
                    cp_async_16_zero(dV + kk * HEAD_DIM + ds * 8, Vw);
                }
            }
            mbar_cp_arrive(&pp_full[b]);
        };

        if constexpr (PP) {
            if (nt > 0) pp_produce(0);
            if (nt > 1) pp_produce(1);
            if (nt > 0 && warp >= N_WARPS) named_bar_sync(1, NT);   // head B starts one QK later
        } else if constexpr (FA2S) {
            if (nt > 0) {
                stage_kv_plane_async<HD, NT>(sK0, Kw, 0, min(BK, T_kv), kv_dim_k, kv_off, bt);
                cp_async_wait_0();
            }
            __syncthreads();
        } else if (nt > 0) {
            stage_kv_tile_async<HD, true, NT>(sK0, sV0, Kw, Vw, 0, min(BK, T_kv), kv_dim_k, kv_dim_v, kv_off, bt);
            if constexpr (NOSTAGE) { cp_async_wait_0(); __syncthreads(); }
        }

        for (int ti = 0; ti < nt; ++ti) {
            const int k0 = ti * BK;
            const int nk = min(BK, T_kv - k0);
            __nv_bfloat16* sK = (NOSTAGE || FA2S) ? sK0 : ((ti & 1) ? sK1 : sK0);
            __nv_bfloat16* sV = FA2S ? sK1 : NOSTAGE ? sV0 : ((ti & 1) ? sV1 : sV0);
            if constexpr (PP) {
                sK = sK0 + (ti % PPST) * 2 * BK * HEAD_DIM;
                sV = sK + BK * HEAD_DIM;
                mbar_wait(&pp_full[ti % PPST], (uint32_t)((ti / PPST) & 1));
            } else if constexpr (FA2S) {
                // V(ti) lands while QK(ti) runs; the plane is free (the last barrier followed PV(ti-1))
                stage_kv_plane_async<HD, NT>(sV, Vw, k0, nk, kv_dim_v, kv_off, bt);
            } else if constexpr (NOSTAGE) {
            } else if constexpr (SB) {
                cp_async_wait_0();
                __syncthreads();   // tile ti visible; every warp is past tile ti-1's stage
                if (ti + 1 < nt) {
                    const int k1 = (ti + 1) * BK;
                    stage_kv_tile_async<HD, true, NT>((ti & 1) ? sK0 : sK1, (ti & 1) ? sV0 : sV1,
                        Kw, Vw, k1, min(BK, T_kv - k1), kv_dim_k, kv_dim_v, kv_off, bt);
                }
            } else {
                if (ti + 1 < nt) {
                    const int k1 = (ti + 1) * BK;
                    stage_kv_tile_async<HD, true, NT>((ti & 1) ? sK0 : sK1, (ti & 1) ? sV0 : sV1,
                        Kw, Vw, k1, min(BK, T_kv - k1), kv_dim_k, kv_dim_v, kv_off, bt);
                    cp_async_wait_1();
                } else {
                    cp_async_wait_0();
                }
                __syncthreads();
            }

            const uint32_t kpl = (ti & 1) ? (uint32_t)(BK * HEAD_DIM * 2) : 0u;          // sK1 : sK0
            const uint32_t vpl = (uint32_t)(2 * BK * HEAD_DIM * 2) + kpl;                // sV1 : sV0
            // ---- GEMM0: QK^T -> 4 score CTiles HELD IN REGISTERS ----
            CTile Sc[BK/N_KEYS];
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) { Sc[g].x[0]=Sc[g].x[1]=Sc[g].x[2]=Sc[g].x[3]=0.0f; }
            if constexpr (QREG == HD_KTILES) {
                for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                    CTile C0, C1;
                    C0.x[0]=C0.x[1]=C0.x[2]=C0.x[3]=0.0f;
                    C1.x[0]=C1.x[1]=C1.x[2]=C1.x[3]=0.0f;
                    #pragma unroll
                    for (int kt = 0; kt < HD_KTILES; ++kt) {
                        ATile Kt;
                        if constexpr (AD) ldsm_k_ad(Kt, kofs[kt & 3] + kpl + (uint32_t)(kg * HEAD_DIM * 2 + 128 * (kt >> 2)));
                        else ld_A_sw(Kt, sK, kg, kt*2, HEAD_DIM/8);
                        BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                        BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                        mma_bf16(C0, Qf[kt], Blo);
                        mma_bf16(C1, Qf[kt], Bhi);
                    }
                    Sc[kg/N_KEYS + 0] = C0;
                    Sc[kg/N_KEYS + 1] = C1;
                }
            } else {
                // k-tile outer, key group inner: each of the four score chains still runs
                // kt = 0..15 in order from zero, so every score is the same float
                #pragma unroll
                for (int kt = 0; kt < HD_KTILES; ++kt) {
                    ATile Qk;
                    if (kt < QREG) Qk = Qf[kt < QREG ? kt : 0];
                    else ld_A_sw(Qk, sQp, 0, (kt - QREG)*2, QS/8);
                    #pragma unroll
                    for (int kg = 0; kg < BK; kg += 2*N_KEYS) {
                        ATile Kt;
                        ld_A_sw(Kt, sK, kg, kt*2, HEAD_DIM/8);
                        BTile Blo; Blo.x[0]=Kt.x[0]; Blo.x[1]=Kt.x[2];
                        BTile Bhi; Bhi.x[0]=Kt.x[1]; Bhi.x[1]=Kt.x[3];
                        mma_bf16(Sc[kg/N_KEYS + 0], Qk, Blo);
                        mma_bf16(Sc[kg/N_KEYS + 1], Qk, Bhi);
                    }
                }
            }

            if constexpr (PP) {
                if (ti == 0 && warp < N_WARPS) named_bar_arrive(1, NT);
            }
            if constexpr (FA2S) {
                cp_async_wait_0();
                __syncthreads();   // every warp is done with K(ti); V(ti) is visible
                if (ti + 1 < nt) {
                    const int k1 = (ti + 1) * BK;
                    stage_kv_plane_async<HD, NT>(sK, Kw, k1, min(BK, T_kv - k1), kv_dim_k, kv_off, bt);
                }
            }

            float alpha_lo = 1.0f, alpha_hi = 1.0f;
            if constexpr (!NOSM) {
            // ---- SOFTMAX on registers (scale + causal mask + 4-lane reduce) ----
            float s_tile_max_lo = NEG_INF, s_tile_max_hi = NEG_INF;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int col = g*N_KEYS + c0 + (l & 1);
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
            alpha_lo = (m_prev_lo == NEG_INF) ? 0.0f : exp2f((m_prev_lo - m_new_lo) * LOG2E);
            alpha_hi = (m_prev_hi == NEG_INF) ? 0.0f : exp2f((m_prev_hi - m_new_hi) * LOG2E);

            float l_part_lo = 0.0f, l_part_hi = 0.0f;
            #pragma unroll
            for (int g = 0; g < BK/N_KEYS; ++g) {
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
            }

            ATile Pf[BK / K_STEP];
            // C's adjacent output columns are exactly A's bf16-pair layout.
            // Preserve each scalar RN conversion; no reduction is introduced.
            #pragma unroll
            for (int kk = 0; kk < BK; kk += K_STEP) {
                const int g = kk / N_KEYS;
                Pf[kk / K_STEP].x[0] = __floats2bfloat162_rn(Sc[g].x[0], Sc[g].x[1]);
                Pf[kk / K_STEP].x[1] = __floats2bfloat162_rn(Sc[g].x[2], Sc[g].x[3]);
                Pf[kk / K_STEP].x[2] = __floats2bfloat162_rn(Sc[g+1].x[0], Sc[g+1].x[1]);
                Pf[kk / K_STEP].x[3] = __floats2bfloat162_rn(Sc[g+1].x[2], Sc[g+1].x[3]);
            }

            #pragma unroll
            for (int c = 0; c < O_NBLK; ++c) {
                O_acc[c].x[0] *= alpha_lo; O_acc[c].x[1] *= alpha_lo;
                O_acc[c].x[2] *= alpha_hi; O_acc[c].x[3] *= alpha_hi;
            }

            for (int d0 = 0; d0 < HEAD_DIM; d0 += 2*N_KEYS) {
                CTile Clo, Chi;
                Clo.x[0]=Clo.x[1]=Clo.x[2]=Clo.x[3]=0.0f;
                Chi.x[0]=Chi.x[1]=Chi.x[2]=Chi.x[3]=0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += K_STEP) {
                    ATile A; ATile Bt;
                    A = Pf[kk / K_STEP];
                    if constexpr (AD) ldsm_vt_ad(Bt, vofs[(d0 / 16) & 3] + vpl + (uint32_t)(kk * HEAD_DIM * 2 + 128 * ((d0 / 16) >> 2)));
                    else ld_A_trans_sw(Bt, sV, kk, d0/8, HEAD_DIM/8);
                    BTile Blo; Blo.x[0]=Bt.x[0]; Blo.x[1]=Bt.x[2];
                    BTile Bhi; Bhi.x[0]=Bt.x[1]; Bhi.x[1]=Bt.x[3];
                    mma_bf16(Clo, A, Blo);
                    mma_bf16(Chi, A, Bhi);
                }
                O_acc[(d0/N_KEYS) + 0].x[0] += Clo.x[0]; O_acc[(d0/N_KEYS) + 0].x[1] += Clo.x[1];
                O_acc[(d0/N_KEYS) + 0].x[2] += Clo.x[2]; O_acc[(d0/N_KEYS) + 0].x[3] += Clo.x[3];
                O_acc[(d0/N_KEYS) + 1].x[0] += Chi.x[0]; O_acc[(d0/N_KEYS) + 1].x[1] += Chi.x[1];
                O_acc[(d0/N_KEYS) + 1].x[2] += Chi.x[2]; O_acc[(d0/N_KEYS) + 1].x[3] += Chi.x[3];
            }
            if constexpr (PP) {
                __syncwarp();
                if (lane == 0) mbar_arrive(&pp_empty[ti % PPST]);
                if (ti + 2 < nt) pp_produce(ti + 2);
            } else if constexpr (FA2S) {
                cp_async_wait_0();
                __syncthreads();   // every warp is done with V(ti); K(ti+1) is visible
            } else if constexpr (!SB && !NOSTAGE) {
                __syncthreads();   // all readers release the tile before overwrite
            }
        }

        #pragma unroll
        for (int c = 0; c < O_NBLK; ++c) {
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                int r = CTile::get_i(l);
                int d = c*N_KEYS + CTile::get_j(l);
                if (r < nqw) {
                    const float lv = (l < 2) ? l_lo : l_hi;
                    float linv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
                    O[((size_t)(qrow_base + r) * n_head + head) * head_dim + d] = O_acc[c].x[l] * linv;
                }
            }
        }
        __syncthreads();
    }
}

#define CAND_KERNEL(NAME, ...)                                                                       \
extern "C" __global__ void __launch_bounds__(2 * N_WARPS * WARP_SZ, 1) NAME(                         \
        const float* Q, const __nv_bfloat16* Kw, const __nv_bfloat16* Vw, float* O,                  \
        int head_dim, int n_head, int n_head_kv, int T, int T_kv,                                    \
        float scale, int causal, int kv_dim_k, int kv_dim_v) {                                       \
    cand_body<__VA_ARGS__>(Q, Kw, Vw, O, head_dim, n_head, n_head_kv,                                \
                           T, T_kv, scale, causal, kv_dim_k, kv_dim_v);                              \
}

CAND_KERNEL(cand_h2c, false)                 // h2 copied (control: same speed, same bytes)
CAND_KERNEL(cand_sb2, true)                  // one barrier per tile
CAND_KERNEL(cand_qh8, true, 8)               // sb2 + Q k-tiles 8..15 from smem
CAND_KERNEL(cand_qh12, true, 12)             // sb2 + Q k-tiles 12..15 from smem
CAND_KERNEL(cand_qh8db, false, 8)            // h2 barriers + Q k-tiles 8..15 from smem
CAND_KERNEL(diag_nostage, false, 16, true)   // ceiling diag: no staging, no barriers
CAND_KERNEL(diag_nosm, false, 16, true, true)// ceiling diag: no staging, no softmax
CAND_KERNEL(diag_qh8ns, false, 8, true)      // ceiling diag: qh8 without staging
CAND_KERNEL(cand_fa2s, false, 0, false, false, true) // all Q in smem, FA2 staging schedule
CAND_KERNEL(cand_pp, false, 16, false, false, false, true) // mbarrier ring, head B one QK behind
CAND_KERNEL(cand_ad, false, 16, false, false, false, false, true) // h2 + ldmatrix addressing
CAND_KERNEL(cand_sbad, true, 16, false, false, false, false, true) // sb2 + ldmatrix addressing
