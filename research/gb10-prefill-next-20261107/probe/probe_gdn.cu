// probe_gdn.cu: GB10 device probe for the GDN chunked prime's state pass (K4) and output pass
// (K5) at the Qwen3.8-27B shape: 48 value heads, 16 compact q/k heads, head dim 128, C = 32.
//
// The shipped kernels come in verbatim through `#include "hybrid.cu"` (built with -I pointing at
// the tree's cu/ dir). Candidates are exact rewrites of the shipped K4: every output element runs
// the same mma.sync instructions on the same operand values in the same k order, so Y, Ssnap,
// state_out and the K5 output o must match the shipped pair bit for bit.
//
//   k4 v0       gdn_chunk_state_mma as shipped (32 state columns per CTA, grid (48, 4))
//   k4 c32      the template below at 32 columns (must equal v0, checks the template itself)
//   k4 c32pf    + register prefetch of the next chunk's U values, gate cumsum and beta
//   k4 c64      64 state columns per CTA, grid (48, 2): one wave of 96 CTAs at 2 CTAs/SM
//   k4 c64pf    c64 + the register prefetch
//   k4 *ns      the same without the Y / Ssnap stores (diagnostic only: what the stores cost)
//   k5 cfast    gdn_chunk_output_mma, chunks-fastest grid (main)
//   k5 hfast    heads-fastest grid (the lane's default)
//
// usage: probe_gdn T REPS
#include "hybrid.cu"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// ------------------------------------------------------------------------------------------
// K4 template. COLS state columns per CTA, 8 warps.
//   Macc: warp (mh = warp / NNQ, nq = warp % NNQ) owns columns mh*16 .. +15 and state rows
//   nq*IPW .. +IPW-1 as NT m16n8 tiles (the shipped mapping at COLS = 32).
//   Step A: warp (mj = warp / 4, g = warp % 4) owns Y rows mj*16 .. +15 and COLS/32 n8 tiles
//   starting at n8 index g * COLS/32 (the shipped mapping at COLS = 32).
// ------------------------------------------------------------------------------------------
template <int COLS, bool PF, bool STORE>
__global__ void __launch_bounds__(256, 2)
k4_tpl(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
       const float* __restrict__ beta,
       const float* __restrict__ U, const __nv_bfloat16* __restrict__ Wb16,
       __half* __restrict__ Y, __half* __restrict__ Ssnap,
       const float* __restrict__ state_in, float* __restrict__ state_out,
       int H, int T, int C, int hk) {
    constexpr int D = GDN_D;
    constexpr int NMH = COLS / 16;          // 16-column groups
    constexpr int NNQ = 8 / NMH;            // state-row groups
    constexpr int IPW = D / NNQ;            // state rows per warp
    constexpr int NT = IPW / 8;             // m16n8 tiles per warp
    constexpr int KBS = COLS == 32 ? 2 : 1; // k-tile stages (64 columns: single, refilled per chunk)
    constexpr int MB_STR = D + 8;
    constexpr int YS = COLS + 8;
    constexpr int NA = COLS / 32;           // step-A n8 tiles per warp
    const int h = blockIdx.x, col0 = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;

    __shared__ __nv_bfloat16 Wb[2][32 * D];
    __shared__ __nv_bfloat16 kb[KBS][32 * D];
    __shared__ __nv_bfloat16 Mb[COLS * MB_STR];
    __shared__ __nv_bfloat16 ys[32 * YS];
    __shared__ float gk[32];

    const int fr = lane / 4, fc = (lane % 4) * 2;
    const int mh = warp / NNQ, nq = warp % NNQ;
    CTile Macc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * IPW + t * 8 + fc + (l & 1);
            Macc[t].x[l] = state_in[((size_t)h * D + col0 + col) * D + i];
        }

    const int NC = (T + C - 1) / C;
    auto stage_w = [&](int c_, int buf_) {
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {
            int r = idx / (D / 8), seg = idx % (D / 8);
            cp_async16_g(&Wb[buf_][r * D + seg * 8],
                         Wb16 + (((size_t)c_ * H + h) * C + r) * D + seg * 8, 16);
        }
    };
    auto stage_k = [&](int c_, int buf_) {
        int t0_ = c_ * C;
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {
            int r = idx / (D / 8), seg = idx % (D / 8);
            cp_async16_g(&kb[buf_][r * D + seg * 8],
                         kb16 + ((size_t)(t0_ + r) * hk + (h % hk)) * D + seg * 8,
                         (t0_ + r < T) ? 16 : 0);
        }
    };
    const int mj = warp / 4, g4 = warp % 4;
    // register prefetch of chunk c's U values (step-A epilogue), gate cumsum and beta
    float upre[NA][4];
    float gpre = 0.f, bpre = 0.f, gCpre = 0.f;
    auto prefetch = [&](int c_) {
        const int t0_ = c_ * C;
        const int Cc_ = min(C, T - t0_);
        #pragma unroll
        for (int a = 0; a < NA; a++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int j = mj * 16 + fr + ((l < 2) ? 0 : 8);
                int n8 = g4 * NA + a;
                int col = n8 * 8 + fc + (l & 1);
                upre[a][l] = (j < Cc_) ? U[(((size_t)c_ * H + h) * C + j) * D + col0 + col] : 0.f;
            }
        gCpre = gcum[(size_t)(t0_ + Cc_ - 1) * H + h];
        if (tid < Cc_) {
            gpre = gcum[(size_t)(t0_ + tid) * H + h];
            bpre = beta[(size_t)(t0_ + tid) * H + h];
        }
    };
    if (PF) prefetch(0);

    stage_w(0, 0);
    stage_k(0, 0);
    cp_commit();
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        const int cur = c & 1;
        const int kcur = KBS == 2 ? cur : 0;
        if (STORE) {
            __half* sc_base = Ssnap + ((size_t)c * H + h) * D * D;
            #pragma unroll
            for (int t = 0; t < NT; t++)
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
                    int i = nq * IPW + t * 8 + fc + (l & 1);
                    int gc = col0 + col;
                    sc_base[(size_t)(gc >> 5) * (D * 32) + (size_t)i * 32 + (gc & 31)] =
                        __float2half(Macc[t].x[l]);
                }
        }
        #pragma unroll
        for (int t = 0; t < NT; t++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
                int i = nq * IPW + t * 8 + fc + (l & 1);
                Mb[col * MB_STR + i] = __float2bfloat16(Macc[t].x[l]);
            }
        float gC;
        if (PF) {
            gC = gCpre;
            if (tid < Cc) gk[tid] = expf(gC - gpre) * bpre;
            else if (tid < 32) gk[tid] = 0.0f;
        } else {
            gC = gcum[(size_t)(t0 + Cc - 1) * H + h];
            if (tid < Cc) {
                gk[tid] = expf(gC - gcum[(size_t)(t0 + tid) * H + h])
                        * beta[(size_t)(t0 + tid) * H + h];
            } else if (tid < 32) {
                gk[tid] = 0.0f;
            }
        }
        float ucur[NA][4];
        if (PF) {
            #pragma unroll
            for (int a = 0; a < NA; a++)
                #pragma unroll
                for (int l = 0; l < 4; l++) ucur[a][l] = upre[a][l];
        }
        cp_wait<0>();
        __syncthreads();
        if (KBS == 2) {
            if (c + 1 < NC) { stage_w(c + 1, cur ^ 1); stage_k(c + 1, cur ^ 1); cp_commit(); }
        } else {
            // 64 columns: k for THIS chunk was staged with W at the previous chunk top's
            // commit only for c == 0; refill the single k buffer now (consumed after step A).
            if (c > 0) { stage_k(c, 0); cp_commit(); }
            if (c + 1 < NC) { stage_w(c + 1, cur ^ 1); cp_commit(); }
        }
        if (PF && c + 1 < NC) prefetch(c + 1);

        // step A: Y = U - W . S  (bf16 mma, k = 128 ascending; one W fragment per k16 step
        // feeds every n8 tile of the warp)
        {
            CTile Sc[NA];
            #pragma unroll
            for (int a = 0; a < NA; a++) Sc[a].x[0] = Sc[a].x[1] = Sc[a].x[2] = Sc[a].x[3] = 0.0f;
            #pragma unroll
            for (int k16 = 0; k16 < D / 16; k16++) {
                ATile A;
                ld_A(A, Wb[cur] + (mj * 16) * D + k16 * 16, D / 2, lane);
                const int colg = NA == 1 ? g4 / 2 : g4;
                ATile Bt;
                ld_A(Bt, Mb + (colg * 16) * MB_STR + k16 * 16, MB_STR / 2, lane);
                if constexpr (NA == 1) {
                    BTile B;
                    if ((g4 & 1) == 0) { B.x[0] = Bt.x[0]; B.x[1] = Bt.x[2]; }
                    else               { B.x[0] = Bt.x[1]; B.x[1] = Bt.x[3]; }
                    mma_bf16(Sc[0], A, B);
                } else {
                    BTile B0, B1;
                    B0.x[0] = Bt.x[0]; B0.x[1] = Bt.x[2];
                    B1.x[0] = Bt.x[1]; B1.x[1] = Bt.x[3];
                    mma_bf16(Sc[0], A, B0);
                    mma_bf16(Sc[1], A, B1);
                }
            }
            #pragma unroll
            for (int a = 0; a < NA; a++)
                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int j = mj * 16 + fr + ((l < 2) ? 0 : 8);
                    int n8 = g4 * NA + a;
                    int col = n8 * 8 + fc + (l & 1);
                    if (j < Cc) {
                        float u = PF ? ucur[a][l]
                                     : U[(((size_t)c * H + h) * C + j) * D + col0 + col];
                        float y = u - Sc[a].x[l];
                        if (STORE) Y[(((size_t)c * H + h) * C + j) * D + col0 + col] = __float2half(y);
                        ys[j * YS + col] = __float2bfloat16(y * gk[j]);
                    } else {
                        ys[j * YS + col] = __float2bfloat16(0.0f);
                    }
                }
        }
        if (KBS == 1) {
            // the k tile of this chunk must have landed; the next W stage may still be in flight
            if (c + 1 < NC) cp_wait<1>(); else cp_wait<0>();
        }
        __syncthreads();

        // step B: S <- bC S + (gk y)^T k
        const float bC = expf(gC);
        #pragma unroll
        for (int t = 0; t < NT; t++)
            #pragma unroll
            for (int l = 0; l < 4; l++) Macc[t].x[l] *= bC;
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A_trans(A, ys + (k16 * 16) * YS + mh * 16, YS / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < NT / 2; p2++) {
                ATile Bt;
                ld_A_trans(Bt, kb[kcur] + (k16 * 16) * D + nq * IPW + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(Macc[p2 * 2 + 0], A, Blo);
                mma_bf16(Macc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < NT; t++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * IPW + t * 8 + fc + (l & 1);
            state_out[((size_t)h * D + col0 + col) * D + i] = Macc[t].x[l];
        }
}

// ------------------------------------------------------------------------------------------
// Fused K4+K5 at 64 state columns per CTA (grid (H, 2), 8 warps, 2 CTAs/SM): the state pass also
// computes the chunk output o, so Y and Ssnap never leave the CTA. Every output element runs the
// shipped kernels' mma chains on the same operand values:
//   step A  Y[j][col] = U - sum_{k16 = 0..7} mma_bf16(W[j][k16], bf16(S)[k16][col])     (K4)
//   phase 1 o[j][col] = sum_{k16 = 0..7} mma_f16(f16(q)[j][k16], f16(S)[k16][col])       (K5)
//           o *= exp(G_j); o += sum_{k16 = 0..1} mma_f16(f16(P)[j][k16], f16(Y)[k16][col]); o *= scale
//   step B  S[col][i] = bC S + sum_{k16 = 0..1} mma_bf16(bf16(gk y)^T[col][k16], k[k16][i])  (K4)
// Warp (mh = warp >> 1, nq = warp & 1) holds S^T for cols mh*16 .. +15 and state rows nq*64 .. +63
// (8 m16n8 tiles). The B fragments of S come straight from those registers (packed exactly as K4's
// Mb / Ssnap stores round them); the k16 chains of step A and phase 1 run blocks 0..3 on nq = 0 and
// continue 4..7 on nq = 1 from nq = 0's accumulators through shared memory, so each chain keeps
// K4's / K5's order. Phase 2 and the output run on nq = 1, with Y's B fragments transposed in
// registers (movmatrix).
__device__ __forceinline__ unsigned k45_pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(unsigned*)&v;
}
__device__ __forceinline__ unsigned k45_pack_f16(float lo, float hi) {
    __half2 v = __floats2half2_rn(lo, hi);
    return *(unsigned*)&v;
}
__device__ __forceinline__ unsigned k45_trans(unsigned a) {
    unsigned d;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
    return d;
}
__device__ __forceinline__ void k45_mma_bf16(CTile& D, const ATile& A, unsigned b0, unsigned b1) {
    BTile B; *(unsigned*)&B.x[0] = b0; *(unsigned*)&B.x[1] = b1;
    mma_bf16(D, A, B);
}
__device__ __forceinline__ void k45_mma_f16(CTile& D, const ATile& A, unsigned b0, unsigned b1) {
    BTile B; *(unsigned*)&B.x[0] = b0; *(unsigned*)&B.x[1] = b1;
    mma_f16(D, A, B);
}

constexpr int K45_QS = GDN_D + 8;     // q tile row stride (halves)
constexpr int K45_YS = 64 + 8;        // ys row stride (bf16)
struct K45Smem {
    __nv_bfloat16 Wb[32 * GDN_D];      // W tile of the chunk (single buffer)
    __nv_bfloat16 kb[32 * GDN_D];      // k tile of the chunk (single buffer)
    __half qs[32 * K45_QS];            // f16 q rows of the chunk
    __half ps[32 * 40];                // f16 P rows (masked, zero above the diagonal)
    __nv_bfloat16 ys[32 * K45_YS];     // bf16(y * gk) for step B
    float4 hand[4][4][32];             // chain hand-off: [mh][tile][lane]
    float gk[32], bj[32];
};

__global__ void __launch_bounds__(256, 2)
k45_c64(const __nv_bfloat16* __restrict__ kb16, const float* __restrict__ gcum,
        const float* __restrict__ beta, const float* __restrict__ U,
        const __nv_bfloat16* __restrict__ Wb16, const float* __restrict__ q,
        const float* __restrict__ P, float* __restrict__ o,
        const float* __restrict__ state_in, float* __restrict__ state_out,
        int H, int T, int C, float scale, int hk) {
    constexpr int D = GDN_D;
    extern __shared__ __align__(16) unsigned char k45_raw[];
    K45Smem& sm = *reinterpret_cast<K45Smem*>(k45_raw);
    const int h = blockIdx.x, col0 = blockIdx.y * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int fr = lane >> 2, fc = (lane & 3) * 2;
    const int mh = warp >> 1, nq = warp & 1;
    CTile Macc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 64 + t * 8 + fc + (l & 1);
            Macc[t].x[l] = state_in[((size_t)h * D + col0 + col) * D + i];
        }
    const int NC = (T + C - 1) / C;
    const int hq = h % hk;
    auto stage_w = [&](int c_) {
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {
            int r = idx / (D / 8), seg = idx % (D / 8);
            cp_async16_g(&sm.Wb[r * D + seg * 8], Wb16 + (((size_t)c_ * H + h) * C + r) * D + seg * 8, 16);
        }
    };
    auto stage_k = [&](int c_) {
        int t0_ = c_ * C;
        for (int idx = tid; idx < 32 * D / 8; idx += 256) {
            int r = idx / (D / 8), seg = idx % (D / 8);
            cp_async16_g(&sm.kb[r * D + seg * 8], kb16 + ((size_t)(t0_ + r) * hk + hq) * D + seg * 8,
                         (t0_ + r < T) ? 16 : 0);
        }
    };
    // register prefetch of the next chunk's q (16 floats), P (4) and gate/beta (tid < 32)
    float4 qpre[4], ppre;
    float gpre = 0.f, bpre = 0.f, gCpre = 0.f;
    auto prefetch = [&](int c_) {
        const int t0_ = c_ * C, Cc_ = min(C, T - t0_);
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int idx = tid + u * 256;                 // 1024 float4 = 32 rows x 32 float4
            int r = idx >> 5, d4 = idx & 31;
            qpre[u] = (r < Cc_) ? *(const float4*)(q + ((size_t)(t0_ + r) * hk + hq) * D + d4 * 4)
                                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        {
            int r = tid >> 3, i4 = tid & 7;          // 256 float4 = 32 rows x 8 float4
            ppre = (r < Cc_) ? *(const float4*)(P + (((size_t)c_ * H + h) * C + r) * C + i4 * 4)
                             : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        gCpre = gcum[(size_t)(t0_ + Cc_ - 1) * H + h];
        if (tid < Cc_) {
            gpre = gcum[(size_t)(t0_ + tid) * H + h];
            bpre = beta[(size_t)(t0_ + tid) * H + h];
        }
    };
    prefetch(0);
    stage_w(0);
    cp_commit();
    for (int c = 0; c < NC; c++) {
        const int t0 = c * C;
        const int Cc = min(C, T - t0);
        // ---- chunk top: q / P tiles (K5's f16 conversions and masks), gk, b_j; k tile for step B
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            int idx = tid + u * 256, r = idx >> 5, d4 = idx & 31;
            __half* qd = &sm.qs[r * K45_QS + d4 * 4];
            qd[0] = __float2half(qpre[u].x); qd[1] = __float2half(qpre[u].y);
            qd[2] = __float2half(qpre[u].z); qd[3] = __float2half(qpre[u].w);
        }
        {
            int r = tid >> 3, i4 = tid & 7;
            float pv[4] = {ppre.x, ppre.y, ppre.z, ppre.w};
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int i = i4 * 4 + e;
                float v = (r < Cc && i < Cc && i <= r) ? pv[e] : 0.0f;
                sm.ps[r * 40 + i] = __float2half(v);
            }
        }
        const float gC = gCpre;
        if (tid < 32) {
            sm.gk[tid] = (tid < Cc) ? expf(gC - gpre) * bpre : 0.0f;
            sm.bj[tid] = (tid < Cc) ? expf(gpre) : 0.0f;
        }
        stage_k(c);
        cp_commit();
        cp_wait<1>();                // W(c) landed; k(c) may still be in flight
        __syncthreads();             // S1
        if (c + 1 < NC) prefetch(c + 1);

        // ---- step A and phase 1, k16 blocks 0..3 on nq = 0 (from zero), 4..7 on nq = 1
        CTile CA[2][2], CP[2][2];
        #pragma unroll
        for (int a = 0; a < 2; a++)
            #pragma unroll
            for (int b2 = 0; b2 < 2; b2++)
                #pragma unroll
                for (int l = 0; l < 4; l++) { CA[a][b2].x[l] = 0.0f; CP[a][b2].x[l] = 0.0f; }
        float uval[2][2][4];
        if (nq == 1) {
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        int j = jt * 16 + fr + ((l < 2) ? 0 : 8);
                        int col = mh * 16 + hf * 8 + fc + (l & 1);
                        uval[jt][hf][l] = (j < Cc) ? U[(((size_t)c * H + h) * C + j) * D + col0 + col] : 0.0f;
                    }
        }
        auto chain = [&](int kb0) {
            #pragma unroll
            for (int kl = 0; kl < 4; kl++) {
                const int kb = kb0 + kl;
                const unsigned bb0 = k45_pack_bf16(Macc[2 * kl].x[0], Macc[2 * kl].x[1]);
                const unsigned bb1 = k45_pack_bf16(Macc[2 * kl + 1].x[0], Macc[2 * kl + 1].x[1]);
                const unsigned bb2 = k45_pack_bf16(Macc[2 * kl].x[2], Macc[2 * kl].x[3]);
                const unsigned bb3 = k45_pack_bf16(Macc[2 * kl + 1].x[2], Macc[2 * kl + 1].x[3]);
                const unsigned bh0 = k45_pack_f16(Macc[2 * kl].x[0], Macc[2 * kl].x[1]);
                const unsigned bh1 = k45_pack_f16(Macc[2 * kl + 1].x[0], Macc[2 * kl + 1].x[1]);
                const unsigned bh2 = k45_pack_f16(Macc[2 * kl].x[2], Macc[2 * kl].x[3]);
                const unsigned bh3 = k45_pack_f16(Macc[2 * kl + 1].x[2], Macc[2 * kl + 1].x[3]);
                #pragma unroll
                for (int jt = 0; jt < 2; jt++) {
                    ATile A;
                    ld_A(A, sm.Wb + (jt * 16) * D + kb * 16, D / 2, lane);
                    k45_mma_bf16(CA[jt][0], A, bb0, bb1);
                    k45_mma_bf16(CA[jt][1], A, bb2, bb3);
                    ATile Aq;
                    ld_A(Aq, (const __nv_bfloat16*)(sm.qs + (jt * 16) * K45_QS + kb * 16), K45_QS / 2, lane);
                    k45_mma_f16(CP[jt][0], Aq, bh0, bh1);
                    k45_mma_f16(CP[jt][1], Aq, bh2, bh3);
                }
            }
        };
        if (nq == 0) {
            chain(0);
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++)
                    sm.hand[mh][jt * 2 + hf][lane] = make_float4(CA[jt][hf].x[0], CA[jt][hf].x[1], CA[jt][hf].x[2], CA[jt][hf].x[3]);
        }
        __syncthreads();             // S2
        unsigned yh[2][2][2];        // f16 Y pairs (nq = 1): [jt][hf][row half]
        if (nq == 1) {
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    float4 v = sm.hand[mh][jt * 2 + hf][lane];
                    CA[jt][hf].x[0] = v.x; CA[jt][hf].x[1] = v.y; CA[jt][hf].x[2] = v.z; CA[jt][hf].x[3] = v.w;
                }
            // The chain continues on the hand-off: zero-start CP here would break phase 1's order,
            // so phase 1's partials come over after this step (below); step A runs blocks 4..7 now.
            #pragma unroll
            for (int kl = 0; kl < 4; kl++) {
                const int kb = 4 + kl;
                const unsigned bb0 = k45_pack_bf16(Macc[2 * kl].x[0], Macc[2 * kl].x[1]);
                const unsigned bb1 = k45_pack_bf16(Macc[2 * kl + 1].x[0], Macc[2 * kl + 1].x[1]);
                const unsigned bb2 = k45_pack_bf16(Macc[2 * kl].x[2], Macc[2 * kl].x[3]);
                const unsigned bb3 = k45_pack_bf16(Macc[2 * kl + 1].x[2], Macc[2 * kl + 1].x[3]);
                #pragma unroll
                for (int jt = 0; jt < 2; jt++) {
                    ATile A;
                    ld_A(A, sm.Wb + (jt * 16) * D + kb * 16, D / 2, lane);
                    k45_mma_bf16(CA[jt][0], A, bb0, bb1);
                    k45_mma_bf16(CA[jt][1], A, bb2, bb3);
                }
            }
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    float yv[4];
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        int j = jt * 16 + fr + ((l < 2) ? 0 : 8);
                        int col = mh * 16 + hf * 8 + fc + (l & 1);
                        float y = uval[jt][hf][l] - CA[jt][hf].x[l];
                        yv[l] = (j < Cc) ? y : 0.0f;
                        sm.ys[j * K45_YS + col] = (j < Cc) ? __float2bfloat16(y * sm.gk[j]) : __float2bfloat16(0.0f);
                    }
                    // K4 stores Y = __float2half(y); K5 reads rows past Cc as 0 (zero-filled)
                    yh[jt][hf][0] = k45_pack_f16(yv[0], yv[1]);
                    yh[jt][hf][1] = k45_pack_f16(yv[2], yv[3]);
                }
        }
        cp_wait<0>();                // k(c) landed
        __syncthreads();             // S3: ys visible, hand free, Wb free
        if (c + 1 < NC) { stage_w(c + 1); cp_commit(); }
        if (nq == 0) {
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++)
                    sm.hand[mh][jt * 2 + hf][lane] = make_float4(CP[jt][hf].x[0], CP[jt][hf].x[1], CP[jt][hf].x[2], CP[jt][hf].x[3]);
        }
        __syncthreads();             // S4
        if (nq == 1) {
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    float4 v = sm.hand[mh][jt * 2 + hf][lane];
                    CP[jt][hf].x[0] = v.x; CP[jt][hf].x[1] = v.y; CP[jt][hf].x[2] = v.z; CP[jt][hf].x[3] = v.w;
                }
            #pragma unroll
            for (int kl = 0; kl < 4; kl++) {
                const int kb = 4 + kl;
                const unsigned bh0 = k45_pack_f16(Macc[2 * kl].x[0], Macc[2 * kl].x[1]);
                const unsigned bh1 = k45_pack_f16(Macc[2 * kl + 1].x[0], Macc[2 * kl + 1].x[1]);
                const unsigned bh2 = k45_pack_f16(Macc[2 * kl].x[2], Macc[2 * kl].x[3]);
                const unsigned bh3 = k45_pack_f16(Macc[2 * kl + 1].x[2], Macc[2 * kl + 1].x[3]);
                #pragma unroll
                for (int jt = 0; jt < 2; jt++) {
                    ATile Aq;
                    ld_A(Aq, (const __nv_bfloat16*)(sm.qs + (jt * 16) * K45_QS + kb * 16), K45_QS / 2, lane);
                    k45_mma_f16(CP[jt][0], Aq, bh0, bh1);
                    k45_mma_f16(CP[jt][1], Aq, bh2, bh3);
                }
            }
            // o *= exp(G_j) (rows past Cc scaled by 0, as K5), then the P . Y chain over j' blocks 0, 1
            #pragma unroll
            for (int jt = 0; jt < 2; jt++) {
                const float b_lo = sm.bj[jt * 16 + fr], b_hi = sm.bj[jt * 16 + fr + 8];
                #pragma unroll
                for (int hf = 0; hf < 2; hf++) {
                    CP[jt][hf].x[0] *= b_lo; CP[jt][hf].x[1] *= b_lo;
                    CP[jt][hf].x[2] *= b_hi; CP[jt][hf].x[3] *= b_hi;
                }
            }
            #pragma unroll
            for (int k16 = 0; k16 < 2; k16++) {
                // B[k = j'][n = col] of the j' block k16: the transposes of Y's two 8-row halves
                const unsigned y00 = k45_trans(yh[k16][0][0]), y01 = k45_trans(yh[k16][0][1]);
                const unsigned y10 = k45_trans(yh[k16][1][0]), y11 = k45_trans(yh[k16][1][1]);
                #pragma unroll
                for (int jt = 0; jt < 2; jt++) {
                    ATile Ap;
                    ld_A(Ap, (const __nv_bfloat16*)(sm.ps + (jt * 16) * 40 + k16 * 16), 20, lane);
                    k45_mma_f16(CP[jt][0], Ap, y00, y01);
                    k45_mma_f16(CP[jt][1], Ap, y10, y11);
                }
            }
            #pragma unroll
            for (int jt = 0; jt < 2; jt++)
                #pragma unroll
                for (int hf = 0; hf < 2; hf++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        int j = jt * 16 + fr + ((l < 2) ? 0 : 8);
                        int col = mh * 16 + hf * 8 + fc + (l & 1);
                        if (j < Cc) o[((size_t)(t0 + j) * H + h) * D + col0 + col] = scale * CP[jt][hf].x[l];
                    }
        }
        // ---- step B (all warps): S <- bC S + (gk y)^T k
        const float bC = expf(gC);
        #pragma unroll
        for (int t = 0; t < 8; t++)
            #pragma unroll
            for (int l = 0; l < 4; l++) Macc[t].x[l] *= bC;
        #pragma unroll
        for (int k16 = 0; k16 < 2; k16++) {
            ATile A;
            ld_A_trans(A, sm.ys + (k16 * 16) * K45_YS + mh * 16, K45_YS / 2, lane);
            #pragma unroll
            for (int p2 = 0; p2 < 4; p2++) {
                ATile Bt;
                ld_A_trans(Bt, sm.kb + (k16 * 16) * D + nq * 64 + p2 * 16, D / 2, lane);
                BTile Blo, Bhi;
                Blo.x[0] = Bt.x[0]; Blo.x[1] = Bt.x[2];
                Bhi.x[0] = Bt.x[1]; Bhi.x[1] = Bt.x[3];
                mma_bf16(Macc[p2 * 2 + 0], A, Blo);
                mma_bf16(Macc[p2 * 2 + 1], A, Bhi);
            }
        }
        __syncthreads();             // S5 (W(c + 1) is waited on at the next chunk top)
    }
    #pragma unroll
    for (int t = 0; t < 8; t++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            int col = mh * 16 + fr + ((l < 2) ? 0 : 8);
            int i = nq * 64 + t * 8 + fc + (l & 1);
            state_out[((size_t)h * D + col0 + col) * D + i] = Macc[t].x[l];
        }
}

// ------------------------------------------------------------------------------------------
static uint32_t rng_s = 12345u;
static float frand() { rng_s = rng_s * 1664525u + 1013904223u; return ((rng_s >> 8) & 0xffffff) / 16777216.0f - 0.5f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = u + 0x7fff + ((u >> 16) & 1); return (uint16_t)(r >> 16); }

struct Bufs {
    __nv_bfloat16* kb16; float* gcum; float* beta; float* U; __nv_bfloat16* Wb16;
    float* q; float* P; float* st_in;
};
struct Outs { __half* Y; __half* Ss; float* st_out; float* o; };

static size_t nY, nSs, nSt, nO;
static Outs alloc_outs() {
    Outs o;
    CK(cudaMalloc(&o.Y, nY * 2)); CK(cudaMalloc(&o.Ss, nSs * 2));
    CK(cudaMalloc(&o.st_out, nSt * 4)); CK(cudaMalloc(&o.o, nO * 4));
    CK(cudaMemset(o.Y, 0, nY * 2)); CK(cudaMemset(o.Ss, 0, nSs * 2));
    CK(cudaMemset(o.st_out, 0, nSt * 4)); CK(cudaMemset(o.o, 0, nO * 4));
    return o;
}
template <typename Tv>
static size_t bitdiff(const Tv* a, const Tv* b, size_t n) {
    std::vector<Tv> ha(n), hb(n);
    CK(cudaMemcpy(ha.data(), a, n * sizeof(Tv), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b, n * sizeof(Tv), cudaMemcpyDeviceToHost));
    size_t d = 0;
    for (size_t i = 0; i < n; i++) if (memcmp(&ha[i], &hb[i], sizeof(Tv))) d++;
    return d;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const int H = 48, hk = 16, D = 128, C = 32;
    const int NC = (T + C - 1) / C;
    nY = (size_t)NC * H * C * D; nSs = (size_t)NC * H * D * D; nSt = (size_t)H * D * D; nO = (size_t)T * H * D;
    // inputs
    std::vector<float> q((size_t)T * hk * D), k((size_t)T * hk * D);
    for (size_t r = 0; r < (size_t)T * hk; r++) {
        float nq = 0, nk = 0;
        for (int i = 0; i < D; i++) { q[r * D + i] = frand(); k[r * D + i] = frand(); nq += q[r*D+i]*q[r*D+i]; nk += k[r*D+i]*k[r*D+i]; }
        for (int i = 0; i < D; i++) { q[r * D + i] /= sqrtf(nq); k[r * D + i] /= sqrtf(nk); }
    }
    std::vector<uint16_t> kb(k.size());
    for (size_t i = 0; i < k.size(); i++) kb[i] = f2bf(k[i]);
    std::vector<float> gc((size_t)T * H), be((size_t)T * H);
    for (int h = 0; h < H; h++)
        for (int c = 0; c < NC; c++) {
            float acc = 0;
            for (int j = 0; j < C && c * C + j < T; j++) {
                int t = c * C + j;
                acc += -0.02f - fabsf(frand()) * 0.5f;
                gc[(size_t)t * H + h] = acc;
                be[(size_t)t * H + h] = 0.3f + fabsf(frand()) * 0.6f;
            }
        }
    std::vector<float> Uh(nY), Wf(nY);
    for (size_t i = 0; i < nY; i++) { Uh[i] = frand(); Wf[i] = frand() * 0.25f; }
    std::vector<uint16_t> Wb(nY);
    for (size_t i = 0; i < nY; i++) Wb[i] = f2bf(Wf[i]);
    std::vector<float> Ph((size_t)NC * H * C * C, 0.f);
    for (int c = 0; c < NC; c++) for (int h = 0; h < H; h++) for (int j = 0; j < C; j++) for (int i = 0; i <= j; i++)
        Ph[(((size_t)c * H + h) * C + j) * C + i] = frand() * 0.5f;
    std::vector<float> st(nSt);
    for (auto& x : st) x = frand();

    Bufs b;
    CK(cudaMalloc(&b.kb16, kb.size() * 2)); CK(cudaMemcpy(b.kb16, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.gcum, gc.size() * 4)); CK(cudaMemcpy(b.gcum, gc.data(), gc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.beta, be.size() * 4)); CK(cudaMemcpy(b.beta, be.data(), be.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.U, nY * 4)); CK(cudaMemcpy(b.U, Uh.data(), nY * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.Wb16, nY * 2)); CK(cudaMemcpy(b.Wb16, Wb.data(), nY * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.q, q.size() * 4)); CK(cudaMemcpy(b.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.P, Ph.size() * 4)); CK(cudaMemcpy(b.P, Ph.data(), Ph.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.st_in, nSt * 4)); CK(cudaMemcpy(b.st_in, st.data(), nSt * 4, cudaMemcpyHostToDevice));
    const float scale = 1.0f / sqrtf((float)D);

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto launch) {
        std::vector<float> ms;
        launch(); CK(cudaDeviceSynchronize());
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(e0)); launch(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    auto k5 = [&](const Outs& o, int hfast) {
        dim3 g = hfast ? dim3(H, NC, 1) : dim3(NC, H, 1);
        gdn_chunk_output_mma<<<g, 256>>>(b.q, b.gcum, b.P, o.Y, o.Ss, o.o, H, T, C, scale, hk, hfast);
        CK(cudaGetLastError());
    };
    // reference: shipped K4 + K5 (chunks fastest)
    Outs ref = alloc_outs();
    auto k4v0 = [&](const Outs& o) {
        gdn_chunk_state_mma<<<dim3(H, 4), 256>>>(b.kb16, b.gcum, b.beta, b.U, b.Wb16, o.Y, o.Ss, b.st_in, o.st_out, H, T, C, hk);
        CK(cudaGetLastError());
    };
    float t_v0 = timeit([&] { k4v0(ref); });
    float t_k5c = timeit([&] { k5(ref, 0); });
    Outs o5h = alloc_outs();
    CK(cudaMemcpy(o5h.Y, ref.Y, nY * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(o5h.Ss, ref.Ss, nSs * 2, cudaMemcpyDeviceToDevice));
    float t_k5h = timeit([&] { k5(o5h, 1); });
    printf("T=%d H=%d hk=%d NC=%d reps=%d (median ms)\n", T, H, hk, NC, reps);
    printf("k4 v0       %8.3f ms\n", t_v0);
    printf("k5 cfast    %8.3f ms\n", t_k5c);
    printf("k5 hfast    %8.3f ms  o bitdiff=%zu\n", t_k5h, bitdiff(ref.o, o5h.o, nO));

    auto run_cand = [&](const char* name, int cols, bool store, auto launch) {
        Outs o = alloc_outs();
        float t = timeit([&] { launch(o); });
        size_t dy = 0, ds = 0, dst = bitdiff(ref.st_out, o.st_out, nSt), dO = 0;
        if (store) {
            dy = bitdiff(ref.Y, o.Y, nY); ds = bitdiff(ref.Ss, o.Ss, nSs);
            k5(o, 0); CK(cudaDeviceSynchronize());
            dO = bitdiff(ref.o, o.o, nO);
            printf("k4 %-8s %8.3f ms  (%.2fx v0)  Y=%zu Ssnap=%zu state=%zu o=%zu %s\n", name, t, t_v0 / t, dy, ds, dst, dO,
                   (dy | ds | dst | dO) ? "MISMATCH" : "EXACT");
        } else {
            printf("k4 %-8s %8.3f ms  (%.2fx v0)  state=%zu (diagnostic, no Y/Ssnap)\n", name, t, t_v0 / t, dst);
        }
        CK(cudaFree(o.Y)); CK(cudaFree(o.Ss)); CK(cudaFree(o.st_out)); CK(cudaFree(o.o));
    };
#define CAND(NAME, COLS, PF, ST) run_cand(NAME, COLS, ST, [&](const Outs& o) { \
        k4_tpl<COLS, PF, ST><<<dim3(H, 128 / COLS), 256>>>(b.kb16, b.gcum, b.beta, b.U, b.Wb16, o.Y, o.Ss, b.st_in, o.st_out, H, T, C, hk); \
        CK(cudaGetLastError()); })
    CAND("c32", 32, false, true);
    CAND("c32pf", 32, true, true);
    CAND("c64", 64, false, true);
    CAND("c64pf", 64, true, true);
    CAND("c32ns", 32, false, false);
    CAND("c32pfns", 32, true, false);
    CAND("c64pfns", 64, true, false);
    {
        Outs o = alloc_outs();
        const size_t smem = sizeof(K45Smem);
        CK(cudaFuncSetAttribute(k45_c64, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        auto launch = [&] {
            k45_c64<<<dim3(H, 2), 256, smem>>>(b.kb16, b.gcum, b.beta, b.U, b.Wb16, b.q, b.P, o.o,
                                                b.st_in, o.st_out, H, T, C, scale, hk);
            CK(cudaGetLastError());
        };
        float t = timeit(launch);
        size_t dO = bitdiff(ref.o, o.o, nO), dst = bitdiff(ref.st_out, o.st_out, nSt);
        printf("k45 c64     %8.3f ms  (K4+K5 shipped %.3f, K4 + K5 hfast %.3f)  smem=%zu B  state=%zu o=%zu %s\n",
               t, t_v0 + t_k5c, t_v0 + t_k5h, smem, dst, dO, (dO | dst) ? "MISMATCH" : "EXACT");
        CK(cudaFree(o.Y)); CK(cudaFree(o.Ss)); CK(cudaFree(o.st_out)); CK(cudaFree(o.o));
    }
    printf("PROBE-GDN-DONE\n");
    return 0;
}
