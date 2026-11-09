"""Make the lever-6 P3 probe TU: the tree's mmq_nvfp4_w4a8.cu plus a v3 mainloop with per-group slot
release (mul_mat_f8f4v3g), taken by f8f4v3_launch when the probe global f8f4v3_group is set.

The 256 x 128 tile's A rows split into 8 groups of 32 (the two warps wm = g read group g) and its B
tokens into 2 groups of 64 (the eight warps wn = h read group h). Per ring slot each group has its
own full and empty mbarrier. Warp (g, 0) refills A group g (its 32 x 64 image box and its scale box)
once both of its warps left the slot; warp (0, h) refills B group h once its eight warps did. So no
refill waits for the slowest of 16 warps, and the A refills spread over 8 issuing warps.

The scale planes land per group ([group][k32 step][rows]); the stage compute reads them at the
matching offsets and is otherwise f8f4v2_compute_stage, so every output is bit-identical to v3.
The tensor maps take the group boxes (64 x 32, 64 x 64, 32 x 2, 64 x 2).
usage: make_v3g.py IN OUT"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub1(a, b):
    global s
    if s.count(a) != 1:
        sys.exit(f"anchor count {s.count(a)}: {a[:70]!r}")
    s = s.replace(a, b)


compute = r'''
// ---- lever 6 P3 probe: per-group slot release ----
static int f8f4v3_group = 0;   // probe switch, set by the driver

// f8f4v2_compute_stage with the scale planes in per-group order: A scales [wm][ks][32] (A group wm,
// 32 rows), B scales [wn][ks][64] (B group wn, 64 tokens). The values and every operation are the
// shipped stage's.
template <int BM, int BN, int WM, int WN, int KS>
static __device__ __forceinline__ void f8f4v3g_compute_stage(
        float (&acc)[BM / WM / 16][BN / WN / 8][4], const unsigned sA,
        const int ar, const int ac, const int br, const int bc,
        const int wm, const int wn, const int g, const int tig) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    constexpr int WRS = BM / WM;
    constexpr int WCS = BN / WN;
    constexpr int TM  = WRS / 16;
    constexpr int TN  = WCS / 8;
    constexpr int KSTEPS = KS / 32;
    const unsigned sB = sA + LY::A_BYTES;
    const unsigned uSA = sB + LY::B_BYTES;
    const unsigned uSB = uSA + LY::SA_BYTES;
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        int a[TM][4];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            f8f4v2_ldsm_x4_s(a[m], sA + LY::swz(ar + m * 16, 2 * ks + ac));
        }
        const float4 dav = f8f4v2_lds4(uSA + 4u * (unsigned) ((wm * KSTEPS + ks) * WRS + g * 4));
        const float da[2][2] = {{dav.x, dav.y}, {dav.z, dav.w}};
#pragma unroll
        for (int pp = 0; pp < TN / 2; ++pp) {
            int b[4];
            f8f4v2_ldsm_x4_s(b, sB + LY::swz(br + pp * 16, 2 * ks + bc));
            const float4 dbv = f8f4v2_lds4(uSB + 4u * (unsigned) ((wn * KSTEPS + ks) * WCS + (pp * 4 + tig) * 4));
            const float db[2][2] = {{dbv.x, dbv.y}, {dbv.z, dbv.w}};
#pragma unroll
            for (int q = 0; q < 2; ++q) {
#pragma unroll
                for (int m = 0; m < TM; ++m) {
                    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    memra_mma_f8f4(C, a[m], b[2 * q + 0], b[2 * q + 1]);
                    float * s = acc[m][2 * pp + q];
                    s[0] += da[m][0] * db[q][0] * C[0];
                    s[1] += da[m][0] * db[q][1] * C[1];
                    s[2] += da[m][1] * db[q][0] * C[2];
                    s[3] += da[m][1] * db[q][1] * C[3];
                }
            }
        }
    }
}
'''

kernel = r'''
template <int BM, int BN, int WM, int WN, int KS, int STAGES, int EPI>
__launch_bounds__(WM * WN * 32, 1)
static __global__ void mul_mat_f8f4v3g(
        const __grid_constant__ F8F4V3Maps maps, float * __restrict__ dst,
        const int out_f, const int K, const int n_tokens, const int nty, const float out_scale, const int G,
        uint8_t * __restrict__ act_out, const float out_scale2, const float * __restrict__ res) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    constexpr int WRS = BM / WM;          // 32 rows per A group
    constexpr int WCS = BN / WN;          // 64 tokens per B group
    constexpr int TM  = WRS / 16;
    constexpr int TN  = WCS / 8;
    constexpr int KSTEPS = KS / 32;
    constexpr int NB = WM + WN;           // groups per slot: A groups 0..WM-1, then B groups
    constexpr unsigned A_BOX  = WRS * KS;              // image bytes per A group
    constexpr unsigned B_BOX  = WCS * KS;              // activation bytes per B group
    constexpr unsigned SA_BOX = KSTEPS * WRS * 4;      // A scale bytes per group
    constexpr unsigned SB_BOX = KSTEPS * WCS * 4;      // B scale bytes per group
    extern __shared__ __align__(1024) uint8_t f8f4v3_smem[];
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(f8f4v3_smem);
    // full[slot][NB] then empty[slot][NB]
    const unsigned bars = smem0 + STAGES * LY::STAGE;
    auto fullb  = [&](const int slot, const int grp) { return bars + 8u * (unsigned) (slot * NB + grp); };
    auto emptyb = [&](const int slot, const int grp) { return bars + 8u * (unsigned) ((STAGES + slot) * NB + grp); };

    const int ntx = (n_tokens + BN - 1) / BN;
    const int Lb = blockIdx.x;
    const int group_size = G * ntx;
    const int grp0 = Lb / group_size;
    const int first = grp0 * G;
    const int gsz = min(nty - first, G);
    const int local = Lb - grp0 * group_size;
    const int it = first + local % gsz;
    const int jt = local / gsz;
    const int row0 = it * BM;
    const int tok0 = jt * BN;
    const int nk = K / KS;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, tig = lane & 3;
    const bool issA = wn == 0 && lane == 0;    // warp (wm, 0) refills A group wm
    const bool issB = wm == 0 && lane == 0;    // warp (0, wn) refills B group wn

    if (tid == 0) {
        for (int sl = 0; sl < STAGES; ++sl) {
            for (int q = 0; q < NB; ++q) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(fullb(sl, q)), "r"(1));
                asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(emptyb(sl, q)), "r"(q < WM ? WN : WM));
            }
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    auto issue_a = [&](const int slot, const int kt) {
        const unsigned sA  = smem0 + slot * LY::STAGE;
        const unsigned sSA = sA + LY::A_BYTES + LY::B_BYTES;
        const unsigned full = fullb(slot, wm);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full), "r"(A_BOX + SA_BOX) : "memory");
        f8f4v3_tma2d(sA + (unsigned) wm * A_BOX, &maps.a, kt * KS, row0 + wm * WRS, full);
        f8f4v3_tma2d(sSA + (unsigned) wm * SA_BOX, &maps.sa, row0 + wm * WRS, kt * KSTEPS, full);
    };
    auto issue_b = [&](const int slot, const int kt) {
        const unsigned sB  = smem0 + slot * LY::STAGE + LY::A_BYTES;
        const unsigned sSB = sB + LY::B_BYTES + LY::SA_BYTES;
        const unsigned full = fullb(slot, WM + wn);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full), "r"(B_BOX + SB_BOX) : "memory");
        f8f4v3_tma2d(sB + (unsigned) wn * B_BOX, &maps.b, kt * KS, tok0 + wn * WCS, full);
        f8f4v3_tma2d(sSB + (unsigned) wn * SB_BOX, &maps.sb, tok0 + wn * WCS, kt * KSTEPS, full);
    };
    for (int sl = 0; sl < STAGES; ++sl) {
        if (sl < nk) {
            if (issA) { issue_a(sl, sl); }
            if (issB) { issue_b(sl, sl); }
        }
    }
    __syncwarp();

    float acc[TM][TN][4];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int p = 0; p < TN; ++p) {
#pragma unroll
            for (int l = 0; l < 4; ++l) { acc[m][p][l] = 0.0f; }
        }
    }
    const int ar = wm * WRS + (lane & 15), ac = lane >> 4;
    const int br = wn * WCS + (lane >> 4) * 8 + (lane & 7), bc = (lane >> 3) & 1;

    for (int kt = 0; kt < nk; ++kt) {
        const int slot = kt % STAGES;
        const unsigned par = (unsigned) (kt / STAGES) & 1u;
        f8f4v3_wait(fullb(slot, wm), par);
        f8f4v3_wait(fullb(slot, WM + wn), par);
        f8f4v3g_compute_stage<BM, BN, WM, WN, KS>(acc, smem0 + slot * LY::STAGE, ar, ac, br, bc, wm, wn, g, tig);
        __syncwarp();
        if (lane == 0) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(emptyb(slot, wm)) : "memory");
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(emptyb(slot, WM + wn)) : "memory");
        }
        const int nx = kt + STAGES;
        if (nx < nk) {
            const unsigned epar = (unsigned) ((nx / STAGES) - 1) & 1u;
            if (issA) { f8f4v3_wait(emptyb(slot, wm), epar); issue_a(slot, nx); }
            if (issB) { f8f4v3_wait(emptyb(slot, WM + wn), epar); issue_b(slot, nx); }
        }
        __syncwarp();
    }

    f8f4v2_epilogue<BM, BN, WM, WN, EPI>(acc, f8f4v3_smem, row0, tok0, wm, wn, g, tig, out_f, n_tokens,
                                          dst, out_scale, act_out, out_scale2, res);
}
'''

sub1("// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.",
     compute + kernel + "\n// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.")

# group boxes in the maps when the switch is set
sub1('''    return f8f4v3_map2d(&mp->a, CU_TENSOR_MAP_DATA_TYPE_UINT8, vals, (uint64_t) K, (uint64_t) Mp, (uint64_t) K,
                        KS, BM, CU_TENSOR_MAP_SWIZZLE_64B) &&
           f8f4v3_map2d(&mp->b, CU_TENSOR_MAP_DATA_TYPE_UINT8, act, (uint64_t) K, (uint64_t) n_tokens,
                        (uint64_t) K, KS, BN, CU_TENSOR_MAP_SWIZZLE_64B) &&
           f8f4v3_map2d(&mp->sa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, scales, (uint64_t) Mp, (uint64_t) (K / 32),
                        (uint64_t) Mp * 4, BM, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE) &&
           f8f4v3_map2d(&mp->sb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, act_sc, (uint64_t) Tp, (uint64_t) (K / 32),
                        (uint64_t) Tp * 4, BN, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE);''',
'''    const uint32_t bm = f8f4v3_group ? 32 : BM, bn = f8f4v3_group ? 64 : BN;
    return f8f4v3_map2d(&mp->a, CU_TENSOR_MAP_DATA_TYPE_UINT8, vals, (uint64_t) K, (uint64_t) Mp, (uint64_t) K,
                        KS, bm, CU_TENSOR_MAP_SWIZZLE_64B) &&
           f8f4v3_map2d(&mp->b, CU_TENSOR_MAP_DATA_TYPE_UINT8, act, (uint64_t) K, (uint64_t) n_tokens,
                        (uint64_t) K, KS, bn, CU_TENSOR_MAP_SWIZZLE_64B) &&
           f8f4v3_map2d(&mp->sa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, scales, (uint64_t) Mp, (uint64_t) (K / 32),
                        (uint64_t) Mp * 4, bm, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE) &&
           f8f4v3_map2d(&mp->sb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, act_sc, (uint64_t) Tp, (uint64_t) (K / 32),
                        (uint64_t) Tp * 4, bn, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE);''')

launch_old = '''    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
launch_new = '''    if (f8f4v3_group) {
        constexpr size_t gsmem = (size_t) STAGES * LY::STAGE + 16 * STAGES * (WM + WN);
        auto gk = mul_mat_f8f4v3g<BM, BN, WM, WN, KS, STAGES, EPI>;
        e = cudaFuncSetAttribute(gk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) gsmem);
        if (e != cudaSuccess) { return e; }
        gk<<<(unsigned) (nty * ntx), WM * WN * 32, gsmem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                                out_scale2, res);
        return cudaGetLastError();
    }
    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
sub1(launch_old, launch_new)
open(out, "w").write(s)
print("ok")
