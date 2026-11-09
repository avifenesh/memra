"""Make the lever-6 P1 probe TU: the tree's mmq_nvfp4_w4a8.cu plus a persistent v3 mainloop
(mul_mat_f8f4v3p: one CTA per SM walks its tiles, the ring runs on across tile boundaries, so the
next tile's first stages load under the current tile's epilogue; the gate/up epilogue gets its own
smem window). f8f4v3_launch takes it when the probe global f8f4v3_persist is set. Every tile runs
the shipped stage compute and epilogue, so the output must equal v3's byte for byte.
usage: make_v3p.py IN OUT"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub1(a, b):
    global s
    if s.count(a) != 1:
        sys.exit(f"anchor count {s.count(a)}: {a[:70]!r}")
    s = s.replace(a, b)


kernel = r'''
// ---- lever 6 P1 probe: the persistent v3 mainloop ----
static int f8f4v3_persist = 0;   // probe switch, set by the driver

template <int BM, int BN, int WM, int WN, int KS, int STAGES, int EPI>
__launch_bounds__(WM * WN * 32, 1)
static __global__ void mul_mat_f8f4v3p(
        const __grid_constant__ F8F4V3Maps maps, float * __restrict__ dst,
        const int out_f, const int K, const int n_tokens, const int nty, const float out_scale, const int G,
        uint8_t * __restrict__ act_out, const float out_scale2, const float * __restrict__ res, const int ntiles) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    constexpr int NW  = WM * WN;
    constexpr int WRS = BM / WM;
    constexpr int WCS = BN / WN;
    constexpr int TM  = WRS / 16;
    constexpr int TN  = WCS / 8;
    constexpr int KSTEPS = KS / 32;
    extern __shared__ __align__(1024) uint8_t f8f4v3_smem[];
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(f8f4v3_smem);
    const unsigned bars = smem0 + STAGES * LY::STAGE;
    uint8_t * epi_smem = f8f4v3_smem + STAGES * LY::STAGE + 16 * STAGES + 1008;   // 1 KB past the bars

    const int ntx = (n_tokens + BN - 1) / BN;
    const int nk = K / KS;
    const int nmy = (ntiles - (int) blockIdx.x + (int) gridDim.x - 1) / (int) gridDim.x;
    const int total = nmy * nk;
    auto tile_of = [&](const int t, int & r0, int & t0) {
        const int Lb = (int) blockIdx.x + t * (int) gridDim.x;
        const int group_size = G * ntx;
        const int grp = Lb / group_size;
        const int first = grp * G;
        const int gsz = min(nty - first, G);
        const int local = Lb - grp * group_size;
        r0 = (first + local % gsz) * BM;
        t0 = (local / gsz) * BN;
    };

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, tig = lane & 3;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bars + 8u * s), "r"(1));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bars + 8u * (STAGES + s)), "r"(NW));
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // global step gs -> tile gs / nk, k step gs % nk
    auto issue = [&](const int slot, const int gs) {
        int r0, t0;
        tile_of(gs / nk, r0, t0);
        const int kt = gs % nk;
        const unsigned sA  = smem0 + slot * LY::STAGE;
        const unsigned sB  = sA + LY::A_BYTES;
        const unsigned sSA = sB + LY::B_BYTES;
        const unsigned sSB = sSA + LY::SA_BYTES;
        const unsigned full = bars + 8u * slot;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full), "r"((unsigned) LY::STAGE) : "memory");
        f8f4v3_tma2d(sA,  &maps.a,  kt * KS, r0, full);
        f8f4v3_tma2d(sB,  &maps.b,  kt * KS, t0, full);
        f8f4v3_tma2d(sSA, &maps.sa, r0, kt * KSTEPS, full);
        f8f4v3_tma2d(sSB, &maps.sb, t0, kt * KSTEPS, full);
    };
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { if (s < total) { issue(s, s); } }
    }
    __syncwarp();

    const int ar = wm * WRS + (lane & 15), ac = lane >> 4;
    const int br = wn * WCS + (lane >> 4) * 8 + (lane & 7), bc = (lane >> 3) & 1;

    for (int t = 0; t < nmy; ++t) {
        int row0, tok0;
        tile_of(t, row0, tok0);
        float acc[TM][TN][4];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int p = 0; p < TN; ++p) {
#pragma unroll
                for (int l = 0; l < 4; ++l) { acc[m][p][l] = 0.0f; }
            }
        }
        for (int kt = 0; kt < nk; ++kt) {
            const int gs = t * nk + kt;
            const int slot = gs % STAGES;
            f8f4v3_wait(bars + 8u * slot, (unsigned) (gs / STAGES) & 1u);
            f8f4v2_compute_stage<BM, BN, WM, WN, KS, true>(
                acc, smem0 + slot * LY::STAGE,
                (const float *) (f8f4v3_smem + slot * LY::STAGE + LY::A_BYTES + LY::B_BYTES),
                ar, ac, br, bc, wm, wn, g, tig);
            __syncwarp();
            if (lane == 0) {
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + 8u * (STAGES + slot)) : "memory");
            }
            if (tid == 0) {
                const int nx = gs + STAGES;
                if (nx < total) {
                    f8f4v3_wait(bars + 8u * (STAGES + slot), (unsigned) ((nx / STAGES) - 1) & 1u);
                    issue(slot, nx);
                }
            }
            __syncwarp();
        }
        f8f4v2_epilogue<BM, BN, WM, WN, EPI>(acc, epi_smem, row0, tok0, wm, wn, g, tig, out_f, n_tokens,
                                              dst, out_scale, act_out, out_scale2, res);
        if constexpr (EPI == 1) { __syncthreads(); }   // the epilogue window is reused by the next tile
    }
}
'''

sub1("// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.",
     kernel + "\n// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.")

launch_old = '''    const int nty = (out_f + BM - 1) / BM;
    const int ntx = (n_tokens + BN - 1) / BN;
    const int g = G < 1 ? 1 : (G > nty ? nty : G);
    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
launch_new = '''    const int nty = (out_f + BM - 1) / BM;
    const int ntx = (n_tokens + BN - 1) / BN;
    const int g = G < 1 ? 1 : (G > nty ? nty : G);
    if (f8f4v3_persist) {
        constexpr size_t psmem = (size_t) STAGES * LY::STAGE + 16 * STAGES + 1008 + (size_t) WM * BN * sizeof(float);
        auto pk = mul_mat_f8f4v3p<BM, BN, WM, WN, KS, STAGES, EPI>;
        e = cudaFuncSetAttribute(pk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) psmem);
        if (e != cudaSuccess) { return e; }
        int nsm = 0;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int ntiles = nty * ntx;
        const int grid = ntiles < nsm ? ntiles : nsm;
        pk<<<(unsigned) grid, WM * WN * 32, psmem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                         out_scale2, res, ntiles);
        return cudaGetLastError();
    }
    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
sub1(launch_old, launch_new)
open(out, "w").write(s)
print("ok")
