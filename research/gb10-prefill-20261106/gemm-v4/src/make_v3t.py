"""Make the lever-6 P4 probe TU: the tree's mmq_nvfp4_w4a8.cu plus the half-height last wave.

When the probe global f8f4v3_tail is set and a launch's tile count leaves a partial last wave whose
tiles fit one wave at half height (2 * rem <= SMs), f8f4v3_launch runs the first ntiles - rem tiles
as today and then the last rem tiles (in the raster's order) as 2 * rem tiles of 128 x 128
(mul_mat_f8f4v3t: 4 x 2 warps, each warp the shipped 32 x 64 span), from their own tensor maps. Every
tile keeps its own full k order and runs the shipped stage compute and epilogue, so the output is
bit-identical to v3's.
usage: make_v3t.py IN OUT"""
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub1(a, b):
    global s
    if s.count(a) != 1:
        sys.exit(f"anchor count {s.count(a)}: {a[:70]!r}")
    s = s.replace(a, b)


kernel = r'''
// ---- lever 6 P4 probe: the half-height last wave ----
static int f8f4v3_tail = 0;   // probe switch, set by the driver

// Half tile h of the tail: parent tile Lb = lb0 + h / 2 in the parent raster (PBM-row tiles, nty
// parent row tiles, groups of G), rows PBM * it + (h % 2) * BM.
template <int BM, int BN, int WM, int WN, int KS, int STAGES, int EPI, int PBM>
__launch_bounds__(WM * WN * 32, 1)
static __global__ void mul_mat_f8f4v3t(
        const __grid_constant__ F8F4V3Maps maps, float * __restrict__ dst,
        const int out_f, const int K, const int n_tokens, const int nty, const float out_scale, const int G,
        uint8_t * __restrict__ act_out, const float out_scale2, const float * __restrict__ res, const int lb0) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    constexpr int NW  = WM * WN;
    constexpr int WRS = BM / WM;
    constexpr int WCS = BN / WN;
    constexpr int TM  = WRS / 16;
    constexpr int TN  = WCS / 8;
    constexpr int KSTEPS = KS / 32;
    static_assert(WRS == F8F4V2_WR, "the image scale permutation assumes 32-row warp spans");
    static_assert(PBM % BM == 0, "half tiles split a parent tile");
    extern __shared__ __align__(1024) uint8_t f8f4v3_smem[];
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(f8f4v3_smem);
    const unsigned bars = smem0 + STAGES * LY::STAGE;

    const int ntx = (n_tokens + BN - 1) / BN;
    const int h = blockIdx.x;
    const int Lb = lb0 + h / (PBM / BM);
    const int group_size = G * ntx;
    const int grp = Lb / group_size;
    const int first = grp * G;
    const int gsz = min(nty - first, G);
    const int local = Lb - grp * group_size;
    const int it = first + local % gsz;
    const int jt = local / gsz;
    const int row0 = it * PBM + (h % (PBM / BM)) * BM;
    const int tok0 = jt * BN;
    const int nk = K / KS;

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

    auto issue = [&](const int slot, const int kt) {
        const unsigned sA  = smem0 + slot * LY::STAGE;
        const unsigned sB  = sA + LY::A_BYTES;
        const unsigned sSA = sB + LY::B_BYTES;
        const unsigned sSB = sSA + LY::SA_BYTES;
        const unsigned full = bars + 8u * slot;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(full), "r"((unsigned) LY::STAGE) : "memory");
        f8f4v3_tma2d(sA,  &maps.a,  kt * KS, row0, full);
        f8f4v3_tma2d(sB,  &maps.b,  kt * KS, tok0, full);
        f8f4v3_tma2d(sSA, &maps.sa, row0, kt * KSTEPS, full);
        f8f4v3_tma2d(sSB, &maps.sb, tok0, kt * KSTEPS, full);
    };
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) { if (s < nk) { issue(s, s); } }
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
        f8f4v3_wait(bars + 8u * slot, (unsigned) (kt / STAGES) & 1u);
        f8f4v2_compute_stage<BM, BN, WM, WN, KS, true>(
            acc, smem0 + slot * LY::STAGE,
            (const float *) (f8f4v3_smem + slot * LY::STAGE + LY::A_BYTES + LY::B_BYTES),
            ar, ac, br, bc, wm, wn, g, tig);
        __syncwarp();
        if (lane == 0) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(bars + 8u * (STAGES + slot)) : "memory");
        }
        if (tid == 0) {
            const int nx = kt + STAGES;
            if (nx < nk) {
                f8f4v3_wait(bars + 8u * (STAGES + slot), (unsigned) ((nx / STAGES) - 1) & 1u);
                issue(slot, nx);
            }
        }
        __syncwarp();
    }

    f8f4v2_epilogue<BM, BN, WM, WN, EPI>(acc, f8f4v3_smem, row0, tok0, wm, wn, g, tig, out_f, n_tokens,
                                          dst, out_scale, act_out, out_scale2, res);
}
'''

sub1("// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.",
     kernel + "\n// The driver's cuTensorMapEncodeTiled, resolved once through the runtime.")

# the launch: the tail split for EPI 0 / 2 (the only users of the 640-tile grids); the tail's maps come
# from the same tensors with 128-row boxes, encoded at the call site (see the entry below)
launch_old = '''    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
launch_new = '''    if constexpr (EPI != 1 && BM == 256) {
        int nsm = 0;
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        const int ntiles = nty * ntx;
        const int rem = ntiles % nsm;
        if (f8f4v3_tail && f8f4v3_tail_maps_ok && rem > 0 && 2 * rem <= nsm && ntiles > rem) {
            kern<<<(unsigned) (ntiles - rem), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g,
                                                                      act_out, out_scale2, res);
            using LT = f8f4v2_layout<128, BN, KS>;
            constexpr size_t tsmem = (size_t) STAGES * LT::STAGE + 16 * STAGES;
            auto tk = mul_mat_f8f4v3t<128, BN, 4, WN, KS, STAGES, EPI, BM>;
            e = cudaFuncSetAttribute(tk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) tsmem);
            if (e != cudaSuccess) { return e; }
            tk<<<(unsigned) (2 * rem), 4 * WN * 32, tsmem, st>>>(f8f4v3_tail_maps, y, out_f, K, n_tokens, nty, out_scale,
                                                                g, act_out, out_scale2, res, ntiles - rem);
            return cudaGetLastError();
        }
    }
    kern<<<(unsigned) (nty * ntx), WM * WN * 32, smem, st>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g, act_out,
                                                             out_scale2, res);
    return cudaGetLastError();'''
sub1(launch_old, launch_new)

# the tail's maps: the plain/residual entry encodes them next to the main maps when the switch is set
sub1('''struct F8F4V3Maps { CUtensorMap a, b, sa, sb; };''',
     '''struct F8F4V3Maps { CUtensorMap a, b, sa, sb; };
static F8F4V3Maps f8f4v3_tail_maps;   // probe: the 128-row-box maps of the current call
static bool f8f4v3_tail_maps_ok = false;''')
entry_old = '''    const bool tma = v3 && f8f4v3_maps<256, 128, 64>(&mp, vals, scales, act, act_sc, Mp, in_f, n_tokens, Tp);'''
if s.count(entry_old) != 2:
    sys.exit("expected two map call sites")
first = s.index(entry_old)
s = s[:first] + entry_old + '''
    f8f4v3_tail_maps_ok = v3 && f8f4v3_tail && f8f4v3_maps<128, 128, 64>(&f8f4v3_tail_maps, vals, scales, act, act_sc, Mp, in_f, n_tokens, Tp);''' + s[first + len(entry_old):]
open(out, "w").write(s)
print("ok")
