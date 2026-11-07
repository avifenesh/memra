// F8F4 tile v3 probe: the shipped v2 tile's numeric program and smem layout, fed by TMA and
// synchronized by mbarriers instead of per-thread cp.async and a CTA-wide __syncthreads per stage.
//
// Per output element the operation sequence is the shipped one: fresh-accumulator block-scale MMA
// per k32 block in ascending k, then s += (dA * dB) * C. The smem bytes are the shipped ones: TMA
// SWIZZLE_64B on 64-byte rows puts chunk c of row r at c ^ ((r >> 1) & 3), which is
// f8f4v2_layout<.., 64>::swz, so the ldmatrix addresses do not change. Tokens past n_tokens are
// zero-filled by TMA instead of repeating the last token; their outputs are dropped either way.
//
// Issue schemes (PROD): 0 one elected consumer thread (warp 0 lane 0) issues stage kt + S - 1 at the
// top of iteration kt after that slot's empty barrier; 1 the same thread issues stage kt + S at the end
// of iteration kt (issue-end); 2 a fifth warpgroup is the producer (setmaxnreg.dec 24 there, .inc 120
// for the 16 consumer warps); 3 / 4 after compute kt, stage kt + S - 1 into the slot left one stage
// earlier, issued by warp 0 / by warp kt % 16; 5 the last warp to leave a slot refills it (an smem
// counter, no empty wait). HINT: mbarrier waits carry a suspend-time hint (the warp sleeps until the
// phase flips) instead of spinning on try_wait. SHL: scale loads use 32-bit shared-window addresses.
// Arms (argv[4], a comma list of ids; default all): see the ARMS table in main().
// build (this dir holds mmq_probe.cu and mmq_common.cuh):
//   nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr tma_probe.cu -lcuda
// usage: tma_probe T REPS SHAPE(-1 all) ARMS
#include "mmq_probe.cu"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
extern "C" void memra_mmq_nvfp4_f8f4_quantize_act(const float *, void *, int, int, int64_t, cudaStream_t) {}

static __device__ __forceinline__ void v3_bar_init(unsigned b, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(count));
}
static __device__ __forceinline__ void v3_expect_tx(unsigned b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
static __device__ __forceinline__ void v3_arrive(unsigned b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
}
template <bool HINT>
static __device__ __forceinline__ void v3_wait(unsigned b, unsigned parity) {
    if constexpr (HINT) {
        asm volatile("{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n @!p bra W%=;\n}\n"
                     :: "r"(b), "r"(parity), "r"(0x989680u) : "memory");
    } else {
        asm volatile("{\n .reg .pred p;\n W%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W%=;\n}\n"
                     :: "r"(b), "r"(parity) : "memory");
    }
}

static __device__ __forceinline__ void v3_tma2d(unsigned dst, const CUtensorMap * map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                 :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

struct V3Maps { CUtensorMap a, b, sa, sb; };

static __device__ __forceinline__ float4 v3_lds4(const unsigned a) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a));
    return v;
}

template <int BM, int BN, int WM, int WN, int KS, int STAGES, int PROD, bool SHL = false, bool HINT = true>
__launch_bounds__(WM * WN * 32 + (PROD == 2 ? 128 : 0), 1)
static __global__ void mul_mat_f8f4v3(const __grid_constant__ V3Maps maps, float * __restrict__ dst,
        const int out_f, const int K, const int n_tokens, const int nty, const float out_scale, const int G) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    constexpr int NC  = WM * WN;          // consumer warps
    constexpr int WRS = BM / WM;
    constexpr int WCS = BN / WN;
    constexpr int TM  = WRS / 16;
    constexpr int TN  = WCS / 8;
    constexpr int KSTEPS = KS / 32;
    static_assert(KS == 64 || KS == 32, "SWIZZLE_64B / SWIZZLE_32B reproduce f8f4v2_layout::swz for 64- / 32-byte rows");
    static_assert(WRS == F8F4V2_WR && TN % 2 == 0, "the shipped warp geometry");

    extern __shared__ __align__(1024) uint8_t v3_smem[];
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(v3_smem);
    const unsigned bars = smem0 + STAGES * LY::STAGE;           // full[S] then empty[S], 8 B each
    int * cnt = (int *) (v3_smem + STAGES * LY::STAGE + 16 * STAGES);   // PROD 5: warps done per slot
    auto full  = [&](int s) { return bars + 8u * s; };
    auto empty = [&](int s) { return bars + 8u * (STAGES + s); };

    const int ntx = (n_tokens + BN - 1) / BN;
    const int Lb = blockIdx.x;
    const int group_size = G * ntx;
    const int grp = Lb / group_size;
    const int first = grp * G;
    const int gsz = min(nty - first, G);
    const int local = Lb - grp * group_size;
    const int it = first + local % gsz;
    const int jt = local / gsz;
    const int row0 = it * BM, tok0 = jt * BN;
    const int nk = K / KS;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    if (tid == 0) {
        if (smem0 & 1023u) { __trap(); }
        if ((LY::STAGE & 511) != 0) { __trap(); }
        for (int s = 0; s < STAGES; ++s) { v3_bar_init(full(s), 1); v3_bar_init(empty(s), NC); cnt[s] = 0; }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    auto issue = [&](const int slot, const int kt) {
        const unsigned sA  = smem0 + slot * LY::STAGE;
        const unsigned sB  = sA + LY::A_BYTES;
        const unsigned sSA = sB + LY::B_BYTES;
        const unsigned sSB = sSA + LY::SA_BYTES;
        v3_expect_tx(full(slot), (unsigned) LY::STAGE);
        v3_tma2d(sA,  &maps.a,  kt * KS, row0, full(slot));
        v3_tma2d(sB,  &maps.b,  kt * KS, tok0, full(slot));
        v3_tma2d(sSA, &maps.sa, row0, kt * KSTEPS, full(slot));
        v3_tma2d(sSB, &maps.sb, tok0, kt * KSTEPS, full(slot));
    };

    if constexpr (PROD == 2) {
        if (warp >= NC) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n");
            if (warp == NC && lane == 0) {
                for (int kt = 0; kt < nk; ++kt) {
                    const int slot = kt % STAGES;
                    if (kt >= STAGES) { v3_wait<HINT>(empty(slot), ((kt / STAGES) - 1) & 1); }
                    issue(slot, kt);
                }
            }
            return;
        }
        asm volatile("setmaxnreg.inc.sync.aligned.u32 120;\n");
    } else {
        if (tid == 0) {
#pragma unroll
            for (int s = 0; s < ((PROD == 1 || PROD == 5) ? STAGES : STAGES - 1); ++s) { if (s < nk) { issue(s, s); } }
        }
        __syncwarp();
    }

    const int wm = warp / WN, wn = warp % WN;
    const int g = lane >> 2, tig = lane & 3;
    float acc[TM][TN][4];
#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int p = 0; p < TN; ++p)
#pragma unroll
            for (int l = 0; l < 4; ++l) acc[m][p][l] = 0.0f;

    const int ar = wm * WRS + (lane & 15), ac = lane >> 4;
    const int br = wn * WCS + (lane >> 4) * 8 + (lane & 7), bc = (lane >> 3) & 1;

    for (int kt = 0; kt < nk; ++kt) {
        const int slot = kt % STAGES;
        if constexpr (PROD == 0) {
            if (tid == 0) {
                const int nx = kt + STAGES - 1;
                if (nx < nk) {
                    const int ns = nx % STAGES;
                    if (nx >= STAGES) { v3_wait<HINT>(empty(ns), ((nx / STAGES) - 1) & 1); }
                    issue(ns, nx);
                }
            }
            __syncwarp();
        }
        v3_wait<HINT>(full(slot), (kt / STAGES) & 1);
        const unsigned sA = smem0 + slot * LY::STAGE;
        const unsigned sB = sA + LY::A_BYTES;
        const float * sSA = (const float *) (v3_smem + slot * LY::STAGE + LY::A_BYTES + LY::B_BYTES);
        const float * sSB = sSA + KSTEPS * BM;
        const unsigned uSA = sB + LY::B_BYTES, uSB = uSA + KSTEPS * BM * 4;
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int a[TM][4];
#pragma unroll
            for (int m = 0; m < TM; ++m) { f8f4v2_ldsm_x4_s(a[m], sA + LY::swz(ar + m * 16, 2 * ks + ac)); }
            const float4 dav = SHL ? v3_lds4(uSA + 4u * (ks * BM + wm * WRS + g * 4)) : *(const float4 *) (sSA + ks * BM + wm * WRS + g * 4);
            const float da[2][2] = {{dav.x, dav.y}, {dav.z, dav.w}};
#pragma unroll
            for (int pp = 0; pp < TN / 2; ++pp) {
                int b[4];
                f8f4v2_ldsm_x4_s(b, sB + LY::swz(br + pp * 16, 2 * ks + bc));
                const float4 dbv = SHL ? v3_lds4(uSB + 4u * (ks * BN + wn * WCS + (pp * 4 + tig) * 4)) : *(const float4 *) (sSB + ks * BN + wn * WCS + (pp * 4 + tig) * 4);
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
        __syncwarp();
        if constexpr (PROD == 5) {
            // the last warp to leave the slot refills it: nobody waits for an empty slot
            if (lane == 0) {
                __threadfence_block();
                const int old = atomicAdd(&cnt[slot], 1);
                if (old == NC - 1) {
                    cnt[slot] = 0;
                    __threadfence_block();
                    const int nx = kt + STAGES;
                    if (nx < nk) { issue(slot, nx); }
                }
            }
            __syncwarp();
        } else {
            if (lane == 0) { v3_arrive(empty(slot)); }
        }
        if constexpr (PROD == 3 || PROD == 4) {
            // after compute kt: stage kt + S - 1 into slot (kt - 1) % S, which every warp left one
            // iteration ago; PROD 4 rotates the issuing warp
            const int issuer = PROD == 3 ? 0 : (kt % NC);
            if (warp == issuer) {
                if (lane == 0) {
                    const int nx = kt + STAGES - 1;
                    if (nx < nk) {
                        const int ns = nx % STAGES;
                        if (nx >= STAGES) { v3_wait<HINT>(empty(ns), ((nx / STAGES) - 1) & 1); }
                        issue(ns, nx);
                    }
                }
                __syncwarp();
            }
        }
        if constexpr (PROD == 1) {
            if (tid == 0) {
                const int nx = kt + STAGES;
                if (nx < nk) {
                    const int ns = nx % STAGES;          // == slot
                    v3_wait<HINT>(empty(ns), ((nx / STAGES) - 1) & 1);
                    issue(ns, nx);
                }
            }
            __syncwarp();
        }
    }

#pragma unroll
    for (int m = 0; m < TM; ++m)
#pragma unroll
        for (int p = 0; p < TN; ++p)
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = row0 + wm * WRS + m * 16 + (l >> 1) * 8 + g;
                const int j = tok0 + wn * WCS + p * 8 + 2 * tig + (l & 1);
                if (i < out_f && j < n_tokens) { dst[(size_t) j * out_f + i] = acc[m][p][l] * out_scale; }
            }
}

static bool map2d(CUtensorMap * m, CUtensorMapDataType ty, int esz, const void * base, uint64_t inner, uint64_t outer,
                  uint64_t row_bytes, uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swz) {
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {row_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t estr[2] = {1, 1};
    (void) esz;
    const CUresult r = cuTensorMapEncodeTiled(m, ty, 2, const_cast<void *>(base), dims, strides, box, estr,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed %d\n", (int) r); return false; }
    return true;
}

template <int BM, int BN, int WM, int WN, int KS, int STAGES, int PROD, bool SHL = false, bool HINT = true>
static cudaError_t v3_launch(const uint8_t * vals, const float * scales, const uint8_t * act, const float * act_sc,
                             float * y, int out_f, int Mp, int K, int n_tokens, int Tp, float out_scale, int G) {
    using LY = f8f4v2_layout<BM, BN, KS>;
    V3Maps mp;
    const CUtensorMapSwizzle swz = KS == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_32B;
    if (!map2d(&mp.a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, vals, K, Mp, K, KS, BM, swz)) { return cudaErrorInvalidValue; }
    if (!map2d(&mp.b, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, act, K, n_tokens, K, KS, BN, swz)) { return cudaErrorInvalidValue; }
    if (!map2d(&mp.sa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, scales, Mp, K / 32, (uint64_t) Mp * 4, BM, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE)) { return cudaErrorInvalidValue; }
    if (!map2d(&mp.sb, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, act_sc, Tp, K / 32, (uint64_t) Tp * 4, BN, KS / 32, CU_TENSOR_MAP_SWIZZLE_NONE)) { return cudaErrorInvalidValue; }
    constexpr size_t smem = (size_t) STAGES * LY::STAGE + 16 * STAGES + 4 * STAGES;
    auto kern = mul_mat_f8f4v3<BM, BN, WM, WN, KS, STAGES, PROD, SHL, HINT>;
    cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    if (e != cudaSuccess) { return e; }
    const int nty = (out_f + BM - 1) / BM, ntx = (n_tokens + BN - 1) / BN;
    const int g = G < 1 ? 1 : (G > nty ? nty : G);
    kern<<<(unsigned) (nty * ntx), WM * WN * 32 + (PROD == 2 ? 128 : 0), smem>>>(mp, y, out_f, K, n_tokens, nty, out_scale, g);
    return cudaGetLastError();
}

static uint32_t rng = 7;
static uint8_t e4m3_rand() { rng = rng * 1664525u + 1013904223u; uint8_t b = (uint8_t)(rng >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; return b; }
static float sc_rand() { rng = rng * 1664525u + 1013904223u; return 0.004f + (rng >> 8) * (1.0f / 16777216.0f) * 0.02f; }
template <typename F>
static double timeit(F f, int reps) {
    std::vector<float> ts; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    f(); cudaDeviceSynchronize();
    for (int r = 0; r < reps; r++) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms); }
    std::sort(ts.begin(), ts.end()); return ts[ts.size() / 2];
}

int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 11;
    const int only = argc > 3 ? atoi(argv[3]) : -1;   // shape index, -1 all
    bool arm_on[32] = {};
    if (argc > 4) { for (char * s = strtok(argv[4], ","); s; s = strtok(nullptr, ",")) { const int i = atoi(s); if (i >= 0 && i < 32) arm_on[i] = true; } }
    else { for (bool & b : arm_on) b = true; }
    int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    struct Shape { const char * name; int out_f, K; } shapes[] = {
        {"gate+up 34816x5120", 34816, 5120}, {"down 5120x17408", 5120, 17408},
        {"wqkv 10240x5120", 10240, 5120}, {"ssm_out 5120x6144", 5120, 6144}, {"wk 1024x5120", 1024, 5120}};
    setvbuf(stdout, nullptr, _IOLBF, 0);
    printf("T=%d reps=%d (median)\n", T, reps);
    for (int si = 0; si < 5; ++si) {
        if (only >= 0 && si != only) { continue; }
        const Shape sh = shapes[si];
        const int Mp = f8f4v2_mp(sh.out_f), K = sh.K, Tp = (T + 255) / 256 * 256;
        std::vector<uint8_t> hv((size_t) Mp * K), ha((size_t) T * K);
        for (auto & b : hv) b = e4m3_rand();
        for (auto & b : ha) b = e4m3_rand();
        std::vector<float> hs((size_t) (K / 32) * Mp), hsa((size_t) (K / 32) * Tp);
        for (auto & x : hs) x = sc_rand();
        for (auto & x : hsa) x = sc_rand();
        uint8_t *v, *a; float *sc, *sa, *y0, *y1;
        const size_t ny = (size_t) T * sh.out_f;
        cudaMalloc(&v, hv.size()); cudaMalloc(&a, ha.size());
        cudaMalloc(&sc, hs.size() * 4); cudaMalloc(&sa, hsa.size() * 4);
        cudaMalloc(&y0, ny * 4); cudaMalloc(&y1, ny * 4);
        cudaMemcpy(v, hv.data(), hv.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(a, ha.data(), ha.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(sc, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(sa, hsa.data(), hsa.size() * 4, cudaMemcpyHostToDevice);
        const double macs = (double) sh.out_f * K * T;
        std::vector<float> r0(ny), r1(ny);
        auto report = [&](const char * name, double ms, cudaError_t e, bool ref) {
            size_t bad = 0;
            if (!ref) {
                cudaMemcpy(r1.data(), y1, ny * 4, cudaMemcpyDeviceToHost);
                for (size_t i = 0; i < ny; i++) bad += memcmp(&r0[i], &r1[i], 4) != 0;
            }
            printf("%-20s %-34s %8.3f ms %6.1f TMAC/s %6.1f MAC/clk/SM@2200  bitdiff=%zu %s%s\n", sh.name, name, ms, macs / ms / 1e9,
                   macs / (ms * 1e-3) / (48 * 2.2e9), bad, ref ? "REF" : (bad ? "FAIL" : "OK"),
                   e == cudaSuccess ? "" : cudaGetErrorString(e));
        };
        cudaError_t e = cudaSuccess;
        double t = timeit([&] { e = f8f4v2_launch<256, 128, 8, 2, 64, 3, 1, 0>(v, sc, a, sa, y0, sh.out_f, Mp, K, T, Tp, 0.83f, 4, 0); return e; }, reps);
        cudaMemcpy(r0.data(), y0, ny * 4, cudaMemcpyDeviceToHost);
        report("v2 shipped 256x128 k64 s3", t, e, true);
#define ARM(ID, BMV, WMV, KSV, STV, PROD, SHLV, HINTV, NAME) \
        if (arm_on[ID]) { cudaMemset(y1, 0xff, ny * 4); \
            t = timeit([&] { e = v3_launch<BMV, 128, WMV, 2, KSV, STV, PROD, SHLV, HINTV>(v, sc, a, sa, y1, sh.out_f, Mp, K, T, Tp, 0.83f, 4); return e; }, reps); \
            e = cudaDeviceSynchronize(); report(NAME, t, e, false); }
        // ARMS: id, tile rows, warp rows, k per stage, stages, issue scheme, shared-window scales, suspend hint.
        // The logs name arms without the plain / hint suffix: stage2.log and stage3.log ran the plain waits,
        // stage4.log to stage6.log the hinted ones (gemm-v3-probe-unit2/README.md maps every log line).
        ARM(0,  256, 8, 64, 3, 0, false, false, "v3 tma issue-top k64 s3 plain")
        ARM(1,  256, 8, 64, 3, 1, false, false, "v3 tma issue-end k64 s3 plain")
        ARM(2,  256, 8, 32, 6, 0, false, false, "v3 tma issue-top k32 s6 plain")
        ARM(3,  256, 8, 32, 6, 1, false, false, "v3 tma issue-end k32 s6 plain")
        ARM(4,  256, 8, 32, 7, 0, false, false, "v3 tma issue-top k32 s7 plain")
        ARM(5,  256, 8, 64, 3, 3, false, false, "v3 tma issue-mid k64 s3 plain")
        ARM(6,  256, 8, 64, 3, 4, false, false, "v3 tma issue-mid-rot k64 s3 plain")
        ARM(7,  256, 8, 64, 3, 2, false, false, "v3 tma producer-wg s3 plain")
        ARM(8,  256, 8, 64, 3, 1, false, true,  "v3 tma issue-end k64 s3 hint")
        ARM(9,  256, 8, 64, 3, 5, false, true,  "v3 tma last-arriver k64 s3 hint")
        ARM(10, 192, 6, 64, 3, 1, false, true,  "v3 192x128 w6x2 issue-end s3 hint")
        ARM(11, 192, 6, 64, 4, 1, false, true,  "v3 192x128 w6x2 issue-end s4 hint")
        ARM(12, 192, 6, 64, 4, 5, false, true,  "v3 192x128 w6x2 last-arr s4 hint")
        ARM(13, 128, 4, 64, 5, 1, false, true,  "v3 128x128 w4x2 issue-end s5 hint")
        ARM(14, 256, 8, 64, 3, 1, true,  true,  "v3 tma issue-end k64 s3 lds.shared hint")
        cudaMemset(y1, 0xff, ny * 4);
        t = timeit([&] { e = f8f4v2_launch<256, 128, 8, 2, 64, 3, 1, 0>(v, sc, a, sa, y1, sh.out_f, Mp, K, T, Tp, 0.83f, 4, 0); return e; }, reps);
        report("v2 shipped (again)", t, e, false);
        cudaFree(v); cudaFree(a); cudaFree(sc); cudaFree(sa); cudaFree(y0); cudaFree(y1);
    }
    printf("cuda: %s, clock attribute %d MHz\n", cudaGetErrorString(cudaDeviceSynchronize()), clk_khz / 1000);
    return 0;
}
