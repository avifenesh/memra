// Probe (scratch): is a hand-written chain of block-scaled MMAs bit-identical (fp32) to CUTLASS 4.8.1's normal
// form when the weights enter the MMA as e2m1 (CUTLASS's operand type) instead of converted to e4m3 (bs1's)?
// D[m][n] = sum_k (X[m][k] * sx[m][k/32]) * (W[n][k] * sw[n][k/32]), X e4m3 [M][K], W MXFP4 [N][K/2] (low nibble
// first), ue8m0 scales. V1: bs1's chain (W -> e4m3, kind::mxf8f6f4 .e4m3.e4m3, W on A). V2: the same chain with W
// as e2m1 in its 8-bit container (nibble << 2) and .e2m1.e4m3. V3: X on A, W on B, .e4m3.e2m1. Each against a
// CUTLASS plain normal-form GEMM with fp32 D, bit for bit. usage: e2m1_probe [M]... (default 1 2 8 16 64)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

#define CK(x)                                                                          \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(2);                                                                   \
        }                                                                              \
    } while (0)

// ---- CUTLASS plain normal form, fp32 D (the engine's PN4 with D = float)
using F8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using F4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using SFT = cutlass::float_ue8m0_t;
struct Cfg {
    using RowM = cutlass::layout::RowMajor;
    using ColM = cutlass::layout::ColumnMajor;
    using Tile = Shape<_128, _128, _128>;
    using Cl = Shape<_1, _1, _1>;
    using Arch = cutlass::arch::Sm120;
    using Op = cutlass::arch::OpClassBlockScaledTensorOp;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, Op, Tile, Cl, cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, RowM, 4, float, RowM,
        4, cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, Op, F8, RowM, 16, F4, ColM, 128, float, Tile, Cl,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using SfCfg = typename Main::Sm1xxBlkScaledConfig;
};

static int cutlass_gemm(const uint8_t* x, const uint8_t* xsf, const uint8_t* w, const uint8_t* wsf, float* d, int M,
                        int N, int K) {
    using G = Cfg::Gemm;
    typename Cfg::Kernel::StrideA sa{};
    cute::get<0>(sa) = K;
    typename Cfg::Kernel::StrideB sb{};
    cute::get<0>(sb) = K;
    typename Cfg::Kernel::StrideD sd{};
    cute::get<0>(sd) = N;
    typename G::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(const cutlass::float_e4m3_t*)x, sa, (const cutlass::float_e2m1_t*)w, sb, (const SFT*)xsf,
         Cfg::SfCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1)), (const SFT*)wsf,
         Cfg::SfCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1))},
        {{}, nullptr, sd, d, sd},
        cutlass::KernelHardwareInfo{}};
    int dev = 0;
    cudaGetDevice(&dev);
    args.hw_info.device_id = dev;
    cudaDeviceGetAttribute(&args.hw_info.sm_count, cudaDevAttrMultiProcessorCount, dev);
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    G gemm;
    if (G::get_workspace_size(args) != 0) return 1;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 2;
    if (gemm.initialize(args, nullptr, 0) != cutlass::Status::kSuccess) return 3;
    if (gemm.run(0) != cutlass::Status::kSuccess) return 4;
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 5;
}

// ---- the chains
__device__ __forceinline__ uint32_t fp4_to_e4m3x4(uint32_t nib16) {  // bs1's conversion
    const uint32_t lut_lo = 0x3C383000u, lut_hi = 0x4C484440u;
    uint32_t mag = __byte_perm(lut_lo, lut_hi, nib16 & 0x7777u);
    uint32_t sgn = ((nib16 & 0x8u) << 4) | ((nib16 & 0x80u) << 8) | ((nib16 & 0x800u) << 12) |
                   ((nib16 & 0x8000u) << 16);
    return mag | sgn;
}
__device__ __forceinline__ uint32_t fp4_container4(uint32_t nib16) {  // 4 nibbles -> 4 bytes, value in bits 5..2
    return ((nib16 & 0xFu) << 2) | (((nib16 >> 4) & 0xFu) << 10) | (((nib16 >> 8) & 0xFu) << 18) |
           (((nib16 >> 12) & 0xFu) << 26);
}
template <int V, int BA, int BB>
__device__ __forceinline__ void mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                    uint32_t sb) {
    if (V == 1)
        asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0),
              "n"(BB), "n"(0));
    else if (V == 2)
        asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0),
              "n"(BB), "n"(0));
    else
        asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0),
              "n"(BB), "n"(0));
}
template <int V>
__device__ __forceinline__ void mma_u(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa,
                                      uint32_t sb, int u) {
    switch (u) {
        case 0: mma<V, 0, 0>(c, a, b0, b1, sa, sb); break;
        case 1: mma<V, 1, 1>(c, a, b0, b1, sa, sb); break;
        case 2: mma<V, 2, 2>(c, a, b0, b1, sa, sb); break;
        default: mma<V, 3, 3>(c, a, b0, b1, sa, sb); break;
    }
}
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

// V1/V2: W on A (16 weight rows), X on B (8 activation rows). Block (n/16, m/8), one warp.
template <int V>
__global__ void chain_w_on_a(const uint8_t* x, const uint8_t* xsf, const uint8_t* w, const uint8_t* wsf, float* d,
                             int M, int N, int K) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 16, m0 = blockIdx.y * 8, KB = K / 32;
    float c[4] = {0, 0, 0, 0};
    for (int kb = 0; kb < KB; ++kb) {
        int u = kb & 3, kb0 = kb & ~3;
        uint32_t a[4];
        const uint8_t* w0 = w + (long)(n0 + g) * (K / 2) + kb * 16;
        const uint8_t* w1 = w + (long)(n0 + g + 8) * (K / 2) + kb * 16;
        uint32_t q[4] = {ld16(w0 + 2 * t), ld16(w1 + 2 * t), ld16(w0 + 8 + 2 * t), ld16(w1 + 8 + 2 * t)};
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = V == 1 ? fp4_to_e4m3x4(q[i]) : fp4_container4(q[i]);
        int mr = m0 + g;
        uint32_t b0 = 0, b1 = 0, sb = 0;
        if (mr < M) {
            b0 = ld32(x + (long)mr * K + kb * 32 + 4 * t);
            b1 = ld32(x + (long)mr * K + kb * 32 + 16 + 4 * t);
            sb = ld32(xsf + (long)mr * KB + kb0);
        }
        int wr = n0 + g + 8 * (t & 1);
        uint32_t sa = ld32(wsf + (long)wr * KB + kb0);
        mma_u<V>(c, a, b0, b1, sa, sb, u);
    }
    // c0: w row g, x row 2t; c1: x row 2t + 1; c2, c3: w row g + 8
    for (int e = 0; e < 4; ++e) {
        int n = n0 + g + (e >= 2 ? 8 : 0), m = m0 + 2 * t + (e & 1);
        if (m < M) d[(long)m * N + n] = c[e];
    }
}
// V3: X on A (16 activation rows), W on B (8 weight rows). Block (n/8, m/16).
__global__ void chain_x_on_a(const uint8_t* x, const uint8_t* xsf, const uint8_t* w, const uint8_t* wsf, float* d,
                             int M, int N, int K) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    int n0 = blockIdx.x * 8, m0 = blockIdx.y * 16, KB = K / 32;
    float c[4] = {0, 0, 0, 0};
    for (int kb = 0; kb < KB; ++kb) {
        int u = kb & 3, kb0 = kb & ~3;
        uint32_t a[4] = {0, 0, 0, 0};
        int r0 = m0 + g, r1 = m0 + g + 8;
        if (r0 < M) {
            a[0] = ld32(x + (long)r0 * K + kb * 32 + 4 * t);
            a[2] = ld32(x + (long)r0 * K + kb * 32 + 16 + 4 * t);
        }
        if (r1 < M) {
            a[1] = ld32(x + (long)r1 * K + kb * 32 + 4 * t);
            a[3] = ld32(x + (long)r1 * K + kb * 32 + 16 + 4 * t);
        }
        int sr = m0 + g + 8 * (t & 1);
        uint32_t sa = sr < M ? ld32(xsf + (long)sr * KB + kb0) : 0;
        const uint8_t* wr = w + (long)(n0 + g) * (K / 2) + kb * 16;
        uint32_t b0 = fp4_container4(ld16(wr + 2 * t)), b1 = fp4_container4(ld16(wr + 8 + 2 * t));
        uint32_t sb = ld32(wsf + (long)(n0 + g) * KB + kb0);
        mma_u<3>(c, a, b0, b1, sa, sb, u);
    }
    // c0: x row g, w col 2t; c1: w col 2t + 1; c2, c3: x row g + 8
    for (int e = 0; e < 4; ++e) {
        int m = m0 + g + (e >= 2 ? 8 : 0), n = n0 + 2 * t + (e & 1);
        if (m < M) d[(long)m * N + n] = c[e];
    }
}

// ---- host
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() {
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (uint32_t)rs;
}
static long sf_off(long r, long kb, long kbn) {
    return (r / 128) * (((kbn + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}
static long sf_bytes(long rows, long k) { return (rows + 127) / 128 * 128 * (((k / 32) + 3) / 4) * 4; }

int main(int argc, char** argv) {
    std::vector<int> ms;
    for (int i = 1; i < argc; ++i) ms.push_back(atoi(argv[i]));
    if (ms.empty()) ms = {1, 2, 8, 16, 64};
    const int N = 2304, K = 5120, KB = K / 32;
    int total_bad[4] = {0, 0, 0, 0};
    for (int M : ms) {
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<uint8_t> x((long)M * K), xs((long)M * KB), w((long)N * K / 2), ws((long)N * KB);
            for (auto& b : x) {
                b = rnd() & 0xFF;
                if ((b & 0x7F) == 0x7F) b ^= 1;
            }
            for (auto& b : xs) b = 118 + rnd() % 13;
            for (auto& b : w) b = rnd() & 0xFF;
            for (auto& b : ws) b = 118 + rnd() % 13;
            // CUTLASS's scale layout
            std::vector<uint8_t> xsl(sf_bytes(M, K), 0), wsl(sf_bytes(N, K), 0);
            for (int r = 0; r < M; ++r)
                for (int kb = 0; kb < KB; ++kb) xsl[sf_off(r, kb, KB)] = xs[(long)r * KB + kb];
            for (int r = 0; r < N; ++r)
                for (int kb = 0; kb < KB; ++kb) wsl[sf_off(r, kb, KB)] = ws[(long)r * KB + kb];
            uint8_t *dx, *dxs, *dw, *dws, *dxsl, *dwsl;
            float *dref, *dv;
            CK(cudaMalloc(&dx, x.size()));
            CK(cudaMalloc(&dxs, xs.size()));
            CK(cudaMalloc(&dw, w.size()));
            CK(cudaMalloc(&dws, ws.size()));
            CK(cudaMalloc(&dxsl, xsl.size()));
            CK(cudaMalloc(&dwsl, wsl.size()));
            CK(cudaMalloc(&dref, (long)M * N * 4));
            CK(cudaMalloc(&dv, (long)M * N * 4));
            CK(cudaMemcpy(dx, x.data(), x.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dxs, xs.data(), xs.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dws, ws.data(), ws.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dxsl, xsl.data(), xsl.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dwsl, wsl.data(), wsl.size(), cudaMemcpyHostToDevice));
            int rc = cutlass_gemm(dx, dxsl, dw, dwsl, dref, M, N, K);
            if (rc) {
                printf("cutlass rc %d\n", rc);
                return 1;
            }
            std::vector<uint32_t> ref((long)M * N), got((long)M * N);
            CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
            for (int v = 1; v <= 3; ++v) {
                CK(cudaMemset(dv, 0xFF, (long)M * N * 4));
                if (v == 1) chain_w_on_a<1><<<dim3(N / 16, (M + 7) / 8), 32>>>(dx, dxs, dw, dws, dv, M, N, K);
                if (v == 2) chain_w_on_a<2><<<dim3(N / 16, (M + 7) / 8), 32>>>(dx, dxs, dw, dws, dv, M, N, K);
                if (v == 3) chain_x_on_a<<<dim3(N / 8, (M + 15) / 16), 32>>>(dx, dxs, dw, dws, dv, M, N, K);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(got.data(), dv, got.size() * 4, cudaMemcpyDeviceToHost));
                long bad = 0, ulp1 = 0;
                for (long i = 0; i < (long)got.size(); ++i)
                    if (got[i] != ref[i]) {
                        ++bad;
                        long dd = (long)got[i] - (long)ref[i];
                        if (dd == 1 || dd == -1) ++ulp1;
                    }
                total_bad[v] += bad;
                printf("M %3d trial %d V%d: %ld of %ld fp32 differ from CUTLASS (%ld by one ulp)\n", M, trial, v, bad,
                       (long)got.size(), ulp1);
            }
            cudaFree(dx); cudaFree(dxs); cudaFree(dw); cudaFree(dws); cudaFree(dxsl); cudaFree(dwsl);
            cudaFree(dref); cudaFree(dv);
        }
    }
    printf("TOTAL differing: V1 (bs1, e4m3.e4m3) %d, V2 (e2m1.e4m3, W on A) %d, V3 (e4m3.e2m1, X on A) %d\n",
           total_bad[1], total_bad[2], total_bad[3]);
    return 0;
}
