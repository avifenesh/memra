// Co-issue probe: what limits the F8F4 tile's exact epilogue on GB10. Per warp and iteration, 16
// independent mma.sync m16n8k32 block-scale QMMAs into a zeroed C (the tile's form) and/or FP32
// work. All scale operands change every iteration (read from a rotating register window fed by a
// shuffle), so nothing is hoisted. 16 warps x 1 CTA per SM unless noted. Reports QMMA cycles per SMSP.
//   qmma        : QMMA only, results folded with one FADD per 4 outputs every 8 iterations
//   fadd4       : QMMA + 4 FADD   (s += C)
//   fmul4       : QMMA + 4 FMUL   (s = t * C, t varies)
//   ffma4       : QMMA + 4 FFMA   (s += t * C, t varies but needs no FMUL)
//   tile        : QMMA + 4 FMUL + 4 FFMA (the exact epilogue: t = dA * dB, s += t * C)
//   tile-indep  : QMMA + 4 FMUL + 4 FFMA on operands that do not depend on C (same count, no dependency)
//   ffma-only   : 8 FFMA per would-be MMA, no QMMA
//   split       : warps 0-7 QMMA + fadd-fold, warps 8-15 8 FFMA per MMA-slot, no data exchange
// build: nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 coissue.cu -o coissue
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define ONE 0x7f7f7f7fu
static __device__ __forceinline__ void qmma(float * c, const int * a, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(ONE), "r"(ONE));
}

enum { QMMA = 0, FADD4, FMUL4, FFMA4, TILE, TILE_INDEP, FFMA_ONLY, SPLIT };

template <int MODE>
__global__ void __launch_bounds__(512, 1) k(float * out, int iters, float seed) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int a[2][4], b[8][2];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int r = 0; r < 4; r++) a[m][r] = 0x38383838 ^ ((tid + m * 7 + r * 3) & 0x07070707);
#pragma unroll
    for (int n = 0; n < 8; n++) { b[n][0] = 0x30303030 ^ ((tid * 3 + n) & 0x03030303); b[n][1] = 0x31313131 ^ ((tid * 5 + n) & 0x03030303); }
    float da[2][2] = {{seed + 1e-3f * lane, seed * 0.5f + 1e-3f * warp}, {seed * 0.25f + 2e-3f * lane, seed * 0.125f + 3e-3f * warp}};
    float db[8][2];
#pragma unroll
    for (int n = 0; n < 8; n++) { db[n][0] = 0.75f + 1e-3f * n + seed; db[n][1] = 1.25f - 1e-3f * n; }
    float s[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int l = 0; l < 4; l++) s[m][n][l] = 0.0f;
    const bool mma_warp = MODE != SPLIT || warp < 8;
    for (int it = 0; it < iters; it++) {
        // new scales every iteration: a shuffle the compiler cannot hoist
#pragma unroll
        for (int m = 0; m < 2; m++) { da[m][0] = __shfl_xor_sync(0xffffffffu, da[m][0], 1); da[m][1] = __shfl_xor_sync(0xffffffffu, da[m][1], 2); }
#pragma unroll
        for (int n = 0; n < 8; n++) { db[n][0] = __shfl_xor_sync(0xffffffffu, db[n][0], 4); }
#pragma unroll
        for (int m = 0; m < 2; m++) {
#pragma unroll
            for (int n = 0; n < 8; n++) {
                float * acc = s[m][n];
                const float dA = da[m][0], dA1 = da[m][1];
                if constexpr (MODE == FFMA_ONLY || MODE == SPLIT) {
                    if (MODE == FFMA_ONLY || !mma_warp) {
                        const float t0 = dA * db[n][0], t1 = dA * db[n][1], t2 = dA1 * db[n][0], t3 = dA1 * db[n][1];
                        acc[0] = fmaf(t0, acc[1], acc[0]); acc[1] = fmaf(t1, acc[2], acc[1]);
                        acc[2] = fmaf(t2, acc[3], acc[2]); acc[3] = fmaf(t3, acc[0], acc[3]);
                    } else {
                        float C[4] = {0.f, 0.f, 0.f, 0.f};
                        qmma(C, a[m], b[n][0], b[n][1]);
#pragma unroll
                        for (int l = 0; l < 4; l++) acc[l] += C[l];
                    }
                } else {
                    float C[4] = {0.f, 0.f, 0.f, 0.f};
                    qmma(C, a[m], b[n][0], b[n][1]);
                    if constexpr (MODE == QMMA) {
                        if ((it & 7) == 0) { acc[0] += C[0] + C[1] + C[2] + C[3]; }
                    } else if constexpr (MODE == FADD4) {
#pragma unroll
                        for (int l = 0; l < 4; l++) acc[l] += C[l];
                    } else if constexpr (MODE == FMUL4) {
                        acc[0] = dA * C[0] * acc[1]; acc[1] = db[n][0] * C[1] * acc[2];
                        acc[2] = dA1 * C[2] * acc[3]; acc[3] = db[n][1] * C[3] * acc[0];
                    } else if constexpr (MODE == FFMA4) {
                        acc[0] = fmaf(dA, C[0], acc[0]); acc[1] = fmaf(db[n][0], C[1], acc[1]);
                        acc[2] = fmaf(dA1, C[2], acc[2]); acc[3] = fmaf(db[n][1], C[3], acc[3]);
                    } else if constexpr (MODE == TILE) {
                        acc[0] = fmaf(dA * db[n][0], C[0], acc[0]); acc[1] = fmaf(dA * db[n][1], C[1], acc[1]);
                        acc[2] = fmaf(dA1 * db[n][0], C[2], acc[2]); acc[3] = fmaf(dA1 * db[n][1], C[3], acc[3]);
                    } else if constexpr (MODE == TILE_INDEP) {
                        acc[0] = fmaf(dA * db[n][0], acc[1], acc[0]); acc[1] = fmaf(dA * db[n][1], acc[2], acc[1]);
                        acc[2] = fmaf(dA1 * db[n][0], acc[3], acc[2]); acc[3] = fmaf(dA1 * db[n][1], acc[0], acc[3]);
                        if ((it & 7) == 0) { acc[0] += C[0] + C[1] + C[2] + C[3]; }
                    }
                }
            }
        }
    }
    float t = 0.0f;
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int l = 0; l < 4; l++) t += s[m][n][l];
    out[blockIdx.x * 512 + tid] = t;
}

template <int MODE>
static void run(const char * name, float * out, int nsm) {
    const int iters = 4096;
    auto f = [&] { k<MODE><<<nsm, 512>>>(out, iters, 0.5f); };
    f(); cudaDeviceSynchronize();
    std::vector<float> ts; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < 9; r++) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms); }
    std::sort(ts.begin(), ts.end());
    const double ms = ts[4];
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    // MMA slots: 16 warps (8 in SPLIT do MMAs) x iters x 16; report cycles per slot per SMSP at the measured clock
    const double slots_per_smsp = (MODE == SPLIT ? 2.0 : 4.0) * iters * 16;   // per SMSP: warps x iters x 16
    const double ghz = 2.1;   // nominal; the caller logs nvidia-smi clocks alongside
    const double cyc = ms * 1e-3 * ghz * 1e9 / slots_per_smsp;
    const double macs = (double) nsm * (MODE == SPLIT ? 8 : 16) * iters * 16 * 4096.0;
    printf("%-12s %8.3f ms  %6.1f cycles per MMA-slot per SMSP at 2.1 GHz  %7.1f TMAC/s (MMA modes)\n", name, ms, cyc,
           (MODE == FFMA_ONLY) ? 0.0 : macs / (ms * 1e-3) / 1e12);
}

int main() {
    int nsm = 0; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    float * out; cudaMalloc(&out, (size_t) nsm * 512 * 4);
    run<QMMA>("qmma", out, nsm);
    run<FADD4>("fadd4", out, nsm);
    run<FMUL4>("fmul4", out, nsm);
    run<FFMA4>("ffma4", out, nsm);
    run<TILE>("tile", out, nsm);
    run<TILE_INDEP>("tile-indep", out, nsm);
    run<FFMA_ONLY>("ffma-only", out, nsm);
    run<SPLIT>("split", out, nsm);
    printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
