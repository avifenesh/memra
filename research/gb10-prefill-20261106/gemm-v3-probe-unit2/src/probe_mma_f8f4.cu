// Probe: the GB10 tensor-core ceiling of the F8F4 v2 GEMM tile's inner step, register operands only.
// The tile issues, per k32 block and warp, TM x TN = 2 x 8 mma.sync m16n8k32 (kind::mxf8f6f4
// block_scale, unit ue8m0 scales, e4m3 x e4m3, f32) into a zeroed C, then applies the per-32
// activation and weight scales in f32: s += da * db * C. Modes:
//   0 chain   : the mma accumulates into s directly (the tensor-core rate, no software scaling)
//   1 add     : zeroed C, then s += C (one FADD per element)
//   2 scaled  : zeroed C, then s += da * db * C (the tile's expression)
//   3 plain   : mode 2 with kind::f8f6f4 (no block_scale qualifier)
//   4 chainp  : mode 0 with kind::f8f6f4
// Every mode runs 16 warps x 1 CTA per SM (the tile's shape) and 8 warps x 2 CTAs. Reports
// MAC/clk/SM from cudaEvent time and the device's current SM clock.
// build: nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 probe_mma_f8f4.cu -o probe_mma_f8f4
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define ONE_UE8M0 0x7f7f7f7fu

template <bool BS>
static __device__ __forceinline__ void mma(float* c, const int* a, int b0, int b1) {
    if constexpr (BS) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
                     ".f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(ONE_UE8M0),
                       "r"(ONE_UE8M0));
    } else {
        asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}

template <int MODE, int NT, int MINB>
__global__ void __launch_bounds__(NT, MINB) k_mma(float* out, int iters) {
    constexpr bool BS = MODE <= 2;
    const int tid = threadIdx.x;
    int a[2][4], b[8][2];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int r = 0; r < 4; r++) a[m][r] = 0x38383838 ^ ((tid + m * 7 + r * 3) & 0x07070707);
#pragma unroll
    for (int n = 0; n < 8; n++) {
        b[n][0] = 0x30303030 ^ ((tid * 3 + n) & 0x03030303);
        b[n][1] = 0x31313131 ^ ((tid * 5 + n) & 0x03030303);
    }
    float da[2][2], db[8][2];
#pragma unroll
    for (int m = 0; m < 2; m++) { da[m][0] = 0.5f + 1e-3f * tid; da[m][1] = 0.25f + 2e-3f * m; }
#pragma unroll
    for (int n = 0; n < 8; n++) { db[n][0] = 0.75f + 1e-3f * n; db[n][1] = 1.25f - 1e-3f * n; }
    float s[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; m++)
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
            for (int l = 0; l < 4; l++) s[m][n][l] = 0.0f;
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int m = 0; m < 2; m++) {
#pragma unroll
            for (int n = 0; n < 8; n++) {
                if constexpr (MODE == 0 || MODE == 4) {
                    mma<BS>(s[m][n], a[m], b[n][0], b[n][1]);
                } else {
                    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma<BS>(C, a[m], b[n][0], b[n][1]);
                    if constexpr (MODE == 1) {
#pragma unroll
                        for (int l = 0; l < 4; l++) s[m][n][l] += C[l];
                    } else {
                        s[m][n][0] += da[m][0] * db[n][0] * C[0];
                        s[m][n][1] += da[m][0] * db[n][1] * C[1];
                        s[m][n][2] += da[m][1] * db[n][0] * C[2];
                        s[m][n][3] += da[m][1] * db[n][1] * C[3];
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
    out[blockIdx.x * NT + tid] = t;
}

template <int MODE, int NT, int MINB>
static void run(const char* name, float* out, int nsm, int clk_mhz) {
    const int iters = 4096;
    const int grid = nsm * MINB;
    auto f = [&] { k_mma<MODE, NT, MINB><<<grid, NT>>>(out, iters); };
    f();
    cudaDeviceSynchronize();
    std::vector<float> ts;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < 9; r++) {
        cudaEventRecord(e0);
        f();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const double ms = ts[4];
    const double macs = (double)grid * (NT / 32) * iters * 16.0 * 16 * 8 * 32;
    const double rate = macs / (ms * 1e-3);
    printf("%-34s %8.3f ms  %7.1f TMAC/s  %7.1f MAC/clk/SM at %d MHz\n", name, ms, rate / 1e12,
           rate / (nsm * clk_mhz * 1e6), clk_mhz);
}

int main() {
    int dev = 0, nsm = 0, clk_khz = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    const int clk_mhz = getenv("PROBE_MHZ") ? atoi(getenv("PROBE_MHZ")) : clk_khz / 1000;
    printf("SMs %d, clock attribute %d MHz, using %d MHz\n", nsm, clk_khz / 1000, clk_mhz);
    float* out;
    cudaMalloc(&out, (size_t)nsm * 2 * 512 * sizeof(float));
    run<0, 512, 1>("chain   block_scale 16w x1", out, nsm, clk_mhz);
    run<1, 512, 1>("add     block_scale 16w x1", out, nsm, clk_mhz);
    run<2, 512, 1>("scaled  block_scale 16w x1", out, nsm, clk_mhz);
    run<3, 512, 1>("scaled  plain       16w x1", out, nsm, clk_mhz);
    run<4, 512, 1>("chain   plain       16w x1", out, nsm, clk_mhz);
    run<0, 256, 2>("chain   block_scale  8w x2", out, nsm, clk_mhz);
    run<2, 256, 2>("scaled  block_scale  8w x2", out, nsm, clk_mhz);
    cudaError_t e = cudaDeviceSynchronize();
    printf("cuda: %s\n", cudaGetErrorString(e));
    return 0;
}
