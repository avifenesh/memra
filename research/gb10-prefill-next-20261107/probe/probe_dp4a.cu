// Probe: the GDN beta/alpha NVFP4 rp dp4a matvec at prefill rows (m 4096, in_f 5120, out_f 48).
// V0 is qmatvec.cu's nvfp4_dp4a_rp_tok<4, 2> verbatim; every variant keeps the per-(token, row)
// arithmetic (same groups per thread, dp4a chains, partial/acc expressions, warp tree, cross-warp
// tree), so outputs must match V0 bit for bit. Times: median of 21 launches each.
// build: nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 probe_dp4a.cu -o probe_dp4a
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__device__ __constant__ signed char kvalues_mxfp4_d[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ int2 get_int_from_table_16_d(int q4, const signed char* table) {
    const uint32_t* table32 = (const uint32_t*)table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210u | ((q4 & 0x88888888u) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16u * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], (uint32_t)q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], (uint32_t)q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
__device__ __forceinline__ float ue4m3_to_f32_d(unsigned char x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int exp = (x >> 3) & 0xF;
    float man = (float)(x & 0x7);
    float raw = (exp == 0) ? ldexpf(man, -9) : ldexpf(1.0f + man / 8.0f, exp - 7);
    return raw * 0.5f;
}
__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// RB: rows per block (blockIdx.y picks the row range; RB >= out_f is the shipped kernel);
// UR: unroll of the row loop.
template <int TT, int NG, int RB, int UR>
__device__ __forceinline__ void tok_body(const unsigned char* __restrict__ W, const signed char* __restrict__ aq,
                                         const float* __restrict__ ad, float* __restrict__ y, int in_f, int out_f,
                                         int m) {
    constexpr int OB = RB < 32 ? RB : 32;
    const int t0 = blockIdx.x * TT;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nsb = in_f >> 5;
    const int nsb64 = in_f >> 6;
    int4 a01[TT][NG], a23[TT][NG];
    float adv[TT][NG];
#pragma unroll
    for (int tt = 0; tt < TT; tt++)
#pragma unroll
        for (int k = 0; k < NG; k++) {
            const int g = tid + 128 * k, t = t0 + tt;
            if (g < nsb && t < m) {
                const int4* aq16 = (const int4*)(aq + (size_t)t * in_f + (size_t)g * 32);
                a01[tt][k] = aq16[0];
                a23[tt][k] = aq16[1];
                adv[tt][k] = ad[(size_t)t * nsb + g];
            } else {
                a01[tt][k] = make_int4(0, 0, 0, 0);
                a23[tt][k] = make_int4(0, 0, 0, 0);
                adv[tt][k] = 0.0f;
            }
        }
    __shared__ float s[TT][OB][4];
    const int r_lo = RB >= 1024 ? 0 : blockIdx.y * RB;
    const int r_hi = RB >= 1024 ? out_f : min(out_f, r_lo + RB);
    for (int ob = r_lo; ob < r_hi; ob += OB) {
        const int on = min(OB, r_hi - ob);
#pragma unroll UR
        for (int oo = 0; oo < on; oo++) {
            const int o = ob + oo;
            const unsigned char* rowq = W + (size_t)o * nsb64 * 32;
            const unsigned char* rows = W + (size_t)out_f * nsb64 * 32 + (size_t)o * nsb64 * 4;
            float acc[TT];
#pragma unroll
            for (int tt = 0; tt < TT; tt++) acc[tt] = 0.0f;
#pragma unroll
            for (int k = 0; k < NG; k++) {
                const int g = tid + 128 * k;
                if (g >= nsb) break;
                const int sblk = g >> 1;
                const int s0 = (g & 1) * 2;
                const int4 qw = *(const int4*)(rowq + (size_t)g * 16);
                const int cscw = *(const int*)(rows + (size_t)sblk * 4);
                const int2 va0 = get_int_from_table_16_d(qw.x, kvalues_mxfp4_d);
                const int2 vb0 = get_int_from_table_16_d(qw.y, kvalues_mxfp4_d);
                const int2 va1 = get_int_from_table_16_d(qw.z, kvalues_mxfp4_d);
                const int2 vb1 = get_int_from_table_16_d(qw.w, kvalues_mxfp4_d);
                const float sc0 = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 0))) & 0xFF));
                const float sc1 = ue4m3_to_f32_d((unsigned char)((cscw >> (8 * (s0 + 1))) & 0xFF));
#pragma unroll
                for (int tt = 0; tt < TT; tt++) {
                    const int aq4[8] = {a01[tt][k].x, a01[tt][k].y, a01[tt][k].z, a01[tt][k].w,
                                        a23[tt][k].x, a23[tt][k].y, a23[tt][k].z, a23[tt][k].w};
                    float partial = 0.0f;
                    int sumi = 0;
                    sumi = dp4a(va0.x, aq4[0], sumi);
                    sumi = dp4a(vb0.x, aq4[1], sumi);
                    sumi = dp4a(va0.y, aq4[2], sumi);
                    sumi = dp4a(vb0.y, aq4[3], sumi);
                    partial += sc0 * (float)sumi;
                    sumi = 0;
                    sumi = dp4a(va1.x, aq4[4], sumi);
                    sumi = dp4a(vb1.x, aq4[5], sumi);
                    sumi = dp4a(va1.y, aq4[6], sumi);
                    sumi = dp4a(vb1.y, aq4[7], sumi);
                    partial += sc1 * (float)sumi;
                    acc[tt] += adv[tt][k] * partial;
                }
            }
#pragma unroll
            for (int tt = 0; tt < TT; tt++) {
                float v = acc[tt];
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
                if (lane == 0) s[tt][oo][warp] = v;
            }
        }
        __syncthreads();
        for (int task = warp; task < TT * on; task += 4) {
            const int tt = task / on, oo = task % on;
            float v = (lane < 4) ? s[tt][oo][lane] : 0.0f;
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0 && t0 + tt < m) y[(size_t)(t0 + tt) * out_f + ob + oo] = v;
        }
        __syncthreads();
    }
}

template <int TT, int NG, int RB, int UR>
__global__ void __launch_bounds__(128) k_tok(const unsigned char* W, const signed char* aq, const float* ad,
                                             float* y, int in_f, int out_f, int m) {
    tok_body<TT, NG, RB, UR>(W, aq, ad, y, in_f, out_f, m);
}

static uint32_t rng = 12345;
static uint32_t nxt() { rng = rng * 1664525u + 1013904223u; return rng; }

template <typename F>
static float timeit(F f) {
    std::vector<float> ts;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    cudaDeviceSynchronize();
    for (int i = 0; i < 21; i++) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    return ts[10];
}

int main() {
    const int m = 4096, in_f = 5120, out_f = 48;
    const int nsb = in_f / 32, nsb64 = in_f / 64;
    const size_t wq = (size_t)out_f * nsb64 * 32, ws = (size_t)out_f * nsb64 * 4;
    std::vector<unsigned char> hW(wq + ws);
    for (size_t i = 0; i < wq; i++) hW[i] = (unsigned char)nxt();
    for (size_t i = 0; i < ws; i++) {
        unsigned char b = (unsigned char)(0x20 + (nxt() % 0x30));
        hW[wq + i] = b;
    }
    std::vector<signed char> haq((size_t)m * in_f);
    for (auto& v : haq) v = (signed char)((int)(nxt() % 255) - 127);
    std::vector<float> had((size_t)m * nsb);
    for (auto& v : had) v = 1e-3f * (float)(1 + nxt() % 1000);
    unsigned char* W;
    signed char* aq;
    float *ad, *y0, *y1;
    cudaMalloc(&W, hW.size());
    cudaMalloc(&aq, haq.size());
    cudaMalloc(&ad, had.size() * 4);
    cudaMalloc(&y0, (size_t)m * 2 * out_f * 4);
    cudaMalloc(&y1, (size_t)m * 2 * out_f * 4);
    cudaMemcpy(W, hW.data(), hW.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(aq, haq.data(), haq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ad, had.data(), had.size() * 4, cudaMemcpyHostToDevice);
    const size_t ny = (size_t)m * out_f;
    std::vector<float> r0(ny), r1(ny);
    auto v0 = [&] { k_tok<4, 2, 1024, 2><<<m / 4, 128>>>(W, aq, ad, y0, in_f, out_f, m); };
    float t0 = timeit(v0);
    cudaMemcpy(r0.data(), y0, ny * 4, cudaMemcpyDeviceToHost);
    printf("V0 t4 (shipped)            %.4f ms\n", t0);
    auto check = [&](const char* name, float t) {
        cudaMemcpy(r1.data(), y1, ny * 4, cudaMemcpyDeviceToHost);
        size_t bad = 0;
        for (size_t i = 0; i < ny; i++) bad += memcmp(&r0[i], &r1[i], 4) != 0;
        printf("%-26s %.4f ms  (%.2fx)  bitdiff=%zu %s\n", name, t, t0 / t, bad, bad ? "FAIL" : "OK");
    };
    cudaMemset(y1, 0xff, ny * 4);
    check("V1 unroll 4", timeit([&] { k_tok<4, 2, 1024, 4><<<m / 4, 128>>>(W, aq, ad, y1, in_f, out_f, m); }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V1b unroll 8", timeit([&] { k_tok<4, 2, 1024, 8><<<m / 4, 128>>>(W, aq, ad, y1, in_f, out_f, m); }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V2 rows/16 per block", timeit([&] {
        k_tok<4, 2, 16, 2><<<dim3(m / 4, out_f / 16), 128>>>(W, aq, ad, y1, in_f, out_f, m);
    }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V2b rows/8 per block", timeit([&] {
        k_tok<4, 2, 8, 2><<<dim3(m / 4, out_f / 8), 128>>>(W, aq, ad, y1, in_f, out_f, m);
    }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V2c rows/24 unroll 4", timeit([&] {
        k_tok<4, 2, 24, 4><<<dim3(m / 4, out_f / 24), 128>>>(W, aq, ad, y1, in_f, out_f, m);
    }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V3 t8", timeit([&] { k_tok<8, 2, 1024, 2><<<m / 8, 128>>>(W, aq, ad, y1, in_f, out_f, m); }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V3b t2", timeit([&] { k_tok<2, 2, 1024, 2><<<m / 2, 128>>>(W, aq, ad, y1, in_f, out_f, m); }));
    cudaMemset(y1, 0xff, ny * 4);
    check("V3c t2 rows/16", timeit([&] {
        k_tok<2, 2, 16, 2><<<dim3(m / 2, out_f / 16), 128>>>(W, aq, ad, y1, in_f, out_f, m);
    }));
    cudaError_t e = cudaDeviceSynchronize();
    printf("cuda: %s\n", cudaGetErrorString(e));
    return 0;
}
