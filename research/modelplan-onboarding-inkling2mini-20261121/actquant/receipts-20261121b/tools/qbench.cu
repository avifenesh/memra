
// Quantizer timing, before (e7204c97a0) and after the bank-conflict fix, same harness: back-to-back
// launches, host wall time per call, best of 5 x 200. OLD selects the old source and its API.
#ifdef OLD
#include "old_actq.cu"
#else
#include "/home/avifenesh/.cache/inkling2mini/wt-actq/crates/memra-engine/cu/inkling2_actq.cu"
#endif
#include <chrono>
#include <cstdio>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda %s\n", cudaGetErrorString(e_)); return 4;} } while (0)
int main() {
    const int H = 2304, F = 1152;
    struct C { int kind, situ, rows, k; } cs[] = {{0,0,1,H},{0,0,8,H},{0,1,8,F},{0,1,64,F},{0,1,2048,F},{1,0,1,H},{1,0,8,H},{1,1,8,F},{1,1,64,F},{1,1,2048,F}};
    void *x, *v, *s, *y; float *a, *g; double *tg, *tu;
    CK(cudaMalloc(&x, 2048 * 2 * F * 2)); CK(cudaMalloc(&v, 2048 * H)); CK(cudaMalloc(&s, 2048 * H / 16)); CK(cudaMalloc(&a, 2048 * 4));
    CK(cudaMalloc(&g, 2048 * 4)); CK(cudaMalloc(&tg, 65536 * 8)); CK(cudaMalloc(&tu, 65536 * 8)); CK(cudaMalloc(&y, 1));
    std::vector<uint16_t> hx(2048 * 2 * F); uint32_t r = 1; for (auto& e : hx) { r = r * 1664525u + 1013904223u; e = 0x3f00 + (r >> 24) % 0x200 | ((r >> 8) & 0x8000); }
    CK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hg(2048, 1.0f); CK(cudaMemcpy(g, hg.data(), 2048 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(tg, 0, 65536 * 8)); CK(cudaMemset(tu, 0, 65536 * 8));
    for (auto c : cs) {
        auto call = [&]() {
#ifdef OLD
            return memra_ink2_actq(c.kind, c.situ, x, g, nullptr, c.rows, v, s, a, nullptr, c.k, 4.0f, 25.0f, 0, 0);
#else
            return memra_ink2_actq(c.kind, c.situ, x, g, nullptr, c.rows, v, s, a, nullptr, c.k, tg, tu, 0);
#endif
        };
        if (call()) { printf("rc\n"); return 5; }
        CK(cudaDeviceSynchronize());
        double best = 1e30;
        for (int t = 0; t < 5; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) call();
            CK(cudaDeviceSynchronize());
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
            if (us < best) best = us;
        }
        printf("%s %s%s rows=%d k=%d: %.2f us/call\n",
#ifdef OLD
               "before",
#else
               "after",
#endif
               c.kind ? "MXFP8" : "NVFP4", c.situ ? " + SiTU" : "", c.rows, c.k, best);
    }
    return 0;
}
