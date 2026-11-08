// The fused gate/up silu launch (EPI 1) on the v2 mainloop and on the v3 mainloop with the CTA-local TMA
// destination, timed in one process (image pre-pass included in both), outputs compared byte for byte.
// The entry keeps v2, so gateup() below is the entry's body with a v3 switch. Outputs compared byte for byte. Random NVFP4 bytes with
// finite block scales, random v2-layout activation. usage: drv_epi [T] [reps] [order: 01 or 10]
#include "mmq_nvfp4_w4a8.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
extern "C" void memra_mmq_nvfp4_f8f4_quantize_act(const float *, void *, int, int, int64_t, cudaStream_t) {}
// mmq_nvfp4_f8f4.cu's sizing (Tp = tokens rounded up to 256)
static size_t memra_mmq_nvfp4_f8f4_v2_act_bytes(int in_f, int n_tokens) {
    return (size_t) n_tokens * in_f + (size_t) (in_f / 32) * (size_t) ((n_tokens + 255) / 256 * 256) * sizeof(float);
}
static uint32_t rng = 11;
static uint8_t rb() { rng = rng * 1664525u + 1013904223u; return (uint8_t) (rng >> 24); }
static void fill_w(std::vector<uint8_t> & w, int rows, int in_f) {
    const int nblk = in_f / QK_NVFP4;
    w.resize((size_t) rows * nblk * sizeof(block_nvfp4));
    block_nvfp4 * b = (block_nvfp4 *) w.data();
    for (size_t i = 0; i < (size_t) rows * nblk; ++i) {
        for (auto & q : b[i].qs) q = rb();
        for (auto & d : b[i].d) d = (uint8_t) (0x30 + (rb() & 0x0f));   // finite ue4m3 scales
    }
}
static void fill_act(std::vector<uint8_t> & a, int T, int in_f) {
    const size_t n = memra_mmq_nvfp4_f8f4_v2_act_bytes(in_f, T);
    a.resize(n);
    const size_t nv = (size_t) T * in_f;
    for (size_t i = 0; i < nv; ++i) { uint8_t v = rb(); if ((v & 0x7f) == 0x7f) v ^= 1; a[i] = v; }
    float * sc = (float *) (a.data() + nv);
    for (size_t i = 0; i < (n - nv) / 4; ++i) sc[i] = 0.004f + (rb() & 0xff) * 1e-5f;
}
template <typename F> static double timeit(F f, int reps) {
    std::vector<float> ts; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    f(); cudaDeviceSynchronize();
    for (int r = 0; r < reps; r++) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms); }
    std::sort(ts.begin(), ts.end()); return ts[ts.size() / 2];
}
static int gateup(const void * W_gate, const void * W_up, const void * act_scratch, void * act_out, int in_f, int nf,
                  int n_tokens, void * img, cudaStream_t st, float scale_gate, float scale_up, int v3) {
    const int rows = 2 * nf;
    const int Mp = f8f4v2_mp(rows);
    const int nblk = in_f / QK_NVFP4;
    uint8_t * vals = (uint8_t *) img;
    float * scales = (float *) (vals + (size_t) Mp * in_f);
    const char * Wg = (const char *) W_gate;
    const char * Wu = (const char *) W_up;
    const dim3 grid((unsigned) ((nblk + 31) / 32), (unsigned) (Mp / 8), 1);
    const dim3 block(32, 8, 1);
    f8f4v2_image_kernel<false, true><<<grid, block, 0, st>>>(Wg, Wg + (size_t) nf * nblk * 32, vals, scales, nf, Mp, nblk,
                                                             Wu, Wu + (size_t) nf * nblk * 32);
    const uint8_t * act = (const uint8_t *) act_scratch;
    const float * act_sc = (const float *) (act + (size_t) n_tokens * in_f);
    const int Tp = (n_tokens + 255) / 256 * 256;
    const cudaError_t e = v3
        ? f8f4v3_launch<256, 128, 8, 2, 64, 3, 1>(vals, scales, act, act_sc, nullptr, rows, Mp, in_f, n_tokens, Tp,
                                                  scale_gate, 4, st, (uint8_t *) act_out, scale_up)
        : f8f4v2_launch<256, 128, 8, 2, 64, 3, 1, 1>(vals, scales, act, act_sc, nullptr, rows, Mp, in_f, n_tokens, Tp,
                                                     scale_gate, 4, st, (uint8_t *) act_out, scale_up);
    return e == cudaSuccess ? 0 : 3000 + (int) e;
}
int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 11;
    const bool rev = argc > 3 && strcmp(argv[3], "10") == 0;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    printf("T=%d reps=%d order=%s (median, image pass included)\n", T, reps, rev ? "v3,v2" : "v2,v3");
    struct S { const char * name; int epi, rows, in_f; } shapes[] = {
        {"gate+up silu 2x17408x5120", 1, 17408, 5120}};
    for (auto & s : shapes) {
        std::vector<uint8_t> wg, wu, ha;
        fill_w(wg, s.rows, s.in_f);
        if (s.epi == 1) fill_w(wu, s.rows, s.in_f);
        fill_act(ha, T, s.in_f);
        uint8_t *dwg, *dwu = nullptr, *da, *dimg, *dout[2]; float *dres = nullptr, *dy[2];
        cudaMalloc(&dwg, wg.size()); cudaMemcpy(dwg, wg.data(), wg.size(), cudaMemcpyHostToDevice);
        if (s.epi == 1) { cudaMalloc(&dwu, wu.size()); cudaMemcpy(dwu, wu.data(), wu.size(), cudaMemcpyHostToDevice); }
        cudaMalloc(&da, ha.size()); cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice);
        const int img_rows = s.epi == 1 ? 2 * s.rows : s.rows;
        cudaMalloc(&dimg, memra_mmq_nvfp4_f8f4_v2_img_bytes(s.in_f, img_rows));
        const size_t out_bytes = s.epi == 1 ? memra_mmq_nvfp4_f8f4_v2_act_bytes(s.rows, T) : (size_t) T * s.rows * 4;
        for (int k = 0; k < 2; ++k) { cudaMalloc(&dout[k], out_bytes); cudaMemset(dout[k], 0, out_bytes); dy[k] = (float *) dout[k]; }
        if (s.epi == 2) { cudaMalloc(&dres, (size_t) T * s.rows * 4); cudaMemset(dres, 0, (size_t) T * s.rows * 4); }
        auto call = [&](int v3) {
            int rc = 0;
            if (s.epi == 1) rc = gateup(dwg, dwu, da, dout[v3], s.in_f, s.rows, T, dimg, nullptr, 0.71f, 1.37f, v3);
            else if (s.epi == 2) rc = memra_mmq_nvfp4_f8f4_v2_preq_resid(dwg, da, dres, dy[v3], s.in_f, s.rows, T, dimg, nullptr, 0.83f, 0, v3);
            else rc = memra_mmq_nvfp4_f8f4_v2_preq(dwg, da, dy[v3], s.in_f, s.rows, T, dimg, nullptr, 0.83f, 0, v3);
            if (rc) printf("rc=%d\n", rc);
        };
        double t[2];
        for (int pass = 0; pass < 2; ++pass) { const int v3 = rev ? 1 - pass : pass; t[v3] = timeit([&] { call(v3); }, reps); }
        std::vector<uint8_t> o0(out_bytes), o1(out_bytes);
        cudaMemcpy(o0.data(), dout[0], out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(o1.data(), dout[1], out_bytes, cudaMemcpyDeviceToHost);
        const size_t bad = memcmp(o0.data(), o1.data(), out_bytes) ? 1 : 0;
        printf("%-28s v2 %8.3f ms  v3 %8.3f ms  %+6.2f%%  bytes %s\n", s.name, t[0], t[1], 100.0 * (t[0] / t[1] - 1.0), bad ? "DIFFER" : "identical");
        cudaFree(dwg); if (dwu) cudaFree(dwu); cudaFree(da); cudaFree(dimg); cudaFree(dout[0]); cudaFree(dout[1]); if (dres) cudaFree(dres);
    }
    printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
