// Lever 6 P1 (round 4): v3 against the persistent v3 mainloop (mmq_v3p.cu), both through the engine entries
// with v3 = 2 (forced, no v2 fallback); the shapes of drv_p0.cu.
// The engine's F8F4 v2 entries with v3 = 0 / 1 timed in one process (image pre-pass included, the same
// in both arms), for the launches v3 runs: EPI 0 plain (down, wqkv, ssm_out), EPI 2 residual (down).
// Built once per TU (two TUs whose v2 SASS is identical) to compare their v3 mainloops. Outputs compared byte for byte. Random NVFP4 bytes with
// finite block scales, random v2-layout activation. usage: drv_epi [T] [reps] [order: 01 or 10]
#include "mmq_v3p.cu"
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
int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 11;
    const bool rev = argc > 3 && strcmp(argv[3], "10") == 0;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    printf("T=%d reps=%d order=%s (median, image pass included)\n", T, reps, rev ? "v3p,v3" : "v3,v3p");
    struct S { const char * name; int epi, rows, in_f; } shapes[] = {
        {"down 5120x17408", 0, 5120, 17408},
        {"down resid 5120x17408", 2, 5120, 17408}, {"wqkv 10240x5120", 0, 10240, 5120},
        {"ssm_out 5120x6144", 0, 5120, 6144}, {"gateup silu 2x17408x5120", 1, 17408, 5120}};
    const int only = argc > 4 ? atoi(argv[4]) : -1;   // run one shape (index), -1 = all
    int si = -1;
    for (auto & s : shapes) {
        ++si;
        if (only >= 0 && si != only) continue;
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
            f8f4v3_persist = v3;
            if (s.epi == 1) rc = memra_mmq_nvfp4_f8f4_v2_gateup_silu(dwg, dwu, da, dout[v3], s.in_f, s.rows, T, dimg, nullptr, 0.83f, 0.91f, 0, 2);
            else if (s.epi == 2) rc = memra_mmq_nvfp4_f8f4_v2_preq_resid(dwg, da, dres, dy[v3], s.in_f, s.rows, T, dimg, nullptr, 0.83f, 0, 2);
            else rc = memra_mmq_nvfp4_f8f4_v2_preq(dwg, da, dy[v3], s.in_f, s.rows, T, dimg, nullptr, 0.83f, 0, 2);
            if (rc) printf("rc=%d\n", rc);
        };
        double t[2];
        for (int pass = 0; pass < 2; ++pass) { const int v3 = rev ? 1 - pass : pass; t[v3] = timeit([&] { call(v3); }, reps); }
        std::vector<uint8_t> o0(out_bytes), o1(out_bytes);
        cudaMemcpy(o0.data(), dout[0], out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(o1.data(), dout[1], out_bytes, cudaMemcpyDeviceToHost);
        const size_t bad = memcmp(o0.data(), o1.data(), out_bytes) ? 1 : 0;
        printf("%-28s v3 %8.3f ms  v3p %8.3f ms  %+6.2f%%  bytes %s\n", s.name, t[0], t[1], 100.0 * (t[0] / t[1] - 1.0), bad ? "DIFFER" : "identical");
        cudaFree(dwg); if (dwu) cudaFree(dwu); cudaFree(da); cudaFree(dimg); cudaFree(dout[0]); cudaFree(dout[1]); if (dres) cudaFree(dres);
    }
    printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
