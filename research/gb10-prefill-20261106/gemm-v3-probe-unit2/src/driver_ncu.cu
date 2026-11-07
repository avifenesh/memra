// ncu driver: one shape (argv[1]: 0 gate+up 34816x5120, 1 down 5120x17408, 2 wqkv 10240x5120,
// 3 ssm_out 5120x6144), T = 4,096, the shipped v2 tile <256,128,8,2,64,3,1,0> in PROBE_MODE
// (0 shipped, 1 no scale FMAs, 2 no loads after the prologue, 3 no loads and no barriers);
// argv[2] launches (default 3), the median time printed. Random e4m3 data, random scales.
#include "mmq_probe.cu"
#include <cstdio>
#include <vector>
#include <algorithm>
extern "C" void memra_mmq_nvfp4_f8f4_quantize_act(const float *, void *, int, int, int64_t, cudaStream_t) {}
static uint32_t rng = 7;
static uint8_t e4m3_rand() { rng = rng * 1664525u + 1013904223u; uint8_t b = (uint8_t)(rng >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; return b; }
static float sc_rand() { rng = rng * 1664525u + 1013904223u; return 0.004f + (rng >> 8) * (1.0f / 16777216.0f) * 0.02f; }
int main(int argc, char** argv) {
    const int T = 4096;
    const int si = argc > 1 ? atoi(argv[1]) : 0;
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    struct Shape { const char* name; int out_f, K; } shapes[] = {
        {"gate+up 34816x5120", 34816, 5120}, {"down 5120x17408", 5120, 17408},
        {"wqkv 10240x5120", 10240, 5120}, {"ssm_out 5120x6144", 5120, 6144}};
    const Shape sh = shapes[si];
    const int Mp = f8f4v2_mp(sh.out_f), K = sh.K, Tp = (T + 255) / 256 * 256;
    std::vector<uint8_t> hv((size_t)Mp * K), ha((size_t)T * K);
    for (auto& b : hv) b = e4m3_rand();
    for (auto& b : ha) b = e4m3_rand();
    std::vector<float> hs((size_t)(K / 32) * Mp), hsa((size_t)(K / 32) * Tp);
    for (auto& x : hs) x = sc_rand();
    for (auto& x : hsa) x = sc_rand();
    uint8_t *v, *a; float *sc, *sa, *y;
    cudaMalloc(&v, hv.size()); cudaMalloc(&a, ha.size());
    cudaMalloc(&sc, hs.size() * 4); cudaMalloc(&sa, hsa.size() * 4); cudaMalloc(&y, (size_t)T * sh.out_f * 4);
    cudaMemcpy(v, hv.data(), hv.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(a, ha.data(), ha.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(sc, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sa, hsa.data(), hsa.size() * 4, cudaMemcpyHostToDevice);
    auto f = [&] { return f8f4v2_launch<256, 128, 8, 2, 64, 3, 1, 0>(v, sc, a, sa, y, sh.out_f, Mp, K, T, Tp, 0.83f, 4, 0); };
    std::vector<float> ts; cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int r = 0; r < reps; r++) { cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms); }
    std::sort(ts.begin(), ts.end());
    const double ms = ts[ts.size() / 2], macs = (double)sh.out_f * K * T;
    printf("mode %d %-22s %7.3f ms  %6.1f TMAC/s  %6.1f MAC/clk/SM at 2200 MHz (n=%d)\n", PROBE_MODE, sh.name, ms, macs / ms / 1e9, macs / (ms * 1e-3) / (48 * 2.2e9), reps);
    printf("cuda: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
    return 0;
}
