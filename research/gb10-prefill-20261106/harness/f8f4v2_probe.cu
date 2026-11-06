// f8f4v2_probe.cu: v1 F8F4 tile against the v2 image + pipelined GEMM on synthetic NVFP4 weights.
// Per shape and token count: bit compare of every output (v2 config x raster vs v1), GEMM-only
// time of each config, the image pre-pass time, and v1's time. Same TU as the engine kernels
// (#include), so every config can be launched without a rebuild of the engine.
// build (on the GB10): nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17
//   --expt-relaxed-constexpr -I crates/memra-engine/cu -o f8f4v2_probe f8f4v2_probe.cu
//   crates/memra-engine/cu/mmq_nvfp4_f8f4.cu
// usage: f8f4v2_probe IN OUT T [reps] [rp] [cfg index or -1] [G or 0 = sweep]
#include "mmq_nvfp4_w4a8.cu"
#include <cstdio>
#include <vector>
#include <random>
#include <functional>

extern "C" void memra_mmq_nvfp4_f8f4_quantize_act(const float * x, void * vy, int in_f, int n_tokens,
                                                  int64_t s01, cudaStream_t st);
extern "C" size_t memra_mmq_nvfp4_f8f4_act_bytes(int in_f, int n_tokens);
extern "C" void memra_mmq_nvfp4_f8f4_v2_quantize_act(const float * x, void * vy, int in_f, int n_tokens,
                                                     int64_t s01, cudaStream_t st);
extern "C" size_t memra_mmq_nvfp4_f8f4_v2_act_bytes(int in_f, int n_tokens);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static float time_ms(cudaStream_t st, int reps, const std::function<void()> & f) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    f(); CK(cudaStreamSynchronize(st));
    cudaEventRecord(a, st);
    for (int r = 0; r < reps; ++r) { f(); }
    cudaEventRecord(b, st); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

struct Cfg { const char * name; cudaError_t (*fn)(const uint8_t *, const float *, const uint8_t *, const float *, float *, int, int, int, int, int, float, int, cudaStream_t); };

int main(int argc, char ** argv) {
    if (argc < 4) { printf("usage: %s IN OUT T [reps] [rp]\n", argv[0]); return 2; }
    const int in_f = atoi(argv[1]), out_f = atoi(argv[2]), T = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 10;
    const int rp = argc > 5 ? atoi(argv[5]) : 0;
    const int only_cfg = argc > 6 ? atoi(argv[6]) : -1;
    const int only_g = argc > 7 ? atoi(argv[7]) : 0;
    cudaStream_t st; CK(cudaStreamCreate(&st));
    const int nblk = in_f / 64;
    std::mt19937 rng(1234 + in_f + out_f);
    // NVFP4 weight: GGUF 36 B blocks (rp=0) or split planes (rp=1); scale bytes over the UE4M3
    // range incl. 0 and the 0x7F NaN code, random nibbles.
    const size_t wbytes = (size_t) out_f * nblk * 36;
    std::vector<uint8_t> w(wbytes);
    for (size_t i = 0; i < wbytes; ++i) { w[i] = (uint8_t) (rng() & 0xFF); }
    if (rp == 0) {
        for (size_t b = 0; b < (size_t) out_f * nblk; ++b) {
            for (int s = 0; s < 4; ++s) {
                const uint32_t u = rng() % 100;
                w[b * 36 + s] = u == 0 ? 0 : (u == 1 ? 0x7F : (uint8_t) (0x18 + rng() % 0x40));
            }
        }
    } else {
        uint8_t * sc = w.data() + (size_t) out_f * nblk * 32;
        for (size_t b = 0; b < (size_t) out_f * nblk * 4; ++b) {
            const uint32_t u = rng() % 100;
            sc[b] = u == 0 ? 0 : (u == 1 ? 0x7F : (uint8_t) (0x18 + rng() % 0x40));
        }
    }
    std::vector<float> x((size_t) T * in_f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        const int tok = (int) (i / in_f);
        x[i] = nd(rng) * powf(10.0f, (float) (tok % 5) - 2.0f) * ((i % 97) == 0 ? 40.0f : 1.0f);
    }
    char * dW; float * dx; void * dact; void * dact2; float * y1; float * y2; uint8_t * img;
    CK(cudaMalloc(&dW, wbytes)); CK(cudaMemcpy(dW, w.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const size_t ab = memra_mmq_nvfp4_f8f4_act_bytes(in_f, T);
    CK(cudaMalloc(&dact, ab));
    const size_t ab2 = memra_mmq_nvfp4_f8f4_v2_act_bytes(in_f, T);
    CK(cudaMalloc(&dact2, ab2));
    const int Tp = (T + 255) / 256 * 256;
    const uint8_t * a2 = (const uint8_t *) dact2;
    const float * a2s = (const float *) (a2 + (size_t) T * in_f);
    CK(cudaMalloc(&y1, (size_t) T * out_f * 4)); CK(cudaMalloc(&y2, (size_t) T * out_f * 4));
    const size_t ib = memra_mmq_nvfp4_f8f4_v2_img_bytes(in_f, out_f);
    CK(cudaMalloc(&img, ib));
    memra_mmq_nvfp4_f8f4_quantize_act(dx, dact, in_f, T, in_f, st);
    memra_mmq_nvfp4_f8f4_v2_quantize_act(dx, dact2, in_f, T, in_f, st);
    CK(cudaStreamSynchronize(st));

    const double flop = 2.0 * in_f * (double) out_f * T;
    // v1
    auto v1 = [&] {
        int rc = memra_mmq_nvfp4_f8f4_preq(dW, dact, y1, in_f, out_f, T, st, 1.0f, rp);
        if (rc) { printf("v1 rc=%d\n", rc); exit(1); }
    };
    const float t1 = time_ms(st, reps, v1);
    printf("shape in=%d out=%d T=%d rp=%d\n", in_f, out_f, T, rp);
    printf("v1          %8.3f ms  %6.1f TFLOPS\n", t1, flop / t1 / 1e9);
    std::vector<float> h1((size_t) T * out_f), h2((size_t) T * out_f);
    CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));

    // image pre-pass
    const int Mp = f8f4v2_mp(out_f);
    uint8_t * vals = img; float * scales = (float *) (img + (size_t) Mp * in_f);
    const char * W_sc = dW + (size_t) out_f * nblk * 32;
    auto image = [&] {
        const dim3 grid((nblk + 31) / 32, Mp / 8), block(32, 8);
        if (rp) { f8f4v2_image_kernel<true><<<grid, block, 0, st>>>(dW, W_sc, vals, scales, out_f, Mp, nblk); }
        else    { f8f4v2_image_kernel<false><<<grid, block, 0, st>>>(dW, W_sc, vals, scales, out_f, Mp, nblk); }
    };
    const float ti = time_ms(st, reps, image);
    const double ibytes = (double) out_f * nblk * 36 + (double) ib;
    printf("image       %8.3f ms  %6.1f GB/s (read+write)\n", ti, ibytes / ti / 1e6);

    // full entry (image + default config) as the engine calls it
    auto full = [&] {
        int rc = memra_mmq_nvfp4_f8f4_v2_preq(dW, dact2, y2, in_f, out_f, T, img, st, 1.0f, rp);
        if (rc) { printf("v2 rc=%d\n", rc); exit(1); }
    };
    const float tf = time_ms(st, reps, full);
    CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < h1.size(); ++i) { bad += __builtin_memcmp(&h1[i], &h2[i], 4) != 0; }
    printf("v2 entry    %8.3f ms  %6.1f TFLOPS incl. image   bit-mismatch %zu/%zu %s\n",
           tf, flop / tf / 1e9, bad, h1.size(), bad ? "FAIL" : "OK");

    image(); CK(cudaStreamSynchronize(st));
    const Cfg cfgs[] = {
        {"c0 128x128 w4x2 k64 s2 2cta", f8f4v2_launch<128, 128, 4, 2, 64, 2, 2>},
        {"c1 128x128 w4x2 k32 s3 2cta", f8f4v2_launch<128, 128, 4, 2, 32, 3, 2>},
        {"c2 128x128 w4x4 k64 s4 1cta", f8f4v2_launch<128, 128, 4, 4, 64, 4, 1>},
        {"c3 256x128 w8x2 k32 s4 1cta", f8f4v2_launch<256, 128, 8, 2, 32, 4, 1>},
        {"c4 128x128 w4x2 k64 s4 1cta", f8f4v2_launch<128, 128, 4, 2, 64, 4, 1>},
        {"c5 256x128 w8x2 k64 s2 1cta", f8f4v2_launch<256, 128, 8, 2, 64, 2, 1>},
        {"c6 256x128 w8x2 k64 s3 1cta", f8f4v2_launch<256, 128, 8, 2, 64, 3, 1>},
        {"c7 256x128 w8x2 k32 s5 1cta", f8f4v2_launch<256, 128, 8, 2, 32, 5, 1>},
        {"c8 256x128 w8x2 k32 s6 1cta", f8f4v2_launch<256, 128, 8, 2, 32, 6, 1>},
        {"c9 256x128 w8x2 k32 s7 1cta", f8f4v2_launch<256, 128, 8, 2, 32, 7, 1>},
        {"c10 128x256 w4x4 k64 s3 1cta", f8f4v2_launch<128, 256, 4, 4, 64, 3, 1>},
    };
    const int Gs[] = {1, 4, 8, 16, 32, 1 << 20};
    int ci = -1;
    for (const Cfg & c : cfgs) {
        ++ci;
        if (only_cfg >= 0 && ci != only_cfg) { continue; }
        for (int G : Gs) {
            if (only_g > 0 && G != only_g) { continue; }
            CK(cudaMemset(y2, 0xFF, (size_t) T * out_f * 4));
            auto run = [&] {
                cudaError_t e = c.fn(vals, scales, a2, a2s, y2, out_f, Mp, in_f, T, Tp, 1.0f, G, st);
                if (e != cudaSuccess) { printf("%s G=%d launch %s\n", c.name, G, cudaGetErrorString(e)); exit(1); }
            };
            const float tg = time_ms(st, reps, run);
            CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
            size_t nb = 0;
            for (size_t i = 0; i < h1.size(); ++i) { nb += __builtin_memcmp(&h1[i], &h2[i], 4) != 0; }
            printf("%-30s G=%-7d %8.3f ms  %6.1f TFLOPS gemm-only  (+image %6.1f TFLOPS)  bit-mismatch %zu %s\n",
                   c.name, G, tg, flop / tg / 1e9, flop / (tg + ti) / 1e9, nb, nb ? "FAIL" : "OK");
        }
    }
    return 0;
}
