// f8f4v2_overlap.cu: price lever (a), the next GEMM's image pass on a side stream under the
// current GEMM, against the engine's serial order (image then GEMM on one stream).
// A GDN-layer sequence of the Qwen3.8-27B trunk (attn_qkv 5120x10240, attn_gate 5120x6144,
// ssm_out 6144x5120, ffn_gate / ffn_up 5120x17408, ffn_down 17408x5120) repeated over LAYERS
// layers with distinct synthetic NVFP4 weights (split-plane layout), T rows of v2-layout e4m3
// activation, the shipped v2 tile <256,128,8,2,64,3,1> at raster group 4.
//   serial : image(W_i) -> gemm(W_i), one stream, one pooled image buffer (the engine today)
//   overlap: image(W_{i+1}) on stream 2 into the other of two image buffers, ordered after
//            gemm(W_{i-1}) (which read that buffer) and before gemm(W_{i+1}) by events
// Both orders compute the same outputs; the last GEMM's output is compared bit for bit.
// build: nvcc -gencode arch=compute_121a,code=sm_121a -O3 -std=c++17 --expt-relaxed-constexpr
//   -I crates/memra-engine/cu -o f8f4v2_overlap f8f4v2_overlap.cu crates/memra-engine/cu/mmq_nvfp4_f8f4.cu
// usage: f8f4v2_overlap T [layers] [reps]
#include "mmq_nvfp4_w4a8.cu"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

extern "C" int memra_mmq_nvfp4_f8f4_v2_quantize_act(const float * x, void * vy, int in_f, int n_tokens,
                                                    int64_t s01, cudaStream_t st);
extern "C" size_t memra_mmq_nvfp4_f8f4_v2_act_bytes(int in_f, int n_tokens);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct Shape { int in_f, out_f; };
struct Weight { char * w; int in_f, out_f; };

static void image(const Weight & W, uint8_t * img, cudaStream_t st) {
    const int Mp = f8f4v2_mp(W.out_f), nblk = W.in_f / QK_NVFP4;
    uint8_t * vals = img; float * scales = (float *) (img + (size_t) Mp * W.in_f);
    const char * W_sc = W.w + (size_t) W.out_f * nblk * 32;
    const dim3 grid((nblk + 31) / 32, Mp / 8), block(32, 8);
    f8f4v2_image_kernel<true><<<grid, block, 0, st>>>(W.w, W_sc, vals, scales, W.out_f, Mp, nblk);
}

static void gemm(const Weight & W, const uint8_t * img, const void * act, float * y, int T, cudaStream_t st) {
    const int Mp = f8f4v2_mp(W.out_f);
    const uint8_t * vals = img; const float * scales = (const float *) (img + (size_t) Mp * W.in_f);
    const uint8_t * a = (const uint8_t *) act; const float * as = (const float *) (a + (size_t) T * W.in_f);
    const int Tp = (T + 255) / 256 * 256;
    const cudaError_t e = f8f4v2_launch<256, 128, 8, 2, 64, 3, 1>(vals, scales, a, as, y, W.out_f, Mp, W.in_f, T, Tp, 1.0f, 4, st);
    CK(e);
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s T [layers] [reps]\n", argv[0]); return 2; }
    const int T = atoi(argv[1]);
    const int layers = argc > 2 ? atoi(argv[2]) : 4;
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    const Shape layer[] = {{5120, 10240}, {5120, 6144}, {6144, 5120}, {5120, 17408}, {5120, 17408}, {17408, 5120}};
    const int per = sizeof(layer) / sizeof(layer[0]);
    std::mt19937 rng(77);
    std::vector<Weight> ws;
    size_t img_bytes = 0, flop_total = 0;
    for (int l = 0; l < layers; ++l) {
        for (int k = 0; k < per; ++k) {
            const Shape s = layer[k];
            const size_t nb = (size_t) s.out_f * (s.in_f / 64) * 36;
            std::vector<uint8_t> h(nb);
            for (auto & b : h) { b = (uint8_t) (rng() & 0xFF); }
            uint8_t * sc = h.data() + (size_t) s.out_f * (s.in_f / 64) * 32;
            for (size_t i = 0; i < (size_t) s.out_f * (s.in_f / 64) * 4; ++i) { sc[i] = (uint8_t) (0x18 + rng() % 0x40); }
            Weight W{nullptr, s.in_f, s.out_f};
            CK(cudaMalloc(&W.w, nb)); CK(cudaMemcpy(W.w, h.data(), nb, cudaMemcpyHostToDevice));
            ws.push_back(W);
            img_bytes = std::max(img_bytes, memra_mmq_nvfp4_f8f4_v2_img_bytes(s.in_f, s.out_f));
            flop_total += 2ull * s.in_f * s.out_f * T;
        }
    }
    // v2-layout activations for in_f 5120 / 6144 / 17408
    void * act[3]; const int ins[3] = {5120, 6144, 17408};
    for (int i = 0; i < 3; ++i) {
        std::vector<float> x((size_t) T * ins[i]);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : x) { v = nd(rng); }
        float * dx; CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&act[i], memra_mmq_nvfp4_f8f4_v2_act_bytes(ins[i], T)));
        if (memra_mmq_nvfp4_f8f4_v2_quantize_act(dx, act[i], ins[i], T, ins[i], 0)) { printf("quantize failed\n"); return 1; }
        CK(cudaFree(dx));
    }
    auto act_for = [&](int in_f) { return in_f == 5120 ? act[0] : (in_f == 6144 ? act[1] : act[2]); };
    uint8_t * img[2]; CK(cudaMalloc(&img[0], img_bytes)); CK(cudaMalloc(&img[1], img_bytes));
    float * y[2]; CK(cudaMalloc(&y[0], (size_t) T * 17408 * 4)); CK(cudaMalloc(&y[1], (size_t) T * 17408 * 4));
    cudaStream_t s1, s2; CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    const int n = (int) ws.size();
    std::vector<cudaEvent_t> g(n), im(n);
    for (int i = 0; i < n; ++i) { CK(cudaEventCreateWithFlags(&g[i], cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&im[i], cudaEventDisableTiming)); }

    auto serial = [&] {
        for (int i = 0; i < n; ++i) {
            image(ws[i], img[0], s1);
            gemm(ws[i], img[0], act_for(ws[i].in_f), y[i & 1], T, s1);
        }
    };
    auto overlap = [&] {
        image(ws[0], img[0], s1);
        for (int i = 0; i < n; ++i) {
            if (i + 1 < n) {
                // buffer (i+1)&1 was last read by gemm i-1
                if (i >= 1) { CK(cudaStreamWaitEvent(s2, g[i - 1], 0)); }
                image(ws[i + 1], img[(i + 1) & 1], s2);
                CK(cudaEventRecord(im[i + 1], s2));
            }
            if (i >= 1) { CK(cudaStreamWaitEvent(s1, im[i], 0)); }
            gemm(ws[i], img[i & 1], act_for(ws[i].in_f), y[i & 1], T, s1);
            CK(cudaEventRecord(g[i], s1));
        }
    };
    auto timed = [&](const std::function<void()> & f) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        f(); CK(cudaDeviceSynchronize());
        std::vector<float> v;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(a, s1)); f(); CK(cudaStreamSynchronize(s2)); CK(cudaEventRecord(b, s1)); CK(cudaEventSynchronize(b));
            float ms; cudaEventElapsedTime(&ms, a, b); v.push_back(ms);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const size_t last = (size_t) T * ws[n - 1].out_f;
    std::vector<float> ys(last), yo(last);
    float ts = timed(serial);
    CK(cudaMemcpy(ys.data(), y[(n - 1) & 1], last * 4, cudaMemcpyDeviceToHost));
    float to = timed(overlap);
    CK(cudaMemcpy(yo.data(), y[(n - 1) & 1], last * 4, cudaMemcpyDeviceToHost));
    float ts2 = timed(serial);
    size_t bad = 0;
    for (size_t i = 0; i < last; ++i) { bad += std::memcmp(&ys[i], &yo[i], 4) != 0; }
    // image-only and gemm-only sums for the attribution
    auto images_only = [&] { for (int i = 0; i < n; ++i) { image(ws[i], img[0], s1); } };
    auto gemms_only = [&] { for (int i = 0; i < n; ++i) { gemm(ws[i], img[0], act_for(ws[i].in_f), y[i & 1], T, s1); } };
    float ti = timed(images_only), tg = timed(gemms_only);
    printf("T=%d layers=%d gemms=%d reps=%d (median)\n", T, layers, n, reps);
    printf("images only   %8.3f ms\n", ti);
    printf("gemms only    %8.3f ms  %6.1f TFLOPS\n", tg, flop_total / tg / 1e9);
    printf("serial        %8.3f ms, %8.3f ms (A B order)\n", ts, ts2);
    printf("overlap       %8.3f ms  saves %.3f ms of %.3f ms image time (%.1f%% of serial)\n",
           to, (ts + ts2) / 2 - to, ti, 100.0 * ((ts + ts2) / 2 - to) / ((ts + ts2) / 2));
    printf("last output bit-mismatch serial vs overlap: %zu/%zu %s\n", bad, last, bad ? "FAIL" : "OK");
    return 0;
}
