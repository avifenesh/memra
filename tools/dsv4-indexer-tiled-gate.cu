// Actual production launcher, scalar witness, CPU anchors, masks and write guards.
// Compile -fmad=false -Xcompiler=-ffp-contract=off, exactly like the engine TU.
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void launch_check(int rc) {
    if (rc) throw std::runtime_error("launch " + std::to_string(rc));
}
template <typename T> struct Dev {
    T* p;
    explicit Dev(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) {
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};
static uint32_t rng(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
static float mirror(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& w, int t, int j, float scale) {
    float total = 0;
    for (int h = 0; h < 64; ++h) {
        float dot = 0;
        for (int x = 0; x < 128; ++x) dot += q[(t * 64 + h) * 128 + x] * k[j * 128 + x];
        float ws = w[t * 64 + h] * scale;
        total += std::fmax(dot, 0.0f) * ws;
    }
    return total;
}
static size_t cell(int s, int nb, int pos0, int lim0, bool zero, bool teeth, bool perf) {
    const int n = s * nb, guard = 17;
    std::vector<float> q(s * 64 * 128), k(nb * 128), w(s * 64);
    uint32_t seed = 0x735d120;
    // General non-dyadic floats detect accidental FMA; zeros force score ties.
    for (auto& x : q) x = zero ? 0.0f : (int(rng(seed) % 20001) - 10000) / 3001.0f;
    for (auto& x : k) x = (int(rng(seed) % 20001) - 10000) / 5003.0f;
    for (auto& x : w) x = (int(rng(seed) % 20001) - 10000) / 7001.0f;
    Dev<float> dq(q), dk(k), dw(w), ref(n + guard), out(n + guard);
    constexpr float scale = 0.011048543f;
    auto run = [&](bool tiled) {
        if (tiled) return memra_dsv4_indexer_score_tiled(dq.p, dk.p, dw.p, scale,
            out.p, s, 64, 128, nb, 4, lim0, pos0, nullptr);
        if (pos0 >= 0) return memra_dsv4_indexer_score_f32acc_pos_m(dq.p, dk.p, dw.p, scale,
            ref.p, s, 64, 128, nb, 4, pos0, nullptr);
        return memra_dsv4_indexer_score_f32acc(dq.p, dk.p, dw.p, scale,
            ref.p, s, 64, 128, nb, 4, lim0, nullptr);
    };
    check(cudaMemset(ref.p, 0xa5, (n + guard) * 4));
    check(cudaMemset(out.p, 0xa5, (n + guard) * 4));
    launch_check(run(false)); launch_check(run(true)); check(cudaDeviceSynchronize());
    if (s == 1 && nb == 129) {
        cudaStream_t probe; cudaGraph_t graph;
        check(cudaStreamCreate(&probe));
        check(cudaStreamBeginCapture(probe, cudaStreamCaptureModeThreadLocal));
        launch_check(memra_dsv4_indexer_score_tiled(dq.p, dk.p, dw.p, scale,
            out.p, s, 64, 128, nb, 4, lim0, pos0, probe));
        check(cudaStreamEndCapture(probe, &graph));
        size_t count = 0; check(cudaGraphGetNodes(graph, nullptr, &count));
        if (count != 1) throw std::runtime_error("unexpected tiled graph node count");
        cudaGraphNode_t node; check(cudaGraphGetNodes(graph, &node, &count));
        cudaKernelNodeParams params{}; check(cudaGraphKernelNodeGetParams(node, &params));
        if (params.func != (void*)dsv4_indexer_score_tiled_kernel ||
            params.gridDim.x != 2 || params.gridDim.y != 1 || params.blockDim.x != 128)
            throw std::runtime_error("tiled launcher did not engage expected kernel/geometry");
        check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(probe));
        puts("ENGAGED tiled kernel, grid 2x1, block 128");
    }
    std::vector<float> a(n + guard), b(n + guard);
    check(cudaMemcpy(a.data(), ref.p, a.size() * 4, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(b.data(), out.p, b.size() * 4, cudaMemcpyDeviceToHost));
    if (teeth) b[0] = std::nextafter(b[0], INFINITY);
    for (int t = 0; t < s; ++t) {
        int lim = pos0 >= 0 ? (pos0 + t + 1) / 4 : (lim0 >= 0 ? lim0 : (t + 1) / 4);
        for (int j = 0; j < nb; ++j) {
            int i = t * nb + j;
            if ((j < lim && !std::isfinite(a[i])) || (j >= lim && a[i] != -INFINITY))
                throw std::runtime_error("invalid reference numeric/mask");
            if (bits(a[i]) != bits(b[i])) throw std::runtime_error("score mismatch at " + std::to_string(i));
        }
        for (int j : {0, std::min(nb, lim) - 1}) {
            if (j >= 0 && j < nb && j < lim && bits(a[t * nb + j]) != bits(mirror(q, k, w, t, j, scale)))
                throw std::runtime_error("CPU anchor mismatch");
        }
    }
    for (int i = n; i < n + guard; ++i)
        if (bits(a[i]) != 0xa5a5a5a5u || bits(b[i]) != 0xa5a5a5a5u)
            throw std::runtime_error("tail overwritten");
    if (perf) {
        cudaEvent_t start, end; check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        for (int repeat = 0; repeat < 5; ++repeat) {
            float ms[2];
            for (int ix = 0; ix < 2; ++ix) {
                int arm = (ix + repeat) % 2;
                check(cudaEventRecord(start));
                for (int r = 0; r < 5; ++r) launch_check(run(arm));
                check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
                check(cudaEventElapsedTime(&ms[arm], start, end));
            }
            printf("TIME s=%d nb=%d repeat=%d scalar_us=%.3f tiled_us=%.3f\n", s, nb, repeat, ms[0]*200, ms[1]*200);
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    printf("EXACT s=%d nb=%d pos0=%d lim0=%d zero=%d values=%d\n", s, nb, pos0, lim0, zero, n);
    return n;
}
int main(int argc, char** argv) {
    try {
        const bool teeth = argc > 1 && std::string(argv[1]) == "--teeth";
        const bool perf = argc > 1 && std::string(argv[1]) == "--perf";
        size_t compared = 0;
        for (int nb : {1, 127, 128, 129, 4096, 4103, 65536, 262147}) {
            compared += cell(1, nb, -1, nb, false, teeth, perf);
            compared += cell(5, nb, std::min(1048576, std::max(0, nb * 4 - 7)), -1, false, false, false);
        }
        for (int s : {1, 5, 8, 32, 64}) {
            compared += cell(s, 129, 0, -1, false, false, false);
            compared += cell(s, 129, -1, 127, true, false, false);
        }
        // Invalid shapes must be rejected before dereferencing the null buffers.
        if (memra_dsv4_indexer_score_tiled(nullptr, nullptr, nullptr, 1, nullptr,
            1, 32, 128, 10, 4, 10, -1, nullptr) != 40009) throw std::runtime_error("shape refusal absent");
        printf("PASS comparisons=%zu\n", compared);
    } catch (const std::exception& e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
