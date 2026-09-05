// Standalone gate for the actual DSV4 selected-expert kernels, with no model load.
// Build with the same -fmad=false as build.rs. The original per-expert kernel
// and a host mirror independently anchor both selected-dispatch realizations.
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void kernel_check(int rc) {
    if (rc) throw std::runtime_error("launcher rc=" + std::to_string(rc));
}
template <typename T> struct DeviceArray {
    T* p = nullptr;
    explicit DeviceArray(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) {
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { cudaFree(p); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
static uint32_t rng(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
static uint32_t bits(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u; }
static float e4m3(uint8_t code) {
    unsigned a = code & 127u;
    if (a == 127) return 0.0f;
    float v = (a >> 3) ? std::ldexp(1.0f + (a & 7) / 8.0f, int(a >> 3) - 7)
                       : (a & 7) / 512.0f;
    return code & 128 ? -v : v;
}
static float e2m1(uint8_t code) {
    static const float v[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return code & 8 ? -v[code & 7] : v[code & 7];
}
static float mirror(const std::vector<uint8_t>& a, const std::vector<float>& as,
                    const std::vector<uint8_t>& w, const std::vector<uint8_t>& sc,
                    float macro, int row, int n, int k, int col, int kind,
                    size_t woff, size_t soff) {
    float part[128] = {};
    const int gs = kind ? 32 : 16;
    for (int tid = 0; tid < 128; ++tid) {
        for (int g = tid; g < k / gs; g += 128) {
            float sub = 0;
            for (int j = 0; j < gs; ++j) {
                const int kk = g * gs + j;
                uint8_t b = w[woff + size_t(col) * (k / 2) + kk / 2];
                const float product = e4m3(a[size_t(row) * k + kk]) *
                    e2m1((kk & 1) ? b >> 4 : b & 15);
                sub += product;
            }
            const uint8_t s = sc[soff + size_t(col) * (k / gs) + g];
            const float ws = kind ? std::ldexp(1.0f, int(s) - 127) : e4m3(s) * macro;
            const float scale = ws * as[size_t(row) * (k / 128) + g * gs / 128];
            const float product = sub * scale;
            part[tid] += product;
        }
    }
    for (int off = 64; off; off >>= 1)
        for (int tid = 0; tid < off; ++tid) part[tid] += part[tid + off];
    return part[0];
}

struct Cell { int kind, n, k, rows, topk; };
static size_t run(Cell c, bool teeth, bool nan_teeth, bool perf) {
    constexpr int experts = 7;
    const int slots = c.rows * c.topk;
    const size_t wstride = size_t(c.n) * c.k / 2;
    const size_t sstride = size_t(c.n) * c.k / (c.kind ? 32 : 16);
    // Cover shared, per-position/top-k, and per-slot activation maps for every
    // projection, both scale encodings, odd output tails and repeated experts.
    std::vector<uint8_t> a(size_t(slots) * c.k), w(experts * 3 * wstride), sc(experts * 3 * sstride);
    std::vector<float> as(size_t(slots) * (c.k / 128)), macros(experts * 3);
    std::vector<int> sel(slots);
    uint32_t seed = 0x735d120u;
    for (auto& v : a) { v = rng(seed) & 255; if ((v & 127) == 127) v = 0; }
    for (auto& v : w) v = rng(seed) & 255;
    for (auto& v : sc) v = c.kind ? 121 + rng(seed) % 13 : 32 + rng(seed) % 80;
    for (auto& v : as) v = std::ldexp(1.0f, int(rng(seed) % 9) - 6);
    for (auto& v : macros) v = std::ldexp(1.0f, int(rng(seed) % 7) - 9);
    for (auto& v : sel) v = rng(seed) % experts;
    DeviceArray<uint8_t> ad(a), wd(w), sd(sc);
    DeviceArray<float> asd(as), md(macros), out(size_t(slots) * c.n + 16), ref(size_t(slots) * c.n + 16);
    DeviceArray<int> ids(sel);
    std::vector<float> witness(size_t(slots) * c.n + 16), candidate(witness.size());
    size_t compared = 0;
    for (int proj = 0; proj < 3; ++proj) {
        for (int mode = 0; mode < 3; ++mode) {
            int agroup = mode == 1 ? c.topk : 0;
            int astride = mode == 2 ? 1 : 0;
            check(cudaMemset(ref.p, 0xa5, witness.size() * 4));
            // The independently retained per-expert kernel is the witness. One
            // call per slot avoids assuming the candidate's activation indexing.
            for (int slot = 0; slot < slots; ++slot) {
                const int ar = mode == 1 ? slot / c.topk : (mode == 2 ? slot : 0);
                const int ep = sel[slot] * 3 + proj;
                kernel_check(memra_dsv4_fp4_gemm(ad.p + size_t(ar) * c.k,
                    asd.p + size_t(ar) * (c.k / 128), wd.p + ep * wstride,
                    sd.p + ep * sstride, macros[ep], c.kind,
                    ref.p + size_t(slot) * c.n, 1, c.n, c.k, nullptr));
            }
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(witness.data(), ref.p, witness.size() * 4, cudaMemcpyDeviceToHost));
            // Also exercise the public launchers, not only direct instantiations.
            // Run this process separately with block/warp/unset to test the door.
            for (int arm : {0, 1, 2}) {
                check(cudaMemset(out.p, 0xa5, candidate.size() * 4));
                const dim3 grid((c.n + 3) / 4, slots);
                if (arm == 2) {
                    if (agroup) {
                        kernel_check(memra_dsv4_fp4_gemm_sel_g(ad.p, asd.p, wd.p, sd.p,
                            md.p, ids.p, proj, astride, c.kind, out.p, slots,
                            c.n, c.k, wstride, sstride, agroup, nullptr));
                    } else {
                        kernel_check(memra_dsv4_fp4_gemm_sel(ad.p, asd.p, wd.p, sd.p,
                            md.p, ids.p, proj, astride, c.kind, out.p, slots,
                            c.n, c.k, wstride, sstride, nullptr));
                    }
                } else if (arm == 1) {
                    dsv4_fp4_gemm_sel_kernel<true><<<grid, 128, DSV4_FP4_SMEM(512)>>>(ad.p, asd.p,
                        wd.p, sd.p, md.p, ids.p, proj, astride, c.kind, out.p,
                        c.n, c.k, wstride, sstride, agroup);
                } else {
                    dsv4_fp4_gemm_sel_kernel<false><<<grid, 128, DSV4_FP4_SMEM(128)>>>(ad.p, asd.p,
                        wd.p, sd.p, md.p, ids.p, proj, astride, c.kind, out.p,
                        c.n, c.k, wstride, sstride, agroup);
                }
                check(cudaDeviceSynchronize());
                check(cudaMemcpy(candidate.data(), out.p, candidate.size() * 4, cudaMemcpyDeviceToHost));
                if (teeth && arm == 1) candidate[0] = candidate[0] + 1.0f;
                if (nan_teeth && arm == 1) candidate[0] = witness[0] = NAN;
                for (size_t i = 0; i < witness.size(); ++i) {
                    if (!std::isfinite(witness[i]) || !std::isfinite(candidate[i]))
                        throw std::runtime_error("non-finite comparison operand");
                    if (bits(witness[i]) != bits(candidate[i])) {
                        throw std::runtime_error("bit mismatch kind=" + std::to_string(c.kind) +
                            " arm=" + std::to_string(arm) + " index=" + std::to_string(i));
                    }
                }
                compared += size_t(slots) * c.n;
            }
            for (int slot : {0, slots - 1}) {
                for (int col : {0, c.n - 1}) {
                    const int ar = mode == 1 ? slot / c.topk : (mode == 2 ? slot : 0);
                    const int ep = sel[slot] * 3 + proj;
                    float expected = mirror(a, as, w, sc, macros[ep], ar, c.n, c.k, col,
                        c.kind, ep * wstride, ep * sstride);
                    if (bits(expected) != bits(witness[size_t(slot) * c.n + col]))
                        throw std::runtime_error("host mirror mismatch");
                }
            }
        }
    }
    std::printf("EXACT kind=%d n=%d k=%d rows=%d topk=%d compared=%zu\n",
        c.kind, c.n, c.k, c.rows, c.topk, compared);
    if (perf) {
        // Optional box-only microbenchmark; correctness-only is the default.
        cudaEvent_t start, end;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
        float samples[2][5];
        const dim3 grid((c.n + 3) / 4, slots);
        for (int rep = 0; rep < 5; ++rep) for (int order = 0; order < 2; ++order) {
            const int warp = (rep + order) % 2;
            check(cudaEventRecord(start));
            for (int i = 0; i < 20; ++i) {
                if (warp) dsv4_fp4_gemm_sel_kernel<true><<<grid, 128, DSV4_FP4_SMEM(512)>>>(
                    ad.p, asd.p, wd.p, sd.p, md.p, ids.p, 0, 0, c.kind, out.p,
                    c.n, c.k, wstride, sstride, c.topk);
                else dsv4_fp4_gemm_sel_kernel<false><<<grid, 128, DSV4_FP4_SMEM(128)>>>(
                    ad.p, asd.p, wd.p, sd.p, md.p, ids.p, 0, 0, c.kind, out.p,
                    c.n, c.k, wstride, sstride, c.topk);
            }
            check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            check(cudaEventElapsedTime(&samples[warp][rep], start, end));
            std::printf("TIMING kind=%d n=%d k=%d rows=%d topk=%d rep=%d warp=%d us=%.3f\n",
                c.kind,c.n,c.k,c.rows,c.topk,rep,warp,samples[warp][rep]*50.0f);
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(end));
    }
    return compared;
}
int main(int argc, char** argv) {
    try {
        const bool teeth = argc > 1 && std::strcmp(argv[1], "--teeth") == 0;
        const bool nan_teeth = argc > 1 && std::strcmp(argv[1], "--nan-teeth") == 0;
        const bool perf = argc > 1 && std::strcmp(argv[1], "--perf") == 0;
        const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
        if (argc > 1 && std::strcmp(argv[1], "--invalid-switch") == 0) {
            if (memra_dsv4_fp4_gemm_sel(nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, 0, 0, 0, nullptr, 1, 4, 128, 256, 32, nullptr) != 40021 ||
                memra_dsv4_fp4_gemm_sel_g(nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, 0, 0, 0, nullptr, 1, 4, 128, 256, 32, 1, nullptr) != 40021)
                throw std::runtime_error("invalid switch did not fail closed");
            std::puts("PASS invalid switch rejected by both launchers");
            return 0;
        }
        if (argc > 1 && !teeth && !nan_teeth && !perf && !quick)
            throw std::runtime_error("unknown argument");
        std::printf("RUNTIME_SWITCH %s\n", std::getenv("MEMRA_DSV4_FP4_REDUCE")
            ? std::getenv("MEMRA_DSV4_FP4_REDUCE") : "<unset>");
        size_t compared = 0;
        for (int kind : {0, 1}) {
            if (quick) {
                compared += run(Cell{kind,7,8192,3,2},teeth,nan_teeth,false);
                continue;
            }
            for (Cell c : {Cell{kind,7,256,3,2}, Cell{kind,2048,4096,1,6},
                           Cell{kind,4096,2048,6,6}, Cell{kind,65,8192,32,6},
                           Cell{kind,129,128,64,6}}) compared += run(c,teeth,nan_teeth,perf);
        }
        std::printf("PASS selected FP4 reduction: %zu bit comparisons; host witnesses and tail canaries checked\n", compared);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what()); return 1;
    }
}
