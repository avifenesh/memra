// M=1-only exact reduction transport experiment. Included by dsv4_gpu.cu;
// inherits its -fmad=false build. Original arithmetic/load bodies stay intact.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

// Host gate selection is thread-local and never read by a device kernel. Capture
// freezes the chosen kernel function, so later selection cannot mutate a graph.
// Gate callers own separate candidate/control states and drain before switching.
// Read once per host thread before its first enqueue. Only explicit 0 rolls back.
// The explicit gate override below still owns A/B selection before capture.
static bool dsv4_dense_exact_tail_environment_default() {
    const char* value = std::getenv("MEMRA_DSV4_DENSE_EXACT_TAIL");
    return !value || std::strcmp(value, "0") != 0;
}
static thread_local bool dsv4_dense_exact_tail_enabled = dsv4_dense_exact_tail_environment_default();
static thread_local int dsv4_dense_exact_tail_suppressed = 0;
static thread_local uint64_t dsv4_dense_exact_tail_enqueues[2] = {};
extern "C" int memra_dsv4_dense_exact_tail_set_for_gate(int enabled) {
    if (enabled != 0 && enabled != 1) return 40075;
    dsv4_dense_exact_tail_enabled = enabled != 0;
    return 0;
}
extern "C" int memra_dsv4_dense_exact_tail_enabled_for_gate() {
    return dsv4_dense_exact_tail_enabled ? 1 : 0;
}
extern "C" int memra_dsv4_dense_exact_tail_restore_default_for_gate() {
    dsv4_dense_exact_tail_enabled = dsv4_dense_exact_tail_environment_default();
    return dsv4_dense_exact_tail_enabled ? 1 : 0;
}
extern "C" int memra_dsv4_dense_exact_tail_counts_for_gate(uint64_t* fp8, uint64_t* dots) {
    if (!fp8 || !dots) return 40075;
    *fp8 = dsv4_dense_exact_tail_enqueues[0];
    *dots = dsv4_dense_exact_tail_enqueues[1];
    return 0;
}
struct Dsv4DenseExactTailControlScope {
    bool suppress;
    explicit Dsv4DenseExactTailControlScope(bool value) : suppress(value) {
        if (suppress) ++dsv4_dense_exact_tail_suppressed;
    }
    ~Dsv4DenseExactTailControlScope() { if (suppress) --dsv4_dense_exact_tail_suppressed; }
};
static bool dsv4_dense_exact_tail_aligned(const void* p, uintptr_t alignment) {
    return p && (reinterpret_cast<uintptr_t>(p) % alignment) == 0;
}
static bool dsv4_dense_exact_tail_fp8_admits(const void* w, const float* sc, int sc_cols,
    const void* x, float* y, int m, int n, int k) {
    return m == 1 && n > 0 && k > 0 && k % 8 == 0 && sc_cols >= (k + 127LL) / 128 &&
        dsv4_dense_exact_tail_aligned(w, 8) && dsv4_dense_exact_tail_aligned(x, 16) &&
        dsv4_dense_exact_tail_aligned(sc, 4) && dsv4_dense_exact_tail_aligned(y, 4);
}
static bool dsv4_dense_exact_tail_dots_admits(const float* x, const void* w,
    int w_is_bf16, float* y, int m, int n, int k) {
    return m == 1 && n > 0 && k > 0 && k % 8 == 0 && (w_is_bf16 == 0 || w_is_bf16 == 1) &&
        dsv4_dense_exact_tail_aligned(x, 16) && dsv4_dense_exact_tail_aligned(w, 16) &&
        dsv4_dense_exact_tail_aligned(y, 4);
}

// Warp 0 replays precisely the old 128-leaf halving tree. All 32 lanes must
// call every shuffle with the full mask; only the old tree's active lanes add.
__device__ __forceinline__ float dsv4_dense_exact_tail_reduce(const float* p) {
    const int lane = threadIdx.x;
    float a = __fadd_rn(p[lane], p[lane + 64]);
    float b = __fadd_rn(p[lane + 32], p[lane + 96]);
    float v = __fadd_rn(a, b);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float partner = __shfl_down_sync(0xffffffffu, v, off);
        if (lane < off) v = __fadd_rn(v, partner);
    }
    return v;
}

template <int M, bool GROUPED = false>
__device__ __forceinline__ void dsv4_dense_exact_tail_fp8_row(const uint8_t* __restrict__ w,
                                       const float* __restrict__ sc, int sc_cols,
                                       const uint16_t* __restrict__ x, float* __restrict__ y,
                                       int n, int k, int xstride, int ystride,
                                       int group_xstride, int group_ystride, int flat) {
    int row = GROUPED ? flat % n : flat;
    if (row >= n) return;
    int group = GROUPED ? flat / n : 0;
    int weight_row = GROUPED ? group * n + row : row;
    const uint16_t* x_group = x + (long)group * group_xstride;
    float* y_group = y + (long)group * group_ystride;
    // smem e4m3 LUT — see dsv4_gemv_fp8_kernel's note (bit-inert decode transport).
    __shared__ float e4m3_tab[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) e4m3_tab[i] = dsv4_e4m3((uint8_t)i);
    __syncthreads();
    const uint8_t* wr = w + (long)weight_row * k;
    const float* srow = sc + (long)(weight_row >> 7) * sc_cols;
    float part[M];
#pragma unroll
    for (int t = 0; t < M; t++) part[t] = 0.0f;
    // Unroll-by-2, early weight loads — the m=1 twin's note applies: load scheduling
    // only, per-(t)-accumulation order verbatim, bit-identical.
    int stride = blockDim.x * 8;
    int i0 = threadIdx.x * 8;
    for (; i0 + stride < k; i0 += 2 * stride) {
        int i1 = i0 + stride;
        uint2 wva = *(const uint2*)(wr + i0);
        uint2 wvb = *(const uint2*)(wr + i1);
        float sa = srow[i0 >> 7];
        float sb = srow[i1 >> 7];
        unsigned wba[2] = {wva.x, wva.y};
        unsigned wbb[2] = {wvb.x, wvb.y};
        float wua[8], wub[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wua[2 * j] = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sa;
            wua[2 * j + 1] = e4m3_tab[(wba[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sa;
            wub[2 * j] = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * sb;
            wub[2 * j + 1] = e4m3_tab[(wbb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * sb;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x_group + (long)t * xstride + i0);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wua[2 * j] * x0;
                acc += wua[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x_group + (long)t * xstride + i1);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wub[2 * j] * x0;
                acc += wub[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
    }
    for (; i0 < k; i0 += stride) {
        uint2 wv = *(const uint2*)(wr + i0);
        float s = srow[i0 >> 7];
        unsigned wb[2] = {wv.x, wv.y};
        float wu[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            wu[2 * j] = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2) * 8)) & 0xFFu] * s;
            wu[2 * j + 1] = e4m3_tab[(wb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 0xFFu] * s;
        }
#pragma unroll
        for (int t = 0; t < M; t++) {
            uint4 xv = *(const uint4*)(x_group + (long)t * xstride + i0);
            unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
            float acc = part[t];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float x0 = __uint_as_float((xw[j] & 0xFFFFu) << 16);
                float x1 = __uint_as_float(xw[j] & 0xFFFF0000u);
                acc += wu[2 * j] * x0;
                acc += wu[2 * j + 1] * x1;
            }
            part[t] = acc;
        }
    }
    static_assert(M == 1, "exact-tail candidate is M=1 only");
    __shared__ float red[128];
    red[threadIdx.x] = part[0];
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = dsv4_dense_exact_tail_reduce(red);
        if (threadIdx.x == 0) y_group[row] = v;
    }
}

template <int M>
__device__ __forceinline__ void dsv4_dense_exact_tail_dots_row(const float* __restrict__ x,
                                             const void* __restrict__ w, int w_is_bf16,
                                             float* __restrict__ y, int k, int n, int j) {
    if (j >= n) return;
    float part[M];
#pragma unroll
    for (int t = 0; t < M; t++) part[t] = 0.0f;
    if (w_is_bf16) {
        const uint16_t* wr = (const uint16_t*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            uint4 wv = *(const uint4*)(wr + i0);
            unsigned ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int t = 0; t < M; t++) {
                const float* xr = x + (long)t * k;
                float4 xa = *(const float4*)(xr + i0);
                float4 xb = *(const float4*)(xr + i0 + 4);
                float xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float acc = part[t];
#pragma unroll
                for (int q2 = 0; q2 < 4; q2++) {
                    float w0 = __uint_as_float((ww[q2] & 0xFFFFu) << 16);
                    float w1 = __uint_as_float(ww[q2] & 0xFFFF0000u);
                    acc += xs[2 * q2] * w0;
                    acc += xs[2 * q2 + 1] * w1;
                }
                part[t] = acc;
            }
        }
    } else {
        const float* wr = (const float*)w + (long)j * k;
        for (int i0 = threadIdx.x * 8; i0 < k; i0 += blockDim.x * 8) {
            float4 wa = *(const float4*)(wr + i0);
            float4 wb = *(const float4*)(wr + i0 + 4);
#pragma unroll
            for (int t = 0; t < M; t++) {
                const float* xr = x + (long)t * k;
                float4 xa = *(const float4*)(xr + i0);
                float4 xb = *(const float4*)(xr + i0 + 4);
                float acc = part[t];
                acc += xa.x * wa.x;
                acc += xa.y * wa.y;
                acc += xa.z * wa.z;
                acc += xa.w * wa.w;
                acc += xb.x * wb.x;
                acc += xb.y * wb.y;
                acc += xb.z * wb.z;
                acc += xb.w * wb.w;
                part[t] = acc;
            }
        }
    }
    static_assert(M == 1, "exact-tail candidate is M=1 only");
    __shared__ float red[128];
    red[threadIdx.x] = part[0];
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = dsv4_dense_exact_tail_reduce(red);
        if (threadIdx.x == 0) y[j] = v;
    }
}

// Existing entry points and batch entries instantiate the SAME row bodies.
// Only the mapping from blockIdx.x to a projection-local row changes.
template <int M, bool GROUPED = false>
__global__ void dsv4_dense_exact_tail_fp8_kernel(const uint8_t* w,
    const float* sc, int sc_cols, const uint16_t* x, float* y,
    int n, int k, int xstride, int ystride, int group_xstride, int group_ystride) {
    dsv4_dense_exact_tail_fp8_row<M, GROUPED>(w, sc, sc_cols, x, y, n, k,
        xstride, ystride, group_xstride, group_ystride, blockIdx.x);
}
template <int M>
__global__ void dsv4_dense_exact_tail_dots_kernel(const float* x, const void* w,
    int w_is_bf16, float* y, int k, int n) {
    dsv4_dense_exact_tail_dots_row<M>(x, w, w_is_bf16, y, k, n, blockIdx.x);
}

struct Dsv4DenseBatchFp8 {
    const void* w;
    const float* sc;
    float* y;
    int n;
    int sc_cols;
};
struct Dsv4DenseBatchDots {
    const void* w;
    float* y;
    int n;
    int w_is_bf16;
};
// Explicit gate-only operand capture. No environment read or production caller.
// Copies real live operands once per projection pair; captures stay on the gate host.
static thread_local std::string dsv4_dense_batch_capture_dir;
static thread_local std::unordered_set<const void*> dsv4_dense_batch_captured;
extern "C" int memra_dsv4_dense_batch_capture_for_gate(const char* dir) {
    dsv4_dense_batch_capture_dir = dir ? dir : "";
    dsv4_dense_batch_captured.clear();
    return 0;
}
static int dsv4_dense_batch_write(FILE* f, const void* ptr, size_t bytes) {
    std::vector<unsigned char> data(bytes);
    auto rc = cudaMemcpy(data.data(), ptr, bytes, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) return 10000 + (int)rc;
    return fwrite(data.data(), 1, bytes, f) == bytes ? 0 : 40075;
}
// Binary tape: eight u32 header words, then x, wa, sca (FP8), wb, scb (FP8).
// Header = magic, kind, K, Na, Nb, scale-cols-a/type-a, scale-cols-b/type-b, device.
static int dsv4_dense_batch_capture(const void* x, int k,
    const Dsv4DenseBatchFp8* fp8, const Dsv4DenseBatchDots* dots, cudaStream_t stream) {
    if (dsv4_dense_batch_capture_dir.empty()) return 0;
    const void* key = fp8 ? fp8[0].w : dots[0].w;
    if (dsv4_dense_batch_captured.count(key)) return 0;
    cudaStreamCaptureStatus status;
    auto rc = cudaStreamIsCapturing(stream, &status);
    if (rc != cudaSuccess) return 10000 + (int)rc;
    if (status != cudaStreamCaptureStatusNone) return 40075;
    rc = cudaStreamSynchronize(stream);
    if (rc != cudaSuccess) return 10000 + (int)rc;
    int dev = -1;
    rc = cudaGetDevice(&dev);
    if (rc != cudaSuccess) return 10000 + (int)rc;
    const std::string path = dsv4_dense_batch_capture_dir + "/pair-" +
        std::to_string(dsv4_dense_batch_captured.size()) + ".bin";
    FILE* f = fopen(path.c_str(), "wbx");
    if (!f) return 40075;
    uint32_t h[8] = {0x44534231u, fp8 ? 0u : 1u, (uint32_t)k,
        (uint32_t)(fp8 ? fp8[0].n : dots[0].n), (uint32_t)(fp8 ? fp8[1].n : dots[1].n),
        (uint32_t)(fp8 ? fp8[0].sc_cols : dots[0].w_is_bf16),
        (uint32_t)(fp8 ? fp8[1].sc_cols : dots[1].w_is_bf16), (uint32_t)dev};
    int err = fwrite(h, sizeof(h), 1, f) == 1 ? 0 : 40075;
    if (!err) err = dsv4_dense_batch_write(f, x, (size_t)k * (fp8 ? 2 : 4));
    for (int i = 0; i < 2 && !err; ++i) {
        if (fp8) {
            err = dsv4_dense_batch_write(f, fp8[i].w, (size_t)fp8[i].n * k);
            if (!err) err = dsv4_dense_batch_write(f, fp8[i].sc,
                (size_t)((fp8[i].n + 127) / 128) * fp8[i].sc_cols * 4);
        } else {
            err = dsv4_dense_batch_write(f, dots[i].w,
                (size_t)dots[i].n * k * (dots[i].w_is_bf16 ? 2 : 4));
        }
    }
    if (fclose(f) != 0 && !err) err = 40075;
    if (!err) dsv4_dense_batch_captured.insert(key);
    return err;
}

static thread_local bool dsv4_dense_batch_enabled = [] {
    const char* value = std::getenv("MEMRA_DSV4_DENSE_BATCH");
    return value && std::strcmp(value, "1") == 0;
}();
static thread_local uint64_t dsv4_dense_batch_enqueues[2] = {};
extern "C" int memra_dsv4_dense_batch_set_for_gate(int enabled) {
    if (enabled != 0 && enabled != 1) return 40075;
    dsv4_dense_batch_enabled = enabled != 0;
    return 0;
}
extern "C" int memra_dsv4_dense_batch_enabled_for_gate() {
    return dsv4_dense_batch_enabled && dsv4_dense_exact_tail_enabled &&
        !dsv4_dense_exact_tail_suppressed;
}
extern "C" int memra_dsv4_dense_batch_counts_for_gate(uint64_t* fp8, uint64_t* dots) {
    if (!fp8 || !dots) return 40075;
    *fp8 = dsv4_dense_batch_enqueues[0];
    *dots = dsv4_dense_batch_enqueues[1];
    return 0;
}
__global__ void dsv4_dense_batch_fp8_kernel(const uint16_t* x, int k,
    Dsv4DenseBatchFp8 a, Dsv4DenseBatchFp8 b) {
    const bool second = blockIdx.x >= a.n;
    const Dsv4DenseBatchFp8 p = second ? b : a;
    const int row = second ? blockIdx.x - a.n : blockIdx.x;
    dsv4_dense_exact_tail_fp8_row<1, false>((const uint8_t*)p.w, p.sc, p.sc_cols,
        x, p.y, p.n, k, k, p.n, 0, 0, row);
}
__global__ void dsv4_dense_batch_dots_kernel(const float* x, int k,
    Dsv4DenseBatchDots a, Dsv4DenseBatchDots b) {
    const bool second = blockIdx.x >= a.n;
    const Dsv4DenseBatchDots p = second ? b : a;
    const int row = second ? blockIdx.x - a.n : blockIdx.x;
    dsv4_dense_exact_tail_dots_row<1>(x, p.w, p.w_is_bf16, p.y, k, p.n, row);
}
static bool dsv4_dense_batch_disjoint(float* a, int na, float* b, int nb) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a);
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b);
    return na > 0 && nb > 0 && (pa < pb ? (pb - pa) / sizeof(float) >= (unsigned)na
        : (pa - pb) / sizeof(float) >= (unsigned)nb);
}
extern "C" int memra_dsv4_dense_batch_fp8(const void* x, int k,
    const Dsv4DenseBatchFp8* pair, void* raw_stream) {
    if (!pair) return 40075;
    const auto a = pair[0], b = pair[1];
    if (!dsv4_dense_exact_tail_fp8_admits(a.w, a.sc, a.sc_cols, x, a.y, 1, a.n, k) ||
        !dsv4_dense_exact_tail_fp8_admits(b.w, b.sc, b.sc_cols, x, b.y, 1, b.n, k) ||
        (int64_t)a.n + b.n > INT32_MAX || !dsv4_dense_batch_disjoint(a.y,a.n,b.y,b.n)) return 40075;
    if (!dsv4_dense_batch_capture_dir.empty()) {
        for (const auto p : {a, b}) {
            dsv4_dense_exact_tail_fp8_kernel<1, false><<<p.n,128,0,(cudaStream_t)raw_stream>>>(
                (const uint8_t*)p.w,p.sc,p.sc_cols,(const uint16_t*)x,p.y,p.n,k,k,p.n,0,0);
            auto rc = cudaGetLastError();
            if (rc != cudaSuccess) return 10000 + (int)rc;
        }
        return dsv4_dense_batch_capture(x,k,pair,nullptr,(cudaStream_t)raw_stream);
    }
    dsv4_dense_batch_fp8_kernel<<<a.n + b.n, 128, 0, (cudaStream_t)raw_stream>>>(
        (const uint16_t*)x, k, a, b);
    auto rc = cudaGetLastError();
    if (rc != cudaSuccess) return 10000 + (int)rc;
    ++dsv4_dense_batch_enqueues[0];
    return 0;
}
extern "C" int memra_dsv4_dense_batch_dots(const float* x, int k,
    const Dsv4DenseBatchDots* pair, void* raw_stream) {
    if (!pair) return 40075;
    const auto a = pair[0], b = pair[1];
    if (!dsv4_dense_exact_tail_dots_admits(x, a.w, a.w_is_bf16, a.y, 1, a.n, k) ||
        !dsv4_dense_exact_tail_dots_admits(x, b.w, b.w_is_bf16, b.y, 1, b.n, k) ||
        (int64_t)a.n + b.n > INT32_MAX || !dsv4_dense_batch_disjoint(a.y,a.n,b.y,b.n)) return 40075;
    if (!dsv4_dense_batch_capture_dir.empty()) {
        for (const auto p : {a, b}) {
            dsv4_dense_exact_tail_dots_kernel<1><<<p.n,128,0,(cudaStream_t)raw_stream>>>(
                x,p.w,p.w_is_bf16,p.y,k,p.n);
            auto rc = cudaGetLastError();
            if (rc != cudaSuccess) return 10000 + (int)rc;
        }
        return dsv4_dense_batch_capture(x,k,nullptr,pair,(cudaStream_t)raw_stream);
    }
    dsv4_dense_batch_dots_kernel<<<a.n + b.n, 128, 0, (cudaStream_t)raw_stream>>>(x,k,a,b);
    auto rc = cudaGetLastError();
    if (rc != cudaSuccess) return 10000 + (int)rc;
    ++dsv4_dense_batch_enqueues[1];
    return 0;
}

// Raw candidates refuse unsupported arguments before enqueue. Production-facing
// existing launchers retain their original control for every unadmitted case.
extern "C" int memra_dsv4_dense_exact_tail_fp8(const void* w, const float* sc, int sc_cols,
    const void* x, float* y, int m, int n, int k, int xstride, int ystride, void* raw_stream) {
    if (!dsv4_dense_exact_tail_fp8_admits(w, sc, sc_cols, x, y, m, n, k) ||
        (xstride > 0 && xstride % 8 != 0)) return 40075;
    dsv4_dense_exact_tail_fp8_kernel<1, false><<<n, 128, 0, (cudaStream_t)raw_stream>>>(
        (const uint8_t*)w, sc, sc_cols, (const uint16_t*)x, y, n, k,
        xstride > 0 ? xstride : k, ystride > 0 ? ystride : n, 0, 0);
    auto rc = cudaGetLastError();
    if (rc != cudaSuccess) return 10000 + (int)rc;
    ++dsv4_dense_exact_tail_enqueues[0];
    return 0;
}
extern "C" int memra_dsv4_dense_exact_tail_dots(const float* x, const void* w,
    int w_is_bf16, float* y, int m, int n, int k, void* raw_stream) {
    if (!dsv4_dense_exact_tail_dots_admits(x, w, w_is_bf16, y, m, n, k)) return 40075;
    dsv4_dense_exact_tail_dots_kernel<1><<<n, 128, 0, (cudaStream_t)raw_stream>>>(
        x, w, w_is_bf16, y, k, n);
    auto rc = cudaGetLastError();
    if (rc != cudaSuccess) return 10000 + (int)rc;
    ++dsv4_dense_exact_tail_enqueues[1];
    return 0;
}
