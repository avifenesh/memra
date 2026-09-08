// M=1-only exact reduction transport experiment. Included by dsv4_gpu.cu;
// inherits its -fmad=false build. Original arithmetic/load bodies stay intact.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
__global__ void dsv4_dense_exact_tail_fp8_kernel(const uint8_t* __restrict__ w,
                                       const float* __restrict__ sc, int sc_cols,
                                       const uint16_t* __restrict__ x, float* __restrict__ y,
                                       int n, int k, int xstride, int ystride,
                                       int group_xstride, int group_ystride) {
    int flat = blockIdx.x;
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
__global__ void dsv4_dense_exact_tail_dots_kernel(const float* __restrict__ x,
                                             const void* __restrict__ w, int w_is_bf16,
                                             float* __restrict__ y, int k, int n) {
    int j = blockIdx.x;
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
