// M1 FP8 N512/N1024 K4096 only. Each S is a distinct numeric class,
// not token-identical to dense exact-tail. No atomics or capture allocations.
#pragma once
static thread_local int dsv4_fp8_ksplit_slices = [] {
    const char* value = std::getenv("MEMRA_DSV4_FP8_KSPLIT");
    if (value && std::strcmp(value, "2") == 0) return 2;
    if (value && std::strcmp(value, "4") == 0) return 4;
    if (value && std::strcmp(value, "8") == 0) return 8;
    return 0;
}();
extern "C" int memra_dsv4_fp8_ksplit_set_for_gate(int slices) {
    if (slices != 0 && slices != 2 && slices != 4 && slices != 8) return 40075;
    dsv4_fp8_ksplit_slices = slices;
    return 0;
}
extern "C" int memra_dsv4_fp8_ksplit_slices_for_gate() {
    return dsv4_fp8_ksplit_slices;
}
template<int S>
__global__ void dsv4_fp8_ksplit_partial_kernel(const uint8_t* __restrict__ w,
    const float* __restrict__ sc, int sc_cols, const uint16_t* __restrict__ x,
    float* __restrict__ partial) {
    const int row = blockIdx.x / S, slice = blockIdx.x % S;
    constexpr int width = 4096 / S;
    __shared__ float lut[256];
    for (int i = threadIdx.x; i < 256; i += 128) lut[i] = dsv4_e4m3(uint8_t(i));
    __syncthreads();
    const uint8_t* wr = w + long(row) * 4096;
    const float* srow = sc + long(row >> 7) * sc_cols;
    float acc = 0.0f;
    // Keep the original leaf identity, stride and element order restricted to
    // this slice. At S8 odd slices start at original leaf 64, not leaf 0.
    for (int i = threadIdx.x * 8; i < 4096; i += 128 * 8) {
        if (i < slice * width || i >= (slice + 1) * width) continue;
        uint2 wv = *(const uint2*)(wr + i);
        uint4 xv = *(const uint4*)(x + i);
        unsigned wb[2] = {wv.x, wv.y};
        unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
        float scale = srow[i >> 7];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float w0 = __fmul_rn(lut[(wb[j >> 1] >> (((j & 1) * 2) * 8)) & 255u], scale);
            float w1 = __fmul_rn(lut[(wb[j >> 1] >> (((j & 1) * 2 + 1) * 8)) & 255u], scale);
            float x0 = __uint_as_float((xw[j] & 65535u) << 16);
            float x1 = __uint_as_float(xw[j] & 0xffff0000u);
            acc = __fadd_rn(acc, __fmul_rn(w0, x0));
            acc = __fadd_rn(acc, __fmul_rn(w1, x1));
        }
    }
    __shared__ float red[128];
    red[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = dsv4_dense_exact_tail_reduce(red);
        if (threadIdx.x == 0) partial[row * S + slice] = value;
    }
}
template<int S>
__global__ void dsv4_fp8_ksplit_reduce_kernel(const float* __restrict__ partial,
    float* __restrict__ y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.0f;
#pragma unroll
    for (int slice = 0; slice < S; ++slice) acc = __fadd_rn(acc, partial[row * S + slice]);
    y[row] = acc;
}
template<int S> static int dsv4_fp8_ksplit_launch(const void* w, const float* sc,
    int sc_cols, const void* x, float* partial, float* y, int n, cudaStream_t stream) {
    dsv4_fp8_ksplit_partial_kernel<S><<<n * S, 128, 0, stream>>>(
        (const uint8_t*)w, sc, sc_cols, (const uint16_t*)x, partial);
    auto rc = cudaGetLastError();
    if (rc != cudaSuccess) return (int)rc;
    dsv4_fp8_ksplit_reduce_kernel<S><<<(n + 127) / 128, 128, 0, stream>>>(partial, y, n);
    return (int)cudaGetLastError();
}
extern "C" int memra_dsv4_fp8_ksplit(const void* w, const float* sc, int sc_cols,
    const void* x, float* partial, int partial_len, float* y, int m, int n, int k,
    void* raw_stream) {
    int slices = dsv4_fp8_ksplit_slices;
    if (!slices || m != 1 || (n != 512 && n != 1024) || k != 4096 ||
        partial_len < n * slices || !dsv4_dense_exact_tail_aligned(partial, 4) ||
        !dsv4_dense_exact_tail_fp8_admits(w, sc, sc_cols, x, y, m, n, k)) return 40075;
    auto stream = (cudaStream_t)raw_stream;
    switch (slices) {
        case 2: return dsv4_fp8_ksplit_launch<2>(w, sc, sc_cols, x, partial, y, n, stream);
        case 4: return dsv4_fp8_ksplit_launch<4>(w, sc, sc_cols, x, partial, y, n, stream);
        case 8: return dsv4_fp8_ksplit_launch<8>(w, sc, sc_cols, x, partial, y, n, stream);
        default: return 40075;
    }
}
