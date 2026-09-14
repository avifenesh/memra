// Standalone native FP4 pair-store candidate.
//
// The production ModelOpt packed helper reads a shared 256-entry half2 LUT whose
// entries are the doubled g_kvalues_mxfp4 codebook. This candidate replaces
// only that LUT read with native cvt.rn.f16x2.e2m1x2, normalizes nibble 8
// (-0) to the helper's +0 policy, doubles the native E2M1 half2, and applies
// the same half2 scale multiply. The GU A/B/MMA/epilogue order is copied from
// the current <108,false,true>/<108,true,true> kernels unchanged.

#define main dsv4_gu_geometry_unused_main
#include "dsv4-gu-geometry-gate.cu"
#undef main

static __device__ __forceinline__ uint8_t r9_fp4_norm_nibble(uint8_t nibble) {
    return nibble == 8u ? 0u : nibble;
}

static __device__ __forceinline__ uint32_t r9_fp4_native_pair(uint8_t packed) {
    const uint16_t normalized = static_cast<uint16_t>(
        (static_cast<uint16_t>(r9_fp4_norm_nibble(packed >> 4)) << 4)
        | r9_fp4_norm_nibble(packed & 0xFu));
    uint32_t out = 0;
    asm volatile("{ .reg .b8 __r9_fp4_lo, __r9_fp4_hi;\n"
                 "  mov.b16 {__r9_fp4_lo, __r9_fp4_hi}, %1;\n"
                 "  cvt.rn.f16x2.e2m1x2 %0, __r9_fp4_lo; }"
                 : "=r"(out) : "h"(normalized));
    return out;
}

static __device__ __forceinline__ void r9_fp4_store_pair(
        const KqRaw& raw, __half* dst) {
    const __half scale = __float2half(raw.f1);
    const __half2 scale2 = __halves2half2(scale, scale);
    const __half2 doubled = __halves2half2(__float2half(2.0f), __float2half(2.0f));
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const uint8_t packed = static_cast<uint8_t>(raw.q[pair >> 2] >> (8 * (pair & 3)));
        const uint32_t native = r9_fp4_native_pair(packed);
        const __half2 codes = *reinterpret_cast<const __half2*>(&native);
        const __half2 modelopt_codes = __hmul2(codes, doubled);
        *reinterpret_cast<__half2*>(dst + pair * 2) = __hmul2(scale2, modelopt_codes);
    }
}

// Exact helper-vs-native exhaustive gate: 256 packed code bytes x 256 signed
// E4M3FN scale bytes, preserving the helper's nibble-8 zero and scale policy.
__global__ void r9_fp4_pair_basis_kernel(uint32_t* current_bits, uint32_t* native_bits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256 * 256) return;
    const uint8_t packed = static_cast<uint8_t>(i & 0xff);
    const uint8_t scale_code = static_cast<uint8_t>(i >> 8);
    __shared__ uint32_t s_h2[256];
    kq_stage_half2_lut<true>(s_h2);
    __syncthreads();
    KqRaw raw{};
    raw.q[0] = packed;
    raw.f1 = g_e4m3fn_to_float(scale_code);
    __half current[16], native[16];
    const uint32_t packed_word = static_cast<uint32_t>(packed) * 0x01010101u;
    raw.q[0] = raw.q[1] = raw.q[2] = raw.q[3] = packed_word;
    kq_store_variant<QT_NVFP4_MODELOPT, true>(raw, current, nullptr, s_h2);
    r9_fp4_store_pair(raw, native);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        current_bits[i * 16 + j] = __half_as_ushort(current[j]);
        native_bits[i * 16 + j] = __half_as_ushort(native[j]);
    }
}

template<bool M1>
static __global__ void __launch_bounds__(128)
r9_moe_kq_sktail_gu_native(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A, float* __restrict__ H,
        const float* __restrict__ row_scale, const float* __restrict__ macro_g,
        const float* __restrict__ macro_u, const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit) {
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SK_BN][SKT_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT_NVFP4_MODELOPT)];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT_NVFP4_MODELOPT>(s_cb);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, 1, 2);
    if (total_tiles < 0) total_tiles = s_pre[n_active];
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane, nkb = in_f / SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int g = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g + 1] - lo;
        if (m_e != 1) continue;
        const int local = t - s_pre[g], m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;
        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const uint8_t* Wg = (const uint8_t*)table[(size_t)0 * n_expert + eid];
        const uint8_t* Sg = (const uint8_t*)table[(size_t)1 * n_expert + eid];
        const uint8_t* Wu = (const uint8_t*)table[(size_t)4 * n_expert + eid];
        const uint8_t* Su = (const uint8_t*)table[(size_t)5 * n_expert + eid];
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* gwrow = Wg + (size_t)bn * row_bytes;
        const uint8_t* gsrow = Sg + (size_t)bn * (in_f / 16);
        const uint8_t* uwrow = Wu + (size_t)bn * row_bytes;
        const uint8_t* usrow = Su + (size_t)bn * (in_f / 16);
        const __half* agp[2]; int asr[2], asc[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = tid + i * 128;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
#define R9_FP4_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for (int i = 0; i < 2; ++i) if (!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while (0)
        R9_FP4_LOAD_A(0, 0); if (nkb > 1) R9_FP4_LOAD_A(1, SKT_BK);
        KqRaw gb0 = kq_fetch<QT_NVFP4_MODELOPT>(gwrow, gsrow, bc0, s_cb, in_f);
        KqRaw gb1 = kq_fetch<QT_NVFP4_MODELOPT>(gwrow, gsrow, bc0 + 16, s_cb, in_f);
        KqRaw ub0 = kq_fetch<QT_NVFP4_MODELOPT>(uwrow, usrow, bc0, s_cb, in_f);
        KqRaw ub1 = kq_fetch<QT_NVFP4_MODELOPT>(uwrow, usrow, bc0 + 16, s_cb, in_f);
        float ag[4][4] = {}, au[4][4] = {};
        for (int kb = 0; kb < nkb; ++kb) {
            const int cur = kb % SKT_STAGES;
            if (kb + 2 < nkb) R9_FP4_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);
            r9_fp4_store_pair(gb0, &Bs[brow][bc0]); r9_fp4_store_pair(gb1, &Bs[brow][bc0 + 16]);
            if (kb + 1 < nkb) {
                gb0 = kq_fetch<QT_NVFP4_MODELOPT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                gb1 = kq_fetch<QT_NVFP4_MODELOPT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if (kb + 2 < nkb) asm volatile("cp.async.wait_group 2;");
            else if (kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (!M1 || ((warp & 1) == 0)) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a, &As[cur][wm][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk * 16], SKT_STRIDE);
                    sk_mma(ag[0], a, b0[0], b0[2]); sk_mma(ag[1], a, b0[1], b0[3]);
                    sk_mma(ag[2], a, b1[0], b1[2]); sk_mma(ag[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
            r9_fp4_store_pair(ub0, &Bs[brow][bc0]); r9_fp4_store_pair(ub1, &Bs[brow][bc0 + 16]);
            if (kb + 1 < nkb) {
                ub0 = kq_fetch<QT_NVFP4_MODELOPT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                ub1 = kq_fetch<QT_NVFP4_MODELOPT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            __syncthreads();
            if (!M1 || ((warp & 1) == 0)) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a, &As[cur][wm][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk * 16], SKT_STRIDE);
                    sk_mma(au[0], a, b0[0], b0[2]); sk_mma(au[1], a, b0[1], b0[3]);
                    sk_mma(au[2], a, b1[0], b1[2]); sk_mma(au[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
#undef R9_FP4_LOAD_A
        const int r0 = m0 + wm + lane / 4, cb = n0 + wn + (lane % 4) * 2;
        const int pair = lo + r0;
        const float rs = r0 < m_e ? row_scale[pair] : 0.0f;
        const float mg = r0 < m_e ? macro_g[pair] : 0.0f;
        const float mu = r0 < m_e ? macro_u[pair] : 0.0f;
        const float rw = r0 < m_e ? route_w[pair] : 0.0f;
        float* hrow = H + (size_t)pair * out_f;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int c = cb + nb * 8;
            if (r0 < m_e) {
                if (c < out_f) {
                    float g = __fmul_rn(__fmul_rn(ag[nb][0], rs), mg);
                    float u = __fmul_rn(__fmul_rn(au[nb][0], rs), mu);
                    u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit);
                    hrow[c] = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u) * rw;
                }
                if (c + 1 < out_f) {
                    float g = __fmul_rn(__fmul_rn(ag[nb][1], rs), mg);
                    float u = __fmul_rn(__fmul_rn(au[nb][1], rs), mu);
                    u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit);
                    hrow[c + 1] = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u) * rw;
                }
            }
        }
    }
}

extern "C" void r9_fp4_native_compile_only_launch(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        long row_bytes, const __half* A, float* H, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        int total_tiles, float limit, cudaStream_t stream) {
    r9_moe_kq_sktail_gu_native<false>
        <<<1, dim3(32, 4, 1), 0, stream>>>(
            table, n_expert, ex_ids, row_bytes, A, H, row_scale, macro_g,
            macro_u, route_w, ex_off, n_active, in_f, out_f, total_tiles, limit);
}

extern "C" void r9_fp4_native_m1_launch(
        const unsigned long long* table, int n_expert, const int* ex_ids,
        long row_bytes, const __half* A, float* H, const float* row_scale,
        const float* macro_g, const float* macro_u, const float* route_w,
        const int* ex_off, int n_active, int in_f, int out_f,
        int total_tiles, float limit, int grid, cudaStream_t stream) {
    r9_moe_kq_sktail_gu_native<true>
        <<<grid, dim3(32, 4, 1), 0, stream>>>(
            table, n_expert, ex_ids, row_bytes, A, H, row_scale, macro_g,
            macro_u, route_w, ex_off, n_active, in_f, out_f, total_tiles, limit);
}

static void check_pair_basis() {
    constexpr int N = 256 * 256;
    constexpr size_t kCells = static_cast<size_t>(N) * 16;
    uint32_t* current = nullptr; uint32_t* native = nullptr;
    CUDA_OK(cudaMalloc(&current, (kCells + 16) * sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&native, (kCells + 16) * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(current, 0xa5, (kCells + 16) * sizeof(uint32_t)));
    CUDA_OK(cudaMemset(native, 0xa5, (kCells + 16) * sizeof(uint32_t)));
    r9_fp4_pair_basis_kernel<<<256, 256>>>(current, native);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint32_t> hc(kCells + 16), hn(kCells + 16);
    CUDA_OK(cudaMemcpy(hc.data(), current, hc.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hn.data(), native, hn.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 16; ++j) {
            const size_t offset = static_cast<size_t>(i) * 16 + j;
            if ((hc[offset] & 0xffff0000u) != 0 || (hn[offset] & 0xffff0000u) != 0) {
                std::fprintf(stderr, "FAIL FP4 basis output not written cell=%d element=%d\n", i, j);
                std::exit(1);
            }
        }
        for (int j = 0; j < 16; ++j) if (hc[static_cast<size_t>(i) * 16 + j]
                                              != hn[static_cast<size_t>(i) * 16 + j]) {
            std::fprintf(stderr, "FAIL FP4 pair basis packed=0x%02x scale=0x%02x current=%08x native=%08x\n",
                         i & 0xff, i >> 8, hc[static_cast<size_t>(i) * 16 + j],
                         hn[static_cast<size_t>(i) * 16 + j]);
            std::exit(1);
        }
    }
    for (size_t i = kCells; i < kCells + 16; ++i)
        if (hc[i] != 0xa5a5a5a5u || hn[i] != 0xa5a5a5a5u) {
            std::fprintf(stderr, "FAIL FP4 basis tail guard offset=%zu\n", i - kCells);
            std::exit(1);
        }
    std::puts("PASS FP4 pair basis cells=65536 current_helper=native exact signedzero_policy=nibble8_pluszero scale_nan=explicit_zero");
    CUDA_OK(cudaFree(current)); CUDA_OK(cudaFree(native));
}

static double run_m1_control(const BankBuffers& b, int active, uint8_t* flush_buffer,
                             size_t flush_bytes, const std::vector<uint32_t>& guard,
                             std::vector<uint32_t>& observed) {
    CUDA_OK(cudaMemcpy(b.h_m1, guard.data(), guard.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr, stop = nullptr; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    const unsigned long long before = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    CUDA_OK(cudaEventRecord(start, b.stream));
    check_rc("actual GU M1 half2", memra_moe_kq_gemm_sk_gu_m1_half2(
        b.table, NE, b.ex_ids, b.act, b.h_m1, b.row_scale, b.macro_g, b.macro_u,
        b.route_w, b.ex_off, NE, HIDDEN, INTER, 6.0f, HIDDEN / 2, b.stream));
    CUDA_OK(cudaEventRecord(stop, b.stream)); CUDA_OK(cudaEventSynchronize(stop));
    const unsigned long long after = memra_moe_kq_gemm_sk_gu_half2_dispatches();
    if (after - before != 1) {
        std::fprintf(stderr, "FAIL actual GU-M1 half2 enqueue delta=%llu expected=1\n", after - before);
        std::exit(1);
    }
    float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    observed = dtoh_bits(b.h_m1, b.h_len, b.stream);
    assert_h_bits_finite_guards("actual-m1", nullptr, observed, active, INTER);
    return ms;
}

static double run_native_m1(const BankBuffers& b, const LaunchGeometry& g,
                            uint8_t* flush_buffer, size_t flush_bytes,
                            const std::vector<uint32_t>& guard,
                            std::vector<uint32_t>& observed) {
    CUDA_OK(cudaMemcpy(b.h_grid, guard.data(), guard.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr, stop = nullptr; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start, b.stream));
    r9_fp4_native_m1_launch(b.table, NE, b.ex_ids, HIDDEN / 2, b.act, b.h_grid,
                            b.row_scale, b.macro_g, b.macro_u, b.route_w, b.ex_off,
                            NE, HIDDEN, INTER, -1, 6.0f, g.m1_full_grid, b.stream);
    CUDA_OK(cudaGetLastError()); CUDA_OK(cudaEventRecord(stop, b.stream)); CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    observed = dtoh_bits(b.h_grid, b.h_len, b.stream);
    assert_h_bits_finite_guards("native-m1", nullptr, observed, b.active, INTER);
    return ms;
}

static void run_active(int active, uint8_t* flush_buffer, size_t flush_bytes) {
    BankBuffers b{}; b.stream = nullptr; alloc_bank(b, active); const LaunchGeometry g = geometry();
    const std::vector<uint32_t> guard(b.h_len, GUARD_BITS);
    std::vector<uint32_t> reference, observed;
    auto run_current = [&](const std::vector<uint32_t>* ref) {
        CUDA_OK(cudaMemcpy(b.h_control, guard.data(), guard.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        return score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes, ref, observed);
    };
    (void)run_current(nullptr); reference = observed;
    (void)run_m1_control(b, active, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("warm-m1", reference, observed, active, INTER, "H");
    (void)run_native_m1(b, g, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("warm-native", reference, observed, active, INTER, "H");
    std::puts("WARMUP native-fp4/nonM1/M1 checked=true scored=false");
    double cur = 0.0, m1 = 0.0, nat_current = 0.0, nat_m1 = 0.0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        cur += run_current(&reference);
        nat_current += run_native_m1(b, g, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("native-vs-current", reference, observed, active, INTER, "H");
        nat_current += run_native_m1(b, g, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("native-vs-current-repeat", reference, observed, active, INTER, "H");
        cur += run_current(&reference);
    }
    std::vector<uint32_t> m1_reference;
    for (int cycle = 0; cycle < 3; ++cycle) {
        m1 += run_m1_control(b, active, flush_buffer, flush_bytes, guard, observed);
        if (m1_reference.empty()) m1_reference = observed;
        else assert_rows_and_guards("m1-repeat", m1_reference, observed, active, INTER, "H");
        nat_m1 += run_native_m1(b, g, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("native-vs-m1", m1_reference, observed, active, INTER, "H");
        nat_m1 += run_native_m1(b, g, flush_buffer, flush_bytes, guard, observed); assert_rows_and_guards("native-vs-m1-repeat", m1_reference, observed, active, INTER, "H");
        m1 += run_m1_control(b, active, flush_buffer, flush_bytes, guard, observed);
        assert_rows_and_guards("m1-repeat-final", m1_reference, observed, active, INTER, "H");
    }
    std::printf("COMPARE active_groups=%d current_native_mean_ms=%.6f m1_native_mean_ms=%.6f "
                "actual_m1_mean_ms=%.6f current_mean_ms=%.6f exact_H=true guards=true "
                "ABBA_cycles=3 native_vs_current=true native_vs_m1=true\n",
                active, nat_current / 6.0, nat_m1 / 6.0, m1 / 6.0, cur / 6.0);
    free_bank(b);
}

int main() {
    check_pair_basis();
    int dev = 0; CUDA_OK(cudaGetDevice(&dev)); cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(prop.l2CacheSize), 1 << 20);
    uint8_t* flush_buffer = nullptr; CUDA_OK(cudaMalloc(&flush_buffer, flush_bytes));
    for (int active : {1, 3, 4, 6}) run_active(active, flush_buffer, flush_bytes);
    CUDA_OK(cudaFree(flush_buffer));
    std::puts("PASS native FP4 pair current/M1 exact gate; basis and both ABBA comparisons executed");
    return 0;
}
