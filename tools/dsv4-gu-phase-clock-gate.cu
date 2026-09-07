// Standalone phase-clock instrumentation of the actual GU half2 program.
//
// The geometry gate supplies the exact bank128/hidden4096/inter2048 fixture,
// current production launcher, guards, and exact-H checks. This file adds a
// duplicate <108,false,true> kernel whose arithmetic/body order is unchanged;
// lane 0 of each warp accumulates clock64() deltas for prefix, activation
// copy/wait, weight decode/store, MMA, barriers, and epilogue. Phase totals are
// per CTA/warp samples and are never summed as wall-time savings.

#define main dsv4_gu_geometry_unused_main
#include "dsv4-gu-geometry-gate.cu"
#undef main

constexpr int R9_PHASES = 6;
constexpr int R9_PREFIX = 0;
constexpr int R9_ACT = 1;
constexpr int R9_DECODE = 2;
constexpr int R9_MMA = 3;
constexpr int R9_BARRIER = 4;
constexpr int R9_EPILOGUE = 5;

static __device__ __forceinline__ void r9_trace_store(
        unsigned long long* trace, int phase_base,
        const unsigned long long (&phase)[R9_PHASES]) {
    for (int p = 0; p < R9_PHASES; ++p) trace[phase_base + p] = phase[p];
}

template<int QT, bool M1 = false, bool PackedStore = true>
static __global__ void __launch_bounds__(128)
r9_moe_kq_sktail_gu_instrumented(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ H, const float* __restrict__ row_scale,
        const float* __restrict__ macro_g, const float* __restrict__ macro_u,
        const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit,
        unsigned long long* __restrict__ trace) {
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[SK_BN][SKT_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    extern __shared__ uint32_t packed_h2[];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int trace_base = (blockIdx.x * 4 + warp) * R9_PHASES;
    unsigned long long phase[R9_PHASES] = {};
    unsigned long long tmark = 0;
    if (lane == 0) tmark = clock64();
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, 1, 2);
    if (total_tiles < 0) total_tiles = s_pre[n_active];
    if (lane == 0) phase[R9_PREFIX] += clock64() - tmark;

    const int nkb = in_f / SKT_BK;
    const int wm = (warp & 1) * 16, wn = (warp >> 1) * 32;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int g = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g + 1] - lo;
        if (m_e != 1) continue;
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * SK_BN;
        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const int qplane = (QT == QT_NVFP4_MODELOPT) ? 2 * 1 : 1;
        const uint8_t* Wg = (const uint8_t*)table[(size_t)0 * n_expert + eid];
        const uint8_t* Sg = (const uint8_t*)table[(size_t)1 * n_expert + eid];
        const uint8_t* Wu = (const uint8_t*)table[(size_t)4 * n_expert + eid];
        const uint8_t* Su = (const uint8_t*)table[(size_t)5 * n_expert + eid];
        (void)qplane;
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
#define R9_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for (int i = 0; i < 2; ++i) if (!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while (0)

        if (lane == 0) tmark = clock64();
        R9_LOAD_A(0, 0);
        if (nkb > 1) R9_LOAD_A(1, SKT_BK);
        if (lane == 0) phase[R9_ACT] += clock64() - tmark;
        if (lane == 0) tmark = clock64();
        KqRaw gb0 = kq_fetch<QT>(gwrow, gsrow, bc0, s_cb, in_f);
        KqRaw gb1 = kq_fetch<QT>(gwrow, gsrow, bc0 + 16, s_cb, in_f);
        KqRaw ub0 = kq_fetch<QT>(uwrow, usrow, bc0, s_cb, in_f);
        KqRaw ub1 = kq_fetch<QT>(uwrow, usrow, bc0 + 16, s_cb, in_f);
        if (lane == 0) phase[R9_DECODE] += clock64() - tmark;
        float ag[4][4] = {}, au[4][4] = {};
        for (int kb = 0; kb < nkb; ++kb) {
            const int cur = kb % SKT_STAGES;
            if (kb + 2 < nkb) {
                if (lane == 0) tmark = clock64();
                R9_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);
                if (lane == 0) phase[R9_ACT] += clock64() - tmark;
            }
            if (lane == 0) tmark = clock64();
            kq_store_variant<QT, true>(gb0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, true>(gb1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if (kb + 1 < nkb) {
                gb0 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                gb1 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if (lane == 0) phase[R9_DECODE] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
            if (kb + 2 < nkb) asm volatile("cp.async.wait_group 2;");
            else if (kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            if (lane == 0) phase[R9_ACT] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
            __syncthreads();
            if (lane == 0) phase[R9_BARRIER] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
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
            if (lane == 0) phase[R9_MMA] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
            __syncthreads();
            if (lane == 0) phase[R9_BARRIER] += clock64() - tmark;

            if (lane == 0) tmark = clock64();
            kq_store_variant<QT, true>(ub0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, true>(ub1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if (kb + 1 < nkb) {
                ub0 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                ub1 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if (lane == 0) phase[R9_DECODE] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
            __syncthreads();
            if (lane == 0) phase[R9_BARRIER] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
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
            if (lane == 0) phase[R9_MMA] += clock64() - tmark;
            if (lane == 0) tmark = clock64();
            __syncthreads();
            if (lane == 0) phase[R9_BARRIER] += clock64() - tmark;
        }
#undef R9_LOAD_A

        if (lane == 0) tmark = clock64();
        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const int pair = lo + r0;
        const float rs = (r0 < m_e) ? row_scale[pair] : 0.0f;
        const float mg = (r0 < m_e) ? macro_g[pair] : 0.0f;
        const float mu = (r0 < m_e) ? macro_u[pair] : 0.0f;
        const float rw = (r0 < m_e) ? route_w[pair] : 0.0f;
        float* hrow = H + (size_t)pair * out_f;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int c = cb + nb * 8;
            if (r0 < m_e) {
#define R9_STORE(col, ga, ua) do { \
                if ((col) < out_f) { \
                    float g = __fmul_rn(__fmul_rn((ga), rs), mg); \
                    float u = __fmul_rn(__fmul_rn((ua), rs), mu); \
                    u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); \
                    float hv = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u); \
                    hrow[(col)] = __fmul_rn(hv, rw); \
                } \
            } while (0)
                R9_STORE(c, ag[nb][0], au[nb][0]); R9_STORE(c + 1, ag[nb][1], au[nb][1]);
#undef R9_STORE
            }
        }
        if (lane == 0) phase[R9_EPILOGUE] += clock64() - tmark;
    }
    if (lane == 0) r9_trace_store(trace, trace_base, phase);
}

static float run_instrumented(const BankBuffers& b, const LaunchGeometry& g,
                              unsigned long long* trace, size_t trace_words,
                              uint8_t* flush_buffer, size_t flush_bytes,
                              std::vector<uint32_t>& observed) {
    CUDA_OK(cudaMemsetAsync(trace, 0, trace_words * sizeof(unsigned long long), b.stream));
    CUDA_OK(cudaStreamSynchronize(b.stream));
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start, b.stream));
    r9_moe_kq_sktail_gu_instrumented<QT_NVFP4_MODELOPT, false, true>
        <<<g.control_full_grid, dim3(32, 4, 1), H2_SMEM_BYTES, b.stream>>>(
            b.table, NE, b.ex_ids, HIDDEN / 2, b.act, b.h_grid, b.row_scale,
            b.macro_g, b.macro_u, b.route_w, b.ex_off, NE, HIDDEN, INTER, -1, 6.0f, trace);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(stop, b.stream)); CUDA_OK(cudaEventSynchronize(stop));
    float ms = 0.0f; CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start)); CUDA_OK(cudaEventDestroy(stop));
    observed = dtoh_bits(b.h_grid, b.h_len, b.stream);
    return ms;
}

static void print_trace_summary(const std::vector<unsigned long long>& trace,
                                int grid, int active) {
    static const char* names[R9_PHASES] = {
        "prefix_lut", "activation_copy_wait", "weight_decode_store",
        "mma", "barriers", "epilogue"};
    std::printf("TRACE active_groups=%d grid=%d samples=cta*warp phase_clock64=true\n",
                active, grid);
    for (int phase = 0; phase < R9_PHASES; ++phase) {
        unsigned long long min_v = ~0ull, max_v = 0, sum = 0;
        size_t count = 0;
        for (int block = 0; block < grid; ++block) {
            for (int warp = 0; warp < 4; ++warp) {
                const unsigned long long value = trace[(block * 4 + warp) * R9_PHASES + phase];
                if (value == 0) continue;
                min_v = std::min(min_v, value); max_v = std::max(max_v, value);
                sum += value; ++count;
            }
        }
        std::printf("TRACE_PHASE active_groups=%d phase=%s samples=%zu min_cycles=%llu "
                    "max_cycles=%llu mean_cycles=%.3f not_wall_sum=true\n",
                    active, names[phase], count, count ? min_v : 0, max_v,
                    count ? static_cast<double>(sum) / count : 0.0);
    }
    // A compact per-CTA/warp receipt for the first four active samples makes
    // skew visible without pretending those samples are additive wall time.
    const int shown = std::min(grid * 4, 8);
    for (int sample = 0; sample < shown; ++sample) {
        std::printf("TRACE_SAMPLE active_groups=%d cta=%d warp=%d phases=",
                    active, sample / 4, sample % 4);
        for (int phase = 0; phase < R9_PHASES; ++phase)
            std::printf("%s%llu", phase ? "," : "", trace[sample * R9_PHASES + phase]);
        std::puts("");
    }
}

static std::vector<unsigned long long> read_trace(const unsigned long long* trace,
                                                   size_t words, cudaStream_t stream) {
    std::vector<unsigned long long> host(words);
    CUDA_OK(cudaMemcpyAsync(host.data(), trace, words * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return host;
}

static void run_active_case(int active, uint8_t* flush_buffer, size_t flush_bytes) {
    BankBuffers b{}; b.stream = nullptr; alloc_bank(b, active);
    const LaunchGeometry g = geometry();
    int instr_occ = 0;
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &instr_occ, r9_moe_kq_sktail_gu_instrumented<QT_NVFP4_MODELOPT, false, true>,
        128, H2_SMEM_BYTES));
    const size_t trace_words = static_cast<size_t>(g.control_full_grid) * 4 * R9_PHASES;
    unsigned long long* trace = nullptr;
    CUDA_OK(cudaMalloc(&trace, trace_words * sizeof(unsigned long long)));
    std::printf("INSTRUMENT_CONFIG active_groups=%d bank=128 hidden=4096 inter=2048 "
                "grid=%d control_occ=%d instrument_occ=%d trace_words=%zu "
                "resource_clock_overhead=true\n",
                active, g.control_full_grid, g.control_occ, instr_occ, trace_words);

    std::vector<uint32_t> reference;
    std::vector<uint32_t> observed;
    // Warmup current and instrumented; neither contributes to the scored means.
    (void)score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes, nullptr, reference);
    (void)run_instrumented(b, g, trace, trace_words, flush_buffer, flush_bytes, observed);
    assert_h_bits_finite_guards("instrumented warmup", &reference, observed, active, INTER);
    std::printf("WARMUP_INSTRUMENT active_groups=%d current=true instrumented=true scored=false h_exact=true\n", active);

    double current_sum = 0.0, instr_sum = 0.0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        current_sum += score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes,
                                    &reference, observed);
        instr_sum += run_instrumented(b, g, trace, trace_words, flush_buffer, flush_bytes, observed);
        assert_h_bits_finite_guards("instrumented scored", &reference, observed, active, INTER);
        print_trace_summary(read_trace(trace, trace_words, b.stream), g.control_full_grid, active);
        instr_sum += run_instrumented(b, g, trace, trace_words, flush_buffer, flush_bytes, observed);
        assert_h_bits_finite_guards("instrumented repeat", &reference, observed, active, INTER);
        current_sum += score_launch(ScoredArm::Current, b, g, flush_buffer, flush_bytes,
                                    &reference, observed);
    }
    std::printf("INSTRUMENT_COMPARE active_groups=%d cycles=3 current_mean_ms=%.6f "
                "instrumented_mean_ms=%.6f overhead_ratio=%.6f phase_totals_not_wall=true "
                "h_bit_exact_finite_guards=true\n",
                active, current_sum / 6.0, instr_sum / 6.0,
                (instr_sum / 6.0) / (current_sum / 6.0));
    CUDA_OK(cudaFree(trace)); free_bank(b);
}

int main() {
    int dev = 0; CUDA_OK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(prop.l2CacheSize), 1 << 20);
    uint8_t* flush_buffer = nullptr; CUDA_OK(cudaMalloc(&flush_buffer, flush_bytes));
    std::printf("CONTROL bank=128 hidden=4096 inter=2048 active=1,3,4,6 "
                "clock64_semantics=per_multiprocessor_counter flush_2x_l2=%zu\n", flush_bytes);
    for (int active : {1, 3, 4, 6}) run_active_case(active, flush_buffer, flush_bytes);
    CUDA_OK(cudaFree(flush_buffer));
    std::puts("PASS GU phase clock instrumentation compile/runtime gate; no production verdict");
    return 0;
}
