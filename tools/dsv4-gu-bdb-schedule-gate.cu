// Standalone exact-order GU schedule candidate.
//
// Candidate seam: double-buffer B so gate and up weight decode/store are both
// issued before the A wait/gate MMA. The gate and up MMA chains, K order,
// stores, and epilogue remain in the current order. This gate compares the
// candidate against both actual current GU launchers on the bank=128 fixture.
// The tile arrays are __half (F16), not BF16.

#define main dsv4_gu_geometry_unused_main
#include "dsv4-gu-geometry-gate.cu"
#undef main

constexpr int R9_BDB_BUFS = 2;
constexpr size_t R9_BDB_STATIC_BYTES =
    static_cast<size_t>(SKT_STAGES * SK_BM * SKT_STRIDE * sizeof(__half))
    + static_cast<size_t>(R9_BDB_BUFS * SK_BN * SKT_STRIDE * sizeof(__half));

template<int QT, bool M1 = false>
static __global__ void __launch_bounds__(128)
r9_moe_kq_sktail_gu_bdb(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ H, const float* __restrict__ row_scale,
        const float* __restrict__ macro_g, const float* __restrict__ macro_u,
        const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit) {
    if (QT != QT_NVFP4_MODELOPT) return;
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][SK_BM][SKT_STRIDE];
    __shared__ __align__(16) __half Bdb[R9_BDB_BUFS][SK_BN][SKT_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    extern __shared__ uint32_t packed_h2[];
    const int ntx = (out_f + SK_BN - 1) / SK_BN;
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<true>(packed_h2);
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, 1, 2);
    if (total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid = warp * 32 + lane;
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
#define R9_BDB_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for (int i = 0; i < 2; ++i) if (!M1 || i == 0) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while (0)
        R9_BDB_LOAD_A(0, 0);
        if (nkb > 1) R9_BDB_LOAD_A(1, SKT_BK);
        KqRaw gb0 = kq_fetch<QT>(gwrow, gsrow, bc0, s_cb, in_f);
        KqRaw gb1 = kq_fetch<QT>(gwrow, gsrow, bc0 + 16, s_cb, in_f);
        KqRaw ub0 = kq_fetch<QT>(uwrow, usrow, bc0, s_cb, in_f);
        KqRaw ub1 = kq_fetch<QT>(uwrow, usrow, bc0 + 16, s_cb, in_f);
        float ag[4][4] = {}, au[4][4] = {};
        for (int kb = 0; kb < nkb; ++kb) {
            const int a_cur = kb % SKT_STAGES;
            const int b_cur = kb & 1;
            if (kb + 2 < nkb) R9_BDB_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);

            // BDB schedule: both projections are decoded/stored before the
            // A wait and gate MMA. The two buffers prevent overwrite while
            // gate reads B[cur] and up reads B[cur^1].
            kq_store_variant<QT, true>(gb0, &Bdb[b_cur][brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, true>(gb1, &Bdb[b_cur][brow][bc0 + 16], s_cb, packed_h2);
            if (kb + 1 < nkb) {
                gb0 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                gb1 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            kq_store_variant<QT, true>(ub0, &Bdb[b_cur ^ 1][brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, true>(ub1, &Bdb[b_cur ^ 1][brow][bc0 + 16], s_cb, packed_h2);
            if (kb + 1 < nkb) {
                ub0 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                ub1 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if (kb + 2 < nkb) asm volatile("cp.async.wait_group 2;");
            else if (kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if (!M1 || ((warp & 1) == 0)) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a, &As[a_cur][wm][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b0, &Bdb[b_cur][wn][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b1, &Bdb[b_cur][wn + 16][kk * 16], SKT_STRIDE);
                    sk_mma(ag[0], a, b0[0], b0[2]); sk_mma(ag[1], a, b0[1], b0[3]);
                    sk_mma(ag[2], a, b1[0], b1[2]); sk_mma(ag[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
            if (!M1 || ((warp & 1) == 0)) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a, &As[a_cur][wm][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b0, &Bdb[b_cur ^ 1][wn][kk * 16], SKT_STRIDE);
                    sk_ldm16x16(b1, &Bdb[b_cur ^ 1][wn + 16][kk * 16], SKT_STRIDE);
                    sk_mma(au[0], a, b0[0], b0[2]); sk_mma(au[1], a, b0[1], b0[3]);
                    sk_mma(au[2], a, b1[0], b1[2]); sk_mma(au[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
#undef R9_BDB_LOAD_A
        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const float rs = (r0 < m_e) ? row_scale[lo + r0] : 0.0f;
        const float mg = (r0 < m_e) ? macro_g[lo + r0] : 0.0f;
        const float mu = (r0 < m_e) ? macro_u[lo + r0] : 0.0f;
        const float rw = (r0 < m_e) ? route_w[lo + r0] : 0.0f;
        float* hrow = H + (size_t)(lo + r0) * out_f;
#pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            const int c = cb + nb * 8;
            if (r0 < m_e) {
#define R9_BDB_STORE(col, ga, ua) do { \
                if ((col) < out_f) { \
                    float g = __fmul_rn(__fmul_rn((ga), rs), mg); \
                    float u = __fmul_rn(__fmul_rn((ua), rs), mu); \
                    u = fminf(fmaxf(u, -limit), limit); g = fminf(g, limit); \
                    float hv = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u); \
                    hrow[(col)] = __fmul_rn(hv, rw); \
                } \
            } while (0)
                R9_BDB_STORE(c, ag[nb][0], au[nb][0]); R9_BDB_STORE(c + 1, ag[nb][1], au[nb][1]);
#undef R9_BDB_STORE
            }
        }
    }
}

struct BdbBuffers {
    float* h_bdb = nullptr;
    size_t h_len = 0;
};

static void alloc_bdb(BdbBuffers& b, size_t h_len) {
    b.h_len = h_len;
    CUDA_OK(cudaMalloc(&b.h_bdb, h_len * sizeof(float)));
    std::vector<uint32_t> guard(h_len, GUARD_BITS);
    CUDA_OK(cudaMemcpy(b.h_bdb, guard.data(), guard.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
}

static void free_bdb(BdbBuffers& b) {
    if (b.h_bdb) CUDA_OK(cudaFree(b.h_bdb));
    b.h_bdb = nullptr;
    b.h_len = 0;
}

struct BdbGeometry {
    LaunchGeometry current;
    int candidate_occ;
    int candidate_grid;
};

static BdbGeometry bdb_geometry() {
    const LaunchGeometry current = geometry();
    int dev = 0;
    int sms = 1;
    int candidate_occ = 1;
    CUDA_OK(cudaGetDevice(&dev));
    CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &candidate_occ, r9_moe_kq_sktail_gu_bdb<QT_NVFP4_MODELOPT, true>, 128,
        H2_SMEM_BYTES));
    if (candidate_occ < 1) candidate_occ = 1;
    return {current, candidate_occ, sms * candidate_occ};
}

enum class BdbArm { Current, CurrentM1Half2, CandidateM1 };

static const char* bdb_arm_name(BdbArm arm) {
    switch (arm) {
    case BdbArm::Current: return "current-gu-half2";
    case BdbArm::CurrentM1Half2: return "current-gu-m1-half2";
    case BdbArm::CandidateM1: return "bdb-m1";
    }
    return "unknown";
}

static float* bdb_output(BdbArm arm, const BankBuffers& b, const BdbBuffers& candidate) {
    switch (arm) {
    case BdbArm::Current: return b.h_control;
    case BdbArm::CurrentM1Half2: return b.h_m1;
    case BdbArm::CandidateM1: return candidate.h_bdb;
    }
    return nullptr;
}

static void bdb_reset_output(float* output, size_t h_len, cudaStream_t stream) {
    std::vector<uint32_t> guard(h_len, GUARD_BITS);
    CUDA_OK(cudaMemcpyAsync(output, guard.data(), guard.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
}

static void bdb_launch(BdbArm arm, const BankBuffers& b, const BdbBuffers& candidate,
                       const BdbGeometry& g) {
    switch (arm) {
    case BdbArm::Current:
        check_rc("GU half2 current", memra_moe_kq_gemm_sk_gu_half2(
            b.table, NE, b.ex_ids, b.act, b.h_control, b.row_scale, b.macro_g, b.macro_u,
            b.route_w, b.ex_off, NE, HIDDEN, INTER, 6.0f, HIDDEN / 2, b.stream));
        break;
    case BdbArm::CurrentM1Half2:
        check_rc("GU M1 half2 current", memra_moe_kq_gemm_sk_gu_m1_half2(
            b.table, NE, b.ex_ids, b.act, b.h_m1, b.row_scale, b.macro_g, b.macro_u,
            b.route_w, b.ex_off, NE, HIDDEN, INTER, 6.0f, HIDDEN / 2, b.stream));
        break;
    case BdbArm::CandidateM1:
        r9_moe_kq_sktail_gu_bdb<QT_NVFP4_MODELOPT, true>
            <<<g.candidate_grid, dim3(32, 4, 1), H2_SMEM_BYTES, b.stream>>>(
                b.table, NE, b.ex_ids, HIDDEN / 2, b.act, candidate.h_bdb, b.row_scale,
                b.macro_g, b.macro_u, b.route_w, b.ex_off, NE, HIDDEN, INTER, -1, 6.0f);
        CUDA_OK(cudaGetLastError());
        break;
    }
}

static unsigned long long bdb_dispatches() {
    return memra_moe_kq_gemm_sk_gu_half2_dispatches();
}

static float bdb_launch_observe(BdbArm arm, const BankBuffers& b,
                                const BdbBuffers& candidate, const BdbGeometry& g,
                                uint8_t* flush_buffer, size_t flush_bytes,
                                const std::vector<uint32_t>* reference,
                                std::vector<uint32_t>& observed, bool timed) {
    float* output = bdb_output(arm, b, candidate);
    bdb_reset_output(output, b.h_len, b.stream);
    flush_l2(b.stream, flush_buffer, flush_bytes);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (timed) {
        CUDA_OK(cudaEventCreate(&start));
        CUDA_OK(cudaEventCreate(&stop));
        CUDA_OK(cudaEventRecord(start, b.stream));
    }
    const unsigned long long before = bdb_dispatches();
    bdb_launch(arm, b, candidate, g);
    if (timed) CUDA_OK(cudaEventRecord(stop, b.stream));
    CUDA_OK(cudaStreamSynchronize(b.stream));
    const unsigned long long delta = bdb_dispatches() - before;
    if (arm != BdbArm::CandidateM1 && delta != 1) {
        std::fprintf(stderr, "FAIL arm=%s control_enqueue_delta=%llu expected=1\n",
                     bdb_arm_name(arm), delta);
        std::exit(1);
    }
    if (arm == BdbArm::CandidateM1 && delta != 0) {
        std::fprintf(stderr, "FAIL arm=%s changed_control_enqueue_delta=%llu\n",
                     bdb_arm_name(arm), delta);
        std::exit(1);
    }
    float ms = 0.0f;
    if (timed) {
        CUDA_OK(cudaEventSynchronize(stop));
        CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_OK(cudaEventDestroy(start));
        CUDA_OK(cudaEventDestroy(stop));
    }
    observed = dtoh_bits(output, b.h_len, b.stream);
    assert_h_bits_finite_guards(bdb_arm_name(arm), reference, observed, b.active, INTER);
    return ms;
}

static void bdb_warm(BdbArm arm, const BankBuffers& b, const BdbBuffers& candidate,
                     const BdbGeometry& g, uint8_t* flush_buffer, size_t flush_bytes,
                     const std::vector<uint32_t>* reference,
                     std::vector<uint32_t>& observed) {
    (void)bdb_launch_observe(arm, b, candidate, g, flush_buffer, flush_bytes,
                             reference, observed, false);
}

static void bdb_compare(const char* sequence, BdbArm control, int active,
                        const BankBuffers& b, const BdbBuffers& candidate,
                        const BdbGeometry& g, uint8_t* flush_buffer, size_t flush_bytes,
                        const std::vector<uint32_t>& reference) {
    double control_sum = 0.0;
    double candidate_sum = 0.0;
    const unsigned long long before = bdb_dispatches();
    for (int cycle = 0; cycle < 3; ++cycle) {
        std::vector<uint32_t> observed;
        control_sum += bdb_launch_observe(control, b, candidate, g, flush_buffer, flush_bytes,
                                          &reference, observed, true);
        candidate_sum += bdb_launch_observe(BdbArm::CandidateM1, b, candidate, g,
                                            flush_buffer, flush_bytes, &reference,
                                            observed, true);
        candidate_sum += bdb_launch_observe(BdbArm::CandidateM1, b, candidate, g,
                                            flush_buffer, flush_bytes, &reference,
                                            observed, true);
        control_sum += bdb_launch_observe(control, b, candidate, g, flush_buffer, flush_bytes,
                                          &reference, observed, true);
    }
    const unsigned long long delta = bdb_dispatches() - before;
    if (delta != 6) {
        std::fprintf(stderr, "FAIL active=%d sequence=%s control_dispatch_delta=%llu expected=6\n",
                     active, sequence, delta);
        std::exit(1);
    }
    std::printf("COMPARE active_groups=%d sequence=%s cycles=3 control=%s candidate=bdb-m1 "
                "control_samples=6 candidate_samples=6 control_mean_ms=%.6f "
                "candidate_mean_ms=%.6f control_dispatch_delta=6 "
                "flush_2x_l2_before_every_scored=true h_bit_exact_finite_guards=true\n",
                active, sequence, bdb_arm_name(control), control_sum / 6.0,
                candidate_sum / 6.0);
}

static void bdb_run_case(int active, uint8_t* flush_buffer, size_t flush_bytes) {
    BankBuffers b{};
    b.stream = nullptr;
    alloc_bank(b, active);
    BdbBuffers candidate{};
    alloc_bdb(candidate, b.h_len);
    const BdbGeometry g = bdb_geometry();

    std::vector<uint32_t> observed;
    bdb_warm(BdbArm::Current, b, candidate, g, flush_buffer, flush_bytes, nullptr, observed);
    const std::vector<uint32_t> reference = observed;
    bdb_warm(BdbArm::CurrentM1Half2, b, candidate, g, flush_buffer, flush_bytes,
             &reference, observed);
    bdb_warm(BdbArm::CandidateM1, b, candidate, g, flush_buffer, flush_bytes,
             &reference, observed);
    std::printf("WARMUP active_groups=%d arms=current-gu-half2,current-gu-m1-half2,bdb-m1 "
                "checked=true scored=false\n", active);

    bdb_compare("current/bdb/bdb/current", BdbArm::Current, active, b, candidate, g,
                flush_buffer, flush_bytes, reference);
    bdb_compare("m1/bdb/bdb/m1", BdbArm::CurrentM1Half2, active, b, candidate, g,
                flush_buffer, flush_bytes, reference);

    std::printf("CASE active_groups=%d bank=128 hidden=4096 inter=2048 selected_experts="
                "0,17,34,51,68,85 candidate=bdb-m1 candidate_grid=%d candidate_occ=%d "
                "candidate_static_f16_bytes=%zu dynamic_packed_h2=%zu "
                "m1_a_stage_index=kb%%SKT_STAGES b_stage_index=kb%%2 "
                "active_route_coverage=true\n",
                active, g.candidate_grid, g.candidate_occ, R9_BDB_STATIC_BYTES,
                H2_SMEM_BYTES);
    free_bdb(candidate);
    free_bank(b);
}

int main() {
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, dev));
    const size_t flush_bytes = std::max<size_t>(2 * static_cast<size_t>(prop.l2CacheSize), 1 << 20);
    uint8_t* flush_buffer = nullptr;
    CUDA_OK(cudaMalloc(&flush_buffer, flush_bytes));
    std::printf("BDB_GATE bank=128 topk_max=6 tile_n=64 active_counts=1,3,4,6 "
                "flush_2x_l2_bytes=%zu candidate_static_f16_bytes=%zu "
                "dynamic_packed_h2=%zu\n",
                flush_bytes, R9_BDB_STATIC_BYTES, H2_SMEM_BYTES);
    for (int active : {1, 3, 4, 6}) bdb_run_case(active, flush_buffer, flush_bytes);
    CUDA_OK(cudaFree(flush_buffer));
    std::puts("PASS GU BDB matched H-bit/finite/guard ABBA3 gate");
    return 0;
}
