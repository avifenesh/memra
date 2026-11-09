// probe_gdn_norm.cu: the GDN out-norm inside the fused state/output pass, on a GB10 unit.
//
// Builds against a tree's cu/hybrid.cu (included verbatim) and runs, at the Qwen3.8-27B shape
// (48 value heads, 16 compact q/k heads, head dim 128, C = 32):
//
//   pair   gdn_chunk_state_output_mma (stores o), then gated_rmsnorm_e4m3v2_f32 (block 128) on o
//   fused  gdn_chunk_state_output_norm_mma (cluster pair per head, o never stored)
//
// and compares the F8F4 v2 activation (e4m3 bytes and scale plane) and state_out bit for bit, then
// times both (median of REPS launches, PASSES passes in alternating order).
//
// usage: probe_gdn_norm T REPS PASSES
#include "hybrid.cu"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static uint32_t rng_s = 12345u;
static float frand() { rng_s = rng_s * 1664525u + 1013904223u; return ((rng_s >> 8) & 0xffffff) / 16777216.0f - 0.5f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = u + 0x7fff + ((u >> 16) & 1); return (uint16_t)(r >> 16); }

// gdn_f8f4v2_act_slot on the host
static long long gdn_f8f4v2_act_slot_host(long long t) {
    const int lt = (int)(t & 63), p = lt >> 3, r8 = lt & 7;
    return (t & ~(long long)63) + ((p >> 1) * 4 + (r8 >> 1)) * 4 + (p & 1) * 2 + (r8 & 1);
}

template <typename Tv>
static size_t bitdiff(const Tv* a, const Tv* b, size_t n, size_t* first) {
    std::vector<Tv> ha(n), hb(n);
    CK(cudaMemcpy(ha.data(), a, n * sizeof(Tv), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b, n * sizeof(Tv), cudaMemcpyDeviceToHost));
    size_t d = 0;
    *first = (size_t)-1;
    for (size_t i = 0; i < n; i++)
        if (memcmp(&ha[i], &hb[i], sizeof(Tv))) { if (!d) *first = i; d++; }
    return d;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const int passes = argc > 3 ? atoi(argv[3]) : 2;
    const int H = 48, hk = 16, D = 128, C = 32;
    const int NC = (T + C - 1) / C;
    const size_t nY = (size_t)NC * H * C * D, nSt = (size_t)H * D * D, nO = (size_t)T * H * D;
    const int in_f = H * D;
    const long long tp = ((long long)T + 255) / 256 * 256;
    const size_t vy_bytes = (size_t)T * in_f + (size_t)(in_f / 32) * tp * 4;

    std::vector<float> q((size_t)T * hk * D), k((size_t)T * hk * D);
    for (size_t r = 0; r < (size_t)T * hk; r++) {
        float nq = 0, nk = 0;
        for (int i = 0; i < D; i++) { q[r * D + i] = frand(); k[r * D + i] = frand(); nq += q[r*D+i]*q[r*D+i]; nk += k[r*D+i]*k[r*D+i]; }
        for (int i = 0; i < D; i++) { q[r * D + i] /= sqrtf(nq); k[r * D + i] /= sqrtf(nk); }
    }
    std::vector<uint16_t> kb(k.size());
    for (size_t i = 0; i < k.size(); i++) kb[i] = f2bf(k[i]);
    std::vector<float> gc((size_t)T * H), be((size_t)T * H);
    for (int h = 0; h < H; h++)
        for (int c = 0; c < NC; c++) {
            float acc = 0;
            for (int j = 0; j < C && c * C + j < T; j++) {
                int t = c * C + j;
                acc += -0.02f - fabsf(frand()) * 0.5f;
                gc[(size_t)t * H + h] = acc;
                be[(size_t)t * H + h] = 0.3f + fabsf(frand()) * 0.6f;
            }
        }
    std::vector<float> Uh(nY), Wf(nY);
    for (size_t i = 0; i < nY; i++) { Uh[i] = frand(); Wf[i] = frand() * 0.25f; }
    std::vector<uint16_t> Wb(nY);
    for (size_t i = 0; i < nY; i++) Wb[i] = f2bf(Wf[i]);
    std::vector<float> Ph((size_t)NC * H * C * C, 0.f);
    for (int c = 0; c < NC; c++) for (int h = 0; h < H; h++) for (int j = 0; j < C; j++) for (int i = 0; i <= j; i++)
        Ph[(((size_t)c * H + h) * C + j) * C + i] = frand() * 0.5f;
    std::vector<float> st(nSt);
    for (auto& x : st) x = frand();
    // the gate: a spread around zero; the norm weight around 1, a few exact zeros in z
    std::vector<float> zh(nO), wh(D);
    for (size_t i = 0; i < nO; i++) zh[i] = (i % 997 == 0) ? 0.0f : frand() * 8.0f;
    for (int i = 0; i < D; i++) wh[i] = 1.0f + frand() * 0.4f;

    __nv_bfloat16 *d_kb, *d_Wb; float *d_gc, *d_be, *d_U, *d_q, *d_P, *d_st, *d_z, *d_w, *d_o, *d_so_a, *d_so_b;
    uint8_t *d_vy_a, *d_vy_b;
    CK(cudaMalloc(&d_kb, kb.size() * 2)); CK(cudaMemcpy(d_kb, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_gc, gc.size() * 4)); CK(cudaMemcpy(d_gc, gc.data(), gc.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_be, be.size() * 4)); CK(cudaMemcpy(d_be, be.data(), be.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_U, nY * 4)); CK(cudaMemcpy(d_U, Uh.data(), nY * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_Wb, nY * 2)); CK(cudaMemcpy(d_Wb, Wb.data(), nY * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_q, q.size() * 4)); CK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_P, Ph.size() * 4)); CK(cudaMemcpy(d_P, Ph.data(), Ph.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_st, nSt * 4)); CK(cudaMemcpy(d_st, st.data(), nSt * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_z, nO * 4)); CK(cudaMemcpy(d_z, zh.data(), nO * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_w, D * 4)); CK(cudaMemcpy(d_w, wh.data(), D * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_o, nO * 4)); CK(cudaMemset(d_o, 0, nO * 4));
    CK(cudaMalloc(&d_so_a, nSt * 4)); CK(cudaMalloc(&d_so_b, nSt * 4));
    CK(cudaMemset(d_so_a, 0, nSt * 4)); CK(cudaMemset(d_so_b, 0, nSt * 4));
    CK(cudaMalloc(&d_vy_a, vy_bytes)); CK(cudaMalloc(&d_vy_b, vy_bytes));
    CK(cudaMemset(d_vy_a, 0xab, vy_bytes)); CK(cudaMemset(d_vy_b, 0xab, vy_bytes));
    const float scale = 1.0f / sqrtf((float)D), eps = 1e-6f;
    const unsigned smem_pair = sizeof(GdnK45Smem), smem_norm = sizeof(GdnK45Smem) + sizeof(GdnK45NormSmem);
    CK(cudaFuncSetAttribute(gdn_chunk_state_output_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_pair));
    CK(cudaFuncSetAttribute(gdn_chunk_state_output_norm_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_norm));

    auto pair_k45 = [&]() {
        gdn_chunk_state_output_mma<<<dim3(H, 2), 256, smem_pair>>>(d_kb, d_gc, d_be, d_U, d_Wb, d_q, d_P, d_o,
                                                                  d_st, d_so_a, H, T, C, scale, hk);
        CK(cudaGetLastError());
    };
    auto pair_norm = [&]() {
        gated_rmsnorm_e4m3v2_f32<<<H * T, 128>>>(d_o, d_w, d_z, nullptr, d_vy_a, D, H, T, tp, eps);
        CK(cudaGetLastError());
    };
    auto fused = [&]() {
        gdn_chunk_state_output_norm_mma<<<dim3(H, 2), 256, smem_norm>>>(d_kb, d_gc, d_be, d_U, d_Wb, d_q, d_P,
                                                                       d_z, d_w, d_vy_b, d_st, d_so_b,
                                                                       H, T, C, scale, hk, D, tp, eps,
                                                                       0LL, (long long)T);
        CK(cudaGetLastError());
    };
    pair_k45(); pair_norm(); fused();
    CK(cudaDeviceSynchronize());
    size_t f1, f2;
    const size_t dv = bitdiff(d_vy_a, d_vy_b, vy_bytes, &f1);
    const size_t ds = bitdiff(d_so_a, d_so_b, nSt, &f2);
    printf("T %d: activation bytes %zu differ (of %zu, first %zd), state_out %zu differ (of %zu, first %zd)\n",
           T, dv, vy_bytes, (ssize_t)f1, ds, nSt, (ssize_t)f2);
    {
        // how many NaN / inf / zero-group scales the inputs produced (the edge cases covered)
        std::vector<float> sc((size_t)(in_f / 32) * tp);
        CK(cudaMemcpy(sc.data(), d_vy_a + (size_t)T * in_f, sc.size() * 4, cudaMemcpyDeviceToHost));
        size_t nz = 0, nn = 0;
        for (int g = 0; g < in_f / 32; g++)
            for (int t = 0; t < T; t++) {
                float v = sc[(size_t)g * tp + (size_t)gdn_f8f4v2_act_slot_host(t)];
                if (v == 0.0f) nz++;
                if (!(v == v) || v > 3.4e38f) nn++;
            }
        printf("T %d: scale plane zero groups %zu, nan/inf %zu\n", T, nz, nn);
    }

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto timeit = [&](auto launch) {
        std::vector<float> ms;
        launch(); CK(cudaDeviceSynchronize());
        for (int r = 0; r < reps; r++) {
            CK(cudaEventRecord(e0)); launch(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    for (int p = 0; p < passes; p++) {
        float a_k45 = 0, a_norm = 0, a_pair = 0, b = 0;
        auto pair = [&]() { pair_k45(); pair_norm(); };
        if (p % 2 == 0) {
            a_k45 = timeit(pair_k45); a_norm = timeit(pair_norm); a_pair = timeit(pair); b = timeit(fused);
        } else {
            b = timeit(fused); a_pair = timeit(pair); a_norm = timeit(pair_norm); a_k45 = timeit(pair_k45);
        }
        printf("T %d pass %d: pair %.3f ms (k45 %.3f + norm %.3f), fused %.3f ms, saved %.3f ms\n",
               T, p, a_pair, a_k45, a_norm, b, a_pair - b);
    }
    return 0;
}
