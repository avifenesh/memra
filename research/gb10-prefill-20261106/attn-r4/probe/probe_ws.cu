// probe_ws.cu: the prime's KV dequant workspace pass. The engine's fa_dequant_kv_ws_bf16 and its
// _prime_table twin (from the tree's flash_attn.cu, the code under test) against the per-element
// loop main shipped before the eight-per-thread body (copied below as ref_ws1, verbatim), on one
// GPU. Every bf16 of each workspace is compared with the reference's.
//
// build (default fatbin formats, q8_0 K / q5_1 V):
//   nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I<tree>/crates/memra-engine/cu probe_ws.cu -o probe_ws
// build (the kf8vf8 fatbin's formats, e4m3 K and V, 1 B per element):
//   ... -DMEMRA_KV_KFMT=1 -DMEMRA_KV_VFMT=2 probe_ws.cu -o probe_ws_f8
// usage: probe_ws KV_DIM[,KV_DIM...] TKV[,TKV...] REPS
#include "flash_attn.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// main's fa_dequant_kv_ws_bf16 body before this change (one thread per element), verbatim.
extern "C" __global__ void ref_ws1(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        __nv_bfloat16* __restrict__ Kw, __nv_bfloat16* __restrict__ Vw,
        int kv_dim_k, int kv_dim_v, int t_kv,
        long k_tok_bytes, long v_tok_bytes)
{
    const long nk = (long)t_kv * kv_dim_k;
    const long nv = (long)t_kv * kv_dim_v;
    const long total = nk + nv;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        if (idx < nk) {
            const long t = idx / kv_dim_k; const int e = (int)(idx % kv_dim_k);
            Kw[idx] = __float2bfloat16(DQ_K_ELEM(K, t, k_tok_bytes, e));
        } else {
            const long j = idx - nk;
            const long t = j / kv_dim_v; const int e = (int)(j % kv_dim_v);
            Vw[j] = __float2bfloat16(DQ_V_ELEM(V, t, v_tok_bytes, e));
        }
    }
}

__global__ void fill_u8(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        uint8_t b = (uint8_t)x;
#if MEMRA_KV_KFMT == 1
        if ((b & 0x7f) == 0x7f) b ^= 0x01;   // e4m3: avoid the NaN codes
#endif
        p[i] = b;
    }
}
#if MEMRA_KV_KFMT == 0
// q8_0 / q5_1 scales: finite f16 in [2^-10, 2^-2] (byte patterns would give NaN/Inf scales).
__global__ void fix_scales(uint8_t* K, uint8_t* V, int t_kv, int kdk, int kdv, long ktb, long vtb) {
    const long nb = (long)t_kv * (kdk / 32);
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < nb; i += (long)gridDim.x * blockDim.x) {
        const long t = i / (kdk / 32), b = i % (kdk / 32);
        half* dk = (half*)(K + t * ktb + b * 34);
        dk[0] = __float2half(ldexpf(1.0f + (float)(i % 7) / 8.0f, -2 - (int)(i % 9)));
        half* dv = (half*)(V + t * vtb + b * 24);
        dv[0] = __float2half(ldexpf(1.0f + (float)(i % 5) / 8.0f, -3 - (int)(i % 8)));
        dv[1] = __float2half(-ldexpf(1.0f + (float)(i % 3) / 8.0f, -1 - (int)(i % 6)));
    }
}
#endif

static std::vector<int> ints(const char* s) {
    std::vector<int> v; std::string t(s);
    for (size_t p = 0; p <= t.size();) {
        size_t q = t.find(',', p); if (q == std::string::npos) q = t.size();
        v.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    const std::vector<int> dims = ints(argc > 1 ? argv[1] : "1024");
    const std::vector<int> tkvs = ints(argc > 2 ? argv[2] : "32768,65536,131072,262144");
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sms %d kfmt %d vfmt %d reps %d\n", prop.name, prop.multiProcessorCount,
           MEMRA_KV_KFMT, MEMRA_KV_VFMT, reps);
    for (int KD : dims) {
        const long ktb = (long)KD / 32 * K_BLK_B, vtb = (long)KD / 32 * V_BLK_B;
        const int tmax = *std::max_element(tkvs.begin(), tkvs.end());
        uint8_t *K, *V; __nv_bfloat16 *W[3][2]; unsigned long long* table;
        CK(cudaMalloc(&K, (size_t)tmax * ktb)); CK(cudaMalloc(&V, (size_t)tmax * vtb));
        for (auto& w : W) for (auto& p : w) CK(cudaMalloc(&p, (size_t)tmax * KD * 2));
        CK(cudaMalloc(&table, 12 * sizeof(unsigned long long)));
        fill_u8<<<1024, 256>>>(K, (size_t)tmax * ktb, 1); fill_u8<<<1024, 256>>>(V, (size_t)tmax * vtb, 2);
#if MEMRA_KV_KFMT == 0
        fix_scales<<<1024, 256>>>(K, V, tmax, KD, KD, ktb, vtb);
#endif
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        for (int tkv : tkvs) {
            unsigned long long tab[12] = {0};
            tab[0] = (unsigned long long)K; tab[1] = (unsigned long long)V; tab[7] = (unsigned long long)tkv;
            CK(cudaMemcpy(table, tab, sizeof(tab), cudaMemcpyHostToDevice));
            const double bytes = (double)tkv * (ktb + vtb + 4.0 * KD);
            const long groups = (long)tkv * (2L * KD) / 8, elems = (long)tkv * (2L * KD);
            // arms: 0 ref (per element), 1 engine, 2 engine _prime_table (fixed grid, as the engine
            // launches it: sm_count * 8 blocks), then ref and engine again
            const int order[5] = {0, 1, 2, 0, 1};
            std::vector<uint16_t> ref, got;
            for (int oi = 0; oi < 5; ++oi) {
                const int arm = order[oi];
                const long items = arm == 0 ? elems : ((KD % 8 == 0) ? groups : elems);
                unsigned nblk = (unsigned)std::min<long>((items + 255) / 256, 65535L * 16);
                if (arm == 2) nblk = (unsigned)std::min<long>((items + 255) / 256, (long)prop.multiProcessorCount * 8);
                std::vector<float> ms;
                for (int r = 0; r <= reps; ++r) {
                    CK(cudaMemset(W[arm][0], 0xff, (size_t)tkv * KD * 2));
                    CK(cudaMemset(W[arm][1], 0xff, (size_t)tkv * KD * 2));
                    CK(cudaEventRecord(e0));
                    if (arm == 0) ref_ws1<<<nblk, 256>>>(K, V, W[0][0], W[0][1], KD, KD, tkv, ktb, vtb);
                    else if (arm == 1) fa_dequant_kv_ws_bf16<<<nblk, 256>>>(K, V, W[1][0], W[1][1], KD, KD, tkv, ktb, vtb);
                    else fa_dequant_kv_ws_bf16_prime_table<<<nblk, 256>>>(table, 0ull, W[2][0], W[2][1], KD, KD, 0, ktb, vtb);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                    float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); if (r > 0) ms.push_back(t);
                }
                std::sort(ms.begin(), ms.end());
                const float med = ms[ms.size() / 2];
                std::vector<uint16_t>& dst = arm == 0 ? ref : got;
                dst.resize((size_t)tkv * KD * 2);
                CK(cudaMemcpy(dst.data(), W[arm][0], (size_t)tkv * KD * 2, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(dst.data() + (size_t)tkv * KD, W[arm][1], (size_t)tkv * KD * 2, cudaMemcpyDeviceToHost));
                size_t bad = 0;
                if (arm != 0) for (size_t i = 0; i < ref.size(); ++i) bad += ref[i] != got[i];
                const char* nm = arm == 0 ? "ref_ws1" : arm == 1 ? "engine" : "engine_pt";
                printf("kv_dim %5d Tkv %7d %-10s grid %8u %8.3f ms %6.1f GB/s mismatch %zu (min %.3f max %.3f)\n",
                       KD, tkv, nm, nblk, med, bytes / (med * 1e-3) / 1e9, bad, ms.front(), ms.back());
                fflush(stdout);
            }
        }
        CK(cudaFree(K)); CK(cudaFree(V)); CK(cudaFree(table));
        for (auto& w : W) for (auto& p : w) CK(cudaFree(p));
    }
    return 0;
}
