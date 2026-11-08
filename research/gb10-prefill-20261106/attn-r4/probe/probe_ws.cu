// probe_ws.cu: the prime's KV dequant workspace pass, the shipped fa_dequant_kv_ws_bf16 (one thread
// per element) against an 8-elements-per-thread form (one 16-byte bf16 store, the block's scale read
// once per group), on the model's KV shape (q8_0 K and q5_1 V, kv_dim 1,024 each). Every bf16 of the
// candidate's workspace is compared with the shipped kernel's.
//
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I<tree>/crates/memra-engine/cu probe_ws.cu -o probe_ws
// usage: probe_ws TKV[,TKV...] REPS
#include "flash_attn.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// Candidate: thread per 8 consecutive elements of one token row (kv_dim % 8 == 0), the same
// __float2bfloat16(DQ_*_ELEM(...)) per element, one 16-byte store.
static __device__ __forceinline__ uint4 ws8_pack(const float* f) {
    uint4 u;
    __nv_bfloat162 p0 = __halves2bfloat162(__float2bfloat16(f[0]), __float2bfloat16(f[1]));
    __nv_bfloat162 p1 = __halves2bfloat162(__float2bfloat16(f[2]), __float2bfloat16(f[3]));
    __nv_bfloat162 p2 = __halves2bfloat162(__float2bfloat16(f[4]), __float2bfloat16(f[5]));
    __nv_bfloat162 p3 = __halves2bfloat162(__float2bfloat16(f[6]), __float2bfloat16(f[7]));
    u.x = *(uint32_t*)&p0; u.y = *(uint32_t*)&p1; u.z = *(uint32_t*)&p2; u.w = *(uint32_t*)&p3;
    return u;
}
extern "C" __global__ void cand_ws8(
        const uint8_t* __restrict__ K, const uint8_t* __restrict__ V,
        __nv_bfloat16* __restrict__ Kw, __nv_bfloat16* __restrict__ Vw,
        int kv_dim_k, int kv_dim_v, int t_kv, long k_tok_bytes, long v_tok_bytes)
{
    const unsigned gk = (unsigned)kv_dim_k / 8u, gv = (unsigned)kv_dim_v / 8u;
    const unsigned nk = (unsigned)t_kv * gk, total = nk + (unsigned)t_kv * gv;
    for (unsigned g = blockIdx.x * blockDim.x + threadIdx.x; g < total; g += gridDim.x * blockDim.x) {
        float f[8];
        if (g < nk) {
            const unsigned t = g / gk; const int e0 = (int)(g - t * gk) * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = DQ_K_ELEM(K, (long)t, k_tok_bytes, e0 + i);
            *(uint4*)(Kw + (size_t)t * kv_dim_k + e0) = ws8_pack(f);
        } else {
            const unsigned j = g - nk;
            const unsigned t = j / gv; const int e0 = (int)(j - t * gv) * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) f[i] = DQ_V_ELEM(V, (long)t, v_tok_bytes, e0 + i);
            *(uint4*)(Vw + (size_t)t * kv_dim_v + e0) = ws8_pack(f);
        }
    }
}

__global__ void fill_u8(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u + seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (uint8_t)x;
    }
}
// Scales: finite f16 in [2^-10, 2^-2] (byte patterns would give NaN/Inf scales).
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

int main(int argc, char** argv) {
    std::vector<int> tkvs;
    {
        std::string s = argc > 1 ? argv[1] : "32768,65536,131072,262144";
        for (size_t p = 0; p <= s.size();) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            tkvs.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1;
        }
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    const int KD = 1024;
    const long ktb = KD / 32 * 34, vtb = KD / 32 * 24;
    const int tmax = *std::max_element(tkvs.begin(), tkvs.end());
    uint8_t *K, *V; __nv_bfloat16 *Kw0, *Vw0, *Kw1, *Vw1;
    CK(cudaMalloc(&K, (size_t)tmax * ktb)); CK(cudaMalloc(&V, (size_t)tmax * vtb));
    CK(cudaMalloc(&Kw0, (size_t)tmax * KD * 2)); CK(cudaMalloc(&Vw0, (size_t)tmax * KD * 2));
    CK(cudaMalloc(&Kw1, (size_t)tmax * KD * 2)); CK(cudaMalloc(&Vw1, (size_t)tmax * KD * 2));
    fill_u8<<<1024, 256>>>(K, (size_t)tmax * ktb, 1); fill_u8<<<1024, 256>>>(V, (size_t)tmax * vtb, 2);
    fix_scales<<<1024, 256>>>(K, V, tmax, KD, KD, ktb, vtb);
    CK(cudaDeviceSynchronize());
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sms %d kv_dim %d reps %d\n", prop.name, prop.multiProcessorCount, KD, reps);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int tkv : tkvs) {
        const double bytes = (double)tkv * (ktb + vtb + 4.0 * KD);
        for (int arm = 0; arm < 4; ++arm) {   // shipped, ws8, shipped, ws8
            const bool c = arm & 1;
            const long total = c ? (long)tkv * (2 * KD) / 8 : (long)tkv * (2 * KD);
            const unsigned nblk = (unsigned)std::min<long>((total + 255) / 256, 65535L * 16);
            std::vector<float> ms;
            for (int r = 0; r <= reps; ++r) {
                CK(cudaMemset(c ? Kw1 : Kw0, 0xff, (size_t)tkv * KD * 2));
                CK(cudaMemset(c ? Vw1 : Vw0, 0xff, (size_t)tkv * KD * 2));
                CK(cudaEventRecord(e0));
                if (c) cand_ws8<<<nblk, 256>>>(K, V, Kw1, Vw1, KD, KD, tkv, ktb, vtb);
                else fa_dequant_kv_ws_bf16<<<nblk, 256>>>(K, V, Kw0, Vw0, KD, KD, tkv, ktb, vtb);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float t = 0; CK(cudaEventElapsedTime(&t, e0, e1)); if (r > 0) ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
            const float med = ms[ms.size() / 2];
            size_t bad = 0;
            if (c) {
                std::vector<uint16_t> a((size_t)tkv * KD), b((size_t)tkv * KD);
                for (int pl = 0; pl < 2; ++pl) {
                    CK(cudaMemcpy(a.data(), pl ? Vw0 : Kw0, a.size() * 2, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(b.data(), pl ? Vw1 : Kw1, b.size() * 2, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < a.size(); ++i) bad += a[i] != b[i];
                }
            }
            printf("Tkv %7d %-8s %8.3f ms %6.1f GB/s mismatch %zu (min %.3f max %.3f)\n", tkv,
                   c ? "ws8" : "shipped", med, bytes / (med * 1e-3) / 1e9, bad, ms.front(), ms.back());
            fflush(stdout);
        }
    }
    return 0;
}
