// probe_pp.cu (from the attention lane's probe_attn.cu): time and byte-compare prime-attention kernels at Qwen3.8-27B's attention shape
// (24 Q / 4 KV heads x 256, causal, bf16 K/V workspace, f32 Q/O) on one GPU.
//
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a -I<tree>/crates/memra-engine/cu \
//          probe_pp.cu -o probe_pp
// usage: probe_pp T TKV[,TKV...] REPS KERNEL[,KERNEL...]
//   The first kernel is the reference: every other kernel's O is compared to it bit for bit.
//   Prints one line per (kernel, T_kv): median ms, causal TFLOPS, mismatching floats.
#include "flash_attn.cu"
#include "attn_pp.cuh"
#include "attn_ex2.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

typedef void (*attn_fn)(const float*, const __nv_bfloat16*, const __nv_bfloat16*, float*,
                        int, int, int, int, int, float, int, int, int);

struct K { const char* name; attn_fn fn; int warps; int smem; int hpc = 2; };
static const K KERNELS[] = {
    {"h2", fa_prefill_qw_h2, 8, 2 * 4 * BK * 256},
    {"h2p", fa_prefill_qw_h2p, 8, 2 * 6 * BK * 256 + 64},
    {"pp0", cand_pp0, 8, 2 * 6 * BK * 256 + 64},
    {"ppw", cand_ppw, 8, 2 * 6 * BK * 256 + 64},
    {"ppg", cand_ppg, 8, 2 * 6 * BK * 256 + 64},
    {"ppl", cand_ppl, 8, 2 * 6 * BK * 256 + 64},
    {"ppv", cand_ppv, 8, 2 * 6 * BK * 256 + 64},
    {"ex2", cand_ex2, 8, 2 * 6 * BK * 256 + 64},
    {"ex2b", cand_ex2b, 8, 2 * 6 * BK * 256 + 64},
};

__device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
// approximately N(0, 1): sum of four uniforms, centered, scaled to unit variance
__device__ __forceinline__ float gauss(uint32_t i, uint32_t seed) {
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) s += (mix32(i * 4u + k + seed * 0x9e3779b9U) >> 8) * (1.0f / 16777216.0f);
    return (s - 2.0f) * 1.7320508f;
}
__global__ void init_f32(float* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = gauss((uint32_t)i, seed);
}
__global__ void init_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2bfloat16(gauss((uint32_t)i, seed));
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    std::vector<int> tkvs;
    {
        std::string s = argc > 2 ? argv[2] : "4096,32768,65536,131072";
        for (size_t p = 0; p <= s.size();) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            tkvs.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1;
        }
    }
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    std::vector<const K*> ks;
    {
        std::string s = argc > 4 ? argv[4] : "h2,sb2";
        for (size_t p = 0; p <= s.size();) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            std::string n = s.substr(p, q - p); p = q + 1;
            const K* f = nullptr;
            for (const K& k : KERNELS) if (n == k.name) f = &k;
            if (!f) { fprintf(stderr, "unknown kernel %s\n", n.c_str()); return 2; }
            ks.push_back(f);
        }
    }
    const int HD = 256, NH = 24, NHKV = 4, KVD = NHKV * HD;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sms %d T %d reps %d\n", prop.name, prop.multiProcessorCount, T, reps);
    for (const K* k : ks) {
        cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, k->fn));
        CK(cudaFuncSetAttribute(k->fn, cudaFuncAttributeMaxDynamicSharedMemorySize, k->smem));
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k->fn, 32 * k->warps, k->smem));
        printf("kernel %-8s regs %d local %zu B smem %d B blocks/SM %d\n", k->name, a.numRegs,
               (size_t)a.localSizeBytes, k->smem, occ);
    }
    const int tkv_max = *std::max_element(tkvs.begin(), tkvs.end());
    float* q; __nv_bfloat16 *kw, *vw;
    CK(cudaMalloc(&q, (size_t)T * NH * HD * 4));
    CK(cudaMalloc(&kw, (size_t)tkv_max * KVD * 2));
    CK(cudaMalloc(&vw, (size_t)tkv_max * KVD * 2));
    std::vector<float*> outs(ks.size());
    for (auto& o : outs) CK(cudaMalloc(&o, (size_t)T * NH * HD * 4));
    init_f32<<<1024, 256>>>(q, (size_t)T * NH * HD, 1);
    init_bf16<<<1024, 256>>>(kw, (size_t)tkv_max * KVD, 2);
    init_bf16<<<1024, 256>>>(vw, (size_t)tkv_max * KVD, 3);
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)T * NH * HD), got((size_t)T * NH * HD);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int tkv : tkvs) {
        if (tkv < T) { fprintf(stderr, "T_kv %d < T %d\n", tkv, T); return 2; }
        const double pairs = (double)T * (tkv - T) + (double)T * (T + 1) / 2.0;
        const double flop = 4.0 * HD * NH * pairs;
        for (size_t i = 0; i < ks.size(); ++i) {
            const K* k = ks[i];
            dim3 grid((T + BLOCK_Q - 1) / BLOCK_Q, NH / k->hpc), block(32, k->warps);
            std::vector<float> ms;
            for (int r = 0; r <= reps; ++r) {   // r = 0 is the warmup
                CK(cudaMemset(outs[i], 0xff, (size_t)T * NH * HD * 4));
                CK(cudaEventRecord(e0));
                k->fn<<<grid, block, k->smem>>>(q, kw, vw, outs[i], HD, NH, NHKV, T, tkv,
                                                1.0f / 16.0f, 1, KVD, KVD);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaGetLastError());
                float t = 0; CK(cudaEventElapsedTime(&t, e0, e1));
                if (r > 0) ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
            const float med = ms[ms.size() / 2];
            CK(cudaMemcpy(i == 0 ? ref.data() : got.data(), outs[i], ref.size() * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0, nonfin = 0;
            const std::vector<float>& v = i == 0 ? ref : got;
            for (size_t j = 0; j < v.size(); ++j) {
                if (!std::isfinite(v[j])) ++nonfin;
                if (i > 0 && memcmp(&ref[j], &got[j], 4) != 0) ++bad;
            }
            printf("T %d Tkv %7d %-8s %9.3f ms %6.2f TFLOPS mismatch %zu nonfinite %zu (min %.3f max %.3f)\n",
                   T, tkv, k->name, med, flop / (med * 1e-3) / 1e12, bad, nonfin, ms.front(), ms.back());
            fflush(stdout);
        }
    }
    return 0;
}
