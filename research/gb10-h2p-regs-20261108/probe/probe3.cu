// probe3.cu: time and byte-compare GB10 prime-attention candidates at Qwen3.8-27B's attention shape
// (24 Q / 4 KV heads x 256, causal, bf16 K/V workspace, f32 Q/O) on one GPU.
//
// build: nvcc -O3 -gencode arch=compute_121a,code=sm_121a probe3.cu -o probe3 -lcuda
//   (flash_attn.cu is main's file, included verbatim for the shipped h2p and its helpers)
// usage: probe3 T TKV[,TKV...] REPS KERNEL[,KERNEL...]
//        PROBE_LOAD=DIR probe3 - - REPS KERNELS   (real data: DIR/meta.txt "T T_kv", q.bin f32,
//                                                  k.bin / v.bin bf16 workspace rows of 1,024)
//        PROBE_CUBIN=FILE: a kernel named "x<name>" runs <name>'s entry symbol from FILE (driver API),
//        with <name>'s geometry: the same PTX assembled by another ptxas.
//   The first kernel is the reference: every other kernel's O is compared to it bit for bit.
//   One line per (kernel, T_kv): median ms of REPS after one warmup, causal TFLOPS and the share
//   of the 102 TFLOPS bf16 mma.sync ceiling, mismatching floats, non-finite outputs.
#include "flash_attn.cu"
#include "attn3.cuh"
#include "attn3b.cuh"
#include <cuda.h>
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

struct K { const char* name; attn_fn fn; int warps; int smem; int bq; int hpc; };
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* m_ = nullptr; cuGetErrorString(r_, &m_); \
    fprintf(stderr, "CU %s at %s:%d: %s\n", #x, __FILE__, __LINE__, m_ ? m_ : "?"); exit(1); } } while (0)
struct Run { std::string name; const K* k; CUfunction xf = nullptr; };
static const K KERNELS[] = {
    {"h2p", fa_prefill_qw_h2p, 8, 2 * 6 * BK * 256 + 64, 64, 2},
    {"h2", fa_prefill_qw_h2, 8, 2 * 4 * BK * 256, 64, 2},
    {"r32", cand_r32, 8, (int)QaLayout<16, false>::SMEM, 64, 2},
    {"r32w", cand_r32w, 8, (int)QaLayout<16, true>::SMEM, 64, 2},
    {"qa12", cand_qa12, 8, (int)QaLayout<12, false>::SMEM, 64, 2},
    {"qa12w", cand_qa12w, 8, (int)QaLayout<12, true>::SMEM, 64, 2},
    {"qa12v1", cand_qa12v1, 8, (int)QaLayout<12, false>::SMEM, 64, 2},
    {"qa12v2", cand_qa12v2, 8, (int)QaLayout<12, false>::SMEM, 64, 2},
    {"rb", cand_rb, 8, (int)QbLayout<16>::SMEM, 64, 2},
    {"qb12", cand_qb12, 8, (int)QbLayout<12>::SMEM, 64, 2},
    {"qb13", cand_qb13, 8, (int)QbLayout<13>::SMEM, 64, 2},
    {"qb14", cand_qb14, 8, (int)QbLayout<14>::SMEM, 64, 2},
    {"pb0", cand_pb0, 12, (int)PbLayout<16>::SMEM, 48, 2},
    {"pb1", cand_pb1, 12, (int)PbLayout<16>::SMEM, 48, 2},
    {"pb2", cand_pb2, 12, (int)PbLayout<16>::SMEM, 48, 2},
    {"pb1q15", cand_pb1q15, 12, (int)PbLayout<15>::SMEM, 48, 2},
    {"pb0q15", cand_pb0q15, 12, (int)PbLayout<15>::SMEM, 48, 2},
    {"r32e", cand_r32e, 8, (int)QaLayout<16, false>::SMEM, 64, 2},
    {"qa12e", cand_qa12e, 8, (int)QaLayout<12, false>::SMEM, 64, 2},
    {"qa13e", cand_qa13e, 8, (int)QaLayout<13, false>::SMEM, 64, 2},
    {"qa14e", cand_qa14e, 8, (int)QaLayout<14, false>::SMEM, 64, 2},
    {"pw0e", cand_pw0e, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1e", cand_pw1e, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw2e", cand_pw2e, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1q15e", cand_pw1q15e, 12, (int)PwLayout<15>::SMEM, 48, 2},
    {"pw0", cand_pw0, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1", cand_pw1, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw2", cand_pw2, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1q15", cand_pw1q15, 12, (int)PwLayout<15>::SMEM, 48, 2},
    {"pw0v2", cand_pw0v2, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1v2", cand_pw1v2, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw2v2", cand_pw2v2, 12, (int)PwLayout<16>::SMEM, 48, 2},
    {"pw1v1", cand_pw1v1, 12, (int)PwLayout<16>::SMEM, 48, 2},
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

static std::vector<int> split_ints(const std::string& s) {
    std::vector<int> v;
    for (size_t p = 0; p <= s.size();) {
        size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
        v.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1;
    }
    return v;
}
static void read_file(const std::string& path, void* dst, size_t bytes) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    if (fread(dst, 1, bytes, f) != bytes) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
    fclose(f);
}

int main(int argc, char** argv) {
    const char* load = getenv("PROBE_LOAD");
    int T = 4096;
    std::vector<int> tkvs;
    if (load) {
        FILE* f = fopen((std::string(load) + "/meta.txt").c_str(), "r");
        int tkv = 0;
        if (!f || fscanf(f, "%d %d", &T, &tkv) != 2) { fprintf(stderr, "bad meta in %s\n", load); return 2; }
        fclose(f);
        tkvs.push_back(tkv);
    } else {
        T = argc > 1 ? atoi(argv[1]) : 4096;
        tkvs = split_ints(argc > 2 ? argv[2] : "4096,32768,65536,131072");
    }
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    std::vector<Run> ks;
    CUmodule xmod = nullptr;
    {
        CK(cudaFree(0));   // the primary context, shared with the driver-API module
        std::string s = argc > 4 ? argv[4] : "h2p,qa12";
        for (size_t p = 0; p <= s.size();) {
            size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
            std::string n = s.substr(p, q - p); p = q + 1;
            const bool x = n.size() > 1 && n[0] == 'x';
            const std::string base = x ? n.substr(1) : n;
            const K* f = nullptr;
            for (const K& k : KERNELS) if (base == k.name) f = &k;
            if (!f) { fprintf(stderr, "unknown kernel %s\n", n.c_str()); return 2; }
            Run r{n, f};
            if (x) {
                const char* cub = getenv("PROBE_CUBIN");
                if (!cub) { fprintf(stderr, "%s needs PROBE_CUBIN\n", n.c_str()); return 2; }
                if (!xmod) CU(cuModuleLoad(&xmod, cub));
                cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, f->fn));
                const char* sym = nullptr;
                CK(cudaFuncGetName(&sym, (const void*)f->fn));
                CU(cuModuleGetFunction(&r.xf, xmod, sym));
            }
            ks.push_back(r);
        }
    }
    const int HD = 256, NH = 24, NHKV = 4, KVD = NHKV * HD;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("device %s sms %d T %d reps %d data %s\n", prop.name, prop.multiProcessorCount, T, reps,
           load ? load : "seeded-gauss");
    for (const Run& r : ks) {
        const K* k = r.k;
        int regs = 0, local = 0, occ = 0;
        if (r.xf) {
            CU(cuFuncSetAttribute(r.xf, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, k->smem));
            CU(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, r.xf));
            CU(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, r.xf));
            CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, r.xf, 32 * k->warps, k->smem));
        } else {
            cudaFuncAttributes a; CK(cudaFuncGetAttributes(&a, k->fn));
            CK(cudaFuncSetAttribute(k->fn, cudaFuncAttributeMaxDynamicSharedMemorySize, k->smem));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k->fn, 32 * k->warps, k->smem));
            regs = a.numRegs; local = (int)a.localSizeBytes;
        }
        printf("kernel %-8s regs %d local %d B smem %d B warps %d bq %d blocks/SM %d\n", r.name.c_str(), regs,
               local, k->smem, k->warps, k->bq, occ);
    }
    const int tkv_max = *std::max_element(tkvs.begin(), tkvs.end());
    float* q; __nv_bfloat16 *kw, *vw;
    CK(cudaMalloc(&q, (size_t)T * NH * HD * 4));
    CK(cudaMalloc(&kw, (size_t)tkv_max * KVD * 2));
    CK(cudaMalloc(&vw, (size_t)tkv_max * KVD * 2));
    std::vector<float*> outs(ks.size());
    for (auto& o : outs) CK(cudaMalloc(&o, (size_t)T * NH * HD * 4));
    if (load) {
        std::vector<char> h((size_t)tkv_max * KVD * 2);
        read_file(std::string(load) + "/k.bin", h.data(), h.size());
        CK(cudaMemcpy(kw, h.data(), h.size(), cudaMemcpyHostToDevice));
        read_file(std::string(load) + "/v.bin", h.data(), h.size());
        CK(cudaMemcpy(vw, h.data(), h.size(), cudaMemcpyHostToDevice));
        h.resize((size_t)T * NH * HD * 4);
        read_file(std::string(load) + "/q.bin", h.data(), h.size());
        CK(cudaMemcpy(q, h.data(), h.size(), cudaMemcpyHostToDevice));
    } else {
        init_f32<<<1024, 256>>>(q, (size_t)T * NH * HD, 1);
        init_bf16<<<1024, 256>>>(kw, (size_t)tkv_max * KVD, 2);
        init_bf16<<<1024, 256>>>(vw, (size_t)tkv_max * KVD, 3);
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> ref((size_t)T * NH * HD), got((size_t)T * NH * HD);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int tkv : tkvs) {
        if (tkv < T) { fprintf(stderr, "T_kv %d < T %d\n", tkv, T); return 2; }
        const double pairs = (double)T * (tkv - T) + (double)T * (T + 1) / 2.0;
        const double flop = 4.0 * HD * NH * pairs;
        for (size_t i = 0; i < ks.size(); ++i) {
            const K* k = ks[i].k;
            dim3 grid((T + k->bq - 1) / k->bq, NH / k->hpc), block(32, k->warps);
            std::vector<float> ms;
            for (int r = 0; r <= reps; ++r) {   // r = 0 is the warmup
                CK(cudaMemset(outs[i], 0xff, (size_t)T * NH * HD * 4));
                CK(cudaEventRecord(e0));
                if (ks[i].xf) {
                    int hd = HD, nh = NH, nhkv = NHKV, tt = T, tk = tkv, cz = 1, kd = KVD, vd = KVD;
                    float sc = 1.0f / 16.0f;
                    float* o = outs[i];
                    void* args[] = {&q, &kw, &vw, &o, &hd, &nh, &nhkv, &tt, &tk, &sc, &cz, &kd, &vd};
                    CU(cuLaunchKernel(ks[i].xf, grid.x, grid.y, 1, 32, k->warps, 1, k->smem, 0, args, nullptr));
                } else {
                    k->fn<<<grid, block, k->smem>>>(q, kw, vw, outs[i], HD, NH, NHKV, T, tkv,
                                                    1.0f / 16.0f, 1, KVD, KVD);
                }
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
            const double tf = flop / (med * 1e-3) / 1e12;
            printf("T %d Tkv %7d %-8s %9.3f ms %6.2f TFLOPS %5.3f of 102 mismatch %zu nonfinite %zu (min %.3f max %.3f)\n",
                   T, tkv, ks[i].name.c_str(), med, tf, tf / 102.0, bad, nonfin, ms.front(), ms.back());
            fflush(stdout);
        }
    }
    return 0;
}
