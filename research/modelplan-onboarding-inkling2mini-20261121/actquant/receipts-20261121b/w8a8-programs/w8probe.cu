
// W8A8 program comparison probe (lane D, #1396 review item 4). Program A: the repo kernel (kind::mxf8f6f4,
// e4m3 x e4m3, ue8m0 per 32). Program B: e4m3 -> bf16 dequant (exact on these inputs, checked on the host), then
// mma.sync m16n8k16 bf16 x bf16 with an fp32 accumulator chained over ascending K. Output per group: acc[g][N].
#include "/home/avifenesh/.cache/inkling2mini/wt-actq/crates/memra-engine/cu/inkling2_actq.cu"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(3); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<uint8_t> v(n);
    if (n && fread(v.data(), 1, n, f) != (size_t)n) exit(3); fclose(f); return v;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "cuda %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(4);} } while (0)
static void* up(const std::vector<uint8_t>& v) { void* d; CK(cudaMalloc(&d, v.size() + 64)); CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice)); return d; }
__global__ void bf16_chain(const uint16_t* __restrict__ W, const uint16_t* __restrict__ A, const uint32_t* __restrict__ groups,
                           float* __restrict__ out, int N, int K) {
    const uint32_t* g = groups + blockIdx.y * 10;
    const uint32_t e = g[0], r = g[2];
    const int lane = threadIdx.x, gq = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 16;
    const uint16_t* w0 = W + ((size_t)e * N + n0 + gq) * K;
    const uint16_t* w1 = w0 + (size_t)8 * K;
    const uint16_t* a = A + (size_t)r * K;  // all 8 B columns = the same row; column gq reads it
    float d[4] = {0, 0, 0, 0};
    for (int k = 0; k < K; k += 16) {
        uint32_t a0 = *(const uint32_t*)(w0 + k + 2 * t), a1 = *(const uint32_t*)(w1 + k + 2 * t);
        uint32_t a2 = *(const uint32_t*)(w0 + k + 8 + 2 * t), a3 = *(const uint32_t*)(w1 + k + 8 + 2 * t);
        uint32_t b0 = *(const uint32_t*)(a + k + 2 * t), b1 = *(const uint32_t*)(a + k + 8 + 2 * t);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (t == 0) {  // column 0 (= the row): d0 row gq, d2 row gq + 8
        out[(size_t)blockIdx.y * N + n0 + gq] = d[0];
        out[(size_t)blockIdx.y * N + n0 + gq + 8] = d[2];
    }
}
int main(int argc, char** argv) {
    std::string d = argv[1]; int N = atoi(argv[2]), K = atoi(argv[3]), R = atoi(argv[4]), ng = atoi(argv[5]);
    void* act = up(rd(d + "/act.bin")); void* actsc = up(rd(d + "/actsc.bin")); float* aam = (float*)up(rd(d + "/actamax.bin"));
    void* w = up(rd(d + "/w.bin")); void* wsc = up(rd(d + "/wsc.bin")); float* wam = (float*)up(rd(d + "/wamax.bin"));
    auto gh = rd(d + "/groups1.bin"); uint32_t* gd = (uint32_t*)up(gh);
    void* out; CK(cudaMalloc(&out, (size_t)R * N * 2)); float* acc; CK(cudaMalloc(&acc, (size_t)R * N * 4));
    std::vector<float> A((size_t)ng * N), B((size_t)ng * N), tmp((size_t)N);
    const uint32_t* gv = (const uint32_t*)gh.data();
    for (int gi = 0; gi < ng; ++gi) {
        int rc = memra_ink2_expert_gemm(1, 1, act, actsc, aam, w, wsc, wam, 1.0f, gd + gi * 10, 1, out, acc, N, K, 0, 0);
        if (rc) { fprintf(stderr, "rc %d\n", rc); return 5; }
        CK(cudaMemcpy(A.data() + (size_t)gi * N, acc + (size_t)gv[gi * 10 + 2] * N, N * 4, cudaMemcpyDeviceToHost));
    }
    void* wbf = up(rd(d + "/wbf.bin")); void* abf = up(rd(d + "/abf.bin")); float* ob; CK(cudaMalloc(&ob, (size_t)ng * N * 4));
    bf16_chain<<<dim3(N / 16, ng), 32>>>((const uint16_t*)wbf, (const uint16_t*)abf, gd, ob, N, K);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(B.data(), ob, B.size() * 4, cudaMemcpyDeviceToHost));
    FILE* f = fopen((d + "/acc.mxf8").c_str(), "wb"); fwrite(A.data(), 4, A.size(), f); fclose(f);
    f = fopen((d + "/acc.bf16mma").c_str(), "wb"); fwrite(B.data(), 4, B.size(), f); fclose(f);
    printf("done %s\n", d.c_str());
    return 0;
}
