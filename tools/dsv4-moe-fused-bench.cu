// Standalone timing bench for the fused MoE pair on one card at the TP/EP partition shape
// (memra #710): a rank's 128 local experts of the 256 bank, 6 slots per token row of which
// `local` are this rank's. Every launch takes a different expert set, so the weights stream from
// DRAM rather than L2. Variants instantiate the kernel templates directly; `packed` streams a
// stage-major copy of the weights (each warp's stage contiguous) through a twin of the issue
// loop, to price the access pattern. Timing only: the outputs are not checked here.
// nvcc -t 8 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo
//   tools/dsv4-moe-fused-bench.cu -lcublasLt -lcublas -ldl -o <dir>/moe_bench
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void api(int rc) { if (rc) throw std::runtime_error("launcher rc=" + std::to_string(rc)); }

__global__ void bench_fill(uint8_t* p, size_t n, uint32_t seed, int scale_bytes) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = scale_bytes ? (uint8_t)(0x30 + h % 9) : (uint8_t)(h >> 24);
    }
}

constexpr int NE = 128, NG = 256, HID = 4096, INTER = 2048, TOPK = 6;

// ---- the stage-major twin. A warp's stage (8 rows x KC/2 code bytes, then 8 rows x KC/16 scale
// bytes) is one contiguous block, so a stage is one run of 16-byte copies from one place.
template<int KC>
struct BenchPack {
    static constexpr int CODE = KC / 2, SCB = KC / 16, BLK = 8 * (CODE + SCB);
};

__global__ void bench_repack(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                             uint8_t* __restrict__ out, int rows, int in, int kc) {
    const int code = kc / 2, scb = kc / 16, blk = 8 * (code + scb), nch = in / kc;
    const long total = (long)(rows / 8) * nch * blk;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
        const long bc = i / blk;
        const int o = (int)(i % blk), b = (int)(bc / nch), c = (int)(bc % nch);
        if (o < 8 * code) {
            const int r = o / code, j = o % code;
            out[i] = codes[(long)(b * 8 + r) * (in / 2) + (long)c * code + j];
        } else {
            const int r = (o - 8 * code) / scb, j = (o - 8 * code) % scb;
            out[i] = scales[(long)(b * 8 + r) * (in / 16) + (long)c * scb + j];
        }
    }
}

template<int KC, int STAGES>
__device__ __forceinline__ void bench_issue_packed(unsigned char* ring, const uint8_t* blk0, int nch,
                                                   int c, int lane, bool live) {
    using R = Dsv4M1Ring<KC, STAGES>;
    using P = BenchPack<KC>;
    static_assert(P::SCB % 16 == 0, "whole 16-byte scale chunks");
    constexpr int CC = 8 * P::CODE / 16, SC = 8 * P::SCB / 16;
    if (live && c < nch) {
        unsigned char* st = ring + (c % STAGES) * R::STAGE;
        const uint8_t* src = blk0 + (size_t)c * P::BLK;
#pragma unroll
        for (int q0 = 0; q0 < CC + SC; q0 += 32) {
            const int q = q0 + lane;
            if (q < CC) {
                const int row = q / (P::CODE / 16), off = (q % (P::CODE / 16)) * 16;
                sk_cp16(st + row * R::PITCH + off, src + row * P::CODE + off);
            } else if (q < CC + SC) {
                const int j = q - CC, row = j / (P::SCB / 16), off = (j % (P::SCB / 16)) * 16;
                sk_cp16(st + row * R::PITCH + P::CODE + off, src + 8 * P::CODE + row * P::SCB + off);
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// dsv4_m1_dot with the packed issue.
template<int KC, int STAGES>
__device__ __forceinline__ void bench_dot_packed(float (&acc)[4], unsigned char* ring,
                                                 const uint32_t* As, const uint8_t* blk0, int nch,
                                                 int lane) {
    using R = Dsv4M1Ring<KC, STAGES>;
    constexpr int CODE = R::CODE, PITCH = R::PITCH;
    const int gq = lane >> 2, t = lane & 3;
    const uint32_t pick = (uint32_t)t | ((uint32_t)(t + 4) << 4);
    for (int c = 0; c < nch; c++) {
        bench_issue_packed<KC, STAGES>(ring, blk0, nch, c + STAGES - 1, lane, true);
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1) : "memory");
        __syncwarp();
        const unsigned char* rowb = ring + (c % STAGES) * R::STAGE + gq * PITCH;
#pragma unroll
        for (int s = 0; s < KC / 128; s++) {
            const unsigned char* rowp = rowb + s * 64;
            const uint2 scw = *reinterpret_cast<const uint2*>(rowb + CODE + s * 8);
            const uint32_t sw[2] = {kqs_e4m3fn_clear_nan(scw.x), kqs_e4m3fn_clear_nan(scw.y)};
            const uint2* ap =
                reinterpret_cast<const uint2*>(As) + ((size_t)c * (KC / 16) + s * 8) * 4 + t;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const uint4 v = *reinterpret_cast<const uint4*>(rowp + q * 16);
                const uint32_t s01 = kqs_e4m3x2_f16x2(q & 1 ? sw[q >> 1] >> 16 : sw[q >> 1]);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t x = __byte_perm(h ? v.z : v.x, h ? v.w : v.y, pick);
                    const uint32_t s2 = __byte_perm(s01, 0u, h ? 0x3232u : 0x1010u);
                    uint32_t b0, b1;
                    kqs_e2m1_f16x4(x, b0, b1);
                    b0 = kqs_hmul2(b0, s2);
                    b1 = kqs_hmul2(b1, s2);
                    const uint2 a2 = ap[(q * 2 + h) * 4];
                    const unsigned a[4] = {a2.x, a2.x, a2.y, a2.y};
                    sk_mma(acc, a, b0, b1);
                }
            }
        }
        __syncwarp();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// dsv4_moe_fused_gu_kernel over the packed bank: ptable[proj * n_expert + le] is the expert's
// packed projection, n8 block b's chunks at (b * nch + c) * BLK.
template<int WP, int KC, int STAGES>
static __global__ void __launch_bounds__(2 * WP * 32)
bench_gu_packed_kernel(const unsigned long long* __restrict__ ptable, int n_expert, int global_experts,
                       int first_expert, int slots_per_row, const int* __restrict__ sel,
                       const float* __restrict__ selw, const float* __restrict__ scale2,
                       const float* __restrict__ xf, float* __restrict__ H, int in_f, int out_f,
                       float limit, int* __restrict__ fault) {
    MEMRA_PDL_CHAIN_ENTRY();
    using R = Dsv4M1Ring<KC, STAGES>;
    using P = BenchPack<KC>;
    extern __shared__ __align__(16) unsigned char fz_smem[];
    __shared__ float s_scale[64];
    __shared__ float s_rs;
    __shared__ float s_up[WP][4][2];
    uint32_t* As = reinterpret_cast<uint32_t*>(fz_smem);
    const int p = blockIdx.y, lane = threadIdx.x, warp = threadIdx.y;
    const int e = sel[p];
    const bool in_range = e >= 0 && e < global_experts;
    const int le = e - first_expert;
    const bool valid = in_range && le >= 0 && le < n_expert;
    if (in_range && !valid) return;
    const bool up = warp >= WP;
    const int proj = up ? 2 : 0;
    const int n0 = (blockIdx.x * WP + (up ? warp - WP : warp)) * 8;
    unsigned char* ring = fz_smem + (size_t)in_f * 2 + (size_t)warp * R::BYTES;
    const int nch = in_f / KC;
    const uint8_t* blk0 = nullptr;
    if (valid) blk0 = (const uint8_t*)ptable[(size_t)proj * n_expert + le] + (size_t)(n0 / 8) * nch * P::BLK;
#pragma unroll
    for (int c = 0; c < STAGES - 1; c++) bench_issue_packed<KC, STAGES>(ring, blk0, nch, c, lane, valid);
    const bool bad = dsv4_moe_fused_mirror<2 * WP>(xf + (size_t)(p / slots_per_row) * in_f, in_f, As,
                                                    s_scale, &s_rs, warp, lane);
    const bool any_bad = __syncthreads_or(bad);
    if (lane == 0 && warp == 0 && fault && any_bad && blockIdx.x == 0) atomicOr(fault, 2);
    const int gq = lane >> 2, t = lane & 3;
    if (!valid) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        if (!up && gq == 0) {
            float* hrow = H + (size_t)p * out_f + n0 + 2 * t;
            hrow[0] = 0.0f;
            hrow[1] = 0.0f;
        }
        return;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bench_dot_packed<KC, STAGES>(acc, ring, As, blk0, nch, lane);
    if (up && gq == 0) {
        s_up[warp - WP][t][0] = acc[0];
        s_up[warp - WP][t][1] = acc[1];
    }
    __syncthreads();
    if (!up && gq == 0) {
        const float rs = s_rs;
        const float mg = scale2[e * 3], mu = scale2[e * 3 + 2], w = selw[p];
        float* hrow = H + (size_t)p * out_f + n0 + 2 * t;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            float gv = acc[j] * rs;
            gv *= mg;
            float uv = s_up[warp][t][j] * rs;
            uv *= mu;
            float u = fminf(fmaxf(uv, -limit), limit);
            float g = fminf(gv, limit);
            float h = g * dsv4_sigmoid(g) * u;
            h *= w;
            hrow[j] = h;
        }
    }
}

// Decomposition twins of the standard dot: MODE 1 streams and waits but skips the decode and the
// MMAs (one byte per chunk keeps the loads live); MODE 2 decodes and runs the MMAs on the first
// stage only, never loading another.
template<int KC, int STAGES, int MODE>
__device__ __forceinline__ void bench_dot_mode(float (&acc)[4], unsigned char* ring,
                                               const uint32_t* As, const uint8_t* wq0,
                                               const uint8_t* ws0, long row_bytes, int sc_row,
                                               int nch, int lane) {
    using R = Dsv4M1Ring<KC, STAGES>;
    constexpr int CODE = R::CODE, PITCH = R::PITCH;
    const int gq = lane >> 2, t = lane & 3;
    const uint32_t pick = (uint32_t)t | ((uint32_t)(t + 4) << 4);
    for (int c = 0; c < nch; c++) {
        if (MODE != 2)
            dsv4_m1_issue<KC, STAGES>(ring, wq0, ws0, row_bytes, sc_row, nch, c + STAGES - 1, lane, true);
        else
            asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1) : "memory");
        __syncwarp();
        const unsigned char* rowb = ring + ((MODE == 2 ? 0 : c) % STAGES) * R::STAGE + gq * PITCH;
        if (MODE == 1) {
            acc[0] += (float)rowb[lane & 63];
            __syncwarp();
            continue;
        }
#pragma unroll
        for (int s = 0; s < KC / 128; s++) {
            const unsigned char* rowp = rowb + s * 64;
            const uint2 scw = *reinterpret_cast<const uint2*>(rowb + CODE + s * 8);
            const uint32_t sw[2] = {kqs_e4m3fn_clear_nan(scw.x), kqs_e4m3fn_clear_nan(scw.y)};
            const uint2* ap =
                reinterpret_cast<const uint2*>(As) + ((size_t)c * (KC / 16) + s * 8) * 4 + t;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const uint4 v = *reinterpret_cast<const uint4*>(rowp + q * 16);
                const uint32_t s01 = kqs_e4m3x2_f16x2(q & 1 ? sw[q >> 1] >> 16 : sw[q >> 1]);
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const uint32_t x = __byte_perm(h ? v.z : v.x, h ? v.w : v.y, pick);
                    const uint32_t s2 = __byte_perm(s01, 0u, h ? 0x3232u : 0x1010u);
                    uint32_t b0, b1;
                    kqs_e2m1_f16x4(x, b0, b1);
                    b0 = kqs_hmul2(b0, s2);
                    b1 = kqs_hmul2(b1, s2);
                    const uint2 a2 = ap[(q * 2 + h) * 4];
                    const unsigned a[4] = {a2.x, a2.x, a2.y, a2.y};
                    sk_mma(acc, a, b0, b1);
                }
            }
        }
        __syncwarp();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

// The gate/up kernel's streaming skeleton with the decomposition dot and no x mirror (MIRROR 0)
// or with it (MIRROR 1).
template<int WP, int KC, int STAGES, int MODE, int MIRROR>
static __global__ void __launch_bounds__(2 * WP * 32)
bench_gu_mode_kernel(const unsigned long long* __restrict__ table, const int* __restrict__ sel,
                     const float* __restrict__ xf, float* __restrict__ H, int in_f, int out_f) {
    using R = Dsv4M1Ring<KC, STAGES>;
    extern __shared__ __align__(16) unsigned char fz_smem[];
    __shared__ float s_scale[64];
    __shared__ float s_rs;
    uint32_t* As = reinterpret_cast<uint32_t*>(fz_smem);
    const int p = blockIdx.y, lane = threadIdx.x, warp = threadIdx.y;
    const int e = sel[p];
    if (e < 0 || e >= NE) return;
    const bool up = warp >= WP;
    const int proj = up ? 2 : 0;
    const int n0 = (blockIdx.x * WP + (up ? warp - WP : warp)) * 8;
    unsigned char* ring = fz_smem + (size_t)in_f * 2 + (size_t)warp * R::BYTES;
    const int nch = in_f / KC, sc_row = in_f / 16;
    const long row_bytes = in_f / 2;
    const uint8_t* wq0 = (const uint8_t*)table[(size_t)(2 * proj) * NE + e] + (size_t)n0 * row_bytes;
    const uint8_t* ws0 = (const uint8_t*)table[(size_t)(2 * proj + 1) * NE + e] + (size_t)n0 * sc_row;
#pragma unroll
    for (int c = 0; c < STAGES - 1; c++)
        dsv4_m1_issue<KC, STAGES>(ring, wq0, ws0, row_bytes, sc_row, nch, c, lane, true);
    if (MIRROR) dsv4_moe_fused_mirror<2 * WP>(xf, in_f, As, s_scale, &s_rs, warp, lane);
    __syncthreads();
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bench_dot_mode<KC, STAGES, MODE>(acc, ring, As, wq0, ws0, row_bytes, sc_row, nch, lane);
    if ((lane >> 2) == 0) H[(size_t)p * out_f + n0 + 2 * (lane & 3)] = acc[0] + acc[1];
}

template<int MODE, int MIRROR>
static void bench_gu_mode(const unsigned long long* table, const int* sel, const float* xf, float* h,
                          int rows, cudaStream_t s) {
    constexpr int WP = 4, KC = 256, ST = 2;
    const size_t smem = (size_t)HID * 2 + (size_t)2 * WP * Dsv4M1Ring<KC, ST>::BYTES;
    bench_gu_mode_kernel<WP, KC, ST, MODE, MIRROR><<<dim3(INTER / (8 * WP), rows * TOPK), dim3(32, 2 * WP), smem, s>>>(
        table, sel, xf, h, HID, INTER);
}

// The down kernel at other template parameters (the partition form: no slot sum).
template<int W, int KC, int ST>
static void bench_down_std(const unsigned long long* table, const int* sel, const float* scale2,
                           const float* h, float* c, int rows, cudaStream_t s) {
    const size_t smem = (size_t)INTER * 2 + (size_t)W * Dsv4M1Ring<KC, ST>::BYTES;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(dsv4_moe_fused_down_kernel<W, KC, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dsv4_moe_fused_down_kernel<W, KC, ST><<<dim3(HID / (8 * W), rows * TOPK), dim3(32, W), smem, s>>>(
        table, NE, NG, 0, sel, scale2, h, c, nullptr, nullptr, nullptr, TOPK, INTER, HID,
        (long)(INTER / 2), nullptr);
}

// The standard gate/up kernel at other template parameters.
template<int WP, int KC, int ST>
static void bench_gu_std(const unsigned long long* table, const int* sel, const float* selw,
                         const float* scale2, const float* xf, float* h, int rows, int* fault,
                         cudaStream_t s) {
    const size_t smem = (size_t)HID * 2 + (size_t)2 * WP * Dsv4M1Ring<KC, ST>::BYTES;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(dsv4_moe_fused_gu_kernel<WP, KC, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dsv4_moe_fused_gu_kernel<WP, KC, ST><<<dim3(INTER / (8 * WP), rows * TOPK), dim3(32, 2 * WP), smem, s>>>(
        table, NE, NG, 0, TOPK, sel, selw, scale2, xf, nullptr, nullptr, h, nullptr, HID, INTER, 10.0f, (long)(HID / 2), fault);
}

template<int WP, int KC, int ST>
static void bench_gu_pk(const unsigned long long* ptable, const int* sel, const float* selw,
                        const float* scale2, const float* xf, float* h, int rows, int* fault,
                        cudaStream_t s) {
    const size_t smem = (size_t)HID * 2 + (size_t)2 * WP * Dsv4M1Ring<KC, ST>::BYTES;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(bench_gu_packed_kernel<WP, KC, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    bench_gu_packed_kernel<WP, KC, ST><<<dim3(INTER / (8 * WP), rows * TOPK), dim3(32, 2 * WP), smem, s>>>(
        ptable, NE, NG, 0, TOPK, sel, selw, scale2, xf, h, HID, INTER, 10.0f, fault);
}

struct Bank {
    uint8_t* codes[3];   // [NE][rows][in/2], proj 0 gate, 1 down, 2 up
    uint8_t* scales[3];  // [NE][rows][in/16]
    int rows[3], in[3];
};

int main(int argc, char** argv) {
    try {
        const int iters = argc > 1 ? atoi(argv[1]) : 200;
        const int local = argc > 2 ? atoi(argv[2]) : 3;
        const int rows = argc > 3 ? atoi(argv[3]) : 1;
        ck(cudaSetDevice(0));
        cudaStream_t s; ck(cudaStreamCreate(&s));
        Bank b{};
        const int R[3] = {INTER, HID, INTER}, I[3] = {HID, INTER, HID};
        std::vector<unsigned long long> table(6 * NE);
        for (int p = 0; p < 3; p++) {
            b.rows[p] = R[p]; b.in[p] = I[p];
            const size_t cb = (size_t)NE * R[p] * (I[p] / 2), sb = (size_t)NE * R[p] * (I[p] / 16);
            ck(cudaMalloc(&b.codes[p], cb)); ck(cudaMalloc(&b.scales[p], sb));
            bench_fill<<<1024, 256, 0, s>>>(b.codes[p], cb, 0x9e3779b9u + p, 0);
            bench_fill<<<1024, 256, 0, s>>>(b.scales[p], sb, 0x7f4a7c15u + p, 1);
            for (int e = 0; e < NE; e++) {
                table[(size_t)(2 * p) * NE + e] = (unsigned long long)(b.codes[p] + (size_t)e * R[p] * (I[p] / 2));
                table[(size_t)(2 * p + 1) * NE + e] = (unsigned long long)(b.scales[p] + (size_t)e * R[p] * (I[p] / 16));
            }
        }
        unsigned long long* dtable; ck(cudaMalloc(&dtable, table.size() * 8));
        ck(cudaMemcpy(dtable, table.data(), table.size() * 8, cudaMemcpyHostToDevice));
        std::vector<float> s2(NG * 3);
        for (int i = 0; i < NG * 3; i++) s2[i] = ldexpf(1.0f, (i % 5) - 6);
        float* dscale2; ck(cudaMalloc(&dscale2, s2.size() * 4)); ck(cudaMemcpy(dscale2, s2.data(), s2.size() * 4, cudaMemcpyHostToDevice));
        // Selections: local slots walk a permutation of the 128 local experts, so consecutive
        // launches share none; the rest are the other rank's.
        const int slots = rows * TOPK;
        std::vector<int> sel((size_t)iters * slots);
        int next = 0;
        for (int it = 0; it < iters; it++)
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < TOPK; k++)
                    sel[((size_t)it * rows + r) * TOPK + k] = k < local ? (int)((next++ * 37u) % NE) : NE + (it * 7 + k) % NE;
        int* dsel; ck(cudaMalloc(&dsel, sel.size() * 4)); ck(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
        std::vector<float> selw(slots, 1.0f / 6.0f), x((size_t)rows * HID);
        for (size_t i = 0; i < x.size(); i++) x[i] = (float)((int)((i * 2654435761u) % 2001) - 1000) / 400.0f;
        float *dselw, *dx, *dh, *dc; int* dfault;
        ck(cudaMalloc(&dselw, slots * 4)); ck(cudaMemcpy(dselw, selw.data(), slots * 4, cudaMemcpyHostToDevice));
        ck(cudaMalloc(&dx, x.size() * 4)); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMalloc(&dh, (size_t)slots * INTER * 4)); ck(cudaMalloc(&dc, (size_t)slots * HID * 4));
        ck(cudaMalloc(&dfault, 4)); ck(cudaMemset(dfault, 0, 4));
        ck(cudaStreamSynchronize(s));
        auto gu = [&](int it) {
            api(memra_dsv4_moe_fused_gu_part(dtable, NE, NG, 0, dsel + (size_t)it * slots, dselw, dscale2, dx, nullptr, nullptr, dh, nullptr, TOPK, rows, HID, INTER, 10.0f, dfault, s));
        };
        auto down = [&](int it) {
            api(memra_dsv4_moe_fused_down_part(dtable, NE, NG, 0, dsel + (size_t)it * slots, dscale2, dh, dc, nullptr, nullptr, nullptr, TOPK, rows, INTER, HID, dfault, s));
        };
        cudaEvent_t e0, e1, e2; ck(cudaEventCreate(&e0)); ck(cudaEventCreate(&e1)); ck(cudaEventCreate(&e2));
        for (int it = 0; it < 20; it++) { gu(it % iters); down(it % iters); }
        ck(cudaStreamSynchronize(s));
        float t_gu = 0, t_down = 0;
        for (int it = 0; it < iters; it++) {
            ck(cudaEventRecord(e0, s)); gu(it); ck(cudaEventRecord(e1, s)); down(it); ck(cudaEventRecord(e2, s));
            ck(cudaEventSynchronize(e2));
            float a, c; ck(cudaEventElapsedTime(&a, e0, e1)); ck(cudaEventElapsedTime(&c, e1, e2));
            t_gu += a; t_down += c;
        }
        const double gu_us = 1e3 * t_gu / iters, down_us = 1e3 * t_down / iters;
        // Distinct local experts per launch: every local slot of every row takes a fresh one.
        const double experts = (double)rows * local;
        const double gu_bytes = experts * 2.0 * ((double)INTER * HID / 2 + (double)INTER * HID / 16);
        const double down_bytes = experts * ((double)HID * INTER / 2 + (double)HID * INTER / 16);
        int fault = 0; ck(cudaMemcpy(&fault, dfault, 4, cudaMemcpyDeviceToHost));
        printf("BENCH rows=%d local=%d iters=%d gu_us=%.2f gu_TBps=%.3f down_us=%.2f down_TBps=%.3f pair_us=%.2f fault=%d\n",
               rows, local, iters, gu_us, gu_bytes / gu_us / 1e6, down_us, down_bytes / down_us / 1e6, gu_us + down_us, fault);
        if (argc > 4 && std::string(argv[4]) == "variants") {
            // The packed bank for gate and up at KC 256 and 512.
            std::vector<unsigned long long> pt256(3 * NE, 0), pt512(3 * NE, 0);
            for (int pj : {0, 2}) {
                for (int kc : {256, 512}) {
                    const size_t per = (size_t)(INTER / 8) * (HID / kc) * 8 * (kc / 2 + kc / 16);
                    uint8_t* buf; ck(cudaMalloc(&buf, per * NE));
                    for (int e2 = 0; e2 < NE; e2++) {
                        bench_repack<<<512, 256, 0, s>>>(b.codes[pj] + (size_t)e2 * INTER * (HID / 2),
                                                         b.scales[pj] + (size_t)e2 * INTER * (HID / 16),
                                                         buf + per * e2, INTER, HID, kc);
                        (kc == 256 ? pt256 : pt512)[(size_t)pj * NE + e2] = (unsigned long long)(buf + per * e2);
                    }
                }
            }
            unsigned long long *dpt256, *dpt512;
            ck(cudaMalloc(&dpt256, pt256.size() * 8)); ck(cudaMalloc(&dpt512, pt512.size() * 8));
            ck(cudaMemcpy(dpt256, pt256.data(), pt256.size() * 8, cudaMemcpyHostToDevice));
            ck(cudaMemcpy(dpt512, pt512.data(), pt512.size() * 8, cudaMemcpyHostToDevice));
            ck(cudaStreamSynchronize(s));
            // The packed twin must reproduce the standard kernel's h bit for bit.
            float* dh2; ck(cudaMalloc(&dh2, (size_t)slots * INTER * 4));
            gu(0);
            bench_gu_pk<4, 256, 2>(dpt256, dsel, dselw, dscale2, dx, dh2, rows, dfault, s);
            ck(cudaGetLastError()); ck(cudaStreamSynchronize(s));
            std::vector<float> h1((size_t)slots * INTER), h2(h1.size());
            ck(cudaMemcpy(h1.data(), dh, h1.size() * 4, cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(h2.data(), dh2, h2.size() * 4, cudaMemcpyDeviceToHost));
            printf("PACKED_EQUAL %d\n", (int)(memcmp(h1.data(), h2.data(), h1.size() * 4) == 0));
            auto time_gu = [&](const char* name, auto&& launch) {
                for (int it = 0; it < 20; it++) launch(it % iters);
                ck(cudaGetLastError()); ck(cudaStreamSynchronize(s));
                ck(cudaEventRecord(e0, s));
                for (int it = 0; it < iters; it++) launch(it);
                ck(cudaEventRecord(e1, s)); ck(cudaEventSynchronize(e1));
                float ms; ck(cudaEventElapsedTime(&ms, e0, e1));
                const double us = 1e3 * ms / iters;
                printf("VARIANT %s rows=%d gu_us=%.2f gu_TBps=%.3f\n", name, rows, us, gu_bytes / us / 1e6);
            };
#define STD(W, K, T) time_gu("std_" #W "_" #K "_" #T, [&](int it) { bench_gu_std<W, K, T>(dtable, dsel + (size_t)it * slots, dselw, dscale2, dx, dh, rows, dfault, s); })
#define PK(W, K, T, PT) time_gu("packed_" #W "_" #K "_" #T, [&](int it) { bench_gu_pk<W, K, T>(PT, dsel + (size_t)it * slots, dselw, dscale2, dx, dh2, rows, dfault, s); })
            STD(4, 256, 2); PK(4, 256, 2, dpt256);
            STD(4, 256, 3); PK(4, 256, 3, dpt256);
            STD(4, 256, 4); PK(4, 256, 4, dpt256);
            STD(4, 512, 2); PK(4, 512, 2, dpt512);
            STD(2, 256, 3); PK(2, 256, 3, dpt256);
            STD(4, 256, 2); PK(4, 256, 2, dpt256);
#define MODE(M, X) time_gu("mode" #M "_mirror" #X, [&](int it) { bench_gu_mode<M, X>(dtable, dsel + (size_t)it * slots, dx, dh2, rows, s); })
            MODE(0, 1); MODE(0, 0); MODE(1, 0); MODE(2, 0); MODE(1, 1); MODE(0, 1);
#undef MODE
            auto time_down = [&](const char* name, auto&& launch) {
                for (int it = 0; it < 20; it++) launch(it % iters);
                ck(cudaGetLastError()); ck(cudaStreamSynchronize(s));
                ck(cudaEventRecord(e0, s));
                for (int it = 0; it < iters; it++) launch(it);
                ck(cudaEventRecord(e1, s)); ck(cudaEventSynchronize(e1));
                float ms; ck(cudaEventElapsedTime(&ms, e0, e1));
                const double us = 1e3 * ms / iters;
                printf("VARIANT %s rows=%d down_us=%.2f down_TBps=%.3f\n", name, rows, us, down_bytes / us / 1e6);
            };
#define DSTD(W, K, T) time_down("down_" #W "_" #K "_" #T, [&](int it) { bench_down_std<W, K, T>(dtable, dsel + (size_t)it * slots, dscale2, dh, dc, rows, s); })
            DSTD(4, 256, 2); DSTD(8, 256, 2); DSTD(16, 256, 2); DSTD(8, 256, 3); DSTD(4, 256, 3); DSTD(4, 256, 2);
            DSTD(16, 256, 2); DSTD(16, 128, 3); DSTD(16, 128, 4);
            STD(4, 256, 2); STD(8, 256, 2); STD(8, 128, 3); STD(4, 256, 2); STD(8, 256, 2);
#undef DSTD
#undef STD
#undef PK
        }
        return 0;
    } catch (const std::exception& e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
