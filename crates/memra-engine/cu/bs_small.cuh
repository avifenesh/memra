// The shared small-batch skeleton of the block-scaled GEMMs: the device MoE plan and its tile list, the two
// cp.async rings of the grouped expert GEMM's small-row body, and the dense small-row body. Header-only: each TU
// that runs it includes this file and instantiates the kernels with a math policy and an epilogue. First caller:
// the DeepSeek-V4.1 route (cu/dsv41_gpu.cu, cu/dsv41_cutlass.cu), whose program is the policy `MxHw32`. The CUTLASS
// forms of the same GEMMs are in cu/bs_cutlass.cuh. Decision record: docs/decisions/BS-SMALL-SKELETON.md.
//
// What is shared, what is a policy and what is the caller's:
//   - the skeleton owns the tile list, the staging (which bytes go to shared memory when), the loop over stages,
//     the m16n8k32 fragment addressing, the order of the K blocks inside the chain (ascending, one accumulator an
//     output, never split over K) and each instance's shared-memory size and limits;
//   - the math policy owns the operand formats (bytes a row and K block on each side), the A register made from
//     staged bytes, the scale stream (where a row's scale word is, and which rows a lane reads), the MMA and what
//     happens at the end of a scale word's K blocks (`block_end`) and of the chain (`finish`);
//   - the epilogue policy owns the output type and its conversion;
//   - the caller owns the launch geometry: which ring and column width a grouped call runs on, the dense body's
//     columns, warps and stage depth, and which calls take a small body at all. Those are speed rules measured on
//     the caller's shapes and cards; the skeleton takes them as arguments and never picks them.
// The grouped body reads the fixed expert table record `ExpW` (three projections' codes and scale streams), not a
// policy type. A ring stages a row's scale stream as PK bytes a stage (one 4-byte word a row and four K blocks of
// 32); the policy decides what the word holds. Rows past the live ones are never copied and never read: their
// fragments and scale words are zeros in registers.
//
// The bodies keep the expression forms of the route kernels they came from (the order operands are formed in,
// scalars rather than aggregates, thread index reads in the kernel): nvcc's output depends on them, and the
// extraction was held to the same SASS kernel for kernel (the decision record has the method).
//
// Every function here is inline or a template, so the header can be included by several TUs of one archive. Two
// units that instantiate the same `bs::` kernel (the same template arguments, such as the plan kernels or a shared
// policy) get one symbol for both: they must compile it identically (the same header revision and the same nvcc
// flags, `-fmad` included), or which unit's copy runs is undefined.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <atomic>

namespace bs {

// ---------------------------------------------------------------------------------------------
// shared records

// The device-planned dispatch (`moe_plan_launch`): one entry per tile of an expert with rows.
struct Tile {
    int row0;    // the expert's first pair
    int rows;    // the expert's pairs
    int m0;      // this tile's first row inside the expert
    int expert;  // index into the rank's expert weight table
};

// The expert table record: three projections' weight codes and scale streams, their second-level scales.
struct ExpW {
    const uint8_t* w[3];
    const uint8_t* s[3];
    float scale2[3];
    int pad;
};
static_assert(sizeof(ExpW) == 64, "the expert table record is 64 bytes");

// Rows a tile of the grouped small-row body holds.
constexpr int kMaxTileRows = 8;
// Rows the dense small-row body takes in one call.
constexpr int kDenseMaxRows = 16;

constexpr int log2c(int x) { return x <= 1 ? 0 : 1 + log2c(x / 2); }

// ---------------------------------------------------------------------------------------------
// device helpers

__device__ __forceinline__ uint32_t lds32(const uint8_t* p) { return *(const uint32_t*)p; }

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool pred) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    int n = pred ? 16 : 0;  // 0: the 16 bytes are zero-filled
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp4(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// The scale layout of CUTLASS's sm_1xx block-scaled kernels at 32 inputs a code (`Sm1xxBlockScaledConfig<32>`, the
// same function for both operands): the code of row r, K block kb of a tensor with kbn blocks a row, over rows
// padded to a multiple of 128. The four codes of K blocks kb .. kb + 3 (kb a multiple of 4) are adjacent bytes.
__host__ __device__ __forceinline__ long sf32_off(long r, long kb, long kbn) {
    return (r / 128) * (((kbn + 3) / 4) * 512) + (kb / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + kb % 4;
}

// ---------------------------------------------------------------------------------------------
// host helpers (each TU keeps its own cache)

// The SMs of the current device (cached a device).
static inline int sms() {
    static int n[64] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!n[dev & 63]) cudaDeviceGetAttribute(&n[dev & 63], cudaDevAttrMultiProcessorCount, dev);
    return n[dev & 63] > 0 ? n[dev & 63] : 1;
}

// The current device's compute capability as major * 10 + minor (cached a device).
static inline int cc() {
    static int c[64] = {0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!c[dev & 63]) {
        int ma = 0, mi = 0;
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, dev);
        c[dev & 63] = ma * 10 + mi;
    }
    return c[dev & 63];
}

// The current device's shared memory: the most a block may opt into, an SM's whole, and what each resident block
// reserves (cached a device).
struct Smem {
    int optin, sm, per_block;
};
static inline Smem smem() {
    static Smem v[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    Smem& s = v[dev & 63];
    if (!s.optin) {
        cudaDeviceGetAttribute(&s.sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&s.per_block, cudaDevAttrReservedSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&s.optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (s.optin <= 0) s.optin = 48 * 1024;
        if (s.sm <= 0) s.sm = s.optin;
    }
    return s;
}

// A kernel's dynamic shared-memory limit is an attribute of the function in a context, not of a launch: set once a
// device, to the most the kernel ever asks for, never to a call's own size (two threads issuing on one card with
// different sizes would otherwise launch above the limit the other had just set). `kernel` goes in parentheses when
// it is a template instance.
#define BS_ONCE_A_DEVICE(stmt)                                           \
    do {                                                                 \
        static std::atomic<unsigned long long> done_{0};                 \
        int dev_ = 0;                                                    \
        cudaGetDevice(&dev_);                                            \
        unsigned long long bit_ = 1ull << (dev_ & 63);                   \
        if (!(done_.load(std::memory_order_relaxed) & bit_)) {           \
            stmt;                                                        \
            done_.fetch_or(bit_, std::memory_order_relaxed);             \
        }                                                                \
    } while (0)
#define BS_SMEM_LIMIT(kernel, bytes) \
    BS_ONCE_A_DEVICE(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes))

// ---------------------------------------------------------------------------------------------
// Math policies. A policy runs on the canonical m16n8k32 fragments (mma.sync, A row-major [16][32], B column-major
// [32][8]): lane (g, t) holds A rows g and g + 8 and B column g. It gives the skeleton:
//   kWB, kWT        a weight row's bytes a K block of 32, and the bytes lane t's A register starts at (t * kWT);
//                   A registers 0, 1 come from rows g, g + 8 at that offset, 2, 3 from kWB / 2 further
//   kAB             an activation row's bytes a K block (the B registers at 4t and 4t + kAB / 2: e4m3)
//   wreg(p)         the 32-bit A register from the staged bytes at p (e4m3: the word; e2m1: two bytes widened)
//   sf_off          the byte offset of a row's scale word of four K blocks in a scale stream
//   wrow(), arow()  which rows' scale words a lane reads (kWRows weight rows, kARows activation rows of an n8 tile);
//                   plain ints, not a struct: an aggregate copy reaches the optimizer later and moves the SASS
//   kAcc, zero()    the accumulator a lane keeps
//   mma(), block_end(), finish()
//                   the four MMAs of a scale word, what follows a scale word's four K blocks, and output element e
// A ring stages a row's scale stream as one 32-bit word a row and four K blocks; the skeleton loads the words the
// lane's rows name and hands them to the policy raw (codes, or a float's bits).

// `MxHw32<WF>`: hardware block-scaled MMA, the DeepSeek-V4.1 program. c += (A * sa) . (B * sb) over each 32-wide K
// block on the tensor core (kind::mxf8f6f4 block_scale scale_vec::1X, ue8m0 codes, one per row and 32 inputs on
// both sides), A the weights (WF 0: e4m3; 1: e2m1 pairs, low nibble first, put in their 8-bit container at bits
// 5..2 as CUTLASS shifts them), B the activations (e4m3), one fp32 accumulator chained over the K blocks in
// ascending order. The scales are applied inside the MMA, so a scale word's end and the chain's end do nothing.
// Scale streams are in `sf32_off`'s layout. Scale selectors (PTX ISA "Block Scaling for mma.sync"): byte u of lane
// t = 0 of quad g holds weight row g's code of K block u of the word and of lane t = 1 row g + 8's; byte u of lane
// t = 0 holds activation column g's. So a lane reads one weight row's word (g + 8 * (t & 1)) and one activation
// row's (g).
template <int WF>
struct MxHw32 {
    static_assert(WF == 0 || WF == 1, "MxHw32: weights e4m3 (0) or e2m1 (1)");
    static constexpr int kWB = WF == 0 ? 32 : 16;  // a weight row's bytes a K block
    static constexpr int kWT = WF == 0 ? 4 : 2;    // lane t's A register at byte t * kWT of a K block's half
    static constexpr int kAB = 32;                 // an activation row's bytes a K block (e4m3)
    static constexpr int kLWB = log2c(kWB / 16);   // 16-byte chunks of a weight row's K block, as a shift
    static constexpr int kLAB = log2c(kAB / 16);
    static constexpr int kAcc = 4;
    static constexpr int kWRows = 1, kARows = 1;  // the skeleton reads one or two of each

    __device__ __forceinline__ static long sf_off(long r, long kb, long kbn) { return sf32_off(r, kb, kbn); }
    // The weight row whose scale word lane (g, t) reads (q < kWRows): t = 0 row g's, t = 1 row g + 8's.
    __device__ __forceinline__ static int wrow(int g, int t, int) { return g + 8 * (t & 1); }
    // The activation row inside an n8 tile whose word it reads (q < kARows): t = 0 column g's.
    __device__ __forceinline__ static int arow(int g, int, int) { return g; }
    __device__ __forceinline__ static void zero(float a[kAcc]) { a[0] = a[1] = a[2] = a[3] = 0.f; }
    __device__ __forceinline__ static uint32_t wreg(const uint8_t* p) {
        if (WF == 0) return lds32(p);
        return e2m1_container4(*(const uint16_t*)p);
    }
    // K block u (0..3) of the scale words; u is a constant after unrolling.
    __device__ __forceinline__ static void mma(float c[kAcc], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                               const uint32_t ws[kWRows], const uint32_t as[kARows], int u) {
        switch (u) {
            case 0: mma_u<0, 0>(c, a, b0, b1, ws[0], as[0]); break;
            case 1: mma_u<1, 1>(c, a, b0, b1, ws[0], as[0]); break;
            case 2: mma_u<2, 2>(c, a, b0, b1, ws[0], as[0]); break;
            default: mma_u<3, 3>(c, a, b0, b1, ws[0], as[0]); break;
        }
    }
    // The end of a scale word's four K blocks: the scales were applied inside the MMA.
    __device__ __forceinline__ static void block_end(float*, const uint32_t*, const uint32_t*) {}
    // Output element e of the lane (weight row g + 8 * (e >= 2), activation row 2t + (e & 1)).
    __device__ __forceinline__ static float finish(const float c[kAcc], int e) { return c[e]; }

    // e2m1 nibbles (four, element 0 lowest) -> four e2m1 bytes, each nibble at bits 5..2
    __device__ __forceinline__ static uint32_t e2m1_container4(uint32_t nib16) {
        return ((nib16 & 0xFu) << 2) | (((nib16 >> 4) & 0xFu) << 10) | (((nib16 >> 8) & 0xFu) << 18) |
               (((nib16 >> 12) & 0xFu) << 26);
    }
    template <int BA, int BB>
    __device__ __forceinline__ static void mma_u(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                                 uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
        if (WF == 0) {
            asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0),
                  "n"(BB), "n"(0));
        } else {
            asm("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {%12, %13}, %11, {%14, %15};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb), "n"(BA), "n"(0),
                  "n"(BB), "n"(0));
        }
#else
        // the block-scaled MMA exists on the 120a family only; callers route here only where it is built
        __trap();
#endif
    }
};

// ---------------------------------------------------------------------------------------------
// The epilogue policy `EpiBf16`: bf16 out, round to nearest even. `conv` makes an output element; the skeleton
// stores it (the value is formed before its address, as in a plain assignment).
struct EpiBf16 {
    typedef uint16_t Out;  // raw bf16 bits
    __device__ __forceinline__ static Out conv(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
};

// ---------------------------------------------------------------------------------------------
// The grouped expert GEMM's small-row body over the plan's tile list (every tile at most kMaxTileRows rows): block
// y runs tile y (none past *ntiles), block z projection z (p0 into d0, p1 into d1), block x the ring's warps of CW
// output columns each. A warp: weight rows rw0 .. rw0 + CW - 1 of its expert's projection against the tile's rows
// (pairs t0 .. t0 + M - 1 of A, scale rows sfrow[pair], or the pair's own row of a layout of the pairs when sfrow
// is null). Two rings, the same chain and the same bytes: which one a call runs is speed only. A ring is a tag with
// its geometry; both ring bodies are in `bs_small_moe_kernel` itself (one compiled a template instance), not in a
// called function: the compiler bounds a thread index read in the kernel by its launch bounds and one read in a
// called function by 1024 only, and the definition order of the index arithmetic decides its form, so the bodies
// keep the reads where the route's kernels had them (#1337 head: the same SASS, kernel for kernel).
enum { kRingWarp6 = 1, kRingBlock2 = 2 };

// Ring `Warp6`: a warp's own ring of six stages of PK = 4 K blocks, each [CW weight rows][WR] + [RA activation
// rows][AR] + the weight rows' scale words [CW][4] + the activation rows' [RA][4]. RA: the tile rows the launch
// holds, so a one-row call keeps one row and the ring goes deep: the body is bound by the bytes it has in flight,
// not by its MMAs. One or two blocks fit an SM.
struct Warp6 {
    static constexpr int kKind = kRingWarp6;
    static constexpr int kThreads = 128;
    static constexpr int NW = 4;  // warps a block
    static constexpr int PS = 6;  // stages
    template <class Math, int CW, int PK>
    struct Geo {
        static_assert(PK == 4, "Warp6 stages one scale word (four K blocks) a row");
        static constexpr int WR = PK * Math::kWB + 16;  // +16: conflict-free fragment loads
        static constexpr int AR = PK * Math::kAB + 16;
        static constexpr int W_B = CW * WR;
        static constexpr int WS_B = (CW * PK + 15) & ~15;
        __host__ __device__ static constexpr int stage(int ra) {
            return W_B + ra * AR + WS_B + ((ra * PK + 15) & ~15);
        }
        __host__ __device__ static constexpr int warp(int ra) { return PS * stage(ra); }
        __host__ __device__ static constexpr int block(int ra) { return NW * warp(ra); }
    };
};

// Ring `Block2`: the same tile list and output columns on a block-staged two-stage ring: a stage is PK K blocks (2
// KB of e2m1 weights a warp at CW x PK = 128), the block's four warps stage their own weight rows and together the
// tile's activation rows, which every warp reads (one copy a block, not a warp), and a barrier a stage hands the
// stage over. Shared memory is the warps' weights and one copy of the activations, so several blocks fit an SM
// where Warp6 holds one or two. The last stage of a K that is not a multiple of PK is shorter (a multiple of four K
// blocks). Which calls take it is the caller's measured rule (the V4.1 route: the RTX PRO 6000).
struct Block2 {
    static constexpr int kKind = kRingBlock2;
    static constexpr int kThreads = 128;
    static constexpr int NW = 4;
    template <class Math, int CW, int PK>
    struct Geo {
        static_assert(PK % 4 == 0, "Block2 stages whole scale words (four K blocks)");
        static constexpr int PW = PK / 4;  // scale words a row a stage
        static constexpr int WR = PK * Math::kWB + 16;
        static constexpr int AR = PK * Math::kAB + 16;
        static constexpr int W_B = CW * WR;
        static constexpr int WST = W_B + ((CW * PK + 15) & ~15);  // a warp's weight rows and their scale words
        __host__ __device__ static constexpr int stage(int ra) { return NW * WST + ra * AR + ((ra * PK + 15) & ~15); }
        __host__ __device__ static constexpr int block(int ra) { return 2 * stage(ra); }
    };
};

template <class Ring, class Math, class Epi, int CW, int PK>
__global__ void __launch_bounds__(Ring::kThreads)
    bs_small_moe_kernel(const uint8_t* __restrict__ A, long lda, const uint8_t* __restrict__ asf,
                        const int* __restrict__ sfrow, const Tile* __restrict__ tiles, const int* __restrict__ ntiles,
                        const ExpW* __restrict__ ew, int p0, int p1, long ldb, typename Epi::Out* __restrict__ d0,
                        typename Epi::Out* __restrict__ d1, long ldc, int N, int K, int BM) {
    static_assert(Math::kWRows >= 1 && Math::kWRows <= 2 && Math::kARows >= 1 && Math::kARows <= 2,
                  "a lane reads one or two weight and activation scale rows");
    using P = typename Ring::template Geo<Math, CW, PK>;
    extern __shared__ __align__(16) uint8_t bs_moe_sm[];
    if constexpr (Ring::kKind == kRingWarp6) {
        constexpr int NW = Warp6::NW, PS = Warp6::PS;
        // 16-byte chunks of a row's stage and their shifts
        constexpr int WQ = PK * (Math::kWB / 16), AQ = PK * (Math::kAB / 16), LWQ = log2c(WQ), LAQ = log2c(AQ);
        // the tile is loaded with the count, not after it: one memory round trip (an entry past the count is inside
        // the list's allocation and is not used)
        const Tile tl = tiles[blockIdx.y];
        if ((int)blockIdx.y >= *ntiles) return;
        const int proj = blockIdx.z ? p1 : p0;
        typename Epi::Out* C = blockIdx.z ? d1 : d0;
        const ExpW& w = ew[tl.expert];
        const uint8_t* __restrict__ W = w.w[proj];
        const uint8_t* __restrict__ WS = w.s[proj];
        const long t0 = (long)tl.row0 + tl.m0;
        const int M = min(tl.rows - tl.m0, BM);
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
        const int rw0 = (blockIdx.x * NW + warp) * CW;
        const int nw = min(CW, N - rw0);  // live weight rows (the rest stay zero)
        if (nw <= 0) return;
        const int KB = K / 32, NS = KB / PK;
        const int STAGE = P::stage(BM), WS_O = P::W_B + BM * P::AR, AS_O = WS_O + P::WS_B;
        uint8_t* ring = bs_moe_sm + warp * P::warp(BM);
        // lane r < M: activation row r's scale row (the pair itself when sfrow is null: a layout of the pairs)
        const int srow = lane < M ? (sfrow ? sfrow[t0 + lane] : (int)(t0 + lane)) : 0;
        float acc[Math::kAcc];
        Math::zero(acc);
        // rows that are never copied (weight rows past N, activation rows past M) are never read: their fragments
        // and scale words are zeros in registers
        auto issue = [&](int i) {
            uint8_t* st = ring + (i % PS) * STAGE;
            int kb0 = i * PK;
            for (int c = lane; c < nw * WQ; c += 32) {
                int r = c >> LWQ, q = c & (WQ - 1);
                cp16(st + r * P::WR + q * 16, W + (long)(rw0 + r) * ldb + kb0 * Math::kWB + q * 16, true);
            }
            for (int c = lane; c < M * AQ; c += 32) {
                int r = c >> LAQ, q = c & (AQ - 1);
                cp16(st + P::W_B + r * P::AR + q * 16, A + (t0 + r) * lda + kb0 * Math::kAB + q * 16, true);
            }
            if (lane < nw) cp4(st + WS_O + lane * 4, WS + Math::sf_off(rw0 + lane, kb0, KB));
            if (lane < M) cp4(st + AS_O + lane * 4, asf + Math::sf_off(srow, kb0, KB));
        };
        for (int s = 0; s < PS - 1; ++s) {
            if (s < NS) issue(s);
            cp_commit();
        }
        const bool w0 = g < nw, w1 = g + 8 < nw;
        const int wr0 = Math::wrow(g, t, 0), ar0 = Math::arow(g, t, 0);  // the lane's scale rows
        const bool wok0 = wr0 < nw;
        for (int i = 0; i < NS; ++i) {
            cp_wait<PS - 2>();
            __syncwarp();  // stage i landed for every lane; stage i - 1's readers are done
            if (i + PS - 1 < NS) issue(i + PS - 1);
            cp_commit();
            const uint8_t* st = ring + (i % PS) * STAGE;
            const uint8_t* sa = st + P::W_B;
            uint32_t wsc[Math::kWRows], asc[Math::kARows];
            wsc[0] = wok0 ? *(const uint32_t*)(st + WS_O + wr0 * 4) : 0u;
            if constexpr (Math::kWRows > 1)
                wsc[1] = Math::wrow(g, t, 1) < nw ? *(const uint32_t*)(st + WS_O + Math::wrow(g, t, 1) * 4) : 0u;
            asc[0] = ar0 < M ? *(const uint32_t*)(st + AS_O + ar0 * 4) : 0u;
            if constexpr (Math::kARows > 1)
                asc[1] = Math::arow(g, t, 1) < M ? *(const uint32_t*)(st + AS_O + Math::arow(g, t, 1) * 4) : 0u;
#pragma unroll
            for (int kk = 0; kk < PK; ++kk) {
                uint32_t af[4] = {0u, 0u, 0u, 0u};
                if (w0) {
                    af[0] = Math::wreg(st + g * P::WR + kk * Math::kWB + Math::kWT * t);
                    af[2] = Math::wreg(st + g * P::WR + kk * Math::kWB + Math::kWB / 2 + Math::kWT * t);
                }
                if (w1) {
                    af[1] = Math::wreg(st + (g + 8) * P::WR + kk * Math::kWB + Math::kWT * t);
                    af[3] = Math::wreg(st + (g + 8) * P::WR + kk * Math::kWB + Math::kWB / 2 + Math::kWT * t);
                }
                uint32_t b0 = 0u, b1 = 0u;
                if (g < M) {
                    const uint8_t* ar = sa + g * P::AR + kk * Math::kAB + 4 * t;
                    b0 = lds32(ar);
                    b1 = lds32(ar + Math::kAB / 2);
                }
                Math::mma(acc, af, b0, b1, wsc, asc, kk);
            }
            Math::block_end(acc, wsc, asc);
        }
        cp_wait<0>();
        // lane (g, t): weight row g against activation rows 2t, 2t + 1 (elements 0, 1) and weight row g + 8 (2, 3)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int wr = g + (e >= 2 ? 8 : 0), row = 2 * t + (e & 1), col = rw0 + wr;
            if (wr < CW && row < M && col < N) C[(t0 + row) * ldc + col] = Epi::conv(Math::finish(acc, e));
        }
    } else {
        constexpr int NW = Block2::NW;
        constexpr int PW = P::PW;
        constexpr int WQ = PK * (Math::kWB / 16), AQ = (Math::kAB / 16) * PK;  // 16-byte chunks a row a stage
        const Tile tl = tiles[blockIdx.y];
        if ((int)blockIdx.y >= *ntiles) return;
        const int proj = blockIdx.z ? p1 : p0;
        typename Epi::Out* C = blockIdx.z ? d1 : d0;
        const ExpW& w = ew[tl.expert];
        const uint8_t* __restrict__ W = w.w[proj];
        const uint8_t* __restrict__ WS = w.s[proj];
        const long t0 = (long)tl.row0 + tl.m0;
        const int M = min(tl.rows - tl.m0, BM);
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
        const int rw0 = (blockIdx.x * NW + warp) * CW;
        // live weight rows; a warp with none still copies activations and meets every barrier
        const int nw = max(0, min(CW, N - rw0));
        const int KB = K / 32, NS = (KB + PK - 1) / PK;
        const int STAGE = P::stage(BM);
        auto wst = [&](int i) { return bs_moe_sm + (i & 1) * STAGE + warp * P::WST; };
        auto ast = [&](int i) { return bs_moe_sm + (i & 1) * STAGE + NW * P::WST; };
        auto issue = [&](int i) {
            uint8_t* sw = wst(i);
            uint8_t* sa = ast(i);
            const int kb0 = i * PK, pk = min(PK, KB - kb0), pw = pk / 4;
            for (int c = lane; c < nw * WQ; c += 32) {
                int r = c / WQ, q = c % WQ;
                if (q < pk * (Math::kWB / 16))
                    cp16(sw + r * P::WR + q * 16, W + (long)(rw0 + r) * ldb + kb0 * Math::kWB + q * 16, true);
            }
            for (int c = lane; c < nw * PW; c += 32) {
                int r = c / PW, j = c % PW;
                if (j < pw) cp4(sw + P::W_B + c * 4, WS + Math::sf_off(rw0 + r, kb0 + 4 * j, KB));
            }
            for (int c = threadIdx.x; c < M * (Math::kAB / 16) * PK; c += NW * 32) {
                int r = c / AQ, q = c % AQ;
                if (q < (Math::kAB / 16) * pk)
                    cp16(sa + r * P::AR + q * 16, A + (t0 + r) * lda + kb0 * Math::kAB + q * 16, true);
            }
            for (int c = threadIdx.x; c < M * PW; c += NW * 32) {
                int r = c / PW, j = c % PW;
                int sr = sfrow ? sfrow[t0 + r] : (int)(t0 + r);
                if (j < pw) cp4(sa + BM * P::AR + c * 4, asf + Math::sf_off(sr, kb0 + 4 * j, KB));
            }
        };
        issue(0);
        cp_commit();
        const bool w0 = g < nw, w1 = g + 8 < nw;
        const int wr0 = Math::wrow(g, t, 0), ar0 = Math::arow(g, t, 0);  // the lane's scale rows
        const bool wok0 = wr0 < nw;
        float acc[Math::kAcc];
        Math::zero(acc);
        for (int i = 0; i < NS; ++i) {
            cp_wait<0>();
            __syncthreads();  // stage i landed for every thread; every warp is done with stage i - 1's buffer
            if (i + 1 < NS) issue(i + 1);
            cp_commit();
            if (nw == 0) continue;
            const uint8_t* st = wst(i);
            const uint8_t* sa = ast(i);
            const int pw = min(PK, KB - i * PK) / 4;
#pragma unroll
            for (int j = 0; j < PW; ++j) {
                if (j < pw) {
                    uint32_t wsc[Math::kWRows], asc[Math::kARows];
                    wsc[0] = wok0 ? *(const uint32_t*)(st + P::W_B + (wr0 * PW + j) * 4) : 0u;
                    if constexpr (Math::kWRows > 1)
                        wsc[1] = Math::wrow(g, t, 1) < nw
                                     ? *(const uint32_t*)(st + P::W_B + (Math::wrow(g, t, 1) * PW + j) * 4)
                                     : 0u;
                    asc[0] = ar0 < M ? *(const uint32_t*)(sa + BM * P::AR + (ar0 * PW + j) * 4) : 0u;
                    if constexpr (Math::kARows > 1)
                        asc[1] = Math::arow(g, t, 1) < M
                                     ? *(const uint32_t*)(sa + BM * P::AR + (Math::arow(g, t, 1) * PW + j) * 4)
                                     : 0u;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int kk = 4 * j + u;
                        uint32_t af[4] = {0u, 0u, 0u, 0u};
                        if (w0) {
                            af[0] = Math::wreg(st + g * P::WR + kk * Math::kWB + Math::kWT * t);
                            af[2] = Math::wreg(st + g * P::WR + kk * Math::kWB + Math::kWB / 2 + Math::kWT * t);
                        }
                        if (w1) {
                            af[1] = Math::wreg(st + (g + 8) * P::WR + kk * Math::kWB + Math::kWT * t);
                            af[3] = Math::wreg(st + (g + 8) * P::WR + kk * Math::kWB + Math::kWB / 2 + Math::kWT * t);
                        }
                        uint32_t b0 = 0u, b1 = 0u;
                        if (g < M) {
                            const uint8_t* ar = sa + g * P::AR + kk * Math::kAB + 4 * t;
                            b0 = lds32(ar);
                            b1 = lds32(ar + Math::kAB / 2);
                        }
                        Math::mma(acc, af, b0, b1, wsc, asc, u);
                    }
                    Math::block_end(acc, wsc, asc);
                }
            }
        }
        cp_wait<0>();
        if (nw == 0) return;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int wr = g + (e >= 2 ? 8 : 0), row = 2 * t + (e & 1), col = rw0 + wr;
            if (wr < CW && row < M && col < N) C[(t0 + row) * ldc + col] = Epi::conv(Math::finish(acc, e));
        }
    }
}

// One instantiation of the grouped body: its shared memory and its launch. The caller picks the instantiation (ring,
// CW, PK) by its own measured rules; nothing here chooses geometry for it. A Warp6 instance's dynamic shared memory
// limit is set once a device to its size at the most tile rows; a Block2 instance stays under the default 48 KB at
// the most tile rows (checked here for every instance, since a policy's operand bytes set the stage size).
template <class Ring, class Math, class Epi, int CW, int PK>
struct SmallRowMoe {
    using G = typename Ring::template Geo<Math, CW, PK>;
    static_assert(Ring::kKind != kRingBlock2 || G::block(kMaxTileRows) <= 48 * 1024,
                  "a Block2 instance over the default 48 KB of dynamic shared memory at the most tile rows");
    static constexpr int kCols = Ring::NW * CW;  // output columns a block
    static int smem(int bm) { return G::block(bm); }
    // max_tiles > 0, 1 <= bm <= kMaxTileRows (contract checked by the caller)
    static void launch(int n, int max_tiles, int nproj, int bm, cudaStream_t st, const uint8_t* a, long lda,
                       const uint8_t* asf, const int* sfrow, const Tile* tiles, const int* ntiles, const ExpW* ew,
                       int p0, int p1, long ldb, typename Epi::Out* d0, typename Epi::Out* d1, long ldc, int k) {
        if constexpr (Ring::kKind == kRingWarp6)
            BS_SMEM_LIMIT((bs_small_moe_kernel<Ring, Math, Epi, CW, PK>), (G::block(kMaxTileRows)));
        dim3 gr((n + kCols - 1) / kCols, max_tiles, nproj);
        bs_small_moe_kernel<Ring, Math, Epi, CW, PK><<<gr, Ring::kThreads, G::block(bm), st>>>(
            a, lda, asf, sfrow, tiles, ntiles, ew, p0, p1, ldb, d0, d1, ldc, n, k, bm);
    }
};

// ---------------------------------------------------------------------------------------------
// The dense small-row body (calls of at most kDenseMaxRows rows): block x holds blockDim / 32 warps of CW output
// columns each, over the call's M <= 8 MR rows and all of K, PK = 1 << LPK K blocks a stage (a power of two from
// 4). A decode call of a few rows is bound by its weight bytes and by the chain over K, not by its MMAs, so the call
// spreads over output columns only (never over K, which would cut the chain): each warp owns CW weight rows and all
// of K and streams them through its own ring of kDenseStages stages. The block stages the call's activation rows
// once a stage for all its warps, in two buffers fed one stage ahead through registers (the activations come from
// L2, written by the call's quantizer, so one stage of lead covers them and the weights' ring keeps the shared
// memory). +16 on a row: conflict-free fragment loads.
constexpr int kDenseStages = 4;
constexpr int kDenseAReg = 8;  // 16-byte activation chunks a thread holds for the next stage
constexpr int kDenseSReg = 2;  // activation scale words a thread holds

// A block's shared memory: two buffers of the call's activation rows for one stage ([M][AR] codes + their scale
// words [M][PK / 4]), then each warp's ring of kDenseStages stages of its weight rows ([CW][WR] + [CW][PK / 4]).
// wb, ab: a weight and an activation row's bytes a K block (the policy's kWB, kAB).
struct DenseGeo {
    int wr, ar, a_st, w_st;
    __host__ __device__ DenseGeo(int wb, int ab, int cw, int m, int pk) {
        wr = pk * wb + 16;
        ar = pk * ab + 16;
        a_st = m * ar + ((m * (pk / 4) * 4 + 15) & ~15);
        w_st = cw * wr + ((cw * (pk / 4) * 4 + 15) & ~15);
    }
    __host__ __device__ int block(int nwb) const { return 2 * a_st + nwb * kDenseStages * w_st; }
};

template <class Math, class Epi, int CW, int MR>
__global__ void __launch_bounds__(128) bs_small_dense_kernel(const uint8_t* __restrict__ A, long lda,
                                                             const uint8_t* __restrict__ asf,
                                                             const uint8_t* __restrict__ W, long ldw,
                                                             const uint8_t* __restrict__ wsf,
                                                             typename Epi::Out* __restrict__ C, long ldc, int M, int N,
                                                             int K, int LPK) {
    static_assert(Math::kWRows >= 1 && Math::kWRows <= 2 && Math::kARows >= 1 && Math::kARows <= 2,
                  "a lane reads one or two weight and activation scale rows");
    constexpr int WB = Math::kWB;  // a weight row's bytes a K block
    extern __shared__ __align__(16) uint8_t bs_dense_sm[];
    const int nthr = blockDim.x, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int rw0 = (blockIdx.x * (nthr >> 5) + warp) * CW;
    const int nw = max(0, min(CW, N - rw0));  // live weight rows (the rest stay zero; a warp with none only copies)
    const int PK = 1 << LPK, KB = K / 32, NS = (KB + PK - 1) / PK;
    const DenseGeo G(Math::kWB, Math::kAB, CW, M, PK);
    uint8_t* abuf = bs_dense_sm;
    uint8_t* wring = bs_dense_sm + 2 * G.a_st + warp * kDenseStages * G.w_st;
    const int AS_O = M * G.ar, WS_O = CW * G.wr;
    // 16-byte chunks of a row's stage (powers of two): weights, activations; scale words a row
    const int lw = LPK + Math::kLWB, la = LPK + Math::kLAB, lp = LPK - 2;
    // rows that are never copied (weight rows past N, activation rows past M) are never read: their fragments and
    // scale words are zeros in registers; a short last stage copies and reads only its own K blocks
    auto issue_w = [&](int i) {
        const int kb0 = i * PK, pk = min(PK, KB - kb0);
        const int wl = pk * WB / 16, pl = pk / 4;
        uint8_t* ws = wring + (i % kDenseStages) * G.w_st;
        for (int c = lane; c < (nw << lw); c += 32) {
            int r = c >> lw, q = c & ((1 << lw) - 1);
            if (q < wl) cp16(ws + r * G.wr + q * 16, W + (long)(rw0 + r) * ldw + (long)kb0 * WB + q * 16, true);
        }
        for (int c = lane; c < (nw << lp); c += 32) {
            int r = c >> lp, u = c & ((1 << lp) - 1);
            if (u < pl) cp4(ws + WS_O + c * 4, wsf + Math::sf_off(rw0 + r, kb0 + 4 * u, KB));
        }
    };
    // the activations of stage i into registers (each thread a fixed set of chunks and words), and from there into
    // buffer i & 1
    uint4 areg[kDenseAReg];
    uint32_t sreg[kDenseSReg];
    auto load_a = [&](int i) {
        const int kb0 = i * PK, pk = min(PK, KB - kb0), al = pk * (Math::kAB / 16), pl = pk / 4;
#pragma unroll
        for (int j = 0; j < kDenseAReg; ++j) {
            int c = j * nthr + tid, r = c >> la, q = c & ((1 << la) - 1);
            if (c < (M << la) && q < al)
                areg[j] = __ldg((const uint4*)(A + (long)r * lda + (long)kb0 * Math::kAB + q * 16));
        }
#pragma unroll
        for (int j = 0; j < kDenseSReg; ++j) {
            int c = j * nthr + tid, r = c >> lp, u = c & ((1 << lp) - 1);
            if (c < (M << lp) && u < pl)
                sreg[j] = __ldg((const unsigned int*)(asf + Math::sf_off(r, kb0 + 4 * u, KB)));
        }
    };
    auto store_a = [&](int i) {
        const int pk = min(PK, KB - i * PK), al = pk * (Math::kAB / 16), pl = pk / 4;
        uint8_t* as = abuf + (i & 1) * G.a_st;
#pragma unroll
        for (int j = 0; j < kDenseAReg; ++j) {
            int c = j * nthr + tid, r = c >> la, q = c & ((1 << la) - 1);
            if (c < (M << la) && q < al) *(uint4*)(as + r * G.ar + q * 16) = areg[j];
        }
#pragma unroll
        for (int j = 0; j < kDenseSReg; ++j) {
            int c = j * nthr + tid, u = c & ((1 << lp) - 1);
            if (c < (M << lp) && u < pl) *(uint32_t*)(as + AS_O + c * 4) = sreg[j];
        }
    };
    for (int s = 0; s < kDenseStages - 1; ++s) {
        if (s < NS) issue_w(s);
        cp_commit();
    }
    load_a(0);
    store_a(0);
    if (NS > 1) load_a(1);
    const bool w0 = g < nw, w1 = g + 8 < nw;
    const int wr0 = Math::wrow(g, t, 0), ar0 = Math::arow(g, t, 0);  // the lane's scale rows
    const bool wok0 = wr0 < nw;
    float acc[MR][Math::kAcc];
#pragma unroll
    for (int mr = 0; mr < MR; ++mr) Math::zero(acc[mr]);
    for (int i = 0; i < NS; ++i) {
        cp_wait<kDenseStages - 2>();
        // weight stage i landed for every thread and activation stage i is in its buffer; every warp is done with
        // stage i - 1 (its weight slot and the other activation buffer)
        __syncthreads();
        if (i + kDenseStages - 1 < NS) issue_w(i + kDenseStages - 1);
        cp_commit();
        if (i + 1 < NS) store_a(i + 1);
        if (i + 2 < NS) load_a(i + 2);
        if (nw == 0) continue;
        const uint8_t* sa = abuf + (i & 1) * G.a_st;
        const uint8_t* sw = wring + (i % kDenseStages) * G.w_st;
        const int pk = min(PK, KB - i * PK);
        for (int j = 0; j < pk / 4; ++j) {
            uint32_t wsc[Math::kWRows];
            wsc[0] = wok0 ? *(const uint32_t*)(sw + WS_O + ((wr0 << lp) + j) * 4) : 0u;
            if constexpr (Math::kWRows > 1)
                wsc[1] = Math::wrow(g, t, 1) < nw
                             ? *(const uint32_t*)(sw + WS_O + ((Math::wrow(g, t, 1) << lp) + j) * 4)
                             : 0u;
            uint32_t asc[MR][Math::kARows];
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {  // the n8 tile of activation rows mr * 8 .. mr * 8 + 7
                asc[mr][0] = mr * 8 + ar0 < M
                                 ? *(const uint32_t*)(sa + AS_O + (((mr * 8 + ar0) << lp) + j) * 4)
                                 : 0u;
                if constexpr (Math::kARows > 1)
                    asc[mr][1] = mr * 8 + Math::arow(g, t, 1) < M
                                     ? *(const uint32_t*)(sa + AS_O + (((mr * 8 + Math::arow(g, t, 1)) << lp) + j) * 4)
                                     : 0u;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int kk = j * 4 + u;
                uint32_t af[4] = {0u, 0u, 0u, 0u};
                if (w0) {
                    af[0] = Math::wreg(sw + g * G.wr + kk * WB + Math::kWT * t);
                    af[2] = Math::wreg(sw + g * G.wr + kk * WB + WB / 2 + Math::kWT * t);
                }
                if (w1) {
                    af[1] = Math::wreg(sw + (g + 8) * G.wr + kk * WB + Math::kWT * t);
                    af[3] = Math::wreg(sw + (g + 8) * G.wr + kk * WB + WB / 2 + Math::kWT * t);
                }
#pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    uint32_t b0 = 0u, b1 = 0u;
                    if (mr * 8 + g < M) {
                        const uint8_t* ar = sa + (mr * 8 + g) * G.ar + kk * Math::kAB + 4 * t;
                        b0 = lds32(ar);
                        b1 = lds32(ar + Math::kAB / 2);
                    }
                    Math::mma(acc[mr], af, b0, b1, wsc, asc[mr], u);
                }
            }
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) Math::block_end(acc[mr], wsc, asc[mr]);
        }
    }
    cp_wait<0>();
    if (nw == 0) return;
    // lane (g, t): weight row g against activation rows mr * 8 + 2t, + 1 (elements 0, 1) and row g + 8 (2, 3)
#pragma unroll
    for (int mr = 0; mr < MR; ++mr)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            int wr = g + (e >= 2 ? 8 : 0), row = mr * 8 + 2 * t + (e & 1);
            if (wr < nw && row < M) C[(long)row * ldc + rw0 + wr] = Epi::conv(Math::finish(acc[mr], e));
        }
}

// Whether a stage of 1 << lpk K blocks of m activation rows fits the registers that carry it to the next stage
// (each of nthr threads holds kDenseAReg 16-byte chunks and kDenseSReg scale words): a fact of the body, which any
// caller's geometry must respect.
template <class Math>
static inline bool dense_carries(int m, int nthr, int lpk) {
    constexpr int la = log2c(Math::kAB / 16);
    return (m << (lpk + la)) <= kDenseAReg * nthr && (m << (lpk - 2)) <= kDenseSReg * nthr;
}

// A dense call's launch geometry, chosen by the caller by its own measured rules: CW output columns a warp (2, 4, 8
// or 16), nwb warps a block, PK = 1 << lpk K blocks a stage, grid blocks, smem bytes a block (`DenseGeo::block`),
// and the opt-in limit the kernel's attribute is set to once a device.
struct DenseLaunch {
    int cw, nwb, lpk;
    unsigned grid;
    int smem, optin;
};

// One instantiation of the dense body and its launch.
template <class Math, class Epi, int CW, int MR>
struct SmallRowDense {
    static void launch(const DenseLaunch& g, cudaStream_t st, const uint8_t* A, long lda, const uint8_t* As,
                       const uint8_t* B, long ldw, const uint8_t* Bs, typename Epi::Out* c, long ldc, int m, int n,
                       int k) {
        BS_SMEM_LIMIT((bs_small_dense_kernel<Math, Epi, CW, MR>), g.optin);
        bs_small_dense_kernel<Math, Epi, CW, MR><<<g.grid, g.nwb * 32, g.smem, st>>>(A, lda, As, B, ldw, Bs, c, ldc, m,
                                                                                    n, k, g.lpk);
    }
};

// One dense small-row call on the caller's geometry (contract checked by the caller: 1 <= m <= kDenseMaxRows, k a
// multiple of 128, aligned bases, g within the body's limits): the instance for g.cw and the call's rows (MR 1 up
// to 8 rows, 2 up to 16). Returns 0 launched, -1 a cw the body has no instance for (nothing launched).
template <class Math, class Epi>
static inline int dense_small_launch(const DenseLaunch& g, cudaStream_t st, const uint8_t* A, long lda,
                                     const uint8_t* As, const uint8_t* B, long ldw, const uint8_t* Bs,
                                     typename Epi::Out* c, long ldc, int m, int n, int k) {
#define BS_DS(CWX)                                                                                             \
    do {                                                                                                       \
        if (m <= 8) SmallRowDense<Math, Epi, CWX, 1>::launch(g, st, A, lda, As, B, ldw, Bs, c, ldc, m, n, k);  \
        else SmallRowDense<Math, Epi, CWX, 2>::launch(g, st, A, lda, As, B, ldw, Bs, c, ldc, m, n, k);         \
    } while (0)
    if (g.cw == 16) BS_DS(16);
    else if (g.cw == 8) BS_DS(8);
    else if (g.cw == 4) BS_DS(4);
    else if (g.cw == 2) BS_DS(2);
    else return -1;
#undef BS_DS
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The MoE dispatch plan on the device: pairs sorted by expert with tokens ascending inside an expert, each token's
// pair indices in ascending expert id, plus the tile list of the grouped expert GEMMs (one entry per `bm` rows of
// an expert that has rows). No host round trip: per-chunk expert counts, one block scanning them, then each chunk
// placing its tokens in order. `err` counts expert ids outside [0, ne) (skipped, their order slot -1), and such an
// id sets bit 0 of `*flag` when it is given and, when `rows_bad` is given, the bit of its row in that bitmap: bit
// `row0 + t` for row t of the plan (`row0`: the slice's first row in its call), the last bit of the map's
// `rows_cap` for any row past it. CH: tokens a chunk (one thread each); NE: the most experts (one scan thread
// each); NP: the most pairs the one-block form takes. A token picks at most kMaxTopk experts.
constexpr int kMaxTopk = 16;

// a plan row's bit in the call's row map (see above)
__device__ __forceinline__ void bs_row_bad(int* rows_bad, int row0, int rows_cap, int t) {
    if (!rows_bad) return;
    int b = min(row0 + t, rows_cap - 1);
    atomicOr(&rows_bad[b >> 5], (int)(1u << (b & 31)));
}

template <int CH>
__global__ void __launch_bounds__(CH) bs_moe_count_kernel(const int* __restrict__ sel, int rows, int topk, int ne,
                                                          int* __restrict__ counts, int* __restrict__ err, int* flag,
                                                          int* rows_bad, int row0, int rows_cap) {
    extern __shared__ int bs_cnt[];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) bs_cnt[e] = 0;
    __syncthreads();
    int t = blockIdx.x * CH + threadIdx.x;
    if (t < rows)
        for (int j = 0; j < topk; ++j) {
            int e = sel[(long)t * topk + j];
            if (e < 0 || e >= ne) {
                atomicAdd(err, 1);
                if (flag) atomicOr(flag, 1);
                bs_row_bad(rows_bad, row0, rows_cap, t);
            } else {
                atomicAdd(&bs_cnt[e], 1);
            }
        }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) counts[(long)blockIdx.x * ne + e] = bs_cnt[e];
}

// counts [nch][ne] -> each chunk's base inside its expert (in place); off [ne + 1]; the tile list of the experts
// [e0, e1) (a rank's share), expert index relative to e0
template <int NE>
__global__ void __launch_bounds__(NE) bs_moe_scan_kernel(int* __restrict__ counts, int nch, int ne, int bm, int e0,
                                                         int e1, int* __restrict__ off, Tile* __restrict__ tiles,
                                                         int* __restrict__ ntiles) {
    __shared__ int s_off[NE], s_til[NE];
    int e = threadIdx.x;
    int run = 0;
    if (e < ne)
        for (int c = 0; c < nch; ++c) {
            int v = counts[(long)c * ne + e];
            counts[(long)c * ne + e] = run;
            run += v;
        }
    s_off[e] = e < ne ? run : 0;
    s_til[e] = (e >= e0 && e < e1) ? (run + bm - 1) / bm : 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, tt = 0;
        for (int x = 0; x < ne; ++x) {
            int c = s_off[x], n = s_til[x];
            s_off[x] = o;
            s_til[x] = tt;
            o += c;
            tt += n;
        }
        off[ne] = o;
        *ntiles = tt;
    }
    __syncthreads();
    if (e < ne) {
        off[e] = s_off[e];
        if (e >= e0 && e < e1)
            for (int i = 0; i * bm < run; ++i) tiles[s_til[e] + i] = Tile{s_off[e], run, i * bm, e - e0};
    }
}

// One thread per token of a CH-token chunk. A pair's slot is its expert's base plus the number of earlier tokens of
// the chunk that picked the same expert (a token's experts are distinct): the chunk's tokens set their bit in a
// per-expert bitmap, and a token counts the bits below its own.
template <int CH>
__global__ void __launch_bounds__(CH) bs_moe_place_kernel(const int* __restrict__ sel, const float* __restrict__ selw,
                                                          int rows, int topk, int ne, const int* __restrict__ bases,
                                                          const int* __restrict__ off, int* __restrict__ ptok,
                                                          float* __restrict__ pw, int* __restrict__ order) {
    extern __shared__ int bs_loc[];  // [ne] the chunk's first slot per expert, then [ne][CH / 32] the token bitmap
    unsigned* mask = (unsigned*)(bs_loc + ne);
    int tid = threadIdx.x;
    for (int e = tid; e < ne; e += CH) bs_loc[e] = off[e] + bases[(long)blockIdx.x * ne + e];
    for (int i = tid; i < ne * (CH / 32); i += CH) mask[i] = 0u;
    __syncthreads();
    int t = blockIdx.x * CH + tid;
    int ev[kMaxTopk];
    float wv[kMaxTopk];
#pragma unroll
    for (int j = 0; j < kMaxTopk; ++j) {
        ev[j] = -1;
        wv[j] = 0.f;
        if (j < topk && t < rows) {
            int e = sel[(long)t * topk + j];
            wv[j] = selw[(long)t * topk + j];
            if (e >= 0 && e < ne) ev[j] = e;
        }
    }
#pragma unroll
    for (int j = 0; j < kMaxTopk; ++j)
        if (ev[j] >= 0) atomicOr(&mask[ev[j] * (CH / 32) + (tid >> 5)], 1u << (tid & 31));
    __syncthreads();
    if (t >= rows) return;
    int n = 0;
#pragma unroll
    for (int j = 0; j < kMaxTopk; ++j) n += ev[j] >= 0;
#pragma unroll
    for (int j = 0; j < kMaxTopk; ++j) {
        int e = ev[j];
        if (e < 0) continue;
        const unsigned* m = mask + e * (CH / 32);
        int before = __popc(m[tid >> 5] & ((1u << (tid & 31)) - 1u));
#pragma unroll
        for (int w = 0; w < CH / 32; ++w)
            if (w < (tid >> 5)) before += __popc(m[w]);
        int p = bs_loc[e] + before;
        ptok[p] = t;
        pw[p] = wv[j];
        int q = 0;  // this expert's rank among the token's experts: its place in the per-token order
#pragma unroll
        for (int k = 0; k < kMaxTopk; ++k) q += (ev[k] >= 0 && ev[k] < e);
        order[(long)t * topk + q] = p;
    }
#pragma unroll
    for (int j = 0; j < kMaxTopk; ++j)
        if (j < topk && j >= n) order[(long)t * topk + j] = -1;
}

// The plan of a call of a few rows (at most NP pairs) in one block instead of three kernels and three zero-fills:
// the same outputs. Nothing is zeroed beforehand: the slots past the placed pairs are written here (token 0, weight
// 0), and err is stored.
template <int NP, int NE>
__global__ void __launch_bounds__(NE) bs_moe_plan_small_kernel(const int* __restrict__ sel,
                                                               const float* __restrict__ selw, int rows, int topk,
                                                               int ne, int bm, int e0, int e1, int* __restrict__ off,
                                                               int* __restrict__ ptok, float* __restrict__ pw,
                                                               int* __restrict__ order, Tile* __restrict__ tiles,
                                                               int* __restrict__ ntiles, int* __restrict__ err,
                                                               int* flag, int* rows_bad, int row0, int rows_cap) {
    __shared__ int s_sel[NP], s_off[NE], s_til[NE], s_placed;
    int tid = threadIdx.x, np = rows * topk;
    if (tid < np) {
        int e = sel[tid];
        bool ok = e >= 0 && e < ne;
        s_sel[tid] = ok ? e : -1;
        if (!ok) bs_row_bad(rows_bad, row0, rows_cap, tid / topk);
    }
    __syncthreads();
    int run = 0;  // expert tid's pairs
    if (tid < ne)
        for (int p = 0; p < np; ++p) run += s_sel[p] == tid;
    s_off[tid] = run;
    s_til[tid] = (tid >= e0 && tid < e1) ? (run + bm - 1) / bm : 0;
    __syncthreads();
    if (tid == 0) {
        int o = 0, tt = 0, bad = 0;
        for (int x = 0; x < ne; ++x) {
            int c = s_off[x], n = s_til[x];
            s_off[x] = o;
            s_til[x] = tt;
            o += c;
            tt += n;
        }
        for (int p = 0; p < np; ++p) bad += s_sel[p] < 0;
        off[ne] = o;
        *ntiles = tt;
        *err = bad;
        if (bad && flag) atomicOr(flag, 1);
        s_placed = o;
    }
    __syncthreads();
    if (tid < ne) {
        off[tid] = s_off[tid];
        if (tid >= e0 && tid < e1)
            for (int i = 0; i * bm < run; ++i) tiles[s_til[tid] + i] = Tile{s_off[tid], run, i * bm, tid - e0};
    }
    if (tid < np) {
        int t = tid / topk, j = tid % topk, e = s_sel[tid];
        int n = 0, q = 0;  // the token's experts in range, and those below e: e's place in the per-token order
        for (int k = 0; k < topk; ++k) {
            int x = s_sel[t * topk + k];
            n += x >= 0;
            q += x >= 0 && x < e;
        }
        if (e >= 0) {
            int before = 0;  // earlier tokens that picked e (a token's experts are distinct)
            for (int p = 0; p < t * topk; ++p) before += s_sel[p] == e;
            int p = s_off[e] + before;
            ptok[p] = t;
            pw[p] = selw[tid];
            order[t * topk + q] = p;
        }
        if (j >= n) order[t * topk + j] = -1;
        if (tid >= s_placed) {
            ptok[tid] = 0;
            pw[tid] = 0.f;
        }
    }
}

// One plan (contract checked by the caller: rows > 0, 0 < ne <= NE, 0 < topk <= kMaxTopk, bm > 0,
// 0 <= e0 <= e1 <= ne, and row0 >= 0, rows_cap > 0 when rows_bad is given). counts: scratch [ceil(rows / CH)][ne]
// ints (unused for at most NP pairs); tiles: room for ceil(rows * topk / bm) + min(ne, rows * topk); the tile list
// covers experts [e0, e1).
template <int CH, int NE, int NP>
static inline void moe_plan_launch(const int* sel, const float* selw, int rows, int topk, int ne, int bm, int e0,
                                   int e1, int* counts, int* off, int* ptok, float* pw, int* order, Tile* tiles,
                                   int* ntiles, int* err, int* flag, int* rows_bad, int row0, int rows_cap,
                                   cudaStream_t st) {
    if (rows * topk <= NP) {
        bs_moe_plan_small_kernel<NP, NE><<<1, NE, 0, st>>>(sel, selw, rows, topk, ne, bm, e0, e1, off, ptok, pw,
                                                           order, tiles, ntiles, err, flag, rows_bad, row0, rows_cap);
        return;
    }
    int nch = (rows + CH - 1) / CH;
    bs_moe_count_kernel<CH><<<nch, CH, ne * 4, st>>>(sel, rows, topk, ne, counts, err, flag, rows_bad, row0,
                                                     rows_cap);
    bs_moe_scan_kernel<NE><<<1, NE, 0, st>>>(counts, nch, ne, bm, e0, e1, off, tiles, ntiles);
    bs_moe_place_kernel<CH><<<nch, CH, ne * 4 * (1 + CH / 32), st>>>(sel, selw, rows, topk, ne, counts, off, ptok, pw,
                                                                     order);
}

}  // namespace bs
