// mmq_mma_i8.cuh — shared int8 ggml_cuda_mma tile machinery (memra-owned), extracted from the
// int8-MMA MMQ vendor TUs under the increment-2 relaxed rule: CODE identical, only comments
// differed (q45k pinned the mma source line; the generic comment lives here, the pin stays in
// the TU). Adopters: mmq_q8_0.cu / mmq_q4_0.cu / mmq_q45k.cu ONLY. mmq_nvfp4_w4a8.cu keeps its
// own variant by design (m16n8k16, 16x4/8x4 tiles, NO_DEVICE_CODE, two load_ldmatrix overloads);
// so do fp4/fp8_blk/iq_experts/f32acc. SASS gate: research/kernel-dedup-20260821/RECEIPTS.md.
//
// Contents: struct tile<I, J, T>, load_generic, load_ldmatrix (m8n8.x4.b16), and the
// mma.sync.m16n8k32.row.col.s32.s8.s8.s32 wrapper. Header-inlined statics only — still no
// cross-TU linkage, no external deps, no ggml headers.

#pragma once

// ======================= mma.cuh: tile<>, loads, int8 mma =======================
namespace ggml_cuda_mma {

    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else {
                __trap();
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else {
                __trap();
                return -1;
            }
        }
    };

    template <int I, int J, typename T>
    static __device__ __forceinline__ void load_generic(tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    // rate-audited 2026-08-06, see research/sm120-empirical-capabilities.md
    //   16.06 cyc/warp-MMA, 309.7 TOP/s = the FASTEST int8 form on sm_120. The pipe is K-FREE
    //   (m16n8k16.s8 costs the same 16.06 cyc for HALF the MACs), so k32 is the right depth and
    //   ptxas rejects m16n8k64.s8 -- there is nothing deeper. OPTIMAL, no swap available.
    // int8 MMA (mma.cuh, Ampere+ path): D(s32) += A(s8) * B(s8).
    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
    }
} // namespace ggml_cuda_mma
