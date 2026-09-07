// R7 dense FP8 -> BF16 MMA candidate, CUDA 13.3 cubin source.
//
// Geometry and arithmetic order are R4's four-warp/64-row, K=128 staging
// candidate. The only decode change is replacing R4's shared BF16 LUT with the
// proven native cvt.rn.bf16x2.e4m3x2 instruction. E4M3 mag==0 and mag==0x7f
// bytes are normalized to 0 before conversion to match dsv4_e4m3; E8M0 scale
// exponents then adjust each exact BF16 half by +/-128 exponent bits.

#include <cuda_runtime.h>

#include <cstdint>

using std::int8_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

namespace r7 {

constexpr int kWarps = 4;
constexpr int kRowsPerWarp = 16;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kTileK = 128;
constexpr int kTileN = 16;

struct CTile { float x[4]; };
struct ATile { uint32_t x[4]; };
struct BTile { uint32_t x[2]; };

static __device__ __forceinline__ uint8_t normalize_e4m3(uint8_t code) {
    const uint8_t mag = code & 0x7fu;
    return (mag == 0u || mag == 0x7fu) ? 0u : code;
}

static __device__ __forceinline__ uint32_t native_e4m3x2_to_bf16x2(uint16_t packed) {
    uint32_t out = 0;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out) : "h"(packed));
    return out;
}

static __device__ __forceinline__ uint16_t scale_bf16_bits(uint16_t base,
                                                             int scale_exp) {
    if ((base & 0x7fffu) == 0u) return 0u;
    return static_cast<uint16_t>(static_cast<int>(base) + scale_exp * 128);
}

static __device__ __forceinline__ void load_a(
        ATile& tile, const uint16_t* base, int stride_bf16, int lane) {
    int* regs = reinterpret_cast<int*>(tile.x);
    const uint32_t* row = reinterpret_cast<const uint32_t*>(base)
        + (lane % 16) * (stride_bf16 / 2) + (lane / 16) * 4;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

static __device__ __forceinline__ void load_b(
        BTile& tile, const uint16_t* base, int stride_bf16, int lane) {
    uint32_t regs[4];
    const uint32_t* row = reinterpret_cast<const uint32_t*>(base)
        + (lane % 16) * (stride_bf16 / 2) + (lane / 16) * 4;
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(regs[0]), "=r"(regs[2]), "=r"(regs[1]), "=r"(regs[3])
        : "r"(addr));
    tile.x[0] = regs[0];
    tile.x[1] = regs[2];
}

static __device__ __forceinline__ int c_row(int lane, int l) {
    return ((l / 2) * 8) + (lane / 4);
}

static __device__ __forceinline__ int c_col(int lane, int l) {
    return ((lane % 4) * 2) + (l % 2);
}

static __device__ __forceinline__ void mma(CTile& d, const ATile& a, const BTile& b) {
    const int* ax = reinterpret_cast<const int*>(a.x);
    const int* bx = reinterpret_cast<const int*>(b.x);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
        : "r"(ax[0]), "r"(ax[1]), "r"(ax[2]), "r"(ax[3]),
          "r"(bx[0]), "r"(bx[1]));
}

extern "C" __global__ void r7_dense_native(
        const uint8_t* __restrict__ codes,
        const int8_t* __restrict__ scale_exp,
        const uint16_t* __restrict__ x,
        float* __restrict__ y, int rows, int k, int sc_cols) {
    __shared__ uint16_t a_s[kWarps][kRowsPerWarp * kTileK];
    __shared__ uint16_t b_s[kTileK * kTileN];
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    CTile acc{};
    acc.x[0] = acc.x[1] = acc.x[2] = acc.x[3] = 0.0f;

    for (int k0 = 0; k0 < k; k0 += kTileK) {
        if (tid < kTileK) {
            const uint16_t value = x[k0 + tid];
#pragma unroll
            for (int col = 0; col < kTileN; ++col)
                b_s[tid * kTileN + col] = value;
        }

        const int row_local = lane / 2;
        const int half = lane & 1;
        const int row = row_base + warp * kRowsPerWarp + row_local;
        const int exponent = row < rows
            ? static_cast<int>(scale_exp[(row >> 7) * sc_cols + (k0 >> 7)]) : 0;
        for (int chunk = half * 8; chunk < kTileK; chunk += 16) {
            uint2 packed = make_uint2(0u, 0u);
            if (row < rows) {
                packed = *reinterpret_cast<const uint2*>(
                    codes + static_cast<long>(row) * k + k0 + chunk);
            }
            const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const uint16_t raw = static_cast<uint16_t>(
                    (words[pair >> 1] >> ((pair & 1) * 16)) & 0xffffu);
                const uint16_t normalized = static_cast<uint16_t>(
                    (static_cast<uint16_t>(normalize_e4m3(
                        static_cast<uint8_t>(raw >> 8))) << 8)
                    | normalize_e4m3(static_cast<uint8_t>(raw)));
                const uint32_t bf16x2 = native_e4m3x2_to_bf16x2(normalized);
                const int dst = row_local * kTileK + chunk + pair * 2;
                a_s[warp][dst] = scale_bf16_bits(
                    static_cast<uint16_t>(bf16x2), exponent);
                a_s[warp][dst + 1] = scale_bf16_bits(
                    static_cast<uint16_t>(bf16x2 >> 16), exponent);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
            ATile a;
            BTile b;
            load_a(a, &a_s[warp][kk], kTileK, lane);
            load_b(b, &b_s[kk * kTileN], kTileN, lane);
            mma(acc, a, b);
        }
        __syncthreads();
    }

#pragma unroll
    for (int l = 0; l < 4; ++l) {
        if (c_col(lane, l) == 0) {
            const int row_out = row_base + warp * kRowsPerWarp + c_row(lane, l);
            if (row_out < rows) y[row_out] = acc.x[l];
        }
    }
}

} // namespace r7

int main() { return 0; }
