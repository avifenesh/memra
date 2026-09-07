// Standalone SM120 probe for the DSV4 mixed FP8-QAT / ModelOpt-FP4 question.
//
// This file deliberately has no engine or FFI dependency.  The default build is
// a device-only fatbin probe.  Defining DSV4_MIXED_MMA_HOST_TEST also builds a
// CPU-only oracle/red-control harness; it does not allocate or launch CUDA work.
//
// Numeric class under test:
//
//   dsv4-mixed-k16-zero-pad-f32-scale-r0
//
// mxf8f6f4 is k32-only on SM120.  ModelOpt has a signed E4M3 scale for every
// K16 group, so this prototype issues two identity block-scale k32 MMAs per
// k32, zero-padding the other K16 half, then applies that group's signed scale
// to the f32 partial before an explicit ordered add.  It keeps raw E2M1 and
// raw E4M3 bytes in their native operands: there is no activation or weight
// requantization.  This is a new reduction-order class, not a byte-identity
// claim for the existing f16 path.

#include <cstdint>

static __device__ __forceinline__ void dsv4_mixed_plain(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void dsv4_mixed_block(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    constexpr unsigned identity_ue8m0 = 0x7f7f7f7fu;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{0,0},{%11},{0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(identity_ue8m0), "r"(identity_ue8m0));
}

// The m1 output-neuron transpose puts the E2M1 weight rows in A and the
// replicated FP8 input vector in B: A is 16 output neurons x K, B is K x 8.
// This is the opposite operand order from dsv4_mixed_block above and is the
// form the down-projection prototype needs.  PTX accepts the mixed pair in
// either operand order; mxf4nvf4 does not provide this FP8 activation path.
static __device__ __forceinline__ void dsv4_mixed_block_w4a8(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    constexpr unsigned identity_ue8m0 = 0x7f7f7f7fu;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},"
        "{%10},{0,0},{%11},{0,0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(identity_ue8m0), "r"(identity_ue8m0));
}

extern "C" __global__ void dsv4_mixed_f8f6f4_plain(
        float* out, const unsigned* in_a, const unsigned* in_b) {
    unsigned a[4] = {in_a[0], in_a[1], in_a[2], in_a[3]};
    unsigned b[2] = {in_b[0], in_b[1]};
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    dsv4_mixed_plain(d, a, b);
    // The MMA is warp-cooperative; only lane 0 publishes the probe result so
    // an optional one-warp launch has no write race.
    if (threadIdx.x == 0) {
        out[0] = d[0];
        out[1] = d[1];
        out[2] = d[2];
        out[3] = d[3];
    }
}

extern "C" __global__ void dsv4_mixed_f8f6f4_block(
        float* out, const unsigned* in_a, const unsigned* in_b) {
    unsigned a[4] = {in_a[0], in_a[1], in_a[2], in_a[3]};
    unsigned b[2] = {in_b[0], in_b[1]};
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    dsv4_mixed_block(d, a, b);
    if (threadIdx.x == 0) {
        out[0] = d[0];
        out[1] = d[1];
        out[2] = d[2];
        out[3] = d[3];
    }
}

// Standard SM80/SM120 8-bit fragment layout, copied only as a small standalone
// loader.  The source rows are byte containers: E2M1 is placed in bits [5:2]
// (code << 2), while E4M3 is already a normal 8-bit value.  The same mapping is
// used by the in-repo FP8/FP4 probes, but this file does not include those TUs.
static __device__ __forceinline__ void dsv4_ld_a8_m16n8k32(
        unsigned (&a)[4], const unsigned char* shared_a) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned* xs = reinterpret_cast<const unsigned*>(shared_a)
                       + (lane % 16u) * 8u + (lane / 16u) * 4u;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(xs));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
}

static __device__ __forceinline__ void dsv4_ld_b8_m16n8k32(
        unsigned (&b)[2], const unsigned char* shared_b) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned* xs = reinterpret_cast<const unsigned*>(shared_b)
                       + (lane % 8u) * 8u + ((lane / 8u) & 1u) * 4u;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(xs));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1},[%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}

static __device__ __forceinline__ float dsv4_e4m3fn_raw(uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v;
    if (exp == 0u) {
        v = static_cast<float>(man) * 0x1p-9f;
    } else {
        v = __uint_as_float(((exp + 120u) << 23) | (man << 20));
    }
    return (x & 0x80u) ? -v : v;
}

// One block covers a 16-output-neuron tile and eight replicated N columns.
// `weight_nibbles` is row-major packed E2M1 (two codes/byte), `act_e4m3` is a
// single FP8 activation row of length K, and `scale_e4m3` is row-major with one
// signed E4M3 byte per output row and K16 group.  This is a compile-only
// prototype; no launcher or production gate is attached here.
extern "C" __global__ void dsv4_mixed_m1_k16_scaled(
        float* out, const unsigned char* weight_nibbles,
        const unsigned char* act_e4m3, const float* act_scales,
        const unsigned char* scale_e4m3, int k) {
    __shared__ __align__(16) unsigned char shared_a[16 * 32];
    __shared__ __align__(16) unsigned char shared_b[8 * 32];

    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int row_base = static_cast<int>(blockIdx.x) * 16;
    const int weight_row_bytes = (k + 1) >> 1;
    const int scale_groups = (k + 15) >> 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < (k + 31) / 32; ++kb) {
        // Two independent k16 partials.  The inactive half is explicitly zero,
        // because SM120 mxf8f6f4 has no m16n8k16 sibling.
        for (int part = 0; part < 2; ++part) {
            for (int i = lane; i < 16 * 32; i += 32) {
                const int row = i / 32;
                const int kk = i & 31;
                const int absolute_k = kb * 32 + part * 16 + kk;
                unsigned char v = 0;
                if (kk < 16 && absolute_k < k) {
                    const unsigned char packed = weight_nibbles[
                        static_cast<size_t>(row_base + row) * weight_row_bytes +
                        (absolute_k >> 1)];
                    const unsigned code = (absolute_k & 1) ? (packed >> 4) : (packed & 0xfu);
                    // The SM120 8-bit container uses the middle four bits for
                    // E2M1.  Do not use the doubled software LUT here.
                    v = static_cast<unsigned char>(code << 2);
                }
                shared_a[i] = v;
            }
            for (int i = lane; i < 8 * 32; i += 32) {
                const int n = i / 32;
                const int kk = i & 31;
                const int absolute_k = kb * 32 + part * 16 + kk;
                shared_b[i] = (kk < 16 && absolute_k < k) ? act_e4m3[absolute_k] : 0;
                (void)n; // B is the same input row in all eight output columns.
            }
            __syncthreads();

            unsigned a[4], b[2];
            dsv4_ld_a8_m16n8k32(a, shared_a);
            dsv4_ld_b8_m16n8k32(b, shared_b);
            float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dsv4_mixed_block_w4a8(partial, a, b);

            for (int l = 0; l < 4; ++l) {
                const int row = (l / 2) * 8 + (lane / 4);
                const int group = kb * 2 + part;
                const float weight_scale = (kb * 32 + part * 16 < k)
                    ? dsv4_e4m3fn_raw(scale_e4m3[
                        static_cast<size_t>(row_base + row) * scale_groups + group])
                    : 0.0f;
                const float activation_scale = (kb * 32 < k)
                    ? act_scales[kb >> 2] : 0.0f; // one scale per K128
                // Named class: identity MMA -> signed E4M3 scale -> ordered
                // f32 add.  This is deliberately not folded into A or B.
                const float weighted = __fmul_rn(partial[l], weight_scale);
                acc[l] = __fadd_rn(acc[l], __fmul_rn(weighted, activation_scale));
            }
            __syncthreads();
        }
    }

    for (int l = 0; l < 4; ++l) {
        const int row = (l / 2) * 8 + (lane / 4);
        const int col = (lane % 4) * 2 + (l & 1);
        out[static_cast<size_t>(row_base + row) * 8 + col] = acc[l];
    }
}

// Register-fed twin of dsv4_mixed_m1_k16_scaled.  It intentionally does not
// share the shared-memory/ldmatrix path: every lane constructs the exact
// 8-bit MMA fragments from global bytes.  The fragment map is the proven
// m16n8k32 row.col map, written out rather than inferred from the output:
//   row = lane / 4, chunk = lane % 4
//   A0 = four E2M1 containers from row, A1 = four from row+8,
//   A2/A3 = zero for the first K16; the second K16 uses A2/A3 instead.
//   B0 = four E4M3 bytes for the replicated input column, B1 = zero for the
//   first K16; the second K16 uses B1 instead.
// This is a correctness prototype.  It deliberately accepts the redundant
// replicated B loads so the shared staging, ldmatrix, and barriers disappear.
static __device__ __forceinline__ void dsv4_mixed_m1_k16_scaled_reg_body(
        float* out, const unsigned char* weight_nibbles,
        const unsigned char* act_e4m3, const float* act_scales,
        const unsigned char* scale_e4m3, int k, int row_base) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int row = lane / 4;
    const int chunk = lane & 3;
    const int weight_row_bytes = (k + 1) >> 1;
    const int scale_groups = (k + 15) >> 4;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = 0; kb < (k + 31) / 32; ++kb) {
        for (int part = 0; part < 2; ++part) {
            unsigned a[4] = {0u, 0u, 0u, 0u};
            unsigned b[2] = {0u, 0u};
            const int start = kb * 32 + part * 16;
            for (int j = 0; j < 4; ++j) {
                const int absolute_k = start + chunk * 4 + j;
                if (absolute_k >= k) continue;
                const unsigned char packed0 = weight_nibbles[
                    static_cast<size_t>(row_base + row) * weight_row_bytes +
                    (absolute_k >> 1)];
                const unsigned char packed8 = weight_nibbles[
                    static_cast<size_t>(row_base + row + 8) * weight_row_bytes +
                    (absolute_k >> 1)];
                const unsigned code0 = (absolute_k & 1) ? (packed0 >> 4) : (packed0 & 0xfu);
                const unsigned code8 = (absolute_k & 1) ? (packed8 >> 4) : (packed8 & 0xfu);
                const unsigned char v0 = static_cast<unsigned char>(code0 << 2);
                const unsigned char v8 = static_cast<unsigned char>(code8 << 2);
                const unsigned char av = act_e4m3[absolute_k];
                const unsigned shift = static_cast<unsigned>(j * 8);
                if (part == 0) {
                    a[0] |= static_cast<unsigned>(v0) << shift;
                    a[1] |= static_cast<unsigned>(v8) << shift;
                    b[0] |= static_cast<unsigned>(av) << shift;
                } else {
                    a[2] |= static_cast<unsigned>(v0) << shift;
                    a[3] |= static_cast<unsigned>(v8) << shift;
                    b[1] |= static_cast<unsigned>(av) << shift;
                }
            }

            float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dsv4_mixed_block_w4a8(partial, a, b);
            for (int l = 0; l < 4; ++l) {
                const int output_row = (l / 2) * 8 + row;
                const int group = kb * 2 + part;
                const float weight_scale = (start < k)
                    ? dsv4_e4m3fn_raw(scale_e4m3[
                        static_cast<size_t>(row_base + output_row) * scale_groups + group])
                    : 0.0f;
                const float activation_scale = (kb * 32 < k) ? act_scales[kb >> 2] : 0.0f;
                const float weighted = __fmul_rn(partial[l], weight_scale);
                acc[l] = __fadd_rn(acc[l], __fmul_rn(weighted, activation_scale));
            }
        }
    }

    for (int l = 0; l < 4; ++l) {
        const int output_row = (l / 2) * 8 + row;
        const int col = (lane % 4) * 2 + (l & 1);
        out[static_cast<size_t>(row_base + output_row) * 8 + col] = acc[l];
    }
}

extern "C" __global__ void dsv4_mixed_m1_k16_scaled_reg(
        float* out, const unsigned char* weight_nibbles,
        const unsigned char* act_e4m3, const float* act_scales,
        const unsigned char* scale_e4m3, int k) {
    dsv4_mixed_m1_k16_scaled_reg_body(
        out, weight_nibbles, act_e4m3, act_scales, scale_e4m3, k,
        static_cast<int>(blockIdx.x) * 16);
}

// Packed sequential/grouped entry points share the exact register-fed body.
// Their only difference is expert pointer arithmetic: the first is launched
// once per expert, the second uses grid.y to select all experts in one launch.
extern "C" __global__ void dsv4_mixed_m1_k16_scaled_reg_packed(
        float* out, const unsigned char* weight_nibbles,
        const unsigned char* act_e4m3, const float* act_scales,
        const unsigned char* scale_e4m3, int k, int expert,
        int out_stride, int weight_stride, int act_stride,
        int act_scale_stride, int scale_stride) {
    const size_t e = static_cast<size_t>(expert);
    dsv4_mixed_m1_k16_scaled_reg_body(
        out + e * out_stride, weight_nibbles + e * weight_stride,
        act_e4m3 + e * act_stride, act_scales + e * act_scale_stride,
        scale_e4m3 + e * scale_stride, k,
        static_cast<int>(blockIdx.x) * 16);
}

extern "C" __global__ void dsv4_mixed_m1_k16_scaled_reg_grouped(
        float* out, const unsigned char* weight_nibbles,
        const unsigned char* act_e4m3, const float* act_scales,
        const unsigned char* scale_e4m3, int k, int out_stride,
        int weight_stride, int act_stride, int act_scale_stride,
        int scale_stride) {
    const int expert = static_cast<int>(blockIdx.y);
    const size_t e = static_cast<size_t>(expert);
    dsv4_mixed_m1_k16_scaled_reg_body(
        out + e * out_stride, weight_nibbles + e * weight_stride,
        act_e4m3 + e * act_stride, act_scales + e * act_scale_stride,
        scale_e4m3 + e * scale_stride, k,
        static_cast<int>(blockIdx.x) * 16);
}

#if defined(DSV4_MIXED_MMA_HOST_TEST) || defined(DSV4_MIXED_MMA_GPU_TEST)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(DSV4_MIXED_MMA_GPU_TEST)
#include <cuda_runtime.h>
#endif

namespace dsv4_mixed_host {

// Software's ModelOpt convention is intentionally different from a hardware
// raw E2M1 decode: the software LUT is doubled [0,1,2,3,4,6,8,12] and its
// signed-E4M3 helper returns raw_scale*0.5.  Their product is the same as
// raw_e2m1 [0,.5,1,1.5,2,3,4,6] times raw signed E4M3.  The oracle computes
// both sides explicitly so a stray extra 0.5 cannot hide in the harness.
static constexpr int kDoubledFp4[16] =
    {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static float f32_mul(const float a, const float b) {
    volatile float r = a * b;
    return r;
}

static float f32_add(const float a, const float b) {
    volatile float r = a + b;
    return r;
}

static float e4m3fn_raw(const std::uint8_t x) {
    const unsigned mag = x & 0x7fu;
    if (mag == 0u || mag == 0x7fu) return 0.0f;
    const unsigned exp = mag >> 3;
    const unsigned man = mag & 7u;
    float v;
    if (exp == 0u) {
        v = static_cast<float>(man) * 0x1p-9f;
    } else {
        const std::uint32_t bits = ((exp + 120u) << 23) | (man << 20);
        std::memcpy(&v, &bits, sizeof(v));
    }
    return (x & 0x80u) ? -v : v;
}

static float modelopt_scale_half(const std::uint8_t x) {
    return f32_mul(e4m3fn_raw(x), 0.5f);
}

static float fp4_e2m1_raw(const std::uint8_t nibble) {
    return f32_mul(static_cast<float>(kDoubledFp4[nibble & 0xfu]), 0.5f);
}

static std::uint8_t activation_pattern(const int k, const int seed) {
    static constexpr std::uint8_t p[] = {
        0x01, 0x77, 0x36, 0xC4, 0x49, 0x0F, 0xA8, 0x51,
        0x18, 0xD1, 0x03, 0x4A, 0xB5, 0x27, 0x70, 0xC1,
    };
    return p[(k * 5 + seed * 3 + (k >> 4)) & 15];
}

static std::vector<float> activation_scales(const int k, const int seed) {
    // DSV4 FP8-QAT uses one f32 power-of-two scale for every K128 group.
    // Keep this plane separate from the FP8 bytes so the GPU gate exercises
    // the same activation-scale epilogue as the intended tile.
    static constexpr float p[] = {1.0f, 0.5f, 2.0f, 0.25f, 4.0f, 0.125f, 1.0f, 0.5f};
    const int groups = (k + 127) >> 7;
    std::vector<float> result(groups);
    for (int g = 0; g < groups; ++g) result[g] = p[(g * 3 + seed) & 7];
    return result;
}

static std::uint8_t weight_nibble(const int row, const int k, const int seed) {
    // Includes positive, negative, zero, and non-adjacent codes; it is not a
    // repeated all-ones fragment that could mask the scale or lane ordering.
    return static_cast<std::uint8_t>((row * 3 + k * 5 + (k >> 4) * 7 + seed * 11) & 15);
}

static std::uint8_t scale_pattern(const int row, const int group, const int seed) {
    static constexpr std::uint8_t p[] = {
        0x44, // +3.0 raw E4M3, +1.5 in the doubled ModelOpt helper
        0xC4, // -3.0 raw E4M3
        0x4C, // +6.0 raw E4M3
        0xB4, // -0.75 raw E4M3
        0x34, // +0.75 raw E4M3
        0xD4, // -6.0 raw E4M3
    };
    return p[(row * 5 + group * 3 + seed) % (sizeof(p) / sizeof(p[0]))];
}

struct Expert {
    int m = 0;
    int k = 0;
    int row_bytes = 0;
    int scale_groups = 0;
    std::vector<std::uint8_t> packed_codes;
    std::vector<std::uint8_t> scales;
};

static Expert make_expert(const int m, const int k, const int seed) {
    Expert e;
    e.m = m;
    e.k = k;
    e.row_bytes = (k + 1) >> 1;
    e.scale_groups = (k + 15) >> 4;
    e.packed_codes.resize(static_cast<std::size_t>(m) * e.row_bytes, 0);
    e.scales.resize(static_cast<std::size_t>(m) * e.scale_groups, 0);
    for (int row = 0; row < m; ++row) {
        for (int k0 = 0; k0 < k; k0 += 2) {
            const std::uint8_t lo = weight_nibble(row, k0, seed);
            const std::uint8_t hi = (k0 + 1 < k) ? weight_nibble(row, k0 + 1, seed) : 0;
            e.packed_codes[static_cast<std::size_t>(row) * e.row_bytes + (k0 >> 1)] =
                static_cast<std::uint8_t>(lo | (hi << 4));
        }
        for (int group = 0; group < e.scale_groups; ++group) {
            e.scales[static_cast<std::size_t>(row) * e.scale_groups + group] =
                scale_pattern(row, group, seed);
        }
    }
    return e;
}

static std::uint8_t read_nibble(const Expert& e, const int row, const int k) {
    const std::uint8_t byte = e.packed_codes[
        static_cast<std::size_t>(row) * e.row_bytes + (k >> 1)];
    return (k & 1) ? static_cast<std::uint8_t>(byte >> 4) : static_cast<std::uint8_t>(byte & 0xfu);
}

// Reference program: decode each raw value and apply its signed K16 scale
// before the f32 accumulation.  The inner product order is ascending K and
// every multiply/add is forced to round at f32, independently of the MMA
// partial program below.
static void explicit_f32_scaled_oracle(
        const Expert& e, const std::vector<std::uint8_t>& act,
        const std::vector<float>& act_scales, const int n,
        std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(e.m) * n, 0.0f);
    for (int row = 0; row < e.m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < e.k; ++k) {
                const std::uint8_t code = read_nibble(e, row, k);
                const std::uint8_t sb = e.scales[
                    static_cast<std::size_t>(row) * e.scale_groups + (k >> 4)];
                // Doubled software LUT * half-scale is exactly raw E2M1 * raw
                // signed E4M3.  Keep the decomposition visible in the oracle.
                const float w = f32_mul(static_cast<float>(kDoubledFp4[code]),
                                        modelopt_scale_half(sb));
                const float raw_term = f32_mul(w, e4m3fn_raw(act[k]));
                const float term = f32_mul(raw_term, act_scales[k >> 7]);
                sum = f32_add(sum, term);
            }
            out[static_cast<std::size_t>(row) * n + col] = sum;
        }
    }
}

// Device prototype's independent CPU twin: two K16 partials per K32.  Each
// partial is the identity mxf8f6f4 product with the other half zero-padded,
// then the signed raw E4M3 scale is applied in f32 and added in K16 order.
static void mixed_k16_zero_padded_f32_scale(
        const Expert& e, const std::vector<std::uint8_t>& act,
        const std::vector<float>& act_scales, const int n,
        std::vector<float>& out) {
    out.assign(static_cast<std::size_t>(e.m) * n, 0.0f);
    for (int row = 0; row < e.m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int base = 0; base < e.k; base += 32) {
                for (int part = 0; part < 2; ++part) {
                    float partial = 0.0f;
                    const int start = base + part * 16;
                    for (int j = 0; j < 16; ++j) {
                        const int k = start + j;
                        if (k >= e.k) continue; // K16 zero padding
                        const float product = f32_mul(
                            fp4_e2m1_raw(read_nibble(e, row, k)), e4m3fn_raw(act[k]));
                        partial = f32_add(partial, product);
                    }
                    const std::uint8_t sb = e.scales[
                        static_cast<std::size_t>(row) * e.scale_groups + (start >> 4)];
                    const float weighted = f32_mul(partial, e4m3fn_raw(sb));
                    const float scaled = f32_mul(weighted, act_scales[base >> 7]);
                    sum = f32_add(sum, scaled);
                }
            }
            out[static_cast<std::size_t>(row) * n + col] = sum;
        }
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::fabs(a[i] - b[i]));
    }
    return result;
}

static float max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float den = std::max(1.0f, std::fabs(b[i]));
        result = std::max(result, std::fabs(a[i] - b[i]) / den);
    }
    return result;
}

static bool all_finite(const std::vector<float>& a) {
    for (const float x : a) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

static void add_in_place(std::vector<float>& dst, const std::vector<float>& src) {
    if (dst.empty()) dst.assign(src.size(), 0.0f);
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = f32_add(dst[i], src[i]);
}

#if !defined(DSV4_MIXED_MMA_GPU_TEST)

static int small_identity_and_red_controls() {
    constexpr int m = 16;
    constexpr int k = 64;
    constexpr int n = 8;
    constexpr int top6 = 6;
    const std::vector<std::uint8_t> act = [&] {
        std::vector<std::uint8_t> v(k);
        for (int i = 0; i < k; ++i) v[i] = activation_pattern(i, 17);
        return v;
    }();
    const std::vector<float> act_scales = activation_scales(k, 17);

    std::vector<float> mixed_sum;
    std::vector<float> oracle_sum;
    for (int expert_id = 0; expert_id < top6; ++expert_id) {
        const Expert e = make_expert(m, k, expert_id + 3);
        std::vector<float> mixed, oracle;
        mixed_k16_zero_padded_f32_scale(e, act, act_scales, n, mixed);
        explicit_f32_scaled_oracle(e, act, act_scales, n, oracle);
        add_in_place(mixed_sum, mixed);
        add_in_place(oracle_sum, oracle);
        if (!all_finite(mixed) || !all_finite(oracle)) {
            std::fprintf(stderr, "FAIL small: non-finite output in expert %d\n", expert_id);
            return 1;
        }
    }

    // Re-running the serial harness must be bit-stable.  This is the host
    // race-free check: no shared scratch or reduction is used by the oracle.
    std::vector<float> repeat_sum;
    for (int expert_id = 0; expert_id < top6; ++expert_id) {
        const Expert e = make_expert(m, k, expert_id + 3);
        std::vector<float> mixed;
        mixed_k16_zero_padded_f32_scale(e, act, act_scales, n, mixed);
        add_in_place(repeat_sum, mixed);
    }
    if (std::memcmp(mixed_sum.data(), repeat_sum.data(),
                    mixed_sum.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL small: repeat is not bit-stable\n");
        return 1;
    }

    const float abs_diff = max_abs_diff(mixed_sum, oracle_sum);
    const float rel_diff = max_rel_diff(mixed_sum, oracle_sum);
    std::printf("small top6 M=%d K=%d N=%d mixed-vs-explicit abs=%.9g rel=%.9g class=K16-zero-pad-f32-scale\n",
                m, k, n, abs_diff, rel_diff);

    // Red control 1: swap adjacent K16 scales.  The result must move; if it
    // does not, a candidate implementation is ignoring the per-16 scale plane.
    Expert swapped = make_expert(m, k, 3);
    for (int row = 0; row < m; ++row) {
        std::swap(swapped.scales[static_cast<std::size_t>(row) * swapped.scale_groups + 0],
                  swapped.scales[static_cast<std::size_t>(row) * swapped.scale_groups + 1]);
    }
    std::vector<float> base_one, swapped_one;
    const Expert base = make_expert(m, k, 3);
    mixed_k16_zero_padded_f32_scale(base, act, act_scales, n, base_one);
    mixed_k16_zero_padded_f32_scale(swapped, act, act_scales, n, swapped_one);
    const float swapped_delta = max_abs_diff(base_one, swapped_one);
    if (!(swapped_delta > 1.0e-4f)) {
        std::fprintf(stderr, "FAIL red control: swapped scales delta=%.9g\n", swapped_delta);
        return 1;
    }

    // Red control 2: clear the sign bit of the negative second K16 group.
    // A path that decodes ModelOpt scales as unsigned must fail this control.
    Expert unsigned_scale = base;
    for (int row = 0; row < m; ++row) {
        unsigned_scale.scales[static_cast<std::size_t>(row) * unsigned_scale.scale_groups + 1]
            &= 0x7fu;
    }
    std::vector<float> unsigned_one;
    mixed_k16_zero_padded_f32_scale(unsigned_scale, act, act_scales, n, unsigned_one);
    const float sign_delta = max_abs_diff(base_one, unsigned_one);
    if (!(sign_delta > 1.0e-4f)) {
        std::fprintf(stderr, "FAIL red control: signed-scale delta=%.9g\n", sign_delta);
        return 1;
    }
    std::printf("small controls PASS: repeat=bit-stable swapped-scale-delta=%.9g signed-scale-delta=%.9g\n",
                swapped_delta, sign_delta);
    std::printf("transpose note: M16 output rows x N8 replicated input; one dot per output row is reused across N8, while the MMA fragment's ordered reduction is two K16 partials per K32.\n");
    return 0;
}

static int full_cpu_benchmark() {
    constexpr int m = 4096;
    constexpr int k = 2048;
    constexpr int n = 8;
    constexpr int top6 = 6;
    std::vector<std::uint8_t> act(k);
    for (int i = 0; i < k; ++i) act[i] = activation_pattern(i, 29);
    const std::vector<float> act_scales = activation_scales(k, 29);
    std::vector<Expert> experts;
    experts.reserve(top6);
    for (int expert_id = 0; expert_id < top6; ++expert_id) {
        experts.push_back(make_expert(m, k, expert_id + 11));
    }

    std::vector<float> mixed_sum;
    const auto mixed_start = std::chrono::steady_clock::now();
    for (const Expert& e : experts) {
        std::vector<float> one;
        mixed_k16_zero_padded_f32_scale(e, act, act_scales, n, one);
        add_in_place(mixed_sum, one);
    }
    const auto mixed_stop = std::chrono::steady_clock::now();

    std::vector<float> oracle_sum;
    const auto oracle_start = std::chrono::steady_clock::now();
    for (const Expert& e : experts) {
        std::vector<float> one;
        explicit_f32_scaled_oracle(e, act, act_scales, n, one);
        add_in_place(oracle_sum, one);
    }
    const auto oracle_stop = std::chrono::steady_clock::now();

    const double mixed_ms = std::chrono::duration<double, std::milli>(mixed_stop - mixed_start).count();
    const double oracle_ms = std::chrono::duration<double, std::milli>(oracle_stop - oracle_start).count();
    std::printf("cpu shape M=%d K=%d N=%d top6 mixed_ms=%.3f explicit_oracle_ms=%.3f mixed_tok_equiv=%.3f\n",
                m, k, n, mixed_ms, oracle_ms, 1000.0 / std::max(1.0, mixed_ms));
    std::printf("cpu full comparison abs=%.9g rel=%.9g; no GPU timing collected\n",
                max_abs_diff(mixed_sum, oracle_sum), max_rel_diff(mixed_sum, oracle_sum));
    return all_finite(mixed_sum) && all_finite(oracle_sum) ? 0 : 1;
}

#endif // !DSV4_MIXED_MMA_GPU_TEST

#if defined(DSV4_MIXED_MMA_GPU_TEST)

// This is intentionally a gate, not a performance claim.  The event interval
// covers only device kernel execution; H2D/D2H and allocation time are outside
// it.  The CPU timings above remain explicitly labeled CPU timings.
// The old aggregate-only bar (0.25 + 2e-3 relative) could hide one expert's
// error behind top6 cancellation.  This bar is intentionally close to f32
// roundoff: 0.0625 absolute for small values and 1e-5 relative for large
// values.  The CPU K16-vs-explicit witness is only 0.007324 abs on the full
// fixture, so this remains a materially tighter component gate.
static constexpr float kGpuAbsTol = 0.0625f;
static constexpr float kGpuRelTol = 1.0e-5f;

static bool cuda_ok(const cudaError_t status, const char* where) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA FAIL at %s: %s\n", where, cudaGetErrorString(status));
    return false;
}

static bool within_gpu_tolerance(const std::vector<float>& got,
                                 const std::vector<float>& expected,
                                 float* max_abs, float* max_rel) {
    *max_abs = 0.0f;
    *max_rel = 0.0f;
    if (got.size() != expected.size()) return false;
    bool pass = true;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - expected[i]);
        const float rel = diff / std::max(1.0f, std::fabs(expected[i]));
        *max_abs = std::max(*max_abs, diff);
        *max_rel = std::max(*max_rel, rel);
        if (!std::isfinite(got[i]) || diff > kGpuAbsTol + kGpuRelTol * std::fabs(expected[i])) {
            pass = false;
        }
    }
    return pass;
}

static bool run_gpu_expert(const Expert& e, const std::vector<std::uint8_t>& act,
                           const std::vector<float>& act_scales, const bool register_fed,
                           std::vector<float>& out, float* kernel_ms) {
    if (e.m % 16 != 0 || e.k % 16 != 0) {
        std::fprintf(stderr, "GPU gate shape must be M/K multiples of 16 (M=%d K=%d)\n", e.m, e.k);
        return false;
    }
    unsigned char* d_codes = nullptr;
    unsigned char* d_act = nullptr;
    float* d_act_scales = nullptr;
    unsigned char* d_scales = nullptr;
    float* d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = true;
    auto cleanup = [&] {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_codes) cudaFree(d_codes);
        if (d_act) cudaFree(d_act);
        if (d_act_scales) cudaFree(d_act_scales);
        if (d_scales) cudaFree(d_scales);
        if (d_out) cudaFree(d_out);
    };
    const std::size_t out_bytes = static_cast<std::size_t>(e.m) * 8 * sizeof(float);
    out.assign(static_cast<std::size_t>(e.m) * 8, 0.0f);
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_codes), e.packed_codes.size()), "cudaMalloc codes")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_act), act.size()), "cudaMalloc act")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_act_scales),
                               act_scales.size() * sizeof(float)), "cudaMalloc act_scales")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_scales), e.scales.size()), "cudaMalloc scales")
        || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_out), out_bytes), "cudaMalloc out")) {
        cleanup();
        return false;
    }
    if (!cuda_ok(cudaMemcpy(d_codes, e.packed_codes.data(), e.packed_codes.size(),
                            cudaMemcpyHostToDevice), "copy codes")
        || !cuda_ok(cudaMemcpy(d_act, act.data(), act.size(), cudaMemcpyHostToDevice), "copy act")
        || !cuda_ok(cudaMemcpy(d_act_scales, act_scales.data(),
                               act_scales.size() * sizeof(float), cudaMemcpyHostToDevice),
                    "copy act_scales")
        || !cuda_ok(cudaMemcpy(d_scales, e.scales.data(), e.scales.size(),
                               cudaMemcpyHostToDevice), "copy scales")
        || !cuda_ok(cudaMemset(d_out, 0, out_bytes), "clear out")
        || !cuda_ok(cudaEventCreate(&start), "event start")
        || !cuda_ok(cudaEventCreate(&stop), "event stop")) {
        cleanup();
        return false;
    }

    cudaEventRecord(start);
    if (register_fed) {
        dsv4_mixed_m1_k16_scaled_reg<<<static_cast<unsigned>(e.m / 16), 32>>>(
            d_out, d_codes, d_act, d_act_scales, d_scales, e.k);
    } else {
        dsv4_mixed_m1_k16_scaled<<<static_cast<unsigned>(e.m / 16), 32>>>(
            d_out, d_codes, d_act, d_act_scales, d_scales, e.k);
    }
    if (!cuda_ok(cudaGetLastError(), register_fed
                                     ? "launch dsv4_mixed_m1_k16_scaled_reg"
                                     : "launch dsv4_mixed_m1_k16_scaled")
        || !cuda_ok(cudaEventRecord(stop), "event stop record")
        || !cuda_ok(cudaEventSynchronize(stop), "event synchronize")) {
        cleanup();
        return false;
    }
    if (!cuda_ok(cudaEventElapsedTime(kernel_ms, start, stop), "event elapsed")
        || !cuda_ok(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost),
                    "copy output")) {
        cleanup();
        return false;
    }
    cleanup();
    return ok;
}

static std::vector<float> cpu_top6_expected(
        const std::vector<Expert>& experts, const std::vector<std::uint8_t>& act,
        const std::vector<float>& act_scales, const bool mixed) {
    std::vector<float> result;
    for (const Expert& e : experts) {
        std::vector<float> one;
        if (mixed) {
            mixed_k16_zero_padded_f32_scale(e, act, act_scales, 8, one);
        } else {
            explicit_f32_scaled_oracle(e, act, act_scales, 8, one);
        }
        add_in_place(result, one);
    }
    return result;
}

static bool compare_gpu_parts(
        const char* label, const std::vector<std::vector<float>>& gpu_parts,
        const std::vector<Expert>& experts, const std::vector<std::uint8_t>& act,
        const std::vector<float>& act_scales, float* worst_abs, float* worst_rel) {
    if (gpu_parts.size() != experts.size()) {
        std::fprintf(stderr, "FAIL GPU %s: expected %zu expert outputs, got %zu\n",
                     label, experts.size(), gpu_parts.size());
        return false;
    }
    *worst_abs = 0.0f;
    *worst_rel = 0.0f;
    bool pass = true;
    for (std::size_t expert_id = 0; expert_id < experts.size(); ++expert_id) {
        std::vector<float> expected;
        mixed_k16_zero_padded_f32_scale(
            experts[expert_id], act, act_scales, 8, expected);
        float abs = 0.0f, rel = 0.0f;
        const bool one_pass = within_gpu_tolerance(gpu_parts[expert_id], expected, &abs, &rel);
        *worst_abs = std::max(*worst_abs, abs);
        *worst_rel = std::max(*worst_rel, rel);
        if (!one_pass) {
            std::fprintf(stderr,
                         "FAIL GPU %s expert=%zu vs-own-CPU-K16 abs=%.9g rel=%.9g\n",
                         label, expert_id, abs, rel);
            pass = false;
        }
    }
    if (pass) {
        std::printf("gpu %s per-expert-vs-own-CPU-K16 PASS n=%zu worst_abs=%.9g worst_rel=%.9g\n",
                    label, experts.size(), *worst_abs, *worst_rel);
    }
    return pass;
}

static bool compare_exact_parts(
        const char* label, const std::vector<std::vector<float>>& lhs,
        const std::vector<std::vector<float>>& rhs) {
    if (lhs.size() != rhs.size()) {
        std::fprintf(stderr, "FAIL GPU %s: exact comparison count mismatch %zu vs %zu\n",
                     label, lhs.size(), rhs.size());
        return false;
    }
    for (std::size_t expert_id = 0; expert_id < lhs.size(); ++expert_id) {
        if (lhs[expert_id].size() != rhs[expert_id].size()
            || std::memcmp(lhs[expert_id].data(), rhs[expert_id].data(),
                           lhs[expert_id].size() * sizeof(float)) != 0) {
            const float abs = (lhs[expert_id].size() == rhs[expert_id].size())
                ? max_abs_diff(lhs[expert_id], rhs[expert_id])
                : std::numeric_limits<float>::infinity();
            std::fprintf(stderr, "FAIL GPU %s expert=%zu bit-diff max_abs=%.9g\n",
                         label, expert_id, abs);
            return false;
        }
    }
    std::printf("gpu %s bit-exact per-expert PASS n=%zu\n", label, lhs.size());
    return true;
}

static bool run_gpu_top6(const char* label, const std::vector<Expert>& experts,
                         const std::vector<std::uint8_t>& act,
                         const std::vector<float>& act_scales,
                         const bool register_fed,
                         std::vector<float>& result, double* kernel_ms,
                         std::vector<std::vector<float>>* per_expert) {
    result.clear();
    *kernel_ms = 0.0;
    if (per_expert) per_expert->clear();
    for (const Expert& e : experts) {
        std::vector<float> one;
        float one_ms = 0.0f;
        if (!run_gpu_expert(e, act, act_scales, register_fed, one, &one_ms)) return false;
        if (per_expert) per_expert->push_back(one);
        add_in_place(result, one);
        *kernel_ms += one_ms;
    }
    std::printf("gpu %s M=%d K=%d top6 kernel_ms=%.3f (H2D/D2H excluded)\n",
                label, experts.front().m, experts.front().k, *kernel_ms);
    return true;
}

struct GroupedFixture {
    int experts = 0;
    int m = 0;
    int k = 0;
    int row_bytes = 0;
    int scale_groups = 0;
    int act_scale_groups = 0;
    int out_stride = 0;
    int weight_stride = 0;
    int act_stride = 0;
    int act_scale_stride = 0;
    int scale_stride = 0;
    std::vector<Expert> weights;
    std::vector<std::vector<std::uint8_t>> acts;
    std::vector<std::vector<float>> act_scales;
};

static GroupedFixture make_grouped_fixture(const int m, const int k, const int seed) {
    GroupedFixture f;
    f.experts = 6;
    f.m = m;
    f.k = k;
    f.row_bytes = (k + 1) >> 1;
    f.scale_groups = (k + 15) >> 4;
    f.act_scale_groups = (k + 127) >> 7;
    f.out_stride = m * 8;
    f.weight_stride = m * f.row_bytes;
    f.act_stride = k;
    f.act_scale_stride = f.act_scale_groups;
    f.scale_stride = m * f.scale_groups;
    f.weights.reserve(f.experts);
    f.acts.reserve(f.experts);
    f.act_scales.reserve(f.experts);
    for (int expert = 0; expert < f.experts; ++expert) {
        const int expert_seed = seed + expert * 17;
        f.weights.push_back(make_expert(m, k, expert_seed + 3));
        f.acts.emplace_back(k);
        for (int i = 0; i < k; ++i) {
            f.acts.back()[i] = activation_pattern(i, expert_seed + 5);
        }
        f.act_scales.push_back(activation_scales(k, expert_seed + 7));

        // The generic fixture generators are intentionally periodic.  Anchor
        // the first K16/output row with an expert-specific, non-periodic
        // record so a full-K expert swap cannot be hidden by cancellation.
#if !defined(DSV4_MIXED_NO_GROUPED_ANCHOR)
        static constexpr std::uint8_t kAnchorScales[6] =
            {0x40, 0x44, 0xC4, 0x4C, 0xB4, 0x34};
        for (int j = 0; j < 16; ++j) {
            const std::uint8_t code = static_cast<std::uint8_t>(
                (expert * 7 + j * 5 + 1) & 0xf);
            std::uint8_t& packed = f.weights.back().packed_codes[j >> 1];
            if (j & 1) packed = static_cast<std::uint8_t>((packed & 0x0f) | (code << 4));
            else packed = static_cast<std::uint8_t>((packed & 0xf0) | code);
        }
        f.weights.back().scales[0] = kAnchorScales[expert];
        f.acts.back()[0] = static_cast<std::uint8_t>(0x31 + expert * 7);
        f.act_scales.back()[0] = (expert & 1) ? 0.5f : 1.0f + static_cast<float>(expert) * 0.25f;
#endif
    }
    return f;
}

static void flatten_grouped_fixture(
        const GroupedFixture& f, std::vector<std::uint8_t>& weights,
        std::vector<std::uint8_t>& acts, std::vector<float>& act_scales,
        std::vector<std::uint8_t>& scales) {
    weights.resize(static_cast<std::size_t>(f.experts) * f.weight_stride);
    acts.resize(static_cast<std::size_t>(f.experts) * f.act_stride);
    act_scales.resize(static_cast<std::size_t>(f.experts) * f.act_scale_stride);
    scales.resize(static_cast<std::size_t>(f.experts) * f.scale_stride);
    for (int expert = 0; expert < f.experts; ++expert) {
        const std::size_t e = static_cast<std::size_t>(expert);
        std::memcpy(weights.data() + e * f.weight_stride,
                    f.weights[expert].packed_codes.data(), f.weight_stride);
        std::memcpy(acts.data() + e * f.act_stride,
                    f.acts[expert].data(), f.act_stride);
        std::memcpy(act_scales.data() + e * f.act_scale_stride,
                    f.act_scales[expert].data(),
                    static_cast<std::size_t>(f.act_scale_stride) * sizeof(float));
        std::memcpy(scales.data() + e * f.scale_stride,
                    f.weights[expert].scales.data(), f.scale_stride);
    }
}

static std::vector<float> cpu_grouped_expected(const GroupedFixture& f) {
    std::vector<float> result(static_cast<std::size_t>(f.experts) * f.out_stride, 0.0f);
    for (int expert = 0; expert < f.experts; ++expert) {
        std::vector<float> one;
        mixed_k16_zero_padded_f32_scale(
            f.weights[expert], f.acts[expert], f.act_scales[expert], 8, one);
        std::memcpy(result.data() + static_cast<std::size_t>(expert) * f.out_stride,
                    one.data(), static_cast<std::size_t>(f.out_stride) * sizeof(float));
    }
    return result;
}

static bool compare_grouped_per_expert(
        const char* label, const GroupedFixture& f,
        const std::vector<float>& got, const std::vector<float>& expected,
        const bool exact, float* worst_abs, float* worst_rel) {
    *worst_abs = 0.0f;
    *worst_rel = 0.0f;
    bool pass = got.size() == expected.size();
    if (!pass) {
        std::fprintf(stderr, "FAIL GPU %s: grouped output size mismatch\n", label);
        return false;
    }
    for (int expert = 0; expert < f.experts; ++expert) {
        const std::size_t off = static_cast<std::size_t>(expert) * f.out_stride;
        std::vector<float> g(got.begin() + off, got.begin() + off + f.out_stride);
        std::vector<float> e(expected.begin() + off, expected.begin() + off + f.out_stride);
        float abs = 0.0f, rel = 0.0f;
        const bool one_pass = exact
            ? std::memcmp(g.data(), e.data(), static_cast<std::size_t>(f.out_stride) * sizeof(float)) == 0
            : within_gpu_tolerance(g, e, &abs, &rel);
        *worst_abs = std::max(*worst_abs, abs);
        *worst_rel = std::max(*worst_rel, rel);
        if (!one_pass) {
            std::fprintf(stderr, "FAIL GPU %s expert=%d abs=%.9g rel=%.9g exact=%d\n",
                         label, expert, abs, rel, exact ? 1 : 0);
            pass = false;
        }
    }
    if (pass) {
        std::printf("gpu %s per-expert %s PASS n=%d worst_abs=%.9g worst_rel=%.9g\n",
                    label, exact ? "bit-exact" : "oracle", f.experts, *worst_abs, *worst_rel);
    }
    return pass;
}

static bool grouped_cpu_control_preflight(
        const char* label, const GroupedFixture& base,
        std::vector<float>& cpu_expected, GroupedFixture& expert_swap,
        GroupedFixture& scale_swap, std::vector<float>& swap_expected,
        std::vector<float>& scale_expected) {
    cpu_expected = cpu_grouped_expected(base);
    expert_swap = base;
    std::swap(expert_swap.weights[0], expert_swap.weights[1]);
    std::swap(expert_swap.acts[0], expert_swap.acts[1]);
    std::swap(expert_swap.act_scales[0], expert_swap.act_scales[1]);
    swap_expected = cpu_grouped_expected(expert_swap);
    float pre_swap_abs = 0.0f, pre_swap_rel = 0.0f;
    const bool swap_preflight_rejected = !within_gpu_tolerance(
        std::vector<float>(swap_expected.begin(), swap_expected.begin() + base.out_stride),
        std::vector<float>(cpu_expected.begin(), cpu_expected.begin() + base.out_stride),
        &pre_swap_abs, &pre_swap_rel);

    scale_swap = base;
    for (int row = 0; row < base.m; ++row) {
        std::swap(scale_swap.weights[0].scales[static_cast<std::size_t>(row) * scale_swap.scale_groups + 0],
                  scale_swap.weights[0].scales[static_cast<std::size_t>(row) * scale_swap.scale_groups + 1]);
    }
    scale_expected = cpu_grouped_expected(scale_swap);
    float pre_scale_abs = 0.0f, pre_scale_rel = 0.0f;
    const bool scale_preflight_rejected = !within_gpu_tolerance(
        std::vector<float>(scale_expected.begin(), scale_expected.begin() + base.out_stride),
        std::vector<float>(cpu_expected.begin(), cpu_expected.begin() + base.out_stride),
        &pre_scale_abs, &pre_scale_rel);
    std::printf("cpu %s control preflight: expert-swap-rejected=%d abs=%.9g rel=%.9g "
                "scale-swap-rejected=%d abs=%.9g rel=%.9g\n",
                label, swap_preflight_rejected ? 1 : 0, pre_swap_abs, pre_swap_rel,
                scale_preflight_rejected ? 1 : 0, pre_scale_abs, pre_scale_rel);
    return swap_preflight_rejected && scale_preflight_rejected;
}

struct GroupedDeviceBuffers {
    unsigned char* d_weights = nullptr;
    unsigned char* d_acts = nullptr;
    float* d_act_scales = nullptr;
    unsigned char* d_scales = nullptr;
    float* d_out = nullptr;
    GroupedFixture shape;

    bool init(const GroupedFixture& f) {
        shape = f;
        std::vector<std::uint8_t> weights, acts, scales;
        std::vector<float> act_scales;
        flatten_grouped_fixture(f, weights, acts, act_scales, scales);
        const std::size_t out_bytes = static_cast<std::size_t>(f.experts) * f.out_stride * sizeof(float);
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights.size()), "grouped malloc weights")
            || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_acts), acts.size()), "grouped malloc acts")
            || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_act_scales),
                                   act_scales.size() * sizeof(float)), "grouped malloc act scales")
            || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_scales), scales.size()), "grouped malloc scales")
            || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&d_out), out_bytes), "grouped malloc out")) {
            release();
            return false;
        }
        if (!cuda_ok(cudaMemcpy(d_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice),
                     "grouped copy weights")
            || !cuda_ok(cudaMemcpy(d_acts, acts.data(), acts.size(), cudaMemcpyHostToDevice),
                        "grouped copy acts")
            || !cuda_ok(cudaMemcpy(d_act_scales, act_scales.data(),
                                   act_scales.size() * sizeof(float), cudaMemcpyHostToDevice),
                        "grouped copy act scales")
            || !cuda_ok(cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice),
                        "grouped copy scales")) {
            release();
            return false;
        }
        return true;
    }

    void release() {
        if (d_weights) cudaFree(d_weights);
        if (d_acts) cudaFree(d_acts);
        if (d_act_scales) cudaFree(d_act_scales);
        if (d_scales) cudaFree(d_scales);
        if (d_out) cudaFree(d_out);
        d_weights = nullptr;
        d_acts = nullptr;
        d_act_scales = nullptr;
        d_scales = nullptr;
        d_out = nullptr;
    }

    bool run(const bool grouped, std::vector<float>& out, float* kernel_ms) {
        const std::size_t out_count = static_cast<std::size_t>(shape.experts) * shape.out_stride;
        const std::size_t out_bytes = out_count * sizeof(float);
        out.assign(out_count, 0.0f);
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (!cuda_ok(cudaMemset(d_out, 0, out_bytes), "grouped clear out")
            || !cuda_ok(cudaEventCreate(&start), "grouped event start")
            || !cuda_ok(cudaEventCreate(&stop), "grouped event stop")) {
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
            return false;
        }
        cudaEventRecord(start);
        if (grouped) {
            dsv4_mixed_m1_k16_scaled_reg_grouped<<<
                dim3(static_cast<unsigned>(shape.m / 16), static_cast<unsigned>(shape.experts), 1), 32>>>(
                d_out, d_weights, d_acts, d_act_scales, d_scales, shape.k,
                shape.out_stride, shape.weight_stride, shape.act_stride,
                shape.act_scale_stride, shape.scale_stride);
        } else {
            for (int expert = 0; expert < shape.experts; ++expert) {
                dsv4_mixed_m1_k16_scaled_reg_packed<<<static_cast<unsigned>(shape.m / 16), 32>>>(
                    d_out, d_weights, d_acts, d_act_scales, d_scales, shape.k,
                    expert, shape.out_stride, shape.weight_stride, shape.act_stride,
                    shape.act_scale_stride, shape.scale_stride);
            }
        }
        if (!cuda_ok(cudaGetLastError(), grouped ? "grouped launch" : "sequential packed launches")
            || !cuda_ok(cudaEventRecord(stop), "grouped event record")
            || !cuda_ok(cudaEventSynchronize(stop), "grouped event synchronize")
            || !cuda_ok(cudaEventElapsedTime(kernel_ms, start, stop), "grouped event elapsed")
            || !cuda_ok(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost),
                        "grouped copy out")) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return true;
    }
};

static bool run_grouped_case(const char* label, const int m, const int k, const int seed) {
    const GroupedFixture base = make_grouped_fixture(m, k, seed);
    std::vector<float> cpu_expected, swap_expected, scale_expected;
    GroupedFixture expert_swap, scale_swap;
    if (!grouped_cpu_control_preflight(
            label, base, cpu_expected, expert_swap, scale_swap, swap_expected, scale_expected)) {
        std::fprintf(stderr, "FAIL CPU %s control preflight; fixture is not discriminating\n", label);
        return false;
    }

    GroupedDeviceBuffers buffers;
    if (!buffers.init(base)) return false;

    std::vector<float> sequential_warm, grouped_warm;
    float sequential_warm_ms = 0.0f, grouped_warm_ms = 0.0f;
    if (!buffers.run(false, sequential_warm, &sequential_warm_ms)
        || !buffers.run(true, grouped_warm, &grouped_warm_ms)) {
        buffers.release();
        return false;
    }
    std::vector<float> sequential, grouped;
    float sequential_ms = 0.0f, grouped_ms = 0.0f;
    if (!buffers.run(false, sequential, &sequential_ms)
        || !buffers.run(true, grouped, &grouped_ms)) {
        buffers.release();
        return false;
    }
    buffers.release();
    float seq_abs = 0.0f, seq_rel = 0.0f, grp_abs = 0.0f, grp_rel = 0.0f;
    const bool seq_cpu = compare_grouped_per_expert(
        "grouped-sequential-cpu", base, sequential, cpu_expected, false, &seq_abs, &seq_rel);
    const bool grp_cpu = compare_grouped_per_expert(
        "grouped-cpu", base, grouped, cpu_expected, false, &grp_abs, &grp_rel);
    const bool warm_exact = sequential_warm.size() == sequential.size()
        && grouped_warm.size() == grouped.size()
        && std::memcmp(sequential_warm.data(), sequential.data(), sequential.size() * sizeof(float)) == 0
        && std::memcmp(grouped_warm.data(), grouped.data(), grouped.size() * sizeof(float)) == 0;
    const bool grouped_exact = grouped.size() == sequential.size()
        && std::memcmp(grouped.data(), sequential.data(), sequential.size() * sizeof(float)) == 0;
    std::printf("gpu %s warm grouped_ms=%.3f sequential6_ms=%.3f; measured grouped_ms=%.3f sequential6_ms=%.3f (CUDA kernel events only)\n",
                label, grouped_warm_ms, sequential_warm_ms, grouped_ms, sequential_ms);
    if (!seq_cpu || !grp_cpu || !warm_exact || !grouped_exact) {
        std::fprintf(stderr, "FAIL GPU %s grouped/sequential or CPU comparison\n", label);
        return false;
    }

    // Expert swap control: swapping complete expert records must swap output
    // slots, and the old per-slot oracle must reject the mutation.
    GroupedDeviceBuffers swap_buffers;
    if (!swap_buffers.init(expert_swap)) return false;
    std::vector<float> swap_out;
    float swap_ms = 0.0f;
    const bool swap_run = swap_buffers.run(true, swap_out, &swap_ms);
    swap_buffers.release();
    float swap_abs = 0.0f, swap_rel = 0.0f;
    const bool swap_cpu = swap_run && compare_grouped_per_expert(
        "expert-swap-cpu", expert_swap, swap_out, swap_expected, false, &swap_abs, &swap_rel);
    const bool swap_rejected = swap_run && swap_out.size() == cpu_expected.size()
        && !within_gpu_tolerance(
            std::vector<float>(swap_out.begin(), swap_out.begin() + base.out_stride),
            std::vector<float>(cpu_expected.begin(), cpu_expected.begin() + base.out_stride),
            &swap_abs, &swap_rel);
    if (!swap_cpu || !swap_rejected) {
        std::fprintf(stderr, "FAIL GPU %s expert-swap control\n", label);
        return false;
    }

    // Scale swap control: only expert 0's adjacent K16 scale bytes move.
    GroupedDeviceBuffers scale_buffers;
    if (!scale_buffers.init(scale_swap)) return false;
    std::vector<float> scale_out;
    float scale_ms = 0.0f;
    const bool scale_run = scale_buffers.run(true, scale_out, &scale_ms);
    scale_buffers.release();
    float scale_abs = 0.0f, scale_rel = 0.0f;
    const bool scale_cpu = scale_run && compare_grouped_per_expert(
        "scale-swap-cpu", scale_swap, scale_out, scale_expected, false, &scale_abs, &scale_rel);
    const bool scale_rejected = scale_run && scale_out.size() == cpu_expected.size()
        && !within_gpu_tolerance(
            std::vector<float>(scale_out.begin(), scale_out.begin() + base.out_stride),
            std::vector<float>(cpu_expected.begin(), cpu_expected.begin() + base.out_stride),
            &scale_abs, &scale_rel);
    if (!scale_cpu || !scale_rejected) {
        std::fprintf(stderr, "FAIL GPU %s scale-swap control\n", label);
        return false;
    }
    std::printf("gpu %s grouped controls PASS: expert-swap and scale-swap rejected original slot oracle\n",
                label);
    return true;
}

[[maybe_unused]] static bool run_gpu_case(const char* label, const int m, const int k, const int seed) {
    constexpr int top6 = 6;
    std::vector<std::uint8_t> act(k);
    for (int i = 0; i < k; ++i) act[i] = activation_pattern(i, seed);
    const std::vector<float> act_scales = activation_scales(k, seed);
    std::vector<Expert> base;
    base.reserve(top6);
    for (int expert_id = 0; expert_id < top6; ++expert_id) {
        base.push_back(make_expert(m, k, seed + expert_id + 3));
    }

    const std::vector<float> cpu_expected = cpu_top6_expected(base, act, act_scales, true);
    const std::vector<float> cpu_explicit = cpu_top6_expected(base, act, act_scales, false);
    if (!all_finite(cpu_expected) || !all_finite(cpu_explicit)) {
        std::fprintf(stderr, "FAIL GPU %s: CPU expectation is non-finite\n", label);
        return false;
    }
    std::vector<float> gpu_base;
    std::vector<std::vector<float>> gpu_base_parts;
    double base_ms = 0.0;
    if (!run_gpu_top6(label, base, act, act_scales, false,
                      gpu_base, &base_ms, &gpu_base_parts)) return false;
    float base_part_abs = 0.0f, base_part_rel = 0.0f;
    if (!compare_gpu_parts(label, gpu_base_parts, base, act, act_scales,
                           &base_part_abs, &base_part_rel)) return false;
    float base_abs = 0.0f, base_rel = 0.0f;
    const bool base_pass = within_gpu_tolerance(gpu_base, cpu_expected, &base_abs, &base_rel);
    std::printf("gpu %s base-vs-cpu-K16 %s abs=%.9g rel=%.9g; cpu-K16-vs-explicit abs=%.9g rel=%.9g\n",
                label, base_pass ? "PASS" : "FAIL", base_abs, base_rel,
                max_abs_diff(cpu_expected, cpu_explicit), max_rel_diff(cpu_expected, cpu_explicit));
    if (!base_pass) return false;

    std::vector<float> gpu_reg;
    std::vector<std::vector<float>> gpu_reg_parts;
    double reg_ms = 0.0;
    if (!run_gpu_top6("reg", base, act, act_scales, true,
                      gpu_reg, &reg_ms, &gpu_reg_parts)) return false;
    float reg_part_abs = 0.0f, reg_part_rel = 0.0f;
    if (!compare_gpu_parts("reg", gpu_reg_parts, base, act, act_scales,
                           &reg_part_abs, &reg_part_rel)
        || !compare_exact_parts("reg-vs-shared", gpu_reg_parts, gpu_base_parts)
        || gpu_reg.size() != gpu_base.size()
        || std::memcmp(gpu_reg.data(), gpu_base.data(), gpu_base.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL GPU %s: register-fed aggregate differs from shared prototype\n", label);
        return false;
    }
    std::printf("gpu %s register-fed-vs-shared aggregate bit-exact PASS kernel_ms=%.3f\n",
                label, reg_ms);

    std::vector<float> gpu_repeat;
    std::vector<std::vector<float>> gpu_repeat_parts;
    double repeat_ms = 0.0;
    if (!run_gpu_top6("repeat", base, act, act_scales, false,
                      gpu_repeat, &repeat_ms,
                      &gpu_repeat_parts)) return false;
    float repeat_part_abs = 0.0f, repeat_part_rel = 0.0f;
    if (!compare_gpu_parts("repeat", gpu_repeat_parts, base, act, act_scales,
                           &repeat_part_abs, &repeat_part_rel)) return false;
    if (gpu_repeat.size() != gpu_base.size()
        || std::memcmp(gpu_repeat.data(), gpu_base.data(), gpu_base.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "FAIL GPU %s: repeated base output is not bit-stable\n", label);
        return false;
    }

    // The controls mutate only expert 0.  Both device output and the CPU K16
    // expectation must move, proving that the real launch consumes the scale
    // plane rather than merely matching a host-side model.
    std::vector<Expert> swapped = base;
    for (int row = 0; row < m; ++row) {
        std::swap(swapped[0].scales[static_cast<std::size_t>(row) * swapped[0].scale_groups + 0],
                  swapped[0].scales[static_cast<std::size_t>(row) * swapped[0].scale_groups + 1]);
    }
    std::vector<float> gpu_swapped, cpu_swapped;
    std::vector<std::vector<float>> gpu_swapped_parts;
    double swapped_ms = 0.0;
    if (!run_gpu_top6("reg-swapped-scale", swapped, act, act_scales, true,
                      gpu_swapped, &swapped_ms,
                      &gpu_swapped_parts)) return false;
    cpu_swapped = cpu_top6_expected(swapped, act, act_scales, true);
    float swapped_part_abs = 0.0f, swapped_part_rel = 0.0f;
    if (!compare_gpu_parts("swapped-scale", gpu_swapped_parts, swapped, act, act_scales,
                           &swapped_part_abs, &swapped_part_rel)) return false;
    float swap_abs = 0.0f, swap_rel = 0.0f;
    const float swap_delta = max_abs_diff(gpu_base, gpu_swapped);
    float swap_original_abs = 0.0f, swap_original_rel = 0.0f;
    const bool swap_wrongly_matches_original = within_gpu_tolerance(
        gpu_swapped, cpu_expected, &swap_original_abs, &swap_original_rel);
    if (!(swap_delta > 1.0e-4f)
        || swap_wrongly_matches_original
        || !within_gpu_tolerance(gpu_swapped, cpu_swapped, &swap_abs, &swap_rel)) {
        std::fprintf(stderr,
                     "FAIL GPU %s swapped-scale delta=%.9g mutated_abs=%.9g mutated_rel=%.9g "
                     "original_abs=%.9g original_rel=%.9g original_match=%d\n",
                     label, swap_delta, swap_abs, swap_rel,
                     swap_original_abs, swap_original_rel, swap_wrongly_matches_original ? 1 : 0);
        return false;
    }

    std::vector<Expert> unsigned_scale = base;
    for (int row = 0; row < m; ++row) {
        unsigned_scale[0].scales[static_cast<std::size_t>(row) * unsigned_scale[0].scale_groups + 1]
            &= 0x7fu;
    }
    std::vector<float> gpu_unsigned, cpu_unsigned;
    std::vector<std::vector<float>> gpu_unsigned_parts;
    double unsigned_ms = 0.0;
    if (!run_gpu_top6("reg-unsigned-scale", unsigned_scale, act, act_scales, true,
                      gpu_unsigned, &unsigned_ms, &gpu_unsigned_parts)) return false;
    cpu_unsigned = cpu_top6_expected(unsigned_scale, act, act_scales, true);
    float unsigned_part_abs = 0.0f, unsigned_part_rel = 0.0f;
    if (!compare_gpu_parts("unsigned-scale", gpu_unsigned_parts, unsigned_scale,
                           act, act_scales, &unsigned_part_abs, &unsigned_part_rel)) return false;
    float sign_abs = 0.0f, sign_rel = 0.0f;
    const float sign_delta = max_abs_diff(gpu_base, gpu_unsigned);
    float sign_original_abs = 0.0f, sign_original_rel = 0.0f;
    const bool sign_wrongly_matches_original = within_gpu_tolerance(
        gpu_unsigned, cpu_expected, &sign_original_abs, &sign_original_rel);
    if (!(sign_delta > 1.0e-4f)
        || sign_wrongly_matches_original
        || !within_gpu_tolerance(gpu_unsigned, cpu_unsigned, &sign_abs, &sign_rel)) {
        std::fprintf(stderr,
                     "FAIL GPU %s signed-scale delta=%.9g mutated_abs=%.9g mutated_rel=%.9g "
                     "original_abs=%.9g original_rel=%.9g original_match=%d\n",
                     label, sign_delta, sign_abs, sign_rel,
                     sign_original_abs, sign_original_rel, sign_wrongly_matches_original ? 1 : 0);
        return false;
    }
    std::printf("gpu %s controls PASS: repeat=bit-stable swapped-delta=%.9g signed-delta=%.9g\n",
                label, swap_delta, sign_delta);
    return true;
}

#endif // DSV4_MIXED_MMA_GPU_TEST

} // namespace dsv4_mixed_host

#if defined(DSV4_MIXED_MMA_GPU_TEST)
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--cpu-preflight") == 0) {
        const dsv4_mixed_host::GroupedFixture base =
            dsv4_mixed_host::make_grouped_fixture(4096, 2048, 29);
        std::vector<float> expected, swap_expected, scale_expected;
        dsv4_mixed_host::GroupedFixture expert_swap, scale_swap;
        return dsv4_mixed_host::grouped_cpu_control_preflight(
                   "full", base, expected, expert_swap, scale_swap,
                   swap_expected, scale_expected) ? 0 : 1;
    }
    if (!dsv4_mixed_host::run_grouped_case("grouped-small", 16, 64, 17)) return 1;
    if (!dsv4_mixed_host::run_grouped_case("grouped-full", 4096, 2048, 29)) return 1;
    return 0;
}
#else
int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--host-test") != 0) {
        std::fprintf(stderr, "compile-only mixed MMA probe; use --host-test for CPU oracle/controls\n");
        return 0;
    }
    const int small = dsv4_mixed_host::small_identity_and_red_controls();
    if (small != 0) return small;
    return dsv4_mixed_host::full_cpu_benchmark();
}
#endif

#endif // DSV4_MIXED_MMA_HOST_TEST || DSV4_MIXED_MMA_GPU_TEST
