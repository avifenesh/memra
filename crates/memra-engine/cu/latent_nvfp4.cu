// Native row-local NVFP4 latent history. No full-precision history shadow.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>

__device__ __forceinline__ float latent_fp4_value(unsigned code) {
    const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (code & 8) ? -grid[code & 7] : grid[code & 7];
}

__device__ __forceinline__ unsigned latent_fp4_code(float x) {
    const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float ax = fabsf(x);
    unsigned best = 0;
    for (unsigned i = 1; i < 8; ++i) {
        float d = fabsf(ax - grid[i]), prev = fabsf(ax - grid[best]);
        if (d < prev || (d == prev && (i & 1) == 0)) best = i;
    }
    return best | (signbit(x) ? 8 : 0);
}

__device__ __forceinline__ float latent_scale_value(unsigned char code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return float(value);
}

struct LatentEncodedBlock {
    unsigned char payload[8];
    unsigned char scale;
    double sse;
};

// Actual rounded storage and consumer arithmetic, not ideal unrounded scales.
// Explicit round-to-nearest operations prevent contraction changing a winner.
__device__ __forceinline__ LatentEncodedBlock latent_encode_block(
    const float* input, float macro, double multiplier) {
    float peak = 0.f;
    for (int j = 0; j < 16; ++j) peak = fmaxf(peak, fabsf(input[j]));
    float ideal = __double2float_rn(__ddiv_rn(
        __dmul_rn(double(peak), multiplier), __dmul_rn(6.0, double(macro))));
    LatentEncodedBlock block = {};
    block.scale = __nv_cvt_float_to_fp8(ideal, __NV_SATFINITE, __NV_E4M3);
    float scale = latent_scale_value(block.scale);
    double divisor = __dmul_rn(double(scale), double(macro));
    for (int j = 0; j < 16; ++j) {
        unsigned code = divisor == 0.0 ? 0 : latent_fp4_code(
            __double2float_rn(__ddiv_rn(double(input[j]), divisor)));
        block.payload[j / 2] |= code << ((j % 2) * 4);
        float value = __fmul_rn(__fmul_rn(latent_fp4_value(code), scale), macro);
        double difference = __dsub_rn(double(value), double(input[j]));
        block.sse = __dadd_rn(block.sse, __dmul_rn(difference, difference));
    }
    return block;
}

// One block per appended token. The macro scale belongs to that token, so later
// appends and speculative overwrites cannot alter an already committed prefix.
__global__ void latent_nvfp4_append_kernel(
    unsigned char* payload, unsigned char* scales, float* macros,
    const float* rows, int* error, int slot, int width, int capacity, const int* slot_d) {
    __shared__ float peaks[256];
    __shared__ int invalid[256];
    __shared__ float macro[2];
    __shared__ double totals[3][256];
    __shared__ int selected;
    int row = blockIdx.x, tid = threadIdx.x;
    if (slot_d) slot = slot_d[0];
    if (slot < 0 || slot > capacity || row >= capacity - slot) {
        if (tid == 0) atomicExch(error, 5);
        return;
    }
    const float* input = rows + (int64_t)row * width;
    float peak = 0.f;
    int bad = 0;
    for (int col = tid; col < width; col += blockDim.x) {
        float x = input[col];
        bad |= !isfinite(x);
        peak = fmaxf(peak, fabsf(x));
    }
    peaks[tid] = peak; invalid[tid] = bad;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (tid < offset) {
            peaks[tid] = fmaxf(peaks[tid], peaks[tid + offset]);
            invalid[tid] |= invalid[tid + offset];
        }
        __syncthreads();
    }
    if (invalid[0]) { if (tid == 0) atomicExch(error, 1); return; }
    if (tid == 0) {
        macro[0] = peaks[0] == 0.f ? 1.f : fmaxf(
            __double2float_rn(__ddiv_rn(double(peaks[0]), 2688.0)), __int_as_float(1));
        macro[1] = peaks[0] == 0.f ? 1.f : fmaxf(
            __double2float_rn(__ddiv_rn(double(peaks[0]), 1536.0)), __int_as_float(1));
    }
    __syncthreads();
    double legacy = 0.0, adaptive448 = 0.0, adaptive256 = 0.0;
    for (int block = tid; block < width / 16; block += blockDim.x) {
        const float* xs = input + block * 16;
        LatentEncodedBlock a = latent_encode_block(xs, macro[0], 1.0);
        LatentEncodedBlock b = latent_encode_block(xs, macro[0], 1.5);
        legacy = __dadd_rn(legacy, a.sse);
        adaptive448 = __dadd_rn(adaptive448, b.sse < a.sse ? b.sse : a.sse);
        LatentEncodedBlock c = latent_encode_block(xs, macro[1], 1.0);
        LatentEncodedBlock d = latent_encode_block(xs, macro[1], 1.5);
        adaptive256 = __dadd_rn(adaptive256, d.sse < c.sse ? d.sse : c.sse);
    }
    totals[0][tid] = legacy;
    totals[1][tid] = adaptive448;
    totals[2][tid] = adaptive256;
    __syncthreads();
    // Same block-stride and reduction tree as the CPU reference, for every width.
    for (int offset = 128; offset; offset >>= 1) {
        if (tid < offset) {
            for (int arm = 0; arm < 3; ++arm)
                totals[arm][tid] = __dadd_rn(totals[arm][tid], totals[arm][tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        selected = 0;
        for (int arm = 1; arm < 3; ++arm)
            if (totals[arm][0] < totals[selected][0]) selected = arm;
        if (!isfinite(totals[selected][0])) atomicExch(error, 4);
        else macros[slot + row] = macro[selected == 2 ? 1 : 0];
    }
    __syncthreads();
    if (!isfinite(totals[selected][0])) return;
    float chosen_macro = macro[selected == 2 ? 1 : 0];
    for (int block = tid; block < width / 16; block += blockDim.x) {
        const float* xs = input + block * 16;
        LatentEncodedBlock encoded = latent_encode_block(xs, chosen_macro, 1.0);
        if (selected != 0) {
            LatentEncodedBlock m4 = latent_encode_block(xs, chosen_macro, 1.5);
            if (m4.sse < encoded.sse) encoded = m4;
        }
        scales[(int64_t)(slot + row) * (width / 16) + block] = encoded.scale;
        for (int pair = 0; pair < 8; ++pair)
            payload[(int64_t)(slot + row) * (width / 2) + block * 8 + pair] = encoded.payload[pair];
    }
}

// Decode only requested positions into bounded attention workspace. Negative
// positions are padding and become zero. Invalid positive positions set a sticky
// error and never read beyond the committed history.
template <typename Output>
__global__ void latent_nvfp4_gather_kernel(
    const unsigned char* payload, const unsigned char* scales, const float* macros,
    const int* positions, Output* out, int* error, int count, int width, int visible) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)count * width) return;
    int selection = i / width, col = i % width, row = positions ? positions[selection] : selection;
    if (row < 0) { out[i] = 0.f; return; }
    if (row >= visible) { atomicExch(error, 2); out[i] = 0.f; return; }
    unsigned char scale = scales[(int64_t)row * (width / 16) + col / 16];
    float macro = macros[row];
    if (scale > 0x7e || !isfinite(macro) || macro <= 0.f) {
        atomicExch(error, 3); out[i] = 0.f; return;
    }
    unsigned char byte = payload[(int64_t)row * (width / 2) + col / 2];
    unsigned code = (col & 1) ? byte >> 4 : byte & 15;
    float value = (latent_fp4_value(code) * latent_scale_value(scale)) * macro;
    if (!isfinite(value)) { atomicExch(error, 4); out[i] = 0.f; return; }
    out[i] = value;
}

extern "C" int memra_latent_nvfp4_append(
    unsigned char* payload, unsigned char* scales, float* macros,
    const float* rows, int* error, int slot, int count, int width,
    int capacity, void* stream) {
    if (slot < 0 || count < 0 || width <= 0 || width % 16 || capacity < 0 ||
        slot > capacity || count > capacity - slot) return 40001;
    if (!count) return 0;
    latent_nvfp4_append_kernel<<<count, 256, 0, (cudaStream_t)stream>>>(
        payload, scales, macros, rows, error, slot, width, capacity, nullptr);
    cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? 0 : 10000 + int(rc);
}

extern "C" int memra_latent_nvfp4_append_live(
    unsigned char* payload, unsigned char* scales, float* macros,
    const float* rows, int* error, const int* slot_d, int count,
    int width, int capacity, void* stream) {
    if (!slot_d || count < 0 || width <= 0 || width % 16 || capacity < 0 || count > capacity) return 40001;
    if (!count) return 0;
    latent_nvfp4_append_kernel<<<count, 256, 0, (cudaStream_t)stream>>>(
        payload, scales, macros, rows, error, 0, width, capacity, slot_d);
    cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? 0 : 10000 + int(rc);
}

extern "C" int memra_latent_nvfp4_gather(
    const unsigned char* payload, const unsigned char* scales, const float* macros,
    const int* positions, float* out, int* error, int count, int width,
    int visible, void* stream) {
    if (count < 0 || width <= 0 || width % 16 || visible < 0) return 40001;
    int64_t elements = (int64_t)count * width;
    if (!elements) return 0;
    int64_t blocks = (elements + 255) / 256;
    if (blocks > INT32_MAX) return 40001;
    latent_nvfp4_gather_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        payload, scales, macros, positions, out, error, count, width, visible);
    cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? 0 : 10000 + int(rc);
}

// Per-layer transient BF16 operand for the existing tensor-core prefill chain.
// It never becomes persistent history and is charged as workspace, not cache.
extern "C" int memra_latent_nvfp4_to_bf16(
    const unsigned char* payload, const unsigned char* scales, const float* macros,
    unsigned short* out, int* error, int visible, int width, void* stream) {
    if (visible < 0 || width <= 0 || width % 16) return 40001;
    int64_t elements = (int64_t)visible * width;
    if (!elements) return 0;
    int64_t blocks = (elements + 255) / 256;
    if (blocks > INT32_MAX) return 40001;
    latent_nvfp4_gather_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        payload, scales, macros, nullptr, (__nv_bfloat16*)out, error, visible, width, visible);
    cudaError_t rc = cudaGetLastError();
    return rc == cudaSuccess ? 0 : 10000 + int(rc);
}
