// R6 native packed E4M3x2 -> BF16x2 cubin source.
//
// The raw kernel characterizes native IEEE behavior, including signed zero
// and the two E4M3 NaN encodings. The engine kernel normalizes E4M3 mag==0
// and mag==0x7f bytes to 0 before issuing the same native conversion, matching
// dsv4_e4m3's serving contract. A CUDA 13.1 driver-only shim validates both
// kernels over all 65536 packed byte pairs.

#include <cuda_runtime.h>

#include <cstdint>

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;

__device__ __forceinline__ uint32_t r6_native_e4m3x2_to_bf16x2(uint16_t packed) {
    uint32_t out = 0;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;"
                 : "=r"(out) : "h"(packed));
    return out;
}

__device__ __forceinline__ uint8_t r6_engine_normalize(uint8_t code) {
    const uint8_t mag = code & 0x7fu;
    return (mag == 0u || mag == 0x7fu) ? 0u : code;
}

extern "C" __global__ void r6_raw_packed_cvt_basis(
        const uint32_t* packed, uint32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = r6_native_e4m3x2_to_bf16x2(static_cast<uint16_t>(packed[i]));
}

extern "C" __global__ void r6_engine_packed_cvt_basis(
        const uint32_t* packed, uint32_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t value = static_cast<uint16_t>(packed[i]);
    const uint16_t normalized = static_cast<uint16_t>(
        (static_cast<uint16_t>(r6_engine_normalize(static_cast<uint8_t>(value >> 8))) << 8)
        | r6_engine_normalize(static_cast<uint8_t>(value)));
    out[i] = r6_native_e4m3x2_to_bf16x2(normalized);
}

int main() { return 0; }
