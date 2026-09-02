#pragma once

// Host/device-constexpr address helpers for Memra's two SM100 block-scaled operand layouts.
// Keeping these in one header lets a CPU-only test exercise the exact formulas the CUDA kernels
// compile, without simulating or opening a CUDA device.
#if defined(__CUDACC__)
#define MEMRA_SM100_HD __host__ __device__
#else
#define MEMRA_SM100_HD
#endif

namespace memra_sm100 {

// Eight-row groups are outermost, followed by 16-byte K cores. Used by the NVFP4 K=64 twin.
MEMRA_SM100_HD constexpr int core_row_outer_offset(
        int row, int byte_col, int bytes_per_row) {
    return (row / 8) * (bytes_per_row / 16) * 128
         + (byte_col / 16) * 128
         + (row % 8) * 16
         + (byte_col % 16);
}

// 16-byte K cores are outermost, followed by eight-row groups. Used by the block-FP8 K=128 twin.
MEMRA_SM100_HD constexpr int core_k_outer_offset(int row, int byte_col, int rows) {
    return (byte_col / 16) * (rows / 8) * 128
         + (row / 8) * 128
         + (row % 8) * 16
         + (byte_col % 16);
}

// One 4X NVFP4 scale vector per row: four UE4M3 bytes occupy the row's full quad.
MEMRA_SM100_HD constexpr int sf4x_offset(int row, int scale_in_vec) {
    return (row % 32) * 16 + (row / 32) * 4 + scale_in_vec;
}

// One 1X FP8 scale byte per row. The other three bytes of the quad stay zero.
MEMRA_SM100_HD constexpr int sf1x_offset(int row) {
    return (row % 32) * 16 + (row / 32) * 4;
}

} // namespace memra_sm100

#undef MEMRA_SM100_HD
