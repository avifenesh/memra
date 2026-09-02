#include "../../crates/memra-engine/cu/sm100_blockscale_layout.cuh"

#include <array>
#include <cassert>
#include <iostream>

template <std::size_t Size>
static void mark_once(std::array<bool, Size> & seen, int offset) {
    assert(offset >= 0 && static_cast<std::size_t>(offset) < Size);
    assert(!seen[offset]);
    seen[offset] = true;
}

int main() {
    {
        std::array<bool, 128 * 32> seen{};
        for (int row = 0; row < 128; ++row) {
            for (int col = 0; col < 32; ++col) {
                mark_once(seen, memra_sm100::core_row_outer_offset(row, col, 32));
            }
        }
        for (bool bit : seen) { assert(bit); }
        assert(memra_sm100::core_row_outer_offset(0, 0, 32) == 0);
        assert(memra_sm100::core_row_outer_offset(0, 16, 32) == 128);
        assert(memra_sm100::core_row_outer_offset(7, 31, 32) == 255);
        assert(memra_sm100::core_row_outer_offset(8, 0, 32) == 256);
        assert(memra_sm100::core_row_outer_offset(127, 31, 32) == 4095);
    }

    {
        std::array<bool, 128 * 128> seen{};
        for (int row = 0; row < 128; ++row) {
            for (int col = 0; col < 128; ++col) {
                mark_once(seen, memra_sm100::core_k_outer_offset(row, col, 128));
            }
        }
        for (bool bit : seen) { assert(bit); }
        assert(memra_sm100::core_k_outer_offset(0, 0, 128) == 0);
        assert(memra_sm100::core_k_outer_offset(0, 16, 128) == 2048);
        assert(memra_sm100::core_k_outer_offset(7, 15, 128) == 127);
        assert(memra_sm100::core_k_outer_offset(8, 0, 128) == 128);
        assert(memra_sm100::core_k_outer_offset(127, 127, 128) == 16383);
    }

    {
        std::array<bool, 512> seen{};
        for (int row = 0; row < 128; ++row) {
            for (int scale = 0; scale < 4; ++scale) {
                mark_once(seen, memra_sm100::sf4x_offset(row, scale));
            }
        }
        for (bool bit : seen) { assert(bit); }
    }

    {
        std::array<bool, 512> seen{};
        for (int row = 0; row < 128; ++row) {
            mark_once(seen, memra_sm100::sf1x_offset(row));
        }
        assert(memra_sm100::sf1x_offset(0) == 0);
        assert(memra_sm100::sf1x_offset(31) == 496);
        assert(memra_sm100::sf1x_offset(32) == 4);
        assert(memra_sm100::sf1x_offset(127) == 508);
    }

    std::cout << "B200-LAYOUT-CONTRACT PASS\n";
}
