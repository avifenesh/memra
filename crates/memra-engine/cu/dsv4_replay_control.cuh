#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Component-stage ABI. Allocations containing these objects must outlive both
// rank executables. One upload per rank/token; no kernel-node parameter updates.
// This header does not enable full-model replay or change a numeric program.
struct Dsv4ReplayInput {
    uint32_t token;
    uint32_t position;
    uint64_t uniform_bits;
    uint32_t window;
    uint32_t capacity;
};

struct Dsv4ReplayControl {
    Dsv4ReplayInput input;
    uint32_t ring_slot;
    uint32_t c4_pending_slot;
    uint32_t c128_pending_slot;
    uint32_t c4_blocks;
    uint32_t c128_blocks;
    uint32_t c4_rope_position;
    uint32_t c128_rope_position;
    uint32_t indexer_count;
    uint32_t attention_slots;
    uint32_t invalid;
};

// Only integer address/cadence control. Compressor pooling, reductions, and
// attention arithmetic remain the existing kernels with their exact loop bounds.
__global__ void dsv4_replay_control_kernel(
    const Dsv4ReplayInput* input, Dsv4ReplayControl* control,
    cudaGraphConditionalHandle c4, cudaGraphConditionalHandle c128) {
    if (threadIdx.x || blockIdx.x) return;
    const auto in = *input;
    *control = {};
    control->input = in;
    control->invalid = !in.window || in.position >= in.capacity ||
                       in.capacity > 0x7fffffffu;
    cudaGraphSetConditional(c4, 0);
    cudaGraphSetConditional(c128, 0);
    if (control->invalid) return;
    control->ring_slot = in.position % in.window;
    control->c4_pending_slot = 4 + in.position % 4; // overlapping C4
    control->c128_pending_slot = in.position % 128;
    control->c4_blocks = (in.position + 1) / 4;
    control->c128_blocks = (in.position + 1) / 128;
    control->c4_rope_position = (in.position / 4) * 4;
    control->c128_rope_position = (in.position / 128) * 128;
    control->indexer_count = control->c4_blocks < 512 ? control->c4_blocks : 512;
    control->attention_slots = in.window + control->indexer_count;
    cudaGraphSetConditional(c4, (in.position + 1) % 4 == 0);
    cudaGraphSetConditional(c128, (in.position + 1) % 128 == 0);
}
