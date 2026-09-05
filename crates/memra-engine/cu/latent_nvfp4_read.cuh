#pragma once
#include <cuda_fp8.h>

// Drop-in read view for the existing attention reduction. The scalar program
// and reduction order stay the same; only the loaded cache operand changes.
struct LatentNvfp4Read {
    const unsigned char* payload;
    const unsigned char* scales;
    const float* macros;
    int* error;
    int width, visible;
    long offset;
    const int* live_pos = nullptr;
    int advance = 0;
    __device__ LatentNvfp4Read operator+(long n) const {
        LatentNvfp4Read r = *this; r.offset += n; return r;
    }
    __device__ float operator[](long i) const {
        i += offset;
        long row = i / width; int col = i % width;
        long limit = live_pos ? (long)live_pos[0] + advance : visible;
        if (i < 0 || limit < 0 || limit > visible || row >= limit) { atomicExch(error, 2); return 0.f; }
        unsigned char sc = scales[row * (width / 16) + col / 16];
        float macro = macros[row];
        if (sc > 0x7e || !isfinite(macro) || macro <= 0.f) { atomicExch(error, 3); return 0.f; }
        __nv_fp8_e4m3 s; s.__x = sc;
        unsigned char packed = payload[row * (width / 2) + col / 2];
        unsigned code = (col & 1) ? packed >> 4 : packed & 15;
        const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        float q = (code & 8) ? -grid[code & 7] : grid[code & 7];
        // E2M1 * E4M3 is exactly representable in f32; apply the f32 macro
        // scale last for one rounding, without FP64 instructions in attention.
        float value = (q * float(s)) * macro;
        if (!isfinite(value)) { atomicExch(error, 4); return 0.f; }
        return value;
    }
};

__device__ __forceinline__ void latent_empty_softmax(const float*) {}
__device__ __forceinline__ void latent_empty_softmax(LatentNvfp4Read cache) {
    atomicExch(cache.error, 6);
}
