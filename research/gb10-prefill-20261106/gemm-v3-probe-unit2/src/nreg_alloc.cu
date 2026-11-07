// Does ptxas allocate the code after setmaxnreg.inc at the new count? 384 threads (168 at entry);
// warpgroup 0 deallocates to 40, warpgroups 1-2 allocate 232 and keep NACC live accumulators.
#ifndef NACC
#define NACC 200
#endif
__global__ void __launch_bounds__(384, 1) k(float* out, const float* in, int iters) {
    const int wg = threadIdx.x / 128;
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
        out[threadIdx.x] = -1.0f;
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = in[i];
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = fmaf(acc[i], 0.5f, (float) i);
    }
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; ++i) t += acc[i];
    out[threadIdx.x] = t;
}
