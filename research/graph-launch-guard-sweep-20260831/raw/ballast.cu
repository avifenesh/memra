// VRAM ballast for admission cells (lane/step37-vram-admission-20260830).
// ./ballast <device> <hold_mb> [grow_mb grow_interval_s max_mb]
// Allocates hold_mb on <device>, prints its state, then (optionally) grows by
// grow_mb every grow_interval_s toward max_mb, RETRYING on failure so freed
// VRAM keeps being consumed (the step-OOM squeeze). Holds forever; SIGTERM to release.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ballast <device> <hold_mb> [grow_mb iv_s max_mb]\n"); return 2; }
    int dev = atoi(argv[1]);
    long hold_mb = atol(argv[2]);
    long grow_mb = argc > 3 ? atol(argv[3]) : 0;
    int iv = argc > 4 ? atoi(argv[4]) : 5;
    long max_mb = argc > 5 ? atol(argv[5]) : hold_mb;
    if (cudaSetDevice(dev) != cudaSuccess) { fprintf(stderr, "ballast: bad device %d\n", dev); return 2; }
    long cur = 0;
    while (cur < hold_mb) {
        size_t step = 256L << 20;
        if ((hold_mb - cur) < 256) step = (size_t)(hold_mb - cur) << 20;
        void* p;
        if (cudaMalloc(&p, step) != cudaSuccess) { printf("ballast dev%d initial alloc FAIL at %ld MB\n", dev, cur); fflush(stdout); break; }
        cudaMemset(p, 1, step);
        cur += (long)(step >> 20);
    }
    printf("ballast dev%d holding %ld MB\n", dev, cur); fflush(stdout);
    while (grow_mb > 0 && cur < max_mb) {
        sleep(iv);
        void* p;
        size_t step = (size_t)grow_mb << 20;
        if (cudaMalloc(&p, step) != cudaSuccess) { printf("ballast dev%d grow blocked at %ld MB (retrying)\n", dev, cur); fflush(stdout); continue; }
        cudaMemset(p, 1, step);
        cur += grow_mb;
        printf("ballast dev%d grew to %ld MB\n", dev, cur); fflush(stdout);
    }
    printf("ballast dev%d steady at %ld MB\n", dev, cur); fflush(stdout);
    while (1) sleep(3600);
}
