// dsv41 perf lane: how long mapping an Engram table's pages up front takes (madvise MADV_POPULATE_READ on
// the file mapping), GB a second, with 1 thread and with T threads on separate ranges; then the gather of G
// random rows from the populated range. usage: populate_probe FILE OFFSET GB_EACH THREADS
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef struct { uint8_t* p; size_t n; int rc; } job_t;
static void* run(void* a) { job_t* j = a; j->rc = madvise(j->p, j->n, MADV_POPULATE_READ); return 0; }
int main(int argc, char** argv) {
    if (argc < 5) return 2;
    size_t off = strtoull(argv[2], 0, 10) & ~4095ull, gb = strtoull(argv[3], 0, 10) << 30;
    int T = atoi(argv[4]);
    int fd = open(argv[1], O_RDONLY);
    uint8_t* m = mmap(0, 2 * gb, PROT_READ, MAP_SHARED, fd, off);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    double t0 = now();
    int rc = madvise(m, gb, MADV_POPULATE_READ);
    double s = now() - t0;
    printf("populate %zu GB, 1 thread: rc %d, %.2f s, %.1f GB/s\n", gb >> 30, rc, s, (gb >> 30) / s);
    pthread_t th[64]; job_t jb[64];
    t0 = now();
    for (int t = 0; t < T; ++t) { jb[t] = (job_t){m + gb + (gb / T) * t, gb / T, 0}; pthread_create(&th[t], 0, run, &jb[t]); }
    for (int t = 0; t < T; ++t) pthread_join(th[t], 0);
    s = now() - t0;
    printf("populate %zu GB, %d threads: %.2f s, %.1f GB/s\n", gb >> 30, T, s, (gb >> 30) / s);
    // rows from the populated range, one thread
    size_t g = 98304, w = 256, nrows = 2 * gb / w;
    uint8_t* dst = malloc(g * w);
    memset(dst, 0, g * w);
    uint64_t sd = 88172645463325252ull;
    t0 = now();
    for (size_t i = 0; i < g; ++i) { sd ^= sd << 13; sd ^= sd >> 7; sd ^= sd << 17; memcpy(dst + i * w, m + (sd % nrows) * w, w); }
    s = now() - t0;
    printf("gather %zu rows from the populated range, 1 thread: %.1f ns a row, %.2f ms\n", g, s / g * 1e9, s * 1e3);
    return 0;
}
