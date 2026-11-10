// dsv41 perf lane: what a row of an Engram table costs the host, by how the table is held.
// The table is N_ROWS rows of W bytes at OFFSET of FILE. G random rows are gathered into a buffer:
//   cold      rows of the file mapping no thread of this process has touched (a minor fault a row)
//   warm 4k   the same rows again after their pages are mapped (caches flushed by a large memset first)
//   anon 4k   rows of a slice read into anonymous memory with 4 KiB pages, all touched
//   anon 2M   the same slice in anonymous memory with transparent huge pages (madvise), all touched
// with 1 thread and with T threads. Prints ns a row and ms for the G rows.
// usage: engram_gather_probe FILE OFFSET N_ROWS W G THREADS SLICE_GB
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static uint64_t rng(uint64_t* s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}
typedef struct { const uint8_t* tab; uint8_t* dst; const uint64_t* idx; size_t n, w; } job_t;
static void* run(void* p) {
    job_t* j = p;
    for (size_t i = 0; i < j->n; ++i) memcpy(j->dst + i * j->w, j->tab + j->idx[i] * j->w, j->w);
    return 0;
}
static double gather(const uint8_t* tab, uint8_t* dst, const uint64_t* idx, size_t g, size_t w, int threads) {
    pthread_t th[64];
    job_t jb[64];
    size_t per = (g + threads - 1) / threads;
    double t0 = now();
    for (int t = 0; t < threads; ++t) {
        size_t a = t * per, b = a + per > g ? g : a + per;
        jb[t] = (job_t){tab, dst + a * w, idx + a, b > a ? b - a : 0, w};
        pthread_create(&th[t], 0, run, &jb[t]);
    }
    for (int t = 0; t < threads; ++t) pthread_join(th[t], 0);
    return now() - t0;
}
static void flush_caches(void) {
    size_t n = 512u << 20;
    uint8_t* x = malloc(n);
    memset(x, 1, n);
    volatile uint8_t s = x[n / 2];
    (void)s;
    free(x);
}
static void line(const char* what, int threads, double s, size_t g) {
    printf("%-22s threads %2d: %7.1f ns a row, %8.2f ms for %zu rows\n", what, threads, s / g * 1e9, s * 1e3, g);
    fflush(stdout);
}
int main(int argc, char** argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s FILE OFFSET N_ROWS W G THREADS SLICE_GB\n", argv[0]); return 2; }
    const char* file = argv[1];
    size_t off = strtoull(argv[2], 0, 10), nrows = strtoull(argv[3], 0, 10), w = strtoull(argv[4], 0, 10);
    size_t g = strtoull(argv[5], 0, 10);
    int threads = atoi(argv[6]);
    size_t slice = strtoull(argv[7], 0, 10) << 30;
    int fd = open(file, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    size_t pg = 4096, map_off = off & ~(pg - 1), lead = off - map_off, len = nrows * w + lead;
    uint8_t* m = mmap(0, len, PROT_READ, MAP_SHARED, fd, map_off);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    const uint8_t* tab = m + lead;
    uint8_t* dst = malloc(g * w);
    memset(dst, 0, g * w);
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    uint64_t* idx = malloc(g * 8);
    // cold, 1 thread and T threads, on two disjoint row sets
    for (int pass = 0; pass < 2; ++pass) {
        int th = pass ? threads : 1;
        for (size_t i = 0; i < g; ++i) idx[i] = rng(&seed) % nrows;
        line("cold file pages", th, gather(tab, dst, idx, g, w, th), g);
        flush_caches();
        line("warm file pages 4k", 1, gather(tab, dst, idx, g, w, 1), g);
        flush_caches();
        line("warm file pages 4k", threads, gather(tab, dst, idx, g, w, threads), g);
    }
    // a slice of the table in anonymous memory, 4 KiB pages and transparent huge pages
    size_t srows = slice / w;
    if (srows > nrows) srows = nrows;
    for (int huge = 0; huge < 2; ++huge) {
        uint8_t* a = mmap(0, srows * w, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (a == MAP_FAILED) { perror("anon"); return 1; }
        madvise(a, srows * w, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        double t0 = now();
        for (size_t o = 0; o < srows * w; o += 1u << 30) {
            size_t n = srows * w - o < (1u << 30) ? srows * w - o : (1u << 30);
            if (pread(fd, a + o, n, off + o) != (ssize_t)n) { perror("pread"); return 1; }
        }
        printf("read %zu GB into anonymous %s pages: %.1f s\n", srows * w >> 30, huge ? "2 MiB (madvise)" : "4 KiB", now() - t0);
        for (size_t i = 0; i < g; ++i) idx[i] = rng(&seed) % srows;
        for (int th = 1; th <= threads; th += threads - 1 > 0 ? threads - 1 : 1) {
            flush_caches();
            line(huge ? "anonymous 2 MiB pages" : "anonymous 4 KiB pages", th, gather(a, dst, idx, g, w, th), g);
            if (threads == 1) break;
        }
        munmap(a, srows * w);
    }
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    (void)f;
    return 0;
}
