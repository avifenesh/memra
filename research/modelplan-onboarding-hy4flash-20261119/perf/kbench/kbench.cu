// Standalone kernel bench + bit-equality check for the hy4f route's small-row GEMMs.
// Build (b23d): nvcc -O3 -arch=sm_121a -std=c++17 -o kbench kbench.cu route.o gpu.o
// where route.o = hy4f_route.cu (default flags), gpu.o = hy4f_gpu.cu (-fmad=false).
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

extern "C" int memra_hy4r_gemm(const float* x, long ldx, long xbs, const void* w, long wbs, const void* sc, long scbs,
                               const float* s2, int fmt, int grid_cols, float* y, long ldy, long ybs, int m, int n,
                               int k, int batch, const int* tiles, int ntiles, int red_dq, int small, int tile_rows,
                               void* stream);
extern "C" int memra_hy4r_tile(const void* src, long rows, long rb, long src_ld, long src_bs, void* dst, long dst_bs,
                               int batch, void* stream);
extern "C" long memra_hy4r_tile_bytes(long rows, long rb);
extern "C" int memra_hy4r_attn(const float* q, const void* kv, const int* list, int ldl, const int* lens, const int* qpos,
                               int mode, int window, int ring, int base, int rows, int H, int C, float scale,
                               const float* sinks, float* po, float* pml, float* out, void* stream);
extern "C" int memra_hy4r_gemm_f32(const float* x, long ldx, const void* w, float* y, long ldy, int m, int n, int k,
                                   void* stream);
extern "C" int memra_hy4f_gemm(const void* x, int64_t ldx, const void* w, const void* scales, int grid_cols,
                               float scale2, int fmt, int round_bf16, void* y, int64_t ldy, int m, int n, int k,
                               void* stream);

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e = (x);                                                         \
    if (e != cudaSuccess) {                                                      \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (uint32_t)(rng >> 16);
}

static void* dev_rand(size_t bytes, int kind) {
  // kind 0: random bytes; 1: bf16 values ~N(0, 0.05); 2: e4m3 scales in [2^-8, 2^-4]; 3: f32 [0.5, 2]
  std::vector<uint8_t> h(bytes);
  if (kind == 0) {
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t)rnd();
    // avoid NaN e4m3 codes (0x7f/0xff) for FP8 weights
    for (size_t i = 0; i < bytes; i++)
      if ((h[i] & 0x7f) == 0x7f) h[i] ^= 1;
  } else if (kind == 1) {
    uint16_t* p = (uint16_t*)h.data();
    for (size_t i = 0; i < bytes / 2; i++) {
      float f = ((int)(rnd() % 2001) - 1000) * 5e-5f;
      uint32_t u;
      memcpy(&u, &f, 4);
      p[i] = (uint16_t)(u >> 16);
    }
  } else if (kind == 2) {
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t)(0x10 + rnd() % 0x18);
  } else {
    float* p = (float*)h.data();
    for (size_t i = 0; i < bytes / 4; i++) p[i] = 0.5f + (rnd() % 1000) * 1.5e-3f;
  }
  void* d;
  CK(cudaMalloc(&d, bytes));
  CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
  return d;
}

static float* dev_x(int m, int k) {
  std::vector<float> h((size_t)m * k);
  for (auto& v : h) v = ((int)(rnd() % 20001) - 10000) * 1e-4f * (1.0f + (rnd() % 7));
  float* d;
  CK(cudaMalloc(&d, h.size() * 4));
  CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  return d;
}

static bool same(const float* a, const float* b, size_t n, const char* what) {
  std::vector<float> ha(n), hb(n);
  CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < n; i++)
    if (memcmp(&ha[i], &hb[i], 4) != 0) {
      if (bad < 3) printf("  %s mismatch at %zu: %.9g vs %.9g\n", what, i, ha[i], hb[i]);
      bad++;
    }
  if (bad) printf("  %s: %zu of %zu differ\n", what, bad, n);
  return bad == 0;
}

__global__ void read_thrash(const uint4* p, size_t n, unsigned* sink) {
  uint4 a = make_uint4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
  }
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x12345678u) *sink = 1;
}

int main(int argc, char** argv) {
  const char* only = argc > 1 ? argv[1] : "";
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  int fails = 0;
  void* thrash;
  const size_t thrash_bytes = 256u << 20;
  CK(cudaMalloc(&thrash, thrash_bytes));
  CK(cudaMemset(thrash, 1, thrash_bytes));
  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  // each rep starts with a cold L2 (a 256 MiB read in between), timed alone
  auto timeit = [&](const char* name, double bytes, auto&& fn) {
    for (int i = 0; i < 3; i++) fn();
    CK(cudaStreamSynchronize(st));
    const int reps = 20;
    double tot = 0;
    for (int i = 0; i < reps; i++) {
      read_thrash<<<192, 256, 0, st>>>((const uint4*)thrash, thrash_bytes / 16, sink);
      CK(cudaEventRecord(e0, st));
      fn();
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      tot += ms;
    }
    const double us = tot * 1e3 / reps;
    printf("%-44s %9.1f us  %7.1f GB/s\n", name, us, bytes / us / 1e3);
  };
  struct Case {
    const char* name;
    int fmt, n, k, batch;
  };
  // dense small-row GEMMs (FMT 0 bf16, 2 fp8, 3 nvfp4 grouped)
  std::vector<Case> cases = {
      {"o_proj fp8 3072x16384", 2, 3072, 16384, 1}, {"o_proj fp8 3072x8192", 2, 3072, 8192, 1},
      {"gate fp8 16384x3072", 2, 16384, 3072, 1},   {"qb fp8 16384x1024", 2, 16384, 1024, 1},
      {"rdown bf16 1536x3072", 0, 1536, 3072, 1},   {"rup bf16 3072x1536", 0, 3072, 1536, 1},
      {"qa bf16 1024x3072", 0, 1024, 3072, 1},      {"kva bf16 576x3072", 0, 576, 3072, 1},
      {"kb_t bf16 64x(512x192)", 0, 512, 192, 64},  {"kvb_v bf16 64x(256x512)", 0, 256, 512, 64},
  };
  for (auto& c : cases) {
    if (*only && !strstr(c.name, only)) continue;
    for (int m : {1, 4}) {
      const size_t wb = c.fmt == 0 ? (size_t)c.n * c.k * 2 : (size_t)c.n * c.k;
      void* wr = dev_rand(wb * c.batch, c.fmt == 0 ? 1 : 0);
      // the route's GEMMs read the tile layout
      const long rb = c.fmt == 0 ? (long)c.k * 2 : (long)c.k;
      void* w;
      CK(cudaMalloc(&w, memra_hy4r_tile_bytes(c.n, rb) * c.batch));
      if (memra_hy4r_tile(wr, c.n, rb, rb, (long)wb, w, memra_hy4r_tile_bytes(c.n, rb), c.batch, st)) fails++;
      CK(cudaFree(wr));
      const int gc = (c.k + 127) / 128;
      void* sc = c.fmt == 2 ? dev_rand((size_t)((c.n + 127) / 128) * gc * 4, 3) : nullptr;
      float* x = dev_x(m * c.batch, c.k);
      float *y0, *y1;
      CK(cudaMalloc(&y0, (size_t)m * c.n * c.batch * 4));
      CK(cudaMalloc(&y1, (size_t)m * c.n * c.batch * 4));
      for (int red = 0; red < (c.fmt == 2 ? 2 : 1); red++) {
        int rc0 = memra_hy4r_gemm(x, c.k, (long)m * c.k, w, (long)wb, sc, 0, nullptr, c.fmt, gc, y0, c.n,
                                  (long)m * c.n, m, c.n, c.k, c.batch, nullptr, 0, red, 0, 0, st);
        int rc1 = memra_hy4r_gemm(x, c.k, (long)m * c.k, w, (long)wb, sc, 0, nullptr, c.fmt, gc, y1, c.n,
                                  (long)m * c.n, m, c.n, c.k, c.batch, nullptr, 0, red, 1, 0, st);
        CK(cudaStreamSynchronize(st));
        char nm[128];
        snprintf(nm, sizeof nm, "%s m%d%s", c.name, m, red ? " red" : "");
        if (rc0 || rc1) {
          printf("%s: rc %d %d\n", nm, rc0, rc1);
          fails++;
          continue;
        }
        if (!same(y0, y1, (size_t)m * c.n * c.batch, nm)) fails++;
        if (!red)
          timeit(nm, (double)wb * c.batch, [&] {
            memra_hy4r_gemm(x, c.k, (long)m * c.k, w, (long)wb, sc, 0, nullptr, c.fmt, gc, y1, c.n, (long)m * c.n, m,
                            c.n, c.k, c.batch, nullptr, 0, 0, 1, 0, st);
          });
      }
      CK(cudaFree(w));
      if (sc) CK(cudaFree(sc));
      CK(cudaFree(x));
      CK(cudaFree(y0));
      CK(cudaFree(y1));
    }
  }
  // decode attention: a sliding layer (64 heads, a 512-row ring at position 2104) and an indexed
  // layer (32 heads, a recall of 2,048 keys of 2,112); times only (the values are checked by the
  // route's tapes)
  if (*only && strstr(only, "attn")) {
    const int cap = 2112, p = 2104;
    float* kv = dev_x(cap, 576);
    float* q = dev_x(64, 576);
    float* sinks = dev_x(1, 64);
    std::vector<int> hl(2048);
    for (int i = 0; i < 2048; i++) hl[i] = (i * 1031) % cap;
    int *list, *lens, *qp;
    CK(cudaMalloc(&list, 2048 * 4));
    CK(cudaMemcpy(list, hl.data(), 2048 * 4, cudaMemcpyHostToDevice));
    int len = 2048, qv = p;
    CK(cudaMalloc(&lens, 4));
    CK(cudaMemcpy(lens, &len, 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&qp, 4));
    CK(cudaMemcpy(qp, &qv, 4, cudaMemcpyHostToDevice));
    float *po, *pml, *out;
    CK(cudaMalloc(&po, 64 * 9 * 512 * 4));
    CK(cudaMalloc(&pml, 64 * 9 * 2 * 4));
    CK(cudaMalloc(&out, 64 * 512 * 4));
    timeit("attn sliding 64h 512 keys", 512.0 * 576 * 4, [&] {
      memra_hy4r_attn(q, kv, nullptr, 0, nullptr, qp, 0, 512, 512, 0, 1, 64, 2, 0.0625f, sinks, po, pml, out, st);
    });
    timeit("attn indexed 32h 2048 keys", 2048.0 * 576 * 4, [&] {
      memra_hy4r_attn(q, kv, list, 2048, lens, qp, 1, 0, 0, 0, 1, 32, 8, 0.0625f, sinks, po, pml, out, st);
    });
    return 0;
  }
  // address-spread check: the o_proj shape over many copies of its weight spread through a large
  // pool (the model reads 12 GB of 87 GB a token), allocated with cudaMalloc or cudaMallocAsync
  if (*only && strstr(only, "spread")) {
    const int n = 3072, k = 16384;
    const size_t wb = (size_t)n * k;
    const int copies = getenv("KB_COPIES") ? atoi(getenv("KB_COPIES")) : 600;
    const bool async = getenv("KB_ASYNC") != nullptr;
    uint8_t* pool;
    if (async) CK(cudaMallocAsync((void**)&pool, wb * copies, st)); else CK(cudaMalloc((void**)&pool, wb * copies));
    CK(cudaMemsetAsync(pool, 0x11, wb * copies, st));
    const int gc = (k + 127) / 128;
    void* sc = dev_rand((size_t)((n + 127) / 128) * gc * 4, 3);
    float* x = dev_x(1, k);
    float* y;
    CK(cudaMalloc(&y, (size_t)n * 4));
    CK(cudaStreamSynchronize(st));
    for (int pass = 0; pass < 2; pass++) {
      CK(cudaEventRecord(e0, st));
      for (int c = 0; c < copies; c++)
        memra_hy4r_gemm(x, k, 0, pool + wb * (size_t)((c * 7919) % copies), 0, sc, 0, nullptr, 2, gc, y, n, 0, 1, n, k, 1,
                        nullptr, 0, 0, 1, 0, st);
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      printf("spread %s copies %d pass %d: %.1f us a GEMV, %.1f GB/s\n", async ? "async" : "malloc", copies, pass,
             ms * 1e3 / copies, (double)wb * copies / (ms * 1e6));
    }
    return 0;
  }
  // grouped NVFP4 experts: 512 experts [n, k], 16 picked, one row each (decode) or t=4 tokens
  if (!*only || strstr("experts", only)) {
    const int E = 512, n = 1536, k = 1536;
    const size_t wbs = (size_t)n * k / 2, scbs = (size_t)n * k / 16;
    void* wr = dev_rand(wbs * E, 0);
    void* w;
    CK(cudaMalloc(&w, wbs * E));
    if (memra_hy4r_tile(wr, n, k / 2, k / 2, (long)wbs, w, (long)wbs, E, st)) fails++;
    CK(cudaFree(wr));
    void* sc = dev_rand(scbs * E, 2);
    float* s2 = (float*)dev_rand(E * 4, 3);
    for (int t : {1, 4}) {
      const int rows = 16 * t;
      // tiles: experts 7*i+3, each with t rows (every token picked all 16, so rows = t)
      const int nexp = getenv("KB_NEXP") ? atoi(getenv("KB_NEXP")) : 16;
      std::vector<int> tl;
      for (int i = 0; i < nexp; i++) {
        tl.push_back(7 * i + 3);
        tl.push_back(i * t);
        tl.push_back(t);
      }
      const int ntiles = nexp + 1;
      tl.push_back(0);
      tl.push_back(0);
      tl.push_back(0);
      int* dt;
      CK(cudaMalloc(&dt, tl.size() * 4));
      CK(cudaMemcpy(dt, tl.data(), tl.size() * 4, cudaMemcpyHostToDevice));
      float* x = dev_x(rows, k);
      float *y0, *y1;
      CK(cudaMalloc(&y0, (size_t)rows * n * 4));
      CK(cudaMalloc(&y1, (size_t)rows * n * 4));
      for (int red = 0; red < 2; red++) {
        int rc0 = memra_hy4r_gemm(x, k, 0, w, wbs, sc, scbs, s2, 3, 0, y0, n, 0, rows, n, k, 1, dt, ntiles, red, 0,
                                  0, st);
        int rc1 = memra_hy4r_gemm(x, k, 0, w, wbs, sc, scbs, s2, 3, 0, y1, n, 0, rows, n, k, 1, dt, ntiles, red, 1,
                                  t, st);
        CK(cudaStreamSynchronize(st));
        char nm[128];
        snprintf(nm, sizeof nm, "experts nvfp4 16x(%dx%d) t%d%s", n, k, t, red ? " red" : "");
        if (rc0 || rc1) {
          printf("%s: rc %d %d\n", nm, rc0, rc1);
          fails++;
          continue;
        }
        if (!same(y0, y1, (size_t)rows * n, nm)) fails++;
        if (!red)
          timeit(nm, (double)nexp * (wbs + scbs), [&] {
            memra_hy4r_gemm(x, k, 0, w, wbs, sc, scbs, s2, 3, 0, y1, n, 0, rows, n, k, 1, dt, ntiles, 0, 1, t, st);
          });
      }
      CK(cudaFree(dt));
      CK(cudaFree(x));
      CK(cudaFree(y0));
      CK(cudaFree(y1));
    }
    CK(cudaFree(w));
    CK(cudaFree(sc));
    CK(cudaFree(s2));
  }
  // F32 islands: router 512x3072, head 120832x3072 (BF16 weights)
  for (int which = 0; which < 2; which++) {
    const int n = which ? 120832 : 512, k = 3072;
    const char* nm0 = which ? "head f32 120832x3072" : "router f32 512x3072";
    if (*only && !strstr(nm0, only)) continue;
    void* w = dev_rand((size_t)n * k * 2, 1);
    for (int m : {1, 4}) {
      float* x = dev_x(m, k);
      float *y0, *y1;
      CK(cudaMalloc(&y0, (size_t)m * n * 4));
      CK(cudaMalloc(&y1, (size_t)m * n * 4));
      int rc0 = memra_hy4f_gemm(x, k, w, nullptr, 0, 1.0f, 0, 0, y0, n, m, n, k, st);
      int rc1 = memra_hy4r_gemm_f32(x, k, w, y1, n, m, n, k, st);
      CK(cudaStreamSynchronize(st));
      char nm[128];
      snprintf(nm, sizeof nm, "%s m%d", nm0, m);
      if (rc0 || rc1) {
        printf("%s: rc %d %d\n", nm, rc0, rc1);
        fails++;
      } else {
        if (!same(y0, y1, (size_t)m * n, nm)) fails++;
        timeit(nm, (double)n * k * 2, [&] { memra_hy4r_gemm_f32(x, k, w, y1, n, m, n, k, st); });
      }
      CK(cudaFree(x));
      CK(cudaFree(y0));
      CK(cudaFree(y1));
    }
    CK(cudaFree(w));
  }
  printf(fails ? "KBENCH FAIL %d\n" : "KBENCH ALL EQUAL\n", fails);
  return fails ? 1 : 0;
}
