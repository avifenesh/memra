// Inkling 2 Mini on one sm_120a card: the kernels of the native GPU route (crate::inkling2_gpu,
// binary ink2-gpu). The program is the typed plan's (memra_gguf::model_plan::inkling2) and the
// reference is memra_reference::inkling2, operation for operation.
//
// Numeric class of this file: activations are f32 on the device between operations. Weights stay
// in the bytes the checkpoint stores (BF16, NVFP4 codes with e4m3 block scales and a per-expert
// amax, MXFP8 values with ue8m0 row scales) and a dequantized weight is never formed and rounded:
// codes multiply the inputs, a block scale multiplies each partial sum and the per-expert factor
// amax / denominator is applied once, in f64, to the finished sum. Reductions that decide a norm,
// a clip, a softmax or a mix run in f64 and round once to f32.
//
// Every kernel computes a row from that row alone and in one fixed order, so a row's bits do not
// depend on how many rows the call holds: a prefill chunk and a decode step are the same program.
//
// The float expert path here (memra_ink2g_gemm with INK2G_W_NVFP4 / INK2G_W_MXFP8 on f32 inputs)
// is the BRING-UP class: it is the oracle step toward the checkpoint's declared act-quant class,
// whose kernels are cu/inkling2_actq.cu. It is never the default.
//
// The tile GEMM is the layout of cu/hy4f_gpu.cu's hy4f_gemm_kernel (perf/hy4flash-decode-20261120),
// kept because its per-output sequential sum is what makes rows independent; the formats, the
// expert tiles, the explicit fused multiply-adds and the f64 epilogue are this program's.
//
// Return codes: 0 ok, 10000 + cudaError_t, 45xxx for a refused contract.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define INK2G_W_BF16 0
#define INK2G_W_NVFP4 1
#define INK2G_W_MXFP8 2
#define INK2G_W_F32 3

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define INK2G_TILE_WORDS (2 + TILE_M)
#define INK2G_MAX_STATES 8
#define INK2G_MAX_EXTENT 1024
// keys one attention tile holds (shared f64 scores)
#define INK2G_ATTN_TILE 1024
#define INK2G_MAX_KERNEL 8

namespace {

__device__ __forceinline__ float ink2g_bf16(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }

// fp32 -> bf16 -> fp32, round to nearest even
__device__ __forceinline__ float ink2g_bf16_round(float x) {
  uint32_t b = __float_as_uint(x);
  b += 0x7fffu + ((b >> 16) & 1u);
  return __uint_as_float(b & 0xffff0000u);
}

// e4m3 to fp32, exactly (the NaN codes are refused at load)
__device__ __forceinline__ float ink2g_e4m3(uint8_t b) {
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  const float mag = e == 0 ? (float)m * 0x1p-9f : (float)(8u + m) * __uint_as_float((e + 117u) << 23);
  return (b & 0x80) ? -mag : mag;
}

__device__ __forceinline__ float ink2g_e2m1(uint8_t nib) {
  const float t[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return t[nib & 15];
}

// Sum of one value per thread over the block, in a fixed order: lanes inside a warp by the shuffle
// tree, warps in ascending order. sh holds one slot per warp.
__device__ __forceinline__ double ink2g_block_sum(double v, double* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  double t = 0.0;
  const int nw = (blockDim.x + 31) >> 5;
  for (int w = 0; w < nw; w++) t += sh[w];
  __syncthreads();
  return t;
}

__device__ __forceinline__ double ink2g_block_max(double v, double* sh) {
  for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  double t = -INFINITY;
  const int nw = (blockDim.x + 31) >> 5;
  for (int w = 0; w < nw; w++) t = fmax(t, sh[w]);
  __syncthreads();
  return t;
}

// ---- tile GEMM ---------------------------------------------------------------------------------
// Y[row, n] = sum_k X[xrow, k] * W[e][n, k]. One 64x64 output tile per block, 16x16 threads, 4x4
// outputs a thread, K 16 at a time. Each output is one thread's sequential fused multiply-add
// chain over k, folded into the accumulator at every 16-input scale boundary.
// tiles == nullptr: rows 0..m of x and y, expert 0. Otherwise blockIdx.y names a record
// {expert, count, pair[64]}: output row = pair, input row = pair / xdiv.
__global__ void ink2g_gemm_kernel(const float* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ w,
                                  const uint8_t* __restrict__ sc, int64_t wstride, int64_t sstride,
                                  const float* __restrict__ amax, float denom, int fmt, int round_bf16,
                                  const uint32_t* __restrict__ tiles, int xdiv, float* __restrict__ y, int64_t ldy,
                                  int m, int n, int k) {
  __shared__ float xs[TILE_K][TILE_M + 1];
  __shared__ float wt[TILE_K][TILE_N + 1];
  __shared__ float ws[TILE_N];
  __shared__ int rx[TILE_M];
  __shared__ int ry[TILE_M];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  const int n0 = blockIdx.x * TILE_N;
  int expert = 0;
  if (tiles) {
    const uint32_t* rec = tiles + (int64_t)blockIdx.y * INK2G_TILE_WORDS;
    expert = (int)rec[0];
    const int count = (int)rec[1];
    if (tid < TILE_M) {
      const int p = tid < count ? (int)rec[2 + tid] : -1;
      ry[tid] = p;
      rx[tid] = p < 0 ? -1 : p / xdiv;
    }
  } else if (tid < TILE_M) {
    const int gm = blockIdx.y * TILE_M + tid;
    rx[tid] = ry[tid] = gm < m ? gm : -1;
  }
  __syncthreads();
  const uint8_t* we = w + (int64_t)expert * wstride;
  const uint8_t* se = sc ? sc + (int64_t)expert * sstride : nullptr;
  const double scale2 = fmt == INK2G_W_NVFP4 ? (double)amax[expert] : 1.0;
  const bool plain = fmt == INK2G_W_BF16 || fmt == INK2G_W_F32 || round_bf16;
  float acc[4][4], part[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = part[i][j] = 0.0f;

  for (int k0 = 0; k0 < k; k0 += TILE_K) {
    for (int e = tid; e < TILE_M * TILE_K; e += 256) {
      const int r = e / TILE_K, c = e % TILE_K;
      const int gm = rx[r], gk = k0 + c;
      xs[c][r] = (gm >= 0 && gk < k) ? x[(int64_t)gm * ldx + gk] : 0.0f;
    }
    for (int e = tid; e < TILE_N * TILE_K; e += 256) {
      const int r = e / TILE_K, c = e % TILE_K;
      const int gn = n0 + r, gk = k0 + c;
      float v = 0.0f;
      if (gn < n && gk < k) {
        if (fmt == INK2G_W_BF16) {
          v = ink2g_bf16(((const uint16_t*)we)[(int64_t)gn * k + gk]);
        } else if (fmt == INK2G_W_F32) {
          v = ((const float*)we)[(int64_t)gn * k + gk];
        } else if (fmt == INK2G_W_MXFP8) {
          v = ink2g_e4m3(we[(int64_t)gn * k + gk]);
        } else {
          const uint8_t b = we[(int64_t)gn * (k / 2) + gk / 2];
          v = ink2g_e2m1((gk & 1) ? (b >> 4) : b);
        }
        if (round_bf16 && fmt == INK2G_W_NVFP4) {
          // RED ARM: the weight a BF16 loader holds, bf16(code * scale * amax / denominator)
          const float cs = v * ink2g_e4m3(se[(int64_t)gn * (k / 16) + gk / 16]);
          v = ink2g_bf16_round((float)((double)cs * scale2 / (double)denom));
        } else if (round_bf16 && fmt == INK2G_W_MXFP8) {
          v = ink2g_bf16_round(scalbnf(v, (int)se[(int64_t)gn * (k / 32) + gk / 32] - 127));
        }
      }
      wt[c][r] = v;
    }
    if (tid < TILE_N) {
      const int gn = n0 + tid;
      float s = 1.0f;
      if (gn < n && !plain) {
        if (fmt == INK2G_W_NVFP4) {
          s = ink2g_e4m3(se[(int64_t)gn * (k / 16) + k0 / 16]);
        } else {
          s = scalbnf(1.0f, (int)se[(int64_t)gn * (k / 32) + k0 / 32] - 127);
        }
      }
      ws[tid] = s;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE_K; kk++) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) a[i] = xs[kk][ty * 4 + i];
#pragma unroll
      for (int j = 0; j < 4; j++) b[j] = wt[kk][tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) part[i][j] = fmaf(a[i], b[j], part[i][j]);
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float s = ws[tx * 4 + j];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        acc[i][j] = plain ? acc[i][j] + part[i][j] : fmaf(part[i][j], s, acc[i][j]);
        part[i][j] = 0.0f;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int gm = ry[ty * 4 + i];
    if (gm < 0) continue;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int gn = n0 + tx * 4 + j;
      if (gn >= n) continue;
      float v = acc[i][j];
      if (fmt == INK2G_W_NVFP4 && !round_bf16) v = (float)((double)v * scale2 / (double)denom);
      y[(int64_t)gm * ldy + gn] = v;
    }
  }
}

// ---- decode GEMV -------------------------------------------------------------------------------
// The same function as ink2g_gemm_kernel for calls of one input row per tile (a decode step): one
// thread per output, the same fused multiply-add chain over k and the same folds, so its bits equal
// the tile kernel's. What changes is the work: no 64-row tile is staged for one row, and each
// thread streams its own weight row once. blockIdx.y names the row (tiles == nullptr: row 0) or a
// tile record of count 1.
#define INK2G_GEMV_MAX_K 4096
__global__ void ink2g_gemv_kernel(const float* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ w,
                                  const uint8_t* __restrict__ sc, int64_t wstride, int64_t sstride,
                                  const float* __restrict__ amax, float denom, int fmt, int round_bf16,
                                  const uint32_t* __restrict__ tiles, int xdiv, float* __restrict__ y, int64_t ldy,
                                  int n, int k) {
  __shared__ float xs[INK2G_GEMV_MAX_K];
  int expert = 0, xrow = 0, yrow = 0;
  if (tiles) {
    const uint32_t* rec = tiles + (int64_t)blockIdx.y * INK2G_TILE_WORDS;
    expert = (int)rec[0];
    yrow = (int)rec[2];
    xrow = yrow / xdiv;
  }
  for (int i = threadIdx.x; i < k; i += blockDim.x) xs[i] = x[(int64_t)xrow * ldx + i];
  __syncthreads();
  const int gn = blockIdx.x * blockDim.x + threadIdx.x;
  if (gn >= n) return;
  const uint8_t* we = w + (int64_t)expert * wstride;
  const uint8_t* se = sc ? sc + (int64_t)expert * sstride : nullptr;
  const double scale2 = fmt == INK2G_W_NVFP4 ? (double)amax[expert] : 1.0;
  const bool plain = fmt == INK2G_W_BF16 || fmt == INK2G_W_F32 || round_bf16;
  float acc = 0.0f;
  for (int k0 = 0; k0 < k; k0 += TILE_K) {
    float part = 0.0f;
    float s = 1.0f;
    if (fmt == INK2G_W_NVFP4) {
      s = ink2g_e4m3(se[(int64_t)gn * (k / 16) + k0 / 16]);
    } else if (fmt == INK2G_W_MXFP8) {
      s = scalbnf(1.0f, (int)se[(int64_t)gn * (k / 32) + k0 / 32] - 127);
    }
#pragma unroll
    for (int kk = 0; kk < TILE_K; kk++) {
      const int gk = k0 + kk;
      float v = 0.0f;
      if (gk < k) {
        if (fmt == INK2G_W_BF16) {
          v = ink2g_bf16(((const uint16_t*)we)[(int64_t)gn * k + gk]);
        } else if (fmt == INK2G_W_F32) {
          v = ((const float*)we)[(int64_t)gn * k + gk];
        } else if (fmt == INK2G_W_MXFP8) {
          v = ink2g_e4m3(we[(int64_t)gn * k + gk]);
        } else {
          const uint8_t b = we[(int64_t)gn * (k / 2) + gk / 2];
          v = ink2g_e2m1((gk & 1) ? (b >> 4) : b);
        }
        if (round_bf16 && fmt == INK2G_W_NVFP4) {
          v = ink2g_bf16_round((float)((double)(v * s) * scale2 / (double)denom));
        } else if (round_bf16 && fmt == INK2G_W_MXFP8) {
          v = ink2g_bf16_round(v * s);
        }
      }
      part = fmaf(gk < k ? xs[gk] : 0.0f, v, part);
    }
    acc = plain ? acc + part : fmaf(part, s, acc);
  }
  if (fmt == INK2G_W_NVFP4 && !round_bf16) acc = (float)((double)acc * scale2 / (double)denom);
  y[(int64_t)yrow * ldy + gn] = acc;
}

// ---- decode GEMV, BF16, at the read bandwidth --------------------------------------------------
// The same function as ink2g_gemv_kernel (and the tile GEMM) for BF16 weights: for each output,
// per 16-input group, a fused multiply-add chain from 0 in input order, and the groups' partials
// added from 0 in group order. Its bits equal theirs. What changes is who computes what: a lane
// computes one group's partial from a 32-byte slice of the row, so a warp reads a weight row
// with coalesced 16-byte loads, and one thread per row adds the row's partials in order after.
// Several matrices that read inputs of the same width run in one launch (one job each).
#define INK2G_JOBS 8
#define INK2G_GV_ROWS 32
#define INK2G_GV_THREADS 256
#define INK2G_GV_MAX_K 4096
struct Ink2gJobs {
  const float* x[INK2G_JOBS];
  const uint16_t* w[INK2G_JOBS];
  float* y[INK2G_JOBS];
  int n[INK2G_JOBS];
  int first[INK2G_JOBS + 1];  // the first block of each job; first[count] = the grid
  int count;
};

template <int ITERS>
__global__ void __launch_bounds__(INK2G_GV_THREADS) ink2g_gemv_bf16_kernel(const Ink2gJobs jobs, int k) {
  extern __shared__ float ink2g_gv[];
  const int groups = k >> 4, pstride = groups + 1;
  float* xs = ink2g_gv;
  float* ps = ink2g_gv + k;
  int j = 0;
  while (j + 1 < jobs.count && (int)blockIdx.x >= jobs.first[j + 1]) j++;
  const int r0 = ((int)blockIdx.x - jobs.first[j]) * INK2G_GV_ROWS;
  const int n = jobs.n[j];
  const float* x = jobs.x[j];
  for (int i = threadIdx.x; i < k; i += INK2G_GV_THREADS) xs[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int rl = warp; rl < INK2G_GV_ROWS; rl += INK2G_GV_THREADS / 32) {
    const int row = r0 + rl;
    if (row >= n) break;
    const uint4* wr = reinterpret_cast<const uint4*>(jobs.w[j] + (int64_t)row * k);
    uint4 lo[ITERS], hi[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; it++) {
      const int g = it * 32 + lane;
      if (g < groups) {
        lo[it] = __ldcs(wr + 2 * g);
        hi[it] = __ldcs(wr + 2 * g + 1);
      }
    }
#pragma unroll
    for (int it = 0; it < ITERS; it++) {
      const int g = it * 32 + lane;
      if (g < groups) {
        const uint32_t wv[8] = {lo[it].x, lo[it].y, lo[it].z, lo[it].w, hi[it].x, hi[it].y, hi[it].z, hi[it].w};
        const float* xg = xs + g * 16;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          part = fmaf(xg[2 * i], __uint_as_float(wv[i] << 16), part);
          part = fmaf(xg[2 * i + 1], __uint_as_float(wv[i] & 0xffff0000u), part);
        }
        ps[rl * pstride + g] = part;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < INK2G_GV_ROWS) {
    const int row = r0 + threadIdx.x;
    if (row < n) {
      const float* p = ps + threadIdx.x * pstride;
      float acc = 0.0f;
      for (int g = 0; g < groups; g++) acc = __fadd_rn(acc, p[g]);
      jobs.y[j][row] = acc;
    }
  }
}

// ---- embedding ---------------------------------------------------------------------------------
// out[r] = norm_w * e * rsqrt(mean(e^2) + eps), e the BF16 row ids[r] of the table.
__global__ void ink2g_embed_kernel(const uint16_t* __restrict__ table, const uint32_t* __restrict__ ids,
                                   const float* __restrict__ nw, double eps, float* __restrict__ out, int H) {
  __shared__ double sh[8];
  const int r = blockIdx.x;
  const uint16_t* row = table + (int64_t)ids[r] * H;
  double ss = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const double v = (double)ink2g_bf16(row[j]);
    ss += v * v;
  }
  ss = ink2g_block_sum(ss, sh);
  const double f = 1.0 / sqrt(ss / (double)H + eps);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    out[(int64_t)r * H + j] = (float)((double)nw[j] * ((double)ink2g_bf16(row[j]) * f));
}

// y[r] = w * x[r] * rsqrt(mean(x[r]^2) + eps)
__global__ void ink2g_norm_kernel(const float* __restrict__ x, const float* __restrict__ nw, double eps,
                                  float* __restrict__ y, int H) {
  __shared__ double sh[8];
  const int r = blockIdx.x;
  const float* row = x + (int64_t)r * H;
  double ss = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const double v = (double)row[j];
    ss += v * v;
  }
  ss = ink2g_block_sum(ss, sh);
  const double f = 1.0 / sqrt(ss / (double)H + eps);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    y[(int64_t)r * H + j] = (float)((double)nw[j] * ((double)row[j] * f));
}

// ---- depth mix ---------------------------------------------------------------------------------
struct Ink2gStates {
  const float* p[INK2G_MAX_STATES];
};

// For each query q: softmax(q . rms(s) / sqrt(H)) over the states, summing the RMS-normalized
// states. out[q] is [R, H] at out + q * out_stride.
__global__ void ink2g_mix_kernel(Ink2gStates st, int ns, const float* __restrict__ queries, int nq, double eps,
                                 float* __restrict__ out, int64_t out_stride, int H) {
  __shared__ double sh[8];
  __shared__ double inv[INK2G_MAX_STATES];
  __shared__ double prob[INK2G_MAX_STATES];
  const int r = blockIdx.x;
  for (int s = 0; s < ns; s++) {
    const float* row = st.p[s] + (int64_t)r * H;
    double ss = 0.0;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
      const double v = (double)row[j];
      ss += v * v;
    }
    ss = ink2g_block_sum(ss, sh);
    if (threadIdx.x == 0) inv[s] = 1.0 / sqrt(ss / (double)H + eps);
    __syncthreads();
  }
  const double scale = 1.0 / sqrt((double)H);
  for (int q = 0; q < nq; q++) {
    const float* qr = queries + (int64_t)q * H;
    double top = -INFINITY;
    for (int s = 0; s < ns; s++) {
      const float* row = st.p[s] + (int64_t)r * H;
      double d = 0.0;
      for (int j = threadIdx.x; j < H; j += blockDim.x) d += (double)qr[j] * ((double)row[j] * inv[s]);
      d = ink2g_block_sum(d, sh) * scale;
      if (threadIdx.x == 0) prob[s] = d;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      for (int s = 0; s < ns; s++) top = fmax(top, prob[s]);
      double den = 0.0;
      for (int s = 0; s < ns; s++) {
        prob[s] = exp(prob[s] - top);
        den += prob[s];
      }
      for (int s = 0; s < ns; s++) prob[s] /= den;
    }
    __syncthreads();
    float* o = out + (int64_t)q * out_stride + (int64_t)r * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
      double a = 0.0;
      for (int s = 0; s < ns; s++) a += prob[s] * ((double)st.p[s][(int64_t)r * H + j] * inv[s]);
      o[j] = (float)a;
    }
    __syncthreads();
  }
}

// ---- loop controller ---------------------------------------------------------------------------
// g = down . cat(rms(x), rms(m)); out = (1 - tanh(forget . g)) * x + tanh(gain . g) * m.
// out may be x.
__global__ void ink2g_controller_kernel(const float* x, const float* __restrict__ msg,
                                        const uint16_t* __restrict__ down, const uint16_t* __restrict__ forget,
                                        const uint16_t* __restrict__ gain, double eps, float* out, int H, int rank) {
  __shared__ double sh[8];
  __shared__ double g[128];
  const int r = blockIdx.x;
  const float* xr = x + (int64_t)r * H;
  const float* mr = msg + (int64_t)r * H;
  double sx = 0.0, sm = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    sx += (double)xr[j] * (double)xr[j];
    sm += (double)mr[j] * (double)mr[j];
  }
  sx = ink2g_block_sum(sx, sh);
  sm = ink2g_block_sum(sm, sh);
  const double fx = 1.0 / sqrt(sx / (double)H + eps), fm = 1.0 / sqrt(sm / (double)H + eps);
  for (int o = 0; o < rank; o++) {
    const uint16_t* dr = down + (int64_t)o * 2 * H;
    double d = 0.0;
    for (int j = threadIdx.x; j < H; j += blockDim.x)
      d += (double)ink2g_bf16(dr[j]) * ((double)xr[j] * fx) + (double)ink2g_bf16(dr[H + j]) * ((double)mr[j] * fm);
    d = ink2g_block_sum(d, sh);
    if (threadIdx.x == 0) g[o] = d;
    __syncthreads();
  }
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    double f = 0.0, a = 0.0;
    for (int o = 0; o < rank; o++) {
      f += (double)ink2g_bf16(forget[(int64_t)j * rank + o]) * g[o];
      a += (double)ink2g_bf16(gain[(int64_t)j * rank + o]) * g[o];
    }
    out[(int64_t)r * H + j] = (float)((1.0 - tanh(f)) * (double)xr[j] + tanh(a) * (double)mr[j]);
  }
}

// ---- loop controller, split over blocks ----------------------------------------------------------
// The same function as ink2g_controller_kernel, with the same per-thread orders and the same block
// reductions (256 threads, ink2g_block_sum), so its bits equal it. Phase 1 runs one block per
// (rank output, row): each block repeats the row's two RMS sums and forms its own g[o] into
// scratch. Phase 2 runs the forget / gain projections with one thread per output channel.
__global__ void __launch_bounds__(256) ink2g_ctrl_down_kernel(const float* __restrict__ x, const float* __restrict__ msg,
                                                             const uint16_t* __restrict__ down, double eps,
                                                             double* __restrict__ g, int H, int rank) {
  __shared__ double sh[8];
  const int o = blockIdx.x, r = blockIdx.y;
  const float* xr = x + (int64_t)r * H;
  const float* mr = msg + (int64_t)r * H;
  double sx = 0.0, sm = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    sx += (double)xr[j] * (double)xr[j];
    sm += (double)mr[j] * (double)mr[j];
  }
  sx = ink2g_block_sum(sx, sh);
  sm = ink2g_block_sum(sm, sh);
  const double fx = 1.0 / sqrt(sx / (double)H + eps), fm = 1.0 / sqrt(sm / (double)H + eps);
  const uint16_t* dr = down + (int64_t)o * 2 * H;
  double d = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    d += (double)ink2g_bf16(dr[j]) * ((double)xr[j] * fx) + (double)ink2g_bf16(dr[H + j]) * ((double)mr[j] * fm);
  d = ink2g_block_sum(d, sh);
  if (threadIdx.x == 0) g[(int64_t)r * rank + o] = d;
}

__global__ void ink2g_ctrl_up_kernel(const float* x, const float* __restrict__ msg,
                                     const uint16_t* __restrict__ forget, const uint16_t* __restrict__ gain,
                                     const double* __restrict__ gin, float* out, int H, int rank) {
  __shared__ double g[128];
  const int r = blockIdx.y;
  for (int o = threadIdx.x; o < rank; o += blockDim.x) g[o] = gin[(int64_t)r * rank + o];
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const float* xr = x + (int64_t)r * H;
  const float* mr = msg + (int64_t)r * H;
  double f = 0.0, a = 0.0;
  for (int o = 0; o < rank; o++) {
    f += (double)ink2g_bf16(forget[(int64_t)j * rank + o]) * g[o];
    a += (double)ink2g_bf16(gain[(int64_t)j * rank + o]) * g[o];
  }
  out[(int64_t)r * H + j] = (float)((1.0 - tanh(f)) * (double)xr[j] + tanh(a) * (double)mr[j]);
}

// ---- depth mix, split over blocks ----------------------------------------------------------------
// The same function as ink2g_mix_kernel, the same orders and block reductions, so the same bits.
// Phase 1: one block per (state, query, row) forms the state's RMS factor and the scaled score
// into scratch [row][query][state][2]. Phase 2: one thread per output channel; each block repeats
// the softmax over the states on thread 0.
__global__ void __launch_bounds__(256) ink2g_mix_dot_kernel(Ink2gStates st, int ns, const float* __restrict__ queries,
                                                           int nq, double eps, double* __restrict__ scr, int H) {
  __shared__ double sh[8];
  const int s = blockIdx.x, q = blockIdx.y, r = blockIdx.z;
  const float* row = st.p[s] + (int64_t)r * H;
  double ss = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const double v = (double)row[j];
    ss += v * v;
  }
  ss = ink2g_block_sum(ss, sh);
  const double inv = 1.0 / sqrt(ss / (double)H + eps);
  const double scale = 1.0 / sqrt((double)H);
  const float* qr = queries + (int64_t)q * H;
  double d = 0.0;
  for (int j = threadIdx.x; j < H; j += blockDim.x) d += (double)qr[j] * ((double)row[j] * inv);
  d = ink2g_block_sum(d, sh) * scale;
  if (threadIdx.x == 0) {
    double* o = scr + (((int64_t)r * nq + q) * ns + s) * 2;
    o[0] = d;
    o[1] = inv;
  }
}

__global__ void ink2g_mix_out_kernel(Ink2gStates st, int ns, int nq, const double* __restrict__ scr,
                                     float* __restrict__ out, int64_t out_stride, int H) {
  __shared__ double inv[INK2G_MAX_STATES];
  __shared__ double prob[INK2G_MAX_STATES];
  const int q = blockIdx.y, r = blockIdx.z;
  if (threadIdx.x == 0) {
    const double* p = scr + ((int64_t)r * nq + q) * ns * 2;
    for (int s = 0; s < ns; s++) {
      prob[s] = p[2 * s];
      inv[s] = p[2 * s + 1];
    }
    double top = -INFINITY;
    for (int s = 0; s < ns; s++) top = fmax(top, prob[s]);
    double den = 0.0;
    for (int s = 0; s < ns; s++) {
      prob[s] = exp(prob[s] - top);
      den += prob[s];
    }
    for (int s = 0; s < ns; s++) prob[s] /= den;
  }
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  double a = 0.0;
  for (int s = 0; s < ns; s++) a += prob[s] * ((double)st.p[s][(int64_t)r * H + j] * inv[s]);
  out[(int64_t)q * out_stride + (int64_t)r * H + j] = (float)a;
}

// ---- head epilogues: clip / tau / value norm ------------------------------------------------------
// One block per (row, head). mode 0: v = x * scale (optional); v *= min(1, thr / (|v| + eps)); v *= tau.
// mode 1: v = x * rsqrt(mean(x^2) + eps). tau = 1 + alpha * ln(max(1, (pos + 1) / floor)) when
// log_floor > 0. The row lands at dst row (pos0 + r) % cap when cap > 0 (a cache ring), else r.
// dst may be x (a block reads its head before it writes it).
__global__ void ink2g_heads_kernel(int mode, const float* x, const float* __restrict__ scale, double thr,
                                   double eps, int64_t pos0, int log_floor, double alpha, float* dst,
                                   int64_t cap, int heads, int width) {
  __shared__ double sh[8];
  const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int64_t pos = pos0 + r;
  const int64_t drow = cap > 0 ? pos % cap : (int64_t)r;
  double v = 0.0;
  if (t < width) {
    v = (double)x[((int64_t)r * heads + h) * width + t];
    if (scale) v *= (double)scale[h * width + t];
  }
  const double ss = ink2g_block_sum(v * v, sh);
  double f;
  if (mode == 0) {
    f = fmin(thr / (sqrt(ss) + eps), 1.0);
    if (log_floor > 0) f *= 1.0 + alpha * log(fmax((double)(pos + 1) / (double)log_floor, 1.0));
  } else {
    f = 1.0 / sqrt(ss / (double)width + eps);
  }
  if (t < width) dst[(drow * heads + h) * width + t] = (float)(v * f);
}

// ---- short convolution with carried state --------------------------------------------------------
// One thread owns one channel and walks the rows in order, so the state it reads is the state it
// wrote. state [K - 1, C], oldest first; inputs before position 0 are zero.
// mode 0 (static): taps [C, K], oldest input first; out = (x + sum) / sqrt(K + 1).
// mode 1 (dynamic): taps [K, C] and dyn [R, groups, K], both by lag with the current token first;
//   out = (x + sum_lag (taps[lag, c] + dyn[r, group(c), lag]) * x[t - lag]) / sqrt(2K + 1).
// add (nullable): out = add + conv. carry == 0 leaves the state untouched (the gate's red arm).
__global__ void ink2g_conv_kernel(int mode, const float* __restrict__ x, const float* __restrict__ state_in,
                                  float* __restrict__ state_out, const float* __restrict__ taps,
                                  const float* __restrict__ dyn, int groups, const float* add, float* out, int R,
                                  int C, int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  double hist[INK2G_MAX_KERNEL];  // hist[l] = input l steps back, l in 1..K-1
  for (int l = 1; l < K; l++) hist[l] = (double)state_in[(int64_t)(K - 1 - l) * C + c];
  double tp[INK2G_MAX_KERNEL];
  for (int j = 0; j < K; j++) tp[j] = mode == 0 ? (double)taps[(int64_t)c * K + j] : (double)taps[(int64_t)j * C + c];
  const int group = mode == 1 ? c / (C / groups) : 0;
  const double div = sqrt(mode == 0 ? (double)(K + 1) : (double)(2 * K + 1));
  for (int r = 0; r < R; r++) {
    const double cur = (double)x[(int64_t)r * C + c];
    double sum = 0.0;
    if (mode == 0) {
      for (int tap = 0; tap < K; tap++) {
        const int lag = K - 1 - tap;
        sum += tp[tap] * (lag == 0 ? cur : hist[lag]);
      }
    } else {
      const float* dr = dyn + ((int64_t)r * groups + group) * K;
      for (int lag = 0; lag < K; lag++) sum += (lag == 0 ? cur : hist[lag]) * (tp[lag] + (double)dr[lag]);
    }
    double o = (sum + cur) / div;
    if (add) o = (double)add[(int64_t)r * C + c] + o;
    out[(int64_t)r * C + c] = (float)o;
    for (int l = K - 1; l > 1; l--) hist[l] = hist[l - 1];
    hist[1] = cur;
  }
  if (state_out)
    for (int l = 1; l < K; l++) state_out[(int64_t)(K - 1 - l) * C + c] = (float)hist[l];
}

// ---- relative-bias attention with the per-query sink ---------------------------------------------
// One block per (row, head). The query at position pos = pos0 + row attends the keys at distance
// d = 0 .. nkeys - 1 back (nkeys = pos + 1, or min(pos + 1, window)); the key at position p is cache
// row p % cap. score = q . k / sqrt(dim) + bias[d], bias[d] = rel . table[:, d] for d < extent; a
// key at d >= extent is masked or takes a zero bias. The sink is one more logit of the softmax
// denominator and has no value: p_d = exp(s_d) / (exp(sink) + sum exp(s)). It is never folded into
// the bias, so every attended key, the zero-bias ones included, sits under it.
// Scores are f64 and never stored per key: pass 1 takes the maximum, pass 2 the denominator, pass 3
// recomputes each score of a tile of INK2G_ATTN_TILE keys into shared memory and folds the tile
// into the output, oldest key first. Every score is the same sequential f64 dot in every pass, so
// the result is the one a stored-score program computes, with shared memory that does not grow
// with the context.
__device__ __forceinline__ double ink2g_attn_score(const float* qs, const float* kc, int64_t cap, int64_t pos,
                                                   int64_t d, int kvw, int kvh, int dim, int extent,
                                                   int mask_beyond, const double* bias, double scale) {
  if (d >= extent && mask_beyond) return -INFINITY;
  const float* kr = kc + ((pos - d) % cap) * kvw + kvh * dim;
  double dot = 0.0;
  for (int i = 0; i < dim; i++) dot += (double)qs[i] * (double)kr[i];
  return dot * scale + (d < extent ? bias[d] : 0.0);
}

__global__ void ink2g_attn_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                  const float* __restrict__ table, const float* __restrict__ sink,
                                  const float* __restrict__ kc, const float* __restrict__ vc, int64_t cap, int64_t pos0,
                                  int window, int extent, int mask_beyond, float* __restrict__ out, int heads,
                                  int kv_heads, int dim, int rank) {
  __shared__ double sh[8];
  __shared__ double bias[INK2G_MAX_EXTENT];
  __shared__ double tile[INK2G_ATTN_TILE];
  __shared__ float qs[256];
  const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x, nt = blockDim.x;
  const int64_t pos = pos0 + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int kvh = h / (heads / kv_heads);
  const int kvw = kv_heads * dim;
  const float* relr = rel + ((int64_t)r * heads + h) * rank;
  for (int d = t; d < extent && d < nkeys; d += nt) {
    double b = 0.0;
    for (int i = 0; i < rank; i++) b += (double)relr[i] * (double)table[(int64_t)i * extent + d];
    bias[d] = b;
  }
  if (t < dim) qs[t] = q[((int64_t)r * heads + h) * dim + t];
  __syncthreads();
  const double scale = 1.0 / sqrt((double)dim);
  const double sk = (double)sink[(int64_t)r * heads + h];
  double top = -INFINITY;
  for (int64_t d = t; d < nkeys; d += nt)
    top = fmax(top, ink2g_attn_score(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale));
  top = fmax(ink2g_block_max(top, sh), sk);
  double den = 0.0;
  for (int64_t d = t; d < nkeys; d += nt)
    den += exp(ink2g_attn_score(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale) - top);
  den = ink2g_block_sum(den, sh) + exp(sk - top);
  double a = 0.0;
  for (int64_t hi = nkeys - 1; hi >= 0; hi -= INK2G_ATTN_TILE) {
    const int64_t lo = hi - INK2G_ATTN_TILE + 1 > 0 ? hi - INK2G_ATTN_TILE + 1 : 0;
    __syncthreads();  // the previous tile is folded
    for (int64_t d = lo + t; d <= hi; d += nt)
      tile[d - lo] =
          exp(ink2g_attn_score(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale) - top);
    __syncthreads();
    if (t < dim)
      for (int64_t d = hi; d >= lo; d--)  // oldest key first, as the reference sums
        a += tile[d - lo] * (double)vc[((pos - d) % cap) * kvw + kvh * dim + t];
  }
  if (t < dim) out[((int64_t)r * heads + h) * dim + t] = (float)(a / den);
}


// y *= sigmoid(g)
__global__ void ink2g_gate_kernel(float* __restrict__ y, const float* __restrict__ g, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)((double)y[i] * (1.0 / (1.0 + exp(-(double)g[i]))));
}

// ---- router ------------------------------------------------------------------------------------
// One block per row. score = sigmoid(logit); the top_k experts by score + bias, a tie going to the
// lower expert (tie_high: the gate's red arm, to the higher), in rank order; weight =
// score / (sum of the selected scores) * scale. margin (nullable) = key of rank top_k - 1 minus key
// of rank top_k.
__global__ void ink2g_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, double scale,
                                   int top_k, int tie_high, uint32_t* __restrict__ ids, float* __restrict__ wts,
                                   float* __restrict__ margin, int E) {
  __shared__ double key[1024];
  __shared__ double score[1024];
  __shared__ double sel[64];
  __shared__ double edge[2];
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const double s = 1.0 / (1.0 + exp(-(double)logits[(int64_t)r * E + i]));
    score[i] = s;
    key[i] = bias ? s + (double)bias[i] : s;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    int rank = 0;
    const double ki = key[i];
    for (int j = 0; j < E; j++) {
      const double kj = key[j];
      const bool first = tie_high ? j > i : j < i;
      rank += (kj > ki || (kj == ki && first)) ? 1 : 0;
    }
    if (rank < top_k) {
      ids[(int64_t)r * top_k + rank] = (uint32_t)i;
      sel[rank] = score[i];
    }
    if (rank == top_k - 1) edge[0] = ki;
    if (rank == top_k) edge[1] = ki;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double total = 0.0;
    for (int j = 0; j < top_k; j++) total += sel[j];
    for (int j = 0; j < top_k; j++) wts[(int64_t)r * top_k + j] = (float)(sel[j] / total * scale);
    if (margin) margin[r] = E > top_k ? (float)(edge[0] - edge[1]) : 0.0f;
  }
}

// act[p, j] = softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap) (* wts[p]) over interleaved
// (g, u) = gu[p, 2j], gu[p, 2j + 1]. softcap(x, c) = c * tanh(x / c); the sigmoid reads the raw gate.
__global__ void ink2g_swiglu_kernel(const float* __restrict__ gu, const float* __restrict__ wts, double gate_cap,
                                    double up_cap, float* __restrict__ act, int64_t n, int F) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double g = (double)gu[2 * i], u = (double)gu[2 * i + 1];
  double v = (tanh(g / gate_cap) * gate_cap) * (1.0 / (1.0 + exp(-g))) * (tanh(u / up_cap) * up_cap);
  if (wts) v *= (double)wts[i / F];
  act[i] = (float)v;
}

// out[r, j] = sum over the row's top_k routed rows, in rank order, of dn[r * top_k + s, j] (* wts).
__global__ void ink2g_sum_kernel(const float* __restrict__ dn, const float* __restrict__ wts,
                                 float* __restrict__ out, int64_t n, int top_k, int H) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / H, j = i % H;
  double a = 0.0;
  for (int s = 0; s < top_k; s++) {
    const double v = (double)dn[(r * top_k + s) * H + j];
    a += wts ? v * (double)wts[r * top_k + s] : v;
  }
  out[i] = (float)a;
}

// The declared class's combine: the routed bf16 rows summed in fp32 in rank order, rounded once to
// bf16 (memra_reference::inkling2_actquant::combine).
__global__ void ink2g_sum_bf16_kernel(const uint16_t* __restrict__ dn, const float* __restrict__ wts,
                                      float* __restrict__ out, int64_t n, int top_k, int H) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t r = i / H, j = i % H;
  float a = 0.0f;
  for (int s = 0; s < top_k; s++) {
    const float v = ink2g_bf16(dn[(r * top_k + s) * H + j]);
    a = __fadd_rn(a, wts ? __fmul_rn(v, wts[r * top_k + s]) : v);  // wts: the gate's red arm only
  }
  out[i] = ink2g_bf16_round(a);
}

// y[p, j] = bf16(x[p / rep, j]) for p < rows * rep: the declared class's MoE input, one copy per
// routed pair. The act-quant GEMM indexes its input row and its output row with the same id, and
// lane D's quantizer writes the row it reads (Rows selects, it does not gather), so each routed pair
// owns a quantized copy of its token row. The copies are bit-identical (one row, one quantizer).
__global__ void ink2g_to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n, int H,
                                     int rep) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t p = i / H, j = i % H;
  y[i] = (uint16_t)(__float_as_uint(ink2g_bf16_round(x[(p / rep) * H + j])) >> 16);
}

// logits = cap * tanh(logits / cap)
__global__ void ink2g_softcap_kernel(float* __restrict__ y, double cap, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)(tanh((double)y[i] / cap) * cap);
}

// The first of the equal maxima of each row, as torch.argmax.
__global__ void ink2g_argmax_kernel(const float* __restrict__ y, uint32_t* __restrict__ best, int n) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  const int r = blockIdx.x;
  const float* row = y + (int64_t)r * n;
  float v = -INFINITY;
  int at = 0x7fffffff;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const float c = row[j];
    if (c > v || (c == v && j < at)) {
      v = c;
      at = j;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, v, o);
    const int oi = __shfl_down_sync(0xffffffffu, at, o);
    if (ov > v || (ov == v && oi < at)) {
      v = ov;
      at = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    bv[threadIdx.x >> 5] = v;
    bi[threadIdx.x >> 5] = at;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    for (int w = 1; w < nw; w++)
      if (bv[w] > v || (bv[w] == v && bi[w] < at)) {
        v = bv[w];
        at = bi[w];
      }
    best[r] = (uint32_t)at;
  }
}

// The decode step's expert tiles, built on the device from the router's picks: one tile per routed
// row (count 1), so a decode step needs no host round trip between the router and the experts.
__global__ void ink2g_tiles_kernel(const uint32_t* __restrict__ ids, uint32_t* __restrict__ tiles, int pairs) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  uint32_t* rec = tiles + (int64_t)p * INK2G_TILE_WORDS;
  rec[0] = ids[p];
  rec[1] = 1;
  rec[2] = (uint32_t)p;
}

// Same, for the act-quant expert GEMM's group records {expert, count, row[8]} (nb = 1). The GEMM
// reads its act row and writes its out row at the same index, so a record names the routed pair.
__global__ void ink2g_groups_kernel(const uint32_t* __restrict__ ids, uint32_t* __restrict__ groups, int pairs) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pairs) return;
  uint32_t* rec = groups + (int64_t)p * 10;
  rec[0] = ids[p];
  rec[1] = 1;
  rec[2] = (uint32_t)p;
  for (int i = 3; i < 10; i++) rec[i] = 0;
}

int ink2g_rc() {
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

// Launch the BF16 decode GEMV over jobs. 0 when launched, -1 when a job does not fit its form
// (the caller then runs the generic GEMV), else an error code.
int ink2g_gemv_jobs(const float* const* xs, const void* const* ws, float* const* ys, const int* ns, int count, int k,
                    cudaStream_t stream) {
  if (count <= 0 || count > INK2G_JOBS || k <= 0 || (k % 16) != 0 || k > INK2G_GV_MAX_K) return -1;
  Ink2gJobs jobs;
  int blocks = 0;
  for (int j = 0; j < INK2G_JOBS; j++) {
    const bool on = j < count;
    jobs.x[j] = on ? xs[j] : nullptr;
    jobs.w[j] = on ? (const uint16_t*)ws[j] : nullptr;
    jobs.y[j] = on ? ys[j] : nullptr;
    jobs.n[j] = on ? ns[j] : 0;
    jobs.first[j] = blocks;
    if (on) {
      if (!xs[j] || !ws[j] || !ys[j] || ns[j] <= 0 || ((uintptr_t)ws[j] & 15) != 0) return -1;
      blocks += (ns[j] + INK2G_GV_ROWS - 1) / INK2G_GV_ROWS;
    }
  }
  jobs.first[INK2G_JOBS] = blocks;
  jobs.count = count;
  const int iters = (k / 16 + 31) / 32;
  const size_t smem = (size_t)k * 4 + (size_t)INK2G_GV_ROWS * (k / 16 + 1) * 4;
  auto go = [&](auto kern) -> int {
    if (smem > 48 * 1024) {
      const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      if (e != cudaSuccess) return 10000 + (int)e;
    }
    kern<<<blocks, INK2G_GV_THREADS, smem, stream>>>(jobs, k);
    return ink2g_rc();
  };
  switch (iters) {
    case 1: return go(ink2g_gemv_bf16_kernel<1>);
    case 2: return go(ink2g_gemv_bf16_kernel<2>);
    case 3: return go(ink2g_gemv_bf16_kernel<3>);
    case 4: return go(ink2g_gemv_bf16_kernel<4>);
    case 5: return go(ink2g_gemv_bf16_kernel<5>);
    case 6: return go(ink2g_gemv_bf16_kernel<6>);
    case 7: return go(ink2g_gemv_bf16_kernel<7>);
    default: return go(ink2g_gemv_bf16_kernel<8>);
  }
}

}  // namespace

extern "C" {

int memra_ink2g_tile_words() { return INK2G_TILE_WORDS; }

// fmt: INK2G_W_*. w / scales point at expert 0; wstride / sstride are the bytes of one expert.
// amax [experts] and denom are NVFP4's. tiles (nullable): ntiles records for the stacked formats.
// single != 0: the call holds one input row (m == 1) or every tile holds one pair; it runs the
// decode GEMV. The caller states it because the tile records live on the device.
int memra_ink2g_gemm(const void* x, int64_t ldx, const void* w, const void* scales, int64_t wstride, int64_t sstride,
                     const float* amax, float denom, int fmt, int round_bf16, const uint32_t* tiles, int ntiles,
                     int xdiv, int single, void* y, int64_t ldy, int m, int n, int k, void* stream) {
  if (single && !tiles && m != 1) return 45006;
  if (!x || !w || !y || n <= 0 || k <= 0 || fmt < 0 || fmt > 3) return 45001;
  if (fmt == INK2G_W_NVFP4 && ((k % 16) != 0 || !scales || !amax || !(denom > 0.0f))) return 45002;
  if (fmt == INK2G_W_MXFP8 && ((k % 32) != 0 || !scales)) return 45003;
  if (tiles ? (ntiles <= 0 || xdiv <= 0) : m <= 0) return 45004;
  if (single && !tiles && fmt == INK2G_W_BF16 && !round_bf16 && ldx >= k) {
    // one BF16 input row: the bandwidth GEMV (the same bits, see the kernel)
    const float* xs[1] = {(const float*)x};
    const void* ws[1] = {w};
    float* ys[1] = {(float*)y};
    const int ns[1] = {n};
    const int rc = ink2g_gemv_jobs(xs, ws, ys, ns, 1, k, (cudaStream_t)stream);
    if (rc != -1) return rc;
  }
  if (single && k <= INK2G_GEMV_MAX_K) {
    // one input row a tile: the decode GEMV (the same bits, see the kernel)
    const dim3 grid((n + 255) / 256, tiles ? ntiles : 1);
    if (grid.y > 65535) return 45005;
    ink2g_gemv_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)x, ldx, (const uint8_t*)w, (const uint8_t*)scales, wstride, sstride, amax, denom, fmt,
        round_bf16, tiles, xdiv, (float*)y, ldy, n, k);
    return ink2g_rc();
  }
  const dim3 block(16, 16);
  const dim3 grid((n + TILE_N - 1) / TILE_N, tiles ? ntiles : (m + TILE_M - 1) / TILE_M);
  if (grid.y > 65535) return 45005;
  ink2g_gemm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      (const float*)x, ldx, (const uint8_t*)w, (const uint8_t*)scales, wstride, sstride, amax, denom, fmt,
      round_bf16, tiles, xdiv, (float*)y, ldy, m, n, k);
  return ink2g_rc();
}

// Several BF16 decode GEMVs (one input row each, the same input width k) in one launch: job j is
// y[j][0..n[j]) = W[j] x[j]. Bits equal memra_ink2g_gemm's for each job alone.
int memra_ink2g_gemv_jobs(const float* const* xs, const void* const* ws, float* const* ys, const int* ns, int count,
                          int k, void* stream) {
  if (!xs || !ws || !ys || !ns) return 45041;
  const int rc = ink2g_gemv_jobs(xs, ws, ys, ns, count, k, (cudaStream_t)stream);
  return rc == -1 ? 45042 : rc;
}

int memra_ink2g_embed(const void* table, const uint32_t* ids, const float* norm_w, double eps, float* out, int R,
                      int H, void* stream) {
  if (!table || !ids || !norm_w || !out || R <= 0 || H <= 0) return 45011;
  ink2g_embed_kernel<<<R, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, ids, norm_w, eps, out, H);
  return ink2g_rc();
}

int memra_ink2g_norm(const float* x, const float* norm_w, double eps, float* y, int R, int H, void* stream) {
  if (!x || !norm_w || !y || R <= 0 || H <= 0) return 45012;
  ink2g_norm_kernel<<<R, 256, 0, (cudaStream_t)stream>>>(x, norm_w, eps, y, H);
  return ink2g_rc();
}

// scratch: at least R * nq * ns * 2 doubles.
int memra_ink2g_mix(const float* const* states, int ns, const float* queries, int nq, double eps, float* out,
                    int64_t out_stride, int R, int H, double* scratch, int64_t scratch_len, void* stream) {
  if (!states || ns <= 0 || ns > INK2G_MAX_STATES || !queries || nq <= 0 || !out || R <= 0 || H <= 0) return 45013;
  if (!scratch || (int64_t)R * nq * ns * 2 > scratch_len || nq > 65535 || R > 65535) return 45043;
  Ink2gStates st;
  for (int s = 0; s < INK2G_MAX_STATES; s++) st.p[s] = s < ns ? states[s] : nullptr;
  ink2g_mix_dot_kernel<<<dim3(ns, nq, R), 256, 0, (cudaStream_t)stream>>>(st, ns, queries, nq, eps, scratch, H);
  ink2g_mix_out_kernel<<<dim3((H + 127) / 128, nq, R), 128, 0, (cudaStream_t)stream>>>(st, ns, nq, scratch, out,
                                                                                    out_stride, H);
  return ink2g_rc();
}

// The #1397 form of the mix (one block per row): the gate's oracle for the split form's bits.
int memra_ink2g_mix_v0(const float* const* states, int ns, const float* queries, int nq, double eps, float* out,
                       int64_t out_stride, int R, int H, void* stream) {
  if (!states || ns <= 0 || ns > INK2G_MAX_STATES || !queries || nq <= 0 || !out || R <= 0 || H <= 0) return 45013;
  Ink2gStates st;
  for (int s = 0; s < INK2G_MAX_STATES; s++) st.p[s] = s < ns ? states[s] : nullptr;
  ink2g_mix_kernel<<<R, 256, 0, (cudaStream_t)stream>>>(st, ns, queries, nq, eps, out, out_stride, H);
  return ink2g_rc();
}

// scratch: at least R * rank doubles. out may be x.
int memra_ink2g_controller(const float* x, const float* msg, const void* down, const void* forget, const void* gain,
                           double eps, float* out, int R, int H, int rank, double* scratch, int64_t scratch_len,
                           void* stream) {
  if (!x || !msg || !down || !forget || !gain || !out || R <= 0 || H <= 0 || rank <= 0 || rank > 128) return 45014;
  if (!scratch || (int64_t)R * rank > scratch_len || R > 65535) return 45044;
  ink2g_ctrl_down_kernel<<<dim3(rank, R), 256, 0, (cudaStream_t)stream>>>(x, msg, (const uint16_t*)down, eps, scratch,
                                                                         H, rank);
  ink2g_ctrl_up_kernel<<<dim3((H + 63) / 64, R), 64, 0, (cudaStream_t)stream>>>(
      x, msg, (const uint16_t*)forget, (const uint16_t*)gain, scratch, out, H, rank);
  return ink2g_rc();
}

// The #1397 form of the controller (one block per row): the gate's oracle for the split form's bits.
int memra_ink2g_controller_v0(const float* x, const float* msg, const void* down, const void* forget, const void* gain,
                              double eps, float* out, int R, int H, int rank, void* stream) {
  if (!x || !msg || !down || !forget || !gain || !out || R <= 0 || H <= 0 || rank <= 0 || rank > 128) return 45014;
  ink2g_controller_kernel<<<R, 256, 0, (cudaStream_t)stream>>>(
      x, msg, (const uint16_t*)down, (const uint16_t*)forget, (const uint16_t*)gain, eps, out, H, rank);
  return ink2g_rc();
}

int memra_ink2g_heads(int mode, const float* x, const float* scale, double thr, double eps, int64_t pos0,
                      int log_floor, double alpha, float* dst, int64_t cap, int R, int heads, int width,
                      void* stream) {
  if ((mode != 0 && mode != 1) || !x || !dst || R <= 0 || heads <= 0 || width <= 0 || width > 256) return 45015;
  if (heads > 65535) return 45016;
  const int threads = ((width + 31) / 32) * 32;
  ink2g_heads_kernel<<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(mode, x, scale, thr, eps, pos0,
                                                                         log_floor, alpha, dst, cap, heads, width);
  return ink2g_rc();
}

// state_in: the history before the call; state_out (nullable): the history after it, a different
// buffer, so a call that fails later leaves the caller's committed history untouched. A null
// state_out drops the history (the gate's red arm).
int memra_ink2g_conv(int mode, const float* x, const float* state_in, float* state_out, const float* taps,
                     const float* dyn, int groups, const float* add, float* out, int R, int C, int K, void* stream) {
  if ((mode != 0 && mode != 1) || !x || !state_in || !taps || !out || R <= 0 || C <= 0) return 45017;
  if (state_out == state_in) return 45041;
  if (K < 2 || K > INK2G_MAX_KERNEL) return 45018;
  if (mode == 1 && (!dyn || groups <= 0 || C % groups != 0)) return 45019;
  if (x == out) return 45020;
  ink2g_conv_kernel<<<(C + 255) / 256, 256, 0, (cudaStream_t)stream>>>(mode, x, state_in, state_out, taps, dyn, groups,
                                                                     add, out, R, C, K);
  return ink2g_rc();
}

int memra_ink2g_attn(const float* q, const float* rel, const float* table, const float* sink, const float* kc,
                     const float* vc, int64_t cap, int64_t pos0, int window, int extent, int mask_beyond, float* out,
                     int R, int heads, int kv_heads, int dim, int rank, void* stream) {
  if (!q || !rel || !table || !sink || !kc || !vc || !out) return 45021;
  if (R <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || dim <= 0 || dim > 256 || rank <= 0)
    return 45022;
  if (extent <= 0 || extent > INK2G_MAX_EXTENT || heads > 65535) return 45023;
  const int64_t last = pos0 + R - 1;
  // every key a row of this call reads is still in the cache: the ring holds the window of its
  // oldest row and the rows of the call
  if ((window > 0 ? (int64_t)window - 1 + R : last + 1) > cap) return 45024;
  ink2g_attn_kernel<<<dim3(R, heads), 256, 0, (cudaStream_t)stream>>>(
      q, rel, table, sink, kc, vc, cap, pos0, window, extent, mask_beyond, out, heads, kv_heads, dim, rank);
  return ink2g_rc();
}

int memra_ink2g_gate(float* y, const float* g, int64_t n, void* stream) {
  if (!y || !g || n <= 0) return 45031;
  ink2g_gate_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, g, n);
  return ink2g_rc();
}

int memra_ink2g_route(const float* logits, const float* bias, double scale, int top_k, int tie_high, uint32_t* ids,
                      float* wts, float* margin, int R, int E, void* stream) {
  if (!logits || !ids || !wts || R <= 0 || E <= 0 || E > 1024 || top_k <= 0 || top_k > 64 || top_k > E)
    return 45032;
  ink2g_route_kernel<<<R, 128, 0, (cudaStream_t)stream>>>(logits, bias, scale, top_k, tie_high, ids, wts, margin,
                                                        E);
  return ink2g_rc();
}

int memra_ink2g_swiglu(const float* gu, const float* wts, double gate_cap, double up_cap, float* act, int P, int F,
                       void* stream) {
  if (!gu || !act || P <= 0 || F <= 0) return 45033;
  const int64_t n = (int64_t)P * F;
  ink2g_swiglu_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(gu, wts, gate_cap, up_cap, act,
                                                                                    n, F);
  return ink2g_rc();
}

int memra_ink2g_sum(const float* dn, const float* wts, float* out, int R, int top_k, int H, void* stream) {
  if (!dn || !out || R <= 0 || top_k <= 0 || H <= 0) return 45034;
  const int64_t n = (int64_t)R * H;
  ink2g_sum_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(dn, wts, out, n, top_k, H);
  return ink2g_rc();
}

// wts (nullable): the routing weight applied after the down projection, a red arm of the gate; the
// declared program folds it into SiTU before the down quantization.
int memra_ink2g_sum_bf16(const void* dn, const float* wts, float* out, int R, int top_k, int H, void* stream) {
  if (!dn || !out || R <= 0 || top_k <= 0 || H <= 0) return 45035;
  const int64_t n = (int64_t)R * H;
  ink2g_sum_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)dn, wts, out,
                                                                                      n, top_k, H);
  return ink2g_rc();
}

// rows source rows of H, each written rep times (rep 1: a plain convert).
int memra_ink2g_to_bf16(const float* x, void* y, int rows, int H, int rep, void* stream) {
  if (!x || !y || rows <= 0 || H <= 0 || rep <= 0) return 45036;
  const int64_t n = (int64_t)rows * rep * H;
  ink2g_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, (uint16_t*)y, n, H, rep);
  return ink2g_rc();
}

// softcap (cap > 0) in place, then the argmax of each row (best nullable).
int memra_ink2g_head(float* logits, double cap, uint32_t* best, int R, int n, void* stream) {
  if (!logits || R <= 0 || n <= 0) return 45037;
  const int64_t total = (int64_t)R * n;
  if (cap > 0.0)
    ink2g_softcap_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(logits, cap, total);
  if (best) ink2g_argmax_kernel<<<R, 1024, 0, (cudaStream_t)stream>>>(logits, best, n);
  return ink2g_rc();
}

int memra_ink2g_tiles(const uint32_t* ids, uint32_t* tiles, int pairs, void* stream) {
  if (!ids || !tiles || pairs <= 0) return 45038;
  ink2g_tiles_kernel<<<(pairs + 63) / 64, 64, 0, (cudaStream_t)stream>>>(ids, tiles, pairs);
  return ink2g_rc();
}

int memra_ink2g_groups(const uint32_t* ids, uint32_t* groups, int pairs, void* stream) {
  if (!ids || !groups || pairs <= 0) return 45039;
  ink2g_groups_kernel<<<(pairs + 63) / 64, 64, 0, (cudaStream_t)stream>>>(ids, groups, pairs);
  return ink2g_rc();
}

int memra_ink2g_copy(void* dst, const void* src, int64_t bytes, void* stream) {
  if (!dst || !src || bytes <= 0) return 45040;
  const cudaError_t e = cudaMemcpyAsync(dst, src, (size_t)bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

}  // extern "C"
