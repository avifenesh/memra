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

// One row of a batched decode call (`Route::decode_batch`): the row is a one-token call of its own
// sequence, so its position, its slot's K / V caches and its four conv histories come from here
// instead of the call's (pos0 + r, the call's cache). Laid out as the host's `RowDesc`.
struct Ink2gRow {
  int64_t pos;
  int64_t cap;
  void* kc;  // float* or uint8_t* (e4m3), the slot's cache element type
  void* vc;
  const float* cin[4];
  float* cout[4];
};
// keys the split fold stages at a time, and the widest head it stages
#define INK2G_FOLD_TILE 32
#define INK2G_FOLD_DIM 128
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

// e2m1 to fp32 exactly, from the bits (a lookup table here would sit in local memory)
__device__ __forceinline__ float ink2g_e2m1(uint8_t nib) {
  const uint32_t m = nib & 7u;
  uint32_t bits = m == 0u ? 0u : (m == 1u ? 0x3f000000u : ((126u + (m >> 1)) << 23) | ((m & 1u) << 22));
  bits |= (uint32_t)(nib & 8u) << 28;
  return __uint_as_float(bits);
}

// Sum of one value per thread over the block, in a fixed order: lanes inside a warp by the shuffle
// tree, warps in ascending order. sh holds one slot per warp.
template <typename T>
__device__ __forceinline__ T ink2g_block_sum(T v, T* sh) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  T t = T(0.0);
  const int nw = (blockDim.x + 31) >> 5;
  for (int w = 0; w < nw; w++) t += sh[w];
  __syncthreads();
  return t;
}

template <typename T>
__device__ __forceinline__ T ink2g_block_max(T v, T* sh) {
  for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
  __syncthreads();
  T t = (T)-INFINITY;
  const int nw = (blockDim.x + 31) >> 5;
  for (int w = 0; w < nw; w++) t = fmax(t, sh[w]);
  __syncthreads();
  return t;
}


// ---- precision arms ----------------------------------------------------------------------------
// The f64 program of #1397 is <double, false>. <float, false> runs the same operations in f32;
// <float, true> also carries every sequential sum as a compensated (Kahan) pair.
struct Ink2gKahan {
  float s, c;
  __device__ Ink2gKahan(float v = 0.0f) : s(v), c(0.0f) {}
  __device__ Ink2gKahan& operator+=(float x) {
    const float y = __fsub_rn(x, c);
    const float t = __fadd_rn(s, y);
    c = __fsub_rn(__fsub_rn(t, s), y);
    s = t;
    return *this;
  }
  __device__ Ink2gKahan& operator=(float v) {
    s = v;
    c = 0.0f;
    return *this;
  }
  __device__ operator float() const { return s; }
};
template <typename T, bool KC>
struct Ink2gAccT {
  using type = T;
};
template <>
struct Ink2gAccT<float, true> {
  using type = Ink2gKahan;
};
#define INK2G_ACC typename Ink2gAccT<T, KC>::type

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
    if (rec[1] == 0u) return;  // an empty record of a device-built table
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

// ---- prefill GEMM, BF16 weights --------------------------------------------------------------------
// The same function as ink2g_gemm_kernel for BF16 weights and no expert tiles: each output is a
// fused multiply-add chain from 0 over each 16-input group in input order, the group partials added
// from 0 in group order, so its bits equal it. What changes is the tiling: 128 x 128 outputs a block,
// 8 x 8 a thread, the next K step staged in registers while the current one computes.
#define INK2G_SG_BM 128
#define INK2G_SG_BN 128
#define INK2G_SG_BK 16
template <bool FP4>
__global__ void __launch_bounds__(256) ink2g_sgemm_bf16_kernel(const float* __restrict__ x, int64_t ldx,
                                                              const void* __restrict__ wv0, const uint8_t* __restrict__ sc,
                                                              const float* __restrict__ amax, float denom,
                                                              float* __restrict__ y, int64_t ldy, int m, int n, int k) {
  __shared__ __align__(16) float as[2][INK2G_SG_BK][INK2G_SG_BM];
  __shared__ __align__(16) float bs[2][INK2G_SG_BK][INK2G_SG_BN];
  __shared__ float ss[2][INK2G_SG_BN];  // FP4: the e4m3 block scale of each weight row for the step
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  const int m0 = blockIdx.y * INK2G_SG_BM, n0 = blockIdx.x * INK2G_SG_BN;
  // loader: row (or weight row) tid / 2, inputs (tid & 1) * 8 .. + 8 of the step
  const int lr = tid >> 1, lk = (tid & 1) * 8;
  const bool xin = m0 + lr < m, win = n0 + lr < n;
  const float* xp = x + (int64_t)(m0 + lr) * ldx + lk;
  const uint16_t* wp = (const uint16_t*)wv0 + (int64_t)(n0 + lr) * k + lk;
  const uint8_t* cp = (const uint8_t*)wv0 + (int64_t)(n0 + lr) * (k / 2) + lk / 2;
  uint32_t cw = 0u;
  uint8_t sb = 0;
  float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
  uint4 wv = make_uint4(0u, 0u, 0u, 0u);
  auto fetch = [&](int k0) {
    if (xin) {
      xa = *reinterpret_cast<const float4*>(xp + k0);
      xb = *reinterpret_cast<const float4*>(xp + k0 + 4);
    }
    if constexpr (FP4) {
      if (win) {
        cw = __ldcs(reinterpret_cast<const uint32_t*>(cp + k0 / 2));
        sb = sc[(int64_t)(n0 + lr) * (k / 16) + k0 / 16];
      }
    } else {
      if (win) wv = __ldcs(reinterpret_cast<const uint4*>(wp + k0));
    }
  };
  auto stash = [&](int buf) {
    const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int i = 0; i < 8; i++) as[buf][lk + i][lr] = xv[i];
    if constexpr (FP4) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint8_t b = (uint8_t)((cw >> (8 * (i >> 1))) & 0xffu);
        bs[buf][lk + i][lr] = win ? ink2g_e2m1((i & 1) ? (uint8_t)(b >> 4) : b) : 0.0f;
      }
      if (lk == 0) ss[buf][lr] = win ? ink2g_e4m3(sb) : 1.0f;
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        bs[buf][lk + 2 * i][lr] = __uint_as_float(ww[i] << 16);
        bs[buf][lk + 2 * i + 1][lr] = __uint_as_float(ww[i] & 0xffff0000u);
      }
    }
  };
  float acc[8][8], part[8][8];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = part[i][j] = 0.0f;
  const int steps = k / INK2G_SG_BK;
  fetch(0);
  stash(0);
  __syncthreads();
  for (int s = 0; s < steps; s++) {
    const int buf = s & 1;
    if (s + 1 < steps) fetch((s + 1) * INK2G_SG_BK);
#pragma unroll
    for (int kk = 0; kk < INK2G_SG_BK; kk++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&as[buf][kk][ty * 4]);
      const float4 a1 = *reinterpret_cast<const float4*>(&as[buf][kk][64 + ty * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&bs[buf][kk][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&bs[buf][kk][64 + tx * 4]);
      const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) part[i][j] = fmaf(a[i], b[j], part[i][j]);
    }
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int j = 0; j < 8; j++) {
        if constexpr (FP4) {
          acc[i][j] = fmaf(part[i][j], ss[buf][j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4], acc[i][j]);
        } else {
          acc[i][j] = __fadd_rn(acc[i][j], part[i][j]);
        }
        part[i][j] = 0.0f;
      }
    if (s + 1 < steps) {
      stash(buf ^ 1);
      __syncthreads();
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int gm = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (gm >= m) continue;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int gn = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      if (gn < n) {
        float v = acc[i][j];
        if constexpr (FP4) v = (float)((double)v * (double)amax[0] / (double)denom);
        y[(int64_t)gm * ldy + gn] = v;
      }
    }
  }
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
  // x transposed: element i of group g at xs[i * groups + g] (consecutive lanes, consecutive words)
  for (int i = threadIdx.x; i < k; i += INK2G_GV_THREADS) xs[(i & 15) * groups + (i >> 4)] = x[i];
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
        const float* xg = xs + g;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          part = fmaf(xg[(2 * i) * groups], __uint_as_float(wv[i] << 16), part);
          part = fmaf(xg[(2 * i + 1) * groups], __uint_as_float(wv[i] & 0xffff0000u), part);
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
template <typename T, bool KC>
__global__ void ink2g_embed_kernel(const uint16_t* __restrict__ table, const uint32_t* __restrict__ ids,
                                   const float* __restrict__ nw, T eps, float* __restrict__ out, int H) {
  __shared__ T sh[8];
  const int r = blockIdx.x;
  const uint16_t* row = table + (int64_t)ids[r] * H;
  INK2G_ACC ss = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const T v = (T)ink2g_bf16(row[j]);
    ss += v * v;
  }
  ss = ink2g_block_sum<T>(ss, sh);
  const T f = T(1.0) / sqrt(ss / (T)H + eps);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    out[(int64_t)r * H + j] = (float)((T)nw[j] * ((T)ink2g_bf16(row[j]) * f));
}

// y[r] = w * x[r] * rsqrt(mean(x[r]^2) + eps)
template <typename T, bool KC>
__global__ void ink2g_norm_kernel(const float* __restrict__ x, const float* __restrict__ nw, T eps,
                                  float* __restrict__ y, int H) {
  __shared__ T sh[8];
  const int r = blockIdx.x;
  const float* row = x + (int64_t)r * H;
  INK2G_ACC ss = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const T v = (T)row[j];
    ss += v * v;
  }
  ss = ink2g_block_sum<T>(ss, sh);
  const T f = T(1.0) / sqrt(ss / (T)H + eps);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    y[(int64_t)r * H + j] = (float)((T)nw[j] * ((T)row[j] * f));
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
template <typename T, bool KC>
__global__ void __launch_bounds__(256) ink2g_ctrl_down_kernel(const float* __restrict__ x, const float* __restrict__ msg,
                                                             const uint16_t* __restrict__ down, T eps,
                                                             double* __restrict__ g, int H, int rank) {
  __shared__ T sh[8];
  const int o = blockIdx.x, r = blockIdx.y;
  const float* xr = x + (int64_t)r * H;
  const float* mr = msg + (int64_t)r * H;
  INK2G_ACC sx = T(0.0), sm = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    sx += (T)xr[j] * (T)xr[j];
    sm += (T)mr[j] * (T)mr[j];
  }
  sx = ink2g_block_sum<T>(sx, sh);
  sm = ink2g_block_sum<T>(sm, sh);
  const T fx = T(1.0) / sqrt(sx / (T)H + eps), fm = T(1.0) / sqrt(sm / (T)H + eps);
  const uint16_t* dr = down + (int64_t)o * 2 * H;
  INK2G_ACC d = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x)
    d += (T)ink2g_bf16(dr[j]) * ((T)xr[j] * fx) + (T)ink2g_bf16(dr[H + j]) * ((T)mr[j] * fm);
  d = ink2g_block_sum<T>(d, sh);
  if (threadIdx.x == 0) g[(int64_t)r * rank + o] = d;
}

template <typename T, bool KC>
__global__ void ink2g_ctrl_up_kernel(const float* x, const float* __restrict__ msg,
                                     const uint16_t* __restrict__ forget, const uint16_t* __restrict__ gain,
                                     const double* __restrict__ gin, float* out, int H, int rank) {
  __shared__ T g[128];
  const int r = blockIdx.y;
  for (int o = threadIdx.x; o < rank; o += blockDim.x) g[o] = gin[(int64_t)r * rank + o];
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const float* xr = x + (int64_t)r * H;
  const float* mr = msg + (int64_t)r * H;
  INK2G_ACC f = T(0.0), a = T(0.0);
  for (int o = 0; o < rank; o++) {
    f += (T)ink2g_bf16(forget[(int64_t)j * rank + o]) * g[o];
    a += (T)ink2g_bf16(gain[(int64_t)j * rank + o]) * g[o];
  }
  out[(int64_t)r * H + j] = (float)((T(1.0) - tanh(f)) * (T)xr[j] + tanh(a) * (T)mr[j]);
}

// ---- depth mix, split over blocks ----------------------------------------------------------------
// The same function as ink2g_mix_kernel, the same orders and block reductions, so the same bits.
// Phase 1: one block per (state, query, row) forms the state's RMS factor and the scaled score
// into scratch [row][query][state][2]. Phase 2: one thread per output channel; each block repeats
// the softmax over the states on thread 0.
template <typename T, bool KC>
__global__ void __launch_bounds__(256) ink2g_mix_dot_kernel(Ink2gStates st, int ns, const float* __restrict__ queries,
                                                           int nq, T eps, double* __restrict__ scr, int H) {
  __shared__ T sh[8];
  const int s = blockIdx.x, q = blockIdx.y, r = blockIdx.z;
  const float* row = st.p[s] + (int64_t)r * H;
  INK2G_ACC ss = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x) {
    const T v = (T)row[j];
    ss += v * v;
  }
  ss = ink2g_block_sum<T>(ss, sh);
  // one thread forms the two factors (an f64 sqrt and divide in every thread cost more than the dot)
  __shared__ T fac[2];
  if (threadIdx.x == 0) {
    fac[0] = T(1.0) / sqrt(ss / (T)H + eps);
    fac[1] = T(1.0) / sqrt((T)H);
  }
  __syncthreads();
  const T inv = fac[0], scale = fac[1];
  const float* qr = queries + (int64_t)q * H;
  INK2G_ACC d = T(0.0);
  for (int j = threadIdx.x; j < H; j += blockDim.x) d += (T)qr[j] * ((T)row[j] * inv);
  d = ink2g_block_sum<T>(d, sh) * scale;
  if (threadIdx.x == 0) {
    double* o = scr + (((int64_t)r * nq + q) * ns + s) * 2;
    o[0] = d;
    o[1] = inv;
  }
}

template <typename T, bool KC>
__global__ void ink2g_mix_out_kernel(Ink2gStates st, int ns, int nq, const double* __restrict__ scr,
                                     float* __restrict__ out, int64_t out_stride, int H) {
  __shared__ T inv[INK2G_MAX_STATES];
  __shared__ T prob[INK2G_MAX_STATES];
  const int q = blockIdx.y, r = blockIdx.z;
  if (threadIdx.x == 0) {
    const double* p = scr + ((int64_t)r * nq + q) * ns * 2;
    for (int s = 0; s < ns; s++) {
      prob[s] = p[2 * s];
      inv[s] = p[2 * s + 1];
    }
    T top = -INFINITY;
    for (int s = 0; s < ns; s++) top = fmax(top, prob[s]);
    INK2G_ACC den = T(0.0);
    for (int s = 0; s < ns; s++) {
      prob[s] = exp(prob[s] - top);
      den += prob[s];
    }
    for (int s = 0; s < ns; s++) prob[s] /= den;
  }
  __syncthreads();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  INK2G_ACC a = T(0.0);
  for (int s = 0; s < ns; s++) a += prob[s] * ((T)st.p[s][(int64_t)r * H + j] * inv[s]);
  out[(int64_t)q * out_stride + (int64_t)r * H + j] = (float)a;
}

// ---- head epilogues: clip / tau / value norm ------------------------------------------------------
// One block per (row, head). mode 0: v = x * scale (optional); v *= min(1, thr / (|v| + eps)); v *= tau.
// mode 1: v = x * rsqrt(mean(x^2) + eps). tau = 1 + alpha * ln(max(1, (pos + 1) / floor)) when
// log_floor > 0. The row lands at dst row (pos0 + r) % cap when cap > 0 (a cache ring), else r.
// dst may be x (a block reads its head before it writes it).
// KV cache elements: f32 as computed, or one e4m3 byte (the nvfp4-rig class: the f32 value rounded
// to the nearest e4m3, ties to even, saturating at 448, by cvt.rn.satfinite.e4m3x2.f32; read back
// exactly).
__device__ __forceinline__ void ink2g_kv_put(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void ink2g_kv_put(uint8_t* p, int64_t i, float v) {
  uint32_t b;
  asm("{ .reg .b16 q; cvt.rn.satfinite.e4m3x2.f32 q, %1, %1; cvt.u32.u16 %0, q; }" : "=r"(b) : "f"(v));
  p[i] = (uint8_t)(b & 0xffu);
}
__device__ __forceinline__ float ink2g_kv_get(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float ink2g_kv_get(const uint8_t* p, int64_t i) { return ink2g_e4m3(p[i]); }

template <typename T, bool KC, typename OUT = float>
__global__ void ink2g_heads_kernel(int mode, const float* x, const float* __restrict__ scale, T thr,
                                   T eps, const int64_t* __restrict__ posd, int64_t pos0, int log_floor,
                                   T alpha, OUT* dst, int64_t cap, int heads, int width,
                                   const Ink2gRow* __restrict__ desc, int which) {
  __shared__ T sh[8];
  const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  // desc (batched rows): the row's own position, and for which = 1 / 2 its own K / V cache; else
  // posd (nullable) is the device copy of pos0 (a replayed graph step)
  const int64_t pos = desc ? desc[r].pos : (posd ? *posd : pos0) + r;
  if (desc && which != 0) {
    dst = (OUT*)(which == 1 ? desc[r].kc : desc[r].vc);
    cap = desc[r].cap;
  }
  const int64_t drow = cap > 0 ? pos % cap : (int64_t)r;
  T v = T(0.0);
  if (t < width) {
    v = (T)x[((int64_t)r * heads + h) * width + t];
    if (scale) v *= (T)scale[h * width + t];
  }
  const T ss = ink2g_block_sum<T>(v * v, sh);
  T f;
  if (mode == 0) {
    f = fmin(thr / (sqrt(ss) + eps), T(1.0));
    if (log_floor > 0) f *= T(1.0) + alpha * log(fmax((T)(pos + 1) / (T)log_floor, T(1.0)));
  } else {
    f = T(1.0) / sqrt(ss / (T)width + eps);
  }
  if (t < width) ink2g_kv_put(dst, (drow * heads + h) * width + t, (float)(v * f));
}

// ---- short convolution with carried state --------------------------------------------------------
// One thread owns one channel and walks the rows in order, so the state it reads is the state it
// wrote. state [K - 1, C], oldest first; inputs before position 0 are zero.
// mode 0 (static): taps [C, K], oldest input first; out = (x + sum) / sqrt(K + 1).
// mode 1 (dynamic): taps [K, C] and dyn [R, groups, K], both by lag with the current token first;
//   out = (x + sum_lag (taps[lag, c] + dyn[r, group(c), lag]) * x[t - lag]) / sqrt(2K + 1).
// add (nullable): out = add + conv. carry == 0 leaves the state untouched (the gate's red arm).
template <typename T, bool KC>
__global__ void ink2g_conv_kernel(int mode, const float* __restrict__ x, const float* __restrict__ state_in,
                                  float* __restrict__ state_out, const float* __restrict__ taps,
                                  const float* __restrict__ dyn, int groups, const float* add, float* out, int R,
                                  int C, int K, const Ink2gRow* __restrict__ desc, int which) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  // desc (batched rows): block row y is a one-row call of its own sequence with its own history
  int r0 = 0;
  if (desc) {
    r0 = blockIdx.y;
    R = r0 + 1;
    state_in = desc[r0].cin[which];
    state_out = desc[r0].cout[which];
  }
  T hist[INK2G_MAX_KERNEL];  // hist[l] = input l steps back, l in 1..K-1
  for (int l = 1; l < K; l++) hist[l] = (T)state_in[(int64_t)(K - 1 - l) * C + c];
  T tp[INK2G_MAX_KERNEL];
  for (int j = 0; j < K; j++) tp[j] = mode == 0 ? (T)taps[(int64_t)c * K + j] : (T)taps[(int64_t)j * C + c];
  const int group = mode == 1 ? c / (C / groups) : 0;
  const T div = sqrt(mode == 0 ? (T)(K + 1) : (T)(2 * K + 1));
  for (int r = r0; r < R; r++) {
    const T cur = (T)x[(int64_t)r * C + c];
    INK2G_ACC sum = T(0.0);
    if (mode == 0) {
      for (int tap = 0; tap < K; tap++) {
        const int lag = K - 1 - tap;
        sum += tp[tap] * (lag == 0 ? cur : hist[lag]);
      }
    } else {
      const float* dr = dyn + ((int64_t)r * groups + group) * K;
      for (int lag = 0; lag < K; lag++) sum += (lag == 0 ? cur : hist[lag]) * (tp[lag] + (T)dr[lag]);
    }
    T o = (sum + cur) / div;
    if (add) o = (T)add[(int64_t)r * C + c] + o;
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
template <typename T, bool KC, typename KV>
__device__ __forceinline__ T ink2g_attn_score(const float* qs, const KV* kc, int64_t cap, int64_t pos,
                                                   int64_t d, int kvw, int kvh, int dim, int extent,
                                                   int mask_beyond, const T* bias, T scale) {
  if (d >= extent && mask_beyond) return -INFINITY;
  const KV* kr = kc + ((pos - d) % cap) * kvw + kvh * dim;
  INK2G_ACC dot = T(0.0);
  for (int i = 0; i < dim; i++) dot += (T)qs[i] * (T)ink2g_kv_get(kr, i);
  return dot * scale + (d < extent ? bias[d] : T(0.0));
}

template <typename T, bool KC, typename KV>
__global__ void ink2g_attn_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                  const float* __restrict__ table, const float* __restrict__ sink,
                                  const KV* __restrict__ kc, const KV* __restrict__ vc, int64_t cap, int64_t pos0,
                                  int window, int extent, int mask_beyond, float* __restrict__ out, int heads,
                                  int kv_heads, int dim, int rank, const Ink2gRow* __restrict__ desc) {
  __shared__ T sh[8];
  __shared__ T bias[INK2G_MAX_EXTENT];
  __shared__ T tile[INK2G_ATTN_TILE];
  __shared__ float qs[256];
  const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x, nt = blockDim.x;
  // desc (batched rows): the row's own position and its own sequence's caches
  if (desc) {
    kc = (const KV*)desc[r].kc;
    vc = (const KV*)desc[r].vc;
    cap = desc[r].cap;
  }
  const int64_t pos = desc ? desc[r].pos : pos0 + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int kvh = h / (heads / kv_heads);
  const int kvw = kv_heads * dim;
  const float* relr = rel + ((int64_t)r * heads + h) * rank;
  for (int d = t; d < extent && d < nkeys; d += nt) {
    INK2G_ACC b = T(0.0);
    for (int i = 0; i < rank; i++) b += (T)relr[i] * (T)table[(int64_t)i * extent + d];
    bias[d] = b;
  }
  if (t < dim) qs[t] = q[((int64_t)r * heads + h) * dim + t];
  __shared__ T scl;
  if (t == 0) scl = T(1.0) / sqrt((T)dim);
  __syncthreads();
  const T scale = scl;
  const T sk = (T)sink[(int64_t)r * heads + h];
  T top = -INFINITY;
  for (int64_t d = t; d < nkeys; d += nt)
    top = fmax(top, ink2g_attn_score<T, KC, KV>(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale));
  top = fmax(ink2g_block_max<T>(top, sh), sk);
  INK2G_ACC den = T(0.0);
  for (int64_t d = t; d < nkeys; d += nt)
    den += exp(ink2g_attn_score<T, KC, KV>(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale) - top);
  den = ink2g_block_sum<T>(den, sh) + exp(sk - top);
  INK2G_ACC a = T(0.0);
  for (int64_t hi = nkeys - 1; hi >= 0; hi -= INK2G_ATTN_TILE) {
    const int64_t lo = hi - INK2G_ATTN_TILE + 1 > 0 ? hi - INK2G_ATTN_TILE + 1 : 0;
    __syncthreads();  // the previous tile is folded
    for (int64_t d = lo + t; d <= hi; d += nt)
      tile[d - lo] =
          exp(ink2g_attn_score<T, KC, KV>(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias, scale) - top);
    __syncthreads();
    if (t < dim)
      for (int64_t d = hi; d >= lo; d--)  // oldest key first, as the reference sums
        a += tile[d - lo] * (T)ink2g_kv_get(vc, ((pos - d) % cap) * kvw + kvh * dim + t);
  }
  if (t < dim) out[((int64_t)r * heads + h) * dim + t] = (float)(a / den);
}


// ---- decode attention, split over blocks ----------------------------------------------------------
// The same function as ink2g_attn_kernel for a call of one row: every score is the same f64
// expression (ink2g_attn_score), the maximum and the denominator run with the same 256-thread
// mapping and block reductions, and the output sums the same exp(score - max) terms in the same
// oldest-first order, so the bits are equal. Phase 1 spreads the scores over blocks of
// INK2G_SC_KEYS keys into f64 scratch [heads][ld]; phase 2 runs one block per head. pos is read from
// posd when it is set (a graph replays one step at every position).
#define INK2G_SC_KEYS 256
template <typename T, bool KC, typename KV>
__global__ void __launch_bounds__(INK2G_SC_KEYS) ink2g_attn_scores_kernel(
    const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
    const KV* __restrict__ kc, int64_t cap, const int64_t* __restrict__ posd, int64_t pos0, int window, int extent,
    int mask_beyond, double* __restrict__ scr, int64_t ld, int heads, int kv_heads, int dim, int rank, int r0) {
  __shared__ T bias[INK2G_SC_KEYS];
  __shared__ float qs[256];
  const int h = blockIdx.y, t = threadIdx.x, r = r0 + (int)blockIdx.z;
  const int64_t pos = (posd ? *posd : pos0) + r;
  q += (int64_t)r * heads * dim;
  rel += (int64_t)r * heads * rank;
  scr += (int64_t)blockIdx.z * heads * ld;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int64_t d0 = (int64_t)blockIdx.x * INK2G_SC_KEYS;
  if (d0 >= nkeys) return;
  const int kvh = h / (heads / kv_heads);
  const int kvw = kv_heads * dim;
  const float* relr = rel + (int64_t)h * rank;
  const int64_t d = d0 + t;
  if (d < extent && d < nkeys) {
    INK2G_ACC b = T(0.0);
    for (int i = 0; i < rank; i++) b += (T)relr[i] * (T)table[(int64_t)i * extent + d];
    bias[t] = b;
  }
  for (int i = t; i < dim; i += blockDim.x) qs[i] = q[(int64_t)h * dim + i];
  __shared__ T scl;
  if (t == 0) scl = T(1.0) / sqrt((T)dim);
  __syncthreads();
  if (d >= nkeys) return;
  const T scale = scl;
  scr[(int64_t)h * ld + d] = ink2g_attn_score<T, KC, KV>(qs, kc, cap, pos, d, kvw, kvh, dim, extent, mask_beyond, bias - d0, scale);
}

// The fold of the split form in three launches, the same f64 (or arm) program as one block a head:
// top: one block a (row, head), the max over the row's scores and the sink (a max is exact in any
// order), stored at scr[ld - 1]; exp: every key's weight exp(score - top) in place, over many blocks;
// out: blocks of INK2G_FOLD_OUT outputs of a head, each forming the denominator with the one-block
// thread mapping (so its bits are the one-block sum) and the weighted sums, oldest key first, from
// 32-key tiles staged in shared memory.
#define INK2G_FOLD_OUT 16
template <typename T, bool KC>
__global__ void __launch_bounds__(256) ink2g_attn_top_kernel(const float* __restrict__ sink,
                                                             const int64_t* __restrict__ posd, int64_t pos0,
                                                             int window, double* __restrict__ scr, int64_t ld,
                                                             int heads, int r0) {
  __shared__ T sh[8];
  const int h = blockIdx.y, t = threadIdx.x, nt = blockDim.x, r = r0 + (int)blockIdx.x;
  const int64_t pos = (posd ? *posd : pos0) + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  double* sc = scr + ((int64_t)blockIdx.x * heads + h) * ld;
  const T sk = (T)sink[(int64_t)r * heads + h];
  T top = -INFINITY;
  for (int64_t d = t; d < nkeys; d += nt) top = fmax(top, (T)sc[d]);
  top = fmax(ink2g_block_max<T>(top, sh), sk);
  if (t == 0) sc[ld - 1] = (double)top;
}

template <typename T, bool KC>
__global__ void __launch_bounds__(256) ink2g_attn_exp_kernel(const int64_t* __restrict__ posd, int64_t pos0,
                                                             int window, double* __restrict__ scr, int64_t ld,
                                                             int heads, int r0) {
  const int h = blockIdx.y, r = r0 + (int)blockIdx.z;
  const int64_t pos = (posd ? *posd : pos0) + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int64_t d = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= nkeys) return;
  double* sc = scr + ((int64_t)blockIdx.z * heads + h) * ld;
  const T top = (T)sc[ld - 1];
  const T e = exp((T)sc[d] - top);
  sc[d] = e;
}

template <typename T, bool KC, typename KV>
__global__ void __launch_bounds__(256) ink2g_attn_out_kernel(const float* __restrict__ sink, const KV* __restrict__ vc,
                                                             int64_t cap, const int64_t* __restrict__ posd,
                                                             int64_t pos0, int window, const double* __restrict__ scr,
                                                             int64_t ld, float* __restrict__ out, int heads,
                                                             int kv_heads, int dim, int r0, int od) {
  __shared__ T sh[8];
  __shared__ T ske;
  __shared__ double tw[INK2G_FOLD_TILE];
  __shared__ float tv[INK2G_FOLD_TILE * INK2G_FOLD_DIM];
  const int h = blockIdx.y, t = threadIdx.x, nt = blockDim.x, r = r0 + (int)blockIdx.x;
  const int c0 = (int)blockIdx.z * od;
  const int oc = dim - c0 < od ? dim - c0 : od;
  const int64_t pos = (posd ? *posd : pos0) + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int kvh = h / (heads / kv_heads);
  const int kvw = kv_heads * dim;
  const double* sc = scr + ((int64_t)blockIdx.x * heads + h) * ld;
  const T top = (T)sc[ld - 1];
  if (t == 0) ske = exp((T)sink[(int64_t)r * heads + h] - top);
  INK2G_ACC den = T(0.0);
  for (int64_t d = t; d < nkeys; d += nt) den += (T)sc[d];
  den = ink2g_block_sum<T>(den, sh);  // its barriers publish ske
  const T dn = (T)den + ske;
  INK2G_ACC a = T(0.0);
  for (int64_t hi = nkeys - 1; hi >= 0; hi -= INK2G_FOLD_TILE) {
    const int nk = (int)(hi + 1 < INK2G_FOLD_TILE ? hi + 1 : INK2G_FOLD_TILE);
    for (int i = t; i < nk; i += nt) tw[i] = sc[hi - i];
    for (int e = t; e < nk * oc; e += nt) {
      const int i = e / oc, c = e - i * oc;
      tv[i * od + c] = ink2g_kv_get(vc, ((pos - (hi - i)) % cap) * kvw + kvh * dim + c0 + c);
    }
    __syncthreads();
    if (t < oc)
      for (int i = 0; i < nk; i++) a += (T)tw[i] * (T)tv[i * od + t];
    __syncthreads();
  }
  if (t < oc) out[((int64_t)r * heads + h) * dim + c0 + t] = (float)(a / dn);
}

// y *= sigmoid(g)
template <typename T, bool KC>
__global__ void ink2g_gate_kernel(float* __restrict__ y, const float* __restrict__ g, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)((T)y[i] * (T(1.0) / (T(1.0) + exp(-(T)g[i]))));
}

// ---- router ------------------------------------------------------------------------------------
// One block per row. score = sigmoid(logit); the top_k experts by score + bias, a tie going to the
// lower expert (tie_high: the gate's red arm, to the higher), in rank order; weight =
// score / (sum of the selected scores) * scale. margin (nullable) = key of rank top_k - 1 minus key
// of rank top_k.
// An order-preserving integer image of a finite value: a < b exactly when ord(a) < ord(b), equal values
// (-0 and +0 too) give equal images. The router ranks on it: an f64 compare is as slow as an f64 fma here.
__device__ __forceinline__ unsigned long long ink2g_ord(double v) {
  unsigned long long b = (unsigned long long)__double_as_longlong(v == 0.0 ? 0.0 : v);
  return (b >> 63) ? ~b : b | 0x8000000000000000ull;
}
__device__ __forceinline__ unsigned long long ink2g_ord(float v) {
  uint32_t b = __float_as_uint(v == 0.0f ? 0.0f : v);
  return (b >> 31) ? (unsigned long long)~b : (unsigned long long)(b | 0x80000000u);
}

template <typename T, bool KC>
__global__ void ink2g_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, T scale,
                                   int top_k, int tie_high, uint32_t* __restrict__ ids, float* __restrict__ wts,
                                   float* __restrict__ margin, int E) {
  __shared__ T key[1024];
  __shared__ T score[1024];
  __shared__ T sel[64];
  __shared__ T edge[2];
  __shared__ unsigned long long kord[1024];
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    const T s = T(1.0) / (T(1.0) + exp(-(T)logits[(int64_t)r * E + i]));
    score[i] = s;
    key[i] = bias ? s + (T)bias[i] : s;
    kord[i] = ink2g_ord(key[i]);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < E; i += blockDim.x) {
    int rank = 0;
    const T ki = key[i];
    const unsigned long long oi = kord[i];
    for (int j = 0; j < E; j++) {
      const unsigned long long oj = kord[j];
      const bool first = tie_high ? j > i : j < i;
      rank += (oj > oi || (oj == oi && first)) ? 1 : 0;
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
    INK2G_ACC total = T(0.0);
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
template <typename T, bool KC>
__global__ void ink2g_softcap_kernel(float* __restrict__ y, T cap, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)(tanh((T)y[i] / cap) * cap);
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

// Expert groups of a call of several rows, built on the device (no host round trip): the first,
// (cap+1)-th, ... occurrence of an expert among the routed pairs leads a record {expert, count,
// pair[cap]} at its own index; its next cap occurrences of that expert, in pair order, fill it. The
// other records stay empty (count 0), and the GEMMs skip them. pairs <= 256.
__global__ void ink2g_groups_dev_kernel(const uint32_t* __restrict__ ids, uint32_t* __restrict__ rec, int pairs,
                                        int words, int cap) {
  __shared__ uint32_t sid[256];
  const int p = threadIdx.x;
  if (p < pairs) sid[p] = ids[p];
  for (int i = p; i < pairs * words; i += blockDim.x) rec[i] = 0u;
  __syncthreads();
  if (p >= pairs) return;
  const uint32_t e = sid[p];
  int idx = 0, total = 0;
  for (int q = 0; q < pairs; q++)
    if (sid[q] == e) {
      if (q < p) idx++;
      total++;
    }
  // the leader: occurrence number (idx / cap) * cap of this expert
  const int want = (idx / cap) * cap;
  int leader = p, seen = 0;
  for (int q = 0; q < pairs; q++)
    if (sid[q] == e) {
      if (seen == want) {
        leader = q;
        break;
      }
      seen++;
    }
  uint32_t* r = rec + (int64_t)leader * words;
  r[2 + idx % cap] = (uint32_t)p;
  if (p == leader) {
    r[0] = e;
    r[1] = (uint32_t)(total - want < cap ? total - want : cap);
  }
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


// ---- decode GEMV over 2 to 4 input rows at a time (a speculative verify, a short chunk) -----------
// Per input row the same function as ink2g_gemv_bf16_kernel / ink2g_gemv_nvfp4_jobs_kernel (and the tile
// GEMM): per 16-input group a fused multiply-add chain from 0, the partials folded in group order
// (BF16: added; NVFP4: fmaf(partial, e4m3 scale, sum), the tensor scale once in f64). A lane reads its
// weight slice once and forms the partial of every input row from it.
#define INK2G_MR_ROWS 32
template <int ITERS, int R, bool FP4>
__global__ void __launch_bounds__(256) ink2g_gemv_rows_kernel(const float* __restrict__ x, int64_t ldx,
                                                             const void* __restrict__ w, const uint8_t* __restrict__ sc,
                                                             const float* __restrict__ amax, float denom,
                                                             float* __restrict__ y, int64_t ldy, int n, int k) {
  extern __shared__ float ink2g_mr[];
  constexpr int WARPS = 8, RPW = INK2G_MR_ROWS / WARPS;
  const int groups = k >> 4, pstride = groups + 1;
  // x transposed per input row: element i of group g at xs[r * k + i * groups + g] (a warp's lanes,
  // consecutive groups, read consecutive words)
  float* xs = ink2g_mr;                                                   // [R][16][groups]
  float* ps = xs + R * k;                                                 // [WARPS][R][pstride]
  uint8_t* sv = reinterpret_cast<uint8_t*>(ps + WARPS * R * pstride);     // [WARPS][pstride] (NVFP4)
  const int r0 = (int)blockIdx.x * INK2G_MR_ROWS;
  for (int i = threadIdx.x; i < R * k; i += 256) {
    const int r = i / k, e = i - r * k;
    xs[r * k + (e & 15) * groups + (e >> 4)] = x[(int64_t)r * ldx + e];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float* pw = ps + warp * R * pstride;
  uint8_t* sw = sv + warp * pstride;
  for (int q = 0; q < RPW; q++) {
    const int row = r0 + warp * RPW + q;
    if (row >= n) break;
#pragma unroll
    for (int it = 0; it < ITERS; it++) {
      const int g = it * 32 + lane;
      if (g >= groups) continue;
      float wv[16];
      if constexpr (FP4) {
        const uint2 c = __ldcs(reinterpret_cast<const uint2*>((const uint8_t*)w + (int64_t)row * (k / 2)) + g);
        const uint32_t cw[2] = {c.x, c.y};
#pragma unroll
        for (int i = 0; i < 16; i++) {
          const uint32_t b = (cw[i >> 3] >> (8 * ((i >> 1) & 3))) & 0xffu;
          wv[i] = ink2g_e2m1((i & 1) ? (uint8_t)(b >> 4) : (uint8_t)b);
        }
        sw[g] = sc[(int64_t)row * groups + g];
      } else {
        const uint4* wr = reinterpret_cast<const uint4*>((const uint16_t*)w + (int64_t)row * k);
        const uint4 lo = __ldcs(wr + 2 * g), hi = __ldcs(wr + 2 * g + 1);
        const uint32_t ww[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
        for (int i = 0; i < 8; i++) {
          wv[2 * i] = __uint_as_float(ww[i] << 16);
          wv[2 * i + 1] = __uint_as_float(ww[i] & 0xffff0000u);
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++) {
        const float* xg = xs + r * k + g;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) part = fmaf(xg[i * groups], wv[i], part);
        pw[r * pstride + g] = part;
      }
    }
    __syncwarp();
    // lane r folds input row r of this weight row, in group order
    if (lane < R) {
      const float* p = pw + lane * pstride;
      float acc = 0.0f;
      if constexpr (FP4) {
        for (int g = 0; g < groups; g++) acc = fmaf(p[g], ink2g_e4m3(sw[g]), acc);
        acc = (float)((double)acc * (double)amax[0] / (double)denom);
      } else {
        for (int g = 0; g < groups; g++) acc = __fadd_rn(acc, p[g]);
      }
      y[(int64_t)lane * ldy + row] = acc;
    }
    __syncwarp();
  }
}

// rows 1..4 of x through the multi-row GEMV; -1 when the shape does not fit
template <int R, bool FP4>
int ink2g_gemv_rows_r(const float* x, int64_t ldx, const void* w, const uint8_t* sc, const float* amax, float denom,
                      float* y, int64_t ldy, int n, int k, cudaStream_t stream) {
  const int iters = (k / 16 + 31) / 32;
  const size_t smem = (size_t)R * k * 4 + (size_t)8 * R * (k / 16 + 1) * 4 + (size_t)8 * (k / 16 + 1);
  if (smem > 99 * 1024) return -1;
  const int blocks = (n + INK2G_MR_ROWS - 1) / INK2G_MR_ROWS;
  auto go = [&](auto kern) -> int {
    const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return 10000 + (int)e;
    kern<<<blocks, 256, smem, stream>>>(x, ldx, w, sc, amax, denom, y, ldy, n, k);
    return ink2g_rc();
  };
  switch (iters) {
    case 1: return go(ink2g_gemv_rows_kernel<1, R, FP4>);
    case 2: return go(ink2g_gemv_rows_kernel<2, R, FP4>);
    case 3: return go(ink2g_gemv_rows_kernel<3, R, FP4>);
    case 4: return go(ink2g_gemv_rows_kernel<4, R, FP4>);
    case 5: return go(ink2g_gemv_rows_kernel<5, R, FP4>);
    case 6: return go(ink2g_gemv_rows_kernel<6, R, FP4>);
    case 7: return go(ink2g_gemv_rows_kernel<7, R, FP4>);
    default: return go(ink2g_gemv_rows_kernel<8, R, FP4>);
  }
}

// m in 2..8 input rows, in passes of up to 4 (a pass re-reads the weights, mostly from L2)
int ink2g_gemv_rows(const float* x, int64_t ldx, const void* w, const uint8_t* sc, const float* amax, float denom,
                    bool fp4, float* y, int64_t ldy, int m, int n, int k, cudaStream_t stream) {
  if (m < 2 || m > 8 || n <= 0 || (k % 16) != 0 || k > INK2G_GV_MAX_K || ((uintptr_t)w & 15) != 0) return -1;
  for (int r0 = 0; r0 < m; r0 += 4) {
    const int r = m - r0 < 4 ? m - r0 : 4;
    const float* xr = x + (int64_t)r0 * ldx;
    float* yr = y + (int64_t)r0 * ldy;
    int rc;
    if (fp4) {
      rc = r == 1 ? ink2g_gemv_rows_r<1, true>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
         : r == 2 ? ink2g_gemv_rows_r<2, true>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
         : r == 3 ? ink2g_gemv_rows_r<3, true>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
                  : ink2g_gemv_rows_r<4, true>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream);
    } else {
      rc = r == 1 ? ink2g_gemv_rows_r<1, false>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
         : r == 2 ? ink2g_gemv_rows_r<2, false>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
         : r == 3 ? ink2g_gemv_rows_r<3, false>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream)
                  : ink2g_gemv_rows_r<4, false>(xr, ldx, w, sc, amax, denom, yr, ldy, n, k, stream);
    }
    if (rc != 0) return rc;
  }
  return 0;
}


// ---- decode GEMV over dense NVFP4 weights, several jobs a launch --------------------------------
// ink2g_gemv_kernel's NVFP4 program per output (per group an fma chain on the e2m1 codes, the partials folded as fmaf(partial, e4m3 scale, sum) in order, the tensor scale once in f64), with jobs like ink2g_gemv_bf16_kernel: the q / r /
// gate / k / v / output projections of a decode step in one launch. Each warp issues the code loads of
// all its rows before it computes, so a warp keeps its rows' bytes in flight together.
struct Ink2gJobs4 {
  const float* x[INK2G_JOBS];
  const uint8_t* w[INK2G_JOBS];
  const uint8_t* sc[INK2G_JOBS];
  const float* amax[INK2G_JOBS];
  float denom[INK2G_JOBS];
  float* y[INK2G_JOBS];
  int n[INK2G_JOBS];
  int first[INK2G_JOBS + 1];
  int count;
};

template <int ITERS>
// three blocks an SM (registers capped for it): two left the DRAM 80% idle on the decode shapes
__global__ void __launch_bounds__(INK2G_GV_THREADS, 3) ink2g_gemv_nvfp4_jobs_kernel(const Ink2gJobs4 jobs, int k) {
  extern __shared__ float ink2g_gj4[];
  constexpr int RPW = INK2G_GV_ROWS / (INK2G_GV_THREADS / 32);  // rows a warp
  const int groups = k >> 4, pstride = groups + 1;
  float* xs = ink2g_gj4;
  float* ps = ink2g_gj4 + k;
  uint8_t* sv = reinterpret_cast<uint8_t*>(ps + INK2G_GV_ROWS * pstride);
  int j = 0;
  while (j + 1 < jobs.count && (int)blockIdx.x >= jobs.first[j + 1]) j++;
  const int r0 = ((int)blockIdx.x - jobs.first[j]) * INK2G_GV_ROWS;
  const int n = jobs.n[j];
  const float* x = jobs.x[j];
  const uint8_t* w = jobs.w[j];
  const uint8_t* sc = jobs.sc[j];
  // x transposed: element i of group g at xs[i * groups + g], so a warp's lanes (consecutive groups)
  // read consecutive words
  for (int i = threadIdx.x; i < k; i += INK2G_GV_THREADS) xs[(i & 15) * groups + (i >> 4)] = x[i];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint2 c[RPW][ITERS];
  uint8_t sb[RPW][ITERS];
#pragma unroll
  for (int q = 0; q < RPW; q++) {
    const int row = r0 + warp + q * (INK2G_GV_THREADS / 32);
#pragma unroll
    for (int it = 0; it < ITERS; it++) {
      const int g = it * 32 + lane;
      if (row < n && g < groups) {
        c[q][it] = __ldcs(reinterpret_cast<const uint2*>(w + (int64_t)row * (k / 2)) + g);
        sb[q][it] = sc[(int64_t)row * groups + g];
      }
    }
  }
  __syncthreads();
  // a group of x is read once into registers for all the warp's rows
#pragma unroll
  for (int it = 0; it < ITERS; it++) {
    const int g = it * 32 + lane;
    if (g >= groups) continue;
    float xv[16];
#pragma unroll
    for (int i = 0; i < 16; i++) xv[i] = xs[i * groups + g];
#pragma unroll
    for (int q = 0; q < RPW; q++) {
      const int rl = warp + q * (INK2G_GV_THREADS / 32), row = r0 + rl;
      if (row >= n) continue;
      const uint32_t cw[2] = {c[q][it].x, c[q][it].y};
      float part = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const uint32_t b = (cw[i >> 3] >> (8 * ((i >> 1) & 3))) & 0xffu;
        part = fmaf(xv[i], ink2g_e2m1((i & 1) ? (uint8_t)(b >> 4) : (uint8_t)b), part);
      }
      ps[rl * pstride + g] = part;
      sv[rl * pstride + g] = sb[q][it];
    }
  }
  // each warp folds its own rows (lane q folds row q of the warp): no block-wide wait
  __syncwarp();
  if (lane < RPW) {
    const int rl = warp + lane * (INK2G_GV_THREADS / 32), row = r0 + rl;
    if (row < n) {
      const float* p = ps + rl * pstride;
      const uint8_t* sp = sv + rl * pstride;
      float acc = 0.0f;
      for (int g = 0; g < groups; g++) acc = fmaf(p[g], ink2g_e4m3(sp[g]), acc);
      jobs.y[j][row] = (float)((double)acc * (double)jobs.amax[j][0] / (double)jobs.denom[j]);
    }
  }
}

int ink2g_gemv_nvfp4_jobs(const float* const* xs, const void* const* ws, const void* const* scs,
                          const float* const* amaxs, const float* denoms, float* const* ys, const int* ns, int count,
                          int k, cudaStream_t stream) {
  if (count <= 0 || count > INK2G_JOBS || k <= 0 || (k % 16) != 0 || k > INK2G_GV_MAX_K) return -1;
  Ink2gJobs4 jobs;
  int blocks = 0;
  for (int j = 0; j < INK2G_JOBS; j++) {
    const bool on = j < count;
    jobs.x[j] = on ? xs[j] : nullptr;
    jobs.w[j] = on ? (const uint8_t*)ws[j] : nullptr;
    jobs.sc[j] = on ? (const uint8_t*)scs[j] : nullptr;
    jobs.amax[j] = on ? amaxs[j] : nullptr;
    jobs.denom[j] = on ? denoms[j] : 1.0f;
    jobs.y[j] = on ? ys[j] : nullptr;
    jobs.n[j] = on ? ns[j] : 0;
    jobs.first[j] = blocks;
    if (on) {
      if (!xs[j] || !ws[j] || !scs[j] || !amaxs[j] || !ys[j] || ns[j] <= 0 || ((uintptr_t)ws[j] & 7) != 0) return -1;
      blocks += (ns[j] + INK2G_GV_ROWS - 1) / INK2G_GV_ROWS;
    }
  }
  jobs.first[INK2G_JOBS] = blocks;
  jobs.count = count;
  const int iters = (k / 16 + 31) / 32;
  const size_t smem = (size_t)k * 4 + (size_t)INK2G_GV_ROWS * (k / 16 + 1) * 5;
  auto go = [&](auto kern) -> int {
    if (smem > 48 * 1024) {
      const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
      if (e != cudaSuccess) return 10000 + (int)e;
    }
    kern<<<blocks, INK2G_GV_THREADS, smem, stream>>>(jobs, k);
    return ink2g_rc();
  };
  switch (iters) {
    case 1: return go(ink2g_gemv_nvfp4_jobs_kernel<1>);
    case 2: return go(ink2g_gemv_nvfp4_jobs_kernel<2>);
    case 3: return go(ink2g_gemv_nvfp4_jobs_kernel<3>);
    case 4: return go(ink2g_gemv_nvfp4_jobs_kernel<4>);
    case 5: return go(ink2g_gemv_nvfp4_jobs_kernel<5>);
    case 6: return go(ink2g_gemv_nvfp4_jobs_kernel<6>);
    case 7: return go(ink2g_gemv_nvfp4_jobs_kernel<7>);
    default: return go(ink2g_gemv_nvfp4_jobs_kernel<8>);
  }
}


// ---- attention arm 4: f32 tiles over keys, f64 merges (a flash-decode split) ---------------------
// Each block takes one tile of keys of a (row, head): the scores q . k / sqrt(dim) + bias and the
// tile's max, exp and value sum in f32 (oldest key first). The merge combines the tiles of a row in
// f64: M = max(sink, tile maxima), L = exp(sink - M) + sum_t exp(m_t - M) l_t, out = sum_t exp(m_t - M)
// o_t / L, tiles oldest first. A different numeric program from arms 0..3: measured by KL.
#define INK2G_T4_MAX_TILE 1024
#define INK2G_T4_MAX_TILES 512
template <typename KV>
__global__ void __launch_bounds__(256) ink2g_attn_t4_kernel(
    const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ table,
    const KV* __restrict__ kc, const KV* __restrict__ vc, int64_t cap, const int64_t* __restrict__ posd,
    int64_t pos0, int window, int extent, int mask_beyond, double* __restrict__ scr, int64_t ld, int heads,
    int kv_heads, int dim, int rank, int r0, int tile) {
  __shared__ float sh[8];
  __shared__ float qs[INK2G_FOLD_DIM];
  __shared__ float sb[INK2G_T4_MAX_TILE];
  __shared__ float sp[INK2G_T4_MAX_TILE];
  __shared__ float tv[INK2G_FOLD_TILE * INK2G_FOLD_DIM];
  __shared__ float scl;
  const int ti = blockIdx.x, h = blockIdx.y, t = threadIdx.x, nt = blockDim.x, r = r0 + (int)blockIdx.z;
  const int64_t pos = (posd ? *posd : pos0) + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int64_t d0 = (int64_t)ti * tile;
  if (d0 >= nkeys) return;
  const int cnt = (int)(nkeys - d0 < tile ? nkeys - d0 : tile);
  const int kvh = h / (heads / kv_heads);
  const int kvw = kv_heads * dim;
  const float* relr = rel + ((int64_t)r * heads + h) * rank;
  double* part = scr + ((int64_t)blockIdx.z * heads + h) * ld + (int64_t)ti * (dim + 2);
  for (int i = t; i < dim; i += nt) qs[i] = q[((int64_t)r * heads + h) * dim + i];
  if (t == 0) scl = 1.0f / sqrtf((float)dim);
  for (int i = t; i < cnt; i += nt) {
    const int64_t d = d0 + i;
    if (d < extent) {
      float b = 0.0f;
      for (int k = 0; k < rank; k++) b += relr[k] * table[(int64_t)k * extent + d];
      sb[i] = b;
    }
  }
  __syncthreads();
  float mx = -INFINITY;
  for (int i = t; i < cnt; i += nt) {
    const float s = ink2g_attn_score<float, false, KV>(qs, kc, cap, pos, d0 + i, kvw, kvh, dim, extent, mask_beyond,
                                                       sb - d0, scl);
    sp[i] = s;
    mx = fmaxf(mx, s);
  }
  const float mt = ink2g_block_max<float>(mx, sh);
  if (mt == -INFINITY) {
    if (t == 0) {
      part[0] = -INFINITY;
      part[1] = 0.0;
    }
    for (int c = t; c < dim; c += nt) part[2 + c] = 0.0;
    return;
  }
  float ls = 0.0f;
  for (int i = t; i < cnt; i += nt) {
    const float e = expf(sp[i] - mt);
    sp[i] = e;
    ls += e;
  }
  const float l = ink2g_block_sum<float>(ls, sh);  // its barriers order the stores above
  float a = 0.0f;
  for (int hi = cnt - 1; hi >= 0; hi -= INK2G_FOLD_TILE) {
    const int n = hi + 1 < INK2G_FOLD_TILE ? hi + 1 : INK2G_FOLD_TILE;
    for (int e = t; e < n * dim; e += nt) {
      const int j = e / dim, c = e - j * dim;
      tv[j * INK2G_FOLD_DIM + c] = ink2g_kv_get(vc, ((pos - (d0 + hi - j)) % cap) * kvw + kvh * dim + c);
    }
    __syncthreads();
    if (t < dim)
      for (int j = 0; j < n; j++) a = fmaf(sp[hi - j], tv[j * INK2G_FOLD_DIM + t], a);
    __syncthreads();
  }
  if (t == 0) {
    part[0] = (double)mt;
    part[1] = (double)l;
  }
  if (t < dim) part[2 + t] = (double)a;
}

__global__ void __launch_bounds__(128) ink2g_attn_t4_merge_kernel(const float* __restrict__ sink,
                                                                  const int64_t* __restrict__ posd, int64_t pos0,
                                                                  int window, const double* __restrict__ scr,
                                                                  int64_t ld, float* __restrict__ out, int heads,
                                                                  int dim, int r0, int tile) {
  __shared__ double w[INK2G_T4_MAX_TILES];
  __shared__ double ms, lsum;
  const int h = blockIdx.y, t = threadIdx.x, nt = blockDim.x, r = r0 + (int)blockIdx.x;
  const int64_t pos = (posd ? *posd : pos0) + r;
  const int64_t nkeys = window > 0 ? (pos + 1 < window ? pos + 1 : window) : pos + 1;
  const int ntiles = (int)((nkeys + tile - 1) / tile);
  const double* base = scr + ((int64_t)blockIdx.x * heads + h) * ld;
  const double sk = (double)sink[(int64_t)r * heads + h];
  if (t == 0) {
    double m = sk;
    for (int i = 0; i < ntiles; i++) m = fmax(m, base[(int64_t)i * (dim + 2)]);
    ms = m;
  }
  __syncthreads();
  const double m = ms;
  for (int i = t; i < ntiles; i += nt) {
    const double mi = base[(int64_t)i * (dim + 2)];
    w[i] = mi == -INFINITY ? 0.0 : exp(mi - m);
  }
  __syncthreads();
  if (t == 0) {
    double l = exp(sk - m);
    for (int i = ntiles - 1; i >= 0; i--) l += w[i] * base[(int64_t)i * (dim + 2) + 1];
    lsum = l;
  }
  __syncthreads();
  if (t < dim) {
    double o = 0.0;
    for (int i = ntiles - 1; i >= 0; i--) o += w[i] * base[(int64_t)i * (dim + 2) + 2 + t];
    out[((int64_t)r * heads + h) * dim + t] = (float)(o / lsum);
  }
}

}  // namespace

template <typename KV>
static int ink2g_attn_any(const float* q, const float* rel, const float* table, const float* sink, const KV* kc,
                          const KV* vc, int64_t cap, const int64_t* posd, int64_t pos0, int window, int extent,
                     int mask_beyond, float* out, int R, int heads, int kv_heads, int dim, int rank, double* scr,
                     int64_t scr_ld, int scr_rows, int prec, const Ink2gRow* desc, void* stream) {
  if (!q || !rel || !table || !sink || ((!kc || !vc) && !desc) || !out) return 45021;
  if (R <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || dim <= 0 || dim > 256 || rank <= 0)
    return 45022;
  if (extent <= 0 || extent > INK2G_MAX_EXTENT || heads > 65535) return 45023;
  const int64_t last = pos0 + R - 1;
  // every key a row of this call reads is still in the cache: the ring holds the window of its
  // oldest row and the rows of the call
  // (a batched call's rows are checked on the host: each row's own cache holds its keys)
  if (!desc && (window > 0 ? (int64_t)window - 1 + R : last + 1) > cap) return 45024;
  // a batched call never takes the split form: its rows are different sequences
  if (scr && !desc && (R == 1 || scr_rows > 0)) {
    // the split form, scr_rows rows a launch pair (scratch [scr_rows][heads][scr_ld]); the most keys a
    // row can attend: a replayed step reads its position on the device
    const int64_t most = window > 0 ? (int64_t)window : (R == 1 ? cap : pos0 + R);
    // a row of scratch holds its scores and, last, the max of the row
    if (dim > INK2G_FOLD_DIM) return 45025;
    const int group = R == 1 ? 1 : scr_rows;
    // arm 4's scratch holds tiles, not scores: its own bound below
    if (prec == 4) {
      // tiles of 256 keys, wider when a row would need more than INK2G_T4_MAX_TILES of them
      // from the cache size, not the call: every call over this cache tiles the keys alike, so the
      // program does not depend on how a prompt is split into calls
      const int64_t tile = 256 * ((cap + 256 * INK2G_T4_MAX_TILES - 1) / (256 * INK2G_T4_MAX_TILES));
      if (tile > INK2G_T4_MAX_TILE || scr_ld < ((most + tile - 1) / tile) * (dim + 2)) return 45026;
      for (int r0 = 0; r0 < R; r0 += group) {
        const int n = R - r0 < group ? R - r0 : group;
        const int64_t keys = R == 1 ? most : (window > 0 ? most : pos0 + r0 + n);
        ink2g_attn_t4_kernel<KV><<<dim3((unsigned)((keys + tile - 1) / tile), heads, n), 256, 0, (cudaStream_t)stream>>>(
            q, rel, table, kc, vc, cap, posd, pos0, window, extent, mask_beyond, scr, scr_ld, heads, kv_heads, dim,
            rank, r0, (int)tile);
        ink2g_attn_t4_merge_kernel<<<dim3(n, heads), 128, 0, (cudaStream_t)stream>>>(sink, posd, pos0, window, scr,
                                                                                     scr_ld, out, heads, dim, r0,
                                                                                     (int)tile);
      }
      return ink2g_rc();
    }
    if (scr_ld < most + 1) return 45025;
    for (int r0 = 0; r0 < R; r0 += group) {
      const int n = R - r0 < group ? R - r0 : group;
      // the keys of the group's last row
      const int64_t keys = R == 1 ? most : (window > 0 ? most : pos0 + r0 + n);
      switch (prec) {
    case 1: ink2g_attn_scores_kernel<float, false, KV><<<dim3((unsigned)((keys + INK2G_SC_KEYS - 1) / INK2G_SC_KEYS), heads, n), INK2G_SC_KEYS,
                                 0, (cudaStream_t)stream>>>(q, rel, table, kc, cap, posd, pos0, window, extent,
                                                            mask_beyond, scr, scr_ld, heads, kv_heads, dim, rank, r0); break;
    case 2: ink2g_attn_scores_kernel<float, true, KV><<<dim3((unsigned)((keys + INK2G_SC_KEYS - 1) / INK2G_SC_KEYS), heads, n), INK2G_SC_KEYS,
                                 0, (cudaStream_t)stream>>>(q, rel, table, kc, cap, posd, pos0, window, extent,
                                                            mask_beyond, scr, scr_ld, heads, kv_heads, dim, rank, r0); break;
    default: ink2g_attn_scores_kernel<double, false, KV><<<dim3((unsigned)((keys + INK2G_SC_KEYS - 1) / INK2G_SC_KEYS), heads, n), INK2G_SC_KEYS,
                                 0, (cudaStream_t)stream>>>(q, rel, table, kc, cap, posd, pos0, window, extent,
                                                            mask_beyond, scr, scr_ld, heads, kv_heads, dim, rank, r0); break;
  }
      const dim3 gx((unsigned)((keys + 255) / 256), heads, n), gh(n, heads),
          go(n, heads, (dim + (R == 1 ? INK2G_FOLD_OUT : dim) - 1) / (R == 1 ? INK2G_FOLD_OUT : dim));
      // a decode row spreads a head's outputs over blocks of INK2G_FOLD_OUT; a chunk has blocks enough
      const int od = R == 1 ? INK2G_FOLD_OUT : dim;
      cudaStream_t sm = (cudaStream_t)stream;
      switch (prec) {
        case 1:
          ink2g_attn_top_kernel<float, false><<<gh, 256, 0, sm>>>(sink, posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_exp_kernel<float, false><<<gx, 256, 0, sm>>>(posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_out_kernel<float, false, KV><<<go, 256, 0, sm>>>(sink, vc, cap, posd, pos0, window, scr, scr_ld, out,
                                                                     heads, kv_heads, dim, r0, od);
          break;
        case 2:
          ink2g_attn_top_kernel<float, true><<<gh, 256, 0, sm>>>(sink, posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_exp_kernel<float, true><<<gx, 256, 0, sm>>>(posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_out_kernel<float, true, KV><<<go, 256, 0, sm>>>(sink, vc, cap, posd, pos0, window, scr, scr_ld, out,
                                                                    heads, kv_heads, dim, r0, od);
          break;
        default:
          ink2g_attn_top_kernel<double, false><<<gh, 256, 0, sm>>>(sink, posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_exp_kernel<double, false><<<gx, 256, 0, sm>>>(posd, pos0, window, scr, scr_ld, heads, r0);
          ink2g_attn_out_kernel<double, false, KV><<<go, 256, 0, sm>>>(sink, vc, cap, posd, pos0, window, scr, scr_ld,
                                                                      out, heads, kv_heads, dim, r0, od);
          break;
      }
    }
    return ink2g_rc();
  }
  switch (prec) {
    case 1: ink2g_attn_kernel<float, false, KV><<<dim3(R, heads), 256, 0, (cudaStream_t)stream>>>(
      q, rel, table, sink, kc, vc, cap, pos0, window, extent, mask_beyond, out, heads, kv_heads, dim, rank, desc); break;
    case 2: ink2g_attn_kernel<float, true, KV><<<dim3(R, heads), 256, 0, (cudaStream_t)stream>>>(
      q, rel, table, sink, kc, vc, cap, pos0, window, extent, mask_beyond, out, heads, kv_heads, dim, rank, desc); break;
    default: ink2g_attn_kernel<double, false, KV><<<dim3(R, heads), 256, 0, (cudaStream_t)stream>>>(
      q, rel, table, sink, kc, vc, cap, pos0, window, extent, mask_beyond, out, heads, kv_heads, dim, rank, desc); break;
  }
  return ink2g_rc();
}

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
  if (single && !tiles && fmt == INK2G_W_NVFP4 && !round_bf16 && ldx >= k) {
    // one input row over dense NVFP4 weights: the bandwidth GEMV (the same bits, see the kernel)
    const float* xs1[1] = {(const float*)x};
    const void* ws1[1] = {w};
    const void* ss1[1] = {scales};
    const float* as1[1] = {amax};
    const float ds1[1] = {denom};
    float* ys1[1] = {(float*)y};
    const int ns1[1] = {n};
    const int rc = ink2g_gemv_nvfp4_jobs(xs1, ws1, ss1, as1, ds1, ys1, ns1, 1, k, (cudaStream_t)stream);
    if (rc != -1) return rc;
  }
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
  if (!tiles && (fmt == INK2G_W_BF16 || fmt == INK2G_W_NVFP4) && !round_bf16 && m >= 2 && m <= 8) {
    // 2..8 input rows: the multi-row GEMV, one weight read for the rows (the same bits per row)
    const int rc = ink2g_gemv_rows((const float*)x, ldx, w, (const uint8_t*)scales, amax, denom, fmt == INK2G_W_NVFP4,
                                   (float*)y, ldy, m, n, k, (cudaStream_t)stream);
    if (rc != -1) return rc;
  }
  if (!tiles && (fmt == INK2G_W_BF16 || fmt == INK2G_W_NVFP4) && !round_bf16 && (k % 16) == 0 && (ldx % 4) == 0 &&
      ((uintptr_t)x & 15) == 0 && ((uintptr_t)w & 15) == 0 && (m + INK2G_SG_BM - 1) / INK2G_SG_BM <= 65535) {
    // the prefill GEMM over BF16 or dense NVFP4 weights (the same bits, see the kernel)
    const dim3 g((n + INK2G_SG_BN - 1) / INK2G_SG_BN, (m + INK2G_SG_BM - 1) / INK2G_SG_BM);
    if (fmt == INK2G_W_NVFP4)
      ink2g_sgemm_bf16_kernel<true><<<g, 256, 0, (cudaStream_t)stream>>>((const float*)x, ldx, w, (const uint8_t*)scales,
                                                                      amax, denom, (float*)y, ldy, m, n, k);
    else
      ink2g_sgemm_bf16_kernel<false><<<g, 256, 0, (cudaStream_t)stream>>>((const float*)x, ldx, w, nullptr, nullptr, 1.0f,
                                                                       (float*)y, ldy, m, n, k);
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

// The #1397 tile GEMM over BF16 weights, rows 0..m: the gate's oracle for the prefill GEMM's bits.
int memra_ink2g_gemm_tile_v0(const void* x, int64_t ldx, const void* w, const void* scales, const float* amax,
                             float denom, int fmt, void* y, int64_t ldy, int m, int n, int k, void* stream) {
  if (!x || !w || !y || m <= 0 || n <= 0 || k <= 0) return 45045;
  const dim3 block(16, 16);
  const dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
  ink2g_gemm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>((const float*)x, ldx, (const uint8_t*)w,
                                                              (const uint8_t*)scales, 0, 0, amax, denom, fmt, 0, nullptr,
                                                              1, (float*)y, ldy, m, n, k);
  return ink2g_rc();
}

// Several dense NVFP4 decode GEMVs of the same input width in one launch (the bits of each alone).
int memra_ink2g_gemv_nvfp4_jobs(const float* const* xs, const void* const* ws, const void* const* scs,
                                const float* const* amaxs, const float* denoms, float* const* ys, const int* ns,
                                int count, int k, void* stream) {
  if (!xs || !ws || !scs || !amaxs || !denoms || !ys || !ns) return 45047;
  const int rc = ink2g_gemv_nvfp4_jobs(xs, ws, scs, amaxs, denoms, ys, ns, count, k, (cudaStream_t)stream);
  return rc == -1 ? 45048 : rc;
}

int memra_ink2g_embed(const void* table, const uint32_t* ids, const float* norm_w, double eps, float* out, int R,
                      int H, int prec, void* stream) {
  if (!table || !ids || !norm_w || !out || R <= 0 || H <= 0) return 45011;
  switch (prec) {
    case 1: ink2g_embed_kernel<float, false><<<R, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, ids, norm_w, eps, out, H); break;
    case 2: ink2g_embed_kernel<float, true><<<R, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, ids, norm_w, eps, out, H); break;
    default: ink2g_embed_kernel<double, false><<<R, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, ids, norm_w, eps, out, H); break;
  }
  return ink2g_rc();
}

int memra_ink2g_norm(const float* x, const float* norm_w, double eps, float* y, int R, int H, int prec, void* stream) {
  if (!x || !norm_w || !y || R <= 0 || H <= 0) return 45012;
  switch (prec) {
    case 1: ink2g_norm_kernel<float, false><<<R, 256, 0, (cudaStream_t)stream>>>(x, norm_w, eps, y, H); break;
    case 2: ink2g_norm_kernel<float, true><<<R, 256, 0, (cudaStream_t)stream>>>(x, norm_w, eps, y, H); break;
    default: ink2g_norm_kernel<double, false><<<R, 256, 0, (cudaStream_t)stream>>>(x, norm_w, eps, y, H); break;
  }
  return ink2g_rc();
}

// scratch: at least R * nq * ns * 2 doubles.
int memra_ink2g_mix(const float* const* states, int ns, const float* queries, int nq, double eps, float* out,
                    int64_t out_stride, int R, int H, double* scratch, int64_t scratch_len, int prec, void* stream) {
  if (!states || ns <= 0 || ns > INK2G_MAX_STATES || !queries || nq <= 0 || !out || R <= 0 || H <= 0) return 45013;
  if (!scratch || (int64_t)R * nq * ns * 2 > scratch_len || nq > 65535 || R > 65535) return 45043;
  Ink2gStates st;
  for (int s = 0; s < INK2G_MAX_STATES; s++) st.p[s] = s < ns ? states[s] : nullptr;
  switch (prec) {
    case 1: ink2g_mix_dot_kernel<float, false><<<dim3(ns, nq, R), 256, 0, (cudaStream_t)stream>>>(st, ns, queries, nq, eps, scratch, H); break;
    case 2: ink2g_mix_dot_kernel<float, true><<<dim3(ns, nq, R), 256, 0, (cudaStream_t)stream>>>(st, ns, queries, nq, eps, scratch, H); break;
    default: ink2g_mix_dot_kernel<double, false><<<dim3(ns, nq, R), 256, 0, (cudaStream_t)stream>>>(st, ns, queries, nq, eps, scratch, H); break;
  }
  switch (prec) {
    case 1: ink2g_mix_out_kernel<float, false><<<dim3((H + 127) / 128, nq, R), 128, 0, (cudaStream_t)stream>>>(st, ns, nq, scratch, out,
                                                                                    out_stride, H); break;
    case 2: ink2g_mix_out_kernel<float, true><<<dim3((H + 127) / 128, nq, R), 128, 0, (cudaStream_t)stream>>>(st, ns, nq, scratch, out,
                                                                                    out_stride, H); break;
    default: ink2g_mix_out_kernel<double, false><<<dim3((H + 127) / 128, nq, R), 128, 0, (cudaStream_t)stream>>>(st, ns, nq, scratch, out,
                                                                                    out_stride, H); break;
  }
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
                           int prec, void* stream) {
  if (!x || !msg || !down || !forget || !gain || !out || R <= 0 || H <= 0 || rank <= 0 || rank > 128) return 45014;
  if (!scratch || (int64_t)R * rank > scratch_len || R > 65535) return 45044;
  switch (prec) {
    case 1: ink2g_ctrl_down_kernel<float, false><<<dim3(rank, R), 256, 0, (cudaStream_t)stream>>>(x, msg, (const uint16_t*)down, eps, scratch,
                                                                         H, rank); break;
    case 2: ink2g_ctrl_down_kernel<float, true><<<dim3(rank, R), 256, 0, (cudaStream_t)stream>>>(x, msg, (const uint16_t*)down, eps, scratch,
                                                                         H, rank); break;
    default: ink2g_ctrl_down_kernel<double, false><<<dim3(rank, R), 256, 0, (cudaStream_t)stream>>>(x, msg, (const uint16_t*)down, eps, scratch,
                                                                         H, rank); break;
  }
  switch (prec) {
    case 1: ink2g_ctrl_up_kernel<float, false><<<dim3((H + 63) / 64, R), 64, 0, (cudaStream_t)stream>>>(
      x, msg, (const uint16_t*)forget, (const uint16_t*)gain, scratch, out, H, rank); break;
    case 2: ink2g_ctrl_up_kernel<float, true><<<dim3((H + 63) / 64, R), 64, 0, (cudaStream_t)stream>>>(
      x, msg, (const uint16_t*)forget, (const uint16_t*)gain, scratch, out, H, rank); break;
    default: ink2g_ctrl_up_kernel<double, false><<<dim3((H + 63) / 64, R), 64, 0, (cudaStream_t)stream>>>(
      x, msg, (const uint16_t*)forget, (const uint16_t*)gain, scratch, out, H, rank); break;
  }
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

// posd (nullable): the device copy of pos0, read by the kernel (a graph replays one step at every
// position); pos0 must still be the call's position. desc (nullable): R rows of a batched decode
// call (each row's position; which 1 / 2 writes the row's K / V cache, which 0 writes dst).
int memra_ink2g_heads(int mode, const float* x, const float* scale, double thr, double eps, const int64_t* posd,
                      int64_t pos0, int log_floor, double alpha, float* dst, int64_t cap, int R, int heads,
                      int width, int prec, const void* desc, int which, void* stream) {
  if ((mode != 0 && mode != 1) || !x || (!dst && !(desc && which != 0)) || R <= 0 || heads <= 0 || width <= 0 ||
      width > 256)
    return 45015;
  if (heads > 65535) return 45016;
  const int threads = ((width + 31) / 32) * 32;
  switch (prec) {
    case 1: ink2g_heads_kernel<float, false><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(mode, x, scale, thr, eps, posd, pos0,
                                                                         log_floor, alpha, dst, cap, heads, width, (const Ink2gRow*)desc, which); break;
    case 2: ink2g_heads_kernel<float, true><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(mode, x, scale, thr, eps, posd, pos0,
                                                                         log_floor, alpha, dst, cap, heads, width, (const Ink2gRow*)desc, which); break;
    default: ink2g_heads_kernel<double, false><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(mode, x, scale, thr, eps, posd, pos0,
                                                                         log_floor, alpha, dst, cap, heads, width, (const Ink2gRow*)desc, which); break;
  }
  return ink2g_rc();
}

// state_in: the history before the call; state_out (nullable): the history after it, a different
// buffer, so a call that fails later leaves the caller's committed history untouched. A null
// state_out drops the history (the gate's red arm).
// desc (nullable): R rows of a batched decode call, each its own sequence's one-row call; which
// names the history (0 K, 1 V, 2 attention output, 3 MLP output) and state_in / state_out are unused.
int memra_ink2g_conv(int mode, const float* x, const float* state_in, float* state_out, const float* taps,
                     const float* dyn, int groups, const float* add, float* out, int R, int C, int K, int prec,
                     const void* desc, int which, void* stream) {
  if ((mode != 0 && mode != 1) || !x || (!state_in && !desc) || !taps || !out || R <= 0 || C <= 0) return 45017;
  if (!desc && state_out == state_in) return 45041;
  if (desc && (which < 0 || which > 3 || R > 65535)) return 45042;
  if (K < 2 || K > INK2G_MAX_KERNEL) return 45018;
  if (mode == 1 && (!dyn || groups <= 0 || C % groups != 0)) return 45019;
  if (x == out) return 45020;
  switch (prec) {
    case 1: ink2g_conv_kernel<float, false><<<dim3((C + 63) / 64, desc ? R : 1), 64, 0, (cudaStream_t)stream>>>(mode, x, state_in, state_out, taps, dyn, groups,
                                                                     add, out, R, C, K, (const Ink2gRow*)desc, which); break;
    case 2: ink2g_conv_kernel<float, true><<<dim3((C + 63) / 64, desc ? R : 1), 64, 0, (cudaStream_t)stream>>>(mode, x, state_in, state_out, taps, dyn, groups,
                                                                     add, out, R, C, K, (const Ink2gRow*)desc, which); break;
    default: ink2g_conv_kernel<double, false><<<dim3((C + 63) / 64, desc ? R : 1), 64, 0, (cudaStream_t)stream>>>(mode, x, state_in, state_out, taps, dyn, groups,
                                                                     add, out, R, C, K, (const Ink2gRow*)desc, which); break;
  }
  return ink2g_rc();
}

// posd (nullable): the device copy of pos0. scr (nullable): f64 scratch [heads][scr_ld] with scr_ld at
// least the keys a row attends (the window, or the context); with it a call of one row runs the split
// form (the same bits).

// kc / vc: f32 rows, or e4m3 bytes when kv_e4m3 (the nvfp4-rig cache; each byte read back exactly).
int memra_ink2g_attn(const float* q, const float* rel, const float* table, const float* sink, const void* kc,
                     const void* vc, int kv_e4m3, int64_t cap, const int64_t* posd, int64_t pos0, int window,
                     int extent, int mask_beyond, float* out, int R, int heads, int kv_heads, int dim, int rank,
                     double* scr, int64_t scr_ld, int scr_rows, int prec, const void* desc, void* stream) {
  const Ink2gRow* rows = (const Ink2gRow*)desc;
  if (kv_e4m3)
    return ink2g_attn_any<uint8_t>(q, rel, table, sink, (const uint8_t*)kc, (const uint8_t*)vc, cap, posd, pos0, window,
                                   extent, mask_beyond, out, R, heads, kv_heads, dim, rank, scr, scr_ld, scr_rows, prec,
                                   rows, stream);
  return ink2g_attn_any<float>(q, rel, table, sink, (const float*)kc, (const float*)vc, cap, posd, pos0, window, extent,
                               mask_beyond, out, R, heads, kv_heads, dim, rank, scr, scr_ld, scr_rows, prec, rows, stream);
}

// The tiled f64 attention over an e4m3 cache (the gate's cell).
int memra_ink2g_attn_e4m3(const float* q, const float* rel, const float* table, const float* sink,
                          const uint8_t* kc, const uint8_t* vc, int64_t cap, int64_t pos0, int window, int extent,
                          int mask_beyond, float* out, int R, int heads, int kv_heads, int dim, int rank,
                          void* stream) {
  return ink2g_attn_any<uint8_t>(q, rel, table, sink, kc, vc, cap, nullptr, pos0, window, extent, mask_beyond, out, R,
                                 heads, kv_heads, dim, rank, nullptr, 0, 0, 0, nullptr, stream);
}

// The K / V epilogue into an e4m3 cache (nvfp4-rig): K (mode 0, no scale, no log scale) and V (mode 1).
int memra_ink2g_heads_e4m3(int mode, const float* x, double thr, double eps, const int64_t* posd, int64_t pos0,
                           uint8_t* dst, int64_t cap, int R, int heads, int width, int prec, const void* desc,
                           int which, void* stream) {
  if ((mode != 0 && mode != 1) || !x || (!dst && !(desc && which != 0)) || R <= 0 || heads <= 0 || width <= 0 ||
      width > 256)
    return 45017;
  if (heads > 65535 || (cap <= 0 && !desc)) return 45018;
  const int threads = ((width + 31) / 32) * 32;
  switch (prec) {
    case 1:
      ink2g_heads_kernel<float, false, uint8_t><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(
          mode, x, nullptr, (float)thr, (float)eps, posd, pos0, 0, 0.0f, dst, cap, heads, width, (const Ink2gRow*)desc, which);
      break;
    case 2:
      ink2g_heads_kernel<float, true, uint8_t><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(
          mode, x, nullptr, (float)thr, (float)eps, posd, pos0, 0, 0.0f, dst, cap, heads, width, (const Ink2gRow*)desc, which);
      break;
    default:
      ink2g_heads_kernel<double, false, uint8_t><<<dim3(R, heads), threads, 0, (cudaStream_t)stream>>>(
          mode, x, nullptr, thr, eps, posd, pos0, 0, 0.0, dst, cap, heads, width, (const Ink2gRow*)desc, which);
  }

  return ink2g_rc();
}

int memra_ink2g_gate(float* y, const float* g, int64_t n, int prec, void* stream) {
  if (!y || !g || n <= 0) return 45031;
  switch (prec) {
    case 1: ink2g_gate_kernel<float, false><<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, g, n); break;
    case 2: ink2g_gate_kernel<float, true><<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, g, n); break;
    default: ink2g_gate_kernel<double, false><<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, g, n); break;
  }
  return ink2g_rc();
}

int memra_ink2g_route(const float* logits, const float* bias, double scale, int top_k, int tie_high, uint32_t* ids,
                      float* wts, float* margin, int R, int E, int prec, void* stream) {
  if (!logits || !ids || !wts || R <= 0 || E <= 0 || E > 1024 || top_k <= 0 || top_k > 64 || top_k > E)
    return 45032;
  switch (prec) {
    case 1: ink2g_route_kernel<float, false><<<R, 128, 0, (cudaStream_t)stream>>>(logits, bias, scale, top_k, tie_high, ids, wts, margin,
                                                        E); break;
    case 2: ink2g_route_kernel<float, true><<<R, 128, 0, (cudaStream_t)stream>>>(logits, bias, scale, top_k, tie_high, ids, wts, margin,
                                                        E); break;
    default: ink2g_route_kernel<double, false><<<R, 128, 0, (cudaStream_t)stream>>>(logits, bias, scale, top_k, tie_high, ids, wts, margin,
                                                        E); break;
  }
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
int memra_ink2g_head(float* logits, double cap, uint32_t* best, int R, int n, int prec, void* stream) {
  if (!logits || R <= 0 || n <= 0) return 45037;
  const int64_t total = (int64_t)R * n;
  if (cap > 0.0)
    switch (prec) {
    case 1: ink2g_softcap_kernel<float, false><<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(logits, cap, total); break;
    case 2: ink2g_softcap_kernel<float, true><<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(logits, cap, total); break;
    default: ink2g_softcap_kernel<double, false><<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(logits, cap, total); break;
  }
  if (best) ink2g_argmax_kernel<<<R, 1024, 0, (cudaStream_t)stream>>>(logits, best, n);
  return ink2g_rc();
}

int memra_ink2g_tiles(const uint32_t* ids, uint32_t* tiles, int pairs, void* stream) {
  if (!ids || !tiles || pairs <= 0) return 45038;
  ink2g_tiles_kernel<<<(pairs + 63) / 64, 64, 0, (cudaStream_t)stream>>>(ids, tiles, pairs);
  return ink2g_rc();
}

int memra_ink2g_groups_dev(const uint32_t* ids, uint32_t* recs, int pairs, int words, int cap, void* stream) {
  if (!ids || !recs || pairs <= 0 || pairs > 256 || cap <= 0 || words != 2 + cap) return 45046;
  ink2g_groups_dev_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(ids, recs, pairs, words, cap);
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
