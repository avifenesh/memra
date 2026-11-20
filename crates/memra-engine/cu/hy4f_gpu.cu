// Hy4-flash native checkpoint runner kernels (`crate::hy4f_gpu`, binary `hy4f-run`).
//
// Two kernels, correctness first; speed is the later GPU route's:
//
// - `memra_hy4f_gemm`: Y[m, n] = sum_k X[m, k] * W[n, k] in f32, the weight read in its stored
//   format: BF16, F32, FP8 E4M3 with a 128x128 F32 block-scale grid (`weight_scale_inv`, multiplied
//   in), or NVFP4 (U8 e2m1 pairs, element 2i in the low nibble, an E4M3 scale per 16 inputs and one
//   F32 `weight_scale_2`). The dequantized weight is never formed: the codes multiply the input
//   (the codes are exact; products and sums round in f32), a scale multiplies each partial sum
//   (per 16 inputs for NVFP4, per 128 for FP8 blocks) and `weight_scale_2` multiplies the finished
//   sum, so no weight is rounded through f32. Each output is one thread's sequential sum over k,
//   so a row's result does not depend on the other rows of the call. `round_bf16` is the red arm
//   of the checkpoint gate: it multiplies by bf16(the dequantized weight), as a BF16 loader would.
// - `memra_hy4f_attn`: gated-MLA attention core over an explicit key-slot list per query (the
//   sliding window, the causal prefix or the indexer's recalled tokens), with the learned sink in
//   the softmax denominator (`eager_attention_forward` with `s_aux`), all in f32: a score is
//   (q.k) * scale, the softmax subtracts the row maximum (sink included) and the output is the
//   sequential f32 sum of probability times value. One block owns one (query, head), so a row's
//   result does not depend on the other rows of the call.
//
// Return codes: 0 ok, 10000 + cudaError_t, 44xxx for a refused contract.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define HY4F_W_BF16 0
#define HY4F_W_F32 1
#define HY4F_W_FP8_BLOCK 2
#define HY4F_W_NVFP4 3

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__device__ __forceinline__ float hy4f_e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

__device__ __forceinline__ float hy4f_e2m1(uint8_t nib) {
  const float t[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return t[nib & 15];
}

// One 64x64 output tile per block, 16x16 threads, 4x4 outputs a thread. K advances 16 at a
// time; the weight tile holds the codes (or the plain values) and `ws` the scale of each of its
// 64 rows for this step.
__global__ void hy4f_gemm_kernel(const float* __restrict__ x, int64_t ldx, const uint8_t* __restrict__ w,
                                 const uint8_t* __restrict__ sc, int grid_cols, float scale2, int fmt,
                                 int round_bf16, float* __restrict__ y, int64_t ldy, int m, int n, int k) {
  __shared__ float xs[TILE_K][TILE_M + 1];
  __shared__ float wt[TILE_K][TILE_N + 1];
  __shared__ float ws[TILE_N];
  const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  const int m0 = blockIdx.y * TILE_M, n0 = blockIdx.x * TILE_N;
  float acc[4][4], part[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = part[i][j] = 0.0f;

  for (int k0 = 0; k0 < k; k0 += TILE_K) {
    // stage x: 64 rows x 16 cols = 1024 values, 4 a thread
    for (int e = tid; e < TILE_M * TILE_K; e += 256) {
      int r = e / TILE_K, c = e % TILE_K;
      int gm = m0 + r, gk = k0 + c;
      xs[c][r] = (gm < m && gk < k) ? x[(int64_t)gm * ldx + gk] : 0.0f;
    }
    // stage w codes: 64 rows x 16 cols
    for (int e = tid; e < TILE_N * TILE_K; e += 256) {
      int r = e / TILE_K, c = e % TILE_K;
      int gn = n0 + r, gk = k0 + c;
      float v = 0.0f;
      if (gn < n && gk < k) {
        if (fmt == HY4F_W_BF16) {
          uint16_t b = ((const uint16_t*)w)[(int64_t)gn * k + gk];
          v = __uint_as_float(((uint32_t)b) << 16);
        } else if (fmt == HY4F_W_F32) {
          v = ((const float*)w)[(int64_t)gn * k + gk];
        } else if (fmt == HY4F_W_FP8_BLOCK) {
          v = hy4f_e4m3(w[(int64_t)gn * k + gk]);
        } else {
          uint8_t b = w[(int64_t)gn * (k / 2) + gk / 2];
          v = hy4f_e2m1((gk & 1) ? (b >> 4) : b);
        }
        if (round_bf16) {
          float s = 1.0f;
          if (fmt == HY4F_W_FP8_BLOCK) s = ((const float*)sc)[(int64_t)(gn / 128) * grid_cols + gk / 128];
          if (fmt == HY4F_W_NVFP4) s = hy4f_e4m3(sc[(int64_t)gn * (k / 16) + gk / 16]) * scale2;
          v = __bfloat162float(__float2bfloat16_rn(v * s));
        }
      }
      wt[c][r] = v;
    }
    if (tid < TILE_N) {
      int gn = n0 + tid;
      float s = 1.0f;
      if (gn < n) {
        if (fmt == HY4F_W_FP8_BLOCK) {
          s = ((const float*)sc)[(int64_t)(gn / 128) * grid_cols + k0 / 128];
        } else if (fmt == HY4F_W_NVFP4) {
          s = hy4f_e4m3(sc[(int64_t)gn * (k / 16) + k0 / 16]);
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
        for (int j = 0; j < 4; j++) part[i][j] += a[i] * b[j];
    }
    // fold the partial sums at a scale boundary: every step for NVFP4 (16 inputs) and the plain
    // formats, every 128 inputs (or at the end) for FP8 blocks
    bool fold = fmt != HY4F_W_FP8_BLOCK || ((k0 + TILE_K) % 128 == 0) || (k0 + TILE_K >= k);
    if (fold) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bool plain = fmt == HY4F_W_BF16 || fmt == HY4F_W_F32 || round_bf16;
        float s = plain ? 1.0f : ws[tx * 4 + j];
#pragma unroll
        for (int i = 0; i < 4; i++) {
          acc[i][j] += plain ? part[i][j] : part[i][j] * s;
          part[i][j] = 0.0f;
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int gm = m0 + ty * 4 + i;
    if (gm >= m) continue;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int gn = n0 + tx * 4 + j;
      if (gn >= n) continue;
      float v = acc[i][j];
      if (fmt == HY4F_W_NVFP4 && !round_bf16) v *= scale2;
      y[(int64_t)gm * ldy + gn] = v;
    }
  }
}

extern "C" int memra_hy4f_gemm(const void* x, int64_t ldx, const void* w, const void* scales, int grid_cols,
                               float scale2, int fmt, int round_bf16, void* y, int64_t ldy, int m, int n, int k,
                               void* stream) {
  if (m < 0 || n <= 0 || k <= 0 || fmt < 0 || fmt > 3) return 44001;
  if (fmt == HY4F_W_NVFP4 && (k % 16) != 0) return 44002;
  if ((fmt == HY4F_W_FP8_BLOCK || fmt == HY4F_W_NVFP4) && scales == nullptr) return 44003;
  if (m == 0) return 0;
  dim3 block(16, 16);
  dim3 grid((n + TILE_N - 1) / TILE_N, (m + TILE_M - 1) / TILE_M);
  if (grid.y > 65535) return 44004;
  hy4f_gemm_kernel<<<grid, block, 0, (cudaStream_t)stream>>>((const float*)x, ldx, (const uint8_t*)w,
                                                             (const uint8_t*)scales, grid_cols, scale2, fmt,
                                                             round_bf16, (float*)y, ldy, m, n, k);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

#define ATTN_THREADS 256
#define ATTN_MAX_KEYS 4096

// One block per (query, head). q [T, H, dk]; k [S, H, dk]; v [S, H, dv]; slots [T, L] (-1 ends
// or pads a row); out [T, H, dv].
__global__ void hy4f_attn_kernel(const float* __restrict__ q, const float* __restrict__ kc,
                                 const float* __restrict__ vc, const float* __restrict__ sinks,
                                 const int* __restrict__ slots, int L, int H, int dk, int dv, float scale,
                                 float* __restrict__ out) {
  __shared__ float qs[512];
  __shared__ float sc[ATTN_MAX_KEYS];
  __shared__ int sl[ATTN_MAX_KEYS];
  __shared__ float red[ATTN_THREADS];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  for (int d = tid; d < dk; d += ATTN_THREADS) qs[d] = q[((int64_t)t * H + h) * dk + d];
  for (int j = tid; j < L; j += ATTN_THREADS) sl[j] = slots[(int64_t)t * L + j];
  __syncthreads();
  float mx = sinks[h];
  for (int j = tid; j < L; j += ATTN_THREADS) {
    int s = sl[j];
    float v = -INFINITY;
    if (s >= 0) {
      const float* kr = kc + ((int64_t)s * H + h) * dk;
      float dot = 0.0f;
      for (int d = 0; d < dk; d++) dot += qs[d] * kr[d];
      v = dot * scale;
    }
    sc[j] = v;
    mx = fmaxf(mx, v);
  }
  red[tid] = mx;
  __syncthreads();
  for (int o = ATTN_THREADS / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (int j = tid; j < L; j += ATTN_THREADS) {
    float e = sl[j] >= 0 ? expf(sc[j] - mx) : 0.0f;
    sc[j] = e;
    sum += e;
  }
  red[tid] = sum;
  __syncthreads();
  for (int o = ATTN_THREADS / 2; o > 0; o >>= 1) {
    if (tid < o) red[tid] += red[tid + o];
    __syncthreads();
  }
  float denom = red[0] + expf(sinks[h] - mx);
  __syncthreads();
  for (int j = tid; j < L; j += ATTN_THREADS) sc[j] = sc[j] / denom;
  __syncthreads();
  for (int d = tid; d < dv; d += ATTN_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < L; j++) {
      int s = sl[j];
      if (s < 0) continue;
      acc += sc[j] * vc[((int64_t)s * H + h) * dv + d];
    }
    out[((int64_t)t * H + h) * dv + d] = acc;
  }
}

extern "C" int memra_hy4f_attn(const void* q, const void* k, const void* v, const void* sinks, const void* slots,
                               int t, int l, int h, int dk, int dv, float scale, void* out, void* stream) {
  if (t < 0 || l < 0 || l > ATTN_MAX_KEYS || h <= 0 || dk <= 0 || dk > 512 || dv <= 0) return 44101;
  if (t == 0) return 0;
  dim3 grid(t, h);
  hy4f_attn_kernel<<<grid, ATTN_THREADS, 0, (cudaStream_t)stream>>>(
      (const float*)q, (const float*)k, (const float*)v, (const float*)sinks, (const int*)slots, l, h, dk, dv,
      scale, (float*)out);
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}
