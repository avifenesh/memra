// Inkling 2 Mini DSpark drafter on one sm_120a card: the kernels of crate::inkling2_dspark.
// The program is the typed plan's (memra_gguf::model_plan::inkling2::Inkling2DrafterPlan) and the
// reference is memra_reference::inkling2_drafter (f64), operation for operation.
//
// Numeric class: BF16 weights as stored, f32 activations between operations; norm reductions,
// the RoPE angles and the fold run in f64; the attention softmax is an f32 online softmax. A proposal never changes a committed token (the
// verify forward is the trunk's own program), so this class decides acceptance only.
//
// Return codes: 0 ok, 10000 + cudaError_t, 46xxx for a refused contract.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float dk_bf16(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }

int dk_rc() {
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

__device__ __forceinline__ float dk_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ double dk_warp_sum_d(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over a block of up to 1024 threads, every thread gets the result.
__device__ double dk_block_sum(double v, double* shared) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
  v = dk_warp_sum_d(v);
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  double s = 0.0;
  for (int w = 0; w < warps; w++) s += shared[w];
  return s;
}

#define DK_MR 8

// y[r, n] = (acc ? y[r, n] : 0) + (bias ? bias[n] : 0) + sum_k x[r, k] * w[n, k]
// One warp per output column, eight per block; each lane reads 8 BF16 weights at a time.
// (dk_gemv_f32w_kernel is the same over f32 weights: the folded recurrent head.)
__global__ void dk_gemv_kernel(const float* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w,
                               int64_t ldw, const float* __restrict__ bias, float* __restrict__ y, int64_t ldy,
                               int m, int n, int k, int acc) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * 8 + warp;
  if (col >= n) return;
  const uint16_t* wr = w + (int64_t)col * ldw;
  for (int r0 = blockIdx.y * DK_MR; r0 < m; r0 += gridDim.y * DK_MR) {
    float s[DK_MR];
#pragma unroll
    for (int i = 0; i < DK_MR; i++) s[i] = 0.f;
    for (int kk = lane * 8; kk < k; kk += 256) {
      const uint4 pk = *reinterpret_cast<const uint4*>(wr + kk);
      float wv[8];
      const uint32_t words[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        wv[2 * j] = __uint_as_float(words[j] << 16);
        wv[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
      }
#pragma unroll
      for (int i = 0; i < DK_MR; i++) {
        if (r0 + i < m) {
          const float4* xp = reinterpret_cast<const float4*>(x + (int64_t)(r0 + i) * ldx + kk);
          const float4 a = xp[0], b = xp[1];
          float t = s[i];
          t = fmaf(wv[0], a.x, t);
          t = fmaf(wv[1], a.y, t);
          t = fmaf(wv[2], a.z, t);
          t = fmaf(wv[3], a.w, t);
          t = fmaf(wv[4], b.x, t);
          t = fmaf(wv[5], b.y, t);
          t = fmaf(wv[6], b.z, t);
          t = fmaf(wv[7], b.w, t);
          s[i] = t;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < DK_MR; i++) {
      const float v = dk_warp_sum(s[i]);
      if (lane == 0 && r0 + i < m) {
        float* out = y + (int64_t)(r0 + i) * ldy + col;
        *out = (acc ? *out : 0.f) + (bias ? bias[col] : 0.f) + v;
      }
    }
  }
}

__global__ void dk_gemv_f32w_kernel(const float* __restrict__ x, const float* __restrict__ w, int64_t ldw,
                                    const float* __restrict__ bias, float* __restrict__ y, int n, int k) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * 8 + warp;
  if (col >= n) return;
  const float* wr = w + (int64_t)col * ldw;
  float s = 0.f;
  for (int kk = lane * 4; kk < k; kk += 128) {
    const float4 a = *reinterpret_cast<const float4*>(wr + kk);
    const float4 b = *reinterpret_cast<const float4*>(x + kk);
    s = fmaf(a.x, b.x, s);
    s = fmaf(a.y, b.y, s);
    s = fmaf(a.z, b.z, s);
    s = fmaf(a.w, b.w, s);
  }
  s = dk_warp_sum(s);
  if (lane == 0) y[col] = (bias ? bias[col] : 0.f) + s;
}

// out[r] = rms(sum_t alpha[t] * taps[t][r]) * w; the mix in f32 (t in order), the norm in f64.
__global__ void dk_mix_norm_kernel(const float* __restrict__ taps, int64_t tap_stride, int64_t row_stride, int ntaps,
                                   const float* __restrict__ alpha, const float* __restrict__ w, double eps,
                                   float* __restrict__ out, int h) {
  __shared__ double red[32];
  extern __shared__ float mixed[];
  const int r = blockIdx.x;
  double ss = 0.0;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float m = 0.f;
    for (int t = 0; t < ntaps; t++) m = fmaf(alpha[t], taps[t * tap_stride + (int64_t)r * row_stride + i], m);
    mixed[i] = m;
    ss += (double)m * (double)m;
  }
  const double total = dk_block_sum(ss, red);
  const float f = (float)(1.0 / sqrt(total / h + eps));
  for (int i = threadIdx.x; i < h; i += blockDim.x) out[(int64_t)r * h + i] = mixed[i] * f * w[i];
}

// x[r] += add[r] (when add), then y[r] = rms(x[r]) * w.
__global__ void dk_norm_kernel(float* __restrict__ x, const float* __restrict__ add, const float* __restrict__ w,
                               double eps, float* __restrict__ y, int h) {
  __shared__ double red[32];
  const int r = blockIdx.x;
  float* xr = x + (int64_t)r * h;
  double ss = 0.0;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    float v = xr[i];
    if (add) {
      v += add[(int64_t)r * h + i];
      xr[i] = v;
    }
    ss += (double)v * (double)v;
  }
  const double total = dk_block_sum(ss, red);
  const float f = (float)(1.0 / sqrt(total / h + eps));
  for (int i = threadIdx.x; i < h; i += blockDim.x) y[(int64_t)r * h + i] = xr[i] * f * w[i];
}

// Normed anchor embedding: out = rms(table[id]) * w.
__global__ void dk_embed_kernel(const uint16_t* __restrict__ table, const uint32_t* __restrict__ id,
                                const float* __restrict__ w, double eps, float* __restrict__ out, int h) {
  __shared__ double red[32];
  const uint16_t* row = table + (int64_t)(*id) * h;
  double ss = 0.0;
  for (int i = threadIdx.x; i < h; i += blockDim.x) {
    const double v = dk_bf16(row[i]);
    ss += v * v;
  }
  const double total = dk_block_sum(ss, red);
  const float f = (float)(1.0 / sqrt(total / h + eps));
  for (int i = threadIdx.x; i < h; i += blockDim.x) out[i] = dk_bf16(row[i]) * f * w[i];
}

// Per-head RMS (when norm) then NeoX RoPE at position pos0 + row, in place; angles in f64.
// grid (rows, heads), block d / 2 threads.
__global__ void dk_head_rope_kernel(float* __restrict__ x, int64_t ld, const float* __restrict__ w, int norm,
                                    double eps, int64_t pos0, double theta, int d) {
  __shared__ double red[32];
  const int r = blockIdx.x, hd = blockIdx.y, i = threadIdx.x, half = d / 2;
  const bool on = i < half;
  float* v = x + (int64_t)r * ld + (int64_t)hd * d;
  float a = on ? v[i] : 0.f, b = on ? v[i + half] : 0.f;
  if (norm) {
    const double total = dk_block_sum((double)a * a + (double)b * b, red);
    const float f = (float)(1.0 / sqrt(total / d + eps));
    if (on) {
      a = a * f * w[i];
      b = b * f * w[i + half];
    }
  }
  if (!on) return;
  const double freq = pow(theta, -(double)(2 * i) / (double)d);
  double sn, cs;
  sincos((double)(pos0 + r) * freq, &sn, &cs);
  const float c = (float)cs, s = (float)sn;
  v[i] = a * c - b * s;
  v[i + half] = b * c + a * s;
}

#define DK_ATT_WARPS 4
#define DK_MAX_DPL 8

// Block attention: query row r (position p0 + r) of head hd sees context positions first..p0-1
// (cache row pos % cap when ring, else pos) and every block row. grid (rows, heads), 4 warps.
__global__ void dk_attn_kernel(const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
                               const float* __restrict__ kc, const float* __restrict__ vc, int64_t cap, int ring,
                               int64_t first, int64_t p0, int rows, int heads, int kv_heads, int d, float scale,
                               float* __restrict__ out) {
  __shared__ float sm[DK_ATT_WARPS], sl[DK_ATT_WARPS];
  __shared__ float sacc[DK_ATT_WARPS][DK_MAX_DPL * 32];
  const int r = blockIdx.x, hd = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kvh = hd / (heads / kv_heads), kvw = kv_heads * d, qw = heads * d;
  float qv[DK_MAX_DPL], acc[DK_MAX_DPL];
  for (int j = 0; j < DK_MAX_DPL; j++) {
    const int c = lane + 32 * j;
    qv[j] = c < d ? q[(int64_t)r * qw + (int64_t)hd * d + c] * scale : 0.f;
    acc[j] = 0.f;
  }
  const int64_t nctx = p0 - first;
  const int64_t total = nctx + rows;
  float m = -INFINITY, l = 0.f;
  for (int64_t j = warp; j < total; j += DK_ATT_WARPS) {
    const float *kr, *vr;
    if (j < nctx) {
      const int64_t pos = first + j;
      const int64_t at = ring ? pos % cap : pos;
      kr = kc + at * kvw + (int64_t)kvh * d;
      vr = vc + at * kvw + (int64_t)kvh * d;
    } else {
      kr = kb + (j - nctx) * kvw + (int64_t)kvh * d;
      vr = vb + (j - nctx) * kvw + (int64_t)kvh * d;
    }
    float s = 0.f;
    for (int jj = 0; jj < DK_MAX_DPL; jj++) {
      const int c = lane + 32 * jj;
      if (c < d) s = fmaf(qv[jj], kr[c], s);
    }
    s = dk_warp_sum(s);
    const float mn = fmaxf(m, s);
    const float corr = expf(m - mn), p = expf(s - mn);
    l = l * corr + p;
    for (int jj = 0; jj < DK_MAX_DPL; jj++) {
      const int c = lane + 32 * jj;
      if (c < d) acc[jj] = acc[jj] * corr + p * vr[c];
    }
    m = mn;
  }
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
  for (int jj = 0; jj < DK_MAX_DPL; jj++) sacc[warp][lane + 32 * jj] = acc[jj];
  __syncthreads();
  if (warp == 0) {
    float M = -INFINITY;
    for (int w = 0; w < DK_ATT_WARPS; w++) M = fmaxf(M, sm[w]);
    float L = 0.f;
    float o[DK_MAX_DPL];
    for (int jj = 0; jj < DK_MAX_DPL; jj++) o[jj] = 0.f;
    for (int w = 0; w < DK_ATT_WARPS; w++) {
      if (sm[w] == -INFINITY) continue;
      const float f = expf(sm[w] - M);
      L += sl[w] * f;
      for (int jj = 0; jj < DK_MAX_DPL; jj++) o[jj] += sacc[w][lane + 32 * jj] * f;
    }
    for (int jj = 0; jj < DK_MAX_DPL; jj++) {
      const int c = lane + 32 * jj;
      if (c < d) out[(int64_t)r * qw + (int64_t)hd * d + c] = o[jj] / L;
    }
  }
}

__global__ void dk_silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ y,
                                   int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float gv = g[i];
    y[i] = gv / (1.f + expf(-gv)) * u[i];
  }
}

// Scatter rows into a cache: dst[(pos0 + r) % cap or pos0 + r] = src[r].
__global__ void dk_scatter_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t cap, int ring,
                                  int64_t pos0, int width) {
  const int r = blockIdx.x;
  const int64_t pos = pos0 + r;
  const int64_t at = ring ? pos % cap : pos;
  for (int i = threadIdx.x; i < width; i += blockDim.x) dst[at * width + i] = src[(int64_t)r * width + i];
}

// [state | w1[prev]] for the recurrent head's step.
__global__ void dk_markov_in_kernel(const float* __restrict__ state, const uint16_t* __restrict__ w1,
                                    const uint32_t* __restrict__ prev, float* __restrict__ out, int rank) {
  const int i = threadIdx.x;
  if (i < rank) {
    out[i] = state[i];
    out[rank + i] = dk_bf16(w1[(int64_t)(*prev) * rank + i]);
  }
}

// raw = ctx + add; g = sigmoid(raw[0:r]); state = g * state + (1 - g) * tanh(raw[r:2r]);
// latent = tanh(raw[2r:3r]).
__global__ void dk_markov_gate_kernel(const float* __restrict__ ctx, const float* __restrict__ add,
                                      const float* __restrict__ add2, float* __restrict__ state,
                                      float* __restrict__ latent, int rank) {
  const int i = threadIdx.x;
  if (i < rank) {
    const float gate = ctx[i] + add[i] + add2[i];
    const float cand = ctx[rank + i] + add[rank + i] + add2[rank + i];
    const float o = ctx[2 * rank + i] + add[2 * rank + i] + add2[2 * rank + i];
    const float g = 1.f / (1.f + expf(-gate));
    state[i] = g * state[i] + (1.f - g) * tanhf(cand);
    latent[i] = tanhf(o);
  }
}

// logits = cap * tanh((base + bias) / cap); per-block first maximum into part.
__global__ void dk_cap_kernel(const float* __restrict__ base, const float* __restrict__ bias, float cap,
                              float* __restrict__ logits, int n, float* __restrict__ part_v,
                              uint32_t* __restrict__ part_i) {
  __shared__ float bv[256];
  __shared__ uint32_t bi[256];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = -INFINITY;
  uint32_t idx = 0xffffffffu;
  if (i < n) {
    v = cap * tanhf((base[i] + bias[i]) / cap);
    logits[i] = v;
    idx = (uint32_t)i;
  }
  bv[threadIdx.x] = v;
  bi[threadIdx.x] = idx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = bv[threadIdx.x + s];
      const uint32_t oi = bi[threadIdx.x + s];
      if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
        bv[threadIdx.x] = ov;
        bi[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_v[blockIdx.x] = bv[0];
    part_i[blockIdx.x] = bi[0];
  }
}

__global__ void dk_pick_kernel(const float* __restrict__ part_v, const uint32_t* __restrict__ part_i, int parts,
                               uint32_t* __restrict__ token, uint32_t* __restrict__ prev) {
  __shared__ float bv[1024];
  __shared__ uint32_t bi[1024];
  float v = -INFINITY;
  uint32_t idx = 0xffffffffu;
  for (int p = threadIdx.x; p < parts; p += blockDim.x) {
    if (part_v[p] > v || (part_v[p] == v && part_i[p] < idx)) {
      v = part_v[p];
      idx = part_i[p];
    }
  }
  bv[threadIdx.x] = v;
  bi[threadIdx.x] = idx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = bv[threadIdx.x + s];
      const uint32_t oi = bi[threadIdx.x + s];
      if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
        bv[threadIdx.x] = ov;
        bi[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *token = bi[0];
    if (prev) *prev = bi[0];
  }
}

// fold[v, j] = sum_h head[v, h] * how[h, j]; fold_b[v] = sum_h head[v, h] * hob[h]; f64 sums.
// grid (vocab), block rank threads.
__global__ void dk_fold_kernel(const uint16_t* __restrict__ head, const uint16_t* __restrict__ how,
                               const uint16_t* __restrict__ hob, float* __restrict__ fold, float* __restrict__ fold_b,
                               int h, int rank) {
  __shared__ double red[32];
  const int v = blockIdx.x, j = threadIdx.x;
  const uint16_t* hr = head + (int64_t)v * h;
  double s = 0.0, b = 0.0;
  for (int i = 0; i < h; i++) {
    const double hv = dk_bf16(hr[i]);
    if (j < rank) s += hv * (double)dk_bf16(how[(int64_t)i * rank + j]);
  }
  for (int i = j; i < h; i += blockDim.x) b += (double)dk_bf16(hr[i]) * (double)dk_bf16(hob[i]);
  const double bias = dk_block_sum(b, red);
  if (j < rank) fold[(int64_t)v * rank + j] = (float)s;
  if (j == 0) fold_b[v] = (float)bias;
}

__global__ void dk_bf16_to_f32_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = dk_bf16(src[i]);
}

}  // namespace

extern "C" {

int memra_ink2d_gemv(const float* x, int64_t ldx, const void* w, int64_t ldw, const float* bias, float* y,
                     int64_t ldy, int m, int n, int k, int acc, void* stream) {
  if (!x || !w || !y || m <= 0 || n <= 0 || k <= 0 || k % 8 || ldw % 8 || ldx % 4 ||
      ((uintptr_t)w & 15) || ((uintptr_t)x & 15))
    return 46001;
  const int groups = (m + DK_MR - 1) / DK_MR;
  dim3 grid((n + 7) / 8, groups < 64 ? groups : 64);
  dk_gemv_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(x, ldx, (const uint16_t*)w, ldw, bias, y, ldy, m, n, k, acc);
  return dk_rc();
}

// y[n] = bias[n] + sum_k x[k] * w[n, k], f32 weights, one row.
int memra_ink2d_gemv_f32w(const float* x, const float* w, int64_t ldw, const float* bias, float* y, int n, int k,
                          void* stream) {
  if (!x || !w || !y || n <= 0 || k <= 0 || k % 4 || ldw % 4 || ((uintptr_t)w & 15) || ((uintptr_t)x & 15))
    return 46014;
  dk_gemv_f32w_kernel<<<(n + 7) / 8, 256, 0, (cudaStream_t)stream>>>(x, w, ldw, bias, y, n, k);
  return dk_rc();
}

int memra_ink2d_mix_norm(const float* taps, int64_t tap_stride, int64_t row_stride, int ntaps, const float* alpha,
                         const float* w, double eps, float* out, int rows, int h, void* stream) {
  if (!taps || !alpha || !w || !out || rows <= 0 || h <= 0 || h > 12288 || ntaps <= 0) return 46002;
  dk_mix_norm_kernel<<<rows, 256, h * sizeof(float), (cudaStream_t)stream>>>(taps, tap_stride, row_stride, ntaps,
                                                                              alpha, w, eps, out, h);
  return dk_rc();
}

int memra_ink2d_norm(float* x, const float* add, const float* w, double eps, float* y, int rows, int h,
                     void* stream) {
  if (!x || !w || !y || rows <= 0 || h <= 0) return 46003;
  dk_norm_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(x, add, w, eps, y, h);
  return dk_rc();
}

int memra_ink2d_embed(const void* table, const uint32_t* id, const float* w, double eps, float* out, int h,
                      void* stream) {
  if (!table || !id || !w || !out || h <= 0) return 46004;
  dk_embed_kernel<<<1, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, id, w, eps, out, h);
  return dk_rc();
}

int memra_ink2d_head_rope(float* x, int64_t ld, const float* w, int norm, double eps, int64_t pos0, double theta,
                          int rows, int heads, int d, void* stream) {
  if (!x || (norm && !w) || rows <= 0 || heads <= 0 || d <= 0 || d % 2 || d > 512) return 46005;
  dk_head_rope_kernel<<<dim3(rows, heads), d / 2 < 32 ? 32 : ((d / 2 + 31) / 32) * 32, 0, (cudaStream_t)stream>>>(x, ld, w, norm, eps, pos0, theta, d);
  return dk_rc();
}

int memra_ink2d_attn(const float* q, const float* kb, const float* vb, const float* kc, const float* vc, int64_t cap,
                     int ring, int64_t first, int64_t p0, int rows, int heads, int kv_heads, int d, float scale,
                     float* out, void* stream) {
  if (!q || !kb || !vb || !out || rows <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || d <= 0 ||
      d > 32 * DK_MAX_DPL || first > p0 || (p0 > first && (!kc || !vc)) || (ring && p0 - first > cap) ||
      (!ring && p0 > cap))
    return 46006;
  dk_attn_kernel<<<dim3(rows, heads), 32 * DK_ATT_WARPS, 0, (cudaStream_t)stream>>>(
      q, kb, vb, kc, vc, cap, ring, first, p0, rows, heads, kv_heads, d, scale, out);
  return dk_rc();
}

int memra_ink2d_silu_mul(const float* g, const float* u, float* y, int64_t n, void* stream) {
  if (!g || !u || !y || n <= 0) return 46007;
  dk_silu_mul_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(g, u, y, n);
  return dk_rc();
}

int memra_ink2d_scatter(const float* src, float* dst, int64_t cap, int ring, int64_t pos0, int rows, int width,
                        void* stream) {
  if (!src || !dst || rows <= 0 || width <= 0 || (!ring && pos0 + rows > cap) || (ring && rows > cap)) return 46008;
  dk_scatter_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(src, dst, cap, ring, pos0, width);
  return dk_rc();
}

int memra_ink2d_markov_in(const float* state, const void* w1, const uint32_t* prev, float* out, int rank,
                          void* stream) {
  if (!state || !w1 || !prev || !out || rank <= 0 || rank > 1024) return 46009;
  dk_markov_in_kernel<<<1, rank, 0, (cudaStream_t)stream>>>(state, (const uint16_t*)w1, prev, out, rank);
  return dk_rc();
}

int memra_ink2d_markov_gate(const float* ctx, const float* add, const float* add2, float* state, float* latent,
                            int rank, void* stream) {
  if (!ctx || !add || !add2 || !state || !latent || rank <= 0 || rank > 1024) return 46010;
  dk_markov_gate_kernel<<<1, rank, 0, (cudaStream_t)stream>>>(ctx, add, add2, state, latent, rank);
  return dk_rc();
}

// Softcapped logits of base + bias, and their first maximum into token (and prev, when given).
// part_v / part_i hold ceil(n / 256) entries.
int memra_ink2d_cap_pick(const float* base, const float* bias, float cap, float* logits, int n, float* part_v,
                         uint32_t* part_i, uint32_t* token, uint32_t* prev, void* stream) {
  if (!base || !bias || !logits || !part_v || !part_i || !token || n <= 0 || cap <= 0.f) return 46011;
  const int parts = (n + 255) / 256;
  dk_cap_kernel<<<parts, 256, 0, (cudaStream_t)stream>>>(base, bias, cap, logits, n, part_v, part_i);
  const int rc = dk_rc();
  if (rc) return rc;
  dk_pick_kernel<<<1, 1024, 0, (cudaStream_t)stream>>>(part_v, part_i, parts, token, prev);
  return dk_rc();
}

int memra_ink2d_fold(const void* head, const void* how, const void* hob, float* fold, float* fold_b, int vocab, int h,
                     int rank, void* stream) {
  if (!head || !how || !hob || !fold || !fold_b || vocab <= 0 || h <= 0 || rank <= 0 || rank > 1024) return 46012;
  const int threads = rank < 32 ? 32 : ((rank + 31) / 32) * 32;
  dk_fold_kernel<<<vocab, threads, 0, (cudaStream_t)stream>>>((const uint16_t*)head, (const uint16_t*)how,
                                                               (const uint16_t*)hob, fold, fold_b, h, rank);
  return dk_rc();
}

int memra_ink2d_bf16_to_f32(const void* src, float* dst, int64_t n, void* stream) {
  if (!src || !dst || n <= 0) return 46013;
  dk_bf16_to_f32_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>((const uint16_t*)src, dst, n);
  return dk_rc();
}

}  // extern "C"
