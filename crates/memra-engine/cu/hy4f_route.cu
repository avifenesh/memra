// Hy4-flash GPU route kernels (`crate::hy4f_route`, binary `hy4f-gpu`), numeric class
// `hy-v4-flash-bf16-activations`.
//
// The class: every weight GEMM carries its input activations as two BF16 terms (hi = bf16(x),
// lo = bf16(x - hi): 16 significant bits) and multiplies both on the tensor cores (mma.sync
// m16n8k16, F32 accumulation, hi then lo at every 16-wide K step) by the weight as stored: BF16 as read, FP8
// E4M3 codes (exact in BF16) with the 128x128 F32 block scale multiplied into each 128-input
// partial sum, NVFP4 e2m1 codes times their E4M3 group scale (the product has at most six
// significant bits, so it is exact in BF16) with the expert's F32 `weight_scale_2` multiplied
// into the finished sum. Everything else runs in F32: the hyper-connection streams, norms,
// RoPE, softmax, router scores, indexer pooling and group logits, the Engram gates. The carried
// cache holds the latent and rope key (576 F32 values a token a layer) and, on the indexed
// layers, the indexer's token key (128 F32) and the pooled group keys (128 F32).
//
// Every kernel computes a row (a token, a query, a group) on its own, in an order that does
// not depend on the other rows of the call or on the tile size: the GEMM never splits K and
// walks K in the same 32-wide steps in both its tile shapes, attention walks a query's key list
// in fixed 256-key chunks merged in order, and reductions are fixed trees. A token therefore
// gets the same bits whether it is scored in a prefill or in a decode call.
//
// `red_dq` is the checkpoint gate's red arm: the quantized weights multiply as bf16 of their
// dequantized value (`code * scale * scale_2`, `code * block scale`), as a BF16 loader would.
//
// Return codes: 0 ok, 10000 + cudaError_t, 45xxx for a refused contract.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define R_BF16 0
#define R_FP8 2
#define R_NVFP4 3

static __device__ __forceinline__ float r_e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

static __device__ __forceinline__ float r_e2m1(uint32_t nib) {
  const float t[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return t[nib & 15];
}

static __device__ __forceinline__ float r_bf(uint16_t b) { return __uint_as_float(((uint32_t)b) << 16); }

static __device__ __forceinline__ uint16_t r_tobf(float v) {
  __nv_bfloat16 h = __float2bfloat16_rn(v);
  return *(uint16_t*)&h;
}

static __device__ __forceinline__ uint32_t r_pack(float lo, float hi) {
  return (uint32_t)r_tobf(lo) | ((uint32_t)r_tobf(hi) << 16);
}

static int r_rc() {
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

// ---- the weight GEMM ------------------------------------------------------------------------

struct RGemm {
  const float* x;    // [m, k] f32, row stride ldx; batch stride xbs
  long ldx, xbs;
  const uint8_t* w;  // stored weight [n, k] (BF16, FP8 codes, NVFP4 packed pairs); batch/expert stride wbs bytes
  long wbs;
  const uint8_t* sc; // FP8: F32 [n/128, grid_cols]; NVFP4: E4M3 [n, k/16]; expert stride scbs bytes
  long scbs;
  const float* s2;   // NVFP4 weight_scale_2 per expert (or null)
  float* y;          // [m, n] f32, row stride ldy; batch stride ybs
  long ldy, ybs;
  int m, n, k, grid_cols;
  const int* tiles;  // grouped mode: per blockIdx.y (expert, row0, rows); null otherwise
  int red_dq;
};

#define RG_BN 64
#define RG_BK 32
#define RG_PAD 8

static __device__ __forceinline__ void r_mma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// BM 64: four warps 2x2, a warp 32 rows x 32 cols. BM 16: four warps 1x4, a warp 16 x 16. Both
// run the same per-element sequence: per 32-wide K step two mma (k 0..15, 16..31) into the same
// accumulator, the FP8 block scale folded in every 128 inputs.
template <int BM, int FMT>
__global__ void __launch_bounds__(128) r_gemm_kernel(RGemm g) {
  constexpr int WR = BM == 64 ? 32 : 16;
  constexpr int WC = BM == 64 ? 32 : 16;
  constexpr int MT = WR / 16;
  constexpr int NT = WC / 8;
  constexpr int LDS = RG_BK + RG_PAD;
  __shared__ __align__(16) uint16_t As[BM * LDS];
  __shared__ __align__(16) uint16_t Al[BM * LDS];
  __shared__ __align__(16) uint16_t Bs[RG_BN * LDS];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = BM == 64 ? (warp >> 1) : 0;
  const int wn = BM == 64 ? (warp & 1) : warp;

  const float* x = g.x;
  const uint8_t* w = g.w;
  const uint8_t* sc = g.sc;
  float* y = g.y;
  float s2 = 1.0f;
  int m = g.m, m0 = 0;
  if (g.tiles) {
    const int e = g.tiles[blockIdx.y * 3], r0 = g.tiles[blockIdx.y * 3 + 1], rows = g.tiles[blockIdx.y * 3 + 2];
    if (rows <= 0) return;
    x += (long)r0 * g.ldx;
    y += (long)r0 * g.ldy;
    w += (long)e * g.wbs;
    if (sc) sc += (long)e * g.scbs;
    if (g.s2) s2 = g.s2[e];
    m = rows;
  } else {
    const int b = blockIdx.z;
    m0 = blockIdx.y * BM;
    x += (long)b * g.xbs;
    y += (long)b * g.ybs;
    w += (long)b * g.wbs;
    if (sc) sc += (long)b * g.scbs;
    if (g.s2) s2 = g.s2[b];
  }
  const int n0 = blockIdx.x * RG_BN;
  const int n = g.n, k = g.k;

  // staging maps: weight row tid/2, 16 inputs from (tid&1)*16; activations likewise for BM 64,
  // row tid/8 and 4 inputs from (tid&7)*4 for BM 16
  const int wr = tid >> 1, wk = (tid & 1) * 16;
  const int gn = n0 + wr;
  const bool wok = gn < n;
  constexpr int AV = BM == 64 ? 16 : 4;
  const int ar = BM == 64 ? (tid >> 1) : (tid >> 3);
  const int ak = BM == 64 ? (tid & 1) * 16 : (tid & 7) * 4;
  const bool aok = m0 + ar < m;

  uint4 wraw[2];
  uint32_t wsc = 0;
  float4 araw[AV / 4];

  auto load = [&](int k0) {
    if (wok) {
      if (FMT == R_BF16) {
        const uint4* p = (const uint4*)(w + ((long)gn * k + k0 + wk) * 2);
        wraw[0] = p[0];
        wraw[1] = p[1];
      } else if (FMT == R_FP8) {
        wraw[0] = *(const uint4*)(w + (long)gn * k + k0 + wk);
      } else {
        uint2 c = *(const uint2*)(w + (long)gn * (k / 2) + (k0 + wk) / 2);
        wraw[0].x = c.x;
        wraw[0].y = c.y;
        wsc = sc[(long)gn * (k / 16) + (k0 + wk) / 16];
      }
    }
    if (aok) {
      const float4* p = (const float4*)(x + (long)(m0 + ar) * g.ldx + k0 + ak);
#pragma unroll
      for (int i = 0; i < AV / 4; i++) araw[i] = p[i];
    }
  };

  auto store = [&](int k0) {
    uint32_t* bw = (uint32_t*)&Bs[wr * LDS + wk];
    if (!wok) {
#pragma unroll
      for (int i = 0; i < 8; i++) bw[i] = 0;
    } else if (FMT == R_BF16) {
      const uint32_t* s = (const uint32_t*)wraw;
#pragma unroll
      for (int i = 0; i < 8; i++) bw[i] = s[i];
    } else if (FMT == R_FP8) {
      const uint8_t* c = (const uint8_t*)&wraw[0];
      float bs = 1.0f;
      if (g.red_dq) bs = ((const float*)sc)[(long)(gn / 128) * g.grid_cols + k0 / 128];
#pragma unroll
      for (int i = 0; i < 8; i++) bw[i] = r_pack(r_e4m3(c[2 * i]) * bs, r_e4m3(c[2 * i + 1]) * bs);
    } else {
      const uint8_t* c = (const uint8_t*)&wraw[0];
      float gs = r_e4m3((uint8_t)wsc);
      if (g.red_dq) gs = gs * s2;
#pragma unroll
      for (int i = 0; i < 8; i++) bw[i] = r_pack(r_e2m1(c[i]) * gs, r_e2m1(c[i] >> 4) * gs);
    }
    uint32_t* aw = (uint32_t*)&As[ar * LDS + ak];
    uint32_t* al = (uint32_t*)&Al[ar * LDS + ak];
    if (!aok) {
#pragma unroll
      for (int i = 0; i < AV / 2; i++) aw[i] = al[i] = 0;
    } else {
#pragma unroll
      for (int i = 0; i < AV / 4; i++) {
        const float v[4] = {araw[i].x, araw[i].y, araw[i].z, araw[i].w};
        float lo[4];
#pragma unroll
        for (int j = 0; j < 4; j++) lo[j] = v[j] - r_bf(r_tobf(v[j]));
        aw[2 * i] = r_pack(v[0], v[1]);
        aw[2 * i + 1] = r_pack(v[2], v[3]);
        al[2 * i] = r_pack(lo[0], lo[1]);
        al[2 * i + 1] = r_pack(lo[2], lo[3]);
      }
    }
  };

  float acc[MT][NT][4];
  float part[MT][NT][4];
#pragma unroll
  for (int a = 0; a < MT; a++)
#pragma unroll
    for (int b = 0; b < NT; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) acc[a][b][c] = part[a][b][c] = 0.0f;

  const bool fold_blocks = FMT == R_FP8 && !g.red_dq;
  load(0);
  for (int k0 = 0; k0 < k; k0 += RG_BK) {
    store(k0);
    __syncthreads();
    if (k0 + RG_BK < k) load(k0 + RG_BK);
#pragma unroll
    for (int kk = 0; kk < RG_BK; kk += 16) {
      uint32_t af[MT][4], afl[MT][4], bfr[NT][2];
#pragma unroll
      for (int a = 0; a < MT; a++) {
        const int r = wm * WR + a * 16 + (lane >> 2);
        const int c = kk + (lane & 3) * 2;
        af[a][0] = *(const uint32_t*)&As[r * LDS + c];
        af[a][1] = *(const uint32_t*)&As[(r + 8) * LDS + c];
        af[a][2] = *(const uint32_t*)&As[r * LDS + c + 8];
        af[a][3] = *(const uint32_t*)&As[(r + 8) * LDS + c + 8];
        afl[a][0] = *(const uint32_t*)&Al[r * LDS + c];
        afl[a][1] = *(const uint32_t*)&Al[(r + 8) * LDS + c];
        afl[a][2] = *(const uint32_t*)&Al[r * LDS + c + 8];
        afl[a][3] = *(const uint32_t*)&Al[(r + 8) * LDS + c + 8];
      }
#pragma unroll
      for (int b = 0; b < NT; b++) {
        const int r = wn * WC + b * 8 + (lane >> 2);
        const int c = kk + (lane & 3) * 2;
        bfr[b][0] = *(const uint32_t*)&Bs[r * LDS + c];
        bfr[b][1] = *(const uint32_t*)&Bs[r * LDS + c + 8];
      }
#pragma unroll
      for (int a = 0; a < MT; a++)
#pragma unroll
        for (int b = 0; b < NT; b++) {
          float* c = fold_blocks ? part[a][b] : acc[a][b];
          r_mma(c, af[a], bfr[b]);
          r_mma(c, afl[a], bfr[b]);
        }
    }
    if (fold_blocks && (((k0 + RG_BK) % 128) == 0 || k0 + RG_BK >= k)) {
      const float s = ((const float*)sc)[(long)(n0 / 128) * g.grid_cols + k0 / 128];
#pragma unroll
      for (int a = 0; a < MT; a++)
#pragma unroll
        for (int b = 0; b < NT; b++)
#pragma unroll
          for (int c = 0; c < 4; c++) {
            acc[a][b][c] = __fmaf_rn(part[a][b][c], s, acc[a][b][c]);
            part[a][b][c] = 0.0f;
          }
    }
    __syncthreads();
  }
  const bool scale2 = FMT == R_NVFP4 && !g.red_dq;
#pragma unroll
  for (int a = 0; a < MT; a++)
#pragma unroll
    for (int b = 0; b < NT; b++)
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int r = m0 + wm * WR + a * 16 + (lane >> 2) + (c >= 2 ? 8 : 0);
        const int col = n0 + wn * WC + b * 8 + (lane & 3) * 2 + (c & 1);
        if (r < m && col < n) {
          float v = acc[a][b][c];
          if (scale2) v = __fmul_rn(v, s2);
          y[(long)r * g.ldy + col] = v;
        }
      }
}

template <int BM>
static int r_gemm_launch(const RGemm& g, int fmt, int batch, int tiles, cudaStream_t st) {
  dim3 grid((g.n + RG_BN - 1) / RG_BN, g.tiles ? tiles : (g.m + BM - 1) / BM, g.tiles ? 1 : batch);
  if (grid.y == 0) return 0;
  if (grid.y > 65535 || grid.z > 65535) return 45004;
  if (fmt == R_BF16)
    r_gemm_kernel<BM, R_BF16><<<grid, 128, 0, st>>>(g);
  else if (fmt == R_FP8)
    r_gemm_kernel<BM, R_FP8><<<grid, 128, 0, st>>>(g);
  else
    r_gemm_kernel<BM, R_NVFP4><<<grid, 128, 0, st>>>(g);
  return r_rc();
}

// `small` picks the 16-row tile (a decode call); the result does not depend on it.
extern "C" int memra_hy4r_gemm(const float* x, long ldx, long xbs, const void* w, long wbs, const void* sc, long scbs,
                               const float* s2, int fmt, int grid_cols, float* y, long ldy, long ybs, int m, int n,
                               int k, int batch, const int* tiles, int ntiles, int red_dq, int small, void* stream) {
  if (n <= 0 || k <= 0 || (k % RG_BK) != 0 || m < 0) return 45001;
  if (fmt != R_BF16 && fmt != R_FP8 && fmt != R_NVFP4) return 45002;
  if (fmt != R_BF16 && sc == nullptr) return 45003;
  if ((ldx % 4) != 0 || (xbs % 4) != 0 || (((uintptr_t)x) % 16) != 0) return 45005;
  if (!tiles && m == 0) return 0;
  RGemm g{x, ldx, xbs, (const uint8_t*)w, wbs, (const uint8_t*)sc, scbs, s2, y, ldy, ybs, m, n, k, grid_cols, tiles, red_dq};
  cudaStream_t st = (cudaStream_t)stream;
  return small ? r_gemm_launch<16>(g, fmt, batch, ntiles, st) : r_gemm_launch<64>(g, fmt, batch, ntiles, st);
}

// ---- reductions -----------------------------------------------------------------------------

static __device__ __forceinline__ float r_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

static __device__ __forceinline__ float r_warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// block of 256: a fixed tree (warp butterflies, then warp 0 over the 8 partials)
static __device__ float r_block_sum(float v, float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = r_warp_sum(v);
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = lane < (int)(blockDim.x >> 5) ? red[lane] : 0.0f;
  t = r_warp_sum(t);
  return t;
}

// ---- elementwise and per-row kernels ----------------------------------------------------------

// y = w * (x * rsqrt(mean(x^2) + eps)), one block (256) per row
__global__ void r_rms_kernel(const float* x, long ldx, const float* w, float* y, long ldy, int d, float eps) {
  __shared__ float red[32];
  const float* xr = x + (long)blockIdx.x * ldx;
  float* yr = y + (long)blockIdx.x * ldy;
  float s = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) s += xr[i] * xr[i];
  s = r_block_sum(s, red);
  const float inv = rsqrtf(s / d + eps);
  for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = (w ? w[i] : 1.0f) * (xr[i] * inv);
}

extern "C" int memra_hy4r_rms(const float* x, long ldx, const float* w, float* y, long ldy, int rows, int d, float eps,
                              void* stream) {
  if (rows == 0) return 0;
  r_rms_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(x, ldx, w, y, ldy, d, eps);
  return r_rc();
}

// interleaved RoPE in place on `heads` slices of `dim` at offset `off` of each row: pairs
// (2i, 2i+1) of the last rope dims; cos/sin from the call's table [rows of positions, dim/2]
__global__ void r_rope_kernel(float* x, long ldrow, long ldhead, int heads, int off, int half, const float2* cs,
                              const int* posrow) {
  const int r = blockIdx.x;
  const float2* c = cs + (long)posrow[r] * half;
  for (int e = threadIdx.x; e < heads * half; e += blockDim.x) {
    const int h = e / half, i = e % half;
    float* p = x + (long)r * ldrow + (long)h * ldhead + off + 2 * i;
    const float x0 = p[0], x1 = p[1];
    const float co = c[i].x, si = c[i].y;
    p[0] = x0 * co - x1 * si;
    p[1] = x1 * co + x0 * si;
  }
}

extern "C" int memra_hy4r_rope(float* x, long ldrow, long ldhead, int rows, int heads, int off, int half,
                               const void* cs, const int* posrow, void* stream) {
  if (rows == 0) return 0;
  r_rope_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(x, ldrow, ldhead, heads, off, half, (const float2*)cs, posrow);
  return r_rc();
}

// the call's cos/sin table: row r is position pos0 + r, inv_freq in f64
__global__ void r_rope_table_kernel(float2* out, int pos0, int rows, int half, const double* inv) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= rows * half) return;
  const int r = e / half, i = e % half;
  double s, c;
  sincos((double)(pos0 + r) * inv[i], &s, &c);
  out[e] = make_float2((float)c, (float)s);
}

extern "C" int memra_hy4r_rope_table(void* out, int pos0, int rows, int half, const double* inv, void* stream) {
  if (rows == 0) return 0;
  const int n = rows * half;
  r_rope_table_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>((float2*)out, pos0, rows, half, inv);
  return r_rc();
}

// HYV4FlashHyperConnection pre: per row, flat = s[r] (n streams x h), inv = rsqrt(mean(flat^2) +
// mix_eps), mix_j = (flat . fn_j) * inv; pre_j = sigmoid(mix_j * scale0 + base_j) + eps;
// post_j = mag * sigmoid(mix_{n+j} * scale1 + base_{n+j}) + eps; x = sum_j pre_j * flat_j.
// `fns` functions (2n for a layer, n for the exit head with post == null).
__global__ void r_hc_kernel(const float* s, const float* fn, const float* base, const float* scale, int nstream, int h,
                            int fns, float mix_eps, float eps, float mag, float* post, float* x) {
  __shared__ float red[32];
  __shared__ float mix[16];
  const int r = blockIdx.x;
  const int width = nstream * h;
  const float* flat = s + (long)r * width;
  float sq = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) sq += flat[i] * flat[i];
  sq = r_block_sum(sq, red);
  const float inv = rsqrtf(sq / width + mix_eps);
  for (int j = 0; j < fns; j++) {
    float d = 0.0f;
    const float* f = fn + (long)j * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) d += flat[i] * f[i];
    d = r_block_sum(d, red);
    if (threadIdx.x == 0) mix[j] = d * inv;
  }
  __syncthreads();
  __shared__ float pre[8];
  if (threadIdx.x < nstream) {
    const int j = threadIdx.x;
    pre[j] = 1.0f / (1.0f + expf(-(mix[j] * scale[0] + base[j]))) + eps;
    if (post) post[r * nstream + j] = mag * (1.0f / (1.0f + expf(-(mix[nstream + j] * scale[1] + base[nstream + j])))) + eps;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < h; d += blockDim.x) {
    float v = 0.0f;
    for (int j = 0; j < nstream; j++) v += pre[j] * flat[(long)j * h + d];
    x[(long)r * h + d] = v;
  }
}

extern "C" int memra_hy4r_hc(const float* s, const float* fn, const float* base, const float* scale, int rows,
                             int nstream, int h, int fns, float mix_eps, float eps, float mag, float* post, float* x,
                             void* stream) {
  if (rows == 0) return 0;
  if (nstream > 8 || fns > 16) return 45101;
  r_hc_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(s, fn, base, scale, nstream, h, fns, mix_eps, eps, mag, post, x);
  return r_rc();
}

// s[r, j, :] = post[r, j] * a[r, :] + s[r, j, :]
__global__ void r_write_back_kernel(float* s, const float* post, const float* a, int nstream, int h, long total) {
  const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total) return;
  const long r = e / ((long)nstream * h);
  const int j = (int)((e / h) % nstream);
  const int d = (int)(e % h);
  s[e] = post[r * nstream + j] * a[r * h + d] + s[e];
}

extern "C" int memra_hy4r_write_back(float* s, const float* post, const float* a, int rows, int nstream, int h,
                                     void* stream) {
  const long total = (long)rows * nstream * h;
  if (total == 0) return 0;
  r_write_back_kernel<<<(unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream>>>(s, post, a, nstream, h, total);
  return r_rc();
}

// out = silu(min(g, lim)) * clamp(u, -lim, lim) (lim < 0: no clamp); `gate_rows` scales: the
// attention output gate, out = o * sigmoid(g) when mode == 1
__global__ void r_act_kernel(const float* g, const float* u, float* out, long n, float lim, int mode) {
  const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  if (mode == 1) {
    out[e] = u[e] * (1.0f / (1.0f + expf(-g[e])));
    return;
  }
  float a = g[e], b = u[e];
  if (lim >= 0.0f) {
    a = fminf(a, lim);
    b = fminf(fmaxf(b, -lim), lim);
  }
  out[e] = a / (1.0f + expf(-a)) * b;
}

extern "C" int memra_hy4r_act(const float* g, const float* u, float* out, long n, float lim, int mode, void* stream) {
  if (n == 0) return 0;
  r_act_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(g, u, out, n, lim, mode);
  return r_rc();
}

// y[r, :] = a[r, :] + b[r, :] (+ bias)
__global__ void r_add_kernel(float* y, const float* a, const float* b, const float* bias, long n, int d) {
  const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  float v = a[e];
  if (b) v = v + b[e];
  if (bias) v = v + bias[e % d];
  y[e] = v;
}

extern "C" int memra_hy4r_add(float* y, const float* a, const float* b, const float* bias, long n, int d, void* stream) {
  if (n == 0) return 0;
  r_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, a, b, bias, n, d);
  return r_rc();
}

// strided copy of `cols` f32 values a row, optionally into BF16 (`to_bf16`)
__global__ void r_copy_kernel(const float* src, long lds, void* dst, long ldd, int cols, int to_bf16, const int* dstrow) {
  const int r = blockIdx.x;
  const long dr = dstrow ? dstrow[r] : r;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = src[(long)r * lds + c];
    if (to_bf16)
      ((uint16_t*)dst)[dr * ldd + c] = r_tobf(v);
    else
      ((float*)dst)[dr * ldd + c] = v;
  }
}

extern "C" int memra_hy4r_copy(const float* src, long lds, void* dst, long ldd, int rows, int cols, int to_bf16,
                               const int* dstrow, void* stream) {
  if (rows == 0) return 0;
  r_copy_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(src, lds, dst, ldd, cols, to_bf16, dstrow);
  return r_rc();
}

// BF16 rows gathered into another BF16 buffer: dst[i] = src[rows[i]] (`words` u32 a row)
__global__ void r_gather_bf_kernel(const uint32_t* src, long lds, const int* rows, uint32_t* dst, long ldd, int words) {
  const int i = blockIdx.x;
  const long sr = rows[i];
  for (int c = threadIdx.x; c < words; c += blockDim.x) dst[(long)i * ldd + c] = src[sr * lds + c];
}

extern "C" int memra_hy4r_gather_bf(const void* src, long lds, const int* rows, void* dst, long ldd, int n, int words,
                                    void* stream) {
  if (n == 0) return 0;
  r_gather_bf_kernel<<<n, 128, 0, (cudaStream_t)stream>>>((const uint32_t*)src, lds, rows, (uint32_t*)dst, ldd, words);
  return r_rc();
}

// embedding rows on every stream: s[r, j, :] = embed[tok[r]]
__global__ void r_embed_kernel(const uint16_t* table, const int* tok, float* s, int nstream, int h) {
  const int r = blockIdx.x;
  const uint16_t* row = table + (long)tok[r] * h;
  for (int d = threadIdx.x; d < h; d += blockDim.x) {
    const float v = r_bf(row[d]);
    for (int j = 0; j < nstream; j++) s[((long)r * nstream + j) * h + d] = v;
  }
}

extern "C" int memra_hy4r_embed(const void* table, const int* tok, float* s, int rows, int nstream, int h, void* stream) {
  if (rows == 0) return 0;
  r_embed_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)table, tok, s, nstream, h);
  return r_rc();
}

// ---- top-k under the CUDA tie order ------------------------------------------------------------

// an order-preserving key: larger value, larger key (no NaN reaches here)
static __device__ __forceinline__ uint32_t r_key(float v) {
  uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block (256) per row: the `k` best of `n` values (equal values: the lower index first, or
// the higher with `reverse_ties`, the red arm), written in ascending index order. A radix
// select finds the k-th key; one ordered pass emits.
__global__ void __launch_bounds__(256) r_topk_kernel(const float* vals, long ldv, int n, int k, int reverse_ties,
                                                     int* out, long ldo) {
  __shared__ int hist[256];
  __shared__ int sh[4];
  __shared__ int wsum[8];
  const float* v = vals + (long)blockIdx.x * ldv;
  int* o = out + (long)blockIdx.x * ldo;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int kk = k < n ? k : n;
  if (kk <= 0) return;
  uint32_t prefix = 0, mask = 0;
  int need = kk;
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += 256) {
      const uint32_t key = r_key(v[i]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0, digit = 0;
      for (int b = 255; b >= 0; b--) {
        if (acc + hist[b] >= need) {
          digit = b;
          break;
        }
        acc += hist[b];
      }
      sh[0] = digit;
      sh[1] = need - acc;
    }
    __syncthreads();
    prefix |= ((uint32_t)sh[0]) << shift;
    mask |= 255u << shift;
    need = sh[1];
    __syncthreads();
  }
  // `need` ties of key == prefix are taken; count all ties for the reversed order
  int eq = 0;
  for (int i = tid; i < n; i += 256) eq += r_key(v[i]) == prefix;
  eq = (int)r_warp_sum((float)eq);
  if (lane == 0) wsum[warp] = eq;
  __syncthreads();
  if (tid == 0) {
    int t = 0;
    for (int w2 = 0; w2 < 8; w2++) t += wsum[w2];
    sh[2] = t;
  }
  __syncthreads();
  const int ties = sh[2];
  int base_out = 0, base_eq = 0;
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int i = c0 + tid;
    uint32_t key = i < n ? r_key(v[i]) : 0u;
    const bool gt = i < n && key > prefix;
    const bool is_eq = i < n && key == prefix;
    // ordered rank of this tie in the chunk
    const unsigned eqm = __ballot_sync(0xffffffffu, is_eq);
    __syncthreads();
    if (lane == 0) wsum[warp] = __popc(eqm);
    __syncthreads();
    int eq_before = base_eq;
    for (int w2 = 0; w2 < warp; w2++) eq_before += wsum[w2];
    eq_before += __popc(eqm & ((1u << lane) - 1u));
    int chunk_eq = 0;
    for (int w2 = 0; w2 < 8; w2++) chunk_eq += wsum[w2];
    const bool take_eq = is_eq && (reverse_ties ? (eq_before >= ties - need) : (eq_before < need));
    const bool take = gt || take_eq;
    const unsigned tm = __ballot_sync(0xffffffffu, take);
    __syncthreads();
    if (lane == 0) wsum[warp] = __popc(tm);
    __syncthreads();
    int pos = base_out;
    for (int w2 = 0; w2 < warp; w2++) pos += wsum[w2];
    pos += __popc(tm & ((1u << lane) - 1u));
    int chunk_take = 0;
    for (int w2 = 0; w2 < 8; w2++) chunk_take += wsum[w2];
    if (take && pos < kk) o[pos] = i;
    base_out += chunk_take;
    base_eq += chunk_eq;
    __syncthreads();
  }
}

extern "C" int memra_hy4r_topk(const float* vals, long ldv, int rows, int n, int k, int reverse_ties, int* out, long ldo,
                               void* stream) {
  if (rows == 0) return 0;
  r_topk_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(vals, ldv, n, k, reverse_ties, out, ldo);
  return r_rc();
}

// ---- router ------------------------------------------------------------------------------------

// scores = sigmoid(logits), choice = scores + bias (both F32, as the program computes them)
__global__ void r_router_scores_kernel(const float* logits, const float* bias, float* scores, float* choice, long n,
                                       int e) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float s = 1.0f / (1.0f + expf(-logits[i]));
  scores[i] = s;
  choice[i] = s + (bias ? bias[i % e] : 0.0f);
}

extern "C" int memra_hy4r_router_scores(const float* logits, const float* bias, float* scores, float* choice, int rows,
                                        int e, void* stream) {
  const long n = (long)rows * e;
  if (n == 0) return 0;
  r_router_scores_kernel<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(logits, bias, scores, choice, n, e);
  return r_rc();
}

// weights of the applied picks: w = scores[pick] (normalized when the plan says so) * scaling
__global__ void r_router_w_kernel(const float* scores, int e, const int* picks, int k, int normalize, float scaling,
                                  float* w, int rows) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float z = 0.0f;
  for (int i = 0; i < k; i++) z += scores[(long)r * e + picks[r * k + i]];
  for (int i = 0; i < k; i++) {
    float v = scores[(long)r * e + picks[r * k + i]];
    if (normalize) v = v / (z + 1e-20f);
    w[r * k + i] = v * scaling;
  }
}

extern "C" int memra_hy4r_router_weights(const float* scores, int rows, int e, const int* picks, int k, int normalize,
                                         float scaling, float* w, void* stream) {
  if (rows == 0) return 0;
  r_router_w_kernel<<<(rows + 127) / 128, 128, 0, (cudaStream_t)stream>>>(scores, e, picks, k, normalize, scaling, w,
                                                                          rows);
  return r_rc();
}

// ---- MoE grouping ------------------------------------------------------------------------------

// counts[e] over the picks; then one block: offsets, tiles (expert, row0, rows) of `bm` rows,
// every unused tile zeroed; cursor = offsets
__global__ void r_moe_count_kernel(const int* picks, int n, int* counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[picks[i]], 1);
}

__global__ void r_moe_plan_kernel(const int* counts, int ne, int bm, int max_tiles, int* cursor, int* tiles) {
  if (threadIdx.x != 0) return;
  int off = 0, t = 0;
  for (int e = 0; e < ne; e++) {
    const int c = counts[e];
    cursor[e] = off;
    for (int r = 0; r < c; r += bm) {
      if (t < max_tiles) {
        tiles[t * 3] = e;
        tiles[t * 3 + 1] = off + r;
        tiles[t * 3 + 2] = c - r < bm ? c - r : bm;
      }
      t++;
    }
    off += c;
  }
  for (; t < max_tiles; t++) {
    tiles[t * 3] = 0;
    tiles[t * 3 + 1] = 0;
    tiles[t * 3 + 2] = 0;
  }
}

// each (token, slot) takes a row of its expert's group (the row order inside a group does not
// change any row's result); xg[row] = x[token]
__global__ void r_moe_place_kernel(const int* picks, int n, int k, int* cursor, int* rowmap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) rowmap[i] = atomicAdd(&cursor[picks[i]], 1);
}

__global__ void r_moe_gather_kernel(const float* x, int d, const int* rowmap, int k, float* xg) {
  const int i = blockIdx.x;  // token * k + slot
  const float* src = x + (long)(i / k) * d;
  float* dst = xg + (long)rowmap[i] * d;
  for (int c = threadIdx.x; c < d; c += blockDim.x) dst[c] = src[c];
}

extern "C" int memra_hy4r_moe_group(const int* picks, int rows, int k, int ne, int bm, int max_tiles, const float* x,
                                    int d, int* counts, int* cursor, int* tiles, int* rowmap, float* xg, void* stream) {
  cudaStream_t st = (cudaStream_t)stream;
  const int n = rows * k;
  if (n == 0) return 0;
  cudaMemsetAsync(counts, 0, sizeof(int) * ne, st);
  r_moe_count_kernel<<<(n + 255) / 256, 256, 0, st>>>(picks, n, counts);
  r_moe_plan_kernel<<<1, 32, 0, st>>>(counts, ne, bm, max_tiles, cursor, tiles);
  r_moe_place_kernel<<<(n + 255) / 256, 256, 0, st>>>(picks, n, k, cursor, rowmap);
  r_moe_gather_kernel<<<n, 256, 0, st>>>(x, d, rowmap, k, xg);
  return r_rc();
}

// out[t, :] = sum over slots in order (ascending expert) of w[t, slot] * y[rowmap[t, slot], :]
__global__ void r_moe_combine_kernel(const float* y, int d, const int* rowmap, const float* w, int k, float* out) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < d; c += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < k; s++) acc += w[t * k + s] * y[(long)rowmap[t * k + s] * d + c];
    out[(long)t * d + c] = acc;
  }
}

extern "C" int memra_hy4r_moe_combine(const float* y, int d, const int* rowmap, const float* w, int rows, int k,
                                      float* out, void* stream) {
  if (rows == 0) return 0;
  r_moe_combine_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(y, d, rowmap, w, k, out);
  return r_rc();
}

// ---- attention (MLA over the latent cache, sink in the denominator) ---------------------------

#define AT_HG 8
#define AT_TILE 32
#define AT_CHUNK 256
#define AT_DK 576
#define AT_DV 512
#define AT_KW 577  // f32 values a key row in shared memory (576 + 1 pad)

struct RAttn {
  const float* q;      // [rows, H, 576]: q_lat (512) then the rotated q_rope (64)
  const float* kv;     // cache rows of 576 F32 (latent, then rope key)
  const int* list;     // explicit key rows per query [rows, ldl] (mode 1)
  int ldl;
  const int* lens;     // keys per query (mode 1)
  const int* qpos;     // position per query (mode 0)
  int mode, window, ring, base, H, C;
  float scale;
  float* po;           // partials [rows, H, C, 512]
  float* pml;          // [rows, H, C, 2]
};

static __device__ __forceinline__ int r_key_row(const RAttn& a, int q, int j, int lo) {
  if (a.mode == 1) return a.list[(long)q * a.ldl + j];
  const int p = lo + j;
  return a.ring ? p % a.ring : p - a.base;
}

__global__ void __launch_bounds__(256) r_attn_kernel(RAttn a) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* qs = (float*)smem;                 // [8][576]
  float* ks = qs + AT_HG * AT_DK;           // [32][577]
  float* ps = ks + AT_TILE * AT_KW;         // [8][32]
  const int qi = blockIdx.x, hg = blockIdx.y, c = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, hl = tid >> 5;
  const int h = hg * AT_HG + hl;
  int len, lo = 0;
  if (a.mode == 1) {
    len = a.lens[qi];
  } else {
    const int p = a.qpos[qi];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    len = p - lo + 1;
  }
  const int j0 = c * AT_CHUNK, j1 = min(len, j0 + AT_CHUNK);
  const long pidx = ((long)qi * a.H + h) * a.C + c;
  float o[16];
#pragma unroll
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  float m = -INFINITY, l = 0.0f;
  if (j0 < j1) {
    for (int e = tid; e < AT_HG * AT_DK; e += 256)
      qs[e] = a.q[((long)qi * a.H + hg * AT_HG) * AT_DK + e];
    for (int t0 = j0; t0 < j1; t0 += AT_TILE) {
      const int nt = min(AT_TILE, j1 - t0);
      __syncthreads();
      for (int e = tid; e < AT_TILE * AT_DK; e += 256) {
        const int r = e / AT_DK, d = e % AT_DK;
        float v = 0.0f;
        if (r < nt) v = a.kv[(long)r_key_row(a, qi, t0 + r, lo) * AT_DK + d];
        ks[r * AT_KW + d] = v;
      }
      __syncthreads();
      float sc = -INFINITY;
      if (lane < nt) {
        const float* qh = qs + hl * AT_DK;
        const float* kr = ks + lane * AT_KW;
        float dot = 0.0f;
        for (int d = 0; d < AT_DK; d++) dot += qh[d] * kr[d];
        sc = dot * a.scale;
      }
      const float mt = r_warp_max(sc);
      const float mn = fmaxf(m, mt);
      const float corr = expf(m - mn);
      const float p = lane < nt ? expf(sc - mn) : 0.0f;
      l = l * corr + r_warp_sum(p);
      m = mn;
#pragma unroll
      for (int i = 0; i < 16; i++) o[i] *= corr;
      ps[hl * AT_TILE + lane] = p;
      __syncwarp();
      for (int kk = 0; kk < nt; kk++) {
        const float pk = ps[hl * AT_TILE + kk];
        const float* vr = ks + kk * AT_KW;
#pragma unroll
        for (int i = 0; i < 16; i++) o[i] += pk * vr[lane + 32 * i];
      }
      __syncwarp();
    }
  }
#pragma unroll
  for (int i = 0; i < 16; i++) a.po[pidx * AT_DV + lane + 32 * i] = o[i];
  if (lane == 0) {
    a.pml[pidx * 2] = m;
    a.pml[pidx * 2 + 1] = l;
  }
}

// out[q, h, :] = merge of the chunks in order, the sink in the max and the denominator
__global__ void r_attn_merge_kernel(const float* po, const float* pml, const float* sinks, int H, int C, float* out) {
  const int qi = blockIdx.x, h = blockIdx.y;
  const long b = ((long)qi * H + h) * C;
  float M = sinks[h];
  for (int c = 0; c < C; c++) M = fmaxf(M, pml[(b + c) * 2]);
  float L = expf(sinks[h] - M);
  for (int c = 0; c < C; c++) {
    const float mc = pml[(b + c) * 2];
    if (mc > -INFINITY) L += pml[(b + c) * 2 + 1] * expf(mc - M);
  }
  for (int d = threadIdx.x; d < AT_DV; d += blockDim.x) {
    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
      const float mc = pml[(b + c) * 2];
      if (mc > -INFINITY) acc += po[(b + c) * AT_DV + d] * expf(mc - M);
    }
    out[((long)qi * H + h) * AT_DV + d] = acc / L;
  }
}

extern "C" int memra_hy4r_attn(const float* q, const void* kv, const int* list, int ldl, const int* lens, const int* qpos,
                               int mode, int window, int ring, int base, int rows, int H, int C, float scale,
                               const float* sinks, float* po, float* pml, float* out, void* stream) {
  if (rows == 0) return 0;
  if (H % AT_HG != 0 || C <= 0) return 45201;
  const size_t smem = sizeof(float) * (AT_HG * AT_DK + AT_TILE * AT_KW + AT_HG * AT_TILE);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(r_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr = true;
  }
  RAttn a{q, (const float*)kv, list, ldl, lens, qpos, mode, window, ring, base, H, C, scale, po, pml};
  dim3 grid(rows, H / AT_HG, C);
  r_attn_kernel<<<grid, 256, smem, (cudaStream_t)stream>>>(a);
  int rc = r_rc();
  if (rc) return rc;
  r_attn_merge_kernel<<<dim3(rows, H), 128, 0, (cudaStream_t)stream>>>(po, pml, sinks, H, C, out);
  return r_rc();
}

// ---- the pooled indexer ------------------------------------------------------------------------

// pooled key of each closing group g (rows gs..gs+n): from its 4 cached F32 token keys,
// pooled[d] = sum_s (kvp[d] . k_s) * softmax_s(kgp[d] . k_s + ape[s, d]), then
// key_norm (F32) and RoPE of the last `rd` dims at the group's first position.
__global__ void r_pool_kernel(const float* tk, int dim, int pool, const float* kvp, const float* kgp,
                              const float* ape, const float* key_norm, float eps, int rd, const float2* cs, int cs_pos0,
                              int g0, float* out) {
  __shared__ float keys[8][128];
  __shared__ float pooled[128];
  __shared__ float red[32];
  const int g = g0 + blockIdx.x;
  const int d = threadIdx.x;  // blockDim == dim == 128
  for (int s = 0; s < pool; s++) keys[s][d] = tk[((long)g * pool + s) * dim + d];
  __syncthreads();
  float gate[8], val[8], mx = -INFINITY;
  for (int s = 0; s < pool; s++) {
    float a = 0.0f, b = 0.0f;
    for (int i = 0; i < dim; i++) {
      a += kgp[(long)d * dim + i] * keys[s][i];
      b += kvp[(long)d * dim + i] * keys[s][i];
    }
    gate[s] = a + ape[s * dim + d];
    val[s] = b;
    mx = fmaxf(mx, gate[s]);
  }
  float z = 0.0f;
  for (int s = 0; s < pool; s++) {
    gate[s] = expf(gate[s] - mx);
    z += gate[s];
  }
  float v = 0.0f;
  for (int s = 0; s < pool; s++) v += val[s] * (gate[s] / z);
  float sq = r_block_sum(v * v, red);
  const float inv = rsqrtf(sq / dim + eps);
  pooled[d] = key_norm[d] * (v * inv);
  __syncthreads();
  if (d >= dim - rd && ((d - (dim - rd)) & 1) == 0) {
    const int i = (d - (dim - rd)) / 2;
    const float2 c = cs[(long)(g * pool - cs_pos0) * (rd / 2) + i];
    const float x0 = pooled[d], x1 = pooled[d + 1];
    pooled[d] = x0 * c.x - x1 * c.y;
    pooled[d + 1] = x1 * c.x + x0 * c.y;
  }
  __syncthreads();
  out[(long)g * dim + d] = pooled[d];
}

extern "C" int memra_hy4r_pool(const void* tk, int dim, int pool, const float* kvp, const float* kgp, const float* ape,
                               const float* key_norm, float eps, int rd, const void* cs, int cs_pos0, int g0, int n,
                               float* out, void* stream) {
  if (n <= 0) return 0;
  if (dim != 128 || pool > 8) return 45301;
  r_pool_kernel<<<n, dim, 0, (cudaStream_t)stream>>>((const float*)tk, dim, pool, kvp, kgp, ape, key_norm, eps, rd,
                                                     (const float2*)cs, cs_pos0, g0, out);
  return r_rc();
}

// group logits of each query: for g < own = pos / pool, sum_h hw[h] * hscale * relu(q_h . gk_g);
// -inf for the other groups of the call's `ng`
__global__ void r_index_logits_kernel(const float* qi, const float* hw, const float* gk, const int* qpos, int heads,
                                      int dim, int pool, int ng, float hscale, float* out) {
  __shared__ float qs[16 * 128];
  __shared__ float ws[16];
  const int r = blockIdx.x;
  for (int e = threadIdx.x; e < heads * dim; e += blockDim.x) qs[e] = qi[(long)r * heads * dim + e];
  if (threadIdx.x < heads) ws[threadIdx.x] = hw[(long)r * heads + threadIdx.x] * hscale;
  __syncthreads();
  const int own = qpos[r] / pool;
  for (int g = threadIdx.x; g < ng; g += blockDim.x) {
    float v = -INFINITY;
    if (g < own) {
      v = 0.0f;
      const float* k = gk + (long)g * dim;
      for (int h = 0; h < heads; h++) {
        float dot = 0.0f;
        for (int i = 0; i < dim; i++) dot += qs[h * dim + i] * k[i];
        v += ws[h] * fmaxf(dot, 0.0f);
      }
    }
    out[(long)r * ng + g] = v;
  }
}

extern "C" int memra_hy4r_index_logits(const float* qi, const float* hw, const float* gk, const int* qpos, int rows,
                                       int heads, int dim, int pool, int ng, float hscale, float* out, void* stream) {
  if (rows == 0) return 0;
  if (heads > 16 || dim > 128) return 45302;
  r_index_logits_kernel<<<rows, 128, 0, (cudaStream_t)stream>>>(qi, hw, gk, qpos, heads, dim, pool, ng, hscale, out);
  return r_rc();
}

// the recalled groups (the picks plus the own group, sorted, the own group kept twice when it is
// also a pick) and the key rows they hold up to the query, ascending
__global__ void r_recall_kernel(const int* picks, int kp, const int* qpos, int pool, int* sel, int* list, int ldl,
                                int* lens) {
  __shared__ int s[1024];
  const int r = blockIdx.x;
  const int own = qpos[r] / pool;
  const int ks = kp + 1;
  if (threadIdx.x == 0) {
    // insertion of `own` into the ascending picks (after its equals)
    int j = 0, w = 0;
    bool placed = false;
    for (int i = 0; i < kp; i++) {
      const int g = picks[(long)r * kp + i];
      if (!placed && own < g) {
        s[w++] = own;
        placed = true;
      }
      s[w++] = g;
      (void)j;
    }
    if (!placed) s[w++] = own;
    for (int i = 0; i < ks; i++) sel[(long)r * ks + i] = s[i];
    const int p = qpos[r];
    int n = 0, prev = -1;
    for (int i = 0; i < ks; i++) {
      const int g = s[i];
      if (g == prev) continue;
      prev = g;
      for (int o = 0; o < pool; o++) {
        const int key = g * pool + o;
        if (key <= p) list[(long)r * ldl + n++] = key;
      }
    }
    lens[r] = n;
  }
}

extern "C" int memra_hy4r_recall(const int* picks, int kp, const int* qpos, int rows, int pool, int* sel, int* list,
                                 int ldl, int* lens, void* stream) {
  if (rows == 0) return 0;
  if (kp + 1 > 1024 || (kp + 1) * pool > ldl) return 45303;
  r_recall_kernel<<<rows, 32, 0, (cudaStream_t)stream>>>(picks, kp, qpos, pool, sel, list, ldl, lens);
  return r_rc();
}

// forced recall (the oracle's sel, own included): only the key rows
__global__ void r_recall_forced_kernel(const int* sel, int ks, const int* qpos, int pool, int* list, int ldl, int* lens) {
  const int r = blockIdx.x;
  if (threadIdx.x != 0) return;
  const int p = qpos[r];
  int n = 0, prev = -1;
  for (int i = 0; i < ks; i++) {
    const int g = sel[(long)r * ks + i];
    if (g == prev) continue;
    prev = g;
    for (int o = 0; o < pool; o++) {
      const int key = g * pool + o;
      if (key <= p) list[(long)r * ldl + n++] = key;
    }
  }
  lens[r] = n;
}

extern "C" int memra_hy4r_recall_forced(const int* sel, int ks, const int* qpos, int rows, int pool, int* list, int ldl,
                                        int* lens, void* stream) {
  if (rows == 0) return 0;
  if (ks * pool > ldl) return 45304;
  r_recall_forced_kernel<<<rows, 32, 0, (cudaStream_t)stream>>>(sel, ks, qpos, pool, list, ldl, lens);
  return r_rc();
}

// ---- Engram ------------------------------------------------------------------------------------

// emb[t, head*dim + d] = rms(e4m3(code) * row scale, head_norm[head]) per (token, head)
__global__ void r_engram_rows_kernel(const uint8_t* codes, const float* scales, const float* norms, int heads, int dim,
                                     float eps, float* emb) {
  const int t = blockIdx.x, hd = threadIdx.y;
  const int lane = threadIdx.x;
  const uint8_t* c = codes + ((long)t * heads + hd) * dim;
  const float s = scales[(long)t * heads + hd];
  float v[4];
  float sq = 0.0f;
  for (int i = 0; i < 4; i++) {
    const int d = lane + 32 * i;
    v[i] = d < dim ? r_e4m3(c[d]) * s : 0.0f;
    sq += v[i] * v[i];
  }
  sq = r_warp_sum(sq);
  const float inv = rsqrtf(sq / dim + eps);
  for (int i = 0; i < 4; i++) {
    const int d = lane + 32 * i;
    if (d < dim) emb[(long)t * heads * dim + hd * dim + d] = norms[hd * dim + d] * (v[i] * inv);
  }
}

extern "C" int memra_hy4r_engram_rows(const void* codes, const float* scales, const float* norms, int rows, int heads,
                                      int dim, float eps, float* emb, void* stream) {
  if (rows == 0) return 0;
  if (dim > 128 || heads > 32) return 45401;
  r_engram_rows_kernel<<<rows, dim3(32, heads), 0, (cudaStream_t)stream>>>((const uint8_t*)codes, scales, norms, heads,
                                                                          dim, eps, emb);
  return r_rc();
}

// per (token, stream j): key = rms(keyraw_j, norm_key_j), query = rms(s_j, norm_query_j),
// dot = key . query / sqrt(h), gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)));
// s_j += mask ? gate * main + scale_j * main : 0
__global__ void r_engram_mix_kernel(float* s, const float* keyraw, const float* main, const float* norm_key,
                                    const float* norm_query, const float* scales, const uint8_t* mask, int nstream, int h,
                                    float eps) {
  __shared__ float red[32];
  const int t = blockIdx.x, j = blockIdx.y;
  float* sr = s + ((long)t * nstream + j) * h;
  const float* kr = keyraw + ((long)j * gridDim.x + t) * h;
  float sk = 0.0f, ss = 0.0f;
  for (int d = threadIdx.x; d < h; d += blockDim.x) {
    sk += kr[d] * kr[d];
    ss += sr[d] * sr[d];
  }
  sk = r_block_sum(sk, red);
  __syncthreads();
  ss = r_block_sum(ss, red);
  const float ik = rsqrtf(sk / h + eps), is = rsqrtf(ss / h + eps);
  float dot = 0.0f;
  for (int d = threadIdx.x; d < h; d += blockDim.x)
    dot += (norm_key[(long)j * h + d] * (kr[d] * ik)) * (norm_query[(long)j * h + d] * (sr[d] * is));
  __syncthreads();
  dot = r_block_sum(dot, red) * rsqrtf((float)h);
  const float sign = dot > 0.0f ? 1.0f : (dot < 0.0f ? -1.0f : 0.0f);
  const float gate = 1.0f / (1.0f + expf(-(sqrtf(fmaxf(fabsf(dot), 1e-6f)) * sign)));
  __syncthreads();
  if (!mask[t]) return;
  for (int d = threadIdx.x; d < h; d += blockDim.x) {
    const float mv = main[(long)t * h + d];
    sr[d] = sr[d] + (gate * mv + scales[(long)j * h + d] * mv);
  }
}

extern "C" int memra_hy4r_engram_mix(float* s, const float* keyraw, const float* main, const float* norm_key,
                                     const float* norm_query, const float* scales, const void* mask, int rows,
                                     int nstream, int h, float eps, void* stream) {
  if (rows == 0) return 0;
  r_engram_mix_kernel<<<dim3(rows, nstream), 256, 0, (cudaStream_t)stream>>>(s, keyraw, main, norm_key, norm_query,
                                                                            scales, (const uint8_t*)mask, nstream, h, eps);
  return r_rc();
}

// ---- copies ------------------------------------------------------------------------------------

extern "C" int memra_hy4r_h2d(void* dst, const void* src, size_t bytes, void* stream) {
  if (bytes == 0) return 0;
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

extern "C" int memra_hy4r_d2h(void* dst, const void* src, size_t bytes, void* stream) {
  if (bytes == 0) return 0;
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
  if (e != cudaSuccess) return 10000 + (int)e;
  e = cudaStreamSynchronize((cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

extern "C" int memra_hy4r_memset(void* dst, int v, size_t bytes, void* stream) {
  if (bytes == 0) return 0;
  cudaError_t e = cudaMemsetAsync(dst, v, bytes, (cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

extern "C" int memra_hy4r_sync(void* stream) {
  cudaError_t e = cudaStreamSynchronize((cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}
