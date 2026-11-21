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

#include <utility>

#define R_BF16 0
#define R_FP8 2
#define R_NVFP4 3

static __device__ __forceinline__ float r_e4m3(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

// e2m1 code -> value {0, 0.5, 1, 1.5, 2, 3, 4, 6} and sign, from its bits (no indexed table,
// which would live in local memory): c >= 2 is 2^((c >> 1) - 1) * (1 + (c & 1) / 2)
static __device__ __forceinline__ float r_e2m1(uint32_t nib) {
  const uint32_t c = nib & 7u;
  const uint32_t mag = c >= 2u ? ((126u + (c >> 1)) << 23) | ((c & 1u) << 22) : (c ? 0x3F000000u : 0u);
  return __uint_as_float(mag | ((nib & 8u) << 28));
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

// ---- programmatic dependent launch ------------------------------------------------------------
//
// Every kernel of the route is launched with programmatic stream serialization (compute
// capability 9.0 and up): a kernel may start while its predecessor finishes, and waits for it
// (`griddepcontrol.wait`: the predecessor complete and its writes visible) before it reads
// anything a predecessor wrote or writes any global memory. Before the wait a kernel only reads
// the resident weights into shared memory (the small-row GEMMs prefetch their first stages
// there), so the overlap changes no value. A kernel lets its dependent launch
// (`launch_dependents`) once its own blocks are running; the dependent then waits as above.

static __device__ __forceinline__ void r_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

static __device__ __forceinline__ void r_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

static bool r_pdl_supported() {
  static int cc = -1;
  if (cc < 0) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cc = major;
  }
  return cc >= 9;
}

template <typename... KArgs, typename... Args>
static void r_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t st, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = r_pdl_supported() ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

// ---- the weight GEMM ------------------------------------------------------------------------
//
// The GEMM weights are resident in a tile layout (`memra_hy4r_tile`): the stored bytes of 32 rows
// x 128 bytes contiguous, tiles of a 32-row block in K order, blocks in row order; rows padded to
// a multiple of 32 with zeros, a row a multiple of 128 bytes. Within a tile, row r's 16-byte
// pieces sit XOR-swizzled by r % 8, so the 8 rows a warp reads land on distinct banks when the
// tile is copied to shared memory as is. A decode GEMM's stage of a 32-row block is one 4 KiB
// bulk copy. The values are the stored ones, so no result moves.

#define RT_ROWS 32
#define RT_BYTES 128

// byte `b` of row `row` in the tile layout of rows of `rb` bytes
static __host__ __device__ __forceinline__ long r_toff(long row, long b, long rb) {
  const long rt = row % RT_ROWS, piece = ((b % RT_BYTES) / 16) ^ (rt & 7);
  return ((row / RT_ROWS) * (rb / RT_BYTES) + b / RT_BYTES) * (RT_ROWS * RT_BYTES) + rt * RT_BYTES + piece * 16 + b % 16;
}

// dst (tile layout, `batch` entries dst_bs bytes apart) from src rows (src_ld bytes apart, batch
// src_bs), one thread a 16-byte piece
__global__ void r_tile_kernel(const uint8_t* src, long rows, long rb, long src_ld, long src_bs, uint8_t* dst,
                              long dst_bs, long pieces) {
  const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= pieces) return;
  const int b = blockIdx.y;
  const long tile = e / (RT_ROWS * RT_BYTES / 16), in = e % (RT_ROWS * RT_BYTES / 16);
  const long nst = rb / RT_BYTES;
  const long cb = tile / nst, st = tile % nst;
  const long rt = in / (RT_BYTES / 16), r = cb * RT_ROWS + rt;
  const long piece = (in % (RT_BYTES / 16)) ^ (rt & 7);  // the stored piece at this place
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (r < rows) v = *(const uint4*)(src + b * src_bs + r * src_ld + st * RT_BYTES + piece * 16);
  *(uint4*)(dst + b * dst_bs + e * 16) = v;
}

// bytes the tile layout of `rows` rows of `rb` bytes takes
extern "C" long memra_hy4r_tile_bytes(long rows, long rb) {
  return (rows + RT_ROWS - 1) / RT_ROWS * RT_ROWS * rb;
}

extern "C" int memra_hy4r_tile(const void* src, long rows, long rb, long src_ld, long src_bs, void* dst, long dst_bs,
                               int batch, void* stream) {
  if (rows <= 0 || rb <= 0 || batch <= 0) return 0;
  if (rb % RT_BYTES != 0 || src_ld % 16 != 0 || src_bs % 16 != 0 || dst_bs % 16 != 0) return 45009;
  if (dst_bs < memra_hy4r_tile_bytes(rows, rb)) return 45009;
  const long pieces = memra_hy4r_tile_bytes(rows, rb) / 16;
  r_tile_kernel<<<dim3((unsigned)((pieces + 255) / 256), batch), 256, 0, (cudaStream_t)stream>>>(
      (const uint8_t*)src, rows, rb, src_ld, src_bs, (uint8_t*)dst, dst_bs, pieces);
  return r_rc();
}

// One output of a small-row launch that runs several GEMMs of the same K, format and rows at once
// (`memra_hy4r_gemm_multi`): its input, weight, output and first column block.
#define R_MAXSEG 6
struct RSeg {
  const float* x;
  long ldx;
  const uint8_t* w;
  const uint8_t* sc;
  const float* s2;
  float* y;
  long ldy;
  int n, grid_cols;
  int gx0;  // first blockIdx.x of this output
};

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
  int tile_rows;     // small-row kernel, grouped mode: the most rows a tile holds
  int nseg;          // small-row kernel: outputs of the launch
  RSeg seg[R_MAXSEG];
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
  r_pdl_wait();
  r_pdl_trigger();
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
        wraw[0] = *(const uint4*)(w + r_toff(gn, ((long)k0 + wk) * 2, (long)k * 2));
        wraw[1] = *(const uint4*)(w + r_toff(gn, ((long)k0 + wk) * 2 + 16, (long)k * 2));
      } else if (FMT == R_FP8) {
        wraw[0] = *(const uint4*)(w + r_toff(gn, (long)k0 + wk, (long)k));
      } else {
        uint2 c = *(const uint2*)(w + r_toff(gn, ((long)k0 + wk) / 2, (long)k / 2));
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

// ---- the small-row GEMM (decode) ----------------------------------------------------------------
//
// Up to 16 rows (a decode call, a grouped expert tile of a few tokens). A block of 4 warps owns
// 32 output columns, a warp one n8 tile, and walks the whole K: the weight bytes, the NVFP4 group
// scales, the FP8 block scale and the activation rows stream through a ring of cp.async stages
// (one block barrier a stage). With up to 4 rows (`MR`), each warp splits a stage's activations
// into their hi/lo BF16 pairs once, in fragment order. Every output element sees the same
// sequence as in `r_gemm_kernel`: per 16-wide K step one mma of the hi term then one of the lo
// term into the same accumulator, the FP8 block scale folded with the same fma after each 128
// inputs (and at the end), `weight_scale_2` on the finished sum; the B operand holds the same
// exact BF16 values (FP8 codes, and NVFP4 codes times their group scale, are exact in BF16). An
// mma's output element depends only on its own A row, B column and accumulator, so the bits
// equal the tiled kernel's. `RED` is the gate's dequantized-weight red arm.

#define GV_WARPS 4
#define GV_COLS (GV_WARPS * 8)
#define GV_ROWB 128               // stored weight bytes a row a stage
#define GV_LDS GV_ROWB            // smem weight row stride: the tile as stored (its swizzle spreads banks)
#define GV_SCS 32                 // NVFP4 scale bytes a row a stage (16 used)
#define GV_SMEM_MAX (96 * 1024)
static_assert(GV_COLS == RT_ROWS && GV_ROWB == RT_BYTES, "a small-row stage is one weight tile");

static __device__ __forceinline__ void r_cp16(void* dst, const void* src, bool ok) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const int sz = ok ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(sz));
}
static __device__ __forceinline__ void r_cp4(void* dst, const void* src) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
// a stage's weight tile: one bulk copy completing on the stage's mbarrier
static __device__ __forceinline__ void r_mb_init(uint64_t* mb) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(mb)));
}
static __device__ __forceinline__ void r_bulk(void* dst, const void* src, unsigned bytes, uint64_t* mb) {
  const unsigned m = (unsigned)__cvta_generic_to_shared(mb), d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(m), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(d),
               "l"(src), "r"(bytes), "r"(m)
               : "memory");
}
static __device__ __forceinline__ void r_mb_wait(uint64_t* mb, unsigned parity) {
  const unsigned m = (unsigned)__cvta_generic_to_shared(mb);
  unsigned ok = 0;
  while (!ok)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok)
                 : "r"(m), "r"(parity)
                 : "memory");
}
static __device__ __forceinline__ void r_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
static __device__ __forceinline__ void r_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

static __device__ __forceinline__ uint32_t r_bf2_bits(__nv_bfloat162 v) { return *(uint32_t*)&v; }

// e4m3 pair (low byte first) -> exact BF16 pair: through F16 (exact), then F32 (exact)
static __device__ __forceinline__ uint32_t r_e4m3x2_bf16x2(uint16_t c) {
  const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(c, __NV_E4M3)));
  return r_bf2_bits(__floats2bfloat162_rn(f.x, f.y));
}

template <int FMT>
struct GvGeom {
  static constexpr int KPS = FMT == R_BF16 ? GV_ROWB / 2 : (FMT == R_FP8 ? GV_ROWB : GV_ROWB * 2);
  static constexpr int NQ = KPS / 16;                                // 16-wide K steps a stage
  static constexpr int WB = GV_COLS * GV_LDS;                        // weight bytes a stage
  static constexpr int SB = FMT == R_NVFP4 ? GV_COLS * GV_SCS : (FMT == R_FP8 ? 16 : 0);
  static constexpr int XLD = KPS * 4 + 16;                           // activation row stride
  static constexpr int LUT = FMT == R_NVFP4 ? 1024 : 0;
  // per warp, split pairs of a stage: NQ x rows x 4 lanes x uint4
  static constexpr int AB1 = NQ * 4 * 16;
  static __host__ __device__ int stage_bytes(int m) { return WB + SB + m * XLD; }
  static __host__ __device__ int fixed_bytes(int mr) { return LUT + (mr ? GV_WARPS * mr * AB1 : 0); }
};

// MR: 1..4 rows (pre-split activations), 0 for up to 16 rows
template <int FMT, int STAGES, int MR, bool RED>
__global__ void __launch_bounds__(GV_WARPS * 32) r_gemv_kernel(RGemm g) {
  // pdl: resident weights before the wait (plain mode; grouped mode waits first for its tiles)
  using G = GvGeom<FMT>;
  constexpr int NT = GV_WARPS * 32;
  extern __shared__ __align__(16) unsigned char gsm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // this block's output: the last whose first column block is at or before blockIdx.x
  int si = 0;
#pragma unroll
  for (int t = 1; t < R_MAXSEG; t++)
    if (t < g.nseg && (int)blockIdx.x >= g.seg[t].gx0) si = t;
  const RSeg& sg = g.seg[si];
  const float* x = sg.x;
  const uint8_t* w = sg.w;
  const uint8_t* sc = sg.sc;
  float* y = sg.y;
  const long ldx = sg.ldx, ldy = sg.ldy;
  float s2 = 1.0f;
  int m = g.m, m0 = 0;
  r_pdl_trigger();
  if (g.tiles) {
    // the tile table is a predecessor's output: wait before reading it
    r_pdl_wait();
    const int e = g.tiles[blockIdx.y * 3], r0 = g.tiles[blockIdx.y * 3 + 1], rows = g.tiles[blockIdx.y * 3 + 2];
    if (rows <= 0) return;
    x += (long)r0 * ldx;
    y += (long)r0 * ldy;
    w += (long)e * g.wbs;
    if (sc) sc += (long)e * g.scbs;
    if (sg.s2) s2 = sg.s2[e];
    m = rows;
  } else {
    const int b = blockIdx.z;
    m0 = blockIdx.y * 16;
    x += (long)b * g.xbs;
    y += (long)b * g.ybs;
    w += (long)b * g.wbs;
    if (sc) sc += (long)b * g.scbs;
    if (sg.s2) s2 = sg.s2[b];
  }
  const int mr = min(MR ? MR : 16, m - m0);  // rows of this block
  constexpr int MS = MR ? MR : 16;           // rows the smem holds
  const int n = sg.n, k = g.k;
  const int nb0 = (blockIdx.x - sg.gx0) * GV_COLS;
  constexpr int SBYTES = G::WB + G::SB + MS * G::XLD;
  unsigned char* fixed = gsm + STAGES * SBYTES;
  uint32_t* lut = (uint32_t*)fixed;
  uint4* ab = (uint4*)(fixed + G::LUT) + warp * (MS * G::AB1 / 16);
  const long wrow = FMT == R_BF16 ? (long)k * 2 : (FMT == R_FP8 ? (long)k : (long)k / 2);
  const int nst = (k + G::KPS - 1) / G::KPS;
  const int srow = k / 16;  // NVFP4 scale bytes a row
  const bool sc_async = FMT == R_NVFP4 && (srow % 16) == 0;
  const float* xb = x + (long)m0 * ldx;
  if (FMT == R_NVFP4 && !RED)
    for (int c = tid; c < 256; c += NT) lut[c] = r_bf2_bits(__floats2bfloat162_rn(r_e2m1(c), r_e2m1(c >> 4)));
  // the block's 32 rows: a stage is one contiguous tile (see r_toff), rows past n zero
  const unsigned char* wtile = w + r_toff(nb0, 0, wrow);
  __shared__ uint64_t mbar[STAGES];  // a stage's weight tile landed
  if (tid == 0) {
    for (int i = 0; i < STAGES; i++) r_mb_init(&mbar[i]);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  // a stage's weight side (resident weights and scales) and its activation side (a
  // predecessor's output)
  auto issue_w = [&](int st) {
    if (st < nst) {
      unsigned char* buf = gsm + (st % STAGES) * SBYTES;
      if (tid == 0) r_bulk(buf, wtile + (long)st * (RT_ROWS * RT_BYTES), RT_ROWS * RT_BYTES, &mbar[st % STAGES]);
      const int k0 = st * G::KPS;
      if (FMT == R_NVFP4) {
        const int s0 = k0 / 16;
        unsigned char* sbuf = buf + G::WB;
        if (sc_async) {
          if (tid < GV_COLS) {
            const int gn = nb0 + tid;
            r_cp16(sbuf + tid * GV_SCS, sc + (long)min(gn, n - 1) * srow + s0, gn < n && s0 < srow);
          }
        } else {
          const int sb = min(G::KPS / 16, srow - s0);
          for (int c = tid; c < GV_COLS * 16; c += NT) {
            const int r = c >> 4, q = c & 15;
            const int gn = nb0 + r;
            sbuf[r * GV_SCS + q] = gn < n && q < sb ? sc[(long)gn * srow + s0 + q] : 0;
          }
        }
      } else if (FMT == R_FP8) {
        // the block scale of these 32 rows (one 128-row block) for this stage's 128 inputs
        if (tid == 0) r_cp4(buf + G::WB, (const float*)sc + (long)(nb0 / 128) * sg.grid_cols + k0 / 128);
      }
    }
  };
  auto issue_x = [&](int st) {
    if (st < nst) {
      unsigned char* buf = gsm + (st % STAGES) * SBYTES;
      const int k0 = st * G::KPS;
      // activation rows: the stage's inputs of each of the block's rows
      constexpr int CX = G::KPS / 4;  // 16-byte chunks a row
      unsigned char* xs = buf + G::WB + G::SB;
      for (int c = tid; c < mr * CX; c += NT) {
        const int r = c / CX, q = c % CX;
        r_cp16(xs + r * G::XLD + q * 16, xb + (long)r * ldx + k0 + q * 4, k0 + q * 4 < k);
      }
    }
  };
  auto issue = [&](int st) {
    issue_w(st);
    issue_x(st);
    r_cp_commit();
  };

  // the first stages' weights stream in before the wait, their activations after it; groups
  // W0..W(S-2), X0..X(S-2): the loop's wait at stage s still covers W(s) and X(s)
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    issue_w(s);
    r_cp_commit();
  }
  if (!g.tiles) r_pdl_wait();
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    issue_x(s);
    r_cp_commit();
  }

  const int r = lane >> 2, j = lane & 3;
  const int n0 = nb0 + warp * 8;
  const bool a0 = r < mr, a1 = r + 8 < mr;
  constexpr bool fold_blocks = FMT == R_FP8 && !RED;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int st = 0; st < nst; st++) {
    r_cp_wait<STAGES - 2>();
    r_mb_wait(&mbar[st % STAGES], (st / STAGES) & 1);
    __syncthreads();  // stage st landed for every thread; stage st - 1 is free to refill
    issue(st + STAGES - 1);
    const unsigned char* buf = gsm + (st % STAGES) * SBYTES;
    const unsigned char* brow = buf + (warp * 8 + r) * GV_LDS;
    const int sw = r & 7;  // the row's piece swizzle (row (warp * 8 + r) % 8 == r)
    const unsigned char* xs = buf + G::WB + G::SB;
    const int kbase = st * G::KPS;
    const int nq = min(G::KPS, k - kbase) / 16;
    float bs = 1.0f;
    if (FMT == R_FP8) bs = *(const float*)(buf + G::WB);
    if (MR) {
      // ab[(q * MR + row) * 4 + jj] = {hi(jj), hi(jj + 4), lo(jj), lo(jj + 4)}, pair p = inputs 2p, 2p + 1
#pragma unroll
      for (int it = 0; it < (MS * G::NQ * 8 + 31) / 32; it++) {
        const int p = lane + it * 32;
        if (p < MS * G::NQ * 8) {
          const int row = p / (G::NQ * 8), rem = p % (G::NQ * 8), q = rem >> 3, pp = rem & 7;
          if (row < mr) {
            const float2 v = *(const float2*)((const float*)(xs + row * G::XLD) + q * 16 + 2 * pp);
            uint32_t* d = (uint32_t*)(ab + (q * MS + row) * 4 + (pp & 3));
            d[pp >> 2] = r_pack(v.x, v.y);
            d[2 + (pp >> 2)] = r_pack(v.x - r_bf(r_tobf(v.x)), v.y - r_bf(r_tobf(v.y)));
          }
        }
      }
      __syncwarp();
    }
    if (n0 < n) {
#pragma unroll
      for (int q = 0; q < G::NQ; q++) {
        if (q >= nq) break;
        const int kk = kbase + q * 16;
        uint32_t ah[4] = {0u, 0u, 0u, 0u}, al[4] = {0u, 0u, 0u, 0u};
        if (MR) {
          if (a0) {
            const uint4 v = ab[(q * MS + r) * 4 + j];
            ah[0] = v.x;
            ah[2] = v.y;
            al[0] = v.z;
            al[2] = v.w;
          }
        } else {
          const float* x0 = (const float*)(xs + r * G::XLD);
          const float* x1 = (const float*)(xs + (r + 8) * G::XLD);
          float2 v00 = make_float2(0.f, 0.f), v02 = v00, v10 = v00, v12 = v00;
          if (a0) {
            v00 = *(const float2*)(x0 + q * 16 + 2 * j);
            v02 = *(const float2*)(x0 + q * 16 + 8 + 2 * j);
          }
          if (a1) {
            v10 = *(const float2*)(x1 + q * 16 + 2 * j);
            v12 = *(const float2*)(x1 + q * 16 + 8 + 2 * j);
          }
          ah[0] = r_pack(v00.x, v00.y);
          ah[1] = r_pack(v10.x, v10.y);
          ah[2] = r_pack(v02.x, v02.y);
          ah[3] = r_pack(v12.x, v12.y);
          al[0] = r_pack(v00.x - r_bf(r_tobf(v00.x)), v00.y - r_bf(r_tobf(v00.y)));
          al[1] = r_pack(v10.x - r_bf(r_tobf(v10.x)), v10.y - r_bf(r_tobf(v10.y)));
          al[2] = r_pack(v02.x - r_bf(r_tobf(v02.x)), v02.y - r_bf(r_tobf(v02.y)));
          al[3] = r_pack(v12.x - r_bf(r_tobf(v12.x)), v12.y - r_bf(r_tobf(v12.y)));
        }
        uint32_t b[2];
        if (FMT == R_BF16) {
          b[0] = *(const uint32_t*)(brow + (((2 * q) ^ sw) * 16) + j * 4);
          b[1] = *(const uint32_t*)(brow + (((2 * q + 1) ^ sw) * 16) + j * 4);
        } else if (FMT == R_FP8) {
          const uint16_t c0 = *(const uint16_t*)(brow + ((q ^ sw) * 16) + 2 * j);
          const uint16_t c1 = *(const uint16_t*)(brow + ((q ^ sw) * 16) + 8 + 2 * j);
          if (!RED) {
            b[0] = r_e4m3x2_bf16x2(c0);
            b[1] = r_e4m3x2_bf16x2(c1);
          } else {
            b[0] = r_pack(r_e4m3((uint8_t)c0) * bs, r_e4m3((uint8_t)(c0 >> 8)) * bs);
            b[1] = r_pack(r_e4m3((uint8_t)c1) * bs, r_e4m3((uint8_t)(c1 >> 8)) * bs);
          }
        } else {
          const unsigned char* cp = brow + (((q >> 1) ^ sw) * 16) + (q & 1) * 8;
          const uint8_t c0 = cp[j], c1 = cp[4 + j];
          const float gsf = r_e4m3(buf[G::WB + (warp * 8 + r) * GV_SCS + q]);
          if (!RED) {
            // code pair times the group scale in BF16: the product is exact, so it equals the
            // F32 product rounded to BF16
            const __nv_bfloat162 gs2 = __float2bfloat162_rn(gsf);
            const uint32_t l0 = lut[c0], l1 = lut[c1];
            b[0] = r_bf2_bits(__hmul2(*(const __nv_bfloat162*)&l0, gs2));
            b[1] = r_bf2_bits(__hmul2(*(const __nv_bfloat162*)&l1, gs2));
          } else {
            const float gs = gsf * s2;
            b[0] = r_pack(r_e2m1(c0) * gs, r_e2m1(c0 >> 4) * gs);
            b[1] = r_pack(r_e2m1(c1) * gs, r_e2m1(c1 >> 4) * gs);
          }
        }
        float* c = fold_blocks ? part : acc;
        r_mma(c, ah, b);
        r_mma(c, al, b);
        if (fold_blocks && (((kk + 16) % 128) == 0 || kk + 16 >= k)) {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            acc[i] = __fmaf_rn(part[i], bs, acc[i]);
            part[i] = 0.0f;
          }
        }
      }
    }
  }
  r_cp_wait<0>();
  if (n0 >= n) return;
  const bool scale2 = FMT == R_NVFP4 && !RED;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int row = r + (i >= 2 ? 8 : 0);
    const int col = n0 + 2 * j + (i & 1);
    if (row < mr && col < n) {
      float v = acc[i];
      if (scale2) v = __fmul_rn(v, s2);
      y[(long)(m0 + row) * ldy + col] = v;
    }
  }
}

template <int FMT, int STAGES, int MR, bool RED>
static int r_gemv_go(const RGemm& g, dim3 grid, cudaStream_t st) {
  using G = GvGeom<FMT>;
  constexpr int MS = MR ? MR : 16;
  const int smem = STAGES * (G::WB + G::SB + MS * G::XLD) + G::fixed_bytes(MR);
  if (smem > GV_SMEM_MAX) return 45007;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(r_gemv_kernel<FMT, STAGES, MR, RED>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         GV_SMEM_MAX);
    attr = true;
  }
  r_launch(r_gemv_kernel<FMT, STAGES, MR, RED>, grid, GV_WARPS * 32, smem, st, g);
  return r_rc();
}

template <int FMT, bool RED>
static int r_gemv_rows(const RGemm& g, dim3 grid, int mr, cudaStream_t st) {
  switch (mr) {
    case 1: return r_gemv_go<FMT, 4, 1, RED>(g, grid, st);
    case 2: return r_gemv_go<FMT, 4, 2, RED>(g, grid, st);
    case 3: return r_gemv_go<FMT, 4, 3, RED>(g, grid, st);
    case 4: return r_gemv_go<FMT, 4, 4, RED>(g, grid, st);
    default: return r_gemv_go<FMT, 3, 0, RED>(g, grid, st);
  }
}

template <int FMT>
static int r_gemv_fmt(const RGemm& g, dim3 grid, int mr, cudaStream_t st) {
  return g.red_dq ? r_gemv_rows<FMT, true>(g, grid, mr, st) : r_gemv_rows<FMT, false>(g, grid, mr, st);
}

static int r_gemv_launch(RGemm g, int fmt, int batch, int tiles, int tile_rows, cudaStream_t st) {
  if (!g.tiles && g.m > 16) return 45007;
  if (g.nseg == 0) {
    // one output: the plain fields
    g.nseg = 1;
    g.seg[0] = RSeg{g.x, g.ldx, g.w, g.sc, g.s2, g.y, g.ldy, g.n, g.grid_cols, 0};
  }
  if (g.nseg > R_MAXSEG) return 45008;
  int gx = 0;
  for (int t = 0; t < g.nseg; t++) {
    if ((g.seg[t].ldx % 4) != 0 || (((uintptr_t)g.seg[t].x) % 16) != 0) return 45005;
    if (g.seg[t].n <= 0) return 45001;
    if (fmt != R_BF16 && g.seg[t].sc == nullptr) return 45003;
    g.seg[t].gx0 = gx;
    gx += (g.seg[t].n + GV_COLS - 1) / GV_COLS;
  }
  dim3 grid(gx, g.tiles ? tiles : (g.m + 15) / 16, g.tiles ? 1 : batch);
  if (grid.y == 0) return 0;
  if (grid.y > 65535 || grid.z > 65535) return 45004;
  // grouped tiles hold up to `tile_rows` rows (a token has one row an expert)
  if (g.tiles && (tile_rows <= 0 || tile_rows > 16)) return 45007;
  const int mr = g.tiles ? tile_rows : (g.m < 16 ? g.m : 16);
  if (fmt == R_BF16) return r_gemv_fmt<R_BF16>(g, grid, mr, st);
  if (fmt == R_FP8) return r_gemv_fmt<R_FP8>(g, grid, mr, st);
  return r_gemv_fmt<R_NVFP4>(g, grid, mr, st);
}

template <int BM>
static int r_gemm_launch(const RGemm& g, int fmt, int batch, int tiles, cudaStream_t st) {
  dim3 grid((g.n + RG_BN - 1) / RG_BN, g.tiles ? tiles : (g.m + BM - 1) / BM, g.tiles ? 1 : batch);
  if (grid.y == 0) return 0;
  if (grid.y > 65535 || grid.z > 65535) return 45004;
  if (fmt == R_BF16)
    r_launch(r_gemm_kernel<BM, R_BF16>, grid, 128, 0, st, g);
  else if (fmt == R_FP8)
    r_launch(r_gemm_kernel<BM, R_FP8>, grid, 128, 0, st, g);
  else
    r_launch(r_gemm_kernel<BM, R_NVFP4>, grid, 128, 0, st, g);
  return r_rc();
}

// `small` picks the small-row kernel (a decode call, up to 16 rows; grouped tiles of up to
// `tile_rows` rows); the result does not depend on it.
extern "C" int memra_hy4r_gemm(const float* x, long ldx, long xbs, const void* w, long wbs, const void* sc, long scbs,
                               const float* s2, int fmt, int grid_cols, float* y, long ldy, long ybs, int m, int n,
                               int k, int batch, const int* tiles, int ntiles, int red_dq, int small, int tile_rows,
                               void* stream) {
  if (n <= 0 || k <= 0 || (k % RG_BK) != 0 || m < 0) return 45001;
  if (fmt != R_BF16 && fmt != R_FP8 && fmt != R_NVFP4) return 45002;
  if (fmt != R_BF16 && sc == nullptr) return 45003;
  if ((ldx % 4) != 0 || (xbs % 4) != 0 || (((uintptr_t)x) % 16) != 0) return 45005;
  // the weight's rows are whole tiles (see r_toff)
  const long rb = fmt == R_BF16 ? (long)k * 2 : (fmt == R_FP8 ? (long)k : (long)k / 2);
  if (rb % RT_BYTES != 0) return 45009;
  if (!tiles && m == 0) return 0;
  RGemm g{x, ldx, xbs, (const uint8_t*)w, wbs, (const uint8_t*)sc, scbs, s2, y, ldy, ybs, m, n, k, grid_cols, tiles, red_dq,
          tile_rows, 0, {}};
  cudaStream_t st = (cudaStream_t)stream;
  return small ? r_gemv_launch(g, fmt, batch, ntiles, tile_rows, st) : r_gemm_launch<64>(g, fmt, batch, ntiles, st);
}

// Up to R_MAXSEG small-row GEMMs of one format and K in one launch (decode: the projections that
// read the same input, the routed experts' gate and up): each output's bits are those of its own
// launch. `segs` holds per output (x, ldx, w, sc, s2, y, ldy, n, grid_cols); the batch strides,
// tile table and rows are shared.
extern "C" int memra_hy4r_gemm_multi(const RSeg* segs, int nseg, int fmt, long xbs, long wbs, long scbs, long ybs, int m,
                                     int k, int batch, const int* tiles, int ntiles, int red_dq, int tile_rows,
                                     void* stream) {
  if (nseg <= 0 || nseg > R_MAXSEG) return 45008;
  if (k <= 0 || (k % RG_BK) != 0 || m < 0) return 45001;
  if (fmt != R_BF16 && fmt != R_FP8 && fmt != R_NVFP4) return 45002;
  if ((fmt == R_BF16 ? (long)k * 2 : (fmt == R_FP8 ? (long)k : (long)k / 2)) % RT_BYTES != 0) return 45009;
  if (!tiles && m == 0) return 0;
  RGemm g{segs[0].x, segs[0].ldx, xbs, segs[0].w, wbs, segs[0].sc, scbs, segs[0].s2, segs[0].y, segs[0].ldy, ybs, m,
          segs[0].n, k, segs[0].grid_cols, tiles, red_dq, tile_rows, nseg, {}};
  for (int t = 0; t < nseg; t++) g.seg[t] = segs[t];
  return r_gemv_launch(g, fmt, batch, ntiles, tile_rows, (cudaStream_t)stream);
}

// ---- the F32 islands' small-row GEMM (router logits, head) ---------------------------------------
//
// y[r, o] = sum_k x[r, k] * w[o, k], BF16 weight, F32 inputs, up to 4 rows. Each output is one
// lane's chain in the order of `hy4f_gemm_kernel` (cu/hy4f_gpu.cu, built -fmad=false): per 16
// inputs a partial sum from zero of separately rounded products, added to the total, so the
// bits equal that kernel's for the same row. A warp owns 32 weight rows (a lane each), streamed
// coalesced through a per-warp cp.async ring.

#define GF_MAXM 4

template <int WARPS, int STAGES, int ROWB>
__global__ void __launch_bounds__(WARPS * 32) r_gemv_f32_kernel(const float* __restrict__ x, long ldx,
                                                               const uint16_t* __restrict__ w, float* __restrict__ y,
                                                               long ldy, int m, int n, int k) {
  // pdl: resident weights before the wait
  constexpr int LDS = ROWB + 16;
  constexpr int XB = ROWB * 2;  // the stage's ROWB / 2 inputs of an activation row, F32
  extern __shared__ __align__(16) unsigned char fsm[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int STAGEB = 32 * LDS + m * XB;
  const int o0 = (blockIdx.x * WARPS + warp) * 32;
  r_pdl_trigger();
  if (o0 >= n) return;
  unsigned char* ring = fsm + warp * (STAGES * STAGEB);
  const long wrow = (long)k * 2;
  const int nst = (int)((wrow + ROWB - 1) / ROWB);
  auto issue_w = [&](int st) {
    if (st < nst) {
      unsigned char* buf = ring + (st % STAGES) * STAGEB;
      const long kb0 = (long)st * ROWB;
      const int cpr = (int)min((long)ROWB, wrow - kb0) >> 4;
      for (int c = lane; c < 32 * cpr; c += 32) {
        const int r = c / cpr, q = c - r * cpr;
        const int o = o0 + r;
        r_cp16(buf + r * LDS + q * 16, (const unsigned char*)w + (long)min(o, n - 1) * wrow + kb0 + q * 16, o < n);
      }
    }
  };
  auto issue_x = [&](int st) {
    if (st < nst) {
      unsigned char* buf = ring + (st % STAGES) * STAGEB;
      const long kb0 = (long)st * ROWB;
      const int cpr = (int)min((long)ROWB, wrow - kb0) >> 4;
      // the activation rows' inputs of this stage
      const int cx = cpr * 2;  // 16-byte chunks of F32 a row
      for (int c = lane; c < m * cx; c += 32) {
        const int i = c / cx, q = c - i * cx;
        r_cp16(buf + 32 * LDS + i * XB + q * 16, x + (long)i * ldx + kb0 / 2 + q * 4, true);
      }
    }
  };
  auto issue = [&](int st) {
    issue_w(st);
    issue_x(st);
    r_cp_commit();
  };
  // weights of the first stages before the wait (see r_gemv_kernel)
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    issue_w(s);
    r_cp_commit();
  }
  r_pdl_wait();
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    issue_x(s);
    r_cp_commit();
  }
  float acc[GF_MAXM];
#pragma unroll
  for (int i = 0; i < GF_MAXM; i++) acc[i] = 0.0f;
  for (int st = 0; st < nst; st++) {
    r_cp_wait<STAGES - 2>();
    __syncwarp();
    issue(st + STAGES - 1);
    const unsigned char* row = ring + (st % STAGES) * STAGEB + lane * LDS;
    const float* xsm = (const float*)(ring + (st % STAGES) * STAGEB + 32 * LDS);
    const int kbase = st * (ROWB / 2);
    const int nk = min(ROWB / 2, k - kbase);
#pragma unroll
    for (int c = 0; c < ROWB / 2; c += 16) {
      if (c >= nk) break;
      const uint4 u0 = *(const uint4*)(row + c * 2);
      const uint4 u1 = *(const uint4*)(row + c * 2 + 16);
      const uint32_t wv[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
      for (int i = 0; i < GF_MAXM; i++) {
        if (i < m) {
          const float4* xp = (const float4*)(xsm + i * (XB / 4) + c);
          const float4 xa = xp[0], xb = xp[1], xc = xp[2], xd = xp[3];
          const float xv[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                                xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};
          float p = 0.0f;
#pragma unroll
          for (int e = 0; e < 16; e++) {
            const uint32_t h = (e & 1) ? (wv[e >> 1] >> 16) : (wv[e >> 1] & 0xffffu);
            p = __fadd_rn(p, __fmul_rn(xv[e], __uint_as_float(h << 16)));
          }
          acc[i] = __fadd_rn(acc[i], p);
        }
      }
    }
  }
  r_cp_wait<0>();
  const int o = o0 + lane;
  if (o < n) {
#pragma unroll
    for (int i = 0; i < GF_MAXM; i++)
      if (i < m) y[(long)i * ldy + o] = acc[i];
  }
}

template <int WARPS, int STAGES, int ROWB>
static void r_gemv_f32_go(const float* x, long ldx, const void* w, float* y, long ldy, int m, int n, int k,
                          cudaStream_t st) {
  const int smem = WARPS * STAGES * (32 * (ROWB + 16) + m * ROWB * 2);
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(r_gemv_f32_kernel<WARPS, STAGES, ROWB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         GV_SMEM_MAX);
    attr = true;
  }
  const int rows = WARPS * 32;
  r_launch(r_gemv_f32_kernel<WARPS, STAGES, ROWB>, (n + rows - 1) / rows, WARPS * 32, smem, st, 
      x, ldx, (const uint16_t*)w, y, ldy, m, n, k);
}

// A narrow output (the router: 512 of 3072): the same chains split at their natural seams. A
// thread computes one 16-input partial sum (from zero, separately rounded products, in input
// order) of one (row, output, step); then one thread an output adds its partials in step order.
// The bits equal one lane's whole chain.
#define GS_OB 4
#define GS_T 256
__global__ void __launch_bounds__(GS_T) r_gemv_f32_split_kernel(const float* __restrict__ x, long ldx,
                                                                const uint16_t* __restrict__ w, float* __restrict__ y,
                                                                long ldy, int m, int n, int k) {
  r_pdl_trigger();
  r_pdl_wait();
  extern __shared__ float part[];  // [m][GS_OB][k / 16]
  const int o0 = blockIdx.x * GS_OB;
  const int steps = k / 16;
  for (int u = threadIdx.x; u < GS_OB * steps; u += GS_T) {
    const int o = u / steps, c = u - o * steps;
    if (o0 + o >= n) continue;
    const uint4* wp = (const uint4*)(w + (long)(o0 + o) * k + c * 16);
    const uint4 u0 = __ldg(wp), u1 = __ldg(wp + 1);
    const uint32_t wv[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    for (int i = 0; i < m; i++) {
      const float4* xp = (const float4*)(x + (long)i * ldx + c * 16);
      const float4 xa = __ldg(xp), xb = __ldg(xp + 1), xc = __ldg(xp + 2), xd = __ldg(xp + 3);
      const float xv[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                            xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};
      float p = 0.0f;
#pragma unroll
      for (int e = 0; e < 16; e++) {
        const uint32_t h = (e & 1) ? (wv[e >> 1] >> 16) : (wv[e >> 1] & 0xffffu);
        p = __fadd_rn(p, __fmul_rn(xv[e], __uint_as_float(h << 16)));
      }
      part[(i * GS_OB + o) * steps + c] = p;
    }
  }
  __syncthreads();
  if (threadIdx.x < m * GS_OB) {
    const int i = threadIdx.x / GS_OB, o = threadIdx.x % GS_OB;
    if (o0 + o < n) {
      const float* pp = part + (i * GS_OB + o) * steps;
      float acc = 0.0f;
      for (int c = 0; c < steps; c++) acc = __fadd_rn(acc, pp[c]);
      y[(long)i * ldy + o0 + o] = acc;
    }
  }
}

// 45006: more rows than the small kernel takes (the caller runs the tiled F32 GEMM). A narrow
// output (the router), or one row of the head, runs the split kernel (244 GB/s on the head at
// one row); more head rows four warps a block, a lane a weight row.
extern "C" int memra_hy4r_gemm_f32(const float* x, long ldx, const void* w, float* y, long ldy, int m, int n, int k,
                                   void* stream) {
  if (m == 0) return 0;
  if (m < 0 || n <= 0 || k <= 0 || (k % 16) != 0) return 45001;
  if (m > GF_MAXM) return 45006;
  if ((ldx % 4) != 0 || (((uintptr_t)x) % 16) != 0 || (((uintptr_t)w) % 16) != 0) return 45005;
  cudaStream_t st = (cudaStream_t)stream;
  if (k <= 4096 && (n <= 4096 || m == 1)) {
    const int smem = m * GS_OB * (k / 16) * 4;
    r_launch(r_gemv_f32_split_kernel, dim3((n + GS_OB - 1) / GS_OB), dim3(GS_T), smem, st, x, ldx,
             (const uint16_t*)w, y, ldy, m, n, k);
  } else
    r_gemv_f32_go<4, 4, 128>(x, ldx, w, y, ldy, m, n, k, st);
  return r_rc();
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
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_rms_kernel, rows, 256, 0, (cudaStream_t)stream, x, ldx, w, y, ldy, d, eps);
  return r_rc();
}

// interleaved RoPE in place on `heads` slices of `dim` at offset `off` of each row: pairs
// (2i, 2i+1) of the last rope dims; cos/sin from the call's table [rows of positions, dim/2]
__global__ void r_rope_kernel(float* x, long ldrow, long ldhead, int heads, int off, int half, const float2* cs,
                              const int* posrow) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_rope_kernel, rows, 128, 0, (cudaStream_t)stream, x, ldrow, ldhead, heads, off, half, (const float2*)cs, posrow);
  return r_rc();
}

// the call's cos/sin table: row r is position pos0 + r, inv_freq in f64
__global__ void r_rope_table_kernel(float2* out, int pos0, int rows, int half, const double* inv) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_rope_table_kernel, (n + 255) / 256, 256, 0, (cudaStream_t)stream, (float2*)out, pos0, rows, half, inv);
  return r_rc();
}

// HYV4FlashHyperConnection pre: per row, flat = s[r] (n streams x h), inv = rsqrt(mean(flat^2) +
// mix_eps), mix_j = (flat . fn_j) * inv; pre_j = sigmoid(mix_j * scale0 + base_j) + eps;
// post_j = mag * sigmoid(mix_{n+j} * scale1 + base_{n+j}) + eps; x = sum_j pre_j * flat_j.
// `fns` functions (2n for a layer, n for the exit head with post == null).
//
// Two kernels: the 1 + fns reductions of a row run as separate blocks (each the same strided
// per-thread sums and block tree a single block ran in turn), then one block a row mixes.
// The 1 + fns reductions of a row, each over the 256 strided per-thread partials a single block
// of 256 threads would hold, then that block's fixed tree. The partials run as 4 blocks of 64
// threads a reduction (`r_hc_dots_kernel`); the mix kernel's 256 threads take partial t each and
// run the tree, the same bits as one block per reduction.
#define HC_T 256
#define HC_SPLIT 4
__global__ void __launch_bounds__(HC_T / HC_SPLIT) r_hc_dots_kernel(const float* s, const float* fn, int width,
                                                                   int fns, float* parts) {
  r_pdl_wait();
  r_pdl_trigger();
  const int r = blockIdx.x, j = blockIdx.y;
  const int t = blockIdx.z * (HC_T / HC_SPLIT) + threadIdx.x;
  const float* flat = s + (long)r * width;
  float d = 0.0f;
  if (j == 0) {
    for (int i = t; i < width; i += HC_T) d += flat[i] * flat[i];
  } else {
    const float* f = fn + (long)(j - 1) * width;
    for (int i = t; i < width; i += HC_T) d += flat[i] * f[i];
  }
  parts[((long)r * (fns + 1) + j) * HC_T + t] = d;
}

__global__ void __launch_bounds__(HC_T) r_hc_mix_kernel(const float* s, const float* parts, const float* base,
                                                        const float* scale, int nstream, int h, int fns, float mix_eps,
                                                        float eps, float mag, float* post, float* x) {
  r_pdl_wait();
  r_pdl_trigger();
  __shared__ float red[17][HC_T / 32];
  __shared__ float dot[17];
  __shared__ float mix[16];
  __shared__ float pre[8];
  const int r = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int width = nstream * h;
  const float* flat = s + (long)r * width;
  // r_block_sum's tree for every reduction at once
  for (int j = 0; j <= fns; j++) {
    const float v = r_warp_sum(parts[((long)r * (fns + 1) + j) * HC_T + threadIdx.x]);
    if (lane == 0) red[j][warp] = v;
  }
  __syncthreads();
  if (warp == 0) {
    for (int j = 0; j <= fns; j++) {
      float t = lane < HC_T / 32 ? red[j][lane] : 0.0f;
      t = r_warp_sum(t);
      if (lane == 0) dot[j] = t;
    }
  }
  __syncthreads();
  const float inv = rsqrtf(dot[0] / width + mix_eps);
  if (threadIdx.x < fns) mix[threadIdx.x] = dot[1 + threadIdx.x] * inv;
  __syncthreads();
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

// `parts` scratch: rows * (fns + 1) * 256 floats
extern "C" int memra_hy4r_hc(const float* s, const float* fn, const float* base, const float* scale, int rows,
                             int nstream, int h, int fns, float mix_eps, float eps, float mag, float* post, float* x,
                             float* parts, void* stream) {
  if (rows == 0) return 0;
  if (nstream > 8 || fns > 16) return 45101;
  cudaStream_t st = (cudaStream_t)stream;
  r_launch(r_hc_dots_kernel, dim3(rows, fns + 1, HC_SPLIT), HC_T / HC_SPLIT, 0, st, s, fn, nstream * h, fns, parts);
  r_launch(r_hc_mix_kernel, rows, HC_T, 0, st, s, parts, base, scale, nstream, h, fns, mix_eps, eps, mag, post, x);
  return r_rc();
}

// s[r, j, :] = post[r, j] * a[r, :] + s[r, j, :]
__global__ void r_write_back_kernel(float* s, const float* post, const float* a, int nstream, int h, long total) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_write_back_kernel, (unsigned)((total + 255) / 256), 256, 0, (cudaStream_t)stream, s, post, a, nstream, h, total);
  return r_rc();
}

// out = silu(min(g, lim)) * clamp(u, -lim, lim) (lim < 0: no clamp); `gate_rows` scales: the
// attention output gate, out = o * sigmoid(g) when mode == 1
__global__ void r_act_kernel(const float* g, const float* u, float* out, long n, float lim, int mode) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_act_kernel, (unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream, g, u, out, n, lim, mode);
  return r_rc();
}

// y[r, :] = a[r, :] + b[r, :] (+ bias)
__global__ void r_add_kernel(float* y, const float* a, const float* b, const float* bias, long n, int d) {
  r_pdl_wait();
  r_pdl_trigger();
  const long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  float v = a[e];
  if (b) v = v + b[e];
  if (bias) v = v + bias[e % d];
  y[e] = v;
}

extern "C" int memra_hy4r_add(float* y, const float* a, const float* b, const float* bias, long n, int d, void* stream) {
  if (n == 0) return 0;
  r_launch(r_add_kernel, (unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream, y, a, b, bias, n, d);
  return r_rc();
}

// strided copy of `cols` f32 values a row, optionally into BF16 (`to_bf16`)
__global__ void r_copy_kernel(const float* src, long lds, void* dst, long ldd, int cols, int to_bf16, const int* dstrow) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_copy_kernel, rows, 128, 0, (cudaStream_t)stream, src, lds, dst, ldd, cols, to_bf16, dstrow);
  return r_rc();
}

// BF16 rows gathered into another BF16 buffer: dst[i] = src[rows[i]] (`words` u32 a row)
__global__ void r_gather_bf_kernel(const uint32_t* src, long lds, const int* rows, uint32_t* dst, long ldd, int words) {
  r_pdl_wait();
  r_pdl_trigger();
  const int i = blockIdx.x;
  const long sr = rows[i];
  for (int c = threadIdx.x; c < words; c += blockDim.x) dst[(long)i * ldd + c] = src[sr * lds + c];
}

extern "C" int memra_hy4r_gather_bf(const void* src, long lds, const int* rows, void* dst, long ldd, int n, int words,
                                    void* stream) {
  if (n == 0) return 0;
  r_launch(r_gather_bf_kernel, n, 128, 0, (cudaStream_t)stream, (const uint32_t*)src, lds, rows, (uint32_t*)dst, ldd, words);
  return r_rc();
}

// embedding rows on every stream: s[r, j, :] = embed[tok[r]]
__global__ void r_embed_kernel(const uint16_t* table, const int* tok, float* s, int nstream, int h) {
  r_pdl_wait();
  r_pdl_trigger();
  const int r = blockIdx.x;
  const uint16_t* row = table + (long)tok[r] * h;
  for (int d = threadIdx.x; d < h; d += blockDim.x) {
    const float v = r_bf(row[d]);
    for (int j = 0; j < nstream; j++) s[((long)r * nstream + j) * h + d] = v;
  }
}

extern "C" int memra_hy4r_embed(const void* table, const int* tok, float* s, int rows, int nstream, int h, void* stream) {
  if (rows == 0) return 0;
  r_launch(r_embed_kernel, rows, 256, 0, (cudaStream_t)stream, (const uint16_t*)table, tok, s, nstream, h);
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
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_topk_kernel, rows, 256, 0, (cudaStream_t)stream, vals, ldv, n, k, reverse_ties, out, ldo);
  return r_rc();
}

// ---- router ------------------------------------------------------------------------------------

// scores = sigmoid(logits), choice = scores + bias (both F32, as the program computes them)
__global__ void r_router_scores_kernel(const float* logits, const float* bias, float* scores, float* choice, long n,
                                       int e) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_router_scores_kernel, (unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream, logits, bias, scores, choice, n, e);
  return r_rc();
}

// weights of the applied picks: w = scores[pick] (normalized when the plan says so) * scaling
__global__ void r_router_w_kernel(const float* scores, int e, const int* picks, int k, int normalize, float scaling,
                                  float* w, int rows) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_router_w_kernel, (rows + 127) / 128, 128, 0, (cudaStream_t)stream, scores, e, picks, k, normalize, scaling, w,
                                                                          rows);
  return r_rc();
}

// ---- MoE grouping ------------------------------------------------------------------------------

// counts[e] over the picks; then one block: offsets, tiles (expert, row0, rows) of `bm` rows,
// every unused tile zeroed; cursor = offsets
__global__ void r_moe_count_kernel(const int* picks, int n, int* counts) {
  r_pdl_wait();
  r_pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[picks[i]], 1);
}

// one block of RM_THREADS: exclusive scans of the counts and of the tiles per expert, then each
// expert writes its own tiles (the same table the serial walk wrote)
#define RM_THREADS 512
__global__ void __launch_bounds__(RM_THREADS) r_moe_plan_kernel(const int* counts, int ne, int bm, int max_tiles,
                                                                int* cursor, int* tiles) {
  r_pdl_wait();
  r_pdl_trigger();
  __shared__ int sc[2][RM_THREADS];
  __shared__ int carry[2];
  const int tid = threadIdx.x;
  if (tid == 0) carry[0] = carry[1] = 0;
  __syncthreads();
  for (int e0 = 0; e0 < ne; e0 += RM_THREADS) {
    const int e = e0 + tid;
    const int c = e < ne ? counts[e] : 0;
    const int nt = (c + bm - 1) / bm;
    int vc = c, vt = nt;
    // inclusive Hillis-Steele scan of (c, nt)
    for (int o = 1; o < RM_THREADS; o <<= 1) {
      sc[0][tid] = vc;
      sc[1][tid] = vt;
      __syncthreads();
      if (tid >= o) {
        vc += sc[0][tid - o];
        vt += sc[1][tid - o];
      }
      __syncthreads();
    }
    const int off = carry[0] + vc - c, t0 = carry[1] + vt - nt;
    if (e < ne) {
      cursor[e] = off;
      for (int i = 0; i < nt; i++) {
        const int t = t0 + i;
        if (t < max_tiles) {
          tiles[t * 3] = e;
          tiles[t * 3 + 1] = off + i * bm;
          tiles[t * 3 + 2] = c - i * bm < bm ? c - i * bm : bm;
        }
      }
    }
    __syncthreads();
    if (tid == RM_THREADS - 1) {
      carry[0] += vc;
      carry[1] += vt;
    }
    __syncthreads();
  }
  for (int t = carry[1] + tid; t < max_tiles; t += RM_THREADS) {
    tiles[t * 3] = 0;
    tiles[t * 3 + 1] = 0;
    tiles[t * 3 + 2] = 0;
  }
}

// each (token, slot) takes a row of its expert's group (the row order inside a group does not
// change any row's result); xg[row] = x[token]
__global__ void r_moe_place_kernel(const int* picks, int n, int k, int* cursor, int* rowmap) {
  r_pdl_wait();
  r_pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) rowmap[i] = atomicAdd(&cursor[picks[i]], 1);
}

__global__ void r_moe_gather_kernel(const float* x, int d, const int* rowmap, int k, float* xg) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_moe_count_kernel, (n + 255) / 256, 256, 0, st, picks, n, counts);
  r_launch(r_moe_plan_kernel, 1, RM_THREADS, 0, st, counts, ne, bm, max_tiles, cursor, tiles);
  r_launch(r_moe_place_kernel, (n + 255) / 256, 256, 0, st, picks, n, k, cursor, rowmap);
  r_launch(r_moe_gather_kernel, n, 256, 0, st, x, d, rowmap, k, xg);
  return r_rc();
}

// out[t, :] = sum over slots in order (ascending expert) of w[t, slot] * y[rowmap[t, slot], :]
__global__ void r_moe_combine_kernel(const float* y, int d, const int* rowmap, const float* w, int k, float* out) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_moe_combine_kernel, rows, 256, 0, (cudaStream_t)stream, y, d, rowmap, w, k, out);
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

// HG heads a block (warps 0..HG-1, one a head); all 8 warps load the key tiles, so a decode call
// can spread its few (head, chunk) units over more SMs with fewer heads a block.
template <int HG>
__global__ void __launch_bounds__(256) r_attn_kernel(RAttn a) {
  r_pdl_wait();
  r_pdl_trigger();
  extern __shared__ __align__(16) unsigned char smem[];
  float* qs = (float*)smem;                 // [HG][576]
  float* ks = qs + HG * AT_DK;              // [32][577]
  float* ps = ks + AT_TILE * AT_KW;         // [HG][32]
  __shared__ long krow[2][AT_TILE];         // a tile's cache rows (element offsets), two tiles
  const int qi = blockIdx.x, hg = blockIdx.y, c = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, hl = tid >> 5;
  const bool head = hl < HG;
  const int h = hg * HG + (head ? hl : 0);
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
    for (int e = tid; e < HG * AT_DK; e += 256)
      qs[e] = a.q[((long)qi * a.H + hg * HG) * AT_DK + e];
    // a tile's rows first, then its values as independent 16-byte loads into registers; the next
    // tile's loads are in flight while this one is scored (the same values the element loop read)
    constexpr int PER = AT_TILE * (AT_DK / 4) / 256;  // float4 a thread a tile
    float4 pre[PER];
    auto rows_of = [&](int t0, int buf) {
      const int nt = min(AT_TILE, j1 - t0);
      if (tid < AT_TILE) krow[buf][tid] = tid < nt ? (long)r_key_row(a, qi, t0 + tid, lo) * AT_DK : -1;
    };
    auto fetch = [&](int buf) {
#pragma unroll
      for (int u = 0; u < PER; u++) {
        const int e = tid + u * 256;
        const int r = e / (AT_DK / 4), d4 = e % (AT_DK / 4);
        const long kr0 = krow[buf][r];
        pre[u] = kr0 >= 0 ? *(const float4*)(a.kv + kr0 + 4 * d4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    };
    rows_of(j0, 0);
    __syncthreads();
    fetch(0);
    int cur = 0;
    for (int t0 = j0; t0 < j1; t0 += AT_TILE) {
      const int nt = min(AT_TILE, j1 - t0);
      __syncthreads();  // the previous tile is scored: its slot is free
#pragma unroll
      for (int u = 0; u < PER; u++) {
        const int e = tid + u * 256;
        const int r = e / (AT_DK / 4), d4 = e % (AT_DK / 4);
        float* dst = ks + r * AT_KW + 4 * d4;
        dst[0] = pre[u].x;
        dst[1] = pre[u].y;
        dst[2] = pre[u].z;
        dst[3] = pre[u].w;
      }
      const bool more = t0 + AT_TILE < j1;
      if (more) rows_of(t0 + AT_TILE, cur ^ 1);
      __syncthreads();
      if (more) fetch(cur ^ 1);
      cur ^= 1;
      if (!head) continue;  // a loading warp: on to the next tile's barrier
      float sc = -INFINITY;
      if (lane < nt) {
        const float4* qh = (const float4*)(qs + hl * AT_DK);  // a broadcast 16-byte read a 4 inputs
        const float* kr = ks + lane * AT_KW;
        float dot = 0.0f;
#pragma unroll 4
        for (int d4 = 0; d4 < AT_DK / 4; d4++) {
          const float4 qv = qh[d4];
          dot += qv.x * kr[4 * d4];
          dot += qv.y * kr[4 * d4 + 1];
          dot += qv.z * kr[4 * d4 + 2];
          dot += qv.w * kr[4 * d4 + 3];
        }
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
  if (!head) return;
#pragma unroll
  for (int i = 0; i < 16; i++) a.po[pidx * AT_DV + lane + 32 * i] = o[i];
  if (lane == 0) {
    a.pml[pidx * 2] = m;
    a.pml[pidx * 2 + 1] = l;
  }
}

// out[q, h, :] = merge of the chunks in order, the sink in the max and the denominator
__global__ void r_attn_merge_kernel(const float* po, const float* pml, const float* sinks, int H, int C, float* out) {
  r_pdl_wait();
  r_pdl_trigger();
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
  RAttn a{q, (const float*)kv, list, ldl, lens, qpos, mode, window, ring, base, H, C, scale, po, pml};
  // the most heads a block that keeps the call's blocks within one wave of the SMs (one block an
  // SM: a key tile is 74 KB of shared memory)
  static int sms = 0;
  if (!sms) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const size_t m8 = sizeof(float) * (8 * AT_DK + AT_TILE * AT_KW + 8 * AT_TILE);
    cudaFuncSetAttribute(r_attn_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)m8);
    cudaFuncSetAttribute(r_attn_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)m8);
    cudaFuncSetAttribute(r_attn_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)m8);
    cudaFuncSetAttribute(r_attn_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)m8);
  }
  int hgs = 8;
  while (hgs > 1 && (long)rows * (H / (hgs / 2)) * C <= sms) hgs /= 2;
  const size_t smem = sizeof(float) * (hgs * AT_DK + AT_TILE * AT_KW + hgs * AT_TILE);
  dim3 grid(rows, H / hgs, C);
  cudaStream_t st = (cudaStream_t)stream;
  if (hgs == 8)
    r_launch(r_attn_kernel<8>, grid, 256, smem, st, a);
  else if (hgs == 4)
    r_launch(r_attn_kernel<4>, grid, 256, smem, st, a);
  else if (hgs == 2)
    r_launch(r_attn_kernel<2>, grid, 256, smem, st, a);
  else
    r_launch(r_attn_kernel<1>, grid, 256, smem, st, a);
  int rc = r_rc();
  if (rc) return rc;
  r_launch(r_attn_merge_kernel, dim3(rows, H), 128, 0, (cudaStream_t)stream, po, pml, sinks, H, C, out);
  return r_rc();
}

// ---- the pooled indexer ------------------------------------------------------------------------

// pooled key of each closing group g (rows gs..gs+n): from its 4 cached F32 token keys,
// pooled[d] = sum_s (kvp[d] . k_s) * softmax_s(kgp[d] . k_s + ape[s, d]), then
// key_norm (F32) and RoPE of the last `rd` dims at the group's first position.
__global__ void r_pool_kernel(const float* tk, int dim, int pool, const float* kvp, const float* kgp,
                              const float* ape, const float* key_norm, float eps, int rd, const float2* cs, int cs_pos0,
                              int g0, float* out) {
  r_pdl_wait();
  r_pdl_trigger();
  __shared__ float keys[8][128];
  __shared__ float pooled[128];
  __shared__ float red[32];
  const int g = g0 + blockIdx.x;
  const int d = threadIdx.x;  // blockDim == dim == 128
  for (int s = 0; s < pool; s++) keys[s][d] = tk[((long)g * pool + s) * dim + d];
  __syncthreads();
  float gate[8], val[8], mx = -INFINITY;
  // the 2 x pool dot chains run side by side, each in input order
  float a[8], b[8];
#pragma unroll
  for (int s = 0; s < 8; s++) a[s] = b[s] = 0.0f;
  for (int i = 0; i < dim; i++) {
    const float kg = kgp[(long)d * dim + i], kv = kvp[(long)d * dim + i];
#pragma unroll
    for (int s = 0; s < 8; s++) {
      if (s < pool) {
        a[s] += kg * keys[s][i];
        b[s] += kv * keys[s][i];
      }
    }
  }
  for (int s = 0; s < pool; s++) {
    gate[s] = a[s] + ape[s * dim + d];
    val[s] = b[s];
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
  r_launch(r_pool_kernel, n, dim, 0, (cudaStream_t)stream, (const float*)tk, dim, pool, kvp, kgp, ape, key_norm, eps, rd,
                                                     (const float2*)cs, cs_pos0, g0, out);
  return r_rc();
}

// group logits of each query: for g < own = pos / pool, sum_h hw[h] * hscale * relu(q_h . gk_g);
// -inf for the other groups of the call's `ng`. One thread a (query, group): a block takes 64
// groups of a query, their keys staged in shared memory; each output is the same chain as
// before (per head a dot in input order, then the heads in order).
#define IL_G 64
__global__ void __launch_bounds__(IL_G) r_index_logits_kernel(const float* qi, const float* hw, const float* gk,
                                                              const int* qpos, int heads, int dim, int pool, int ng,
                                                              float hscale, float* out) {
  r_pdl_wait();
  r_pdl_trigger();
  __shared__ float qs[16 * 128];
  __shared__ float ws[16];
  __shared__ float ks[IL_G * 129];
  const int r = blockIdx.x, g0 = blockIdx.y * IL_G;
  for (int e = threadIdx.x; e < heads * dim; e += blockDim.x) qs[e] = qi[(long)r * heads * dim + e];
  if (threadIdx.x < heads) ws[threadIdx.x] = hw[(long)r * heads + threadIdx.x] * hscale;
  const int own = qpos[r] / pool;
  const int gn = min(IL_G, min(own, ng) - g0);  // groups of this block that score
  for (int e = threadIdx.x; e < max(gn, 0) * dim; e += blockDim.x) {
    const int gi = e / dim, i = e % dim;
    ks[gi * 129 + i] = gk[(long)(g0 + gi) * dim + i];
  }
  __syncthreads();
  const int g = g0 + threadIdx.x;
  if (g >= ng) return;
  float v = -INFINITY;
  if (g < own) {
    v = 0.0f;
    const float* k = ks + threadIdx.x * 129;
    for (int h = 0; h < heads; h++) {
      float dot = 0.0f;
      for (int i = 0; i < dim; i++) dot += qs[h * dim + i] * k[i];
      v += ws[h] * fmaxf(dot, 0.0f);
    }
  }
  out[(long)r * ng + g] = v;
}

extern "C" int memra_hy4r_index_logits(const float* qi, const float* hw, const float* gk, const int* qpos, int rows,
                                       int heads, int dim, int pool, int ng, float hscale, float* out, void* stream) {
  if (rows == 0 || ng <= 0) return 0;
  if (heads > 16 || dim > 128) return 45302;
  r_launch(r_index_logits_kernel, dim3(rows, (ng + IL_G - 1) / IL_G), IL_G, 0, (cudaStream_t)stream, qi, hw, gk, qpos,
           heads, dim, pool, ng, hscale, out);
  return r_rc();
}

// the recalled groups (the picks plus the own group, sorted, the own group kept twice when it is
// also a pick) and the key rows they hold up to the query, ascending
// One block of RC_T threads a query, a thread an entry of the recall (kp + 1 <= 1024): own's
// place among the ascending picks (after its equals), then each distinct group's keys up to the
// query at its exclusive prefix offset; the same `sel`, list and length as a serial walk.
#define RC_T 1024
__global__ void __launch_bounds__(RC_T) r_recall_kernel(const int* picks, int kp, const int* qpos, int pool, int* sel,
                                                         int* list, int ldl, int* lens) {
  r_pdl_wait();
  r_pdl_trigger();
  __shared__ int s[RC_T];
  __shared__ int wsum[RC_T / 32];
  __shared__ int tot[2];
  const int r = blockIdx.x, i = threadIdx.x, lane = i & 31, warp = i >> 5;
  const int p = qpos[r];
  const int own = p / pool;
  const int ks = kp + 1;
  const int* pk = picks + (long)r * kp;
  // own's place: the number of picks <= own
  int le = (i < kp && pk[i] <= own) ? 1 : 0;
  le = __reduce_add_sync(0xffffffffu, le);
  if (lane == 0) wsum[warp] = le;
  __syncthreads();
  if (i == 0) {
    int t = 0;
    for (int w = 0; w < RC_T / 32; w++) t += wsum[w];
    tot[0] = t;
  }
  __syncthreads();
  const int pos = tot[0];
  int g = 0;
  if (i < ks) {
    g = i < pos ? pk[i] : (i == pos ? own : pk[i - 1]);
    s[i] = g;
    sel[(long)r * ks + i] = g;
  }
  __syncthreads();
  // keys of this entry: none for a repeat, else those of g up to the query
  int c = 0;
  if (i < ks && (i == 0 || s[i - 1] != g)) c = min(max(p - g * pool + 1, 0), pool);
  // exclusive scan of c over the block
  int incl = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  __syncthreads();
  if (lane == 31) wsum[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int wv = wsum[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, wv, o);
      if (lane >= o) wv += v;
    }
    wsum[lane] = wv;  // inclusive over warps
    if (lane == 31) tot[1] = wv;
  }
  __syncthreads();
  const int off = (warp ? wsum[warp - 1] : 0) + incl - c;
  for (int o = 0; o < c; o++) list[(long)r * ldl + off + o] = g * pool + o;
  if (i == 0) lens[r] = tot[1];
}

extern "C" int memra_hy4r_recall(const int* picks, int kp, const int* qpos, int rows, int pool, int* sel, int* list,
                                 int ldl, int* lens, void* stream) {
  if (rows == 0) return 0;
  if (kp + 1 > 1024 || (kp + 1) * pool > ldl) return 45303;
  r_launch(r_recall_kernel, rows, RC_T, 0, (cudaStream_t)stream, picks, kp, qpos, pool, sel, list, ldl, lens);
  return r_rc();
}

// forced recall (the oracle's sel, own included): only the key rows
__global__ void r_recall_forced_kernel(const int* sel, int ks, const int* qpos, int pool, int* list, int ldl, int* lens) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_recall_forced_kernel, rows, 32, 0, (cudaStream_t)stream, sel, ks, qpos, pool, list, ldl, lens);
  return r_rc();
}

// ---- Engram ------------------------------------------------------------------------------------

// emb[t, head*dim + d] = rms(e4m3(code) * row scale, head_norm[head]) per (token, head)
__global__ void r_engram_rows_kernel(const uint8_t* codes, const float* scales, const float* norms, int heads, int dim,
                                     float eps, float* emb) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_engram_rows_kernel, rows, dim3(32, heads), 0, (cudaStream_t)stream, (const uint8_t*)codes, scales, norms, heads,
                                                                          dim, eps, emb);
  return r_rc();
}

// per (token, stream j): key = rms(keyraw_j, norm_key_j), query = rms(s_j, norm_query_j),
// dot = key . query / sqrt(h), gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)));
// s_j += mask ? gate * main + scale_j * main : 0
__global__ void r_engram_mix_kernel(float* s, const float* keyraw, const float* main, const float* norm_key,
                                    const float* norm_query, const float* scales, const uint8_t* mask, int nstream, int h,
                                    float eps) {
  r_pdl_wait();
  r_pdl_trigger();
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
  r_launch(r_engram_mix_kernel, dim3(rows, nstream), 256, 0, (cudaStream_t)stream, s, keyraw, main, norm_key, norm_query,
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

// page-locked host memory: an upload from it does not wait for the stream
extern "C" int memra_hy4r_pinned(void** p, size_t bytes) {
  cudaError_t e = cudaHostAlloc(p, bytes, cudaHostAllocDefault);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

extern "C" int memra_hy4r_pinned_free(void* p) {
  cudaError_t e = cudaFreeHost(p);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}

extern "C" int memra_hy4r_sync(void* stream) {
  cudaError_t e = cudaStreamSynchronize((cudaStream_t)stream);
  return e == cudaSuccess ? 0 : 10000 + (int)e;
}
