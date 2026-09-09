// Standalone numeric experiment. No runtime dispatch or serving integration.
// Reuse Memra's own validated lane maps and baseline, never external kernels.
#include "../../../crates/memra-engine/cu/flash_attn.cu"

__device__ __forceinline__ float fa2_exp2(float x) {
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

template<int TILE>
__device__ __forceinline__ void fa2_stage(__nv_bfloat16* sk, __nv_bfloat16* sv,
        const __nv_bfloat16* k, const __nv_bfloat16* v, int start, int depth, int head) {
    for (int i = threadIdx.y * 32 + threadIdx.x; i < TILE * 32; i += 128) {
        int row = i / 32, col = i % 32;
        int dst = row * 256 + (col ^ (row & 7)) * 8;
        size_t src = (size_t)(start + row) * 1024 + head * 256 + col * 8;
        if (start + row < depth) {
            cp_async_16(sk + dst, k + src);
            cp_async_16(sv + dst, v + src);
        } else {
            *(uint4*)(sk + dst) = make_uint4(0,0,0,0);
            *(uint4*)(sv + dst) = make_uint4(0,0,0,0);
        }
    }
    cp_async_commit();
}

template<bool PACKED, bool SKIP_SCALE, bool LOG2 = false>
__device__ __forceinline__ void fa2_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, float* __restrict__ out, int rows, int depth) {
    constexpr int TILE = 16;
    int lane = threadIdx.x, warp = threadIdx.y;
    int base = blockIdx.x * 64, wr = base + warp * 16;
    int kh = PACKED ? blockIdx.y : blockIdx.y / 6;
    extern __shared__ __align__(16) unsigned char raw[];
    auto s = reinterpret_cast<__nv_bfloat16*>(raw);
    auto sq = s + warp * 16 * 256;
    for (int i = lane; i < 16 * 256; i += 32) {
        int row = i / 256, col = i % 256;
        int token = PACKED ? (wr + row) / 6 : wr + row;
        int head = PACKED ? kh * 6 + (wr + row) % 6 : blockIdx.y;
        float x = token < rows ? q[((size_t)token * 24 + head) * 256 + col] : 0;
        sq[row * 256 + ((col / 8) ^ (row & 7)) * 8 + col % 8] = __float2bfloat16_rn(x);
    }
    __syncwarp();
    ATile qf[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) ld_A_sw(qf[i], sq, 0, i * 2, 32);
    __syncthreads();
    CTile accum[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) accum[i].x[j] = 0;
    float ml = NEG_INF, mh = NEG_INF, ll = 0, lh = 0;
    int pos_l = depth - rows + (PACKED ? (wr + lane / 4) / 6 : wr + lane / 4);
    int pos_h = depth - rows + (PACKED ? (wr + lane / 4 + 8) / 6 : wr + lane / 4 + 8);
    int end = min(depth, depth - rows + (PACKED ? (base + 63) / 6 : base + 63) + 1);
    int tiles = (end + TILE - 1) / TILE;
    fa2_stage<TILE>(s, s + TILE * 512, k, v, 0, depth, kh);
    for (int tile = 0; tile < tiles; ++tile) {
        auto sk = s + (tile & 1) * TILE * 256;
        auto sv = sk + TILE * 512;
        if (tile + 1 < tiles) {
            fa2_stage<TILE>(s + ((tile + 1) & 1) * TILE * 256,
                           s + ((tile + 1) & 1) * TILE * 256 + TILE * 512,
                           k, v, (tile + 1) * TILE, depth, kh);
            cp_async_wait_1();
        } else cp_async_wait_0();
        __syncthreads();
        CTile scores[2] = {};
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ATile b; ld_A_sw(b, sk, 0, i * 2, 32);
            BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
            mma_bf16(scores[0], qf[i], b0);
            mma_bf16(scores[1], qf[i], b1);
        }
        float tl = NEG_INF, th = NEG_INF;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int key = tile * TILE + i * 8 + (lane % 4) * 2 + (j & 1);
                float x = scores[i].x[j] * (LOG2 ? LOG2E / 16 : 1.0f / 16);
                if (key >= depth || key > (j < 2 ? pos_l : pos_h)) x = NEG_INF;
                scores[i].x[j] = x;
                if (j < 2) tl = fmaxf(tl, x); else th = fmaxf(th, x);
            }
        }
        float nl = fmaxf(ml, row_max4(tl)), nh = fmaxf(mh, row_max4(th));
        float al = ml == NEG_INF ? 0 : fa2_exp2((ml - nl) * (LOG2 ? 1.0f : LOG2E));
        float ah = mh == NEG_INF ? 0 : fa2_exp2((mh - nh) * (LOG2 ? 1.0f : LOG2E));
        ml = nl; mh = nh;
        float pl = 0, ph = 0;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x = scores[i].x[j];
                float p = x == NEG_INF ? 0 : fa2_exp2((x - (j < 2 ? nl : nh)) * (LOG2 ? 1.0f : LOG2E));
                // FA2 class: denominator sums the same BF16 probabilities used by PV.
                p = __bfloat162float(__float2bfloat16_rn(p));
                scores[i].x[j] = p;
                if (j < 2) pl += p; else ph += p;
            }
        }
        ll = ll * al + row_sum4(pl); lh = lh * ah + row_sum4(ph);
        ATile p;
        p.x[0] = __floats2bfloat162_rn(scores[0].x[0], scores[0].x[1]);
        p.x[1] = __floats2bfloat162_rn(scores[0].x[2], scores[0].x[3]);
        p.x[2] = __floats2bfloat162_rn(scores[1].x[0], scores[1].x[1]);
        p.x[3] = __floats2bfloat162_rn(scores[1].x[2], scores[1].x[3]);
        if (!SKIP_SCALE || __any_sync(0xffffffff, al != 1 || ah != 1)) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                accum[i].x[0] *= al; accum[i].x[1] *= al;
                accum[i].x[2] *= ah; accum[i].x[3] *= ah;
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            ATile b; ld_A_trans_sw(b, sv, 0, i * 2, 32);
            BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
            mma_bf16(accum[i * 2], p, b0);
            mma_bf16(accum[i * 2 + 1], p, b1);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = wr + CTile::get_i(j);
            int token = PACKED ? row / 6 : row;
            int head = PACKED ? kh * 6 + row % 6 : blockIdx.y;
            if (token < rows) out[((size_t)token * 24 + head) * 256 + i * 8 + CTile::get_j(j)] =
                accum[i].x[j] / (j < 2 ? ll : lh);
        }
    }
}

#define FA2_STAMP(NAME, PACKED, SKIP) \
extern "C" __global__ __launch_bounds__(128, 2) void NAME( \
        const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out, \
        int rows, int depth) { fa2_body<PACKED, SKIP>(q, k, v, out, rows, depth); }
FA2_STAMP(fa2_gqa16, true, false)
FA2_STAMP(fa2_gqa16_skip, true, true)
FA2_STAMP(fa2_qw16_skip, false, true)

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa16_log2(
        const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out,
        int rows, int depth) { fa2_body<true, false, true>(q, k, v, out, rows, depth); }

// Wider softmax group, rotating three planes: overlap next V with softmax/PV.
template<bool PACKED, bool SKIP_SCALE, bool LOG2 = false, bool DEN_MMA = false>
__device__ __forceinline__ void fa2_wide_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, float* __restrict__ out, int rows, int depth) {
    constexpr int TILE = 32;
    int lane = threadIdx.x, warp = threadIdx.y;
    int base = blockIdx.x * 64, wr = base + warp * 16;
    int kh = PACKED ? blockIdx.y : blockIdx.y / 6;
    extern __shared__ __align__(16) unsigned char raw[];
    auto s = reinterpret_cast<__nv_bfloat16*>(raw);
    auto sq = s + warp * 16 * 256;
    for (int i = lane; i < 16 * 256; i += 32) {
        int row = i / 256, col = i % 256;
        int token = PACKED ? (wr + row) / 6 : wr + row;
        int head = PACKED ? kh * 6 + (wr + row) % 6 : blockIdx.y;
        float x = token < rows ? q[((size_t)token * 24 + head) * 256 + col] : 0;
        sq[row * 256 + ((col / 8) ^ (row & 7)) * 8 + col % 8] = __float2bfloat16_rn(x);
    }
    __syncwarp();
    ATile qf[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) ld_A_sw(qf[i], sq, 0, i * 2, 32);
    __syncthreads();
    CTile accum[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) accum[i].x[j] = 0;
    float ml = NEG_INF, mh = NEG_INF, ll = 0, lh = 0;
    int pos_l = depth - rows + (PACKED ? (wr + lane / 4) / 6 : wr + lane / 4);
    int pos_h = depth - rows + (PACKED ? (wr + lane / 4 + 8) / 6 : wr + lane / 4 + 8);
    int end = min(depth, depth - rows + (PACKED ? (base + 63) / 6 : base + 63) + 1);
    int tiles = (end + TILE - 1) / TILE;
    auto sk = s; auto sv = s + TILE * 256; auto spare = s + TILE * 512;
    fa2_stage<TILE>(sk, sv, k, v, 0, depth, kh);
    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait_0();
        __syncthreads();
        if (tile + 1 < tiles) stage_kv_plane_async<256>(spare, k, (tile + 1) * TILE,
            min(TILE, depth - (tile + 1) * TILE), 1024, kh * 256, warp * 32 + lane);
        CTile scores[4] = {};
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                ATile b; ld_A_sw(b, sk, g * 16, i * 2, 32);
                BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
                mma_bf16(scores[g * 2], qf[i], b0);
                mma_bf16(scores[g * 2 + 1], qf[i], b1);
            }
        }
        __syncthreads();
        if (tile + 1 < tiles) stage_kv_plane_async<256>(sk, v, (tile + 1) * TILE,
            min(TILE, depth - (tile + 1) * TILE), 1024, kh * 256, warp * 32 + lane);
        float tl = NEG_INF, th = NEG_INF;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int key = tile * TILE + i * 8 + (lane % 4) * 2 + (j & 1);
                float x = scores[i].x[j] * (LOG2 ? LOG2E / 16 : 1.0f / 16);
                if (key >= depth || key > (j < 2 ? pos_l : pos_h)) x = NEG_INF;
                scores[i].x[j] = x;
                if (j < 2) tl = fmaxf(tl, x); else th = fmaxf(th, x);
            }
        }
        float nl = fmaxf(ml, row_max4(tl)), nh = fmaxf(mh, row_max4(th));
        float al = ml == NEG_INF ? 0 : fa2_exp2((ml - nl) * (LOG2 ? 1.0f : LOG2E));
        float ah = mh == NEG_INF ? 0 : fa2_exp2((mh - nh) * (LOG2 ? 1.0f : LOG2E));
        ml = nl; mh = nh;
        float pl = 0, ph = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x = scores[i].x[j];
                float p = x == NEG_INF ? 0 : fa2_exp2((x - (j < 2 ? nl : nh)) * (LOG2 ? 1.0f : LOG2E));
                // FA2 class: denominator sums the same BF16 probabilities used by PV.
                if constexpr (!DEN_MMA) p = __bfloat162float(__float2bfloat16_rn(p));
                scores[i].x[j] = p;
                if constexpr (!DEN_MMA) { if (j < 2) pl += p; else ph += p; }
            }
        }
        if constexpr (!DEN_MMA) { ll = ll * al + row_sum4(pl); lh = lh * ah + row_sum4(ph); }
        ATile p[2];
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            p[g].x[0] = __floats2bfloat162_rn(scores[g*2].x[0], scores[g*2].x[1]);
            p[g].x[1] = __floats2bfloat162_rn(scores[g*2].x[2], scores[g*2].x[3]);
            p[g].x[2] = __floats2bfloat162_rn(scores[g*2+1].x[0], scores[g*2+1].x[1]);
            p[g].x[3] = __floats2bfloat162_rn(scores[g*2+1].x[2], scores[g*2+1].x[3]);
        }
        if constexpr (DEN_MMA) {
            CTile denom{{ll * al, ll * al, lh * ah, lh * ah}};
            BTile ones{{__floats2bfloat162_rn(1, 1), __floats2bfloat162_rn(1, 1)}};
            mma_bf16(denom, p[0], ones);
            mma_bf16(denom, p[1], ones);
            ll = denom.x[0]; lh = denom.x[2];
        }
        if (!SKIP_SCALE || __any_sync(0xffffffff, al != 1 || ah != 1)) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                accum[i].x[0] *= al; accum[i].x[1] *= al;
                accum[i].x[2] *= ah; accum[i].x[3] *= ah;
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                ATile b; ld_A_trans_sw(b, sv, g * 16, i * 2, 32);
                BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
                mma_bf16(accum[i * 2], p[g], b0);
                mma_bf16(accum[i * 2 + 1], p[g], b1);
            }
        }
        __syncthreads();
        auto old_v = sv; sv = sk; sk = spare; spare = old_v;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = wr + CTile::get_i(j);
            int token = PACKED ? row / 6 : row;
            int head = PACKED ? kh * 6 + row % 6 : blockIdx.y;
            if (token < rows) out[((size_t)token * 24 + head) * 256 + i * 8 + CTile::get_j(j)] =
                accum[i].x[j] / (j < 2 ? ll : lh);
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate(
        const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out,
        int rows, int depth) { fa2_wide_body<true, false, true>(q, k, v, out, rows, depth); }

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate_denmma(
        const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out,
        int rows, int depth) { fa2_wide_body<true, false, true, true>(q, k, v, out, rows, depth); }

// Arithmetic isolation: 0 baseline order, 1 direct PV, 2 rounded MMA denominator, 3 log2.
template<int MODE>
__device__ __forceinline__ void fa2_control_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, float* __restrict__ out, int rows, int depth) {
    constexpr bool PACKED=true, SKIP_SCALE=false, LOG2=(MODE>=3), DEN_MMA=(MODE>=2);
    constexpr int TILE = 32;
    int lane = threadIdx.x, warp = threadIdx.y;
    int base = blockIdx.x * 64, wr = base + warp * 16;
    int kh = PACKED ? blockIdx.y : blockIdx.y / 6;
    extern __shared__ __align__(16) unsigned char raw[];
    auto s = reinterpret_cast<__nv_bfloat16*>(raw);
    auto sq = s + warp * 16 * 256;
    for (int i = lane; i < 16 * 256; i += 32) {
        int row = i / 256, col = i % 256;
        int token = PACKED ? (wr + row) / 6 : wr + row;
        int head = PACKED ? kh * 6 + (wr + row) % 6 : blockIdx.y;
        float x = token < rows ? q[((size_t)token * 24 + head) * 256 + col] : 0;
        sq[row * 256 + ((col / 8) ^ (row & 7)) * 8 + col % 8] = __float2bfloat16_rn(x);
    }
    __syncwarp();
    ATile qf[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) ld_A_sw(qf[i], sq, 0, i * 2, 32);
    __syncthreads();
    CTile accum[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) accum[i].x[j] = 0;
    float ml = NEG_INF, mh = NEG_INF, ll = 0, lh = 0;
    int pos_l = depth - rows + (PACKED ? (wr + lane / 4) / 6 : wr + lane / 4);
    int pos_h = depth - rows + (PACKED ? (wr + lane / 4 + 8) / 6 : wr + lane / 4 + 8);
    int end = min(depth, depth - rows + (PACKED ? (base + 63) / 6 : base + 63) + 1);
    int tiles = (end + TILE - 1) / TILE;
    auto sk = s; auto sv = s + TILE * 256; auto spare = s + TILE * 512;
    fa2_stage<TILE>(sk, sv, k, v, 0, depth, kh);
    for (int tile = 0; tile < tiles; ++tile) {
        cp_async_wait_0();
        __syncthreads();
        if (tile + 1 < tiles) stage_kv_plane_async<256>(spare, k, (tile + 1) * TILE,
            min(TILE, depth - (tile + 1) * TILE), 1024, kh * 256, warp * 32 + lane);
        CTile scores[4] = {};
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                ATile b; ld_A_sw(b, sk, g * 16, i * 2, 32);
                BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
                mma_bf16(scores[g * 2], qf[i], b0);
                mma_bf16(scores[g * 2 + 1], qf[i], b1);
            }
        }
        __syncthreads();
        if (tile + 1 < tiles) stage_kv_plane_async<256>(sk, v, (tile + 1) * TILE,
            min(TILE, depth - (tile + 1) * TILE), 1024, kh * 256, warp * 32 + lane);
        float tl = NEG_INF, th = NEG_INF;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int key = tile * TILE + i * 8 + (lane % 4) * 2 + (j & 1);
                float x = scores[i].x[j] * (LOG2 ? LOG2E / 16 : 1.0f / 16);
                if (key >= depth || key > (j < 2 ? pos_l : pos_h)) x = NEG_INF;
                scores[i].x[j] = x;
                if (j < 2) tl = fmaxf(tl, x); else th = fmaxf(th, x);
            }
        }
        float nl = fmaxf(ml, row_max4(tl)), nh = fmaxf(mh, row_max4(th));
        float al = ml == NEG_INF ? 0 : exp2f((ml - nl) * (LOG2 ? 1.0f : LOG2E));
        float ah = mh == NEG_INF ? 0 : exp2f((mh - nh) * (LOG2 ? 1.0f : LOG2E));
        ml = nl; mh = nh;
        float pl = 0, ph = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float x = scores[i].x[j];
                float p = x == NEG_INF ? 0 : exp2f((x - (j < 2 ? nl : nh)) * (LOG2 ? 1.0f : LOG2E));
                // FA2 class: denominator sums the same BF16 probabilities used by PV.
                // MODE 0/1 retain the baseline FP32 denominator.
                scores[i].x[j] = p;
                if constexpr (!DEN_MMA) { if (j < 2) pl += p; else ph += p; }
            }
        }
        if constexpr (!DEN_MMA) { ll = ll * al + row_sum4(pl); lh = lh * ah + row_sum4(ph); }
        ATile p[2];
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            p[g].x[0] = __floats2bfloat162_rn(scores[g*2].x[0], scores[g*2].x[1]);
            p[g].x[1] = __floats2bfloat162_rn(scores[g*2].x[2], scores[g*2].x[3]);
            p[g].x[2] = __floats2bfloat162_rn(scores[g*2+1].x[0], scores[g*2+1].x[1]);
            p[g].x[3] = __floats2bfloat162_rn(scores[g*2+1].x[2], scores[g*2+1].x[3]);
        }
        if constexpr (DEN_MMA) {
            CTile denom{{ll * al, ll * al, lh * ah, lh * ah}};
            BTile ones{{__floats2bfloat162_rn(1, 1), __floats2bfloat162_rn(1, 1)}};
            mma_bf16(denom, p[0], ones);
            mma_bf16(denom, p[1], ones);
            ll = denom.x[0]; lh = denom.x[2];
        }
        if (!SKIP_SCALE || __any_sync(0xffffffff, al != 1 || ah != 1)) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                accum[i].x[0] *= al; accum[i].x[1] *= al;
                accum[i].x[2] *= ah; accum[i].x[3] *= ah;
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            CTile partial0{}, partial1{};
            #pragma unroll
            for (int g = 0; g < 2; ++g) {
                ATile b; ld_A_trans_sw(b, sv, g * 16, i * 2, 32);
                BTile b0{{b.x[0], b.x[2]}}, b1{{b.x[1], b.x[3]}};
                if constexpr (MODE == 0) {
                    mma_bf16(partial0, p[g], b0); mma_bf16(partial1, p[g], b1);
                } else {
                    mma_bf16(accum[i * 2], p[g], b0); mma_bf16(accum[i * 2 + 1], p[g], b1);
                }
            }
            if constexpr (MODE == 0) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    accum[i*2].x[j] += partial0.x[j]; accum[i*2+1].x[j] += partial1.x[j];
                }
            }
        }
        __syncthreads();
        auto old_v = sv; sv = sk; sk = spare; spare = old_v;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = wr + CTile::get_i(j);
            int token = PACKED ? row / 6 : row;
            int head = PACKED ? kh * 6 + row % 6 : blockIdx.y;
            if (token < rows) out[((size_t)token * 24 + head) * 256 + i * 8 + CTile::get_j(j)] =
                accum[i].x[j] * (1.0f / (j < 2 ? ll : lh));
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate_control0(
 const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out, int rows, int depth) { fa2_control_body<0>(q,k,v,out,rows,depth); }

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate_control1(
 const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out, int rows, int depth) { fa2_control_body<1>(q,k,v,out,rows,depth); }

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate_control2(
 const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out, int rows, int depth) { fa2_control_body<2>(q,k,v,out,rows,depth); }

extern "C" __global__ __launch_bounds__(128, 2) void fa2_gqa32_rotate_control3(
 const float* q, const __nv_bfloat16* k, const __nv_bfloat16* v, float* out, int rows, int depth) { fa2_control_body<3>(q,k,v,out,rows,depth); }
