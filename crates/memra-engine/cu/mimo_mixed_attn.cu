// Split decode attention for the pinned MiMo global-layer geometry.
// K: GGUF q8_0, 34 bytes / 32 values, [seq, 4, 192].
// V: GGUF NVFP4, 36 bytes / 64 values, [seq, 4, 128].
// Q: f32 [64, 192]. Output: f32 [64, 128].
// This is a source-inspection component, not a serving admission path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kHeads = 64;
constexpr int kKvHeads = 4;
constexpr int kQk = 192;
constexpr int kValue = 128;
constexpr int kQ8RowBytes = (kQk / 32) * 34;
constexpr int kNvfp4RowBytes = (kValue / 64) * 36;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTile = 256;
constexpr int kGroupedTile = 64;
constexpr int kDeepSplit = 512;
constexpr int kDeepTile = 32;
constexpr int kPartial = kValue + 2;
constexpr int kReduce = 128;
constexpr int kMaxSeq = 1048576;

__device__ __forceinline__ float ue4m3_scale(uint8_t code) {
    if (code == 0 || code == 0x7f) return 0.0f;
    const int exp = (code >> 3) & 15;
    const int man = code & 7;
    return exp == 0 ? static_cast<float>(man) * 0x1p-10f
                    : (1.0f + static_cast<float>(man) * 0.125f) *
                          ldexpf(1.0f, exp - 8);
}

__device__ __forceinline__ float nvfp4_value(const uint8_t* row, int dim) {
    const int block = dim / 64;
    const int sub = (dim & 63) / 16;
    const int local = dim & 15;
    const uint8_t* bytes = row + block * 36;
    const uint8_t packed = bytes[4 + sub * 8 + (local & 7)];
    const uint8_t code = local < 8 ? packed & 15 : packed >> 4;
    const int mag_index = code & 7;
    const float mag =
        mag_index <= 4 ? static_cast<float>(mag_index)
                       : mag_index == 5 ? 6.0f
                       : mag_index == 6 ? 8.0f
                                        : 12.0f;
    return (code & 8 ? -mag : mag) * ue4m3_scale(bytes[sub]);
}

__device__ __forceinline__ float q8_scale(const uint8_t* block) {
    return __half2float(*reinterpret_cast<const __half*>(block));
}

__global__ void mixed_tile(const float* __restrict__ q,
                           const uint8_t* __restrict__ k,
                           const uint8_t* __restrict__ v,
                           float* __restrict__ partial,
                           int seq, int tiles) {
    const int head = blockIdx.x;
    const int tile = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int kv_head = head / (kHeads / kKvHeads);
    const int first = tile * kTile;
    const int last = min(first + kTile, seq);
    const float scale = 1.0f / sqrtf(static_cast<float>(kQk));
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int token = first + warp; token < last; token += kWarps) {
        const uint8_t* key_row =
            k + (static_cast<size_t>(token) * kKvHeads + kv_head) * kQ8RowBytes;
        float dot = 0.0f;
#pragma unroll
        for (int block = 0; block < kQk / 32; ++block) {
            const uint8_t* packed = key_row + block * 34;
            const float key_value =
                q8_scale(packed) * static_cast<float>(
                    static_cast<int8_t>(packed[2 + lane]));
            dot = fmaf(q[head * kQk + block * 32 + lane], key_value, dot);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;
        const float next_m = fmaxf(m, score);
        const float alpha = expf(m - next_m);
        const float beta = expf(score - next_m);
        l = l * alpha + beta;
        m = next_m;
        const uint8_t* value_row =
            v + (static_cast<size_t>(token) * kKvHeads + kv_head) * kNvfp4RowBytes;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = nvfp4_value(value_row, j * 32 + lane);
            acc[j] = acc[j] * alpha + beta * value;
        }
    }

    __shared__ float warp_m[kWarps];
    __shared__ float warp_l[kWarps];
    __shared__ float warp_acc[kWarps][kValue];
    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        warp_acc[warp][j * 32 + lane] = acc[j];
    }
    __syncthreads();

    if (warp == 0) {
        float max_m = -CUDART_INF_F;
        for (int i = 0; i < kWarps; ++i) max_m = fmaxf(max_m, warp_m[i]);
        float denom = 0.0f;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = 0; i < kWarps; ++i) {
            const float weight = expf(warp_m[i] - max_m);
            denom += weight * warp_l[i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum[j] += weight * warp_acc[i][j * 32 + lane];
            }
        }
        float* out =
            partial + (static_cast<size_t>(head) * tiles + tile) * kPartial;
        if (lane == 0) {
            out[0] = max_m;
            out[1] = denom;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[2 + j * 32 + lane] = sum[j];
        }
    }
}

// Keep one KV tile in shared memory while all 16 query heads in its GQA group
// consume it. Two warps per query head process alternating tokens.
__global__ void mixed_grouped_tile(const float* __restrict__ q,
                                   const uint8_t* __restrict__ k,
                                   const uint8_t* __restrict__ v,
                                   float* __restrict__ partial,
                                   int seq, int tiles) {
    const int kv_head = blockIdx.x;
    const int tile = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int relative_head = warp / 2;
    const int token_partition = warp & 1;
    const int head = kv_head * (kHeads / kKvHeads) + relative_head;
    const int first = tile * kGroupedTile;
    const int count = min(kGroupedTile, seq - first);
    const float scale = 1.0f / sqrtf(static_cast<float>(kQk));

    __shared__ uint8_t key_tile[kGroupedTile * kQ8RowBytes];
    __shared__ uint8_t value_tile[kGroupedTile * kNvfp4RowBytes];
    __shared__ float warp_m[32];
    __shared__ float warp_l[32];
    __shared__ float warp_acc[32][kValue];
    const uint8_t* key_base =
        k + (static_cast<size_t>(first) * kKvHeads + kv_head) * kQ8RowBytes;
    const uint8_t* value_base =
        v + (static_cast<size_t>(first) * kKvHeads + kv_head) * kNvfp4RowBytes;
    // Source rows are token-major across four KV heads. Stage only this
    // group's row for each token rather than treating its rows as contiguous.
    for (int byte = threadIdx.x; byte < count * kQ8RowBytes; byte += blockDim.x) {
        const int token = byte / kQ8RowBytes;
        const int within = byte % kQ8RowBytes;
        key_tile[byte] =
            key_base[static_cast<size_t>(token) * kKvHeads * kQ8RowBytes + within];
    }
    for (int byte = threadIdx.x; byte < count * kNvfp4RowBytes; byte += blockDim.x) {
        const int token = byte / kNvfp4RowBytes;
        const int within = byte % kNvfp4RowBytes;
        value_tile[byte] =
            value_base[static_cast<size_t>(token) * kKvHeads * kNvfp4RowBytes + within];
    }
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int token = token_partition; token < count; token += 2) {
        const uint8_t* key_row = key_tile + token * kQ8RowBytes;
        float dot = 0.0f;
#pragma unroll
        for (int block = 0; block < kQk / 32; ++block) {
            const uint8_t* packed = key_row + block * 34;
            const float key_value =
                q8_scale(packed) * static_cast<float>(
                    static_cast<int8_t>(packed[2 + lane]));
            dot = fmaf(q[head * kQk + block * 32 + lane], key_value, dot);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        const float score = __shfl_sync(0xffffffff, dot, 0) * scale;
        const float next_m = fmaxf(m, score);
        const float alpha = expf(m - next_m);
        const float beta = expf(score - next_m);
        l = l * alpha + beta;
        m = next_m;
        const uint8_t* value_row = value_tile + token * kNvfp4RowBytes;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = nvfp4_value(value_row, j * 32 + lane);
            acc[j] = acc[j] * alpha + beta * value;
        }
    }

    if (lane == 0) {
        warp_m[warp] = m;
        warp_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        warp_acc[warp][j * 32 + lane] = acc[j];
    }
    __syncthreads();
    if (token_partition == 0) {
        const int first_warp = relative_head * 2;
        const float max_m = fmaxf(warp_m[first_warp], warp_m[first_warp + 1]);
        const float w0 = expf(warp_m[first_warp] - max_m);
        const float w1 = expf(warp_m[first_warp + 1] - max_m);
        float* out =
            partial + (static_cast<size_t>(head) * tiles + tile) * kPartial;
        if (lane == 0) {
            out[0] = max_m;
            out[1] = w0 * warp_l[first_warp] + w1 * warp_l[first_warp + 1];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dim = j * 32 + lane;
            out[2 + dim] = w0 * warp_acc[first_warp][dim] +
                           w1 * warp_acc[first_warp + 1][dim];
        }
    }
}

// Deep-context variant: each warp scores 32 different keys in parallel.
// Eight query heads share the staged K/V tile; two CTAs cover the 16 heads
// that map to one MiMo global KV head. Keep f32 Q arithmetic in this rung so
// the tile schedule can be priced separately from a future dp4a Q rewrite.
__global__ void mixed_deep_tile(const float* __restrict__ q,
                                const uint8_t* __restrict__ k,
                                const uint8_t* __restrict__ v,
                                float* __restrict__ partial,
                                int seq, int splits) {
    const int kv_head = blockIdx.x;
    const int head_group = blockIdx.y;
    const int split = blockIdx.z;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int head = kv_head * (kHeads / kKvHeads) + head_group * 8 + warp;
    const int first = split * kDeepSplit;
    const int last = min(first + kDeepSplit, seq);
    const float scale = 1.0f / sqrtf(static_cast<float>(kQk));

    alignas(16) __shared__ uint8_t key_tile[kDeepTile * kQ8RowBytes];
    __shared__ float value_tile[kDeepTile * kValue];
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t0 = first; t0 < last; t0 += kDeepTile) {
        const int count = min(kDeepTile, last - t0);
        const int thread = warp * 32 + lane;
        for (int byte = thread; byte < count * kQ8RowBytes; byte += 256) {
            const int token = byte / kQ8RowBytes;
            const int within = byte % kQ8RowBytes;
            key_tile[byte] =
                k[(static_cast<size_t>(t0 + token) * kKvHeads + kv_head) *
                      kQ8RowBytes +
                  within];
        }
        for (int index = thread; index < count * kValue; index += 256) {
            const int token = index / kValue;
            const int dim = index % kValue;
            const uint8_t* row =
                v + (static_cast<size_t>(t0 + token) * kKvHeads + kv_head) *
                        kNvfp4RowBytes;
            value_tile[index] = nvfp4_value(row, dim);
        }
        __syncthreads();

        float score = -CUDART_INF_F;
        if (lane < count) {
            const uint8_t* row = key_tile + lane * kQ8RowBytes;
            float dot = 0.0f;
#pragma unroll
            for (int block = 0; block < kQk / 32; ++block) {
                const uint8_t* packed = row + block * 34;
                const float d = q8_scale(packed);
#pragma unroll
                for (int dim = 0; dim < 32; ++dim) {
                    dot = fmaf(
                        q[head * kQk + block * 32 + dim],
                        d * static_cast<float>(
                                static_cast<int8_t>(packed[2 + dim])),
                        dot);
                }
            }
            score = dot * scale;
        }
        float tile_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max,
                             __shfl_xor_sync(0xffffffff, tile_max, offset));
        }
        const float next_m = fmaxf(m, tile_max);
        const float alpha = m == -CUDART_INF_F ? 0.0f : expf(m - next_m);
        const float p = lane < count ? expf(score - next_m) : 0.0f;
        float weight_sum = p;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            weight_sum += __shfl_xor_sync(0xffffffff, weight_sum, offset);
        }
        l = l * alpha + weight_sum;
        m = next_m;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] *= alpha;
        for (int token = 0; token < count; ++token) {
            const float weight = __shfl_sync(0xffffffff, p, token);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j] += weight * value_tile[token * kValue + j * 32 + lane];
            }
        }
        __syncthreads();
    }
    float* out =
        partial + (static_cast<size_t>(head) * splits + split) * kPartial;
    if (lane == 0) {
        out[0] = m;
        out[1] = l;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        out[2 + j * 32 + lane] = acc[j];
    }
}

__global__ void reduce_partials(const float* __restrict__ input,
                                float* __restrict__ output,
                                int input_tiles, int output_tiles) {
    const int head = blockIdx.x;
    const int group = blockIdx.y;
    const int dim = threadIdx.x;
    if (dim >= kValue) return;
    const int first = group * kReduce;
    const int last = min(first + kReduce, input_tiles);
    float max_m = -CUDART_INF_F;
    for (int i = first; i < last; ++i) {
        const float* p = input + (static_cast<size_t>(head) * input_tiles + i) * kPartial;
        max_m = fmaxf(max_m, p[0]);
    }
    float denom = 0.0f;
    float sum = 0.0f;
    for (int i = first; i < last; ++i) {
        const float* p = input + (static_cast<size_t>(head) * input_tiles + i) * kPartial;
        const float weight = expf(p[0] - max_m);
        denom += weight * p[1];
        sum += weight * p[2 + dim];
    }
    float* out = output + (static_cast<size_t>(head) * output_tiles + group) * kPartial;
    if (dim == 0) {
        out[0] = max_m;
        out[1] = denom;
    }
    out[2 + dim] = sum;
}

__global__ void normalize(const float* __restrict__ partial,
                          float* __restrict__ output) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    if (dim >= kValue) return;
    const float* row = partial + static_cast<size_t>(head) * kPartial;
    output[head * kValue + dim] = row[2 + dim] / row[1];
}

}  // namespace

// baseline scratch1: 64*ceil(seq/256)*130 f32
// grouped scratch1: 64*ceil(seq/64)*130 f32
// scratch2: 64*ceil(tiles/128)*130 f32
// scratch3: 64*130 f32
// 0 accepts an asynchronous launch. 41001..41004 refuse invalid contracts;
// 10000 + cudaError_t identifies an asynchronous-predecessor or launch error.
static int dispatch(
    const float* q, const uint8_t* k, const uint8_t* v,
    float* output, float* scratch1, float* scratch2, float* scratch3,
    int seq, int heads, int kv_heads, int qk_dim, int v_dim,
    size_t scratch1_floats, size_t scratch2_floats,
    size_t scratch3_floats, void* stream_v, int program) {
    if (heads != kHeads || kv_heads != kKvHeads ||
        qk_dim != kQk || v_dim != kValue) return 41001;
    if (seq <= 0 || seq > kMaxSeq) return 41002;
    if (!q || !k || !v || !output || !scratch1 || !scratch2 ||
        !scratch3 || !stream_v) return 41003;
    const int tile_size =
        program == 2 ? kDeepSplit : program == 1 ? kGroupedTile : kTile;
    const int tiles = (seq + tile_size - 1) / tile_size;
    const int groups = (tiles + kReduce - 1) / kReduce;
    if (scratch1_floats < static_cast<size_t>(kHeads) * tiles * kPartial ||
        scratch2_floats < static_cast<size_t>(kHeads) * groups * kPartial ||
        scratch3_floats < static_cast<size_t>(kHeads) * kPartial ||
        groups > kReduce) return 41004;
    const cudaError_t prior = cudaPeekAtLastError();
    if (prior != cudaSuccess) return 10000 + static_cast<int>(prior);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
    const char* timing_env = seq == kMaxSeq ? std::getenv("MEMRA_MIMO_SPLIT_STAGE_TIMING") : nullptr;
    const bool timing = timing_env != nullptr && timing_env[0] == '1' && timing_env[1] == '\0';
    cudaEvent_t marks[5] = {};
    auto fail = [&](cudaError_t error) {
        for (cudaEvent_t mark : marks) {
            if (mark != nullptr) cudaEventDestroy(mark);
        }
        return 10000 + static_cast<int>(error);
    };
    if (timing) {
        for (cudaEvent_t& mark : marks) {
            const cudaError_t created = cudaEventCreate(&mark);
            if (created != cudaSuccess) return fail(created);
        }
        const cudaError_t recorded = cudaEventRecord(marks[0], stream);
        if (recorded != cudaSuccess) return fail(recorded);
    }
    if (program == 2) {
        mixed_deep_tile<<<dim3(kKvHeads, 2, tiles), dim3(32, 8), 0, stream>>>(
            q, k, v, scratch1, seq, tiles);
    } else if (program == 1) {
        mixed_grouped_tile<<<dim3(kKvHeads, tiles), 1024, 0, stream>>>(
            q, k, v, scratch1, seq, tiles);
    } else {
        mixed_tile<<<dim3(kHeads, tiles), kThreads, 0, stream>>>(
            q, k, v, scratch1, seq, tiles);
    }
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) return fail(launch);
    if (timing && (launch = cudaEventRecord(marks[1], stream)) != cudaSuccess) {
        return fail(launch);
    }
    reduce_partials<<<dim3(kHeads, groups), kValue, 0, stream>>>(
        scratch1, scratch2, tiles, groups);
    launch = cudaGetLastError();
    if (launch != cudaSuccess) return fail(launch);
    if (timing && (launch = cudaEventRecord(marks[2], stream)) != cudaSuccess) {
        return fail(launch);
    }
    reduce_partials<<<dim3(kHeads, 1), kValue, 0, stream>>>(
        scratch2, scratch3, groups, 1);
    launch = cudaGetLastError();
    if (launch != cudaSuccess) return fail(launch);
    if (timing && (launch = cudaEventRecord(marks[3], stream)) != cudaSuccess) {
        return fail(launch);
    }
    normalize<<<kHeads, kValue, 0, stream>>>(scratch3, output);
    launch = cudaGetLastError();
    if (launch != cudaSuccess) return fail(launch);
    if (timing) {
        launch = cudaEventRecord(marks[4], stream);
        if (launch != cudaSuccess) return fail(launch);
        launch = cudaEventSynchronize(marks[4]);
        if (launch != cudaSuccess) return fail(launch);
        float stage_ms[4] = {};
        for (int i = 0; i < 4; ++i) {
            launch = cudaEventElapsedTime(&stage_ms[i], marks[i], marks[i + 1]);
            if (launch != cudaSuccess) return fail(launch);
        }
        std::fprintf(stderr, "mimo_split_stage_ms\t%s\t%.6f\t%.6f\t%.6f\t%.6f\n",
                     program == 2 ? "deep" : program == 1 ? "grouped" : "baseline",
                     stage_ms[0], stage_ms[1],
                     stage_ms[2], stage_ms[3]);
        for (cudaEvent_t mark : marks) cudaEventDestroy(mark);
    }
    return 0;
}

extern "C" int memra_mimo_global_q8_nvfp4_decode(
    const float* q, const uint8_t* k, const uint8_t* v,
    float* output, float* scratch1, float* scratch2, float* scratch3,
    int seq, int heads, int kv_heads, int qk_dim, int v_dim,
    size_t scratch1_floats, size_t scratch2_floats,
    size_t scratch3_floats, void* stream_v) {
    return dispatch(q, k, v, output, scratch1, scratch2, scratch3,
                    seq, heads, kv_heads, qk_dim, v_dim,
                    scratch1_floats, scratch2_floats, scratch3_floats,
                    stream_v, 0);
}

extern "C" int memra_mimo_global_q8_nvfp4_decode_grouped(
    const float* q, const uint8_t* k, const uint8_t* v,
    float* output, float* scratch1, float* scratch2, float* scratch3,
    int seq, int heads, int kv_heads, int qk_dim, int v_dim,
    size_t scratch1_floats, size_t scratch2_floats,
    size_t scratch3_floats, void* stream_v) {
    return dispatch(q, k, v, output, scratch1, scratch2, scratch3,
                    seq, heads, kv_heads, qk_dim, v_dim,
                    scratch1_floats, scratch2_floats, scratch3_floats,
                    stream_v, 1);
}

extern "C" int memra_mimo_global_q8_nvfp4_decode_deep(
    const float* q, const uint8_t* k, const uint8_t* v,
    float* output, float* scratch1, float* scratch2, float* scratch3,
    int seq, int heads, int kv_heads, int qk_dim, int v_dim,
    size_t scratch1_floats, size_t scratch2_floats,
    size_t scratch3_floats, void* stream_v) {
    return dispatch(q, k, v, output, scratch1, scratch2, scratch3,
                    seq, heads, kv_heads, qk_dim, v_dim,
                    scratch1_floats, scratch2_floats, scratch3_floats,
                    stream_v, 2);
}
