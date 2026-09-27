// One pinned MiMo bundled audio-tokenizer encoder layer. Activations are f32
// carriers of BF16 values. GEMMs use the resident BF16 checkpoint matrices.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float read_bf16(const uint16_t* values, int index) {
    return __uint_as_float(static_cast<uint32_t>(values[index]) << 16);
}

__global__ void check_values(const float* values, int elements, bool require_bf16, int* fault) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float value = values[index];
    if (!isfinite(value) || (require_bf16 && (__float_as_uint(value) & 0xffffu) != 0)) {
        atomicOr(fault, 1);
    }
}

__global__ void layer_norm(const float* input, const uint16_t* weight,
                           const uint16_t* bias, float* output) {
    const int row = blockIdx.x;
    __shared__ float mean;
    __shared__ float scale;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int col = 0; col < 1024; ++col) sum += input[row * 1024 + col];
        mean = sum / 1024.0f;
        float variance = 0.0f;
        for (int col = 0; col < 1024; ++col) {
            const float delta = input[row * 1024 + col] - mean;
            variance += delta * delta;
        }
        // torch.nn.LayerNorm default epsilon, not MiMo text RMS epsilon.
        scale = rsqrtf(variance / 1024.0f + 1e-5f);
    }
    __syncthreads();
    for (int col = threadIdx.x; col < 1024; col += blockDim.x) {
        const float value = (input[row * 1024 + col] - mean) * scale;
        output[row * 1024 + col] =
            bf16(value * read_bf16(weight, col) + read_bf16(bias, col));
    }
}

__global__ void linear_direct(const float* input, const uint16_t* weight,
                              float* output, int rows, int in_features, int out_features) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * out_features) return;
    const int row = index / out_features;
    const int feature = index % out_features;
    float sum = 0.0f;
    for (int col = 0; col < in_features; ++col) {
        sum = fmaf(input[row * in_features + col],
                   read_bf16(weight, feature * in_features + col), sum);
    }
    output[index] = sum;
}

// Reorder `[tokens,1024]` to `[ceil(tokens/2),1024*2]` in Conv1D's
// `[input_channel,kernel_tap]` weight order. Missing odd-tail taps are zero.
__global__ void downsample_columns(const float* input, float* columns, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int extent = ((tokens + 1) / 2) * 1024 * 2;
    if (index >= extent) return;
    const int tap = index % 2;
    const int channel = (index / 2) % 1024;
    const int row = index / (1024 * 2);
    const int source = row * 2 + tap;
    columns[index] = source < tokens ? input[source * 1024 + channel] : 0.0f;
}

__global__ void epilogue(const float* input, const uint16_t* bias,
                         const float* residual, float* output,
                         int elements, int width, int operation) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = input[index];
    if (operation == 1) value += read_bf16(bias, index % width);
    if (operation == 2) value += residual[index];
    if (operation == 3) value = 0.5f * value *
        (1.0f + erff(value * 0.7071067811865475f));
    output[index] = bf16(value);
}

__global__ void rope(float* query, float* key, int tokens) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * 16 * 32) return;
    const int half = index % 32;
    const int head = (index / 32) % 16;
    const int row = index / (16 * 32);
    const int offset = row * 1024 + head * 64 + half;
    const float angle = static_cast<float>(row) /
        powf(10000.0f, static_cast<float>(2 * half) / 64.0f);
    const float c = bf16(cosf(angle));
    const float s = bf16(sinf(angle));
    const float qa = query[offset];
    const float qb = query[offset + 32];
    const float ka = key[offset];
    const float kb = key[offset + 32];
    query[offset] = bf16(bf16(qa * c) + bf16(-qb * s));
    query[offset + 32] = bf16(bf16(qb * c) + bf16(qa * s));
    key[offset] = bf16(bf16(ka * c) + bf16(-kb * s));
    key[offset + 32] = bf16(bf16(kb * c) + bf16(ka * s));
}

__global__ void causal_attention(const float* query, const float* key,
                                 const float* value, float* output,
                                 int tokens, int window) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int past = threadIdx.x;
    __shared__ float scores[256];
    const int first = window < 0 ? 0 : max(0, row - window);
    float score = -INFINITY;
    if (past >= first && past <= row && past < tokens) {
        score = 0.0f;
        const int q = row * 1024 + head * 64;
        const int k = past * 1024 + head * 64;
        for (int col = 0; col < 64; ++col)
            score = fmaf(query[q + col], key[k + col], score);
        score *= 0.125f;
    }
    scores[past] = score;
    __syncthreads();
    if (past == 0) {
        float maximum = -INFINITY;
        for (int token = first; token <= row; ++token)
            maximum = fmaxf(maximum, scores[token]);
        float normalizer = 0.0f;
        for (int token = first; token <= row; ++token) {
            scores[token] = expf(scores[token] - maximum);
            normalizer += scores[token];
        }
        for (int token = first; token <= row; ++token)
            scores[token] /= normalizer;
    }
    __syncthreads();
    if (past < 64) {
        float sum = 0.0f;
        for (int token = first; token <= row; ++token)
            sum = fmaf(scores[token], value[token * 1024 + head * 64 + past], sum);
        output[row * 1024 + head * 64 + past] = bf16(sum);
    }
}

int launch_status() {
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : 10000 + static_cast<int>(error);
}

}  // namespace

extern "C" int memra_mimo_codec_layer_check_values(
    const float* values, int elements, int require_bf16, int* fault, void* stream_v) {
    if (!values || !fault || !stream_v || elements < 1 ||
        elements > 256 * 4096 || (require_bf16 != 0 && require_bf16 != 1))
        return 40001;
    check_values<<<(elements + 255) / 256, 256, 0,
                   static_cast<cudaStream_t>(stream_v)>>>(
        values, elements, require_bf16 != 0, fault);
    return launch_status();
}

extern "C" int memra_mimo_codec_layer_norm(
    const float* input, const uint16_t* weight, const uint16_t* bias,
    float* output, int tokens, void* stream_v) {
    if (!input || !weight || !bias || !output || !stream_v ||
        tokens < 1 || tokens > 256) return 40001;
    layer_norm<<<tokens, 256, 0, static_cast<cudaStream_t>(stream_v)>>>(
        input, weight, bias, output);
    return launch_status();
}

extern "C" int memra_mimo_codec_layer_linear_direct(
    const float* input, const uint16_t* weight, float* output,
    int rows, int in_features, int out_features, void* stream_v) {
    if (!input || !weight || !output || !stream_v ||
        rows < 1 || rows > 256 || in_features < 1 || in_features > 4096 ||
        out_features < 1 || out_features > 4096) return 40001;
    const int elements = rows * out_features;
    linear_direct<<<(elements + 255) / 256, 256, 0,
                    static_cast<cudaStream_t>(stream_v)>>>(
        input, weight, output, rows, in_features, out_features);
    return launch_status();
}

extern "C" int memra_mimo_codec_downsample_columns(
    const float* input, float* columns, int tokens, void* stream_v) {
    if (!input || !columns || !stream_v || tokens < 1 || tokens > 256)
        return 40001;
    const int elements = ((tokens + 1) / 2) * 1024 * 2;
    downsample_columns<<<(elements + 255) / 256, 256, 0,
                          static_cast<cudaStream_t>(stream_v)>>>(
        input, columns, tokens);
    return launch_status();
}

extern "C" int memra_mimo_codec_layer_epilogue(
    const float* input, const uint16_t* bias, const float* residual,
    float* output, int elements, int width, int operation, void* stream_v) {
    if (!input || !output || !stream_v || elements < 1 ||
        elements > 256 * 4096 || width < 1 || width > 4096 ||
        elements % width != 0 || operation < 0 || operation > 3 ||
        (operation == 1 && !bias) || (operation == 2 && !residual)) return 40001;
    epilogue<<<(elements + 255) / 256, 256, 0,
               static_cast<cudaStream_t>(stream_v)>>>(
        input, bias, residual, output, elements, width, operation);
    return launch_status();
}

extern "C" int memra_mimo_codec_layer_rope(
    float* query, float* key, int tokens, void* stream_v) {
    if (!query || !key || !stream_v || tokens < 1 || tokens > 256) return 40001;
    const int elements = tokens * 16 * 32;
    rope<<<(elements + 255) / 256, 256, 0,
           static_cast<cudaStream_t>(stream_v)>>>(query, key, tokens);
    return launch_status();
}

extern "C" int memra_mimo_codec_layer_attention(
    const float* query, const float* key, const float* value,
    float* output, int tokens, int window, void* stream_v) {
    if (!query || !key || !value || !output || !stream_v ||
        tokens < 1 || tokens > 256 || (window != -1 && window != 128))
        return 40001;
    causal_attention<<<dim3(tokens, 16), 256, 0,
                       static_cast<cudaStream_t>(stream_v)>>>(
        query, key, value, output, tokens, window);
    return launch_status();
}
