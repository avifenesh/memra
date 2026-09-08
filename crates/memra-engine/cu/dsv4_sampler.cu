// Memra-owned DSV4 plain sampler. Exact logit/ID order and position-keyed draw.
// Numeric class: device-f64-exp-tree-cdf-v1 (see docs/FLAGS.md).
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

static constexpr int B = 256;
__global__ void dsv4_sample_prepare(const float* logits, float* values,
    uint64_t* keys, const int* counts, int n, float repeat, float freq,
    float present, unsigned* result) {
    int i = blockIdx.x * B + threadIdx.x;
    if (i >= n) return;
    float v = logits[i];
    int count = counts[i];
    if (count) {
        if (repeat != 1.0f) v = v > 0.0f ? __fdiv_rn(v, repeat) : __fmul_rn(v, repeat);
        v = __fsub_rn(v, __fmul_rn(freq, (float)count));
        v = __fsub_rn(v, present);
    }
    if (!isfinite(v)) atomicExch(result + 1, 1u);
    values[i] = v;
    unsigned bits = v == 0.0f ? 0u : __float_as_uint(v);
    unsigned asc = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    keys[i] = ((uint64_t)~asc << 32) | (unsigned)i;
}
// Unique keys include token ID. Each item binary-searches the adjacent sorted run;
// its rank plus its within-run offset is a unique output location. Ragged safe.
__global__ void dsv4_sample_merge(const uint64_t* src, uint64_t* dst, int n, int width) {
    int i = blockIdx.x * B + threadIdx.x;
    if (i >= n) return;
    int base = (i / (2 * width)) * (2 * width);
    bool right = i >= base + width;
    int own = right ? base + width : base;
    int other = right ? base : min(base + width, n);
    int end = right ? min(base + width, n) : min(base + 2 * width, n);
    int lo = other, hi = end;
    uint64_t key = src[i];
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (src[mid] < key) lo = mid + 1; else hi = mid;
    }
    dst[base + (i - own) + (lo - other)] = key;
}
__global__ void dsv4_sample_exp_scan(const float* values, const uint64_t* keys,
    double* prefix, double* blocks, int k, double temperature) {
    __shared__ double x[B];
    int lane = threadIdx.x, i = blockIdx.x * B + lane;
    double m = (double)values[(unsigned)keys[0]];
    x[lane] = i < k ? exp(((double)values[(unsigned)keys[i]] - m) / temperature) : 0.0;
    __syncthreads();
    for (int shift = 1; shift < B; shift *= 2) {
        double add = lane >= shift ? x[lane - shift] : 0.0;
        __syncthreads();
        x[lane] = __dadd_rn(x[lane], add);
        __syncthreads();
    }
    if (i < k) prefix[i] = x[lane];
    if (lane == B - 1) blocks[blockIdx.x] = x[lane];
}
__global__ void dsv4_sample_offsets(double* blocks, int nb) {
    if (threadIdx.x || blockIdx.x) return;
    double sum = 0.0;
    for (int i = 0; i < nb; ++i) {
        double v = blocks[i];
        blocks[i] = sum;
        sum = __dadd_rn(sum, v);
    }
    blocks[nb] = sum;
}
__device__ double dsv4_sample_cdf(const double* prefix, const double* blocks, int i) {
    return __dadd_rn(prefix[i], blocks[i / B]);
}
__global__ void dsv4_sample_draw(const uint64_t* keys, const double* prefix,
    const double* blocks, int k, double top_p, double uniform, unsigned* result, const double* uniform_dev = nullptr) {
    if (uniform_dev) uniform = *uniform_dev;
    if (threadIdx.x || blockIdx.x) return;
    if (result[1]) { result[0] = 0xffffffffu; return; }
    double total = blocks[(k + B - 1) / B];
    if (!(isfinite(total) && total > 0.0)) { result[0] = 0xffffffffu; return; }
    double threshold = __dmul_rn(top_p, total);
    int lo = 0, hi = k - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (dsv4_sample_cdf(prefix, blocks, mid) < threshold) lo = mid + 1; else hi = mid;
    }
    int last = lo;
    double draw = __dmul_rn(uniform, dsv4_sample_cdf(prefix, blocks, last));
    lo = 0; hi = last;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (draw < dsv4_sample_cdf(prefix, blocks, mid)) hi = mid; else lo = mid + 1;
    }
    result[0] = (unsigned)keys[lo];
}
static int dsv4_sample_device_enqueue(const float* logits, float* values,
    uint64_t* keys0, uint64_t* keys1, double* prefix, double* blocks,
    const int* counts, unsigned* result, int n, int k, double temperature,
    double top_p, double uniform, float repeat, float freq, float present, void* raw_stream, const double* uniform_dev) {
    if (n <= 0 || k <= 0 || k > n || !(temperature > 0.0) || !(top_p > 0.0 && top_p <= 1.0)) return 40001;
    cudaStream_t stream = (cudaStream_t)raw_stream;
    cudaError_t err = cudaMemsetAsync(result, 0, 2 * sizeof(unsigned), stream);
    if (err != cudaSuccess) return 10000 + (int)err;
    dsv4_sample_prepare<<<(n + B - 1) / B, B, 0, stream>>>(logits, values, keys0, counts, n, repeat, freq, present, result);
    err = cudaGetLastError(); if (err != cudaSuccess) return 10000 + (int)err;
    for (int width = 1; width < n; width *= 2) {
        dsv4_sample_merge<<<(n + B - 1) / B, B, 0, stream>>>(keys0, keys1, n, width);
        err = cudaGetLastError(); if (err != cudaSuccess) return 10000 + (int)err;
        uint64_t* tmp = keys0; keys0 = keys1; keys1 = tmp;
    }
    int nb = (k + B - 1) / B;
    dsv4_sample_exp_scan<<<nb, B, 0, stream>>>(values, keys0, prefix, blocks, k, temperature);
    err = cudaGetLastError(); if (err != cudaSuccess) return 10000 + (int)err;
    dsv4_sample_offsets<<<1, 1, 0, stream>>>(blocks, nb);
    err = cudaGetLastError(); if (err != cudaSuccess) return 10000 + (int)err;
    dsv4_sample_draw<<<1, 1, 0, stream>>>(keys0, prefix, blocks, k, top_p, uniform, result, uniform_dev);
    return (int)cudaGetLastError();
}

extern "C" int memra_dsv4_sample_device(const float* logits, float* values,
    uint64_t* keys0, uint64_t* keys1, double* prefix, double* blocks,
    const int* counts, unsigned* result, int n, int k, double temperature,
    double top_p, double uniform, float repeat, float freq, float present, void* stream) {
    return dsv4_sample_device_enqueue(logits,values,keys0,keys1,prefix,blocks,counts,result,
        n,k,temperature,top_p,uniform,repeat,freq,present,stream,nullptr);
}
extern "C" int memra_dsv4_sample_device_replay(const float* logits, float* values,
    uint64_t* keys0, uint64_t* keys1, double* prefix, double* blocks,
    const int* counts, unsigned* result, int n, int k, double temperature,
    double top_p, const double* uniform, void* stream) {
    if (!uniform) return 40001;
    return dsv4_sample_device_enqueue(logits,values,keys0,keys1,prefix,blocks,counts,result,
        n,k,temperature,top_p,0.0,1.0f,0.0f,0.0f,stream,uniform);
}
