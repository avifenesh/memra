// Small-message cross-rank all-reduce for TP decode (lane/tp-allreduce-20260906).
//
// WHY. The TP-2 join costs about 500 us today and there are ~90 of them per token, which is
// why TP-2 measured 3.1x SLOWER than the pipeline split it should replace. None of that is
// transport. `tp_transport.rs`'s default is `host-canonical`: every hop is `dtoh` -> host ->
// `htod`, and `Engine::dtoh` ends in `stream().synchronize()`, so each leg is a FULL STREAM
// DRAIN. A drain does not cost bytes, it costs everything the stream has pending, i.e. that
// layer's compute, twice per join. The `peer-pull` arm replaced the host bounce with event
// ordering but kept the shape: a consumer-side PULL still makes the reading rank wait for the
// producing rank at every layer, so the two cards stay single-file. The link on the served
// pair is NV18, eighteen NVLink links at 53.125 GB/s = 956 GB/s; an 8 KB two-rank all-reduce
// on that is single-digit microseconds.
//
// THE SHAPE. One-shot, symmetric, no host boundary and no drain. Each rank PUSHES its partial
// straight into the peer's staging buffer over NVLink (peer access is already granted by
// `tp::grant_peer_access`, so a peer pointer is directly dereferenceable from a kernel in this
// context), then each rank FOLDS the buffer its peer wrote into its own accumulator. Ordering
// between the two halves is a cross-stream event, the same contract `tp_transport`'s
// `PeerPullLink::publish` already uses: the folding stream simply does not start until the
// peer's push event fires.
//
// WHY AN EVENT AND NOT A FLAG. The first cut had the fold spin on a peer-armed flag. It is
// correct on two cards and it deadlocks on one: the same-device gate runs both ranks as two
// contexts on one GPU, the spinning fold fills the SMs, and the peer's push can never be
// scheduled to arm the flag. Measured 2026-09-06 on the rig: bitwise-correct at 4 KiB and
// 64 KiB, every element wrong at 256 KiB, which is the spin winning the whole device. An event
// wait occupies nothing, cannot starve the peer, and needs no timeout to reason about.
#include <cuda_runtime.h>

#define TP_AR_ERR()                                            \
    do {                                                       \
        cudaError_t ce_ = cudaGetLastError();                  \
        if (ce_ != cudaSuccess) return 10000 + (int)ce_;       \
    } while (0)

// Grid-stride push of `n` floats into the peer's staging buffer. float4 while the tail allows
// it: the pointers are cudaMalloc'd and therefore 256 B aligned, so only `n` decides.
__global__ void memra_tp_ar_push_kernel(const float* __restrict__ src, float* __restrict__ peer,
                                        long n) {
    long stride = (long)gridDim.x * blockDim.x;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n4 = n / 4;
    const float4* s4 = (const float4*)src;
    float4* p4 = (float4*)peer;
    for (long j = i; j < n4; j += stride) p4[j] = s4[j];
    for (long j = n4 * 4 + i; j < n; j += stride) peer[j] = src[j];
}

// `dst += stage`, run only once the peer's push event has fired.
__global__ void memra_tp_ar_fold_kernel(float* __restrict__ dst, const float* __restrict__ stage,
                                        long n) {
    long stride = (long)gridDim.x * blockDim.x;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n4 = n / 4;
    float4* d4 = (float4*)dst;
    const float4* s4 = (const float4*)stage;
    for (long j = i; j < n4; j += stride) {
        float4 a = d4[j], b = s4[j];
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        a.w += b.w;
        d4[j] = a;
    }
    for (long j = n4 * 4 + i; j < n; j += stride) dst[j] += stage[j];
}

static inline int memra_tp_ar_blocks(long units) {
    long want = (units + 255) / 256;
    if (want < 1) return 1;
    return (int)(want > 1024 ? 1024 : want);
}

extern "C" int memra_tp_ar_push(const float* src, float* peer_stage, long n, void* stream_v) {
    if (n <= 0) return 40041;
    cudaStream_t stream = (cudaStream_t)stream_v;
    memra_tp_ar_push_kernel<<<memra_tp_ar_blocks(n / 4), 256, 0, stream>>>(src, peer_stage, n);
    TP_AR_ERR();
    return 0;
}

extern "C" int memra_tp_ar_fold(float* dst, const float* stage, long n, void* stream_v) {
    if (n <= 0) return 40041;
    cudaStream_t stream = (cudaStream_t)stream_v;
    memra_tp_ar_fold_kernel<<<memra_tp_ar_blocks(n / 4), 256, 0, stream>>>(dst, stage, n);
    TP_AR_ERR();
    return 0;
}

// Strided push: `rows` rows of `row_len` floats, `src_stride` apart in the source and
// `dst_stride` apart in the peer. This is the shape the TP gather actually needs: the full
// attention matrix is TOKEN-MAJOR, so rank r's part lands at `tok * full + r * part` for every
// token, not as one contiguous run. At t=1 it degenerates to the contiguous push.
__global__ void memra_tp_ar_push_2d_kernel(const float* __restrict__ src, float* __restrict__ peer,
                                           long rows, long row_len, long src_stride,
                                           long dst_stride) {
    long stride = (long)gridDim.x * blockDim.x;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        const float* s = src + r * src_stride;
        float* d = peer + r * dst_stride;
        for (long j = (long)blockIdx.x * blockDim.x + threadIdx.x; j < row_len; j += stride)
            d[j] = s[j];
    }
}

extern "C" int memra_tp_ar_push_2d(const float* src, float* peer_stage, long rows, long row_len,
                                   long src_stride, long dst_stride, void* stream_v) {
    if (rows <= 0 || row_len <= 0) return 40041;
    cudaStream_t stream = (cudaStream_t)stream_v;
    unsigned gy = (unsigned)(rows > 65535 ? 65535 : rows);
    dim3 grid((unsigned)memra_tp_ar_blocks(row_len), gy);
    memra_tp_ar_push_2d_kernel<<<grid, 256, 0, stream>>>(src, peer_stage, rows, row_len, src_stride,
                                                          dst_stride);
    TP_AR_ERR();
    return 0;
}
