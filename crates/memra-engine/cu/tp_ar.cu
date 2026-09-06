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

// ---------------------------------------------------------------- one-shot all-reduce
//
// SHAPE TAKEN FROM vLLM's `csrc/custom_all_reduce.cuh` (`cross_device_reduce_1stage` plus the
// `barrier_at_start` / `barrier_at_end` pair in `custom_collective_common.cuh`), because the
// push-then-fold pipeline above is the wrong shape and its cost says so: 4 kernel launches and 8
// cross-context CUDA event operations per reduce, which measured 20-26 us for 16 KB on a pair
// whose fabric is 956 GB/s. That is host overhead, not bandwidth.
//
// The right shape is ONE KERNEL PER RANK AND NO EVENTS AT ALL. Each rank's kernel synchronises
// through flags written into the peer's memory, reads BOTH ranks' input buffers directly (peer
// access makes the peer pointer dereferenceable from this context), and computes the whole sum
// locally. No staging buffer, no copy, and nothing for the host to do per reduce beyond the two
// launches.
//
// BITWISE IDENTICAL ACROSS RANKS BY CONSTRUCTION, and vLLM's comment names the reason: the
// operand order is indexed by GLOBAL RANK and is therefore the same on every rank, so both
// compute the same expression rather than mirror images of it.
//
// TWO COUNTER SETS, and the reason is subtle enough to copy verbatim rather than rediscover: a
// peer block can reach the SECOND barrier while this block is still spinning on the FIRST, and
// with one counter it would write counter+1 into the value this block is waiting on. `start` and
// `end` alternate so that cannot happen.
//
// The spin is BOUNDED here where vLLM's is not. A device-side barrier can hang the card if the
// peer launch never arrives, and a bounded wait turns a wiring bug into a readable refusal
// instead of a wedged GPU. The ranks are on different devices (ArLink refuses a same-device
// pairing), so no legitimate peer can be starved by this kernel's own occupancy.
#define MEMRA_AR_MAX_BLOCKS 72
#define MEMRA_AR_RANKS 2

struct MemraArSignal {
    alignas(128) unsigned start[MEMRA_AR_MAX_BLOCKS][MEMRA_AR_RANKS];
    alignas(128) unsigned end[MEMRA_AR_MAX_BLOCKS][MEMRA_AR_RANKS];
    alignas(128) unsigned seq[MEMRA_AR_MAX_BLOCKS];
};

extern "C" int memra_tp_ar_signal_bytes(void) { return (int)sizeof(MemraArSignal); }

__device__ __forceinline__ void memra_ar_st_release(unsigned* addr, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(addr));
}

__device__ __forceinline__ unsigned memra_ar_ld_acquire(const unsigned* addr) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// Returns 0 on success, 1 if the bounded wait expired.
__device__ __forceinline__ int memra_ar_barrier(unsigned (*peer_ctr)[MEMRA_AR_RANKS],
                                                unsigned (*self_ctr)[MEMRA_AR_RANKS], unsigned flag,
                                                int rank, long long spin_limit) {
    __shared__ int expired;
    if (threadIdx.x == 0) expired = 0;
    __syncthreads();
    if (threadIdx.x < MEMRA_AR_RANKS) {
        // Every rank's slot in EVERY rank's array, this one's included: the wait below spins on
        // self_ctr[b][t] for t in 0..RANKS, and slot `rank` of the local array has no other
        // writer. Without this store the barrier can only expire (tpar2 2026-09-06 on the pair:
        // the one-shot returned on 40043 with x untouched, which the gate read as "rank 0 got
        // its own operand back", the same bits staged or in place).
        if (threadIdx.x == 0) memra_ar_st_release(&self_ctr[blockIdx.x][rank], flag);
        memra_ar_st_release(&peer_ctr[blockIdx.x][rank], flag);
        long long t0 = clock64();
        while (memra_ar_ld_acquire(&self_ctr[blockIdx.x][threadIdx.x]) != flag) {
            if (clock64() - t0 > spin_limit) {
                expired = 1;
                break;
            }
        }
    }
    __syncthreads();
    return expired;
}

// `in_rank0` and `in_rank1` are the two ranks' input buffers in GLOBAL RANK ORDER, one of which is
// local and one of which is the peer's. `out` may alias this rank's input: each thread reads both
// operands at index i before writing index i, and no other thread touches that index.
__global__ void __launch_bounds__(512, 1) memra_tp_ar_1stage_kernel(
        const float* __restrict__ in_rank0, const float* __restrict__ in_rank1,
        float* __restrict__ out, MemraArSignal* self_sg, MemraArSignal* peer_sg, int rank, long n,
        int* __restrict__ err, long long spin_limit) {
    unsigned flag = self_sg->seq[blockIdx.x] + 1;
    if (memra_ar_barrier(peer_sg->start, self_sg->start, flag, rank, spin_limit)) {
        if (threadIdx.x == 0) {
            *(volatile int*)err = 40043;
            self_sg->seq[blockIdx.x] = flag;
        }
        return;
    }
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x) {
        out[i] = in_rank0[i] + in_rank1[i];
    }
    if (memra_ar_barrier(peer_sg->end, self_sg->end, flag, rank, spin_limit)) {
        if (threadIdx.x == 0) *(volatile int*)err = 40044;
    }
    if (threadIdx.x == 0) self_sg->seq[blockIdx.x] = flag;
}

extern "C" int memra_tp_ar_1stage(const float* in_rank0, const float* in_rank1, float* out,
                                  void* self_sg, void* peer_sg, int rank, long n, int* err,
                                  long long spin_limit, int blocks, void* stream_v) {
    if (n <= 0) return 40041;
    if (spin_limit <= 0) return 40042;
    if (rank < 0 || rank >= MEMRA_AR_RANKS) return 40045;
    if (blocks < 1 || blocks > MEMRA_AR_MAX_BLOCKS) return 40046;
    cudaStream_t stream = (cudaStream_t)stream_v;
    memra_tp_ar_1stage_kernel<<<(unsigned)blocks, 512u, 0, stream>>>(
        in_rank0, in_rank1, out, (MemraArSignal*)self_sg, (MemraArSignal*)peer_sg, rank, n, err,
        spin_limit);
    TP_AR_ERR();
    return 0;
}
