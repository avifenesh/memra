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
    // The fused reduce+post kernel: block 0 alone crosses the fabric; these two words carry its
    // verdict to the other blocks (grid_go) and their completion back to it (grid_done), both
    // device-local. hp_seq counts the fused rounds so grid_done's target is known.
    alignas(128) unsigned grid_go;
    alignas(128) unsigned grid_done;
    alignas(128) unsigned hp_seq;
};

extern "C" int memra_tp_ar_signal_bytes(void) { return (int)sizeof(MemraArSignal); }
extern "C" int memra_tp_ar_seq_offset_bytes(void) { return (int)offsetof(MemraArSignal, seq); }

__device__ __forceinline__ void memra_ar_st_release(unsigned* addr, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(addr));
}

__device__ __forceinline__ unsigned memra_ar_ld_acquire(const unsigned* addr) {
    unsigned v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}

// ---------------------------------------------------------------- phase instrument
//
// WHY THIS SHAPE. The replay map ranks the residual 85 all-reduces at 1.977316 ms/step on rank 0
// and records them as "highest exposure, lowest readiness: needs a lower-overhead causal
// instrument" (darklanes research/dsv4f-replay-map-nf2-20260909). Every tool we already own
// either cannot see inside the collective or costs as much as the thing it measures:
//
//  - Nsight graph-node tracing is REFUSED by receipt, not by taste: it inflated the first-AR
//    residence to 4.288 ms and launch API to 4.152 ms against 26.656 us and 13.545 us on the same
//    program measured with events (darklanes #519, TRAP dsv4-nsys-graph-node-tracing-inflates-
//    launch-and-first-ar). A profiler that moves the quantity by two orders of magnitude cannot
//    attribute it.
//  - Per-AR CUDA event pairs cannot be created "outside capture" for a captured program at all.
//    Inside stream capture a cudaEventRecord becomes an EVENT RECORD NODE in the graph, so 86 ARs
//    would add 172 nodes per rank per step to a graph whose whole forward is 2870 nodes, change
//    the topology being measured, and pay host-visible event bookkeeping per replay. #519 used
//    event pairs successfully on ONE AR per step; that does not scale to the pool.
//
// So the stamps go INSIDE the kernel that is already a graph node. No new node, no new launch, no
// host API per reduce, and replay re-executes the same node so the instrument survives capture.
//
// TWO CLOCKS ON PURPOSE. `clock64()` is an SM-local cycle counter: cheap (a few cycles) and exact
// WITHIN one kernel, because block 0 never migrates off its SM mid-kernel, but meaningless across
// kernels or across SMs. `%globaltimer` is a device-wide nanosecond clock, comparable across
// kernels, but a much more expensive read. The phase DECOMPOSITION (peer wait / reduce / tail
// wait) therefore uses clock64 and pays four cheap reads, while TWO globaltimer reads per AR place
// the record on a device timeline (so the analyzer can ask whether an AR sits on the critical path
// or in slack) and calibrate ticks to nanoseconds from this very kernel's own span. The calibration
// is checkable: `(gt_exit - gt_entry)` against `(clk_exit - clk_entry)` must land in the card's
// clock band, and both units must close on the same phase sum. A stamp set that does not close is
// thrown out by the gate rather than reported.
//
// GATE-ONLY, AND BIT-IDENTICAL BY CONSTRUCTION. `trace == nullptr` is the ordinary program: every
// branch below is kernel-parameter-uniform, the reduce loop's operand order, indexing and
// arithmetic are untouched, and nothing the instrument writes is ever read back into the tape. The
// instrument is therefore NOT a new numeric class and the gate refuses on the first differing bit
// rather than on a tolerance.
struct MemraArPhase {
    unsigned long long gt_entry;    // %globaltimer, kernel entry
    unsigned long long gt_exit;     // %globaltimer, after the end barrier
    unsigned long long clk_entry;   // clock64, kernel entry
    unsigned long long clk_started; // clock64, start barrier satisfied  -> peer wait
    unsigned long long clk_reduced; // clock64, block reduce complete    -> transport + add
    unsigned long long clk_exit;    // clock64, end barrier satisfied    -> tail wait
    unsigned int site;              // 2*layer for the attention join, 2*layer+1 for the expert join
    unsigned int flag;              // this rank's AR epoch for block 0
    unsigned int smid;              // which SM block 0 ran on, so a stamp set is attributable
    unsigned int blocks;            // gridDim.x
    unsigned int elems;             // n
    unsigned int rank;
};

extern "C" int memra_tp_ar_phase_record_bytes(void) { return (int)sizeof(MemraArPhase); }

__device__ __forceinline__ unsigned long long memra_ar_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ unsigned int memra_ar_smid() {
    unsigned int s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
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
        int* __restrict__ err, long long spin_limit,
        const unsigned long long* fault = nullptr, int site = -1,
        MemraArPhase* trace = nullptr, unsigned long long* trace_cursor = nullptr,
        long trace_capacity = 0, unsigned int phase_site = 0u, int null_arm = 0,
        long long delay_ticks = 0) {
    // Live request-local diagnostic, before the real waits so a real timeout
    // takes precedence. The ordinary entry passes null and keeps its program.
    if (fault && threadIdx.x == 0) {
        unsigned long long word = *fault;
        if ((unsigned)(word >> 32) && (int)(word & 0xffffu) == site &&
            (int)((word >> 16) & 0xffffu) == rank) *(volatile int*)err = (int)(word >> 32);
    }
    // Red-arm delay. Gate-only, and the CALLER keys it: it passes a nonzero tick count to exactly
    // one rank's launch at one site. The instrument is only honest if a known delay injected on
    // rank 1 appears as rank 0's start-barrier wait at that site and nowhere else.
    if (delay_ticks > 0 && threadIdx.x == 0) {
        long long d0 = clock64();
        while (clock64() - d0 < delay_ticks) {
        }
    }
    if (delay_ticks > 0) __syncthreads();
    const int stamping = (trace != nullptr) && (blockIdx.x == 0);
    unsigned long long gt_entry = 0, clk_entry = 0, clk_started = 0, clk_reduced = 0;
    if (stamping && threadIdx.x == 0) {
        gt_entry = memra_ar_globaltimer();
        clk_entry = (unsigned long long)clock64();
    }
    unsigned flag = self_sg->seq[blockIdx.x] + 1;
    if (memra_ar_barrier(peer_sg->start, self_sg->start, flag, rank, spin_limit)) {
        if (threadIdx.x == 0) {
            *(volatile int*)err = 40043;
            self_sg->seq[blockIdx.x] = flag;
        }
        return;
    }
    if (stamping && threadIdx.x == 0) clk_started = (unsigned long long)clock64();
    if (null_arm) {
        // NULL-AR INSTRUMENT ARM, never a product arm: the answer is wrong on purpose. Launch
        // geometry, both barriers, the epoch counters and the refusal words are exactly the product
        // arm's; the only difference is that this rank reads its OWN operand twice instead of one
        // local and one peer operand, so the fabric is never touched. The difference between the
        // two arms' reduce phases is the transport and nothing else. Reading through the existing
        // restrict-qualified parameter (never a second alias for the same storage) keeps the
        // aliasing contract the product arm has.
        const float* s = (rank == 0) ? in_rank0 : in_rank1;
        for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (long)gridDim.x * blockDim.x) {
            out[i] = s[i] + s[i];
        }
    } else {
        for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += (long)gridDim.x * blockDim.x) {
            out[i] = in_rank0[i] + in_rank1[i];
        }
    }
    // Kernel-parameter-uniform, so this is a legal block-wide barrier: it makes `clk_reduced` the
    // block's reduce completion rather than thread 0's.
    if (trace != nullptr) __syncthreads();
    if (stamping && threadIdx.x == 0) clk_reduced = (unsigned long long)clock64();
    if (memra_ar_barrier(peer_sg->end, self_sg->end, flag, rank, spin_limit)) {
        if (threadIdx.x == 0) *(volatile int*)err = 40044;
    }
    if (threadIdx.x == 0) self_sg->seq[blockIdx.x] = flag;
    if (stamping && threadIdx.x == 0 && trace_capacity > 0) {
        // The cursor is device state and advances on every replay, so slots are never reused inside
        // a collection window. The host refuses a window whose cursor advanced past the capacity
        // rather than reporting aliased records.
        unsigned long long slot = atomicAdd(trace_cursor, 1ull);
        MemraArPhase* r = &trace[slot % (unsigned long long)trace_capacity];
        r->gt_entry = gt_entry;
        r->gt_exit = memra_ar_globaltimer();
        r->clk_entry = clk_entry;
        r->clk_started = clk_started;
        r->clk_reduced = clk_reduced;
        r->clk_exit = (unsigned long long)clock64();
        r->site = phase_site;
        r->flag = flag;
        r->smid = memra_ar_smid();
        r->blocks = gridDim.x;
        r->elems = (unsigned int)n;
        r->rank = (unsigned int)rank;
    }
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

extern "C" int memra_tp_ar_1stage_replay(const float* in_rank0, const float* in_rank1, float* out,
    void* self_sg, void* peer_sg, int rank, long n, int* err, long long spin_limit, int blocks,
    void* stream_v, const void* fault, int site) {
    if (n <= 0) return 40041;
    if (spin_limit <= 0) return 40042;
    if (rank < 0 || rank >= MEMRA_AR_RANKS) return 40045;
    if (blocks < 1 || blocks > MEMRA_AR_MAX_BLOCKS) return 40046;
    if (!fault || site < 0 || site >= 43) return 40047;
    memra_tp_ar_1stage_kernel<<<(unsigned)blocks, 512u, 0, (cudaStream_t)stream_v>>>(
        in_rank0,in_rank1,out,(MemraArSignal*)self_sg,(MemraArSignal*)peer_sg,rank,n,err,
        spin_limit,(const unsigned long long*)fault,site);
    TP_AR_ERR();
    return 0;
}

// Instrumented entry. Gate-only: `dsv4_ep.rs` reaches it only when the process armed the
// `MEMRA_DSV4_AR_PHASE` door, and no serving path arms it. It is a superset of both ordinary
// entries so the ONE kernel stays the thing under measurement: a separate instrumented kernel
// would be a different kernel and could not claim the product arm's timing. `fault` may be null
// here (the eager walk has no replay fault word) while `trace` may not.
extern "C" int memra_tp_ar_1stage_instr(const float* in_rank0, const float* in_rank1, float* out,
    void* self_sg, void* peer_sg, int rank, long n, int* err, long long spin_limit, int blocks,
    void* stream_v, const void* fault, int site, void* trace, void* trace_cursor,
    long trace_capacity, unsigned int phase_site, int null_arm, long long delay_ticks) {
    if (n <= 0) return 40041;
    if (spin_limit <= 0) return 40042;
    if (rank < 0 || rank >= MEMRA_AR_RANKS) return 40045;
    if (blocks < 1 || blocks > MEMRA_AR_MAX_BLOCKS) return 40046;
    if (fault && (site < 0 || site >= 43)) return 40047;
    if (!trace || !trace_cursor || trace_capacity <= 0) return 40048;
    if (null_arm < 0 || null_arm > 1) return 40049;
    if (delay_ticks < 0) return 40050;
    memra_tp_ar_1stage_kernel<<<(unsigned)blocks, 512u, 0, (cudaStream_t)stream_v>>>(
        in_rank0, in_rank1, out, (MemraArSignal*)self_sg, (MemraArSignal*)peer_sg, rank, n, err,
        spin_limit, (const unsigned long long*)fault, fault ? site : -1, (MemraArPhase*)trace,
        (unsigned long long*)trace_cursor, trace_capacity, phase_site, null_arm, delay_ticks);
    TP_AR_ERR();
    return 0;
}

// One-shot + hc post in ONE launch, t = 1 only: f = in_rank0 + in_rank1 (the reduce), then the
// hyper-connection post-branch for every stream k, out[k, i] = post[k] * f[i] + sum_j
// comb[j*hc + k] * residual[j*d + i]. BITWISE the pair memra_tp_ar_1stage + dsv4_hc_post_kernel:
// dsv4_gpu.cu is compiled -fmad=false, so the post arithmetic here is written with the
// non-contracting intrinsics (the 2026-09-07 tpwalk9 tape moved on exactly that). Block 0 alone
// runs the two fabric barriers (tp-ar-bench: every barrier block is a flag round trip); it
// publishes the start verdict to the other blocks through grid_go and gathers their completion
// through grid_done before the end barrier, so the post spreads over the grid the way the
// stand-alone hc_post did (64 blocks) while the crossing pays one block's round trips.
__device__ __forceinline__ void memra_ar_st_release_gpu(unsigned* addr, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%1], %0;" ::"r"(v), "l"(addr));
}
__device__ __forceinline__ unsigned memra_ar_ld_acquire_gpu(const unsigned* addr) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(addr));
    return v;
}
__global__ void __launch_bounds__(256) memra_tp_ar_1stage_hcpost_kernel(
        const float* __restrict__ in_rank0, const float* __restrict__ in_rank1,
        const float* __restrict__ residual, const float* __restrict__ post,
        const float* __restrict__ comb, float* __restrict__ out, int hc, int d,
        MemraArSignal* self_sg, MemraArSignal* peer_sg, int rank, int* __restrict__ err,
        long long spin_limit) {
    __shared__ int ok;
    const unsigned flag = self_sg->seq[0] + 1;
    const unsigned round = self_sg->hp_seq + 1;
    if (blockIdx.x == 0) {
        int expired = memra_ar_barrier(peer_sg->start, self_sg->start, flag, rank, spin_limit);
        if (threadIdx.x == 0) {
            if (expired) *(volatile int*)err = 40043;
            memra_ar_st_release_gpu(&self_sg->grid_go, expired ? (flag | 0x80000000u) : flag);
        }
    }
    if (threadIdx.x == 0) {
        long long t0 = clock64();
        unsigned v;
        while (((v = memra_ar_ld_acquire_gpu(&self_sg->grid_go)) & 0x7fffffffu) != flag) {
            if (clock64() - t0 > spin_limit) {
                v = flag | 0x80000000u;
                break;
            }
        }
        ok = (v & 0x80000000u) ? 0 : 1;
    }
    __syncthreads();
    if (ok) {
        for (int i = (int)blockIdx.x * blockDim.x + threadIdx.x; i < d;
             i += (int)gridDim.x * blockDim.x) {
            float f = __fadd_rn(in_rank0[i], in_rank1[i]);
            for (int k = 0; k < hc; k++) {
                float acc = __fmul_rn(post[k], f);
                for (int j = 0; j < hc; j++)
                    acc = __fadd_rn(acc, __fmul_rn(comb[j * hc + k], residual[(long)j * d + i]));
                out[(long)k * d + i] = acc;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(&self_sg->grid_done, 1u);
    }
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            const unsigned target = round * gridDim.x;
            long long t0 = clock64();
            while (memra_ar_ld_acquire_gpu(&self_sg->grid_done) < target) {
                if (clock64() - t0 > spin_limit) {
                    *(volatile int*)err = 40044;
                    break;
                }
            }
        }
        __syncthreads();
        if (memra_ar_barrier(peer_sg->end, self_sg->end, flag, rank, spin_limit)) {
            if (threadIdx.x == 0) *(volatile int*)err = 40044;
        }
        if (threadIdx.x == 0) {
            self_sg->seq[0] = flag;
            self_sg->hp_seq = round;
        }
    }
}

extern "C" int memra_tp_ar_1stage_hcpost(const float* in_rank0, const float* in_rank1,
                                         const float* residual, const float* post,
                                         const float* comb, float* out, int hc, int d,
                                         void* self_sg, void* peer_sg, int rank, int* err,
                                         long long spin_limit, int blocks, void* stream_v) {
    if (d <= 0 || hc <= 0) return 40041;
    if (spin_limit <= 0) return 40042;
    if (rank < 0 || rank >= MEMRA_AR_RANKS) return 40045;
    if (blocks < 1 || blocks > MEMRA_AR_MAX_BLOCKS) return 40046;
    cudaStream_t stream = (cudaStream_t)stream_v;
    memra_tp_ar_1stage_hcpost_kernel<<<(unsigned)blocks, 256u, 0, stream>>>(
        in_rank0, in_rank1, residual, post, comb, out, hc, d, (MemraArSignal*)self_sg,
        (MemraArSignal*)peer_sg, rank, err, spin_limit);
    TP_AR_ERR();
    return 0;
}
