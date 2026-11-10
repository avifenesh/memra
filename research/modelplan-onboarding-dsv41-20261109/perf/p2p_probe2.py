#!/usr/bin/env python3
"""Why the engine's four-card gather copies run at ~2 GB/s a rank while the same pattern in p2p_probe.py runs at
26-31: the engine's factors one at a time, on the four cards, no model.

The engine (dsv41_gpu/tp.rs gather_packed): blocks from the stream-ordered pool (cuMemAllocAsync; the default pool's
access granted to every peer with cuMemPoolSetAccess, plus cuCtxEnablePeerAccess), copied with cuMemcpyDtoDAsync
(UVA, no device named) on the CONSUMER's stream, in ring order (consumer c pulls c+1, c+2, c+3, one after another).
p2p_probe.py used cudaMalloc blocks, cudaMemcpyPeerAsync, consumer streams in index order.

Arms (an all-gather of S bytes a rank, 3 warm + REPS timed, ms a gather and GB/s a rank = 3 S / ms):
  alloc  malloc (cudaMalloc) | pool (cudaMallocAsync from the default pool, access set as the engine does)
  api    peer (cudaMemcpyPeerAsync) | uva (cudaMemcpyAsync DeviceToDevice, the engine's call)
  dir    pull (the consumer's stream, ring order: the engine) | push (the producer's stream, ring order)
  plus   pool_fresh: pool blocks allocated for every gather (the engine before kept blocks)
Sizes 1 MiB (the engine's own probe: 2.3 GB/s a rank at 1 MiB) and 16 MiB.
usage: p2p_probe2.py --out DIR [--build-only] [--reps N] [--pairs-only]\n(per-pair rows: one copy alone a card pair, GB/s of that copy)
"""
import json
import os
import statistics
import sys
import time

ARGS = sys.argv[1:]
OUT = ARGS[ARGS.index("--out") + 1] if "--out" in ARGS else "p2p-probe2"
REPS = int(ARGS[ARGS.index("--reps") + 1]) if "--reps" in ARGS else 15
WARM = 3
SIZES = [1 << 20, 16 << 20]
EXT_DIR = os.environ.get("P2P_PROBE_EXT", os.path.join(os.getcwd(), "p2p-probe2-ext"))
N = 4

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(w) + ": " + cudaGetErrorString(e));
}
void setup(int64_t n) {  // peer access between every pair, and every default pool open to every peer
    for (int a = 0; a < n; ++a) {
        ck(cudaSetDevice(a), "set device");
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else ck(e, "enable peer");
        }
    }
    for (int o = 0; o < n; ++o) {
        cudaMemPool_t pool;
        ck(cudaDeviceGetDefaultMemPool(&pool, o), "default pool");
        for (int a = 0; a < n; ++a) {
            if (a == o) continue;
            cudaMemAccessDesc d = {};
            d.location.type = cudaMemLocationTypeDevice;
            d.location.id = a;
            d.flags = cudaMemAccessFlagsProtReadWrite;
            ck(cudaMemPoolSetAccess(pool, &d, 1), "pool access");
        }
    }
}
int64_t stream_new(int64_t dev) {
    ck(cudaSetDevice((int)dev), "set device");
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    return (int64_t)s;
}
void sync_dev(int64_t dev) { ck(cudaSetDevice((int)dev), "set device"); ck(cudaDeviceSynchronize(), "sync"); }
void sync_stream(int64_t s) { ck(cudaStreamSynchronize((cudaStream_t)s), "stream sync"); }
int64_t alloc(int64_t dev, int64_t bytes, int64_t pooled, int64_t stream) {
    ck(cudaSetDevice((int)dev), "set device");
    void* p = nullptr;
    if (pooled) ck(cudaMallocAsync(&p, (size_t)bytes, (cudaStream_t)stream), "malloc async");
    else ck(cudaMalloc(&p, (size_t)bytes), "malloc");
    return (int64_t)p;
}
void dealloc(int64_t dev, int64_t p, int64_t pooled, int64_t stream) {
    ck(cudaSetDevice((int)dev), "set device");
    if (pooled) ck(cudaFreeAsync((void*)p, (cudaStream_t)stream), "free async"); else ck(cudaFree((void*)p), "free");
}
void fill(int64_t dev, int64_t p, int64_t v, int64_t bytes) {
    ck(cudaSetDevice((int)dev), "set device");
    ck(cudaMemset((void*)p, (int)v, (size_t)bytes), "memset");
}
int64_t peek(int64_t dev, int64_t p) {
    ck(cudaSetDevice((int)dev), "set device");
    unsigned char b = 0;
    ck(cudaMemcpy(&b, (void*)p, 1, cudaMemcpyDeviceToHost), "peek");
    return b;
}
void copy(int64_t dst, int64_t dst_dev, int64_t src, int64_t src_dev, int64_t bytes, int64_t uva, int64_t stream) {
    if (uva) ck(cudaMemcpyAsync((void*)dst, (const void*)src, (size_t)bytes, cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream), "memcpy uva");
    else ck(cudaMemcpyPeerAsync((void*)dst, (int)dst_dev, (const void*)src, (int)src_dev, (size_t)bytes,
                                (cudaStream_t)stream), "memcpy peer");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("setup", &setup); m.def("stream_new", &stream_new); m.def("sync_dev", &sync_dev);
    m.def("sync_stream", &sync_stream); m.def("alloc", &alloc); m.def("dealloc", &dealloc);
    m.def("fill", &fill); m.def("peek", &peek); m.def("copy", &copy);
}
"""


def ext():
    from torch.utils.cpp_extension import load_inline
    os.makedirs(EXT_DIR, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return load_inline("p2p_probe2_ext", cpp_sources="", cuda_sources=SRC, build_directory=EXT_DIR,
                       extra_cuda_cflags=["-O3"], verbose=False)


def stats(ts, s):
    ms = sorted(t * 1e3 for t in ts)
    med = statistics.median(ms)
    return {"ms_median": round(med, 3), "ms_p10": round(ms[int(0.1 * (len(ms) - 1))], 3),
            "ms_p90": round(ms[int(round(0.9 * (len(ms) - 1)))], 3),
            "gbs_rank": round(3 * s / (med * 1e-3) / 1e9, 2), "reps": len(ms)}


def run(res):
    x = ext()
    x.setup(N)
    st = [x.stream_new(d) for d in range(N)]
    for s in SIZES:
        for pooled in (0, 1):
            src = [x.alloc(r, s, pooled, st[r]) for r in range(N)]
            dst = [x.alloc(r, N * s, pooled, st[r]) for r in range(N)]
            for r in range(N):
                x.sync_dev(r)
                x.fill(r, src[r], r + 1, s)

            def gather(uva, push, srcs=src, dsts=dst):
                if push:  # producer p pushes to p+1, p+2, p+3 on its own stream
                    for p in range(N):
                        for j in range(1, N):
                            c = (p + j) % N
                            x.copy(dsts[c] + p * s, c, srcs[p], p, s, uva, st[p])
                else:  # consumer c pulls from c+1, c+2, c+3 on its own stream (the engine)
                    for c in range(N):
                        for j in range(1, N):
                            p = (c + j) % N
                            x.copy(dsts[c] + p * s, c, srcs[p], p, s, uva, st[c])
                for r in range(N):
                    x.sync_stream(st[r])

            for uva in (0, 1):
                for push in (0, 1):
                    name = f"{'pool' if pooled else 'malloc'}-{'uva' if uva else 'peer'}-{'push' if push else 'pull'}"
                    try:
                        for _ in range(WARM):
                            gather(uva, push)
                        ts = []
                        for _ in range(REPS):
                            t0 = time.perf_counter()
                            gather(uva, push)
                            ts.append(time.perf_counter() - t0)
                        ok = all(x.peek(c, dst[c] + p * s) == p + 1 and x.peek(c, dst[c] + p * s + s - 1) == p + 1
                                 for c in range(N) for p in range(N) if p != c)
                        res.append({"arm": name, "bytes_a_rank": s, **stats(ts, s), "data_ok": ok})
                    except Exception as e:
                        res.append({"arm": name, "bytes_a_rank": s, "error": repr(e)[:300]})
                    print(json.dumps(res[-1]), flush=True)
            if pooled:
                # fresh pool blocks every gather: allocated, copied into, freed (the engine before kept blocks)
                try:
                    ts = []
                    for i in range(WARM + REPS):
                        t0 = time.perf_counter()
                        fs = [x.alloc(r, s, 1, st[r]) for r in range(N)]
                        fd = [x.alloc(r, N * s, 1, st[r]) for r in range(N)]
                        for r in range(N):
                            x.sync_stream(st[r])
                        gather(1, 0, fs, fd)
                        for r in range(N):
                            x.dealloc(r, fs[r], 1, st[r])
                            x.dealloc(r, fd[r], 1, st[r])
                            x.sync_stream(st[r])
                        if i >= WARM:
                            ts.append(time.perf_counter() - t0)
                    res.append({"arm": "pool_fresh-uva-pull", "bytes_a_rank": s, **stats(ts, s), "data_ok": True})
                except Exception as e:
                    res.append({"arm": "pool_fresh-uva-pull", "bytes_a_rank": s, "error": repr(e)[:300]})
                print(json.dumps(res[-1]), flush=True)
            for r in range(N):
                x.sync_dev(r)
                x.dealloc(r, src[r], pooled, st[r])
                x.dealloc(r, dst[r], pooled, st[r])
                x.sync_dev(r)


def per_pair(res, s=16 << 20, reps=9):
    """One copy alone for every ordered pair of cards (pool blocks, issued on the consumer's stream), by device
    address and by the peer call: GB/s of that one copy. Shows which card pairs the slow combination hits."""
    x = ext()
    x.setup(N)
    st = [x.stream_new(d) for d in range(N)]
    src = [x.alloc(r, s, 1, st[r]) for r in range(N)]
    dst = [x.alloc(r, N * s, 1, st[r]) for r in range(N)]
    for r in range(N):
        x.sync_dev(r)
        x.fill(r, src[r], r + 1, s)
    for uva in (1, 0):
        for p in range(N):
            for c in range(N):
                if p == c:
                    continue
                ts = []
                for i in range(WARM + reps):
                    t0 = time.perf_counter()
                    x.copy(dst[c] + p * s, c, src[p], p, s, uva, st[c])
                    x.sync_stream(st[c])
                    if i >= WARM:
                        ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)
                res.append({"arm": f"pair {p}->{c} pool-{'uva' if uva else 'peer'}-pull", "bytes_a_rank": s,
                            "ms_median": round(med * 1e3, 3), "ms_p10": round(min(ts) * 1e3, 3),
                            "ms_p90": round(max(ts) * 1e3, 3), "gbs_rank": round(s / med / 1e9, 2), "reps": reps,
                            "data_ok": x.peek(c, dst[c] + p * s) == p + 1})
                print(json.dumps(res[-1]), flush=True)
    for r in range(N):
        x.sync_dev(r)
        x.dealloc(r, src[r], 1, st[r])
        x.dealloc(r, dst[r], 1, st[r])
        x.sync_dev(r)


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--build-only" in ARGS:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        t0 = time.perf_counter()
        ext()
        print(f"built in {time.perf_counter() - t0:.1f} s: {EXT_DIR}")
        return
    res = []
    t0 = time.perf_counter()
    if "--pairs-only" not in ARGS:
        run(res)
    per_pair(res)
    json.dump({"results": res, "s": round(time.perf_counter() - t0, 1)}, open(os.path.join(OUT, "probe2.json"), "w"),
              indent=1)
    lines = ["| arm | MiB a rank | ms a gather median [p10, p90] | GB/s a rank | data |", "|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            lines.append(f"| {r['arm']} | {r['bytes_a_rank'] >> 20} | ERROR {r['error'][:90]} | | |")
        else:
            lines.append(f"| {r['arm']} | {r['bytes_a_rank'] >> 20} | {r['ms_median']} [{r['ms_p10']}, {r['ms_p90']}] | "
                         f"{r['gbs_rank']} | {'ok' if r['data_ok'] else 'WRONG'} |")
    open(os.path.join(OUT, "probe2.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
