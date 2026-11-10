#!/usr/bin/env python3
"""Four-card transport probe for the dsv41 prefill gathers: no model, no engine. An all-gather of S bytes a rank
(every rank's region lands in the three other ranks) by six methods, each repeated REPS times after WARM warm-ups:

  ce_serial    copy-engine peer copies (cudaMemcpyPeerAsync), the 12 copies one at a time
  ce_consumer  each consumer pulls its 3 regions on its own stream, the 4 consumers at once (4 in flight)
  ce_all       the 12 copies on 12 streams at once (12 in flight: today's large-gather regime)
  ks_all       kernel stores: each producer's kernel stores its region into the 3 peers, the 4 producers at once
  ks_serial    kernel stores, one producer at a time
  nccl         torch.distributed all_gather_into_tensor, NCCL (the venv's), 4 processes a card each

Sizes (bytes a rank): 10 MiB, 16 MiB (wo_a's 16.8 MB), 64 MiB. Output: ms a gather (median, p10, p90 over the
reps; a gather ends when its last copy ends) and GB/s a rank = 3 x S / ms (what each rank receives).
usage: p2p_probe.py --out DIR [--build-only] [--reps N] [--warm N]
"""
import json
import os
import statistics
import sys
import time

ARGS = sys.argv[1:]


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


OUT = arg("--out", "p2p-probe")
REPS = int(arg("--reps", "15"))
WARM = int(arg("--warm", "3"))
SIZES = [10 << 20, 16 << 20, 64 << 20]
EXT_DIR = os.environ.get("P2P_PROBE_EXT", os.path.join(os.getcwd(), "p2p-probe-ext"))
N = 4

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__global__ void store3(const uint4* __restrict__ s, uint4* __restrict__ a, uint4* __restrict__ b,
                       uint4* __restrict__ c, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x, st = (long)gridDim.x * blockDim.x;
    for (; i < n; i += st) { uint4 v = s[i]; a[i] = v; b[i] = v; c[i] = v; }
}

void enable_peers(int64_t n) {
    for (int a = 0; a < n; ++a) {
        ck(cudaSetDevice(a), "set device");
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;
            int ok = 0;
            ck(cudaDeviceCanAccessPeer(&ok, a, b), "can access peer");
            if (!ok) throw std::runtime_error("no peer access " + std::to_string(a) + " -> " + std::to_string(b));
            cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else ck(e, "enable peer access");
        }
    }
}

void peer_copy(int64_t dst, int64_t dst_dev, int64_t src, int64_t src_dev, int64_t bytes, int64_t stream) {
    ck(cudaMemcpyPeerAsync((void*)dst, (int)dst_dev, (const void*)src, (int)src_dev, (size_t)bytes,
                           (cudaStream_t)stream), "memcpy peer");
}

void peer_store(int64_t dev, int64_t src, std::vector<int64_t> dsts, int64_t bytes, int64_t blocks, int64_t stream) {
    ck(cudaSetDevice((int)dev), "set device");
    long n = bytes / 16;
    store3<<<(int)blocks, 256, 0, (cudaStream_t)stream>>>((const uint4*)src, (uint4*)dsts[0], (uint4*)dsts[1],
                                                          (uint4*)dsts[2], n);
    ck(cudaGetLastError(), "store kernel");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enable_peers", &enable_peers);
    m.def("peer_copy", &peer_copy);
    m.def("peer_store", &peer_store);
}
"""


def ext():
    from torch.utils.cpp_extension import load_inline
    os.makedirs(EXT_DIR, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return load_inline("p2p_probe_ext", cpp_sources="", cuda_sources=SRC, build_directory=EXT_DIR,
                       extra_cuda_cflags=["-O3"], verbose=False)


def stats(ts, s):
    ms = sorted(t * 1e3 for t in ts)
    med = statistics.median(ms)
    p10 = ms[max(0, int(0.1 * (len(ms) - 1)))]
    p90 = ms[min(len(ms) - 1, int(round(0.9 * (len(ms) - 1))))]
    return {"ms_median": round(med, 3), "ms_p10": round(p10, 3), "ms_p90": round(p90, 3),
            "gbs_rank": round(3 * s / (med * 1e-3) / 1e9, 2), "reps": len(ms)}


def single(res):
    import torch
    x = ext()
    x.enable_peers(N)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    for s in SIZES:
        src = [torch.full((s,), r + 1, dtype=torch.uint8, device=f"cuda:{r}") for r in range(N)]
        dst = [torch.zeros((N * s,), dtype=torch.uint8, device=f"cuda:{r}") for r in range(N)]
        pairs = [(p, c) for c in range(N) for p in range(N) if p != c]  # (producer, consumer)
        streams = {k: torch.cuda.Stream(device=f"cuda:{k[1]}") for k in pairs}
        cstream = [torch.cuda.Stream(device=f"cuda:{c}") for c in range(N)]
        pstream = [torch.cuda.Stream(device=f"cuda:{p}") for p in range(N)]

        def sync():
            for r in range(N):
                torch.cuda.synchronize(r)

        def cp(p, c, st):
            x.peer_copy(dst[c].data_ptr() + p * s, c, src[p].data_ptr(), p, s, st.cuda_stream)

        def ce_serial():
            for p, c in pairs:
                cp(p, c, cstream[c])
                torch.cuda.synchronize(c)

        def ce_consumer():
            for p, c in pairs:
                cp(p, c, cstream[c])
            sync()

        def ce_all():
            for k in pairs:
                cp(k[0], k[1], streams[k])
            sync()

        def ks(p):
            ds = [dst[c].data_ptr() + p * s for c in range(N) if c != p]
            x.peer_store(p, src[p].data_ptr(), ds, s, 2 * sm, pstream[p].cuda_stream)

        def ks_all():
            for p in range(N):
                ks(p)
            sync()

        def ks_serial():
            for p in range(N):
                ks(p)
                torch.cuda.synchronize(p)

        def one():  # a single copy 0 -> 1 alone (reference rate)
            cp(0, 1, cstream[1])
            torch.cuda.synchronize(1)

        for name, fn in [("ce_one_alone", one), ("ce_serial", ce_serial), ("ce_consumer", ce_consumer),
                         ("ce_all", ce_all), ("ks_all", ks_all), ("ks_serial", ks_serial)]:
            try:
                sync()
                for _ in range(WARM):
                    fn()
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    fn()
                    ts.append(time.perf_counter() - t0)
            except Exception as e:  # one method failing does not stop the others
                res.append({"method": name, "bytes_a_rank": s, "error": repr(e)[:300]})
                print(json.dumps(res[-1]), flush=True)
                continue
            st = stats(ts, s)
            if name == "ce_one_alone":
                st["gbs_rank"] = round(s / (st["ms_median"] * 1e-3) / 1e9, 2)  # one copy's rate
            # the data landed: every consumer holds every producer's fill value
            ok = all(int(dst[c][p * s]) == p + 1 and int(dst[c][p * s + s - 1]) == p + 1
                     for c in range(N) for p in range(N) if p != c) if name != "ce_one_alone" else True
            st["data_ok"] = ok
            res.append({"method": name, "bytes_a_rank": s, **st})
            print(json.dumps(res[-1]), flush=True)
            for d in dst:
                d.zero_()
        del src, dst
        torch.cuda.empty_cache()


def nccl_worker(rank, path, q):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    t0 = time.perf_counter()
    dist.init_process_group("nccl", init_method=f"file://{path}", rank=rank, world_size=N,
                            device_id=torch.device(f"cuda:{rank}"))
    out = {"init_s": None, "sizes": {}}
    w = torch.zeros(1, device="cuda")
    dist.all_reduce(w)
    torch.cuda.synchronize()
    out["init_s"] = round(time.perf_counter() - t0, 2)
    for s in SIZES:
        inp = torch.full((s,), rank + 1, dtype=torch.uint8, device="cuda")
        o = torch.zeros((N * s,), dtype=torch.uint8, device="cuda")
        for _ in range(WARM):
            dist.all_gather_into_tensor(o, inp)
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_gather_into_tensor(o, inp)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        ok = all(int(o[p * s]) == p + 1 and int(o[p * s + s - 1]) == p + 1 for p in range(N))
        out["sizes"][s] = {"ts": ts, "ok": ok}
    q.put((rank, out))
    dist.destroy_process_group()


def nccl(res):
    import torch.multiprocessing as mp
    path = os.path.join(OUT, "nccl-init")
    if os.path.exists(path):
        os.remove(path)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=nccl_worker, args=(r, path, q)) for r in range(N)]
    for p in ps:
        p.start()
    got = dict(q.get(timeout=150) for _ in range(N))
    for p in ps:
        p.join(timeout=30)
    print(json.dumps({"nccl_init_s": [got[r]["init_s"] for r in range(N)]}), flush=True)
    for s in SIZES:
        # a gather ends when its slowest rank ends: the per-rep maximum over the ranks
        ts = [max(got[r]["sizes"][s]["ts"][i] for r in range(N)) for i in range(REPS)]
        st = stats(ts, s)
        st["data_ok"] = all(got[r]["sizes"][s]["ok"] for r in range(N))
        res.append({"method": "nccl", "bytes_a_rank": s, **st})
        print(json.dumps(res[-1]), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--build-only" in ARGS:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        t0 = time.perf_counter()
        ext()
        print(f"built in {time.perf_counter() - t0:.1f} s: {EXT_DIR}")
        return
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_FILE", os.path.join(OUT, "nccl.%h.%p.log"))
    t0 = time.perf_counter()
    res = []
    for part in (single, nccl):
        try:
            part(res)
        except Exception as e:
            res.append({"method": part.__name__, "bytes_a_rank": 0, "error": repr(e)[:300]})
            print(json.dumps(res[-1]), flush=True)
        if part is single:
            t1 = time.perf_counter()
    t2 = time.perf_counter()
    with open(os.path.join(OUT, "probe.json"), "w") as f:
        json.dump({"results": res, "single_s": round(t1 - t0, 1), "nccl_s": round(t2 - t1, 1)}, f, indent=1)
    lines = ["| method | MiB a rank | ms a gather median [p10, p90] | GB/s a rank | data |", "|---|---|---|---|---|"]
    for r in res:
        if "error" in r:
            lines.append(f"| {r['method']} | {r['bytes_a_rank'] >> 20} | ERROR {r['error'][:80]} | | |")
            continue
        lines.append(f"| {r['method']} | {r['bytes_a_rank'] >> 20} | {r['ms_median']} [{r['ms_p10']}, {r['ms_p90']}] | "
                     f"{r['gbs_rank']} | {'ok' if r['data_ok'] else 'WRONG'} |")
    lines.append(f"\nsingle-process part {t1 - t0:.1f} s, NCCL part {t2 - t1:.1f} s (incl. spawn and init); "
                 "ce_one_alone's GB/s is one copy's rate")
    open(os.path.join(OUT, "probe.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
