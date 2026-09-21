#!/usr/bin/env python3
"""Pinned host arena reserve harness (memra#385): one cuMemHostAlloc versus N parallel chunks.

Measures exactly the driver call the arena makes at startup (`PinnedHostArena::reserve`,
crates/memra-engine/src/pinned_host.rs: cudarc `malloc_host` = cuMemHostAlloc with
CU_MEMHOSTALLOC_PORTABLE) through ctypes on libcuda, so no engine binary and no engine change
is needed to compare reserve mechanisms. Arms:

  single   one cuMemHostAlloc(bytes)                        (today's arena, pinned_host.rs:102)
  chunked  N threads, each cuMemHostAlloc(bytes / N), joined (candidate 1 of #385)

Interleaved A/B pairs in both orders (AB x pairs, then BA x pairs) inside ONE process and one
window; every allocation is freed before the next so the box never holds two arenas. The
default size is 75 percent of MemFree at start (the task's "largest arena the box's free host
RAM allows minus 25 percent", read literally so the pin never evicts another tenant's page
cache on a shared box; `--basis available` sizes on MemAvailable for a box that runs nothing
else), rounded down to 2 MiB. Prints one JSON row per
measurement and a summary with per-arm medians. A harness receipt: the numbers describe the
box that ran it and nothing else; the decision cell for #385 is the 2x B200 pair.

usage: pinned-host-reserve-bench.py [--bytes N | --fraction 0.75 [--basis free|available]] [--chunks 8]
                                     [--pairs-per-order 5] [--device 0] [--out receipt.json]
Exit 0 = every allocation succeeded and was freed; 2 = a driver call failed (the code is printed).
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time

CU_MEMHOSTALLOC_PORTABLE = 0x01
ALIGN = 2 << 20


def meminfo(field):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise SystemExit(f"REFUSED: /proc/meminfo has no {field}")


def gpu_state():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        return out
    except Exception as e:  # noqa: BLE001 - a receipt field, never a verdict
        return f"unavailable: {e}"


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


class Cuda:
    def __init__(self, device):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cuMemHostAlloc.restype = ctypes.c_int
        self.lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cuMemFreeHost.restype = ctypes.c_int
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.check(self.lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), dev), "cuDevicePrimaryCtxRetain")
        self.check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    @staticmethod
    def check(rc, what):
        if rc != 0:
            raise SystemExit(f"REFUSED: {what} returned CUresult {rc}")

    def bind_thread(self):
        self.check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent(thread)")

    def alloc(self, nbytes):
        p = ctypes.c_void_p()
        rc = self.lib.cuMemHostAlloc(ctypes.byref(p), nbytes, CU_MEMHOSTALLOC_PORTABLE)
        return rc, p

    def free(self, p):
        return self.lib.cuMemFreeHost(p)


def measure_single(cuda, nbytes):
    t0 = time.perf_counter()
    rc, p = cuda.alloc(nbytes)
    t1 = time.perf_counter()
    if rc != 0:
        return {"ok": False, "rc": rc, "alloc_ms": (t1 - t0) * 1e3}
    t2 = time.perf_counter()
    frc = cuda.free(p)
    t3 = time.perf_counter()
    return {"ok": frc == 0, "rc": frc, "alloc_ms": (t1 - t0) * 1e3, "free_ms": (t3 - t2) * 1e3}


def measure_chunked(cuda, nbytes, chunks):
    per = (nbytes // chunks) // ALIGN * ALIGN
    results = [None] * chunks

    def worker(i):
        cuda.bind_thread()
        s = time.perf_counter()
        rc, p = cuda.alloc(per)
        results[i] = (rc, p, (time.perf_counter() - s) * 1e3)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(chunks)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.perf_counter()
    rcs = [r[0] for r in results]
    row = {"ok": all(rc == 0 for rc in rcs), "rc": rcs, "alloc_ms": (t1 - t0) * 1e3,
           "chunk_bytes": per, "chunk_alloc_ms": [round(r[2], 3) for r in results]}
    t2 = time.perf_counter()
    frcs = [cuda.free(r[1]) for r in results if r[0] == 0]
    t3 = time.perf_counter()
    row["free_ms"] = (t3 - t2) * 1e3
    row["ok"] = row["ok"] and all(rc == 0 for rc in frcs)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--bytes", type=int, help="arena size in bytes (rounded down to 2 MiB)")
    size.add_argument("--fraction", type=float, default=0.75, help="fraction of the --basis field at start (default 0.75)")
    ap.add_argument("--basis", choices=["free", "available"], default="free",
                    help="MemFree (default: never evicts another tenant's page cache) or MemAvailable")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--pairs-per-order", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", help="write the receipt JSON here as well as to stdout")
    args = ap.parse_args()
    if args.chunks < 2 or args.pairs_per_order < 1:
        raise SystemExit("REFUSED: --chunks >= 2 and --pairs-per-order >= 1")
    avail0 = meminfo("MemAvailable")
    free0 = meminfo("MemFree")
    total = meminfo("MemTotal")
    basis0 = free0 if args.basis == "free" else avail0
    nbytes = args.bytes if args.bytes else int(basis0 * args.fraction)
    nbytes = nbytes // ALIGN * ALIGN
    if nbytes <= 0 or nbytes >= avail0:
        raise SystemExit(f"REFUSED: {nbytes} bytes is not below MemAvailable {avail0}")
    cuda = Cuda(args.device)
    receipt = {
        "kind": "pinned-host-reserve-bench",
        "issue": "memra#385",
        "bytes": nbytes,
        "basis": args.basis if not args.bytes else "bytes",
        "fraction": args.fraction if not args.bytes else None,
        "chunks": args.chunks,
        "pairs_per_order": args.pairs_per_order,
        "mem_total": total,
        "mem_available_at_start": avail0,
        "mem_free_at_start": free0,
        "gpu_at_start": gpu_state(),
        "loadavg_at_start": loadavg(),
        "rows": [],
    }
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}), flush=True)
    arms = {"single": lambda: measure_single(cuda, nbytes),
            "chunked": lambda: measure_chunked(cuda, nbytes, args.chunks)}
    failed = False
    for order in ("AB", "BA"):
        seq = ["single", "chunked"] if order == "AB" else ["chunked", "single"]
        for pair in range(args.pairs_per_order):
            for arm in seq:
                avail = meminfo("MemAvailable")
                free = meminfo("MemFree")
                row = arms[arm]()
                row.update(order=order, pair=pair, arm=arm, mem_available_before=avail, mem_free_before=free,
                           utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                receipt["rows"].append(row)
                print(json.dumps(row), flush=True)
                failed |= not row["ok"]
                if not row["ok"]:
                    break
            if failed:
                break
        if failed:
            break
    receipt["gpu_at_end"] = gpu_state()
    receipt["loadavg_at_end"] = loadavg()
    receipt["mem_available_at_end"] = meminfo("MemAvailable")
    receipt["mem_free_at_end"] = meminfo("MemFree")
    summary = {}
    for arm in arms:
        ok = [r for r in receipt["rows"] if r["arm"] == arm and r["ok"]]
        if ok:
            summary[arm] = {
                "n": len(ok),
                "alloc_ms_median": statistics.median(r["alloc_ms"] for r in ok),
                "alloc_ms_min": min(r["alloc_ms"] for r in ok),
                "alloc_ms_max": max(r["alloc_ms"] for r in ok),
                "free_ms_median": statistics.median(r["free_ms"] for r in ok),
                "alloc_gib_per_s_median": statistics.median((nbytes / (1 << 30)) / (r["alloc_ms"] / 1e3) for r in ok),
            }
            for order in ("AB", "BA"):
                sub = [r["alloc_ms"] for r in ok if r["order"] == order]
                if sub:
                    summary[arm][f"alloc_ms_median_{order}"] = statistics.median(sub)
    receipt["summary"] = summary
    receipt["status"] = "failed" if failed else "executed-not-qualified"
    print(json.dumps({"summary": summary, "status": receipt["status"]}, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=1)
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
