#!/usr/bin/env python3
"""Driver-independent read-rate probe for pinned host memory (WP-A day 13, the discrepancy between
the gate's write-combined hash rate and lane C's whole-demote line): cuMemHostAlloc through ctypes
on libcuda with flags 4 (write-combined) and 0 (cached), a memset, then a sequential read of the
same bytes by hashlib.sha256 and by a bulk copy, interleaved A B A B and B A B A, in one process.
No engine binary, no cudarc: what the CPU sees when it reads each attribute on this host.
Run ONLY through tier-battery (it takes the device's primary context).
usage: pinned-read-probe.py --bytes N [--pairs N] [--device 0]
"""
import argparse
import ctypes
import hashlib
import json
import statistics
import sys
import time

WC, CACHED = 4, 0


class Cuda:
    def __init__(self, device):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cuMemHostAlloc.restype = ctypes.c_int
        self.lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cuMemHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
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
            raise SystemExit(f"{what} failed: CUresult {rc}")

    def alloc(self, nbytes, flags):
        p = ctypes.c_void_p()
        self.check(self.lib.cuMemHostAlloc(ctypes.byref(p), nbytes, flags), f"cuMemHostAlloc(flags={flags})")
        got = ctypes.c_uint()
        self.check(self.lib.cuMemHostGetFlags(ctypes.byref(got), p), "cuMemHostGetFlags")
        return p, got.value

    def free(self, p):
        self.check(self.lib.cuMemFreeHost(p), "cuMemFreeHost")


def roundtrip(cuda, nbytes, flags):
    t0 = time.perf_counter()
    p, got = cuda.alloc(nbytes, flags)
    ctypes.memset(p, 0x5a, nbytes)
    alloc_ms = (time.perf_counter() - t0) * 1e3
    view = (ctypes.c_char * nbytes).from_address(p.value)
    t0 = time.perf_counter()
    digest = hashlib.sha256(view).hexdigest()
    sha_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    copy = bytes(view)
    copy_ms = (time.perf_counter() - t0) * 1e3
    exact = copy == b"\x5a" * nbytes and digest == hashlib.sha256(copy).hexdigest()
    cuda.free(p)
    return {"flags": flags, "driver_flags": got, "alloc_ms": alloc_ms, "sha256_ms": sha_ms, "copy_ms": copy_ms, "exact": exact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    cuda = Cuda(a.device)
    rows = []
    for arm in (WC, CACHED):
        r = roundtrip(cuda, a.bytes, arm)
        print("PROBE warmup", json.dumps(r))
    for order, arms in enumerate([(WC, CACHED), (CACHED, WC)], start=1):
        for pair in range(1, a.pairs + 1):
            for arm in arms:
                r = roundtrip(cuda, a.bytes, arm)
                r.update(order=order, pair=pair)
                rows.append(r)
                print("PROBE timed", json.dumps(r))
    summary = {}
    for arm, name in ((WC, "write-combined"), (CACHED, "cached")):
        xs = [r for r in rows if r["flags"] == arm]
        summary[name] = {
            "n": len(xs),
            "sha256_ms_median": statistics.median(r["sha256_ms"] for r in xs),
            "copy_ms_median": statistics.median(r["copy_ms"] for r in xs),
            "alloc_ms_median": statistics.median(r["alloc_ms"] for r in xs),
            "sha256_MBps": a.bytes / 1e6 / (statistics.median(r["sha256_ms"] for r in xs) / 1e3),
            "driver_flags": sorted({r["driver_flags"] for r in xs}),
            "exact_all": all(r["exact"] for r in xs),
        }
    print("RESULT " + json.dumps({"cell": "pinned-read-probe", "bytes": a.bytes, "pairs_per_order": a.pairs, "python_sha256": True, "summary": summary, "qualification": False}))
    return 0 if all(r["exact"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
