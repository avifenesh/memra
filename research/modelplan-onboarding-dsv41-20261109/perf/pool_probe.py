#!/usr/bin/env python3
"""Does a block reserved in the device's default pool and freed serve later allocations of other sizes without the
pool growing? (dsv41 Runner::load_resident maps the runs' reserve this way.) Release threshold pinned to the maximum,
as memra's Engine::new does. Three passes over one list of sizes (a 32,768-token prefill's first allocations: the
stream 1.34 GB, the ids-sized rows, then workspaces), each allocation timed on the host:
  cold      the pool empty (trimmed to 0)
  reserved  after one 7.34 GiB block was allocated and freed
  again     a second pass (the pool's own high-water mark)
Prints each allocation's host time and the pool's reserved bytes after each pass. stdlib only (ctypes, libcuda).
usage: pool_probe.py [DEVICE]"""
import ctypes as C
import sys
import time

dev_i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cu = C.CDLL("libcuda.so.1")


def ok(rc, what):
    if rc != 0:
        sys.exit(f"{what}: CUresult {rc}")


ok(cu.cuInit(0), "cuInit")
dev = C.c_int()
ok(cu.cuDeviceGet(C.byref(dev), dev_i), "cuDeviceGet")
ctx = C.c_void_p()
ok(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
ok(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
pool = C.c_void_p()
ok(cu.cuDeviceGetDefaultMemPool(C.byref(pool), dev), "cuDeviceGetDefaultMemPool")
RELEASE_THRESHOLD, RESERVED_CURRENT, USED_CURRENT = 4, 5, 7
thr = C.c_uint64(2**64 - 1)
ok(cu.cuMemPoolSetAttribute(pool, RELEASE_THRESHOLD, C.byref(thr)), "set release threshold")
st = C.c_void_p()
ok(cu.cuStreamCreate(C.byref(st), 0), "cuStreamCreate")
cu.cuMemAllocAsync.argtypes = [C.POINTER(C.c_uint64), C.c_size_t, C.c_void_p]
cu.cuMemFreeAsync.argtypes = [C.c_uint64, C.c_void_p]
cu.cuMemPoolTrimTo.argtypes = [C.c_void_p, C.c_size_t]


def attr(a):
    v = C.c_uint64()
    ok(cu.cuMemPoolGetAttribute(pool, a, C.byref(v)), "pool attribute")
    return v.value


def alloc(n):
    p = C.c_uint64()
    t = time.perf_counter()
    ok(cu.cuMemAllocAsync(C.byref(p), n, st), f"alloc {n}")
    return p.value, (time.perf_counter() - t) * 1e3


def free(p):
    ok(cu.cuMemFreeAsync(p, st), "free")


def sync():
    ok(cu.cuStreamSynchronize(st), "sync")


MB = 1 << 20
sizes = [1342177280, 335544320, 268435456, 134217728, 671088640, 50331648, 402653184, 201326592, 1073741824, 16 * MB]
reserve = int(7.34 * (1 << 30))


def run(name):
    ps, ts = [], []
    for n in sizes:
        p, t = alloc(n)
        ps.append(p)
        ts.append(t)
    sync()
    used = attr(USED_CURRENT)
    for p in ps:
        free(p)
    sync()
    print(f"{name:9s} allocations ms: " + " ".join(f"{t:.2f}" for t in ts) + f"  (sum {sum(ts):.1f} ms, max {max(ts):.2f})"
          f"; reserved after {attr(RESERVED_CURRENT) / (1 << 30):.2f} GiB, used at peak {used / (1 << 30):.2f} GiB")


cu.cuMemPoolTrimTo(pool, 0)
sync()
print(f"device {dev_i}: sizes MB {[n // MB for n in sizes]} (sum {sum(sizes) / (1 << 30):.2f} GiB); reserve "
      f"{reserve / (1 << 30):.2f} GiB; reserved at start {attr(RESERVED_CURRENT) / (1 << 30):.2f} GiB")
run("cold")
cu.cuMemPoolTrimTo(pool, 0)
sync()
p, t = alloc(reserve)
free(p)
sync()
print(f"reserve   one {reserve / (1 << 30):.2f} GiB block {t:.2f} ms, freed; reserved {attr(RESERVED_CURRENT) / (1 << 30):.2f} GiB")
run("reserved")
run("again")
cu.cuMemPoolTrimTo(pool, 0)
sync()
