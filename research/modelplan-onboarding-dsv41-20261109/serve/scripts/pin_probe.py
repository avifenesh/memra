#!/usr/bin/env python3
"""Pinned expert store probe. A read-only file mmap cannot be page-locked on this driver
(READ_ONLY_HOST_REGISTER_SUPPORTED=0; PROT_READ + PORTABLE is EINVAL), so the store is anonymous memory
filled from the checkpoint mmap and then registered. Measures copy-in, register and H2D rates.
Usage: pin_probe.py SHARD_FILE GB"""
import ctypes, ctypes.util, os, sys, time

shard, gb = sys.argv[1], float(sys.argv[2])
cu = ctypes.CDLL("libcuda.so.1")
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.memcpy.restype = ctypes.c_void_p
libc.memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]


def ck(name, rc):
    if rc != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(rc, ctypes.byref(s))
        raise SystemExit(f"{name}: {rc} {s.value}")


ck("cuInit", cu.cuInit(0))
dev = ctypes.c_int()
ck("cuDeviceGet", cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = ctypes.c_void_p()
ck("retain", cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev))
ck("setcur", cu.cuCtxSetCurrent(ctx))

fd = os.open(shard, os.O_RDONLY)
size = os.fstat(fd).st_size
n = min(size, int(gb * (1 << 30))) & ~4095
src = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
dst = libc.mmap(None, n, 3, 0x22, -1, 0)  # PROT_READ|WRITE, MAP_PRIVATE|MAP_ANONYMOUS
t = time.time()
libc.memcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), n)
cp_s = time.time() - t
print(f"copy-in {n / 2**30:.2f} GiB (one thread, page cache -> anonymous) in {cp_s:.2f} s = {n / 2**30 / cp_s:.1f} GiB/s")
t = time.time()
ck("register", cu.cuMemHostRegister_v2(ctypes.c_void_p(dst), ctypes.c_size_t(n), 1))
reg_s = time.time() - t
print(f"register {n / 2**30:.2f} GiB in {reg_s:.2f} s = {n / 2**30 / reg_s:.1f} GiB/s")
dptr = ctypes.c_uint64()
chunk = 1 << 30
ck("alloc", cu.cuMemAlloc_v2(ctypes.byref(dptr), ctypes.c_size_t(chunk)))
st = ctypes.c_void_p()
ck("stream", cu.cuStreamCreate(ctypes.byref(st), 1))
ck("warm", cu.cuMemcpyHtoDAsync_v2(dptr, ctypes.c_void_p(dst), ctypes.c_size_t(chunk), st))
ck("sync", cu.cuStreamSynchronize(st))
t = time.time()
moved, off = 0, 0
while off + chunk <= n:
    ck("h2d", cu.cuMemcpyHtoDAsync_v2(dptr, ctypes.c_void_p(dst + off), ctypes.c_size_t(chunk), st))
    moved += chunk
    off += chunk
ck("sync", cu.cuStreamSynchronize(st))
s = time.time() - t
print(f"H2D pinned 1 GiB copies: {moved / 2**30:.0f} GiB in {s:.2f} s = {moved / 2**30 / s:.1f} GiB/s")
# one expert = six pieces (w1/w3 5,898,240 B, w2 5,898,240 B, scales 368,640 B each), back to back
pieces = [5898240, 368640] * 3
t = time.time()
off, cnt = 0, 0
for e in range(48):
    for pz in pieces:
        ck("h2d", cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dptr.value + (off % (chunk - 6000000))), ctypes.c_void_p(dst + off), ctypes.c_size_t(pz), st))
        off += pz
    cnt += 1
ck("sync", cu.cuStreamSynchronize(st))
s = time.time() - t
print(f"H2D {cnt} experts as six pieces each: {s * 1e3:.1f} ms = {off / 2**30 / s:.1f} GiB/s, {s * 1e3 / cnt:.2f} ms per expert")
t = time.time()
ck("h2d", cu.cuMemcpyHtoDAsync_v2(dptr, ctypes.c_void_p(src), ctypes.c_size_t(1 << 28), st))
ck("sync", cu.cuStreamSynchronize(st))
s = time.time() - t
print(f"H2D pageable 256 MiB straight from the file mmap: {s * 1e3:.1f} ms = {0.25 / s:.1f} GiB/s")
ck("unregister", cu.cuMemHostUnregister(ctypes.c_void_p(dst)))
cu.cuMemFree_v2(dptr)
print("ok")
