#!/usr/bin/env python3
"""probe_alloc.py OUT.jsonl LABEL [SIZES_MIB] [REPS]: memra#1307's instrument check on a GB10 host,
under the caller's lease. Brackets a device allocation of each size with the two free-memory
readings a delta could use and prints how far each moved:

- raw: `cuMemGetInfo` free, what main's draft-state bracket reads at both ends;
- MemAvailable (`/proc/meminfo`), which the allocatable reading takes on unified memory;
- MemFree + Cached from the same read (FreePlusCached), the delta reading memra#1307 adopts;
- MemFree, Cached, Buffers, SReclaimable and Shmem, for context.

Each sample: read, `cuMemAlloc` SIZE, `cuMemsetD8` over it and synchronize (so every page is
backed), read again, free. A delta that tracks the allocation reads SIZE. Standard library and
libcuda only; allocates nothing on the host beyond the process itself and changes no setting.
SIZES_MIB defaults to 34,256 (the draft state Qwen3.8-27B parks per session, and one prime chunk's
block class), REPS to 8."""
import ctypes
import json
import sys
import time

out, label = sys.argv[1], sys.argv[2]
sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "34,256").split(",")]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 8
cu = ctypes.CDLL("libcuda.so.1")


def ck(r, what):
    if r != 0:
        sys.exit(f"{what} failed: CUresult {r}")


ck(cu.cuInit(0), "cuInit")
dev = ctypes.c_int()
ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
ctx = ctypes.c_void_p()
ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
ck(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
integrated = ctypes.c_int()
ck(cu.cuDeviceGetAttribute(ctypes.byref(integrated), 18, dev), "integrated attr")  # CU_DEVICE_ATTRIBUTE_INTEGRATED


def raw_free():
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    ck(cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    return free.value


def meminfo():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "Cached", "Buffers", "SReclaimable", "Shmem"):
                m[k] = int(v.split()[0]) * 1024
    # The candidate delta reading memra#1307 adopts: free memory plus the page cache, one read.
    m["FreePlusCached"] = m["MemFree"] + m["Cached"]
    return m


def reading():
    r = raw_free()
    m = meminfo()
    return {"raw": r, **m}


MIB = 1 << 20
with open(out, "a") as f:
    for size in sizes:
        for rep in range(reps):
            ptr = ctypes.c_uint64()
            a = reading()
            ck(cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size * MIB)), "cuMemAlloc")
            ck(cu.cuMemsetD8_v2(ptr, ctypes.c_ubyte(1), ctypes.c_size_t(size * MIB)), "cuMemsetD8")
            ck(cu.cuCtxSynchronize(), "cuCtxSynchronize")
            b = reading()
            ck(cu.cuMemFree_v2(ptr), "cuMemFree")
            row = {
                "label": label, "t": round(time.time(), 3), "integrated": integrated.value,
                "size_mib": size, "rep": rep,
                "before_mib": {k: round(v / MIB, 1) for k, v in a.items()},
                "after_mib": {k: round(v / MIB, 1) for k, v in b.items()},
                "delta_mib": {k: round((a[k] - b[k]) / MIB, 1) for k in a},
            }
            f.write(json.dumps(row) + "\n")
            print(f"{label} size={size} rep={rep} " + " ".join(
                f"{k}={row['delta_mib'][k]:+.1f}" for k in ("raw", "MemAvailable", "FreePlusCached", "MemFree", "Cached")))
            time.sleep(0.5)
