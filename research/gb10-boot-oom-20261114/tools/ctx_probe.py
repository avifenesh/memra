#!/usr/bin/env python3
"""ctx_probe.py: memra#1316's driver-level boot probe for a GB10 host, under the caller's lease.

    ctx_probe.py one [RETRY_MAX]          one boot in this process, one JSON line on stdout
    ctx_probe.py loop OUT.jsonl N GAP_MS [RETRY_MAX]
                                          N boots, each in a fresh process, GAP_MS apart
    ctx_probe.py armed MB WAIT_S [RETRY_MAX]
                                          one boot that waits (at most WAIT_S) until MemFree is
                                          under MB before it creates the context, and samples
                                          MemFree every 5 ms through the call
    ctx_probe.py churn OUT.jsonl SECONDS [RETRY_MAX]
                                          one process that creates and destroys a context over
                                          and over for SECONDS, one record per context
    ctx_probe.py hold GIB SECONDS         hold GIB of backed device memory for SECONDS, then exit
                                          cleanly (stands in for a server that exits)

A boot walks the driver calls `Engine::new` makes before it loads a kernel module, in the same
order: cuInit, cuDeviceGet, cuDevicePrimaryCtxRetain (cudarc's `CudaContext::new`), cuCtxSetCurrent,
two cuStreamCreate, then one small device allocation. Each step's CUresult and duration are
recorded, with MemFree/MemAvailable/Cached and the Normal zone's free-block counts read right
before the context is created. `cuMemGetInfo` needs a context, so it is read right after the
context exists (first try or a later one).

When a step returns CUDA_ERROR_OUT_OF_MEMORY and RETRY_MAX > 0, the same call is repeated in the
same process, 50 ms apart, up to RETRY_MAX times; the record keeps how many tries and how long it
took to succeed. The first try is the boot's verdict; the retries measure how long the state lasts.

Standard library and libcuda only. Reads /proc; changes no setting."""
import ctypes
import json
import subprocess
import sys
import threading
import time

OOM = 2
RETRY_STEP_S = 0.05


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "Cached", "Buffers", "SReclaimable"):
                out[k + "_mb"] = int(v.split()[0]) // 1024
    return out


def buddy_normal():
    with open("/proc/buddyinfo") as f:
        for line in f:
            p = line.split()
            if p[3] == "Normal":
                return [int(x) for x in p[4:]]
    return []


def mem_free_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) // 1024
    return -1


def one(retry_max, arm_mb=None, arm_wait_s=0.0):
    rec = {"t": round(time.time(), 3), "steps": [], "ok": True}
    trace = []
    tracing = threading.Event()
    cu = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    s1, s2 = ctypes.c_void_p(), ctypes.c_void_p()
    ptr = ctypes.c_ulonglong()
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    integ = ctypes.c_int()

    def step(name, fn):
        """Run fn; on OOM retry up to retry_max. False stops the boot."""
        t0 = time.monotonic()
        rc = fn()
        s = {"call": name, "rc": rc, "ms": round((time.monotonic() - t0) * 1e3, 2)}
        if rc == OOM:
            s["mem_at_fail"] = meminfo()
            s["buddy_at_fail"] = buddy_normal()
            tries = 0
            while rc == OOM and tries < retry_max:
                time.sleep(RETRY_STEP_S)
                tries += 1
                rc = fn()
            s["retries"] = tries
            s["retry_rc"] = rc
            s["retry_ms"] = round((time.monotonic() - t0) * 1e3, 2)
            if rc == 0:
                s["mem_at_ok"] = meminfo()
        rec["steps"].append(s)
        if s["rc"] != 0:
            rec["ok"] = False
            rec.setdefault("first_fail", name)
        return rc == 0

    def after_ctx():
        cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        rec["cu_free_mb"] = free.value >> 20
        rec["cu_total_mb"] = total.value >> 20
        return 0

    seq = [
        ("cuInit", lambda: cu.cuInit(0)),
        ("cuDeviceGet", lambda: cu.cuDeviceGet(ctypes.byref(dev), 0)),
        # CU_DEVICE_ATTRIBUTE_INTEGRATED = 18: readable before any context exists.
        ("cuDeviceGetAttribute", lambda: cu.cuDeviceGetAttribute(ctypes.byref(integ), 18, dev)),
        ("cuDevicePrimaryCtxRetain", lambda: cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)),
        ("cuCtxSetCurrent", lambda: cu.cuCtxSetCurrent(ctx)),
        ("cuMemGetInfo", after_ctx),
        ("cuStreamCreate#1", lambda: cu.cuStreamCreate(ctypes.byref(s1), 1)),
        ("cuStreamCreate#2", lambda: cu.cuStreamCreate(ctypes.byref(s2), 1)),
        ("cuMemAlloc 64 MiB", lambda: cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(64 << 20))),
        ("cuMemsetD8", lambda: cu.cuMemsetD8_v2(ptr, ctypes.c_ubyte(0), ctypes.c_size_t(64 << 20))),
        ("cuCtxSynchronize", lambda: cu.cuCtxSynchronize()),
    ]
    for name, fn in seq:
        if name == "cuDevicePrimaryCtxRetain":
            if arm_mb is not None:
                # Armed: hold here until MemFree falls under the threshold, then create the
                # context at once, with MemFree sampled every 5 ms through the call.
                t_arm = time.monotonic()
                fired = "timeout"
                while time.monotonic() - t_arm < arm_wait_s:
                    if mem_free_mb() < arm_mb:
                        fired = "threshold"
                        break
                    time.sleep(0.002)
                rec["armed"] = {"mb": arm_mb, "waited_ms": round((time.monotonic() - t_arm) * 1e3, 1),
                                "fired": fired}
                rec["t"] = round(time.time(), 3)
                t_fire = time.monotonic()

                def sample():
                    while not tracing.is_set():
                        trace.append((round((time.monotonic() - t_fire) * 1e3), mem_free_mb()))
                        time.sleep(0.005)

                threading.Thread(target=sample, daemon=True).start()
            rec["mem_before_ctx"] = meminfo()
            rec["buddy_before_ctx"] = buddy_normal()
        if not step(name, fn):
            break
    tracing.set()
    if trace:
        rec["free_trace_ms_mb"] = trace[:120]
        rec["free_min_mb"] = min(v for _, v in trace)
    rec["integrated"] = integ.value
    if ptr.value:
        cu.cuMemFree_v2(ptr)
    if ctx.value:
        cu.cuDevicePrimaryCtxRelease(dev)
    print(json.dumps(rec), flush=True)


def loop(out, n, gap_ms, retry_max):
    fails = 0
    with open(out, "a") as f:
        for i in range(n):
            p = subprocess.run(
                [sys.executable, sys.argv[0], "one", str(retry_max)],
                capture_output=True, text=True, timeout=120,
            )
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            try:
                rec = json.loads(line)
            except ValueError:
                rec = {"ok": False, "first_fail": "probe died", "stderr": p.stderr[-400:]}
            rec["boot"] = i
            rec["exit"] = p.returncode
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if not rec["ok"]:
                fails += 1
                print(f"boot {i}: FAIL at {rec.get('first_fail')}", flush=True)
            time.sleep(gap_ms / 1e3)
    print(f"loop done: boots={n} failed={fails}", flush=True)


def churn(out, seconds, retry_max):
    """Create and destroy one context after another in this process for `seconds`: each
    cuCtxCreate makes the driver allocate a channel's context buffers again, so every sample asks
    the kernel for the same blocks a boot needs, about five times a second."""
    cu = ctypes.CDLL("libcuda.so.1")
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    for name, rc in (("cuInit", cu.cuInit(0)), ("cuDeviceGet", cu.cuDeviceGet(ctypes.byref(dev), 0))):
        if rc:
            sys.exit(f"churn: {name} CUresult {rc}")
    if hasattr(cu, "cuCtxCreate_v2"):
        create = lambda: cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)  # noqa: E731
    else:
        create = lambda: cu.cuCtxCreate_v4(ctypes.byref(ctx), None, 0, dev)  # noqa: E731
    end = time.monotonic() + seconds
    n = fails = 0
    with open(out, "a") as f:
        while time.monotonic() < end:
            row = {"t": round(time.time(), 3), "mem": meminfo(), "buddy": buddy_normal()}
            t0 = time.monotonic()
            rc = create()
            row["rc"] = rc
            row["ms"] = round((time.monotonic() - t0) * 1e3, 1)
            if rc == OOM:
                fails += 1
                row["mem_at_fail"] = meminfo()
                row["buddy_at_fail"] = buddy_normal()
                tries = 0
                while rc == OOM and tries < retry_max:
                    time.sleep(RETRY_STEP_S)
                    tries += 1
                    rc = create()
                row["retries"] = tries
                row["retry_rc"] = rc
                row["retry_ms"] = round((time.monotonic() - t0) * 1e3, 1)
            if rc == 0:
                # While the context lives: what its creation took from each block order.
                row["buddy_after"] = buddy_normal()
                cu.cuCtxDestroy_v2(ctx)
            elif rc != OOM:
                row["stopped"] = True
            n += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("stopped"):
                break
    print(f"churn done: contexts={n} first-try failures={fails}", flush=True)


def hold(gib, seconds):
    cu = ctypes.CDLL("libcuda.so.1")
    dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_ulonglong()
    for name, rc in (
        ("cuInit", cu.cuInit(0)),
        ("cuDeviceGet", cu.cuDeviceGet(ctypes.byref(dev), 0)),
        ("cuDevicePrimaryCtxRetain", cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)),
        ("cuCtxSetCurrent", cu.cuCtxSetCurrent(ctx)),
    ):
        if rc:
            sys.exit(f"hold: {name} CUresult {rc}")
    held = []
    for _ in range(gib):
        rc = cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(1 << 30))
        if rc:
            sys.exit(f"hold: cuMemAlloc CUresult {rc} after {len(held)} GiB")
        cu.cuMemsetD8_v2(ctypes.c_ulonglong(ptr.value), ctypes.c_ubyte(1), ctypes.c_size_t(1 << 30))
        held.append(ptr.value)
    cu.cuCtxSynchronize()
    print(f"hold: {gib} GiB backed, {json.dumps(meminfo())}", flush=True)
    time.sleep(seconds)
    for p in held:
        cu.cuMemFree_v2(ctypes.c_ulonglong(p))
    cu.cuDevicePrimaryCtxRelease(dev)
    print(f"hold: released {round(time.time(), 3)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "one":
        one(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    elif mode == "armed":
        one(int(sys.argv[4]) if len(sys.argv) > 4 else 0, int(sys.argv[2]), float(sys.argv[3]))
    elif mode == "loop":
        loop(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
             int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    elif mode == "churn":
        churn(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif mode == "hold":
        hold(int(sys.argv[2]), float(sys.argv[3]))
    else:
        sys.exit(__doc__)
