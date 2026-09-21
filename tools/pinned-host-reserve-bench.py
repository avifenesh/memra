#!/usr/bin/env python3
"""Pinned host arena reserve harness (memra#385): today's one cuMemHostAlloc against a candidate.

Measures exactly the driver call the arena makes at startup (`PinnedHostArena::reserve`,
crates/memra-engine/src/pinned_host.rs: cudarc `malloc_host` = cuMemHostAlloc with
CU_MEMHOSTALLOC_PORTABLE, flag bits 1: PORTABLE plus the cached pinned kind, no write-combined bit)
through ctypes on libcuda, so no engine binary and no engine change is needed to compare reserve
mechanisms. Arms (`--arms A,B`; A is always compared against B, the candidate):

  single   one cuMemHostAlloc(bytes, PORTABLE)                                 (today's arena)
  chunked  N threads, each cuMemHostAlloc(bytes / N, PORTABLE), joined         (candidate 1 of #385)
  thp      mmap(MAP_PRIVATE|MAP_ANONYMOUS) aligned to 2 MiB, madvise(MADV_HUGEPAGE),
           cuMemHostRegister(PORTABLE); freed with cuMemHostUnregister + munmap (candidate 2 of #385,
           the THP form: the mapping's AnonHugePages is read from /proc/self/smaps after the register
           call and reported; the arm counts only if it is backed by huge pages)

Interleaved A/B pairs in both orders (AB x pairs, then BA x pairs) inside ONE process and one window;
every allocation is freed before the next so the box never holds two arenas. `--roundtrip` runs a
correctness pass per arm BEFORE the timed pairs (A then B, full size): host fill with a per-2 MiB-block
byte pattern, H2D into a device buffer, host wipe, D2H back, memcmp per block (mismatched blocks must
be 0), and the driver's flags read back with cuMemHostGetFlags (the write-combined bit must be absent;
the PORTABLE bit set for the cuMemHostAlloc arms). Its reserve time is reported, not pooled.

Sizing: `--bytes N`, or `--fraction F --basis free|available` (F of MemFree or MemAvailable at start),
or `--basis admissible` (MemAvailable minus the engine's 32 GiB startup margin,
crates/memra-server/src/worker/host_memory.rs, the largest arena `check_headroom` admits); rounded
down to 2 MiB. Prints one JSON row per measurement, a summary with per-arm medians, and when
`--cell` is given the pre-registered rule line (research/spill-a-20260919/DAY15.md): the candidate
wins on this card if every allocation succeeded, both roundtrips are byte exact, the write-combined
bit is absent everywhere, a `thp` candidate is at least `--hugepage-min-fraction` huge-page backed,
its reserve time is strictly below single's at every pair and in both orders' medians, and the pooled
medians' ratio single / candidate is at least `--floor`. A harness receipt: the numbers describe the
box that ran it and nothing else.

usage: pinned-host-reserve-bench.py [--bytes N | --fraction 0.75 --basis free|available | --basis admissible]
                                     [--arms single,chunked|single,thp] [--chunks 8] [--pairs-per-order 5]
                                     [--roundtrip] [--cell NAME] [--floor 1.10] [--device 0] [--out receipt.json]
Exit 0 = every allocation succeeded and was freed (a rule verdict of inconclusive or void is still exit 0:
the receipt carries it); 2 = a driver call failed (the code is printed).
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
CU_MEMHOSTALLOC_DEVICEMAP = 0x02
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
CU_MEMHOSTREGISTER_PORTABLE = 0x01
ALIGN = 2 << 20
MARGIN = 32 << 30  # host_memory::MARGIN
PROT_READ, PROT_WRITE = 0x1, 0x2
MAP_PRIVATE, MAP_ANONYMOUS = 0x02, 0x20
MADV_HUGEPAGE = 14
ARMS = ("single", "chunked", "thp")


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


def thp_mode():
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            return f.read().strip()
    except OSError as e:
        return f"unavailable: {e}"


def smaps_anon_huge(addr):
    """AnonHugePages (bytes) of the /proc/self/smaps mapping containing addr, or -1 if not found."""
    with open("/proc/self/smaps") as f:
        inside = False
        for line in f:
            if not line[0].isspace() and "-" in line.split(" ", 1)[0]:
                lo, hi = (int(x, 16) for x in line.split(" ", 1)[0].split("-"))
                inside = lo <= addr < hi
            elif inside and line.startswith("AnonHugePages:"):
                return int(line.split()[1]) * 1024
    return -1


class Cuda:
    def __init__(self, device):
        self.lib = lib = ctypes.CDLL("libcuda.so.1")
        self.libc = ctypes.CDLL(None, use_errno=True)
        lib.cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        lib.cuMemFreeHost.argtypes = [ctypes.c_void_p]
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuMemHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
        # The un-suffixed cuMemAlloc / cuMemFree / cuMemcpy* / cuMemHostRegister exports are the legacy
        # 32-bit-size entry points; the headers map the names to the _v2 symbols, and so does this.
        self.cuMemAlloc = lib.cuMemAlloc_v2
        self.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
        self.cuMemFree = lib.cuMemFree_v2
        self.cuMemFree.argtypes = [ctypes.c_ulonglong]
        self.cuMemcpyHtoD = lib.cuMemcpyHtoD_v2
        self.cuMemcpyHtoD.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]
        self.cuMemcpyDtoH = lib.cuMemcpyDtoH_v2
        self.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
        self.cuMemHostRegister = lib.cuMemHostRegister_v2
        self.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        lib.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        for fn in (lib.cuMemHostAlloc, lib.cuMemFreeHost, lib.cuCtxSetCurrent, lib.cuMemHostGetFlags,
                   self.cuMemAlloc, self.cuMemFree, self.cuMemcpyHtoD, self.cuMemcpyDtoH,
                   self.cuMemHostRegister, lib.cuMemHostUnregister):
            fn.restype = ctypes.c_int
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.restype = ctypes.c_int
        self.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.libc.madvise.restype = ctypes.c_int
        self.libc.memcmp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.libc.memcmp.restype = ctypes.c_int
        self.check(lib.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self.check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), dev), "cuDevicePrimaryCtxRetain")
        self.check(lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

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

    def host_flags(self, p):
        flags = ctypes.c_uint()
        rc = self.lib.cuMemHostGetFlags(ctypes.byref(flags), p)
        return rc, flags.value


class Region:
    """One arm's reserved region: base pointers and how to release them."""

    def __init__(self, arm, ptrs, release, ok, rc, nbytes):
        self.arm, self.ptrs, self.release, self.ok, self.rc, self.nbytes = arm, ptrs, release, ok, rc, nbytes


def reserve_single(cuda, nbytes):
    t0 = time.perf_counter()
    rc, p = cuda.alloc(nbytes)
    t1 = time.perf_counter()
    row = {"ok": rc == 0, "rc": rc, "alloc_ms": (t1 - t0) * 1e3}
    ptrs = [(p.value, nbytes)] if rc == 0 else []
    return row, Region("single", ptrs, lambda: [cuda.free(q) for q, _ in ptrs], rc == 0, rc, nbytes)


def reserve_chunked(cuda, nbytes, chunks):
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
    ptrs = [(r[1].value, per) for r in results if r[0] == 0]
    return row, Region("chunked", ptrs, lambda: [cuda.free(q) for q, _ in ptrs], row["ok"], rcs, per * chunks)


def reserve_thp(cuda, nbytes):
    length = nbytes + ALIGN
    t0 = time.perf_counter()
    base = cuda.libc.mmap(None, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    if base is None or base == ctypes.c_void_p(-1).value:
        err = ctypes.get_errno()
        return {"ok": False, "rc": f"mmap errno {err}", "alloc_ms": 0.0}, Region("thp", [], lambda: None, False, err, nbytes)
    aligned = (base + ALIGN - 1) // ALIGN * ALIGN
    t1 = time.perf_counter()
    madv = cuda.libc.madvise(aligned, nbytes, MADV_HUGEPAGE)
    madv_errno = ctypes.get_errno() if madv != 0 else 0
    t2 = time.perf_counter()
    rc = cuda.cuMemHostRegister(aligned, nbytes, CU_MEMHOSTREGISTER_PORTABLE)
    t3 = time.perf_counter()
    huge = smaps_anon_huge(aligned) if rc == 0 else -1
    row = {"ok": rc == 0 and madv == 0, "rc": rc, "alloc_ms": (t3 - t0) * 1e3,
           "mmap_ms": (t1 - t0) * 1e3, "madvise_ms": (t2 - t1) * 1e3, "madvise_rc": madv,
           "madvise_errno": madv_errno, "register_ms": (t3 - t2) * 1e3,
           "anon_huge_bytes": huge, "anon_huge_fraction": (huge / nbytes) if huge >= 0 else None}

    def release():
        out = []
        if rc == 0:
            out.append(cuda.lib.cuMemHostUnregister(aligned))
        out.append(cuda.libc.munmap(base, length))
        return out

    return row, Region("thp", [(aligned, nbytes)] if rc == 0 else [], release, row["ok"], rc, nbytes)


def measure(cuda, arm, nbytes, chunks):
    if arm == "single":
        row, region = reserve_single(cuda, nbytes)
    elif arm == "chunked":
        row, region = reserve_chunked(cuda, nbytes, chunks)
    else:
        row, region = reserve_thp(cuda, nbytes)
    t2 = time.perf_counter()
    frcs = region.release() or []
    t3 = time.perf_counter()
    row["free_ms"] = (t3 - t2) * 1e3
    row["free_rc"] = frcs
    row["ok"] = bool(row["ok"]) and all(r == 0 for r in frcs)
    return row


def roundtrip(cuda, arm, nbytes, chunks):
    """Correctness pass: fill, H2D, wipe, D2H, memcmp per 2 MiB block; driver flags read back."""
    if arm == "single":
        row, region = reserve_single(cuda, nbytes)
    elif arm == "chunked":
        row, region = reserve_chunked(cuda, nbytes, chunks)
    else:
        row, region = reserve_thp(cuda, nbytes)
    out = {"arm": arm, "reserve": row, "ok": region.ok}
    if not region.ok:
        region.release()
        return out
    flags = []
    for p, _ in region.ptrs:
        rc, f = cuda.host_flags(p)
        flags.append({"rc": rc, "flags": f})
    out["driver_flags"] = flags
    out["wc_bit_absent"] = all((f["rc"] == 0 and f["flags"] & CU_MEMHOSTALLOC_WRITECOMBINED == 0) or f["rc"] != 0 for f in flags)
    out["flags_read_ok"] = all(f["rc"] == 0 for f in flags)
    if arm in ("single", "chunked"):
        out["portable_bit_set"] = all(f["rc"] == 0 and f["flags"] & CU_MEMHOSTALLOC_PORTABLE for f in flags)
    total = sum(n for _, n in region.ptrs)
    dptr = ctypes.c_ulonglong()
    rc = cuda.cuMemAlloc(ctypes.byref(dptr), total)
    if rc != 0:
        out.update(ok=False, device_alloc_rc=rc)
        region.release()
        return out
    refs = {}

    def ref(val):
        if val not in refs:
            buf = ctypes.create_string_buffer(ALIGN)
            ctypes.memset(buf, val, ALIGN)
            refs[val] = buf
        return refs[val]

    t0 = time.perf_counter()
    block = 0
    for p, n in region.ptrs:
        for off in range(0, n, ALIGN):
            ctypes.memset(p + off, (7 * block + 13) & 0xFF, min(ALIGN, n - off))
            block += 1
    t1 = time.perf_counter()
    doff, h2d_rc = 0, []
    for p, n in region.ptrs:
        h2d_rc.append(cuda.cuMemcpyHtoD(dptr.value + doff, p, n))
        doff += n
    t2 = time.perf_counter()
    for p, n in region.ptrs:
        ctypes.memset(p, 0, n)
    t3 = time.perf_counter()
    doff, d2h_rc = 0, []
    for p, n in region.ptrs:
        d2h_rc.append(cuda.cuMemcpyDtoH(p, dptr.value + doff, n))
        doff += n
    t4 = time.perf_counter()
    block, mismatched = 0, 0
    for p, n in region.ptrs:
        for off in range(0, n, ALIGN):
            ln = min(ALIGN, n - off)
            if cuda.libc.memcmp(p + off, ref((7 * block + 13) & 0xFF), ln) != 0:
                mismatched += 1
            block += 1
    t5 = time.perf_counter()
    frc = cuda.cuMemFree(dptr.value)
    rel = region.release() or []
    out.update(blocks=block, mismatched_blocks=mismatched, byte_exact=(mismatched == 0 and all(r == 0 for r in h2d_rc + d2h_rc)),
               h2d_rc=h2d_rc, d2h_rc=d2h_rc, device_free_rc=frc, release_rc=rel,
               fill_ms=(t1 - t0) * 1e3, h2d_ms=(t2 - t1) * 1e3, wipe_ms=(t3 - t2) * 1e3,
               d2h_ms=(t4 - t3) * 1e3, compare_ms=(t5 - t4) * 1e3,
               h2d_gib_per_s=(total / (1 << 30)) / ((t2 - t1) or 1e-9), d2h_gib_per_s=(total / (1 << 30)) / ((t4 - t3) or 1e-9))
    out["ok"] = out["byte_exact"] and frc == 0 and all(r == 0 for r in rel)
    return out


def rule(receipt, floor, huge_min):
    """The pre-registered rule of DAY15.md, from the receipt's rows; returns (dict, line)."""
    a, b = receipt["arms"]
    rows = receipt["rows"]
    pairs = {}
    for r in rows:
        pairs.setdefault((r["order"], r["pair"]), {})[r["arm"]] = r
    complete = [pairs[k] for k in sorted(pairs) if a in pairs[k] and b in pairs[k] and pairs[k][a]["ok"] and pairs[k][b]["ok"]]
    n_pairs = len(complete)
    below = sum(1 for p in complete if p[b]["alloc_ms"] < p[a]["alloc_ms"])
    med = {}
    for order in ("AB", "BA"):
        for arm in (a, b):
            xs = [r["alloc_ms"] for r in rows if r["order"] == order and r["arm"] == arm and r["ok"]]
            med[(order, arm)] = statistics.median(xs) if xs else float("nan")
    medians_both = all(med[(o, b)] < med[(o, a)] for o in ("AB", "BA"))
    pooled = {arm: statistics.median([r["alloc_ms"] for r in rows if r["arm"] == arm and r["ok"]] or [float("nan")]) for arm in (a, b)}
    ratio = pooled[a] / pooled[b] if pooled[b] and pooled[b] == pooled[b] else float("nan")
    alloc_ok = all(r["ok"] for r in rows) and n_pairs == receipt["pairs_per_order"] * 2
    rt = {x["arm"]: x for x in receipt.get("roundtrips", [])}
    exact = {arm: bool(rt.get(arm, {}).get("byte_exact", False)) for arm in (a, b)}
    wc_absent = all(rt.get(arm, {}).get("wc_bit_absent", False) for arm in (a, b))
    flags_ok = all(rt.get(arm, {}).get("flags_read_ok", False) for arm in (a, b))
    portable_ok = all(rt.get(arm, {}).get("portable_bit_set", True) for arm in (a, b))
    huge_fracs = [r["anon_huge_fraction"] for r in rows if r["arm"] == "thp" and r.get("anon_huge_fraction") is not None]
    if "thp" in rt and rt["thp"].get("reserve", {}).get("anon_huge_fraction") is not None:
        huge_fracs.append(rt["thp"]["reserve"]["anon_huge_fraction"])
    huge_min_seen = min(huge_fracs) if huge_fracs else None
    huge_ok = (huge_min_seen is not None and huge_min_seen >= huge_min) if b == "thp" else None
    integrity = alloc_ok and exact[a] and exact[b] and wc_absent and flags_ok and portable_ok and (huge_ok is not False)
    speed = n_pairs > 0 and below == n_pairs and medians_both
    material = ratio == ratio and ratio >= floor
    verdict = "wins-on-this-card" if integrity and speed and material else ("inconclusive" if integrity else "void")
    if b == "thp" and huge_ok is False:
        verdict = "void (hugepage-requested-not-granted)"
    d = {"cell": receipt.get("cell"), "candidate": b, "bytes": receipt["bytes"], "n_per_order": receipt["pairs_per_order"],
         "pooled": n_pairs, "alloc_ok": alloc_ok, "roundtrip_exact": exact, "wc_bit_absent": wc_absent,
         "flags_read_ok": flags_ok, "portable_bit_set": portable_ok, "hugepage_min_fraction_seen": huge_min_seen,
         "hugepage_ok": huge_ok, "cand_below_single_pairs": f"{below}/{n_pairs}", "medians_both_orders": medians_both,
         "median_AB": {a: med[("AB", a)], b: med[("AB", b)]}, "median_BA": {a: med[("BA", a)], b: med[("BA", b)]},
         "pooled_median_ms": pooled, "ratio_single_over_candidate": ratio, "floor": floor, "materiality": material,
         "integrity": integrity, "speed": speed, "candidate_arm": verdict}
    hf = "na" if huge_min_seen is None else f"{huge_min_seen:.4f}"
    line = (f"ARENA-RESERVE rule cell={d['cell']} candidate={b} bytes={d['bytes']} n_per_order={d['n_per_order']} "
            f"pooled={n_pairs} alloc_ok={str(alloc_ok).lower()} roundtrip_exact_single={str(exact[a]).lower()} "
            f"roundtrip_exact_candidate={str(exact[b]).lower()} wc_bit_absent={str(wc_absent).lower()} "
            f"flags_read_ok={str(flags_ok).lower()} portable_bit_set={str(portable_ok).lower()} hugepage_fraction={hf} "
            f"hugepage_ok={'na' if huge_ok is None else str(huge_ok).lower()} cand_below_single_pairs={below}/{n_pairs} "
            f"medians_both_orders={str(medians_both).lower()} pooled_single_ms={pooled[a]:.3f} pooled_candidate_ms={pooled[b]:.3f} "
            f"ratio={ratio:.4f} floor={floor:.2f} materiality={str(material).lower()} candidate_arm={verdict}")
    return d, line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    size = ap.add_mutually_exclusive_group()
    size.add_argument("--bytes", type=int, help="arena size in bytes (rounded down to 2 MiB)")
    size.add_argument("--fraction", type=float, default=0.75, help="fraction of the --basis field at start (default 0.75; ignored for admissible)")
    ap.add_argument("--basis", choices=["free", "available", "admissible"], default="free",
                    help="MemFree (default: never evicts another tenant's page cache), MemAvailable, or MemAvailable minus the engine's 32 GiB margin")
    ap.add_argument("--arms", default="single,chunked", help="A,B with A=single and B in {chunked,thp}")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--pairs-per-order", type=int, default=5)
    ap.add_argument("--roundtrip", action="store_true", help="correctness pass per arm before the timed pairs")
    ap.add_argument("--cell", help="cell name; prints the pre-registered rule line (DAY15.md)")
    ap.add_argument("--floor", type=float, default=1.10, help="materiality floor on pooled median single / candidate")
    ap.add_argument("--hugepage-min-fraction", type=float, default=0.9)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", help="write the receipt JSON here as well as to stdout")
    args = ap.parse_args()
    arms = tuple(args.arms.split(","))
    if len(arms) != 2 or arms[0] != "single" or arms[1] not in ("chunked", "thp"):
        raise SystemExit("REFUSED: --arms must be single,chunked or single,thp")
    if args.chunks < 2 or args.pairs_per_order < 1:
        raise SystemExit("REFUSED: --chunks >= 2 and --pairs-per-order >= 1")
    avail0 = meminfo("MemAvailable")
    free0 = meminfo("MemFree")
    total = meminfo("MemTotal")
    if args.bytes:
        nbytes, basis = args.bytes, "bytes"
    elif args.basis == "admissible":
        nbytes, basis = avail0 - MARGIN, "admissible"
    else:
        nbytes, basis = int((free0 if args.basis == "free" else avail0) * args.fraction), args.basis
    nbytes = nbytes // ALIGN * ALIGN
    if nbytes <= 0 or nbytes >= avail0:
        raise SystemExit(f"REFUSED: {nbytes} bytes is not below MemAvailable {avail0}")
    cuda = Cuda(args.device)
    receipt = {
        "kind": "pinned-host-reserve-bench",
        "issue": "memra#385",
        "cell": args.cell,
        "arms": list(arms),
        "bytes": nbytes,
        "basis": basis,
        "fraction": args.fraction if basis in ("free", "available") else None,
        "margin_bytes": MARGIN if basis == "admissible" else None,
        "chunks": args.chunks if arms[1] == "chunked" else None,
        "pairs_per_order": args.pairs_per_order,
        "floor": args.floor,
        "hugepage_min_fraction": args.hugepage_min_fraction,
        "mem_total": total,
        "mem_available_at_start": avail0,
        "mem_free_at_start": free0,
        "anon_huge_at_start": meminfo("AnonHugePages"),
        "hugepages_total": meminfo("HugePages_Total") // 1024,
        "thp_mode": thp_mode(),
        "cpus": os.cpu_count(),
        "gpu_at_start": gpu_state(),
        "loadavg_at_start": loadavg(),
        "rows": [],
        "roundtrips": [],
    }
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("rows", "roundtrips")}), flush=True)
    failed = False
    if args.roundtrip:
        for arm in arms:
            rt = roundtrip(cuda, arm, nbytes, args.chunks)
            rt.update(mem_available_before=meminfo("MemAvailable"), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            receipt["roundtrips"].append(rt)
            print(json.dumps({"roundtrip": rt}), flush=True)
            failed |= not rt["ok"]
        if failed:
            print("REFUSED: a correctness pass failed; no timed pairs", flush=True)
    if not failed:
        for order in ("AB", "BA"):
            seq = list(arms) if order == "AB" else list(reversed(arms))
            for pair in range(args.pairs_per_order):
                for arm in seq:
                    avail = meminfo("MemAvailable")
                    free = meminfo("MemFree")
                    row = measure(cuda, arm, nbytes, args.chunks)
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
    receipt["anon_huge_at_end"] = meminfo("AnonHugePages")
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
            if arm == "thp":
                summary[arm]["register_ms_median"] = statistics.median(r["register_ms"] for r in ok)
                summary[arm]["anon_huge_fraction_min"] = min(r["anon_huge_fraction"] for r in ok if r["anon_huge_fraction"] is not None)
            for order in ("AB", "BA"):
                sub = [r["alloc_ms"] for r in ok if r["order"] == order]
                if sub:
                    summary[arm][f"alloc_ms_median_{order}"] = statistics.median(sub)
    receipt["summary"] = summary
    receipt["status"] = "failed" if failed else "executed-not-qualified"
    if args.cell:
        d, line = rule(receipt, args.floor, args.hugepage_min_fraction)
        receipt["rule"] = d
        receipt["rule_line"] = line
    print(json.dumps({"summary": summary, "status": receipt["status"]}, indent=1), flush=True)
    if args.cell:
        print(receipt["rule_line"], flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=1)
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
