#!/usr/bin/env python3
"""M1 host page-cache regimes, file-scoped (M1-PREREG.md section B; OWED 11).

  cold FILE...       fdatasync + POSIX_FADV_DONTNEED per file; mincore must then show 0 resident
                     pages (bounded retries), else FAIL naming the files still resident
  warm FILE...       one full buffered read per file; mincore must then show every page resident
  residency FILE...  report resident/total pages per file (no change)
  balloon --bytes N --floor-bytes F [--hold-s S]
                     mmap + mlock N anonymous bytes to bound the page cache; refuse when
                     RLIMIT_MEMLOCK is below N or MemAvailable - N would fall under F; while held,
                     release and exit 4 if MemAvailable drops under F; hold until SIGTERM or S

Page-cache states only: nothing here drops a global cache, touches a device, or needs
privilege. POSIX_FADV_DONTNEED cannot evict pages another process has mapped; cold then FAILS,
it does not pretend. Every run prints one M1-CACHE line per file (or balloon) and exits 0 only
when every file reached the requested state.
"""
import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import resource
import signal
import sys
import time

PAGE = mmap.PAGESIZE
libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PROT_READ, MAP_SHARED = 0x1, 0x01
MAP_FAILED = ctypes.c_void_p(-1).value


def residency(path):
    size = os.path.getsize(path)
    pages = (size + PAGE - 1) // PAGE
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, MAP_FAILED):
            e = ctypes.get_errno()
            raise OSError(e, f"mmap for mincore failed: {os.strerror(e)}")
        try:
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                e = ctypes.get_errno()
                raise OSError(e, f"mincore failed: {os.strerror(e)}")
            return sum(b & 1 for b in vec), pages
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def meminfo_kb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1])
    raise KeyError(key)


def report(**row):
    print("M1-CACHE " + json.dumps(row, sort_keys=True), flush=True)


def cold(files, retries=5):
    ok = True
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fdatasync(fd)
            resident = pages = None
            for attempt in range(1, retries + 1):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                resident, pages = residency(path)
                if resident == 0:
                    break
                time.sleep(0.05 * attempt)
        finally:
            os.close(fd)
        state = "PASS" if resident == 0 else "FAIL"
        ok &= state == "PASS"
        report(mode="cold", file=os.path.basename(path), resident_pages=resident, pages=pages,
               attempts=attempt, state=state,
               **({} if state == "PASS" else {"reason": "pages still resident after DONTNEED "
                                                        "(mapped by another process, or dirty)"}))
    return ok


def warm(files):
    ok = True
    buf = bytearray(16 << 20)
    for path in files:
        started = time.monotonic_ns()
        with open(path, "rb", buffering=0) as f:
            while f.readinto(buf):
                pass
        resident, pages = residency(path)
        state = "PASS" if resident == pages else "FAIL"
        ok &= state == "PASS"
        report(mode="warm", file=os.path.basename(path), resident_pages=resident, pages=pages,
               read_ms=round((time.monotonic_ns() - started) / 1e6, 3), state=state,
               **({} if state == "PASS" else {"reason": "page cache could not hold the file"}))
    return ok


def show(files):
    for path in files:
        resident, pages = residency(path)
        report(mode="residency", file=os.path.basename(path), resident_pages=resident, pages=pages,
               share=round(resident / pages, 6) if pages else None)
    return True


def balloon(nbytes, floor_bytes, hold_s):
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    available = meminfo_kb("MemAvailable") * 1024
    if soft != resource.RLIM_INFINITY and soft < nbytes:
        report(mode="balloon", state="REFUSED", bytes=nbytes, rlimit_memlock=soft,
               reason="RLIMIT_MEMLOCK below the balloon size")
        return 2
    if available - nbytes < floor_bytes:
        report(mode="balloon", state="REFUSED", bytes=nbytes, mem_available=available,
               floor_bytes=floor_bytes, reason="MemAvailable minus the balloon would fall under the floor")
        return 2
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    anchor = ctypes.c_char.from_buffer(region)  # export: must be dropped before close()
    addr = ctypes.addressof(anchor)
    if libc.mlock(addr, nbytes) != 0:
        e = ctypes.get_errno()
        report(mode="balloon", state="REFUSED", bytes=nbytes, errno=e, reason=f"mlock: {os.strerror(e)}")
        del anchor
        region.close()
        return 2
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    report(mode="balloon", state="LOCKED", bytes=nbytes, mem_available_after=meminfo_kb("MemAvailable") * 1024,
           floor_bytes=floor_bytes)
    code = 0
    end = time.monotonic() + hold_s if hold_s else None
    while not stop["flag"] and (end is None or time.monotonic() < end):
        if meminfo_kb("MemAvailable") * 1024 < floor_bytes:
            report(mode="balloon", state="RELEASED-FLOOR", bytes=nbytes,
                   mem_available=meminfo_kb("MemAvailable") * 1024, floor_bytes=floor_bytes)
            code = 4
            break
        time.sleep(0.25)
    libc.munlock(addr, nbytes)
    del anchor
    region.close()
    if code == 0:
        report(mode="balloon", state="RELEASED", bytes=nbytes)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("cold", "warm", "residency"):
        sub.add_parser(name).add_argument("files", nargs="+")
    b = sub.add_parser("balloon")
    b.add_argument("--bytes", type=int, required=True)
    b.add_argument("--floor-bytes", type=int, required=True)
    b.add_argument("--hold-s", type=float)
    args = ap.parse_args(argv)
    if args.cmd == "balloon":
        return balloon(args.bytes, args.floor_bytes, args.hold_s)
    fn = {"cold": cold, "warm": warm, "residency": show}[args.cmd]
    return 0 if fn(args.files) else 3


if __name__ == "__main__":
    sys.exit(main())
