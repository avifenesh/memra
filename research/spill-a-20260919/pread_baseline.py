#!/usr/bin/env python3
"""Linux N=1 positioned-read plumbing; invoke through tier-battery, not scored.

Only the owned fixture's cache is changed (fadvise), never global drop_caches.
O_DIRECT bypasses page cache; cold/warm denote measured residency BEFORE reading,
not cold/warm storage-controller caches. No runtime dependency or dispatch added.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import time

LABEL = "block-device ext4 (virtio; NVMe ancestry provider-claimed, not proven)"
TOTAL = 64 << 20
PATTERN = bytes(range(256)) * 4096


def residency(fd, size, libc):
    with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
        anchor = ctypes.c_char.from_buffer(mapping)
        vec = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        if libc.mincore(ctypes.addressof(anchor), size, vec):
            raise OSError(ctypes.get_errno(), "mincore")
        del anchor
        return sum(bool(x & 1) for x in vec), len(vec)


def physical_reads():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
    return int(fields['read_bytes'])


def run(path):
    if platform.system() != 'Linux' or not hasattr(os, 'O_DIRECT'):
        raise RuntimeError('Linux O_DIRECT required; no fallback')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong]
    libc.pread.restype = ctypes.c_ssize_t
    libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    libc.posix_memalign.restype = ctypes.c_int
    libc.free.argtypes = [ctypes.c_void_p]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    expected = hashlib.sha256(PATTERN * (TOTAL // len(PATTERN))).hexdigest()
    # Exclusive creation prevents truncating another lane's data.
    with path.open('xb') as fixture:
        for _ in range(TOTAL // len(PATTERN)):
            fixture.write(PATTERN)
        fixture.flush()
        os.fsync(fixture.fileno())
    try:
        for chunk in (1 << 20, 16 << 20):
            for direct in (False, True):
                for cache in ('cold', 'warm'):
                    buffered = os.open(path, os.O_RDONLY)
                    pointer = ctypes.c_void_p()
                    fd = None
                    try:
                        os.posix_fadvise(buffered, 0, TOTAL, os.POSIX_FADV_DONTNEED)
                        if cache == 'warm':
                            for offset in range(0, TOTAL, len(PATTERN)):
                                if len(os.pread(buffered, len(PATTERN), offset)) != len(PATTERN):
                                    raise RuntimeError('short cache preparation read')
                        resident, pages = residency(buffered, TOTAL, libc)
                        if resident != (0 if cache == 'cold' else pages):
                            raise RuntimeError(f'cache precondition failed: {cache} {resident}/{pages}')
                        error = libc.posix_memalign(ctypes.byref(pointer), 4096, chunk)
                        if error:
                            raise OSError(error, 'posix_memalign')
                        # Fault/initialize destination outside the read interval.
                        ctypes.memset(pointer, 0, chunk)
                        fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
                        read_ns = 0
                        physical_before = physical_reads()
                        wall_start = time.perf_counter_ns()
                        actual = hashlib.sha256()
                        for offset in range(0, TOTAL, chunk):
                            started = time.perf_counter_ns()
                            got = libc.pread(fd, pointer, chunk, offset)
                            read_ns += time.perf_counter_ns() - started
                            if got != chunk:
                                raise RuntimeError(f'pread short/error: got={got} errno={ctypes.get_errno()}')
                            actual.update(ctypes.string_at(pointer, got))
                        wall_ns = time.perf_counter_ns() - wall_start
                        physical_bytes = physical_reads() - physical_before
                        if actual.hexdigest() != expected:
                            raise RuntimeError('byte checksum mismatch')
                        print(json.dumps(dict(schema_version=1, operation='synchronous-libc-pread',
                            mode='O_DIRECT' if direct else 'buffered', cache_before=cache,
                            resident_pages_before=resident, total_pages=pages, chunk_bytes=chunk,
                            bytes=TOTAL, calls=TOTAL//chunk, N=1, read_ns=read_ns,
                            mib_per_second=(TOTAL/(1<<20))/(read_ns/1e9),
                            wall_with_hash_ns=wall_ns, process_read_bytes=physical_bytes,
                            expected_sha256=expected, actual_sha256=actual.hexdigest(),
                            byte_exact=True, storage_class=LABEL, qualification=False,
                            scope='development plumbing; not a median or spill speed')), flush=True)
                    finally:
                        if fd is not None:
                            os.close(fd)
                        libc.free(pointer)
                        os.close(buffered)
    finally:
        path.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    args = parser.parse_args()
    run(args.fixture)
