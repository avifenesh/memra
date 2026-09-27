#!/usr/bin/env python3
"""Day 74 inducer for cell `induce` (research/spill-c-20260919/DAY74.md sections 1 and 1a): fragments host memory,
then, from the run's gate (SIGUSR1 from the cell), asks for huge pages in a loop so the kernel compacts, and triggers
compaction through /proc/sys/vm/compact_memory when that file is writable, until it is stopped (SIGTERM from the
cell, by its pid). Everything it holds is released when it exits.

Setup: map F GiB anonymous with MADV_NOHUGEPAGE, touch every 4 KiB page, return every other page with MADV_DONTNEED,
write `ready`. After SIGUSR1: write `loop trigger=<writable|none>` and loop: (when writable) write 1 to
compact_memory; map 256 MiB with MADV_HUGEPAGE, touch every 4 KiB page, read the process's AnonHugePages while it is
mapped, unmap. One row per second: `t\tmaps\tanon_huge_kb_last\tcpu_s\thalf_held_kb\ttriggers`.

usage: day74-compactor.py <out.tsv> <F GiB>
"""
import mmap
import resource
import signal
import sys
import time
from datetime import datetime, timezone

PAGE = 4096
BURST = 256 << 20


def stamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def anon_huge_kb():
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if line.startswith("AnonHugePages:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main():
    out = open(sys.argv[1], "a", buffering=1)
    size = int(float(sys.argv[2]) * (1 << 30)) // PAGE * PAGE
    stop, go = [], []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGUSR1, lambda *_: go.append(1))
    t0 = time.monotonic()
    frag = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    frag.madvise(mmap.MADV_NOHUGEPAGE)
    for off in range(0, size, PAGE):
        frag[off] = 1
    for off in range(0, size, 2 * PAGE):
        frag.madvise(mmap.MADV_DONTNEED, off, PAGE)
    out.write(f"{stamp()}\tready\tfragmented_gib={size / (1 << 30):.1f}\tsetup_s={time.monotonic() - t0:.1f}"
              f"\tcpu_s={cpu_s():.1f}\n")
    while not go and not stop:
        time.sleep(0.01)
    try:
        trigger = open("/proc/sys/vm/compact_memory", "w", buffering=1)
    except OSError:
        trigger = None
    out.write(f"{stamp()}\tloop\ttrigger={'writable' if trigger else 'none'}\n")
    maps, triggers, last_huge, next_row = 0, 0, -1, time.monotonic() + 1.0
    while not stop:
        if trigger is not None:
            try:
                trigger.write("1\n")
                triggers += 1
            except OSError:
                trigger = None
        burst = mmap.mmap(-1, BURST, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        burst.madvise(mmap.MADV_HUGEPAGE)
        for off in range(0, BURST, PAGE):
            burst[off] = 1
        last_huge = anon_huge_kb()
        burst.close()
        maps += 1
        if time.monotonic() >= next_row:
            next_row += 1.0
            out.write(f"{stamp()}\t{maps}\t{last_huge}\t{cpu_s():.1f}\t{size // 2 // 1024}\t{triggers}\n")
    out.write(f"{stamp()}\tstopped\tmaps={maps}\ttriggers={triggers}\tcpu_s={cpu_s():.1f}\n")
    frag.close()


if __name__ == "__main__":
    main()
