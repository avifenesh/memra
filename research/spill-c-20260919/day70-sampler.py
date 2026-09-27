#!/usr/bin/env python3
"""Day 70 sampler for cell `sched` (research/spill-c-20260919/DAY70.md section 1): log only, one process, started and
stopped by the cell, pinned by it to a CPU outside the run's pin and its hardware-thread siblings.

Every 250 ms it writes every CPU's `/proc/stat` line (`C` rows) and, for each `run-gen` process, its main thread's
`schedstat` and last CPU (`O` rows); every 1 s every host thread whose user or system time moved since the last pass
(`T` rows: pid, tid, name, last CPU, the move in clock ticks). Rows are tab-separated with a UTC time first.

usage: day70-sampler.py <out.tsv>
"""
import os
import sys
import time
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def stat_fields(text):
    """`/proc/.../stat`: (name, fields after the name); the name may hold spaces and parentheses."""
    left, right = text.find("("), text.rfind(")")
    return text[left + 1:right], text[right + 2:].split()


def run_gens():
    out = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            comm = read(f"/proc/{pid}/comm")
            if comm and comm.startswith("run-gen"):
                out.append((pid, comm.strip()))
    return out


def main():
    out = open(sys.argv[1], "a", buffering=1)
    last = {}
    next_threads = 0.0
    while True:
        now = time.monotonic()
        t = stamp()
        rows = []
        text = read("/proc/stat") or ""
        for line in text.splitlines():
            if line.startswith("cpu") and line[3:4].isdigit():
                rows.append(f"{t}\tC\t{line}")
        for pid, comm in run_gens():
            sched = read(f"/proc/{pid}/task/{pid}/schedstat")
            st = read(f"/proc/{pid}/task/{pid}/stat")
            if sched and st:
                _, fields = stat_fields(st)
                rows.append(f"{t}\tO\t{pid}\t{comm}\t{fields[36]}\t{sched.strip()}")
        if now >= next_threads:
            next_threads = now + 1.0
            seen = {}
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    tids = os.listdir(f"/proc/{pid}/task")
                except OSError:
                    continue
                for tid in tids:
                    st = read(f"/proc/{pid}/task/{tid}/stat")
                    if not st:
                        continue
                    name, fields = stat_fields(st)
                    ticks = int(fields[11]) + int(fields[12])
                    key = (pid, tid)
                    seen[key] = ticks
                    moved = ticks - last.get(key, ticks)
                    if moved > 0:
                        rows.append(f"{t}\tT\t{pid}\t{tid}\t{name}\t{fields[36]}\t{moved}")
            last = seen
        out.write("\n".join(rows) + "\n")
        time.sleep(max(0.0, 0.25 - (time.monotonic() - now)))


if __name__ == "__main__":
    main()
