#!/usr/bin/env python3
"""Diagnostic only (never a gate): per-process storage I/O attribution on the 5090 rig while the
5090 queue runs. Every 5 s, read `/proc/<pid>/io` (read_bytes, write_bytes) for every readable pid
and log each process whose bytes moved, with its command name. Explains the foreign device bytes
that unscored the capped regime; `/proc` only, no storage traffic of its own.
Usage: m1-io-attribution.py --out FILE [--interval-s 5]
"""
import argparse
import json
import os
import time


def snapshot():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            io = dict(line.split(": ") for line in open(f"/proc/{pid}/io").read().splitlines())
            comm = open(f"/proc/{pid}/comm").read().strip()
        except (OSError, ValueError):
            continue
        out[pid] = (comm, int(io["read_bytes"]), int(io["write_bytes"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-s", type=float, default=5.0)
    a = ap.parse_args()
    prev = snapshot()
    with open(a.out, "a") as f:
        while True:
            time.sleep(a.interval_s)
            cur = snapshot()
            moved = []
            for pid, (comm, rb, wb) in cur.items():
                if pid in prev:
                    dr, dw = rb - prev[pid][1], wb - prev[pid][2]
                    if dr > 0 or dw > 0:
                        moved.append({"pid": int(pid), "comm": comm, "read": dr, "write": dw})
            moved.sort(key=lambda m: -(m["read"] + m["write"]))
            f.write(json.dumps({"unix": time.time(), "moved": moved[:20]}) + "\n")
            f.flush()
            prev = cur


if __name__ == "__main__":
    main()
