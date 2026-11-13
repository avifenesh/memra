#!/usr/bin/env python3
"""memra#1258: per served cold prime in a diag-build serve.log run with MEMRA_ALLOC_TRACE=1, the first
prime chunk's allocation sequence ([alloc-trace] lines between the first dspark-prime-chunk-before
and the next chunk-after or step OOM), its pool growth from the [diag1255] snapshots, and where its
sequence first differs from the first prime's.
usage: cmp-chunk1.py serve.log"""
import re
import sys

RE_PASS = re.compile(r"gate-pass id=(\S+) prompt=(\d+) path=(\w+)")
RE_SNAP = re.compile(r"\[diag1255\] (dspark-prime-chunk-before|dspark-prime-chunk-after|dspark-prime-done) id=(\S+): "
                     r"driver_free=(\d+)MB pool_res=(\d+)MB pool_used=(\d+)MB pool_unused=(\d+)MB")
RE_AT = re.compile(r"^\[alloc-trace\] (\d+) bytes from (\S+)")

primes = []
cur = None
state = None
for line in open(sys.argv[1], errors="replace"):
    if (m := RE_PASS.search(line)) and int(m.group(2)) > 1000:
        cur = {"id": m.group(1), "prompt": int(m.group(2)), "path": m.group(3), "seq": [], "before": None,
               "after": None, "oom": False}
        primes.append(cur)
        state = "admitted"
        continue
    if cur is None:
        continue
    if (m := RE_SNAP.search(line)) and m.group(2) == cur["id"]:
        snap = tuple(int(m.group(i)) for i in range(3, 7))
        if m.group(1) == "dspark-prime-chunk-before" and state == "admitted":
            cur["before"] = snap
            state = "chunk1"
        elif state == "chunk1":
            cur["after"] = snap
            state = "done"
        continue
    if state == "chunk1":
        if (m := RE_AT.match(line)):
            cur["seq"].append((int(m.group(1)), m.group(2)))
        elif "out of memory" in line:
            cur["oom"] = True
            state = "done"

ref = primes[0]["seq"] if primes else []
print("| prompt | path | allocs | bytes allocated MiB | pool res before -> after MiB | growth MiB | first diff vs first prime | OOM |")
print("|---|---|---|---|---|---|---|---|")
for p in primes:
    seq = p["seq"]
    diff = next((i for i, (a, b) in enumerate(zip(seq, ref)) if a != b), None)
    if diff is None and len(seq) != len(ref):
        diff = min(len(seq), len(ref))
    where = "-" if diff is None else f"#{diff}: {seq[diff] if diff < len(seq) else None} vs {ref[diff] if diff < len(ref) else None}"
    b, a = p["before"], p["after"]
    growth = (a[1] - b[1]) if (a and b) else None
    print(f"| {p['prompt']} | {p['path']} | {len(seq)} | {sum(x for x, _ in seq) >> 20} | "
          f"{b[1] if b else None} -> {a[1] if a else None} | {growth} | {where} | {p['oom']} |")
