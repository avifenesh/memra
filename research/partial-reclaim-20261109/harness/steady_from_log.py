"""Steady-cell counts from serve.log alone, for an arm whose harness never wrote rows.json (cut mid-run): every
admission's `[spec-k]` line names its namespace and its cached tokens. A namespace's first admission is new (ns-f* in
the fill, ns-s* in the steady stream), a later one is a repeat; cached > 0 is a hit. The flushes or passes, OOM lines
and the requests the server aborted at the cut come from the same log.

  python3 steady_from_log.py ARM_DIR...
"""
import re, sys
from pathlib import Path

print("| arm | fill admissions | steady new | steady repeats, hits, cached tokens | reclaims (partial) | OOM lines | aborted with no output |")
print("|---|---|---|---|---|---|---|")
for d in map(Path, sys.argv[1:]):
    log = (d / "serve.log").read_text(errors="replace")
    seen, fill, new, rep, hits, cached = set(), 0, 0, 0, 0, 0
    for ns, c in re.findall(r'\[spec-k\] model="gate" tenant="(ns-[fs]\d+)" .*? cached=(\d+)', log):
        if ns in seen:
            rep += 1
            hits += int(c) > 0
            cached += int(c)
        else:
            seen.add(ns)
            fill += ns.startswith("ns-f")
            new += ns.startswith("ns-s")
    reclaims = len(re.findall(r"reclaim-on-defer: evicted", log))
    partial = len(re.findall(r"reclaim-on-defer \(partial\)", log))
    oom = len(re.findall(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY", log, re.I))
    aborted = len(re.findall(r"\[abort\] client disconnected: .*? 0 generated", log))
    print(f"| {d.name} | {fill} | {new} | {rep}, {hits}, {cached} | {reclaims} ({partial}) | {oom} | {aborted} |")
