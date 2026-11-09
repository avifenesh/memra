#!/usr/bin/env python3
"""Tokens per round of one speculative burst, by draft length (memra#1093).

Reads the `[spec-k]` and `[spec-acc]` lines of every server log under the given run directories.
A `[spec-acc] burst=A/D` line is one burst of a request: D draft tokens offered and A accepted,
so D / K rounds and D / K + A tokens. Only cold-long requests (prompt of 1,024+ tokens) and
bursts of at least 8 rounds count; a request's last burst is the remainder of its budget.

usage: burst_spread.py RUN_DIR [RUN_DIR ...]
"""
import glob
import re
import statistics
import sys

per_k = {}
for run in sys.argv[1:]:
    for log in sorted(glob.glob(run + "/server-*.log")):
        k, long_prompt = None, False
        for line in open(log, errors="replace"):
            m = re.search(r"\[spec-k\] model.* K=(\d+) source=\S+ prompt=(\d+)", line)
            if m:
                k, long_prompt = int(m.group(1)), int(m.group(2)) >= 1024
                continue
            m = re.search(r"\[spec-acc\] ctx=\d+ burst=(\d+)/(\d+)", line)
            if m and k and long_prompt:
                accepted, drafted = int(m.group(1)), int(m.group(2))
                rounds = drafted / k
                if rounds >= 8:
                    per_k.setdefault(k, []).append(1 + accepted / rounds)
print("| K | bursts | tokens per round, min to max | mean | standard deviation over mean |")
print("|---|---:|---|---:|---:|")
for k in sorted(per_k, reverse=True):
    v = per_k[k]
    print(f"| {k} | {len(v)} | {min(v):.2f} to {max(v):.2f} | {statistics.mean(v):.2f} | "
          f"{100 * statistics.pstdev(v) / statistics.mean(v):.1f}% |")
