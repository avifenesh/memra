#!/usr/bin/env python3
"""Where a tie between two cost-model slots can change the verdict (memra#1212).

The spec cost model prices (n, K) from its boot table and, when (n, K) has no realized slot of
its own, scales that prediction by the realized/predicted ratio of one other slot. When two
slots tie in rank (the common case: the same K at n-1 and at n+1), the choice between them only
matters if their ratios fall on opposite sides of the threshold

    t(n) = plain(n) / max_K spec(n, K)

below which the scaled speculative price loses to plain. This tool reads a calibration line
(`[spec-cost] model=... depth=2048 ...` from a boot log) and prints, for n = 1..8, the table's
plain and speculative predictions and t(n). The formulas are the server's
(`crates/memra-server/src/worker.rs`: `spec_round_ms`, `spec_plain_ms`,
`spec_expected_tokens`; `crates/memra-engine/src/spec_cost.rs`: `round_cost_at`).

usage: tie_thresholds.py LOGFILE [--accept 0.7,0.5,0.35] [--rung 3] [--model NAME]
The acceptance is per draft position (the server reads it from its recent window); the default
is the one the server's own cost-model tests use.
"""
import re
import sys

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--accept", default="0.7,0.5,0.35")
ap.add_argument("--rung", type=int, default=3)
ap.add_argument("--model")
a = ap.parse_args()
log, rung, want = a.log, a.rung, a.model
accept = [float(x) for x in a.accept.split(",")]

line = None
for l in open(log, errors="replace"):
    if "[spec-cost] model=" in l and "depth=" in l and "verify_ms" in l:
        if want is None or f'model="{want}"' in l:
            line = l
            break
if line is None:
    raise SystemExit("no calibration line")
model = re.search(r'model="([^"]+)"', line).group(1)
pooled = re.search(r"pooled=(\w+)", line).group(1) == "true"
plain_cap = int(re.search(r"plain_cap=(\d+)", line).group(1))
overhead = float(re.search(r"round_overhead=([0-9.]+)ms", line).group(1))
m = re.search(r"draft_step=([0-9.]+)ms", line)
draft_step = float(m.group(1)) if m else 0.0
m = re.search(r"pooled_overhead=Some\(\((\d+), ([0-9.]+)\)\)", line)
pooled_overhead = (int(m.group(1)), float(m.group(2))) if m else None
m = re.search(r"pooled_draft_step=Some\(([0-9.]+)\)", line)
pooled_draft_step = float(m.group(1)) if m else None
verify = [float(x) for x in re.search(r"verify_ms\[1\.\.16\]=\[([^\]]+)\]", line).group(1).split()]
plain = [float(x) for x in re.search(r"plain_ms\[1\.\.\]=\[([^\]]+)\]", line).group(1).split()]


def interp(solo, cohort, m, n):
    if n >= m:
        return cohort
    if n > 1:
        return solo + (cohort - solo) * (n - 1) / (m - 1)
    return solo


def round_cost_at(n, k):
    o = interp(overhead, pooled_overhead[1], pooled_overhead[0], n) if pooled_overhead else overhead
    if pooled_overhead and pooled_draft_step is not None:
        step = interp(draft_step, pooled_draft_step, pooled_overhead[0], n)
    else:
        step = draft_step
    return o + k * step


def spec_round_ms(n, k):
    rows = k + 1
    if pooled and rows <= 16:
        per_chunk = 16 // rows
        full, rem = divmod(n, per_chunk)
        v = full * verify[per_chunk * rows - 1] + (verify[rem * rows - 1] if rem else 0.0)
    else:
        v = n * verify[min(rows, 16) - 1]
    o = round_cost_at(n, k) if pooled else overhead + k * draft_step
    return v + n * o


def spec_plain_ms(n):
    cap = max(1, min(plain_cap, len(plain)))
    full, rem = divmod(n, cap)
    return full * plain[cap - 1] + (plain[rem - 1] if rem else 0.0)


def expected(k):
    return 1.0 + sum(accept[:k])


print(f"model {model}: pooled={pooled} plain_cap={plain_cap} acceptance per position {accept}")
print("| n | plain tok/s | spec tok/s K=1 / 2 / 3 | best K | t(n) = plain / best spec |")
print("|---:|---:|---|---:|---:|")
for n in range(1, 9):
    p = n / spec_plain_ms(n) * 1e3
    s = [n * expected(k) / spec_round_ms(n, k) * 1e3 for k in range(1, rung + 1)]
    best = max(range(len(s)), key=lambda i: s[i])
    print(f"| {n} | {p:.1f} | {' / '.join(f'{x:.1f}' for x in s)} | {best + 1} | {p / s[best]:.2f} |")
