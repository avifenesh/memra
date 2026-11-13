#!/usr/bin/env python3
"""Per-run figures of a dsv41-tpbench A/B (perf_cfix_four.sh's runs/runN-ARM/cells.jsonl), the
token-tape hashes of its decode cells, and the cards' telemetry during each run.

A call's steady figure is steady.py's: calls that start at least 0.5 s after the run's first call,
a run's first call of each kind left out (it builds the lazy state). steady.py pools every run of
an arm; this prints each run's median, the arm's median of run medians, and each adjacent A/B pair
of the interleaved order (A B B A gives the pairs run1/run2 and run3/run4) as B / A.

Telemetry (lease.telemetry.csv, 250 ms nvidia-smi lines) is cut to each run's span from its
cells' unix times: per card, the median SM clock and power, the highest temperature and the share
of lines whose clock-event reasons carry a thermal slowdown bit (0x20 sw thermal, 0x40 hw
thermal, 0x08 hw slowdown).

usage: per_run.py DIR [--steady S] [--cards 0,1]   (DIR holds runs/ and lease.telemetry.csv; either file may be
       gzipped, as the receipts here are)
"""
import datetime
import glob
import gzip
import json
import os
import statistics
import sys

d = sys.argv[1]
steady = float(sys.argv[sys.argv.index("--steady") + 1]) if "--steady" in sys.argv else 0.5
cards = sys.argv[sys.argv.index("--cards") + 1].split(",") if "--cards" in sys.argv else None
THERMAL = 0x20 | 0x40 | 0x08


def lines(path):
    """The text lines of PATH or of PATH.gz, whichever exists; none when neither does."""
    if os.path.exists(path):
        return open(path)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    return []


def calls_of(path):
    out, hashes, span = [], {}, [None, None]
    for line in lines(path):
        if not line.strip():
            continue
        c = json.loads(line)
        span[0] = c["start_unix"] if span[0] is None else min(span[0], c["start_unix"])
        span[1] = c["end_unix"] if span[1] is None else max(span[1], c["end_unix"])
        if c.get("kind") == "prefill":
            out.append((c["start_unix"], f"{c['prompt_len']} rows", c["ttft_s"] * 1e3))
        elif c.get("kind") == "decode":
            t = c["start_unix"] + c.get("ttft_s", 0.0)
            for s in c.get("step_s", []):
                out.append((t, f"decode c{c.get('c', 1)}", s * 1e3))
                t += s
            if c.get("gen_sha256"):
                hashes.setdefault(c["cell"], set()).add(tuple(c["gen_sha256"]))
    return sorted(out), hashes, span


def run_key(r):
    return int(os.path.basename(r)[3:].split("-")[0])


def kind_key(k):
    head = k.split()[0]
    return (0, int(head)) if head.isdigit() else (1, 0)


runs = sorted((r for r in glob.glob(os.path.join(d, "runs", "run*")) if os.path.isdir(r)), key=run_key)
per = {}  # (run, kind) -> steady median
spans = {}
toks = {}
for r in runs:
    name = os.path.basename(r)
    calls, hashes, span = calls_of(os.path.join(r, "cells.jsonl"))
    spans[name] = span
    for cell, hs in hashes.items():
        for h in hs:
            toks.setdefault(cell, {}).setdefault(h, []).append(name)
    if not calls:
        continue
    t0, seen, by = calls[0][0], set(), {}
    for t, kind, ms in calls:
        if kind not in seen:
            seen.add(kind)
            continue
        if t - t0 >= steady:
            by.setdefault(kind, []).append(ms)
    for kind, v in by.items():
        per[(name, kind)] = (statistics.median(v), len(v))

names = [os.path.basename(r) for r in runs]
kinds = sorted({k for (_, k) in per}, key=kind_key)
print(f"## {os.path.basename(os.path.abspath(d))}: steady medians a run, ms (calls)\n")
print("| call | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for k in kinds:
    cells = [f"{per[(n, k)][0]:.3f} ({per[(n, k)][1]})" if (n, k) in per else "-" for n in names]
    print(f"| {k} | " + " | ".join(cells) + " |")

print("\nArms (median of the run medians) and each adjacent A/B pair as B / A:\n")
pairs = [(names[i], names[i + 1]) for i in range(0, len(names) - 1, 2)]
print("| call | A | B | B / A | " + " | ".join(f"{a} + {b}" for a, b in pairs) + " |")
print("|---|---|---|---|" + "---|" * len(pairs))
for k in kinds:
    arm = {}
    for n in names:
        if (n, k) in per:
            arm.setdefault(n.split("-", 1)[1], []).append(per[(n, k)][0])
    a = statistics.median(arm["A"]) if arm.get("A") else None
    b = statistics.median(arm["B"]) if arm.get("B") else None
    row = [f"{a:.3f}" if a else "-", f"{b:.3f}" if b else "-", f"{b / a:.3f}" if a and b else "-"]
    for x, y in pairs:
        px, py = per.get((x, k)), per.get((y, k))
        if px and py:
            ax, bx = (px[0], py[0]) if x.endswith("-A") else (py[0], px[0])
            row.append(f"{bx / ax:.3f}")
        else:
            row.append("-")
    print(f"| {k} | " + " | ".join(row) + " |")

print("\nDecode token tapes (gen_sha256 of each round's tokens, by cell):\n")
for cell, hs in sorted(toks.items()):
    for h, where in hs.items():
        arms = sorted({w.split("-", 1)[1] for w in where})
        print(f"- {cell}: {h[0][:16]}... in {len(where)} runs (arms {', '.join(arms)})")

tel = os.path.join(d, "lease.telemetry.csv")
if os.path.exists(tel) or os.path.exists(tel + ".gz"):
    rows = []
    for line in lines(tel):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        try:
            ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append((ts, p[1], float(p[2].split()[0]), int(p[3]), int(p[5].split()[0]), int(p[8], 16)))
        except ValueError:
            continue
    print("\nTelemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):\n")
    for n in names:
        s, e = spans[n]
        cur = [r for r in rows if s <= r[0] <= e]
        parts = []
        for card in sorted({r[1] for r in cur}):
            if cards and card not in cards:
                continue
            cr = [r for r in cur if r[1] == card]
            th = sum(1 for r in cr if r[5] & THERMAL) / len(cr)
            parts.append(
                f"card {card}: {statistics.median(r[4] for r in cr):.0f} MHz, "
                f"{statistics.median(r[2] for r in cr):.0f} W, {max(r[3] for r in cr)} C, thermal {th:.0%}"
            )
        print(f"- {n} ({len(cur)} lines, {e - s:.1f} s): " + "; ".join(parts))
