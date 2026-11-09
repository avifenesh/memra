#!/usr/bin/env python3
"""Tables for the cold-long K=2 vs K=3 A/B (cold_long_ab.py output).

usage: analyze.py OUT   (reads OUT/runs.jsonl and OUT/telemetry-*.csv; prints markdown)

Three reads of the same rows:
- per arm and class: n, TTFT p50 / p95 / range, decode median / range / pooled, ITL medians,
  acceptance;
- drift-normalized: each long row's decode rate over its own boot's t1-short reference median (the
  cold-short rung, K=3 on both arms), so a boot that runs hot or cool cancels;
- per boot: GPU temperature, SM clock and power medians from the 250 ms telemetry.
"""
import csv
import json
import pathlib
import statistics
import sys

OUT = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in open(OUT / "runs.jsonl")]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def med(v):
    return statistics.median(v) if v else None


ref = {}
for b in sorted({r["boot"] for r in rows}):
    v = [r["decode_tok_s"] for r in rows if r["boot"] == b and r["class"] == "t1-short" and r["decode_tok_s"]]
    ref[b] = med(v)

print("| class | arm | K | n | TTFT p50 / p95 (s) | TTFT range (s) | decode median (tok/s) | decode range | pooled | normalized decode | ITL p50 / p95 / p99 (ms) | acceptance |")
print("|---|---|---|---:|---|---|---:|---|---:|---:|---|---:|")
groups = [("t1-long 1.4k", "t1-long", "-1k"), ("t1-long 4k", "t1-long", "-4k"), ("t1-long 6k", "t1-long", "-6k"),
          ("t1-long all", "t1-long", ""), ("greedy-long", "greedy-long", ""), ("t1-short (reference)", "t1-short", "")]
for label, cls, tag in groups:
    for arm in ("k3", "k2"):
        sel = [r for r in rows if r["arm"] == arm and r["class"] == cls and r["prompt"].endswith(tag)]
        if not sel:
            continue
        rates = [r["decode_tok_s"] for r in sel if r["decode_tok_s"]]
        ttfts = [r["ttft_s"] for r in sel if r["ttft_s"]]
        norm = [r["decode_tok_s"] / ref[r["boot"]] for r in sel if r["decode_tok_s"] and ref.get(r["boot"])]
        toks = sum(r["completion_tokens"] for r in sel)
        dec = sum(r["total_s"] - r["ttft_s"] for r in sel if r["ttft_s"])
        k = "/".join(sorted({r["k"][0] for r in sel if r["k"]}))
        itl = [r["itl_ms"] for r in sel if r["itl_ms"]]
        print(f"| {label} | {arm} | {k} | {len(sel)} | {pct(ttfts, .5):.3f} / {pct(ttfts, .95):.3f} | "
              f"{min(ttfts):.3f}-{max(ttfts):.3f} | {med(rates):.2f} | {min(rates):.2f}-{max(rates):.2f} | "
              f"{toks / dec:.2f} | {med(norm):.4f} | {med([i['p50'] for i in itl]):.1f} / "
              f"{med([i['p95'] for i in itl]):.1f} / {med([i['p99'] for i in itl]):.1f} | "
              f"{med([r['acceptance'] for r in sel if r['acceptance'] is not None]):.3f} |")

print()
print("| boot | arm | t1-short reference decode | GPU temp median (C) | SM clock median (MHz) | power median (W) |")
print("|---:|---|---:|---:|---:|---:|")
for b in sorted(ref):
    arm = next(r["arm"] for r in rows if r["boot"] == b)
    tel = OUT / f"telemetry-{b}-{arm}.csv"
    temps, clocks, power = [], [], []
    if tel.exists():
        for row in csv.reader(open(tel)):
            if len(row) < 6 or row[0].startswith("timestamp"):
                continue
            try:
                temps.append(float(row[1]))
                clocks.append(float(row[2].split()[0]))
                power.append(float(row[4].split()[0]))
            except ValueError:
                continue
    print(f"| {b} | {arm} | {ref[b]:.2f} | {med(temps) if temps else '-'} | {med(clocks) if clocks else '-'} | "
          f"{round(med(power), 1) if power else '-'} |")

print()
same = {}
for r in rows:
    if r["class"] == "greedy-long":
        same.setdefault(r["prompt"], {}).setdefault(r["arm"], set()).add(r["sha"])
print("greedy texts per prompt and arm (one sha each, equal across arms = K-invariant):",
      {p: {a: sorted(s) for a, s in v.items()} for p, v in same.items()})
