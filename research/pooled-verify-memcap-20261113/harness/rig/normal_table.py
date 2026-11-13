#!/usr/bin/env python3
"""normal_table.py CAMPAIGN_DIR: memra#1266. The normal-concurrency cells (normal_cell.py), one row
per boot and c in boot order, the text-set check per c across boots, and each boot's summary.txt
(OOM, capped forwards, pre-cohort reclaims, cohort engagement). Then per c the arm medians."""
import json
import pathlib
import statistics
import sys

camp = pathlib.Path(sys.argv[1])
rows = []
for d in sorted(p for p in camp.iterdir() if p.is_dir() and (p / "normal.jsonl").exists()):
    for line in (d / "normal.jsonl").read_text().splitlines():
        rows.append((d.name, json.loads(line)))
print("| boot | c | requests, errors | completion tokens | agg tok/s | TTFT p50 s | ITL p50 / p99 ms |")
print("|---|---|---|---|---|---|---|")
sets = {}
for name, r in rows:
    sets.setdefault(r["c"], set()).add(tuple(sorted((x["prompt"], x.get("sha")) for x in r["rows"])))
    print(f"| {name} | {r['c']} | {r['requests']}, {r['errors']} | {r['completion_tokens']} | {r['agg_tok_s']} | "
          f"{r['ttft_p50_s']} | {r['itl_p50_ms']} / {r['itl_p99_ms']} |")
print()
for c, s in sorted(sets.items()):
    print(f"- c={c}: {'one' if len(s) == 1 else len(s)} set of (prompt, text sha) across the boots")
for d in sorted(p for p in camp.iterdir() if p.is_dir() and (p / "summary.txt").exists()):
    print(f"- {d.name}: " + "; ".join(l.strip() for l in (d / "summary.txt").read_text().splitlines()))
print()
for c in sorted(sets):
    by = {}
    for name, r in rows:
        if r["c"] == c:
            by.setdefault(name.split("-")[-1], []).append(r["agg_tok_s"])
    print(f"- c={c} agg tok/s: " + ", ".join(f"{arm} {statistics.median(v):.2f} ({', '.join(map(str, v))})" for arm, v in sorted(by.items())))
