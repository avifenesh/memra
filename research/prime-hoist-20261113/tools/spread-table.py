#!/usr/bin/env python3
"""memra#1258: per-boot tables of b-spread.py rows: each prime call's demand (pool reserved after
the call - pool used before it, MiB) and growth (newly mapped), with min/median/max per call index.
usage: spread-table.py SPREAD_DIR..."""
import json
import statistics
import sys
from pathlib import Path

for d in sys.argv[1:]:
    print(f"## {Path(d).name}")
    print("| boot | mode | n | call | demand MiB min / median / max | growth MiB min / median / max | primes with a trim line | step OOM |")
    print("|---|---|---|---|---|---|---|---|")
    for boot in sorted(p for p in Path(d).glob("b*-mode*") if p.is_dir()):
        rows = [json.loads(l) for l in (boot / "rows.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("path") == "spec"]
        if not rows:
            continue
        ncalls = max(len(r.get("calls") or []) for r in rows)
        for c in range(ncalls):
            dem = [r["calls"][c]["demand_mib"] for r in rows if len(r.get("calls") or []) > c]
            gro = [r["calls"][c]["growth_mib"] for r in rows if len(r.get("calls") or []) > c]
            fires = sum(1 for r in rows if any(" fire " in t or " drain " in t for t in r.get("trim_lines", [])))
            oom = sum(r.get("oom_lines", 0) for r in rows)
            print(f"| {boot.name} | {rows[0]['mode']} | {len(dem)} | {c + 1} | {min(dem)} / {statistics.median(dem):.0f} / {max(dem)} | "
                  f"{min(gro)} / {statistics.median(gro):.0f} / {max(gro)} | {fires} | {oom} |")
