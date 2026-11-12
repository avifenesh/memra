#!/usr/bin/env python3
"""memra#1255: one row per harness/reclaim-cell.py run (summary.json), in run order.
usage: reclaim-table.py RECLAIM_DIR"""
import json
import sys
from pathlib import Path

runs = [d for d in Path(sys.argv[1]).iterdir() if (d / "summary.json").exists()]
runs.sort(key=lambda d: (d / "cell.log").read_text().splitlines()[0] if (d / "cell.log").exists() else d.name)
print("| run | binary sha256 (first 16) | budget MB | A, B charge MB | placed MB (band) | partial reclaim: need MB "
      "(owed MB), freed MB | settle waits | B's route | pair in one pass | out of memory | A, B status |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for d in runs:
    s = json.loads((d / "summary.json").read_text())
    if "partial_reclaims" not in s:
        continue
    learn = s.get("learn", {})
    pr = "; ".join(f"{p['need_mb']} ({p['owed_mb']}), {p['freed_mb']}" for p in s["partial_reclaims"]) or "none"
    routes = s.get("routes") or []
    ans = s.get("answers", {})
    sha = (d / "cell.log").read_text().splitlines()[1][:16]
    print(f"| `{d.name}` | `{sha}` | {s.get('budget_mb')} | "
          f"{learn.get('A', {}).get('charge', 0) // 1_000_000}, {learn.get('B', {}).get('charge', 0) // 1_000_000} | "
          f"{s.get('free_before_pair_mb')} ({s.get('lo_mb')}, {s.get('hi_mb')}) | {pr} | "
          f"{s.get('settle_lines')} | {routes[1] if len(routes) > 1 else 'n/a'} | {s.get('both_in_one_pass')} | "
          f"{s.get('oom_lines')} | {ans.get('A', {}).get('status')}, {ans.get('B', {}).get('status')} |")
