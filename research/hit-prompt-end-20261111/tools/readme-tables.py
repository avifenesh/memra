#!/usr/bin/env python3
"""memra#1233: the README's cell tables from the banked runs. usage: readme-tables.py RECEIPTS_DIR"""
import json, subprocess, sys
from pathlib import Path
R = Path(sys.argv[1]); T = R / "tools" / "cells-table.py"
def rows(d):
    out = subprocess.run([sys.executable, str(T), str(R / d), "--json"], check=True, capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines()]
def f(v, nd=0): return "-" if v is None else f"{v:,.{nd}f}"
NAME = {"main": "main", "bonly": "b-only", "fix": "fix", "tf": "tails-first", "lru": "LRU-item"}
# GB10 boots 000849 and 001403 ran the tails-first build while it was still the lane's `fix` binary.
RELABEL = {"fix-000849": "tf", "fix-001403": "tf"}
def table(d, arms, rename=None):
    print("| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows(d):
        r["arm"] = RELABEL.get(r["label"], r["arm"]) if "confirm" not in d else r["arm"]
        if r["arm"] not in arms: continue
        arm = (rename or {}).get(r["arm"], NAME[r["arm"]])
        tails = "-" if r["tail_attaches"] is None else f"{r['tail_attaches']} / {r['tail_hits']} / {r['tail_skips']} / {r['tail_strips']}"
        print(f"| {r['label']} | {arm} | {r['restored']} ({r['whole']} + {r['suffix']}) | {f(r['hit_p50'])} / {f(r['hit_p95'])} | {f(r['whole_p50'])} / {f(r['whole_p95'])} | "
              f"{f(r['suffix_p50'])} / {f(r['suffix_p95'])} | {f(r['suffix_prime_wait_p50'])} / {f(r['suffix_prime_wait_p95'])} | {f(r['cold_p50'])} / {f(r['cold_p95'])} | {f(r['tok_s'],1)} | {tails} |")
which = sys.argv[2]
if which == "gb10": table("gb10/gb10-hitcold", {"main", "bonly", "fix"})
elif which == "gb10-policy": table("gb10/gb10-hitcold", {"lru", "tf", "fix"})
elif which == "rtx": table("rtx5090/abba", {"main", "bonly", "fix"})
elif which == "rtx2": table("rtx5090/abba-telemetry", {"main", "bonly", "fix"})
elif which == "rtx-confirm": table("rtx5090/confirm/abba", {"main", "fix"})
elif which == "gb10-confirm": table("gb10/confirm/gb10-hitcold", {"main", "fix"})
elif which == "rtx-policy": table("rtx5090/policy-middle", {"tf", "fix"})
elif which == "rtx-tf": table("rtx5090/tails-first/abba", {"main", "bonly", "fix"}, {"fix": "tails-first", "main": "main", "bonly": "b-only"})
