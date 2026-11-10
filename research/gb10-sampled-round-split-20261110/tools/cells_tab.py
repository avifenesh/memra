#!/usr/bin/env python3
"""cells_tab.py CELL_DIR...: the client's clock per request of each cell (reqs.jsonl).

decode tok/s = completion tokens / (total less TTFT); wall ms/round = (total less TTFT) / rounds;
issue figure = 1000 x (1 + K x acceptance) / decode tok/s, the round time of memra#1186's table;
in brackets the change against the first greedy request of the same boot.
"""
import json
import pathlib
import sys

print(f"{'cell':9s} {'class':9s} {'rounds':>6s} {'drafted':>7s} {'accepted':>8s} {'decode tok/s':>12s} {'wall ms/round':>20s} {'issue figure':>20s}  text sha")
for d in map(pathlib.Path, sys.argv[1:]):
    reqs = [json.loads(l) for l in open(d / "reqs.jsonl") if l.strip()]
    base = None
    for q in reqs:
        dec = (q["total_s"] - q["ttft_s"]) * 1e3
        wall = dec / q["rounds"]
        tpr = 1 + q["drafted"] / q["rounds"] * q["accepted"] / q["drafted"]
        issue = dec / q["completion_tokens"] * tpr
        if q["class"] == "greedy":
            base = (wall, issue)
        b = base or (wall, issue)
        mark = "" if q["error"] is None else f"  ERROR {q['error']}"
        print(f"{d.name:9s} {q['class']:9s} {q['rounds']:6d} {q['drafted']:7d} {q['accepted']:8d} {q['completion_tokens'] / dec * 1e3:12.2f} "
              f"{wall:10.2f} [{wall - b[0]:+7.2f}] {issue:10.2f} [{issue - b[1]:+7.2f}]  {q['sha']}{mark}")
