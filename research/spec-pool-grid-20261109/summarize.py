#!/usr/bin/env python3
"""Join the base and fix pair-cell verdicts into one markdown table.

usage: summarize.py BASE_VERDICTS[,MORE...] FIX_VERDICTS[,MORE...]
Each argument is a comma list of verdicts.jsonl files; a later row for a case replaces an earlier one.
"""
import json
import sys


def load(arg):
    rows = {}
    for path in arg.split(","):
        for line in open(path):
            v = json.loads(line)
            rows[v["case"]] = v
    return rows


def path(v):
    w = v["warm"]
    parts = []
    for (n,) in w["spec_parked_end"]:
        parts.append(f"parked end {n}")
    for pos, _ in w["spec_affinity"]:
        parts.append(f"affinity {pos}" + ("" if pos % 32 == 0 else " (off grid)"))
    for c, g in w["spec_grid_rewind"]:
        parts.append(f"rewind {c}->{g}")
    if w["spec_grid_decline"]:
        parts.append(f"{len(w['spec_grid_decline'])} grid decline")
    if w["spec_affinity_grid_decline"]:
        parts.append(f"{len(w['spec_affinity_grid_decline'])} affinity grid decline")
    if w["spec_park_skip"]:
        parts.append(f"{len(w['spec_park_skip'])} park skip")
    if w["spec_restore"]:
        parts.append("prefix restore " + "/".join(str(a) for a, _ in w["spec_restore"]))
    return ", ".join(parts) or "none"


def cell(v):
    if v is None:
        return "not run", ""
    t = v["turns"]
    eq = "/".join("=" if t[k]["equal"] else f"{t[k]['warm_len']}!={t[k]['cold_len']}" for k in ("2", "3"))
    return ("PASS" if v["bytes"] else "FAIL") + f" ({eq})", path(v)


base, fix = load(sys.argv[1]), load(sys.argv[2])
print("| case | sampling | base bytes (t2/t3) | base warm path | fix bytes (t2/t3) | fix warm path |")
print("|---|---|---|---|---|---|")
for name in list(dict.fromkeys(list(base) + list(fix))):
    v = base.get(name) or fix.get(name)
    env = v["env"]
    samp = "T=1" if "_TEMP" in env else "greedy"
    if "_FREQ" in env:
        samp += " pen"
    b, bp = cell(base.get(name))
    f, fp = cell(fix.get(name))
    print(f"| {name} | {samp} | {b} | {bp} | {f} | {fp} |")
