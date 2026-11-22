#!/usr/bin/env python3
"""Step 11 readings: per arm, the long bench (medians of the non-warm-up samples) and the six chat
turns (median walls), with the drafts each kept; arms read from SERVE_DIR/<tag>/.

usage: summary.py SERVE_DIR TAG... [--json OUT]
"""
import json
import os
import re
import statistics as st
import sys

args = sys.argv[1:]
out_json = None
if "--json" in args:
    i = args.index("--json")
    out_json = args[i + 1]
    del args[i:i + 2]
root, tags = args[0], args[1:]
DEPTHS = re.compile(r"\[hy4f-draft\] request_id=\"([^\"]+)\" rounds=(\d+) drafted=(\d+) accepted=(\d+) tokens=(\d+)(?: depths=\[([0-9, ]*)\])?")
WARM = re.compile(r"one-token call ([\d.]+) ms, (\d+)-row call ([\d.]+) ms, draft of 1 ([\d.]+) ms, of (\d+) ([\d.]+) ms")


def med(xs):
    return st.median(xs) if xs else None


res = {}
for tag in tags:
    d = os.path.join(root, tag)
    row = {"tag": tag}
    try:
        row["run"] = open(os.path.join(d, "run.txt")).read().splitlines()[0]
    except OSError:
        row["run"] = None
    bench = []
    if os.path.exists(os.path.join(d, "bench.jsonl")):
        bench = [json.loads(x) for x in open(os.path.join(d, "bench.jsonl")) if x.strip()]
    for w in ("long_c1", "long_c4"):
        rs = [r for r in bench if r["workload"] == w and not r["warmup"]]
        if not rs:
            continue
        dec = [st.mean(q["decode_tok_s"] for q in r["requests"]) for r in rs]
        agg = [r["aggregate_tok_s"] for r in rs]
        dr = rs[0].get("drafts") or {}
        row[w] = {"n": len(rs), "decode_tok_s": med(dec), "aggregate_tok_s": med(agg),
                  "decode_runs": dec, "agg_runs": agg,
                  "drafted": dr.get("drafted"), "accepted": dr.get("accepted"), "rounds": dr.get("rounds")}
    turns = []
    if os.path.exists(os.path.join(d, "turns.json")):
        turns = json.load(open(os.path.join(d, "turns.json")))
    tt = {}
    for t in turns:
        sp = t.get("spec") or {}
        tt[t["turn"]] = {"tokens": t["completion_tokens"], "wall_s": t["wall_s"], "walls_s": t["walls_s"],
                         "tok_s": t["completion_tokens"] / t["wall_s"] if t["wall_s"] else None,
                         "rounds": sp.get("rounds", 0), "drafted": sp.get("drafted", 0), "accepted": sp.get("accepted", 0)}
    if tt:
        row["turns"] = tt
        row["turns_total"] = {"tokens": sum(v["tokens"] for v in tt.values()),
                              "wall_s": sum(v["wall_s"] for v in tt.values()),
                              "drafted": sum(v["drafted"] for v in tt.values()),
                              "accepted": sum(v["accepted"] for v in tt.values()),
                              "rounds": sum(v["rounds"] for v in tt.values())}
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        m = WARM.search(txt)
        if m:
            row["warm_prices_ms"] = {"verify_1": float(m.group(1)), f"verify_{m.group(2)}": float(m.group(3)),
                                     "draft_1": float(m.group(4)), f"draft_{m.group(5)}": float(m.group(6))}
        depths = [0] * 6
        for m in DEPTHS.finditer(txt):
            if m.group(6):
                for i, v in enumerate(int(x) for x in m.group(6).split(",") if x.strip()):
                    if i < len(depths):
                        depths[i] += v
        row["depths_all_requests"] = depths
    res[tag] = row


def f(x, p=2):
    return "-" if x is None else f"{x:.{p}f}"


print("| arm | long c1 decode tok/s | long c1 kept/drafted | long c4 agg tok/s | long c4 kept/drafted | turns tok/s | turns kept/drafted (rate, a round) |")
print("|---|---:|---:|---:|---:|---:|---:|")
for tag, r in res.items():
    c1, c4, tt = r.get("long_c1", {}), r.get("long_c4", {}), r.get("turns_total", {})
    tr = f"{tt.get('accepted')}/{tt.get('drafted')} ({f(tt['accepted'] / tt['drafted'] if tt.get('drafted') else None)}, {f(tt['accepted'] / tt['rounds'] if tt.get('rounds') else None)})" if tt else "-"
    print(f"| {tag} | {f(c1.get('decode_tok_s'))} | {c1.get('accepted')}/{c1.get('drafted')} | {f(c4.get('aggregate_tok_s'))} | "
          f"{c4.get('accepted')}/{c4.get('drafted')} | {f(tt['tokens'] / tt['wall_s'] if tt else None)} | {tr} |")
print()
names = list(next((r["turns"] for r in res.values() if "turns" in r), {}).keys())
if names:
    print("| turn | " + " | ".join(res) + " |")
    print("|---|" + "---:|" * len(res))
    for n in names:
        cells = []
        for r in res.values():
            t = r.get("turns", {}).get(n)
            cells.append("-" if not t else f"{f(t['tok_s'])} ({t['accepted']}/{t['drafted']})")
        print(f"| {n} | " + " | ".join(cells) + " |")
print()
for tag, r in res.items():
    print(tag, "warm prices", r.get("warm_prices_ms"), "depths", r.get("depths_all_requests"))
if out_json:
    json.dump(res, open(out_json, "w"), indent=1)
