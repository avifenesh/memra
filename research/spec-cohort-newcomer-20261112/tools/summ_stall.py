#!/usr/bin/env python3
"""Table of the stall cells with LONGS long streams (stall_cell.py rows) for memra#1227.
usage: summ_stall.py RUNDIR [RUNDIR...]   (each holds stall.jsonl and texts/)
Per (cell, arm): tokens of each long stream inside the 30 s window, their longest gap, the short
requests served and their latency, stall marks set in the window, and the newcomers admitted to
plain decode. A short request is one token, so "all tokens" is the long streams' tokens
plus the short requests served. Text check: each long stream's texts are prefixes of the longest across arms."""
import collections
import json
import pathlib
import sys

rows = []
for d in sys.argv[1:]:
    d = pathlib.Path(d)
    for l in open(d / "stall.jsonl"):
        if l.strip():
            r = json.loads(l)
            r["_dir"] = d
            rows.append(r)
print("| cell (clients:think ms) | class | arm | long tokens in the window, per stream | longest gap s, per stream | short requests | all tokens (long + short) | short p50 s | short p95 s | stall marks | newcomers plain | demotions |")
print("|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|")
for r in rows:
    toks = [r["tokens_during"]] + [x["tokens_during"] for x in r.get("extra_longs", [])]
    gaps = [r["longest_gap_in_window_s"]] + [x["longest_gap_in_window_s"] for x in r.get("extra_longs", [])]
    sk = r["census_window"]["spec_k"]
    plain = sum(v for k, v in sk.items() if k.startswith("K0/"))
    print(f"| {r['clients']}:{int(r['think_ms'])} | {r['class']} | {r['arm']} | {' + '.join(map(str, toks))} = {sum(toks)} | {', '.join(map(str, gaps))} | {r['short_requests']} | {sum(toks) + r['short_requests']} | "
          f"{r['short_lat_p50_s']} | {r['short_lat_p95_s']} | {r['census_window']['stall_marks']} | {plain} of {sum(sk.values())} | {r['demotions']} |")
bad = n = 0
by = collections.defaultdict(list)
for r in rows:
    by[(r["clients"], r["think_ms"], r["class"])].append(r)
for key, rs in sorted(by.items()):
    streams = [[(r["arm"], (r["_dir"] / r["text_file"]).read_text()) for r in rs]]
    for i in range(len(rs[0].get("extra_longs", []))):
        streams.append([(r["arm"], (r["_dir"] / r["extra_longs"][i]["text_file"]).read_text()) for r in rs])
    for i, texts in enumerate(streams):
        longest = max(texts, key=lambda t: len(t[1]))[1]
        for arm, t in texts:
            n += 1
            if not t or not longest.startswith(t):
                bad += 1
                print(f"PREFIX MISMATCH cell {key} stream {i} arm {arm}")
print(f"\nprefix check: {n} long texts, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")
