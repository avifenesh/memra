#!/usr/bin/env python3
"""Per-run table and arm medians of the w15 served A/B (A = MEMRA_DSV41_BATCH_ROWS=8, B = 16).
usage: ab_table.py AB_DIR > per_run.md"""
import json
import os
import statistics
import sys

ab = sys.argv[1]
runs = sorted(d for d in os.listdir(ab) if d.startswith("run"))
rows = []
hashes = set()
for r in runs:
    arm = r.split("-")[1]
    env = open(os.path.join(ab, r, "arm.env")).read().split()
    hashes.add(open(os.path.join(ab, r, "sha256.txt")).read().split()[0])
    for line in open(os.path.join(ab, r, "bench", "samples.jsonl")):
        s = json.loads(line)
        g = list(s["gpu"].values())
        rows.append({
            "run": r, "arm": arm, "env": [e for e in env if "BATCH_ROWS" in e][0], "w": s["workload"],
            "agg": s["agg_out_tok_s"], "req": s["per_req_decode_tok_s"]["median"],
            "req_min": s["per_req_decode_tok_s"]["min"], "ttft_med": s["ttft_s"]["median"],
            "ttft_max": s["ttft_s"]["max"], "tok": s["completion_tokens"], "wall": s["wall_s"],
            "err": s["n_err"], "pw": max(x["power_mean_w"] for x in g), "mhz": min(x["sm_mean_mhz"] for x in g),
            "temp": max(x["temp_max_c"] for x in g), "reasons": sorted({x["clock_event_reasons"] for x in g}),
        })
print("# w15 served A/B: MEMRA_DSV41_BATCH_ROWS 8 (A) against 16 (B), four ranks\n")
print(f"memra-server sha256 (every run): {', '.join(sorted(hashes))}\n")
print("| run | arm | workload | agg out tok/s | per-request decode tok/s (median, min) | TTFT median / max s "
      "| tokens | wall s | errors | power mean max W | SM mean min MHz | temp max C | clock events |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print(f"| {x['run']} | {x['arm']} ({x['env']}) | {x['w']} | {x['agg']} | {x['req']}, {x['req_min']} | "
          f"{x['ttft_med']} / {x['ttft_max']} | {x['tok']} | {x['wall']} | {x['err']} | {x['pw']} | {x['mhz']} | "
          f"{x['temp']} | {' '.join(x['reasons'])} |")
print("\n## Arm medians (N = 3 runs an arm, one sample a run)\n")
print("| workload | metric | A (8) median [min, max] | B (16) median [min, max] | B/A |")
print("|---|---|---|---|---|")
for w in sorted({x["w"] for x in rows}):
    for m, name in (("agg", "agg out tok/s"), ("req", "per-request decode tok/s"), ("ttft_med", "TTFT median s"),
                    ("ttft_max", "TTFT max s")):
        a = [x[m] for x in rows if x["w"] == w and x["arm"] == "A"]
        b = [x[m] for x in rows if x["w"] == w and x["arm"] == "B"]
        ma, mb = statistics.median(a), statistics.median(b)
        print(f"| {w} | {name} | {ma} [{min(a)}, {max(a)}] | {mb} [{min(b)}, {max(b)}] | {mb / ma:.3f} |")
