#!/usr/bin/env python3
"""Summarize an interleaved dsv41-bench layer A/B (perf_ab.sh: runs run<i>-<A|B>.json in order).

Per layer and phase: each run's median (prefill: over its reps; decode: over steps 2..), the arm
median of the run medians, the within-arm spread ((max - min) / median), the B/A ratio, whether the
two arms' run ranges overlap, and whether every run of both arms wrote the same output hashes
(the byte-identity check: a schedule change must not move them). Telemetry: GPU0 power, temperature,
SM clock and fan over the runs, and the co-tenant cards' power (GPUs 2 and 3, another session).

usage: ab_summary.py AB_DIR [--md OUT.md] [--json OUT.json]
"""
import csv
import glob
import json
import os
import re
import statistics
import sys


def med(v):
    return statistics.median(v) if v else float("nan")


def main():
    d = sys.argv[1]
    out_md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    out_js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "run*-?.json")), key=lambda p: int(re.search(r"run(\d+)-", p).group(1))):
        arm = f[-6]
        runs.append((arm, json.load(open(f)), f))
    per = {}  # (layer, phase) -> arm -> [run medians]
    hashes = {}  # (layer, phase) -> set of hash strings per arm
    for arm, js, _ in runs:
        for L in js["layers"]:
            key = (L["layer"], f"prefill {js['tokens']}")
            per.setdefault(key, {}).setdefault(arm, []).append(L["median_ms"])
            hashes.setdefault(key, {}).setdefault(arm, set()).add(L["out_sha"] + ":" + L["pre_sha"])
        for D in js.get("decode", []):
            for pl in D["per_layer"]:
                key = (pl["layer"], f"decode at {D['from_pos']}")
                per.setdefault(key, {}).setdefault(arm, []).append(pl["median_ms_after_first"])
            key = (tuple(D["run"]), f"decode steps at {D['from_pos']}")
            hashes.setdefault(key, {}).setdefault(arm, set()).add("|".join(D["step_sha"]))
    rows = []
    md = ["| layer | phase | A runs (ms) | B runs (ms) | A median | B median | B/A | A spread | B spread | ranges |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(per, key=lambda k: (str(k[1]), k[0])):
        a, b = per[key].get("A", []), per[key].get("B", [])
        ma, mb = med(a), med(b)
        sa = (max(a) - min(a)) / ma if a else float("nan")
        sb = (max(b) - min(b)) / mb if b else float("nan")
        overlap = not (max(b) < min(a) or max(a) < min(b)) if a and b else None
        rows.append({"layer": key[0], "phase": key[1], "A": a, "B": b, "A_median": ma, "B_median": mb,
                     "ratio": mb / ma, "A_spread": sa, "B_spread": sb, "overlap": overlap})
        md.append(f"| {key[0]} | {key[1]} | {', '.join(f'{x:.3f}' for x in a)} | {', '.join(f'{x:.3f}' for x in b)} | "
                  f"{ma:.3f} | {mb:.3f} | {mb / ma:.3f} | {100 * sa:.2f}% | {100 * sb:.2f}% | "
                  f"{'OVERLAP' if overlap else 'NO OVERLAP'} |")
    md.append("")
    md.append("| layer | phase | hashes A | hashes B | identical |")
    md.append("|---|---|---|---|---|")
    ident = {}
    for key, hv in sorted(hashes.items(), key=lambda x: str(x[0])):
        ha, hb = hv.get("A", set()), hv.get("B", set())
        ok = len(ha) == 1 and ha == hb
        ident[str(key)] = ok
        md.append(f"| {key[0]} | {key[1]} | {len(ha)} distinct | {len(hb)} distinct | {'yes' if ok else 'NO'} |")
    # telemetry
    tel = {"gpu0_power_w": [], "gpu0_temp_c": [], "gpu0_sm_mhz": [], "gpu0_fan": [], "cotenant_power_w": []}
    for f in glob.glob(os.path.join(d, "run*-?.telemetry.csv")) + glob.glob(os.path.join(d, "telemetry.csv")):
        for r in csv.reader(open(f)):
            if len(r) < 9:
                continue
            try:
                idx = int(r[1])
                pw = float(r[2].strip().split()[0])
            except ValueError:
                continue
            if idx == 0:
                tel["gpu0_power_w"].append(pw)
                tel["gpu0_temp_c"].append(float(r[3]))
                tel["gpu0_sm_mhz"].append(float(r[5].strip().split()[0]))
                tel["gpu0_fan"].append(float(r[4].strip().split()[0]))
            elif idx in (2, 3):
                tel["cotenant_power_w"].append(pw)
    tsum = {k: {"n": len(v), "median": med(v), "max": max(v) if v else None} for k, v in tel.items()}
    md.append("")
    md.append("Telemetry (250 ms, all runs): " + ", ".join(
        f"{k} median {v['median']:.0f} max {v['max']:.0f}" for k, v in tsum.items() if v["n"]))
    text = "\n".join(md)
    print(text)
    if out_md:
        open(out_md, "w").write(text + "\n")
    if out_js:
        json.dump({"rows": rows, "identical": ident, "telemetry": tsum,
                   "order": [a for a, _, _ in runs]}, open(out_js, "w"), indent=1)


if __name__ == "__main__":
    main()
