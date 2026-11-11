#!/usr/bin/env python3
"""Summarize an A B B A timing proxy dir (time_c.sh): layer-2 prefill median, decode-step medians, and the GEMM
bench medians per shape, arm A vs arm B (median of the arm's two runs), with B / A."""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = {"A": [], "B": []}
for log in sorted(d.glob("run*-layer.log")):
    a = log.name.split("-")[1]
    t = log.read_text(errors="replace")
    m = re.search(r"reps \[([^\]]*)\] ms, median ([0-9.]+) ms", t)
    reps = [float(x) for x in m.group(1).split(",")] if m else []
    steady = sorted(reps[2:])[len(reps[2:]) // 2] if len(reps) > 2 else float("nan")
    dec = re.search(r"layer 2: decode at [^:]*: steps \[[^\]]*\] ms, median \(steps 2\.\.\) ([0-9.]+) ms", t)
    whole = re.search(r"whole decode steps[^\]]*\] ms, median \(steps 2\.\.\) ([0-9.]+) ms", t)
    same = re.search(r"(\d+)/(\d+) reruns byte-identical", t)
    arms[a].append(("layer", steady, float(dec.group(1)) if dec else float("nan"),
                    float(whole.group(1)) if whole else float("nan"), same.group(0) if same else "?"))
gemm = {"A": {}, "B": {}}
for log in sorted(d.glob("run*-gemm.log")):
    a = log.name.split("-")[1]
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"(gemm .*?): median ([0-9.]+) ms", line)
        if m:
            key = re.sub(r" (bm \d+|form \d+)", "", m.group(1))
            key = key.replace("tiles ", "").replace("grouped ", "").replace(" groups 6 with rows 6", "")
            key = re.sub(r" groups (\d+) with rows (\d+)", r" experts \2 of \1", key)
            gemm[a].setdefault(key, []).append(float(m.group(2)))
print(f"timing proxy {d.name}")
for a in "AB":
    for row in arms[a]:
        print(f"  arm {a}: layer-2 4k prefill steady median {row[1]:.2f} ms, layer decode step {row[2]:.3f} ms, "
              f"whole decode step {row[3]:.3f} ms, {row[4]}")
def med(v):
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
for k in gemm["B"]:
    if k in gemm["A"]:
        a, b = med(gemm["A"][k]), med(gemm["B"][k])
        print(f"  {k}: A {a:.4f} ms, B {b:.4f} ms, B/A {b / a:.3f}")
    else:
        print(f"  {k}: B {med(gemm['B'][k]):.4f} ms (no A shape)")
