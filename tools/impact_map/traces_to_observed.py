#!/usr/bin/env python3
"""Kernel traces -> observed.json ({kernel, family, route, device} per executed memra kernel) plus a
compact per-run kernel list for the repo. Usage: traces_to_observed.py TRACES_DIR DEVICE > observed.json"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_check import base_name
d = Path(sys.argv[1]); dev = sys.argv[2]
obs, runs = [], []
for line in (d / "index.tsv").read_text().splitlines():
    if not line.strip():
        continue
    name, fam, route = line.split("\t")
    p = d / f"{name}_cuda_gpu_kern_sum.csv"
    if not p.exists():
        continue
    ks = sorted({base_name(r["Name"]) for r in csv.DictReader(open(p))})
    runs.append({"run": name, "family": fam, "route": route, "device": dev, "kernels": ks})
    obs += [{"kernel": k, "family": fam, "route": route, "device": dev} for k in ks]
json.dump({"schema": "memra-impact-observed-v1", "runs": runs, "observed": obs}, sys.stdout, indent=0)
