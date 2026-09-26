#!/usr/bin/env python3
"""Pool one 5090 B3 regime's round cells into its verdict (M1-PREREG.md section D).

Usage: m1-b3-pool.py <regime dir written by m1-5090-rounds.py>  -> pooled-summary.json

Each round is its own collector cell (`round-KK/`, or `round-KK-attemptN/` after a lost lock
race; a lost race leaves no visits). Every round's visits come from the one cell that ran it.
The registered verdict uses the registered co-tenancy gate (runner.verdicts, the visit's own
`clean_timing`, which already folds in the per-visit GPU co-tenant gate). The B1-amendment read
gate is computed beside it, labelled post-hoc, exactly as m1-b3-resummarize.py does for BOX27.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load("runner", "m1-spill-runner.py")


def read_gate(v):
    v = copy.deepcopy(v)
    c = v["contamination"]
    fr = max(0, c["device_read_bytes"] - v["proc_io"]["read_bytes"])
    v["clean_timing"] = bool(v["telemetry_ok"] and v["thermal_ok"] and v["identity_after_ok"]
                             and v.get("regime_ok", True) and not v.get("gpu_cotenant", False)
                             and fr <= max(0.02 * c["device_read_bytes"], 1 << 20))
    v["scored"] = bool(v["clean_timing"] and not v["correctness_problems"] and v["exit_code"] == 0
                       and not v["timed_out"])
    return v


def main():
    d = Path(sys.argv[1])
    cells = {}
    for cell in sorted(d.glob("round-[0-9][0-9]*")):
        if cell.is_dir() and (cell / "visits").is_dir():
            cells[int(cell.name[6:8])] = cell
    visits, lock_arms = [], []
    for k, cell in sorted(cells.items()):
        for p in sorted((cell / "visits").glob("r*/visit.json")):
            v = json.loads(p.read_text())
            if v["round"] != k:
                raise SystemExit(f"REFUSED: {p} carries round {v['round']}, cell is round {k}")
            v["_cell"] = cell.name
            visits.append(v)
            if v["arm"] not in lock_arms:
                lock_arms.append(v["arm"])
    refused = sorted({v["arm"] for v in visits if v["correctness_problems"]})
    arms = [a for a in lock_arms if a not in refused]
    kept = [v for v in visits if v["arm"] in arms]
    reg = R.verdicts(kept, arms, "worker16")
    post = R.verdicts([read_gate(v) for v in kept], arms, "worker16")
    gpu_unclean = sum(1 for v in visits if v.get("gpu_cotenant"))
    summary = {"rounds": sorted(cells), "cells": {k: c.name for k, c in cells.items()}, "visits": len(visits),
               "refused_arms": refused, "gpu_cotenant_unclean_visits": gpu_unclean,
               "registered": reg, "post_hoc_read_gate": post}
    (d / "pooled-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for label, s in (("M1-5090-VERDICT", reg), ("M1-5090-VERDICT-READGATE(post-hoc)", post)):
        for arm, x in s["arms"].items():
            print(f"{label} arm={arm} vs worker16: {x['verdict']} median_ratio={x['median_ratio']} pairs={x['n_pairs']}")
        print(f"{label} regime_scored={s['regime_scored']} contaminated={s['contaminated_visits']}")
    print(f"rounds={sorted(cells)} visits={len(visits)} refused={refused} gpu_cotenant_unclean={gpu_unclean}")


if __name__ == "__main__":
    main()
