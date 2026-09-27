#!/usr/bin/env python3
"""Recompute B1 scoring and verdicts with the registered B1 amendment gate (reads), offline.

Usage: m1-b1-resummarize.py <b1 run dir>  (writes summary-amended-gate.json beside the run's own
summary.json, which is kept unchanged). Inputs are each visit's recorded fields only.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b1", HERE / "m1-b1-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
LIMIT = 0.02
FLOOR = 1 << 20


def main():
    run = Path(sys.argv[1])
    visits = []
    for f in sorted(run.glob("*/r*/visit.json")):
        v = json.loads(f.read_text())
        c = v["contamination"]
        c["foreign_read_bytes"] = max(0, c["device_read_bytes"] - v["own_io"]["read_bytes"])
        c["read_gate_limit"] = max(LIMIT * c["device_read_bytes"], FLOOR)
        v["clean_timing_amended"] = bool(v["telemetry_ok"] and v.get("regime_ok", True)
                                         and c["foreign_read_bytes"] <= c["read_gate_limit"])
        v["scored_in_run"] = v["scored"]
        v["scored"] = bool(v["clean_timing_amended"] and not v["problems"])
        visits.append(v)
    sizes = sorted({v["size"] for v in visits})
    out = {"gate": "B1 amendment: foreign read bytes <= max(2% of device read bytes, 1 MiB)",
           "visits": len(visits), "scored": sum(v["scored"] for v in visits),
           "failed_problems": sum(1 for v in visits if v["problems"]),
           "cells": R.summarize(visits, sizes, 10), "qualified": False}
    (run / "summary-amended-gate.json").write_text(json.dumps(out, indent=1) + "\n")
    for key, cell in out["cells"].items():
        print("M1-B1-CELL " + key + " " + " ".join(
            f"{m}={c['median'] / 1e6:.1f}MB/s(n={c['n_scored']})" + (f",{c['verdict_vs_buffered']}" if "verdict_vs_buffered" in c else "")
            for m, c in cell.items() if c["median"] is not None))
    print(f"M1-B1-RESUMMARY visits={out['visits']} scored={out['scored']} failed={out['failed_problems']}")


if __name__ == "__main__":
    main()
