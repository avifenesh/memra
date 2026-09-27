#!/usr/bin/env python3
"""G2 5090 half summary (OWED 23; M1-PREREG.md section D): D's replay rules for the ten sizes.

Usage: m1-g2-5090-summary.py <cell dir: the collector's --out, with visits/>  -> g2-summary.json

Same replay as research/spill-d-20260919/summarize-g2.py: every raw probe log is re-parsed and
must equal its recorded samples with the recorded hash; no competing compute application in any
visit; the calibrated copy count is fixed; N=5 in each order per size, direction and arm; 250 ms
collector telemetry with no gap above 1 s. Differences, as registered: ten sizes and 400 scored
samples; the power check is the 5090's constant-envelope rule (the probe's power fields identical
on every sample, recorded in power-envelope.json) instead of 600/600 W.
"""
from collections import defaultdict
import csv
import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g2w", HERE / "m1-g2-5090.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
G, SIZES = W.G, W.SIZES


def summarize(root):
    G.B.validate_cell(root / "CELL.jsonl")
    capture = json.loads((root / "command.capture.json").read_text())
    G.B.require(capture["exit_code"] == 0 and capture["result"]["samples"] == 400, "campaign did not complete")
    telemetry = capture["gpu_telemetry"]
    G.B.require(telemetry["interval_ms"] == 250, "telemetry not continuously captured")
    visits = root / "visits"
    rows = [json.loads(s) for s in (visits / "samples.jsonl").read_text().splitlines()]
    scored = [r for r in rows if r["phase"] == "scored"]
    expected = [f"score-{size}-{pair}-{order}" for size in SIZES for pair in range(5) for order in ("ab", "ba")]
    G.B.require([r["visit"] for r in scored[::4]] == expected and len(scored) == 400, "incomplete/reordered campaign")
    calibrated = json.loads((visits / "calibration.json").read_text())
    envelope = json.loads((visits / "power-envelope.json").read_text())
    grouped = defaultdict(list)
    for name in dict.fromkeys(r["visit"] for r in rows):
        recorded = [r for r in rows if r["visit"] == name]
        first = recorded[0]
        raw = visits / first["raw_log"]
        samples = G.C.check(raw.read_text(), first["bytes"], first["copies"], first["order"])
        G.B.require(len(recorded) == 4 and not (visits / (name + ".compute.log")).read_text().strip(),
                    "incomplete visit or competing GPU process")
        for sample, record in zip(samples, recorded):
            G.B.require(all(record[k] == v for k, v in sample.items()) and record["raw_sha256"] == G.B.digest(raw),
                        "raw sample/hash mismatch")
            G.B.require({k: record["power_before"][k] for k in envelope} == envelope, "power envelope changed")
            if record["phase"] == "scored":
                G.B.require(record["copies"] == calibrated[str(record["bytes"])], "copy count changed")
                G.B.require(record["wall_ns"] >= G.MIN_NS, "scored visit shorter than 250 ms")
                grouped[(record["bytes"], record["direction"], record["arm"])].append(record)
    with (root / telemetry["raw_csv"]["path"]).open() as stream:
        gpu = list(csv.DictReader(stream, skipinitialspace=True))
    stamp = [datetime.datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for r in gpu]
    gaps = [(b - a).total_seconds() * 1000 for a, b in zip(stamp, stamp[1:])]
    G.B.require(len(gpu) >= capture["elapsed_seconds"] / 0.5 and all(0 < v <= 1000 for v in gaps),
                "missing/nonmonotonic telemetry")

    def numbers(column):
        vals = [float(r[column].split()[0]) for r in gpu if r[column].split()[0] not in ("[N/A]",)]
        G.B.require(all(math.isfinite(v) for v in vals), "nonfinite telemetry")
        return vals
    thermal = {"samples": len(gpu), "interval_median_ms": statistics.median(gaps), "interval_max_ms": max(gaps),
               "power_envelope": envelope}
    for column in ["temperature.gpu", "clocks.current.sm [MHz]", "power.draw [W]",
                   "pcie.link.gen.current", "pcie.link.width.current"]:
        vals = numbers(column)
        thermal[column] = {"min": min(vals), "max": max(vals)}
    result = []
    for (size, direction, arm), samples in sorted(grouped.items()):
        G.B.require(len(samples) == 10 and all(sum(r["order"] == o for r in samples) == 5 for o in ("ab", "ba")),
                    "requires N=5 in each order")
        result.append({"bytes": size, "direction": direction, "arm": arm, "n": 10, "copies_per_visit": samples[0]["copies"],
                       "median_us_per_copy": statistics.median(r["wall_ns"] / r["copies"] / 1000 for r in samples),
                       "median_GiB_s": statistics.median(r["completed_bytes"] / (r["wall_ns"] / 1e9) / (1 << 30) for r in samples),
                       "minimum_visit_ms": min(r["wall_ns"] / 1e6 for r in samples)})
    G.B.require(len(result) == 40, "incomplete size/direction/arm matrix")
    return {"kind": "G2-5090-development", "qualification": False, "capture_seconds": capture["elapsed_seconds"],
            "scored_samples": len(scored), "calibration_samples": len(rows) - len(scored), "thermal": thermal,
            "medians": result}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = summarize(root)
    (root / "g2-summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for m in out["medians"]:
        print(f"G2-5090 {m['bytes']:>10} {m['direction']} {m['arm']:<18} {m['median_GiB_s']:7.2f} GiB/s {m['median_us_per_copy']:10.2f} us")
