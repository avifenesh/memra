#!/usr/bin/env python3
"""Summarize raw exact-oracle and paired ABBA component receipts."""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
raw = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
result = {"weights": {"2": "48/162", "4": "103/162", "7": "11/162"}, "oracle": {}, "bench": {}}
all_pass = True
for t in (2, 4, 7):
    p = raw / f"t{t}/oracle.tsv"
    rows = list(csv.DictReader(p.open(), delimiter="\t")) if p.exists() else []
    passed = len(rows) == 42 * t * 4 and all(r["status"] == "PASS" and int(r["bit_diffs"]) == 0 and float(r["mean_delta"]) == float(r["max_delta"]) == 0 for r in rows)
    all_pass &= passed
    result["oracle"][str(t)] = {"rows": len(rows), "expected_rows": 42*t*4, "layers": len({r["layer"] for r in rows}), "pass": passed, "failures": [r for r in rows if r["status"] != "PASS"]}
per_layer = []
for t in (2, 4, 7):
    p = raw / f"t{t}/bench.tsv"
    if not p.exists():
        continue
    if not all_pass:
        raise SystemExit("timing exists without all three complete oracles")
    rows = list(csv.DictReader(p.open(), delimiter="\t"))
    if len(rows) != 42*5*4:
        raise SystemExit(f"incomplete timings t={t}: {len(rows)}")
    totals = {arm: [0.0]*5 for arm in ("current", "dual")}
    for il in range(3,45):
        lr = [r for r in rows if int(r["layer"]) == il]
        for arm in totals:
            for b in range(5):
                values = [float(r["us"]) for r in lr if r["arm"] == arm and int(r["block"]) == b]
                if len(values) != 2:
                    raise SystemExit("unbalanced ABBA block")
                totals[arm][b] += statistics.mean(values)/1000
        a = statistics.median(float(r["us"]) for r in lr if r["arm"] == "current")
        b = statistics.median(float(r["us"]) for r in lr if r["arm"] == "dual")
        per_layer.append({"t": t, "layer": il, "current_us": a, "dual_us": b, "saving_us": a-b})
    diffs = [a-b for a,b in zip(totals["current"],totals["dual"])]
    result["bench"][str(t)] = {"current_ms": statistics.median(totals["current"]), "dual_ms": statistics.median(totals["dual"]), "saving_ms": statistics.median(diffs), "paired_block_savings_ms": diffs, "block_totals_ms": totals}
if len(result["bench"]) == 3:
    weights = {"2":48/162,"4":103/162,"7":11/162}
    result["weighted_saving_ms"] = sum(weights[t]*r["saving_ms"] for t,r in result["bench"].items())
    result["weighted_eligible_saving_ms_control_zero"] = sum(weights[t]*result["bench"][t]["saving_ms"] for t in ("2","4"))
    result["verdict"] = "CANDIDATE" if min(result["weighted_saving_ms"],result["weighted_eligible_saving_ms_control_zero"]) >= 0.5 else "NEGATIVE"
else:
    result["verdict"] = "ORACLE_PASS_TIMING_PENDING" if all_pass else "ORACLE_INCOMPLETE_OR_FAILED"
if per_layer:
    with (out/"per-layer-timing.tsv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=per_layer[0].keys(),delimiter="\t");w.writeheader();w.writerows(per_layer)
(out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
manifest={str(p.relative_to(raw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.rglob("*")) if p.is_file() and p.name!="f16-source.tar.gz"}
(out/"raw-sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(result,indent=2))
