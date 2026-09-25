#!/usr/bin/env python3
"""Rescore one B3 regime with the B1-amendment read gate, offline, beside the registered result.

Usage: m1-b3-resummarize.py <regime visits dir>  -> summary-read-gate.json (summary.json kept).

Registered B3 gate: foreign device bytes (reads plus writes) at most 2% of the visit's device
bytes. It is ill-posed when a visit reads nothing from the device (the warm regime by design):
tens of MB of log and journal writes, which per-process accounting cannot attribute, then make
every visit "contaminated". This rescoring applies the read gate registered for B1 (foreign read
bytes at most max(2% of device read bytes, 1 MiB)); every other condition is unchanged. It is an
amendment made AFTER seeing the warm data and is reported as such, next to the registered verdict.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runner", HERE / "m1-spill-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main():
    d = Path(sys.argv[1])
    visits = [json.loads(p.read_text()) for p in sorted(d.glob("r*/visit.json"))]
    lock_arms = []
    for v in visits:
        c = v["contamination"]
        fr = max(0, c["device_read_bytes"] - v["proc_io"]["read_bytes"])
        c["foreign_read_bytes"] = fr
        c["read_gate_limit"] = max(0.02 * c["device_read_bytes"], 1 << 20)
        v["clean_timing_registered"] = v["clean_timing"]
        v["clean_timing"] = bool(v["telemetry_ok"] and v["thermal_ok"] and v["identity_after_ok"]
                                 and v.get("regime_ok", True) and fr <= c["read_gate_limit"])
        v["scored"] = bool(v["clean_timing"] and not v["correctness_problems"] and v["exit_code"] == 0
                           and not v["timed_out"])
        if v["arm"] not in lock_arms:
            lock_arms.append(v["arm"])
    refused = sorted({v["arm"] for v in visits if v["correctness_problems"]})
    arms = [a for a in lock_arms if a not in refused]
    s = R.verdicts([v for v in visits if v["arm"] in arms], arms, "worker16")
    s.update(gate="read gate (B1 amendment), applied after the data", refused_arms=refused, visits=len(visits))
    (d / "summary-read-gate.json").write_text(json.dumps(s, indent=1) + "\n")
    for arm, x in s["arms"].items():
        print(f"M1-VERDICT-READGATE arm={arm} vs worker16: {x['verdict']} median_ratio={x['median_ratio']} pairs={x['n_pairs']}")
    print(f"regime_scored={s['regime_scored']} contaminated={s['contaminated_visits']} refused={refused}")


if __name__ == "__main__":
    main()
