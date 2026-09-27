#!/usr/bin/env python3
"""Day 29: read a twin-gate run dir and print, per turn, both boots' reclaim-on-defer lines beside the V3 state error.

usage: verify-day29.py <run-dir>/twin27-off
Prints the verdict line verbatim, then one row per turn: V3 error, measured-boot and calibration-boot parked
sessions released (from the server logs' own lines), and the parked bytes the measured reclaim moved beyond its
prefix bytes (effective free move minus `evicted_prefix_bytes` of the settle line); a final line says whether the
first broken window's parked bytes match |V3 error| within the gate's 64 MiB slack. Read-only.
"""
import json
import re
import sys
from pathlib import Path

RE_RECLAIM = re.compile(r"reclaim-on-defer: evicted (\d+) prefix entries \+ (\d+) plain \+ (\d+) spec \+ (\d+) dspark .*; effective free (\d+)MB -> (\d+)MB")
RE_SETTLE = re.compile(r"reclaim settle \(reclaim-on-defer\): dev0 evicted_prefix_bytes=(\d+)")

out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text())
print((out / "VERDICT.txt").read_text().strip())
turns = summary["turns"]


def releases(window):
    rel, settles = [], []
    for ln in window["lines"]:
        m = RE_RECLAIM.search(ln)
        if m:
            rel.append(tuple(int(x) for x in m.groups()))
    return rel


cal_turns = summary["calibration"]["turns"]
for t, c in zip(turns, cal_turns):
    mr = releases(t["window"])
    cr = releases(c["window"])
    print(f"turn {t['turn']}: v3_error={t['v3_state_error_bytes']} measured_reclaims={mr} calibration_reclaims={cr}")
# The settle lines are not in the gate's window lines; read them from the measured log in order.
settles = [int(m.group(1)) for m in (RE_SETTLE.search(ln) for ln in (out / "measured" / "server.log").read_text().splitlines()) if m]
print(f"measured settle evicted_prefix_bytes in order: {settles}")
card = summary.get("boot", {}).get("card_at_boot") or summary.get("v3_premise", {})
print("card at measured boot:", json.dumps(summary.get("boot", {}).get("card_at_boot")))
print("card at calibration boot:", json.dumps(summary.get("calibration", {}).get("card_at_boot")))
print("v3_premise:", json.dumps(summary.get("v3_premise", {}).get("holds")))
