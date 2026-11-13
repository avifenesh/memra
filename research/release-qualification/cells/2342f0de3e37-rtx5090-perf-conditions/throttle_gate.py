#!/usr/bin/env python3
"""Owner's 5090 heat rule for a timed take (2026-11-13): a take is bankable when the card started below
86 C and neither arm had software thermal slowdown on more than 5% of its 1 Hz samples. Joins the 1 Hz
nvidia-smi log with the perf run logs (NN-A|B-model.log, end time = file mtime, as arm_conditions.py).
A take is also void when the 1 Hz AC/profile log (ac_log.py: HH:MM:SS, ac_online, profile) shows the
rig off AC, or on another platform profile than the lease asked for, at any second of the window
(a take that overlaps any time on battery is void).
usage: throttle_gate.py THROTTLE_CSV PERF_DIR START_ISO START_TEMP_C OUT_JSON [AC_TSV PROFILE]
(exit 0 bankable, 1 not)"""
import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

throttle, perf, start, start_temp, out = sys.argv[1:6]
ac_tsv, want_profile = (sys.argv[6], sys.argv[7]) if len(sys.argv) > 7 else (None, None)
rows = list(csv.reader(open(throttle), skipinitialspace=True))
head = [h.split(" [")[0] for h in rows[0]]
samples = []
for r in rows[1:]:
    if len(r) != len(head):
        continue
    d = dict(zip(head, r))
    try:
        d["t"] = dt.datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        continue
    samples.append(d)
SW = "clocks_event_reasons.sw_thermal_slowdown"
edge = dt.datetime.fromisoformat(start).replace(tzinfo=None)
arms, runs = {}, []
for log in sorted(Path(perf).glob("*.log")):
    m = re.match(r"\d+-([AB])-(.*)\.log$", log.name)
    if not m:
        continue
    end = dt.datetime.fromtimestamp(log.stat().st_mtime)
    window = [s for s in samples if edge <= s["t"] <= end]
    sw = sum(1 for s in window if s.get(SW, "Not Active") != "Not Active")
    temps = [int(s["temperature.gpu"]) for s in window if s["temperature.gpu"].strip().isdigit()]
    runs.append({"run": log.name, "samples": len(window), "sw_thermal": sw,
                 "temp_max": max(temps) if temps else None})
    a = arms.setdefault(m.group(1), {"samples": 0, "sw_thermal": 0})
    a["samples"] += len(window)
    a["sw_thermal"] += sw
    edge = end
for a in arms.values():
    a["share"] = round(a["sw_thermal"] / a["samples"], 4) if a["samples"] else None
start_c = int(start_temp)
worst = max((a["share"] for a in arms.values() if a["share"] is not None), default=None)
power = None
if ac_tsv:
    t0 = dt.datetime.fromisoformat(start).replace(tzinfo=None)
    t1 = edge
    bad = []
    for line in Path(ac_tsv).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        try:
            t = dt.datetime.combine(t0.date(), dt.time.fromisoformat(parts[0]))
        except ValueError:
            continue
        if t0 <= t <= t1 and (parts[1] != "1" or parts[2] != want_profile):
            bad.append(line)
    power = {"profile": want_profile, "off_ac_or_profile_seconds": len(bad), "first": bad[:5]}
ok = bool(runs) and start_c < 86 and worst is not None and worst <= 0.05 and (power is None or not power["off_ac_or_profile_seconds"])
result = {"start_temp_c": start_c, "arms": arms, "worst_arm_share": worst, "power": power, "bankable": ok,
          "runs": runs}
Path(out).write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: result[k] for k in ("start_temp_c", "arms", "worst_arm_share", "power", "bankable")}))
sys.exit(0 if ok else 1)
