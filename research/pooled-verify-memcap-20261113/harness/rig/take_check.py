#!/usr/bin/env python3
"""take_check.py: memra#1266, the owner's rule for a timed take on the RTX 5090 (2026-11-13).
A take does not count when, in any of its arms, software thermal slowdown is active in more than
5% of the 1 Hz samples, or the card was at 86 C or hotter when the arm started. The share is
taken over the arm's samples under load (utilization >= 50%), the stricter reading: the boot and
model load idle at low clocks and cannot throttle.

  take_check.py dirs CMD          the arm directories a step command writes (one per line)
  take_check.py check TAKE DIR... judge each arm; writes DIR/take.json; exit 0 when all count
"""
import gzip
import json
import pathlib
import re
import sys

RUNS = pathlib.Path.home() / ".cache/memra-1266/runs"
SHARE_MAX = 0.05
START_MAX_C = 86


def dirs(cmd):
    out = []
    for m in re.finditer(r"(?:rtx_cell|normal_5090)\.sh (\S+) (\S+)", cmd):
        out.append(RUNS / m.group(1) / m.group(2))
    for m in re.finditer(r"bench_5090\.sh (\S+)", cmd):
        out.append(pathlib.Path(m.group(1)))
    return out


def judge(d):
    f, g = d / "telemetry.csv.gz", d / "telemetry.csv"
    text = gzip.open(f, "rt").read() if f.exists() else g.read_text() if g.exists() else ""
    rows = []
    for line in text.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        normal = p[1].endswith("MHz")  # normal cells: timestamp, clock, draw, limit, temp, util, ...
        try:
            temp = float(p[4] if normal else p[1])
            util = float(p[5].rstrip(" %"))
        except ValueError:
            continue
        rows.append((p[0], temp, util, p[7] == "Active", p[8] == "Active"))
    if not rows:
        return {"dir": d.name, "counts": False, "why": "no telemetry with throttle reasons"}
    load = [r for r in rows if r[2] >= 50]
    sw_load = sum(1 for r in load if r[3])
    share = sw_load / len(load) if load else 0.0
    start = rows[0][1]
    why = []
    if share > SHARE_MAX:
        why.append(f"sw thermal slowdown in {share:.1%} of the samples under load (> 5%)")
    if start >= START_MAX_C:
        why.append(f"started at {start:.0f} C (>= 86 C)")
    # Any second off AC in the arm's window voids it (owner, 2026-11-13); ac.txt is the bench's
    # 1 Hz log of /sys/class/power_supply/ADP0/online.
    ac = d / "ac.txt"
    off_ac = 0
    if ac.exists():
        off_ac = sum(1 for line in ac.read_text().splitlines() if line.split()[-1:] == ["0"])
        if off_ac:
            why.append(f"off AC for {off_ac} s")
    return {"dir": d.name, "start": rows[0][0], "start_temp_c": start, "max_temp_c": max(r[1] for r in rows),
            "samples": len(rows), "load_samples": len(load), "sw_thermal_load_samples": sw_load,
            "sw_thermal_load_share": round(share, 4),
            "sw_thermal_all_share": round(sum(1 for r in rows if r[3]) / len(rows), 4),
            "hw_thermal_samples": sum(1 for r in rows if r[4]), "off_ac_s": off_ac, "counts": not why,
            "why": "; ".join(why)}


if sys.argv[1] == "dirs":
    print("\n".join(map(str, dirs(sys.argv[2]))))
    sys.exit(0)
take = sys.argv[2]
ok = True
for d in map(pathlib.Path, sys.argv[3:]):
    j = judge(d)
    j["take"] = int(take)
    (d / "take.json").write_text(json.dumps(j) + "\n")
    print(json.dumps(j))
    ok = ok and j["counts"]
sys.exit(0 if ok else 1)
