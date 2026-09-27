#!/usr/bin/env python3
"""Day 66 reader for cell `freq` (research/spill-c-20260919/DAY66.md section 1, registered before the cell's script):
integrity, R1 the slow door boots, R2 each door run's owner-core clock, R3 its huge-page backing and context switches,
and the verdict.

usage: day66-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
SLOW_OVER_REF = 0.030
TAIL = timedelta(seconds=3.0)
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def clock_mhz(text):
    """`scaling_cur_freq` in kHz, or a `/proc/cpuinfo` value ending in MHz; None when absent."""
    if text.endswith("MHz"):
        return float(text[:-3])
    try:
        return float(text) / 1000.0
    except ValueError:
        return None


def hms(text):
    return datetime.strptime(text[:12], "%H:%M:%S.%f")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    settings = [l for l in (ev / "topology.txt").read_text().splitlines() if "cpufreq/" in l or "amd_pstate" in l or "thp/" in l]
    print(f"DAY66 HOST rig={rig} " + " | ".join(settings))
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r*.log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        fill = FILL.search(text)
        run["fill_ms"] = float(fill.group(1)) if fill else None
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        runs[log.stem] = run
    fails = []
    if len(runs) != 40 or sorted({r["arm"] for r in runs.values()}) != ["i15", "ref"]:
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if r["arm"] == "i15" and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] == "i15"}) != 1:
        fails.append("host demand sequences differ across door runs")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY66 FREQ CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))

    ref = med([r["gen_s"] for r in runs.values() if r["arm"] == "ref"])
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        label, _, what = m.partition(" ")
        marks.setdefault(label, {})[what.split()[0]] = datetime.strptime(t[11:23], "%H:%M:%S.%f")
    clocks = []
    for line in (ev / "clock.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 5 and parts[3].startswith("run-gen-i15"):
            clocks.append((hms(parts[0]), clock_mhz(parts[4])))
    memory = []
    for line in (ev / "memory.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 6 and parts[2].startswith("run-gen-i15"):
            memory.append((hms(parts[0]), parts[3], parts[4], parts[5]))
    rows = []
    for label in sorted(l for l in runs if runs[l]["arm"] == "i15"):
        r = runs[label]
        a, b = marks[label]["start"], marks[label]["end"]
        mine = [(t, c) for t, c in clocks if a <= t <= b and c is not None]
        whole = med([c for _, c in mine])
        tail = med([c for t, c in mine if t >= b - TAIL])
        mem = [(t, h, v, i) for t, h, v, i in memory if a <= t <= b]
        huge = max((int(h) for _, h, _, _ in mem if h.isdigit()), default=None)
        last = mem[-1] if mem else None
        slow = r["gen_s"] > ref + SLOW_OVER_REF
        rows.append((label, slow, whole, huge))
        print(f"DAY66 R2R3 {label}: gen={r['gen_s']:.3f} {'slow' if slow else 'fast'} fill_ms={r['fill_ms']}"
              f" clock_mhz whole={whole:.0f} tail={tail:.0f} (samples {len(mine)})"
              f" anon_huge_kb={huge} ctxt vol={last[2] if last else '-'} invol={last[3] if last else '-'}")
    slows = [x for x in rows if x[1]]
    fasts = [x for x in rows if not x[1]]
    print(f"DAY66 R1 ref_median={ref:.3f} slow={len(slows)} fast={len(fasts)} of {len(rows)} door runs")
    if integrity != "ok":
        print(f"DAY66 FREQ VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY66 FREQ VERDICT rig={rig} integrity=ok -> not_reproduced")
        return 0
    clock_known = all(x[2] == x[2] for x in rows)
    clock = ("clock_tracks_mode" if clock_known and max(x[2] for x in slows) < min(x[2] for x in fasts)
             else "clock_does_not_track")
    huge_known = all(x[3] is not None for x in rows)
    thp = "thp_does_not_track"
    if huge_known and (max(x[3] for x in slows) < min(x[3] for x in fasts)
                       or min(x[3] for x in slows) > max(x[3] for x in fasts)):
        thp = "thp_tracks_mode"
    print(f"DAY66 FREQ VERDICT rig={rig} integrity=ok -> {clock} {thp}"
          + ("" if clock_known else " (clock absent in some runs)") + ("" if huge_known else " (huge pages absent in some runs)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
