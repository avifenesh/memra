#!/usr/bin/env python3
"""Day 67 reader for cell `probe` (research/spill-c-20260919/DAY67.md section 1, registered before the cell's script):
integrity, R1 the slow door boots, R2 every run's `[cpu-probe]` line, R3 each door run's largest migration count, and
the verdict.

usage: day67-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
SLOW_OVER_REF = 0.030
PROBE = re.compile(r"\[cpu-probe\] (.*)")
FIELDS = ("compute_ns", "l1_ns", "l2_ns", "dram_ns")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r*.log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        probe = PROBE.search(text)
        run["probe"] = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([0-9.]+)", probe.group(1))) if probe else None
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
        if r["probe"] is None or any(k not in r["probe"] for k in FIELDS):
            fails.append(f"{label} no probe line")
        if r["arm"] == "i15" and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] == "i15"}) != 1:
        fails.append("host demand sequences differ across door runs")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY67 PROBE CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY67 PROBE VERDICT rig={rig} integrity=FAIL -> void")
        return 1

    ref = med([r["gen_s"] for r in runs.values() if r["arm"] == "ref"])
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        label, _, what = m.partition(" ")
        marks.setdefault(label, {})[what.split()[0]] = datetime.strptime(t[11:23], "%H:%M:%S.%f")
    sched = []
    for line in (ev / "sched.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 6 and parts[3].isdigit():
            sched.append((datetime.strptime(parts[0][:12], "%H:%M:%S.%f"), parts[2], int(parts[3])))
    for label in sorted(runs):
        r = runs[label]
        a, b = marks[label]["start"], marks[label]["end"]
        mig = [m for t, comm, m in sched if a <= t <= b and comm.startswith("run-gen")]
        r["migrations"] = max(mig) if mig else None
        r["slow"] = r["arm"] == "i15" and r["gen_s"] > ref + SLOW_OVER_REF
        p = r["probe"]
        print(f"DAY67 R2 {label}: gen={r['gen_s']:.3f} {'slow' if r['slow'] else 'fast'}"
              f" compute_ns={p['compute_ns']:.3f} l1_ns={p['l1_ns']:.3f} l2_ns={p['l2_ns']:.3f}"
              f" dram_ns={p['dram_ns']:.3f} compute2_ns={p.get('compute2_ns', float('nan')):.3f}"
              f" cpu={int(p.get('cpu', -1))}/{int(p.get('cpu_after', -1))} fill_ms={r['fill_ms']}"
              f" migrations={r['migrations']}")
    doors = [r for r in runs.values() if r["arm"] == "i15"]
    slows = [r for r in doors if r["slow"]]
    fasts = [r for r in doors if not r["slow"]]
    print(f"DAY67 R1 ref_median={ref:.3f} slow={len(slows)} fast={len(fasts)} of {len(doors)} door runs")
    refs = [r for r in runs.values() if r["arm"] == "ref"]
    for k in FIELDS:
        v = [r["probe"][k] for r in refs]
        print(f"DAY67 REF {k}: min={min(v):.3f} median={med(v):.3f} max={max(v):.3f} (N={len(v)})")
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY67 PROBE VERDICT rig={rig} integrity=ok -> not_reproduced")
        return 0
    tracks = []
    for k in FIELDS:
        if min(r["probe"][k] for r in slows) > max(r["probe"][k] for r in fasts):
            tracks.append(f"{k}_tracks")
    if all(r["migrations"] is not None for r in doors) and min(r["migrations"] for r in slows) > max(
            r["migrations"] for r in fasts):
        tracks.append("migrations_tracks")
    print(f"DAY67 PROBE VERDICT rig={rig} integrity=ok -> {' '.join(tracks) if tracks else 'none_tracks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
