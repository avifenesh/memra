#!/usr/bin/env python3
"""Day 68 reader for cell `eclock` (research/spill-c-20260919/DAY68.md section 1, registered before the cell's
script): integrity, R1 the slow door boots, R2 every run's phase probes and after-timing probe, R3 each door run's
effective clock and hottest Tctl, and the strict verdict with a count per field beside it.

usage: day68-read.py <cell-dir> [--rig NAME]
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
PHASES = ("start", "gate", "generate", "warm", "window")
PHASE = re.compile(r"\[cpu-probe\] phase=(\w+) cpu=(-?\d+) compute_ns=([0-9.]+)")
PROBE = re.compile(r"\[cpu-probe\] cpu=(.*)")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


def tctl(field):
    """The `Tctl` sensor in degrees from a `label=millidegrees,...` field, else the hottest sensor, else None."""
    if field == "absent":
        return None
    values = {}
    for part in field.split(","):
        if "=" in part:
            label, _, v = part.rpartition("=")
            if v.isdigit():
                values[label] = int(v) / 1000.0
    if "Tctl" in values:
        return values["Tctl"]
    return max(values.values()) if values else None


def separates(slows, fasts, key, higher):
    """The strict rule: every slow run beyond every fast one; and how many slow runs lie beyond the fast extreme."""
    s = [key(r) for r in slows]
    f = [key(r) for r in fasts]
    if any(v is None or v != v for v in s + f):
        return False, None
    if higher:
        return min(s) > max(f), sum(v > max(f) for v in s)
    return max(s) < min(f), sum(v < min(f) for v in s)


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
        run["phases"] = {m.group(1): float(m.group(3)) for m in PHASE.finditer(text)}
        probe = PROBE.search(text)
        run["probe"] = (dict((k, float(v)) for k, v in re.findall(r"(\w+)=([0-9.]+)", "cpu=" + probe.group(1)))
                        if probe else None)
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
        if any(p not in r["phases"] for p in PHASES) or r["probe"] is None:
            fails.append(f"{label} probe lines missing ({sorted(r['phases'])})")
        if r["arm"] == "i15" and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] == "i15"}) != 1:
        fails.append("host demand sequences differ across door runs")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY68 ECLOCK CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY68 ECLOCK VERDICT rig={rig} integrity=FAIL -> void")
        return 1

    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        label, _, what = m.partition(" ")
        marks.setdefault(label, {})[what.split()[0]] = datetime.strptime(t[11:23], "%H:%M:%S.%f")
    samples = []
    for line in (ev / "eclock.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 7:
            samples.append((datetime.strptime(parts[0][:12], "%H:%M:%S.%f"), parts[3], number(parts[4]),
                            number(parts[5]), tctl(parts[6])))
    ref = med([r["gen_s"] for r in runs.values() if r["arm"] == "ref"])
    for label in sorted(runs):
        r = runs[label]
        a, b = marks[label]["start"], marks[label]["end"]
        mine = [s for s in samples if a <= s[0] <= b and s[1].startswith("run-gen-p68")]
        r["eclock"] = med([s[2] for s in mine if s[2] is not None])
        r["requested"] = med([s[3] / 1000.0 for s in mine if s[3] is not None])
        temps = [s[4] for s in mine if s[4] is not None]
        r["tctl"] = max(temps) if temps else None
        r["slow"] = r["arm"] == "i15" and r["gen_s"] > ref + SLOW_OVER_REF
        ph = " ".join(f"{p}={r['phases'][p]:.3f}" for p in PHASES)
        print(f"DAY68 R2R3 {label}: gen={r['gen_s']:.3f} {'slow' if r['slow'] else 'fast'} fill_ms={r['fill_ms']}"
              f" phases {ph} | probe compute_ns={r['probe'].get('compute_ns', float('nan')):.3f}"
              f" compute2_ns={r['probe'].get('compute2_ns', float('nan')):.3f}"
              f" | eclock_mhz={r['eclock']:.0f} requested_mhz={r['requested']:.0f} tctl_max={r['tctl']}"
              f" (samples {len(mine)})")
    doors = [r for r in runs.values() if r["arm"] == "i15"]
    slows = [r for r in doors if r["slow"]]
    fasts = [r for r in doors if not r["slow"]]
    print(f"DAY68 R1 ref_median={ref:.3f} slow={len(slows)} fast={len(fasts)} of {len(doors)} door runs")
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY68 ECLOCK VERDICT rig={rig} integrity=ok -> not_reproduced")
        return 0
    fields = (
        ("generate_tracks", lambda r: r["phases"]["generate"], True),
        ("gate_tracks", lambda r: r["phases"]["gate"], True),
        ("eclock_tracks", lambda r: r["eclock"], False),
        ("tctl_tracks", lambda r: r["tctl"], True),
    )
    tracks = []
    for name, key, higher in fields:
        strict, count = separates(slows, fasts, key, higher)
        print(f"DAY68 COUNT {name}: {count} of {len(slows)} slow runs beyond the fast runs' extreme (deciding nothing)")
        if strict:
            tracks.append(name)
    print(f"DAY68 ECLOCK VERDICT rig={rig} integrity=ok -> {' '.join(tracks) if tracks else 'none_tracks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
