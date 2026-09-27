#!/usr/bin/env python3
"""Day 61 reader for cell `i11` (research/spill-c-20260919/DAY61.md section 2, registered before the improvements'
code): integrity, I11 against ON, I12 against I11, and the door against REF.

usage: day61-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
ARMS = ("ref", "on", "i11", "i12")
DOORS = ("on", "i11", "i12")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ")
SLOT = re.compile(r" slot=\d+")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        run["fill_complete"] = bool(FILL.search(text))
        # The host demand sequence: the trace lines without their stamp and slot number.
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        run["trace_lines"] = len(trace)
        runs[log.stem] = run
    fails = []
    if len(runs) != 40 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if r["arm"] in DOORS:
            if not r["fill_complete"]:
                fails.append(f"{label} no fill-complete line")
            if r.get("physical_reads") != 0:
                fails.append(f"{label} physical_reads={r.get('physical_reads')}")
            if r["trace_lines"] == 0:
                fails.append(f"{label} no host trace")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    door_traces = {r["trace"] for r in runs.values() if r["arm"] in DOORS}
    if len(door_traces) != 1:
        fails.append(f"host demand sequences differ across door runs ({len(door_traces)} distinct)")
    print(f"DAY61 host demand sequence sha256 {sorted(door_traces)[0][:16] if door_traces else '-'}"
          f" lines={sorted({r['trace_lines'] for r in runs.values() if r['arm'] in DOORS})}")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY61 I11 CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def compare(new, old, key):
        noise = max(iqr(values(new, key)), iqr(values(old, key)))
        diff = {o: med(values(new, key, o)) - med(values(old, key, o)) for o in (None, "o1", "o2")}
        return noise, diff

    verdicts = {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        medians = " ".join(f"{a}={med(values(a, key)):.3f}" for a in ARMS)
        print(f"DAY61 {label} medians (N=10 each): {medians}")
        for new, old in (("i11", "on"), ("i12", "i11")):
            noise, diff = compare(new, old, key)
            if all(d < -noise for d in diff.values()):
                name = "improves"
            elif diff["o1"] > noise and diff["o2"] > noise:
                name = "regresses"
            else:
                name = "flat"
            verdicts[(new, key)] = name
            print(f"DAY61 STEP {new}_vs_{old} {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f}"
                  f" o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}")
    door = "i11" if verdicts.get(("i12", "gen_s")) == "regresses" else "i12"
    out = {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        noise, diff = compare(door, "ref", key)
        if all(d < -noise for d in diff.values()):
            name = "beats"
        elif diff["o1"] > noise and diff["o2"] > noise:
            name = "loses"
        else:
            name = "matches"
        out[key] = name
        print(f"DAY61 DOOR {door}_vs_ref {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f}"
              f" o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}")
    if integrity != "ok":
        print(f"DAY61 VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    print(f"DAY61 VERDICT rig={rig} integrity=ok i11={verdicts[('i11', 'gen_s')]} i12={verdicts[('i12', 'gen_s')]}"
          f" door={door} vs_ref={out['gen_s']} (window: i11={verdicts[('i11', 'window_s')]}"
          f" i12={verdicts[('i12', 'window_s')]} vs_ref={out['window_s']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
