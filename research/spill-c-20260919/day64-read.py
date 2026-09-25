#!/usr/bin/env python3
"""Day 64 reader for cell `i15` (research/spill-c-20260919/DAY64.md section 3, registered before the cell's script):
integrity, I14 against I13 and I15 against I14, the door against REF, and I15S's stage split per window token.

usage: day64-read.py <cell-dir> [--rig NAME]
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
ARMS = ("ref", "i13", "i14", "i15", "i15s")
DOORS = ("i13", "i14", "i15", "i15s")
STAGES = re.compile(r"\[experts-via-tier\] stages phase=(\w+) (.*)")
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
        # The stage line's segments (cache | owner | fill | bank | proc), each its own key=value dict.
        run["segments"] = {}
        for m in STAGES.finditer(text):
            parts = m.group(2).split("|")
            run["segments"][m.group(1)] = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", p)) for p in parts]
        runs[log.stem] = run
    fails = []
    if len(runs) != 50 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
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
        if r["arm"] == "i15s":
            bank = r["segments"].get("window", [{}] * 4)[3] if len(r["segments"].get("window", [])) > 3 else {}
            if "stage_lookup_ns" not in bank or "warm" not in r["segments"]:
                fails.append(f"{label} no day-63 stage split")
        elif r["segments"]:
            fails.append(f"{label} a stage line without the flag")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    door_traces = {r["trace"] for r in runs.values() if r["arm"] in DOORS}
    if len(door_traces) != 1:
        fails.append(f"host demand sequences differ across door runs ({len(door_traces)} distinct)")
    print(f"DAY64 host demand sequence sha256 {sorted(door_traces)[0][:16] if door_traces else '-'}"
          f" lines={sorted({r['trace_lines'] for r in runs.values() if r['arm'] in DOORS})}")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY64 I15 CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
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
        print(f"DAY64 {label} medians (N=10 each): {medians}")
        for new, old in (("i14", "i13"), ("i15", "i14")):
            noise, diff = compare(new, old, key)
            if all(d < -noise for d in diff.values()):
                name = "improves"
            elif diff["o1"] > noise and diff["o2"] > noise:
                name = "regresses"
            else:
                name = "flat"
            verdicts[(new, key)] = name
            print(f"DAY64 STEP {new}_vs_{old} {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f}"
                  f" o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}")
    door = "i15"
    if verdicts.get(("i15", "gen_s")) == "regresses":
        door = "i13" if verdicts.get(("i14", "gen_s")) == "regresses" else "i14"
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
        print(f"DAY64 DOOR {door}_vs_ref {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f}"
              f" o2={diff['o2']:+.4f} noise={noise:.4f} -> {name}")
    # I15S's split per window token (window minus warm over the 32 window tokens), medians over its runs.
    split = [r["segments"] for r in runs.values() if r["arm"] == "i15s" and "warm" in r["segments"]]
    names = ("cache", "owner", "fill", "bank")
    for index, name in enumerate(names):
        keys = sorted({k for sg in split for k in sg["window"][index]}) if split else []
        cells = []
        for k in keys:
            v = med([(sg["window"][index].get(k, 0) - sg["warm"][index].get(k, 0)) / N for sg in split])
            cells.append(f"{k}={v / 1e6:.4f}ms" if k.endswith("_ns") else f"{k}={v:.1f}")
        print(f"DAY64 SPLIT {name} per window token (N={len(split)} runs): " + " ".join(cells))
    if integrity != "ok":
        print(f"DAY64 VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    print(f"DAY64 VERDICT rig={rig} integrity=ok i14={verdicts[('i14', 'gen_s')]} i15={verdicts[('i15', 'gen_s')]}"
          f" door={door} vs_ref={out['gen_s']} (window: i14={verdicts[('i14', 'window_s')]}"
          f" i15={verdicts[('i15', 'window_s')]} vs_ref={out['window_s']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
