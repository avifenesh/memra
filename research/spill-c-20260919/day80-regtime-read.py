#!/usr/bin/env python3
"""Day 80 reader for cell `regtime` (research/spill-c-20260919/DAY80.md section 1, registered before the cell's
script): integrity (one host demand sequence within each door arm, and DR's equal to D's), DAY64 section 5's
admissibility clause, DR against D (`DAY61.md` section 2's improves / regresses / flat, gen-only primary, the window
beside it), DR against REF, and the verdict line.

usage: day80-regtime-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d40 = load("day40_attrib", "day40-attrib.py")

N = 32
ADMISSIBLE_IQR = 0.005
ARMS = ("ref", "d", "dr")
DOORS = ("d", "dr")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")
CLOCK = re.compile(r"\[moe-cache\] dispatch-clock phase=(\w+) (.*)")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


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
        run["registered"] = "host pinned pool" in text and any(
            l.endswith(" registered") for l in text.splitlines() if "host pinned pool" in l)
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        run["trace_lines"] = len(trace)
        runs[log.stem] = run
    fails = []
    if len(runs) != 30 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if r["arm"] in DOORS and (not r["fill_complete"] or r.get("physical_reads") != 0 or r["trace_lines"] == 0):
            fails.append(f"{label} fill={r['fill_complete']} physical_reads={r.get('physical_reads')}")
        if r["arm"] in DOORS and r["registered"] != (r["arm"] == "dr"):
            fails.append(f"{label} pool line not the arm's")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    for arm in DOORS:
        traces = {r["trace"] for r in runs.values() if r["arm"] == arm}
        if len(traces) != 1:
            fails.append(f"host demand sequences differ within {arm} ({len(traces)} distinct)")
        print(f"DAY80 host demand sequence {arm} sha256 {sorted(traces)[0][:16] if traces else '-'}")
    if {r["trace"] for r in runs.values() if r["arm"] == "dr"} != {r["trace"] for r in runs.values() if r["arm"] == "d"}:
        fails.append("DR's host demand sequence differs from D's")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY80 REGTIME CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY80 REGTIME VERDICT rig={rig} integrity=FAIL -> void")
        return 1

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    def compare(new, old, key):
        noise = max(iqr(values(new, key)), iqr(values(old, key)))
        diff = {o: med(values(new, key, o)) - med(values(old, key, o)) for o in (None, "o1", "o2")}
        return noise, diff

    failing = [f"{a}:{k}={iqr(values(a, k)):.4f}" for a in ARMS for k in ("gen_s", "window_s")
               if not iqr(values(a, k)) <= ADMISSIBLE_IQR]
    admissible = not failing
    worst = {k: max(iqr(values(a, k)) for a in ARMS) for k in ("gen_s", "window_s")}
    print(f"DAY80 ADMISSIBILITY rig={rig} ceiling={ADMISSIBLE_IQR} max_iqr_gen={worst['gen_s']:.4f}"
          f" max_iqr_window={worst['window_s']:.4f} failing={failing}"
          f" -> {'admissible' if admissible else 'inadmissible'}")
    step, out = {}, {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        print(f"DAY80 {label} medians (N=10 each): " + " ".join(f"{a}={med(values(a, key)):.3f}" for a in ARMS))
        noise, diff = compare("dr", "d", key)
        if all(d < -noise for d in diff.values()):
            step[key] = "improves"
        elif diff["o1"] > noise and diff["o2"] > noise:
            step[key] = "regresses"
        else:
            step[key] = "flat"
        print(f"DAY80 STEP dr_vs_d {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f}"
              f" noise={noise:.4f} -> {step[key]}")
        noise, diff = compare("dr", "ref", key)
        if all(d < -noise for d in diff.values()):
            out[key] = "beats"
        elif diff["o1"] > noise and diff["o2"] > noise:
            out[key] = "loses"
        else:
            out[key] = "matches"
        print(f"DAY80 DOOR dr_vs_ref {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f}"
              f" noise={noise:.4f} -> {out[key]}")
    tail = (f"dr={step['gen_s']} vs_ref={out['gen_s']} (window: dr={step['window_s']} vs_ref={out['window_s']})")
    if not admissible:
        print(f"DAY80 REGTIME VERDICT rig={rig} integrity=ok -> void (inadmissible) [as read: {tail}]")
    else:
        print(f"DAY80 REGTIME VERDICT rig={rig} integrity=ok {tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
