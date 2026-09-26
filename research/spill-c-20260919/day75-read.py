#!/usr/bin/env python3
"""Day 75 reader for cell `i16` (research/spill-c-20260919/DAY75.md section 2, registered before the cell's script):
integrity (one host demand sequence within each door arm), DAY64 section 5's admissibility clause, I16 against I15
(`DAY61.md` section 2's improves / regresses / flat, gen-only primary, the window beside it), the door against REF,
I16C's dispatch-clock brackets, and Part B (REF and I16 under Nsight Systems, `day72-read.py`'s window reading),
deciding nothing. Then the verdict line.

usage: day75-read.py <cell-dir> [--rig NAME]
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
d72 = load("day72_read", "day72-read.py")

N = 32
ADMISSIBLE_IQR = 0.005
ARMS = ("ref", "i15", "i16", "i16c")
DOORS = ("i15", "i16", "i16c")
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
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        run["trace_lines"] = len(trace)
        run["clocks"] = {m.group(1): {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2))}
                         for m in CLOCK.finditer(text)}
        runs[log.stem] = run
    fails = []
    if len(runs) != 40 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if r["arm"] in DOORS and (not r["fill_complete"] or r.get("physical_reads") != 0 or r["trace_lines"] == 0):
            fails.append(f"{label} fill={r['fill_complete']} physical_reads={r.get('physical_reads')}"
                         f" trace_lines={r['trace_lines']}")
        if (r["arm"] == "i16c") != bool(r["clocks"]):
            fails.append(f"{label} dispatch-clock lines {'missing' if r['arm'] == 'i16c' else 'without the flag'}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    for arm in DOORS:
        traces = {r["trace"] for r in runs.values() if r["arm"] == arm}
        if len(traces) != 1:
            fails.append(f"host demand sequences differ within {arm} ({len(traces)} distinct)")
        print(f"DAY75 host demand sequence {arm} sha256 {sorted(traces)[0][:16] if traces else '-'}"
              f" lines={sorted({r['trace_lines'] for r in runs.values() if r['arm'] == arm})}")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY75 I16 CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY75 VERDICT rig={rig} integrity=FAIL -> void")
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
    print(f"DAY75 ADMISSIBILITY rig={rig} ceiling={ADMISSIBLE_IQR} max_iqr_gen={worst['gen_s']:.4f}"
          f" max_iqr_window={worst['window_s']:.4f} failing={failing}"
          f" -> {'admissible' if admissible else 'inadmissible'}")
    step = {}
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        print(f"DAY75 {label} medians (N=10 each): " + " ".join(f"{a}={med(values(a, key)):.3f}" for a in ARMS))
        noise, diff = compare("i16", "i15", key)
        if all(d < -noise for d in diff.values()):
            name = "improves"
        elif diff["o1"] > noise and diff["o2"] > noise:
            name = "regresses"
        else:
            name = "flat"
        step[key] = name
        print(f"DAY75 STEP i16_vs_i15 {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f}"
              f" noise={noise:.4f} -> {name}")
    door = "i15" if step["gen_s"] == "regresses" else "i16"
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
        print(f"DAY75 DOOR {door}_vs_ref {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f}"
              f" noise={noise:.4f} -> {name}")
    # I16C's brackets per window token (window minus warm over the 32 window tokens), medians over its runs.
    keys = ("dispatch_ns", "prefetch_ns", "pf_demand_ns", "pf_resident_ns", "pf_retire_ns", "pf_stage_ns")
    per = {k: med([(r["clocks"]["window"].get(k, 0) - r["clocks"]["warm"].get(k, 0)) / N / 1e6
                   for r in runs.values() if r["arm"] == "i16c" and "window" in r["clocks"] and "warm" in r["clocks"]])
           for k in keys}
    print("DAY75 I16C brackets per window token (ms, medians): " + " ".join(f"{k}={v:.4f}" for k, v in per.items()))
    got = {"ref": [], "on": []}
    for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
        res, why = d72.profiled(ev, label)
        if res is None:
            print(f"DAY75 B {label}: not read ({why})")
            continue
        got[label.split("-")[1]].append(res)
        print(f"DAY75 B {label}: window_ms={res['window_ms']:.1f} gpu_busy={res['gpu_busy']:.4f}"
              f" gpu_idle={res['gpu_idle']:.4f} h2d_exposed={res['h2d_exposed']:.4f} (ms per window token)")
    if len(got["ref"]) == 2 and len(got["on"]) == 2:
        diff = {k: med([x[k] for x in got["on"]]) - med([x[k] for x in got["ref"]])
                for k in ("gpu_busy", "gpu_idle", "h2d_exposed", "kernel_sum")}
        print("DAY75 B i16_minus_ref per window token (ms, medians of two; deciding nothing): "
              + " ".join(f"{k}={v:+.4f}" for k, v in diff.items()))
    tail = (f"i16={step['gen_s']} door={door} vs_ref={out['gen_s']}"
            f" (window: i16={step['window_s']} vs_ref={out['window_s']})")
    if not admissible:
        print(f"DAY75 VERDICT rig={rig} integrity=ok -> void (inadmissible) [as read: {tail}]")
    else:
        print(f"DAY75 VERDICT rig={rig} integrity=ok {tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
