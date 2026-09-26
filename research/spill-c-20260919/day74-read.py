#!/usr/bin/env python3
"""Day 74 reader for cell `induce` (research/spill-c-20260919/DAY74.md sections 1 and 1a, registered before the cell's
scripts): integrity, R1 each run's cycles per chain step at `gate` and `window` and whether its span had compaction,
R2 per arm the medians, R3 per arm the timing medians (deciding nothing), and the verdict. DAY71's reader supplies the
run parsing, the sampler rows and the counters.

With --induce-b (DAY74 section 4): where the counters check reads `counters=unavailable`, the state is the phase
probe's wall nanoseconds per chain step (`compute_ns`) and the counters are not an integrity condition.

usage: day74-read.py <cell-dir> [--rig NAME] [--induce-b]
"""
import importlib.util
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day71_read", HERE / "day71-read.py")
d71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d71)

N = 32
STEPS = 1 << 20
ARMS = ("ref", "refi", "i15", "i15i")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def wall_ns(r, phase):
    p = r["phases"].get(phase)
    return p["ns"] if p else None


def cycles(r, phase):
    p = r["phases"].get(phase)
    c = p and p["counters"]
    if not c or c["cpu_after"] != p["cpu"] or c["aperf"] <= 0:
        return None
    return c["aperf"] / STEPS


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    inducer = (ev / "inducer.txt").read_text().strip() if (ev / "inducer.txt").exists() else "inducer.txt missing"
    print(f"DAY74 INDUCER rig={rig} {inducer.splitlines()[0]}")
    if "induce=1" not in inducer:
        print(f"DAY74 INDUCE VERDICT rig={rig} -> not_run")
        return 0
    counters_on = (ev / "counters.txt").exists() and \
        (ev / "counters.txt").read_text().strip() == "counters=--cpu-probe-counters"
    check = (ev / "counters-check.txt").read_text() if (ev / "counters-check.txt").exists() else ""
    wall_mode = "--induce-b" in sys.argv and not counters_on and "counters=unavailable" in check
    unit = "ns_per_step" if wall_mode else "cycles"
    state = wall_ns if wall_mode else cycles
    print(f"DAY74 STATE rig={rig} reading={unit}")
    runs = d71.parse_runs(ev)
    fails = []
    if len(runs) != 24 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if any(p not in r["phases"] for p in ("gate", "window")):
            fails.append(f"{label} phase lines missing")
        if r["arm"] in ("i15", "i15i") and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if not counters_on and not wall_mode:
        fails.append("the counters check did not pass (the cycle readings need it)")
    s = d71.load_sampler(ev)
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        lab, _, what = m.partition(" ")
        marks.setdefault(lab, {})[what.split()[0]] = d71.hms(t[11:23])
    spans = {}
    for label, r in runs.items():
        if label not in marks or any(p not in r["phases"] for p in ("gate", "window")):
            continue
        s0, s1 = marks[label]["start"], marks[label]["end"]
        g, w = r["phases"]["gate"]["t"], r["phases"]["window"]["t"]
        a, b = d71.at(s["V"], g, True), d71.at(s["V"], w, False)
        if a is None or b is None or a[0] < s0 or b[0] > s1:
            fails.append(f"{label} vmstat rows do not bracket its span")
            continue
        spans[label] = (a[0], b[0])
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY74 INDUCE CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY74 INDUCE VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    per = defaultdict(list)
    for label in sorted(runs, key=lambda k: marks[k]["start"]):
        r = runs[label]
        t0, t1 = spans[label]
        wall = (t1 - t0).total_seconds()
        vm = defaultdict(int)
        for t, d in s["V"]:
            if t0 < t <= t1:
                for k, x in d.items():
                    vm[k] += x
        r["cg"], r["cw"] = state(r, "gate"), state(r, "window")
        r["iso"], r["fail"] = vm["compact_isolated"] / wall, vm["pgmigrate_fail"] / wall
        r["compacted"] = vm["compact_isolated"] > 0
        per[r["arm"]].append(r)

        def show(x):
            return "not_read" if x is None else f"{x:.3f}"

        print(f"DAY74 R1 {label}: gen={r['gen_s']:.3f} window={r['window_s']:.3f} gate_{unit}={show(r['cg'])}"
              f" window_{unit}={show(r['cw'])} compacted={'yes' if r['compacted'] else 'no'}"
              f" compact_isolated={r['iso']:.0f}/s migrate_fail={r['fail']:.0f}/s")
    for arm in ARMS:
        rs = per[arm]
        cw = [r["cw"] for r in rs if r["cw"] is not None]
        print(f"DAY74 R2 rig={rig} {arm}: window_{unit} median={med(cw):.3f} (N={len(cw)})"
              f" compacted={sum(r['compacted'] for r in rs)} of {len(rs)}"
              f" compact_isolated median={med([r['iso'] for r in rs]):.0f}/s"
              f" migrate_fail median={med([r['fail'] for r in rs]):.0f}/s"
              f" | R3 gen median={med([r['gen_s'] for r in rs]):.3f} window median={med([r['window_s'] for r in rs]):.3f}")
    induced = {arm: sum(r["compacted"] for r in per[arm]) for arm in ("refi", "i15i")}
    if induced["refi"] < 5 or induced["i15i"] < 5:
        print(f"DAY74 INDUCE VERDICT rig={rig} integrity=ok -> not_induced (compaction in {induced['refi']} of 6 REF+I"
              f" and {induced['i15i']} of 6 door+I spans)")
        return 0
    fields = []
    for name, plain, ind in (("door_slows", "i15", "i15i"), ("ref_slows", "ref", "refi")):
        a = [r["cw"] for r in per[ind]]
        b = [r["cw"] for r in per[plain]]
        if any(x is None for x in a + b):
            print(f"DAY74 COUNT {name}: not read in every run (deciding nothing)")
            continue
        print(f"DAY74 COUNT {name}: {sum(x > max(b) for x in a)} of {len(a)} induced runs above every uninduced run"
              " (deciding nothing)")
        if min(a) > max(b):
            fields.append(name)
    print(f"DAY74 INDUCE VERDICT rig={rig} integrity=ok -> {' '.join(fields) if fields else 'neither'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
