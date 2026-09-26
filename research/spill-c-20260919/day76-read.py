#!/usr/bin/env python3
"""Day 76 reader for cell `chunk` (research/spill-c-20260919/DAY76.md section 1, registered before the cell's
scripts): integrity, per run the window state (APERF cycles per chain step where the counters run, else the phase
probe's nanoseconds per step) and whether its span had compaction, per arm the counts and medians, and the verdict.
The run parsing and the sampler rows come from DAY71's reader, the state helpers from DAY74's.

usage: day76-read.py <cell-dir> [--rig NAME]
"""
import importlib.util
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d71 = load("day71_read", "day71-read.py")
d74 = load("day74_read", "day74-read.py")

N = 32
ARMS = ("refi", "di", "dci")
SLOW = 1.25
POOL = re.compile(r"\[experts-via-tier\] host pinned pool bytes=(\d+) .*?(?: chunk_bytes=(\d+) allocations=(\d+))?$", re.M)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    inducer = (ev / "inducer.txt").read_text().strip() if (ev / "inducer.txt").exists() else "inducer.txt missing"
    print(f"DAY76 INDUCER rig={rig} {inducer.splitlines()[0]}")
    if "induce=1" not in inducer:
        print(f"DAY76 CHUNK VERDICT rig={rig} -> not_run")
        return 0
    counters_on = (ev / "counters.txt").exists() and \
        (ev / "counters.txt").read_text().strip() == "counters=--cpu-probe-counters"
    state = d74.cycles if counters_on else d74.wall_ns
    unit = "cycles" if counters_on else "ns_per_step"
    runs = d71.parse_runs(ev)
    fails = []
    if len(runs) != 24 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    pools = {}
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N or r.get("window_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')} window_n={r.get('window_n')}")
        if any(p not in r["phases"] for p in ("gate", "window")):
            fails.append(f"{label} phase lines missing")
        if r["arm"] in ("di", "dci"):
            if r["fill_ms"] is None or r.get("physical_reads") != 0:
                fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
            m = POOL.search((ev / f"{label}.log").read_text(errors="replace"))
            if not m or (r["arm"] == "dci") != (m.group(2) is not None):
                fails.append(f"{label} pool line {'missing' if not m else 'not the arm'}s")
            elif r["arm"] == "dci":
                pools[label] = (int(m.group(1)), int(m.group(3)))
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
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
        a, b = d71.at(s["V"], r["phases"]["gate"]["t"], True), d71.at(s["V"], r["phases"]["window"]["t"], False)
        if a is None or b is None or a[0] < s0 or b[0] > s1:
            fails.append(f"{label} vmstat rows do not bracket its span")
            continue
        spans[label] = (a[0], b[0])
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY76 CHUNK CHECKS rig={rig} runs={len(runs)} state={unit} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY76 CHUNK VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    if pools:
        print(f"DAY76 POOL dci: bytes={sorted({b for b, _ in pools.values()})} allocations="
              f"{sorted({n for _, n in pools.values()})}")
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
        r["cw"] = state(r, "window")
        r["compacted"] = vm["compact_isolated"] > 0
        r["iso"], r["fail"] = vm["compact_isolated"] / wall, vm["pgmigrate_fail"] / wall
        per[r["arm"]].append(r)
    ref_state = med([r["cw"] for r in per["refi"] if r["cw"] is not None])
    for arm in ARMS:
        for r in per[arm]:
            r["slow"] = r["cw"] is not None and r["cw"] >= SLOW * ref_state
    for label in sorted(runs, key=lambda k: marks[k]["start"]):
        r = runs[label]
        print(f"DAY76 R {label}: gen={r['gen_s']:.3f} window={r['window_s']:.3f} window_{unit}="
              f"{'not_read' if r['cw'] is None else format(r['cw'], '.3f')} {'slow' if r['slow'] else 'fast'}"
              f" compacted={'yes' if r['compacted'] else 'no'} compact_isolated={r['iso']:.0f}/s"
              f" migrate_fail={r['fail']:.0f}/s")
    count = {}
    for arm in ARMS:
        rs = per[arm]
        count[arm] = (sum(r["compacted"] for r in rs), sum(r["slow"] for r in rs))
        print(f"DAY76 ARM rig={rig} {arm}: compacted={count[arm][0]} of {len(rs)} slow={count[arm][1]} of {len(rs)}"
              f" window_{unit} median={med([r['cw'] for r in rs if r['cw'] is not None]):.3f}"
              f" gen median={med([r['gen_s'] for r in rs]):.3f} window median={med([r['window_s'] for r in rs]):.3f}")
    if count["di"][0] < 3:
        verdict = "not_reproduced"
    elif count["dci"][0] == 0 and count["dci"][1] == 0:
        verdict = "chunk_clears"
    elif count["dci"][0] >= 3:
        verdict = "chunk_does_not"
    else:
        verdict = "undecided"
    print(f"DAY76 CHUNK VERDICT rig={rig} integrity=ok -> {verdict} (di compacted {count['di'][0]} of 8, dci"
          f" {count['dci'][0]} of 8, refi {count['refi'][0]} of 8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
