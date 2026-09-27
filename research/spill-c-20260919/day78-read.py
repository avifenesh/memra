#!/usr/bin/env python3
"""Day 78 reader for cell `pages` (research/spill-c-20260919/DAY78.md section 1, registered before the cell's
scripts): integrity (the timed runs and the census runs), per timed run the window state and the span's compaction
with `fail_heavy` (isolated > 0 and failed / isolated >= 0.5), per arm the counts, the verdict, and the census
tables beside it (what the process could read, and per census run its largest mappings), deciding nothing.

usage: day78-read.py <cell-dir> [--rig NAME]
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
d40 = load("day40_attrib", "day40-attrib.py")
d74 = load("day74_read", "day74-read.py")

N = 32
ARMS = ("refi", "di", "dpi")
FAIL_HEAVY = 0.5
SLOW = 1.25
POOL = re.compile(r"\[experts-via-tier\] host pinned pool bytes=(\d+) .*$", re.M)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    inducer = (ev / "inducer.txt").read_text().strip() if (ev / "inducer.txt").exists() else "inducer.txt missing"
    print(f"DAY78 INDUCER rig={rig} {inducer.splitlines()[0]}")
    if "induce=1" not in inducer:
        print(f"DAY78 PAGES VERDICT rig={rig} -> not_run")
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
        if r["arm"] in ("di", "dpi"):
            if r["fill_ms"] is None or r.get("physical_reads") != 0:
                fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
            m = POOL.search((ev / f"{label}.log").read_text(errors="replace"))
            if not m or (r["arm"] == "dpi") != m.group(0).endswith(" pageable"):
                fails.append(f"{label} pool line {'missing' if not m else 'not the arm'}s")
            else:
                pools[label] = int(m.group(1))
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    # The census runs: exit 0, MATCH, the same tape, a census file that finished.
    census = {}
    for log in sorted(ev.glob("c[12]-*-r1.log")):
        r = d40.parse_run(log)
        r["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        census[log.stem] = r
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH") or r.get("tokens") != next(iter(runs.values())).get("tokens"):
            fails.append(f"{log.stem} exit={r['exit']} match={r.get('match')}")
        pages = ev / f"{log.stem}.pages.tsv"
        if not pages.exists() or "\tdone" not in pages.read_text():
            fails.append(f"{log.stem} page census missing or unfinished")
    if len(census) != 6:
        fails.append(f"census runs={len(census)}")
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
    print(f"DAY78 PAGES CHECKS rig={rig} runs={len(runs)} state={unit} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY78 PAGES VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    if pools:
        print(f"DAY78 POOL door arms: bytes={sorted(set(pools.values()))}")
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
        r["fail_heavy"] = r["compacted"] and vm["pgmigrate_fail"] / vm["compact_isolated"] >= FAIL_HEAVY
        r["iso"], r["fail"] = vm["compact_isolated"] / wall, vm["pgmigrate_fail"] / wall
        per[r["arm"]].append(r)
    ref_state = med([r["cw"] for r in per["refi"] if r["cw"] is not None])
    for arm in ARMS:
        for r in per[arm]:
            r["slow"] = r["cw"] is not None and r["cw"] >= SLOW * ref_state
    for label in sorted(runs, key=lambda k: marks[k]["start"]):
        r = runs[label]
        print(f"DAY78 R {label}: gen={r['gen_s']:.3f} window={r['window_s']:.3f} window_{unit}="
              f"{'not_read' if r['cw'] is None else format(r['cw'], '.3f')} {'slow' if r['slow'] else 'fast'}"
              f" compacted={'yes' if r['compacted'] else 'no'} fail_heavy={'yes' if r['fail_heavy'] else 'no'}"
              f" compact_isolated={r['iso']:.0f}/s migrate_fail={r['fail']:.0f}/s")
    count = {}
    for arm in ARMS:
        rs = per[arm]
        count[arm] = (sum(r["fail_heavy"] for r in rs), sum(r["slow"] for r in rs))
        print(f"DAY78 ARM rig={rig} {arm}: fail_heavy={count[arm][0]} of {len(rs)} slow={count[arm][1]} of {len(rs)}"
              f" compacted={sum(r['compacted'] for r in rs)} of {len(rs)}"
              f" window_{unit} median={med([r['cw'] for r in rs if r['cw'] is not None]):.3f}"
              f" gen median={med([r['gen_s'] for r in rs]):.3f} window median={med([r['window_s'] for r in rs]):.3f}")
    if count["di"][0] < 3:
        verdict = "not_reproduced"
    elif count["dpi"][0] == 0:
        verdict = "pool_draws"
    elif count["dpi"][0] >= 3:
        verdict = "pool_does_not"
    else:
        verdict = "undecided"
    # The census, deciding nothing: what the process could read, and each census run's largest mappings.
    for label in sorted(census):
        text = (ev / f"{label}.pages.tsv").read_text().splitlines()
        access = next((l.split("\t", 2)[2] for l in text if "\taccess\t" in l), "-")
        maps = [l.split("\t") for l in text if "\tmap\t0\t" in l]
        big = sorted(maps, key=lambda f: -int(f[7].split("=")[1]))[:4]
        moved = [l.split("\t")[4] for l in text if "\tmoved\t" in l and not l.endswith("changed_frames=0")]
        print(f"DAY78 CENSUS {label}: {access}")
        for f in big:
            print(f"DAY78 CENSUS {label} map: {f[3]} {f[4]} {f[5][:48]} {f[6]} {f[7]} {f[9]} {f[10]} {f[11]} {f[12]}"
                  f" {f[14]} {f[15][:60]} {f[16]} {f[17][:80]}")
        print(f"DAY78 CENSUS {label}: mappings with frames changed between the passes: {len(moved)}"
              + (f" ({', '.join(moved[:4])})" if moved else ""))
    print(f"DAY78 PAGES VERDICT rig={rig} integrity=ok -> {verdict} (fail_heavy: di {count['di'][0]} of 8,"
          f" dpi {count['dpi'][0]} of 8, refi {count['refi'][0]} of 8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
