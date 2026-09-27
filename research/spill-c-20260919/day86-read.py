#!/usr/bin/env python3
"""Day 86 reader for cell `slow86` (research/spill-c-20260919/DAY86.md section 1, registered before this script):
integrity (every run exits 0 with MATCH, one tape, one host demand sequence across the door arms), then per run the
gen-only seconds, the slow mark (gen-only above REF's median by more than 0.030 s, DAY80 section 3's) and the
compaction deltas from the run's boundary snapshots; per arm the slow boots in each order; the rule's verdicts
(`not_reproduced`, the one-sided Fisher exact tests of I21 against I20 and I22 against I21, the registered-pool
control); beside them, deciding nothing, the slow boots against compaction and by position in the round.

usage: day86-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import math
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
NATURAL_SLOW = 0.030
ALPHA = 0.05
SLOT = re.compile(r" slot=\d+")
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
COUNTERS = ("compact_isolated", "compact_migrate_scanned", "compact_fail", "compact_stall", "pgmigrate_fail",
            "pgmigrate_success", "thp_fault_alloc", "thp_fault_fallback")


def vmstat(snap):
    """The snapshot's /proc/vmstat counters (the section after `## vmstat`)."""
    out, inside = {}, False
    for line in snap.read_text(errors="replace").splitlines():
        if line.startswith("## "):
            inside = line == "## vmstat"
            continue
        if inside:
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                out[parts[0]] = int(parts[1])
    return out


def fisher_greater(a_slow, a_n, b_slow, b_n):
    """One-sided Fisher exact p that arm a has more slow boots than arm b (hypergeometric upper tail)."""
    total_slow, n = a_slow + b_slow, a_n + b_n

    def pmf(k):
        return math.comb(a_n, k) * math.comb(b_n, total_slow - k) / math.comb(n, total_slow)

    return sum(pmf(k) for k in range(a_slow, min(a_n, total_slow) + 1) if 0 <= total_slow - k <= b_n)


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        order, arm, rep = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"], run["rep"] = order, arm, rep
        text = log.read_text(errors="replace")
        run["fill_complete"] = bool(FILL.search(text))
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        run["trace_lines"] = len(trace)
        before, after = vmstat(ev / f"{log.stem}.before.snap"), vmstat(ev / f"{log.stem}.after.snap")
        run["compaction"] = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS if k in before and k in after}
        runs[log.stem] = run
    arms = sorted({r["arm"] for r in runs.values()}, key=["ref", "i20", "i21", "i22", "i22r"].index)
    doors = [a for a in arms if a != "ref"]
    fails = []
    if arms not in (["ref", "i20", "i21"], ["ref", "i20", "i21", "i22", "i22r"]):
        fails.append(f"arms={arms}")
    for arm in arms:
        count = sum(r["arm"] == arm for r in runs.values())
        if count != 10:
            fails.append(f"{arm} runs={count}")
    for label, r in runs.items():
        if r["exit"] != 0 or not r.get("match", "").endswith("MATCH"):
            fails.append(f"{label} exit={r['exit']} match={r.get('match')}")
        if r.get("gen_n") != N:
            fails.append(f"{label} gen_n={r.get('gen_n')}")
        if r["arm"] in doors and (not r["fill_complete"] or r.get("physical_reads") != 0 or r["trace_lines"] == 0):
            fails.append(f"{label} fill={r['fill_complete']} physical_reads={r.get('physical_reads')}"
                         f" trace_lines={r['trace_lines']}")
        if not r["compaction"]:
            fails.append(f"{label} no vmstat counters in its snapshots")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] in doors}) != 1:
        fails.append("the door arms' host demand sequences differ")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY86 SLOW CHECKS rig={rig} runs={len(runs)} arms={','.join(arms)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY86 VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    ref_median = statistics.median(r["gen_s"] for r in runs.values() if r["arm"] == "ref")
    mark = ref_median + NATURAL_SLOW
    for label in sorted(runs, key=lambda s: (s.split("-")[0], s.split("-")[2], arms.index(s.split("-")[1]))):
        r = runs[label]
        r["slow"] = r["gen_s"] > mark
        c = r["compaction"]
        print(f"DAY86 RUN {label} gen_s={r['gen_s']:.3f} slow={'yes' if r['slow'] else 'no'} "
              + " ".join(f"{k}={c.get(k, '-')}" for k in COUNTERS))
    slow = {a: sum(r["slow"] for r in runs.values() if r["arm"] == a) for a in arms}
    by_order = {a: {o: sum(r["slow"] for r in runs.values() if r["arm"] == a and r["order"] == o) for o in ("o1", "o2")}
                for a in arms}
    print(f"DAY86 SLOW rig={rig} ref_median={ref_median:.3f} mark={mark:.3f} "
          + " ".join(f"{a}={slow[a]}/10 (o1 {by_order[a]['o1']}, o2 {by_order[a]['o2']})" for a in arms))
    default_slow = sum(slow[a] for a in ("i20", "i21", "i22") if a in slow)
    if default_slow < 3:
        print(f"DAY86 VERDICT rig={rig} integrity=ok -> not_reproduced (default-pool door arms slow={default_slow})")
        return 0
    p21 = fisher_greater(slow["i21"], 10, slow["i20"], 10)
    v21 = "i21_more_slow" if slow["i21"] > slow["i20"] and p21 < ALPHA else "i21_not_shown"
    print(f"DAY86 TEST i21_vs_i20 slow {slow['i21']}/10 against {slow['i20']}/10 one-sided p={p21:.4f} -> {v21}")
    verdicts = [v21]
    if "i22" in slow:
        p22 = fisher_greater(slow["i22"], 10, slow["i21"], 10)
        v22 = "i22_more_slow" if slow["i22"] > slow["i21"] and p22 < ALPHA else "i22_not_shown"
        print(f"DAY86 TEST i22_vs_i21 slow {slow['i22']}/10 against {slow['i21']}/10 one-sided p={p22:.4f} -> {v22}")
        control = ("registered_clears" if slow["i22r"] == 0
                   else "registered_does_not" if slow["i22r"] >= 2 else "undecided")
        print(f"DAY86 CONTROL i22r slow {slow['i22r']}/10 -> {control}")
        verdicts += [v22, control]
    # Beside, deciding nothing: slow boots against compaction, and by position in the round.
    fast_fail = [r["compaction"].get("pgmigrate_fail", 0) for r in runs.values() if r["arm"] in doors and not r["slow"]]
    slow_fail = [r["compaction"].get("pgmigrate_fail", 0) for r in runs.values() if r["arm"] in doors and r["slow"]]
    top = max(fast_fail) if fast_fail else 0
    print(f"DAY86 BESIDE compaction (deciding nothing): pgmigrate_fail fast boots max={top} median="
          f"{statistics.median(fast_fail) if fast_fail else '-'}; slow boots min={min(slow_fail) if slow_fail else '-'} "
          f"median={statistics.median(slow_fail) if slow_fail else '-'}; slow boots above the fast maximum "
          f"{sum(f > top for f in slow_fail)}/{len(slow_fail)}")
    positions = {}
    for r in runs.values():
        if r["arm"] in doors:
            seq = arms if r["order"] == "o1" else list(reversed(arms))
            positions.setdefault(seq.index(r["arm"]), []).append(r["slow"])
    print("DAY86 BESIDE position in the round (deciding nothing): "
          + " ".join(f"pos{p}={sum(v)}/{len(v)}" for p, v in sorted(positions.items())))
    print(f"DAY86 VERDICT rig={rig} integrity=ok -> {' '.join(verdicts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
