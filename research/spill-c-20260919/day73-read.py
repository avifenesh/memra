#!/usr/bin/env python3
"""Day 73 reader for cell `compact` (research/spill-c-20260919/DAY73.md section 1, registered before the cell's
scripts): integrity, R1 the state per door run from its gate probe, R2 compaction over each door run's gate-to-window
span (and REF's), R3 the process's pinned, locked and huge-page memory at the gate, R4 buddyinfo's high orders at the
gate, R5 Part B's system calls, and the strict verdict with a count beside each field. DAY71's reader supplies the run
parsing, the sampler's shared rows and the gate counters.

usage: day73-read.py <cell-dir> [--rig NAME]
"""
import importlib.util
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day71_read", HERE / "day71-read.py")
d71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d71)

N = 32
SLOW_CYCLES = 7.5
SLOW_NS_OVER_REF = 1.25


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def load_extra(ev):
    buddy, proc = [], defaultdict(list)
    for line in (ev / "sched.tsv").read_text().splitlines():
        p = line.split("\t")
        if len(p) == 4 and p[1] == "B":
            buddy.append((d71.hms(p[0]), p[2], [int(x) for x in p[3].split(",")]))
        elif len(p) == 4 and p[1] == "P":
            proc[p[2]].append((d71.hms(p[0]), dict(x.split("=") for x in p[3].split(",") if "=" in x)))
    return buddy, proc


def strace_counts(path):
    """`strace -c` summary: {syscall: calls}."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) >= 5 and f[-1] not in ("total", "syscall") and f[0].replace(".", "").isdigit():
            try:
                out[f[-1]] = int(f[3])
            except ValueError:
                continue
    return out


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    print(f"DAY73 PINS rig={rig} " + " ".join((ev / "pins.txt").read_text().split()))
    fails = []
    if not (ev / "counters.txt").exists():
        fails.append("counters.txt missing")
    counters_on = (ev / "counters.txt").exists() and \
        (ev / "counters.txt").read_text().strip() == "counters=--cpu-probe-counters"
    runs = d71.parse_runs(ev)
    checks = d71.run_checks(runs, counters_on)
    fails += [f for f in checks if not f.startswith("runs=")]
    if len(runs) != 20 or sorted({r["arm"] for r in runs.values()}) != ["i15", "ref"]:
        fails.append(f"runs={len(runs)} arms={sorted({r['arm'] for r in runs.values()})}")
    s = d71.load_sampler(ev)
    buddy, proc = load_extra(ev)
    if not s["O"] or not s["V"] or not proc:
        fails.append("owner, vmstat or process rows missing")
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        lab, _, what = m.partition(" ")
        marks.setdefault(lab, {})[what.split()[0]] = d71.hms(t[11:23])
    spans = {}
    for label, r in runs.items():
        if any(p not in r["phases"] for p in ("gate", "window")):
            continue
        s0, s1 = marks[label]["start"], marks[label]["end"]
        g, w = r["phases"]["gate"]["t"], r["phases"]["window"]["t"]
        pid = next((p for p, rows in s["O"].items() if any(s0 <= x[0] <= s1 for x in rows)), None)
        a = b = None
        if pid is not None:
            a, b = d71.at(s["O"][pid], g, True), d71.at(s["O"][pid], w, False)
        if pid is None or a is None or b is None or a[0] < s0 or b[0] > s1 or not proc.get(pid):
            if r["arm"] == "i15":
                fails.append(f"{label} sampler rows do not bracket its gate-to-window span")
            continue
        spans[label] = (pid, a[0], b[0])
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY73 COMPACT CHECKS rig={rig} runs={len(runs)} counters={'on' if counters_on else 'off'}"
          f" integrity={integrity}" + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY73 COMPACT VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    ref_ns = med([r["phases"]["gate"]["ns"] for r in runs.values() if r["arm"] == "ref"])
    order = sorted(runs, key=lambda k: marks[k]["start"])
    ref_rank = {k: i for i, k in enumerate(k for k in order if runs[k]["arm"] == "ref")}
    doors, ref_compacts = [], []
    for label in order:
        r = runs[label]
        if label not in spans:
            continue
        pid, t0, t1 = spans[label]
        wall = (t1 - t0).total_seconds()
        vm = defaultdict(int)
        for t, d in s["V"]:
            if t0 < t <= t1:
                for k, x in d.items():
                    vm[k] += x
        rates = {"migrate_fail": vm["pgmigrate_fail"] / wall, "compact_isolated": vm["compact_isolated"] / wall,
                 "migrate_ok": vm["pgmigrate_success"] / wall}
        g = r["phases"]["gate"]
        c = d71.gate_counters(r)
        if c is not None:
            state_by = f"cycles_per_step={c['cycles_per_step']:.3f}"
            slow = c["cycles_per_step"] >= SLOW_CYCLES
        else:
            state_by = f"gate_ns={g['ns']:.3f} ref_gate_ns={ref_ns:.3f}"
            slow = g["ns"] >= SLOW_NS_OVER_REF * ref_ns
        p = d71.at(proc[pid], g["t"], True) or proc[pid][0]
        mem = " ".join(f"{k}={p[1].get(k, '?')}" for k in ("VmPin", "VmLck", "AnonHugePages", "Locked", "VmRSS"))
        bg = d71.at(buddy, g["t"], True)
        normal = [row for row in buddy if row[0] == (bg[0] if bg else None) and "Normal" in row[1]]
        high = sum(sum(x[9:]) for _, _, x in normal) if normal else None
        if r["arm"] == "ref":
            if ref_rank[label] >= 2 and vm["compact_isolated"] > 0:
                ref_compacts.append(label)
            print(f"DAY73 REF {label}: gen={r['gen_s']:.3f} {state_by} migrate_fail={rates['migrate_fail']:.0f}"
                  f" compact_isolated={rates['compact_isolated']:.0f} migrate_ok={rates['migrate_ok']:.0f} | {mem}"
                  f" | normal_free_order9plus={high}")
            continue
        r["slow"], r["rates"] = slow, rates
        doors.append(r)
        print(f"DAY73 R1R2 {label}: gen={r['gen_s']:.3f} {'slow' if slow else 'fast'} {state_by} span_ms={wall * 1e3:.0f}"
              f" migrate_fail={rates['migrate_fail']:.0f} compact_isolated={rates['compact_isolated']:.0f}"
              f" migrate_ok={rates['migrate_ok']:.0f} | R3 {mem} | R4 normal_free_order9plus={high}")
    got = defaultdict(list)
    strace = (ev / "strace.txt").read_text().strip() if (ev / "strace.txt").exists() else "strace.txt missing"
    for label in ("s-ref-r1", "s-i15-r1", "s-i15-r2", "s-ref-r2"):
        path = ev / f"{label}.strace"
        if path.exists():
            got[label.split("-")[1]].append(strace_counts(path))
    if len(got["ref"]) == 2 and len(got["i15"]) == 2:
        names = set().union(*got["ref"], *got["i15"])
        diff = {n: med([x.get(n, 0) for x in got["i15"]]) - med([x.get(n, 0) for x in got["ref"]]) for n in names}
        top = sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:10]
        print(f"DAY73 R5 rig={rig} ({strace.splitlines()[0] if strace else '-'}) calls, door minus REF (medians of two):"
              " " + ", ".join(f"{n}={v:+.0f}" for n, v in top))
    else:
        print(f"DAY73 R5 rig={rig} not read ({strace.splitlines()[0] if strace else '-'})")
    slows = [r for r in doors if r["slow"]]
    fasts = [r for r in doors if not r["slow"]]
    print(f"DAY73 R1 rig={rig} slow={len(slows)} fast={len(fasts)} of {len(doors)} door runs"
          f" (slow: gate cycles per step >= {SLOW_CYCLES}, or gate ns >= {SLOW_NS_OVER_REF} x REF's without counters)")
    print(f"DAY73 REF_COMPACTS rig={rig} {'ref_compacts ' + ','.join(ref_compacts) if ref_compacts else 'none'}"
          " (REF spans from its third run on; deciding nothing)")
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY73 COMPACT VERDICT rig={rig} integrity=ok -> not_reproduced")
        return 0
    tracks = []
    for name, key in (("migrate_fail_tracks", "migrate_fail"), ("isolated_tracks", "compact_isolated")):
        sv = [r["rates"][key] for r in slows]
        fv = [r["rates"][key] for r in fasts]
        print(f"DAY73 COUNT {name}: {sum(x > max(fv) for x in sv)} of {len(sv)} slow runs beyond the fast runs'"
              " extreme (deciding nothing)")
        if min(sv) > max(fv):
            tracks.append(name)
    print(f"DAY73 COMPACT VERDICT rig={rig} integrity=ok -> {' '.join(tracks) if tracks else 'none_tracks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
