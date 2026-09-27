#!/usr/bin/env python3
"""Day 70 reader for cell `sched` (research/spill-c-20260919/DAY70.md section 1, registered before the cell's script):
integrity, R1 the slow door boots, R2 the owner thread's run-queue wait, R3 the owner CPU's other busy time and its
SMT sibling's, R4 the host's busiest threads over each door run's gate-to-window span, and the strict verdict with a
count per field beside it.

usage: day70-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
SLOW_OVER_REF = 0.030
TICK_NS = 10_000_000  # USER_HZ 100
PHASE = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3})\t\[cpu-probe\] phase=(\w+) cpu=(-?\d+) compute_ns=([0-9.]+)", re.M)
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")


def hms(text):
    return datetime.strptime(text[:12], "%H:%M:%S.%f")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def expand(cpus):
    out = set()
    for part in cpus.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    siblings = {}
    for line in (ev / "topology.txt").read_text().splitlines():
        m = re.match(r"cpu(\d+) l3=\S* siblings=(\S+)", line)
        if m:
            siblings[int(m.group(1))] = expand(m.group(2)) - {int(m.group(1))}
    print(f"DAY70 PINS rig={rig} " + " ".join((ev / "pins.txt").read_text().split()))
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r*.log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        run["phases"] = {m.group(2): (hms(m.group(1)), float(m.group(4))) for m in PHASE.finditer(text)}
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
        if any(p not in r["phases"] for p in ("gate", "window")):
            fails.append(f"{label} phase lines missing")
        if r["arm"] == "i15" and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] == "i15"}) != 1:
        fails.append("host demand sequences differ across door runs")

    cpu_rows = defaultdict(list)  # cpu -> [(t, busy_ticks)]
    owner_rows = defaultdict(list)  # pid -> [(t, cpu, run_ns, wait_ns)]
    thread_rows = []  # (t, pid, tid, name, cpu, moved)
    for line in (ev / "sched.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        t = hms(parts[0])
        if parts[1] == "C":
            f = parts[2].split()
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            cpu_rows[int(f[0][3:])].append((t, sum(v[:8]) - idle))
        elif parts[1] == "O" and len(parts) == 6:
            run_ns, wait_ns, _ = parts[5].split()
            owner_rows[parts[2]].append((t, int(parts[4]), int(run_ns), int(wait_ns)))
        elif parts[1] == "T" and len(parts) == 7:
            thread_rows.append((t, parts[2], parts[3], parts[4], parts[5], int(parts[6])))
    if not owner_rows:
        fails.append("no owner samples")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY70 SCHED CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY70 SCHED VERDICT rig={rig} integrity=FAIL -> void")
        return 1

    def at(rows, t, before):
        """The sample at or before `t` (before=True) or at or after it."""
        pick = None
        for row in rows:
            if before and row[0] <= t:
                pick = row
            if not before and row[0] >= t:
                return row
        return pick

    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        lab, _, what = m.partition(" ")
        marks.setdefault(lab, {})[what.split()[0]] = hms(t[11:23])
    # Each door run's owner: the run-gen process sampled inside the run's marks, with a sample at or before its gate
    # line and one at or after its window line, both inside the marks.
    spans = {}
    for label, r in runs.items():
        if r["arm"] != "i15":
            continue
        s0, s1 = marks[label]["start"], marks[label]["end"]
        g, w = r["phases"]["gate"][0], r["phases"]["window"][0]
        pid = next((p for p, rows in owner_rows.items() if any(s0 <= x[0] <= s1 for x in rows)), None)
        a = b = None
        if pid is not None:
            a, b = at(owner_rows[pid], g, True), at(owner_rows[pid], w, False)
        if pid is None or a is None or b is None or a[0] < s0 or b[0] > s1:
            fails.append(f"{label} owner samples do not bracket its gate-to-window span")
        else:
            spans[label] = (pid, a, b)
    if fails:
        print(f"DAY70 SCHED CHECKS (spans) rig={rig} integrity=FAIL failed=" + "; ".join(fails))
        print(f"DAY70 SCHED VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    ref = med([r["gen_s"] for r in runs.values() if r["arm"] == "ref"])
    doors = []
    for label in sorted(spans):
        r = runs[label]
        r["slow"] = r["gen_s"] > ref + SLOW_OVER_REF
        pid, a, b = spans[label]
        rows = owner_rows[pid]
        wall = (b[0] - a[0]).total_seconds() * 1e9
        r["rq"] = (b[3] - a[3]) / wall if wall > 0 else None
        cpus = Counter(x[1] for x in rows if a[0] <= x[0] <= b[0])
        cpu = cpus.most_common(1)[0][0]
        ca, cb = at(cpu_rows[cpu], a[0], True), at(cpu_rows[cpu], b[0], False)
        busy_ns = (cb[1] - ca[1]) * TICK_NS
        r["own"] = max(0.0, busy_ns - (b[2] - a[2])) / wall if wall > 0 else None
        sib = sorted(siblings.get(cpu, set()))
        if sib:
            sa, sb = at(cpu_rows[sib[0]], a[0], True), at(cpu_rows[sib[0]], b[0], False)
            r["sib"] = (sb[1] - sa[1]) * TICK_NS / wall if wall > 0 else None
        else:
            r["sib"] = None
        movers = defaultdict(int)
        where = defaultdict(set)
        for t, p, tid, name, c, moved in thread_rows:
            if a[0] - timedelta(seconds=1) <= t <= b[0] + timedelta(seconds=1) and not (p == pid and tid == pid):
                movers[(p, tid, name)] += moved
                where[(p, tid, name)].add(c)
        top = sorted(movers.items(), key=lambda kv: -kv[1])[:3]
        top_text = ", ".join(f"{name}[{p}/{tid}] cpu={','.join(sorted(where[(p, tid, name)]))} ticks={v}"
                             for (p, tid, name), v in top)
        print(f"DAY70 R2R3R4 {label}: gen={r['gen_s']:.3f} {'slow' if r['slow'] else 'fast'}"
              f" gate_ns={r['phases']['gate'][1]:.3f} span_ms={wall / 1e6:.0f} owner_cpu={cpu} sibling={sib}"
              f" rqwait={r['rq']:.3f} owncpu_other={r['own']:.3f} sibling_busy="
              f"{r['sib'] if r['sib'] is None else round(r['sib'], 3)} | top: {top_text}")
        doors.append(r)
    slows = [r for r in doors if r["slow"]]
    fasts = [r for r in doors if not r["slow"]]
    print(f"DAY70 R1 ref_median={ref:.3f} slow={len(slows)} fast={len(fasts)} of {len(doors)} door runs")
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY70 SCHED VERDICT rig={rig} integrity=ok -> not_reproduced")
        return 0
    tracks = []
    for name, key in (("rqwait_tracks", "rq"), ("owncpu_tracks", "own"), ("sibling_tracks", "sib")):
        s = [r[key] for r in slows]
        f = [r[key] for r in fasts]
        if any(v is None for v in s + f):
            print(f"DAY70 COUNT {name}: not read in every run (deciding nothing)")
            continue
        print(f"DAY70 COUNT {name}: {sum(v > max(f) for v in s)} of {len(s)} slow runs beyond the fast runs' extreme"
              " (deciding nothing)")
        if min(s) > max(f):
            tracks.append(name)
    print(f"DAY70 SCHED VERDICT rig={rig} integrity=ok -> {' '.join(tracks) if tracks else 'none_tracks'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
