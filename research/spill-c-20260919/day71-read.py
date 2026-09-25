#!/usr/bin/env python3
"""Day 71 reader for cell `core` (research/spill-c-20260919/DAY71.md section 1, registered before the cell's script):
integrity, R1 the slow door boots, R2 the core's counters at the gate probe, R3 the owner CPU's interrupts and
softirqs, its sibling's POLL time, the package's power, the card's PCIe receive rate and the owner's system share over
each door run's gate-to-window span, R4 the named sources beside them, and the strict verdict with a count per field.
DAY71 section 3 adds `intr_rate` (the host's interrupt total per second, from `/proc/stat`) and `migrate_fail` (failed
page migrations per second) and reads a hidden source as not read. With --class, the class verdict from two cells.

usage: day71-read.py <cell-dir> [--rig NAME]
       day71-read.py --class <cell-dir-a> <cell-dir-b>
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
PHASE_STEPS = 1 << 20
PHASE = re.compile(
    r"^(\d\d:\d\d:\d\d\.\d{3})\t\[cpu-probe\] phase=(\w+) cpu=(-?\d+) compute_ns=([0-9.]+)"
    r"(?: cpu_after=(-?\d+) wall_ns=(\d+) tsc=(\d+) mperf=(\d+) aperf=(\d+)| (counters=unavailable))?$", re.M)
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")
SLOT = re.compile(r" slot=\d+")
# (field, direction): +1 tracks when every slow value is above every fast one, -1 when below. The last two joined in
# DAY71 section 3 (registered after machine b's reading, before BOX15's half).
FIELDS = (("delivered_ghz", -1), ("ref_share", -1), ("counted_ghz", -1), ("cycles_per_step", +1),
          ("irq_rate", +1), ("softirq_rate", +1), ("sibling_poll", +1), ("pkg_w", +1), ("pcie_rx", +1),
          ("owner_sys", +1), ("intr_rate", +1), ("migrate_fail", +1))


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


def parse_runs(ev):
    runs = {}
    for log in sorted(ev.glob("o[12]-*-r*.log")):
        order, arm, _ = log.stem.split("-")
        run = d40.parse_run(log)
        run["exit"] = int((ev / f"{log.stem}.exit").read_text().strip())
        run["order"], run["arm"] = order, arm
        text = log.read_text(errors="replace")
        run["phases"] = {}
        for m in PHASE.finditer(text):
            c = None
            if m.group(5) is not None:
                c = {"cpu_after": int(m.group(5)), "wall_ns": int(m.group(6)), "tsc": int(m.group(7)),
                     "mperf": int(m.group(8)), "aperf": int(m.group(9))}
            run["phases"][m.group(2)] = {"t": hms(m.group(1)), "cpu": int(m.group(3)), "ns": float(m.group(4)),
                                         "counters": c, "unavailable": m.group(10) is not None}
        fill = FILL.search(text)
        run["fill_ms"] = float(fill.group(1)) if fill else None
        trace = [SLOT.sub("", line.partition("\t")[2]) for line in text.splitlines() if "[expert-host-slru] key=" in line]
        run["trace"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
        runs[log.stem] = run
    return runs


def run_checks(runs, counters_on):
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
        elif counters_on and any(r["phases"][p]["counters"] is None for p in ("gate", "window")):
            fails.append(f"{label} counters asked for but not on its phase lines")
        elif not counters_on and any(r["phases"][p]["counters"] is not None for p in ("gate", "window")):
            fails.append(f"{label} counters on its phase lines without the flag")
        if r["arm"] == "i15" and (r["fill_ms"] is None or r.get("physical_reads") != 0):
            fails.append(f"{label} fill={r['fill_ms']} physical_reads={r.get('physical_reads')}")
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] == "i15"}) != 1:
        fails.append("host demand sequences differ across door runs")
    return fails


def gate_counters(r):
    """R2: the gate probe's derived counters, or None when absent or the probe migrated."""
    g = r["phases"]["gate"]
    c = g["counters"]
    if c is None or c["cpu_after"] != g["cpu"] or c["wall_ns"] <= 0 or c["tsc"] <= 0 or c["mperf"] <= 0:
        return None
    return {"delivered_ghz": c["aperf"] / c["wall_ns"], "ref_share": c["mperf"] / c["tsc"],
            "counted_ghz": c["aperf"] / c["mperf"] * (c["tsc"] / c["wall_ns"]),
            "cycles_per_step": c["aperf"] / PHASE_STEPS}


def load_sampler(ev):
    s = {"C": defaultdict(list), "O": defaultdict(list), "T": [], "I": [], "S": [], "IH": {}, "SH": {}, "D": [],
         "DH": {}, "V": [], "E": defaultdict(list), "EH": {}, "H": [], "N": []}
    for line in (ev / "sched.tsv").read_text().splitlines():
        p = line.split("\t")
        if len(p) < 3:
            continue
        t, tag = hms(p[0]), p[1]
        if tag == "C":
            f = p[2].split()
            v = [int(x) for x in f[1:]]
            s["C"][int(f[0][3:])].append((t, sum(v[:8]) - v[3] - (v[4] if len(v) > 4 else 0)))
        elif tag == "O" and len(p) == 10:
            run_ns, wait_ns, _ = p[5].split()
            s["O"][p[2]].append((t, int(p[4]), int(run_ns), int(wait_ns), int(p[8]), int(p[9])))
        elif tag == "T" and len(p) == 7:
            s["T"].append((t, p[2], p[3], p[4], p[5], int(p[6])))
        elif tag in ("I", "S") and len(p) == 4:
            s[tag].append((t, p[2], {k: int(v) for k, v in (x.split(":") for x in p[3].split(","))}))
        elif tag in ("IH", "SH") and len(p) >= 3:
            s[tag][p[2]] = p[3] if len(p) > 3 else ""
        elif tag == "N" and len(p) == 4:
            s["N"].append((t, int(p[2]), int(p[3])))
        elif tag == "D" and len(p) == 4:
            moved = {}
            for x in p[3].split(","):
                k, _, v = x.partition(":")
                tm, _, _ = v.partition("/")
                moved[int(k)] = int(tm)
            s["D"].append((t, int(p[2]), moved))
        elif tag == "DH" and len(p) == 5:
            s["DH"][(int(p[2]), int(p[3]))] = p[4]
        elif tag == "V" and len(p) == 3:
            s["V"].append((t, {k: int(v) for k, v in (x.rsplit(":", 1) for x in p[2].split(","))}))
        elif tag == "E" and len(p) == 4:
            s["E"][p[2]].append((t, int(p[3])))
        elif tag == "EH" and len(p) == 5:
            s["EH"][p[2]] = (p[3], p[4])
        elif tag == "H" and len(p) == 3:
            s["H"].append((t, dict(x.rsplit("=", 1) for x in p[2].split(","))))
    return s


def at(rows, t, before):
    """The sample at or before `t` (before=True) or at or after it."""
    pick = None
    for row in rows:
        if before and row[0] <= t:
            pick = row
        if not before and row[0] >= t:
            return row
    return pick


def load_pcie(ev):
    out = []
    path = ev / "pcie.log"
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        stamp_, _, rest = line.partition("\t")
        f = rest.split()
        if len(f) >= 4 and not rest.lstrip().startswith("#") and f[2].isdigit() and f[3].isdigit():
            out.append((hms(stamp_), int(f[2]), int(f[3])))
    return out


def ref_median(runs):
    return med([r["gen_s"] for r in runs.values() if r["arm"] == "ref"])


def read_cell(cell, rig):
    ev = cell / "ev"
    siblings = {}
    for line in (ev / "topology.txt").read_text().splitlines():
        m = re.match(r"cpu(\d+) l3=\S* siblings=(\S+)", line)
        if m:
            siblings[int(m.group(1))] = expand(m.group(2)) - {int(m.group(1))}
    print(f"DAY71 PINS rig={rig} " + " ".join((ev / "pins.txt").read_text().split()))
    fails = []
    if not (ev / "counters.txt").exists() or not (ev / "counters-check.txt").exists():
        fails.append("counters.txt or counters-check.txt missing")
    counters_on = (ev / "counters.txt").exists() and \
        (ev / "counters.txt").read_text().strip() == "counters=--cpu-probe-counters"
    check = (ev / "counters-check.txt").read_text().strip().replace("\n", " | ") \
        if (ev / "counters-check.txt").exists() else "missing"
    print(f"DAY71 COUNTERS rig={rig} {'on' if counters_on else 'off'} check: {check}")
    runs = parse_runs(ev)
    fails += run_checks(runs, counters_on)
    s = load_sampler(ev)
    if not s["O"]:
        fails.append("no owner samples")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY71 CORE CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))
    if integrity != "ok":
        print(f"DAY71 CORE VERDICT rig={rig} integrity=FAIL -> void")
        return None
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        lab, _, what = m.partition(" ")
        marks.setdefault(lab, {})[what.split()[0]] = hms(t[11:23])
    spans = {}
    for label, r in runs.items():
        if r["arm"] != "i15":
            continue
        s0, s1 = marks[label]["start"], marks[label]["end"]
        g, w = r["phases"]["gate"]["t"], r["phases"]["window"]["t"]
        pid = next((p for p, rows in s["O"].items() if any(s0 <= x[0] <= s1 for x in rows)), None)
        a = b = None
        if pid is not None:
            a, b = at(s["O"][pid], g, True), at(s["O"][pid], w, False)
        if pid is None or a is None or b is None or a[0] < s0 or b[0] > s1:
            fails.append(f"{label} owner samples do not bracket its gate-to-window span")
        else:
            spans[label] = (pid, a, b)
    if fails:
        print(f"DAY71 CORE CHECKS (spans) rig={rig} integrity=FAIL failed=" + "; ".join(fails))
        print(f"DAY71 CORE VERDICT rig={rig} integrity=FAIL -> void")
        return None

    ref = ref_median(runs)
    refs = [gate_counters(r) for r in runs.values() if r["arm"] == "ref"]
    for name in ("delivered_ghz", "ref_share", "counted_ghz", "cycles_per_step"):
        vals = [c[name] for c in refs if c is not None]
        if vals:
            print(f"DAY71 REF gate {name}: min={min(vals):.3f} median={med(vals):.3f} max={max(vals):.3f}"
                  f" (N={len(vals)} of {len(refs)})")
        else:
            print(f"DAY71 REF gate {name}: not read")
    pcie = load_pcie(ev)
    has_pkg = [z for z, (name, ok) in s["EH"].items() if name.startswith("package") and ok == "readable"]
    doors = []
    for label in sorted(spans):
        r = runs[label]
        r["slow"] = r["gen_s"] > ref + SLOW_OVER_REF
        pid, a, b = spans[label]
        t0, t1 = a[0], b[0]
        wall_s = (t1 - t0).total_seconds()
        cpu = Counter(x[1] for x in s["O"][pid] if t0 <= x[0] <= t1).most_common(1)[0][0]
        sib = sorted(siblings.get(cpu, set()))
        v = dict.fromkeys(name for name, _ in FIELDS)
        v.update(gate_counters(r) or {})

        def rate(tag, c):
            # DAY71 section 3: a source with no header rows (hidden or absent) is not read, not zero.
            if not s[tag + "H"]:
                return None
            return sum(d.get(str(c), 0) for t, _, d in s[tag] if t0 < t <= t1) / wall_s

        v["irq_rate"] = rate("I", cpu)
        v["softirq_rate"] = rate("S", cpu)
        na, nb = at(s["N"], t0, True), at(s["N"], t1, False)
        if na and nb and nb[0] > na[0]:
            v["intr_rate"] = (nb[1] - na[1]) / (nb[0] - na[0]).total_seconds()
        if s["V"]:
            v["migrate_fail"] = sum(d.get("pgmigrate_fail", 0) for t, d in s["V"] if t0 < t <= t1) / wall_s
        if sib and any(c == sib[0] for c, _ in s["DH"]):
            poll = [k for (c, k), name in s["DH"].items() if c == sib[0] and name == "POLL"]
            us = sum(m.get(k, 0) for t, c, m in s["D"] if c == sib[0] and t0 < t <= t1 for k in poll)
            v["sibling_poll"] = us / 1e6 / wall_s
        if has_pkg:
            ea, eb = at(s["E"][has_pkg[0]], t0, True), at(s["E"][has_pkg[0]], t1, False)
            if ea and eb and eb[1] >= ea[1] and eb[0] > ea[0]:
                v["pkg_w"] = (eb[1] - ea[1]) / 1e6 / (eb[0] - ea[0]).total_seconds()
        # dmon's rows each report the second before their stamp: the rows stamped in (t0, t1 + 1 s] cover the span.
        rx = [x[1] for x in pcie if t0 < x[0] <= t1 + timedelta(seconds=1)]
        tx = [x[2] for x in pcie if t0 < x[0] <= t1 + timedelta(seconds=1)]
        v["pcie_rx"] = med(rx) if rx else None
        ut, st = b[4] - a[4], b[5] - a[5]
        v["owner_sys"] = st / (ut + st) if ut + st > 0 else None
        irq_rows = defaultdict(int)
        for t, name, d in s["I"]:
            if t0 < t <= t1:
                irq_rows[name] += d.get(str(cpu), 0)
        top_irq = ", ".join(f"{n}({s['IH'].get(n, '')[:40].strip()})={c}"
                            for n, c in sorted(irq_rows.items(), key=lambda kv: -kv[1])[:3] if c)
        vm = defaultdict(int)
        for t, d in s["V"]:
            if t0 < t <= t1:
                for k, x in d.items():
                    vm[k] += x
        top_vm = ", ".join(f"{k}={x}" for k, x in sorted(vm.items(), key=lambda kv: -abs(kv[1]))[:5])
        hot = None
        for t, d in s["H"]:
            if t0 <= t <= t1:
                for k, x in d.items():
                    if "temp" in k and x.lstrip("-").isdigit() and (hot is None or int(x) > hot[1]):
                        hot = (k, int(x))
        movers = defaultdict(int)
        for t, p, tid, name, c, moved in s["T"]:
            if t0 < t <= t1 and not (p == pid and tid == pid):
                movers[(p, tid, name)] += moved
        top_t = ", ".join(f"{name}[{p}/{tid}]={x}" for (p, tid, name), x in
                          sorted(movers.items(), key=lambda kv: -kv[1])[:3])

        def show(x):
            return "not_read" if x is None else f"{x:.3f}"

        print(f"DAY71 R2R3 {label}: gen={r['gen_s']:.3f} {'slow' if r['slow'] else 'fast'}"
              f" gate_ns={r['phases']['gate']['ns']:.3f} span_ms={wall_s * 1e3:.0f} owner_cpu={cpu} sibling={sib} "
              + " ".join(f"{name}={show(v[name])}" for name, _ in FIELDS))
        print(f"DAY71 R4 {label}: irq_top: {top_irq or '-'} | vmstat_top: {top_vm or '-'} | hottest:"
              f" {hot[0] + '=' + str(hot[1]) if hot else '-'} | pcie_tx={med(tx) if tx else 'not_read'}"
              f" | movers: {top_t or '-'}")
        r["v"] = v
        doors.append(r)
    slows = [r for r in doors if r["slow"]]
    fasts = [r for r in doors if not r["slow"]]
    print(f"DAY71 R1 rig={rig} ref_median={ref:.3f} slow={len(slows)} fast={len(fasts)} of {len(doors)} door runs")
    if len(slows) < 2 or len(fasts) < 2:
        print(f"DAY71 CORE VERDICT rig={rig} integrity=ok -> not_reproduced")
        return len(slows), len(fasts)
    tracks = []
    for name, sign in FIELDS:
        sv = [r["v"][name] for r in slows]
        fv = [r["v"][name] for r in fasts]
        if any(x is None for x in sv + fv):
            print(f"DAY71 COUNT {name}: not read in every door run (deciding nothing)")
            continue
        if sign > 0:
            beyond = sum(x > max(fv) for x in sv)
        else:
            beyond = sum(x < min(fv) for x in sv)
        print(f"DAY71 COUNT {name}: {beyond} of {len(sv)} slow runs beyond the fast runs' extreme"
              f" ({'higher' if sign > 0 else 'lower'}; deciding nothing)")
        if beyond == len(sv):
            tracks.append(name)
    print(f"DAY71 CORE VERDICT rig={rig} integrity=ok -> {' '.join(tracks) if tracks else 'none_tracks'}")
    return len(slows), len(fasts)


def class_verdict(cell_a, cell_b):
    got = {}
    for tag, cell in (("a", cell_a), ("b", cell_b)):
        got[tag] = read_cell(cell, f"class-{tag}")
    if got["a"] is None or got["b"] is None:
        print("DAY71 CLASS VERDICT integrity=FAIL -> void")
        return
    (sa, fa), (sb, fb) = got["a"], got["b"]
    if sb >= 2:
        verdict = "class"
    elif sb == 0 and sa >= 2 and fa >= 2:
        verdict = "this_host"
    else:
        verdict = "undecided"
    print(f"DAY71 CLASS VERDICT a=slow{sa}/fast{fa} b=slow{sb}/fast{fb} -> {verdict}")


def main():
    if "--class" in sys.argv:
        i = sys.argv.index("--class")
        class_verdict(Path(sys.argv[i + 1]), Path(sys.argv[i + 2]))
        return 0
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    read_cell(Path(sys.argv[1]), rig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
