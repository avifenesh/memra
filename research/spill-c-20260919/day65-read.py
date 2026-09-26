#!/usr/bin/env python3
"""Day 65 reader for cell `pin` (research/spill-c-20260919/DAY65.md section 1, registered before the cell's script):
integrity, R1 the medians and IQRs, R2 the slow door boots, R3 each door run's placement, R4 the door against REF on
one L3 domain under the admissibility clause, and the verdict.

usage: day65-read.py <cell-dir> [--rig NAME]
"""
import hashlib
import importlib.util
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
ARMS = ("ref2", "i15two", "ref1", "i15one")
DOORS = ("i15two", "i15one")
SLOW_OVER_REF = 0.030
ADMISSIBLE_IQR = 0.005
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ")
SLOT = re.compile(r" slot=\d+")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


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


def stamp(text):
    return datetime.strptime(text[:12], "%H:%M:%S.%f")


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    if (ev / "NOT_APPLICABLE").exists():
        print(f"DAY65 PIN VERDICT rig={rig} -> not_applicable ({(ev / 'NOT_APPLICABLE').read_text().strip()})")
        return 0
    pins = dict(line.split("=", 1) for line in (ev / "pins.txt").read_text().split())
    home = expand(pins["home_l3"])
    print(f"DAY65 PINS rig={rig} two={pins['two']} one={pins['one']} home_l3={pins['home_l3']}"
          f" two_l3_domains={pins['two_l3_domains']}")
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
        runs[log.stem] = run
    fails = []
    if len(runs) != 40 or sorted({r["arm"] for r in runs.values()}) != sorted(ARMS):
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
    if len({r.get("tokens") for r in runs.values()}) != 1:
        fails.append("tapes differ across arms")
    if len({r["trace"] for r in runs.values() if r["arm"] in DOORS}) != 1:
        fails.append("host demand sequences differ across door runs")
    integrity = "ok" if not fails else "FAIL"
    print(f"DAY65 PIN CHECKS rig={rig} runs={len(runs)} integrity={integrity}"
          + ("" if not fails else " failed=" + "; ".join(fails)))

    def values(arm, key, order=None):
        return [r[key] for r in runs.values() if r["arm"] == arm and key in r and (order is None or r["order"] == order)]

    for arm in ARMS:
        print(f"DAY65 R1 {arm}: gen median={med(values(arm, 'gen_s')):.3f} iqr={iqr(values(arm, 'gen_s')):.4f}"
              f" | window median={med(values(arm, 'window_s')):.3f} iqr={iqr(values(arm, 'window_s')):.4f} (N=10)")
    ref_of = {"i15two": "ref2", "i15one": "ref1"}
    slow = {}
    for door in DOORS:
        bound = med(values(ref_of[door], "gen_s")) + SLOW_OVER_REF
        slow[door] = sorted(label for label, r in runs.items() if r["arm"] == door and r["gen_s"] > bound)
        print(f"DAY65 R2 {door}: slow={len(slow[door])} of 10 (gen-only over {ref_of[door]} median + {SLOW_OVER_REF})"
              f" {slow[door]}")
    # R3: each door run's placement share on CPU 0's L3 domain, from the sampler inside the run's marks.
    marks = {}
    for line in (ev / "marks.tsv").read_text().splitlines():
        t, m = line.split("\t")
        label, _, what = m.partition(" ")
        marks.setdefault(label, {})[what.split()[0]] = datetime.strptime(t[11:23], "%H:%M:%S.%f")
    samples = []
    for line in (ev / "placement.tsv").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            samples.append((stamp(parts[0]), int(parts[2]), parts[3]))
    tracks = True
    for label in sorted(r for r in runs if runs[r]["arm"] == "i15two" or runs[r]["arm"] == "i15one"):
        a, b = marks[label]["start"], marks[label]["end"]
        mine = [cpu for t, cpu, comm in samples if a <= t <= b and comm.startswith("run-gen-i15")]
        share = sum(cpu in home for cpu in mine) / len(mine) if mine else float("nan")
        is_slow = label in slow[runs[label]["arm"]]
        print(f"DAY65 R3 {label}: samples={len(mine)} home_share={share:.2f} {'slow' if is_slow else 'fast'}")
        if runs[label]["arm"] == "i15two" and not (share == share and ((share < 0.5) == is_slow)):
            tracks = False
    # R4: the door against REF on ONE, under the admissibility clause.
    admissible = all(iqr(values(a, k)) <= ADMISSIBLE_IQR for a in ("ref1", "i15one") for k in ("gen_s", "window_s"))
    readings = []
    for key, label in (("gen_s", "gen-only decode"), ("window_s", "steady window")):
        noise = max(iqr(values("i15one", key)), iqr(values("ref1", key)))
        diff = {o: med(values("i15one", key, o)) - med(values("ref1", key, o)) for o in (None, "o1", "o2")}
        if not admissible:
            name = "void (inadmissible)"
        elif all(d < -noise for d in diff.values()):
            name = "beats"
        elif diff["o1"] > noise and diff["o2"] > noise:
            name = "loses"
        else:
            name = "matches"
        readings.append(name)
        print(f"DAY65 R4 i15one_vs_ref1 {label}: pooled={diff[None]:+.4f} o1={diff['o1']:+.4f} o2={diff['o2']:+.4f}"
              f" noise={noise:.4f} -> {name}")
    if integrity != "ok":
        print(f"DAY65 PIN VERDICT rig={rig} integrity=FAIL -> void")
        return 1
    if not slow["i15two"]:
        verdict = "not_reproduced"
    elif not slow["i15one"]:
        verdict = "pin_removes"
    elif len(slow["i15one"]) >= 2:
        verdict = "pin_does_not"
    else:
        verdict = "inconclusive"
    placement = "placement_tracks_mode" if tracks else "placement_does_not_track"
    print(f"DAY65 PIN VERDICT rig={rig} integrity=ok -> {verdict} {placement}; one-domain door vs REF:"
          f" gen {readings[0]}, window {readings[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
