#!/usr/bin/env python3
"""WP-A day 50 reader (DAY50.md section 1: OWED item 9, the promote's tick placement on the current tree), written
before it runs on any receipt.

usage: day50-reading.py BOOT_DIR [BOOT_DIR ...]
Each BOOT_DIR holds server.log and promote/receipt.json (stall_cell.py --mode promote). Per promote-arm run, C's tick
definition (lane C's day39-tick-split.py, unchanged): f = fire_at - 1; a gap is stretched when itl_ms[i] > 3 x the run's
median gap; tick 1 is the first stretched gap at an index >= f, tick 2 the next. Per boot, the server's `promote
published off the tick` lines (the day-33 timeline): the submission's tick, the publication's tick, and their distance.
Printed per boot and pooled: the runs with a tick 1, with a tick 2, the medians of tick 1 and tick 2 (where defined) and
of the stretched gaps' count; the timeline's submission-to-publication tick distance (steady: the 2nd line on of a boot)
and the submission's own owner segment (`owner segment X ms` of `promote submitted off the tick`).
A reading: nothing here is a clause.
"""
import json
import os
import re
import statistics
import sys

PUB = re.compile(r"promote published off the tick: .*timeline from t0: submitted [+-][\d.]+ms \(tick (\d+), its top "
                 r"[+-][\d.]+ms\);.* published [+-][\d.]+ms \(tick (\d+), its top")
SEG = re.compile(r"promote submitted off the tick: .* owner segment ([\d.]+)ms")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def ticks(run, fire_at):
    itl = run.get("itl_ms") or []
    if not itl:
        return None
    p50 = statistics.median(itl)
    f = fire_at - 1
    stretched = [g for i, g in enumerate(itl) if i >= f and g > 3.0 * p50]
    return stretched


def main():
    pooled = {"runs": 0, "t1": [], "t2": [], "n": [], "dist": [], "seg": []}
    for d in sys.argv[1:]:
        rec = os.path.join(d, "promote", "receipt.json")
        if not os.path.exists(rec):
            print(f"DAY50 BOOT {d} no receipt")
            continue
        j = json.load(open(rec))
        fire_at = j.get("fire_at", 24)
        t1, t2, ns = [], [], []
        runs = 0
        for r in j["runs"]:
            if r.get("arm") != "promote":
                continue
            s = ticks(r, fire_at)
            if s is None:
                continue
            runs += 1
            ns.append(len(s))
            if len(s) >= 1:
                t1.append(s[0])
            if len(s) >= 2:
                t2.append(s[1])
        dist, seg = [], []
        k = 0
        for ln in open(os.path.join(d, "server.log"), errors="replace"):
            m = PUB.search(ln)
            if m:
                k += 1
                if k >= 2:
                    dist.append(int(m.group(2)) - int(m.group(1)))
            m = SEG.search(ln)
            if m:
                seg.append(float(m.group(1)))
        print(f"DAY50 BOOT {os.path.basename(d.rstrip('/'))} runs={runs} tick1={len(t1)}/{runs} tick2={len(t2)}/{runs} "
              f"tick1_med={med(t1):.2f} tick2_med={med(t2):.2f} stretched_med={med(ns):.0f} "
              f"submit_to_publish_ticks={sorted(set(dist))} owner_segment_med={med(seg):.2f}")
        pooled["runs"] += runs
        pooled["t1"] += t1
        pooled["t2"] += t2
        pooled["n"] += ns
        pooled["dist"] += dist
        pooled["seg"] += seg
    p = pooled
    dcount = {x: p["dist"].count(x) for x in sorted(set(p["dist"]))}
    print(f"DAY50 POOLED runs={p['runs']} tick1={len(p['t1'])} tick2={len(p['t2'])} tick1_med={med(p['t1']):.2f} "
          f"tick2_med={med(p['t2']):.2f} stretched_med={med(p['n']):.0f} submit_to_publish_ticks={dcount} "
          f"owner_segment_med={med(p['seg']):.2f}")


if __name__ == "__main__":
    main()
