#!/usr/bin/env python3
"""WP-A day 62 (DAY62.md section 1 step 2, OWED item 13) reader: the retire seam's price and the design it selects,
written before its cell runs.

usage: day62-reading.py ROOT
Input: ROOT/seam/ab/o{1,2}/bNN-{prime,retire-seam,retire-seam-other}: five boots per mode per order. Complete: 30
boots with receipts, `errors` empty, 30 `STALL REPLAY: PASS`.
The holds: every `capture published off the tick` line settled by `a session retire (source retiring: yes|no)`; its
`the settle held the owner thread X ms` and its `copy stream in flight at submission: K`. The SOURCE shape's holds are
the `yes` lines of the `retire-seam` boots, the NO-SOURCE shape's the `no` lines of the `retire-seam-other` boots.
The rule (section 1): per shape, the median hold above 1.0 ms in both orders selects its design (no-source: R1;
source: R2); both, both; neither, the seam closes as priced. A shape with no line in an order is named and selects
nothing (its shape did not form; the cell is revised under a new registration).
Readings: the tenant's stall of each mode against `prime`, per order; the in-flight kinds seen per shape.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

PUB = re.compile(r"\[prefix-cache\] capture published off the tick .*settled synchronously by a session retire "
                 r"\(source retiring: (yes|no)\); the settle held the owner thread ([\d.]+)ms.*copy stream in flight at "
                 r"submission: ([^)]*)\)")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    root = os.path.join(sys.argv[1], "seam", "ab")
    complete, notes = True, []
    stall = {}
    holds = collections.defaultdict(list)
    kinds = collections.defaultdict(collections.Counter)
    other = collections.Counter()
    for order in ("o1", "o2"):
        for mode in ("prime", "retire-seam", "retire-seam-other"):
            ds = sorted(glob.glob(os.path.join(root, order, f"b*-{mode}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{order} {mode} boots={len(ds)}")
            st = []
            for d in ds:
                rec = os.path.join(d, mode, "receipt.json")
                if not os.path.exists(rec):
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} no receipt")
                    continue
                j = json.load(open(rec))
                if j.get("summary", {}).get("errors"):
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} errors")
                st += [r["stall_ms"] for r in j["runs"] if r.get("arm") == mode and "stall_ms" in r]
                for m in PUB.finditer(open(os.path.join(d, "server.log"), errors="replace").read()):
                    want = {"retire-seam": "yes", "retire-seam-other": "no"}.get(mode)
                    if m.group(1) == want:
                        shape = "source" if want == "yes" else "no-source"
                        holds[(order, shape)].append(float(m.group(2)))
                        kinds[shape][m.group(3)] += 1
                    else:
                        other[(mode, m.group(1))] += 1
            stall[(order, mode)] = st
    passes = open(os.path.join(root, "replays.log"), errors="replace").read().count("STALL REPLAY: PASS") \
        if os.path.exists(os.path.join(root, "replays.log")) else 0
    if passes != 30:
        complete = False
        notes.append(f"replays PASS={passes} of 30")
    for order in ("o1", "o2"):
        p = med(stall[(order, "prime")])
        print(f"DAY62 READING order={order} stall prime={p:.2f} retire-seam={med(stall[(order, 'retire-seam')]):.2f} "
              f"retire-seam-other={med(stall[(order, 'retire-seam-other')]):.2f} ms | holds source N="
              f"{len(holds[(order, 'source')])} median={med(holds[(order, 'source')]):.2f} no-source N="
              f"{len(holds[(order, 'no-source')])} median={med(holds[(order, 'no-source')]):.2f} ms")
    for shape in ("source", "no-source"):
        print(f"DAY62 IN-FLIGHT shape={shape} {dict(kinds[shape])}")
    if other:
        print(f"DAY62 OTHER retire settles (not a shape's): {dict(other)}")
    if not complete:
        print(f"DAY62 INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        return
    missing = [f"{o} {s}" for o in ("o1", "o2") for s in ("source", "no-source") if not holds[(o, s)]]
    if missing:
        print(f"DAY62 SHAPE MISSING ({', '.join(missing)}) -> that shape selects nothing")
    sel = []
    if all(holds[(o, "no-source")] and med(holds[(o, "no-source")]) > 1.0 for o in ("o1", "o2")):
        sel.append("R1 (no-source retires do not settle)")
    if all(holds[(o, "source")] and med(holds[(o, "source")]) > 1.0 for o in ("o1", "o2")):
        sel.append("R2 (the source's planes held by the capture)")
    print(f"DAY62 SELECT -> {' and '.join(sel) if sel else 'NEITHER (the seam closes as priced)'}")


if __name__ == "__main__":
    main()
