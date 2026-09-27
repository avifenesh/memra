#!/usr/bin/env python3
"""WP-A day 64 (DAY64.md section 1 step 2, OWED item 18) reader: the late promotes placed, written before its cell runs.

usage: day64-reading.py ROOT
Input: ROOT/promote/ab/o{1,2}/bNN-promote/{server.log,promote/receipt.json} (stall_cell.py --mode promote --n 5; 20
boots). Complete: 20 boots with receipts, `errors` empty, 20 `STALL REPLAY: PASS`.
Per boot, every `promote published off the tick` line after the first (steady): the submission's tick and the
publication's tick (the day-33 timeline). A promote is LATE when it publishes after the next tick top (publication
tick > submission tick + 1). For a late promote, the poll at the next tick top (submission tick + 1) is its extra poll,
and its label (`pending on X`, step 1) names what it still waited on.
The placing rule (section 1): the requirement (`copies`, `receipt`, `sources`) named at the extra poll in at least half
of the late promotes is the place; otherwise NOT PLACED. The designs by the place: receipt -> the non-blocking
after-launch poll; sources -> DAY65's T-H; copies -> recorded as the copy stream's price.
"""
import collections
import glob
import json
import os
import re
import sys

PUB = re.compile(r"promote published off the tick: .*timeline from t0: submitted [+-][\d.]+ms \(tick (\d+), its top "
                 r"[+-][\d.]+ms\);(.*) published [+-][\d.]+ms \(tick (\d+), its top")
POLL = re.compile(r"poll \d+ at [+-][\d.]+ms \(tick (\d+), its top [+-][\d.]+ms\) (pending on ([a-z+]+)|pending|complete)")


def main():
    root = os.path.join(sys.argv[1], "promote", "ab")
    ds = sorted(glob.glob(os.path.join(root, "o*", "b*-promote")))
    complete, notes = len(ds) == 20, [] if len(ds) == 20 else [f"boots={len(ds)}"]
    steady = late = 0
    places = collections.Counter()
    for d in ds:
        rec = os.path.join(d, "promote", "receipt.json")
        if not os.path.exists(rec):
            complete = False
            notes.append(f"{os.path.basename(d)} no receipt")
            continue
        if json.load(open(rec)).get("summary", {}).get("errors"):
            complete = False
            notes.append(f"{os.path.basename(d)} errors")
        lines = [m for m in map(PUB.search, open(os.path.join(d, "server.log"), errors="replace")) if m]
        for m in lines[1:]:
            steady += 1
            sub, pub = int(m.group(1)), int(m.group(3))
            if pub <= sub + 1:
                continue
            late += 1
            label = "unlabelled"
            for p in POLL.finditer(m.group(2)):
                if int(p.group(1)) == sub + 1:
                    label = p.group(3) or p.group(2)
            for part in label.split("+"):
                places[part] += 1
    rp = os.path.join(root, "replays.log")
    passes = open(rp, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(rp) else 0
    if passes != 20:
        complete = False
        notes.append(f"replays PASS={passes} of 20")
    print(f"DAY64 READING steady promotes={steady} late={late} labels at the extra poll={dict(places)}")
    if not complete:
        print(f"DAY64 INCOMPLETE ({'; '.join(notes)}) -> nothing is placed; the cell repeats whole once")
        return
    if late == 0:
        print("DAY64 PLACE -> NONE LATE (0 late promotes on this tree: item 18 closes as read)")
        return
    named = [k for k in ("receipt", "sources", "copies") if places[k] * 2 >= late]
    print(f"DAY64 PLACE -> {' and '.join(named) if named else 'NOT PLACED'} ({late} late of {steady})")


if __name__ == "__main__":
    main()
