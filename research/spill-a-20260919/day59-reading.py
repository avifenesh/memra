#!/usr/bin/env python3
"""WP-A day 59 (DAY59.md section 1, OWED item 10) reader: the fanout's owner-time attribution and the design it
selects, written before its cell runs.

usage: day59-reading.py ROOT
Input: ROOT/short/ab/o{1,2}/bNN-{fanout,prime-short}/{server.log,<mode>/receipt.json} and ROOT/short/ab/replays.log
(DAY54's paired cell shape, five boots per mode per order). Complete: 20 boots with receipts, `errors` empty, 20
`STALL REPLAY: PASS`, and an on-tick split line for every fanout on-tick publish line.
Steady: the second and later fanout tick of each fanout boot. Per order, medians of the snapshot and restores' owner
times (`on-tick publish`) and of the split's parts (`on-tick split`).
The rule (DAY59 section 1, with section 2's reading of the length sets as device calls):
  calls = snapshot copies + clones + restores copies + length sets; allocs = snapshot allocations;
  calls >= 60% of (snapshot + restores) in both orders -> DESIGN B1 (batched copies);
  allocs >= 60% in both orders -> DESIGN B2 (one pool reservation); otherwise NEITHER (both priced, the larger first).
Readings: the DAY54 stall price (fanout minus prime-short) on this card, per order.
"""
import glob
import json
import os
import re
import statistics
import sys

PUB = re.compile(r"\[prefix-dedup\] on-tick publish: snapshot ([\d.]+) ms \(([\d.]+) MB\), (\d+) sibling restore\(s\) "
                 r"([\d.]+) ms, insert ([\d.]+) ms")
SPLIT = re.compile(r"\[prefix-dedup\] on-tick split: snapshot alloc ([\d.]+) ms over (\d+), copies ([\d.]+) ms over "
                   r"(\d+), clones ([\d.]+) ms over (\d+); restores copies ([\d.]+) ms over (\d+), len sets ([\d.]+) ms "
                   r"over (\d+)")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    root = os.path.join(sys.argv[1], "short")
    complete, notes = True, []
    pools = {}
    for order in ("o1", "o2"):
        for mode in ("fanout", "prime-short"):
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{mode}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{order} {mode} boots={len(ds)}")
            p = {"stall": [], "snap": [], "rest": [], "alloc": [], "calls": [], "sets": [], "copies": [], "clones": []}
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
                p["stall"] += [r["stall_ms"] for r in j["runs"] if r.get("arm") == mode and "stall_ms" in r]
                if mode != "fanout":
                    continue
                lines = open(os.path.join(d, "server.log"), errors="replace").read().splitlines()
                pubs = [m for m in map(PUB.search, lines) if m]
                splits = [m for m in map(SPLIT.search, lines) if m]
                if len(pubs) != len(splits):
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} publish lines={len(pubs)} split lines={len(splits)}")
                for m, sp in list(zip(pubs, splits))[1:]:
                    p["snap"].append(float(m.group(1)))
                    p["rest"].append(float(m.group(4)))
                    a, c, cl, rc, st = (float(sp.group(k)) for k in (1, 3, 5, 7, 9))
                    p["alloc"].append(a)
                    p["copies"].append(c + rc)
                    p["clones"].append(cl)
                    p["sets"].append(st)
                    p["calls"].append(c + cl + rc + st)
            pools[(order, mode)] = p
    rp = os.path.join(root, "ab", "replays.log")
    passes = open(rp, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(rp) else 0
    if passes != 20:
        complete = False
        notes.append(f"replays PASS={passes} of 20")
    for order in ("o1", "o2"):
        f, c = pools[(order, "fanout")], pools[(order, "prime-short")]
        print(f"DAY59 READING order={order} N={len(f['snap'])} snapshot={med(f['snap']):.2f} restores={med(f['rest']):.2f} "
              f"| alloc={med(f['alloc']):.2f} copies={med(f['copies']):.2f} clones={med(f['clones']):.2f} "
              f"sets={med(f['sets']):.2f} ms | stall fanout={med(f['stall']):.2f} prime-short={med(c['stall']):.2f} "
              f"fanout-minus-prime={med(f['stall']) - med(c['stall']):+.2f} ms")
    if not complete:
        print(f"DAY59 INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        return
    shares = []
    for order in ("o1", "o2"):
        f = pools[(order, "fanout")]
        total = med(f["snap"]) + med(f["rest"])
        shares.append((med(f["calls"]) / total, med(f["alloc"]) / total))
        print(f"DAY59 SHARES order={order} calls={shares[-1][0]:.2f} allocs={shares[-1][1]:.2f} of {total:.2f} ms")
    if all(c >= 0.6 for c, _ in shares):
        v = "DESIGN B1 (batched copies)"
    elif all(a >= 0.6 for _, a in shares):
        v = "DESIGN B2 (one pool reservation)"
    else:
        v = "NEITHER (both priced; the larger designed first)"
    print(f"DAY59 SELECT -> {v}")


if __name__ == "__main__":
    main()
