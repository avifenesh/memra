#!/usr/bin/env python3
"""WP-A day 30 reading (DAY30.md sections 2 and 2a, fixed before the run).

A2 (the pre-submit census): over the double-park cell's ON boots (`<ev>/o*/b*-on/server.log`), the ledger line's
`pre-submit P` of every demote after the third of its boot (the steady set). Pass: N >= 80, median <= 1.5 ms,
max <= 3.0 ms. Readings (not clauses): demotes 1 to 3 of each boot, the helper's `hashed in H ms`, the copy's
`from submission to completion` figure.

A3 (the completion lines), over every server log under the given roots: each `demote copy complete off the tick`
line carries `items=N (K KV, S f32 spans)` with N == K + S, S == --spans and K in --kv; the same ticket's
`contracts door D2H receipt` line carries `items=K` and the suffix `; S f32 spans landed under the ticket and taken
back before the retire`. A receipt with no copy-complete line of its ticket is an on-tick demote (not the door's
off-tick route; counted, not a clause).

    day30-reading.py --double-park <ev_dir> --spans 96 --kv 32,34 [--a3-root <dir> ...]
"""
import argparse
import glob
import os
import re
import statistics

LEDGER = re.compile(r"demote digests landed off the tick: ticket seq=(\d+), (\d+) payloads \(([\d.]+)MB\) hashed in "
                    r"([\d.]+)ms on the hash helper, .*pre-submit ([\d.]+), ")
COPY = re.compile(r"demote copy complete off the tick: ticket seq=(\d+) complete after (\d+) poll\(s\), ([\d.]+)ms "
                  r"from submission to completion \(([^)]*)\); items=(\d+) \((\d+) KV, (\d+) f32 spans\); (\d+) heap "
                  r"payloads \(([\d.]+)MB\)")
COPY_ANY = re.compile(r"demote copy complete off the tick: ticket seq=(\d+)")
RECEIPT = re.compile(r"contracts door D2H receipt: ticket issuer=\d+ seq=(\d+) \S+ items=(\d+) \((\d+) KV planes(, draft)?\)"
                     r".* retired acknowledged(?:; (\d+) f32 spans landed under the ticket and taken back before the "
                     r"retire)?$")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def census(ev):
    steady, first3, helper, copy = [], {1: [], 2: [], 3: []}, [], []
    boots = sorted(glob.glob(os.path.join(ev, "o*", "b*-on", "server.log")))
    per_boot = []
    for log in boots:
        n = 0
        for ln in open(log, errors="replace"):
            m = LEDGER.search(ln)
            if m:
                n += 1
                p = float(m.group(5))
                helper.append(float(m.group(4)))
                (first3[n] if n <= 3 else steady).append(p)
            c = COPY.search(ln)
            if c:
                copy.append(float(c.group(3)))
        per_boot.append(n)
    ok = len(steady) >= 80 and med(steady) <= 1.5 and max(steady, default=float("inf")) <= 3.0
    print(f"DAY30 A2 pre-submit steady N={len(steady)} median={med(steady):.2f} min={min(steady, default=float('nan')):.2f} "
          f"max={max(steady, default=float('nan')):.2f} boots_on={len(boots)} demotes_per_boot={sorted(set(per_boot))} "
          f"rule N>=80 median<=1.5 max<=3.0 -> {'PASS' if ok else 'FAIL'}")
    for k in (1, 2, 3):
        xs = first3[k]
        print(f"DAY30 READING pre-submit demote {k} of its boot N={len(xs)} median={med(xs):.2f} "
              f"min={min(xs, default=float('nan')):.2f} max={max(xs, default=float('nan')):.2f}")
    print(f"DAY30 READING helper hashed_in_ms N={len(helper)} median={med(helper):.1f} min={min(helper, default=float('nan')):.1f} "
          f"max={max(helper, default=float('nan')):.1f}; copy submission-to-completion N={len(copy)} median={med(copy):.1f}")
    return ok


def a3(roots, spans, kvs):
    logs = sorted({p for r in roots for p in glob.glob(os.path.join(r, "**", "*.log"), recursive=True)})
    copies = bad = receipts_paired = receipts_ontick = 0
    for log in logs:
        cc, rc = {}, {}
        for ln in open(log, errors="replace"):
            ln = ln.rstrip("\n")
            if COPY_ANY.search(ln):
                m = COPY.search(ln)
                seq = int(COPY_ANY.search(ln).group(1))
                if not m:
                    bad += 1
                    print(f"  A3 BAD copy-complete line (no items= term) in {log}: {ln[:200]}")
                    continue
                cc[seq] = (int(m.group(5)), int(m.group(6)), int(m.group(7)))
            m = RECEIPT.search(ln)
            if m:
                rc[int(m.group(1))] = (int(m.group(2)), int(m.group(5)) if m.group(5) else None)
        for seq, (n, k, s) in cc.items():
            copies += 1
            r = rc.get(seq)
            why = []
            if n != k + s:
                why.append(f"items={n} != {k}+{s}")
            if s != spans:
                why.append(f"spans={s} != {spans}")
            if k not in kvs:
                why.append(f"KV={k} not in {sorted(kvs)}")
            if r is None:
                why.append("no D2H receipt line for the ticket")
            else:
                receipts_paired += 1
                if r[0] != k:
                    why.append(f"receipt items={r[0]} != KV {k}")
                if r[1] != s:
                    why.append(f"receipt span suffix={r[1]} != {s}")
            if why:
                bad += 1
                print(f"  A3 BAD seq={seq} in {log}: {'; '.join(why)}")
        receipts_ontick += len(set(rc) - set(cc))
    print(f"DAY30 A3 logs={len(logs)} copy_complete_lines={copies} receipts_paired={receipts_paired} "
          f"receipts_without_copy_line(on-tick)={receipts_ontick} bad={bad} "
          f"rule items==KV+spans, spans={spans}, KV in {sorted(kvs)}, receipt items==KV and suffix==spans "
          f"-> {'PASS' if bad == 0 and copies > 0 else 'FAIL'}")
    return bad == 0 and copies > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--double-park")
    ap.add_argument("--spans", type=int, required=True)
    ap.add_argument("--kv", required=True)
    ap.add_argument("--a3-root", action="append", default=[])
    a = ap.parse_args()
    kvs = {int(x) for x in a.kv.split(",")}
    fails = 0
    if a.double_park:
        fails += not census(a.double_park)
    if a.a3_root:
        fails += not a3(a.a3_root, a.spans, kvs)
    print(f"DAY30 VERDICT clauses_failed={fails} -> {'ALL PASS' if fails == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
