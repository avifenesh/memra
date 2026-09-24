#!/usr/bin/env python3
"""WP-A day 38 design G reader (DAY38.md section 3: acceptance (c) and (d) on the 5090 A/B), written before it runs.

Input: ROOT/ab/o{1,2}/bNN-{base,g}/server.log and .../demote/receipt.json (stall_cell.py's demote arm), ROOT/ab/replays.log.
Complete: 20 boots, five per arm per order, every boot with a receipt, every receipt `errors` empty and every intruder
without an error, 20 `STALL REPLAY: PASS` lines. Nothing is read from an incomplete cell.
Steady demotes: the second and later `demote digests landed off the tick` ledger line of each boot, every settle mode.
(c) On the g boots: the ledger's `copy settle` median <= 1.5 ms and max <= 3.0 ms over N >= 20 steady demotes.
(d) Per order: the steady demotes' `wall .. t0 to publication` median on g <= base + 5.0 ms, and the demoting
    intruder's e2e (`wall_ms` of every demote-arm run) median on g <= base + 1.0 ms.
Readings: take-back, owner-held and the helper's time on both arms; on g the D2H receipt kernel's copy-stream time
(`receipts on the receipt stream (source digests, X.XXms)`, DAY38 section 5a) and the count of D2H receipt lines naming it;
the tenant's `stall_ms` over the demote-arm runs. Usage: day38-reading.py ROOT
"""
import glob
import json
import os
import re
import statistics
import sys

LEDGER = re.compile(
    r"demote digests landed off the tick: ticket seq=\d+, (\d+) payloads \(([\d.]+)MB\) hashed in ([\d.]+)ms on the "
    r"hash helper, landed after \d+ poll\(s\) \(([^)]*)\); the owner thread held ([\d.]+)ms across the demote: "
    r"pre-submit ([\d.]+), copy settle ([\d.]+) over (\d+) poll\(s\), hashing polls ([\d.]+), take-back bind and "
    r"publish ([\d.]+); owner in-completion ([\d.]+)ms; wall ([\d.]+)ms t0 to publication")
RECEIPTS = re.compile(r"demote receipts on the hash helper: ticket seq=\d+, (\d+) KV receipts \(([\d.]+)MB in ([\d.]+)ms\)")
LEASES = re.compile(r"demote KV leases on the hash helper: ticket seq=\d+, (\d+) lease views \(([\d.]+)MB\)")
ARMS = ("base", "g")
KERNEL = re.compile(r"receipts on the receipt stream \(source digests(?:, ([\d.]+)ms)?\)")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}"


def boot(d):
    out = {"copy": [], "take": [], "wall": [], "held": [], "mode": [], "helper": [], "e2e": [], "stall": [],
           "receipts": [], "leases": 0, "landed": 0, "ok": True, "why": [], "kernel": [], "kernel_lines": 0}
    rec = os.path.join(d, "demote", "receipt.json")
    if not os.path.exists(rec):
        out["ok"] = False
        out["why"].append("no receipt")
    else:
        j = json.load(open(rec))
        if j.get("summary", {}).get("errors"):
            out["ok"] = False
            out["why"].append(f"errors={len(j['summary']['errors'])}")
        for r in j["runs"]:
            if r.get("arm") != "demote":
                continue
            i = r.get("intruder") or {}
            if "error" in i or "wall_ms" not in i:
                out["ok"] = False
                out["why"].append("intruder error")
                continue
            out["e2e"].append(i["wall_ms"])
            if r.get("stall_ms") is not None:
                out["stall"].append(r["stall_ms"])
    log = os.path.join(d, "server.log")
    if os.path.exists(log):
        k = 0
        for ln in open(log, errors="replace"):
            m = LEDGER.search(ln)
            if m:
                out["landed"] += 1
                k += 1
                if k >= 2:
                    out["copy"].append(float(m.group(7)))
                    out["take"].append(float(m.group(10)))
                    out["wall"].append(float(m.group(12)))
                    out["held"].append(float(m.group(5)))
                    out["mode"].append(m.group(4))
                    out["helper"].append(float(m.group(3)))
            m = RECEIPTS.search(ln)
            if m:
                out["receipts"].append(float(m.group(3)))
            if LEASES.search(ln):
                out["leases"] += 1
            m = KERNEL.search(ln)
            if m:
                out["kernel_lines"] += 1
                if m.group(1):
                    out["kernel"].append(float(m.group(1)))
    return out


def main():
    root = sys.argv[1]
    cells = {}
    complete = True
    notes = []
    for order in ("o1", "o2"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{arm}")))
            bs = [boot(d) for d in ds]
            if len(bs) != 5:
                complete = False
                notes.append(f"{order} {arm} boots={len(bs)}")
            for d, b in zip(ds, bs):
                if not b["ok"]:
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} {';'.join(b['why'])}")
            cells[(order, arm)] = bs
    replays = os.path.join(root, "ab", "replays.log")
    passes = open(replays, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(replays) else 0
    if passes != 20:
        complete = False
        notes.append(f"replays PASS={passes} of 20")
    pooled = lambda arm, key, order=None: [x for (o, a), bs in cells.items() if a == arm and order in (None, o)
                                           for b in bs for x in b[key]]
    for arm in ARMS:
        modes = pooled(arm, "mode")
        kinds = sorted(set(modes))
        by_mode = {k: [c for c, md in zip(pooled(arm, "copy"), modes) if md == k] for k in kinds}
        landed = sum(b["landed"] for (o, a), bs in cells.items() if a == arm for b in bs)
        klines = sum(b["kernel_lines"] for (o, a), bs in cells.items() if a == arm for b in bs)
        print(f"DAY38 READING arm={arm} steady {stat('copy-settle', pooled(arm, 'copy'))} | "
              f"{stat('take-back', pooled(arm, 'take'))} | {stat('owner-held', pooled(arm, 'held'))} | "
              f"{stat('helper-hash', pooled(arm, 'helper'))} | modes {[(k, len(v)) for k, v in by_mode.items()]} | "
              f"landed={landed} copy-stream-receipt-lines={klines} {stat('receipt-kernel-ms', pooled(arm, 'kernel'))}")
        for order in ("o1", "o2"):
            print(f"DAY38 READING order={order} arm={arm} {stat('wall', pooled(arm, 'wall', order))} | "
                  f"{stat('e2e', pooled(arm, 'e2e', order))} | {stat('tenant-stall', pooled(arm, 'stall', order))}")
    if not complete:
        print(f"DAY38 G INCOMPLETE ({'; '.join(notes)}) -> nothing is read")
        sys.exit(2)
    rc = 0
    copy = pooled("g", "copy")
    ok = len(copy) >= 20 and med(copy) <= 1.5 and max(copy) <= 3.0
    print(f"DAY38 G C {stat('copy-settle', copy)} rule N>=20 median<=1.5 max<=3.0 -> {'PASS' if ok else 'FAIL'}")
    rc |= 0 if ok else 1
    for order in ("o1", "o2"):
        wb, wg = pooled("base", "wall", order), pooled("g", "wall", order)
        eb, eg = pooled("base", "e2e", order), pooled("g", "e2e", order)
        ok = bool(wb and wg and eb and eg) and med(wg) <= med(wb) + 5.0 and med(eg) <= med(eb) + 1.0
        print(f"DAY38 G D order={order} wall base={med(wb):.2f} g={med(wg):.2f} g-minus-base={med(wg) - med(wb):+.2f} "
              f"rule <=+5.0 | e2e base={med(eb):.2f} g={med(eg):.2f} g-minus-base={med(eg) - med(eb):+.2f} "
              f"rule <=+1.0 -> {'PASS' if ok else 'FAIL'}")
        rc |= 0 if ok else 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
