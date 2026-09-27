#!/usr/bin/env python3
"""WP-A day 42 design S2 reader (DAY42.md section 1: clauses (c) and (d), S's bounds unchanged), written before its
cells run: day40-reading.py with the arms g4 and s2.

usage: day42-reading.py demote|promote ROOT
Input: ROOT/ab/o{1,2}/bNN-{g4,s2}/server.log and .../<mode>/receipt.json (stall_cell.py), ROOT/ab/replays.log; 20 boots,
five per arm per order. Complete: every boot with a receipt, `errors` empty, every intruder without an error, one
`STALL REPLAY: PASS` per boot; nothing is read from an incomplete cell.
demote (c): per order, the steady demotes' `wall .. t0 to publication` median (the second and later `demote digests
    landed off the tick` ledger line of each boot) on s2 <= g4 + 8.0 ms, and the demoting intruder's e2e (`wall_ms` of
    every demote-arm run) median on s2 <= g4 + 1.0 ms.
promote (d): per order, PIN (the `[prefix-host] promote: .. in Y ms` lines, the second and later of each boot) median
    on s2 <= g4 + 1.0 ms, and the promoting intruder's e2e median on s2 <= g4 + 1.0 ms.
Readings: the copy settle and the owner's hold (demote), the steady polls (promote).
"""
import glob
import json
import os
import re
import statistics
import sys

LEDGER = re.compile(r"demote digests landed off the tick: .*copy settle ([\d.]+) over \d+ poll\(s\).*; owner "
                    r"in-completion [\d.]+ms; wall ([\d.]+)ms t0 to publication")
HELD = re.compile(r"the owner thread held ([\d.]+)ms across the demote")
PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PUB = re.compile(r"promote published off the tick: ticket complete after (\d+) poll\(s\)")
ARMS = ("g4", "s2")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def stat(name, xs):
    return f"{name} N={len(xs)} median={med(xs):.2f} min={min(xs):.2f} max={max(xs):.2f}" if xs else f"{name} N=0"


def boot(d, mode):
    out = {"e2e": [], "wall": [], "copy": [], "held": [], "pin": [], "polls": [], "ok": True, "why": []}
    rec = os.path.join(d, mode, "receipt.json")
    if not os.path.exists(rec):
        out["ok"] = False
        out["why"].append("no receipt")
        return out
    j = json.load(open(rec))
    if j.get("summary", {}).get("errors"):
        out["ok"] = False
        out["why"].append(f"errors={len(j['summary']['errors'])}")
    for r in j["runs"]:
        if r.get("arm") != mode:
            continue
        i = r.get("intruder") or {}
        if "error" in i or "wall_ms" not in i:
            out["ok"] = False
            out["why"].append("intruder error")
            continue
        out["e2e"].append(i["wall_ms"])
    log = os.path.join(d, "server.log")
    k = p = q = 0
    for ln in open(log, errors="replace") if os.path.exists(log) else []:
        m = LEDGER.search(ln)
        if m:
            k += 1
            if k >= 2:
                out["copy"].append(float(m.group(1)))
                out["wall"].append(float(m.group(2)))
                h = HELD.search(ln)
                if h:
                    out["held"].append(float(h.group(1)))
        m = PROMOTE.search(ln)
        if m:
            p += 1
            if p >= 2:
                out["pin"].append(float(m.group(1)))
        m = PUB.search(ln)
        if m:
            q += 1
            if q >= 2:
                out["polls"].append(int(m.group(1)))
    return out


def main():
    mode, root = sys.argv[1], sys.argv[2]
    cells, complete, notes = {}, True, []
    for order in ("o1", "o2"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{arm}")))
            bs = [boot(d, mode) for d in ds]
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
    pool = lambda order, arm, key: [x for b in cells[(order, arm)] for x in b[key]]
    for order in ("o1", "o2"):
        for arm in ARMS:
            polls = pool(order, arm, "polls")
            keys = sorted(set(polls))
            print(f"DAY42 READING mode={mode} order={order} arm={arm} {stat('e2e', pool(order, arm, 'e2e'))} | "
                  f"{stat('wall', pool(order, arm, 'wall'))} | {stat('copy-settle', pool(order, arm, 'copy'))} | "
                  f"{stat('owner-held', pool(order, arm, 'held'))} | {stat('pin', pool(order, arm, 'pin'))} | "
                  f"steady polls {keys} (counts {[polls.count(p) for p in keys]})")
    if not complete:
        print(f"DAY42 S2 {mode.upper()} INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    ok_all = True
    for order in ("o1", "o2"):
        if mode == "demote":
            terms = (("wall", 8.0), ("e2e", 1.0))
        else:
            terms = (("pin", 1.0), ("e2e", 1.0))
        parts = []
        ok = True
        for key, bound in terms:
            g4, s2 = med(pool(order, "g4", key)), med(pool(order, "s2", key))
            ok &= s2 - g4 <= bound
            parts.append(f"{key} g4={g4:.2f} s2={s2:.2f} s2-minus-g4={s2 - g4:+.2f} rule <=+{bound}")
        ok_all &= ok
        clause = "C" if mode == "demote" else "D"
        print(f"DAY42 S2 {clause} order={order} " + " | ".join(parts) + f" -> {'PASS' if ok else 'FAIL'}")
    print(f"DAY42 S2 {mode.upper()} -> {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
