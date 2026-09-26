#!/usr/bin/env python3
"""WP-A design T-H (DAY65.md section 1) reader, written before its sitting runs.

usage: th-reading.py R
(a) every R/gates/*.exit 0 (11 gates on the th binary); the bitwise CPU cell is on record (day65/).
Cells R/{demote,chain,promote}/ab/o{1,2}/bNN-{base,th} (stall_cell.py --mode demote, promote-long, promote; five boots
    per arm per order); R/hump/bNN-x{base,th} (4 boots, --mode demote --n 8). Complete: 20 boots per A/B cell with
    receipts, `errors` empty, 20 `STALL REPLAY: PASS` each; the hump's 4 boots with 16 demote runs each.
Steady: the second and later line of each boot.
(b) demote: th's helper time (`demote helper split: .. (helper Y ms)`) median at most 0.5 x base's, and th's wall t0 to
    publication (`demote digests landed .. wall Xms t0 to publication`) median at most base's - 20 ms, per order.
(c) chain: th's `chain_wall_ms` median at most base's - 30 ms, per order.
(d) every A/B cell: th's tenant stall median at most base's + 1.0 ms; promote: th's PIN (`[prefix-host] promote: .. in Y
    ms`) and intruder e2e medians at most base's + 1.0 ms; the hump (day38's HUMP per boot): th's median at most base's
    + 0.15 ms. Every per-order clause in both orders.
Readings: th's helper thread count and its summed copy and hash thread time.
Verdict: ADOPT when (a) to (d) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics as st
import sys

SPLIT = re.compile(r"demote helper split: ticket seq=\d+ copy ([\d.]+) ms over [\d.]+ MB \(minflt \+-?\d+\), hash "
                   r"([\d.]+) ms \(helper ([\d.]+) ms\)(?:; (\d+) threads)?")
WALL = re.compile(r"demote digests landed off the tick: .*; wall ([\d.]+)ms t0 to publication")
PIN = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
MODES = {"demote": "demote", "chain": "promote-long", "promote": "promote"}


def med(xs):
    return st.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def ab(root, cell):
    mode = MODES[cell]
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "th"):
            ds = sorted(glob.glob(os.path.join(root, cell, "ab", order, f"b*-{arm}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{cell} {order} {arm} boots={len(ds)}")
            p = {k: [] for k in ("stall", "e2e", "chain", "helper", "wall", "pin", "threads", "cpu")}
            for d in ds:
                rec = os.path.join(d, mode, "receipt.json")
                if not os.path.exists(rec):
                    complete = False
                    notes.append(f"{cell}/{order}/{os.path.basename(d)} no receipt")
                    continue
                j = json.load(open(rec))
                if j.get("summary", {}).get("errors"):
                    complete = False
                    notes.append(f"{cell}/{order}/{os.path.basename(d)} errors")
                for r in j["runs"]:
                    if r.get("arm") != mode:
                        continue
                    if "stall_ms" in r:
                        p["stall"].append(r["stall_ms"])
                    i = r.get("intruder") or {}
                    if "wall_ms" in i:
                        p["e2e"].append(i["wall_ms"])
                    if "chain_wall_ms" in i:
                        p["chain"].append(i["chain_wall_ms"])
                text = open(os.path.join(d, "server.log"), errors="replace").read()
                sp = list(SPLIT.finditer(text))[1:]
                p["helper"] += [float(m.group(3)) for m in sp]
                p["cpu"] += [float(m.group(1)) + float(m.group(2)) for m in sp]
                p["threads"] += [int(m.group(4)) for m in sp if m.group(4)]
                p["wall"] += [float(m.group(1)) for m in list(WALL.finditer(text))[1:]]
                p["pin"] += [float(m.group(1)) for m in list(PIN.finditer(text))[1:]]
            pools[(order, arm)] = p
    passes = (rd(os.path.join(root, cell, "ab", "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 20:
        complete = False
        notes.append(f"{cell} replays PASS={passes} of 20")
    return complete, notes, pools


def hump(root):
    by = {}
    for f in sorted(glob.glob(os.path.join(root, "hump", "b*-x*", "demote", "receipt.json"))):
        arm = f.split(os.sep)[-3].split("-")[1]
        rs = [r for r in json.load(open(f))["runs"] if r.get("arm") == "demote" and r.get("itl_ms")]
        itl = [st.median(r["itl_ms"][:22]) for r in rs]
        if len(itl) < 16:
            return None
        base = st.median(itl[:3])
        by.setdefault(arm, []).append(max(itl[3:16]) - base)
    return by


def main():
    root = sys.argv[1]
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a = len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"TH (a) gates {exits}")
    cells = {c: ab(root, c) for c in MODES}
    complete = all(v[0] for v in cells.values())
    notes = [n for v in cells.values() for n in v[1]]
    b_ok, c_ok, d_ok = [], [], []
    for order in ("o1", "o2"):
        for cell, (_, _, pools) in cells.items():
            B, T = pools[(order, "base")], pools[(order, "th")]
            d_ok.append(med(T["stall"]) <= med(B["stall"]) + 1.0)
            line = f"TH READING cell={cell} order={order} stall base={med(B['stall']):.2f} th={med(T['stall']):.2f}"
            if cell == "demote":
                b_ok.append(med(T["helper"]) <= 0.5 * med(B["helper"]) and med(T["wall"]) <= med(B["wall"]) - 20.0)
                line += (f" | helper base={med(B['helper']):.1f} th={med(T['helper']):.1f} ms (th threads "
                         f"{sorted(set(T['threads']))}, thread time {med(T['cpu']):.1f} ms) | wall base="
                         f"{med(B['wall']):.1f} th={med(T['wall']):.1f} ms")
            if cell == "chain":
                c_ok.append(med(T["chain"]) <= med(B["chain"]) - 30.0)
                line += f" | chain base={med(B['chain']):.1f} th={med(T['chain']):.1f} ms"
            if cell == "promote":
                d_ok.append(med(T["pin"]) <= med(B["pin"]) + 1.0 and med(T["e2e"]) <= med(B["e2e"]) + 1.0)
                line += (f" | pin base={med(B['pin']):.2f} th={med(T['pin']):.2f} | e2e base={med(B['e2e']):.1f} "
                         f"th={med(T['e2e']):.1f} ms")
            print(line)
    hb = hump(root)
    if hb is None or set(hb) != {"xbase", "xth"}:
        complete = False
        notes.append("hump incomplete")
    else:
        d_ok.append(med(hb["xth"]) <= med(hb["xbase"]) + 0.15)
        print(f"TH READING hump base={med(hb['xbase']):+.3f} th={med(hb['xth']):+.3f} ms")
    if not complete:
        print(f"TH INCOMPLETE ({'; '.join(notes)}) -> (b) to (d) read nothing; the cells repeat whole once")
    res = {k: complete and all(v) for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok))}
    for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok)):
        print(f"TH ({k}) {'PASS' if res[k] else 'FAIL'} {v}")
    failed = [k for k in ("b", "c", "d") if not res[k]]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the cells whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (T-H is the naked program)"
    print(f"TH VERDICT -> {v}")


if __name__ == "__main__":
    main()
