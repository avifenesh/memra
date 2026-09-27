#!/usr/bin/env python3
"""WP-A design R1 (DAY62.md section 8) reader, written before its sitting runs.

usage: r1-reading.py ROOT
(a) every ROOT/gates/*.exit 0 (11 gates on the r1 binary).
The paired cell ROOT/seam/ab/o{1,2}/bNN-{base,r1}-{retire-seam-nosource,retire-seam,prime}: five boots per arm per
    mode per order. Complete: 60 boots with receipts, `errors` empty, 60 `STALL REPLAY: PASS`.
(b) retire-seam-nosource: r1's no-source retire hold (the median of any `source retiring: no` settle on r1; 0.0 when
    none) at most 0.5 ms per order, and r1's skip-line count within 10% of base's no-source settle count per order.
(c) every mode: r1's tenant stall median and intruder e2e median (in retire-seam-nosource also the short's e2e) at
    most base's + 1.0 ms, per order.
(d) where the wait goes: per mode and order, r1's `a second capture` and `source retiring: yes` settle medians at most
    base's + 1.0 ms (a kind absent on base and present on r1 compares against 0.0).
Readings: the tenant's mid-gap excess (every gap in (p50 + 5, 150) ms, summed per run) medians.
Verdict: ADOPT when (a) to (d) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics
import sys

SETTLE = re.compile(r"settled synchronously by (a session retire \(source retiring: (?:yes|no)\)|a second capture); the "
                    r"settle held the owner thread ([\d.]+)ms")
SKIP = "retire with a capture pending: no retiring session is its source"
MODES = ("retire-seam-nosource", "retire-seam", "prime")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def main():
    root = sys.argv[1]
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a = len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"R1 (a) gates {exits}")
    base = os.path.join(root, "seam", "ab")
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "r1"):
            for mode in MODES:
                ds = sorted(glob.glob(os.path.join(base, order, f"b*-{arm}-{mode}")))
                if len(ds) != 5:
                    complete = False
                    notes.append(f"{order} {arm} {mode} boots={len(ds)}")
                p = {"stall": [], "e2e": [], "short": [], "mid": [], "skip": 0,
                     "no": [], "yes": [], "second": []}
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
                    for r in j["runs"]:
                        if r.get("arm") != mode:
                            continue
                        if "stall_ms" in r:
                            p["stall"].append(r["stall_ms"])
                            p["mid"].append(sum(g - r["p50"] for g in r["itl_ms"] if r["p50"] + 5 < g < 150))
                        i = r.get("intruder") or {}
                        if "wall_ms" in i:
                            p["e2e"].append(i["wall_ms"])
                        if "short_wall_ms" in (i.get("seam") or {}):
                            p["short"].append(i["seam"]["short_wall_ms"])
                    text = open(os.path.join(d, "server.log"), errors="replace").read()
                    p["skip"] += text.count(SKIP)
                    for m in SETTLE.finditer(text):
                        k = "second" if m.group(1) == "a second capture" else ("yes" if "yes" in m.group(1) else "no")
                        p[k].append(float(m.group(2)))
                pools[(order, arm, mode)] = p
    passes = (rd(os.path.join(base, "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 60:
        complete = False
        notes.append(f"replays PASS={passes} of 60")
    b_ok, c_ok, d_ok = [], [], []
    for order in ("o1", "o2"):
        for mode in MODES:
            B, R = pools[(order, "base", mode)], pools[(order, "r1", mode)]
            c = med(R["stall"]) <= med(B["stall"]) + 1.0 and med(R["e2e"]) <= med(B["e2e"]) + 1.0
            if mode == "retire-seam-nosource":
                c = c and med(R["short"]) <= med(B["short"]) + 1.0
                hold = med(R["no"]) if R["no"] else 0.0
                n_base = len(B["no"])
                b_ok.append(hold <= 0.5 and n_base > 0 and abs(R["skip"] - n_base) <= 0.1 * n_base)
            c_ok.append(c)
            for k in ("second", "yes"):
                if R[k]:
                    d_ok.append(med(R[k]) <= (med(B[k]) if B[k] else 0.0) + 1.0)
            print(f"R1 READING order={order} mode={mode} | no-source settles base N={len(B['no'])} median="
                  f"{med(B['no']):.2f} r1 N={len(R['no'])} skips r1={R['skip']} | second-capture base N="
                  f"{len(B['second'])} {med(B['second']):.2f} r1 N={len(R['second'])} {med(R['second']):.2f} | "
                  f"source-retire base N={len(B['yes'])} {med(B['yes']):.2f} r1 N={len(R['yes'])} {med(R['yes']):.2f} | "
                  f"stall base={med(B['stall']):.2f} r1={med(R['stall']):.2f} | e2e base={med(B['e2e']):.1f} "
                  f"r1={med(R['e2e']):.1f} short base={med(B['short']):.1f} r1={med(R['short']):.1f} | mid-gap "
                  f"base={med(B['mid']):.1f} r1={med(R['mid']):.1f} ms")
    if not complete:
        print(f"R1 INCOMPLETE ({'; '.join(notes)}) -> (b) to (d) read nothing; the cell repeats whole once")
    res = {k: complete and all(v) for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok))}
    for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok)):
        print(f"R1 ({k}) {'PASS' if res[k] else 'FAIL'} {v}")
    failed = [k for k in ("b", "c", "d") if not res[k]]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the paired cell whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (R1 is the naked program)"
    print(f"R1 VERDICT -> {v}")


if __name__ == "__main__":
    main()
