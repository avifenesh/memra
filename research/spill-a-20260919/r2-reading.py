#!/usr/bin/env python3
"""WP-A design R2 (DAY62.md section 5) reader, written before its sitting runs.

usage: r2-reading.py ROOT
(a) every ROOT/gates/*.exit 0 (the identity gate default and plain door OFF and ON, the fault and contract fault gates
    default and plain, the pause gate, the hit gate OFF and ON: 11); no `retire QUARANTINED` line in any r2 boot.
The paired cell ROOT/seam/ab/o{1,2}/bNN-{base,r2}-{prime,retire-seam}: five boots per arm per mode per order.
    Complete: 40 boots with receipts, `errors` empty, 40 `STALL REPLAY: PASS`; every r2 boot has deferral lines and
    as many `retire released` lines as deferrals.
    (b) r2's deferral hold (the `retire deferred: .. (X ms)` line) median at most 0.5 ms, per mode, per order;
    (c) r2's tenant stall median and intruder e2e median at most base's + 1.0 ms, per mode, per order.
Readings: base's retire settle hold (`source retiring: yes`), r2's stall against base's.
Verdict: ADOPT when (a) to (c) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics
import sys

DEFER = re.compile(r"\[prefix-cache\] retire deferred: .*ticket seq=\d+; ([\d.]+) ms\)")
RELEASED = re.compile(r"\[prefix-cache\] retire released: .*the landing observed")
QUAR = "[prefix-cache] retire QUARANTINED"
HOLD = re.compile(r"settled synchronously by a session retire \(source retiring: yes\); the settle held the owner "
                  r"thread ([\d.]+)ms")


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
    a_gates = len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"R2 (a) gates {exits}")
    base = os.path.join(root, "seam", "ab")
    complete, notes, pools, quarantined = True, [], {}, 0
    for order in ("o1", "o2"):
        for arm in ("base", "r2"):
            for mode in ("prime", "retire-seam"):
                ds = sorted(glob.glob(os.path.join(base, order, f"b*-{arm}-{mode}")))
                if len(ds) != 5:
                    complete = False
                    notes.append(f"{order} {arm} {mode} boots={len(ds)}")
                p = {"stall": [], "e2e": [], "defer": [], "hold": []}
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
                        i = r.get("intruder") or {}
                        if "wall_ms" in i:
                            p["e2e"].append(i["wall_ms"])
                    text = open(os.path.join(d, "server.log"), errors="replace").read()
                    defers = [float(m.group(1)) for m in DEFER.finditer(text)]
                    p["defer"] += defers
                    p["hold"] += [float(m.group(1)) for m in HOLD.finditer(text)]
                    if arm == "r2":
                        quarantined += text.count(QUAR)
                        if not defers or len(RELEASED.findall(text)) != len(defers):
                            complete = False
                            notes.append(f"{order}/{os.path.basename(d)} deferrals={len(defers)} "
                                         f"released={len(RELEASED.findall(text))}")
                pools[(order, arm, mode)] = p
    passes = (rd(os.path.join(base, "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 40:
        complete = False
        notes.append(f"replays PASS={passes} of 40")
    a = a_gates and quarantined == 0
    print(f"R2 (a) quarantined lines in r2 boots: {quarantined}")
    b_ok, c_ok = [], []
    for order in ("o1", "o2"):
        for mode in ("prime", "retire-seam"):
            g = lambda arm, k: med(pools[(order, arm, mode)][k])
            dr = g("r2", "defer")
            sb, sr, eb, er = g("base", "stall"), g("r2", "stall"), g("base", "e2e"), g("r2", "e2e")
            b_ok.append(dr <= 0.5)
            c_ok.append(sr <= sb + 1.0 and er <= eb + 1.0)
            print(f"R2 READING order={order} mode={mode} base hold={g('base', 'hold'):.2f} ms (N="
                  f"{len(pools[(order, 'base', mode)]['hold'])}) | r2 deferral={dr:.3f} ms (N="
                  f"{len(pools[(order, 'r2', mode)]['defer'])}), r2 retire settles={len(pools[(order, 'r2', mode)]['hold'])}"
                  f" | stall base={sb:.2f} r2={sr:.2f} ({sr - sb:+.2f}) | e2e base={eb:.1f} r2={er:.1f} ms")
    if not complete:
        print(f"R2 INCOMPLETE ({'; '.join(notes)}) -> (b), (c) read nothing; the cell repeats whole once")
    print(f"R2 (b) {'PASS' if complete and all(b_ok) else 'FAIL'} {b_ok}")
    print(f"R2 (c) {'PASS' if complete and all(c_ok) else 'FAIL'} {c_ok}")
    failed = [k for k, ok in (("b", complete and all(b_ok)), ("c", complete and all(c_ok))) if not ok]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the paired cell whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (R2 is the naked program)"
    print(f"R2 VERDICT -> {v}")


if __name__ == "__main__":
    main()
