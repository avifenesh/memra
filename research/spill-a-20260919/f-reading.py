#!/usr/bin/env python3
"""WP-A design F (DAY64.md section 5) reader, written before its sitting runs.

usage: f-reading.py R
(a) R/unit/run.log's `UNIT native=0 censuses=0` and every R/gates/*.exit 0 (11 gates).
The promote cell R/promote/ab/o{1,2}/bNN-{base,f} (stall_cell.py --mode promote, five boots per arm per order) and the
    hump R/hump/bNN-x{base,f}. Complete: 20 boots with receipts, `errors` empty, 20 `STALL REPLAY: PASS`; the hump's 4
    boots with 16 demote runs each.
Steady: the second and later `promote published off the tick` line of each boot. LATE: published after the next tick
    top (publication tick > submission tick + 1), day64-reading.py's rule.
(b) f's late promotes at most 5% of its steady promotes (9 of 180), pooled over both orders.
(c) f's intruder e2e median at most base's - 5.0 ms and f's PIN (`[prefix-host] promote: .. in Y ms`) median at most
    base's + 1.0 ms, per order.
(d) f's tenant stall median at most base's + 1.0 ms per order; f's median HUMP at most base's + 0.15 ms.
Readings: the late count on base; the span receipt's phases per arm.
Verdict: ADOPT when (a) to (d) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics as st
import sys

PUB = re.compile(r"promote published off the tick: .*timeline from t0: submitted [+-][\d.]+ms \(tick (\d+), its top "
                 r"[+-][\d.]+ms\);.* published [+-][\d.]+ms \(tick (\d+), its top")
PIN = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PH = re.compile(r"span receipt: fill ([\d.]+) ms, copies ([\d.]+) ms, digests ([\d.]+) ms")


def med(xs):
    return st.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def main():
    root = sys.argv[1]
    unit = [ln for ln in (rd(os.path.join(root, "unit", "run.log")) or "").splitlines() if ln.startswith("UNIT ")]
    u = unit[-1] if unit else "UNIT missing"
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a = u == "UNIT native=0 censuses=0" and len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"F (a) {u}")
    print(f"F (a) gates {exits}")
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "f"):
            ds = sorted(glob.glob(os.path.join(root, "promote", "ab", order, f"b*-{arm}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{order} {arm} boots={len(ds)}")
            p = {"stall": [], "e2e": [], "pin": [], "steady": 0, "late": 0, "ph": []}
            for d in ds:
                rec = os.path.join(d, "promote", "receipt.json")
                if not os.path.exists(rec):
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} no receipt")
                    continue
                j = json.load(open(rec))
                if j.get("summary", {}).get("errors"):
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} errors")
                for r in j["runs"]:
                    if r.get("arm") != "promote":
                        continue
                    if "stall_ms" in r:
                        p["stall"].append(r["stall_ms"])
                    i = r.get("intruder") or {}
                    if "wall_ms" in i:
                        p["e2e"].append(i["wall_ms"])
                text = open(os.path.join(d, "server.log"), errors="replace").read()
                pubs = list(PUB.finditer(text))[1:]
                p["steady"] += len(pubs)
                p["late"] += sum(1 for m in pubs if int(m.group(2)) > int(m.group(1)) + 1)
                p["pin"] += [float(m.group(1)) for m in list(PIN.finditer(text))[1:]]
                p["ph"] += [tuple(float(x) for x in m.groups()) for m in list(PH.finditer(text))[1:]]
            pools[(order, arm)] = p
    passes = (rd(os.path.join(root, "promote", "ab", "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 20:
        complete = False
        notes.append(f"replays PASS={passes} of 20")
    c_ok, d_ok = [], []
    late_f = sum(pools[(o, "f")]["late"] for o in ("o1", "o2"))
    steady_f = sum(pools[(o, "f")]["steady"] for o in ("o1", "o2"))
    late_b = sum(pools[(o, "base")]["late"] for o in ("o1", "o2"))
    steady_b = sum(pools[(o, "base")]["steady"] for o in ("o1", "o2"))
    b_ok = steady_f > 0 and late_f <= 0.05 * steady_f
    for order in ("o1", "o2"):
        B, F = pools[(order, "base")], pools[(order, "f")]
        c_ok.append(med(F["e2e"]) <= med(B["e2e"]) - 5.0 and med(F["pin"]) <= med(B["pin"]) + 1.0)
        d_ok.append(med(F["stall"]) <= med(B["stall"]) + 1.0)
        phase = lambda P, k: med([x[k] for x in P["ph"]])
        print(f"F READING order={order} late base={B['late']}/{B['steady']} f={F['late']}/{F['steady']} | e2e base="
              f"{med(B['e2e']):.1f} f={med(F['e2e']):.1f} | pin base={med(B['pin']):.2f} f={med(F['pin']):.2f} | stall "
              f"base={med(B['stall']):.2f} f={med(F['stall']):.2f} ms | phases base fill {phase(B, 0):.2f} copies "
              f"{phase(B, 1):.2f} digests {phase(B, 2):.2f}; f fill {phase(F, 0):.2f} copies {phase(F, 1):.2f} "
              f"digests {phase(F, 2):.2f} ms")
    by = {}
    for fpath in sorted(glob.glob(os.path.join(root, "hump", "b*-x*", "demote", "receipt.json"))):
        arm = fpath.split(os.sep)[-3].split("-")[1]
        rs = [r for r in json.load(open(fpath))["runs"] if r.get("arm") == "demote" and r.get("itl_ms")]
        itl = [st.median(r["itl_ms"][:22]) for r in rs]
        if len(itl) >= 16:
            by.setdefault(arm, []).append(max(itl[3:16]) - st.median(itl[:3]))
    if set(by) != {"xbase", "xf"}:
        complete = False
        notes.append("hump incomplete")
    else:
        d_ok.append(med(by["xf"]) <= med(by["xbase"]) + 0.15)
        print(f"F READING hump base={med(by['xbase']):+.3f} f={med(by['xf']):+.3f} ms")
    print(f"F READING late pooled base={late_b}/{steady_b} f={late_f}/{steady_f}")
    if not complete:
        print(f"F INCOMPLETE ({'; '.join(notes)}) -> (b) to (d) read nothing; the cells repeat whole once")
    res = {"b": complete and b_ok, "c": complete and all(c_ok), "d": complete and all(d_ok)}
    for k, v in (("b", [b_ok]), ("c", c_ok), ("d", d_ok)):
        print(f"F ({k}) {'PASS' if res[k] else 'FAIL'} {v}")
    failed = [k for k in ("b", "c", "d") if not res[k]]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the cells whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (F is the naked program)"
    print(f"F VERDICT -> {v}")


if __name__ == "__main__":
    main()
