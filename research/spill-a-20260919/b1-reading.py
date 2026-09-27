#!/usr/bin/env python3
"""WP-A design B1 (DAY59.md section 7) reader, written before its sitting runs.

usage: b1-reading.py ROOT
(a) ROOT/unit/run.log's `UNIT` line: a1 and a2 green on b1, both red on the red arm with the marker printed, the
    censuses green; ROOT/gates/identity-*.exit all 0; ROOT/gates/hitgate-{off,on}.exit both 0.
(b) to (d) ROOT/short/ab/o{1,2}/bNN-{base,b1}-{fanout,prime-short}: five boots per arm per mode per order. Complete:
    40 boots with receipts, `errors` empty, 40 `STALL REPLAY: PASS`, a split line for every fanout publish line.
    Steady: the second and later fanout tick of each fanout boot.
    (b) b1's snapshot-plus-restores median <= 0.5 x base's, per order;
    (c) base's (fanout - prime-short) stall minus b1's >= 1.0 ms, in both orders;
    (d) b1's fanout members' wall_ms median <= base's + 1.0 ms, per order.
Verdict: ADOPT when (a) to (d) all pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics
import sys

PUB = re.compile(r"\[prefix-dedup\] on-tick publish: snapshot ([\d.]+) ms \(([\d.]+) MB\), (\d+) sibling restore\(s\) "
                 r"([\d.]+) ms, insert ([\d.]+) ms")
SPLIT = re.compile(r"\[prefix-dedup\] on-tick split: ")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def main():
    root = sys.argv[1]
    unit = [ln for ln in (rd(os.path.join(root, "unit", "run.log")) or "").splitlines() if ln.startswith("UNIT ")]
    u = unit[-1] if unit else "UNIT missing"
    a_unit = bool(unit) and all(f"{k}=0" in u for k in ("a1-green", "a2-green", "censuses")) and \
        "a1-red=0 " not in u and "a2-red=0 " not in u and "(marker 0)" not in u
    gates = {}
    for n in ("identity-default-off", "identity-default-on", "identity-plain-off", "identity-plain-on"):
        gates[n] = rd(os.path.join(root, "gates", f"{n}.exit"))
    for n in ("off", "on"):
        gates[f"hitgate-{n}"] = rd(os.path.join(root, "gates", f"hitgate-{n}.exit"))
    a_gates = all(v == "0" for v in gates.values())
    print(f"B1 (a) {u}")
    print(f"B1 (a) gates {gates}")
    a = a_unit and a_gates
    base = os.path.join(root, "short", "ab")
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "b1"):
            for mode in ("fanout", "prime-short"):
                ds = sorted(glob.glob(os.path.join(base, order, f"b*-{arm}-{mode}")))
                if len(ds) != 5:
                    complete = False
                    notes.append(f"{order} {arm} {mode} boots={len(ds)}")
                p = {"stall": [], "own": [], "members": []}
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
                        for m in (r.get("intruder") or {}).get("fanout") or []:
                            p["members"].append(m["wall_ms"])
                    if mode != "fanout":
                        continue
                    lines = open(os.path.join(d, "server.log"), errors="replace").read().splitlines()
                    pubs = [m for m in map(PUB.search, lines) if m]
                    nsplit = sum(1 for ln in lines if SPLIT.search(ln))
                    if len(pubs) != nsplit:
                        complete = False
                        notes.append(f"{order}/{os.path.basename(d)} publish={len(pubs)} split={nsplit}")
                    for m in pubs[1:]:
                        p["own"].append(float(m.group(1)) + float(m.group(4)))
                pools[(order, arm, mode)] = p
    rp = os.path.join(base, "replays.log")
    passes = (rd(rp) or "").count("STALL REPLAY: PASS")
    if passes != 40:
        complete = False
        notes.append(f"replays PASS={passes} of 40")
    res = {"b": [], "c": [], "d": []}
    for order in ("o1", "o2"):
        g = lambda arm, mode, k: med(pools[(order, arm, mode)][k])
        own_b, own_1 = g("base", "fanout", "own"), g("b1", "fanout", "own")
        pr_b = g("base", "fanout", "stall") - g("base", "prime-short", "stall")
        pr_1 = g("b1", "fanout", "stall") - g("b1", "prime-short", "stall")
        mem_b, mem_1 = g("base", "fanout", "members"), g("b1", "fanout", "members")
        res["b"].append(own_1 <= 0.5 * own_b)
        res["c"].append(pr_b - pr_1 >= 1.0)
        res["d"].append(mem_1 <= mem_b + 1.0)
        print(f"B1 READING order={order} N_own={len(pools[(order, 'b1', 'fanout')]['own'])} "
              f"own base={own_b:.2f} b1={own_1:.2f} ms | fanout-minus-prime base={pr_b:+.2f} b1={pr_1:+.2f} "
              f"(gain {pr_b - pr_1:+.2f}) ms | members wall base={mem_b:.1f} b1={mem_1:.1f} ms")
    if not complete:
        print(f"B1 INCOMPLETE ({'; '.join(notes)}) -> (b) to (d) read nothing; the cell repeats whole once")
    for k in ("b", "c", "d"):
        print(f"B1 ({k}) {'PASS' if complete and all(res[k]) else 'FAIL'} per order {res[k]}")
    failed = [k for k in ("b", "c", "d") if not (complete and all(res[k]))]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the paired cell whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (B1 is the naked program)"
    print(f"B1 VERDICT -> {v}")


if __name__ == "__main__":
    main()
