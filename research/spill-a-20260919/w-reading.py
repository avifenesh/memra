#!/usr/bin/env python3
"""WP-A design W (DAY61.md section 1) reader, written before its sittings run.

usage: w-reading.py --card {5090,target} ROOT
(a) ROOT/unit/run.log's `UNIT` line (the host and pinned identity cells green on w, both red on the red arm with the
    marker printed, the census green) and every ROOT/gates/*.exit 0 (the identity gate default and plain, door OFF
    and ON; the contract fault gate default and plain).
The paired cell ROOT/tier/ab/o{1,2}/bNN-{base,w}-{demote,promote}: five boots per arm per mode per order. Complete: 40
    boots with receipts, `errors` empty, 40 `STALL REPLAY: PASS`, a helper split line in every demote boot and a
    helper-checksum term in every promote boot. Steady: the second and later of each boot's lines.
  helper times: demote, the bind's re-hash share of the helper's job = helper - copy - payload hash (the `demote
    helper split` line); promote, the `Sources` job (`KV checksums on the hash helper (.. in Y ms)`).
  (b) on the 5090: w's median at most half of base's, per mode, per order;
  (c) on the target: w's median at most base's x 1.10, per mode, per order;
  (d) on either: the tenant's stall median and the intruder's e2e median of w at most base's + 1.0 ms, per mode, per
    order.
Verdict per card: PASS when (a), its own clause and (d) pass; otherwise FAIL naming the clauses.
"""
import glob
import json
import os
import re
import statistics
import sys

SPLIT = re.compile(r"\[prefix-host\] demote helper split: ticket seq=(\d+) copy ([\d.]+) ms over [\d.]+ MB \(minflt "
                   r"\+\d+\), hash ([\d.]+) ms \(helper ([\d.]+) ms\)")
SUMS = re.compile(r"KV checksums on the hash helper \(([\d.]+)MB in ([\d.]+)ms\)")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def main():
    card, root = sys.argv[2], sys.argv[3]
    assert sys.argv[1] == "--card" and card in ("5090", "target")
    unit = [ln for ln in (rd(os.path.join(root, "unit", "run.log")) or "").splitlines() if ln.startswith("UNIT ")]
    u = unit[-1] if unit else "UNIT missing"
    a_unit = bool(unit) and all(f"{k}=0" in u for k in ("a1-green", "a2-green", "censuses")) and \
        "a1-red=0 " not in u and "a2-red=0 " not in u and "(marker 0)" not in u
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a_gates = len(exits) == 6 and all(v == "0" for v in exits.values())
    print(f"W (a) {u}")
    print(f"W (a) gates {exits}")
    a = a_unit and a_gates
    base = os.path.join(root, "tier", "ab")
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "w"):
            for mode in ("demote", "promote"):
                ds = sorted(glob.glob(os.path.join(base, order, f"b*-{arm}-{mode}")))
                if len(ds) != 5:
                    complete = False
                    notes.append(f"{order} {arm} {mode} boots={len(ds)}")
                p = {"stall": [], "e2e": [], "helper": []}
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
                    if mode == "demote":
                        hs = [float(m.group(4)) - float(m.group(2)) - float(m.group(3)) for m in SPLIT.finditer(text)]
                    else:
                        hs = [float(m.group(2)) for m in SUMS.finditer(text)]
                    if not hs:
                        complete = False
                        notes.append(f"{order}/{os.path.basename(d)} no helper line")
                    p["helper"] += hs[1:]
                pools[(order, arm, mode)] = p
    passes = (rd(os.path.join(base, "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 40:
        complete = False
        notes.append(f"replays PASS={passes} of 40")
    own, d_ok = [], []
    for order in ("o1", "o2"):
        for mode in ("demote", "promote"):
            g = lambda arm, k: med(pools[(order, arm, mode)][k])
            hb, hw = g("base", "helper"), g("w", "helper")
            sb, sw, eb, ew = g("base", "stall"), g("w", "stall"), g("base", "e2e"), g("w", "e2e")
            own.append(hw <= 0.5 * hb if card == "5090" else hw <= 1.10 * hb)
            d_ok.append(sw <= sb + 1.0 and ew <= eb + 1.0)
            print(f"W READING card={card} order={order} mode={mode} N_helper={len(pools[(order, 'w', mode)]['helper'])} "
                  f"helper base={hb:.2f} w={hw:.2f} ms (ratio {hw / hb if hb else float('nan'):.2f}) | stall base={sb:.2f} "
                  f"w={sw:.2f} | e2e base={eb:.1f} w={ew:.1f} ms")
    clause = "b" if card == "5090" else "c"
    if not complete:
        print(f"W INCOMPLETE ({'; '.join(notes)}) -> the paired cell reads nothing; it repeats whole once")
    print(f"W ({clause}) {'PASS' if complete and all(own) else 'FAIL'} {own}")
    print(f"W (d) {'PASS' if complete and all(d_ok) else 'FAIL'} {d_ok}")
    failed = [k for k, ok in (("a", a), (clause, complete and all(own)), ("d", complete and all(d_ok))) if not ok]
    v = "PASS" if not failed else ("INCOMPLETE" if not complete and a else f"FAIL ({', '.join(failed)})")
    print(f"W VERDICT card={card} -> {v}")


if __name__ == "__main__":
    main()
