#!/usr/bin/env python3
"""WP-A design L (DAY63.md sections 1 and 2) reader, written before its sitting runs.

usage: l-reading.py R
(a) R/unit/run.log's `UNIT` line (both native cells green on l, the length cell red on red with the marker, the
    censuses green) and every R/gates/*.exit 0 (11 gates).
The cells R/{chain,demote}/ab/o{1,2}/bNN-{base,l}: five boots per arm per order each, stall_cell.py --mode
    promote-long (chain) and --mode demote. Complete: 20 boots per cell with receipts, `errors` empty, 20 `STALL
    REPLAY: PASS` per cell.
(b) chain: l's replaced-twin `kv` drop (the `demote publication split` lines with leases, `kv X ms over K leases`, K >
    0) median at most 1.0 ms per order; l's long pre-submit `leases` (`demote pre-submit split` lines of 100 MB or
    more, each boot's third and later: the chain's steady state) median at most 2.0 ms per order.
(c) demote: l's first demote's `spans` (each boot's first `demote pre-submit split` line) median at most 2.0 ms per
    order. Reading: the boot line `span staging set allocated at boot: .. in Y ms`.
(d) both cells: l's tenant stall median and intruder e2e median at most base's + 1.0 ms per order. Reading: the chain's
    `chain_wall_ms`.
Verdict: ADOPT when (a) to (d) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics
import sys

PUBSPLIT = re.compile(r"demote publication split: ticket seq=\d+ .*\(twin: meta [\d.]+ ms, kv ([\d.]+) ms over (\d+) "
                      r"leases")
PRESUB = re.compile(r"demote pre-submit split: ticket seq=\d+ leases ([\d.]+) ms \((\d+) pinned, (?:pooled (\d+) of \d+, )?"
                    r"([\d.]+) MB, minflt \+-?\d+\), register [\d.]+ ms, spans ([\d.]+) ms")
BOOTSTAGE = re.compile(r"span staging set allocated at boot: (\d+) buffers, ([\d.]+) MB in ([\d.]+) ms")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def cell(root, name, mode):
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", "l"):
            ds = sorted(glob.glob(os.path.join(root, name, "ab", order, f"b*-{arm}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{name} {order} {arm} boots={len(ds)}")
            p = {"stall": [], "e2e": [], "chain": [], "twin_kv": [], "leases": [], "pooled": [], "first_spans": [],
                 "boot_stage": []}
            for d in ds:
                rec = os.path.join(d, mode, "receipt.json")
                if not os.path.exists(rec):
                    complete = False
                    notes.append(f"{name}/{order}/{os.path.basename(d)} no receipt")
                    continue
                j = json.load(open(rec))
                if j.get("summary", {}).get("errors"):
                    complete = False
                    notes.append(f"{name}/{order}/{os.path.basename(d)} errors")
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
                p["twin_kv"] += [float(m.group(1)) for m in PUBSPLIT.finditer(text) if int(m.group(2)) > 0]
                pres = list(PRESUB.finditer(text))
                if pres:
                    p["first_spans"].append(float(pres[0].group(5)))
                longs = [m for m in pres if float(m.group(4)) >= 100.0]
                p["leases"] += [float(m.group(1)) for m in longs[2:]]
                p["pooled"] += [int(m.group(3)) for m in longs[2:] if m.group(3) is not None]
                p["boot_stage"] += [float(m.group(3)) for m in BOOTSTAGE.finditer(text)]
            pools[(order, arm)] = p
    passes = (rd(os.path.join(root, name, "ab", "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 20:
        complete = False
        notes.append(f"{name} replays PASS={passes} of 20")
    return complete, notes, pools


def main():
    root = sys.argv[1]
    unit = [ln for ln in (rd(os.path.join(root, "unit", "run.log")) or "").splitlines() if ln.startswith("UNIT ")]
    u = unit[-1] if unit else "UNIT missing"
    a_unit = bool(unit) and all(f"{k}=0" in u for k in ("a1-green", "a2-green", "censuses")) and \
        "a1-red=0 " not in u and "(marker 0)" not in u
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a = a_unit and len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"L (a) {u}")
    print(f"L (a) gates {exits}")
    cc, cn, chain = cell(root, "chain", "promote-long")
    dc, dn, dem = cell(root, "demote", "demote")
    b_ok, c_ok, d_ok = [], [], []
    for order in ("o1", "o2"):
        B, L = chain[(order, "base")], chain[(order, "l")]
        b_ok.append(med(L["twin_kv"]) <= 1.0 and med(L["leases"]) <= 2.0)
        d_ok.append(med(L["stall"]) <= med(B["stall"]) + 1.0 and med(L["e2e"]) <= med(B["e2e"]) + 1.0)
        print(f"L READING cell=chain order={order} twin kv base={med(B['twin_kv']):.2f} l={med(L['twin_kv']):.2f} ms "
              f"(N={len(L['twin_kv'])}) | long leases base={med(B['leases']):.2f} l={med(L['leases']):.2f} ms "
              f"(N={len(L['leases'])}, pooled median {med(L['pooled'])}) | stall base={med(B['stall']):.2f} "
              f"l={med(L['stall']):.2f} | e2e base={med(B['e2e']):.1f} l={med(L['e2e']):.1f} | chain base="
              f"{med(B['chain']):.1f} l={med(L['chain']):.1f} ms")
        B, L = dem[(order, "base")], dem[(order, "l")]
        c_ok.append(med(L["first_spans"]) <= 2.0)
        d_ok.append(med(L["stall"]) <= med(B["stall"]) + 1.0 and med(L["e2e"]) <= med(B["e2e"]) + 1.0)
        print(f"L READING cell=demote order={order} first spans base={med(B['first_spans']):.2f} "
              f"l={med(L['first_spans']):.2f} ms (N={len(L['first_spans'])}) | boot staging l={med(L['boot_stage']):.1f} "
              f"ms | stall base={med(B['stall']):.2f} l={med(L['stall']):.2f} | e2e base={med(B['e2e']):.1f} "
              f"l={med(L['e2e']):.1f} ms")
    complete = cc and dc
    if not complete:
        print(f"L INCOMPLETE ({'; '.join(cn + dn)}) -> (b) to (d) read nothing; the cells repeat whole once")
    res = {k: complete and all(v) for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok))}
    for k, v in (("b", b_ok), ("c", c_ok), ("d", d_ok)):
        print(f"L ({k}) {'PASS' if res[k] else 'FAIL'} {v}")
    failed = [k for k in ("b", "c", "d") if not res[k]]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the cells whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (L is the naked program)"
    print(f"L VERDICT -> {v}")


if __name__ == "__main__":
    main()
