#!/usr/bin/env python3
"""WP-A day 52 (DAY52.md section 1, OWED item 17) reader: design P2 against base, written before its cells run
(day51-reading.py with p2 in p's place, the chain cell's third arm p, the placing rule and the arming readings).

usage: day52-reading.py R
Input (the pro-single-p2 sitting's layout):
  R/{demote,free,promote,chain}/ab/o{1,2}/bNN-{base,p2[,p]}/{server.log,rss.txt,<mode>/receipt.json} and each
  R/<cell>/ab/replays.log (stall_cell.py; demote and free run --mode demote, promote --mode promote, chain
  --mode promote-long; --n 5; 20 boots per cell, five per arm per order; the chain cell 30 boots with p);
  R/hump/reading-hump.log (day38-hump-reading.py over xgpp xp2 xp2 xgpp);
  R/gates/*.exit, R/gates/hitgate-{off,on}.exit and R/unit/run.log (clause (a) on the card).
Complete, per A/B cell: five boots per arm per order, each with a receipt, `errors` empty, every intruder without an
error (and with `chain_wall_ms` in chain), one `STALL REPLAY: PASS` per boot, and in demote and free one `demote helper split`
line per `demote digests landed` line. An incomplete cell is not read (it repeats whole once).
Steady: the second and later demote of a boot in demote, the fourth and later in free; the second and later promote.
Terms, per order, medians over an arm's pooled boots:
  wall   the steady `demote digests landed off the tick: .. wall Xms t0 to publication`
  e2e    the intruder's `wall_ms` (demote, free, promote)
  copy   the steady `demote helper split: .. copy X ms over B MB (minflt +M) ..` X, with M as `minflt`
  hits   the fraction of steady demotes reading `reserve N of N staged` with N > 0
  pin    the steady `[prefix-host] promote: .. in Y ms`
  chain  chain's `chain_wall_ms`; first: chain's `wall_ms`
The rule (P2 is the door's copy program on this class only if every clause holds, each in both orders; p2 in p's place):
  (a) every gate exit 0 (identity x4, failure x2, contract-fault x2, twin x2, pause-demote, hitgate off and on) and
      the unit cells' last line all green
  (b) demote: p minflt <= 0.25 x pages (pages = p's copy bytes / 4096); p copy <= base copy - 8.0; p hits >= 0.90
  (c) demote: p wall <= base wall - 8.0; p e2e <= base e2e + 1.0
  (d) promote: p pin <= base pin + 1.0; p e2e <= base e2e + 1.0
  (e) hump: xp2 median HUMP <= 0.15 with the xgpp control above 0.15 (a control at or below 0.15: UNREAD)
  (f) free: p wall <= base wall + 2.0; p copy <= base copy + 1.0
  (g) chain: p chain <= base chain + 1.0; p first <= base first + 1.0
Readings: the refill lines (`payload reserve ready: N buffers, B MB in X ms (minflt +M, yielded Y time(s))`), the
arming lines (`payload reserve armed|disarmed: ..`, per boot: how many, and the demote count at the first disarm),
VmRSS / VmHWM per arm per cell, and the publication split per arm per cell (the third and later split lines of each
boot: bind, reclaim, insert, twin meta, kv, f32, rest, evictions, pause release, and `publish` = the landed line's
`take-back bind and publish`).
The placing rule (chain cell, p against base, per order; decides no adoption): D = the publish delta; a group whose
median delta is at least D / 2 in both orders (D > 0) is named the place of P's millisecond; otherwise `not placed`.
"""
import glob
import json
import os
import re
import statistics
import sys

LEDGER = re.compile(r"demote digests landed off the tick: .*; owner in-completion [\d.]+ms; wall ([\d.]+)ms t0 to "
                    r"publication")
SPLIT = re.compile(r"demote helper split: ticket seq=\d+ copy ([\d.]+) ms over ([\d.]+) MB \(minflt \+(-?\d+)\), "
                   r"hash [\d.]+ ms \(helper [\d.]+ ms\)(?:; reserve (\d+) of (\d+) staged)?")
PROMOTE = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
REFILL = re.compile(r"payload reserve ready: (\d+) buffers, ([\d.]+) MB in ([\d.]+) ms \(minflt \+(-?\d+), yielded "
                    r"(\d+) time\(s\)\)")
RSS = re.compile(r"^(VmRSS|VmHWM):\s+(\d+) kB", re.M)
PUBSPLIT = re.compile(r"demote publication split: ticket seq=\d+ bind ([\d.]+) ms, reclaim ([\d.]+) ms, insert ([\d.]+) ms "
                      r"\(twin: meta ([\d.]+) ms, kv ([\d.]+) ms over \d+ leases, f32 ([\d.]+) ms over \d+ payloads, rest "
                      r"([\d.]+) ms; \d+ evicted in ([\d.]+) ms\), pause release ([\d.]+) ms")
PUBLISH = re.compile(r"take-back bind and publish ([\d.]+); owner")
ARMING = re.compile(r"payload reserve (armed|disarmed): the job took")
GROUPS = ("bind", "reclaim", "insert", "meta", "kv", "f32", "rest", "evict", "pause")
ARMS = ("base", "p2")
ARMS_OF = {"demote": ("base", "p2"), "free": ("base", "p2"), "promote": ("base", "p2"), "chain": ("base", "p", "p2")}
CELLS = {"demote": "demote", "free": "demote", "promote": "promote", "chain": "promote-long"}
STEADY_FROM = {"demote": 1, "free": 3, "promote": 1, "chain": 1}
GATES = ("identity-default-off", "identity-default-on", "identity-plain-off", "identity-plain-on", "failure-off",
         "failure-on", "contract-fault", "contract-fault-plain", "twin-off", "twin-on", "pause-demote", "hitgate-off",
         "hitgate-on")
UNIT_GREEN = re.compile(r"unit-cells parallel=3/3 engine-serial-rc=0 door-rc=0 cpu-rc=0 engine-census-rc=0 tier-rc=0")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def boot(d, cell):
    mode = CELLS[cell]
    out = {k: [] for k in ("wall", "e2e", "copy", "minflt", "mb", "hit", "pin", "chain", "first", "refill_ms",
                           "refill_minflt", "refill_yields", "rss", "hwm", "publish", "insert_other") + GROUPS}
    out["arming"], out["first_disarm"] = [], []
    out["ok"], out["why"] = True, []
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
        if "error" in i or "wall_ms" not in i or (mode == "promote-long" and "chain_wall_ms" not in i):
            out["ok"] = False
            out["why"].append("intruder error")
            continue
        if mode == "promote-long":
            out["first"].append(i["wall_ms"])
            out["chain"].append(i["chain_wall_ms"])
        else:
            out["e2e"].append(i["wall_ms"])
    k = STEADY_FROM[cell]
    ledgers, splits, pins, pubs, publish = [], [], [], [], []
    for ln in open(os.path.join(d, "server.log"), errors="replace"):
        m = PUBSPLIT.search(ln)
        if m:
            pubs.append([float(x) for x in m.groups()])
        m = PUBLISH.search(ln)
        if m:
            publish.append(float(m.group(1)))
        m = ARMING.search(ln)
        if m:
            out["arming"].append((m.group(1), len(ledgers)))
        m = LEDGER.search(ln)
        if m:
            ledgers.append(float(m.group(1)))
        m = SPLIT.search(ln)
        if m:
            splits.append(m)
        m = PROMOTE.search(ln)
        if m:
            pins.append(float(m.group(1)))
        m = REFILL.search(ln)
        if m:
            out["refill_ms"].append(float(m.group(3)))
            out["refill_minflt"].append(int(m.group(4)))
            out["refill_yields"].append(int(m.group(5)))
    if cell in ("demote", "free"):
        if len(splits) != len(ledgers) or len(ledgers) <= k:
            out["ok"] = False
            out["why"].append(f"split lines={len(splits)} ledger lines={len(ledgers)}")
        out["wall"] = ledgers[k:]
        for m in splits[k:]:
            out["copy"].append(float(m.group(1)))
            out["mb"].append(float(m.group(2)))
            out["minflt"].append(int(m.group(3)))
            hits, staged = (int(m.group(4)), int(m.group(5))) if m.group(4) is not None else (0, 0)
            out["hit"].append(1.0 if staged > 0 and hits == staged else 0.0)
    if cell == "promote":
        out["pin"] = pins[k:]
    for row in pubs[2:]:
        for name, v in zip(GROUPS, row):
            out[name].append(v)
        out["insert_other"].append(row[2] - sum(row[3:8]))
    out["publish"] = publish[2:]
    disarms = [n for state, n in out["arming"] if state == "disarmed"]
    out["first_disarm"] = disarms[:1]
    rss = os.path.join(d, "rss.txt")
    if os.path.exists(rss):
        for name, kb in RSS.findall(open(rss).read()):
            out["rss" if name == "VmRSS" else "hwm"].append(int(kb) / 1024)
    return out


def read_cell(root, cell):
    base = os.path.join(root, cell)
    cells, complete, notes = {}, True, []
    for order in ("o1", "o2"):
        for arm in ARMS_OF[cell]:
            ds = sorted(glob.glob(os.path.join(base, "ab", order, f"b*-{arm}")))
            bs = [boot(d, cell) for d in ds]
            if len(bs) != 5:
                complete = False
                notes.append(f"{order} {arm} boots={len(bs)}")
            for d, b in zip(ds, bs):
                if not b["ok"]:
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} {';'.join(b['why'])}")
            cells[(order, arm)] = bs
    replays = os.path.join(base, "ab", "replays.log")
    passes = open(replays, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(replays) else 0
    want = 10 * len(ARMS_OF[cell])
    if passes != want:
        complete = False
        notes.append(f"replays PASS={passes} of {want}")
    return cells, complete, notes


def main():
    root = sys.argv[1]
    # WP-A day 67 (`DAY67.md` section 1): a sitting without DAY52's diagnostic `p` arm reads the chain cell as base
    # against p2; the placing rule then prints `not placed` with its deltas absent.
    if not glob.glob(os.path.join(root, "chain", "ab", "o*", "b*-p")):
        ARMS_OF["chain"] = ("base", "p2")
    failed, unread = False, False
    pools = {}
    for cell in ("demote", "free", "promote", "chain"):
        cells, complete, notes = read_cell(root, cell)
        pool = lambda order, arm, key, cells=cells: [x for b in cells.get((order, arm), []) for x in b[key]]
        pools[cell] = pool
        for order in ("o1", "o2"):
            for arm in ARMS_OF[cell]:
                keys = {"demote": ("wall", "e2e", "copy", "minflt", "hit"), "free": ("wall", "e2e", "copy", "minflt",
                        "hit"), "promote": ("pin", "e2e"), "chain": ("chain", "first")}[cell]
                terms = " ".join(f"{k}={med(pool(order, arm, k)):.2f}(N={len(pool(order, arm, k))})" for k in keys)
                rf = pool(order, arm, "refill_ms")
                print(f"DAY52 READING cell={cell} order={order} arm={arm} {terms} | refill N={len(rf)} "
                      f"ms={med(rf):.2f} minflt={med(pool(order, arm, 'refill_minflt')):.0f} "
                      f"yields={sum(pool(order, arm, 'refill_yields'))} | VmRSS={med(pool(order, arm, 'rss')):.0f} MB "
                      f"VmHWM={med(pool(order, arm, 'hwm')):.0f} MB")
                arms_lines = [b["arming"] for b in cells.get((order, arm), [])]
                split = " ".join(f"{g}={med(pool(order, arm, g)):.2f}" for g in ("publish",) + GROUPS + ("insert_other",))
                print(f"DAY52 SPLIT cell={cell} order={order} arm={arm} N={len(pool(order, arm, 'publish'))} {split} | "
                      f"arming lines per boot={[len(a) for a in arms_lines]} first disarm at demote="
                      f"{[b['first_disarm'][0] if b['first_disarm'] else None for b in cells.get((order, arm), [])]}")
        if not complete:
            print(f"DAY52 CELL {cell} INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
            unread = True
            pools[cell] = None
    rules = {
        "b": ("demote", lambda P, B, o: [
            ("minflt", med(P(o, "p2", "minflt")), 0.25 * med(P(o, "p2", "mb")) * 1e6 / 4096, "<=", "p2-median (0.25 x pages)"),
            ("copy", med(P(o, "p2", "copy")) - med(P(o, "base", "copy")), -8.0, "<=", "p2-minus-base"),
            ("hits", sum(P(o, "p2", "hit")) / max(1, len(P(o, "p2", "hit"))), 0.90, ">=", "fraction of steady demotes"),
        ]),
        "c": ("demote", lambda P, B, o: [
            ("wall", med(P(o, "p2", "wall")) - med(P(o, "base", "wall")), -8.0, "<=", "p2-minus-base"),
            ("e2e", med(P(o, "p2", "e2e")) - med(P(o, "base", "e2e")), 1.0, "<=", "p2-minus-base"),
        ]),
        "d": ("promote", lambda P, B, o: [
            ("pin", med(P(o, "p2", "pin")) - med(P(o, "base", "pin")), 1.0, "<=", "p2-minus-base"),
            ("e2e", med(P(o, "p2", "e2e")) - med(P(o, "base", "e2e")), 1.0, "<=", "p2-minus-base"),
        ]),
        "f": ("free", lambda P, B, o: [
            ("wall", med(P(o, "p2", "wall")) - med(P(o, "base", "wall")), 2.0, "<=", "p2-minus-base"),
            ("copy", med(P(o, "p2", "copy")) - med(P(o, "base", "copy")), 1.0, "<=", "p2-minus-base"),
        ]),
        "g": ("chain", lambda P, B, o: [
            ("chain", med(P(o, "p2", "chain")) - med(P(o, "base", "chain")), 1.0, "<=", "p2-minus-base"),
            ("first", med(P(o, "p2", "first")) - med(P(o, "base", "first")), 1.0, "<=", "p2-minus-base"),
        ]),
    }
    for clause, (cell, terms) in rules.items():
        if pools[cell] is None:
            print(f"DAY52 P2 ({clause}) cell={cell} -> UNREAD (incomplete)")
            continue
        for order in ("o1", "o2"):
            parts, ok = [], True
            for name, value, bound, op, what in terms(pools[cell], None, order):
                good = value <= bound if op == "<=" else value >= bound
                ok &= good
                parts.append(f"{name} {what}={value:+.2f} rule {op}{bound:+.2f}")
            failed |= not ok
            print(f"DAY52 P2 ({clause}) cell={cell} order={order} " + " | ".join(parts) + f" -> {'PASS' if ok else 'FAIL'}")
    if pools["chain"] is not None:
        P = pools["chain"]
        named = None
        deltas = {}
        for order in ("o1", "o2"):
            D = med(P(order, "p", "publish")) - med(P(order, "base", "publish"))
            deltas[order] = (D, {g: med(P(order, "p", g)) - med(P(order, "base", g)) for g in GROUPS + ("insert_other",)})
        both = [g for g in GROUPS + ("insert_other",)
                if all(deltas[o][0] > 0 and deltas[o][1][g] >= deltas[o][0] / 2 for o in ("o1", "o2"))]
        for order in ("o1", "o2"):
            D, ds = deltas[order]
            print(f"DAY52 PLACING order={order} publish p-minus-base={D:+.2f} | "
                  + " ".join(f"{g}={v:+.2f}" for g, v in ds.items()))
        print(f"DAY52 PLACING -> {('placed in ' + ', '.join(both)) if both else 'not placed'}")
    else:
        print("DAY52 PLACING -> UNREAD (the chain cell is incomplete)")
    hump = os.path.join(root, "hump", "reading-hump.log")
    arms = {}
    if os.path.exists(hump):
        for m in re.finditer(r"HUMP arm=(x\w+) boots=(\d+) median-hump=([+-][\d.]+)", open(hump).read()):
            arms[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    if "xp2" not in arms or "xgpp" not in arms or arms["xp2"][0] != 2 or arms["xgpp"][0] != 2:
        print(f"DAY52 P2 (e) hump -> UNREAD (arms {arms})")
        unread = True
    elif arms["xgpp"][1] <= 0.15:
        print(f"DAY52 P2 (e) hump xp2={arms['xp2'][1]:+.3f} control xgpp={arms['xgpp'][1]:+.3f} -> UNREAD (the control "
              f"did not hump; the cell repeats once)")
        unread = True
    else:
        ok = arms["xp2"][1] <= 0.15
        failed |= not ok
        print(f"DAY52 P2 (e) hump xp2={arms['xp2'][1]:+.3f} rule <=0.15 control xgpp={arms['xgpp'][1]:+.3f} (humps) -> "
              f"{'PASS' if ok else 'FAIL'}")
    exits = {}
    for g in GATES:
        f = os.path.join(root, "gates", f"{g}.exit")
        exits[g] = open(f).read().strip() if os.path.exists(f) else "missing"
    unit = os.path.join(root, "unit", "run.log")
    unit_line = open(unit).read().strip().splitlines()[-1] if os.path.exists(unit) and open(unit).read().strip() else ""
    unit_ok = bool(UNIT_GREEN.search(unit_line))
    gates_ok = all(v == "0" for v in exits.values())
    if any(v == "missing" for v in exits.values()) or not unit_line:
        unread = True
    else:
        failed |= not (gates_ok and unit_ok)
    print(f"DAY52 P2 (a) gates {' '.join(f'{k}={v}' for k, v in exits.items())} | unit [{unit_line}] -> "
          f"{'PASS' if gates_ok and unit_ok else 'FAIL'}")
    print(f"DAY52 P2 -> {'FAIL' if failed else ('INCOMPLETE' if unread else 'PASS')}")


if __name__ == "__main__":
    main()
