#!/usr/bin/env python3
"""WP-A day 43 (DAY43.md section 1, OWED item 15) reader: G''' against G4 at 4096-token entries, written before its
cells run.

usage: item15-reading.py ROOT
Input: ROOT/demote-long/ab/o{1,2}/bNN-{g4,g3}/{server.log,demote-long/receipt.json} and
ROOT/promote-long/ab/o{1,2}/bNN-{g4,g3}/{server.log,promote-long/receipt.json} (stall_cell.py, --n 5: 5 timed runs of
the arm per boot), each ab/replays.log; 20 boots per mode, five per arm per order. Complete: every boot with a receipt,
`errors` empty, every intruder (and every chain) without an error, one `STALL REPLAY: PASS` per boot, and in
demote-long at least as many `demote copy complete off the tick` lines as timed runs plus the seed's (every timed run
demoted); nothing is read from an incomplete cell (it repeats whole once).
Terms, per order, medians over the pooled boots of an arm (steady = the second and later such line of a boot):
  copy   demote-long: the steady `demote copy complete off the tick: .. Xms from submission to completion`
  wall   demote-long: the steady `demote digests landed off the tick: .. wall Xms t0 to publication`
  chain  promote-long: the chained request's e2e (`chain_wall_ms`), the path that waits on the demote
  first  promote-long: the first intruder's e2e (`wall_ms`, a host hit that promotes)
  tenant both modes: the tenant's e2e in the arm runs (`tenant_wall_ms`)
  itl    both modes: the tenant's pre-fire inter-token latency median per arm run (its first FIRE_AT - 1 gaps)
The rule (G''' is adopted for the RTX PRO 6000 class only if every term holds in both orders):
  (1) chain: g4 - g3 >= 2.0 ms          (2) copy: g4 - g3 >= 2.0 ms
  (3) no regression: tenant (each mode) g3 - g4 <= +1.0 ms; first g3 - g4 <= +1.0 ms; wall g3 - g4 <= +5.0 ms;
      itl (each mode) g3 - g4 <= +0.05 ms
  (4) the long-entry hump (ROOT/hump, day38-hump-reading.py's line for arm xg3): median HUMP <= 0.15 ms
Otherwise G4 stays the single placement.
"""
import glob
import json
import os
import re
import statistics
import sys

COPY = re.compile(r"demote copy complete off the tick: ticket seq=\d+ complete after \d+ poll\(s\), ([\d.]+)ms from "
                  r"submission to completion")
WALL = re.compile(r"demote digests landed off the tick: .*; wall ([\d.]+)ms t0 to publication")
HUMP = re.compile(r"HUMP arm=(\S+) boots=(\d+) median-hump=([+-][\d.]+) humps=(\S+)")
ARMS = ("g4", "g3")
FIRE_AT = 24


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def boot(d, mode):
    out = {"copy": [], "wall": [], "chain": [], "first": [], "tenant": [], "itl": [], "ok": True, "why": [],
           "copies": 0, "runs": 0}
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
        out["runs"] += 1
        i = r.get("intruder") or {}
        if "error" in i or "wall_ms" not in i or (mode == "promote-long" and "chain_wall_ms" not in i):
            out["ok"] = False
            out["why"].append("intruder error")
            continue
        out["first"].append(i["wall_ms"])
        if mode == "promote-long":
            out["chain"].append(i["chain_wall_ms"])
        if r.get("tenant_wall_ms") is not None:
            out["tenant"].append(r["tenant_wall_ms"])
        pre = r.get("itl_ms", [])[: FIRE_AT - 1]
        if pre:
            out["itl"].append(med(pre))
    log = os.path.join(d, "server.log")
    c = w = 0
    for ln in open(log, errors="replace") if os.path.exists(log) else []:
        m = COPY.search(ln)
        if m:
            c += 1
            if c >= 2:
                out["copy"].append(float(m.group(1)))
        m = WALL.search(ln)
        if m:
            w += 1
            if w >= 2:
                out["wall"].append(float(m.group(1)))
    out["copies"] = c
    if mode == "demote-long" and c < out["runs"] + 1:
        out["ok"] = False
        out["why"].append(f"copy-complete lines={c} for {out['runs']} timed runs plus the seed")
    return out


def cell(root, mode):
    cells, complete, notes = {}, True, []
    for order in ("o1", "o2"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(root, mode, "ab", order, f"b*-{arm}")))
            bs = [boot(d, mode) for d in ds]
            if len(bs) != 5:
                complete = False
                notes.append(f"{mode} {order} {arm} boots={len(bs)}")
            for d, b in zip(ds, bs):
                if not b["ok"]:
                    complete = False
                    notes.append(f"{mode} {order}/{os.path.basename(d)} {';'.join(b['why'])}")
            cells[(order, arm)] = bs
    replays = os.path.join(root, mode, "ab", "replays.log")
    passes = open(replays, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(replays) else 0
    if passes != 20:
        complete = False
        notes.append(f"{mode} replays PASS={passes} of 20")
    return cells, complete, notes


def main():
    root = sys.argv[1]
    pools, complete, notes = {}, True, []
    for mode in ("demote-long", "promote-long"):
        cells, ok, n = cell(root, mode)
        complete &= ok
        notes += n
        for (order, arm), bs in cells.items():
            for key in ("copy", "wall", "chain", "first", "tenant", "itl"):
                pools[(mode, order, arm, key)] = [x for b in bs for x in b[key]]
    for mode in ("demote-long", "promote-long"):
        for order in ("o1", "o2"):
            for arm in ARMS:
                parts = []
                for key in ("copy", "wall", "chain", "first", "tenant", "itl"):
                    xs = pools[(mode, order, arm, key)]
                    if xs:
                        parts.append(f"{key} N={len(xs)} median={med(xs):.2f}")
                print(f"ITEM15 READING mode={mode} order={order} arm={arm} " + " | ".join(parts))
    hump = {}
    hl = os.path.join(root, "hump", "reading-hump.log")
    for ln in open(hl, errors="replace") if os.path.exists(hl) else []:
        m = HUMP.search(ln)
        if m:
            hump[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    print(f"ITEM15 READING hump {hump}")
    if "xg3" not in hump or hump["xg3"][0] != 2:
        complete = False
        notes.append("hump: no two-boot xg3 line")
    if not complete:
        print(f"ITEM15 INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    ok_all = True
    for order in ("o1", "o2"):
        d = lambda mode, key: med(pools[(mode, order, "g4", key)]) - med(pools[(mode, order, "g3", key)])
        terms = [
            ("(1) chain g4-g3", d("promote-long", "chain"), lambda v: v >= 2.0, ">=+2.0"),
            ("(2) copy g4-g3", d("demote-long", "copy"), lambda v: v >= 2.0, ">=+2.0"),
            ("(3) demote-long tenant g3-g4", -d("demote-long", "tenant"), lambda v: v <= 1.0, "<=+1.0"),
            ("(3) promote-long tenant g3-g4", -d("promote-long", "tenant"), lambda v: v <= 1.0, "<=+1.0"),
            ("(3) first g3-g4", -d("promote-long", "first"), lambda v: v <= 1.0, "<=+1.0"),
            ("(3) wall g3-g4", -d("demote-long", "wall"), lambda v: v <= 5.0, "<=+5.0"),
            ("(3) demote-long itl g3-g4", -d("demote-long", "itl"), lambda v: v <= 0.05, "<=+0.05"),
            ("(3) promote-long itl g3-g4", -d("promote-long", "itl"), lambda v: v <= 0.05, "<=+0.05"),
        ]
        ok = all(f(v) for _, v, f, _ in terms)
        ok_all &= ok
        print(f"ITEM15 order={order} " + " | ".join(f"{n}={v:+.2f} rule {r}" for n, v, _, r in terms)
              + f" -> {'HOLDS' if ok else 'FAILS'}")
    h = hump["xg3"][1]
    ok4 = h <= 0.15
    print(f"ITEM15 (4) hump xg3 median={h:+.3f} rule <=0.15 -> {'HOLDS' if ok4 else 'FAILS'}")
    ok_all &= ok4
    print("ITEM15 -> " + ("G''' ADOPTED for the RTX PRO 6000 class (the per-card placement's design is pre-registered next)"
                          if ok_all else "G4 STAYS the single placement"))


if __name__ == "__main__":
    main()
