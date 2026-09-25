#!/usr/bin/env python3
"""WP-A day 54 (DAY54.md section 1, OWED item 10) reader: the on-tick publishers' census and prices, written before
its cells run.

usage: day54-reading.py R
Input (the pro-single-day54 sitting's layout):
  R/{short,long}/ab/o{1,2}/bNN-<mode>/{server.log,<mode>/receipt.json} and R/<cell>/ab/replays.log (stall_cell.py
  --n 5; short: fanout against prime-short, long: fanout-long against prime; 20 boots per cell, five per mode per order);
  R/pause/bNN/{server.log,pause/receipt.json} (three boots); R/census/** (the gates' logs, every file read).
Complete, per paired cell: five boots per mode per order, each with a receipt, `errors` empty, every intruder without
an error, 20 `STALL REPLAY: PASS`. Nothing is read from an incomplete cell.
Terms, per cell, order and mode (pooled boots): the tenant stall (`stall_ms` of the mode's runs, max gap minus the
run's median gap), its median; the intruder's e2e (`wall_ms`) median; in the fanout modes the `[prefix-dedup] B=N`
lines (how many timed fanouts grouped all four) and the on-tick line's snapshot, restores and insert medians.
The rule (DAY54 section 1): per publisher, the tenant-visible share. Fanout (each length): the stall median of the
fanout mode minus the prime control's, per order; above 1.0 ms in both orders -> DESIGN NEXT; at or below 1.0 ms in
both -> CLOSED AS PRICED; otherwise NOT PLACED (the orders disagree). The pause park and every census row: the owner
time median (snapshot plus insert); above 1.0 ms -> DESIGN NEXT, else CLOSED AS PRICED. The publishers this card does
not exercise are printed NOT MEASURED HERE.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

DEDUP_B = re.compile(r"\[prefix-dedup\] B=(\d+) prefix=(\d+)")
DEDUP_ON = re.compile(r"\[prefix-dedup\] on-tick publish: snapshot ([\d.]+) ms \(([\d.]+) MB\), (\d+) sibling "
                      r"restore\(s\) ([\d.]+) ms, insert ([\d.]+) ms")
PARK = re.compile(r"\[prefix-host\] pause park snapshot on the tick: ([\d.]+) ms \(([\d.]+) MB\)")
ONTICK = re.compile(r"\[prefix-cache\] on-tick publish: publisher=(\S+) \(the (capture|spec) route answered on-tick: "
                    r"(.*?)\); snapshot ([\d.]+) ms, insert ([\d.]+) ms, ([\d.]+) MB")
CELLS = {"short": ("fanout", "prime-short"), "long": ("fanout-long", "prime")}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def boot(d, mode):
    out = {"stall": [], "e2e": [], "groups": [], "snap": [], "restore": [], "insert": [], "siblings": [], "ok": True,
           "why": []}
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
        if "stall_ms" in r:
            out["stall"].append(r["stall_ms"])
    for ln in open(os.path.join(d, "server.log"), errors="replace"):
        m = DEDUP_B.search(ln)
        if m:
            out["groups"].append(int(m.group(1)))
        m = DEDUP_ON.search(ln)
        if m:
            out["snap"].append(float(m.group(1)))
            out["siblings"].append(int(m.group(3)))
            out["restore"].append(float(m.group(4)))
            out["insert"].append(float(m.group(5)))
    return out


def main():
    root = sys.argv[1]
    verdicts = []
    for cell, (fan, prime) in CELLS.items():
        base = os.path.join(root, cell)
        complete, notes, pools = True, [], {}
        for order in ("o1", "o2"):
            for mode in (fan, prime):
                ds = sorted(glob.glob(os.path.join(base, "ab", order, f"b*-{mode}")))
                bs = [boot(d, mode) for d in ds]
                if len(bs) != 5:
                    complete = False
                    notes.append(f"{order} {mode} boots={len(bs)}")
                for d, b in zip(ds, bs):
                    if not b["ok"]:
                        complete = False
                        notes.append(f"{order}/{os.path.basename(d)} {';'.join(b['why'])}")
                pools[(order, mode)] = {k: [x for b in bs for x in b[k]] for k in
                                        ("stall", "e2e", "groups", "snap", "restore", "insert", "siblings")}
        rp = os.path.join(base, "ab", "replays.log")
        passes = open(rp, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(rp) else 0
        if passes != 20:
            complete = False
            notes.append(f"replays PASS={passes} of 20")
        for order in ("o1", "o2"):
            for mode in (fan, prime):
                p = pools[(order, mode)]
                g = collections.Counter(p["groups"])
                print(f"DAY54 READING cell={cell} order={order} mode={mode} stall N={len(p['stall'])} "
                      f"median={med(p['stall']):.2f} | e2e median={med(p['e2e']):.1f} | dedup groups {dict(g)} | "
                      f"on-tick snapshot={med(p['snap']):.2f} ms restores={med(p['restore']):.2f} ms over "
                      f"{med(p['siblings'])} insert={med(p['insert']):.2f} ms")
        if not complete:
            print(f"DAY54 CELL {cell} INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
            verdicts.append((f"fanout ({cell})", "UNREAD"))
            continue
        diffs = [med(pools[(o, fan)]["stall"]) - med(pools[(o, prime)]["stall"]) for o in ("o1", "o2")]
        if all(d > 1.0 for d in diffs):
            v = "DESIGN NEXT"
        elif all(d <= 1.0 for d in diffs):
            v = "CLOSED AS PRICED"
        else:
            v = "NOT PLACED (the orders disagree)"
        print(f"DAY54 PRICE fanout ({cell}) stall fanout-minus-prime o1={diffs[0]:+.2f} o2={diffs[1]:+.2f} ms "
              f"rule >1.0 both -> {v}")
        verdicts.append((f"fanout ({cell})", v))
    # The pause park.
    park, pstall = [], []
    for d in sorted(glob.glob(os.path.join(root, "pause", "b*"))):
        for ln in open(os.path.join(d, "server.log"), errors="replace"):
            m = PARK.search(ln)
            if m:
                park.append(float(m.group(1)))
        rec = os.path.join(d, "pause", "receipt.json")
        if os.path.exists(rec):
            pstall += [r["stall_ms"] for r in json.load(open(rec))["runs"] if r.get("arm") == "pause" and "stall_ms" in r]
    if park:
        v = "DESIGN NEXT" if med(park) > 1.0 else "CLOSED AS PRICED"
        print(f"DAY54 PRICE pause park snapshot owner N={len(park)} median={med(park):.2f} ms (pause stall median "
              f"{med(pstall):.2f} ms, N={len(pstall)}) rule >1.0 -> {v}")
        verdicts.append(("pause park", v))
    else:
        print("DAY54 PRICE pause park -> UNREAD (no park line)")
        verdicts.append(("pause park", "UNREAD"))
    # The census.
    rows = collections.defaultdict(list)
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(f)]
    for f in files:
        try:
            text = open(f, errors="replace").read()
        except OSError:
            continue
        for m in ONTICK.finditer(text):
            rows[(m.group(1), m.group(2), m.group(3))].append(float(m.group(4)) + float(m.group(5)))
    if not rows:
        print("DAY54 CENSUS no on-tick publish line from either capture route on this card")
    for (pub, route, reason), xs in sorted(rows.items()):
        v = "DESIGN NEXT" if med(xs) > 1.0 else "CLOSED AS PRICED"
        print(f"DAY54 CENSUS publisher={pub} route={route} reason={reason!r} N={len(xs)} owner median={med(xs):.2f} ms "
              f"-> {v}")
        verdicts.append((f"{pub} on-tick ({reason})", v))
    for name in ("dspark-boundary (a DFlash drafter tail)", "glm5-boundary (latent tails, tensor parallel)",
                 "latent planes (MLA models)"):
        print(f"DAY54 PRICE {name} -> NOT MEASURED HERE (not exercised by the 27B on one card)")
    print("DAY54 VERDICTS " + "; ".join(f"{k}: {v}" for k, v in verdicts))


if __name__ == "__main__":
    main()
