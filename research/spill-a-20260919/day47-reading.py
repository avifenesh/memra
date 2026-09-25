#!/usr/bin/env python3
"""WP-A day 47 reader (DAY47.md section 1: design V's clauses (c) and (d)), written before its cells run.

usage: day47-reading.py ROOT
Input: ROOT/ab/o{1,2}/bNN-{base,v}/server.log and .../pause/receipt.json (stall_cell.py --mode pause, --n 5), and
ROOT/ab/replays.log; 20 boots, five per arm per order. Complete: every boot with a receipt, `errors` empty, every intruder
answered with `finish_reason` `tool_calls`, one `STALL REPLAY: PASS` per boot, and at least five counted runs per boot.
A run is counted when its pause window holds a `pause demote` line in the run's server log tail (the pause fired while
the tenant streamed). Per counted run: the pause window starts 200 ms after the intruder's response returned
(`fired_at_ms + wall_ms + 200`, the tenant's clock); the pause-window stall is the largest tenant inter-token gap that
ends inside the window, minus the run's ITL p50.
(c) per order: median(V's pause-window stalls) <= median(base's) / 4, and every run's tenant text identical across both
    arms (`tenant_text_sha`).
(d) per boot: the count of `pause demote: .. released` lines (either form) equal between every base and V boot of an
    order (the same candidates demote), and every V boot's `demote digests landed` count at least its release count.
"""
import glob
import json
import os
import re
import statistics
import sys

RELEASED = re.compile(r"\[prefix-host\] pause demote: (plain park released|device prefix entry released)")
LANDED = "demote digests landed off the tick"
ARMS = ("base", "v")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def window_stall(run):
    i = run.get("intruder") or {}
    if "wall_ms" not in i or "fired_at_ms" not in i or run.get("ttft_ms") is None:
        return None
    start = i["fired_at_ms"] + i["wall_ms"] + 200.0
    t = run["ttft_ms"]
    gaps = []
    for g in run.get("itl_ms", []):
        t += g
        if t > start:
            gaps.append(g)
    if not gaps or "p50" not in run:
        return None
    return max(gaps) - run["p50"]


def boot(d):
    out = {"stalls": [], "shas": set(), "ok": True, "why": [], "released": 0, "landed": 0}
    rec = os.path.join(d, "pause", "receipt.json")
    if not os.path.exists(rec):
        out["ok"] = False
        out["why"].append("no receipt")
        return out
    j = json.load(open(rec))
    if j.get("summary", {}).get("errors"):
        out["ok"] = False
        out["why"].append(f"errors={len(j['summary']['errors'])}")
    for r in j["runs"]:
        out["shas"].add(r.get("tenant_text_sha"))
        if r.get("arm") != "pause":
            continue
        i = r.get("intruder") or {}
        if i.get("finish_reason") != "tool_calls":
            out["ok"] = False
            out["why"].append(f"intruder finish={i.get('finish_reason')}")
            continue
        if not any("pause demote" in ln for ln in r.get("server_log_lines", [])):
            continue
        s = window_stall(r)
        if s is not None:
            out["stalls"].append(s)
    if len(out["stalls"]) < 5:
        out["ok"] = False
        out["why"].append(f"counted runs={len(out['stalls'])}")
    log = os.path.join(d, "server.log")
    for ln in open(log, errors="replace") if os.path.exists(log) else []:
        if RELEASED.search(ln):
            out["released"] += 1
        if LANDED in ln:
            out["landed"] += 1
    return out


def main():
    root = sys.argv[1]
    cells, complete, notes = {}, True, []
    for order in ("o1", "o2"):
        for arm in ARMS:
            ds = sorted(glob.glob(os.path.join(root, "ab", order, f"b*-{arm}")))
            bs = [boot(d) for d in ds]
            if len(bs) != 5:
                complete = False
                notes.append(f"{order} {arm} boots={len(bs)}")
            for d, b in zip(ds, bs):
                if not b["ok"]:
                    complete = False
                    notes.append(f"{order}/{os.path.basename(d)} {';'.join(b['why'])}")
            cells[(order, arm)] = bs
    replays = os.path.join(root, "ab", "replays.log")
    passes = open(replays, errors="replace").read().count("STALL REPLAY: PASS") if os.path.exists(replays) else 0
    if passes != 20:
        complete = False
        notes.append(f"replays PASS={passes} of 20")
    for (order, arm), bs in sorted(cells.items()):
        xs = [x for b in bs for x in b["stalls"]]
        print(f"DAY47 READING order={order} arm={arm} pause-window-stall N={len(xs)} median={med(xs):.2f} "
              f"released={[b['released'] for b in bs]} landed={[b['landed'] for b in bs]}")
    if not complete:
        print(f"DAY47 V INCOMPLETE ({'; '.join(notes)}) -> nothing is read; the cell repeats whole once")
        sys.exit(2)
    shas = set().union(*(b["shas"] for bs in cells.values() for b in bs))
    ok_all = True
    for order in ("o1", "o2"):
        base = med([x for b in cells[(order, "base")] for x in b["stalls"]])
        v = med([x for b in cells[(order, "v")] for x in b["stalls"]])
        c = v <= base / 4.0
        rel = {b["released"] for arm in ARMS for b in cells[(order, arm)]}
        d = len(rel) == 1 and all(b["landed"] >= b["released"] for b in cells[(order, "v")])
        ok = c and d and len(shas) == 1
        ok_all &= ok
        print(f"DAY47 V C order={order} stall base={base:.2f} v={v:.2f} rule v<=base/4={base / 4.0:.2f} "
              f"tenant_texts={len(shas)} -> {'PASS' if c and len(shas) == 1 else 'FAIL'} | D released={sorted(rel)} "
              f"-> {'PASS' if d else 'FAIL'}")
    print(f"DAY47 V -> {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
