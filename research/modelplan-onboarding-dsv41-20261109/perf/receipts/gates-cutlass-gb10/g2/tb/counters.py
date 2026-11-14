#!/usr/bin/env python3
"""The gather counters of `dsv41-tpbench` row-block calls, by rows, for each build that ran.

usage: counters.py TB_DIR LAYERS TAG [TAG...]   (TB_DIR holds <tag>-<devices>/cells.jsonl or .gz; markdown on stdout)

A call's row carries `comm`: its collectives, its copies and its bytes (a region counted once a peer). They depend on
the program a build runs, not on the card or on how many cards the ranks sit on. Round 0 is left out; a build's
counters at a width must be one value over its calls. A build with no cells file for a device list is listed as not
run (`runs.txt` has its exit status, its log the reason).
"""
import glob
import gzip
import json
import os
import sys


def lines(path):
    if os.path.exists(path):
        return open(path)
    return gzip.open(path + ".gz", "rt")


def counters(d):
    out = {}
    for line in lines(os.path.join(d, "cells.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] != "prefill" or r["round"] == 0:
            continue
        c = (r["comm"]["collectives"], r["comm"]["copies"], r["comm"]["bytes"])
        out.setdefault(r["prompt_len"], set()).add(c)
    return out


def has(d):
    return os.path.exists(os.path.join(d, "cells.jsonl")) or os.path.exists(os.path.join(d, "cells.jsonl.gz"))


def main():
    root, layers, tags = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3:]
    devs = sorted({os.path.basename(p)[len(t) + 1:].replace(".log.gz", "").replace(".log", "")
                   for t in tags for p in glob.glob(os.path.join(root, f"{t}-0*"))}, key=lambda s: -len(s))
    for dev in devs:
        ranks = dev.count(",") + 1
        ran = [t for t in tags if has(os.path.join(root, f"{t}-{dev}"))]
        print(f"## {ranks} ranks on the card (devices {dev}), {layers} layers\n")
        if not ran:
            print(f"No build ran ({', '.join(tags)}: rc 1 in `runs.txt`; the reason is the last line of each run's log).\n")
            continue
        got = {t: counters(os.path.join(root, f"{t}-{dev}")) for t in ran}
        print("| rows | " + " | ".join(f"{t}: collectives / copies / bytes a call | {t}: bytes a row and layer" for t in ran) + " |")
        print("|---|" + "---|---|" * len(ran))
        for m in sorted(got[ran[0]]):
            cells = []
            for t in ran:
                v = sorted(got[t].get(m, ()))
                cells.append(" ; ".join(f"{a} / {b} / {c:,}" for a, b, c in v))
                cells.append(" ; ".join(f"{c / m / layers:,.0f}" for _, _, c in v))
            print(f"| {m} | " + " | ".join(cells) + " |")
        missing = [t for t in tags if t not in ran]
        if missing:
            print(f"\nNot run: {', '.join(missing)} (rc 1 in `runs.txt`; the reason is the last line of each run's log).")
        print()


main()
