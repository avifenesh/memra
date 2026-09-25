#!/usr/bin/env python3
"""WP-A day 52 section 3, a reading written AFTER the verdict (no clause): the chain cell per boot.

usage: day52-chain-per-boot.py ROOT   (ROOT = the pro-single-p2 receipts: chain/ab/o{1,2}/bNN-{base,p,p2})
Per boot: the replaced twin's KV lease frees (`kv X ms over K leases` of the publication split, K > 0; the median of
the third and later), the chained request's and the first intruder's e2e medians, the arming lines, and the boot's
start temperature and SM clock.
"""
import glob
import json
import re
import statistics
import sys

SPLIT = re.compile(r"demote publication split: .*kv ([\d.]+) ms over (\d+) leases")
ARM = re.compile(r"payload reserve (armed|disarmed): the job took (\d+) fresh pages of (\d+)")
root = sys.argv[1]
for order in ("o1", "o2"):
    for d in sorted(glob.glob(f"{root}/chain/ab/{order}/b*")):
        lines = open(f"{d}/server.log", errors="replace").read().splitlines()
        kv = [float(m.group(1)) for m in map(SPLIT.search, lines) if m and m.group(2) != "0"]
        arms = [f"{m.group(1)}@{m.group(2)}/{m.group(3)}" for m in map(ARM.search, lines) if m]
        runs = [r for r in json.load(open(f"{d}/promote-long/receipt.json"))["runs"] if r.get("arm") == "promote-long"]
        chain = statistics.median(r["intruder"]["chain_wall_ms"] for r in runs)
        first = statistics.median(r["intruder"]["wall_ms"] for r in runs)
        start = open(f"{d}/BOOT.txt").read().splitlines()[1].split(": ", 1)[1]
        print(f"DAY52 CHAIN BOOT {order} {d.rsplit('/', 1)[1]} kv_med={statistics.median(kv[2:]):.2f} (N={len(kv)}) "
              f"chain={chain:.1f} first={first:.1f} arming={arms} start={start}")
