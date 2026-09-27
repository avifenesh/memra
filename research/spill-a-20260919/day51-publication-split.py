#!/usr/bin/env python3
"""WP-A day 51 section 3, a reading written AFTER the verdict (no clause): where the chain cell's +1.4 ms sits.

usage: day51-publication-split.py ROOT   (ROOT = the pro-single-p receipts: <cell>/ab/o{1,2}/bNN-{base,p}/server.log)
Per cell, order and arm, medians over the pooled boots of each boot's third and later lines: the demote's owner hold
(`the owner thread held X ms across the demote: pre-submit A, copy settle B .., hashing polls C, take-back bind and
publish D`), the pre-submit split's leases, the demote's wall and helper time and polls, and the promote's PIN, its
submission-to-completion time and polls.
"""
import glob
import re
import statistics
import sys

PAT = {
    "held": r"the owner thread held ([\d.]+)ms across the demote",
    "pre": r"across the demote: pre-submit ([\d.]+)",
    "settle": r"copy settle ([\d.]+) over",
    "hpolls": r"hashing polls ([\d.]+)",
    "bind_publish": r"take-back bind and publish ([\d.]+)",
    "leases": r"demote pre-submit split: ticket seq=\d+ leases ([\d.]+) ms",
    "wall": r"demote digests landed off the tick: .*wall ([\d.]+)ms t0 to publication",
    "helper": r"demote digests landed off the tick: .*hashed in ([\d.]+)ms",
    "dpolls": r"demote digests landed off the tick: .*landed after (\d+) poll",
    "pin": r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms",
    "p_submit_to_done": r"promote published off the tick: ticket complete after \d+ poll\(s\), ([\d.]+)ms from submission",
    "ppolls": r"promote published off the tick: ticket complete after (\d+) poll",
}
PAT = {k: re.compile(v) for k, v in PAT.items()}
root = sys.argv[1]
for cell in ("demote", "free", "promote", "chain"):
    for order in ("o1", "o2"):
        for arm in ("base", "p"):
            vals = {k: [] for k in PAT}
            for d in sorted(glob.glob(f"{root}/{cell}/ab/{order}/b*-{arm}")):
                lines = open(f"{d}/server.log", errors="replace").read().splitlines()
                for k, p in PAT.items():
                    vals[k] += [float(m.group(1)) for m in map(p.search, lines) if m][2:]
            terms = " ".join(f"{k}={statistics.median(v):.2f}(N={len(v)})" for k, v in vals.items() if v)
            print(f"DAY51 SPLIT cell={cell} order={order} arm={arm} {terms}")
