#!/usr/bin/env python3
"""memra#1211: the c = 2 start mode per arm over a directory of census boots (normal.jsonl and
server-ARM-N.log, PLAN 1,2 repeated). Per phase: whether the first pair's two requests were
admitted in the same tick, and whether they ended in the same burst. Prints the per-phase rows
(c2mode.py's), the same-tick counts per arm with a two-sided Fisher exact p, the aggregate
tok/s per arm conditioned on the start mode, and the arms' mean aggregate.
usage: c2stats.py RUNDIR PLAN ARM_A ARM_B   (arms by name prefix, e.g. main7 head)"""
import collections
import json
import math
import pathlib
import re
import statistics
import sys

def read_log(path):
    """A server log, plain or gzip-compressed (the receipts keep them as .log.gz)."""
    import gzip
    import os
    path = str(path)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace").read()
    return open(path, errors="replace").read()


d, plan = pathlib.Path(sys.argv[1]), [int(c) for c in sys.argv[2].split(",")]
arm_a, arm_b = sys.argv[3], sys.argv[4]
rows = [json.loads(l) for l in open(d / "normal.jsonl") if l.strip()]
boots = collections.OrderedDict()
for r in rows:
    boots.setdefault(r["arm"], []).append(r)
phases = []  # (arm prefix, boot, phase, agg, same_tick, together, marks)
print("| boot | phase | agg tok/s | first pair admitted | ended in one burst | marks | first pair's tokens after each burst |")
print("|---|---:|---:|---|---|---:|---|")
for boot, rs in boots.items():
    L = read_log(d / f"server-{boot}.log").splitlines()
    adm = [i for i, l in enumerate(L) if "[spec-k] model=" in l]
    at = 0
    for ph, c in enumerate(plan):
        n = 3 * c
        if c == 2:
            seg = adm[at:at + n]
            text = "\n".join(l for l in L[seg[0]:seg[2]] if l.startswith("[tick-spec]"))
            ends = [int(b) for _, b in re.findall(r"generated=(\d+)->(\d+)", text)]
            same = len(ends) >= 4 and ends[2] < 50 and ends[3] < 50
            together = ends[-1] == ends[-2] == 160
            r = rs[ph]
            marks = r.get("stall_marks") or 0
            phases.append((boot.rsplit("-", 1)[0], boot, ph, r["agg_tok_s"], same, together, marks))
            print(f"| {boot} | {ph} | {r['agg_tok_s']} | {'same tick' if same else 'one tick apart'} | "
                  f"{'yes' if together else 'no'} | {marks} | {' '.join(map(str, ends))} |")
        at += n


def fisher(a, b, c, e):
    """Two-sided Fisher exact p for the table [[a, b], [c, e]]."""
    n1, n2, k = a + b, c + e, a + c

    def p(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)

    obs = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(n1, k) + 1) if p(x) <= obs * (1 + 1e-9))


print()
cnt = {}
for arm in (arm_a, arm_b):
    ps = [x for x in phases if x[0] == arm]
    cnt[arm] = (sum(x[4] for x in ps), len(ps))
    print(f"- {arm}: {cnt[arm][0]} of {cnt[arm][1]} c = 2 phases started in the same tick; "
          f"mean aggregate {statistics.mean(x[3] for x in ps):.2f} tok/s; marks {sum(x[6] for x in ps)}")
(a, na), (b, nb) = cnt[arm_a], cnt[arm_b]
print(f"- Fisher exact, two-sided, same-tick starts {a}/{na} against {b}/{nb}: p = {fisher(a, na - a, b, nb - b):.3f}")
print()
print("| start mode | " + " | ".join(f"{arm}: phases, agg tok/s mean (range)" for arm in (arm_a, arm_b)) + " |")
print("|---|---|---|")
for name, test in (("same tick, ended in one burst", lambda x: x[4] and x[5]),
                   ("same tick, ended apart", lambda x: x[4] and not x[5]),
                   ("one tick apart", lambda x: not x[4])):
    cells = []
    for arm in (arm_a, arm_b):
        v = [x[3] for x in phases if x[0] == arm and test(x)]
        cells.append(f"{len(v)}, {statistics.mean(v):.2f} ({min(v)}-{max(v)})" if v else "0")
    print(f"| {name} | {cells[0]} | {cells[1]} |")
