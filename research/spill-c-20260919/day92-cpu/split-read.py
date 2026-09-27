#!/usr/bin/python3
"""DAY92 section 3's sizing reader for queue v22's split (day89-cpu/split-read.py with v22's arms, registered before
this script): per arm and phase, each door-only leaf's median over the arm's runs and their sum; the summed change
p88s -> i24s, i24s -> i25s and p88s -> i25s (DAY89 section 2's rule, cumulative: short below 155 us per generated
token), and i25t minus i25s beside it (the trace's clocked cost, deciding nothing). Integrity is day83-read.py
--check --traced's, run by the queue beside this.

usage: split-read.py <ev-dir> [--rig <rig>]
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("day83_read", HERE / "day83-read.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

THRESHOLD_US = 155.0
ev = Path(sys.argv[1])
rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
ARMS = ["p88s", "i24s", "i25s", "i25t"]
PHASES = (("generate", "gate", "generate"), ("window", "warm", "window"))


def us(ns):
    return None if ns is None else ns / r.TOKENS / 1000


table = {}
for arm in ARMS:
    runs = [r.parse(p) for p in sorted(ev.glob(f"o[12]-{arm}-r*.log"))]
    for phase, a, b in PHASES:
        per = [r.terms(x, a, b) for x in runs]
        if per:
            table[(arm, phase)] = (len(per), {k: r.med([p[k] for p in per]) for k in per[0]})

verdict = None
for phase, _, _ in PHASES:
    sums = {}
    for arm in ARMS:
        if (arm, phase) not in table:
            continue
        n, m = table[(arm, phase)]
        vals = [m[k] for k in r.LEAVES]
        sums[arm] = None if any(v is None for v in vals) else us(sum(vals))
        shown = "-" if sums[arm] is None else f"{sums[arm]:.1f}"
        print(f"DAY92 SPLIT rig={rig} {arm} {phase} N={n} door-only leaves summed={shown} us per token")
    for a, b in (("p88s", "i24s"), ("i24s", "i25s"), ("p88s", "i25s"), ("i25s", "i25t")):
        if sums.get(a) is None or sums.get(b) is None:
            print(f"DAY92 CHANGE rig={rig} {phase} {b} minus {a}: unread")
            continue
        change = sums[b] - sums[a]
        print(f"DAY92 CHANGE rig={rig} {phase} {b} minus {a}: {change:+.1f} us per token")
        if phase == "generate" and (a, b) == ("p88s", "i25s"):
            verdict = "reaches" if -change >= THRESHOLD_US else "short"
            print(f"DAY92 SIZING rig={rig} generate door-only change p88s->i25s={change:+.1f} us per token, "
                  f"threshold -{THRESHOLD_US:.0f} -> {verdict}")
if verdict is None:
    print(f"DAY92 SIZING rig={rig} unread")
    sys.exit(1)
