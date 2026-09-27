#!/usr/bin/python3
"""DAY89 section 2's sizing reader for queue v21's split (registered before this script): per arm and phase, each
door-only leaf's median over the arm's runs (day83-read.py's terms and LEAVES, the arithmetic behind DAY85's 346 us),
their sum, and the summed change p88s -> i23s, i23s -> i24s and p88s -> i24s; beside it the dispatch-clock-only arms'
`pf_retire` (no stage clock settle). The rule: the cuts are short if the door-only leaves fall by less than 155 us per
generated token from p88s to i24s (100 us on the 285K class, scaled by 346 against 223). Integrity is day83-read.py
--check's, run by the queue beside this.

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
ARMS = ["p88s", "i23s", "i24s", "p88d", "i23d", "i24d"]
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
    for arm in ("p88s", "i23s", "i24s"):
        if (arm, phase) not in table:
            continue
        n, m = table[(arm, phase)]
        vals = [m[k] for k in r.LEAVES]
        sums[arm] = None if any(v is None for v in vals) else us(sum(vals))
        shown = "-" if sums[arm] is None else f"{sums[arm]:.1f}"
        print(f"DAY89 SPLIT rig={rig} {arm} {phase} N={n} door-only leaves summed={shown} us per token")
    for a, b in (("p88s", "i23s"), ("i23s", "i24s"), ("p88s", "i24s")):
        if sums.get(a) is None or sums.get(b) is None:
            print(f"DAY89 CHANGE rig={rig} {phase} {b} minus {a}: unread")
            continue
        change = sums[b] - sums[a]
        print(f"DAY89 CHANGE rig={rig} {phase} {b} minus {a}: {change:+.1f} us per token")
        if phase == "generate" and (a, b) == ("p88s", "i24s"):
            verdict = "reaches" if -change >= THRESHOLD_US else "short"
            print(f"DAY89 SIZING rig={rig} generate door-only change p88s->i24s={change:+.1f} us per token, "
                  f"threshold -{THRESHOLD_US:.0f} -> {verdict}")
    retire = {arm: us(table[(arm, phase)][1]["pf_retire_ns"]) for arm in ("p88d", "i23d", "i24d")
              if (arm, phase) in table}
    if retire:
        print(f"DAY89 BESIDE rig={rig} {phase} pf_retire without the stage clock (us per token, medians): "
              + " ".join(f"{arm}={'-' if v is None else f'{v:.1f}'}" for arm, v in retire.items()))
if verdict is None:
    print(f"DAY89 SIZING rig={rig} unread")
    sys.exit(1)
