#!/usr/bin/env python3
"""Greedy-law gate: looped output never enters a perf row. Tail-cycle and repeated-line
detectors over every rollout continuation; any flagged rollout gets excluded and the
exclusion stated in the receipt."""
import json

rows = json.load(open("/root/dfp2/rollouts.json"))
out = []
for r in rows:
    t = r["text"]
    flag = ""
    tail = t[-360:]
    for u in range(12, 61):
        unit = tail[-u:]
        if len(tail) >= 3 * u and tail[-3 * u:] == unit * 3:
            flag = "LOOP unit=%r" % unit
            break
    lines = [l for l in t.split("\n") if l.strip()]
    maxrun, run = 1, 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        maxrun = max(maxrun, run)
    verdict = flag or "clean"
    out.append({"name": r["name"], "verdict": verdict, "max_line_run": maxrun,
                "finish": r["finish_reason"]})
    print(r["name"], verdict, "maxlinerun=%d" % maxrun, r["finish_reason"])
json.dump(out, open("/root/dfp2/loop_check.json", "w"), indent=1)
