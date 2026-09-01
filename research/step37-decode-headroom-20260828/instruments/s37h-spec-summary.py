#!/usr/bin/env python3
"""Summarise the step37 spec sweep: median/min/max/spread per arm, deltas, engagement audit.

A cell with no valid row is INVALID and never a pass. An arm whose spec engagement is not
proved in the direction its name claims is reported as such rather than quoted: a spec arm with
zero rounds is not a spec measurement, and a plain arm with rounds is not a control.
"""
import re
import statistics
import sys

ARMS = ("plain", "spec", "specv", "plainv")
rows = {a: [] for a in ARMS}
eng = {a: [] for a in ARMS}
invalid = []
for line in open(sys.argv[1]):
    m = re.search(r"arm=(\S+)\s+tok/s_med=([\d.]+)", line)
    if m and m.group(1) in rows:
        rows[m.group(1)].append(float(m.group(2)))
    mi = re.search(r"arm=(\S+).*(NO VALID ROWS|booted=NO)", line)
    if mi:
        invalid.append(line.strip())
    me = re.search(r"arm=(\S+).*spec_usage=(\S+)", line)
    if me and me.group(1) in eng:
        eng[me.group(1)].append(me.group(2))

print()
print("=== VENDOR-DEFAULT SAMPLED, curve-0400, interleaved x5 ===")
print("arm     n  median   min    max    spread   vs plain")
base = statistics.median(rows["plain"]) if rows["plain"] else None
for a in ARMS:
    v = sorted(rows[a])
    if not v:
        print("%-7s NO VALID ROWS - INVALID" % a)
        continue
    med = statistics.median(v)
    d = "" if base is None or a == "plain" else "  %+.2f tok/s (%+.1f%%)" % (med - base, 100.0 * (med - base) / base)
    print("%-7s %d  %-7.2f %-6.2f %-6.2f %-7.2f%s" % (a, len(v), med, v[0], v[-1], v[-1] - v[0], d))

print()
print("=== ENGAGEMENT AUDIT (usage.spec, response body, both directions) ===")
for a in ARMS:
    want_spec = a.startswith("spec")
    seen = eng[a]
    engaged = [s for s in seen if "rounds=" in s and "rounds=0" not in s and "rounds=None" not in s]
    absent = [s for s in seen if "ABSENT" in s or "rounds=0" in s or "rounds=None" in s]
    if want_spec:
        ok = len(engaged) == len(seen) and seen
        verdict = "ENGAGED in %d/%d cells" % (len(engaged), len(seen)) if ok else \
                  "NOT PROVEN (%d/%d cells engaged) - this arm is NOT a spec measurement" % (len(engaged), len(seen))
    else:
        ok = len(absent) == len(seen) and seen
        verdict = "spec absent in %d/%d cells (correct control)" % (len(absent), len(seen)) if ok else \
                  "CONTROL IS DIRTY (%d/%d cells show rounds) - this arm is not a spec-off control" % (len(engaged), len(seen))
    print("  %-7s %s" % (a, verdict))
    if seen:
        print("          sample: %s" % seen[0])

if invalid:
    print()
    print("=== INVALID CELLS (never counted as a pass) ===")
    for l in invalid:
        print("  " + l)

print()
if base:
    for a in ARMS:
        v = rows[a]
        if v:
            med = statistics.median(v)
            print("  %-7s median %.2f tok/s = %.2f ms/token   %s 90 tok/s (11.11 ms)"
                  % (a, med, 1000.0 / med, "CLEARS" if med >= 90 else "short of"))
