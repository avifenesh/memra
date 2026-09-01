#!/usr/bin/env python3
"""Compare the three gate arms' byte tapes and state each candidate's verdict at ITS OWN bar.

  spec   vs plain  -> SPEC claims numerical exactness (the verify arbitrates), so the bar is
                      FULL byte identity on real prompts, first token AND 128-token tape.
  plainv vs plain  -> MEMRA_W8_VIEW is a weight-precision door in the MEMRA_STEP_TP_W8 class,
                      so the bar is the argmax gate: the first token at max_tokens=1, one
                      forward with no cascade possible. Tape divergence there is ordinary
                      tie-break cascade and is NOT a gate result; it is reported separately.

A cell whose completion was empty is INVALID and never a pass: this is a THINKING model, and
sha256("") compares equal to itself.
"""
import json
import os

arms = {}
for a in ("plain", "spec", "plainv"):
    p = "/root/s37h-specgate-%s.json" % a
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        continue
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            arms[a] = json.loads(line)["cells"]

if "plain" not in arms:
    print("GATE INVALID: no plain arm tape, nothing to compare against")
    raise SystemExit(1)


def compare(other, label, bar):
    if other not in arms:
        print("  %-8s MISSING ARM - GATE INVALID (not a pass)" % other)
        return
    firsts, tapes, bad = [], [], 0
    for k in sorted(arms["plain"]):
        a = arms["plain"][k]
        b = arms[other].get(k, ["MISSING", 0])
        if "EMPTY-INVALID" in (a[0], b[0]) or b[0] == "MISSING":
            bad += 1
            v = "INVALID"
        else:
            v = "IDENTICAL" if a[0] == b[0] else "DIFFER"
        (firsts if k.endswith("/first") else tapes).append(v)
        print("    %-22s plain=%s(%d) %s=%s(%d)  %s" % (k, a[0], a[1], other, b[0], b[1], v))
    ok_first = firsts and all(v == "IDENTICAL" for v in firsts)
    ok_tape = tapes and all(v == "IDENTICAL" for v in tapes)
    print("  %s: %s" % (label, bar(ok_first, ok_tape, bad)))


def spec_bar(ok_first, ok_tape, bad):
    if bad:
        return "INVALID (%d empty/missing cells) - never a pass" % bad
    if ok_first and ok_tape:
        return "PASS - byte-identical to spec-off on every prompt, first token AND full tape"
    if ok_first:
        return ("FAIL AT ITS OWN BAR - first tokens match but the tape DIVERGES, and spec is "
                "claimed numerically exact, so a tape difference is a real finding here")
    return "FAIL - first token differs from spec-off"


def w8v_bar(ok_first, ok_tape, bad):
    if bad:
        return "INVALID (%d empty/missing cells) - never a pass" % bad
    if ok_first:
        return ("ARGMAX GATE PASS - first token identical on all four real prompts"
                + ("; tape identical too" if ok_tape
                   else "; tape diverges, which is ordinary tie-break cascade for a numeric-class door"))
    return "ARGMAX GATE FAIL - the first token FLIPPED. Door REJECTED; do not flip it to hit a number."


print()
print("=== GATE: spec vs plain (bar = FULL byte identity) ===")
compare("spec", "SPEC VERDICT", spec_bar)
print()
print("=== GATE: MEMRA_W8_VIEW vs plain (bar = argmax, first token at max_tokens=1) ===")
compare("plainv", "W8_VIEW VERDICT", w8v_bar)
