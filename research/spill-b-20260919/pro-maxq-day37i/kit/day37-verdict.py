#!/usr/bin/env python3
"""DAY37 1.7 rule over the clause lines of day37-read.py (the corrected reader, tree 3476ea4dc).

The Workstation sittings printed one line per clause and the rule was read from them (DAY37.md 2.6, :704-708). This
applies the same rule to the same lines and prints one VERDICT line in the reader's own form:

    DAY37 VERDICT card=<card> rule=1.7 A1=.. A2=.. A3-i=.. A3-ii=.. A5=.. A6=.. A7=.. A4=..(reading) -> <reading>

<reading> is PROMOTE-ELIGIBLE (A1, A2, A5, A6, A7 PASS and A3 (i) and (ii) PASS), DELETE (every correctness clause
PASS, A3 fails), FAIL (no reading) (a correctness clause fails), or INCOMPLETE (owed: ...) when a clause has no line
yet (the gap run before its timed slot). A5-MAPPER's `N/A (inline: no mapper)` counts as met (addendum F item 2).
A4 is a benefit reading outside the rule (1.7); it is printed, never ruled on.

usage:
  day37-verdict.py <card> <read.log>                         one receipt root (the Max-Q cell)
  day37-verdict.py <card> --take <read.log>:<clause,..> ...   a reading assembled from several roots, as DAY37 2.6
                                                              assembled the Workstation class from two sittings
"""
import re, sys

GATE_CELLS = ["admit-mem-burst", "fault-default", "fault-plain", "hit-off", "hit-on", "identity-default-on",
              "serve-smoke", "spec-ctx-edge", "twin27-off", "twin27-on"]
MIX = [("spec", "O1"), ("spec", "O2"), ("plain", "O1"), ("plain", "O2")]
SHAPES = ["g2", "l64", "boff"]


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = re.match(r"DAY37 (\S+) card=(\S+) (.*)$", line.rstrip("\n"))
        if not m or m.group(1) in ("VERDICT", "A4-IDLE"):
            continue
        clause, card, rest = m.groups()
        if clause == "A2":
            res = "PASS" if " GROW-G1 PASS " in f" {rest} " else "FAIL"
        else:
            r = re.search(r"-> (PASS|FAIL|N/A[^\n]*)$", rest)
            res = r.group(1) if r else None
        key = clause
        if clause == "A1-GATE":
            key += ":" + re.search(r"cell=(\S+)", rest).group(1)
        elif clause == "A1-MIX":
            key += ":" + re.search(r"kind=(\S+)", rest).group(1) + ":" + re.search(r"order=(\S+)", rest).group(1)
        elif clause == "A3-ii":
            key += ":" + re.search(r"order=(\S+)", rest).group(1)
        elif clause == "A7":
            key += ":" + re.search(r"shape=(\S+)", rest).group(1)
        rows.append(dict(key=key, clause=clause, card=card, res=res, src=path, line=line.rstrip("\n")))
    return rows


def main(argv):
    card = argv[1]
    rows = []
    if argv[2] == "--take":
        args = argv[3:]
        while args:
            spec = args.pop(0)
            if spec == "--take":
                continue
            path, clauses = spec.rsplit(":", 1)
            want = set(clauses.split(","))
            rows += [r for r in parse(path) if r["clause"] in want]
    else:
        rows = parse(argv[2])
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)

    def one(key):
        rs = by.get(key, [])
        if not rs:
            return "MISSING"
        if any(r["res"] is None for r in rs):
            return "FAIL"
        return "PASS" if all(r["res"] == "PASS" or (key == "A5-MAPPER" and r["res"].startswith("N/A (inline")) for r in rs) else "FAIL"

    groups = {
        "A1": [f"A1-GATE:{c}" for c in GATE_CELLS] + [f"A1-MIX:{k}:{o}" for k, o in MIX] + ["A1-STREAM"],
        "A2": ["A2"],
        "A5": ["A5-MAPPER", "A5-ENSURE", "A5-BUILD1", "A5-BUILD64"],
        "A6": ["A6"],
        "A7": [f"A7:{s}" for s in SHAPES],
        "A3-i": ["A3-i"],
        "A3-ii": ["A3-ii:both"],
    }
    clauses = {}
    for name, keys in groups.items():
        st = [one(k) for k in keys]
        missing = [k for k, s in zip(keys, st) if s == "MISSING"]
        clauses[name] = ("FAIL" if "FAIL" in st else "MISSING" if missing else "PASS", missing)
        for k, s in zip(keys, st):
            if s == "FAIL":
                for r in by[k]:
                    print(f"failing: {r['line'][:240]}")
    a4 = one("A4")
    correctness = ["A1", "A2", "A5", "A6", "A7"]
    owed = [k for name, (s, miss) in clauses.items() for k in miss]
    if any(clauses[c][0] == "FAIL" for c in correctness):
        reading = "FAIL (no reading)"
    elif owed:
        reading = f"INCOMPLETE (owed: {','.join(owed)})"
    elif clauses["A3-i"][0] == "PASS" and clauses["A3-ii"][0] == "PASS":
        reading = "PROMOTE-ELIGIBLE"
    else:
        reading = "DELETE"
    cards = sorted({r["card"] for r in rows})
    srcs = sorted({r["src"] for r in rows})
    line = (f"DAY37 VERDICT card={card} rule=1.7 " + " ".join(f"{n}={s}" for n, (s, _) in clauses.items())
            + f" A4={a4}(reading) lines={len(rows)} reader_cards={','.join(cards)} -> {reading}")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
