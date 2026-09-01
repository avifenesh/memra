#!/usr/bin/env python3
"""Loop-law screening for the card3 acceptance probe outputs.

Greedy looping is a known artifact, never a finding; looped rows never enter an
aggregate (owner law, codified 2026-08-21). This screens the banked decoded outputs:
a row is FLAGGED when its token tape (or decoded text) degenerates into repetition —
(a) a terminal n-gram (n in 2..8) repeated >= 4 times consecutively at the tail, or
(b) any decoded line repeated >= 4 times consecutively.

Usage: looplaw_screen.py <out_dir> [...]  (dirs holding *-plain.ids / *-k*.txt)
Prints one row per file: FLAGGED/clean + the repeating unit if flagged.
"""
import re
import sys
from pathlib import Path


def tail_ngram_loop(ids, n_max=8, min_reps=4):
    for n in range(2, n_max + 1):
        if len(ids) < n * min_reps:
            continue
        unit = ids[-n:]
        reps = 1
        i = len(ids) - 2 * n
        while i >= 0 and ids[i : i + n] == unit:
            reps += 1
            i -= n
        if reps >= min_reps:
            return n, reps
    return None


def line_loop(text, min_reps=4):
    lines = [l for l in text.splitlines() if l.strip()]
    run, prev = 1, None
    for l in lines:
        if l == prev:
            run += 1
            if run >= min_reps:
                return l[:60]
        else:
            run, prev = 1, l
    return None


def main():
    flagged = []
    for d in sys.argv[1:]:
        for f in sorted(Path(d).iterdir()):
            if f.suffix == ".ids":
                ids = [int(x) for x in f.read_text().split()]
                hit = tail_ngram_loop(ids)
                if hit:
                    print(f"FLAGGED {f.name}: tail {hit[0]}-gram x{hit[1]}")
                    flagged.append(f.name)
            elif f.suffix == ".txt" and re.search(r"-(plain|k\d)", f.stem):
                hit = line_loop(f.read_text())
                if hit:
                    print(f"FLAGGED {f.name}: repeated line {hit!r}")
                    flagged.append(f.name)
    print(f"\n{len(flagged)} flagged file(s)" if flagged else "\nno loops flagged")


if __name__ == "__main__":
    main()
