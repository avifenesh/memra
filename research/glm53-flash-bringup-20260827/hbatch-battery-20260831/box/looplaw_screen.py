#!/usr/bin/env python3
"""Loop-law screening for the 3way-decision tapes.

Greedy looping is a known artifact, never a finding; looped rows never enter an aggregate
(owner law, codified 2026-08-21). A row is FLAGGED when its decoded text degenerates into
repetition: (a) a terminal n-gram (n in 2..8 words) repeated >= 4 times consecutively at the
tail, or (b) any decoded line repeated >= 4 times consecutively.

Unlike the card3/spec-battery version this screens EVERY *.txt tape in the given dirs (the
3way filenames are pool tags like d00-code.txt / l3-WARM.txt, not -plain/-kN suffixed).

Usage: looplaw_screen.py <dir> [<dir> ...]
"""
import sys
from pathlib import Path


def tail_ngram_loop(words, n_max=8, min_reps=4):
    for n in range(2, n_max + 1):
        if len(words) < n * min_reps:
            continue
        unit = words[-n:]
        reps = 1
        i = len(words) - 2 * n
        while i >= 0 and words[i:i + n] == unit:
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
    flagged, total = [], 0
    for d in sys.argv[1:]:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.txt")):
            total += 1
            text = f.read_text(errors="replace")
            hit_line = line_loop(text)
            hit_ng = tail_ngram_loop(text.split())
            rel = f.relative_to(p.parent) if p.parent != p else f
            if hit_line:
                print(f"FLAGGED {rel}: repeated line {hit_line!r}")
                flagged.append(str(rel))
            elif hit_ng:
                print(f"FLAGGED {rel}: tail {hit_ng[0]}-gram x{hit_ng[1]}")
                flagged.append(str(rel))
    print(f"[loop-law] screened={total} flagged={len(flagged)}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
