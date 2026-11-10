#!/usr/bin/env python3
"""Compare two gate receipts row by row: the argmax margin probe's window rows (pos, prefill
top1, margin_p, decode top1, margin_d, spread, agree) and its summary lines, or for any other
receipt every verdict-bearing line (logits differ, acceptance, MATCH, PASS, FAIL, ALL GREEN).
Prints one line per receipt pair: identical, or the first rows that differ.
usage: same_rows.py OLD NEW"""
import re, sys

KEEP = re.compile(r"logits differ|acceptance|MATCH|PASS|FAIL|ALL GREEN|margin distribution|DECISION|agreement|flip|SERVED")


def rows(path):
    win, other = [], []
    for line in open(path):
        line = line.rstrip("\n")
        if re.match(r"^\d{4,6}\s", line):
            win.append(line.split())
        elif KEEP.search(line) and not line.startswith(("  raw:", "# ")):
            other.append(re.sub(r"\d+(\.\d+)?s\b|in \d+\.\d+s|= \d+\.\d+ tok/s|\(\d+\.\d+x vs generate; this run's prime \d+\.\d+s\)", "", line))
    return win, other


a, b = rows(sys.argv[1]), rows(sys.argv[2])
diffs = []
for x, y in zip(a[0], b[0]):
    if x != y:
        diffs.append(f"row {x[0]}: {' '.join(x[1:])} | {' '.join(y[1:])}")
if len(a[0]) != len(b[0]):
    diffs.append(f"window rows {len(a[0])} vs {len(b[0])}")
for x, y in zip(a[1], b[1]):
    if x != y:
        diffs.append(f"line: {x.strip()[:120]} | {y.strip()[:120]}")
if len(a[1]) != len(b[1]):
    diffs.append(f"verdict lines {len(a[1])} vs {len(b[1])}")
name = sys.argv[2].rsplit("/", 1)[-1]
if diffs:
    print(f"{name}: DIFFERS in {len(diffs)} place(s)")
    for d in diffs[:12]:
        print("   ", d)
else:
    print(f"{name}: identical ({len(a[0])} window rows, {len(a[1])} verdict lines)")
