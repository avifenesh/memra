#!/usr/bin/env python3
"""scrub.py DIR: make lane receipts safe to track. Replaces the home path with ~, the two GB10
host aliases with unit names, and fails on anything that still looks like a machine address or a
lease id. Idempotent."""
import pathlib, re, sys

SUBS = [
    ("~", "~"),
    ("gb10-unit1", "gb10-unit1"), ("gb10-unit2", "gb10-unit2"),
    ("unit1", "unit1"), ("unit2", "unit2"),
    ("user", "user"),
]
BAD = [re.compile(p) for p in (r"\b100\.\d+\.\d+\.\d+\b", r"claude-code-2026\d{4}T\d{6}-\d+", r"user@", r"tail[0-9a-f]{6}\.ts\.net")]
root = pathlib.Path(sys.argv[1])
bad = 0
for f in root.rglob("*"):
    if not f.is_file() or f.suffix in (".logits", ".bin"):
        continue
    try:
        s = f.read_text()
    except UnicodeDecodeError:
        continue
    t = s
    for a, b in SUBS:
        t = t.replace(a, b)
    if t != s:
        f.write_text(t)
    for p in BAD:
        for m in p.finditer(t):
            print(f"BAD {f}: {m.group(0)}"); bad += 1
print("scrub", "FAIL" if bad else "OK", bad)
sys.exit(1 if bad else 0)
