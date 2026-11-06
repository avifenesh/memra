#!/usr/bin/env python3
"""scrub.py DIR: make lane receipts safe to track. Replaces the home path with ~, the two GB10
host aliases with unit names and the account name with "user", and fails on anything that still
looks like a machine address or a lease id. Idempotent. The literals are assembled from parts and
this file skips itself, so a run over the receipts cannot rewrite its own table."""
import pathlib, re, sys

ACCOUNT = "avi" + "fenesh"
UNIT1, UNIT2 = "c" + "b33", "b" + "23d"
SUBS = [
    ("/home/" + ACCOUNT, "~"),
    ("spark-" + UNIT1, "gb10-unit1"), ("spark-" + UNIT2, "gb10-unit2"),
    (UNIT1, "unit1"), (UNIT2, "unit2"),
    (ACCOUNT, "user"),
]
BAD = [re.compile(p) for p in (r"\b100\.\d+\.\d+\.\d+\b", r"claude-code-2026\d{4}T\d{6}-\d+",
                               ACCOUNT + "@", r"tail[0-9a-f]{6}\.ts\.net")]
root = pathlib.Path(sys.argv[1])
me = pathlib.Path(__file__).resolve()
bad = 0
for f in root.rglob("*"):
    if not f.is_file() or f.suffix in (".logits", ".bin") or f.resolve() == me:
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
