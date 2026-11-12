#!/usr/bin/env python3
"""Prefix check of the long texts (memra#1211): for every row of RUN/stall.jsonl whose arm is not
main, the long text must be a prefix of the main row's text for the same cell and class, or the
other way round (the stream is cut at a time, not a token count). The main rows are the arms
whose name starts with REF (default `main`). usage: prefix1211.py RUNDIR [REF]"""
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
REF = sys.argv[2] if len(sys.argv) > 2 else "main"
rows = [json.loads(l) for l in open(d / "stall.jsonl") if l.strip()]
text = lambda r: (d / r["text_file"]).read_text()
ref = {}
for r in rows:
    if r["arm"].startswith(REF):
        ref.setdefault((r["clients"], r["think_ms"], r["class"]), r)
bad = 0
for r in rows:
    m = ref.get((r["clients"], r["think_ms"], r["class"]))
    if m is None or r is m:
        continue
    a, b = text(r), text(m)
    ok = a.startswith(b) or b.startswith(a)
    bad += not ok
    print(r["arm"], f"{r['clients']}:{int(r['think_ms'])}", r["class"], "prefix-equal" if ok else "DIFFERS", len(a), len(b))
print("prefix check:", "PASS" if bad == 0 else f"FAIL ({bad})")
