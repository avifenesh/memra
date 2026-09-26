#!/usr/bin/env python3
"""DAY82 section 2's local check reader: each run's exit, MATCH, gen-only and window seconds, and host demand sequence
(the `[expert-host-slru] key=` lines without the slot, SHA-256, first 16 hex, as day82-read.py reads it); then each
run's tape (the `tokens:` line) and sequence against i18-a's. usage: gpu-check-read.py <out-dir>
"""
import hashlib
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
SLOT = re.compile(r" slot=\d+")
runs, fails = {}, []
for label in ("i18-a", "i20-a", "i20-b", "i18-b"):
    text = (out / f"{label}.log").read_text(errors="replace")
    rc = (out / f"{label}.exit").read_text().strip()
    lines = text.splitlines()
    gen = next((l for l in lines if l.startswith("generated ")), "")
    win = next((l for l in lines if "STEADY-STATE window:" in l), "")
    tape = next((l for l in lines if l.startswith("tokens: ")), "")
    trace = [SLOT.sub("", l) for l in lines if "[expert-host-slru] key=" in l]
    h = hashlib.sha256("\n".join(trace).encode()).hexdigest()[:16]
    match = any("MATCH" in l and "argmax" in l for l in lines)
    runs[label] = (tape, h)
    g = re.search(r"generated \d+ tokens in [0-9.]+s", gen)
    w = re.search(r"window: .*", win)
    print(f"{label} rc={rc} {'MATCH' if match else 'NO-MATCH'} {g.group(0) if g else 'no gen line'} "
          f"{w.group(0) if w else 'no window line'} trace={h} lines={len(trace)}")
    if rc != "0" or not match or not tape or not trace:
        fails.append(f"{label}: rc={rc} match={match} tape={bool(tape)} trace_lines={len(trace)}")
base = runs["i18-a"]
for label in ("i20-a", "i20-b", "i18-b"):
    tape, h = runs[label]
    same_tape, same_seq = tape == base[0], h == base[1]
    print(f"{label}: {'the same tape' if same_tape else 'A DIFFERENT TAPE'} and "
          f"{'the same host demand sequence' if same_seq else 'A DIFFERENT HOST DEMAND SEQUENCE'} as i18-a")
    if not (same_tape and same_seq):
        fails.append(f"{label} differs from i18-a")
print("DAY82 GPU CHECK " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
sys.exit(1 if fails else 0)
