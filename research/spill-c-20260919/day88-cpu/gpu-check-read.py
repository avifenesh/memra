#!/usr/bin/env python3
"""DAY88 section 4's local check reader: each run's exit, MATCH, gen-only and window seconds, process wall, host
demand sequence (the `[expert-host-slru] key=` lines without the slot, SHA-256, first 16 hex) and the door's lines;
then: every run `MATCH` with one tape; p88's and i22's door runs one host demand sequence; p88 prints `qualified`,
`installed`, the registered pool and `[moe-prefetch] default=on`; i22 prints `installed`; the rollback prints `off:
MEMRA_EXPERTS_VIA_TIER=0`, the prefetch default on, and no host demand line. usage: gpu-check-read.py <out-dir>
"""
import hashlib
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
SLOT = re.compile(r" slot=\d+")
runs, fails = {}, []
for label in ("p88-a", "i22-a", "i22-b", "p88-b", "leg"):
    text = (out / f"{label}.log").read_text(errors="replace")
    rc = (out / f"{label}.exit").read_text().strip()
    wall = (out / f"{label}.wall").read_text().strip() if (out / f"{label}.wall").exists() else "-"
    lines = text.splitlines()
    gen = re.search(r"generated \d+ tokens in [0-9.]+s", text)
    win = re.search(r"STEADY-STATE window: (.*)", text)
    tape = next((l for l in lines if l.startswith("tokens: ")), "")
    trace = [SLOT.sub("", l) for l in lines if "[expert-host-slru] key=" in l]
    h = hashlib.sha256("\n".join(trace).encode()).hexdigest()[:16] if trace else "-"
    match = any("MATCH" in l and "argmax" in l for l in lines)
    marks = {
        "qualified": "[experts-via-tier] qualified sha256=df27a780" in text,
        "installed": "[experts-via-tier] installed artifact_sha256=df27a780" in text,
        "registered": bool(re.search(r"host pinned pool bytes=.* registered$", text, re.M)),
        "allocated": bool(re.search(r"host pinned pool bytes=.* allocated$", text, re.M)),
        "pf_on": "[moe-prefetch] default=on" in text,
        "off_rollback": "[experts-via-tier] off: MEMRA_EXPERTS_VIA_TIER=0" in text,
    }
    runs[label] = (tape, h, marks)
    print(f"{label} rc={rc} {'MATCH' if match else 'NO-MATCH'} {gen.group(0) if gen else 'no gen line'} "
          f"window: {win.group(1) if win else '-'} wall={wall}s trace={h} lines={len(trace)} "
          + " ".join(k for k, v in marks.items() if v))
    if rc != "0" or not match or not tape:
        fails.append(f"{label}: rc={rc} match={match} tape={bool(tape)}")
base = runs["p88-a"][0]
for label, (tape, _, _) in runs.items():
    if tape != base:
        fails.append(f"{label}: a different tape")
door = {runs[l][1] for l in ("p88-a", "i22-a", "i22-b", "p88-b")}
if len(door) != 1 or "-" in door:
    fails.append(f"door runs' host demand sequences: {sorted(door)}")
for l in ("p88-a", "p88-b"):
    m = runs[l][2]
    if not (m["qualified"] and m["installed"] and m["registered"] and m["pf_on"]):
        fails.append(f"{l}: the default door's lines {m}")
for l in ("i22-a", "i22-b"):
    if not runs[l][2]["installed"]:
        fails.append(f"{l}: no installed line")
m = runs["leg"][2]
if not (m["off_rollback"] and m["pf_on"] and not m["installed"]) or runs["leg"][1] != "-":
    fails.append(f"leg: the rollback's lines {m} trace={runs['leg'][1]}")
print("DAY88 GPU CHECK " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
sys.exit(1 if fails else 0)
