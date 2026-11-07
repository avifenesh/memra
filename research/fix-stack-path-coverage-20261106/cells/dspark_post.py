#!/usr/bin/env python3
"""Checks on a dspark-serve-smoke evidence dir that the smoke itself does not make (RTX 5090):
  - the #1069 boot line: doors default OFF on this card, all three read false with no door env;
  - the pooled DFlash2 cohort: the concurrent LOW=2 pair (on-c1 = P2, on-c2 = P3, greedy, 96
    tokens) must be byte-equal to the same prompts served solo (on-r2/on-r3) and spec-off
    (off-r2/off-r3), and the cohort must have engaged.
usage: dspark_post.py EVDIR    (prints CELL lines; exit 1 on a FAIL)
"""
import glob
import hashlib
import json
import os
import re
import sys

ev = sys.argv[1]
fails = 0


def cell(name, ok, why=""):
    global fails
    print(f"CELL {name} {'PASS' if ok else 'FAIL'} {why}".rstrip())
    fails += 0 if ok else 1


def text(name):
    try:
        m = json.load(open(os.path.join(ev, name + ".json")))["choices"][0]["message"]
    except (OSError, KeyError, IndexError, ValueError) as e:
        return f"<unreadable {name}: {e}>"
    return (m.get("reasoning") or "") + "\x1f" + (m.get("content") or "")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


logs = {os.path.basename(p): open(p, errors="replace").read() for p in glob.glob(os.path.join(ev, "*.log"))}
boot = [(n, m.group(0)) for n, t in logs.items() for m in re.finditer(r"\[dspark\] prefix restore doors default [^\n]*", t)]
cell("door-boot-line-present", bool(boot), "; ".join(f"{n}: {b[:140]}" for n, b in boot[:3]))
off = [b for _, b in boot if "default OFF" in b and "restore=false partial=false capture=false" in b]
cell("door-default-off-on-5090", bool(off), off[0][:160] if off else "no boot line reads default OFF with all three false")
for c, r in (("on-c1", "2"), ("on-c2", "3")):
    a, solo, plain = text(c), text("on-r" + r), text("off-r" + r)
    cell(f"pooled-{c}-identity", a == solo == plain, f"{c}={sha(a)} on-r{r}={sha(solo)} off-r{r}={sha(plain)}")
eng = [l.strip()[:160] for t in logs.values() for l in t.splitlines() if "pooled DFlash2 cohort engaged" in l]
cell("pooled-dflash2-engaged", bool(eng), eng[0] if eng else "no engagement line in the server logs")
sys.exit(1 if fails else 0)
