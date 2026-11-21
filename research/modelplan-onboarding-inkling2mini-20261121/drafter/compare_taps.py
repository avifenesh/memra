#!/usr/bin/env python3
"""Lane S: the route's captured drafter taps against the oracle trunk's own (independent hooks on
the PR model, oracle_taps.py), per tap: relative L2 error over every position of every prompt.
usage: compare_taps.py PP_DIR [ORACLE_PP_DIR]   (oracle taps default to PP_DIR's <id>.oracle.taps.f32)"""
import json
import sys
from pathlib import Path

import numpy as np

pp = Path(sys.argv[1])
odir = Path(sys.argv[2]) if len(sys.argv) > 2 else pp
H = 2304
num = np.zeros(4)
den = np.zeros(4)
worst = np.zeros(4)
for jf in sorted(pp.glob("*.json")):
    if jf.name.endswith(".oracle.json"):
        continue
    m = json.load(open(jf))
    o = odir / f"{m['id']}.oracle.taps.f32"
    if not o.exists():
        continue
    n = len(m["seq"])
    g = np.fromfile(pp / f"{m['id']}.taps.f32", dtype=np.float32).reshape(n, 4, H).astype(np.float64)
    r = np.fromfile(o, dtype=np.float32).reshape(n, 4, H).astype(np.float64)
    d = ((g - r) ** 2).sum(-1)
    s = (r ** 2).sum(-1)
    num += d.sum(0)
    den += s.sum(0)
    worst = np.maximum(worst, np.sqrt(d / np.maximum(s, 1e-30)).max(0))
print("tap\trel L2 (all positions)\tworst position rel L2")
for t, name in enumerate(["slot 5 input", "slot 29 input", "slot 47 input", "final norm"]):
    print(f"{name}\t{np.sqrt(num[t] / den[t]):.3e}\t{worst[t]:.3e}")
