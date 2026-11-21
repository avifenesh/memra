#!/usr/bin/env python3
"""Lane S: the clampcheck rule over a 250 ms nvidia-smi log: a loaded sample (power > 60 W) at P2 or
with memory at 11001 MHz voids the take. usage: smi_check.py smi.csv"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = [h.strip() for h in rows[0]]
loaded = bad = 0
sm, mem, pw = [], set(), []
for r in rows[1:]:
    d = dict(zip(hdr, [x.strip() for x in r]))
    try:
        p = float(d["power.draw [W]"].split()[0])
    except (KeyError, ValueError):
        continue
    if p <= 60:
        continue
    loaded += 1
    m = d["clocks.current.memory [MHz]"].split()[0]
    mem.add(m)
    sm.append(int(d["clocks.current.sm [MHz]"].split()[0]))
    pw.append(p)
    if d["pstate"] == "P2" or m == "11001":
        bad += 1
sm.sort(); pw.sort()
print(f"loaded samples {loaded}, void samples (P2 or 11001 MHz) {bad}, memory clocks {sorted(mem)}, "
      f"SM p50 {sm[len(sm) // 2] if sm else 0} MHz, power p50 {pw[len(pw) // 2] if pw else 0:.0f} W")
sys.exit(1 if bad else 0)
