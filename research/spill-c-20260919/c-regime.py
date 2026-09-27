#!/usr/bin/env python3
"""The card's regime over one collector hold, from its 250 ms `command.gpu.csv`: SM clock, power, temperature and
memory used (min, median, max, N). The day-40 regime lines' form. usage: c-regime.py <cell-dir>"""
import csv
import statistics
import sys
from pathlib import Path

rows = list(csv.reader(open(Path(sys.argv[1]) / "command.gpu.csv")))
head = [h.strip() for h in rows[0]]


def col(prefix):
    return next(i for i, h in enumerate(head) if h.startswith(prefix))


for name, prefix in (("sm_mhz", "clocks.current.sm"), ("power_w", "power.draw"), ("temp_c", "temperature.gpu"),
                     ("mem_mib", "memory.used")):
    i = col(prefix)
    vals = []
    for r in rows[1:]:
        try:
            vals.append(float(r[i].split()[0]))
        except (ValueError, IndexError):
            pass
    print(f"{name} min={min(vals)} median={statistics.median(vals)} max={max(vals)} N={len(vals)}")
