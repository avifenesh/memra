#!/usr/bin/env python3
"""Kernel time by (name, grid) from an nsys sqlite export, for the F8F4 GEMM kernels and the total.
usage: kshapes.py prof.sqlite [name-substring ...]"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
subs = sys.argv[2:] or ["mul_mat_f8f4v", "f8f4v2_image"]
rows = db.execute(
    "SELECT s.value, k.gridX, k.gridY, COUNT(*), SUM(k.end - k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.shortName GROUP BY s.value, k.gridX, k.gridY").fetchall()
tot = sum(r[4] for r in rows)
print(f"all kernels: {tot / 1e9:.3f} s")
sel = [r for r in rows if any(x in r[0] for x in subs)]
for name, gx, gy, n, t in sorted(sel, key=lambda r: -r[4]):
    print(f"{t / 1e9:9.3f} s  {n:6d} x {t / n / 1e6:8.3f} ms  grid {gx}x{gy}  {name[:60]}")
print(f"selected: {sum(r[4] for r in sel) / 1e9:.3f} s")
