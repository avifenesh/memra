"""Summarize cta-probe.sh / e1-probe.sh logs: per (T, shape, build) the v2 and v3 medians over the
runs and the v3 gain over v2; with builds A and B, also v3(A) / v3(B) - 1 (the B mainloop's gain).
usage: probe_sum.py OUTDIR"""
import re
import statistics
import sys
from pathlib import Path

rows = {}
pat = re.compile(r"^(.+?)\s+v2\s+([\d.]+) ms\s+v3\s+([\d.]+) ms\s+[+-][\d.]+%\s+bytes (\w+)")
for f in sorted(Path(sys.argv[1]).glob("*-T*-*.log")):
    m = re.match(r"\d+-T(\d+)-(\w)-(\d+)\.log", f.name)
    T, build = int(m.group(1)), m.group(2)
    for line in f.read_text().splitlines():
        p = pat.match(line)
        if p:
            key = (T, p.group(1).strip(), build)
            rows.setdefault(key, []).append((float(p.group(2)), float(p.group(3)), p.group(4)))
builds = sorted({k[2] for k in rows})
print(f"{'T':>5}  {'shape':<27} {'build':<5} {'n':>2} {'v2 ms':>8} {'v3 ms':>8} {'v3 vs v2':>9}  bytes")
for key in sorted(rows):
    v = rows[key]
    v2 = statistics.median(x[0] for x in v)
    v3 = statistics.median(x[1] for x in v)
    same = "identical" if all(x[2] == "identical" for x in v) else "DIFFER"
    print(f"{key[0]:>5}  {key[1]:<27} {key[2]:<5} {len(v):>2} {v2:8.3f} {v3:8.3f} {100 * (v2 / v3 - 1):+8.2f}%  {same}")
if builds == ["A", "B"]:
    print()
    print("v3 mainloop B against A (median v3 times; v2 is the same SASS in both builds):")
    for T, shape in sorted({(k[0], k[1]) for k in rows}):
        a = statistics.median(x[1] for x in rows[(T, shape, "A")])
        b = statistics.median(x[1] for x in rows[(T, shape, "B")])
        a2 = statistics.median(x[0] for x in rows[(T, shape, "A")])
        b2 = statistics.median(x[0] for x in rows[(T, shape, "B")])
        print(f"{T:>5}  {shape:<27} v3 A {a:7.3f}  B {b:7.3f}  {100 * (a / b - 1):+6.2f}%   (v2 A {a2:.3f} B {b2:.3f})")
