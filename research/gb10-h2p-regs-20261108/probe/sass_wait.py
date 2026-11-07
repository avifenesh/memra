#!/usr/bin/env python3
"""sass_wait.py CSV: stall samples by opcode class and the top instructions by wait stalls."""
import csv, sys, collections, re
rows = list(csv.reader(open(sys.argv[1])))
h = rows[1]
ix = {n: i for i, n in enumerate(h)}
def f(r, n):
    try: return float(r[ix[n]].replace(",", ""))
    except: return 0.0
by = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
tot = [0.0, 0.0]
top = []
for r in rows[2:]:
    if len(r) < len(h): continue
    src = r[ix["Source"]].strip()
    op = re.split(r"[ .]", src.split(" ", 1)[0] if not src.startswith("@") else src.split(" ", 2)[1])[0]
    w, a, ex = f(r, "stall_wait"), f(r, "Warp Stall Sampling (All Samples)"), f(r, "Instructions Executed")
    b = by[op]; b[0] += a; b[1] += w; b[2] += ex; b[3] += f(r, "stall_math")
    tot[0] += a; tot[1] += w
    top.append((w, src[:90]))
print(f"all samples {tot[0]:.0f}, wait {tot[1]:.0f}")
print("op        all%   wait%  math%  executed")
for op, b in sorted(by.items(), key=lambda kv: -kv[1][0])[:16]:
    print(f"{op:8s} {100*b[0]/tot[0]:6.1f} {100*b[1]/tot[1]:6.1f} {100*b[3]/max(1,tot[0]):6.1f} {b[2]:.3e}")
print("top wait instructions:")
for w, s in sorted(top, reverse=True)[:14]:
    print(f"{w:9.0f}  {s}")
