#!/usr/bin/env python3
"""Compare the boot and request lines of two served cells' server logs (B and T): every line
tagged [server], [worker], [spec-k], [spec-gate], [route-contract], [spec-pool] or [admission],
with the timestamp, the build id and time or memory figures taken out, and every line that names
a penalty, a sampler or a retired name. Prints the lines both arms print alike, the lines only
one prints, and the count of penalty/sampler lines (0 means neither server printed one).
usage: boot_lines.py B_server.log T_server.log [--lines]   (--lines also prints every tagged line)"""
import re, sys

TAG = re.compile(r"^\S+ \[(server|worker|spec-k|spec-gate|route-contract|spec-pool|admission)\]")
PEN = re.compile(r"penalt|devpen|sampler|SERVE_BATCH|retired", re.I)
def norm(l):
    l = re.sub(r"^\S+ ", "", l)
    l = re.sub(r"memra-0\.\d+\.\d+-[0-9a-f]+", "memra-<build>", l)
    l = re.sub(r"\b\d+(\.\d+)?(s|ms|MB|GB)\b", "<n>", l)
    l = re.sub(r"in \d+\.\d+s", "in <n>", l)
    return l.strip()
def lines(path):
    tagged, pen = [], []
    for l in open(path, errors="replace"):
        if PEN.search(l):
            pen.append(norm(l))
        if TAG.match(l):
            tagged.append(norm(l))
    return tagged, pen
(bt, bp), (tt, tp) = lines(sys.argv[1]), lines(sys.argv[2])
print(f"penalty/sampler/retired-name lines: B {len(bp)}, T {len(tp)}")
for l in sorted(set(bp) | set(tp)):
    print("  ", ("B+T" if l in bp and l in tp else "B  " if l in bp else "  T"), l[:200])
same = [l for l in bt if l in tt]
if "--lines" in sys.argv:
    for l in bt:
        print("  B:", l[:220])
    for l in tt:
        print("  T:", l[:220])
print(f"tagged lines: B {len(bt)}, T {len(tt)}, alike {len(same)}")
for l in bt:
    if l not in tt:
        print("  B only:", l[:200])
for l in tt:
    if l not in bt:
        print("  T only:", l[:200])
