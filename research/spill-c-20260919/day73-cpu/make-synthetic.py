#!/usr/bin/env python3
"""DAY73: a synthetic `compact` cell for the reader's mechanics dry check, from machine b's DAY71 receipts: its first
five runs of each arm in each order (20 runs), its sampler rows plus invented `B` and `P` rows at each pass, and four
invented `strace -c` summaries. The `B`, `P` and strace values are invented; the reading means nothing.
usage: make-synthetic.py <day71-b-cell-dir> <out-cell-dir>
"""
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
keep = {f"o{o}-{a}-r{i}" for o in (1, 2) for a in ("ref", "i15") for i in range(1, 6)}
for f in src.iterdir():
    stem = f.name.split(".")[0]
    if re.match(r"o[12]-", stem) and stem not in keep:
        continue
    if f.name == "marks.tsv":
        (out / f.name).write_text("".join(l for l in f.read_text().splitlines(True) if l.split("\t")[1].split()[0] in keep))
        continue
    if f.name != "sched.tsv":
        shutil.copy(f, out / f.name)
rows, pids = [], set()
for line in (src / "sched.tsv").read_text().splitlines():
    rows.append(line)
    p = line.split("\t")
    if len(p) > 3 and p[1] == "O":
        pids.add((p[0], p[2]))
    if len(p) > 2 and p[1] == "C" and p[2].startswith("cpu0 "):
        rows.append(f"{p[0]}\tB\t0 Normal\t100,90,80,70,60,50,40,30,20,10,5")
for t, pid in sorted(pids):
    rows.append(f"{t}\tP\t{pid}\tVmRSS=30000000,VmLck=0,VmPin=26000000,AnonHugePages=0,Shared_Hugetlb=0,Swap=0,Locked=0")
(out / "sched.tsv").write_text("\n".join(sorted(rows, key=lambda r: r[:12])) + "\n")
(out / "strace.txt").write_text("strace -- version synthetic\n")
summary = """% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 50.00    0.100000          10     {ioctl}           ioctl
 30.00    0.060000          10      {mmap}           mmap
 20.00    0.040000          10      {madv}           madvise
------ ----------- ----------- --------- --------- ----------------
100.00    0.200000                 {tot}           total
"""
for label, (a, b, c) in (("s-ref-r1", (10000, 500, 10)), ("s-i15-r1", (12000, 900, 4000)),
                         ("s-i15-r2", (12100, 910, 4100)), ("s-ref-r2", (10100, 510, 12))):
    (out / f"{label}.strace").write_text(summary.format(ioctl=a, mmap=b, madv=c, tot=a + b + c))
print(f"synthetic cell {out.parent}: 20 runs from {src}")
