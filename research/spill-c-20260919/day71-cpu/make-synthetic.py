#!/usr/bin/env python3
"""DAY71: a synthetic `core` cell for the reader's mechanics dry check, built from DAY70's receipts: DAY70's run logs
with made-up counter fields on every phase line, DAY70's sampler rows with the day-71 owner fields and made-up
interrupt, softirq, cpuidle, vmstat, powercap and hwmon rows at each pass, and a made-up dmon log. Every added value is
invented; the reading it gives means nothing. usage: make-synthetic.py <day70-cell-dir> <out-cell-dir>
"""
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.suffix in (".exit", ".snap") or f.name in ("marks.tsv", "topology.txt", "pins.txt"):
        shutil.copy(f, out / f.name)
PH = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3}\t\[cpu-probe\] phase=\w+ cpu=(-?\d+) compute_ns=([0-9.]+))$", re.M)


def counted(m):
    ns = float(m.group(3))
    wall = int(ns * (1 << 20))
    tsc = int(wall * 4.3)
    return f"{m.group(1)} cpu_after={m.group(2)} wall_ns={wall} tsc={tsc} mperf={tsc} aperf={int(6 * (1 << 20))}"


for log in src.glob("o[12]-*-r*.log"):
    (out / log.name).write_text(PH.sub(counted, log.read_text(errors="replace")))
(out / "counters.txt").write_text("counters=--cpu-probe-counters\n")
(out / "counters-check.txt").write_text("[cpu-probe] counters-check cpu=1 rdpru=ok cpu_after=1 wall_ns=1 tsc=1 mperf=1"
                                         " aperf=1\nrc=0\n")
rows, seen, first = [], set(), None
for line in (src / "sched.tsv").read_text().splitlines():
    p = line.split("\t")
    if len(p) < 3:
        continue
    t = p[0]
    if first is None:
        first = t
        rows += [f"{t}\tIH\tLOC\tLocal timer interrupts", f"{t}\tIH\tCAL\tFunction call interrupts",
                 f"{t}\tSH\tTIMER\t", f"{t}\tEH\tintel-rapl:0\tpackage-0\treadable"]
        rows += [f"{t}\tDH\t{c}\t{k}\t{n}" for c in range(32) for k, n in enumerate(("POLL", "C1", "C2"))]
    if p[1] == "O" and len(p) == 6:
        tick = int(p[5].split()[0]) // 10_000_000
        rows.append(f"{line}\t100\t0\t{tick - tick // 10}\t{tick // 10}")
        continue
    rows.append(line)
    if p[1] == "C" and p[2].startswith("cpu0 ") and t not in seen:
        seen.add(t)
        n = int(t[6:8])
        rows += [f"{t}\tI\tLOC\t" + ",".join(f"{c}:{250 + (n % 3)}" for c in range(32)), f"{t}\tI\tCAL\t1:{n % 5 + 1}",
                 f"{t}\tS\tTIMER\t1:{n % 4 + 1},4:2,17:1", f"{t}\tD\t17\t0:{n * 10}/1,2:240000/3",
                 f"{t}\tV\tpgfault:{n + 10},nr_free_pages:-{n}", f"{t}\tE\tintel-rapl:0\t{n * 60_000_000 + 10**9}",
                 f"{t}\tH\thwmon1:k10temp/temp1_input={60000 + n * 10}"]
(out / "sched.tsv").write_text("\n".join(rows) + "\n")
t0 = datetime.strptime(first[:8], "%H:%M:%S")
pcie = []
for i in range(0, 600):
    t = (t0 + timedelta(seconds=i)).strftime("%H:%M:%S")
    pcie.append(f"{t}.000\t {t}   0   {1000 + i % 7}    {20 + i % 3}")
(out / "pcie.log").write_text("\n".join([f"{first}\t#Time        gpu  rxpci  txpci"] + pcie) + "\n")
print(f"synthetic cell {out.parent}: {len(list(out.glob('o[12]-*-r*.log')))} run logs, {len(rows)} sampler rows")
