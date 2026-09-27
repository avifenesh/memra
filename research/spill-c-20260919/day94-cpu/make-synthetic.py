#!/usr/bin/python3
"""DAY94: a synthetic `where285` from the I24 sitting's promo receipts (pro-single-day89/promo/ev), for the reader's
dry check only (its readings mean nothing): wn, pn, wnc, pnc from `naked` with the trace lines removed (I25's door
writes none untraced), wo from `naked` as it is, wl, pl, wlc, plc from `legacy`; two traced twins from o1-naked-r1 and
o2-naked-r1; the cpu files and marks rewritten. usage: make-synthetic.py <out-root>"""
import re
import shutil
import sys
from pathlib import Path

src = Path(__file__).resolve().parent.parent / "pro-single-day89" / "promo" / "ev"
out = Path(sys.argv[1]) / "where285" / "ev"
out.mkdir(parents=True)
(out / "cpus.txt").write_text("wide=0-11\npcores=0-7\n")
SOURCE = {"wn": "naked", "pn": "naked", "wnc": "naked", "pnc": "naked", "wo": "naked",
          "wl": "legacy", "pl": "legacy", "wlc": "legacy", "plc": "legacy"}
marks = []
start = {}
for line in (src / "marks.tsv").read_text().splitlines():
    when, _, what = line.partition("\t")
    label, _, kind = what.partition(" ")
    if kind == "start":
        start[label] = when
for arm, base in SOURCE.items():
    for order in ("o1", "o2"):
        for i in range(1, 6):
            s = f"{order}-{base}-r{i}"
            d = f"{order}-{arm}-r{i}"
            text = (src / f"{s}.log").read_text(errors="replace")
            if arm != "wo":
                text = "\n".join(l for l in text.splitlines() if "[expert-host-slru] key=" not in l) + "\n"
            (out / f"{d}.log").write_text(text)
            shutil.copy(src / f"{s}.exit", out / f"{d}.exit")
            (out / f"{d}.cpus").write_text("0-7\n" if arm.startswith("p") else "0-11\n")
            marks.append(f"{start[s]}\t{d} start")
for n, s in ((1, "o1-naked-r1"), (2, "o2-naked-r1")):
    shutil.copy(src / f"{s}.log", out / f"t-pnt-r{n}.log")
    shutil.copy(src / f"{s}.exit", out / f"t-pnt-r{n}.exit")
(out / "marks.tsv").write_text("\n".join(marks) + "\n")
print(f"synthetic where285 at {out}: {len(list(out.glob('o[12]-*.log')))} timed runs")
