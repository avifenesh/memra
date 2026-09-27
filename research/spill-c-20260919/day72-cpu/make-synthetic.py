#!/usr/bin/env python3
"""DAY72: a synthetic `gap15` cell for the reader's mechanics dry check. Part A is DAY60's own `gap` receipts
(pro-single-day61/gap/, copied); Part B is four invented runs: DAY60 run logs copied under the profiled labels and a
made-up SQLite per run with Nsight Systems' table shapes (session start, string ids, kernels, copies) placed around
the log's window. Every Part B value is invented; the reading it gives means nothing.
usage: make-synthetic.py <day60-gap-cell-dir> <out-cell-dir>
"""
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
import re

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2]) / "ev"
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    shutil.copy(f, out / f.name)
(out / "nsys.txt").write_text("nsys=/usr/local/cuda/bin/nsys\nNVIDIA Nsight Systems version synthetic\n")
WINDOW = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3})\tMoE cache STEADY-STATE window: (\d+) decode steps in ([0-9.]+)s", re.M)
for label, source, idle_ms in (("p-ref-r1", "o1-ref-r1", 0.5), ("p-on-r1", "o1-on-r1", 0.8), ("p-on-r2", "o1-on-r2", 0.8),
                               ("p-ref-r2", "o1-ref-r2", 0.5)):
    for ext in (".log", ".exit"):
        shutil.copy(src / f"{source}{ext}", out / f"{label}{ext}")
    text = (out / f"{label}.log").read_text(errors="replace")
    m = WINDOW.search(text)
    day = datetime(2026, 9, 24, tzinfo=timezone.utc)
    end = datetime.combine(day.date(), datetime.strptime(m.group(1), "%H:%M:%S.%f").time(), tzinfo=timezone.utc)
    start_ns = int(end.timestamp() * 1e9) - 20 * 10**9  # the session began 20 s before the window's end
    hi = 20 * 10**9
    lo = hi - int(float(m.group(3)) * 1e9)
    db = sqlite3.connect(str(out / f"{label}.sqlite"))
    db.execute("create table TARGET_INFO_SESSION_START_TIME (utcEpochNs int, utcTime text, localTime text)")
    db.execute("insert into TARGET_INFO_SESSION_START_TIME values (?, 'x', 'x')", (start_ns,))
    db.execute("create table StringIds (id int, value text)")
    db.executemany("insert into StringIds values (?, ?)", [(1, "moe_gemv"), (2, "attn"), (3, "argmax")])
    db.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, shortName int)")
    db.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start int, end int, bytes int, copyKind int)")
    step = (hi - lo) // 32
    rows, copies = [], []
    for t in range(32):
        s0 = lo + t * step
        busy = step - int(idle_ms * 1e6)
        rows += [(s0, s0 + busy // 2, 1), (s0 + busy // 2, s0 + busy, 2 if t % 2 else 3)]
        copies.append((s0 + busy, s0 + busy + int(0.1e6), 860160, 1))
    db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?)", rows)
    db.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?)", copies)
    db.commit()
print(f"synthetic cell {out.parent}: Part A from {src}, Part B four invented profiles")
