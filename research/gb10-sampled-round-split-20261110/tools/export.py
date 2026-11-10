#!/usr/bin/env python3
"""export.py RUNS_DIR TABLES_DIR DEST CELL...: the receipts of the cells, small files only.

Per cell from RUNS_DIR/CELL: reqs.jsonl, server.log, guard.log, run.log, telemetry.csv. From
TABLES_DIR/CELL (when it exists): what split.py and fold.py wrote for a profiled cell. Receipts name
paths from the home directory down, `ls` lines keep the size and the file name, and the profiler
export's progress line is dropped. Profiles (prof.nsys-rep, prof.sqlite) stay on the machine.
"""
import os
import pathlib
import re
import sys

runs, tables, dest = (pathlib.Path(p) for p in sys.argv[1:4])
home = os.path.expanduser("~")


def clean(text):
    text = re.sub(r"/home/[A-Za-z0-9_.-]+", "~", text.replace(home, "~"))
    lines = []
    for line in text.splitlines():
        if line.startswith("Processing ") and "events:" in line:
            continue
        m = re.match(r"^-[rwx-]{9}\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\w+\s+\d+\s+[\d:]+\s+(.*)$", line)
        if m:
            line = f"{m.group(1)} bytes  {m.group(2)}"
        lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


for cell in sys.argv[4:]:
    out = dest / cell
    out.mkdir(parents=True, exist_ok=True)
    for f in ("reqs.jsonl", "server.log", "guard.log", "run.log", "telemetry.csv"):
        src = runs / cell / f
        if src.exists():
            (out / f).write_text(clean(src.read_text(errors="replace")))
    if (tables / cell).is_dir():
        for src in sorted((tables / cell).iterdir()):
            (out / src.name).write_text(clean(src.read_text()))
    print(cell, sorted(p.name for p in out.iterdir()))
