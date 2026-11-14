#!/usr/bin/env python3
"""assemble1307.py RECORD_DIR SRC=DST...: copy campaign receipts into the research record, home
paths written as ~ (as memra#1221's and memra#1257's records are), trailing whitespace and
trailing blank lines dropped (the 1 Hz memory logs end every line with a space).

SRC is a campaign part's directory as campaign1307.sh wrote it (RUNS/CAMPAIGN/PART), DST its place
under RECORD_DIR. A cell directory (NN-ARM) keeps rows.jsonl, guard.log, run.log, trace.txt (the
server's [diag1307], [spec], [spec-pool], [admit-trim], [device-trim] and OOM lines), memfree.log,
and for the short-memory cells holder.log, mem.jsonl, kernel-oom-count.txt and the kernel log's OOM
lines. A cell that died before its trace was cut keeps its whole server.log. A probe directory keeps probe.jsonl and its logs. The part's campaign.log comes along."""
import re
import sys
from pathlib import Path

rec = Path(sys.argv[1])
HOME = str(Path.home())
CELL = ("rows.jsonl", "guard.log", "run.log", "trace.txt", "memfree.log", "holder.log", "kernel-oom-count.txt")
OOM = re.compile(r"out of memory|oom-kill|oom_reaper|invoked oom-killer", re.I)


def clean(text):
    text = re.sub(r"/home/[a-z0-9_-]+", "~", text.replace(HOME, "~"))
    lines = [x.rstrip() for x in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(clean(src.read_text(errors="replace")))


for pair in sys.argv[2:]:
    s, d = pair.split("=", 1)
    src, dst = Path(s), rec / d
    if (src / "campaign.log").exists():
        put(src / "campaign.log", dst / "campaign.log")
    for f in sorted(src.glob("*.log")) + sorted(src.glob("probe.jsonl")):
        if f.name not in ("campaign.log", "memfree.log"):
            put(f, dst / f.name)
    for cell in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in CELL:
            if (cell / f).exists():
                put(cell / f, dst / cell.name / f)
        if not (cell / "trace.txt").exists() and (cell / "server.log").exists():
            put(cell / "server.log", dst / cell.name / "server.log")  # a cell that died before its trace
        if (cell / "holder.log").exists() and (cell / "mem.jsonl").exists():
            put(cell / "mem.jsonl", dst / cell.name / "mem.jsonl")
        k = cell / "kernel.log"
        if k.exists():
            lines = [x for x in k.read_text(errors="replace").splitlines() if OOM.search(x)]
            (dst / cell.name / "kernel-oom-lines.txt").write_text(clean("\n".join(lines) + ("\n" if lines else "")))
        print(d, cell.name, sorted(p.name for p in (dst / cell.name).iterdir()))
