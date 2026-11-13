#!/usr/bin/env python3
"""memra#1257: copy the GB10 cells' receipts into the research record, sanitized like memra#1221's
(home paths as ~): ab.jsonl, guard.log, run.log, and trace.txt (the server's [admit-trim], [spec],
[device-trim], [admission], [admit-cal] and OOM lines), plus the MemFree log of the cells that kept
a background reader. usage: assemble.py RUNS_DIR RECORD_DIR"""
import re
import shutil
import sys
from pathlib import Path

runs, rec = Path(sys.argv[1]), Path(sys.argv[2])
HOME = str(Path.home())
groups = {"h1257": "gb10/full-cache", "h1257s": "gb10/short"}
pat = re.compile(r"\[(admit-trim|spec|device-trim|admission|admit-cal|admit-oom|admit-mem)\]|OUT_OF_MEMORY|out of memory")


def clean(text: str) -> str:
    return text.replace(HOME, "~")


for camp, sub in groups.items():
    for cell in sorted((runs / camp).glob("[0-9][0-9]-*")):
        name = cell.name
        dst = rec / (sub if not name.startswith("3") else "gb10/cold-first") / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("ab.jsonl", "guard.log", "run.log", "holder.log", "kernel-oom-count.txt"):
            if (cell / f).exists():
                (dst / f).write_text(clean((cell / f).read_text(errors="replace")))
        srv = cell / "server.log"
        if srv.exists():
            lines = [l for l in srv.read_text(errors="replace").splitlines() if pat.search(l)]
            (dst / "trace.txt").write_text(clean("\n".join(lines) + "\n"))
        # fill_cache3.sh names its MemFree log by the memra-server pid it follows (run.log says which).
        m = re.search(r"follows memra-server (\d+)", (cell / "run.log").read_text(errors="replace")) if (cell / "run.log").exists() else None
        mf = runs / f"memfree-{m.group(1)}.log" if m else None
        if mf and mf.exists():
            shutil.copy(mf, dst / "memfree.log")
        print(dst.relative_to(rec), sorted(p.name for p in dst.iterdir()))
