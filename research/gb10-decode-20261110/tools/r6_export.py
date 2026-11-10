#!/usr/bin/env python3
"""r6_export.py RUNS_DIR DEST: the small receipts of the round 6 re-scout cells.

RUNS_DIR holds one directory per cell (r6-decode, r6-nsys1k, r6-nsys131k, r6-nsys260k, r6-kN). Per
cell: pf.jsonl, guard.log (the binary sha256 and the server env), trace.txt (the server log's lines the tables read: the spec K table and verdicts, the
draft load, the cost calibration, the prefix cache and every `[tick-spec]` burst) and the tables the
analysis tools wrote next to the profile (kernels.txt, copies-gaps.txt, parts.txt, roundgap.txt,
big.txt, round6.txt, tab.txt). Profiles, GPU telemetry and memory samples stay in RUNS_DIR.
usage: r6_export.py RUNS_DIR DEST name=dest_name ...   (e.g. r6-decode=decode-unit2)
"""
import os
import pathlib
import re
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
keep = re.compile(r"\[(tick-spec|spec-k|spec-cost|mtp-draft|device-trim|spec-pool|prefix-cache)\]")
home = os.path.expanduser("~")


def put(path, text):
    """Receipts name paths from the home directory down."""
    path.write_text(re.sub(r"/home/[A-Za-z0-9_.-]+", "~", text.replace(home, "~")))


for pair in sys.argv[3:]:
    name, dest = pair.split("=")
    cell, out = src / name, dst / dest
    out.mkdir(parents=True, exist_ok=True)
    for f in ("pf.jsonl", "guard.log", "kernels.txt", "copies-gaps.txt", "parts.txt", "roundgap.txt", "big.txt", "round6.txt", "tab.txt"):
        if (cell / f).exists():
            put(out / f, (cell / f).read_text())
    log = cell / "server.log"
    if log.exists():
        put(out / "trace.txt", "".join(l for l in log.read_text(errors="replace").splitlines(keepends=True) if keep.search(l)))
    print(dest, sorted(p.name for p in out.iterdir()))
