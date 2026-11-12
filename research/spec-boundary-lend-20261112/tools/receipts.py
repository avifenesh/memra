#!/usr/bin/env python3
"""receipts.py SRC_RUN_DIR DST_DIR: copy one campaign's cells into the receipts (memra#1221 part 2).
Per cell: ab.jsonl / cells.jsonl / requests rows / replay summary / verdicts, guard.log, run.log,
summary.txt as they are; server.log and mem.jsonl gzipped (server logs that are already gzipped are
copied); gpu.csv and telemetry are left on the machine. Then campaign.log and table.txt. The home
directory becomes `~` in every file (no user path in the receipts)."""
import gzip
import pathlib
import sys

HOME = str(pathlib.Path.home()).encode()
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
KEEP = {"ab.jsonl", "cells.jsonl", "guard.log", "run.log", "summary.txt", "replay.json", "replay.log",
        "metrics.json", "verdicts.jsonl", "cells.log", "table.txt", "campaign.log", "binary.sha"}


def scrub(data: bytes) -> bytes:
    return data.replace(HOME, b"~")


for p in sorted(src.rglob("*")):
    if p.is_dir():
        continue
    name = p.name
    out = dst / p.relative_to(src)
    turn_json = name.endswith(".json") and ("warm-t" in name or "cold-t" in name)
    if name in KEEP or name.endswith(".requests.jsonl") or turn_json:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(scrub(p.read_bytes()))
    elif name in ("server.log", "mem.jsonl") or name.endswith("-server.log"):
        out = out.with_name(name + ".gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wb", 9) as g:
            g.write(scrub(p.read_bytes()))
    elif name.endswith(".log.gz"):
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out, "wb", 9) as g:
            g.write(scrub(gzip.open(p).read()))
total = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
print(f"{dst}: {total / 1e6:.2f} MB")
