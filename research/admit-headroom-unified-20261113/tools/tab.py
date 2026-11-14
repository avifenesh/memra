#!/usr/bin/env python3
"""memra#1257 GB10 cells: one row per cell from a local copy of ~/.cache/memra-1221/runs/CAMPAIGN.
Columns: arm, MemFree after the fill, the rung's [admit-trim] lines (count, the reading and the
raw/allocatable value of the first, bytes released), the pre-capture guard's lines, CUDA OOM lines,
the lowest MemAvailable, and per request: TTFT, completion tokens, finish, error and text sha.
usage: tab.py RUNS_DIR [CAMPAIGN...]"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
camps = sys.argv[2:] or ["h1257", "h1257s"]
for camp in camps:
    for cell in sorted((root / camp).glob("[0-9][0-9]-*")):
        srv = cell / "server.log"
        text = srv.read_text(errors="replace") if srv.exists() else ""
        run = (cell / "run.log").read_text(errors="replace") if (cell / "run.log").exists() else ""
        trims = re.findall(r"\[admit-trim\] dev\d+ driver_free_bytes=(\d+) floor_bytes=\d+ reclaimed_bytes=(\d+).*?(?:reading=(\w+) )?before (\w+)", text)
        cap = re.findall(r"\[spec\] (pre-capture pool trim[^\n]*|capture refused[^\n]*|draft-graph capture[^\n]*refus[^\n]*)", text)
        ooms = len(re.findall(r"OUT_OF_MEMORY|out of memory", text))
        fill = re.findall(r"fill2? done: MemFree: (\d+) kB", run)
        mem = [json.loads(line) for line in (cell / "mem.jsonl").read_text().splitlines() if line.strip()] if (cell / "mem.jsonl").exists() else []
        low = min((m["mem_available_kb"] for m in mem), default=0) // 1024
        rows = []
        if (cell / "ab.jsonl").exists():
            for line in (cell / "ab.jsonl").read_text().splitlines():
                if line.startswith("{"):
                    r = json.loads(line)
                    rows.append(f"{r.get('req')}: ttft {r.get('ttft_s')} s, {r.get('completion_tokens')} tok, {r.get('finish')}, "
                                f"error {r.get('error')}, sha {str(r.get('text_sha256'))[:12]}")
        t0 = f"first {int(trims[0][0]) >> 20} MiB ({trims[0][2] or 'raw'}), released {sum(int(t[1]) for t in trims) >> 20} MiB in all" if trims else ""
        print(f"{camp}/{cell.name}: fill MemFree {int(fill[0]) >> 10 if fill else '-'} MB; admit-trim {len(trims)} {t0}; "
              f"capture lines {len(cap)}; OOM lines {ooms}; lowest MemAvailable {low} MB")
        for c in cap[:3]:
            print(f"    [spec] {c[:200]}")
        for r in rows:
            print(f"    {r}")
