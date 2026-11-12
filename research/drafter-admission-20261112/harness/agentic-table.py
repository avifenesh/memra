#!/usr/bin/env python3
"""memra#1173: the agentic replay boots, one row per boot in run order.
usage: agentic-table.py AGENTIC_DIR"""
import json
import sys
from pathlib import Path

print("| boot | turns ok | tokens | wall s | agg gen tok/s | TTFT median s | cost lines path=spec / path=plain | "
      "drafter refused (busy-fits / busy-unfundable / idle-fits / idle-short) | eager-arm admit | VRAM defer lines | "
      "capacity rejects | out of memory |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(Path(sys.argv[1]).iterdir(), key=lambda p: p.name.split("-")[-1]):
    if not (d / "cells.log").exists():
        continue
    log = (d / "serve.log").read_text(errors="replace")
    c = log.count
    s = json.loads((d / "cells.log").read_text().strip().splitlines()[-1])
    print(f"| {d.name} | {s['ok']}/{s['turns']} | {s['tokens']:,} | {s['wall_s']} | {s['aggregate_gen_tps']} | "
          f"{s['ttft_median_s']} | {c('request cost:') - c('path=plain =')} / {c('path=plain =')} | "
          f"{c('drafter refused:')} ({c('arm busy-fits')} / {c('arm busy-unfundable')} / {c('arm idle-fits')} / "
          f"{c('arm idle-short')}) | {c('eager-arm admit')} | {c('VRAM defer')} | {c('capacity reject')} | "
          f"{c('out of memory')} |")
