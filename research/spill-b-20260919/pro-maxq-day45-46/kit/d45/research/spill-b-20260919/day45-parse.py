#!/usr/bin/env python3
"""DAY45 boot report (run-day26-cell.sh's PARSER): the boot's own counts for the run log. Verdicts are day45-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
print(f"DAY45 BOOT rows={len(rows)} ok={sum(r['status'] == 200 for r in rows)} "
      f"w_booked={len(re.findall(r'\[admit-book\] w-booked', log))} w_release={len(re.findall(r'\[admit-book\] w-release', log))} "
      f"predict_lines={len(re.findall(r'\[admit-predict\]', log))} oom={len(re.findall(r'CUDA_ERROR_OUT_OF_MEMORY', log))}")
