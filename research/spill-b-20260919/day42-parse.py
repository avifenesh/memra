#!/usr/bin/env python3
"""DAY42 boot report (run-day26-cell.sh's PARSER): the boot's own counts for the run log. Verdicts are day42-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
print(f"DAY42 BOOT rows={len(rows)} ok={sum(r['status'] == 200 for r in rows)} "
      f"r429={sum(r['status'] == 429 for r in rows)} "
      f"ontick_demoted={len(re.findall(r'\\[admit-mem\\] reclaim demoted', log))} "
      f"plans={len(re.findall(r'reclaim off-tick: plan', log))} "
      f"submitted={len(re.findall(r'reclaim off-tick: submitted', log))} "
      f"oom={len(re.findall(r'CUDA_ERROR_OUT_OF_MEMORY', log))}")
