#!/usr/bin/env python3
"""DAY41 boot report (run-day26-cell.sh's PARSER): the boot's own counts for the run log. Verdicts are day41-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
print(f"DAY41 BOOT rows={len(rows)} ok={sum(r['status'] == 200 for r in rows)} "
      f"grid_rewind={len(re.findall(r'\\[kv-reuse\\] grid-rewind: \\w+ extension', log))} "
      f"grid_declined={len(re.findall(r'\\[kv-reuse\\] grid-rewind: \\w+ declined', log))} "
      f"door_line={len(re.findall(r'MEMRA_RESUME_GRID_REWIND=1', log))} "
      f"oom={len(re.findall(r'CUDA_ERROR_OUT_OF_MEMORY', log))}")
