#!/usr/bin/env python3
"""DAY48 boot report (run-day26-cell.sh's PARSER): the boot's own counts for the run log. Verdicts are day48-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
print(f"DAY48 BOOT rows={len(rows)} ok={sum(r['status'] == 200 and not r['error'] for r in rows)} "
      f"r429={sum(r['status'] == 429 for r in rows)} rejects={len(re.findall(r'verdict=reject-kv', log))} "
      f"vg_debt_lines={len(re.findall(r'vg_debt=[1-9]', log))} pool_engaged={len(re.findall(r'verify-graph pool ENGAGED', log))} oom={len(re.findall(r'CUDA_ERROR_OUT_OF_MEMORY', log))}")
