#!/usr/bin/env python3
"""DAY44 boot report (run-day26-cell.sh's PARSER): the boot's own counts for the run log. Verdicts are day44-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
resumes = len(re.findall(r"\[kv-reuse\] exact: \w+ resume from", log))
settles = len(re.findall(r"\[kv-reuse\] exact: settle \d+", log))
declined = len(re.findall(r"\[kv-reuse\] exact: declined", log))
door = len(re.findall(r"MEMRA_RESUME_EXACT=1", log))
oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", log))
print(f"DAY44 BOOT rows={len(rows)} ok={sum(r['status'] == 200 for r in rows)} exact_resumes={resumes} "
      f"settles={settles} declined={declined} door_line={door} oom={oom}")
