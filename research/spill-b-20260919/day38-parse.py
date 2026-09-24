#!/usr/bin/env python3
"""DAY38 boot report (run-day26-cell.sh's PARSER): the boot's own counts, for the run log. The verdicts are day38-read.py's."""
import json, os, re, sys

c = sys.argv[1]
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
rows = [json.loads(l) for l in open(os.path.join(c, "client.jsonl"))] if os.path.exists(os.path.join(c, "client.jsonl")) else []
print(f"DAY38 BOOT rows={len(rows)} ok={sum(r['status'] == 200 for r in rows)} "
      f"park_compact={len(re.findall(r'\\[kv-reuse\\] park-compact: ', log))} "
      f"grow={len(re.findall(r'\\[kv-reuse\\] park-compact grow: ', log))} "
      f"failed={len(re.findall(r'park-compact (?:grow )?failed', log))} "
      f"affinity_rewound={len(re.findall(r'plain-affinity: rewound to', log))} "
      f"step_oom_parked={len(re.findall(r'step OOM parked session back to queue', log))} "
      f"door_park={len(re.findall(r'MEMRA_KV_PARK_COMPACT', log))} kv_vmm_door_on={len(re.findall(r'\\[kv-vmm\\] door=ON', log))}")
