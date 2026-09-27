#!/usr/bin/env python3
"""DAY37 stream boot report: the client's SUMMARY line plus the boot's own facts (run-day26-cell.sh's PARSER)."""
import json, os, re, sys

c = sys.argv[1]
summary = open(os.path.join(c, "SUMMARY.txt")).read().strip() if os.path.exists(os.path.join(c, "SUMMARY.txt")) else "no SUMMARY"
log = open(os.path.join(c, "server.log"), errors="replace").read() if os.path.exists(os.path.join(c, "server.log")) else ""
door = re.findall(r"\[kv-vmm\] door=\S+", log)
grows = len(re.findall(r"\[kv-vmm\] grow ", log))
waited = len(re.findall(r"\[kv-vmm\] grow .* waited=1", log))
fails = len(re.findall(r"\[kv-vmm\] (?:grow failed|ensure failed)", log))
oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY|out of memory", log))
print(summary)
print(f"DAY37 STREAM-BOOT door={door[0] if door else 'absent'} owner_grows={grows} waited={waited} grow_failures={fails} oom_lines={oom}")
