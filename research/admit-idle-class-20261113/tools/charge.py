#!/usr/bin/env python3
"""charge.py SERVER_LOG MARGIN_MB [--explain]: the MemAvailable target (MiB) that leaves an idle
server just above one request's admission charge. The charge is read from the server's own lines
for the log's last admitted request: `[admission] request cost: ... = N MB`, the
`[admission] per-session draft-state charge: ... +N MB` before it when the route has one, and the
transient reserve of `[admit-cal] boot calibration done: ... transient floor N MB` (MiB there).
Prints the target, or with --explain the terms."""
import re
import sys

log = open(sys.argv[1], errors="replace").read()
margin = int(sys.argv[2])
cost = re.findall(r"\[admission\] request cost: .*? = (\d+)MB", log)
draft = re.findall(r"\[admission\] per-session draft-state charge: .*?\+(\d+)MB", log)
floor = re.findall(r"\[admit-cal\] boot calibration done: .*?transient floor (\d+)MB", log)
if not cost or not floor:
    sys.exit("charge.py: no request cost or transient floor in the log")
cost_mb, draft_mb = int(cost[-1]), int(draft[-1]) if draft else 0
reserve_mb = int(floor[-1]) * 1048576 // 1000000
required = cost_mb + draft_mb + reserve_mb
target = (required + margin) * 1000000 // 1048576
if "--explain" in sys.argv:
    print(f"cost {cost_mb} MB + draft state {draft_mb} MB + reserve {reserve_mb} MB = {required} MB; "
          f"+ margin {margin} MB = {target} MiB")
else:
    print(target)
