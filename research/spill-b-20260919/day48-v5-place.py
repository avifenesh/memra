#!/usr/bin/env python3
"""DAY48 2.2: places V5's predictive lines that no physical line follows (read after the nineteenth sitting).

For each enforce-vg boot: every [admit-predict] line with vg_debt above 0 is classed by what follows it before the
next predictive line. Paired: a physical `[admission] dspark verify-graph pool debt` line follows. Unpaired reject: the
verdict is a reject, so no physical gate runs. Unpaired admit: an admission with no physical line. For each unpaired
admission it also checks whether a `[admission] request cost` line sits between the previous predictive line and this
one. The server prints both lines only under `log_estimate`, which is false when the admission's (cap, spec, cost) key
equals the last logged one.
usage: day48-v5-place.py <receipt root>   (reads <root>/boots/*-enforce-vg/server.log)
"""
import os
import re
import sys

root = sys.argv[1]
PRED = re.compile(r"\[admit-predict\] id=(\S+) .*?verdict=(\S+).*?vg_debt=(\d+)")
boots = sorted(d for d in os.listdir(os.path.join(root, "boots"))
               if d.endswith("-enforce-vg") and os.path.isdir(os.path.join(root, "boots", d)))
for b in boots:
    ev = []
    for line in open(os.path.join(root, "boots", b, "server.log"), errors="replace"):
        m = PRED.search(line)
        if m:
            ev.append(("P", m.group(1), m.group(2), int(m.group(3))))
        elif "[admission] dspark verify-graph pool debt: +" in line:
            ev.append(("D",))
        elif "[admission] request cost:" in line:
            ev.append(("C",))
    paired, rejects, admits, admits_with_cost = 0, 0, [], 0
    for k, e in enumerate(ev):
        if e[0] != "P" or e[3] == 0:
            continue
        j = k + 1
        while j < len(ev) and ev[j][0] != "P":
            j += 1
        if any(x[0] == "D" for x in ev[k + 1:j]):
            paired += 1
            continue
        if e[2].startswith("reject"):
            rejects += 1
            continue
        i = k - 1
        while i >= 0 and ev[i][0] != "P":
            i -= 1
        has_cost = any(x[0] == "C" for x in ev[i + 1:k])
        admits_with_cost += has_cost
        admits.append(e[1])
    print(f"DAY48 V5-PLACE boot={b} predictive_vg_debt_lines={paired + rejects + len(admits)} paired={paired} "
          f"unpaired_reject={rejects} unpaired_admit={len(admits)} unpaired_admit_with_request_cost_line="
          f"{admits_with_cost} unpaired_admit_ids={admits}")
