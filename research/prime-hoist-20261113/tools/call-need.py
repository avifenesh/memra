#!/usr/bin/env python3
"""Per-call need of the last (placed) prime in each boot's serve.log: pool reserved after the call
minus pool used before it, MiB (the record's definition, tools/b-spread.py). usage: call-need.py DIR..."""
import re, sys
R = re.compile(r"\[diag1255\] dspark-prime-(chunk-before|chunk-after|done) id=(\S+): driver_free=(\d+)MB pool_res=(\d+)MB pool_used=(\d+)MB")
for d in sys.argv[1:]:
    ev = [(m.group(1), m.group(2), int(m.group(4)), int(m.group(5))) for m in map(R.search, open(d + "/serve.log", errors="replace")) if m]
    last = ev[-1][1]
    ev = [e for e in ev if e[1] == last]
    needs = []
    before = None
    for kind, _, res, used in ev:
        if kind == "chunk-before":
            before = used
        elif before is not None:
            needs.append(res - before)
            before = None
    print(d.split("/")[-2].replace("cap-", "").replace("-first-8k-m100", ""), d.split("/")[-1], "calls", len(needs), "need", needs, "ended" if ev[-1][0] == "done" else "NOT DONE")
