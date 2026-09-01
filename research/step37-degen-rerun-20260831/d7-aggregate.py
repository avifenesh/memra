#!/usr/bin/env python3
import json, glob, os, collections, statistics
LANE = os.environ.get("LANE", "/home/ubuntu/degen-rerun/lane")
rows = [json.load(open(f)) for f in sorted(glob.glob(LANE + "/gen/*-s*.json"))]
by = collections.defaultdict(list)
for r in rows:
    by[r["arm"]].append(r)
print("%-9s %2s | %-6s %-7s %-9s | repfrac: med max #>=0.25 | t5>t8 | think_med" %
      ("arm", "n", "len", "cont>0", "stop"))
for arm in ["ctrl", "clean", "cleanlow", "ctrllow", "clean4k", "empty", "t1"]:
    rs = by.get(arm, [])
    if not rs: continue
    n = len(rs)
    fl = sum(1 for r in rs if r.get("finish") == "length")
    c0 = sum(1 for r in rs if (r.get("content_chars") or 0) > 0)
    st = sum(1 for r in rs if r.get("finish") == "stop")
    reps = [r.get("repfrac", 0) for r in rs]
    loop = sum(1 for x in reps if x >= 0.25)
    t5gt = sum(1 for r in rs if r.get("t5_keys", 0) > r.get("t8_keys", 0))
    think = [r.get("reasoning_chars", 0) for r in rs]
    print("%-9s %2d | len=%d content>0=%d stop=%d | rep med=%.3f max=%.3f loops=%d | t5>t8=%d | think_med=%d" %
          (arm, n, fl, c0, st, statistics.median(reps), max(reps), loop, t5gt,
           statistics.median(think)))
print()
for arm in ["ctrl", "clean", "cleanlow", "ctrllow", "clean4k", "empty", "t1"]:
    for r in by.get(arm, []):
        print("%s s%d finish=%s r=%5d c=%5d rep=%.3f t5=%d t8=%d %s" %
              (arm.ljust(8), r["sample"], (r.get("finish") or "?").ljust(6),
               r.get("reasoning_chars", 0), r.get("content_chars", 0),
               r.get("repfrac", 0), r.get("t5_keys", 0), r.get("t8_keys", 0),
               "" if r.get("valid") else "INVALID:" + str(r.get("invalid_reason"))))
