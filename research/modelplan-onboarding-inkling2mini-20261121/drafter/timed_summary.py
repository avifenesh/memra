#!/usr/bin/env python3
"""Lane S: decode tok/s with and without drafting per K from ink2-spec --check-plain runs.
Decode time excludes prefill; tok/s counts the tokens after the first. Sums over prompts.
K "plain" is decode without drafting; K "adapt" is the adaptive length (--ks 0)."""
import json
import sys
from collections import defaultdict

ORDER = ["plain", "adapt"] + [str(k) for k in range(1, 8)]
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    agg = defaultdict(lambda: [0.0, 0, 0, 0, 0, 0])
    for r in rows:
        key = "plain" if r["mode"] == "plain" else ("adapt" if r["k"] == 0 else str(r["k"]))
        for kind in (r["kind"], "all"):
            a = agg[(kind, key)]
            a[0] += r["decode_s"]
            a[1] += r["generated"] - 1
            a[5] += 1
            if key != "plain":
                a[2] += r["steps"]; a[3] += r["accepted"]; a[4] += 1 if r.get("equal_plain") else 0
    print(path)
    print("kind\tK\tdecode tok/s\tvs plain\taccepted/verify\tms per verify\tequal plain")
    for kind in ["chat", "code", "agentic", "all"]:
        if (kind, "plain") not in agg:
            continue
        base = agg[(kind, "plain")][1] / agg[(kind, "plain")][0]
        for key in ORDER:
            if (kind, key) not in agg:
                continue
            s, t, st, acc, eq, n = agg[(kind, key)]
            tps = t / s
            extra = f"{acc / st:.3f}\t{1000 * s / st:.1f}\t{eq}/{n}" if key != "plain" else "\t\t"
            print(f"{kind}\t{key}\t{tps:.1f}\t{tps / base:.3f}\t{extra}")
