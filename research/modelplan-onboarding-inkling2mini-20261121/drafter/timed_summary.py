#!/usr/bin/env python3
"""Lane S: decode tok/s with and without drafting per K from ink2-spec --check-plain runs.
Decode time excludes prefill; tok/s counts the tokens after the first. Sums over prompts."""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    agg = defaultdict(lambda: [0.0, 0, 0, 0, 0])  # (kind, k) -> seconds, tokens, steps, accepted, equal
    for r in rows:
        k = 0 if r["mode"] == "plain" else r["k"]
        for kind in (r["kind"], "all"):
            a = agg[(kind, k)]
            a[0] += r["decode_s"]
            a[1] += r["generated"] - 1
            if k:
                a[2] += r["steps"]; a[3] += r["accepted"]; a[4] += 1 if r.get("equal_plain") else 0
    print(path)
    print("kind\tK\tdecode tok/s\tspeedup vs plain\taccepted/verify\tms per verify\tequal plain")
    for kind in ["chat", "code", "agentic", "all"]:
        base = agg[(kind, 0)][1] / agg[(kind, 0)][0]
        for k in range(0, 8):
            if (kind, k) not in agg:
                continue
            s, t, st, acc, eq = agg[(kind, k)]
            tps = t / s
            extra = f"{acc / st:.3f}\t{1000 * s / st:.1f}\t{eq}" if k else "\t\t"
            print(f"{kind}\t{k}\t{tps:.1f}\t{tps / base:.3f}\t{extra}")
