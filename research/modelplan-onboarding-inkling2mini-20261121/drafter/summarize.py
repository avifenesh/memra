#!/usr/bin/env python3
"""Lane S: acceptance per kind from ink2-spec JSON lines. usage: summarize.py FILE.jsonl..."""
import json
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path) if l.strip() and '"mode"' in l]
    spec = [r for r in rows if r["mode"] != "plain"]
    by = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in spec:
        for key in (r["kind"], "all"):
            b = by[(key, r["k"])]
            b[0] += r["steps"]; b[1] += r["accepted"]; b[2] += r["generated"]; b[3] += 1
            b[4] += 1 if r.get("equal_plain") else 0
    print(path)
    print("kind\tk\tprompts\tverify steps\taccepted\tacceptance length\ttokens per verify\tequal plain")
    for (kind, k), (steps, acc, gen, n, eq) in sorted(by.items()):
        print(f"{kind}\t{k}\t{n}\t{steps}\t{acc}\t{acc / steps:.3f}\t{(acc + steps) / steps:.3f}\t{eq}/{n}")
