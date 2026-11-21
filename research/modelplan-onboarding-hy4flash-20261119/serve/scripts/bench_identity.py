"""Two bench.py outputs of one binary, drafts off and on: every request generated the same text.

usage: python3 bench_identity.py OFF_JSONL ON_JSONL [OUT_JSON]

bench.py seeds every sampled request, so a drafting server must generate what the drafts-off
server generates. Requests are matched by workload, run and index; their token counts and the
sha256 of their text must be equal, and the runs must be sampled. Exits 1 on any difference.
"""

import json
import sys


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out = {}
    for row in rows:
        for i, r in enumerate(row["requests"]):
            out[(row["workload"], row["run"], i)] = (r["tokens"], r["seed"], r["text_sha256"])
    sampled = bool(rows) and all((row.get("sampling") or {}).get("temperature", 0) > 0 for row in rows)
    return out, sampled


off, off_sampled = load(sys.argv[1])
on, on_sampled = load(sys.argv[2])
differ = sorted(str(k) for k in set(off) | set(on) if off.get(k) != on.get(k))
ok = bool(off) and not differ and off_sampled and on_sampled
result = {"pass": ok, "requests_compared": len(off), "sampled": off_sampled and on_sampled,
          "differences": differ[:50]}
if len(sys.argv) > 3:
    json.dump(result, open(sys.argv[3], "w"), indent=1)
print(json.dumps(result))
print("BENCH-IDENTITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
