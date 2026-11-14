#!/usr/bin/env python3
"""idle_req.py OUT.jsonl TAG [TOKENS]: memra#1304's request. short_req.py's request (the same
document, nonce and greedy 512-token answer, TOKENS 1024 = g1k, 8000 = g8k, 28000 = g28k, 131072 = g131k), so its text compares
with memra#1221 and memra#1257's rows of that size; on an HTTP refusal the row also keeps the
status, the retry headers and the error body. BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eos_ab  # noqa: E402
import pf_long_memra as pf  # noqa: E402

refusal = {}
_urlopen = urllib.request.urlopen


def urlopen(*args, **kwargs):
    try:
        return _urlopen(*args, **kwargs)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        refusal.update({
            "status": e.code,
            "retry_after": e.headers.get("Retry-After"),
            "retry_after_ms": e.headers.get("retry-after-ms"),
            "x_should_retry": e.headers.get("x-should-retry"),
            "body": body,
        })
        raise


urllib.request.urlopen = urlopen

n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
name = {1024: "g1k", 8000: "g8k", 28000: "g28k", 131072: "g131k"}[n]
row = eos_ab.stream(
    [{"role": "user", "content": pf.doc(pf.lines_for(n, "fairness-bench-v1"), f"memra-rows nonce {name}")}],
    512, {"temperature": 0.0})
row.update({"tag": sys.argv[2], "req": name, "status": refusal.get("status", 200), "refusal": refusal or None})
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps({k: row[k] for k in ("tag", "req", "status", "ttft_s", "total_s", "completion_tokens",
                                      "finish", "text_sha256")} | {"refusal": row["refusal"]}))
