#!/usr/bin/env python3
"""short_req.py OUT.jsonl TAG [TOKENS]: memra#1221's truly-short cell request: specpub_ab.py's
request of TOKENS prompt tokens (default 1024: g1k; 28000: g28k), the same document, nonce and
greedy 512-token answer, so its text compares with the other cells' row of that size.
BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402
from eos_ab import stream  # noqa: E402

n = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
name = {1024: "g1k", 28000: "g28k", 131072: "g131k"}[n]
row = stream([{"role": "user", "content": pf.doc(pf.lines_for(n, "fairness-bench-v1"), f"memra-rows nonce {name}")}],
             512, {"temperature": 0.0})
row.update({"tag": sys.argv[2], "req": name})
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(row) + "\n")
