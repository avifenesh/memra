#!/usr/bin/env python3
"""Arithmetic replay of a labeled synthetic native-ID fixture, not SSD timing."""
import argparse
import hashlib
import json
from pathlib import Path

lane = Path(__file__).resolve().parent
fixture = lane / "fixtures/ple-ngram-synthetic.json"
trace = json.loads(fixture.read_text())
assert trace["synthetic"] is True
rows = []
for granularity in (512, 4096, 16384):
    for shape, stride in (("packed", 264), ("sparse", 32768)):
        for step in trace["steps"]:
            # Native ple_block consumes only last chunk, preserving token/head order.
            logical_ids = step["ids"][-step["chunk"] * len(trace["sizes"]):]
            ids = sorted(set(logical_ids))
            spans = []
            straddles = 0
            for row in ids:
                start = row * stride
                end = start + 264
                lo = start // granularity * granularity
                hi = (end + granularity - 1) // granularity * granularity
                straddles += hi - lo > granularity
                if spans and lo <= spans[-1][1]:
                    spans[-1][1] = max(spans[-1][1], hi)
                else:
                    spans.append([lo, hi])
            io = sum(hi-lo for lo, hi in spans)
            useful = 264 * len(ids)
            rows.append({"synthetic": True, "shape": shape, "step": step["label"],
                "granularity": granularity, "logical_rows": len(logical_ids),
                "unique_rows": len(ids), "useful_bytes": useful, "requested_bytes": io,
                "straddles": straddles, "extents": len(spans), "amplification": io/useful})
result = {"fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
          "measurement": "deterministic aligned-request arithmetic; no physical IO or timings", "rows": rows}
encoded = json.dumps(result, indent=2) + "\n"
output = lane / "fixtures/ple-trace-policy.json"
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
if args.check:
    assert output.read_text() == encoded, "trace-policy fixture drift"
else:
    output.write_text(encoded)
for shape in ("packed", "sparse"):
    for granularity in (512, 4096, 16384):
        selected = [r for r in rows if r["shape"] == shape and r["granularity"] == granularity]
        useful = sum(r["useful_bytes"] for r in selected)
        requested = sum(r["requested_bytes"] for r in selected)
        print(f"SYNTHETIC {shape} granularity={granularity} useful={useful} requested={requested} amplification={requested/useful:.6f} straddles={sum(r['straddles'] for r in selected)}")
