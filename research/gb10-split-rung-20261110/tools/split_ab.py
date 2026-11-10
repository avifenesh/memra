#!/usr/bin/env python3
"""split_ab.py OUT.jsonl --set {12k|32k|131k|260k} [--tag ARM]: one served cell for the deep split
rung's B/T (GB10 decode round 6, lever 1; research/gb10-verify-rows-20261109/tools/rows_ab.py with
more depths). Requests on a fresh server (the pf_long_memra.py document and question), greedy, 512
max tokens (every answer ends on EOS): g1k, then the set's request (g12k: 12,288 tokens, g32k:
32,768, g70k: 69,632, g131k: 131,072, g260k: 260,000).
Each row: TTFT, total, completion tokens, finish and the text sha256 (the arms' identity check).
BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402
from eos_ab import stream  # noqa: E402

DEPTHS = {"12k": 12288, "32k": 32768, "70k": 69632, "131k": 131072, "260k": 260000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--set", default="131k", choices=sorted(DEPTHS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    greedy = {"temperature": 0.0}
    with open(a.out, "a") as f:
        def put(name, row):
            row.update({"tag": a.tag, "req": name})
            f.write(json.dumps(row) + "\n"); f.flush()

        def doc(target, nonce):
            return [{"role": "user", "content": pf.doc(pf.lines_for(target, "fairness-bench-v1"), nonce)}]
        put("g1k", stream(doc(1024, "memra-rows nonce g1k"), 512, greedy))
        nonce = "memra-rows" if a.set in ("131k", "260k") else "memra-split"
        put(f"g{a.set}", stream(doc(DEPTHS[a.set], f"{nonce} nonce g{a.set}"), 512, greedy))

if __name__ == "__main__":
    main()
