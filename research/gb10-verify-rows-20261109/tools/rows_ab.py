#!/usr/bin/env python3
"""rows_ab.py OUT.jsonl --set {1k|28k|131k|260k} [--tag ARM]: one served cell for the verify-rows A/B
(round 5 lever 2). Requests on a fresh server (the pf_long_memra.py document and question), greedy,
512 max tokens (every answer ends on EOS), by set:
  1k     g1k, then s1k (sampled, temperature 0.8, top_p 0.95, seed 1234)
  28k    g1k, then g28k (a 32k-context server)
  131k   g1k, then g131k
  260k   g1k, then g260k
Each row: TTFT, total, completion tokens, finish and the text sha256 (the arms' identity check).
BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402
from eos_ab import stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--set", default="1k", choices=["1k", "28k", "131k", "260k"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    greedy = {"temperature": 0.0}
    sampled = {"temperature": 0.8, "top_p": 0.95, "seed": 1234}
    with open(a.out, "a") as f:
        def put(name, row):
            row.update({"tag": a.tag, "req": name})
            f.write(json.dumps(row) + "\n"); f.flush()

        def doc(target, nonce):
            return [{"role": "user", "content": pf.doc(pf.lines_for(target, "fairness-bench-v1"), nonce)}]
        put("g1k", stream(doc(1024, "memra-rows nonce g1k"), 512, greedy))
        if a.set == "1k":
            put("s1k", stream(doc(1024, "memra-rows nonce s1k"), 512, sampled))
        elif a.set == "28k":
            put("g28k", stream(doc(28000, "memra-rows nonce g28k"), 512, greedy))
        elif a.set == "131k":
            put("g131k", stream(doc(131072, "memra-rows nonce g131k"), 512, greedy))
        else:
            put("g260k", stream(doc(260000, "memra-rows nonce g260k"), 512, greedy))


if __name__ == "__main__":
    main()
