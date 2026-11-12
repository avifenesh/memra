#!/usr/bin/env python3
"""specpub_ab.py OUT.jsonl --set {131k|260k} [--tag ARM]: one served cell of the spec boundary
publication B/T (GB10 decode round 6, lever 5). Requests on a fresh server, greedy, 512 max tokens
(every answer ends on EOS): g1k, then the set's cold request (g131k: 131,072 tokens, g260k:
260,000), each with its own nonce line, so neither hits the other's prefix entry. The document,
needles, question and sizes are pf_long_memra.py's (research/gb10-prefill-20261106/harness/r4/),
the stream reader is eos_ab.py's (research/gb10-decode-20261109/tools/); the prompts are the deep
split rung's served cells', token for token.
Each row: TTFT, total, completion tokens, finish and the text sha256 (the arms' identity check).
BENCH_BASE / BENCH_MODEL as pf_long_memra.py."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402
from eos_ab import stream  # noqa: E402

DEPTHS = {"28k": 28000, "60k": 60000, "100k": 100000, "131k": 131072, "260k": 260000}


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
        put(f"g{a.set}", stream(doc(DEPTHS[a.set], f"memra-rows nonce g{a.set}"), 512, greedy))


if __name__ == "__main__":
    main()
