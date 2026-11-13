#!/usr/bin/env python3
"""openout_ab.py OUT.jsonl [--tokens N] [--tag ARM]: memra#1221 part 2 review round 1 (should-fix
7), one served open-output cell on GB10. Requests on a fresh server, greedy, every answer ending on
EOS: g1k (512 max tokens), then two cold open-output requests (no max_tokens: the session's KV
capacity is the served context, 262,144 rows) of N tokens each with their own nonce lines, so
neither hits the other's prefix entry. With MEMRA_REUSE_POOL=0 the first request's session does
not park, so its lent planes move to its entry at retirement (adopted, charged their whole
capacity); the second request's insert is where compaction runs when the cache needs room.
The document and question are pf_long_memra.py's, the stream reader eos_ab.py's (specpub_ab.py).
Each row: TTFT, total, completion tokens, finish and the text sha256."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_long_memra as pf  # noqa: E402
from eos_ab import stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=60000)
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
        put("open-a", stream(doc(a.tokens, "memra-rows nonce open-a"), None, greedy))
        put("open-b", stream(doc(a.tokens, "memra-rows nonce open-b"), None, greedy))


if __name__ == "__main__":
    main()
