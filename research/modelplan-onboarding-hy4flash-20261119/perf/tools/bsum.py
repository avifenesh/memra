#!/usr/bin/env python3
"""median/min/max of the timed bench runs (warm-up dropped). usage: bsum.py FILE..."""
import json, sys, statistics
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.strip()]
    rows = [r for r in rows if not r["warmup"]]
    def s(k):
        v = [r[k] for r in rows]
        return f"{statistics.median(v):8.2f} [{min(v):.2f}-{max(v):.2f}]"
    ids = {r["ids_fnv1a"] for r in rows}
    r0 = rows[0]
    mode = r0.get("decoding", "greedy")
    if mode == "sampled":
        mode = f"sampled T={r0['temperature']} top_k={r0['top_k']} top_p={r0['top_p']} seed={r0['seed']}"
    print(f"{f.split('/')[-1]:22s} N={len(rows)} prefill tok/s {s('prefill_tok_s')} decode tok/s {s('decode_tok_s')} step p50 ms {s('decode_step_p50_ms')} ids {','.join(ids)} {mode}")
