#!/usr/bin/env python3
"""tok_time.py: time /v1/tokenize on the served prefill documents (pf_long_memra.py's prompts) and
print tokens, seconds and tokens per second. usage: tok_time.py TARGET [TARGET ...]"""
import json, sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
import pf_long_memra as p

for t in map(int, sys.argv[1:]):
    nl = p.lines_for(t, "fairness-bench-v1")
    text = p.doc(nl, f"memra-prefill nonce t{t} r9")
    for rep in range(2):
        t0 = time.time()
        r = p.post("/v1/tokenize", {"model": p.MODEL, "messages": [{"role": "user", "content": text}],
                                    "chat_template_kwargs": p.KW}, timeout=600)
        dt = time.time() - t0
        n = next((int(r[k]) for k in ("count", "num_tokens", "length") if k in r), None) or len(r["tokens"])
        print(json.dumps({"target": t, "rep": rep, "chars": len(text), "tokens": n, "tokenize_s": round(dt, 3),
                          "tok_per_s": round(n / dt)}), flush=True)
