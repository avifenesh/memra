#!/usr/bin/env python3
"""Decode-rate receipt rows for the speedup arithmetic: streamed chat, one greedy row
(the instrument) and one vendor-default sampled row (NO sampling params, the traffic
shape we serve), reasoning_effort low, on the A4630 full prompt."""
import json
import time
import urllib.request

EP = "http://127.0.0.1:18402/v1/chat/completions"
P = json.load(open("/root/gpf-ab/prompts.json"))


def row(name, sampled, max_tokens=160):
    body = {
        "model": "zai/glm-5.3-flash",
        "messages": [{"role": "user", "content": P["A4630"]}],
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not sampled:
        body["temperature"] = 0.0
    req = urllib.request.Request(
        EP, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    t0 = time.time()
    first = last = None
    n = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                t = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if t:
                    n += 1
                    last = time.time() - t0
                    if first is None:
                        first = last
    dec = (n - 1) / (last - first) if n > 1 and last > first else None
    out = {
        "row": name,
        "sampled": sampled,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "ttfd_s": round(first, 3) if first else None,
        "decode_tok_s": round(dec, 2) if dec else None,
    }
    print(json.dumps(out))
    return out


rows = [row("A4630-greedy", False), row("A4630-vendor-default-sampled", True)]
json.dump(rows, open("/root/dfp2/decode_rate.json", "w"))
