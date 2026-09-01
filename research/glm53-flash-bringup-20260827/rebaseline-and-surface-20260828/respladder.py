#!/usr/bin/env python3
"""/v1/responses GREEDY effort ladder: does an OMITTING client get Max, as it does on chat?

The chat-surface ladder (followup.py Q2) proved omitted is byte-identical to max, matching the
vendor card. Battery case 07b then implied the OPPOSITE on this surface: a no-effort request
full-hit (cached_tokens 176 of 176) the prefix entry seeded by an effort-low request, and a
whole-entry full hit means byte-identical prompt. If both held, the same omitting client would
get Max on /v1/chat/completions and Low on /v1/responses, which the standard-surface law forbids.
Cache counters INFER; greedy output shas MEASURE. This settles it.
"""
import hashlib, json, urllib.request

EP = "http://127.0.0.1:18400"
M = "zai/glm-5.3-flash"
PROMPT = "In one sentence: why is the sky blue?"


def text_of(j):
    out = []
    for it in j.get("output") or []:
        for c in it.get("content") or []:
            out.append(c.get("text", "") or "")
        for c in it.get("summary") or []:
            out.append(c.get("text", "") or "")
    return "".join(out)


shas = {}
for lvl in [None, "low", "high", "max"]:
    body = {"model": M, "input": PROMPT, "max_output_tokens": 220, "temperature": 0.0}
    if lvl:
        body["reasoning"] = {"effort": lvl}
    req = urllib.request.Request(EP + "/v1/responses", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            j = json.loads(r.read().decode())
    except Exception as e:
        print(f"  effort={lvl}: ERROR {type(e).__name__}: {e}")
        continue
    txt = text_of(j)
    u = j.get("usage") or {}
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
    sha = hashlib.sha256(txt.encode()).hexdigest()[:16]
    shas[lvl or "omitted"] = sha
    print(f"  effort={str(lvl):>8}: sha={sha} in_tok={u.get('input_tokens')} "
          f"out_tok={u.get('output_tokens')} cached={cached} chars={len(txt)}")

print()
print("  shas:", json.dumps(shas))
print("  omitted == max ?", shas.get("omitted") == shas.get("max"),
      "  (chat surface measured TRUE)")
print("  omitted == low ?", shas.get("omitted") == shas.get("low"))
print("  low == high  ?", shas.get("low") == shas.get("high"))
