#!/usr/bin/env python3
"""One greedy row (for PRIME_PROF lines / engagement), then N vendor-default sampled
max_tokens=1 requests on B5550: the first-token distribution is the near-tie proxy the
missing logprobs surface would otherwise answer. usage: tie-probe.py <tag> [n]"""
import json, sys, time, urllib.request

TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
EP = "http://127.0.0.1:18402/v1/chat/completions"
P = json.load(open("/root/gpf-ab/prompts.json"))


def one(body):
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=200) as r:
        j = json.loads(r.read())
    ch = j["choices"][0]
    msg = ch.get("message") or {}
    tok = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    return tok, round(time.time() - t0, 2)


# greedy anchor (also produces the PRIME_PROF phase lines on a prof boot)
tok, el = one({"model": "zai/glm-5.3-flash",
               "messages": [{"role": "user", "content": P["C6470"]}],
               "max_tokens": 1, "temperature": 0.0, "reasoning_effort": "low"})
print(f"{TAG} C6470 greedy first={tok!r} elapsed={el}s", flush=True)

counts = {}
for i in range(N):
    tok, el = one({"model": "zai/glm-5.3-flash",
                   "messages": [{"role": "user", "content": P["B5550"]}],
                   "max_tokens": 1, "reasoning_effort": "low"})
    counts[tok] = counts.get(tok, 0) + 1
    print(f"{TAG} B5550 sampled[{i}] first={tok!r} elapsed={el}s", flush=True)
print(f"{TAG} B5550 sampled first-token distribution over {N}: {counts}", flush=True)
