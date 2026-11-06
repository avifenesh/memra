#!/usr/bin/env python3
"""GB10 coverage check 1: one streaming /v1/completions request whose prefill runs past 90 s
(a ~150K-token prompt) must get its response headers and its tokens, never an HTTP 408.
usage: longprefill_client.py BASE MODEL TARGET_TOKENS OUT.json"""
import json
import sys
import time
import urllib.error
import urllib.request

base, model, target, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]


def post(path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def para(i, n=60):
    words = ("memory bandwidth decode kernel scheduler prefix cache draft tail verify accept "
             "token session window boundary restore prime chunk tensor stream").split()
    return " ".join(f"{words[(i * 7 + k * 3) % len(words)]}{(i * 31 + k) % 97}" for k in range(n))


def count(text):
    return len(post("/v1/tokenize", {"model": model, "prompt": text})["tokens"])


probe = "\n".join(para(i) for i in range(20))
per_para = count(probe) / 20
n = int(target / per_para) + 1
text = "Reference log.\n" + "\n".join(para(i) for i in range(n)) + "\nQuestion: name one word above.\nAnswer:"
prompt_tokens = count(text)

result = {"target_tokens": target, "prompt_tokens": prompt_tokens, "max_tokens": 16}
body = {"model": model, "prompt": text, "max_tokens": 16, "temperature": 0, "stream": True}
req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    resp = urllib.request.urlopen(req, timeout=1800)
    result["status"] = resp.status
    result["headers_s"] = round(time.time() - t0, 2)
    chunks, first_token_s, text_out, usage = 0, None, "", None
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunks += 1
        v = json.loads(line[5:])
        piece = (v.get("choices") or [{}])[0].get("text") or ""
        if piece and first_token_s is None:
            first_token_s = round(time.time() - t0, 2)
        text_out += piece
        usage = v.get("usage") or usage
    result.update({"first_token_s": first_token_s, "total_s": round(time.time() - t0, 2),
                   "data_chunks": chunks, "text": text_out, "usage": usage})
except urllib.error.HTTPError as e:
    result.update({"status": e.code, "headers_s": round(time.time() - t0, 2),
                   "error_body": e.read().decode(errors="replace")[:500]})
ok = result.get("status") == 200 and result.get("first_token_s") is not None and \
    (result.get("headers_s") or 0) >= 0 and result["prompt_tokens"] >= 0.9 * target
result["prefill_over_90s"] = (result.get("first_token_s") or 0) > 90
result["verdict"] = "PASS" if ok else "FAIL"
json.dump(result, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "text"}))
