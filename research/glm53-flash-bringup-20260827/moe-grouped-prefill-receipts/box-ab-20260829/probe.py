#!/usr/bin/env python3
"""Grouped-prefill A/B battery, one boot = one call.
usage: probe.py <arm-tag> <outfile.json>

Per boot: warmup (short real prompt), then 3 greedy streamed rows at the real ~4.6k/5.5k/6.5k
prompts (max_tokens=32, temperature=0, reasoning_effort low), then the vendor-default sampled
twin (NO sampling params) on the 4.6k prompt. TTFD = first generated-token delta (content or
reasoning channel); first_line_s = first SSE data line (the banked ctxprobe method) is recorded
alongside for comparability. prefill tok/s = usage.prompt_tokens / ttfd. The first greedy token
text per prompt is the argmax-gate row."""
import json, sys, time, urllib.request, urllib.error, hashlib

TAG, OUT = sys.argv[1], sys.argv[2]
EP = "http://127.0.0.1:18402/v1/chat/completions"
MODEL = "zai/glm-5.3-flash"
P = json.load(open("/root/gpf-ab/prompts.json"))


def stream_row(name, prompt, max_tokens, sampled):
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "reasoning_effort": "low",
            "stream": True, "stream_options": {"include_usage": True}}
    if not sampled:
        body["temperature"] = 0.0
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    first_line = None
    first_tok_t = None
    first_tok_text = None
    last_tok_t = None
    n_deltas = 0
    text = []
    usage = {}
    status = None
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            status = r.status
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if first_line is None:
                    first_line = time.time() - t0
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    tok = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                    if tok:
                        n_deltas += 1
                        last_tok_t = time.time() - t0
                        if first_tok_t is None:
                            first_tok_t = last_tok_t
                            first_tok_text = tok
                        text.append(tok)
    except urllib.error.HTTPError as e:
        status = e.code
        err = e.read().decode()[:200]
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    full = "".join(text)
    pt = usage.get("prompt_tokens")
    row = {"arm": TAG, "row": name, "sampled": sampled, "status": status,
           "prompt_tokens": pt, "completion_tokens": usage.get("completion_tokens"),
           "first_line_s": round(first_line, 4) if first_line else None,
           "ttfd_s": round(first_tok_t, 4) if first_tok_t else None,
           "prefill_tok_s": round(pt / first_tok_t, 2) if (pt and first_tok_t) else None,
           "decode_tok_s": round((n_deltas - 1) / (last_tok_t - first_tok_t), 2)
                           if (n_deltas > 1 and last_tok_t and first_tok_t and last_tok_t > first_tok_t) else None,
           "first_token": first_tok_text,
           "text_sha16": hashlib.sha256(full.encode()).hexdigest()[:16],
           "text_head": full[:60],
           "err": err or ("EMPTY_STREAM: 200 with no tokens and no usage (engine-error in server log)"
                          if (status == 200 and n_deltas == 0 and not usage) else None)}
    print(json.dumps(row), flush=True)
    return row


rows = []
rows.append(stream_row("warmup", P["WARM"], 8, False))
for name in ["A4630", "B5550", "C6470"]:
    rows.append(stream_row(name, P[name], 32, False))
rows.append(stream_row("A4630-sampled", P["A4630"], 64, True))
json.dump(rows, open(OUT, "w"), indent=1)
