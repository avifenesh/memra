#!/usr/bin/env python3
"""WP-B day 42 client (DAY42.md 1.4): the reclaim flush cell, one boot = one arm.

Phases, all `/v1/completions` with `prompt_ids`, greedy:
  1. warm     --warm distinct prompts of --warm-tokens ids, max_tokens=1, one namespace (they seed the prefix cache)
  2. tenants  --tenants streamed requests of 1,024 ids, max_tokens=--tenant-tokens, started together; each records its
              inter-token gaps (the stall the flush imposes), run through the burst
  3. burst    --burst open-output requests of 2,048 distinct ids released together (no max_tokens), streamed so TTFT is
              the wait; status, Retry-After and TTFT recorded
  4. warmth   the warm prompts again with max_tokens=32, each followed by its cold twin in a fresh namespace

Per row (client.jsonl): tag, phase, status, submit/done ms, ttft_ms, gaps_ms (tenants), completion_tokens,
cached_tokens, retry_after, content_sha256, error. marks.json: each phase's start and end ms.
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--n", default=None, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--order", default=None, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--warm", type=int, default=8)
ap.add_argument("--warm-tokens", type=int, default=8192)
ap.add_argument("--tenants", type=int, default=4)
ap.add_argument("--tenant-tokens", type=int, default=2048)
ap.add_argument("--burst", type=int, default=32)
ap.add_argument("--timeout-s", type=int, default=3600)
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, "client.jsonl")
open(out_path, "w").close()
lock = threading.Lock()
marks = {}


def now():
    return time.time() * 1000.0


def post_json(path, body):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
        return json.load(r)


text = open(a.serving_md, encoding="utf-8").read()
stream = []
need = a.warm * a.warm_tokens + (a.tenants + 1) * 1024 + (a.burst + 1) * 2048 + 4096
while len(stream) < need and len(stream) < 64 * 400000:
    stream += post_json("/v1/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})["tokens"]
if len(stream) < need:
    sys.exit(f"tokenized stream {len(stream)} shorter than {need}")
cursor = [0]


def take(n):
    s = stream[cursor[0]:cursor[0] + n]
    cursor[0] += n
    return s


def call(tag, phase, ids, max_tokens, salt, record_gaps=False):
    body = {"model": a.model, "prompt_ids": ids, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": salt}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    submit = now()
    first = last = None
    gaps, parts = [], []
    usage, status, err, retry = None, None, None, None
    try:
        req = urllib.request.Request(a.base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout_s) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                j = json.loads(payload)
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    t = ch.get("text") or ""
                    if t:
                        ts = now()
                        if first is None:
                            first = ts
                        elif record_gaps:
                            gaps.append(ts - last)
                        last = ts
                        parts.append(t)
    except urllib.error.HTTPError as e:
        status, err, retry = e.code, e.read().decode("utf-8", "replace")[:400], e.headers.get("Retry-After")
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        status, err = -1, repr(e)[:400]
    done = now()
    content = "".join(parts)
    row = dict(tag=tag, phase=phase, status=status, submit_ms=submit, done_ms=done,
               ttft_ms=None if first is None else first - submit, gaps_ms=gaps if record_gaps else None,
               completion_tokens=(usage or {}).get("completion_tokens"),
               cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               retry_after=retry, error=err,
               content_sha256=hashlib.sha256(content.encode()).hexdigest() if status == 200 else None)
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{tag} status={status} G={row['completion_tokens']} cached={row['cached_tokens']} ttft={row['ttft_ms']}", flush=True)


# 1. warm
marks["warm_start_ms"] = now()
warm = [take(a.warm_tokens) for _ in range(a.warm)]
for i, ids in enumerate(warm):
    call(f"warm-{i}", "warm", ids, 1, "warm")
marks["warm_end_ms"] = now()
# 2. tenants, started together, then 3. the burst while they run
tenants = [threading.Thread(target=call, args=(f"tenant-{i}", "tenant", take(1024), a.tenant_tokens, f"tenant-{i}", True))
           for i in range(a.tenants)]
marks["tenants_start_ms"] = now()
for th in tenants:
    th.start()
time.sleep(3.0)
bursts = [threading.Thread(target=call, args=(f"burst-{i}", "burst", take(2048), None, f"burst-{i}"))
          for i in range(a.burst)]
marks["burst_start_ms"] = now()
for th in bursts:
    th.start()
for th in bursts:
    th.join()
marks["burst_end_ms"] = now()
for th in tenants:
    th.join()
marks["tenants_end_ms"] = now()
# 4. warmth
marks["warmth_start_ms"] = now()
for i, ids in enumerate(warm):
    call(f"warmth-{i}", "warmth", ids, 32, "warm")
    call(f"warmth-{i}-cold", "warmth", ids, 32, f"warm-cold-{i}")
marks["warmth_end_ms"] = now()
with open(os.path.join(a.out, "marks.json"), "w") as f:
    json.dump(marks, f)
rows = [json.loads(l) for l in open(out_path)]
print(f"DAY42 CLIENT rows={len(rows)} ok={sum(1 for r in rows if r['status'] == 200)}")
