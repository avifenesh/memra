#!/usr/bin/env python3
"""WP-B day 45 client (DAY45.md 1.4): the W-release cell, one boot = one arm.

4 warm requests of --warm-tokens ids, sequential; then --burst requests of --length ids each (distinct windows of the
tokenized docs/SERVING.md stream), released together, max_tokens --max-tokens; after the burst drains and --idle-s, one
probe request of --warm-tokens ids. `/v1/completions` with `prompt_ids`, greedy, not streamed.

Per row (client.jsonl): tag, phase, status, submit/done ms, e2e_ms, completion_tokens, content_sha256, prompt_sha256,
error.
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--n", type=int, default=0, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--order", default=None, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--warm-tokens", type=int, default=512)
ap.add_argument("--burst", type=int, default=32)
ap.add_argument("--length", type=int, default=6144)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--idle-s", type=float, default=5.0)
ap.add_argument("--timeout-s", type=int, default=3600)
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, "client.jsonl")
open(out_path, "w").close()
lock = threading.Lock()


def post_json(path, body):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
        return json.load(r)


text = open(a.serving_md, encoding="utf-8").read()
need = max(a.length + 997 * (a.burst + 8), 60_000 + a.warm_tokens)
stream = []
while len(stream) < need and len(stream) < 64 * 400000:
    stream += post_json("/v1/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})["tokens"]
if len(stream) < need:
    sys.exit(f"tokenized stream {len(stream)} shorter than {need}")


def complete(tag, phase, ids, max_tokens):
    body = {"model": a.model, "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0,
            "cache_salt": f"d45-{tag}"}
    submit = time.time() * 1000.0
    status, err, content, usage = None, None, "", {}
    try:
        req = urllib.request.Request(a.base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout_s) as resp:
            status = resp.status
            j = json.load(resp)
            content = "".join(c.get("text") or "" for c in j.get("choices") or [])
            usage = j.get("usage") or {}
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode("utf-8", "replace")[:400]
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        status, err = -1, repr(e)[:400]
    done = time.time() * 1000.0
    row = dict(tag=tag, phase=phase, status=status, submit_ms=submit, done_ms=done, e2e_ms=done - submit,
               completion_tokens=usage.get("completion_tokens"),
               content_sha256=hashlib.sha256(content.encode()).hexdigest() if status == 200 else None,
               prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), error=err)
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{tag} status={status} G={row['completion_tokens']} e2e={row['e2e_ms']:.0f}", flush=True)


for i in range(4):
    off = 50_000 + i * 997
    complete(f"warm-{i}", "warm", stream[off:off + a.warm_tokens], 16)
threads = []
for i in range(a.burst):
    off = i * 997
    th = threading.Thread(target=complete, args=(f"burst-{i}", "burst", stream[off:off + a.length], a.max_tokens))
    threads.append(th)
for th in threads:
    th.start()
for th in threads:
    th.join()
time.sleep(a.idle_s)
off = 60_000
complete("probe", "probe", stream[off:off + a.warm_tokens], 16)
rows = [json.loads(l) for l in open(out_path)]
print(f"DAY45 CLIENT rows={len(rows)} ok={sum(1 for r in rows if r['status'] == 200)}")
