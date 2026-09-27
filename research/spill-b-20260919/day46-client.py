#!/usr/bin/env python3
"""WP-B day 46 client (DAY46.md 1.2 and addendum A): the enforcing predictive door's cell, one boot = one arm.

DAY24 1.1's sequence (one 64-token warm request; (a) four concurrent requests of 6,000 prompt tokens, max_tokens=96;
(b) one of 12,000; (c) the four of (a) again, the same salts so their prefixes are retained), then --idle-s, then a
burst of --burst requests of --length tokens released together (distinct windows, max_tokens --max-tokens), then
(addendum A) a second wave of burst/2 released when the burst's first request completes, then --idle-s and one
--probe-tokens probe. `/v1/completions` with `prompt_ids`, greedy, streamed with usage.

Per row (client.jsonl): tag, phase, salt, status, retry_after, submit/done ms, ttft_ms, e2e_ms, completion_tokens,
content_sha256, prompt_sha256, error. The cache_salt is `d46-<salt>` and appears as the tenant on the server's
`[admit-predict]` line; every salt is used by exactly one request except (a)i and (c)i, which share `seq<i>`.
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--n", type=int, default=0, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--order", default=None, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--burst", type=int, default=32)
ap.add_argument("--length", type=int, default=6144)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--probe-tokens", type=int, default=512)
ap.add_argument("--idle-s", type=float, default=5.0)
ap.add_argument("--timeout-s", type=int, default=3600)
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, "client.jsonl")
open(out_path, "w").close()
lock = threading.Lock()
first_burst_done = threading.Event()


def post_json(path, body):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
        return json.load(r)


text = open(a.serving_md, encoding="utf-8").read()
SEQ_OFF = 200_000  # the DAY24 sequence reads its own windows, clear of the burst's
need = max(a.length + 997 * (a.burst + a.burst // 2 + 8), SEQ_OFF + 12_000 + 5 * 7_001, 300_000 + a.probe_tokens)
stream = []
while len(stream) < need and len(stream) < 64 * 400000:
    stream += post_json("/v1/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})["tokens"]
if len(stream) < need:
    sys.exit(f"tokenized stream {len(stream)} shorter than {need}")


def complete(tag, phase, salt, ids, max_tokens):
    body = {"model": a.model, "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": f"d46-{salt}"}
    submit = time.time() * 1000.0
    first, usage, parts = None, None, []
    status, err, retry_after = None, None, None
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
                if j.get("error"):
                    err = json.dumps(j["error"])[:400]
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    t = ch.get("text") or ""
                    if t:
                        if first is None:
                            first = time.time() * 1000.0
                        parts.append(t)
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode("utf-8", "replace")[:400]
        retry_after = e.headers.get("Retry-After")
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        status, err = -1, repr(e)[:400]
    done = time.time() * 1000.0
    content = "".join(parts)
    row = dict(tag=tag, phase=phase, salt=salt, status=status, retry_after=retry_after, submit_ms=submit, done_ms=done,
               e2e_ms=done - submit, ttft_ms=None if first is None else first - submit,
               completion_tokens=(usage or {}).get("completion_tokens"),
               content_sha256=hashlib.sha256(content.encode()).hexdigest() if status == 200 and err is None else None,
               prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), error=err)
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{tag} status={status} G={row['completion_tokens']} ttft={row['ttft_ms']} e2e={row['e2e_ms']:.0f}", flush=True)
    if phase == "burst":
        first_burst_done.set()


def together(specs):
    ts = [threading.Thread(target=complete, args=s) for s in specs]
    for t in ts:
        t.start()
    return ts


def seq_ids(i, n):
    off = SEQ_OFF + i * 7_001
    return stream[off:off + n]


# DAY24 1.1's sequence
complete("warm", "warm", "warm", stream[SEQ_OFF - 1_000:SEQ_OFF - 1_000 + 64], 64)
time.sleep(2.0)
for t in together([(f"a{i}", "seq-a", f"seq{i}", seq_ids(i, 6_000), 96) for i in range(4)]):
    t.join()
time.sleep(2.0)
complete("b0", "seq-b", "seqb", seq_ids(4, 12_000), 96)
time.sleep(2.0)
for t in together([(f"c{i}", "seq-c", f"seq{i}", seq_ids(i, 6_000), 96) for i in range(4)]):
    t.join()
time.sleep(a.idle_s)
# the burst, then (addendum A) the second wave at the burst's first completion
burst = together([(f"burst-{i}", "burst", f"burst{i}", stream[i * 997:i * 997 + a.length], a.max_tokens)
                  for i in range(a.burst)])
first_burst_done.wait(timeout=a.timeout_s)
with lock:
    with open(os.path.join(a.out, "wave2.txt"), "w") as f:
        f.write(f"wave2_release_ms={time.time() * 1000.0:.1f}\n")
wave2 = together([(f"wave2-{i}", "wave2", f"wave2{i}", stream[(a.burst + 4 + i) * 997:(a.burst + 4 + i) * 997 + a.length],
                   a.max_tokens) for i in range(a.burst // 2)])
for t in burst + wave2:
    t.join()
time.sleep(a.idle_s)
complete("probe", "probe", "probe", stream[300_000:300_000 + a.probe_tokens], 16)
rows = [json.loads(l) for l in open(out_path)]
print(f"DAY46 CLIENT rows={len(rows)} ok={sum(1 for r in rows if r['status'] == 200 and not r['error'])}")
