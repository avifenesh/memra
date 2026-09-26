#!/usr/bin/env python3
"""WP-B day 44 client (DAY44.md 1.6): day 41's client plus --turn-gap-ms (the RXg shape: a sleep before turns 2
and 3, so an off-path settle can land). Otherwise unchanged from day41-client.py.

Day 41's description follows: the grid-checkpoint rewind arm's pricing workload, one boot = one arm.

Shape RX (a raw agent loop), `/v1/completions` with `prompt_ids`, greedy, streamed with usage:
  turn 1 = L ids (L on the grid), max_ctx = L + --max-ctx-pad, max_tokens = G
  turn 2 = turn 1's ids + turn 1's completion tokenized (`/v1/tokenize`) + 64 stream ids, max_tokens = G
  turn 3 = likewise from turn 2
  each turn's cold twin: the same ids and max_ctx in a fresh namespace, after the conversation.
  G in --gens (every G in every boot, as separate conversations), L in --lengths, N conversations per (L, G).
Shape FX (the fanout cost): --fanout requests sharing a --fanout-prefix-token prefix in one namespace, released
together, max_tokens 16.

Per row (client.jsonl): tag, shape, L, G, turn, cold, rep, status, ttft_ms (first streamed token text), e2e_ms,
completion_tokens, cached_tokens, finish_reason, content_sha256, prompt_sha256, pool-hit delta, submit/done ms, error.
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--order", default=None, help="run-day26-cell.sh passes it; unused")
ap.add_argument("--lengths", default="6144,30720")
ap.add_argument("--gens", default="32,256")
ap.add_argument("--shapes", default="RX,FX")
ap.add_argument("--max-ctx-pad", type=int, default=2048)
ap.add_argument("--fanout", type=int, default=8)
ap.add_argument("--fanout-prefix", type=int, default=4096)
ap.add_argument("--timeout-s", type=int, default=3600)
ap.add_argument("--turn-gap-ms", type=int, default=0, help="sleep before turns 2 and 3 (RXg)")
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
lengths = [int(x) for x in a.lengths.split(",")]
gens = [int(x) for x in a.gens.split(",")]
shapes = a.shapes.split(",")
for L in lengths:
    if L % 32:
        sys.exit(f"length {L} is not a multiple of 32 (the prime grid)")
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, "client.jsonl")
open(out_path, "w").close()
lock = threading.Lock()


def post_json(path, body):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
        return json.load(r)


def hits():
    try:
        with urllib.request.urlopen(a.base + "/metrics", timeout=5) as r:
            return json.load(r).get("continuation_pool_hits")
    except Exception:
        return None


text = open(a.serving_md, encoding="utf-8").read()
need = max(lengths + [a.fanout_prefix]) + 997 * a.n * len(gens) + 64 * 8 + 4096
stream = []
while len(stream) < need and len(stream) < 64 * 400000:
    stream += post_json("/v1/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})["tokens"]
if len(stream) < need:
    sys.exit(f"tokenized stream {len(stream)} shorter than {need}")


def stream_completion(tag, meta, ids, max_tokens, salt, max_ctx=None):
    body = {"model": a.model, "prompt_ids": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_salt": salt}
    if max_ctx is not None:
        body["max_ctx"] = max_ctx
    h0 = hits()
    submit = time.time() * 1000.0
    first = None
    usage = None
    finish = None
    parts = []
    status, err = None, None
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
                        if first is None:
                            first = time.time() * 1000.0
                        parts.append(t)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        status, err = e.code, e.read().decode("utf-8", "replace")[:400]
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        status, err = -1, repr(e)[:400]
    done = time.time() * 1000.0
    content = "".join(parts)
    h1 = hits()
    row = dict(meta, tag=tag, status=status, submit_ms=submit, done_ms=done, e2e_ms=done - submit,
               ttft_ms=None if first is None else first - submit, error=err, finish_reason=finish,
               completion_tokens=(usage or {}).get("completion_tokens"),
               cached_tokens=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               content_sha256=hashlib.sha256(content.encode()).hexdigest() if status == 200 else None,
               prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), max_ctx=max_ctx,
               pool_hits_delta=None if h0 is None or h1 is None else h1 - h0)
    with lock:
        with open(out_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{tag} status={status} G={row['completion_tokens']} cached={row['cached_tokens']} "
          f"ttft={row['ttft_ms']} e2e={row['e2e_ms']:.0f}", flush=True)
    return content if status == 200 else None


def tokenize(text_):
    if not text_:
        return []
    return post_json("/v1/tokenize", {"model": a.model, "prompt": text_, "add_special_tokens": False})["tokens"]


if "RX" in shapes:
    k = 0
    for L in lengths:
        for G in gens:
            for rep in range(a.n):
                off = k * 997
                k += 1
                cap = L + a.max_ctx_pad
                ns = f"rx-{L}-{G}-{rep}"
                p = stream[off:off + L]
                nxt = off + L
                prompts = []
                for turn in (1, 2, 3):
                    if turn > 1 and a.turn_gap_ms > 0:
                        time.sleep(a.turn_gap_ms / 1000.0)
                    meta = dict(shape="RX", L=L, G=G, turn=turn, cold=False, rep=rep, gap_ms=a.turn_gap_ms)
                    prompts.append((turn, p))
                    out = stream_completion(f"RX-{L}-g{G}-r{rep}-t{turn}", meta, p, G, ns, cap)
                    g = tokenize(out)
                    p = p + g + stream[nxt:nxt + 64]
                    nxt += 64
                for turn, pp in prompts:
                    meta = dict(shape="RX", L=L, G=G, turn=turn, cold=True, rep=rep)
                    stream_completion(f"RX-{L}-g{G}-r{rep}-t{turn}-cold", meta, pp, G, f"rxc-{L}-{G}-{rep}-{turn}", cap)

if "FX" in shapes:
    base = stream[:a.fanout_prefix]
    threads = []
    for i in range(a.fanout):
        ids = base + stream[a.fanout_prefix + 97 * i:a.fanout_prefix + 97 * i + 64]
        meta = dict(shape="FX", L=a.fanout_prefix, G=16, turn=1, cold=False, rep=i)
        # One namespace: the shared prefix is what the in-batch fanout can serve.
        th = threading.Thread(target=stream_completion, args=(f"FX-r{i}", meta, ids, 16, "fx"))
        threads.append(th)
    for th in threads:
        th.start()
    for th in threads:
        th.join()

rows = [json.loads(l) for l in open(out_path)]
print(f"DAY41 CLIENT rows={len(rows)} ok={sum(1 for r in rows if r['status'] == 200)}")
