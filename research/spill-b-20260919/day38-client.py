#!/usr/bin/env python3
"""WP-B day 38 client (DAY38.md 1.2): the MEMRA_KV_PARK_COMPACT deciding cell's resume workload, one boot = one arm.

The server runs the plain path with the prefix cache off (MEMRA_SERVE_SPEC=0, MEMRA_PREFIX_CACHE_MB=0), so a
continuation meets the continuation pool or runs cold. Every conversation has its own cache namespace (cache_salt).

Shape X (exact extension, the compact -> grow -> re-park cycle), /v1/completions with prompt_ids:
  t1 = base (L ids, L a multiple of 32), max_tokens=1  -> parks the prompt itself
  t2 = t1 + 64 ids, max_tokens=1                       -> resumes t1's entry, parks again
  t3 = t2 + 64 ids, max_tokens=32                      -> resumes t2's entry
  each turn's cold twin: the same prompt_ids in a fresh namespace.
Shape A (affinity rewind), /v1/chat/completions with an x-session-id header:
  t1 = system + user over the length-L text, max_tokens=16
  t2 = t1's messages + a fixed assistant reply + a new user message, max_tokens=32 (the render diverges after the
       checkpoint, so the plain-affinity path rewinds)
  each turn's cold twin: the same messages in a fresh namespace, no session id.

Per request (client.jsonl): tag, shape, turn, cold, length, rep, status, usage, finish_reason, content_sha256
(the completion text), submit/done ms, error, and /metrics before and after the request.
"""
import argparse, hashlib, json, os, sys, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--order", default=None, help="run-day26-cell.sh passes it; this workload has no order")
ap.add_argument("--lengths", default="6144,30720")
ap.add_argument("--shapes", default="X,A")
ap.add_argument("--timeout-s", type=int, default=3600)
ap.add_argument("--turn4", action="store_true", help="the fault boots (addendum B): a fourth shape-X turn")
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
lengths = [int(x) for x in a.lengths.split(",")]
for L in lengths:
    if L % 32:
        sys.exit(f"length {L} is not a multiple of 32 (the prime grid)")
shapes = a.shapes.split(",")
os.makedirs(a.out, exist_ok=True)
KEYS = ["cuda_driver_free_bytes", "cuda_pool_cached_bytes", "cuda_pool_reserved_bytes", "continuation_pool_entries",
        "continuation_pool_hits", "active_sessions", "kv_vmm_mapped_bytes", "kv_vmm_reserved_bytes"]


def post(path, body, headers=None, timeout=None):
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=timeout or a.timeout_s) as r:
        return r.status, json.load(r)


def metrics():
    try:
        with urllib.request.urlopen(a.base + "/metrics", timeout=5) as r:
            m = json.load(r)
    except Exception:
        return {k: None for k in KEYS}
    out = {}
    for k in KEYS:
        v = m.get(k)
        if isinstance(v, dict):
            v = sum(x for x in v.values() if isinstance(x, (int, float)))
        out[k] = v
    return out


# One long reversible id stream from the SERVING.md text, repeated until it covers the longest prompt plus room.
text = open(a.serving_md, encoding="utf-8").read()
need = max(lengths) + 5 * 997 * a.n + 4096
stream = []
reps = 0
while len(stream) < need and reps < 64:
    _, t = post("/v1/tokenize", {"model": a.model, "prompt": text, "add_special_tokens": False})
    stream += t["tokens"]
    reps += 1
if len(stream) < need:
    sys.exit(f"tokenized stream {len(stream)} shorter than {need}")
with open(os.path.join(a.out, "stream.json"), "w") as f:
    json.dump({"tokens": len(stream), "sha256": hashlib.sha256(",".join(map(str, stream)).encode()).hexdigest()}, f)


def ids(start, n):
    return stream[start:start + n]


rows = []


def run(tag, shape, turn, cold, L, rep, path, body, headers=None):
    before = metrics()
    t0 = int(time.time() * 1000)
    status, usage, finish, content, err = None, None, None, None, None
    try:
        status, j = post(path, body, headers)
        ch = (j.get("choices") or [{}])[0]
        usage = j.get("usage")
        finish = ch.get("finish_reason")
        content = ch.get("text")
        if content is None:
            msg = ch.get("message") or {}
            content = (msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or "")
    except urllib.error.HTTPError as e:
        status = e.code
        err = e.read().decode("utf-8", "replace")[:400]
    except Exception as e:  # noqa: BLE001 - recorded verbatim
        status = -1
        err = repr(e)[:400]
    t1 = int(time.time() * 1000)
    after = metrics()
    row = dict(tag=tag, shape=shape, turn=turn, cold=cold, length=L, rep=rep, status=status, usage=usage,
               finish_reason=finish, submit_ms=t0, done_ms=t1, error=err,
               content_sha256=None if content is None else hashlib.sha256(content.encode()).hexdigest(),
               content_chars=None if content is None else len(content), metrics_before=before, metrics_after=after)
    rows.append(row)
    with open(os.path.join(a.out, "client.jsonl"), "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"{tag} status={status} finish={finish} G={(usage or {}).get('completion_tokens')} "
          f"cached={((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens')} ms={t1 - t0}", flush=True)


def completion(ids_, max_tokens, salt):
    return {"model": a.model, "prompt_ids": ids_, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "cache_salt": salt}


def chat(messages, max_tokens, salt):
    return {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "cache_salt": salt}


open(os.path.join(a.out, "client.jsonl"), "w").close()
for L in lengths:
    for rep in range(a.n):
        off = rep * 997
        if "X" in shapes:
            t1 = ids(off, L)
            t2 = t1 + ids(off + L, 64)
            t3 = t2 + ids(off + L + 64, 64)
            t4 = t3 + ids(off + L + 128, 64)
            ns = f"x-{L}-{rep}"
            turns = [(1, t1, 1), (2, t2, 1), (3, t3, 32)] + ([(4, t4, 32)] if a.turn4 else [])
            for turn, p, mt in turns:
                run(f"X-{L}-r{rep}-t{turn}", "X", turn, False, L, rep, "/v1/completions", completion(p, mt, ns))
            for turn, p, mt in turns:
                run(f"X-{L}-r{rep}-t{turn}-cold", "X", turn, True, L, rep, "/v1/completions",
                    completion(p, mt, f"xc-{L}-{rep}-{turn}"))
        if "A" in shapes:
            _, dt = post("/v1/detokenize", {"model": a.model, "tokens": ids(off, L)})
            body_text = dt.get("text") or dt.get("prompt") or ""
            m1 = [{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": f"[conversation {L}-{rep}] Summarize this section:\n\n{body_text}"}]
            m2 = m1 + [{"role": "assistant", "content": "Here is a short summary of the section."},
                       {"role": "user", "content": "Add one more sentence to that summary."}]
            sid = {"x-session-id": f"a-{L}-{rep}"}
            ns = f"a-{L}-{rep}"
            run(f"A-{L}-r{rep}-t1", "A", 1, False, L, rep, "/v1/chat/completions", chat(m1, 16, ns), sid)
            run(f"A-{L}-r{rep}-t2", "A", 2, False, L, rep, "/v1/chat/completions", chat(m2, 32, ns), sid)
            run(f"A-{L}-r{rep}-t1-cold", "A", 1, True, L, rep, "/v1/chat/completions", chat(m1, 16, f"ac-{L}-{rep}-1"))
            run(f"A-{L}-r{rep}-t2-cold", "A", 2, True, L, rep, "/v1/chat/completions", chat(m2, 32, f"ac-{L}-{rep}-2"))
ok = sum(1 for r in rows if r["status"] == 200)
print(f"DAY38 CLIENT rows={len(rows)} ok={ok}")
