#!/usr/bin/env python3
"""Day 26 memra#539 cell client (DAY26.md section 2): the fixed request mix plus a 250 ms device/metrics sampler.

Arms, per prompt length L (three lengths), N reps each, distinct salt per rep so no rep shares a prefix with another:
  (i)   open:    max_tokens omitted (the OpenAI default), one-sentence summary prompt
  (ii)  bounded: max_tokens = --bounded (96, the serving-density shape)
  (iii) warm:    the (i) conversation continued (user P_k, assistant reply_k, user "one more sentence"), max_tokens omitted
Order AB runs (i) then (ii) inside each length, BA runs (ii) then (i); (iii) follows after both arms of every length.
Writes <out>/samples.csv (250 ms: epoch_ms, mem_used_mib, cuda_driver_free_bytes, cuda_pool_used_bytes,
cuda_pool_reserved_bytes, admission_booked_bytes, active_sessions, prefix_cache_bytes) and <out>/client.jsonl
(one row per request: arm, length index, rep, submit/done ms, status, usage, driver free and pool used before/after).
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="q9")
ap.add_argument("--order", choices=["AB", "BA"], required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--chars", default="5000,10000,20000", help="prompt slice lengths in characters (about 3.5 chars per token)")
ap.add_argument("--bounded", type=int, default=96)
ap.add_argument("--warm-immediate", action="store_true",
                help="run each (iii) continuation right after its (i) request instead of after every first turn (the hit shape)")
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
text = open(a.serving_md, encoding="utf-8").read()
chars = [int(c) for c in a.chars.split(",")]
stop = threading.Event()
rows = []
KEYS = ["cuda_driver_free_bytes", "cuda_pool_used_bytes", "cuda_pool_reserved_bytes", "admission_booked_bytes",
        "active_sessions", "prefix_cache_bytes"]


def metrics():
    try:
        with urllib.request.urlopen(a.base + "/metrics", timeout=1) as r:
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


def sampler():
    with open(os.path.join(a.out, "samples.csv"), "w") as f:
        f.write("epoch_ms,mem_used_mib," + ",".join(KEYS) + "\n")
        while not stop.is_set():
            t = int(time.time() * 1000)
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
            except Exception:
                used = ""
            m = metrics()
            f.write(f"{t},{used}," + ",".join("" if m[k] is None else str(m[k]) for k in KEYS) + "\n"); f.flush()
            time.sleep(max(0.0, 0.25 - (time.time() * 1000 - t) / 1000))


def chat(tag, arm, li, rep, messages, max_tokens):
    body = {"model": a.model, "messages": messages, "temperature": 0.0, "stream": False}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    before = metrics()
    t0 = int(time.time() * 1000); status = None; usage = None; err = None; content = None; j = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status = r.status; j = json.load(r); usage = j.get("usage")
            content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content")
    except urllib.error.HTTPError as e:
        status = e.code; err = e.read().decode(errors="replace")[:400]
    except Exception as e:
        err = repr(e)[:400]
    t1 = int(time.time() * 1000)
    after = metrics()
    row = {"tag": tag, "arm": arm, "length_idx": li, "rep": rep, "submit_ms": t0, "done_ms": t1, "status": status,
           "prompt_chars": sum(len(m["content"]) for m in messages), "max_tokens": max_tokens, "usage": usage,
           "err": err, "before": before, "after": after, "content_chars": None if content is None else len(content),
           # day 27: completion digest so two arms (or a before and an after binary) compare byte-for-byte
           "content_sha256": None if content is None else hashlib.sha256(content.encode()).hexdigest(),
           "finish_reason": ((j.get("choices") or [{}])[0].get("finish_reason") if status == 200 else None)}
    rows.append(row); print(json.dumps({k: v for k, v in row.items() if k not in ("before", "after")}), flush=True)
    return content


def prompt(c, salt):
    # distinct salt per rep: every request tokenizes to its own length and shares no prefix with another rep
    return f"[{salt}] Summarize the following in one sentence.\n\n" + text[salt * 13:(salt * 13) + c]


th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(1.0)
chat("warmup", "warmup", -1, -1, [{"role": "user", "content": prompt(300, 1)}], 32)
time.sleep(2.0)
replies = {}
for li, c in enumerate(chars):
    arms = ["i", "ii"] if a.order == "AB" else ["ii", "i"]
    for arm in arms:
        for rep in range(a.n):
            salt = 100 + li * 10 + rep if arm == "i" else 500 + li * 10 + rep
            msgs = [{"role": "user", "content": prompt(c, salt)}]
            reply = chat(f"{arm}-L{li}-r{rep}", arm, li, rep, msgs, None if arm == "i" else a.bounded)
            if arm == "i":
                replies[(li, rep)] = (msgs, reply)
                if a.warm_immediate:
                    cont = msgs + [{"role": "assistant", "content": reply or ""},
                                   {"role": "user", "content": "Add one more sentence to that summary."}]
                    chat(f"iii-L{li}-r{rep}", "iii", li, rep, cont, None)
    time.sleep(1.0)
for li, c in enumerate(chars):
    for rep in range(a.n):
        if a.warm_immediate:
            break
        msgs, reply = replies[(li, rep)]
        cont = msgs + [{"role": "assistant", "content": reply or ""},
                       {"role": "user", "content": "Add one more sentence to that summary."}]
        chat(f"iii-L{li}-r{rep}", "iii", li, rep, cont, None)
time.sleep(2.0)
stop.set(); th.join(timeout=3)
with open(os.path.join(a.out, "client.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
bad = [r for r in rows if r["status"] != 200]
print(f"requests={len(rows)} non-200={len(bad)}")
sys.exit(1 if bad else 0)
