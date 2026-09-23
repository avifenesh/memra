#!/usr/bin/env python3
"""Day 31 MEMRA_ADMIT_BY_MEMORY cell client (DAY31.md section 1). One boot = one arm; this client runs the whole
pre-registered workload against it and never changes with the arm.

Workload, in this order, every request greedy (temperature 0.0), non-streaming, one at a time except the burst:
  warmup  one 300-char summary, max_tokens=32
  mix     the day-26 mix unchanged (day26-client.py): per prompt length L in 5000/10000/20000 chars, N reps,
          (i) open (max_tokens omitted), (ii) max_tokens=96, (iii) the (i) conversation continued with
          "Add one more sentence to that summary." (max_tokens omitted), deferred after every first turn.
          Order AB runs (i) then (ii) inside each length, BA runs (ii) then (i).
  iv      the long-generation open class, max_tokens omitted, three reps of each type:
          iv-a  an operator manual over a 6000-char SERVING.md slice (salt 700+r)
          iv-b  a multiplication table from 1 x 1 to k x k, k = 24, 40, 72 for r = 0, 1, 2 (salt 800+r)
          AB runs iv-a then iv-b, BA runs iv-b then iv-a.
  burst   B concurrent open (i)-shape requests at the 5000-char length, salts 900..900+B-1, all released on one
          barrier; excluded from the identity gate, read for admitted concurrency, 429s and Retry-After.

Per request row (client.jsonl): tag, class, submit/done ms, HTTP status, x-request-id, Retry-After (on a non-200),
usage, finish_reason, prompt_sha256 (the messages array, sorted-key JSON), message_sha256 (the whole response
message object, sorted-key JSON: reasoning plus content plus any tool calls), content_sha256, reasoning and content
character counts, the error body (first 400 chars), and /metrics before and after.
samples.csv: 250 ms nvidia-smi memory.used plus the /metrics keys, as day 26.
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="q9")
ap.add_argument("--order", choices=["AB", "BA"], required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--chars", default="5000,10000,20000")
ap.add_argument("--bounded", type=int, default=96)
ap.add_argument("--burst", type=int, default=32)
ap.add_argument("--timeout-s", type=int, default=3600, help="client socket timeout per request")
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
text = open(a.serving_md, encoding="utf-8").read()
chars = [int(c) for c in a.chars.split(",")]
stop = threading.Event()
rows = []
rows_lock = threading.Lock()
KEYS = ["cuda_driver_free_bytes", "cuda_pool_used_bytes", "cuda_pool_reserved_bytes", "admission_booked_bytes",
        "active_sessions", "prefix_cache_bytes"]
IV_K = [24, 40, 72]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


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


def chat(tag, cls, messages, max_tokens, quiet_metrics=False):
    body = {"model": a.model, "messages": messages, "temperature": 0.0, "stream": False}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    before = None if quiet_metrics else metrics()
    t0 = int(time.time() * 1000); status = None; usage = None; err = None; msg = None; j = {}
    rid = None; retry_after = None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
            status = r.status; rid = r.headers.get("x-request-id"); j = json.load(r); usage = j.get("usage")
            msg = ((j.get("choices") or [{}])[0].get("message"))
    except urllib.error.HTTPError as e:
        status = e.code; rid = e.headers.get("x-request-id"); retry_after = e.headers.get("Retry-After")
        err = e.read().decode(errors="replace")[:400]
    except Exception as e:
        err = repr(e)[:400]
    t1 = int(time.time() * 1000)
    after = None if quiet_metrics else metrics()
    content = None if msg is None else msg.get("content")
    reasoning = None if msg is None else (msg.get("reasoning") or msg.get("reasoning_content"))
    row = {"tag": tag, "class": cls, "submit_ms": t0, "done_ms": t1, "status": status, "request_id": rid,
           "retry_after": retry_after, "prompt_chars": sum(len(m["content"]) for m in messages),
           "max_tokens": max_tokens, "usage": usage, "err": err, "before": before, "after": after,
           "prompt_sha256": sha(messages),
           "message_sha256": None if msg is None else sha(msg),
           "content_sha256": None if content is None else hashlib.sha256(content.encode()).hexdigest(),
           "content_chars": None if content is None else len(content),
           "reasoning_chars": None if reasoning is None else len(reasoning),
           "finish_reason": ((j.get("choices") or [{}])[0].get("finish_reason") if status == 200 else None)}
    with rows_lock:
        rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k not in ("before", "after")}), flush=True)
    return content


def prompt(c, salt):
    # day 26 byte-for-byte: distinct salt per rep, no shared prefix between reps
    return f"[{salt}] Summarize the following in one sentence.\n\n" + text[salt * 13:(salt * 13) + c]


def iv_a(r):
    salt = 700 + r
    return (f"[{salt}] Write a complete operator manual for the server described in the notes below. Give every "
            "section a heading, explain every setting and endpoint you can find in the notes with a worked example "
            "for each, and finish with a troubleshooting section of at least ten problems and their fixes.\n\n"
            + text[salt * 17:(salt * 17) + 6000])


def iv_b(r):
    salt = 800 + r; k = IV_K[r]
    return (f"[{salt}] Write out the full multiplication table from 1 x 1 up to {k} x {k}: every product on its own "
            "line in the form 'a x b = c', in order, with no line skipped and no ellipsis. After the last line, "
            "state the sum of all the products.")


th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(1.0)
chat("warmup", "warmup", [{"role": "user", "content": prompt(300, 1)}], 32)
time.sleep(2.0)
replies = {}
for li, c in enumerate(chars):
    arms = ["i", "ii"] if a.order == "AB" else ["ii", "i"]
    for arm in arms:
        for rep in range(a.n):
            salt = 100 + li * 10 + rep if arm == "i" else 500 + li * 10 + rep
            msgs = [{"role": "user", "content": prompt(c, salt)}]
            reply = chat(f"{arm}-L{li}-r{rep}", arm, msgs, None if arm == "i" else a.bounded)
            if arm == "i":
                replies[(li, rep)] = (msgs, reply)
    time.sleep(1.0)
for li, c in enumerate(chars):
    for rep in range(a.n):
        msgs, reply = replies[(li, rep)]
        cont = msgs + [{"role": "assistant", "content": reply or ""},
                       {"role": "user", "content": "Add one more sentence to that summary."}]
        chat(f"iii-L{li}-r{rep}", "iii", cont, None)
time.sleep(1.0)
iv_types = [("iv-a", iv_a), ("iv-b", iv_b)] if a.order == "AB" else [("iv-b", iv_b), ("iv-a", iv_a)]
for name, fn in iv_types:
    for r in range(3):
        chat(f"{name}-r{r}", name, [{"role": "user", "content": fn(r)}], None)
time.sleep(2.0)
# burst: every thread builds its request, then all release on one barrier
barrier = threading.Barrier(a.burst)
marks = {}


def burst_one(b):
    msgs = [{"role": "user", "content": prompt(5000, 900 + b)}]
    barrier.wait()
    chat(f"burst-b{b}", "burst", msgs, None, quiet_metrics=True)


marks["burst_start_ms"] = int(time.time() * 1000)
ts = [threading.Thread(target=burst_one, args=(b,)) for b in range(a.burst)]
for t in ts:
    t.start()
for t in ts:
    t.join()
marks["burst_end_ms"] = int(time.time() * 1000)
time.sleep(2.0)
stop.set(); th.join(timeout=3)
with open(os.path.join(a.out, "client.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(a.out, "marks.json"), "w") as f:
    json.dump(marks, f)
bad = [r for r in rows if r["status"] != 200 and r["class"] != "burst"]
print(f"requests={len(rows)} non-200-outside-burst={len(bad)} burst-non-200={sum(1 for r in rows if r['class'] == 'burst' and r['status'] != 200)}")
sys.exit(1 if bad else 0)
