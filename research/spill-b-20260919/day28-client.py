#!/usr/bin/env python3
"""Day 28 memra#476 shape-walk client (DAY28.md 1.4): warm, S1, S2, S4, S8, L, S8b, plus a 250 ms device/metrics sampler.
Writes <out>/samples.csv (epoch_ms, mem_used_mib, admission_booked_bytes, cuda_driver_free_bytes, pool_reserved, pool_used,
pool_cached) and <out>/client.jsonl (one row per request: tag, submit_ms, done_ms, status, prompt_chars, usage)."""
import argparse, json, subprocess, sys, threading, time, urllib.request, urllib.error, os

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--model", default="q9")
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
a = ap.parse_args()
text = open(a.serving_md, encoding="utf-8").read()
stop = threading.Event()
rows = []

def metrics():
    try:
        with urllib.request.urlopen(a.base + "/metrics", timeout=1) as r:
            m = json.load(r)
        booked = m.get("admission_booked_bytes") or {}
        booked = sum(booked.values()) if isinstance(booked, dict) else booked
        return (booked, m.get("cuda_driver_free_bytes"), m.get("cuda_pool_reserved_bytes"),
                m.get("cuda_pool_used_bytes"), m.get("cuda_pool_cached_bytes"))
    except Exception:
        return (None,) * 5

def sampler():
    with open(os.path.join(a.out, "samples.csv"), "w") as f:
        f.write("epoch_ms,mem_used_mib,admission_booked_bytes,cuda_driver_free_bytes,pool_reserved,pool_used,pool_cached\n")
        while not stop.is_set():
            t = int(time.time() * 1000)
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
            except Exception:
                used = ""
            vals = metrics()
            f.write(f"{t},{used}," + ",".join("" if v is None else str(v) for v in vals) + "\n"); f.flush()
            time.sleep(max(0.0, 0.25 - (time.time() * 1000 - t) / 1000))

def chat(tag, prompt, max_tokens):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = int(time.time() * 1000); status = None; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            status = r.status; usage = json.load(r).get("usage")
    except urllib.error.HTTPError as e:
        status = e.code; err = e.read().decode(errors="replace")[:300]
    except Exception as e:
        err = repr(e)[:300]
    row = {"tag": tag, "submit_ms": t0, "done_ms": int(time.time() * 1000), "status": status,
           "prompt_chars": len(prompt), "max_tokens": max_tokens, "usage": usage, "err": err}
    rows.append(row); print(json.dumps(row), flush=True)

def slice_prompt(chars, salt):
    # distinct salt per request so every request tokenizes to its own length and gets its own request-cost line
    return f"[{salt}] Summarize the following in one sentence.\n\n" + text[salt * 7:(salt * 7) + chars]

def burst(tag, chars_list, max_tokens, salt0):
    ts = [threading.Thread(target=chat, args=(f"{tag}-{i}", slice_prompt(c, salt0 + i), max_tokens)) for i, c in enumerate(chars_list)]
    [t.start() for t in ts]; [t.join() for t in ts]
    time.sleep(2.0)

th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(1.0)
chat("warm", slice_prompt(200, 1), 64)
time.sleep(2.0)                                          # idle baseline: 8 samples
two_k = [7000 + 300 * i for i in range(8)]               # about 2k prompt tokens each, distinct lengths
burst("S1", two_k[:1], 96, 10)                           # B=1, spec (the first serving draft-chain capture)
burst("S2", two_k[:2], 96, 20)                           # B=2, spec (LOW=2)
burst("S4", two_k[:4], 96, 30)                           # B=4, plain batched trunk
burst("S8", two_k[:8], 96, 40)                           # B=8, the trunk's cap
burst("L", [80000], 96, 50)                              # about 23k prompt tokens alone: a new t_kv shape
burst("S8b", two_k[:8], 96, 60)                          # control: every shape already presented
stop.set(); th.join(timeout=3)
with open(os.path.join(a.out, "client.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
bad = [r for r in rows if r["status"] != 200]
print(f"requests={len(rows)} non-200={len(bad)}")
sys.exit(1 if bad else 0)
