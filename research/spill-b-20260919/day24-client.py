#!/usr/bin/env python3
"""Day 24 memra#476 cell client: the fixed request sequence plus a 250 ms device/metrics sampler.
Writes <out>/samples.csv (epoch_ms, mem_used_mib, admission_booked_bytes, cuda_driver_free_bytes) and
<out>/client.jsonl (one row per request: tag, submit_ms, done_ms, status, prompt_chars, usage)."""
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
        return booked, m.get("cuda_driver_free_bytes")
    except Exception:
        return None, None

def sampler():
    with open(os.path.join(a.out, "samples.csv"), "w") as f:
        f.write("epoch_ms,mem_used_mib,admission_booked_bytes,cuda_driver_free_bytes\n")
        while not stop.is_set():
            t = int(time.time() * 1000)
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=2).stdout.strip().splitlines()[0]
            except Exception:
                used = ""
            b, fr = metrics()
            f.write(f"{t},{used},{'' if b is None else b},{'' if fr is None else fr}\n"); f.flush()
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

th = threading.Thread(target=sampler, daemon=True); th.start()
time.sleep(1.0)
chat("warm", slice_prompt(200, 1), 64)
time.sleep(2.0)                      # idle baseline: 8 samples
def burst(tag, chars_list, max_tokens):
    ts = [threading.Thread(target=chat, args=(f"{tag}{i}", slice_prompt(c, 10 + i), max_tokens)) for i, c in enumerate(chars_list)]
    [t.start() for t in ts]; [t.join() for t in ts]
burst("a", [21000, 22000, 23000, 24000], 96)   # about 6k tokens each, four concurrent
time.sleep(2.0)
burst("b", [46000], 96)                        # about 12k tokens, alone
time.sleep(2.0)
burst("c", [21000, 22000, 23000, 24000], 96)   # the same four again (prefixes retained)
time.sleep(2.0)
stop.set(); th.join(timeout=3)
with open(os.path.join(a.out, "client.jsonl"), "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
bad = [r for r in rows if r["status"] != 200]
print(f"requests={len(rows)} non-200={len(bad)}")
sys.exit(1 if bad else 0)
