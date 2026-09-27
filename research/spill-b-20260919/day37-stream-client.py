#!/usr/bin/env python3
"""WP-B day 37 streaming load client (DAY37.md 1.6 A3 (ii)): TTFT, ITL, TPOT, E2E and throughput of one boot.

One boot = one arm. The client never changes with the arm. Workload: `--requests` greedy chat requests over one
`--chars`-long slice of docs/SERVING.md with a per-request salt, `--concurrency` in flight at a time (a new request
starts as one finishes), streaming (`stream: true`, `stream_options.include_usage: true`). `max_tokens` cycles
through `--max-tokens` (a comma list), so long requests cross KV granule boundaries while short ones churn
admission and retirement.

Per request (client.jsonl): tag, salt, max_tokens, HTTP status, submit and done ms (epoch), ttft_ms (first chunk
carrying content or reasoning), e2e_ms, completion_tokens (usage), chunk gaps in ms (ITL, one per chunk after the
first), finish_reason, the sha256 of the concatenated reasoning and content (so arms compare byte for byte), error.
SUMMARY: `DAY37 STREAM arm=<arm> N=<ok> ttft_p50=.. ttft_p95=.. ttft_p99=.. itl_p50=.. itl_p95=.. itl_p99=..
tpot_p50=.. tpot_p95=.. tpot_p99=.. e2e_p50=.. e2e_p95=.. e2e_p99=.. req_per_s=.. out_tok_per_s=.. wall_s=..`.
"""
import argparse, hashlib, json, os, subprocess, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--model", default="q9")
ap.add_argument("--requests", type=int, default=32)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--chars", type=int, default=5000)
ap.add_argument("--max-tokens", default="256,2600")
ap.add_argument("--salt", type=int, default=3700)
ap.add_argument("--timeout-s", type=int, default=1800)
ap.add_argument("--serving-md", default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "SERVING.md"))
# run-day26-cell.sh passes these to every client; this workload has no order or rep count.
ap.add_argument("--order", default=None)
ap.add_argument("--n", default=None)
a = ap.parse_args()
text = open(a.serving_md, encoding="utf-8").read()
max_tokens = [int(x) for x in a.max_tokens.split(",")]
os.makedirs(a.out, exist_ok=True)


def now_ms():
    return time.time() * 1000.0


def prompt(i):
    start = (i * 997) % max(1, len(text) - a.chars)
    body = text[start:start + a.chars]
    return f"[salt {a.salt + i}] Summarize the following section in plain words.\n\n{body}"


def one(i):
    mt = max_tokens[i % len(max_tokens)]
    req = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt(i)}],
        "max_tokens": mt,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(req).encode()
    row = {"tag": f"s{i}", "salt": a.salt + i, "max_tokens": mt}
    submit = now_ms()
    row["submit_ms"] = submit
    first = None
    last = None
    gaps = []
    usage = None
    finish = None
    digest = hashlib.sha256()
    try:
        r = urllib.request.Request(a.base + "/v1/chat/completions", data=data,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=a.timeout_s) as resp:
            row["status"] = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (d.get("reasoning") or "") + (d.get("reasoning_content") or "") + (d.get("content") or "")
                    if piece:
                        t = now_ms()
                        if first is None:
                            first = t
                        else:
                            gaps.append(round(t - last, 3))
                        last = t
                        digest.update(piece.encode())
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        row["status"] = e.code
        row["retry_after"] = e.headers.get("Retry-After")
        row["error"] = e.read().decode("utf-8", "replace")[:400]
    except Exception as e:  # noqa: BLE001 - every failure is recorded verbatim, never dropped
        row["status"] = row.get("status", -1)
        row["error"] = repr(e)[:400]
    done = now_ms()
    row["done_ms"] = done
    row["e2e_ms"] = round(done - submit, 3)
    row["ttft_ms"] = round(first - submit, 3) if first is not None else None
    row["completion_tokens"] = (usage or {}).get("completion_tokens")
    row["finish_reason"] = finish
    row["itl_ms"] = gaps
    row["text_sha256"] = digest.hexdigest()
    return row


KEYS = ["cuda_driver_free_bytes", "cuda_pool_used_bytes", "cuda_pool_reserved_bytes", "cuda_pool_cached_bytes",
        "active_sessions", "prefix_cache_bytes", "kv_vmm_mapped_bytes", "kv_vmm_reserved_bytes", "kv_vmm_owed_bytes",
        "kv_vmm_graveyard_bytes", "kv_vmm_grows_total", "kv_vmm_mapper_grows_total", "kv_vmm_grow_waits_total"]
stop = threading.Event()


def metrics():
    try:
        with urllib.request.urlopen(a.base + "/metrics", timeout=1) as r:
            m = json.load(r)
    except Exception:
        return {k: None for k in KEYS}
    return {k: m.get(k) for k in KEYS}


def sampler():
    """250 ms telemetry: the card's temperature, power, SM clock and memory, and the /metrics memory keys."""
    with open(os.path.join(a.out, "samples.csv"), "w") as f:
        f.write("epoch_ms,temp_c,power_w,sm_mhz,mem_used_mib," + ",".join(KEYS) + "\n")
        while not stop.is_set():
            t = int(time.time() * 1000)
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=2).stdout.strip().splitlines()[0].replace(" ", "")
            except Exception:
                gpu = ",,,"
            m = metrics()
            f.write(f"{t},{gpu}," + ",".join("" if m[k] is None else str(m[k]) for k in KEYS) + "\n")
            f.flush()
            time.sleep(max(0.0, 0.25 - (time.time() * 1000 - t) / 1000))


rows = []
lock = threading.Lock()
next_i = [0]


def worker():
    while True:
        with lock:
            i = next_i[0]
            if i >= a.requests:
                return
            next_i[0] += 1
        row = one(i)
        with lock:
            rows.append(row)


samp = threading.Thread(target=sampler, daemon=True)
samp.start()
with open(os.path.join(a.out, "metrics-before.json"), "w") as f:
    json.dump(metrics(), f, sort_keys=True)
t0 = now_ms()
threads = [threading.Thread(target=worker) for _ in range(a.concurrency)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall_s = (now_ms() - t0) / 1000.0
time.sleep(1.0)
with open(os.path.join(a.out, "metrics-after.json"), "w") as f:
    json.dump(metrics(), f, sort_keys=True)
stop.set()
samp.join(timeout=5)
rows.sort(key=lambda r: r["salt"])
with open(os.path.join(a.out, "client.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


ok = [r for r in rows if r.get("status") == 200 and r.get("ttft_ms") is not None]
itl = [g for r in ok for g in r["itl_ms"]]
tpot = [(r["e2e_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
        for r in ok if (r.get("completion_tokens") or 0) > 1]
out_tokens = sum(r.get("completion_tokens") or 0 for r in ok)
line = (f"DAY37 STREAM arm={a.arm} N={len(ok)} of {len(rows)} "
        + " ".join(f"{name}_p{int(q*100)}={pct(vals, q):.2f}" for name, vals in
                   (("ttft", [r["ttft_ms"] for r in ok]), ("itl", itl), ("tpot", tpot), ("e2e", [r["e2e_ms"] for r in ok]))
                   for q in (0.5, 0.95, 0.99))
        + f" req_per_s={len(ok) / wall_s:.4f} out_tok_per_s={out_tokens / wall_s:.2f} wall_s={wall_s:.1f}")
with open(os.path.join(a.out, "SUMMARY.txt"), "w") as f:
    f.write(line + "\n")
print(line)
