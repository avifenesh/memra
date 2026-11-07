"""spec-cache-mixed shape with streamed requests (TTFT per row) and prefix-cache counters.

Same workload as tools/spec-cache-mixed-gate.sh: one greedy warm request with a ~520-token
shared prompt, then a c8 window of unique-prompt batch load with six greedy repeats of
PROMPT + EXT_i fired 3 s apart. Every request streams, so each row carries TTFT.

  python3 mixed.py PORT OUT.json [--conc 8] [--repeats 6] [--otoks 60]
"""
import argparse, json, re, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("out")
ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--repeats", type=int, default=6)
ap.add_argument("--otoks", type=int, default=60)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

PROMPT = ("You are cataloguing the maintenance history of a small observatory. " * 40).strip()
EXTS = [f" Report focus area {i}: summarize the highest-risk subsystem." for i in range(1, a.repeats + 1)]


def metrics():
    raw = urllib.request.urlopen(BASE + "/metrics", timeout=15).read().decode()
    out = {}
    for m in re.finditer(r"^([A-Za-z_:]*(?:prefix_cache|evict|reclaim)[A-Za-z_:]*)(?:\{[^}]*\})? ([0-9.e+-]+)$", raw, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


rows, lock, stop_batch = [], threading.Lock(), threading.Event()


def fire(prompt, tag):
    body = {"model": "gate", "prompt": prompt, "max_tokens": a.otoks, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ttft, text, usage = None, [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            v = json.loads(line[5:])
            if v.get("usage"):
                usage = v["usage"]
            for c in v.get("choices") or []:
                if c.get("text"):
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(c["text"])
    wall = time.time() - t0
    with lock:
        rows.append({
            "tag": tag, "t_start": t0, "ttft_s": ttft, "wall_s": wall,
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "spec": (usage or {}).get("spec"),
            "text": "".join(text),
        })


m0 = metrics()
fire(PROMPT, "warm")


def batch_worker(i):
    n = 0
    while not stop_batch.is_set():
        fire(f"Unique load stream {i} item {n}. " + PROMPT[: 400 + 13 * i], f"batch-{i}-{n}")
        n += 1


t_start = time.time()
workers = [threading.Thread(target=batch_worker, args=(i,)) for i in range(a.conc)]
for w in workers:
    w.start()
try:
    for i, ext in enumerate(EXTS):
        time.sleep(3)
        fire(PROMPT + ext, f"repeat-{i}")
finally:
    stop_batch.set()
    for w in workers:
        w.join()
window_s = time.time() - t_start
m1 = metrics()
batch = [r for r in rows if r["tag"].startswith("batch-")]
rep = sorted((r for r in rows if r["tag"].startswith("repeat-")), key=lambda r: r["tag"])
agg = sum(r["completion_tokens"] or 0 for r in batch) / window_s
summary = {
    "window_s": round(window_s, 2), "batch_rows": len(batch), "batch_agg_tok_s": round(agg, 2),
    "repeat_cached": [r["cached"] for r in rep],
    "repeat_hits": sum(1 for r in rep if (r["cached"] or 0) > 0),
    "repeat_ttft_ms": [round(1000 * r["ttft_s"], 1) if r["ttft_s"] else None for r in rep],
    "batch_ttft_ms_p50": sorted(1000 * r["ttft_s"] for r in batch if r["ttft_s"])[len(batch) // 2] if batch else None,
    "metrics_delta": {k: m1.get(k, 0) - m0.get(k, 0) for k in m1},
    "metrics_end": m1,
}
json.dump({"rows": rows, "summary": summary}, open(a.out, "w"), indent=1)
print(json.dumps(summary))
