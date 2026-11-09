"""Steady traffic at the prefix cache's limit on a discrete card: fill the cache and the parked-session pool as the
#1090 lane's limit.py does, then hold a continuous stream for --duration seconds. W workers each send their next
request as soon as the last one ends; every --repeat-every-th request repeats an earlier prompt (same text, same
`cache_salt` namespace) drawn from --back-min to --back-max requests back in issue order, every other one is a fresh
unique prompt in its own namespace. No bursts: the stream's concurrency is W throughout.

  python3 steady.py PORT OUT.json [--tok 24000] [--fill 64] [--fill-conc 4] [--workers 6] [--duration 900]
                                  [--repeat-every 4] [--back-min 16] [--back-max 48] [--otoks 64] [--seed 7]
"""
import argparse, json, random, re, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("out")
ap.add_argument("--tok", type=int, default=24000)
ap.add_argument("--fill", type=int, default=64)
ap.add_argument("--fill-conc", type=int, default=4)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--duration", type=float, default=900.0)
ap.add_argument("--repeat-every", type=int, default=4)
ap.add_argument("--back-min", type=int, default=16)
ap.add_argument("--back-max", type=int, default=48)
ap.add_argument("--otoks", type=int, default=64)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

WORDS = ("ledger valve turbine gasket relay sensor bracket conduit spindle flange piston rotor bearing coupling "
         "manifold actuator governor damper impeller housing").split()


def post(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def ntok(text):
    v = post("/v1/tokenize", {"model": "gate", "prompt": text})
    return v.get("count") or len(v.get("tokens") or [])


def body_text(tag, n_items):
    # deterministic, unique from the first token (no shared prefix between prompts)
    parts = [f"Inspection log {tag}."]
    for k in range(n_items):
        w = WORDS[(k * 7 + len(tag)) % len(WORDS)]
        parts.append(f"Entry {tag}-{k}: the {w} at bay {k % 97} read {(k * 31 + len(tag) * 17) % 1000} units.")
    parts.append("Summarize the three most unusual entries.")
    return " ".join(parts)


# size the prompts once: items per prompt for ~a.tok tokens
probe_items = 400
per_item = ntok(body_text("probe", probe_items)) / probe_items
items = max(1, int(a.tok / per_item))


def metrics():
    req = urllib.request.Request(BASE + "/metrics", headers={"Accept": "text/plain"})  # Prometheus text, not JSON
    raw = urllib.request.urlopen(req, timeout=15).read().decode()
    out = {}
    for m in re.finditer(r"^([A-Za-z_:]*(?:prefix_cache|evict|reclaim|park|reuse|defer)[A-Za-z_:]*)(?:\{[^}]*\})? ([0-9.e+-]+)$", raw, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


rows, lock = [], threading.Lock()


def fire(prompt, tag, salt, phase):
    body = {"model": "gate", "prompt": prompt, "max_tokens": a.otoks, "temperature": 0.0, "cache_salt": salt,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ttft, text, usage, err = None, [], None, None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
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
    except Exception as e:  # a refused or dropped request is a result, not a harness failure
        err = repr(e)[:300]
    with lock:
        rows.append({"phase": phase, "tag": tag, "salt": salt, "t_start": t0, "ttft_s": ttft,
                     "wall_s": time.time() - t0, "error": err,
                     "prompt_tokens": (usage or {}).get("prompt_tokens"),
                     "completion_tokens": (usage or {}).get("completion_tokens"),
                     "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "text": "".join(text)})


def run_parallel(jobs, conc):
    it, it_lock = iter(jobs), threading.Lock()

    def worker():
        while True:
            with it_lock:
                job = next(it, None)
            if job is None:
                return
            fire(*job)

    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


fill_jobs = [(body_text(f"F{i}", items), f"fill-{i}", f"ns-f{i}", "fill") for i in range(a.fill)]
issued = [(p, t, s) for p, t, s, _ in fill_jobs]  # issue order, the repeat pool
rng = random.Random(a.seed)
seq = {"n": 0}
seq_lock = threading.Lock()


def next_job():
    with seq_lock:
        seq["n"] += 1
        n = seq["n"]
        if n % a.repeat_every == 0 and len(issued) > a.back_min:
            lo = max(0, len(issued) - a.back_max)
            hi = len(issued) - a.back_min
            p, t, s = issued[rng.randint(lo, hi)]
            return (p, f"repeat-{n}-of-{t}", s, "steady-repeat")
        tag = f"S{n}"
        job = (body_text(tag, items), f"new-{n}", f"ns-s{n}")
        issued.append(job)
        return (*job, "steady-new")


marks = {"m0": metrics(), "t0": time.time()}
run_parallel(fill_jobs, a.fill_conc)
marks["m_fill"], marks["t_fill"] = metrics(), time.time()
deadline = marks["t_fill"] + a.duration


def steady_worker():
    while time.time() < deadline:
        fire(*next_job())


ts = [threading.Thread(target=steady_worker) for _ in range(a.workers)]
for t in ts:
    t.start()
for t in ts:
    t.join()
marks["m_steady"], marks["t_steady"] = metrics(), time.time()


def phase_summary(rs):
    ttft = sorted(1000 * r["ttft_s"] for r in rs if r["ttft_s"])
    q = lambda f: round(ttft[min(len(ttft) - 1, int(f * len(ttft)))], 1) if ttft else None
    return {"rows": len(rs), "errors": sum(1 for r in rs if r["error"]),
            "hits": sum(1 for r in rs if (r["cached"] or 0) > 0),
            "cached_tokens": sum(r["cached"] or 0 for r in rs),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
            "ttft_ms_p50": q(0.5), "ttft_ms_p95": q(0.95), "ttft_ms_max": round(ttft[-1], 1) if ttft else None}


def delta(m1, m0):
    return {k: m1.get(k, 0) - m0.get(k, 0) for k in m1 if m1.get(k, 0) != m0.get(k, 0)}


steady = [r for r in rows if r["phase"].startswith("steady")]
win = 300.0
windows = []
for w in range(int(a.duration // win) + 1):
    lo, hi = marks["t_fill"] + w * win, marks["t_fill"] + (w + 1) * win
    rs = [r for r in steady if r["phase"] == "steady-repeat" and lo <= r["t_start"] < hi]
    if rs:
        windows.append({"from_s": w * win, "repeats": len(rs), "hits": sum(1 for r in rs if (r["cached"] or 0) > 0)})
summary = {"items_per_prompt": items, "tokens_per_item": round(per_item, 3), "workers": a.workers,
           "duration_s": a.duration, "repeat_every": a.repeat_every, "back": [a.back_min, a.back_max],
           "fill": phase_summary([r for r in rows if r["phase"] == "fill"]),
           "steady_new": phase_summary([r for r in rows if r["phase"] == "steady-new"]),
           "steady_repeat": phase_summary([r for r in rows if r["phase"] == "steady-repeat"]),
           "repeat_windows_300s": windows,
           "phase_s": {"fill": round(marks["t_fill"] - marks["t0"], 1), "steady": round(marks["t_steady"] - marks["t_fill"], 1)},
           "metrics_delta": {"fill": delta(marks["m_fill"], marks["m0"]), "steady": delta(marks["m_steady"], marks["m_fill"])},
           "metrics_end": marks["m_steady"]}
json.dump({"rows": rows, "summary": summary, "marks_t": {k: v for k, v in marks.items() if k.startswith("t")}},
          open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k not in ("metrics_end",)}))
