"""The prefix cache at its limit on a discrete card: fill it, fill the parked-session pool, then put the
interactive cap under long unique prompts, and see what a defer does to the cache.

Three phases, every request greedy and streamed (TTFT per row), each in its own `cache_salt` namespace so that
finished sessions park (2 per namespace, 16 process-wide by default):

  fill     F unique prompts of ~TOK tokens, FILL_CONC at a time: each leaves one prefix entry and parks its
           session, so the cache and the parked pool both fill
  pressure P unique prompts of ~TOK tokens, all at once (the interactive cap): sessions that the session
           peak counts, beside a full cache and a full parked pool, so admission may defer and reclaim
  repeat   R of the fill prompts again, same namespaces, sequential: what the cache kept

  python3 limit.py PORT OUT.json [--tok 24000] [--fill 64] [--fill-conc 4] [--pressure 12] [--repeat 16]
                                 [--otoks 64]

Prometheus prefix-cache, eviction and reclaim counters are read before and after each phase.
"""
import argparse, json, re, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("out")
ap.add_argument("--tok", type=int, default=24000)
ap.add_argument("--fill", type=int, default=64)
ap.add_argument("--fill-conc", type=int, default=4)
ap.add_argument("--pressure", type=int, default=12)
ap.add_argument("--repeat", type=int, default=16)
ap.add_argument("--otoks", type=int, default=64)
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
press_jobs = [(body_text(f"P{i}", items), f"pressure-{i}", f"ns-p{i}", "pressure") for i in range(a.pressure)]
# repeat the most recent fill prompts first: the ones an LRU-ordered cache would still hold
rep_idx = list(range(a.fill - 1, a.fill - 1 - a.repeat, -1))
rep_jobs = [(fill_jobs[i][0], f"repeat-{i}", fill_jobs[i][2], "repeat") for i in rep_idx]

marks = {"m0": metrics(), "t0": time.time()}
run_parallel(fill_jobs, a.fill_conc)
marks["m_fill"], marks["t_fill"] = metrics(), time.time()
run_parallel(press_jobs, len(press_jobs))
marks["m_press"], marks["t_press"] = metrics(), time.time()
run_parallel(rep_jobs, 1)
marks["m_rep"], marks["t_rep"] = metrics(), time.time()


def phase_summary(name):
    rs = [r for r in rows if r["phase"] == name]
    ttft = sorted(1000 * r["ttft_s"] for r in rs if r["ttft_s"])
    return {"rows": len(rs), "errors": sum(1 for r in rs if r["error"]),
            "hits": sum(1 for r in rs if (r["cached"] or 0) > 0),
            "cached_tokens": sum(r["cached"] or 0 for r in rs),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
            "ttft_ms_p50": round(ttft[len(ttft) // 2], 1) if ttft else None,
            "ttft_ms_max": round(ttft[-1], 1) if ttft else None}


def delta(m1, m0):
    return {k: m1.get(k, 0) - m0.get(k, 0) for k in m1 if m1.get(k, 0) != m0.get(k, 0)}


summary = {"items_per_prompt": items, "tokens_per_item": round(per_item, 3),
           "fill": phase_summary("fill"), "pressure": phase_summary("pressure"), "repeat": phase_summary("repeat"),
           "phase_s": {"fill": round(marks["t_fill"] - marks["t0"], 1), "pressure": round(marks["t_press"] - marks["t_fill"], 1),
                       "repeat": round(marks["t_rep"] - marks["t_press"], 1)},
           "metrics_delta": {"fill": delta(marks["m_fill"], marks["m0"]), "pressure": delta(marks["m_press"], marks["m_fill"]),
                             "repeat": delta(marks["m_rep"], marks["m_press"])},
           "metrics_after_fill": marks["m_fill"], "metrics_end": marks["m_rep"]}
json.dump({"rows": rows, "summary": summary, "marks_t": {k: v for k, v in marks.items() if k.startswith("t")}},
          open(a.out, "w"), indent=1)
print(json.dumps(summary))
