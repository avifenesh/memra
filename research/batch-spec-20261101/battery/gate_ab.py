#!/usr/bin/env python3
"""Spec-gate A/B (lane/batch-spec step 4): cost-model gate vs the LOW/HIGH band.

usage: gate_ab.py BIN OUTDIR [--reps 3] [--model-path GGUF+DRAFT] [--cs 1,2,4,8] [--tokens 256]

Arms, interleaved per rep in alternating order (ABBA...), one fresh boot per (arm, rep):
  band   MEMRA_SPEC_GATE_MODEL=0 (the LOW/HIGH band, the shipped policy)
  model  default (the cost-model gate)
Each boot: a warm-up request at c=1 (fills the acceptance window past 64 rounds), then waves
at each concurrency. Waves use distinct prompts and alternate greedy and sampled (T=0.7, seeded)
requests, all arriving together, with the shipped serving defaults (admission yield on).
Records aggregate tok/s per wave and the server's [spec-cost] / [spec-k] / [spec-gate] /
[spec-pool] lines. Medians per (arm, c) go to OUTDIR/summary.json.
"""
import argparse, json, os, statistics, subprocess, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("bin")
ap.add_argument("out")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--models", default=(
    "q9=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
    "+/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"))
ap.add_argument("--model-name", default="q9")
ap.add_argument("--cs", default="1,2,4,8")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--port", type=int, default=8361)
ap.add_argument("--ctx", default="8192")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"
os.makedirs(a.out, exist_ok=True)
CS = [int(c) for c in a.cs.split(",")]
TASKS = ["Explain how a hash map handles collisions, with a short Python example.",
         "Write a bash script that rotates log files older than 7 days and explain each line.",
         "Summarize the causes of the 2008 financial crisis in five bullet points, then critique them.",
         "Plan a three-step refactor of a 2,000-line Python module into packages; list risks.",
         "Derive the formula for the sum of a geometric series and give two applications.",
         "Write a SQL query for the top 3 customers by revenue per region, then explain the plan.",
         "Compare TCP congestion control algorithms Reno, Cubic and BBR in a short table.",
         "Draft a polite email declining a meeting and proposing two alternative times."]


def post(i, c, rep, res):
    body = {"model": a.model_name, "messages": [{"role": "user", "content":
            f"[{rep}.{c}.{i}] " + TASKS[i % len(TASKS)]}], "max_tokens": a.tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if i % 2:
        body.update({"temperature": 0.7, "seed": 7 + i})
    else:
        body["temperature"] = 0.0
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res[i] = json.loads(r.read())


def wave(c, rep):
    res = {}
    th = [threading.Thread(target=post, args=(i, c, rep, res)) for i in range(c)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    toks = sum(res[i].get("usage", {}).get("completion_tokens", 0) for i in range(c))
    return toks / wall


def boot(arm, rep):
    log = os.path.join(a.out, f"{arm}-r{rep}-server.log")
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": a.models, "MEMRA_ADDR": f"127.0.0.1:{a.port}", "MEMRA_CTX": a.ctx})
    if arm == "band":
        env["MEMRA_SPEC_GATE_MODEL"] = "0"
    with open(log, "w") as lf:
        p = subprocess.Popen([a.bin], env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
                break
            except Exception:
                if p.poll() is not None:
                    raise SystemExit(f"{arm} r{rep}: server exited rc={p.returncode}; see {log}")
                time.sleep(1)
        wave(1, f"warm{rep}")  # acceptance history for the cost model
        out = {c: wave(c, rep) for c in CS}
    finally:
        p.terminate()
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            p.kill()
    lines = [l.strip() for l in open(log)
             if any(t in l for t in ("[spec-cost]", "[spec-gate]", "[spec-pool]"))]
    return out, lines


results = {"band": {c: [] for c in CS}, "model": {c: [] for c in CS}}
logs = {}
for rep in range(a.reps):
    order = ["band", "model"] if rep % 2 == 0 else ["model", "band"]
    for arm in order:
        out, lines = boot(arm, rep)
        for c in CS:
            results[arm][c].append(round(out[c], 2))
        logs[f"{arm}-r{rep}"] = lines[:60]
        print(arm, rep, {c: round(out[c], 1) for c in CS}, flush=True)
summary = {"runs": results, "median": {arm: {c: statistics.median(v) for c, v in d.items()}
                                       for arm, d in results.items()}, "lines": logs}
with open(os.path.join(a.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
for c in CS:
    b, m = summary["median"]["band"][c], summary["median"]["model"][c]
    print(f"c={c}: band {b:.1f}  model {m:.1f}  ({m / b:.2f}x)")
