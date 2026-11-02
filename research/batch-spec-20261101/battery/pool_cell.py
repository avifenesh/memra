#!/usr/bin/env python3
"""Pooled-vs-serial served-stream cell (lane/batch-spec step 2).

usage: pool_cell.py BIN OUTDIR [PORT]

One fresh memra-server boot (q9 + owntrim draft, MEMRA_CTX=4096, MEMRA_SPEC_GATE=0 so every
session speculates at every concurrency). Waves of c = 2, 4, 8 concurrent requests, distinct
prompts, alternating greedy and sampled (T=0.8, fixed per-request seeds), max_tokens 200.
Writes OUTDIR/summary.json: per (c, request) sha256 of reasoning\\0content, finish reason,
completion tokens, and the wave wall time; plus the server's [spec-pool] lines.
Run it once on the serial binary and once on the pooled binary; every stream must match.
"""
import hashlib, json, os, re, subprocess, sys, threading, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8351
BASE = f"http://127.0.0.1:{PORT}"
Q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
DRAFT = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
os.makedirs(OUT, exist_ok=True)
TOPICS = ["a lighthouse keeper", "a clockmaker's apprentice", "a desert caravan", "a deep-sea diver",
          "a mountain observatory", "a river ferry", "a city of bridges", "an orbital greenhouse"]


def prompt(c, j):
    return (f"Wave {c}, writer {j}: write a vivid short story about {TOPICS[j % len(TOPICS)]}, "
            f"then list three lessons from it.")


def post(c, j, res):
    body = {"model": "q9", "messages": [{"role": "user", "content": prompt(c, j)}],
            "max_tokens": 200, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if j % 2 == 0:
        body["temperature"] = 0.0
    else:
        body["temperature"] = 0.8
        body["seed"] = 1000 + 17 * j + c
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        res[j] = json.loads(r.read())


log = os.path.join(OUT, "server.log")
env = dict(os.environ)
env.update({"MEMRA_MODELS": f"q9={Q9}+{DRAFT}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
            "MEMRA_CTX": "4096", "MEMRA_SPEC_GATE": "0",
            # Fixed burst boundaries: a sampled stream draws its boundary token from the session
            # Philox stream at every burst end, so arrival-timed early burst ends (admission
            # yield) would make the reference itself timing-dependent.
            "MEMRA_ADMIT_YIELD": "0"})
with open(log, "w") as lf:
    p = subprocess.Popen([BIN], env=env, stdout=lf, stderr=subprocess.STDOUT)
summary = {"bin": BIN}
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
            break
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"server exited rc={p.returncode}; see {log}")
            time.sleep(1)
    for c in (2, 4, 8):
        res = {}
        th = [threading.Thread(target=post, args=(c, j, res)) for j in range(c)]
        t0 = time.time()
        # Staggered arrivals: each request primes alone (a cross-request batched prime is a
        # different numeric class, so simultaneous arrivals make the reference itself
        # timing-dependent); the bursts still overlap.
        for t in th:
            t.start()
            time.sleep(0.25)
        for t in th:
            t.join()
        wall = time.time() - t0
        toks = 0
        for j in range(c):
            r = res[j]
            m = r["choices"][0]["message"]
            text = (m.get("reasoning") or m.get("reasoning_content") or "") + "\0" + (m.get("content") or "")
            u = r.get("usage", {})
            toks += u.get("completion_tokens", 0)
            summary[f"c{c}/r{j}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                                     "finish": r["choices"][0].get("finish_reason"),
                                     "completion_tokens": u.get("completion_tokens")}
            with open(os.path.join(OUT, f"c{c}-r{j}.json"), "w") as f:
                json.dump(r, f, indent=1)
        summary[f"c{c}/wall_s"] = round(wall, 3)
        summary[f"c{c}/agg_tok_s"] = round(toks / wall, 2)
finally:
    p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
summary["spec_pool_lines"] = [l.strip() for l in open(log) if "[spec-pool]" in l]
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if "/" in k and ("wall" in k or "agg" in k)}))
print("\n".join(summary["spec_pool_lines"]))
