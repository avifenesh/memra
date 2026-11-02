#!/usr/bin/env python3
"""Pooled-vs-serial served-stream cell for the DFlash2 (dspark) route (lane/batch-spec step 5).

usage: dspark_pool_cell.py BIN OUTDIR [PORT]

One fresh memra-server boot: Qwen3.8-27B NVFP4 GGUF with the DFlash2 drafter
(MEMRA_DSPARK_SPEC=1, MEMRA_DSPARK_DRAFT), MEMRA_MTP_SKIP=1 (the owner rule: with DFlash as the
drafter the MTP head is not loaded), MEMRA_CTX=4096, the spec gate ON at LOW=8 / HIGH=16 so
every request of every wave rides the DFlash2 route (gate OFF admits dspark solo only),
MEMRA_ADMIT_YIELD=0. Waves of c = 1, 2, 4, 8 concurrent requests, distinct prompts, four arms
by request index (greedy; sampled T=0.8 with a fixed per-request seed; greedy + presence
penalty; sampled + frequency penalty + top_p), max_tokens 200, arrivals staggered 0.25 s (each
request primes alone; the bursts overlap).
Writes OUTDIR/summary.json: per (c, request) sha256 of reasoning\\0content, finish reason,
completion tokens, spec rounds (None = served plain), and the wave wall time; plus the server's [spec-pool] and [dspark-acc] lines.
Run it on the serial binary (twice: the reference must reproduce) and on the pooled binary;
every stream must match.
"""
import hashlib, json, os, subprocess, sys, threading, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8361
BASE = f"http://127.0.0.1:{PORT}"
Q38 = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DRAFT = "/data/ai-ml/models/q38-dflash2"
os.makedirs(OUT, exist_ok=True)
TOPICS = ["a lighthouse keeper", "a clockmaker's apprentice", "a desert caravan", "a deep-sea diver",
          "a mountain observatory", "a river ferry", "a city of bridges", "an orbital greenhouse"]
WAVES = (1, 2, 4, 8)


def prompt(c, j):
    return (f"Wave {c}, writer {j}: write a vivid short story about {TOPICS[j % len(TOPICS)]}, "
            f"then list three lessons from it.")


def post(c, j, res):
    body = {"model": "q38", "messages": [{"role": "user", "content": prompt(c, j)}],
            "max_tokens": 200, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    # Four arms by j % 4: greedy, sampled, greedy + presence penalty (the logits-verify
    # penalised-argmax arm), sampled + frequency penalty + top_p.
    if j % 2 == 0:
        body["temperature"] = 0.0
    else:
        body["temperature"] = 0.8
        body["seed"] = 1000 + 17 * j + c
    if j % 4 == 2:
        body["presence_penalty"] = 0.5
    if j % 4 == 3:
        body["frequency_penalty"] = 0.3
        body["top_p"] = 0.9
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res[j] = json.loads(r.read())


log = os.path.join(OUT, "server.log")
env = dict(os.environ)
env.update({"MEMRA_MODELS": f"q38={Q38}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
            "MEMRA_DSPARK_SPEC": "1", "MEMRA_DSPARK_DRAFT": DRAFT, "MEMRA_MTP_SKIP": "1",
            # Gate ON with a wide LOW band: with MEMRA_SPEC_GATE=0 the dspark admission is
            # solo-only (`dspark_load_admits_with`: gate off admits only at n_active == 0), so
            # no cohort could form. LOW=8 admits every row of the c=8 wave to the DFlash2
            # route, HIGH=16 keeps the demotion watermark out of reach.
            "MEMRA_CTX": "4096", "MEMRA_MAX_SESSIONS": "8",
            "MEMRA_SPEC_GATE_LOW": "8", "MEMRA_SPEC_GATE_HIGH": "16",
            # Fixed burst boundaries (the MTP pool cell's reproducibility condition).
            "MEMRA_ADMIT_YIELD": "0"})
with open(log, "w") as lf:
    p = subprocess.Popen([BIN], env=env, stdout=lf, stderr=subprocess.STDOUT)
summary = {"bin": BIN}
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=3).read()
            break
        except Exception:
            if p.poll() is not None:
                raise SystemExit(f"server exited rc={p.returncode}; see {log}")
            time.sleep(1)
    for c in WAVES:
        res = {}
        th = [threading.Thread(target=post, args=(c, j, res)) for j in range(c)]
        t0 = time.time()
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
                                     "completion_tokens": u.get("completion_tokens"),
                                     # spec usage: None = the request was served plain
                                     "spec_rounds": (u.get("spec") or {}).get("rounds")}
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
lines = open(log, errors="replace").read().splitlines()
summary["spec_pool_lines"] = [l.strip() for l in lines if "[spec-pool]" in l]
summary["dspark_acc_lines"] = len([l for l in lines if "[dspark-acc]" in l])
summary["dspark_engaged"] = any("DSPARK SPEC route armed" in l for l in lines)
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if "/" in k and ("wall" in k or "agg" in k)}))
print("\n".join(summary["spec_pool_lines"]))
