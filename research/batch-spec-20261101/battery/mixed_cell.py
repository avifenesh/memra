#!/usr/bin/env python3
"""Mixed spec + plain served-stream cell (lane/batch-spec v2, the unified step).

usage: mixed_cell.py BIN OUTDIR [PORT] [demote]

With `demote`, `MEMRA_SPEC_DEMOTE_AT=40` also hands each spec session to plain decode after 40
committed tokens (exact Philox handoff for sampled rows), so demoted rows ride next to the
other spec session.

One fresh memra-server boot (q9 + owntrim draft, MEMRA_CTX=4096). The band gate with explicit
LOW=2 / HIGH=16 (`MEMRA_SPEC_GATE_MODEL=0`): the first two requests of a wave are admitted to
MTP spec, every later one to plain decode, and nothing demotes. Waves of c = 3, 4, 6, 8, staggered
0.25 s, `MEMRA_ADMIT_YIELD=0` (fixed burst boundaries), 200 tokens each. Request kinds by index:

  r0 greedy (spec)          r1 T=0.8 seeded (spec)
  r2 greedy (plain)         r3 T=0.8 seeded (plain)
  r4 T=0.7 top_k 40 top_p 0.9 (plain, filtered device draw)
  r5 T=0.8 presence 1.5 (plain, device penalties)
  r6 greedy + frequency 0.5 (plain, host-sampled: stays in phase (c))
  r7 greedy json_schema (plain, constrained: stays in phase (c))

Writes OUTDIR/summary.json: per (c, request) sha256 of reasoning\\0content, finish reason,
completion tokens and end-to-end seconds; per-wave wall time and aggregate tok/s; the server's [spec-pool] and
[spec-gate] lines. Run on the reference and the candidate binary; every stream must
match.
"""
import hashlib, json, os, subprocess, sys, threading, time, urllib.request

BIN, OUT = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8353
DEMOTE = len(sys.argv) > 4 and sys.argv[4] == "demote"
BASE = f"http://127.0.0.1:{PORT}"
Q9 = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"
DRAFT = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/draft-9b-owntrim-nvfp4head-q4blk.gguf"
os.makedirs(OUT, exist_ok=True)
TOPICS = ["a lighthouse keeper", "a clockmaker's apprentice", "a desert caravan", "a deep-sea diver",
          "a mountain observatory", "a river ferry", "a city of bridges", "an orbital greenhouse"]
SCHEMA = {"type": "object",
          "properties": {"name": {"type": "string"}, "year": {"type": "integer"},
                         "ships": {"type": "array", "items": {"type": "integer"}}},
          "required": ["name", "year", "ships"]}


def body(c, j):
    b = {"model": "q9",
         "messages": [{"role": "user", "content":
                       f"Wave {c}, writer {j}: write a vivid short story about "
                       f"{TOPICS[j % len(TOPICS)]}, then list three lessons from it."}],
         "max_tokens": 200, "stream": False, "temperature": 0.0,
         "chat_template_kwargs": {"enable_thinking": False}}
    seed = 2000 + 17 * j + c
    if j in (1, 3):
        b.update(temperature=0.8, seed=seed)
    elif j == 4:
        b.update(temperature=0.7, top_k=40, top_p=0.9, seed=seed)
    elif j == 5:
        b.update(temperature=0.8, presence_penalty=1.5, seed=seed)
    elif j == 6:
        b.update(frequency_penalty=0.5)
    elif j == 7:
        b["messages"][0]["content"] = (f"Wave {c}: give a JSON record for {TOPICS[j % 8]}: name, "
                                       f"year, and ships seen on five days.")
        b["max_tokens"] = 160
        b["response_format"] = {"type": "json_schema",
                                "json_schema": {"name": "rec", "schema": SCHEMA}}
    return b


def post(c, j, res):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body(c, j)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res[j] = json.loads(r.read())
    res[j]["_e2e_s"] = round(time.time() - t0, 3)


log = os.path.join(OUT, "server.log")
env = dict(os.environ)
env.update({"MEMRA_MODELS": f"q9={Q9}+{DRAFT}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
            "MEMRA_CTX": "4096", "MEMRA_SPEC_GATE_MODEL": "0",
            "MEMRA_SPEC_GATE_LOW": "2", "MEMRA_SPEC_GATE_HIGH": "16",
            "MEMRA_ADMIT_YIELD": "0"})
if DEMOTE:
    env["MEMRA_SPEC_DEMOTE_AT"] = "40"
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
    for c in (3, 4, 6, 8):
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
                                     "e2e_s": r.get("_e2e_s")}
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
summary["spec_gate_lines"] = [l.strip() for l in lines if "[spec-gate]" in l][:40]
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: v for k, v in summary.items() if "/" in k and ("wall" in k or "agg" in k)}))
print("\n".join(summary["spec_pool_lines"]))
