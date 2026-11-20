#!/usr/bin/env python3
"""Readings of the hy_v4_flash route through memra-server (a measurement, not a tuned claim).

usage: bench.py ADDR API_KEY MODEL REQUESTS_DIR OUT_JSONL [RUNS]

Streams /v1/completions in memra's own shape (one event a token) with `prompt_ids`, greedy and
`ignore_eos`, so every sample generates its full budget. One warm-up, then RUNS (default 3)
samples a workload:
  chat_c1    the chat probe's 36-token prompt, 128 tokens, alone
  long_c1    the long probe's 2,104-token prompt, 64 tokens, alone
  chat_c4    four chat-probe requests at once, 128 tokens each
  long_c4    four long-probe requests at once, 32 tokens each (their prefills run one chunk a step)
Per request: TTFT (request to first token), prefill tokens/s (prompt / TTFT), decode tokens/s
(tokens after the first over the time after the first); per sample: the aggregate output
tokens/s over the wall time and the GPU's temperature, power and SM clock before and after."""
import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.request

addr, key, model, req_dir, out_path = sys.argv[1:6]
RUNS = int(sys.argv[6]) if len(sys.argv) > 6 else 3
REQ = {n: json.load(open(f"{req_dir}/{n}.json")) for n in ("chat", "long")}
PROMPT = {n: r["token_ids"][: r["decode_from"]] for n, r in REQ.items()}


def gpu():
    q = "temperature.gpu,power.draw,clocks.sm"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        t, p, c = [x.strip() for x in out.split(",")]
        return {"temp_c": t, "power_w": p, "sm_mhz": c}
    except Exception as e:  # a reading, never a failure
        return {"error": str(e)}


def one(ids, n):
    body = {"model": model, "prompt_ids": ids, "max_tokens": n, "temperature": 0,
            "ignore_eos": True, "stream": True}
    req = urllib.request.Request(
        f"http://{addr}/v1/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    stamps = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if "text" in j and "stop_reason" not in j and "n_tokens" not in j:
                stamps.append(time.time())
    t1 = time.time()
    n_out = len(stamps)
    ttft = stamps[0] - t0 if stamps else None
    dec = (n_out - 1) / (stamps[-1] - stamps[0]) if n_out > 1 and stamps[-1] > stamps[0] else None
    return {"prompt_tokens": len(ids), "tokens": n_out, "ttft_s": ttft,
            "prefill_tok_s": len(ids) / ttft if ttft else None, "decode_tok_s": dec,
            "e2e_s": t1 - t0}


WORK = [("chat_c1", "chat", 1, 128), ("long_c1", "long", 1, 64),
        ("chat_c4", "chat", 4, 128), ("long_c4", "long", 4, 32)]
with open(out_path, "w") as out:
    for name, p, c, n in WORK:
        for run in range(RUNS + 1):
            before = gpu()
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(c) as ex:
                reqs = [f.result() for f in [ex.submit(one, PROMPT[p], n) for _ in range(c)]]
            wall = time.time() - t0
            row = {"workload": name, "run": run, "warmup": run == 0, "concurrency": c,
                   "requests": reqs, "wall_s": wall,
                   "aggregate_tok_s": sum(r["tokens"] for r in reqs) / wall,
                   "gpu_before": before, "gpu_after": gpu()}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(name, run, f"wall {wall:.1f}s agg {row['aggregate_tok_s']:.2f} tok/s",
                  [(round(r["ttft_s"] or 0, 2), round(r["decode_tok_s"] or 0, 2)) for r in reqs],
                  flush=True)
print("BENCH-DONE")
