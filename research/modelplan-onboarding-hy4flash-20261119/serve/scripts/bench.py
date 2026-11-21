#!/usr/bin/env python3
"""Readings of the hy_v4_flash route through memra-server (a measurement, not a tuned claim).

usage: bench.py ADDR API_KEY MODEL REQUESTS_DIR OUT_JSONL [RUNS] [--server-log PATH]

Streams /v1/completions in memra's own shape (one event a token) with `prompt_ids`, sampled at
the artifact's `generation_config.json` (temperature 0.9, no top-k, no top-p) with seed 20261121
and `ignore_eos`, so every sample generates its full budget. Speed is not read on greedy
decoding. Every request carries the same seed, so two servers of one binary, drafting or not,
generate the same tokens: each request's row keeps the sha256 of its text for that check
(`bench_identity.py`). With `--server-log`, each sample also keeps the draft rounds, drafts and
kept drafts the server logged for its requests (`[hy4f-draft]` lines). One warm-up, then RUNS
(default 3) samples a workload:
  long_c1    the long probe's 2,104-token prompt, 64 tokens, alone
  long_c4    four long-probe requests at once, 32 tokens each (their prefills run one chunk a step)
The chat probe is not a workload here: its first sampled token is the end token, so a run past
it generates no answer. Chat speed is read on real turns (`real_turns.py`).
Per request: TTFT (request to first token), prefill tokens/s (prompt / TTFT), decode tokens/s
(tokens after the first over the time after the first); per sample: the aggregate output
tokens/s over the wall time and the GPU's temperature, power and SM clock before and after."""
import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request

argv = sys.argv[1:]
SERVER_LOG = None
if "--server-log" in argv:
    at = argv.index("--server-log")
    SERVER_LOG = argv[at + 1]
    del argv[at:at + 2]
addr, key, model, req_dir, out_path = argv[:5]
RUNS = int(argv[5]) if len(argv) > 5 else 3
SAMPLING = {"temperature": 0.9}
SEED = 20261121
DRAFT = re.compile(r"\[hy4f-draft\].* rounds=(\d+) drafted=(\d+) accepted=(\d+) tokens=(\d+)")
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


def log_size():
    try:
        with open(SERVER_LOG, "rb") as f:
            return f.seek(0, 2)
    except (OSError, TypeError):
        return None


def drafts_since(offset):
    """The draft usage the server logged since `offset`: one line a request that drafted."""
    if offset is None:
        return None
    with open(SERVER_LOG, "rb") as f:
        f.seek(offset)
        text = f.read().decode(errors="replace")
    got = [tuple(int(x) for x in m.groups()) for m in DRAFT.finditer(text)]
    rounds, drafted, accepted = (sum(g[i] for g in got) for i in range(3))
    return {"requests": len(got), "rounds": rounds, "drafted": drafted, "accepted": accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "kept_per_round": accepted / rounds if rounds else None}


def one(ids, n, seed):
    body = {"model": model, "prompt_ids": ids, "max_tokens": n, "seed": seed,
            "ignore_eos": True, "stream": True} | SAMPLING
    req = urllib.request.Request(
        f"http://{addr}/v1/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    stamps = []
    text = hashlib.sha256()
    done = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}")
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if "error" in j:
                raise RuntimeError(f"in-band error: {j['error']}")
            if "stop_reason" in j or "n_tokens" in j:
                done = j
            elif "text" in j:
                stamps.append(time.time())
                text.update(j["text"].encode())
    t1 = time.time()
    n_out = len(stamps)
    # a sample counts only when the stream ended with its terminal event at the full budget
    if done is None or done.get("n_tokens") != n or n_out != n or done.get("stop_reason") != "MaxNew":
        raise RuntimeError(f"stream ended at {n_out} of {n} tokens, terminal event {done}")
    ttft = stamps[0] - t0 if stamps else None
    dec = (n_out - 1) / (stamps[-1] - stamps[0]) if n_out > 1 and stamps[-1] > stamps[0] else None
    return {"prompt_tokens": len(ids), "tokens": n_out, "seed": seed, "text_sha256": text.hexdigest(),
            "ttft_s": ttft,
            "prefill_tok_s": len(ids) / ttft if ttft else None, "decode_tok_s": dec,
            "e2e_s": t1 - t0}


WORK = [("long_c1", "long", 1, 64), ("long_c4", "long", 4, 32)]
with open(out_path, "w") as out:
    for name, p, c, n in WORK:
        for run in range(RUNS + 1):
            before = gpu()
            mark = log_size()
            t0 = time.time()
            with concurrent.futures.ThreadPoolExecutor(c) as ex:
                reqs = [f.result() for f in [ex.submit(one, PROMPT[p], n, SEED) for _ in range(c)]]
            wall = time.time() - t0
            row = {"workload": name, "run": run, "warmup": run == 0, "concurrency": c,
                   "sampling": SAMPLING, "requests": reqs, "wall_s": wall, "drafts": drafts_since(mark),
                   "aggregate_tok_s": sum(r["tokens"] for r in reqs) / wall,
                   "gpu_before": before, "gpu_after": gpu()}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(name, run, f"wall {wall:.1f}s agg {row['aggregate_tok_s']:.2f} tok/s",
                  [(round(r["ttft_s"] or 0, 2), round(r["decode_tok_s"] or 0, 2)) for r in reqs],
                  "drafts", row["drafts"] and (row["drafts"]["rounds"], row["drafts"]["drafted"],
                                               row["drafts"]["accepted"]), flush=True)
print("BENCH-DONE")
