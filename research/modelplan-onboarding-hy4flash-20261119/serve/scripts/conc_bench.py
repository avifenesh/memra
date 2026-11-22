#!/usr/bin/env python3
"""Concurrency readings of the hy_v4_flash route through memra-server (a measurement).

usage: conc_bench.py ADDR API_KEY MODEL REQUESTS_DIR OUT_JSONL --server-log PATH
                     [--conc 1,2,4,8] [--runs 3] [--work long,chat]

Every request is sampled at the artifact's generation_config (temperature 0.9, no top-k, no
top-p). Request i of a sample carries seed 20261121 + i, so the requests of one sample generate
different texts, and request i's text must equal what the same request generated alone
(`identity` below): the c=1 sample runs every seed the larger samples use.

Workloads (one warm-up, then RUNS samples each, concurrency c):
  long   c copies of the long probe's 2,104-token prompt (`prompt_ids`, /v1/completions),
         64 tokens each, `ignore_eos`, streamed one event a token
  chat   c of the six real chat turns (cycled; /v1/chat/completions, streamed), each up to 256
         tokens to its own end

Per request: TTFT, decode tokens/s (tokens after the first over the time after the first), e2e,
the sha256 of its text. Per sample: aggregate output tokens/s over the wall; the decode-phase
aggregate (tokens all requests emitted while every request was past its first token and none had
ended, over that window); from the server log's `[hy4f-step]` lines in the sample's window: the
steps, step time p50/p95 (all steps and decode-only steps), decode rows a step, and on the
decode-only steps the forward calls a step (logged as `calls=`, else one a running request) and
the decode rows a forward call.
"""
import concurrent.futures
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.request

argv = sys.argv[1:]


def opt(name, default):
    if name in argv:
        at = argv.index(name)
        v = argv[at + 1]
        del argv[at:at + 2]
        return v
    return default


SERVER_LOG = opt("--server-log", None)
CONC = [int(x) for x in opt("--conc", "1,2,4,8").split(",")]
RUNS = int(opt("--runs", "3"))
WORK = opt("--work", "long,chat").split(",")
addr, key, model, req_dir, out_path = argv[:5]
SEED = 20261121
TEMP = 0.9
STEP = re.compile(r"\[hy4f-step\] step=(\d+) prefill_rows=(\d+) decode_rows=(\d+)(?: calls=(\d+))? "
                  r"step_s=([0-9.]+) running=(\d+)")
DRAFT = re.compile(r"\[hy4f-draft\].* rounds=(\d+) drafted=(\d+) accepted=(\d+) tokens=(\d+)")
LONG = json.load(open(f"{req_dir}/long.json"))
LONG_IDS = LONG["token_ids"][: LONG["decode_from"]]
TURNS = [
    ([{"role": "user", "content": "What is the capital of Japan, and what is it known for? Two sentences."}],
     "no_think"),
    ([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number "
                                  "iteratively, with a docstring."}], "no_think"),
    ([{"role": "user", "content": "Explain in a short paragraph why the sky is blue."}], "no_think"),
    ([{"role": "user", "content": "List five tips for writing clear technical documentation."}], "no_think"),
    ([{"role": "user", "content": "A train leaves at 9:40 and arrives at 13:05. How long is the trip?"}],
     "high"),
    ([{"role": "user", "content": "Translate to French: The meeting is moved to Thursday at noon."}],
     "no_think"),
]


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


def log_since(offset):
    if offset is None:
        return ""
    with open(SERVER_LOG, "rb") as f:
        f.seek(offset)
        return f.read().decode(errors="replace")


def pct(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def steps_of(text):
    st = [m.groups() for m in STEP.finditer(text)]
    rows = [(int(p), int(d), int(c) if c is not None else None, float(s), int(r))
            for _, p, d, c, s, r in st]
    alls = [r[3] for r in rows]
    dec = [r for r in rows if r[0] == 0 and r[1] > 0]
    calls = [r[2] for r in rows if r[2] is not None and r[1] > 0]
    # forward calls of a decode-only step: logged (`calls=`), else one a running request (a
    # server that runs one call a request a step)
    dcalls = [r[2] if r[2] is not None else r[4] for r in dec]
    got = [tuple(int(x) for x in m.groups()) for m in DRAFT.finditer(text)]
    rounds, drafted, accepted = (sum(g[i] for g in got) for i in range(3))
    return {
        "steps": len(rows),
        "step_s_p50": pct(alls, 0.5), "step_s_p95": pct(alls, 0.95),
        "decode_steps": len(dec),
        "decode_step_s_p50": pct([r[3] for r in dec], 0.5),
        "decode_step_s_p95": pct([r[3] for r in dec], 0.95),
        "decode_rows_per_step": statistics.mean([r[1] for r in dec]) if dec else None,
        "calls_logged": bool(calls),
        "calls_per_step": statistics.mean(calls) if calls else None,
        "rows_per_call": (sum(r[1] + r[0] for r in rows if r[2]) / sum(r[2] for r in rows if r[2]))
        if calls else None,
        "decode_calls_per_step": statistics.mean(dcalls) if dcalls else None,
        "decode_rows_per_call": sum(r[1] for r in dec) / sum(dcalls) if dcalls and sum(dcalls) else None,
        "drafts": {"rounds": rounds, "drafted": drafted, "accepted": accepted},
    }


def stream(path, body):
    req = urllib.request.Request(
        f"http://{addr}{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    stamps, text, done = [], [], None
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
            piece = None
            if "choices" in j:
                ch = (j["choices"] or [{}])[0]
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning") or d.get("reasoning_content") or "")
                if ch.get("finish_reason"):
                    done = j
                if j.get("usage"):
                    done = j
                if not piece:
                    piece = None
            elif "stop_reason" in j or "n_tokens" in j:
                done = j
            elif "text" in j:
                piece = j["text"]
            if piece is not None:
                stamps.append(time.time())
                text.append(piece)
    return t0, time.time(), stamps, "".join(text), done


def one_long(i, n=64):
    body = {"model": model, "prompt_ids": LONG_IDS, "max_tokens": n, "seed": SEED + i,
            "ignore_eos": True, "stream": True, "temperature": TEMP}
    t0, t1, stamps, text, done = stream("/v1/completions", body)
    if done is None or done.get("n_tokens") != n or len(stamps) != n:
        raise RuntimeError(f"long request {i}: {len(stamps)} of {n} tokens, terminal {done}")
    return t0, t1, stamps, text, n


def one_chat(i):
    messages, effort = TURNS[i % len(TURNS)]
    body = {"model": model, "messages": messages, "max_tokens": 256, "seed": SEED + i,
            "temperature": TEMP, "reasoning_effort": effort, "stream": True,
            "stream_options": {"include_usage": True}}
    t0, t1, stamps, text, done = stream("/v1/chat/completions", body)
    n = ((done or {}).get("usage") or {}).get("completion_tokens") or len(stamps)
    return t0, t1, stamps, text, n


def sample(work, c):
    fn = one_long if work == "long" else one_chat
    before, mark = gpu(), log_size()
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(c) as ex:
        res = [f.result() for f in [ex.submit(fn, i) for i in range(c)]]
    wall = time.time() - t0
    reqs = []
    for i, (a, b, stamps, text, n) in enumerate(res):
        dec = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else None
        reqs.append({"i": i, "seed": SEED + i, "tokens": n, "events": len(stamps),
                     "ttft_s": stamps[0] - a if stamps else None, "decode_tok_s": dec,
                     "e2e_s": b - a, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    # the decode phase: every request past its first token, none ended
    lo = max(r[2][0] for r in res if r[2])
    hi = min(r[2][-1] for r in res if r[2])
    in_win = sum(sum(1 for s in r[2] if lo < s <= hi) for r in res)
    phase = in_win / (hi - lo) if hi > lo else None
    return {"workload": work, "concurrency": c, "requests": reqs, "wall_s": wall,
            "aggregate_tok_s": sum(r["tokens"] for r in reqs) / wall,
            "decode_phase": {"window_s": hi - lo, "tokens": in_win, "agg_tok_s": phase},
            "server": steps_of(log_since(mark)), "gpu_before": before, "gpu_after": gpu()}


ref = {}
ok = True
with open(out_path, "w") as out:
    for work in WORK:
        # the references: every request of the largest sample, each alone
        fn = one_long if work == "long" else one_chat
        for i in range(max(CONC)):
            ref[(work, i)] = hashlib.sha256(fn(i)[3].encode()).hexdigest()
        out.write(json.dumps({"workload": work, "references": {str(i): ref[(work, i)]
                                                                for i in range(max(CONC))}}) + "\n")
        for c in CONC:
            for run in range(RUNS + 1):
                row = sample(work, c)
                row["run"] = run
                row["warmup"] = run == 0
                for r in row["requests"]:
                    r["identity"] = ref[(work, r["i"])] == r["text_sha256"]
                    ok &= r["identity"]
                out.write(json.dumps(row) + "\n")
                out.flush()
                sv = row["server"]
                print(work, f"c={c}", run, f"wall {row['wall_s']:.1f}s agg {row['aggregate_tok_s']:.2f}",
                      f"phase {row['decode_phase']['agg_tok_s'] or 0:.2f}",
                      "per-req", [round(r["decode_tok_s"] or 0, 2) for r in row["requests"]],
                      f"dstep p50 {sv['decode_step_s_p50']} p95 {sv['decode_step_s_p95']}",
                      f"rows/step {sv['decode_rows_per_step']} rows/call {sv['decode_rows_per_call']}",
                      "ident", all(r["identity"] for r in row["requests"]), flush=True)
print("CONC-BENCH", "IDENTITY-PASS" if ok else "IDENTITY-FAIL")
sys.exit(0 if ok else 1)
