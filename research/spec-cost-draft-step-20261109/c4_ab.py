#!/usr/bin/env python3
"""Four requests at once, base against flip binary, on one RTX PRO 6000 Max-Q (memra#1093).

Derived from research/spec-k-cold-long-c2-20261109/c2_ab.py (its docstring follows). Here the arms
are two binaries at their device defaults (`base`, `flip`; the driver ab_bins.py passes each its
own), and every boot serves three cold groups of four requests sent at once: four T=1 prompts of
about 3,600 tokens, four of about 1,240 and four of 34 tokens. The cost model owns K on this card
at every concurrency, and the cost-model fixes move its table prices at 4 sessions; this cell
checks what the served rate does there.

Original docstring:
Cold-long K=2 vs K=3 at concurrency 2 on the RTX 5090 (memra#1093, before the cold-long flip).

The single-stream A/B (research/spec-k-cold-long-20261109/) found K=2 ahead of the cold-long K=3.
Spec serves c1 and c2 on a single card (the gate demotes from c3), and at c2 two sessions pool one
verify (2 x (K+1) rows), so K=2 must not lose there. Arms, one binary: `k3` the device default,
`k2` MEMRA_SPEC_K_COLD_LONG=2. One boot per call (the driver gives each its own lease). Every boot
serves the same cold prompt pairs, each pair's two requests sent at once, pairs one after another:

  t1-4k     two pairs of T=1 prompts of about 3,600 tokens
  t1-6k     one pair of T=1 prompts of about 5,800 tokens
  t1-1k     one pair of T=1 prompts of about 1,240 tokens
  greedy-4k one pair of greedy prompts of about 3,600 tokens (text must match across arms)
  short     one pair of T=1 prompts of 34 tokens: the cold-short rung, K=3 on both arms (drift)

Answers capped at 400 tokens (requests for about 300 words), thinking off, streamed. Per request:
the [spec-k] line, TTFT, decode rate after TTFT, ITL, acceptance, sha. Per pair: aggregate
completion tokens over the pair's wall. 250 ms GPU telemetry per boot.

usage: c2_ab.py OUT BINARY ARM   (BOOT_START=n; ARM k3 or k2) | c2_ab.py OUT - -  (summary only)
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN, ARM = sys.argv[2], sys.argv[3]
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("CELL_MODEL", "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf")
COMMON = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
          "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "4"}
ARMS = {"base": {}, "flip": {}}
KLINE = r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)"


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


def t1(seed):
    return {"temperature": 1.0, "seed": seed}


PAIRS = [  # (class, four (id, content, sampling)), the four sent at once
    ("t1-4k", [(f"{t}-4k", long_prompt(t, 130), t1(401 + i)) for i, t in enumerate(("rivers", "bridges", "glaciers", "harbors"))]),
    ("t1-1k", [(f"{t}-1k", long_prompt(t, 45), t1(411 + i)) for i, t in enumerate(("islands", "meadows", "deserts", "forests"))]),
    ("short", [(f"{t}-short", short_prompt(t), t1(421 + i)) for i, t in enumerate(("comet", "tide", "dune", "reef"))]),
]


def boot(log, env):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(env)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=3).read()
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise SystemExit("server not ready in 900 s")


def stop(p):
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def stream(body, out):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, text, usage = [], [], None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        times.append(time.perf_counter() - t0)
                        text.append(piece)
        out.update(times=times, total=time.perf_counter() - t0, text="".join(text), usage=usage, error=None)
    except Exception as exc:  # recorded, never swallowed
        out.update(times=times, total=time.perf_counter() - t0, text="".join(text), usage=usage, error=repr(exc))


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def run_boot(n):
    log = OUT / f"server-{n}-{ARM}.log"
    sel = ["-i", os.environ["TELEMETRY_GPU"]] if os.environ.get("TELEMETRY_GPU") else []
    tel = subprocess.Popen(["nvidia-smi", *sel, "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
                            "--format=csv", "-lms", "250"], stdout=open(OUT / f"telemetry-{n}-{ARM}.csv", "w"),
                           stderr=subprocess.DEVNULL)
    p = boot(log, dict(COMMON, **ARMS[ARM]))
    try:
        for pi, (cls, pair) in enumerate(PAIRS):
            outs = [{} for _ in pair]
            before = log.stat().st_size
            t0 = time.perf_counter()
            threads = []
            for (pid, content, samp), o in zip(pair, outs):
                body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": 400,
                        "chat_template_kwargs": {"enable_thinking": False}, **samp}
                th = threading.Thread(target=stream, args=(body, o))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
            wall = time.perf_counter() - t0
            time.sleep(0.3)
            with open(log, errors="replace") as f:
                f.seek(before)
                seg = f.read()
            klines = re.findall(KLINE, seg)
            pair_toks = 0
            for (pid, _, samp), o in zip(pair, outs):
                usage = o.get("usage") or {}
                toks = usage.get("completion_tokens") or 0
                pair_toks += toks
                times = o.get("times") or []
                ttft = times[0] if times else None
                gaps = [b - a for a, b in zip(times, times[1:])]
                ptoks = usage.get("prompt_tokens")
                k = next((kl for kl in klines if ptoks and int(kl[2]) == ptoks), None)
                k = k or next((kl for kl in klines if ptoks and abs(int(kl[2]) - ptoks) <= 1), None)
                r = {"boot": n, "arm": ARM, "pair": pi, "class": cls, "prompt": pid, "k": k,
                     "prompt_tokens": ptoks, "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     "completion_tokens": toks, "ttft_s": round(ttft, 4) if ttft else None,
                     "total_s": round(o.get("total", 0), 4),
                     "decode_tok_s": round(toks / (o["total"] - ttft), 2) if ttft and o["total"] > ttft and toks else None,
                     "itl_ms": {"p50": round(1e3 * pct(gaps, .5), 2), "p95": round(1e3 * pct(gaps, .95), 2),
                                "p99": round(1e3 * pct(gaps, .99), 2)} if gaps else None,
                     "acceptance": (usage.get("spec") or {}).get("acceptance_rate"),
                     "sha": hashlib.sha256((o.get("text") or "").encode()).hexdigest()[:16], "error": o.get("error")}
                with open(OUT / "runs.jsonl", "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(ARM, n, pid, r["k"], r["ttft_s"], r["decode_tok_s"], toks, r["error"], flush=True)
            with open(OUT / "pairs.jsonl", "a") as f:
                f.write(json.dumps({"boot": n, "arm": ARM, "pair": pi, "class": cls, "wall_s": round(wall, 4),
                                    "tokens": pair_toks, "agg_tok_s": round(pair_toks / wall, 2),
                                    "klines": klines}) + "\n")
    finally:
        stop(p)
        tel.terminate()
        try:
            tel.wait(timeout=15)
        except subprocess.TimeoutExpired:
            tel.kill()
            tel.wait()


def summarize():
    rows = [json.loads(l) for l in open(OUT / "runs.jsonl")]
    pairs = [json.loads(l) for l in open(OUT / "pairs.jsonl")]
    boots = sorted({r["boot"] for r in rows})
    arm_of = {r["boot"]: r["arm"] for r in rows}
    ref = {b: statistics.median([r["decode_tok_s"] for r in rows if r["boot"] == b and r["class"] == "short"
                                 and r["decode_tok_s"]]) for b in boots}

    def u(a, b):
        return sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)

    out = {"errors": [r for r in rows if r["error"]]}
    for cls in ("t1-4k", "t1-1k", "short"):
        per = {"base": [], "flip": []}
        for b in boots:
            sel = [r for r in rows if r["boot"] == b and r["class"] == cls and r["decode_tok_s"]]
            ps = [p for p in pairs if p["boot"] == b and p["class"] == cls]
            if not sel:
                continue
            per[arm_of[b]].append({"decode": statistics.median(r["decode_tok_s"] for r in sel),
                                   "norm": statistics.median(r["decode_tok_s"] / ref[b] for r in sel),
                                   "agg": statistics.median(p["agg_tok_s"] for p in ps),
                                   "ttft": statistics.median(r["ttft_s"] for r in sel if r["ttft_s"])})
        res = {}
        for key in ("decode", "norm", "agg", "ttft"):
            a = [x[key] for x in per["flip"]]
            b = [x[key] for x in per["base"]]
            if a and b:
                res[key] = {"base": round(statistics.median(b), 4), "flip": round(statistics.median(a), 4),
                            "ratio": round(statistics.median(a) / statistics.median(b), 4), "U_flip_gt_base": u(a, b),
                            "pairs": len(a) * len(b)}
        ks = {arm: sorted({r["k"][0] for r in rows if r["arm"] == arm and r["class"] == cls and r["k"]}) for arm in ARMS}
        acc = {arm: statistics.median([r["acceptance"] for r in rows if r["arm"] == arm and r["class"] == cls
                                       and r["acceptance"] is not None] or [0]) for arm in ARMS}
        itl = {arm: statistics.median([r["itl_ms"]["p95"] for r in rows if r["arm"] == arm and r["class"] == cls
                                       and r["itl_ms"]] or [0]) for arm in ARMS}
        out[cls] = {"per_boot": res, "k": ks, "acceptance_median": acc, "itl_p95_ms_median": itl}
    shas = {}
    for r in rows:
        if r["class"] == "greedy-4k":
            shas.setdefault(r["prompt"], {}).setdefault(r["arm"], set()).add(r["sha"])
    out["greedy_shas"] = {p: {a: sorted(v) for a, v in d.items()} for p, d in shas.items()}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if BIN == "-":
    summarize()
else:
    run_boot(int(os.environ.get("BOOT_START", "1")))
