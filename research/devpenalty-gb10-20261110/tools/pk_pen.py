#!/usr/bin/env python3
"""Device penalty arm A/B (MEMRA_SERVE_DEVPENALTY off against on), one boot per call.

The request shape is the Qwen non-thinking vendor shape: temperature 0.7, top_p 0.8, top_k 20,
presence_penalty 1.5, a fixed seed per prompt, thinking off. Per boot:
  c1  three prompts one at a time (34, 1.2k, 1.2k tokens), 300-token answers
  c4  two groups of four prompts sent at once (two short, two 1.2k), 300-token answers
Per request: TTFT, decode rate after the first chunk, the [spec-k] line, acceptance, text sha.
Per group: aggregate tokens per second over the group's wall time.

usage: pk_pen.py OUT BINARY ARM   (ARM = off | on; env: BOOT_START, CELL_MODEL, CELL_PORT,
MAX_TOKENS)
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN, ARM = sys.argv[2], sys.argv[3]
PORT = int(os.environ.get("CELL_PORT", "18195"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("CELL_MODEL", "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "300"))
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "4", "MEMRA_SERVE_DEVPENALTY": "1" if ARM == "on" else "0"}
KLINE = r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)"


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic):
    return f"{topic.capitalize()} field notes. {notes(topic, 45)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


def shape(seed):
    return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": seed}


def boot(log):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
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


def stream(content, samp, out):
    body = {"model": "q", "messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOKENS, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, **samp}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
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


def group(n, log, cls, items):
    outs = [{} for _ in items]
    before = log.stat().st_size
    t0 = time.perf_counter()
    threads = [threading.Thread(target=stream, args=(c, s, o)) for (_, c, s), o in zip(items, outs)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    time.sleep(0.3)
    with open(log, errors="replace") as f:
        f.seek(before)
        klines = re.findall(KLINE, f.read())
    total = 0
    for (pid, _, _), o in zip(items, outs):
        usage = o.get("usage") or {}
        toks = usage.get("completion_tokens") or 0
        total += toks
        times = o.get("times") or []
        ttft = times[0] if times else None
        ptoks = usage.get("prompt_tokens")
        k = next((kl for kl in klines if ptoks and abs(int(kl[2]) - ptoks) <= 1), None)
        text = o.get("text") or ""
        r = {"boot": n, "arm": ARM, "class": cls, "prompt": pid, "k": k, "prompt_tokens": ptoks,
             "completion_tokens": toks, "ttft_s": round(ttft, 4) if ttft else None, "total_s": round(o.get("total", 0), 4),
             "decode_tok_s": round(toks / (o["total"] - ttft), 2) if ttft and o["total"] > ttft and toks else None,
             "acceptance": (usage.get("spec") or {}).get("acceptance_rate"),
             "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "error": o.get("error")}
        with open(OUT / "runs.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        print(ARM, n, cls, pid, r["k"], r["ttft_s"], r["decode_tok_s"], r["acceptance"], toks, r["sha"], r["error"], flush=True)
    g = {"boot": n, "arm": ARM, "class": cls, "requests": len(items), "tokens": total, "wall_s": round(wall, 3),
         "aggregate_tok_s": round(total / wall, 2) if wall else None}
    with open(OUT / "groups.jsonl", "a") as f:
        f.write(json.dumps(g) + "\n")
    print(ARM, n, cls, "aggregate", g["aggregate_tok_s"], flush=True)


def main():
    n = int(os.environ.get("BOOT_START", "1"))
    log = OUT / f"server-{n}-{ARM}.log"
    tel = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_throttle_reasons.active",
                            "--format=csv", "-lms", "250"], stdout=open(OUT / f"telemetry-{n}-{ARM}.csv", "w"), stderr=subprocess.DEVNULL)
    p = boot(log)
    try:
        for i, (pid, content) in enumerate((("s1", short_prompt("delta")), ("l1", long_prompt("estuary")), ("l2", long_prompt("tundra")))):
            group(n, log, "c1", [(pid, content, shape(801 + i))])
        for gi, topics in enumerate((("reef", "steppe", "marsh", "fjord"), ("atoll", "mesa", "bog", "plateau"))):
            items = [(f"g{gi}{j}", short_prompt(t) if j < 2 else long_prompt(t), shape(811 + 10 * gi + j)) for j, t in enumerate(topics)]
            group(n, log, "c4", items)
    finally:
        stop(p)
        tel.terminate()
        try:
            tel.wait(timeout=15)
        except subprocess.TimeoutExpired:
            tel.kill()
            tel.wait()


main()
