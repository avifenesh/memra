#!/usr/bin/env python3
"""Draft-side accept leftovers, before against after (memra#1169).

The #1154 cell program (`research/position-keyed-sampling-20261110/tools/pk_ab.py`, MODE perf)
with one more sampler class appended after its nine prompts. One boot per call. Arms are two
binaries (the driver passes each its own): `before`, the #1154 branch, and `after`, this change.

Twelve cold prompts, one at a time, three per sampler (short 34 tokens, 1.2k, 3.6k):
  t1     temperature 1, fixed seed                 (pure-temperature sampled draft graph)
  chat   temperature 0.7, top_p 0.95, fixed seed   (filtered sampled draft graph)
  greedy temperature 0 (the control: its text and rate must not move between the arms)
  pen    temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, fixed seed
         (a penalised request drafts on the eager sampled chain)
Per request: the [spec-k] line, TTFT, decode rate after the first chunk, acceptance, text sha.

Each row also records the request's speculative rounds, drafted and accepted tokens.
MODE inv is pk_ab.py's, unchanged.

usage: ab_cell.py OUT BINARY ARM   (env: BOOT_START, CELL_MODEL, CELL_PORT, TELEMETRY_GPU, MODE, SPEC_K)
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
(OUT / "texts").mkdir(exist_ok=True)
BIN, ARM = sys.argv[2], sys.argv[3]
MODE = os.environ.get("MODE", "perf")
PORT = int(os.environ.get("CELL_PORT", "18193"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("CELL_MODEL", "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf")
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
SPEC_K = os.environ.get("SPEC_K")
if SPEC_K is not None:
    ENV["MEMRA_SPEC_K"] = SPEC_K
KLINE = r"\[spec-k\] model=\"q\" tenant=\"[^\"]*\" K=(\d+) source=(\S+) prompt=(\d+) cached=(\d+)"


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def short_prompt(topic):
    return f"{topic.capitalize()} in brief: write about 300 words on how a {topic} survey is planned and run."


SAMPLERS = {"t1": lambda seed: {"temperature": 1.0, "seed": seed},
            "chat": lambda seed: {"temperature": 0.7, "top_p": 0.95, "seed": seed},
            "greedy": lambda seed: {"temperature": 0},
            "pen": lambda seed: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": seed}}
PERF = []
for si, (sname, topics) in enumerate((("t1", ("comet", "rivers", "bridges")), ("chat", ("tide", "glaciers", "harbors")),
                                      ("greedy", ("dune", "islands", "deserts")), ("pen", ("orchards", "lakes", "valleys")))):
    for ti, (size, topic) in enumerate(zip(("short", "1k", "4k"), topics)):
        content = short_prompt(topic) if size == "short" else long_prompt(topic, 45 if size == "1k" else 130)
        PERF.append((f"{sname}-{size}", f"{topic}-{size}", content, SAMPLERS[sname](501 + 10 * si + ti)))


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


def stream(messages, samp, out, max_tokens=400):
    body = {"model": "q", "messages": messages, "max_tokens": max_tokens, "stream": True,
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


def record(n, cls, pid, o, klines, extra=None):
    usage = o.get("usage") or {}
    toks = usage.get("completion_tokens") or 0
    times = o.get("times") or []
    ttft = times[0] if times else None
    ptoks = usage.get("prompt_tokens")
    k = next((kl for kl in klines if ptoks and abs(int(kl[2]) - ptoks) <= 1), None)
    spec = usage.get("spec") or {}
    text = o.get("text") or ""
    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    r = {"boot": n, "arm": ARM, "mode": MODE, "spec_k_pin": SPEC_K, "class": cls, "prompt": pid, "k": k,
         "prompt_tokens": ptoks, "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
         "completion_tokens": toks, "ttft_s": round(ttft, 4) if ttft else None,
         "total_s": round(o.get("total", 0), 4),
         "decode_tok_s": round(toks / (o["total"] - ttft), 2) if ttft and o["total"] > ttft and toks else None,
         "acceptance": spec.get("acceptance_rate"), "rounds": spec.get("rounds"), "drafted": spec.get("drafted"),
         "accepted": spec.get("accepted"), "sha": sha, "error": o.get("error")}
    if extra:
        r.update(extra)
    with open(OUT / "runs.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    (OUT / "texts" / f"{ARM}-k{SPEC_K}-b{n}-{cls}-{pid}.txt").write_text(text)
    print(ARM, n, cls, pid, r["k"], r["ttft_s"], r["decode_tok_s"], r["acceptance"], toks, sha, r["error"], flush=True)
    return text


def serve(n, log, group):
    """One request, or several at once: group = [(cls, pid, messages, samp), ...]."""
    outs = [{} for _ in group]
    before = log.stat().st_size
    threads = [threading.Thread(target=stream, args=(m, s, o)) for (_, _, m, s), o in zip(group, outs)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    time.sleep(0.3)
    with open(log, errors="replace") as f:
        f.seek(before)
        klines = re.findall(KLINE, f.read())
    return [record(n, cls, pid, o, klines) for (cls, pid, _, _), o in zip(group, outs)]


def user(content):
    return [{"role": "user", "content": content}]


def t1(seed):
    return {"temperature": 1.0, "seed": seed}


def main():
    n = int(os.environ.get("BOOT_START", "1"))
    log = OUT / f"server-{n}-{ARM}.log"
    sel = ["-i", os.environ["TELEMETRY_GPU"]] if os.environ.get("TELEMETRY_GPU") else []
    tel = subprocess.Popen(["nvidia-smi", *sel, "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_throttle_reasons.active",
                            "--format=csv", "-lms", "250"], stdout=open(OUT / f"telemetry-{n}-{ARM}.csv", "w"),
                           stderr=subprocess.DEVNULL)
    p = boot(log)
    try:
        if MODE == "perf":
            for cls, pid, content, samp in PERF:
                serve(n, log, [(cls, pid, user(content), samp)])
            return
        part = os.environ.get("PART", "x")
        a, b = long_prompt("volcanoes", 45), short_prompt("meadow")
        c, d = long_prompt("canyons", 45), long_prompt("forests", 45)
        e1 = long_prompt("lagoons", 45)
        e2 = "Now give three concrete next steps for the same survey, in about 150 words."
        if part == "x":
            serve(n, log, [("alone", "a", user(a), t1(701))])
            serve(n, log, [("alone", "b", user(b), t1(702))])
            serve(n, log, [("pair", "c", user(c), t1(703)), ("pair", "d", user(d), t1(704))])
            answer = serve(n, log, [("turn1", "e1", user(e1), t1(705))])[0]
            (OUT / "texts" / f"E1-{ARM}-k{SPEC_K}.txt").write_text(answer)
            conv = user(e1) + [{"role": "assistant", "content": answer}, {"role": "user", "content": e2}]
            serve(n, log, [("resumed", "e2", conv, t1(706))])
        else:
            serve(n, log, [("solo", "c", user(c), t1(703))])
            serve(n, log, [("solo", "d", user(d), t1(704))])
            answer = pathlib.Path(os.environ["E1_FILE"]).read_text()
            conv = user(e1) + [{"role": "assistant", "content": answer}, {"role": "user", "content": e2}]
            serve(n, log, [("cold", "e2", conv, t1(706))])
    finally:
        stop(p)
        tel.terminate()
        try:
            tel.wait(timeout=15)
        except subprocess.TimeoutExpired:
            tel.kill()
            tel.wait()


main()
