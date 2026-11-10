#!/usr/bin/env python3
"""The end of a request, base against candidate (memra#1186): what a request writes, what its
usage counts, how long the response stays open after its last chunk, and what the next request
on the same session gets.

Before the carry a sampled burst committed its pending token with a solo pass at its tail, so
a sampled request's last burst did too and the response ended one pass after its last chunk.
With the carry the token stays pending on the session, as a greedy request's does, and what
follows the request decides what happens to it: nothing (the session is freed), a flush when
the session is parked or demoted, or a drop when the client is gone.

One boot per call. Per sampler class (greedy as the control, t1, chat, pen) the boot serves,
streamed, one request at a time:

MODE=close (default)
  eos           an answer that ends on EOS
  budget        an answer cut by max_tokens (96)
  stop          the same request cut by a stop string (a word of the budget answer, taken once
                and kept in stops.json so both arms use the same one)
  turn1, turn2  a two-turn conversation: the second turn resumes the session of the first
  park1, park2  a text completion, then its prompt plus its answer (the session park1 leaves
                holds exactly park2's prompt)
  other         a request with another prefix, right after
  cancelled     a request whose client hangs up after 40 chunks (mid burst)
  after-cancel  the same request again, whole
MODE=demote (MEMRA_SPEC_DEMOTE_AT=40: speculation hands each request to plain decode after 40
tokens, so the first request of a pair ends demoted)
  turn1, turn2, park1, park2 as above

Each row: text sha, finish reason, prompt, cached and completion tokens, the speculative counts
of the usage block, the time of the last chunk and close_ms = total - last chunk.

usage: close_cell.py OUT BINARY ARM   (env: CELL_MODEL, CELL_PORT, BOOT_START, MODE)
"""
import hashlib
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BIN, ARM = sys.argv[2], sys.argv[3]
MODE = os.environ.get("MODE", "close")
PORT = int(os.environ.get("CELL_PORT", "18195"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
BOOT = int(os.environ.get("BOOT_START", "1"))
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "2"}
if MODE == "demote":
    ENV["MEMRA_SPEC_DEMOTE_AT"] = "40"
SAMPLERS = {"greedy": lambda seed: {"temperature": 0},
            "t1": lambda seed: {"temperature": 1.0, "seed": seed},
            "chat": lambda seed: {"temperature": 0.7, "top_p": 0.95, "seed": seed},
            "pen": lambda seed: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": seed}}
TOPICS = {"greedy": ("tide", "reef", "lagoons", "glaciers"), "t1": ("rain", "marsh", "moors", "tundra"),
          "chat": ("wind", "delta", "dunes", "steppe"), "pen": ("snow", "fjord", "cliffs", "plateau")}


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n=45):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def user(content):
    return [{"role": "user", "content": content}]


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


def stream(payload, samp, max_tokens, stop=None, hang_up_after=None):
    """payload: chat messages, or a raw text for /v1/completions. hang_up_after: close the
    connection after that many content chunks."""
    body = {"model": "q", "max_tokens": max_tokens, "stream": True, **samp}
    if stop:
        body["stop"] = [stop]
    if isinstance(payload, str):
        path, body["prompt"] = "/v1/completions", payload
    else:
        path = "/v1/chat/completions"
        body.update(messages=payload, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, text, usage, finish = [], [], None, None
    try:
        r = urllib.request.urlopen(req, timeout=900)
        try:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[5:])
                usage = obj.get("usage") or usage
                for ch in obj.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    piece = ch.get("text") if "text" in ch else (ch.get("delta") or {}).get("content")
                    if piece:
                        times.append(time.perf_counter() - t0)
                        text.append(piece)
                if hang_up_after and len(times) >= hang_up_after:
                    break
        finally:
            r.close()
        err = None
    except Exception as exc:  # recorded, never swallowed
        err = repr(exc)
    return {"times": times, "total": time.perf_counter() - t0, "text": "".join(text), "usage": usage,
            "finish": finish, "error": err}


def record(cls, kind, o, stop=None):
    usage = o.get("usage") or {}
    spec = usage.get("spec") or {}
    times = o["times"]
    text = o["text"]
    r = {"boot": BOOT, "arm": ARM, "mode": MODE, "class": cls, "kind": kind, "stop": stop, "finish": o["finish"],
         "prompt_tokens": usage.get("prompt_tokens"),
         "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
         "completion_tokens": usage.get("completion_tokens"),
         "rounds": spec.get("rounds"), "drafted": spec.get("drafted"), "accepted": spec.get("accepted"),
         "chunks": len(times), "last_chunk_s": round(times[-1], 4) if times else None,
         "total_s": round(o["total"], 4),
         "close_ms": round(1000 * (o["total"] - times[-1]), 2) if times else None,
         "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "error": o["error"]}
    with open(OUT / f"{MODE}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    print(ARM, BOOT, MODE, cls, kind, r["finish"], r["prompt_tokens"], r["cached"], r["completion_tokens"], r["rounds"],
          r["close_ms"], r["sha"], r["error"], flush=True)
    return text


def pick_stop(text):
    words = text.split()
    for i in range(30, len(words)):
        w = words[i].strip(".,;:!?()\"'*#-")
        if len(w) >= 5 and w.isalpha() and text.find(w) >= len(" ".join(words[:i])) - 1:
            return w
    return None


def pairs(cls, samp, i):
    a, b, c, d = TOPICS[cls]
    e1 = long_prompt(c)
    turn1 = record(cls, "turn1", stream(user(e1), samp(921 + i), 96))
    conv = user(e1) + [{"role": "assistant", "content": turn1},
                       {"role": "user", "content": "Now give three concrete next steps for the same survey, in about 80 words."}]
    record(cls, "turn2", stream(conv, samp(931 + i), 96))
    p1 = f"{d.capitalize()} survey log. {notes(d, 8)}\n\nSummary of the log:"
    park1 = record(cls, "park1", stream(p1, samp(951 + i), 64))
    record(cls, "park2", stream(p1 + park1, samp(961 + i), 96))


def main():
    log = OUT / f"server-{MODE}-{BOOT}-{ARM}.log"
    stops_file = OUT / "stops.json"
    stops = json.loads(stops_file.read_text()) if stops_file.exists() else {}
    tel = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_throttle_reasons.active",
                            "--format=csv", "-lms", "250"], stdout=open(OUT / f"telemetry-{MODE}-{BOOT}-{ARM}.csv", "w"),
                           stderr=subprocess.DEVNULL)
    p = boot(log)
    try:
        for i, cls in enumerate(("greedy", "t1", "chat", "pen")):
            samp = SAMPLERS[cls]
            a, b, c, d = TOPICS[cls]
            if MODE == "demote":
                pairs(cls, samp, i)
                continue
            record(cls, "eos", stream(user(f"In about 45 words, explain how a {a} gauge is read."), samp(901 + i), 200))
            body = f"{b.capitalize()} in brief: write about 300 words on how such a survey is planned and run."
            answer = record(cls, "budget", stream(user(body), samp(911 + i), 96))
            if cls not in stops:
                stops[cls] = pick_stop(answer)
                stops_file.write_text(json.dumps(stops))
            if stops.get(cls):
                record(cls, "stop", stream(user(body), samp(911 + i), 96, stop=stops[cls]), stop=stops[cls])
            pairs(cls, samp, i)
            record(cls, "other", stream(user(f"List four tools a {b} survey needs, one line each."), samp(941 + i), 96))
            again = f"{a.capitalize()} in brief: write about 300 words on how a {a} survey is planned and run."
            record(cls, "cancelled", stream(user(again), samp(971 + i), 160, hang_up_after=40))
            time.sleep(1.5)
            record(cls, "after-cancel", stream(user(again), samp(971 + i), 160))
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        tel.terminate()
        try:
            tel.wait(timeout=15)
        except subprocess.TimeoutExpired:
            tel.kill()
            tel.wait()
    text = log.read_text(errors="replace")
    handoffs = len(re.findall(r"\[spec-gate\] demoted session to batched decode", text))
    aborts = len(re.findall(r"abort", text))
    print(f"{ARM} {MODE}: {handoffs} demotion handoffs and {aborts} abort lines in the server log", flush=True)


main()
