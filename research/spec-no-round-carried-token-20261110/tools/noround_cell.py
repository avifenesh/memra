#!/usr/bin/env python3
"""What a no-round continuation burst costs the next turn in serving (memra#1196).

A greedy speculative session that enters a continuation burst on a carried pending token while
a request waits for admission runs no round. Before the fix that burst pushed the carried token
into the session's committed list with no cache row behind it. This cell asks what the next
request on that session then gets.

One boot per call, greedy, MEMRA_SPEC_SETUP_TRACE=1, MEMRA_MAX_SESSIONS=8, MEMRA_SPEC_GATE=0 (no
demotion under load, so the long request keeps speculating). Four first requests, each followed
by the request that continues its session:

  turn       a 1.2k-token chat prompt, 400 tokens; then the same conversation plus the answer
             and a second user turn
  park       a text completion, 400 tokens; then its prompt plus its answer
  turn-quiet, park-quiet   the same two with nothing else arriving (no no-round burst)

`turn` and `park` first run alone for LEAD seconds (1.5), so the request has run rounds and
holds a carried token. Then two clients send short one-token requests back to back for WINDOW
seconds (3) and stop. A burst of the long request that begins while one of them waits for
admission runs no round; one that entered on a carried token emits nothing. The short requests
are below the reuse floor: they are never parked and publish no prefix entry, so they cannot
evict the session under test.

Per first request: its continuation bursts and how many ran no round (their round loop took
under 1 ms), from the server log. Per follow-up: prompt tokens, cached tokens, completion
tokens, text sha, and the server's own resume lines for it.

MODE=ctx: an open request (no max_tokens, so it runs to the window) on a small context
(MEMRA_CTX=640), alone and with the same arrivals: its completion tokens against the no-round
bursts it took (the context-full check reads the committed list).

usage: noround_cell.py OUT BINARY ARM   (env: CELL_MODEL, CELL_PORT, MODE, LEAD, WINDOW)
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
MODE = os.environ.get("MODE", "resume")
PORT = int(os.environ.get("CELL_PORT", "18192"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "640" if MODE == "ctx" else "8192", "MEMRA_MAX_SESSIONS": "8", "MEMRA_SPEC_SETUP_TRACE": "1",
       # no demotion under load: the long request must stay on the speculative route
       "MEMRA_SPEC_GATE": "0"}
LEAD, WINDOW = float(os.environ.get("LEAD", "1.5")), float(os.environ.get("WINDOW", "3"))
SETUP = re.compile(r"\[spec-setup\] .* rounds=([0-9.]+)ms .* out=(\d+) cont=(true|false)")
RESUME = re.compile(r"^\[(?:worker|kv-reuse|prefix-cache)\][^\n]*(?:spec-reuse|spec-affinity|grid|rewound|resumed|hit:)[^\n]*", re.M)


def notes(topic, n):
    return " ".join(f"Note {i}: the {topic} survey records one measurement per site, and a later visit "
                    f"extends the same record with a new reading." for i in range(1, n + 1))


def long_prompt(topic, n=45):
    return f"{topic.capitalize()} field notes. {notes(topic, n)}\n\nWrite about 300 words on what these notes say about {topic}."


def user(content):
    return [{"role": "user", "content": content}]


def post(payload, max_tokens):
    body = {"model": "q", "stream": False, "temperature": 0}
    if max_tokens is not None:  # an open request runs to the server's context window
        body["max_tokens"] = max_tokens
    if isinstance(payload, str):
        path, body["prompt"] = "/v1/completions", payload
    else:
        path = "/v1/chat/completions"
        body.update(messages=payload, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            obj = json.loads(r.read())
        ch = obj["choices"][0]
        text = ch.get("text") if "text" in ch else (ch.get("message") or {}).get("content")
        return {"text": text or "", "usage": obj.get("usage") or {}, "finish": ch.get("finish_reason"), "error": None}
    except Exception as exc:  # recorded, never swallowed
        return {"text": "", "usage": {}, "finish": None, "error": repr(exc)}


class Arrivals:
    """Other requests while one request generates. The long request first runs alone for
    LEAD seconds, so it has run rounds and holds a carried token. Then two clients send short
    one-token requests back to back for WINDOW seconds, and stop. A short prompt is below
    the reuse floor, so these sessions are never parked and publish no prefix entry: they
    cannot evict the session under test. Each one sits in the worker's command channel for a
    moment, and a burst of the long request that begins in that moment runs no round."""

    def __init__(self, tag):
        self.tag, self.done, self.sent = tag, threading.Event(), 0
        self.threads = [threading.Thread(target=self.run, args=(i,)) for i in range(2)]

    def run(self, i):
        self.done.wait(LEAD + 0.05 * i)
        end = time.monotonic() + WINDOW
        n = 0
        while not self.done.is_set() and time.monotonic() < end:
            n += 1
            post(user(f"Name one {self.tag} tool, number {i}-{n}."), 1)
            self.sent += 1

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        for t in self.threads:
            t.join()


def one(log, kind, payload, max_tokens, arrivals):
    mark = log.stat().st_size
    if arrivals:
        with Arrivals(kind.replace("-", "")):
            o = post(payload, max_tokens)
    else:
        o = post(payload, max_tokens)
    threading.Event().wait(1.5)
    with open(log, errors="replace") as f:
        f.seek(mark)
        chunk = f.read()
    bursts = [(float(ms), int(n)) for ms, n, cont in SETUP.findall(chunk) if cont == "true"]
    usage = o["usage"]
    row = {"arm": ARM, "mode": MODE, "kind": kind, "arrivals": bool(arrivals), "finish": o["finish"],
           "prompt_tokens": usage.get("prompt_tokens"),
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "completion_tokens": usage.get("completion_tokens"),
           "continuation_bursts": len(bursts), "no_round_bursts": sum(1 for ms, _ in bursts if ms < 1.0),
           # a no-round burst that emitted nothing entered on a carried token
           "no_round_on_a_carried_token": sum(1 for ms, n in bursts if ms < 1.0 and n == 0),
           "resume_lines": [line[:200] for line in RESUME.findall(chunk)][:6],
           "sha": hashlib.sha256(o["text"].encode()).hexdigest()[:16], "error": o["error"]}
    with open(OUT / f"{MODE}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(ARM, MODE, kind, row["finish"], row["prompt_tokens"], row["cached"], row["completion_tokens"],
          "continuation bursts", row["continuation_bursts"], "no-round", row["no_round_bursts"],
          "on a carried token", row["no_round_on_a_carried_token"], row["sha"], row["error"] or "",
          flush=True)
    for line in row["resume_lines"]:
        print("   ", line, flush=True)
    return o["text"]


def ready(p, log):
    for _ in range(900):
        if p.poll() is not None:
            raise SystemExit(f"server exited {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(URL + "/readyz", timeout=3).read()
            return
        except Exception:
            threading.Event().wait(1)
    raise SystemExit("server not ready in 900 s")


def main():
    log = OUT / f"server-{MODE}-{ARM}.log"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        ready(p, log)
        if MODE == "ctx":
            one(log, "ctx-quiet", user("Meadow in brief: write a long report on how a meadow survey is planned and run."), None, False)
            one(log, "ctx", user("Orchard in brief: write a long report on how an orchard survey is planned and run."), None, True)
            return
        follow = "Now give three concrete next steps for the same survey, in about 80 words."
        for kind, topic, arrivals in (("turn-quiet", "lagoons", False), ("turn", "moors", True)):
            e1 = long_prompt(topic)
            answer = one(log, kind, user(e1), 400, arrivals)
            conv = user(e1) + [{"role": "assistant", "content": answer}, {"role": "user", "content": follow}]
            one(log, kind + "-next", conv, 32, False)
        for kind, topic, arrivals in (("park-quiet", "glaciers", False), ("park", "tundra", True)):
            p1 = f"{topic.capitalize()} survey log. {notes(topic, 40)}\n\nSummary of the log:"
            answer = one(log, kind, p1, 400, arrivals)
            one(log, kind + "-next", p1 + answer, 32, False)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


main()
