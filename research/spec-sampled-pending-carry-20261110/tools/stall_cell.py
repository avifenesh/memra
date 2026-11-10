#!/usr/bin/env python3
"""Does a session that holds a carried token advance while other requests keep arriving?

A burst that begins while a request waits for admission runs no round (the admission yield at
the burst's entry). A session with no carried token still emits its boundary token in such a
burst, one token per tick. A session that entered on a carried token emits nothing. This cell
measures what a long request does under sustained arrivals when it holds a carried token.

One boot per call. Per class (greedy, then temperature 1 with a fixed seed) one long streamed
request (a 34-token prompt, max_tokens 3000). It runs alone for LEAD seconds, so it has run
rounds and holds a carried token. Then four closed-loop clients send short one-token requests
back to back for WINDOW seconds, then stop, and the long request runs alone for TAIL seconds
more before the cell hangs up on it. Streaming keeps the 90 s non-streaming deadline out.

Per class: content chunks (one per token) before, during and after the arrivals, the rate in
each phase, the longest gap between chunks inside the arrival window, the short requests
served, and from the server log the long request's continuation bursts in the window: how many
ran no round, and how many of those emitted nothing.

usage: stall_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT, LEAD (4), WINDOW (20), TAIL (8), GATE (0 = MEMRA_SPEC_GATE=0, the
     default; "default" leaves the demotion gate at its default)
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
PORT = int(os.environ.get("CELL_PORT", "18191"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, WINDOW, TAIL = (float(os.environ.get(k, d)) for k, d in (("LEAD", "4"), ("WINDOW", "20"), ("TAIL", "8")))
GATE = os.environ.get("GATE", "0")
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "8", "MEMRA_SPEC_SETUP_TRACE": "1"}
if GATE == "0":
    ENV["MEMRA_SPEC_GATE"] = "0"
SETUP = re.compile(r"\[spec-setup\] .* rounds=([0-9.]+)ms .* out=(\d+) cont=(true|false)")
CLASSES = (("greedy", {"temperature": 0}), ("t1", {"temperature": 1.0, "seed": 611}))
PROMPT = "Write a very long, detailed manual, at least 3000 words, on how a coastal survey is planned, run and written up."


def short(tag, stop, counter):
    n = 0
    while not stop.is_set():
        n += 1
        body = {"model": "q", "max_tokens": 1, "stream": False, "temperature": 0,
                "messages": [{"role": "user", "content": f"Name one {tag} tool, number {n}."}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=120).read()
            counter[0] += 1
        except Exception:  # counted, never swallowed
            counter[1] += 1


def long_request(samp, times, done, hang_up):
    body = {"model": "q", "max_tokens": 3000, "stream": True, **samp,
            "messages": [{"role": "user", "content": PROMPT}], "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = []
    try:
        r = urllib.request.urlopen(req, timeout=600)
        try:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                for ch in json.loads(line[5:]).get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        times.append(time.perf_counter())
                        text.append(piece)
                if hang_up.is_set():
                    break
        finally:
            r.close()
    except Exception as exc:  # recorded, never swallowed
        done["error"] = repr(exc)
    done["text"] = "".join(text)
    done["end"] = time.perf_counter()


def main():
    log = OUT / f"server-{ARM}-gate{GATE}.log"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(900):
            if p.poll() is not None:
                raise SystemExit(f"server exited {p.returncode}; see {log}")
            try:
                urllib.request.urlopen(URL + "/readyz", timeout=3).read()
                break
            except Exception:
                threading.Event().wait(1)
        for cls, samp in CLASSES:
            times, done, hang_up = [], {}, threading.Event()
            mark0 = log.stat().st_size
            t0 = time.perf_counter()
            th = threading.Thread(target=long_request, args=(samp, times, done, hang_up))
            th.start()
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            stop, counter = threading.Event(), [0, 0]
            clients = [threading.Thread(target=short, args=(f"{cls}{i}", stop, counter)) for i in range(4)]
            t1 = time.perf_counter()
            for c in clients:
                c.start()
            threading.Event().wait(WINDOW)
            stop.set()
            for c in clients:
                c.join()
            t2 = time.perf_counter()
            mark2 = log.stat().st_size
            threading.Event().wait(TAIL)
            t3 = time.perf_counter()
            hang_up.set()
            th.join(30)
            before = [t for t in times if t < t1]
            during = [t for t in times if t1 <= t < t2]
            after = [t for t in times if t2 <= t < t3]
            edges = [t1] + during + [t2]
            gap = max(b - a for a, b in zip(edges, edges[1:]))
            with open(log, errors="replace") as f:
                f.seek(mark1)
                chunk = f.read(mark2 - mark1)
            bursts = [(float(ms), int(n)) for ms, n, cont in SETUP.findall(chunk) if cont == "true"]
            no_round = [n for ms, n in bursts if ms < 1.0]
            demoted = "[spec-gate] demoted session" in open(log, errors="replace").read()[mark0:]
            row = {"arm": ARM, "gate": GATE, "class": cls, "lead_s": LEAD, "window_s": round(t2 - t1, 2), "tail_s": TAIL,
                   "tokens_before": len(before), "tokens_during": len(during), "tokens_after": len(after),
                   "rate_before": round(len(before) / (t1 - t0), 2), "rate_during": round(len(during) / (t2 - t1), 2),
                   "rate_after": round(len(after) / (t3 - t2), 2), "longest_gap_in_window_s": round(gap, 2),
                   "short_requests": counter[0], "short_errors": counter[1],
                   "continuation_bursts_in_window": len(bursts), "no_round_bursts": len(no_round),
                   "no_round_bursts_that_emitted_nothing": sum(1 for n in no_round if n == 0),
                   "demoted_to_plain": demoted, "finished": th.is_alive() is False and "end" in done and not hang_up.is_set(),
                   "sha": hashlib.sha256(done.get("text", "").encode()).hexdigest()[:16], "error": done.get("error")}
            with open(OUT / "stall.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            threading.Event().wait(2)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


main()
