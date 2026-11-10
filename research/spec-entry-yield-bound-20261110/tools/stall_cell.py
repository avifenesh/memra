#!/usr/bin/env python3
"""Does a speculating request advance while short requests keep arriving? (memra#1201)

A burst that begins while a request waits for admission runs no round (the admission yield at
the burst's entry). A session that holds a carried pending token emits nothing in such a burst,
so under arrivals that keep coming it can stop emitting. This cell measures the long request's
progress and the short requests' latency under closed-loop arrivals with a think time.

One server boot per call, several cells in it (CELLS). Per cell: one long streamed request
(a 34-token prompt, max_tokens 3000) runs alone for LEAD seconds, so it has run rounds and holds
a carried token; then CLIENTS closed-loop clients send short one-token chat requests for WINDOW
seconds, each waiting THINK ms after every answer; then they stop, the long request runs alone
for TAIL seconds more, and the cell hangs up on it.

Per cell row: the long request's content chunks (one per token) before, during and after the
arrivals, the rate in each phase, the longest gap between chunks inside the window, the short
requests served and their latency (p50, p95, max), and from the server log inside the window:
continuation bursts of the long request, how many ran no round, how many of those emitted
nothing, the entry-yield bound's forced rounds, starved markers, flush-arm flushes, and
demotions. The long request's full text goes to texts/ (greedy texts compare by prefix: the
stream is cut at a time, not a token count).

usage: stall_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18201), CELLS ("4:0" = 4 clients, 0 ms; comma list; an optional
     third field picks the class: greedy (default) or t1), LEAD (8), WINDOW (30), TAIL (12),
     GATE ("default" leaves the gate at its default; "0" sets MEMRA_SPEC_GATE=0),
     EXTRA_ENV ("K=V;K=V" added to the server's environment), TAG (the row and log label;
     default ARM), KNOBS_FILE (the probe build's knob file: each cell's fourth field, "N=1+STARVED=2",
     is written to it one line per knob before the cell starts; empty = the probe's off state)
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
(OUT / "texts").mkdir(exist_ok=True)
BIN = sys.argv[2]
ARM = os.environ.get("TAG") or sys.argv[3]
PORT = int(os.environ.get("CELL_PORT", "18201"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, WINDOW, TAIL = (float(os.environ.get(k, d)) for k, d in (("LEAD", "8"), ("WINDOW", "30"), ("TAIL", "12")))
GATE = os.environ.get("GATE", "default")
CELLS = [c.split(":") for c in os.environ.get("CELLS", "4:0").split(",") if c]
KNOBS = os.environ.get("KNOBS_FILE", "")
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "8", "MEMRA_SPEC_SETUP_TRACE": "1"}
if GATE == "0":
    ENV["MEMRA_SPEC_GATE"] = "0"
if KNOBS:
    ENV["PROBE1201_KNOBS"] = KNOBS
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(";")):
    k, v = kv.split("=", 1)
    ENV[k] = v
SETUP = re.compile(r"\[spec-setup\] .* rounds=([0-9.]+)ms .* out=(\d+) cont=(true|false)")
CLASSES = {"greedy": {"temperature": 0}, "t1": {"temperature": 1.0, "seed": 611}}
PROMPT = "Write a very long, detailed manual, at least 3000 words, on how a coastal survey is planned, run and written up."


def short(tag, stop, think, lat, errors):
    n = 0
    while not stop.is_set():
        n += 1
        body = {"model": "q", "max_tokens": 1, "stream": False, "temperature": 0,
                "messages": [{"role": "user", "content": f"Name one {tag} tool, number {n}."}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            urllib.request.urlopen(req, timeout=120).read()
            lat.append(time.perf_counter() - t)
        except Exception as exc:  # counted and kept, never swallowed
            errors.append(repr(exc))
        if think > 0:
            stop.wait(think)


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
                    if ch.get("finish_reason"):
                        done["finish"] = ch["finish_reason"]
                if hang_up.is_set():
                    break
        finally:
            r.close()
    except Exception as exc:  # recorded, never swallowed
        done["error"] = repr(exc)
    done["text"] = "".join(text)
    done["end"] = time.perf_counter()


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


def main():
    log = OUT / f"server-{ARM}.log"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
    if KNOBS:
        pathlib.Path(KNOBS).write_text("\n")  # the boot starts with every probe knob off
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
        for cell in CELLS:
            clients, think_ms = int(cell[0]), float(cell[1])
            cls = cell[2] if len(cell) > 2 and cell[2] else "greedy"
            knobs = cell[3] if len(cell) > 3 else ""
            if KNOBS:
                pathlib.Path(KNOBS).write_text("\n".join(knobs.split("+")) + "\n")
                threading.Event().wait(0.5)
            times, done, hang_up = [], {}, threading.Event()
            mark0 = log.stat().st_size
            t0 = time.perf_counter()
            th = threading.Thread(target=long_request, args=(CLASSES[cls], times, done, hang_up))
            th.start()
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            stop, lat, errors = threading.Event(), [], []
            ths = [threading.Thread(target=short, args=(f"{cls}{clients}x{int(think_ms)}c{i}", stop, think_ms / 1e3, lat, errors))
                   for i in range(clients)]
            t1 = time.perf_counter()
            for c in ths:
                c.start()
            threading.Event().wait(WINDOW)
            stop.set()
            for c in ths:
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
                f.seek(mark0)
                whole = f.read()
            bursts = [(float(ms), int(n)) for ms, n, cont in SETUP.findall(chunk) if cont == "true"]
            no_round = [n for ms, n in bursts if ms < 1.0]
            text = done.get("text", "")
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            name = f"{ARM}-{cls}-c{clients}-t{int(think_ms)}" + (f"-{knobs}" if knobs else "")
            name = f"{name}-{int(time.time())}"
            (OUT / "texts" / f"{name}.txt").write_text(text)
            row = {"arm": ARM, "gate": GATE, "extra_env": os.environ.get("EXTRA_ENV", ""), "class": cls, "knobs": knobs,
                   "clients": clients, "think_ms": think_ms, "lead_s": LEAD, "window_s": round(t2 - t1, 2), "tail_s": TAIL,
                   "tokens_before": len(before), "tokens_during": len(during), "tokens_after": len(after),
                   "rate_before": round(len(before) / (t1 - t0), 2), "rate_during": round(len(during) / (t2 - t1), 2),
                   "rate_after": round(len(after) / (t3 - t2), 2), "longest_gap_in_window_s": round(gap, 2),
                   "short_requests": len(lat), "short_errors": len(errors), "short_error_samples": errors[:3],
                   "short_lat_p50_s": pct(lat, 0.5), "short_lat_p95_s": pct(lat, 0.95),
                   "short_lat_max_s": round(max(lat), 3) if lat else None,
                   "short_lat_mean_s": round(statistics.fmean(lat), 3) if lat else None,
                   "continuation_bursts_in_window": len(bursts), "no_round_bursts": len(no_round),
                   "no_round_bursts_that_emitted_nothing": sum(1 for n in no_round if n == 0),
                   "bound_rounds_in_window": chunk.count("[spec-yield] bound:"),
                   "starved_marks_in_window": chunk.count("[spec-yield] starved:"),
                   "flushes_in_window": chunk.count("[spec-yield] flush arm:"),
                   "demotions": whole.count("[spec-gate] demoted session"),
                   "spec_cost_lines": [l for l in whole.splitlines() if l.startswith("[spec-cost] model=")][:12],
                   "finish": done.get("finish"), "chars": len(text), "sha": sha, "text_file": f"texts/{name}.txt",
                   "error": done.get("error")}
            with open(OUT / "stall.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "spec_cost_lines"}), flush=True)
            threading.Event().wait(3)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


main()
