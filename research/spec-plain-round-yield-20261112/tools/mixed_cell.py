#!/usr/bin/env python3
"""Plain requests of several classes and lanes beside speculating streams (memra#1244 round-2 review).

One server boot per call, several cells in it (CELLS). Per cell: L long greedy streamed requests
start together and run alone for LEAD seconds (a pooled cohort in continuation bursts). Then the
cell's plain requests start, each after its own delay, each a streamed chat request from its own
prompt, and the cell waits for all of them. After TAIL seconds the cell hangs up on the streams.
Then the same plain requests run again with the same delays and nothing else in flight.

A cell is "L:ITEM+ITEM+...", an ITEM "CLASS[@LANE][~DELAY_S][#TOKENS]": CLASS pen (greedy +
frequency penalty 0.3: admitted plain, host-sampled) or t1json (temperature 1, fixed seed, JSON
object: constrained); LANE interactive (default), judge or harvest (sent as `x-lane`); DELAY_S
after the first item's start (default 0); TOKENS max tokens (default NEWTOK). The n-th item of a
class in a cell takes that class's n-th prompt and seed, so the same item writes the same text in
every cell, arm and phase.

Per cell row: each item's TTFT, token gaps, E2E, tokens and tokens inside the window, text and
sha; the window from the first interactive item's start to the last one's end ("live"); the
streams' rate inside it and before it; the summed rate of all requests inside it.

usage: mixed_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18245), CELLS, LEAD (8), NEWTOK (64), LONGTOK (6000), TAIL (3),
     EXTRA_ENV ("K=V;K=V"), TAG (row label; default ARM), MAX_SESSIONS (12), CTX (4096)
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
(OUT / "texts").mkdir(parents=True, exist_ok=True)
BIN = sys.argv[2]
ARM = os.environ.get("TAG") or sys.argv[3]
PORT = int(os.environ.get("CELL_PORT", "18245"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, TAIL = (float(os.environ.get(k, d)) for k, d in (("LEAD", "8"), ("TAIL", "3")))
NEWTOK, LONGTOK = int(os.environ.get("NEWTOK", "64")), int(os.environ.get("LONGTOK", "6000"))
ITEM = re.compile(r"^(?P<cls>[a-z0-9]+)(?:@(?P<lane>[a-z]+))?(?:~(?P<delay>[0-9.]+))?(?:#(?P<tok>[0-9]+))?$")
CELLS = []
for c in filter(None, os.environ.get("CELLS", "2:pen~0+t1json~1").split(",")):
    longs, items = c.split(":", 1)
    parsed = []
    for it in items.split("+"):
        m = ITEM.match(it)
        if not m:
            raise SystemExit(f"bad item {it!r}")
        parsed.append({"cls": m["cls"], "lane": m["lane"] or "interactive", "delay": float(m["delay"] or 0),
                       "tokens": int(m["tok"] or NEWTOK), "spec": it})
    CELLS.append((int(longs), parsed, c))
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": os.environ.get("CTX", "4096"), "MEMRA_MAX_SESSIONS": os.environ.get("MAX_SESSIONS", "12"),
       "MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_TICK_TRACE": "1"}
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(";")):
    k, v = kv.split("=", 1)
    ENV[k] = v
LONG_TOPICS = ["a coastal survey", "a mountain rescue exercise", "a harbor dredging project", "a regional seed bank"]
NEW_TOPICS = ["a river delta forms", "a bridge is inspected", "bread dough rises", "a ship's hull is painted",
              "a forest fire is mapped", "a census is planned", "a violin is tuned", "a library catalogs books"]
CLASSES = {"pen": lambda n: {"temperature": 0, "frequency_penalty": 0.3},
           "t1json": lambda n: {"temperature": 1.0, "seed": 811 + n, "response_format": {"type": "json_object"}}}
CLASS_OFFSET = {"pen": 0, "t1json": 4}


def stream(body, lane, times, done, hang_up=None, delay=0.0, t_zero=None):
    if delay and t_zero is not None:
        threading.Event().wait(max(0.0, t_zero + delay - time.perf_counter()))
    headers = {"Content-Type": "application/json"}
    if lane:
        headers["x-lane"] = lane
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    text = []
    done["start"] = time.perf_counter()
    try:
        r = urllib.request.urlopen(req, timeout=900)
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
                if hang_up is not None and hang_up.is_set():
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
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)


def wave(ci, items, label):
    """The cell's plain requests, each after its delay; returns their rows."""
    rows, ths, seen = [], [], {}
    t_zero = time.perf_counter()
    for k, it in enumerate(items):
        n = seen.get(it["cls"], 0)
        seen[it["cls"]] = n + 1
        it["n"] = n
        topic = NEW_TOPICS[(CLASS_OFFSET[it["cls"]] + n) % len(NEW_TOPICS)]
        body = {"model": "q", "max_tokens": it["tokens"], "stream": True, **CLASSES[it["cls"]](n),
                "messages": [{"role": "user", "content": f"Explain how {topic}, step by step, in plain words. "
                              f"Case {it['cls']}-{n}."}],
                "chat_template_kwargs": {"enable_thinking": False}}
        rows.append(([], {}))
        th = threading.Thread(target=stream, args=(body, it["lane"], rows[k][0], rows[k][1]),
                              kwargs={"delay": it["delay"], "t_zero": t_zero})
        th.start()
        ths.append(th)
    for th in ths:
        th.join(900)
    out = []
    for k, (times, done) in enumerate(rows):
        it = items[k]
        gaps = [round(b - a, 4) for a, b in zip(times, times[1:])]
        text = done.get("text", "")
        name = f"{ARM}-c{ci}-{label}{k}-{it['cls']}-{it['lane']}-{int(time.time())}.txt"
        (OUT / "texts" / name).write_text(text)
        out.append({"k": k, "cls": it["cls"], "lane": it["lane"], "n": it["n"], "spec": it["spec"], "max_tokens": it["tokens"],
                    "ts": done["start"], "end": done["end"], "times": times,
                    "ttft_s": round(times[0] - done["start"], 4) if times else None,
                    "gaps_s": gaps, "itl_p50_s": pct(gaps, 0.5), "itl_p95_s": pct(gaps, 0.95),
                    "e2e_s": round(done["end"] - done["start"], 4), "chunks": len(times),
                    "finish": done.get("finish"), "error": done.get("error"),
                    "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
    return out


def main():
    log = OUT / f"server-{ARM}.log"
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"port {PORT} busy")
    e = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
    e.update(ENV)
    p = subprocess.Popen([BIN], env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    print(f"server pid {p.pid}", flush=True)
    try:
        for _ in range(900):
            if p.poll() is not None:
                raise SystemExit(f"server exited {p.returncode}; see {log}")
            try:
                urllib.request.urlopen(URL + "/readyz", timeout=3).read()
                break
            except Exception:
                threading.Event().wait(1)
        for ci, (longs, items, spec) in enumerate(CELLS):
            hang_up = threading.Event()
            ltimes, ldone, lths = [], [], []
            mark0 = log.stat().st_size
            t0 = time.perf_counter()
            for i in range(longs):
                body = {"model": "q", "max_tokens": LONGTOK, "stream": True, "temperature": 0,
                        "messages": [{"role": "user", "content": f"Write a very long, detailed manual, at least 3000 "
                                      f"words, on how {LONG_TOPICS[i]} is planned, run and written up."}],
                        "chat_template_kwargs": {"enable_thinking": False}}
                ltimes.append([])
                ldone.append({})
                th = threading.Thread(target=stream, args=(body, None, ltimes[i], ldone[i], hang_up))
                th.start()
                lths.append(th)
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            beside = wave(ci, items, "b")
            mark2 = log.stat().st_size
            threading.Event().wait(TAIL)
            t3 = time.perf_counter()
            hang_up.set()
            for th in lths:
                th.join(60)
            inter = [x for x in beside if x["lane"] == "interactive"] or beside
            a, b = min(x["ts"] for x in inter), max(x["end"] for x in inter)
            live_s = b - a
            lrows = []
            for i in range(longs):
                times = ltimes[i]
                text = ldone[i].get("text", "")
                name = f"{ARM}-c{ci}-l{longs}-s{i}-{int(time.time())}.txt"
                (OUT / "texts" / name).write_text(text)
                n_live = sum(1 for t in times if a <= t < b)
                n_before = sum(1 for t in times if t0 + 2 <= t < min(x["ts"] for x in beside))
                lrows.append({"stream": i, "tokens": len(times), "tokens_live": n_live,
                              "rate_live": round(n_live / live_s, 2),
                              "rate_before": round(n_before / max(1e-9, min(x["ts"] for x in beside) - t0 - 2), 2),
                              "ended_in_window": bool(ldone[i].get("finish")) and ldone[i].get("end", t3) < b,
                              "finish": ldone[i].get("finish"), "error": ldone[i].get("error"),
                              "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
            for x in beside:
                x["tokens_live"] = sum(1 for t in x["times"] if a <= t < b)
                x["rate_live"] = round(x["tokens_live"] / live_s, 2)
            threading.Event().wait(1.0)
            mark3 = log.stat().st_size
            alone = wave(ci, items, "a")
            long_live = sum(r["tokens_live"] for r in lrows)
            plain_live = sum(x["tokens_live"] for x in beside)
            for x in beside + alone:
                x.pop("times")
                x["ts"] = round(x["ts"] - a, 3)
                x["end"] = round(x["end"] - a, 3)
            row = {"arm": ARM, "cell": ci, "spec": spec, "longs": longs, "extra_env": os.environ.get("EXTRA_ENV", ""),
                   "live_s": round(live_s, 3), "plain_errors": sum(1 for x in beside + alone if x["error"]),
                   "longs_ended_in_window": sum(1 for r in lrows if r["ended_in_window"]),
                   "long_rate_live": round(long_live / live_s, 2),
                   "long_rate_before": round(sum(r["rate_before"] for r in lrows), 2),
                   "plain_tok_s_live": round(plain_live / live_s, 2),
                   "agg_tok_s_live": round((plain_live + long_live) / live_s, 2),
                   "log": log.name, "log_from": mark1, "log_to": mark2, "log_cell_from": mark0, "log_alone_from": mark3,
                   "beside": beside, "alone": alone, "long_streams": lrows}
            with open(OUT / "mixed.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("beside", "alone", "long_streams")}), flush=True)
    finally:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()


main()
