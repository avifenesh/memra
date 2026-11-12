#!/usr/bin/env python3
"""How long does a newcomer's first token wait behind speculating streams? (memra#1227)

One server boot per call, several cells in it (CELLS). Per cell: LONGS long streamed requests
(distinct prompts, max_tokens 3000) start together and run alone for LEAD seconds, so they are in
steady continuation bursts (two or more same-model speculative sessions form a pooled cohort).
Then NEW newcomers are sent one at a time: each is a streamed chat request of NEWTOK tokens from
its own prompt, sent GAPS[i % len(GAPS)] seconds after the previous newcomer finished (a fixed
list of offsets, so the arrivals fall at different points of the streams' bursts and both arms
get the same schedule). After the last newcomer the long streams run alone for TAIL seconds and
the cell hangs up on them.

Per cell row: each newcomer's time to first token, time from first to second token, end to end
time, token count and text sha; the long streams' tokens before, during and after the newcomer
window, their rates and longest gap, and their texts (texts/, compared by prefix: a stream is
cut at a time, not a token count). The server log of the window is kept by offset (log_from,
log_to) for the tick trace.

usage: newcomer_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18227), CELLS ("2:greedy" = 2 long streams, greedy; comma list;
     class greedy or t1 (temperature 1, fixed seeds); an optional third field is the newcomers'
     class when it differs, e.g. "2:greedy:pen": pen is greedy with a frequency penalty, which
     the server admits to plain decode; an optional fourth field "long" gives each newcomer a
     prompt of about LONGWORDS words (default 4200, about 6,000 tokens), so its prime spans
     several chunks; each long prompt starts differently, so none is a prefix-cache hit), LEAD (8), NEW (12), NEWTOK (16), TAIL (4),
     GAPS ("0.9,1.7,0.5,2.3,1.1,0.3,1.9,0.7,2.9,1.3,0.1,2.1"), GAP (3: idle seconds after each
     cell), EXTRA_ENV ("K=V;K=V" added to the server's environment), TAG (row and log label;
     default ARM)
"""
import hashlib
import json
import os
import pathlib
import socket
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
PORT = int(os.environ.get("CELL_PORT", "18227"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, TAIL, GAP = (float(os.environ.get(k, d)) for k, d in (("LEAD", "8"), ("TAIL", "4"), ("GAP", "3")))
NEW, NEWTOK = int(os.environ.get("NEW", "12")), int(os.environ.get("NEWTOK", "16"))
GAPS = [float(x) for x in os.environ.get("GAPS", "0.9,1.7,0.5,2.3,1.1,0.3,1.9,0.7,2.9,1.3,0.1,2.1").split(",")]
CELLS = [c.split(":") for c in os.environ.get("CELLS", "2:greedy").split(",") if c]
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "8", "MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_TICK_TRACE": "1"}
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(";")):
    k, v = kv.split("=", 1)
    ENV[k] = v
LONG_TOPICS = ["a coastal survey", "a mountain rescue exercise", "a harbor dredging project", "a regional seed bank"]
NEW_TOPICS = ["a river delta forms", "a bridge is inspected", "bread dough rises", "a ship's hull is painted",
              "a forest fire is mapped", "a census is planned", "a violin is tuned", "a library catalogs books",
              "a city plans bus routes", "a vineyard is pruned", "a glacier is measured", "a museum lends art",
              "a printing press works", "a dam controls floods", "a lighthouse is kept", "a road is resurfaced"]
LONGWORDS = int(os.environ.get("LONGWORDS", "4200"))
LONG_SUBJECTS = ["the tide gauge", "the survey boat", "the depth sounder", "the shore marker", "the field notebook", "the weather mast",
                 "the harbor pilot", "the chart table", "the mooring line", "the signal lamp", "the supply crate", "the base station"]
LONG_VERBS = ["was checked before dawn by", "was moved to the north pier by", "was repaired on the second day by", "was logged twice by",
              "was left in the shed by", "was carried up the slope by", "was cleaned and stored by"]
LONG_PEOPLE = ["the junior surveyor", "the boat crew", "the night watch", "the visiting engineer", "the project lead", "the two students"]


def long_prompt(ci, n):
    """About LONGWORDS words of plain sentences, a different start for every (cell, newcomer)."""
    out, k, words = [f"Report {ci}-{n}, entry order {n * 7 + ci}."], n * 7 + ci * 3, 0
    while words < LONGWORDS:
        sent = (f"{LONG_SUBJECTS[k % len(LONG_SUBJECTS)].capitalize()} {LONG_VERBS[(k // 3) % len(LONG_VERBS)]} "
                f"{LONG_PEOPLE[(k // 5) % len(LONG_PEOPLE)]} on day {k % 29 + 1}.")
        out.append(sent)
        words += len(sent.split())
        k += 1
    out.append("In one short sentence, say which item was moved to the north pier most often.")
    return " ".join(out)


CLASSES = {"greedy": lambda i: {"temperature": 0}, "t1": lambda i: {"temperature": 1.0, "seed": 611 + i},
           # a greedy request with a penalty is admitted to plain decode (K=0, eligibility)
           "pen": lambda i: {"temperature": 0, "frequency_penalty": 0.3}}


def stream(body, times, done, hang_up=None):
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
        for ci, cell in enumerate(CELLS):
            longs = int(cell[0])
            cls = cell[1] if len(cell) > 1 and cell[1] else "greedy"
            ncls = cell[2] if len(cell) > 2 and cell[2] else cls
            long_new = len(cell) > 3 and cell[3] == "long"
            hang_up = threading.Event()
            ltimes, ldone, lths = [], [], []
            mark0 = log.stat().st_size
            t0 = time.perf_counter()
            for i in range(longs):
                body = {"model": "q", "max_tokens": 3000, "stream": True, **CLASSES[cls](i),
                        "messages": [{"role": "user", "content": f"Write a very long, detailed manual, at least 3000 "
                                      f"words, on how {LONG_TOPICS[i]} is planned, run and written up."}],
                        "chat_template_kwargs": {"enable_thinking": False}}
                ltimes.append([])
                ldone.append({})
                th = threading.Thread(target=stream, args=(body, ltimes[i], ldone[i], hang_up))
                th.start()
                lths.append(th)
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            t1 = time.perf_counter()
            news = []
            for n in range(NEW):
                threading.Event().wait(GAPS[n % len(GAPS)])
                body = {"model": "q", "max_tokens": NEWTOK, "stream": True, **CLASSES[ncls](100 + n),
                        "messages": [{"role": "user", "content": long_prompt(ci, n) if long_new else
                                      f"Explain how {NEW_TOPICS[(ci * 5 + n) % len(NEW_TOPICS)]}, "
                                      f"step by step, in plain words. Case {ci}-{n}."}],
                        "chat_template_kwargs": {"enable_thinking": False}}
                times, done = [], {}
                ts = time.perf_counter()
                stream(body, times, done)
                news.append({"n": n, "sent_at_s": round(ts - t1, 3),
                             "ttft_s": round(times[0] - ts, 4) if times else None,
                             "second_s": round(times[1] - times[0], 4) if len(times) > 1 else None,
                             "e2e_s": round(done["end"] - ts, 4), "chunks": len(times), "finish": done.get("finish"),
                             "error": done.get("error"),
                             "sha": hashlib.sha256(done.get("text", "").encode()).hexdigest()[:16]})
            t2 = time.perf_counter()
            mark2 = log.stat().st_size
            threading.Event().wait(TAIL)
            t3 = time.perf_counter()
            hang_up.set()
            for th in lths:
                th.join(30)
            lrows = []
            for i in range(longs):
                times = ltimes[i]
                during = [t for t in times if t1 <= t < t2]
                edges = [t1] + during + [t2]
                text = ldone[i].get("text", "")
                name = f"{ARM}-c{ci}-{cls}-l{longs}-s{i}-{int(time.time())}.txt"
                (OUT / "texts" / name).write_text(text)
                lrows.append({"stream": i, "tokens_before": sum(1 for t in times if t < t1), "tokens_during": len(during),
                              "tokens_after": sum(1 for t in times if t2 <= t < t3),
                              "rate_before": round(sum(1 for t in times if t < t1) / (t1 - t0), 2),
                              "rate_during": round(len(during) / (t2 - t1), 2),
                              "rate_after": round(sum(1 for t in times if t2 <= t < t3) / (t3 - t2), 2),
                              "longest_gap_in_window_s": round(max(b - a for a, b in zip(edges, edges[1:])), 3),
                              "itl_p50_ms": round(pct([b - a for a, b in zip(during, during[1:])], 0.5) * 1e3, 1) if len(during) > 1 else None,
                              "itl_p99_ms": round(pct([b - a for a, b in zip(during, during[1:])], 0.99) * 1e3, 1) if len(during) > 1 else None,
                              "gaps_over_150ms_s": sorted((round(b - a, 3) for a, b in zip(during, during[1:]) if b - a > 0.15), reverse=True)[:40],
                              "finish": ldone[i].get("finish"), "error": ldone[i].get("error"),
                              "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
            tt = [x["ttft_s"] for x in news if x["ttft_s"] is not None]
            sec = [x["second_s"] for x in news if x["second_s"] is not None]
            row = {"arm": ARM, "cell": ci, "class": cls, "newcomer_class": ncls, "long_prompts": long_new, "longs": longs, "extra_env": os.environ.get("EXTRA_ENV", ""),
                   "lead_s": LEAD, "window_s": round(t2 - t1, 2), "tail_s": TAIL, "newcomers": len(news),
                   "newcomer_errors": sum(1 for x in news if x["error"]),
                   "ttft_p50_s": pct(tt, 0.5), "ttft_p95_s": pct(tt, 0.95), "ttft_max_s": max(tt) if tt else None,
                   "ttft_mean_s": round(sum(tt) / len(tt), 4) if tt else None,
                   "second_p50_s": pct(sec, 0.5), "second_max_s": max(sec) if sec else None,
                   "long_tokens_during": sum(r["tokens_during"] for r in lrows),
                   "long_rate_during": round(sum(r["tokens_during"] for r in lrows) / (t2 - t1), 2),
                   "long_rate_before": round(sum(r["tokens_before"] for r in lrows) / (t1 - t0), 2),
                   "log": log.name, "log_from": mark1, "log_to": mark2, "log_cell_from": mark0,
                   "news": news, "long_streams": lrows}
            with open(OUT / "newcomer.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("news", "long_streams")}), flush=True)
            print("ttfts", [x["ttft_s"] for x in news], flush=True)
            threading.Event().wait(GAP)
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


main()
