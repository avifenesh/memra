#!/usr/bin/env python3
"""Several host-sampled plain requests at once beside speculating streams (memra#1244 review).

memra#1266 copy of research/spec-plain-round-yield-20261112/tools/riders_cell.py: the same cell,
plus a /metrics dump at the end of each boot (metrics-ARM.json).

One server boot per call, several cells in it (CELLS). Per cell: LONGS long greedy streamed
requests start together and run alone for LEAD seconds (steady continuation bursts, a pooled
cohort). Then R plain requests of class CLS start at once, each a streamed chat request of
NEWTOK tokens from its own prompt, and the cell waits for all of them. After TAIL seconds alone
the cell hangs up on the streams. Then the same R requests run at once with nothing else in
flight (the riders' own rate).

Per cell row: each plain request's TTFT, token gaps, E2E, tokens, text and sha; the window from
the first plain request's start to the last one's end ("live"); the streams' rate inside it and
outside it; the plain requests' summed rate inside it; the aggregate (streams + plain) inside it.

usage: riders_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18245), CELLS ("2:4:vendor" = 2 long greedy streams, 4 plain
     requests of class vendor; comma list), classes: vendor (the vendor's non-thinking shape:
     temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, fixed seeds), pen (greedy +
     frequency penalty 0.3), LEAD (8), NEWTOK (64), LONGTOK (6000), TAIL (3), GAP (3), EXTRA_ENV
     ("K=V;K=V"), TAG (row label; default ARM), MAX_SESSIONS (12), CTX (4096)
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
PORT = int(os.environ.get("CELL_PORT", "18245"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, TAIL, GAP = (float(os.environ.get(k, d)) for k, d in (("LEAD", "8"), ("TAIL", "3"), ("GAP", "3")))
NEWTOK, LONGTOK = int(os.environ.get("NEWTOK", "64")), int(os.environ.get("LONGTOK", "6000"))
CELLS = [c.split(":") for c in os.environ.get("CELLS", "2:4:vendor").split(",") if c]
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": os.environ.get("CTX", "4096"), "MEMRA_MAX_SESSIONS": os.environ.get("MAX_SESSIONS", "12"),
       "MEMRA_SPEC_SETUP_TRACE": "1", "MEMRA_TICK_TRACE": "1"}
for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(";")):
    k, v = kv.split("=", 1)
    ENV[k] = v
LONG_TOPICS = ["a coastal survey", "a mountain rescue exercise", "a harbor dredging project", "a regional seed bank"]
NEW_TOPICS = ["a river delta forms", "a bridge is inspected", "bread dough rises", "a ship's hull is painted",
              "a forest fire is mapped", "a census is planned", "a violin is tuned", "a library catalogs books",
              "a city plans bus routes", "a vineyard is pruned", "a glacier is measured", "a museum lends art"]
CLASSES = {"vendor": lambda i: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 911 + i},
           "pen": lambda i: {"temperature": 0, "frequency_penalty": 0.3}}


def stream(body, times, done, hang_up=None):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
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


def plain_wave(ci, nr, ncls, label):
    """nr plain requests at once; returns their rows and (first start, last end)."""
    rows, ths = [], []
    for n in range(nr):
        body = {"model": "q", "max_tokens": NEWTOK, "stream": True, **CLASSES[ncls](n),
                "messages": [{"role": "user", "content": f"Explain how {NEW_TOPICS[(ci * 3 + n) % len(NEW_TOPICS)]}, "
                              f"step by step, in plain words. Case {ci}-{n}."}],
                "chat_template_kwargs": {"enable_thinking": False}}
        rows.append(([], {}))
        th = threading.Thread(target=stream, args=(body, rows[n][0], rows[n][1]))
        th.start()
        ths.append(th)
    for th in ths:
        th.join(900)
    out = []
    for n, (times, done) in enumerate(rows):
        gaps = [round(b - a, 4) for a, b in zip(times, times[1:])]
        text = done.get("text", "")
        name = f"{ARM}-c{ci}-{label}{n}-{int(time.time())}.txt"
        (OUT / "texts" / name).write_text(text)
        out.append({"n": n, "ts": done["start"], "end": done["end"], "times": times,
                    "ttft_s": round(times[0] - done["start"], 4) if times else None,
                    "gaps_s": gaps, "itl_p50_s": pct(gaps, 0.5), "e2e_s": round(done["end"] - done["start"], 4),
                    "chunks": len(times), "finish": done.get("finish"), "error": done.get("error"),
                    "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
    return out, (min(x["ts"] for x in out), max(x["end"] for x in out))


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
            longs, nr, ncls = int(cell[0]), int(cell[1]), cell[2]
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
                th = threading.Thread(target=stream, args=(body, ltimes[i], ldone[i], hang_up))
                th.start()
                lths.append(th)
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            riders, (a, b) = plain_wave(ci, nr, ncls, "r")
            mark2 = log.stat().st_size
            threading.Event().wait(TAIL)
            t3 = time.perf_counter()
            hang_up.set()
            for th in lths:
                th.join(60)
            live_s = b - a
            lrows = []
            for i in range(longs):
                times = ltimes[i]
                text = ldone[i].get("text", "")
                name = f"{ARM}-c{ci}-l{longs}-s{i}-{int(time.time())}.txt"
                (OUT / "texts" / name).write_text(text)
                n_live = sum(1 for t in times if a <= t < b)
                n_before = sum(1 for t in times if t0 + 2 <= t < a)
                lrows.append({"stream": i, "tokens": len(times), "tokens_live": n_live,
                              "rate_live": round(n_live / live_s, 2),
                              "rate_before": round(n_before / max(1e-9, a - t0 - 2), 2),
                              "ended_in_window": bool(ldone[i].get("finish")) and ldone[i].get("end", t3) < b,
                              "finish": ldone[i].get("finish"), "error": ldone[i].get("error"),
                              "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
            threading.Event().wait(1.0)
            mark3 = log.stat().st_size
            solos, (sa, sb) = plain_wave(ci, nr, ncls, "solo")
            plain_live = sum(x["chunks"] for x in riders)
            long_live = sum(r["tokens_live"] for r in lrows)
            gaps = [g for x in riders for g in x["gaps_s"]]
            sgaps = [g for x in solos for g in x["gaps_s"]]
            for x in riders + solos:
                x.pop("times")
                x["ts"] = round(x["ts"] - a, 3)
                x["end"] = round(x["end"] - a, 3)
            row = {"arm": ARM, "cell": ci, "longs": longs, "riders": nr, "plain_class": ncls,
                   "extra_env": os.environ.get("EXTRA_ENV", ""), "newtok": NEWTOK, "live_s": round(live_s, 3),
                   "plain_errors": sum(1 for x in riders + solos if x["error"]),
                   "longs_ended_in_window": sum(1 for r in lrows if r["ended_in_window"]),
                   "plain_tok_s_live": round(plain_live / live_s, 2), "long_rate_live": round(long_live / live_s, 2),
                   "long_rate_before": round(sum(r["rate_before"] for r in lrows), 2),
                   "agg_tok_s_live": round((plain_live + long_live) / live_s, 2),
                   "ttft_p50_s": pct([x["ttft_s"] for x in riders if x["ttft_s"] is not None], 0.5),
                   "itl_p50_s": pct(gaps, 0.5), "itl_p95_s": pct(gaps, 0.95),
                   "e2e_max_s": max(x["e2e_s"] for x in riders),
                   "solo_live_s": round(sb - sa, 3), "solo_tok_s": round(sum(x["chunks"] for x in solos) / (sb - sa), 2),
                   "solo_itl_p50_s": pct(sgaps, 0.5),
                   "log": log.name, "log_from": mark1, "log_to": mark2, "log_cell_from": mark0, "log_solo_from": mark3,
                   "news": riders, "solos": solos, "long_streams": lrows}
            with open(OUT / "riders.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("news", "solos", "long_streams")}), flush=True)
            threading.Event().wait(GAP)
    finally:
        # memra#1266: the server's /metrics at the end of the boot (the memory cap's rider
        # counter among them), before it stops.
        try:
            (OUT / f"metrics-{ARM}.json").write_bytes(urllib.request.urlopen(URL + "/metrics", timeout=10).read())
        except Exception as exc:
            print(f"metrics: {exc}", flush=True)
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


main()
