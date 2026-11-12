#!/usr/bin/env python3
"""What does a plain request that cannot ride get beside speculating streams? (memra#1244)

One server boot per call, several cells in it (CELLS). Per cell: LONGS long streamed requests
(distinct prompts, max_tokens LONGTOK) start together and run alone for LEAD seconds, so they
are in steady continuation bursts. Then NEW plain requests are sent one at a time, each a
streamed chat request of NEWTOK tokens from its own prompt, sent GAPS[i % len(GAPS)] seconds
after the previous one finished (fixed offsets: both arms get the same schedule). After the
last one the long streams run alone for TAIL seconds and the cell hangs up on them. Then SOLO
plain requests of the same class run alone (nothing else in flight): the request's own rate.

Per cell row: each plain request's time to first token, every gap between its tokens, end to
end time, token count, text and sha; the long streams' tokens while a plain request was live
and while none was (the idle gaps, lead and tail), their rates and longest gap, and their
texts (texts/, compared by prefix across arms: a stream is cut at a time, not a token count).
The server log of the window is kept by offset (log_from, log_to) for the tick trace.

usage: plain_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18244), CELLS ("2:greedy:pen" = 2 long greedy streams, plain
     requests of class pen; comma list; long class greedy or t1 (temperature 1, fixed seeds);
     plain class pen (greedy + frequency penalty), ppen (greedy + presence penalty), rpen
     (greedy + repetition penalty), t1json (temperature 1, fixed seed, JSON object output: plain
     with a grammar mask), greedy or t1 (speculative newcomers, the control)),
     LEAD (8), NEW (6), NEWTOK (24), LONGTOK (6000), TAIL (4), SOLO (2),
     GAPS ("0.9,1.7,0.5,2.3,1.1,0.3"), GAP (3: idle seconds after each cell),
     EXTRA_ENV ("K=V;K=V" added to the server's environment), TAG (row and log label; default ARM),
     PAD (0: that many numbered filler sentences, about 20 tokens each and distinct per request,
     go in front of each plain request's prompt: a long prime; the solo request of the same n
     gets its own filler, so nothing is restored from the first one's cache)
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
PORT = int(os.environ.get("CELL_PORT", "18244"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
LEAD, TAIL, GAP = (float(os.environ.get(k, d)) for k, d in (("LEAD", "8"), ("TAIL", "4"), ("GAP", "3")))
NEW, NEWTOK = int(os.environ.get("NEW", "6")), int(os.environ.get("NEWTOK", "24"))
LONGTOK, SOLO = int(os.environ.get("LONGTOK", "6000")), int(os.environ.get("SOLO", "2"))
PAD = int(os.environ.get("PAD", "0"))
GAPS = [float(x) for x in os.environ.get("GAPS", "0.9,1.7,0.5,2.3,1.1,0.3").split(",")]
CELLS = [c.split(":") for c in os.environ.get("CELLS", "2:greedy:pen").split(",") if c]
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
CLASSES = {"greedy": lambda i: {"temperature": 0}, "t1": lambda i: {"temperature": 1.0, "seed": 611 + i},
           # greedy with a penalty is admitted to plain decode (K=0, eligibility) and samples on the host
           "pen": lambda i: {"temperature": 0, "frequency_penalty": 0.3},
           "ppen": lambda i: {"temperature": 0, "presence_penalty": 0.5},
           "rpen": lambda i: {"temperature": 0, "repetition_penalty": 1.1},
           # sampled and constrained: plain decode with a grammar mask, never a rider
           "t1json": lambda i: {"temperature": 1.0, "seed": 811 + i, "response_format": {"type": "json_object"}}}


def stream(body, times, done, hang_up=None):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = []
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


def plain_request(ci, n, ncls, label):
    pad = "".join(f"Note {label}-{ci}-{n}-{k}: station {k * 7 + n} logged {k * 13 % 97} readings and {k % 11} faults on day {k}.\n"
                  for k in range(PAD))
    body = {"model": "q", "max_tokens": NEWTOK, "stream": True, **CLASSES[ncls](100 + n),
            "messages": [{"role": "user", "content": f"{pad}Explain how {NEW_TOPICS[(ci * 5 + n) % len(NEW_TOPICS)]}, "
                          f"step by step, in plain words. Case {ci}-{n}."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    times, done = [], {}
    ts = time.perf_counter()
    stream(body, times, done)
    gaps = [round(b - a, 4) for a, b in zip(times, times[1:])]
    text = done.get("text", "")
    name = f"{ARM}-c{ci}-{label}{n}-{int(time.time())}.txt"
    (OUT / "texts" / name).write_text(text)
    return {"n": n, "ts": ts, "end": done["end"], "first": times[0] if times else done["end"],
            "ttft_s": round(times[0] - ts, 4) if times else None,
            "gaps_s": gaps, "itl_p50_s": pct(gaps, 0.5), "itl_max_s": max(gaps) if gaps else None,
            "e2e_s": round(done["end"] - ts, 4), "chunks": len(times), "finish": done.get("finish"),
            "error": done.get("error"), "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "text_file": f"texts/{name}"}


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
            ncls = cell[2] if len(cell) > 2 and cell[2] else "pen"
            hang_up = threading.Event()
            ltimes, ldone, lths = [], [], []
            mark0 = log.stat().st_size
            t0 = time.perf_counter()
            for i in range(longs):
                body = {"model": "q", "max_tokens": LONGTOK, "stream": True, **CLASSES[cls](i),
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
                news.append(plain_request(ci, n, ncls, "p"))
            t2 = time.perf_counter()
            mark2 = log.stat().st_size
            threading.Event().wait(TAIL)
            t3 = time.perf_counter()
            hang_up.set()
            for th in lths:
                th.join(60)
            live = [(x["ts"], x["end"]) for x in news]
            live_s = sum(b - a for a, b in live)
            idle_s = (t3 - t0) - live_s
            lrows = []
            for i in range(longs):
                times = ltimes[i]
                in_live = [t for t in times if any(a <= t < b for a, b in live)]
                during = [t for t in times if t1 <= t < t2]
                edges = [t1] + during + [t2]
                text = ldone[i].get("text", "")
                name = f"{ARM}-c{ci}-{cls}-l{longs}-s{i}-{int(time.time())}.txt"
                (OUT / "texts" / name).write_text(text)
                ended_in_window = bool(ldone[i].get("finish")) and ldone[i].get("end", t3) < t2
                lrows.append({"stream": i, "tokens": len(times), "tokens_live": len(in_live),
                              "tokens_idle": sum(1 for t in times if t < t3) - len(in_live),
                              "rate_live": round(len(in_live) / live_s, 2) if live_s else None,
                              "rate_idle": round((sum(1 for t in times if t < t3) - len(in_live)) / idle_s, 2),
                              "rate_before": round(sum(1 for t in times if t < t1) / (t1 - t0), 2),
                              "longest_gap_in_window_s": round(max(b - a for a, b in zip(edges, edges[1:])), 3),
                              "itl_p50_ms": round(pct([b - a for a, b in zip(during, during[1:])], 0.5) * 1e3, 1) if len(during) > 1 else None,
                              "itl_p99_ms": round(pct([b - a for a, b in zip(during, during[1:])], 0.99) * 1e3, 1) if len(during) > 1 else None,
                              "ended_in_window": ended_in_window,
                              "finish": ldone[i].get("finish"), "error": ldone[i].get("error"),
                              "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text_file": f"texts/{name}"})
            for x in news:  # the streams' tokens while the request waited for its first token
                x["long_tokens_to_first"] = sum(1 for times in ltimes for t in times if x["ts"] <= t < x["first"])
                x["long_rate_to_first"] = round(x["long_tokens_to_first"] / max(1e-9, x["first"] - x["ts"]), 2)
            # the plain request alone: nothing else in flight
            threading.Event().wait(1.0)
            mark3 = log.stat().st_size
            solos = [plain_request(ci, n, ncls, "solo") for n in range(SOLO)]
            for x in news + solos:
                x["first"] = round(x["first"] - t1, 3)
                x["ts"] = round(x["ts"] - t1, 3)
                x["end"] = round(x["end"] - t1, 3)
            tt = [x["ttft_s"] for x in news if x["ttft_s"] is not None]
            gaps = [g for x in news for g in x["gaps_s"]]
            sgaps = [g for x in solos for g in x["gaps_s"]]
            row = {"arm": ARM, "cell": ci, "class": cls, "plain_class": ncls, "longs": longs,
                   "extra_env": os.environ.get("EXTRA_ENV", ""), "lead_s": LEAD, "window_s": round(t2 - t1, 2),
                   "live_s": round(live_s, 2), "tail_s": TAIL, "plain_requests": len(news), "newtok": NEWTOK, "pad": PAD,
                   "plain_errors": sum(1 for x in news + solos if x["error"]),
                   "longs_ended_in_window": sum(1 for r in lrows if r["ended_in_window"]),
                   "ttft_p50_s": pct(tt, 0.5), "ttft_max_s": max(tt) if tt else None,
                   "itl_p50_s": pct(gaps, 0.5), "itl_p95_s": pct(gaps, 0.95), "itl_max_s": max(gaps) if gaps else None,
                   "itl_mean_s": round(sum(gaps) / len(gaps), 4) if gaps else None,
                   "e2e_p50_s": pct([x["e2e_s"] for x in news], 0.5), "e2e_max_s": max(x["e2e_s"] for x in news) if news else None,
                   "plain_tok_s_live": round(sum(x["chunks"] for x in news) / live_s, 2) if live_s else None,
                   "long_rate_live": round(sum(r["tokens_live"] for r in lrows) / live_s, 2) if live_s else None,
                   "long_rate_idle": round(sum(r["tokens_idle"] for r in lrows) / idle_s, 2),
                   "long_rate_before": round(sum(r["rate_before"] for r in lrows), 2),
                   "solo_ttft_s": [x["ttft_s"] for x in solos], "solo_itl_p50_s": pct(sgaps, 0.5),
                   "solo_e2e_s": [x["e2e_s"] for x in solos],
                   "log": log.name, "log_from": mark1, "log_to": mark2, "log_cell_from": mark0, "log_solo_from": mark3,
                   "news": news, "solos": solos, "long_streams": lrows}
            with open(OUT / "plain.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("news", "solos", "long_streams")}), flush=True)
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
