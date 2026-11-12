#!/usr/bin/env python3
"""Does a speculating request advance while short requests keep arriving? (memra#1201, memra#1211)

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
     is written to it one line per knob before the cell starts; empty = the probe's off state),
     GAP (3: idle seconds after each cell; a stall mark lives 64 s, so 70 keeps cells apart),
     SEG (0: when > 0, the row also splits the window into SEG-second segments: long tokens,
     short requests and their p50 per segment), FOLLOW ("c:gap": after each cell's tail, wait gap
     seconds, then c closed-loop clients send 3 greedy 160-token requests each, as in
     normal_cell.py; the row gets their aggregate and the admissions and verdicts they saw; a
     cell's fifth field "c/gap" sets it for that cell, e.g. "2:0:greedy::2/5"). A sixth field
     "live" keeps the long request open through the follow and hangs up after it, so the follow's
     admissions see the demoted session live (memra#1211's hold); the follow row records whether
     the long stream was still open at the follow's start and end and its tokens meanwhile.
memra#1211 census per row: stall marks set (`stalled n=`), stalled verdicts, `[spec-k]`
admissions in the window by K and source, and demotions by reason.
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
SPECK = re.compile(r"\[spec-k\] model=\"q\" .* K=(\d+) source=(\S+)")
GAP = float(os.environ.get("GAP", "3"))
# memra#1227: LONGS long streams per cell (default 1). Streams 2.. use their own prompts and are
# reported under "extra_longs"; with two or more the streams form a pooled cohort.
LONGS = int(os.environ.get("LONGS", "1"))
EXTRA_PROMPTS = [f"Write a very long, detailed manual, at least 3000 words, on how {t} is planned, run and written up."
                 for t in ("a mountain rescue exercise", "a harbor dredging project", "a regional seed bank")]
SEG = float(os.environ.get("SEG", "0"))
FOLLOW = os.environ.get("FOLLOW", "")
TOPICS = ["how a river delta forms", "how a bridge is inspected", "how bread dough rises", "how a ship's hull is painted",
          "how a forest fire is mapped", "how a census is planned", "how a violin is tuned", "how a library catalogs books",
          "how a city plans bus routes", "how a vineyard is pruned", "how a glacier is measured", "how a museum lends art",
          "how a printing press works", "how a dam controls floods", "how a lighthouse is kept", "how a road is resurfaced"]
PROMPTS = [f"Explain {t} in a careful, step by step way for a curious reader. Cover the tools, the people, "
           f"the order of the work, what can go wrong and how it is checked. Use plain words." for t in TOPICS]
CLASSES = {"greedy": {"temperature": 0}, "t1": {"temperature": 1.0, "seed": 611}}
PROMPT = "Write a very long, detailed manual, at least 3000 words, on how a coastal survey is planned, run and written up."


def short(tag, stop, think, lat, errors, starts):
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
            starts.append(t)
        except Exception as exc:  # counted and kept, never swallowed
            errors.append(repr(exc))
        if think > 0:
            stop.wait(think)


def long_request(samp, times, done, hang_up, prompt=None):
    body = {"model": "q", "max_tokens": 3000, "stream": True, **samp,
            "messages": [{"role": "user", "content": prompt or PROMPT}], "chat_template_kwargs": {"enable_thinking": False}}
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


def follow_client(i, rows):
    for r in range(3):
        idx = (i * 3 + r) % len(PROMPTS)
        body = {"model": "q", "max_tokens": 160, "stream": False, "temperature": 0,
                "messages": [{"role": "user", "content": PROMPTS[idx]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=300).read())
            rows.append({"prompt": idx, "completion_tokens": out["usage"]["completion_tokens"],
                         "sha": hashlib.sha256((out["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:16]})
        except Exception as exc:  # recorded, never swallowed
            rows.append({"prompt": idx, "error": repr(exc)})


def census(chunk):
    speck = {}
    for k, src in SPECK.findall(chunk):
        speck[f"K{k}/{src}"] = speck.get(f"K{k}/{src}", 0) + 1
    dem = {}
    for line in chunk.splitlines():
        if line.startswith("[spec-gate] demoted"):
            why = line.split(": ", 1)[1].split(" (model")[0] if ": " in line else line
            dem[why] = dem.get(why, 0) + 1
    return {"stall_marks": chunk.count("[spec-cost] model=\"q\" stalled n="),
            "stall_mark_lines": [l for l in chunk.splitlines() if " stalled n=" in l and l.startswith("[spec-cost]")][:8],
            "stalled_verdicts": chunk.count("(speculation stalled at n"),
            "hold_waits": sum(int(x) for x in re.findall(r"(?:hold|form)_waits=(\d+)", chunk)),
            "hold_wait_ms": round(sum(float(x) for x in re.findall(r"(?:hold|form)_wait_ms=([0-9.]+)", chunk)), 2),
            "ticks": chunk.count("[tick] act="),
            "stall_refreshes": sum(int(x) for x in re.findall(r"stall_refresh=(\d+)", chunk)),
            "held_wave_marks": chunk.count("the wave holds a session"),
            "spec_k": speck, "demotion_reasons": dem}


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
            extras = []
            for x in range(LONGS - 1):
                xt, xd = [], {}
                xth = threading.Thread(target=long_request, args=(CLASSES[cls], xt, xd, hang_up, EXTRA_PROMPTS[x]))
                xth.start()
                extras.append((xt, xd, xth))
            threading.Event().wait(LEAD)
            mark1 = log.stat().st_size
            stop, lat, errors, starts = threading.Event(), [], [], []
            ths = [threading.Thread(target=short, args=(f"{cls}{clients}x{int(think_ms)}c{i}", stop, think_ms / 1e3, lat, errors, starts))
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
            live = len(cell) > 5 and cell[5] == "live"
            if not live:
                hang_up.set()
                th.join(30)
            before = [t for t in times if t < t1]
            during = [t for t in times if t1 <= t < t2]
            after = [t for t in times if t2 <= t < t3]
            edges = [t1] + during + [t2]
            gap = max(b - a for a, b in zip(edges, edges[1:]))
            itl = [y - x for x, y in zip(during, during[1:])]
            itl_all = [y - x for x, y in zip(times, times[1:])]
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
            segs = []
            if SEG > 0:
                a = t1
                while a < t2 - 0.5:
                    b = min(a + SEG, t2)
                    sl = [l for l, st in zip(lat, starts) if a <= st < b]
                    segs.append({"from_s": round(a - t1, 1), "to_s": round(b - t1, 1),
                                 "long_tokens": sum(1 for t in during if a <= t < b),
                                 "short_requests": len(sl), "short_p50_s": pct(sl, 0.5)})
                    a = b
            follow = None
            fspec = cell[4].replace("/", ":") if len(cell) > 4 and cell[4] else FOLLOW
            if fspec:
                fc, fgap = fspec.split(":")
                threading.Event().wait(float(fgap))
                fmark = log.stat().st_size
                live_at_start = th.is_alive() and "finish" not in done
                frows, ft0 = [], time.perf_counter()
                fths = [threading.Thread(target=follow_client, args=(i, frows)) for i in range(int(fc))]
                for t in fths:
                    t.start()
                for t in fths:
                    t.join()
                fwall = time.perf_counter() - ft0
                live_at_end = th.is_alive() and "finish" not in done
                long_during_follow = sum(1 for t in times if ft0 <= t)
                with open(log, errors="replace") as f:
                    f.seek(fmark)
                    fchunk = f.read()
                ftoks = sum(r.get("completion_tokens", 0) for r in frows)
                follow = {"clients": int(fc), "after_s": float(fgap), "wall_s": round(fwall, 2), "completion_tokens": ftoks,
                          "agg_tok_s": round(ftoks / fwall, 2), "errors": sum(1 for r in frows if "error" in r),
                          "spec_cost_lines": [l for l in fchunk.splitlines() if l.startswith("[spec-cost] model=")],
                          **census(fchunk), "rows": frows, "long_live": live,
                          "long_open_at_start": live_at_start, "long_open_at_end": live_at_end,
                          "long_tokens_during_follow": long_during_follow}
            if live:
                hang_up.set()
                th.join(30)
                # the long request's text exists once its thread ends
                text = done.get("text", "")
                sha = hashlib.sha256(text.encode()).hexdigest()[:16]
                (OUT / "texts" / f"{name}.txt").write_text(text)
            with open(log, errors="replace") as f:
                f.seek(mark1)
                cwin = f.read(mark2 - mark1)
            hang_up.set()
            extra_rows = []
            for x, (xt, xd, xth) in enumerate(extras):
                xth.join(30)
                xtext = xd.get("text", "")
                xname = f"{name}-x{x + 1}.txt"
                (OUT / "texts" / xname).write_text(xtext)
                xdur = [t for t in xt if t1 <= t < t2]
                xedges = [t1] + xdur + [t2]
                extra_rows.append({"stream": x + 1, "tokens_before": sum(1 for t in xt if t < t1), "tokens_during": len(xdur),
                                   "tokens_after": sum(1 for t in xt if t2 <= t < t3),
                                   "longest_gap_in_window_s": round(max(b - a for a, b in zip(xedges, xedges[1:])), 2),
                                   "finish": xd.get("finish"), "error": xd.get("error"),
                                   "sha": hashlib.sha256(xtext.encode()).hexdigest()[:16], "text_file": f"texts/{xname}"})
            row = {"arm": ARM, "gate": GATE, "extra_env": os.environ.get("EXTRA_ENV", ""), "class": cls, "knobs": knobs,
                   "clients": clients, "think_ms": think_ms, "lead_s": LEAD, "window_s": round(t2 - t1, 2), "tail_s": TAIL,
                   "tokens_before": len(before), "tokens_during": len(during), "tokens_after": len(after),
                   "rate_before": round(len(before) / (t1 - t0), 2), "rate_during": round(len(during) / (t2 - t1), 2),
                   "rate_after": round(len(after) / (t3 - t2), 2), "longest_gap_in_window_s": round(gap, 2),
                   "long_itl_p50_ms": round(pct(itl, 0.5) * 1e3, 1) if itl else None,
                   "long_itl_p99_ms": round(pct(itl, 0.99) * 1e3, 1) if itl else None,
                   "long_itl_all_p50_ms": round(pct(itl_all, 0.5) * 1e3, 1) if itl_all else None,
                   "long_itl_all_p99_ms": round(pct(itl_all, 0.99) * 1e3, 1) if itl_all else None,
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
                   "error": done.get("error"), "census_window": census(cwin), "census_cell": census(whole),
                   "segments": segs, "follow": follow, "longs": LONGS, "extra_longs": extra_rows}
            with open(OUT / "stall.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("spec_cost_lines", "follow", "census_cell")}), flush=True)
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
