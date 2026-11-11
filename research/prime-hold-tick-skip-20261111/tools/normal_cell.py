#!/usr/bin/env python3
"""Normal-concurrency cell (memra#1201, memra#1211 census): closed-loop clients at c = 1, 2, 4, 8, no one-token
flood. Shows what the cost model chose (the draft length at each admission, its verdict lines,
the demotions) and how often a session yielded at two burst entries in a row without running
(the probe's `[spec-yield] starved` marker, the trigger of the starved-demotion arm).

One server boot per call. PLAN is a comma list of "c[:knobs]" phases run in order; knobs as in
stall_cell.py ("STARVED=2+N=1"), written to KNOBS_FILE before the phase. Each phase: c clients,
each sending REQS greedy chat requests back to back (max_tokens MAXTOK) from a fixed prompt
list; client i of a phase starts at prompt (phase index * 7 + i * 3) so phases differ.

Per phase row: wall, completion tokens, aggregate tok/s, requests and errors, the K and source
of each `[spec-k]` admission line in the phase, the `[spec-cost]` verdict lines, demotions,
starved markers, bound rounds, no-round bursts, and per request the text sha.

usage: normal_cell.py OUT BINARY ARM
env: CELL_MODEL, CELL_PORT (18202), PLAN ("1,2,4,8"), REQS (3), MAXTOK (160), KNOBS_FILE, TAG,
     STREAM (1 = streamed requests with per-request TTFT and inter-token gaps), EXTRA_ENV ("K=V;K=V")
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
BIN = sys.argv[2]
ARM = os.environ.get("TAG") or sys.argv[3]
PORT = int(os.environ.get("CELL_PORT", "18202"))
URL = f"http://127.0.0.1:{PORT}"
MODEL = os.environ["CELL_MODEL"]
PLAN = [p.split(":") for p in os.environ.get("PLAN", "1,2,4,8").split(",") if p]
REQS = int(os.environ.get("REQS", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "160"))
KNOBS = os.environ.get("KNOBS_FILE", "")
EXTRA = dict(kv.split("=", 1) for kv in filter(None, os.environ.get("EXTRA_ENV", "").split(";")))
ENV = {"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"q={MODEL}", "MEMRA_ADDR": f"127.0.0.1:{PORT}",
       "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "8", "MEMRA_SPEC_SETUP_TRACE": "1"}
if KNOBS:
    ENV["PROBE1201_KNOBS"] = KNOBS
ENV.update(EXTRA)
SETUP = re.compile(r"\[spec-setup\] .* rounds=([0-9.]+)ms .* out=(\d+) cont=(true|false)")
SPECK = re.compile(r"\[spec-k\] model=\"q\" .* K=(\d+) source=(\S+)")
TOPICS = ["how a river delta forms", "how a bridge is inspected", "how bread dough rises", "how a ship's hull is painted",
          "how a forest fire is mapped", "how a census is planned", "how a violin is tuned", "how a library catalogs books",
          "how a city plans bus routes", "how a vineyard is pruned", "how a glacier is measured", "how a museum lends art",
          "how a printing press works", "how a dam controls floods", "how a lighthouse is kept", "how a road is resurfaced"]
PROMPTS = [f"Explain {t} in a careful, step by step way for a curious reader. Cover the tools, the people, "
           f"the order of the work, what can go wrong and how it is checked. Use plain words." for t in TOPICS]


STREAM = os.environ.get("STREAM", "0") == "1"


def stream_request(body):
    """Streamed chat request: (text, completion tokens = content chunks, chunk arrival times)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text, times, usage = [], [], None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    times.append(time.perf_counter())
                    text.append(piece)
    return "".join(text), (usage or {}).get("completion_tokens", len(times)), times


def client(phase, i, n, rows):
    for r in range(n):
        idx = (phase * 7 + i * 3 + r) % len(PROMPTS)
        if STREAM:
            body = {"model": "q", "max_tokens": MAXTOK, "temperature": 0,
                    "messages": [{"role": "user", "content": PROMPTS[idx]}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            t = time.perf_counter()
            try:
                text, toks, times = stream_request(body)
                gaps = [y - x for x, y in zip(times, times[1:])]
                rows.append({"prompt": idx, "s": round(time.perf_counter() - t, 3), "completion_tokens": toks,
                             "ttft_s": round(times[0] - t, 3) if times else None, "itl_ms": [round(g * 1e3, 1) for g in gaps],
                             "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
            except Exception as exc:  # recorded, never swallowed
                rows.append({"prompt": idx, "error": repr(exc)})
            continue
        body = {"model": "q", "max_tokens": MAXTOK, "stream": False, "temperature": 0,
                "messages": [{"role": "user", "content": PROMPTS[idx]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=300).read())
            text = out["choices"][0]["message"].get("content") or ""
            rows.append({"prompt": idx, "s": round(time.perf_counter() - t, 3),
                         "completion_tokens": out["usage"]["completion_tokens"],
                         "finish": out["choices"][0].get("finish_reason"),
                         "sha": hashlib.sha256(text.encode()).hexdigest()[:16]})
        except Exception as exc:  # recorded, never swallowed
            rows.append({"prompt": idx, "error": repr(exc)})


def pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


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
        for ph, step in enumerate(PLAN):
            c = int(step[0])
            knobs = step[1] if len(step) > 1 else ""
            if KNOBS:
                pathlib.Path(KNOBS).write_text("\n".join(knobs.split("+")) + "\n")
                threading.Event().wait(0.5)
            mark = log.stat().st_size
            rows = []
            t0 = time.perf_counter()
            ths = [threading.Thread(target=client, args=(ph, i, REQS, rows)) for i in range(c)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            wall = time.perf_counter() - t0
            with open(log, errors="replace") as f:
                f.seek(mark)
                chunk = f.read()
            toks = sum(r.get("completion_tokens", 0) for r in rows)
            bursts = [(float(ms), int(n)) for ms, n, cont in SETUP.findall(chunk) if cont == "true"]
            row = {"arm": ARM, "phase": ph, "c": c, "knobs": knobs, "wall_s": round(wall, 2), "completion_tokens": toks,
                   "agg_tok_s": round(toks / wall, 2), "requests": len(rows), "errors": sum(1 for r in rows if "error" in r),
                   "spec_k": [f"K{k}/{src}" for k, src in SPECK.findall(chunk)],
                   "spec_cost_lines": [l for l in chunk.splitlines() if l.startswith("[spec-cost] model=")],
                   "demotions": chunk.count("[spec-gate] demoted session"),
                   "starved_marks": chunk.count("[spec-yield] starved:"),
                   "stall_marks": chunk.count("[spec-cost] model=\"q\" stalled n="),
                   "stalled_verdicts": chunk.count("(speculation stalled at n"),
                   "bound_rounds": chunk.count("[spec-yield] bound:"),
                   "continuation_bursts": len(bursts), "no_round_bursts": sum(1 for ms, n in bursts if ms < 1.0),
                   "no_round_silent": sum(1 for ms, n in bursts if ms < 1.0 and n == 0),
                   "itl_p50_ms": pctl([g for r in rows for g in r.get("itl_ms", [])], 0.5),
                   "itl_p99_ms": pctl([g for r in rows for g in r.get("itl_ms", [])], 0.99),
                   "ttft_p50_s": pctl([r["ttft_s"] for r in rows if r.get("ttft_s") is not None], 0.5),
                   "hold_waits": sum(int(x) for x in re.findall(r"hold_waits=(\d+)", chunk)),
                   "hold_wait_ms": round(sum(float(x) for x in re.findall(r"hold_wait_ms=([0-9.]+)", chunk)), 2),
                   "ticks_priming_unprimed": sum(1 for m in re.finditer(r"\[tick\] act=\d+ int=\d+ priming=(\d+) .*prefill_single_calls=(\d+) .*prefill_batch_calls=(\d+)", chunk) if int(m.group(1)) > 0 and m.group(2) == "0" and m.group(3) == "0"),
                   "ticks": chunk.count("[tick] act="),
                   "rows": rows}
            with open(OUT / "normal.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("rows", "spec_cost_lines", "spec_k")}), flush=True)
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
