#!/usr/bin/env python3
"""Agentic replay with per-request rows (memra#1093 follow-up, the DSpark-pool grow).

The same workload, bodies and phases as the 2026-11-05 decision cell's `ab_agent.py run` (a prime
pass at max_tokens 1, then the timed pass), so its aggregates compare with the earlier receipts:
aggregate generated tok/s over the timed pass and the per-user median. Two changes: each request
is recorded on its own row (response id, HTTP status or error, TTFT, wall, usage, a sha256 of the
generated text), and a refused or timed-out request is recorded instead of failing the cell, so a
cell reports its HTTP 400 and 408 counts. The response id joins a row to the server log's
`[meter] admit id=` segment, which names the pool path that request took.

The workload is derived from private agent sessions and never leaves the machine: rows carry no
prompt or generated text, only lengths, ids and hashes.

  python3 agent_turns.py WORKLOAD.json OUTDIR ARM --users N [--think 1] [--temperature 1.0]
      [--seed 1234] [--max-tokens 768]
Env: BENCH_BASE (default http://127.0.0.1:8093), BENCH_MODEL (default q38), CLIENT_MAX_TIME (s).
"""
import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8093")
MODEL = os.environ.get("BENCH_MODEL", "q38")
MAX_TIME = float(os.environ.get("CLIENT_MAX_TIME", "1800"))


def stream(body):
    """One streamed chat completion. Returns a row; never raises on an HTTP or stream error."""
    data = json.dumps(dict(body, stream=True, stream_options={"include_usage": True})).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data, {"Content-Type": "application/json"})
    started = time.time()
    t0 = time.perf_counter()
    row = {"started_unix": started, "status": 200, "error": None, "id": None, "ttft_s": None,
           "usage": None, "finish": None}
    text, reasoning, calls = [], [], []
    try:
        with urllib.request.urlopen(req, timeout=MAX_TIME) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if line == "data: [DONE]":
                    break
                if not line.startswith("data:"):
                    continue
                value = json.loads(line[5:])
                if value.get("error"):
                    row["error"] = str(value["error"])[:300]
                    break
                row["id"] = row["id"] or value.get("id")
                row["usage"] = value.get("usage") or row["usage"]
                for choice in value.get("choices", []):
                    row["finish"] = choice.get("finish_reason") or row["finish"]
                    d = choice.get("delta", {})
                    chunk = d.get("content") or ""
                    thought = d.get("reasoning_content") or d.get("reasoning") or ""
                    if (chunk or thought or d.get("tool_calls")) and row["ttft_s"] is None:
                        row["ttft_s"] = time.perf_counter() - t0
                    text.append(chunk)
                    reasoning.append(thought)
                    for c in d.get("tool_calls", []) or []:
                        calls.append(json.dumps(c.get("function", {}), sort_keys=True))
    except urllib.error.HTTPError as e:
        row["status"] = e.code
        try:
            row["error"] = e.read().decode(errors="replace")[:300]
        except Exception:  # noqa: BLE001 - the status is the record
            row["error"] = str(e)[:300]
    except Exception as e:  # noqa: BLE001 - a dropped stream is recorded, not raised
        row["status"] = -1
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["wall_s"] = time.perf_counter() - t0
    row["finished_unix"] = started + row["wall_s"]
    out = "".join(text) + "\x00" + "".join(reasoning) + "\x00" + "\x01".join(calls)
    row["text_sha"] = hashlib.sha256(out.encode()).hexdigest()[:16]
    row["ok"] = row["status"] == 200 and row["error"] is None and row["usage"] is not None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("outdir")
    ap.add_argument("arm")
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    work = json.loads(Path(a.workload).read_text())
    kw = {"enable_thinking": True} if a.think else {"enable_thinking": False}
    bodies = [{"model": MODEL, "messages": w["messages"], "tools": w["tools"], "max_tokens": a.max_tokens,
               "temperature": a.temperature, "chat_template_kwargs": kw, "seed": a.seed + k}
              for k, w in enumerate(work)]
    # Prime pass, as ab_agent.py: prefill each body once so the timed pass measures decode.
    with ThreadPoolExecutor(a.users) as pool:
        primes = list(pool.map(lambda b: stream(dict(b, max_tokens=1)), bodies))
    t0 = time.time()
    with ThreadPoolExecutor(a.users) as pool:
        rows = list(pool.map(stream, bodies))
    wall = time.time() - t0
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    tag = f"{a.arm}-c{a.users}"
    with (out / f"{tag}.requests.jsonl").open("w") as f:
        for k, (w, r, p) in enumerate(zip(work, rows, primes)):
            sess = hashlib.sha256(w["session"].encode()).hexdigest()[:10]
            seen[sess] = seen.get(sess, 0) + 1
            r.update({"k": k, "session": sess, "session_turn": seen[sess], "prime_status": p["status"],
                      "prime_id": p["id"]})
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if r["ok"]]
    toks = sum(r["usage"]["completion_tokens"] for r in ok)
    first = [r["started_unix"] + r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    span = (max(r["finished_unix"] for r in ok) - min(first)) if ok and first else 0
    gen = []
    for r in ok:
        n = r["usage"]["completion_tokens"]
        if r["ttft_s"] is not None and r["wall_s"] > r["ttft_s"] and n > 1:
            gen.append((n - 1) / (r["wall_s"] - r["ttft_s"]))
    ttft = sorted(r["ttft_s"] for r in ok if r["ttft_s"] is not None)
    cell = {"arm": a.arm, "users": a.users, "max_tokens": a.max_tokens, "seed": a.seed, "think": a.think,
            "temperature": a.temperature, "turns": len(rows), "ok": len(ok),
            "http_400": sum(r["status"] == 400 for r in rows), "http_408": sum(r["status"] == 408 for r in rows),
            "other_failures": sum((not r["ok"]) and r["status"] not in (400, 408) for r in rows),
            "tokens": toks, "wall_s": round(wall, 1),
            "aggregate_gen_tps": round(toks / span, 2) if span else None,
            "per_user_gen_tps_median": round(statistics.median(gen), 2) if gen else None,
            "ttft_median_s": round(statistics.median(ttft), 3) if ttft else None,
            "ttft_sum_s": round(sum(ttft), 2),
            "e2e_sum_s": round(sum(r["wall_s"] for r in ok), 2)}
    with (out / "cells.jsonl").open("a") as f:
        f.write(json.dumps(cell) + "\n")
    print(json.dumps(cell), flush=True)


if __name__ == "__main__":
    main()
