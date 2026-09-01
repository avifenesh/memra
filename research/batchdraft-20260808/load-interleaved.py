#!/usr/bin/env python3
"""Controlled c=4 speculative-serving load for the batchdraft anatomy study.

Two c=4 arms are rep-major interleaved. ``sync`` sends four byte-identical greedy
prompts (the best-case round-alignment envelope); ``divergent`` sends four distinct
prompts (a ragged-acceptance stress). Every request has a stable ``trace_id`` so the
server's diagnostic phase records can be joined back to the client point.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import statistics
import threading
import time
import urllib.request


BASE_PROMPT = (
    "Summarize the operational state of a GPU serving cluster in exactly three "
    "sentences, then list four risks. Context follows. "
    + (
        "The quick brown fox jumps over the lazy dog while the seasoned engineer "
        "measures throughput, latency, and saturation across every replica. "
    )
    * 8
)

DIVERGENT_SUFFIXES = (
    " Emphasize admission control, queue fairness, and tail latency.",
    " Emphasize kernel occupancy, memory traffic, and launch overhead.",
    " Emphasize cache locality, request shape, and batching efficiency.",
    " Emphasize failure recovery, observability, and operator safeguards.",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def request_one(
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    trace_id: str,
    barrier: threading.Barrier,
    timeout: float,
) -> dict:
    body = {
        "model": model,
        "prompt": prompt,
        "chat": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "trace_id": trace_id,
        # Prevent the prefix pool from turning a later point into a cached-prefix arm.
        "cache_salt": trace_id,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-lane": "harvest"},
    )
    barrier.wait()
    start_utc = utc_now()
    start_ns = time.monotonic_ns()
    payload = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
            response_id = response.headers.get("x-request-id")
        end_ns = time.monotonic_ns()
        if "choices" in payload:
            shape = "openai"
            text = payload["choices"][0]["text"]
            usage = payload.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens", 0))
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            finish_reason = payload["choices"][0].get("finish_reason")
            response_id = payload.get("id") or response_id
        else:
            # MEMRA_COMPAT is intentionally unset in the measured process, so the native
            # validation shape is expected: text/tokens/stop_reason plus direct usage fields.
            shape = "native"
            text = payload["text"]
            completion_tokens = int(payload["n_tokens"])
            prompt_tokens = int(payload["prompt_tokens"])
            finish_reason = payload.get("stop_reason")
        return {
            "kind": "request",
            "trace_id": trace_id,
            "ok": True,
            "start_utc": start_utc,
            "end_utc": utc_now(),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "latency_s": (end_ns - start_ns) / 1e9,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "finish_reason": finish_reason,
            "response_id": response_id,
            "response_shape": shape,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    except Exception as exc:  # the raw receipt must retain the exact client failure
        end_ns = time.monotonic_ns()
        return {
            "kind": "request",
            "trace_id": trace_id,
            "ok": False,
            "start_utc": start_utc,
            "end_utc": utc_now(),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "latency_s": (end_ns - start_ns) / 1e9,
            "error": f"{type(exc).__name__}: {exc}",
            "payload_keys": sorted(payload) if isinstance(payload, dict) else None,
        }


def run_point(args: argparse.Namespace, rep: int, arm: str) -> tuple[dict, list[dict]]:
    prompts = (
        [BASE_PROMPT] * args.concurrency
        if arm == "sync"
        else [BASE_PROMPT + DIVERGENT_SUFFIXES[i] for i in range(args.concurrency)]
    )
    barrier = threading.Barrier(args.concurrency)
    point_start_utc = utc_now()
    point_start_ns = time.monotonic_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_one,
                args.base,
                args.model,
                prompts[i],
                args.max_tokens,
                f"r{rep}-{arm}-q{i}",
                barrier,
                args.timeout,
            )
            for i in range(args.concurrency)
        ]
        rows = [future.result() for future in futures]
    point_end_ns = time.monotonic_ns()
    oks = [row for row in rows if row["ok"]]
    wall_s = (point_end_ns - point_start_ns) / 1e9
    starts = [row["start_ns"] for row in rows]
    point = {
        "kind": "point",
        "rep": rep,
        "arm": arm,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "start_utc": point_start_utc,
        "end_utc": utc_now(),
        "wall_s": wall_s,
        "start_skew_ms": (max(starts) - min(starts)) / 1e6,
        "n_ok": len(oks),
        "n_err": len(rows) - len(oks),
        "completion_tokens": sum(row.get("completion_tokens", 0) for row in oks),
        "aggregate_output_tok_s": (
            sum(row.get("completion_tokens", 0) for row in oks) / wall_s
            if wall_s
            else 0.0
        ),
        "latency_median_s": statistics.median(row["latency_s"] for row in oks)
        if oks
        else None,
        "traces": [row["trace_id"] for row in rows],
    }
    return point, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--cooldown", type=float, default=1.0)
    args = parser.parse_args()
    if args.concurrency != 4:
        parser.error("this frozen study harness requires concurrency=4")

    with open(args.out, "w", encoding="utf-8") as output:
        def emit(row: dict) -> None:
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            print(line, file=output, flush=True)

        emit({
            "kind": "protocol",
            "utc": utc_now(),
            "reps": args.reps,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "order": "odd=sync,divergent; even=divergent,sync",
            "sync_contract": "four identical greedy prompts with isolated cache salts",
            "divergent_contract": "four prompt suffixes, greedy, isolated cache salts",
        })

        # Unscored graph/cache warmup at the scored concurrency.
        warm_args = argparse.Namespace(**vars(args))
        warm_args.max_tokens = min(32, args.max_tokens)
        point, rows = run_point(warm_args, 0, "sync")
        point["arm"] = "warmup"
        emit(point)
        for row in rows:
            emit(row)
        if any(not row["ok"] for row in rows):
            return 1
        time.sleep(args.cooldown)

        failures = 0
        for rep in range(1, args.reps + 1):
            order = ("sync", "divergent") if rep % 2 else ("divergent", "sync")
            for arm in order:
                point, rows = run_point(args, rep, arm)
                emit(point)
                for row in rows:
                    emit(row)
                    failures += int(not row["ok"])
                time.sleep(args.cooldown)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
