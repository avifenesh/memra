#!/usr/bin/env python3
"""Barrier-synchronized, distinct-prefix prime bursts against memra-server."""

import argparse
import concurrent.futures
import datetime
import json
import math
import pathlib
import statistics
import threading
import time
import urllib.request


def utc_from_epoch(value):
    return datetime.datetime.fromtimestamp(
        value, datetime.timezone.utc
    ).isoformat()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def prompt_text(args, burst, request):
    """A normal-text 4k-class prompt whose first tokens differ per request."""
    return (
        f"Distinct request {args.label}-{burst}-{request}. Context begins. "
        f"{args.prompt_text} End of context. Reply with the single word OK."
    )


def stream_completion(base, body, timeout):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    request_id = None
    usage = {}
    first_visible = None
    visible_chunks = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(json.dumps(event["error"], sort_keys=True))
            request_id = event.get("id") or request_id
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if piece:
                    first_visible = first_visible or time.monotonic()
                    visible_chunks += 1
    return request_id, usage, first_visible, visible_chunks


def one_request(args, burst, request, ready, go):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text(args, burst, request)}],
        "reasoning": {"enabled": False},
        "max_ctx": 8192,
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": f"{args.label}-b{burst}-q{request}-{time.time_ns()}",
    }
    ready.wait()
    go.wait()
    started = time.monotonic()
    request_id, usage, first_visible, visible_chunks = stream_completion(
        args.base, body, args.timeout
    )
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError("request completed without a visible token")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or abs(prompt_tokens - args.prompt_tokens) > args.prompt_tolerance
        or cached != 0
    ):
        raise RuntimeError(
            "unexpected usage: "
            f"prompt={prompt_tokens} target={args.prompt_tokens}+/-{args.prompt_tolerance} "
            f"cached={cached}"
        )
    return {
        "id": request_id,
        "request": request,
        "client_ttft_s": first_visible - started,
        "client_wall_s": ended - started,
        "first_visible_at": first_visible,
        "visible_chunks": visible_chunks,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
    }


def burst(args, burst_index, concurrency, repeat):
    ready = threading.Barrier(concurrency + 1)
    go = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, args, burst_index, request, ready, go)
            for request in range(concurrency)
        ]
        ready.wait()
        started_epoch = time.time()
        started = time.monotonic()
        go.set()
        rows = [future.result() for future in futures]
    ended = max(row["first_visible_at"] for row in rows)
    wall = ended - started
    ttfts = [row["client_ttft_s"] for row in rows]
    prompt_counts = [row["prompt_tokens"] for row in rows]
    total_prompt = sum(prompt_counts)
    summary = {
        "kind": "summary",
        "label": args.label,
        "burst": burst_index,
        "concurrency": concurrency,
        "repeat": repeat,
        "n_requests": len(rows),
        "prompt_tokens_each_min": min(prompt_counts),
        "prompt_tokens_each_max": max(prompt_counts),
        "aggregate_prompt_tokens": total_prompt,
        "burst_start_utc": utc_from_epoch(started_epoch),
        "last_first_token_utc": utc_from_epoch(started_epoch + wall),
        "wall_to_last_first_token_s": wall,
        "client_aggregate_prompt_tps": total_prompt / wall,
        "client_ttft_p50_s": statistics.median(ttfts),
        "client_ttft_p95_s": percentile(ttfts, 95),
        "client_ttft_min_s": min(ttfts),
        "client_ttft_max_s": max(ttfts),
        "request_ids": [row["id"] for row in rows],
    }
    return rows, summary


def emit_burst(output, args, burst_index, concurrency, repeat, excluded=False):
    rows, summary = burst(args, burst_index, concurrency, repeat)
    summary["excluded"] = excluded
    for row in rows:
        row.pop("first_visible_at")
        output.write(
            json.dumps(
                {
                    "kind": "request",
                    "label": args.label,
                    "burst": burst_index,
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "excluded": excluded,
                    **row,
                },
                sort_keys=True,
            )
            + "\n"
        )
    output.write(json.dumps(summary, sort_keys=True) + "\n")
    output.flush()
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="q9")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--prompt-tolerance", type=int, default=128)
    parser.add_argument("--order", default="1,2,4,4,2,1,2,4,1")
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    order = [int(value) for value in args.order.split(",")]
    if args.prompt_tokens < 1 or args.prompt_tolerance < 0 or any(value not in (1, 2, 4) for value in order):
        parser.error("prompt-tokens must be positive and order may contain only 1,2,4")
    args.prompt_text = pathlib.Path(args.prompt_file).read_text()

    repeats = {1: 0, 2: 0, 4: 0}
    with open(args.out, "a", encoding="utf-8") as output:
        emit_burst(output, args, 0, 1, 0, excluded=True)
        time.sleep(args.cooldown)
        for burst_index, concurrency in enumerate(order, start=1):
            repeats[concurrency] += 1
            emit_burst(
                output,
                args,
                burst_index,
                concurrency,
                repeats[concurrency],
            )
            time.sleep(args.cooldown)


if __name__ == "__main__":
    main()
