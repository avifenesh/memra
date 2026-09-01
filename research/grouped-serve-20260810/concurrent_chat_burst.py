#!/usr/bin/env python3
"""Cold concurrent chat-prefill burst probe with visible streaming output."""

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


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def one_request(args, prompt, repeat, request_index, ready, go):
    cache_salt = (
        f"{args.label}-r{repeat}-q{request_index}-{time.time_ns()}"
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
    }
    request = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    ready.wait()
    go.wait()
    started = time.monotonic()
    first_visible = None
    request_id = None
    usage = {}
    finish_reason = None
    visible_chunks = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
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
                visible = (
                    (delta.get("content") or "")
                    + (delta.get("reasoning") or "")
                    + (delta.get("reasoning_content") or "")
                )
                if visible:
                    visible_chunks += 1
                    first_visible = first_visible or time.monotonic()
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.monotonic()

    if first_visible is None:
        raise RuntimeError(
            f"repeat {repeat} request {request_index}: no visible SSE delta"
        )
    if not request_id:
        raise RuntimeError(f"repeat {repeat} request {request_index}: no response id")
    prompt_tokens = usage.get("prompt_tokens")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get(
        "cached_tokens"
    )
    if prompt_tokens != args.expect_prompt_tokens or cached_tokens != 0:
        raise RuntimeError(
            f"repeat {repeat} request {request_index}: unexpected usage "
            f"prompt={prompt_tokens} cached={cached_tokens}"
        )

    return {
        "kind": "request",
        "label": args.label,
        "repeat": repeat,
        "request": request_index,
        "id": request_id,
        "cache_salt": cache_salt,
        "client_ttft_s": first_visible - started,
        "client_wall_s": ended - started,
        "first_visible_at": first_visible,
        "visible_chunks": visible_chunks,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
    }


def burst(args, prompt, concurrency, repeat):
    ready = threading.Barrier(concurrency + 1)
    go = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as pool:
        futures = [
            pool.submit(
                one_request,
                args,
                prompt,
                repeat,
                request_index,
                ready,
                go,
            )
            for request_index in range(concurrency)
        ]
        ready.wait()
        started_utc = utc_now()
        started = time.monotonic()
        go.set()
        rows = [future.result() for future in futures]

    last_first_visible = max(row["first_visible_at"] for row in rows)
    wall = last_first_visible - started
    ttfts = [row["client_ttft_s"] for row in rows]
    summary = {
        "kind": "summary",
        "label": args.label,
        "concurrency": concurrency,
        "repeat": repeat,
        "n_requests": len(rows),
        "prompt_tokens_each": args.expect_prompt_tokens,
        "aggregate_prompt_tokens": concurrency * args.expect_prompt_tokens,
        "burst_start_utc": started_utc,
        "last_first_visible_utc": utc_now(),
        "wall_to_last_first_token_s": wall,
        "aggregate_prefill_tps": (
            concurrency * args.expect_prompt_tokens / wall
        ),
        "ttft_p50_s": statistics.median(ttfts),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_min_s": min(ttfts),
        "ttft_max_s": max(ttfts),
    }
    for row in rows:
        row.pop("first_visible_at")
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--expect-prompt-tokens", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("concurrency/repeats must be positive and warmup nonnegative")

    prompt = pathlib.Path(args.prompt_file).read_text()
    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for warmup in range(args.warmup):
            rows, summary = burst(args, prompt, 1, -(warmup + 1))
            summary["kind"] = "warmup"
            output.write(json.dumps(summary, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(summary, sort_keys=True), flush=True)

        for repeat in range(1, args.repeats + 1):
            rows, summary = burst(
                args, prompt, args.concurrency, repeat
            )
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.write(json.dumps(summary, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
