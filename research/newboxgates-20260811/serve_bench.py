#!/usr/bin/env python3
"""Streaming serving probe for the new-box N=5 receipt."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request


SHORT_PROMPT = "Say OK."
DECODE_PROMPT = (
    "Write a detailed technical essay of at least 1,200 words about reliable GPU inference "
    "serving. Cover scheduling, memory, observability, and failure recovery."
)


def one_request(
    args: argparse.Namespace,
    prompt: str,
    request_index: int,
    ready: threading.Barrier,
    go: threading.Event,
) -> dict:
    cache_salt = f"{args.label}-q{request_index}-{time.time_ns()}"
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 3407,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    ready.wait()
    go.wait()
    started = time.monotonic()
    row = {
        "kind": "request",
        "label": args.label,
        "shape": args.shape,
        "concurrency": args.concurrency,
        "request_index": request_index,
        "cache_salt": cache_salt,
        "ok": False,
    }
    pieces: list[str] = []
    usage: dict = {}
    first_visible = None
    finish_reason = None
    request_id = None
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            row["http_status"] = response.status
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    row["done"] = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                request_id = event.get("id") or request_id
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (
                        (delta.get("content") or "")
                        + (delta.get("reasoning") or "")
                        + (delta.get("reasoning_content") or "")
                    )
                    if piece:
                        first_visible = first_visible or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode(errors="replace")[:500]
    except Exception as error:  # receipt records the concrete client failure
        row["error"] = f"{type(error).__name__}: {error}"[:500]

    ended = time.monotonic()
    encoded = "".join(pieces).encode()
    row.update(
        {
            "request_id": request_id,
            "ttft_s": first_visible - started if first_visible is not None else None,
            "latency_s": ended - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish_reason,
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
            "first_visible_at": first_visible,
            "ended_at": ended,
        }
    )
    row["ok"] = bool(
        row.get("http_status") == 200
        and row.get("done")
        and first_visible is not None
        and request_id
        and not row.get("error")
    )
    if args.expect_prompt_tokens is not None:
        row["ok"] = bool(
            row["ok"]
            and row["prompt_tokens"] == args.expect_prompt_tokens
            and row["cached_tokens"] == 0
        )
    if args.require_length:
        row["ok"] = bool(
            row["ok"]
            and row["completion_tokens"] == args.max_tokens
            and row["finish_reason"] == "length"
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shape", choices=("warmup", "short", "4k", "decode"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--expect-prompt-tokens", type=int)
    parser.add_argument("--require-length", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    if args.shape == "4k":
        if args.prompt_file is None or not args.prompt_file.is_file():
            parser.error("4k shape requires --prompt-file")
        prompt = args.prompt_file.read_text()
    elif args.shape == "decode":
        prompt = DECODE_PROMPT
    else:
        prompt = SHORT_PROMPT

    ready = threading.Barrier(args.concurrency + 1)
    go = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args, prompt, index, ready, go)
            for index in range(args.concurrency)
        ]
        ready.wait()
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        started = time.monotonic()
        go.set()
        rows = [future.result() for future in futures]
    ended = time.monotonic()

    oks = [row for row in rows if row["ok"]]
    ttfts = [float(row["ttft_s"]) for row in oks]
    completion_tokens = sum(int(row["completion_tokens"] or 0) for row in oks)
    decode_tokens = sum(max(0, int(row["completion_tokens"] or 0) - 1) for row in oks)
    first_visible = [float(row["first_visible_at"]) for row in oks]
    last_end = [float(row["ended_at"]) for row in oks]
    wall_s = ended - started
    decode_window_s = (
        max(last_end) - min(first_visible) if first_visible and last_end else None
    )
    summary = {
        "kind": "summary",
        "label": args.label,
        "shape": args.shape,
        "started_utc": started_utc,
        "concurrency": args.concurrency,
        "n_requests": len(rows),
        "n_ok": len(oks),
        "n_error": len(rows) - len(oks),
        "max_tokens": args.max_tokens,
        "wall_s": wall_s,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_min_s": min(ttfts) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "completion_tokens_total": completion_tokens,
        "total_window_tok_s": completion_tokens / wall_s if wall_s else None,
        "decode_window_s": decode_window_s,
        "decode_window_tok_s": (
            decode_tokens / decode_window_s if decode_window_s else None
        ),
        "prompt_tokens": sorted({row.get("prompt_tokens") for row in oks}),
        "cached_tokens": sorted({row.get("cached_tokens") for row in oks}),
        "finish_reasons": sorted({row.get("finish_reason") for row in oks}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as output:
        for row in rows:
            row.pop("first_visible_at", None)
            row.pop("ended_at", None)
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if len(oks) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
