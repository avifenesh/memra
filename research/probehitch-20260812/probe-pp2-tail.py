#!/usr/bin/env python3
"""Measure one cache-warm streaming request around the PP owner-thread receipt gate."""

import argparse
import datetime
import hashlib
import json
import pathlib
import time
import urllib.request


def stream_request(args: argparse.Namespace, prompt: str, phase: str) -> dict:
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": args.cache_salt,
    }
    request = urllib.request.Request(
        args.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.monotonic()
    first_visible = None
    request_id = None
    usage = {}
    finish_reason = None
    text = []
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
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
                    if first_visible is None:
                        first_visible = time.monotonic()
                    text.append(visible)
                finish_reason = choice.get("finish_reason") or finish_reason

    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError(f"{args.label} {phase}: no visible SSE delta")
    if not request_id:
        raise RuntimeError(f"{args.label} {phase}: no response id")
    details = usage.get("prompt_tokens_details") or {}
    rendered = "".join(text).encode()
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "phase": phase,
        "id": request_id,
        "cache_salt": args.cache_salt,
        "client_ttft_ms": (first_visible - started) * 1_000.0,
        "latency_ms": (ended - started) * 1_000.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "text_sha256": hashlib.sha256(rendered).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="step35")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--idle-settle", type=float, default=1.0)
    parser.add_argument("--post-settle", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompt = pathlib.Path(args.prompt_file).read_text()
    gate = pathlib.Path(args.gate)
    if gate.exists():
        raise RuntimeError(f"receipt gate already exists: {gate}")

    warmup = stream_request(args, prompt, "warmup")
    print(json.dumps(warmup, sort_keys=True), flush=True)
    time.sleep(args.idle_settle)
    gate.touch(exist_ok=False)
    measured = stream_request(args, prompt, "measured")
    print(json.dumps(measured, sort_keys=True), flush=True)
    if measured["cached_tokens"] <= 0:
        raise RuntimeError(
            f"{args.label}: measured request was not a cache hit: {measured}"
        )
    pathlib.Path(args.out).write_text(
        json.dumps(warmup, sort_keys=True) + "\n"
        + json.dumps(measured, sort_keys=True) + "\n"
    )
    time.sleep(args.post_settle)


if __name__ == "__main__":
    main()
