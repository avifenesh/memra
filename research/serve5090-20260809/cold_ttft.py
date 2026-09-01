#!/usr/bin/env python3
"""One warmup plus one cold TTFT row in a bounded cache namespace.

Warmup and measured prompts differ at their first user tokens, so neither the continuation
pool nor the prefix cache can resume them. Keeping one cache_salt avoids retaining one parked
spec session for every cold probe on a VRAM-tight single card.
"""

import argparse
import datetime
import json
import pathlib
import time
import urllib.request


def request(args, prompt, phase):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": args.cache_salt,
    }
    req = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_visible = None
    usage = {}
    response_id = None
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
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
            response_id = event.get("id") or response_id
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                visible = (
                    (delta.get("content") or "")
                    + (delta.get("reasoning") or "")
                    + (delta.get("reasoning_content") or "")
                )
                if visible and first_visible is None:
                    first_visible = time.monotonic()
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError(f"{phase}: stream completed without visible text")
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if cached != 0:
        raise RuntimeError(f"{phase}: expected a cold prompt, got {cached} cached tokens")
    return {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "shape": args.shape,
        "phase": phase,
        "id": response_id,
        "cache_salt": args.cache_salt,
        "client_ttft_ms": (first_visible - started) * 1000.0,
        "latency_ms": (ended - started) * 1000.0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": usage.get("completion_tokens"),
        "spec": usage.get("spec"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shape", choices=("short", "4k"), required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompt = pathlib.Path(args.prompt_file).read_text()
    rows = [
        request(args, "Trial A. " + prompt, "warmup"),
        request(args, "Trial B. " + prompt, "measured"),
    ]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
    measured = rows[-1]
    print(json.dumps({
        "summary": {
            "label": args.label,
            "shape": args.shape,
            "n": 1,
            "prompt_tokens": measured["prompt_tokens"],
            "client_ttft_p50_ms": measured["client_ttft_ms"],
        }
    }, sort_keys=True))


if __name__ == "__main__":
    main()
