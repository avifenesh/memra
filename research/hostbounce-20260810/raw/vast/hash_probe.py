#!/usr/bin/env python3
"""Sequential c=1 exact-output probe for the Step3.7 serve path."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
import urllib.request
from pathlib import Path


FILLER = (
    "The operator measures latency, allocator state, checkpoint durability, and exact output "
    "while a lower-priority optimizer yields to interactive traffic. "
)
PROMPT = (
    "In exactly four concise bullets, explain why a GPU background job must yield to an "
    "interactive inference request. Include one point about memory. Context: " + FILLER * 5
)


def request(
    base: str, model: str, prompt: str, max_tokens: int, timeout: float, index: int
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 3407,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    pieces: list[str] = []
    usage: dict = {}
    request_id = None
    finish_reason = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                request_id = event.get("id") or request_id
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                    if piece:
                        first = first or time.monotonic()
                        pieces.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        text = "".join(pieces)
        encoded = text.encode()
        return {
            "index": index,
            "ok": True,
            "request_id": request_id,
            "ttft_s": None if first is None else first - started,
            "latency_s": time.monotonic() - started,
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": text,
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "latency_s": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else PROMPT
    rows = [
        request(args.base, args.model, prompt, args.max_tokens, args.timeout, index)
        for index in range(args.requests)
    ]
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    good = [row for row in rows if row["ok"]]
    hashes = collections.Counter(row["text_sha256"] for row in good)
    summary = {
        "requests": args.requests,
        "prompt_bytes": len(prompt.encode()),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "n_ok": len(good),
        "n_error": len(rows) - len(good),
        "hash_counts": dict(sorted(hashes.items())),
        "expected_sha256": args.expected_sha256,
        "expected_matches": sum(
            row["text_sha256"] == args.expected_sha256 for row in good
        ) if args.expected_sha256 else None,
        "single_hash": len(good) == args.requests and len(hashes) == 1,
        "errors": [row.get("error") for row in rows if not row["ok"]],
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    if not summary["single_hash"]:
        return 1
    if args.expected_sha256 and summary["expected_matches"] != args.requests:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
