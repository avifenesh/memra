#!/usr/bin/env python3
"""Send one frozen sellgate request to a locally profiled memra server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request


PROMPT_TOKENS = 4_860
COMPLETION_TOKENS = 60
MAX_CTX = PROMPT_TOKENS + COMPLETION_TOKENS + 8
PROMPT_SHA256 = "eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb"


def frozen_prompt_ids() -> list[int]:
    offset = 105
    family_seed = 1_008
    prompt = [
        5_000 + ((position + offset + family_seed * 131) % 1_024)
        for position in range(PROMPT_TOKENS)
    ]
    encoded = json.dumps(prompt, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != PROMPT_SHA256:
        raise RuntimeError(f"frozen prompt hash mismatch: {actual}")
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--max-tokens", type=int, default=COMPLETION_TOKENS)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    prompt = frozen_prompt_ids()
    body = {
        "model": args.model,
        "prompt_ids": prompt,
        "max_ctx": MAX_CTX,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 3_407,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": args.cache_salt,
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.monotonic()
    first_visible = None
    finish_reason = None
    usage = {}
    pieces: list[str] = []
    done = False
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text and first_visible is None:
                first_visible = time.monotonic()
            pieces.append(text)
            finish_reason = choices[0].get("finish_reason") or finish_reason
    ended = time.monotonic()

    print(
        json.dumps(
            {
                "schema": "memra.fa3softmax.profile-request.v1",
                "model": args.model,
                "prompt_tokens": PROMPT_TOKENS,
                "prompt_ids_sha256_canonical_json": PROMPT_SHA256,
                "max_ctx": MAX_CTX,
                "max_tokens": args.max_tokens,
                "cache_salt": args.cache_salt,
                "http_status": status,
                "done": done,
                "finish_reason": finish_reason,
                "ttft_ms": (
                    (first_visible - started) * 1_000 if first_visible is not None else None
                ),
                "elapsed_ms": (ended - started) * 1_000,
                "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
                "usage": usage,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
