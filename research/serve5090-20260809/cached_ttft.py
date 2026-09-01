#!/usr/bin/env python3
"""Measure one long-prompt cached TTFT through /v1/completions.

Repeat mode sends the exact setup prompt again. Continuation mode appends the setup response
byte-for-byte under the same cache_salt; that is the serving contract used by the K-policy
calibration, so the measured request must resume at least 1024 prompt tokens and select
automatic K=2 when speculation is enabled.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import time
import urllib.request


def stream_request(base, model, prompt, salt, max_tokens, timeout):
    body = {
        "model": model,
        "prompt": prompt,
        "cache_salt": salt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_visible = None
    usage = {}
    pieces = []
    response_id = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
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
                text = choice.get("text") or ""
                if text:
                    if first_visible is None:
                        first_visible = time.monotonic()
                    pieces.append(text)
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError("stream completed without visible text")
    return {
        "id": response_id,
        "text": "".join(pieces),
        "usage": usage,
        "ttft_ms": (first_visible - started) * 1000.0,
        "wall_ms": (ended - started) * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--cache-salt", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect-spec", choices=("on", "off"), required=True)
    parser.add_argument("--mode", choices=("repeat", "continuation"), required=True)
    parser.add_argument("--setup-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompt = pathlib.Path(args.prompt_file).read_text()
    salt = args.cache_salt
    setup = stream_request(
        args.base, args.model, prompt, salt, args.setup_tokens, args.timeout
    )
    if args.mode == "repeat":
        measured_prompt = prompt
    else:
        measured_prompt = (
            prompt
            + setup["text"]
            + "\n\nContinue from the prior answer. State the most important remaining "
              "regression test in one paragraph.\n"
        )
    measured = stream_request(
        args.base, args.model, measured_prompt, salt, args.max_tokens, args.timeout
    )
    usage = measured["usage"]
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    spec = usage.get("spec")
    if args.mode == "continuation" and cached < 1024:
        raise RuntimeError(f"cached continuation resumed only {cached} tokens")
    if args.mode == "repeat" and args.expect_spec == "off" and cached < 1024:
        raise RuntimeError(f"exact repeat reused only {cached} prefix tokens")
    if args.expect_spec == "on" and spec is None:
        raise RuntimeError("default-policy cached continuation carried no usage.spec")
    if args.expect_spec == "off" and spec is not None:
        raise RuntimeError("MEMRA_SERVE_SPEC=0 continuation unexpectedly carried usage.spec")

    row = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "shape": f"cached-{args.mode}-4k",
        "cache_salt": salt,
        "expect_spec": args.expect_spec,
        "setup_prompt_tokens": setup["usage"].get("prompt_tokens"),
        "setup_completion_tokens": setup["usage"].get("completion_tokens"),
        "setup_wall_ms": setup["wall_ms"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "completion_tokens": usage.get("completion_tokens"),
        "client_ttft_ms": measured["ttft_ms"],
        "latency_ms": measured["wall_ms"],
        "spec": spec,
        "text_sha256": hashlib.sha256(measured["text"].encode()).hexdigest(),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
