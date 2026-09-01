#!/usr/bin/env python3
"""Probe HY3 first-token identity and sampled c1/c4 served throughput."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


PROMPTS = [
    "Write a Rust function that parses a decimal u64 without allocating, then explain its overflow check.",
    "Diagnose a production API whose p95 latency doubled after a database index migration.",
    "Explain why expert parallel MoE inference can be compute-bound even when GPUs communicate every layer.",
    "Plan a zero-downtime PostgreSQL column rename for a 50-million-row table with rollback points.",
]


def post(endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1_200) as response:
        return json.load(response)


def completion_text(response: dict) -> str:
    message = response["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning") or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    first_tokens = []
    for prompt in PROMPTS:
        response = post(
            args.endpoint,
            {
                "model": "hy3",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "temperature": 0.0,
            },
        )
        first_tokens.append(
            {
                "prompt": prompt,
                "text": completion_text(response),
                "response": response,
            }
        )

    c1_response = post(
        args.endpoint,
        {
            "model": "hy3",
            "messages": [{"role": "user", "content": PROMPTS[0]}],
            "max_tokens": 128,
        },
    )
    c1_usage = c1_response["usage"]
    c1_tokens = int(c1_usage["completion_tokens"])
    c1_elapsed = float(c1_usage["elapsed_s"])

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                post,
                args.endpoint,
                {
                    "model": "hy3",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 128,
                },
            )
            for prompt in PROMPTS
        ]
        c4_responses = [future.result() for future in futures]
    c4_wall = time.monotonic() - started
    c4_tokens = sum(int(response["usage"]["completion_tokens"]) for response in c4_responses)

    report = {
        "format": "memra-hy3-served-ab-v1",
        "label": args.label,
        "endpoint": args.endpoint,
        "first_tokens": first_tokens,
        "sampled_c1": {
            "sampling_fields": [],
            "completion_tokens": c1_tokens,
            "elapsed_s": c1_elapsed,
            "tok_s": c1_tokens / c1_elapsed,
            "response": c1_response,
        },
        "sampled_c4": {
            "sampling_fields": [],
            "completion_tokens": c4_tokens,
            "client_wall_s": c4_wall,
            "aggregate_tok_s": c4_tokens / c4_wall,
            "responses": c4_responses,
        },
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{args.label}: c1={report['sampled_c1']['tok_s']:.2f} tok/s "
        f"c4={report['sampled_c4']['aggregate_tok_s']:.2f} aggregate tok/s"
    )


if __name__ == "__main__":
    main()
