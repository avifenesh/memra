#!/usr/bin/env python3
"""Run one full-prefix-hit, streaming decode-throughput point.

One unscored cold request seeds a long-form chat prompt, one unscored cache-hit
wave warms the requested batch width, and one cache-hit wave is scored.  The
prompt explicitly demands far more than 512 tokens so natural EOS cannot make
the two runtime policies perform different amounts of work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LONG_PROMPT = (
    "Write a rigorous technical essay of at least 5,000 words about reliable GPU inference "
    "serving. Continue until every requested section is complete; do not conclude early. "
    "Cover model loading, quantization, KV-cache geometry, prefix caching, request admission, "
    "continuous batching, CUDA graphs, kernel selection, memory pools, PCIe transfer, NUMA, "
    "thermal stability, concurrency knees, latency percentiles, throughput measurement, "
    "failure recovery, observability, correctness gates, deterministic replay, capacity "
    "planning, and incident response. For each section explain mechanisms, failure modes, "
    "measurement traps, and concrete mitigations, using complete paragraphs and examples. "
    "Include a long comparative discussion of cold requests, full-prefix hits, mixed traffic, "
    "single-user latency, and saturated revenue traffic. End only after a detailed checklist "
    "and a multi-paragraph conclusion."
)


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def one_request(
    base: str,
    model: str,
    salt: str,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None = None,
    go: threading.Event | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 3407,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=60)
    if go is not None:
        go.wait(timeout=60)

    started = time.monotonic()
    first_visible: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    request_id = None
    done = False
    http_status = None
    error_text = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
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
        http_status = error.code
        error_text = error.read().decode(errors="replace")[:1000]
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"[:1000]

    ended = time.monotonic()
    encoded = "".join(pieces).encode()
    prompt_tokens = usage.get("prompt_tokens")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    completion_tokens = usage.get("completion_tokens")
    row = {
        "request_id": request_id,
        "http_status": http_status,
        "done": done,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": (
            (first_visible - started) * 1000.0 if first_visible is not None else None
        ),
        "latency_ms": (ended - started) * 1000.0,
        "text_bytes": len(encoded),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "error": error_text,
        "_started": started,
        "_first_visible": first_visible,
        "_ended": ended,
    }
    row["ok"] = bool(
        http_status == 200
        and done
        and first_visible is not None
        and request_id
        and prompt_tokens
        and isinstance(cached_tokens, int)
        and completion_tokens == max_tokens
        and finish_reason == "length"
        and not error_text
    )
    return row


def run_wave(
    base: str,
    model: str,
    salt: str,
    max_tokens: int,
    concurrency: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float, float]:
    barrier = threading.Barrier(concurrency + 1)
    go = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                one_request,
                base,
                model,
                salt,
                max_tokens,
                timeout,
                barrier,
                go,
            )
            for _ in range(concurrency)
        ]
        barrier.wait(timeout=60)
        released = time.monotonic()
        go.set()
        rows = [future.result() for future in futures]
    ended = max(float(row["_ended"]) for row in rows)
    return rows, released, ended


def validate_rows(
    rows: list[dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
    label: str,
) -> None:
    for index, row in enumerate(rows):
        if not row.get("ok"):
            raise RuntimeError(f"{label} request {index} failed: {public(row)}")
        expected = {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": "length",
        }
        drift = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if drift:
            raise RuntimeError(f"{label} request {index} shape drift: {drift}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", choices=("q27", "q35"), required=True)
    parser.add_argument("--policy-arm", choices=("repaired", "eager"), required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--label", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.rep < 0 or args.concurrency < 1 or args.max_tokens < 1:
        parser.error("rep must be non-negative; concurrency/max-tokens must be positive")

    salt = args.namespace
    seed = one_request(
        args.base,
        args.model,
        salt,
        args.warmup_tokens,
        args.timeout,
    )
    prompt_tokens = int(seed.get("prompt_tokens") or 0)
    if (
        not seed.get("ok")
        or prompt_tokens < 64
        or seed.get("cached_tokens") != 0
        or seed.get("completion_tokens") != args.warmup_tokens
    ):
        raise RuntimeError(f"cold cache seed failed: {public(seed)}")

    warmup_rows, _, _ = run_wave(
        args.base,
        args.model,
        salt,
        args.warmup_tokens,
        args.concurrency,
        args.timeout,
    )
    validate_rows(warmup_rows, prompt_tokens, args.warmup_tokens, "warmup")

    rows, released, ended = run_wave(
        args.base,
        args.model,
        salt,
        args.max_tokens,
        args.concurrency,
        args.timeout,
    )
    validate_rows(rows, prompt_tokens, args.max_tokens, "scored")

    first_visible = [float(row["_first_visible"]) for row in rows]
    completion_total = args.concurrency * args.max_tokens
    decode_tokens = args.concurrency * max(0, args.max_tokens - 1)
    wall_s = ended - released
    decode_window_s = ended - min(first_visible)
    summary = {
        "kind": "summary",
        "label": args.label,
        "target": args.target,
        "policy_arm": args.policy_arm,
        "rep": args.rep,
        "concurrency": args.concurrency,
        "n_requests": len(rows),
        "n_ok": sum(bool(row.get("ok")) for row in rows),
        "n_error": sum(not bool(row.get("ok")) for row in rows),
        "max_tokens": args.max_tokens,
        "wall_s": wall_s,
        "ttft_p50_s": statistics.median(
            float(row["ttft_ms"]) / 1000.0 for row in rows
        ),
        "ttft_min_s": min(float(row["ttft_ms"]) / 1000.0 for row in rows),
        "ttft_max_s": max(float(row["ttft_ms"]) / 1000.0 for row in rows),
        "completion_tokens_total": completion_total,
        "total_window_tok_s": completion_total / wall_s,
        "decode_window_s": decode_window_s,
        "decode_window_tok_s": decode_tokens / decode_window_s,
        "prompt_tokens": sorted({row.get("prompt_tokens") for row in rows}),
        "cached_tokens": sorted({row.get("cached_tokens") for row in rows}),
        "finish_reasons": sorted({row.get("finish_reason") for row in rows}),
        "prompt_utf8_sha256": hashlib.sha256(LONG_PROMPT.encode()).hexdigest(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "kind": "protocol",
                    "label": args.label,
                    "shape": (
                        "one cold long-form-chat+16 seed; one unscored full-hit "
                        "width-matched +16 wave; one scored full-hit +512 wave"
                    ),
                    "prompt_utf8_sha256": hashlib.sha256(
                        LONG_PROMPT.encode()
                    ).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        output.write(
            json.dumps({**public(seed), "kind": "seed", "label": args.label},
                       sort_keys=True)
            + "\n"
        )
        for index, row in enumerate(warmup_rows):
            output.write(
                json.dumps(
                    {
                        **public(row),
                        "kind": "warmup_request",
                        "label": args.label,
                        "request_index": index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        for index, row in enumerate(rows):
            output.write(
                json.dumps(
                    {
                        **public(row),
                        "kind": "request",
                        "label": args.label,
                        "request_index": index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
