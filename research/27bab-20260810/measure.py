#!/usr/bin/env python3
"""Streaming measurement client for the 27B-beside-Step Vast campaign.

Every invocation writes immutable request rows plus one summary.  Token counts and
speculative-acceptance counters come from memra's response/metrics surfaces; text is
retained as a bounded prefix and a full SHA-256 so BOS-only corruption cannot become a
performance point.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request


FILLER = (
    "The operator measures latency, allocator state, checkpoint durability, and exact output "
    "while a lower-priority optimizer yields to interactive traffic. "
)

PROMPTS = {
    "sanity": "Say OK.",
    "workload": (
        "In exactly four concise bullets, explain why a GPU background job must yield to an "
        "interactive inference request. Include one point about memory. Context: " + FILLER * 5
    ),
    # This is deliberately tokenizer-measured on the server rather than described as exactly
    # 4,096 tokens.  On the Step/Qwen tokenizer family it is the requested ~4k prime class.
    "long4k": (
        "Summarize the following operating record in one sentence. Context: " + FILLER * 157
    ),
}

SPECIAL_MARKERS = (
    "<|begin_of_sentence|>",
    "<|end_of_sentence|>",
    "<|im_start|>",
    "<|im_end|>",
)


def fetch_json(url: str, timeout: float = 10.0) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except Exception as error:  # retained in the receipt; request success remains separate
        return {"fetch_error": f"{type(error).__name__}: {error}"}


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


def spec_counter(metrics: dict, model: str) -> dict:
    value = (metrics.get("spec") or {}).get(model) or {}
    return {
        "rounds": int(value.get("rounds", 0)),
        "drafted": int(value.get("drafted", 0)),
        "accepted": int(value.get("accepted", 0)),
    }


def fetch_settled_metrics(
    url: str, before: dict, completed_requests: int, timeout: float = 10.0
) -> tuple[dict, bool, int | None]:
    """Wait until request-finalization counters include this invocation.

    The streaming response can close a few milliseconds before memra publishes its
    completed/spec counters.  Sampling immediately would attribute this invocation's
    counters to the following cell.
    """
    before_completed = before.get("completed")
    if not isinstance(before_completed, int):
        return fetch_json(url), False, None
    expected_completed = before_completed + completed_requests
    deadline = time.monotonic() + timeout
    after: dict = {}
    while True:
        after = fetch_json(url)
        observed = after.get("completed")
        if isinstance(observed, int) and observed >= expected_completed:
            return after, True, expected_completed
        if time.monotonic() >= deadline:
            return after, False, expected_completed
        time.sleep(0.01)


def request_once(
    *,
    base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    cache_salt: str,
    session_id: str,
    timeout: float,
    index: int,
    zero: float,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": cache_salt,
        "session_id": session_id,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    pieces: list[str] = []
    usage: dict = {}
    request_id = None
    finish_reason = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                request_id = event.get("id") or request_id
                if event.get("usage"):
                    usage = event["usage"]
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                    if piece:
                        if first is None:
                            first = time.monotonic()
                        pieces.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        ended = time.monotonic()
        text = "".join(pieces)
        encoded = text.encode()
        completion_tokens = int(usage.get("completion_tokens", 0))
        marker_only = bool(text.strip()) and all(
            part.strip() in SPECIAL_MARKERS for part in text.split() if part.strip()
        )
        bos_garbage = (
            not text.strip()
            or marker_only
            or sum(text.count(marker) for marker in SPECIAL_MARKERS) >= 4
        )
        ttft = None if first is None else first - started
        decode_s = None if ttft is None else (ended - started) - ttft
        decode_tok_s = None
        if decode_s is not None and decode_s > 0 and completion_tokens > 1:
            decode_tok_s = (completion_tokens - 1) / decode_s
        row = {
            "index": index,
            "ok": status == 200 and not bos_garbage and completion_tokens > 0,
            "http_status": status,
            "request_id": request_id,
            "started_offset_s": started - zero,
            "ended_offset_s": ended - zero,
            "started_monotonic_s": started,
            "ended_monotonic_s": ended,
            "ttft_s": ttft,
            "latency_s": ended - started,
            "decode_s": decode_s,
            "decode_tok_s": decode_tok_s,
            "finish_reason": finish_reason,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": completion_tokens,
            "usage_spec": usage.get("spec"),
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
            "text_prefix": text[:512],
            "bos_garbage": bos_garbage,
            "cache_salt": cache_salt,
            "session_id": session_id,
        }
        if not row["ok"]:
            row["error"] = "empty/special-token-only output or missing token usage"
        return row
    except Exception as error:
        ended = time.monotonic()
        return {
            "index": index,
            "ok": False,
            "started_offset_s": started - zero,
            "ended_offset_s": ended - zero,
            "started_monotonic_s": started,
            "ended_monotonic_s": ended,
            "latency_s": ended - started,
            "session_id": session_id,
            "error": f"{type(error).__name__}: {error}"[:800],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--session-key")
    parser.add_argument("--rows", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < 1 or args.max_tokens < 1:
        parser.error("concurrency, requests, and max-tokens must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("duration must be positive")
    if args.rows.exists() or args.summary.exists():
        parser.error("refusing to overwrite an existing receipt")

    prompt = (
        args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file is not None
        else PROMPTS[args.prompt]
    )
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    before = fetch_json(args.base.rstrip("/") + "/metrics")
    nonce = f"{time.time_ns()}"
    zero = time.monotonic()
    deadline = zero + args.duration if args.duration is not None else None
    next_index = 0
    index_lock = threading.Lock()
    rows: list[dict] = []
    rows_lock = threading.Lock()
    release = threading.Barrier(args.concurrency + 1)

    def worker(worker_index: int) -> None:
        nonlocal next_index
        release.wait()
        while True:
            with index_lock:
                if deadline is None:
                    if next_index >= args.requests:
                        return
                elif time.monotonic() >= deadline:
                    return
                index = next_index
                next_index += 1
            unique_key = f"cx27-{args.label}-{nonce}-{worker_index}-{index}"
            # memra's speculative pool affinity follows cache_salt, not the OpenAI
            # extension's session_id. A session key deliberately stabilizes both.
            cache_salt = (
                f"{args.session_key}-worker-{worker_index}"
                if args.session_key is not None
                else unique_key
            )
            session_id = cache_salt
            row = request_once(
                base=args.base,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.seed + index,
                cache_salt=cache_salt,
                session_id=session_id,
                timeout=args.timeout,
                index=index,
                zero=zero,
            )
            row["worker"] = worker_index
            with rows_lock:
                rows.append(row)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.concurrency)]
    for thread in threads:
        thread.start()
    release.wait()
    for thread in threads:
        thread.join()

    rows.sort(key=lambda row: row["index"])
    with args.rows.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    good = [row for row in rows if row.get("ok")]
    after, metrics_settled, expected_completed = fetch_settled_metrics(
        args.base.rstrip("/") + "/metrics", before, len(good)
    )
    starts = [float(row["started_offset_s"]) for row in good]
    ends = [float(row["ended_offset_s"]) for row in good]
    wall = max(ends) - min(starts) if starts and ends else None
    completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in good)
    ttfts = [float(row["ttft_s"]) for row in good if row.get("ttft_s") is not None]
    decodes = [
        float(row["decode_tok_s"])
        for row in good
        if row.get("decode_tok_s") is not None
    ]
    hashes = collections.Counter(row["text_sha256"] for row in good)
    before_spec = spec_counter(before, args.model)
    after_spec = spec_counter(after, args.model)
    spec_delta = {
        key: after_spec[key] - before_spec[key] for key in ("rounds", "drafted", "accepted")
    }
    spec_delta["acceptance_rate"] = (
        spec_delta["accepted"] / spec_delta["drafted"] if spec_delta["drafted"] > 0 else None
    )
    prompt_token_values = sorted(
        {int(row["prompt_tokens"]) for row in good if row.get("prompt_tokens") is not None}
    )
    expected_matches = (
        sum(row["text_sha256"] == args.expected_sha256 for row in good)
        if args.expected_sha256
        else None
    )
    summary = {
        "label": args.label,
        "base": args.base,
        "model": args.model,
        "prompt_kind": args.prompt,
        "prompt_bytes": len(prompt.encode()),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens_observed": prompt_token_values,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "requested_requests": None if args.duration is not None else args.requests,
        "duration_target_s": args.duration,
        "max_tokens": args.max_tokens,
        "n": len(rows),
        "n_ok": len(good),
        "n_error": len(rows) - len(good),
        "completion_tokens": completion_tokens,
        "wall_s": wall,
        "aggregate_output_tok_s": completion_tokens / wall if wall and wall > 0 else None,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p99_s": nearest_rank(ttfts, 99),
        "ttft_values_s": sorted(ttfts),
        "decode_tok_s_median": statistics.median(decodes) if decodes else None,
        "decode_tok_s_values": sorted(decodes),
        "text_hash_counts": dict(sorted(hashes.items())),
        "text_prefixes": sorted({row["text_prefix"] for row in good})[:8],
        "expected_sha256": args.expected_sha256,
        "session_key": args.session_key,
        "expected_matches": expected_matches,
        "bos_garbage_count": sum(bool(row.get("bos_garbage")) for row in rows),
        "spec_metrics_delta": spec_delta,
        "metrics_settled": metrics_settled,
        "metrics_expected_completed": expected_completed,
        "metrics_before": before,
        "metrics_after": after,
        "errors": [row.get("error") for row in rows if not row.get("ok")],
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)

    clean = len(good) == len(rows) and len(rows) >= args.concurrency and metrics_settled
    if args.expected_sha256:
        clean = clean and expected_matches == len(rows)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
