#!/usr/bin/env python3
"""Exactness and timing gate for prefix-cache reuse through a PP-2 server."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def cached_tokens(usage: dict) -> int:
    return int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def request(
    base: str,
    model: str,
    prompt: list[int],
    salt: str,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None = None,
) -> dict:
    body = {
        "model": model,
        "prompt_ids": prompt,
        "max_ctx": len(prompt) + max_tokens + 8,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 3407,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait()
    started = time.monotonic()
    first_visible = None
    pieces: list[str] = []
    usage: dict = {}
    request_id = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
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
                piece = choice.get("text") or ""
                piece += delta.get("content") or ""
                piece += delta.get("reasoning") or ""
                piece += delta.get("reasoning_content") or ""
                if piece:
                    if first_visible is None:
                        first_visible = time.monotonic()
                    pieces.append(piece)
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.monotonic()
    if first_visible is None:
        raise RuntimeError("stream completed without a visible text token")
    encoded = "".join(pieces).encode()
    return {
        "request_id": request_id,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens(usage),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "ttft_ms": round((first_visible - started) * 1000.0, 6),
        "wall_ms": round((ended - started) * 1000.0, 6),
        "text_bytes": len(encoded),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "text_utf8_b64": base64.b64encode(encoded).decode(),
    }


def burst(
    base: str,
    model: str,
    prompt: list[int],
    salt: str,
    max_tokens: int,
    timeout: float,
    concurrency: int,
) -> list[dict]:
    barrier = threading.Barrier(concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(request, base, model, prompt, salt, max_tokens, timeout, barrier)
            for _ in range(concurrency)
        ]
        return [future.result() for future in futures]


def metrics(base: str, timeout: float) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as response:
        return json.load(response)


def metric_snapshot(row: dict) -> dict:
    dual = row.get("dual_pp") or {}
    return {
        "prefix_cache_hits": int(row.get("prefix_cache_hits") or 0),
        "prefix_cache_misses": int(row.get("prefix_cache_misses") or 0),
        "prefix_cache_inserts": int(row.get("prefix_cache_inserts") or 0),
        "prefix_cache_evictions": int(row.get("prefix_cache_evictions") or 0),
        "prefix_cache_hit_tokens": int(row.get("prefix_cache_hit_tokens") or 0),
        "prefix_cache_entries": int(row.get("prefix_cache_entries") or 0),
        "prefix_cache_bytes": int(row.get("prefix_cache_bytes") or 0),
        "cached_tokens_in": int(row.get("cached_tokens_in") or 0),
        "dual_pp_slot_pairs": int(dual.get("slot_pairs") or 0),
        "dual_pp_slot_collisions": int(dual.get("slot_collisions") or 0),
    }


def delta(after: dict, before: dict, key: str) -> int:
    return int(after[key]) - int(before[key])


def median(rows: list[dict], label: str) -> float:
    return statistics.median(row["ttft_ms"] for row in rows if row["case"] == label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", default="prefixmoney")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--require-dual", action="store_true")
    args = parser.parse_args()
    if args.reps < 1 or args.prefix_tokens < 64 or args.suffix_tokens < 1:
        parser.error("require reps>=1, prefix-tokens>=64, suffix-tokens>=1")
    if args.concurrency < 2:
        parser.error("--concurrency must be at least 2")

    prefix = list(range(2000, 2000 + args.prefix_tokens))
    suffix_a = list(range(100000, 100000 + args.suffix_tokens))
    suffix_b = list(range(100256, 100256 + args.suffix_tokens))
    prompt_a = prefix + suffix_a
    prompt_b = prefix + suffix_b
    before = metric_snapshot(metrics(args.base, args.timeout))
    rows: list[dict] = []
    failures: list[str] = []

    for rep in range(1, args.reps + 1):
        salt = f"{args.namespace}-r{rep}"
        a1 = {"kind": "request", "rep": rep, "case": "repeat-cold", **request(
            args.base, args.model, prompt_a, salt, args.max_tokens, args.timeout)}
        b1 = {"kind": "request", "rep": rep, "case": "shared-cold", **request(
            args.base, args.model, prompt_b, salt, args.max_tokens, args.timeout)}
        b2 = {"kind": "request", "rep": rep, "case": "shared-hit", **request(
            args.base, args.model, prompt_b, salt, args.max_tokens, args.timeout)}
        a2 = {"kind": "request", "rep": rep, "case": "repeat-hit", **request(
            args.base, args.model, prompt_a, salt, args.max_tokens, args.timeout)}
        concurrent = burst(
            args.base,
            args.model,
            prompt_a,
            salt,
            args.max_tokens,
            args.timeout,
            args.concurrency,
        )
        concurrent_rows = [
            {"kind": "request", "rep": rep, "case": "repeat-hit-concurrent", "index": i, **row}
            for i, row in enumerate(concurrent)
        ]
        rows.extend([a1, b1, b2, a2, *concurrent_rows])

        expected_prompt_a = len(prompt_a)
        expected_prompt_b = len(prompt_b)
        checks = [
            (a1["cached_tokens"] == 0, f"rep {rep}: repeat cold credited cache"),
            (b1["cached_tokens"] == 0, f"rep {rep}: shared learning request credited cache"),
            (b2["cached_tokens"] == args.prefix_tokens,
             f"rep {rep}: shared hit cached {b2['cached_tokens']} != {args.prefix_tokens}"),
            (a2["cached_tokens"] == expected_prompt_a,
             f"rep {rep}: full hit cached {a2['cached_tokens']} != {expected_prompt_a}"),
            (a1["prompt_tokens"] == expected_prompt_a,
             f"rep {rep}: repeat prompt token count drift"),
            (b1["prompt_tokens"] == expected_prompt_b,
             f"rep {rep}: shared prompt token count drift"),
            (a1["text_utf8_b64"] == a2["text_utf8_b64"],
             f"rep {rep}: repeated-prompt hit changed output bytes"),
            (b1["text_utf8_b64"] == b2["text_utf8_b64"],
             f"rep {rep}: shared-prefix hit changed output bytes"),
        ]
        for i, row in enumerate(concurrent_rows):
            checks.extend([
                (row["cached_tokens"] == expected_prompt_a,
                 f"rep {rep} concurrent {i}: cached {row['cached_tokens']} != {expected_prompt_a}"),
                (row["text_utf8_b64"] == a1["text_utf8_b64"],
                 f"rep {rep} concurrent {i}: cache-hit output changed bytes"),
            ])
        failures.extend(message for passed, message in checks if not passed)

    after = metric_snapshot(metrics(args.base, args.timeout))
    expected_hits = args.reps * (2 + args.concurrency)
    hit_delta = delta(after, before, "prefix_cache_hits")
    miss_delta = delta(after, before, "prefix_cache_misses")
    pair_delta = delta(after, before, "dual_pp_slot_pairs")
    collision_delta = delta(after, before, "dual_pp_slot_collisions")
    if hit_delta < expected_hits:
        failures.append(f"prefix hit counter delta {hit_delta} < expected {expected_hits}")
    if miss_delta < args.reps * 2:
        failures.append(f"prefix miss counter delta {miss_delta} < expected {args.reps * 2}")
    if collision_delta != 0:
        failures.append(f"dual PP slot collisions increased by {collision_delta}")
    if args.require_dual and pair_delta <= 0:
        failures.append("concurrent cache-hit traffic produced no dual PP slot pairs")

    repeat_cold = median(rows, "repeat-cold")
    repeat_hit = median(rows, "repeat-hit")
    shared_cold = median(rows, "shared-cold")
    shared_hit = median(rows, "shared-hit")
    summary = {
        "kind": "summary",
        "schema": "memra.prefixmoney.exactness.v1",
        "model": args.model,
        "reps": args.reps,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "repeated_prompt": {
            "cold_ttft_ms_median": repeat_cold,
            "hit_ttft_ms_median": repeat_hit,
            "delta_ms": round(repeat_hit - repeat_cold, 6),
            "speedup": round(repeat_cold / repeat_hit, 6),
            "byte_identity": f"{args.reps}/{args.reps}",
        },
        "shared_prefix": {
            "learning_ttft_ms_median": shared_cold,
            "hit_ttft_ms_median": shared_hit,
            "delta_ms": round(shared_hit - shared_cold, 6),
            "speedup": round(shared_cold / shared_hit, 6),
            "byte_identity": f"{args.reps}/{args.reps}",
        },
        "metrics_before": before,
        "metrics_after": after,
        "metrics_delta": {
            "prefix_cache_hits": hit_delta,
            "prefix_cache_misses": miss_delta,
            "dual_pp_slot_pairs": pair_delta,
            "dual_pp_slot_collisions": collision_delta,
        },
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    for row in rows:
        printable = {key: value for key, value in row.items() if key != "text_utf8_b64"}
        print(json.dumps(printable, sort_keys=True), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
