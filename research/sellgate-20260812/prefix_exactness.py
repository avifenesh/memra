#!/usr/bin/env python3
"""Serial partial/full prefix-cache exactness gate for one sellgate model.

This is the c=1 exactness portion of prefixmoney's prefix_gate.py, using the
sellgate's locked 81:1 prompt size and the established visible-output synthetic
prompt family. Batched serving is scored separately by sellgate_replay.py; a
c=4-vs-c=1 text comparison is not a cache exactness oracle on MoE near ties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sellgate_replay import (
    Endpoint,
    fixed_prompt_ids,
    load_workload,
    metric_delta,
    metric_value,
    request,
    scrape,
)


def credited(row: dict[str, Any]) -> int:
    return int(row.get("cached_tokens") or 0)


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    parts = args.endpoint.split(",", 2)
    if len(parts) != 3 or not all(parts):
        parser.error("--endpoint must be LABEL,BASE_URL,MODEL")
    endpoint = Endpoint(parts[0], parts[1].rstrip("/"), parts[2])
    workload = load_workload(args.workload_lock)
    config = workload["partial_prefix_exactness"]
    prefix_n = int(config["prefix_tokens"])
    suffix_n = int(config["suffix_tokens"])
    reps = int(config["repetitions"])
    if int(config["concurrency"]) != 1:
        raise ValueError("prefix-cache exactness must use the serial decode oracle")
    if prefix_n + suffix_n != int(workload["prompt_tokens"]):
        raise ValueError("partial-prefix exactness prompt does not match workload prompt size")

    prefix = fixed_prompt_ids(prefix_n, 370)
    suffix_a = fixed_prompt_ids(suffix_n, 407)
    suffix_b = fixed_prompt_ids(suffix_n, 444)
    prompt_a = prefix + suffix_a
    prompt_b = prefix + suffix_b
    before = scrape(endpoint, args.timeout)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for rep in range(1, reps + 1):
        salt = f"{args.namespace}-r{rep}"
        a1 = request(endpoint, prompt_a, salt, workload, args.timeout)
        b1 = request(endpoint, prompt_b, salt, workload, args.timeout)
        b2 = request(endpoint, prompt_b, salt, workload, args.timeout)
        a2 = request(endpoint, prompt_a, salt, workload, args.timeout)
        named = (
            ("repeat-cold", a1),
            ("shared-cold", b1),
            ("shared-hit", b2),
            ("repeat-hit", a2),
        )
        for case, row in named:
            row.update({"kind": "request", "target": endpoint.label, "rep": rep, "case": case})
            rows.append(row)
            if not row.get("ok"):
                failures.append(f"rep {rep} {case}: request failed: {row.get('error')}")

        checks = (
            (credited(a1) == 0, "repeat cold credited cache"),
            (credited(b1) == 0, "shared learning request credited cache"),
            (credited(b2) == prefix_n, f"shared hit cached {credited(b2)} != {prefix_n}"),
            (
                credited(a2) == len(prompt_a),
                f"full hit cached {credited(a2)} != {len(prompt_a)}",
            ),
            (a1.get("prompt_tokens") == len(prompt_a), "repeat prompt count drift"),
            (b1.get("prompt_tokens") == len(prompt_b), "shared prompt count drift"),
            (a1.get("text_sha256") == a2.get("text_sha256"), "full cache hit changed output"),
            (b1.get("text_sha256") == b2.get("text_sha256"), "partial cache hit changed output"),
        )
        failures.extend(f"rep {rep}: {message}" for passed, message in checks if not passed)

    after = scrape(endpoint, args.timeout)
    prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    cached_total = sum(int(row.get("cached_tokens") or 0) for row in rows)
    deltas = {
        key: metric_delta(after, before, key)
        for key in (
            "prompt_tokens_in",
            "cached_tokens_in",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "prefix_cache_hit_tokens",
        )
    }
    expected_hits = reps * 2
    expected_misses = reps * 2
    checks = (
        (deltas["prompt_tokens_in"] == prompt_total, "prompt_tokens_in drift"),
        (deltas["cached_tokens_in"] == cached_total, "cached_tokens_in drift"),
        (
            deltas["prefix_cache_hit_tokens"] == cached_total,
            "prefix_cache_hit_tokens drift",
        ),
        (deltas["prefix_cache_hits"] == expected_hits, "prefix hit count drift"),
        (deltas["prefix_cache_misses"] == expected_misses, "prefix miss count drift"),
    )
    failures.extend(message for passed, message in checks if not passed)
    summary = {
        "kind": "summary",
        "schema": "memra.sellgate.prefix-exactness.v1",
        "target": endpoint.label,
        "model": endpoint.model,
        "reps": reps,
        "prompt_tokens": len(prompt_a),
        "prefix_tokens": prefix_n,
        "suffix_tokens": suffix_n,
        "completion_tokens": int(workload["completion_tokens"]),
        "metrics_before": {
            key: metric_value(before, key)
            for key in (
                "prompt_tokens_in",
                "cached_tokens_in",
                "prefix_cache_hits",
                "prefix_cache_misses",
                "prefix_cache_hit_tokens",
            )
        },
        "metrics_after": {
            key: metric_value(after, key)
            for key in (
                "prompt_tokens_in",
                "cached_tokens_in",
                "prefix_cache_hits",
                "prefix_cache_misses",
                "prefix_cache_hit_tokens",
            )
        },
        "metrics_delta": deltas,
        "cached_token_usage_total": cached_total,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(public(row), sort_keys=True) + "\n")
        output.write(json.dumps(summary, sort_keys=True) + "\n")
    for row in rows:
        printable = public(row)
        printable.pop("text_utf8_b64", None)
        print(json.dumps(printable, sort_keys=True), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
