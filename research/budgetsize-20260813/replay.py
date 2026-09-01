#!/usr/bin/env python3
"""Strict frozen-prefix replay for cx-budgetsize sequential and c=64 cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import statistics
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


COUNTERS = (
    "admitted",
    "completed",
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_skips_budget",
    "prefix_cache_skips_pinned",
    "prefix_cache_hit_tokens",
    "prefix_cache_entries",
    "prefix_cache_bytes",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected_sha256: str) -> ModuleType:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: expected {expected_sha256}, got {actual}")
    spec = importlib.util.spec_from_file_location("budgetsize_frozen_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def emit(output: TextIO, row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True)
    output.write(line + "\n")
    output.flush()
    print(line, flush=True)


def metric(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def counter_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {key: metric(after, key) - metric(before, key) for key in COUNTERS}


def public_request(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def validate_request(
    row: dict[str, Any], expected_cached: int, expected_hash: str | None
) -> list[str]:
    failures: list[str] = []
    if not row.get("ok"):
        failures.append(f"request failed: {row.get('error')}")
    if row.get("prompt_tokens") != 4860:
        failures.append(f"prompt_tokens={row.get('prompt_tokens')} != 4860")
    if row.get("completion_tokens") != 60:
        failures.append(f"completion_tokens={row.get('completion_tokens')} != 60")
    if row.get("cached_tokens") != expected_cached:
        failures.append(f"cached_tokens={row.get('cached_tokens')} != {expected_cached}")
    if expected_hash is not None and row.get("text_sha256") != expected_hash:
        failures.append("greedy output hash differs from the cold seed")
    return failures


def require_deltas(
    actual: dict[str, int], expected: dict[str, int], failures: list[str]
) -> None:
    for key, value in expected.items():
        if actual[key] != value:
            failures.append(f"{key} delta={actual[key]} != {value}")


def run_sequential(
    frozen: ModuleType,
    endpoint: Any,
    workload: dict[str, Any],
    arm: str,
    repetition: int,
    namespace: str,
    timeout: float,
    output: TextIO,
) -> tuple[list[str], dict[str, Any]]:
    prompt = frozen.scored_prompt_ids(workload)
    before = frozen.scrape(endpoint, timeout)
    failures: list[str] = []
    for key in ("prefix_cache_skips_budget", "prefix_cache_skips_pinned"):
        if key not in before:
            failures.append(f"operator metrics missing {key}")

    rows: list[dict[str, Any]] = []
    golden: str | None = None
    for request_index in range(1, 6):
        row = frozen.request(endpoint, prompt, namespace, workload, timeout)
        after_request = frozen.wait_settled(
            endpoint, metric(before, "completed"), request_index, timeout
        )
        expected_cached = 0 if arm == "baseline" or request_index == 1 else len(prompt)
        row_failures = validate_request(row, expected_cached, golden)
        if golden is None and row.get("ok"):
            golden = str(row["text_sha256"])
        failures.extend(f"request {request_index}: {item}" for item in row_failures)
        emitted = public_request(row)
        emitted.update(
            {
                "kind": "request",
                "arm": arm,
                "repetition": repetition,
                "request_index": request_index,
                "expected_cached_tokens": expected_cached,
                "failures": row_failures,
                "metrics_after_request": {
                    key: metric(after_request, key) for key in COUNTERS
                },
            }
        )
        rows.append(emitted)
        emit(output, emitted)

    after = frozen.wait_settled(endpoint, metric(before, "completed"), 5, timeout)
    deltas = counter_delta(after, before)
    expected = {
        "admitted": 5,
        "completed": 5,
        "prompt_tokens_in": 5 * len(prompt),
        "admission_session_defers": 0,
        "admission_vram_defers": 0,
        "step_oom_parks": 0,
        "prefix_cache_evictions": 0,
        "prefix_cache_skips_pinned": 0,
    }
    if arm == "baseline":
        expected.update(
            {
                "cached_tokens_in": 0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 5,
                "prefix_cache_inserts": 0,
                "prefix_cache_skips_budget": 5,
                "prefix_cache_hit_tokens": 0,
                "prefix_cache_entries": 0,
                "prefix_cache_bytes": 0,
            }
        )
    else:
        expected.update(
            {
                "cached_tokens_in": 4 * len(prompt),
                "prefix_cache_hits": 4,
                "prefix_cache_misses": 1,
                "prefix_cache_inserts": 1,
                "prefix_cache_skips_budget": 0,
                "prefix_cache_hit_tokens": 4 * len(prompt),
                "prefix_cache_entries": 1,
                "prefix_cache_bytes": 301_215_744,
            }
        )
    require_deltas(deltas, expected, failures)
    hit_ttfts = [float(row["ttft_ms"]) for row in rows[1:] if row.get("ttft_ms") is not None]
    summary = {
        "kind": "summary",
        "schema": "memra.cx-budgetsize.replay.v1",
        "mode": "sequential",
        "arm": arm,
        "repetition": repetition,
        "requests": len(rows),
        "counter_deltas": deltas,
        "cold_ttft_ms": rows[0].get("ttft_ms"),
        "first_hit_ttft_ms": rows[1].get("ttft_ms") if arm != "baseline" else None,
        "hit_ttft_median_ms": statistics.median(hit_ttfts) if arm != "baseline" else None,
        "golden_text_sha256": golden,
        "metrics_before": {key: before.get(key) for key in COUNTERS},
        "metrics_after": {key: after.get(key) for key in COUNTERS},
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    return failures, summary


def run_c64(
    frozen: ModuleType,
    endpoint: Any,
    workload: dict[str, Any],
    arm: str,
    repetition: int,
    namespace: str,
    timeout: float,
    output: TextIO,
) -> tuple[list[str], dict[str, Any]]:
    prompt = frozen.scored_prompt_ids(workload)
    before_seed = frozen.scrape(endpoint, timeout)
    seed = frozen.request(endpoint, prompt, namespace, workload, timeout)
    after_seed = frozen.wait_settled(endpoint, metric(before_seed, "completed"), 1, timeout)
    failures = validate_request(seed, 0, None)
    seed_delta = counter_delta(after_seed, before_seed)
    require_deltas(
        seed_delta,
        {
            "completed": 1,
            "prefix_cache_misses": 1,
            "prefix_cache_inserts": 1,
            "prefix_cache_skips_budget": 0,
            "prefix_cache_skips_pinned": 0,
            "prefix_cache_entries": 1,
            "prefix_cache_bytes": 301_215_744,
            "admission_session_defers": 0,
            "admission_vram_defers": 0,
            "step_oom_parks": 0,
        },
        failures,
    )
    golden = str(seed.get("text_sha256") or "")
    emitted_seed = public_request(seed)
    emitted_seed.update(
        {
            "kind": "seed",
            "arm": arm,
            "repetition": repetition,
            "counter_deltas": seed_delta,
            "failures": list(failures),
        }
    )
    emit(output, emitted_seed)

    barrier = threading.Barrier(64)
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        futures = [
            pool.submit(
                frozen.request,
                endpoint,
                prompt,
                namespace,
                workload,
                timeout,
                barrier,
                None,
            )
            for _ in range(64)
        ]
        rows = [future.result() for future in futures]

    after = frozen.wait_settled(endpoint, metric(after_seed, "completed"), 64, timeout)
    for request_index, row in enumerate(rows, 1):
        row_failures = validate_request(row, len(prompt), golden)
        failures.extend(f"request {request_index}: {item}" for item in row_failures)
        emitted = public_request(row)
        emitted.update(
            {
                "kind": "request",
                "arm": arm,
                "repetition": repetition,
                "request_index": request_index,
                "failures": row_failures,
            }
        )
        emit(output, emitted)

    deltas = counter_delta(after, after_seed)
    require_deltas(
        deltas,
        {
            "admitted": 64,
            "completed": 64,
            "prompt_tokens_in": 64 * len(prompt),
            "cached_tokens_in": 64 * len(prompt),
            "prefix_cache_hits": 64,
            "prefix_cache_misses": 0,
            "prefix_cache_inserts": 0,
            "prefix_cache_evictions": 0,
            "prefix_cache_skips_budget": 0,
            "prefix_cache_skips_pinned": 0,
            "prefix_cache_hit_tokens": 64 * len(prompt),
            "prefix_cache_entries": 0,
            "prefix_cache_bytes": 0,
            "admission_session_defers": 0,
            "admission_vram_defers": 0,
            "step_oom_parks": 0,
        },
        failures,
    )
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ttft_ms") is not None]
    summary = {
        "kind": "summary",
        "schema": "memra.cx-budgetsize.replay.v1",
        "mode": "c64",
        "arm": arm,
        "repetition": repetition,
        "requests": len(rows),
        "counter_deltas": deltas,
        "seed_counter_deltas": seed_delta,
        "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
        "ttft_max_ms": max(ttfts) if ttfts else None,
        "golden_text_sha256": golden,
        "metrics_before_burst": {key: after_seed.get(key) for key in COUNTERS},
        "metrics_after_burst": {key: after.get(key) for key in COUNTERS},
        "cuda_driver_free_bytes_after": after.get("cuda_driver_free_bytes"),
        "cuda_pool_cached_bytes_after": after.get("cuda_pool_cached_bytes"),
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    return failures, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "baseline",
            "derived",
            "explicit4096-a",
            "explicit4096-b",
            "derived-c64",
            "explicit4096-a-c64",
        ),
        required=True,
    )
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    repo = args.protocol.parents[2]
    frozen_path = repo / protocol["frozen_replay"]
    workload_path = repo / protocol["frozen_workload"]
    frozen = load_module(frozen_path, protocol["frozen_replay_sha256"])
    if sha256_file(workload_path) != protocol["frozen_workload_sha256"]:
        parser.error("frozen workload hash mismatch")
    workload = frozen.load_workload(workload_path)
    if (
        int(workload["prompt_tokens"]) != protocol["prompt_tokens"]
        or int(workload["completion_tokens"]) != protocol["completion_tokens"]
    ):
        parser.error("frozen workload dimensions differ from protocol")
    endpoint = frozen.parse_endpoint(f"q27,{args.endpoint},q27")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        emit(
            output,
            {
                "kind": "protocol",
                "schema": "memra.cx-budgetsize.replay.v1",
                "arm": args.arm,
                "repetition": args.repetition,
                "endpoint": args.endpoint,
                "namespace": args.namespace,
                "frozen_replay_sha256": sha256_file(frozen_path),
                "frozen_workload_sha256": sha256_file(workload_path),
                "prompt_sha256": frozen.prompt_sha256(frozen.scored_prompt_ids(workload)),
            },
        )
        if args.arm in ("derived-c64", "explicit4096-a-c64"):
            failures, summary = run_c64(
                frozen,
                endpoint,
                workload,
                args.arm,
                args.repetition,
                args.namespace,
                args.timeout,
                output,
            )
        else:
            failures, summary = run_sequential(
                frozen,
                endpoint,
                workload,
                args.arm,
                args.repetition,
                args.namespace,
                args.timeout,
                output,
            )
        emit(output, summary)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
