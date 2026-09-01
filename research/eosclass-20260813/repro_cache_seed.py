#!/usr/bin/env python3
"""Reproduce cache-hit output drift inherited from concurrent prefix seeding."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any


COUNTERS = (
    "admitted",
    "completed",
    "tokens_out",
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_hit_tokens",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("eosclass_frozen_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def metric(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {key: metric(after, key) - metric(before, key) for key in COUNTERS}


def emit(row: dict[str, Any]) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18427")
    parser.add_argument("--model", default="q27-eosclass")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--seed-concurrency", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--hit-concurrency", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--hit-repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--expect",
        choices=("observe", "early-eos", "full-length"),
        default="observe",
    )
    args = parser.parse_args()
    if args.seed_count < 1 or args.hit_repetitions < 1:
        parser.error("seed-count and hit-repetitions must be positive")

    replay = load_module(repo / "research/sellgate-20260812/sellgate_replay.py")
    workload = replay.load_workload(repo / "research/sellgate-20260812/workload.lock.json")
    expected = {
        "prompt_tokens": 4860,
        "completion_tokens": 60,
        "temperature": 0,
        "seed": 3407,
    }
    actual = {key: workload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"frozen workload drift: expected {expected}, got {actual}")

    endpoint = replay.Endpoint(
        label="q27",
        base=args.base.rstrip("/"),
        model=args.model,
    )
    prompt = replay.scored_prompt_ids(workload)
    emit(
        {
            "kind": "protocol",
            "schema": "memra.eosclass.cache-seed-repro.v1",
            "model": args.model,
            "namespace": args.namespace,
            "seed_count": args.seed_count,
            "seed_concurrency": args.seed_concurrency,
            "hit_concurrency": args.hit_concurrency,
            "hit_repetitions": args.hit_repetitions,
            "prompt_tokens": len(prompt),
            "prompt_ids_sha256_canonical_json": hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest(),
            "temperature": int(workload["temperature"]),
            "seed": int(workload["seed"]),
        }
    )
    seed_workload = dict(workload)
    seed_workload["completion_tokens"] = 1
    before = replay.scrape(endpoint, args.timeout)

    seed_rows: list[dict[str, Any]] = []
    for start in range(0, args.seed_count, args.seed_concurrency):
        ids = list(range(start, min(start + args.seed_concurrency, args.seed_count)))
        barrier = threading.Barrier(len(ids)) if len(ids) > 1 else None

        def seed_one(prefix_id: int) -> dict[str, Any]:
            row = replay.request(
                endpoint,
                prompt,
                f"{args.namespace}-hot-{prefix_id}",
                seed_workload,
                args.timeout,
                barrier=barrier,
            )
            row.update(
                {
                    "kind": "seed",
                    "prefix_id": prefix_id,
                    "seed_concurrency": len(ids),
                }
            )
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
            rows = [future.result() for future in (pool.submit(seed_one, i) for i in ids)]
        seed_rows.extend(rows)
        for row in rows:
            emit(public(row))

    after_seed = replay.wait_settled(
        endpoint,
        metric(before, "completed"),
        args.seed_count,
        args.timeout,
    )

    hit_rows: list[dict[str, Any]] = []
    for repetition in range(args.hit_repetitions):
        for start in range(0, args.seed_count, args.hit_concurrency):
            ids = list(range(start, min(start + args.hit_concurrency, args.seed_count)))
            barrier = threading.Barrier(len(ids)) if len(ids) > 1 else None

            def hit_one(prefix_id: int) -> dict[str, Any]:
                row = replay.request(
                    endpoint,
                    prompt,
                    f"{args.namespace}-hot-{prefix_id}",
                    workload,
                    args.timeout,
                    barrier=barrier,
                )
                row.update(
                    {
                        "kind": "hit",
                        "prefix_id": prefix_id,
                        "repetition": repetition + 1,
                        "seed_concurrency": args.seed_concurrency,
                        "hit_concurrency": len(ids),
                    }
                )
                return row

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
                rows = [future.result() for future in (pool.submit(hit_one, i) for i in ids)]
            hit_rows.extend(rows)
            for row in rows:
                emit(public(row))

    after_hits = replay.wait_settled(
        endpoint,
        metric(after_seed, "completed"),
        args.seed_count * args.hit_repetitions,
        args.timeout,
    )
    early = [
        {
            "prefix_id": row["prefix_id"],
            "repetition": row["repetition"],
            "request_id": row.get("request_id"),
            "completion_tokens": row.get("completion_tokens"),
            "finish_reason": row.get("finish_reason"),
            "cached_tokens": row.get("cached_tokens"),
            "text_sha256": row.get("text_sha256"),
        }
        for row in hit_rows
        if row.get("completion_tokens") != 60 or row.get("finish_reason") != "length"
    ]
    full_hits = [
        row
        for row in hit_rows
        if row.get("cached_tokens") == len(prompt)
        and row.get("completion_tokens") == 60
        and row.get("finish_reason") == "length"
    ]
    summary = {
        "kind": "summary",
        "schema": "memra.eosclass.cache-seed-repro.v1",
        "seed_count": args.seed_count,
        "seed_concurrency": args.seed_concurrency,
        "hit_concurrency": args.hit_concurrency,
        "hit_repetitions": args.hit_repetitions,
        "prompt_tokens": len(prompt),
        "expected_completion_tokens": 60,
        "seed_rows_ok": sum(bool(row.get("ok")) for row in seed_rows),
        "full_cache_hits": sum(row.get("cached_tokens") == len(prompt) for row in hit_rows),
        "full_length_hits": len(full_hits),
        "early_or_non_length": early,
        "seed_counter_deltas": deltas(after_seed, before),
        "hit_counter_deltas": deltas(after_hits, after_seed),
    }
    emit(summary)

    if args.expect == "early-eos":
        return 0 if any(
            row.get("cached_tokens") == len(prompt)
            and row.get("completion_tokens", 60) < 60
            and row.get("finish_reason") == "stop"
            for row in hit_rows
        ) else 1
    if args.expect == "full-length":
        return 0 if len(full_hits) == len(hit_rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
