#!/usr/bin/env python3
"""Serial Q27 seed followed by one restored hit mixed with cold peers."""

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
    parser.add_argument("--cold-peers", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--expect",
        choices=("observe", "early-eos", "full-length"),
        default="observe",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")

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
    prompt_hash = hashlib.sha256(
        json.dumps(prompt, separators=(",", ":")).encode()
    ).hexdigest()
    emit(
        {
            "kind": "protocol",
            "schema": "memra.eosclass.restore-mix-repro.v1",
            "model": args.model,
            "namespace": args.namespace,
            "cold_peers": args.cold_peers,
            "repetitions": args.repetitions,
            "prompt_tokens": len(prompt),
            "prompt_ids_sha256_canonical_json": prompt_hash,
            "temperature": int(workload["temperature"]),
            "seed": int(workload["seed"]),
            "shape": "one serial full-length seed, then one restored hit plus cold peers",
        }
    )

    before = replay.scrape(endpoint, args.timeout)
    # Keep the historical failing key label so the reducer can join snapshot/restore receipts.
    target_namespace = f"{args.namespace}-hot-87"
    baseline = replay.request(
        endpoint,
        prompt,
        target_namespace,
        workload,
        args.timeout,
    )
    baseline.update({"kind": "seed_baseline", "role": "target"})
    emit(public(baseline))
    after_seed = replay.wait_settled(
        endpoint,
        metric(before, "completed"),
        1,
        args.timeout,
    )

    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        jobs = [("hit", target_namespace)] + [
            ("cold", f"{args.namespace}-cold-r{repetition}-p{peer}")
            for peer in range(args.cold_peers)
        ]
        barrier = threading.Barrier(len(jobs))

        def one(index: int, role: str, cache_namespace: str) -> dict[str, Any]:
            row = replay.request(
                endpoint,
                prompt,
                cache_namespace,
                workload,
                args.timeout,
                barrier=barrier,
            )
            row.update(
                {
                    "kind": "mixed_request",
                    "repetition": repetition,
                    "job_index": index,
                    "role": role,
                }
            )
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [
                pool.submit(one, index, role, cache_namespace)
                for index, (role, cache_namespace) in enumerate(jobs)
            ]
            repetition_rows = [future.result() for future in futures]
        rows.extend(repetition_rows)
        for row in repetition_rows:
            emit(public(row))

    after = replay.wait_settled(
        endpoint,
        metric(after_seed, "completed"),
        len(rows),
        args.timeout,
    )
    hit_rows = [row for row in rows if row["role"] == "hit"]
    cold_rows = [row for row in rows if row["role"] == "cold"]
    early_hits = [
        row
        for row in hit_rows
        if int(row.get("cached_tokens") or 0) == len(prompt)
        and int(row.get("completion_tokens") or 0) < int(workload["completion_tokens"])
        and row.get("finish_reason") == "stop"
    ]
    full_hits = [
        row
        for row in hit_rows
        if int(row.get("cached_tokens") or 0) == len(prompt)
        and int(row.get("completion_tokens") or 0) == int(workload["completion_tokens"])
        and row.get("finish_reason") == "length"
    ]
    protocol_failures = []
    if not baseline.get("ok") or baseline.get("cached_tokens") != 0:
        protocol_failures.append("serial target seed was not a clean miss")
    if any(int(row.get("cached_tokens") or 0) != len(prompt) for row in hit_rows):
        protocol_failures.append("target request was not a full cache hit")
    if any(int(row.get("cached_tokens") or 0) != 0 for row in cold_rows):
        protocol_failures.append("cold peer received cache credit")

    summary = {
        "kind": "summary",
        "schema": "memra.eosclass.restore-mix-repro.v1",
        "cold_peers": args.cold_peers,
        "repetitions": args.repetitions,
        "baseline_text_sha256": baseline.get("text_sha256"),
        "hit_text_sha256": [row.get("text_sha256") for row in hit_rows],
        "cold_text_sha256": [row.get("text_sha256") for row in cold_rows],
        "full_length_hits": len(full_hits),
        "early_eos_hits": [
            {
                "repetition": row["repetition"],
                "request_id": row.get("request_id"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "text_sha256": row.get("text_sha256"),
            }
            for row in early_hits
        ],
        "counter_deltas": deltas(after, before),
        "protocol_failures": protocol_failures,
    }
    emit(summary)

    if protocol_failures:
        return 2
    if args.expect == "early-eos":
        return 0 if early_hits else 1
    if args.expect == "full-length":
        return 0 if len(full_hits) == len(hit_rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
