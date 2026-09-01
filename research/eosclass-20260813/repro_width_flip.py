#!/usr/bin/env python3
"""Sweep the Q27 B=1 -> B>=2 transition using restored-hit peers only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import time
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


def parse_delays(value: str) -> list[int]:
    delays = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not delays or any(delay < 0 for delay in delays):
        raise argparse.ArgumentTypeError("delays must be non-negative comma-separated integers")
    return delays


def full_hit(row: dict[str, Any], prompt_tokens: int) -> bool:
    return (
        row.get("http_status") == 200
        and row.get("done") is True
        and row.get("cached_tokens") == prompt_tokens
        and row.get("finish_reason") in ("stop", "length")
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18427")
    parser.add_argument("--model", default="q27-eosclass")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--peer-count", type=int, choices=range(1, 8), default=3)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--delays-ms",
        type=parse_delays,
        default=parse_delays(",".join(str(value) for value in range(0, 601, 25))),
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--expect",
        choices=("observe", "early-eos", "full-length", "stable-full-length"),
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
    prompt_tokens = len(prompt)
    target_namespace = f"{args.namespace}-hot-87"
    peer_namespaces = [
        f"{args.namespace}-hot-peer-{peer}" for peer in range(args.peer_count)
    ]
    emit(
        {
            "kind": "protocol",
            "schema": "memra.eosclass.width-flip-repro.v1",
            "model": args.model,
            "namespace": args.namespace,
            "peer_count": args.peer_count,
            "repetitions": args.repetitions,
            "delays_ms": args.delays_ms,
            "prompt_tokens": prompt_tokens,
            "prompt_ids_sha256_canonical_json": hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest(),
            "temperature": int(workload["temperature"]),
            "seed": int(workload["seed"]),
            "shape": (
                "serial seed, solo restored-hit control, then target restored hit starts "
                "before already-restored peers"
            ),
        }
    )

    before = replay.scrape(endpoint, args.timeout)
    seed_workload = dict(workload)
    seed_workload["completion_tokens"] = 1
    seed_rows: list[dict[str, Any]] = []
    for index, cache_namespace in enumerate([target_namespace, *peer_namespaces]):
        trace_id = "seed-target" if index == 0 else f"seed-peer-{index - 1}"
        row = replay.request(
            endpoint,
            prompt,
            cache_namespace,
            seed_workload,
            args.timeout,
            trace_id=trace_id,
        )
        row.update(
            {
                "kind": "seed",
                "role": "target" if index == 0 else "peer",
                "peer_index": None if index == 0 else index - 1,
                "trace_id": trace_id,
            }
        )
        seed_rows.append(row)
        emit(public(row))
    after_seed = replay.wait_settled(
        endpoint,
        metric(before, "completed"),
        len(seed_rows),
        args.timeout,
    )

    solo = replay.request(
        endpoint,
        prompt,
        target_namespace,
        workload,
        args.timeout,
        trace_id="solo-control",
    )
    solo.update({"kind": "solo_control", "role": "target", "trace_id": "solo-control"})
    emit(public(solo))
    settled = replay.wait_settled(
        endpoint,
        metric(after_seed, "completed"),
        1,
        args.timeout,
    )

    rows: list[dict[str, Any]] = []
    expected_completed = metric(settled, "completed")
    for repetition in range(1, args.repetitions + 1):
        for delay_ms in args.delays_ms:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.peer_count + 1
            ) as pool:
                target_future = pool.submit(
                    replay.request,
                    endpoint,
                    prompt,
                    target_namespace,
                    workload,
                    args.timeout,
                    trace_id=f"r{repetition}-d{delay_ms}-target",
                )
                time.sleep(delay_ms / 1000.0)
                peer_futures = [
                    pool.submit(
                        replay.request,
                        endpoint,
                        prompt,
                        cache_namespace,
                        workload,
                        args.timeout,
                        trace_id=f"r{repetition}-d{delay_ms}-peer-{peer}",
                    )
                    for peer, cache_namespace in enumerate(peer_namespaces)
                ]
                cell_rows = [target_future.result()] + [
                    future.result() for future in peer_futures
                ]

            for index, row in enumerate(cell_rows):
                row.update(
                    {
                        "kind": "width_flip_request",
                        "role": "target" if index == 0 else "peer",
                        "peer_index": None if index == 0 else index - 1,
                        "delay_ms": delay_ms,
                        "repetition": repetition,
                        "trace_id": (
                            f"r{repetition}-d{delay_ms}-target"
                            if index == 0
                            else f"r{repetition}-d{delay_ms}-peer-{index - 1}"
                        ),
                    }
                )
                rows.append(row)
                emit(public(row))
            expected_completed += len(cell_rows)
            settled = replay.wait_settled(
                endpoint,
                expected_completed - len(cell_rows),
                len(cell_rows),
                args.timeout,
            )

    target_rows = [row for row in rows if row["role"] == "target"]
    early_targets = [
        row
        for row in target_rows
        if full_hit(row, prompt_tokens)
        and int(row.get("completion_tokens") or 0) < int(workload["completion_tokens"])
        and row.get("finish_reason") == "stop"
    ]
    protocol_failures = []
    if any(row.get("cached_tokens") not in (0, prompt_tokens) for row in seed_rows):
        protocol_failures.append("seed did not resolve to a clean miss or existing full entry")
    if not full_hit(solo, prompt_tokens):
        protocol_failures.append("solo control was not a valid full-prefix hit")
    if int(solo.get("completion_tokens") or 0) != int(workload["completion_tokens"]):
        protocol_failures.append("solo control did not reach the full completion budget")
    if any(not full_hit(row, prompt_tokens) for row in rows):
        protocol_failures.append("one or more transition-cell requests were not full-prefix hits")

    after = replay.wait_settled(
        endpoint,
        expected_completed,
        0,
        args.timeout,
    )
    summary = {
        "kind": "summary",
        "schema": "memra.eosclass.width-flip-repro.v1",
        "peer_count": args.peer_count,
        "repetitions": args.repetitions,
        "delays_ms": args.delays_ms,
        "solo_completion_tokens": solo.get("completion_tokens"),
        "solo_text_sha256": solo.get("text_sha256"),
        "target_text_classes": sorted(
            {str(row.get("text_sha256")) for row in target_rows}
        ),
        "all_targets_match_solo": all(
            row.get("text_sha256") == solo.get("text_sha256") for row in target_rows
        ),
        "target_text_sha256_by_delay": [
            {
                "delay_ms": row["delay_ms"],
                "repetition": row["repetition"],
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "text_sha256": row.get("text_sha256"),
                "request_id": row.get("request_id"),
                "trace_id": row.get("trace_id"),
            }
            for row in target_rows
        ],
        "early_eos_targets": [
            {
                "delay_ms": row["delay_ms"],
                "repetition": row["repetition"],
                "completion_tokens": row.get("completion_tokens"),
                "text_sha256": row.get("text_sha256"),
                "request_id": row.get("request_id"),
                "trace_id": row.get("trace_id"),
            }
            for row in early_targets
        ],
        "counter_deltas": deltas(after, before),
        "protocol_failures": protocol_failures,
    }
    emit(summary)

    if protocol_failures:
        return 2
    if args.expect == "early-eos":
        return 0 if early_targets else 1
    if args.expect == "full-length":
        return 0 if all(
            row.get("completion_tokens") == int(workload["completion_tokens"])
            and row.get("finish_reason") == "length"
            for row in target_rows
        ) else 1
    if args.expect == "stable-full-length":
        return 0 if all(
            row.get("completion_tokens") == int(workload["completion_tokens"])
            and row.get("finish_reason") == "length"
            and row.get("text_sha256") == solo.get("text_sha256")
            for row in target_rows
        ) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
