#!/usr/bin/env python3
"""Measure one prefix snapshot's exact device-byte accounting at fixed token lengths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected_sha256: str) -> ModuleType:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: expected {expected_sha256}, got {actual}")
    spec = importlib.util.spec_from_file_location("cachesize_frozen_replay", path)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", choices=("q27", "q35"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frozen = load_module(args.frozen_replay, protocol["frozen_replay_sha256"])
    if sha256_file(args.workload_lock) != protocol["frozen_workload_sha256"]:
        parser.error("frozen workload hash mismatch")
    workload = frozen.load_workload(args.workload_lock)
    endpoint = frozen.parse_endpoint(f"{args.model},{args.endpoint},{args.model}")
    lengths = [int(value) for value in protocol["entry_probe_prefix_tokens"]]
    request_workload = dict(workload)
    request_workload["completion_tokens"] = 1

    failures: list[str] = []
    measurements: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        emit(output, {
            "kind": "protocol",
            "schema": "memra.cachesize.entry-bytes.v1",
            "model": args.model,
            "endpoint": args.endpoint,
            "prefix_tokens": lengths,
            "server_ctx": int(protocol["entry_probe_ctx"]),
            "measurement": (
                "prefix_cache_bytes delta after one cold full-prompt seed; this is the "
                "runtime's exact device-resident KV plus recurrent-state accounting"
            ),
            "frozen_replay_sha256": sha256_file(args.frozen_replay),
            "frozen_workload_sha256": sha256_file(args.workload_lock),
        })

        for token_count in lengths:
            prompt = frozen.fixed_prompt_ids(token_count, 105, 1_008)
            before = frozen.scrape(endpoint, args.timeout)
            row = frozen.request(
                endpoint,
                prompt,
                f"{args.namespace}-{args.model}-t{token_count}",
                request_workload,
                args.timeout,
            )
            after = frozen.wait_settled(
                endpoint,
                metric(before, "completed"),
                1,
                args.timeout,
            )
            public = {key: value for key, value in row.items() if not key.startswith("_")}
            deltas = {
                key: metric(after, key) - metric(before, key)
                for key in (
                    "admitted",
                    "completed",
                    "prompt_tokens_in",
                    "cached_tokens_in",
                    "prefix_cache_hits",
                    "prefix_cache_misses",
                    "prefix_cache_inserts",
                    "prefix_cache_evictions",
                    "prefix_cache_entries",
                    "prefix_cache_bytes",
                    "admission_session_defers",
                    "admission_vram_defers",
                    "step_oom_parks",
                )
            }
            row_failures: list[str] = []
            if not row.get("ok"):
                row_failures.append(f"request failed: {row.get('error')}")
            if row.get("prompt_tokens") != token_count:
                row_failures.append(
                    f"response prompt_tokens={row.get('prompt_tokens')} != {token_count}"
                )
            if row.get("cached_tokens") != 0:
                row_failures.append(f"cold seed cached_tokens={row.get('cached_tokens')}")
            expected = {
                "admitted": 1,
                "completed": 1,
                "prompt_tokens_in": token_count,
                "cached_tokens_in": 0,
                "prefix_cache_hits": 0,
                "prefix_cache_misses": 1,
                "prefix_cache_inserts": 1,
                "prefix_cache_evictions": 0,
                "prefix_cache_entries": 1,
                "admission_session_defers": 0,
                "admission_vram_defers": 0,
                "step_oom_parks": 0,
            }
            for key, value in expected.items():
                if deltas[key] != value:
                    row_failures.append(f"{key} delta={deltas[key]} != {value}")
            if deltas["prefix_cache_bytes"] <= 0:
                row_failures.append(
                    f"prefix_cache_bytes delta={deltas['prefix_cache_bytes']} is not positive"
                )
            measurement = {
                "kind": "entry_bytes",
                "schema": "memra.cachesize.entry-bytes.v1",
                "model": args.model,
                "prefix_tokens": token_count,
                "device_bytes": deltas["prefix_cache_bytes"],
                "device_mib": deltas["prefix_cache_bytes"] / (1024 * 1024),
                "bytes_per_token": deltas["prefix_cache_bytes"] / token_count,
                "metrics_before": {
                    "prefix_cache_entries": metric(before, "prefix_cache_entries"),
                    "prefix_cache_bytes": metric(before, "prefix_cache_bytes"),
                },
                "metrics_after": {
                    "prefix_cache_entries": metric(after, "prefix_cache_entries"),
                    "prefix_cache_bytes": metric(after, "prefix_cache_bytes"),
                },
                "counter_deltas": deltas,
                "request": public,
                "failures": row_failures,
                "clean": not row_failures,
            }
            measurements.append(measurement)
            failures.extend(f"t{token_count}: {failure}" for failure in row_failures)
            emit(output, measurement)

        emit(output, {
            "kind": "summary",
            "schema": "memra.cachesize.entry-bytes.v1",
            "model": args.model,
            "measurements": len(measurements),
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
