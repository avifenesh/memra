#!/usr/bin/env python3
"""Run one grouped-regate boot over the frozen Q35 c=4/c=40 serve cells."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


FROZEN_REPLAY_SHA256 = "91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b"
FROZEN_WORKLOAD_SHA256 = "85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34"
FROZEN_LEVELS = [4, 40]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_replay(path: Path) -> ModuleType:
    actual = sha256_file(path)
    if actual != FROZEN_REPLAY_SHA256:
        raise ValueError(f"{path}: frozen replay hash {actual} != {FROZEN_REPLAY_SHA256}")
    spec = importlib.util.spec_from_file_location("frozen_sellgate_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import frozen replay from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--dispatch-arm", choices=("off", "grouped"), required=True)
    parser.add_argument("--repetition", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    workload_hash = sha256_file(args.workload_lock)
    if workload_hash != FROZEN_WORKLOAD_SHA256:
        parser.error(
            f"frozen workload hash {workload_hash} != {FROZEN_WORKLOAD_SHA256}"
        )

    frozen = load_frozen_replay(args.frozen_replay)
    endpoint = frozen.parse_endpoint(args.endpoint)
    if endpoint.label != "q35":
        parser.error("the grouped-regate serve cell is frozen to endpoint label q35")
    workload = frozen.load_workload(args.workload_lock)
    if int(workload["repetitions"]) != 5:
        parser.error("frozen workload must retain five repetitions")
    if int(workload["prompt_tokens"]) != 4860:
        parser.error("frozen workload must retain 4,860 prompt tokens")
    if int(workload["completion_tokens"]) != 60:
        parser.error("frozen workload must retain 60 completion tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dispatch_arm": args.dispatch_arm,
        "physical_gpu_index": args.physical_gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "source_commit": args.source_commit,
    }
    failures: list[str] = []
    goldens: dict[tuple[str, int], str] = {}
    cells: list[dict[str, Any]] = []
    order = FROZEN_LEVELS if args.repetition % 2 else list(reversed(FROZEN_LEVELS))

    with args.out.open("x", encoding="utf-8") as output:
        def emit(row: dict[str, Any]) -> None:
            merged = {**row, **metadata}
            output.write(json.dumps(merged, sort_keys=True) + "\n")
            output.flush()

        protocol = {
            "kind": "protocol",
            "schema": "memra.groupedregate.serve-ab.v1",
            "target": endpoint.label,
            "endpoint": dataclasses.asdict(endpoint),
            "global_repetition": args.repetition,
            "total_repetitions": 5,
            "levels": FROZEN_LEVELS,
            "width_order": order,
            "workload": workload,
            "workload_lock_sha256": workload_hash,
            "frozen_replay_sha256": sha256_file(args.frozen_replay),
            "prompt_ids_sha256_canonical_json": frozen.prompt_sha256(
                frozen.scored_prompt_ids(workload)
            ),
            "cache_shape": (
                "frozen mixed90: nine full-prompt hits and one cold miss per ten "
                "equal-sized prompts"
            ),
            "latency_clock": "first visible response content, not SSE keepalive",
        }
        emit(protocol)
        print(json.dumps({**protocol, **metadata}, sort_keys=True), flush=True)

        for position, concurrency in enumerate(order):
            pair_index = (args.repetition - 1) * len(order) + position
            workload_arms = (
                ["cold", "mixed90"] if pair_index % 2 == 0 else ["mixed90", "cold"]
            )
            for workload_arm in workload_arms:
                if workload_arm == "mixed90":
                    seed_rows, seed_failures = frozen.seed_hot_set(
                        [endpoint], workload, args.namespace, args.timeout, goldens
                    )
                    for row in seed_rows:
                        emit(public(row))
                    failures.extend(seed_failures)
                    for failure in seed_failures:
                        print(
                            json.dumps(
                                {"kind": "seed_failure", "error": failure, **metadata}
                            ),
                            flush=True,
                        )

                request_rows, sample_rows, cell_rows = frozen.run_cell(
                    [endpoint],
                    workload,
                    args.namespace,
                    workload_arm,
                    args.repetition,
                    concurrency,
                    args.timeout,
                    goldens,
                )
                for row in [*sample_rows, *request_rows, *cell_rows]:
                    emit(row)
                cells.extend(cell_rows)
                for row in cell_rows:
                    print(json.dumps({**row, **metadata}, sort_keys=True), flush=True)
                    if not row["clean"]:
                        failures.append(
                            f"q35 {workload_arm} r{args.repetition} c{concurrency}: "
                            + "; ".join(row["integrity_failures"])
                        )

        summary = {
            "kind": "summary",
            "schema": "memra.groupedregate.serve-ab.v1",
            "target": endpoint.label,
            "global_repetition": args.repetition,
            "levels_run": FROZEN_LEVELS,
            "cells": len(cells),
            "cells_clean": len(cells) == 4 and all(row["clean"] for row in cells),
            "failures": failures,
        }
        summary["verdict"] = (
            "PASS" if summary["cells_clean"] and not failures else "FAIL"
        )
        emit(summary)
        print(json.dumps({**summary, **metadata}, sort_keys=True), flush=True)

    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
