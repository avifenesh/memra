#!/usr/bin/env python3
"""Run one global repetition of the frozen sold-shape replay on one endpoint."""

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
LEVELS_BY_TARGET = {
    "q27": [1, 2, 4, 8, 12, 16, 20],
    "q35": [1, 2, 4, 8, 16, 32, 40, 48],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_replay(path: Path) -> ModuleType:
    if sha256_file(path) != FROZEN_REPLAY_SHA256:
        raise ValueError(f"{path}: frozen replay hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_sellgate_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import frozen replay from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_levels(raw: str) -> list[int]:
    try:
        levels = [int(value) for value in raw.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from error
    if not levels or any(value <= 0 for value in levels) or len(levels) != len(set(levels)):
        raise argparse.ArgumentTypeError("levels must be unique positive integers")
    return levels


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--target", choices=sorted(LEVELS_BY_TARGET), required=True)
    parser.add_argument("--levels", type=parse_levels, required=True)
    parser.add_argument("--repetition", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--record-failures", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if args.levels != LEVELS_BY_TARGET[args.target]:
        parser.error(
            f"{args.target} levels must remain frozen at {LEVELS_BY_TARGET[args.target]}"
        )
    if sha256_file(args.workload_lock) != FROZEN_WORKLOAD_SHA256:
        parser.error("frozen workload hash mismatch")

    frozen = load_frozen_replay(args.frozen_replay)
    endpoint = frozen.parse_endpoint(args.endpoint)
    if endpoint.label != args.target:
        parser.error("endpoint label must match --target")
    workload = frozen.load_workload(args.workload_lock)
    repetitions = int(workload["repetitions"])
    if repetitions != 5:
        parser.error("frozen workload must retain five repetitions")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    requests: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    failures: list[str] = []
    goldens: dict[tuple[str, int], str] = {}
    order = frozen.width_orders(args.levels, repetitions)[args.repetition - 1]

    with args.out.open("x", encoding="utf-8") as output:
        def emit(row: dict[str, Any]) -> None:
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()

        protocol = {
            "kind": "protocol",
            "schema": "memra.requal2.single-replay.v1",
            "target": args.target,
            "global_repetition": args.repetition,
            "total_repetitions": repetitions,
            "levels": args.levels,
            "width_order": order,
            "endpoint": dataclasses.asdict(endpoint),
            "workload": workload,
            "workload_lock_sha256": sha256_file(args.workload_lock),
            "frozen_replay_sha256": sha256_file(args.frozen_replay),
            "prompt_ids_sha256_canonical_json": frozen.prompt_sha256(
                frozen.scored_prompt_ids(workload)
            ),
            "cache_shape": (
                "unchanged frozen mixed90: nine full-prompt hits and one cold miss "
                "per ten equal-sized prompts"
            ),
            "latency_clock": "first visible response content, not SSE keepalive",
        }
        emit(protocol)
        print(json.dumps(protocol, sort_keys=True), flush=True)

        for position, concurrency in enumerate(order):
            pair_index = (args.repetition - 1) * len(args.levels) + position
            arms = ["cold", "mixed90"] if pair_index % 2 == 0 else ["mixed90", "cold"]
            for arm in arms:
                if arm == "mixed90":
                    seed_rows, seed_failures = frozen.seed_hot_set(
                        [endpoint], workload, args.namespace, args.timeout, goldens
                    )
                    for row in seed_rows:
                        emit(public(row))
                    failures.extend(seed_failures)
                    for failure in seed_failures:
                        print(json.dumps({"kind": "seed_failure", "error": failure}), flush=True)

                cell_requests, samples, cell_rows = frozen.run_cell(
                    [endpoint],
                    workload,
                    args.namespace,
                    arm,
                    args.repetition,
                    concurrency,
                    args.timeout,
                    goldens,
                )
                for row in [*samples, *cell_requests, *cell_rows]:
                    emit(row)
                requests.extend(cell_requests)
                cells.extend(cell_rows)
                for row in cell_rows:
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if not row["clean"]:
                        failures.append(
                            f"{args.target} {arm} r{args.repetition} c{concurrency}: "
                            + "; ".join(row["integrity_failures"])
                        )

        for aggregate in frozen.aggregate_rows([endpoint], args.levels, requests, cells):
            emit(aggregate)

        base_cells = [row for row in cells if int(row["concurrency"]) in (1, 2, 4, 8)]
        final = {
            "kind": "summary",
            "schema": "memra.requal2.single-replay.v1",
            "target": args.target,
            "global_repetition": args.repetition,
            "levels_run": args.levels,
            "cells": len(cells),
            "required_base_cells": len(base_cells),
            "required_base_clean": len(base_cells) == 8 and all(row["clean"] for row in base_cells),
            "all_cells_clean": len(cells) == len(args.levels) * 2
            and all(row["clean"] for row in cells),
            "seed_and_cell_failures": failures,
        }
        final["verdict"] = (
            "PASS"
            if final["required_base_clean"] and final["all_cells_clean"] and not failures
            else "FAIL"
        )
        emit(final)
        print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if final["verdict"] == "PASS" or args.record_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
