#!/usr/bin/env python3
"""Single-model frozen mixed90 concurrency sweep for cx-kneeraise."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


METRIC_KEYS = (
    "admitted",
    "completed",
    "tokens_out",
    "step_p50_ms",
    "step_p99_ms",
    "prompt_tokens_in",
    "cached_tokens_in",
    "computed_tokens_in",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_hit_tokens",
    "prefix_cache_entries",
    "prefix_cache_bytes",
    "active_sessions",
    "queued_requests",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
    "cuda_driver_free_bytes",
    "cuda_pool_reserved_bytes",
    "cuda_pool_used_bytes",
    "cuda_pool_cached_bytes",
)


def load_replay(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kneeraise_frozen_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen replay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fetch_json(base: str, route: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(base.rstrip("/") + route, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError(f"{route}: expected a JSON object")
    return value


def telemetry(base: str, timeout: float) -> dict[str, Any]:
    metrics = fetch_json(base, "/metrics", timeout)
    health = fetch_json(base, "/health", timeout)
    yield_metrics = fetch_json(base, "/yield/metrics", timeout)
    return {
        "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
        "spec_tau": metrics.get("spec_tau"),
        "spec_accept_by_position": metrics.get("spec_accept_by_position"),
        "worker": health.get("worker"),
        "yield_metrics": yield_metrics,
    }


def emit(output: TextIO, row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True)
    output.write(line + "\n")
    output.flush()
    if row.get("kind") in {"cell", "aggregate", "summary"}:
        print(line, flush=True)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="q27")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workload-lock", required=True, type=Path)
    parser.add_argument("--frozen-replay", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--namespace", default="cx-kneeraise")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--concurrency", default="8,12,16,20,24")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    levels = [int(value) for value in args.concurrency.split(",")]
    if args.reps < 1 or args.rep_start < 1 or not levels or min(levels) < 1:
        parser.error("reps, rep-start, and concurrency levels must be positive")
    if len(set(levels)) != len(levels):
        parser.error("concurrency levels must be unique")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    replay = load_replay(args.frozen_replay)
    workload = replay.load_workload(args.workload_lock)
    endpoint = replay.Endpoint("q27", args.base.rstrip("/"), args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_requests: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    goldens: dict[tuple[str, int], str] = {}
    failures: list[str] = []

    with args.out.open("x", encoding="utf-8") as output:
        emit(output, {
            "kind": "protocol",
            "schema": "memra.kneeraise.sweep.v1",
            "label": args.label,
            "levels": levels,
            "repetitions": args.reps,
            "rep_start": args.rep_start,
            "arm": "mixed90",
            "target": "q27",
            "model": args.model,
            "workload_lock_sha256": replay.sha256_file(args.workload_lock),
            "frozen_replay_sha256": replay.sha256_file(args.frozen_replay),
            "prompt_ids_sha256_canonical_json": replay.prompt_sha256(
                replay.scored_prompt_ids(workload)
            ),
            "workload": workload,
            "level_order": "rotated per repetition, reversed on alternating repetitions",
            "thermal_regime": "one server boot; no clock changes or artificial cooldown",
        })

        # Preserve the frozen global repetition ordering when an interleaved A/B
        # driver runs one repetition per server boot.  Calling width_orders with
        # args.reps alone would restart every N=1 arm at the first width.
        orders = replay.width_orders(
            levels, args.rep_start + args.reps - 1
        )[args.rep_start - 1:]
        for local_rep, order in enumerate(orders):
            rep = args.rep_start + local_rep
            for concurrency in order:
                emit(output, {
                    "kind": "telemetry_boundary",
                    "boundary": "before_seed",
                    "label": args.label,
                    "rep": rep,
                    "concurrency": concurrency,
                    "timestamp_unix_s": time.time(),
                    **telemetry(args.base, min(args.timeout, 10.0)),
                })
                seed_rows, seed_failures = replay.seed_hot_set(
                    [endpoint], workload, args.namespace, args.timeout, goldens
                )
                for row in seed_rows:
                    emit(output, public_row(row) | {
                        "label": args.label,
                        "rep": rep,
                        "concurrency": concurrency,
                    })
                failures.extend(seed_failures)

                requests, samples, cells = replay.run_cell(
                    [endpoint], workload, args.namespace, "mixed90", rep,
                    concurrency, args.timeout, goldens
                )
                for row in samples:
                    emit(output, row | {"label": args.label})
                for row in requests:
                    emit(output, row | {"label": args.label})
                for row in cells:
                    emit(output, row | {"label": args.label})
                    if not row["clean"]:
                        failures.append(
                            f"r{rep} c{concurrency}: "
                            + "; ".join(row["integrity_failures"])
                        )
                all_requests.extend(requests)
                all_cells.extend(cells)
                emit(output, {
                    "kind": "telemetry_boundary",
                    "boundary": "after_cell",
                    "label": args.label,
                    "rep": rep,
                    "concurrency": concurrency,
                    "timestamp_unix_s": time.time(),
                    **telemetry(args.base, min(args.timeout, 10.0)),
                })

        aggregates = [
            row for row in replay.aggregate_rows(
                [endpoint], levels, all_requests, all_cells
            )
            if row["arm"] == "mixed90"
        ]
        aggregates.sort(key=lambda row: int(row["concurrency"]))
        for row in aggregates:
            emit(output, row | {"label": args.label})

        knee = levels[0]
        path: list[dict[str, Any]] = []
        previous = aggregates[0]
        for current in aggregates[1:]:
            rose = bool(
                previous["all_clean"]
                and current["all_clean"]
                and float(current["output_tok_s_median"])
                > float(previous["output_tok_s_median"])
            )
            path.append({
                "from": previous["concurrency"],
                "to": current["concurrency"],
                "from_output_tok_s": previous["output_tok_s_median"],
                "to_output_tok_s": current["output_tok_s_median"],
                "clean_rise": rose,
            })
            if not rose:
                break
            knee = int(current["concurrency"])
            previous = current

        expected_cells = len(levels) * args.reps
        if len(all_cells) != expected_cells:
            failures.append(f"observed {len(all_cells)} cells, expected {expected_cells}")
        if any(len([
            cell for cell in all_cells if int(cell["concurrency"]) == level
        ]) != args.reps for level in levels):
            failures.append("one or more widths are missing repetitions")
        if not all(bool(cell["clean"]) for cell in all_cells):
            failures.append("one or more cells failed integrity checks")

        emit(output, {
            "kind": "summary",
            "schema": "memra.kneeraise.sweep.v1",
            "label": args.label,
            "levels": levels,
            "repetitions": args.reps,
            "cells": len(all_cells),
            "requests": len(all_requests),
            "clean_throughput_knee": knee,
            "capacity_path": path,
            "median_output_tok_s": {
                str(row["concurrency"]): row["output_tok_s_median"]
                for row in aggregates
            },
            "median_of_cell_rates": {
                str(level): statistics.median(
                    float(cell["output_tok_s"])
                    for cell in all_cells
                    if int(cell["concurrency"]) == level
                )
                for level in levels
            },
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
