#!/usr/bin/env python3
"""Qualify a fixed long-prompt identity before the scored sellgate replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path
from typing import Any

from sellgate_replay import Endpoint, load_workload, metric_delta, request, scrape


COUNTERS = (
    "admitted",
    "completed",
    "tokens_out",
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hit_tokens",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "step_oom_parks",
)


def fixed_prompt(token_count: int, offset: int) -> list[int]:
    family_seed = 1_008
    return [
        5_000 + ((position + offset + family_seed * 131) % 1_024)
        for position in range(token_count)
    ]


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    endpoints = []
    for raw in args.endpoint:
        parts = raw.split(",", 2)
        if len(parts) != 3 or not all(parts):
            parser.error("--endpoint must be LABEL,BASE_URL,MODEL")
        endpoints.append(Endpoint(parts[0], parts[1].rstrip("/"), parts[2]))
    if len(endpoints) != 2:
        parser.error("the sold shape requires exactly two endpoints")
    levels = [int(value) for value in args.concurrency.split(",")]
    if args.repetitions < 1 or levels != [1, 2, 4, 8]:
        parser.error("require repetitions >= 1 and concurrency 1,2,4,8")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    workload = load_workload(args.workload_lock)
    prompt = fixed_prompt(int(workload["prompt_tokens"]), args.offset)
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        protocol = {
            "kind": "protocol",
            "schema": "memra.sellgate.prompt-pilot.v1",
            "prompt_family": "safe-c8-v1-fixed-offset",
            "prompt_family_seed": 1_008,
            "prompt_offset": args.offset,
            "prompt_tokens": len(prompt),
            "completion_tokens": int(workload["completion_tokens"]),
            "concurrency": levels,
            "repetitions": args.repetitions,
            "selection_rule": (
                "PASS only if both models return exactly the requested completion length "
                "with visible content for every cold request at every required width"
            ),
        }
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        print(json.dumps(protocol, sort_keys=True), flush=True)

        for rep in range(1, args.repetitions + 1):
            order = levels[rep - 1 :] + levels[: rep - 1]
            if rep % 2 == 0:
                order = list(reversed(order))
            for concurrency in order:
                before = {endpoint.label: scrape(endpoint, args.timeout) for endpoint in endpoints}
                barrier = threading.Barrier(concurrency * len(endpoints))

                def one(endpoint: Endpoint, index: int) -> dict[str, Any]:
                    salt = (
                        f"{args.namespace}-{endpoint.label}-r{rep}-c{concurrency}-i{index}"
                    )
                    row = request(
                        endpoint,
                        prompt,
                        salt,
                        workload,
                        args.timeout,
                        barrier=barrier,
                    )
                    row.update(
                        {
                            "kind": "request",
                            "target": endpoint.label,
                            "rep": rep,
                            "concurrency": concurrency,
                            "index": index,
                            "prompt_offset": args.offset,
                        }
                    )
                    row["cold_usage_ok"] = bool(
                        row.get("prompt_tokens") == len(prompt)
                        and row.get("cached_tokens") == 0
                        and row.get("completion_tokens")
                        == int(workload["completion_tokens"])
                        and row.get("finish_reason") == "length"
                    )
                    row["ok"] = bool(row.get("ok") and row["cold_usage_ok"])
                    return row

                futures = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency * len(endpoints)
                ) as pool:
                    for endpoint in endpoints:
                        for index in range(concurrency):
                            futures.append(pool.submit(one, endpoint, index))
                    wave_rows = [future.result() for future in futures]
                rows.extend(wave_rows)
                after = {endpoint.label: scrape(endpoint, args.timeout) for endpoint in endpoints}
                for row in wave_rows:
                    output.write(json.dumps(public(row), sort_keys=True) + "\n")

                for endpoint in endpoints:
                    selected = [row for row in wave_rows if row["target"] == endpoint.label]
                    deltas = {
                        key: metric_delta(after[endpoint.label], before[endpoint.label], key)
                        for key in COUNTERS
                    }
                    prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in selected)
                    completion_total = sum(
                        int(row.get("completion_tokens") or 0) for row in selected
                    )
                    failures = []
                    if any(not row.get("ok") for row in selected):
                        failures.append("one or more requests failed MaxNew/usage checks")
                    if deltas["admitted"] != concurrency or deltas["completed"] != concurrency:
                        failures.append("admitted/completed counter drift")
                    if deltas["prompt_tokens_in"] != prompt_total:
                        failures.append("prompt token counter drift")
                    if deltas["tokens_out"] != completion_total:
                        failures.append("completion token counter drift")
                    if any(
                        deltas[key] != 0
                        for key in (
                            "cached_tokens_in",
                            "prefix_cache_hit_tokens",
                            "prefix_cache_hits",
                            "step_oom_parks",
                        )
                    ):
                        failures.append("cold/cache/OOM counters invalid")
                    if deltas["prefix_cache_misses"] != concurrency:
                        failures.append("prefix-cache miss counter drift")
                    cell = {
                        "kind": "cell",
                        "schema": "memra.sellgate.prompt-pilot.v1",
                        "target": endpoint.label,
                        "rep": rep,
                        "concurrency": concurrency,
                        "requests_n": len(selected),
                        "requests_ok": sum(bool(row.get("ok")) for row in selected),
                        "completion_tokens": completion_total,
                        "counter_deltas": deltas,
                        "failures": failures,
                        "clean": not failures,
                    }
                    cells.append(cell)
                    output.write(json.dumps(cell, sort_keys=True) + "\n")
                    print(json.dumps(cell, sort_keys=True), flush=True)
                output.flush()

        target_clean = {
            endpoint.label: all(
                cell["clean"] for cell in cells if cell["target"] == endpoint.label
            )
            for endpoint in endpoints
        }
        summary = {
            "kind": "summary",
            "schema": "memra.sellgate.prompt-pilot.v1",
            "prompt_offset": args.offset,
            "cells_n": len(cells),
            "requests_n": len(rows),
            "target_clean": target_clean,
            "verdict": "PASS" if all(target_clean.values()) else "FAIL",
        }
        output.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if all(target_clean.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
