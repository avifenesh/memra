#!/usr/bin/env python3
"""Measure synthetic owner-thread stalls for the old and new peer-probe policies.

The injected durations are the complete N=1 cloudbox rung costs captured in
research/sec9-20260812/raw/cloudbox/driver.log. This deliberately measures scheduler blocking with
wall-clock sleeps; it is not a replacement for a two-GPU transport benchmark.
"""

import argparse
import json
import math
import statistics
import time


RUNGS = [
    {"tokens": 1, "source_ms": 0.724},
    {"tokens": 8, "source_ms": 1.161},
    {"tokens": 16, "source_ms": 1.909},
    {"tokens": 4096, "source_ms": 431.188},
]
BUDGET_MS = 5.0


def measured_sleep_ms(duration_ms: float) -> float:
    started = time.perf_counter_ns()
    time.sleep(duration_ms / 1000.0)
    return (time.perf_counter_ns() - started) / 1e6


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summary(policy: str, samples: list[float], cycles: int) -> dict:
    return {
        "kind": "summary",
        "policy": policy,
        "cycles": cycles,
        "samples": len(samples),
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(percentile(samples, 0.95), 6),
        "max_ms": round(max(samples), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=9)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")

    print(json.dumps({
        "kind": "config",
        "cycles": args.cycles,
        "budget_ms": BUDGET_MS,
        "source": "research/sec9-20260812/raw/cloudbox/driver.log",
        "rungs": RUNGS,
        "measurement": "CPU-only wall-clock sleep injection; no CUDA work",
    }, sort_keys=True))

    before: list[float] = []
    after_busy: list[float] = []
    after_idle: list[float] = []
    for cycle in range(args.cycles):
        for rung_index, rung in enumerate(RUNGS):
            old_ms = measured_sleep_ms(rung["source_ms"])
            before.append(old_ms)
            print(json.dumps({
                "kind": "sample",
                "policy": "before_all_inline",
                "cycle": cycle,
                "rung": rung_index,
                "tokens": rung["tokens"],
                "stall_ms": round(old_ms, 6),
            }, sort_keys=True))

            idle_only = rung_index == len(RUNGS) - 1 or rung["source_ms"] > BUDGET_MS
            started = time.perf_counter_ns()
            if not idle_only:
                time.sleep(rung["source_ms"] / 1000.0)
            busy_ms = (time.perf_counter_ns() - started) / 1e6
            after_busy.append(busy_ms)
            print(json.dumps({
                "kind": "sample",
                "policy": "after_busy_boundary",
                "cycle": cycle,
                "rung": rung_index,
                "tokens": rung["tokens"],
                "idle_only": idle_only,
                "stall_ms": round(busy_ms, 6),
            }, sort_keys=True))

            if idle_only:
                idle_ms = measured_sleep_ms(rung["source_ms"])
                after_idle.append(idle_ms)
                print(json.dumps({
                    "kind": "sample",
                    "policy": "after_idle_drain",
                    "cycle": cycle,
                    "rung": rung_index,
                    "tokens": rung["tokens"],
                    "stall_ms": round(idle_ms, 6),
                }, sort_keys=True))

    print(json.dumps(summary("before_all_inline", before, args.cycles), sort_keys=True))
    print(json.dumps(summary("after_busy_boundary", after_busy, args.cycles), sort_keys=True))
    print(json.dumps(summary("after_idle_drain", after_idle, args.cycles), sort_keys=True))


if __name__ == "__main__":
    main()
