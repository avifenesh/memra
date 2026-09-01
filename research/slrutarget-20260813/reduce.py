#!/usr/bin/env python3
"""Reduce tee-first SLRU simulation JSONL into a compact decision artifact."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row["metrics"][field]) for row in rows)


def aggregate(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "hit_rate_mean": mean(rows, "hit_rate"),
        "hit_rate_min": min(float(row["metrics"]["hit_rate"]) for row in rows),
        "hit_rate_max": max(float(row["metrics"]["hit_rate"]) for row in rows),
        "returning_hit_rate_mean": mean(rows, "returning_hit_rate"),
        "reuse_hit_rate_mean": mean(rows, "reuse_hit_rate"),
        "post_first_hit_misses_mean": mean(rows, "post_first_hit_misses"),
        "direct_scan_caused_post_first_hit_misses_mean": mean(
            rows, "direct_scan_caused_post_first_hit_misses"
        ),
        "proven_returning_evictions_mean": mean(rows, "proven_returning_evictions"),
        "evictions_mean": mean(rows, "evictions"),
        "scan_hits_sum": sum(int(row["metrics"]["scan_hits"]) for row in rows),
        "refusals_sum": sum(int(row["metrics"]["refusals"]) for row in rows),
    }


def paired_summary(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    metadata: dict[tuple, dict] = {}
    for row in rows:
        values: list[object] = []
        for field in group_fields:
            if field in row:
                values.append(row[field])
            else:
                values.append(row["parameters"][field])
        key = tuple(values)
        grouped[key][row["policy"]].append(row)
        metadata[key] = {field: value for field, value in zip(group_fields, values)}
    out: list[dict] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        arms = grouped[key]
        if set(arms) != {"lru", "slru"}:
            raise ValueError(f"unpaired policies for {metadata[key]}: {sorted(arms)}")
        lru = aggregate(arms["lru"])
        slru = aggregate(arms["slru"])
        out.append(
            {
                **metadata[key],
                "lru": lru,
                "slru": slru,
                "delta": {
                    "hit_rate_pp": 100 * (slru["hit_rate_mean"] - lru["hit_rate_mean"]),
                    "returning_hit_rate_pp": 100 * (
                        slru["returning_hit_rate_mean"] - lru["returning_hit_rate_mean"]
                    ),
                    "reuse_hit_rate_pp": 100 * (
                        slru["reuse_hit_rate_mean"] - lru["reuse_hit_rate_mean"]
                    ),
                    "post_first_hit_misses_mean": (
                        slru["post_first_hit_misses_mean"]
                        - lru["post_first_hit_misses_mean"]
                    ),
                    "proven_returning_evictions_mean": (
                        slru["proven_returning_evictions_mean"]
                        - lru["proven_returning_evictions_mean"]
                    ),
                },
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines()]
    validation = next(row for row in rows if row.get("kind") == "validation")
    if validation["verdict"] != "PASS":
        raise SystemExit("simulation validation failed")
    simulations = [row for row in rows if row.get("kind") == "simulation"]
    primary_rows = [row for row in simulations if row["scenario"] == "primary_hot_scan"]
    primary = paired_summary(primary_rows, ("variant", "budget_mib"))
    sensitivity_rows = [
        row for row in simulations if row["scenario"] == "sensitivity_hot_scan"
    ]
    sensitivity = paired_summary(
        sensitivity_rows,
        ("variant", "budget_mib", "logical_sessions", "returning_fraction", "zipf_alpha"),
    )
    stationary_rows = [row for row in simulations if row["scenario"] == "stationary_cycle"]
    stationary = paired_summary(
        stationary_rows, ("variant", "budget_mib", "logical_sessions", "cycles")
    )
    turnover_rows = [row for row in simulations if row["scenario"] == "hotset_turnover_cycle"]
    turnover = paired_summary(
        turnover_rows,
        ("variant", "budget_mib", "old_protected_logical_sessions",
         "new_logical_sessions", "cycles"),
    )
    tolerance = 1e-12
    worse_sensitivity = [
        row for row in sensitivity if row["delta"]["hit_rate_pp"] < -tolerance
    ]
    better_sensitivity = [
        row for row in sensitivity if row["delta"]["hit_rate_pp"] > tolerance
    ]
    result = {
        "schema": "memra.slrutarget.analysis.v1",
        "validation": validation,
        "capacities": [row for row in rows if row.get("kind") == "capacity"],
        "primary": primary,
        "sensitivity": {
            "scenarios": sensitivity,
            "scenario_count": len(sensitivity),
            "slru_better_count": len(better_sensitivity),
            "equal_count": len(sensitivity) - len(better_sensitivity) - len(worse_sensitivity),
            "slru_worse_count": len(worse_sensitivity),
            "best_hit_rate_delta": max(
                better_sensitivity, key=lambda row: row["delta"]["hit_rate_pp"], default=None
            ),
            "worst_hit_rate_delta": min(
                sensitivity, key=lambda row: row["delta"]["hit_rate_pp"], default=None
            ),
        },
        "controls": {
            "stationary_cycle": stationary,
            "hotset_turnover_cycle": turnover,
            "turnover_worst": min(
                turnover, key=lambda row: row["delta"]["hit_rate_pp"], default=None
            ),
        },
        "decision_checks": {
            "primary_nonnegative": all(
                row["delta"]["hit_rate_pp"] >= -tolerance
                and row["delta"]["returning_hit_rate_pp"] >= -tolerance
                for row in primary
            ),
            "primary_slru_post_first_hit_misses_zero": all(
                row["slru"]["post_first_hit_misses_mean"] == 0 for row in primary
            ),
            "primary_scan_hits_zero": all(row["slru"]["scan_hits_sum"] == 0 for row in primary),
            "primary_refusals_zero": all(row["slru"]["refusals_sum"] == 0 for row in primary),
            "losing_shape_found": any(
                row["delta"]["hit_rate_pp"] < -tolerance for row in turnover
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
