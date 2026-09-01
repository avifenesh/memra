#!/usr/bin/env python3
"""Reduce the frozen Box1 cold-prefill before/after binary A/B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LEVELS = (8, 12, 16, 20, 24)
BASE_SOURCE = "d2fba620031920032b253b700443af5ef1ec7866"
CANDIDATE_SOURCE = "b37d77c6f6403d8b3b87099470fc3b5c2cd62cee"
HARNESS_SOURCE = "ca80b88dbe7cc74e8c3c5d31355e6bc23a500050"
BASE_SERVER_SHA256 = "b5e31c8db47f2d5f04a2ffb8729c921fd4b68cb6f090819b8234eb0996385ef3"
CANDIDATE_SERVER_SHA256 = "f00f1bd5d08fbf0476a540e497b51d749d813873c4b885a67fc5fce120120748"
MODEL_SHA256 = "d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517"
WORKLOAD_SHA256 = "85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34"
PROMPT_SHA256 = "eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb"
CHILD_RE = re.compile(
    r"^r(?P<round>\d+)-o(?P<order>\d+)-(?P<arm>baseline|candidate)$"
)
PRIME_RE = re.compile(
    r"^\[prime-batch\] B=(?P<batch>\d+) tokens=(?P<tokens>\d+) "
    r"carried=(?P<carried>\d+)(?: partial=(?P<partial>\d+))? "
    r"in (?P<milliseconds>[0-9.]+)ms$"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_n}: expected JSON object")
        rows.append(value)
    return rows


def one(rows: list[dict[str, Any]], kind: str, path: Path) -> dict[str, Any]:
    selected = [row for row in rows if row.get("kind") == kind]
    if len(selected) != 1:
        raise ValueError(f"{path}: expected one {kind}, got {len(selected)}")
    return selected[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and " " not in key:
            result[key] = value
    return result


def input_hashes(path: Path) -> list[str]:
    rows = [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 5:
        raise ValueError(f"{path}: expected five frozen input hashes")
    return rows


def rate_path(rates: dict[int, float]) -> tuple[int, list[dict[str, Any]]]:
    knee = LEVELS[0]
    path: list[dict[str, Any]] = []
    for left, right in zip(LEVELS, LEVELS[1:]):
        rose = rates[right] > rates[left]
        path.append(
            {
                "from": left,
                "to": right,
                "from_output_tok_s": rates[left],
                "to_output_tok_s": rates[right],
                "clean_rise": rose,
            }
        )
        if not rose:
            break
        knee = right
    return knee, path


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p95_ms": nearest_rank(values, 0.95),
        "max_ms": max(values) if values else None,
    }


def gpu_summary(paths: list[Path]) -> dict[str, Any]:
    temperatures: list[float] = []
    powers: list[float] = []
    clocks: list[float] = []
    utils: list[float] = []
    by_boot: list[dict[str, Any]] = []
    for path in paths:
        boot_t: list[float] = []
        boot_p: list[float] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) != 11:
                    continue
                try:
                    temperature = float(row[3])
                    power = float(row[4])
                    clock = float(row[6])
                    util = float(row[10])
                except ValueError:
                    continue
                temperatures.append(temperature)
                powers.append(power)
                clocks.append(clock)
                utils.append(util)
                boot_t.append(temperature)
                boot_p.append(power)
        if not boot_t:
            raise ValueError(f"{path}: no numeric GPU samples")
        by_boot.append(
            {
                "path": str(path),
                "samples_250ms": len(boot_t),
                "temperature_c_min": min(boot_t),
                "temperature_c_max": max(boot_t),
                "power_w_max": max(boot_p),
            }
        )
    return {
        "samples_250ms": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "power_w_max": max(powers),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
        "gpu_util_percent_median": statistics.median(utils),
        "gpu_util_percent_max": max(utils),
        "by_boot": by_boot,
    }


def integrity(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counter_keys = (
        "admission_session_defers",
        "admission_vram_defers",
        "step_oom_parks",
        "prefix_cache_evictions",
        "prefix_cache_hits",
        "prefix_cache_misses",
        "tokens_out",
    )
    return {
        "cells": len(cells),
        "clean_cells": sum(bool(cell["clean"]) for cell in cells),
        "requests": sum(int(cell["requests_n"]) for cell in cells),
        "requests_ok": sum(int(cell["requests_ok"]) for cell in cells),
        "completion_tokens": sum(int(cell["completion_tokens"]) for cell in cells),
        "cached_tokens": sum(int(cell["cached_tokens"]) for cell in cells),
        "computed_prompt_tokens": sum(
            int(cell["computed_prompt_tokens"]) for cell in cells
        ),
        "cached_token_drift_abs": sum(
            abs(int(cell["cached_tokens_in_drift"])) for cell in cells
        ),
        "prefix_hit_token_drift_abs": sum(
            abs(int(cell["prefix_cache_hit_tokens_drift"])) for cell in cells
        ),
        "prompt_token_drift_abs": sum(
            abs(int(cell["prompt_tokens_in_drift"])) for cell in cells
        ),
        "golden_required_failures": sum(
            int(cell["golden_required_failures"]) for cell in cells
        ),
        "counter_totals": {
            key: sum(int(cell["counter_deltas"][key]) for cell in cells)
            for key in counter_keys
        },
    }


def arm_summary(
    cells: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    gpu_paths: list[Path],
    health_paths: list[Path],
) -> dict[str, Any]:
    by_level: dict[int, list[dict[str, Any]]] = {}
    for level in LEVELS:
        selected = sorted(
            (cell for cell in cells if int(cell["concurrency"]) == level),
            key=lambda row: int(row["rep"]),
        )
        if len(selected) != 5 or [int(row["rep"]) for row in selected] != list(range(1, 6)):
            raise ValueError(f"c={level}: expected repetitions 1..5")
        by_level[level] = selected
    rates = {
        level: statistics.median(float(row["output_tok_s"]) for row in rows)
        for level, rows in by_level.items()
    }
    knee, path = rate_path(rates)
    return {
        "clean_throughput_knee": knee,
        "capacity_path": path,
        "output_tok_s_median": {str(level): rates[level] for level in LEVELS},
        "output_tok_s_by_rep": {
            str(level): [float(row["output_tok_s"]) for row in by_level[level]]
            for level in LEVELS
        },
        "hit_ttft_p95_ms_median": {
            str(level): statistics.median(
                float(row["ttft_hit"]["p95_ms"]) for row in by_level[level]
            )
            for level in LEVELS
        },
        "hit_ttft_p95_ms_by_rep": {
            str(level): [
                float(row["ttft_hit"]["p95_ms"]) for row in by_level[level]
            ]
            for level in LEVELS
        },
        "hit_ttft_ms_pooled": {
            str(level): distribution(
                [
                    float(row["ttft_ms"])
                    for row in requests
                    if int(row["concurrency"]) == level
                    and row["cache_role"] == "hit"
                ]
            )
            for level in LEVELS
        },
        "miss_ttft_p50_ms_median": {
            str(level): statistics.median(
                float(row["ttft_miss"]["p50_ms"]) for row in by_level[level]
            )
            for level in LEVELS
        },
        "integrity": integrity(cells),
        "health_tick_max_ms": max(
            int(read_json(path)["worker"]["tick_max_ms"]) for path in health_paths
        ),
        "gpu": gpu_summary(gpu_paths),
    }


def prime_telemetry(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, float | int | str | None]] = []
    failures: list[str] = []
    by_boot: list[dict[str, Any]] = []
    for path in paths:
        boot_rows: list[dict[str, float | int | str | None]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("[prime-batch] failed"):
                failures.append(f"{path}: {line}")
            match = PRIME_RE.match(line)
            if match is None:
                continue
            row: dict[str, float | int | str | None] = {
                "path": str(path),
                "batch": int(match.group("batch")),
                "tokens": int(match.group("tokens")),
                "carried": int(match.group("carried")),
                "partial": (
                    int(match.group("partial"))
                    if match.group("partial") is not None
                    else None
                ),
                "milliseconds": float(match.group("milliseconds")),
            }
            rows.append(row)
            boot_rows.append(row)
        by_boot.append(
            {
                "path": str(path),
                "batch_calls": len(boot_rows),
                "partial_batch_calls": sum(
                    int(row["partial"] or 0) > 0 for row in boot_rows
                ),
            }
        )
    return {
        "batch_calls": len(rows),
        "partial_batch_calls": sum(int(row["partial"] or 0) > 0 for row in rows),
        "carried_batch_calls": sum(int(row["carried"]) > 0 for row in rows),
        "failed_batch_calls": len(failures),
        "batch_widths": sorted({int(row["batch"]) for row in rows}),
        "tokens_per_call_median": (
            statistics.median(int(row["tokens"]) for row in rows) if rows else None
        ),
        "milliseconds_per_call_median": (
            statistics.median(float(row["milliseconds"]) for row in rows) if rows else None
        ),
        "by_boot": by_boot,
        "failures": failures,
    }


def paired_summary(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in LEVELS:
        base_rows = sorted(
            (row for row in baseline if int(row["concurrency"]) == level),
            key=lambda row: int(row["rep"]),
        )
        candidate_rows = sorted(
            (row for row in candidate if int(row["concurrency"]) == level),
            key=lambda row: int(row["rep"]),
        )
        if [int(row["rep"]) for row in base_rows] != [int(row["rep"]) for row in candidate_rows]:
            raise ValueError(f"c={level}: before/after rounds do not align")

        def deltas(getter: Any) -> list[float]:
            return [
                (getter(after) / getter(before) - 1.0) * 100.0
                for before, after in zip(base_rows, candidate_rows)
            ]

        throughput = deltas(lambda row: float(row["output_tok_s"]))
        hit_ttft = deltas(lambda row: float(row["ttft_hit"]["p95_ms"]))
        miss_ttft = deltas(lambda row: float(row["ttft_miss"]["p50_ms"]))
        result[str(level)] = {
            "output_tok_s_delta_percent_by_round": throughput,
            "output_tok_s_delta_percent_median": statistics.median(throughput),
            "hit_ttft_p95_delta_percent_by_round": hit_ttft,
            "hit_ttft_p95_delta_percent_median": statistics.median(hit_ttft),
            "miss_ttft_p50_delta_percent_by_round": miss_ttft,
            "miss_ttft_p50_delta_percent_median": statistics.median(miss_ttft),
        }
    return result


def output_comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
        return {
            (int(row["rep"]), int(row["concurrency"]), int(row["index"])): row
            for row in rows
        }

    before = keyed(baseline)
    after = keyed(candidate)
    if before.keys() != after.keys():
        raise ValueError("before/after request identities do not align")
    for key in before:
        if before[key]["cache_role"] != after[key]["cache_role"]:
            raise ValueError(f"{key}: before/after cache roles differ")
    by_level = {}
    for level in LEVELS:
        keys = [key for key in before if key[1] == level]
        matches = sum(before[key]["text_sha256"] == after[key]["text_sha256"] for key in keys)
        by_level[str(level)] = {"requests": len(keys), "matching_text_sha256": matches}
    total = len(before)
    matches = sum(before[key]["text_sha256"] == after[key]["text_sha256"] for key in before)
    return {
        "requests": total,
        "matching_text_sha256": matches,
        "different_text_sha256": total - matches,
        "by_level": by_level,
        "interpretation": (
            "observational only: concurrent batch composition is an admitted numeric class; "
            "the required output gates are the model-backed argmax and K=1..8 receipts"
        ),
    }


def daily_gross(output_tok_s: float, workload: dict[str, Any]) -> dict[str, float]:
    input_price = 0.287
    output_price = 2.751
    completion = int(workload["completion_tokens"])
    prompt = int(workload["prompt_tokens"])
    requests_s = output_tok_s / completion
    input_tok_s = requests_s * prompt
    return {
        "requests_s": requests_s,
        "billed_input_tok_s": input_tok_s,
        "input_usd_day": input_tok_s * 86_400 * input_price / 1_000_000,
        "output_usd_day": output_tok_s * 86_400 * output_price / 1_000_000,
        "gross_usd_day": (
            input_tok_s * 86_400 * input_price / 1_000_000
            + output_tok_s * 86_400 * output_price / 1_000_000
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab", required=True, type=Path)
    parser.add_argument("--workload-lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if not (args.ab / "ab.ok").exists():
        raise ValueError("A/B completion sentinel is absent")

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gpu_paths: dict[str, list[Path]] = defaultdict(list)
    health_paths: dict[str, list[Path]] = defaultdict(list)
    server_paths: dict[str, list[Path]] = defaultdict(list)
    provenances: dict[str, list[dict[str, str]]] = defaultdict(list)
    input_sets: dict[str, list[list[str]]] = defaultdict(list)
    boot_order: list[dict[str, Any]] = []
    boot_knees: dict[str, list[dict[str, int]]] = defaultdict(list)

    for child in sorted(path for path in args.ab.iterdir() if path.is_dir()):
        match = CHILD_RE.match(child.name)
        if match is None:
            raise ValueError(f"unexpected A/B child: {child.name}")
        arm = match.group("arm")
        round_n = int(match.group("round"))
        order_n = int(match.group("order"))
        if not (child / "run.ok").exists():
            raise ValueError(f"{child}: run sentinel absent")
        rows = read_jsonl(child / "sweep.jsonl")
        protocol = one(rows, "protocol", child / "sweep.jsonl")
        summary = one(rows, "summary", child / "sweep.jsonl")
        if protocol["levels"] != list(LEVELS):
            raise ValueError(f"{child}: concurrency levels drifted")
        if int(protocol["rep_start"]) != round_n or int(protocol["repetitions"]) != 1:
            raise ValueError(f"{child}: repetition metadata mismatch")
        if protocol["workload_lock_sha256"] != WORKLOAD_SHA256:
            raise ValueError(f"{child}: workload hash drifted")
        if protocol["prompt_ids_sha256_canonical_json"] != PROMPT_SHA256:
            raise ValueError(f"{child}: prompt hash drifted")
        if summary["verdict"] != "PASS" or int(summary["cells"]) != len(LEVELS):
            raise ValueError(f"{child}: incomplete or failed sweep")
        cells[arm].extend(row for row in rows if row.get("kind") == "cell")
        requests[arm].extend(row for row in rows if row.get("kind") == "request")
        gpu_paths[arm].append(child / "gpu-250ms.csv")
        health_paths[arm].append(child / "health-final.json")
        server_paths[arm].append(child / "server.log")
        provenances[arm].append(parse_provenance(child / "provenance.txt"))
        input_sets[arm].append(input_hashes(child / "SHA256SUMS.input"))
        boot_order.append({"round": round_n, "order": order_n, "arm": arm})
        boot_knees[arm].append(
            {"round": round_n, "clean_throughput_knee": int(summary["clean_throughput_knee"])}
        )

    boot_order.sort(key=lambda row: (row["round"], row["order"]))
    expected_order = [
        {"round": round_n, "order": order_n, "arm": arm}
        for round_n in range(1, 6)
        for order_n, arm in enumerate(
            ("baseline", "candidate") if round_n % 2 else ("candidate", "baseline"), 1
        )
    ]
    if boot_order != expected_order:
        raise ValueError("A/B boot order is not the frozen alternating order")

    expected = {
        "baseline": (BASE_SOURCE, BASE_SERVER_SHA256),
        "candidate": (CANDIDATE_SOURCE, CANDIDATE_SERVER_SHA256),
    }
    frozen_configs: dict[str, list[tuple[str, ...]]] = {}
    for arm in ("baseline", "candidate"):
        expected_source, expected_binary = expected[arm]
        for provenance, hashes in zip(provenances[arm], input_sets[arm]):
            if provenance["harness_source"] != HARNESS_SOURCE:
                raise ValueError(f"{arm}: harness source drifted")
            if provenance["runtime_source"] != expected_source:
                raise ValueError(f"{arm}: runtime source drifted")
            if provenance["runtime_binary_sha256"] != expected_binary:
                raise ValueError(f"{arm}: runtime binary drifted")
            if provenance["MEMRA_PRIME_BATCH"] != "<unset>":
                raise ValueError(f"{arm}: scored arm is not the naked default")
            if hashes[0] != MODEL_SHA256 or hashes[1] != expected_binary:
                raise ValueError(f"{arm}: model/server input hash drifted")
            if hashes[2:] != input_sets[arm][0][2:]:
                raise ValueError(f"{arm}: frozen harness input hash drifted")
        frozen_configs[arm] = sorted(
            {
                (
                    row["MEMRA_PREFIX_CACHE_MB"],
                    row["MEMRA_PREFIX_DEDUP"],
                    row["MEMRA_REUSE_POOL"],
                    row["MEMRA_AFFINITY"],
                    row["MEMRA_MAX_SESSIONS"],
                    row["MEMRA_SERVE_SPEC"],
                    row["MEMRA_DECODE_BATCH_CAP"],
                    row["MEMRA_PREFILL_TICK"],
                    row["MEMRA_PRIME_BATCH"],
                )
                for row in provenances[arm]
            }
        )
        if len(frozen_configs[arm]) != 1:
            raise ValueError(f"{arm}: configuration drifted across boots")
    if input_sets["baseline"][0][2:] != input_sets["candidate"][0][2:]:
        raise ValueError("before/after frozen harness inputs differ")

    arms = {
        arm: arm_summary(
            cells[arm], requests[arm], gpu_paths[arm], health_paths[arm]
        )
        for arm in ("baseline", "candidate")
    }
    for arm in arms:
        arms[arm]["per_boot_knees"] = sorted(
            boot_knees[arm], key=lambda row: row["round"]
        )
        arms[arm]["configuration"] = frozen_configs[arm][0]
        arms[arm]["prime_batch_telemetry"] = prime_telemetry(server_paths[arm])
    paired = paired_summary(cells["baseline"], cells["candidate"])
    outputs = output_comparison(requests["baseline"], requests["candidate"])
    pooled_hit_ttft = {
        str(level): {
            "baseline_p95_ms": arms["baseline"]["hit_ttft_ms_pooled"][str(level)][
                "p95_ms"
            ],
            "candidate_p95_ms": arms["candidate"]["hit_ttft_ms_pooled"][str(level)][
                "p95_ms"
            ],
            "delta_percent": (
                float(
                    arms["candidate"]["hit_ttft_ms_pooled"][str(level)]["p95_ms"]
                )
                / float(
                    arms["baseline"]["hit_ttft_ms_pooled"][str(level)]["p95_ms"]
                )
                - 1.0
            )
            * 100.0,
        }
        for level in LEVELS
    }
    workload = read_json(args.workload_lock)
    base_knee = int(arms["baseline"]["clean_throughput_knee"])
    candidate_knee = int(arms["candidate"]["clean_throughput_knee"])
    base_rate = float(arms["baseline"]["output_tok_s_median"][str(base_knee)])
    candidate_rate = float(arms["candidate"]["output_tok_s_median"][str(candidate_knee)])
    base_gross = daily_gross(base_rate, workload)
    candidate_gross = daily_gross(candidate_rate, workload)

    result = {
        "schema": "memra.coldhol.analysis.v1",
        "provenance": {
            "harness_source": HARNESS_SOURCE,
            "baseline_runtime_source": BASE_SOURCE,
            "candidate_runtime_source": CANDIDATE_SOURCE,
            "baseline_server_sha256": BASE_SERVER_SHA256,
            "candidate_server_sha256": CANDIDATE_SERVER_SHA256,
            "model_sha256": MODEL_SHA256,
            "workload_lock_sha256": sha256_file(args.workload_lock),
            "prompt_ids_sha256_canonical_json": PROMPT_SHA256,
            "ab_manifest_sha256": sha256_file(args.ab / "MANIFEST.sha256"),
            "thermal_regime": (
                "ten alternating whole-server boots under one Box1 flock hold; "
                "physical GPU0 only, no clock changes or artificial cooldown"
            ),
        },
        "boot_order": boot_order,
        "baseline": arms["baseline"],
        "candidate": arms["candidate"],
        "paired": paired,
        "pooled_hit_ttft_p95": pooled_hit_ttft,
        "output_hash_observation": outputs,
        "verdict": {
            "knee_before": base_knee,
            "knee_after": candidate_knee,
            "knee_moved": candidate_knee > base_knee,
            "all_cells_clean": all(
                arms[arm]["integrity"]["clean_cells"] == 25
                for arm in ("baseline", "candidate")
            ),
            "candidate_partial_batches_observed": (
                arms["candidate"]["prime_batch_telemetry"]["partial_batch_calls"] > 0
            ),
            "batch_failures": sum(
                int(arms[arm]["prime_batch_telemetry"]["failed_batch_calls"])
                for arm in ("baseline", "candidate")
            ),
        },
        "revenue": {
            "assumption": (
                "24h continuous frozen 4860-input/60-output mix billed at the stated "
                "input/output prices; gross usage revenue only, not demand or margin"
            ),
            "input_usd_per_million_tokens": 0.287,
            "output_usd_per_million_tokens": 2.751,
            "baseline_at_knee": base_gross,
            "candidate_at_knee": candidate_gross,
            "gross_usd_day_delta": (
                candidate_gross["gross_usd_day"] - base_gross["gross_usd_day"]
            ),
            "knee_concurrency_delta": candidate_knee - base_knee,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "knee_before": base_knee,
                "knee_after": candidate_knee,
                "c8_delta_percent": paired["8"]["output_tok_s_delta_percent_median"],
                "c12_delta_percent": paired["12"]["output_tok_s_delta_percent_median"],
                "c16_delta_percent": paired["16"]["output_tok_s_delta_percent_median"],
                "gross_usd_day_delta": result["revenue"]["gross_usd_day_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
