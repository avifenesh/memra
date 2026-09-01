#!/usr/bin/env python3
"""Reduce the Q27 knee baseline, tick trace, and interleaved config A/B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LEVELS = (8, 12, 16, 20, 24)
CHILD_RE = re.compile(r"^r(?P<round>\d+)-o(?P<order>\d+)-(?P<arm>baseline|candidate)$")
TICK_RE = re.compile(
    r"^\[tick\] act=(?P<active>\d+) int=(?P<interactive>\d+) "
    r"priming=(?P<priming>\d+) ready=(?P<ready>\d+).*?"
    r"prefill_single_calls=(?P<single_calls>\d+) "
    r"prefill_single_tokens=(?P<single_tokens>\d+) "
    r"prefill_batch_calls=(?P<batch_calls>\d+) "
    r"prefill_batch_tokens=(?P<batch_tokens>\d+) "
    r"prefill_ms=(?P<prefill_ms>[0-9.]+) decode_ms=(?P<decode_ms>[0-9.]+)$"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(rows: list[dict[str, Any]], kind: str, source: Path) -> dict[str, Any]:
    selected = [row for row in rows if row.get("kind") == kind]
    if len(selected) != 1:
        raise ValueError(f"{source}: expected one {kind} row, got {len(selected)}")
    return selected[0]


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


def parse_provenance(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key and " " not in key:
                result[key] = value
    return result


def gpu_summary(paths: list[Path]) -> dict[str, Any]:
    temperatures: list[float] = []
    powers: list[float] = []
    clocks: list[float] = []
    utils: list[float] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) != 11:
                    continue
                try:
                    temperatures.append(float(row[3]))
                    powers.append(float(row[4]))
                    clocks.append(float(row[6]))
                    utils.append(float(row[10]))
                except ValueError:
                    continue
    if not temperatures:
        raise ValueError("GPU sampler receipt has no numeric rows")
    return {
        "samples_250ms": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "power_w_max": max(powers),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
        "gpu_util_percent_median": statistics.median(utils),
        "gpu_util_percent_max": max(utils),
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
    by_level: dict[int, list[dict[str, Any]]] = {
        level: sorted(
            (cell for cell in cells if int(cell["concurrency"]) == level),
            key=lambda cell: int(cell["rep"]),
        )
        for level in LEVELS
    }
    for level, selected in by_level.items():
        if len(selected) != 5 or [int(row["rep"]) for row in selected] != list(range(1, 6)):
            raise ValueError(f"c={level}: expected repetitions 1..5")
    rates = {
        level: statistics.median(float(row["output_tok_s"]) for row in selected)
        for level, selected in by_level.items()
    }
    knee, path = rate_path(rates)
    return {
        "clean_throughput_knee": knee,
        "capacity_path": path,
        "output_tok_s_median": {str(key): value for key, value in rates.items()},
        "output_tok_s_by_rep": {
            str(level): [float(row["output_tok_s"]) for row in selected]
            for level, selected in by_level.items()
        },
        "ttft_hit_p95_ms_median": {
            str(level): statistics.median(
                float(row["ttft_hit"]["p95_ms"]) for row in selected
            )
            for level, selected in by_level.items()
        },
        "ttft_miss_p50_ms_median": {
            str(level): statistics.median(
                float(row["ttft_miss"]["p50_ms"]) for row in selected
            )
            for level, selected in by_level.items()
        },
        "first_wave_miss_requests_by_rep": {
            str(level): [
                sum(
                    row.get("cache_role") == "miss" and int(row["index"]) < level
                    for row in requests
                    if int(row["concurrency"]) == level and int(row["rep"]) == rep
                )
                for rep in range(1, 6)
            ]
            for level in LEVELS
        },
        "integrity": integrity(cells),
        "health_tick_max_ms": max(
            int(read_json(path)["worker"]["tick_max_ms"]) for path in health_paths
        ),
        "gpu": gpu_summary(gpu_paths),
    }


def diagnostic_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path / "sweep.jsonl")
    sweep = one(rows, "summary", path / "sweep.jsonl")
    tick_rows: list[dict[str, float | int]] = []
    for line in (path / "server.log").read_text(encoding="utf-8").splitlines():
        match = TICK_RE.match(line)
        if match is None:
            continue
        row: dict[str, float | int] = {
            key: (float(value) if key.endswith("_ms") else int(value))
            for key, value in match.groupdict().items()
        }
        tick_rows.append(row)
    decode: dict[int, list[float]] = defaultdict(list)
    prefill: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(list)
    for row in tick_rows:
        ready = int(row["ready"])
        decode_ms = float(row["decode_ms"])
        if int(row["priming"]) == 0 and ready > 0 and decode_ms > 0:
            decode[ready].append(decode_ms)
        prefill_ms = float(row["prefill_ms"])
        if prefill_ms > 0:
            key = (
                int(row["priming"]),
                int(row["single_calls"]),
                int(row["single_tokens"]),
                int(row["batch_calls"]),
                int(row["batch_tokens"]),
            )
            prefill[key].append(prefill_ms)
    decode_receipt = {
        str(width): {
            "ticks": len(samples),
            "decode_ms_median": statistics.median(samples),
            "decode_ms_min": min(samples),
            "decode_ms_max": max(samples),
            "row_output_tok_s": width / (statistics.median(samples) / 1000.0),
        }
        for width, samples in sorted(decode.items())
        if width in {4, 8, 12, 16, 20, 24}
    }
    partition_receipts: dict[str, Any] = {}
    for width, parts in ((20, (16, 4)), (24, (16, 8))):
        if all(str(part) in decode_receipt for part in parts) and str(width) in decode_receipt:
            expected = sum(
                float(decode_receipt[str(part)]["decode_ms_median"]) for part in parts
            )
            observed = float(decode_receipt[str(width)]["decode_ms_median"])
            partition_receipts[str(width)] = {
                "partition": list(parts),
                "component_sum_ms": expected,
                "observed_ready_width_ms": observed,
                "absolute_difference_ms": observed - expected,
            }
    return {
        "single_run_only": True,
        "levels": sweep["levels"],
        "output_tok_s": sweep["median_output_tok_s"],
        "clean_throughput_knee": sweep["clean_throughput_knee"],
        "tick_rows": len(tick_rows),
        "steady_decode": decode_receipt,
        "serial_partition_receipts": partition_receipts,
        "prefill_signatures": [
            {
                "priming_after_phase": key[0],
                "single_calls": key[1],
                "single_tokens": key[2],
                "batch_calls": key[3],
                "batch_tokens": key[4],
                "ticks": len(samples),
                "prefill_ms_median": statistics.median(samples),
                "prefill_ms_min": min(samples),
                "prefill_ms_max": max(samples),
            }
            for key, samples in sorted(prefill.items())
        ],
        "gpu": gpu_summary([path / "gpu-250ms.csv"]),
        "health": read_json(path / "health-final.json"),
    }


def daily_gross(output_tok_s: float, workload: dict[str, Any]) -> dict[str, float]:
    output_price = 2.751
    input_price = 0.287
    completion = int(workload["completion_tokens"])
    prompt = int(workload["prompt_tokens"])
    request_s = output_tok_s / completion
    billed_prompt_tok_s = request_s * prompt
    seconds_per_day = 86_400
    input_daily = billed_prompt_tok_s * input_price / 1_000_000 * seconds_per_day
    output_daily = output_tok_s * output_price / 1_000_000 * seconds_per_day
    return {
        "requests_s": request_s,
        "billed_prompt_tok_s": billed_prompt_tok_s,
        "input_usd_day": input_daily,
        "output_usd_day": output_daily,
        "gross_usd_day": input_daily + output_daily,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--ab", required=True, type=Path)
    parser.add_argument("--workload-lock", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    baseline_rows = read_jsonl(args.baseline / "sweep.jsonl")
    baseline_protocol = one(baseline_rows, "protocol", args.baseline / "sweep.jsonl")
    baseline_summary_row = one(baseline_rows, "summary", args.baseline / "sweep.jsonl")
    baseline_cells = [row for row in baseline_rows if row.get("kind") == "cell"]
    baseline_requests = [row for row in baseline_rows if row.get("kind") == "request"]
    baseline = arm_summary(
        baseline_cells,
        baseline_requests,
        [args.baseline / "gpu-250ms.csv"],
        [args.baseline / "health-final.json"],
    )
    if baseline["clean_throughput_knee"] != baseline_summary_row["clean_throughput_knee"]:
        raise ValueError("baseline knee disagrees with captured summary")

    ab_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ab_requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ab_gpu: dict[str, list[Path]] = defaultdict(list)
    ab_health: dict[str, list[Path]] = defaultdict(list)
    ab_configs: dict[str, list[dict[str, str]]] = defaultdict(list)
    boot_knees: dict[str, list[int]] = defaultdict(list)
    boot_order: list[dict[str, Any]] = []
    for child in sorted(path for path in args.ab.iterdir() if path.is_dir()):
        match = CHILD_RE.match(child.name)
        if match is None:
            raise ValueError(f"unexpected A/B child: {child.name}")
        arm = match.group("arm")
        round_n = int(match.group("round"))
        order_n = int(match.group("order"))
        rows = read_jsonl(child / "sweep.jsonl")
        protocol = one(rows, "protocol", child / "sweep.jsonl")
        summary = one(rows, "summary", child / "sweep.jsonl")
        if int(protocol["rep_start"]) != round_n or int(protocol["repetitions"]) != 1:
            raise ValueError(f"{child}: repetition metadata mismatch")
        if summary["verdict"] != "PASS" or summary["cells"] != len(LEVELS):
            raise ValueError(f"{child}: incomplete or failed sweep")
        ab_cells[arm].extend(row for row in rows if row.get("kind") == "cell")
        ab_requests[arm].extend(row for row in rows if row.get("kind") == "request")
        ab_gpu[arm].append(child / "gpu-250ms.csv")
        ab_health[arm].append(child / "health-final.json")
        ab_configs[arm].append(parse_provenance(child / "provenance.txt"))
        boot_knees[arm].append(int(summary["clean_throughput_knee"]))
        boot_order.append({"round": round_n, "order": order_n, "arm": arm})
    if len(boot_order) != 10:
        raise ValueError(f"expected ten A/B boots, got {len(boot_order)}")
    boot_order.sort(key=lambda row: (row["round"], row["order"]))
    expected_order = [
        {"round": round_n, "order": order_n, "arm": arm}
        for round_n in range(1, 6)
        for order_n, arm in enumerate(
            ("baseline", "candidate") if round_n % 2 else ("candidate", "baseline"),
            1,
        )
    ]
    if boot_order != expected_order:
        raise ValueError("A/B boot order is not alternating by round")

    ab = {
        arm: arm_summary(ab_cells[arm], ab_requests[arm], ab_gpu[arm], ab_health[arm])
        for arm in ("baseline", "candidate")
    }
    for arm in ab:
        ab[arm]["per_boot_knees"] = sorted(boot_knees[arm])
        ab[arm]["configurations"] = sorted(
            {
                (
                    row["MEMRA_PREFIX_CACHE_MB"],
                    row["MEMRA_MAX_SESSIONS"],
                    row["MEMRA_DECODE_BATCH_CAP"],
                    row["MEMRA_PREFILL_TICK"],
                )
                for row in ab_configs[arm]
            }
        )
    paired: dict[str, Any] = {}
    for level in LEVELS:
        baseline_level = sorted(
            (row for row in ab_cells["baseline"] if int(row["concurrency"]) == level),
            key=lambda row: int(row["rep"]),
        )
        candidate_level = sorted(
            (row for row in ab_cells["candidate"] if int(row["concurrency"]) == level),
            key=lambda row: int(row["rep"]),
        )
        deltas = [
            (float(candidate["output_tok_s"]) / float(base["output_tok_s"]) - 1.0) * 100.0
            for base, candidate in zip(baseline_level, candidate_level)
        ]
        hit_p95_deltas = [
            (
                float(candidate["ttft_hit"]["p95_ms"])
                / float(base["ttft_hit"]["p95_ms"])
                - 1.0
            )
            * 100.0
            for base, candidate in zip(baseline_level, candidate_level)
        ]
        miss_p50_deltas = [
            (
                float(candidate["ttft_miss"]["p50_ms"])
                / float(base["ttft_miss"]["p50_ms"])
                - 1.0
            )
            * 100.0
            for base, candidate in zip(baseline_level, candidate_level)
        ]
        paired[str(level)] = {
            "output_tok_s_delta_percent_by_round": deltas,
            "output_tok_s_delta_percent_median": statistics.median(deltas),
            "hit_ttft_p95_delta_percent_by_round": hit_p95_deltas,
            "hit_ttft_p95_delta_percent_median": statistics.median(hit_p95_deltas),
            "miss_ttft_p50_delta_percent_by_round": miss_p50_deltas,
            "miss_ttft_p50_delta_percent_median": statistics.median(miss_p50_deltas),
        }

    workload = read_json(args.workload_lock)
    baseline_rate = float(baseline["output_tok_s_median"][str(baseline["clean_throughput_knee"])])
    ab_base_rate = float(
        ab["baseline"]["output_tok_s_median"][str(ab["baseline"]["clean_throughput_knee"])]
    )
    ab_candidate_rate = float(
        ab["candidate"]["output_tok_s_median"][str(ab["candidate"]["clean_throughput_knee"])]
    )
    result = {
        "schema": "memra.kneeraise.analysis.v1",
        "provenance": {
            "runtime_source": parse_provenance(args.baseline / "provenance.txt")[
                "runtime_source"
            ],
            "model_sha256": (args.baseline / "SHA256SUMS.input").read_text(
                encoding="utf-8"
            ).split()[0],
            "workload_lock_sha256": sha256_file(args.workload_lock),
            "prompt_ids_sha256_canonical_json": baseline_protocol[
                "prompt_ids_sha256_canonical_json"
            ],
            "baseline_manifest_sha256": sha256_file(args.baseline / "MANIFEST.sha256"),
            "diagnostic_manifest_sha256": sha256_file(args.diagnostic / "MANIFEST.sha256"),
            "ab_manifest_sha256": sha256_file(args.ab / "MANIFEST.sha256"),
        },
        "baseline": baseline,
        "diagnostic": diagnostic_summary(args.diagnostic),
        "prefill2048_ab": {
            "boot_order": boot_order,
            "baseline": ab["baseline"],
            "candidate": ab["candidate"],
            "paired": paired,
            "knee_before": ab["baseline"]["clean_throughput_knee"],
            "knee_after": ab["candidate"]["clean_throughput_knee"],
            "knee_moved": (
                ab["candidate"]["clean_throughput_knee"]
                > ab["baseline"]["clean_throughput_knee"]
            ),
        },
        "binding_receipts": {
            "first_wave_misses": baseline["first_wave_miss_requests_by_rep"],
            "admission_session_defers": baseline["integrity"]["counter_totals"][
                "admission_session_defers"
            ],
            "admission_vram_defers": baseline["integrity"]["counter_totals"][
                "admission_vram_defers"
            ],
            "step_oom_parks": baseline["integrity"]["counter_totals"]["step_oom_parks"],
            "prefix_cache_bytes_final": read_json(args.baseline / "metrics-final.json")[
                "prefix_cache_bytes"
            ],
            "prefix_cache_entries_final": read_json(args.baseline / "metrics-final.json")[
                "prefix_cache_entries"
            ],
            "cuda_driver_free_bytes_final": read_json(args.baseline / "metrics-final.json")[
                "cuda_driver_free_bytes"
            ],
        },
        "revenue": {
            "assumption": (
                "24h continuous benchmark mix billed at published input/output list prices; "
                "gross usage revenue only, not demand, margin, or a sold-cap recommendation"
            ),
            "input_usd_per_million_tokens": 0.287,
            "output_usd_per_million_tokens": 2.751,
            "characterization_baseline_at_knee": daily_gross(baseline_rate, workload),
            "interleaved_baseline_at_knee": daily_gross(ab_base_rate, workload),
            "prefill2048_candidate_at_knee": daily_gross(ab_candidate_rate, workload),
            "knee_concurrency_gain": (
                ab["candidate"]["clean_throughput_knee"]
                - ab["baseline"]["clean_throughput_knee"]
            ),
            "sold_cap": 4,
            "headroom_percent_before": (
                ab["baseline"]["clean_throughput_knee"] / 4 - 1.0
            )
            * 100.0,
            "headroom_percent_after": (
                ab["candidate"]["clean_throughput_knee"] / 4 - 1.0
            )
            * 100.0,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_knee": baseline["clean_throughput_knee"],
                "ab_knee_before": result["prefill2048_ab"]["knee_before"],
                "ab_knee_after": result["prefill2048_ab"]["knee_after"],
                "c16_paired_delta_percent_median": paired["16"][
                    "output_tok_s_delta_percent_median"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
