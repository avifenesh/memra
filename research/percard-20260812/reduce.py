#!/usr/bin/env python3
"""Reduce the per-card campaign into paired curves and resident-idle controls."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics


WIDTHS = (1, 2, 4, 8, 12, 16, 24)
TARGETS = ("q27", "q35")
CONDITIONS = ("paired", "q27", "q35")
LABEL = re.compile(r"r(?P<rep>\d+)-p(?P<position>\d+)-c(?P<width>\d+)-(?P<condition>paired|q27|q35)$")
PRICES = {
    "q27": {"input_per_m": 0.285, "output_per_m": 2.816},
    "q35": {"input_per_m": 0.125, "output_per_m": 1.065},
}


def median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def thermal(path: pathlib.Path) -> dict:
    per_gpu: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if len(row) < 11:
                continue
            try:
                gpu = row[1].strip()
                values = {
                    "temperature_c": float(row[3]),
                    "power_w": float(row[4]),
                    "sm_clock_mhz": float(row[6]),
                    "memory_used_mib": float(row[8]),
                    "utilization_pct": float(row[10]),
                }
            except ValueError:
                continue
            bucket = per_gpu.setdefault(gpu, {key: [] for key in values})
            for key, value in values.items():
                bucket[key].append(value)
    if not per_gpu:
        raise ValueError(f"no GPU samples parsed from {path}")
    return {
        "sample_interval_ms": 250,
        "artificial_cooldown": False,
        "rows": sum(len(bucket["temperature_c"]) for bucket in per_gpu.values()),
        "gpus": {
            gpu: {
                f"{key}_min": min(values)
                for key, values in bucket.items()
            }
            | {
                f"{key}_max": max(values)
                for key, values in bucket.items()
            }
            for gpu, bucket in sorted(per_gpu.items())
        },
    }


def collect(root: pathlib.Path) -> list[dict]:
    points = []
    for path in sorted((root / "perf").glob("*/score.jsonl")):
        match = LABEL.fullmatch(path.parent.name)
        if not match:
            raise ValueError(f"bad score directory label: {path.parent.name}")
        metadata = {key: int(value) if key != "condition" else value for key, value in match.groupdict().items()}
        summaries = [row for row in read_jsonl(path) if row.get("kind") == "target_summary"]
        expected_targets = TARGETS if metadata["condition"] == "paired" else (metadata["condition"],)
        if sorted(row["target"] for row in summaries) != sorted(expected_targets):
            raise ValueError(f"target set mismatch in {path}: {summaries}")
        for summary in summaries:
            point = {**metadata, **summary, "receipt": str(path.relative_to(root))}
            if point["n_error"] or point["n_ok"] != point["n_requests"]:
                raise ValueError(f"request failure in {path}: {point}")
            expected_tokens = int(point["concurrency"]) * int(point["max_tokens"])
            counters = point["counter_deltas"]
            if int(point["completion_tokens_total"]) != expected_tokens:
                raise ValueError(f"token total mismatch in {path}: {point}")
            if any(int(counters[key]) != expected for key, expected in (
                ("admitted", int(point["concurrency"])),
                ("completed", int(point["concurrency"])),
                ("tokens_out", expected_tokens),
            )):
                raise ValueError(f"worker accounting mismatch in {path}: {point}")
            points.append(point)
    expected_points = 3 * len(WIDTHS) * 4  # paired contributes two targets; two solo controls.
    if len(points) != expected_points:
        raise ValueError(f"wanted {expected_points} target points, found {len(points)}")
    for rep in range(1, 4):
        for width in WIDTHS:
            for condition in CONDITIONS:
                targets = TARGETS if condition == "paired" else (condition,)
                for target in targets:
                    matches = [
                        point
                        for point in points
                        if point["rep"] == rep
                        and point["width"] == width
                        and point["condition"] == condition
                        and point["target"] == target
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"wanted one r{rep} c{width} {condition}/{target}, got {len(matches)}"
                        )
    return points


def aggregate_curve(points: list[dict], target: str, condition: str) -> dict:
    result = {}
    for width in WIDTHS:
        rows = sorted(
            (
                point
                for point in points
                if point["target"] == target
                and point["condition"] == condition
                and point["width"] == width
            ),
            key=lambda point: point["rep"],
        )
        if len(rows) != 3:
            raise ValueError(f"wanted N=3 for {target}/{condition}/c{width}")
        spec_deltas = {
            key: [
                int((row.get("spec_after") or {}).get(key, 0))
                - int((row.get("spec_before") or {}).get(key, 0))
                for row in rows
            ]
            for key in ("rounds", "drafted", "accepted")
        }
        result[f"c{width}"] = {
            "N": 3,
            "aggregate_tok_s": [float(row["aggregate_tok_s"]) for row in rows],
            "aggregate_tok_s_median": median(rows, "aggregate_tok_s"),
            "aggregate_prompt_tok_s": [float(row["aggregate_prompt_tok_s"]) for row in rows],
            "aggregate_prompt_tok_s_median": median(rows, "aggregate_prompt_tok_s"),
            "ttft_p50_s": [float(row["ttft_p50_s"]) for row in rows],
            "ttft_p50_s_median": median(rows, "ttft_p50_s"),
            "ttft_p99_s": [float(row["ttft_p99_s"]) for row in rows],
            "ttft_p99_s_median": median(rows, "ttft_p99_s"),
            "latency_p99_s": [float(row["latency_p99_s"]) for row in rows],
            "latency_p99_s_median": median(rows, "latency_p99_s"),
            "request_start_spread_ms": [float(row["request_start_spread_ms"]) for row in rows],
            "replicate_positions": [int(row["position"]) for row in rows],
            "spec_rounds": spec_deltas["rounds"],
            "spec_drafted": spec_deltas["drafted"],
            "spec_accepted": spec_deltas["accepted"],
            "spec_active_replicates": sum(
                value > 0 for value in spec_deltas["rounds"]
            ),
            "receipts": [row["receipt"] for row in rows],
        }
    return result


def capacity(curve: dict, target: str) -> dict:
    first_non_rising = None
    first_tail_crossing = None
    previous = None
    eligible = []
    for width in WIDTHS:
        row = curve[f"c{width}"]
        throughput = float(row["aggregate_tok_s_median"])
        if previous is not None and throughput <= previous and first_non_rising is None:
            first_non_rising = width
        if float(row["ttft_p99_s_median"]) > 15.0 and first_tail_crossing is None:
            first_tail_crossing = width
        if float(row["ttft_p99_s_median"]) <= 15.0:
            eligible.append(width)
        previous = throughput
    if not eligible:
        raise ValueError(f"no {target} width meets the 15-second TTFT bar")
    throughput_optimal = max(
        eligible,
        key=lambda width: float(curve[f"c{width}"]["aggregate_tok_s_median"]),
    )
    optimal_tps = float(
        curve[f"c{throughput_optimal}"]["aggregate_tok_s_median"]
    )
    knee = min(
        width
        for width in eligible
        if float(curve[f"c{width}"]["aggregate_tok_s_median"])
        >= optimal_tps * 0.95
    )
    row = curve[f"c{knee}"]
    output_tps = float(row["aggregate_tok_s_median"])
    input_tps = float(row["aggregate_prompt_tok_s_median"])
    prices = PRICES[target]
    output_per_day = output_tps * 86_400
    input_per_day = input_tps * 86_400
    return {
        "knee_rule": "lowest measured width reaching at least 95% of the best paired median aggregate tok/s among widths whose median window p99 TTFT is at most 15s",
        "first_non_rising_width": first_non_rising,
        "first_ttft_over_15s_width": first_tail_crossing,
        "knee_width": knee,
        "throughput_optimal_rule": "highest paired median aggregate tok/s among measured widths with median window p99 TTFT at most 15s",
        "throughput_optimal_width": throughput_optimal,
        "throughput_optimal_output_tok_s": optimal_tps,
        "largest_width_under_15s_ttft": max(eligible),
        "aggregate_output_tok_s": output_tps,
        "aggregate_input_tok_s": input_tps,
        "output_tokens_per_day": output_per_day,
        "input_tokens_per_day": input_per_day,
        "price_usd_per_million": prices,
        "output_only_usd_per_day": output_per_day / 1_000_000 * prices["output_per_m"],
        "input_usd_per_day": input_per_day / 1_000_000 * prices["input_per_m"],
        "benchmark_mix_usd_per_day": (
            output_per_day / 1_000_000 * prices["output_per_m"]
            + input_per_day / 1_000_000 * prices["input_per_m"]
        ),
    }


def interference(paired: dict, solo: dict) -> dict:
    result = {}
    for width in WIDTHS:
        p = paired[f"c{width}"]
        s = solo[f"c{width}"]
        pt = float(p["aggregate_tok_s_median"])
        st = float(s["aggregate_tok_s_median"])
        p50 = float(p["ttft_p50_s_median"])
        s50 = float(s["ttft_p50_s_median"])
        p99 = float(p["ttft_p99_s_median"])
        s99 = float(s["ttft_p99_s_median"])
        plat = float(p["latency_p99_s_median"])
        slat = float(s["latency_p99_s_median"])
        throughput_rep_pct = [
            (float(paired_value) / float(solo_value) - 1.0) * 100.0
            for paired_value, solo_value in zip(
                p["aggregate_tok_s"], s["aggregate_tok_s"], strict=True
            )
        ]
        ttft_p99_rep_delta = [
            float(paired_value) - float(solo_value)
            for paired_value, solo_value in zip(
                p["ttft_p99_s"], s["ttft_p99_s"], strict=True
            )
        ]
        latency_p99_rep_delta = [
            float(paired_value) - float(solo_value)
            for paired_value, solo_value in zip(
                p["latency_p99_s"], s["latency_p99_s"], strict=True
            )
        ]
        result[f"c{width}"] = {
            "N": 3,
            "paired_vs_solo_output_tok_s_pct": (pt / st - 1.0) * 100.0,
            "paired_vs_solo_output_tok_s_pct_by_rep": throughput_rep_pct,
            "paired_vs_solo_output_tok_s_pct_by_rep_median": statistics.median(
                throughput_rep_pct
            ),
            "paired_minus_solo_ttft_p50_s": p50 - s50,
            "paired_vs_solo_ttft_p50_pct": (p50 / s50 - 1.0) * 100.0 if s50 else None,
            "paired_minus_solo_ttft_p99_s": p99 - s99,
            "paired_minus_solo_ttft_p99_s_by_rep": ttft_p99_rep_delta,
            "paired_minus_solo_ttft_p99_s_by_rep_median": statistics.median(
                ttft_p99_rep_delta
            ),
            "paired_vs_solo_ttft_p99_pct": (p99 / s99 - 1.0) * 100.0 if s99 else None,
            "paired_minus_solo_latency_p99_s": plat - slat,
            "paired_minus_solo_latency_p99_s_by_rep": latency_p99_rep_delta,
            "paired_minus_solo_latency_p99_s_by_rep_median": statistics.median(
                latency_p99_rep_delta
            ),
            "paired_vs_solo_latency_p99_pct": (plat / slat - 1.0) * 100.0 if slat else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    points = collect(root)
    goldens = {
        target: json.loads((root / "exactness" / f"golden-{target}.json").read_text())
        for target in TARGETS
    }
    for target, receipt in goldens.items():
        if (
            receipt["verdict"] != "PASS"
            or receipt["matches"] != 10
            or receipt["repeats"] != 10
        ):
            raise ValueError(f"bad golden receipt for {target}: {receipt}")
    final_metrics = {
        target: json.loads((root / f"metrics-{target}-final.json").read_text())
        for target in TARGETS
    }
    server_logs = {
        target: (root / f"server-{target}.log").read_text(errors="replace")
        for target in TARGETS
    }
    curves = {}
    controls = {}
    capacities = {}
    interference_rows = {}
    spec = {}
    for target in TARGETS:
        curves[target] = aggregate_curve(points, target, "paired")
        controls[target] = aggregate_curve(points, target, target)
        capacities[target] = capacity(curves[target], target)
        interference_rows[target] = interference(curves[target], controls[target])
        spec_metrics = (final_metrics[target].get("spec") or {}).get(target)
        spec[target] = {
            "drafter_attached": "+draft" in server_logs[target]
            or "regime draft" in server_logs[target],
            "observed_spec_metrics": spec_metrics,
            "rounds": int((spec_metrics or {}).get("rounds", 0)),
            "drafted": int((spec_metrics or {}).get("drafted", 0)),
            "accepted": int((spec_metrics or {}).get("accepted", 0)),
            "gate_demotions_in_log": server_logs[target].count("[spec-gate] demoted"),
        }
    summary = {
        "schema": "memra.percard.capacity.v1",
        "source_commit": args.source,
        "bundled_main": "250ba819e83f868d395c01c6f315a4c6344f54cb",
        "rig": "london, 2x RTX PRO 6000 Blackwell Server Edition, one model per card",
        "protocol": "persistent simultaneous servers; exactness first; N=3 order-rotated paired and resident-idle controls; barrier-released 128-token requests; no artificial cooldown",
        "metric": "per-model completion tokens / global barrier release to that model's final drain",
        "goldens": goldens,
        "spec_decode": spec,
        "paired_curves": curves,
        "resident_idle_controls": controls,
        "capacity": capacities,
        "interference": interference_rows,
        "thermal_regime": thermal(root / "gpu-250ms.csv"),
        "price_source": "research/modelpick-20260812/REPORT.md effective input/output prices",
        "receipt": {
            "scored_target_points": len(points),
            "paired_windows": 3 * len(WIDTHS),
            "solo_control_windows": 2 * 3 * len(WIDTHS),
            "request_errors": sum(int(point["n_error"]) for point in points),
            "completion_tokens": sum(int(point["completion_tokens_total"]) for point in points),
        },
    }
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
