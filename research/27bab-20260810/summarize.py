#!/usr/bin/env python3
"""Summarize the immutable A/B/C receipts for the 27B-beside-Step campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty measurement series")
    return {
        "n": len(values),
        "values": values,
        "median": statistics.median(values),
        "p99_nearest_rank": nearest_rank(values, 99),
        "min": min(values),
        "max": max(values),
    }


def load_series(directory: pathlib.Path, pattern: str, key: str, expected_n: int = 5) -> dict:
    paths = sorted(directory.glob(pattern))
    if len(paths) != expected_n:
        raise ValueError(f"{directory}/{pattern}: expected {expected_n}, found {len(paths)}")
    values = [float(json.loads(path.read_text())[key]) for path in paths]
    return describe(values)


def weighted_spec(directory: pathlib.Path, pattern: str, expected_n: int = 5) -> dict:
    paths = sorted(directory.glob(pattern))
    if len(paths) != expected_n:
        raise ValueError(f"{directory}/{pattern}: expected {expected_n}, found {len(paths)}")
    accepted = 0
    drafted = 0
    rounds = 0
    for path in paths:
        delta = json.loads(path.read_text())["spec_metrics_delta"]
        accepted += int(delta["accepted"])
        drafted += int(delta["drafted"])
        rounds += int(delta["rounds"])
    return {
        "n": len(paths),
        "rounds": rounds,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def pct(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def thermal(directory: pathlib.Path) -> dict:
    result: dict[str, dict] = {}
    with (directory / "gpu.csv").open(newline="") as handle:
        for raw in csv.reader(handle):
            if len(raw) < 9:
                continue
            gpu = raw[1].strip()
            row = result.setdefault(
                gpu,
                {
                    "samples": 0,
                    "max_temp_c": -math.inf,
                    "max_power_w": -math.inf,
                    "max_memory_used_mib": -math.inf,
                    "min_memory_free_mib": math.inf,
                },
            )
            row["samples"] += 1
            row["max_temp_c"] = max(row["max_temp_c"], float(raw[3]))
            row["max_power_w"] = max(row["max_power_w"], float(raw[4]))
            row["max_memory_used_mib"] = max(row["max_memory_used_mib"], float(raw[6]))
            row["min_memory_free_mib"] = min(row["min_memory_free_mib"], float(raw[7]))
    return result


GPU_LINE = re.compile(
    r"^(?P<index>[01]), .*?, (?P<total>\d+) MiB, (?P<used>\d+) MiB, "
    r"(?P<free>\d+) MiB, (?P<temp>\d+),"
)


def vram_snapshot(path: pathlib.Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        match = GPU_LINE.match(line)
        if match:
            data = {key: int(value) for key, value in match.groupdict().items()}
            index = str(data.pop("index"))
            result[index] = data
    if set(result) != {"0", "1"}:
        raise ValueError(f"failed to parse both GPUs from {path}")
    return result


def validate_receipts(*directories: pathlib.Path) -> dict:
    summaries = []
    hash_checks = 0
    settlement_checks = 0
    failure_scans = 0
    for directory in directories:
        rc = int((directory / "runner.rc").read_text().strip())
        if rc != 0:
            raise ValueError(f"{directory}: runner rc {rc}")
        for path in directory.glob("*.summary.json"):
            summary = json.loads(path.read_text())
            if summary["n_error"] or summary["bos_garbage_count"]:
                raise ValueError(f"unclean receipt {path}")
            if summary.get("expected_sha256") is not None:
                hash_checks += 1
                if summary.get("expected_matches") != summary["n"]:
                    raise ValueError(f"known-output mismatch {path}")
            if "metrics_settled" in summary:
                settlement_checks += 1
                if summary["metrics_settled"] is not True:
                    raise ValueError(f"unsettled counters {path}")
            summaries.append(path)
        for path in directory.glob("*server.failures.txt"):
            failure_scans += 1
            if path.read_text().strip():
                raise ValueError(f"runtime failure signature {path}")
    c_driver = directories[-1] / "driver.log"
    overlap_assertions = sum(
        line.startswith("overlap_s=") for line in c_driver.read_text().splitlines()
    )
    if overlap_assertions != 15:
        raise ValueError(f"expected 15 overlap assertions, found {overlap_assertions}")
    return {
        "summary_files": len(summaries),
        "known_output_hash_checks": hash_checks,
        "settled_counter_checks": settlement_checks,
        "empty_runtime_failure_scans": failure_scans,
        "overlap_assertions": overlap_assertions,
        "all_clean": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    a = args.raw / "A"
    b = args.raw / "B"
    c = args.raw / "C"

    result = {
        "validation": validate_receipts(a, b, c),
        "step_alone": {
            "short_ttft_s": load_series(a, "step-short-r*.summary.json", "ttft_p50_s"),
            "ttft_4k_s": load_series(a, "step-4k-r*.summary.json", "ttft_p50_s"),
            "decode_c1_tok_s": load_series(
                a, "step-decode-c1-r*.summary.json", "decode_tok_s_median"
            ),
            "decode_c4_aggregate_tok_s": load_series(
                a, "step-decode-c4-r*.summary.json", "aggregate_output_tok_s"
            ),
        },
        "q27_alone": {
            "short_ttft_s": load_series(b, "q27-short-r*.summary.json", "ttft_p50_s"),
            "decode_c1_tok_s": load_series(
                b, "q27-decode-c1-r*.summary.json", "decode_tok_s_median"
            ),
            "decode_c1_spec": weighted_spec(b, "q27-decode-c1-r*.summary.json"),
            "decode_c4_aggregate_tok_s": load_series(
                b, "q27-decode-c4-r*.summary.json", "aggregate_output_tok_s"
            ),
            "decode_c4_spec": weighted_spec(b, "q27-decode-c4-r*.summary.json"),
        },
        "co_serve": {
            "step_idle_short_ttft_s": load_series(
                c, "step-short-idle-r*.summary.json", "ttft_p50_s"
            ),
            "step_active_short_ttft_s": load_series(
                c, "step-short-active-r*.summary.json", "ttft_p50_s"
            ),
            "step_idle_decode_c1_tok_s": load_series(
                c, "step-decode-idle-r*.summary.json", "decode_tok_s_median"
            ),
            "step_active_decode_c1_tok_s": load_series(
                c, "step-decode-active-r*.summary.json", "decode_tok_s_median"
            ),
            "q27_background_c2_aggregate_tok_s": load_series(
                c, "q27-bg-r*.summary.json", "aggregate_output_tok_s"
            ),
            "q27_background_c2_spec": weighted_spec(c, "q27-bg-r*.summary.json"),
            "q27_step_idle_ttft_s": load_series(
                c, "q27-step-idle-r*.summary.json", "ttft_p50_s"
            ),
            "q27_step_prime_ttft_s": load_series(
                c, "q27-under-step-prime-r*.summary.json", "ttft_p50_s"
            ),
            "q27_step_idle_decode_tok_s": load_series(
                c, "q27-step-idle-r*.summary.json", "decode_tok_s_median"
            ),
            "q27_step_prime_decode_tok_s": load_series(
                c, "q27-under-step-prime-r*.summary.json", "decode_tok_s_median"
            ),
            "step_prime_4k_ttft_s": load_series(
                c, "step-prime-r*.summary.json", "ttft_p50_s"
            ),
        },
        "vram": {
            "both_resident_before": vram_snapshot(c / "vram-both-resident-before.txt"),
            "both_resident_after": vram_snapshot(c / "vram-both-resident-after.txt"),
        },
        "q27_pool_final": {
            key: json.loads((c / "sanity-q27-final.summary.json").read_text())["metrics_after"].get(key)
            for key in (
                "cuda_driver_free_bytes",
                "cuda_pool_reserved_bytes",
                "cuda_pool_used_bytes",
                "spec_pool_entries",
                "spec_pool_hits",
                "spec_pool_misses",
                "spec_pool_evictions",
                "step_oom_parks",
            )
        },
        "thermal": {"A": thermal(a), "B": thermal(b), "C": thermal(c)},
    }

    step_alone = result["step_alone"]
    co = result["co_serve"]
    co["regressions_pct"] = {
        "step_resident_idle_short_p50_vs_alone": pct(
            co["step_idle_short_ttft_s"]["median"], step_alone["short_ttft_s"]["median"]
        ),
        "step_resident_idle_decode_vs_alone": pct(
            co["step_idle_decode_c1_tok_s"]["median"],
            step_alone["decode_c1_tok_s"]["median"],
        ),
        "step_active_short_p50_vs_alone": pct(
            co["step_active_short_ttft_s"]["median"], step_alone["short_ttft_s"]["median"]
        ),
        "step_active_short_p99_vs_alone": pct(
            co["step_active_short_ttft_s"]["p99_nearest_rank"],
            step_alone["short_ttft_s"]["p99_nearest_rank"],
        ),
        "step_active_short_p50_vs_resident_idle": pct(
            co["step_active_short_ttft_s"]["median"],
            co["step_idle_short_ttft_s"]["median"],
        ),
        "step_active_short_p99_vs_resident_idle": pct(
            co["step_active_short_ttft_s"]["p99_nearest_rank"],
            co["step_idle_short_ttft_s"]["p99_nearest_rank"],
        ),
        "step_active_decode_vs_alone": pct(
            co["step_active_decode_c1_tok_s"]["median"],
            step_alone["decode_c1_tok_s"]["median"],
        ),
        "step_active_decode_vs_resident_idle": pct(
            co["step_active_decode_c1_tok_s"]["median"],
            co["step_idle_decode_c1_tok_s"]["median"],
        ),
        "q27_prime_ttft_vs_step_idle": pct(
            co["q27_step_prime_ttft_s"]["median"], co["q27_step_idle_ttft_s"]["median"]
        ),
        "q27_prime_decode_vs_step_idle": pct(
            co["q27_step_prime_decode_tok_s"]["median"],
            co["q27_step_idle_decode_tok_s"]["median"],
        ),
        "step_prime_4k_ttft_vs_alone": pct(
            co["step_prime_4k_ttft_s"]["median"], step_alone["ttft_4k_s"]["median"]
        ),
        "q27_resident_idle_ttft_vs_alone": pct(
            co["q27_step_idle_ttft_s"]["median"],
            result["q27_alone"]["short_ttft_s"]["median"],
        ),
        "q27_resident_idle_decode_vs_alone": pct(
            co["q27_step_idle_decode_tok_s"]["median"],
            result["q27_alone"]["decode_c1_tok_s"]["median"],
        ),
    }
    q27_background = co["q27_background_c2_aggregate_tok_s"]["median"]
    million_tokens_day = q27_background * 86_400.0 / 1_000_000.0
    result["economy"] = {
        "continuous_c2_output_million_tokens_per_day": million_tokens_day,
        "gemini_3_1_flash_lite_output_usd_per_million": 1.50,
        "value_per_day_at_1_50_usd": million_tokens_day * 1.50,
        "gemini_3_5_flash_lite_ga_output_usd_per_million": 2.50,
        "value_per_day_at_2_50_usd": million_tokens_day * 2.50,
    }

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
