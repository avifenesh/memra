#!/usr/bin/env python3
"""Validate and reduce the box1 gscost raw JSONL after all scoring has finished."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


MAIN_LABEL = re.compile(r"^main-(q27|q35)-c(1|4|16|40)-r([1-5])-(repaired|eager)$")
MIXED_LABEL = re.compile(r"^mixed-(q27|q35)-c(4|16|40)-r([1-5])-(repaired|eager)$")
MAIN_WIDTHS = {"q27": (1, 4, 16, 40), "q35": (1, 4, 16, 40)}
MIXED_WIDTHS = {"q27": (4, 16), "q35": (4, 40)}
ARMS = ("repaired", "eager")
MAIN_ORDER = (("q27", 1), ("q35", 4), ("q27", 16), ("q35", 40),
              ("q35", 1), ("q27", 4), ("q35", 16), ("q27", 40))
MIXED_ORDER = (("q27", 4), ("q35", 40), ("q27", 16), ("q35", 4))
NEIGHBOUR_BOOKKEEPING_MEMORY_MIB = 1.0
NEIGHBOUR_IDLE_SM_CLOCK_MAX_MHZ = 200.0
NEIGHBOUR_BOOKKEEPING_MAX_RUN_SECONDS = 2.0
NEIGHBOUR_BOOKKEEPING_MAX_FRACTION = 0.005
THERMAL_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def rows(path: Path) -> list[dict[str, Any]]:
    parsed = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return parsed


def median(values: Iterable[float]) -> float:
    materialized = list(values)
    if len(materialized) != 5:
        raise ValueError(f"expected N=5, got {len(materialized)}: {materialized}")
    return float(statistics.median(materialized))


def expected_labels(prefix: str, order: tuple[tuple[str, int], ...]) -> list[str]:
    expected = []
    for rep in range(1, 6):
        multiplier = 3 if prefix == "main" else 1
        offset = (rep - 1) * multiplier % len(order)
        arms = ARMS if rep % 2 == 1 else tuple(reversed(ARMS))
        for point in range(len(order)):
            model, concurrency = order[(point + offset) % len(order)]
            expected.extend(
                f"{prefix}-{model}-c{concurrency}-r{rep}-{arm}" for arm in arms
            )
    return expected


def group_summary(
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]], metric: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (model, concurrency, arm), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: int(row["rep"]))
        values = [float(row[metric]) for row in selected]
        result.setdefault(model, {}).setdefault(str(concurrency), {})[arm] = {
            "N": 5,
            "samples_tok_s": values,
            "median_tok_s": median(values),
            "min_tok_s": min(values),
            "max_tok_s": max(values),
        }

    for model, widths in result.items():
        for concurrency, arms in widths.items():
            repaired = arms["repaired"]
            eager = arms["eager"]
            repaired_samples = repaired["samples_tok_s"]
            eager_samples = eager["samples_tok_s"]
            paired = [
                (r / e - 1.0) * 100.0
                for r, e in zip(repaired_samples, eager_samples, strict=True)
            ]
            ranges_overlap = not (
                repaired["max_tok_s"] < eager["min_tok_s"]
                or eager["max_tok_s"] < repaired["min_tok_s"]
            )
            arms["comparison"] = {
                "repaired_delta_vs_eager_pct":
                    (repaired["median_tok_s"] / eager["median_tok_s"] - 1.0) * 100.0,
                "repair_cost_pct":
                    (1.0 - repaired["median_tok_s"] / eager["median_tok_s"]) * 100.0,
                "paired_delta_pct": paired,
                "paired_delta_pct_median": float(statistics.median(paired)),
                "ranges_overlap": ranges_overlap,
                "verdict": "FLAT" if ranges_overlap else (
                    "REPAIR_LOSS" if repaired["median_tok_s"] < eager["median_tok_s"]
                    else "REPAIR_WIN"
                ),
            }
    return result


def reduce_main(path: Path) -> dict[str, Any]:
    parsed = rows(path)
    summaries = [row for row in parsed if row.get("kind") == "summary"]
    requests = [row for row in parsed if row.get("kind") == "request"]
    expected_points = sum(len(widths) for widths in MAIN_WIDTHS.values()) * len(ARMS) * 5
    if len(summaries) != expected_points:
        raise ValueError(f"main summaries: expected {expected_points}, got {len(summaries)}")
    actual_labels = [str(row.get("label")) for row in summaries]
    if actual_labels != expected_labels("main", MAIN_ORDER):
        raise ValueError("main point order/arm alternation drift")

    grouped_decode: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_total: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        match = MAIN_LABEL.fullmatch(str(row.get("label")))
        if match is None:
            raise ValueError(f"bad main label: {row.get('label')}")
        model, concurrency_raw, rep_raw, arm = match.groups()
        concurrency = int(concurrency_raw)
        rep = int(rep_raw)
        if int(row["concurrency"]) != concurrency:
            raise ValueError(f"concurrency drift in {row['label']}")
        if int(row["rep"]) != rep or row["target"] != model or row["policy_arm"] != arm:
            raise ValueError(f"label metadata drift in {row['label']}")
        if int(row["n_error"]) != 0 or int(row["n_ok"]) != concurrency:
            raise ValueError(f"request failure in {row['label']}: {row}")
        if int(row["completion_tokens_total"]) != concurrency * 512:
            raise ValueError(f"short completion in {row['label']}: {row}")
        if row["finish_reasons"] != ["length"]:
            raise ValueError(f"shape drift in {row['label']}: {row}")
        if len(row["prompt_tokens"]) != 1 or int(row["prompt_tokens"][0]) < 64:
            raise ValueError(f"prompt shape drift in {row['label']}: {row}")
        if row["cached_tokens"] != row["prompt_tokens"]:
            raise ValueError(f"cache-hit shape drift in {row['label']}: {row}")
        decorated = {**row, "model": model, "rep": rep, "arm": arm}
        grouped_decode[(model, concurrency, arm)].append(decorated)
        grouped_total[(model, concurrency, arm)].append(decorated)

    expected_requests = sum(sum(widths) for widths in MAIN_WIDTHS.values()) * len(ARMS) * 5
    if len(requests) != expected_requests:
        raise ValueError(f"main requests: expected {expected_requests}, got {len(requests)}")
    return {
        "metric": (
            "fully cached long-form chat prompt; aggregate generated tokens after each request's "
            "first visible token / shared decode window"
        ),
        "cache_hit_token_ratio": 1.0,
        "max_tokens_per_request": 512,
        "decode_window": group_summary(grouped_decode, "decode_window_tok_s"),
        "end_to_end": group_summary(grouped_total, "total_window_tok_s"),
        "receipt": {"points": len(summaries), "requests": len(requests), "errors": 0},
    }


def reduce_mixed(path: Path) -> dict[str, Any]:
    parsed = rows(path)
    cells = [row for row in parsed if row.get("kind") == "cell"]
    requests = [row for row in parsed if row.get("kind") == "request"]
    protocols = [row for row in parsed if row.get("kind") == "single_mixed_protocol"]
    expected_points = sum(len(widths) for widths in MIXED_WIDTHS.values()) * len(ARMS) * 5
    if len(cells) != expected_points or len(protocols) != expected_points:
        raise ValueError(
            f"mixed cells/protocols: expected {expected_points}, got {len(cells)}/{len(protocols)}"
        )
    actual_labels = [str(row.get("label")) for row in cells]
    if actual_labels != expected_labels("mixed", MIXED_ORDER):
        raise ValueError("mixed point order/arm alternation drift")

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        match = MIXED_LABEL.fullmatch(str(row.get("label")))
        if match is None:
            raise ValueError(f"bad mixed label: {row.get('label')}")
        model, concurrency_raw, rep_raw, arm = match.groups()
        concurrency = int(concurrency_raw)
        rep = int(rep_raw)
        if concurrency not in MIXED_WIDTHS[model]:
            raise ValueError(f"unexpected mixed width: {row['label']}")
        if int(row["rep"]) != rep or row["target"] != model or row["policy_arm"] != arm:
            raise ValueError(f"label metadata drift in {row['label']}")
        if not bool(row["clean"]) or row["integrity_failures"]:
            raise ValueError(f"unclean mixed cell {row['label']}: {row['integrity_failures']}")
        if int(row["requests_ok"]) != int(row["requests_n"]):
            raise ValueError(f"mixed request failure: {row['label']}")
        if not math.isclose(float(row["cache_hit_token_ratio"]), 0.9, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"mixed cache ratio drift: {row['label']}")
        if int(row["cached_tokens_in_drift"]) != 0 or int(row["prefix_cache_hit_tokens_drift"]) != 0:
            raise ValueError(f"mixed cached-token drift: {row['label']}")
        decorated = {**row, "model": model, "rep": rep, "arm": arm}
        grouped[(model, concurrency, arm)].append(decorated)

    expected_requests = 0
    for model, widths in MIXED_WIDTHS.items():
        for concurrency in widths:
            request_count = max(20, concurrency)
            request_count = ((request_count + 9) // 10) * 10
            expected_requests += request_count * len(ARMS) * 5
    if len(requests) != expected_requests:
        raise ValueError(f"mixed requests: expected {expected_requests}, got {len(requests)}")
    return {
        "metric": "completion tokens / full mixed-cell wall time, including hit TTFT and real misses",
        "prompt_tokens_per_request": 4860,
        "max_tokens_per_request": 60,
        "cache_hit_token_ratio": 0.9,
        "output_window": group_summary(grouped, "output_tok_s"),
        "receipt": {"points": len(cells), "requests": len(requests), "errors": 0},
    }


def is_idle_bookkeeping_sample(sample: dict[str, Any]) -> bool:
    return (
        sample["memory_used_mib"] == NEIGHBOUR_BOOKKEEPING_MEMORY_MIB
        and sample["utilization_pct"] == 0
        and sample["pstate"] == "P8"
        and sample["sm_clock_mhz"] <= NEIGHBOUR_IDLE_SM_CLOCK_MAX_MHZ
    )


def thermal_evidence(samples: list[dict[str, Any]]) -> dict[str, Any]:
    selected = samples if len(samples) <= 8 else samples[:4] + samples[-4:]
    return {
        "samples": [
            {
                "timestamp": sample["timestamp"],
                "pstate": sample["pstate"],
                "sm_clock_mhz": sample["sm_clock_mhz"],
                "memory_used_mib": sample["memory_used_mib"],
                "utilization_pct": sample["utilization_pct"],
            }
            for sample in selected
        ],
        "omitted_samples": len(samples) - len(selected),
    }


def reduce_thermal(path: Path) -> dict[str, Any]:
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8", errors="replace") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 10:
                raise ValueError(f"{path}:{line_number}: short thermal row: {row}")
            try:
                index = int(row[1].strip())
                timestamp = row[0].strip()
                by_gpu[index].append(
                    {
                        "timestamp": timestamp,
                        "timestamp_s": datetime.strptime(
                            timestamp, THERMAL_TIMESTAMP_FORMAT
                        ).timestamp(),
                        "pstate": row[2].strip(),
                        "temperature_c": float(row[3].strip()),
                        "power_w": float(row[4].strip()),
                        "power_limit_w": float(row[5].strip()),
                        "sm_clock_mhz": float(row[6].strip()),
                        "memory_used_mib": float(row[7].strip()),
                        "utilization_pct": float(row[9].strip()),
                    }
                )
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed thermal row: {row}"
                ) from error
    if set(by_gpu) != {0, 1}:
        raise ValueError(f"thermal samples missing GPU: {sorted(by_gpu)}")
    if len(by_gpu[0]) != len(by_gpu[1]):
        raise ValueError(
            f"thermal sample-count mismatch: GPU0={len(by_gpu[0])}, GPU1={len(by_gpu[1])}"
        )

    result: dict[str, Any] = {}
    active_by_gpu: dict[int, list[dict[str, Any]]] = {}
    bookkeeping_by_gpu: dict[int, list[dict[str, Any]]] = {}
    for index, samples in sorted(by_gpu.items()):
        for previous, current in zip(samples, samples[1:]):
            if current["timestamp_s"] < previous["timestamp_s"]:
                raise ValueError(
                    f"thermal timestamps moved backwards on GPU{index}: "
                    f"{previous['timestamp']} -> {current['timestamp']}"
                )
        if any(sample["memory_used_mib"] < 0 for sample in samples):
            raise ValueError(f"negative memory telemetry on GPU{index}")

        # nvidia-smi reports FB memory in whole MiB. Attempt 3 contains a 1 MiB
        # quantum on both devices in the same samples, including while GPU0 is
        # at P0, though the neighbour stays at P8 / 180 MHz / 0%. One quantum
        # is therefore not tenant evidence by itself. This exemption is narrow:
        # any larger allocation or any independent work signal is active, and
        # the exact idle signature must also remain brief and rare below. Do not
        # widen these limits without new raw evidence; the shared-PIX doctrine
        # remains fail-closed.
        bookkeeping = [sample for sample in samples if is_idle_bookkeeping_sample(sample)]
        active = [
            sample for sample in samples
            if (
                sample["memory_used_mib"] > 0
                and not is_idle_bookkeeping_sample(sample)
            )
            or sample["utilization_pct"] > 0
            or sample["pstate"] != "P8"
            or sample["sm_clock_mhz"] > NEIGHBOUR_IDLE_SM_CLOCK_MAX_MHZ
        ]
        active_by_gpu[index] = active
        bookkeeping_by_gpu[index] = bookkeeping

        max_run_samples = 0
        max_run_seconds = 0.0
        run_samples = 0
        run_start_s = 0.0
        for sample in samples:
            if is_idle_bookkeeping_sample(sample):
                if run_samples == 0:
                    run_start_s = sample["timestamp_s"]
                run_samples += 1
                max_run_samples = max(max_run_samples, run_samples)
                max_run_seconds = max(
                    max_run_seconds, sample["timestamp_s"] - run_start_s
                )
            else:
                run_samples = 0

        result[f"gpu{index}"] = {
            "samples": len(samples),
            "active_samples": len(active),
            "one_mib_zero_util_samples": len(bookkeeping),
            "one_mib_zero_util_fraction": len(bookkeeping) / len(samples),
            "one_mib_zero_util_max_run_samples": max_run_samples,
            "one_mib_zero_util_max_run_seconds": max_run_seconds,
            "one_mib_zero_util_first_timestamp": (
                bookkeeping[0]["timestamp"] if bookkeeping else None
            ),
            "one_mib_zero_util_last_timestamp": (
                bookkeeping[-1]["timestamp"] if bookkeeping else None
            ),
            "pstate_values": sorted({sample["pstate"] for sample in samples}),
            "temperature_c_min": min(sample["temperature_c"] for sample in samples),
            "temperature_c_max": max(sample["temperature_c"] for sample in samples),
            "active_sm_clock_mhz_min": min(
                (sample["sm_clock_mhz"] for sample in active), default=None
            ),
            "active_sm_clock_mhz_max": max(
                (sample["sm_clock_mhz"] for sample in active), default=None
            ),
            "power_limit_w_values": sorted({sample["power_limit_w"] for sample in samples}),
            "memory_used_mib_max": max(sample["memory_used_mib"] for sample in samples),
            "utilization_pct_max": max(sample["utilization_pct"] for sample in samples),
        }

    gpu1 = result["gpu1"]
    gpu1["idle_guard"] = {
        "bookkeeping_memory_mib_exact": NEIGHBOUR_BOOKKEEPING_MEMORY_MIB,
        "bookkeeping_max_fraction": NEIGHBOUR_BOOKKEEPING_MAX_FRACTION,
        "bookkeeping_max_run_seconds": NEIGHBOUR_BOOKKEEPING_MAX_RUN_SECONDS,
        "idle_pstate": "P8",
        "idle_sm_clock_max_mhz": NEIGHBOUR_IDLE_SM_CLOCK_MAX_MHZ,
        "idle_utilization_pct": 0,
    }
    refusal_reasons = []
    if gpu1["active_samples"] != 0:
        refusal_reasons.append(
            "substantive allocation or an independent pstate/clock/utilization signal"
        )
    if gpu1["one_mib_zero_util_max_run_seconds"] > NEIGHBOUR_BOOKKEEPING_MAX_RUN_SECONDS:
        refusal_reasons.append(
            "1 MiB idle signature exceeded the maximum continuous duration"
        )
    if gpu1["one_mib_zero_util_fraction"] > NEIGHBOUR_BOOKKEEPING_MAX_FRACTION:
        refusal_reasons.append("1 MiB idle signature exceeded the campaign noise budget")
    if refusal_reasons:
        suspicious = active_by_gpu[1] or bookkeeping_by_gpu[1]
        evidence = {
            "reasons": refusal_reasons,
            "summary": gpu1,
            "evidence": thermal_evidence(suspicious),
        }
        raise ValueError(
            f"neighbour GPU was not idle: {json.dumps(evidence, sort_keys=True)}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "schema": "memra.gscost.summary.v1",
        "source_commit": args.source,
        "binary_sha256": args.binary_sha256,
        "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition; GPU0 scored, GPU1 idle",
        "protocol": (
            "one /tmp/memra-gpu.lock hold; fresh server per point; N=5; odd reps "
            "REPAIRED-first, even reps EAGER-first; no clock changes or artificial cooldown"
        ),
        "arms": {
            "repaired": "MEMRA_SERVE_B1FAST and MEMRA_SERVE_GS unset",
            "eager": "MEMRA_SERVE_B1FAST=1 and MEMRA_SERVE_GS=1",
        },
        "main": reduce_main(args.main),
        "mixed": reduce_mixed(args.mixed),
        "thermal": reduce_thermal(args.thermal),
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
