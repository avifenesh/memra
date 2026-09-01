#!/usr/bin/env python3
"""Validate and reduce the local RTX 5090 eager-arm campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FULL_LABEL = re.compile(r"^full-q27-c(1|4|16)-r([1-5])-(repaired|eager)$")
MIXED_LABEL = re.compile(r"^mixed-q27-c4-r([1-5])-(repaired|eager)$")
WIDTHS = (1, 4, 16)
ARMS = ("repaired", "eager")


def read_rows(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return parsed


def median5(values: Iterable[float]) -> float:
    materialized = list(values)
    if len(materialized) != 5:
        raise ValueError(f"expected N=5, got {len(materialized)}: {materialized}")
    return float(statistics.median(materialized))


def summarize(
    grouped: dict[tuple[int, str], list[dict[str, Any]]], metric: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (concurrency, arm), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: int(row["rep"]))
        samples = [float(row[metric]) for row in selected]
        result.setdefault(str(concurrency), {})[arm] = {
            "N": 5,
            "samples_tok_s": samples,
            "median_tok_s": median5(samples),
            "min_tok_s": min(samples),
            "max_tok_s": max(samples),
        }

    for concurrency, arms in result.items():
        repaired = arms["repaired"]
        eager = arms["eager"]
        paired = [
            (eager_value / repaired_value - 1.0) * 100.0
            for repaired_value, eager_value in zip(
                repaired["samples_tok_s"], eager["samples_tok_s"], strict=True
            )
        ]
        ranges_overlap = not (
            repaired["max_tok_s"] < eager["min_tok_s"]
            or eager["max_tok_s"] < repaired["min_tok_s"]
        )
        median_delta = (
            eager["median_tok_s"] / repaired["median_tok_s"] - 1.0
        ) * 100.0
        arms["comparison"] = {
            "eager_delta_vs_repaired_pct": median_delta,
            "paired_delta_pct": paired,
            "paired_delta_pct_median": float(statistics.median(paired)),
            "ranges_overlap": ranges_overlap,
            "verdict": (
                "FLAT"
                if ranges_overlap
                else "EAGER_WIN"
                if median_delta > 0
                else "REPAIRED_WIN"
            ),
        }
    return result


def hashes_by_arm(requests: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in requests:
        match = FULL_LABEL.fullmatch(str(row.get("label"))) or MIXED_LABEL.fullmatch(
            str(row.get("label"))
        )
        if match is None or not row.get("text_sha256"):
            raise ValueError(f"request has bad label/hash: {row}")
        arm = str(row.get("policy_arm") or match.groups()[-1])
        result[arm].add(str(row["text_sha256"]))
    return {arm: sorted(values) for arm, values in sorted(result.items())}


def hashes_by_arm_and_role(requests: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in requests:
        arm = str(row["policy_arm"])
        role = str(row["cache_role"])
        result[arm][role].add(str(row["text_sha256"]))
    return {
        arm: {role: sorted(values) for role, values in sorted(roles.items())}
        for arm, roles in sorted(result.items())
    }


def reduce_full(path: Path) -> dict[str, Any]:
    parsed = read_rows(path)
    summaries = [row for row in parsed if row.get("kind") == "summary"]
    requests = [row for row in parsed if row.get("kind") == "request"]
    expected_points = len(WIDTHS) * len(ARMS) * 5
    if len(summaries) != expected_points:
        raise ValueError(f"full summaries: expected {expected_points}, got {len(summaries)}")

    grouped_decode: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_total: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        match = FULL_LABEL.fullmatch(str(row.get("label")))
        if match is None:
            raise ValueError(f"bad full label: {row.get('label')}")
        concurrency, rep, arm = int(match[1]), int(match[2]), match[3]
        if int(row["concurrency"]) != concurrency:
            raise ValueError(f"concurrency drift in {row['label']}")
        if int(row["n_error"]) != 0 or int(row["n_ok"]) != concurrency:
            raise ValueError(f"request failure in {row['label']}: {row}")
        if int(row["completion_tokens_total"]) != concurrency * 512:
            raise ValueError(f"short completion in {row['label']}: {row}")
        if row["finish_reasons"] != ["length"]:
            raise ValueError(f"finish drift in {row['label']}: {row['finish_reasons']}")
        if row["cached_tokens"] != row["prompt_tokens"]:
            raise ValueError(f"cache-hit drift in {row['label']}")
        decorated = {**row, "rep": rep, "arm": arm}
        grouped_decode[(concurrency, arm)].append(decorated)
        grouped_total[(concurrency, arm)].append(decorated)

    expected_requests = sum(WIDTHS) * len(ARMS) * 5
    if len(requests) != expected_requests:
        raise ValueError(f"full requests: expected {expected_requests}, got {len(requests)}")
    exactness: dict[str, Any] = {}
    valid_widths: set[int] = set()
    for concurrency in WIDTHS:
        selected = [
            row
            for row in requests
            if FULL_LABEL.fullmatch(str(row.get("label")))
            and int(FULL_LABEL.fullmatch(str(row["label"]))[1]) == concurrency
        ]
        arm_hashes = hashes_by_arm(selected)
        all_hashes = {value for values in arm_hashes.values() for value in values}
        valid = len(all_hashes) == 1
        mismatch_reps: list[int] = []
        for rep in range(1, 6):
            rep_rows = [
                row
                for row in selected
                if int(FULL_LABEL.fullmatch(str(row["label"]))[2]) == rep
            ]
            rep_hashes = {
                str(row["text_sha256"])
                for row in rep_rows
                if row.get("text_sha256")
            }
            if len(rep_hashes) != 1:
                mismatch_reps.append(rep)

        repaired_hashes = arm_hashes.get("repaired", [])
        divergent_requests_by_arm: dict[str, int] = {}
        if len(repaired_hashes) == 1:
            repaired_hash = repaired_hashes[0]
            for arm in ARMS:
                divergent_requests_by_arm[arm] = sum(
                    1
                    for row in selected
                    if FULL_LABEL.fullmatch(str(row["label"]))[3] == arm
                    and str(row["text_sha256"]) != repaired_hash
                )
        if valid:
            valid_widths.add(concurrency)
        exactness[str(concurrency)] = {
            "requests": len(selected),
            "requests_by_arm": {arm: len(selected) // len(ARMS) for arm in ARMS},
            "hashes_by_arm": arm_hashes,
            "mismatch_reps": mismatch_reps,
            "requests_not_matching_stable_repaired_hash_by_arm": divergent_requests_by_arm,
            "verdict": (
                "BYTE IDENTICAL across arms and reps" if valid else "BYTE MISMATCH; throughput withheld"
            ),
        }

    filtered_decode = {
        key: value for key, value in grouped_decode.items() if key[0] in valid_widths
    }
    filtered_total = {
        key: value for key, value in grouped_total.items() if key[0] in valid_widths
    }

    return {
        "shape": "fully restored long-form chat, 512 completion tokens per request",
        "primary_metric": "aggregate tokens after first visible token / shared decode window",
        "decode_window": summarize(filtered_decode, "decode_window_tok_s"),
        "end_to_end": summarize(filtered_total, "total_window_tok_s"),
        "throughput_withheld_concurrency": [
            concurrency for concurrency in WIDTHS if concurrency not in valid_widths
        ],
        "exactness": exactness,
        "receipt": {"points": len(summaries), "requests": len(requests), "errors": 0},
    }


def reduce_mixed(path: Path) -> dict[str, Any]:
    parsed = read_rows(path)
    cells = [row for row in parsed if row.get("kind") == "cell"]
    requests = [row for row in parsed if row.get("kind") == "request"]
    seeds = [row for row in parsed if row.get("kind") == "seed"]
    protocols = [row for row in parsed if row.get("kind") == "single_mixed_protocol"]
    expected_points = len(ARMS) * 5
    if len(cells) != expected_points or len(protocols) != expected_points:
        raise ValueError(
            f"mixed cells/protocols: expected {expected_points}, got {len(cells)}/{len(protocols)}"
        )

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        match = MIXED_LABEL.fullmatch(str(row.get("label")))
        if match is None:
            raise ValueError(f"bad mixed label: {row.get('label')}")
        rep, arm = int(match[1]), match[2]
        if not bool(row["clean"]) or row["integrity_failures"]:
            raise ValueError(f"unclean mixed cell {row['label']}: {row['integrity_failures']}")
        if int(row["requests_ok"]) != 20 or int(row["requests_n"]) != 20:
            raise ValueError(f"mixed request count drift in {row['label']}")
        if not math.isclose(float(row["cache_hit_token_ratio"]), 0.9, abs_tol=1e-12):
            raise ValueError(f"mixed cache ratio drift in {row['label']}")
        if int(row["cached_tokens_in_drift"]) or int(row["prefix_cache_hit_tokens_drift"]):
            raise ValueError(f"mixed cached-token accounting drift in {row['label']}")
        grouped[(4, arm)].append({**row, "rep": rep, "arm": arm})

    if len(requests) != 200:
        raise ValueError(f"mixed requests: expected 200, got {len(requests)}")
    if len(seeds) != 80:
        raise ValueError(f"mixed seeds: expected 80, got {len(seeds)}")
    if any(row.get("ok") is not True or not row.get("text_sha256") for row in seeds):
        raise ValueError("mixed seed failure")
    for row in requests:
        if (
            row.get("ok") is not True
            or int(row.get("completion_tokens") or 0) != 60
            or row.get("finish_reason") != "length"
        ):
            raise ValueError(f"mixed exact-token failure: {row}")

    arm_hashes = hashes_by_arm(requests)
    all_hashes = {value for values in arm_hashes.values() for value in values}
    seed_hashes: dict[str, set[str]] = defaultdict(set)
    for row in seeds:
        seed_hashes[str(row["policy_arm"])].add(str(row["text_sha256"]))

    restored_hit_golden: dict[str, Any] = {}
    mismatch_cells_by_arm: dict[str, list[int]] = {}
    for arm in ARMS:
        hits = [
            row
            for row in requests
            if row["policy_arm"] == arm and row["cache_role"] == "hit"
        ]
        if any(not row.get("golden_sha256") for row in hits):
            raise ValueError(f"missing restored-hit golden hash for {arm}")
        mismatches = [row for row in hits if row.get("golden_ok") is not True]
        reported_mismatches = sum(
            int(row["golden_mismatches_observed"])
            for row in cells
            if row["policy_arm"] == arm
        )
        if reported_mismatches != len(mismatches):
            raise ValueError(
                f"mixed golden mismatch accounting drift for {arm}: "
                f"cells={reported_mismatches}, requests={len(mismatches)}"
            )
        restored_hit_golden[arm] = {
            "requests": len(hits),
            "mismatches": len(mismatches),
        }
        mismatch_cells_by_arm[arm] = sorted(
            int(MIXED_LABEL.fullmatch(str(row["label"]))[1])
            for row in cells
            if row["policy_arm"] == arm
            and int(row["golden_mismatches_observed"]) > 0
        )

    pair_mismatch_reps: list[int] = []
    for rep in range(1, 6):
        rep_hashes = {
            str(row["text_sha256"])
            for row in requests
            if int(MIXED_LABEL.fullmatch(str(row["label"]))[1]) == rep
        }
        if len(rep_hashes) != 1:
            pair_mismatch_reps.append(rep)

    golden_mismatches = sum(
        int(details["mismatches"]) for details in restored_hit_golden.values()
    )
    valid = len(all_hashes) == 1 and golden_mismatches == 0
    eager_golden = restored_hit_golden["eager"]
    return {
        "shape": "frozen 4860+60 prompt; 90% full hits / 10% real cold misses at c=4",
        "primary_metric": "completion tokens / full mixed-cell wall time",
        "output_window": summarize(grouped, "output_tok_s") if valid else {},
        "throughput_withheld": not valid,
        "exactness": {
            "requests": len(requests),
            "requests_by_arm": {arm: 100 for arm in ARMS},
            "hashes_by_arm": arm_hashes,
            "hashes_by_arm_and_cache_role": hashes_by_arm_and_role(requests),
            "seed_hashes_by_arm": {
                arm: sorted(values) for arm, values in sorted(seed_hashes.items())
            },
            "pair_mismatch_reps": pair_mismatch_reps,
            "restored_hit_golden": restored_hit_golden,
            "restored_hit_golden_mismatch_cells_by_arm": mismatch_cells_by_arm,
            "verdict": (
                "BYTE IDENTICAL across hit/miss, arms, and reps"
                if valid
                else (
                    "BYTE MISMATCH; EAGER restored-hit golden mismatches "
                    f"{eager_golden['mismatches']}/{eager_golden['requests']}; "
                    "throughput withheld"
                )
            ),
        },
        "receipt": {"points": len(cells), "requests": len(requests), "errors": 0},
    }


def reduce_thermal(path: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    power_limit_raw_values: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as source:
        for row in csv.reader(source):
            if len(row) < 10:
                continue
            try:
                temperature_c = float(row[3].strip())
                power_w = float(row[4].strip())
                sm_clock_mhz = float(row[6].strip())
                memory_used_mib = float(row[7].strip())
                utilization_pct = float(row[9].strip())
            except ValueError:
                continue
            power_limit_raw = row[5].strip()
            power_limit_raw_values.add(power_limit_raw)
            try:
                power_limit_w: float | None = float(power_limit_raw)
            except ValueError:
                power_limit_w = None
            samples.append(
                {
                    "temperature_c": temperature_c,
                    "power_w": power_w,
                    "power_limit_w": power_limit_w,
                    "sm_clock_mhz": sm_clock_mhz,
                    "memory_used_mib": memory_used_mib,
                    "utilization_pct": utilization_pct,
                }
            )
    if not samples:
        raise ValueError("no GPU telemetry samples")
    active = [sample for sample in samples if sample["memory_used_mib"] > 1000]
    if not active:
        raise ValueError("no active GPU telemetry samples")
    busy = [sample for sample in active if sample["utilization_pct"] > 0]
    if not busy:
        raise ValueError("no busy GPU telemetry samples")
    clock_max = max(sample["sm_clock_mhz"] for sample in active)
    if clock_max > 1200:
        raise ValueError(f"owner clock cap drift: observed active SM clock {clock_max} MHz")
    return {
        "samples": len(samples),
        "active_samples": len(active),
        "busy_samples": len(busy),
        "temperature_c_min": min(sample["temperature_c"] for sample in samples),
        "temperature_c_max": max(sample["temperature_c"] for sample in samples),
        "active_sm_clock_mhz_min": min(sample["sm_clock_mhz"] for sample in active),
        "active_sm_clock_mhz_max": clock_max,
        "active_power_w_min": min(sample["power_w"] for sample in active),
        "active_power_w_max": max(sample["power_w"] for sample in active),
        "busy_sm_clock_mhz_min": min(sample["sm_clock_mhz"] for sample in busy),
        "busy_sm_clock_mhz_max": max(sample["sm_clock_mhz"] for sample in busy),
        "busy_power_w_min": min(sample["power_w"] for sample in busy),
        "busy_power_w_max": max(sample["power_w"] for sample in busy),
        "power_limit_w_values": sorted(
            {
                sample["power_limit_w"]
                for sample in samples
                if sample["power_limit_w"] is not None
            }
        ),
        "power_limit_raw_values": sorted(power_limit_raw_values),
        "memory_used_mib_max": max(sample["memory_used_mib"] for sample in samples),
        "utilization_pct_max": max(sample["utilization_pct"] for sample in samples),
        "clock_cap": (
            "owner-declared 210-1200 MHz; no clock-changing command run; "
            "telemetry ceiling verified at 1200 MHz"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "schema": "memra.armrig5090.summary.v1",
        "binary_source_commit": args.source,
        "binary_sha256": args.binary_sha256,
        "rig": "local NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB, sm_120 / 82 SM",
        "protocol": (
            "one /tmp/memra-5090.lock hold; fresh server per point; N=5; odd reps "
            "REPAIRED-first, even reps EAGER-first; no clock changes or artificial cooldown"
        ),
        "arms": {
            "repaired": "MEMRA_SERVE_B1FAST and MEMRA_SERVE_GS unset",
            "eager": "MEMRA_SERVE_B1FAST=1 and MEMRA_SERVE_GS=1",
        },
        "full_hit": reduce_full(args.full),
        "mixed90": reduce_mixed(args.mixed),
        "thermal": reduce_thermal(args.thermal),
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
