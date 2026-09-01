#!/usr/bin/env python3
"""Reduce the sealed dual-model sellgate receipts into one auditable summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


BASE_LEVELS = [1, 2, 4, 8]
SOLD_CAP = 4
TTFT_BAR_MS = 2_000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError(f"missing sealed manifest: {manifest}")
    checked = 0
    failures = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.lstrip("*").removeprefix("./")
        actual = sha256_file(path) if path.is_file() else None
        checked += 1
        if actual != expected:
            failures.append({"path": str(path), "expected": expected, "actual": actual})
    if failures:
        raise ValueError(f"manifest verification failed: {failures}")
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "files_checked": checked,
        "verified": True,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p75_ms": nearest_rank(values, 0.75),
        "p90_ms": nearest_rank(values, 0.90),
        "p95_ms": nearest_rank(values, 0.95),
        "p99_ms": nearest_rank(values, 0.99),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def summarize_cell_group(
    target: str,
    arm: str,
    concurrency: int,
    requests: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_requests = [
        row
        for row in requests
        if row["target"] == target
        and row["arm"] == arm
        and int(row["concurrency"]) == concurrency
    ]
    selected_cells = [
        row
        for row in cells
        if row["target"] == target
        and row["arm"] == arm
        and int(row["concurrency"]) == concurrency
    ]
    hit_rows = [row for row in selected_requests if row["cache_role"] == "hit"]
    miss_rows = [row for row in selected_requests if row["cache_role"] == "miss"]

    def times(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    def output_hash_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(
            str(row["text_sha256"])
            for row in rows
            if row.get("text_sha256") is not None
        ).items()))

    counter_totals = {
        key: sum(int(cell["counter_deltas"][key]) for cell in selected_cells)
        for key in (
            "admitted",
            "completed",
            "tokens_out",
            "prompt_tokens_in",
            "cached_tokens_in",
            "prefix_cache_hits",
            "prefix_cache_misses",
            "prefix_cache_evictions",
            "prefix_cache_hit_tokens",
            "admission_session_defers",
            "admission_vram_defers",
            "step_oom_parks",
        )
    }
    prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in selected_requests)
    cached_total = sum(int(row.get("cached_tokens") or 0) for row in selected_requests)
    return {
        "target": target,
        "arm": arm,
        "concurrency": concurrency,
        "n_cells": len(selected_cells),
        "n_requests": len(selected_requests),
        "all_clean": bool(selected_cells) and all(bool(cell["clean"]) for cell in selected_cells),
        "requests_ok": sum(bool(row.get("ok")) for row in selected_requests),
        "early_or_short_completions": sum(
            int(row.get("completion_tokens") or 0) != 60 for row in selected_requests
        ),
        "cache_hit_token_ratio": cached_total / prompt_total if prompt_total else None,
        "prompt_tokens": prompt_total,
        "cached_tokens_from_usage": cached_total,
        "counter_totals": counter_totals,
        "cached_tokens_in_drift": counter_totals["cached_tokens_in"] - cached_total,
        "prefix_cache_hit_tokens_drift": (
            counter_totals["prefix_cache_hit_tokens"] - cached_total
        ),
        "requests_per_s_median": statistics.median(
            float(cell["requests_per_s"]) for cell in selected_cells
        )
        if selected_cells
        else None,
        "output_tok_s_median": statistics.median(
            float(cell["output_tok_s"]) for cell in selected_cells
        )
        if selected_cells
        else None,
        "billed_prompt_tok_s_median": statistics.median(
            float(cell["billed_prompt_tok_s"]) for cell in selected_cells
        )
        if selected_cells
        else None,
        "computed_prompt_tok_s_median": statistics.median(
            float(cell["computed_prompt_tok_s"]) for cell in selected_cells
        )
        if selected_cells
        else None,
        "ttft_all": distribution(times(selected_requests, "ttft_ms")),
        "ttft_hit": distribution(times(hit_rows, "ttft_ms")),
        "ttft_miss": distribution(times(miss_rows, "ttft_ms")),
        "latency_all": distribution(times(selected_requests, "latency_ms")),
        "latency_hit": distribution(times(hit_rows, "latency_ms")),
        "latency_miss": distribution(times(miss_rows, "latency_ms")),
        "inter_token_all": distribution(times(selected_requests, "inter_token_ms")),
        "inter_token_hit": distribution(times(hit_rows, "inter_token_ms")),
        "inter_token_miss": distribution(times(miss_rows, "inter_token_ms")),
        "golden_mismatches_observed": sum(
            not bool(row.get("golden_ok")) for row in selected_requests
        ),
        "golden_required_failures": sum(
            bool(row.get("golden_required")) and not bool(row.get("golden_ok"))
            for row in selected_requests
        ),
        "output_sha256_counts": {
            "all": output_hash_counts(selected_requests),
            "hit": output_hash_counts(hit_rows),
            "miss": output_hash_counts(miss_rows),
        },
        "prefix_cache_bytes_peak_cell_end": max(
            (int(cell["prefix_cache_bytes_after"]) for cell in selected_cells),
            default=0,
        ),
    }


def parse_sha256s(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        result[name] = digest
    return result


def correctness_summary(gates: Path) -> dict[str, dict[str, Any]]:
    kernel = {}
    for gpu in (0, 1):
        text = (gates / f"kernel-gpu{gpu}.log").read_text(encoding="utf-8")
        kernel[str(gpu)] = {
            "all_green": "ALL GREEN" in text,
            "failure_marker": "MISMATCH" in text or "FAIL" in text,
        }
    targets = {}
    for target in ("q27", "q35"):
        gen = (gates / f"run-gen-{target}.log").read_text(encoding="utf-8")
        spec = (gates / f"run-spec-{target}.log").read_text(encoding="utf-8")
        targets[target] = {
            "kernel_all_green_both_gpus": all(
                row["all_green"] and not row["failure_marker"] for row in kernel.values()
            ),
            "run_gen_argmax_match": "argmax=" in gen
            and "MATCH" in gen
            and "MISMATCH" not in gen,
            "run_spec_k1_k8_pass_count": spec.count("self-consistency: PASS"),
            "run_spec_overall_pass": "=== SELF-CONSISTENCY PASS ===" in spec
            and "SELF-CONSISTENCY FAIL" not in spec,
        }
        targets[target]["pass"] = bool(
            targets[target]["kernel_all_green_both_gpus"]
            and targets[target]["run_gen_argmax_match"]
            and targets[target]["run_spec_k1_k8_pass_count"] == 8
            and targets[target]["run_spec_overall_pass"]
        )
    return {"kernel": kernel, "targets": targets}


def exactness_summary(campaign: Path, target: str) -> dict[str, Any]:
    rows = read_jsonl(campaign / "exactness" / f"{target}.jsonl")
    summary = [row for row in rows if row.get("kind") == "summary"]
    if len(summary) != 1:
        raise ValueError(f"{target}: expected exactly one cache-exactness summary")
    return summary[0]


def thermal_summary(path: Path) -> dict[str, Any]:
    by_gpu: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 11:
            continue
        try:
            gpu = fields[1]
            temperature = float(fields[3])
            power = float(fields[4])
            memory = float(fields[8])
            utilization = float(fields[10])
        except ValueError:
            continue
        row = by_gpu.setdefault(
            gpu,
            {
                "max_temperature_c": temperature,
                "max_power_w": power,
                "max_memory_used_mib": memory,
                "max_utilization_percent": utilization,
            },
        )
        row["max_temperature_c"] = max(row["max_temperature_c"], temperature)
        row["max_power_w"] = max(row["max_power_w"], power)
        row["max_memory_used_mib"] = max(row["max_memory_used_mib"], memory)
        row["max_utilization_percent"] = max(row["max_utilization_percent"], utilization)
    return by_gpu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    if not (args.campaign / "campaign.ok").is_file():
        raise ValueError("campaign does not carry the completion sentinel")
    if not (args.gates / "gates.ok").is_file():
        raise ValueError("correctness receipt does not carry the completion sentinel")

    campaign_manifest = verify_manifest(args.campaign)
    gates_manifest = verify_manifest(args.gates)
    rows = read_jsonl(args.campaign / "replay.jsonl")
    protocol = [row for row in rows if row.get("kind") == "protocol"]
    final = [row for row in rows if row.get("kind") == "summary"]
    if len(protocol) != 1 or len(final) != 1:
        raise ValueError("replay must carry exactly one protocol and final summary")
    protocol = protocol[0]
    final = final[0]
    requests = [row for row in rows if row.get("kind") == "request"]
    cells = [row for row in rows if row.get("kind") == "cell"]
    levels = [int(value) for value in final["levels_run"]]
    repetitions = int(protocol["workload"]["repetitions"])
    if levels[: len(BASE_LEVELS)] != BASE_LEVELS:
        raise ValueError(f"replay omitted or reordered required base levels: {levels}")
    expected_cells = len(levels) * 2 * 2 * repetitions
    if len(cells) != expected_cells:
        raise ValueError(f"replay has {len(cells)} cells, expected {expected_cells}")
    correctness = correctness_summary(args.gates)

    groups = {
        target: {
            arm: {
                str(level): summarize_cell_group(target, arm, level, requests, cells)
                for level in levels
            }
            for arm in ("cold", "mixed90")
        }
        for target in ("q27", "q35")
    }

    targets = {}
    for target in ("q27", "q35"):
        mixed = groups[target]["mixed90"]
        capacity_width = SOLD_CAP
        capacity_path = []
        previous = SOLD_CAP
        for level in [value for value in levels if value > SOLD_CAP]:
            prior = mixed[str(previous)]
            current = mixed[str(level)]
            rose = bool(
                prior["all_clean"]
                and current["all_clean"]
                and float(current["output_tok_s_median"])
                > float(prior["output_tok_s_median"])
            )
            capacity_path.append(
                {
                    "from": previous,
                    "to": level,
                    "from_output_tok_s": prior["output_tok_s_median"],
                    "to_output_tok_s": current["output_tok_s_median"],
                    "clean_rise": rose,
                }
            )
            if not rose:
                break
            capacity_width = level
            previous = level
        headroom_percent = (capacity_width / SOLD_CAP - 1.0) * 100.0
        c4 = mixed[str(SOLD_CAP)]
        c4_cold = groups[target]["cold"][str(SOLD_CAP)]
        c4_cold_hashes = set(c4_cold["output_sha256_counts"]["all"])
        c4_hit_hashes = set(c4["output_sha256_counts"]["hit"])
        exactness = exactness_summary(args.campaign, target)
        criteria = {
            "standard_correctness": correctness["targets"][target]["pass"],
            "serial_cache_exactness": exactness["verdict"] == "PASS",
            "all_required_base_cells_clean": bool(final["target_base_clean"][target]),
            "c4_mixed_cell_integrity": c4["all_clean"],
            "c4_cache_hit_ttft_p95_lt_2s": float(c4["ttft_hit"]["p95_ms"])
            < TTFT_BAR_MS,
            "c4_all_traffic_ttft_p50_lt_2s": float(c4["ttft_all"]["p50_ms"])
            < TTFT_BAR_MS,
            "c4_cached_token_accounting_zero_drift": (
                c4["cached_tokens_in_drift"] == 0
                and c4["prefix_cache_hit_tokens_drift"] == 0
            ),
            "c4_cache_hits_introduce_no_new_output_hash": bool(c4_hit_hashes)
            and c4_hit_hashes <= c4_cold_hashes,
            "capacity_headroom_ge_25_percent": headroom_percent >= 25.0,
        }
        targets[target] = {
            "verdict": "SELLABLE" if all(criteria.values()) else "NOT at c=4",
            "criteria": criteria,
            "correctness": correctness["targets"][target],
            "cache_exactness": exactness,
            "sold_cap": SOLD_CAP,
            "capacity_width": capacity_width,
            "capacity_headroom_percent": headroom_percent,
            "capacity_path": capacity_path,
            "c4_mixed90": c4,
            "c4_cold": c4_cold,
        }

    pair_c4_rates = []
    for rep in range(1, repetitions + 1):
        matching = [
            cell
            for cell in cells
            if cell["arm"] == "mixed90"
            and int(cell["concurrency"]) == SOLD_CAP
            and int(cell["rep"]) == rep
        ]
        if len(matching) != 2:
            raise ValueError(f"c4 mixed rep {rep}: expected a two-model cell pair")
        pair_c4_rates.append(
            sum(int(cell["completion_tokens"]) for cell in matching)
            / max(float(cell["wall_s"]) for cell in matching)
        )

    sha256s = parse_sha256s(args.campaign / "SHA256SUMS.input")
    result = {
        "schema": "memra.sellgate.summary.v1",
        "overall_verdict": (
            "GO" if any(row["verdict"] == "SELLABLE" for row in targets.values()) else "NO-GO"
        ),
        "targets": targets,
        "pair_shape": {
            "both_models_active_simultaneously": True,
            "c4_mixed_pair_output_tok_s_median_same_window": statistics.median(pair_c4_rates),
            "c4_mixed_pair_output_tok_s_replicates": pair_c4_rates,
        },
        "protocol": protocol,
        "replay_final": final,
        "replay_counts": {
            "requests": len(requests),
            "cells": len(cells),
            "levels": levels,
        },
        "measurements_by_target_arm_concurrency": groups,
        "correctness": correctness,
        "thermal": thermal_summary(args.campaign / "gpu-250ms.csv"),
        "input_sha256s": sha256s,
        "manifests": {"campaign": campaign_manifest, "gates": gates_manifest},
        "gate_semantics": {
            "two_second_metric": "content-bearing TTFT; full-response latency is published separately",
            "percentile_method": "nearest-rank except p50 uses the median of the full request population",
            "capacity_headroom": (
                "highest consecutive clean mixed90 width above c4 whose N=5 median output "
                "throughput rises over the preceding measured width"
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall_verdict": result["overall_verdict"], "targets": {
        target: row["verdict"] for target, row in targets.items()
    }}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
