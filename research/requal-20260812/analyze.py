#!/usr/bin/env python3
"""Extract the explicit c=2 regression matrix and compare Q27 to sellgate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PERCENTILES = ("p50_ms", "p75_ms", "p90_ms", "p95_ms", "p99_ms")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def regression_percent(old: float, new: float, higher_is_better: bool) -> float:
    if old == 0:
        return 0.0 if new == old else float("inf")
    delta = old - new if higher_is_better else new - old
    return delta / abs(old) * 100.0


def compare_number(
    rows: list[dict[str, Any]],
    metric: str,
    old: Any,
    new: Any,
    higher_is_better: bool,
) -> None:
    if old is None or new is None:
        return
    old_value = float(old)
    new_value = float(new)
    regression = regression_percent(old_value, new_value, higher_is_better)
    rows.append(
        {
            "metric": metric,
            "old": old_value,
            "fresh": new_value,
            "higher_is_better": higher_is_better,
            "regression_percent": regression,
            "regression_gt_2_percent": regression > 2.0,
        }
    )


def compare_q27(original: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    old_groups = original["measurements_by_target_arm_concurrency"]["q27"]
    new_groups = fresh["measurements_by_target_arm_concurrency"]["q27"]
    comparisons: list[dict[str, Any]] = []
    old_c4_mixed = old_groups["mixed90"]["4"]
    new_c4_mixed = new_groups["mixed90"]["4"]
    old_c4_cold = old_groups["cold"]["4"]
    new_c4_cold = new_groups["cold"]["4"]

    # These are the original Q27 one-page's published c=4 rates and distributions.
    for metric in (
        "requests_per_s_median",
        "output_tok_s_median",
        "billed_prompt_tok_s_median",
        "computed_prompt_tok_s_median",
    ):
        compare_number(
            comparisons,
            f"q27.c4.mixed90.{metric}",
            old_c4_mixed[metric],
            new_c4_mixed[metric],
            True,
        )
    compare_number(
        comparisons,
        "q27.c4.cold.output_tok_s_median",
        old_c4_cold["output_tok_s_median"],
        new_c4_cold["output_tok_s_median"],
        True,
    )
    for arm, old_group, new_group, distributions in (
        (
            "mixed90",
            old_c4_mixed,
            new_c4_mixed,
            (
                "ttft_all", "ttft_hit", "ttft_miss",
                "latency_all", "latency_hit", "latency_miss",
                "inter_token_all", "inter_token_hit", "inter_token_miss",
            ),
        ),
        ("cold", old_c4_cold, new_c4_cold, ("ttft_all", "latency_all", "inter_token_all")),
    ):
        for distribution in distributions:
            for percentile in PERCENTILES:
                compare_number(
                    comparisons,
                    f"q27.c4.{arm}.{distribution}.{percentile}",
                    old_group[distribution].get(percentile),
                    new_group[distribution].get(percentile),
                    False,
                )

    # The capacity board publishes five values per width. c=4 is already covered above.
    common_levels = sorted(set(old_groups["mixed90"]) & set(new_groups["mixed90"]), key=int)
    for level in (value for value in common_levels if value != "4"):
        old_mixed = old_groups["mixed90"][level]
        new_mixed = new_groups["mixed90"][level]
        old_cold = old_groups["cold"][level]
        new_cold = new_groups["cold"][level]
        compare_number(
            comparisons,
            f"q27.capacity.c{level}.cold.output_tok_s_median",
            old_cold["output_tok_s_median"],
            new_cold["output_tok_s_median"],
            True,
        )
        compare_number(
            comparisons,
            f"q27.capacity.c{level}.mixed90.output_tok_s_median",
            old_mixed["output_tok_s_median"],
            new_mixed["output_tok_s_median"],
            True,
        )
        for distribution, percentile in (
            ("ttft_hit", "p95_ms"),
            ("ttft_all", "p50_ms"),
            ("ttft_all", "p99_ms"),
        ):
            compare_number(
                comparisons,
                f"q27.capacity.c{level}.mixed90.{distribution}.{percentile}",
                old_mixed[distribution][percentile],
                new_mixed[distribution][percentile],
                False,
            )

    old_target = original["targets"]["q27"]
    new_target = fresh["targets"]["q27"]
    compare_number(
        comparisons,
        "q27.capacity_width",
        old_target["capacity_width"],
        new_target["capacity_width"],
        True,
    )
    compare_number(
        comparisons,
        "q27.capacity_headroom_percent",
        old_target["capacity_headroom_percent"],
        new_target["capacity_headroom_percent"],
        True,
    )
    regressions = [row for row in comparisons if row["regression_gt_2_percent"]]
    return {
        "threshold_percent": 2.0,
        "old_levels": original["replay_counts"]["levels"],
        "fresh_levels": fresh["replay_counts"]["levels"],
        "scope": "every non-exact published Q27 number in the original RESULTS one-page and capacity table",
        "comparisons": comparisons,
        "regressions_gt_2_percent": regressions,
        "regression_count": len(regressions),
    }


def regression_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = [row for row in rows if row.get("kind") == "protocol"]
    if len(protocol) != 1:
        raise ValueError("fresh replay must contain exactly one protocol row")
    result: dict[str, Any] = {
        "workload_lock_sha256": protocol[0]["workload_lock_sha256"],
        "prompt_ids_sha256_canonical_json": protocol[0][
            "prompt_ids_sha256_canonical_json"
        ],
        "arm": "mixed90",
        "concurrency": 2,
        "repetitions": 5,
        "targets": {},
    }
    for target in ("q27", "q35"):
        cells = sorted(
            (
                row
                for row in rows
                if row.get("kind") == "cell"
                and row.get("target") == target
                and row.get("arm") == "mixed90"
                and int(row.get("concurrency", 0)) == 2
            ),
            key=lambda row: int(row["rep"]),
        )
        if len(cells) != 5 or [int(row["rep"]) for row in cells] != [1, 2, 3, 4, 5]:
            raise ValueError(f"{target}: expected mixed-c2 repetitions 1..5")
        requests = [
            row
            for row in rows
            if row.get("kind") == "request"
            and row.get("target") == target
            and row.get("arm") == "mixed90"
            and int(row.get("concurrency", 0)) == 2
        ]
        matrix = []
        for cell in cells:
            rep = int(cell["rep"])
            rep_requests = [row for row in requests if int(row["rep"]) == rep]
            matrix.append(
                {
                    "rep": rep,
                    "requests": len(rep_requests),
                    "requests_ok": sum(bool(row.get("ok")) for row in rep_requests),
                    "short_completions": sum(
                        int(row.get("completion_tokens") or 0) != 60
                        for row in rep_requests
                    ),
                    "response_usage_tokens": sum(
                        int(row.get("completion_tokens") or 0) for row in rep_requests
                    ),
                    "engine_tokens_out": int(cell["counter_deltas"]["tokens_out"]),
                    "cached_tokens_in_drift": int(cell["cached_tokens_in_drift"]),
                    "prefix_cache_hit_tokens_drift": int(
                        cell["prefix_cache_hit_tokens_drift"]
                    ),
                    "clean": bool(cell["clean"]),
                }
            )
        result["targets"][target] = {
            "matrix": matrix,
            "all_clean": all(row["clean"] for row in matrix),
            "requests": sum(row["requests"] for row in matrix),
            "response_usage_tokens": sum(row["response_usage_tokens"] for row in matrix),
            "engine_tokens_out": sum(row["engine_tokens_out"] for row in matrix),
            "short_completions": sum(row["short_completions"] for row in matrix),
        }
    result["both_models_clean"] = all(
        row["all_clean"] for row in result["targets"].values()
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-summary", type=Path, required=True)
    parser.add_argument("--fresh-replay", type=Path, required=True)
    parser.add_argument("--original-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    fresh = read_json(args.fresh_summary)
    original = read_json(args.original_summary)
    if fresh.get("schema") != "memra.sellgate.summary.v1":
        raise ValueError("fresh summary has the wrong schema")
    if original.get("schema") != "memra.sellgate.summary.v1":
        raise ValueError("original summary has the wrong schema")
    result = {
        "schema": "memra.requal.analysis.v1",
        "regression_matrix": regression_matrix(read_jsonl(args.fresh_replay)),
        "q27_comparison": compare_q27(original, fresh),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "both_regression_cells_clean": result["regression_matrix"]["both_models_clean"],
        "q27_regressions_gt_2_percent": result["q27_comparison"]["regression_count"],
    }, sort_keys=True))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
