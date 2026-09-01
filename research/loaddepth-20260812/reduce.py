#!/usr/bin/env python3
"""Reduce immutable cx-loaddepth box1 receipts into the scored curve."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import statistics


WIDTHS = (8, 10, 12, 16, 20, 24)
ANCHOR_C8_TOK_S = 158.065


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_summary(path: pathlib.Path) -> dict:
    rows = [row for row in jsonl(path) if row.get("kind") == "summary"]
    if len(rows) != 1:
        raise ValueError(f"expected one summary in {path}, found {len(rows)}")
    return rows[0]


def thermal(path: pathlib.Path) -> dict:
    by_gpu: dict[int, dict[str, list[float]]] = {}
    with path.open(newline="", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) < 9:
                continue
            try:
                gpu = int(row[1].strip())
                values = {
                    "temperature_c": float(row[3].strip()),
                    "power_w": float(row[4].strip()),
                    "sm_clock_mhz": float(row[6].strip()),
                    "memory_used_mib": float(row[7].strip()),
                }
            except ValueError:
                continue
            target = by_gpu.setdefault(gpu, {key: [] for key in values})
            for key, value in values.items():
                target[key].append(value)
    if not by_gpu:
        raise ValueError(f"no thermal samples in {path}")
    return {
        "artificial_cooldown": False,
        "sample_interval_ms": 250,
        "rows": sum(len(values["temperature_c"]) for values in by_gpu.values()),
        "intervals": min(len(values["temperature_c"]) for values in by_gpu.values()),
        "gpus": {
            str(gpu): {
                **{key + "_min": min(values) for key, values in fields.items()},
                **{key + "_max": max(values) for key, values in fields.items()},
            }
            for gpu, fields in sorted(by_gpu.items())
        },
        "all_gpus": {
            key + "_min": min(value for fields in by_gpu.values() for value in fields[key])
            for key in ("temperature_c", "power_w", "sm_clock_mhz", "memory_used_mib")
        }
        | {
            key + "_max": max(value for fields in by_gpu.values() for value in fields[key])
            for key in ("temperature_c", "power_w", "sm_clock_mhz", "memory_used_mib")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    exactness = json.loads((args.root / "exactness-verdict.json").read_text())
    allowed = [width for width in WIDTHS if width < int(exactness.get("failed_width") or 10**9)]
    points: list[dict] = []
    pattern = re.compile(r"r(\d+)-p(\d+)-c(\d+)")
    for path in sorted((args.root / "perf").glob("*/score.jsonl")):
        row = one_summary(path)
        match = pattern.fullmatch(str(row["label"]))
        if not match:
            raise ValueError(f"unexpected label {row['label']!r} in {path}")
        row = {**row, "rep": int(match.group(1)), "position": int(match.group(2))}
        width = int(match.group(3))
        if width != int(row["concurrency"]):
            raise ValueError(f"label/concurrency mismatch in {path}")
        points.append(row)

    expected_points = len(allowed) * 3
    if len(points) != expected_points:
        raise ValueError(f"expected {expected_points} score points, found {len(points)}")
    widths: dict[str, dict] = {}
    for width in allowed:
        rows = [row for row in points if int(row["concurrency"]) == width]
        if sorted(int(row["rep"]) for row in rows) != [1, 2, 3]:
            raise ValueError(f"c={width} replicate mismatch")
        for row in rows:
            counters = row["admission_counters"]
            if row["n_ok"] != row["n_requests"] or row["n_error"] != 0:
                raise ValueError(f"request error at c={width} r={row['rep']}")
            if row["completion_tokens_total"] != width * 128:
                raise ValueError(f"short completion at c={width} r={row['rep']}")
            if counters["tokens_out"] != width * 128:
                raise ValueError(f"worker token mismatch at c={width} r={row['rep']}")
            dual = row["dual_pp"]
            if dual["slot_pairs"] <= 0 or dual["slot_collisions"] != 0:
                raise ValueError(f"dual slot failure at c={width} r={row['rep']}")

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in sorted(rows, key=lambda item: item["rep"])]

        aggregate = values("aggregate_tok_s")
        ttft50 = values("ttft_p50_s")
        ttft99 = values("ttft_p99_s")
        step99 = values("step_p99_ms")
        widths[f"c{width}"] = {
            "N": 3,
            "aggregate_tok_s": aggregate,
            "aggregate_tok_s_median": statistics.median(aggregate),
            "aggregate_tok_s_min": min(aggregate),
            "aggregate_tok_s_max": max(aggregate),
            "ttft_p50_s": ttft50,
            "ttft_p50_s_median": statistics.median(ttft50),
            "ttft_p99_s": ttft99,
            "ttft_p99_s_median": statistics.median(ttft99),
            "step_p99_ms": step99,
            "step_p99_ms_median": statistics.median(step99),
            "admission_counters_total": {
                key: sum(int(row["admission_counters"][key]) for row in rows)
                for key in (
                    "admitted",
                    "completed",
                    "tokens_out",
                    "admission_session_defers",
                    "admission_vram_defers",
                    "step_oom_parks",
                )
            },
            "peak_active_sessions_sampled": max(
                int(row["peak_active_sessions_sampled"]) for row in rows
            ),
            "peak_queued_requests_sampled": max(
                int(row["peak_queued_requests_sampled"]) for row in rows
            ),
            "replicate_positions": [int(row["position"]) for row in sorted(rows, key=lambda item: item["rep"])],
        }

    non_rising = None
    latency_crossing = None
    previous = None
    for width in allowed:
        row = widths[f"c{width}"]
        throughput = float(row["aggregate_tok_s_median"])
        if previous is not None and throughput <= previous and non_rising is None:
            non_rising = width
        if float(row["ttft_p99_s_median"]) > 15.0 and latency_crossing is None:
            latency_crossing = width
        previous = throughput
    knee_candidates = [value for value in (non_rising, latency_crossing) if value is not None]
    knee = min(knee_candidates) if knee_candidates else None
    serve_ready = [
        width
        for width in allowed
        if float(widths[f"c{width}"]["ttft_p99_s_median"]) <= 15.0
    ]
    optimal = (
        max(serve_ready, key=lambda width: float(widths[f"c{width}"]["aggregate_tok_s_median"]))
        if serve_ready
        else None
    )
    optimal_tok_s = (
        float(widths[f"c{optimal}"]["aggregate_tok_s_median"])
        if optimal is not None
        else None
    )
    c8 = float(widths["c8"]["aggregate_tok_s_median"]) if "c8" in widths else None
    summary = {
        "schema": "memra.cx-loaddepth.v1",
        "source_commit": args.source,
        "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
        "protocol": "N=3 interleaved widths; fresh naked server per point; same-width discarded warmup; 128 tokens/request; one lock hold",
        "metric": "aggregate completion tokens / barrier-release-to-final-drain wall second",
        "exactness": exactness,
        "widths": widths,
        "anchor": {
            "flip_battery_c8_tok_s": ANCHOR_C8_TOK_S,
            "measured_c8_tok_s": c8,
            "delta_pct": ((c8 / ANCHOR_C8_TOK_S) - 1.0) * 100.0 if c8 else None,
        },
        "knee": {
            "rule": "first width where median aggregate tok/s does not rise or median window p99 TTFT exceeds 15s",
            "first_non_rising_width": non_rising,
            "first_ttft_over_15s_width": latency_crossing,
            "width": knee,
        },
        "revenue_optimum": {
            "rule": "highest median aggregate tok/s among widths with median window p99 TTFT <= 15s",
            "width": optimal,
            "aggregate_tok_s": optimal_tok_s,
            "tokens_per_day": optimal_tok_s * 86_400 if optimal_tok_s is not None else None,
        },
        "thermal_regime": thermal(args.root / "gpu.csv"),
        "receipt": {
            "scored_points": len(points),
            "requests": sum(int(row["n_requests"]) for row in points),
            "request_errors": sum(int(row["n_error"]) for row in points),
            "completion_tokens": sum(int(row["completion_tokens_total"]) for row in points),
        },
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
