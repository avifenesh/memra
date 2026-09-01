#!/usr/bin/env python3
"""Attach thermal samples and reduce the frozen MoESD B x gamma matrix."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


BATCHES = (1, 2, 4, 8, 16, 24, 32)
GAMMAS = (1, 2, 3, 4, 6, 8)
RUNS = 5
PIVOT_B = 8
PIVOT_GAMMA = 4
X = 1.5
PLAIN_B8_TOKS = 173.62


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def thermal_for(
    row: dict[str, Any], samples: list[dict[str, Any]]
) -> tuple[list[float | None], list[float | None], list[float | None], int]:
    start = int(row["started_unix_ms"])
    finish = int(row["finished_unix_ms"])
    selected = [
        sample for sample in samples if start - 300 <= int(sample["unix_ms"]) <= finish + 300
    ]
    if not selected:
        midpoint = (start + finish) // 2
        selected = [min(samples, key=lambda sample: abs(int(sample["unix_ms"]) - midpoint))]
    indices = sorted(
        {int(gpu["index"]) for sample in selected for gpu in sample.get("gpus", [])}
    )
    max_c: list[float | None] = []
    max_w: list[float | None] = []
    avg_c: list[float | None] = []
    for index in indices:
        gpus = [
            gpu
            for sample in selected
            for gpu in sample.get("gpus", [])
            if int(gpu["index"]) == index
        ]
        temperatures = [float(gpu["temperature_C"]) for gpu in gpus if gpu["temperature_C"] is not None]
        powers = [float(gpu["power_W"]) for gpu in gpus if gpu["power_W"] is not None]
        max_c.append(max(temperatures) if temperatures else None)
        max_w.append(max(powers) if powers else None)
        avg_c.append(statistics.fmean(temperatures) if temperatures else None)
    return max_c, max_w, avg_c, len(selected)


def derive(
    measurements: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not samples:
        raise ValueError("thermal sample stream is empty")
    expected = {(run, batch, gamma) for run in range(1, RUNS + 1) for batch in BATCHES for gamma in GAMMAS}
    observed = {(int(row["run"]), int(row["B"]), int(row["gamma"])) for row in measurements}
    if observed != expected or len(measurements) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"matrix is incomplete or duplicated: rows={len(measurements)} expected={len(expected)} "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    expected_order = []
    for run in range(1, RUNS + 1):
        batches = BATCHES if run % 2 else tuple(reversed(BATCHES))
        gammas = GAMMAS if run % 2 else tuple(reversed(GAMMAS))
        expected_order.extend((run, batch, gamma) for batch in batches for gamma in gammas)
    observed_order = [
        (int(row["run"]), int(row["B"]), int(row["gamma"])) for row in measurements
    ]
    if observed_order != expected_order:
        mismatch = next(
            index
            for index, (observed_cell, expected_cell) in enumerate(
                zip(observed_order, expected_order, strict=True)
            )
            if observed_cell != expected_cell
        )
        raise ValueError(
            f"matrix interleave differs at row {mismatch + 1}: "
            f"observed={observed_order[mismatch]} expected={expected_order[mismatch]}"
        )
    baseline = {
        (int(row["run"]), int(row["B"])): float(row["ms_step"])
        for row in measurements
        if int(row["gamma"]) == 1
    }
    enriched: list[dict[str, Any]] = []
    for measurement in measurements:
        row = dict(measurement)
        run, batch, gamma = int(row["run"]), int(row["B"]), int(row["gamma"])
        paper_eff = baseline[(run, batch)] / float(row["ms_step"])
        row["target_eff"] = paper_eff
        row["serial_amortization"] = gamma * paper_eff
        max_c, max_w, avg_c, count = thermal_for(row, samples)
        row["thermal_max_C"] = max_c
        row["thermal_max_W"] = max_w
        row["thermal_avg_C"] = avg_c
        row["thermal_samples"] = count
        enriched.append(row)

    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        groups[(int(row["B"]), int(row["gamma"]))].append(row)
    summaries: list[dict[str, Any]] = []
    median_ms: dict[tuple[int, int], float] = {}
    for key, rows in groups.items():
        median_ms[key] = median([float(row["ms_step"]) for row in rows])
    for batch in BATCHES:
        for gamma in GAMMAS:
            rows = groups[(batch, gamma)]
            if len(rows) != RUNS:
                raise ValueError(f"B={batch} gamma={gamma} has N={len(rows)}, expected {RUNS}")
            times = [float(row["ms_step"]) for row in rows]
            paper_eff_runs = median([float(row["target_eff"]) for row in rows])
            serial_eff_runs = median([float(row["serial_amortization"]) for row in rows])
            paper_eff_ratio = median_ms[(batch, 1)] / median_ms[(batch, gamma)]
            serial_eff_ratio = gamma * paper_eff_ratio
            layer_ids = [int(layer["id"]) for layer in rows[0]["layers"]]
            union_median = []
            for layer_id in layer_ids:
                values = [
                    int(next(layer for layer in row["layers"] if int(layer["id"]) == layer_id)["union"])
                    for row in rows
                ]
                union_median.append(
                    {"id": layer_id, "union": int(statistics.median(values)), "range": [min(values), max(values)]}
                )
            verdict = "amortized" if serial_eff_ratio > X else "marginal" if serial_eff_ratio > 1.2 else "serial"
            summaries.append(
                {
                    "summary": True,
                    "B": batch,
                    "gamma": gamma,
                    "N": RUNS,
                    "ms_step_median": median(times),
                    "ms_step_range": [min(times), max(times)],
                    "union_median": union_median,
                    "target_eff_median": paper_eff_runs,
                    "serial_amortization_median": serial_eff_runs,
                    "target_eff_ratio_of_medians": paper_eff_ratio,
                    "serial_amortization_ratio_of_medians": serial_eff_ratio,
                    "effective_toks_median": median([float(row["effective_toks"]) for row in rows]),
                    "realistic_toks_median": median([float(row["realistic_toks"]) for row in rows]),
                    "verdict": verdict,
                }
            )
    pivot = next(
        row for row in summaries if row["B"] == PIVOT_B and row["gamma"] == PIVOT_GAMMA
    )
    if float(pivot["serial_amortization_ratio_of_medians"]) > X:
        if float(pivot["realistic_toks_median"]) > PLAIN_B8_TOKS:
            verdict = "GO"
            reason = "batch-verify amortization is live at c=8; DSpark training gains PP-2 as a second venue"
        else:
            verdict = "HOLD"
            reason = "amortization exists but acceptance is too weak; focus DSpark acceptance training first"
    else:
        verdict = "CLOSED"
        reason = "no amortization at c=8; K=0 remains correct and the next dollar goes to DSpark acceptance-only"
    decision = {
        "decision": True,
        "pivot_B": PIVOT_B,
        "pivot_gamma": PIVOT_GAMMA,
        "X": X,
        "plain_B8_toks": PLAIN_B8_TOKS,
        "ms_step_B8_gamma1_median": median_ms[(PIVOT_B, 1)],
        "ms_step_B8_gamma4_median": median_ms[(PIVOT_B, PIVOT_GAMMA)],
        "target_eff": pivot["target_eff_ratio_of_medians"],
        "serial_amortization": pivot["serial_amortization_ratio_of_medians"],
        "target_eff_runs_median": pivot["target_eff_median"],
        "serial_amortization_runs_median": pivot["serial_amortization_median"],
        "realistic_toks": pivot["realistic_toks_median"],
        "go_no_go": "GO" if verdict == "GO" else "NO-GO",
        "verdict": verdict,
        "reason": reason,
        "reopens_pp2_87_94": False,
    }
    return enriched, summaries, decision


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            print(json.dumps(row, separators=(",", ":"), sort_keys=True), file=output)


def self_test() -> None:
    measurements = []
    for run in range(1, RUNS + 1):
        batches = BATCHES if run % 2 else tuple(reversed(BATCHES))
        gammas = GAMMAS if run % 2 else tuple(reversed(GAMMAS))
        for batch in batches:
            for gamma in gammas:
                ms = 10.0 if gamma == 1 else 5.0 * gamma
                measurements.append(
                    {
                        "run": run,
                        "B": batch,
                        "gamma": gamma,
                        "ms_step": ms,
                        "started_unix_ms": 1_000 + len(measurements) * 10,
                        "finished_unix_ms": 1_005 + len(measurements) * 10,
                        "layers": [{"id": 3, "union": min(256, batch * gamma * 8)}],
                        "effective_toks": batch * gamma / (ms / 1000),
                        "realistic_toks": batch * 1.108 / (ms / 1000),
                    }
                )
    samples = [
        {
            "unix_ms": 1_000 + index * 10,
            "gpus": [
                {"index": 0, "temperature_C": 55.0, "power_W": 400.0},
                {"index": 1, "temperature_C": 56.0, "power_W": 410.0},
            ],
        }
        for index in range(len(measurements) + 1)
    ]
    enriched, summaries, decision = derive(measurements, samples)
    assert len(enriched) == 210
    assert len(summaries) == 42
    assert decision["serial_amortization"] == 2.0
    assert decision["go_no_go"] == "GO"
    assert decision["verdict"] == "GO"
    print("summarize self-test: PASS")


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(sys.argv) != 6:
        print(
            "usage: summarize.py <measurements.jsonl> <thermal.jsonl> <raw.jsonl> <RESULTS.jsonl> <summary.json>",
            file=sys.stderr,
        )
        return 2
    measurements_path, thermal_path, raw_path, results_path, summary_path = map(Path, sys.argv[1:])
    enriched, summaries, decision = derive(
        load_jsonl(measurements_path), load_jsonl(thermal_path)
    )
    write_jsonl(raw_path, enriched)
    write_jsonl(results_path, summaries + [decision])
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(
            {"format": "memra-moesd-target-efficiency-v1", "cells": summaries, "decision": decision},
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
