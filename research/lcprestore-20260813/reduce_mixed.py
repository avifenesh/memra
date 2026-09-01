#!/usr/bin/env python3
"""Reduce five frozen Q27 sold-shape replays for feature-off/on comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


LEVELS = [1, 2, 4, 8, 12, 16, 20]
KNEE_GRID = [4, 8, 12, 16, 20]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("lcprestore_frozen_replay", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_arm(root: Path, frozen) -> dict[str, object]:
    files = sorted(root.glob("r??.jsonl"))
    if len(files) != 5:
        raise ValueError(f"{root}: expected five repetitions, found {len(files)}")
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
    cells = [row for row in rows if row.get("kind") == "cell"]
    requests = [row for row in rows if row.get("kind") == "request"]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    if len(summaries) != 5 or any(row.get("verdict") != "PASS" for row in summaries):
        raise ValueError(f"{root}: one or more repetition summaries are not PASS")
    expected_cells = 5 * len(LEVELS) * 2
    if len(cells) != expected_cells or any(not row.get("clean") for row in cells):
        raise ValueError(f"{root}: cell integrity failed ({len(cells)} != {expected_cells})")
    groups = {
        arm: {
            level: frozen.summarize_cell_group("q27", arm, level, requests, cells)
            for level in LEVELS
        }
        for arm in ("cold", "mixed90")
    }
    mixed = groups["mixed90"]
    knee = KNEE_GRID[0]
    path = []
    for previous, level in zip(KNEE_GRID, KNEE_GRID[1:]):
        rose = bool(
            mixed[previous]["all_clean"]
            and mixed[level]["all_clean"]
            and float(mixed[level]["output_tok_s_median"])
            > float(mixed[previous]["output_tok_s_median"])
        )
        path.append(
            {
                "from": previous,
                "to": level,
                "from_output_tok_s": mixed[previous]["output_tok_s_median"],
                "to_output_tok_s": mixed[level]["output_tok_s_median"],
                "clean_rise": rose,
            }
        )
        if not rose:
            break
        knee = level
    return {
        "repetitions": 5,
        "cells": len(cells),
        "all_cells_clean": True,
        "c4_mixed90": mixed[4],
        "c4_cold": groups["cold"][4],
        "knee": knee,
        "knee_path": path,
    }


def pct(candidate: float, control: float) -> float | None:
    return (candidate / control - 1.0) * 100.0 if control else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    frozen = load_module(args.frozen_replay)
    control = load_arm(args.control, frozen)
    candidate = load_arm(args.candidate, frozen)
    c = control["c4_mixed90"]
    n = candidate["c4_mixed90"]
    deltas = {
        "c4_output_tok_s_pct": pct(
            float(n["output_tok_s_median"]), float(c["output_tok_s_median"])
        ),
        "c4_hit_ttft_p50_pct": pct(
            float(n["ttft_hit"]["p50_ms"]), float(c["ttft_hit"]["p50_ms"])
        ),
        "c4_hit_ttft_p95_pct": pct(
            float(n["ttft_hit"]["p95_ms"]), float(c["ttft_hit"]["p95_ms"])
        ),
        "knee_width": int(candidate["knee"]) - int(control["knee"]),
    }
    # Pre-declared regression bounds: 5% throughput, 10% cache-hit TTFT, or a lower clean knee.
    # These are guardrails, not claims of statistical significance; every raw repetition remains.
    failures = []
    if float(deltas["c4_output_tok_s_pct"]) < -5.0:
        failures.append("c4 mixed output throughput regressed by more than 5%")
    if float(deltas["c4_hit_ttft_p50_pct"]) > 10.0:
        failures.append("c4 hit TTFT p50 regressed by more than 10%")
    if float(deltas["c4_hit_ttft_p95_pct"]) > 10.0:
        failures.append("c4 hit TTFT p95 regressed by more than 10%")
    if int(deltas["knee_width"]) < 0:
        failures.append("clean mixed-serve knee moved lower")
    summary = {
        "schema": "memra.lcprestore.mixed-regression.v1",
        "protocol": {
            "repetitions": 5,
            "feature_arms_alternated": True,
            "levels": LEVELS,
            "knee_grid": KNEE_GRID,
            "physical_gpu_index": args.physical_gpu,
            "physical_gpu_uuid": args.gpu_uuid,
            "regression_bounds": {
                "c4_output_tok_s_pct_min": -5.0,
                "c4_hit_ttft_p50_pct_max": 10.0,
                "c4_hit_ttft_p95_pct_max": 10.0,
                "knee_width_delta_min": 0,
            },
        },
        "control": control,
        "candidate": candidate,
        "candidate_vs_control": deltas,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
