#!/usr/bin/env python3
"""Reduce the fixed increment-2 versus increment-1 Step A/B matrix."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"^r(?P<rep>[1-5])-(?P<arm>default|inc1)-c(?P<c>1|8)$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cells: dict[tuple[int, str, int], dict] = {}
    for line in args.points.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "summary":
            continue
        match = LABEL.fullmatch(str(row.get("label", "")))
        if not match:
            continue
        key = (int(match["rep"]), match["arm"], int(match["c"]))
        if key in cells:
            raise SystemExit(f"duplicate point: {key}")
        if row.get("n_error") != 0 or row.get("n_ok") != row.get("n_requests"):
            raise SystemExit(f"failed requests in {key}: {row}")
        cells[key] = row

    expected = {
        (rep, arm, concurrency)
        for rep in range(1, 6)
        for arm in ("default", "inc1")
        for concurrency in (1, 8)
    }
    if cells.keys() != expected:
        missing = sorted(expected - cells.keys())
        extra = sorted(cells.keys() - expected)
        raise SystemExit(f"matrix mismatch: missing={missing} extra={extra}")

    summary: dict[str, object] = {
        "schema": "memra.sigrouter2.perf.v1",
        "metric": "decode_window_tok_s",
        "runs_per_arm": 5,
        "comparison": "default device-resident routing versus increment-1 MEMRA_MOE_DEV=0",
        "thermal_regime": (
            "one exclusive GPU-lock hold; fresh server and warmup per arm; arm order alternated "
            "by repetition; c1/c8 order reversed between arms; 250 ms NVML sampling"
        ),
        "points": {},
    }
    points: dict[str, object] = summary["points"]  # type: ignore[assignment]
    for concurrency in (1, 8):
        default = [
            float(cells[(rep, "default", concurrency)]["decode_window_tok_s"])
            for rep in range(1, 6)
        ]
        inc1 = [
            float(cells[(rep, "inc1", concurrency)]["decode_window_tok_s"])
            for rep in range(1, 6)
        ]
        paired = [100.0 * (new / old - 1.0) for new, old in zip(default, inc1)]
        default_median = statistics.median(default)
        inc1_median = statistics.median(inc1)
        points[f"c{concurrency}"] = {
            "default_tok_s": default,
            "inc1_tok_s": inc1,
            "default_median_tok_s": default_median,
            "inc1_median_tok_s": inc1_median,
            "median_of_arm_medians_delta_pct": 100.0 * (default_median / inc1_median - 1.0),
            "paired_delta_pct": paired,
            "paired_delta_median_pct": statistics.median(paired),
            "paired_wins": sum(delta > 0.0 for delta in paired),
        }

    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
