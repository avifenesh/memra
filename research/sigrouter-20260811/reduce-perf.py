#!/usr/bin/env python3
"""Reduce the fixed sigrouter x5 A/B matrix without selecting on public scores."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


LABEL = re.compile(r"^r(?P<rep>[1-5])-(?P<arm>default|rollback)-c(?P<c>1|8)$")


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
        for arm in ("default", "rollback")
        for concurrency in (1, 8)
    }
    missing = sorted(expected - cells.keys())
    extra = sorted(cells.keys() - expected)
    if missing or extra:
        raise SystemExit(f"matrix mismatch: missing={missing} extra={extra}")

    summary: dict[str, object] = {
        "schema": "memra.sigrouter.perf.v1",
        "metric": "decode_window_tok_s",
        "runs_per_arm": 5,
        "thermal_regime": (
            "one exclusive GPU-lock hold; fresh server per arm; one warmup burst; "
            "default and rollback order alternated by repetition; c1 and c8 order reversed "
            "between arms; continuous 250 ms NVML sampling"
        ),
        "points": {},
    }
    points: dict[str, object] = summary["points"]  # type: ignore[assignment]
    for concurrency in (1, 8):
        default = [
            float(cells[(rep, "default", concurrency)]["decode_window_tok_s"])
            for rep in range(1, 6)
        ]
        rollback = [
            float(cells[(rep, "rollback", concurrency)]["decode_window_tok_s"])
            for rep in range(1, 6)
        ]
        paired = [100.0 * (new / old - 1.0) for new, old in zip(default, rollback)]
        default_median = statistics.median(default)
        rollback_median = statistics.median(rollback)
        points[f"c{concurrency}"] = {
            "default_tok_s": default,
            "rollback_tok_s": rollback,
            "default_median_tok_s": default_median,
            "rollback_median_tok_s": rollback_median,
            "median_of_arm_medians_delta_pct": 100.0 * (default_median / rollback_median - 1.0),
            "paired_delta_pct": paired,
            "paired_delta_median_pct": statistics.median(paired),
            "paired_wins": sum(delta > 0.0 for delta in paired),
        }

    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
