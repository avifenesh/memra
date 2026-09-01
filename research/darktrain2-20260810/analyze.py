#!/usr/bin/env python3
"""Reduce darktrain2 box1 receipts without replacing the raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
from typing import Any


CELL_RE = re.compile(r"rep(?P<rep>[1-3])-(?P<arm>absent|running|parked)$")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def span(values: list[float]) -> list[float] | None:
    return [round(min(values), 6), round(max(values), 6)] if values else None


def delta_pct(value: float | None, base: float | None) -> float | None:
    if value is None or base in (None, 0):
        return None
    return round((value - base) / base * 100, 3)


def qos_summary(raw: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for summary_path in raw.glob("**/qos-summary.json"):
        match = CELL_RE.match(summary_path.parent.name)
        if not match:
            continue
        row = load_json(summary_path)
        metrics_path = summary_path.parent / "metrics-after.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        cells.append(
            {
                "rep": int(match.group("rep")),
                "arm": match.group("arm"),
                "ttft_p50_s": row["ttft_p50_s"],
                "ttft_p99_s": row["ttft_p99_s"],
                "latency_p50_s": row["latency_p50_s"],
                "latency_p99_s": row["latency_p99_s"],
                "step_p50_ms": metrics.get("step_p50_ms"),
                "step_p99_ms": metrics.get("step_p99_ms"),
                "yield_or_prepark_ms": row.get("watch_latency_ms"),
                "watch_state_before": row.get("watch_state_before"),
                "watch_terminal": row.get("watch_terminal"),
                "exactness": row.get("exactness"),
                "text_sha256": row.get("text_sha256"),
                "n_ok": row.get("n_ok"),
                "n_error": row.get("n_error"),
            }
        )
    cells.sort(key=lambda row: (row["rep"], row["arm"]))
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("absent", "running", "parked"):
        rows = [row for row in cells if row["arm"] == arm]
        arm_summary: dict[str, Any] = {"n_cells": len(rows), "requests_per_cell": 8}
        for key in (
            "ttft_p50_s",
            "ttft_p99_s",
            "latency_p50_s",
            "latency_p99_s",
            "step_p50_ms",
            "step_p99_ms",
        ):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            arm_summary[key] = median(values)
            arm_summary[key + "_range"] = span(values)
        arms[arm] = arm_summary
    base = arms.get("absent", {})
    for arm in ("running", "parked"):
        for key in ("ttft_p50_s", "ttft_p99_s", "step_p50_ms"):
            arms[arm][key + "_delta_pct"] = delta_pct(arms[arm].get(key), base.get(key))
    running_yields = [
        float(row["yield_or_prepark_ms"])
        for row in cells
        if row["arm"] == "running" and row.get("yield_or_prepark_ms") is not None
    ]
    campaign_complete = all(arms[arm]["n_cells"] == 3 for arm in arms)
    return {
        "cells": cells,
        "arms": arms,
        "campaign_complete": campaign_complete,
        "expected_cells": 9,
        "observed_cells": len(cells),
        "running_sigstop_ms_median": median(running_yields),
        "running_sigstop_ms_range": span(running_yields),
        "all_exact": bool(cells) and all(row["exactness"] == "match" for row in cells),
        "all_requests_ok": bool(cells)
        and all(row["n_ok"] == 8 and row["n_error"] == 0 for row in cells),
    }


def allocator_summary(raw: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("default", "maxsplit"):
        candidates = list(raw.glob(f"**/{arm}/trainer-events.jsonl"))
        if not candidates:
            continue
        events = load_jsonl(candidates[0])
        selected: dict[str, Any] = {}
        for name in ("process_start", "torch_imported", "cuda_probed", "cuda_initialized",
                     "setup_complete", "optimizer_step"):
            rows = [row for row in events if row.get("event") == name]
            if rows:
                selected[name] = rows[-1]
        steps = [row for row in events if row.get("event") == "optimizer_step"]
        selected["budget_violations"] = sum(
            row.get("event") == "budget_violation" for row in events
        )
        selected["optimizer_step_p50_ms"] = median(
            [float(row["step_ms"]) for row in steps if row.get("step_ms") is not None]
        )
        result[arm] = selected
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = {
        "allocator": allocator_summary(args.raw),
        "qos": qos_summary(args.raw),
        "checkpoint_resume": (
            load_json(args.raw / "checkpoint" / "checkpoint-resume.json")
            if (args.raw / "checkpoint" / "checkpoint-resume.json").exists()
            else None
        ),
        "refusal_metrics": (
            load_json(args.raw / "refusal" / "metrics.json")
            if (args.raw / "refusal" / "metrics.json").exists()
            else None
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
