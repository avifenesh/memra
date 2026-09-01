#!/usr/bin/env python3
"""Summarize cx-throughput raw receipts without mutating the raw JSONL."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(path: pathlib.Path) -> dict:
    rows = [row for row in jsonl(path) if row.get("kind") == "summary"]
    if len(rows) != 1:
        raise ValueError(f"expected one summary in {path}, found {len(rows)}")
    return rows[0]


def decode_window(path: pathlib.Path, output_tokens: int) -> dict:
    """Measure first content-bearing event through final request drain.

    This deliberately stays separate from the primary end-to-end aggregate, which
    starts at barrier release and therefore includes prefill/TTFT.
    """
    requests = [row for row in jsonl(path) if row.get("kind") == "request"]
    first_text = [
        float(row["started_unix_s"]) + float(row["ttft_s"])
        for row in requests
        if row.get("ttft_s") is not None
    ]
    drained = [
        float(row["started_unix_s"]) + float(row["wall_s"])
        for row in requests
        if row.get("wall_s") is not None
    ]
    if len(first_text) != len(requests) or len(drained) != len(requests) or not requests:
        return {}
    elapsed = max(drained) - min(first_text)
    return {
        "first_text_to_drain_s": round(elapsed, 6),
        "first_text_to_drain_output_tok_s": (
            round(output_tokens / elapsed, 6) if elapsed > 0 else None
        ),
    }


def step_rate(row: dict) -> dict:
    """Translate the rolling steady-state step p50 into output-token rate."""
    step_ms = row.get("step_p50_ms")
    concurrency = row.get("concurrency")
    if not isinstance(step_ms, (int, float)) or step_ms <= 0:
        return {}
    if not isinstance(concurrency, int) or concurrency <= 0:
        return {}
    return {"step_p50_implied_output_tok_s": round(concurrency * 1_000 / step_ms, 6)}


def thermal(server_dir: pathlib.Path) -> dict:
    maxima: dict[int, dict[str, float]] = {}
    path = server_dir / "gpu.csv"
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 8:
            continue
        try:
            index = int(fields[1])
            temp = float(fields[3])
            power = float(fields[4])
            memory = float(fields[6])
        except ValueError:
            continue
        values = maxima.setdefault(index, {"temp_c": 0.0, "power_w": 0.0, "memory_mib": 0.0})
        values["temp_c"] = max(values["temp_c"], temp)
        values["power_w"] = max(values["power_w"], power)
        values["memory_mib"] = max(values["memory_mib"], memory)
    return {f"gpu{index}": values for index, values in sorted(maxima.items())}


def median(values: list[float]) -> float:
    return statistics.median(values)


def aggregate(rows: list[dict], block: str, arm: str, cell: str) -> dict:
    fields = (
        "aggregate_output_tok_s",
        "first_text_to_drain_output_tok_s",
        "first_text_to_drain_s",
        "step_p50_implied_output_tok_s",
        "step_p50_ms",
        "step_p99_ms",
        "ttft_p50_s",
        "ttft_p95_s",
        "wall_s",
    )
    result = {
        "kind": "aggregate",
        "block": block,
        "arm": arm,
        "cell": cell,
        "n": len(rows),
        "replicates": [row["rep"] for row in rows],
        "requests_ok": sum(row["requests_ok"] for row in rows),
        "requests_n": sum(row["requests_n"] for row in rows),
        "expected_output_tokens": sum(row["expected_output_tokens"] for row in rows),
        "metrics_output_tokens": sum(row["metrics_output_tokens"] for row in rows),
        "admission_vram_defers": sum(row["admission_vram_defers"] for row in rows),
        "admission_session_defers": sum(row["admission_session_defers"] for row in rows),
        "step_oom_parks": sum(row["step_oom_parks"] for row in rows),
    }
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            result[field + "_median"] = median(values)
            result[field + "_min"] = min(values)
            result[field + "_max"] = max(values)
    return result


def collect(root: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for block_dir in sorted((root / "raw").glob("block-baseline-*")):
        for server_dir in sorted(block_dir.glob("rep*-*")):
            match = re.fullmatch(r"rep(\d+)-(off|on)", server_dir.name)
            if not match:
                continue
            rep = int(match.group(1))
            arm = match.group(2)
            therm = thermal(server_dir)
            for cell_dir in sorted(server_dir.glob("decode-c*")) + sorted(server_dir.glob("mixed-c16")):
                path = cell_dir / "requests.jsonl"
                if not path.exists():
                    continue
                row = summary(path)
                derived = decode_window(path, int(row["metrics_output_tokens"]))
                derived.update(step_rate(row))
                rows.append(
                    {
                        "kind": "replicate",
                        "block": "baseline",
                        "block_dir": block_dir.name,
                        "rep": rep,
                        "arm": arm,
                        "cell": cell_dir.name,
                        "thermal": therm,
                        **{key: value for key, value in row.items() if key not in ("kind", "n")},
                        **derived,
                    }
                )
    for block_dir in sorted((root / "raw").glob("block-knob-*")):
        for server_dir in sorted(block_dir.glob("rep*-*")):
            match = re.fullmatch(r"rep(\d+)-(default|tick2048)", server_dir.name)
            if not match:
                continue
            path = server_dir / "mixed-c16" / "requests.jsonl"
            if not path.exists():
                continue
            row = summary(path)
            derived = decode_window(path, int(row["metrics_output_tokens"]))
            derived.update(step_rate(row))
            rows.append(
                {
                    "kind": "replicate",
                    "block": "knob",
                    "block_dir": block_dir.name,
                    "rep": int(match.group(1)),
                    "arm": match.group(2),
                    "cell": "mixed-c16",
                    "thermal": thermal(server_dir),
                    **{key: value for key, value in row.items() if key not in ("kind", "n")},
                    **derived,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()

    replicates = collect(args.root)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in replicates:
        grouped.setdefault((row["block"], row["arm"], row["cell"]), []).append(row)
    aggregates = [
        aggregate(rows, block, arm, cell)
        for (block, arm, cell), rows in sorted(grouped.items())
    ]
    output = replicates + aggregates
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
