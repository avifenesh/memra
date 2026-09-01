#!/usr/bin/env python3
"""Reduce and assert the new-box matrix, correctness, and performance receipts."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics


def write_receipt(receipt: dict, destination: pathlib.Path) -> None:
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


def reduce_matrix(root: pathlib.Path, destination: pathlib.Path) -> None:
    rows = []
    cells = collections.Counter()
    for path in sorted(root.glob("*/*/qos-rows.jsonl")):
        condition = path.parents[1].name
        cells[condition] += 1
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    hashes = collections.Counter(row.get("text_sha256") for row in rows if row.get("ok"))
    receipt = {
        "conditions": dict(sorted(cells.items())),
        "cells": sum(cells.values()),
        "requests": len(rows),
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "golden_matches": sum(bool(row.get("golden_match")) for row in rows),
        "completion_hashes": dict(sorted(hashes.items())),
    }
    assert cells == {"h2-c1": 10, "same": 5}, receipt
    assert len(rows) == 50, receipt
    assert receipt["requests_ok"] == receipt["golden_matches"] == len(rows), receipt
    assert len(hashes) == 1, receipt
    receipt["verdict"] = "PASS"
    write_receipt(receipt, destination)


def reduce_correctness(root: pathlib.Path, destination: pathlib.Path) -> None:
    logs = {path.stem: path.read_text(errors="replace") for path in root.glob("*.log")}
    checks = {
        "kernel_check": "ALL GREEN: kernels match CPU reference" in logs.get("kernel-check", ""),
        "decode_batch": "pp mode verdict: 0 failing arm(s)" in logs.get("decode-batch-gate", ""),
        "run_gen_prefill_decode": "prefill argmax=" in logs.get("run-gen", "")
        and "  MATCH" in logs.get("run-gen", ""),
        "run_gen_batched_tokenwise": "batched-prime argmax=" in logs.get("run-gen", "")
        and "  MATCH" in logs.get("run-gen", ""),
        "run_spec_k1_8": logs.get("run-spec", "").count("self-consistency: PASS") == 8
        and "=== SELF-CONSISTENCY PASS ===" in logs.get("run-spec", ""),
        "chunk_naked": "chunk-invariance-gate: PASS" in logs.get("chunk-naked", ""),
        "chunk_canary": "canary broke the assertion as required" in logs.get("chunk-canary", ""),
        "tick_naked": "tick-invariance-gate: PASS" in logs.get("tick-naked", ""),
        "tick_canary": "canary broke the assertion as required" in logs.get("tick-canary", ""),
    }
    receipt = {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
    assert receipt["verdict"] == "PASS", receipt
    write_receipt(receipt, destination)


def reduce_bench(points: pathlib.Path, destination: pathlib.Path) -> None:
    rows = [json.loads(line) for line in points.read_text().splitlines() if line.strip()]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    groups: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    for row in summaries:
        groups[(row["shape"], int(row["concurrency"]))].append(row)
    expected = {("short", 1), ("4k", 1), ("decode", 1), ("decode", 4), ("decode", 8)}
    assert set(groups) == expected, sorted(groups)
    receipt = {
        "n_per_point": 5,
        "thermal_regime": (
            "five fresh server boots; forward/reverse point order alternated by rep; "
            "continuous 500 ms NVML sampling; exclusive GPU lock"
        ),
        "points": {},
    }
    for key in sorted(groups):
        group = groups[key]
        assert len(group) == 5, (key, len(group))
        assert all(row["n_ok"] == row["n_requests"] for row in group), key
        shape, concurrency = key
        name = shape if shape != "decode" else f"decode_c{concurrency}"
        point = {
            "n": len(group),
            "ttft_median_s": statistics.median(row["ttft_p50_s"] for row in group),
            "ttft_samples_s": [row["ttft_p50_s"] for row in group],
            "wall_median_s": statistics.median(row["wall_s"] for row in group),
            "wall_samples_s": [row["wall_s"] for row in group],
        }
        if shape == "decode":
            point.update(
                {
                    "total_window_tok_s_median": statistics.median(
                        row["total_window_tok_s"] for row in group
                    ),
                    "total_window_tok_s_samples": [
                        row["total_window_tok_s"] for row in group
                    ],
                    "decode_window_tok_s_median": statistics.median(
                        row["decode_window_tok_s"] for row in group
                    ),
                    "decode_window_tok_s_samples": [
                        row["decode_window_tok_s"] for row in group
                    ],
                }
            )
        receipt["points"][name] = point
    receipt["verdict"] = "PASS"
    write_receipt(receipt, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("matrix", "correctness"):
        sub = subparsers.add_parser(name)
        sub.add_argument("root", type=pathlib.Path)
        sub.add_argument("destination", type=pathlib.Path)
    bench = subparsers.add_parser("bench")
    bench.add_argument("points", type=pathlib.Path)
    bench.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "matrix":
        reduce_matrix(args.root, args.destination)
    elif args.command == "correctness":
        reduce_correctness(args.root, args.destination)
    else:
        reduce_bench(args.points, args.destination)


if __name__ == "__main__":
    main()
