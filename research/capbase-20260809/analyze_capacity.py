#!/usr/bin/env python3
"""Extract one honest capacity cell from its raw box1 receipt."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re


COST = re.compile(
    r"\[admission\] request cost: .* ctx=(\d+) path=(\w+) = (\d+) "
    r"B/token x ctx \+ ([0-9.]+)MB fixed = ([0-9.]+)MB"
)
DEFER = re.compile(
    r"\[admit-oom\] VRAM defer: (\d+) active, effective free ([0-9.]+)MB .* "
    r"< cost ([0-9.]+)MB \+ reserve ([0-9.]+)MB"
)
FAILURE = re.compile(r"CUDA_ERROR|out of memory|panicked at|memory allocation.*failed", re.I)


def gpu_peaks(path: pathlib.Path) -> dict:
    per_gpu: dict[int, dict] = {}
    combined: dict[str, float] = {}
    max_temperature: dict[int, float] = {}
    with path.open(newline="") as source:
        for fields in csv.reader(source):
            if len(fields) != 8:
                continue
            try:
                stamp = fields[0].strip()
                index = int(fields[1].strip())
                temperature = float(fields[3].strip())
                used = float(fields[6].strip())
                free = float(fields[7].strip())
            except ValueError:
                continue
            combined[stamp] = combined.get(stamp, 0.0) + used
            max_temperature[index] = max(max_temperature.get(index, temperature), temperature)
            if index not in per_gpu or used > per_gpu[index]["used_mib"]:
                per_gpu[index] = {
                    "used_mib": used,
                    "free_mib": free,
                    "temperature_c_at_peak": temperature,
                    "timestamp": stamp,
                }
    for index, value in per_gpu.items():
        value["max_temperature_c"] = max_temperature[index]
    return {
        "per_gpu": {str(index): value for index, value in sorted(per_gpu.items())},
        "combined_peak_used_mib": max(combined.values(), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []

    rows = [
        json.loads(line)
        for line in (args.root / "requests.jsonl").read_text().splitlines()
        if line.strip()
    ]
    run = next(row for row in rows if row.get("kind") == "run")
    requests = [row for row in rows if row.get("kind") == "request"]
    client_summary = next(row for row in rows if row.get("kind") == "summary")
    if not requests or any(not row.get("ok") for row in requests):
        failures.append("not all offered requests completed cleanly")

    metrics = json.loads((args.root / "metrics-final.json").read_text())
    if metrics.get("step_oom_parks") != 0:
        failures.append(f"step_oom_parks={metrics.get('step_oom_parks')}")

    costs = []
    defers = []
    captured_failures = []
    for lineno, line in enumerate(
        (args.root / "server.log").read_text(errors="replace").splitlines(), 1
    ):
        if match := COST.search(line):
            costs.append(
                {
                    "line": lineno,
                    "effective_ctx_cap": int(match.group(1)),
                    "path": match.group(2),
                    "bytes_per_token": int(match.group(3)),
                    "fixed_mb": float(match.group(4)),
                    "cost_mb": float(match.group(5)),
                }
            )
        if match := DEFER.search(line):
            defers.append(
                {
                    "line": lineno,
                    "active": int(match.group(1)),
                    "effective_free_mb": float(match.group(2)),
                    "cost_mb": float(match.group(3)),
                    "reserve_mb": float(match.group(4)),
                }
            )
        if FAILURE.search(line):
            captured_failures.append({"line": lineno, "text": line})

    requested = run["requested_max_ctx"]
    if requested not in {row["effective_ctx_cap"] for row in costs}:
        failures.append(f"request-cost log lacks explicit ctx={requested}")
    if captured_failures:
        failures.append("captured CUDA/OOM/panic failure line")

    if defers:
        result = {
            "kind": "first_defer",
            "sessions": defers[0]["active"],
        }
    elif client_summary.get("peak_active_sessions_sampled") == run["offered_concurrency"]:
        result = {
            "kind": "lower_bound_no_defer",
            "sessions": run["offered_concurrency"],
        }
    else:
        result = {
            "kind": "inconclusive_no_defer",
            "sessions": client_summary.get("peak_active_sessions_sampled"),
        }
        failures.append("no defer and sampled active peak did not reach offered concurrency")

    receipt = {
        "n": 1,
        "thermal_regime": "continuous one-second nvidia-smi sampling under one exclusive lock",
        "server_ctx": run["server_ctx"],
        "requested_max_ctx": requested,
        "offered_concurrency": run["offered_concurrency"],
        "capacity_result": result,
        "first_defer": defers[0] if defers else None,
        "requests_ok": sum(bool(row.get("ok")) for row in requests),
        "requests_n": len(requests),
        "peak_active_sessions_sampled": client_summary.get("peak_active_sessions_sampled"),
        "request_start_spread_ms": client_summary.get("request_start_spread_ms"),
        "request_cost_lines": costs,
        "admission_vram_defers_metric": metrics.get("admission_vram_defers"),
        "step_oom_parks": metrics.get("step_oom_parks"),
        "nvidia_smi_peak": gpu_peaks(args.root / "gpu.csv"),
        "captured_failure_lines": captured_failures,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
