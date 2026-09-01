#!/usr/bin/env python3
"""Extract the honest one-run box1 capacity row from raw receipts."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

from analyze_admission import COST


DEFER = re.compile(
    r"\[admit-oom\] VRAM defer: (\d+) active, effective free ([0-9.]+)MB .* "
    r"< cost ([0-9.]+)MB \+ reserve ([0-9.]+)MB"
)


def gpu_peaks(path: pathlib.Path) -> dict:
    per_gpu: dict[int, dict] = {}
    combined: dict[str, float] = {}
    with path.open(newline="") as source:
        for fields in csv.reader(source):
            if len(fields) != 8:
                continue
            try:
                stamp = fields[0].strip()
                index = int(fields[1].strip())
                used = float(fields[6].strip())
                free = float(fields[7].strip())
                temp = float(fields[3].strip())
            except ValueError:
                continue
            combined[stamp] = combined.get(stamp, 0.0) + used
            if index not in per_gpu or used > per_gpu[index]["used_mib"]:
                per_gpu[index] = {
                    "used_mib": used,
                    "free_mib": free,
                    "temperature_c": temp,
                    "timestamp": stamp,
                }
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
    requests = [row for row in rows if row.get("kind") == "request"]
    run = next((row for row in rows if row.get("kind") == "run"), {})
    if not requests or any(not row.get("ok") for row in requests):
        failures.append("not all capacity requests completed cleanly")

    metrics = json.loads((args.root / "metrics-final.json").read_text())
    if metrics.get("step_oom_parks") != 0:
        failures.append(f"step_oom_parks={metrics.get('step_oom_parks')}")

    log_lines = (args.root / "server.log").read_text(errors="replace").splitlines()
    defers = []
    costs = []
    captured_failures = []
    for lineno, line in enumerate(log_lines, 1):
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
        if match := COST.search(line):
            costs.append(
                {
                    "line": lineno,
                    "effective_ctx_cap": int(match.group(1)),
                    "path": match.group(2),
                    "bytes_per_token": int(match.group(3)),
                    "cost_mb": float(match.group(5)),
                }
            )
        if re.search(r"CUDA_ERROR|out of memory|panicked at|memory allocation.*failed", line, re.I):
            captured_failures.append({"line": lineno, "text": line})

    if not defers:
        failures.append(
            f"no admission defer at offered concurrency {run.get('concurrency')}; capacity is only a lower bound"
        )
    if captured_failures:
        failures.append("captured CUDA/OOM/panic failure line")

    receipt = {
        "n": 1,
        "thermal_regime": "continuous one-second nvidia-smi sampling under one exclusive lock",
        "server_ctx": run.get("server_ctx"),
        "requested_max_ctx": run.get("requested_max_ctx"),
        "offered_concurrency": run.get("concurrency"),
        "requests_ok": sum(bool(row.get("ok")) for row in requests),
        "requests_n": len(requests),
        "first_defer": defers[0] if defers else None,
        "admitted_before_first_defer": defers[0]["active"] if defers else None,
        "all_defer_lines": defers,
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
