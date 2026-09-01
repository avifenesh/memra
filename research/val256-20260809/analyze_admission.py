#!/usr/bin/env python3
"""Check the box1 mixed-context admission receipts without repairing failures."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


COST = re.compile(
    r"\[admission\] request cost: .* ctx=(\d+) path=(\w+) = (\d+) "
    r"B/token x ctx \+ ([0-9.]+)MB fixed = ([0-9.]+)MB"
)


def jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def log_receipt(path: pathlib.Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    costs = []
    for lineno, line in enumerate(lines, 1):
        match = COST.search(line)
        if match:
            costs.append(
                {
                    "line": lineno,
                    "ctx_cap": int(match.group(1)),
                    "path": match.group(2),
                    "bytes_per_token": int(match.group(3)),
                    "fixed_mb": float(match.group(4)),
                    "cost_mb": float(match.group(5)),
                }
            )
    return {
        "costs": costs,
        "reclaim_lines": [i for i, line in enumerate(lines, 1) if "reclaim-on-defer" in line],
        "defer_lines": [i for i, line in enumerate(lines, 1) if "VRAM defer" in line],
        "step_oom_lines": [
            i
            for i, line in enumerate(lines, 1)
            if "step OOM" in line or "step-time CUDA OOM" in line
        ],
        "failure_lines": [
            {"line": i, "text": line}
            for i, line in enumerate(lines, 1)
            if re.search(r"CUDA_ERROR|out of memory|panicked at|memory allocation.*failed", line, re.I)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []
    orders = {}
    for order in ("forward", "inverse"):
        arm = args.root / order
        rows = jsonl(arm / "requests.jsonl")
        requests = [row for row in rows if row.get("kind") == "request"]
        final_metrics = json.loads((arm / "metrics-final.json").read_text())
        logs = log_receipt(arm / "server.log")
        orders[order] = {
            "requests_ok": sum(bool(row.get("ok")) for row in requests),
            "requests_n": len(requests),
            "admission_vram_defers": final_metrics.get("admission_vram_defers"),
            "step_oom_parks": final_metrics.get("step_oom_parks"),
            **logs,
        }
        if not requests or any(not row.get("ok") for row in requests):
            failures.append(f"{order}: not all requests completed cleanly")
        if final_metrics.get("step_oom_parks") != 0 or logs["step_oom_lines"]:
            failures.append(f"{order}: step-OOM park observed")
        if logs["failure_lines"]:
            failures.append(f"{order}: captured CUDA/OOM/panic failure line")

    forward = orders["forward"]
    contexts = {row["ctx_cap"]: row["cost_mb"] for row in forward["costs"]}
    expected = (8192, 131072, 262144)
    if any(ctx not in contexts for ctx in expected):
        failures.append(
            f"forward: request-cost log lacks distinct 8k/128k/256k ctx caps; saw {sorted(contexts)}"
        )
    elif not (contexts[8192] < contexts[131072] < contexts[262144]):
        failures.append("forward: request cost did not increase with ctx cap")
    if not forward["reclaim_lines"]:
        failures.append("forward: reclaim-on-defer never fired")
    elif forward["defer_lines"] and min(forward["defer_lines"]) < min(forward["reclaim_lines"]):
        failures.append("forward: a VRAM defer was logged before reclaim-on-defer")

    inverse = orders["inverse"]
    if inverse["admission_vram_defers"] != 0 or inverse["defer_lines"]:
        failures.append("inverse: small requests were over-gated after the 256k-first request")

    receipt = {
        "n_per_order": 1,
        "server_ctx": 262144,
        "expected_request_ctx_caps": list(expected),
        "orders": orders,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
