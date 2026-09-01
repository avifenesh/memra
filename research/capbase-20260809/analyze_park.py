#!/usr/bin/env python3
"""Gate the full-context park plus 8k pressure receipt."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


RECLAIM = re.compile(
    r"reclaim-on-defer: evicted (\d+) plain \+ (\d+) spec .* free ([0-9.]+)MB -> ([0-9.]+)MB"
)
FAILURE = re.compile(r"CUDA_ERROR|out of memory|panicked at|memory allocation.*failed", re.I)


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
    summary = next(row for row in rows if row.get("kind") == "summary")
    parked_entries = (summary.get("parked_plain_entries") or 0) + (
        summary.get("parked_spec_entries") or 0
    )
    if not summary.get("park_ok") or parked_entries < 1:
        failures.append("full-context request did not leave a parked session")
    if summary.get("burst_ok") != summary.get("burst_n"):
        failures.append("not all c=4 pressure requests completed")
    if summary.get("step_oom_parks") != 0:
        failures.append(f"step_oom_parks={summary.get('step_oom_parks')}")

    reclaim = []
    defer_lines = []
    captured_failures = []
    for lineno, line in enumerate(
        (args.root / "server.log").read_text(errors="replace").splitlines(), 1
    ):
        if match := RECLAIM.search(line):
            reclaim.append(
                {
                    "line": lineno,
                    "plain_evicted": int(match.group(1)),
                    "spec_evicted": int(match.group(2)),
                    "free_before_mb": float(match.group(3)),
                    "free_after_mb": float(match.group(4)),
                }
            )
        if "VRAM defer" in line:
            defer_lines.append(lineno)
        if FAILURE.search(line):
            captured_failures.append({"line": lineno, "text": line})

    if not reclaim:
        failures.append("reclaim-on-defer did not evict the parked session")
    elif reclaim[0]["plain_evicted"] + reclaim[0]["spec_evicted"] < 1:
        failures.append("reclaim-on-defer logged zero evictions")
    if reclaim and defer_lines and min(defer_lines) < reclaim[0]["line"]:
        failures.append("VRAM defer preceded reclaim-on-defer")
    if captured_failures:
        failures.append("captured CUDA/OOM/panic failure line")

    receipt = {
        "n": 1,
        "thermal_regime": "continuous one-second nvidia-smi sampling under one exclusive lock",
        "parked_entries_after_park": parked_entries,
        "reclaim_events": reclaim,
        "vram_defer_lines": defer_lines,
        "burst_ok": summary.get("burst_ok"),
        "burst_n": summary.get("burst_n"),
        "service_order": summary.get("service_order"),
        "ttfb_span_s": summary.get("ttfb_span_s"),
        "captured_failure_lines": captured_failures,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
