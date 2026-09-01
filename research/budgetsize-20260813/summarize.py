#!/usr/bin/env python3
"""Fail-closed reduction for the sealed cx-budgetsize local receipts."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SEQUENTIAL = {
    "A": ["local-ab/01-a-r1", "local-ab/04-a-r2", "local-ab/06-a-r3"],
    "B": ["local-ab/03-b-r1", "local-ab/05-b-r2", "local-ab/07-b-r3"],
}
EXPLICIT = {
    "A": "local-explicit/08-explicit-a-r1",
    "B": "local-explicit/09-explicit-b-r1",
}
C64 = "local-c64/10-b-c64-r1"
FAILURE_KEYS = (
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    if not rows:
        raise ValueError(f"{path}: empty")
    return rows


def gpu_samples(path: Path) -> dict[str, int]:
    clocks: list[int] = []
    used_mib: list[int] = []
    free_mib: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 12:
            raise ValueError(f"{path}:{line_number}: expected 12 CSV fields")
        clocks.append(int(fields[6]))
        used_mib.append(int(fields[9]))
        free_mib.append(int(fields[10]))
    if not clocks or min(clocks) < 210 or max(clocks) > 1200:
        raise ValueError(f"{path}: samples escaped the locked 210-1200 MHz regime")
    return {
        "n": len(clocks),
        "clock_min_mhz": min(clocks),
        "clock_max_mhz": max(clocks),
        "gpu_used_peak_mib": max(used_mib),
        "gpu_free_min_mib": min(free_mib),
    }


def load_cell(relative: str, expected_arm: str, binary_sha256: str) -> dict[str, Any]:
    cell = RAW / relative
    required = (
        "replay.jsonl",
        "replay.exit",
        "orchestrator.log",
        "preflight.log",
        "server-failure-scan.log",
        "gpu-250ms.csv",
    )
    missing = [name for name in required if not (cell / name).is_file()]
    if missing:
        raise ValueError(f"{relative}: missing {missing}")
    if (cell / "replay.exit").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"{relative}: replay did not exit zero")
    if (cell / "server-failure-scan.log").stat().st_size:
        raise ValueError(f"{relative}: non-empty server failure scan")
    cell_pass = "CELL_PASS" in (cell / "orchestrator.log").read_text(encoding="utf-8")
    recovered_a1 = False
    if not cell_pass:
        recovery_files = (
            "RECOVERY.md",
            "clock-validation-recovery.log",
            "recovery-audit.log",
        )
        recovered_a1 = (
            relative == "local-ab/01-a-r1"
            and all((cell / name).is_file() for name in recovery_files)
            and (cell / "clock-validation-recovery.log")
            .read_text(encoding="utf-8")
            .strip()
            == "samples=181 min_sm_mhz=210 max_sm_mhz=1192 escapes=0"
            and (cell / "recovery-audit.log").read_text(encoding="utf-8").strip()
            == "A1_RECOVERY_AUDIT_PASS"
        )
        if not recovered_a1:
            raise ValueError(f"{relative}: missing CELL_PASS or exact A1 recovery receipt")
    if binary_sha256 not in (cell / "preflight.log").read_text(encoding="utf-8"):
        raise ValueError(f"{relative}: frozen binary hash absent from preflight")

    rows = read_jsonl(cell / "replay.jsonl")
    summary = rows[-1]
    if (
        summary.get("kind") != "summary"
        or summary.get("arm") != expected_arm
        or summary.get("verdict") != "PASS"
        or summary.get("failures")
    ):
        raise ValueError(f"{relative}: invalid terminal summary")
    requests = [row for row in rows if row.get("kind") == "request"]
    return {
        "path": f"raw/{relative}",
        "summary": summary,
        "requests": requests,
        "gpu": gpu_samples(cell / "gpu-250ms.csv"),
        "server_log": (cell / "server.log").read_text(encoding="utf-8"),
        "completion_receipt": "a1-clock-validator-recovery" if recovered_a1 else "CELL_PASS",
    }


def load_failed_c64(relative: str, binary_sha256: str) -> dict[str, Any]:
    """Accept only the exact observed c64 admission/exactness failure class."""
    cell = RAW / relative
    required = (
        "replay.jsonl",
        "replay.exit",
        "orchestrator.log",
        "preflight.log",
        "server-failure-scan.log",
        "server.log",
        "gpu-250ms.csv",
    )
    missing = [name for name in required if not (cell / name).is_file()]
    if missing:
        raise ValueError(f"{relative}: missing {missing}")
    if (cell / "replay.exit").read_text(encoding="utf-8").strip() != "1":
        raise ValueError(f"{relative}: expected replay exit 1")
    if (cell / "server-failure-scan.log").stat().st_size:
        raise ValueError(f"{relative}: non-empty server failure scan")
    if binary_sha256 not in (cell / "preflight.log").read_text(encoding="utf-8"):
        raise ValueError(f"{relative}: frozen binary hash absent from preflight")

    rows = read_jsonl(cell / "replay.jsonl")
    summary = rows[-1]
    requests = [row for row in rows if row.get("kind") == "request"]
    if (
        summary.get("kind") != "summary"
        or summary.get("arm") != "derived-c64"
        or summary.get("mode") != "c64"
        or summary.get("verdict") != "FAIL"
        or int(summary.get("requests") or 0) != 64
        or len(requests) != 64
    ):
        raise ValueError(f"{relative}: not the complete c64 failure cell")
    if any(
        not request.get("ok")
        or int(request.get("prompt_tokens") or 0) != 4_860
        or int(request.get("completion_tokens") or 0) != 60
        or int(request.get("cached_tokens") or 0) != 4_860
        for request in requests
    ):
        raise ValueError(f"{relative}: a c64 request failed its transport/cache shape")
    golden = str(summary["golden_text_sha256"])
    if any(str(request.get("text_sha256")) == golden for request in requests):
        raise ValueError(f"{relative}: expected every concurrent stream to differ from seed")

    expected_counters = {
        "admitted": 64,
        "completed": 64,
        "prompt_tokens_in": 311_040,
        "cached_tokens_in": 311_040,
        "prefix_cache_hits": 64,
        "prefix_cache_misses": 0,
        "prefix_cache_inserts": 0,
        "prefix_cache_evictions": 0,
        "prefix_cache_skips_budget": 0,
        "prefix_cache_skips_pinned": 0,
        "prefix_cache_hit_tokens": 311_040,
        "prefix_cache_entries": 0,
        "prefix_cache_bytes": 0,
        "admission_session_defers": 0,
        "step_oom_parks": 0,
    }
    counters = summary["counter_deltas"]
    for key, expected in expected_counters.items():
        if int(counters[key]) != expected:
            raise ValueError(f"{relative}: {key}={counters[key]} != {expected}")
    if int(counters["admission_vram_defers"]) <= 0:
        raise ValueError(f"{relative}: expected the observed VRAM-deferral failure")
    seed = summary["seed_counter_deltas"]
    for key, expected in {
        "completed": 1,
        "prefix_cache_misses": 1,
        "prefix_cache_inserts": 1,
        "prefix_cache_bytes": 301_215_744,
        "prefix_cache_entries": 1,
        "admission_session_defers": 0,
        "admission_vram_defers": 0,
        "step_oom_parks": 0,
    }.items():
        if int(seed[key]) != expected:
            raise ValueError(f"{relative}: seed {key}={seed[key]} != {expected}")
    failures = list(summary.get("failures") or [])
    if (
        len(failures) != 65
        or sum("greedy output hash differs" in failure for failure in failures) != 64
        or sum(failure.startswith("admission_vram_defers delta=") for failure in failures) != 1
    ):
        raise ValueError(f"{relative}: unexpected c64 failure set")
    server_log = (cell / "server.log").read_text(encoding="utf-8")
    if "[admit-oom] VRAM defer:" not in server_log:
        raise ValueError(f"{relative}: missing quoted VRAM-deferral receipt")
    return {
        "path": f"raw/{relative}",
        "summary": summary,
        "requests": requests,
        "gpu": gpu_samples(cell / "gpu-250ms.csv"),
        "server_log": server_log,
        "completion_receipt": "EXPECTED_FAIL",
    }


def total_counters(cells: list[dict[str, Any]]) -> dict[str, int]:
    keys = cells[0]["summary"]["counter_deltas"].keys()
    return {
        key: sum(int(cell["summary"]["counter_deltas"][key]) for cell in cells)
        for key in keys
    }


def sequential_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    counters = total_counters(cells)
    requests = [request for cell in cells for request in cell["requests"]]
    hit_requests = [request for request in requests if int(request["cached_tokens"]) > 0]
    misses = counters["prefix_cache_misses"]
    hits = counters["prefix_cache_hits"]
    return {
        "n_boots": len(cells),
        "n_requests": len(requests),
        "hit_rate": hits / (hits + misses),
        "counter_totals": counters,
        "cold_request_ttft_ms": distribution(
            [float(request["ttft_ms"]) for request in requests if not request["cached_tokens"]]
        ),
        "first_request_ttft_ms": distribution(
            [float(cell["summary"]["cold_ttft_ms"]) for cell in cells]
        ),
        "first_hit_ttft_ms": (
            distribution(
                [
                    float(cell["summary"]["first_hit_ttft_ms"])
                    for cell in cells
                    if cell["summary"]["first_hit_ttft_ms"] is not None
                ]
            )
            if any(cell["summary"]["first_hit_ttft_ms"] is not None for cell in cells)
            else None
        ),
        "hit_request_ttft_ms": (
            distribution([float(request["ttft_ms"]) for request in hit_requests])
            if hit_requests
            else None
        ),
        "output_sha256": sorted({str(request["text_sha256"]) for request in requests}),
        "cells": [
            {
                "path": cell["path"],
                "repetition": cell["summary"]["repetition"],
                "gpu": cell["gpu"],
                "completion_receipt": cell["completion_receipt"],
            }
            for cell in cells
        ],
    }


def request_behavior(cell: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: request[key]
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "text_sha256")
        }
        for request in cell["requests"]
    ]


def main() -> int:
    protocol = json.loads((ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    hashes = {arm: protocol["arms"][arm]["binary_sha256"] for arm in ("A", "B")}
    a_cells = [load_cell(path, "baseline", hashes["A"]) for path in SEQUENTIAL["A"]]
    b_cells = [load_cell(path, "derived", hashes["B"]) for path in SEQUENTIAL["B"]]
    explicit_a = load_cell(EXPLICIT["A"], "explicit4096-a", hashes["A"])
    explicit_b = load_cell(EXPLICIT["B"], "explicit4096-b", hashes["B"])
    c64 = load_failed_c64(C64, hashes["B"])

    arm_a = sequential_summary(a_cells)
    arm_b = sequential_summary(b_cells)
    if arm_a["hit_rate"] != 0 or (
        arm_a["counter_totals"]["prefix_cache_skips_budget"]
        != arm_a["counter_totals"]["prefix_cache_misses"]
    ):
        raise ValueError("arm A did not reproduce the default-budget refusal")
    if (
        arm_b["hit_rate"] < 0.8
        or arm_b["counter_totals"]["prefix_cache_inserts"] < 3
        or arm_b["counter_totals"]["prefix_cache_skips_budget"] != 0
    ):
        raise ValueError("arm B did not satisfy the derived-budget acceptance criteria")
    if arm_a["output_sha256"] != arm_b["output_sha256"]:
        raise ValueError("A/B greedy output bytes differ")

    geometry_pattern = re.compile(
        r"derived: 2 x 400162816 B max entry.*requested 800325632 B; "
        r"boot driver free ([0-9]+) B, post-reserve clamp ([0-9]+) B"
    )
    boot_geometry = []
    for cell in b_cells:
        match = geometry_pattern.search(cell["server_log"])
        if not match:
            raise ValueError(f"{cell['path']}: missing exact derived geometry receipt")
        free_bytes, clamp_bytes = (int(value) for value in match.groups())
        if 800_325_632 > free_bytes or 800_325_632 > clamp_bytes:
            raise ValueError(f"{cell['path']}: derived budget exceeds boot clamp")
        boot_geometry.append(
            {
                "path": cell["path"],
                "entry_bytes": 400_162_816,
                "entry_count": 2,
                "requested_budget_bytes": 800_325_632,
                "boot_driver_free_bytes": free_bytes,
                "post_reserve_clamp_bytes": clamp_bytes,
            }
        )

    explicit_equal = (
        request_behavior(explicit_a) == request_behavior(explicit_b)
        and explicit_a["summary"]["counter_deltas"]
        == explicit_b["summary"]["counter_deltas"]
    )
    if not explicit_equal:
        raise ValueError("explicit 4096 MiB A/B behavior differs")

    c64_counters = c64["summary"]["counter_deltas"]
    c64_pass = (
        all(int(c64_counters[key]) == 0 for key in FAILURE_KEYS)
        and int(c64_counters["prefix_cache_evictions"]) == 0
        and not c64["summary"]["failures"]
    )

    result = {
        "schema": "memra.cx-budgetsize.summary.v1",
        "thermal_regime": protocol["local_thermal_regime"],
        "arms": {
            "A": {"protocol": protocol["arms"]["A"], "measurement": arm_a},
            "B": {"protocol": protocol["arms"]["B"], "measurement": arm_b},
        },
        "derived_boot_geometry": boot_geometry,
        "explicit_4096_mib": {
            "behavior_equal": explicit_equal,
            "a_path": explicit_a["path"],
            "b_path": explicit_b["path"],
            "counter_deltas": explicit_a["summary"]["counter_deltas"],
            "output_sha256": sorted(
                {request["text_sha256"] for request in explicit_a["requests"]}
            ),
        },
        "c64": {
            "path": c64["path"],
            "requests": c64["summary"]["requests"],
            "counter_deltas": c64_counters,
            "seed_counter_deltas": c64["summary"]["seed_counter_deltas"],
            "ttft_median_ms": c64["summary"]["ttft_median_ms"],
            "ttft_max_ms": c64["summary"]["ttft_max_ms"],
            "cuda_driver_free_bytes_after": c64["summary"][
                "cuda_driver_free_bytes_after"
            ],
            "cuda_pool_cached_bytes_after": c64["summary"][
                "cuda_pool_cached_bytes_after"
            ],
            "output_sha256_counts": dict(sorted(Counter(
                str(request["text_sha256"]) for request in c64["requests"]
            ).items())),
            "seed_output_sha256": c64["summary"]["golden_text_sha256"],
            "failure_count": len(c64["summary"]["failures"]),
            "verdict": "PASS" if c64_pass else "FAIL",
            "gpu": c64["gpu"],
        },
        "acceptance": {
            "arm_a_reproduces_defect": "PASS",
            "arm_b_n3_hit_and_geometry": "PASS",
            "explicit_4096_mib_parity": "PASS",
            "c64_zero_defers_and_oom_parks": "PASS" if c64_pass else "FAIL",
            "overall": "PASS" if c64_pass else "FAIL",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if c64_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
