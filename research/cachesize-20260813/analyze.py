#!/usr/bin/env python3
"""Reduce the box1 cache-size campaign into reproducible capacity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


BUDGETS = [1024, 4096, 8192, 16384, 32768, 49152]
LEVELS = {
    "q27": [4, 5, 6, 7, 8, 12, 16],
    "q35": [4, 5, 6, 7, 8, 16, 24, 32, 40],
}
ENTRY_PREFIX_TOKENS = [4096, 4860, 8192]
REPETITIONS = 5
WORKING_SET_ENTRIES = 96
SOLD_HIT_P95_MS = {"q27": 22.0, "q35": 11.0}
MODEL_KNEE = {"q27": 16, "q35": 40}
LIVE_USED_MIB = 36_880
LIVE_TOTAL_MIB = 97_887
LIVE_PREFIX_CACHE_MIB = 4_096
FROZEN_REPLAY_SHA256 = "91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b"
FROZEN_WORKLOAD_SHA256 = "85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34"
PROMPT_SHA256 = "eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb"
RUNTIME_SOURCE = "18885ec479d897a3e8c42b0d408a71fa3edaa708"
CYCLE_BASE_SEED = 3407
REUSED_BOOT_KEYS = {("q27", 1024, 1), ("q27", 4096, 1)}
EXPECTED_EXCLUDED_BOOT = ("q27", 8192, 1)
PHYSICAL_GPU0_UUID = "GPU-54dd2b6f-9311-dd31-672b-60be2ed28a79"
EOS_TEXT_SHA256 = "ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73"
COST_EVENT = re.compile(
    r"CUDA_ERROR_OUT_OF_MEMORY|out of memory|\[admit-oom\].*(?:VRAM defer|step OOM)|"
    r"\[prefix-cache\].*(?:alloc OOM|allocation failed|grow OOM)|"
    r"(?:session cache|cache|spec session|plain-affinity grow) alloc(?:ation)? fail|"
    r"reclaim-on-(?:defer|alloc-oom)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(value)
    return rows


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = list(values)
    return {
        "n": len(samples),
        "p50_ms": statistics.median(samples) if samples else None,
        "p95_ms": nearest_rank(samples, 0.95),
        "min_ms": min(samples) if samples else None,
        "max_ms": max(samples) if samples else None,
    }


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "MANIFEST.sha256"
    entries = 0
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"{manifest}:{number}: malformed entry") from error
        path = root / relative.removeprefix("./")
        if sha256_file(path) != digest:
            raise ValueError(f"manifest mismatch: {relative}")
        entries += 1
    return {"entries": entries, "sha256": sha256_file(manifest)}


def parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_boot_gpu(path: Path) -> dict[str, float]:
    """Parse timestamp,index,pstate,temp,power,limit,sm,mem,total,used,free,util."""
    maxima: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 12:
            continue
        for key, index in (
            ("max_temperature_c", 3),
            ("max_power_w", 4),
            ("max_clock_mhz", 6),
            ("total_memory_mib", 8),
            ("max_memory_used_mib", 9),
            ("max_utilization_percent", 11),
        ):
            value = parse_float(fields[index])
            if value is not None:
                maxima[key] = max(maxima.get(key, value), value)
    if "max_memory_used_mib" not in maxima:
        raise ValueError(f"{path}: no GPU samples")
    return maxima


def parse_snapshot_csv(row: dict[str, Any]) -> dict[str, float]:
    fields = [field.strip() for field in str(row["csv"]).split(",")]
    names = list(row["query_fields"])
    if len(fields) != len(names):
        raise ValueError(f"GPU snapshot field mismatch: {row}")
    parsed: dict[str, float] = {}
    for name, value in zip(names, fields, strict=True):
        numeric = parse_float(value)
        if numeric is not None:
            parsed[name] = numeric
    return parsed


def parse_all_gpu_monitor(path: Path) -> dict[str, Any]:
    timestamps: list[str] = []
    rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", line):
            timestamps.append(line)
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        try:
            gpu_index = int(fields[0])
            rows[gpu_index].append({
                "temperature_c": float(fields[3]),
                "power_w": float(fields[4]),
                "total_memory_mib": float(fields[5]),
                "memory_used_mib": float(fields[6]),
                "utilization_percent": float(fields[7]),
            })
        except ValueError:
            continue
    if not timestamps or not rows[0] or not rows[1]:
        raise ValueError(f"{path}: incomplete two-GPU sidecar monitor")
    if max(row["memory_used_mib"] for row in rows[1]) != 0:
        raise ValueError(f"{path}: GPU1 memory use was not zero throughout the sidecar")
    if max(row["utilization_percent"] for row in rows[1]) != 0:
        raise ValueError(f"{path}: GPU1 utilization was not zero throughout the sidecar")
    return {
        "started_utc": timestamps[0],
        "ended_utc": timestamps[-1],
        "timestamp_samples": len(timestamps),
        "gpu0_rows": len(rows[0]),
        "gpu1_rows": len(rows[1]),
        "gpu1_max_memory_used_mib": max(row["memory_used_mib"] for row in rows[1]),
        "gpu1_max_utilization_percent": max(row["utilization_percent"] for row in rows[1]),
    }


def campaign_timing(path: Path, require_complete: bool) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    acquired = re.findall(r"CACHESIZE_LOCK_ACQUIRED ts=([^ ]+)", text)
    complete = re.findall(r"CACHESIZE_COMPLETE ts=([^ ]+)", text)
    if len(acquired) != 1 or len(complete) > 1:
        raise ValueError(f"{path}: invalid lock timing")
    if require_complete and len(complete) != 1:
        raise ValueError(f"{path}: complete lock timing is absent")
    start = datetime.fromisoformat(acquired[0].replace("Z", "+00:00"))
    result: dict[str, Any] = {
        "lock_acquired_utc": acquired[0],
        "completed_utc": complete[0] if complete else None,
        "lock_hold_seconds": None,
        "complete": bool(complete),
    }
    if complete:
        end = datetime.fromisoformat(complete[0].replace("Z", "+00:00"))
        result["lock_hold_seconds"] = (end - start).total_seconds()
    return result


def snapshot_time(path: Path) -> datetime:
    match = re.search(
        r"^ts=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$",
        path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{path}: timestamp is absent")
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))


def sweep_path(raw: Path, model: str, budget: int, rep: int) -> Path:
    return raw / "campaign" / f"r{rep:02d}-{model}-b{budget:05d}" / "sweep.jsonl"


def validate_boot_order(
    raws: list[Path], excluded_paths: set[Path]
) -> dict[str, Any]:
    observed: list[tuple[str, int, int]] = []
    segments: list[dict[str, Any]] = []
    for raw in raws:
        text = (raw / "orchestrator.log").read_text(
            encoding="utf-8", errors="replace"
        )
        segment_observed: list[tuple[str, int, int]] = []
        segment_excluded = 0
        for model, budget_text, rep_text in re.findall(
            r"BUDGET_BOOT_START model=(q27|q35) budget_mb=(\d+) rep=(\d+)", text
        ):
            budget = int(budget_text)
            rep = int(rep_text)
            if sweep_path(raw, model, budget, rep).resolve() in excluded_paths:
                segment_excluded += 1
                continue
            segment_observed.append((model, budget, rep))
        observed.extend(segment_observed)
        segments.append({
            "raw": str(raw),
            "valid_boot_starts": len(segment_observed),
            "excluded_boot_starts": segment_excluded,
        })
    expected: list[tuple[str, int, int]] = []
    for rep in range(1, REPETITIONS + 1):
        if rep % 2:
            models = ("q27", "q35")
            budgets = BUDGETS
        else:
            models = ("q35", "q27")
            budgets = list(reversed(BUDGETS))
        expected.extend(
            (model, budget, rep) for model in models for budget in budgets
        )
    expected = [key for key in expected if key != EXPECTED_EXCLUDED_BOOT]
    if observed != expected:
        raise ValueError("budget/model boot order drift")
    return {
        "boots": len(observed),
        "segments": segments,
        "odd_repetitions": "q27 then q35, budgets ascending",
        "even_repetitions": "q35 then q27, budgets descending",
    }


def entry_measurements(raw: Path) -> tuple[dict[str, Any], dict[str, int]]:
    result: dict[str, Any] = {}
    sold_bytes: dict[str, int] = {}
    for model in LEVELS:
        path = raw / "entry-bytes" / model / "entry-bytes.jsonl"
        rows = read_jsonl(path)
        measurements = [row for row in rows if row.get("kind") == "entry_bytes"]
        summaries = [row for row in rows if row.get("kind") == "summary"]
        if len(measurements) != 3 or [int(row["prefix_tokens"]) for row in measurements] != ENTRY_PREFIX_TOKENS:
            raise ValueError(f"{path}: entry-measurement grid drift")
        if len(summaries) != 1 or summaries[0].get("verdict") != "PASS":
            raise ValueError(f"{path}: entry probe did not pass")
        if any(not row.get("clean") for row in measurements):
            raise ValueError(f"{path}: unclean entry measurement")
        by_tokens = {
            str(row["prefix_tokens"]): {
                "device_bytes": int(row["device_bytes"]),
                "device_mib": float(row["device_mib"]),
                "bytes_per_token": float(row["bytes_per_token"]),
                "counter_deltas": row["counter_deltas"],
            }
            for row in measurements
        }
        sold_bytes[model] = int(by_tokens["4860"]["device_bytes"])
        result[model] = by_tokens
    return result, sold_bytes


def median(values: Iterable[float | int]) -> float:
    return float(statistics.median(list(values)))


def expected_width_order(levels: list[int], rep: int) -> list[int]:
    index = rep - 1
    offset = index % len(levels)
    order = levels[offset:] + levels[:offset]
    return list(reversed(order)) if index % 2 else order


def validate_excluded_boot(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    protocols = [row for row in rows if row.get("kind") == "protocol"]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    if len(protocols) != 1 or len(summaries) != 1:
        raise ValueError(f"{path}: excluded boot lacks one protocol and summary")
    protocol, summary = protocols[0], summaries[0]
    key = (
        str(protocol.get("model")),
        int(protocol.get("budget_mb") or 0),
        int(protocol.get("repetition") or 0),
    )
    exit_code = (path.parent / "sweep.exit").read_text(encoding="utf-8").strip()
    failure_scan = (path.parent / "server-failure-scan.log").read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    failed_requests = [
        row
        for row in rows
        if row.get("kind") == "request" and not row.get("ok")
    ]
    if key != EXPECTED_EXCLUDED_BOOT:
        raise ValueError(f"{path}: unexpected excluded boot {key}")
    if exit_code != "1" or summary.get("verdict") != "FAIL" or not summary.get("failures"):
        raise ValueError(f"{path}: excluded boot is not a captured failure")
    if failure_scan:
        raise ValueError(f"{path}: excluded boot server failure scan is nonempty")
    if len(failed_requests) != 1:
        raise ValueError(f"{path}: expected one failed response receipt")
    failed = failed_requests[0]
    if (
        int(failed.get("prefix_id") or -1) != 87
        or int(failed.get("completion_tokens") or 0) != 11
        or failed.get("finish_reason") != "stop"
        or failed.get("text_sha256") != EOS_TEXT_SHA256
        or int(failed.get("cached_tokens") or 0) != 4860
    ):
        raise ValueError(f"{path}: excluded EOS receipt drift")
    return {
        "model": key[0],
        "budget_mb": key[1],
        "rep": key[2],
        "path": str(path),
        "sweep_exit": exit_code,
        "verdict": summary.get("verdict"),
        "failures": summary.get("failures"),
        "request_id": failed.get("request_id"),
        "completion_tokens": failed.get("completion_tokens"),
        "finish_reason": failed.get("finish_reason"),
        "text_sha256": failed.get("text_sha256"),
        "server_failure_scan_lines": 0,
    }


def reduce_campaign(
    raws: list[Path], sold_bytes: dict[str, int], excluded_paths: set[Path]
) -> dict[str, Any]:
    all_paths = [
        path
        for raw in raws
        for path in sorted((raw / "campaign").glob("r??-*-b*/sweep.jsonl"))
    ]
    resolved_paths = {path.resolve() for path in all_paths}
    if not excluded_paths <= resolved_paths:
        raise ValueError(
            f"excluded sweep paths are absent: {sorted(excluded_paths - resolved_paths)}"
        )
    excluded_boots = [
        validate_excluded_boot(path)
        for path in all_paths
        if path.resolve() in excluded_paths
    ]
    if len(excluded_boots) != 1:
        raise ValueError(f"expected one excluded boot, got {len(excluded_boots)}")
    paths = [path for path in all_paths if path.resolve() not in excluded_paths]
    expected_paths = len(BUDGETS) * len(LEVELS) * REPETITIONS - 1
    if len(paths) != expected_paths:
        raise ValueError(f"expected {expected_paths} sweep files, got {len(paths)}")

    cells: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    boots: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        rows = read_jsonl(path)
        protocols = [row for row in rows if row.get("kind") == "protocol"]
        seed_requests = [row for row in rows if row.get("kind") == "seed"]
        seeds = [row for row in rows if row.get("kind") == "seed_summary"]
        summaries = [row for row in rows if row.get("kind") == "summary"]
        snapshots = [row for row in rows if row.get("kind") == "gpu_snapshot"]
        if len(protocols) != 1 or len(seeds) != 1 or len(summaries) != 1:
            raise ValueError(f"{path}: expected one protocol, seed summary, and run summary")
        protocol, seed, summary = protocols[0], seeds[0], summaries[0]
        model = str(protocol["model"])
        budget = int(protocol["budget_mb"])
        rep = int(protocol["repetition"])
        key = (model, budget, rep)
        if key in seen:
            raise ValueError(f"duplicate boot: {key}")
        seen.add(key)
        if model not in LEVELS or budget not in BUDGETS or rep not in range(1, REPETITIONS + 1):
            raise ValueError(f"unexpected boot: {key}")
        if int(protocol["working_set_entries"]) != WORKING_SET_ENTRIES:
            raise ValueError(f"{path}: working-set drift")
        if int(protocol.get("seed_concurrency") or 0) != 1:
            raise ValueError(f"{path}: seed-concurrency drift")
        if not str(protocol.get("seed_method") or "").startswith("sequential"):
            raise ValueError(f"{path}: seed-method drift")
        if not str(protocol.get("budget_pairing") or "").startswith(
            "within one model and repetition"
        ):
            raise ValueError(f"{path}: budget-pairing contract drift")
        if "/tmp/memra-gpu-1.lock exclusion" not in str(
            protocol.get("thermal_regime") or ""
        ):
            raise ValueError(f"{path}: GPU1 lock contract drift")
        model_seed = 27 if model == "q27" else 35
        expected_cycle_seed = (
            CYCLE_BASE_SEED + model_seed * 1_000_003 + rep * 10_007
        )
        if int(protocol.get("working_set_cycle_seed") or 0) != expected_cycle_seed:
            raise ValueError(f"{path}: working-set cycle seed drift")
        if [int(value) for value in protocol.get("concurrency_order") or []] != expected_width_order(
            LEVELS[model], rep
        ):
            raise ValueError(f"{path}: concurrency-order drift")
        if (
            protocol.get("frozen_replay_sha256") != FROZEN_REPLAY_SHA256
            or protocol.get("frozen_workload_sha256") != FROZEN_WORKLOAD_SHA256
            or protocol.get("prompt_ids_sha256_canonical_json") != PROMPT_SHA256
            or int(protocol.get("prompt_tokens") or 0) != 4860
            or int(protocol.get("completion_tokens") or 0) != 60
        ):
            raise ValueError(f"{path}: frozen workload identity drift")
        tenant_checks = [
            row for row in rows if row.get("kind") == "compute_app_check"
        ]
        if key in REUSED_BOOT_KEYS:
            if tenant_checks:
                raise ValueError(f"{path}: reused pre-steering boot has tenant checks")
        else:
            expected_boundaries = ["before_seed"] + [
                f"before_c{value}" for value in protocol["concurrency_order"]
            ]
            if [str(row.get("boundary")) for row in tenant_checks] != expected_boundaries:
                raise ValueError(f"{path}: per-cell compute-app check grid drift")
            if any(
                row.get("verdict") != "PASS"
                or row.get("failures")
                or int(row.get("query_exit_code") or 0) != 0
                or len(row.get("apps") or []) != 1
                or int((row.get("apps") or [{}])[0].get("pid") or 0)
                != int(row.get("expected_server_pid") or 0)
                or (row.get("apps") or [{}])[0].get("gpu_uuid")
                != PHYSICAL_GPU0_UUID
                for row in tenant_checks
            ):
                raise ValueError(f"{path}: compute-app tenant check failed")
            if any(
                int(row.get("physical_gpu_index", -1)) != 0
                or row.get("physical_gpu_uuid") != PHYSICAL_GPU0_UUID
                or not row.get("physical_gpu_name")
                for row in rows
            ):
                raise ValueError(f"{path}: physical GPU identity missing from a row")
        if summary.get("verdict") != "PASS" or not seed.get("clean"):
            raise ValueError(f"{path}: boot did not pass")
        if len(seed_requests) != WORKING_SET_ENTRIES or any(
            not row.get("ok")
            or int(row.get("prompt_tokens") or 0) != 4860
            or int(row.get("completion_tokens") or 0) != 1
            or int(row.get("cached_tokens") or 0) != 0
            for row in seed_requests
        ):
            raise ValueError(f"{path}: seed response/usage drift")
        if (path.parent / "sweep.exit").read_text(encoding="utf-8").strip() != "0":
            raise ValueError(f"{path}: nonzero sweep exit receipt")
        if (path.parent / "server-failure-scan.log").read_text(
            encoding="utf-8", errors="replace"
        ).strip():
            raise ValueError(f"{path}: server failure scan is nonempty")
        run_cells = [row for row in rows if row.get("kind") == "cell"]
        if sorted(int(row["concurrency"]) for row in run_cells) != LEVELS[model]:
            raise ValueError(f"{path}: concurrency grid drift")
        if any(not row.get("clean") for row in run_cells):
            raise ValueError(f"{path}: unclean cell")
        run_requests = [row for row in rows if row.get("kind") == "request"]
        expected_requests = sum(max(20, math.ceil(level / 10) * 10) for level in LEVELS[model])
        if len(run_requests) != expected_requests:
            raise ValueError(f"{path}: expected {expected_requests} requests, got {len(run_requests)}")
        if any(
            not row.get("ok")
            or int(row.get("prompt_tokens") or 0) != 4860
            or int(row.get("completion_tokens") or 0) != 60
            or int(row.get("cached_tokens") or 0) not in (0, 4860)
            for row in run_requests
        ):
            raise ValueError(f"{path}: request response/usage drift")
        if any(
            (row["intended_role"] == "cold" and row["actual_cache_role"] != "miss")
            or (row["actual_cache_role"] == "hit" and row["intended_role"] != "working")
            for row in run_requests
        ):
            raise ValueError(f"{path}: cache-role drift")
        if any(
            int(row["counter_deltas"].get("admitted") or 0) != int(row["requests_n"])
            or int(row["counter_deltas"].get("completed") or 0) != int(row["requests_n"])
            or int(row["counter_deltas"].get("prefix_cache_hits") or 0) != int(row["hit_requests"])
            or int(row["counter_deltas"].get("prefix_cache_misses") or 0) != int(row["miss_requests"])
            or int(row["counter_deltas"].get("prefix_cache_inserts") or 0) != int(row["miss_requests"])
            or int(row["counter_deltas"].get("prefix_cache_hit_tokens") or 0) != int(row["cached_tokens"])
            or int(row["counter_deltas"].get("cached_tokens_in") or 0) != int(row["cached_tokens"])
            or int(row["counter_deltas"].get("tokens_out") or 0) != int(row["completion_tokens"])
            or int(row["counter_deltas"].get("prompt_tokens_in") or 0) != int(row["prompt_tokens"])
            or int(row["admission_session_defers"]) != 0
            or int(row["admission_vram_defers"]) != 0
            or int(row["step_oom_parks"]) != 0
            for row in run_cells
        ):
            raise ValueError(f"{path}: cell counter reconciliation drift")
        for concurrency in LEVELS[model]:
            request_n = max(20, math.ceil(concurrency / 10) * 10)
            cell_working = [
                int(row["prefix_id"])
                for row in run_requests
                if int(row["concurrency"]) == concurrency
                and row["intended_role"] == "working"
            ]
            if len(cell_working) != len(set(cell_working)):
                raise ValueError(
                    f"{path}: c{concurrency} repeats a working key within one cell"
                )
            cell_cold_n = sum(
                row["intended_role"] == "cold"
                for row in run_requests
                if int(row["concurrency"]) == concurrency
            )
            if len(cell_working) != request_n * 9 // 10 or cell_cold_n != request_n // 10:
                raise ValueError(f"{path}: c{concurrency} 9:1 role mix drift")
        cycle_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in run_requests:
            if row["intended_role"] == "working":
                cycle_rows[int(row["cycle_index"])].append(
                    (int(row["cycle_position"]), int(row["prefix_id"]))
                )
        for cycle_index, values in sorted(cycle_rows.items()):
            ordered = sorted(values)
            if [position for position, _ in ordered] != list(range(len(ordered))):
                raise ValueError(f"{path}: cycle {cycle_index} position drift")
            ids = [prefix_id for _, prefix_id in ordered]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path}: cycle {cycle_index} repeats a key")
            if len(ids) == WORKING_SET_ENTRIES and set(ids) != set(range(WORKING_SET_ENTRIES)):
                raise ValueError(f"{path}: cycle {cycle_index} is not a full permutation")
            if len(ids) < WORKING_SET_ENTRIES and cycle_index != max(cycle_rows):
                raise ValueError(f"{path}: non-final cycle {cycle_index} is incomplete")
        cells.extend(run_cells)
        requests.extend(run_requests)
        after_seed = [row for row in snapshots if row.get("boundary") == "after_seed"]
        if len(after_seed) != 1:
            raise ValueError(f"{path}: missing after-seed GPU snapshot")
        snapshot_fields = [
            value.strip() for value in str(after_seed[0]["csv"]).split(",")
        ]
        if len(snapshot_fields) < 3 or snapshot_fields[2] != PHYSICAL_GPU0_UUID:
            raise ValueError(f"{path}: after-seed physical GPU identity drift")
        gpu = parse_boot_gpu(path.parent / "gpu-250ms.csv")
        after_seed_gpu = parse_snapshot_csv(after_seed[0])
        ready_seconds = (
            snapshot_time(path.parent / "gpu-ready.log")
            - snapshot_time(path.parent / "gpu-before.log")
        ).total_seconds()
        if ready_seconds < 0:
            raise ValueError(f"{path}: negative boot-to-ready duration")
        expected_retained = min(
            WORKING_SET_ENTRIES,
            budget * 1024 * 1024 // sold_bytes[model],
        )
        if int(seed["retained_entries_after_seed"]) != expected_retained:
            raise ValueError(f"{path}: retained-entry accounting drift")
        if int(seed["retained_bytes_after_seed"]) != expected_retained * sold_bytes[model]:
            raise ValueError(f"{path}: retained-byte accounting drift")
        cost_event_lines = [
            line
            for line in (path.parent / "server.log").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if COST_EVENT.search(line)
        ]
        per_rep_counters = defaultdict(int)
        for counter, value in seed["counter_deltas"].items():
            per_rep_counters[counter] += int(value)
        for cell in run_cells:
            for counter, value in cell["counter_deltas"].items():
                per_rep_counters[counter] += int(value)
        boots.append({
            "source_raw": str(path.parents[2]),
            "model": model,
            "budget_mb": budget,
            "rep": rep,
            "seed_retained_entries": int(seed["retained_entries_after_seed"]),
            "seed_retained_bytes": int(seed["retained_bytes_after_seed"]),
            "seed_evictions": int(seed["counter_deltas"]["prefix_cache_evictions"]),
            "after_seed_memory_used_mib": after_seed_gpu["memory.used"],
            "boot_to_ready_seconds": ready_seconds,
            "seed_ttft_ms": [float(row["ttft_ms"]) for row in seed_requests],
            "seed_latency_ms": [float(row["latency_ms"]) for row in seed_requests],
            "gpu": gpu,
            "counters": dict(per_rep_counters),
            "cost_event_lines": cost_event_lines,
        })

    expected = {
        (model, budget, rep)
        for model in LEVELS
        for budget in BUDGETS
        for rep in range(1, REPETITIONS + 1)
    }
    expected.remove(EXPECTED_EXCLUDED_BOOT)
    if seen != expected:
        raise ValueError(f"boot grid mismatch: missing={sorted(expected - seen)}")
    for model in LEVELS:
        for rep in range(1, REPETITIONS + 1):
            traces: list[list[tuple[int, str, int | None]]] = []
            for budget in BUDGETS:
                if (model, budget, rep) not in seen:
                    continue
                selected = [
                    row
                    for row in requests
                    if row["model"] == model
                    and int(row["budget_mb"]) == budget
                    and int(row["rep"]) == rep
                ]
                traces.append([
                    (
                        int(row["concurrency"]),
                        str(row["intended_role"]),
                        int(row["prefix_id"]) if row["prefix_id"] is not None else None,
                    )
                    for row in selected
                ])
            if any(trace != traces[0] for trace in traces[1:]):
                raise ValueError(f"{model} r{rep}: budget-paired access trace drift")

    models: dict[str, Any] = {}
    all_counter_names = (
        "prefix_cache_hits",
        "prefix_cache_misses",
        "prefix_cache_inserts",
        "prefix_cache_evictions",
        "admission_session_defers",
        "admission_vram_defers",
        "step_oom_parks",
    )
    for model, levels in LEVELS.items():
        budget_rows: dict[str, Any] = {}
        for budget in BUDGETS:
            selected_boots = [row for row in boots if row["model"] == model and row["budget_mb"] == budget]
            boot_n = REPETITIONS - int((model, budget) == ("q27", 8192))
            if len(selected_boots) != boot_n:
                raise ValueError(f"{model} b{budget}: expected N={boot_n} boots")
            budget_requests = [
                row
                for row in requests
                if row["model"] == model and int(row["budget_mb"]) == budget
            ]
            budget_hits = [
                row for row in budget_requests if row["actual_cache_role"] == "hit"
            ]
            budget_misses = [
                row for row in budget_requests if row["actual_cache_role"] == "miss"
            ]
            budget_working = [
                row for row in budget_requests if row["intended_role"] == "working"
            ]
            budget_cells = [
                row
                for row in cells
                if row["model"] == model and int(row["budget_mb"]) == budget
            ]
            if len(budget_cells) != len(levels) * boot_n:
                raise ValueError(
                    f"{model} b{budget}: expected N={len(levels) * boot_n} cells"
                )
            concurrency_rows: dict[str, Any] = {}
            for concurrency in levels:
                selected_cells = [
                    row for row in cells
                    if row["model"] == model
                    and int(row["budget_mb"]) == budget
                    and int(row["concurrency"]) == concurrency
                ]
                selected_requests = [
                    row for row in requests
                    if row["model"] == model
                    and int(row["budget_mb"]) == budget
                    and int(row["concurrency"]) == concurrency
                ]
                if len(selected_cells) != boot_n:
                    raise ValueError(f"{model} b{budget} c{concurrency}: expected N={boot_n} cells")
                hits = [row for row in selected_requests if row["actual_cache_role"] == "hit"]
                misses = [row for row in selected_requests if row["actual_cache_role"] == "miss"]
                working = [row for row in selected_requests if row["intended_role"] == "working"]
                concurrency_rows[str(concurrency)] = {
                    "cell_n": len(selected_cells),
                    "requests_n": len(selected_requests),
                    "working_requests_n": len(working),
                    "hit_requests_n": len(hits),
                    "miss_requests_n": len(misses),
                    "working_set_hit_rate": len(hits) / len(working),
                    "all_request_hit_rate": len(hits) / len(selected_requests),
                    "cache_hit_token_ratio": sum(int(row["cached_tokens"]) for row in selected_requests)
                    / sum(int(row["prompt_tokens"]) for row in selected_requests),
                    "hit_ttft": distribution(float(row["ttft_ms"]) for row in hits),
                    "miss_ttft": distribution(float(row["ttft_ms"]) for row in misses),
                    "hit_ttft_p50_by_rep": [
                        row["ttft_hit"]["p50_ms"]
                        for row in sorted(selected_cells, key=lambda value: int(value["rep"]))
                    ],
                    "hit_ttft_p95_by_rep": [
                        row["ttft_hit"]["p95_ms"]
                        for row in sorted(selected_cells, key=lambda value: int(value["rep"]))
                    ],
                    "miss_ttft_p50_by_rep": [
                        row["ttft_miss"]["p50_ms"]
                        for row in sorted(selected_cells, key=lambda value: int(value["rep"]))
                    ],
                    "miss_ttft_p95_by_rep": [
                        row["ttft_miss"]["p95_ms"]
                        for row in sorted(selected_cells, key=lambda value: int(value["rep"]))
                    ],
                    "output_tok_s_median": median(float(row["output_tok_s"]) for row in selected_cells),
                    "output_tok_s_by_rep": [
                        float(row["output_tok_s"])
                        for row in sorted(selected_cells, key=lambda value: int(value["rep"]))
                    ],
                    "prefix_cache_evictions_total": sum(int(row["prefix_cache_evictions"]) for row in selected_cells),
                    "admission_session_defers_total": sum(int(row["admission_session_defers"]) for row in selected_cells),
                    "admission_vram_defers_total": sum(int(row["admission_vram_defers"]) for row in selected_cells),
                    "step_oom_parks_total": sum(int(row["step_oom_parks"]) for row in selected_cells),
                    "peak_queued_requests_sampled_max": max(
                        int(row["peak_queued_requests_sampled"]) for row in selected_cells
                    ),
                }

            budget_rows[str(budget)] = {
                "boot_n": boot_n,
                "configured_mib": budget,
                "formula_entry_capacity_at_4860": budget * 1024 * 1024 // sold_bytes[model],
                "seed_retained_entries_median": median(row["seed_retained_entries"] for row in selected_boots),
                "seed_retained_entries_by_rep": [row["seed_retained_entries"] for row in sorted(selected_boots, key=lambda value: value["rep"])],
                "seed_retained_mib_median": median(row["seed_retained_bytes"] / 1024 / 1024 for row in selected_boots),
                "seed_evictions_median": median(row["seed_evictions"] for row in selected_boots),
                "seed_prime_ttft": distribution(
                    value
                    for row in selected_boots
                    for value in row["seed_ttft_ms"]
                ),
                "seed_prime_latency": distribution(
                    value
                    for row in selected_boots
                    for value in row["seed_latency_ms"]
                ),
                "seed_prime_ttft_median_by_rep": [
                    median(row["seed_ttft_ms"])
                    for row in sorted(selected_boots, key=lambda value: value["rep"])
                ],
                "boot_to_ready_seconds_median": median(
                    row["boot_to_ready_seconds"] for row in selected_boots
                ),
                "boot_to_ready_seconds_by_rep": [
                    row["boot_to_ready_seconds"]
                    for row in sorted(selected_boots, key=lambda value: value["rep"])
                ],
                "vram": {
                    "total_mib": median(row["gpu"]["total_memory_mib"] for row in selected_boots),
                    "after_seed_used_mib_median": median(row["after_seed_memory_used_mib"] for row in selected_boots),
                    "peak_used_mib_median": median(row["gpu"]["max_memory_used_mib"] for row in selected_boots),
                    "peak_used_mib_min": min(row["gpu"]["max_memory_used_mib"] for row in selected_boots),
                    "peak_used_mib_max": max(row["gpu"]["max_memory_used_mib"] for row in selected_boots),
                },
                "counter_totals": {
                    counter: sum(int(row["counters"].get(counter, 0)) for row in selected_boots)
                    for counter in all_counter_names
                },
                "counter_median_per_boot": {
                    counter: median(int(row["counters"].get(counter, 0)) for row in selected_boots)
                    for counter in all_counter_names
                },
                "mixed_counter_totals": {
                    counter: sum(
                        int(cell["counter_deltas"].get(counter, 0))
                        for cell in budget_cells
                    )
                    for counter in all_counter_names
                },
                "captured_allocation_or_oom_event_lines": [
                    {"rep": row["rep"], "line": line}
                    for row in sorted(selected_boots, key=lambda value: value["rep"])
                    for line in row["cost_event_lines"]
                ],
                "pooled_mixed_load": {
                    "cell_n": len(levels) * boot_n,
                    "requests_n": len(budget_requests),
                    "working_requests_n": len(budget_working),
                    "hit_requests_n": len(budget_hits),
                    "miss_requests_n": len(budget_misses),
                    "working_set_hit_rate": len(budget_hits) / len(budget_working),
                    "all_request_hit_rate": len(budget_hits) / len(budget_requests),
                    "hit_ttft": distribution(
                        float(row["ttft_ms"]) for row in budget_hits
                    ),
                    "miss_ttft": distribution(
                        float(row["ttft_ms"]) for row in budget_misses
                    ),
                },
                "concurrency": concurrency_rows,
            }

        full_residency = [
            budget
            for budget in BUDGETS
            if all(
                math.isclose(
                    float(budget_rows[str(budget)]["concurrency"][str(level)]["working_set_hit_rate"]),
                    1.0,
                    rel_tol=0,
                    abs_tol=0,
                )
                for level in levels
            )
        ]
        if not full_residency:
            raise ValueError(f"{model}: no budget retained the full working set")
        knee_budget = min(full_residency)
        sold_level = MODEL_KNEE[model]
        sold_by_budget = {
            str(budget): {
                "hit_p95_ms": budget_rows[str(budget)]["concurrency"][str(sold_level)]["hit_ttft"]["p95_ms"],
                "working_set_hit_rate": budget_rows[str(budget)]["concurrency"][str(sold_level)]["working_set_hit_rate"],
                "output_tok_s_median": budget_rows[str(budget)]["concurrency"][str(sold_level)]["output_tok_s_median"],
            }
            for budget in BUDGETS
        }
        largest_sold_concurrency_by_budget: dict[str, int | None] = {}
        for budget in BUDGETS:
            eligible = [
                level
                for level in levels
                if budget_rows[str(budget)]["concurrency"][str(level)]["working_set_hit_rate"] == 1.0
                and budget_rows[str(budget)]["concurrency"][str(level)]["hit_ttft"]["p95_ms"] is not None
                and float(budget_rows[str(budget)]["concurrency"][str(level)]["hit_ttft"]["p95_ms"])
                <= SOLD_HIT_P95_MS[model]
            ]
            largest_sold_concurrency_by_budget[str(budget)] = max(eligible) if eligible else None
        models[model] = {
            "entry_bytes_4860": sold_bytes[model],
            "full_working_set_budget_knee_mib": knee_budget,
            "full_residency_budgets_mib": full_residency,
            "sold_hit_p95_limit_ms": SOLD_HIT_P95_MS[model],
            "sold_throughput_knee_concurrency": sold_level,
            "sold_knee_by_budget": sold_by_budget,
            "largest_measured_concurrency_with_full_residency_and_sold_hit_p95": largest_sold_concurrency_by_budget,
            "budgets": budget_rows,
        }

    q27_bytes = sold_bytes["q27"]
    q35_bytes = sold_bytes["q35"]
    production_capacity = {
        str(budget): {
            "q27_only_entries": budget * 1024 * 1024 // q27_bytes,
            "q35_only_entries": budget * 1024 * 1024 // q35_bytes,
            "paired_q27_plus_q35_sessions": budget * 1024 * 1024 // (q27_bytes + q35_bytes),
        }
        for budget in BUDGETS
    }
    production_memory = {
        str(budget): {
            "used_mib_if_current_4g_cache_was_full": LIVE_USED_MIB + budget - LIVE_PREFIX_CACHE_MIB,
            "free_mib_if_current_4g_cache_was_full": LIVE_TOTAL_MIB - (LIVE_USED_MIB + budget - LIVE_PREFIX_CACHE_MIB),
            "used_mib_if_current_4g_cache_was_empty": LIVE_USED_MIB + budget,
            "free_mib_if_current_4g_cache_was_empty": LIVE_TOTAL_MIB - (LIVE_USED_MIB + budget),
        }
        for budget in BUDGETS
    }
    q27_n96_mib = q27_bytes * WORKING_SET_ENTRIES / 1024 / 1024
    q35_n96_mib = q35_bytes * WORKING_SET_ENTRIES / 1024 / 1024
    paired_n96_mib = (q27_bytes + q35_bytes) * WORKING_SET_ENTRIES / 1024 / 1024
    base_if_current_cache_full = LIVE_USED_MIB - LIVE_PREFIX_CACHE_MIB
    base_if_current_cache_empty = LIVE_USED_MIB
    production_working_set = {
        "minimum_budget_mib_for_q27_n96": math.ceil(q27_n96_mib),
        "minimum_budget_mib_for_q35_n96": math.ceil(q35_n96_mib),
        "minimum_budget_mib_for_paired_q27_q35_n96": math.ceil(paired_n96_mib),
        "q27_n96_device_mib": q27_n96_mib,
        "q35_n96_device_mib": q35_n96_mib,
        "paired_q27_q35_n96_device_mib": paired_n96_mib,
        "worst_one_prefix_per_session_n96_live_bounds": {
            "model": "q27",
            "used_mib_if_current_4g_cache_was_full": base_if_current_cache_full + q27_n96_mib,
            "free_mib_if_current_4g_cache_was_full": LIVE_TOTAL_MIB - base_if_current_cache_full - q27_n96_mib,
            "used_mib_if_current_4g_cache_was_empty": base_if_current_cache_empty + q27_n96_mib,
            "free_mib_if_current_4g_cache_was_empty": LIVE_TOTAL_MIB - base_if_current_cache_empty - q27_n96_mib,
        },
    }
    source_boot_counts: dict[str, int] = defaultdict(int)
    for boot in boots:
        source_boot_counts[str(boot["source_raw"])] += 1
    valid_boot_thermal = {
        "max_temperature_c": max(row["gpu"]["max_temperature_c"] for row in boots),
        "max_power_w": max(row["gpu"]["max_power_w"] for row in boots),
        "max_clock_mhz": max(row["gpu"]["max_clock_mhz"] for row in boots),
        "total_memory_mib": median(row["gpu"]["total_memory_mib"] for row in boots),
        "max_memory_used_mib": max(row["gpu"]["max_memory_used_mib"] for row in boots),
        "max_utilization_percent": max(
            row["gpu"]["max_utilization_percent"] for row in boots
        ),
    }
    return {
        "models": models,
        "production_capacity_arithmetic": production_capacity,
        "production_memory_arithmetic": production_memory,
        "production_working_set_arithmetic": production_working_set,
        "live_observation_supplied_by_owner": {
            "prefix_cache_mib": LIVE_PREFIX_CACHE_MIB,
            "memory_used_mib": LIVE_USED_MIB,
            "memory_total_mib": LIVE_TOTAL_MIB,
            "warning": "The cache is lazy and its current occupancy was not supplied, so the bounds treat the present 4 GiB allowance as either empty or full. The live serve box was not touched by this lane.",
        },
        "excluded_boots": excluded_boots,
        "valid_boots_by_raw_segment": dict(source_boot_counts),
        "valid_boot_thermal_gpu0": valid_boot_thermal,
        "campaign_counts": {
            "boots": len(boots),
            "cells": len(cells),
            "requests": len(requests),
            "repetitions_per_cell": {
                "default": REPETITIONS,
                "q27_budget_8192": REPETITIONS - 1,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-raw", type=Path, required=True)
    parser.add_argument(
        "--campaign-raw", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--exclude-sweep", type=Path, action="append", default=[]
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    campaign_raws = list(args.campaign_raw)
    if len({raw.resolve() for raw in campaign_raws}) != len(campaign_raws):
        raise ValueError("campaign raw segments must be unique")
    excluded_paths = {path.resolve() for path in args.exclude_sweep}
    if len(excluded_paths) != 1:
        raise ValueError(f"expected one explicit excluded sweep, got {len(excluded_paths)}")
    if args.entry_raw.resolve() not in {raw.resolve() for raw in campaign_raws}:
        raise ValueError("entry receipt root must also be a campaign raw segment")
    completed_segments = [
        raw
        for raw in campaign_raws
        if (raw / "campaign.complete").is_file()
        or (raw / "resume-segment.complete").is_file()
    ]
    if len(completed_segments) != 1:
        raise ValueError(
            f"expected one completed continuation segment, got {len(completed_segments)}"
        )
    manifests = [
        {"raw": str(raw), **verify_manifest(raw)} for raw in campaign_raws
    ]
    entries, sold_bytes = entry_measurements(args.entry_raw)
    campaign = reduce_campaign(campaign_raws, sold_bytes, excluded_paths)
    segment_timing = [
        {
            "raw": str(raw),
            **campaign_timing(raw / "orchestrator.log", raw in completed_segments),
        }
        for raw in campaign_raws
    ]
    sidecars = [
        {
            "raw": str(raw),
            **parse_all_gpu_monitor(raw / "operator-gpu-both-250ms.csv"),
        }
        for raw in campaign_raws
        if (raw / "operator-gpu-both-250ms.csv").is_file()
    ]
    if len(sidecars) != 1 or sidecars[0]["raw"] != str(completed_segments[0]):
        raise ValueError("completed continuation segment lacks its two-GPU sidecar")
    result = {
        "schema": "memra.cachesize.analysis.v2",
        "verdict": "PASS",
        "runtime": {"source": RUNTIME_SOURCE, "tag": "v0.81.2"},
        "frozen_workload": {
            "prompt_tokens": 4860,
            "completion_tokens": 60,
            "working_set_entries": WORKING_SET_ENTRIES,
            "nominal_repetitions": REPETITIONS,
            "scored_exception": "Q27 8192 MiB has N=4; repetition 1 is the explicit EOS exclusion",
            "replay_sha256": FROZEN_REPLAY_SHA256,
            "workload_sha256": FROZEN_WORKLOAD_SHA256,
            "prompt_sha256": PROMPT_SHA256,
        },
        "manifests": manifests,
        "campaign_segments": segment_timing,
        "segmentation_note": (
            "Two passing Q27 repetition-1 boots were preserved from the interrupted first lock "
            "segment; the other 57 valid boots ran in the completed continuation lock segment."
        ),
        "boot_order": validate_boot_order(campaign_raws, excluded_paths),
        "all_gpu_sidecars": sidecars,
        "entry_measurements": entries,
        **campaign,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
