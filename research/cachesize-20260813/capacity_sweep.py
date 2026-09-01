#!/usr/bin/env python3
"""One cache-budget boot of the frozen sold-shape workload over a 96-key set."""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import importlib.util
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO


RUN_IDENTITY: dict[str, Any] = {}

COUNTERS = (
    "admitted",
    "completed",
    "tokens_out",
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_hit_tokens",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, expected_sha256: str) -> ModuleType:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: expected {expected_sha256}, got {actual}")
    spec = importlib.util.spec_from_file_location("cachesize_frozen_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def emit(output: TextIO, row: dict[str, Any], announce: bool = False) -> None:
    payload = {**RUN_IDENTITY, **row}
    line = json.dumps(payload, sort_keys=True)
    output.write(line + "\n")
    output.flush()
    if announce:
        print(line, flush=True)


def metric(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def deltas(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {key: metric(after, key) - metric(before, key) for key in COUNTERS}


def gpu_snapshot(boundary: str) -> dict[str, Any]:
    fields = (
        "index,name,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,"
        "clocks.mem,memory.total,memory.used,memory.free,utilization.gpu"
    )
    command = [
        "nvidia-smi",
        "-i",
        "0",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "kind": "gpu_snapshot",
        "boundary": boundary,
        "timestamp_unix_s": time.time(),
        "query_fields": fields.split(","),
        "csv": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "exit_code": completed.returncode,
    }


def gpu_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=index,uuid,name",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = list(csv.reader(completed.stdout.splitlines()))
    if completed.returncode != 0 or len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(
            "physical GPU0 identity query failed: "
            f"exit={completed.returncode} stdout={completed.stdout!r} "
            f"stderr={completed.stderr!r}"
        )
    index, uuid, name = (value.strip() for value in rows[0])
    if index != "0" or not uuid:
        raise RuntimeError(f"unexpected physical GPU identity: {rows[0]!r}")
    return {
        "physical_gpu_index": 0,
        "physical_gpu_uuid": uuid,
        "physical_gpu_name": name,
    }


def compute_app_check(expected_server_pid: int, boundary: str) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    apps: list[dict[str, Any]] = []
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"nvidia-smi compute-app query exited {completed.returncode}")
    else:
        for number, fields in enumerate(csv.reader(completed.stdout.splitlines()), 1):
            if len(fields) != 4:
                failures.append(f"compute-app row {number} has {len(fields)} fields")
                continue
            gpu_uuid, pid_text, process_name, used_memory = (
                value.strip() for value in fields
            )
            try:
                pid = int(pid_text)
            except ValueError:
                failures.append(f"compute-app row {number} has invalid pid {pid_text!r}")
                continue
            apps.append({
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": used_memory,
            })
    expected_uuid = str(RUN_IDENTITY["physical_gpu_uuid"])
    expected = [
        app
        for app in apps
        if app["pid"] == expected_server_pid and app["gpu_uuid"] == expected_uuid
    ]
    foreign = [
        app
        for app in apps
        if app["pid"] != expected_server_pid or app["gpu_uuid"] != expected_uuid
    ]
    if len(expected) != 1:
        failures.append(
            f"expected one physical-GPU0 server pid {expected_server_pid}, got {len(expected)}"
        )
    if foreign:
        failures.append(f"foreign compute apps present: {foreign!r}")
    return {
        "kind": "compute_app_check",
        "schema": "memra.cachesize.compute-app-check.v1",
        "boundary": boundary,
        "expected_server_pid": expected_server_pid,
        "apps": apps,
        "query_stderr": completed.stderr.strip(),
        "query_exit_code": completed.returncode,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


class WorkingSetCycle:
    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self.cycle = -1
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.position = 0
        self.cycle += 1

    def take(self, exclude: set[int] | None = None) -> tuple[int, int, int]:
        if self.position == self.size:
            self._reshuffle()
        excluded = exclude or set()
        if self.order[self.position] in excluded:
            replacement = next(
                (
                    index
                    for index in range(self.position + 1, self.size)
                    if self.order[index] not in excluded
                ),
                None,
            )
            if replacement is None:
                raise RuntimeError("working-set cycle cannot provide a distinct key")
            self.order[self.position], self.order[replacement] = (
                self.order[replacement],
                self.order[self.position],
            )
        position = self.position
        value = self.order[position]
        self.position += 1
        return value, self.cycle, position


def run_seed(
    frozen: ModuleType,
    endpoint: Any,
    workload: dict[str, Any],
    namespace: str,
    working_set_n: int,
    timeout: float,
    output: TextIO,
) -> tuple[dict[str, Any], list[str]]:
    prompt = frozen.scored_prompt_ids(workload)
    seed_workload = dict(workload)
    seed_workload["completion_tokens"] = 1
    before = frozen.scrape(endpoint, timeout)
    rows: list[dict[str, Any]] = []
    # Match the frozen sell-gate's hot-cache setup: one seed request at a time. Prefix
    # snapshots inherit the numerical class of the prime configuration that produced them.
    batch_size = 1
    for start in range(0, working_set_n, batch_size):
        ids = list(range(start, min(start + batch_size, working_set_n)))
        barrier = threading.Barrier(len(ids))

        def one(prefix_id: int) -> dict[str, Any]:
            row = frozen.request(
                endpoint,
                prompt,
                f"{namespace}-hot-{prefix_id}",
                seed_workload,
                timeout,
                barrier=barrier,
            )
            row.update({
                "kind": "seed",
                "prefix_id": prefix_id,
                "working_set_entries": working_set_n,
            })
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ids)) as pool:
            futures = [pool.submit(one, prefix_id) for prefix_id in ids]
            batch_rows = [future.result() for future in futures]
        rows.extend(batch_rows)
        for row in batch_rows:
            emit(output, {key: value for key, value in row.items() if not key.startswith("_")})

    after = frozen.wait_settled(endpoint, metric(before, "completed"), working_set_n, timeout)
    counter_deltas = deltas(after, before)
    failures: list[str] = []
    if any(not row.get("ok") for row in rows):
        failures.append("one or more seed requests failed")
    if any(row.get("prompt_tokens") != len(prompt) for row in rows):
        failures.append("one or more seed prompt-token receipts drifted")
    if any(row.get("cached_tokens") != 0 for row in rows):
        failures.append("one or more fresh working-set seeds received cache credit")
    exact = {
        "admitted": working_set_n,
        "completed": working_set_n,
        "tokens_out": working_set_n,
        "prompt_tokens_in": working_set_n * len(prompt),
        "cached_tokens_in": 0,
        "prefix_cache_hits": 0,
        "prefix_cache_misses": working_set_n,
        "prefix_cache_inserts": working_set_n,
        "prefix_cache_hit_tokens": 0,
        "step_oom_parks": 0,
    }
    for key, expected in exact.items():
        if counter_deltas[key] != expected:
            failures.append(f"seed {key} delta={counter_deltas[key]} != {expected}")
    summary = {
        "kind": "seed_summary",
        "schema": "memra.cachesize.sweep.v1",
        "requests": len(rows),
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "working_set_entries": working_set_n,
        "retained_entries_after_seed": metric(after, "prefix_cache_entries"),
        "retained_bytes_after_seed": metric(after, "prefix_cache_bytes"),
        "counter_deltas": counter_deltas,
        "admission_session_defers": counter_deltas["admission_session_defers"],
        "admission_vram_defers": counter_deltas["admission_vram_defers"],
        "failures": failures,
        "clean": not failures,
    }
    emit(output, summary, announce=True)
    return summary, failures


def make_jobs(
    frozen: ModuleType,
    workload: dict[str, Any],
    namespace: str,
    model: str,
    budget_mb: int,
    repetition: int,
    concurrency: int,
    cycle: WorkingSetCycle,
) -> list[dict[str, Any]]:
    request_n = frozen.cell_request_count(workload, concurrency)
    prompt = frozen.scored_prompt_ids(workload)
    hit_per_cycle = int(workload["hit_requests_per_cycle"])
    miss_per_cycle = int(workload["miss_requests_per_cycle"])
    role_cycle = hit_per_cycle + miss_per_cycle
    working_n = request_n * hit_per_cycle // role_cycle
    roles = ["working"] * working_n + ["cold"] * (request_n - working_n)
    random.Random(
        int(workload["seed"]) + repetition * 1009 + concurrency * 9173
    ).shuffle(roles)
    jobs: list[dict[str, Any]] = []
    cell_prefix_ids: set[int] = set()
    for index, role in enumerate(roles):
        if role == "working":
            prefix_id, cycle_index, cycle_position = cycle.take(cell_prefix_ids)
            cell_prefix_ids.add(prefix_id)
            salt = f"{namespace}-hot-{prefix_id}"
        else:
            prefix_id = None
            cycle_index = None
            cycle_position = None
            salt = (
                f"{namespace}-{model}-b{budget_mb}-r{repetition}-c{concurrency}-cold-{index}"
            )
        jobs.append({
            "index": index,
            "intended_role": role,
            "prefix_id": prefix_id,
            "cycle_index": cycle_index,
            "cycle_position": cycle_position,
            "prompt": prompt,
            "salt": salt,
        })
    return jobs


def run_cell(
    frozen: ModuleType,
    endpoint: Any,
    workload: dict[str, Any],
    namespace: str,
    model: str,
    budget_mb: int,
    repetition: int,
    concurrency: int,
    cycle: WorkingSetCycle,
    timeout: float,
    output: TextIO,
) -> tuple[dict[str, Any], list[str]]:
    jobs = make_jobs(
        frozen,
        workload,
        namespace,
        model,
        budget_mb,
        repetition,
        concurrency,
        cycle,
    )
    before = frozen.scrape(endpoint, timeout)
    barrier = threading.Barrier(concurrency + 1)
    go = threading.Event()
    release_box: list[float | None] = [None]

    def one(job: dict[str, Any], first_wave: bool) -> dict[str, Any]:
        row = frozen.request(
            endpoint,
            job["prompt"],
            job["salt"],
            workload,
            timeout,
            barrier=barrier if first_wave else None,
            go=go,
        )
        release = release_box[0]
        assert release is not None
        cached = int(row.get("cached_tokens") or 0)
        actual_role = "hit" if cached == len(job["prompt"]) else "miss"
        usage_ok = bool(
            row.get("prompt_tokens") == len(job["prompt"])
            and cached in (0, len(job["prompt"]))
            and row.get("completion_tokens") == int(workload["completion_tokens"])
            and (job["intended_role"] != "cold" or cached == 0)
        )
        row.update({
            "kind": "request",
            "model": model,
            "budget_mb": budget_mb,
            "rep": repetition,
            "concurrency": concurrency,
            "index": job["index"],
            "intended_role": job["intended_role"],
            "actual_cache_role": actual_role,
            "prefix_id": job["prefix_id"],
            "cycle_index": job["cycle_index"],
            "cycle_position": job["cycle_position"],
            "usage_ok": usage_ok,
            "request_start_offset_ms": (float(row["_started"]) - release) * 1000.0,
        })
        row["ok"] = bool(row.get("ok") and usage_ok)
        return row

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    futures = [
        executor.submit(one, job, index < concurrency)
        for index, job in enumerate(jobs)
    ]
    barrier.wait(timeout=60)
    release_box[0] = time.monotonic()
    go.set()
    samples: list[dict[str, Any]] = []
    while not all(future.done() for future in futures):
        try:
            sample = frozen.scrape(endpoint, min(timeout, 10.0))
            samples.append({
                "kind": "metrics_sample",
                "model": model,
                "budget_mb": budget_mb,
                "rep": repetition,
                "concurrency": concurrency,
                "elapsed_s": time.monotonic() - float(release_box[0]),
                "active_sessions": sample.get("active_sessions"),
                "queued_requests": sample.get("queued_requests"),
                "prefix_cache_entries": sample.get("prefix_cache_entries"),
                "prefix_cache_bytes": sample.get("prefix_cache_bytes"),
                "admission_session_defers": sample.get("admission_session_defers"),
                "admission_vram_defers": sample.get("admission_vram_defers"),
                "step_oom_parks": sample.get("step_oom_parks"),
            })
        except Exception as error:
            samples.append({
                "kind": "metrics_sample",
                "model": model,
                "budget_mb": budget_mb,
                "rep": repetition,
                "concurrency": concurrency,
                "elapsed_s": time.monotonic() - float(release_box[0]),
                "error": f"{type(error).__name__}: {error}",
            })
        time.sleep(0.1)
    rows = [future.result() for future in futures]
    executor.shutdown(wait=True)
    after = frozen.wait_settled(
        endpoint,
        metric(before, "completed"),
        len(jobs),
        timeout,
    )
    counter_deltas = deltas(after, before)
    release = float(release_box[0])
    wall_s = max(float(row["_ended"]) for row in rows) - release
    hit_rows = [row for row in rows if row["actual_cache_role"] == "hit"]
    miss_rows = [row for row in rows if row["actual_cache_role"] == "miss"]
    working_rows = [row for row in rows if row["intended_role"] == "working"]
    prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    cached_total = sum(int(row.get("cached_tokens") or 0) for row in rows)
    completion_total = sum(int(row.get("completion_tokens") or 0) for row in rows)
    failures: list[str] = []
    if any(not row.get("ok") for row in rows):
        failures.append("one or more requests failed response or usage checks")
    exact = {
        "admitted": len(rows),
        "completed": len(rows),
        "tokens_out": completion_total,
        "prompt_tokens_in": prompt_total,
        "cached_tokens_in": cached_total,
        "prefix_cache_hits": len(hit_rows),
        "prefix_cache_misses": len(miss_rows),
        "prefix_cache_inserts": len(miss_rows),
        "prefix_cache_hit_tokens": cached_total,
        "step_oom_parks": 0,
    }
    for key, expected in exact.items():
        if counter_deltas[key] != expected:
            failures.append(f"{key} delta={counter_deltas[key]} != {expected}")

    def times(selected: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in selected if row.get(key) is not None]

    cell = {
        "kind": "cell",
        "schema": "memra.cachesize.sweep.v1",
        "model": model,
        "budget_mb": budget_mb,
        "rep": repetition,
        "concurrency": concurrency,
        "requests_n": len(rows),
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "working_requests": len(working_rows),
        "hit_requests": len(hit_rows),
        "miss_requests": len(miss_rows),
        "working_set_hit_rate": (
            sum(row["actual_cache_role"] == "hit" for row in working_rows)
            / len(working_rows)
            if working_rows else None
        ),
        "request_hit_rate": len(hit_rows) / len(rows),
        "cache_hit_token_ratio": cached_total / prompt_total if prompt_total else None,
        "prompt_tokens": prompt_total,
        "cached_tokens": cached_total,
        "completion_tokens": completion_total,
        "wall_s": wall_s,
        "requests_per_s": len(rows) / wall_s,
        "output_tok_s": completion_total / wall_s,
        "ttft_hit": frozen.distribution(times(hit_rows, "ttft_ms")),
        "ttft_miss": frozen.distribution(times(miss_rows, "ttft_ms")),
        "ttft_all": frozen.distribution(times(rows, "ttft_ms")),
        "latency_hit": frozen.distribution(times(hit_rows, "latency_ms")),
        "latency_miss": frozen.distribution(times(miss_rows, "latency_ms")),
        "counter_deltas": counter_deltas,
        "prefix_cache_entries_before": metric(before, "prefix_cache_entries"),
        "prefix_cache_entries_after": metric(after, "prefix_cache_entries"),
        "prefix_cache_bytes_before": metric(before, "prefix_cache_bytes"),
        "prefix_cache_bytes_after": metric(after, "prefix_cache_bytes"),
        "peak_active_sessions_sampled": max(
            (int(row["active_sessions"]) for row in samples if isinstance(row.get("active_sessions"), int)),
            default=0,
        ),
        "peak_queued_requests_sampled": max(
            (int(row["queued_requests"]) for row in samples if isinstance(row.get("queued_requests"), int)),
            default=0,
        ),
        "admission_session_defers": counter_deltas["admission_session_defers"],
        "admission_vram_defers": counter_deltas["admission_vram_defers"],
        "step_oom_parks": counter_deltas["step_oom_parks"],
        "prefix_cache_evictions": counter_deltas["prefix_cache_evictions"],
        "request_start_spread_ms": max(float(row["request_start_offset_ms"]) for row in rows)
        - min(float(row["request_start_offset_ms"]) for row in rows),
        "failures": failures,
        "clean": not failures,
    }
    for sample in samples:
        emit(output, sample)
    for row in rows:
        emit(output, {key: value for key, value in row.items() if not key.startswith("_")})
    emit(output, cell, announce=True)
    return cell, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", choices=("q27", "q35"), required=True)
    parser.add_argument("--budget-mb", type=int, required=True)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--expected-server-pid", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    budgets = [int(value) for value in protocol["prefix_cache_mb"]]
    repetitions = int(protocol["repetitions"])
    if args.budget_mb not in budgets:
        parser.error(f"budget must be one of {budgets}")
    if args.repetition not in range(1, repetitions + 1):
        parser.error(f"repetition must be in 1..{repetitions}")
    frozen = load_module(args.frozen_replay, protocol["frozen_replay_sha256"])
    if sha256_file(args.workload_lock) != protocol["frozen_workload_sha256"]:
        parser.error("frozen workload hash mismatch")
    workload = frozen.load_workload(args.workload_lock)
    endpoint = frozen.parse_endpoint(f"{args.model},{args.endpoint},{args.model}")
    RUN_IDENTITY.update(gpu_identity())
    working_set_n = int(protocol["working_set_entries"])
    levels = [int(value) for value in protocol["concurrency"][args.model]]
    order = frozen.width_orders(levels, repetitions)[args.repetition - 1]
    model_seed = 27 if args.model == "q27" else 35
    cycle_seed = (
        int(protocol["seed"])
        + model_seed * 1_000_003
        + args.repetition * 10_007
    )
    cycle = WorkingSetCycle(working_set_n, cycle_seed)
    failures: list[str] = []
    cells: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        emit(output, {
            "kind": "protocol",
            "schema": "memra.cachesize.sweep.v1",
            "model": args.model,
            "budget_mb": args.budget_mb,
            "repetition": args.repetition,
            "concurrency_order": order,
            "working_set_entries": working_set_n,
            "seed_concurrency": int(protocol["seed_concurrency"]),
            "seed_method": protocol["seed_method"],
            "working_set": protocol["working_set"],
            "working_set_cycle": protocol["working_set_cycle"],
            "working_set_cell_rule": protocol["working_set_cell_rule"],
            "budget_pairing": protocol["budget_pairing"],
            "working_set_cycle_seed": cycle_seed,
            "prompt_tokens": int(workload["prompt_tokens"]),
            "completion_tokens": int(workload["completion_tokens"]),
            "hit_requests_per_cycle": int(workload["hit_requests_per_cycle"]),
            "miss_requests_per_cycle": int(workload["miss_requests_per_cycle"]),
            "frozen_replay_sha256": sha256_file(args.frozen_replay),
            "frozen_workload_sha256": sha256_file(args.workload_lock),
            "prompt_ids_sha256_canonical_json": frozen.prompt_sha256(
                frozen.scored_prompt_ids(workload)
            ),
            "thermal_regime": protocol["thermal_regime"],
        }, announce=True)
        pre_seed_check = compute_app_check(args.expected_server_pid, "before_seed")
        emit(output, pre_seed_check, announce=True)
        seed_failures: list[str] = []
        if pre_seed_check["failures"]:
            failures.extend(
                f"before_seed: {failure}"
                for failure in pre_seed_check["failures"]
            )
        else:
            _, seed_failures = run_seed(
                frozen,
                endpoint,
                workload,
                args.namespace,
                working_set_n,
                args.timeout,
                output,
            )
            emit(output, gpu_snapshot("after_seed"))
            failures.extend(seed_failures)
        if not failures:
            for concurrency in order:
                tenant_check = compute_app_check(
                    args.expected_server_pid, f"before_c{concurrency}"
                )
                emit(output, tenant_check, announce=True)
                if tenant_check["failures"]:
                    failures.extend(
                        f"c{concurrency}: {failure}"
                        for failure in tenant_check["failures"]
                    )
                    break
                cell, cell_failures = run_cell(
                    frozen,
                    endpoint,
                    workload,
                    args.namespace,
                    args.model,
                    args.budget_mb,
                    args.repetition,
                    concurrency,
                    cycle,
                    args.timeout,
                    output,
                )
                cells.append(cell)
                failures.extend(f"c{concurrency}: {failure}" for failure in cell_failures)
                emit(output, gpu_snapshot(f"after_c{concurrency}"))
        emit(output, {
            "kind": "summary",
            "schema": "memra.cachesize.sweep.v1",
            "model": args.model,
            "budget_mb": args.budget_mb,
            "repetition": args.repetition,
            "cells": len(cells),
            "expected_cells": len(levels),
            "failures": failures,
            "verdict": "PASS" if not failures and len(cells) == len(levels) else "FAIL",
        }, announce=True)
    return 0 if not failures and len(cells) == len(levels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
