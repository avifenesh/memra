#!/usr/bin/env python3
"""Hourly public-path health and latency receipt for the two-day pair serve test."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pathlib
import subprocess
from typing import Any

from public_gate import (
    Client,
    completion_payload,
    fixed_prompt_ids,
    rate_headers_ok,
    usage_from,
    utc_now,
    validate_usage,
    write_json,
    write_manifest,
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def process_snapshot(pid_file: pathlib.Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"pid_file": str(pid_file), "alive": False}
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        system_uptime = float(pathlib.Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        snapshot.update({
            "pid": pid,
            "alive": True,
            "state": stat[2],
            "uptime_s": round(system_uptime - int(stat[21]) / ticks, 3),
        })
    except (FileNotFoundError, PermissionError, ValueError, IndexError, ProcessLookupError) as exc:
        snapshot["error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def append_ledger(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> int:
    lock_path = pathlib.Path(args.lock_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("hourly monitor already running")
        return 75

    started_utc = utc_now()
    stamp = started_utc.replace(":", "").replace("-", "").replace(".", "")
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    final_dir = out_root / stamp
    temp_dir = out_root / f".{stamp}.{os.getpid()}.tmp"
    temp_dir.mkdir()

    base_url = pathlib.Path(args.base_url_file).read_text(encoding="utf-8").strip().rstrip("/")
    metrics_base_url = (args.metrics_base_url or base_url).rstrip("/")
    api_key = pathlib.Path(args.api_key_file).read_text(encoding="utf-8").strip()
    metrics_token = pathlib.Path(args.metrics_token_file).read_text(encoding="utf-8").strip()
    client = Client(base_url, api_key, args.timeout)
    metrics_client = Client(metrics_base_url, metrics_token, args.timeout)

    health = client.get_json("/health")
    ready = client.get_json("/readyz")
    write_json(temp_dir / "health.json", health.body)
    write_json(temp_dir / "ready.json", ready.body)

    metrics = metrics_client.get_json("/metrics", authenticated=True)
    write_json(temp_dir / "metrics.json", metrics.body)

    process = process_snapshot(pathlib.Path(args.pid_file))
    write_json(temp_dir / "process.json", process)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,uuid,temperature.gpu,power.draw,power.limit,"
            "memory.used,memory.total,utilization.gpu,pstate",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    (temp_dir / "gpu.csv").write_text(gpu.stdout, encoding="utf-8")
    if gpu.stderr:
        (temp_dir / "gpu.stderr").write_text(gpu.stderr, encoding="utf-8")

    # Reuse the frozen sellgate prompt so every sample is expected to reach the
    # token cap; arbitrary short token-id prompts may legitimately emit EOS.
    prompt = fixed_prompt_ids()
    request_rows: list[dict[str, Any]] = []
    ttfts_by_model: dict[str, list[float]] = {model: [] for model in args.models}
    latencies_by_model: dict[str, list[float]] = {model: [] for model in args.models}
    errors_by_model: dict[str, int] = {model: 0 for model in args.models}
    errors = 0
    for index in range(args.samples):
        for model in args.models:
            result = client.post_sse(
                "/v1/completions",
                completion_payload(
                    model,
                    prompt,
                    "servetest-hourly-monitor",
                    stream=True,
                    max_tokens=60,
                ),
            )
            usage_ok, usage_detail = validate_usage(
                usage_from(result), expected_prompt=len(prompt), expected_completion=60
            )
            ok = (
                result.status == 200
                and result.error is None
                and result.done is True
                and result.request_id is not None
                and result.first_content_ms is not None
                and usage_ok
                and rate_headers_ok(result)
            )
            if ok:
                ttfts_by_model[model].append(float(result.first_content_ms))
                latencies_by_model[model].append(float(result.elapsed_ms))
            else:
                errors += 1
                errors_by_model[model] += 1
            row = {
                "schema": "memra.cx-servetest.monitor-request.v2",
                "monitor_started_utc": started_utc,
                "model": model,
                "sample": index,
                "ok": ok,
                "usage_detail": usage_detail,
                **result.receipt(),
            }
            request_rows.append(row)

    with (temp_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    ready_models = (
        set(ready.body.get("models") or []) if isinstance(ready.body, dict) else set()
    )
    per_model = {}
    for model in args.models:
        ttfts = ttfts_by_model[model]
        latencies = latencies_by_model[model]
        model_errors = errors_by_model[model]
        per_model[model] = {
            "samples": args.samples,
            "successes": args.samples - model_errors,
            "errors": model_errors,
            "error_rate": model_errors / args.samples,
            "ttft_ms": {
                "p50": round(percentile(ttfts, 0.50), 3) if ttfts else None,
                "p95": round(percentile(ttfts, 0.95), 3) if ttfts else None,
            },
            "latency_ms": {
                "p50": round(percentile(latencies, 0.50), 3) if latencies else None,
                "p95": round(percentile(latencies, 0.95), 3) if latencies else None,
            },
        }
    models_ready = all(model in ready_models for model in args.models)
    total_samples = args.samples * len(args.models)
    summary = {
        "schema": "memra.cx-servetest.hourly.v2",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "base_url": base_url,
        "metrics_base_url": metrics_base_url,
        "models": args.models,
        "models_ready": models_ready,
        "samples_per_model": args.samples,
        "samples": total_samples,
        "successes": total_samples - errors,
        "errors": errors,
        "error_rate": errors / total_samples,
        "per_model": per_model,
        "health_status": health.status,
        "ready_status": ready.status,
        "metrics_status": metrics.status,
        "server": process,
        "gpu_probe_rc": gpu.returncode,
    }
    write_json(temp_dir / "summary.json", summary)
    write_manifest(temp_dir)
    append_ledger(pathlib.Path(args.ledger), request_rows)
    temp_dir.rename(final_dir)
    latest = out_root / "latest"
    link_path = out_root / f".{latest.name}.{os.getpid()}.tmp"
    try:
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(final_dir.name)
        link_path.replace(latest)
    finally:
        link_path.unlink(missing_ok=True)
    print(json.dumps(summary, sort_keys=True))
    return 0 if (
        errors == 0
        and models_ready
        and health.status == 200
        and ready.status == 200
        and metrics.status == 200
        and process.get("alive") is True
        and gpu.returncode == 0
    ) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url-file", required=True)
    parser.add_argument("--metrics-base-url")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--metrics-token-file", required=True)
    parser.add_argument("--model", dest="models", action="append", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--lock-file", default="/run/memra-servetest-monitor.lock")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
