#!/usr/bin/env python3
"""Drive the four capbase box1 measurement workloads."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

VAL256 = pathlib.Path(__file__).resolve().parents[1] / "val256-20260809"
sys.path.insert(0, str(VAL256))

from run_admission_workload import append, burst, metrics, request  # noqa: E402


def capacity(args: argparse.Namespace) -> int:
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")
    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "server_ctx": 262144,
            "requested_max_ctx": args.max_ctx,
            "offered_concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
    )
    append(args.rows, {"kind": "metrics", "phase": "before", "value": metrics(args.base)})

    release = threading.Barrier(args.concurrency)
    rows: list[dict] = []
    write_lock = threading.Lock()

    def one(index: int) -> None:
        row = request(
            args.base,
            f"capacity-{args.max_ctx}",
            f"burst-{args.max_ctx}",
            index,
            args.max_ctx,
            args.max_tokens,
            release,
        )
        with write_lock:
            rows.append(row)
            append(args.rows, row)

    threads = [threading.Thread(target=one, args=(index,)) for index in range(args.concurrency)]
    for thread in threads:
        thread.start()

    peak_active = 0
    last_active: int | None = None
    while any(thread.is_alive() for thread in threads):
        sample = metrics(args.base)
        active = sample.get("active_sessions")
        if isinstance(active, int):
            peak_active = max(peak_active, active)
        if active != last_active:
            with write_lock:
                append(
                    args.rows,
                    {"kind": "metrics_sample", "active_sessions": active, "value": sample},
                )
            last_active = active
        time.sleep(0.05)
    for thread in threads:
        thread.join()

    append(args.rows, {"kind": "metrics", "phase": "after", "value": metrics(args.base)})
    completed = sum(bool(row.get("ok")) for row in rows)
    starts = [row["started_unix_s"] for row in rows]
    append(
        args.rows,
        {
            "kind": "summary",
            "n": 1,
            "requests_ok": completed,
            "requests_n": len(rows),
            "peak_active_sessions_sampled": peak_active,
            "request_start_spread_ms": round((max(starts) - min(starts)) * 1000, 3),
        },
    )
    return 0 if completed == len(rows) == args.concurrency else 1


def burst_only(args: argparse.Namespace) -> int:
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")
    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "server_ctx": 262144,
            "requested_max_ctx": 8192,
            "concurrency": args.concurrency,
            "max_tokens": 64,
            "temperature": 0,
        },
    )
    append(args.rows, {"kind": "metrics", "phase": "before", "value": metrics(args.base)})
    rows = burst(
        args.base,
        args.rows,
        f"burst-c{args.concurrency}",
        "small8k",
        args.concurrency,
        8192,
        64,
    )
    final = metrics(args.base)
    append(args.rows, {"kind": "metrics", "phase": "after", "value": final})
    ordered = sorted(rows, key=lambda row: row.get("ttfb_s", float("inf")))
    ttfb = [row["ttfb_s"] for row in rows if "ttfb_s" in row]
    starts = [row["started_unix_s"] for row in rows]
    summary = {
        "kind": "summary",
        "n": 1,
        "concurrency": args.concurrency,
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "requests_n": len(rows),
        "service_order": [
            {"index": row["index"], "ttfb_s": row.get("ttfb_s")} for row in ordered
        ],
        "ttfb_span_s": round(max(ttfb) - min(ttfb), 3) if len(ttfb) == len(rows) else None,
        "request_start_spread_ms": round((max(starts) - min(starts)) * 1000, 3),
        "step_p50_ms": final.get("step_p50_ms"),
        "step_p99_ms": final.get("step_p99_ms"),
        "step_oom_parks": final.get("step_oom_parks"),
    }
    append(args.rows, summary)
    return 0 if summary["requests_ok"] == args.concurrency else 1


def prompt_request(base: str, worker: int, sequence: int) -> dict:
    prompt_ids = [
        5_000 + ((index + sequence * 131 + worker * 17) % 1_024)
        for index in range(8_000)
    ]
    started_wall = time.time()
    started_mono = time.monotonic()
    body = {
        "model": "step",
        "prompt_ids": prompt_ids,
        "max_ctx": 8192,
        "max_tokens": 128,
        "temperature": 0,
        "stream": True,
        "cache_salt": "capbase-sustained",
        "session_id": f"capbase-sustained-{worker}-{sequence}",
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    row = {
        "kind": "request",
        "worker": worker,
        "sequence": sequence,
        "prompt_tokens_requested": len(prompt_ids),
        "max_ctx": 8192,
        "max_tokens": 128,
        "started_unix_s": round(started_wall, 6),
        "ok": False,
        "chunks": 0,
        "done": False,
        "finish_reason": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            row["http_status"] = response.status
            first_byte = None
            for raw in response:
                if first_byte is None:
                    first_byte = time.monotonic()
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    row["done"] = True
                    break
                row["chunks"] += 1
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    row["bad_json"] = payload[:300]
                    continue
                if "error" in event:
                    error = event["error"]
                    row["server_error"] = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )[:500]
                    continue
                choices = event.get("choices") or [{}]
                if choices[0].get("finish_reason"):
                    row["finish_reason"] = choices[0]["finish_reason"]
            if first_byte is not None:
                row["ttfb_s"] = round(first_byte - started_mono, 3)
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode("utf-8", "replace")[:500]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:500]
    row["wall_s"] = round(time.monotonic() - started_mono, 3)
    row["ok"] = (
        row.get("http_status") == 200
        and row["chunks"] > 0
        and row["done"]
        and row["finish_reason"] in ("stop", "length")
        and not any(name in row for name in ("bad_json", "server_error", "error"))
    )
    return row


def sustained(args: argparse.Namespace) -> int:
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")
    before = metrics(args.base)
    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "server_ctx": 262144,
            "requested_max_ctx": 8192,
            "prompt_tokens": 8000,
            "max_tokens": 128,
            "concurrency": args.concurrency,
            "target_duration_s": args.duration,
            "temperature": 0,
        },
    )
    append(args.rows, {"kind": "metrics", "phase": "before", "value": before})

    release = threading.Barrier(args.concurrency + 1)
    stop = threading.Event()
    write_lock = threading.Lock()
    rows: list[dict] = []

    def worker(worker_id: int) -> None:
        release.wait()
        sequence = 0
        while not stop.is_set():
            row = prompt_request(args.base, worker_id, sequence)
            with write_lock:
                rows.append(row)
                append(args.rows, row)
            sequence += 1

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.concurrency)]
    for thread in threads:
        thread.start()
    started = time.monotonic()
    release.wait()
    stop.wait(args.duration)
    stop.set()
    window_end = metrics(args.base)
    window_elapsed = time.monotonic() - started
    append(args.rows, {"kind": "metrics", "phase": "window_end", "value": window_end})
    for thread in threads:
        thread.join()
    final = metrics(args.base)
    append(args.rows, {"kind": "metrics", "phase": "after_drain", "value": final})

    token_delta = window_end.get("tokens_out", 0) - before.get("tokens_out", 0)
    wall_times = [row["wall_s"] for row in rows if row.get("ok")]
    summary = {
        "kind": "summary",
        "n": 1,
        "thermal_regime": "continuous one-second nvidia-smi sampling under one exclusive lock",
        "target_duration_s": args.duration,
        "window_elapsed_s": round(window_elapsed, 3),
        "tokens_out_window": token_delta,
        "aggregate_tok_s": round(token_delta / window_elapsed, 3),
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "requests_n": len(rows),
        "latency_p50_s": round(statistics.median(wall_times), 3) if wall_times else None,
        "step_p50_ms": window_end.get("step_p50_ms"),
        "step_p99_ms": window_end.get("step_p99_ms"),
        "step_oom_parks": window_end.get("step_oom_parks"),
        "admission_vram_defers": window_end.get("admission_vram_defers"),
        "active_at_window_end": window_end.get("active_sessions"),
    }
    append(args.rows, summary)
    return 0 if summary["requests_ok"] == summary["requests_n"] and token_delta > 0 else 1


def park(args: argparse.Namespace) -> int:
    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")
    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "server_ctx": 262144,
            "sequence": "one explicit-262k park; c=4 explicit-8k greedy burst",
        },
    )
    parked = request(args.base, "parkfull", "park262k", 0, 262144, 16)
    append(args.rows, parked)
    if not parked.get("ok"):
        return 2
    after_park = metrics(args.base)
    append(args.rows, {"kind": "metrics", "phase": "after_park", "value": after_park})
    rows = burst(args.base, args.rows, "pressure", "burst8k", 4, 8192, 64)
    final = metrics(args.base)
    append(args.rows, {"kind": "metrics", "phase": "after_burst", "value": final})
    ordered = sorted(rows, key=lambda row: row.get("ttfb_s", float("inf")))
    ttfb = [row["ttfb_s"] for row in rows if "ttfb_s" in row]
    append(
        args.rows,
        {
            "kind": "summary",
            "n": 1,
            "park_ok": bool(parked.get("ok")),
            "parked_plain_entries": after_park.get("continuation_pool_entries"),
            "parked_spec_entries": after_park.get("spec_pool_entries"),
            "burst_ok": sum(bool(row.get("ok")) for row in rows),
            "burst_n": len(rows),
            "service_order": [
                {"index": row["index"], "ttfb_s": row.get("ttfb_s")} for row in ordered
            ],
            "ttfb_span_s": round(max(ttfb) - min(ttfb), 3) if len(ttfb) == 4 else None,
            "step_oom_parks": final.get("step_oom_parks"),
            "admission_vram_defers": final.get("admission_vram_defers"),
            "continuation_pool_evictions": final.get("continuation_pool_evictions"),
            "spec_pool_evictions": final.get("spec_pool_evictions"),
        },
    )
    return 0 if all(row.get("ok") for row in rows) else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capacity")
    cap.add_argument("base")
    cap.add_argument("rows", type=pathlib.Path)
    cap.add_argument("--max-ctx", type=int, required=True)
    cap.add_argument("--concurrency", type=int, default=24)
    cap.add_argument("--max-tokens", type=int, default=64)
    cap.set_defaults(func=capacity)

    timing = sub.add_parser("burst")
    timing.add_argument("base")
    timing.add_argument("rows", type=pathlib.Path)
    timing.add_argument("--concurrency", type=int, required=True)
    timing.set_defaults(func=burst_only)

    load = sub.add_parser("sustained")
    load.add_argument("base")
    load.add_argument("rows", type=pathlib.Path)
    load.add_argument("--concurrency", type=int, default=8)
    load.add_argument("--duration", type=float, default=600.0)
    load.set_defaults(func=sustained)

    pressure = sub.add_parser("park")
    pressure.add_argument("base")
    pressure.add_argument("rows", type=pathlib.Path)
    pressure.set_defaults(func=park)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
