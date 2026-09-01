#!/usr/bin/env python3
"""Minimal Q35 mixed-c=2 exactness recorder.

The sellgate harness treated response usage as the client token count. This
reducer counts blank-line-delimited SSE token events independently. In native
compatibility mode it also records every token id, so content identity can be
checked without retokenizing streamed text.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, separators=(",", ":")).encode())


def load_workload(path: Path) -> dict[str, Any]:
    workload = json.loads(path.read_text(encoding="utf-8"))
    if workload.get("schema") != "memra.sellgate.workload.v1":
        raise ValueError("unsupported workload lock schema")
    if int(workload["prompt_tokens"]) != 4_860:
        raise ValueError("this reducer is pinned to the sellgate Q35 prompt length")
    if int(workload["completion_tokens"]) != 60:
        raise ValueError("this reducer is pinned to the sellgate completion length")
    return workload


def prompt_ids(workload: dict[str, Any]) -> list[int]:
    family = workload["prompt_family"]
    count = int(workload["prompt_tokens"])
    offset = int(family["offset"])
    seed = int(family["seed"])
    ids = [5_000 + ((position + offset + seed * 131) % 1_024) for position in range(count)]
    actual = canonical_sha256(ids)
    expected = str(family["prompt_ids_sha256_canonical_json"])
    if actual != expected:
        raise ValueError(f"prompt hash mismatch: {actual} != {expected}")
    return ids


def scrape(base: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(base + "/metrics", timeout=timeout) as response:
        return json.load(response)


def metric(row: dict[str, Any], name: str) -> int:
    return int(row.get(name) or 0)


def sse_events(response: Any) -> Iterator[tuple[str, str]]:
    """Yield complete WHATWG SSE event blocks from an HTTP response."""
    event_name = "message"
    data: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield event_name, "\n".join(data)
            event_name = "message"
            data = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data.append(value)
    # A response that closes without the blank-line dispatch delimiter is incomplete.


def normalize_native_finish(stop_reason: str | None) -> str | None:
    if stop_reason in ("Eos", "Callback"):
        return "stop"
    if stop_reason in ("MaxNew", "ContextFull"):
        return "length"
    return None


def request(
    base: str,
    model: str,
    compat: str,
    prompt: list[int],
    salt: str,
    workload: dict[str, Any],
    timeout: float,
    barrier: threading.Barrier | None = None,
    go: threading.Event | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt_ids": prompt,
        "max_ctx": len(prompt) + int(workload["completion_tokens"]) + 8,
        "max_tokens": int(workload["completion_tokens"]),
        "temperature": int(workload["temperature"]),
        "seed": int(workload["seed"]),
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    http_request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=60)
    if go is not None:
        go.wait(timeout=60)

    started = time.monotonic()
    http_status: int | None = None
    token_ids: list[int] = []
    text: list[str] = []
    usage: dict[str, Any] = {}
    native_done: dict[str, Any] = {}
    request_id: str | None = None
    finish_reason: str | None = None
    sse_event_count = 0
    token_event_count = 0
    terminal_event_count = 0
    done_marker_count = 0
    error: str | None = None

    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            http_status = response.status
            for event_name, payload in sse_events(response):
                sse_event_count += 1
                if payload == "[DONE]":
                    done_marker_count += 1
                    continue
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                if compat == "native":
                    if event_name == "done":
                        terminal_event_count += 1
                        native_done = event
                        finish_reason = normalize_native_finish(event.get("stop_reason"))
                    else:
                        token_event_count += 1
                        token_ids.append(int(event["id"]))
                        text.append(str(event.get("text") or ""))
                else:
                    request_id = event.get("id") or request_id
                    usage = event.get("usage") or usage
                    choices = event.get("choices") or []
                    for choice in choices:
                        choice_finish = choice.get("finish_reason")
                        if choice_finish is None:
                            token_event_count += 1
                            text.append(str(choice.get("text") or ""))
                        else:
                            terminal_event_count += 1
                            finish_reason = str(choice_finish)
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        error = exc.read().decode(errors="replace")[:2_000]
    except Exception as exc:  # preserve the exact transport/parser exception in the receipt
        error = f"{type(exc).__name__}: {exc}"[:2_000]

    ended = time.monotonic()
    encoded = "".join(text).encode()
    reported_tokens = (
        native_done.get("n_tokens") if compat == "native" else usage.get("completion_tokens")
    )
    prompt_tokens = (
        native_done.get("prompt_tokens") if compat == "native" else usage.get("prompt_tokens")
    )
    cached = (
        native_done.get("cached_tokens")
        if compat == "native"
        else (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    )
    terminal_ok = terminal_event_count == 1 and (
        done_marker_count == 0 if compat == "native" else done_marker_count == 1
    )
    row = {
        "http_status": http_status,
        "request_id": request_id,
        "sse_event_count": sse_event_count,
        "token_event_count": token_event_count,
        "terminal_event_count": terminal_event_count,
        "done_marker_count": done_marker_count,
        "reported_completion_tokens": reported_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached,
        "finish_reason": finish_reason,
        "native_stop_reason": native_done.get("stop_reason"),
        "text_bytes": len(encoded),
        "text_sha256": sha256_bytes(encoded),
        "token_ids": token_ids if compat == "native" else None,
        "token_ids_sha256": canonical_sha256(token_ids) if compat == "native" else None,
        "latency_ms": (ended - started) * 1_000.0,
        "error": error,
    }
    row["wire_count_exact"] = bool(
        isinstance(reported_tokens, int) and token_event_count == reported_tokens
    )
    row["transport_ok"] = bool(
        http_status == 200
        and error is None
        and terminal_ok
        and finish_reason in ("stop", "length")
        and row["wire_count_exact"]
    )
    return row


def hot_salt(namespace: str, template: int) -> str:
    return f"{namespace}-q35-hot-{template}"


def jobs(workload: dict[str, Any], namespace: str, rep: int) -> list[dict[str, Any]]:
    concurrency = 2
    request_n = max(int(workload["minimum_requests_per_cell"]), concurrency)
    cycle = int(workload["hit_requests_per_cycle"]) + int(workload["miss_requests_per_cycle"])
    request_n = math.ceil(request_n / cycle) * cycle
    hit_n = request_n * int(workload["hit_requests_per_cycle"]) // cycle
    roles = ["hit"] * hit_n + ["miss"] * (request_n - hit_n)
    random.Random(int(workload["seed"]) + rep * 1009 + concurrency * 9173).shuffle(roles)
    prompt = prompt_ids(workload)
    hit_index = 0
    result: list[dict[str, Any]] = []
    for index, role in enumerate(roles):
        if role == "hit":
            template = hit_index % int(workload["hot_cache_entries"])
            hit_index += 1
            salt = hot_salt(namespace, template)
            expected_cached = len(prompt)
        else:
            template = None
            salt = f"{namespace}-q35-mixed90-r{rep}-c2-i{index}"
            expected_cached = 0
        result.append({
            "index": index,
            "cache_role": role,
            "template": template,
            "salt": salt,
            "expected_cached_tokens": expected_cached,
        })
    return result


def wait_settled(base: str, completed_before: int, expected: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + min(timeout, 180.0)
    current = scrape(base, timeout)
    while time.monotonic() < deadline:
        if metric(current, "completed") >= completed_before + expected and metric(current, "active_sessions") == 0:
            return current
        time.sleep(0.1)
        current = scrape(base, timeout)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", default="q35")
    parser.add_argument("--compat", choices=("openai", "native"), required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1_800.0)
    parser.add_argument("--expect-clean", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    workload = load_workload(args.workload_lock)
    prompt = prompt_ids(workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    golden_text: str | None = None
    golden_ids: str | None = None
    all_rows: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    with args.out.open("x", encoding="utf-8") as output:
        protocol = {
            "kind": "protocol",
            "schema": "memra.q35bug.repro.v1",
            "compat": args.compat,
            "base": args.base,
            "model": args.model,
            "concurrency": 2,
            "repetitions": args.repetitions,
            "workload_lock_sha256": sha256_bytes(args.workload_lock.read_bytes()),
            "prompt_ids_sha256_canonical_json": canonical_sha256(prompt),
            "client_count_definition": "blank-line-delimited SSE token events",
            "content_identity": "canonical token-id SHA-256" if args.compat == "native" else "streamed-text SHA-256",
        }
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(protocol, sort_keys=True), flush=True)

        for rep in range(1, args.repetitions + 1):
            seed_before = scrape(args.base, args.timeout)
            for template in range(int(workload["hot_cache_entries"])):
                row = request(
                    args.base, args.model, args.compat, prompt,
                    hot_salt(args.namespace, template), workload, args.timeout,
                )
                row.update({"kind": "seed", "rep": rep, "template": template})
                if row["transport_ok"] and golden_text is None:
                    golden_text = str(row["text_sha256"])
                    golden_ids = row["token_ids_sha256"]
                row["content_matches_golden"] = bool(
                    row["token_ids_sha256"] == golden_ids
                    if args.compat == "native"
                    else row["text_sha256"] == golden_text
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
                all_rows.append(row)
            output.flush()

            # Event::Done reaches the socket before the worker's retire sweep publishes
            # metrics. Wait for all eight seed retires, or a stale idle snapshot can leak
            # their final counter updates into the scored cell delta.
            before = wait_settled(
                args.base,
                metric(seed_before, "completed"),
                int(workload["hot_cache_entries"]),
                args.timeout,
            )
            cell_jobs = jobs(workload, args.namespace, rep)
            barrier = threading.Barrier(3)
            go = threading.Event()

            def one(job: dict[str, Any], first_wave: bool) -> dict[str, Any]:
                row = request(
                    args.base, args.model, args.compat, prompt, str(job["salt"]),
                    workload, args.timeout, barrier if first_wave else None, go,
                )
                row.update({
                    "kind": "request", "rep": rep, "concurrency": 2,
                    "index": job["index"], "cache_role": job["cache_role"],
                    "template": job["template"],
                    "expected_cached_tokens": job["expected_cached_tokens"],
                })
                row["cache_exact"] = row["cached_tokens"] == job["expected_cached_tokens"]
                row["content_matches_golden"] = bool(
                    row["token_ids_sha256"] == golden_ids
                    if args.compat == "native"
                    else row["text_sha256"] == golden_text
                )
                return row

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(one, job, index < 2) for index, job in enumerate(cell_jobs)]
                barrier.wait(timeout=60)
                go.set()
                rows = [future.result(timeout=args.timeout) for future in futures]

            after = wait_settled(args.base, metric(before, "completed"), len(rows), args.timeout)
            client_total = sum(int(row["token_event_count"]) for row in rows)
            response_total = sum(int(row["reported_completion_tokens"] or 0) for row in rows)
            engine_total = metric(after, "tokens_out") - metric(before, "tokens_out")
            cell = {
                "kind": "cell",
                "rep": rep,
                "concurrency": 2,
                "requests": len(rows),
                "client_token_event_total": client_total,
                "response_completion_total": response_total,
                "engine_tokens_out_delta": engine_total,
                "client_response_delta": client_total - response_total,
                "response_engine_delta": response_total - engine_total,
                "finish_reasons": {reason: sum(row["finish_reason"] == reason for row in rows)
                                   for reason in ("length", "stop")},
                "early_stop_requests": sum(int(row["reported_completion_tokens"] or 0) < 60 for row in rows),
                "wire_count_mismatches": sum(not row["wire_count_exact"] for row in rows),
                "cache_mismatches": sum(not row["cache_exact"] for row in rows),
                "content_mismatches": sum(not row["content_matches_golden"] for row in rows),
                "transport_failures": sum(not row["transport_ok"] for row in rows),
            }
            cell["clean"] = bool(
                cell["transport_failures"] == 0
                and cell["wire_count_mismatches"] == 0
                and cell["cache_mismatches"] == 0
                and cell["content_mismatches"] == 0
                and client_total == response_total == engine_total == len(rows) * 60
            )
            for row in rows:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.write(json.dumps(cell, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(cell, sort_keys=True), flush=True)
            all_rows.extend(rows)
            all_cells.append(cell)

        requests = [row for row in all_rows if row["kind"] == "request"]
        summary = {
            "kind": "summary",
            "schema": "memra.q35bug.repro.v1",
            "compat": args.compat,
            "cells": len(all_cells),
            "all_clean": all(bool(cell["clean"]) for cell in all_cells),
            "client_token_event_total": sum(int(row["token_event_count"]) for row in requests),
            "response_completion_total": sum(int(row["reported_completion_tokens"] or 0) for row in requests),
            "engine_tokens_out_delta": sum(int(cell["engine_tokens_out_delta"]) for cell in all_cells),
            "early_stop_requests": sum(int(row["reported_completion_tokens"] or 0) < 60 for row in requests),
            "content_mismatches": sum(not row["content_matches_golden"] for row in requests),
            "wire_count_mismatches": sum(not row["wire_count_exact"] for row in requests),
            "transport_failures": sum(not row["transport_ok"] for row in requests),
        }
        output.write(json.dumps(summary, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(summary, sort_keys=True), flush=True)

    if args.expect_clean and not summary["all_clean"]:
        return 1
    return 0 if summary["transport_failures"] == 0 and summary["wire_count_mismatches"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
