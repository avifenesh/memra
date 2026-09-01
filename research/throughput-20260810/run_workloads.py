#!/usr/bin/env python3
"""Run one exact-shape Step-3.7 serving cell and emit an append-only JSONL receipt."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request


def append(path: pathlib.Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def metrics(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
            return json.load(response)
    except Exception as error:
        return {"metrics_error": f"{type(error).__name__}: {error}"}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def prompt_ids(n_tokens: int, request_index: int, seed: int) -> list[int]:
    # Pin the eight variants that all reached MaxNew in the first c=8 pilot. Higher
    # concurrencies repeat those variants; each request still has its own cache namespace,
    # so repetition cannot turn the cell into prefix reuse/dedup. The caller seed is a cell
    # identity only and deliberately does not change the model input across A/B arms.
    del seed
    family_seed = 1_008
    variant = request_index % 8
    return [
        5_000 + ((position + variant * 17 + family_seed * 131) % 1_024)
        for position in range(n_tokens)
    ]


def one_request(
    base: str,
    label: str,
    request_index: int,
    n_prompt: int,
    max_tokens: int,
    seed: int,
    release: threading.Barrier,
    timeout: float,
) -> dict:
    ids = prompt_ids(n_prompt, request_index, seed)
    body = {
        "model": "step",
        "prompt_ids": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "cache_salt": f"cx-throughput-{label}-{request_index}",
        "session_id": f"cx-throughput-{label}-{request_index}",
    }
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    release.wait()
    started_wall = time.time()
    started_mono = time.monotonic()
    row = {
        "kind": "request",
        "label": label,
        "index": request_index,
        "prompt_tokens_requested": n_prompt,
        "max_tokens": max_tokens,
        "max_ctx": None,
        "started_unix_s": round(started_wall, 6),
        "ok": False,
        "sse_events": 0,
        "content_events": 0,
        "done": False,
        "finish_reason": None,
    }
    first_byte_mono: float | None = None
    first_text_mono: float | None = None
    usage: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            row["http_status"] = response.status
            for raw in response:
                now = time.monotonic()
                if first_byte_mono is None:
                    first_byte_mono = now
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    row["done"] = True
                    break
                row["sse_events"] += 1
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    row["bad_json"] = payload[:300]
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                if "error" in event:
                    error = event["error"]
                    row["server_error"] = (
                        error.get("message", str(error))
                        if isinstance(error, dict)
                        else str(error)
                    )[:500]
                    continue
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (
                        (choice.get("text") or "")
                        + (delta.get("content") or "")
                        + (delta.get("reasoning") or "")
                    )
                    if piece:
                        row["content_events"] += 1
                        if first_text_mono is None:
                            first_text_mono = now
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode("utf-8", "replace")[:500]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:500]
    ended_mono = time.monotonic()
    row["wall_s"] = round(ended_mono - started_mono, 6)
    if first_byte_mono is not None:
        row["ttfb_s"] = round(first_byte_mono - started_mono, 6)
    if first_text_mono is not None:
        row["ttft_s"] = round(first_text_mono - started_mono, 6)
    if usage:
        row["usage"] = usage
    row["ok"] = (
        row.get("http_status") == 200
        and row["content_events"] > 0
        and row["done"]
        and row["finish_reason"] in ("stop", "length")
        and not any(name in row for name in ("bad_json", "server_error", "error"))
    )
    row["_started_mono"] = started_mono
    row["_ended_mono"] = ended_mono
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("rows", type=pathlib.Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    if args.rows.exists():
        raise SystemExit(f"refusing to append a second run to {args.rows}")
    if args.concurrency < 1 or args.prompt_tokens < 1 or args.max_tokens < 1:
        raise SystemExit("concurrency, prompt tokens, and max tokens must be positive")
    args.rows.parent.mkdir(parents=True, exist_ok=True)

    before = metrics(args.base)
    append(
        args.rows,
        {
            "kind": "run",
            "n": 1,
            "label": args.label,
            "concurrency": args.concurrency,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "request_max_ctx": None,
            "server_ctx": 262_144,
            "cell_seed": args.seed,
            "prompt_family": "safe-c8-v1",
            "prompt_family_seed": 1_008,
            "release": "simultaneous threading.Barrier",
        },
    )
    append(args.rows, {"kind": "metrics", "phase": "before", "value": before})

    release = threading.Barrier(args.concurrency + 1)
    rows: list[dict] = []
    rows_lock = threading.Lock()

    def worker(index: int) -> None:
        row = one_request(
            args.base,
            args.label,
            index,
            args.prompt_tokens,
            args.max_tokens,
            args.seed,
            release,
            args.timeout,
        )
        with rows_lock:
            rows.append(row)
            public = {key: value for key, value in row.items() if not key.startswith("_")}
            append(args.rows, public)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.concurrency)]
    for thread in threads:
        thread.start()
    release.wait()

    peak_active = 0
    peak_queued = 0
    while any(thread.is_alive() for thread in threads):
        sample = metrics(args.base)
        active = sample.get("active_sessions")
        queued = sample.get("queued_requests")
        if isinstance(active, int):
            peak_active = max(peak_active, active)
        if isinstance(queued, int):
            peak_queued = max(peak_queued, queued)
        append(
            args.rows,
            {
                "kind": "metrics_sample",
                "active_sessions": active,
                "queued": queued,
                "tokens_out": sample.get("tokens_out"),
                "admission_vram_defers": sample.get("admission_vram_defers"),
            },
        )
        time.sleep(1.0)
    for thread in threads:
        thread.join()

    expected_completed = int(before.get("completed", 0)) + args.concurrency
    after = metrics(args.base)
    for _ in range(100):
        if (
            after.get("completed", 0) >= expected_completed
            and after.get("active_sessions", 0) == 0
        ):
            break
        time.sleep(0.1)
        after = metrics(args.base)
    append(args.rows, {"kind": "metrics", "phase": "after", "value": after})

    starts = [row["_started_mono"] for row in rows]
    ends = [row["_ended_mono"] for row in rows]
    wall_s = max(ends) - min(starts) if starts and ends else 0.0
    token_delta = int(after.get("tokens_out", 0)) - int(before.get("tokens_out", 0))
    defers_delta = int(after.get("admission_vram_defers", 0)) - int(
        before.get("admission_vram_defers", 0)
    )
    session_defers_delta = int(after.get("admission_session_defers", 0)) - int(
        before.get("admission_session_defers", 0)
    )
    ttfts = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    ttft_p50 = statistics.median(ttfts) if ttfts else None
    expected_tokens = args.concurrency * args.max_tokens
    summary = {
        "kind": "summary",
        "n": 1,
        "label": args.label,
        "thermal_regime": "one-second GPU sampling; arm order interleaved under one lock",
        "concurrency": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "requests_ok": sum(bool(row.get("ok")) for row in rows),
        "requests_n": len(rows),
        "length_finishes": sum(row.get("finish_reason") == "length" for row in rows),
        "expected_output_tokens": expected_tokens,
        "metrics_output_tokens": token_delta,
        "wall_s": round(wall_s, 6),
        "aggregate_output_tok_s": round(token_delta / wall_s, 6) if wall_s > 0 else None,
        "ttft_p50_s": round(ttft_p50, 6) if ttft_p50 is not None else None,
        "ttft_p95_s": round(percentile(ttfts, 95), 6) if ttfts else None,
        "request_start_spread_ms": round((max(starts) - min(starts)) * 1_000, 3),
        "step_p50_ms": after.get("step_p50_ms"),
        "step_p99_ms": after.get("step_p99_ms"),
        "admission_vram_defers": defers_delta,
        "admission_session_defers": session_defers_delta,
        "step_oom_parks": int(after.get("step_oom_parks", 0))
        - int(before.get("step_oom_parks", 0)),
        "peak_active_sessions_sampled": peak_active,
        "peak_queued_sampled": peak_queued,
    }
    append(args.rows, summary)
    clean = (
        summary["requests_ok"] == summary["requests_n"] == args.concurrency
        and summary["length_finishes"] == args.concurrency
        and token_delta == expected_tokens
        and defers_delta == 0
        and session_defers_delta == 0
        and summary["step_oom_parks"] == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
