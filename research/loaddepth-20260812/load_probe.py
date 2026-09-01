#!/usr/bin/env python3
"""One simultaneous Step-3.7 load window with request and worker telemetry."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request


COUNTERS = (
    "admitted",
    "completed",
    "tokens_out",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


def get_metrics(base: str) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as response:
        return json.load(response)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def dual_delta(after: dict, before: dict) -> dict:
    a = after.get("dual_pp") or {}
    b = before.get("dual_pp") or {}
    a_uses = a.get("slot_uses") or [0, 0]
    b_uses = b.get("slot_uses") or [0, 0]
    return {
        "overlaps": int(a.get("overlaps", 0)) - int(b.get("overlaps", 0)),
        "slot_pairs": int(a.get("slot_pairs", 0)) - int(b.get("slot_pairs", 0)),
        "slot_uses": [int(a_uses[i]) - int(b_uses[i]) for i in range(2)],
        "slot_collisions": int(a.get("slot_collisions", 0))
        - int(b.get("slot_collisions", 0)),
    }


def one_request(
    args: argparse.Namespace,
    index: int,
    barrier: threading.Barrier,
    release_box: list[float | None],
) -> dict:
    ordinal = index + 1
    body = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": f"Count upward from {ordinal} listing one integer per line.",
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        args.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait(timeout=30)
    release = release_box[0]
    assert release is not None
    started = time.monotonic()
    row = {
        "kind": "request",
        "label": args.label,
        "index": index,
        "request_start_offset_ms": round((started - release) * 1_000, 3),
        "ok": False,
    }
    first_visible: float | None = None
    pieces: list[str] = []
    usage: dict = {}
    finish_reason = None
    request_id = None
    done = False
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            row["http_status"] = response.status
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"], sort_keys=True))
                request_id = event.get("id") or request_id
                usage = event.get("usage") or usage
                for choice in event.get("choices") or []:
                    delta_body = choice.get("delta") or {}
                    piece = (
                        (choice.get("text") or "")
                        + (delta_body.get("content") or "")
                        + (delta_body.get("reasoning") or "")
                        + (delta_body.get("reasoning_content") or "")
                    )
                    if piece:
                        first_visible = first_visible or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode(errors="replace")[:500]
    except Exception as error:  # The receipt preserves the concrete client failure.
        row["error"] = f"{type(error).__name__}: {error}"[:500]

    ended = time.monotonic()
    encoded = "".join(pieces).encode()
    row.update(
        {
            "request_id": request_id,
            "done": done,
            "ttft_s": first_visible - started if first_visible is not None else None,
            "latency_s": ended - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish_reason,
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
            "_started": started,
            "_ended": ended,
        }
    )
    row["ok"] = bool(
        row.get("http_status") == 200
        and done
        and first_visible is not None
        and request_id
        and finish_reason == "length"
        and usage.get("completion_tokens") == args.max_tokens
        and not row.get("error")
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sample-ms", type=float, default=250.0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.sample_ms <= 0:
        parser.error("concurrency, max-tokens, and sample-ms must be positive")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    before = get_metrics(args.base)
    barrier = threading.Barrier(args.concurrency + 1)
    release_box: list[float | None] = [None]
    samples: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args, index, barrier, release_box)
            for index in range(args.concurrency)
        ]
        release_box[0] = time.monotonic()
        barrier.wait(timeout=30)
        while not all(future.done() for future in futures):
            try:
                sample = get_metrics(args.base)
                samples.append(
                    {
                        "kind": "metrics_sample",
                        "elapsed_s": round(time.monotonic() - release_box[0], 6),
                        "active_sessions": sample.get("active_sessions"),
                        "queued_requests": sample.get("queued_requests"),
                        "tokens_out": sample.get("tokens_out"),
                        "admission_session_defers": sample.get("admission_session_defers"),
                        "admission_vram_defers": sample.get("admission_vram_defers"),
                        "step_oom_parks": sample.get("step_oom_parks"),
                    }
                )
            except Exception as error:
                samples.append(
                    {
                        "kind": "metrics_sample",
                        "elapsed_s": round(time.monotonic() - release_box[0], 6),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            time.sleep(args.sample_ms / 1_000.0)
        rows = [future.result() for future in futures]

    expected_completed = int(before.get("completed", 0)) + args.concurrency
    after = get_metrics(args.base)
    for _ in range(100):
        if (
            int(after.get("completed", 0)) >= expected_completed
            and int(after.get("active_sessions", 0)) == 0
        ):
            break
        time.sleep(0.1)
        after = get_metrics(args.base)

    release = release_box[0]
    assert release is not None
    ended = max(float(row["_ended"]) for row in rows)
    wall_s = ended - release
    ttfts = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    counter_deltas = {key: delta(after, before, key) for key in COUNTERS}
    dual = dual_delta(after, before)
    summary = {
        "kind": "summary",
        "label": args.label,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "n_requests": len(rows),
        "n_ok": sum(bool(row["ok"]) for row in rows),
        "n_error": sum(not bool(row["ok"]) for row in rows),
        "completion_tokens_total": completion_tokens,
        "wall_s": wall_s,
        "aggregate_tok_s": completion_tokens / wall_s if wall_s > 0 else None,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p99_s": nearest_rank(ttfts, 0.99),
        "ttft_min_s": min(ttfts) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "request_start_spread_ms": max(
            float(row["request_start_offset_ms"]) for row in rows
        )
        - min(float(row["request_start_offset_ms"]) for row in rows),
        "step_p50_ms": after.get("step_p50_ms"),
        "step_p99_ms": after.get("step_p99_ms"),
        "admission_counters": counter_deltas,
        "peak_active_sessions_sampled": max(
            (int(row["active_sessions"]) for row in samples if isinstance(row.get("active_sessions"), int)),
            default=0,
        ),
        "peak_queued_requests_sampled": max(
            (int(row["queued_requests"]) for row in samples if isinstance(row.get("queued_requests"), int)),
            default=0,
        ),
        "dual_pp": dual,
    }
    public_rows = []
    for row in rows:
        public_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
    receipt = [
        {
            "kind": "run",
            "label": args.label,
            "protocol": "simultaneous barrier; flip-battery prompt family; streaming content TTFT",
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "cache_namespace": "unset, matching dualpp flip battery",
        },
        {"kind": "metrics", "phase": "before", "value": before},
        *samples,
        *public_rows,
        {"kind": "metrics", "phase": "after", "value": after},
        summary,
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipt),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)

    expected_tokens = args.concurrency * args.max_tokens
    clean = (
        summary["n_ok"] == summary["n_requests"] == args.concurrency
        and completion_tokens == expected_tokens
        and counter_deltas["admitted"] == args.concurrency
        and counter_deltas["completed"] == args.concurrency
        and counter_deltas["tokens_out"] == expected_tokens
        and dual["slot_pairs"] > 0
        and dual["slot_uses"] == [dual["slot_pairs"], dual["slot_pairs"]]
        and dual["slot_collisions"] == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
