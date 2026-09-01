#!/usr/bin/env python3
"""Barrier/stagger streaming exactness probe derived from darktrain2 qos_probe.py."""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time
import urllib.request


FILLER = (
    "The operator measures latency, allocator state, checkpoint durability, and exact output "
    "while a lower-priority optimizer yields to interactive traffic. "
)
PROMPT = (
    "In exactly four concise bullets, explain why a GPU background job must yield to an "
    "interactive inference request. Include one point about memory. Context: " + FILLER * 5
)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))]


def one_request(
    base: str,
    model: str,
    max_tokens: int,
    timeout: float,
    barrier: threading.Barrier,
    release_box: list[float | None],
    delay_ms: float,
    index: int,
    lane: str,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 3407,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    headers = {"Content-Type": "application/json"}
    if lane:
        headers["x-lane"] = lane
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers=headers,
    )
    barrier.wait()
    release = release_box[0]
    assert release is not None
    target = release + delay_ms / 1000.0
    remaining = target - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    t0 = time.monotonic()
    start_offset_ms = (t0 - release) * 1000.0
    ttft = None
    pieces: list[str] = []
    usage: dict = {}
    rid = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                rid = event.get("id") or rid
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning") or "")
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        pieces.append(piece)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        text = "".join(pieces)
        encoded = text.encode()
        return {
            "index": index,
            "lane": lane or "interactive",
            "scheduled_delay_ms": round(delay_ms, 3),
            "request_start_offset_ms": round(start_offset_ms, 3),
            "ok": True,
            "rid": rid,
            "ttft_s": ttft,
            "first_token_offset_ms": round(start_offset_ms + (ttft or 0.0) * 1000.0, 3),
            "latency_s": time.monotonic() - t0,
            "finish_reason": finish,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "text_utf8_b64": base64.b64encode(encoded).decode(),
            "text_bytes": len(encoded),
            "text_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    except Exception as exc:
        return {
            "index": index,
            "lane": lane or "interactive",
            "scheduled_delay_ms": round(delay_ms, 3),
            "request_start_offset_ms": round(start_offset_ms, 3),
            "ok": False,
            "ttft_s": ttft,
            "latency_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def request_delays(args: argparse.Namespace) -> list[float]:
    if args.delays_ms:
        delays = [float(part) for part in args.delays_ms.split(",")]
        if len(delays) != args.requests:
            raise ValueError(
                f"--delays-ms supplied {len(delays)} values for {args.requests} requests"
            )
        if any(delay < 0 for delay in delays):
            raise ValueError("request delays must be non-negative")
        return delays
    if args.stagger_max_ms < 0:
        raise ValueError("--stagger-max-ms must be non-negative")
    if args.requests == 1:
        return [0.0]
    return [args.stagger_max_ms * i / (args.requests - 1) for i in range(args.requests)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--stagger-max-ms", type=float, default=0.0)
    ap.add_argument("--delays-ms", help="comma-separated release offsets, one per request")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument(
        "--lanes",
        help="comma-separated x-lane sequence, cycled across requests",
    )
    args = ap.parse_args()

    if args.requests < 1:
        ap.error("--requests must be positive")
    if not args.golden.is_file():
        ap.error(f"golden completion missing: {args.golden}")
    try:
        delays = request_delays(args)
    except ValueError as exc:
        ap.error(str(exc))
    lanes = [part.strip() for part in (args.lanes or "").split(",") if part.strip()]
    if any(lane not in {"interactive", "judge", "harvest"} for lane in lanes):
        ap.error("--lanes values must be interactive, judge, or harvest")
    request_lanes = [lanes[i % len(lanes)] if lanes else "" for i in range(args.requests)]

    expected = args.golden.read_bytes()
    expected_sha = hashlib.sha256(expected).hexdigest()
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    barrier = threading.Barrier(args.requests + 1)
    release_box: list[float | None] = [None]
    rows: list[dict | None] = [None] * args.requests

    def worker(index: int) -> None:
        rows[index] = one_request(
            args.base,
            args.model,
            args.max_tokens,
            args.timeout,
            barrier,
            release_box,
            delays[index],
            index,
            request_lanes[index],
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.requests)]
    for thread in threads:
        thread.start()
    release_box[0] = time.monotonic()
    barrier.wait()
    release = release_box[0]
    for thread in threads:
        thread.join()
    assert release is not None
    wall_s = time.monotonic() - release

    final_rows = [row for row in rows if row is not None]
    for row in final_rows:
        if row.get("ok"):
            row["golden_match"] = base64.b64decode(row["text_utf8_b64"]) == expected
    with args.rows.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps({"label": args.label, **row}, sort_keys=True) + "\n")

    oks = [row for row in final_rows if row.get("ok")]
    errors = [row for row in final_rows if not row.get("ok")]
    ttfts = [float(row["ttft_s"]) for row in oks if row.get("ttft_s") is not None]
    lats = [float(row["latency_s"]) for row in oks]
    hash_counts = collections.Counter(str(row["text_sha256"]) for row in oks)
    golden_matches = sum(bool(row.get("golden_match")) for row in oks)
    summary = {
        "label": args.label,
        "requests": args.requests,
        "n_ok": len(oks),
        "n_error": len(errors),
        "wall_s": round(wall_s, 6),
        "scheduled_delays_ms": [round(delay, 3) for delay in delays],
        "lanes": request_lanes,
        "actual_start_offsets_ms": [row.get("request_start_offset_ms") for row in final_rows],
        "ttft_p50_s": round(statistics.median(ttfts), 6) if ttfts else None,
        "ttft_p99_s": round(nearest_rank(ttfts, 0.99), 6) if ttfts else None,
        "latency_p50_s": round(statistics.median(lats), 6) if lats else None,
        "latency_p99_s": round(nearest_rank(lats, 0.99), 6) if lats else None,
        "hash_counts": dict(sorted(hash_counts.items())),
        "expected_sha256": expected_sha,
        "golden_matches": golden_matches,
        "golden_divergences": len(oks) - golden_matches,
        "exactness": "match" if len(oks) == args.requests and golden_matches == len(oks)
        else "mismatch",
        "errors": [row.get("error") for row in errors[:3]],
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    if errors or len(oks) != args.requests:
        return 1
    if golden_matches != len(oks):
        print(f"P0 exactness failure: {len(oks) - golden_matches}/{len(oks)} diverged", flush=True)
        return 86
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
