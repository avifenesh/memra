#!/usr/bin/env python3
"""One barrier-released load window across one or two memra servers."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
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
    "prompt_tokens_in",
    "cached_tokens_in",
    "prefix_cache_hits",
    "admission_session_defers",
    "admission_vram_defers",
    "step_oom_parks",
)


@dataclasses.dataclass(frozen=True)
class Endpoint:
    label: str
    base: str
    model: str


def parse_endpoint(raw: str) -> Endpoint:
    parts = raw.split(",", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "endpoint must be LABEL,BASE_URL,MODEL"
        )
    return Endpoint(parts[0].strip(), parts[1].rstrip("/"), parts[2].strip())


def get_metrics(endpoint: Endpoint) -> dict:
    with urllib.request.urlopen(endpoint.base + "/metrics", timeout=10) as response:
        return json.load(response)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def counter_delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def one_request(
    endpoint: Endpoint,
    args: argparse.Namespace,
    index: int,
    barrier: threading.Barrier,
    go: threading.Event,
    release_box: list[float | None],
) -> dict:
    ordinal = index + 1
    body = {
        "model": endpoint.model,
        "temperature": 0,
        "seed": 3407 + index,
        "max_tokens": args.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": f"Count upward from {ordinal} listing one integer per line.",
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": f"{args.label}-{endpoint.label}-{index}",
    }
    request = urllib.request.Request(
        endpoint.base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait(timeout=60)
    go.wait(timeout=60)
    release = release_box[0]
    assert release is not None
    started = time.monotonic()
    row = {
        "kind": "request",
        "label": args.label,
        "target": endpoint.label,
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
                    delta = choice.get("delta") or {}
                    piece = (
                        (choice.get("text") or "")
                        + (delta.get("content") or "")
                        + (delta.get("reasoning") or "")
                        + (delta.get("reasoning_content") or "")
                    )
                    if piece:
                        first_visible = first_visible or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        row["http_status"] = error.code
        row["error"] = error.read().decode(errors="replace")[:500]
    except Exception as error:  # Preserve the concrete client failure in the receipt.
        row["error"] = f"{type(error).__name__}: {error}"[:500]

    ended = time.monotonic()
    encoded = "".join(pieces).encode()
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    row.update(
        {
            "request_id": request_id,
            "done": done,
            "ttft_s": first_visible - started if first_visible is not None else None,
            "latency_s": ended - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached_tokens,
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
        and cached_tokens in (None, 0)
        and not row.get("error")
    )
    return row


def summarize_target(
    endpoint: Endpoint,
    args: argparse.Namespace,
    rows: list[dict],
    samples: list[dict],
    before: dict,
    after: dict,
    release: float,
) -> dict:
    ended = max(float(row["_ended"]) for row in rows)
    wall_s = ended - release
    ttfts = [float(row["ttft_s"]) for row in rows if row.get("ttft_s") is not None]
    latencies = [float(row["latency_s"]) for row in rows]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    relevant_samples = [row for row in samples if row["target"] == endpoint.label]
    deltas = {key: counter_delta(after, before, key) for key in COUNTERS}
    return {
        "kind": "target_summary",
        "label": args.label,
        "target": endpoint.label,
        "base": endpoint.base,
        "model": endpoint.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "n_requests": len(rows),
        "n_ok": sum(bool(row["ok"]) for row in rows),
        "n_error": sum(not bool(row["ok"]) for row in rows),
        "completion_tokens_total": completion_tokens,
        "prompt_tokens_total": prompt_tokens,
        "wall_s": wall_s,
        "aggregate_tok_s": completion_tokens / wall_s if wall_s > 0 else None,
        "aggregate_prompt_tok_s": prompt_tokens / wall_s if wall_s > 0 else None,
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p99_s": nearest_rank(ttfts, 0.99),
        "ttft_min_s": min(ttfts) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p99_s": nearest_rank(latencies, 0.99),
        "latency_min_s": min(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "request_start_spread_ms": max(
            float(row["request_start_offset_ms"]) for row in rows
        )
        - min(float(row["request_start_offset_ms"]) for row in rows),
        "counter_deltas": deltas,
        "peak_active_sessions_sampled": max(
            (
                int(row["active_sessions"])
                for row in relevant_samples
                if isinstance(row.get("active_sessions"), int)
            ),
            default=0,
        ),
        "peak_queued_requests_sampled": max(
            (
                int(row["queued_requests"])
                for row in relevant_samples
                if isinstance(row.get("queued_requests"), int)
            ),
            default=0,
        ),
        "spec_before": before.get("spec", {}).get(endpoint.model),
        "spec_after": after.get("spec", {}).get(endpoint.model),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        action="append",
        type=parse_endpoint,
        required=True,
        help="LABEL,BASE_URL,MODEL; pass once for solo or twice for paired",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sample-ms", type=float, default=250.0)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.sample_ms <= 0:
        parser.error("concurrency, max-tokens, and sample-ms must be positive")
    if len(args.endpoint) not in (1, 2):
        parser.error("pass one or two --endpoint values")
    if len({endpoint.label for endpoint in args.endpoint}) != len(args.endpoint):
        parser.error("endpoint labels must be unique")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    before = {endpoint.label: get_metrics(endpoint) for endpoint in args.endpoint}
    request_count = args.concurrency * len(args.endpoint)
    barrier = threading.Barrier(request_count + 1)
    go = threading.Event()
    release_box: list[float | None] = [None]
    samples: list[dict] = []
    jobs = [
        (endpoint, index)
        for endpoint in args.endpoint
        for index in range(args.concurrency)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as pool:
        futures = [
            pool.submit(one_request, endpoint, args, index, barrier, go, release_box)
            for endpoint, index in jobs
        ]
        barrier.wait(timeout=60)
        release_box[0] = time.monotonic()
        go.set()
        while not all(future.done() for future in futures):
            for endpoint in args.endpoint:
                try:
                    sample = get_metrics(endpoint)
                    samples.append(
                        {
                            "kind": "metrics_sample",
                            "target": endpoint.label,
                            "elapsed_s": round(
                                time.monotonic() - release_box[0], 6
                            ),
                            "active_sessions": sample.get("active_sessions"),
                            "queued_requests": sample.get("queued_requests"),
                            "tokens_out": sample.get("tokens_out"),
                            "admission_session_defers": sample.get(
                                "admission_session_defers"
                            ),
                            "admission_vram_defers": sample.get(
                                "admission_vram_defers"
                            ),
                            "step_oom_parks": sample.get("step_oom_parks"),
                        }
                    )
                except Exception as error:
                    samples.append(
                        {
                            "kind": "metrics_sample",
                            "target": endpoint.label,
                            "elapsed_s": round(
                                time.monotonic() - release_box[0], 6
                            ),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            time.sleep(args.sample_ms / 1_000.0)
        rows = [future.result() for future in futures]

    after: dict[str, dict] = {}
    for endpoint in args.endpoint:
        expected_completed = int(before[endpoint.label].get("completed", 0)) + args.concurrency
        current = get_metrics(endpoint)
        for _ in range(100):
            if (
                int(current.get("completed", 0)) >= expected_completed
                and int(current.get("active_sessions", 0)) == 0
            ):
                break
            time.sleep(0.1)
            current = get_metrics(endpoint)
        after[endpoint.label] = current

    release = release_box[0]
    assert release is not None
    summaries = []
    clean = True
    for endpoint in args.endpoint:
        target_rows = [row for row in rows if row["target"] == endpoint.label]
        summary = summarize_target(
            endpoint,
            args,
            target_rows,
            samples,
            before[endpoint.label],
            after[endpoint.label],
            release,
        )
        summaries.append(summary)
        expected_tokens = args.concurrency * args.max_tokens
        deltas = summary["counter_deltas"]
        clean = clean and bool(
            summary["n_ok"] == summary["n_requests"] == args.concurrency
            and summary["completion_tokens_total"] == expected_tokens
            and deltas["admitted"] == args.concurrency
            and deltas["completed"] == args.concurrency
            and deltas["tokens_out"] == expected_tokens
        )

    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    run = {
        "kind": "run",
        "label": args.label,
        "protocol": "global barrier across all target requests; streaming content TTFT",
        "targets": [dataclasses.asdict(endpoint) for endpoint in args.endpoint],
        "concurrency_per_target": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "cache_namespace": "unique per request; cached_tokens must be zero when reported",
    }
    receipt = [run]
    for endpoint in args.endpoint:
        receipt.append(
            {
                "kind": "metrics",
                "phase": "before",
                "target": endpoint.label,
                "value": before[endpoint.label],
            }
        )
    receipt.extend(samples)
    receipt.extend(public_rows)
    for endpoint in args.endpoint:
        receipt.append(
            {
                "kind": "metrics",
                "phase": "after",
                "target": endpoint.label,
                "value": after[endpoint.label],
            }
        )
    receipt.extend(summaries)
    receipt.append(
        {
            "kind": "window_summary",
            "label": args.label,
            "targets": [summary["target"] for summary in summaries],
            "concurrency_per_target": args.concurrency,
            "clean": clean,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in receipt),
        encoding="utf-8",
    )
    print(json.dumps(receipt[-1], sort_keys=True), flush=True)
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
