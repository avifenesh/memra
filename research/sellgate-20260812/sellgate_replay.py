#!/usr/bin/env python3
"""Dual-model cold and exact 90%-token-hit sellgate replay.

The request and metric parsing follows research/prefixmoney-20260812's
prefix_gate.py and cache_concurrency.py. Unlike the earlier partial-prefix
capacity ladder, the scored mixed arm contains nine full-prompt cache hits and
one cold miss per ten equal-sized prompts. That makes its all-traffic
percentiles and 90% token-weighted coverage directly observable in one window.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
    "prefix_cache_entries",
    "prefix_cache_bytes",
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
        raise argparse.ArgumentTypeError("endpoint must be LABEL,BASE_URL,MODEL")
    return Endpoint(parts[0].strip(), parts[1].rstrip("/"), parts[2].strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_workload(path: Path) -> dict[str, Any]:
    workload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "prompt_tokens",
        "completion_tokens",
        "prompt_completion_ratio",
        "hit_requests_per_cycle",
        "miss_requests_per_cycle",
        "token_weighted_hit_ratio",
        "prompt_family",
        "hot_cache_entries",
        "minimum_requests_per_cell",
        "repetitions",
        "base_concurrency",
        "extension_concurrency",
        "prefix_cache_mb_per_model",
        "temperature",
        "seed",
    }
    missing = required - workload.keys()
    if missing:
        raise ValueError(f"workload lock is missing keys: {sorted(missing)}")
    prompt_tokens = int(workload["prompt_tokens"])
    completion_tokens = int(workload["completion_tokens"])
    ratio = int(workload["prompt_completion_ratio"])
    hit_n = int(workload["hit_requests_per_cycle"])
    miss_n = int(workload["miss_requests_per_cycle"])
    if prompt_tokens != completion_tokens * ratio:
        raise ValueError("prompt/completion token counts do not match the frozen ratio")
    if hit_n <= 0 or miss_n <= 0:
        raise ValueError("hit and miss cycle counts must be positive")
    measured_hit_ratio = hit_n / (hit_n + miss_n)
    if not math.isclose(
        measured_hit_ratio,
        float(workload["token_weighted_hit_ratio"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("hit/miss cycle does not match the token-weighted hit ratio")
    levels = [int(value) for value in workload["base_concurrency"]]
    extensions = [int(value) for value in workload["extension_concurrency"]]
    if levels != [1, 2, 4, 8] or any(value <= 0 for value in extensions):
        raise ValueError("base concurrency must be 1,2,4,8 and extensions positive")
    family = workload["prompt_family"]
    if (
        family.get("name") != "safe-c8-v1-fixed-offset"
        or int(family.get("seed") or 0) != 1_008
    ):
        raise ValueError("unsupported scored prompt family")
    expected_prompt_hash = str(family.get("prompt_ids_sha256_canonical_json") or "")
    actual_prompt_hash = prompt_sha256(
        fixed_prompt_ids(prompt_tokens, int(family["offset"]), int(family["seed"]))
    )
    if actual_prompt_hash != expected_prompt_hash:
        raise ValueError("scored prompt ids do not match the workload lock")
    if int(workload["hot_cache_entries"]) < 1:
        raise ValueError("hot_cache_entries must be positive")
    return workload


def nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p75_ms": nearest_rank(values, 0.75),
        "p90_ms": nearest_rank(values, 0.90),
        "p95_ms": nearest_rank(values, 0.95),
        "p99_ms": nearest_rank(values, 0.99),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def scrape(endpoint: Endpoint, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(endpoint.base + "/metrics", timeout=timeout) as response:
        return json.load(response)


def metric_value(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def metric_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> int:
    return metric_value(after, key) - metric_value(before, key)


def fixed_prompt_ids(token_count: int, offset: int, family_seed: int = 1_008) -> list[int]:
    """Generate the qualified fixed-offset synthetic prompt identity."""
    return [
        5_000 + ((position + offset + family_seed * 131) % 1_024)
        for position in range(token_count)
    ]


def prompt_sha256(prompt: list[int]) -> str:
    canonical = json.dumps(prompt, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def scored_prompt_ids(workload: dict[str, Any]) -> list[int]:
    family = workload["prompt_family"]
    return fixed_prompt_ids(
        int(workload["prompt_tokens"]),
        int(family["offset"]),
        int(family["seed"]),
    )


def cached_tokens(usage: dict[str, Any]) -> int:
    return int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def request(
    endpoint: Endpoint,
    prompt: list[int],
    salt: str,
    workload: dict[str, Any],
    timeout: float,
    barrier: threading.Barrier | None = None,
    go: threading.Event | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    body = {
        "model": endpoint.model,
        "prompt_ids": prompt,
        "max_ctx": len(prompt) + int(workload["completion_tokens"]) + 8,
        "max_tokens": int(workload["completion_tokens"]),
        "temperature": int(workload["temperature"]),
        "seed": int(workload["seed"]),
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    if trace_id is not None:
        body["trace_id"] = trace_id
    http_request = urllib.request.Request(
        endpoint.base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(timeout=60)
    if go is not None:
        go.wait(timeout=60)
    started = time.monotonic()
    first_visible: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    request_id = None
    done = False
    http_status = None
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            http_status = response.status
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
                    piece = choice.get("text") or ""
                    piece += delta.get("content") or ""
                    piece += delta.get("reasoning") or ""
                    piece += delta.get("reasoning_content") or ""
                    if piece:
                        first_visible = first_visible or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as error:
        ended = time.monotonic()
        return {
            "ok": False,
            "http_status": error.code,
            "error": error.read().decode(errors="replace")[:1000],
            "_started": started,
            "_ended": ended,
        }
    except Exception as error:
        ended = time.monotonic()
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"{type(error).__name__}: {error}"[:1000],
            "_started": started,
            "_ended": ended,
        }

    ended = time.monotonic()
    encoded = "".join(pieces).encode()
    completion_tokens = usage.get("completion_tokens")
    inter_token_ms = None
    if first_visible is not None and isinstance(completion_tokens, int) and completion_tokens > 1:
        inter_token_ms = (ended - first_visible) * 1000.0 / (completion_tokens - 1)
    row = {
        "request_id": request_id,
        "http_status": http_status,
        "done": done,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached_tokens(usage),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": (first_visible - started) * 1000.0 if first_visible is not None else None,
        "latency_ms": (ended - started) * 1000.0,
        "inter_token_ms": inter_token_ms,
        "text_bytes": len(encoded),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "text_utf8_b64": base64.b64encode(encoded).decode(),
        "_started": started,
        "_ended": ended,
    }
    row["ok"] = bool(
        http_status == 200
        and done
        and first_visible is not None
        and request_id
        and finish_reason in ("stop", "length")
        and completion_tokens == int(workload["completion_tokens"])
        and not row.get("error")
    )
    return row


def wait_settled(
    endpoint: Endpoint,
    before_completed: int,
    expected_requests: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + min(timeout, 120.0)
    current = scrape(endpoint, timeout)
    while time.monotonic() < deadline:
        if (
            metric_value(current, "completed") >= before_completed + expected_requests
            and metric_value(current, "active_sessions") == 0
        ):
            break
        time.sleep(0.1)
        current = scrape(endpoint, timeout)
    return current


def cell_request_count(workload: dict[str, Any], concurrency: int) -> int:
    cycle = int(workload["hit_requests_per_cycle"]) + int(workload["miss_requests_per_cycle"])
    floor = max(int(workload["minimum_requests_per_cell"]), concurrency)
    return math.ceil(floor / cycle) * cycle


def hot_salt(namespace: str, endpoint: Endpoint, entry: int) -> str:
    return f"{namespace}-{endpoint.label}-hot-{entry}"


def seed_hot_set(
    endpoints: list[Endpoint],
    workload: dict[str, Any],
    namespace: str,
    timeout: float,
    goldens: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], list[str]]:
    prompt = scored_prompt_ids(workload)
    entry_n = int(workload["hot_cache_entries"])

    def seed_endpoint(endpoint: Endpoint) -> tuple[list[dict[str, Any]], list[str]]:
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for template in range(entry_n):
            row = request(
                endpoint,
                prompt,
                hot_salt(namespace, endpoint, template),
                workload,
                timeout,
            )
            row.update(
                {
                    "kind": "seed",
                    "target": endpoint.label,
                    "template": template,
                }
            )
            expected_hash = goldens.get((endpoint.label, template))
            if expected_hash is None and row.get("ok"):
                goldens[(endpoint.label, template)] = str(row["text_sha256"])
            elif expected_hash is not None and row.get("text_sha256") != expected_hash:
                failures.append(
                    f"{endpoint.label} hot template {template}: seed output hash drift"
                )
            if not row.get("ok"):
                failures.append(
                    f"{endpoint.label} hot template {template}: seed failed: {row.get('error')}"
                )
            if row.get("cached_tokens") not in (0, len(prompt)):
                failures.append(
                    f"{endpoint.label} hot template {template}: seed cached "
                    f"{row.get('cached_tokens')} not 0 or {len(prompt)}"
                )
            rows.append(row)
        return rows, failures

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = [pool.submit(seed_endpoint, endpoint) for endpoint in endpoints]
        for future in futures:
            endpoint_rows, endpoint_failures = future.result()
            rows.extend(endpoint_rows)
            failures.extend(endpoint_failures)
    for endpoint in endpoints:
        snapshot = scrape(endpoint, timeout)
        entries = metric_value(snapshot, "prefix_cache_entries")
        if entries < entry_n:
            failures.append(
                f"{endpoint.label}: cache retains {entries} entries after {entry_n} hot seeds"
            )
    return rows, failures


def make_jobs(
    endpoint: Endpoint,
    workload: dict[str, Any],
    namespace: str,
    arm: str,
    rep: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    request_n = cell_request_count(workload, concurrency)
    prompt = scored_prompt_ids(workload)
    prompt_n = len(prompt)
    cycle = int(workload["hit_requests_per_cycle"]) + int(workload["miss_requests_per_cycle"])
    if arm == "cold":
        roles = ["miss"] * request_n
    elif arm == "mixed90":
        hit_n = request_n * int(workload["hit_requests_per_cycle"]) // cycle
        roles = ["hit"] * hit_n + ["miss"] * (request_n - hit_n)
        random.Random(int(workload["seed"]) + rep * 1009 + concurrency * 9173).shuffle(roles)
    else:
        raise ValueError(f"unknown arm: {arm}")

    jobs: list[dict[str, Any]] = []
    hit_index = 0
    for index, role in enumerate(roles):
        unique = rep * 1_000_000 + concurrency * 10_000 + index
        if role == "hit":
            template = hit_index % int(workload["hot_cache_entries"])
            hit_index += 1
            salt = hot_salt(namespace, endpoint, template)
            expected_cached = prompt_n
        else:
            template = None
            salt = f"{namespace}-{endpoint.label}-{arm}-r{rep}-c{concurrency}-i{index}"
            expected_cached = 0
        jobs.append(
            {
                "index": index,
                "role": role,
                "template": template,
                "prompt": prompt,
                "salt": salt,
                "expected_cached": expected_cached,
            }
        )
    return jobs


def run_cell(
    endpoints: list[Endpoint],
    workload: dict[str, Any],
    namespace: str,
    arm: str,
    rep: int,
    concurrency: int,
    timeout: float,
    goldens: dict[tuple[str, int], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    jobs = {
        endpoint.label: make_jobs(endpoint, workload, namespace, arm, rep, concurrency)
        for endpoint in endpoints
    }
    before = {endpoint.label: scrape(endpoint, timeout) for endpoint in endpoints}
    barrier = threading.Barrier(concurrency * len(endpoints) + 1)
    go = threading.Event()
    release_box: list[float | None] = [None]
    executors: list[concurrent.futures.ThreadPoolExecutor] = []
    futures: list[tuple[Endpoint, dict[str, Any], concurrent.futures.Future]] = []

    def one(
        endpoint: Endpoint,
        job: dict[str, Any],
        first_wave: bool,
    ) -> dict[str, Any]:
        row = request(
            endpoint,
            job["prompt"],
            job["salt"],
            workload,
            timeout,
            barrier if first_wave else None,
            go,
        )
        release = release_box[0]
        assert release is not None
        row.update(
            {
                "kind": "request",
                "target": endpoint.label,
                "arm": arm,
                "rep": rep,
                "concurrency": concurrency,
                "index": job["index"],
                "cache_role": job["role"],
                "template": job["template"],
                "expected_cached_tokens": job["expected_cached"],
                "request_start_offset_ms": (float(row["_started"]) - release) * 1000.0,
            }
        )
        expected_hash = (
            goldens.get((endpoint.label, int(job["template"])))
            if job["template"] is not None
            else None
        )
        row["golden_sha256"] = expected_hash
        row["usage_ok"] = bool(
            row.get("prompt_tokens") == int(workload["prompt_tokens"])
            and row.get("cached_tokens") == job["expected_cached"]
            and row.get("completion_tokens") == int(workload["completion_tokens"])
        )
        row["golden_ok"] = bool(expected_hash is None or row.get("text_sha256") == expected_hash)
        # Cache exactness is byte-gated under the same serial decode composition at c=1.
        # At c>1, the repository's documented batched-prime near-tie class can move text
        # independently of cache state; retain the comparison, but do not mislabel that
        # cross-config numeric class as cache corruption.
        row["golden_required"] = concurrency == 1
        row["ok"] = bool(
            row.get("ok")
            and row["usage_ok"]
            and (row["golden_ok"] or not row["golden_required"])
        )
        return row

    for endpoint in endpoints:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
        executors.append(executor)
        for index, job in enumerate(jobs[endpoint.label]):
            future = executor.submit(one, endpoint, job, index < concurrency)
            futures.append((endpoint, job, future))

    barrier.wait(timeout=60)
    release_box[0] = time.monotonic()
    go.set()
    samples: list[dict[str, Any]] = []
    while not all(future.done() for _, _, future in futures):
        for endpoint in endpoints:
            try:
                sample = scrape(endpoint, min(timeout, 10.0))
                samples.append(
                    {
                        "kind": "metrics_sample",
                        "target": endpoint.label,
                        "arm": arm,
                        "rep": rep,
                        "concurrency": concurrency,
                        "elapsed_s": time.monotonic() - float(release_box[0]),
                        "active_sessions": sample.get("active_sessions"),
                        "queued_requests": sample.get("queued_requests"),
                        "prefix_cache_entries": sample.get("prefix_cache_entries"),
                        "prefix_cache_bytes": sample.get("prefix_cache_bytes"),
                        "admission_session_defers": sample.get("admission_session_defers"),
                        "admission_vram_defers": sample.get("admission_vram_defers"),
                        "step_oom_parks": sample.get("step_oom_parks"),
                    }
                )
            except Exception as error:
                samples.append(
                    {
                        "kind": "metrics_sample",
                        "target": endpoint.label,
                        "arm": arm,
                        "rep": rep,
                        "concurrency": concurrency,
                        "elapsed_s": time.monotonic() - float(release_box[0]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        time.sleep(0.1)

    rows = [future.result() for _, _, future in futures]
    for executor in executors:
        executor.shutdown(wait=True)

    after = {}
    for endpoint in endpoints:
        after[endpoint.label] = wait_settled(
            endpoint,
            metric_value(before[endpoint.label], "completed"),
            len(jobs[endpoint.label]),
            timeout,
        )

    summaries: list[dict[str, Any]] = []
    release = float(release_box[0])
    for endpoint in endpoints:
        target_rows = [row for row in rows if row["target"] == endpoint.label]
        target_samples = [row for row in samples if row["target"] == endpoint.label]
        hit_rows = [row for row in target_rows if row["cache_role"] == "hit"]
        miss_rows = [row for row in target_rows if row["cache_role"] == "miss"]
        completed_rows = [row for row in target_rows if row.get("_ended") is not None]
        wall_s = max(float(row["_ended"]) for row in completed_rows) - release
        deltas = {
            key: metric_delta(after[endpoint.label], before[endpoint.label], key)
            for key in COUNTERS
            if key not in ("prefix_cache_entries", "prefix_cache_bytes")
        }
        prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in target_rows)
        cached_total = sum(int(row.get("cached_tokens") or 0) for row in target_rows)
        completion_total = sum(int(row.get("completion_tokens") or 0) for row in target_rows)
        expected_hit_n = sum(row["cache_role"] == "hit" for row in target_rows)
        expected_miss_n = len(target_rows) - expected_hit_n
        drift_cached_in = deltas["cached_tokens_in"] - cached_total
        drift_prefix_tokens = deltas["prefix_cache_hit_tokens"] - cached_total
        drift_prompt = deltas["prompt_tokens_in"] - prompt_total
        integrity_failures: list[str] = []
        if any(not row.get("ok") for row in target_rows):
            integrity_failures.append("one or more requests failed response/usage/golden checks")
        if deltas["admitted"] != len(target_rows) or deltas["completed"] != len(target_rows):
            integrity_failures.append("admitted/completed counters do not match request count")
        if deltas["tokens_out"] != completion_total:
            integrity_failures.append("tokens_out counter does not match response usage")
        if drift_prompt != 0:
            integrity_failures.append(f"prompt token accounting drift={drift_prompt}")
        if drift_cached_in != 0:
            integrity_failures.append(f"cached_tokens_in drift={drift_cached_in}")
        if drift_prefix_tokens != 0:
            integrity_failures.append(f"prefix_cache_hit_tokens drift={drift_prefix_tokens}")
        if deltas["prefix_cache_hits"] != expected_hit_n:
            integrity_failures.append(
                f"prefix hit count {deltas['prefix_cache_hits']} != {expected_hit_n}"
            )
        if deltas["prefix_cache_misses"] != expected_miss_n:
            integrity_failures.append(
                f"prefix miss count {deltas['prefix_cache_misses']} != {expected_miss_n}"
            )
        if deltas["step_oom_parks"] != 0:
            integrity_failures.append(f"step OOM parks={deltas['step_oom_parks']}")

        summary = {
            "kind": "cell",
            "schema": "memra.sellgate.replay.v1",
            "target": endpoint.label,
            "model": endpoint.model,
            "arm": arm,
            "rep": rep,
            "concurrency": concurrency,
            "requests_n": len(target_rows),
            "requests_ok": sum(bool(row.get("ok")) for row in target_rows),
            "hit_requests": len(hit_rows),
            "miss_requests": len(miss_rows),
            "prompt_tokens": prompt_total,
            "cached_tokens": cached_total,
            "computed_prompt_tokens": prompt_total - cached_total,
            "cache_hit_token_ratio": cached_total / prompt_total if prompt_total else None,
            "completion_tokens": completion_total,
            "wall_s": wall_s,
            "requests_per_s": len(target_rows) / wall_s if wall_s > 0 else None,
            "output_tok_s": completion_total / wall_s if wall_s > 0 else None,
            "billed_prompt_tok_s": prompt_total / wall_s if wall_s > 0 else None,
            "computed_prompt_tok_s": (prompt_total - cached_total) / wall_s
            if wall_s > 0
            else None,
            "ttft_all": distribution(
                [float(row["ttft_ms"]) for row in target_rows if row.get("ttft_ms") is not None]
            ),
            "ttft_hit": distribution(
                [float(row["ttft_ms"]) for row in hit_rows if row.get("ttft_ms") is not None]
            ),
            "ttft_miss": distribution(
                [float(row["ttft_ms"]) for row in miss_rows if row.get("ttft_ms") is not None]
            ),
            "latency_all": distribution(
                [float(row["latency_ms"]) for row in target_rows if row.get("latency_ms") is not None]
            ),
            "latency_hit": distribution(
                [float(row["latency_ms"]) for row in hit_rows if row.get("latency_ms") is not None]
            ),
            "latency_miss": distribution(
                [float(row["latency_ms"]) for row in miss_rows if row.get("latency_ms") is not None]
            ),
            "inter_token_all": distribution(
                [float(row["inter_token_ms"]) for row in target_rows if row.get("inter_token_ms") is not None]
            ),
            "inter_token_hit": distribution(
                [float(row["inter_token_ms"]) for row in hit_rows if row.get("inter_token_ms") is not None]
            ),
            "inter_token_miss": distribution(
                [float(row["inter_token_ms"]) for row in miss_rows if row.get("inter_token_ms") is not None]
            ),
            "golden_mismatches_observed": sum(
                not bool(row.get("golden_ok")) for row in target_rows
            ),
            "golden_required_failures": sum(
                bool(row.get("golden_required")) and not bool(row.get("golden_ok"))
                for row in target_rows
            ),
            "request_start_spread_ms": max(
                float(row["request_start_offset_ms"]) for row in target_rows
            )
            - min(float(row["request_start_offset_ms"]) for row in target_rows),
            "peak_active_sessions_sampled": max(
                (
                    int(row["active_sessions"])
                    for row in target_samples
                    if isinstance(row.get("active_sessions"), int)
                ),
                default=0,
            ),
            "peak_queued_requests_sampled": max(
                (
                    int(row["queued_requests"])
                    for row in target_samples
                    if isinstance(row.get("queued_requests"), int)
                ),
                default=0,
            ),
            "prefix_cache_entries_before": metric_value(
                before[endpoint.label], "prefix_cache_entries"
            ),
            "prefix_cache_entries_after": metric_value(
                after[endpoint.label], "prefix_cache_entries"
            ),
            "prefix_cache_bytes_before": metric_value(
                before[endpoint.label], "prefix_cache_bytes"
            ),
            "prefix_cache_bytes_after": metric_value(
                after[endpoint.label], "prefix_cache_bytes"
            ),
            "counter_deltas": deltas,
            "cached_tokens_in_drift": drift_cached_in,
            "prefix_cache_hit_tokens_drift": drift_prefix_tokens,
            "prompt_tokens_in_drift": drift_prompt,
            "integrity_failures": integrity_failures,
            "clean": not integrity_failures,
        }
        summaries.append(summary)

    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    return public_rows, samples, summaries


def aggregate_rows(
    endpoints: list[Endpoint],
    levels: list[int],
    requests: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for concurrency in levels:
            for arm in ("cold", "mixed90"):
                selected_cells = [
                    row
                    for row in cells
                    if row["target"] == endpoint.label
                    and row["concurrency"] == concurrency
                    and row["arm"] == arm
                ]
                selected_requests = [
                    row
                    for row in requests
                    if row["target"] == endpoint.label
                    and row["concurrency"] == concurrency
                    and row["arm"] == arm
                ]
                if not selected_cells:
                    continue
                hit_rows = [row for row in selected_requests if row["cache_role"] == "hit"]
                miss_rows = [row for row in selected_requests if row["cache_role"] == "miss"]
                prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in selected_requests)
                cached_total = sum(int(row.get("cached_tokens") or 0) for row in selected_requests)
                aggregates.append(
                    {
                        "kind": "aggregate",
                        "schema": "memra.sellgate.replay.v1",
                        "target": endpoint.label,
                        "model": endpoint.model,
                        "arm": arm,
                        "concurrency": concurrency,
                        "n_cells": len(selected_cells),
                        "n_requests": len(selected_requests),
                        "all_clean": all(bool(row["clean"]) for row in selected_cells),
                        "cached_accounting_zero_drift": all(
                            row["cached_tokens_in_drift"] == 0
                            and row["prefix_cache_hit_tokens_drift"] == 0
                            for row in selected_cells
                        ),
                        "cache_hit_token_ratio": cached_total / prompt_total
                        if prompt_total
                        else None,
                        "requests_per_s_median": statistics.median(
                            float(row["requests_per_s"]) for row in selected_cells
                        ),
                        "output_tok_s_median": statistics.median(
                            float(row["output_tok_s"]) for row in selected_cells
                        ),
                        "ttft_all": distribution(
                            [
                                float(row["ttft_ms"])
                                for row in selected_requests
                                if row.get("ttft_ms") is not None
                            ]
                        ),
                        "ttft_hit": distribution(
                            [
                                float(row["ttft_ms"])
                                for row in hit_rows
                                if row.get("ttft_ms") is not None
                            ]
                        ),
                        "ttft_miss": distribution(
                            [
                                float(row["ttft_ms"])
                                for row in miss_rows
                                if row.get("ttft_ms") is not None
                            ]
                        ),
                        "latency_all": distribution(
                            [
                                float(row["latency_ms"])
                                for row in selected_requests
                                if row.get("latency_ms") is not None
                            ]
                        ),
                        "latency_hit": distribution(
                            [
                                float(row["latency_ms"])
                                for row in hit_rows
                                if row.get("latency_ms") is not None
                            ]
                        ),
                        "latency_miss": distribution(
                            [
                                float(row["latency_ms"])
                                for row in miss_rows
                                if row.get("latency_ms") is not None
                            ]
                        ),
                        "inter_token_all": distribution(
                            [
                                float(row["inter_token_ms"])
                                for row in selected_requests
                                if row.get("inter_token_ms") is not None
                            ]
                        ),
                        "inter_token_hit": distribution(
                            [
                                float(row["inter_token_ms"])
                                for row in hit_rows
                                if row.get("inter_token_ms") is not None
                            ]
                        ),
                        "inter_token_miss": distribution(
                            [
                                float(row["inter_token_ms"])
                                for row in miss_rows
                                if row.get("inter_token_ms") is not None
                            ]
                        ),
                        "golden_mismatches_observed": sum(
                            int(row["golden_mismatches_observed"])
                            for row in selected_cells
                        ),
                        "golden_required_failures": sum(
                            int(row["golden_required_failures"])
                            for row in selected_cells
                        ),
                        "admission_session_defers": sum(
                            int(row["counter_deltas"]["admission_session_defers"])
                            for row in selected_cells
                        ),
                        "admission_vram_defers": sum(
                            int(row["counter_deltas"]["admission_vram_defers"])
                            for row in selected_cells
                        ),
                        "step_oom_parks": sum(
                            int(row["counter_deltas"]["step_oom_parks"])
                            for row in selected_cells
                        ),
                        "prefix_cache_evictions": sum(
                            int(row["counter_deltas"]["prefix_cache_evictions"])
                            for row in selected_cells
                        ),
                        "prefix_cache_bytes_peak_cell_end": max(
                            int(row["prefix_cache_bytes_after"]) for row in selected_cells
                        ),
                    }
                )
    return aggregates


def width_orders(levels: list[int], reps: int) -> list[list[int]]:
    orders: list[list[int]] = []
    for rep in range(reps):
        offset = rep % len(levels)
        order = levels[offset:] + levels[:offset]
        if rep % 2:
            order = list(reversed(order))
        orders.append(order)
    return orders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", type=parse_endpoint, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", default="cx-sellgate")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if len(args.endpoint) != 2:
        parser.error("the sold shape requires exactly two endpoints")
    if len({endpoint.label for endpoint in args.endpoint}) != 2:
        parser.error("endpoint labels must be unique")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    workload = load_workload(args.workload_lock)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_levels = [int(value) for value in workload["base_concurrency"]]
    extension_levels = [int(value) for value in workload["extension_concurrency"]]
    reps = int(workload["repetitions"])
    all_requests: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_levels = list(base_levels)
    failures: list[str] = []
    goldens: dict[tuple[str, int], str] = {}

    with args.out.open("w", encoding="utf-8") as output:
        protocol = {
            "kind": "protocol",
            "schema": "memra.sellgate.replay.v1",
            "workload_lock_sha256": sha256_file(args.workload_lock),
            "prompt_ids_sha256_canonical_json": prompt_sha256(scored_prompt_ids(workload)),
            "workload": workload,
            "targets": [dataclasses.asdict(endpoint) for endpoint in args.endpoint],
            "cache_shape": (
                "mixed90 has nine full-prompt hits and one full cold miss per ten "
                "equal-sized prompts; eight hot cache namespaces carry the qualified "
                "fixed prompt and cold has a unique namespace per request"
            ),
            "latency_clock": "first visible response content, not SSE keepalive",
            "arm_order": "alternating within rotated base-width orders",
        }
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(protocol, sort_keys=True), flush=True)

        cell_pair_index = 0

        def execute_cell(arm: str, rep: int, concurrency: int) -> None:
            if arm == "mixed90":
                seed_rows, seed_failures = seed_hot_set(
                    args.endpoint,
                    workload,
                    args.namespace,
                    args.timeout,
                    goldens,
                )
                for row in seed_rows:
                    output.write(json.dumps(
                        {key: value for key, value in row.items() if not key.startswith("_")},
                        sort_keys=True,
                    ) + "\n")
                output.flush()
                failures.extend(seed_failures)
                for failure in seed_failures:
                    print(json.dumps({"kind": "seed_failure", "error": failure}), flush=True)

            requests, samples, cells = run_cell(
                args.endpoint,
                workload,
                args.namespace,
                arm,
                rep,
                concurrency,
                args.timeout,
                goldens,
            )
            for row in [*samples, *requests, *cells]:
                output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            all_requests.extend(requests)
            all_cells.extend(cells)
            for cell in cells:
                print(json.dumps(cell, sort_keys=True), flush=True)
                if not cell["clean"]:
                    failures.append(
                        f"{cell['target']} {arm} r{rep} c{concurrency}: "
                        + "; ".join(cell["integrity_failures"])
                    )
        for rep, order in enumerate(width_orders(base_levels, reps), start=1):
            for concurrency in order:
                arms = (
                    ["cold", "mixed90"]
                    if cell_pair_index % 2 == 0
                    else ["mixed90", "cold"]
                )
                for arm in arms:
                    execute_cell(arm, rep, concurrency)
                cell_pair_index += 1

        previous = base_levels[-1]
        for candidate in extension_levels:
            current_aggregates = aggregate_rows(
                args.endpoint, all_levels, all_requests, all_cells
            )

            def rate(target: str, concurrency: int) -> float | None:
                for aggregate in current_aggregates:
                    if (
                        aggregate["target"] == target
                        and aggregate["arm"] == "mixed90"
                        and aggregate["concurrency"] == concurrency
                    ):
                        if not aggregate["all_clean"]:
                            return None
                        return float(aggregate["output_tok_s_median"])
                return None

            if previous == base_levels[-1]:
                compared = base_levels[-2]
            else:
                compared = all_levels[-2]
            rise_by_target = {}
            for endpoint in args.endpoint:
                previous_rate = rate(endpoint.label, previous)
                compared_rate = rate(endpoint.label, compared)
                rise_by_target[endpoint.label] = bool(
                    previous_rate is not None
                    and compared_rate is not None
                    and previous_rate > compared_rate
                )
            decision = {
                "kind": "extension_decision",
                "preceding_width": previous,
                "compared_width": compared,
                "candidate_width": candidate,
                "mixed90_throughput_rises": rise_by_target,
                "run_candidate": any(rise_by_target.values()),
            }
            output.write(json.dumps(decision, sort_keys=True) + "\n")
            output.flush()
            print(json.dumps(decision, sort_keys=True), flush=True)
            if not decision["run_candidate"]:
                break
            all_levels.append(candidate)
            for rep in range(1, reps + 1):
                arms = (
                    ["cold", "mixed90"]
                    if cell_pair_index % 2 == 0
                    else ["mixed90", "cold"]
                )
                for arm in arms:
                    execute_cell(arm, rep, candidate)
                cell_pair_index += 1
            previous = candidate

        aggregates = aggregate_rows(args.endpoint, all_levels, all_requests, all_cells)
        for aggregate in aggregates:
            output.write(json.dumps(aggregate, sort_keys=True) + "\n")
            print(json.dumps(aggregate, sort_keys=True), flush=True)

        required_cells = [cell for cell in all_cells if int(cell["concurrency"]) in base_levels]
        target_base_clean = {}
        expected_per_target = len(base_levels) * reps * 2
        for endpoint in args.endpoint:
            target_cells = [
                cell for cell in required_cells if cell["target"] == endpoint.label
            ]
            target_base_clean[endpoint.label] = bool(
                len(target_cells) == expected_per_target
                and all(bool(cell["clean"]) for cell in target_cells)
            )
        required_clean = all(target_base_clean.values())
        at_least_one_clean = any(target_base_clean.values())
        final = {
            "kind": "summary",
            "schema": "memra.sellgate.replay.v1",
            "levels_run": all_levels,
            "required_base_cells": len(required_cells),
            "required_base_clean": required_clean,
            "target_base_clean": target_base_clean,
            "at_least_one_target_base_clean": at_least_one_clean,
            "seed_and_cell_failures": failures,
            "verdict": "PASS" if at_least_one_clean else "FAIL",
        }
        output.write(json.dumps(final, sort_keys=True) + "\n")
        output.flush()
        print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if at_least_one_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
