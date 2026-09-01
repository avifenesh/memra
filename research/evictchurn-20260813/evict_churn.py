#!/usr/bin/env python3
"""Serial multi-tenant prefix-cache contention workload.

This extends prefixmoney's request/receipt shape: direct prompt token ids, per-request
``cache_salt``, greedy streaming completions, and worker-truth ``/metrics`` deltas.  Requests
are serial on purpose.  The lane measures eviction decisions, not scheduler throughput, and a
serial trace makes every hit, insert, eviction, and refusal attributable to one request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path


PREFIXMONEY = Path(__file__).resolve().parents[1] / "prefixmoney-20260812"
sys.path.insert(0, str(PREFIXMONEY))
from prefix_gate import cached_tokens as prefixmoney_cached_tokens  # noqa: E402


COUNTERS = (
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_inserts",
    "prefix_cache_evictions",
    "prefix_cache_hit_tokens",
)
REFUSAL = re.compile(r"\[prefix-cache\] skip (?:pinned )?.*? insert:")


def request(
    base: str,
    model: str,
    prompt: list[int],
    salt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """Prefixmoney request shape, retaining valid EOS-only completions.

    Prefixmoney's exactness client requires visible UTF-8 because its prompts are chosen to emit
    text. A contention scan deliberately covers many synthetic prefixes; a valid first sample can
    be EOS and therefore have no visible bytes. For those rows TTFT is explicitly the terminal EOS
    event time. Cache accounting and empty-byte identity remain observable in the final usage event.
    """
    body = {
        "model": model,
        "prompt_ids": prompt,
        "max_ctx": len(prompt) + max_tokens + 8,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 3407,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": salt,
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_event = None
    first_visible = None
    pieces: list[str] = []
    usage: dict = {}
    request_id = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(json.dumps(event["error"], sort_keys=True))
            request_id = event.get("id") or request_id
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            if choices and first_event is None:
                first_event = time.monotonic()
            for choice in choices:
                delta = choice.get("delta") or {}
                piece = choice.get("text") or ""
                piece += delta.get("content") or ""
                piece += delta.get("reasoning") or ""
                piece += delta.get("reasoning_content") or ""
                if piece:
                    if first_visible is None:
                        first_visible = time.monotonic()
                    pieces.append(piece)
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.monotonic()
    if first_event is None:
        raise RuntimeError("stream completed without a token or terminal choice event")
    encoded = "".join(pieces).encode()
    ttft_at = first_visible if first_visible is not None else first_event
    return {
        "request_id": request_id,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": prefixmoney_cached_tokens(usage),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "ttft_ms": round((ttft_at - started) * 1000.0, 6),
        "ttft_basis": "first_visible_text" if first_visible is not None else "terminal_eos_event",
        "wall_ms": round((ended - started) * 1000.0, 6),
        "text_bytes": len(encoded),
        "text_sha256": hashlib.sha256(encoded).hexdigest(),
        "text_utf8_b64": base64.b64encode(encoded).decode(),
    }


def scrape(base: str, timeout: float) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as response:
        return json.load(response)


def metric_snapshot(row: dict) -> dict[str, int]:
    return {
        **{key: int(row.get(key) or 0) for key in COUNTERS},
        "prefix_cache_entries": int(row.get("prefix_cache_entries") or 0),
        "prefix_cache_bytes": int(row.get("prefix_cache_bytes") or 0),
        "completed": int(row.get("completed") or 0),
    }


def metric_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after[key]) - int(before[key]) for key in COUNTERS}


def after_retire(base: str, timeout: float, completed_before: int) -> dict[str, int]:
    """Wait until the worker has retired the streamed request and published its counters."""
    deadline = time.monotonic() + min(timeout, 10.0)
    latest = metric_snapshot(scrape(base, timeout))
    while latest["completed"] <= completed_before and time.monotonic() < deadline:
        time.sleep(0.01)
        latest = metric_snapshot(scrape(base, timeout))
    return latest


class RefusalLog:
    """Count existing ``skip ... insert`` instrumentation appended during the workload."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = path.stat().st_size
        self.total = 0

    def read_new(self) -> int:
        with self.path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(self.offset)
            text = stream.read()
            self.offset = stream.tell()
        found = len(REFUSAL.findall(text))
        self.total += found
        return found


def prompt_ids(prefix_tokens: int, suffix_tokens: int, prefix_id: int) -> list[int]:
    """Prefixmoney's synthetic token-id shape, with a distinct prefix per working-set key."""
    prefix = [5_000 + ((prefix_id * 257 + position * 17) % 2_048)
              for position in range(prefix_tokens)]
    suffix = [8_000 + ((prefix_id * 131 + position) % 1_024)
              for position in range(suffix_tokens)]
    return prefix + suffix


def hotset_schedule(
    working_set: int,
    requests: int,
    hot_fraction: float,
    hot_request_fraction: float,
    zipf_alpha: float,
    seed: int,
) -> tuple[list[int], set[int]]:
    hot_n = max(1, math.ceil(working_set * hot_fraction))
    if hot_n >= working_set:
        raise ValueError("hot subset must leave at least one cold prefix")
    hot = list(range(hot_n))
    cold = list(range(hot_n, working_set))
    hot_requests = round(requests * hot_request_fraction)
    cold_requests = requests - hot_requests
    if hot_requests < len(hot) or cold_requests < len(cold):
        raise ValueError(
            "hotset needs enough requests to exercise every hot and cold working-set key"
        )
    labels = [True] * hot_requests + [False] * (requests - hot_requests)
    rng = random.Random(seed)
    rng.shuffle(labels)
    weights = [1.0 / ((rank + 1) ** zipf_alpha) for rank in range(hot_n)]
    # Guarantee that W really is the exercised working set, then distribute the remaining hot
    # requests with Zipf weights. Cold traffic is an explicit scan over the cold subset.
    hot_values = hot + rng.choices(hot, weights=weights, k=hot_requests - len(hot))
    rng.shuffle(hot_values)
    hot_draws = iter(hot_values)
    cold_draws = iter(cold[index % len(cold)] for index in range(cold_requests))
    return [next(hot_draws) if is_hot else next(cold_draws) for is_hot in labels], set(hot)


def schedule(args: argparse.Namespace) -> tuple[list[int], set[int]]:
    if args.pattern == "round-robin":
        return [index % args.working_set for index in range(args.requests)], set()
    if args.pattern == "sequential-scan":
        if args.requests > args.working_set:
            raise ValueError("sequential-scan requires requests <= working-set (every key is new)")
        return list(range(args.requests)), set()
    return hotset_schedule(
        args.working_set,
        args.requests,
        args.hot_fraction,
        args.hot_request_fraction,
        args.zipf_alpha,
        args.rng_seed,
    )


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def ttft_summary(rows: list[dict], hit: bool) -> dict:
    values = [float(row["ttft_ms"]) for row in rows
              if row.get("ok") and row.get("cache_hit") is hit]
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 6) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--namespace", default="evictchurn")
    parser.add_argument(
        "--pattern", required=True, choices=("round-robin", "hotset", "sequential-scan")
    )
    parser.add_argument("--working-set", type=int, default=40)
    parser.add_argument("--tenants", type=int, default=4)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--hot-fraction", type=float, default=0.2)
    parser.add_argument("--hot-request-fraction", type=float, default=0.8)
    parser.add_argument("--zipf-alpha", type=float, default=1.0)
    parser.add_argument("--rng-seed", type=int, default=3407)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--thermal-regime",
        default="local RTX 5090 Laptop; global 210-1200 MHz cap; no clock changes",
    )
    args = parser.parse_args()
    if args.working_set < 2 or args.tenants < 1 or args.tenants > args.working_set:
        parser.error("require working-set>=2 and 1<=tenants<=working-set")
    if args.requests < 1 or args.prefix_tokens < 64 or args.suffix_tokens < 0:
        parser.error("require requests>=1, prefix-tokens>=64, and suffix-tokens>=0")
    if not (0 < args.hot_fraction < 1 and 0 < args.hot_request_fraction < 1):
        parser.error("hot fractions must be strictly between zero and one")
    if args.zipf_alpha <= 0:
        parser.error("zipf-alpha must be positive")
    if not args.server_log.is_file():
        parser.error(f"server log does not exist: {args.server_log}")

    try:
        request_schedule, hot_ids = schedule(args)
    except ValueError as error:
        parser.error(str(error))
    distinct_prefixes = {
        tuple(prompt_ids(args.prefix_tokens, args.suffix_tokens, prefix_id)[:64])
        for prefix_id in range(args.working_set)
    }
    if len(distinct_prefixes) != args.working_set:
        parser.error("synthetic prefix generator collided inside the first 64 tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    before_all = metric_snapshot(scrape(args.base, args.timeout))
    refusal_log = RefusalLog(args.server_log)
    rows: list[dict] = []
    failures: list[str] = []
    seen: set[tuple[int, int]] = set()
    cacheable_once: set[tuple[int, int]] = set()
    golden_sha: dict[tuple[int, int], str] = {}
    byte_identity_checks = 0
    byte_identity_matches = 0

    protocol = {
        "kind": "protocol",
        "schema": "memra.evictchurn.v1",
        "model": args.model,
        "pattern": args.pattern,
        "working_set": args.working_set,
        "tenants": args.tenants,
        "requests": args.requests,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "max_tokens": args.max_tokens,
        "hot_prefix_fraction": args.hot_fraction if args.pattern == "hotset" else None,
        "hot_request_fraction": args.hot_request_fraction if args.pattern == "hotset" else None,
        "zipf_alpha": args.zipf_alpha if args.pattern == "hotset" else None,
        "rng_seed": args.rng_seed,
        "thermal_regime": args.thermal_regime,
        "request_order": "serial (one request -> retire metrics -> next request)",
        "thrash_definition": (
            "a repeated exact (tenant cache_salt, prompt_ids) request misses after that key "
            "previously produced a cache insert or hit; it would have hit if its entry remained"
        ),
        "insert_refusal_source": (
            "count of existing server lines matching '[prefix-cache] skip ... insert:'"
        ),
    }

    with args.out.open("w", encoding="utf-8") as output:
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        print(json.dumps(protocol, sort_keys=True), flush=True)
        for index, prefix_id in enumerate(request_schedule):
            tenant = prefix_id % args.tenants
            key = (tenant, prefix_id)
            prompt = prompt_ids(args.prefix_tokens, args.suffix_tokens, prefix_id)
            cache_salt = f"{args.namespace}-t{tenant}"
            before = metric_snapshot(scrape(args.base, args.timeout))
            seen_before = key in seen
            was_cacheable = key in cacheable_once
            try:
                result = request(
                    args.base,
                    args.model,
                    prompt,
                    cache_salt,
                    args.max_tokens,
                    args.timeout,
                )
                ok = (
                    result.get("finish_reason") in ("stop", "length")
                    and int(result.get("completion_tokens") or 0) > 0
                    and int(result.get("prompt_tokens") or -1) == len(prompt)
                )
                if not ok:
                    result["error"] = "incomplete response metadata"
            except Exception as error:  # Preserve exact refusal/failure text in the raw row.
                result = {
                    "error": f"{type(error).__name__}: {error}",
                    "prompt_tokens": len(prompt),
                    "cached_tokens": None,
                }
                ok = False
            after = after_retire(args.base, args.timeout, before["completed"])
            deltas = metric_delta(after, before)
            refusal_delta = refusal_log.read_new()
            cached = result.get("cached_tokens")
            cache_hit = ok and int(cached or 0) == len(prompt)
            cache_miss = ok and int(cached or 0) == 0
            evicted_before_reuse = seen_before and was_cacheable and cache_miss

            if ok and not (cache_hit or cache_miss):
                failures.append(
                    f"request {index}: partial/unexpected cached_tokens={cached} for exact key"
                )
            if ok and deltas["prefix_cache_hits"] + deltas["prefix_cache_misses"] != 1:
                failures.append(
                    f"request {index}: prefix probe delta is "
                    f"{deltas['prefix_cache_hits']} hits + {deltas['prefix_cache_misses']} misses"
                )
            if not ok:
                failures.append(f"request {index}: {result.get('error')}")

            if ok:
                sha = str(result.get("text_sha256"))
                if key not in golden_sha:
                    golden_sha[key] = sha
                elif cache_hit:
                    byte_identity_checks += 1
                    if sha == golden_sha[key]:
                        byte_identity_matches += 1
                    else:
                        failures.append(f"request {index}: cache hit changed output bytes")

            row = {
                "kind": "request",
                "pattern": args.pattern,
                "index": index,
                "prefix_id": prefix_id,
                "tenant": tenant,
                "cache_salt": cache_salt,
                "is_hot": prefix_id in hot_ids,
                "seen_before": seen_before,
                "cacheable_before": was_cacheable,
                "cache_hit": cache_hit,
                "cache_miss": cache_miss,
                "evicted_before_reuse": evicted_before_reuse,
                "metric_delta": deltas,
                "insert_refusals": refusal_delta,
                "resident_entries_after": after["prefix_cache_entries"],
                "resident_bytes_after": after["prefix_cache_bytes"],
                "ok": ok,
                **result,
            }
            rows.append(row)
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
            printable = {key_: value for key_, value in row.items() if key_ != "text_utf8_b64"}
            print(json.dumps(printable, sort_keys=True), flush=True)

            seen.add(key)
            if cache_hit or deltas["prefix_cache_inserts"] > 0:
                cacheable_once.add(key)

        # Capture any final line emitted between the retire scrape and the file read above.
        refusal_log.read_new()
        after_all = metric_snapshot(scrape(args.base, args.timeout))
        totals = metric_delta(after_all, before_all)
        good = [row for row in rows if row["ok"]]
        hits = sum(bool(row["cache_hit"]) for row in good)
        misses = sum(bool(row["cache_miss"]) for row in good)
        reuse = [row for row in good if row["seen_before"]]
        reuse_hits = sum(bool(row["cache_hit"]) for row in reuse)
        hot = [row for row in good if row["is_hot"]]
        hot_hits = sum(bool(row["cache_hit"]) for row in hot)
        hot_reuse = [row for row in hot if row["seen_before"]]
        hot_reuse_hits = sum(bool(row["cache_hit"]) for row in hot_reuse)
        thrash = sum(bool(row["evicted_before_reuse"]) for row in good)
        summary = {
            "kind": "summary",
            "schema": "memra.evictchurn.v1",
            "pattern": args.pattern,
            "requests_ok": len(good),
            "requests_n": len(rows),
            "distinct_prefixes_requested": len({row["prefix_id"] for row in rows}),
            "hits": hits,
            "misses": misses,
            "hit_rate": rate(hits, len(good)),
            "reuse_opportunities": len(reuse),
            "reuse_hits": reuse_hits,
            "reuse_hit_rate": rate(reuse_hits, len(reuse)),
            "hot_subset": {
                "prefixes": len(hot_ids),
                "requests": len(hot),
                "hits": hot_hits,
                "hit_rate": rate(hot_hits, len(hot)),
                "reuse_opportunities": len(hot_reuse),
                "reuse_hits": hot_reuse_hits,
                "reuse_hit_rate": rate(hot_reuse_hits, len(hot_reuse)),
            } if args.pattern == "hotset" else None,
            "prefix_cache_inserts": totals["prefix_cache_inserts"],
            "prefix_cache_evictions": totals["prefix_cache_evictions"],
            "insert_refusals": refusal_log.total,
            "evictions_per_100_requests": round(100 * totals["prefix_cache_evictions"] / len(rows), 6),
            "evicted_before_reuse": thrash,
            "thrash_per_100_requests": round(100 * thrash / len(rows), 6),
            "thrash_share_of_reuse_opportunities": rate(thrash, len(reuse)),
            "ttft": {
                "hit": ttft_summary(good, True),
                "miss": ttft_summary(good, False),
            },
            "cache_hit_byte_identity": {
                "matches": byte_identity_matches,
                "checks": byte_identity_checks,
            },
            "resident_entries_final": after_all["prefix_cache_entries"],
            "resident_bytes_final": after_all["prefix_cache_bytes"],
            "metrics_before": before_all,
            "metrics_after": after_all,
            "metrics_delta": totals,
            "thermal_regime": args.thermal_regime,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }
        output.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
