#!/usr/bin/env python3
"""Interleaved cold-vs-90%-prefix-hit concurrency ladder for a PP-2 server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from prefix_gate import request


def scrape(base: str, timeout: float) -> dict:
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as response:
        return json.load(response)


def counter(row: dict, key: str) -> int:
    return int(row.get(key) or 0)


def dual_counter(row: dict, key: str) -> int:
    return int((row.get("dual_pp") or {}).get(key) or 0)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))]


def prompt_ids(prefix_tokens: int, suffix_tokens: int, index: int) -> list[int]:
    prefix = [5_000 + (position % 1_024) for position in range(prefix_tokens)]
    suffix = [7_000 + ((index * 509 + position) % 1_024) for position in range(suffix_tokens)]
    return prefix + suffix


def run_wave(
    args: argparse.Namespace,
    arm: str,
    rep: int,
    concurrency: int,
    salt: str,
) -> tuple[list[dict], dict]:
    barrier = threading.Barrier(concurrency)
    before = scrape(args.base, args.timeout)

    def one(index: int) -> dict:
        prompt = prompt_ids(args.prefix_tokens, args.suffix_tokens, index)
        request_salt = salt if arm == "hit90" else f"{salt}-cold-{index}"
        try:
            row = request(
                args.base,
                args.model,
                prompt,
                request_salt,
                args.max_tokens,
                args.timeout,
                barrier,
            )
            row["ok"] = (
                row.get("finish_reason") in ("stop", "length")
                and int(row.get("completion_tokens") or 0) > 0
                and row.get("prompt_tokens") is not None
                and row.get("cached_tokens") is not None
            )
            if not row["ok"]:
                row["error"] = "incomplete response metadata"
            row["_completed_mono"] = time.monotonic()
            return row
        except Exception as error:  # Preserve refusals and resource failures in the receipt.
            return {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "prompt_tokens": len(prompt),
                "cached_tokens": None,
                "_completed_mono": time.monotonic(),
            }

    started = time.monotonic()
    peak_active = 0
    peak_queued = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, index) for index in range(concurrency)]
        while not all(future.done() for future in futures):
            sample = scrape(args.base, args.timeout)
            peak_active = max(peak_active, counter(sample, "active_sessions"))
            peak_queued = max(peak_queued, counter(sample, "queued_requests"))
            time.sleep(0.1)
        rows = [future.result() for future in futures]
    wall_s = max(row["_completed_mono"] for row in rows) - started
    after = scrape(args.base, args.timeout)

    for index, row in enumerate(rows):
        row.pop("_completed_mono")
        row.update(
            {
                "kind": "request",
                "arm": arm,
                "rep": rep,
                "concurrency": concurrency,
                "index": index,
            }
        )

    good = [row for row in rows if row.get("ok")]
    ttfts = [float(row["ttft_ms"]) for row in good]
    prompt_total = sum(int(row.get("prompt_tokens") or 0) for row in good)
    cached_total = sum(int(row.get("cached_tokens") or 0) for row in good)
    expected_cached = args.prefix_tokens if arm == "hit90" else 0
    expected_prompt = args.prefix_tokens + args.suffix_tokens
    usage_ok = all(
        int(row["cached_tokens"]) == expected_cached
        and int(row["prompt_tokens"]) == expected_prompt
        for row in good
    )
    summary = {
        "kind": "cell",
        "arm": arm,
        "rep": rep,
        "concurrency": concurrency,
        "requests_ok": len(good),
        "requests_n": len(rows),
        "usage_ok": usage_ok and len(good) == len(rows),
        "prompt_tokens": prompt_total,
        "cached_tokens": cached_total,
        "cache_hit_token_ratio": cached_total / prompt_total if prompt_total else None,
        "wall_s": round(wall_s, 6),
        "requests_per_s": round(len(good) / wall_s, 6) if wall_s else None,
        "output_tok_s": round(
            sum(int(row.get("completion_tokens") or 0) for row in good) / wall_s, 6
        ) if wall_s else None,
        "ttft_p50_ms": round(statistics.median(ttfts), 6) if ttfts else None,
        "ttft_p95_ms": round(percentile(ttfts, 0.95), 6) if ttfts else None,
        "peak_active_sessions_sampled": peak_active,
        "peak_queued_sampled": peak_queued,
        "admission_session_defers": counter(after, "admission_session_defers")
        - counter(before, "admission_session_defers"),
        "admission_vram_defers": counter(after, "admission_vram_defers")
        - counter(before, "admission_vram_defers"),
        "step_oom_parks": counter(after, "step_oom_parks") - counter(before, "step_oom_parks"),
        "dual_pp_slot_pairs": dual_counter(after, "slot_pairs")
        - dual_counter(before, "slot_pairs"),
        "dual_pp_slot_collisions": dual_counter(after, "slot_collisions")
        - dual_counter(before, "slot_collisions"),
        "prefix_cache_evictions": counter(after, "prefix_cache_evictions")
        - counter(before, "prefix_cache_evictions"),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--namespace", default="prefixmoney-capacity")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4,8,16,24,32")
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--suffix-tokens", type=int, default=455)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    levels = [int(value) for value in args.concurrency.split(",")]
    if args.reps < 1 or not levels or min(levels) < 1:
        parser.error("reps and concurrency levels must be positive")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    before_all = scrape(args.base, args.timeout)
    cells: list[dict] = []
    failures: list[str] = []
    with args.out.open("w", encoding="utf-8") as output:
        protocol = {
            "kind": "protocol",
            "schema": "memra.prefixmoney.capacity.v1",
            "model": args.model,
            "reps": args.reps,
            "concurrency": levels,
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "requested_hit_ratio": args.prefix_tokens
            / (args.prefix_tokens + args.suffix_tokens),
            "max_tokens": args.max_tokens,
            "arm_order": "interleaved by rep and reversed on alternating cells",
            "thermal_regime": "one server boot and one GPU sampler; no clock changes",
        }
        output.write(json.dumps(protocol, sort_keys=True) + "\n")
        print(json.dumps(protocol, sort_keys=True), flush=True)

        cell_index = 0
        for rep in range(1, args.reps + 1):
            for concurrency in levels:
                arms = ["cold", "hit90"] if cell_index % 2 == 0 else ["hit90", "cold"]
                cell_index += 1
                for arm in arms:
                    salt = f"{args.namespace}-r{rep}-c{concurrency}-{arm}"
                    if arm == "hit90":
                        try:
                            seed = request(
                                args.base,
                                args.model,
                                prompt_ids(args.prefix_tokens, 0, 0),
                                salt,
                                args.max_tokens,
                                args.timeout,
                            )
                            seed_row = {
                                "kind": "seed",
                                "arm": arm,
                                "rep": rep,
                                "concurrency": concurrency,
                                "ok": True,
                                **seed,
                            }
                        except Exception as error:
                            seed = None
                            seed_row = {
                                "kind": "seed",
                                "arm": arm,
                                "rep": rep,
                                "concurrency": concurrency,
                                "ok": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        output.write(json.dumps(seed_row, sort_keys=True) + "\n")
                        print(json.dumps({k: v for k, v in seed_row.items()
                                          if k != "text_utf8_b64"}, sort_keys=True), flush=True)
                        if seed is None:
                            failures.append(f"r{rep} c{concurrency}: hit seed failed")
                        elif (
                            seed["cached_tokens"] != 0
                            or seed["prompt_tokens"] != args.prefix_tokens
                            or int(seed.get("completion_tokens") or 0) <= 0
                        ):
                            failures.append(
                                f"r{rep} c{concurrency}: hit seed usage mismatch: "
                                f"prompt={seed['prompt_tokens']} cached={seed['cached_tokens']} "
                                f"completion={seed['completion_tokens']}"
                            )
                    rows, cell = run_wave(args, arm, rep, concurrency, salt)
                    for row in rows:
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                    output.write(json.dumps(cell, sort_keys=True) + "\n")
                    output.flush()
                    cells.append(cell)
                    print(json.dumps(cell, sort_keys=True), flush=True)
                    if cell["requests_ok"] != cell["requests_n"]:
                        failures.append(
                            f"r{rep} c{concurrency} {arm}: "
                            f"{cell['requests_ok']}/{cell['requests_n']} requests clean"
                        )
                    if not cell["usage_ok"]:
                        failures.append(f"r{rep} c{concurrency} {arm}: cached-token usage mismatch")
                    if cell["step_oom_parks"] or cell["dual_pp_slot_collisions"]:
                        failures.append(
                            f"r{rep} c{concurrency} {arm}: oom_parks={cell['step_oom_parks']} "
                            f"slot_collisions={cell['dual_pp_slot_collisions']}"
                        )

        def aggregate(arm: str, concurrency: int) -> dict:
            selected = [
                cell for cell in cells
                if cell["arm"] == arm and cell["concurrency"] == concurrency
            ]
            ttfts = [float(cell["ttft_p95_ms"]) for cell in selected
                     if cell["ttft_p95_ms"] is not None]
            request_rates = [float(cell["requests_per_s"]) for cell in selected
                             if cell["requests_per_s"] is not None]
            output_rates = [float(cell["output_tok_s"]) for cell in selected
                            if cell["output_tok_s"] is not None]
            return {
                "arm": arm,
                "concurrency": concurrency,
                "n_cells": len(selected),
                "all_clean": all(
                    cell["requests_ok"] == cell["requests_n"] and cell["usage_ok"]
                    and cell["step_oom_parks"] == 0
                    and cell["dual_pp_slot_collisions"] == 0
                    and cell["admission_session_defers"] == 0
                    and cell["admission_vram_defers"] == 0
                    for cell in selected
                ),
                "ttft_p95_ms_median": statistics.median(ttfts) if ttfts else None,
                "requests_per_s_median": statistics.median(request_rates)
                if request_rates else None,
                "output_tok_s_median": statistics.median(output_rates)
                if output_rates else None,
            }

        aggregates = [aggregate(arm, concurrency) for arm in ("cold", "hit90")
                      for concurrency in levels]
        cold_one = next(row for row in aggregates
                        if row["arm"] == "cold" and row["concurrency"] == 1)
        parity_limit = cold_one["ttft_p95_ms_median"]
        if parity_limit is None:
            failures.append("cold c=1 produced no TTFT parity baseline")

        def parity_capacity(arm: str) -> int:
            eligible = [
                row["concurrency"] for row in aggregates
                if row["arm"] == arm and row["all_clean"]
                and parity_limit is not None
                and row["ttft_p95_ms_median"] is not None
                and row["ttft_p95_ms_median"] <= parity_limit
            ]
            return max(eligible, default=0)

        after_all = scrape(args.base, args.timeout)
        cold_capacity = parity_capacity("cold")
        hit_capacity = parity_capacity("hit90")
        pair_delta = dual_counter(after_all, "slot_pairs") - dual_counter(before_all, "slot_pairs")
        collision_delta = (
            dual_counter(after_all, "slot_collisions")
            - dual_counter(before_all, "slot_collisions")
        )
        if pair_delta <= 0:
            failures.append("concurrency ladder produced no dual PP slot pairs")
        if collision_delta != 0:
            failures.append(f"dual PP slot collisions increased by {collision_delta}")
        summary = {
            "kind": "summary",
            "schema": "memra.prefixmoney.capacity.v1",
            "aggregates": aggregates,
            "latency_parity_definition": (
                "largest clean tested concurrency whose median-of-cell-p95 TTFT is no worse "
                "than the cold c=1 median-of-cell-p95 TTFT"
            ),
            "latency_parity_limit_ms": parity_limit,
            "cold_latency_parity_concurrency": cold_capacity,
            "hit90_latency_parity_concurrency": hit_capacity,
            "hit90_concurrency_multiplier": (
                hit_capacity / cold_capacity if cold_capacity else None
            ),
            "dual_pp_slot_pairs": pair_delta,
            "dual_pp_slot_collisions": collision_delta,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        }
        output.write(json.dumps(summary, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
