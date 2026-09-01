#!/usr/bin/env python3
"""Serial exactness oracle after filling the 96-entry cache working set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO

import capacity_sweep as capacity


def emit(output: TextIO, row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True)
    output.write(line + "\n")
    output.flush()
    print(line, flush=True)


def public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-replay", type=Path, required=True)
    parser.add_argument("--workload-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--budget-mb", type=int, default=8192)
    parser.add_argument("--target-prefix-id", type=int, default=87)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.budget_mb not in [int(value) for value in protocol["prefix_cache_mb"]]:
        parser.error("budget is outside the locked ladder")
    frozen = capacity.load_module(
        args.frozen_replay, protocol["frozen_replay_sha256"]
    )
    if capacity.sha256_file(args.workload_lock) != protocol["frozen_workload_sha256"]:
        parser.error("frozen workload hash mismatch")
    workload = frozen.load_workload(args.workload_lock)
    endpoint = frozen.parse_endpoint(f"q27,{args.endpoint},q27")
    prompt = frozen.scored_prompt_ids(workload)
    working_set_n = int(protocol["working_set_entries"])
    if args.target_prefix_id not in range(working_set_n):
        parser.error("target prefix id is outside the working set")

    failures: list[str] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        emit(output, {
            "kind": "protocol",
            "schema": "memra.cachesize.restore-oracle.v1",
            "model": "q27",
            "budget_mb": args.budget_mb,
            "working_set_entries": working_set_n,
            "target_prefix_id": args.target_prefix_id,
            "repetitions": args.repetitions,
            "prompt_tokens": len(prompt),
            "completion_tokens": int(workload["completion_tokens"]),
            "seed_method": "serial; target emits the full baseline, all other seeds emit one token",
            "comparison": "same prime configuration, same exact cache entry, serial cold versus serial restored hit",
        })
        before = frozen.scrape(endpoint, args.timeout)
        baseline: dict[str, Any] | None = None
        for prefix_id in range(working_set_n):
            seed_workload = dict(workload)
            expected_completion = 1
            if prefix_id == args.target_prefix_id:
                expected_completion = int(workload["completion_tokens"])
            else:
                seed_workload["completion_tokens"] = 1
            row = frozen.request(
                endpoint,
                prompt,
                f"{args.namespace}-hot-{prefix_id}",
                seed_workload,
                args.timeout,
            )
            row.update({
                "kind": "seed",
                "prefix_id": prefix_id,
                "target": prefix_id == args.target_prefix_id,
            })
            emit(output, public(row))
            if (
                not row.get("ok")
                or int(row.get("prompt_tokens") or 0) != len(prompt)
                or int(row.get("cached_tokens") or 0) != 0
                or int(row.get("completion_tokens") or 0) != expected_completion
            ):
                failures.append(f"seed {prefix_id} response/usage drift")
            if prefix_id == args.target_prefix_id:
                baseline = row

        after_seed = frozen.wait_settled(
            endpoint, int(before.get("completed") or 0), working_set_n, args.timeout
        )
        emit(output, capacity.gpu_snapshot("after_seed"))
        sold_bytes = 301_215_744
        expected_retained = min(
            working_set_n,
            args.budget_mb * 1024 * 1024 // sold_bytes,
        )
        if int(after_seed.get("prefix_cache_entries") or 0) != expected_retained:
            failures.append("retained-entry accounting drift")
        if int(after_seed.get("prefix_cache_bytes") or 0) != expected_retained * sold_bytes:
            failures.append("retained-byte accounting drift")
        if baseline is None:
            failures.append("target baseline is absent")

        hits: list[dict[str, Any]] = []
        if not failures:
            for repetition in range(1, args.repetitions + 1):
                row = frozen.request(
                    endpoint,
                    prompt,
                    f"{args.namespace}-hot-{args.target_prefix_id}",
                    workload,
                    args.timeout,
                )
                row.update({
                    "kind": "serial_hit",
                    "prefix_id": args.target_prefix_id,
                    "repetition": repetition,
                })
                emit(output, public(row))
                hits.append(row)
                if (
                    not row.get("ok")
                    or int(row.get("prompt_tokens") or 0) != len(prompt)
                    or int(row.get("cached_tokens") or 0) != len(prompt)
                    or row.get("text_sha256") != baseline.get("text_sha256")
                ):
                    failures.append(f"serial hit {repetition} changed response/usage")

        after = frozen.wait_settled(
            endpoint,
            int(before.get("completed") or 0),
            working_set_n + len(hits),
            args.timeout,
        )
        emit(output, capacity.gpu_snapshot("after_hits"))
        counter_deltas = capacity.deltas(after, before)
        expected_counters = {
            "admitted": working_set_n + args.repetitions,
            "completed": working_set_n + args.repetitions,
            "tokens_out": working_set_n - 1
                + int(workload["completion_tokens"]) * (1 + args.repetitions),
            "prompt_tokens_in": len(prompt) * (working_set_n + args.repetitions),
            "cached_tokens_in": len(prompt) * args.repetitions,
            "prefix_cache_hits": args.repetitions,
            "prefix_cache_misses": working_set_n,
            "prefix_cache_inserts": working_set_n,
            "prefix_cache_hit_tokens": len(prompt) * args.repetitions,
            "admission_session_defers": 0,
            "admission_vram_defers": 0,
            "step_oom_parks": 0,
        }
        for key, expected in expected_counters.items():
            if counter_deltas[key] != expected:
                failures.append(
                    f"{key} delta={counter_deltas[key]} != {expected}"
                )
        emit(output, {
            "kind": "summary",
            "schema": "memra.cachesize.restore-oracle.v1",
            "model": "q27",
            "budget_mb": args.budget_mb,
            "working_set_entries": working_set_n,
            "target_prefix_id": args.target_prefix_id,
            "repetitions": args.repetitions,
            "baseline_text_sha256": baseline.get("text_sha256") if baseline else None,
            "hit_text_sha256": [row.get("text_sha256") for row in hits],
            "retained_entries_after_seed": after_seed.get("prefix_cache_entries"),
            "retained_bytes_after_seed": after_seed.get("prefix_cache_bytes"),
            "counter_deltas": counter_deltas,
            "failures": failures,
            "verdict": "PASS" if not failures else "FAIL",
        })
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
