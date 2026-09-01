#!/usr/bin/env python3
"""Reduce and validate the P0 isolation receipts into committed evidence tables."""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
GOLDEN = "21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de"
TRANSITION = "7a5032f2d723e3cf9ef788fdc9d4067fe2eb909157189b666430b7997a56961f"
SOLO = "d35be2307889b24ec1ba4361eb22fdc6ceabda65864df261bd66c08f37f192c1"
HASH_CLASS = {GOLDEN: "golden", TRANSITION: "solo-to-batch", SOLO: "all-solo"}
CONDITIONS = {
    "same": (20, 8),
    "stagger": (20, 8),
    "dedup-off": (20, 8),
    "h2-c2": (10, 2),
    "h2-first-late": (10, 8),
    "h2-c1": (10, 1),
}
READY_RE = re.compile(r"^\[tick\].*\bready=(\d+)\b", re.MULTILINE)
ADMIT_RE = re.compile(r"^\[meter\] admit id=([^ ]+)", re.MULTILINE)
FAIL_RE = re.compile(
    r"CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|prefix fanout .*FAILED",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def classify_history(positive_ready: list[int]) -> str:
    assert positive_ready, "cell has no positive-ready decode tick"
    if max(positive_ready) == 1:
        return "all-solo"
    if positive_ready[0] == 1:
        return "solo-to-batch"
    return "batched-from-first"


def main() -> None:
    completion_rows: list[dict] = []
    cell_rows: list[dict] = []
    matrix: list[dict] = []

    for condition, (expected_cells, expected_requests) in CONDITIONS.items():
        condition_dir = RAW / condition
        cells = sorted(condition_dir.glob("cell-*"))
        assert len(cells) == expected_cells, (condition, len(cells), expected_cells)

        condition_hashes: collections.Counter[str] = collections.Counter()
        history_counts: collections.Counter[str] = collections.Counter()
        first_ready_counts: collections.Counter[int] = collections.Counter()
        divergent_cells = 0
        transition_cells = 0
        golden_only_cells = 0
        total_errors = 0
        total_prefix_hits = 0
        total_prefix_misses = 0
        transition_ttft_leads_ms: list[float] = []

        for cell_dir in cells:
            cell = cell_dir.name
            rows = load_jsonl(cell_dir / "qos-rows.jsonl")
            summary = load_json(cell_dir / "qos-summary.json")
            metrics = load_json(cell_dir / "metrics-after.json")
            server_log = (cell_dir / "server.log").read_text(encoding="utf-8")
            env = (cell_dir / "server-env.txt").read_text(encoding="utf-8")
            assert len(rows) == expected_requests, (condition, cell, len(rows))
            assert summary["requests"] == expected_requests
            assert summary["n_ok"] == expected_requests
            assert summary["n_error"] == 0
            assert not FAIL_RE.search(server_log), (condition, cell, "server failure")
            assert ("MEMRA_PREFIX_DEDUP=0" in env) == (condition == "dedup-off")

            ready = [int(value) for value in READY_RE.findall(server_log)]
            positive_ready = [value for value in ready if value > 0]
            history = classify_history(positive_ready)
            history_counts[history] += 1
            first_ready_counts[positive_ready[0]] += 1

            admitted = ADMIT_RE.findall(server_log)
            assert len(admitted) == expected_requests, (condition, cell, len(admitted))
            assert len(set(admitted)) == expected_requests
            admission_rank = {rid: rank for rank, rid in enumerate(admitted)}

            cell_hashes: collections.Counter[str] = collections.Counter()
            non_golden_ranks: list[int] = []
            for row in rows:
                assert row["ok"] is True
                payload = base64.b64decode(row["text_utf8_b64"], validate=True)
                digest = hashlib.sha256(payload).hexdigest()
                assert digest == row["text_sha256"]
                assert len(payload) == row["text_bytes"]
                assert digest in HASH_CLASS, (condition, cell, digest)
                assert row["golden_match"] == (digest == GOLDEN)
                rank = admission_rank[row["rid"]]
                if digest != GOLDEN:
                    non_golden_ranks.append(rank)
                cell_hashes[digest] += 1
                condition_hashes[digest] += 1
                completion_rows.append(
                    {
                        "admission_rank": rank,
                        "cell": cell,
                        "class": HASH_CLASS[digest],
                        "condition": condition,
                        "first_positive_ready": positive_ready[0],
                        "index": row["index"],
                        "max_ready": max(positive_ready),
                        "request_start_offset_ms": row["request_start_offset_ms"],
                        "rid": row["rid"],
                        "scheduled_delay_ms": row["scheduled_delay_ms"],
                        "text_bytes": row["text_bytes"],
                        "text_sha256": digest,
                        "ttft_s": row["ttft_s"],
                    }
                )

            assert dict(cell_hashes) == summary["hash_counts"]
            is_divergent = cell_hashes[GOLDEN] != expected_requests
            divergent_cells += int(is_divergent)
            transition_cells += int(cell_hashes[TRANSITION] > 0)
            golden_only_cells += int(cell_hashes[GOLDEN] == expected_requests)
            total_errors += summary["n_error"]
            total_prefix_hits += metrics["prefix_cache_hits"]
            total_prefix_misses += metrics["prefix_cache_misses"]
            assert metrics["prefix_cache_hits"] == 0
            assert metrics["prefix_cache_misses"] == expected_requests

            transition_rows = [row for row in rows if row["text_sha256"] == TRANSITION]
            if transition_rows:
                peer_ttft = [
                    row["first_token_offset_ms"]
                    for row in rows
                    if row["text_sha256"] != TRANSITION
                ]
                assert len(transition_rows) == 1 and peer_ttft
                transition_ttft_leads_ms.append(
                    transition_rows[0]["first_token_offset_ms"] - statistics.median(peer_ttft)
                )

            if history == "solo-to-batch":
                assert cell_hashes == collections.Counter({GOLDEN: expected_requests - 1, TRANSITION: 1})
                assert non_golden_ranks == [0]
            elif history == "batched-from-first":
                assert cell_hashes == collections.Counter({GOLDEN: expected_requests})
                assert not non_golden_ranks
            else:
                assert expected_requests == 1
                assert cell_hashes == collections.Counter({SOLO: 1})
                assert non_golden_ranks == [0]

            cell_rows.append(
                {
                    "cell": cell,
                    "condition": condition,
                    "decode_history": history,
                    "first_positive_ready": positive_ready[0],
                    "golden_matches": cell_hashes[GOLDEN],
                    "hash_counts": dict(sorted(cell_hashes.items())),
                    "max_ready": max(positive_ready),
                    "non_golden_admission_ranks": non_golden_ranks,
                    "prefix_cache_hits": metrics["prefix_cache_hits"],
                    "prefix_cache_misses": metrics["prefix_cache_misses"],
                    "ready_prefix": positive_ready[:8],
                    "requests": expected_requests,
                }
            )

        request_count = expected_cells * expected_requests
        matrix.append(
            {
                "cells": expected_cells,
                "condition": condition,
                "decode_history_counts": dict(sorted(history_counts.items())),
                "divergent_cells_vs_golden": divergent_cells,
                "errors": total_errors,
                "first_positive_ready_counts": {
                    str(key): value for key, value in sorted(first_ready_counts.items())
                },
                "golden_only_cells": golden_only_cells,
                "hash_counts": dict(sorted(condition_hashes.items())),
                "non_golden_requests": request_count - condition_hashes[GOLDEN],
                "prefix_cache_hits": total_prefix_hits,
                "prefix_cache_misses": total_prefix_misses,
                "requests": request_count,
                "transition_cells": transition_cells,
                "transition_ttft_delta_vs_peer_median_ms": (
                    {
                        "max": round(max(transition_ttft_leads_ms), 6),
                        "median": round(statistics.median(transition_ttft_leads_ms), 6),
                        "min": round(min(transition_ttft_leads_ms), 6),
                        "n": len(transition_ttft_leads_ms),
                    }
                    if transition_ttft_leads_ms
                    else None
                ),
            }
        )

    aggregate_hashes = collections.Counter(row["text_sha256"] for row in completion_rows)
    aggregate_histories = collections.Counter(row["decode_history"] for row in cell_rows)
    assert len(cell_rows) == 90
    assert len(completion_rows) == 590
    assert aggregate_hashes == collections.Counter({GOLDEN: 505, TRANSITION: 75, SOLO: 10})
    assert aggregate_histories == collections.Counter(
        {"solo-to-batch": 75, "batched-from-first": 5, "all-solo": 10}
    )

    result = {
        "aggregate": {
            "cells": len(cell_rows),
            "decode_history_counts": dict(sorted(aggregate_histories.items())),
            "hash_counts": dict(sorted(aggregate_hashes.items())),
            "requests": len(completion_rows),
        },
        "completion_classes": {
            "all_solo": SOLO,
            "golden": GOLDEN,
            "solo_to_batch": TRANSITION,
        },
        "conditions": matrix,
    }
    write_json(RAW / "reproduction-matrix.json", result)
    write_jsonl(RAW / "cell-analysis.jsonl", cell_rows)
    write_jsonl(RAW / "completion-hashes.jsonl", completion_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
