#!/usr/bin/env python3
"""Merge restartable split-map cells and correlate outcomes with named runtime geometry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TOTAL_TOKENS = 4860
MIN_PREFIX = 64
PREFILL_TICK = 1024
SOLO_PREFILL_TICK = 8192
PREFILL_BLOCK_Q = 64
PREFILL_BK = 32
PREFILL_HD512_SP_M = 16
GEMMA_FA512_MIN = 512
GEMMA_SWA_WINDOW = 1024
GEMMA_GLOBAL_SP = 32
GEMMA_SWA_SP = 64
GLOBAL_ROW_BYTES = 512
SWA_ROW_BYTES = 2048
PAGE_BYTES = 4096


def features(split: int) -> dict[str, int | str | bool]:
    first_tkv = split + 1
    fallback_split = 16 if first_tkv <= 2048 else 64
    return {
        "prefix_min_eligible": split >= MIN_PREFIX,
        "cold_prefill_execution": "gemma4_prime-monolithic-t4860",
        "restored_suffix_execution": "decode_step-tokenwise-t1",
        "worker_prefill_tick": PREFILL_TICK,
        "worker_solo_prefill_tick": SOLO_PREFILL_TICK,
        "decode_batch_provenance": "eager-b1-width-row-null",
        "split_mod_block_q64": split % PREFILL_BLOCK_Q,
        "split_mod_bk32": split % PREFILL_BK,
        "split_mod_sp_m16": split % PREFILL_HD512_SP_M,
        "suffix_tokens": TOTAL_TOKENS - split,
        "suffix_mod_block_q64": (TOTAL_TOKENS - split) % PREFILL_BLOCK_Q,
        "first_suffix_tkv": first_tkv,
        "global_first_suffix_class": (
            f"fa_decode_kvmod-scalar-sp{fallback_split}"
            if first_tkv < GEMMA_FA512_MIN
            else f"fa_decode_rows-sp{GEMMA_GLOBAL_SP}"
        ),
        "swa_first_suffix_class": (
            f"fa_decode_kvmod-vec-sp{fallback_split}"
            if first_tkv <= GEMMA_SWA_WINDOW
            else f"fa_decode_rows_w-sp{GEMMA_SWA_SP}"
        ),
        # fa_split_keys names a 16->64 rung at t_kv=2049 on this 188-SM rig, but Gemma's
        # rows/rows_w arms have already taken over at 512/1025. Keep both the nominal rung and
        # whether it is actually consumed by either first-suffix attention arm.
        "big_rig_fallback_split_ladder": fallback_split,
        "big_rig_fallback_split_live": (
            first_tkv < GEMMA_FA512_MIN or first_tkv <= GEMMA_SWA_WINDOW
        ),
        "actual_global_partition_keys": (
            fallback_split if first_tkv < GEMMA_FA512_MIN else GEMMA_GLOBAL_SP
        ),
        "actual_swa_partition_keys": (
            fallback_split if first_tkv <= GEMMA_SWA_WINDOW else GEMMA_SWA_SP
        ),
        "global_plane_page_offset": (split * GLOBAL_ROW_BYTES) % PAGE_BYTES,
        "swa_plane_page_offset": (split * SWA_ROW_BYTES) % PAGE_BYTES,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--require-dense", action="store_true")
    args = parser.parse_args()

    cells: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    prompt_modes: set[str] = set()
    target_prompt_hashes: set[str] = set()
    summaries_without_target_hash = 0
    for path in args.requests:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summaries = [row for row in rows if row.get("kind") == "summary"]
        if len(summaries) != 1 or summaries[0].get("verdict") != "MAP-COMPLETE":
            failures.append(f"{path}: missing MAP-COMPLETE summary")
        else:
            # Receipts predating the controlled constructor are the frozen lcprestore map.
            summary = summaries[0]
            prompt_modes.add(str(summary.get("map_prompts", "lcprestore")))
            hashes = summary.get("target_prompt_ids_sha256_canonical_json")
            if hashes is None:
                summaries_without_target_hash += 1
            elif not isinstance(hashes, list) or not all(isinstance(value, str) for value in hashes):
                failures.append(f"{path}: invalid target prompt hash receipt")
            else:
                target_prompt_hashes.update(hashes)
        for row in rows:
            if row.get("kind") != "map-cell":
                continue
            split = int(row["split"])
            if split in cells and cells[split]["verdict"] != row["verdict"]:
                failures.append(
                    f"split {split}: conflicting duplicate verdicts "
                    f"{cells[split]['verdict']} and {row['verdict']}"
                )
            cells[split] = row

    if len(prompt_modes) != 1:
        failures.append(f"mixed map prompt modes: {sorted(prompt_modes)}")
    map_prompts = next(iter(prompt_modes), "unknown")
    if map_prompts == "fixed-target":
        if summaries_without_target_hash:
            failures.append(
                f"fixed-target map has {summaries_without_target_hash} summary receipt(s) "
                "without a target prompt hash"
            )
        if len(target_prompt_hashes) != 1:
            failures.append(
                "fixed-target map does not have exactly one target prompt hash: "
                f"{sorted(target_prompt_hashes)}"
            )

    expected = list(range(64, 4375, 64))
    if 4374 not in expected:
        expected.append(4374)
    expected.sort()
    missing = sorted(set(expected) - cells.keys())
    if args.require_dense and missing:
        failures.append(f"dense map missing splits: {missing}")

    table = []
    for split, cell in sorted(cells.items()):
        row = {
            "split": split,
            "verdict": cell["verdict"],
            "restored_sha256": cell.get("restored_sha256"),
            "genuinely_cold_sha256": cell.get("genuinely_cold_sha256"),
        }
        row.update(features(split))
        table.append(row)

    feature_names = (
        "prefix_min_eligible",
        "cold_prefill_execution",
        "restored_suffix_execution",
        "worker_prefill_tick",
        "worker_solo_prefill_tick",
        "decode_batch_provenance",
        "split_mod_block_q64",
        "split_mod_bk32",
        "split_mod_sp_m16",
        "suffix_mod_block_q64",
        "global_first_suffix_class",
        "swa_first_suffix_class",
        "big_rig_fallback_split_ladder",
        "big_rig_fallback_split_live",
        "actual_global_partition_keys",
        "actual_swa_partition_keys",
        "global_plane_page_offset",
        "swa_plane_page_offset",
    )
    correlation: dict[str, dict[str, dict[str, int]]] = {}
    exact_discriminators: list[str] = []
    for name in feature_names:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0})
        for row in table:
            counts[str(row[name])][row["verdict"]] += 1
        correlation[name] = dict(counts)
        populated = [count for count in counts.values() if sum(count.values())]
        if len(populated) > 1 and all(not (count["PASS"] and count["FAIL"]) for count in populated):
            exact_discriminators.append(name)

    transitions = []
    targeted = set()
    for left, right in zip(table, table[1:]):
        if left["verdict"] == right["verdict"]:
            continue
        a, b = int(left["split"]), int(right["split"])
        transitions.append(
            {
                "left_split": a,
                "left_verdict": left["verdict"],
                "right_split": b,
                "right_verdict": right["verdict"],
            }
        )
        for value in (a + 1, b - 1):
            if MIN_PREFIX <= value < TOTAL_TOKENS and value not in cells:
                targeted.add(value)

    named_boundary_candidates = set()
    # These are split positions, not post-append t_kv values. The global rows arm begins when
    # first_tkv = split + 1 reaches GEMMA_FA512_MIN, hence its boundary is split 511.
    for boundary in (MIN_PREFIX, GEMMA_FA512_MIN - 1, GEMMA_SWA_WINDOW, 2048):
        for value in (boundary - 1, boundary, boundary + 1):
            if MIN_PREFIX <= value < TOTAL_TOKENS and value not in cells:
                named_boundary_candidates.add(value)
    targeted.update(named_boundary_candidates)

    summary = {
        "schema": "memra.splitiso.boundary-correlation.v1",
        "map_prompts": map_prompts,
        "target_prompt_ids_sha256_canonical_json": sorted(target_prompt_hashes),
        "summaries_without_target_hash": summaries_without_target_hash,
        "source_requests": [str(path) for path in args.requests],
        "expected_dense_splits": expected,
        "missing_dense_splits": missing,
        "table": table,
        "correlation": correlation,
        "exact_discriminators": exact_discriminators,
        "sampled_transitions": transitions,
        "targeted_candidates": sorted(targeted),
        "named_boundary_candidates": sorted(named_boundary_candidates),
        "execution_shape": {
            "cold": "worker.rs fresh eager-only prompt -> one 4860-token gemma4_prime",
            "restored": "worker.rs carried eager-only suffix -> one decode_step per token",
            "prefill_tiles": (
                "cold-only SWA BLOCK_Q=64/BK=32 (paired BLOCK_QH=32); global "
                "SP_M_ROWS=16/BK=32 (fallback BLOCK_Q=32/BK=32); split does not select them"
            ),
            "decode_batch": "Gemma eager per-session path; observed width and row are null",
            "kv_planes": "global 512 B/token, SWA 2048 B/token, allocations rows*token_bytes+8",
            "prefix_cache_min_tokens": MIN_PREFIX,
        },
        "failures": failures,
        "verdict": "COMPLETE" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# Split-boundary map reduction",
        "",
        f"Prompt construction: `{map_prompts}`.",
        "",
        "Target prompt token SHA-256 (canonical JSON): "
        f"`{','.join(sorted(target_prompt_hashes)) or 'not-recorded'}`.",
        "",
        "## Pass/fail table",
        "",
        "| Split | Output | Restored SHA-256 | Cold SHA-256 |",
        "|---:|:---:|---|---|",
    ]
    for row in table:
        markdown.append(
            f"| {row['split']} | {row['verdict']} | {row['restored_sha256']} | "
            f"{row['genuinely_cold_sha256']} |"
        )
    markdown.extend(
        [
            "",
            "## Correlation counts",
            "",
            "| Feature | Value | Pass | Fail |",
            "|---|---|---:|---:|",
        ]
    )
    for name, values in correlation.items():
        for value, count in sorted(values.items()):
            markdown.append(f"| `{name}` | `{value}` | {count['PASS']} | {count['FAIL']} |")
    markdown.extend(
        [
            "",
            f"Exact discriminators: `{', '.join(exact_discriminators) or 'none'}`.",
            "",
            f"Sampled transitions: `{json.dumps(transitions, sort_keys=True)}`.",
            "",
            f"Targeted candidates: `{','.join(map(str, sorted(targeted)))}`.",
            "",
            f"Named-boundary candidates: `{','.join(map(str, sorted(named_boundary_candidates)))}`.",
            "",
        ]
    )
    args.markdown.write_text("\n".join(markdown))
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
