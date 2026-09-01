#!/usr/bin/env python3
"""Reduce targeted split-state receipts after raw server logs have been captured."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DETAIL = "[prefix-cache-split-detail] "
SESSION = "[prefix-cache-split-session] "


def load(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    details: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    errors: list[str] = []
    pending: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if line.startswith(DETAIL):
                payload = line[len(DETAIL) :]
                if payload.startswith("ERROR"):
                    errors.append(f"{path}:{line_number}: {payload}")
                    continue
                row = json.loads(payload)
                row["_path"] = str(path)
                row["_line"] = line_number
                details.append(row)
                pending[(int(row["split"]), row["role"], row["why"])].append(row)
            elif line.startswith(SESSION):
                payload = line[len(SESSION) :]
                if payload.startswith("ERROR"):
                    errors.append(f"{path}:{line_number}: {payload}")
                    continue
                row = json.loads(payload)
                row["_path"] = str(path)
                row["_line"] = line_number
                key = (int(row["split"]), row["role"], row["why"])
                if pending[key]:
                    row["cache_detail"] = pending[key].pop()
                sessions.append(row)
    return details, sessions, errors


def one(rows: list[dict[str, Any]], description: str, failures: list[str]) -> dict[str, Any] | None:
    if len(rows) != 1:
        failures.append(f"{description}: found {len(rows)}, expected 1")
        return None
    return rows[0]


def layer_difference(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    left_layers = {int(row["layer"]): row for row in left.get("kv_layers", [])}
    right_layers = {int(row["layer"]): row for row in right.get("kv_layers", [])}
    if left_layers.keys() != right_layers.keys():
        return {
            "field": "kv.layer-set",
            "left": sorted(left_layers),
            "right": sorted(right_layers),
        }
    for layer in sorted(left_layers):
        for field in fields:
            if left_layers[layer].get(field) != right_layers[layer].get(field):
                return {
                    "field": f"kv.layer.{layer}.{field}",
                    "left": left_layers[layer].get(field),
                    "right": right_layers[layer].get(field),
                }
    return None


def value_difference(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    for field in fields:
        if left.get(field) != right.get(field):
            return {"field": field, "left": left.get(field), "right": right.get(field)}
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matrix-out", type=Path)
    args = parser.parse_args()

    details, sessions, parse_errors = load(args.logs)
    splits = sorted({int(row["split"]) for row in details + sessions})
    failures = list(parse_errors)
    reduced: dict[int, dict[str, Any]] = {}

    for split in splits:
        source = one(
            [
                row for row in details
                if int(row["split"]) == split
                and row["role"] == "source"
                and row["why"] == "immediate-partial"
            ],
            f"split {split} source detail",
            failures,
        )
        restored_boundary = one(
            [
                row for row in details
                if int(row["split"]) == split
                and row["role"] == "restored"
                and row["why"] == "immediate-partial"
            ],
            f"split {split} restored-boundary detail",
            failures,
        )
        boundary_cold = one(
            [
                row for row in details
                if int(row["split"]) == split
                and row["role"] == "boundary-cold"
                and row["why"] == "seed"
            ],
            f"split {split} boundary-cold detail",
            failures,
        )
        restored_full = one(
            [
                row for row in sessions
                if int(row["split"]) == split
                and row["role"] == "first-sample"
                and int(row["cached_tokens"]) == split
            ],
            f"split {split} restored first sample",
            failures,
        )
        cold_full = one(
            [
                row for row in sessions
                if int(row["split"]) == split
                and row["role"] == "first-sample"
                and "genuinely-cold-" in row["cache_namespace"]
            ],
            f"split {split} genuinely-cold first sample",
            failures,
        )

        source_restore_diff = None
        source_cold_diff = None
        final_cache_diff = None
        final_session_diff = None
        final_prefix_diff = None
        final_first_suffix_diff = None
        final_suffix_diff = None
        final_len_d_diff = None
        final_recurrent_diff = None
        first_logical_model_state_diff = None
        if source is not None and restored_boundary is not None:
            restored_normalized = dict(restored_boundary)
            restored_normalized["kv_layers"] = [
                dict(row, logical_len=row.get("len"))
                for row in restored_boundary.get("kv_layers", [])
            ]
            source_restore_diff = layer_difference(
                source,
                restored_normalized,
                ("logical_len", "k_tok_bytes", "v_tok_bytes", "k_sha256", "v_sha256"),
            )
            for row in restored_boundary.get("kv_layers", []):
                if row.get("len_d") != [split]:
                    source_restore_diff = {
                        "field": f"kv.layer.{row['layer']}.len_d",
                        "left": f"derived [{split}]",
                        "right": row.get("len_d"),
                    }
                    break
        if source is not None and boundary_cold is not None:
            source_cold_diff = layer_difference(
                source,
                boundary_cold,
                ("k_tok_bytes", "v_tok_bytes", "k_sha256", "v_sha256"),
            )
        if restored_full is not None and cold_full is not None:
            restored_cache = restored_full.get("cache_detail") or {}
            cold_cache = cold_full.get("cache_detail") or {}
            final_prefix_diff = layer_difference(
                restored_cache,
                cold_cache,
                ("prefix_k_sha256", "prefix_v_sha256"),
            )
            final_first_suffix_diff = layer_difference(
                restored_cache,
                cold_cache,
                ("first_suffix_k_sha256", "first_suffix_v_sha256"),
            )
            final_suffix_diff = layer_difference(
                restored_cache,
                cold_cache,
                ("suffix_k_sha256", "suffix_v_sha256", "k_sha256", "v_sha256"),
            )
            final_len_d_diff = layer_difference(restored_cache, cold_cache, ("len_d",))
            final_recurrent_diff = value_difference(
                restored_cache,
                cold_cache,
                ("recurrent_layers",),
            )
            # Compare persistent logical model state in token-boundary order. len_d is reported
            # separately: a fresh Gemma prime leaves that device mirror at zero until a rows arm
            # syncs it, while its host logical len is already authoritative for this eager path.
            first_logical_model_state_diff = (
                final_prefix_diff
                or final_recurrent_diff
                or final_first_suffix_diff
                or final_suffix_diff
            )
            final_cache_diff = layer_difference(
                restored_cache,
                cold_cache,
                (
                    "len",
                    "len_d",
                    "k_tok_bytes",
                    "v_tok_bytes",
                    "prefix_k_sha256",
                    "prefix_v_sha256",
                    "first_suffix_k_sha256",
                    "first_suffix_v_sha256",
                    "suffix_k_sha256",
                    "suffix_v_sha256",
                    "k_sha256",
                    "v_sha256",
                ),
            )
            if final_cache_diff is None:
                final_cache_diff = value_difference(
                    restored_cache,
                    cold_cache,
                    ("recurrent_layers", "last_logits_dev_sha256"),
                )
            final_session_diff = value_difference(
                restored_full,
                cold_full,
                (
                    "cache_pos",
                    "rope_next_position",
                    "engine_scoped_flags",
                    "sampler",
                    "boundary_logits",
                    "logits_producer",
                    "decode_batch_width",
                    "decode_batch_row",
                    "selected",
                    "selected_is_eos",
                ),
            )

        first = None
        first_scope = None
        for scope, difference in (
            ("source-vs-restored-boundary", source_restore_diff),
            ("source-vs-cold-boundary", source_cold_diff),
            ("restored-vs-cold-full-cache", final_cache_diff),
            ("restored-vs-cold-first-sample", final_session_diff),
        ):
            if difference is not None:
                first_scope = scope
                first = difference
                break
        reduced[split] = {
            "source_restore_difference": source_restore_diff,
            "source_boundary_cold_difference": source_cold_diff,
            "restored_cold_full_cache_difference": final_cache_diff,
            "restored_cold_first_sample_difference": final_session_diff,
            "restored_cold_prefix_kv_difference": final_prefix_diff,
            "restored_cold_first_suffix_kv_difference": final_first_suffix_diff,
            "restored_cold_suffix_aggregate_difference": final_suffix_diff,
            "restored_cold_len_d_difference": final_len_d_diff,
            "restored_cold_recurrent_difference": final_recurrent_diff,
            "first_logical_model_state_difference": first_logical_model_state_diff,
            "comparison_matrix": {
                "prefix_kv_equal": final_prefix_diff is None,
                "first_suffix_kv_equal": final_first_suffix_diff is None,
                "suffix_aggregate_equal": final_suffix_diff is None,
                "len_d_equal": final_len_d_diff is None,
                "recurrent_and_spare_equal": final_recurrent_diff is None,
                "cache_pos_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("cache_pos") == cold_full.get("cache_pos")
                ),
                "rope_position_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("rope_next_position")
                    == cold_full.get("rope_next_position")
                ),
                "sampler_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("sampler") == cold_full.get("sampler")
                ),
                "engine_scoped_flags_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("engine_scoped_flags")
                    == cold_full.get("engine_scoped_flags")
                ),
                "decode_batch_provenance_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("decode_batch_width")
                    == cold_full.get("decode_batch_width")
                    and restored_full.get("decode_batch_row")
                    == cold_full.get("decode_batch_row")
                ),
                "boundary_logits_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("boundary_logits") == cold_full.get("boundary_logits")
                ),
                "selected_first_token_equal": (
                    restored_full is not None
                    and cold_full is not None
                    and restored_full.get("selected") == cold_full.get("selected")
                ),
            },
            "first_divergence_scope": first_scope,
            "first_divergence": first,
            "restored_first_sample": restored_full,
            "genuinely_cold_first_sample": cold_full,
        }

    summary = {
        "schema": "memra.splitiso.targeted-detail.v1",
        "logs": [str(path) for path in args.logs],
        "splits": reduced,
        "failures": failures,
        "verdict": "COMPLETE" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.matrix_out is not None:
        matrix = {
            "schema": "memra.splitiso.field-comparison.v1",
            "source_logs": [str(path) for path in args.logs],
            "splits": {
                str(split): {
                    "comparison_matrix": row["comparison_matrix"],
                    "first_logical_model_state_difference": row[
                        "first_logical_model_state_difference"
                    ],
                    "len_d_difference": row["restored_cold_len_d_difference"],
                    "boundary_session_difference": row[
                        "restored_cold_first_sample_difference"
                    ],
                    "restored": {
                        "logits_producer": (row["restored_first_sample"] or {}).get(
                            "logits_producer"
                        ),
                        "decode_batch_width": (row["restored_first_sample"] or {}).get(
                            "decode_batch_width"
                        ),
                        "decode_batch_row": (row["restored_first_sample"] or {}).get(
                            "decode_batch_row"
                        ),
                        "selected": (row["restored_first_sample"] or {}).get("selected"),
                    },
                    "genuinely_cold": {
                        "logits_producer": (row["genuinely_cold_first_sample"] or {}).get(
                            "logits_producer"
                        ),
                        "decode_batch_width": (row["genuinely_cold_first_sample"] or {}).get(
                            "decode_batch_width"
                        ),
                        "decode_batch_row": (row["genuinely_cold_first_sample"] or {}).get(
                            "decode_batch_row"
                        ),
                        "selected": (row["genuinely_cold_first_sample"] or {}).get("selected"),
                    },
                }
                for split, row in sorted(reduced.items())
            },
            "failures": failures,
            "verdict": "COMPLETE" if not failures else "FAIL",
        }
        args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
        args.matrix_out.write_text(
            json.dumps(matrix, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
