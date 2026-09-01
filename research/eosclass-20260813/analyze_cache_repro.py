#!/usr/bin/env python3
"""Compare sequential and batched Q27 prefix snapshots and EOS receipts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SNAPSHOT_PREFIX = "[eosclass-snapshot] "
RESTORE_PREFIX = "[eosclass-restore] "
SAMPLE_PREFIX = "[eosclass-sample] "
PREFIX_ID = re.compile(r"-hot-(\d+)$")


def tagged_rows(path: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        marker = line.find(prefix)
        if marker < 0:
            continue
        try:
            row = json.loads(line[marker + len(prefix) :])
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: malformed receipt: {error}") from error
        row["_line"] = line_number
        rows.append(row)
    return rows


def client_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: malformed client row: {error}") from error
        row["_line"] = line_number
        rows.append(row)
    return rows


def prefix_id(row: dict[str, Any]) -> int:
    namespace = str(row.get("cache_namespace") or "")
    match = PREFIX_ID.search(namespace)
    if match is None:
        raise ValueError(f"snapshot namespace has no hot-prefix id: {namespace!r}")
    return int(match.group(1))


def by_layer(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["layer"]): row for row in rows}


def snapshot_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_count": row["token_count"],
        "tokens_sha256_u32le": row["tokens_sha256_u32le"],
        "pos": row["pos"],
        "bytes": row["bytes"],
        "kv_sha256": row["kv_sha256"],
        "recurrent_sha256": row["recurrent_sha256"],
        "last_logits_sha256": row["last_logits"]["sha256_f32le"],
    }


def restored_state_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_count": row["token_count"],
        "tokens_sha256_u32le": row["tokens_sha256_u32le"],
        "pos": row["pos"],
        "kv_sha256": row["kv_sha256"],
        "recurrent_sha256": row["recurrent_sha256"],
        "last_logits_sha256": row["last_logits"]["sha256_f32le"],
    }


def first_layer_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    left_by_layer = by_layer(left)
    right_by_layer = by_layer(right)
    if left_by_layer.keys() != right_by_layer.keys():
        return {
            "kind": "layer-set",
            "left": sorted(left_by_layer),
            "right": sorted(right_by_layer),
        }
    for layer in sorted(left_by_layer):
        changed = [
            field
            for field in fields
            if left_by_layer[layer].get(field) != right_by_layer[layer].get(field)
        ]
        if changed:
            return {"kind": "field", "layer": layer, "fields": changed}
    return None


def compare_snapshots(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_sig = snapshot_signature(left)
    right_sig = snapshot_signature(right)
    changed = [key for key in left_sig if left_sig[key] != right_sig[key]]
    return {
        "same": not changed,
        "changed_top_level": changed,
        "first_kv_difference": first_layer_difference(
            left["kv_layers"],
            right["kv_layers"],
            ("len", "k_bytes", "v_bytes", "k_sha256", "v_sha256"),
        ),
        "first_recurrent_difference": first_layer_difference(
            left["recurrent_layers"],
            right["recurrent_layers"],
            ("conv_len", "ssm_len", "conv_sha256", "ssm_sha256"),
        ),
        "sequential": left_sig,
        "batched": right_sig,
    }


def selected_eos_logit(row: dict[str, Any]) -> dict[str, Any] | None:
    selected = int(row["selected"])
    return next(
        (entry for entry in row["logits"]["eos"] if int(entry["id"]) == selected),
        None,
    )


def is_hit(row: dict[str, Any]) -> bool:
    return row.get("kind") == "hit" or (
        row.get("kind") == "mixed_request" and row.get("role") == "hit"
    )


def width_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in rows:
        width = row.get("batch_width")
        index = int(row["generated_index"])
        if runs and runs[-1]["width"] == width:
            runs[-1]["last_generated_index"] = index
            runs[-1]["samples"] += 1
        else:
            runs.append(
                {
                    "width": width,
                    "first_generated_index": index,
                    "last_generated_index": index,
                    "samples": 1,
                }
            )
    return runs


def summarize_run(path: Path) -> dict[str, Any]:
    snapshots = [
        row
        for row in tagged_rows(path / "server.log", SNAPSHOT_PREFIX)
        if row.get("why") == "seed"
    ]
    restores = tagged_rows(path / "server.log", RESTORE_PREFIX)
    samples = tagged_rows(path / "server.log", SAMPLE_PREFIX)
    clients = client_rows(path / "client.jsonl")
    by_prefix: dict[int, dict[str, Any]] = {}
    for row in snapshots:
        identity = prefix_id(row)
        if identity in by_prefix:
            raise ValueError(f"{path}: duplicate seed snapshot for prefix {identity}")
        by_prefix[identity] = row

    requests = {
        str(row.get("request_id")): row
        for row in clients
        if row.get("request_id")
    }
    restore_comparisons = []
    for row in restores:
        identity = prefix_id(row)
        snapshot = by_prefix.get(identity)
        restored = restored_state_signature(row)
        seeded = restored_state_signature(snapshot) if snapshot else None
        restore_comparisons.append(
            {
                "prefix_id": identity,
                "trace_id": row.get("trace_id"),
                "server_line": row["_line"],
                "same_as_seed_snapshot": seeded == restored,
                "seed_snapshot": seeded,
                "restored": restored,
            }
        )
    eos_samples = []
    for row in samples:
        if not row.get("selected_is_eos"):
            continue
        eos_logit = selected_eos_logit(row)
        top1 = row["logits"]["top1"].get("id")
        eos_samples.append(
            {
                "trace_id": row.get("trace_id"),
                "sample_source": row.get("sample_source"),
                "selected": row.get("selected"),
                "generated_index": row.get("generated_index"),
                "batch_width": row.get("batch_width"),
                "batch_row": row.get("batch_row"),
                "top1_id": top1,
                "selected_eos_rank": eos_logit.get("rank") if eos_logit else None,
                "selected_eos_logit": eos_logit.get("logit") if eos_logit else None,
                "host_logits_favor_eos": bool(
                    row.get("sample_source") != "device"
                    and top1 == row.get("selected")
                    and eos_logit
                    and eos_logit.get("rank") == 1
                ),
                "client": requests.get(str(row.get("trace_id"))),
                "server_line": row["_line"],
            }
        )

    hits = [row for row in clients if is_hit(row)]
    early_hits = [
        row
        for row in hits
        if int(row.get("cached_tokens") or 0) == 4860
        and int(row.get("completion_tokens") or 0) < 60
        and row.get("finish_reason") == "stop"
    ]
    signatures = {
        json.dumps(snapshot_signature(row), sort_keys=True)
        for row in snapshots
    }
    samples_by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        samples_by_trace.setdefault(str(row.get("trace_id")), []).append(row)
    numeric_paths = []
    for row in hits:
        trace_id = str(row.get("request_id"))
        path = samples_by_trace.get(trace_id, [])
        runs = width_runs(path)
        numeric_paths.append(
            {
                "trace_id": trace_id,
                "role": row.get("role"),
                "repetition": row.get("repetition"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "text_sha256": row.get("text_sha256"),
                "sample_count": len(path),
                "width_runs": runs,
                "width_transitions": [
                    {
                        "generated_index": run["first_generated_index"],
                        "from": runs[index - 1]["width"],
                        "to": run["width"],
                    }
                    for index, run in enumerate(runs)
                    if index > 0
                ],
            }
        )
    return {
        "path": str(path),
        "snapshot_count": len(snapshots),
        "snapshot_classes": len(signatures),
        "snapshot_prefix_ids": sorted(by_prefix),
        "restore_count": len(restores),
        "restore_comparisons": restore_comparisons,
        "snapshots": by_prefix,
        "hit_count": len(hits),
        "full_cache_hits": sum(int(row.get("cached_tokens") or 0) == 4860 for row in hits),
        "early_eos_hits": [
            {
                "prefix_id": row.get("prefix_id"),
                "repetition": row.get("repetition"),
                "request_id": row.get("request_id"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "text_sha256": row.get("text_sha256"),
                "client_line": row["_line"],
            }
            for row in early_hits
        ],
        "numeric_paths": numeric_paths,
        "eos_samples": eos_samples,
    }


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "snapshots"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", required=True, type=Path)
    parser.add_argument("--batched", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sequential = summarize_run(args.sequential)
    batched = summarize_run(args.batched)
    shared = sorted(set(sequential["snapshots"]) & set(batched["snapshots"]))
    comparisons = {
        str(identity): compare_snapshots(
            sequential["snapshots"][identity], batched["snapshots"][identity]
        )
        for identity in shared
    }
    summary = {
        "schema": "memra.eosclass.cache-state-comparison.v1",
        "sequential": public_run(sequential),
        "batched": public_run(batched),
        "shared_prefix_ids": shared,
        "comparisons": comparisons,
        "different_snapshots": sum(not row["same"] for row in comparisons.values()),
        "verdict": "DIFFERENT" if any(not row["same"] for row in comparisons.values()) else "SAME",
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
