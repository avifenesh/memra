#!/usr/bin/env python3
"""Reject an A/B pair before its throughput can enter the campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--kind", choices=("full", "mixed"), required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    args = parser.parse_args()

    expected_n = args.concurrency if args.kind == "full" else 20
    expected_tokens = 512 if args.kind == "full" else 60
    selected: dict[str, list[dict[str, Any]]] = {}
    for label in (args.left_label, args.right_label):
        label_rows = [
            row
            for row in rows(args.path)
            if row.get("kind") == "request" and row.get("label") == label
        ]
        if len(label_rows) != expected_n:
            raise ValueError(f"{label}: expected {expected_n} requests, got {len(label_rows)}")
        for index, row in enumerate(label_rows):
            expected = {
                "ok": True,
                "completion_tokens": expected_tokens,
                "finish_reason": "length",
            }
            drift = {
                key: (row.get(key), value)
                for key, value in expected.items()
                if row.get(key) != value
            }
            if drift:
                raise ValueError(f"{label} request {index} shape drift: {drift}")
            if not row.get("text_sha256"):
                raise ValueError(f"{label} request {index} has no output hash")
        selected[label] = label_rows

    hashes = {
        str(row["text_sha256"])
        for label_rows in selected.values()
        for row in label_rows
    }
    if len(hashes) != 1:
        by_label = {
            label: sorted({str(row["text_sha256"]) for row in label_rows})
            for label, label_rows in selected.items()
        }
        receipt = {
            "kind": "pair_exactness",
            "shape": args.kind,
            "concurrency": args.concurrency,
            "labels": [args.left_label, args.right_label],
            "requests_per_arm": expected_n,
            "completion_tokens_per_request": expected_tokens,
            "hashes_by_label": by_label,
            "verdict": "BYTE MISMATCH",
        }
        print(json.dumps(receipt, sort_keys=True))
        print(f"BYTE MISMATCH across A/B pair: {by_label}", file=sys.stderr)
        return 2

    receipt = {
        "kind": "pair_exactness",
        "shape": args.kind,
        "concurrency": args.concurrency,
        "labels": [args.left_label, args.right_label],
        "requests_per_arm": expected_n,
        "completion_tokens_per_request": expected_tokens,
        "text_sha256": next(iter(hashes)),
        "verdict": "BYTE IDENTICAL",
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
