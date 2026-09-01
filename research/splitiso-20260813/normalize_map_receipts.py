#!/usr/bin/env python3
"""Normalize an already-captured boundary map whose only strict-helper failure was normal EOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def map_response_ok(row: dict[str, Any]) -> bool:
    if row.get("ok"):
        return True
    completion_tokens = row.get("completion_tokens")
    return bool(
        row.get("http_status") == 200
        and row.get("done")
        and row.get("request_id")
        and row.get("finish_reason") == "stop"
        and isinstance(completion_tokens, int)
        and completion_tokens > 0
        and not row.get("error")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    summaries = [row for row in rows if row.get("kind") == "summary"]
    if len(summaries) != 1:
        raise ValueError(f"found {len(summaries)} summaries, expected one")
    original = summaries[0]
    splits = [int(value) for value in original["split_boundaries"]]
    failures: list[str] = []
    expected_cached = {
        "request1-seed": 0,
        "request2-restored": None,
        "request2-genuinely-cold": 0,
        "boundary-cold": 0,
    }
    for split in splits:
        for case, expected in expected_cached.items():
            found = [
                row for row in rows
                if row.get("kind") == "request"
                and int(row.get("split", -1)) == split
                and row.get("case") == case
            ]
            if len(found) != 1:
                failures.append(f"split={split} case={case}: found {len(found)}, expected 1")
                continue
            row = found[0]
            row["map_response_ok"] = map_response_ok(row)
            if not row["map_response_ok"]:
                failures.append(f"split={split} case={case}: response invalid: {row.get('error')}")
            wanted = split if expected is None else expected
            actual = int(row.get("cached_tokens") or 0)
            if actual != wanted:
                failures.append(
                    f"split={split} case={case}: cached_tokens {actual} != {wanted}"
                )

    summary = dict(original)
    summary["infrastructure_failures"] = failures
    summary["verdict"] = "MAP-COMPLETE" if not failures else "ERROR"
    summary["normalization"] = {
        "schema": "memra.splitiso.map-eos-normalization.v1",
        "input": str(args.input),
        "rule": "HTTP 200 + SSE DONE + request id + positive completion + finish_reason stop",
        "original_verdict": original.get("verdict"),
        "original_infrastructure_failures": original.get("infrastructure_failures"),
    }
    output_rows = [row for row in rows if row.get("kind") != "summary"] + [summary]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for row in output_rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
