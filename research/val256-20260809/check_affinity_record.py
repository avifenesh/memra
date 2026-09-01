#!/usr/bin/env python3
"""Fail before A/B if the recorded workload is not deep rewritten history."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--token-floor", type=int, default=32768)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    sequential = [
        row
        for row in rows
        if row.get("type") == "request" and row.get("phase") == "sequential"
    ]
    failures = []
    max_prompt = max((row.get("prompt_tokens") or 0 for row in sequential), default=0)
    not_rewritten = [row.get("phase_index") for row in sequential if not row.get("history_rewritten")]
    if max_prompt <= args.token_floor:
        failures.append(f"max prompt tokens {max_prompt} did not cross {args.token_floor}")
    if not_rewritten:
        failures.append(f"reasoning was not stripped from sequential turns {not_rewritten}")
    receipt = {
        "sequential_turns": len(sequential),
        "min_prompt_tokens": min((row.get("prompt_tokens") or 0 for row in sequential), default=0),
        "max_prompt_tokens": max_prompt,
        "history_rewritten_turns": len(sequential) - len(not_rewritten),
        "not_rewritten_turns": not_rewritten,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
