"""Explain native heldout timing by prompt length and output phase."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def rate(rows):
    tokens = sum(row["tokens"] for row in rows)
    seconds = sum(row["seconds"] for row in rows)
    return {
        "cells": len(rows),
        "tokens": tokens,
        "seconds": seconds,
        "tok_s": tokens / seconds if seconds else None,
    }


def score(root):
    summary = json.loads((root / "heldout-summary.json").read_text())
    by_arm = defaultdict(list)
    for record in summary["records"]:
        by_arm[record["variant"]].append(record)
    baseline = by_arm["k3-c0"]
    if len(baseline) != 6:
        raise ValueError("fresh K3/C0 baseline is not six conversations")
    prompt_lengths = sorted(
        int(row["prompt_tokens"])
        for record in baseline
        for row in table(root / record["name"] / "turns.tsv")
    )
    low = prompt_lengths[len(prompt_lengths) // 3]
    high = prompt_lengths[2 * len(prompt_lengths) // 3]
    report = {}
    for arm, records in by_arm.items():
        prompt_cells = defaultdict(list)
        phase_cells = defaultdict(list)
        transitions = 0
        continuation = []
        for record in records:
            session = root / record["name"]
            for row in table(session / "turns.tsv"):
                length = int(row["prompt_tokens"])
                band = "short" if length < low else "medium" if length < high else "long"
                prompt_cells[band].append({
                    "tokens": int(row["output_tokens"]),
                    "seconds": float(row["elapsed_s"]),
                })
                if int(row["turn"]) > 1:
                    continuation.append({
                        "cached": int(row["cached_tokens"]),
                        "new": int(row["new_input_tokens"]),
                        "resumed": row["resumed"] == "true",
                    })
            for row in table(session / "spans.tsv"):
                if row["eligible"] != "true":
                    continue
                before = row["context_before"]
                after = row["context_after"]
                transitions += before != after
                phase_cells[before].append({
                    "tokens": int(row["output_end"]) - int(row["output_start"]),
                    "seconds": int(row["elapsed_ns"]) / 1e9,
                })
        if len(continuation) != 42 or any(
            not row["resumed"] or row["cached"] <= 0 or row["new"] <= 0
            for row in continuation
        ):
            raise ValueError("fresh arm has a missing native continuing-session receipt")
        report[arm] = {
            "by_prompt_length_complete_request": {
                name: rate(rows) for name, rows in prompt_cells.items()
            },
            "by_output_phase_round_only": {
                name: rate(rows) for name, rows in phase_cells.items()
            },
            "eligible_output_phase_transitions": transitions,
            "later_turns_with_positive_cached_and_new_tokens": len(continuation),
        }
    return {
        "schema": 1,
        "scope": "fresh code heldout descriptive cells; round cells are not E2E",
        "baseline_prompt_tertiles": [low, high],
        "arms": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.root)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({"arms": len(result["arms"])}))


if __name__ == "__main__":
    main()
