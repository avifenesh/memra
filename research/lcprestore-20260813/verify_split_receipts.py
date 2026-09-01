#!/usr/bin/env python3
"""Parse split-state logs after capture; never score through a pipe."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RECEIPT = re.compile(
    r"\[prefix-cache-split-state\] role=(?P<role>\S+) why=(?P<why>\S+) "
    r"split=(?P<split>\d+) .*?state_sha256=(?P<state>[0-9a-f]{64}) "
    r"boundary_logits_sha256=(?P<logits>\S+)"
)


def parse(path: Path) -> list[dict[str, str | int]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = RECEIPT.search(line)
        if match:
            row: dict[str, str | int] = match.groupdict()
            row["split"] = int(row["split"])
            row["line"] = line_number
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-lock", required=True)
    args = parser.parse_args()

    request_rows = [json.loads(line) for line in args.requests.read_text().splitlines()]
    summaries = [row for row in request_rows if row.get("kind") == "summary"]
    if len(summaries) != 1 or summaries[0].get("verdict") != "PASS":
        raise ValueError("request exactness summary is absent or not PASS")
    cells = Counter(
        int(row["split"])
        for row in request_rows
        if row.get("kind") == "request"
        and row.get("arm") == "control"
        and row.get("case") == "request2"
    )
    control = parse(args.control_log)
    candidate = parse(args.candidate_log)
    errors = [
        line
        for path in (args.control_log, args.candidate_log)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "[prefix-cache-split-state] ERROR" in line
    ]
    failures = list(errors)
    by_split: dict[int, dict[str, object]] = {}
    for split, expected_cells in sorted(cells.items()):
        baseline = [
            row for row in control
            if row["split"] == split and row["role"] == "snapshot" and row["why"] == "lcp-split"
        ]
        source = [
            row for row in candidate
            if row["split"] == split and row["role"] == "source"
            and row["why"] == "immediate-partial"
        ]
        restored = [
            row for row in candidate
            if row["split"] == split and row["role"] == "restored"
            and row["why"] == "immediate-partial"
        ]
        expected_partials = expected_cells * 2
        if len(baseline) != expected_cells:
            failures.append(
                f"split {split}: control boundary snapshots {len(baseline)} != {expected_cells}"
            )
        partial_counts_match = (
            len(source) == expected_partials and len(restored) == expected_partials
        )
        if not partial_counts_match:
            failures.append(
                f"split {split}: source/restored counts {len(source)}/{len(restored)} "
                f"!= {expected_partials}/{expected_partials}"
            )
        states = {
            "control_lcp_boundary": sorted({str(row["state"]) for row in baseline}),
            "source": sorted({str(row["state"]) for row in source}),
            "restored": sorted({str(row["state"]) for row in restored}),
        }
        mismatched_pairs = [
            index
            for index, (source_row, restored_row) in enumerate(zip(source, restored), 1)
            if source_row["state"] != restored_row["state"]
        ]
        if mismatched_pairs:
            failures.append(
                f"split {split}: source/restored state mismatch at pairs {mismatched_pairs}: "
                f"{states}"
            )
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(row["logits"])) for row in baseline):
            failures.append(f"split {split}: cold boundary logits digest missing")
        if any(row["logits"] != "n/a-not-consumed" for row in source + restored):
            failures.append(f"split {split}: partial path claimed to consume boundary logits")
        by_split[split] = {
            "control_boundary_snapshots": len(baseline),
            "candidate_sources": len(source),
            "candidate_restores": len(restored),
            "control_lcp_boundary_state_sha256": states["control_lcp_boundary"],
            "source_state_sha256": states["source"],
            "restored_state_sha256": states["restored"],
            "source_restore_pairs_equal": partial_counts_match and not mismatched_pairs,
            "control_lcp_boundary_logits_sha256": sorted(
                {str(row["logits"]) for row in baseline}
            ),
            "partial_boundary_logits": "n/a-not-consumed; non-empty suffix is mandatory",
        }

    summary = {
        "schema": "memra.lcprestore.split-state-receipts.v1",
        "control_log": str(args.control_log),
        "candidate_log": str(args.candidate_log),
        "physical_gpu_index": args.physical_gpu,
        "physical_gpu_uuid": args.gpu_uuid,
        "gpu_lock": args.gpu_lock,
        "splits": by_split,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
