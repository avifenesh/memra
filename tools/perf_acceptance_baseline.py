#!/usr/bin/env python3
"""Resolve a perf cell's acceptance baseline with evidence validation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_baseline(
    manifest_path: Path, history_path: Path, cell_id: str, repo_root: Path
) -> float:
    manifest = _load_json(manifest_path)
    cell = next(
        (entry for entry in manifest["cells"] if entry["id"] == cell_id), None
    )
    if cell is None:
        raise ValueError(f"unknown perf cell: {cell_id}")

    settled = cell.get("acceptance_baseline")
    if settled is not None:
        evidence_path = repo_root / settled["evidence"]
        evidence = _load_json(evidence_path)
        field = settled.get("evidence_field", "acceptance_both_arms")
        evidence_value = float(evidence[field])
        declared_value = float(settled["value"])
        if abs(evidence_value - declared_value) > 1e-9:
            raise ValueError(
                f"{cell_id}: declared acceptance baseline {declared_value} "
                f"does not match {evidence_path}:{field}={evidence_value}"
            )
        return declared_value

    values: list[float] = []
    if history_path.exists():
        with history_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("cell") == cell_id and row.get("accept") is not None:
                    values.append(float(row["accept"]))
    return statistics.median_high(values[-5:]) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(
        resolve_baseline(
            args.manifest, args.history, args.cell, args.repo_root.resolve()
        )
    )


if __name__ == "__main__":
    main()
