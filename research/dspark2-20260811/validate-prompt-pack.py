#!/usr/bin/env python3
"""Fail-closed validation for the frozen DSpark prompt text and assignment metadata."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_key(prompt: str) -> str:
    normalized = " ".join(prompt.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate(root: Path, prefix: int) -> dict:
    pack_path = root / "prompts.promptpack"
    metadata_path = root / "prompts.tsv"
    summary_path = root / "prompt-pack-summary.json"
    summary = json.loads(summary_path.read_text())
    payload = pack_path.read_bytes()
    if not payload.endswith(b"\0"):
        raise ValueError("prompt pack is not NUL terminated")
    prompts = [value.decode("utf-8") for value in payload[:-1].split(b"\0")]
    lines = metadata_path.read_text().splitlines()
    expected_header = "id\tsplit\tmode\tcategory\tsource\tprompt_sha256"
    if not lines or lines[0] != expected_header:
        raise ValueError("unexpected prompt metadata header")
    if len(lines) - 1 != len(prompts) or len(prompts) != int(summary["limit"]):
        raise ValueError("prompt text/metadata/summary row counts differ")
    if not 1 <= prefix <= len(prompts):
        raise ValueError("prefix must be inside the prompt pack")
    if sha256(pack_path) != summary["prompt_pack_sha256"]:
        raise ValueError("prompt pack hash disagrees with summary")
    if sha256(metadata_path) != summary["prompt_tsv_sha256"]:
        raise ValueError("prompt metadata hash disagrees with summary")

    keys: set[str] = set()
    full_cells: collections.Counter[str] = collections.Counter()
    prefix_cells: collections.Counter[str] = collections.Counter()
    prefix_categories: collections.Counter[str] = collections.Counter()
    prefix_modes: collections.Counter[str] = collections.Counter()
    prefix_splits: collections.Counter[str] = collections.Counter()
    for expected_id, (prompt, line) in enumerate(zip(prompts, lines[1:])):
        fields = line.split("\t")
        if len(fields) != 6 or int(fields[0]) != expected_id:
            raise ValueError(f"malformed metadata row {expected_id}")
        split, mode, category, key = fields[1], fields[2], fields[3], fields[5]
        if split not in {"train", "heldout"}:
            raise ValueError(f"invalid split at row {expected_id}")
        if mode not in {"think", "nothink"}:
            raise ValueError(f"invalid mode at row {expected_id}")
        if category not in {"chat", "math", "code", "if"}:
            raise ValueError(f"invalid category at row {expected_id}")
        if key != prompt_key(prompt) or key in keys:
            raise ValueError(f"prompt hash mismatch or duplicate at row {expected_id}")
        keys.add(key)
        cell = f"{category}/{mode}/{split}"
        full_cells[cell] += 1
        if expected_id < prefix:
            prefix_cells[cell] += 1
            prefix_categories[category] += 1
            prefix_modes[mode] += 1
            prefix_splits[split] += 1

    if dict(sorted(full_cells.items())) != summary["assignment_cells"]:
        raise ValueError("assignment cell counts disagree with summary")
    if len(full_cells) != 16 or len(prefix_cells) != 16:
        raise ValueError("full pack or pilot prefix is missing an assignment cell")
    full_modes = collections.Counter()
    full_splits = collections.Counter()
    for cell, count in full_cells.items():
        _, mode, split = cell.split("/")
        full_modes[mode] += count
        full_splits[split] += count
    if dict(sorted(full_modes.items())) != summary["mode_counts"]:
        raise ValueError("mode counts disagree with summary")
    if full_splits != collections.Counter(
        {"train": int(summary["train"]), "heldout": int(summary["heldout"])}
    ):
        raise ValueError("split counts disagree with summary")

    return {
        "format": "memra-dspark-prompt-validation-v1",
        "assignment_version": summary["assignment_version"],
        "dataset_revision": summary["dataset_revision"],
        "prompts": len(prompts),
        "unique_prompt_keys": len(keys),
        "prompt_pack_sha256": sha256(pack_path),
        "prompt_tsv_sha256": sha256(metadata_path),
        "full_cells": dict(sorted(full_cells.items())),
        "prefix": prefix,
        "prefix_cells": dict(sorted(prefix_cells.items())),
        "prefix_categories": dict(sorted(prefix_categories.items())),
        "prefix_modes": dict(sorted(prefix_modes.items())),
        "prefix_splits": dict(sorted(prefix_splits.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prefix", type=int, default=2_000)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = validate(args.root, args.prefix)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
