#!/usr/bin/env python3
"""Reduce a contiguous DSpark chunk range into one evidence receipt."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        raise ValueError("cannot take percentile of an empty sequence")
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def verify_manifest(chunk: Path) -> str:
    manifest = chunk / "sha256.txt"
    if not manifest.is_file() or not (chunk / ".remote-verified").is_file():
        raise ValueError(f"{chunk}: missing manifest or remote-verification marker")
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{manifest}:{line_number}: malformed manifest row")
        expected, relative = fields
        path = chunk / relative.strip()
        if path.resolve().parent != chunk.resolve() and chunk.resolve() not in path.resolve().parents:
            raise ValueError(f"{manifest}:{line_number}: path escapes chunk")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path}: expected {expected}, got {actual}")
    return sha256(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.end <= args.start:
        raise SystemExit("--end must exceed --start")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.label):
        raise SystemExit("invalid --label")

    pattern = re.compile(rf"^{re.escape(args.label)}-(\d{{5}})-(\d{{5}})$")
    chunks: list[tuple[int, int, Path]] = []
    for path in args.root.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_dir():
            begin, finish = map(int, match.groups())
            if begin >= args.start and finish <= args.end:
                chunks.append((begin, finish, path))
    chunks.sort()
    if not chunks:
        raise ValueError("no matching chunks")

    next_id = args.start
    manifest_rows = []
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    splits: collections.Counter[str] = collections.Counter()
    modes: collections.Counter[str] = collections.Counter()
    categories: collections.Counter[str] = collections.Counter()
    assignment_cells: collections.Counter[str] = collections.Counter()
    deficient_records_by_cell: collections.Counter[str] = collections.Counter()
    zero_anchor_pairs_by_cell: collections.Counter[str] = collections.Counter()
    anchor_counts: list[int] = []
    requested_anchors: int | None = None
    records = 0
    skipped_short = 0
    validation_weight = 0
    weighted_tail = 0.0
    weighted_top64 = 0.0
    max_tail = 0.0
    max_mass_error = 0.0
    max_ratio_error = 0.0

    for begin, finish, chunk in chunks:
        if begin != next_id or finish <= begin:
            raise ValueError(f"non-contiguous chunk {chunk.name}, expected begin {next_id}")
        manifest_hash = verify_manifest(chunk)
        manifest_rows.append({"chunk": chunk.name, "sha256": manifest_hash})
        pairs_path = chunk / "generated" / "pairs.tsv"
        lines = pairs_path.read_text().splitlines()
        if not lines or lines[0] != "# memra-dspark-pairs-v1":
            raise ValueError(f"{pairs_path}: invalid header")
        if len(lines) - 1 != finish - begin:
            raise ValueError(f"{pairs_path}: row count does not match chunk range")
        chunk_cells: list[str] = []
        for expected_id, line in enumerate(lines[1:], begin):
            fields = line.split("\t", 7)
            if len(fields) != 8 or int(fields[0]) != expected_id:
                raise ValueError(f"{pairs_path}: non-contiguous or malformed pair {expected_id}")
            prompt_len, response_len, total_len = map(int, fields[4:7])
            if prompt_len + response_len != total_len:
                raise ValueError(f"{pairs_path}: length mismatch at pair {expected_id}")
            prompt_lengths.append(prompt_len)
            response_lengths.append(response_len)
            splits[fields[1]] += 1
            modes[fields[2]] += 1
            categories[fields[3]] += 1
            cell = f"{fields[3]}/{fields[2]}/{fields[1]}"
            assignment_cells[cell] += 1
            chunk_cells.append(cell)

        extraction = json.loads((chunk / "extracted" / "extraction.meta.json").read_text())
        validation = json.loads((chunk / "validation.json").read_text())
        chunk_records = int(extraction["records"])
        if int(validation["records"]) != chunk_records:
            raise ValueError(f"{chunk}: validation/extraction record mismatch")
        chunk_requested = int(extraction["anchors_per_pair"])
        if requested_anchors is None:
            requested_anchors = chunk_requested
        elif requested_anchors != chunk_requested:
            raise ValueError(f"{chunk}: anchors-per-pair changed within the frozen range")
        index_path = chunk / "extracted" / "index.tsv"
        index_lines = index_path.read_text().splitlines()
        expected_header = "record\tpair_id\tanchor_pos\tprompt_len\tsplit\tmode\tcategory"
        if not index_lines or index_lines[0] != expected_header:
            raise ValueError(f"{index_path}: invalid header")
        if len(index_lines) - 1 != chunk_records:
            raise ValueError(f"{index_path}: row count does not match extraction metadata")
        chunk_anchor_counts: collections.Counter[int] = collections.Counter()
        for expected_record, line in enumerate(index_lines[1:]):
            fields = line.split("\t")
            pair_id = int(fields[1]) if len(fields) == 7 else -1
            if (
                len(fields) != 7
                or int(fields[0]) != expected_record
                or not begin <= pair_id < finish
            ):
                raise ValueError(f"{index_path}: malformed record {expected_record}")
            chunk_anchor_counts[pair_id] += 1
        chunk_counts = [chunk_anchor_counts[pair_id] for pair_id in range(begin, finish)]
        if any(count > chunk_requested for count in chunk_counts):
            raise ValueError(f"{index_path}: pair exceeds requested anchor count")
        if sum(count == 0 for count in chunk_counts) != int(extraction["skipped_short"]):
            raise ValueError(f"{index_path}: zero-anchor count disagrees with metadata")
        for cell, count in zip(chunk_cells, chunk_counts):
            if count < chunk_requested:
                deficient_records_by_cell[cell] += chunk_requested - count
            if count == 0:
                zero_anchor_pairs_by_cell[cell] += 1
        anchor_counts.extend(chunk_counts)
        records += chunk_records
        skipped_short += int(extraction["skipped_short"])
        validation_weight += chunk_records
        weighted_tail += float(validation["tail_mass_mean"]) * chunk_records
        weighted_top64 += float(validation["sampled_token_top64_rate"]) * chunk_records
        max_tail = max(max_tail, float(validation["tail_mass_max"]))
        max_mass_error = max(max_mass_error, float(validation["max_probability_mass_error"]))
        max_ratio_error = max(
            max_ratio_error, float(validation["max_logit_probability_ratio_error"])
        )
        next_id = finish

    if next_id != args.end:
        raise ValueError(f"range ends at {next_id}, expected {args.end}")

    fingerprint = hashlib.sha256()
    for row in manifest_rows:
        fingerprint.update(row["chunk"].encode())
        fingerprint.update(b"\0")
        fingerprint.update(row["sha256"].encode())
        fingerprint.update(b"\n")
    sorted_prompt = sorted(prompt_lengths)
    sorted_response = sorted(response_lengths)
    summary = {
        "format": "memra-dspark-corpus-summary-v2",
        "label": args.label,
        "start": args.start,
        "end": args.end,
        "pairs": len(response_lengths),
        "chunks": len(chunks),
        "records": records,
        "skipped_short_pairs": skipped_short,
        "anchor_sampling": {
            "requested_per_pair": requested_anchors,
            "actual_records": sum(anchor_counts),
            "maximum_records": len(anchor_counts) * int(requested_anchors),
            "deficient_records": len(anchor_counts) * int(requested_anchors)
            - sum(anchor_counts),
            "pairs_below_requested": sum(
                count < int(requested_anchors) for count in anchor_counts
            ),
            "min_per_pair": min(anchor_counts),
            "max_per_pair": max(anchor_counts),
            "deficient_records_by_cell": dict(sorted(deficient_records_by_cell.items())),
            "zero_anchor_pairs_by_cell": dict(sorted(zero_anchor_pairs_by_cell.items())),
        },
        "prompt_tokens": {
            "total": sum(prompt_lengths),
            "min": sorted_prompt[0],
            "p50": percentile(sorted_prompt, 0.50),
            "p95": percentile(sorted_prompt, 0.95),
            "max": sorted_prompt[-1],
        },
        "response_tokens": {
            "total": sum(response_lengths),
            "min": sorted_response[0],
            "p50": percentile(sorted_response, 0.50),
            "p95": percentile(sorted_response, 0.95),
            "max": sorted_response[-1],
            "max_new_512_count": sum(length == 512 for length in response_lengths),
        },
        "splits": dict(sorted(splits.items())),
        "modes": dict(sorted(modes.items())),
        "categories": dict(sorted(categories.items())),
        "assignment_cells": dict(sorted(assignment_cells.items())),
        "validation": {
            "sampled_token_top64_rate": weighted_top64 / validation_weight,
            "tail_mass_mean": weighted_tail / validation_weight,
            "tail_mass_max": max_tail,
            "max_probability_mass_error": max_mass_error,
            "max_logit_probability_ratio_error": max_ratio_error,
        },
        "manifest_fingerprint": fingerprint.hexdigest(),
        "manifests": manifest_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
