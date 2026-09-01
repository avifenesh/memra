#!/usr/bin/env python3
"""Compare one or more memra-checkpoint-oracle-v1 files to a BF16 oracle."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def parse_oracle(path: Path) -> dict:
    metadata = {}
    logits = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if fields[0] == "logit" and len(fields) == 3:
            index = int(fields[1])
            if index in logits:
                raise ValueError(f"{path}: duplicate logit {index}")
            logits[index] = struct.unpack("<f", struct.pack("<I", int(fields[2], 16)))[0]
        elif len(fields) == 2:
            metadata[fields[0]] = fields[1]
    if metadata.get("format") != "memra-checkpoint-oracle-v1":
        raise ValueError(f"{path}: wrong or missing format")
    vocab = int(metadata["vocab"])
    if set(logits) != set(range(vocab)):
        raise ValueError(f"{path}: logits are not contiguous 0..{vocab - 1}")
    values = [logits[index] for index in range(vocab)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: non-finite logits")
    return {"metadata": metadata, "logits": values}


def stable_top(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def compare(reference: dict, candidate: dict, path: Path) -> dict:
    ref_meta, cand_meta = reference["metadata"], candidate["metadata"]
    if ref_meta["tokens"] != cand_meta["tokens"] or ref_meta["vocab"] != cand_meta["vocab"]:
        raise ValueError(f"{path}: token/vocab identity differs from reference")
    ref, cand = reference["logits"], candidate["logits"]
    errors = [got - want for want, got in zip(ref, cand)]
    abs_errors = [abs(value) for value in errors]
    rel_errors = [error / max(abs(want), 1e-6) for want, error in zip(ref, abs_errors)]
    dot = sum(left * right for left, right in zip(ref, cand))
    ref_norm = math.sqrt(sum(value * value for value in ref))
    cand_norm = math.sqrt(sum(value * value for value in cand))
    top20_ref = stable_top(ref, 20)
    top20_cand = stable_top(cand, 20)
    worst = max(range(len(abs_errors)), key=abs_errors.__getitem__)
    return {
        "candidate": str(path),
        "engine": cand_meta.get("engine"),
        "numeric_class": cand_meta.get("numeric_class"),
        "argmax": stable_top(cand, 1)[0],
        "reference_argmax": stable_top(ref, 1)[0],
        "argmax_match": stable_top(cand, 1)[0] == stable_top(ref, 1)[0],
        "top20_overlap": len(set(top20_ref) & set(top20_cand)),
        "max_abs": max(abs_errors),
        "max_abs_index": worst,
        "max_rel": max(rel_errors),
        "mean_abs": sum(abs_errors) / len(abs_errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "cosine": dot / (ref_norm * cand_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reference = parse_oracle(args.reference)
    report = {
        "format": "memra-checkpoint-oracle-comparison-v1",
        "reference": str(args.reference),
        "reference_engine": reference["metadata"].get("engine"),
        "reference_numeric_class": reference["metadata"].get("numeric_class"),
        "comparisons": [compare(reference, parse_oracle(path), path) for path in args.candidates],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
