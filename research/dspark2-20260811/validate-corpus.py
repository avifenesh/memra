#!/usr/bin/env python3
"""Fail-closed validation for one compact DSpark extraction chunk."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


VOCAB_SIZE = 248_320


def exact_memmap(path: Path, dtype: str, count: int) -> np.memmap:
    itemsize = np.dtype(dtype).itemsize
    expected = count * itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path}: {actual} bytes, expected {expected}")
    return np.memmap(path, dtype=dtype, mode="r", shape=(count,))


def validate(root: Path) -> dict:
    meta = json.loads((root / "extraction.meta.json").read_text())
    if meta.get("format") != "memra-dspark-anchors-v1":
        raise ValueError("unexpected extraction format")
    records = int(meta["records"])
    pairs = int(meta["pairs"])
    hidden_size = int(meta["hidden_size"])
    gamma = int(meta["gamma"])
    top_k = int(meta["top_k"])
    temperature = float(meta["temperature"])
    anchors_per_pair = int(meta["anchors_per_pair"])
    if not records or not pairs:
        raise ValueError("empty extraction")
    if (hidden_size, gamma, top_k, temperature) != (4096, 5, 64, 0.7):
        raise ValueError(
            f"non-frozen extraction shape: hidden={hidden_size} gamma={gamma} "
            f"top_k={top_k} temperature={temperature}"
        )

    hidden = exact_memmap(root / "hiddens.bf16", "<u2", records * hidden_size)
    tokens = exact_memmap(root / "tokens.u32", "<u4", records * (gamma + 1)).reshape(
        records, gamma + 1
    )
    top_ids = exact_memmap(
        root / "top_ids.u32", "<u4", records * gamma * top_k
    ).reshape(records, gamma, top_k)
    top_logits = exact_memmap(
        root / "top_logits.f32", "<f4", records * gamma * top_k
    ).reshape(records, gamma, top_k)
    top_probs = exact_memmap(
        root / "top_probs.f32", "<f4", records * gamma * top_k
    ).reshape(records, gamma, top_k)
    tail = exact_memmap(root / "tail_probs.f32", "<f4", records * gamma).reshape(
        records, gamma
    )

    if np.any((hidden & np.uint16(0x7F80)) == np.uint16(0x7F80)):
        raise ValueError("hiddens contain non-finite BF16 values")
    if np.any(tokens >= VOCAB_SIZE) or np.any(top_ids >= VOCAB_SIZE):
        raise ValueError("token id outside target vocabulary")
    if not np.all(np.isfinite(top_logits)):
        raise ValueError("target logits contain non-finite values")
    if not np.all(np.isfinite(top_probs)) or np.any(top_probs <= 0.0):
        raise ValueError("target top probabilities must be finite and positive")
    if not np.all(np.isfinite(tail)) or np.any((tail < 0.0) | (tail > 1.0)):
        raise ValueError("target tail probabilities must be finite in [0,1]")

    mass = top_probs.sum(axis=-1, dtype=np.float64) + tail
    max_mass_error = float(np.max(np.abs(mass - 1.0)))
    if max_mass_error > 2.0e-5:
        raise ValueError(f"target probability mass error {max_mass_error}")
    if np.any(top_logits[..., 1:] > top_logits[..., :-1]):
        raise ValueError("top logits are not monotonically descending")
    if np.any(top_probs[..., 1:] > top_probs[..., :-1]):
        raise ValueError("top probabilities are not monotonically descending")
    if np.any(np.sort(top_ids, axis=-1)[..., 1:] == np.sort(top_ids, axis=-1)[..., :-1]):
        raise ValueError("duplicate target id within a top-k row")

    ratio_error = np.abs(
        np.log(top_probs[..., :1].astype(np.float64) / top_probs.astype(np.float64))
        - (top_logits[..., :1] - top_logits).astype(np.float64) / temperature
    )
    max_ratio_error = float(np.max(ratio_error))
    if max_ratio_error > 3.0e-4:
        raise ValueError(f"logit/probability temperature mismatch {max_ratio_error}")

    index_lines = (root / "index.tsv").read_text().splitlines()
    expected_header = "record\tpair_id\tanchor_pos\tprompt_len\tsplit\tmode\tcategory"
    if not index_lines or index_lines[0] != expected_header:
        raise ValueError("unexpected index header")
    if len(index_lines) != records + 1:
        raise ValueError(f"index has {len(index_lines) - 1} rows, expected {records}")
    pair_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for expected_record, line in enumerate(index_lines[1:]):
        fields = line.split("\t")
        if len(fields) != 7 or int(fields[0]) != expected_record:
            raise ValueError(f"malformed index record {expected_record}")
        pair_counts[int(fields[1])] += 1
        split_counts[fields[4]] += 1
        if fields[4] not in {"train", "heldout"} or fields[5] not in {"think", "nothink"}:
            raise ValueError(f"invalid split/mode in index record {expected_record}")
    if len(pair_counts) + int(meta["skipped_short"]) != pairs:
        raise ValueError("indexed plus skipped pair count does not match metadata")
    if any(count > anchors_per_pair for count in pair_counts.values()):
        raise ValueError("pair produced more than the frozen anchors-per-pair")

    sampled = tokens[:, 1:]
    sampled_is_top64 = np.any(top_ids == sampled[..., None], axis=-1)
    result = {
        "format": "memra-dspark-validation-v1",
        "records": records,
        "pairs": pairs,
        "split_records": dict(sorted(split_counts.items())),
        "sampled_token_top64_rate": float(np.mean(sampled_is_top64)),
        "tail_mass_mean": float(np.mean(tail, dtype=np.float64)),
        "tail_mass_max": float(np.max(tail)),
        "max_probability_mass_error": max_mass_error,
        "max_logit_probability_ratio_error": max_ratio_error,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = validate(args.extracted)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        args.receipt.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
