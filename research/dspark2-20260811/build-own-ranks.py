#!/usr/bin/env python3
"""Build the frozen DSpark d2t vocabulary from exact target-generation token tapes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

VOCAB_SIZE = 248_320
DEFAULT_SIZE = 32_768
# Frozen Qwen3.5 chat/control ids present in the target's existing own-generation ranking.
FROZEN_SPECIAL_IDS = (248_044, 248_045, 248_046, 248_068, 248_069)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pairs(
    paths: list[Path],
) -> tuple[collections.Counter[int], dict[str, collections.Counter[int]], set[int], dict]:
    ranking_frequency: collections.Counter[int] = collections.Counter()
    split_frequency: dict[str, collections.Counter[int]] = {
        "train": collections.Counter(),
        "heldout": collections.Counter(),
    }
    high_ids: set[int] = set()
    all_response_ids: set[int] = set()
    seen_pairs: set[int] = set()
    response_tokens = 0
    split_pairs: collections.Counter[str] = collections.Counter()

    for path in paths:
        lines = path.read_text().splitlines()
        if not lines or lines[0] != "# memra-dspark-pairs-v1":
            raise ValueError(f"{path}: unexpected or missing pairs header")
        for line_number, line in enumerate(lines[1:], 2):
            fields = line.split("\t", 7)
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TSV fields")
            pair_id = int(fields[0])
            split = fields[1]
            prompt_len = int(fields[4])
            response_len = int(fields[5])
            total_len = int(fields[6])
            tokens = [int(value) for value in fields[7].split()]
            if pair_id in seen_pairs:
                raise ValueError(f"duplicate pair id {pair_id} in {path}:{line_number}")
            if split not in split_frequency:
                raise ValueError(f"{path}:{line_number}: invalid split {split!r}")
            if len(tokens) != total_len or prompt_len + response_len != total_len:
                raise ValueError(f"{path}:{line_number}: token length mismatch")
            if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
                raise ValueError(f"{path}:{line_number}: token outside target vocabulary")

            response = tokens[prompt_len:]
            split_frequency[split].update(response)
            all_response_ids.update(response)
            if split == "train":
                ranking_frequency.update(response)
                high_ids.update(token for token in response if token >= 248_000)
            response_tokens += len(response)
            split_pairs[split] += 1
            seen_pairs.add(pair_id)

    if not seen_pairs or response_tokens == 0:
        raise ValueError("no generated response tokens found")
    stats = {
        "pairs": len(seen_pairs),
        "pair_id_min": min(seen_pairs),
        "pair_id_max": max(seen_pairs),
        "split_pairs": dict(sorted(split_pairs.items())),
        "response_tokens": response_tokens,
        "distinct_response_ids": len(all_response_ids),
        "ranking_split": "train",
        "ranking_response_tokens": sum(split_frequency["train"].values()),
        "ranking_distinct_response_ids": len(ranking_frequency),
    }
    return ranking_frequency, split_frequency, high_ids, stats


def load_backfill(path: Path | None) -> list[int]:
    if path is None:
        return []
    ids = [int(value) for value in path.read_text().split()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids in backfill ranking")
    if any(token < 0 or token >= VOCAB_SIZE for token in ids):
        raise ValueError(f"{path}: id outside target vocabulary")
    return ids


def select_ids(
    frequency: collections.Counter[int], high_ids: set[int], backfill: list[int], size: int
) -> tuple[list[int], int]:
    forced = list(FROZEN_SPECIAL_IDS) + sorted(high_ids - set(FROZEN_SPECIAL_IDS))
    ranked = sorted(frequency, key=lambda token: (-frequency[token], token))
    selected: list[int] = []
    selected_set: set[int] = set()
    backfill_used = 0
    for token in forced + ranked:
        if token not in selected_set:
            selected.append(token)
            selected_set.add(token)
        if len(selected) == size:
            return selected, backfill_used
    for token in backfill:
        if token not in selected_set:
            selected.append(token)
            selected_set.add(token)
            backfill_used += 1
        if len(selected) == size:
            return selected, backfill_used
    raise ValueError(
        f"only {len(selected)} unique own-generation/backfill ids are available; need {size}"
    )


def coverage(counter: collections.Counter[int], selected: set[int]) -> float:
    total = sum(counter.values())
    return sum(count for token, count in counter.items() if token in selected) / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--backfill", type=Path)
    args = parser.parse_args()
    if not 1 <= args.size <= VOCAB_SIZE:
        raise SystemExit(f"--size must be in 1..{VOCAB_SIZE}")

    paths = sorted({path.resolve() for path in args.pairs})
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise SystemExit(f"missing pairs files: {missing}")
    frequency, split_frequency, high_ids, stats = parse_pairs(paths)
    backfill = load_backfill(args.backfill)
    selected, backfill_used = select_ids(frequency, high_ids, backfill, args.size)
    selected_set = set(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"{token}\n" for token in selected))
    summary = {
        "format": "memra-dspark-d2t-v1",
        "target_vocab_size": VOCAB_SIZE,
        "draft_vocab_size": len(selected),
        "forced_special_ids": list(FROZEN_SPECIAL_IDS),
        "observed_high_ids": sorted(high_ids),
        "backfill": str(args.backfill.resolve()) if args.backfill else None,
        "backfill_sha256": sha256(args.backfill) if args.backfill else None,
        "backfill_ids_used": backfill_used,
        "coverage": {
            split: coverage(counter, selected_set)
            for split, counter in sorted(split_frequency.items())
        },
        "pairs_files": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "ranks_sha256": sha256(args.out),
        **stats,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
