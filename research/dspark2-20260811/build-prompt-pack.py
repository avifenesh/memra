#!/usr/bin/env python3
"""Build the frozen 30K DSpark prompt mix without copying teacher responses."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from collections import deque
from pathlib import Path

DATASET = "mlabonne/open-perfectblend"
DATASET_REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"
SEED = 20260811
MODE_ORDER = ("think", "nothink")
CATEGORY_ORDER = ("chat", "math", "code", "if")
QUOTAS = {
    "chat": 5_280,
    "math": 11_820,
    "code": 11_670,
    "if": 1_230,
}

BUILTIN_PROMPTS = (
    ("code", "Write a Python function that parses an ISO-8601 timestamp string and returns the number of seconds since the Unix epoch, handling timezone offsets correctly. Include error handling and a few unit tests."),
    ("code", "Refactor a Rust config loader that uses unwrap into idiomatic error-propagating code. Explain each change briefly."),
    ("code", "Implement a thread-safe LRU cache in C++ with get and put in O(1), and explain the iterator-invalidation pitfall."),
    ("chat", "Explain the difference between TCP and UDP to someone who knows basic networking, with one concrete example for each."),
    ("chat", "I have chicken thighs, rice, onions, and soy sauce. Suggest a simple dinner I can cook in 30 minutes, with steps."),
    ("code", "A repository test expects HTTP 200 but receives 404. Give a concrete diagnostic runbook with the commands you would run."),
    ("if", "Plan a zero-downtime PostgreSQL migration that renames a column in a 50-million-row table. Include rollback points."),
    ("math", "A train leaves city A at 09:00 at 80 km/h. Another leaves city B, 240 km away, at 09:30 toward A at 100 km/h. When do they meet? Show the algebra."),
    ("math", "Three friends split an 87.50 restaurant bill plus a 15 percent tip. One dish cost 12 more than each of the other equal dishes. Compute each fair share."),
    ("if", "Write a 150-word release note for a CLI that speeds up model inference, aimed at developers, and include one command example."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=sum(QUOTAS.values()))
    parser.add_argument("--agentic-dir", type=Path)
    return parser.parse_args()


def category_for(source: str, prompt: str) -> str | None:
    lower_source = source.lower()
    lower_prompt = prompt.lower()
    if "autoif" in lower_source:
        return "if"
    if "metamath" in lower_source or "orca-math" in lower_source:
        return "math"
    if "evol-code" in lower_source:
        return "code"
    if "ultrachat" in lower_source or "ultrafeedback" in lower_source or "lmsys" in lower_source:
        return "chat"
    if "ultrainteract" in lower_source:
        code_markers = (
            "write code",
            "python",
            "c++",
            "java",
            "rust",
            "algorithm",
            "input\n",
            "output\n",
            "implement",
            "function",
        )
        return "code" if any(marker in lower_prompt for marker in code_markers) else "math"
    return None


def first_user_prompt(row: dict) -> str | None:
    for message in row.get("conversations", []):
        if message.get("from") == "human":
            prompt = str(message.get("value", "")).strip()
            if prompt:
                return prompt
    return None


def prompt_key(prompt: str) -> str:
    normalized = " ".join(prompt.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assignment_score(prompt: str, purpose: str) -> str:
    material = f"{SEED}\0{purpose}\0{prompt_key(prompt)}".encode()
    return hashlib.sha256(material).hexdigest()


def assign_mode_and_split(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Assign exact, independently stratified mode/split cells without id-period aliases."""
    assigned = [dict(row) for row in rows]
    category_indices: dict[str, list[int]] = {category: [] for category in CATEGORY_ORDER}
    for index, row in enumerate(assigned):
        category_indices[row["category"]].append(index)

    for category in CATEGORY_ORDER:
        indices = sorted(
            category_indices[category],
            key=lambda index: (
                assignment_score(assigned[index]["prompt"], "mode"),
                prompt_key(assigned[index]["prompt"]),
            ),
        )
        think_count = (len(indices) + 1) // 2
        for rank, index in enumerate(indices):
            assigned[index]["mode"] = "think" if rank < think_count else "nothink"

    cells: dict[tuple[str, str], list[int]] = {
        (category, mode): [] for category in CATEGORY_ORDER for mode in MODE_ORDER
    }
    for index, row in enumerate(assigned):
        cells[(row["category"], row["mode"])].append(index)

    heldout_total = round(len(assigned) * 0.05)
    raw_quotas = {
        cell: len(indices) * heldout_total / len(assigned) for cell, indices in cells.items()
    }
    heldout_quotas = {cell: int(value) for cell, value in raw_quotas.items()}
    remaining = heldout_total - sum(heldout_quotas.values())
    category_rank = {category: rank for rank, category in enumerate(CATEGORY_ORDER)}
    mode_rank = {mode: rank for rank, mode in enumerate(MODE_ORDER)}
    priority = sorted(
        cells,
        key=lambda cell: (
            -(raw_quotas[cell] - heldout_quotas[cell]),
            category_rank[cell[0]],
            mode_rank[cell[1]],
        ),
    )
    for cell in priority[:remaining]:
        heldout_quotas[cell] += 1

    for row in assigned:
        row["split"] = "train"
    for cell, indices in cells.items():
        ranked = sorted(
            indices,
            key=lambda index: (
                assignment_score(assigned[index]["prompt"], "split"),
                prompt_key(assigned[index]["prompt"]),
            ),
        )
        for index in ranked[: heldout_quotas[cell]]:
            assigned[index]["split"] = "heldout"

    counts: collections.Counter[str] = collections.Counter()
    for row in assigned:
        counts[f"{row['category']}/{row['mode']}/{row['split']}"] += 1
    return assigned, dict(sorted(counts.items()))


def iter_agentic_prompts(root: Path | None):
    if root is None or not root.is_dir():
        return
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("outcome", {}).get("verified", False):
                continue
            for message in row.get("messages", []):
                if message.get("role") == "user" and str(message.get("content", "")).strip():
                    yield str(message["content"]).strip(), path.name
                    break


def scaled_quotas(limit: int) -> dict[str, int]:
    if limit == sum(QUOTAS.values()):
        return dict(QUOTAS)
    raw = {key: limit * value / sum(QUOTAS.values()) for key, value in QUOTAS.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(result, key=lambda item: raw[item] - result[item], reverse=True):
        if sum(result.values()) == limit:
            break
        result[key] += 1
    return result


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    args = parse_args()
    if args.limit <= 0 or args.limit > sum(QUOTAS.values()):
        raise SystemExit(f"--limit must be in [1, {sum(QUOTAS.values())}]")
    quotas = scaled_quotas(args.limit)
    selected: list[dict] = []
    counts = {key: 0 for key in quotas}
    seen: set[str] = set()

    def admit(prompt: str, category: str, source: str) -> bool:
        if counts[category] >= quotas[category]:
            return False
        key = prompt_key(prompt)
        if key in seen:
            return False
        seen.add(key)
        selected.append({"prompt": prompt, "category": category, "source": source})
        counts[category] += 1
        return True

    for category, prompt in BUILTIN_PROMPTS:
        admit(prompt, category, "frspec-builtin")

    agentic_count = 0
    for prompt, source_name in iter_agentic_prompts(args.agentic_dir) or ():
        if admit(prompt, "code", f"sft-verified:{source_name}"):
            agentic_count += 1

    stream = load_dataset(DATASET, revision=DATASET_REVISION, split="train", streaming=True)
    scanned = 0
    for row in stream:
        scanned += 1
        prompt = first_user_prompt(row)
        if prompt is None:
            continue
        category = category_for(str(row.get("source", "")), prompt)
        if category is None:
            continue
        admit(prompt, category, str(row.get("source", "unknown")))
        if counts == quotas:
            break
    if counts != quotas:
        raise RuntimeError(f"prompt quotas unfilled after {scanned} rows: {counts} != {quotas}")

    # Stable category interleave prevents long generation chunks from being domain-monocultures.
    category_rows = {
        category: deque(row for row in selected if row["category"] == category)
        for category in quotas
    }
    ordered: list[dict] = []
    while len(ordered) < args.limit:
        for category in CATEGORY_ORDER:
            rows = category_rows[category]
            if rows:
                ordered.append(rows.popleft())
    ordered, assignment_cells = assign_mode_and_split(ordered)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_pack = args.output_dir / "prompts.promptpack"
    metadata_path = args.output_dir / "prompts.tsv"
    with prompt_pack.open("wb") as pack, metadata_path.open("w", encoding="utf-8") as meta:
        meta.write("id\tsplit\tmode\tcategory\tsource\tprompt_sha256\n")
        for idx, row in enumerate(ordered):
            prompt = row["prompt"].encode("utf-8")
            if b"\0" in prompt:
                raise ValueError(f"prompt {idx} contains NUL")
            pack.write(prompt)
            pack.write(b"\0")
            split = row["split"]
            mode = row["mode"]
            source = row["source"].replace("\t", " ").replace("\n", " ")
            meta.write(
                f"{idx}\t{split}\t{mode}\t{row['category']}\t{source}\t{prompt_key(row['prompt'])}\n"
            )

    revision = HfApi().dataset_info(DATASET, revision=DATASET_REVISION).sha
    if revision != DATASET_REVISION:
        raise RuntimeError(f"dataset revision drift: expected {DATASET_REVISION}, got {revision}")
    split_counts = collections.Counter(row["split"] for row in ordered)
    mode_counts = collections.Counter(row["mode"] for row in ordered)
    summary = {
        "assignment_cells": assignment_cells,
        "assignment_version": "stratified-hash-v2",
        "agentic_verified_prompts": agentic_count,
        "counts": counts,
        "dataset": DATASET,
        "dataset_revision": revision,
        "heldout": split_counts["heldout"],
        "limit": args.limit,
        "prompt_pack_sha256": hashlib.sha256(prompt_pack.read_bytes()).hexdigest(),
        "prompt_tsv_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "scanned_source_rows": scanned,
        "seed": SEED,
        "mode_counts": dict(sorted(mode_counts.items())),
        "train": split_counts["train"],
    }
    (args.output_dir / "prompt-pack-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
