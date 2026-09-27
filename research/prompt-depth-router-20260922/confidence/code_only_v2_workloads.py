"""Versioned code-only qualifier with six untouched held-out scenarios."""

import argparse
import json
from pathlib import Path
import shutil
import sys

PREFIX = Path(__file__).resolve().parents[1] / "prefix"
sys.path.insert(0, str(PREFIX))
from workloads import Counter, LENGTHS, digest, reference  # noqa: E402
from workloads_simple import instruction, make_cell  # noqa: E402


QUALIFICATION = (
    "tuple utilities",
    "swap_pair(pair: tuple[int, int]) -> tuple[int, int] swaps its components; "
    "first_or_none(values: tuple[int, ...]) -> int | None returns the first element or None; "
    "last_or_none(values: tuple[int, ...]) -> int | None returns the last element or None; "
    "pair_sum(pair: tuple[int, int]) -> int adds its two components",
    "swap a pair, choose the first or last tuple element, and add a pair's components",
)


def build(binary, model, original, out):
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((original / "manifest.json").read_text())
    if previous["schema"] != 1 or previous["binary_sha256"] != digest(binary):
        raise ValueError("original untouched scenarios have another source or binary")
    scenarios = {}
    for index in range(6):
        entry = previous["scenarios"][str(index)]
        if entry["seed"] != 20760000 + index:
            raise ValueError("original scenario seeds changed")
        source = original / entry["file"]
        if digest(source) != entry["sha256"]:
            raise ValueError("original held-out prompt changed")
        target = out / entry["file"]
        shutil.copyfile(source, target)
        scenarios[str(index)] = entry

    counter = Counter(binary, model, out / "tokenizer.stderr")
    try:
        context = reference(997000, QUALIFICATION[0])
        prompts, cells = [], []
        for length in LENGTHS:
            for kind in ("prose", "code"):
                prompt, cell = make_cell(
                    counter, instruction(QUALIFICATION, kind), context, length, kind
                )
                cell.pop("predictions", None)
                prompts.append(prompt)
                cells.append(cell)
    finally:
        counter.close()
    qualifier = out / "qwen-c-code-v2-qualification.txt"
    qualifier.write_text("\n---TURN---\n".join(prompts) + "\n")
    manifest = {
        "schema": 2,
        "scope": "code-only v2 qualification; original mixed scenario prompts unchanged",
        "generator_sha256": digest(Path(__file__)),
        "original_manifest_sha256": digest(original / "manifest.json"),
        "prefix_helper_sha256": digest(PREFIX / "workloads.py"),
        "prompt_helper_sha256": digest(PREFIX / "workloads_simple.py"),
        "binary_sha256": digest(binary),
        "length_targets": LENGTHS,
        "scenarios": scenarios,
        "qualification": {
            "file": qualifier.name,
            "sha256": digest(qualifier),
            "cells": cells,
            "seed": 20760020,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": "code-only-v2-prompts-registered",
        "scenarios": len(scenarios),
        "qualification_seed": 20760020,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.binary, args.model, args.original, args.out)


if __name__ == "__main__":
    main()
