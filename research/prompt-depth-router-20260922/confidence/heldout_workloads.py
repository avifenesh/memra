"""Frozen, disjoint Qwen code/prose prompts for the conditional C follow-up."""

import argparse
import json
from pathlib import Path
import sys

PREFIX = Path(__file__).resolve().parents[1] / "prefix"
sys.path.insert(0, str(PREFIX))
from workloads import Counter, LENGTHS, digest, reference  # noqa: E402
from workloads_simple import instruction, make_cell  # noqa: E402


SCENARIOS = (
    (
        "dictionary utilities",
        "get_or_default(data: dict[str, int], key: str, default: int) -> int reads a key; "
        "count_keys(data: dict[str, int]) -> int counts keys; "
        "merged_copy(left: dict[str, int], right: dict[str, int]) -> dict[str, int] combines copies; "
        "invert_unique(data: dict[str, int]) -> dict[int, str] inverts unique values",
        "read a dictionary with a default, count keys, merge two copies, and invert unique values",
    ),
    (
        "sequence summaries",
        "minimum_or_none(values: list[float]) -> float | None finds a minimum or None; "
        "maximum_or_none(values: list[float]) -> float | None finds a maximum or None; "
        "mean_or_zero(values: list[float]) -> float computes a mean or zero; "
        "count_above(values: list[float], threshold: float) -> int counts values above a threshold",
        "find the minimum and maximum of a list, compute a mean, and count values above a threshold",
    ),
    (
        "coordinate utilities",
        "translate_point(point: tuple[int, int], dx: int, dy: int) -> tuple[int, int] shifts a point; "
        "midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float] finds a midpoint; "
        "manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int measures grid distance; "
        "in_rectangle(point: tuple[int, int], low: tuple[int, int], high: tuple[int, int]) -> bool checks bounds",
        "translate a point, find a midpoint, measure grid distance, and check rectangular bounds",
    ),
    (
        "text cleaning",
        "normalize_space(text: str) -> str collapses whitespace; "
        "initials(text: str) -> str takes first letters of words; "
        "nonempty_lines(text: str) -> list[str] keeps stripped nonempty lines; "
        "trim_prefix(text: str, prefix: str) -> str removes a matching leading prefix",
        "normalize whitespace, take initials, collect nonempty lines, and remove a matching prefix",
    ),
    (
        "clock utilities",
        "minutes_to_hours_parts(minutes: int) -> tuple[int, int] splits minutes; "
        "seconds_to_minutes_parts(seconds: int) -> tuple[int, int] splits seconds; "
        "clamp_hour(hour: int) -> int clamps a value to 0 through 23; "
        "format_hhmm(hour: int, minute: int) -> str formats a two-part clock",
        "split durations, clamp an hour, and format an hour and minute",
    ),
    (
        "set utilities",
        "union_copy(left: set[int], right: set[int]) -> set[int] unions two sets; "
        "intersection_copy(left: set[int], right: set[int]) -> set[int] intersects them; "
        "only_left(left: set[int], right: set[int]) -> set[int] removes shared values; "
        "is_subset(left: set[int], right: set[int]) -> bool tests containment",
        "form a union and intersection, take only-left values, and check subset containment",
    ),
)
QUALIFICATION = (
    "boolean helpers",
    "xor_bool(a: bool, b: bool) -> bool returns exclusive-or; "
    "nand_bool(a: bool, b: bool) -> bool returns not-both; "
    "all_true(values: list[bool]) -> bool tests whether every value is true; "
    "any_true(values: list[bool]) -> bool tests whether at least one value is true",
    "compute exclusive-or and not-both, then test whether all or any booleans are true",
)


def build(binary, model, out):
    out.mkdir(parents=True, exist_ok=False)
    counter = Counter(binary, model, out / "tokenizer.stderr")
    try:
        scenarios = {}
        for index, spec in enumerate(SCENARIOS):
            context = reference(995000 + index, spec[0])
            order = [(length, kind) for length in LENGTHS for kind in ("prose", "code")]
            shift = index % len(order)
            order = order[shift:] + order[:shift]
            if index % 2:
                order.reverse()
            prompts, cells = [], []
            for length, kind in order:
                prompt, cell = make_cell(
                    counter, instruction(spec, kind), context, length, kind
                )
                cell.pop("predictions", None)  # This follow-up uses fixed depths only.
                prompts.append(prompt)
                cells.append(cell)
            path = out / f"qwen-c-heldout-scenario-{index}.txt"
            path.write_text("\n---TURN---\n".join(prompts) + "\n")
            scenarios[str(index)] = {
                "file": path.name,
                "sha256": digest(path),
                "topic": spec[0],
                "cells": cells,
                "seed": 20760000 + index,
            }
        context = reference(996000, QUALIFICATION[0])
        prompts, cells = [], []
        for length in LENGTHS:
            for kind in ("prose", "code"):
            prompt, cell = make_cell(
                counter, instruction(QUALIFICATION, kind), context, length, kind
            )
            cell.pop("predictions", None)
            prompts.append(prompt)
                cells.append(cell)
        path = out / "qwen-c-heldout-qualification.txt"
        path.write_text("\n---TURN---\n".join(prompts) + "\n")
        qualification = {
            "file": path.name,
            "sha256": digest(path),
            "cells": cells,
            "seed": 20760010,
        }
    finally:
        counter.close()
    manifest = {
        "schema": 1,
        "scope": "disjoint simple Qwen code/prose requests for selected C validation",
        "generator_sha256": digest(Path(__file__)),
        "prefix_helper_sha256": digest(PREFIX / "workloads.py"),
        "prompt_helper_sha256": digest(PREFIX / "workloads_simple.py"),
        "binary_sha256": digest(binary),
        "length_targets": LENGTHS,
        "scenarios": scenarios,
        "qualification": qualification,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "heldout-prompts-registered", "scenarios": len(scenarios)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.binary, args.model, args.out)


if __name__ == "__main__":
    main()
