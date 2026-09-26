"""Versioned prompt-length corpus with prose and code that can reach a final answer."""

import argparse
import hashlib
import json
from pathlib import Path

from workloads import Counter, LENGTHS, digest, reference, sized_prompt


SCENARIOS = (
    (
        "string helpers",
        "reverse_text(text: str) -> str reverses text; "
        "count_vowels(text: str) -> int counts English vowels; "
        "first_word(text: str) -> str returns the first word or an empty string; "
        "text_length(text: str) -> int returns its length",
        "reverse text, count vowels, find the first word, and measure a string's length",
    ),
    (
        "integer helpers",
        "double_number(n: int) -> int doubles n; "
        "square_number(n: int) -> int squares n; "
        "negate_number(n: int) -> int negates n; "
        "absolute_number(n: int) -> int returns its absolute value",
        "double, square, negate, and take the absolute value of an integer",
    ),
    (
        "integer predicates",
        "is_odd(n: int) -> bool checks whether n is odd; "
        "is_positive(n: int) -> bool checks whether n is positive; "
        "is_negative(n: int) -> bool checks whether n is negative; "
        "is_zero(n: int) -> bool checks whether n is zero",
        "check whether an integer is odd, positive, negative, or zero",
    ),
    (
        "list helpers",
        "sum_numbers(values: list[int]) -> int adds the values; "
        "count_items(values: list[int]) -> int counts them; "
        "first_or_zero(values: list[int]) -> int returns the first value or zero; "
        "last_or_zero(values: list[int]) -> int returns the last value or zero",
        "sum and count list items, then choose the first or last item",
    ),
    (
        "geometry helpers",
        "rectangle_area(width: float, height: float) -> float gives its area; "
        "rectangle_perimeter(width: float, height: float) -> float gives its perimeter; "
        "square_area(side: float) -> float gives its area; "
        "square_perimeter(side: float) -> float gives its perimeter",
        "calculate the area and perimeter of rectangles and squares",
    ),
    (
        "unit conversions",
        "celsius_to_fahrenheit(value: float) -> float converts temperature; "
        "fahrenheit_to_celsius(value: float) -> float converts it back; "
        "meters_to_centimeters(value: float) -> float converts distance; "
        "minutes_to_seconds(value: float) -> float converts time",
        "convert temperatures, distances, and times between common units",
    ),
)

QUALIFICATION = (
    "simple arithmetic utilities",
    "add_numbers(a: int, b: int) -> int returns their sum; "
    "subtract_numbers(a: int, b: int) -> int returns their difference; "
    "multiply_numbers(a: int, b: int) -> int returns their product; "
    "is_even(n: int) -> bool reports whether n is even",
    "add, subtract, and multiply integers, then check whether one is even",
)


def instruction(specification, kind):
    topic, functions, prose_description = specification
    if kind == "code":
        return (
            f"Write a Python module for {topic} with four simple functions: "
            f"{functions}. Use straightforward implementations and type hints. "
            "Return one fenced Python code block only, without a prose explanation."
        )
    return (
        f"Explain in ordinary language how to {prose_description}. "
        "Give one worked example and one edge case for each. Write at least "
        "three short paragraphs, without syntax blocks or tables."
    )


def make_cell(counter, task, context, length, kind):
    prompt, info = sized_prompt(counter, task, context, length)
    return prompt, {
        "kind": kind,
        "length_target": length,
        **info,
        "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def build(models, binaries, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 2,
        "generator_sha256": digest(Path(__file__)),
        "helper_sha256": digest(Path(__file__).with_name("workloads.py")),
        "length_targets": LENGTHS,
        "prefix_budgets": (64, 128, 256),
        "scope": "simple-helper code/prose; independent requests; separate from failed complex-contract qualification",
        "families": {},
    }
    for family in ("qwen", "gemma"):
        binary = binaries / f"{family}-prefix-study"
        counter = Counter(binary, models / family / "target.gguf", out / f"{family}-tokenizer.stderr")
        try:
            scenarios = {}
            for index, specification in enumerate(SCENARIOS):
                context = reference(980000 + index, specification[0])
                order = [(length, kind) for length in LENGTHS for kind in ("prose", "code")]
                shift = index % len(order)
                order = order[shift:] + order[:shift]
                if index % 2:
                    order.reverse()
                prompts = []
                cells = []
                for length, kind in order:
                    prompt, cell = make_cell(
                        counter, instruction(specification, kind), context, length, kind
                    )
                    prompts.append(prompt)
                    cells.append(cell)
                path = out / f"{family}-simple-scenario-{index}.txt"
                path.write_text("\n---TURN---\n".join(prompts) + "\n")
                scenarios[str(index)] = {
                    "file": path.name,
                    "sha256": digest(path),
                    "topic": specification[0],
                    "cells": cells,
                    "seed": (20740000 if family == "qwen" else 20750000) + index,
                }
            context = reference(990000, QUALIFICATION[0])
            prompts = []
            cells = []
            for length in LENGTHS:
                for kind in ("prose", "code"):
                    prompt, cell = make_cell(
                        counter, instruction(QUALIFICATION, kind), context, length, kind
                    )
                    prompts.append(prompt)
                    cells.append(cell)
            path = out / f"{family}-simple-qualification.txt"
            path.write_text("\n---TURN---\n".join(prompts) + "\n")
            qualification = {
                "file": path.name,
                "sha256": digest(path),
                "cells": cells,
                "seed": 20740010 if family == "qwen" else 20750010,
            }
            robustness = []
            context = reference(990001, SCENARIOS[0][0])
            for length in LENGTHS:
                for kind in ("prose", "code"):
                    # The task follows the long reference in this separate cell.
                    prompt, info = sized_prompt(
                        counter, instruction(SCENARIOS[0], kind), context, length, late=True
                    )
                    path = out / f"{family}-simple-late-{kind}-{length}.txt"
                    path.write_text(prompt)
                    robustness.append(
                        {
                            "file": path.name,
                            "sha256": digest(path),
                            "kind": kind,
                            "length_target": length,
                            **info,
                            "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        }
                    )
            manifest["families"][family] = {
                "binary_sha256": digest(binary),
                "scenarios": scenarios,
                "late_task_robustness": robustness,
                "qualification": qualification,
            }
        finally:
            counter.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("models", "binaries", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.models, args.binaries, args.out), indent=2))
