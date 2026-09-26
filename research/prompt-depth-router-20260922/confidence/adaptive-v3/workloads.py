"""Freeze disjoint eight-turn code conversations without reading model output."""

import argparse
import hashlib
import json
from pathlib import Path
import random


QUALIFICATION = (
    "integer rectangles",
    (
        ("rect_area", "width: int, height: int -> int", "Return width times height."),
        ("rect_perimeter", "width: int, height: int -> int", "Return twice the sum of width and height."),
        ("is_square", "width: int, height: int -> bool", "Return whether width equals height."),
        ("scale_rect", "width: int, height: int, factor: int -> tuple[int, int]", "Multiply both dimensions by factor."),
        ("rect_fits", "width: int, height: int, max_width: int, max_height: int -> bool", "Return whether both dimensions fit their limits."),
        ("swap_rect", "width: int, height: int -> tuple[int, int]", "Return the dimensions in reverse order."),
        ("rect_diagonal_squared", "width: int, height: int -> int", "Return width squared plus height squared."),
        ("rect_border_cells", "width: int, height: int -> int", "For positive dimensions, count integer grid cells on the border without double counting corners."),
    ),
)

CALIBRATION = (
    (
        "set operations",
        (
            ("insert_value", "values: set[int], value: int -> set[int]", "Return a new set with value added."),
            ("remove_value", "values: set[int], value: int -> set[int]", "Return a new set with value removed if present."),
            ("union_values", "left: set[int], right: set[int] -> set[int]", "Return the set union."),
            ("common_values", "left: set[int], right: set[int] -> set[int]", "Return the set intersection."),
            ("only_left", "left: set[int], right: set[int] -> set[int]", "Return values in left but not right."),
            ("symmetric_values", "left: set[int], right: set[int] -> set[int]", "Return values in exactly one input."),
            ("is_subset", "left: set[int], right: set[int] -> bool", "Return whether left is a subset of right."),
            ("sorted_values", "values: set[int] -> list[int]", "Return all values in ascending order."),
        ),
    ),
    (
        "index operations",
        (
            ("last_index", "values: list[int] -> int | None", "Return the final valid index or None for an empty list."),
            ("valid_index", "values: list[int], index: int -> bool", "Return whether index is a nonnegative valid index."),
            ("value_at", "values: list[int], index: int -> int | None", "Return the value at a nonnegative valid index, else None."),
            ("indices_of", "values: list[int], target: int -> list[int]", "Return every index holding target."),
            ("index_pairs", "values: list[int] -> list[tuple[int, int]]", "Return index and value pairs in order."),
            ("even_indices", "values: list[int] -> list[int]", "Return values at even indices."),
            ("odd_indices", "values: list[int] -> list[int]", "Return values at odd indices."),
            ("reverse_indices", "values: list[int] -> list[int]", "Return valid indices in descending order."),
        ),
    ),
    (
        "pair comparisons",
        (
            ("pair_min", "pair: tuple[int, int] -> int", "Return the smaller component."),
            ("pair_max", "pair: tuple[int, int] -> int", "Return the larger component."),
            ("pair_diff", "pair: tuple[int, int] -> int", "Return the first component minus the second."),
            ("pair_equal", "pair: tuple[int, int] -> bool", "Return whether both components match."),
            ("pair_sorted", "pair: tuple[int, int] -> tuple[int, int]", "Return the components in ascending order."),
            ("pair_product", "pair: tuple[int, int] -> int", "Return their product."),
            ("pair_nonnegative", "pair: tuple[int, int] -> bool", "Return whether both components are nonnegative."),
            ("pair_sum_abs", "pair: tuple[int, int] -> int", "Return the sum of their absolute values."),
        ),
    ),
)

HELDOUT = (
    (
        "bit flags",
        (
            ("set_bit", "value: int, bit: int -> int", "For a nonnegative bit index, set that bit."),
            ("clear_bit", "value: int, bit: int -> int", "For a nonnegative bit index, clear that bit."),
            ("has_bit", "value: int, bit: int -> bool", "For a nonnegative bit index, test that bit."),
            ("toggle_bit", "value: int, bit: int -> int", "For a nonnegative bit index, toggle that bit."),
            ("count_ones", "value: int -> int", "For nonnegative value, count its set bits."),
            ("is_power_of_two", "value: int -> bool", "Return true exactly for positive powers of two."),
            ("lowest_set_bit", "value: int -> int", "For nonnegative value, return its lowest set bit as a mask, or zero."),
            ("mask_below", "bit: int -> int", "For nonnegative bit, return a mask with all lower bits set."),
        ),
    ),
    (
        "closed intervals",
        (
            ("clamp_int", "value: int, low: int, high: int -> int", "Clamp value into a valid closed interval."),
            ("in_closed_range", "value: int, low: int, high: int -> bool", "Test membership in a valid closed interval."),
            ("range_size", "low: int, high: int -> int", "Count integers in a valid closed interval."),
            ("shift_range", "low: int, high: int, delta: int -> tuple[int, int]", "Shift both ends of a valid interval."),
            ("overlap_range", "a: tuple[int, int], b: tuple[int, int] -> bool", "Test whether valid closed intervals overlap."),
            ("contains_range", "outer: tuple[int, int], inner: tuple[int, int] -> bool", "Test whether one valid interval contains another."),
            ("intersect_range", "a: tuple[int, int], b: tuple[int, int] -> tuple[int, int] | None", "Return their closed intersection or None."),
            ("merge_touching", "a: tuple[int, int], b: tuple[int, int] -> tuple[int, int] | None", "Merge valid closed intervals if they overlap or touch, else None."),
        ),
    ),
    (
        "sequence slices",
        (
            ("take_prefix", "values: list[int], count: int -> list[int]", "Take at most nonnegative count leading values."),
            ("take_suffix", "values: list[int], count: int -> list[int]", "Take at most nonnegative count trailing values."),
            ("drop_prefix", "values: list[int], count: int -> list[int]", "Drop at most nonnegative count leading values."),
            ("drop_suffix", "values: list[int], count: int -> list[int]", "Drop at most nonnegative count trailing values."),
            ("rotate_left", "values: list[int], count: int -> list[int]", "Rotate left by count, returning an empty list for empty input."),
            ("rotate_right", "values: list[int], count: int -> list[int]", "Rotate right by count, returning an empty list for empty input."),
            ("every_other", "values: list[int] -> list[int]", "Return values at even indices."),
            ("adjacent_pairs", "values: list[int] -> list[tuple[int, int]]", "Return each consecutive pair."),
        ),
    ),
    (
        "mapping counts",
        (
            ("merge_counts", "left: dict[str, int], right: dict[str, int] -> dict[str, int]", "Add counts for shared keys without mutating inputs."),
            ("keys_sorted", "counts: dict[str, int] -> list[str]", "Return keys in ascending order."),
            ("filter_positive", "counts: dict[str, int] -> dict[str, int]", "Keep only strictly positive counts."),
            ("rename_key", "counts: dict[str, int], old: str, new: str -> dict[str, int]", "Return a copy moving old to new when old exists."),
            ("get_or_zero", "counts: dict[str, int], key: str -> int", "Return the key's count or zero."),
            ("increment_key", "counts: dict[str, int], key: str -> dict[str, int]", "Return a copy with that key incremented by one."),
            ("count_values", "counts: dict[str, int] -> int", "Return the sum of all counts."),
            ("invert_unique", "mapping: dict[str, int] -> dict[int, str]", "Swap keys and values, rejecting duplicate values."),
        ),
    ),
    (
        "whitespace text",
        (
            ("normalize_spaces", "text: str -> str", "Collapse every run of whitespace to one ordinary space and strip ends."),
            ("strip_prefix_once", "text: str, prefix: str -> str", "Remove prefix once if present."),
            ("strip_suffix_once", "text: str, suffix: str -> str", "Remove suffix once if present."),
            ("line_count", "text: str -> int", "Count lines in the same way as str.splitlines()."),
            ("first_line", "text: str -> str", "Return the first splitline or an empty string."),
            ("last_line", "text: str -> str", "Return the last splitline or an empty string."),
            ("nonempty_lines", "text: str -> list[str]", "Return nonempty stripped lines in order."),
            ("join_lines", "lines: list[str] -> str", "Join lines with newline characters."),
        ),
    ),
    (
        "integer arithmetic",
        (
            ("ceil_div", "value: int, divisor: int -> int", "For positive divisor, return mathematical ceiling division."),
            ("floor_div", "value: int, divisor: int -> int", "For positive divisor, return mathematical floor division."),
            ("divides", "divisor: int, value: int -> bool", "For nonzero divisor, test exact divisibility."),
            ("gcd_pair", "left: int, right: int -> int", "Return their nonnegative greatest common divisor."),
            ("lcm_pair", "left: int, right: int -> int", "Return their nonnegative least common multiple."),
            ("digit_sum", "value: int -> int", "Sum base-ten digits of the absolute value."),
            ("reverse_digits", "value: int -> int", "Reverse base-ten digits of a nonnegative integer."),
            ("signum", "value: int -> int", "Return minus one, zero or one according to the sign."),
        ),
    ),
)

PADDING = (4096, 256, 1024, 4096, 256, 1024, 4096, 256)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(seed, topic):
    rng = random.Random(seed)
    lines = []
    for index in range(400):
        lines.append(
            f"Example {index} for {topic}: item_{index} has value "
            f"{rng.randrange(10, 10000)} and revision {rng.randrange(1, 900)}. "
            "These are inert examples, not function requirements.\n"
        )
    return "".join(lines)


def conversation(specification, seed, initial_padding):
    topic, functions = specification
    source = reference(seed, topic)
    prompts, records = [], []
    for turn, (name, signature, meaning) in enumerate(functions, 1):
        padding = initial_padding if turn == 1 else PADDING[turn - 1]
        parameters, returns = signature.rsplit(" -> ", 1)
        prompt = (
            f"Write one Python function named {name} with signature "
            f"{name}({parameters}) -> {returns}. {meaning} "
            "Use type hints and a straightforward "
            "implementation. Return only one fenced Python code block, without "
            "a prose explanation. If an input is outside the stated domain, "
            "raise ValueError. Earlier turns are context, not a request to "
            "repeat earlier functions.\n\n"
            "<reference>\n" + source[:padding] + "\n</reference>"
        )
        prompts.append(prompt)
        records.append({
            "turn": turn, "function": name, "signature": signature,
            "padding_chars": padding,
            "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
    return "\n---TURN---\n".join(prompts) + "\n", records


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for role, entries, start in (
        ("qualification", (QUALIFICATION,), 20771000),
        ("calibration", CALIBRATION, 20772000),
        ("heldout", HELDOUT, 20773000),
    ):
        groups[role] = []
        for index, item in enumerate(entries):
            seed = start + index
            first_padding = 16384 if index % 2 else 4096
            text, records = conversation(item, seed, first_padding)
            path = out / f"{role}-{index}.txt"
            path.write_text(text)
            groups[role].append({
                "file": path.name, "sha256": digest(path),
                "seed": seed, "topic": item[0], "turns": records,
            })
    manifest = {
        "schema": 1, "generator_sha256": digest(Path(__file__)),
        "scope": "disjoint eight-turn code conversations; fixed source before native model output",
        "requested_format": "one fenced Python function per turn",
        "padding_chars_after_first": PADDING,
        "groups": groups,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.out)
    print(json.dumps({
        "schema": result["schema"],
        "counts": {role: len(rows) for role, rows in result["groups"].items()},
    }))


if __name__ == "__main__":
    main()
