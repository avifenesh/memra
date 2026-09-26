"""Freeze fresh Qwen code conversations before any joint-control GPU output."""

import argparse
import hashlib
import json
from pathlib import Path
import random


QUALIFICATION = (
    "score transforms",
    (
        ("increase_score", "score: int, delta: int -> int", "Return score plus delta."),
        ("double_score", "score: int -> int", "Return twice score."),
        ("score_is_even", "score: int -> bool", "Return whether score is even."),
        ("clip_score", "score: int, low: int, high: int -> int", "Clamp score into a valid inclusive interval."),
        ("score_gap", "left: int, right: int -> int", "Return the absolute difference."),
        ("larger_score", "left: int, right: int -> int", "Return the larger score."),
        ("nonnegative_score", "score: int -> int", "Return score if nonnegative, otherwise zero."),
        ("score_bucket", "score: int, width: int -> int", "For positive width, return the floor-division bucket index."),
    ),
)

HELDOUT = (
    (
        "graph adjacency",
        (
            ("graph_neighbors", "adjacency: dict[int, set[int]], node: int -> set[int]", "Return a new set of this node's neighbors, or an empty set if absent."),
            ("graph_degree", "adjacency: dict[int, set[int]], node: int -> int", "Return this node's number of neighbors, or zero if absent."),
            ("graph_has_edge", "adjacency: dict[int, set[int]], left: int, right: int -> bool", "Return whether right is in left's neighbor set."),
            ("graph_add_edge", "adjacency: dict[int, set[int]], left: int, right: int -> dict[int, set[int]]", "Return a deep copy with an undirected edge added; do not mutate input."),
            ("graph_remove_edge", "adjacency: dict[int, set[int]], left: int, right: int -> dict[int, set[int]]", "Return a deep copy with an undirected edge removed if present."),
            ("graph_shared_neighbors", "adjacency: dict[int, set[int]], left: int, right: int -> set[int]", "Return neighbors shared by both nodes."),
            ("graph_isolated_nodes", "adjacency: dict[int, set[int]] -> set[int]", "Return keys whose neighbor sets are empty."),
            ("graph_edges_once", "adjacency: dict[int, set[int]] -> set[tuple[int, int]]", "Return each undirected edge once as an ascending endpoint pair."),
        ),
    ),
    (
        "grid coordinates",
        (
            ("grid_translate", "row: int, col: int, dr: int, dc: int -> tuple[int, int]", "Return the translated coordinates."),
            ("grid_manhattan", "a: tuple[int, int], b: tuple[int, int] -> int", "Return Manhattan distance between positions."),
            ("grid_contains", "row: int, col: int, height: int, width: int -> bool", "Test zero-based membership in a positive-size grid."),
            ("grid_neighbors4", "row: int, col: int, height: int, width: int -> list[tuple[int, int]]", "Return valid up, down, left, right neighbors in that order."),
            ("grid_flat_index", "row: int, col: int, width: int -> int", "Return row-major flat index for nonnegative coordinates and positive width."),
            ("grid_unflat_index", "index: int, width: int -> tuple[int, int]", "Return row and column for nonnegative flat index and positive width."),
            ("grid_rotate_clockwise", "row: int, col: int, height: int, width: int -> tuple[int, int]", "Return the position after a clockwise quarter-turn of a valid grid."),
            ("grid_clamp", "row: int, col: int, height: int, width: int -> tuple[int, int]", "Clamp a position into a positive-size grid."),
        ),
    ),
    (
        "byte sequences",
        (
            ("bytes_xor", "left: bytes, right: bytes -> bytes", "XOR equal-length byte strings positionwise."),
            ("bytes_prefix", "data: bytes, count: int -> bytes", "Take at most nonnegative count leading bytes."),
            ("bytes_suffix", "data: bytes, count: int -> bytes", "Take at most nonnegative count trailing bytes."),
            ("bytes_chunks", "data: bytes, size: int -> list[bytes]", "Split data into consecutive chunks of positive size."),
            ("bytes_first_nonzero", "data: bytes -> int | None", "Return the first index holding a nonzero byte, or None."),
            ("bytes_checksum", "data: bytes -> int", "Return the sum of bytes modulo 256."),
            ("bytes_reverse", "data: bytes -> bytes", "Return the bytes in reverse order."),
            ("bytes_count", "data: bytes, value: int -> int", "Count occurrences of byte value in the inclusive range 0..255."),
        ),
    ),
    (
        "integer record tables",
        (
            ("rows_pick_fields", "record: dict[str, int], keys: list[str] -> dict[str, int]", "Return the requested fields present in record."),
            ("rows_where_equal", "rows: list[dict[str, int]], key: str, value: int -> list[dict[str, int]]", "Keep rows whose present key equals value, preserving order."),
            ("rows_sort_key", "rows: list[dict[str, int]], key: str -> list[dict[str, int]]", "Return rows sorted ascending by a key present in every row."),
            ("rows_unique_key", "rows: list[dict[str, int]], key: str -> list[dict[str, int]]", "Keep the first row for each key value, preserving first-seen order."),
            ("rows_sum_key", "rows: list[dict[str, int]], key: str -> int", "Sum the key across rows, treating a missing key as zero."),
            ("rows_index_key", "rows: list[dict[str, int]], key: str -> dict[int, dict[str, int]]", "Index rows by unique integer values of a key present in every row."),
            ("rows_rename_field", "record: dict[str, int], old: str, new: str -> dict[str, int]", "Return a copy with old field renamed to new when present."),
            ("rows_merge", "left: dict[str, int], right: dict[str, int] -> dict[str, int]", "Return a new merged record, with right values winning collisions."),
        ),
    ),
    (
        "prefix totals",
        (
            ("totals_running", "values: list[int] -> list[int]", "Return the inclusive running sums."),
            ("totals_between", "values: list[int], start: int, end: int -> int", "Sum the valid half-open slice from start to end."),
            ("totals_before", "values: list[int], index: int -> int", "Sum values strictly before a valid insertion index."),
            ("totals_first_at_least", "values: list[int], threshold: int -> int | None", "Return the first running-sum index meeting threshold, else None."),
            ("totals_differences", "values: list[int] -> list[int]", "Return each later value minus its predecessor."),
            ("totals_running_max", "values: list[int] -> list[int]", "Return the inclusive running maximum at each position."),
            ("totals_balance_indices", "values: list[int] -> list[int]", "Return indices where the sum before equals the sum after, excluding the value at that index."),
            ("totals_scale", "values: list[int], factor: int -> list[int]", "Return a new list with every value multiplied by factor."),
        ),
    ),
    (
        "whitespace words",
        (
            ("words_lower", "text: str -> list[str]", "Return whitespace-separated words lowercased in order."),
            ("words_count", "text: str -> int", "Count whitespace-separated words."),
            ("words_unique", "text: str -> list[str]", "Return first-seen lowercase words without duplicates."),
            ("words_longest", "text: str -> str", "Return the first longest whitespace-separated word, or empty string."),
            ("words_initials", "text: str -> str", "Concatenate the first character of each whitespace-separated word."),
            ("words_starts_with", "text: str, word: str -> bool", "Return whether the first whitespace-separated word equals word."),
            ("words_strip_outer_punct", "text: str -> str", "Strip ASCII punctuation from both ends of text, preserving interior characters."),
            ("words_join", "words: list[str] -> str", "Join words with one ordinary space."),
        ),
    ),
)

PADDING = (4096, 256, 1024, 4096, 256, 1024, 4096, 256)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(seed, topic):
    rng = random.Random(seed)
    return "".join(
        f"Example {index} for {topic}: item_{index} has value "
        f"{rng.randrange(10, 10000)} and revision {rng.randrange(1, 900)}. "
        "These are inert examples, not function requirements.\n"
        for index in range(400)
    )


def conversation(item, seed, first_padding):
    topic, functions = item
    source = reference(seed, topic)
    prompts, records = [], []
    for turn, (name, signature, meaning) in enumerate(functions, 1):
        padding = first_padding if turn == 1 else PADDING[turn - 1]
        parameters, returns = signature.rsplit(" -> ", 1)
        prompt = (
            f"Write one Python function named {name} with signature "
            f"{name}({parameters}) -> {returns}. {meaning} "
            "Use type hints and a straightforward implementation. Return only "
            "one fenced Python code block, without a prose explanation. "
            "If an input is outside the stated domain, raise ValueError. "
            "Earlier turns are context, not a request to repeat earlier functions.\n\n"
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
    for role, topics, start in (
        ("qualification", (QUALIFICATION,), 20774000),
        ("heldout", HELDOUT, 20775000),
    ):
        groups[role] = []
        for index, item in enumerate(topics):
            seed = start + index
            text, turns = conversation(item, seed, 16384 if index % 2 else 4096)
            path = out / f"{role}-{index}.txt"
            path.write_text(text)
            groups[role].append({
                "file": path.name, "sha256": sha(path),
                "seed": seed, "topic": item[0], "turns": turns,
            })
    manifest = {
        "schema": 1,
        "scope": "fresh Qwen code qualification and heldout split; no model output read",
        "generator_sha256": sha(Path(__file__)),
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
    print(json.dumps({key: len(value) for key, value in result["groups"].items()}))


if __name__ == "__main__":
    main()
