"""One bounded functional probe per frozen code request, outside request time."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


FORMAT = re.compile(r"^```(?:python|py)\n(?P<code>[\s\S]+?)\n```\s*$")
CASES = {
    "increase_score": ((4, -2), 2),
    "double_score": ((-3,), -6),
    "score_is_even": ((8,), True),
    "clip_score": ((9, 0, 4), 4),
    "score_gap": ((-3, 5), 8),
    "larger_score": ((-3, 5), 5),
    "nonnegative_score": ((-2,), 0),
    "score_bucket": ((9, 4), 2),
    "graph_neighbors": (({1: {2, 3}}, 1), {2, 3}),
    "graph_degree": (({1: {2, 3}}, 1), 2),
    "graph_has_edge": (({1: {2}}, 1, 2), True),
    "graph_add_edge": (({1: {2}, 2: {1}}, 2, 3), {1: {2}, 2: {1, 3}, 3: {2}}),
    "graph_remove_edge": (({1: {2}, 2: {1, 3}, 3: {2}}, 2, 3), {1: {2}, 2: {1}, 3: set()}),
    "graph_shared_neighbors": (({1: {2, 3}, 4: {3}}, 1, 4), {3}),
    "graph_isolated_nodes": (({1: set(), 2: {3}},), {1}),
    "graph_edges_once": (({1: {2}, 2: {1, 3}, 3: {2}},), {(1, 2), (2, 3)}),
    "grid_translate": ((2, 3, -1, 4), (1, 7)),
    "grid_manhattan": (((1, 2), (4, -2)), 7),
    "grid_contains": ((1, 2, 3, 4), True),
    "grid_neighbors4": ((1, 1, 3, 3), [(0, 1), (2, 1), (1, 0), (1, 2)]),
    "grid_flat_index": ((2, 3, 5), 13),
    "grid_unflat_index": ((13, 5), (2, 3)),
    "grid_rotate_clockwise": ((0, 1, 2, 3), (1, 1)),
    "grid_clamp": ((-1, 8, 3, 4), (0, 3)),
    "bytes_xor": ((b"\x01\xff", b"\x02\x01"), b"\x03\xfe"),
    "bytes_prefix": ((b"abcdef", 3), b"abc"),
    "bytes_suffix": ((b"abcdef", 2), b"ef"),
    "bytes_chunks": ((b"abcde", 2), [b"ab", b"cd", b"e"]),
    "bytes_first_nonzero": ((b"\x00\x00\x02",), 2),
    "bytes_checksum": ((bytes([255, 2]),), 1),
    "bytes_reverse": ((b"abc",), b"cba"),
    "bytes_count": ((b"\x01\x02\x01", 1), 2),
    "rows_pick_fields": (({"a": 2, "b": 3}, ["b", "x"]), {"b": 3}),
    "rows_where_equal": (([{"a": 2}, {"a": 3}, {"b": 2}], "a", 2), [{"a": 2}]),
    "rows_sort_key": (([{"a": 3}, {"a": 1}], "a"), [{"a": 1}, {"a": 3}]),
    "rows_unique_key": (([{"a": 2, "b": 1}, {"a": 2, "b": 3}, {"a": 4}], "a"), [{"a": 2, "b": 1}, {"a": 4}]),
    "rows_sum_key": (([{"a": 2}, {"b": 5}, {"a": 3}], "a"), 5),
    "rows_index_key": (([{"a": 2, "b": 1}, {"a": 4, "b": 3}], "a"), {2: {"a": 2, "b": 1}, 4: {"a": 4, "b": 3}}),
    "rows_rename_field": (({"a": 2, "b": 3}, "a", "z"), {"z": 2, "b": 3}),
    "rows_merge": (({"a": 1, "b": 2}, {"b": 4, "c": 5}), {"a": 1, "b": 4, "c": 5}),
    "totals_running": (([2, -1, 4],), [2, 1, 5]),
    "totals_between": (([2, -1, 4], 1, 3), 3),
    "totals_before": (([2, -1, 4], 2), 1),
    "totals_first_at_least": (([2, -1, 4], 3), 2),
    "totals_differences": (([2, -1, 4],), [-3, 5]),
    "totals_running_max": (([2, -1, 4],), [2, 2, 4]),
    "totals_balance_indices": (([1, 2, 1],), [1]),
    "totals_scale": (([2, -1, 4], -2), [-4, 2, -8]),
    "words_lower": ((" Ab  CD  ",), ["ab", "cd"]),
    "words_count": ((" Ab  CD  ",), 2),
    "words_unique": ((" Ab ab  CD ",), ["ab", "cd"]),
    "words_longest": ((" a bbbb cccc ",), "bbbb"),
    "words_initials": ((" Ab  CD  ",), "AC"),
    "words_starts_with": ((" Ab  CD  ", "Ab"), True),
    "words_strip_outer_punct": (("!hi,there?",), "hi,there"),
    "words_join": ((["a", "b", "c"],), "a b c"),
}


def probe(answer, name):
    match = FORMAT.fullmatch(answer.strip())
    if match is None:
        return {"pass": False, "reason": "format"}
    if name not in CASES:
        raise ValueError("no frozen case for requested function " + name)
    args, expected = CASES[name]
    immutability = name in ("graph_add_edge", "graph_remove_edge")
    program = (
        "import sys, copy\n"
        "space = {}\n"
        "exec(compile(sys.stdin.read(), 'candidate', 'exec'), space)\n"
        f"arguments = {repr(args)}\n"
        "before = copy.deepcopy(arguments)\n"
        f"actual = space[{repr(name)}](*arguments)\n"
        f"assert actual == {repr(expected)}, repr(actual)\n"
        f"assert {not immutability} or arguments == before\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", program],
            input=match["code"], text=True, capture_output=True, timeout=2,
        )
    except subprocess.TimeoutExpired:
        return {"pass": False, "reason": "timeout"}
    if result.returncode:
        return {"pass": False, "reason": "wrong-or-error",
                "stderr_tail": result.stderr[-1000:]}
    return {"pass": True}


def score(root, manifest):
    sessions = []
    for phase in ("qualification", "heldout"):
        source = root / f"{phase}-summary.json"
        if not source.exists():
            continue
        summary = json.loads(source.read_text())
        for record in summary["records"]:
            label = record["name"]
            index = int(label.split("-")[1])
            item = manifest["groups"][phase][index]
            results = [
                {
                    "turn": turn,
                    "function": expected["function"],
                    **probe(
                        (root / label / f"turn-{turn}.answer.txt").read_text(),
                        expected["function"],
                    ),
                }
                for turn, expected in enumerate(item["turns"], 1)
            ]
            sessions.append({
                "session": label, "variant": record["variant"],
                "passed": sum(row["pass"] for row in results),
                "turns": results,
            })
    return {"schema": 1, "scope": "one bounded function case per generated turn",
            "sessions": sessions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.root, json.loads(args.manifest.read_text()))
    with args.out.open("x") as target:
        json.dump(report, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({row["session"]: row["passed"] for row in report["sessions"]}))


if __name__ == "__main__":
    main()
