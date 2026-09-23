"""Bounded functional probes for the frozen heldout Python-function requests."""

import argparse
import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys


# Each pair is a literal tuple of valid-domain arguments and its expected value.
# These probes detect incorrect implementations; passing one does not establish
# general functional correctness.
CASES = {
    "set_bit": ("(5, 1)", "7"),
    "clear_bit": ("(7, 1)", "5"),
    "has_bit": ("(5, 2)", "True"),
    "toggle_bit": ("(5, 1)", "7"),
    "count_ones": ("(13,)", "3"),
    "is_power_of_two": ("(16,)", "True"),
    "lowest_set_bit": ("(12,)", "4"),
    "mask_below": ("(4,)", "15"),
    "clamp_int": ("(11, 3, 8)", "8"),
    "in_closed_range": ("(5, 3, 8)", "True"),
    "range_size": ("(3, 8)", "6"),
    "shift_range": ("(3, 8, 2)", "(5, 10)"),
    "overlap_range": ("((1, 4), (4, 7))", "True"),
    "contains_range": ("((1, 8), (3, 5))", "True"),
    "intersect_range": ("((1, 5), (3, 8))", "(3, 5)"),
    "merge_touching": ("((1, 3), (4, 6))", "(1, 6)"),
    "take_prefix": ("([1, 2, 3, 4], 2)", "[1, 2]"),
    "take_suffix": ("([1, 2, 3, 4], 2)", "[3, 4]"),
    "drop_prefix": ("([1, 2, 3, 4], 2)", "[3, 4]"),
    "drop_suffix": ("([1, 2, 3, 4], 2)", "[1, 2]"),
    "rotate_left": ("([1, 2, 3, 4], 1)", "[2, 3, 4, 1]"),
    "rotate_right": ("([1, 2, 3, 4], 1)", "[4, 1, 2, 3]"),
    "every_other": ("([1, 2, 3, 4],)", "[1, 3]"),
    "adjacent_pairs": ("([1, 2, 3, 4],)", "[(1, 2), (2, 3), (3, 4)]"),
    "merge_counts": ('({"a": 2, "b": 1}, {"a": 3, "c": 4})', '{"a": 5, "b": 1, "c": 4}'),
    "keys_sorted": ('({"b": 1, "a": 2},)', '["a", "b"]'),
    "filter_positive": ('({"a": 0, "b": 2, "c": -1},)', '{"b": 2}'),
    "rename_key": ('({"a": 1, "b": 2}, "a", "c")', '{"b": 2, "c": 1}'),
    "get_or_zero": ('({"a": 2}, "b")', "0"),
    "increment_key": ('({"a": 2}, "a")', '{"a": 3}'),
    "count_values": ('({"a": 2, "b": 3},)', "5"),
    "invert_unique": ('({"a": 2, "b": 3},)', '{2: "a", 3: "b"}'),
    "normalize_spaces": ('(" a\\t b\\n",)', '"a b"'),
    "strip_prefix_once": ('("foobar", "foo")', '"bar"'),
    "strip_suffix_once": ('("foobar", "bar")', '"foo"'),
    "line_count": ('("a\\nb\\n",)', "2"),
    "first_line": ('("a\\nb",)', '"a"'),
    "last_line": ('("a\\nb",)', '"b"'),
    "nonempty_lines": ('("a\\n \\nb",)', '["a", "b"]'),
    "join_lines": ('(["a", "b"],)', '"a\\nb"'),
    "ceil_div": ("(10, 3)", "4"),
    "floor_div": ("(10, 3)", "3"),
    "divides": ("(3, 12)", "True"),
    "gcd_pair": ("(12, 18)", "6"),
    "lcm_pair": ("(4, 6)", "12"),
    "digit_sum": ("(-123,)", "6"),
    "reverse_digits": ("(1234,)", "4321"),
    "signum": ("(-7,)", "-1"),
}
MUST_PRESERVE_ARGS = {"merge_counts", "rename_key", "increment_key"}


def worker():
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
    payload = json.load(sys.stdin)
    text = payload["text"].strip()
    if not text.startswith(("```python\n", "```py\n")) or not text.endswith("\n```"):
        raise ValueError("one fenced Python function required")
    source = text.split("\n", 1)[1][:-len("\n```")]
    tree = ast.parse(source)
    if (
        len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].name != payload["function"]
    ):
        raise ValueError("fence contains another module shape")
    allowed = {
        name: getattr(builtins, name) for name in (
            "ValueError", "TypeError", "Exception", "int", "str", "bool", "float", "list",
            "tuple", "set", "dict", "len", "range", "enumerate", "zip",
            "sum", "min", "max", "abs", "sorted", "reversed", "all",
            "any", "isinstance", "round", "pow", "divmod", "filter", "map",
        )
    }
    def restricted_import(name, *args, **kwargs):
        if name not in ("math", "collections", "itertools"):
            raise ImportError("module outside functional-probe allowance")
        return builtins.__import__(name, *args, **kwargs)
    allowed["__import__"] = restricted_import
    scope = {"__builtins__": allowed}
    exec(compile(tree, "<generated>", "exec"), scope)
    arguments, expected = CASES[payload["function"]]
    args = ast.literal_eval(arguments)
    before = copy.deepcopy(args)
    got = scope[payload["function"]](*args)
    want = ast.literal_eval(expected)
    input_ok = payload["function"] not in MUST_PRESERVE_ARGS or args == before
    print(json.dumps({
        "pass": type(got) is type(want) and got == want and input_ok,
        "actual_type": type(got).__name__,
        "preserved_inputs": input_ok,
    }))


def evaluate(root, manifest):
    rows = []
    for index, session in enumerate(manifest["groups"]["heldout"]):
        for arm in ("learn-c3", "monitor-c3", "fixed-c3", "fixed:3", "fixed:2"):
            path = root / f"heldout-{index}-{arm}"
            for turn in session["turns"]:
                answer = path / f"turn-{turn['turn']}.answer.txt"
                payload = {"text": answer.read_text(), "function": turn["function"]}
                try:
                    result = subprocess.run(
                        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                        input=json.dumps(payload), text=True, capture_output=True,
                        timeout=3, check=True,
                    )
                    finding = json.loads(result.stdout)
                except subprocess.CalledProcessError as error:
                    finding = {
                        "pass": False,
                        "error": (
                            error.stderr.strip().splitlines()[-1].split(":", 1)[0]
                            if error.stderr and error.stderr.strip()
                            else "worker exited without a result"
                        )[:80],
                    }
                except (subprocess.TimeoutExpired, ValueError) as error:
                    finding = {"pass": False, "error": type(error).__name__}
                rows.append({
                    "session": index, "arm": arm, "turn": turn["turn"],
                    "function": turn["function"], "answer_sha256": hashlib.sha256(answer.read_bytes()).hexdigest(),
                    **finding,
                })
    return {"schema": 1, "scope": "one valid-domain fixture per requested function, isolated CPU subprocess",
            "rows": rows}


def main():
    if sys.argv[1:] == ["--worker"]:
        worker()
        return
    if sys.argv[1:] == ["--self-check"]:
        correct = {
            "text": "```python\ndef mask_below(bit: int) -> int:\n    return (1 << bit) - 1\n```",
            "function": "mask_below",
        }
        wrong = {
            "text": "```python\ndef mask_below(bit: int) -> int:\n    return bit\n```",
            "function": "mask_below",
        }
        for payload, expected in ((correct, True), (wrong, False)):
            result = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=3, check=True,
            )
            if json.loads(result.stdout)["pass"] is not expected:
                raise ValueError("functional probe failed its bounded execution check")
        print("functional probe self-check passed")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    if {turn["function"] for session in manifest["groups"]["heldout"] for turn in session["turns"]} != set(CASES):
        raise ValueError("functional probes do not match the frozen heldout functions")
    result = evaluate(args.root, manifest)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        arm: {"pass": sum(row["pass"] for row in result["rows"] if row["arm"] == arm),
              "total": sum(row["arm"] == arm for row in result["rows"])}
        for arm in ("learn-c3", "monitor-c3", "fixed-c3", "fixed:3", "fixed:2")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
