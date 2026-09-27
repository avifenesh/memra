"""Score frozen MBPP tests after native timing, one isolated process per turn."""

import argparse
import ast
import json
from pathlib import Path
import re
import resource
import subprocess
import sys


FORMAT = re.compile(r"^```(?:python|py)\n(?P<code>[\s\S]+?)\n```\s*$")
PROGRAM = (
    "import json, sys\n"
    "item = json.load(sys.stdin)\n"
    "space = {}\n"
    "exec(compile(item['code'], 'candidate', 'exec'), space)\n"
    "if item['setup']:\n"
    "    exec(compile(item['setup'], 'setup', 'exec'), space)\n"
    "for number, test in enumerate(item['tests'], 1):\n"
    "    exec(compile(test, f'test-{number}', 'exec'), space)\n"
)


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))


def probe(answer, task):
    match = FORMAT.fullmatch(answer.strip())
    if match is None:
        return {"format": False, "syntax": False, "tests": 0, "reason": "format"}
    code = match["code"]
    try:
        ast.parse(code)
    except SyntaxError:
        return {"format": True, "syntax": False, "tests": 0, "reason": "syntax"}
    tests = task["test_list"][task["visible_test_count"]:]
    if len(tests) < 2:
        raise ValueError("frozen MBPP task lacks two hidden tests")
    passed = 0
    for test in tests:
        item = {
            "code": code,
            "setup": task["test_setup_code"],
            "tests": [test],
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", PROGRAM],
                input=json.dumps(item), text=True, capture_output=True,
                timeout=4, preexec_fn=limits,
            )
        except subprocess.TimeoutExpired:
            return {
                "format": True, "syntax": True, "tests": passed,
                "reason": "timeout",
            }
        if result.returncode:
            return {
                "format": True, "syntax": True, "tests": passed,
                "reason": "test-failed",
                "stderr_tail": result.stderr[-600:],
            }
        passed += 1
    return {"format": True, "syntax": True, "tests": passed, "reason": "pass"}


def score(root, manifest, phases):
    sessions = []
    for phase in phases:
        entries = manifest["groups"][phase]
        for record in sorted(root.glob(f"{phase}-*.result.json")):
            result = json.loads(record.read_text())
            index = int(result["name"].split("-")[1])
            expected = entries[index]
            if result["task_ids"] != expected["task_ids"]:
                raise ValueError("quality tasks differ from native result")
            results = [
                {
                    "turn": turn,
                    "task_id": task["task_id"],
                    **probe(
                        (root / result["name"] / f"turn-{turn}.answer.txt").read_text(),
                        task,
                    ),
                }
                for turn, task in enumerate(expected["turns"], 1)
            ]
            sessions.append({
                "session": result["name"],
                "variant": result["variant"],
                "format_pass": sum(item["format"] for item in results),
                "syntax_pass": sum(item["syntax"] for item in results),
                "all_tests_pass": sum(
                    item["tests"] == len(
                        expected["turns"][item["turn"] - 1]["test_list"]
                    ) - expected["turns"][item["turn"] - 1]["visible_test_count"]
                    for item in results
                ),
                "turns": results,
            })
    return {
        "schema": 1,
        "scope": "custom disjoint MBPP tasks scored after native request clocks",
        "sessions": sessions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", default=["training"])
    args = parser.parse_args()
    result = score(args.root, json.loads(args.manifest.read_text()), args.phases)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "sessions": len(result["sessions"]),
        "format_pass": sum(row["format_pass"] for row in result["sessions"]),
        "all_tests_pass": sum(row["all_tests_pass"] for row in result["sessions"]),
    }))


if __name__ == "__main__":
    main()
