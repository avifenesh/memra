"""Grade fresh MBPP and GSM8K turns outside native request clocks."""

import argparse
import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
FORMAT = re.compile(r"^```(?:python|py)\n(?P<code>[\s\S]+?)\n```\s*$")
ANSWER = re.compile(
    r"####[ \t]+([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
)
PROGRAM = (
    "import json, sys\n"
    "item = json.load(sys.stdin)\n"
    "space = {}\n"
    "exec(compile(item['code'], 'candidate', 'exec'), space)\n"
    "for number, test in enumerate(item['tests'], 1):\n"
    "    exec(compile(test, f'test-{number}', 'exec'), space)\n"
)
SANDBOX = (
    "/usr/bin/bwrap", "--die-with-parent", "--unshare-all", "--new-session",
    "--ro-bind", "/usr", "/usr",
    "--ro-bind-try", "/lib", "/lib",
    "--ro-bind-try", "/lib64", "/lib64",
    "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    "--chdir", "/tmp", "--clearenv", "--setenv", "LANG", "C.UTF-8",
    "--", "/usr/bin/python3", "-I", "-S", "-c", PROGRAM,
)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def sandbox_preflight():
    command = (*SANDBOX[:-1], "print('sandbox-ready')")
    result = subprocess.run(
        command, text=True, capture_output=True, timeout=5,
        preexec_fn=limits,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
    )
    if result.returncode or result.stdout != "sandbox-ready\n":
        raise RuntimeError("credential-free MBPP sandbox is unavailable")


def code_grade(answer, task):
    matched = FORMAT.fullmatch(answer.strip())
    if matched is None:
        return {"pass": False, "format": False, "syntax": False,
                "hidden_tests_pass": 0, "reason": "format"}
    try:
        ast.parse(matched["code"])
    except SyntaxError:
        return {"pass": False, "format": True, "syntax": False,
                "hidden_tests_pass": 0, "reason": "syntax"}
    if task["visible_test_count"] != 1 or task["test_setup_code"]:
        raise ValueError("fresh MBPP hidden test contract differs")
    tests = task["test_list"][1:]
    if len(tests) < 2:
        raise ValueError("fresh MBPP lacks two hidden tests")
    passed = 0
    for test in tests:
        try:
            result = subprocess.run(
                SANDBOX,
                input=json.dumps({"code": matched["code"], "tests": [test]}),
                text=True, capture_output=True, timeout=5, preexec_fn=limits,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
        except subprocess.TimeoutExpired:
            return {
                "pass": False, "format": True, "syntax": True,
                "hidden_tests_pass": passed, "reason": "timeout",
            }
        if result.returncode:
            return {
                "pass": False, "format": True, "syntax": True,
                "hidden_tests_pass": passed, "reason": "hidden-test",
            }
        passed += 1
    return {
        "pass": True, "format": True, "syntax": True,
        "hidden_tests_pass": passed, "reason": "pass",
    }


def math_answer(text):
    lines = text.strip().splitlines()
    matched = ANSWER.fullmatch(lines[-1].strip()) if lines else None
    return Decimal(matched.group(1).replace(",", "")) if matched else None


def math_grade(answer, task):
    gold = math_answer("#### " + task["gold_answer"])
    if gold is None:
        raise ValueError("fresh GSM8K gold answer differs")
    observed = math_answer(answer)
    return {
        "pass": observed == gold,
        "numeric_answer_present": observed is not None,
    }


def inspect(args):
    expected = {"validation": VALIDATION_SHA, "final": FULL_SHA}[args.phase]
    if sha(args.workloads / "manifest.json") != expected:
        raise ValueError("fresh quality phase workload changed")
    manifest = json.loads((args.workloads / "manifest.json").read_text())
    arms = json.loads(args.arms.read_text())
    if (
        manifest["schema"] != 1
        or arms["schema"] != 1
        or arms["phase"] != args.phase
        or arms["source_manifest_sha256"] != FULL_SHA
        or len({item["label"] for item in arms["arms"]})
        != len(arms["arms"])
    ):
        raise ValueError("fresh quality arm inventory changed")
    return manifest, arms


def grade(args):
    manifest, arms = inspect(args)
    sandbox_preflight()
    domains = {}
    for domain, grader in (("code", code_grade), ("math", math_grade)):
        sessions = []
        for index, entry in enumerate(manifest["groups"][args.phase][domain]):
            for arm in arms["arms"]:
                name = f"{args.phase}-{domain}-{index}-{arm['label']}"
                native = json.loads(
                    (args.root / f"{name}.result.json").read_text()
                )
                if (
                    native["name"] != name
                    or native["task_ids"] != entry["task_ids"]
                    or native["cached_later_turns"] != 7
                ):
                    raise ValueError("fresh quality task lacks native receipt")
                turns = []
                for turn, task in enumerate(entry["turns"], 1):
                    output = args.root / name / f"turn-{turn}.answer.txt"
                    turns.append({
                        "turn": turn, "task_id": task["task_id"],
                        **grader(output.read_text(), task),
                    })
                sessions.append({
                    "session": name, "variant": arm["label"],
                    "task_pass": sum(item["pass"] for item in turns),
                    "turns": turns,
                })
        domains[domain] = {"sessions": sessions}
    return {
        "schema": 1, "phase": args.phase,
        "scope": "fresh MBPP hidden tests and GSM8K strict numeric",
        "arms_sha256": sha(args.arms),
        "workloads_sha256": sha(args.workloads / "manifest.json"),
        "domains": domains,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "workloads", "arms", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final"),
                        required=True)
    args = parser.parse_args()
    for name in ("root", "workloads", "arms", "out"):
        setattr(args, name, getattr(args, name).resolve())
    result = grade(args)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "phase": args.phase,
        "sessions": {
            domain: len(item["sessions"])
            for domain, item in result["domains"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
