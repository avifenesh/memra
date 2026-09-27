"""Grade IFEval prose and GSM8K math after native request timing."""

import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import sys


ANS_RE = re.compile(
    r"####[ \t]+([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"
)
DOMAINS = ("ifeval", "gsm8k")


def ifeval_grader(source):
    sys.path.insert(0, str(source.parent))
    import langdetect
    from instruction_following_eval import evaluation_lib

    langdetect.DetectorFactory.seed = 0
    return evaluation_lib


def gsm_answer(text):
    lines = text.strip().splitlines()
    match = ANS_RE.fullmatch(lines[-1].strip()) if lines else None
    return Decimal(match.group(1).replace(",", "")) if match else None


def grade(domain, task, answer, evaluator):
    if domain == "ifeval":
        inp = evaluator.InputExample(
            key=task["task_id"],
            instruction_id_list=task["instruction_id_list"],
            prompt=task["prompt"],
            kwargs=task["kwargs"],
        )
        result = evaluator.test_instruction_following_strict(
            inp, {task["prompt"]: answer}
        )
        if len(result.follow_instruction_list) != len(task["instruction_id_list"]):
            raise ValueError("IFEval instruction inventory changed")
        return {
            "pass": result.follow_all_instructions,
            "instruction_pass": sum(result.follow_instruction_list),
            "instruction_total": len(result.follow_instruction_list),
        }
    if domain == "gsm8k":
        expected = gsm_answer("#### " + task["gold_answer"])
        if expected is None:
            raise ValueError("GSM8K gold answer cannot be parsed")
        observed = gsm_answer(answer)
        return {
            "pass": observed == expected,
            "numeric_answer_present": observed is not None,
        }
    raise ValueError("unknown non-code stratum")


def score(root, manifest, arms, phase, evaluator):
    results = {}
    for domain in DOMAINS:
        sessions = []
        entries = manifest["groups"][phase][domain]
        for index, entry in enumerate(entries):
            for arm in arms["arms"]:
                name = f"{phase}-{domain}-{index}-{arm['label']}"
                path = root / f"{name}.result.json"
                native = json.loads(path.read_text())
                if (
                    native["name"] != name
                    or native["task_ids"] != entry["task_ids"]
                    or native["cached_later_turns"] != 7
                ):
                    raise ValueError("native result differs from quality task")
                turns = []
                for turn, task in enumerate(entry["turns"], 1):
                    answer = (
                        root / name / f"turn-{turn}.answer.txt"
                    ).read_text()
                    turns.append({
                        "turn": turn,
                        "task_id": task["task_id"],
                        **grade(domain, task, answer, evaluator),
                    })
                sessions.append({
                    "session": name,
                    "variant": arm["label"],
                    "task_pass": sum(item["pass"] for item in turns),
                    "instruction_pass": sum(
                        item.get("instruction_pass", 0) for item in turns
                    ),
                    "instruction_total": sum(
                        item.get("instruction_total", 0) for item in turns
                    ),
                    "numeric_answer_present": sum(
                        item.get("numeric_answer_present", False) for item in turns
                    ),
                    "turns": turns,
                })
        results[domain] = {"sessions": sessions}
    return {
        "schema": 1,
        "phase": phase,
        "scope": "IFEval strict and GSM8K final numeric checks outside native clocks",
        "domains": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--ifeval-source", type=Path, required=True)
    parser.add_argument("--phase", choices=("qualification", "heldout"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    arms = json.loads(args.arms.read_text())
    evaluator = ifeval_grader(args.ifeval_source)
    result = score(args.root, manifest, arms, args.phase, evaluator)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        domain: {
            "sessions": len(group["sessions"]),
            "task_pass": sum(row["task_pass"] for row in group["sessions"]),
        }
        for domain, group in result["domains"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
