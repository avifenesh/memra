"""Freeze independent eight-turn MBPP conversations before native scoring."""

import argparse
import hashlib
import json
from pathlib import Path
import random


MBPP_COMMIT = "d36068b845da4c2b24927fee2cea1e6ef98dadda"
MBPP_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
SPLITS = (("training", 24), ("validation", 8), ("heldout", 16),
          ("qualification", 1))
SEED = 20924003
EXCLUDED_DIAGNOSTIC_IDS = {561, 729, 694, 720, 264, 467, 174, 613}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conversation(tasks, seed, index):
    prompts, turns = [], []
    for turn, task in enumerate(tasks, 1):
        prompt = (
            "Write Python code for this task. Give only one fenced Python "
            "code block, with no reasoning or prose. Earlier turns are "
            "context, not requests to repeat earlier code.\n\n"
            + task["prompt"].strip()
            + "\n\nThe function must satisfy this example:\n"
            + task["test_list"][0].strip()
        )
        prompts.append(prompt)
        turns.append({
            "turn": turn,
            "task_id": task["task_id"],
            "user_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "test_setup_code": "\n".join(task["test_imports"]),
            "test_list": task["test_list"],
            "visible_test_count": 1,
        })
    return "\n---TURN---\n".join(prompts) + "\n", turns


def build(source, out):
    if sha(source) != MBPP_SHA256:
        raise ValueError("MBPP source differs from pinned Google Research commit")
    tasks = json.loads(source.read_text())
    if len(tasks) != 427 or len({task["task_id"] for task in tasks}) != 427:
        raise ValueError("MBPP task inventory differs")
    usable = [
        task for task in tasks
        if task["task_id"] > 10
        and task["task_id"] not in EXCLUDED_DIAGNOSTIC_IDS
        and not task["test_imports"]
        and len(task["test_list"]) >= 3
    ]
    rng = random.Random(SEED)
    rng.shuffle(usable)
    needed = sum(count for _, count in SPLITS) * 8
    if len(usable) < needed:
        raise ValueError("insufficient distinct MBPP tasks for frozen split")
    out.mkdir(parents=True, exist_ok=False)
    offset = 0
    groups = {}
    for role, count in SPLITS:
        entries = []
        for index in range(count):
            chosen = usable[offset:offset + 8]
            offset += 8
            seed = 20925000 + offset
            text, turns = conversation(chosen, seed, index)
            path = out / f"{role}-{index}.txt"
            path.write_text(text)
            entries.append({
                "file": path.name,
                "sha256": sha(path),
                "seed": seed,
                "task_ids": [task["task_id"] for task in chosen],
                "turns": turns,
            })
        groups[role] = entries
    manifest = {
        "schema": 1,
        "scope": "custom disjoint sanitized MBPP eight-turn Qwen C/K/D workload split",
        "source_commit": MBPP_COMMIT,
        "source_sha256": MBPP_SHA256,
        "excluded_diagnostic_task_ids": sorted(EXCLUDED_DIAGNOSTIC_IDS),
        "generator_sha256": sha(Path(__file__)),
        "assignment_seed": SEED,
        "groups": groups,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.out)
    print(json.dumps({
        "source_sha256": result["source_sha256"],
        "groups": {name: len(group) for name, group in result["groups"].items()},
    }))


if __name__ == "__main__":
    main()
