"""Freeze disjoint non-code training, selection, and final conversations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


IFEVAL_SHA = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
GSM8K_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
V10_MANIFEST_SHA = "2142c62394fbb3f90c684a942e09a227f24f8a32d0011f550dd7b6841ed32dc1"
CODE_LIKE = (
    r"\b(?:code|coding|python|script|function|program|regex|javascript|"
    r"sql|java|html|css|algorithm|implement)\b|C\+\+"
)
SPLITS = (("qualification", 1), ("training", 16),
          ("validation", 8), ("heldout", 16))
TURNS = 8
DELIMITER = "\n---TURN---\n"
SEEDS = {"ifeval": 25194001, "gsm8k": 25194002}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def candidates(ifeval_path, gsm_path, v10_path):
    if (
        sha(ifeval_path) != IFEVAL_SHA
        or sha(gsm_path) != GSM8K_SHA
        or sha(v10_path) != V10_MANIFEST_SHA
    ):
        raise ValueError("non-code source or v10 frozen split differs")
    earlier = json.loads(v10_path.read_text())
    if earlier["schema"] != 1 or earlier["code_exclusion_regex"] != CODE_LIKE:
        raise ValueError("v10 non-code split contract differs")
    used = {}
    for domain in SEEDS:
        ids = [
            task
            for phase in ("qualification", "heldout")
            for session in earlier["groups"][phase][domain]
            for task in session["task_ids"]
        ]
        if len(ids) != 136 or len(set(ids)) != len(ids):
            raise ValueError(f"v10 {domain} task exclusions differ")
        used[domain] = set(ids)
    ifeval = [
        row for row in (
            json.loads(line) for line in ifeval_path.read_text().splitlines()
        )
        if row["key"] not in used["ifeval"]
        and not re.search(CODE_LIKE, row["prompt"], re.I)
        and row["prompt"] == row["prompt"].strip()
        and DELIMITER.strip() not in row["prompt"]
        and len(row["instruction_id_list"]) == len(row["kwargs"])
    ]
    gsm = [
        {**row, "source_index": index}
        for index, row in enumerate(
            json.loads(line) for line in gsm_path.read_text().splitlines()
        )
        if index not in used["gsm8k"]
        and DELIMITER.strip() not in row["question"]
        and "#### " in row["answer"]
    ]
    if len(ifeval) < sum(n for _, n in SPLITS) * TURNS:
        raise ValueError("too few untouched IFEval prompts")
    if len(gsm) < sum(n for _, n in SPLITS) * TURNS:
        raise ValueError("too few untouched GSM8K questions")
    if len({row["key"] for row in ifeval}) != len(ifeval):
        raise ValueError("IFEval task keys repeat")
    for domain, rows in (("ifeval", ifeval), ("gsm8k", gsm)):
        random.Random(SEEDS[domain]).shuffle(rows)
    return {"ifeval": ifeval, "gsm8k": gsm}, used


def task_prompt(domain, row):
    if domain == "ifeval":
        return row["prompt"]
    return (
        row["question"].strip()
        + "\n\nShow your work and end with #### <number>."
    )


def conversation(domain, rows, seed, path):
    prompts = [task_prompt(domain, row) for row in rows]
    path.write_text(DELIMITER.join(prompts) + "\n")
    turns = []
    for number, (row, text) in enumerate(zip(rows, prompts), 1):
        item = {
            "turn": number,
            "user_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if domain == "ifeval":
            item.update({
                "task_id": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            })
        else:
            item.update({
                "task_id": row["source_index"],
                "gold_answer": row["answer"].split("#### ")[-1].strip(),
            })
        turns.append(item)
    return {
        "file": path.name,
        "sha256": sha(path),
        "seed": seed,
        "domain": domain,
        "task_ids": [item["task_id"] for item in turns],
        "turns": turns,
    }


def build(ifeval_path, gsm_path, v10_path, out):
    by_domain, used = candidates(ifeval_path, gsm_path, v10_path)
    out.mkdir(parents=True, exist_ok=False)
    groups = {name: {} for name, _ in SPLITS}
    for domain, rows in by_domain.items():
        offset = 0
        seed_base = 25195000 if domain == "ifeval" else 25196000
        for phase, count in SPLITS:
            sessions = []
            for index in range(count):
                chosen = rows[offset:offset + TURNS]
                offset += TURNS
                path = out / f"{phase}-{domain}-{index}.txt"
                sessions.append(conversation(
                    domain, chosen, seed_base + offset, path
                ))
            groups[phase][domain] = sessions
        assigned = [
            task
            for phase, _ in SPLITS
            for session in groups[phase][domain]
            for task in session["task_ids"]
        ]
        if len(assigned) != 328 or len(set(assigned)) != 328:
            raise ValueError(f"v11 {domain} split repeats tasks")
        if set(assigned) & used[domain]:
            raise ValueError(f"v11 {domain} overlaps v10 task")
    manifest = {
        "schema": 1,
        "scope": "Qwen non-code mixed-training follow-up on untouched tasks",
        "generator_sha256": sha(Path(__file__)),
        "ifeval_source_sha256": IFEVAL_SHA,
        "gsm8k_source_sha256": GSM8K_SHA,
        "v10_manifest_sha256": V10_MANIFEST_SHA,
        "code_exclusion_regex": CODE_LIKE,
        "shuffle_seeds": SEEDS,
        "splits": dict(SPLITS),
        "groups": groups,
    }
    save(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--v10", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.ifeval, args.gsm8k, args.v10, args.out)
    print(json.dumps({
        domain: {
            phase: len(result["groups"][phase][domain])
            for phase, _ in SPLITS
        }
        for domain in SEEDS
    }, sort_keys=True))


if __name__ == "__main__":
    main()
