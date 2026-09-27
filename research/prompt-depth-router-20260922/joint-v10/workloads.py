"""Freeze non-code IFEval and GSM8K continuing conversations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re


IFEVAL_COMMIT = "d36068b845da4c2b24927fee2cea1e6ef98dadda"
IFEVAL_SHA = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
CODE_LIKE = (
    r"\b(?:code|coding|python|script|function|program|regex|javascript|"
    r"sql|java|html|css|algorithm|implement)\b|C\+\+"
)
TURNS = 8
HELDOUT_CONVERSATIONS = 16
DELIMITER = "\n---TURN---\n"


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


def candidates(ifeval_path, gsm_path):
    if sha(ifeval_path) != IFEVAL_SHA or sha(gsm_path) != GSM8K_SHA:
        raise ValueError("non-code benchmark source differs from pinned revision")
    ifeval = [json.loads(line) for line in ifeval_path.read_text().splitlines()]
    gsm = [json.loads(line) for line in gsm_path.read_text().splitlines()]
    if len(ifeval) != 541 or len(gsm) != 1319:
        raise ValueError("non-code benchmark row inventory differs")
    if len({row["key"] for row in ifeval}) != 541:
        raise ValueError("IFEval keys are not unique")
    if len({row["prompt"] for row in ifeval}) != 541:
        raise ValueError("IFEval prompts are not unique")
    ifeval = [
        row for row in ifeval
        if not re.search(CODE_LIKE, row["prompt"], re.I)
        and "---TURN---" not in row["prompt"]
        and row["prompt"] == row["prompt"].strip()
        and len(row["instruction_id_list"]) == len(row["kwargs"])
    ]
    gsm = [
        {**row, "source_index": index}
        for index, row in enumerate(gsm)
        if "---TURN---" not in row["question"]
        and "#### " in row["answer"]
    ]
    if len(ifeval) < 136 or len(gsm) < 136:
        raise ValueError("too few frozen non-code prompts")
    random.Random(25190001).shuffle(ifeval)
    random.Random(25190002).shuffle(gsm)
    return ifeval[:136], gsm[:136]


def prompt(domain, row):
    if domain == "ifeval":
        return row["prompt"]
    if domain == "gsm8k":
        return (
            row["question"].strip()
            + "\n\nShow your work and end with #### <number>."
        )
    raise ValueError("unknown non-code domain")


def conversation(domain, rows, seed, path):
    prompts = [prompt(domain, row) for row in rows]
    path.write_text(DELIMITER.join(prompts) + "\n")
    turns = []
    for turn, (row, text) in enumerate(zip(rows, prompts), 1):
        record = {
            "turn": turn,
            "user_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        if domain == "ifeval":
            record.update({
                "task_id": row["key"],
                "prompt": row["prompt"],
                "instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"],
            })
        else:
            record.update({
                "task_id": row["source_index"],
                "gold_answer": row["answer"].split("#### ")[-1].strip(),
            })
        turns.append(record)
    return {
        "file": path.name,
        "sha256": sha(path),
        "seed": seed,
        "domain": domain,
        "task_ids": [row["task_id"] for row in turns],
        "turns": turns,
    }


def build(ifeval_path, gsm_path, out):
    by_domain = dict(zip(("ifeval", "gsm8k"), candidates(ifeval_path, gsm_path)))
    out.mkdir(parents=True, exist_ok=False)
    groups = {"qualification": {}, "heldout": {}}
    for domain, rows in by_domain.items():
        base = 25191000 if domain == "ifeval" else 25192000
        qualifier = out / f"qualification-{domain}-0.txt"
        groups["qualification"][domain] = [
            conversation(domain, rows[:TURNS], base, qualifier)
        ]
        heldout = []
        for index in range(HELDOUT_CONVERSATIONS):
            subset = rows[TURNS + index * TURNS:TURNS + (index + 1) * TURNS]
            path = out / f"heldout-{domain}-{index}.txt"
            heldout.append(conversation(domain, subset, base + index + 1, path))
        groups["heldout"][domain] = heldout
    manifest = {
        "schema": 1,
        "scope": "Qwen code-trained C/K/D transfer to disjoint non-code IFEval and GSM8K",
        "generator_sha256": sha(Path(__file__)),
        "ifeval_source_commit": IFEVAL_COMMIT,
        "ifeval_source_sha256": IFEVAL_SHA,
        "gsm8k_source_commit": GSM8K_COMMIT,
        "gsm8k_source_sha256": GSM8K_SHA,
        "code_exclusion_regex": CODE_LIKE,
        "ifeval_shuffle_seed": 25190001,
        "gsm8k_shuffle_seed": 25190002,
        "groups": groups,
    }
    save(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifeval", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.ifeval, args.gsm8k, args.out)
    print(json.dumps({
        "ifeval_qualification": len(result["groups"]["qualification"]["ifeval"]),
        "gsm8k_qualification": len(result["groups"]["qualification"]["gsm8k"]),
        "ifeval_heldout": len(result["groups"]["heldout"]["ifeval"]),
        "gsm8k_heldout": len(result["groups"]["heldout"]["gsm8k"]),
    }))


if __name__ == "__main__":
    main()
