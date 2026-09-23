"""Build the K=1/2/3 marginal round-cost table from matched native sessions."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from learn import loop_candidate


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def cost_table(arms, source_sha256, model_sha256):
    if set(arms) != {1, 2, 3}:
        raise ValueError("cost control requires K=1/2/3")
    controls = {}
    for k, root in arms.items():
        turns = rows(root / "turns.tsv")
        rounds = rows(root / "rounds.tsv")
        if len(turns) != 8 or not rounds:
            raise ValueError("cost arm lacks eight native turns")
        loops = set()
        for turn in range(1, 9):
            ids = [int(token) for token in (
                root / f"turn-{turn}.output.ids"
            ).read_text().split()]
            if loop_candidate(ids):
                loops.add(turn)
        controls[k] = {
            "turns": turns, "rounds": rounds, "loops": loops,
            "turns_sha256": sha(root / "turns.tsv"),
            "rounds_sha256": sha(root / "rounds.tsv"),
        }
    excluded = set.union(*(arm["loops"] for arm in controls.values()))
    result = {}
    for k, control in controls.items():
        durations = []
        for row in control["rounds"]:
            if set(row) != {
                "turn", "round", "draft_depth", "emitted",
                "elapsed_ns", "eligible_for_learning",
            }:
                raise ValueError("native round receipt has another schema")
            turn = int(row["turn"])
            if turn in excluded or row["eligible_for_learning"] != "true":
                continue
            if int(row["draft_depth"]) != k or int(row["emitted"]) > k + 1:
                raise ValueError("cost control changed draft depth")
            ns = int(row["elapsed_ns"])
            if ns <= 0:
                raise ValueError("invalid native round cost")
            durations.append(ns)
        if len(durations) < 8:
            raise ValueError("insufficient matched native cost rounds")
        mean = sum(durations) / len(durations)
        variance = sum((value - mean) ** 2 for value in durations) / (
            len(durations) - 1
        )
        result[str(k)] = {
            "round_ns": mean,
            "rounds": len(durations),
            "standard_error_ns": math.sqrt(variance / len(durations)),
            "turns_sha256": control["turns_sha256"],
            "rounds_sha256": control["rounds_sha256"],
        }
    if not result["1"]["round_ns"] < result["2"]["round_ns"] < result["3"]["round_ns"]:
        raise ValueError("measured K costs are not strictly increasing")
    return {
        "schema": 1,
        "scope": "eligible, nonlooped native rounds; matched turn exclusions",
        "source_sha256": source_sha256,
        "model_sha256": model_sha256,
        "excluded_turns": sorted(excluded),
        "round_ns": {k: value["round_ns"] for k, value in result.items()},
        "detail": result,
    }


def main():
    parser = argparse.ArgumentParser()
    for k in (1, 2, 3):
        parser.add_argument(f"--k{k}", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = cost_table(
        {k: getattr(args, f"k{k}") for k in (1, 2, 3)},
        args.source_sha256, args.model_sha256,
    )
    with args.out.open("x") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"round_ns": report["round_ns"], "excluded": report["excluded_turns"]}))


if __name__ == "__main__":
    main()
