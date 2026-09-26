"""Check whether native first-token K features distinguish non-code topics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import fit_k


WORKLOAD_SHA = "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
DOMAINS = ("ifeval", "gsm8k")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def prefix(root, turn, expected_sha):
    if sha(root / f"turn-{turn}.user.txt") != expected_sha:
        raise ValueError("topic visibility prompt differs")
    ids = [
        int(value) for value in
        (root / f"turn-{turn}.user-prefix.ids").read_text().split()
    ]
    if not 1 <= len(ids) <= 32:
        raise ValueError("native first-token prefix differs")
    return ids


def examples(root, manifest, phase):
    features = []
    labels = []
    counts = {}
    for label, domain in enumerate(DOMAINS):
        sessions = manifest["groups"][phase][domain]
        if len(sessions) != manifest["splits"][phase]:
            raise ValueError("topic visibility split count differs")
        counts[domain] = 0
        for index, entry in enumerate(sessions):
            name = (
                f"training-{label * 16 + index}-k20-fixed-d3"
                if phase == "training"
                else f"validation-{domain}-{index}-fixed-k20-d3-c0"
            )
            session = root / name
            result = json.loads((root / f"{name}.result.json").read_text())
            if result["name"] != name or result["task_ids"] != entry["task_ids"]:
                raise ValueError("topic visibility native session differs")
            for turn, task in enumerate(entry["turns"], 1):
                ids = prefix(session, turn, task["user_sha256"])
                features.append(ids)
                labels.append(label)
                counts[domain] += 1
    return features, np.array(labels, dtype=np.float64), counts


def fit(training, targets, variant):
    matrix = np.stack([
        fit_k.vector({
            "first32_user_ids": ids,
            "previous_turn_acceptance": None,
        }, variant)
        for ids in training
    ])
    penalty = np.eye(matrix.shape[1]) * 10
    penalty[0, 0] = 0
    weights = np.linalg.solve(
        matrix.T @ matrix + penalty, matrix.T @ targets
    )
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite prefix visibility probe")
    return weights


def report(train_root, validation_root, workloads):
    if sha(workloads / "manifest.json") != WORKLOAD_SHA:
        raise ValueError("frozen topic visibility source differs")
    manifest = json.loads((workloads / "manifest.json").read_text())
    training, targets, training_counts = examples(
        train_root, manifest, "training"
    )
    validation, expected, validation_counts = examples(
        validation_root, manifest, "validation"
    )
    result = {
        "schema": 1,
        "scope": "native first-token topic visibility, not K utility",
        "workload_sha256": WORKLOAD_SHA,
        "training_turns": training_counts,
        "validation_turns": validation_counts,
        "variants": {},
    }
    for variant in ("first16", "first32"):
        weights = fit(training, targets, variant)
        matrix = np.stack([
            fit_k.vector({
                "first32_user_ids": ids,
                "previous_turn_acceptance": None,
            }, variant)
            for ids in validation
        ])
        scores = matrix @ weights
        predicted = scores >= 0.5
        correct = predicted == expected
        result["variants"][variant] = {
            "weights": [float(value) for value in weights],
            "validation_accuracy": float(np.mean(correct)),
            "by_domain_accuracy": {
                domain: float(np.mean(correct[expected == label]))
                for label, domain in enumerate(DOMAINS)
            },
            "confusion": {
                f"{domain}_as_{choice}": int(np.sum(
                    (expected == label) & (predicted == chosen)
                ))
                for label, domain in enumerate(DOMAINS)
                for chosen, choice in enumerate(DOMAINS)
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(
        args.training_root, args.validation_root, args.workloads
    )
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        variant: row["validation_accuracy"]
        for variant, row in result["variants"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
