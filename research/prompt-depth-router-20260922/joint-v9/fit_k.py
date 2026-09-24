"""Fit draft-only K=3/10/20 controllers from measured complete turns."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_COUNT = {"first16": 10, "first32": 10, "prior": 11}
ACTIONS = (3, 10, 20)


def vector(row, variant):
    count = 16 if variant == "first16" else 32
    prefix = row["first32_user_ids"][:count]
    values = np.zeros(FEATURE_COUNT[variant], dtype=np.float64)
    values[0] = 1
    for token in prefix:
        bucket = ((token * 0x9E3779B1) & 0xFFFFFFFF) % 8
        values[1 + bucket] += 1 / len(prefix)
    values[9] = len(prefix) / count
    if variant == "prior" and row["previous_turn_acceptance"] is not None:
        values[10] = row["previous_turn_acceptance"]
    return values


def fit(rows, variant, lam, alpha):
    weights = {}
    for k in ACTIONS:
        selected = [row for row in rows if row["draft_k"] == k]
        if len(selected) < 32:
            raise ValueError(f"too few measured K={k} turns")
        x = np.stack([vector(row, variant) for row in selected])
        y = np.array([
            row["output_tokens"] - lam * row["complete_request_seconds"]
            for row in selected
        ])
        penalty = np.eye(x.shape[1]) * alpha
        penalty[0, 0] = 0
        weights[k] = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        if not np.isfinite(weights[k]).all():
            raise ValueError(f"nonfinite K={k} weights")
    return weights


def predict(row, variant, weights):
    x = vector(row, variant)
    best = 20
    best_score = float(x @ weights[20])
    for k in ACTIONS:
        value = float(x @ weights[k])
        if value > best_score + 1e-9:
            best = k
            best_score = value
    return best


def train(data, out):
    with gzip.open(data / "k.jsonl.gz", "rt") as source:
        rows = [json.loads(line) for line in source]
    if (
        len(rows) != 192
        or {row["draft_k"] for row in rows} != set(ACTIONS)
        or {row["source"] for row in rows} != {"v6", "v8"}
    ):
        raise ValueError("the v6/v8 measured-turn inventory differs")
    baseline = [
        row for row in rows
        if row["source"] == "v8" and row["draft_k"] == 20
    ]
    if len(baseline) != 48:
        raise ValueError("missing v8 fixed-K20 reference turns")
    lam = sum(row["output_tokens"] for row in baseline) / sum(
        row["complete_request_seconds"] for row in baseline
    )
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for alpha in (20, 100):
        directory = out / f"ridge-{alpha}"
        directory.mkdir()
        for variant in FEATURE_COUNT:
            weights = fit(rows, variant, lam, alpha)
            lines = [f"joint-v6-draft-topk\t1\t{variant}\t{lam:.12g}\t{alpha}"]
            for k in ACTIONS:
                lines.append(
                    f"K\t{k}\t" +
                    ",".join(f"{value:.17g}" for value in weights[k])
                )
            payload = ("\n".join(lines) + "\n").encode()
            path = directory / f"topk-{variant}.tsv"
            path.write_bytes(payload)
            results.append({
                "alpha": alpha,
                "variant": variant,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "training_only_actions": dict(
                    sorted(Counter(
                        predict(row, variant, weights) for row in rows
                    ).items())
                ),
            })
    manifest = {
        "schema": 1,
        "use": "training-only",
        "input_manifest_sha256": hashlib.sha256(
            (data / "manifest.json").read_bytes()
        ).hexdigest(),
        "input_k_sha256": hashlib.sha256(
            (data / "k.jsonl.gz").read_bytes()
        ).hexdigest(),
        "reference_tok_s": lam,
        "sampled_turns": {
            str(k): sum(row["draft_k"] == k for row in rows)
            for k in ACTIONS
        },
        "models": results,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.data, args.out)
    print(json.dumps({
        "reference_tok_s": result["reference_tok_s"],
        "sampled_turns": result["sampled_turns"],
        "training_only_actions": [
            [row["alpha"], row["variant"], row["training_only_actions"]]
            for row in result["models"]
        ],
    }))


if __name__ == "__main__":
    main()
