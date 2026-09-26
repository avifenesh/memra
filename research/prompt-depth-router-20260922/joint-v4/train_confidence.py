"""Fit after-offer acceptance heads; score stopping by marginal round cost."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np

from train_depth import FEATURES, features, table, token_classes


def probability(bits):
    value = struct.unpack("<f", struct.pack("<I", int(bits)))[0]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid sampled chosen-token probability")
    return value


def q_logit(value):
    value = min(1 - 1e-6, max(1e-6, value))
    return max(-12.0, min(12.0, math.log(value / (1 - value)))) / 12


def samples(root, classes, variant):
    summary = json.loads((root / "training-summary.json").read_text())
    sessions = [row for row in summary["records"] if row["arm"] == "explore-d"]
    if len(sessions) != 6:
        raise ValueError("confidence fit needs six randomized development conversations")
    rows = defaultdict(list)
    for record in sessions:
        session = root / record["name"]
        spans = {
            (int(row["turn"]), int(row["round"])): row
            for row in table(session / "spans.tsv")
        }
        prior = None
        for turn in range(1, 9):
            output = [int(value) for value in (
                session / f"turn-{turn}.output.ids"
            ).read_text().split()]
            for row in table(session / f"turn-{turn}.confidence.tsv"):
                key = turn, int(row["round"])
                span = spans[key]
                d = int(row["drafted"])
                accepted = int(row["accepted_prefix"])
                elapsed_ms = int(row["elapsed_ns"]) / 1e6
                offered = [probability(bits) for bits in row["q_bits"].split(",")]
                if (
                    len(offered) != d or d not in (1, 2, 3, 4)
                    or row["eligible"] != span["eligible"]
                    or int(span["k"]) != d
                    or not 0 <= accepted <= d
                ):
                    raise ValueError("sampled confidence and randomized D differ")
                if row["eligible"] == "true":
                    start = int(span["output_start"])
                    if start > len(output) or elapsed_ms <= 0:
                        raise ValueError("eligible confidence row refers to future output")
                    history = output[:start][-16:]
                    base = features(history, prior, classes, variant)
                    for j, q in enumerate(offered):
                        if accepted < j:
                            break
                        x = np.concatenate((base, [q_logit(q)]))
                        rows[j].append((x, float(accepted > j)))
                    prior = accepted / d, elapsed_ms
    if any(len(rows[j]) < 150 for j in range(4)):
        raise ValueError("censored confidence labels lack depth-4 exposure")
    return rows


def logistic(rows, alpha):
    x = np.stack([row[0] for row in rows])
    y = np.array([row[1] for row in rows], dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0
    for _ in range(40):
        z = np.clip(x @ weights, -30, 30)
        chance = 1 / (1 + np.exp(-z))
        grad = x.T @ (y - chance) - penalty @ weights
        curvature = (x.T * (chance * (1 - chance))) @ x + penalty
        change = np.linalg.solve(curvature, grad)
        weights += change
        if np.max(np.abs(change)) < 1e-8:
            break
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite conditional acceptance weights")
    return weights


def fit(root, out, variant):
    if variant not in FEATURES:
        raise ValueError("confidence feature version differs from D model")
    analysis = json.loads((root / "training-analysis.json").read_text())
    lam = analysis["fixed_baseline_tok_s"]
    costs = analysis["randomized_d"]
    classes = token_classes(root)
    data = samples(root, classes, variant)
    lines = [f"joint-v4-confidence\t1\t{variant}\t{lam:.12g}"]
    records = []
    for j in range(3):
        weights = logistic(data[j], 20)
        next_rate = sum(label for _, label in data[j + 1]) / len(data[j + 1])
        marginal_ms = 1000 * (
            costs[str(j + 2)]["seconds_per_round"]
            - costs[str(j + 1)]["seconds_per_round"]
        )
        lines.append(
            f"C\t{j}\t{next_rate:.17g}\t{marginal_ms:.17g}\t"
            + ",".join(f"{weight:.17g}" for weight in weights)
        )
        records.append({
            "position": j, "labels": len(data[j]),
            "next_acceptance_rate": next_rate,
            "marginal_ms": marginal_ms,
        })
    for token, klass in sorted(classes.items()):
        lines.append(f"B\t{token}\t{klass}")
    payload = ("\n".join(lines) + "\n").encode()
    with out.open("xb") as target:
        target.write(payload)
    return {
        "variant": variant,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "training": records,
        "class_table": len(classes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    records = [
        fit(args.root, args.out_dir / f"confidence-{variant}.tsv", variant)
        for variant in FEATURES
    ]
    with (args.out_dir / "confidence-models.json").open("x") as target:
        json.dump(records, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
