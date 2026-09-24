"""Fit C/D models with new GPU time costs and optional old acceptance data."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


V4 = Path(__file__).resolve().parent.parent / "joint-v4"
sys.path.insert(0, str(V4))
import train_depth
import train_confidence


def rows(path):
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source]


def token_classes(root, old_data):
    classes = {}
    for session in sorted(root.glob("training-*-explore-d")):
        path = session / "token-bytes.tsv"
        if not path.is_file():
            raise ValueError(f"missing tokenizer bytes: {session}")
        with path.open() as source:
            for row in csv.DictReader(source, delimiter="\t"):
                token = int(row["id"])
                klass = train_depth.classify(bytes.fromhex(row["hex"]))
                if token in classes and classes[token] != klass:
                    raise ValueError("token byte class differs across sessions")
                classes[token] = klass
    historical = json.loads((old_data / "token-classes.json").read_text())
    for token_text, klass in historical.items():
        token = int(token_text)
        if token in classes and classes[token] != klass:
            raise ValueError("new and historical token byte classes differ")
        classes[token] = klass
    if len(classes) < 1000:
        raise ValueError("too few token classes for C/D features")
    return classes


def x(row, classes, variant):
    prior = (
        None if row["previous_round_acceptance"] is None else (
            row["previous_round_acceptance"],
            row["previous_round_ms"],
        )
    )
    return train_depth.features(
        row["committed_history_ids"], prior, classes, variant
    )


def depth_weights(data, classes, variant, alpha):
    result = {}
    for d in range(1, 5):
        chosen = [row for row in data if row["draft_depth"] == d]
        if len(chosen) < 150:
            raise ValueError(f"D={d} lacks measured exposure")
        matrix = np.stack([x(row, classes, variant) for row in chosen])
        penalty = np.eye(matrix.shape[1]) * alpha
        penalty[0, 0] = 0
        accepted = np.array([row["accepted_prefix"] for row in chosen])
        result[d] = np.linalg.solve(
            matrix.T @ matrix + penalty, matrix.T @ accepted
        )
        if not np.isfinite(result[d]).all():
            raise ValueError(f"nonfinite D={d} acceptance weights")
    return result


def time_weights(data, classes, variant, alpha):
    result = {}
    for d in range(1, 5):
        chosen = [row for row in data if row["draft_depth"] == d]
        if len(chosen) < 150:
            raise ValueError(f"D={d} lacks new GPU time exposure")
        matrix = np.stack([x(row, classes, variant) for row in chosen])
        penalty = np.eye(matrix.shape[1]) * alpha
        penalty[0, 0] = 0
        elapsed = np.array([row["round_ms"] for row in chosen])
        result[d] = np.linalg.solve(
            matrix.T @ matrix + penalty, matrix.T @ elapsed
        )
        if not np.isfinite(result[d]).all():
            raise ValueError(f"nonfinite D={d} time weights")
    return result


def confidence_weights(data, classes, variant):
    result = {}
    for position in range(3):
        chosen = [
            row for row in data if row["offer_position"] == position
        ]
        if len(chosen) < 150:
            raise ValueError(f"C position {position} lacks observed labels")
        samples = [
            (
                np.concatenate((
                    x(row, classes, variant),
                    [train_confidence.q_logit(row["chosen_probability"])],
                )),
                float(row["accepted_offer"]),
            )
            for row in chosen
        ]
        result[position] = train_confidence.logistic(samples, 20)
    return result


def format_weights(weights):
    return ",".join(f"{value:.17g}" for value in weights)


def build(new_data, old_data, native, out):
    new_d = rows(new_data / "d.jsonl.gz")
    new_c = rows(new_data / "c.jsonl.gz")
    old_d = rows(old_data / "d.jsonl.gz")
    old_c = rows(old_data / "c.jsonl.gz")
    if not new_d or not new_c or not old_d or not old_c:
        raise ValueError("new or historical C/D observation set is empty")
    classes = token_classes(native, old_data)
    new_manifest = json.loads((new_data / "manifest.json").read_text())
    included = set(new_manifest["included_training_topics"])
    fixed = [
        json.loads(path.read_text())
        for path in native.glob("training-*-k20-fixed-d3.result.json")
        if int(path.name.split("-")[1]) in included
    ]
    if len(fixed) != len(included) or len(included) < 20 or any(row["loops"] for row in fixed):
        raise ValueError("new GPU K20 timing baseline incomplete")
    lam = sum(row["tokens"] for row in fixed) / sum(
        row["seconds"] for row in fixed
    )
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": 1,
        "reference_tok_s_new_gpu": lam,
        "time_training": "new-randomized-only",
        "acceptance_training": {
            "new-only": ["v9 randomized"],
            "augmented": ["v9 randomized", "v6 randomized", "v8 realized D"],
        },
        "models": [],
    }
    for source in ("new-only", "augmented"):
        base = out / source
        base.mkdir()
        for k in (3, 10, 20):
            dest = base / f"topk{k}"
            dest.mkdir()
            current_d = [row for row in new_d if row["draft_k"] == k]
            current_c = [row for row in new_c if row["draft_k"] == k]
            acceptance_d = current_d if source == "new-only" else (
                current_d + [row for row in old_d if row["draft_k"] == k]
            )
            acceptance_c = current_c if source == "new-only" else (
                current_c + [row for row in old_c if row["draft_k"] == k]
            )
            means = {
                d: np.mean([
                    row["round_ms"] for row in current_d
                    if row["draft_depth"] == d
                ])
                for d in range(1, 5)
            }
            if not all(np.isfinite(value) for value in means.values()):
                raise ValueError("missing randomized time cost for a D action")
            for variant in train_depth.FEATURES:
                accepted = depth_weights(acceptance_d, classes, variant, 100)
                elapsed = time_weights(current_d, classes, variant, 100)
                depth_lines = [
                    f"joint-v4-linear\t1\t{variant}\t4\t{lam:.12g}\t100"
                ]
                for d in range(1, 5):
                    depth_lines.append(f"A\t{d}\t{format_weights(accepted[d])}")
                    depth_lines.append(f"T\t{d}\t{format_weights(elapsed[d])}")
                confidence = confidence_weights(
                    acceptance_c, classes, variant
                )
                confidence_lines = [
                    f"joint-v4-confidence\t1\t{variant}\t{lam:.12g}"
                ]
                for position in range(3):
                    next_rows = [
                        row for row in acceptance_c
                        if row["offer_position"] == position + 1
                    ]
                    next_rate = sum(
                        row["accepted_offer"] for row in next_rows
                    ) / len(next_rows)
                    marginal = means[position + 2] - means[position + 1]
                    confidence_lines.append(
                        f"C\t{position}\t{next_rate:.17g}\t{marginal:.17g}\t"
                        f"{format_weights(confidence[position])}"
                    )
                for token, klass in sorted(classes.items()):
                    line = f"B\t{token}\t{klass}"
                    depth_lines.append(line)
                    confidence_lines.append(line)
                for kind, lines in (
                    ("depth", depth_lines),
                    ("confidence", confidence_lines),
                ):
                    payload = ("\n".join(lines) + "\n").encode()
                    path = dest / f"{kind}-{variant}.tsv"
                    path.write_bytes(payload)
                    summary["models"].append({
                        "source": source,
                        "k": k,
                        "kind": kind,
                        "variant": variant,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    })
    (out / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.new, args.old, args.native, args.out)
    print(json.dumps({
        "reference_tok_s_new_gpu": result["reference_tok_s_new_gpu"],
        "models": len(result["models"]),
    }))


if __name__ == "__main__":
    main()
