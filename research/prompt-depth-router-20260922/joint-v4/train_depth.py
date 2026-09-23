"""Fit a tiny accepted-prefix and cycle-cost model from randomized D."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np


CLASSES = ("word", "number", "space", "line", "symbol", "other")
FEATURES = {"token": 23, "history": 35, "prior": 37}


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def classify(raw):
    if b"\n" in raw or b"\r" in raw:
        return 3
    if raw and all(byte in b" \t\v\f" for byte in raw):
        return 2
    if any(48 <= byte <= 57 for byte in raw):
        return 1
    if any(65 <= byte <= 90 or 97 <= byte <= 122 for byte in raw):
        return 0
    if any(byte in b"`{}[]()=+-*/:;,.<>|" for byte in raw):
        return 4
    return 5


def features(history, prior, classes, variant):
    values = np.zeros(37, dtype=np.float64)
    values[0] = 1
    if history:
        token = history[-1]
        values[1 + classes.get(token, 5)] = 1
        values[7 + ((token * 0x9E3779B1 & 0xFFFFFFFF) % 16)] = 1
    if variant in ("history", "prior"):
        for offset, size in ((23, 4), (29, 16)):
            window = history[-size:]
            if window:
                for token in window:
                    values[offset + classes.get(token, 5)] += 1 / len(window)
    if variant == "prior" and prior is not None:
        values[35] = prior[0]
        values[36] = min(1, prior[1] / 80)
    return values[:FEATURES[variant]]


def samples(root, classes, variant):
    summary = json.loads((root / "training-summary.json").read_text())
    records = [
        row for row in summary["records"] if row["arm"] == "explore-d"
    ]
    if len(records) != 6:
        raise ValueError("training requires six randomized conversations")
    data = []
    for record in records:
        session = root / record["name"]
        spans = {
            (int(row["turn"]), int(row["round"])): row
            for row in table(session / "spans.tsv")
        }
        prior = None
        for turn in range(1, 9):
            prompt = [int(value) for value in (
                session / f"turn-{turn}.prompt.ids"
            ).read_text().split()]
            output = [int(value) for value in (
                session / f"turn-{turn}.output.ids"
            ).read_text().split()]
            for row in table(session / f"turn-{turn}.confidence.tsv"):
                key = turn, int(row["round"])
                span = spans[key]
                start = int(span["output_start"])
                d = int(row["drafted"])
                accepted = int(row["accepted_prefix"])
                emitted = int(row["emitted"])
                elapsed_ms = int(row["elapsed_ns"]) / 1e6
                if (
                    d not in (1, 2, 3, 4)
                    or row["eligible"] != span["eligible"]
                    or int(span["k"]) != d + 1
                    or int(span["elapsed_ns"]) != int(row["elapsed_ns"])
                ):
                    raise ValueError("sampled depth and accepted-prefix receipts disagree")
                if row["eligible"] == "true":
                    if (
                        start > len(output) or emitted != accepted + 1
                        or not 0 <= accepted <= d or elapsed_ms <= 0
                    ):
                        raise ValueError("eligible randomized D cannot train utility")
                    history = (prompt[-16:] + output[:start])[-16:]
                    data.append((
                        d, features(history, prior, classes, variant),
                        float(accepted), elapsed_ms,
                    ))
                    prior = accepted / d, elapsed_ms
    if not data:
        raise ValueError("randomized collector has no uncensored D labels")
    return data


def token_classes(root):
    summary = json.loads((root / "training-summary.json").read_text())
    mapping = {}
    for record in summary["records"]:
        if record["arm"] != "explore-d":
            continue
        for row in table(root / record["name"] / "token-bytes.tsv"):
            token = int(row["id"])
            klass = classify(bytes.fromhex(row["hex"]))
            if token in mapping and mapping[token] != klass:
                raise ValueError("token byte class changed across training conversations")
            mapping[token] = klass
    return mapping


def ridge(rows, target, alpha):
    x = np.stack([row[1] for row in rows])
    y = np.array([row[target] for row in rows], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite fitted depth weights")
    return weights


def fit(root, out, variant, alpha):
    if variant not in FEATURES or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("invalid frozen feature variant or ridge penalty")
    analysis = json.loads((root / "training-analysis.json").read_text())
    lam = analysis["fixed_baseline_tok_s"]
    classes = token_classes(root)
    data = samples(root, classes, variant)
    by_d = {d: [row for row in data if row[0] == d] for d in range(1, 5)}
    if any(len(rows) < 150 for rows in by_d.values()):
        raise ValueError("randomized depth lacks enough training exposure")
    lines = [f"joint-v4-linear\t1\t{variant}\t4\t{lam:.12g}\t{alpha:.12g}"]
    for d, rows in by_d.items():
        accepted = ridge(rows, 2, alpha)
        elapsed = ridge(rows, 3, alpha)
        lines.append(
            f"A\t{d}\t" + ",".join(f"{weight:.17g}" for weight in accepted)
        )
        lines.append(
            f"T\t{d}\t" + ",".join(f"{weight:.17g}" for weight in elapsed)
        )
    for token, klass in sorted(classes.items()):
        lines.append(f"B\t{token}\t{klass}")
    payload = ("\n".join(lines) + "\n").encode()
    with out.open("xb") as target:
        target.write(payload)
    return {
        "variant": variant, "alpha": alpha,
        "lambda_tok_s": lam,
        "training_rows": {str(d): len(rows) for d, rows in by_d.items()},
        "feature_count": FEATURES[variant],
        "class_table": len(classes),
        "model_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    records = []
    for variant in FEATURES:
        path = args.out_dir / f"depth-{variant}.tsv"
        records.append(fit(args.root, path, variant, 100))
    with (args.out_dir / "models.json").open("x") as target:
        json.dump(records, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
