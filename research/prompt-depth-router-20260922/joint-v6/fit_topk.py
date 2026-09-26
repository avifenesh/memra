"""Fit first-user-token-prefix MTP draft K from randomized turns."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


FEATURES = {"first16": 10, "first32": 10, "prior": 11}
TOP_K = (3, 10, 20)


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def features(prefix, previous, variant):
    count = 16 if variant == "first16" else 32
    values = np.zeros(FEATURES[variant], dtype=np.float64)
    values[0] = 1
    window = prefix[:count]
    if window:
        for token in window:
            bucket = ((token * 0x9E3779B1) & 0xFFFFFFFF) % 8
            values[1 + bucket] += 1 / len(window)
    values[9] = len(window) / count
    if variant == "prior" and previous is not None:
        values[10] = previous
    return values


def eligible_fixed(summary, qualifier):
    allowed = {}
    for k in TOP_K:
        rows = [
            row for row in summary["records"]
            if row["draft_top_k"] == k and row["arm"] == "fixed:3"
        ]
        if len(rows) != 6:
            raise ValueError("fixed top-k development control is incomplete")
        tokens = sum(row["tokens"] for row in rows)
        seconds = sum(row["seconds"] for row in rows)
        allowed[str(k)] = {
            "qualifier_pass": bool(qualifier["eligible"][str(k)]),
            "format_pass": sum(row["format"] for row in rows),
            "loops": sum(row["loops"] for row in rows),
            "tokens": tokens, "seconds": seconds,
            "tok_s": tokens / seconds,
        }
    return allowed


def randomized_rows(root, summary, variant, allowed):
    data = defaultdict(list)
    scheduled = [
        row for row in summary["records"] if row["variant"] == "random-draftk"
    ]
    if len(scheduled) != 6 and len(allowed) > 1:
        raise ValueError("K training lacks six randomized conversations")
    for record in scheduled:
        if record["loops"]:
            continue
        session = root / record["name"]
        previous = None
        turns = table(session / "turns.tsv")
        if len(turns) != 8:
            raise ValueError("randomized K session is not eight continuing turns")
        for turn, row in enumerate(turns, 1):
            k = int(row["draft_top_k"])
            prefix = [
                int(value) for value in (
                    session / f"turn-{turn}.user-prefix.ids"
                ).read_text().split()
            ]
            if not 1 <= len(prefix) <= 32:
                raise ValueError("K model prefix is empty or unbounded")
            if k in allowed:
                data[k].append((
                    features(prefix, previous, variant),
                    int(row["output_tokens"]),
                    float(row["elapsed_s"]),
                ))
            drafted = int(row["drafted"])
            previous = int(row["accepted"]) / drafted if drafted else None
    return data


def ridge(rows, lam, alpha):
    x = np.stack([row[0] for row in rows])
    y = np.array([
        tokens - lam * seconds for _, tokens, seconds in rows
    ], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite tiny K router weights")
    return weights


def fit(root, models):
    summary = json.loads((root / "development-summary.json").read_text())
    qualifier = json.loads((root / "qualification-result.json").read_text())
    if summary["schema"] != 1 or qualifier["schema"] != 1:
        raise ValueError("another K development schema")
    fixed = eligible_fixed(summary, qualifier)
    allowed = [
        k for k in TOP_K
        if fixed[str(k)]["qualifier_pass"]
        and fixed[str(k)]["format_pass"] == 48
        and fixed[str(k)]["loops"] == 0
    ]
    if 20 not in allowed:
        raise ValueError("quality-eligible recommended K=20 is required")
    lam = max(fixed[str(k)]["tok_s"] for k in allowed)
    models.mkdir(exist_ok=True)
    fallback = (
        f"joint-v6-draft-topk\t1\tfirst16\t{lam:.12g}\t20\n"
        "K\t20\t0,0,0,0,0,0,0,0,0,0\n"
    ).encode()
    with (models / "topk-fixed20.tsv").open("xb") as target:
        target.write(fallback)
    records = []
    for variant in FEATURES:
        data = randomized_rows(root, summary, variant, set(allowed))
        if len(allowed) > 1 and any(len(data[k]) < 8 for k in allowed):
            raise ValueError("randomized K action lacks enough input-prefix examples")
        lines = [f"joint-v6-draft-topk\t1\t{variant}\t{lam:.12g}\t20"]
        for k in allowed:
            weights = (
                ridge(data[k], lam, 20)
                if len(allowed) > 1 else np.zeros(FEATURES[variant])
            )
            lines.append(
                f"K\t{k}\t" + ",".join(f"{weight:.17g}" for weight in weights)
            )
        payload = ("\n".join(lines) + "\n").encode()
        with (models / f"topk-{variant}.tsv").open("xb") as target:
            target.write(payload)
        records.append({
            "variant": variant,
            "allowed_top_k": allowed,
            "sampled_turns": {str(k): len(data[k]) for k in allowed},
            "lambda_tok_s": lam,
            "model_sha256": hashlib.sha256(payload).hexdigest(),
        })
    save(models / "topk-models.json", records)
    save(root / "topk-training-summary.json", {
        "schema": 1,
        "scope": "draft-only K from first bounded user tokenizer prefix; target top-k=20",
        "fixed": fixed, "quality_eligible_top_k": allowed,
        "fixed20_model_sha256": hashlib.sha256(fallback).hexdigest(),
        "models": records,
    })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.root, args.models)))


if __name__ == "__main__":
    main()
