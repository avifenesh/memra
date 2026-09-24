"""Fit new-GPU draft K controllers with and without old measured turns."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from fit_k import ACTIONS, FEATURE_COUNT, vector


def rows(path):
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source]


def fit(data, variant, alpha, reference):
    weights = {}
    for k in ACTIONS:
        selected = [row for row in data if row["draft_k"] == k]
        if len(selected) < 128:
            raise ValueError(f"K={k} lacks new-GPU training coverage")
        matrix = np.stack([vector(row, variant) for row in selected])
        utility = np.array([
            row["output_tokens"]
            - reference[row["source"]] * row["complete_request_seconds"]
            for row in selected
        ])
        penalty = np.eye(FEATURE_COUNT[variant]) * alpha
        penalty[0, 0] = 0
        weights[k] = np.linalg.solve(
            matrix.T @ matrix + penalty, matrix.T @ utility
        )
        if not np.isfinite(weights[k]).all():
            raise ValueError(f"K={k} model has nonfinite weights")
    return weights


def train(new, old, out):
    current = rows(new / "k.jsonl.gz")
    historical = rows(old / "k.jsonl.gz")
    new_manifest = json.loads((new / "manifest.json").read_text())
    topics = new_manifest["conversation_count"]
    if (
        not 20 <= topics <= 24
        or len(current) != topics * 8 * 3 or len(historical) != 192
        or {row["source"] for row in current} != {"v9"}
    ):
        raise ValueError("new or historical measured K inventory differs")
    fixed20 = [
        row for row in current if row["draft_k"] == 20
    ]
    if len(fixed20) != topics * 8:
        raise ValueError("new K20 reference is incomplete")
    new_lam = sum(row["output_tokens"] for row in fixed20) / sum(
        row["complete_request_seconds"] for row in fixed20
    )
    old_manifest = json.loads((old / "manifest.json").read_text())
    if old_manifest["use"] != "training-only":
        raise ValueError("old observations have another provenance")
    old_fixed20 = [
        row for row in historical
        if row["source"] == "v8" and row["draft_k"] == 20
    ]
    old_lam = sum(row["output_tokens"] for row in old_fixed20) / sum(
        row["complete_request_seconds"] for row in old_fixed20
    )
    references = {"v6": old_lam, "v8": old_lam, "v9": new_lam}
    out.mkdir(parents=True, exist_ok=False)
    models = []
    for source in ("new-only", "augmented"):
        data = current if source == "new-only" else current + historical
        for alpha in (20, 100):
            directory = out / source / f"ridge-{alpha}"
            directory.mkdir(parents=True, exist_ok=True)
            for variant in FEATURE_COUNT:
                weights = fit(data, variant, alpha, references)
                lines = [f"joint-v6-draft-topk\t1\t{variant}\t{new_lam:.12g}\t20"]
                for k in ACTIONS:
                    lines.append(
                        f"K\t{k}\t" +
                        ",".join(f"{value:.17g}" for value in weights[k])
                    )
                payload = ("\n".join(lines) + "\n").encode()
                path = directory / f"topk-{variant}.tsv"
                path.write_bytes(payload)
                chosen = Counter(
                    max(
                        ACTIONS,
                        key=lambda k: (float(vector(row, variant) @ weights[k]), k)
                    )
                    for row in current
                )
                models.append({
                    "source": source,
                    "alpha": alpha,
                    "variant": variant,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "training_only_actions": dict(sorted(chosen.items())),
                })
    manifest = {
        "schema": 1,
        "references_tok_s": references,
        "new_turns": len(current),
        "old_turns": len(historical),
        "models": models,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.new, args.old, args.out)
    print(json.dumps({
        "new_turns": result["new_turns"],
        "old_turns": result["old_turns"],
        "new_reference_tok_s": result["references_tok_s"]["v9"],
        "models": len(result["models"]),
    }))


if __name__ == "__main__":
    main()
