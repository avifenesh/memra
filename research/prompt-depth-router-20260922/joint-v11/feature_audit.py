"""Check whether token history adds C-label signal beyond proposal q."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

V4 = Path(__file__).resolve().parent.parent / "joint-v4"
sys.path.insert(0, str(V4))
import train_confidence
import train_depth


DOMAINS = ("ifeval", "gsm8k")
VARIANTS = (
    "q-only", "token", "history", "prior",
    "prompt16", "prompt32", "prompt-history",
)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def prompt_features(ids, count):
    if not 1 <= len(ids) <= 32:
        raise ValueError("C prompt feature lacks bounded tokenizer input")
    prefix = ids[:count]
    values = np.zeros(10, dtype=np.float64)
    values[0] = 1
    for token in prefix:
        bucket = ((token * 0x9E3779B1) & 0xFFFFFFFF) % 8
        values[1 + bucket] += 1 / len(prefix)
    values[9] = len(prefix) / count
    return values


def features(row, classes, variant):
    q = train_confidence.q_logit(row["chosen_probability"])
    if variant == "q-only":
        return np.array([1.0, q])
    previous = (
        None if row["previous_round_acceptance"] is None
        else (
            row["previous_round_acceptance"],
            row["previous_round_ms"],
        )
    )
    if variant in ("prompt16", "prompt32", "prompt-history"):
        prefix = prompt_features(
            row["first32_user_ids"],
            16 if variant == "prompt16" else 32,
        )
        if variant == "prompt-history":
            history = train_depth.features(
                row["committed_history_ids"], previous, classes, "history"
            )
            return np.concatenate((prefix, history, [q]))
        return np.concatenate((prefix, [q]))
    base = train_depth.features(
        row["committed_history_ids"], previous, classes, variant
    )
    return np.concatenate((base, [q]))


def heldout_group(row):
    # The first split component after "training-" is the frozen global
    # conversation index from the v11 collector.
    name = row["conversation"]
    if not name.startswith("training-"):
        raise ValueError("C label does not name a v11 training conversation")
    index = int(name.split("-")[1])
    if not 0 <= index < 32:
        raise ValueError("v11 training group index differs")
    return index % 4 == 0


def load(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != 1 or manifest["use"] != "training-only":
        raise ValueError("C visibility source is not training-only")
    path = root / "c.jsonl.gz"
    if sha(path) != manifest["rows"]["c"]["sha256"]:
        raise ValueError("training C labels differ")
    classes = {
        int(token): kind
        for token, kind in json.loads(
            (root / "token-classes.json").read_text()
        ).items()
    }
    if sha(root / "token-classes.json") != manifest["token_classes_sha256"]:
        raise ValueError("C feature byte classes changed")
    grouped = defaultdict(lambda: {"train": [], "test": []})
    with gzip.open(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            if row["source"] not in ("v11-ifeval", "v11-gsm8k"):
                continue
            if row["domain"] not in DOMAINS:
                raise ValueError("fresh C label topic differs")
            if not 1 <= len(row.get("first32_user_ids", [])) <= 32:
                raise ValueError("fresh C label lacks user-token prefix")
            position = row["offer_position"]
            k = row["draft_k"]
            if position not in (0, 1, 2) or k not in (3, 10, 20):
                continue
            split = "test" if heldout_group(row) else "train"
            grouped[k, position][split].append(row)
    if any(
        len(group[split]) < 150
        for group in grouped.values()
        for split in ("train", "test")
    ) or set(grouped) != {
        (k, position) for k in (3, 10, 20) for position in (0, 1, 2)
    }:
        raise ValueError("too few conversation-disjoint C labels")
    return grouped, classes, sha(path)


def fit(rows, classes, variant):
    samples = [
        (
            features(row, classes, variant),
            float(row["accepted_offer"]),
        )
        for row in rows
    ]
    return train_confidence.logistic(samples, 20)


def losses(rows, classes, variant, weights):
    metrics = {
        domain: {"count": 0, "log_loss": 0.0, "brier": 0.0}
        for domain in DOMAINS
    }
    for row in rows:
        z = float(features(row, classes, variant) @ weights)
        chance = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
        chance = max(1e-9, min(1 - 1e-9, chance))
        label = float(row["accepted_offer"])
        item = metrics[row["domain"]]
        item["count"] += 1
        item["log_loss"] += -(
            label * math.log(chance)
            + (1 - label) * math.log(1 - chance)
        )
        item["brier"] += (label - chance) ** 2
    for item in metrics.values():
        if item["count"] == 0:
            raise ValueError("topic lacks C calibration labels")
        item["log_loss"] /= item["count"]
        item["brier"] /= item["count"]
    return metrics


def audit(root):
    grouped, classes, source_sha = load(root)
    result = {
        "schema": 1,
        "scope": "training-only per-K conditional C calibration, not native tok/s",
        "source_c_sha256": source_sha,
        "positions": {},
    }
    for (k, position), split in sorted(grouped.items()):
        models = {}
        for variant in VARIANTS:
            weights = fit(split["train"], classes, variant)
            models[variant] = {
                "weights": [float(value) for value in weights],
                "test": losses(split["test"], classes, variant, weights),
            }
            print(json.dumps({
                "calibration_k": k, "offer_position": position,
                "variant": variant,
            }, sort_keys=True), flush=True)
        baseline = models["q-only"]["test"]
        for variant in VARIANTS[1:]:
            for domain in DOMAINS:
                models[variant]["test"][domain]["log_loss_delta_vs_q_only"] = (
                    models[variant]["test"][domain]["log_loss"]
                    - baseline[domain]["log_loss"]
                )
        result["positions"][f"K{k}:P{position}"] = {
            "training_labels": len(split["train"]),
            "heldout_labels": len(split["test"]),
            "models": models,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rows)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        position: {
            variant: {
                domain: metrics["log_loss"]
                for domain, metrics in model["test"].items()
            }
            for variant, model in group["models"].items()
        }
        for position, group in result["positions"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
