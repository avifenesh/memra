"""Check training-only task visibility in the controller's first 16/32 tokens."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v11"))
import fit_mixed as previous
sys.path.insert(0, str(BASE / "universal-v12"))
import fit_shared


DOMAINS = ("code", "prose", "math")
SPLIT_SEED = "mtp-v12-training-prefix"


def vector(row, length):
    values = previous.np.zeros(10, dtype=previous.np.float64)
    values[0] = 1
    prefix = row["first32_user_ids"][:length]
    if not prefix:
        raise ValueError("training-only prefix is empty")
    for token in prefix:
        bucket = ((token * 0x9E3779B1) & 0xFFFFFFFF) % 8
        values[1 + bucket] += 1 / len(prefix)
    values[9] = len(prefix) / length
    return values


def conversations(rows, domain):
    by_session = {}
    for row in rows:
        if (
            row["draft_k"] != 20
            or row["assignment"] != "fixed-arm"
        ):
            continue
        name = row["conversation"]
        turns = by_session.setdefault(name, {})
        prior = turns.get(row["turn"])
        if prior is not None and (
            prior["first32_user_ids"] != row["first32_user_ids"]
        ):
            raise ValueError(f"{domain} duplicate prefix differs")
        turns[row["turn"]] = row
    if len(by_session) < 16 or any(
        len(turns) != 8 for turns in by_session.values()
    ):
        raise ValueError(f"{domain} prefix preflight lacks whole training sessions")
    ordered = sorted(
        by_session,
        key=lambda name: hashlib.sha256(
            (SPLIT_SEED + domain + name).encode()
        ).hexdigest(),
    )
    heldout = set(ordered[:4])
    return (
        [
            row for session in ordered if session not in heldout
            for _, row in sorted(by_session[session].items())
        ],
        [
            row for session in ordered if session in heldout
            for _, row in sorted(by_session[session].items())
        ],
        sorted(heldout),
    )


def experiment(rows, length):
    train = []
    heldout = []
    names = {}
    for domain_index, domain in enumerate(DOMAINS):
        training, final, sessions = conversations(rows[domain], domain)
        names[domain] = sessions
        train.extend((vector(row, length), domain_index)
                     for row in training)
        heldout.extend((vector(row, length), domain_index)
                       for row in final)
    x = previous.np.stack([item[0] for item in train])
    y = previous.np.eye(len(DOMAINS))[
        [item[1] for item in train]
    ]
    penalty = previous.np.eye(x.shape[1]) * 0.1
    penalty[0, 0] = 0
    weights = previous.np.linalg.solve(
        x.T @ x + penalty, x.T @ y,
    )
    if not previous.np.isfinite(weights).all():
        raise ValueError("prefix visibility model has nonfinite weights")
    confusion = {
        domain: Counter() for domain in DOMAINS
    }
    for features, expected in heldout:
        predicted = int(previous.np.argmax(features @ weights))
        confusion[DOMAINS[expected]][DOMAINS[predicted]] += 1
    recalls = {
        domain: confusion[domain][domain]
        / sum(confusion[domain].values())
        for domain in DOMAINS
    }
    return {
        "first_tokens": length,
        "training_examples": len(train),
        "heldout_examples": len(heldout),
        "heldout_conversations": names,
        "confusion": {
            domain: dict(sorted(confusion[domain].items()))
            for domain in DOMAINS
        },
        "recall": recalls,
        "balanced_accuracy":
        sum(recalls.values()) / len(recalls),
    }


def build(fresh_rows, fresh_replay):
    fresh, _, _ = fit_shared.read_fresh(
        fresh_rows, fresh_replay,
    )
    rows = {
        domain: [
            row for row in fresh["k"]
            if row["domain"] == domain
        ]
        for domain in DOMAINS
    }
    results = {
        str(length): experiment(rows, length)
        for length in (16, 32)
    }
    visible = any(
        item["balanced_accuracy"] >= 0.6
        and min(item["recall"].values()) >= 0.5
        for item in results.values()
    )
    return {
        "schema": 1,
        "scope": "training-only conversation-heldout prefix visibility",
        "status": "training-prefix-visible" if visible
        else "training-prefix-not-visible",
        "no_field_route": True,
        "v12_fresh_training_manifest_sha256":
        previous.sha(fresh_rows / "manifest.json"),
        "v12_fresh_training_replay_sha256":
        previous.sha(fresh_replay),
        "experiments": results,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("fresh-rows", "fresh-replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.fresh_rows, args.fresh_replay)
    previous.save(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "balanced_accuracy": {
            key: value["balanced_accuracy"]
            for key, value in result["experiments"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
