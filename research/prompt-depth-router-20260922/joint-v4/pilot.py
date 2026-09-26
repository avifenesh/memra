"""Development-only test of token/history features for accepted-prefix calibration.

This does not estimate counterfactual E2E throughput. The source archive's
calibration and C=0 monitor traces are both development data for v4.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import tarfile


ARMS = ("q", "token", "history", "prior_round")
CLASSES = ("word", "number", "space", "line", "symbol", "other")
ARCHIVE_SHA = "8631202585850659769481529485f89a53f78381a06fcce67c2d7bdd1f10fb9c"
WORKLOADS_SHA = "00a1b6d97fbae3ecf69b46de7097cf6dd0639ca7aed60ed84585d17a2a8bfdef"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(data):
    return list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))


def ids(data):
    return [int(value) for value in data.split()]


def category(raw):
    if not raw:
        return "other"
    text = raw.decode("utf-8", errors="ignore")
    if "\n" in text or "\r" in text:
        return "line"
    if text.isspace():
        return "space"
    if any(ch.isdigit() for ch in text):
        return "number"
    if any(ch.isalpha() for ch in text):
        return "word"
    if any(ch in "`{}[]()=+-*/:;,.<>|" for ch in text):
        return "symbol"
    return "other"


def probability(bits):
    value = struct.unpack("<f", struct.pack("<I", int(bits)))[0]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("recorded chosen probability is outside [0, 1]")
    return value


def make_labels(q, accepted, eligible):
    if not eligible:
        return []
    if not 0 <= accepted <= len(q):
        raise ValueError("accepted prefix differs from offered proposals")
    return [
        (position, value, int(accepted > position))
        for position, value in enumerate(q)
        if accepted >= position
    ]


class Archive:
    def __init__(self, path):
        self.archive = tarfile.open(path, "r:gz")
        self.names = set(self.archive.getnames())

    def read(self, name):
        if name not in self.names:
            raise ValueError("missing pinned native record: " + name)
        member = self.archive.getmember(name)
        if not member.isfile():
            raise ValueError("native record is not a regular file")
        return self.archive.extractfile(member).read()

    def close(self):
        self.archive.close()


def session(archive, root, topic):
    byte_rows = table(archive.read(f"{root}/token-bytes.tsv"))
    token_bytes = {int(row["id"]): bytes.fromhex(row["hex"]) for row in byte_rows}
    spans = table(archive.read(f"{root}/spans.tsv"))
    span_map = {(int(row["turn"]), int(row["round"])): row for row in spans}
    samples = []
    seen = set()
    for turn in range(1, 9):
        prompt = ids(archive.read(f"{root}/turn-{turn}.prompt.ids"))
        output = ids(archive.read(f"{root}/turn-{turn}.output.ids"))
        rounds = table(archive.read(f"{root}/turn-{turn}.confidence.tsv"))
        previous = None
        for row in rounds:
            number = int(row["round"])
            span = span_map[(turn, number)]
            seen.add((turn, number))
            drafted = int(row["drafted"])
            accepted = int(row["accepted_prefix"])
            q = [probability(bits) for bits in row["q_bits"].split(",")]
            eligible = row["eligible"] == "true"
            if (
                len(q) != drafted or drafted != 3
                or span["eligible"] != row["eligible"]
                or int(span["elapsed_ns"]) != int(row["elapsed_ns"])
            ):
                raise ValueError("K=3 confidence and native span receipts differ")
            start = int(span["output_start"])
            if start < 0 or start > len(output):
                if eligible:
                    raise ValueError("eligible round refers to an unavailable token prefix")
                previous = row
                continue
            history = (prompt[-16:] + output[:start])[-16:]
            for position, value, label in make_labels(q, accepted, eligible):
                samples.append({
                    "session": root, "topic": topic,
                    "turn": turn, "round": number, "position": position,
                    "q": value, "label": label,
                    "last": history[-1] if history else -1,
                    "history": history,
                    "prior_accept": (
                        int(previous["accepted_prefix"]) / int(previous["drafted"])
                        if previous else None
                    ),
                    "prior_ms": (
                        int(previous["elapsed_ns"]) / 1e6 if previous else None
                    ),
                    "phase": span["context_before"],
                })
            previous = row
    if seen != set(span_map):
        raise ValueError("native span and offered-prefix round inventories differ")
    if not samples:
        raise ValueError("uncensored session supplied no eligible acceptance labels")
    return samples


def top_tokens(samples):
    counts = Counter(row["last"] for row in samples if row["last"] >= 0)
    return [token for token, _ in sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    )[:32]]


def features(row, variant, token_bytes, top_ids):
    q = min(1 - 1e-6, max(1e-6, row["q"]))
    out = {
        "bias": 1.0,
        "q_logit": max(-12.0, min(12.0, math.log(q / (1 - q)))) / 12,
        f"position:{row['position']}": 1.0,
    }
    if variant == "q":
        return out
    last = row["last"]
    out[f"last:{last if last in top_ids else 'other'}"] = 1.0
    out[f"last_class:{category(token_bytes.get(last, b''))}"] = 1.0
    if variant == "token":
        return out
    for size in (4, 16):
        window = row["history"][-size:]
        if not window:
            continue
        counts = Counter(category(token_bytes.get(token, b"")) for token in window)
        for kind in CLASSES:
            out[f"prev{size}:{kind}"] = counts[kind] / len(window)
    if variant == "history":
        return out
    if row["prior_accept"] is not None:
        out["prior_accept"] = row["prior_accept"]
        out["prior_ms"] = min(4.0, row["prior_ms"] / 20.0) / 4.0
    return out


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def fit(rows, variant, token_bytes, top_ids):
    weights = defaultdict(float)
    rng = random.Random(20774001)
    for epoch in range(12):
        order = list(range(len(rows)))
        rng.shuffle(order)
        rate = 0.035 / (1 + epoch / 4)
        for index in order:
            row = rows[index]
            feature = features(row, variant, token_bytes, top_ids)
            score = sum(weights[name] * value for name, value in feature.items())
            error = row["label"] - sigmoid(score)
            for name, value in feature.items():
                weights[name] += rate * (error * value - 1e-5 * weights[name])
    return dict(weights)


def predict(weights, row, variant, token_bytes, top_ids):
    return min(1 - 1e-9, max(1e-9, sigmoid(sum(
        weights.get(name, 0.0) * value
        for name, value in features(row, variant, token_bytes, top_ids).items()
    ))))


def loss(rows, weights, variant, token_bytes, top_ids):
    logloss = brier = 0.0
    by_session = defaultdict(list)
    by_phase = defaultdict(list)
    for row in rows:
        chance = predict(weights, row, variant, token_bytes, top_ids)
        label = row["label"]
        logloss += -(label * math.log(chance) + (1 - label) * math.log1p(-chance))
        brier += (chance - label) ** 2
        by_session[row["session"]].append((chance, label))
        by_phase[row["phase"]].append((chance, label))
    def rows_score(pairs):
        return {
            "labels": len(pairs),
            "logloss": sum(
                -(y * math.log(p) + (1 - y) * math.log1p(-p)) for p, y in pairs
            ) / len(pairs),
            "brier": sum((p - y) ** 2 for p, y in pairs) / len(pairs),
            "acceptance_rate": sum(y for _, y in pairs) / len(pairs),
        }
    return {
        "labels": len(rows), "logloss": logloss / len(rows),
        "brier": brier / len(rows),
        "by_session": {name: rows_score(data) for name, data in by_session.items()},
        "by_output_phase": {name: rows_score(data) for name, data in by_phase.items()},
    }


def study(archive_path, workload_path):
    if digest(archive_path) != ARCHIVE_SHA or digest(workload_path) != WORKLOADS_SHA:
        raise ValueError("observational pilot input differs from pinned v3 development records")
    manifest = json.loads(Path(workload_path).read_text())
    archive = Archive(archive_path)
    try:
        training, evaluation = [], []
        token_bytes = {}
        for index, item in enumerate(manifest["groups"]["calibration"]):
            root = f"native/calibration-{index}-trace-c3"
            training.extend(session(archive, root, item["topic"]))
            token_bytes.update({
                int(row["id"]): bytes.fromhex(row["hex"]) for row in
                table(archive.read(f"{root}/token-bytes.tsv"))
            })
        for index, item in enumerate(manifest["groups"]["heldout"]):
            root = f"native/heldout-{index}-monitor-c3"
            evaluation.extend(session(archive, root, item["topic"]))
            token_bytes.update({
                int(row["id"]): bytes.fromhex(row["hex"]) for row in
                table(archive.read(f"{root}/token-bytes.tsv"))
            })
    finally:
        archive.close()
    if {row["session"] for row in training} & {row["session"] for row in evaluation}:
        raise ValueError("pilot training and evaluation reuse a conversation")
    top_ids = top_tokens(training)
    models = {}
    for variant in ARMS:
        weights = fit(training, variant, token_bytes, top_ids)
        models[variant] = {
            "weights": weights,
            "train": loss(training, weights, variant, token_bytes, top_ids),
            "evaluation": loss(evaluation, weights, variant, token_bytes, top_ids),
        }
    return {
        "schema": 1,
        "status": "observational acceptance-feature pilot; no E2E result",
        "archive_sha256": digest(archive_path),
        "workloads_sha256": digest(workload_path),
        "pilot_sha256": digest(Path(__file__)),
        "split": {
            "training": sorted({row["session"] for row in training}),
            "evaluation": sorted({row["session"] for row in evaluation}),
        },
        "top_training_token_ids": top_ids,
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to replace a prior pilot")
    result = study(args.archive, args.workloads)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "training_labels": result["models"]["q"]["train"]["labels"],
        "evaluation_labels": result["models"]["q"]["evaluation"]["labels"],
        "eval_logloss": {
            name: model["evaluation"]["logloss"] for name, model in result["models"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
