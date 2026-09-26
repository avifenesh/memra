"""Measure first-token topic visibility inside training before K arm selection."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

import training_replay
import visibility


WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def report(root, workloads, archive_sha):
    if sha(workloads / "manifest.json") != WORKLOAD_SHA:
        raise ValueError("topic preflight uses another prompt split")
    manifest = json.loads((workloads / "manifest.json").read_text())
    prefixes, labels, counts = visibility.examples(
        root, manifest, "training"
    )
    if counts != {"ifeval": 128, "gsm8k": 128}:
        raise ValueError("topic preflight lacks balanced training turns")
    groups = np.repeat(np.arange(32), 8)
    if len(prefixes) != len(groups):
        raise ValueError("topic preflight group inventory differs")
    heldout = groups % 4 == 0
    result = {
        "schema": 1,
        "scope": "training-only conversation-heldout first-token visibility",
        "workload_sha256": WORKLOAD_SHA,
        "training_archive_sha256": archive_sha,
        "training_conversations_per_domain": 12,
        "heldout_conversations_per_domain": 4,
        "variants": {},
    }
    for variant in ("first16", "first32"):
        weights = visibility.fit(
            [prefixes[i] for i in np.flatnonzero(~heldout)],
            labels[~heldout],
            variant,
        )
        matrix = np.stack([
            visibility.fit_k.vector({
                "first32_user_ids": prefixes[i],
                "previous_turn_acceptance": None,
            }, variant)
            for i in np.flatnonzero(heldout)
        ])
        expected = labels[heldout]
        predicted = matrix @ weights >= 0.5
        result["variants"][variant] = {
            "heldout_accuracy": float(np.mean(predicted == expected)),
            "by_domain_accuracy": {
                domain: float(np.mean(predicted[expected == label] == label))
                for label, domain in enumerate(visibility.DOMAINS)
            },
        }
    return result


def sealed_report(archive, manifest_path, replay_path):
    manifest = json.loads(manifest_path.read_text())
    replay = json.loads(replay_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or manifest["workloads_sha256"] != WORKLOAD_SHA
        or replay["status"] != "training-native-random-K-D-C-KV-replay-match"
        or replay["archive_sha256"] != manifest["archive_sha256"]
        or replay["sessions"] != 224
    ):
        raise ValueError("prefix preflight lacks sealed randomized training")
    with tempfile.TemporaryDirectory(prefix="mtp-v11-visibility-") as temp:
        root = Path(temp)
        training_replay.extract(archive, manifest, root)
        return report(
            root / "native/training-results",
            root / "inputs/workloads",
            manifest["archive_sha256"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = sealed_report(args.archive, args.manifest, args.replay)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result["variants"], sort_keys=True))


if __name__ == "__main__":
    main()
