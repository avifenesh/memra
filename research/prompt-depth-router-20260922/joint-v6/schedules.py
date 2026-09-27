"""Freeze draft-only top-k assignments before any v6 model output."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import secrets


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def schedule(seed, actions):
    rng = random.Random(seed)
    result = []
    while len(result) < 8:
        batch = list(actions)
        rng.shuffle(batch)
        result.extend(batch)
    return result[:8]


def freeze(manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    topics = manifest["groups"]["calibration"] + manifest["groups"]["heldout"][:3]
    if len(topics) != 6:
        raise ValueError("K exploration requires six distinct training topics")
    rows = []
    for index, topic in enumerate(topics):
        seed = secrets.randbits(64)
        rows.append({
            "training_index": index,
            "topic": topic["topic"],
            "workload_sha256": topic["sha256"],
            "schedule_seed": seed,
            "topk_3_10_20": schedule(seed, (3, 10, 20)),
            "topk_3_20": schedule(seed, (3, 20)),
            "topk_10_20": schedule(seed, (10, 20)),
        })
    record = {
        "schema": 1,
        "scope": "independent per-turn MTP draft K assignments, frozen before v6 GPU output",
        "workloads_sha256": sha(manifest_path),
        "generator_sha256": sha(Path(__file__)),
        "training": rows,
    }
    with out.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.manifest, args.out)
    print(json.dumps({"training_conversations": len(result["training"])}))


if __name__ == "__main__":
    main()
