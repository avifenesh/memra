"""Pin the v6 training parent without treating its unengaged C as v7 proof."""

import argparse
import hashlib
import json
from pathlib import Path


CONTROLS = (
    "development-summary.json",
    "model-training-summary.json",
    "topk-training-summary.json",
    "depth-grid-analysis.json",
    "fixed-confidence-candidates.json",
    "selected-fixed.json",
    "selection-summary.json",
    "selected-topk.json",
    "joint-selection-summary.json",
    "selected-joint.json",
)
COPIED = (
    "topk-training-summary.json",
    "fixed-confidence-candidates.json",
    "selected-fixed.json",
    "selected-topk.json",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ("parent-results", "parent-models", "results", "models"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent_results.resolve()
    models = args.models.resolve()
    if (parent / "heldout-summary.json").exists():
        raise ValueError("v6 training parent unexpectedly opened fresh heldout")
    joint = json.loads((parent / "joint-selection-summary.json").read_text())
    if sum(
        int(row["confidence_decisions"])
        for record in joint["records"] if record["variant"].startswith("joint-")
        for row in table(parent / record["name"] / "turns.tsv")
    ) != 0:
        raise ValueError("v6 training parent no longer has the recorded C engagement fault")
    control_hashes = {name: sha(parent / name) for name in CONTROLS}
    for name in COPIED:
        if sha(args.results / name) != control_hashes[name]:
            raise ValueError("copied v7 training control differs: " + name)
    model_hashes = {
        path.relative_to(args.parent_models).as_posix(): sha(path)
        for path in sorted(args.parent_models.rglob("*"))
        if path.is_file()
    }
    if not model_hashes or any(
        sha(models / name) != digest for name, digest in model_hashes.items()
    ):
        raise ValueError("copied v7 K/C/D weights differ from v6 parent")
    record = {
        "schema": 1,
        "scope": "v6 C0 and randomized K/D training parent only; learned C unengaged",
        "parent_source_sha256": "a391ff6337434e45acac4def721b816ccb4c6027ea93aad83cb90d40390c729a",
        "parent_binary_sha256": "07033ec249485f9da202f266d07d5432554a1f8b65ea6d14f55bee0387327c15",
        "control_sha256": control_hashes,
        "model_sha256": model_hashes,
    }
    with (args.results / "training-parent.json").open("x") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({"controls": len(control_hashes), "models": len(model_hashes)}))


def table(path):
    import csv
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


if __name__ == "__main__":
    main()
