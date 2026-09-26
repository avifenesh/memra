"""Freeze code-trained C/K/D transfer arms against exact model files."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm(label, k, action, cap, *extra):
    return {
        "label": label,
        "k": k,
        "arm": action,
        "cap": cap,
        "extra": list(extra),
    }


def freeze(models, out):
    models = models.resolve()
    source = models / "manifest.json"
    manifest = json.loads(source.read_text())
    if (
        manifest["schema"] != 1
        or manifest["source_archive_sha256"]
        != "a914e20a4f823acbdf189202806415785d60507fa0f775ab588b10083d3c034b"
        or len(manifest["files"]) != 9
    ):
        raise ValueError("code-trained controller provenance differs")
    for relative, expected in manifest["files"].items():
        if sha(models / relative) != expected:
            raise ValueError(f"controller model changed: {relative}")
    router = models / "k-models/augmented/ridge-100/topk-prior.tsv"
    cd_new = models / "cd-models/new-only/topk20"
    cd_joint = models / "cd-models/augmented"
    rows = [
        arm(f"fixed-k{k}-d3-c0", k, "fixed:3", 3)
        for k in (3, 10, 20)
    ]
    rows.extend([
        arm("fixed-k20-d4-c0", 20, "fixed:4", 4),
        arm(
            "fixed-k20-c-low", 20, "fixed-c3", 3,
            "confidence-fixed=0,0.50844276",
        ),
        arm(
            "fixed-k20-c-mid", 20, "fixed-c3", 3,
            "confidence-fixed=0,0.937437713",
        ),
        arm(
            "cd-new-only", 20, "joint-cd", 4,
            f"depth-model={cd_new / 'depth-history.tsv'}",
            f"confidence-model={cd_new / 'confidence-history.tsv'}",
        ),
        arm(
            "cd-noop-new-only", 20, "noop-cd", 4,
            f"depth-model={cd_new / 'depth-history.tsv'}",
            f"confidence-model={cd_new / 'confidence-history.tsv'}",
        ),
        arm(
            "joint-augmented", 20, "joint-ckd", 4,
            f"topk-model={router}",
            f"joint-model-dir={cd_joint}",
            "depth-variant=history",
            "confidence-variant=history",
        ),
        arm(
            "joint-noop-augmented", 20, "noop-ckd", 4,
            f"topk-model={router}",
            f"joint-model-dir={cd_joint}",
            "depth-variant=history",
            "confidence-variant=history",
        ),
    ])
    if len(rows) != 10:
        raise ValueError("non-code arm inventory differs")
    record = {
        "schema": 1,
        "source_archive_sha256": manifest["source_archive_sha256"],
        "model_manifest_sha256": sha(source),
        "arms": rows,
    }
    with out.open("x") as target:
        json.dump(record, target, indent=2, sort_keys=True)
        target.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.models, args.out)
    print(json.dumps({
        "arms": len(result["arms"]),
        "source_archive_sha256": result["source_archive_sha256"],
    }))


if __name__ == "__main__":
    main()
