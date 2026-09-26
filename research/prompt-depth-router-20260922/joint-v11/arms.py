"""Freeze bounded native validation arms from v11 training-only models."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


SOURCES = ("noncode-only", "mixed", "augmented")
FIXED_K = (3, 10, 20)
QUANTILES = (0.25, 0.50, 0.75)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def quantiles(rows):
    with gzip.open(rows / "c.jsonl.gz", "rt") as source:
        values = sorted(
            item["chosen_probability"]
            for line in source
            if (item := json.loads(line))["source"].startswith("v11-")
            and item["draft_k"] == 20
            and item["offer_position"] == 1
        )
    if len(values) < 150:
        raise ValueError("fresh non-code K20 C probabilities are insufficient")
    return {
        str(int(q * 100)): values[int(q * (len(values) - 1))]
        for q in QUANTILES
    }


def model_inventory(models):
    manifest = json.loads((models / "manifest.json").read_text())
    if (
        manifest["schema"] != 1
        or manifest["use"] != "v11 validation candidates only"
        or {item["source"] for item in manifest["models"]} != set(SOURCES)
    ):
        raise ValueError("v11 model candidate inventory differs")
    files = {}
    for group in manifest["models"]:
        for item in group["k_models"] + group["cd_models"]:
            path = models / group["source"] / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"v11 policy weight changed: {path}")
            files[(group["source"], item["path"])] = path.resolve()
    return manifest, files


def arm(label, k, mode, cap, *extra):
    return {
        "label": label,
        "k": k,
        "arm": mode,
        "cap": cap,
        "extra": list(extra),
    }


def freeze(models, training_rows, visibility_preflight, out):
    models = models.resolve()
    training_rows = training_rows.resolve()
    manifest, files = model_inventory(models)
    rows_manifest = json.loads((training_rows / "manifest.json").read_text())
    if (
        rows_manifest["schema"] != 1
        or rows_manifest["use"] != "training-only"
        or sha(training_rows / "manifest.json")
        != manifest["v11_training_manifest_sha256"]
        or sha(training_rows / "c.jsonl.gz")
        != rows_manifest["rows"]["c"]["sha256"]
    ):
        raise ValueError("C thresholds lack matching randomized training")
    preflight = json.loads(visibility_preflight.read_text())
    if (
        preflight["schema"] != 1
        or preflight["scope"]
        != "training-only conversation-heldout first-token visibility"
        or preflight["workload_sha256"]
        != rows_manifest["v11_workload_sha256"]
        or preflight["training_archive_sha256"]
        != rows_manifest["v11_training_archive_sha256"]
        or preflight["training_conversations_per_domain"] != 12
        or preflight["heldout_conversations_per_domain"] != 4
        or set(preflight["variants"]) != {"first16", "first32"}
    ):
        raise ValueError("K arm lacks frozen training-only prefix preflight")
    cutoffs = quantiles(training_rows)

    def weight(source, name):
        key = source, name
        if key not in files:
            raise ValueError(f"v11 policy variant missing: {key}")
        return files[key]

    arms = [
        arm(f"fixed-k{k}-d3-c0", k, "fixed:3", 3)
        for k in FIXED_K
    ]
    arms.append(arm("fixed-k20-d4-c0", 20, "fixed:4", 4))
    for label, value in cutoffs.items():
        arms.append(arm(
            f"fixed-k20-c-q{label}", 20, "fixed-c3", 3,
            f"confidence-fixed=0,{value:.9g}",
        ))
    for source in SOURCES:
        depth = weight(source, "topk20/depth-history.tsv")
        confidence = weight(source, "topk20/confidence-history.tsv")
        extra = (f"depth-model={depth}", f"confidence-model={confidence}")
        arms.extend((
            arm(f"cd-{source}", 20, "joint-cd", 4, *extra),
            arm(f"cd-noop-{source}", 20, "noop-cd", 4, *extra),
        ))
    router = weight("mixed", "ridge-100/topk-prior.tsv")
    noncode_router = weight("noncode-only", "ridge-100/topk-prior.tsv")
    for k in FIXED_K:
        for kind in ("depth", "confidence"):
            weight("mixed", f"topk{k}/{kind}-history.tsv")
    joint_extra = (
        f"topk-model={router}",
        f"joint-model-dir={models / 'mixed'}",
        "depth-variant=history",
        "confidence-variant=history",
    )
    arms.extend((
        arm("joint-mixed", 20, "joint-ckd", 4, *joint_extra),
        arm("joint-noop-mixed", 20, "noop-ckd", 4, *joint_extra),
        arm("k-mixed", 20, "learn-topk", 3, f"topk-model={router}"),
        arm("k-noop-mixed", 20, "noop-topk", 3, f"topk-model={router}"),
        arm("k-noncode-only", 20, "learn-topk", 3,
            f"topk-model={noncode_router}"),
        arm("k-noop-noncode-only", 20, "noop-topk", 3,
            f"topk-model={noncode_router}"),
    ))
    if len(arms) != 19 or len({item["label"] for item in arms}) != 19:
        raise ValueError("v11 bounded validation arms differ")
    out.mkdir(exist_ok=False)
    for phase in ("qualification", "validation"):
        save(out / f"{phase}-arms.json", {
            "schema": 1,
            "phase": phase,
            "model_manifest_sha256": sha(models / "manifest.json"),
            "training_manifest_sha256": sha(training_rows / "manifest.json"),
            "visibility_preflight_sha256": sha(visibility_preflight),
            "fixed_c_quantiles": cutoffs,
            "arms": arms,
        })
    return {
        "arms": len(arms),
        "fixed_c_quantiles": cutoffs,
        "model_manifest_sha256": sha(models / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--visibility-preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        freeze(args.models, args.training_rows, args.visibility_preflight,
               args.out),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
