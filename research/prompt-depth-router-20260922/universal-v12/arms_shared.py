"""Freeze one bounded mixed-domain validation menu from training only."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v9"))
import eval as v9_eval


FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
SOURCES = ("mixed-fresh", "augmented-fresh", "prose-balanced")
QUANTILES = (25, 50, 75)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def model_inventory(models):
    manifest = json.loads((models / "manifest.json").read_text())
    if (
        manifest["schema"] != 1
        or manifest["use"] != "v12 one-policy validation candidates only"
        or {item["source"] for item in manifest["models"]}
        != set(SOURCES)
        or manifest["v12_workload_training_projection_sha256"]
        != TRAIN_SHA
        or manifest["model_sha256"] != v9_eval.MODEL_SHA256
        or manifest["binary_sha256"] != v9_eval.BINARY_SHA256
    ):
        raise ValueError("mixed learned model inventory differs")
    files = {}
    for group in manifest["models"]:
        for item in group["k_models"] + group["cd_models"]:
            path = models / group["source"] / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError("mixed learned weight changed")
            files[(group["source"], item["path"])] = path.resolve()
    return manifest, files


def quantiles(v11_rows, prose_rows):
    values = []
    for root in (v11_rows, prose_rows):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] != 1 or manifest["use"] != "training-only":
            raise ValueError("C fixed control lacks training-only source")
        source = root / "c.jsonl.gz"
        if sha(source) != manifest["rows"]["c"]["sha256"]:
            raise ValueError("C fixed control training rows changed")
        with gzip.open(source, "rt") as stream:
            for line in stream:
                item = json.loads(line)
                if (
                    item["draft_k"] == 20
                    and item["offer_position"] == 1
                    and item["source"].startswith(("v11-", "v12-"))
                ):
                    values.append(item["chosen_probability"])
    values.sort()
    if len(values) < 150:
        raise ValueError("K20 C fixed quantiles lack randomized offers")
    return {
        str(q): values[int((q / 100) * (len(values) - 1))]
        for q in QUANTILES
    }


def arm(label, role, k, mode, cap, *extra, noop_label=None,
        selectable=False):
    entry = {
        "label": label, "role": role,
        "k": k, "arm": mode, "cap": cap,
        "extra": list(extra),
    }
    if role in ("learned", "noop"):
        entry["policy_sha256"] = digest(
            v9_eval.model_hashes(entry["extra"])
        )
    if noop_label:
        entry["noop_label"] = noop_label
    if role == "learned":
        entry["selectable"] = selectable
    return entry


def freeze(models, v11_rows, prose_rows, prose_replay,
           preflight_path, out):
    models = models.resolve()
    manifest, files = model_inventory(models)
    preflight = json.loads(preflight_path.read_text())
    if (
        sha(v11_rows / "manifest.json")
        != manifest["v11_training_manifest_sha256"]
        or sha(prose_rows / "manifest.json")
        != manifest["v12_prose_training_manifest_sha256"]
        or sha(prose_replay)
        != manifest["v12_prose_training_replay_sha256"]
        or preflight["schema"] != 1
        or preflight["status"] != "training-prefix-visible"
        or preflight["no_field_route"] is not True
        or preflight["scope"]
        != "training-only conversation-heldout prefix visibility"
        or preflight["v9_code_training_manifest_sha256"]
        != manifest["code_training_manifest_sha256"]
        or preflight["v11_training_manifest_sha256"]
        != manifest["v11_training_manifest_sha256"]
        or preflight["v12_prose_training_manifest_sha256"]
        != manifest["v12_prose_training_manifest_sha256"]
        or preflight["v12_prose_training_replay_sha256"]
        != manifest["v12_prose_training_replay_sha256"]
    ):
        raise ValueError("mixed candidate training or prefix preflight differs")
    cutoffs = quantiles(v11_rows, prose_rows)

    def weight(source, name):
        try:
            return files[source, name]
        except KeyError as error:
            raise ValueError(f"mixed candidate weight missing: {name}") from error

    arms = [
        arm(f"fixed-k{k}-d3-c0", "fixed", k, "fixed:3", 3)
        for k in (3, 10, 20)
    ]
    for depth in (1, 2, 4):
        arms.append(arm(
            f"fixed-k20-d{depth}-c0", "fixed", 20,
            f"fixed:{depth}", depth,
        ))
    for q, cutoff in cutoffs.items():
        arms.append(arm(
            f"fixed-k20-d3-cq{q}", "fixed", 20, "fixed-c3", 3,
            f"confidence-fixed=0,{cutoff:.9g}",
        ))
    for source in SOURCES:
        router = weight(source, "ridge-100/topk-prior.tsv")
        for k in (3, 10, 20):
            for kind in ("depth", "confidence"):
                weight(source, f"topk{k}/{kind}-history.tsv")
        extra = (
            f"topk-model={router}",
            f"joint-model-dir={models / source}",
            "depth-variant=history",
            "confidence-variant=history",
        )
        label = f"joint-{source}"
        noop = f"joint-noop-{source}"
        arms.extend((
            arm(label, "learned", 20, "joint-ckd", 4,
                *extra, noop_label=noop, selectable=True),
            arm(noop, "noop", 20, "noop-ckd", 4, *extra),
        ))
    source = "prose-balanced"
    depth = weight(source, "topk20/depth-history.tsv")
    confidence = weight(source, "topk20/confidence-history.tsv")
    extra = (
        f"depth-model={depth}", f"confidence-model={confidence}",
    )
    arms.extend((
        arm("cd-prose-balanced", "learned", 20, "joint-cd", 4,
            *extra, noop_label="cd-noop-prose-balanced"),
        arm("cd-noop-prose-balanced", "noop", 20, "noop-cd", 4,
            *extra),
    ))
    router = weight(source, "ridge-100/topk-prior.tsv")
    extra = (f"topk-model={router}",)
    arms.extend((
        arm("k-prose-balanced", "learned", 20, "learn-topk", 3,
            *extra, noop_label="k-noop-prose-balanced"),
        arm("k-noop-prose-balanced", "noop", 20, "noop-topk", 3,
            *extra),
    ))
    if len(arms) != 19 or len({item["label"] for item in arms}) != 19:
        raise ValueError("mixed validation arm menu differs")
    out.mkdir(exist_ok=False)
    common = {
        "schema": 1,
        "source_manifest_sha256": FULL_SHA,
        "training_workloads_sha256": TRAIN_SHA,
        "model_manifest_sha256": sha(models / "manifest.json"),
        "v11_training_rows_sha256": sha(v11_rows / "manifest.json"),
        "v12_prose_training_rows_sha256":
        sha(prose_rows / "manifest.json"),
        "v12_prose_training_replay_sha256":
        sha(prose_replay),
        "training_prefix_preflight_sha256": sha(preflight_path),
        "fixed_c_quantiles": cutoffs,
        "domains": ["code", "prose", "math"],
        "arms": arms,
    }
    qualification = out / "qualification-arms.json"
    save(qualification, {**common, "phase": "qualification"})
    save(out / "validation-arms.json", {
        **common, "phase": "validation",
        "qualification_arms_sha256": sha(qualification),
    })
    return {
        "arms": len(arms),
        "model_manifest_sha256": common["model_manifest_sha256"],
        "qualification_arms_sha256": sha(qualification),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "models", "v11-rows", "prose-rows", "prose-replay",
        "preflight", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        freeze(
            args.models, args.v11_rows, args.prose_rows,
            args.prose_replay, args.preflight, args.out,
        ),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
