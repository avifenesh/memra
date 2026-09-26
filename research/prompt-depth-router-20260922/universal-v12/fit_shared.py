"""Fit task-blind K/C/D candidates from code, math, IF and fresh prose."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys


os.environ["OPENBLAS_NUM_THREADS"] = "1"
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "joint-v11"))
import fit_mixed as previous


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
SOURCES = ("mixed-fresh", "augmented-fresh", "prose-balanced")
KINDS = ("k", "d", "c")


def read_prose(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or manifest["use"] != "training-only"
        or manifest["training_workloads_sha256"] != TRAIN_SHA
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or manifest["expected_training_sessions"] != 112
    ):
        raise ValueError("fresh prose rows lack randomized training lineage")
    result = {}
    for kind in KINDS:
        file = path / f"{kind}.jsonl.gz"
        if previous.sha(file) != manifest["rows"][kind]["sha256"]:
            raise ValueError(f"fresh prose {kind} row hash changed")
        with gzip.open(file, "rt") as source:
            rows = [json.loads(line) for line in source]
        if (
            len(rows) != manifest["rows"][kind]["count"]
            or any(
                row["source"] != "v12-prose"
                or row["domain"] != "prose"
                for row in rows
            )
        ):
            raise ValueError(f"fresh prose {kind} row source differs")
        result[kind] = rows
    classes_path = path / "token-classes.json"
    if previous.sha(classes_path) != manifest["token_classes_sha256"]:
        raise ValueError("fresh prose token classes changed")
    classes = {
        int(token): kind
        for token, kind in json.loads(classes_path.read_text()).items()
    }
    reference = manifest["reference_tok_s"]["v12-prose"]
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("fresh prose K price differs")
    return result, classes, reference


def merge_classes(base, additional):
    for token, kind in additional.items():
        if token in base and base[token] != kind:
            raise ValueError("mixed tokenizer byte classes disagree")
        base[token] = kind


def pooled_reference(rows):
    reference = [
        row for row in rows
        if row["draft_k"] == 20
        and row["source"].startswith(("v11-", "v12-"))
        and row["assignment"] == "fixed-arm"
    ]
    if not reference:
        raise ValueError("current GPU K20 reference missing")
    return (
        sum(row["output_tokens"] for row in reference)
        / sum(row["complete_request_seconds"] for row in reference)
    )


def build(v9_new, v9_old, v9_k_manifest, v9_table, v11_rows,
          prose_rows, out):
    code_new, code_old, noncode, classes, references = (
        previous.read_inputs(
            v9_new, v9_old, v9_k_manifest, v9_table, v11_rows,
        )
    )
    fresh, prose_classes, prose_reference = read_prose(prose_rows)
    merge_classes(classes, prose_classes)
    references["v12-prose"] = prose_reference
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": 1,
        "use": "v12 one-policy validation candidates only",
        "model_sha256": previous.v9_collect.MODEL_SHA256,
        "binary_sha256": previous.v9_collect.BINARY_SHA256,
        "code_training_manifest_sha256":
        previous.V9_NEW_MANIFEST_SHA,
        "v11_training_manifest_sha256":
        previous.sha(v11_rows / "manifest.json"),
        "v12_prose_training_manifest_sha256":
        previous.sha(prose_rows / "manifest.json"),
        "v12_workload_training_projection_sha256": TRAIN_SHA,
        "source_reference_tok_s": references,
        "models": [],
    }
    for source in SOURCES:
        output = out / source
        output.mkdir()
        chosen = {
            kind: (
                noncode[kind] + code_new[kind]
                + (code_old[kind] if source != "mixed-fresh" else [])
                + fresh[kind] * (3 if source == "prose-balanced" else 1)
            )
            for kind in KINDS
        }
        randomized = [
            row for row in chosen["d"]
            if row["assignment"] == "randomized-round"
            and row["source"].startswith(("v11-", "v12-"))
        ]
        rate = pooled_reference(chosen["k"])
        k_models = []
        for alpha in (20, 100):
            directory = output / f"ridge-{alpha}"
            directory.mkdir()
            for variant in previous.fit_k.FEATURE_COUNT:
                weights = previous.fit_k_weights(
                    chosen["k"], references, variant, alpha,
                )
                lines = [
                    f"joint-v6-draft-topk\t1\t{variant}\t"
                    f"{rate:.12g}\t{alpha}"
                ]
                for k in previous.fit_k.ACTIONS:
                    lines.append(
                        f"K\t{k}\t"
                        f"{previous.fit_cd.format_weights(weights[k])}"
                    )
                path = directory / f"topk-{variant}.tsv"
                path.write_text("\n".join(lines) + "\n")
                predicted = Counter(
                    previous.fit_k.predict(row, variant, weights)
                    for row in chosen["k"]
                )
                k_models.append({
                    "kind": "k", "alpha": alpha,
                    "variant": variant,
                    "path": str(path.relative_to(output)),
                    "sha256": previous.sha(path),
                    "training_only_actions":
                    dict(sorted(predicted.items())),
                })
        cd_models = previous.cd_models(
            chosen["d"], randomized, chosen["c"],
            classes, rate, output,
        )
        result["models"].append({
            "source": source,
            "prose_weight": 3 if source == "prose-balanced" else 1,
            "reference_tok_s": rate,
            "training_rows": {
                kind: len(chosen[kind]) for kind in KINDS
            },
            "randomized_time_rows": len(randomized),
            "k_models": k_models,
            "cd_models": cd_models,
        })
    previous.save(out / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "v9-new", "v9-old", "v9-k-manifest", "v9-table",
        "v11-rows", "prose-rows", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.v9_new, args.v9_old, args.v9_k_manifest, args.v9_table,
        args.v11_rows, args.prose_rows, args.out,
    )
    print(json.dumps({
        group["source"]: group["training_rows"]
        for group in result["models"]
    }, sort_keys=True))


if __name__ == "__main__":
    main()
