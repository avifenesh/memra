"""Fit task-blind K/C/D candidates from fresh and historical measurements."""

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
SOURCES = ("fresh-only", "mixed-history", "augmented-history")
KINDS = ("k", "d", "c")
DOMAINS = ("code", "prose", "math")


def read_fresh(path, replay_path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replay = json.loads(replay_path.read_text())
    if (
        manifest["schema"] != 1
        or manifest["use"] != "training-only"
        or manifest["training_workloads_sha256"] != TRAIN_SHA
        or manifest["source_full_manifest_sha256"] != FULL_SHA
        or manifest["expected_training_sessions"] != 336
        or replay["schema"] != 1
        or replay["status"]
        != "fresh-mixed-training-native-K-D-C-replay-match"
        or replay["training_workloads_sha256"] != TRAIN_SHA
        or replay["source_full_manifest_sha256"] != FULL_SHA
        or replay["training_rows_manifest_sha256"]
        != previous.sha(manifest_path)
        or replay["row_counts"] != {
            kind: manifest["rows"][kind]["count"]
            for kind in KINDS
        }
    ):
        raise ValueError("fresh mixed rows lack randomized training lineage")
    result = {}
    for kind in KINDS:
        file = path / f"{kind}.jsonl.gz"
        if previous.sha(file) != manifest["rows"][kind]["sha256"]:
            raise ValueError(f"fresh mixed {kind} row hash changed")
        with gzip.open(file, "rt") as source:
            rows = [json.loads(line) for line in source]
        if (
            len(rows) != manifest["rows"][kind]["count"]
            or any(
                row["domain"] not in DOMAINS
                or row["source"] != f"v12-{row['domain']}"
                for row in rows
            )
            or {
                domain: sum(
                    row["domain"] == domain for row in rows
                )
                for domain in DOMAINS
            } != manifest["rows"][kind]["by_domain"]
        ):
            raise ValueError(f"fresh mixed {kind} row source differs")
        result[kind] = rows
    classes_path = path / "token-classes.json"
    if previous.sha(classes_path) != manifest["token_classes_sha256"]:
        raise ValueError("fresh mixed token classes changed")
    classes = {
        int(token): kind
        for token, kind in json.loads(classes_path.read_text()).items()
    }
    reference = manifest["reference_tok_s"]
    if set(reference) != {
        f"v12-{domain}" for domain in DOMAINS
    } or any(
        not math.isfinite(value) or value <= 0
        for value in reference.values()
    ):
        raise ValueError("fresh mixed K prices differ")
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
        and row["source"].startswith("v12-")
        and row["assignment"] == "fixed-arm"
    ]
    if not reference:
        raise ValueError("current GPU K20 reference missing")
    return (
        sum(row["output_tokens"] for row in reference)
        / sum(row["complete_request_seconds"] for row in reference)
    )


def build(v9_new, v9_old, v9_k_manifest, v9_table, v11_rows,
          fresh_rows, fresh_replay, out):
    code_new, code_old, noncode, classes, references = (
        previous.read_inputs(
            v9_new, v9_old, v9_k_manifest, v9_table, v11_rows,
        )
    )
    fresh, fresh_classes, fresh_references = read_fresh(
        fresh_rows, fresh_replay,
    )
    merge_classes(classes, fresh_classes)
    references.update(fresh_references)
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
        "v12_fresh_training_manifest_sha256":
        previous.sha(fresh_rows / "manifest.json"),
        "v12_fresh_training_replay_sha256":
        previous.sha(fresh_replay),
        "v12_workload_training_projection_sha256": TRAIN_SHA,
        "source_reference_tok_s": references,
        "models": [],
    }
    for source in SOURCES:
        output = out / source
        output.mkdir()
        chosen = {
            kind: (
                fresh[kind]
                + (
                    noncode[kind] + code_new[kind]
                    if source != "fresh-only" else []
                )
                + (
                    code_old[kind]
                    if source == "augmented-history" else []
                )
            )
            for kind in KINDS
        }
        randomized = [
            row for row in chosen["d"]
            if row["assignment"] == "randomized-round"
            and row["source"].startswith("v12-")
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
            "historical_measurements": (
                "none" if source == "fresh-only"
                else "code-new-v10-v11"
                if source == "mixed-history"
                else "code-new-and-old-v10-v11"
            ),
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
        "v11-rows", "fresh-rows", "fresh-replay", "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.v9_new, args.v9_old, args.v9_k_manifest, args.v9_table,
        args.v11_rows, args.fresh_rows, args.fresh_replay,
        args.out,
    )
    print(json.dumps({
        group["source"]: group["training_rows"]
        for group in result["models"]
    }, sort_keys=True))


if __name__ == "__main__":
    main()
