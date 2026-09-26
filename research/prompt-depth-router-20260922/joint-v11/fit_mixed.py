"""Fit small C/K/D controllers from code and non-code training receipts."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

V9 = Path(__file__).resolve().parent.parent / "joint-v9"
sys.path.insert(0, str(V9))
import fit_cd
import fit_k
import collect as v9_collect


V9_NEW_MANIFEST_SHA = "ecde6665d9627f43ca299ce90602158eab6bd7ae0c0d1ba72228bb7ac5f8dc0a"
V9_OLD_MANIFEST_SHA = "196ece7fcd89091615b8c16e249f5f9b3373d46fc8326a1772618ce5565ee073"
V9_K_MANIFEST_SHA = "ac5c14da112616d7c0459c3b109e07429ff2aab2cb1dc70f08bad9abb6eb878a"
V9_CLASS_TABLE_SHA = "34a4b7d67f83f6fa66d967d0bb5f5d74d9aa791469d4c07d6086b401f5962e75"
VARIANTS = ("token", "history", "prior")
SOURCES = ("noncode-only", "mixed", "augmented")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def rows(path):
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source]


def save(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def classes_from_table(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("B\t"):
            _, token, kind = line.split("\t")
            result[int(token)] = int(kind)
    if len(result) < 1000:
        raise ValueError("v9 code token class table is incomplete")
    return result


def token_classes(v9_table, v11_path):
    classes = classes_from_table(v9_table)
    new = json.loads(v11_path.read_text())
    for token, kind in new.items():
        token = int(token)
        if token in classes and classes[token] != kind:
            raise ValueError("code and non-code token byte classes disagree")
        classes[token] = kind
    return classes


def read_inputs(v9_new, v9_old, v9_k_manifest, v9_table, v11):
    if (
        sha(v9_new / "manifest.json") != V9_NEW_MANIFEST_SHA
        or sha(v9_old / "manifest.json") != V9_OLD_MANIFEST_SHA
        or sha(v9_k_manifest) != V9_K_MANIFEST_SHA
        or sha(v9_table) != V9_CLASS_TABLE_SHA
    ):
        raise ValueError("v9 code training provenance differs")
    old_manifest = json.loads((v9_old / "manifest.json").read_text())
    new_manifest = json.loads((v9_new / "manifest.json").read_text())
    mixed_manifest = json.loads((v11 / "manifest.json").read_text())
    if (
        old_manifest["use"] != "training-only"
        or new_manifest["phase"] != "training"
        or mixed_manifest["use"] != "training-only"
    ):
        raise ValueError("input is not assigned to training")
    code_new = {kind: rows(v9_new / f"{kind}.jsonl.gz")
                for kind in ("k", "d", "c")}
    code_old = {kind: rows(v9_old / f"{kind}.jsonl.gz")
                for kind in ("k", "d", "c")}
    noncode = {kind: rows(v11 / f"{kind}.jsonl.gz")
               for kind in ("k", "d", "c")}
    for kind in ("k", "d", "c"):
        if (
            len(noncode[kind]) != mixed_manifest["rows"][kind]["count"]
            or sha(v11 / f"{kind}.jsonl.gz")
            != mixed_manifest["rows"][kind]["sha256"]
            or not code_new[kind] or not code_old[kind]
        ):
            raise ValueError(f"{kind} training row inventory differs")
    if sha(v11 / "token-classes.json") != (
        mixed_manifest["token_classes_sha256"]
    ):
        raise ValueError("non-code token classes changed")
    classes = token_classes(v9_table, v11 / "token-classes.json")
    references = {
        **json.loads(v9_k_manifest.read_text())["references_tok_s"],
        **mixed_manifest["reference_tok_s"],
    }
    for group in (code_new, code_old, noncode):
        for row in group["k"]:
            if row["source"] not in references:
                raise ValueError("K utility lacks source-specific time price")
    return code_new, code_old, noncode, classes, references


def pooled_reference(k_rows):
    control = [
        row for row in k_rows
        if row["draft_k"] == 20
        and row["source"].startswith("v11-")
        and row["assignment"] == "fixed-arm"
    ]
    if not control:
        raise ValueError("no current-GPU fixed K=20 reference")
    return (
        sum(row["output_tokens"] for row in control)
        / sum(row["complete_request_seconds"] for row in control)
    )


def fit_k_weights(data, references, variant, alpha):
    weights = {}
    for k in fit_k.ACTIONS:
        chosen = [row for row in data if row["draft_k"] == k]
        if len(chosen) < 128:
            raise ValueError(f"K={k} lacks source-balanced turn coverage")
        matrix = np.stack([fit_k.vector(row, variant) for row in chosen])
        utility = np.array([
            row["output_tokens"]
            - references[row["source"]] * row["complete_request_seconds"]
            for row in chosen
        ])
        penalty = np.eye(matrix.shape[1]) * alpha
        penalty[0, 0] = 0
        weights[k] = np.linalg.solve(
            matrix.T @ matrix + penalty, matrix.T @ utility
        )
        if not np.isfinite(weights[k]).all():
            raise ValueError("K fit has nonfinite weights")
    return weights


def cd_models(accepted_d, randomized_d, accepted_c, classes, lam, out):
    inventory = []
    for k in fit_k.ACTIONS:
        subdir = out / f"topk{k}"
        subdir.mkdir()
        accepted = [row for row in accepted_d if row["draft_k"] == k]
        timed = [row for row in randomized_d if row["draft_k"] == k]
        offers = [row for row in accepted_c if row["draft_k"] == k]
        if not accepted or not timed or not offers:
            raise ValueError(f"K={k} lacks D/C labels")
        mean_ms = {}
        for depth in range(1, 5):
            observed = [
                row["round_ms"] for row in timed
                if row["draft_depth"] == depth
            ]
            if len(observed) < 150:
                raise ValueError(f"K={k}/D={depth} lacks randomized time")
            mean_ms[depth] = float(np.mean(observed))
        if not all(np.isfinite(mean_ms[d]) for d in range(1, 5)):
            raise ValueError(f"K={k} randomized D costs are nonfinite")
        for variant in VARIANTS:
            a_weights = fit_cd.depth_weights(
                accepted, classes, variant, 100
            )
            t_weights = fit_cd.time_weights(
                timed, classes, variant, 100
            )
            c_weights = fit_cd.confidence_weights(
                offers, classes, variant
            )
            depth_lines = [
                f"joint-v4-linear\t1\t{variant}\t4\t{lam:.12g}\t100"
            ]
            for depth in range(1, 5):
                depth_lines.extend((
                    f"A\t{depth}\t{fit_cd.format_weights(a_weights[depth])}",
                    f"T\t{depth}\t{fit_cd.format_weights(t_weights[depth])}",
                ))
            confidence_lines = [
                f"joint-v4-confidence\t1\t{variant}\t{lam:.12g}"
            ]
            for position in range(3):
                next_offers = [
                    row for row in offers
                    if row["offer_position"] == position + 1
                ]
                if len(next_offers) < 150:
                    raise ValueError(
                        f"K={k}/C={position + 1} lacks offered labels"
                    )
                next_rate = (
                    sum(row["accepted_offer"] for row in next_offers)
                    / len(next_offers)
                )
                marginal_ms = mean_ms[position + 2] - mean_ms[position + 1]
                confidence_lines.append(
                    f"C\t{position}\t{next_rate:.17g}\t"
                    f"{marginal_ms:.17g}\t"
                    f"{fit_cd.format_weights(c_weights[position])}"
                )
            class_lines = [
                f"B\t{token}\t{kind}" for token, kind in sorted(classes.items())
            ]
            for kind, lines in (
                ("depth", depth_lines), ("confidence", confidence_lines)
            ):
                path = subdir / f"{kind}-{variant}.tsv"
                path.write_text("\n".join(lines + class_lines) + "\n")
                inventory.append({
                    "kind": kind,
                    "k": k,
                    "variant": variant,
                    "path": str(path.relative_to(out)),
                    "sha256": sha(path),
                })
            print(json.dumps({
                "fit_source": out.name, "k": k,
                "cd_variant": variant,
            }, sort_keys=True), flush=True)
    return inventory


def build(v9_new, v9_old, v9_k_manifest, v9_table, v11, out):
    code_new, code_old, noncode, classes, references = read_inputs(
        v9_new, v9_old, v9_k_manifest, v9_table, v11
    )
    out.mkdir(parents=True, exist_ok=False)
    results = {
        "schema": 1,
        "use": "v11 validation candidates only",
        "model_sha256": v9_collect.MODEL_SHA256,
        "binary_sha256": v9_collect.BINARY_SHA256,
        "code_training_manifest_sha256": V9_NEW_MANIFEST_SHA,
        "v11_training_manifest_sha256": sha(v11 / "manifest.json"),
        "source_reference_tok_s": references,
        "models": [],
    }
    for source in SOURCES:
        base = out / source
        base.mkdir()
        use_code = source != "noncode-only"
        use_old = source == "augmented"
        chosen = {
            kind: (
                noncode[kind]
                + (code_new[kind] if use_code else [])
                + (code_old[kind] if use_old else [])
            )
            for kind in ("k", "d", "c")
        }
        randomized = [
            row for row in chosen["d"]
            if row["assignment"] == "randomized-round"
            and row["source"].startswith("v11-")
        ]
        lam = pooled_reference(chosen["k"])
        k_models = []
        for alpha in (20, 100):
            k_dir = base / f"ridge-{alpha}"
            k_dir.mkdir()
            for variant in fit_k.FEATURE_COUNT:
                weights = fit_k_weights(
                    chosen["k"], references, variant, alpha
                )
                lines = [
                    f"joint-v6-draft-topk\t1\t{variant}\t"
                    f"{lam:.12g}\t{alpha}"
                ]
                for k in fit_k.ACTIONS:
                    lines.append(
                        f"K\t{k}\t{fit_cd.format_weights(weights[k])}"
                    )
                path = k_dir / f"topk-{variant}.tsv"
                path.write_text("\n".join(lines) + "\n")
                predicted = Counter(
                    fit_k.predict(row, variant, weights)
                    for row in chosen["k"]
                )
                k_models.append({
                    "kind": "k",
                    "alpha": alpha,
                    "variant": variant,
                    "path": str(path.relative_to(base)),
                    "sha256": sha(path),
                    "training_only_actions": dict(sorted(predicted.items())),
                })
                print(json.dumps({
                    "fit_source": source, "k_alpha": alpha,
                    "k_variant": variant,
                }, sort_keys=True), flush=True)
        models = cd_models(
            chosen["d"], randomized, chosen["c"], classes, lam, base
        )
        results["models"].append({
            "source": source,
            "reference_tok_s": lam,
            "training_rows": {
                kind: len(chosen[kind]) for kind in ("k", "d", "c")
            },
            "randomized_time_rows": len(randomized),
            "k_models": k_models,
            "cd_models": models,
        })
    save(out / "manifest.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ("v9-new", "v9-old", "v9-k-manifest", "v9-table",
                 "v11-rows", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.v9_new, args.v9_old, args.v9_k_manifest, args.v9_table,
        args.v11_rows, args.out,
    )
    print(json.dumps({
        item["source"]: item["training_rows"] for item in result["models"]
    }, sort_keys=True))


if __name__ == "__main__":
    main()
