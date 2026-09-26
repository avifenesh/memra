"""Exercise randomized K and D/C on disjoint qualifier prompts before training."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import quality
import prepare_code_training
import prepare_transfer_inputs


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"
VALIDATION_SHA = "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
FINAL_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def collector():
    path = Path(__file__).resolve().parent / "collect.py"
    spec = importlib.util.spec_from_file_location(
        "v11_pilot_collect", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(binary, model, workloads, source_manifest, out):
    if os.environ.get("MEMRA_CAPTURE_DIR"):
        raise ValueError("research pilot cannot hold customer capture")
    if (
        sha(binary) != BINARY_SHA
        or sha(model) != MODEL_SHA
        or sha(workloads / "manifest.json") != WORKLOAD_SHA
    ):
        raise ValueError("v11 pilot artifact or workload differs")
    source = json.loads(source_manifest.read_text())
    for name, expected in source["files"].items():
        path = Path(__file__).resolve().parent / name
        if sha(path) != expected:
            raise ValueError(f"v11 pilot source changed: {name}")
        if path.suffix == ".py":
            compile(path.read_bytes(), str(path), "exec")
    base = source_manifest.parent
    finalizer_source = base / "ops/finalize_v11.py"
    compile(finalizer_source.read_bytes(), str(finalizer_source), "exec")
    code = prepare_code_training.prepare(
        base / "v9-parent/native-data.tar.gz",
        base / "v9-parent/manifest.json",
        base / "code-training",
    )
    transfer = prepare_transfer_inputs.prepare(
        base / "v10-parent/native-data.tar.gz",
        base / "v10-parent/manifest.json",
        base / "v10-parent/custody.json",
        base / "transfer-inputs",
    )
    for name in ("feature_audit.py", "fit_mixed.py"):
        path = Path(__file__).resolve().parent / name
        spec = importlib.util.spec_from_file_location(
            "v11_pilot_" + name.removesuffix(".py"), path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    rows_path = Path(__file__).resolve().parent / "measurement_rows.py"
    spec = importlib.util.spec_from_file_location(
        "measurement_rows", rows_path
    )
    rows_module = importlib.util.module_from_spec(spec)
    sys.modules["measurement_rows"] = rows_module
    spec.loader.exec_module(rows_module)
    for name in (
        "arms", "evaluation_seal", "evaluation_replay",
        "preflight_visibility", "score", "select",
        "training_chain_replay", "training_seal", "training_supervise",
    ):
        path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            "v11_pilot_" + name, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    behavior_path = Path(__file__).resolve().parent / "behavior.py"
    spec = importlib.util.spec_from_file_location(
        "v11_pilot_behavior", behavior_path
    )
    behavior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(behavior)
    archive = behavior.V10Archive(
        base / "v10-parent/native-data.tar.gz",
        base / "v10-parent/manifest.json",
    )
    try:
        behavior_result = behavior.analyze(
            archive, base / "v10-parent/custody.json"
        )
        fixed_rows, excluded_fixed, reference_rates = (
            rows_module.v10_rows(
                archive, base / "v10-parent/custody.json", {}
            )
        )
    finally:
        archive.close()
    fixed_counts = {
        kind: len(rows) for kind, rows in fixed_rows.items()
    }
    if (
        fixed_counts["k"] == 0
        or fixed_counts["d"] == 0
        or fixed_counts["c"] != 0
        or set(reference_rates) != {"v10-ifeval", "v10-gsm8k"}
    ):
        raise ValueError("v10 fixed-arm training preflight differs")
    del fixed_rows
    (base / "behavior-v10.json").write_text(
        json.dumps(behavior_result, indent=2, sort_keys=True) + "\n"
    )
    manifest = json.loads((workloads / "manifest.json").read_text())
    if (
        set(manifest["groups"]) != {"qualification", "training"}
        or manifest["source_full_manifest_sha256"] != FINAL_SHA
        or manifest["validation_projection_sha256"] != VALIDATION_SHA
    ):
        raise ValueError("pilot stage exposes a reserved task phase")
    os.environ["NLTK_DATA"] = str(base / "transfer-inputs/nltk_data")
    evaluator = quality.ifeval_grader(
        base / "transfer-inputs/third_party/instruction_following_eval"
    )
    graded_inputs = 0
    for phase in ("qualification", "training"):
        for entry in manifest["groups"][phase]["ifeval"]:
            for task in entry["turns"]:
                quality.grade("ifeval", task, "test", evaluator)
                graded_inputs += 1
        for entry in manifest["groups"][phase]["gsm8k"]:
            for task in entry["turns"]:
                if quality.gsm_answer("#### " + task["gold_answer"]) is None:
                    raise ValueError("v11 pilot math key cannot be graded")
    out.mkdir(exist_ok=False)
    args = SimpleNamespace(binary=binary, model=model,
                           workloads=workloads, out=out)
    collect = collector()
    k_entry = manifest["groups"]["qualification"]["ifeval"][0]
    d_entry = manifest["groups"]["qualification"]["gsm8k"][0]
    k_result = collect.run_random(args, k_entry, 32)
    d_result = collect.v9_collect.run_one(
        args, d_entry, 33, 20, "explore-d"
    )
    for row in (k_result, d_result):
        collect.v9_collect.save(
            out / f"{row['name']}.result.json", row
        )
    offers = list(rows_module.old_rows.round_rows(
        rows_module.Directory(out),
        {"records": [d_result]},
        "v9",
        lambda _: True,
    ))
    c_labels = sum(kind == "c" for kind, _ in offers)
    d_labels = sum(kind == "d" for kind, _ in offers)
    if (
        k_result["cached_later_turns"] != 7
        or len(k_result["k_actions"]) != 3
        or d_result["cached_later_turns"] != 7
        or len([d for d, n in d_result["d_exposure"].items() if n]) < 2
        or c_labels == 0
        or d_labels == 0
    ):
        raise ValueError("v11 K/D pilot did not engage")
    return {
        "schema": 1,
        "status": "random-K-and-D-C-pilot-qualified",
        "training_use": False,
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workloads_sha256": WORKLOAD_SHA,
        "source_manifest_sha256": sha(source_manifest),
        "code_parent_status": code["status"],
        "transfer_parent_status": transfer["status"],
        "behavior_sha256": sha(base / "behavior-v10.json"),
        "v10_fixed_rows": fixed_counts,
        "v10_excluded_fixed_conversations": excluded_fixed,
        "graded_instruction_inputs": graded_inputs,
        "sessions": [k_result["name"], d_result["name"]],
        "k_actions": k_result["k_actions"],
        "d_exposure": d_result["d_exposure"],
        "c_offer_labels": c_labels,
        "d_round_labels": d_labels,
        "native_kv_later_turns": [
            k_result["cached_later_turns"],
            d_result["cached_later_turns"],
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("binary", "model", "workloads", "source-manifest",
                 "out", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("binary", "model", "workloads", "source_manifest",
                 "out", "receipt"):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args.binary, args.model, args.workloads,
                 args.source_manifest, args.out)
    with args.receipt.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "status": result["status"], "sessions": result["sessions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
