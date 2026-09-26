"""Recompute v11 native, task-quality, selection and speed receipts."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58"
VALIDATION_SHA = "e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd"
FINAL_SHA = "71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df"
HELDOUT_COMMIT = "ec85aaaf044cca42779c12144b3c6ef1acb96a96155e866e6db6c93c356a1f62"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def check_phase_files(root, tasks, phases, groups):
    if set(tasks["groups"]) != set(groups):
        raise ValueError("evaluation replay phase metadata differs")
    expected = {"manifest.json"} | {
        entry["file"]
        for phase in phases
        for domain in ("ifeval", "gsm8k")
        for entry in tasks["groups"][phase][domain]
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("evaluation replay phase file allowlist differs")


def extract(archive, manifest, root):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in manifest["members"]
                or name in seen
                or not member.isfile()
                or logical.is_absolute()
                or ".." in logical.parts
                or str(logical) != name
                or member.size != manifest["members"][name]["bytes"]
            ):
                raise ValueError(f"unsafe or changed evaluation member: {name}")
            seen.add(name)
            path = root.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as input_file, path.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != manifest["members"][name]["sha256"]:
                raise ValueError(f"evaluation member hash differs: {name}")
    if seen != set(manifest["members"]):
        raise ValueError("sealed evaluation omitted members")
    return len(seen)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy_paths(arm):
    options = dict(item.split("=", 1) for item in arm.get("extra", []))
    paths = [
        Path(options[key]) for key in (
            "topk-model", "depth-model", "confidence-model"
        ) if key in options
    ]
    if "joint-model-dir" in options:
        directory = Path(options["joint-model-dir"])
        for k in (3, 10, 20):
            for kind in ("depth", "confidence"):
                paths.append(
                    directory / f"topk{k}/{kind}-{options[kind + '-variant']}.tsv"
                )
    return paths


def verify_policies(root, arm, command):
    expected = {str(path) for path in policy_paths(arm)}
    if set(command["policy_sha256"]) != expected:
        raise ValueError("native evaluation policy inventory differs")
    for original in expected:
        parts = Path(original).parts
        if "policy-models" not in parts or ".." in parts:
            raise ValueError("native policy path leaves frozen model directory")
        index = parts.index("policy-models")
        relative = Path(*parts[index + 1:])
        staged = root / "policy/models" / relative
        if sha(staged) != command["policy_sha256"][original]:
            raise ValueError(f"native policy weight differs: {original}")


def verify_models(root):
    manifest = json.loads(
        (root / "policy/models/manifest.json").read_text()
    )
    for group in manifest["models"]:
        for item in group["k_models"] + group["cd_models"]:
            path = root / "policy/models" / group["source"] / item["path"]
            if sha(path) != item["sha256"]:
                raise ValueError(f"fitted policy model changed: {path}")


def verify_native(root, tasks, arms, phase, v9_eval):
    native = root / "native/eval-results"
    inputs = root / "inputs" / (
        "final-workloads" if phase == "heldout"
        else "validation-workloads"
    )
    for domain in ("ifeval", "gsm8k"):
        entries = tasks["groups"][phase][domain]
        for index, entry in enumerate(entries):
            if sha(inputs / entry["file"]) != entry["sha256"]:
                raise ValueError(f"v11 {phase} prompt file differs")
            for arm in arms["arms"]:
                name = f"{phase}-{domain}-{index}-{arm['label']}"
                result = json.loads((native / f"{name}.result.json").read_text())
                observed = v9_eval.verify(native / name, entry, arm)
                command = json.loads((native / f"{name}.command.json").read_text())
                exit_row = json.loads((native / f"{name}.exit.json").read_text())
                argv = command["argv"]
                if (
                    canonical(result) != canonical(observed)
                    or command["model_sha256"] != v9_eval.MODEL_SHA256
                    or command["binary_sha256"] != v9_eval.BINARY_SHA256
                    or command["workload_sha256"] != entry["sha256"]
                    or exit_row["returncode"] != 0
                    or argv[2] != "embedded"
                    or Path(argv[3]).name != entry["file"]
                    or Path(argv[4]).name != name
                    or argv[5] != arm["arm"]
                    or argv[6] != str(entry["seed"])
                    or argv[7:10] != ["4096", "65536", "1.0"]
                    or argv[10:13] != [
                        f"cap={arm['cap']}", "sampler-top-k=20",
                        f"draft-top-k={arm['k']}",
                    ]
                    or argv[13:] != arm.get("extra", [])
                ):
                    raise ValueError(f"native v11 result or command differs: {name}")
                log = (native / f"{name}.stderr.log").read_text()
                if (
                    "full_vocab=248320 draft_vocab=248320 mtp=embedded"
                    not in log or "target_top_k=20" not in log
                ):
                    raise ValueError(f"v11 full-head engagement differs: {name}")
                verify_policies(root, arm, command)
            print(json.dumps({
                "replay_phase": phase,
                "domain": domain,
                "conversation": index,
            }, sort_keys=True), flush=True)


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or manifest["model_sha256"] != MODEL_SHA
        or manifest["binary_sha256"] != BINARY_SHA
        or manifest["workloads_sha256"] != WORKLOAD_SHA
        or manifest["validation_workloads_sha256"] != VALIDATION_SHA
        or manifest["full_workloads_sha256"] != FINAL_SHA
        or manifest["phase_counts"]["qualification"] != 38
        or manifest["phase_counts"]["validation"] != 304
    ):
        raise ValueError("sealed evaluation archive differs")
    with tempfile.TemporaryDirectory(prefix="mtp-v11-evaluation-replay-") as temp:
        root = Path(temp)
        members = extract(archive, manifest, root)
        source = json.loads(
            (root / "inputs/source-manifest.json").read_text()
        )
        metadata_path = root / "native/run-meta.json"
        metadata = json.loads(metadata_path.read_text())
        tasks_path = root / "inputs/workloads/manifest.json"
        tasks = json.loads(tasks_path.read_text())
        validation_path = (
            root / "inputs/validation-workloads/manifest.json"
        )
        validation_tasks = json.loads(validation_path.read_text())
        check_phase_files(
            root / "inputs/workloads", tasks,
            ("qualification", "training"),
            ("qualification", "training"),
        )
        check_phase_files(
            root / "inputs/validation-workloads", validation_tasks,
            ("qualification", "training", "validation"),
            ("qualification", "training", "validation"),
        )
        training = json.loads(
            (root / "training/manifest.json").read_text()
        )
        training_replay = json.loads(
            (root / "training/replay.json").read_text()
        )
        selection = json.loads(
            (root / "policy/arms/selected.json").read_text()
        )
        if (
            sha(metadata_path) != manifest["run_meta_sha256"]
            or sha(tasks_path) != manifest["workloads_sha256"]
            or sha(validation_path)
            != manifest["validation_workloads_sha256"]
            or sha(root / "inputs/source-manifest.json")
            != manifest["source_manifest_sha256"]
            or metadata["source_manifest_sha256"]
            != manifest["source_manifest_sha256"]
            or metadata["model_sha256"] != manifest["model_sha256"]
            or metadata["binary_sha256"] != manifest["binary_sha256"]
            or metadata["workloads_sha256"] != WORKLOAD_SHA
            or metadata["validation_workloads_sha256"] != VALIDATION_SHA
            or metadata["full_workloads_sha256"] != FINAL_SHA
            or tasks["validation_projection_sha256"] != VALIDATION_SHA
            or tasks["source_full_manifest_sha256"] != FINAL_SHA
            or tasks["heldout_group_sha256"] != HELDOUT_COMMIT
            or validation_tasks["source_full_manifest_sha256"] != FINAL_SHA
            or validation_tasks["heldout_group_sha256"]
            != HELDOUT_COMMIT
            or set(tasks["groups"]) != {"qualification", "training"}
            or set(validation_tasks["groups"])
            != {"qualification", "training", "validation"}
            or any(
                tasks["groups"][phase]
                != validation_tasks["groups"][phase]
                for phase in ("qualification", "training")
            )
            or metadata["v9_archive_sha256"]
            != manifest["v9_archive_sha256"]
            or metadata["v10_archive_sha256"]
            != manifest["v10_archive_sha256"]
            or metadata["v10_parent_archive_sha256"]
            != manifest["v10_parent_archive_sha256"]
            or metadata["customer_capture"] is not False
            or sha(root / "source/private_ops/training_runmeta.py")
            != metadata["ops_source_sha256"]
            or sha(root / "source/private_ops/finalize_v11.py")
            != metadata["finalizer_source_sha256"]
            or sha(root / "source/private_ops/prepare_v10_projection.py")
            != metadata["v10_projection_source_sha256"]
            or training["archive_sha256"]
            != manifest["training_archive_sha256"]
            or training_replay["status"]
            != "training-native-random-K-D-C-KV-replay-match"
            or training_replay["archive_sha256"]
            != manifest["training_archive_sha256"]
            or selection["status"] != manifest["selection_status"]
            or source["schema"] != 1
        ):
            raise ValueError("v11 evaluation source or parent differs")
        validation_ready = json.loads(
            (root / "native/validation-ready.json").read_text()
        )
        validation_request = json.loads(
            (root / "native/needs-validation.json").read_text()
        )
        if (
            validation_ready != {
                "phase": "validation", "manifest_sha256": VALIDATION_SHA,
            }
            or validation_request != validation_ready
        ):
            raise ValueError("validation phase release differs")
        for name, expected in source["files"].items():
            if sha(root / "source/joint-v11" / name) != expected:
                raise ValueError(f"v11 source hash differs: {name}")
        for name, expected in metadata["v9_source_sha256"].items():
            if sha(root / "source/joint-v9" / name) != expected:
                raise ValueError(f"v9 source hash differs: {name}")
        for name, expected in metadata["v4_source_sha256"].items():
            if sha(root / "source/joint-v4" / name) != expected:
                raise ValueError(f"v4 feature source hash differs: {name}")
        for name, expected in metadata["evaluator_source_sha256"].items():
            if sha(
                root / "third_party/instruction_following_eval" / name
            ) != expected:
                raise ValueError(f"IFEval grader source differs: {name}")
        if sha(root / "source/joint-v9/collect.py") != (
            metadata["v9_collect_sha256"]
        ):
            raise ValueError("v9 collector source differs")
        if sha(root / "training/behavior-v10.json") != (
            metadata["behavior_v10_sha256"]
        ):
            raise ValueError("v10 behavior diagnostic differs")
        packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True,
        ).splitlines()
        if sorted(packages) != sorted(metadata["python_packages"]):
            raise ValueError("v11 evaluator environment differs")
        verify_models(root)
        sys.path.insert(0, str(root / "source/joint-v9"))
        import eval as v9_eval
        v11_eval = load_module(
            "v11_native_eval", root / "source/joint-v11/eval.py"
        )
        sys.path.insert(0, str(root / "source/joint-v11"))
        import quality
        import score
        select_arms = load_module(
            "v11_arm_select", root / "source/joint-v11/select.py"
        )

        os.environ["NLTK_DATA"] = str(root / "nltk_data")
        evaluator = quality.ifeval_grader(
            root / "third_party/instruction_following_eval"
        )
        arms_dir = root / "policy/arms"
        native = root / "native/eval-results"
        arms = {}
        for phase in ("qualification", "validation"):
            arms_path = arms_dir / f"{phase}-arms.json"
            arms[phase] = json.loads(arms_path.read_text())
            verify_native(
                root, validation_tasks, arms[phase], phase, v9_eval
            )
            observed = quality.score(
                native, validation_tasks, arms[phase], phase, evaluator
            )
            saved_path = root / f"native/{phase}-quality.json"
            if canonical(observed) != json.loads(saved_path.read_text()):
                raise ValueError(f"v11 {phase} quality replay differs")
        observed_qualifier = v11_eval.qualifier(
            SimpleNamespace(
                out=native,
                workloads=root / "inputs/validation-workloads",
                arms=arms_dir / "qualification-arms.json",
            ),
            arms["qualification"],
        )
        qualifier_path = root / "native/qualification-result.json"
        if canonical(observed_qualifier) != json.loads(
            qualifier_path.read_text()
        ):
            raise ValueError("v11 qualifier replay differs")
        score_path = arms_dir / "validation-score.json"
        observed_validation = score.attach_receipts(
            score.score(
                native,
                json.loads(
                    (root / "native/validation-quality.json").read_text()
                ),
                arms["validation"],
            ),
            root / "native/validation-quality.json",
            arms_dir / "validation-arms.json",
        )
        if canonical(observed_validation) != json.loads(score_path.read_text()):
            raise ValueError("v11 validation rate replay differs")
        selected, final, _ = select_arms.choose(
            score_path, arms_dir / "validation-arms.json", qualifier_path
        )
        if canonical(selected) != selection:
            raise ValueError("v11 validation selection replay differs")
        if selection["status"] == "selected":
            final_tasks_path = root / "inputs/final-workloads/manifest.json"
            final_tasks = json.loads(final_tasks_path.read_text())
            check_phase_files(
                root / "inputs/final-workloads", final_tasks,
                ("heldout",),
                ("qualification", "training", "validation", "heldout"),
            )
            if (
                sha(final_tasks_path) != FINAL_SHA
                or manifest["final_workloads_sha256"] != FINAL_SHA
                or any(
                    validation_tasks["groups"][phase]
                    != final_tasks["groups"][phase]
                    for phase in ("qualification", "training", "validation")
                )
                or hashlib.sha256(
                    (json.dumps(
                        final_tasks["groups"]["heldout"],
                        indent=2, sort_keys=True,
                    ) + "\n").encode()
                ).hexdigest() != HELDOUT_COMMIT
                or json.loads(
                    (root / "native/heldout-ready.json").read_text()
                ) != {
                    "phase": "heldout", "manifest_sha256": FINAL_SHA,
                }
                or json.loads(
                    (root / "native/needs-heldout.json").read_text()
                ) != {
                    "phase": "heldout", "manifest_sha256": FINAL_SHA,
                }
            ):
                raise ValueError("heldout phase release differs")
            final_path = arms_dir / "heldout-arms.json"
            arms["heldout"] = json.loads(final_path.read_text())
            if canonical(final) != arms["heldout"]["arms"]:
                raise ValueError("v11 selected final arm inventory differs")
            verify_native(
                root, final_tasks, arms["heldout"], "heldout", v9_eval
            )
            observed_quality = quality.score(
                native, final_tasks, arms["heldout"], "heldout", evaluator
            )
            quality_path = root / "native/heldout-quality.json"
            if canonical(observed_quality) != json.loads(
                quality_path.read_text()
            ):
                raise ValueError("v11 final quality replay differs")
            observed_score = score.attach_receipts(
                score.score(native, observed_quality, arms["heldout"]),
                quality_path, final_path,
            )
            if canonical(observed_score) != json.loads(
                (root / "native/heldout-score.json").read_text()
            ):
                raise ValueError("v11 final rate replay differs")
        else:
            if (
                manifest["phase_counts"]["heldout"] != 0
                or manifest["final_workloads_sha256"] is not None
                or (root / "inputs/final-workloads").exists()
            ):
                raise ValueError("no-go archive contains final arms")
        return {
            "status": "evaluation-native-quality-selection-and-score-match",
            "members": members,
            "selection_status": selection["status"],
            "phase_counts": manifest["phase_counts"],
            "archive_sha256": manifest["archive_sha256"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.archive, args.manifest)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
