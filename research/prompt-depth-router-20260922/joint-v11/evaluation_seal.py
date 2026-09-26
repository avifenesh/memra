"""Seal v11 validation and final native receipts without model weights."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


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


def files_from(prefix, root):
    if not root.is_dir():
        raise ValueError(f"evaluation evidence group is missing: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evaluation evidence contains a symlink: {path}")
        if path.is_file() and path.suffix != ".pyc":
            yield f"{prefix}/{path.relative_to(root).as_posix()}", path


def check_workload_files(root, phases, groups):
    manifest = json.loads((root / "manifest.json").read_text())
    if set(manifest["groups"]) != set(groups):
        raise ValueError("evaluation workload phase metadata differs")
    expected = {"manifest.json"}
    for phase in phases:
        for domain in ("ifeval", "gsm8k"):
            for entry in manifest["groups"][phase][domain]:
                name = entry["file"]
                if Path(name).name != name:
                    raise ValueError("evaluation prompt path leaves phase root")
                if sha(root / name) != entry["sha256"]:
                    raise ValueError("evaluation prompt bytes differ")
                expected.add(name)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if (
        actual != expected
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ValueError("evaluation workload file allowlist differs")
    return manifest


def inventory(base):
    training_tasks = check_workload_files(
        base / "workloads",
        ("qualification", "training"),
        ("qualification", "training"),
    )
    validation_tasks = check_workload_files(
        base / "validation-workloads",
        ("qualification", "training", "validation"),
        ("qualification", "training", "validation"),
    )
    groups = {
        "source/joint-v11": base / "joint-v11",
        "source/joint-v9": base / "joint-v9",
        "source/joint-v4": base / "joint-v4",
        "source/private_ops": base / "ops",
        "third_party/instruction_following_eval": (
            base / "transfer-inputs/third_party/instruction_following_eval"
        ),
        "nltk_data": base / "transfer-inputs/nltk_data",
        "inputs/workloads": base / "workloads",
        "inputs/validation-workloads": base / "validation-workloads",
        "inputs/code-training": base / "code-training",
        "training/rows": base / "training-rows",
        "policy/models": base / "policy-models",
        "policy/arms": base / "policy-arms",
        "native/eval-results": base / "eval-results",
    }
    files = {}
    for prefix, root in groups.items():
        files.update(files_from(prefix, root))
    for name, source in (
        ("native/run-meta.json", base / "run-meta.json"),
        ("inputs/rental.json", base / "rental.json"),
        ("inputs/source-manifest.json", base / "source-manifest.json"),
        ("inputs/v9-parent-manifest.json", base / "v9-parent/manifest.json"),
        ("inputs/v10-parent-manifest.json", base / "v10-parent/manifest.json"),
        ("inputs/v10-parent-custody.json", base / "v10-parent/custody.json"),
        ("training/manifest.json", base / "training-sealed/manifest.json"),
        ("training/replay.json", base / "training-replay.json"),
        ("training/feature-audit.json", base / "feature-audit.json"),
        ("training/behavior-v10.json", base / "behavior-v10.json"),
        ("training/prefix-preflight.json", base / "prefix-preflight.json"),
        ("native/qualification-result.json", base / "qualification-result.json"),
        ("native/qualification-quality.json", base / "qualification-quality.json"),
        ("native/validation-quality.json", base / "validation-quality.json"),
        ("native/visibility-validation.json", base / "visibility-validation.json"),
        ("native/needs-validation.json", base / "needs-validation.json"),
        ("native/validation-ready.json", base / "validation-ready.json"),
    ):
        if not source.is_file():
            raise ValueError(f"evaluation receipt is missing: {name}")
        files[name] = source
    selection = json.loads((base / "policy-arms/selected.json").read_text())
    if selection["status"] not in ("selected", "no-go"):
        raise ValueError("v11 validation selection differs")
    final_arms = 0
    if selection["status"] == "selected":
        check_workload_files(
            base / "final-workloads",
            ("heldout",),
            ("qualification", "training", "validation", "heldout"),
        )
        files.update(files_from(
            "inputs/final-workloads", base / "final-workloads"
        ))
        files["native/needs-heldout.json"] = base / "needs-heldout.json"
        files["native/heldout-ready.json"] = base / "heldout-ready.json"
        final = json.loads(
            (base / "policy-arms/heldout-arms.json").read_text()
        )
        final_arms = len(final["arms"])
        for name in ("heldout-quality.json", "heldout-score.json"):
            path = base / name
            if not path.is_file():
                raise ValueError(f"v11 final receipt is missing: {name}")
            files[f"native/{name}"] = path
    elif (
        (base / "policy-arms/heldout-arms.json").exists()
        or (base / "final-workloads").exists()
        or (base / "needs-heldout.json").exists()
        or (base / "heldout-ready.json").exists()
        or (base / "heldout-quality.json").exists()
        or (base / "heldout-score.json").exists()
    ):
        raise ValueError("v11 no-go may not carry final output")
    counts = {
        "qualification": 2 * 19,
        "validation": 2 * 8 * 19,
        "heldout": 2 * 16 * final_arms,
    }
    for phase, expected in counts.items():
        actual = len(list(
            (base / "eval-results").glob(f"{phase}-*.result.json")
        ))
        if actual != expected:
            raise ValueError(f"v11 {phase} native inventory differs")
    if (base / "pipeline-failed.json").exists():
        raise ValueError("failed v11 pipeline cannot be sealed")
    metadata = json.loads((base / "run-meta.json").read_text())
    training = json.loads(
        (base / "training-sealed/manifest.json").read_text()
    )
    replay = json.loads((base / "training-replay.json").read_text())
    if (
        sha(base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf") != MODEL_SHA
        or sha(base / "mtp-depth-study") != BINARY_SHA
        or sha(base / "workloads/manifest.json") != WORKLOAD_SHA
        or sha(base / "validation-workloads/manifest.json")
        != VALIDATION_SHA
        or metadata["model_sha256"] != MODEL_SHA
        or metadata["binary_sha256"] != BINARY_SHA
        or metadata["workloads_sha256"] != WORKLOAD_SHA
        or metadata["validation_workloads_sha256"] != VALIDATION_SHA
        or metadata["full_workloads_sha256"] != FINAL_SHA
        or metadata["heldout_group_sha256"] != HELDOUT_COMMIT
        or metadata["source_manifest_sha256"]
        != sha(base / "source-manifest.json")
        or replay["status"]
        != "training-native-random-K-D-C-KV-replay-match"
        or replay["sessions"] != 224
        or replay["archive_sha256"] != training["archive_sha256"]
    ):
        raise ValueError("v11 model, source or training parent differs")
    if selection["status"] == "selected" and (
        sha(base / "final-workloads/manifest.json") != FINAL_SHA
    ):
        raise ValueError("v11 final workload release differs")
    if (
        set(training_tasks["groups"]) != {"qualification", "training"}
        or set(validation_tasks["groups"])
        != {"qualification", "training", "validation"}
        or training_tasks["validation_projection_sha256"] != VALIDATION_SHA
        or training_tasks["source_full_manifest_sha256"] != FINAL_SHA
        or validation_tasks["source_full_manifest_sha256"] != FINAL_SHA
        or validation_tasks["heldout_group_sha256"] != HELDOUT_COMMIT
    ):
        raise ValueError("v11 phase projections differ")
    return dict(sorted(files.items())), counts, selection["status"]


def seal(base, out):
    files, counts, status = inventory(base)
    if out.exists():
        raise ValueError("evaluation seal destination exists")
    out.mkdir()
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in files.items()
    }
    archive = out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name, path in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    training = json.loads(
        (base / "training-sealed/manifest.json").read_text()
    )
    metadata = json.loads((base / "run-meta.json").read_text())
    manifest = {
        "schema": 1,
        "scope": "v11 non-code validation and selected final native evaluation",
        "archive_sha256": sha(archive),
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "workloads_sha256": WORKLOAD_SHA,
        "validation_workloads_sha256": VALIDATION_SHA,
        "full_workloads_sha256": FINAL_SHA,
        "final_workloads_sha256": (
            FINAL_SHA if status == "selected" else None
        ),
        "source_manifest_sha256": sha(base / "source-manifest.json"),
        "run_meta_sha256": sha(base / "run-meta.json"),
        "training_archive_sha256": training["archive_sha256"],
        "v9_archive_sha256": metadata["v9_archive_sha256"],
        "v10_archive_sha256": metadata["v10_archive_sha256"],
        "v10_parent_archive_sha256":
        metadata["v10_parent_archive_sha256"],
        "selection_status": status,
        "phase_counts": counts,
        "members": members,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "archive_sha256": manifest["archive_sha256"],
        "phase_counts": counts,
        "selection_status": status,
        "members": len(members),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.base, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
