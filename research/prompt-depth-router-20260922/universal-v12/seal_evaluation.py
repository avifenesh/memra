"""Seal mixed native, quality, and selection receipts without credentials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile


TRAIN_SHA = "2bca403c1edff44a1d707abd60710767f9dbbe2a63721757032b0fbe223c5e00"
VALIDATION_SHA = "bf920b82e0176c4304bcc562ccce34a28384082c888355a280620163a45e5313"
FULL_SHA = "b35374d34e2a28b31cca0399e6394fa3db3b0f27e9d593b17032a7856b2a5477"
MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def add_tree(files, prefix, root):
    if not root.is_dir():
        raise ValueError(f"evaluation evidence group missing: {prefix}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("evaluation evidence contains symlink")
        if path.is_file() and path.suffix != ".pyc":
            files[f"{prefix}/{path.relative_to(root).as_posix()}"] = path


def inventory(base, selected):
    files = {}
    for prefix, root in (
        ("native/eval-results", base / "eval-results"),
        ("models/policy-models", base / "policy-models"),
        ("models/policy-arms", base / "policy-arms"),
        ("inputs/phase-validation", base / "phase-validation"),
        ("quality/validation-packets", base / "validation-packets"),
        ("quality/validation-judge", base / "validation-judge"),
        ("source/universal-v12", base / "universal-v12"),
    ):
        add_tree(files, prefix, root)
    if selected:
        for prefix, root in (
            ("inputs/phase-final", base / "phase-final"),
            ("quality/final-packets", base / "final-packets"),
            ("quality/final-judge", base / "final-judge"),
        ):
            add_tree(files, prefix, root)
    for name in (
        "run-meta.json", "rental.json", "cuda-accept.json",
        "judge-config.json", "wildbench-pairwise-template.md",
        "qualification-result.json", "validation-task-quality.json",
        "validation-prose-quality.json",
    ) + ((
        "final-task-quality.json", "final-prose-quality.json",
        "final-score.json", "final-ready.json",
    ) if selected else ()):
        path = base / name
        if not path.is_file():
            raise ValueError(f"evaluation receipt missing: {name}")
        files[f"inputs/{name}"] = path
    for name in ("validation-ready.json", "fresh-replay.json"):
        path = base / name
        if not path.is_file():
            raise ValueError(f"evaluation lineage receipt missing: {name}")
        files[f"inputs/{name}"] = path
    return dict(sorted(files.items()))


def seal(base, out):
    if (
        sha(base / "Qwen3.8-27B-NVFP4-Q5K-mtp.gguf") != MODEL_SHA
        or sha(base / "mtp-depth-study") != BINARY_SHA
        or sha(base / "phase-training/manifest.json") != TRAIN_SHA
        or sha(base / "phase-validation/manifest.json")
        != VALIDATION_SHA
        or sha(base / "judge-config.json")
        != "624cbb8478326ec7662d6e5aaa959e713cb3bf0330128dd42a7e0dc9b8a05bdd"
    ):
        raise ValueError("mixed evaluation model, source or judge changed")
    arms = base / "policy-arms"
    selected_record = json.loads(
        (arms / "shared-selected.json").read_text()
    )
    meta = json.loads((base / "run-meta.json").read_text())
    if (
        selected_record["gpu_uuid"] != meta["gpu_uuid"]
        or meta["customer_capture"] is not False
    ):
        raise ValueError("mixed result moved physical GPU or host role")
    selected = selected_record["status"] == "selected"
    if selected_record["status"] not in ("selected", "global-no-go"):
        raise ValueError("mixed evaluation selection status differs")
    if selected != (base / "phase-final").exists():
        raise ValueError("final phase released without shared selection")
    if selected and sha(
        base / "phase-final/manifest.json"
    ) != FULL_SHA:
        raise ValueError("mixed final prompt package changed")
    if not selected and (
        (arms / "final-arms.json").exists()
        or (base / "final-judge").exists()
        or (base / "final-score.json").exists()
    ):
        raise ValueError("no-go opened final result")
    qualifier = json.loads(
        (arms / "qualification-arms.json").read_text()
    )
    validation = json.loads(
        (arms / "validation-arms.json").read_text()
    )
    final = (
        json.loads((arms / "final-arms.json").read_text())
        if selected else None
    )
    expected_sessions = (
        3 * len(qualifier["arms"])
        + 3 * 8 * len(validation["arms"])
        + (3 * 24 * len(final["arms"]) if final else 0)
    )
    results = list((base / "eval-results").glob("*.result.json"))
    if len(results) != expected_sessions or (
        any(
            json.loads(path.read_text())["name"]
            != path.name.removesuffix(".result.json")
            for path in results
        )
    ):
        raise ValueError("mixed evaluation native session count differs")
    files = inventory(base, selected)
    if out.exists():
        raise ValueError("evaluation seal destination exists")
    out.mkdir()
    members = {
        name: {"bytes": path.stat().st_size, "sha256": sha(path)}
        for name, path in files.items()
    }
    archive = out / "native-data.tar.gz"
    with archive.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0,
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as tar:
                for name, path in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    manifest = {
        "schema": 1,
        "scope": "one shared C/K/D mixed-domain native and quality result",
        "model_sha256": MODEL_SHA,
        "binary_sha256": BINARY_SHA,
        "training_workloads_sha256": TRAIN_SHA,
        "validation_workloads_sha256": VALIDATION_SHA,
        "source_full_manifest_sha256": FULL_SHA,
        "selection_status": selected_record["status"],
        "gpu_uuid": meta["gpu_uuid"],
        "selection_sha256": sha(arms / "shared-selected.json"),
        "validation_score_sha256":
        sha(arms / "validation-score.json"),
        "final_score_sha256":
        sha(base / "final-score.json") if selected else None,
        "training_archive_sha256":
        sha(base / "training-sealed/native-data.tar.gz"),
        "expected_native_sessions": expected_sessions,
        "archive_sha256": sha(archive),
        "members": members,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "selection_status": manifest["selection_status"],
        "archive_sha256": manifest["archive_sha256"],
        "members": len(members),
        "sessions": expected_sessions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        seal(args.base.resolve(), args.out.resolve()),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
