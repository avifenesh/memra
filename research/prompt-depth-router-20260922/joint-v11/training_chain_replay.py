"""Refit V11 policies from copied V10/V11 archives and compare sealed outputs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def run(source_root, name, *argv):
    command = [sys.executable, str(source_root / "joint-v11" / f"{name}.py"),
               *map(str, argv)]
    result = subprocess.run(
        command, cwd=source_root.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=3 * 3600 if name in (
            "feature_audit", "fit_mixed"
        ) else 3600,
    )
    if result.returncode:
        raise RuntimeError(
            f"copied-byte {name} replay failed: "
            f"{result.stderr[-900:]}"
        )


def extract_copied_inputs(archive, manifest, root):
    prefixes = (
        "source/joint-v11/", "source/joint-v9/",
        "source/joint-v4/", "inputs/code-training/",
        "policy/arms/",
    )
    wanted = {
        name: info for name, info in manifest["members"].items()
        if name.startswith(prefixes)
    }
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name
            if name not in wanted:
                continue
            logical = PurePosixPath(name)
            if (
                name in seen or not member.isfile()
                or logical.is_absolute() or ".." in logical.parts
                or member.size != wanted[name]["bytes"]
            ):
                raise ValueError(f"copied source member differs: {name}")
            path = root.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as input_file, path.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != wanted[name]["sha256"]:
                raise ValueError(f"copied source member hash differs: {name}")
            seen.add(name)
    if seen != set(wanted):
        raise ValueError("copied source archive omits a member")
    if sha(root / "source/joint-v11/training_chain_replay.py") != (
        sha(Path(__file__))
    ):
        raise ValueError("live chain orchestrator differs from copied source")
    return root / "source", root / "inputs/code-training"


def normalized_arms(value):
    result = json.loads(json.dumps(value))
    for arm in result["arms"]:
        extra = []
        for option in arm.get("extra", []):
            key, sep, text = option.partition("=")
            if sep and key in (
                "topk-model", "depth-model",
                "confidence-model", "joint-model-dir",
            ):
                parts = Path(text).parts
                marker = (
                    "policy-models" if "policy-models" in parts
                    else "models"
                )
                if marker not in parts:
                    raise ValueError("arm model path leaves fitted weights")
                text = "<models>/" + "/".join(
                    parts[parts.index(marker) + 1:]
                )
            extra.append(key + sep + text)
        arm["extra"] = extra
    return result


def match_group(root, prefix, members):
    observed = {
        f"{prefix}/{path.relative_to(root).as_posix()}": path
        for path in root.rglob("*") if path.is_file()
    }
    expected = {
        name: info for name, info in members.items()
        if name.startswith(prefix + "/")
    }
    if set(observed) != set(expected):
        raise ValueError(f"copied-byte {prefix} inventory differs")
    for name, path in observed.items():
        if sha(path) != expected[name]["sha256"]:
            raise ValueError(f"copied-byte {prefix} file differs: {name}")
    return len(observed)


def replay(v10, training, train_replay, evaluation, out):
    v10_manifest = json.loads((v10 / "manifest.json").read_text())
    training_manifest = json.loads(
        (training / "manifest.json").read_text()
    )
    eval_manifest = json.loads(
        (evaluation / "manifest.json").read_text()
    )
    if (
        sha(v10 / "native-data.tar.gz")
        != v10_manifest["archive_sha256"]
        or sha(v10 / "manifest.json")
        != eval_manifest["members"][
            "inputs/v10-parent-manifest.json"
        ]["sha256"]
        or sha(v10 / "custody.json")
        != eval_manifest["members"][
            "inputs/v10-parent-custody.json"
        ]["sha256"]
        or v10_manifest["parent_archive_sha256"]
        != eval_manifest["v10_parent_archive_sha256"]
        or sha(training / "native-data.tar.gz")
        != training_manifest["archive_sha256"]
        or sha(evaluation / "native-data.tar.gz")
        != eval_manifest["archive_sha256"]
        or eval_manifest["v10_archive_sha256"]
        != v10_manifest["archive_sha256"]
        or eval_manifest["training_archive_sha256"]
        != training_manifest["archive_sha256"]
    ):
        raise ValueError("copied V10/V11 chain parent hash differs")
    out.mkdir(exist_ok=False)
    source_root, code = extract_copied_inputs(
        evaluation / "native-data.tar.gz",
        eval_manifest,
        out / "copied",
    )
    rows = out / "rows"
    run(
        source_root, "measurement_rows",
        "--v10-archive", v10 / "native-data.tar.gz",
        "--v10-manifest", v10 / "manifest.json",
        "--v10-custody", v10 / "custody.json",
        "--v11-archive", training / "native-data.tar.gz",
        "--v11-manifest", training / "manifest.json",
        "--v11-replay", train_replay,
        "--out", rows,
    )
    feature = out / "feature-audit.json"
    run(source_root, "feature_audit", "--rows", rows, "--out", feature)
    prefix = out / "prefix-preflight.json"
    run(
        source_root, "preflight_visibility",
        "--archive", training / "native-data.tar.gz",
        "--manifest", training / "manifest.json",
        "--replay", train_replay, "--out", prefix,
    )
    models = out / "models"
    run(
        source_root, "fit_mixed",
        "--v9-new", code / "new",
        "--v9-old", code / "old",
        "--v9-k-manifest", code / "k-models/manifest.json",
        "--v9-table", code / "table.tsv",
        "--v11-rows", rows, "--out", models,
    )
    arms = out / "arms"
    run(
        source_root, "arms",
        "--models", models,
        "--training-rows", rows,
        "--visibility-preflight", prefix,
        "--out", arms,
    )
    for phase in ("qualification", "validation"):
        generated = normalized_arms(json.loads(
            (arms / f"{phase}-arms.json").read_text()
        ))
        archived = normalized_arms(json.loads(
            (
                out / "copied/policy/arms"
                / f"{phase}-arms.json"
            ).read_text()
        ))
        if generated != archived:
            raise ValueError(f"copied-byte {phase} label-to-model mapping differs")
    members = eval_manifest["members"]
    count = sum((
        match_group(code, "inputs/code-training", members),
        match_group(rows, "training/rows", members),
        match_group(models, "policy/models", members),
    ))
    for path, name in (
        (feature, "training/feature-audit.json"),
        (prefix, "training/prefix-preflight.json"),
    ):
        if sha(path) != members[name]["sha256"]:
            raise ValueError(f"copied-byte training diagnostic differs: {name}")
    return {
        "schema": 1,
        "status": "copied-V10-V11-rows-and-policy-fit-match",
        "v10_archive_sha256": v10_manifest["archive_sha256"],
        "training_archive_sha256": training_manifest["archive_sha256"],
        "evaluation_archive_sha256": eval_manifest["archive_sha256"],
        "rows_manifest_sha256": sha(rows / "manifest.json"),
        "models_manifest_sha256": sha(models / "manifest.json"),
        "regenerated_arm_labels": len(json.loads(
            (arms / "validation-arms.json").read_text()
        )["arms"]),
        "matched_files": count + 2,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "v10", "training", "training-replay", "evaluation",
        "out", "receipt",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "v10", "training", "training_replay", "evaluation",
        "out", "receipt",
    ):
        setattr(args, name, getattr(args, name).resolve())
    result = replay(
        args.v10, args.training, args.training_replay,
        args.evaluation, args.out,
    )
    with args.receipt.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
