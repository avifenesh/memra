"""Verify a sealed v9 archive and recompute native, quality and score receipts."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def extract(archive, manifest, target):
    expected = manifest["members"]
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in expected or name in seen
                or not member.isfile() or logical.is_absolute()
                or ".." in logical.parts or str(logical) != name
                or member.size != expected[name]["bytes"]
            ):
                raise ValueError(f"unsafe or changed v9 archive member: {name}")
            seen.add(name)
            destination = target.joinpath(*logical.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as input_file, destination.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"member hash differs: {name}")
    if seen != set(expected):
        raise ValueError("sealed archive omits a member")
    return len(seen)


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != 1 or sha(archive) != manifest["archive_sha256"]:
        raise ValueError("v9 archive differs from manifest")
    with tempfile.TemporaryDirectory(prefix="mtp-v9-replay-") as temporary:
        root = Path(temporary)
        members = extract(archive, manifest, root)
        sys.path.insert(0, str(root / "source/joint-v9"))
        import collect
        import eval as native_eval
        import quality
        import qualify
        import score

        inputs = json.loads((root / "inputs/workloads/manifest.json").read_text())
        if sha(root / "inputs/workloads/manifest.json") != manifest["workloads_sha256"]:
            raise ValueError("frozen workload manifest differs")
        for group in inputs["groups"].values():
            for entry in group:
                if sha(root / "inputs/workloads" / entry["file"]) != entry["sha256"]:
                    raise ValueError("frozen conversation text differs")
        native = root / "native/results"
        training = list(native.glob("training-*.result.json"))
        if len(training) != 144:
            raise ValueError("training evidence count differs")
        for path in training:
            row = json.loads(path.read_text())
            index = int(row["name"].split("-")[1])
            entry = inputs["groups"]["training"][index]
            label = row["variant"].split("-", 1)[1]
            observed = collect.verify_session(
                native / row["name"], entry, label, row["k"], "training",
            )
            if observed != row:
                raise ValueError(f"native training result differs: {row['name']}")
        for phase, arms_path in (
            ("qualification", root / "selection/arms/qualification-arms.json"),
            ("validation", root / "selection/arms/validation-arms.json"),
            ("heldout", root / "selection/final/heldout-arms.json"),
        ):
            specs = {
                item["label"]: item
                for item in json.loads(arms_path.read_text())["arms"]
            }
            for path in native.glob(f"{phase}-*.result.json"):
                row = json.loads(path.read_text())
                index = int(row["name"].split("-")[1])
                entry = inputs["groups"][phase][index]
                observed = native_eval.verify(
                    native / row["name"], entry, specs[row["variant"]],
                )
                if observed != row:
                    raise ValueError(f"native {phase} result differs: {row['name']}")
        qualification = qualify.qualify(
            native, root / "selection/arms/qualification-arms.json",
        )
        if qualification != json.loads(
            (root / "native/qualification-result.json").read_text()
        ):
            raise ValueError("no-op or C engagement replay differs")
        for phase, arms_path in (
            ("qualification", None),
            ("validation", root / "selection/arms/validation-arms.json"),
            ("heldout", root / "selection/final/heldout-arms.json"),
        ):
            scored_quality = quality.score(native, inputs, [phase])
            quality_path = root / f"native/{phase}-quality.json"
            if scored_quality != json.loads(quality_path.read_text()):
                raise ValueError(f"{phase} code-quality replay differs")
            if arms_path is not None:
                scored = score.score(native, quality_path, arms_path, phase)
                if scored != json.loads(
                    (root / f"native/{phase}-score.json").read_text()
                ):
                    raise ValueError(f"{phase} pooled request score differs")
    return {
        "status": "archive-hashes-native-KV-noops-C-quality-and-E2E-match",
        "members": members,
        "training_arms": 144,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.archive, args.manifest)
    with args.out.open("x") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
