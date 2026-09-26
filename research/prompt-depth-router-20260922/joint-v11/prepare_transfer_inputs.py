"""Verify v10 custody and extract only pinned non-code grader inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
BINARY_SHA = "84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f"
WORKLOAD_SHA = "dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(name):
    if name.startswith("third_party/instruction_following_eval/"):
        return Path(name)
    if name.startswith("nltk_data/"):
        return Path(name)
    return None


def prepare(archive, manifest_path, custody_path, out):
    manifest = json.loads(manifest_path.read_text())
    custody = json.loads(custody_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["archive_sha256"]
        or manifest["model_sha256"] != MODEL_SHA
        or manifest["binary_sha256"] != BINARY_SHA
        or manifest["workloads_sha256"] != WORKLOAD_SHA
        or custody["status"] != "verified-training-projection"
        or custody["archive_sha256"] != manifest["archive_sha256"]
        or custody["parent_archive_sha256"]
        != manifest["parent_archive_sha256"]
        or custody["parent_replay"]["status"]
        != "archive-native-IFEval-GSM8K-noops-and-E2E-match"
        or custody["parent_replay"]["heldout_arms"] != 320
        or custody["sessions"] != 256
        or any(name.startswith("inputs/datasets/")
               for name in manifest["members"])
    ):
        raise ValueError("v10 training projection lacks parent replay custody")
    out.mkdir(exist_ok=False)
    copied = set()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or set(names) != set(manifest["members"])
        ):
            raise ValueError("v10 projected tar contains undeclared members")
        for member in source:
            relative = wanted(member.name)
            if relative is None:
                continue
            if member.name in copied or not member.isfile():
                raise ValueError("v10 grader input member differs")
            expected = manifest["members"][member.name]
            if member.size != expected["bytes"]:
                raise ValueError("v10 grader input size differs")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as input_file, target.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("v10 grader input hash differs")
            copied.add(member.name)
    if (
        not (out / "third_party/instruction_following_eval/evaluation_lib.py").is_file()
        or not (out / "nltk_data").is_dir()
    ):
        raise ValueError("v10 pinned grader source differs")
    return {
        "status": "pinned-transfer-grader-inputs-extracted",
        "members": len(copied),
        "v10_archive_sha256": manifest["archive_sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--custody", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        prepare(args.archive, args.manifest, args.custody, args.out),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
