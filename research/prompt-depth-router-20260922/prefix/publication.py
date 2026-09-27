"""Verify a sealed scientific projection and reproduce its native comparison."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from audit import sha
from reproduce import reproduce


def unpack(archive_path, destination, expected):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("archive has missing, duplicate or extra members")
        for member in members:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe scientific archive member")
            record = expected[member.name]
            if member.size != record["bytes"]:
                raise ValueError("archive member length differs")
            path = destination.joinpath(*name.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, path.open("xb") as out:
                shutil.copyfileobj(src, out)
            if sha(path) != record["sha256"]:
                raise ValueError("archive member digest differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipts, out = args.receipts.resolve(), args.out.resolve()
    if sha(receipts / "manifest.json") != args.manifest_sha256:
        raise ValueError("scientific manifest differs from its external pin")
    manifest = json.loads((receipts / "manifest.json").read_text())
    expected_archives = {"native-data.tar.gz", "runtime-source.tar.gz", "harness-source.tar.gz"}
    if set(manifest["files"]) != expected_archives:
        raise ValueError("scientific archive coverage changed")
    for name, expected in manifest["files"].items():
        path = receipts / name
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("sealed archive differs: " + name)
    out.mkdir(parents=True, exist_ok=False)
    root = out / "data"
    unpack(receipts / "native-data.tar.gz", root, manifest["raw_members"])
    source = json.loads((root / "source.json").read_text())
    if (source["source_recipe_commit"] != manifest["source_recipe_commit"]
            or source["harness_commit"] != manifest["harness_commit"]
            or source["runtime_source_sha256"] != manifest["files"]["runtime-source.tar.gz"]["sha256"]
            or source["harness_source_sha256"] != manifest["files"]["harness-source.tar.gz"]["sha256"]
            or source["native_build_metadata_sha256"] != sha(root / "native-build-source.json")):
        raise ValueError("native source/build/staging identities differ")
    with tarfile.open(receipts / "harness-source.tar.gz") as archive:
        members = archive.getmembers()
        if len(members) != len(source["harness_files"]) or {m.name for m in members} != set(source["harness_files"]):
            raise ValueError("staged measurement source inventory changed")
        for member in members:
            if not member.isfile() or hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() != source["harness_files"][member.name]:
                raise ValueError("staged measurement source differs")
    result = reproduce(root, receipts / "runtime-source.tar.gz",
                       manifest["files"]["runtime-source.tar.gz"]["sha256"], out / "RESULTS.json")
    print(json.dumps(result["independent_replay"]))


if __name__ == "__main__":
    main()
