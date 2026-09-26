"""Verify the closed data inventory before reconstructing native results."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from native_report import inspect, markdown

FILES = {
    "native-data.tar.gz", "native-data.members.jsonl",
    "runtime-source.tar.gz", "harness-source.tar.gz",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(receipts, expected_pin, destination):
    if sha(receipts / "manifest.json") != expected_pin:
        raise ValueError("native manifest identity changed")
    manifest = json.loads((receipts / "manifest.json").read_text())
    if set(manifest["files"]) != FILES:
        raise ValueError("native archive inventory changed")
    if {p.name for p in receipts.iterdir()} != FILES | {"manifest.json", "manifest.sha256"}:
        raise ValueError("unexpected native receipt files")
    for name, entry in manifest["files"].items():
        path = receipts / name
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("native archive identity changed: " + name)
    members = [json.loads(line) for line in (receipts / "native-data.members.jsonl").read_text().splitlines()]
    expected = {}
    for row in members:
        name = row["file"]
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name in expected:
            raise ValueError("unsafe or duplicate receipt member")
        if type(row["bytes"]) is not int or not 0 <= row["bytes"] <= 64 * 1024 * 1024:
            raise ValueError("receipt member exceeds size bound")
        expected[name] = row
    if len(expected) != manifest["member_count"] or sum(x["bytes"] for x in members) != manifest["expanded_data_bytes"]:
        raise ValueError("expanded inventory counts changed")
    if manifest["expanded_data_bytes"] > 4 * 1024**3:
        raise ValueError("expanded native receipts exceed bound")
    seen = set()
    with tarfile.open(receipts / "native-data.tar.gz") as archive:
        for member in archive:
            row = expected.get(member.name)
            if row is None or member.name in seen or not member.isfile() or member.size != row["bytes"]:
                raise ValueError("archive/member inventory mismatch")
            data = archive.extractfile(member).read()
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("native member bytes changed")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("native archive is incomplete")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--parent-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="prompt-depth-native-replay-") as directory:
        root = Path(directory)
        manifest = unpack(args.receipts, args.manifest_sha256, root)
        source = json.loads((root / "source.json").read_text())
        build = json.loads((root / "native-build-source.json").read_text())
        if (sha(root / "native-build-source.json") != source["native_build_metadata_sha256"]
                or build["source_recipe_commit"] != source["source_recipe_commit"]
                or build["binaries"] != source["binaries"]
                or sha(args.receipts / "runtime-source.tar.gz") != source["runtime_source_sha256"]):
            raise ValueError("compiled runtime/source bindings disagree")
        runtime_seen = set()
        binding_digest = None
        with tarfile.open(args.receipts / "runtime-source.tar.gz") as archive:
            for member in archive:
                expected = build["source_files"].get(member.name)
                if (not member.isfile() or member.name in runtime_seen or expected is None
                        or member.size != expected["bytes"]):
                    raise ValueError("runtime source inventory changed")
                payload = archive.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError("runtime source member changed")
                if member.name == "REQUEST-ROUTING-SOURCE.json":
                    binding_digest = hashlib.sha256(payload).hexdigest()
                runtime_seen.add(member.name)
        if runtime_seen != set(build["source_files"]):
            raise ValueError("runtime source archive is incomplete")
        freeze = json.loads((root / "native/FREEZE.json").read_text())
        if binding_digest != freeze["runtime_binding_sha256"]:
            raise ValueError("native program differs from the executed source freeze")
        for name in ("router.rs", "request_routing.rs"):
            path = "crates/memra-engine/src/bin/depth_study_io/" + name
            if build["source_files"][path]["sha256"] != source["harness_files"][name]:
                raise ValueError("native and CPU routing source differ")
        result = inspect(
            root, args.parent_archive, args.receipts / "harness-source.tar.gz",
            manifest["files"]["harness-source.tar.gz"]["sha256"],
        )
        text = markdown(result)
        (args.out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out / "RESULTS.md").write_text(text)
        if args.check and args.check.read_text() != text:
            raise ValueError("native report does not reproduce")
        print(text)


if __name__ == "__main__":
    main()
