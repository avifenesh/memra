"""Extract only pinned code training rows from the replayed v9 archive."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


ARCHIVE_SHA = "a914e20a4f823acbdf189202806415785d60507fa0f775ab588b10083d3c034b"
NEW_SHA = "ecde6665d9627f43ca299ce90602158eab6bd7ae0c0d1ba72228bb7ac5f8dc0a"
OLD_SHA = "196ece7fcd89091615b8c16e249f5f9b3373d46fc8326a1772618ce5565ee073"
K_SHA = "ac5c14da112616d7c0459c3b109e07429ff2aab2cb1dc70f08bad9abb6eb878a"
TABLE_SHA = "34a4b7d67f83f6fa66d967d0bb5f5d74d9aa791469d4c07d6086b401f5962e75"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wanted(name):
    for prefix in ("training/new/", "training/old/"):
        if name.startswith(prefix):
            basename = name.removeprefix(prefix)
            if basename in {"k.jsonl.gz", "d.jsonl.gz", "c.jsonl.gz",
                            "manifest.json"}:
                return Path(prefix.removeprefix("training/")) / basename
    if name == "training/k-models/manifest.json":
        return Path("k-models/manifest.json")
    if name == "training/cd-models/augmented/topk20/depth-history.tsv":
        return Path("table.tsv")
    return None


def prepare(archive, manifest_path, out):
    manifest = json.loads(manifest_path.read_text())
    if manifest["archive_sha256"] != ARCHIVE_SHA or sha(archive) != ARCHIVE_SHA:
        raise ValueError("v9 code training archive differs")
    out.mkdir(exist_ok=False)
    copied = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            relative = wanted(member.name)
            if relative is None:
                continue
            if member.name in copied or not member.isfile():
                raise ValueError("v9 code training member differs")
            expected = manifest["members"][member.name]
            if member.size != expected["bytes"]:
                raise ValueError("v9 code training member size differs")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.extractfile(member) as input_file, target.open("xb") as output:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("v9 code training member hash differs")
            copied.add(member.name)
    if len(copied) != 10 or any(
        sha(out / path) != expected
        for path, expected in (
            ("new/manifest.json", NEW_SHA),
            ("old/manifest.json", OLD_SHA),
            ("k-models/manifest.json", K_SHA),
            ("table.tsv", TABLE_SHA),
        )
    ):
        raise ValueError("v9 code training input inventory differs")
    return {"status": "pinned-code-training-extracted", "members": len(copied)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.manifest, args.out),
                     sort_keys=True))


if __name__ == "__main__":
    main()
