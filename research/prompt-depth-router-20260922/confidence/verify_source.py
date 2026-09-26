"""Verify the measured confidence runtime differs from the sealed base once."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from patch_source import BASE_SPEC_SHA256, NEW, OLD


SPEC = "crates/memra-engine/src/spec.rs"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(base, patched, patch_receipt, source_record, binary=None):
    patch = json.loads(patch_receipt.read_text())
    source = json.loads(source_record.read_text())
    if (sha(base) != source["base_runtime_source_sha256"]
            or sha(patched) != source["runtime_source_sha256"]
            or sha(patch_receipt) != source["source_patch_receipt_sha256"]):
        raise ValueError("measured source pin differs")
    if binary is not None and sha(binary) != source["binaries"]["qwen-prefix-study"]:
        raise ValueError("measured binary pin differs")
    if patch["before_sha256"] != BASE_SPEC_SHA256 or patch["after_sha256"] != source["patched_spec_sha256"]:
        raise ValueError("guard patch receipt differs")
    if sha(Path(__file__).with_name("patch_source.py")) != source["source_patcher_sha256"]:
        raise ValueError("guard patcher differs")
    changed = []
    members = 0
    with tarfile.open(base, "r:gz") as old, tarfile.open(patched, "r:gz") as new:
        for a, b in zip(old, new, strict=True):
            if not a.isfile() or not b.isfile() or a.name != b.name:
                raise ValueError("source archive inventory changed")
            before, after = old.extractfile(a).read(), new.extractfile(b).read()
            if before != after:
                changed.append(a.name)
                if (a.name != SPEC or hashlib.sha256(before).hexdigest() != BASE_SPEC_SHA256
                        or hashlib.sha256(after).hexdigest() != patch["after_sha256"]
                        or before.decode().count(OLD) != 1
                        or before.decode().replace(OLD, NEW).encode() != after):
                    raise ValueError("source has a change outside the reviewed guard")
            members += 1
    if changed != [SPEC] or members != source["source_members"]:
        raise ValueError("source changed more than the fixed-confidence guard")
    return {
        "status": "source-and-binary-verified" if binary is not None else "source-verified",
        "members": members,
        "changed": changed,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "patched", "patch_receipt", "source_record"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(
        args.base, args.patched, args.patch_receipt, args.source_record, args.binary
    )))


if __name__ == "__main__":
    main()
