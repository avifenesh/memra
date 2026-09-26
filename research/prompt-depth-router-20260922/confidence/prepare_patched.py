"""Seal the single-file confidence patch and the remote research binaries."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile


BASE_ARCHIVE_SHA256 = "53cfab8e8a4c4a01362d54d9d96fd69100f99ab445b62ba81be1c096a17634bf"
SPEC = "crates/memra-engine/src/spec.rs"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "runtime-source.tar.gz"
    repo = root / "repo-confidence"
    patch = json.loads((root / "source-patch.json").read_text())
    if sha(base) != BASE_ARCHIVE_SHA256:
        raise ValueError("base runtime archive changed")
    if sha(repo / SPEC) != patch["after_sha256"]:
        raise ValueError("patched spec.rs changed")
    if sha(root / "harness/confidence/patch_source.py") != patch["patcher_sha256"]:
        raise ValueError("source patcher changed")

    sealed = root / "runtime-source-confidence.tar.gz"
    if sealed.exists():
        raise ValueError("refusing to overwrite a source archive")
    count = 0
    with tarfile.open(base, "r:gz") as original, sealed.open("xb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for member in original:
                    if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                        raise ValueError("unsafe source archive member")
                    before = original.extractfile(member).read()
                    after = (repo / member.name).read_bytes()
                    if member.name == SPEC:
                        if hashlib.sha256(before).hexdigest() != patch["before_sha256"]:
                            raise ValueError("patched source has another base")
                    elif before != after:
                        raise ValueError("a second source file changed: " + member.name)
                    info = tarfile.TarInfo(member.name)
                    info.size = len(after)
                    info.mode = member.mode
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(after))
                    count += 1

    binaries = root / "binaries-confidence"
    binaries.mkdir(exist_ok=False)
    for name in ("qwen-prefix-study", "run-spec"):
        source = root / "repo/target/release" / name
        shutil.copyfile(source, binaries / name)
        (binaries / name).chmod(0o755)
    source_record = {
        "kind": "qwen-fixed-confidence-research",
        "base_source_recipe_commit": "bdf9f4305b1093c4ac35a8e387c689047c938c5e",
        "base_runtime_source_sha256": sha(base),
        "runtime_source_archive": sealed.name,
        "runtime_source_sha256": sha(sealed),
        "source_members": count,
        "source_patch_receipt_sha256": sha(root / "source-patch.json"),
        "patched_spec_sha256": patch["after_sha256"],
        "source_patcher_sha256": patch["patcher_sha256"],
        "binaries": {"qwen-prefix-study": sha(binaries / "qwen-prefix-study")},
        "oracle_binary_sha256": sha(binaries / "run-spec"),
        "build_log_sha256": sha(root / "confidence-build.log"),
        "runner_sha256": sha(root / "harness/prefix/run.py"),
        "grid_sha256": sha(root / "harness/confidence/fixed_grid.py"),
        "reporter_sha256": sha(root / "harness/confidence/report_fixed.py"),
    }
    destination = root / "source-confidence.json"
    destination.write_text(json.dumps(source_record, indent=2) + "\n")
    print(json.dumps({
        "patched_source_sha256": source_record["runtime_source_sha256"],
        "binary_sha256": source_record["binaries"]["qwen-prefix-study"],
        "oracle_sha256": source_record["oracle_binary_sha256"],
        "source_members": count,
    }))


if __name__ == "__main__":
    main()
