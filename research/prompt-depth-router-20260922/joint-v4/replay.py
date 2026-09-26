"""Verify every sealed v4 member and independently reproduce scored reports."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile


REPORTS = (
    ("analyze_training.py", "training-analysis.json"),
    ("analyze_selection.py", "selection-analysis.json"),
    ("diagnostics.py", "diagnostics.json"),
    ("analyze_final.py", "final-analysis.json"),
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, manifest, target):
    expected = manifest["native_members"]
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = member.name
            logical = PurePosixPath(name)
            if (
                name not in expected or name in seen
                or not member.isfile() or logical.is_absolute()
                or ".." in logical.parts or str(logical) != name
                or member.size != expected[name]["bytes"]
            ):
                raise ValueError("unsafe or unmatched sealed evidence member: " + name)
            seen.add(name)
            digest = hashlib.sha256()
            output = target.joinpath(*logical.parts) if logical.parts[0] in (
                "native", "models",
            ) else None
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source:
                sink = output.open("xb") if output is not None else None
                try:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        if sink is not None:
                            sink.write(chunk)
                finally:
                    if sink is not None:
                        sink.close()
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError("sealed evidence digest differs: " + name)
    if seen != set(expected):
        raise ValueError("sealed archive lacks required evidence members")
    return len(seen)


def replay(archive, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["schema"] != 1
        or sha(archive) != manifest["native_archive_sha256"]
    ):
        raise ValueError("another v4 receipt archive or manifest")
    with tempfile.TemporaryDirectory(prefix="joint-v4-replay-") as tmp:
        base = Path(tmp)
        members = unpack(archive, manifest, base)
        native = base / "native"
        lane = Path(__file__).resolve().parent
        for program, result_name in REPORTS:
            generated = base / result_name
            subprocess.run(
                [sys.executable, str(lane / program), "--root", str(native),
                 "--out", str(generated)],
                check=True, capture_output=True, text=True,
            )
            frozen = native / result_name
            if not frozen.is_file() or json.loads(generated.read_text()) != json.loads(
                frozen.read_text()
            ):
                raise ValueError("independent v4 report differs: " + result_name)
    return {"status": "all-hashes-and-reports-match", "members": members}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.archive, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
