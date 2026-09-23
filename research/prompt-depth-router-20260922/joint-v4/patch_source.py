"""Rebuild the exact v4 randomized-depth research source."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


RUST = (
    "crates/memra-engine/src/learned_depth.rs",
    "crates/memra-engine/src/bin/mtp_depth_study.rs",
)
PYTHON = ()
DESTINATION = "research/prompt-depth-router-20260922/joint-v4"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(source, target):
    seen = set()
    with tarfile.open(source, "r:gz") as stream:
        for member in stream:
            logical = PurePosixPath(member.name)
            if (
                not member.isfile() or logical.is_absolute()
                or ".." in logical.parts or str(logical) != member.name
                or member.name in seen
            ):
                raise ValueError("unsafe or duplicated measured source member")
            seen.add(member.name)
            path = target.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as payload, path.open("xb") as out:
                out.write(payload.read())
    return len(seen)


def seal(source, patch, pins_path, out):
    pins = json.loads(pins_path.read_text())
    if pins["schema"] != 1 or sha(source) != pins["base_archive_sha256"]:
        raise ValueError("adaptive source base differs from its sealed parent")
    if sha(patch) != pins["patch_sha256"]:
        raise ValueError("adaptive source patch changed")
    out.mkdir(parents=True, exist_ok=False)
    repo = out / "repo"
    repo.mkdir()
    parent_members = unpack(source, repo)
    if parent_members != pins["parent_members"]:
        raise ValueError("adaptive source parent member count differs")
    for name in RUST:
        if sha(repo / name) != pins["rust"][name]["before_sha256"]:
            raise ValueError("research Rust source differs before patch: " + name)
    for command in (
        ["git", "apply", "--check", str(patch.resolve())],
        ["git", "apply", str(patch.resolve())],
    ):
        subprocess.run(command, cwd=repo, check=True)
    for name in RUST:
        if sha(repo / name) != pins["rust"][name]["after_sha256"]:
            raise ValueError("research Rust source differs after patch: " + name)
    for name in PYTHON:
        source_path = Path(__file__).with_name(name)
        if sha(source_path) != pins["python"][name]:
            raise ValueError("research learner source changed: " + name)
        path = repo / DESTINATION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(source_path.read_bytes())
    archive = out / "runtime-source-joint-v4.tar.gz"
    members = 0
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as stream:
                for path in sorted(repo.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    data = path.read_bytes()
                    info = tarfile.TarInfo(path.relative_to(repo).as_posix())
                    info.size = len(data)
                    info.mtime = info.uid = info.gid = 0
                    info.mode = 0o644
                    info.uname = info.gname = ""
                    stream.addfile(info, BytesIO(data))
                    members += 1
    record = {
        "schema": 1, "base_archive_sha256": pins["base_archive_sha256"],
        "patch_sha256": pins["patch_sha256"],
        "patched_rust": {
            name: sha(repo / name) for name in RUST
        },
        "python": {name: sha(repo / DESTINATION / name) for name in PYTHON},
        "parent_members": parent_members, "source_members": members,
        "source_archive_sha256": sha(archive),
        "patcher_sha256": sha(Path(__file__)),
    }
    with (out / "source.json").open("x") as target:
        target.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    result = seal(
        args.base, directory / "source.patch", directory / "source-pins.json",
        args.out,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
