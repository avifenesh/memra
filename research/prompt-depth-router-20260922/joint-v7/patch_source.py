"""Seal the v7 sampled single-head learned-C hook over pinned v6 source."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


SPEC = "crates/memra-engine/src/spec.rs"


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
                raise ValueError("unsafe or duplicated v6 source member")
            seen.add(member.name)
            path = target.joinpath(*logical.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as payload, path.open("xb") as out:
                out.write(payload.read())
    return len(seen)


def seal(source, out):
    directory = Path(__file__).resolve().parent
    patch = directory / "source.patch"
    pins = json.loads((directory / "source-pins.json").read_text())
    if (
        pins["schema"] != 1
        or sha(source) != pins["base_archive_sha256"]
        or sha(patch) != pins["patch_sha256"]
    ):
        raise ValueError("v7 parent source or single-head patch differs")
    out.mkdir(parents=True, exist_ok=False)
    repo = out / "repo"
    repo.mkdir()
    parent_members = unpack(source, repo)
    if sha(repo / SPEC) != pins["before_spec_sha256"]:
        raise ValueError("v7 spec.rs before-patch hash differs")
    for command in (
        ["git", "apply", "--check", str(patch.resolve())],
        ["git", "apply", str(patch.resolve())],
    ):
        subprocess.run(command, cwd=repo, check=True)
    if sha(repo / SPEC) != pins["after_spec_sha256"]:
        raise ValueError("v7 spec.rs after-patch hash differs")
    archive = out / "runtime-source-joint-v7.tar.gz"
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
        "schema": 1,
        "base_archive_sha256": pins["base_archive_sha256"],
        "patch_sha256": pins["patch_sha256"],
        "before_spec_sha256": pins["before_spec_sha256"],
        "after_spec_sha256": pins["after_spec_sha256"],
        "parent_members": parent_members,
        "source_members": members,
        "source_archive_sha256": sha(archive),
        "patcher_sha256": sha(Path(__file__)),
    }
    (out / "source.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.base, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
