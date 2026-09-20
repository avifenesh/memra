"""Actual checkout and model-input boundaries for native qualification producers."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import stat
import struct

import release_qualification as q


def actual_blob(path, mode):
    """Hash compiler-visible bytes without Git index flags, filters or stat-cache hints."""
    info = path.lstat()
    if mode == "120000":
        q.require(stat.S_ISLNK(info.st_mode), f"tracked symlink changed type: {path}")
        data = os.fsencode(os.readlink(path))
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    q.require(stat.S_ISREG(info.st_mode), f"tracked input changed type: {path}")
    q.require(bool(info.st_mode & stat.S_IXUSR) == (mode == "100755"),
              f"tracked input executable mode changed: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        q.require((before.st_dev, before.st_ino) == (info.st_dev, info.st_ino),
                  f"tracked input replaced before hashing: {path}")
        h = hashlib.sha1(b"blob " + str(before.st_size).encode() + b"\0")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
        q.require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                  (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                  f"tracked input changed while hashing: {path}")
    return h.hexdigest()


def verify_checkout(repo, head="HEAD"):
    repo = repo.resolve()
    head = q.commit(repo, head)
    expected = q.tree_files(repo, head)
    links = q.source_symlink_targets(repo, head, expected)
    for name, entry in expected.items():
        path = repo / name
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise q.GateError(f"tracked input cannot resolve: {name}") from error
        q.require(resolved.is_relative_to(repo), f"tracked input escaped checkout: {name}")
        if name in links:
            q.require(resolved == repo / links[name], f"actual symlink closure differs from Git: {name}")
        q.require(actual_blob(path, entry["mode"]) == entry["blob"],
                  f"actual tracked input differs from Git source: {name}")
    # Ignored build inputs must not disappear behind .gitignore. Cargo output is
    # outside these source roots; Python caches are redirected by the producer.
    for ignored in (False, True):
        args = ["ls-files", "--others", "--exclude-standard", "-z"]
        if ignored:
            args.append("--ignored")
        args.extend(["--", ".cargo", "crates", "tools", "Cargo.toml", "Cargo.lock",
                     "rust-toolchain", "rust-toolchain.toml", "build.rs"])
        for raw in q.git(repo, *args).split(b"\0"):
            if not raw:
                continue
            name = raw.decode()
            if "__pycache__" in Path(name).parts and name.endswith(".pyc"):
                continue  # the producer/children never consume this cache
            raise q.GateError(f"untracked or ignored build/gate input: {name}")
    configs = []
    for directory in (repo, *repo.parents):
        for name in (".cargo/config", ".cargo/config.toml"):
            path = directory / name
            if not path.exists() and not path.is_symlink():
                continue
            q.require(directory == repo and name in expected,
                      f"unrecorded ancestor/Cargo configuration: {path}")
            # The repository currently needs only build.jobs. Refuse env forcing,
            # target overrides, rustflags and compiler wrappers rather than assert
            # DOCS_RS/architecture values that Cargo can silently override.
            text = "\n".join(line.split("#", 1)[0] for line in path.read_text().splitlines()).strip()
            q.require(not text or re.fullmatch(r"\[build\]\s+jobs\s*=\s*[1-9][0-9]*\s*", text),
                      f"unsupported effective Cargo configuration: {name}; only build.jobs is admitted")
            configs.append({"path": name, "sha256": q.sha256_file(path)})
    return configs


def single_file_gguf(path):
    """Refuse split models by parsed metadata, even if their entry file was renamed.

    Qualification of split inputs is intentionally unsupported until a complete
    loaded-shard closure is recorded. Only headers are read; tensor bytes are hashed
    by the caller after this format boundary succeeds.
    """
    scalar = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
              4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("B", 1),
              10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        def read(n):
            q.require(n >= 0 and stream.tell() + n <= size, f"truncated GGUF header: {path}")
            data = stream.read(n)
            q.require(len(data) == n, f"short GGUF read: {path}")
            return data
        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]
        def skip(n):
            q.require(n >= 0 and stream.tell() + n <= size, f"invalid GGUF metadata extent: {path}")
            stream.seek(n, 1)
        def value(kind, keep=False):
            if kind in scalar:
                fmt, count = scalar[kind]
                return number(fmt) if keep else skip(count)
            q.require(not keep, f"invalid split.count metadata type: {path}")
            if kind == 8:
                return skip(number("Q"))
            q.require(kind == 9, f"unknown GGUF metadata type: {path}")
            element, count = number("I"), number("Q")
            q.require(count <= size and element != 9, f"invalid GGUF metadata array: {path}")
            if element in scalar:
                return skip(count * scalar[element][1])
            q.require(element == 8, f"unknown GGUF array type: {path}")
            for _ in range(count):
                skip(number("Q"))
        q.require(read(4) == b"GGUF" and number("I") in (2, 3), f"unsupported GGUF input: {path}")
        number("Q")  # tensor count; no tensor contents are interpreted here
        entries = number("Q")
        q.require(entries <= 1_000_000, f"excessive GGUF metadata entries: {path}")
        split_seen = False
        for _ in range(entries):
            length = number("Q")
            q.require(length <= 1_048_576, f"oversized GGUF metadata key: {path}")
            key = read(length).decode("utf-8")
            kind = number("I")
            if key == "split.count":
                q.require(not split_seen and kind in (0, 1, 2, 3, 4, 5, 10, 11),
                          f"invalid or duplicated split.count: {path}")
                split_seen = True
                count = value(kind, True)
                q.require(0 <= count <= 1,
                          f"split GGUF qualification is unsupported; full shard closure required: {path}")
            else:
                value(kind)
