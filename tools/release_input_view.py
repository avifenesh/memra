"""Compiler-visible source view and Linux filesystem boundary for release builds."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

import release_qualification as q
from release_inputs import actual_blob

POLICY = "fingerprinted-input-view-v1"
SANDBOX_POLICY = "linux-input-view-v1"


def identity_files(source):
    q.require(q.COMMIT.fullmatch(source["commit"]), "invalid input-view Git identity")
    # This supports the build's optional rev-parse field, without objects, refs to
    # other commits, remotes, alternates, hooks or a path to the provenance clone.
    return {".git/HEAD": (source["commit"] + "\n").encode(),
            ".git/config": b"[core]\nrepositoryformatversion = 0\nbare = false\n"}


def expected_files(source):
    files = dict(source["files"])
    q.require(files and all(not q.publication_metadata(p) and not p.startswith(".git/")
                            and p != ".git" for p in files), "metadata in compiler input manifest")
    for name, data in identity_files(source).items():
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        files[name] = {"mode": "100644", "blob": blob}
    return files


def identity(source):
    return q.object_digest({"policy": POLICY, "files": expected_files(source)})


def verify(view, source):
    view = view.resolve()
    expected = expected_files(source)
    directories = {".git", ".git/objects", ".git/refs"}
    q.require(all((view / name).is_dir() and not (view / name).is_symlink() for name in directories),
              "input-view Git identity directories changed")
    for name in expected:
        directories.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
    seen = set()
    for root, dirs, files in os.walk(view, followlinks=False):
        for name in list(dirs):
            path = Path(root) / name
            relative = str(path.relative_to(view))
            if path.is_symlink():
                seen.add(relative)
                dirs.remove(name)
            else:
                q.require(relative in directories, f"unexpected input-view directory: {relative}")
        seen.update(str((Path(root) / name).relative_to(view)) for name in files)
    q.require(seen == set(expected), "input-view file inventory changed: "
              + ", ".join(sorted(seen ^ set(expected))[:20]))
    for name, item in expected.items():
        path = view / name
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise q.GateError(f"input-view path cannot resolve: {name}") from error
        q.require(resolved.is_relative_to(view), f"input-view path escapes: {name}")
        q.require(actual_blob(path, item["mode"]) == item["blob"], f"input-view bytes/mode changed: {name}")
    return identity(source)


def materialize(provenance, source, view):
    view.mkdir()
    for name, item in source["files"].items():
        q.safe_path(name)
        q.require(not q.publication_metadata(name), "metadata cannot be a compiler input")
        target = view / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = provenance / name
        if item["mode"] == "120000":
            target.symlink_to(os.readlink(original))
        else:
            shutil.copy2(original, target, follow_symlinks=False)
    for name in (".git/objects", ".git/refs"):
        (view / name).mkdir(parents=True)
    for name, data in identity_files(source).items():
        (view / name).write_bytes(data)
        (view / name).chmod(0o644)
    verify(view, source)
    return view


def sandbox(view, out, toolchain, toolkit, environment, bwrap, hidden_paths):
    """Empty-root compiler namespace; no network, host homes, provenance or GPUs."""
    q.require(sys.platform == "linux", "controlled native builds require Linux input isolation")
    for path in (out / "target", out / "python-cache"):
        path.mkdir(exist_ok=True)
    command = [str(bwrap), "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv"]
    mounts = [(Path("/usr"), Path("/usr"))]
    for name in ("bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            command += ["--symlink", os.readlink(path), str(path)]
        elif path.exists():
            mounts.append((path, path))
    for name in ("/etc/ld.so.cache", "/etc/localtime", "/etc/alternatives"):
        path = Path(name)
        if path.exists():
            mounts.append((path, path))
    mounts += [(toolchain.resolve(), Path("/toolchain")), (toolkit.resolve(), Path("/cuda")),
               (view.resolve(), Path("/source"))]
    for original, target in mounts:
        command += ["--ro-bind", str(original), str(target)]
    command += ["--bind", str(out / "cargo-home"), "/cargo",
                "--bind", str(out / "target"), "/target",
                "--bind", str(out / "python-cache"), "/python-cache",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    # A caller could put its full checkout under a system/toolchain root. Hide
    # those known provenance paths even when an enclosing system mount is visible.
    masked_paths = set()
    for hidden in hidden_paths:
        hidden = hidden.resolve()
        for original, target in mounts[:-1]:
            if hidden.is_relative_to(original.resolve()):
                masked = target / hidden.relative_to(original.resolve())
                q.require(masked != target, "provenance overlaps a required compiler root")
                masked_paths.add(masked)
    for masked in sorted(masked_paths):
        if not any(masked != other and masked.is_relative_to(other) for other in masked_paths):
            command += ["--tmpfs", str(masked)]
    env = {k: v for k, v in environment.items() if k in ("HOME", "USER", "LOGNAME")}
    env.update(PATH="/toolchain/bin:/cuda/bin:/usr/bin:/bin", LANG="C.UTF-8", LC_ALL="C.UTF-8",
               TMPDIR="/tmp", CUDA_VISIBLE_DEVICES="", MEMRA_CUDA_ARCH="120a",
               MEMRA_NVCC="/cuda/bin/nvcc", CUDA_HOME="/cuda", CUDA_PATH="/cuda",
               CARGO_HOME="/cargo", CARGO_TARGET_DIR="/target", RUSTC="/toolchain/bin/rustc",
               RUSTUP_TOOLCHAIN="1.97.1", PYTHONPYCACHEPREFIX="/python-cache", PYTHONDONTWRITEBYTECODE="1")
    for name, value in sorted(env.items()):
        command += ["--setenv", name, value]
    command += ["--chdir", "/source", "--"]
    return command, env
