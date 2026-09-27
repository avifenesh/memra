#!/usr/bin/env python3
"""Reuse exact, verified controlled-build capsules without sharing a Cargo target.

The caller retains an expectation exported from a trusted build. This is a local
content cache, not a signer, compiler cache, or model qualification mechanism.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

sys.dont_write_bytecode = True
import release_qualification as q
import release_inputs

_SPEC = importlib.util.spec_from_file_location("local_build_gpu_ci", Path(__file__).with_name("gpu-ci.py"))
ci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ci)

SCHEMA = "memra-local-build-expectation-v1"


def read_json(path):
    return q.json_bytes(Path(path).read_bytes())


def expectation_key(expected):
    q.require(isinstance(expected, dict) and set(expected) ==
              {"schema", "source", "build", "oracle_manifest", "payloads"}
              and expected["schema"] == SCHEMA, "unsupported build expectation")
    q.require(isinstance(expected["source"], dict)
              and isinstance(expected["source"].get("commit"), str)
              and q.COMMIT.fullmatch(expected["source"]["commit"]), "expectation needs an exact candidate")
    q.require(isinstance(expected["payloads"], dict)
              and set(expected["payloads"]) == ci.CAPSULE_FILES, "expectation capsule inventory differs")
    for name, identity in expected["payloads"].items():
        # A successful command may emit an empty log. The native verifier still
        # requires nonempty binaries and validates all metadata and log bindings.
        q.require(isinstance(identity, dict) and set(identity) == {"bytes", "sha256"}
                  and type(identity["bytes"]) is int and identity["bytes"] >= 0
                  and isinstance(identity["sha256"], str) and q.SHA.fullmatch(identity["sha256"]),
                  f"invalid capsule payload identity: {name}")
    return q.object_digest(expected)


def admit_checkout(repo, expected):
    """Use the release producer's byte/mode/config checks, then forbid commit equivalence."""
    expectation_key(expected)
    commit = expected["source"]["commit"]
    ci.candidate(repo, commit)
    release_inputs.verify_checkout(repo, commit)
    q.verify_source(expected["source"], repo, commit)
    q.require(q.source_snapshot(repo, commit) == expected["source"], "cache requires the exact source snapshot")
    return commit


def capsule_description(build):
    build = Path(build).resolve()
    payloads = {}
    for name in sorted(ci.CAPSULE_FILES):
        path = build / name
        q.require(path.is_file() and not path.is_symlink()
                  and path.resolve().is_relative_to(build), f"capsule member is not a contained regular file: {name}")
        payloads[name] = q.file_identity(path)
    oracle = read_json(build / "oracle-manifest.json")
    return {"schema": SCHEMA, "source": read_json(build / "source.json"),
            "build": read_json(build / "build.json"),
            "oracle_manifest": ci.manifest(q.canonical(oracle)),
            "payloads": payloads}


def describe(repo, build):
    """Export all build context and byte identities from an admitted capsule."""
    expected = capsule_description(build)
    commit = admit_checkout(repo, expected)
    ci.verify_build(repo, build, commit)
    # Catch capsule edits during admission before exporting its identities.
    q.require(capsule_description(build) == expected, "build capsule changed during admission")
    return expected


def verify_capsule(repo, build, expected):
    q.require(capsule_description(build) == expected, "capsule does not match the caller's expectation")
    ci.verify_build(repo, build, expected["source"]["commit"])


def outside_checkout(path, repo):
    path = Path(path).absolute()
    q.require(not path.resolve().is_relative_to(repo.resolve()), "cache and output must be outside the source checkout")
    return path


def cache_entry(cache, expected):
    return Path(cache) / (expectation_key(expected) + ".tar")


def unpack_verified(repo, archive, out, expected):
    q.require(archive.is_file() and not archive.is_symlink(), "cache entry missing or not a regular file")
    ci.unpack_build(archive, out)
    verify_capsule(repo, out, expected)


def store(repo, build, cache, expected):
    """Publish by atomic, no-replace hard link from private scratch on the same filesystem."""
    commit = admit_checkout(repo, expected)
    verify_capsule(repo, build, expected)
    cache = outside_checkout(cache, repo)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache_entry(cache, expected)
    with tempfile.TemporaryDirectory(prefix=".build-writer-", dir=cache) as scratch:
        scratch = Path(scratch)
        if target.exists() or target.is_symlink():
            unpack_verified(repo, target, scratch / "existing", expected)
            admit_checkout(repo, expected)
            return {"status": "already-present", "key": expectation_key(expected), "archive": str(target)}
        archive = scratch / "capsule.tar"
        ci.pack_build(repo, build, commit, archive)
        # Verify bytes actually packed, not only the original mutable directory.
        unpack_verified(repo, archive, scratch / "checked", expected)
        admit_checkout(repo, expected)
        archive.chmod(0o444)
        with archive.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(archive, target)
            status = "stored"
        except FileExistsError:
            unpack_verified(repo, target, scratch / "existing", expected)
            status = "already-present"
    return {"status": status, "key": expectation_key(expected), "archive": str(target)}


def load(repo, cache, expected, out):
    """Restore to private scratch and validate before exposing a fresh output directory."""
    admit_checkout(repo, expected)
    cache = outside_checkout(cache, repo)
    out = outside_checkout(out, repo)
    q.require(not out.exists() and not out.is_symlink(), "output already exists; use a fresh per-writer path")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-reader-", dir=out.parent) as scratch:
        restored = Path(scratch) / "restored"
        unpack_verified(repo, cache_entry(cache, expected), restored, expected)
        admit_checkout(repo, expected)
        # No caller can mutate a shared target: these are new files, not hard links.
        # Reserve the destination name exclusively before publishing the verified tree.
        out.mkdir()
        try:
            os.rename(restored, out)
        except BaseException:
            out.rmdir()
            raise
    return {"status": "restored", "key": expectation_key(expected), "build": str(out),
            "qualification": "build-only; model and GPU gates still required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "store", "load"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", type=Path, help="existing controlled build directory")
    parser.add_argument("--expect", type=Path, help="caller-pinned descriptor, retained independently of cache")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/memra/controlled-builds-v1")
    parser.add_argument("--out", type=Path, help="new descriptor file (describe) or build directory (load)")
    args = parser.parse_args()
    for field in {"describe": ("build", "out"), "store": ("build", "expect"), "load": ("expect", "out")}[args.command]:
        if getattr(args, field) is None:
            parser.error(f"{args.command} requires --{field}")
    started = time.monotonic()
    try:
        args.repo = args.repo.resolve()
        if args.command == "describe":
            expected = describe(args.repo, args.build)
            out = outside_checkout(args.out, args.repo)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("xb") as stream:
                stream.write(q.canonical(expected) + b"\n")
            result = {"status": "described", "key": expectation_key(expected), "expect": str(out)}
        elif args.command == "store":
            result = store(args.repo, args.build, args.cache, read_json(args.expect))
        else:
            result = load(args.repo, args.cache, read_json(args.expect), args.out)
        result["seconds"] = round(time.monotonic() - started, 6)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (q.GateError, ci.Refused, OSError, ValueError, TypeError, KeyError, tarfile.TarError) as error:
        print(f"build-cache refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
