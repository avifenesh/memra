#!/usr/bin/env python3
"""Rebuild the DSv4 gate prompt tape from git (memra #657).

The DSv4 perf and identity gates read `Review this inference engine source:\n\n{tape}` and
consume its first 256 or 8192 tokens. The originally pinned tape (sha256 f6e175a6...) was cut
from a dirty tree at 9e3c8b550 and no reachable machine holds it. This recipe rebuilds the clean
tape at the same commit: every tracked `crates/**/*.rs|cu|cuh` file, sorted by path, each once,
each preceded by `\n\n===== SOURCE: <path> =====\n`. The result is byte-identical to the pinned
tape for its first 3,736,115 bytes, which is everything the gates tokenize.

usage: tools/dsv4-source-tape.py <output.txt> [--repo <memra checkout>]
Writes <output.txt> and <output.txt>.manifest.json, refuses to overwrite, and exits nonzero
unless the rebuilt digest is the pinned rebuild digest.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys

BASE = "9e3c8b550a1020d7592e94f3c81f5f7bb706ea20"
REBUILD_SHA256 = "11e4bd80352f4a24504ffdee519b33bbc527b3b9bcfeb531815e5731b190cb8c"
REBUILD_BYTES = 22_013_722


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    git = lambda *args: subprocess.run(
        ["git", "-C", a.repo, *args], check=True, capture_output=True
    ).stdout
    names = sorted(
        n
        for n in git("ls-tree", "-r", "-z", "--name-only", BASE, "crates").decode().split("\0")
        if re.search(r"\.(rs|cu|cuh)$", n)
    )
    files, parts = [], []
    for name in names:
        body = git("cat-file", "blob", f"{BASE}:{name}")
        files.append({"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        parts += [f"\n\n===== SOURCE: {name} =====\n".encode(), body]
    tape = b"".join(parts)
    digest = hashlib.sha256(tape).hexdigest()
    if digest != REBUILD_SHA256 or len(tape) != REBUILD_BYTES:
        sys.exit(f"REBUILD_MISMATCH sha256={digest} bytes={len(tape)}")
    with open(a.output, "xb") as f:
        f.write(tape)
    manifest = {"head": BASE, "dirty": "", "files": files, "bytes": len(tape), "sha256": digest,
                "recipe": "sorted tracked Rust/CUDA source; each file once; no repetition or padding"}
    with open(a.output + ".manifest.json", "x") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": len(tape), "sha256": digest, "output": a.output}))


if __name__ == "__main__":
    main()
