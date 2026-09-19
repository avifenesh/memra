#!/usr/bin/env python3
"""Bounded CPU-only checks; capture combined raw output before reporting results."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
checks = [
    ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
    ("mac-check", ["cargo", "check", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets"]),
    ("linux-check", ["cargo", "check", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"]),
    ("tests", ["cargo", "test", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--no-fail-fast"]),
    ("clippy", ["cargo", "clippy", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("diff", ["git", "diff", "--check"]),
    ("flags", ["bash", "tools/check-flags.sh"]),
    ("docs", ["bash", "tools/docs-registry-census.sh"]),
    ("wire", ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"]),
    ("frozen", ["git", "diff", "--exit-code", "b3487a03b0ee3f833c1157e7b7d68f2cb35a3843", "--", "crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts/fixtures.json", "Cargo.toml", "Cargo.lock"]),
]
manifest = {}
for crate in ["memra-tier", "memra-kv"]:
    for path in sorted((repo / "crates" / crate).rglob("*.rs")):
        manifest[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
result = {"source_commit": source, "kind": "cpu-cross-compile-not-qualified", "checks": []}
for name, argv in checks:
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        proc = subprocess.run(argv, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240, check=False)
        raw, code = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as error:
        raw, code = (error.stdout or b"") + b"\nCPU verification timeout (240 seconds)\n", 124
    archive = gzip.compress(raw, mtime=0)
    (out / (name + ".log.gz")).write_bytes(archive)
    result["checks"].append({"name": name, "argv": argv, "started_utc": start, "exit_code": code, "raw_sha256": hashlib.sha256(raw).hexdigest(), "archive_sha256": hashlib.sha256(archive).hexdigest()})
    (out / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: exit {code}", flush=True)
    if name == "tests":
        for line in raw.decode(errors="replace").splitlines():
            if "test result:" in line or "FAILED" in line:
                print(line, flush=True)
    if code:
        print(raw.decode(errors="replace")[-8000:], flush=True)
raise SystemExit(any(check["exit_code"] for check in result["checks"]))
