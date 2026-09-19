#!/usr/bin/env python3
"""CPU-only freeze-v1.1 receipt. Every command runs, including known-red cells."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/spill-lead-20260919/v1.1"
COMMANDS = [
    ("fmt", ["cargo", "fmt", "--all", "--", "--check"]),
    ("check-macos", ["cargo", "check", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets"]),
    ("check-linux", ["cargo", "check", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets", "--target", "x86_64-unknown-linux-gnu"]),
    ("test-required", ["cargo", "test", "-p", "memra-tier", "-p", "memra-kv", "--offline"]),
    ("test-all", ["cargo", "test", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--no-fail-fast"]),
    ("clippy", ["cargo", "clippy", "-p", "memra-tier", "-p", "memra-kv", "--offline", "--all-targets", "--no-deps", "--", "-D", "warnings"]),
    ("diff", ["git", "diff", "--check"]),
    ("flags", ["bash", "tools/check-flags.sh"]),
    ("fixtures", ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"]),
    ("schema-unchanged", ["git", "diff", "--exit-code", "98e558dc", "--", "crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts/fixtures.json", "Cargo.toml", "Cargo.lock"]),
]

def main():
    OUT.mkdir(exist_ok=True)
    tip = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipts = []
    for name, command in COMMANDS:
        started = time.time()
        run = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240, check=False)
        path = OUT / f"{name}.log.gz"
        path.write_bytes(gzip.compress(run.stdout, mtime=0))
        receipts.append({"name": name, "command": command, "exit": run.returncode, "seconds": time.time() - started,
                         "raw_sha256": hashlib.sha256(run.stdout).hexdigest(), "log": path.name})
        print(f"{name}: exit {run.returncode}", flush=True)
    paths = subprocess.check_output(["git", "ls-files", "crates/memra-tier", "crates/memra-kv"], cwd=ROOT, text=True).splitlines()
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if p.endswith((".rs", ".toml", ".json"))}
    (OUT / "commands.json").write_text(json.dumps({"source_commit": tip, "commands": receipts, "sources": sources}, indent=2) + "\n")

if __name__ == "__main__":
    main()
