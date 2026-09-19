#!/usr/bin/env python3
"""Run the CPU freeze checks without shell evaluation, env dumps, or GPU builds."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
commands = [
    ["git", "rev-parse", "HEAD"],
    ["rustc", "--version"],
    ["cargo", "--version"],
    ["cargo", "fmt", "--all", "--", "--check"],
    ["cargo", "check", "-p", "memra-tier", "--offline", "--all-targets"],
    ["cargo", "test", "-p", "memra-tier", "--offline"],
    ["cargo", "check", "-p", "memra-kv", "--offline"],
    ["git", "diff", "--check"],
    ["bash", "tools/check-flags.sh"],
    ["cargo", "clippy", "-p", "memra-tier", "--offline", "--all-targets", "--", "-D", "warnings"],
    ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"],
]
log_path = Path(__file__).with_name("verification.log")
with log_path.open("w") as log:
    def emit(text):
        log.write(text)
        log.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    emit("Repository: avifenesh/memra; lane/spill-lead-20260919\n")
    emit(datetime.now(timezone.utc).isoformat() + "\n")
    for command in commands:
        emit("\n$ " + " ".join(command) + "\n")
        result = subprocess.run(command, cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        emit(result.stdout)
        emit(f"exit={result.returncode}\n")
        if result.returncode:
            sys.exit(result.returncode)
    for name in ["crates/memra-tier/src/contracts.rs",
                 "crates/memra-tier/tests/contracts/fixtures.json"]:
        emit(hashlib.sha256((root / name).read_bytes()).hexdigest() + "  " + name + "\n")
