#!/usr/bin/env python3
"""Bounded CPU verification, raw output retained before summarizing. No GPU/env inspection."""
import datetime
import hashlib
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
commands = [
    ["git", "rev-parse", "HEAD"],
    ["cargo", "fmt", "--all", "--", "--check"],
    ["cargo", "check", "-p", "memra-kv", "-p", "memra-tier", "--offline", "--all-targets"],
    ["cargo", "test", "-p", "memra-kv", "-p", "memra-tier", "--offline"],
    ["cargo", "clippy", "-p", "memra-kv", "--offline", "--all-targets", "--no-deps"],
    ["cargo", "clippy", "-p", "memra-tier", "--offline", "--all-targets", "--no-deps", "--", "-D", "warnings"],
    ["git", "diff", "--check"],
    ["bash", "tools/check-flags.sh"],
    ["python3", "crates/memra-tier/tests/contracts/fixture_reference.py", "--check"],
    ["git", "diff", "--exit-code", "259ff819", "--", "crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts"],
]
log = pathlib.Path(__file__).with_name("day2-verification.log")
failed = False
with log.open("w") as out:
    out.write("UTC " + datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
    for command in commands:
        line = "$ " + " ".join(command) + "\n"
        out.write(line)
        out.flush()
        try:
            run = subprocess.run(command, cwd=root, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=180, text=True)
            output, code = run.stdout, run.returncode
        except subprocess.TimeoutExpired as exc:
            output, code = str(exc.stdout or "") + "\nTIMEOUT\n", 124
        out.write(output + f"EXIT {code}\n\n")
        out.flush()
        print(line + output + f"EXIT {code}", flush=True)
        failed |= code != 0
    for path in ["crates/memra-tier/src/contracts.rs", "crates/memra-tier/tests/contracts/fixtures.json"]:
        out.write(f"SHA256 {path} {hashlib.sha256((root / path).read_bytes()).hexdigest()}\n")
sys.exit(1 if failed else 0)
